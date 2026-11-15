//! What a command buffer of fenced encoders costs: its encoders' opens and ends on the host, its
//! `vkQueueSubmit` call, the GPU's start after it, and its GPU time from its first command's start
//! to its last's end, by the shape of its encoders' fences (`encoder_fold.rs`'s **Fences**). The
//! shapes, each [`ENCODERS`] serial compute encoders of one dispatch over [`THREADS`] threads, a
//! command buffer at a time, waited for:
//! - **Independent, updating**: each writes a buffer of its own from one input and updates one
//!   fence; none waits for it. Nothing orders them but what the runtime records.
//! - **Independent**: the same with no fence.
//! - **Chain, wait first**: each waits for the fence, then dispatches, reading what the one before
//!   wrote through untracked buffers (only the fence orders them), then updates the fence.
//! - **Chain, wait after the dispatch**: the same with the wait after the dispatch, which still
//!   covers it (`waitForFence:`, "a fence that the pass waits for before it runs any of its
//!   commands").
//!
//! Each shape runs [`COMMAND_BUFFERS`] command buffers a round, [`ROUNDS`] rounds, interleaved, on
//! a device that writes GPU timestamps. Printed (`FENCE-FOLD`), per shape: the
//! `VkCommandBuffer`s a submission and the encoders folded a command buffer, the mean host time
//! of an encoder's open (`compute_command_encoder`) and of its `end_encoding`, the mean call, and
//! the median and 90th percentile of the GPU's start after the call (`gpu_start_time` from
//! `kernel_end_time`) and of the GPU time (`gpu_end_time` from `gpu_start_time`); and the counts
//! named `folds_refused_*` and `fold_fence_barriers` that moved, read from the counts' `Debug`
//! form so that the bench builds where they do not exist.
//!
//! Ignored: run it in release under `--cfg m2v_gpu`,
//! `cargo test --release -p metal2vulkan-runtime --lib fence_fold_bench -- --ignored --nocapture`.

use super::blit_holds::{plan_device, Scratch, WAIT};
use crate::{
    Buffer, CommandBuffer, CommandQueue, ComputePipelineState, Device, Fence, HazardTrackingMode,
    ResourceOptions, Size, WorkCounts,
};
use std::collections::BTreeMap;
use std::time::Instant;

/// The encoders of each command buffer.
const ENCODERS: usize = 24;
/// The threads of each dispatch, in threadgroups of [`GROUP`].
const THREADS: u64 = 1 << 14;
const GROUP: u64 = 64;
/// The command buffers of each shape a round.
const COMMAND_BUFFERS: usize = 200;
/// The rounds.
const ROUNDS: usize = 3;

/// A shape of a command buffer's encoders.
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    IndependentUpdating,
    Independent,
    WaitFirst,
    WaitAfter,
}

const SHAPES: [(Shape, &str); 4] = [
    (Shape::IndependentUpdating, "independent, updating"),
    (Shape::Independent, "independent"),
    (Shape::WaitFirst, "chain, wait first"),
    (Shape::WaitAfter, "chain, wait after the dispatch"),
];

/// The kernel (`output[i] = input[i] * scale + i`) and what its shapes read and write.
struct Rig {
    device: Device,
    queue: CommandQueue,
    state: ComputePipelineState,
    scale: Buffer,
    input: Buffer,
    outputs: Vec<Buffer>,
    chain: [Buffer; 2],
    fence: Fence,
}

/// The host time a command buffer's encoders' opens and ends took, in nanoseconds.
#[derive(Default)]
struct Host {
    opens: u128,
    ends: u128,
}

impl Rig {
    fn new(device: Device) -> Rig {
        let floats =
            |words: u64, options: ResourceOptions| device.new_buffer(words * 4, options).unwrap();
        let untracked =
            ResourceOptions::private().with_hazard_tracking_mode(HazardTrackingMode::Untracked);
        Rig {
            state: super::dispatch_pending::state(&device),
            scale: device
                .new_buffer_with_bytes(&1f32.to_le_bytes(), ResourceOptions::shared())
                .unwrap(),
            input: floats(THREADS, ResourceOptions::private()),
            outputs: (0..ENCODERS)
                .map(|_| floats(THREADS, ResourceOptions::private()))
                .collect(),
            chain: [floats(THREADS, untracked), floats(THREADS, untracked)],
            fence: device.new_fence(),
            queue: device.new_command_queue().unwrap(),
            device,
        }
    }

    /// A command buffer of `shape`, committed, with the host time its encoders' opens and ends
    /// took.
    fn commit(&self, shape: Shape) -> (CommandBuffer, Host) {
        let command_buffer = self.queue.command_buffer().unwrap();
        let mut host = Host::default();
        for k in 0..ENCODERS {
            let opened = Instant::now();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            host.opens += opened.elapsed().as_nanos();
            if shape == Shape::WaitFirst {
                encoder.wait_for_fence(&self.fence).unwrap();
            }
            let (input, output) = match shape {
                Shape::WaitFirst | Shape::WaitAfter => {
                    (&self.chain[k % 2], &self.chain[(k + 1) % 2])
                }
                _ => (&self.input, &self.outputs[k]),
            };
            encoder.set_compute_pipeline_state(&self.state).unwrap();
            encoder.set_buffer(Some(input), 0, 0).unwrap();
            encoder.set_buffer(Some(output), 0, 1).unwrap();
            encoder.set_buffer(Some(&self.scale), 0, 2).unwrap();
            encoder
                .dispatch_threads(Size::new(THREADS, 1, 1), Size::new(GROUP, 1, 1))
                .unwrap();
            if shape == Shape::WaitAfter {
                encoder.wait_for_fence(&self.fence).unwrap();
            }
            if shape != Shape::Independent {
                encoder.update_fence(&self.fence).unwrap();
            }
            let ending = Instant::now();
            encoder.end_encoding().unwrap();
            host.ends += ending.elapsed().as_nanos();
        }
        command_buffer.commit().unwrap();
        (command_buffer, host)
    }
}

/// The counts named `folds_refused_*` and `fold_fence_barriers` in `counts`' `Debug` form.
fn fold_counts(counts: &WorkCounts) -> BTreeMap<String, u64> {
    let text = format!("{counts:?}");
    text.split([',', '{', '}'])
        .filter_map(|field| {
            let (name, value) = field.split_once(':')?;
            let name = name.trim();
            if !(name.starts_with("folds_refused_") || name == "fold_fence_barriers") {
                return None;
            }
            Some((name.to_owned(), value.trim().parse().ok()?))
        })
        .collect()
}

/// What a shape's command buffers showed: the counts' change, the host times, and each one's GPU
/// start after its call and GPU time, in seconds.
#[derive(Default)]
struct Seen {
    command_buffers: u64,
    submissions: u64,
    vk_command_buffers: u64,
    folded: u64,
    call_nanos: u64,
    opens: u128,
    ends: u128,
    fold: BTreeMap<String, u64>,
    starts: Vec<f64>,
    gpu: Vec<f64>,
}

impl Seen {
    fn add(
        &mut self,
        (before, after): (&WorkCounts, &WorkCounts),
        command_buffer: &CommandBuffer,
        host: Host,
    ) {
        self.command_buffers += 1;
        self.submissions += after.queue_submissions - before.queue_submissions;
        self.vk_command_buffers +=
            after.command_buffers_submitted - before.command_buffers_submitted;
        self.folded += after.encoders_folded - before.encoders_folded;
        self.call_nanos += after.batch_submit_nanos - before.batch_submit_nanos;
        self.opens += host.opens;
        self.ends += host.ends;
        let earlier = fold_counts(before);
        for (name, value) in fold_counts(after) {
            let moved = value - earlier.get(&name).copied().unwrap_or(0);
            *self.fold.entry(name).or_default() += moved;
        }
        let (start, end) = (
            command_buffer.gpu_start_time().unwrap(),
            command_buffer.gpu_end_time().unwrap(),
        );
        self.starts.push(start - command_buffer.kernel_end_time());
        self.gpu.push(end - start);
    }
}

fn quantile(values: &[f64], q: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() as f64 * q) as usize).min(sorted.len() - 1)] * 1e6
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn fence_fold_bench() {
    let scratch = Scratch::new("fence-fold");
    let rig = Rig::new(plan_device(&scratch.0));
    println!("FENCE-FOLD on {}", rig.device.name());
    // Warm: the pipelines made, so no dispatch leaves work for later.
    for (shape, _) in SHAPES {
        rig.commit(shape).0.wait_until_completed_for(WAIT).unwrap();
    }
    let mut seen: Vec<Seen> = SHAPES.iter().map(|_| Seen::default()).collect();
    for _ in 0..ROUNDS {
        for ((shape, _), seen) in SHAPES.iter().zip(&mut seen) {
            for _ in 0..COMMAND_BUFFERS {
                let before = rig.device.work_counts();
                let (command_buffer, host) = rig.commit(*shape);
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                let after = rig.device.work_counts();
                seen.add((&before, &after), &command_buffer, host);
            }
        }
    }
    for ((_, name), seen) in SHAPES.iter().zip(&seen) {
        let submissions = seen.submissions.max(1);
        let encoders = (seen.command_buffers * ENCODERS as u64) as f64;
        let moved: Vec<String> = (seen.fold.iter())
            .filter(|(_, value)| **value > 0)
            .map(|(name, value)| format!("{name} {value}"))
            .collect();
        println!(
            "FENCE-FOLD {name}: {:.1} VkCommandBuffers a submission, {:.1} encoders folded a \
             command buffer; an encoder's open {:.2} us, its end {:.2} us; mean call {:.1} us; GPU \
             start after the call median {:.1} us, p90 {:.1} us; GPU time median {:.1} us, p90 \
             {:.1} us; moved: {}",
            seen.vk_command_buffers as f64 / submissions as f64,
            seen.folded as f64 / seen.command_buffers as f64,
            seen.opens as f64 / encoders / 1e3,
            seen.ends as f64 / encoders / 1e3,
            seen.call_nanos as f64 / submissions as f64 / 1e3,
            quantile(&seen.starts, 0.5),
            quantile(&seen.starts, 0.9),
            quantile(&seen.gpu, 0.5),
            quantile(&seen.gpu, 0.9),
            match moved.is_empty() {
                true => "none".to_owned(),
                false => moved.join(", "),
            },
        );
    }
}
