//! What a guest's buffers over one shared dma-buf cost its command buffers when they come and go
//! one at a time, with the file kept (`Device::import_shared_dma_buf`) and without. Each command
//! buffer runs [`ENCODERS`] dispatches apart and one more whose output is a no-copy buffer of
//! [`BYTES`] made just before its commit over the next piece of one `udmabuf` of a sealed
//! `memfd`'s pages (the same `Arc`, as a consumer hands out its guest RAM's file), committed and
//! waited for, as a guest that times its kernels commits them; then the buffer is dropped and its
//! release settled. Without a handle the file's import goes with each buffer, so the next imports
//! it again; with one, each joins the kept import.
//!
//! The arms, interleaved over [`ROUNDS`] rounds of [`COMMITS`] command buffers: **unkept** and
//! **kept**. Printed (`KEPT-FILE`), per arm: the imports, frees and pieces joined a command
//! buffer (`WorkCounts::dma_buf_imports`, the shared imports freed, `dma_buf_pieces_joined`), the
//! submission calls' median and 90th percentile (`WorkCounts::batch_submit_nanos`), and the
//! kernel end (the call's start) to the GPU start's median and 90th percentile.
//!
//! Ignored: run it in release under `--cfg m2v_gpu` where `/dev/udmabuf` opens,
//! `cargo test --release -p metal2vulkan-runtime --lib kept_file_bench -- --ignored --nocapture`.

use super::blit_holds::{Scratch, WAIT};
use super::compute_parked::Kernel;
use super::dma_buf_shared_files::{no_copy, shared_source};
use super::host_residency::{udmabuf, udmabuf_openable, Ram};
use crate::{ResourceOptions, WorkCounts};
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The dispatches apart a command buffer, before the one over the buffer.
const ENCODERS: u64 = 4;
/// The bytes of each buffer over a piece of the file.
const BYTES: usize = 1 << 20;
/// The pieces of the file, used in turn.
const PIECES: usize = 16;
/// The command buffers of an arm a round.
const COMMITS: usize = 150;
/// The rounds.
const ROUNDS: usize = 3;
/// How long a buffer's release may take to settle.
const SETTLE: Duration = Duration::from_secs(10);

/// What an arm measured: per command buffer, its calls' nanoseconds and its kernel end to GPU
/// start in seconds; and the counts it moved.
#[derive(Default)]
struct Arm {
    calls: Vec<f64>,
    gaps: Vec<f64>,
    commits: u64,
    imports: u64,
    frees: u64,
    joined: u64,
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

/// Wait until the device's live dma-buf memories fall to `count`, at most [`SETTLE`].
fn settled_to(kernel: &Kernel, count: u64) -> WorkCounts {
    let start = Instant::now();
    loop {
        let counts = kernel.device.work_counts();
        if counts.dma_buf_memories_live <= count || start.elapsed() > SETTLE {
            return counts;
        }
        std::thread::sleep(Duration::from_micros(200));
    }
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn kept_file_bench() {
    if !udmabuf_openable("kept_file_bench") {
        return;
    }
    let scratch = Scratch::new("kept-file");
    let kernel = Kernel::new(&scratch);
    kernel.warm();
    let device = &kernel.device;
    println!("KEPT-FILE on {}", device.name());
    let len = PIECES * BYTES;
    let ram = Ram::new(len, 0, len);
    let file = Arc::new(udmabuf(ram.file(), 0, len as u64).unwrap());
    let alignment = Arc::new(AtomicU64::new(0));
    let source = shared_source(&ram, &file, 0, &alignment);
    let queue = device.new_command_queue().unwrap();
    let output = kernel.output(ENCODERS);
    let mut arms = [Arm::default(), Arm::default()];
    let base = device.work_counts().dma_buf_memories_live;
    for _ in 0..ROUNDS {
        for (kept, arm) in [false, true].into_iter().zip(&mut arms) {
            let handle = kept.then(|| {
                device
                    .import_shared_dma_buf(&file, ResourceOptions::default())
                    .unwrap()
            });
            let floor = device.work_counts().dma_buf_memories_live;
            for commit in 0..COMMITS {
                let command_buffer = queue.command_buffer().unwrap();
                for at in 0..ENCODERS {
                    kernel.dispatch(&command_buffer, (&output, at * 256));
                }
                let before = device.work_counts();
                let piece = (commit % PIECES) * BYTES;
                let buffer = no_copy(device, &ram, (piece, BYTES), Arc::clone(&source)).unwrap();
                kernel.dispatch(&command_buffer, (&buffer, 0));
                let committed = device.work_counts();
                command_buffer.commit().unwrap();
                command_buffer.wait_until_completed_for(WAIT).unwrap();
                let done = device.work_counts();
                drop(buffer);
                let after = settled_to(&kernel, floor);
                let calls = (done.queue_submissions - committed.queue_submissions).max(1) as f64;
                arm.calls
                    .push((done.batch_submit_nanos - committed.batch_submit_nanos) as f64 / calls);
                if let Ok(start) = command_buffer.gpu_start_time() {
                    arm.gaps.push(start - command_buffer.kernel_end_time());
                }
                arm.commits += 1;
                arm.imports += after.dma_buf_imports - before.dma_buf_imports;
                arm.joined += after.dma_buf_pieces_joined - before.dma_buf_pieces_joined;
                let made = after.dma_buf_imports - before.dma_buf_imports;
                let live = |counts: &WorkCounts| counts.dma_buf_shared_imports_live as i64;
                arm.frees += (made as i64 - (live(&after) - live(&before))).max(0) as u64;
            }
            drop(handle);
            settled_to(&kernel, base);
        }
    }
    for (name, arm) in ["unkept", "kept"].into_iter().zip(&arms) {
        let commits = arm.commits.max(1) as f64;
        println!(
            "KEPT-FILE {name}: {} command buffers; {:.2} imports, {:.2} frees, {:.2} pieces \
             joined a command buffer; call median {:.1} us, p90 {:.1} us; kernel end -> GPU \
             start median {:.1} us, p90 {:.1} us",
            arm.commits,
            arm.imports as f64 / commits,
            arm.frees as f64 / commits,
            arm.joined as f64 / commits,
            quantile(&arm.calls, 0.5) / 1e3,
            quantile(&arm.calls, 0.9) / 1e3,
            quantile(&arm.gaps, 0.5) * 1e6,
            quantile(&arm.gaps, 0.9) * 1e6,
        );
    }
}
