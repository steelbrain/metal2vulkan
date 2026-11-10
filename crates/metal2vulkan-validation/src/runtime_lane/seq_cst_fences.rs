//! Sequentially consistent fences against store buffering, through the runtime, on the device the
//! cfg selects, under the validation layer.
//!
//! The store-buffering litmus (`metal2vulkan_devtools::litmus::store_buffering`): of each pair
//! of threads, one stores `x[i]`, fences and loads `y[i]`, the other stores `y[i]`, fences and
//! loads `x[i]`. Under Metal's `seq_cst` device fence no pair reads 0 twice (none
//! of 13.1 M pairs on an M3, where 2,684 did without a fence). The translation keeps the order
//! through an acquire-release increment of the command buffer's status word after each such
//! fence; Vulkan's own `SequentiallyConsistent` is `AcquireRelease`. The lane runs
//! [`ROUNDS`] command buffers of [`PAIRS`] pairs under the `seq_cst` fence and requires none to
//! read 0 twice; it prints the count without a fence and under an `acq_rel` fence beside it,
//! which the device may or may not show (the hardware's own order), so the run says whether the
//! device could buffer stores at all.

use super::tile_states::on_device_with;
use super::{at_plan, buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::litmus::store_buffering;
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Size,
};

/// Threads a threadgroup, as the kernel indexes them.
const THREADS: u64 = 256;

/// Threadgroup pairs a command buffer.
const GROUP_PAIRS: u64 = 1024;

/// Thread pairs a command buffer.
const PAIRS: u64 = THREADS * GROUP_PAIRS;

/// Command buffers a fence.
const ROUNDS: usize = 8;

/// The kernel's state with `fence` between each thread's store and load.
fn state(device: &Device, fence: &str) -> ComputePipelineState {
    let bitcode = llvm_assemble_text(&store_buffering(fence)).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name("sb").unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The pairs of [`ROUNDS`] command buffers of `kernel` that read 0 twice.
fn both_old(device: &Device, kernel: &ComputePipelineState) -> u64 {
    let queue = device.new_command_queue().unwrap();
    let mut seen = 0;
    for _ in 0..ROUNDS {
        let zeros = vec![0u8; (PAIRS * 4) as usize];
        let (x, y) = (
            buffer_with(device, &zeros).unwrap(),
            buffer_with(device, &zeros).unwrap(),
        );
        let r = buffer_with(device, &vec![0xAB; (PAIRS * 8) as usize]).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(kernel).unwrap();
        for (index, buffer) in [&x, &y, &r].into_iter().enumerate() {
            encoder.set_buffer(Some(buffer), 0, index as u32).unwrap();
        }
        encoder
            .dispatch_threadgroups(Size::new(2 * GROUP_PAIRS, 1, 1), Size::new(THREADS, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let words: Vec<u32> = read_buffer(&r, 0, PAIRS * 8)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        for (i, pair) in words.as_chunks::<2>().0.iter().enumerate() {
            assert!(pair.iter().all(|&v| v <= 1), "pair {i} read {pair:?}");
            seen += u64::from(*pair == [0, 0]);
        }
    }
    seen
}

/// No pair reads 0 twice under the `seq_cst` device fence (module docs); the counts without a
/// fence and under an `acq_rel` one are printed.
#[test]
fn a_seq_cst_fence_forbids_store_buffering() {
    on_device_with(at_plan, |device| {
        let name = device.name().to_owned();
        let mut counts = Vec::new();
        for (what, fence) in [
            ("no fence", ""),
            (
                "acq_rel fence",
                "  tail call void @air.atomic.fence(i32 1, i32 4, i32 2)",
            ),
            (
                "seq_cst fence",
                "  tail call void @air.atomic.fence(i32 1, i32 5, i32 2)",
            ),
        ] {
            let seen = both_old(device, &state(device, fence));
            eprintln!(
                "{name}: {what}: {seen} of {} pairs read 0 twice",
                PAIRS * ROUNDS as u64
            );
            counts.push((what, seen));
        }
        assert_eq!(
            counts.last(),
            Some(&("seq_cst fence", 0)),
            "{name}: {counts:?}"
        );
    });
}
