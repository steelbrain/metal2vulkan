//! The widest-pin rule's integer kernels on the GPU the cfg selects: the owned shuffle kernels of
//! `engine/wps_shuffles`, each run through the runtime under the validation layer, give the CPU's
//! answer exactly. Whether the rule fires for a kernel on this device is read from the module the
//! translator compiles for the opened device's queried description (its strategy census), and
//! printed. It times nothing.

use super::engine_compute::case;
use super::*;
use crate::candidate::engine::{ComputeOutput, ComputeRun};
use crate::check::admission::ExecutorError;
use metal2vulkan::bind::Dispatch;
use metal2vulkan_devtools::{engine_dir, llvm_assemble};

/// The strategy census name of a kernel the rule emulates (`lower::simd`'s `WIDEST_PIN`).
const WIDEST_PIN: &str = "simd.emulated.widest-pin";

/// The integer kernels over the rule's bounds: `wps_uint_k{K}_s{S}`, `K` live accumulators over `S`
/// steps.
const KERNELS: [(u32, u32); 4] = [(32, 64), (48, 16), (40, 32), (24, 32)];

/// The grid's threads, in threadgroups of [`GROUP`]: whole simdgroups of 32, so thread `t` is lane
/// `t % 32` of its simdgroup.
const THREADS: u32 = 1024;
const GROUP: u32 = 64;

/// Input word `i`: a multiplicative hash of `i`.
fn input(words: u32) -> Vec<u32> {
    (0..words)
        .map(|i| (i ^ 0x5bd1_e995).wrapping_mul(0x9e37_79b9).rotate_left(13))
        .collect()
}

/// What `wps_uint_k{k}_s{s}` writes for `input`, as the MSL states it: each step adds lane
/// `lane ^ (1 << (step % 5))`'s accumulator `step % k` to accumulator `(step + 1) % k`, then sets
/// each accumulator `i` in turn to `a[i] * 2654435761 + (a[(i + 1) % k] >> 3)`.
fn expected(k: u32, s: u32, input: &[u32]) -> Vec<u32> {
    let k = k as usize;
    let mut a: Vec<Vec<u32>> = input.chunks(k).map(<[u32]>::to_vec).collect();
    for step in 0..s as usize {
        let (p, q, mask) = (step % k, (step + 1) % k, 1usize << (step % 5));
        let shuffled: Vec<u32> = (0..a.len())
            .map(|t| a[(t & !31) | ((t & 31) ^ mask)][p])
            .collect();
        for (t, value) in shuffled.into_iter().enumerate() {
            a[t][q] = a[t][q].wrapping_add(value);
        }
        for thread in &mut a {
            for i in 0..k {
                thread[i] = thread[i]
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(thread[(i + 1) % k] >> 3);
            }
        }
    }
    a.concat()
}

/// Run `entry` over `input` through the runtime.
fn run(entry: &str, bitcode: &[u8], input: &[u32]) -> Result<ComputeRun, ExecutorError> {
    let bytes: Vec<u8> = input.iter().flat_map(|w| w.to_le_bytes()).collect();
    let mut kernel = case(
        entry,
        bytes.clone(),
        Dispatch::Threads {
            threads: [THREADS, 1, 1],
            threads_per_group: [GROUP, 1, 1],
        },
    );
    kernel.buffers.insert(1, vec![0; bytes.len()]);
    kernel.output = ComputeOutput::Buffer {
        binding: 1,
        offset: 0,
        length: bytes.len() as u64,
    };
    super::engine_compute::run_through_runtime(&kernel, bitcode)
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// Whether the translator lowers `entry` of `bitcode` in the rule's emulated form for the opened
/// device's queried description.
fn fires(entry: &str, bitcode: &[u8]) -> bool {
    use metal2vulkan::library::Library;
    use metal2vulkan::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
    let opened = platform::opened_worker_device(crate::gpu::TEST_DEVICE).unwrap();
    let function = Library::from_air(bitcode)
        .and_then(|library| library.function(entry))
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(function)));
    let compiled = opened
        .device()
        .compile_compute_pipeline(&descriptor, &metal2vulkan::Budget::standard())
        .unwrap();
    compiled.stage.strategies.entries.contains_key(WIDEST_PIN)
}

#[test]
fn the_widest_pin_rule_s_integer_kernels_give_their_exact_answers_through_the_runtime() {
    let bitcode = llvm_assemble(&engine_dir().join("wps_shuffles.ll")).unwrap();
    for (k, s) in KERNELS {
        let entry = format!("wps_uint_k{k}_s{s}");
        let input = input(THREADS * k);
        let answer = expected(k, s, &input);
        eprintln!(
            "{entry}: the widest-pin rule fires on this device: {}",
            fires(&entry, &bitcode)
        );
        let ran = run(&entry, &bitcode, &input).unwrap();
        assert!(
            words(&ran.output) == answer,
            "{entry}: the runtime's output"
        );
    }
}
