//! The widest-pin rule's two forms on the GPU the cfg selects: the owned shuffle kernels of
//! `engine/wps_shuffles` compiled from the opened device's queried description, and again from it
//! with the rule's trigger cleared (its smallest subgroup size raised to its largest, so the rule's
//! `min < 32` fails and the kernel pins its subgroups instead). Each form runs through the engine
//! compute candidate under the validation layer, and each integer kernel's output is the CPU's
//! answer exactly in both. Where the queried description does not fire the rule there is no second
//! form to run, and the test says so and skips. It times nothing.

use super::engine_compute::{case, BACKEND};
use super::*;
use crate::candidate::engine::{compile_case, ComputeOutput, ComputeRun};
use crate::check::admission::ExecutorError;
use crate::worker::compile::WorkerDevice;
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

/// Run `entry` over `input` with its pipeline compiled for `target`, or, given `None`, for the
/// opened device.
fn run(
    entry: &str,
    bitcode: &[u8],
    input: &[u32],
    target: Option<&WorkerDevice>,
) -> Result<ComputeRun, ExecutorError> {
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
    platform::execute_engine(
        crate::gpu::TEST_DEVICE,
        "engine-test",
        &kernel,
        |opened, shape| {
            compile_case(
                compile_worker(),
                target.unwrap_or(opened),
                &crate::acceptance::retained(bitcode),
                &[],
                &kernel,
                shape,
            )
        },
        |_, token, _, _| Err(format!("no template has pipeline token {token}").into()),
        &[],
        BACKEND,
    )
    .result
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// Whether `run`'s kernel ran on the rule's emulated form, and the subgroup size it pinned.
fn form(run: &ComputeRun) -> (bool, Option<u32>) {
    let module = &run.census.modules[0];
    (
        module.strategies.contains(WIDEST_PIN),
        module.subgroup.required_size,
    )
}

#[test]
fn the_widest_pin_rule_s_two_forms_give_the_integer_kernels_their_exact_answers() {
    let test = "the_widest_pin_rule_s_two_forms_give_the_integer_kernels_their_exact_answers";
    let opened = platform::opened_worker_device(crate::gpu::TEST_DEVICE).unwrap();
    let mut description = opened.description().clone();
    let (min, max) = (description.subgroup.min_size, description.subgroup.max_size);
    description.subgroup.min_size = max;
    let cleared = WorkerDevice::new(description, *opened.policy()).unwrap();
    let bitcode = llvm_assemble(&engine_dir().join("wps_shuffles.ll")).unwrap();
    for (k, s) in KERNELS {
        let entry = format!("wps_uint_k{k}_s{s}");
        let input = input(THREADS * k);
        let answer = expected(k, s, &input);
        let picked = run(&entry, &bitcode, &input, None).unwrap();
        if !form(&picked).0 {
            eprintln!(
                "SKIP {test}: the widest-pin rule does not fire for {entry} on this device \
                 (subgroup sizes {min} to {max}; it fires below 32 to 32 with subgroup size \
                 control), so it has no second form here"
            );
            return;
        }
        assert_eq!(form(&picked), (true, None), "{entry}: the rule's form");
        assert!(
            words(&picked.output) == answer,
            "{entry}: the rule's emulated form's output"
        );
        let other = run(&entry, &bitcode, &input, Some(&cleared)).unwrap();
        assert_eq!(
            form(&other),
            (false, Some(32)),
            "{entry}: the trigger cleared"
        );
        assert!(
            words(&other.output) == answer,
            "{entry}: the pinned form's output"
        );
    }
}
