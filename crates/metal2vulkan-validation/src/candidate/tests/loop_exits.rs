//! The loop kernels of `metal2vulkan_devtools::loop_kernels` on the test's Vulkan device: each
//! runs through the runtime under the validation layer, in one threadgroup, and every thread's
//! word is the one its source gives. The kernels whose exits read values their own trips defined
//! (one loop of three exits, and an outer loop of four whose inner loop leaves both) run with trip
//! bounds that reach every exit, and their modules store an undefined value at a loop trip's top
//! (`spirv::carry`): the test counts those stores in the module the translator compiles for the
//! kernel (`Device::compile_compute_pipeline` for `runtime-floor`, the runtime's own module of it),
//! prints the count and requires one. The loop whose threads leave a barrier loop early is walked
//! predicated, so its module holds no such store. It times nothing.

use crate::candidate::engine::{ComputeCase, ComputeOutput};
use metal2vulkan::bind::Dispatch;
use metal2vulkan::library::FunctionConstantValues;
use metal2vulkan_devtools::loop_kernels::{self, BOUNDS, LOOP_LEAVER, NESTED_EXITS, TRIP_EXITS};
use std::collections::{BTreeMap, BTreeSet};

/// The poison every output word starts as.
const POISON: u32 = 0xABAB_ABAB;

/// SPIR-V's `OpUndef` and `OpStore` opcodes.
const OP_UNDEF: u32 = 1;
const OP_STORE: u32 = 62;

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// The stores of an `OpUndef` result in the SPIR-V module `spirv`.
fn undefined_stores(spirv: &[u32]) -> usize {
    let mut insts = Vec::new();
    let mut at = 5;
    while let Some(&first) = spirv.get(at) {
        let count = (first >> 16) as usize;
        assert!(count > 0, "an instruction of no words at word {at}");
        insts.push(
            spirv
                .get(at..at + count)
                .expect("an instruction past the module's end"),
        );
        at += count;
    }
    let undefs: BTreeSet<u32> = insts
        .iter()
        .filter(|inst| inst[0] & 0xFFFF == OP_UNDEF)
        .filter_map(|inst| inst.get(2).copied())
        .collect();
    insts
        .iter()
        .filter(|inst| inst[0] & 0xFFFF == OP_STORE)
        .filter(|inst| inst.get(2).is_some_and(|object| undefs.contains(object)))
        .count()
}

/// One dispatch of kernel `entry` of `bitcode` through the runtime in one threadgroup of `n`
/// threads, `limits` in buffer 0 and `n` poisoned words in buffer 1: the words it wrote, and the
/// stores of an undefined value in the module the translator compiles for the kernel.
fn run(entry: &str, bitcode: &[u8], n: u32, limits: [u32; 2]) -> (Vec<u32>, usize) {
    super::gpu_fuzz::device_wait();
    let poison: Vec<u8> = (0..n).flat_map(|_| POISON.to_le_bytes()).collect();
    let length = poison.len() as u64;
    let case = ComputeCase {
        out_of_bounds: Vec::new(),
        creation_bound: crate::budgets::PIPELINE_CREATION_CEILING,
        entry: entry.into(),
        constants: FunctionConstantValues::defaults(),
        buffers: BTreeMap::from([
            (0, limits.iter().flat_map(|w| w.to_le_bytes()).collect()),
            (1, poison),
        ]),
        argument_buffer_buffers: BTreeMap::new(),
        argument_buffer_textures: BTreeMap::new(),
        argument_buffer_samplers: BTreeMap::new(),
        textures: BTreeMap::new(),
        samplers: BTreeMap::new(),
        threadgroup_lengths: BTreeMap::new(),
        imageblock: None,
        dispatch: Dispatch::Threads {
            threads: [n, 1, 1],
            threads_per_group: [n, 1, 1],
        },
        output: ComputeOutput::Buffer {
            binding: 1,
            offset: 0,
            length,
        },
    };
    let run = super::engine_compute::run_through_runtime(&case, bitcode)
        .unwrap_or_else(|error| panic!("{entry} over {n} threads, limits {limits:?}: {error}"));
    (
        words(&run.output),
        undefined_stores(&compiled(entry, bitcode)),
    )
}

/// The SPIR-V words of kernel `entry` of `bitcode` as the translator compiles it for
/// `runtime-floor`, with no function-constant value.
fn compiled(entry: &str, bitcode: &[u8]) -> std::sync::Arc<[u32]> {
    use metal2vulkan::library::Library;
    use metal2vulkan::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};
    let device = crate::engine::load_device("runtime-floor").expect("the runtime-floor profile");
    let function = Library::from_air(bitcode)
        .and_then(|library| library.function(entry))
        .unwrap_or_else(|error| panic!("{entry}: {error}"));
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(function)));
    device
        .compile_compute_pipeline(&descriptor, &metal2vulkan::Budget::standard())
        .unwrap_or_else(|error| panic!("compile {entry}: {error}"))
        .stage
        .spirv
}

/// What thread `t` of a kernel writes for `limits`.
type Answer = fn([u32; 2], u32) -> u32;

/// Each exit kernel's threads write what its source gives at every trip bound, on the module that
/// ends its trips' values at each trip's top.
#[test]
fn loop_exits_read_the_values_their_own_trips_defined_on_the_device() {
    let kernels: [(&str, &str, Answer); 2] = [
        ("trip_exits_k", TRIP_EXITS, |[n, _], t| {
            loop_kernels::trip_exits(n, t)
        }),
        ("nested_exits_k", NESTED_EXITS, |[n, m], t| {
            loop_kernels::nested_exits(n, m, t)
        }),
    ];
    for (name, body, answer) in kernels {
        let bitcode = super::gpu_fuzz::device::assemble(name, &loop_kernels::module(name, body))
            .expect("assemble");
        let mut ran = 0;
        for n in [32, 64] {
            for limits in BOUNDS {
                let (out, stores) = run(name, &bitcode, n, limits);
                if ran == 0 {
                    eprintln!("loop_exits {name}: {stores} stores of an undefined value ran");
                }
                assert!(stores > 0, "{name}: no value ended at a trip's top");
                let want: Vec<u32> = (0..n).map(|t| answer(limits, t)).collect();
                assert_eq!(out, want, "{name} over {n} threads, limits {limits:?}");
                ran += 1;
            }
        }
        eprintln!("loop_exits {name}: {ran} dispatches gave their source's words");
    }
}

/// A thread that leaves the predicated barrier loop early keeps its last trip's value: the walk
/// runs the later trips for it too, so no store ends a value at their tops.
#[test]
fn a_thread_that_leaves_a_vote_loop_early_keeps_its_value_on_the_device() {
    let name = "loop_leaver_k";
    let bitcode = super::gpu_fuzz::device::assemble(name, &loop_kernels::module(name, LOOP_LEAVER))
        .expect("assemble");
    for n in [8, 32, 40, 64, 96, 128] {
        let (out, stores) = run(name, &bitcode, n, [0, 0]);
        assert_eq!(
            stores, 0,
            "{name}: a predicated loop's value ended at a trip's top"
        );
        let want: Vec<u32> = (0..n).map(loop_kernels::loop_leaver).collect();
        assert_eq!(out, want, "{name} over {n} threads");
    }
    eprintln!(
        "loop_exits {name}: 6 dispatches gave their source's words, 0 stores of an undefined value"
    );
}
