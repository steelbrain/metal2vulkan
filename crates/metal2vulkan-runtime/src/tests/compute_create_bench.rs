//! What a warm compute pipeline state's creation costs the calling thread (ignored,
//! `COMPUTE-CREATE`; run in release): `new_function_with_name_constant_values` and
//! `new_compute_pipeline_state` of a kernel with a function constant, as a guest's
//! `newFunctionWithName:constantValues:error:` and `newComputePipelineStateWithDescriptor:` make
//! one, on the device the cfg selects under the product's default options.
//!
//! Each round makes [`STATES`] states, one for each value of the kernel's constant, keeps them
//! until the round ends, as a process keeps its states, and drops them all:
//!
//! - **cold**: a fresh device over an empty shader cache, which translates each;
//! - **warm, same device**: [`PASSES`] more rounds on that device, which finds each translation
//!   in its memo;
//! - **warm, new device**: [`PASSES`] rounds, each on a new device over the same shader cache
//!   directory, which finds each translation on disk.
//!
//! Printed for each: the median and slowest function and creation times, and the round's work
//! counts (translations, cache reads, the shader modules, layouts and `VkPipeline`s made).
//!
//! [`compute_create_profile`] is a target for a profiler: one cold round, then many warm rounds
//! on one device.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). The kernel
//! is the owned fixture `kernel_inverse_hyperbolics_and_a_gated_bias`
//! (`validation/fixtures/public/*.ll`, `llvm-as`).

use super::blit_holds::{options, Scratch};
use crate::{ComputePipelineState, Device, Library, PipelineOptions, WorkCounts};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::DataType;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::path::Path;
use std::time::{Duration, Instant};

/// The kernel, and its `short` function constant.
const KERNEL: &str = "kernel_inverse_hyperbolics_and_a_gated_bias";
const CONSTANT: &str = "has_bias";

/// The states of a round: one for each value of the constant.
const STATES: i16 = 8;

/// The warm rounds of each kind.
const PASSES: usize = 4;

/// The kernel's library on `device`.
fn library(device: &Device) -> Library {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public")
        .join(format!("{KERNEL}.ll"));
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    device.new_library_with_compile_library(
        metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
    )
}

/// One round's times: each function's and each creation's.
#[derive(Default)]
struct Times {
    functions: Vec<Duration>,
    creations: Vec<Duration>,
}

/// Make a state for each value of the constant on `device` from `library`, kept until all are
/// made, then dropped: their times into `times`.
fn round(library: &Library, device: &Device, times: &mut Times) {
    let options = PipelineOptions::default();
    let mut states: Vec<ComputePipelineState> = Vec::new();
    for value in 0..STATES {
        let constant = FunctionConstantValue::new(DataType::Short, value.to_le_bytes().to_vec());
        let constants = FunctionConstantValues::defaults().with_name(CONSTANT, constant);
        let began = Instant::now();
        let function = library
            .new_function_with_name_constant_values(KERNEL, &constants)
            .unwrap();
        times.functions.push(began.elapsed());
        let descriptor =
            ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        let began = Instant::now();
        let state = device
            .new_compute_pipeline_state(&descriptor, &options)
            .unwrap();
        times.creations.push(began.elapsed());
        states.push(state);
    }
    drop(states);
}

/// The median and slowest of `times`, in microseconds.
fn spread(times: &[Duration]) -> String {
    let mut times = times.to_vec();
    times.sort();
    let us = |d: Duration| d.as_secs_f64() * 1e6;
    format!(
        "median {:.1} us, slowest {:.1} us",
        us(times[times.len() / 2]),
        us(times[times.len() - 1])
    )
}

/// What a round did, from `before` to `after`.
fn work(before: &WorkCounts, after: &WorkCounts) -> String {
    format!(
        "translated {}, shared {}, entries specialized {} shared {}, modules made {}, layouts made \
         {}, VkPipelines made {} (at creation {}), driver compute {:.0} us, cache reads {:.0} us",
        after.functions_translated - before.functions_translated,
        after.functions_shared - before.functions_shared,
        after.stage_entries_specialized - before.stage_entries_specialized,
        after.stage_entries_shared - before.stage_entries_shared,
        after.shader_modules_made - before.shader_modules_made,
        after.pipeline_layouts_made - before.pipeline_layouts_made,
        after.variant_pipelines_made - before.variant_pipelines_made,
        after.creation_pipelines_made - before.creation_pipelines_made,
        (after.compute_pipeline_create_nanos - before.compute_pipeline_create_nanos) as f64 / 1e3,
        (after.shader_cache_read_nanos - before.shader_cache_read_nanos) as f64 / 1e3,
    )
}

/// Print one kind of round's figures.
fn report(device: &Device, kind: &str, times: &Times, work: &str) {
    eprintln!(
        "COMPUTE-CREATE {} {kind}: {} creations; function {}; creation {}; {work}",
        device.name(),
        times.creations.len(),
        spread(&times.functions),
        spread(&times.creations),
    );
}

/// The bench (the module docs).
#[test]
#[ignore = "a release-mode benchmark"]
fn compute_create_bench() {
    let scratch = Scratch::new("compute-create");
    let first = Device::system_default_with(&options(&scratch.0)).unwrap();
    let library_first = library(&first);
    let (before, mut times) = (first.work_counts(), Times::default());
    round(&library_first, &first, &mut times);
    report(&first, "cold", &times, &work(&before, &first.work_counts()));
    let (before, mut times) = (first.work_counts(), Times::default());
    for _ in 0..PASSES {
        round(&library_first, &first, &mut times);
    }
    report(
        &first,
        "warm, same device",
        &times,
        &work(&before, &first.work_counts()),
    );
    drop((library_first, first));
    let mut times = Times::default();
    let mut works = Vec::new();
    for _ in 0..PASSES {
        let device = Device::system_default_with(&options(&scratch.0)).unwrap();
        let library = library(&device);
        let before = device.work_counts();
        round(&library, &device, &mut times);
        works.push(work(&before, &device.work_counts()));
        if works.len() == PASSES {
            report(&device, "warm, new device", &times, &works.join(" | "));
        }
    }
}

/// The rounds [`compute_create_profile`] makes once the first translated.
const PROFILE_ROUNDS: usize = 2000;

/// For a profiler (ignored; run in release): one cold round, then [`PROFILE_ROUNDS`] warm rounds
/// on the same device, which find each translation in its memo, printed as the bench prints.
#[test]
#[ignore = "a release-mode profile target"]
fn compute_create_profile() {
    let scratch = Scratch::new("compute-create-profile");
    let device = Device::system_default_with(&options(&scratch.0)).unwrap();
    let library = library(&device);
    round(&library, &device, &mut Times::default());
    let (before, mut times) = (device.work_counts(), Times::default());
    for _ in 0..PROFILE_ROUNDS {
        round(&library, &device, &mut times);
    }
    report(
        &device,
        "profiled, same device",
        &times,
        &work(&before, &device.work_counts()),
    );
}
