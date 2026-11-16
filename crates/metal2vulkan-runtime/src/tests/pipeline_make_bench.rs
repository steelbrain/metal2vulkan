//! What one warm making of a compute pipeline state, and of a library, costs the calling thread,
//! step by step, as a guest's pipeline compile makes one (ignored, `PIPELINE-MAKE`; run in
//! release): the function (`newFunctionWithName:constantValues:error:`), the state
//! (`newComputePipelineStateWithDescriptor:`), the six properties a guest's reply reads, and the
//! state's release, under the product's default options with a pipeline record hook that only
//! counts, as a consumer's hands each record to a writer thread.
//!
//! Each round makes [`STATES`] states of a kernel, one for each set of its constants' values, after
//! one cold round on the same device. Two kinds of warm making:
//!
//! - **kept**: every state of the kernel released before the round, so each making takes back the
//!   objects a released variant left (`pipeline/kept.rs`);
//! - **live**: each state made again while the round before holds its twin, so each making shares
//!   the live variant (`pipeline/memo.rs`).
//!
//! Both with and without the record hook, which says what the record's making costs a creation.
//! Then a library's making from a metallib of [`LIBRARY_SIZES`] functions, and the lookup of its
//! last function by name.
//!
//! [`pipeline_make_profile`] is a target for a profiler: [`PROFILE_ROUNDS`] rounds of each kind.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). The kernels
//! are the compute creation bench's (`compute_create_bench.rs`).

use super::blit_holds::{options, Scratch};
use super::compute_create_bench::{library, Kernel, CONV, SMALL};
use crate::{ComputePipelineState, Device, Function, Library, PipelineOptions};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::metallib::{metallib, wrap, MtlbFunction};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The states of a round: one for each set of constants.
const STATES: u16 = 8;

/// The timed rounds of each kind.
const ROUNDS: usize = 64;

/// A device over the shader cache in `scratch`, with a pipeline record hook that counts the
/// records in `records` where given.
fn device_of(scratch: &Scratch, records: Option<&Arc<AtomicU64>>) -> Device {
    let mut options = options(&scratch.0);
    if let Some(records) = records {
        let records = Arc::clone(records);
        options.pipeline_records = Some(Arc::new(move |record| {
            drop(record);
            records.fetch_add(1, Ordering::Relaxed);
        }));
    }
    Device::system_default_with(&options).unwrap()
}

/// One kind's times, by step.
#[derive(Default)]
struct Steps {
    function: Vec<Duration>,
    state: Vec<Duration>,
    properties: Vec<Duration>,
    release: Vec<Duration>,
}

/// What a guest's reply reads of a made state, summed so nothing is optimised out.
fn properties(state: &ComputePipelineState) -> u64 {
    u64::from(state.max_total_threads_per_threadgroup())
        + u64::from(state.thread_execution_width())
        + u64::from(state.static_threadgroup_memory_length())
        + u64::from(state.support_indirect_command_buffers())
        + state.gpu_resource_id().0
        + state.unique_identifier()
}

/// Make a state of `kernel` for each set of its constants on `device` from `library`: the states,
/// each step's time into `steps`.
fn round(
    kernel: &Kernel,
    library: &Library,
    device: &Device,
    steps: &mut Steps,
) -> Vec<(Function, ComputePipelineState)> {
    let options = PipelineOptions::default();
    let mut made = Vec::with_capacity(usize::from(STATES));
    let mut read = 0;
    for n in 0..STATES {
        let constants = (kernel.constants)(n);
        let began = Instant::now();
        let function = library
            .new_function_with_name_constant_values(kernel.name, &constants)
            .unwrap();
        steps.function.push(began.elapsed());
        let began = Instant::now();
        let descriptor =
            ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        let state = device
            .new_compute_pipeline_state(&descriptor, &options)
            .unwrap();
        steps.state.push(began.elapsed());
        let began = Instant::now();
        read += properties(&state);
        steps.properties.push(began.elapsed());
        made.push((function, state));
    }
    assert!(read > 0);
    made
}

/// Release `made`, each state and its function, timed into `steps`.
fn release(made: Vec<(Function, ComputePipelineState)>, steps: &mut Steps) {
    for pair in made {
        let began = Instant::now();
        drop(pair);
        steps.release.push(began.elapsed());
    }
}

/// The 10th, 50th and 90th percentiles of `times`, in microseconds.
fn spread(times: &[Duration]) -> String {
    let mut times = times.to_vec();
    times.sort();
    let us = |at: usize| times[at.min(times.len() - 1)].as_secs_f64() * 1e6;
    let n = times.len();
    format!(
        "p10 {:.1} p50 {:.1} p90 {:.1} us",
        us(n / 10),
        us(n / 2),
        us(n * 9 / 10)
    )
}

/// Print one kind's steps.
fn report(device: &Device, kernel: &Kernel, kind: &str, steps: &Steps) {
    eprintln!(
        "PIPELINE-MAKE {} {} {kind}: {} makings; function {}; state {}; properties {}; release {}",
        device.name(),
        kernel.name,
        steps.state.len(),
        spread(&steps.function),
        spread(&steps.state),
        spread(&steps.properties),
        spread(&steps.release),
    );
}

/// The two warm kinds of `kernel`'s makings on a device with the record hook or without it.
fn kinds(kernel: &Kernel, hooked: bool) {
    let scratch = Scratch::new("pipeline-make");
    let records = Arc::new(AtomicU64::new(0));
    let device = device_of(&scratch, hooked.then_some(&records));
    let library = library(&device, kernel);
    let mut cold = Steps::default();
    let made = round(kernel, &library, &device, &mut cold);
    report(&device, kernel, &format!("cold, hook {hooked}"), &cold);
    release(made, &mut cold);
    let (mut kept, mut live) = (Steps::default(), Steps::default());
    let before = device.work_counts();
    for _ in 0..ROUNDS {
        let first = round(kernel, &library, &device, &mut kept);
        let again = round(kernel, &library, &device, &mut live);
        release(first, &mut kept);
        release(again, &mut live);
    }
    let after = device.work_counts();
    report(&device, kernel, &format!("kept, hook {hooked}"), &kept);
    report(&device, kernel, &format!("live, hook {hooked}"), &live);
    eprintln!(
        "PIPELINE-MAKE {} {} warm, hook {hooked}: records {}, translated {}, shared {}, variants \
         taken back {}, modules made {}, layouts made {}, VkPipelines made {}",
        device.name(),
        kernel.name,
        records.load(Ordering::Relaxed),
        after.functions_translated - before.functions_translated,
        after.functions_shared - before.functions_shared,
        after.variants_taken_back - before.variants_taken_back,
        after.shader_modules_made - before.shader_modules_made,
        after.pipeline_layouts_made - before.pipeline_layouts_made,
        after.variant_pipelines_made - before.variant_pipelines_made,
    );
}

/// The function counts of the libraries [`libraries`] makes.
const LIBRARY_SIZES: [usize; 3] = [1, 256, 2048];

/// The makings of each library timed.
const LIBRARY_MAKES: usize = 8;

/// The lookups of a library's last function timed.
const LOOKUPS: usize = 256;

/// A library's making from a metallib of each of [`LIBRARY_SIZES`] functions, each the small
/// kernel's bitcode under a name of its own, and the lookup of its last function by name.
fn libraries(device: &Device) {
    let ll = metal2vulkan_devtools::fixtures_dir().join(format!("{}.ll", SMALL.name));
    let bitcode = wrap(&metal2vulkan_devtools::llvm_assemble(&ll).unwrap());
    for size in LIBRARY_SIZES {
        let names: Vec<String> = (0..size).map(|i| format!("kernel_{i:05}")).collect();
        let functions: Vec<MtlbFunction<'_>> = (names.iter())
            .map(|name| MtlbFunction {
                name,
                type_code: 2,
                bitcode: bitcode.clone(),
                constants: Vec::new(),
                public: Vec::new(),
                tags: Vec::new(),
            })
            .collect();
        let bytes = metallib(&functions);
        let mut makes = Vec::with_capacity(LIBRARY_MAKES);
        let mut releases = Vec::with_capacity(LIBRARY_MAKES);
        for _ in 0..LIBRARY_MAKES {
            let began = Instant::now();
            let library = device.new_library_with_data(&bytes).unwrap();
            makes.push(began.elapsed());
            let began = Instant::now();
            drop(library);
            releases.push(began.elapsed());
        }
        let library = device.new_library_with_data(&bytes).unwrap();
        let last = names.last().unwrap();
        let lookups: Vec<Duration> = (0..LOOKUPS)
            .map(|_| {
                let began = Instant::now();
                let function = library.new_function_with_name(last).unwrap();
                let took = began.elapsed();
                drop(function);
                took
            })
            .collect();
        eprintln!(
            "PIPELINE-MAKE-LIBRARY {} {size} functions, {} bytes: make {}; release {}; its last \
             function by name {}",
            device.name(),
            bytes.len(),
            spread(&makes),
            spread(&releases),
            spread(&lookups),
        );
    }
}

/// The bench (the module docs).
#[test]
#[ignore = "a release-mode benchmark"]
fn pipeline_make_bench() {
    for kernel in [SMALL, CONV] {
        for hooked in [true, false] {
            kinds(&kernel, hooked);
        }
    }
    let scratch = Scratch::new("pipeline-make-libraries");
    libraries(&device_of(&scratch, None));
}

/// The rounds [`pipeline_make_profile`] makes of each kind.
const PROFILE_ROUNDS: usize = 4000;

/// For a profiler (ignored; run in release): one cold round of [`SMALL`] on a device with the
/// record hook, then [`PROFILE_ROUNDS`] rounds of each warm kind, printed as the bench prints.
#[test]
#[ignore = "a release-mode profile target"]
fn pipeline_make_profile() {
    let scratch = Scratch::new("pipeline-make-profile");
    let records = Arc::new(AtomicU64::new(0));
    let device = device_of(&scratch, Some(&records));
    let library = library(&device, &SMALL);
    let mut cold = Steps::default();
    release(round(&SMALL, &library, &device, &mut cold), &mut cold);
    let (mut kept, mut live) = (Steps::default(), Steps::default());
    for _ in 0..PROFILE_ROUNDS {
        let first = round(&SMALL, &library, &device, &mut kept);
        let again = round(&SMALL, &library, &device, &mut live);
        release(first, &mut kept);
        release(again, &mut live);
    }
    report(&device, &SMALL, "profiled kept, hook true", &kept);
    report(&device, &SMALL, "profiled live, hook true", &live);
}
