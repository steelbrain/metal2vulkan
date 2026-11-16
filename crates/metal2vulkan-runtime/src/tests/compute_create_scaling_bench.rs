//! What a warm compute pipeline state's creation costs as the device holds more objects
//! (ignored, `COMPUTE-CREATE-SCALING`; run in release), on several threads at once, as a guest's
//! pipeline compiles make them.
//!
//! The device first holds a base of `n` live states (each of its own SPIR-V words, so each holds
//! its own shader module), for each `n` of [`LIVE`]; then each of `threads` threads makes and
//! releases states from a small working set of its own, [`ROUNDS`] times over: each making after
//! the first takes back the objects the released variant left (`pipeline/kept.rs`), as a guest that
//! releases a state and makes it again does. Printed for each `n` and thread count: the median,
//! 90th and 99th percentile creation times, the device's maps' sizes (what a settle of them
//! walks), and the round's work counts (translations, variants taken back, modules made).
//!
//! The kernel is the owned fixture `kernel_vector_function_constant`
//! (`validation/fixtures/public`), assembled by `llvm-as`, whose `uint4` constant's lanes are
//! folded into the value it stores: each sum is a module of its own.
//!
//! [`compute_create_scaling_profile_empty`] and [`compute_create_scaling_profile_full`] are
//! targets for a profiler: four threads' churn alone, over no live base and over the largest.
//!
//! [`compute_create_key_costs`] (`COMPUTE-CREATE-KEYS`) times the keys a creation derives on its
//! own: the function's compile key and the compute pipeline's key, and the `Debug` text each
//! formats today, and the layout text that names a new variant's makings.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::blit_holds::{options, Scratch};
use crate::{ComputePipelineState, Device, Library, PipelineOptions, WorkCounts};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::DataType;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The kernel.
const KERNEL: &str = "kernel_vector_function_constant";

/// The live base sizes measured, in order: the base grows to each.
const LIVE: [u32; 5] = [0, 512, 1024, 2048, 4096];

/// The thread counts measured at each base size.
const THREADS: [u32; 2] = [1, 4];

/// The states of each thread's working set.
const WORKING: u32 = 16;

/// How many times each thread makes and releases its working set at each base size.
const ROUNDS: u32 = 24;

/// Where a working set's sums start: past every base state's.
const WORKING_BASE: u32 = 1 << 24;

/// The kernel's library on `device`.
fn library(device: &Device) -> Library {
    let ll = metal2vulkan_devtools::fixtures_dir().join(format!("{KERNEL}.ll"));
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    device.new_library_with_compile_library(
        metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
    )
}

/// The kernel's constants whose stored sum is `sum`.
fn constants(sum: u32) -> FunctionConstantValues {
    let lanes: Vec<u8> = [sum, 0, 0, 0]
        .iter()
        .flat_map(|l| l.to_le_bytes())
        .collect();
    FunctionConstantValues::defaults().with_name(
        "vector_value",
        FunctionConstantValue::new(DataType::UInt4, lanes),
    )
}

/// The state whose stored sum is `sum`, made on `device` from `library`, and how long the
/// creation took.
fn state(device: &Device, library: &Library, sum: u32) -> (ComputePipelineState, Duration) {
    let function = library
        .new_function_with_name_constant_values(KERNEL, &constants(sum))
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    let began = Instant::now();
    let state = device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    (state, began.elapsed())
}

/// The states of sums `from..to`, made on `threads` threads, each thread with a library of its
/// own.
fn base(device: &Device, from: u32, to: u32, threads: u32) -> Vec<ComputePipelineState> {
    std::thread::scope(|scope| {
        let parts: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let library = library(device);
                    (from..to)
                        .filter(|sum| sum % threads == t)
                        .map(|sum| state(device, &library, sum).0)
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        parts
            .into_iter()
            .flat_map(|part| part.join().unwrap())
            .collect()
    })
}

/// Each of `threads` threads makes and releases its working set `rounds` times: every creation's
/// time.
fn churn(device: &Device, threads: u32, rounds: u32) -> Vec<Duration> {
    std::thread::scope(|scope| {
        let parts: Vec<_> = (0..threads)
            .map(|t| {
                scope.spawn(move || {
                    let library = library(device);
                    let first = WORKING_BASE + t * WORKING;
                    let mut times = Vec::new();
                    for _ in 0..rounds {
                        for sum in first..first + WORKING {
                            let (state, took) = state(device, &library, sum);
                            drop(state);
                            times.push(took);
                        }
                    }
                    times
                })
            })
            .collect();
        parts
            .into_iter()
            .flat_map(|part| part.join().unwrap())
            .collect()
    })
}

/// The median, 90th and 99th percentile of `times`, in microseconds.
fn spread(times: &[Duration]) -> String {
    let mut times = times.to_vec();
    times.sort();
    let at = |q: f64| times[((times.len() - 1) as f64 * q) as usize].as_secs_f64() * 1e6;
    format!(
        "p50 {:.1} us, p90 {:.1} us, p99 {:.1} us",
        at(0.5),
        at(0.9),
        at(0.99)
    )
}

/// The device's maps' sizes: what a settle of them walks.
fn maps(device: &Device) -> String {
    let shared = device.shared();
    let modules = crate::device::lock(&shared.shader_modules).len();
    let layouts = crate::device::lock(&shared.pipeline_layouts).len();
    let sets = crate::device::lock(&shared.sampler_sets).len();
    let libraries = shared.libraries.as_ref().map_or(0, |l| l.len());
    let (kept, _) = crate::device::lock(&shared.kept).held();
    format!(
        "modules {modules}, layouts {layouts}, sampler sets {sets}, libraries and links \
         {libraries}, kept instances {kept}"
    )
}

/// What the churn did, from `before` to `after`.
fn work(before: &WorkCounts, after: &WorkCounts) -> String {
    format!(
        "translated {}, taken back {}, modules made {}, VkPipelines made {}",
        after.functions_translated - before.functions_translated,
        after.variants_taken_back - before.variants_taken_back,
        after.shader_modules_made - before.shader_modules_made,
        after.variant_pipelines_made - before.variant_pipelines_made,
    )
}

/// A device over the shader cache in `scratch`, with a record hook that only counts into
/// `records`, as a consumer's hands each record to a writer thread; each thread's working set
/// made once on it, cold, and released, so every later making of one is warm.
fn device_of(scratch: &Scratch, records: &Arc<AtomicU64>) -> Device {
    let mut options = options(&scratch.0);
    let counted = Arc::clone(records);
    options.pipeline_records = Some(Arc::new(move |record| {
        drop(record);
        counted.fetch_add(1, Ordering::Relaxed);
    }));
    let device = Device::system_default_with(&options).unwrap();
    churn(&device, *THREADS.iter().max().unwrap(), 1);
    device
}

/// The bench (the module docs).
#[test]
#[ignore = "a release-mode benchmark"]
fn compute_create_scaling_bench() {
    let scratch = Scratch::new("compute-create-scaling");
    let records = Arc::new(AtomicU64::new(0));
    let device = device_of(&scratch, &records);
    let mut live: Vec<ComputePipelineState> = Vec::new();
    for n in LIVE {
        let began = Instant::now();
        let made = live.len() as u32;
        live.extend(base(&device, made, n, 4));
        let grown = began.elapsed();
        for threads in THREADS {
            let before = device.work_counts();
            let began = Instant::now();
            let times = churn(&device, threads, ROUNDS);
            let wall = began.elapsed();
            eprintln!(
                "COMPUTE-CREATE-SCALING {} live {n} threads {threads}: {} creations in {:.1} ms \
                 ({:.1} us each); {}; {}; {} (base grown in {:.1} s)",
                device.name(),
                times.len(),
                wall.as_secs_f64() * 1e3,
                wall.as_secs_f64() * 1e6 / times.len() as f64,
                spread(&times),
                maps(&device),
                work(&before, &device.work_counts()),
                grown.as_secs_f64(),
            );
        }
    }
    eprintln!(
        "COMPUTE-CREATE-SCALING {}: {} records handed over",
        device.name(),
        records.load(Ordering::Relaxed)
    );
    drop(live);
}

/// The rounds of each thread's churn in a profile target.
const PROFILE_ROUNDS: u32 = 400;

/// Four threads' churn of [`PROFILE_ROUNDS`] rounds over a live base of `live` states, printed as
/// the bench prints.
fn profile(live: u32) {
    let scratch = Scratch::new("compute-create-scaling-profile");
    let records = Arc::new(AtomicU64::new(0));
    let device = device_of(&scratch, &records);
    let base = base(&device, 0, live, 4);
    let began = Instant::now();
    let times = churn(&device, 4, PROFILE_ROUNDS);
    let wall = began.elapsed();
    eprintln!(
        "COMPUTE-CREATE-SCALING-PROFILE {} live {live} threads 4: {} creations in {:.1} ms; {}; {}",
        device.name(),
        times.len(),
        wall.as_secs_f64() * 1e3,
        spread(&times),
        maps(&device),
    );
    drop(base);
}

/// For a profiler (ignored; run in release): four threads' churn over no live base, what a
/// creation costs apart from what the device holds.
#[test]
#[ignore = "a release-mode profile target"]
fn compute_create_scaling_profile_empty() {
    profile(0);
}

/// For a profiler (ignored; run in release): four threads' churn over the largest live base.
#[test]
#[ignore = "a release-mode profile target"]
fn compute_create_scaling_profile_full() {
    profile(*LIVE.iter().max().unwrap());
}

/// The calls each of [`compute_create_key_costs`]' pieces is timed over.
const KEY_CALLS: u32 = 20_000;

/// The mean of [`KEY_CALLS`] runs of `piece`, in microseconds.
fn mean_of<T>(mut piece: impl FnMut() -> T) -> f64 {
    let began = Instant::now();
    for _ in 0..KEY_CALLS {
        std::hint::black_box(piece());
    }
    began.elapsed().as_secs_f64() * 1e6 / f64::from(KEY_CALLS)
}

/// What the keys a warm creation of the bench's kernel derives cost on their own (ignored,
/// `COMPUTE-CREATE-KEYS`; run in release), each the mean of [`KEY_CALLS`] calls: the function's
/// compile key (`Variant::compiled`), the compute pipeline's key of the compiled kernel
/// (`Sources::kernel`), the `Debug` text of the lowering inputs and of the descriptor's fields
/// those keys format, and the text of the layout that names a new variant's makings
/// (`objects::variant_content`).
#[test]
#[ignore = "a release-mode benchmark"]
fn compute_create_key_costs() {
    use metal2vulkan::pipeline::LoweringInputs;
    let scratch = Scratch::new("compute-create-keys");
    let device = Device::system_default_with(&options(&scratch.0)).unwrap();
    let library = library(&device);
    let function = library
        .new_function_with_name_constant_values(KERNEL, &constants(7))
        .unwrap();
    let descriptor = function.function_descriptor();
    let compile = device.compile_device();
    let inputs = LoweringInputs::default();
    let budget = metal2vulkan::Budget::standard();
    let compiled = compile
        .compile_function(&descriptor, &inputs, &budget)
        .unwrap();
    let kernel = ComputePipelineDescriptor::new(StageSource::Compiled(compiled));
    let pipeline = compile.compile_compute_pipeline(&kernel, &budget).unwrap();
    let function_key = mean_of(|| compile.function_compile_key(&descriptor, &inputs));
    let pipeline_key = mean_of(|| compile.compute_pipeline_key(&kernel));
    let inputs_text = mean_of(|| {
        format!(
            "LoweringInputs {{ vertex_fetch: {:?}, storage_images: {:?}, psb_slots: {:?} }}",
            inputs.vertex_fetch, inputs.storage_images, inputs.psb_slots
        )
    });
    let fields_text = mean_of(|| {
        format!(
            "{:?}",
            (
                kernel.stage_input.as_ref(),
                kernel.max_total_threads_per_threadgroup,
                kernel.threadgroup_size_is_multiple_of_execution_width,
                kernel.support_indirect_command_buffers,
            )
        )
    });
    let layout_text = mean_of(|| format!("{:?}", pipeline.layout));
    eprintln!(
        "COMPUTE-CREATE-KEYS {}: function_compile_key {function_key:.2} us (its inputs' text \
         {inputs_text:.2} us); compute_pipeline_key {pipeline_key:.2} us (its fields' text \
         {fields_text:.2} us); a variant's layout text {layout_text:.2} us (means of {KEY_CALLS})",
        device.name(),
    );
}
