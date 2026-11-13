//! What a warm compute pipeline state's creation costs the calling thread (ignored,
//! `COMPUTE-CREATE`; run in release): `new_function_with_name_constant_values` and
//! `new_compute_pipeline_state` of a kernel with function constants, as a guest's
//! `newFunctionWithName:constantValues:error:` and `newComputePipelineStateWithDescriptor:` make
//! one, on the device the cfg selects under the product's default options. Two kernels: a small
//! one with one constant, and [`CONV`], a convolution over texture arrays of a framework
//! convolution kernel's shape, with ten.
//!
//! Each round makes [`STATES`] states of a kernel, one for each set of its constants' values,
//! keeps them until the round ends, as a process keeps its states, and drops them all:
//!
//! - **cold**: a fresh device over an empty shader cache, which translates each;
//! - **warm, same device**: [`PASSES`] more rounds on that device, which finds each translation
//!   in its memo;
//! - **warm, new device**: [`PASSES`] rounds, each on a new device over the same shader cache
//!   directory, which finds each translation on disk;
//! - **warm, memo of 1 byte**: [`PASSES`] rounds on one device whose memo keeps nothing past a
//!   round, as a device whose memo a run's many states evicted: each round finds each on disk.
//!
//! Printed for each: the median and slowest function and creation times, and the round's work
//! counts (translations, cache reads, the shader modules, layouts and `VkPipeline`s made).
//!
//! [`compute_create_profile`] is a target for a profiler: one cold round of [`CONV`], then many
//! rounds on its device with a memo of one byte.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). The kernels
//! are the owned fixtures `kernel_inverse_hyperbolics_and_a_gated_bias`
//! (`validation/fixtures/public/*.ll`) and `ccb_conv_array` (`validation/fixtures/public/engine`),
//! assembled by `llvm-as`.

use super::blit_holds::{options, Scratch};
use crate::{ComputePipelineState, Device, Library, PipelineOptions, WorkCounts};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::DataType;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::path::Path;
use std::time::{Duration, Instant};

/// A kernel of the bench: its fixture's directory under `validation/fixtures/public`, its name,
/// and the constants of its state `n`.
struct Kernel {
    directory: &'static str,
    name: &'static str,
    constants: fn(u16) -> FunctionConstantValues,
}

/// The small kernel: `has_bias`, a `short`, at `n`.
const SMALL: Kernel = Kernel {
    directory: "",
    name: "kernel_inverse_hyperbolics_and_a_gated_bias",
    constants: |n| FunctionConstantValues::defaults().with_name("has_bias", short(n as i16)),
};

/// The convolution: a 3x3 kernel, its strides, bias, activation, input slices and edges moving
/// with `n`.
const CONV: Kernel = Kernel {
    directory: "engine",
    name: "ccb_conv_array",
    constants: |n| {
        let ushort =
            |v: u16| FunctionConstantValue::new(DataType::UShort, v.to_le_bytes().to_vec());
        let pair = |ty, a: u16, b: u16| {
            let mut bytes = a.to_le_bytes().to_vec();
            bytes.extend(b.to_le_bytes());
            FunctionConstantValue::new(ty, bytes)
        };
        let flag = |v: bool| FunctionConstantValue::new(DataType::Bool, vec![v as u8]);
        FunctionConstantValues::defaults()
            .with_name("ccb_kernel_width", ushort(3))
            .with_name("ccb_kernel_height", ushort(3))
            .with_name("ccb_stride", pair(DataType::UShort2, 1 + n % 2, 1 + n % 2))
            .with_name("ccb_dilation", pair(DataType::UShort2, 1, 1))
            .with_name("ccb_has_bias", flag(n % 2 == 1))
            .with_name("ccb_activation", ushort(n % 8))
            .with_name("ccb_offset", pair(DataType::Short2, 0xffff, 0xffff))
            .with_name("ccb_input_slices", ushort(4 + n))
            .with_name("ccb_clamp_edges", flag(n % 4 == 0))
            .with_name(
                "ccb_alpha",
                FunctionConstantValue::new(DataType::Float, 0.1f32.to_le_bytes().to_vec()),
            )
    },
};

/// A `short` constant's value.
fn short(v: i16) -> FunctionConstantValue {
    FunctionConstantValue::new(DataType::Short, v.to_le_bytes().to_vec())
}

/// The states of a round: one for each set of constants.
const STATES: u16 = 8;

/// The warm rounds of each kind.
const PASSES: usize = 4;

/// `kernel`'s library on `device`.
fn library(device: &Device, kernel: &Kernel) -> Library {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public")
        .join(kernel.directory)
        .join(format!("{}.ll", kernel.name));
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

/// Make a state of `kernel` for each set of its constants on `device` from `library`, kept until
/// all are made, then dropped: their times into `times`.
fn round(kernel: &Kernel, library: &Library, device: &Device, times: &mut Times) {
    let options = PipelineOptions::default();
    let mut states: Vec<ComputePipelineState> = Vec::new();
    for n in 0..STATES {
        let constants = (kernel.constants)(n);
        let began = Instant::now();
        let function = library
            .new_function_with_name_constant_values(kernel.name, &constants)
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
fn report(device: &Device, kernel: &Kernel, kind: &str, times: &Times, work: &str) {
    eprintln!(
        "COMPUTE-CREATE {} {} {kind}: {} creations; function {}; creation {}; {work}",
        device.name(),
        kernel.name,
        times.creations.len(),
        spread(&times.functions),
        spread(&times.creations),
    );
}

/// A device over the shader cache in `scratch`, whose memo holds `memo` bytes at most.
fn device_of(scratch: &Scratch, memo: u64) -> Device {
    let mut options = options(&scratch.0);
    options.shader_cache.memory_capacity = memo;
    Device::system_default_with(&options).unwrap()
}

/// The default bound of a device's memo.
const MEMO: u64 = crate::shader_cache::DEFAULT_MEMORY_CAPACITY;

/// The bench (the module docs).
#[test]
#[ignore = "a release-mode benchmark"]
fn compute_create_bench() {
    for kernel in [SMALL, CONV] {
        let scratch = Scratch::new("compute-create");
        let first = device_of(&scratch, MEMO);
        let library_first = library(&first, &kernel);
        let (before, mut times) = (first.work_counts(), Times::default());
        round(&kernel, &library_first, &first, &mut times);
        let work_cold = work(&before, &first.work_counts());
        report(&first, &kernel, "cold", &times, &work_cold);
        let (before, mut times) = (first.work_counts(), Times::default());
        for _ in 0..PASSES {
            round(&kernel, &library_first, &first, &mut times);
        }
        let work_warm = work(&before, &first.work_counts());
        report(&first, &kernel, "warm, same device", &times, &work_warm);
        drop((library_first, first));
        let mut times = Times::default();
        let mut works = Vec::new();
        for _ in 0..PASSES {
            let device = device_of(&scratch, MEMO);
            let library = library(&device, &kernel);
            let before = device.work_counts();
            round(&kernel, &library, &device, &mut times);
            works.push(work(&before, &device.work_counts()));
            if works.len() == PASSES {
                report(
                    &device,
                    &kernel,
                    "warm, new device",
                    &times,
                    &works.join(" | "),
                );
            }
        }
        // A memo of one byte keeps nothing past a round: each round finds each on disk.
        let device = device_of(&scratch, 1);
        let library = library(&device, &kernel);
        let (before, mut times) = (device.work_counts(), Times::default());
        for _ in 0..PASSES {
            round(&kernel, &library, &device, &mut times);
        }
        let work_disk = work(&before, &device.work_counts());
        report(&device, &kernel, "warm, memo of 1 byte", &times, &work_disk);
    }
}

/// The rounds [`compute_create_profile`] makes once the first translated.
const PROFILE_ROUNDS: usize = 2000;

/// For a profiler (ignored; run in release): one cold round of [`CONV`], then [`PROFILE_ROUNDS`]
/// rounds on the same device, whose memo of one byte keeps nothing past a round, so each finds
/// each translation on disk, printed as the bench prints.
#[test]
#[ignore = "a release-mode profile target"]
fn compute_create_profile() {
    let scratch = Scratch::new("compute-create-profile");
    let device = device_of(&scratch, 1);
    let library = library(&device, &CONV);
    round(&CONV, &library, &device, &mut Times::default());
    let (before, mut times) = (device.work_counts(), Times::default());
    for _ in 0..PROFILE_ROUNDS {
        round(&CONV, &library, &device, &mut times);
    }
    let work_disk = work(&before, &device.work_counts());
    report(
        &device,
        &CONV,
        "profiled, memo of 1 byte",
        &times,
        &work_disk,
    );
}
