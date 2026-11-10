//! Released render states made again from the device's kept variants (`pipeline/kept.rs`), on the
//! device the cfg selects. A program makes [`STATES`] states of distinct compiled pipelines, each
//! released as soon as it is made, then makes them all again in the same order, as a guest's
//! CoreAnimation makes and drops its render states: between a key's use and its next one a cold
//! boot's RenderBox made 51 other keys at the median and fewer than 256 at most. The second round
//! takes every state's variant back, its objects and its `VkPipeline`, so it makes no `VkPipeline`,
//! no shader module and no pipeline layout; the first makes one `VkPipeline` per state. With the
//! kept variants bounded at 16 instances, each is evicted before its key comes back.
//!
//! The states differ in their fragment function's constant (`fragment_tint`), so each is its own
//! compiled pipeline (the memo's `PipelineKey`, which is what the kept variants are keyed by).
//! Each creation makes its predicted `VkPipeline` on the creating thread
//! (`PipelineWaitPolicy::Exact`), so the counts do not depend on the device's pool.
//!
//! Printed (`KEPT-VARIANTS`): each round's creations' median and slowest time, and the process's
//! memory after each (`spares_bench::process_memory`).
//!
//! [`what_a_kept_instance_holds`] is a release benchmark, ignored: it measures what a kept
//! instance costs in memory and what a creation costs at the bound.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). The
//! functions are those of the owned fixture
//! `validation/fixtures/public/engine/unspecialized_stage_functions.ll` (`llvm-as`).

use super::blit_holds::{exact_waits_device, Scratch};
use super::spares_bench::process_memory;
use crate::pipeline::kept::KEPT;
use crate::{Device, Function, Library, PipelineOptions, WorkCounts};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{DataType, PixelFormat};
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::Path;
use std::time::{Duration, Instant};

/// The distinct states each round makes: past a guest's median reuse distance (51), well past
/// the 16 the kept variants once held.
const STATES: usize = 64;

/// The fixture's library on `device`.
fn library(device: &Device) -> Library {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/unspecialized_stage_functions.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    device.new_library_with_compile_library(
        metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
    )
}

/// The descriptor of state `n`: the constant-free vertex function and the fragment function with
/// `fragment_tint` at `n`, one `BGRA8Unorm` target.
fn descriptor(library: &Library, vertex: &Function, n: usize) -> RenderPipelineDescriptor {
    let tint = FunctionConstantValue::new(DataType::Float, (n as f32).to_le_bytes().to_vec());
    let fragment = library
        .new_function_with_name_constant_values(
            "unspecialized_fragment",
            &FunctionConstantValues::defaults().with_name("fragment_tint", tint),
        )
        .unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::BGRA8Unorm));
    descriptor
}

/// Make and release a state of each descriptor in turn: each creation's time.
fn round(device: &Device, descriptors: &[RenderPipelineDescriptor]) -> Vec<Duration> {
    let options = PipelineOptions::default();
    descriptors
        .iter()
        .map(|descriptor| {
            let start = Instant::now();
            let state = device
                .new_render_pipeline_state(descriptor, &options)
                .unwrap();
            let took = start.elapsed();
            drop(state);
            took
        })
        .collect()
}

/// The work a round did: `VkPipeline`s its variants made, of them those made at creation,
/// variants taken back, shader modules and pipeline layouts made.
fn work(before: &WorkCounts, after: &WorkCounts) -> [u64; 5] {
    [
        after.variant_pipelines_made - before.variant_pipelines_made,
        after.creation_render_pipelines_made - before.creation_render_pipelines_made,
        after.variants_taken_back - before.variants_taken_back,
        after.shader_modules_made - before.shader_modules_made,
        after.pipeline_layouts_made - before.pipeline_layouts_made,
    ]
}

/// The median and slowest of `times`, in microseconds.
fn spread(times: &[Duration]) -> String {
    let mut times = times.to_vec();
    times.sort();
    let us = |d: Duration| d.as_secs_f64() * 1e6;
    format!(
        "median {:.0} us, slowest {:.0} us",
        us(times[times.len() / 2]),
        us(times[times.len() - 1])
    )
}

/// The second round, every state released before it, makes nothing: each takes its variant
/// back. The first makes each state's predicted `VkPipeline`, so the second's zeros are not
/// vacuous.
#[test]
fn released_states_made_again_take_their_variants_back() {
    let scratch = Scratch::new("kept-variants");
    let device = exact_waits_device(&scratch.0);
    let library = library(&device);
    let vertex = library
        .new_function_with_name("unspecialized_vertex_free")
        .unwrap();
    let descriptors: Vec<_> = (0..STATES)
        .map(|n| descriptor(&library, &vertex, n))
        .collect();
    let start = device.work_counts();
    let first_times = round(&device, &descriptors);
    let first = device.work_counts();
    let first_memory = process_memory();
    let second_times = round(&device, &descriptors);
    let second = device.work_counts();
    let second_memory = process_memory();
    println!(
        "KEPT-VARIANTS kept={KEPT} states={STATES} first: {} work {:?} memory {first_memory:?}",
        spread(&first_times),
        work(&start, &first),
    );
    println!(
        "KEPT-VARIANTS kept={KEPT} states={STATES} second: {} work {:?} memory {second_memory:?}",
        spread(&second_times),
        work(&first, &second),
    );
    let [_, at_creation, taken_back, ..] = work(&start, &first);
    assert!(
        at_creation >= STATES as u64,
        "the first round made each state's VkPipeline at its creation: {at_creation}"
    );
    assert_eq!(taken_back, 0, "the first round took nothing back");
    assert_eq!(
        work(&first, &second),
        [0, 0, STATES as u64, 0, 0],
        "the second round: VkPipelines made, at creation, variants taken back, shader modules and \
         pipeline layouts made"
    );
}

/// What a kept instance costs in memory, and a creation at the bound: a release benchmark.
/// [`KEPT`] states of distinct pipelines, each released as made, fill the kept variants; as many
/// more then evict one instance each. The first run's growth per state is a kept instance's and
/// the memo's, the second's the memo's alone (each eviction frees an instance as one is kept), so
/// their difference is what a kept instance holds. Then the second run's states are made again,
/// each taking its variant back while [`KEPT`] are kept, the scan of a full deque included.
/// Printed (`KEPT-INSTANCE`). Ignored: run it alone, in release, under `--cfg m2v_gpu` (or
/// `m2v_cpu_icd`): `cargo nextest run --release -p metal2vulkan-runtime --lib --run-ignored all
/// what_a_kept_instance_holds --no-capture`.
#[test]
#[ignore = "a release benchmark: run with --run-ignored all --no-capture"]
fn what_a_kept_instance_holds() {
    let scratch = Scratch::new("kept-instance");
    let device = exact_waits_device(&scratch.0);
    let library = library(&device);
    let vertex = library
        .new_function_with_name("unspecialized_vertex_free")
        .unwrap();
    let descriptors: Vec<_> = (0..2 * KEPT)
        .map(|n| descriptor(&library, &vertex, n))
        .collect();
    let (filling, evicting) = descriptors.split_at(KEPT);
    let empty = process_memory();
    let start = device.work_counts();
    let fill_times = round(&device, filling);
    let full = process_memory();
    let filled = device.work_counts();
    let evict_times = round(&device, evicting);
    let evicted = process_memory();
    let after_evicting = device.work_counts();
    let again_times = round(&device, evicting);
    let again = device.work_counts();
    let per_state = |from: &std::collections::BTreeMap<String, u64>,
                     to: &std::collections::BTreeMap<String, u64>| {
        to.iter()
            .map(|(key, kib)| {
                let grown = *kib as f64 - from.get(key).copied().unwrap_or(0) as f64;
                format!("{key} {:.1} KiB", grown / KEPT as f64)
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!("KEPT-INSTANCE kept={KEPT} memory empty {empty:?} full {full:?} evicted {evicted:?}");
    println!(
        "KEPT-INSTANCE filling: per state {} ({}, work {:?})",
        per_state(&empty, &full),
        spread(&fill_times),
        work(&start, &filled)
    );
    println!(
        "KEPT-INSTANCE evicting: per state {} ({}, work {:?})",
        per_state(&full, &evicted),
        spread(&evict_times),
        work(&filled, &after_evicting)
    );
    println!(
        "KEPT-INSTANCE made again at the bound: {} (work {:?})",
        spread(&again_times),
        work(&after_evicting, &again)
    );
}
