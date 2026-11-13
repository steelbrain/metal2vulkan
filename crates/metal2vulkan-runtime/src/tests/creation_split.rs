//! Which part of a pipeline state's creation is the driver's and which is ours
//! (`WorkCounts::render_pipeline_create_nanos`, `compute_pipeline_create_nanos`,
//! `shader_cache_read_nanos`, and `ShaderCacheStatistics`' hits and misses by kind).
//!
//! # Why these are read out here
//!
//! A probe outside the runtime times a creation whole, so it cannot tell the driver's compile from
//! the runtime's own work, and nothing in a `WorkCounts` said how long the driver was in
//! `vkCreateGraphicsPipelines` or how long the shader cache's directory was under a creation. This
//! test is the proof that the figures reach a `WorkCounts` through the device's real door
//! (`device/compiles.rs`) and are attributed to the right field. It asserts no magnitude: what
//! they are on a given device is the device's business. That the two driver calls are counted
//! apart is held host-independently against a fake driver in
//! `device::compiles::tests::a_making_s_time_in_the_driver_is_counted_by_the_call_it_made`.
//!
//! # The attribution it holds
//!
//! - **A counted making moved its figure.** A compute state's creation makes its region's
//!   pipeline at the dispatch, not at the creation, so on a device that predicts no region the
//!   sweep here makes none, and on one that does it makes some; either way, where a making is
//!   counted the call's figure must have moved. That direction is what this leg holds, and on a
//!   device that predicts no region it is the render creation below that carries it.
//!   **The converse is not ours to assert.** The figures are the time in *every* making the
//!   runtime asked the device for through the one door (`device/compiles.rs`), and makings that
//!   no `WorkCounts` field counts reach that door too: the device's own answer about a dynamic
//!   binding of several variables, asked inside the first creation whose plan raises the
//!   question and paid for whichever way the driver answers (`pipeline/objects.rs`'s
//!   `takes_multi_variable_dynamic`, a compile this host's driver refuses); priming's kernels
//!   (`device/prime.rs`); and the runtime's own blit, depth, stencil, resolve and plane kernels.
//!   A moved figure therefore names no making, and a creation that made no `VkPipeline` can
//!   still have spent driver time. Counting those makings is not this test's business; until
//!   something counts them, the figure is a cost, not a census.
//! - **A render creation's time is the graphics call's.** It moves the render figure and leaves
//!   the compute figure where it was.
//! - **Every creation asks the directory**, so the reads move in both legs. A creation of this
//!   shape hands every entry to the cache thread, so it writes none itself and the two figures of
//!   its own writes stay at zero together.
//! - **The hits and misses by kind add up to the whole hits and misses**, and a cold compute sweep
//!   looks up no render pipeline's entry and gives the directory its translated functions alone:
//!   it links each compute pipeline from its function and keeps none (`pipeline/sources.rs`).
//!   Without the split, a creation's one answered lookup and one unanswered one could not be told
//!   apart by kind.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). The kernel
//! and the function pair are the owned fixtures
//! `validation/fixtures/public/engine/kernel_stage_in.ll` and `ubo_bind_offsets.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch};
use super::descriptor_key_once::descriptor;
use crate::{Device, PipelineOptions, ShaderCacheStatistics};
use metal2vulkan::metal::{AttributeFormat, StepFunction};
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, StageInputAttributeDescriptor, StageInputDescriptor,
    StageInputLayoutDescriptor, StageSource,
};
use std::path::Path;

/// The compute states the sweep makes.
const SWEEP: u32 = 4;

/// The attributes the fixture's kernel declares and reads.
const READ: [(u32, AttributeFormat, u32); 5] = [
    (0, AttributeFormat::Float4, 0),
    (1, AttributeFormat::UInt4, 16),
    (2, AttributeFormat::Int, 32),
    (3, AttributeFormat::Half2, 36),
    (4, AttributeFormat::Int4, 48),
];

/// A compute state of the fixture's `[[stage_in]]` kernel, its layout's stride `step` bytes wider
/// than the last, so every row of the sweep is its own descriptor and its own pipeline.
fn compute(device: &Device, library: &crate::Library, step: u32) {
    let function = library
        .new_function_with_name("stage_in_fetch_wide")
        .unwrap();
    let mut made =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    let mut stage_input = StageInputDescriptor::new();
    for (index, format, offset) in READ {
        stage_input
            .attributes
            .insert(index, StageInputAttributeDescriptor::new(format, offset, 0));
    }
    let mut layout = StageInputLayoutDescriptor::new(Some(64 + step));
    layout.step_function = StepFunction::ThreadPositionInGridX;
    stage_input.layouts.insert(0, layout);
    made.stage_input = Some(stage_input);
    device
        .new_compute_pipeline_state(&made, &PipelineOptions::default())
        .unwrap();
}

/// The hits and misses by kind add up to the whole counts.
fn adds_up(made: &ShaderCacheStatistics) {
    assert_eq!(
        made.function_hits + made.render_pipeline_hits,
        made.hits,
        "the hits by kind are every hit: {made:?}"
    );
    assert_eq!(
        made.function_misses + made.render_pipeline_misses,
        made.misses,
        "the misses by kind are every miss: {made:?}"
    );
}

/// The time the driver spends making a state's pipelines reaches the state's device's
/// `WorkCounts`, by the call it made, and the shader cache directory's reads reach it beside them.
#[test]
fn a_creation_s_driver_time_and_directory_reads_reach_the_work_counts() {
    let scratch = Scratch::new("creation-split");
    let device = device(&scratch.0);
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/kernel_stage_in.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = device.new_library_with_compile_library(
        metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
    );
    let before = device.work_counts();
    for row in 0..SWEEP {
        compute(&device, &library, 16 * (row + 1));
    }
    let swept = device.work_counts();
    // A compute state's creation makes its region's pipeline at the dispatch unless the device
    // predicts one, so a sweep here may make none. Where it made one, the compute figure must
    // have moved; a figure that moved on its own names no making (the module's note).
    let made_a_pipeline = swept.variant_pipelines_made > before.variant_pipelines_made
        || swept.creation_pipelines_made > before.creation_pipelines_made;
    assert!(
        !made_a_pipeline
            || swept.compute_pipeline_create_nanos > before.compute_pipeline_create_nanos,
        "the sweep made a `VkPipeline` and the compute figure stood still: {} to {} ns, \
         {} to {} variants and {} to {} at creation",
        before.compute_pipeline_create_nanos,
        swept.compute_pipeline_create_nanos,
        before.variant_pipelines_made,
        swept.variant_pipelines_made,
        before.creation_pipelines_made,
        swept.creation_pipelines_made
    );
    assert_eq!(
        swept.render_pipeline_create_nanos, before.render_pipeline_create_nanos,
        "a compute sweep makes no graphics pipeline, so the graphics figure stands still"
    );
    assert!(
        swept.shader_cache_read_nanos > before.shader_cache_read_nanos,
        "a compute sweep asked the directory nothing: {} to {} ns",
        before.shader_cache_read_nanos,
        swept.shader_cache_read_nanos
    );
    let cold = device.shader_cache_statistics();
    adds_up(&cold);
    assert!(
        cold.function_misses > 0,
        "a cold compute sweep must translate its kernel: {cold:?}"
    );
    assert_eq!(
        (cold.render_pipeline_hits, cold.render_pipeline_misses),
        (0, 0),
        "a compute sweep looks up no render pipeline: {cold:?}"
    );
    assert_eq!(
        cold.writes, cold.function_misses,
        "a compute sweep gives the directory each function it translated and no pipeline, which \
         it links from the function: {cold:?}"
    );
    // A render state of the other fixture: now the graphics call is made, and it is the graphics
    // figure that moves.
    let render = descriptor(&device);
    let state = device
        .new_render_pipeline_state(&render, &PipelineOptions::default())
        .unwrap();
    let after = device.work_counts();
    assert!(
        after.variant_pipelines_made > swept.variant_pipelines_made
            || after.creation_pipelines_made > swept.creation_pipelines_made,
        "the render creation must make a `VkPipeline` at all: {after:?}"
    );
    assert!(
        after.render_pipeline_create_nanos > swept.render_pipeline_create_nanos,
        "a render creation's `vkCreateGraphicsPipelines` read no time: {} to {} ns",
        swept.render_pipeline_create_nanos,
        after.render_pipeline_create_nanos
    );
    assert_eq!(
        after.compute_pipeline_create_nanos, swept.compute_pipeline_create_nanos,
        "a render creation makes no compute pipeline, so the compute figure stands still"
    );
    assert!(
        after.shader_cache_read_nanos > swept.shader_cache_read_nanos,
        "a render creation asked the directory nothing"
    );
    assert_eq!(
        (
            after.shader_cache_write_nanos > 0,
            after.shader_cache_writes_on_thread > 0
        ),
        (false, false),
        "no creation of this shape writes its own cache entry: the queue takes them all \
         (`tests::shader_cache_off_thread`), and {} ns of writes with {} such writes would \
         disagree",
        after.shader_cache_write_nanos,
        after.shader_cache_writes_on_thread
    );
    let warm = device.shader_cache_statistics();
    adds_up(&warm);
    assert!(
        warm.function_misses > cold.function_misses,
        "the render creation must translate its pair: {warm:?}"
    );
    println!(
        "creation split: {SWEEP} compute states and one render state; driver {} us graphics and \
         {} us compute; directory {} us read, {} us written by a creating thread ({} entries), \
         {} us on the cache thread; by kind, functions {}/{} hit/missed, render pipelines {}/{}",
        after.render_pipeline_create_nanos / 1_000,
        after.compute_pipeline_create_nanos / 1_000,
        after.shader_cache_read_nanos / 1_000,
        after.shader_cache_write_nanos / 1_000,
        after.shader_cache_writes_on_thread,
        after.shader_cache_off_thread_nanos / 1_000,
        warm.function_hits,
        warm.function_misses,
        warm.render_pipeline_hits,
        warm.render_pipeline_misses,
    );
    drop(state);
}
