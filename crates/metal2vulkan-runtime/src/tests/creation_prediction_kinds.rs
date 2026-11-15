//! The two predictions a state's creation makes a `VkPipeline` for are counted apart: a render
//! state's predicted key (`pipeline/predicted.rs`) moves
//! `WorkCounts::creation_render_pipelines_made` alone, and a compute state's predicted region
//! (`pipeline/dispatch_making.rs`) moves `creation_dispatch_pipelines_made` alone. The whole
//! count is their sum.
//!
//! # Why this is a count
//!
//! The two predictions guess at very different things -- a render key the first draw will bind,
//! and a threadgroup the first dispatch will take -- and they are right at wildly different
//! rates. `creation_pipelines_made` and `creation_pipelines_drawn` cover both, so neither one's
//! hit rate can be read off them: a reading of 100 made and 60 drawn is as consistent with one
//! prediction always right and the other always wrong as with both being right three fifths of
//! the time. Splitting them makes each rate readable, and this holds the split to the kind of
//! state it belongs to.
//!
//! Both legs assert their own count moved before asserting the other's did not: a creation that
//! predicts nothing at all would read zero for both kinds and make the "the other kind did not
//! move" assertion vacuous.
//!
//! # Which compute creations predict at all
//!
//! A compute state predicts a region only where its descriptor states
//! `requiredThreadsPerThreadgroup`, which Metal then makes the threadgroup of every dispatch of
//! the state (`pipeline/dispatch_making.rs`). So the compute leg states one, and a third leg
//! creates the same kernel's state **without** one and holds the kind's count still: a guess at
//! the threadgroup where the request names none was taken by 0 dispatches of 141 made over the
//! whole conformance suite, so it is not made.
//!
//! Every reading here is a delta across one creation on this test's own device, never a
//! device-wide total: a total would be moved by anything else the process did and would pass with
//! the behaviour put back.
//!
//! The assertion is a count, never a time: it is a fact of the structure, so it holds on any host
//! under any load. The device takes a shader cache directory of its own, empty, so no run reads a
//! pipeline an earlier run stored.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixtures
//! `validation/fixtures/public/engine/ubo_bind_offsets.ll` and
//! `validation/fixtures/public/examples/runtime_examples.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch};
use super::descriptor_key_once::descriptor;
use crate::{PipelineOptions, WorkCounts};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};

/// The two predictions' makings, and the whole count: render made, dispatch made, all made.
fn made(counts: &WorkCounts) -> [u64; 3] {
    [
        counts.creation_render_pipelines_made,
        counts.creation_dispatch_pipelines_made,
        counts.creation_pipelines_made,
    ]
}

/// A render creation's predicted pipeline counts as the render prediction's and nothing else; a
/// compute creation's as the dispatch prediction's. The whole count is the two together.
#[test]
fn the_render_and_dispatch_predictions_are_counted_apart() {
    let scratch = Scratch::new("creation-prediction-kinds");
    let device = device(&scratch.0);
    let options = PipelineOptions::default();

    let before = made(&device.work_counts());
    let _render = device
        .new_render_pipeline_state(&descriptor(&device), &options)
        .unwrap();
    let after_render = made(&device.work_counts());
    let moved = |from: [u64; 3], to: [u64; 3]| [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
    let render = moved(before, after_render);
    assert!(
        render[0] > 0,
        "a render creation must make its predicted pipeline at all, not {} times: a creation that \
         predicts nothing makes the dispatch count's zero below a vacuous zero",
        render[0]
    );
    assert_eq!(
        render[1], 0,
        "a render creation made {} pipeline(s) counted as a compute state's predicted region",
        render[1]
    );
    assert_eq!(
        render[2], render[0],
        "the whole count moved {} where the render prediction moved {}",
        render[2], render[0]
    );

    let ll = metal2vulkan_devtools::fixtures_dir().join("examples/runtime_examples.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let function = device
        .new_library_with_compile_library(
            metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
        )
        .new_function_with_name("scale_and_offset")
        .unwrap();
    // The threadgroup every dispatch of the state must take, so the region of its first dispatch
    // is part of the request and the creation makes that region's pipeline.
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.required_threads_per_threadgroup = Some([32, 1, 1]);
    let _compute = device
        .new_compute_pipeline_state(&descriptor, &options)
        .unwrap();
    let after_compute = made(&device.work_counts());
    let compute = moved(after_render, after_compute);
    assert!(
        compute[1] > 0,
        "a compute creation whose descriptor requires a threadgroup must make its predicted \
         pipeline at all, not {} times",
        compute[1]
    );
    assert_eq!(
        compute[0], 0,
        "a compute creation made {} pipeline(s) counted as a render state's predicted key",
        compute[0]
    );
    assert_eq!(
        compute[2], compute[1],
        "the whole count moved {} where the dispatch prediction moved {}",
        compute[2], compute[1]
    );

    // And the same kernel with no required threadgroup: nothing in the request says what the
    // dispatch will ask for, so nothing is predicted. Read as the delta across this one creation.
    let _unrequired = device
        .new_compute_pipeline_state_with_function(&function, &options)
        .unwrap();
    let after_unrequired = made(&device.work_counts());
    let unrequired = moved(after_compute, after_unrequired);
    assert_eq!(
        unrequired[1], 0,
        "a compute creation whose descriptor requires no threadgroup made {} predicted \
         pipeline(s) for a threadgroup the request never named; such a guess was taken by 0 \
         dispatches of 141 made over the whole conformance suite",
        unrequired[1]
    );
    assert_eq!(
        unrequired[2], 0,
        "the whole count moved {} over a creation that predicts nothing",
        unrequired[2]
    );
}
