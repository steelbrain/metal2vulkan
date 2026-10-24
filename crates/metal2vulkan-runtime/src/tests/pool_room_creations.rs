//! A run of pipeline creations asks the memory pools for room once, not once a creation
//! (`memory.rs`'s `Pool::room`, `tests::pool_room`).
//!
//! # What this adds to the pure count
//!
//! `tests::pool_room` holds the pool itself to one derivation over pool shapes it builds. This
//! holds the runtime to it on a device: every creation call that may wait keeps room in the pools
//! the encodings and commits place their own resources in (`memory::keep_room`), so a run of
//! creations is where the per-call walk was paid, and a compute pipeline state places nothing in
//! those pools -- the blocks it needs are the ones the first creation made.
//!
//! It is a **count**, never a time, for the same reason the pure one is: six streams share this
//! host and no timing window is judgeable, where the derivations a run costs are a fact of the
//! structure. The warm-up's end holds the count from below -- the pools must be asked at all, so
//! `keep_room` dropped from the creation path turns the run's zero into a vacuous zero and fails
//! here -- and the run's end holds it from above.

use crate::PipelineOptions;
use metal2vulkan::metal::{AttributeFormat, StepFunction};
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, StageInputAttributeDescriptor, StageInputDescriptor,
    StageInputLayoutDescriptor, StageSource,
};
use std::path::Path;

/// The creations the warm-up makes: the first makes the blocks the pools lack, the second finds
/// them and derives each pool's longest free range, and the third reads what the second held.
const WARM: u32 = 3;

/// The creations the run makes once the pools hold their answer.
const RUN: u32 = 20;

/// A compute pipeline state of the fixture's `[[stage_in]]` kernel, its layout's stride `step`
/// bytes wider than the last: every creation of a run is its own descriptor, as the attribute
/// sweeps that measured this are.
fn create(device: &crate::Device, library: &crate::Library, step: u32) {
    let function = library
        .new_function_with_name("stage_in_fetch_wide")
        .unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    let mut stage_input = StageInputDescriptor::new();
    for (index, format, offset) in [
        (0, AttributeFormat::Float4, 0),
        (1, AttributeFormat::UInt4, 16),
        (2, AttributeFormat::Int, 32),
        (3, AttributeFormat::Half2, 36),
        (4, AttributeFormat::Int4, 48),
    ] {
        stage_input
            .attributes
            .insert(index, StageInputAttributeDescriptor::new(format, offset, 0));
    }
    let mut layout = StageInputLayoutDescriptor::new(Some(64 + step));
    layout.step_function = StepFunction::ThreadPositionInGridX;
    stage_input.layouts.insert(0, layout);
    descriptor.stage_input = Some(stage_input);
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
}

/// A run of compute pipeline creations leaves the pools' blocks and free ranges as they are, and
/// derives their longest free ranges not once a creation but not at all: the answer the warm-up's
/// creations derived is the answer every one of them reads.
#[test]
fn a_run_of_creations_derives_the_pools_room_no_further() {
    let device = crate::Device::system_default().unwrap();
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/kernel_stage_in.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = device.new_library_with_compile_library(
        metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
    );
    let room_work = || crate::device::lock(&device.shared().memory).room_work();
    for step in 0..WARM {
        create(&device, &library, step);
    }
    let (walked, visited, live, ranges) = room_work();
    assert!(
        walked > 0 && visited > 0,
        "the creations must ask the pools for room at all: \
         {walked} block slots walked and {visited} free ranges visited"
    );
    for step in 0..RUN {
        create(&device, &library, WARM + step);
    }
    assert_eq!(
        room_work(),
        (walked, visited, live, ranges),
        "{RUN} creations that place nothing in the pools must leave their {live} block slots and \
         {ranges} free ranges as they are and derive their room no further, which is {walked} \
         slots walked and {visited} ranges visited however many creations read it"
    );
}
