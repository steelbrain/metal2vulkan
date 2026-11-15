//! The programs a buffer whose records the GPU may write keeps of the pipeline states the
//! program releases (`released_programs.rs`) stay bounded: at most four of an ID, however many
//! states take the ID and give it back; and the device's freed IDs keep one program each, which a
//! buffer first written after keeps, until a state takes the ID.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The function is the owned fixture
//! `validation/fixtures/public/examples/runtime_examples.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch};
use crate::{IndirectCommandBufferDescriptor, PipelineOptions, ResourceId, ResourceOptions};
use metal2vulkan::metal::IndirectCommandType;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::path::Path;

/// The states made and released on one ID.
const CYCLES: usize = 1000;

/// A buffer the GPU may write keeps the programs of the states made with
/// `supportIndirectCommandBuffers` the program releases, at most four of an ID: after 1000 states
/// took ID 1 and gave it back, it keeps four, the last released, and the buffer made after it
/// that the GPU never writes keeps none.
#[test]
fn a_buffer_keeps_at_most_four_released_programs_of_an_id() {
    let scratch = Scratch::new("released-programs");
    let device = device(&scratch.0);
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/examples/runtime_examples.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("scale_and_offset")
        .unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = true;
    let mut commands = IndirectCommandBufferDescriptor::new();
    commands.command_types = IndirectCommandType::CONCURRENT_DISPATCH_THREADS;
    commands.max_kernel_buffer_bind_count = 3;
    let make = || {
        device
            .new_indirect_command_buffer(&commands, 1, ResourceOptions::shared())
            .unwrap()
    };
    let written = make();
    // The hold of a command buffer that may write its records, completed at once.
    drop(written.gpu_write());
    let unwritten = make();
    for cycle in 0..CYCLES {
        let state = device
            .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
        assert_eq!(state.gpu_resource_id(), ResourceId(1), "cycle {cycle}");
        drop(state);
        assert!(written.kept_count() <= 4, "cycle {cycle}");
    }
    assert_eq!(written.kept_count(), 4);
    assert_eq!(unwritten.kept_count(), 0);
    // The freed ID keeps its last holder's program alone, which a buffer the GPU first writes now
    // keeps; once a state took the ID, none.
    let later = make();
    drop(later.gpu_write());
    assert_eq!(later.kept_count(), 1, "the freed ID's program");
    let taker = device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    assert_eq!(taker.gpu_resource_id(), ResourceId(1));
    let after = make();
    drop(after.gpu_write());
    assert_eq!(after.kept_count(), 0, "the ID taken");
}
