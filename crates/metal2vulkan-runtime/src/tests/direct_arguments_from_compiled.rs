//! A compute creation with `supportIndirectCommandBuffers` answers Metal's direct-argument
//! refusal from the kernel its own compile produced, not from the made descriptor's AIR
//! (`pipeline/mod.rs`).
//!
//! # What this guards
//!
//! Metal answers nil for a compute state that supports indirect command buffers whose kernel uses
//! a direct texture or sampler, so every such creation asks
//! `ComputePipelineDescriptor::uses_direct_arguments`. Asked of the **made** descriptor, whose
//! kernel is a `Function` stage, that decodes the kernel's AIR -- immediately after the creation's
//! own compile decoded it and kept the answer. Asked of the **compiled** kernel the compile
//! returned, it decodes nothing and reads what the compile kept. The render creation asks its
//! equivalent before its compile and keeps the answer in the shader cache; the compute creation
//! asked after it and kept nothing.
//!
//! The check is a **count** of the creations that fell back to the AIR, which an ordinary creation
//! must not do: before the fix every such creation took that path, and now none does. A count is
//! deterministic on any host under any load. It is a process counter read as a difference across
//! the creation this test makes, and nextest gives each test its own process, so no other test's
//! creations are in it.
//!
//! What it does not prove by itself is that no decode happened: that rests on the compile layer's
//! contract that a `StageSource::Compiled` stage answers from what its compile kept and decodes
//! nothing. What the count proves is which of the two paths the creation took.

use super::blit_holds::{device, Scratch};
use crate::pipeline::direct_arguments_from_air;
use crate::PipelineOptions;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::path::Path;

/// A compute state of `runtime_examples`' `scale_and_offset` kernel with
/// `supportIndirectCommandBuffers`, which is what makes the creation ask: it is made, and the
/// creation answers from the compiled kernel, adding nothing to the AIR count.
#[test]
fn a_creation_answers_from_its_compiled_kernel() {
    let scratch = Scratch::new("direct-arguments-from-compiled");
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
    let before = direct_arguments_from_air();
    // Two states, so a creation that asked the AIR would add two: the kernel is the same, and the
    // answer does not depend on which object asks it.
    for _ in 0..2 {
        device
            .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
            .unwrap();
    }
    assert_eq!(
        direct_arguments_from_air() - before,
        0,
        "a creation whose compile kept the answer read it from the compiled kernel, and decoded \
         the kernel's AIR again for none of it"
    );
}
