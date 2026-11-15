//! A run of compute pipeline creations over one kernel specializes its AIR **once**, however many
//! siblings the run makes (`WorkCounts::stage_entries_specialized`, `stage_entries_shared`), as a
//! run of render creations over one vertex-and-fragment pair does.
//!
//! # Why this is a count
//!
//! A sibling pipeline of one kernel differs only in its stage-input descriptor, which is its
//! lowering inputs: its function compiles again, under its own compile key, so no memo of compiled
//! functions can spare it. What the siblings share is everything the compile does *before*
//! lowering -- decoding the library's AIR, cross-checking the container, binding the function
//! constants, resolving the linked functions, inlining and running SCCP -- which the kernel's
//! descriptor and its call-stack depth alone decide. The device holds that specialized entry
//! ([`StageEntries`](metal2vulkan::pipeline::StageEntries)), so the second sibling onwards reads
//! it rather than decoding and specializing the kernel again.
//!
//! The assertion is a count of specializations, never a time: it is a fact of the structure, so it
//! holds on any host under any load, needs no quiet host and depends on nothing that ran before it
//! in the process (the warm-up's own creation is subtracted). Both halves are needed. The count of
//! specializations holds the sharing from above; the count of **shader modules** holds it from
//! below, because sharing an entry would also read zero if the run never compiled at all -- a
//! sibling found whole in the pipeline memo asks for no entry, and a test that only counted
//! specializations would pass on a run that did nothing.
//!
//! The device takes a shader cache directory of its own, empty, so that no run of this test reads
//! a pipeline an earlier run stored: a sibling found whole on disk compiles nothing and asks for
//! no entry, which is the warmth this count must not depend on.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernel is the owned fixture
//! `validation/fixtures/public/engine/kernel_stage_in.ll`'s `stage_in_fetch_wide` (`llvm-as`).

use super::blit_holds::{device, Scratch};
use crate::PipelineOptions;
use metal2vulkan::metal::{AttributeFormat, StepFunction};
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, StageInputAttributeDescriptor, StageInputDescriptor,
    StageInputLayoutDescriptor, StageSource,
};

/// The siblings the run makes after the warm-up's one.
const RUN: u64 = 16;

/// A compute pipeline state of the fixture's `[[stage_in]]` kernel, its layout's stride `step`
/// bytes wider than the last: every creation is its own descriptor, and so its own compile, as
/// the attribute sweeps that measured this are.
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

/// The entry asks one compute creation of a kernel with a stage-input descriptor makes: one to work
/// out which `[[stage_in]]` attributes the kernel reads, so the compile key names those and nothing
/// else (`Device::kernel_inputs_read`), and one for the compile itself
/// (`compile_function_from_entries`). Both read the device's entry and neither specializes it again.
const ASKS: u64 = 2;

/// `RUN` sibling compute pipelines of one kernel compile `RUN` modules of their own and specialize
/// the kernel's AIR not once each but not at all: the entry the warm-up's creation specialized is
/// the one every one of them reads.
#[test]
fn a_run_of_compute_creations_specializes_one_kernel_once() {
    let scratch = Scratch::new("stage-entries-shared");
    let device = device(&scratch.0);
    let ll = metal2vulkan_devtools::engine_dir().join("kernel_stage_in.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = device.new_library_with_compile_library(
        metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
    );
    // The warm-up's creation is the one that specializes the kernel.
    create(&device, &library, 0);
    let before = device.work_counts();
    assert!(
        before.stage_entries_specialized > 0,
        "the first creation must specialize the kernel at all, not {} times: a creation that \
         specializes nothing makes the run's zero a vacuous zero",
        before.stage_entries_specialized
    );
    for step in 0..RUN {
        create(&device, &library, 1 + step as u32);
    }
    let after = device.work_counts();
    let modules = after.shader_modules_made - before.shader_modules_made;
    assert_eq!(
        modules, RUN,
        "every sibling must compile a module of its own, else the run shares an entry it never \
         asked for: {modules} modules made over {RUN} creations"
    );
    assert_eq!(
        after.stage_entries_specialized - before.stage_entries_specialized,
        0,
        "AIR specializations {RUN} sibling creations of one kernel did, each of them compiling"
    );
    assert_eq!(
        after.stage_entries_shared - before.stage_entries_shared,
        ASKS * RUN,
        "entry reads {RUN} siblings did, {ASKS} each and specializing none of them"
    );
}
