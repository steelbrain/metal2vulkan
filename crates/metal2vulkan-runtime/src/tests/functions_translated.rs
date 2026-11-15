//! What a run of pipeline states costs in **translations**: the count between the AIR decode and
//! the shader module (`WorkCounts::functions_translated`, `functions_shared`).
//!
//! # Why this count had to exist
//!
//! A sweep of 36 compute states over one kernel, differing only in stage-input attributes the
//! kernel never reads, translated the kernel 36 times and made **one** shader module of the result,
//! because every translation came out byte-identical. Neither counter around it saw that:
//! `stage_entries_specialized` reads 1 and `stage_entries_shared` 35, because the decode and
//! specialization *before* lowering never depended on the unread attributes; `shader_modules_made`
//! reads 1 and `shader_modules_shared` 35, because the SPIR-V *after* lowering was the same words.
//! The 35 translations in between moved nothing a bound could hold, so collapsing them to one moved
//! no bound either -- and a tree that re-translated them would pass every bound the suite holds.
//!
//! # Why these are counts
//!
//! A translation is a memo miss under a compile key, so the number of them a run costs is a fact of
//! the keys and not of the host: it holds under any load, needs no quiet machine, and the two halves
//! add up to every ask the run made. Both halves are asserted, in both directions, because each
//! alone passes on a run that did nothing: a run that compiled no pipeline at all shares nothing
//! and translates nothing.
//!
//! The device takes an empty shader cache directory of its own, so no run of this test reads a
//! function an earlier run stored: a function found on disk is counted as shared, correctly, and
//! would make the first row's translation vanish.
//!
//! # The two halves this holds
//!
//! - [`a_sweep_over_unread_attributes_translates_once`]: the narrowing's own shape. The kernel reads
//!   attributes 0 to 4; each row adds a sixth the kernel never reads, with its own format and
//!   offset. One translation, the rest shared.
//! - [`a_sweep_over_a_read_layout_translates_each_row`]: the control, and the sharper of the two. The
//!   rows move the **read** layout's stride, which every fetch of the kernel depends on, so each row
//!   is its own compile key and its own translation. A counter that read 1 here would be worse than
//!   no counter: it would say the runtime shared work it did again.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). The kernel is
//! the owned fixture `validation/fixtures/public/engine/kernel_stage_in.ll`'s `stage_in_fetch_wide`
//! (`llvm-as`).

use super::blit_holds::{device, Scratch};
use crate::PipelineOptions;
use metal2vulkan::metal::{AttributeFormat, StepFunction};
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, StageInputAttributeDescriptor, StageInputDescriptor,
    StageInputLayoutDescriptor, StageSource,
};

/// The rows each sweep makes after the warm-up's one.
const RUN: u64 = 16;

/// The attributes the fixture's kernel declares and reads, and the stride of the layout they read.
const READ: [(u32, AttributeFormat, u32); 5] = [
    (0, AttributeFormat::Float4, 0),
    (1, AttributeFormat::UInt4, 16),
    (2, AttributeFormat::Int, 32),
    (3, AttributeFormat::Half2, 36),
    (4, AttributeFormat::Int4, 48),
];
const READ_STRIDE: u32 = 64;

/// A buffer index no read attribute names, for an unread attribute's own layout.
const SPARE: u32 = 4;

/// The library of the fixture's kernel, on a device with an empty cache of its own.
fn opened(scratch: &Scratch) -> (crate::Device, crate::Library) {
    let device = device(&scratch.0);
    let ll = metal2vulkan_devtools::engine_dir().join("kernel_stage_in.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = device.new_library_with_compile_library(
        metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
    );
    (device, library)
}

/// A compute state of the kernel: the read attributes on a layout of stride `stride`, and where
/// `unread` is given, a sixth attribute of that format at that offset on its own spare layout --
/// which the kernel's `[[stage_in]]` never reads.
fn create(
    device: &crate::Device,
    library: &crate::Library,
    stride: u32,
    unread: Option<(AttributeFormat, u32)>,
) {
    let function = library
        .new_function_with_name("stage_in_fetch_wide")
        .unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    let mut stage_input = StageInputDescriptor::new();
    for (index, format, offset) in READ {
        stage_input
            .attributes
            .insert(index, StageInputAttributeDescriptor::new(format, offset, 0));
    }
    let mut layout = StageInputLayoutDescriptor::new(Some(stride));
    layout.step_function = StepFunction::ThreadPositionInGridX;
    stage_input.layouts.insert(0, layout);
    if let Some((format, offset)) = unread {
        stage_input
            .attributes
            .insert(5, StageInputAttributeDescriptor::new(format, offset, SPARE));
        let mut spare = StageInputLayoutDescriptor::new(Some(64));
        spare.step_function = StepFunction::ThreadPositionInGridX;
        stage_input.layouts.insert(SPARE, spare);
    }
    descriptor.stage_input = Some(stage_input);
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
}

/// The formats an unread attribute takes, cycled over the sweep's rows.
const FORMATS: [AttributeFormat; 8] = [
    AttributeFormat::UInt,
    AttributeFormat::Float4,
    AttributeFormat::UChar4Normalized,
    AttributeFormat::Half2,
    AttributeFormat::Int4,
    AttributeFormat::Short2,
    AttributeFormat::UInt1010102Normalized,
    AttributeFormat::UShort4,
];

/// `RUN` states over one kernel whose descriptors differ only in an attribute the kernel never
/// reads translate the kernel **not once each but not at all**: the warm-up's translation is the one
/// every row of the sweep shares.
#[test]
fn a_sweep_over_unread_attributes_translates_once() {
    let scratch = Scratch::new("functions-translated-unread");
    let (device, library) = opened(&scratch);
    // The warm-up's creation is the one that translates the kernel.
    create(&device, &library, READ_STRIDE, None);
    let before = device.work_counts();
    assert!(
        before.functions_translated > 0,
        "the first creation must translate the kernel at all, not {} times: a creation that \
         translates nothing makes the sweep's zero a vacuous zero",
        before.functions_translated
    );
    for row in 0..RUN {
        let format = FORMATS[(row as usize) % FORMATS.len()];
        let offset = 4 * (row as u32 % 5);
        create(&device, &library, READ_STRIDE, Some((format, offset)));
    }
    let after = device.work_counts();
    // The two counters around this one read the sharing **whether or not the sweep translates
    // again**, which is the whole reason this count exists: the decode before lowering never
    // depended on the unread attribute, and the SPIR-V after it comes out the same words. Asserted
    // first, so a run against a tree that re-translates says so here -- these stay at zero -- and
    // then fails on the count that can see it.
    assert_eq!(
        after.stage_entries_specialized - before.stage_entries_specialized,
        0,
        "the AIR decode before lowering is shared over the sweep either way"
    );
    assert_eq!(
        after.shader_modules_made - before.shader_modules_made,
        0,
        "the module made from the SPIR-V after lowering is shared over the sweep either way"
    );
    assert_eq!(
        after.functions_translated - before.functions_translated,
        0,
        "translations {RUN} states over one kernel did, each of them stating an attribute the \
         kernel never reads: the attribute must not reach the compile key"
    );
    assert_eq!(
        after.functions_shared - before.functions_shared,
        RUN,
        "each of the {RUN} rows must ask for the kernel's function and be answered by the memo, \
         once: a row that asks for nothing shares nothing and the zero above is vacuous"
    );
}

/// `RUN` states over one kernel whose descriptors move the layout **the kernel reads** translate it
/// each: the stride reaches every fetch, so each row is its own compile key and its own
/// translation.
///
/// This is the control the other half needs. A narrowing that dropped a field the kernel does read
/// would read 1 here and look like a win.
#[test]
fn a_sweep_over_a_read_layout_translates_each_row() {
    let scratch = Scratch::new("functions-translated-read");
    let (device, library) = opened(&scratch);
    create(&device, &library, READ_STRIDE, None);
    let before = device.work_counts();
    for row in 0..RUN {
        create(&device, &library, READ_STRIDE + 4 * (1 + row as u32), None);
    }
    let after = device.work_counts();
    assert_eq!(
        after.functions_translated - before.functions_translated,
        RUN,
        "each of the {RUN} rows moves the stride every fetch of the kernel reads, so each must \
         translate its own function: a count that read fewer would say work was shared that was \
         done again"
    );
    assert_eq!(
        after.functions_shared - before.functions_shared,
        0,
        "no row of this sweep may be answered by the memo: its key is its own"
    );
}
