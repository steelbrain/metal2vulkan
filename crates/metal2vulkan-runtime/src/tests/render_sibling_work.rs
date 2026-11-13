//! A run of render pipeline creations over one vertex-and-fragment pair, differing only in
//! fixed-function state, makes a `VkPipeline` each and **nothing else**: no shader module, no
//! pipeline layout and no AIR specialization of its own
//! (`WorkCounts::variant_pipelines_made`, `shader_modules_made`, `pipeline_layouts_made`,
//! `stage_entries_specialized`, `functions_translated`).
//!
//! It is the render half of the translation counts: both stages of every sibling ask the device's
//! memo for their function and are answered by it, so one function pair costs one translation a
//! stage however many states the sweep makes (`tests::functions_translated` holds the compute form,
//! where the ask comes from the kernel rather than from a render variant's two stages).
//!
//! # Why this is a count
//!
//! A sweep of states over one function pair -- a blend sweep, a vertex-format sweep -- is what a
//! render pipeline state costs when nothing about the shaders changes. Each sibling is its own
//! descriptor, so each needs a pipeline of its own; what it must not need is the work that belongs
//! to the shaders. The stages compile under keys the sweep does not move, so the device's memo of
//! compiled functions answers them; their SPIR-V is word-for-word the first's, so the device's
//! content-keyed modules answer them; their interface is the first's, so its pipeline layout
//! answers them; and their specialized entries are the device's, so no sibling decodes AIR
//! (`tests::stage_entries_shared` holds the compute form of that last one).
//!
//! The assertion is a count, never a time: it is a fact of the structure, so it holds on any host
//! under any load. `variant_pipelines_made` is the floor that keeps a pass from being vacuous --
//! a run whose siblings were all found whole in a memo would read zero for everything, and a
//! sweep that makes no pipeline is not the sweep this measures.
//!
//! `stage_entries_specialized` counts the **device's** entries, so a creation that kept entries of
//! its own would move it not at all and read as perfect sharing. The warm-up's own count is
//! therefore asserted to be above zero before the run, which is what makes the run's zero mean
//! shared rather than uncounted.
//!
//! The device takes a shader cache directory of its own, empty, so no run reads a pipeline an
//! earlier run stored.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/ubo_bind_offsets.ll`'s (`llvm-as`).

use super::blit_holds::{device, Scratch};
use super::descriptor_key_once::descriptor;
use crate::PipelineOptions;
use metal2vulkan::metal::{BlendFactor, BlendOperation};

/// The siblings the run makes after the warm-up's one.
const RUN: u64 = 16;

/// Blend factors no two of which make one pipeline: each sibling is its own descriptor.
const FACTORS: [BlendFactor; 8] = [
    BlendFactor::Zero,
    BlendFactor::One,
    BlendFactor::SourceColor,
    BlendFactor::OneMinusSourceColor,
    BlendFactor::SourceAlpha,
    BlendFactor::OneMinusSourceAlpha,
    BlendFactor::DestinationColor,
    BlendFactor::DestinationAlpha,
];

/// `RUN` sibling render pipelines of one function pair make a pipeline each and share every piece
/// that belongs to the shaders.
#[test]
fn a_run_of_render_creations_shares_every_piece_of_its_shaders() {
    let scratch = Scratch::new("render-sibling-work");
    let device = device(&scratch.0);
    let base = descriptor(&device);
    let options = PipelineOptions::default();
    let sibling = |i: u64| {
        let mut d = base.clone();
        let color = d.color_attachments[0]
            .as_mut()
            .expect("the fixture's target");
        color.blending_enabled = true;
        color.source_rgb_blend_factor = FACTORS[(i % 8) as usize];
        color.destination_rgb_blend_factor = FACTORS[(i / 8 % 8) as usize];
        color.rgb_blend_operation = BlendOperation::Add;
        d
    };
    // The warm-up's creation is the one that compiles the stages, makes their modules and their
    // layout, and specializes their AIR.
    let _first = device
        .new_render_pipeline_state(&sibling(0), &options)
        .unwrap();
    let before = device.work_counts();
    assert!(
        before.stage_entries_specialized > 0,
        "the first creation must specialize its stages into the device's entries at all, not {} \
         times: a creation that specializes nothing there, because it keeps entries of its own, \
         makes the run's zero a vacuous zero",
        before.stage_entries_specialized
    );
    let mut kept = Vec::with_capacity(RUN as usize);
    for i in 0..RUN {
        kept.push(
            device
                .new_render_pipeline_state(&sibling(1 + i), &options)
                .unwrap(),
        );
    }
    let after = device.work_counts();
    let made = |of: fn(&crate::WorkCounts) -> u64| of(&after) - of(&before);
    let pipelines = made(|c| c.variant_pipelines_made);
    assert_eq!(
        pipelines, RUN,
        "every sibling must make a pipeline of its own, else the run shares pieces it never asked \
         for: {pipelines} pipelines made over {RUN} creations"
    );
    assert_eq!(
        [
            made(|c| c.shader_modules_made),
            made(|c| c.pipeline_layouts_made),
            made(|c| c.stage_entries_specialized),
            made(|c| c.functions_translated),
        ],
        [0, 0, 0, 0],
        "the shader modules, pipeline layouts, AIR specializations and translations {RUN} sibling \
         render creations of one function pair did, each of them making its own pipeline"
    );
    // The floor under the translation zero, and the count that says the render path reaches the
    // same memo the compute path does: both stages of every sibling asked for their function and
    // were answered.
    let shared = made(|c| c.functions_shared);
    assert_eq!(
        shared,
        2 * RUN,
        "each sibling's vertex and fragment must ask for its function and be answered by the \
         memo: {shared} asks spared over {RUN} creations of two stages"
    );
}
