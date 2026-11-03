//! Render states whose whole key is equal, each a variant of its own, share one linked
//! `VkPipeline`: the device's, found by exactly what its linking takes, its four libraries and its
//! pipeline layout (`pipeline/libraries.rs`, `WorkCounts::linked_pipelines_shared`).
//!
//! The siblings here are descriptors of one vertex-and-fragment pair that differ only in
//! `maxFragmentCallStackDepth`, which sizes the call stack of a fragment function's recursion
//! through visible functions: each compiles its fragment function under a depth of its own, and
//! so is a compiled pipeline and a variant of its own, while the fixture's fragment function calls
//! no visible function, so every sibling's stages come out as the first creation's words, and
//! every part of its pipelines, every library and every link, is the first creation's. Each
//! sibling's creation takes the first creation's linked pipelines with no driver call, and each
//! sibling draws with them the pixels the fixture's constants give.
//!
//! A descriptor that differs only in what no compile reads (a buffer's mutability) is not a
//! sibling of this kind: it is the first creation's compiled pipeline, whose live variant it
//! shares whole (`memo.rs`).
//!
//! On a device that makes its render pipelines whole (no `VK_EXT_graphics_pipeline_library` with
//! fast linking: MoltenVK) nothing is linked, so nothing is shared; the run says so and checks
//! the draws alone.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. The functions are the owned fixture
//! `validation/fixtures/public/engine/ubo_bind_offsets.ll`'s (`llvm-as`).

use super::blit_holds::{device, Scratch};
use super::descriptor_key_once::descriptor;
use super::uniform_bind_offsets::{draw, idle, Bound};
use crate::{PipelineOptions, WorkCounts};
use std::time::Instant;

/// The siblings made after the first state.
const RUN: u32 = 16;

/// The draw each state makes: the fragment's constants at byte 0 and the vertex's at byte 256 of
/// the constants buffer, offsets every device's uniform alignment divides.
const CASE: Bound = Bound::Buffer(0, 256);

#[test]
fn render_states_of_one_whole_key_share_one_linked_pipeline() {
    let scratch = Scratch::new("linked-pipelines-shared");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let base = descriptor(&device);
    let options = PipelineOptions::default();
    let first = device.new_render_pipeline_state(&base, &options).unwrap();
    draw(&device, &queue, &first, &[CASE]);
    let before = idle(&device);
    let mut siblings = Vec::with_capacity(RUN as usize);
    let mut times = Vec::with_capacity(RUN as usize);
    for i in 0..RUN {
        let mut d = base.clone();
        // Past Metal's default, 1, each a depth of its own.
        d.max_fragment_call_stack_depth = 2 + i;
        let started = Instant::now();
        siblings.push(device.new_render_pipeline_state(&d, &options).unwrap());
        times.push(started.elapsed().as_secs_f64() * 1e3);
    }
    for sibling in &siblings {
        draw(&device, &queue, sibling, &[CASE]);
    }
    let after = idle(&device);
    let made = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
    let (linked, shared) = (
        made(|c| c.pipelines_linked),
        made(|c| c.linked_pipelines_shared),
    );
    let libraries = made(|c| c.pipeline_libraries_made);
    let pipelines = made(|c| c.variant_pipelines_made);
    times.sort_by(f64::total_cmp);
    println!(
        "LINKED-PIPELINES-SHARED: {RUN} siblings: {linked} linked, {shared} shared, {libraries} \
         libraries made, {pipelines} variant pipelines made; creation median {:.3} ms, max {:.3} ms",
        times[times.len() / 2],
        times[times.len() - 1],
    );
    if device.shared().libraries.is_none() {
        assert_eq!(
            shared, 0,
            "a device that makes its render pipelines whole links none, so shares none"
        );
        return;
    }
    // Each sibling is a variant of its own: it made no library and linked nothing, and took at
    // least one linked pipeline, its creation's, from the first state.
    assert_eq!(
        (linked, libraries, pipelines),
        (0, 0, 0),
        "every one of {RUN} siblings of one whole key must take the first state's linked \
         pipelines and libraries: {linked} linked, {libraries} libraries and {pipelines} variant \
         pipelines made, {shared} shared"
    );
    assert!(
        shared >= u64::from(RUN),
        "every one of {RUN} siblings must take at least its creation's linked pipeline from the \
         first state: {shared} taken"
    );
}
