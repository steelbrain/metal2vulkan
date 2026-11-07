//! A render or mesh pipeline creation builds its descriptor's key **once**
//! (`descriptor_keys_built`), however many sites of the creation read it, and the state it makes
//! holds that key rather than building it again. The one creation that builds a second is one
//! whose fragment stage it drops because the state does not rasterize: what it compiles is then
//! not the descriptor its checks asked their questions of, and the key tells the two apart.
//!
//! # Why this is a count
//!
//! The key is a digest of every field of the descriptor, each stage by its compile key: building
//! one formats a dozen of the descriptor's fields in their derived `Debug` form and hashes two
//! function compile keys. Nothing of it depends on anything the creation works out, so a creation
//! that builds it at each site that wants it -- its linked-use answers, its variant compiles, its
//! reflection, its state -- pays for the same digest several times over. The assertion is a count
//! of builds, never a time: it is a fact of the structure, so it holds on any host under any
//! load, needs no quiet host and depends on nothing that ran before it in the process.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The functions are those of the owned
//! fixture `validation/fixtures/public/engine/ubo_bind_offsets.ll` (`llvm-as`).

use super::blit_holds::{device, Scratch};
use crate::{Device, PipelineOption, PipelineOptions, RenderPipelineState};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource};
use std::path::Path;

/// The fixture's render descriptor, one `RGBA8Unorm` target.
pub(super) fn descriptor(device: &Device) -> RenderPipelineDescriptor {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/ubo_bind_offsets.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = device.new_library_with_compile_library(
        metal2vulkan::library::Library::from_air(&bitcode).unwrap(),
    );
    let vertex = library
        .new_function_with_name("ubo_offsets_vertex")
        .unwrap();
    let fragment = library
        .new_function_with_name("ubo_offsets_fragment")
        .unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    descriptor
}

/// How many descriptor keys `device` built, and the other work a creation does: a removal of
/// repeated key building moves the first and leaves the rest where they were.
pub(super) fn built(device: &Device) -> [u64; 6] {
    let c = device.work_counts();
    [
        c.descriptor_keys_built,
        c.shader_modules_made,
        c.pipeline_layouts_made,
        c.variant_pipelines_made,
        c.stage_entries_specialized,
        c.linked_use_questions_asked,
    ]
}

/// A creation builds one key, whether or not it reflects, and the state's own key is the one its
/// creation built: its `compiledReflection` builds none. The second creation is of the same
/// descriptor, so its answers come from the shader cache and it asks no linked-use question; the
/// key it builds is the one it looks them up by.
#[test]
fn a_render_creation_builds_one_descriptor_key() {
    let scratch = Scratch::new("descriptor-key-once");
    let device = device(&scratch.0);
    let descriptor = descriptor(&device);
    let plain = PipelineOptions::default();
    let before = built(&device);
    let state = device
        .new_render_pipeline_state(&descriptor, &plain)
        .unwrap();
    let made = built(&device);
    assert_eq!(
        made[0] - before[0],
        1,
        "descriptor keys a creation without reflection built"
    );
    // The state answers from the key its creation built.
    let _ = state.compiled_reflection().unwrap();
    assert_eq!(
        built(&device)[0],
        made[0],
        "descriptor keys a state's reflection read built"
    );
    let reflecting = PipelineOptions {
        options: PipelineOption::BINDING_INFO,
        ..PipelineOptions::default()
    };
    let (_state, reflection) = device
        .new_render_pipeline_state_with_reflection(&descriptor, &reflecting)
        .unwrap();
    assert!(reflection.is_some(), "the creation reflected");
    let reflected = built(&device);
    assert_eq!(
        reflected[0] - made[0],
        1,
        "descriptor keys a creation with reflection built"
    );
    // Only the key building moved: the second creation is of the same descriptor, so it made no
    // module, layout or pipeline of its own, shared the first's stage entries and read the first's
    // linked-use answers from the shader cache -- the same counts it read before the key was built
    // once instead of three times.
    assert_eq!(
        &reflected[1..],
        &made[1..],
        "the second creation's other work: nothing made again for one descriptor"
    );
}

/// A plain render creation -- rasterizing, a vertex and a fragment function, no indirect command
/// buffers -- on a fresh device asks two linked-use questions (the fragment's depth write and the
/// unwritten fragment inputs) and specializes its two stages once each. Whether its vertex
/// function returns a value is no question of its own: the compile reads it from the vertex entry
/// it specializes anyway, and its link refuses a `void` one (`pipeline/unrasterized.rs`). A
/// question for it cost every such creation a third question and a third stage entry, the
/// answer's, in the cost records' render creations.
#[test]
fn a_plain_render_creation_asks_two_questions_and_specializes_two_stages() {
    let scratch = Scratch::new("plain-render-counts");
    let device = device(&scratch.0);
    let descriptor = descriptor(&device);
    let before = built(&device);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let made = built(&device);
    assert_eq!(
        [made[4] - before[4], made[5] - before[5]],
        [2, 2],
        "stage entries specialized and linked-use questions asked"
    );
}

/// A state made of the descriptor again builds one key of its own: the key belongs to the
/// creation, not to the state, so a second state of one descriptor neither keeps nor reuses the
/// first's.
#[test]
fn every_creation_of_one_descriptor_builds_one_key() {
    let scratch = Scratch::new("descriptor-key-each");
    let device = device(&scratch.0);
    let descriptor = descriptor(&device);
    let plain = PipelineOptions::default();
    let before = built(&device);
    let states: Vec<RenderPipelineState> = (0..4)
        .map(|_| {
            device
                .new_render_pipeline_state(&descriptor, &plain)
                .unwrap()
        })
        .collect();
    assert_eq!(states.len(), 4);
    assert_eq!(
        built(&device)[0] - before[0],
        4,
        "descriptor keys four creations of one descriptor built"
    );
}

/// A creation the checks refuse builds the one key they asked their questions under, and no key
/// for a compile it never reaches: a descriptor that does not rasterize whose vertex function
/// returns a value is `Invalid`, which Metal answers before it compiles anything
/// (`pipeline/unrasterized.rs`).
#[test]
fn a_refused_creation_builds_one_descriptor_key() {
    let scratch = Scratch::new("descriptor-key-refused");
    let device = device(&scratch.0);
    let mut descriptor = descriptor(&device);
    descriptor.rasterization_enabled = false;
    let before = built(&device);
    let refused = device.new_render_pipeline_state(&descriptor, &PipelineOptions::default());
    assert!(refused.is_err(), "the creation was refused: {refused:?}");
    assert_eq!(
        built(&device)[0] - before[0],
        1,
        "descriptor keys a refused creation built"
    );
}
