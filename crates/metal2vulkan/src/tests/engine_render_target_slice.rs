//! A vertex function's render target slice against the pass's render target array length, over the
//! owned fixture `engine/vertex_amplification` (a vertex function that chooses its slice,
//! `amp_vertex_layer`, and a fragment function that reads it, `amp_fragment_layer`) and the public
//! `vertex_amplification_id_layer`:
//!
//! - A vertex function that writes `[[render_target_array_index]]` depends on
//!   `RuntimeFact::RenderTargetArrayLength`: it writes `Layer` as 0 at or past the pass's length,
//!   which it reads from `PushField::RenderTargetArrayLength`, in a pass of any length (Metal's
//!   default length 0 and 1 render every index into slice 0). It also writes the index as written
//!   to a flat varying (`IoRole::RenderTargetArrayIndex`), which a fragment function that reads the
//!   slice reads in place of `Layer`.
//! - The render plan asks for no variant at any length.
//! - Amplified draws are never views: they draw one amplification at a time.
//!
//! Every module that compiles passes L3 (`spirv-val` and requirements within the profile).
//! `llvm-as` assembles the fixtures; a missing tool skips visibly. No GPU is used.

use std::path::Path;

use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Error, RuntimeInput};
use crate::library::Library;
use crate::pipeline::descriptor::{FunctionDescriptor, RenderPipelineDescriptor, StageSource};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::stage_io::{IoRole, StageBuiltin};
use crate::pipeline::sym::{Stage, VaryingKey};
use crate::pipeline::testing::{engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::RenderPipeline;
use crate::spirv::testing::check_words_valid;

fn assemble(path: &Path) -> Option<Vec<u8>> {
    match llvm_assemble(path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {path:?}: {why}"),
    }
}

fn description(stem: &str) -> DeviceDescription {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    DeviceDescription::from_khronos_json(&text, None).expect("description")
}

fn device(stem: &str) -> Device {
    Device::from_queries(description(stem), DevicePolicy::RESIDENT_TABLES).expect("device")
}

/// A render pipeline of `vertex` and `fragment` from `bc` with `maxVertexAmplificationCount`
/// `max`.
fn render(bc: &[u8], vertex: &str, fragment: Option<&str>, max: u32) -> RenderPipelineDescriptor {
    let lib = Library::from_air(bc).expect("library");
    let stage = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            lib.function(name).expect("function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(stage(vertex));
    d.fragment = fragment.map(stage);
    d.max_vertex_amplification_count = max;
    d
}

/// The `RenderTargetArrayLength` fact of `p`'s vertex function, if it chooses its slice.
fn layer_fact(p: &RenderPipeline) -> Option<bool> {
    p.dependencies.iter().find_map(|d| match d.fact {
        RuntimeFact::RenderTargetArrayLength { layer_dropped } => Some(layer_dropped),
        _ => None,
    })
}

/// Every stage of `p` passes L3 on `dev`.
fn valid(p: &RenderPipeline, dev: &Device, what: &str) {
    for m in &p.stages {
        check_words_valid(&m.spirv, &p.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
    }
}

/// A vertex function that writes `[[render_target_array_index]]` writes the index as written to a
/// flat varying (`IoRole::RenderTargetArrayIndex`), stores `Layer` and reads the pass's length from
/// its push data. In its pipeline a fragment function that reads the slice compiles under
/// `render_target_array_index_varying` and reads that varying, as `render_pipeline_key` names; a
/// fragment function that does not read it
/// keeps the key it has without the field. A reading fragment compiled for the other kind of vertex
/// function is `NeedsInput(RenderTargetArrayIndex)` either way. Every module passes L3.
#[test]
fn a_fragment_reads_the_slice_index_its_vertex_function_wrote() {
    let Some(bc) = assemble(&engine_dir().join("vertex_amplification.ll")) else {
        return;
    };
    let lib = Library::from_air(&bc).expect("library");
    let function = |name: &str| {
        FunctionDescriptor::new(lib.function(name).unwrap_or_else(|e| panic!("{name}: {e}")))
    };
    let dev = device("amd-780m-radv");
    let budget = Budget::standard();
    let reader = function("amp_fragment_layer");
    let d = render(&bc, "amp_vertex_layer", Some("amp_fragment_layer"), 1);
    let key = VariantKey::new();
    let what = "the vertex function's pipeline";
    let vertex_inputs = key.lowering_inputs(Stage::Vertex);
    let vertex = dev
        .compile_function(&function("amp_vertex_layer"), &vertex_inputs, &budget)
        .expect("vertex");
    let io = &vertex.artifact().interface.io;
    let raw: Vec<_> = io
        .outputs
        .iter()
        .filter(|v| v.role == IoRole::RenderTargetArrayIndex)
        .collect();
    assert_eq!(raw.len(), 1, "{what}");
    assert_eq!(
        raw[0].key,
        VaryingKey::render_target_array_index(),
        "{what}"
    );
    // The function stores its slice as `Layer`.
    assert!(io.builtins.contains(&StageBuiltin::Layer), "{what}");

    let keys = dev
        .render_variant_key(&d, &key, &budget, &crate::pipeline::StageEntries::new())
        .expect("keys");
    let inputs = keys.fragment_inputs.clone().expect("fragment inputs");
    assert!(inputs.render_target_array_index_varying, "{what}");
    // The vertex function writes every input the reader reads: none is absent.
    assert!(inputs.absent_varyings.is_empty(), "{what}");
    let pipe = dev
        .compile_render_variant(&d, &key, &budget)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    assert_eq!(pipe.key, keys.pipeline, "{what}");
    assert!(pipe.graphics.declares_layer, "{what}");
    valid(&pipe, &dev, what);
    let fragment = dev
        .compile_function(&reader, &inputs, &budget)
        .expect("fragment");
    assert_eq!(Some(*fragment.key()), keys.fragment, "{what}");
    let io = &fragment.artifact().interface.io;
    assert!(
        io.inputs
            .iter()
            .any(|v| v.role == IoRole::RenderTargetArrayIndex),
        "{what}"
    );
    assert!(!io.builtins.contains(&StageBuiltin::Layer), "{what}");

    // A fragment that does not read the slice keeps its key: its only output is the color 0 it
    // writes, which the pipeline has no attachment for. Its stage input `color` (`user(locn0)`),
    // which the vertex function does not write, reaches only that color, so the linking drops it
    // and the fragment compiles with it absent: its key is the one with exactly that absent input
    // and its push fields from the vertex function's push end, and not the one without the input.
    let plain = function("amp_fragment");
    let keys = dev
        .render_pipeline_key(
            &render(&bc, "amp_vertex_layer", Some("amp_fragment"), 1),
            &budget,
        )
        .expect("keys");
    let mut unattached = LoweringInputs::new();
    unattached.unattached_colors = [0].into();
    let absent: std::collections::BTreeSet<VaryingKey> = [VaryingKey::user(b"user(locn0)")].into();
    let fragment_inputs = keys.fragment_inputs.clone().expect("fragment inputs");
    eprintln!("absent varyings: {:?}", fragment_inputs.absent_varyings);
    assert_eq!(fragment_inputs.absent_varyings, absent);
    // Its push fields start at the vertex function's push end.
    assert_ne!(fragment_inputs.push_base, 0);
    unattached.push_base = fragment_inputs.push_base;
    let mut dropped = unattached.clone();
    dropped.absent_varyings = absent;
    assert_eq!(fragment_inputs, dropped);
    assert_eq!(
        keys.fragment,
        Some(dev.function_compile_key(&plain, &dropped))
    );
    assert_ne!(
        keys.fragment,
        Some(dev.function_compile_key(&plain, &unattached))
    );

    // Compiled for the other kind of vertex function: refused, never a wrong read.
    let without = dev
        .compile_function(&reader, &LoweringInputs::new(), &budget)
        .expect("fragment");
    let mut under = LoweringInputs::new();
    under.render_target_array_index_varying = true;
    let under = dev
        .compile_function(&reader, &under, &budget)
        .expect("fragment");
    let with = |vertex: &str, fragment| {
        let mut d = render(&bc, vertex, None, 1);
        d.fragment = Some(StageSource::Compiled(fragment));
        d
    };
    for d in [with("amp_vertex_layer", without), with("amp_vertex", under)] {
        assert_eq!(
            dev.compile_render_pipeline(&d, &budget).err(),
            Some(Error::NeedsInput(RuntimeInput::RenderTargetArrayIndex))
        );
    }
}

/// A compiled render pipeline states which indices its vertex function writes, whatever its
/// fragment function reads, as the runtime's `writes_render_target_array_index` (the
/// `RenderTargetArrayLength` fact) and `writes_viewport_array_index`
/// (`GraphicsState::writes_viewport_index`) answer them: `amp_vertex_layer` writes the slice and no
/// viewport, `amp_vertex` the viewport and no slice, with no fragment function or one that reads
/// either index; `amp_vertex_store` writes neither, and memory.
#[test]
fn the_pipeline_states_which_indices_its_vertex_function_writes() {
    let Some(bc) = assemble(&engine_dir().join("vertex_amplification.ll")) else {
        return;
    };
    let budget = Budget::standard();
    for stem in ["amd-780m-radv", "lavapipe"] {
        let dev = device(stem);
        for (vertex, fragment, layer, viewport, memory) in [
            ("amp_vertex_layer", None, true, false, false),
            (
                "amp_vertex_layer",
                Some("amp_fragment_layer"),
                true,
                false,
                false,
            ),
            ("amp_vertex", None, false, true, false),
            ("amp_vertex", Some("amp_fragment"), false, true, false),
            ("amp_vertex", Some("amp_fragment_layer"), false, true, false),
            ("amp_vertex_store", None, false, false, true),
        ] {
            let what = format!("{stem} {vertex} {fragment:?}");
            let p = dev
                .compile_render_pipeline(&render(&bc, vertex, fragment, 1), &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(layer_fact(&p).is_some(), layer, "{what}");
            assert_eq!(p.graphics.writes_viewport_index, viewport, "{what}");
            assert_eq!(p.graphics.vertex_writes_memory, memory, "{what}");
            assert!(!p.graphics.writes_point_size, "{what}");
        }
    }
}
