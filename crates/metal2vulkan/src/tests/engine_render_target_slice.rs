//! A vertex function's render target slice against the pass's render target array length, over the
//! owned fixture `engine/vertex_amplification` (a vertex function that chooses its slice,
//! `amp_vertex_layer`, and a fragment function that reads it, `amp_fragment_layer`) and the public
//! `vertex_amplification_id_layer`:
//!
//! - A vertex function that writes `[[render_target_array_index]]` depends on
//!   `RuntimeFact::RenderTargetArrayLength`. Compiled as is it writes `Layer` as 0 at or past the
//!   pass's length, which it reads from `PushField::RenderTargetArrayLength`; under
//!   `VariantKey::dropped_vertex_layer` it declares no `Layer` (so needs no `shaderOutputLayer`).
//!   Both write the index as written to a flat varying (`IoRole::RenderTargetArrayIndex`), which a
//!   fragment function that reads the slice behind either reads in place of `Layer`.
//! - The render plan asks for the dropped variant in a pass of at most one layer (Metal's
//!   default length 0, and 1) and for the variant that writes `Layer` in a pass of more; each
//!   variant's plan for its own length states its own key, and a draw of a plan whose pipeline is
//!   the other variant is `BindError::NeedsVariant`.
//! - Amplified draws are never views on either variant: they draw one amplification at a time.
//!
//! Every module that compiles passes L3 (`spirv-val` and requirements within the profile), and
//! the dropped vertex module converts to MSL and compiles as MoltenVK's does (`spirv-cross` and
//! `xcrun metal`, compile only). `llvm-as` assembles the fixtures; a missing tool skips visibly.
//! No GPU is used.

use std::path::Path;

use crate::bind::plan::FieldLocation;
use crate::bind::{BindState, DrawCall, VertexAmplification, ViewMapping, ViewSetup};
use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::requirements::{VkExtension, VkFeature as F};
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{BindError, Capability, Error, Feature, Needs, RuntimeInput};
use crate::library::Library;
use crate::pipeline::descriptor::{FunctionDescriptor, RenderPipelineDescriptor, StageSource};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::stage_io::{IoRole, StageBuiltin};
use crate::pipeline::sym::{PushField, Stage, VaryingKey};
use crate::pipeline::testing::{
    engine_dir, fixtures_dir, llvm_assemble, moltenvk_msl_compile, AssembleError,
};
use crate::pipeline::{AmplificationDraws, RenderPipeline};
use crate::spirv::testing::check_words_valid;
use crate::vk;

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
    Device::from_queries(description(stem), DevicePolicy::default()).expect("device")
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

/// A bind state of a pass of `length` layers.
fn pass(length: u32) -> BindState {
    BindState {
        render_target_array_length: length,
        ..Default::default()
    }
}

fn draw() -> DrawCall {
    DrawCall::Primitives {
        vertex_start: 0,
        vertex_count: 3,
        instance_count: 1,
        base_instance: 0,
    }
}

/// The `RenderTargetArrayLength` fact of `p`'s vertex function, if it chooses its slice.
fn layer_fact(p: &RenderPipeline) -> Option<bool> {
    p.dependencies.iter().find_map(|d| match d.fact {
        RuntimeFact::RenderTargetArrayLength { layer_dropped } => Some(layer_dropped),
        _ => None,
    })
}

/// Whether `p`'s vertex stage reads the pass's length from its push data.
fn reads_length(p: &RenderPipeline) -> bool {
    p.interface.push.iter().any(|pl| {
        pl.stage == Stage::Vertex
            && pl
                .fields
                .iter()
                .any(|f| f.field == PushField::RenderTargetArrayLength)
    })
}

/// The bytes of `p`'s vertex push data that hold the pass's length, in the draw of a pass of
/// `length` layers.
fn length_word(p: &RenderPipeline, dev: &Device, length: u32) -> [u8; 4] {
    let state = pass(length);
    let plan = p.plan(dev, &state.shape()).expect("plan");
    let placed = plan
        .push
        .fields
        .iter()
        .find(|f| f.stage == Stage::Vertex && f.field == PushField::RenderTargetArrayLength)
        .expect("placed");
    let FieldLocation::PushWindow { offset } = placed.location else {
        panic!("the length spilled: {placed:?}");
    };
    let commands = plan.instantiate_draw(&state, &draw()).expect("draw");
    let write = commands
        .push_constants
        .iter()
        .find(|w| w.stages.contains(vk::ShaderStageFlags::VERTEX))
        .expect("vertex push");
    let at = (offset - write.offset) as usize;
    write.bytes[at..at + 4].try_into().expect("a word")
}

/// Every stage of `p` passes L3 on `dev`.
fn valid(p: &RenderPipeline, dev: &Device, what: &str) {
    for m in &p.stages {
        check_words_valid(&m.spirv, &p.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
    }
}

/// A pass of at most one layer runs the vertex function that chooses its slice without its `Layer`
/// output, a pass of more with it; each plan states the variant its length asks for, and a draw on
/// the other variant is `NeedsVariant`. Only the variant that writes `Layer` reads the pass's
/// length, which each of its draws writes.
#[test]
fn a_pass_of_one_layer_runs_the_vertex_function_without_its_layer() {
    let Some(bc) = assemble(&engine_dir().join("vertex_amplification.ll")) else {
        return;
    };
    let Some(public) = assemble(&fixtures_dir().join("vertex_amplification_id_layer.ll")) else {
        return;
    };
    let budget = Budget::standard();
    let pipelines = [
        (&bc, "amp_vertex_layer", None),
        (&bc, "amp_vertex_layer", Some("amp_fragment_layer")),
        (&public, "vertex_amplification_id_layer", None),
    ];
    for stem in ["moltenvk-m3", "lavapipe"] {
        let dev = device(stem);
        for (bc, vertex, fragment) in pipelines {
            let what = format!("{stem} {vertex} {fragment:?}");
            let d = render(bc, vertex, fragment, 2);
            let written = dev
                .compile_render_pipeline(&d, &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            valid(&written, &dev, &what);
            assert_eq!(layer_fact(&written), Some(false), "{what}");
            assert!(written.graphics.declares_layer, "{what}");
            assert!(reads_length(&written), "{what}");
            assert_eq!(length_word(&written, &dev, 4), 4u32.to_le_bytes(), "{what}");

            // Metal's default length (0) and a one-slice pass ask for the dropped variant, which
            // the pipeline that writes `Layer` does not draw.
            let mut dropped_key = None;
            for length in [0, 1] {
                let plan = written.plan(&dev, &pass(length).shape()).expect("plan");
                assert!(plan.variant.dropped_vertex_layer, "{what} {length}");
                assert_eq!(
                    plan.instantiate_draw(&pass(length), &draw()),
                    Err(BindError::NeedsVariant {
                        variant: Box::new(plan.variant.clone())
                    }),
                    "{what} {length}"
                );
                dropped_key = Some(plan.variant);
            }
            let dropped_key = dropped_key.expect("a key");
            // A layered pass keeps it.
            let plan = written.plan(&dev, &pass(4).shape()).expect("plan");
            assert!(plan.variant.is_empty(), "{what}");
            plan.instantiate_draw(&pass(4), &draw())
                .unwrap_or_else(|e| panic!("{what}: {e:?}"));

            let keys = dev
                .render_variant_key(
                    &d,
                    &dropped_key,
                    &budget,
                    &crate::pipeline::StageEntries::new(),
                )
                .expect("keys");
            assert!(
                keys.vertex_inputs
                    .as_ref()
                    .expect("inputs")
                    .dropped_vertex_layer,
                "{what}"
            );
            assert_ne!(
                keys.vertex,
                dev.render_pipeline_key(&d, &budget).expect("keys").vertex,
                "{what}"
            );
            let dropped = dev
                .compile_render_variant(&d, &dropped_key, &budget)
                .unwrap_or_else(|e| panic!("{what}: {e}"));
            valid(&dropped, &dev, &what);
            assert_eq!(dropped.key, keys.pipeline, "{what}");
            assert_eq!(layer_fact(&dropped), Some(true), "{what}");
            // No stage declares `Layer` now: a fragment function that reads the slice reads the
            // raw varying.
            assert!(!dropped.graphics.declares_layer, "{what}");
            assert!(!reads_length(&dropped), "{what}");
            for length in [0, 1] {
                let plan = dropped.plan(&dev, &pass(length).shape()).expect("plan");
                assert_eq!(plan.variant, dropped_key, "{what} {length}");
                plan.instantiate_draw(&pass(length), &draw())
                    .unwrap_or_else(|e| panic!("{what} {length}: {e:?}"));
            }
            let plan = dropped.plan(&dev, &pass(4).shape()).expect("plan");
            assert_eq!(
                plan.instantiate_draw(&pass(4), &draw()),
                Err(BindError::NeedsVariant {
                    variant: Box::new(VariantKey::new())
                }),
                "{what}"
            );

            // Neither variant draws amplified as views: each plans the draws of one amplification
            // at a time, without `Layer` in a pass of one layer.
            for p in [&written, &dropped] {
                let mut amplified = pass(1);
                amplified.vertex_amplification = VertexAmplification::new(
                    2,
                    vec![ViewMapping::new(0, 0), ViewMapping::new(0, 1)],
                );
                let plan = p
                    .plan(&dev, &amplified.shape())
                    .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                assert_eq!(plan.views, ViewSetup::default(), "{what}");
                assert!(plan.variant.dropped_vertex_layer, "{what}");
                assert_eq!(
                    plan.variant.amplification_draws,
                    Some(AmplificationDraws::new(false, false)),
                    "{what}"
                );
            }

            if stem == "moltenvk-m3" {
                let words = &dropped
                    .stages
                    .iter()
                    .find(|m| m.stage == vk::ShaderStage::VERTEX)
                    .expect("vertex")
                    .spirv;
                match moltenvk_msl_compile(words) {
                    Ok(()) => {}
                    Err(AssembleError::Missing(why)) => eprintln!("SKIPPED: {why}"),
                    Err(AssembleError::Rejected(why)) => panic!("{what}: MoltenVK's MSL: {why}"),
                }
            }
        }
    }
}

/// A fragment function that reads its slice behind a vertex function that does not choose one
/// asks for no variant at any length: there is no `Layer` output to drop.
#[test]
fn a_fragment_reading_its_slice_alone_asks_for_no_variant() {
    let Some(bc) = assemble(&engine_dir().join("vertex_amplification.ll")) else {
        return;
    };
    let dev = device("moltenvk-m3");
    let p = dev
        .compile_render_pipeline(
            &render(&bc, "amp_vertex", Some("amp_fragment_layer"), 1),
            &Budget::standard(),
        )
        .expect("pipeline");
    assert_eq!(layer_fact(&p), None);
    for length in [0, 1, 4] {
        let plan = p.plan(&dev, &pass(length).shape()).expect("plan");
        assert!(plan.variant.is_empty(), "{length}");
        plan.instantiate_draw(&pass(length), &draw())
            .unwrap_or_else(|e| panic!("{length}: {e:?}"));
    }
}

/// The dropped variant declares no `Layer` output, so a device that cannot write `Layer` from a
/// vertex function runs a pass of one layer exactly, where the variant that writes `Layer` is
/// refused typed when the device has no geometry stage to write it either (MoltenVK has none).
/// The device keeps `VK_EXT_shader_viewport_index_layer`: without `shaderOutputLayer` it grants no
/// `Layer` output (the capability a module declares at the 1.2 floor is the feature's), so the
/// refusal names what would store the slice rather than failing as a requirement beyond the
/// profile: `geometryShader`, as `amp_vertex_layer`'s slice (`slices[vid & 1]`) differs among a
/// primitive's vertices, which only the synthesized geometry stage stores without the output
/// MoltenVK hands to Metal.
#[test]
fn the_dropped_variant_needs_no_output_layer() {
    let Some(bc) = assemble(&engine_dir().join("vertex_amplification.ll")) else {
        return;
    };
    let mut desc = description("moltenvk-m3");
    assert!(desc.features.remove(&F::ShaderOutputLayer));
    assert!(desc
        .extensions
        .contains(&VkExtension::ExtShaderViewportIndexLayer));
    let dev = Device::from_queries(desc, DevicePolicy::default()).expect("device");
    let d = render(&bc, "amp_vertex_layer", None, 1);
    let budget = Budget::standard();
    match dev.compile_render_pipeline(&d, &budget) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::LayeredRendering);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![Capability::Feature(F::GeometryShader)]])
            );
        }
        other => panic!("{other:?}"),
    }
    let mut key = VariantKey::new();
    key.dropped_vertex_layer = true;
    let p = dev
        .compile_render_variant(&d, &key, &budget)
        .expect("dropped variant");
    valid(&p, &dev, "without shaderOutputLayer");
    assert!(!p.requirements.features.contains(&F::ShaderOutputLayer));
    let plan = p.plan(&dev, &pass(1).shape()).expect("plan");
    assert_eq!(plan.variant, key);
    plan.instantiate_draw(&pass(1), &draw()).expect("draw");
}

/// A vertex function that writes `[[render_target_array_index]]` writes the index as written to a
/// flat varying (`IoRole::RenderTargetArrayIndex`) in both variants, and in the one that writes
/// `Layer` reads the pass's length from its push data. In its pipeline a fragment function that
/// reads the slice compiles under `render_target_array_index_varying` and reads that varying, as
/// `render_pipeline_key` names, behind either variant; a fragment function that does not read it
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
    let dev = device("moltenvk-m3");
    let budget = Budget::standard();
    let reader = function("amp_fragment_layer");
    let d = render(&bc, "amp_vertex_layer", Some("amp_fragment_layer"), 1);
    let mut dropped = VariantKey::new();
    dropped.dropped_vertex_layer = true;
    for key in [VariantKey::new(), dropped] {
        let what = format!("dropped {}", key.dropped_vertex_layer);
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
        assert_eq!(
            io.builtins.contains(&StageBuiltin::Layer),
            !key.dropped_vertex_layer,
            "{what}"
        );

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
        valid(&pipe, &dev, &what);
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
    }

    // A fragment that does not read the slice keeps its key: its only output is the color 0 it
    // writes, which the pipeline has no attachment for. Its stage input `color` (`user(locn0)`),
    // which the vertex function does not write, reaches only that color, so the linking drops it
    // and the fragment compiles with it absent: its key is the one with exactly that absent input,
    // and not the one without.
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
    for stem in ["moltenvk-m3", "lavapipe"] {
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
