//! The render target slice and viewport a vertex function selects, over the owned fixtures
//! `engine/rt6_layered` (a vertex function that chooses its slice per instance, and fragment
//! functions that read it, one with the barycentric weights and the primitive ID),
//! `engine/rt6_viewports` (a vertex function that selects its viewport per instance, and a
//! fragment function that reads it), `engine/rt6_per_primitive` (a vertex function that reads its
//! slice and viewport per triangle from a device buffer at `vertex_id / 3`),
//! `engine/rt6_first_vertex` (one that reads them per vertex) and `engine/rt6_unwritten` (a vertex
//! function that declares both and writes neither):
//!
//! - An index is stored to `Layer` or `ViewportIndex` as the function writes it, where the device
//!   stores the builtin from a vertex function, whatever its structure shows of it: Metal and
//!   Vulkan both ask the same value at every vertex of a primitive. Without `shaderOutputLayer`
//!   (`shaderOutputViewportIndex`) it is refused needing that feature alone.
//! - Without `multiViewport` a draw has one viewport, so the vertex function stores no viewport
//!   index at all.
//! - An index declared and never written, which Metal gives no value, is stored as 0.
//! - A fragment function that reads an index no stage writes, on a device that cannot declare its
//!   builtin, reads 0 (`StageIo::zero_indices`).
//!
//! Every module that compiles passes L3 (`spirv-val` and requirements within the profile).
//! `llvm-as` assembles the fixtures; a missing tool skips visibly. No GPU is used.

use std::path::Path;

use crate::budget::Budget;
use crate::device::requirements::VkFeature as F;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Capability, Error, Feature, Needs};
use crate::library::Library;
use crate::metal::PixelFormat;
use crate::pipeline::descriptor::{
    ColorAttachmentDescriptor, FunctionDescriptor, RenderPipelineDescriptor, StageSource,
};
use crate::pipeline::stage_io::{IoRole, StageBuiltin};
use crate::pipeline::testing::{compile, engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::RenderPipeline;
use crate::spirv::testing::check_words_valid;
use crate::spv::{BuiltIn, Capability as Cap};

fn assemble(name: &str) -> Option<Vec<u8>> {
    let path = engine_dir().join(name);
    match llvm_assemble(&path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {path:?}: {why}"),
    }
}

/// Committed profile `stem`'s description without `withheld`.
fn device(stem: &str, withheld: &[F]) -> Device {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let mut desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    for f in withheld {
        desc.features.remove(f);
    }
    Device::from_queries(desc, DevicePolicy::RESIDENT_TABLES).expect("device")
}

/// A render pipeline of `vertex` and `fragment` from `bc`, drawing into one color of `format`.
fn render(
    bc: &[u8],
    vertex: &str,
    fragment: &str,
    format: PixelFormat,
) -> RenderPipelineDescriptor {
    let lib = Library::from_air(bc).expect("library");
    let stage = |name: &str| {
        StageSource::Function(FunctionDescriptor::new(
            lib.function(name).expect("function"),
        ))
    };
    let mut d = RenderPipelineDescriptor::new(stage(vertex));
    d.fragment = Some(stage(fragment));
    d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(format));
    d
}

/// Every stage of `p` valid on `dev`.
fn validate(p: &RenderPipeline, dev: &Device, what: &str) {
    for m in &p.stages {
        check_words_valid(&m.spirv, &p.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
    }
}

/// The roles of the vertex outputs `entry` of `bc` declares on `dev`, and its builtins.
fn vertex_io(bc: &[u8], entry: &str, dev: &Device) -> (Vec<IoRole>, Vec<StageBuiltin>) {
    let a = compile(bc, entry, dev.profile()).unwrap_or_else(|e| panic!("{entry}: {e}"));
    let io = &a.interface.io;
    (
        io.outputs.iter().map(|o| o.role.clone()).collect(),
        io.builtins.clone(),
    )
}

/// `result` is the refusal of layered rendering with `needs`.
fn layered(result: Result<RenderPipeline, Error>, needs: Needs, what: &str) {
    match result {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::LayeredRendering, "{what}");
            assert_eq!(u.needs, needs, "{what}");
        }
        other => panic!("{what}: {:?}", other.map(|_| ())),
    }
}

fn needs(feature: F) -> Needs {
    Needs::AnyOf(vec![vec![Capability::Feature(feature)]])
}

/// A slice written the same at every vertex of a primitive is stored to `Layer` where the device
/// stores it from a vertex function, and refused needing `shaderOutputLayer` alone without it,
/// `geometryShader` or not.
#[test]
fn a_uniform_slice_takes_the_output_and_nothing_else() {
    let Some(bc) = assemble("rt6_layered.ll") else {
        return;
    };
    let budget = Budget::standard();
    let d = render(
        &bc,
        "rt6_layered_vertex",
        "rt6_layered_fragment",
        PixelFormat::R32Uint,
    );
    let native = device("lavapipe", &[]);
    let (outputs, builtins) = vertex_io(&bc, "rt6_layered_vertex", &native);
    assert!(builtins.contains(&StageBuiltin::Layer), "{builtins:?}");
    assert!(
        outputs.contains(&IoRole::RenderTargetArrayIndex),
        "{outputs:?}"
    );
    let p = native
        .compile_render_pipeline(&d, &budget)
        .expect("pipeline");
    validate(&p, &native, "native");
    assert!(p.graphics.declares_layer);
    assert!(p.requirements.features.contains(&F::ShaderOutputLayer));
    for (stem, without) in [
        ("lavapipe", device("lavapipe", &[F::ShaderOutputLayer])),
        (
            "runtime-floor",
            device("runtime-floor", &[F::ShaderOutputLayer]),
        ),
    ] {
        layered(
            without.compile_render_pipeline(&d, &budget),
            needs(F::ShaderOutputLayer),
            stem,
        );
    }
}

/// A viewport written the same at every vertex of a primitive is stored to `ViewportIndex` where
/// the device stores it from a vertex function, through `ShaderViewportIndex`; a device with
/// `multiViewport` but without `shaderOutputViewportIndex` refuses it needing that feature alone,
/// and a device without `multiViewport` draws through one viewport and stores none, its pipeline
/// still stating that the vertex function writes the index.
#[test]
fn a_uniform_viewport_takes_the_output_or_goes_nowhere() {
    let Some(bc) = assemble("rt6_viewports.ll") else {
        return;
    };
    let budget = Budget::standard();
    let d = render(
        &bc,
        "rt6_select",
        "rt6_viewport_read",
        PixelFormat::RGBA8Unorm,
    );
    let native = device("lavapipe", &[]);
    let (outputs, builtins) = vertex_io(&bc, "rt6_select", &native);
    assert!(
        builtins.contains(&StageBuiltin::ViewportIndex),
        "{builtins:?}"
    );
    assert!(outputs.contains(&IoRole::ViewportIndex), "{outputs:?}");
    let p = native
        .compile_render_pipeline(&d, &budget)
        .expect("pipeline");
    validate(&p, &native, "native");
    assert!(p.graphics.writes_viewport_index);
    let index = crate::device::SpirvCapability::from_raw(Cap::ShaderViewportIndex as u32);
    assert!(p.requirements.spirv_capabilities.contains(&index));
    layered(
        device("lavapipe", &[F::ShaderOutputViewportIndex]).compile_render_pipeline(&d, &budget),
        needs(F::ShaderOutputViewportIndex),
        "multiViewport without the output",
    );
    // Without `multiViewport`, below the floor too, which requires it: the raw index alone.
    for dev in [
        device(
            "lavapipe",
            &[F::ShaderOutputViewportIndex, F::MultiViewport],
        ),
        device(
            "runtime-floor",
            &[F::ShaderOutputViewportIndex, F::MultiViewport],
        ),
    ] {
        let (outputs, builtins) = vertex_io(&bc, "rt6_select", &dev);
        assert_eq!(outputs, [IoRole::ViewportIndex], "{builtins:?}");
        assert!(!builtins.contains(&StageBuiltin::ViewportIndex));
        let p = dev.compile_render_pipeline(&d, &budget).expect("pipeline");
        validate(&p, &dev, "one viewport");
        // The function still writes the index: the runtime refuses its draw while the encoder
        // holds more than one viewport.
        assert!(p.graphics.writes_viewport_index);
    }
}

/// A vertex function that reads its slice and viewport per triangle from a device buffer
/// (`engine/rt6_per_primitive`), or per vertex (`engine/rt6_first_vertex`), which its structure
/// does not show the same at every vertex of a primitive, stores both to `Layer` and
/// `ViewportIndex` as it writes them, in a pipeline of its vertex and fragment stages alone, on
/// every device with the outputs, with `geometryShader` and without it; a device without
/// `shaderOutputLayer` refuses it needing that feature alone.
#[test]
fn indices_are_stored_as_the_vertex_function_writes_them() {
    let Some(per_primitive) = assemble("rt6_per_primitive.ll") else {
        return;
    };
    let Some(first_vertex) = assemble("rt6_first_vertex.ll") else {
        return;
    };
    let budget = Budget::standard();
    let pipelines = [
        (
            render(
                &per_primitive,
                "rt6_per_primitive",
                "rt6_per_primitive_read",
                PixelFormat::R32Uint,
            ),
            &per_primitive,
            &["rt6_per_primitive"][..],
        ),
        (
            render(
                &first_vertex,
                "rt6_first_vertex",
                "rt6_first_vertex_read",
                PixelFormat::R32Uint,
            ),
            &first_vertex,
            &["rt6_first_vertex", "rt6_first_vertex_point"][..],
        ),
    ];
    for (stem, dev) in [
        ("lavapipe", device("lavapipe", &[])),
        ("moltenvk-m3", device("moltenvk-m3", &[])),
        (
            "lavapipe without geometryShader",
            device("lavapipe", &[F::GeometryShader]),
        ),
        ("runtime-floor", device("runtime-floor", &[])),
    ] {
        for (d, bc, entries) in &pipelines {
            for &entry in *entries {
                let (_, builtins) = vertex_io(bc, entry, &dev);
                for b in [StageBuiltin::Layer, StageBuiltin::ViewportIndex] {
                    assert!(builtins.contains(&b), "{stem} {entry}: {builtins:?}");
                }
            }
            let p = (dev.compile_render_pipeline(d, &budget))
                .unwrap_or_else(|e| panic!("{stem} {entries:?}: {e}"));
            validate(&p, &dev, stem);
            let stages: Vec<_> = p.stages.iter().map(|m| m.stage).collect();
            assert_eq!(
                stages,
                [
                    crate::vk::ShaderStage::VERTEX,
                    crate::vk::ShaderStage::FRAGMENT
                ],
                "{stem} {entries:?}"
            );
            assert!(p.graphics.declares_layer, "{stem} {entries:?}");
            assert!(p.graphics.writes_viewport_index, "{stem} {entries:?}");
        }
    }
    let (d, _, _) = &pipelines[0];
    layered(
        device("runtime-floor", &[F::ShaderOutputLayer]).compile_render_pipeline(d, &budget),
        needs(F::ShaderOutputLayer),
        "runtime-floor without shaderOutputLayer",
    );
}

/// A vertex function that declares its slice and viewport and writes neither
/// (`engine/rt6_unwritten`, both `undef` in its return) stores each as 0, the same at every vertex
/// of a primitive: its stage declares `Layer` and `ViewportIndex`, and no value it stores to either
/// reads an `OpUndef`, in a pipeline with a fragment function that reads both, on a full device
/// and on the floor.
#[test]
fn indices_a_vertex_function_never_writes_are_stored_as_zero() {
    let Some(bc) = assemble("rt6_unwritten.ll") else {
        return;
    };
    let d = render(
        &bc,
        "rt6_unwritten",
        "rt6_unwritten_read",
        PixelFormat::R32Uint,
    );
    for (stem, dev) in [
        ("lavapipe", device("lavapipe", &[])),
        ("runtime-floor", device("runtime-floor", &[])),
    ] {
        let a =
            compile(&bc, "rt6_unwritten", dev.profile()).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let io = &a.interface.io;
        for b in [StageBuiltin::Layer, StageBuiltin::ViewportIndex] {
            assert!(io.builtins.contains(&b), "{stem}: {:?}", io.builtins);
        }
        let p = dev
            .compile_render_pipeline(&d, &Budget::standard())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        validate(&p, &dev, stem);
        let vertex = (p.stages.iter())
            .find(|m| m.stage == crate::vk::ShaderStage::VERTEX)
            .expect("vertex stage");
        for b in [BuiltIn::Layer, BuiltIn::ViewportIndex] {
            let stores = stores_read_undef(&vertex.spirv, b);
            assert!(!stores.is_empty(), "{stem}: no store to {b:?}");
            assert!(!stores.contains(&true), "{stem}: {b:?} stores {stores:?}");
        }
    }
}

/// For each `OpStore` of `words` through the variable builtin `b` decorates, whether its value
/// reads an `OpUndef`: through the instructions that define every id it reads, transitively.
fn stores_read_undef(words: &[u32], b: BuiltIn) -> Vec<bool> {
    use crate::spirv_binary::{parse_bytes, Operand};
    use crate::spv::{Decoration, Op};
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let parsed = parse_bytes(&bytes).expect("parse");
    let insts = &parsed.instructions;
    let decorated: Vec<u32> = (insts.iter())
        .filter(|i| i.opcode == Op::Decorate)
        .filter_map(|i| match i.operands.as_slice() {
            [Operand::IdRef(id), Operand::Word(d), Operand::Word(v)]
                if *d == Decoration::BuiltIn as u32 && *v == b as u32 =>
            {
                Some(*id)
            }
            _ => None,
        })
        .collect();
    let reads_undef = |value: u32| {
        let mut seen = std::collections::BTreeSet::new();
        let mut stack = vec![value];
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            let Some(def) = insts.iter().find(|i| i.result_id == Some(id)) else {
                continue;
            };
            if def.opcode == Op::Undef {
                return true;
            }
            stack.extend(def.operands.iter().filter_map(|o| match o {
                Operand::IdRef(r) => Some(*r),
                _ => None,
            }));
        }
        false
    };
    (insts.iter())
        .filter(|i| i.opcode == Op::Store)
        .filter_map(|i| match i.operands.as_slice() {
            [Operand::IdRef(pointer), Operand::IdRef(value), ..] if decorated.contains(pointer) => {
                Some(reads_undef(*value))
            }
            _ => None,
        })
        .collect()
}

/// A fragment function that reads the slice or viewport no stage writes reads 0 on a device that
/// cannot declare the builtin, which it lists: below the floor, the viewport without
/// `multiViewport` and `shaderOutputViewportIndex`, and the slice without `shaderOutputLayer`,
/// each of which the floor requires.
#[test]
fn below_the_floor_a_read_of_an_index_no_stage_writes_is_zero() {
    let Some(amp) = assemble("vertex_amplification.ll") else {
        return;
    };
    for (entry, zero, floor) in [
        (
            "amp_fragment_layer",
            vec![StageBuiltin::Layer],
            device("runtime-floor", &[F::ShaderOutputLayer]),
        ),
        (
            "amp_fragment",
            vec![StageBuiltin::ViewportIndex],
            device(
                "runtime-floor",
                &[F::ShaderOutputViewportIndex, F::MultiViewport],
            ),
        ),
    ] {
        let a = compile(&amp, entry, floor.profile()).unwrap_or_else(|e| panic!("{entry}: {e}"));
        assert_eq!(a.interface.io.zero_indices, zero, "{entry}");
        assert!(!a
            .interface
            .io
            .builtins
            .iter()
            .any(|b| matches!(b, StageBuiltin::Layer | StageBuiltin::ViewportIndex)));
    }
}
