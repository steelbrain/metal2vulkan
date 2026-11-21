//! The render target slice and viewport a vertex function selects, over the owned fixtures
//! `engine/rt6_layered` (a vertex function that chooses its slice per instance, and fragment
//! functions that read it, one with the barycentric weights and the primitive ID),
//! `engine/rt6_viewports` (a vertex function that selects its viewport per instance, and a
//! fragment function that reads it) and `engine/rt6_first_vertex` (a vertex function whose slice
//! and viewport differ among the vertices of a primitive):
//!
//! - An index written the same at every vertex of a primitive is stored to `Layer` or
//!   `ViewportIndex` where the device stores the builtin from a vertex function; without
//!   `shaderOutputLayer` (`shaderOutputViewportIndex`) it is refused needing that feature alone.
//! - Without `multiViewport` a draw has one viewport, so the vertex function stores no viewport
//!   index at all.
//! - An index not shown the same at every vertex of a primitive is refused as not implemented on
//!   every device: Metal leaves its primitive undefined and Vulkan asks every vertex to write the
//!   same.
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
use crate::pipeline::sym::VaryingKey;
use crate::pipeline::testing::{compile, engine_dir, llvm_assemble, AssembleError};
use crate::pipeline::RenderPipeline;
use crate::spirv::testing::check_words_valid;
use crate::spv::Capability as Cap;

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
    Device::from_queries(desc, DevicePolicy::default()).expect("device")
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

/// A vertex function whose slice and viewport differ among the vertices of a primitive
/// (`engine/rt6_first_vertex`) writes neither alike, as its structure shows, where one whose slice
/// is the same at every vertex (`rt6_layered_vertex`) writes it alike; the first is refused as not
/// implemented on every device, with the outputs, with `geometryShader` and without it.
#[test]
fn indices_that_differ_among_a_primitive_s_vertices_are_refused() {
    let Some(layered_bc) = assemble("rt6_layered.ll") else {
        return;
    };
    let Some(bc) = assemble("rt6_first_vertex.ll") else {
        return;
    };
    let dev = device("lavapipe", &[]);
    let uniform = |bc: &[u8], entry: &str| {
        let a = compile(bc, entry, dev.profile()).unwrap_or_else(|e| panic!("{entry}: {e}"));
        a.interface.io.uniform_outputs.clone()
    };
    let alike = uniform(&layered_bc, "rt6_layered_vertex");
    assert!(
        alike.contains(&VaryingKey::render_target_array_index()),
        "{alike:?}"
    );
    let d = render(
        &bc,
        "rt6_first_vertex",
        "rt6_first_vertex_read",
        PixelFormat::R32Uint,
    );
    for (stem, dev) in [
        ("lavapipe", device("lavapipe", &[])),
        ("moltenvk-m3", device("moltenvk-m3", &[])),
        (
            "lavapipe without geometryShader",
            device("lavapipe", &[F::GeometryShader]),
        ),
        ("runtime-floor", device("runtime-floor", &[])),
    ] {
        for entry in ["rt6_first_vertex", "rt6_first_vertex_point"] {
            match compile(&bc, entry, dev.profile()) {
                Err(Error::Unsupported(u)) => {
                    assert_eq!(u.feature, Feature::LayeredRendering, "{stem} {entry}");
                    assert_eq!(u.needs, Needs::NotImplemented, "{stem} {entry}");
                }
                other => panic!("{stem} {entry}: {:?}", other.map(|_| ())),
            }
        }
        layered(
            dev.compile_render_pipeline(&d, &Budget::standard()),
            Needs::NotImplemented,
            stem,
        );
    }
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
