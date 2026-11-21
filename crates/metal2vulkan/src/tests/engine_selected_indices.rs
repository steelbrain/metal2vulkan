//! The render target slice and viewport a vertex function selects on a device that cannot store
//! `Layer` or `ViewportIndex` from a vertex function, over the owned fixtures `engine/rt6_layered`
//! (a vertex function that chooses its slice per instance, and fragment functions that read it,
//! one with the barycentric weights and the primitive ID) and `engine/rt6_viewports` (a vertex
//! function that selects its viewport per instance, and a fragment function that reads it):
//!
//! - Without `shaderOutputLayer`, on a device with `geometryShader`, the vertex function stores the
//!   slice it selects to a flat varying (`IoRole::SelectedLayer`) and declares no `Layer`; its
//!   pipeline carries the synthesized geometry stages of the point, line and triangle classes,
//!   which store `Layer` from the primitive's first vertex. The variant for a pass of one layer
//!   selects nothing and carries no stage.
//! - Without `shaderOutputViewportIndex`, on a device with `multiViewport` and `geometryShader`,
//!   the viewport alike (`IoRole::SelectedViewport`), the stages storing `ViewportIndex` through
//!   `MultiViewport`. Without `multiViewport` a draw has one viewport, so the vertex function
//!   stores no viewport index at all and the pipeline carries no stage.
//! - One set of stages serves the barycentric weights, the primitive ID and the slice together.
//! - A fragment function that reads an index no stage writes, on a device that cannot declare its
//!   builtin, reads 0 (`StageIo::zero_indices`).
//! - Without `geometryShader` either, the variant that selects a slice is refused needing either
//!   feature, and the one for a pass of one layer compiles.
//! - A vertex function whose slice and viewport index differ among the vertices of a primitive
//!   (`engine/rt6_first_vertex`) stores them through the stages behind the device's own outputs
//!   too, as Vulkan takes `Layer` and `ViewportIndex` only where every vertex of a primitive
//!   stores the same and Metal takes the first vertex's; MoltenVK, which hands them to Metal,
//!   stores them itself. Without `geometryShader` elsewhere it is refused needing it.
//!
//! Every module that compiles passes L3 (`spirv-val` and requirements within the profile). `llvm-as`
//! assembles the fixtures; a missing tool skips visibly. No GPU is used.

use std::path::Path;

use crate::budget::Budget;
use crate::cache::VariantKey;
use crate::device::requirements::{VkFeature as F, VkProperty};
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
use crate::vk;

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

/// The geometry stages of `p`, each valid on `dev`, every other stage too.
fn geometry_stages(p: &RenderPipeline, dev: &Device, what: &str) -> usize {
    for m in &p.stages {
        check_words_valid(&m.spirv, &p.requirements, dev.profile())
            .unwrap_or_else(|e| panic!("{what} {:?}: L3: {e}", m.stage));
    }
    p.stages
        .iter()
        .filter(|m| m.stage == vk::ShaderStage::GEOMETRY)
        .count()
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

fn any_of(alternatives: &[F]) -> Needs {
    Needs::AnyOf(
        alternatives
            .iter()
            .map(|&f| vec![Capability::Feature(f)])
            .collect(),
    )
}

/// Without `shaderOutputLayer` a vertex function selects its slice through the synthesized
/// geometry stages, and its variant for a pass of one layer through none.
#[test]
fn without_the_output_the_slice_goes_through_the_geometry_stage() {
    let Some(bc) = assemble("rt6_layered.ll") else {
        return;
    };
    let dev = device(
        "lavapipe",
        &[F::ShaderOutputLayer, F::ShaderOutputViewportIndex],
    );
    let (outputs, builtins) = vertex_io(&bc, "rt6_layered_vertex", &dev);
    assert!(outputs.contains(&IoRole::SelectedLayer), "{outputs:?}");
    assert!(
        outputs.contains(&IoRole::RenderTargetArrayIndex),
        "{outputs:?}"
    );
    assert!(!builtins.contains(&StageBuiltin::Layer), "{builtins:?}");
    let budget = Budget::standard();
    let d = render(
        &bc,
        "rt6_layered_vertex",
        "rt6_layered_fragment",
        PixelFormat::R32Uint,
    );
    let p = dev.compile_render_pipeline(&d, &budget).expect("pipeline");
    assert_eq!(geometry_stages(&p, &dev, "layered"), 3);
    assert!(p.graphics.declares_layer);
    assert!(!p.graphics.writes_viewport_index);
    assert!(!p.requirements.features.contains(&F::ShaderOutputLayer));
    assert!(p.requirements.features.contains(&F::GeometryShader));
    let mut key = VariantKey::new();
    key.dropped_vertex_layer = true;
    let one = dev
        .compile_render_variant(&d, &key, &budget)
        .expect("dropped variant");
    assert_eq!(geometry_stages(&one, &dev, "one layer"), 0);
    assert!(!one.requirements.features.contains(&F::GeometryShader));
    // Behind the device's own `Layer` output, no stage.
    let native = device("lavapipe", &[]);
    let p = native
        .compile_render_pipeline(&d, &budget)
        .expect("pipeline");
    assert_eq!(geometry_stages(&p, &native, "native"), 0);
    assert!(p.requirements.features.contains(&F::ShaderOutputLayer));
}

/// Without `shaderOutputViewportIndex`, a device with `multiViewport` selects the viewport through
/// the synthesized geometry stages, whose `ViewportIndex` needs `MultiViewport`; a device without
/// `multiViewport` draws through one viewport and stores none, its pipeline still stating that the
/// vertex function writes the index.
#[test]
fn without_the_output_the_viewport_goes_through_the_geometry_stage_or_nowhere() {
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
    let dev = device("lavapipe", &[F::ShaderOutputViewportIndex]);
    let (outputs, builtins) = vertex_io(&bc, "rt6_select", &dev);
    assert!(outputs.contains(&IoRole::SelectedViewport), "{outputs:?}");
    assert!(outputs.contains(&IoRole::ViewportIndex), "{outputs:?}");
    assert!(
        !builtins.contains(&StageBuiltin::ViewportIndex),
        "{builtins:?}"
    );
    let p = dev.compile_render_pipeline(&d, &budget).expect("pipeline");
    assert_eq!(geometry_stages(&p, &dev, "multiViewport"), 3);
    assert!(p.graphics.writes_viewport_index);
    assert!(!p.graphics.declares_layer);
    assert!(p.requirements.features.contains(&F::MultiViewport));
    assert!(!p
        .requirements
        .features
        .contains(&F::ShaderOutputViewportIndex));
    let mv = crate::device::SpirvCapability::from_raw(Cap::MultiViewport as u32);
    assert!(p.requirements.spirv_capabilities.contains(&mv));
    // Without `multiViewport`, on the floor too: the raw index alone.
    for dev in [
        device(
            "lavapipe",
            &[F::ShaderOutputViewportIndex, F::MultiViewport],
        ),
        device("runtime-floor", &[]),
    ] {
        let (outputs, builtins) = vertex_io(&bc, "rt6_select", &dev);
        assert_eq!(outputs, [IoRole::ViewportIndex], "{builtins:?}");
        assert!(!builtins.contains(&StageBuiltin::ViewportIndex));
        let p = dev.compile_render_pipeline(&d, &budget).expect("pipeline");
        assert_eq!(geometry_stages(&p, &dev, "one viewport"), 0);
        // The function still writes the index: the runtime refuses its draw while the encoder
        // holds more than one viewport.
        assert!(p.graphics.writes_viewport_index);
    }
}

/// One set of synthesized geometry stages carries the barycentric weights, the primitive ID and the
/// selected slice of a pipeline that needs the stages for each.
#[test]
fn one_geometry_stage_serves_the_weights_the_primitive_and_the_slice() {
    let Some(bc) = assemble("rt6_layered.ll") else {
        return;
    };
    let dev = device(
        "lavapipe",
        &[F::ShaderOutputLayer, F::FragmentShaderBarycentric],
    );
    let d = render(
        &bc,
        "rt6_layered_vertex",
        "rt6_layered_weights",
        PixelFormat::R32Uint,
    );
    let p = dev
        .compile_render_pipeline(&d, &Budget::standard())
        .expect("pipeline");
    assert_eq!(geometry_stages(&p, &dev, "weights and slice"), 3);
    assert!(p.graphics.reads_barycentric_coord);
    assert!(p.graphics.declares_layer);
    assert!(!p
        .requirements
        .features
        .contains(&F::FragmentShaderBarycentric));
    // With the device's own `Layer` output beside the weights' stage: not implemented.
    let mixed = device("lavapipe", &[F::FragmentShaderBarycentric]);
    match mixed.compile_render_pipeline(&d, &Budget::standard()) {
        Err(Error::Unsupported(u)) => assert_eq!(u.feature, Feature::Barycentrics),
        other => panic!("{other:?}"),
    }
}

/// A fragment function that reads the slice or viewport no stage writes reads 0 on a device that
/// cannot declare the builtin, which it lists; below the floor (`runtime-floor` without
/// `shaderOutputLayer`, which the floor requires) a vertex function that selects its slice is
/// refused needing either feature, and its variant for a pass of one layer compiles.
#[test]
fn below_the_floor_a_read_is_zero_and_a_selected_slice_needs_either_feature() {
    let Some(layered) = assemble("rt6_layered.ll") else {
        return;
    };
    let Some(amp) = assemble("vertex_amplification.ll") else {
        return;
    };
    let floor = device("runtime-floor", &[F::ShaderOutputLayer]);
    for (entry, zero) in [
        ("amp_fragment_layer", vec![StageBuiltin::Layer]),
        ("amp_fragment", vec![StageBuiltin::ViewportIndex]),
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
    let budget = Budget::standard();
    let d = render(
        &layered,
        "rt6_layered_vertex",
        "rt6_layered_fragment",
        PixelFormat::R32Uint,
    );
    match floor.compile_render_pipeline(&d, &budget) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::LayeredRendering);
            assert_eq!(u.needs, any_of(&[F::ShaderOutputLayer, F::GeometryShader]));
        }
        other => panic!("{other:?}"),
    }
    let mut key = VariantKey::new();
    key.dropped_vertex_layer = true;
    let one = floor
        .compile_render_variant(&d, &key, &budget)
        .expect("dropped variant");
    assert_eq!(
        geometry_stages(&one, &floor, "below the floor, one layer"),
        0
    );
}

/// A vertex function whose slice and viewport index differ among the vertices of a primitive
/// (`engine/rt6_first_vertex`) stores them through the synthesized geometry stages behind the
/// device's own outputs too, as Vulkan takes `Layer` and `ViewportIndex` only where every vertex
/// of a primitive stores the same; one whose slice is the same at every vertex
/// (`rt6_layered_vertex`) keeps the builtin. No module requires
/// `VkProperty::MetalFirstVertexIndices`. Without `geometryShader` (MoltenVK's, and lavapipe's with
/// the feature removed), the pipeline is refused needing `geometryShader`.
#[test]
fn indices_that_differ_among_a_primitive_s_vertices_go_through_the_geometry_stage() {
    let Some(layered) = assemble("rt6_layered.ll") else {
        return;
    };
    let Some(bc) = assemble("rt6_first_vertex.ll") else {
        return;
    };
    let dev = device("lavapipe", &[]);
    let uniform = |bc: &[u8], entry: &str, dev: &Device| {
        let a = compile(bc, entry, dev.profile()).unwrap_or_else(|e| panic!("{entry}: {e}"));
        a.interface.io.uniform_outputs.clone()
    };
    let alike = uniform(&layered, "rt6_layered_vertex", &dev);
    assert!(
        alike.contains(&VaryingKey::render_target_array_index()),
        "{alike:?}"
    );
    let (outputs, builtins) = vertex_io(&layered, "rt6_layered_vertex", &dev);
    assert!(builtins.contains(&StageBuiltin::Layer), "{builtins:?}");
    assert!(!outputs.contains(&IoRole::SelectedLayer), "{outputs:?}");
    let per_vertex = uniform(&bc, "rt6_first_vertex", &dev);
    for key in [
        VaryingKey::render_target_array_index(),
        VaryingKey::viewport_index(),
    ] {
        assert!(!per_vertex.contains(&key), "{per_vertex:?}");
    }
    let (outputs, builtins) = vertex_io(&bc, "rt6_first_vertex", &dev);
    assert!(outputs.contains(&IoRole::SelectedLayer), "{outputs:?}");
    assert!(outputs.contains(&IoRole::SelectedViewport), "{outputs:?}");
    assert!(
        !builtins
            .iter()
            .any(|b| matches!(b, StageBuiltin::Layer | StageBuiltin::ViewportIndex)),
        "{builtins:?}"
    );
    let budget = Budget::standard();
    let d = render(
        &bc,
        "rt6_first_vertex",
        "rt6_first_vertex_read",
        PixelFormat::R32Uint,
    );
    let p = dev.compile_render_pipeline(&d, &budget).expect("pipeline");
    assert_eq!(geometry_stages(&p, &dev, "per vertex"), 3);
    assert!(p.graphics.declares_layer);
    assert!(p.graphics.writes_viewport_index);
    assert!(p.requirements.features.contains(&F::GeometryShader));
    assert!(!p
        .requirements
        .properties
        .contains(&VkProperty::MetalFirstVertexIndices));
    for without in [
        device("moltenvk-m3", &[]),
        device("lavapipe", &[F::GeometryShader]),
    ] {
        match without.compile_render_pipeline(&d, &budget) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::LayeredRendering);
                assert_eq!(u.needs, any_of(&[F::GeometryShader]));
            }
            other => panic!("{other:?}"),
        }
    }
}
