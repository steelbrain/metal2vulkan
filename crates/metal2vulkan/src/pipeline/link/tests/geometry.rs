//! The synthesized geometry stages (`link::geometry`) of a fragment function that reads
//! `[[barycentric_coord]]` as varyings and of a vertex function that selects its slice or viewport
//! through them: what they pass through, where the weights live, what they store, each class's
//! module, and the typed refusals.

use super::*;
use crate::error::Needs;
use crate::metal::PrimitiveTopologyClass as Class;
use crate::pipeline::link::geometry::{plan, GeometryStages, CLASSES};
use crate::pipeline::stage_io::StageBuiltin;

/// The weights' `float2` input of form `perspective`.
fn weights(perspective: bool) -> IoVariable {
    IoVariable {
        key: VaryingKey::barycentric(perspective),
        role: IoRole::Barycentric { perspective },
        ty: io_ty(IoNumeric::Float, 2),
    }
}

fn stage(stage: Stage, io: StageIo) -> FunctionArtifact {
    let mut i = iface(stage, vec![], vec![]);
    i.io = io;
    artifact(i, None)
}

/// A vertex function writing `a` (`float4`) and `t` (`uint`) and its position, and a fragment
/// function reading `a`, the perspective weights and the primitive ID.
fn pipeline() -> (FunctionArtifact, FunctionArtifact) {
    let f4 = io_ty(IoNumeric::Float, 4);
    let u1 = io_ty(IoNumeric::Uint, 1);
    let mut vio = stage_io(vec![], vec![varying("a", f4, None), varying("t", u1, None)]);
    vio.builtins = vec![StageBuiltin::Position { invariant: false }];
    let mut fio = stage_io(
        vec![varying("a", f4, SMOOTH), weights(true)],
        vec![color(0, 0, f4)],
    );
    fio.builtins = vec![StageBuiltin::PrimitiveId];
    (stage(Stage::Vertex, vio), stage(Stage::Fragment, fio))
}

fn render() -> LinkInputs {
    let mut r = RenderLinkInputs::new();
    r.color_attachments[0] = Some(crate::pipeline::descriptor::ColorAttachmentDescriptor::new(
        crate::metal::PixelFormat::RGBA32Float,
    ));
    LinkInputs::Render(Box::new(r))
}

/// The geometry stages pass through the vertex outputs the fragment stage reads, and the weights
/// take whole locations after every vertex output's; each class's module is valid on the device.
#[test]
fn the_weights_follow_the_vertex_outputs_and_each_class_has_its_stage() {
    for stem in ["lavapipe", "intel-irisxe-rplp-anv", "nvidia-pascal-linux"] {
        let p = profile(stem);
        let (v, f) = pipeline();
        let linked = link(&[&v, &f], &render(), &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let b = linked.geometry.as_ref().expect("geometry stages");
        assert_eq!(
            b.varyings,
            [(key("a"), io_ty(IoNumeric::Float, 4))],
            "{stem}"
        );
        assert_eq!(b.forms, [true], "{stem}");
        assert!(b.primitive_id && !b.multiview, "{stem}");
        assert_eq!(linked.table.locations[&key("a")], (0, 0), "{stem}");
        assert_eq!(
            linked.table.locations[&VaryingKey::barycentric(true)],
            (2, 0),
            "{stem}"
        );
        assert!(
            linked
                .graphics
                .as_ref()
                .expect("graphics")
                .reads_barycentric_coord,
            "{stem}"
        );
        let made = b
            .stages(&linked, &p, &mut crate::pipeline::testing::meter())
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert_eq!(made.len(), CLASSES.len());
        for ((module, requirements), (class, emitted)) in
            made.iter().zip(CLASSES.iter().zip([1, 2, 3]))
        {
            assert_eq!(module.stage, vk::ShaderStage::GEOMETRY, "{stem} {class:?}");
            assert!(requirements.features.contains(&VkFeature::GeometryShader));
            crate::spirv::testing::check_words_valid(&module.spirv, requirements, &p)
                .unwrap_or_else(|e| panic!("{stem} {class:?}: {e}"));
            let emits = count_ops(&module.spirv, crate::spv::Op::EmitVertex as u32);
            assert_eq!(emits, emitted, "{stem} {class:?}");
            assert_eq!(
                count_ops(&module.spirv, crate::spv::Op::EndPrimitive as u32),
                1,
                "{stem} {class:?}"
            );
        }
    }
}

/// How many instructions of `words` have opcode `op`.
fn count_ops(words: &[u32], op: u32) -> usize {
    let mut i = 5;
    let mut n = 0;
    while let Some(&w) = words.get(i) {
        let len = (w >> 16) as usize;
        n += usize::from(w & 0xFFFF == op);
        if len == 0 {
            break;
        }
        i += len;
    }
    n
}

fn refused_needing(e: Error, capability: Capability) {
    match e {
        Error::Unsupported(u) => {
            assert_eq!(u.feature, Feature::Barycentrics);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![
                    vec![Capability::Feature(VkFeature::FragmentShaderBarycentric)],
                    vec![capability],
                ])
            );
        }
        other => panic!("{other:?}"),
    }
}

fn not_implemented(e: Error) {
    match e {
        Error::Unsupported(u) => {
            assert_eq!(u.feature, Feature::Barycentrics);
            assert_eq!(u.needs, Needs::NotImplemented);
        }
        other => panic!("{other:?}"),
    }
}

/// What the geometry stages cannot carry is refused typed: a mesh pipeline, a post-tessellation
/// vertex function and a vertex function's own layer or viewport as not implemented; multiview
/// without `multiviewGeometryShader`; a vertex function's point size in the point class without
/// `shaderTessellationAndGeometryPointSize`; and the geometry limits. A fragment function that
/// reads no weights has no stages.
#[test]
fn what_a_geometry_stage_cannot_carry_is_refused() {
    let p = profile("lavapipe");
    let (v, f) = pipeline();
    let ifaces = |v: &FunctionArtifact, f: &FunctionArtifact| -> Vec<StageInterface> {
        vec![v.interface.clone(), f.interface.clone()]
    };
    let run = |v: &FunctionArtifact,
               f: &FunctionArtifact,
               class,
               amplification,
               mesh,
               p: &DeviceProfile| {
        let i = ifaces(v, f);
        let refs: Vec<&StageInterface> = i.iter().collect();
        plan(&refs, class, amplification, mesh, None, p)
    };
    assert!(run(&v, &f, Class::Triangle, 1, false, &p)
        .unwrap()
        .is_some());
    not_implemented(run(&v, &f, Class::Triangle, 1, true, &p).unwrap_err());
    for builtin in [StageBuiltin::Layer, StageBuiltin::ViewportIndex] {
        let mut w = v.clone();
        w.interface.io.builtins.push(builtin);
        not_implemented(run(&w, &f, Class::Triangle, 1, false, &p).unwrap_err());
    }
    let mut tess = v.clone();
    tess.interface.tessellation = Some(crate::pipeline::interface::TessellationStage::new(
        crate::pipeline::interface::PatchDomain::Triangle,
        3,
    ));
    not_implemented(run(&tess, &f, Class::Triangle, 1, false, &p).unwrap_err());
    // Multiview.
    let mut lacking = p.clone();
    lacking.edit(|x| {
        x.enabled
            .features
            .remove(&VkFeature::MultiviewGeometryShader);
    });
    let views = p.fields().limits.max_multiview_view_count;
    assert!(views > 1);
    let b = run(&v, &f, Class::Triangle, 2, false, &p).unwrap().unwrap();
    assert!(b.multiview);
    refused_needing(
        run(&v, &f, Class::Triangle, 2, false, &lacking).unwrap_err(),
        Capability::Feature(VkFeature::MultiviewGeometryShader),
    );
    assert!(run(&v, &f, Class::Triangle, 1, false, &lacking).is_ok());
    // A written point size, in the classes that draw points.
    let mut sized = v.clone();
    sized
        .interface
        .io
        .builtins
        .push(StageBuiltin::PointSize { default: false });
    let mut no_size = p.clone();
    no_size.edit(|x| {
        x.enabled
            .features
            .remove(&VkFeature::ShaderTessellationAndGeometryPointSize);
    });
    for class in [Class::Point, Class::Unspecified] {
        refused_needing(
            run(&sized, &f, class, 1, false, &no_size).unwrap_err(),
            Capability::Feature(VkFeature::ShaderTessellationAndGeometryPointSize),
        );
        let b: GeometryStages = run(&sized, &f, class, 1, false, &p).unwrap().unwrap();
        assert!(b.point_size && b.writes_point_size);
    }
    let b = run(&sized, &f, Class::Triangle, 1, false, &no_size)
        .unwrap()
        .unwrap();
    assert!(b.point_size && !b.writes_point_size);
    // No weights, no stages.
    let mut plain = f.clone();
    plain.interface.io.inputs.retain(|x| x.key == key("a"));
    assert!(run(&v, &plain, Class::Triangle, 1, true, &p)
        .unwrap()
        .is_none());
    // The limits, in whole locations.
    let mut narrow = p.clone();
    narrow.edit(|x| x.limits.max_geometry_output_components = 8);
    let linked = link(&[&v, &f], &render(), &narrow).unwrap_err();
    refused_needing(
        linked,
        Capability::Limit {
            limit: VkLimit::MaxGeometryOutputComponents,
            required: 12,
            available: 8,
        },
    );
}

/// The flat `uint` vertex output that carries the selected slice (`IoRole::SelectedLayer`) or
/// viewport (`IoRole::SelectedViewport`) to the geometry stages.
fn selected(role: IoRole) -> IoVariable {
    let key = match role {
        IoRole::SelectedLayer => VaryingKey::selected_layer(),
        _ => VaryingKey::selected_viewport(),
    };
    IoVariable {
        key,
        role,
        ty: io_ty(IoNumeric::Uint, 1),
    }
}

/// [`pipeline`]'s fragment function without the weights and the primitive ID.
fn plain(f: &FunctionArtifact) -> FunctionArtifact {
    let mut plain = f.clone();
    plain.interface.io.inputs.retain(|x| x.key == key("a"));
    plain.interface.io.builtins.clear();
    plain
}

/// The stages [`plan`] makes for `v` and `f`.
fn planned(
    v: &FunctionArtifact,
    f: &FunctionArtifact,
    amplification: u32,
    mesh: bool,
    p: &DeviceProfile,
) -> Result<Option<GeometryStages>, Error> {
    plan(
        &[&v.interface, &f.interface],
        Class::Triangle,
        amplification,
        mesh,
        None,
        p,
    )
}

/// Whether `words` decorate a variable as builtin `builtin`.
fn decorates_builtin(words: &[u32], builtin: crate::spv::BuiltIn) -> bool {
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let len = (w >> 16) as usize;
        if w & 0xFFFF == crate::spv::Op::Decorate as u32
            && words.get(i + 2) == Some(&(crate::spv::Decoration::BuiltIn as u32))
            && words.get(i + 3) == Some(&(builtin as u32))
        {
            return true;
        }
        if len == 0 {
            break;
        }
        i += len;
    }
    false
}

/// A vertex function's selected slice and viewport take the stages alone, behind a fragment
/// function that reads neither the weights nor the primitive ID: each class's module stores
/// `Layer` and `ViewportIndex` (that one through `MultiViewport`) and is valid on a device without
/// the outputs; the pipeline states both. Behind a fragment function that reads the weights and
/// the primitive ID, one set of stages serves all of them.
#[test]
fn a_selected_slice_and_viewport_take_the_stages() {
    let mut p = profile("lavapipe");
    for f in [
        VkFeature::ShaderOutputLayer,
        VkFeature::ShaderOutputViewportIndex,
    ] {
        p.edit(|x| {
            x.enabled.features.remove(&f);
        });
    }
    let (mut v, f) = pipeline();
    v.interface.io.outputs.push(selected(IoRole::SelectedLayer));
    v.interface
        .io
        .outputs
        .push(selected(IoRole::SelectedViewport));
    let linked = link(&[&v, &plain(&f)], &render(), &p).expect("indices alone");
    let b = linked.geometry.as_ref().expect("geometry stages");
    assert!(b.layer && b.viewport);
    assert!(b.forms.is_empty() && !b.primitive_id && !b.multiview);
    assert_eq!(b.varyings, [(key("a"), io_ty(IoNumeric::Float, 4))]);
    let g = linked.graphics.as_ref().expect("graphics");
    assert!(g.declares_layer && g.writes_viewport_index);
    let made = b
        .stages(&linked, &p, &mut crate::pipeline::testing::meter())
        .expect("stages");
    assert_eq!(made.len(), CLASSES.len());
    let mv = crate::device::SpirvCapability::from_raw(crate::spv::Capability::MultiViewport as u32);
    for ((module, requirements), class) in made.iter().zip(CLASSES) {
        crate::spirv::testing::check_words_valid(&module.spirv, requirements, &p)
            .unwrap_or_else(|e| panic!("{class:?}: {e}"));
        for builtin in [
            crate::spv::BuiltIn::Layer,
            crate::spv::BuiltIn::ViewportIndex,
        ] {
            assert!(
                decorates_builtin(&module.spirv, builtin),
                "{class:?}: {builtin:?}"
            );
        }
        assert!(requirements.spirv_capabilities.contains(&mv), "{class:?}");
        assert!(requirements.features.contains(&VkFeature::MultiViewport));
        assert!(!requirements
            .features
            .contains(&VkFeature::ShaderOutputLayer));
    }
    let linked = link(&[&v, &f], &render(), &p).expect("with the weights");
    let b = linked.geometry.as_ref().expect("geometry stages");
    assert!(b.layer && b.viewport && b.primitive_id);
    assert_eq!(b.forms, [true]);
}

/// What the stages cannot carry for a selected index is refused typed, naming
/// `LayeredRendering`: a post-tessellation vertex function and a vertex function's own
/// `ViewportIndex` beside a selected slice as not implemented, and a selected viewport drawn as
/// multiview without `multiviewGeometryShader` as needing the output instead. A selected slice is
/// never drawn as multiview.
#[test]
fn what_the_stages_cannot_carry_for_an_index_is_refused() {
    let p = profile("lavapipe");
    let (v, f) = pipeline();
    let f = plain(&f);
    let layered = |e: Error, needs: Needs| match e {
        Error::Unsupported(u) => {
            assert_eq!(u.feature, Feature::LayeredRendering);
            assert_eq!(u.needs, needs);
        }
        other => panic!("{other:?}"),
    };
    let mut slice = v.clone();
    slice
        .interface
        .io
        .outputs
        .push(selected(IoRole::SelectedLayer));
    assert!(planned(&slice, &f, 1, false, &p).unwrap().is_some());
    let mut tess = slice.clone();
    tess.interface.tessellation = Some(crate::pipeline::interface::TessellationStage::new(
        crate::pipeline::interface::PatchDomain::Triangle,
        3,
    ));
    layered(
        planned(&tess, &f, 1, false, &p).unwrap_err(),
        Needs::NotImplemented,
    );
    let mut both = slice.clone();
    both.interface.io.builtins.push(StageBuiltin::ViewportIndex);
    layered(
        planned(&both, &f, 1, false, &p).unwrap_err(),
        Needs::NotImplemented,
    );
    assert!(planned(&slice, &f, 1, true, &p).unwrap().is_none());
    let mut lacking = p.clone();
    lacking.edit(|x| {
        x.enabled
            .features
            .remove(&VkFeature::MultiviewGeometryShader);
    });
    let b = planned(&slice, &f, 2, false, &lacking).unwrap().unwrap();
    assert!(!b.multiview);
    let mut viewport = v.clone();
    viewport
        .interface
        .io
        .outputs
        .push(selected(IoRole::SelectedViewport));
    assert!(
        planned(&viewport, &f, 2, false, &p)
            .unwrap()
            .unwrap()
            .multiview
    );
    layered(
        planned(&viewport, &f, 2, false, &lacking).unwrap_err(),
        Needs::AnyOf(vec![
            vec![Capability::Feature(VkFeature::ShaderOutputViewportIndex)],
            vec![Capability::Feature(VkFeature::MultiviewGeometryShader)],
        ]),
    );
}
