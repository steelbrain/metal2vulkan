//! The synthesized geometry stages (`link::geometry`) of a fragment function that reads
//! `[[barycentric_coord]]` as varyings: what they pass through, where the weights live, each
//! class's module, and the typed refusals.

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
