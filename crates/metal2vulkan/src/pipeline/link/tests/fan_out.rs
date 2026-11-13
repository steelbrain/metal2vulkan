//! The fan-out variant's geometry stages (`link::fan_out`): its triangle class's stage emits as
//! many copies as the device's limits leave room for, a probe's none, each module valid on the
//! device; a probe pushes its flag's index; and the refusals.

use super::*;
use crate::error::Needs;
use crate::metal::PrimitiveTopologyClass as Class;
use crate::pipeline::link::geometry::{plan, GeometryStages, CLASSES};
use crate::pipeline::stage_io::StageBuiltin;
use crate::pipeline::RegionFanOut;

fn stage(stage: Stage, io: StageIo) -> FunctionArtifact {
    let mut i = iface(stage, vec![], vec![]);
    i.io = io;
    artifact(i, None)
}

/// A vertex function writing `a` (`float4`), its position and two clip distances, and a
/// fragment function reading `a` and the primitive ID.
fn pipeline() -> (FunctionArtifact, FunctionArtifact) {
    let f4 = io_ty(IoNumeric::Float, 4);
    let mut vio = stage_io(vec![], vec![varying("a", f4, None)]);
    vio.builtins = vec![
        StageBuiltin::Position { invariant: false },
        StageBuiltin::ClipDistance { count: 2 },
    ];
    let mut fio = stage_io(vec![varying("a", f4, SMOOTH)], vec![color(0, 0, f4)]);
    fio.builtins = vec![StageBuiltin::PrimitiveId];
    (stage(Stage::Vertex, vio), stage(Stage::Fragment, fio))
}

fn fanned(mode: RegionFanOut) -> LinkInputs {
    let mut r = RenderLinkInputs::new();
    r.color_attachments[0] = Some(crate::pipeline::descriptor::ColorAttachmentDescriptor::new(
        crate::metal::PixelFormat::RGBA32Float,
    ));
    r.input_primitive_topology = Class::Triangle;
    r.region_fan_out = Some(mode);
    LinkInputs::Render(Box::new(r))
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

/// A draw's triangle stage emits three vertices a copy, as many copies as its state says, and a
/// probe's none and pushes its flag's index in the geometry stage's own range; every class's
/// module is valid on the device.
#[test]
fn a_fan_out_variant_s_triangle_stage_emits_its_copies() {
    for stem in ["lavapipe", "intel-irisxe-rplp-anv", "nvidia-pascal-linux"] {
        let p = profile(stem);
        let (v, f) = pipeline();
        for mode in [RegionFanOut::Draw, RegionFanOut::Probe] {
            let linked =
                link(&[&v, &f], &fanned(mode), &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
            let g = linked.geometry.as_ref().expect("geometry stages");
            let fan = g.fan_out.expect("the fan-out stage");
            assert!((1..=16).contains(&fan.copies), "{stem}: {}", fan.copies);
            let state = linked
                .graphics
                .as_ref()
                .and_then(|g| g.region_fan_out)
                .expect("the fan-out state");
            assert_eq!(
                (state.mode, state.copies, state.push_offset),
                (mode, fan.copies, fan.push_offset),
                "{stem}"
            );
            let pushes = (linked.layout.push_constants.iter())
                .filter(|r| r.stages == vk::ShaderStageFlags::GEOMETRY)
                .map(|r| (r.offset, r.size))
                .collect::<Vec<_>>();
            match mode {
                RegionFanOut::Draw => assert!(pushes.is_empty(), "{stem}"),
                RegionFanOut::Probe => assert_eq!(pushes, [(fan.push_offset, 4)], "{stem}"),
            }
            let made = g
                .stages(&linked, &p, &mut crate::pipeline::testing::meter())
                .unwrap_or_else(|e| panic!("{stem} {mode:?}: {e}"));
            for ((module, requirements), class) in made.iter().zip(CLASSES) {
                crate::spirv::testing::check_words_valid(&module.spirv, requirements, &p)
                    .unwrap_or_else(|e| panic!("{stem} {mode:?} {class:?}: {e}"));
                let emits = count_ops(&module.spirv, crate::spv::Op::EmitVertex as u32);
                let expected = match (class, mode) {
                    (Class::Point, _) => 1,
                    (Class::Line, _) => 2,
                    (_, RegionFanOut::Draw) => 3 * fan.copies as usize,
                    (_, RegionFanOut::Probe) => 0,
                };
                assert_eq!(emits, expected, "{stem} {mode:?} {class:?}");
            }
        }
    }
}

fn not_implemented(e: Error) {
    match e {
        Error::Unsupported(u) => {
            assert_eq!(u.feature, Feature::RasterizationRateMap);
            assert_eq!(u.needs, Needs::NotImplemented);
        }
        other => panic!("{other:?}"),
    }
}

/// A fan-out beside another reason for the stages (a fragment function reading the weights where
/// the device reads none), of a mesh or post-tessellation vertex function, or of a pipeline that
/// amplifies its vertices is not implemented; a probe on a device without
/// `vertexPipelineStoresAndAtomics` needs it, where a draw takes the stages.
#[test]
fn what_the_fan_out_cannot_carry_is_refused() {
    let mut p = profile("lavapipe");
    p.edit(|x| {
        x.enabled
            .features
            .remove(&VkFeature::FragmentShaderBarycentric);
    });
    let (v, f) = pipeline();
    let run = |v: &FunctionArtifact,
               f: &FunctionArtifact,
               amplification,
               mesh,
               mode,
               p: &DeviceProfile| {
        let i = [v.interface.clone(), f.interface.clone()];
        let refs: Vec<&StageInterface> = i.iter().collect();
        plan(&refs, Class::Triangle, amplification, mesh, Some(mode), p)
    };
    for mode in [RegionFanOut::Draw, RegionFanOut::Probe] {
        let g: GeometryStages = run(&v, &f, 1, false, mode, &p).unwrap().unwrap();
        assert_eq!(g.fan_out.map(|s| s.mode), Some(mode));
        assert_eq!(g.clip_distances, 2);
        let mut weights = f.clone();
        weights.interface.io.inputs.push(IoVariable {
            key: VaryingKey::barycentric(true),
            role: IoRole::Barycentric { perspective: true },
            ty: io_ty(IoNumeric::Float, 2),
        });
        not_implemented(run(&v, &weights, 1, false, mode, &p).unwrap_err());
        not_implemented(run(&v, &f, 2, false, mode, &p).unwrap_err());
        not_implemented(run(&v, &f, 1, true, mode, &p).unwrap_err());
        let mut tess = v.clone();
        tess.interface.tessellation = Some(crate::pipeline::interface::TessellationStage::new(
            crate::pipeline::interface::PatchDomain::Triangle,
            3,
        ));
        not_implemented(run(&tess, &f, 1, false, mode, &p).unwrap_err());
    }
    let mut no_stores = p.clone();
    no_stores.edit(|x| {
        x.enabled
            .features
            .remove(&VkFeature::VertexPipelineStoresAndAtomics);
    });
    assert!(run(&v, &f, 1, false, RegionFanOut::Draw, &no_stores).is_ok());
    match run(&v, &f, 1, false, RegionFanOut::Probe, &no_stores).unwrap_err() {
        Error::Unsupported(u) => {
            assert_eq!(u.feature, Feature::RasterizationRateMap);
            assert_eq!(
                u.needs,
                Needs::AnyOf(vec![vec![Capability::Feature(
                    VkFeature::VertexPipelineStoresAndAtomics
                )]])
            );
        }
        other => panic!("{other:?}"),
    }
}
