//! The fan-out variant's geometry stages (`link::fan_out`): its triangle class's stage emits as
//! many copies as the device's limits leave room for, a probe's none, each module valid on the
//! device; a probe pushes its flag's index and adds a triangle's copies into the count its
//! primitive's index picks; and the refusals.

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

/// A draw's triangle stage emits three vertices a copy, in a loop of as many copies as its state
/// says, its output vertices three a copy, its copies' arithmetic `NoContraction`, and a probe's
/// none and pushes its flag's index in the geometry stage's own range; its `Position` and
/// `ClipDistance` are `Invariant`; every class's module is valid on the device.
#[test]
fn a_fan_out_variant_s_triangle_stage_emits_its_copies() {
    for stem in ["lavapipe", "intel-irisxe-rplp-anv", "nvidia-pascal-linux"] {
        let p = profile(stem);
        let (v, f) = pipeline();
        for mode in [
            RegionFanOut::Draw,
            RegionFanOut::Probe,
            RegionFanOut::DrawFew,
        ] {
            let linked =
                link(&[&v, &f], &fanned(mode), &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
            let g = linked.geometry.as_ref().expect("geometry stages");
            let fan = g.fan_out.expect("the fan-out stage");
            let most = match mode {
                RegionFanOut::DrawFew => crate::bind::fan_out::FEW_COPIES,
                _ => 16,
            };
            assert!((1..=most).contains(&fan.copies), "{stem}: {}", fan.copies);
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
                RegionFanOut::Probe => assert_eq!(pushes, [(fan.push_offset, 4)], "{stem}"),
                _ => assert!(pushes.is_empty(), "{stem} {mode:?}"),
            }
            let made = g
                .stages(&linked, &p, &mut crate::pipeline::testing::meter())
                .unwrap_or_else(|e| panic!("{stem} {mode:?}: {e}"));
            for ((module, requirements), class) in made.iter().zip(CLASSES) {
                crate::spirv::testing::check_words_valid(&module.spirv, requirements, &p)
                    .unwrap_or_else(|e| panic!("{stem} {mode:?} {class:?}: {e}"));
                let emits = count_ops(&module.spirv, crate::spv::Op::EmitVertex as u32);
                let (expected, outputs) = match (class, mode) {
                    (Class::Point, _) => (1, 1),
                    (Class::Line, _) => (2, 2),
                    (_, RegionFanOut::Probe) => (0, 3),
                    (_, _) => (3, 3 * fan.copies),
                };
                assert_eq!(emits, expected, "{stem} {mode:?} {class:?}");
                let loops = count_ops(&module.spirv, crate::spv::Op::LoopMerge as u32);
                let looped = class == Class::Triangle && mode != RegionFanOut::Probe;
                assert_eq!(loops, usize::from(looped), "{stem} {mode:?} {class:?}");
                let declared = (instructions(&module.spirv, crate::spv::Op::ExecutionMode))
                    .into_iter()
                    .filter(|m| {
                        m.get(1) == Some(&(crate::spv::ExecutionMode::OutputVertices as u32))
                    })
                    .map(|m| m[2])
                    .collect::<Vec<u32>>();
                assert_eq!(declared, [outputs], "{stem} {mode:?} {class:?}");
                if class != Class::Triangle {
                    continue;
                }
                let decorated = |decoration: crate::spv::Decoration| {
                    (instructions(&module.spirv, crate::spv::Op::Decorate).into_iter())
                        .filter(|d| d.get(1) == Some(&(decoration as u32)))
                        .count()
                };
                // Of each of a copy's three vertices, each axis's coordinate and its two clip
                // distances: five products and three sums.
                let contracted = match mode {
                    RegionFanOut::Probe => 0,
                    _ => 3 * 2 * 8,
                };
                let no_contraction = decorated(crate::spv::Decoration::NoContraction);
                assert_eq!(no_contraction, contracted, "{stem} {mode:?}");
                let invariant = decorated(crate::spv::Decoration::Invariant);
                assert_eq!(invariant, 2, "{stem} {mode:?}");
            }
        }
    }
}

/// The instructions of `words` with opcode `op`, each its operand words after the opcode's.
fn instructions(words: &[u32], op: crate::spv::Op) -> Vec<&[u32]> {
    let mut i = 5;
    let mut found = Vec::new();
    while let Some(&w) = words.get(i) {
        let len = (w >> 16) as usize;
        if len == 0 {
            break;
        }
        if w & 0xFFFF == op as u32 {
            found.extend(words.get(i + 1..i + len));
        }
        i += len;
    }
    found
}

/// A probe's triangle stage adds each triangle's copies once, into the count of its draw's
/// [`PROBE_COUNT_WORDS`](crate::bind::fan_out::PROBE_COUNT_WORDS) that its `PrimitiveId` picks
/// (masked by their number less one), whether or not the fragment function reads the primitive's
/// index: no one word takes every triangle's add. A flagged triangle raises its draw's first's and
/// last's words once each, adds one into its flagged count once, and sets its block's bit once,
/// the block its primitive's index shifted by the block's triangles picks.
#[test]
fn a_probe_adds_a_triangle_s_copies_into_the_count_its_primitive_picks() {
    use crate::spv::{BuiltIn, Decoration, Op};
    let p = profile("nvidia-pascal-linux");
    let f4 = io_ty(IoNumeric::Float, 4);
    let (v, reads_id) = pipeline();
    let fio = stage_io(vec![varying("a", f4, SMOOTH)], vec![color(0, 0, f4)]);
    let reads_none = stage(Stage::Fragment, fio);
    for (f, reads) in [(&reads_id, true), (&reads_none, false)] {
        let linked = link(&[&v, f], &fanned(RegionFanOut::Probe), &p).expect("linked");
        let g = linked.geometry.as_ref().expect("geometry stages");
        let made = (g.stages(&linked, &p, &mut crate::pipeline::testing::meter())).expect("made");
        let ((module, _), _) = (made.iter().zip(CLASSES))
            .find(|(_, class)| *class == Class::Triangle)
            .expect("the triangle stage");
        let words = &module.spirv;
        // The copies' add, and a flagged triangle's count.
        assert_eq!(instructions(words, Op::AtomicIAdd).len(), 2, "two adds");
        // The flag's store, and the wide word's.
        assert_eq!(instructions(words, Op::AtomicStore).len(), 2, "two stores");
        // A flagged triangle's raise of the first's and the last's words.
        assert_eq!(instructions(words, Op::AtomicUMax).len(), 2, "two raises");
        // A flagged triangle's block bit.
        assert_eq!(instructions(words, Op::AtomicOr).len(), 1, "one block bit");
        let shift = crate::bind::fan_out::BLOCK_TRIANGLES.ilog2();
        let shifts: Vec<u32> = (instructions(words, Op::Constant).into_iter())
            .filter(|c| c.get(2) == Some(&shift))
            .map(|c| c[1])
            .collect();
        let blocked = (instructions(words, Op::ShiftRightLogical).into_iter())
            .any(|a| a[3..].iter().any(|operand| shifts.contains(operand)));
        assert!(blocked, "the primitive's index shifted to its block");
        let primitive = (instructions(words, Op::Decorate).into_iter())
            .filter(|d| d[1..] == [Decoration::BuiltIn as u32, BuiltIn::PrimitiveId as u32])
            .count();
        // Its input, and the output it passes on where the fragment function reads it.
        assert_eq!(
            primitive,
            1 + usize::from(reads),
            "the primitive's index read"
        );
        let mask = crate::bind::fan_out::PROBE_COUNT_WORDS - 1;
        let masks: Vec<u32> = (instructions(words, Op::Constant).into_iter())
            .filter(|c| c.get(2) == Some(&mask))
            .map(|c| c[1])
            .collect();
        let masked = (instructions(words, Op::BitwiseAnd).into_iter())
            .any(|a| a[2..].iter().any(|operand| masks.contains(operand)));
        assert!(masked, "the primitive's index masked to the counts");
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

/// A region-copy variant draws no geometry stage: its vertex function draws its region's copy, at
/// the first region it reads pushed (`PushField::RegionFirst`), which its graphics state names; a
/// vertex function that reads none is refused.
#[test]
fn a_regions_variant_draws_its_vertex_function_s_copies() {
    let p = profile("lavapipe");
    let (mut v, f) = pipeline();
    let regions = fanned(RegionFanOut::RegionCopies);
    not_implemented(link(&[&v, &f], &regions, &p).unwrap_err());
    let fields = BTreeSet::from([PushField::RegionFirst]);
    v.interface.push = crate::lower::push::place(Stage::Vertex, &fields);
    let offset = match v.interface.push.fields.as_slice() {
        [placed] => match placed.loc {
            crate::pipeline::interface::PushLoc::Push { offset } => offset,
            other => panic!("{other:?}"),
        },
        other => panic!("{other:?}"),
    };
    let linked = link(&[&v, &f], &regions, &p).unwrap_or_else(|e| panic!("{e}"));
    assert!(linked.geometry.is_none());
    let fan = linked.graphics.and_then(|g| g.region_fan_out);
    assert_eq!(
        fan,
        Some(crate::pipeline::graphics::FanOutState {
            mode: RegionFanOut::RegionCopies,
            copies: 0,
            push_offset: offset,
        })
    );
}
