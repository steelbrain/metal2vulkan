//! The stage-IO tail: `bool` varyings, the fragment builtins whose enabling capability has
//! alternatives, and barycentric coordinates.
//!
//! The owned fixture `engine/stage_io_tail` goes through the per-function pipeline and link:
//!
//! - A `bool` varying (scalar and vector) is a flat `uint` in the interface, written as `select(b,
//!   1, 0)` and read as `x != 0`; the vertex and fragment entries link as one render pipeline.
//! - A fragment `PrimitiveId`, `Layer`, or `ViewportIndex` input is declared through the first of
//!   its enabling capabilities the profile enables (`Geometry` or `Tessellation`; `Geometry` or
//!   `ShaderLayer`; `MultiViewport` or `ShaderViewportIndex`), and refused as needing any one of
//!   their features on a profile with none.
//! - `[[barycentric_coord]]` reads `BaryCoordKHR` or `BaryCoordNoPerspKHR`, its leading lanes for a
//!   `float` or `float2`, and for a `float3` its first two lanes and `1 - (x + y)` of them as the
//!   third, Metal's, on a profile with `fragmentShaderBarycentric`, and is refused needing the
//!   feature on a profile without it; the linked graphics state states whether the fragment
//!   function reads it.
//!
//! Every module that compiles passes L3 (`spirv-val` and requirements within the profile) under
//! the stage-only numbering. `llvm-as` assembles the fixture; a missing tool skips visibly. No
//! GPU is used.

use crate::device::requirements::VkFeature as F;
use crate::device::{DevicePolicy, DeviceProfile};
use crate::error::{Capability, Error, Feature, Needs};
use crate::library::FunctionArtifact;
use crate::library::FunctionConstantValues;
use crate::metal::PixelFormat;
use crate::pipeline::compile::{plan_bitcode, EntryRef};
use crate::pipeline::descriptor::ColorAttachmentDescriptor;
use crate::pipeline::link::{link, LinkInputs, RenderLinkInputs};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::relocate::relocate;
use crate::pipeline::stage_io::{Interpolation, IoNumeric, IoRole, IoType, StageBuiltin};
use crate::pipeline::testing::{
    compile, compile_kept, compile_with, engine_dir, fragment_inputs, llvm_assemble, meter,
    profile, profile_without, AssembleError,
};
use crate::spirv::ir::{BinOp, ConstKind, Constant, Contraction, Inst, TypedModule, ValueOrigin};
use crate::spirv::testing::check_words_valid;

use crate::device::SpirvCapability;
use crate::spv::Capability as Cap;

/// A grammar capability as the API's raw-value `SpirvCapability`.
fn spirv_cap(c: Cap) -> SpirvCapability {
    SpirvCapability::from_raw(c as u32)
}

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("stage_io_tail.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected stage_io_tail.ll: {why}"),
    }
}

fn prof(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::default()).expect("committed profile")
}

/// Link `a` alone, relocate it, and run L3; the relocated words.
fn validate(a: &FunctionArtifact, p: &DeviceProfile) -> Vec<u32> {
    let t = link(&[a], &LinkInputs::standalone(), p)
        .unwrap_or_else(|e| panic!("link: {e}"))
        .table;
    let words = relocate(a, &t).expect("relocate");
    check_words_valid(&words, &a.requirements, p).unwrap_or_else(|e| panic!("L3: {e}"));
    words
}

/// Whether `words` hold an instruction with opcode `op`.
fn has_op(words: &[u32], op: u32) -> bool {
    let mut i = 5;
    while let Some(&w) = words.get(i) {
        let len = (w >> 16) as usize;
        if w & 0xFFFF == op {
            return true;
        }
        if len == 0 {
            return false;
        }
        i += len;
    }
    false
}

const OP_SELECT: u32 = 169;
const OP_I_NOT_EQUAL: u32 = 171;

fn refused(a: Result<std::sync::Arc<FunctionArtifact>, Error>, entry: &str) -> (Feature, Needs) {
    match a {
        Err(Error::Unsupported(u)) => (u.feature, u.needs),
        other => panic!("{entry}: {other:?}"),
    }
}

fn any_of(features: &[F]) -> Needs {
    Needs::AnyOf(
        features
            .iter()
            .map(|f| vec![Capability::Feature(*f)])
            .collect(),
    )
}

/// `bool` varyings are flat `uint`s, converted at both ends, and the stages link.
#[test]
fn bool_varyings_are_flat_uints_converted_at_both_ends() {
    let Some(bc) = fixture() else { return };
    // (numeric, components, AIR bits) of a `bool` varying of `n` lanes.
    let bool_ty = |n: u8| (IoNumeric::Uint, n, 1u8);
    let parts = |t: &IoType| (t.numeric, t.components, t.source_bits);
    for stem in ["moltenvk-m3", "lavapipe", "runtime-floor"] {
        let p = prof(stem);
        let v = compile(&bc, "tail_vertex_bool", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let mut r = RenderLinkInputs::new();
        r.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        let f = compile_with(&bc, "tail_fragment_bool", &p, &fragment_inputs(&r, &p))
            .unwrap_or_else(|e| panic!("{stem}: {e}"));
        let out: Vec<(IoNumeric, u8, u8)> = v
            .interface
            .io
            .outputs
            .iter()
            .map(|o| parts(&o.ty))
            .collect();
        assert_eq!(
            out[..2],
            [bool_ty(1), bool_ty(2)],
            "{stem}: lit, mask ({out:?})"
        );
        let inputs: Vec<((IoNumeric, u8, u8), Option<Interpolation>)> = f
            .interface
            .io
            .inputs
            .iter()
            .map(|x| match &x.role {
                IoRole::Varying { interpolation, .. } => (parts(&x.ty), *interpolation),
                r => panic!("{stem}: fragment input role {r:?}"),
            })
            .collect();
        assert_eq!(
            inputs[..2],
            [
                (bool_ty(1), Some(Interpolation::Flat)),
                (bool_ty(2), Some(Interpolation::Flat)),
            ],
            "{stem}"
        );
        assert!(
            has_op(&validate(&v, &p), OP_SELECT),
            "{stem}: select(b, 1, 0)"
        );
        assert!(has_op(&validate(&f, &p), OP_I_NOT_EQUAL), "{stem}: x != 0");
        let l = link(&[&v, &f], &LinkInputs::Render(Box::new(r)), &p)
            .unwrap_or_else(|e| panic!("{stem}: link: {e}"));
        for a in [&v, &f] {
            let words = relocate(a, &l.table).expect("relocate");
            check_words_valid(&words, &a.requirements, &p)
                .unwrap_or_else(|e| panic!("{stem}: L3: {e}"));
        }
    }
}

/// A fragment `PrimitiveId` is declared through `Geometry` where the profile has `geometryShader`,
/// else through `Tessellation` where it has `tessellationShader`, else read from a varying.
#[test]
fn a_fragment_primitive_id_takes_the_enabled_alternative() {
    let Some(bc) = fixture() else { return };
    for (stem, cap, feature) in [
        ("moltenvk-m3", Cap::Tessellation, F::TessellationShader),
        ("lavapipe", Cap::Geometry, F::GeometryShader),
    ] {
        let p = prof(stem);
        let a =
            compile(&bc, "tail_fragment_primitive", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let r = &a.requirements;
        assert!(
            r.spirv_capabilities.contains(&spirv_cap(cap)),
            "{stem}: {r:?}"
        );
        assert!(r.features.contains(&feature), "{stem}: {r:?}");
        assert_eq!(
            r.spirv_capabilities
                .intersection(&[spirv_cap(Cap::Geometry), spirv_cap(Cap::Tessellation)].into())
                .count(),
            1,
            "{stem}: one alternative"
        );
        assert!(a.interface.io.builtins.contains(&StageBuiltin::PrimitiveId));
        validate(&a, &p);
    }
    // `runtime-floor` without `tessellationShader` (below the runtime's floor) has neither: the ID
    // is the flat varying the pipeline's vertex function writes (`engine_primitive_ids`).
    let floor = profile_without(
        "runtime-floor",
        &[
            F::TessellationShader,
            F::ShaderTessellationAndGeometryPointSize,
            F::MultiviewTessellationShader,
        ],
        DevicePolicy::default(),
    )
    .expect("below the runtime's floor");
    let a = compile(&bc, "tail_fragment_primitive", &floor).expect("floor");
    assert!(a
        .interface
        .io
        .inputs
        .iter()
        .any(|v| v.role == IoRole::PrimitiveId));
    assert!(!a.interface.io.builtins.contains(&StageBuiltin::PrimitiveId));
    validate(&a, &floor);
}

/// Fragment `Layer` and `ViewportIndex` inputs take their enabled alternatives. The floor declares
/// `Layer` through `shaderOutputLayer` and cannot declare `ViewportIndex`, so no stage can have
/// written the viewport index and the function reads it as 0, Metal's value with no vertex write
/// (`StageIo::zero_indices`). Below the floor, without `shaderOutputLayer`, it reads both as 0 and
/// declares neither.
#[test]
fn fragment_layer_and_viewport_inputs_take_the_enabled_alternatives() {
    let Some(bc) = fixture() else { return };
    for (stem, caps) in [
        ("moltenvk-m3", [Cap::ShaderLayer, Cap::MultiViewport]),
        ("lavapipe", [Cap::Geometry, Cap::MultiViewport]),
    ] {
        let p = prof(stem);
        let a = compile(&bc, "tail_fragment_layer", &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        for cap in caps {
            assert!(
                a.requirements.spirv_capabilities.contains(&spirv_cap(cap)),
                "{stem}: {cap:?} in {:?}",
                a.requirements.spirv_capabilities
            );
        }
        validate(&a, &p);
    }
    let floor = prof("runtime-floor");
    let a = compile(&bc, "tail_fragment_layer", &floor).expect("floor");
    let io = &a.interface.io;
    assert_eq!(io.zero_indices, [StageBuiltin::ViewportIndex]);
    assert!(io.builtins.contains(&StageBuiltin::Layer));
    assert!(!io.builtins.contains(&StageBuiltin::ViewportIndex));
    assert!(a
        .requirements
        .spirv_capabilities
        .contains(&spirv_cap(Cap::ShaderLayer)));
    assert!(!a
        .requirements
        .spirv_capabilities
        .contains(&spirv_cap(Cap::MultiViewport)));
    validate(&a, &floor);
    let below = profile_without(
        "runtime-floor",
        &[F::ShaderOutputLayer],
        DevicePolicy::default(),
    )
    .expect("below the runtime's floor");
    let a = compile(&bc, "tail_fragment_layer", &below).expect("below the floor");
    let io = &a.interface.io;
    assert_eq!(
        io.zero_indices,
        [StageBuiltin::Layer, StageBuiltin::ViewportIndex]
    );
    assert!(!io.builtins.contains(&StageBuiltin::Layer));
    assert!(!io.builtins.contains(&StageBuiltin::ViewportIndex));
    assert!(!a
        .requirements
        .spirv_capabilities
        .contains(&spirv_cap(Cap::MultiViewport)));
    validate(&a, &below);
}

/// `[[barycentric_coord]]` reads `BaryCoordKHR` (perspective) or `BaryCoordNoPerspKHR`, through
/// `SPV_KHR_fragment_shader_barycentric`, on a profile with the feature (RADV's and MoltenVK's): a
/// function reading both forms, and one reading one form alone.
#[test]
fn barycentric_coordinates_read_the_khr_builtins() {
    let Some(bc) = fixture() else { return };
    for stem in ["amd-780m-radv", "moltenvk-m3"] {
        let p = prof(stem);
        let a = compile(&bc, "tail_fragment_barycentric", &p).expect("barycentric");
        assert_eq!(
            a.interface.io.builtins,
            [
                StageBuiltin::BaryCoord { perspective: false },
                StageBuiltin::BaryCoord { perspective: true },
            ],
            "{stem}"
        );
        let r = &a.requirements;
        assert!(r
            .spirv_capabilities
            .contains(&spirv_cap(Cap::FragmentBarycentricKHR)));
        assert!(r.features.contains(&F::FragmentShaderBarycentric));
        validate(&a, &p);
    }
    for stem in ["amd-780m-radv", "moltenvk-m3"] {
        let p = prof(stem);
        let s = compile(&bc, "tail_fragment_barycentric_scalar", &p).expect("scalar barycentric");
        assert_eq!(
            s.interface.io.builtins,
            [StageBuiltin::BaryCoord { perspective: true }],
            "{stem}"
        );
        validate(&s, &p);
    }
}

/// Without `fragmentShaderBarycentric`, `[[barycentric_coord]]` is refused needing the feature,
/// with `geometryShader` (`lavapipe`, `intel-irisxe-rplp-anv`, `nvidia-pascal-linux`) or without
/// it (`runtime-floor`).
#[test]
fn without_the_feature_the_weights_are_refused() {
    let Some(bc) = fixture() else { return };
    for stem in [
        "lavapipe",
        "intel-irisxe-rplp-anv",
        "nvidia-pascal-linux",
        "runtime-floor",
    ] {
        let p = prof(stem);
        assert!(!p.fields().fragment.barycentric, "{stem}");
        for entry in [
            "tail_fragment_barycentric",
            "tail_fragment_barycentric_scalar",
        ] {
            assert_eq!(
                refused(compile(&bc, entry, &p), entry),
                (
                    Feature::Barycentrics,
                    any_of(&[F::FragmentShaderBarycentric])
                ),
                "{entry} on {stem}"
            );
        }
    }
}

/// The third weights `m` computes as Metal does: each `OpFSub` decorated `NoContraction` of the
/// constant `1.0` and an `OpFAdd` decorated `NoContraction` of the first two lanes of one value.
fn third_weights(m: &TypedModule) -> usize {
    let insts = || m.blocks().iter().flat_map(|b| b.insts.iter());
    let defs: std::collections::HashMap<_, &Inst> = insts()
        .filter_map(|n| n.result.map(|r| (r, &n.inst)))
        .collect();
    let one = |v| match m.value(v).map(|d| d.origin) {
        Ok(ValueOrigin::Const(c)) => matches!(
            m.consts().get(c).map(|d| &d.kind),
            Ok(ConstKind::Plain(Constant::Scalar { bits, .. })) if bits.0 == 0x3F80_0000
        ),
        _ => false,
    };
    let lane = |v, i: u32| match defs.get(&v) {
        Some(Inst::CompositeExtract { composite, indices }) if indices[..] == [i] => {
            Some(*composite)
        }
        _ => None,
    };
    insts()
        .filter(|n| match &n.inst {
            Inst::Binary {
                op: BinOp::FSub,
                contract: Contraction::Forbidden,
                a,
                b,
            } if one(*a) => matches!(
                defs.get(b),
                Some(Inst::Binary {
                    op: BinOp::FAdd,
                    contract: Contraction::Forbidden,
                    a: x,
                    b: y,
                }) if lane(*x, 0).is_some() && lane(*x, 0) == lane(*y, 1)
            ),
            _ => false,
        })
        .count()
}

/// A `float3` `[[barycentric_coord]]` takes `BaryCoordKHR`'s first two lanes and `1 - (x + y)` of
/// them as its third weight, Metal's (M3 Max probes: every pixel and sample of triangles and lines,
/// perspective or not), where the device's own third lane is not (the RTX 5080 Laptop GPU's, 42 of
/// 574 pixels, and exactly 0 on a line); a `float` or `float2` read computes none. The emitter
/// reads a parameter at each use, so `tail_fragment_barycentric`, which uses its weights' three
/// lanes, computes them three times.
#[test]
fn the_third_barycentric_weight_is_one_minus_the_first_two() {
    let Some(bc) = fixture() else { return };
    for stem in ["amd-780m-radv", "nvidia-rtx5080-laptop", "moltenvk-m3"] {
        let p = prof(stem);
        for (entry, weights) in [
            ("tail_fragment_barycentric", 3),
            ("tail_fragment_barycentric_scalar", 0),
        ] {
            if stem == "moltenvk-m3" && weights > 0 {
                // MoltenVK refuses this entry, which reads both forms (the driver's defect).
                continue;
            }
            let (a, _, kept) =
                compile_kept(&bc, entry, &p).unwrap_or_else(|e| panic!("{entry} on {stem}: {e:?}"));
            assert_eq!(third_weights(&kept), weights, "{entry} on {stem}");
            validate(&a, &p);
        }
    }
}

/// A render pipeline's graphics state states whether its fragment function reads
/// `[[barycentric_coord]]`, read from the compiled AIR: both forms together and the scalar
/// perspective form do; `bool` varyings, the primitive ID, and the layer and viewport do not.
#[test]
fn the_graphics_state_states_whether_the_fragment_function_reads_barycentric_coordinates() {
    let Some(bc) = fixture() else { return };
    let p = prof("amd-780m-radv");
    let mut r = RenderLinkInputs::new();
    r.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let v = compile(&bc, "tail_vertex_bool", &p).expect("vertex");
    for (entry, want) in [
        ("tail_fragment_barycentric", true),
        ("tail_fragment_barycentric_scalar", true),
        ("tail_fragment_bool", false),
        ("tail_fragment_primitive", false),
        ("tail_fragment_layer", false),
    ] {
        let f = compile_with(&bc, entry, &p, &fragment_inputs(&r, &p))
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
        let graphics = link(&[&v, &f], &LinkInputs::Render(Box::new(r.clone())), &p)
            .unwrap_or_else(|e| panic!("{entry}: link: {e}"))
            .graphics
            .expect("graphics");
        assert_eq!(graphics.reads_barycentric_coord, want, "{entry}");
    }
}

/// The plan's size estimate bounds the emitted words of every entry, the `bool` conversions and the
/// barycentric lane reads included.
#[test]
fn the_estimate_bounds_every_tail_entry() {
    let Some(bc) = fixture() else { return };
    // A profile that compiles every tail entry (on MoltenVK, both barycentric forms are refused).
    let p = prof("amd-780m-radv");
    for entry in [
        "tail_vertex_bool",
        "tail_fragment_bool",
        "tail_fragment_primitive",
        "tail_fragment_layer",
        "tail_fragment_barycentric",
        "tail_fragment_barycentric_scalar",
    ] {
        let plan = plan_bitcode(
            &bc,
            EntryRef::Name(entry),
            &FunctionConstantValues::defaults(),
            &p,
            &LoweringInputs::new(),
            &mut meter(),
        )
        .expect("plan");
        let a = compile(&bc, entry, &p).expect("compile");
        let words = a.module.words.len() as u64;
        assert!(
            plan.estimate.words >= words,
            "{entry}: estimate {} < emitted {words}",
            plan.estimate.words
        );
    }
}
