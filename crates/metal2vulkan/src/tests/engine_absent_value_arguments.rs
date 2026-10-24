//! Arguments passed by value that a function constant gates off, read without a guard.
//!
//! MSL 5.8.1.2 and 5.8.1.3: with its constant false or undefined, such an argument is not
//! declared in the function signature, and the language gives its read no value. The engine
//! reads the all-zero value of its type and declares nothing for the argument: no attribute,
//! varying, color input, or builtin, and so neither the capability nor the fact that reading it
//! would need. With the constant true the argument is an ordinary input. The owned fixtures
//! `engine/fca_kernel_absent_builtin`, `fca_vertex_absent_attribute`, and
//! `fca_fragment_absent_inputs` are the sources of authored Metal cases; here the kernel runs on
//! the CPU interpreter under both constant values, and the graphics entries are checked through
//! their interfaces and L3 (`spirv-val` within each committed profile).
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly. No GPU is used.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::budget::SPIRV_WORDS;
use crate::device::{DevicePolicy, DeviceProfile};
use crate::error::{Error, RuntimeInput};
use crate::library::{FunctionArtifact, FunctionConstantValue, FunctionConstantValues};
use crate::lower::select::Plan;
use crate::metal::{DataType, PixelFormat};
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::descriptor::ColorAttachmentDescriptor;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::lower_inputs::{FramebufferFetchInputs, LoweringInputs};
use crate::pipeline::stage_io::{IoRole, StageBuiltin};
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    engine_dir, link_and_validate, llvm_assemble, meter, profile, typed_module_specialized,
    AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;

fn assemble(stem: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {stem}.ll: {why}"),
    }
}

fn prof(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::default()).expect("committed profile")
}

/// The boolean constants `on` true, every other constant undefined (so its gate is off).
fn constants(on: &[u32]) -> FunctionConstantValues {
    let mut values = FunctionConstantValues::defaults();
    for &i in on {
        values = values.with_index(i, FunctionConstantValue::new(DataType::Bool, [1]));
    }
    values
}

/// `entry` compiled with the boolean constants `on` true.
fn compile_with(
    bc: &[u8],
    entry: &str,
    on: &[u32],
    p: &DeviceProfile,
) -> Result<Arc<FunctionArtifact>, Error> {
    compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &constants(on),
        p,
        &LoweringInputs::new(),
        &mut meter(),
        SPIRV_WORDS,
    )
}

/// Every word of `out` after one workgroup of four threads ran `m` (planned as `plan`), with
/// `out` filled with a sentinel first; identical in both scheduling orders.
fn run_kernel(plan: &Plan, m: &TypedModule) -> Vec<u32> {
    let mut spec = BTreeMap::new();
    for (axis, v) in [(Axis::X, 4), (Axis::Y, 1), (Axis::Z, 1)] {
        spec.insert(SpecKey::LocalSize(axis), v);
    }
    spec.insert(SpecKey::FlatLocalSize, 4);
    // Zero push bytes: each buffer is bound at its start, and the dispatch is one region at the
    // grid's origin.
    for f in &plan.push.fields {
        assert!(matches!(f.loc, PushLoc::Push { .. }), "{f:?}");
        assert!(
            matches!(
                f.field,
                PushField::SlotBindOffset(_) | PushField::DispatchRegion
            ),
            "unexpected push field {f:?}"
        );
    }
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: [0xab; 16].to_vec(),
        }],
        push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            order,
            ..Config::default()
        };
        let out = run(m, &cfg, res.clone()).unwrap_or_else(|e| panic!("{order:?}: {e:?}"));
        outs.push(out.buffers);
    }
    assert_eq!(outs[0], outs[1], "scheduling orders disagree");
    outs[0][0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// A kernel builtin gated off reads zero: every thread stores 0; gated on, it is
/// `threads_per_threadgroup`, 4. On every committed profile.
#[test]
fn an_absent_kernel_builtin_reads_zero() {
    let Some(bc) = assemble("fca_kernel_absent_builtin") else {
        return;
    };
    let entry = "fca_kernel_absent_builtin";
    for (stem, p) in committed_profiles().expect("profiles") {
        for (on, want) in [(false, 0), (true, 4)] {
            let gates: &[u32] = if on { &[0] } else { &[] };
            let (plan, m) = typed_module_specialized(&bc, entry, &p, &constants(gates))
                .unwrap_or_else(|e| panic!("{stem}, has_width {on}: {e}"));
            assert_eq!(run_kernel(&plan, &m), [want; 4], "{stem}, has_width {on}");
            let a = compile_with(&bc, entry, gates, &p)
                .unwrap_or_else(|e| panic!("{stem}, has_width {on}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}, has_width {on}: {e}"));
        }
    }
}

/// The vertex attributes `a` reads, fixed-function or pulled.
fn attributes(a: &FunctionArtifact) -> Vec<u32> {
    let mut out: Vec<u32> = a
        .interface
        .io
        .inputs
        .iter()
        .map(|x| match x.role {
            IoRole::Attribute { index, .. } => index,
            ref r => panic!("vertex input role {r:?}"),
        })
        .chain(a.interface.io.pulled.iter().map(|x| x.index))
        .collect();
    out.sort_unstable();
    out
}

/// A vertex attribute gated off is not declared: the function reads attribute 0 alone and still
/// writes its `tint` varying (zero); gated on, it reads attribute 1 too.
#[test]
fn an_absent_vertex_attribute_is_not_declared() {
    let Some(bc) = assemble("fca_vertex_absent_attribute") else {
        return;
    };
    let entry = "fca_vertex_absent_attribute";
    for stem in ["moltenvk-m3", "vk12-floor"] {
        let p = prof(stem);
        for (on, want) in [(false, vec![0]), (true, vec![0, 1])] {
            let gates: &[u32] = if on { &[0] } else { &[] };
            let a = compile_with(&bc, entry, gates, &p)
                .unwrap_or_else(|e| panic!("{stem}, has_tint {on}: {e}"));
            assert_eq!(attributes(&a), want, "{stem}, has_tint {on}");
            let varyings: Vec<String> = a
                .interface
                .io
                .outputs
                .iter()
                .map(|o| match &o.role {
                    IoRole::Varying { user, .. } => String::from_utf8_lossy(&user.0).into_owned(),
                    r => panic!("vertex output role {r:?}"),
                })
                .collect();
            assert_eq!(varyings, ["generated(4tintDv4_f)"], "{stem}, has_tint {on}");
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}, has_tint {on}: {e}"));
        }
    }
}

const HAS_TINT: u32 = 0;
const HAS_PRIOR: u32 = 1;
const HAS_PRIMITIVE: u32 = 2;
const HAS_BARYCENTRIC: u32 = 3;

/// With every gate off, the fragment function declares none of its four gated inputs: no
/// varying, no color input (no framebuffer fetch), and neither `PrimitiveId` nor `BaryCoordKHR`,
/// so it compiles and passes L3 on every committed profile, including those without the
/// geometry, tessellation, or barycentric features those builtins need.
#[test]
fn absent_fragment_inputs_declare_nothing() {
    let Some(bc) = assemble("fca_fragment_absent_inputs") else {
        return;
    };
    let entry = "fca_fragment_absent_inputs";
    for (stem, p) in committed_profiles().expect("profiles") {
        let a = compile_with(&bc, entry, &[], &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        assert!(
            a.interface.io.inputs.is_empty(),
            "{stem}: {:?}",
            a.interface.io.inputs
        );
        assert!(
            !a.interface.io.builtins.iter().any(|b| matches!(
                b,
                StageBuiltin::PrimitiveId | StageBuiltin::BaryCoord { .. }
            )),
            "{stem}: {:?}",
            a.interface.io.builtins
        );
        assert!(a.facts.is_empty(), "{stem}: no fetch: {:?}", a.facts);
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
    }
}

/// Each gate on alone declares its input as any such input is declared: the varying, the
/// `PrimitiveId` and `BaryCoordKHR` builtins (on `moltenvk-m3`, which has what they need), and
/// the color input, a framebuffer fetch `moltenvk-m3` reads through the pixel interlock under its
/// pipeline's state.
#[test]
fn each_fragment_input_gated_on_is_declared() {
    let Some(bc) = assemble("fca_fragment_absent_inputs") else {
        return;
    };
    let entry = "fca_fragment_absent_inputs";
    let p = prof("moltenvk-m3");
    let tint = compile_with(&bc, entry, &[HAS_TINT], &p).expect("has_tint");
    let users: Vec<String> = tint
        .interface
        .io
        .inputs
        .iter()
        .map(|i| match &i.role {
            IoRole::Varying { user, .. } => String::from_utf8_lossy(&user.0).into_owned(),
            r => panic!("fragment input role {r:?}"),
        })
        .collect();
    assert_eq!(users, ["generated(4tintDv4_f)"]);
    link_and_validate(&tint, &p).expect("has_tint: L3");
    let primitive = compile_with(&bc, entry, &[HAS_PRIMITIVE], &p).expect("has_primitive");
    assert!(
        primitive
            .interface
            .io
            .builtins
            .contains(&StageBuiltin::PrimitiveId),
        "{:?}",
        primitive.interface.io.builtins
    );
    link_and_validate(&primitive, &p).expect("has_primitive: L3");
    let barycentric = compile_with(&bc, entry, &[HAS_BARYCENTRIC], &p).expect("has_barycentric");
    assert!(
        barycentric
            .interface
            .io
            .builtins
            .contains(&StageBuiltin::BaryCoord { perspective: true }),
        "{:?}",
        barycentric.interface.io.builtins
    );
    link_and_validate(&barycentric, &p).expect("has_barycentric: L3");
    // `moltenvk-m3` orders a fetch only through the pixel interlock: the color input reads its
    // attachment in the interlocked storage form under its pipeline's state, which a standalone
    // compile does not have.
    match compile_with(&bc, entry, &[HAS_PRIOR], &p) {
        Err(Error::NeedsInput(RuntimeInput::FramebufferFetch)) => {}
        other => panic!("declared color input: {:?}", other.map(|_| ())),
    }
    let mut inputs = LoweringInputs::new();
    inputs.framebuffer_fetch = Some(FramebufferFetchInputs::of(
        [0],
        &[Some(ColorAttachmentDescriptor::new(
            PixelFormat::RGBA8Unorm,
        ))],
        false,
        false,
        1,
    ));
    let prior = compile_bitcode(
        &bc,
        EntryRef::Name(entry),
        &constants(&[HAS_PRIOR]),
        &p,
        &inputs,
        &mut meter(),
        SPIRV_WORDS,
    )
    .expect("has_prior");
    assert!(
        prior.facts.iter().any(|f| matches!(
            f,
            RuntimeFact::InterlockedFramebufferFetch {
                color: 0,
                writes: true,
                ..
            }
        )),
        "{:?}",
        prior.facts
    );
    link_and_validate(&prior, &p).expect("has_prior: L3");
}
