//! Stage outputs and arguments a function constant gates off.
//!
//! - A gated-off output is not declared (MSL 5.8.1.6), so it leaves the interface: the vertex and
//!   fragment entries of the owned fixture `engine/function_constant_interface` state only
//!   their gated-on outputs, and with the constants true every output is back.
//! - An observed read, query, or sample of a gated-off texture, or a sample of a present one
//!   through a gated-off sampler, is an absent read: the texture is not declared in the specialized
//!   function, so Metal leaves the read undefined, and its compiler's answer is that undefined
//!   behaviour's fallout (a store of the value dropped, arithmetic on it NaN, a texture write of it
//!   zero: the write of `kernel_absent_texture_reads_zero`'s authored case). The kernel sets
//!   `StatusBits::ABSENT_ARGUMENT_READ` where the read runs and takes zero for it, and states
//!   `RuntimeFact::AbsentArgumentRead`, reaching the status by address where the profile addresses
//!   buffers and through the stage's binding of it elsewhere. An element of a gated-off texture
//!   table, and a
//!   gated-off sampler merged with a present one before the sample, are the absent handle where the
//!   read reaches them (`src/tests/engine_absent_handles.rs` follows the handle through memory,
//!   aggregates and merges). A write to (or fence of) the gated-off texture itself is dropped, as
//!   Metal drops it on every device and layer measured. Gated on, the same kernels compile, without
//!   the fact.
//! - A gated-off input that only feeds an output gated by the same constant is not observed (the
//!   return stores only declared outputs), so the entry compiles without either.
//! - A gated-off value argument whose read nothing observes compiles (the read is `undef`); an
//!   observed read reads zero (`src/tests/engine_stage_io.rs` and
//!   `src/tests/engine_absent_value_arguments.rs`).
//! - A gated-off varying stored into a private struct whose field nothing reads, and whose address
//!   sits in a pointer cell nothing reads, is not observed; a read of the field is, and reads zero.
//! - What nothing observes after specialization goes before lowering: the element address of a
//!   gated-off texture array that only the constant's branch read compiles.
//! - Two color outputs at one `[[color(0)]]`, each gated by its own constant, compile under every
//!   set that leaves at most one; the set that leaves both is the invalid request
//!   `DuplicateColorOutput`, as Metal refuses it when it makes the function.
//!
//! `llvm-as` assembles the fixtures; without it the tests skip visibly. No GPU is used.

use std::path::Path;
use std::sync::Arc;

use crate::budget::SPIRV_WORDS;
use crate::device::{DevicePolicy, DeviceProfile};
use crate::error::{DescriptorError, Error, Feature, RuntimeInput};
use crate::library::{FunctionArtifact, FunctionConstantValue, FunctionConstantValues};
use crate::metal::DataType;
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::link::link;
use crate::pipeline::link::LinkInputs;
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::relocate::relocate;
use crate::pipeline::stage_io::{IoRole, StageBuiltin};
use crate::pipeline::sym::{SlotClass, Stage};
use crate::pipeline::testing::{
    engine_dir, fixtures_dir, llvm_assemble, meter, profile, status_reach, status_reach_on,
    AssembleError, Scratch,
};
use crate::spirv::testing::{check_words_valid, committed_profiles};

const HAS_TINT: u32 = 0;
const HAS_SIZE: u32 = 1;
const HAS_SECOND: u32 = 2;
const HAS_SOURCE: u32 = 3;
const HAS_BLEND: u32 = 4;

fn assemble(path: &Path) -> Option<Vec<u8>> {
    match llvm_assemble(path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {}: {why}", path.display()),
    }
}

fn fixture() -> Option<Vec<u8>> {
    assemble(&engine_dir().join("function_constant_interface.ll"))
}

fn prof(stem: &str) -> DeviceProfile {
    profile(stem, DevicePolicy::default()).expect("committed profile")
}

/// `entry` compiled with the boolean constants `on` true and every other constant undefined.
fn compile_with(bc: &[u8], entry: &str, on: &[u32], p: &DeviceProfile) -> Arc<FunctionArtifact> {
    let bools: Vec<(u32, bool)> = on.iter().map(|&i| (i, true)).collect();
    compile_bools(bc, entry, &bools, p)
}

/// `entry` compiled with each boolean constant of `bools` set and every other constant undefined.
fn compile_bools(
    bc: &[u8],
    entry: &str,
    bools: &[(u32, bool)],
    p: &DeviceProfile,
) -> Arc<FunctionArtifact> {
    let mut values = FunctionConstantValues::defaults();
    for &(i, v) in bools {
        values = values.with_index(i, FunctionConstantValue::new(DataType::Bool, [u8::from(v)]));
    }
    compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &values,
        p,
        &LoweringInputs::new(),
        &mut meter(),
        SPIRV_WORDS,
    )
    .unwrap_or_else(|e| panic!("{entry} with {bools:?}: {e}"))
}

/// Link `a` alone, relocate it, and run L3.
fn validate(a: &FunctionArtifact, p: &DeviceProfile) {
    let t = link(&[a], &LinkInputs::standalone(), p)
        .unwrap_or_else(|e| panic!("link: {e}"))
        .table;
    let words = relocate(a, &t).expect("relocate");
    check_words_valid(&words, &a.requirements, p).unwrap_or_else(|e| panic!("L3: {e}"));
}

/// The AIR user keys of the varyings `a` outputs.
fn varyings(a: &FunctionArtifact) -> Vec<String> {
    a.interface
        .io
        .outputs
        .iter()
        .map(|o| match &o.role {
            IoRole::Varying { user, .. } => String::from_utf8_lossy(&user.0).into_owned(),
            r => panic!("vertex output role {r:?}"),
        })
        .collect()
}

/// A gated-off varying and point size leave the vertex interface; gated on, they return.
#[test]
fn gated_off_vertex_outputs_leave_the_interface() {
    let Some(bc) = fixture() else { return };
    for stem in ["moltenvk-m3", "vk12-floor"] {
        let p = prof(stem);
        let off = compile_with(&bc, "fc_vertex_gated_outputs", &[], &p);
        assert_eq!(varyings(&off), ["generated(5shadeDv4_f)"], "{stem}");
        assert_eq!(
            off.interface.io.builtins,
            [
                StageBuiltin::VertexIndex,
                StageBuiltin::Position { invariant: true },
                StageBuiltin::PointSize { default: true },
            ],
            "{stem}: a vertex function whose point size is gated off writes the default"
        );
        validate(&off, &p);

        let on = compile_with(&bc, "fc_vertex_gated_outputs", &[HAS_TINT, HAS_SIZE], &p);
        assert_eq!(
            varyings(&on),
            ["generated(4tintDv4_f)", "generated(5shadeDv4_f)"],
            "{stem}"
        );
        assert!(
            on.interface
                .io
                .builtins
                .contains(&StageBuiltin::PointSize { default: false }),
            "{stem}: {:?}",
            on.interface.io.builtins
        );
        validate(&on, &p);
    }
}

/// A gated-off color output leaves the fragment interface.
#[test]
fn a_gated_off_color_output_leaves_the_interface() {
    let Some(bc) = fixture() else { return };
    let colors = |a: &FunctionArtifact| -> Vec<u32> {
        a.interface
            .io
            .outputs
            .iter()
            .map(|o| match o.role {
                IoRole::Color { attachment, .. } => attachment,
                ref r => panic!("fragment output role {r:?}"),
            })
            .collect()
    };
    for stem in ["moltenvk-m3", "vk12-floor"] {
        let p = prof(stem);
        let off = compile_with(&bc, "fc_fragment_gated_color", &[], &p);
        assert_eq!(colors(&off), [0], "{stem}");
        validate(&off, &p);
        let on = compile_with(&bc, "fc_fragment_gated_color", &[HAS_SECOND], &p);
        assert_eq!(colors(&on), [0, 1], "{stem}");
        validate(&on, &p);
    }
}

/// A gated-off color input that only a blend the gate selects away reads is never observed, so the
/// entry compiles without it.
#[test]
fn an_unobserved_gated_off_input_compiles() {
    let Some(bc) = fixture() else { return };
    for stem in ["moltenvk-m3", "vk12-floor"] {
        let p = prof(stem);
        let off = compile_with(&bc, "fc_fragment_unobserved_input", &[], &p);
        assert!(off.interface.io.inputs.is_empty(), "{stem}");
        assert!(off.facts.is_empty(), "{stem}: {:?}", off.facts);
        validate(&off, &p);
    }
    // Defined (and so declared), the color input is read: a framebuffer fetch. `vk12-floor` has
    // neither ordered form, so the default policy refuses it; `moltenvk-m3` orders a fetch only
    // through the pixel interlock, in a storage form that needs its pipeline's state, which a
    // standalone compile does not have.
    for stem in ["vk12-floor", "moltenvk-m3"] {
        let r = compile_bitcode(
            &bc,
            EntryRef::Name("fc_fragment_unobserved_input"),
            &FunctionConstantValues::defaults()
                .with_index(HAS_BLEND, FunctionConstantValue::new(DataType::Bool, [1])),
            &prof(stem),
            &LoweringInputs::new(),
            &mut meter(),
            SPIRV_WORDS,
        );
        match (stem, r) {
            ("vk12-floor", Err(Error::Unsupported(u))) => {
                assert_eq!(u.feature, Feature::FramebufferFetch)
            }
            ("moltenvk-m3", Err(Error::NeedsInput(RuntimeInput::FramebufferFetch))) => {}
            (_, other) => panic!("{stem}: declared color input: {:?}", other.map(|_| ())),
        }
    }
}

/// `entry` compiled with each boolean constant of `bools` set and every other constant undefined,
/// or its error.
fn compile_result(
    bc: &[u8],
    entry: &str,
    bools: &[(u32, bool)],
    p: &DeviceProfile,
) -> Result<Arc<FunctionArtifact>, Error> {
    let mut values = FunctionConstantValues::defaults();
    for &(i, v) in bools {
        values = values.with_index(i, FunctionConstantValue::new(DataType::Bool, [u8::from(v)]));
    }
    compile_bitcode(
        bc,
        EntryRef::Name(entry),
        &values,
        p,
        &LoweringInputs::new(),
        &mut meter(),
        SPIRV_WORDS,
    )
}

/// Assert that `entry` with `bools` set compiles, on every committed profile, into a kernel that
/// states its absent reads (`AbsentArgumentRead`, and the `ExecutionStatus` the read marks,
/// reached by address where the profile addresses buffers and through the stage's binding of it
/// elsewhere) and validates, or needs a storage texture's format.
fn absent_read(bc: &[u8], entry: &str, bools: &[(u32, bool)]) {
    for (stem, p) in committed_profiles().expect("profiles") {
        match compile_result(bc, entry, bools, &p) {
            Ok(a) => {
                let fact = RuntimeFact::AbsentArgumentRead {
                    stage: Stage::Kernel,
                };
                assert!(a.facts.contains(&fact), "{stem}: {:?}", a.facts);
                assert!(
                    a.facts.contains(&RuntimeFact::ExecutionStatus),
                    "{stem}: {:?}",
                    a.facts
                );
                assert_eq!(status_reach(&a), status_reach_on(&p), "{stem}: {entry}");
                validate(&a, &p);
            }
            Err(Error::NeedsInput(_)) => {}
            other => panic!("{stem}: {entry} {bools:?}: {:?}", other.map(|_| ())),
        }
    }
}

/// Assert that `entry` with `bools` set compiles and validates on every committed profile, or
/// needs a storage texture's format, and reads no absent argument.
fn compiles(bc: &[u8], entry: &str, bools: &[(u32, bool)]) {
    for (stem, p) in committed_profiles().expect("profiles") {
        match compile_result(bc, entry, bools, &p) {
            Ok(a) => {
                assert!(
                    !a.facts
                        .iter()
                        .any(|f| matches!(f, RuntimeFact::AbsentArgumentRead { .. })),
                    "{stem}: {:?}",
                    a.facts
                );
                validate(&a, &p);
            }
            Err(Error::NeedsInput(_)) => {}
            Err(e) => panic!("{stem}: {entry} {bools:?}: {e}"),
        }
    }
}

/// With `provide_absent` false, `absent_source` is read and `absent_sink` written though the
/// pipeline provides neither: the write is dropped, as Metal drops it, and the read of
/// `absent_source` is an absent read. Metal wrote zero for that read into `dest`, but its compiler
/// drops a buffer store of the same read and gives arithmetic on it NaN: the fallout of undefined
/// behaviour, not a rule, so a run of the read fails the command buffer. (With `provide_absent`
/// true the two gated textures' computed slots coincide, a set the specialization planner keeps
/// apart.)
#[test]
fn an_observed_operation_on_an_absent_texture_is_an_absent_read() {
    let Some(bc) = assemble(&fixtures_dir().join("kernel_absent_texture_reads_zero.ll")) else {
        return;
    };
    absent_read(&bc, "absent_texture_reads_zero", &[]);
}

/// An unguarded read and sample of a gated-off texture through a gated-off sampler are absent
/// reads; gated on, they are reads.
#[test]
fn a_read_of_an_absent_texture_and_sampler_is_an_absent_read() {
    let Some(bc) = fixture() else { return };
    let entry = "fc_kernel_absent_resources";
    absent_read(&bc, entry, &[]);
    compiles(&bc, entry, &[(HAS_SOURCE, true)]);
}

/// Sampling a present texture through a gated-off sampler has no Metal result, so it is an absent
/// read; gated on, it is a sample.
#[test]
fn a_present_texture_through_an_absent_sampler_is_an_absent_read() {
    let Some(bc) = fixture() else { return };
    let entry = "fc_kernel_absent_sampler";
    absent_read(&bc, entry, &[]);
    compiles(&bc, entry, &[(HAS_SOURCE, true)]);
}

/// A gated-off attribute feeding only a varying the same constant gates, and a gated-off color
/// input feeding only a color output the same constant gates, are unobserved.
#[test]
fn an_absent_input_feeding_only_an_absent_output_compiles() {
    let Some(bc) = fixture() else { return };
    for stem in ["moltenvk-m3", "vk12-floor"] {
        let p = prof(stem);
        let v = compile_with(&bc, "fc_vertex_input_to_output", &[], &p);
        assert!(varyings(&v).is_empty(), "{stem}");
        let attrs: Vec<u32> = v
            .interface
            .io
            .inputs
            .iter()
            .map(|x| match x.role {
                IoRole::Attribute { index, .. } => index,
                ref r => panic!("{stem}: vertex input role {r:?}"),
            })
            .collect();
        assert_eq!(attrs, [0], "{stem}");
        validate(&v, &p);
        let f = compile_with(&bc, "fc_fragment_input_to_output", &[], &p);
        let colors: Vec<u32> = f
            .interface
            .io
            .outputs
            .iter()
            .map(|o| match o.role {
                IoRole::Color { attachment, .. } => attachment,
                ref r => panic!("{stem}: fragment output role {r:?}"),
            })
            .collect();
        assert_eq!(colors, [1], "{stem}");
        assert!(f.facts.is_empty(), "{stem}: no fetch: {:?}", f.facts);
        validate(&f, &p);
    }
}

/// A gated-off sampler merged with a present one before a sample reaches the sample beside a
/// present texture, whose result Metal does not define where the merge chose the gated-off one: the
/// sample dispatches on the sampler, and the gated-off sampler's arm is an absent read.
#[test]
fn an_absent_sampler_merged_with_a_present_one_is_an_absent_read_where_chosen() {
    let Some(bc) = fixture() else { return };
    absent_read(&bc, "fc_kernel_sampler_select", &[]);
}

/// Two fragment functions in the shape the frontend gives a surface-shader pipeline: every stage
/// input, the gated-off `uv` among them, is stored into a private `VertexOut`, whose address is
/// stored into a private argument struct (a pointer cell) that nothing reads. The first then reads
/// only the present `st` back; the second reads `uv`.
const PRIVATE_INPUT: &str = r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.VertexOut = type { <4 x float>, <2 x float>, <2 x float> }
%struct.Surface = type { ptr, <4 x float> }

@has_uv.MTL_FC_INIT_0_b = linkonce_odr hidden local_unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@__metal_implicit_fc_pred_0 = internal addrspace(2) global i8 0, align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_, ptr null }]

define <4 x float> @fc_fragment_private_input(<4 x float> %0, <2 x float> %1, <2 x float> %2) {
  %in = alloca %struct.VertexOut, align 16
  %args = alloca %struct.Surface, align 16
  %pos = getelementptr inbounds %struct.VertexOut, ptr %in, i64 0, i32 0
  store <4 x float> %0, ptr %pos, align 16
  %uv = getelementptr inbounds %struct.VertexOut, ptr %in, i64 0, i32 1
  store <2 x float> %1, ptr %uv, align 16
  %st = getelementptr inbounds %struct.VertexOut, ptr %in, i64 0, i32 2
  store <2 x float> %2, ptr %st, align 8
  %cell = getelementptr inbounds %struct.Surface, ptr %args, i64 0, i32 0
  store ptr %in, ptr %cell, align 16
  %x = load <2 x float>, ptr %st, align 8
  %c = shufflevector <2 x float> %x, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  ret <4 x float> %c
}

define <4 x float> @fc_fragment_private_input_read(<4 x float> %0, <2 x float> %1, <2 x float> %2) {
  %in = alloca %struct.VertexOut, align 16
  %args = alloca %struct.Surface, align 16
  %pos = getelementptr inbounds %struct.VertexOut, ptr %in, i64 0, i32 0
  store <4 x float> %0, ptr %pos, align 16
  %uv = getelementptr inbounds %struct.VertexOut, ptr %in, i64 0, i32 1
  store <2 x float> %1, ptr %uv, align 16
  %st = getelementptr inbounds %struct.VertexOut, ptr %in, i64 0, i32 2
  store <2 x float> %2, ptr %st, align 8
  %cell = getelementptr inbounds %struct.Surface, ptr %args, i64 0, i32 0
  store ptr %in, ptr %cell, align 16
  %x = load <2 x float>, ptr %uv, align 16
  %c = shufflevector <2 x float> %x, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  ret <4 x float> %c
}

declare i8 @air.normalize_function_constant_predicate.i8(i8)

define internal void @_GLOBAL__sub_I_() section "air.static_init" {
  %1 = load i8, ptr addrspace(2) @has_uv.MTL_FC_INIT_0_b, align 1
  %2 = tail call i8 @air.normalize_function_constant_predicate.i8(i8 %1)
  store i8 %2, ptr addrspace(2) @__metal_implicit_fc_pred_0, align 1
  ret void
}

!air.fragment = !{!0, !8}
!air.function_constants = !{!10}
!air.version = !{!20}
!air.language_version = !{!21}

!0 = !{ptr @fc_fragment_private_input, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4, !5, !7}
!4 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!5 = !{i32 1, !"air.function_constant", !6, !"air.fragment_input", !"generated(2uvDv2_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"uv"}
!6 = !{ptr addrspace(2) @__metal_implicit_fc_pred_0, !"bool", !"has_uv"}
!7 = !{i32 2, !"air.fragment_input", !"generated(2stDv2_f)", !"air.center", !"air.perspective", !"air.arg_type_name", !"float2", !"air.arg_name", !"st"}
!8 = !{ptr @fc_fragment_private_input_read, !1, !3}
!10 = !{ptr addrspace(2) @has_uv.MTL_FC_INIT_0_b, !"bool", !"has_uv", i32 0}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// [`PRIVATE_INPUT`] assembled in a scratch directory removed before this returns.
fn private_input() -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-fc-private-input");
    let dir = scratch.path();
    let ll = dir.join("private_input.ll");
    std::fs::write(&ll, PRIVATE_INPUT).expect("write");
    assemble(&ll)
}

/// The AIR user keys of the varyings `a` reads.
fn inputs(a: &FunctionArtifact) -> Vec<String> {
    a.interface
        .io
        .inputs
        .iter()
        .map(|i| match &i.role {
            IoRole::Varying { user, .. } => String::from_utf8_lossy(&user.0).into_owned(),
            r => panic!("fragment input role {r:?}"),
        })
        .collect()
}

/// A gated-off varying stored into a private struct whose field nothing reads (the struct's address
/// sits in a pointer cell nothing reads either) is not observed, so the fragment compiles without
/// it. Declared, nothing observes it either: the store goes, and the fragment reads only `st`. A
/// read of the gated-off field is observed and reads zero: the fragment still declares no input for
/// it.
#[test]
fn an_absent_input_stored_into_private_memory_nothing_reads_compiles() {
    let Some(bc) = private_input() else { return };
    for stem in ["moltenvk-m3", "vk12-floor"] {
        let p = prof(stem);
        let off = compile_with(&bc, "fc_fragment_private_input", &[], &p);
        assert_eq!(inputs(&off), ["generated(2stDv2_f)"], "{stem}");
        validate(&off, &p);
        let on = compile_with(&bc, "fc_fragment_private_input", &[0], &p);
        assert_eq!(inputs(&on), ["generated(2stDv2_f)"], "{stem}");
        validate(&on, &p);
        let read = compile_with(&bc, "fc_fragment_private_input_read", &[], &p);
        assert!(
            !inputs(&read).contains(&"generated(2uvDv2_f)".to_owned()),
            "{stem}: {:?}",
            inputs(&read)
        );
        validate(&read, &p);
    }
}

/// A kernel over a texture array `ts` (`array<texture2d<float>, 4> [[texture(0),
/// function_constant(has_ts)]]`, a thread pointer to its handles) and `out` (`uint *`): it takes
/// the address of element `out[0]` before the constant's branch, and only the branch reads it,
/// storing that texture's width to `out[1]`. The shape the frontend leaves when it hoists an
/// element address above a function-constant test.
const ABSENT_TABLE: &str = r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@has_ts.MTL_FC_INIT_0_b = linkonce_odr hidden local_unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@__metal_implicit_fc_pred_0 = internal addrspace(2) global i8 0, align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_, ptr null }]

define void @fc_kernel_absent_table(ptr readonly byval([4 x ptr addrspace(1)]) %ts, ptr addrspace(1) %out) {
entry:
  %k = load i32, ptr addrspace(1) %out, align 4
  %i = zext i32 %k to i64
  %e = getelementptr inbounds [4 x ptr addrspace(1)], ptr %ts, i64 0, i64 %i
  %fc = load i8, ptr addrspace(2) @has_ts.MTL_FC_INIT_0_b, align 1
  %on = icmp ne i8 %fc, 0
  br i1 %on, label %read, label %done

read:
  %h = load ptr addrspace(1), ptr %e, align 8
  %x = call i32 @air.get_width_texture_2d(ptr addrspace(1) %h, i32 0)
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %x, ptr addrspace(1) %o, align 4
  br label %done

done:
  ret void
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare i8 @air.normalize_function_constant_predicate.i8(i8)

define internal void @_GLOBAL__sub_I_() section "air.static_init" {
  %1 = load i8, ptr addrspace(2) @has_ts.MTL_FC_INIT_0_b, align 1
  %2 = tail call i8 @air.normalize_function_constant_predicate.i8(i8 %1)
  store i8 %2, ptr addrspace(2) @__metal_implicit_fc_pred_0, align 1
  ret void
}

!air.kernel = !{!0}
!air.function_constants = !{!10}
!air.version = !{!20}
!air.language_version = !{!21}

!0 = !{ptr @fc_kernel_absent_table, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.function_constant", !5, !"air.texture", !"air.location_index", i32 0, i32 4, !"air.sample", !"air.arg_type_name", !"array<texture2d<float, sample>, 4>", !"air.arg_name", !"ts"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!5 = !{ptr addrspace(2) @__metal_implicit_fc_pred_0, !"bool", !"has_ts"}
!10 = !{ptr addrspace(2) @has_ts.MTL_FC_INIT_0_b, !"bool", !"has_ts", i32 0}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The texture slots `a` binds.
fn texture_slots(a: &FunctionArtifact) -> Vec<u32> {
    a.interface
        .slots
        .keys()
        .filter(|s| s.class == SlotClass::Texture)
        .map(|s| s.index)
        .collect()
}

/// Gated off, the constant's branch goes, leaving the texture array's element address with no use:
/// its read of the array is unobserved and reads `undef`. Nothing observes the address, so it goes
/// before lowering, which never sees an element address into an absent table; gated on, the kernel
/// binds all four textures.
#[test]
fn an_absent_texture_array_s_element_address_nothing_uses_compiles() {
    let scratch = Scratch::new("m2v-fc-absent-table");
    let dir = scratch.path();
    let ll = dir.join("absent_table.ll");
    std::fs::write(&ll, ABSENT_TABLE).expect("write");
    let bc = assemble(&ll);
    let Some(bc) = bc else { return };
    for (stem, p) in committed_profiles().expect("profiles") {
        let off = compile_bools(&bc, "fc_kernel_absent_table", &[(0, false)], &p);
        assert!(texture_slots(&off).is_empty(), "{stem}");
        validate(&off, &p);
        let on = compile_with(&bc, "fc_kernel_absent_table", &[0], &p);
        assert_eq!(texture_slots(&on), [0, 1, 2, 3], "{stem}");
        validate(&on, &p);
    }
}

/// Two `float4` outputs at `[[color(0)]]` of `fc_fragment_shared_color`, gated by the
/// constants `use_a` (0) and `use_b` (1). MSL accepts the declaration; Metal refuses the
/// function only when both constants are true (makeFunction: "duplicated indices pair").
const SHARED_COLOR: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@use_a.MTL_FC_INIT_0_b = linkonce_odr hidden local_unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@use_b.MTL_FC_INIT_1_b = linkonce_odr hidden local_unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@__metal_implicit_fc_pred_0 = internal addrspace(2) global i8 0, align 1
@__metal_implicit_fc_pred_1 = internal addrspace(2) global i8 0, align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_, ptr null }]

define <{ <4 x float>, <4 x float> }> @fc_fragment_shared_color(<4 x float> noundef %0) {
  ret <{ <4 x float>, <4 x float> }> <{ <4 x float> <float 1.0, float 0.0, float 0.0, float 1.0>, <4 x float> <float 0.0, float 1.0, float 0.0, float 1.0> }>
}

declare i8 @air.normalize_function_constant_predicate.i8(i8)

define internal void @_GLOBAL__sub_I_() section "air.static_init" {
  %1 = load i8, ptr addrspace(2) @use_a.MTL_FC_INIT_0_b, align 1
  %2 = load i8, ptr addrspace(2) @use_b.MTL_FC_INIT_1_b, align 1
  %3 = tail call i8 @air.normalize_function_constant_predicate.i8(i8 %1)
  store i8 %3, ptr addrspace(2) @__metal_implicit_fc_pred_0, align 1
  %4 = tail call i8 @air.normalize_function_constant_predicate.i8(i8 %2)
  store i8 %4, ptr addrspace(2) @__metal_implicit_fc_pred_1, align 1
  ret void
}

!air.fragment = !{!0}
!air.function_constants = !{!10, !11}
!air.version = !{!20}
!air.language_version = !{!21}

!0 = !{ptr @fc_fragment_shared_color, !1, !6}
!1 = !{!2, !4}
!2 = !{!"air.function_constant", !3, !"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"a"}
!3 = !{ptr addrspace(2) @__metal_implicit_fc_pred_0, !"bool", !"use_a"}
!4 = !{!"air.function_constant", !5, !"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4", !"air.arg_name", !"b"}
!5 = !{ptr addrspace(2) @__metal_implicit_fc_pred_1, !"bool", !"use_b"}
!6 = !{!7}
!7 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos", !"air.arg_unused"}
!10 = !{ptr addrspace(2) @use_a.MTL_FC_INIT_0_b, !"bool", !"use_a", i32 0}
!11 = !{ptr addrspace(2) @use_b.MTL_FC_INIT_1_b, !"bool", !"use_b", i32 1}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The `[[color(n)]]` attachments `a` outputs.
fn color_outputs(a: &FunctionArtifact) -> Vec<u32> {
    a.interface
        .io
        .outputs
        .iter()
        .map(|o| match &o.role {
            IoRole::Color { attachment, .. } => *attachment,
            r => panic!("fragment output role {r:?}"),
        })
        .collect()
}

/// Each set that leaves at most one of the two `[[color(0)]]` outputs compiles with the one it
/// leaves; the set that leaves both is refused as an invalid request naming the pair, never
/// translated with one store winning.
#[test]
fn two_remaining_outputs_at_one_color_are_an_invalid_request() {
    let scratch = Scratch::new("m2v-fc-shared-color");
    let dir = scratch.path();
    let ll = dir.join("shared_color.ll");
    std::fs::write(&ll, SHARED_COLOR).expect("write");
    let bc = assemble(&ll);
    let Some(bc) = bc else { return };
    let entry = "fc_fragment_shared_color";
    for (stem, p) in committed_profiles().expect("profiles") {
        for (a, b, colors) in [(false, false, 0), (true, false, 1), (false, true, 1)] {
            let f = compile_bools(&bc, entry, &[(0, a), (1, b)], &p);
            assert_eq!(color_outputs(&f), vec![0; colors], "{stem} {a} {b}");
            validate(&f, &p);
        }
        let mut both = FunctionConstantValues::defaults();
        for i in [0, 1] {
            both = both.with_index(i, FunctionConstantValue::new(DataType::Bool, [1]));
        }
        let refused = compile_bitcode(
            &bc,
            EntryRef::Name(entry),
            &both,
            &p,
            &LoweringInputs::new(),
            &mut meter(),
            SPIRV_WORDS,
        );
        assert!(
            matches!(
                refused,
                Err(Error::InvalidDescriptor(
                    DescriptorError::DuplicateColorOutput { color: 0, index: 0 }
                ))
            ),
            "{stem}: {refused:?}"
        );
    }
}

/// The same kernel with the read unguarded reads an element of the gated-off table whatever the
/// constant: every element of the table is the absent handle, so the read is an absent read, which
/// fails the command buffer that runs it. Metal dropped the whole body of a kernel that reads such
/// an element unconditionally, its stores to a present buffer included: the fallout of
/// undefined behaviour, which no translation reproduces, and which reports no result as good.
#[test]
fn an_observed_element_of_an_absent_texture_table_is_an_absent_read() {
    let guard = "br i1 %on, label %read, label %done";
    assert!(ABSENT_TABLE.contains(guard));
    let text = ABSENT_TABLE.replace(guard, "br label %read");
    let scratch = Scratch::new("m2v-fc-absent-element");
    let dir = scratch.path();
    let ll = dir.join("absent_element.ll");
    std::fs::write(&ll, text).expect("write");
    let bc = assemble(&ll);
    let Some(bc) = bc else { return };
    let entry = "fc_kernel_absent_table";
    absent_read(&bc, entry, &[(0, false)]);
    compiles(&bc, entry, &[(0, true)]);
}
