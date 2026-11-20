//! A texture or sampler a function constant removes, reached through memory, aggregates, table
//! elements and merges.
//!
//! A removed texture or sampler is not declared in the specialized function (MSL 5.8.1.2), and
//! Metal gives a read through one no value. Read only as a texture call's own operand, the call
//! is made an absent read while specializing (`canon::interface::absent_reads`). Read any other
//! way, the argument stays in the body (`Entry::absent_handles`) and lowering follows its handle
//! as it follows a present one's: every handle it holds is the absent handle
//! (`memory::handles::Candidate::Absent`), and each texture operation the handle reaches is an
//! absent read: it takes zero and marks nothing (`texture.absent_argument`), alone or as the arm
//! of a switch-at-use over the operation's candidates. A run that never reaches the read is
//! Metal's exactly.
//!
//! The kernels below are our own, each reading the removed texture `t` (or sampler `s`) only where
//! the run-time flag `out[0]` is nonzero, otherwise writing 7, through one shape:
//!
//! - `ah_stack_struct`: `t` stored into a stack struct, the field loaded back;
//! - `ah_two_level`: `t` stored into a stack struct whose address another stack struct holds, read
//!   through both;
//! - `ah_struct_value`: `t` inserted into a struct value, the member extracted;
//! - `ah_table_element`: an element of the removed table `ts` at a run-time index;
//! - `ah_select`, `ah_phi`: `t` merged with the present texture `p` by a run-time condition;
//! - `ah_sampler_stack`: the present texture `p` sampled through the removed sampler `s`, stored
//!   into a stack struct and loaded back.
//!
//! With `has_t` false each compiles, on every committed profile, into a kernel whose absent arm
//! takes zero and marks nothing (no fact, no status push field or binding), and validates.
//! With `has_t` true each compiles too. Previously every one was refused
//! `FunctionConstantArgument` while specializing.
//!
//! `llvm-as` assembles the module; without it the tests skip visibly. No GPU is used.

use std::sync::Arc;

use crate::budget::SPIRV_WORDS;
use crate::device::DeviceProfile;
use crate::error::Error;
use crate::library::{FunctionArtifact, FunctionConstantValue, FunctionConstantValues};
use crate::metal::DataType;
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::relocate::relocate;
use crate::pipeline::sym::{SlotClass, Stage};
use crate::pipeline::testing::{
    llvm_assemble, meter, status_reach, AssembleError, Scratch, StatusReach,
};
use crate::spirv::testing::{check_words_valid, committed_profiles};

/// The kernels (module docs). Argument 0 is the removed texture `t` (a table `ts` in
/// `ah_table_element`, the removed sampler `s` in `ah_sampler_stack`), gated by `has_t`;
/// argument 1 the present texture `p` where a kernel has one; then `out`.
const MODULE: &str = r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@has_t.MTL_FC_INIT_0_b = linkonce_odr hidden local_unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@__metal_implicit_fc_pred_0 = internal addrspace(2) global i8 0, align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_, ptr null }]

define void @ah_stack_struct(ptr addrspace(1) %t, ptr addrspace(1) %out) {
entry:
  %slot = alloca { ptr addrspace(1), i32 }, align 8
  %f = getelementptr inbounds { ptr addrspace(1), i32 }, ptr %slot, i64 0, i32 0
  store ptr addrspace(1) %t, ptr %f, align 8
  %k = load i32, ptr addrspace(1) %out, align 4
  %on = icmp ne i32 %k, 0
  br i1 %on, label %read, label %done

read:
  %h = load ptr addrspace(1), ptr %f, align 8
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %h, i32 0)
  br label %done

done:
  %v = phi i32 [ %w, %read ], [ 7, %entry ]
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

define void @ah_two_level(ptr addrspace(1) %t, ptr addrspace(1) %out) {
entry:
  %inner = alloca { ptr addrspace(1) }, align 8
  %outer = alloca { ptr, i32 }, align 8
  %fi = getelementptr inbounds { ptr addrspace(1) }, ptr %inner, i64 0, i32 0
  store ptr addrspace(1) %t, ptr %fi, align 8
  %fo = getelementptr inbounds { ptr, i32 }, ptr %outer, i64 0, i32 0
  store ptr %inner, ptr %fo, align 8
  %k = load i32, ptr addrspace(1) %out, align 4
  %on = icmp ne i32 %k, 0
  br i1 %on, label %read, label %done

read:
  %a = load ptr, ptr %fo, align 8
  %af = getelementptr inbounds { ptr addrspace(1) }, ptr %a, i64 0, i32 0
  %h = load ptr addrspace(1), ptr %af, align 8
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %h, i32 0)
  br label %done

done:
  %v = phi i32 [ %w, %read ], [ 7, %entry ]
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

define void @ah_struct_value(ptr addrspace(1) %t, ptr addrspace(1) %out) {
entry:
  %k = load i32, ptr addrspace(1) %out, align 4
  %agg = insertvalue { ptr addrspace(1), i32 } undef, ptr addrspace(1) %t, 0
  %agg2 = insertvalue { ptr addrspace(1), i32 } %agg, i32 %k, 1
  %on = icmp ne i32 %k, 0
  br i1 %on, label %read, label %done

read:
  %h = extractvalue { ptr addrspace(1), i32 } %agg2, 0
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %h, i32 0)
  br label %done

done:
  %v = phi i32 [ %w, %read ], [ 7, %entry ]
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

define void @ah_table_element(ptr readonly byval([2 x ptr addrspace(1)]) %ts, ptr addrspace(1) %out) {
entry:
  %k = load i32, ptr addrspace(1) %out, align 4
  %on = icmp ne i32 %k, 0
  br i1 %on, label %read, label %done

read:
  %j = and i32 %k, 1
  %i = zext i32 %j to i64
  %e = getelementptr inbounds [2 x ptr addrspace(1)], ptr %ts, i64 0, i64 %i
  %h = load ptr addrspace(1), ptr %e, align 8
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %h, i32 0)
  br label %done

done:
  %v = phi i32 [ %w, %read ], [ 7, %entry ]
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

define void @ah_select(ptr addrspace(1) %t, ptr addrspace(1) %p, ptr addrspace(1) %out) {
entry:
  %k = load i32, ptr addrspace(1) %out, align 4
  %on = icmp ne i32 %k, 0
  br i1 %on, label %read, label %done

read:
  %c = icmp ugt i32 %k, 1
  %h = select i1 %c, ptr addrspace(1) %p, ptr addrspace(1) %t
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %h, i32 0)
  br label %done

done:
  %v = phi i32 [ %w, %read ], [ 7, %entry ]
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

define void @ah_phi(ptr addrspace(1) %t, ptr addrspace(1) %p, ptr addrspace(1) %out) {
entry:
  %k = load i32, ptr addrspace(1) %out, align 4
  %on = icmp ne i32 %k, 0
  br i1 %on, label %pick, label %done

pick:
  %c = icmp ugt i32 %k, 1
  br i1 %c, label %present, label %read

present:
  br label %read

read:
  %h = phi ptr addrspace(1) [ %p, %present ], [ %t, %pick ]
  %w = call i32 @air.get_width_texture_2d(ptr addrspace(1) %h, i32 0)
  br label %done

done:
  %v = phi i32 [ %w, %read ], [ 7, %entry ]
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

define void @ah_sampler_stack(ptr addrspace(2) %s, ptr addrspace(1) %p, ptr addrspace(1) %out) {
entry:
  %slot = alloca { ptr addrspace(2) }, align 8
  %f = getelementptr inbounds { ptr addrspace(2) }, ptr %slot, i64 0, i32 0
  store ptr addrspace(2) %s, ptr %f, align 8
  %k = load i32, ptr addrspace(1) %out, align 4
  %on = icmp ne i32 %k, 0
  br i1 %on, label %read, label %done

read:
  %h = load ptr addrspace(2), ptr %f, align 8
  %r = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %p, ptr addrspace(2) %h, <2 x float> splat (float 5.000000e-01), i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
  %x = extractvalue { <4 x float>, i8 } %r, 0
  %x0 = extractelement <4 x float> %x, i64 0
  %w = bitcast float %x0 to i32
  br label %done

done:
  %v = phi i32 [ %w, %read ], [ 7, %entry ]
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare i8 @air.normalize_function_constant_predicate.i8(i8)

define internal void @_GLOBAL__sub_I_() section "air.static_init" {
  %1 = load i8, ptr addrspace(2) @has_t.MTL_FC_INIT_0_b, align 1
  %2 = tail call i8 @air.normalize_function_constant_predicate.i8(i8 %1)
  store i8 %2, ptr addrspace(2) @__metal_implicit_fc_pred_0, align 1
  ret void
}

!air.kernel = !{!30, !31, !32, !33, !34, !35, !36}
!air.function_constants = !{!10}
!air.version = !{!20}
!air.language_version = !{!21}

!1 = !{}
!5 = !{ptr addrspace(2) @__metal_implicit_fc_pred_0, !"bool", !"has_t"}
!10 = !{ptr addrspace(2) @has_t.MTL_FC_INIT_0_b, !"bool", !"has_t", i32 0}
!11 = !{i32 0, !"air.function_constant", !5, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"t"}
!12 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 0, !"air.function_constant", !5, !"air.texture", !"air.location_index", i32 0, i32 2, !"air.read", !"air.arg_type_name", !"array<texture2d<float, read>, 2>", !"air.arg_name", !"ts"}
!14 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.read", !"air.arg_type_name", !"texture2d<float, read>", !"air.arg_name", !"p"}
!15 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!16 = !{i32 0, !"air.function_constant", !5, !"air.sampler", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"s"}
!17 = !{i32 1, !"air.texture", !"air.location_index", i32 1, i32 1, !"air.sample", !"air.arg_type_name", !"texture2d<float, sample>", !"air.arg_name", !"p"}
!22 = !{!11, !12}
!23 = !{!13, !12}
!24 = !{!11, !14, !15}
!25 = !{!16, !17, !15}
!30 = !{ptr @ah_stack_struct, !1, !22}
!31 = !{ptr @ah_two_level, !1, !22}
!32 = !{ptr @ah_struct_value, !1, !22}
!33 = !{ptr @ah_table_element, !1, !23}
!34 = !{ptr @ah_select, !1, !24}
!35 = !{ptr @ah_phi, !1, !24}
!36 = !{ptr @ah_sampler_stack, !1, !25}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// Every kernel of [`MODULE`].
const KERNELS: [&str; 7] = [
    "ah_stack_struct",
    "ah_two_level",
    "ah_struct_value",
    "ah_table_element",
    "ah_select",
    "ah_phi",
    "ah_sampler_stack",
];

/// [`MODULE`] assembled, or `None` (skipped visibly) without `llvm-as`.
fn module() -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-absent-handles");
    let dir = scratch.path();
    let ll = dir.join("absent_handles.ll");
    std::fs::write(&ll, MODULE).expect("write");
    let r = llvm_assemble(&ll);
    match r {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the module: {why}"),
    }
}

/// `entry` compiled on `p` with `has_t` set to `on`.
fn compile(
    bc: &[u8],
    entry: &str,
    on: bool,
    p: &DeviceProfile,
) -> Result<Arc<FunctionArtifact>, Error> {
    let values = FunctionConstantValues::defaults().with_index(
        0,
        FunctionConstantValue::new(DataType::Bool, [u8::from(on)]),
    );
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

/// Link `a` alone, relocate it, and run L3.
fn validate(a: &FunctionArtifact, p: &DeviceProfile) {
    let t = link(&[a], &LinkInputs::standalone(), p)
        .unwrap_or_else(|e| panic!("link: {e}"))
        .table;
    let words = relocate(a, &t).expect("relocate");
    check_words_valid(&words, &a.requirements, p).unwrap_or_else(|e| panic!("L3: {e}"));
}

/// Whether `a` states an absent read.
fn absent(a: &FunctionArtifact) -> bool {
    a.facts.contains(&RuntimeFact::AbsentArgumentRead {
        stage: Stage::Kernel,
    })
}

/// The texture slots `a` binds.
fn texture_slots(a: &FunctionArtifact) -> Vec<u32> {
    a.interface
        .slots
        .keys()
        .filter(|s| s.class == SlotClass::Texture)
        .map(|s| s.index)
        .collect()
}

/// With `has_t` false, each shape compiles on every committed profile into an absent read that
/// takes zero and marks nothing (no fact, no status push field or binding), binds only the
/// present texture a merge may still choose (no slot for the removed one), and validates.
#[test]
fn a_removed_handle_reached_through_any_shape_is_an_absent_read() {
    let Some(bc) = module() else { return };
    for (stem, p) in committed_profiles().expect("profiles") {
        for entry in KERNELS {
            let a =
                compile(&bc, entry, false, &p).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            assert!(
                !absent(&a) && !a.facts.contains(&RuntimeFact::ExecutionStatus),
                "{stem} {entry}: {:?}",
                a.facts
            );
            assert_eq!(status_reach(&a), StatusReach::None, "{stem} {entry}");
            // A merge may still choose `p`; a sample through the removed sampler is the
            // absent read whatever it samples, so it reads no texture.
            let present: &[u32] = match entry {
                "ah_select" | "ah_phi" => &[1],
                _ => &[],
            };
            assert_eq!(texture_slots(&a), present, "{stem} {entry}");
            validate(&a, &p);
        }
    }
}

/// With `has_t` true the same kernels read their textures and samplers.
#[test]
fn a_present_handle_through_the_same_shapes_is_read() {
    let Some(bc) = module() else { return };
    for (stem, p) in committed_profiles().expect("profiles") {
        for entry in KERNELS {
            match compile(&bc, entry, true, &p) {
                Ok(a) => {
                    assert!(!absent(&a), "{stem} {entry}: {:?}", a.facts);
                    validate(&a, &p);
                }
                Err(Error::NeedsInput(_)) => {}
                Err(e) => panic!("{stem} {entry}: {e}"),
            }
        }
    }
}
