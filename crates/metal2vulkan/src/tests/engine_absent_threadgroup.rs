//! Threadgroup memory a function constant removes, still read and written.
//!
//! Removed threadgroup memory is not declared in the specialized function (MSL 5.8.1.2): what an
//! access through it reads or changes is undefined, and on Metal it varies. When the body reaches
//! the memory only as the address of loads, stores, and atomics (at its own address, or one derived
//! from it), specialization makes each load and atomic with a result an absent read of its type
//! (`FunctionConstant::AbsentMemory`): it takes zero and marks nothing, like a removed texture's
//! read. Each store, and each atomic without a result, is dropped. A run that never reaches a read
//! is Metal's.
//!
//! The kernels below are our own, each gated by `has_tg`, touching the removed memory `tg` only
//! where the run-time flag `out[0]` is nonzero, and writing their result to `out[1]` (7 where the
//! flag is zero):
//!
//! - `at_accesses`: `tg[tid] = 5`, a barrier, `out[1] = tg[tid] + atomic_fetch_add(tg, 1)`;
//! - `at_stores`: `tg[tid] = 5` and `atomic_store(&tg[1], 2)`, then `out[1] = 7`: only writes;
//! - `at_compared`: `out[1] = (tg == null)`, a use that is no access.
//!
//! With `has_tg` false `at_accesses` and `at_stores` compile, on every committed profile, into
//! kernels that mark nothing (no fact, no status push field or binding) and validate: the reads
//! take zero and the writes go. `at_compared` stays refused
//! `FunctionConstantArgument` while specializing. With `has_tg` true each kernel reads its
//! memory, none absent.
//!
//! `llvm-as` assembles the module; without it the tests skip visibly. No GPU is used.

use std::sync::Arc;

use crate::budget::SPIRV_WORDS;
use crate::device::DeviceProfile;
use crate::error::{Error, Feature};
use crate::library::{FunctionArtifact, FunctionConstantValue, FunctionConstantValues};
use crate::metal::DataType;
use crate::pipeline::compile::{compile_bitcode, EntryRef};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::link::{link, LinkInputs};
use crate::pipeline::lower_inputs::LoweringInputs;
use crate::pipeline::relocate::relocate;
use crate::pipeline::testing::{
    llvm_assemble, meter, status_reach, AssembleError, Scratch, StatusReach,
};
use crate::spirv::testing::threadgroup_profiles;
use crate::spirv::testing::{check_words_valid, committed_profiles};

/// The kernels (module docs). Argument 0 is the removed threadgroup memory `tg`, gated by
/// `has_tg`; argument 1 `out`; argument 2 `tid`.
const MODULE: &str = r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@has_tg.MTL_FC_INIT_0_b = linkonce_odr hidden local_unnamed_addr addrspace(2) externally_initialized constant i8 undef, section "air.fc_initializer", align 1
@__metal_implicit_fc_pred_0 = internal addrspace(2) global i8 0, align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_, ptr null }]

define void @at_accesses(ptr addrspace(3) noundef %tg, ptr addrspace(1) noundef %out, i32 noundef %tid) {
entry:
  %k = load i32, ptr addrspace(1) %out, align 4
  %go = icmp ne i32 %k, 0
  br i1 %go, label %access, label %done

access:
  %e = getelementptr inbounds i32, ptr addrspace(3) %tg, i32 %tid
  store i32 5, ptr addrspace(3) %e, align 4
  call void @air.wg.barrier(i32 2, i32 1)
  %x = load i32, ptr addrspace(3) %e, align 4
  %old = call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) %tg, i32 1, i32 0, i32 1, i1 true)
  %s = add i32 %x, %old
  br label %done

done:
  %v = phi i32 [ %s, %access ], [ 7, %entry ]
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

define void @at_stores(ptr addrspace(3) noundef %tg, ptr addrspace(1) noundef %out, i32 noundef %tid) {
entry:
  %k = load i32, ptr addrspace(1) %out, align 4
  %go = icmp ne i32 %k, 0
  br i1 %go, label %access, label %done

access:
  %e = getelementptr inbounds i32, ptr addrspace(3) %tg, i32 %tid
  store i32 5, ptr addrspace(3) %e, align 4
  %f = getelementptr inbounds i32, ptr addrspace(3) %tg, i64 1
  call void @air.atomic.local.store.i32(ptr addrspace(3) %f, i32 2, i32 0, i32 1, i1 true)
  br label %done

done:
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 7, ptr addrspace(1) %o, align 4
  ret void
}

define void @at_compared(ptr addrspace(3) noundef %tg, ptr addrspace(1) noundef %out, i32 noundef %tid) {
entry:
  %k = load i32, ptr addrspace(1) %out, align 4
  %go = icmp ne i32 %k, 0
  br i1 %go, label %access, label %done

access:
  %n = icmp eq ptr addrspace(3) %tg, null
  %w = zext i1 %n to i32
  br label %done

done:
  %v = phi i32 [ %w, %access ], [ 7, %entry ]
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 1
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

declare void @air.wg.barrier(i32, i32)
declare i32 @air.atomic.local.add.u.i32(ptr addrspace(3), i32, i32, i32, i1)
declare void @air.atomic.local.store.i32(ptr addrspace(3), i32, i32, i32, i1)

declare i8 @air.normalize_function_constant_predicate.i8(i8)

define internal void @_GLOBAL__sub_I_() section "air.static_init" {
  %1 = load i8, ptr addrspace(2) @has_tg.MTL_FC_INIT_0_b, align 1
  %2 = tail call i8 @air.normalize_function_constant_predicate.i8(i8 %1)
  store i8 %2, ptr addrspace(2) @__metal_implicit_fc_pred_0, align 1
  ret void
}

!air.kernel = !{!30, !31, !32}
!air.function_constants = !{!10}
!air.version = !{!20}
!air.language_version = !{!21}

!1 = !{}
!5 = !{ptr addrspace(2) @__metal_implicit_fc_pred_0, !"bool", !"has_tg"}
!10 = !{ptr addrspace(2) @has_tg.MTL_FC_INIT_0_b, !"bool", !"has_tg", i32 0}
!11 = !{i32 0, !"air.function_constant", !5, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"tg"}
!12 = !{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!13 = !{i32 2, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!22 = !{!11, !12, !13}
!30 = !{ptr @at_accesses, !1, !22}
!31 = !{ptr @at_stores, !1, !22}
!32 = !{ptr @at_compared, !1, !22}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// [`MODULE`] assembled, or `None` (skipped visibly) without `llvm-as`.
fn module() -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-absent-threadgroup");
    let dir = scratch.path();
    let ll = dir.join("absent_threadgroup.ll");
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

/// `entry` compiled on `p` with `has_tg` set to `on`.
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

/// With `has_tg` false, the loads and the atomic of `at_accesses` are absent reads that take zero,
/// and the writes of `at_stores` go: on every profile neither kernel marks anything (no fact, no
/// status push field or binding), and both validate.
#[test]
fn absent_threadgroup_memory_reads_zero_and_drops_its_writes() {
    let Some(bc) = module() else { return };
    for (stem, p) in committed_profiles().expect("profiles") {
        for entry in ["at_accesses", "at_stores"] {
            let a =
                compile(&bc, entry, false, &p).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            assert!(
                !a.facts.contains(&RuntimeFact::ExecutionStatus),
                "{stem} {entry}: {:?}",
                a.facts
            );
            assert_eq!(status_reach(&a), StatusReach::None, "{stem} {entry}");
            validate(&a, &p);
        }
    }
}

/// Removed threadgroup memory used other than as an access's address stays refused while
/// specializing, on every profile.
#[test]
fn absent_threadgroup_memory_compared_is_refused() {
    let Some(bc) = module() else { return };
    for (stem, p) in committed_profiles().expect("profiles") {
        match compile(&bc, "at_compared", false, &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::FunctionConstantArgument, "{stem}");
            }
            other => panic!("{stem}: {:?}", other.map(|_| ())),
        }
    }
}

/// With `has_tg` true each kernel reads its threadgroup memory: no absent read.
#[test]
fn present_threadgroup_memory_is_read() {
    let Some(bc) = module() else { return };
    for (stem, p) in threadgroup_profiles().expect("profiles") {
        for entry in ["at_accesses", "at_stores", "at_compared"] {
            match compile(&bc, entry, true, &p) {
                Ok(a) => validate(&a, &p),
                Err(Error::NeedsInput(_)) => {}
                Err(e) => panic!("{stem} {entry}: {e}"),
            }
        }
    }
}
