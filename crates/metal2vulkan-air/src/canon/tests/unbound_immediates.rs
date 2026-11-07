//! An intrinsic's immediate that is a parameter of the function calling it (Metal's compiler leaves
//! OpenCL C's `barrier(flags)` and `mem_fence(flags)` out of line, passing the flags on): decode
//! keeps the call unbound, and specialization binds each inlined copy with its call site's
//! constants, folding the integer arithmetic between, or refuses it where they are not constants.
//! The modules are LLVM text written here, assembled with `llvm-as`; a run without it skips.

use super::*;
use crate::intrinsics::{BarrierName, MemFlags, MemoryOrder, Sync, ThreadScope};

const HEAD: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n";

/// The ABI of a kernel `@k(ptr addrspace(1) %x)`, `x` a read-write `int` buffer at index 0.
const KERNEL_ABI: &str = "!air.kernel = !{!0}\n\
!air.version = !{!5}\n\
!air.language_version = !{!6}\n\
!0 = !{ptr @k, !1, !2}\n\
!1 = !{}\n\
!2 = !{!3}\n\
!3 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"int\", !\"air.arg_name\", !\"x\"}\n\
!5 = !{i32 2, i32 8, i32 0}\n\
!6 = !{!\"Metal\", i32 3, i32 2, i32 0}\n";

/// `body` (functions and declarations) with the kernel ABI, decoded; `None` without `llvm-as`.
fn decoded(test: &str, body: &str) -> Option<Result<crate::bitcode::Decoded, Error>> {
    let text = format!("{HEAD}{body}{KERNEL_ABI}");
    let bitcode = match metal2vulkan_devtools::llvm_assemble_text(&text) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP {test}: {e:?}");
            return None;
        }
    };
    Some(crate::bitcode::decode(&bitcode, &mut meter()))
}

/// The kernel `@k` of `body` specialized, and the work units specialization metered.
fn specialized(test: &str, body: &str) -> Option<(Result<Entry, Error>, u64)> {
    let d = decoded(test, body)?.expect("decoded");
    let req = SpecRequest {
        entry: "k".into(),
        ..SpecRequest::default()
    };
    let mut m = meter();
    let entry = specialize(d.module, &d.abi, &req, &mut m);
    Some((entry, m.work()))
}

/// The barriers and fences of the specialized entry, in order.
fn syncs(entry: &Entry) -> Vec<Sync> {
    let body = entry
        .module
        .func(entry.func)
        .and_then(|f| f.body.as_ref())
        .expect("the entry's body");
    body.insts
        .values()
        .filter_map(|i| match &i.kind {
            InstKind::Call(call) => match call.callee {
                Callee::Air {
                    op: AirIntrinsic::Sync(s),
                    ..
                } => Some(s),
                _ => None,
            },
            _ => None,
        })
        .collect()
}

fn flags(raw: u64) -> MemFlags {
    MemFlags::from_raw(raw).expect("flags")
}

/// `@wait(flags)`, the barrier its caller's flags name, called from the kernel with `calls`.
fn barrier_module(barrier_flags: &str, calls: &str) -> String {
    format!(
        "define void @k(ptr addrspace(1) %x) #0 {{\n\
         \x20 store i32 1, ptr addrspace(1) %x, align 4\n\
         {calls}\
         \x20 ret void\n}}\n\
         define internal fastcc void @wait(i32 %flags) #1 {{\n\
         \x20 tail call void @air.wg.barrier(i32 {barrier_flags}, i32 1) #2\n\
         \x20 ret void\n}}\n\
         declare void @air.wg.barrier(i32, i32) #2\n\
         attributes #0 = {{ convergent nounwind }}\n\
         attributes #1 = {{ convergent noinline nounwind }}\n\
         attributes #2 = {{ convergent nounwind willreturn }}\n"
    )
}

#[test]
fn a_barrier_whose_flags_are_its_callers_binds_each_call_site_with_its_own() {
    let calls = "  tail call fastcc void @wait(i32 2)\n  tail call fastcc void @wait(i32 1)\n";
    let Some((entry, _)) = specialized(
        "a_barrier_whose_flags_are_its_callers_binds_each_call_site_with_its_own",
        &barrier_module("%flags", calls),
    ) else {
        return;
    };
    let entry = entry.expect("specialized");
    let scope = ThreadScope::from_raw(1).expect("scope");
    let barrier = |raw| Sync::Barrier {
        name: BarrierName::Threadgroup,
        flags: flags(raw),
        order: None,
        scope,
        memory: scope,
    };
    assert_eq!(syncs(&entry), [barrier(2), barrier(1)]);
}

/// Binding is no pass of its own: the kernel whose callee passes its parameter on as the
/// barrier's flags specializes to the module of the kernel whose callee writes the same constant,
/// metering no more work than it.
#[test]
fn binding_a_callers_flags_does_no_work_of_its_own() {
    let test = "binding_a_callers_flags_does_no_work_of_its_own";
    let calls = "  tail call fastcc void @wait(i32 2)\n";
    let (Some((passed, passed_work)), Some((written, written_work))) = (
        specialized(test, &barrier_module("%flags", calls)),
        specialized(test, &barrier_module("2", calls)),
    ) else {
        return;
    };
    let (passed, written) = (passed.expect("passed on"), written.expect("written"));
    assert_eq!(print(&passed.module), print(&written.module));
    assert!(
        passed_work <= written_work,
        "{passed_work} units passing the flags on, {written_work} writing them"
    );
}

#[test]
fn a_fence_whose_flags_its_callers_flags_compute_binds_after_folding() {
    // OpenCL C's `mem_fence(flags)` as Metal's compiler leaves it: CLK_GLOBAL_MEM_FENCE (2)
    // becomes `mem_device` (1) by reversing the two flag bits, an `i2` operation.
    let body = "define void @k(ptr addrspace(1) %x) #0 {\n\
         \x20 store i32 1, ptr addrspace(1) %x, align 4\n\
         \x20 tail call fastcc void @fence(i32 2)\n\
         \x20 store i32 2, ptr addrspace(1) %x, align 4\n\
         \x20 ret void\n}\n\
         define internal fastcc void @fence(i32 %flags) #1 {\n\
         \x20 %t = trunc i32 %flags to i2\n\
         \x20 %r = tail call i2 @llvm.bitreverse.i2(i2 %t)\n\
         \x20 %m = zext i2 %r to i32\n\
         \x20 tail call void @air.atomic.fence(i32 %m, i32 3, i32 1) #2\n\
         \x20 ret void\n}\n\
         declare void @air.atomic.fence(i32, i32, i32) #2\n\
         declare i2 @llvm.bitreverse.i2(i2) #3\n\
         attributes #0 = { nounwind }\n\
         attributes #1 = { noinline nounwind }\n\
         attributes #2 = { nounwind willreturn }\n\
         attributes #3 = { nounwind readnone speculatable willreturn }\n";
    let Some((entry, _)) = specialized(
        "a_fence_whose_flags_its_callers_flags_compute_binds_after_folding",
        body,
    ) else {
        return;
    };
    let entry = entry.expect("specialized");
    let fence = Sync::Fence {
        flags: flags(1),
        order: MemoryOrder::from_raw(3).expect("order"),
        scope: ThreadScope::from_raw(1).expect("scope"),
    };
    assert_eq!(syncs(&entry), [fence]);
}

#[test]
fn a_barrier_whose_flags_the_run_computes_is_refused_at_the_call() {
    // The flags are read from the buffer: no call site makes them a constant.
    let calls =
        "  %f = load i32, ptr addrspace(1) %x, align 4\n  tail call fastcc void @wait(i32 %f)\n";
    let Some((entry, _)) = specialized(
        "a_barrier_whose_flags_the_run_computes_is_refused_at_the_call",
        &barrier_module("%flags", calls),
    ) else {
        return;
    };
    let Err(Error::InvalidInput(e)) = entry else {
        panic!("{entry:?}");
    };
    assert_eq!(
        e.kind,
        InputErrorKind::Intrinsic(crate::error::IntrinsicError::ImmediateNotConstant)
    );
    // At the call in the inlined entry, not at a bitstream position.
    let (f, block, inst) = e.at.in_function().expect("a function location");
    assert_eq!(
        (f.to_string(), block.is_some(), inst.is_some()),
        ("k".to_owned(), true, true)
    );
}
