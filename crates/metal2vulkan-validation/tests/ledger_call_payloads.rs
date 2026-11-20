//! The byte ledger carries what an AIR instruction holds outside its slot.
//!
//! A call keeps its payload in a box (the callee, the argument list, the attributes and their
//! lists), and a PHI, GEP, switch, shuffle, aggregate access, or ranged load keeps its operand
//! list in one. Canonicalization rebuilds a body pass by pass and several copies are live at once,
//! so a call-heavy entry's allocator peak lies in canonicalization, before emission. While only
//! the instruction slots were charged, an entry refused before emission (`NeedsInput` at
//! selection) ended with a ledger peak near 0.64 of its allocator peak, under the floor.
//!
//! This binary installs the worker's counting allocator and runs one test, so no other test
//! thread allocates while it measures. The module is generated here: an entry whose inlined
//! helper is a chain of calls carrying function and argument string attributes, and an entry with
//! every other boxed operand list. Decoding it must be charged at least what it allocates and at
//! least its bodies' footprint, every decoded instruction's clone must allocate exactly its
//! `InstKind::heap_bytes`, and planning the call-heavy entry (decode through selection, which is
//! all a compile refused at selection runs) must keep its ledger peak at the floor and, since
//! canonicalization returns each body it frees (`Body::release`), within 5/4 of the allocator
//! peak.

use metal2vulkan::pipeline::{EntryDescriptor, EntryRef, LoweringInputs};
use metal2vulkan::Budget;
use metal2vulkan_air::budget::Meter;
use metal2vulkan_air::ir::InstKind;
use metal2vulkan_devtools::{llvm_assemble, AssembleError};
use metal2vulkan_validation::budgets::ledger_covers;
use metal2vulkan_validation::engine::load_profile;
use metal2vulkan_validation::worker::alloc::{measure, CountingAllocator};

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Calls in the inlined helper's chain.
const CALLS: usize = 512;

/// The module: `call_payloads` loads two floats, passes them to an internal helper that folds
/// them through [`CALLS`] alternating `air.fast_fmin`/`air.fast_fmax` calls, each carrying a
/// string attribute on the call and on both arguments, and stores the result; `operand_lists`
/// holds a ranged load, a call with a `!range` and a return string attribute, a shuffle,
/// aggregate indices, a GEP, a switch, and a PHI.
fn module() -> String {
    let mut chain = String::new();
    let (mut a, mut b) = ("%a".to_owned(), "%b".to_owned());
    for i in 0..CALLS {
        let op = if i % 2 == 0 { "fmin" } else { "fmax" };
        chain.push_str(&format!(
            "  %c{i} = tail call fast float @air.fast_{op}.f32(float \"operand\"=\"first\" {a}, \
             float \"operand\"=\"second\" {b}) #1\n"
        ));
        (a, b) = (format!("%c{i}"), a);
    }
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"
target triple = \"air64_v28-apple-macosx26.0.0\"

define void @call_payloads(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{
entry:
  %a = load float, ptr addrspace(1) %0, align 4
  %p = getelementptr inbounds float, ptr addrspace(1) %0, i64 1
  %b = load float, ptr addrspace(1) %p, align 4
  %r = call fastcc float @chain(float %a, float %b)
  store float %r, ptr addrspace(1) %0, align 4
  ret void
}}

define internal fastcc float @chain(float %a, float %b) {{
entry:
{chain}  ret float {a}
}}

define void @operand_lists(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{
entry:
  %w = load i32, ptr addrspace(1) %0, align 4, !range !30
  %n = call \"result\"=\"bits\" i32 @llvm.ctpop.i32(i32 %w), !range !31
  %v = insertelement <4 x i32> poison, i32 %w, i64 0
  %s = shufflevector <4 x i32> %v, <4 x i32> poison, <4 x i32> <i32 0, i32 0, i32 poison, i32 3>
  %agg = insertvalue {{ i32, [2 x i32] }} poison, i32 %w, 1, 0
  %x = extractvalue {{ i32, [2 x i32] }} %agg, 1, 0
  %e = extractelement <4 x i32> %s, i64 1
  %q = getelementptr inbounds [4 x i32], ptr addrspace(1) %0, i64 0, i64 2
  switch i32 %w, label %join [
    i32 1, label %one
    i32 2, label %two
  ]
one:
  br label %join
two:
  br label %join
join:
  %m = phi i32 [ %x, %entry ], [ %e, %one ], [ %n, %two ]
  store i32 %m, ptr addrspace(1) %q, align 4
  ret void
}}

declare float @air.fast_fmin.f32(float, float) #0
declare float @air.fast_fmax.f32(float, float) #0
declare i32 @llvm.ctpop.i32(i32) #0

attributes #0 = {{ nounwind willreturn memory(none) }}
attributes #1 = {{ nounwind willreturn memory(none) \"call\"=\"payload\" }}

!llvm.module.flags = !{{!0}}
!air.kernel = !{{!20, !21}}
!air.version = !{{!17}}
!air.language_version = !{{!18}}

!0 = !{{i32 7, !\"air.max_device_buffers\", i32 31}}
!10 = !{{}}
!11 = !{{!12}}
!12 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"data\"}}
!17 = !{{i32 2, i32 8, i32 0}}
!18 = !{{!\"Metal\", i32 4, i32 0, i32 0}}
!20 = !{{ptr @call_payloads, !10, !11}}
!21 = !{{ptr @operand_lists, !10, !11}}
!30 = !{{i32 0, i32 100}}
!31 = !{{i32 0, i32 33}}
"
    )
}

/// The module's bitcode, or `None` (skipped visibly) without `llvm-as`.
fn assemble() -> Option<Vec<u8>> {
    let dir = std::env::temp_dir().join(format!("m2v-ledger-calls-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let ll = dir.join("call_payloads.ll");
    std::fs::write(&ll, module()).expect("write");
    let assembled = llvm_assemble(&ll);
    std::fs::remove_dir_all(&dir).expect("remove temp dir");
    match assembled {
        Ok(bitcode) => Some(bitcode),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the module: {why}"),
    }
}

#[test]
fn the_ledger_carries_the_payloads_of_a_call_heavy_body() {
    let Some(bitcode) = assemble() else {
        return;
    };

    // The decoder charges each list of a record before allocating it and a call's two boxes as it
    // builds them, and holds the instruction slots twice at its peak (the record vector and the
    // arena it moves into, both charged), so decoding is charged at least what it allocates. The
    // chain's boxes uncharged would leave it about a quarter short.
    let mut decoding = Meter::new(&Budget::default());
    let (decoded, allocator) = measure(|| metal2vulkan_air::decode(&bitcode, &mut decoding));
    let decoded = decoded.expect("the module decodes");
    let allocator = allocator.expect("the counting allocator is installed");
    assert!(
        decoding.ledger_peak() >= allocator,
        "decoding: ledger peak {} < allocator peak {allocator}",
        decoding.ledger_peak()
    );

    // A decoded body returns at most what decoding charged when a pass frees it.
    let footprints: usize = decoded
        .module
        .parts()
        .funcs
        .values()
        .filter_map(|function| function.body.as_ref())
        .map(|body| body.footprint())
        .sum();
    assert!(
        decoding.ledger() >= footprints as u64,
        "decoding holds {} on the ledger, under its bodies' footprint {footprints}",
        decoding.ledger()
    );

    // `heap_bytes` is what the allocator holds: a clone allocates exactly it.
    let (mut kinds, mut boxed, mut payload, mut ranged_calls) = (0usize, 0usize, 0u64, 0usize);
    for function in decoded.module.parts().funcs.values() {
        for inst in function.body.iter().flat_map(|body| body.insts.values()) {
            let (clone, allocated) = measure(|| inst.kind.clone());
            assert_eq!(clone, inst.kind);
            let allocated = allocated.expect("the counting allocator is installed");
            assert_eq!(
                allocated,
                inst.kind.heap_bytes() as u64,
                "{:?}: a clone allocates its heap bytes",
                inst.kind.opcode()
            );
            kinds += 1;
            boxed += usize::from(allocated > 0);
            payload += allocated;
            ranged_calls += usize::from(matches!(&inst.kind, InstKind::Call(call)
                if call.range.is_some()
                    && call.attrs.as_ref().is_some_and(|a| !a.ret.strings.is_empty())));
        }
    }
    // `operand_lists`' `ctpop`, whose range and return attributes `Call::heap_bytes` counts.
    assert_eq!(
        ranged_calls, 1,
        "one call keeps a range and a return string attribute"
    );
    // The chain's calls, the entries' GEPs, and every list of `operand_lists`.
    assert!(
        boxed > CALLS,
        "{boxed} of {kinds} instructions hold a payload"
    );

    // Planning the call-heavy entry keeps the ledger at the floor of the allocator peak.
    let profile = load_profile("runtime-floor").expect("profile");
    let entry = EntryDescriptor::new(&bitcode, EntryRef::Name("call_payloads"));
    let budget = Budget::default();
    let ((plan, report), allocator) =
        measure(|| profile.plan_entry(&entry, &LoweringInputs::new(), &budget));
    plan.expect("call_payloads plans");
    let allocator = allocator.expect("the counting allocator is installed");
    assert!(
        allocator > payload,
        "the allocator peak {allocator} holds at least one copy of the payloads ({payload})"
    );
    assert!(
        ledger_covers(report.ledger_peak, allocator),
        "ledger peak {} < 7/10 of allocator peak {allocator}",
        report.ledger_peak
    );
    // And it does not keep the bodies canonicalization freed: the ledger peak stays within 5/4 of
    // the allocator peak (1.42 while every replaced body stayed charged).
    assert!(
        u128::from(report.ledger_peak) * 4 <= u128::from(allocator) * 5,
        "ledger peak {} > 5/4 of allocator peak {allocator}",
        report.ledger_peak
    );
}
