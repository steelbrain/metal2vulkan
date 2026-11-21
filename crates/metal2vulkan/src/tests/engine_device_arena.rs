//! A kernel's private arena in the runtime's device memory (`memory::plan::ArenaLayout::device`)
//! under the identity domains, the runtime's.
//!
//! Owned synthetic kernel `branches` keeps a private array of 7176 words (28,704 bytes, past the
//! floor's 16 KiB, so it lives in the runtime's memory): it writes the word its thread ID picks in
//! the entry block, reads one in one arm of a branch, writes one and reads one in the other, and
//! reads one after the branch joins.
//!
//! On every committed profile `Device::from_queries` derives under the identity domains whose
//! plan keeps the arena in device memory, with the default lowering inputs and as a kernel whose
//! threadgroups may be partial, the invocation's slot (`EmitCx::arena_slot`) is derived once in
//! the function, and each push-block or `ArgumentTable` member, the arena's address among them,
//! is loaded once: the floor's tail guard (`push::emit_tail_guard`) reads the tail from the load
//! the first block made, and a hook after the guard's construct recalls the first block's loads.
//!
//! `llvm-as` assembles the kernel; without it the test skips visibly.

use super::engine_dispatch::field_loads;
use super::engine_word_phases::identity_profiles;
use crate::lower::select::LowerInputs;
use crate::pipeline::testing::{compile_with, link_and_validate, typed_module_with, AssembleError};
use crate::spirv::ir::{BinOp, Inst, TypedModule, ValueOrigin};

/// The arena's words.
const WORDS: u32 = 7176;

/// The module holding `branches`.
fn module() -> Option<Vec<u8>> {
    let n = WORDS;
    let text = format!(
        r#"; Owned synthetic kernel: a private array of {n} words reached in several blocks.
source_filename = "device_arena.metal"

define void @branches(ptr addrspace(1) %out, i32 %gid) {{
entry:
  %a = alloca [{n} x i32], align 4
  %w = urem i32 %gid, {n}
  %p = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 %w
  store i32 %gid, ptr %p, align 4
  %c = icmp ult i32 %gid, 4
  br i1 %c, label %left, label %right
left:
  %r1 = mul i32 %gid, 7
  %i1 = urem i32 %r1, {n}
  %q1 = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 %i1
  %x1 = load i32, ptr %q1, align 4
  br label %done
right:
  %r2 = mul i32 %gid, 11
  %i2 = urem i32 %r2, {n}
  %q2 = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 %i2
  store i32 %r2, ptr %q2, align 4
  %x2 = load i32, ptr %p, align 4
  br label %done
done:
  %x = phi i32 [ %x1, %left ], [ %x2, %right ]
  %r3 = mul i32 %gid, 13
  %i3 = urem i32 %r3, {n}
  %q3 = getelementptr inbounds [{n} x i32], ptr %a, i32 0, i32 %i3
  %y = load i32, ptr %q3, align 4
  %s = add i32 %x, %y
  %g = zext i32 %gid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 %s, ptr addrspace(1) %o, align 4
  ret void
}}

!air.kernel = !{{!0}}
!0 = !{{ptr @branches, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
"#
    );
    assemble(&text)
}

/// `text` assembled, or `None` where `llvm-as` is missing.
fn assemble(text: &str) -> Option<Vec<u8>> {
    match metal2vulkan_devtools::llvm_assemble_text(text) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the kernels: {why}"),
    }
}

/// How many times `m` derives the invocation's slot of an arena of `slot_bytes` a slot: the
/// multiplications by the slot's bytes (`dispatch::KernelDecl::arena_slot`, `i · slot_bytes`).
fn slot_derivations(m: &TypedModule, slot_bytes: u32) -> usize {
    let is_bytes = |v| match m.value(v).map(|d| d.origin) {
        Ok(ValueOrigin::Const(c)) => {
            m.consts().scalar_bits(c).map(|b| b.0) == Some(u64::from(slot_bytes))
        }
        _ => false,
    };
    m.blocks()
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|n| {
            matches!(&n.inst, Inst::Binary { op: BinOp::IMul, a, b, .. }
                if is_bytes(*a) || is_bytes(*b))
        })
        .count()
}

/// The slot is derived once and every push field, the arena's address among them, is loaded
/// once, however many blocks reach the arena: with the default lowering inputs, and as a kernel
/// whose threadgroups may be partial (`LowerInputs::partial_threadgroups`), whose entry hooks
/// return past the grid in a construct after the slot is asked for.
#[test]
fn the_arena_slot_is_made_once_and_its_push_field_read_once() {
    let Some(bc) = module() else {
        return;
    };
    let partial = LowerInputs {
        partial_threadgroups: true,
        ..LowerInputs::new()
    };
    let mut checked = 0;
    for (stem, p) in identity_profiles() {
        for (what, inputs) in [("default", &LowerInputs::new()), ("partial", &partial)] {
            let a = compile_with(&bc, "branches", &p, inputs)
                .unwrap_or_else(|e| panic!("{stem} {what}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {what}: L3: {e}"));
            let (plan, m) = typed_module_with(&bc, "branches", &p, inputs)
                .unwrap_or_else(|e| panic!("{stem} {what}: {e}"));
            if !plan.families.access.local.private_device {
                continue;
            }
            checked += 1;
            assert_eq!(slot_derivations(&m, 4 * WORDS), 1, "{stem} {what}");
            let loads = field_loads(&m);
            assert!(!loads.is_empty(), "{stem} {what}: no push field read");
            assert!(loads.values().all(|&n| n == 1), "{stem} {what}: {loads:?}");
        }
    }
    assert!(
        checked > 0,
        "no identity profile keeps the arena in device memory"
    );
}
