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
//! What one access of the arena emits, on a generated module: each shape is two kernels, one that
//! makes the access once and one that makes it twice, everything else equal, as in
//! `engine_access_size`; the second's entry weight (`spirv::roll::entry_weight`) less the first's
//! is what one access costs, its index arithmetic (a `mul` and a `urem` of the thread ID), its
//! containment and the add that keeps its value live included. The shapes: a word load and a word
//! store at a run-time index, a `float4` load (four words) at a run-time index, and a word load
//! at a constant index. Each word is an access chain from the slot's words, placed once
//! (`EmitCx::arena_slot`), and a load or store, so each shape stays within its bound on every
//! identity profile. The table of the first profile is printed.
//!
//! `llvm-as` assembles the kernels; without it the tests skip visibly.

use std::fmt::Write as _;

use super::engine_dispatch::field_loads;
use super::engine_word_phases::identity_profiles;
use crate::lower::select::LowerInputs;
use crate::pipeline::testing::{
    compile_kept, compile_with, link_and_validate, meter, typed_module_with, AssembleError,
};
use crate::spirv::ir::{BinOp, Inst, TypedModule, ValueOrigin};
use crate::spirv::roll::entry_weight;

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

/// One access shape of the arena: its name, the instructions of its `j`th access (`%x{j}` its
/// loaded value), how two values are combined, and the most instructions one access may cost.
struct Shape {
    name: &'static str,
    access: fn(u32) -> String,
    /// The loaded value's type and the instruction combining two of them (`None` for a store).
    value: Option<(&'static str, &'static str)>,
    bound: u64,
}

/// The `j`th access's run-time word index `%k{j}`, below `WORDS - 4`, and its pointer `%q{j}`.
fn run_time_index(j: u32) -> String {
    let n = WORDS - 4;
    let m = 7 + 2 * j;
    format!(
        "  %m{j} = mul i32 %gid, {m}\n  %k{j} = urem i32 %m{j}, {n}\n  %q{j} = getelementptr inbounds [{WORDS} x i32], ptr %a, i32 0, i32 %k{j}\n"
    )
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "word load at a run-time index",
        access: |j| {
            format!(
                "{}  %x{j} = load i32, ptr %q{j}, align 4\n",
                run_time_index(j)
            )
        },
        value: Some(("i32", "add")),
        bound: 12,
    },
    Shape {
        name: "word store at a run-time index",
        access: |j| {
            format!(
                "{}  store i32 %gid, ptr %q{j}, align 4\n",
                run_time_index(j)
            )
        },
        value: None,
        bound: 11,
    },
    Shape {
        name: "float4 load at a run-time index",
        access: |j| {
            format!(
                "{}  %x{j} = load <4 x float>, ptr %q{j}, align 4\n",
                run_time_index(j)
            )
        },
        value: Some(("<4 x float>", "fadd")),
        bound: 23,
    },
    Shape {
        name: "word load at a constant index",
        access: |j| {
            let at = 100 + j;
            format!(
                "  %q{j} = getelementptr inbounds [{WORDS} x i32], ptr %a, i32 0, i32 {at}\n  %x{j} = load i32, ptr %q{j}, align 4\n"
            )
        },
        value: Some(("i32", "add")),
        bound: 3,
    },
];

/// The kernel of `shape` making its access `n` times: its name and its text. Its entry block
/// writes the word the thread ID picks, so the array stays memory, in the runtime's.
fn shape_kernel(s: usize, shape: &Shape, n: u32) -> (String, String) {
    let name = format!("arena{s}_{n}");
    let mut body = format!(
        "  %a = alloca [{WORDS} x i32], align 4\n  %w = urem i32 %gid, {WORDS}\n  %p = getelementptr inbounds [{WORDS} x i32], ptr %a, i32 0, i32 %w\n  store i32 %gid, ptr %p, align 4\n  %g = zext i32 %gid to i64\n"
    );
    for j in 0..n {
        body.push_str(&(shape.access)(j));
    }
    match shape.value {
        Some((ty, op)) => {
            let mut acc = String::from("%x0");
            for j in 1..n {
                let _ = writeln!(body, "  %c{j} = {op} {ty} {acc}, %x{j}");
                acc = format!("%c{j}");
            }
            let _ = writeln!(
                body,
                "  %o = getelementptr inbounds {ty}, ptr addrspace(1) %out, i64 %g\n  store {ty} {acc}, ptr addrspace(1) %o, align 4"
            );
        }
        // The stored words are read back, so the stores are not dead.
        None => {
            let _ = writeln!(
                body,
                "  %y = load i32, ptr %p, align 4\n  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g\n  store i32 %y, ptr addrspace(1) %o, align 4"
            );
        }
    }
    let text = format!(
        "define void @{name}(ptr addrspace(1) %out, i32 %gid) {{\nentry:\n{body}  ret void\n}}\n\n"
    );
    (name, text)
}

/// Every shape's two kernels in one module, with their metadata.
fn shapes_module() -> (String, Vec<(String, String)>) {
    let mut text =
        String::from("; Owned synthetic kernels: one access of a private arena, once and twice.\n");
    let mut names = Vec::new();
    for (s, shape) in SHAPES.iter().enumerate() {
        let (one, a) = shape_kernel(s, shape, 1);
        let (two, b) = shape_kernel(s, shape, 2);
        text.push_str(&a);
        text.push_str(&b);
        names.push((one, two));
    }
    let kernels: Vec<String> = (0..names.len() * 2)
        .map(|k| format!("!{}", 100 + k))
        .collect();
    let _ = writeln!(
        text,
        "!air.kernel = !{{{}}}\n!1 = !{{}}\n!2 = !{{!3, !4}}\n\
         !3 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}}\n\
         !4 = !{{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}}",
        kernels.join(", ")
    );
    for (k, (one, two)) in names.iter().enumerate() {
        let _ = writeln!(text, "!{} = !{{ptr @{one}, !1, !2}}", 100 + 2 * k);
        let _ = writeln!(text, "!{} = !{{ptr @{two}, !1, !2}}", 101 + 2 * k);
    }
    (text, names)
}

/// Each shape's instructions for one access on profile `p`, or `None` where the profile keeps the
/// arena out of device memory.
fn measure(
    bc: &[u8],
    names: &[(String, String)],
    stem: &str,
    p: &crate::device::DeviceProfile,
) -> Option<Vec<u64>> {
    let mut costs = Vec::new();
    for (one, two) in names {
        let mut weights = [0u64; 2];
        for (w, n) in weights.iter_mut().zip([one, two]) {
            let (a, plan, module) =
                compile_kept(bc, n, p).unwrap_or_else(|e| panic!("{stem} {n}: {e}"));
            link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem} {n}: L3: {e}"));
            if !plan.families.access.local.private_device {
                return None;
            }
            *w = entry_weight(&module, &mut meter()).expect("a weight");
        }
        costs.push(weights[1].saturating_sub(weights[0]));
    }
    Some(costs)
}

/// Each access of the arena's words costs at most its shape's bound, on every identity profile
/// that keeps the arena in device memory.
#[test]
fn an_arena_word_is_one_access_chain_and_one_load_or_store() {
    let (text, names) = shapes_module();
    let Some(bc) = assemble(&text) else {
        return;
    };
    let mut printed = false;
    for (stem, p) in identity_profiles() {
        let Some(costs) = measure(&bc, &names, &stem, &p) else {
            continue;
        };
        if !printed {
            printed = true;
            eprintln!("one arena access on {stem}: instructions");
            for (shape, cost) in SHAPES.iter().zip(&costs) {
                eprintln!("  {:<40} {cost:>4}", shape.name);
            }
        }
        for (shape, cost) in SHAPES.iter().zip(&costs) {
            assert!(
                *cost <= shape.bound,
                "{stem}: {} costs {cost} instructions, over {}",
                shape.name,
                shape.bound
            );
        }
    }
    assert!(
        printed,
        "no identity profile keeps the arena in device memory"
    );
}
