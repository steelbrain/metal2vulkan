//! Oracle-fuel tests. CPU only: nothing here loads a library on a device.
//!
//! - Synthetic modules (`llvm-as`, skipped visibly where it is absent) check the plan and the
//!   rewrite: which loops get which check, the threaded state, barrier budgets, and the typed
//!   refusals. Every instrumented function is checked for the bound itself: with its check blocks
//!   removed, its CFG is acyclic, so every cycle passes a check; or, where the oracle left counted
//!   loops unchecked, every cycle left is a counted loop within [`ORACLE_NEST_LIMIT`].
//! - Round trips (`xcrun`, skipped visibly where it is absent) write the instrumented module,
//!   assemble it with `metal-as`, decode it again, check the bound again, and hand the AIR to
//!   `metal-opt -passes=verify` and `metallib`, the oracle's next steps: synthetic typed MSL and
//!   every public fixture whose entry reaches a loop.

use std::ffi::OsString;
use std::path::PathBuf;

use metal2vulkan::Budget;
use metal2vulkan_air::bitcode::{Decoded, EnumAttrTable, EnumAttrs};
use metal2vulkan_air::budget::Meter;
use metal2vulkan_air::graph::{Cfg, Graph, LoopForest};
use metal2vulkan_air::intrinsics::{AirIntrinsic, Atomic, AtomicOp};
use metal2vulkan_air::ir::{
    ArgIdx, BlockId, Body, Callee, Const, FuncId, InstId, InstKind, IntPredicate, Md, Module,
    TailKind, Type, Value,
};

use super::plan::barrier_budget;
use super::status::{check_status, initial_status, STATUS_BYTES};
use super::*;
use crate::air_writer::tests::compile_msl;
use crate::air_writer::{decode_as_written, metal_as, write};
use crate::gpu::compile::xcrun;
use crate::ScratchDir;

const SLOT: u32 = 29;

fn config() -> FuelConfig {
    FuelConfig::oracle(SLOT)
}

// ---- Inputs ----

/// Whether `llvm-as` is available; prints a visible skip line when it is not. Under
/// `METAL2VULKAN_TEST_REQUIRE_PINNED_LLVM=1` (CI's Linux job) a missing `llvm-as` fails instead,
/// so these tests cannot pass there by skipping.
fn llvm_as(test: &str) -> bool {
    match crate::check::loop_free::assemble("define void @probe() {\n ret void\n}\n") {
        Ok(_) => true,
        Err(why) => {
            assert!(
                std::env::var_os("METAL2VULKAN_TEST_REQUIRE_PINNED_LLVM").is_none_or(|v| v != "1"),
                "{test}: METAL2VULKAN_TEST_REQUIRE_PINNED_LLVM=1 and llvm-as is unavailable: {why}"
            );
            eprintln!("SKIP {test}: {why}");
            false
        }
    }
}

fn have_xcrun(test: &str) -> bool {
    let ok = cfg!(target_os = "macos") && crate::gpu::compile::has_tool("metal-as");
    if !ok {
        eprintln!("SKIP {test}: xcrun metal-as not found");
    }
    ok
}

/// Decode as the oracle does: keeping the enum attributes the writer spells.
fn decode_bytes(bytes: &[u8]) -> Decoded {
    decode_as_written(bytes, &mut Meter::new(&Budget::default())).expect("decode")
}

/// A kernel `@k(ptr addrspace(1) %out, i32 %n)` with `body` and extra `decls`, assembled with
/// `llvm-as`.
fn kernel(body: &str, extra: &str) -> Decoded {
    decode_bytes(&kernel_bitcode(body, extra))
}

/// The bitcode of [`kernel`].
fn kernel_bitcode(body: &str, extra: &str) -> Vec<u8> {
    let text = format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v27-apple-macosx15.0.0"

define void @k(ptr addrspace(1) %out, i32 %n) {{
{body}
}}

{extra}

declare void @air.wg.barrier(i32, i32)

!air.kernel = !{{!0}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"n"}}
"#
    );
    crate::check::loop_free::assemble(&text).expect("llvm-as")
}

fn run(d: Decoded) -> Result<Instrumented, FuelError> {
    let pointees = Pointees::from_table(&d.pointees, &d.module);
    instrument(d.module, pointees, d.enum_attrs, "k", &config())
}

fn func(m: &Module, name: &str) -> FuncId {
    m.parts()
        .funcs
        .iter()
        .find(|(_, f)| m.symbols().get(f.name) == Some(name.as_bytes()))
        .map(|(id, _)| id)
        .unwrap_or_else(|| panic!("no @{name}"))
}

fn body<'m>(m: &'m Module, name: &str) -> &'m Body {
    m.func(func(m, name)).unwrap().body.as_ref().unwrap()
}

fn param_count(m: &Module, name: &str) -> usize {
    match m.types().get(m.func(func(m, name)).unwrap().ty) {
        Some(Type::Function { params, .. }) => params.len(),
        _ => panic!("not a function type"),
    }
}

// ---- The bound, checked structurally ----

/// The forced-exit blocks of a body: they set `EXHAUSTED` with an atomic `or` and return.
fn exits(b: &Body) -> Vec<BlockId> {
    b.blocks
        .ids()
        .filter(|&blk| {
            let mut insts = b.block_insts(blk).map(|(_, i)| &i.kind);
            let sets = insts.any(|k| {
                matches!(k, InstKind::Call(c) if matches!(c.callee, Callee::Air {
                    op: AirIntrinsic::Atomic(Atomic { op: AtomicOp::Or, .. }),
                    ..
                }))
            });
            sets && matches!(b.terminator(blk).map(|t| &t.kind), Some(InstKind::Ret(_)))
        })
        .collect()
}

/// The checks of a body: blocks that load a counter, compare it with zero, and branch to a
/// forced exit when it is zero.
fn checks(m: &Module, b: &Body) -> Vec<BlockId> {
    let exits = exits(b);
    b.blocks
        .ids()
        .filter(|&blk| {
            let insts: Vec<_> = b.block_insts(blk).map(|(_, i)| &i.kind).collect();
            match insts.as_slice() {
                [InstKind::Load { .. }, InstKind::ICmp {
                    pred: IntPredicate::Eq,
                    a: Value::Inst(_),
                    b: Value::Const(zero),
                }, InstKind::CondBr { t, .. }] => {
                    exits.contains(t)
                        && matches!(m.consts().get(*zero).map(|c| &c.value), Some(Const::Int(0)))
                }
                _ => false,
            }
        })
        .collect()
}

/// Every cycle of `b` passes a check: without the check blocks its CFG is acyclic.
struct Without<'b> {
    body: &'b Body,
    removed: Vec<BlockId>,
}

impl Graph for Without<'_> {
    type Node = BlockId;
    fn node_count(&self) -> usize {
        self.body.blocks.len()
    }
    fn entry(&self) -> BlockId {
        BlockId::new(0)
    }
    fn successors(&self, node: BlockId) -> impl Iterator<Item = BlockId> + '_ {
        let cut = self.removed.contains(&node);
        self.body.successors(node).filter(move |_| !cut)
    }
}

fn assert_bounded(m: &Module, name: &str) -> usize {
    let b = body(m, name);
    let removed = checks(m, b);
    let cfg = Cfg::build(
        &Without {
            body: b,
            removed: removed.clone(),
        },
        &mut Meter::new(&Budget::default()),
    )
    .unwrap();
    let cyclic = cfg
        .preorder()
        .any(|x| cfg.successors(x).any(|y| cfg.is_retreating(x, y)));
    assert!(!cyclic, "@{name} has a cycle that passes no check");
    removed.len()
}

/// The bound of a function the oracle may leave counted loops unchecked in: with its check blocks
/// removed, every loop of its CFG is counted within [`ORACLE_NEST_LIMIT`]. Returns the number of
/// checks.
fn assert_bounded_or_counted(m: &Module, name: &str) -> usize {
    let b = body(m, name);
    let removed = checks(m, b);
    let meter = &mut Meter::new(&Budget::default());
    let cfg = Cfg::build(
        &Without {
            body: b,
            removed: removed.clone(),
        },
        meter,
    )
    .unwrap();
    let forest = LoopForest::compute(&cfg, meter).unwrap();
    let mut preds =
        std::collections::BTreeMap::<BlockId, std::collections::BTreeSet<BlockId>>::new();
    for blk in b.blocks.ids() {
        for s in b.successors(blk) {
            preds.entry(s).or_default().insert(blk);
        }
    }
    for l in forest.loops() {
        assert!(
            super::counted::header_visits(m, b, &forest, &preds, l, ORACLE_NEST_LIMIT).is_some(),
            "@{name} has a cycle that passes no check and is not a counted loop"
        );
    }
    removed.len()
}

// ---- Barrier budgets ----

#[test]
fn barrier_budget_is_the_integer_root_with_floor_two() {
    assert_eq!(barrier_budget(1000, 1), 1000);
    assert_eq!(barrier_budget(1000, 2), 31);
    assert_eq!(barrier_budget(1000, 3), 10);
    assert_eq!(barrier_budget(1024, 10), 2);
    assert_eq!(barrier_budget(1000, 10), 2);
    assert_eq!(barrier_budget(0, 1), 2);
    assert_eq!(barrier_budget(1, 1), 2);
    assert_eq!(barrier_budget(u32::MAX, 1), u32::MAX);
    assert_eq!(barrier_budget(u32::MAX, 2), 65535);
    assert_eq!(barrier_budget(u32::MAX, 64), 2);
    assert_eq!(barrier_budget(7, 0), 7);
    for f in [2u32, 3, 17, 1 << 20, 999_999, u32::MAX] {
        for d in 1..6 {
            let k = u64::from(barrier_budget(f, d));
            if k > 2 {
                assert!(k.pow(d) <= u64::from(f) && (k + 1).pow(d) > u64::from(f));
            }
        }
    }
}

// ---- Status ----

#[test]
fn the_status_buffer_carries_the_case_fuel_and_reports_exhaustion() {
    let bytes = initial_status(77);
    assert_eq!(bytes.len(), STATUS_BYTES);
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect();
    // Status bits clear, no chunk consumed, the pool, the case fuel, and the rest 0: the bytes
    // the oracle wrote when it took the layout from the product's buffer.
    assert_eq!(words, [0, 0, 1 << 20, 77, 0, 0, 0, 0]);
    assert_eq!(check_status(&bytes).unwrap(), Ok(()));
    let mut exhausted = bytes;
    exhausted[0] = 1;
    exhausted[4] = 5;
    // A word the transform never writes is reported as read.
    exhausted[28] = 0xab;
    assert_eq!(
        check_status(&exhausted).unwrap(),
        Err(LoopFuelExhausted {
            consumed_chunks: 5,
            words: [1, 5, 1 << 20, 77, 0, 0, 0, 0xab],
        })
    );
    assert!(check_status(&bytes[..4]).is_err());
    let message = check_status(&exhausted).unwrap().unwrap_err().to_string();
    assert!(message.starts_with("LoopFuelExhausted"), "{message}");
    assert!(
        message.contains("0x00000005") && message.contains("0x000000ab"),
        "{message}"
    );
}

// ---- Synthetic modules ----

const NESTED: &str = r#"entry:
  br label %outer
outer:
  %i = phi i32 [ 0, %entry ], [ %i1, %outer.latch ]
  br label %inner
inner:
  %j = phi i32 [ 0, %outer ], [ %j1, %inner ]
  %j1 = add i32 %j, 1
  %more = icmp ult i32 %j1, %n
  br i1 %more, label %inner, label %outer.latch
outer.latch:
  %i1 = add i32 %i, 1
  %again = icmp ult i32 %i1, %n
  br i1 %again, label %outer, label %done
done:
  store i32 %i1, ptr addrspace(1) %out, align 4
  ret void"#;

#[test]
fn nested_loops_get_a_fuel_check_on_every_latch() {
    if !llvm_as("nested_loops_get_a_fuel_check_on_every_latch") {
        return;
    }
    let out = run(kernel(NESTED, "")).unwrap();
    assert_eq!(
        out.report,
        FuelReport {
            functions: 1,
            loops: 2,
            barrier_loops: 0,
            tail_calls: 0,
            ..FuelReport::default()
        }
    );
    // The status buffer is the entry's new last parameter, a device pointer to `uint`.
    assert_eq!(param_count(&out.module, "k"), 3);
    let k = func(&out.module, "k");
    let status_ty = out
        .module
        .value_ty(k, Value::Arg(metal2vulkan_air::ir::ArgIdx::new(2)));
    assert!(matches!(
        status_ty.and_then(|t| out.module.types().get(t)),
        Some(Type::Ptr {
            space: metal2vulkan_air::ir::AddrSpace::Device
        })
    ));
    assert_eq!(
        out.pointees.param(k, 2),
        out.module.types().lookup(&Type::Int(32))
    );
    assert_eq!(assert_bounded(&out.module, "k"), 2);
    assert_status_argument(&out.module, SLOT, 2);
}

/// The entry's stage argument list names the status buffer at `slot`, argument `index`.
fn assert_status_argument(m: &Module, slot: u32, index: u64) {
    let int = |md: Option<metal2vulkan_air::ir::MdId>| match md.and_then(|md| m.md(md)) {
        Some(Md::Value(c)) => match m.consts().get(*c).map(|c| &c.value) {
            Some(Const::Int(v)) => Some(*v),
            _ => None,
        },
        _ => None,
    };
    let string = |md: Option<metal2vulkan_air::ir::MdId>| match md.and_then(|md| m.md(md)) {
        Some(Md::String(s)) => m.symbols().as_str(*s).map(str::to_owned),
        _ => None,
    };
    let list = m
        .named_md()
        .iter()
        .find(|n| m.symbols().get(n.name) == Some(b"air.kernel".as_slice()))
        .unwrap();
    let Some(Md::Node { ops, .. }) = m.md(list.ops[0]) else {
        panic!("entry node")
    };
    let Some(Md::Node { ops: args, .. }) = m.md(ops[2].unwrap()) else {
        panic!("argument list")
    };
    let Some(Md::Node { ops: arg, .. }) = m.md(args.last().unwrap().unwrap()) else {
        panic!("status argument")
    };
    assert_eq!(int(arg[0]), Some(index));
    assert_eq!(string(arg[1]).as_deref(), Some("air.buffer"));
    assert_eq!(int(arg[3]), Some(u64::from(slot)));
    assert_eq!(string(arg[5]).as_deref(), Some("air.read_write"));
}

const CALLS: &str = r#"entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  %v = call i32 @sum(i32 %i, i32 %n)
  %w = call i32 @plain(i32 %v)
  %i1 = add i32 %i, %w
  %again = icmp ult i32 %i1, %n
  br i1 %again, label %loop, label %done
done:
  %r = call i32 @sum(i32 %i1, i32 3)
  store i32 %r, ptr addrspace(1) %out, align 4
  ret void"#;

const CALLEES: &str = r#"define internal i32 @sum(i32 %a, i32 %n) noinline {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  %s = phi i32 [ %a, %entry ], [ %s1, %loop ]
  %s1 = add i32 %s, %i
  %i1 = add i32 %i, 1
  %again = icmp ult i32 %i1, %n
  br i1 %again, label %loop, label %done
done:
  ret i32 %s1
}

define internal i32 @plain(i32 %a) noinline {
  %b = add i32 %a, 1
  ret i32 %b
}

define internal void @dead(i32 %n) noinline {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  %i1 = add i32 %i, 1
  %again = icmp ult i32 %i1, %n
  br i1 %again, label %loop, label %done
done:
  ret void
}"#;

#[test]
fn calls_in_loops_draw_from_the_callers_fuel() {
    if !llvm_as("calls_in_loops_draw_from_the_callers_fuel") {
        return;
    }
    let original = kernel(CALLS, CALLEES);
    let plain_before = original
        .module
        .func(func(&original.module, "plain"))
        .cloned();
    let dead_before = original
        .module
        .func(func(&original.module, "dead"))
        .cloned();
    let out = run(original).unwrap();
    assert_eq!(
        out.report,
        FuelReport {
            functions: 2,
            loops: 2,
            barrier_loops: 0,
            tail_calls: 0,
            ..FuelReport::default()
        }
    );
    let m = &out.module;
    // @sum takes the state and the status buffer; its forced exit returns zero.
    assert_eq!(param_count(m, "sum"), 4);
    let sum = body(m, "sum");
    let returns_zero = exits(sum).iter().any(|&x| {
        matches!(sum.terminator(x).map(|t| &t.kind), Some(InstKind::Ret(Some(Value::Const(c))))
            if m.consts().get(*c).map(|c| &c.value) == Some(&Const::Null))
    });
    assert!(returns_zero, "a forced exit from @sum returns zero");
    assert_eq!(assert_bounded(m, "sum"), 1);
    assert_eq!(assert_bounded(m, "k"), 1);
    // Both calls to @sum pass the state and test `forced` right after.
    let k = body(m, "k");
    let calls = k
        .insts
        .values()
        .filter(|i| matches!(&i.kind, InstKind::Call(c) if c.callee == Callee::Direct(func(m, "sum")) && c.args.len() == 4))
        .count();
    assert_eq!(calls, 2);
    let k_exits = exits(k);
    let after_call = k
        .blocks
        .ids()
        .filter(|&b| {
            let insts: Vec<_> = k.block_insts(b).map(|(_, i)| &i.kind).collect();
            matches!(insts.as_slice(), [.., InstKind::Call(c), InstKind::Load { .. }, InstKind::ICmp { pred: IntPredicate::Ne, .. }, InstKind::CondBr { t, .. }]
                if c.callee == Callee::Direct(func(m, "sum")) && k_exits.contains(t))
        })
        .count();
    assert_eq!(
        after_call, 2,
        "every call to @sum is followed by the forced test"
    );
    // Loop-free and unexecuted functions are untouched.
    assert_eq!(m.func(func(m, "plain")).cloned(), plain_before);
    assert_eq!(m.func(func(m, "dead")).cloned(), dead_before);
}

const BARRIERS: &str = r#"entry:
  br label %outer
outer:
  %i = phi i32 [ 0, %entry ], [ %i1, %outer.latch ]
  br label %inner
inner:
  %j = phi i32 [ 0, %outer ], [ %j1, %inner ]
  call void @air.wg.barrier(i32 2, i32 1)
  %j1 = add i32 %j, 1
  %more = icmp ult i32 %j1, %n
  br i1 %more, label %inner, label %outer.latch
outer.latch:
  call void @step(i32 %n)
  %i1 = add i32 %i, 1
  %again = icmp ult i32 %i1, %n
  br i1 %again, label %outer, label %free
free:
  %f = phi i32 [ 0, %outer.latch ], [ %f1, %free ]
  %f1 = add i32 %f, 1
  %fmore = icmp ult i32 %f1, %n
  br i1 %fmore, label %free, label %done
done:
  ret void"#;

const BARRIER_CALLEE: &str = r#"define internal void @step(i32 %n) noinline {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  call void @air.wg.barrier(i32 2, i32 1)
  %i1 = add i32 %i, 1
  %again = icmp ult i32 %i1, %n
  br i1 %again, label %loop, label %done
done:
  ret void
}"#;

/// The editor rebuilds a body through `Body::push_inst`: the body is charged exactly its footprint,
/// each payload as it is after the edits, here a call whose arguments `extend_call` lengthened in
/// place.
#[test]
fn an_edited_body_is_charged_exactly_its_footprint() {
    if !llvm_as("an_edited_body_is_charged_exactly_its_footprint") {
        return;
    }
    let decoded = kernel(CALLS, CALLEES);
    let original = body(&decoded.module, "k").clone();
    let (call, arg) = original
        .insts
        .iter()
        .find_map(|(id, inst)| match &inst.kind {
            InstKind::Call(c) => Some((id, *c.args.first()?)),
            _ => None,
        })
        .expect("k calls @sum");
    let mut editor = edit::Editor::new(original);
    editor
        .extend_call(call, &[arg, arg], &std::collections::BTreeMap::new())
        .unwrap();
    let mut meter = Meter::new(&Budget::default());
    let (edited, _) = editor.finish(&mut meter).expect("the body is rebuilt");
    let lengthened = edited
        .insts
        .values()
        .any(|inst| matches!(&inst.kind, InstKind::Call(c) if c.args.len() == 4));
    assert!(lengthened, "the call keeps its two added arguments");
    assert_eq!(meter.ledger(), edited.footprint() as u64);
}

/// Every `store i32 <constant>` to a counter in `b`: the barrier budgets its resets write.
fn reset_budgets(m: &Module, b: &Body) -> Vec<u64> {
    let mut out: Vec<u64> = b
        .blocks
        .ids()
        .filter_map(|blk| {
            let insts: Vec<_> = b.block_insts(blk).map(|(_, i)| &i.kind).collect();
            match insts.as_slice() {
                [InstKind::Store {
                    val: Value::Const(c),
                    ..
                }, InstKind::Br { .. }] => match m.consts().get(*c).map(|c| &c.value) {
                    Some(Const::Int(v)) => Some(*v),
                    _ => None,
                },
                _ => None,
            }
        })
        .collect();
    out.sort_unstable();
    out
}

#[test]
fn barrier_loops_count_uniformly_by_their_nesting_depth() {
    if !llvm_as("barrier_loops_count_uniformly_by_their_nesting_depth") {
        return;
    }
    let out = run(kernel(BARRIERS, BARRIER_CALLEE)).unwrap();
    // `outer` holds a barrier loop and calls @step, whose loop reaches a barrier: three barrier
    // loops on chains of length 2 (outer -> inner, outer -> @step's loop). `free` follows them
    // and reaches none.
    assert_eq!(
        out.report,
        FuelReport {
            functions: 2,
            loops: 1,
            barrier_loops: 3,
            tail_calls: 0,
            ..FuelReport::default()
        }
    );
    let f_b = config().barrier_fuel;
    let two = u64::from(barrier_budget(f_b, 2));
    assert_eq!(
        reset_budgets(&out.module, body(&out.module, "k")),
        vec![two, two]
    );
    assert_eq!(
        reset_budgets(&out.module, body(&out.module, "step")),
        vec![two]
    );
    assert_eq!(assert_bounded(&out.module, "k"), 3);
    assert_eq!(assert_bounded(&out.module, "step"), 1);
}

#[test]
fn an_irreducible_loop_is_bounded_unless_it_reaches_a_barrier() {
    if !llvm_as("an_irreducible_loop_is_bounded_unless_it_reaches_a_barrier") {
        return;
    }
    let ping_pong = |barrier: &str| {
        format!(
            r#"entry:
  %c = icmp ult i32 %n, 7
  br i1 %c, label %a, label %b
a:
  %x = phi i32 [ 0, %entry ], [ %y1, %b ]
  %x1 = add i32 %x, 1
  %xa = icmp ult i32 %x1, %n
  br i1 %xa, label %b, label %done
b:
  %y = phi i32 [ 1, %entry ], [ %x1, %a ]
  {barrier}
  %y1 = add i32 %y, 2
  %yb = icmp ult i32 %y1, %n
  br i1 %yb, label %a, label %done
done:
  ret void"#
        )
    };
    let out = run(kernel(&ping_pong(""), "")).unwrap();
    assert_eq!(out.report.loops, 1);
    assert_eq!(assert_bounded(&out.module, "k"), 1);
    let refused = run(kernel(
        &ping_pong("call void @air.wg.barrier(i32 2, i32 1)"),
        "",
    ));
    assert!(
        matches!(
            refused,
            Err(FuelError::Unsupported(FuelUnsupported::IrreducibleBarrierLoop { ref function })) if function == "k"
        ),
        "{:?}",
        refused.err()
    );
}

#[test]
fn unthreadable_loops_and_call_cycles_are_refused() {
    if !llvm_as("unthreadable_loops_and_call_cycles_are_refused") {
        return;
    }
    // A looping function whose address is taken cannot take the state.
    let taken = kernel(
        "entry:\n  store ptr @spin, ptr addrspace(1) %out, align 8\n  call void @spin(i32 %n)\n  ret void",
        "define internal void @spin(i32 %n) noinline {\nentry:\n  br label %l\nl:\n  %i = phi i32 [ 0, %entry ], [ %i1, %l ]\n  %i1 = add i32 %i, 1\n  %c = icmp ult i32 %i1, %n\n  br i1 %c, label %l, label %d\nd:\n  ret void\n}",
    );
    assert!(matches!(
        run(taken),
        Err(FuelError::Unsupported(FuelUnsupported::UnthreadableLoop { ref function })) if function == "spin"
    ));
    // A function outside the plan (reached only through its escaping address) cannot pass the
    // state to a looping function it calls directly.
    let relay = kernel(
        "entry:\n  store ptr @relay, ptr addrspace(1) %out, align 8\n  call void @spin(i32 %n)\n  ret void",
        "define internal void @relay(i32 %n) noinline {\n  call void @spin(i32 %n)\n  ret void\n}\ndefine internal void @spin(i32 %n) noinline {\nentry:\n  br label %l\nl:\n  %i = phi i32 [ 0, %entry ], [ %i1, %l ]\n  %i1 = add i32 %i, 1\n  %c = icmp ult i32 %i1, %n\n  br i1 %c, label %l, label %d\nd:\n  ret void\n}",
    );
    assert!(matches!(
        run(relay),
        Err(FuelError::Unsupported(FuelUnsupported::UnthreadableLoop { ref function })) if function == "spin"
    ));
    // Recursion is a call cycle.
    let recursive = kernel(
        "entry:\n  call void @r(i32 %n)\n  ret void",
        "define internal void @r(i32 %n) noinline {\n  %c = icmp eq i32 %n, 0\n  br i1 %c, label %d, label %again\nagain:\n  %m = sub i32 %n, 1\n  call void @r(i32 %m)\n  br label %d\nd:\n  ret void\n}",
    );
    assert!(matches!(
        run(recursive),
        Err(FuelError::Unsupported(FuelUnsupported::CallCycle { ref function })) if function == "r"
    ));
    // An entry the module does not define.
    let d = kernel(NESTED, "");
    let pointees = Pointees::from_table(&d.pointees, &d.module);
    assert!(matches!(
        instrument(d.module, pointees, d.enum_attrs, "missing", &config()),
        Err(FuelError::Unsupported(FuelUnsupported::MissingEntry { .. }))
    ));
}

/// A closed indirect call: `%s1` chooses `@spin` (a loop), `@flat` or nothing, is tested against
/// null, and is called; `extra` is spliced in before the test. The entry stores `%n` to `%out`,
/// which types the buffer as its metadata declares (`uint`).
fn chooser(extra: &str) -> String {
    format!(
        "entry:\n  store i32 %n, ptr addrspace(1) %out, align 4\n  %p = icmp ult i32 %n, 7\n  %q = icmp ugt i32 %n, 100\n\
  %s0 = select i1 %p, ptr @spin, ptr null\n  %s1 = select i1 %q, ptr @flat, ptr %s0\n{extra}\
  %none = icmp eq ptr %s1, null\n  br i1 %none, label %done, label %call\n\
call:\n  call void %s1(i32 %n)\n  br label %done\ndone:\n  ret void"
    )
}

const CHOSEN: &str = "define internal void @spin(i32 %n) noinline {\nentry:\n  br label %l\nl:\n  %i = phi i32 [ 0, %entry ], [ %i1, %l ]\n  %i1 = add i32 %i, 1\n  %c = icmp ult i32 %i1, %n\n  br i1 %c, label %l, label %d\nd:\n  ret void\n}\ndefine internal void @flat(i32 %n) noinline {\n  ret void\n}";

#[test]
fn a_closed_indirect_call_passes_the_state_to_every_target() {
    if !llvm_as("a_closed_indirect_call_passes_the_state_to_every_target") {
        return;
    }
    let out = run(kernel(&chooser(""), CHOSEN)).unwrap();
    assert_eq!(
        out.report,
        FuelReport {
            functions: 3,
            loops: 1,
            barrier_loops: 0,
            tail_calls: 0,
            ..FuelReport::default()
        }
    );
    let m = &out.module;
    // Both targets take the state and the status buffer, @flat although it has no loop, so the
    // call's targets keep one prototype, which the call now names.
    assert_eq!(param_count(m, "spin"), 3);
    assert_eq!(param_count(m, "flat"), 3);
    let grown = m.func(func(m, "spin")).unwrap().ty;
    assert_eq!(m.func(func(m, "flat")).unwrap().ty, grown);
    assert_eq!(assert_bounded(m, "spin"), 1);
    let k = body(m, "k");
    let k_exits = exits(k);
    let tested = k
        .blocks
        .ids()
        .filter(|&b| {
            let insts: Vec<_> = k.block_insts(b).map(|(_, i)| &i.kind).collect();
            matches!(insts.as_slice(), [.., InstKind::Call(c), InstKind::Load { .. }, InstKind::ICmp { pred: IntPredicate::Ne, .. }, InstKind::CondBr { t, .. }]
                if matches!(c.callee, Callee::Indirect { fn_ty, .. } if fn_ty == grown)
                    && c.args.len() == 3
                    && k_exits.contains(t))
        })
        .count();
    assert_eq!(
        tested, 1,
        "the call passes the state and is followed by the forced test"
    );
    // A target whose address escapes cannot take the state.
    let stored = kernel(
        &chooser("  store ptr @flat, ptr addrspace(1) %out, align 8\n"),
        CHOSEN,
    );
    assert!(matches!(
        run(stored),
        Err(FuelError::Unsupported(FuelUnsupported::UnthreadableLoop { ref function })) if function == "flat"
    ));
    // Nor can any leaf of a choice that escapes.
    let open = kernel(
        &chooser("  store ptr %s1, ptr addrspace(1) %out, align 8\n"),
        CHOSEN,
    );
    assert!(matches!(
        run(open),
        Err(FuelError::Unsupported(FuelUnsupported::UnthreadableLoop { ref function })) if function == "spin"
    ));
    // Apple's toolchain takes the call through the grown prototype (the module names its AIR
    // and language versions, as `metal-opt` requires).
    if have_xcrun("a_closed_indirect_call_passes_the_state_to_every_target") {
        let versions = "!air.version = !{!50}\n!air.language_version = !{!51}\n!50 = !{i32 2, i32 7, i32 0}\n!51 = !{!\"Metal\", i32 3, i32 2, i32 0}\n\
!llvm.module.flags = !{!52, !53, !54, !55, !56, !57}\n\
!52 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n!53 = !{i32 7, !\"air.max_constant_buffers\", i32 31}\n\
!54 = !{i32 7, !\"air.max_threadgroup_buffers\", i32 31}\n!55 = !{i32 7, !\"air.max_textures\", i32 128}\n\
!56 = !{i32 7, !\"air.max_read_write_textures\", i32 8}\n!57 = !{i32 7, !\"air.max_samplers\", i32 16}";
        let out = run(kernel(&chooser(""), &format!("{CHOSEN}\n{versions}"))).unwrap();
        let scratch = ScratchDir::new("oracle-fuel-closed-call").unwrap();
        round_trip(&out, &scratch).unwrap();
    }
}

/// An entry with a loop of its own that first calls `@spin`, which has another.
const ENTRY_AND_CALLEE_LOOPS: &str = "entry:\n  store i32 %n, ptr addrspace(1) %out, align 4\n  call void @spin(i32 %n)\n  br label %l\nl:\n  %i = phi i32 [ 0, %entry ], [ %i1, %l ]\n  %i1 = add i32 %i, 1\n  %c = icmp ult i32 %i1, %n\n  br i1 %c, label %l, label %d\nd:\n  ret void";

#[test]
fn a_compile_probe_checks_only_the_loops_it_selects() {
    if !llvm_as("a_compile_probe_checks_only_the_loops_it_selects") {
        return;
    }
    let spin = "define internal void @spin(i32 %n) noinline {\nentry:\n  br label %l\nl:\n  %i = phi i32 [ 0, %entry ], [ %i1, %l ]\n  %i1 = add i32 %i, 1\n  %c = icmp ult i32 %i1, %n\n  br i1 %c, label %l, label %d\nd:\n  ret void\n}";
    let probe = |loops| {
        let d = kernel(ENTRY_AND_CALLEE_LOOPS, spin);
        let pointees = Pointees::from_table(&d.pointees, &d.module);
        let config = FuelConfig { loops, ..config() };
        instrument(d.module, pointees, d.enum_attrs, "k", &config).unwrap()
    };
    let report = |functions, loops| FuelReport {
        functions,
        loops,
        barrier_loops: 0,
        tail_calls: 0,
        ..FuelReport::default()
    };
    // The oracle's own: both loops checked, the state threaded to @spin.
    let all = probe(LoopSelection::All);
    assert_eq!(all.report, report(2, 2));
    assert_eq!(assert_bounded(&all.module, "k"), 1);
    assert_eq!(assert_bounded(&all.module, "spin"), 1);
    assert_eq!(param_count(&all.module, "spin"), 3);
    // The entry's loop only: @spin runs as compiled, with its prototype unchanged.
    let entry = probe(LoopSelection::Entry);
    assert_eq!(entry.report, report(1, 1));
    assert_eq!(assert_bounded(&entry.module, "k"), 1);
    assert_eq!(checks(&entry.module, body(&entry.module, "spin")).len(), 0);
    assert_eq!(param_count(&entry.module, "spin"), 1);
    // The callees' loops only: the entry passes the state to @spin and checks nothing itself.
    let callees = probe(LoopSelection::Callees);
    assert_eq!(callees.report, report(2, 1));
    assert_eq!(checks(&callees.module, body(&callees.module, "k")).len(), 0);
    assert_eq!(assert_bounded(&callees.module, "spin"), 1);
    assert_eq!(param_count(&callees.module, "spin"), 3);
}

/// Seven loops in a row: three counted, three not, one with a barrier.
///
/// - `a`: tested at its latch on the sum, `0, 1, ...` until `i + 1 < 16` fails (counted, 16).
/// - `b`: tested at its header on the phi, exits when it equals 8 (counted, 9 visits).
/// - `c`: bounded by the argument `%n` (not counted).
/// - `d`: `sub nsw` by 3 from 10 while the result is positive (counted: 10, 7, 4, 1, -2).
/// - `e`: an `i8` from 250 by `add nuw` 1 until it is 3: the sum wraps at 255, which `nuw`
///   makes poison (not counted; without the flag it would be).
/// - `f`: by 1 until 100000000, past the limit (not counted).
/// - `g`: tested on a constant on the left, exits once the sum passes 4 (counted, 5 visits); it
///   reaches a barrier.
const LOOP_KINDS: &str = "entry:\n  br label %a\n\
a:\n  %i = phi i32 [ 0, %entry ], [ %i1, %a ]\n  %i1 = add nuw nsw i32 %i, 1\n  %ca = icmp ult i32 %i1, 16\n  br i1 %ca, label %a, label %bh\n\
bh:\n  %j = phi i32 [ 0, %a ], [ %j1, %bb ]\n  %cb = icmp eq i32 %j, 8\n  br i1 %cb, label %c, label %bb\n\
bb:\n  %j1 = add nuw i32 %j, 1\n  br label %bh\n\
c:\n  %x = phi i32 [ 0, %bh ], [ %x1, %c ]\n  %x1 = add i32 %x, 1\n  %cc = icmp ult i32 %x1, %n\n  br i1 %cc, label %c, label %d\n\
d:\n  %k = phi i32 [ 10, %c ], [ %k1, %d ]\n  %k1 = sub nsw i32 %k, 3\n  %cd = icmp sgt i32 %k1, 0\n  br i1 %cd, label %d, label %e\n\
e:\n  %m = phi i8 [ -6, %d ], [ %m1, %e ]\n  %m1 = add nuw i8 %m, 1\n  %ce = icmp ne i8 %m1, 3\n  br i1 %ce, label %e, label %f\n\
f:\n  %y = phi i32 [ 0, %e ], [ %y1, %f ]\n  %y1 = add i32 %y, 1\n  %cf = icmp ult i32 %y1, 100000000\n  br i1 %cf, label %f, label %g\n\
g:\n  %z = phi i32 [ 0, %f ], [ %z1, %g ]\n  call void @air.wg.barrier(i32 2, i32 1)\n  %z1 = add i32 %z, 1\n  %cg = icmp ult i32 4, %z1\n  br i1 %cg, label %done, label %g\n\
done:\n  store i32 %i1, ptr addrspace(1) %out, align 4\n  ret void";

#[test]
fn a_compile_probe_selects_loops_by_kind() {
    if !llvm_as("a_compile_probe_selects_loops_by_kind") {
        return;
    }
    let probe = |loops| {
        let d = kernel(LOOP_KINDS, "");
        let pointees = Pointees::from_table(&d.pointees, &d.module);
        let config = FuelConfig { loops, ..config() };
        instrument(d.module, pointees, d.enum_attrs, "k", &config)
            .unwrap()
            .report
    };
    let report = |loops, barrier_loops| FuelReport {
        functions: 1,
        loops,
        barrier_loops,
        tail_calls: 0,
        ..FuelReport::default()
    };
    assert_eq!(probe(LoopSelection::All), report(6, 1));
    assert_eq!(probe(LoopSelection::Barrier), report(0, 1));
    assert_eq!(probe(LoopSelection::NonBarrier), report(6, 0));
    // a, b, d and g are counted; c, e and f are not.
    assert_eq!(probe(LoopSelection::Counted), report(3, 1));
    assert_eq!(probe(LoopSelection::Uncounted), report(3, 0));
}

/// The oracle checks only [`LOOP_KINDS`]'s uncounted loops: the four counted ones, the barrier loop
/// `g` among them, are one deep and within [`ORACLE_NEST_LIMIT`], so they run as written; the
/// largest is `a`'s 16 visits.
#[test]
fn the_oracle_leaves_counted_loops_unchecked() {
    if !llvm_as("the_oracle_leaves_counted_loops_unchecked") {
        return;
    }
    let out = run(kernel(LOOP_KINDS, "")).unwrap();
    assert_eq!(
        out.report,
        FuelReport {
            functions: 1,
            loops: 3,
            barrier_loops: 0,
            tail_calls: 0,
            exempt_loops: 4,
            exempt_nest: 16,
        }
    );
    // One check on each uncounted loop's latch; the counted loops' cycles carry none.
    assert_eq!(assert_bounded_or_counted(&out.module, "k"), 3);
}

/// Two counted loops, `outer` visiting its header `outer_visits` times and `inner` (nested in
/// it) `inner_visits` times per entry, both tested at their latch.
fn counted_nest(outer_visits: u32, inner_visits: u32) -> String {
    format!(
        "entry:\n  br label %outer\n\
         outer:\n  %i = phi i32 [ 0, %entry ], [ %i1, %outer.latch ]\n  br label %inner\n\
         inner:\n  %j = phi i32 [ 0, %outer ], [ %j1, %inner ]\n  \
         %j1 = add nuw nsw i32 %j, 1\n  %cj = icmp ult i32 %j1, {inner_visits}\n  \
         br i1 %cj, label %inner, label %outer.latch\n\
         outer.latch:\n  %i1 = add nuw nsw i32 %i, 1\n  \
         %ci = icmp ult i32 %i1, {outer_visits}\n  br i1 %ci, label %outer, label %done\n\
         done:\n  store i32 %i1, ptr addrspace(1) %out, align 4\n  ret void"
    )
}

/// A nest is exempt while the product of its header visits stays within [`ORACLE_NEST_LIMIT`];
/// past it, the inner loop stays exempt and the outer one is checked, once per inner run.
#[test]
fn a_counted_nest_is_exempt_up_to_the_limit_on_its_product() {
    if !llvm_as("a_counted_nest_is_exempt_up_to_the_limit_on_its_product") {
        return;
    }
    // 32 x 32 = 2^10: nothing needs the state, so the module runs as written.
    let within = run(kernel(&counted_nest(32, 32), "")).unwrap();
    assert_eq!(
        within.report,
        FuelReport {
            exempt_loops: 2,
            exempt_nest: ORACLE_NEST_LIMIT,
            ..FuelReport::default()
        }
    );
    assert_eq!(param_count(&within.module, "k"), 2);
    // 64 x 32 = 2^11: the outer loop is checked.
    let past = run(kernel(&counted_nest(64, 32), "")).unwrap();
    assert_eq!(
        past.report,
        FuelReport {
            functions: 1,
            loops: 1,
            exempt_loops: 1,
            exempt_nest: 32,
            ..FuelReport::default()
        }
    );
    assert_eq!(
        assert_bounded_or_counted(&past.module, "k"),
        1,
        "one check, on the outer loop's latch"
    );
    // A loop past the limit on its own is checked.
    let alone = run(kernel(&counted_nest(1, 2048), "")).unwrap();
    assert_eq!((alone.report.loops, alone.report.exempt_loops), (1, 1));
    assert_eq!(alone.report.exempt_nest, 1);
    assert_eq!(assert_bounded_or_counted(&alone.module, "k"), 1);
}

/// The product runs across calls: a counted loop calling a function whose counted loop is exempt
/// multiplies by its visits; and through a checked loop, whose first pass takes no back edge, so
/// draws no fuel.
#[test]
fn a_counted_nest_counts_across_calls_and_checked_loops() {
    if !llvm_as("a_counted_nest_counts_across_calls_and_checked_loops") {
        return;
    }
    let caller = |visits: u32| {
        format!(
            "entry:\n  br label %l\n\
             l:\n  %i = phi i32 [ 0, %entry ], [ %i1, %l ]\n  \
             call void @row(ptr addrspace(1) %out)\n  %i1 = add nuw nsw i32 %i, 1\n  \
             %c = icmp ult i32 %i1, {visits}\n  br i1 %c, label %l, label %done\n\
             done:\n  ret void"
        )
    };
    let row = "define internal void @row(ptr addrspace(1) %out) noinline {\nentry:\n  \
               br label %l\nl:\n  %j = phi i32 [ 0, %entry ], [ %j1, %l ]\n  \
               store i32 %j, ptr addrspace(1) %out, align 4\n  %j1 = add nuw nsw i32 %j, 1\n  \
               %c = icmp ult i32 %j1, 32\n  br i1 %c, label %l, label %d\nd:\n  ret void\n}";
    let within = run(kernel(&caller(32), row)).unwrap();
    assert_eq!(
        (
            within.report.loops,
            within.report.exempt_loops,
            within.report.exempt_nest
        ),
        (0, 2, ORACLE_NEST_LIMIT)
    );
    let past = run(kernel(&caller(64), row)).unwrap();
    assert_eq!(
        past.report,
        FuelReport {
            functions: 1,
            loops: 1,
            exempt_loops: 1,
            exempt_nest: 32,
            ..FuelReport::default()
        }
    );
    // @row keeps its prototype: it needs no state.
    assert_eq!(param_count(&past.module, "row"), 1);
    assert_eq!(assert_bounded(&past.module, "k"), 1);

    // A loop bounded by %n between two counted ones: counted 16 or 32, then %n, then 64.
    let through = |outer: u32| {
        format!(
            "entry:\n  br label %o\n\
             o:\n  %i = phi i32 [ 0, %entry ], [ %i1, %o.latch ]\n  br label %m\n\
             m:\n  %x = phi i32 [ 0, %o ], [ %x1, %m.latch ]\n  br label %in\n\
             in:\n  %j = phi i32 [ 0, %m ], [ %j1, %in ]\n  %j1 = add nuw nsw i32 %j, 1\n  \
             %cj = icmp ult i32 %j1, 64\n  br i1 %cj, label %in, label %m.latch\n\
             m.latch:\n  %x1 = add i32 %x, 1\n  %cx = icmp ult i32 %x1, %n\n  \
             br i1 %cx, label %m, label %o.latch\n\
             o.latch:\n  %i1 = add nuw nsw i32 %i, 1\n  %ci = icmp ult i32 %i1, {outer}\n  \
             br i1 %ci, label %o, label %done\n\
             done:\n  store i32 %i1, ptr addrspace(1) %out, align 4\n  ret void"
        )
    };
    let fits = run(kernel(&through(16), "")).unwrap();
    assert_eq!(
        (
            fits.report.loops,
            fits.report.exempt_loops,
            fits.report.exempt_nest
        ),
        (1, 2, ORACLE_NEST_LIMIT)
    );
    let over = run(kernel(&through(32), "")).unwrap();
    assert_eq!(
        (
            over.report.loops,
            over.report.exempt_loops,
            over.report.exempt_nest
        ),
        (2, 1, 64)
    );
    assert_eq!(assert_bounded_or_counted(&over.module, "k"), 2);
    assert_eq!(assert_bounded_or_counted(&fits.module, "k"), 1);
}

/// Without `nuw`, loop `e` of [`LOOP_KINDS`] wraps at 255 and then reaches 3: counted.
#[test]
fn a_loop_that_wraps_without_a_flag_is_counted() {
    if !llvm_as("a_loop_that_wraps_without_a_flag_is_counted") {
        return;
    }
    let body = LOOP_KINDS.replace("%m1 = add nuw i8", "%m1 = add i8");
    let d = kernel(&body, "");
    let pointees = Pointees::from_table(&d.pointees, &d.module);
    let config = FuelConfig {
        loops: LoopSelection::Uncounted,
        ..config()
    };
    let report = instrument(d.module, pointees, d.enum_attrs, "k", &config)
        .unwrap()
        .report;
    assert_eq!((report.loops, report.barrier_loops), (2, 0));
}

#[test]
fn an_entry_without_a_stage_argument_list_is_refused() {
    if !llvm_as("an_entry_without_a_stage_argument_list_is_refused") {
        return;
    }
    let text = "define void @k(i32 %n) {\nentry:\n  br label %l\nl:\n  %i = phi i32 [ 0, %entry ], [ %i1, %l ]\n  %i1 = add i32 %i, 1\n  %c = icmp ult i32 %i1, %n\n  br i1 %c, label %l, label %d\nd:\n  ret void\n}\n";
    let d = decode_bytes(&crate::check::loop_free::assemble(text).unwrap());
    let pointees = Pointees::from_table(&d.pointees, &d.module);
    assert!(matches!(
        instrument(d.module, pointees, d.enum_attrs, "k", &config()),
        Err(FuelError::Unsupported(
            FuelUnsupported::EntryNotAStage { .. }
        ))
    ));
}

// ---- Tail markers ----

/// The entry passes its fuel state to a looping callee through a `tail` call, the shape of the
/// MPS filter case Metal 26.7 refused (`tail call fastcc` of the helper, then `ret`), beside a
/// `notail` call to the same callee and a `tail` call to a loop-free one.
const TAIL_CALLS: &str = r#"entry:
  tail call fastcc void @fill(ptr addrspace(1) %out, i32 %n)
  notail call fastcc void @fill(ptr addrspace(1) %out, i32 %n)
  tail call void @mark(ptr addrspace(1) %out)
  ret void"#;

const TAIL_CALLEES: &str = r#"define internal fastcc void @fill(ptr addrspace(1) %out, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i32 %i
  store i32 %i, ptr addrspace(1) %p, align 4
  %i1 = add i32 %i, 1
  %again = icmp ult i32 %i1, %n
  br i1 %again, label %loop, label %done
done:
  ret void
}

define internal void @mark(ptr addrspace(1) %out) {
  store i32 1, ptr addrspace(1) %out, align 4
  ret void
}"#;

/// The tail markers and argument counts of the calls `caller` makes to `callee`, in layout order.
fn calls_to(m: &Module, caller: &str, callee: &str) -> Vec<(TailKind, usize)> {
    let target = Callee::Direct(func(m, callee));
    body(m, caller)
        .insts
        .values()
        .filter_map(|i| match &i.kind {
            InstKind::Call(c) if c.callee == target => Some((c.tail, c.args.len())),
            _ => None,
        })
        .collect()
}

/// `tail` promises that the callee accesses none of the caller's allocas, and the fuel state is
/// the entry's alloca: every call that passes the state loses the marker, or the Metal toolchain
/// deletes the state's initialization and the callee reads an uninitialized fuel word. `notail`
/// promises nothing about the callee's accesses, and a call that passes no state keeps its marker.
#[test]
fn a_call_that_passes_the_fuel_state_drops_its_tail_marker() {
    if !llvm_as("a_call_that_passes_the_fuel_state_drops_its_tail_marker") {
        return;
    }
    let original = kernel(TAIL_CALLS, TAIL_CALLEES);
    assert_eq!(
        calls_to(&original.module, "k", "fill"),
        [(TailKind::Tail, 2), (TailKind::NoTail, 2)]
    );
    let out = run(original).unwrap();
    assert_eq!(
        out.report,
        FuelReport {
            functions: 2,
            loops: 1,
            barrier_loops: 0,
            tail_calls: 1,
            ..FuelReport::default()
        }
    );
    let m = &out.module;
    assert_eq!(
        calls_to(m, "k", "fill"),
        [(TailKind::None, 4), (TailKind::NoTail, 4)]
    );
    assert_eq!(calls_to(m, "k", "mark"), [(TailKind::Tail, 1)]);
    assert_eq!(assert_bounded(m, "fill"), 1);
}

/// A function that takes the fuel state writes it and the status buffer through volatile atomics,
/// so `speculatable` (no side effects) and `nosync` (volatile accesses synchronize) stop being
/// true of it and of every call that passes it the state, as its memory effects do; they are
/// dropped. Every other attribute stays as written: `nounwind`, `willreturn`, and `mustprogress`
/// here, and everything on the loop-free `@mark` and its call.
#[test]
fn the_fuel_state_drops_only_the_attributes_it_falsifies() {
    if !llvm_as("the_fuel_state_drops_only_the_attributes_it_falsifies") {
        return;
    }
    let attrs = "mustprogress nofree nosync nounwind speculatable willreturn";
    let callees = TAIL_CALLEES
        .replace(
            "define internal fastcc void @fill(ptr addrspace(1) %out, i32 %n) {",
            &format!("define internal fastcc void @fill(ptr addrspace(1) %out, i32 %n) {attrs} {{"),
        )
        .replace(
            "define internal void @mark(ptr addrspace(1) %out) {",
            &format!("define internal void @mark(ptr addrspace(1) %out) {attrs} {{"),
        );
    let calls = TAIL_CALLS
        .replace("%n)\n  notail", "%n) nosync nounwind\n  notail")
        .replace(
            "@mark(ptr addrspace(1) %out)",
            "@mark(ptr addrspace(1) %out) nosync nounwind",
        );
    let kinds = |e: EnumAttrs| e.kinds().collect::<Vec<_>>();
    let all = [
        EnumAttrs::NO_UNWIND,
        EnumAttrs::SPECULATABLE,
        EnumAttrs::WILL_RETURN,
        EnumAttrs::NO_FREE,
        EnumAttrs::NO_SYNC,
        EnumAttrs::MUST_PROGRESS,
    ];
    let kept = [
        EnumAttrs::NO_UNWIND,
        EnumAttrs::WILL_RETURN,
        EnumAttrs::NO_FREE,
        EnumAttrs::MUST_PROGRESS,
    ];
    // The kinds of each call from `@k` to `callee`, in order: `@k` is the entry, rebuilt with
    // the checks, so its calls' kinds must follow them to their new instruction ids.
    let call_enums = |m: &Module, t: &EnumAttrTable, callee: &str| -> Vec<Vec<u32>> {
        let (k, target) = (func(m, "k"), Callee::Direct(func(m, callee)));
        body(m, "k")
            .insts
            .iter()
            .filter_map(|(id, i)| match &i.kind {
                InstKind::Call(c) if c.callee == target => {
                    Some(t.call(k, id).map_or(Vec::new(), |e| kinds(e.func)))
                }
                _ => None,
            })
            .collect()
    };
    let fn_enums = |m: &Module, t: &EnumAttrTable, f: &str| {
        kinds(t.func(func(m, f)).map(|e| e.func).unwrap_or_default())
    };
    let original = kernel(&calls, &callees);
    let (m, t) = (&original.module, &original.enum_attrs);
    for f in ["fill", "mark"] {
        assert_eq!(fn_enums(m, t, f), all, "@{f}");
    }
    assert_eq!(
        call_enums(m, t, "fill"),
        [vec![EnumAttrs::NO_UNWIND, EnumAttrs::NO_SYNC], vec![]]
    );
    let out = run(original).unwrap();
    let (m, t) = (&out.module, &out.enums);
    let fill = &m.func(func(m, "fill")).unwrap().attrs;
    assert_eq!((fill.memory, fn_enums(m, t, "fill")), (None, kept.to_vec()));
    assert_eq!(fn_enums(m, t, "mark"), all);
    assert_eq!(
        call_enums(m, t, "fill"),
        [vec![EnumAttrs::NO_UNWIND], vec![]]
    );
    assert_eq!(
        call_enums(m, t, "mark"),
        [vec![EnumAttrs::NO_UNWIND, EnumAttrs::NO_SYNC]]
    );
}

/// A `musttail` call requires its caller's prototype to match its callee's, and every function
/// that takes the state grows (the entry by one parameter, the others by two); one that passed
/// the state would also make `tail`'s promise and return before the forced test. So a function
/// that takes the state and makes a `musttail` call is refused, typed and named, whatever it
/// calls. A `musttail` call in a function the transform leaves alone stays, and the module
/// assembles with `metal-as` where `xcrun` is available.
#[test]
fn a_function_that_takes_the_fuel_state_and_makes_a_musttail_call_is_refused() {
    let test = "a_function_that_takes_the_fuel_state_and_makes_a_musttail_call_is_refused";
    if !llvm_as(test) {
        return;
    }
    const LOOP: &str = "entry:\n  br label %l\nl:\n  %i = phi i32 [ 0, %entry ], [ %i1, %l ]\n  \
                        %i1 = add i32 %i, 1\n  %c = icmp ult i32 %i1, %n\n  \
                        br i1 %c, label %l, label %d\nd:\n";
    let define = |name: &str, body: &str| {
        format!("define internal void @{name}(ptr addrspace(1) %out, i32 %n) {{\n{body}\n}}\n")
    };
    let store = "  store i32 %n, ptr addrspace(1) %out, align 4\n  ret void";
    let refused = |body: &str, extra: &str| match run(kernel(body, extra)) {
        Err(FuelError::Unsupported(FuelUnsupported::MustTailCall { function })) => function,
        other => panic!("{test}: expected MustTailCall, got {:?}", other.err()),
    };
    // The entry passes the state through it.
    let looping = define("spin", &format!("{LOOP}  ret void"));
    assert_eq!(
        refused(
            "entry:\n  musttail call void @spin(ptr addrspace(1) %out, i32 %n)\n  ret void",
            &looping
        ),
        "k"
    );
    // The entry's own loop makes it grow; the callee needs no state.
    let flat = define("spin", store);
    assert_eq!(
        refused(
            &format!("{LOOP}  musttail call void @spin(ptr addrspace(1) %out, i32 %n)\n  ret void"),
            &flat
        ),
        "k"
    );
    // A looping callee the entry calls plainly makes a musttail call of its own.
    let relay = format!(
        "{}{}",
        define(
            "walk",
            &format!("{LOOP}  musttail call void @leaf(ptr addrspace(1) %out, i32 %n)\n  ret void")
        ),
        define("leaf", store)
    );
    assert_eq!(
        refused(
            "entry:\n  call void @walk(ptr addrspace(1) %out, i32 %n)\n  ret void",
            &relay
        ),
        "walk"
    );
    // A musttail call in a loop-free helper the transform leaves alone stays.
    let untouched = format!(
        "{}{}{}",
        define(
            "relay",
            "  musttail call void @leaf(ptr addrspace(1) %out, i32 %n)\n  ret void"
        ),
        define("leaf", store),
        define("spin", &format!("{LOOP}  ret void"))
    );
    let out = run(kernel(
        "entry:\n  call void @spin(ptr addrspace(1) %out, i32 %n)\n  \
         call void @relay(ptr addrspace(1) %out, i32 %n)\n  ret void",
        &untouched,
    ))
    .unwrap();
    assert_eq!(out.report.functions, 2);
    assert_eq!(param_count(&out.module, "relay"), 2);
    assert_eq!(
        calls_to(&out.module, "relay", "leaf"),
        [(TailKind::MustTail, 2)]
    );
    if have_xcrun(test) {
        let text = write(&out.module, &out.pointees, &out.enums).unwrap();
        metal_as(&text).unwrap_or_else(|e| panic!("{test}: {e}\n--- text ---\n{text}"));
    }
}

// ---- Round trips through metal-as ----

/// The oracle's steps after instrumentation, CPU only: write, `metal-as`, decode again and check
/// the bound again, `metal-opt -passes=verify` at the full level, `metallib`.
fn round_trip(out: &Instrumented, scratch: &ScratchDir) -> Result<(), String> {
    round_trip_at(out, scratch, FULL)
}

/// `metal-opt`'s full AIR verification: the compiler's checks and every AIR intrinsic declaration
/// against the toolchain's own table.
const FULL: &str = "--air-verification-level=full";

/// `metal-opt`'s compiler checks alone, without the intrinsic table, which lacks intrinsics the
/// Metal compiler itself emits (`air.get_null_simdgroup_event` and the other simdgroup event
/// functions among them).
const COMPILER: &str = "--air-verification-level=compiler";

/// The strictest `metal-opt` verification level the AIR `bytes` passes, as the oracle picks it
/// for its input (`metal::platform::libraries`, `Verification::Strictest`): instrumented AIR is
/// held to that level.
fn strictest_level(bytes: &[u8], scratch: &ScratchDir) -> Result<&'static str, String> {
    let input = scratch.path().join("input.air");
    std::fs::write(&input, bytes).map_err(|e| e.to_string())?;
    let mut errors = Vec::new();
    for level in [FULL, COMPILER] {
        let output = scratch.path().join("input.ll");
        let args = ["-S", "-passes=verify", level]
            .into_iter()
            .map(OsString::from)
            .chain([input.clone().into(), "-o".into(), output.into()]);
        let result = xcrun("metal-opt", args).map_err(|e| e.to_string())?;
        if result.success {
            return Ok(level);
        }
        errors.push(result.stderr);
    }
    Err(format!(
        "metal-opt: the input passes no level: {}",
        errors.join("\n")
    ))
}

/// [`round_trip`] with `metal-opt` verifying at `level`.
fn round_trip_at(out: &Instrumented, scratch: &ScratchDir, level: &str) -> Result<(), String> {
    let text = write(&out.module, &out.pointees, &out.enums).map_err(|e| format!("write: {e}"))?;
    let air = metal_as(&text).map_err(|e| format!("{e}\n--- text ---\n{text}"))?;
    let again = decode_as_written(&air, &mut Meter::new(&Budget::default()))
        .map_err(|e| format!("re-decode: {e}"))?;
    for (f, function) in again.module.parts().funcs.iter() {
        if function.body.is_some() && out.module.func(f).is_some_and(|g| g.body.is_some()) {
            let name = String::from_utf8_lossy(again.module.symbols().get(function.name).unwrap())
                .into_owned();
            let before = checks(&out.module, body(&out.module, &name)).len();
            if before > 0 && out.report.exempt_loops == 0 {
                assert_eq!(assert_bounded(&again.module, &name), before, "@{name}");
            } else if before > 0 {
                assert_eq!(
                    assert_bounded_or_counted(&again.module, &name),
                    before,
                    "@{name}"
                );
            }
        }
    }
    let input = scratch.path().join("fuel.air");
    std::fs::write(&input, &air).map_err(|e| e.to_string())?;
    for (tool, args, output) in [
        ("metal-opt", vec!["-S", "-passes=verify", level], "fuel.ll"),
        ("metallib", vec![], "fuel.metallib"),
    ] {
        let output = scratch.path().join(output);
        let args = args.into_iter().map(OsString::from).chain([
            input.clone().into(),
            "-o".into(),
            output.into(),
        ]);
        let result = xcrun(tool, args).map_err(|e| e.to_string())?;
        if !result.success {
            return Err(format!("{tool}: {}\n--- text ---\n{text}", result.stderr));
        }
    }
    Ok(())
}

const MSL: &str = r#"#include <metal_stdlib>
using namespace metal;

__attribute__((noinline)) uint walk(device const uint* in, uint n) {
    uint s = 0;
    for (uint i = 0; i < n; i++) { s += in[i]; }
    return s;
}

kernel void k(device uint* out [[buffer(0)]],
              device const uint* in [[buffer(1)]],
              constant uint& n [[buffer(2)]],
              threadgroup uint* scratch [[threadgroup(0)]],
              uint id [[thread_position_in_grid]],
              uint lid [[thread_position_in_threadgroup]]) {
    uint acc = 0;
    for (uint i = 0; i < n; i++) {
        for (uint j = 0; j < i; j++) { acc += walk(in, j); }
    }
    for (uint r = 0; r < n; r++) {
        scratch[lid] = acc + r;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc += scratch[(lid + 1) % 4];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[id] = acc;
}

vertex float4 v(device const float4* p [[buffer(0)]], constant uint& n [[buffer(1)]],
                uint vid [[vertex_id]]) {
    float4 s = 0;
    for (uint i = 0; i < n; i++) { s += p[vid + i]; }
    return s;
}

fragment float4 f(float4 pos [[position]], constant uint& n [[buffer(0)]]) {
    float4 s = pos;
    for (uint i = 0; i < n; i++) { s = s * 0.5 + 1.0; }
    return s;
}
"#;

#[test]
fn instrumented_msl_round_trips_through_metal_as() {
    if !have_xcrun("instrumented_msl_round_trips_through_metal_as") {
        return;
    }
    let scratch = ScratchDir::new("oracle-fuel-msl").unwrap();
    let source = scratch.path().join("fuel.metal");
    std::fs::write(&source, MSL).unwrap();
    let air = compile_msl(&source, &scratch).unwrap();
    for (entry, loops, barrier_loops) in [("k", 3, 1), ("v", 1, 0), ("f", 1, 0)] {
        let d = decode_bytes(&air);
        let pointees = Pointees::from_table(&d.pointees, &d.module);
        let out = instrument(d.module, pointees, d.enum_attrs, entry, &config()).unwrap();
        assert_eq!(out.report.loops, loops, "@{entry}");
        assert_eq!(out.report.barrier_loops, barrier_loops, "@{entry}");
        round_trip(&out, &scratch).unwrap_or_else(|e| panic!("@{entry}: {e}"));
    }
    // `walk` is compiled `readonly argmemonly`; once it writes the fuel state and the status
    // buffer those effects would be false, so they are dropped.
    let d = decode_bytes(&air);
    let walk = func(&d.module, "_Z4walkPU9MTLdeviceKjj");
    assert!(d.module.func(walk).unwrap().attrs.memory.is_some());
    let pointees = Pointees::from_table(&d.pointees, &d.module);
    let out = instrument(d.module, pointees, d.enum_attrs, "k", &config()).unwrap();
    assert_eq!(out.module.func(walk).unwrap().attrs.memory, None);
    // The one-call form the oracle uses.
    let (bytes, report) = instrument_air(&air, "k", &config()).unwrap();
    assert!(!bytes.is_empty());
    assert_eq!(report.functions, 2);
}

fn fixtures() -> PathBuf {
    metal2vulkan_devtools::fixtures_dir()
}

/// Every stage entry a module lists.
fn stage_entries(m: &Module) -> Vec<String> {
    let mut out = Vec::new();
    for list in m.named_md() {
        if !matches!(
            m.symbols().get(list.name),
            Some(b"air.kernel" | b"air.vertex" | b"air.fragment")
        ) {
            continue;
        }
        for &node in list.ops.iter() {
            let Some(Md::Node { ops, .. }) = m.md(node) else {
                continue;
            };
            if let Some(Md::Value(c)) = ops.first().copied().flatten().and_then(|o| m.md(o)) {
                if let Some(Const::FuncAddr(f)) = m.consts().get(*c).map(|c| &c.value) {
                    out.push(
                        String::from_utf8_lossy(m.symbols().get(m.func(*f).unwrap().name).unwrap())
                            .into_owned(),
                    );
                }
            }
        }
    }
    out
}

/// Whether any function of `m` has a CFG cycle among its reachable blocks.
fn has_loop(m: &Module) -> bool {
    m.parts().funcs.values().any(|f| {
        f.body.as_ref().is_some_and(|b| {
            let cfg = Cfg::build(b, &mut Meter::new(&Budget::default())).unwrap();
            let cyclic = cfg
                .preorder()
                .any(|x| cfg.successors(x).any(|y| cfg.is_retreating(x, y)));
            cyclic
        })
    })
}

/// Every public fixture with a loop, as typed AIR (`.air.ll` through `metal-as`, `.metal`
/// through `metal -c`): each entry that reaches a loop is instrumented and round-trips. Typed
/// refusals are listed visibly; any other error fails.
#[test]
fn public_fixtures_with_loops_round_trip_instrumented() {
    if !have_xcrun("public_fixtures_with_loops_round_trip_instrumented") {
        return;
    }
    let scratch = ScratchDir::new("oracle-fuel-fixtures").unwrap();
    let mut inputs = Vec::new();
    for dir in [fixtures(), fixtures().join("engine")] {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.to_string_lossy().into_owned();
            if name.ends_with(".air.ll") || name.ends_with(".metal") {
                inputs.push(path);
            }
        }
    }
    inputs.sort();
    let (mut instrumented, mut refused, mut failures) = (0, Vec::new(), Vec::new());
    for path in &inputs {
        let name = path.to_string_lossy().into_owned();
        let bytes = if name.ends_with(".air.ll") {
            std::fs::read_to_string(path)
                .map_err(|e| e.to_string())
                .and_then(|t| metal_as(&t).map_err(|e| e.to_string()))
        } else {
            compile_msl(path, &scratch)
        };
        let Ok(bytes) = bytes else {
            failures.push(format!("{name}: {}", bytes.unwrap_err()));
            continue;
        };
        let Ok(d) = decode_as_written(&bytes, &mut Meter::new(&Budget::default())) else {
            continue; // outside the writer's input
        };
        if !has_loop(&d.module) {
            continue;
        }
        // The instrumented AIR is held to the strictest level the fixture's own AIR passes, as
        // the oracle holds it (rfs_async_copy calls simdgroup event functions the full level's
        // table lacks).
        let level = match strictest_level(&bytes, &scratch) {
            Ok(level) => level,
            Err(e) => {
                failures.push(format!("{name}: {e}"));
                continue;
            }
        };
        for entry in stage_entries(&d.module) {
            let d = decode_bytes(&bytes);
            let pointees = Pointees::from_table(&d.pointees, &d.module);
            match instrument(d.module, pointees, d.enum_attrs, &entry, &config()) {
                Ok(out) if out.report.loops + out.report.barrier_loops == 0 => {}
                Ok(out) => match round_trip_at(&out, &scratch, level) {
                    Ok(()) => instrumented += 1,
                    Err(e) => failures.push(format!("{name} @{entry}: {e}")),
                },
                Err(FuelError::Unsupported(u)) => refused.push(format!("{name} @{entry}: {u}")),
                Err(e) => failures.push(format!("{name} @{entry}: {e}")),
            }
        }
    }
    for r in &refused {
        eprintln!("public_fixtures_with_loops_round_trip_instrumented: REFUSED {r}");
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
    assert!(instrumented > 0, "no fixture entry reached a loop");
    eprintln!(
        "public_fixtures_with_loops_round_trip_instrumented: {instrumented} entries instrumented, {} refused",
        refused.len()
    );
}

/// The status atomics take the form of the module's AIR version: a module that states AIR 2.9
/// (Apple's compiler for macOS 27) gets the `mem_flags` `i32` between the scope and the volatile
/// flag, which `metal-opt` requires of it (it refused the AIR 2.8 load in every `tns_mm_v2` case);
/// one that states AIR 2.8, or no version, keeps the AIR 2.8 form. Every call passes as many
/// operands as its declaration takes.
#[test]
fn the_status_atomics_take_the_form_of_the_modules_air_version() {
    if !llvm_as("the_status_atomics_take_the_form_of_the_modules_air_version") {
        return;
    }
    const LOOP: &str = "entry:\n  br label %l\nl:\n  %i = phi i32 [ 0, %entry ], [ %i1, %l ]\n  \
                        %i1 = add i32 %i, 1\n  %c = icmp ult i32 %i1, %n\n  \
                        br i1 %c, label %l, label %d\nd:\n  ret void";
    let version =
        |minor: u32| format!("!air.version = !{{!9}}\n!9 = !{{i32 2, i32 {minor}, i32 0}}");
    for (stated, flagged) in [
        (String::new(), false),
        (version(8), false),
        (version(9), true),
    ] {
        let out = run(kernel(LOOP, &stated)).unwrap();
        assert!(out.report.loops > 0, "{stated}: no fuel loop");
        let text = write(&out.module, &out.pointees, &out.enums).unwrap();
        let flags = if flagged { "i32, " } else { "" };
        for declaration in [
            format!("@\"air.atomic.global.load.i32\"(i32 addrspace(1)*, i32, i32, {flags}i1)"),
            format!(
                "@\"air.atomic.global.add.u.i32\"(i32 addrspace(1)*, i32, i32, i32, {flags}i1)"
            ),
            format!("@\"air.atomic.global.or.u.i32\"(i32 addrspace(1)*, i32, i32, i32, {flags}i1)"),
        ] {
            assert!(
                text.contains(&declaration),
                "{stated}: {declaration}\n{text}"
            );
        }
        for name in [
            "air.atomic.global.load.i32",
            "air.atomic.global.add.u.i32",
            "air.atomic.global.or.u.i32",
        ] {
            let decl = func(&out.module, name);
            let params = param_count(&out.module, name);
            let calls = out
                .module
                .parts()
                .funcs
                .iter()
                .filter_map(|(_, f)| f.body.as_ref())
                .flat_map(|b| b.insts.iter().map(|(_, i)| &i.kind))
                .filter_map(|k| match k {
                    InstKind::Call(c) => match c.callee {
                        Callee::Air { decl: d, .. } if d == decl => Some(c.args.len()),
                        _ => None,
                    },
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert!(!calls.is_empty(), "{stated}: @{name} is never called");
            assert!(
                calls.iter().all(|&n| n == params),
                "{stated}: @{name}: {calls:?} != {params}"
            );
        }
    }
}

/// A module whose loops are all barrier loops calls neither the atomic load nor the add, so no
/// use constrains their pointer parameter: the transform records it, and the declarations are
/// written as Metal declares them (Q2: `metal-opt` refused `air.atomic.global.load.i32` with an
/// `i8 addrspace(1)*` parameter in 45 cases).
#[test]
fn the_status_atomics_keep_their_air_signature_when_only_barrier_loops_call_them() {
    if !llvm_as("the_status_atomics_keep_their_air_signature_when_only_barrier_loops_call_them") {
        return;
    }
    let out = run(kernel(
        "entry:\n  br label %loop\nloop:\n  %i = phi i32 [ 0, %entry ], [ %i2, %loop ]\n  \
         call void @air.wg.barrier(i32 2, i32 1)\n  %i2 = add i32 %i, 1\n  \
         %c = icmp ult i32 %i2, %n\n  br i1 %c, label %loop, label %exit\nexit:\n  ret void",
        "",
    ))
    .unwrap();
    assert_eq!((out.report.loops, out.report.barrier_loops), (0, 1));
    let i32_ty = out
        .module
        .types()
        .iter()
        .find(|(_, t)| **t == Type::Int(32))
        .map(|(id, _)| id);
    for name in [
        "air.atomic.global.load.i32",
        "air.atomic.global.add.u.i32",
        "air.atomic.global.or.u.i32",
    ] {
        assert_eq!(
            out.pointees.param(func(&out.module, name), 0),
            i32_ty,
            "@{name}"
        );
    }
    let text = write(&out.module, &out.pointees, &out.enums).unwrap();
    for declaration in [
        "@\"air.atomic.global.load.i32\"(i32 addrspace(1)*, i32, i32, i1)",
        "@\"air.atomic.global.add.u.i32\"(i32 addrspace(1)*, i32, i32, i32, i1)",
        "@\"air.atomic.global.or.u.i32\"(i32 addrspace(1)*, i32, i32, i32, i1)",
    ] {
        assert!(text.contains(declaration), "{declaration}\n{text}");
    }
}

/// The MSL form of the test above: a kernel whose only loop reaches a barrier round-trips through
/// `metal-as` and `metal-opt`'s full AIR verification.
#[test]
fn a_kernel_with_only_barrier_loops_round_trips_instrumented() {
    if !have_xcrun("a_kernel_with_only_barrier_loops_round_trips_instrumented") {
        return;
    }
    let scratch = ScratchDir::new("oracle-fuel-barrier-only").unwrap();
    let source = scratch.path().join("barrier.metal");
    std::fs::write(
        &source,
        r#"#include <metal_stdlib>
using namespace metal;

kernel void b(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]],
              threadgroup uint* scratch [[threadgroup(0)]],
              uint lid [[thread_position_in_threadgroup]]) {
    uint acc = lid;
    for (uint r = 0; r < n; r++) {
        scratch[lid] = acc + r;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc += scratch[(lid + 1) % 4];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[lid] = acc;
}
"#,
    )
    .unwrap();
    let air = compile_msl(&source, &scratch).unwrap();
    let d = decode_bytes(&air);
    let pointees = Pointees::from_table(&d.pointees, &d.module);
    let out = instrument(d.module, pointees, d.enum_attrs, "b", &config()).unwrap();
    assert_eq!((out.report.loops, out.report.barrier_loops), (0, 1));
    round_trip(&out, &scratch).unwrap();
}

/// The module the oracle loads may be compiled apart from the one the checker classified, and
/// the Metal compiler folds many loops away (Q2: `kernel_bounded_loop_word.metal` compiles to a
/// closed form). An entry that reaches no loop is left exactly as it is: no status argument is
/// declared for a parameter the entry does not have (`metal-opt` crashed on that).
#[test]
fn an_entry_that_reaches_no_loop_is_left_as_it_is() {
    if !llvm_as("an_entry_that_reaches_no_loop_is_left_as_it_is") {
        return;
    }
    // A looping helper nothing executes does not count.
    let dead = "define internal void @dead(i32 %n) {\nentry:\n  br label %loop\nloop:\n  \
                %i = phi i32 [ 0, %entry ], [ %i2, %loop ]\n  %i2 = add i32 %i, 1\n  \
                %c = icmp ult i32 %i2, %n\n  br i1 %c, label %loop, label %exit\nexit:\n  \
                ret void\n}";
    let flat = kernel_bitcode("  store i32 %n, ptr addrspace(1) %out\n  ret void", dead);
    let before = decode_bytes(&flat);
    let pointees = Pointees::from_table(&before.pointees, &before.module);
    let again = decode_bytes(&flat);
    let out = instrument(
        again.module,
        pointees.clone(),
        again.enum_attrs,
        "k",
        &config(),
    )
    .unwrap();
    assert_eq!(out.report, FuelReport::default());
    assert_eq!(param_count(&out.module, "k"), 2);
    assert!(out.module == before.module && out.pointees == pointees);
    assert_eq!(out.enums, before.enum_attrs);
}

/// The MSL form: the loop `metal -c` folds to a closed form round-trips uninstrumented.
#[test]
fn a_loop_the_metal_compiler_folds_round_trips_uninstrumented() {
    if !have_xcrun("a_loop_the_metal_compiler_folds_round_trips_uninstrumented") {
        return;
    }
    let scratch = ScratchDir::new("oracle-fuel-folded").unwrap();
    let source = scratch.path().join("folded.metal");
    std::fs::write(
        &source,
        r#"#include <metal_stdlib>
using namespace metal;

kernel void w(device uint* out [[buffer(0)]], const device uint* count [[buffer(1)]]) {
    uint sum = 0;
    for (uint index = 0; index < count[0]; ++index) {
        sum += index + 1;
    }
    out[0] = sum;
}
"#,
    )
    .unwrap();
    let air = compile_msl(&source, &scratch).unwrap();
    if has_loop(&decode_bytes(&air).module) {
        eprintln!(
            "SKIP a_loop_the_metal_compiler_folds_round_trips_uninstrumented: this compiler \
             keeps the loop"
        );
        return;
    }
    let d = decode_bytes(&air);
    let pointees = Pointees::from_table(&d.pointees, &d.module);
    let out = instrument(d.module, pointees, d.enum_attrs, "w", &config()).unwrap();
    assert_eq!(out.report, FuelReport::default());
    round_trip(&out, &scratch).unwrap();
}

/// The function passes `metal-opt` runs before inlining that act on a call's `tail` marker: alias
/// analysis takes a `tail` call to leave the caller's allocas alone, so `gvn` folds the forced
/// test after it and `dse` deletes the store of the fuel before it.
const TAIL_SENSITIVE_PASSES: &str =
    "-passes=function(sroa,early-cse<memssa>,gvn,instcombine,simplifycfg,dse)";

/// The words of the status buffer, argument `status`, that `b` loads and stores the value of.
fn stored_status_words(m: &Module, b: &Body, status: u32) -> Vec<u64> {
    let stored: Vec<InstId> = b
        .insts
        .values()
        .filter_map(|i| match i.kind {
            InstKind::Store {
                val: Value::Inst(v),
                ..
            } => Some(v),
            _ => None,
        })
        .collect();
    let word = |ptr: Value| match ptr {
        Value::Inst(p) => match &b.insts.get(p)?.kind {
            InstKind::Gep { base, indices, .. } if *base == Value::Arg(ArgIdx::new(status)) => {
                match indices.as_ref() {
                    [Value::Const(c)] => match m.consts().get(*c).map(|c| &c.value) {
                        Some(Const::Int(k)) => Some(*k),
                        _ => None,
                    },
                    _ => None,
                }
            }
            _ => None,
        },
        _ => None,
    };
    stored
        .into_iter()
        .filter_map(|v| match &b.insts.get(v)?.kind {
            InstKind::Load { ptr, .. } => word(*ptr),
            _ => None,
        })
        .collect()
}

/// Whether `b` sets `EXHAUSTED`: the forced test's exit.
fn sets_exhausted(b: &Body) -> bool {
    b.insts.values().any(|i| {
        matches!(&i.kind, InstKind::Call(c) if matches!(c.callee, Callee::Air {
            op: AirIntrinsic::Atomic(Atomic { op: AtomicOp::Or, .. }),
            ..
        }))
    })
}

/// The MSL form of [`a_call_that_passes_the_fuel_state_drops_its_tail_marker`], run through the
/// toolchain: the Metal compiler calls a looping helper with `tail call` in tail position, the
/// shape of the case Metal 26.7 refused with no pool chunk consumed. The instrumented AIR keeps
/// the entry's load of the invocation fuel, its store into the state, and the forced test after
/// the call through `metal-opt`'s passes that act on `tail`.
#[test]
fn the_toolchain_keeps_the_fuel_state_a_former_tail_call_passes() {
    let test = "the_toolchain_keeps_the_fuel_state_a_former_tail_call_passes";
    if !have_xcrun(test) {
        return;
    }
    let scratch = ScratchDir::new("oracle-fuel-tail").unwrap();
    let source = scratch.path().join("tail.metal");
    std::fs::write(
        &source,
        r#"#include <metal_stdlib>
using namespace metal;

__attribute__((noinline)) void fill(device uint* out, constant uint& n, uint id) {
    for (uint i = id; i < n; i += 4) { out[i] = i * 3 + 1; }
}

kernel void t(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]],
              uint id [[thread_position_in_grid]]) {
    fill(out, n, id);
}
"#,
    )
    .unwrap();
    let air = compile_msl(&source, &scratch).unwrap();
    let fill = "_Z4fillPU9MTLdevicejRU11MTLconstantKjj";
    let d = decode_bytes(&air);
    assert_eq!(calls_to(&d.module, "t", fill), [(TailKind::Tail, 3)]);
    let pointees = Pointees::from_table(&d.pointees, &d.module);
    let out = instrument(d.module, pointees, d.enum_attrs, "t", &config()).unwrap();
    assert_eq!(out.report.tail_calls, 1);
    assert_eq!(calls_to(&out.module, "t", fill), [(TailKind::None, 5)]);
    round_trip(&out, &scratch).unwrap();

    let text = write(&out.module, &out.pointees, &out.enums).unwrap();
    let input = scratch.path().join("tail.air");
    std::fs::write(&input, metal_as(&text).unwrap()).unwrap();
    let optimized = scratch.path().join("tail-opt.air");
    let args = [
        OsString::from(TAIL_SENSITIVE_PASSES),
        input.into(),
        "-o".into(),
        optimized.clone().into(),
    ];
    let result = xcrun("metal-opt", args).unwrap();
    assert!(result.success, "metal-opt: {}", result.stderr);
    let after = decode_bytes(&std::fs::read(&optimized).unwrap());
    let entry = body(&after.module, "t");
    // The status buffer is the entry's fourth argument; word 3 is the invocation fuel.
    assert_eq!(
        stored_status_words(&after.module, entry, 3),
        [super::status::WORD_INVOCATION_FUEL],
        "{test}: the entry no longer initializes the fuel state\n--- input ---\n{text}"
    );
    assert!(
        sets_exhausted(entry),
        "{test}: the forced test after the call folded away\n--- input ---\n{text}"
    );
}
