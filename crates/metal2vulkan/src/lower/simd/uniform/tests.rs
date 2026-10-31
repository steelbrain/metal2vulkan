//! L0 tests of the workgroup-uniformity analysis on test-builder IR.

use super::*;
use crate::air::abi::{
    AbiInt, Access, ArgTypeInfo, BufferArg, EntryPoint, ParamBinding, Slot, Stage as AbiStage,
};
use crate::air::ir::build::{FunctionBuilder, ModuleBuilder};
use crate::air::ir::{Align, DataLayout, FuncId, IntPredicate, Module, TypeId};
use crate::error::DiagName;
use crate::graph::{Cfg, DomTree};
use crate::pipeline::testing::meter;

const AIR_DL: &str = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

/// The kernel's parameters: `%0` a device buffer, `%1` a constant buffer, `%2`
/// `thread_index_in_threadgroup`, `%3` `threadgroup_position_in_grid`.
const DEVICE: u32 = 0;
const CONSTANT: u32 = 1;
const TID: u32 = 2;
const GROUP: u32 = 3;

fn buffer(index: u32, space: AddrSpace) -> ParamRole {
    ParamRole::Buffer(BufferArg {
        location: Slot {
            index: AbiInt::Const(index),
            count: AbiInt::Const(1),
        },
        access: Access::Read,
        space,
        buffer_size: None,
        layout: None,
        raster_order_group: None,
    })
}

struct Types {
    i1: TypeId,
    i32: TypeId,
    /// `air.simd_sum.u.i32`, a simdgroup reduction.
    sum: FuncId,
}

/// A kernel whose body `body` builds, with what `body` returns.
fn kernel<R>(body: impl FnOnce(&mut FunctionBuilder<'_>, &Types) -> R) -> (Entry, R) {
    let mut mb = ModuleBuilder::new();
    mb.set_data_layout(DataLayout::parse(AIR_DL).expect("datalayout"));
    let void = mb.void();
    let (i1, i32) = (mb.int(1), mb.int(32));
    let sum_ty = mb.fn_ty(i32, &[i32]);
    let types = Types {
        i1,
        i32,
        sum: mb.declare("air.simd_sum.u.i32", sum_ty),
    };
    let dev = mb.ptr(AddrSpace::Device);
    let cst = mb.ptr(AddrSpace::Constant);
    let fty = mb.fn_ty(void, &[dev, cst, types.i32, types.i32]);
    let mut fb = mb.define("k", fty);
    let out = body(&mut fb, &types);
    let func = fb.finish();
    let module = mb.finish().expect("module");
    let roles = vec![
        buffer(0, AddrSpace::Device),
        buffer(1, AddrSpace::Constant),
        ParamRole::Builtin(Builtin::ThreadIndexInThreadgroup),
        ParamRole::Builtin(Builtin::ThreadgroupPositionInGrid),
    ];
    (entry(module, func, roles), out)
}

/// The uniformity of a kernel whose body `body` builds, with the blocks it returns.
fn analyze(
    body: impl FnOnce(&mut FunctionBuilder<'_>, &Types) -> Vec<BlockId>,
) -> (Uniformity, Vec<BlockId>) {
    let (e, blocks) = kernel(body);
    let body = e
        .module
        .func(e.func)
        .and_then(|f| f.body.as_ref())
        .expect("body");
    let mut m = meter();
    let u = Uniformity::of(&e, body, &mut m).expect("uniformity");
    (u, blocks)
}

/// Whether each value `body` returns is uniform among the threads of one simdgroup, under the
/// barrier contract.
fn simdgroup_uniform(
    body: impl FnOnce(&mut FunctionBuilder<'_>, &Types) -> Vec<Value>,
) -> Vec<bool> {
    let (e, values) = kernel(body);
    let body = e
        .module
        .func(e.func)
        .and_then(|f| f.body.as_ref())
        .expect("body");
    let mut m = meter();
    let loops = LoopForest::compute(&e.cfg, &mut m).expect("loops");
    let u =
        Uniformity::under_barriers(&e, body, Among::Simdgroup, &loops, &mut m).expect("uniformity");
    values
        .into_iter()
        .map(|v| u.is_uniform_value(&e, v))
        .collect()
}

/// `air.simd_sum.u.i32(v)`.
fn simd_sum(fb: &mut FunctionBuilder<'_>, t: &Types, v: Value) -> Value {
    let op = AirIntrinsic::parse("air.simd_sum.u.i32").expect("a simdgroup reduction");
    fb.push(
        t.i32,
        InstKind::Call(Box::new(crate::air::ir::Call {
            callee: Callee::Air { decl: t.sum, op },
            args: Box::new([v]),
            fmf: crate::air::ir::FastMath::NONE,
            tail: crate::air::ir::TailKind::None,
            attrs: None,
            range: None,
        })),
    )
}

fn entry(module: Module, func: FuncId, roles: Vec<ParamRole>) -> Entry {
    let mut m = meter();
    let body = module
        .func(func)
        .and_then(|f| f.body.as_ref())
        .expect("body");
    let cfg = Cfg::build(body, &mut m).expect("cfg");
    let dom = DomTree::compute(&cfg, &mut m).expect("dom");
    let params = roles
        .into_iter()
        .enumerate()
        .map(|(i, role)| ParamBinding {
            param: i as u32,
            role,
            unused: false,
            predicate: None,
            ty: ArgTypeInfo::default(),
            name: None,
        })
        .collect();
    Entry {
        func,
        ep: EntryPoint {
            func,
            stage: AbiStage::Kernel {
                max_total_threads: None,
            },
            params,
            outputs: Vec::new(),
            name: DiagName::new("k"),
            user_annotation: None,
        },
        absent_handles: Vec::new(),
        structs: Vec::new(),
        language: None,
        compile: Default::default(),
        layout: module.data_layout().clone(),
        cfg,
        dom,
        static_samplers: Vec::new(),
        linked_sampler_groups: Vec::new(),
        unnumbered_linked_samplers: false,
        intersection_candidates: false,
        function_constants: false,
        undefined_reads: Vec::new(),
        function_tables: Default::default(),
        lockstep: Default::default(),
        call_stack: None,
        unmeasured_speculation: false,
        module,
    }
}

fn icmp(fb: &mut FunctionBuilder<'_>, t: &Types, pred: IntPredicate, a: Value, b: Value) -> Value {
    fb.push(t.i1, InstKind::ICmp { pred, a, b })
}

fn load(fb: &mut FunctionBuilder<'_>, ty: TypeId, ptr: Value) -> Value {
    fb.push(
        ty,
        InstKind::Load {
            ptr,
            align: Align::from_bytes(4),
            volatile: false,
            atomic: None,
            range: None,
        },
    )
}

/// `if (c) { then } join; ret`, with `c` computed in the entry by `cond`.
fn diamond(
    cond: impl FnOnce(&mut FunctionBuilder<'_>, &Types) -> Value,
) -> (Uniformity, Vec<BlockId>) {
    analyze(|fb, t| {
        let entry = BlockId::new(0);
        let (then, join) = (fb.block(), fb.block());
        let c = cond(fb, t);
        fb.cond_br(c, then, join);
        fb.switch_to(then);
        fb.br(join);
        fb.switch_to(join);
        fb.ret(None);
        vec![entry, then, join]
    })
}

/// A branch on a per-threadgroup value keeps both paths uniform; one on the thread's index makes
/// the arm it guards divergent, and the join post-dominates the branch, so it is uniform again.
#[test]
fn branches_on_per_invocation_values_make_their_arms_divergent() {
    let (u, b) = diamond(|fb, t| {
        let zero = fb.module().const_int(t.i32, 0);
        icmp(fb, t, IntPredicate::Eq, fb.arg(GROUP), zero)
    });
    assert!(b.iter().all(|&b| u.is_uniform(b)), "{u:?}");

    let (u, b) = diamond(|fb, t| {
        let zero = fb.module().const_int(t.i32, 0);
        icmp(fb, t, IntPredicate::Eq, fb.arg(TID), zero)
    });
    assert!(u.is_uniform(b[0]));
    assert!(!u.is_uniform(b[1]));
    assert!(u.is_uniform(b[2]));
}

/// Loads of `constant` memory through a uniform pointer are uniform (it is not written while the
/// dispatch runs); device memory may be, so its loads are divergent.
#[test]
fn only_constant_memory_loads_are_uniform() {
    let (u, b) = diamond(|fb, t| {
        let n = load(fb, t.i32, fb.arg(CONSTANT));
        let zero = fb.module().const_int(t.i32, 0);
        icmp(fb, t, IntPredicate::Eq, n, zero)
    });
    assert!(u.is_uniform(b[1]));

    let (u, b) = diamond(|fb, t| {
        let n = load(fb, t.i32, fb.arg(DEVICE));
        let zero = fb.module().const_int(t.i32, 0);
        icmp(fb, t, IntPredicate::Eq, n, zero)
    });
    assert!(!u.is_uniform(b[1]));
}

/// An `undef` operand may read a different value in each invocation.
#[test]
fn undef_is_divergent() {
    let (u, b) = diamond(|fb, t| {
        let undef = fb.module().constant(t.i1, Const::Undef);
        Value::Const(undef)
    });
    assert!(!u.is_uniform(b[1]));
}

/// An early return of some invocations makes everything after it divergent.
#[test]
fn an_early_return_makes_the_rest_divergent() {
    let (u, b) = analyze(|fb, t| {
        let (early, rest, tail) = (fb.block(), fb.block(), fb.block());
        let n = load(fb, t.i32, fb.arg(CONSTANT));
        let out = icmp(fb, t, IntPredicate::Uge, fb.arg(TID), n);
        fb.cond_br(out, early, rest);
        fb.switch_to(early);
        fb.ret(None);
        fb.switch_to(rest);
        fb.br(tail);
        fb.switch_to(tail);
        fb.ret(None);
        vec![BlockId::new(0), early, rest, tail]
    });
    assert!(u.is_uniform(b[0]));
    assert!(!u.is_uniform(b[1]));
    assert!(!u.is_uniform(b[2]));
    assert!(!u.is_uniform(b[3]));
}

/// A loop whose trip count is uniform is uniform; one left on a per-invocation condition makes
/// its header and latch divergent, and its counter, read after the loop, keeps the branch that
/// reads it divergent.
#[test]
fn loops_are_divergent_exactly_when_their_exit_is() {
    for (per_thread, uniform_loop) in [(false, true), (true, false)] {
        let (u, b) = analyze(|fb, t| {
            let entry = BlockId::new(0);
            let (header, latch, exit, then, join) =
                (fb.block(), fb.block(), fb.block(), fb.block(), fb.block());
            let n = if per_thread {
                fb.arg(TID)
            } else {
                load(fb, t.i32, fb.arg(CONSTANT))
            };
            let zero = fb.module().const_int(t.i32, 0);
            let one = fb.module().const_int(t.i32, 1);
            fb.br(header);
            fb.switch_to(header);
            // Provisional ids are handed out in push order: the entry pushed the load (when
            // there is one) and its branch, so the phi and the increment follow them.
            let phi = u32::from(!per_thread) + 1;
            let next_ref = Value::Inst(InstId::new(phi + 1));
            let i = fb.push(
                t.i32,
                InstKind::Phi {
                    incoming: Box::new([(zero, entry), (next_ref, latch)]),
                    fmf: crate::air::ir::FastMath::NONE,
                },
            );
            assert_eq!(i, Value::Inst(InstId::new(phi)));
            let next = fb.bin(crate::air::ir::BinOp::Add, i, one);
            assert_eq!(next, next_ref);
            let more = icmp(fb, t, IntPredicate::Ult, next, n);
            fb.cond_br(more, latch, exit);
            fb.switch_to(latch);
            fb.br(header);
            fb.switch_to(exit);
            // After the loop, a branch on the counter.
            let big = icmp(fb, t, IntPredicate::Ugt, next, one);
            fb.cond_br(big, then, join);
            fb.switch_to(then);
            fb.br(join);
            fb.switch_to(join);
            fb.ret(None);
            vec![entry, header, latch, exit, then, join]
        });
        assert!(u.is_uniform(b[0]));
        assert_eq!(u.is_uniform(b[1]), uniform_loop, "header");
        assert_eq!(u.is_uniform(b[2]), uniform_loop, "latch");
        assert!(u.is_uniform(b[3]), "the loop exit post-dominates it");
        assert_eq!(
            u.is_uniform(b[4]),
            uniform_loop,
            "a branch on the trip count"
        );
        assert!(u.is_uniform(b[5]));
    }
}

/// A simdgroup reduction in a block the whole simdgroup runs together gives every thread one
/// result, whatever its operand; in an arm some threads skip, the threads that skip it hold no
/// result of it.
#[test]
fn a_collective_is_uniform_only_in_a_uniform_block() {
    let u = simdgroup_uniform(|fb, t| {
        let (then, join) = (fb.block(), fb.block());
        let together = simd_sum(fb, t, fb.arg(TID));
        let zero = fb.module().const_int(t.i32, 0);
        let first = icmp(fb, t, IntPredicate::Eq, fb.arg(TID), zero);
        fb.cond_br(first, then, join);
        fb.switch_to(then);
        let apart = simd_sum(fb, t, fb.arg(TID));
        fb.br(join);
        fb.switch_to(join);
        fb.ret(None);
        vec![together, apart]
    });
    assert_eq!(u, [true, false]);
}

/// A simdgroup reduction in a loop every thread of the simdgroup leaves together is uniform; in
/// one its threads leave after different iterations, each keeps the result of its own last
/// iteration, so after the loop the threads hold different results.
#[test]
fn a_collective_in_a_loop_left_apart_is_divergent() {
    for (per_thread, together) in [(false, true), (true, false)] {
        let u = simdgroup_uniform(|fb, t| {
            let entry = BlockId::new(0);
            let (header, latch, exit) = (fb.block(), fb.block(), fb.block());
            let n = if per_thread {
                fb.arg(TID)
            } else {
                load(fb, t.i32, fb.arg(CONSTANT))
            };
            let zero = fb.module().const_int(t.i32, 0);
            let one = fb.module().const_int(t.i32, 1);
            fb.br(header);
            fb.switch_to(header);
            // Provisional ids are handed out in push order: the entry pushed the load (when
            // there is one) and its branch, so the phi and the increment follow them.
            let phi = u32::from(!per_thread) + 1;
            let next_ref = Value::Inst(InstId::new(phi + 1));
            let i = fb.push(
                t.i32,
                InstKind::Phi {
                    incoming: Box::new([(zero, entry), (next_ref, latch)]),
                    fmf: crate::air::ir::FastMath::NONE,
                },
            );
            let next = fb.bin(crate::air::ir::BinOp::Add, i, one);
            assert_eq!(next, next_ref);
            let sum = simd_sum(fb, t, fb.arg(TID));
            let more = icmp(fb, t, IntPredicate::Ult, next, n);
            fb.cond_br(more, latch, exit);
            fb.switch_to(latch);
            fb.br(header);
            fb.switch_to(exit);
            fb.ret(None);
            vec![sum]
        });
        assert_eq!(u, [together], "a loop left per thread: {per_thread}");
    }
}
