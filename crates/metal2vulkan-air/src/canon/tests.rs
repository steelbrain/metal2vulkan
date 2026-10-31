//! Canonicalization tests over IR built with the typed test builder.

use std::collections::BTreeMap;

use super::*;
use crate::abi::{
    AbiInt, Access, AirAbi, ArgTypeInfo, BufferArg, CompileOptions, EntryPoint,
    FunctionConstantDecl, ModuleLimits, ParamBinding, ParamRole, Predicate, Slot, Stage,
};
use crate::budget::{Budget, Meter};
use crate::error::{CanonError, InputErrorKind};
use crate::error::{Limit, MissingInput};
use crate::intrinsics::{AbsentAccess, AirIntrinsic, Elem, FunctionConstant, NumTy, TexOp};
use crate::ir::build::{FunctionBuilder, ModuleBuilder};
use crate::ir::print::print;
use crate::ir::{
    AddrSpace, Align, BinOp, Call, CastOp, Const, ConstExpr, FastMath, Global, IntPredicate,
    Linkage, Section, TailKind, Value,
};

mod aggregates;
mod command_buffers;
mod global_builtins;
mod private_memory;
mod table_copies;
mod texture_writes;
mod unobserved;

fn meter() -> Meter {
    Meter::new(&Budget::default())
}

fn abi(entries: Vec<EntryPoint>, fcs: Vec<FunctionConstantDecl>) -> AirAbi {
    AirAbi {
        air_version: None,
        language: None,
        compile: CompileOptions::default(),
        limits: ModuleLimits::default(),
        entries,
        function_constants: fcs,
        static_samplers: Vec::new(),
        visible_function_refs: Vec::new(),
        visible_functions: Vec::new(),
        core_image_kernels: Vec::new(),
        linkage_signature: None,
        structs: Vec::new(),
        producer: None,
    }
}

fn buffer(param: u32, index: u32) -> ParamBinding {
    ParamBinding {
        param,
        role: ParamRole::Buffer(BufferArg {
            location: Slot {
                index: AbiInt::Const(index),
                count: AbiInt::Const(1),
            },
            access: Access::ReadWrite,
            space: AddrSpace::Device,
            buffer_size: None,
            layout: None,
            raster_order_group: None,
        }),
        unused: false,
        predicate: None,
        ty: ArgTypeInfo::default(),
        name: None,
    }
}

fn kernel(func: FuncId, params: Vec<ParamBinding>) -> EntryPoint {
    EntryPoint {
        func,
        stage: Stage::Kernel {
            max_total_threads: None,
        },
        params,
        outputs: Vec::new(),
        name: DiagName::new("k"),
        user_annotation: None,
    }
}

/// An `air.fc_initializer` global of `ty`, as the frontend writes it.
pub(super) fn fc_global(m: &mut ModuleBuilder, name: &str, ty: TypeId) -> GlobalId {
    let undef = m.constant(ty, Const::Undef);
    let name = m.sym(name);
    m.global(Global {
        name,
        value_ty: ty,
        space: AddrSpace::Constant,
        constant: true,
        init: Some(undef),
        externally_initialized: true,
        visibility: Default::default(),
        unnamed_addr: Default::default(),
        linkage: Linkage::LinkOnceOdr,
        align: None,
        section: Some(Section::FcInitializer),
    })
}

pub(super) fn fc_decl(index: u32, global: GlobalId, ty: TypeId) -> FunctionConstantDecl {
    FunctionConstantDecl {
        index,
        global,
        ty,
        flag: None,
        data_type: None,
        name: DiagName::new("c"),
        type_name: DiagName::new("int"),
    }
}

fn call(f: &mut FunctionBuilder<'_>, ty: TypeId, callee: Callee, args: &[Value]) -> Value {
    f.push(
        ty,
        InstKind::Call(Box::new(Call {
            callee,
            args: args.into(),
            fmf: FastMath::NONE,
            tail: TailKind::None,
            attrs: None,
            range: None,
        })),
    )
}

fn store(f: &mut FunctionBuilder<'_>, val: Value, ptr: Value) {
    f.push_void(InstKind::Store {
        val,
        ptr,
        align: Align::from_bytes(4),
        volatile: false,
        atomic: None,
    });
}

fn load(f: &mut FunctionBuilder<'_>, ty: TypeId, ptr: Value) -> Value {
    f.push(
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

fn arg(i: u32) -> Value {
    Value::Arg(crate::ir::ArgIdx::new(i))
}

fn int(v: u32) -> ConstValue {
    ConstValue {
        ty: DataType::Int,
        bytes: v.to_le_bytes().into(),
    }
}

fn request(constants: &[(u32, ConstValue)]) -> SpecRequest {
    SpecRequest {
        call_stack_depth: 1,
        entry: "k".into(),
        constants: constants.iter().cloned().collect::<BTreeMap<_, _>>(),
    }
}

/// The printed module after its header line (the data layout).
fn body_text(e: &Entry) -> String {
    let text = print(&e.module);
    text.lines()
        .filter(|l| !l.starts_with("; datalayout"))
        .map(|l| format!("{l}\n"))
        .collect()
}

/// `kernel k(device i32* out)`: if `is_function_constant_defined(@mode)` store `@mode`, else 7.
fn gated_store() -> (Module, AirAbi) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let cptr = m.ptr(AddrSpace::Constant);
    let mode = fc_global(&mut m, "mode", i32t);
    let is_def_ty = m.fn_ty(i1, &[cptr]);
    let is_def = m.declare("air.is_function_constant_defined", is_def_ty);
    let fty = m.fn_ty(void, &[dev]);
    let seven = m.const_int(i32t, 7);
    let mode_addr = m.global_addr(mode);
    let mut f = m.define("k", fty);
    let (then, other, exit) = (f.block(), f.block(), f.block());
    let defined = call(
        &mut f,
        i1,
        Callee::Air {
            decl: is_def,
            op: AirIntrinsic::FunctionConstant(FunctionConstant::IsDefined),
        },
        &[mode_addr],
    );
    f.cond_br(defined, then, other);
    f.switch_to(then);
    let v = load(&mut f, i32t, mode_addr);
    store(&mut f, v, arg(0));
    f.br(exit);
    f.switch_to(other);
    store(&mut f, seven, arg(0));
    f.br(exit);
    f.switch_to(exit);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(
        vec![kernel(k, vec![buffer(0, 0)])],
        vec![fc_decl(0, mode, i32t)],
    );
    (module, abi)
}

#[test]
fn a_supplied_constant_is_baked_and_its_gate_folds() {
    let (module, abi) = gated_store();
    let e = specialize(module, &abi, &request(&[(0, int(5))]), &mut meter()).unwrap();
    assert_eq!(
        body_text(&e),
        "%t2 = type void (ptr addrspace(1))\n\
         \n\
         define %t2 @f0 \"k\" {\n\
         bb0:\n  br bb1\n\
         bb1:\n  store i32 5, %a0, align 4\n  br bb2\n\
         bb2:\n  ret void\n\
         }\n"
    );
    assert_eq!(e.func, FuncId::new(0));
    assert_eq!(e.ep.func, e.func);
    assert_eq!(e.cfg.reachable_count(), 3);
    assert_eq!(e.dom.idom(BlockId::new(2)), Some(BlockId::new(1)));
}

/// The entry's control-flow graph and dominator tree are the pair the verification of the pruned
/// module built for its own dominance check, not a second pair computed over the same body.
///
/// `gated_store`'s module holds one function with a body, so the whole specialization computes one
/// dominator tree. Rebuilding the pair after the verification reads two.
#[test]
fn a_specialization_computes_the_entry_dominator_tree_once() {
    let (module, abi) = gated_store();
    let before = crate::graph::trees_computed();
    let e = specialize(module, &abi, &request(&[(0, int(5))]), &mut meter()).unwrap();
    let computed = crate::graph::trees_computed() - before;
    // The pair is the entry's own and is usable: the same answers the folding test reads.
    assert_eq!(e.cfg.reachable_count(), 3);
    assert_eq!(e.dom.idom(BlockId::new(2)), Some(BlockId::new(1)));
    assert_eq!(
        computed, 1,
        "the specialization computed {computed} dominator trees for one body"
    );
}

#[test]
fn an_unsupplied_constant_is_undefined_and_its_gate_folds_off() {
    let (module, abi) = gated_store();
    let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
    assert_eq!(
        body_text(&e),
        "%t2 = type void (ptr addrspace(1))\n\
         \n\
         define %t2 @f0 \"k\" {\n\
         bb0:\n  br bb1\n\
         bb1:\n  store i32 7, %a0, align 4\n  br bb2\n\
         bb2:\n  ret void\n\
         }\n"
    );
    assert!(
        e.module.parts().globals.is_empty(),
        "the FC global is pruned"
    );
}

#[test]
fn canonicalization_is_deterministic() {
    let run = || {
        let (module, abi) = gated_store();
        print(
            &specialize(module, &abi, &request(&[(0, int(9))]), &mut meter())
                .unwrap()
                .module,
        )
    };
    assert_eq!(run(), run());
}

/// `kernel k(device i32* out)` whose body `use_mode` builds from a load of the unguarded `@mode`.
fn unguarded(use_mode: impl FnOnce(&mut FunctionBuilder<'_>, Value, TypeId)) -> (Module, AirAbi) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let mode = fc_global(&mut m, "mode", i32t);
    let fty = m.fn_ty(void, &[dev]);
    let mode_addr = m.global_addr(mode);
    let mut f = m.define("k", fty);
    let v = load(&mut f, i32t, mode_addr);
    use_mode(&mut f, v, i32t);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(
        vec![kernel(k, vec![buffer(0, 0)])],
        vec![fc_decl(3, mode, i32t)],
    );
    (module, abi)
}

#[test]
fn a_surviving_value_use_of_an_unsupplied_constant_needs_input() {
    let stored = |f: &mut FunctionBuilder<'_>, v: Value, i32t: TypeId| {
        let one = f.module().const_int(i32t, 1);
        let w = f.bin(BinOp::Add, v, one);
        store(f, w, arg(0));
        f.ret(None);
    };
    let (module, abi) = unguarded(stored);
    let err = specialize(module, &abi, &request(&[]), &mut meter()).unwrap_err();
    assert_eq!(
        err,
        Error::NeedsInput(MissingInput::FunctionConstant { index: 3 })
    );
    // Supplying it bakes the value and folds the arithmetic.
    let (module, abi) = unguarded(stored);
    let e = specialize(module, &abi, &request(&[(3, int(41))]), &mut meter()).unwrap();
    assert!(
        body_text(&e).contains("store i32 42, %a0"),
        "{}",
        body_text(&e)
    );
}

#[test]
fn a_branch_on_an_unsupplied_constant_needs_input() {
    let (module, abi) = unguarded(|f, v, i32t| {
        let zero = f.module().const_int(i32t, 0);
        let i1 = f.module().int(1);
        let c = f.push(
            i1,
            InstKind::ICmp {
                pred: IntPredicate::Eq,
                a: v,
                b: zero,
            },
        );
        let (t, e) = (f.block(), f.block());
        f.cond_br(c, t, e);
        f.switch_to(t);
        f.ret(None);
        f.switch_to(e);
        f.ret(None);
    });
    let err = specialize(module, &abi, &request(&[]), &mut meter()).unwrap_err();
    assert_eq!(
        err,
        Error::NeedsInput(MissingInput::FunctionConstant { index: 3 })
    );
}

#[test]
fn a_dead_pure_use_of_an_unsupplied_constant_is_dropped() {
    let (module, abi) = unguarded(|f, v, i32t| {
        let one = f.module().const_int(i32t, 1);
        f.bin(BinOp::Mul, v, one);
        f.ret(None);
    });
    let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
    let text = body_text(&e);
    assert!(!text.contains("load") && !text.contains("mul"), "{text}");
    assert!(e.module.parts().globals.is_empty());
}

#[test]
fn a_branch_on_undef_is_refused() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let undef = m.constant(i1, Const::Undef);
    let fty = m.fn_ty(void, &[]);
    let mut f = m.define("k", fty);
    let (t, e) = (f.block(), f.block());
    f.cond_br(Value::Const(undef), t, e);
    f.switch_to(t);
    f.ret(None);
    f.switch_to(e);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, Vec::new())], Vec::new());
    let err = specialize(module, &abi, &request(&[]), &mut meter()).unwrap_err();
    let Error::InvalidInput(e) = err else {
        panic!("{err:?}");
    };
    assert_eq!(e.kind, InputErrorKind::Canon(CanonError::UndefCondition));
}

#[test]
fn sccp_folds_integers_phis_and_switches_and_deletes_dead_blocks() {
    // bb0: x = 3 * 4; switch x [12 -> bb1] default bb2
    // bb1: br bb3            bb2: br bb3 (dead)
    // bb3: p = phi [x, bb1], [0, bb2]; c = icmp ult p, 20; br c, bb4, bb5 (dead)
    // bb4: store p + 1; ret  bb5: ret
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let fty = m.fn_ty(void, &[dev]);
    let (three, four, zero, one, twenty) = (
        m.const_int(i32t, 3),
        m.const_int(i32t, 4),
        m.const_int(i32t, 0),
        m.const_int(i32t, 1),
        m.const_int(i32t, 20),
    );
    let mut f = m.define("k", fty);
    let (b1, b2, b3, b4, b5) = (f.block(), f.block(), f.block(), f.block(), f.block());
    let x = f.bin(BinOp::Mul, three, four);
    f.push_void(InstKind::Switch {
        v: x,
        default: b2,
        cases: Box::new([(12, b1)]),
        hints: None,
    });
    f.switch_to(b1);
    f.br(b3);
    f.switch_to(b2);
    f.br(b3);
    f.switch_to(b3);
    let p = f.push(
        i32t,
        InstKind::Phi {
            incoming: Box::new([(x, b1), (zero, b2)]),
            fmf: FastMath::NONE,
        },
    );
    let c = f.push(
        i1,
        InstKind::ICmp {
            pred: IntPredicate::Ult,
            a: p,
            b: twenty,
        },
    );
    f.cond_br(c, b4, b5);
    f.switch_to(b4);
    let q = f.bin(BinOp::Add, p, one);
    store(&mut f, q, arg(0));
    f.ret(None);
    f.switch_to(b5);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, vec![buffer(0, 0)])], Vec::new());
    let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
    assert_eq!(
        body_text(&e),
        "%t2 = type void (ptr addrspace(1))\n\
         \n\
         define %t2 @f0 \"k\" {\n\
         bb0:\n  br bb1\n\
         bb1:\n  br bb2\n\
         bb2:\n  br bb3\n\
         bb3:\n  store i32 13, %a0, align 4\n  ret void\n\
         }\n"
    );
}

#[test]
fn a_folded_branch_with_parallel_edges_keeps_one_phi_entry_per_edge() {
    // bb0: br a0 ? bb1 : bb1 on a non-constant, then bb1 phis stay two-entry; a constant
    // `br true, bb2, bb2` in bb1 leaves one edge into bb2, whose phi keeps one entry.
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let fty = m.fn_ty(void, &[dev, i1, i32t]);
    let t = m.const_int(i1, 1);
    let mut f = m.define("k", fty);
    let (b1, b2) = (f.block(), f.block());
    f.cond_br(f.arg(1), b1, b1);
    f.switch_to(b1);
    let p = f.push(
        i32t,
        InstKind::Phi {
            incoming: Box::new([(f.arg(2), BlockId::new(0)), (f.arg(2), BlockId::new(0))]),
            fmf: FastMath::NONE,
        },
    );
    f.cond_br(t, b2, b2);
    f.switch_to(b2);
    let q = f.push(
        i32t,
        InstKind::Phi {
            incoming: Box::new([(p, b1), (p, b1)]),
            fmf: FastMath::NONE,
        },
    );
    store(&mut f, q, arg(0));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, vec![buffer(0, 0)])], Vec::new());
    let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
    let text = body_text(&e);
    assert!(
        text.contains("%i1 = phi i32 [%a2, bb0], [%a2, bb0]"),
        "{text}"
    );
    assert!(text.contains("%i3 = phi i32 [%i1, bb1]\n"), "{text}");
}

#[test]
fn no_op_pointer_casts_are_erased() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let fty = m.fn_ty(void, &[dev]);
    let one = m.const_int(i32t, 1);
    let mut f = m.define("k", fty);
    let p = f.push(
        dev,
        InstKind::Cast {
            op: CastOp::Bitcast,
            v: f.arg(0),
        },
    );
    let q = f.push(
        dev,
        InstKind::Cast {
            op: CastOp::Bitcast,
            v: p,
        },
    );
    store(&mut f, one, q);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, vec![buffer(0, 0)])], Vec::new());
    let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
    assert_eq!(
        body_text(&e),
        "%t2 = type void (ptr addrspace(1))\n\
         \n\
         define %t2 @f0 \"k\" {\n\
         bb0:\n  store i32 1, %a0, align 4\n  ret void\n\
         }\n"
    );
}

/// A constant `getelementptr` with only zero indices is its base (the typed-pointer spelling of an
/// array's first element); one with a nonzero index stays.
#[test]
fn zero_constant_geps_are_erased() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let i64t = m.int(64);
    let tg = m.ptr(AddrSpace::Threadgroup);
    let dev = m.ptr(AddrSpace::Device);
    let arr = m.array(i32t, 4);
    let g = m.simple_global("g", arr, AddrSpace::Threadgroup, None);
    let Value::Const(base) = m.global_addr(g) else {
        panic!("a global address is a constant");
    };
    let zero = m.constant(i64t, Const::Int(0));
    let two = m.constant(i64t, Const::Int(2));
    let gep = |m: &mut ModuleBuilder, i| {
        Value::Const(m.constant(
            tg,
            Const::Expr(ConstExpr::Gep {
                src_ty: arr,
                inbounds: true,
                ops: vec![base, zero, i].into(),
            }),
        ))
    };
    let first = gep(&mut m, zero);
    let third = gep(&mut m, two);
    let one = m.const_int(i32t, 1);
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    store(&mut f, one, first);
    store(&mut f, one, third);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, vec![buffer(0, 0)])], Vec::new());
    let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
    let text = body_text(&e);
    assert!(
        text.contains("store i32 1, ptr addrspace(3) @g0, align 4"),
        "{text}"
    );
    assert!(text.contains("getelementptr inbounds"), "{text}");
}

/// `helper(i32 x) -> i32`: `x > 4 ? x * 4 : x`, with two `ret`s.
fn two_return_helper(m: &mut ModuleBuilder) -> FuncId {
    let i1 = m.int(1);
    let i32t = m.int(32);
    let hty = m.fn_ty(i32t, &[i32t]);
    let four = m.const_int(i32t, 4);
    let mut h = m.define("helper", hty);
    let (t, e) = (h.block(), h.block());
    let x = h.arg(0);
    let big = h.push(
        i1,
        InstKind::ICmp {
            pred: IntPredicate::Ugt,
            a: x,
            b: four,
        },
    );
    h.cond_br(big, t, e);
    h.switch_to(t);
    let y = h.bin(BinOp::Mul, x, four);
    h.ret(Some(y));
    h.switch_to(e);
    h.ret(Some(x));
    h.finish()
}

/// `kernel k(device i32* p)`: `*p = helper(arg)`, where `arg` is `*p` or a constant.
fn calls_helper(constant: Option<u64>) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let helper = two_return_helper(&mut m);
    let fty = m.fn_ty(void, &[dev]);
    let c = constant.map(|v| m.const_int(i32t, v));
    let mut f = m.define("k", fty);
    let x = match c {
        Some(c) => c,
        None => load(&mut f, i32t, arg(0)),
    };
    let r = call(&mut f, i32t, Callee::Direct(helper), &[x]);
    store(&mut f, r, arg(0));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, vec![buffer(0, 0)])], Vec::new());
    specialize(module, &abi, &request(&[]), &mut meter())
}

#[test]
fn a_call_is_inlined_with_its_returns_joined_by_a_phi() {
    let e = calls_helper(None).unwrap();
    assert_eq!(
        body_text(&e),
        "%t2 = type void (ptr addrspace(1))\n\
         \n\
         define %t2 @f0 \"k\" {\n\
         bb0:\n  %i0 = load i32, %a0, align 4\n  br bb1\n\
         bb1:\n  %i2 = icmp ugt %i0, i32 4\n  br %i2, bb2, bb3\n\
         bb2:\n  %i4 = mul i32 %i0, i32 4\n  br bb4\n\
         bb3:\n  br bb4\n\
         bb4:\n  %i7 = phi i32 [%i4, bb2], [%i0, bb3]\n  store %i7, %a0, align 4\n  ret void\n\
         }\n"
    );
    assert_eq!(e.module.parts().funcs.len(), 1, "the helper is pruned");
    // SCCP sees the constant argument through the inlined body.
    let e = calls_helper(Some(5)).unwrap();
    assert!(
        body_text(&e).contains("store i32 20, %a0"),
        "{}",
        body_text(&e)
    );
}

/// Inlining charges exactly the body it builds: the continuation PHI's incoming list is charged
/// once, by `Body::push_inst` as the PHI is pushed, and each frame's argument list is scratch the
/// inliner returns with its tables.
#[test]
fn an_inlined_body_is_charged_exactly_its_footprint() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let helper = two_return_helper(&mut m);
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    let x = load(&mut f, i32t, arg(0));
    let r = call(&mut f, i32t, Callee::Direct(helper), &[x]);
    store(&mut f, r, arg(0));
    f.ret(None);
    let k = f.finish();
    let mut parts = m.finish().unwrap().into_parts();
    let mut ledger = meter();
    let body = inline::inline(
        &parts.funcs,
        (&parts.types, &parts.symbols),
        &mut parts.consts,
        k,
        &mut ledger,
        |_, _| Ok(()),
    )
    .unwrap()
    .expect("the helper is inlined");
    let joins: Vec<usize> = body
        .insts
        .values()
        .filter_map(|inst| match &inst.kind {
            InstKind::Phi { incoming, .. } => Some(incoming.len()),
            _ => None,
        })
        .collect();
    assert_eq!(joins, [2], "the helper's two returns meet in one PHI");
    assert_eq!(ledger.ledger(), body.footprint() as u64);
    // The inliner's counts are exact, so its arenas hold exactly the body: no doubling slack,
    // which at two million instructions was half the arena.
    let exact = |b: &Body| {
        b.insts.capacity_bytes() == b.insts.len() * std::mem::size_of::<crate::ir::Inst>()
            && b.blocks.capacity_bytes() == b.blocks.len() * std::mem::size_of::<crate::ir::Block>()
    };
    assert!(exact(&body));
    // So does a rewrite's body, which counts its survivors before it builds them.
    let store = body
        .insts
        .iter()
        .find_map(|(i, inst)| matches!(inst.kind, InstKind::Store { .. }).then_some(i))
        .expect("a store");
    let rebuilt = rewrite::scoped(&mut ledger, |m, held| {
        let mut edit = rewrite::BodyEdit::keep_all(&body, m, held)?;
        edit.delete(store);
        rewrite::apply(&body, &edit, &|c| c, m)
    })
    .unwrap();
    assert_eq!(rebuilt.insts.len() + 1, body.insts.len());
    assert!(exact(&rebuilt));
    assert_eq!(
        ledger.ledger(),
        (body.footprint() + rebuilt.footprint()) as u64
    );
}

/// Every defined function is erased before inlining, so erasing the body the inliner built from
/// them finds nothing to rewrite and copies nothing: the ledger and its peak do not move (that copy
/// beside a two-million-instruction inlined body was its translation's peak).
#[test]
fn an_inlined_body_of_erased_functions_is_not_copied_to_erase_it() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let seven = m.const_int(i32t, 7);
    // `helper(p)`: *(p as the same pointer type) = 7, a no-op cast to erase.
    let hty = m.fn_ty(void, &[dev]);
    let mut h = m.define("helper", hty);
    let q = h.push(
        dev,
        InstKind::Cast {
            op: CastOp::Bitcast,
            v: h.arg(0),
        },
    );
    store(&mut h, seven, q);
    h.ret(None);
    let helper = h.finish();
    let mut f = m.define("k", hty);
    for _ in 0..8 {
        call(&mut f, void, Callee::Direct(helper), &[arg(0)]);
    }
    f.ret(None);
    let k = f.finish();
    let mut parts = m.finish().unwrap().into_parts();
    let mut ledger = meter();
    let emap = erase::erase_consts(&mut parts.consts, &mut ledger).unwrap();
    erase_functions(&mut parts, &emap, &mut ledger).unwrap();
    let inlined = inline::inline(
        &parts.funcs,
        (&parts.types, &parts.symbols),
        &mut parts.consts,
        k,
        &mut ledger,
        |_, _| Ok(()),
    )
    .unwrap()
    .expect("the helper is inlined");
    let stores = |b: &Body| {
        b.insts
            .values()
            .filter(|i| matches!(i.kind, InstKind::Store { .. }))
            .count()
    };
    let casts = |b: &Body| {
        b.insts
            .values()
            .filter(|i| matches!(i.kind, InstKind::Cast { .. }))
            .count()
    };
    assert_eq!((stores(&inlined), casts(&inlined)), (8, 0));
    let before = (ledger.ledger(), ledger.ledger_peak());
    let body = erased_entry(&parts, k, Some(inlined), &emap, &mut ledger).unwrap();
    assert_eq!((ledger.ledger(), ledger.ledger_peak()), before, "no copy");
    assert_eq!((stores(&body), casts(&body)), (8, 0));
}

#[test]
fn nested_void_and_non_returning_callees_inline() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let seven = m.const_int(i32t, 7);
    // `inner(p)`: *p = 7. `outer(p)`: inner(p); inner(p). `trap() -> i32`: unreachable.
    let sty = m.fn_ty(void, &[dev]);
    let mut inner = m.define("inner", sty);
    store(&mut inner, seven, arg(0));
    inner.ret(None);
    let inner = inner.finish();
    let mut outer = m.define("outer", sty);
    call(&mut outer, void, Callee::Direct(inner), &[arg(0)]);
    call(&mut outer, void, Callee::Direct(inner), &[arg(0)]);
    outer.ret(None);
    let outer = outer.finish();
    let tty = m.fn_ty(i32t, &[]);
    let mut trap = m.define("trap", tty);
    trap.push_void(InstKind::Unreachable);
    let trap = trap.finish();
    let fty = m.fn_ty(void, &[dev, dev]);
    let i1 = m.int(1);
    let one = m.const_int(i1, 1);
    let mut f = m.define("k", fty);
    let (t, e) = (f.block(), f.block());
    call(&mut f, void, Callee::Direct(outer), &[arg(1)]);
    f.cond_br(one, t, e);
    f.switch_to(t);
    f.ret(None);
    f.switch_to(e);
    let r = call(&mut f, i32t, Callee::Direct(trap), &[]);
    store(&mut f, r, arg(0));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(
        vec![kernel(k, vec![buffer(0, 0), buffer(1, 1)])],
        Vec::new(),
    );
    let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
    let text = body_text(&e);
    assert_eq!(text.matches("store i32 7, %a1").count(), 2, "{text}");
    assert!(
        !text.contains("unreachable"),
        "the dead call is gone: {text}"
    );
    assert_eq!(e.module.parts().funcs.len(), 1);
}

/// A call cycle the entry is outside of runs on an explicit call stack in the entry; one through
/// the entry itself is `Recursion`.
#[test]
fn recursion_runs_on_a_call_stack_and_through_the_entry_is_refused() {
    for through_entry in [false, true] {
        let mut m = ModuleBuilder::new();
        let void = m.void();
        let fty = m.fn_ty(void, &[]);
        // f0 = a, f1 = b, f2 = k: a calls b, b calls a (or the kernel k), k calls a.
        let mut a = m.define("a", fty);
        call(&mut a, void, Callee::Direct(FuncId::new(1)), &[]);
        a.ret(None);
        let a = a.finish();
        let mut b = m.define("b", fty);
        let closing = if through_entry { FuncId::new(2) } else { a };
        call(&mut b, void, Callee::Direct(closing), &[]);
        b.ret(None);
        b.finish();
        let mut f = m.define("k", fty);
        call(&mut f, void, Callee::Direct(a), &[]);
        f.ret(None);
        let k = f.finish();
        let module = m.finish().unwrap();
        let abi = abi(vec![kernel(k, Vec::new())], Vec::new());
        let result = specialize(module, &abi, &request(&[]), &mut meter());
        if through_entry {
            let Err(Error::Unsupported(u)) = result else {
                panic!("{result:?}");
            };
            assert_eq!(u.feature, Feature::Recursion);
        } else {
            let e = result.unwrap_or_else(|e| panic!("{e}"));
            let text = body_text(&e);
            assert_eq!(e.module.parts().funcs.len(), 1, "{text}");
            // Metal's byte model at depth 1: the cycle's two 16-byte frames, so two frames, the
            // root's activation and one call.
            assert!(text.contains("= type [2 x "), "{text}");
        }
    }
}

#[test]
fn the_inline_cap_is_hit_before_the_body_is_built() {
    // f_i calls f_{i+1} twice: the inlined entry has about 2^25 instructions from 26 tiny
    // functions. The pre-pass refuses it; nothing proportional to 2^25 is ever charged.
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let fty = m.fn_ty(void, &[]);
    let mut leaf = m.define("f25", fty);
    leaf.ret(None);
    let mut next = leaf.finish();
    for i in (0..25).rev() {
        let mut f = m.define(&format!("f{i}"), fty);
        call(&mut f, void, Callee::Direct(next), &[]);
        call(&mut f, void, Callee::Direct(next), &[]);
        f.ret(None);
        next = f.finish();
    }
    let mut f = m.define("k", fty);
    call(&mut f, void, Callee::Direct(next), &[]);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, Vec::new())], Vec::new());
    let mut meter = meter();
    let err = specialize(module, &abi, &request(&[]), &mut meter).unwrap_err();
    assert_eq!(err, Error::LimitExceeded(Limit::InlinedInstructions));
    assert!(meter.ledger_peak() < 64 * 1024, "{}", meter.ledger_peak());
}

#[test]
fn a_byval_callee_is_refused_typed() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let thread = m.ptr(AddrSpace::Thread);
    let hty = m.fn_ty(void, &[thread]);
    let mut h = m.define("helper", hty);
    h.ret(None);
    let helper = h.finish();
    m.func_mut(helper).unwrap().params[0].attrs.byval = Some(i32t);
    let fty = m.fn_ty(void, &[]);
    let one = m.const_int(i32t, 1);
    let mut f = m.define("k", fty);
    let slot = f.push(
        thread,
        InstKind::Alloca {
            ty: i32t,
            count: one,
            align: None,
        },
    );
    call(&mut f, void, Callee::Direct(helper), &[slot]);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, Vec::new())], Vec::new());
    let Error::Unsupported(u) = specialize(module, &abi, &request(&[]), &mut meter()).unwrap_err()
    else {
        panic!();
    };
    assert_eq!(u.feature, Feature::DirectCall);
    // At the call, naming the callee.
    assert_eq!(u.symbol, Some(DiagName::new("helper")));
    assert_eq!(
        u.location,
        Some(Location::function(DiagName::new("k"), Some(0), Some(1)))
    );
}

#[test]
fn other_entries_and_unreached_globals_are_pruned() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let seven = m.constant(i32t, Const::Int(7));
    let table = m.simple_global("table", i32t, AddrSpace::Constant, Some(seven));
    let unused = m.simple_global("unused", i32t, AddrSpace::Constant, Some(seven));
    let fty = m.fn_ty(void, &[dev]);
    let taddr = m.global_addr(table);
    let uaddr = m.global_addr(unused);
    let mut other = m.define("other", fty);
    let v = load(&mut other, i32t, uaddr);
    store(&mut other, v, arg(0));
    other.ret(None);
    let other = other.finish();
    let mut f = m.define("k", fty);
    let v = load(&mut f, i32t, taddr);
    store(&mut f, v, arg(0));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(
        vec![
            kernel(other, vec![buffer(0, 0)]),
            kernel(k, vec![buffer(0, 0)]),
        ],
        Vec::new(),
    );
    let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
    let parts = e.module.parts();
    assert_eq!(parts.funcs.len(), 1);
    assert_eq!(
        parts.globals.len(),
        1,
        "the loaded global stays (it is not constant)"
    );
    assert!(parts.metadata.is_empty() && parts.named_md.is_empty());
    let name = parts.globals.values().next().unwrap().name;
    assert_eq!(parts.symbols.get(name), Some(&b"table"[..]));
}

#[test]
fn predicate_gated_parameters_resolve_from_their_globals() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i8t = m.int(8);
    let dev = m.ptr(AddrSpace::Device);
    let (off, on) = (
        m.constant(i8t, Const::Int(0)),
        m.constant(i8t, Const::Int(1)),
    );
    let p_off = m.simple_global("p_off", i8t, AddrSpace::Constant, Some(off));
    let p_on = m.simple_global("p_on", i8t, AddrSpace::Constant, Some(on));
    let fty = m.fn_ty(void, &[dev, dev, dev]);
    let mut f = m.define("k", fty);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let gated = |param, index, global| ParamBinding {
        predicate: Some(Predicate {
            global,
            name: DiagName::new("p"),
        }),
        ..buffer(param, index)
    };
    let mut params = vec![buffer(0, 0), gated(1, 1, p_off), gated(2, 2, p_on)];
    if let ParamRole::Buffer(b) = &mut params[0].role {
        b.location.count = AbiInt::Global(p_on);
    }
    let abi = abi(vec![kernel(k, params)], Vec::new());
    let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
    let kept: Vec<u32> = e.ep.params.iter().map(|p| p.param).collect();
    assert_eq!(kept, [0, 2]);
    assert!(e.ep.params.iter().all(|p| p.predicate.is_none()));
    let ParamRole::Buffer(b) = &e.ep.params[0].role else {
        panic!();
    };
    assert_eq!(b.location.count, AbiInt::Const(1));
}

#[test]
fn request_errors_are_typed() {
    let spec = |constants: &[(u32, ConstValue)], entry: &str| {
        let (module, abi) = gated_store();
        let mut r = request(constants);
        r.entry = entry.into();
        specialize(module, &abi, &r, &mut meter()).unwrap_err()
    };
    assert_eq!(
        spec(&[(1, int(0))], "k"),
        Error::InvalidRequest(RequestError::FunctionConstantIndex { index: 1 })
    );
    let float = ConstValue {
        ty: DataType::Float,
        bytes: 1f32.to_le_bytes().into(),
    };
    assert_eq!(
        spec(&[(0, float)], "k"),
        Error::InvalidRequest(RequestError::FunctionConstantType { index: 0 })
    );
    let short = ConstValue {
        ty: DataType::Int,
        bytes: Box::new([1, 2]),
    };
    assert_eq!(
        spec(&[(0, short)], "k"),
        Error::InvalidRequest(RequestError::FunctionConstantType { index: 0 })
    );
    assert_eq!(
        spec(&[], "missing"),
        Error::InvalidRequest(RequestError::EntryNotFound {
            requested: DiagName::new("missing"),
            available: 1,
        })
    );
}

/// An internal `air.static_init` function whose body `body` writes; `k` is the entry.
fn static_init(m: &mut ModuleBuilder, name: &str, body: impl FnOnce(&mut FunctionBuilder<'_>)) {
    let void = m.void();
    let fty = m.fn_ty(void, &[]);
    let mut init = m.define(name, fty);
    body(&mut init);
    init.ret(None);
    let init = init.finish();
    let f = m.func_mut(init).unwrap();
    f.section = Some(Section::StaticInit);
    f.linkage = Linkage::Internal;
}

fn normalize(f: &mut FunctionBuilder<'_>, ty: TypeId, v: Value) -> Value {
    let decl_ty = f.module().fn_ty(ty, &[ty]);
    let decl = f
        .module()
        .declare("air.normalize_function_constant_predicate.i8", decl_ty);
    call(
        f,
        ty,
        Callee::Air {
            decl,
            op: AirIntrinsic::FunctionConstant(FunctionConstant::NormalizePredicate(
                crate::intrinsics::Elem::I8,
            )),
        },
        &[v],
    )
}

fn boolean(v: u8) -> ConstValue {
    ConstValue {
        ty: DataType::Bool,
        bytes: Box::new([v]),
    }
}

/// The frontend's shape: the static initializer copies the `bool` constant `flag` into the
/// internal `copy` and stores its normalized predicate into `pred`, which gates parameter 1. The
/// kernel stores 7 to parameter 0 when `copy` is set.
fn copied_flag() -> (Module, AirAbi) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i8t = m.int(8);
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let flag = fc_global(&mut m, "flag", i8t);
    let undef = m.constant(i8t, Const::Undef);
    let zero = m.constant(i8t, Const::Int(0));
    let copy = m.simple_global("copy", i8t, AddrSpace::Constant, Some(undef));
    let pred = m.simple_global("pred", i8t, AddrSpace::Constant, Some(zero));
    let (flag_addr, copy_addr, pred_addr) = (
        m.global_addr(flag),
        m.global_addr(copy),
        m.global_addr(pred),
    );
    static_init(&mut m, "init", |f| {
        let v = load(f, i8t, flag_addr);
        store(f, v, copy_addr);
        let p = normalize(f, i8t, v);
        store(f, p, pred_addr);
    });
    let fty = m.fn_ty(void, &[dev, dev]);
    let (seven, zero8) = (m.const_int(i32t, 7), m.const_int(i8t, 0));
    let i1 = m.int(1);
    let mut f = m.define("k", fty);
    let (then, exit) = (f.block(), f.block());
    let c = load(&mut f, i8t, copy_addr);
    let set = f.push(
        i1,
        InstKind::ICmp {
            pred: IntPredicate::Ne,
            a: c,
            b: zero8,
        },
    );
    f.cond_br(set, then, exit);
    f.switch_to(then);
    store(&mut f, seven, arg(0));
    f.br(exit);
    f.switch_to(exit);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let gated = ParamBinding {
        predicate: Some(Predicate {
            global: pred,
            name: DiagName::new("pred"),
        }),
        ..buffer(1, 1)
    };
    let abi = abi(
        vec![kernel(k, vec![buffer(0, 0), gated])],
        vec![fc_decl(0, flag, i8t)],
    );
    (module, abi)
}

#[test]
fn a_static_initializer_bakes_its_stores_into_constant_globals() {
    let (module, abi) = copied_flag();
    let e = specialize(module, &abi, &request(&[(0, boolean(1))]), &mut meter()).unwrap();
    let kept: Vec<u32> = e.ep.params.iter().map(|p| p.param).collect();
    assert_eq!(kept, [0, 1], "the predicate is on");
    assert_eq!(
        body_text(&e),
        "%t2 = type void (ptr addrspace(1), ptr addrspace(1))\n\
         \n\
         define %t2 @f0 \"k\" {\n\
         bb0:\n  br bb1\n\
         bb1:\n  store i32 7, %a0, align 4\n  br bb2\n\
         bb2:\n  ret void\n\
         }\n"
    );

    let (module, abi) = copied_flag();
    let e = specialize(module, &abi, &request(&[(0, boolean(0))]), &mut meter()).unwrap();
    let kept: Vec<u32> = e.ep.params.iter().map(|p| p.param).collect();
    assert_eq!(kept, [0], "the predicate is off");
    assert!(!body_text(&e).contains("store"), "{}", body_text(&e));
}

#[test]
fn static_values_resolve_abi_globals_without_canonicalizing_a_body() {
    // The static initializer's stores answer the predicate for each constant, and the module is
    // not consumed: the kernel's value use of the copy, which `specialize` refuses when the
    // constant is unsupplied, is never looked at.
    let (module, abi) = copied_flag();
    let ParamRole::Buffer(_) = &abi.entries[0].params[1].role else {
        panic!();
    };
    let pred = abi.entries[0].params[1].predicate.as_ref().unwrap().global;
    let flag = abi.function_constants[0].global;
    let on = static_values(
        module.clone(),
        &abi,
        &[(0, boolean(1))].into(),
        &[pred],
        &mut meter(),
    );
    assert_eq!(on.unwrap(), [Ok(1)]);
    let off = static_values(
        module.clone(),
        &abi,
        &[(0, boolean(0))].into(),
        &[pred],
        &mut meter(),
    );
    assert_eq!(off.unwrap(), [Ok(0)]);
    // The predicate of an unsupplied constant is 0; the constant itself needs input.
    let unsupplied = static_values(
        module.clone(),
        &abi,
        &BTreeMap::new(),
        &[pred, flag],
        &mut meter(),
    );
    assert_eq!(
        unsupplied.unwrap(),
        [
            Ok(0),
            Err(Error::NeedsInput(MissingInput::FunctionConstant {
                index: 0
            }))
        ]
    );
    // Without globals only the constants are checked; nothing is evaluated.
    assert_eq!(
        static_values(
            module.clone(),
            &abi,
            &[(0, boolean(1))].into(),
            &[],
            &mut meter()
        ),
        Ok(Vec::new())
    );
    // Request errors are the ones `specialize` reports.
    assert_eq!(
        static_values(
            module,
            &abi,
            &[(1, boolean(1))].into(),
            &[pred],
            &mut meter()
        ),
        Err(Error::InvalidRequest(RequestError::FunctionConstantIndex {
            index: 1
        }))
    );
}

/// A module whose static initializer computes `pred` from function constant 0 and `bits` through
/// float arithmetic the interpreter does not evaluate (`frem`, then a bitcast), with `branch`
/// choosing whether it then branches on the float result.
fn float_beside_flag(branch: bool) -> (Module, AirAbi, GlobalId, GlobalId) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i8t = m.int(8);
    let i32t = m.int(32);
    let f32t = m.float();
    let dev = m.ptr(AddrSpace::Device);
    let flag = fc_global(&mut m, "flag", i8t);
    let zero8 = m.constant(i8t, Const::Int(0));
    let zero32 = m.constant(i32t, Const::Int(0));
    let pred = m.simple_global("pred", i8t, AddrSpace::Constant, Some(zero8));
    let bits = m.simple_global("bits", i32t, AddrSpace::Constant, Some(zero32));
    let one = m.constant(f32t, Const::Float(f32::to_bits(1.0).into()));
    let (flag_addr, pred_addr, bits_addr) = (
        m.global_addr(flag),
        m.global_addr(pred),
        m.global_addr(bits),
    );
    static_init(&mut m, "init", |f| {
        let v = load(f, i8t, flag_addr);
        let p = normalize(f, i8t, v);
        store(f, p, pred_addr);
        let sum = f.bin(BinOp::FRem, Value::Const(one), Value::Const(one));
        let as_int = f.push(
            i32t,
            InstKind::Cast {
                op: CastOp::Bitcast,
                v: sum,
            },
        );
        store(f, as_int, bits_addr);
        if branch {
            let i1 = f.module().int(1);
            let zero = f.module().const_int(i32t, 0);
            let nonzero = f.push(
                i1,
                InstKind::ICmp {
                    pred: IntPredicate::Ne,
                    a: as_int,
                    b: zero,
                },
            );
            let (then, exit) = (f.block(), f.block());
            f.cond_br(nonzero, then, exit);
            f.switch_to(then);
            f.br(exit);
            f.switch_to(exit);
        }
    });
    let fty = m.fn_ty(void, &[dev, dev]);
    let mut f = m.define("k", fty);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let gated = ParamBinding {
        predicate: Some(Predicate {
            global: pred,
            name: DiagName::new("pred"),
        }),
        ..buffer(1, 1)
    };
    let abi = abi(
        vec![kernel(k, vec![buffer(0, 0), gated])],
        vec![fc_decl(0, flag, i8t)],
    );
    (module, abi, pred, bits)
}

#[test]
fn an_unevaluated_float_operation_leaves_only_the_globals_it_reaches_unknown() {
    let static_init_unsupported = |result: &Result<u64, Error>| matches!(result, Err(Error::Unsupported(u)) if u.feature == Feature::StaticInitializer);
    let (module, abi, pred, bits) = float_beside_flag(false);
    let values = static_values(
        module.clone(),
        &abi,
        &[(0, boolean(1))].into(),
        &[pred, bits],
        &mut meter(),
    )
    .unwrap();
    // The predicate is a pure function of the constant; the float result taints only `bits`.
    assert_eq!(values[0], Ok(1));
    assert!(static_init_unsupported(&values[1]), "{values:?}");
    // Specialization still refuses the module: it must bake every stored global.
    let err = specialize(module, &abi, &request(&[(0, boolean(1))]), &mut meter()).unwrap_err();
    assert!(
        matches!(&err, Error::Unsupported(u) if u.feature == Feature::StaticInitializer),
        "{err:?}"
    );
    // A branch on the float result leaves the path unknown: the whole evaluation refuses.
    let (module, abi, pred, _) = float_beside_flag(true);
    let err = static_values(
        module,
        &abi,
        &[(0, boolean(1))].into(),
        &[pred],
        &mut meter(),
    )
    .unwrap_err();
    assert!(
        matches!(&err, Error::Unsupported(u) if u.feature == Feature::StaticInitializer),
        "{err:?}"
    );
}

#[test]
fn an_unsupplied_constant_normalizes_to_a_false_predicate() {
    // The predicate of an undefined constant is 0, so the gated parameter is absent; the kernel's
    // value use of the copy still needs the constant.
    let (module, flag_abi) = copied_flag();
    let err = specialize(module, &flag_abi, &request(&[]), &mut meter()).unwrap_err();
    assert_eq!(
        err,
        Error::NeedsInput(MissingInput::FunctionConstant { index: 0 })
    );

    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i8t = m.int(8);
    let dev = m.ptr(AddrSpace::Device);
    let flag = fc_global(&mut m, "flag", i8t);
    let zero = m.constant(i8t, Const::Int(0));
    let pred = m.simple_global("pred", i8t, AddrSpace::Constant, Some(zero));
    let (flag_addr, pred_addr) = (m.global_addr(flag), m.global_addr(pred));
    static_init(&mut m, "init", |f| {
        let v = load(f, i8t, flag_addr);
        let one = f.module().const_int(i8t, 1);
        let not = f.bin(BinOp::Xor, v, one);
        let p = normalize(f, i8t, not);
        store(f, p, pred_addr);
    });
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let gated = ParamBinding {
        predicate: Some(Predicate {
            global: pred,
            name: DiagName::new("pred"),
        }),
        ..buffer(0, 0)
    };
    let abi = abi(vec![kernel(k, vec![gated])], vec![fc_decl(0, flag, i8t)]);
    let e = specialize(module.clone(), &abi, &request(&[]), &mut meter()).unwrap();
    assert!(e.ep.params.is_empty(), "!undefined is still off");
    let e = specialize(module, &abi, &request(&[(0, boolean(0))]), &mut meter()).unwrap();
    assert_eq!(e.ep.params.len(), 1, "!false is on");
}

/// A kernel storing the `i32` global `g` to its buffer, after `init` ran.
fn stores_global(init: impl FnOnce(&mut ModuleBuilder, GlobalId)) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let zero = m.constant(i32t, Const::Int(0));
    let g = m.simple_global("g", i32t, AddrSpace::Constant, Some(zero));
    init(&mut m, g);
    let gaddr = m.global_addr(g);
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    let v = load(&mut f, i32t, gaddr);
    store(&mut f, v, arg(0));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, vec![buffer(0, 0)])], Vec::new());
    specialize(module, &abi, &request(&[]), &mut meter())
}

fn stored(e: &Entry) -> String {
    body_text(e)
        .lines()
        .find(|l| l.contains("store"))
        .unwrap_or("")
        .trim()
        .to_string()
}

#[test]
fn static_initializers_run_in_global_ctors_order() {
    let e = stores_global(|m, g| {
        let i32t = m.int(32);
        let gaddr = m.global_addr(g);
        let (two, three) = (m.const_int(i32t, 2), m.const_int(i32t, 3));
        static_init(m, "first_defined", |f| store(f, two, gaddr));
        static_init(m, "second_defined", |f| store(f, three, gaddr));
        // Priority 2 runs `first_defined` after `second_defined` (priority 1).
        let ptr0 = m.ptr(AddrSpace::Thread);
        let entry_ty = m.struct_ty(&[i32t, ptr0, ptr0]);
        let list_ty = m.array(entry_ty, 2);
        let null = m.constant(ptr0, Const::Null);
        let mut entries = Vec::new();
        for (prio, f) in [(2, 0u32), (1, 1)] {
            let p = m.constant(i32t, Const::Int(prio));
            let fa = m.constant(ptr0, Const::FuncAddr(FuncId::new(f)));
            entries.push(m.constant(entry_ty, Const::Aggregate(vec![p, fa, null].into())));
        }
        let list = m.constant(list_ty, Const::Aggregate(entries.into()));
        let name = m.sym("llvm.global_ctors");
        m.global(Global {
            name,
            value_ty: list_ty,
            space: AddrSpace::Thread,
            constant: false,
            init: Some(list),
            externally_initialized: false,
            visibility: Default::default(),
            unnamed_addr: Default::default(),
            linkage: Linkage::Appending,
            align: None,
            section: None,
        });
    })
    .unwrap();
    assert_eq!(stored(&e), "store i32 2, %a0, align 4");
}

/// A kernel storing `g + h` to its buffer, where `inits` defines the static initializers (no
/// `llvm.global_ctors` list) over the `i32` globals `g` and `h`, both initially 0.
fn stores_sum(inits: impl FnOnce(&mut ModuleBuilder, Value, Value)) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let zero = m.constant(i32t, Const::Int(0));
    let g = m.simple_global("g", i32t, AddrSpace::Constant, Some(zero));
    let h = m.simple_global("h", i32t, AddrSpace::Constant, Some(zero));
    let (gaddr, haddr) = (m.global_addr(g), m.global_addr(h));
    inits(&mut m, gaddr, haddr);
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    let (v, w) = (load(&mut f, i32t, gaddr), load(&mut f, i32t, haddr));
    let sum = f.bin(BinOp::Add, v, w);
    store(&mut f, sum, arg(0));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, vec![buffer(0, 0)])], Vec::new());
    specialize(module, &abi, &request(&[]), &mut meter())
}

#[test]
fn unlisted_static_initializers_that_commute_all_run() {
    // Several initializers and no `llvm.global_ctors`: AIR states no order, and none is needed
    // when no global one writes is touched by another: `writes_h` reads `h` only after its own
    // store, so every order computes 2 + 3.
    let e = stores_sum(|m, g, h| {
        let i32t = m.int(32);
        let (two, three) = (m.const_int(i32t, 2), m.const_int(i32t, 3));
        static_init(m, "writes_g", |f| store(f, two, g));
        static_init(m, "writes_h", |f| {
            store(f, three, h);
            let again = load(f, i32t, h);
            store(f, again, h);
        });
    })
    .unwrap();
    assert_eq!(stored(&e), "store i32 5, %a0, align 4");
}

#[test]
fn unlisted_static_initializers_whose_order_could_matter_refuse() {
    // Without a list, two stores to one global, or a read of a global another initializer
    // writes, compute different values in different orders: refuse rather than pick one.
    let refused = |e: Result<Entry, Error>| match e {
        Err(Error::Unsupported(u)) => u.feature == Feature::StaticInitializer,
        _ => false,
    };
    let both_write = stores_sum(|m, g, _| {
        let i32t = m.int(32);
        let (two, three) = (m.const_int(i32t, 2), m.const_int(i32t, 3));
        static_init(m, "first", |f| store(f, two, g));
        static_init(m, "second", |f| store(f, three, g));
    });
    assert!(refused(both_write));
    let one_reads = stores_sum(|m, g, h| {
        let i32t = m.int(32);
        let two = m.const_int(i32t, 2);
        static_init(m, "writes_g", |f| store(f, two, g));
        static_init(m, "copies_g", |f| {
            let v = load(f, i32t, g);
            store(f, v, h);
        });
    });
    assert!(refused(one_reads));
    // Reading the same global is no conflict (`g` itself stays a variable: nothing wrote it).
    let both_read = stores_sum(|m, g, h| {
        let i32t = m.int(32);
        static_init(m, "reads_g", |f| {
            let v = load(f, i32t, g);
            store(f, v, h);
        });
        static_init(m, "reads_g_too", |f| {
            load(f, i32t, g);
        });
    });
    assert!(both_read.is_ok(), "{both_read:?}");
}

#[test]
fn unlisted_static_initializers_refuse_what_they_cannot_follow() {
    // Commuting is judged per named module global, so an unlisted initializer that writes through
    // a pointer naming no global, calls a function without a body, or touches memory other than a
    // constant-space module global must refuse, beside one that commutes with it.
    let with = |second: &dyn Fn(&mut ModuleBuilder, Value)| {
        stores_sum(|m, g, h| {
            let i32t = m.int(32);
            let two = m.const_int(i32t, 2);
            static_init(m, "writes_g", |f| store(f, two, g));
            second(m, h);
        })
    };
    let unnamed = with(&|m, _| {
        let (i32t, i64t) = (m.int(32), m.int(64));
        let ptr = m.ptr(AddrSpace::Constant);
        let addr = m.constant(i64t, Const::Int(64));
        let cast = Const::Expr(ConstExpr::Cast {
            op: CastOp::IntToPtr,
            ops: [addr],
        });
        let p = Value::Const(m.constant(ptr, cast));
        let three = m.const_int(i32t, 3);
        static_init(m, "writes_an_address", |f| store(f, three, p));
    });
    let opaque = with(&|m, _| {
        let void = m.void();
        let fty = m.fn_ty(void, &[]);
        let decl = m.declare("opaque", fty);
        static_init(m, "calls_a_declaration", |f| {
            call(f, void, Callee::Direct(decl), &[]);
        });
    });
    let private = with(&|m, _| {
        let i32t = m.int(32);
        let thread = m.ptr(AddrSpace::Thread);
        let (one, three) = (m.const_int(i32t, 1), m.const_int(i32t, 3));
        static_init(m, "writes_its_stack", |f| {
            let slot = f.push(
                thread,
                InstKind::Alloca {
                    ty: i32t,
                    count: one,
                    align: None,
                },
            );
            store(f, three, slot);
        });
    });
    let device = with(&|m, _| {
        let i32t = m.int(32);
        let zero = m.constant(i32t, Const::Int(0));
        let d = m.simple_global("d", i32t, AddrSpace::Device, Some(zero));
        let daddr = m.global_addr(d);
        let three = m.const_int(i32t, 3);
        static_init(m, "writes_device_memory", |f| store(f, three, daddr));
    });
    for (what, e) in [
        ("an address naming no global", unnamed),
        ("a call without a body", opaque),
        ("a stack slot", private),
        ("a device global", device),
    ] {
        match e {
            Err(Error::Unsupported(u)) => {
                assert_eq!(u.feature, Feature::StaticInitializer, "{what}")
            }
            other => panic!("{what}: {other:?}"),
        }
    }
}

#[test]
fn a_deeply_nested_written_global_is_rebuilt_on_a_small_stack() {
    // Nothing caps type nesting, and a chain of one-element arrays has two leaves at any depth,
    // so rebuilding the written global's initializer must not recurse per level.
    const DEPTH: usize = 20_000;
    let run = || {
        let mut m = ModuleBuilder::new();
        let void = m.void();
        let i32t = m.int(32);
        let dev = m.ptr(AddrSpace::Device);
        let mut ty = m.struct_ty(&[i32t, i32t]);
        for _ in 0..DEPTH {
            ty = m.array(ty, 1);
        }
        let null = m.constant(ty, Const::Null);
        let g = m.simple_global("g", ty, AddrSpace::Constant, Some(null));
        let gaddr = m.global_addr(g);
        let five = m.const_int(i32t, 5);
        static_init(&mut m, "init", |f| store(f, five, gaddr));
        let fty = m.fn_ty(void, &[dev]);
        let mut f = m.define("k", fty);
        let v = load(&mut f, i32t, gaddr);
        store(&mut f, v, arg(0));
        f.ret(None);
        let k = f.finish();
        let module = m.finish().unwrap();
        let abi = abi(vec![kernel(k, vec![buffer(0, 0)])], Vec::new());
        let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
        // The global's rebuilt initializer: descend its first members to the first leaf,
        // counting the aggregate levels on the way.
        let parts = e.module.parts();
        let init = parts
            .globals
            .iter()
            .filter_map(|(_, g)| g.init)
            .find(|&c| matches!(parts.consts.get(c).unwrap().value, Const::Aggregate(_)))
            .unwrap();
        let (mut c, mut levels) = (init, 0);
        while let Const::Aggregate(elems) = &parts.consts.get(c).unwrap().value {
            c = elems[0];
            levels += 1;
        }
        (levels, parts.consts.get(c).unwrap().value.clone())
    };
    let (levels, first) = std::thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(run)
        .unwrap()
        .join()
        .unwrap();
    assert_eq!(
        levels,
        DEPTH + 1,
        "every array level and the struct are rebuilt"
    );
    assert_eq!(first, Const::Int(5));
}

#[test]
fn static_initializers_call_helpers_and_branch_on_constants() {
    let e = stores_global(|m, g| {
        let i32t = m.int(32);
        let i1 = m.int(1);
        let gaddr = m.global_addr(g);
        let hty = m.fn_ty(i32t, &[i32t]);
        let mut h = m.define("helper", hty);
        let (t, e) = (h.block(), h.block());
        let x = h.arg(0);
        let four = h.module().const_int(i32t, 4);
        let big = h.push(
            i1,
            InstKind::ICmp {
                pred: IntPredicate::Ugt,
                a: x,
                b: four,
            },
        );
        h.cond_br(big, t, e);
        h.switch_to(t);
        let y = h.bin(BinOp::Mul, x, four);
        h.ret(Some(y));
        h.switch_to(e);
        h.ret(Some(x));
        let helper = h.finish();
        let five = m.const_int(i32t, 5);
        static_init(m, "init", |f| {
            let r = call(f, i32t, Callee::Direct(helper), &[five]);
            store(f, r, gaddr);
        });
    })
    .unwrap();
    assert_eq!(stored(&e), "store i32 20, %a0, align 4");
}

/// The value a kernel stores after a static initializer computes the `ty` global `g` as
/// `op(uitofp(c), k)` (`op(k, uitofp(c))` with `swap`), where `c` is the `u32` function constant
/// 0 supplied as `n` and `k` the `ty` constant with bits `k`: the specialized entry's store, or
/// the error.
fn float_static_init(ty: Type, op: BinOp, k: u64, swap: bool, n: u32) -> Result<String, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let fty_ = m.ty(ty);
    let dev = m.ptr(AddrSpace::Device);
    let c = fc_global(&mut m, "c", i32t);
    let zero = m.constant(fty_, Const::Float(0));
    let g = m.simple_global("g", fty_, AddrSpace::Constant, Some(zero));
    let k = m.constant(fty_, Const::Float(k));
    let (caddr, gaddr) = (m.global_addr(c), m.global_addr(g));
    static_init(&mut m, "init", |f| {
        let v = load(f, i32t, caddr);
        let x = f.push(
            fty_,
            InstKind::Cast {
                op: CastOp::UIToFP,
                v,
            },
        );
        let (a, b) = match swap {
            false => (x, Value::Const(k)),
            true => (Value::Const(k), x),
        };
        let r = f.bin(op, a, b);
        store(f, r, gaddr);
    });
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    let v = load(&mut f, fty_, gaddr);
    store(&mut f, v, arg(0));
    f.ret(None);
    let kfn = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(
        vec![kernel(kfn, vec![buffer(0, 0)])],
        vec![fc_decl(0, c, i32t)],
    );
    let e = specialize(module, &abi, &request(&[(0, int(n))]), &mut meter())?;
    Ok(stored(&e))
}

#[test]
fn static_initializer_float_arithmetic_folds_where_every_ieee_evaluation_agrees() {
    // MPSHistogram's `kMaxBinIndex = float(kNumBins) - 1`: 2^2 - 1, and the exact cancellation 1 -
    // 1, which is +0 under round to nearest even.
    let minus_one = u64::from(f32::to_bits(-1.0));
    let fold = |ty, op, k, swap, n| float_static_init(ty, op, k, swap, n).unwrap();
    assert_eq!(
        fold(Type::Float, BinOp::FAdd, minus_one, false, 4),
        "store float 0x40400000, %a0, align 4"
    );
    assert_eq!(
        fold(Type::Float, BinOp::FAdd, minus_one, false, 1),
        "store float 0x0, %a0, align 4"
    );
    // Deblock's `kNormalizePixel = 1 / half(255)`: the half nearest 1/255 (1028 * 2^-18).
    assert_eq!(
        fold(Type::Half, BinOp::FDiv, 0x3c00, true, 255),
        "store half 0x1C04, %a0, align 4"
    );
    // 3 * 2^24, and 1/3 rounded to the nearest float.
    assert_eq!(
        fold(Type::Float, BinOp::FMul, 0x4b80_0000, false, 3),
        "store float 0x4C400000, %a0, align 4"
    );
    assert_eq!(
        fold(
            Type::Float,
            BinOp::FDiv,
            u64::from(f32::to_bits(1.0)),
            true,
            3
        ),
        "store float 0x3EAAAAAB, %a0, align 4"
    );
    assert_eq!(
        fold(Type::Double, BinOp::FDiv, f64::to_bits(1.0), true, 3),
        "store double 0x3FD5555555555555, %a0, align 4"
    );
    assert_eq!(
        fold(
            Type::Float,
            BinOp::FSub,
            u64::from(f32::to_bits(0.5)),
            false,
            3
        ),
        "store float 0x40200000, %a0, align 4"
    );

    // Where evaluations can disagree, the initializer stays refused: a zero operand (the sign
    // of the result under `nsz`), an infinite operand (`half(65535)` overflows), a subnormal
    // result (flushed or not), an overflowing result, and `frem`.
    let refused = |ty, op, k, swap, n| {
        let err = float_static_init(ty, op, k, swap, n).unwrap_err();
        assert!(
            matches!(&err, Error::Unsupported(u) if u.feature == Feature::StaticInitializer),
            "{err:?}"
        );
    };
    refused(Type::Float, BinOp::FAdd, minus_one, false, 0);
    refused(Type::Half, BinOp::FDiv, 0x3c00, true, 65535);
    refused(Type::Half, BinOp::FDiv, 0x3c00, true, 32768);
    refused(Type::Float, BinOp::FMul, 0x7f00_0000, false, 4);
    refused(Type::Float, BinOp::FRem, minus_one, false, 4);
}

/// The value a kernel stores after a static initializer stores the `i64` `first` (if any) at
/// byte 0 of the global `t`, whose type and initializer `table` makes, then copies an integer of
/// `width` bits loaded at byte `at` of `t` into the global the kernel reads: the specialized
/// entry's store, or the error.
fn table_load(
    table: impl FnOnce(&mut ModuleBuilder) -> (TypeId, crate::ir::ConstId),
    first: Option<u64>,
    width: u8,
    at: u64,
) -> Result<String, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i8t = m.int(8);
    let i64t = m.int(64);
    let wt = m.int(width);
    let dev = m.ptr(AddrSpace::Device);
    let cptr = m.ptr(AddrSpace::Constant);
    let (ty, init) = table(&mut m);
    let t = m.simple_global("t", ty, AddrSpace::Constant, Some(init));
    let zero = m.constant(wt, Const::Int(0));
    let g = m.simple_global("g", wt, AddrSpace::Constant, Some(zero));
    let (taddr, gaddr) = (m.global_addr(t), m.global_addr(g));
    let first = first.map(|v| m.const_int(i64t, v));
    let off = m.const_int(i64t, at);
    static_init(&mut m, "init", |f| {
        if let Some(v) = first {
            store(f, v, taddr);
        }
        let p = f.push(
            cptr,
            InstKind::Gep {
                src_ty: i8t,
                base: taddr,
                indices: Box::new([off]),
                inbounds: true,
            },
        );
        let v = load(f, wt, p);
        store(f, v, gaddr);
    });
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    let v = load(&mut f, wt, gaddr);
    store(&mut f, v, arg(0));
    f.ret(None);
    let kfn = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(kfn, vec![buffer(0, 0)])], Vec::new());
    let e = specialize(module, &abi, &request(&[]), &mut meter())?;
    Ok(stored(&e))
}

/// MPS's kernel-info table: the union `{ i64, [8 x i8] }`, its bytes 8..16 `0xa1..=0xa8`.
fn kernel_info(m: &mut ModuleBuilder) -> (TypeId, crate::ir::ConstId) {
    let i8t = m.int(8);
    let i64t = m.int(64);
    let bytes = m.array(i8t, 8);
    let ty = m.struct_ty(&[i64t, bytes]);
    let tail: Vec<_> = (0xa1..=0xa8u64)
        .map(|b| m.constant(i8t, Const::Int(b)))
        .collect();
    let tail = m.constant(bytes, Const::Aggregate(tail.into_boxed_slice()));
    let head = m.constant(i64t, Const::Int(0));
    (ty, m.constant(ty, Const::Aggregate(Box::new([head, tail]))))
}

#[test]
fn a_static_initializer_loads_an_integer_over_a_tables_fields() {
    // MPS's quantized matmul stores its 64-bit kernel info at byte 0 of the union `{ i64, [8 x i8]
    // }` and reads the `i40` field of a packed `<{ i32, i40 }>` view of it as an `i64` at byte 4:
    // the stored value's high four bytes, then the array's first four, in the module's
    // little-endian order.
    let info = 0x1122_3344_5566_7788;
    let load = |width, at| table_load(kernel_info, Some(info), width, at).unwrap();
    assert_eq!(
        load(64, 4),
        format!("store i64 {}, %a0, align 4", 0xa4a3_a2a1_1122_3344u64)
    );
    // Narrower loads inside the stored field and across its end.
    assert_eq!(load(16, 2), format!("store i16 {}, %a0, align 4", 0x5566));
    assert_eq!(
        load(32, 6),
        format!("store i32 {}, %a0, align 4", 0xa2a1_1122u32)
    );
    assert_eq!(
        load(64, 8),
        format!("store i64 {}, %a0, align 4", 0xa8a7_a6a5_a4a3_a2a1u64)
    );
    // A field read whole is the field, as before.
    assert_eq!(load(64, 0), format!("store i64 {info}, %a0, align 4"));

    // Bytes no integer leaf of whole bytes holds stay unread: past the table's end, padding, a
    // float's bits, and an `i1`'s unused bits.
    let refused = |r: Result<String, Error>| {
        let err = r.unwrap_err();
        assert!(
            matches!(&err, Error::Unsupported(u) if u.feature == Feature::StaticInitializer),
            "{err:?}"
        );
    };
    refused(table_load(kernel_info, Some(info), 64, 12));
    let padded = |m: &mut ModuleBuilder| {
        let i8t = m.int(8);
        let i32t = m.int(32);
        let ty = m.struct_ty(&[i8t, i32t]);
        let (a, b) = (
            m.constant(i8t, Const::Int(1)),
            m.constant(i32t, Const::Int(2)),
        );
        (ty, m.constant(ty, Const::Aggregate(Box::new([a, b]))))
    };
    refused(table_load(padded, None, 32, 0));
    assert_eq!(
        table_load(padded, None, 16, 4).unwrap(),
        "store i16 2, %a0, align 4",
        "the second field's low half"
    );
    let float = |m: &mut ModuleBuilder| {
        let f32t = m.float();
        let i32t = m.int(32);
        let ty = m.struct_ty(&[f32t, i32t]);
        let (a, b) = (
            m.constant(f32t, Const::Float(0x3f80_0000)),
            m.constant(i32t, Const::Int(2)),
        );
        (ty, m.constant(ty, Const::Aggregate(Box::new([a, b]))))
    };
    // A float's bytes are its IEEE 754 encoding: `1.0f` then `2`.
    assert_eq!(
        table_load(float, None, 64, 0).unwrap(),
        format!("store i64 {}, %a0, align 4", 0x0000_0002_3f80_0000u64)
    );
    let pointer = |m: &mut ModuleBuilder| {
        let p = m.ptr(AddrSpace::Constant);
        let i32t = m.int(32);
        let ty = m.struct_ty(&[p, i32t]);
        let (a, b) = (m.constant(p, Const::Null), m.constant(i32t, Const::Int(2)));
        (ty, m.constant(ty, Const::Aggregate(Box::new([a, b]))))
    };
    refused(table_load(pointer, None, 64, 0));
    let flag = |m: &mut ModuleBuilder| {
        let i1 = m.int(1);
        let i8t = m.int(8);
        let ty = m.struct_ty(&[i1, i8t]);
        let (a, b) = (
            m.constant(i1, Const::Int(1)),
            m.constant(i8t, Const::Int(2)),
        );
        (ty, m.constant(ty, Const::Aggregate(Box::new([a, b]))))
    };
    refused(table_load(flag, None, 16, 0));
}

/// The lanes' bits (one for a scalar) of the constant the specialized entry `e` stores.
fn stored_lanes(e: &Entry) -> Vec<u64> {
    let parts = e.module.parts();
    let body = parts.funcs.get(e.func).unwrap().body.as_ref().unwrap();
    let c = body
        .insts
        .iter()
        .find_map(|(_, i)| match i.kind {
            InstKind::Store {
                val: Value::Const(c),
                ..
            } => Some(c),
            _ => None,
        })
        .unwrap();
    let bits = |k| match parts.consts.get(k).unwrap().value {
        Const::Int(x) | Const::Float(x) => x,
        Const::Null => 0,
        ref v => panic!("{v:?}"),
    };
    match &parts.consts.get(c).unwrap().value {
        Const::Aggregate(elems) => elems.iter().map(|&k| bits(k)).collect(),
        _ => vec![bits(c)],
    }
}

/// A static initializer copies function constant 0 (of the type `ty` makes, given `bytes` as
/// `data` when `Some`) through an integer of `width` bits into the global `g` of the same type,
/// whose initializer is `undef`; the kernel stores `g`. The specialized entry, or the error.
fn punned_copy(
    ty: impl FnOnce(&mut ModuleBuilder) -> TypeId,
    width: u8,
    value: Option<(DataType, &[u8])>,
) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let dev = m.ptr(AddrSpace::Device);
    let ty = ty(&mut m);
    let wt = m.int(width);
    let c = fc_global(&mut m, "c", ty);
    let undef = m.constant(ty, Const::Undef);
    let g = m.simple_global("g", ty, AddrSpace::Constant, Some(undef));
    let (caddr, gaddr) = (m.global_addr(c), m.global_addr(g));
    static_init(&mut m, "init", |f| {
        let v = load(f, wt, caddr);
        store(f, v, gaddr);
    });
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    let v = load(&mut f, ty, gaddr);
    store(&mut f, v, arg(0));
    f.ret(None);
    let kfn = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(
        vec![kernel(kfn, vec![buffer(0, 0)])],
        vec![fc_decl(0, c, ty)],
    );
    let constants: Vec<(u32, ConstValue)> = value
        .map(|(ty, bytes)| {
            (
                0,
                ConstValue {
                    ty,
                    bytes: bytes.into(),
                },
            )
        })
        .into_iter()
        .collect();
    specialize(module, &abi, &request(&constants), &mut meter())
}

#[test]
fn a_static_initializer_copies_a_float_or_vector_constant_through_an_integer() {
    // Initializers copy a `float`, `half` or `short2` function constant into an internal global as
    // an integer of its size (`load i32` of the constant, `store i32` into the global): the global
    // holds the constant's bytes, so the kernel reads the constant.
    let float = |m: &mut ModuleBuilder| m.float();
    let e = punned_copy(float, 32, Some((DataType::Float, &0.75f32.to_le_bytes()))).unwrap();
    assert_eq!(stored_lanes(&e), [u64::from(0.75f32.to_bits())]);
    let half = |m: &mut ModuleBuilder| m.ty(Type::Half);
    let e = punned_copy(half, 16, Some((DataType::Half, &[0x00, 0x3c]))).unwrap();
    assert_eq!(stored_lanes(&e), [0x3c00]);
    let short2 = |m: &mut ModuleBuilder| {
        let i16t = m.int(16);
        m.vector(i16t, 2)
    };
    let bytes = [0x34, 0x12, 0xcd, 0xab];
    let e = punned_copy(short2, 32, Some((DataType::Short2, &bytes))).unwrap();
    assert_eq!(stored_lanes(&e), [0x1234, 0xabcd]);
    // Unsupplied, the copy is undefined as before: the global is derived from the constant.
    let err = punned_copy(float, 32, None).unwrap_err();
    assert_eq!(
        err,
        Error::NeedsInput(MissingInput::FunctionConstant { index: 0 }),
        "{err:?}"
    );
    // Through an integer narrower than the constant, a copy covers the global's float only in
    // part, and the rest of an `undef` float has no bytes to keep.
    let err = punned_copy(float, 16, Some((DataType::Float, &1f32.to_le_bytes()))).unwrap_err();
    assert!(
        matches!(&err, Error::Unsupported(u) if u.feature == Feature::StaticInitializer),
        "{err:?}"
    );
}

/// A static initializer stores the constant `value` (of the type `vty` makes) at byte `at` of
/// the global `t` (whose type and initializer `table` makes), then loads `<4 x i16>` at byte 0
/// of `t` into the global the kernel stores. The specialized entry, or the error.
fn store_over(
    table: impl FnOnce(&mut ModuleBuilder) -> (TypeId, crate::ir::ConstId),
    value: impl FnOnce(&mut ModuleBuilder) -> crate::ir::ConstId,
    at: u64,
) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i8t = m.int(8);
    let i16t = m.int(16);
    let i64t = m.int(64);
    let v4 = m.vector(i16t, 4);
    let dev = m.ptr(AddrSpace::Device);
    let cptr = m.ptr(AddrSpace::Constant);
    let (ty, init) = table(&mut m);
    let t = m.simple_global("t", ty, AddrSpace::Constant, Some(init));
    let undef = m.constant(v4, Const::Undef);
    let g = m.simple_global("g", v4, AddrSpace::Constant, Some(undef));
    let (taddr, gaddr) = (m.global_addr(t), m.global_addr(g));
    let value = Value::Const(value(&mut m));
    let off = m.const_int(i64t, at);
    static_init(&mut m, "init", |f| {
        let p = f.push(
            cptr,
            InstKind::Gep {
                src_ty: i8t,
                base: taddr,
                indices: Box::new([off]),
                inbounds: true,
            },
        );
        store(f, value, p);
        let v = load(f, v4, taddr);
        store(f, v, gaddr);
    });
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    let v = load(&mut f, v4, gaddr);
    store(&mut f, v, arg(0));
    f.ret(None);
    let kfn = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(kfn, vec![buffer(0, 0)])], Vec::new());
    specialize(module, &abi, &request(&[]), &mut meter())
}

#[test]
fn a_static_initializer_stores_bytes_over_leaves_of_other_types() {
    // A store's bytes replace those of the leaves it overlaps, and a leaf it covers in part keeps
    // its other bytes: an `i32` at byte 2 of `<4 x i16> <1, 2, 3, 4>` writes lanes 1 and 2.
    let lanes = |m: &mut ModuleBuilder| {
        let i16t = m.int(16);
        let v4 = m.vector(i16t, 4);
        let l: Vec<_> = (1..=4).map(|x| m.constant(i16t, Const::Int(x))).collect();
        (v4, m.constant(v4, Const::Aggregate(l.into_boxed_slice())))
    };
    let word = |m: &mut ModuleBuilder| {
        let i32t = m.int(32);
        m.constant(i32t, Const::Int(0xaaaa_bbbb))
    };
    let e = store_over(lanes, word, 2).unwrap();
    assert_eq!(stored_lanes(&e), [1, 0xbbbb, 0xaaaa, 4]);
    // Over two `i32` fields of a struct, the second covered in part.
    let pair = |m: &mut ModuleBuilder| {
        let i32t = m.int(32);
        let ty = m.struct_ty(&[i32t, i32t]);
        let (a, b) = (
            m.constant(i32t, Const::Int(0x1111_2222)),
            m.constant(i32t, Const::Int(0x3333_4444)),
        );
        (ty, m.constant(ty, Const::Aggregate(Box::new([a, b]))))
    };
    let e = store_over(pair, word, 2).unwrap();
    assert_eq!(stored_lanes(&e), [0x2222, 0xbbbb, 0xaaaa, 0x3333]);
    // A float stored over an integer leaf leaves its encoding.
    let one = |m: &mut ModuleBuilder| {
        let f32t = m.float();
        m.constant(f32t, Const::Float(u64::from(1f32.to_bits())))
    };
    let e = store_over(pair, one, 0).unwrap();
    assert_eq!(stored_lanes(&e), [0, 0x3f80, 0x4444, 0x3333]);

    // A stored byte no leaf holds (padding), a byte beside an `undef` leaf's, and a leaf of a
    // pointer stay refused.
    let refused = |r: Result<Entry, Error>| {
        let err = r.unwrap_err();
        assert!(
            matches!(&err, Error::Unsupported(u) if u.feature == Feature::StaticInitializer),
            "{err:?}"
        );
    };
    let padded = |m: &mut ModuleBuilder| {
        let i16t = m.int(16);
        let i32t = m.int(32);
        let i64t = m.int(64);
        let ty = m.struct_ty(&[i16t, i32t, i64t]);
        let (a, b, c) = (
            m.constant(i16t, Const::Int(1)),
            m.constant(i32t, Const::Int(2)),
            m.constant(i64t, Const::Int(3)),
        );
        (ty, m.constant(ty, Const::Aggregate(Box::new([a, b, c]))))
    };
    refused(store_over(padded, word, 0));
    let undefined = |m: &mut ModuleBuilder| {
        let i16t = m.int(16);
        let v4 = m.vector(i16t, 4);
        (v4, m.constant(v4, Const::Undef))
    };
    refused(store_over(undefined, word, 2));
    let pointers = |m: &mut ModuleBuilder| {
        let p = m.ptr(AddrSpace::Constant);
        (p, m.constant(p, Const::Null))
    };
    refused(store_over(pointers, word, 0));
}

/// A constant global of `ty` holding `init`, in `space`.
fn constant_global(
    m: &mut ModuleBuilder,
    name: &str,
    ty: TypeId,
    space: AddrSpace,
    init: crate::ir::ConstId,
) -> GlobalId {
    let name = m.sym(name);
    m.global(Global {
        name,
        value_ty: ty,
        space,
        constant: true,
        init: Some(init),
        externally_initialized: false,
        visibility: Default::default(),
        unnamed_addr: Default::default(),
        linkage: Linkage::Internal,
        align: None,
        section: None,
    })
}

/// A kernel that picks one of two thread-space constant structs `{ ptr addrspace(2), i64 }`
/// (each holding the address of a constant `i64`, 111 or 222, then a 333 or 444 of its own) by
/// the `bool` function constant 0, then loads a field of the picked struct and stores what it
/// reads. `how`: 0 loads the pointer and stores the `i64` it addresses; 1 stores the `i64`
/// field; 2 copies the pointer through an `i64` (`load i64` of field 0, then `inttoptr`) and
/// stores the `i64` it addresses; 3 does the same through a load at the struct's own address.
fn picked_field(flag: Option<u8>, how: u8) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i8t = m.int(8);
    let i32t = m.int(32);
    let i64t = m.int(64);
    let dev = m.ptr(AddrSpace::Device);
    let cptr = m.ptr(AddrSpace::Constant);
    let tptr = m.ptr(AddrSpace::Thread);
    let pair = m.struct_ty(&[cptr, i64t]);
    let mut picks = Vec::new();
    for (k, (state, own)) in [(111, 333), (222, 444)].into_iter().enumerate() {
        let v = m.constant(i64t, Const::Int(state));
        let g = constant_global(&mut m, &format!("state{k}"), i64t, AddrSpace::Constant, v);
        let addr = m.constant(cptr, Const::GlobalAddr(g));
        let own = m.constant(i64t, Const::Int(own));
        let init = m.constant(pair, Const::Aggregate(Box::new([addr, own])));
        let s = constant_global(
            &mut m,
            &format!("sampler{k}"),
            pair,
            AddrSpace::Thread,
            init,
        );
        picks.push(m.constant(tptr, Const::GlobalAddr(s)));
    }
    let flag_g = fc_global(&mut m, "flag", i8t);
    let flag_addr = m.global_addr(flag_g);
    let zero = m.const_int(i32t, 0);
    let idx = m.const_int(i32t, u64::from(how == 1));
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    let b = load(&mut f, i8t, flag_addr);
    let c = f.push(
        i1,
        InstKind::Cast {
            op: CastOp::Trunc,
            v: b,
        },
    );
    let p = f.push(
        tptr,
        InstKind::Select {
            c,
            t: Value::Const(picks[0]),
            f: Value::Const(picks[1]),
            fmf: FastMath::NONE,
        },
    );
    let at = f.push(
        tptr,
        InstKind::Gep {
            src_ty: pair,
            base: p,
            indices: Box::new([zero, idx]),
            inbounds: true,
        },
    );
    let v = match how {
        0 => {
            let s = load(&mut f, cptr, at);
            load(&mut f, i64t, s)
        }
        1 => load(&mut f, i64t, at),
        _ => {
            let bits = load(&mut f, i64t, if how == 3 { p } else { at });
            let s = f.push(
                cptr,
                InstKind::Cast {
                    op: CastOp::IntToPtr,
                    v: bits,
                },
            );
            load(&mut f, i64t, s)
        }
    };
    store(&mut f, v, arg(0));
    f.ret(None);
    let kfn = f.finish();
    let module = m.finish().unwrap();
    let mut decl = fc_decl(0, flag_g, i8t);
    decl.type_name = DiagName::new("bool");
    let abi = abi(vec![kernel(kfn, vec![buffer(0, 0)])], vec![decl]);
    let constants: Vec<(u32, ConstValue)> = flag.map(|b| (0, boolean(b))).into_iter().collect();
    specialize(module, &abi, &request(&constants), &mut meter())
}

#[test]
fn a_pointer_loaded_from_a_constant_globals_field_is_the_fields_constant() {
    // With the constant supplied, SCCP folds the select to one struct, and the pointer loaded from
    // its field is the field's constant: the kernel reads the picked state global directly, and the
    // other struct and its state are pruned. A copy of the pointer through an `i64` (`inttoptr` of
    // its integer load) reads the same, at the field's address or at the struct's own, whose first
    // member it is.
    for how in [0, 2, 3] {
        for (flag, kept, gone) in [(1, "111", "222"), (0, "222", "111")] {
            let e = picked_field(Some(flag), how).unwrap();
            let text = body_text(&e);
            assert!(!text.contains("getelementptr"), "{text}");
            assert!(
                !text.contains("load ptr") && !text.contains("inttoptr"),
                "{text}"
            );
            assert!(text.contains(kept) && !text.contains(gone), "{text}");
        }
    }
    // A field of any other type is left to the memory plan: its load stays.
    let text = body_text(&picked_field(Some(1), 1).unwrap());
    assert!(text.contains("getelementptr"), "{text}");
    // Unsupplied, the select stays undecided, as before.
    assert_eq!(
        picked_field(None, 0).unwrap_err(),
        Error::NeedsInput(MissingInput::FunctionConstant { index: 0 })
    );
}

#[test]
fn an_unevaluable_static_initializer_is_refused_typed() {
    // A loop.
    let err = stores_global(|m, _| {
        let void = m.void();
        let fty = m.fn_ty(void, &[]);
        let mut init = m.define("init", fty);
        let l = init.block();
        init.br(l);
        init.switch_to(l);
        init.br(l);
        let init = init.finish();
        m.func_mut(init).unwrap().section = Some(Section::StaticInit);
    })
    .unwrap_err();
    let Error::Unsupported(u) = err else {
        panic!("{err:?}");
    };
    assert_eq!(u.feature, Feature::StaticInitializer);

    // A store outside the constant address space.
    let err = stores_global(|m, _| {
        let i32t = m.int(32);
        let zero = m.constant(i32t, Const::Int(0));
        let d = m.simple_global("d", i32t, AddrSpace::Device, Some(zero));
        let daddr = m.global_addr(d);
        let one = m.const_int(i32t, 1);
        static_init(m, "init", |f| store(f, one, daddr));
    })
    .unwrap_err();
    let Error::Unsupported(u) = err else {
        panic!("{err:?}");
    };
    assert_eq!(u.feature, Feature::StaticInitializer);

    // Recursion.
    let err = stores_global(|m, _| {
        let void = m.void();
        let fty = m.fn_ty(void, &[]);
        let mut r = m.define("r", fty);
        call(&mut r, void, Callee::Direct(FuncId::new(0)), &[]);
        r.ret(None);
        let r = r.finish();
        static_init(m, "init", |f| {
            call(f, void, Callee::Direct(r), &[]);
        });
    })
    .unwrap_err();
    let Error::Unsupported(u) = err else {
        panic!("{err:?}");
    };
    assert_eq!(u.feature, Feature::Recursion);
}

#[test]
fn a_static_initializer_branch_on_an_unsupplied_constant_needs_input() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i8t = m.int(8);
    let flag = fc_global(&mut m, "flag", i8t);
    let zero = m.constant(i8t, Const::Int(0));
    let g = m.simple_global("g", i8t, AddrSpace::Constant, Some(zero));
    let (faddr, gaddr) = (m.global_addr(flag), m.global_addr(g));
    let fty = m.fn_ty(void, &[]);
    let mut init = m.define("init", fty);
    let (t, e) = (init.block(), init.block());
    let v = load(&mut init, i8t, faddr);
    let c = init.push(
        i1,
        InstKind::Cast {
            op: CastOp::Trunc,
            v,
        },
    );
    init.cond_br(c, t, e);
    init.switch_to(t);
    store(&mut init, v, gaddr);
    init.br(e);
    init.switch_to(e);
    init.ret(None);
    let init = init.finish();
    m.func_mut(init).unwrap().section = Some(Section::StaticInit);
    let mut f = m.define("k", fty);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, Vec::new())], vec![fc_decl(3, flag, i8t)]);
    assert_eq!(
        specialize(module.clone(), &abi, &request(&[]), &mut meter()).unwrap_err(),
        Error::NeedsInput(MissingInput::FunctionConstant { index: 3 })
    );
    let r = SpecRequest {
        call_stack_depth: 1,
        entry: "k".into(),
        constants: [(3, boolean(1))].into_iter().collect(),
    };
    specialize(module, &abi, &r, &mut meter()).unwrap();
}

/// A PHI in a live block keeps no entry, and no value, from a block SCCP deleted, even when the
/// value is defined only in that block.
#[test]
fn a_phi_drops_the_value_of_a_deleted_predecessor() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let fty = m.fn_ty(void, &[dev]);
    let f0 = m.const_int(i1, 0);
    let one = m.const_int(i32t, 1);
    let mut f = m.define("k", fty);
    let (dead, live, join) = (f.block(), f.block(), f.block());
    f.cond_br(f0, dead, live);
    f.switch_to(dead);
    let x = load(&mut f, i32t, arg(0));
    f.br(join);
    f.switch_to(live);
    let y = load(&mut f, i32t, arg(0));
    let z = f.bin(BinOp::Add, y, one);
    f.br(join);
    f.switch_to(join);
    let p = f.push(
        i32t,
        InstKind::Phi {
            incoming: vec![(x, dead), (z, live)].into(),
            fmf: FastMath::NONE,
        },
    );
    store(&mut f, p, arg(0));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, vec![buffer(0, 0)])], Vec::new());
    let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
    assert!(
        body_text(&e).contains("phi i32 [%i2, bb1]"),
        "{}",
        body_text(&e)
    );
}

#[test]
fn an_unknown_intrinsic_fails_only_when_it_survives() {
    let build = |gate: bool| {
        let mut m = ModuleBuilder::new();
        let void = m.void();
        let i1 = m.int(1);
        let fty = m.fn_ty(void, &[]);
        let decl = m.declare("air.some_future_op", fty);
        let c = m.const_int(i1, u64::from(gate));
        let mut f = m.define("k", fty);
        let (t, e) = (f.block(), f.block());
        f.cond_br(c, t, e);
        f.switch_to(t);
        call(&mut f, void, Callee::UnknownIntrinsic(decl), &[]);
        f.ret(None);
        f.switch_to(e);
        f.ret(None);
        let k = f.finish();
        let module = m.finish().unwrap();
        let abi = abi(vec![kernel(k, Vec::new())], Vec::new());
        specialize(module, &abi, &request(&[]), &mut meter())
    };
    let e = build(false).unwrap();
    assert_eq!(e.module.parts().funcs.len(), 1, "the declaration is pruned");
    let Error::Unsupported(u) = build(true).unwrap_err() else {
        panic!();
    };
    assert_eq!(u.feature, Feature::UnknownIntrinsic);
}

#[test]
fn a_vector_constant_and_a_bool_bake_from_packed_bytes() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i8t = m.int(8);
    let f32t = m.float();
    let v3 = m.vector(f32t, 3);
    let dev = m.ptr(AddrSpace::Device);
    let flag = fc_global(&mut m, "flag", i8t);
    let color = fc_global(&mut m, "color", v3);
    let fty = m.fn_ty(void, &[dev, dev]);
    let (faddr, caddr) = (m.global_addr(flag), m.global_addr(color));
    let mut f = m.define("k", fty);
    let b = load(&mut f, i8t, faddr);
    store(&mut f, b, arg(0));
    let c = load(&mut f, v3, caddr);
    store(&mut f, c, arg(1));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(
        vec![kernel(k, vec![buffer(0, 0), buffer(1, 1)])],
        vec![fc_decl(0, flag, i8t), fc_decl(1, color, v3)],
    );
    let bytes: Vec<u8> = [1f32, 2.0, 3.0]
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let r = request(&[
        (
            0,
            ConstValue {
                ty: DataType::Bool,
                bytes: Box::new([7]),
            },
        ),
        (
            1,
            ConstValue {
                ty: DataType::Float3,
                bytes: bytes.into(),
            },
        ),
    ]);
    let e = specialize(module, &abi, &r, &mut meter()).unwrap();
    let text = body_text(&e);
    assert!(text.contains("store i8 1, %a0"), "{text}");
    assert!(text.contains("store <3 x float> c4, %a1"), "{text}");
    assert!(
        text.contains("[0x3F800000, 0x40000000, 0x40400000]"),
        "{text}"
    );
}

/// `kernel k(device i32* out)` branching with `branch` on a constant expression over `@g`'s
/// address, storing 1 on one successor and 2 on the other.
fn constexpr_branch(
    branch: impl FnOnce(&mut FunctionBuilder<'_>, Value, BlockId, BlockId),
    width: u8,
) -> (Module, AirAbi) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let i64t = m.int(64);
    let cty = m.int(width);
    let dev = m.ptr(AddrSpace::Device);
    let zero = m.constant(i32t, Const::Int(0));
    let g = m.simple_global("g", i32t, AddrSpace::Constant, Some(zero));
    let Value::Const(addr) = m.global_addr(g) else {
        panic!();
    };
    let as_int = m.constant(
        i64t,
        Const::Expr(ConstExpr::Cast {
            op: CastOp::PtrToInt,
            ops: [addr],
        }),
    );
    let cond = if width == 64 {
        as_int
    } else {
        m.constant(
            cty,
            Const::Expr(ConstExpr::Cast {
                op: CastOp::Trunc,
                ops: [as_int],
            }),
        )
    };
    let (one, two) = (m.const_int(i32t, 1), m.const_int(i32t, 2));
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    let (t, e) = (f.block(), f.block());
    branch(&mut f, Value::Const(cond), t, e);
    f.switch_to(t);
    store(&mut f, one, arg(0));
    f.ret(None);
    f.switch_to(e);
    store(&mut f, two, arg(0));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, vec![buffer(0, 0)])], Vec::new());
    (module, abi)
}

#[test]
fn a_branch_on_a_constant_expression_keeps_every_successor() {
    // `br i1 trunc (i64 ptrtoint (ptr @g to i64) to i1)` is valid input that SCCP cannot
    // decide: both successors stay (only a literal `undef`/`poison` is refused).
    let (module, abi) = constexpr_branch(|f, c, t, e| f.cond_br(c, t, e), 1);
    let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
    let text = body_text(&e);
    assert!(text.contains("store i32 1"), "{text}");
    assert!(text.contains("store i32 2"), "{text}");
    // `switch i64 ptrtoint (ptr @g to i64)` likewise keeps its default and every case.
    let (module, abi) = constexpr_branch(
        |f, v, t, e| {
            f.push_void(InstKind::Switch {
                v,
                default: e,
                cases: Box::new([(0, t)]),
                hints: None,
            })
        },
        64,
    );
    let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
    let text = body_text(&e);
    assert!(text.contains("store i32 1"), "{text}");
    assert!(text.contains("store i32 2"), "{text}");
}

#[test]
fn a_predicate_global_without_an_integer_value_is_an_unevaluated_static_initializer() {
    // No `air.static_init`, and the predicate's global holds `undef`: the single error mapping
    // classes `StaticInitNotEvaluable` as `Unsupported{StaticInitializer}`, not `InvalidInput`.
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i8t = m.int(8);
    let dev = m.ptr(AddrSpace::Device);
    let undef = m.constant(i8t, Const::Undef);
    let p = m.simple_global("p", i8t, AddrSpace::Constant, Some(undef));
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let param = ParamBinding {
        predicate: Some(Predicate {
            global: p,
            name: DiagName::new("p"),
        }),
        ..buffer(0, 0)
    };
    let abi = abi(vec![kernel(k, vec![param])], Vec::new());
    let err = specialize(module, &abi, &request(&[]), &mut meter()).unwrap_err();
    let Error::Unsupported(u) = err else {
        panic!("{err:?}");
    };
    assert_eq!(u.feature, Feature::StaticInitializer);
}

/// `wide(p)`: `n` stores to `p`, a straight-line body the inliner copies once per call.
fn wide_helper(m: &mut ModuleBuilder, n: usize) -> FuncId {
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let hty = m.fn_ty(void, &[dev]);
    let one = m.const_int(i32t, 1);
    let mut h = m.define("wide", hty);
    for _ in 0..n {
        store(&mut h, one, arg(0));
    }
    h.ret(None);
    h.finish()
}

const WIDE: usize = 256;
const CALLS: usize = 64;

/// The corpus's shape: the static initializer derives `@unroll` from the constant `@mode` (index
/// 3), and the kernel switches on `@unroll` into cases that call `wide` `CALLS` times, so the
/// inlined body has about `WIDE * CALLS` instructions, far more than the module. `call_first` puts
/// one call to `wide` in the entry block before the switch.
fn switch_on_derived(call_first: bool) -> (Module, AirAbi) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i16t = m.int(16);
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let mode = fc_global(&mut m, "mode", i32t);
    let zero = m.constant(i16t, Const::Int(0));
    let unroll = m.simple_global("unroll", i16t, AddrSpace::Constant, Some(zero));
    let (mode_addr, unroll_addr) = (m.global_addr(mode), m.global_addr(unroll));
    static_init(&mut m, "init", |f| {
        let v = load(f, i32t, mode_addr);
        let t = f.push(
            i16t,
            InstKind::Cast {
                op: CastOp::Trunc,
                v,
            },
        );
        store(f, t, unroll_addr);
    });
    let wide = wide_helper(&mut m, WIDE);
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    let (one, two, exit) = (f.block(), f.block(), f.block());
    if call_first {
        call(&mut f, void, Callee::Direct(wide), &[arg(0)]);
    }
    let u = load(&mut f, i16t, unroll_addr);
    f.push_void(InstKind::Switch {
        v: u,
        default: exit,
        cases: Box::new([(1, one), (2, two)]),
        hints: None,
    });
    for b in [one, two] {
        f.switch_to(b);
        for _ in 0..CALLS / 2 {
            call(&mut f, void, Callee::Direct(wide), &[arg(0)]);
        }
        f.br(exit);
    }
    f.switch_to(exit);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(
        vec![kernel(k, vec![buffer(0, 0)])],
        vec![fc_decl(3, mode, i32t)],
    );
    (module, abi)
}

#[test]
fn an_entry_block_refusal_comes_before_the_inlined_body() {
    let run = |call_first| {
        let (module, abi) = switch_on_derived(call_first);
        let mut meter = meter();
        let err = specialize(module, &abi, &request(&[]), &mut meter).unwrap_err();
        (err, meter.work(), meter.ledger_peak())
    };
    let needs = Error::NeedsInput(MissingInput::FunctionConstant { index: 3 });
    let inlined = (WIDE * CALLS) as u64;
    // The entry block's switch on the derived global is refused before anything proportional to
    // the inlined body is built or visited.
    let (early, work, peak) = run(false);
    assert_eq!(early, needs);
    assert!(work < inlined / 8, "{work} work units");
    // Behind a call, SCCP over the inlined body refuses the same switch, with the same error,
    // after the work the early refusal does not do.
    let (late, late_work, late_peak) = run(true);
    assert_eq!(late, needs);
    assert!(late_work > inlined, "{late_work} work units");
    assert!(peak * 8 < late_peak, "ledger peaks {peak} and {late_peak}");
    // Supplied, the switch folds and the taken case is inlined.
    let (module, abi) = switch_on_derived(false);
    let e = specialize(module, &abi, &request(&[(3, int(2))]), &mut meter()).unwrap();
    let stores = body_text(&e).matches("store i32 1, %a0").count();
    assert_eq!(stores, WIDE * CALLS / 2);
}

/// `kernel k(device i32* p)` whose entry block `entry` builds from a load of the unsupplied
/// `@mode` (index 3) and ends by calling `wide`, so the entry is inlined; `helper` defines a
/// function `entry` may call.
fn entry_block(
    helper: impl FnOnce(&mut ModuleBuilder) -> FuncId,
    entry: impl FnOnce(&mut FunctionBuilder<'_>, Value, FuncId),
) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let mode = fc_global(&mut m, "mode", i32t);
    let mode_addr = m.global_addr(mode);
    let wide = wide_helper(&mut m, 2);
    let h = helper(&mut m);
    let fty = m.fn_ty(void, &[dev]);
    let mut f = m.define("k", fty);
    let v = load(&mut f, i32t, mode_addr);
    entry(&mut f, v, h);
    call(&mut f, void, Callee::Direct(wide), &[arg(0)]);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(
        vec![kernel(k, vec![buffer(0, 0)])],
        vec![fc_decl(3, mode, i32t)],
    );
    specialize(module, &abi, &request(&[]), &mut meter())
}

#[test]
fn the_entry_block_refusal_defers_what_it_does_not_decide() {
    let needs = Error::NeedsInput(MissingInput::FunctionConstant { index: 3 });
    // A certain refusal: a store of the undefined value.
    let err = entry_block(|m| wide_helper(m, 1), |f, v, _| store(f, v, arg(0))).unwrap_err();
    assert_eq!(err, needs);

    // The inliner's own refusals come first: a call cycle through the entry (the kernel `k`,
    // defined next).
    let recursive = |m: &mut ModuleBuilder| {
        let void = m.void();
        let dev = m.ptr(AddrSpace::Device);
        let fty = m.fn_ty(void, &[dev]);
        let mut a = m.define("a", fty);
        let k = FuncId::new(a.id().raw() + 1);
        call(&mut a, void, Callee::Direct(k), &[arg(0)]);
        a.ret(None);
        a.finish()
    };
    let err = entry_block(recursive, |f, v, a| {
        store(f, v, arg(0));
        let void = f.module().void();
        call(f, void, Callee::Direct(a), &[arg(0)]);
    })
    .unwrap_err();
    let Error::Unsupported(u) = err else {
        panic!("{err:?}");
    };
    assert_eq!(u.feature, Feature::Recursion);

    // An earlier `BadValue` in the block is SCCP's first refusal.
    let err = entry_block(
        |m| wide_helper(m, 1),
        |f, v, _| {
            let i1 = f.module().int(1);
            let cptr = f.module().ptr(AddrSpace::Constant);
            let is_def_ty = f.module().fn_ty(i1, &[cptr]);
            let is_def = f
                .module()
                .declare("air.is_function_constant_defined", is_def_ty);
            let null = f.module().constant(cptr, Const::Null);
            call(
                f,
                i1,
                Callee::Air {
                    decl: is_def,
                    op: AirIntrinsic::FunctionConstant(FunctionConstant::IsDefined),
                },
                &[Value::Const(null)],
            );
            store(f, v, arg(0));
        },
    )
    .unwrap_err();
    let Error::InvalidInput(e) = err else {
        panic!("{err:?}");
    };
    assert_eq!(
        e.kind,
        InputErrorKind::Abi(crate::error::AbiError::BadValue)
    );

    // An inlined call's argument is not a use: the callee drops its pure use.
    let pure_use = |m: &mut ModuleBuilder| {
        let void = m.void();
        let i32t = m.int(32);
        let hty = m.fn_ty(void, &[i32t]);
        let one = m.const_int(i32t, 1);
        let mut h = m.define("pure_use", hty);
        let x = h.arg(0);
        h.bin(BinOp::Add, x, one);
        h.ret(None);
        h.finish()
    };
    entry_block(pure_use, |f, v, h| {
        let void = f.module().void();
        call(f, void, Callee::Direct(h), &[v]);
    })
    .unwrap();

    // A select between a tainted and a clean value is undecided without folding: SCCP folds the
    // condition and stores the clean arm.
    let e = entry_block(
        |m| wide_helper(m, 1),
        |f, v, _| {
            let i1 = f.module().int(1);
            let i32t = f.module().int(32);
            let yes = f.module().const_int(i1, 1);
            let seven = f.module().const_int(i32t, 7);
            let s = f.push(
                i32t,
                InstKind::Select {
                    c: yes,
                    t: seven,
                    f: v,
                    fmf: FastMath::NONE,
                },
            );
            store(f, s, arg(0));
        },
    )
    .unwrap();
    assert!(
        body_text(&e).contains("store i32 7, %a0"),
        "{}",
        body_text(&e)
    );
}

/// A kernel `k` whose one block calls `callee` (built from the module and its declarations) and
/// returns, specialized.
fn specialize_call(
    callee: impl FnOnce(&mut ModuleBuilder, TypeId) -> Callee,
) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let fty = m.fn_ty(void, &[]);
    let callee = callee(&mut m, fty);
    let mut f = m.define("k", fty);
    call(&mut f, void, callee, &[]);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, Vec::new())], Vec::new());
    specialize(module, &abi, &request(&[]), &mut meter())
}

fn refusal(r: Result<Entry, Error>) -> Unsupported {
    match r {
        Err(Error::Unsupported(u)) => u,
        Err(other) => panic!("{other}"),
        Ok(_) => panic!("specialized"),
    }
}

/// A surviving call is refused at its location in the inlined entry, naming its callee, as its
/// typed feature (not implemented yet): binding an intersection function table (an undecoded
/// `air.set_intersection_function_table.*` intrinsic, by its family), a multisample imageblock's
/// operation (imageblocks), a call to link, and an indirect call; another unknown intrinsic is
/// `UnknownIntrinsic`.
#[test]
fn surviving_calls_are_refused_by_family_naming_the_callee() {
    let at = Location::function(DiagName::new("k"), Some(0), Some(0));
    let u = refusal(specialize_call(|m, fty| {
        Callee::UnknownIntrinsic(m.declare("air.set_intersection_function_table.p1i8", fty))
    }));
    assert_eq!(u.feature, Feature::IntersectionFunctionTable);
    assert_eq!(u.location, Some(at.clone()));
    assert_eq!(
        u.symbol,
        Some(DiagName::new("air.set_intersection_function_table.p1i8"))
    );

    // A multisample imageblock's operation is imageblocks, naming the operation.
    let u = refusal(specialize_call(|m, fty| {
        Callee::UnknownIntrinsic(m.declare("air.get_imageblock_num_colors", fty))
    }));
    assert_eq!(u.feature, Feature::Imageblocks);
    assert_eq!(u.location, Some(at.clone()));
    assert_eq!(
        u.symbol,
        Some(DiagName::new("air.get_imageblock_num_colors"))
    );

    let u = refusal(specialize_call(|m, fty| {
        Callee::UnknownIntrinsic(m.declare("air.some_future_op", fty))
    }));
    assert_eq!(u.feature, Feature::UnknownIntrinsic);
    assert_eq!(u.location, Some(at.clone()));
    assert_eq!(u.symbol, Some(DiagName::new("air.some_future_op")));

    // An undefined external function (the shape of the tensor-ops library calls) needs linking.
    let u = refusal(specialize_call(|m, fty| {
        Callee::Direct(m.declare("__tensorops_impl_matmul2d_op_cooperative_tensor_init", fty))
    }));
    assert_eq!(u.feature, Feature::Linking);
    assert_eq!(u.location, Some(at.clone()));
    assert_eq!(
        u.symbol,
        Some(DiagName::new(
            "__tensorops_impl_matmul2d_op_cooperative_tensor_init"
        ))
    );

    let u = refusal(specialize_call(|m, fty| {
        let ptr = m.ptr(AddrSpace::Device);
        let callee = m.constant(ptr, Const::Null);
        Callee::Indirect {
            callee: Value::Const(callee),
            fn_ty: fty,
        }
    }));
    assert_eq!(u.feature, Feature::DynamicCall);
    assert_eq!(u.location, Some(at));
    assert_eq!(u.symbol, None);
}

/// The decoder keeps a call through a function's own constant address indirect when the call's
/// function type is not the function's. Such a call is classified by that function, not as a
/// dynamic call: a defined function whose parameters cannot receive the call's arguments is invalid
/// input, since nothing binds them (`CanonError::CallTypeMismatch`; here the call passes no
/// argument to a one-parameter function), an undefined one needs linking (not implemented, naming
/// the function), and an intrinsic called at another type is a signature mismatch. Each is refused
/// at the call.
#[test]
fn a_call_through_a_function_address_at_another_type_is_classified_by_the_function() {
    let at = Location::function(DiagName::new("k"), Some(0), Some(0));
    // A call as `void ()` through the address of `name`, a `void (i32)` function.
    let through = |name: &'static str, defined: bool| {
        specialize_call(move |m, fty| {
            let void = m.void();
            let i32t = m.int(32);
            let hty = m.fn_ty(void, &[i32t]);
            let h = if defined {
                let mut h = m.define(name, hty);
                h.ret(None);
                h.finish()
            } else {
                m.declare(name, hty)
            };
            let thread = m.ptr(AddrSpace::Thread);
            let addr = m.constant(thread, Const::FuncAddr(h));
            Callee::Indirect {
                callee: Value::Const(addr),
                fn_ty: fty,
            }
        })
    };
    match through("helper", true) {
        Err(Error::InvalidInput(e)) => {
            assert_eq!(e.kind, InputErrorKind::Canon(CanonError::CallTypeMismatch));
            assert_eq!(e.at, at);
        }
        Err(other) => panic!("{other}"),
        Ok(_) => panic!("specialized"),
    }

    let u = refusal(through("external", false));
    assert_eq!(u.feature, Feature::Linking);
    assert_eq!(u.location, Some(at.clone()));
    assert_eq!(u.symbol, Some(DiagName::new("external")));

    match through("air.some_op", false) {
        Err(Error::InvalidInput(e)) => {
            assert_eq!(
                e.kind,
                InputErrorKind::Intrinsic(IntrinsicError::SignatureMismatch)
            );
            assert_eq!(e.at, at);
        }
        Err(other) => panic!("{other}"),
        Ok(_) => panic!("specialized"),
    }
}

/// A call through a defined function's address at another type is invalid input only when its
/// arguments differ from the function's parameters in count or in shape. A call that differs only
/// in its return type, or passes an identically shaped struct of another name, gives each parameter
/// its argument: the inliner cannot take it (`DirectCall`, naming the function), as linking does
/// not resolve a reference of another isomorphic type.
#[test]
fn a_call_at_another_type_is_invalid_only_when_its_arguments_differ() {
    let at = Location::function(DiagName::new("k"), Some(0), Some(0));
    /// `k` calls `helper` through its address at another type, in one of four shapes.
    #[derive(Clone, Copy, Debug)]
    enum Shape {
        /// `i32 (i32)` called as `void (i32)`.
        ReturnOnly,
        /// `void (%a)` called as `void (%b)`, both `{ i32, i32 }`.
        RenamedStruct,
        /// `void (i32)` called as `void (float)`.
        ArgumentType,
        /// `void (i32)` called as `void (i32, i32)`.
        ArgumentCount,
    }
    let through = |shape: Shape| {
        let mut m = ModuleBuilder::new();
        let void = m.void();
        let i32t = m.int(32);
        let float = m.float();
        let kty = m.fn_ty(void, &[]);
        let (def_ty, call_args) = match shape {
            Shape::ReturnOnly => (m.fn_ty(i32t, &[i32t]), vec![i32t]),
            Shape::RenamedStruct => {
                let a = pair_struct(&mut m, "a");
                let b = pair_struct(&mut m, "b");
                (m.fn_ty(void, &[a]), vec![b])
            }
            Shape::ArgumentType => (m.fn_ty(void, &[i32t]), vec![float]),
            Shape::ArgumentCount => (m.fn_ty(void, &[i32t]), vec![i32t, i32t]),
        };
        let call_ty = m.fn_ty(void, &call_args);
        let ret = match shape {
            Shape::ReturnOnly => Some(m.const_int(i32t, 7)),
            _ => None,
        };
        let mut h = m.define("helper", def_ty);
        h.ret(ret);
        let h = h.finish();
        let thread = m.ptr(AddrSpace::Thread);
        let addr = m.constant(thread, Const::FuncAddr(h));
        let args: Vec<Value> = call_args
            .iter()
            .map(|&t| Value::Const(m.constant(t, Const::Null)))
            .collect();
        let mut f = m.define("k", kty);
        let callee = Callee::Indirect {
            callee: Value::Const(addr),
            fn_ty: call_ty,
        };
        call(&mut f, void, callee, &args);
        f.ret(None);
        let k = f.finish();
        let module = m.finish().unwrap();
        let abi = abi(vec![kernel(k, Vec::new())], Vec::new());
        specialize(module, &abi, &request(&[]), &mut meter())
    };
    for shape in [Shape::ReturnOnly, Shape::RenamedStruct] {
        let u = refusal(through(shape));
        assert_eq!(u.feature, Feature::DirectCall, "{shape:?}");
        assert_eq!(u.location, Some(at.clone()), "{shape:?}");
        assert_eq!(u.symbol, Some(DiagName::new("helper")), "{shape:?}");
    }
    for shape in [Shape::ArgumentType, Shape::ArgumentCount] {
        match through(shape) {
            Err(Error::InvalidInput(e)) => {
                assert_eq!(
                    e.kind,
                    InputErrorKind::Canon(CanonError::CallTypeMismatch),
                    "{shape:?}"
                );
                assert_eq!(e.at, at, "{shape:?}");
            }
            Err(other) => panic!("{shape:?}: {other}"),
            Ok(_) => panic!("{shape:?}: specialized"),
        }
    }
}

/// An identified struct `name` of two `i32` fields.
fn pair_struct(m: &mut ModuleBuilder, name: &str) -> TypeId {
    let i32t = m.int(32);
    let name = m.sym(name);
    m.ty(Type::Struct {
        fields: vec![i32t, i32t].into(),
        packed: false,
        name: Some(name),
    })
}

/// A linkage signature describes how a closed module was produced, so a stitched module and a
/// linked-constants module that declares no constant specialize like any other. A linked-constants
/// module that still declares a constant has no reading: `Linking`, located at the signature.
#[test]
fn a_linkage_signature_describes_a_closed_module() {
    use crate::abi::{LinkageSignature, LinkageSignatureKind};
    let build = |kind: LinkageSignatureKind, declares: bool| {
        let mut m = ModuleBuilder::new();
        let void = m.void();
        let i32t = m.int(32);
        let c = declares.then(|| fc_global(&mut m, "c", i32t));
        let fty = m.fn_ty(void, &[]);
        let mut f = m.define("k", fty);
        f.ret(None);
        let k = f.finish();
        let module = m.finish().unwrap();
        let fcs = c.map(|c| vec![fc_decl(0, c, i32t)]).unwrap_or_default();
        let mut abi = abi(vec![kernel(k, Vec::new())], fcs);
        abi.linkage_signature = Some(LinkageSignature { kind, operands: 2 });
        (module, abi)
    };
    for kind in [
        LinkageSignatureKind::StitchingModules,
        LinkageSignatureKind::LinkedConstants,
    ] {
        let (module, abi) = build(kind, false);
        let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
        assert_eq!(e.ep.name, DiagName::new("k"), "{kind:?}");
    }
    let (module, abi) = build(LinkageSignatureKind::LinkedConstants, true);
    let u = refusal(specialize(module, &abi, &request(&[]), &mut meter()));
    assert_eq!(u.feature, Feature::Linking);
    assert_eq!(
        u.location,
        Some(Location::metadata(None, None, Some("air.signature")))
    );
}

/// A function whose address the entry takes is a function pointer (a dynamic call), named.
#[test]
fn a_taken_function_address_is_a_dynamic_call() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let fty = m.fn_ty(void, &[]);
    let mut h = m.define("helper", fty);
    h.ret(None);
    let helper = h.finish();
    let dev = m.ptr(AddrSpace::Device);
    let thread = m.ptr(AddrSpace::Thread);
    let kty = m.fn_ty(void, &[dev]);
    let addr = m.constant(thread, Const::FuncAddr(helper));
    let mut f = m.define("k", kty);
    store(&mut f, Value::Const(addr), arg(0));
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let abi = abi(vec![kernel(k, vec![buffer(0, 0)])], Vec::new());
    let u = refusal(specialize(module, &abi, &request(&[]), &mut meter()));
    assert_eq!(u.feature, Feature::DynamicCall);
    assert_eq!(u.symbol, Some(DiagName::new("helper")));
}

/// A post-tessellation vertex function `k(ptr cps, i32 i)` of a triangle patch whose
/// control-point input (parameter 0) names the declaration `k.MTL_CONTROL_POINT_FN`, returning
/// `{ i32 }`; `body` fills the entry with the function, the struct type, and the module builder.
fn post_tessellation(
    body: impl FnOnce(&mut FunctionBuilder<'_>, FuncId, TypeId),
) -> (crate::ir::Module, AirAbi) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let ptr = m.ptr(AddrSpace::Thread);
    let point = m.struct_ty(&[i32t]);
    let fetch_ty = m.fn_ty(point, &[i32t, ptr]);
    let control_point = m.declare("k.MTL_CONTROL_POINT_FN", fetch_ty);
    let fty = m.fn_ty(void, &[ptr, i32t]);
    let mut f = m.define("k", fty);
    body(&mut f, control_point, point);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let mut ep = kernel(k, Vec::new());
    ep.stage = Stage::Vertex {
        patch: Some(crate::abi::PatchInfo {
            kind: crate::abi::PatchKind::Triangle,
            control_points: Some(3),
        }),
    };
    let mut control_points = buffer(0, 0);
    control_points.role = ParamRole::PatchInput(crate::abi::PatchInput::ControlPoints {
        function: control_point,
        fields: Vec::new(),
    });
    ep.params = vec![control_points, buffer(1, 1)];
    (module, abi(vec![ep], Vec::new()))
}

/// A post-tessellation vertex function's call to its control-point function, with its control-point
/// input, is the one call to a declaration specialization keeps; the input names the declaration in
/// the specialized module.
#[test]
fn a_control_point_fetch_survives_specialization() {
    let (module, abi) = post_tessellation(|f, cp, point| {
        call(f, point, Callee::Direct(cp), &[arg(1), arg(0)]);
    });
    let e = specialize(module, &abi, &request(&[]), &mut meter()).expect("specialized");
    let ParamRole::PatchInput(crate::abi::PatchInput::ControlPoints { function, .. }) =
        &e.ep.params[0].role
    else {
        panic!("{:?}", e.ep.params);
    };
    let body = e.module.func(e.func).and_then(|f| f.body.as_ref()).unwrap();
    let called = body.insts.values().any(
        |inst| matches!(&inst.kind, InstKind::Call(c) if c.callee == Callee::Direct(*function)),
    );
    assert!(called, "{}", body_text(&e));
    assert!(e.module.func(*function).is_some_and(|f| f.body.is_none()));
}

/// The control-point function called with a pointer other than the input is not a fetch Metal
/// defines: tessellation, naming the function.
#[test]
fn a_control_point_fetch_of_another_pointer_is_tessellation() {
    let (module, abi) = post_tessellation(|f, cp, point| {
        let ptr = f.module().ptr(AddrSpace::Thread);
        let i32t = f.module().int(32);
        let one = f.module().const_int(i32t, 1);
        let local = f.push(
            ptr,
            InstKind::Alloca {
                ty: i32t,
                count: one,
                align: None,
            },
        );
        call(f, point, Callee::Direct(cp), &[arg(1), local]);
    });
    let u = refusal(specialize(module, &abi, &request(&[]), &mut meter()));
    assert_eq!(u.feature, Feature::Tessellation);
    assert_eq!(u.symbol, Some(DiagName::new("k.MTL_CONTROL_POINT_FN")));
}

/// A control-point input the body reads directly is tessellation at the entry's argument also when
/// the body calls its function, which then survives pruning.
#[test]
fn a_control_point_input_read_beside_its_fetch_is_tessellation_at_its_argument() {
    let (module, abi) = post_tessellation(|f, cp, point| {
        call(f, point, Callee::Direct(cp), &[arg(1), arg(0)]);
        let i32t = f.module().int(32);
        let x = load(f, i32t, arg(0));
        store(f, x, arg(0));
    });
    let u = refusal(specialize(module, &abi, &request(&[]), &mut meter()));
    assert_eq!(u.feature, Feature::Tessellation);
    assert_eq!(
        u.location,
        Some(Location::metadata(Some(DiagName::new("k")), Some(0), None))
    );
}

/// A control-point input the body reads other than through its function (which it never calls) is
/// tessellation, at the entry's argument.
#[test]
fn a_control_point_input_read_directly_is_tessellation_at_its_argument() {
    let (module, abi) = post_tessellation(|f, _, _| {
        let i32t = f.module().int(32);
        let x = load(f, i32t, arg(0));
        store(f, x, arg(0));
    });
    let u = refusal(specialize(module, &abi, &request(&[]), &mut meter()));
    assert_eq!(u.feature, Feature::Tessellation);
    assert_eq!(
        u.location,
        Some(Location::metadata(Some(DiagName::new("k")), Some(0), None))
    );
}

/// A predicate global holding `v`, as a static initializer leaves one.
fn predicate(m: &mut ModuleBuilder, name: &str, v: u64) -> Predicate {
    let i8t = m.int(8);
    let c = m.constant(i8t, Const::Int(v));
    Predicate {
        global: m.simple_global(name, i8t, AddrSpace::Constant, Some(c)),
        name: DiagName::new(name),
    }
}

/// `kernel k(device i32* gone [[off]], uint tid [[off]], device i32* out)`: `tid + 1` is computed
/// and, when `observe`, stored; `*gone` is stored.
fn absent_reads(observe: bool) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let off = predicate(&mut m, "off", 0);
    let fty = m.fn_ty(void, &[dev, i32t, dev]);
    let one = m.const_int(i32t, 1);
    let mut f = m.define("k", fty);
    let x = load(&mut f, i32t, arg(0));
    store(&mut f, x, arg(2));
    let y = f.bin(BinOp::Add, arg(1), one);
    if observe {
        store(&mut f, y, arg(2));
    }
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let tid = ParamBinding {
        param: 1,
        role: ParamRole::Builtin(crate::abi::Builtin::ThreadPositionInGrid),
        unused: false,
        predicate: Some(off.clone()),
        ty: ArgTypeInfo::default(),
        name: Some(DiagName::new("tid")),
    };
    let gone = ParamBinding {
        predicate: Some(off),
        ..buffer(0, 0)
    };
    let abi = abi(vec![kernel(k, vec![gone, tid, buffer(2, 1)])], Vec::new());
    specialize(module, &abi, &request(&[]), &mut meter())
}

/// A read of a gated-off buffer reads the null pointer, and an unobserved read of a gated-off
/// builtin reads `undef`, which nothing then observes, so it goes with the read; neither argument
/// stays in the entry point.
#[test]
fn absent_arguments_read_their_absent_values() {
    let e = absent_reads(false).unwrap();
    let kept: Vec<u32> = e.ep.params.iter().map(|p| p.param).collect();
    assert_eq!(kept, [2]);
    let text = body_text(&e);
    assert!(
        text.contains("load i32, ptr addrspace(1) zeroinitializer"),
        "{text}"
    );
    assert!(!text.contains("add i32"), "{text}");
    assert!(!text.contains("%a0") && !text.contains("%a1"), "{text}");
}

/// An observed read of a gated-off argument passed by value has no value Metal defines (the
/// argument is not declared, MSL 5.8.1.2), so it reads the all-zero value of its type, which
/// refines the undefined one; the argument still leaves the entry point.
#[test]
fn an_observed_read_of_an_absent_value_argument_reads_zero() {
    let e = absent_reads(true).unwrap();
    let kept: Vec<u32> = e.ep.params.iter().map(|p| p.param).collect();
    assert_eq!(kept, [2]);
    let text = body_text(&e);
    assert!(
        text.contains("add i32 i32 zeroinitializer, i32 1"),
        "{text}"
    );
    assert!(!text.contains("undef"), "{text}");
    assert!(!text.contains("%a0") && !text.contains("%a1"), "{text}");
}

/// How the body of [`absent_threadgroup`] reaches its gated-off threadgroup memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TgUse {
    /// `out[0] = *tg`.
    Read,
    /// Through element addresses (`&tg[1]`, and `&(&tg[2])[1]`, two steps): `tg[1] = 5`,
    /// `out[0] = tg[1]`, `out[1] = atomic_fetch_add(&tg[3], 1)`, `atomic_store(&tg[3], 2)`.
    Accesses,
    /// `out[0] = (tg == null)`: the pointer is compared, which is no access through it.
    Compared,
    /// `out[0] = atomic_compare_exchange_weak(&tg[0], &expected, 1)`, `expected` a stack slot the
    /// exchange writes.
    Exchanged,
}

/// `kernel k(threadgroup i32* tg [[off]], device i32* out)` using `tg` as `how` says.
fn absent_threadgroup(how: TgUse) -> Result<Entry, Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let (i1, i32t) = (m.int(1), m.int(32));
    let dev = m.ptr(AddrSpace::Device);
    let tg = m.ptr(AddrSpace::Threadgroup);
    let thread = m.ptr(AddrSpace::Thread);
    let off = predicate(&mut m, "off", 0);
    let atomic = |m: &mut ModuleBuilder, name: &str, ty: TypeId| {
        let decl = m.declare(name, ty);
        let op = AirIntrinsic::parse(name).expect("an atomic parses");
        Callee::Air { decl, op }
    };
    let rmw_ty = m.fn_ty(i32t, &[tg, i32t, i32t, i32t, i1]);
    let add = atomic(&mut m, "air.atomic.local.add.u.i32", rmw_ty);
    let put_ty = m.fn_ty(void, &[tg, i32t, i32t, i32t, i1]);
    let put = atomic(&mut m, "air.atomic.local.store.i32", put_ty);
    let cas_ty = m.fn_ty(i32t, &[tg, thread, i32t, i32t, i32t, i32t, i1]);
    let cas = atomic(&mut m, "air.atomic.local.cmpxchg.weak.i32", cas_ty);
    let k = |m: &mut ModuleBuilder, v: u64| m.const_int(i32t, v);
    let (zero, one, two, five) = (k(&mut m, 0), k(&mut m, 1), k(&mut m, 2), k(&mut m, 5));
    let (relaxed, scope, no) = (k(&mut m, 0), k(&mut m, 1), m.const_int(i1, 0));
    let null = Value::Const(m.constant(tg, Const::Null));
    let fty = m.fn_ty(void, &[tg, dev]);
    let mut f = m.define("k", fty);
    let at = |f: &mut FunctionBuilder<'_>, ty: TypeId, base: Value, index: Value| {
        f.push(
            ty,
            InstKind::Gep {
                src_ty: i32t,
                base,
                indices: Box::new([index]),
                inbounds: true,
            },
        )
    };
    match how {
        TgUse::Read => {
            let x = load(&mut f, i32t, arg(0));
            store(&mut f, x, arg(1));
        }
        TgUse::Accesses => {
            let first = at(&mut f, tg, arg(0), one);
            store(&mut f, five, first);
            let x = load(&mut f, i32t, first);
            store(&mut f, x, arg(1));
            let second = at(&mut f, tg, arg(0), two);
            let third = at(&mut f, tg, second, one);
            let old = call(&mut f, i32t, add, &[third, one, relaxed, scope, no]);
            let out1 = at(&mut f, dev, arg(1), one);
            store(&mut f, old, out1);
            call(&mut f, void, put, &[third, two, relaxed, scope, no]);
        }
        TgUse::Compared => {
            let is_null = f.push(
                i1,
                InstKind::ICmp {
                    pred: IntPredicate::Eq,
                    a: arg(0),
                    b: null,
                },
            );
            let wide = f.push(
                i32t,
                InstKind::Cast {
                    op: CastOp::ZExt,
                    v: is_null,
                },
            );
            store(&mut f, wide, arg(1));
        }
        TgUse::Exchanged => {
            let expected = f.push(
                thread,
                InstKind::Alloca {
                    ty: i32t,
                    count: one,
                    align: None,
                },
            );
            store(&mut f, zero, expected);
            let won = call(
                &mut f,
                i32t,
                cas,
                &[arg(0), expected, one, relaxed, relaxed, scope, no],
            );
            store(&mut f, won, arg(1));
        }
    }
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let ParamBinding {
        role: ParamRole::Buffer(b),
        ..
    } = buffer(0, 0)
    else {
        unreachable!("buffer() binds a buffer")
    };
    let gone = ParamBinding {
        role: ParamRole::Threadgroup(BufferArg {
            space: AddrSpace::Threadgroup,
            ..b
        }),
        predicate: Some(off),
        name: Some(DiagName::new("tg")),
        ..buffer(0, 0)
    };
    let abi = abi(vec![kernel(k, vec![gone, buffer(1, 0)])], Vec::new());
    specialize(module, &abi, &request(&[]), &mut meter())
}

/// The absent threadgroup accesses in `e`'s entry, in body order.
fn absent_memory_ops(e: &Entry) -> Vec<AbsentAccess> {
    let body = e
        .module
        .func(e.ep.func)
        .and_then(|f| f.body.as_ref())
        .expect("the entry has a body");
    body.insts
        .values()
        .filter_map(|i| match &i.kind {
            InstKind::Call(c) => match c.callee {
                Callee::Air {
                    op: AirIntrinsic::FunctionConstant(FunctionConstant::AbsentMemory(a)),
                    ..
                } => {
                    assert!(c.args.is_empty(), "an absent read takes no operand");
                    Some(a)
                }
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// Threadgroup memory a function constant removed has no contents Metal defines (measured: it
/// varies with the validation layer). A load through it, at its own address or one
/// derived from it in one or more steps, and an atomic on it are the absent read of their type (an
/// atomic's stays though nothing observes it); a store into it, a plain one or an atomic one, is
/// dropped. The argument leaves the entry point, and no address derived from it stays.
#[test]
fn an_access_through_absent_threadgroup_memory_is_an_absent_read_or_dropped() {
    let i32 = NumTy::scalar(Elem::I32);
    for (how, want) in [
        (
            TgUse::Read,
            vec![AbsentAccess {
                ty: i32,
                atomic: false,
            }],
        ),
        (
            TgUse::Accesses,
            vec![
                AbsentAccess {
                    ty: i32,
                    atomic: false,
                },
                AbsentAccess {
                    ty: i32,
                    atomic: true,
                },
            ],
        ),
    ] {
        let e = absent_threadgroup(how).unwrap();
        let params: Vec<u32> = e.ep.params.iter().map(|p| p.param).collect();
        assert_eq!(params, [1], "{how:?}");
        assert_eq!(absent_memory_ops(&e), want, "{how:?}");
        let text = body_text(&e);
        assert!(!text.contains("%a0"), "{how:?}: {text}");
        // Only `out[1]`'s address stays.
        let geps = text.matches("getelementptr").count();
        assert_eq!(geps, usize::from(how == TgUse::Accesses), "{how:?}: {text}");
        assert!(!text.contains("air.atomic"), "{how:?}: {text}");
        assert!(!text.contains("i32 5"), "{how:?}: {text}");
    }
}

/// Absent threadgroup memory used other than as the address of a load, a store, or an atomic
/// (compared with null), or by an atomic that also writes thread memory (a compare-exchange's
/// `expected`), has no contained answer: it stays refused at its metadata, naming it.
#[test]
fn absent_threadgroup_memory_used_another_way_is_refused() {
    for how in [TgUse::Compared, TgUse::Exchanged] {
        let u = refusal(absent_threadgroup(how));
        assert_eq!(u.feature, Feature::FunctionConstantArgument, "{how:?}");
        assert_eq!(
            u.location,
            Some(Location::metadata(Some(DiagName::new("k")), Some(0), None)),
            "{how:?}"
        );
        assert_eq!(u.symbol, Some(DiagName::new("tg")), "{how:?}");
    }
}

/// What sits between an absent texture and the store of its sample in [`absent_sample`].
#[derive(Clone, Copy, PartialEq, Eq)]
enum SampleUse {
    /// The sample's color is stored.
    Stored,
    /// The sample's result goes nowhere.
    Dropped,
    /// The texture is also compared with null, a use that is no texture call.
    Compared,
    /// The texture is stored into a stack slot and sampled as loaded back from it.
    Parked,
}

/// `kernel k(texture2d tex, sampler smp, device float4* out)` sampling `tex` through `smp` and
/// using the result as `how` says; `gone` (0, the texture, or 1, the sampler) is gated off.
fn absent_sample(gone: u32, how: SampleUse) -> Result<Entry, Error> {
    use crate::abi::TextureArg;
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let (i1, i8t, i32t, f32t) = (m.int(1), m.int(8), m.int(32), m.float());
    let (v2f, v2i, v4f) = (m.vector(f32t, 2), m.vector(i32t, 2), m.vector(f32t, 4));
    let ret = m.struct_ty(&[v4f, i8t]);
    let (dev, cst) = (m.ptr(AddrSpace::Device), m.ptr(AddrSpace::Constant));
    let off = predicate(&mut m, "off", 0);
    let name = "air.sample_texture_2d.v4f32";
    let sample_ty = m.fn_ty(ret, &[dev, cst, v2f, i1, v2i, i1, f32t, f32t, i32t]);
    let decl = m.declare(name, sample_ty);
    let op = AirIntrinsic::parse(name).expect("a sample parses");
    let (half, yes, zero2, zf, z) = (
        Value::Const(m.constant(v2f, Const::Null)),
        m.const_int(i1, 1),
        Value::Const(m.constant(v2i, Const::Null)),
        Value::Const(m.constant(f32t, Const::Null)),
        m.const_int(i32t, 0),
    );
    let fty = m.fn_ty(void, &[dev, cst, dev]);
    let thread = m.ptr(AddrSpace::Thread);
    let one = m.const_int(i32t, 1);
    let mut f = m.define("k", fty);
    let tex = match how {
        SampleUse::Parked => {
            let slot = f.push(
                thread,
                InstKind::Alloca {
                    ty: dev,
                    count: one,
                    align: None,
                },
            );
            store(&mut f, arg(0), slot);
            load(&mut f, dev, slot)
        }
        SampleUse::Stored | SampleUse::Dropped | SampleUse::Compared => arg(0),
    };
    let r = call(
        &mut f,
        ret,
        Callee::Air { decl, op },
        &[tex, arg(1), half, yes, zero2, yes, zf, zf, z],
    );
    match how {
        SampleUse::Stored | SampleUse::Parked => {
            let color = f.push(
                v4f,
                InstKind::ExtractValue {
                    agg: r,
                    indices: Box::new([0]),
                },
            );
            store(&mut f, color, arg(2));
        }
        SampleUse::Dropped => {}
        SampleUse::Compared => {
            let null = Value::Const(f.module().constant(dev, Const::Null));
            let is_null = f.push(
                i1,
                InstKind::ICmp {
                    pred: IntPredicate::Eq,
                    a: arg(0),
                    b: null,
                },
            );
            let wide = f.push(
                i32t,
                InstKind::Cast {
                    op: CastOp::ZExt,
                    v: is_null,
                },
            );
            store(&mut f, wide, arg(2));
        }
    }
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let at = |index: u32| Slot {
        index: AbiInt::Const(index),
        count: AbiInt::Const(1),
    };
    let gate = |p: u32| (p == gone).then(|| off.clone());
    let tex = ParamBinding {
        role: ParamRole::Texture(TextureArg {
            location: at(0),
            access: Access::Sample,
            raster_order_group: None,
        }),
        predicate: gate(0),
        name: Some(DiagName::new("tex")),
        ..buffer(0, 0)
    };
    let smp = ParamBinding {
        role: ParamRole::Sampler { location: at(0) },
        predicate: gate(1),
        name: Some(DiagName::new("smp")),
        ..buffer(1, 0)
    };
    let abi = abi(vec![kernel(k, vec![tex, smp, buffer(2, 0)])], Vec::new());
    specialize(module, &abi, &request(&[]), &mut meter())
}

/// The operations of the absent reads in `e`'s entry, in body order.
fn absent_read_ops(e: &Entry) -> Vec<TexOp> {
    let body = e
        .module
        .func(e.ep.func)
        .and_then(|f| f.body.as_ref())
        .expect("the entry has a body");
    body.insts
        .values()
        .filter_map(|i| match &i.kind {
            InstKind::Call(c) => match c.callee {
                Callee::Air {
                    op: AirIntrinsic::FunctionConstant(FunctionConstant::AbsentRead(t)),
                    ..
                } => {
                    assert!(c.args.is_empty(), "an absent read takes no operand");
                    Some(t.op)
                }
                _ => None,
            },
            _ => None,
        })
        .collect()
}

/// An observed sample of a gated-off texture, or through a gated-off sampler, is an absent read of
/// the sample: no operand, the sample's result type, and the gated-off argument leaves the entry
/// point. Metal gives the read no value; lowering marks the command buffer.
#[test]
fn an_observed_sample_of_an_absent_texture_or_through_an_absent_sampler_is_an_absent_read() {
    for (gone, kept) in [(0, [1, 2]), (1, [0, 2])] {
        let e = absent_sample(gone, SampleUse::Stored).unwrap();
        let params: Vec<u32> = e.ep.params.iter().map(|p| p.param).collect();
        assert_eq!(params, kept, "gone {gone}");
        let ops = absent_read_ops(&e);
        assert_eq!(ops.len(), 1, "gone {gone}");
        assert!(matches!(ops[0], TexOp::Sample { .. }), "gone {gone}");
        let text = body_text(&e);
        assert!(!text.contains("air.sample_texture_2d"), "{text}");
        assert!(text.contains("absent argument read"), "{text}");
        assert!(!text.contains(&format!("%a{gone}")), "{text}");
    }
}

/// A sample of an absent texture whose result nothing observes goes with the rest of what nothing
/// observes, so it marks nothing.
#[test]
fn an_unobserved_sample_of_an_absent_texture_goes() {
    let e = absent_sample(0, SampleUse::Dropped).unwrap();
    assert!(absent_read_ops(&e).is_empty());
    let text = body_text(&e);
    assert!(!text.contains("absent argument read"), "{text}");
    assert!(!text.contains("air.sample_texture_2d"), "{text}");
}

/// An absent texture read other than as a texture call's own operand (compared with null, or stored
/// into a stack slot and sampled as loaded back) stays in the body, and the entry lists it for
/// lowering, which follows its handle to the calls it reaches; its call is not rewritten here.
#[test]
fn an_absent_texture_read_beyond_texture_calls_stays_for_lowering() {
    for how in [SampleUse::Compared, SampleUse::Parked] {
        let e = absent_sample(0, how).unwrap();
        let params: Vec<u32> = e.ep.params.iter().map(|p| p.param).collect();
        assert_eq!(params, [1, 2]);
        let kept: Vec<u32> = e.absent_handles.iter().map(|p| p.param).collect();
        assert_eq!(kept, [0]);
        assert_eq!(e.absent_handles[0].name, Some(DiagName::new("tex")));
        assert!(absent_read_ops(&e).is_empty());
        let text = body_text(&e);
        assert!(text.contains("air.sample_texture_2d"), "{text}");
        assert!(text.contains("%a0"), "{text}");
    }
    // Read only by its call, it is rewritten as before and not listed.
    let e = absent_sample(0, SampleUse::Stored).unwrap();
    assert!(e.absent_handles.is_empty());
}

/// An output gated off leaves the entry point; the outputs that stay name their field of the
/// returned struct.
#[test]
fn a_gated_off_output_leaves_the_entry_point() {
    use crate::abi::{OutputBinding, OutputRole, UserKey};
    let mut m = ModuleBuilder::new();
    let f32t = m.float();
    let v4 = m.vector(f32t, 4);
    let ret = m.struct_ty(&[v4, v4, v4]);
    let (off, on) = (predicate(&mut m, "off", 0), predicate(&mut m, "on", 1));
    let fty = m.fn_ty(ret, &[]);
    let undef = m.constant(ret, Const::Undef);
    let mut f = m.define("k", fty);
    f.ret(Some(Value::Const(undef)));
    let k = f.finish();
    let module = m.finish().unwrap();
    let output = |field: u32, role: OutputRole, predicate: Option<Predicate>| OutputBinding {
        field,
        role,
        shared: false,
        predicate,
        ty: ArgTypeInfo::default(),
        name: None,
    };
    let mut ep = kernel(k, Vec::new());
    ep.stage = Stage::Vertex { patch: None };
    ep.outputs = vec![
        output(0, OutputRole::Position { invariant: false }, None),
        output(
            1,
            OutputRole::VertexOutput(UserKey(b"a".as_slice().into())),
            Some(off),
        ),
        output(
            2,
            OutputRole::VertexOutput(UserKey(b"b".as_slice().into())),
            Some(on),
        ),
    ];
    let abi = abi(vec![ep], Vec::new());
    let e = specialize(module, &abi, &request(&[]), &mut meter()).unwrap();
    let fields: Vec<u32> = e.ep.outputs.iter().map(|o| o.field).collect();
    assert_eq!(fields, [0, 2]);
    assert!(e.ep.outputs.iter().all(|o| o.predicate.is_none()));
}

/// A vertex entry `(v4 pos [[on]], v4 uv [[off]]) -> {pos, uv}` whose returned struct inserts
/// both inputs; the `uv` output is gated by `off` when `gate_output`.
fn absent_into_output(gate_output: bool) -> Result<Entry, Error> {
    use crate::abi::{Builtin, OutputBinding, OutputRole, UserKey};
    let mut m = ModuleBuilder::new();
    let f32t = m.float();
    let v4 = m.vector(f32t, 4);
    let ret = m.struct_ty(&[v4, v4]);
    let off = predicate(&mut m, "off", 0);
    let fty = m.fn_ty(ret, &[v4, v4]);
    let undef = m.constant(ret, Const::Undef);
    let mut f = m.define("k", fty);
    let a = f.push(
        ret,
        InstKind::InsertValue {
            agg: Value::Const(undef),
            val: arg(0),
            indices: [0].into(),
        },
    );
    let b = f.push(
        ret,
        InstKind::InsertValue {
            agg: a,
            val: arg(1),
            indices: [1].into(),
        },
    );
    f.ret(Some(b));
    let k = f.finish();
    let module = m.finish().unwrap();
    let param = |param: u32, predicate: Option<Predicate>, name: &str| ParamBinding {
        param,
        role: ParamRole::Builtin(Builtin::VertexId),
        unused: false,
        predicate,
        ty: ArgTypeInfo::default(),
        name: Some(DiagName::new(name)),
    };
    let output = |field: u32, role: OutputRole, predicate: Option<Predicate>| OutputBinding {
        field,
        role,
        shared: false,
        predicate,
        ty: ArgTypeInfo::default(),
        name: None,
    };
    let mut ep = kernel(
        k,
        vec![param(0, None, "pos"), param(1, Some(off.clone()), "uv")],
    );
    ep.stage = Stage::Vertex { patch: None };
    ep.outputs = vec![
        output(0, OutputRole::Position { invariant: false }, None),
        output(
            1,
            OutputRole::VertexOutput(UserKey(b"uv".as_slice().into())),
            gate_output.then_some(off),
        ),
    ];
    let abi = abi(vec![ep], Vec::new());
    specialize(module, &abi, &request(&[]), &mut meter())
}

/// A gated-off input that only feeds a returned field whose output the same gate removed is not
/// observed (the return stores only declared fields), so it reads `undef`; with the output
/// declared, the same read is observed and reads zero.
#[test]
fn an_absent_input_feeding_only_an_absent_output_is_unobserved() {
    let e = absent_into_output(true).unwrap();
    let fields: Vec<u32> = e.ep.outputs.iter().map(|o| o.field).collect();
    assert_eq!(fields, [0]);
    let text = body_text(&e);
    assert!(!text.contains("%a1"), "{text}");
    assert!(text.contains("<4 x float> undef, 1"), "{text}");
    let e = absent_into_output(false).unwrap();
    let fields: Vec<u32> = e.ep.outputs.iter().map(|o| o.field).collect();
    assert_eq!(fields, [0, 1]);
    let text = body_text(&e);
    assert!(!text.contains("%a1"), "{text}");
    assert!(text.contains("<4 x float> zeroinitializer, 1"), "{text}");
}

/// An entry that returns its one output itself, gated off, stores nothing, so a gated-off input it
/// returns is not observed; declared, the output observes it, and it reads zero.
#[test]
fn a_returned_value_whose_output_is_gated_off_observes_nothing() {
    use crate::abi::{Builtin, OutputBinding, OutputRole, UserKey};
    let run = |gate_output: bool| {
        let mut m = ModuleBuilder::new();
        let f32t = m.float();
        let off = predicate(&mut m, "off", 0);
        let fty = m.fn_ty(f32t, &[f32t]);
        let mut f = m.define("k", fty);
        f.ret(Some(arg(0)));
        let k = f.finish();
        let module = m.finish().unwrap();
        let mut ep = kernel(
            k,
            vec![ParamBinding {
                param: 0,
                role: ParamRole::Builtin(Builtin::VertexId),
                unused: false,
                predicate: Some(off.clone()),
                ty: ArgTypeInfo::default(),
                name: Some(DiagName::new("x")),
            }],
        );
        ep.stage = Stage::Vertex { patch: None };
        ep.outputs = vec![OutputBinding {
            field: 0,
            role: OutputRole::VertexOutput(UserKey(b"x".as_slice().into())),
            shared: false,
            predicate: gate_output.then_some(off),
            ty: ArgTypeInfo::default(),
            name: None,
        }];
        let abi = abi(vec![ep], Vec::new());
        specialize(module, &abi, &request(&[]), &mut meter())
    };
    let e = run(true).unwrap();
    assert!(e.ep.outputs.is_empty());
    assert!(
        body_text(&e).contains("ret float undef"),
        "{}",
        body_text(&e)
    );
    let e = run(false).unwrap();
    assert_eq!(e.ep.outputs.len(), 1);
    assert!(e.ep.params.is_empty());
    let text = body_text(&e);
    assert!(text.contains("ret float zeroinitializer"), "{text}");
}

/// A body a pass rebuilds is on the ledger with what its instructions hold outside their slots:
/// `rewrite::apply` appends through `Body::push_inst`, so once the rebuild's own tables are
/// returned the ledger holds exactly the arenas' capacity, reserved at the survivors' count with no
/// doubling slack (`Body::with_capacity`), and every instruction's boxed payload: each call's box,
/// arguments, attributes with their string and per-argument lists, and the PHI's incoming edges.
/// Freeing it with `Body::release` returns exactly that charge.
#[test]
fn a_rebuilt_body_is_charged_its_calls_payloads() {
    use crate::ir::{Block, CallAttrs, FloatPredicate, FnAttrs, Inst};
    use crate::ir::{Opcode, ParamAttrs};
    let mut m = ModuleBuilder::new();
    let f32t = m.float();
    let i1 = m.int(1);
    let binary = m.fn_ty(f32t, &[f32t, f32t]);
    let g = m.declare("g", binary);
    let (key, first, second) = (m.sym("payload"), m.sym("first"), m.sym("second"));
    let attrs = CallAttrs {
        func: FnAttrs {
            strings: [(key, first)].into(),
            ..FnAttrs::default()
        },
        ret: ParamAttrs::default(),
        params: [first, second]
            .map(|value| ParamAttrs {
                strings: [(key, value)].into(),
                ..ParamAttrs::default()
            })
            .into(),
    };
    let mut f = m.define("k", binary);
    let (then, other, join) = (f.block(), f.block(), f.block());
    let (mut a, mut b) = (arg(0), arg(1));
    for _ in 0..64 {
        let c = f.push(
            f32t,
            InstKind::Call(Box::new(Call {
                callee: Callee::Direct(g),
                args: [a, b].into(),
                fmf: FastMath::NONE,
                tail: TailKind::Tail,
                attrs: Some(Box::new(attrs.clone())),
                range: None,
            })),
        );
        (a, b) = (c, a);
    }
    let lt = f.push(
        i1,
        InstKind::FCmp {
            pred: FloatPredicate::Olt,
            fmf: FastMath::NONE,
            a,
            b,
        },
    );
    f.cond_br(lt, then, other);
    f.switch_to(then);
    f.br(join);
    f.switch_to(other);
    f.br(join);
    f.switch_to(join);
    let merged = f.push(
        f32t,
        InstKind::Phi {
            incoming: [(a, then), (b, other)].into(),
            fmf: FastMath::NONE,
        },
    );
    f.ret(Some(merged));
    let k = f.finish();
    let module = m.finish().unwrap();
    let body = module.func(k).and_then(|f| f.body.as_ref()).unwrap();

    let payload: usize = body.insts.values().map(|i| i.kind.heap_bytes()).sum();
    let calls = body
        .insts
        .values()
        .filter(|i| i.kind.opcode() == Opcode::Call)
        .count();
    let per_call = std::mem::size_of::<Call>()
        + 2 * std::mem::size_of::<Value>()
        + std::mem::size_of::<CallAttrs>()
        + 2 * std::mem::size_of::<ParamAttrs>()
        + 3 * 2 * std::mem::size_of::<crate::ir::Sym>();
    let phi = 2 * std::mem::size_of::<(Value, BlockId)>();
    assert_eq!(calls, 64);
    assert_eq!(payload, calls * per_call + phi);

    let slots = body.insts.len() * std::mem::size_of::<Inst>()
        + body.blocks.len() * std::mem::size_of::<Block>();
    let edit = rewrite::BodyEdit::keep_all(body, &mut meter(), &mut 0).unwrap();
    let mut ledger = meter();
    let rebuilt = rewrite::apply(body, &edit, &|c| c, &mut ledger).unwrap();
    assert_eq!(rebuilt, *body);
    assert_eq!(ledger.ledger(), (slots + payload) as u64);

    // Freeing the body returns exactly that charge.
    assert_eq!(rebuilt.footprint() as u64, ledger.ledger());
    rebuilt.release(&mut ledger);
    assert_eq!(ledger.ledger(), 0);
}

/// The specialization of a kernel `k(p)` whose one parameter, of address space `space`, has role
/// `role`; its slot count may be function constant 0 (`n`, supplied as `n`).
fn slot_spec(
    role: impl FnOnce(GlobalId) -> ParamRole,
    space: AddrSpace,
    n: u32,
) -> Result<(), Error> {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let count = fc_global(&mut m, "n", i32t);
    let p = m.ptr(space);
    let fty = m.fn_ty(void, &[p]);
    let mut f = m.define("k", fty);
    f.ret(None);
    let k = f.finish();
    let module = m.finish().unwrap();
    let mut b = buffer(0, 0);
    b.role = role(count);
    let abi = abi(vec![kernel(k, vec![b])], vec![fc_decl(0, count, i32t)]);
    specialize(module, &abi, &request(&[(0, int(n))]), &mut meter()).map(|_| ())
}

/// An entry parameter's argument-table slots lie in Metal's table for its kind (31 buffer, 31
/// threadgroup memory, 128 texture, and 16 sampler entries per function, every GPU family's
/// maximum; an acceleration structure or function table binds at a buffer index): the last slot
/// of each table is accepted, and one past it, from the index, the count, or a count a function
/// constant gives, is invalid input at the parameter's `air.location_index`.
#[test]
fn argument_table_slots_past_metal_s_tables_are_invalid_input() {
    use crate::abi::{AccelerationStructureKind, ObjectArg, TextureArg};
    use crate::error::{AbiError, Location};
    let konst = |index: u32, count: u32| Slot {
        index: AbiInt::Const(index),
        count: AbiInt::Const(count),
    };
    let buffer_at = |location: Slot| {
        move |_| match buffer(0, 0).role {
            ParamRole::Buffer(b) => ParamRole::Buffer(BufferArg { location, ..b }),
            other => other,
        }
    };
    let texture_at = |location: Slot| {
        move |_| {
            ParamRole::Texture(TextureArg {
                location,
                access: Access::Read,
                raster_order_group: None,
            })
        }
    };
    let sampler_at = |location: Slot| move |_| ParamRole::Sampler { location };
    let object_at = |location: Slot| {
        move |_| ParamRole::AccelerationStructure {
            kind: AccelerationStructureKind::Primitive,
            arg: ObjectArg {
                location,
                access: None,
            },
        }
    };
    let table_at = |location: Slot| {
        move |_| {
            ParamRole::IntersectionTable(ObjectArg {
                location,
                access: None,
            })
        }
    };
    let threadgroup_at = |location: Slot| {
        move |_| match buffer(0, 0).role {
            ParamRole::Buffer(b) => ParamRole::Threadgroup(BufferArg {
                location,
                space: AddrSpace::Threadgroup,
                ..b
            }),
            other => other,
        }
    };
    let past = |r: Result<(), Error>| match r {
        Err(Error::InvalidInput(e)) => {
            e.kind == InputErrorKind::Abi(AbiError::SlotPastTable)
                && e.at
                    == Location::metadata(
                        Some(DiagName::new("k")),
                        Some(0),
                        Some("air.location_index"),
                    )
        }
        _ => false,
    };
    let (dev, cst) = (AddrSpace::Device, AddrSpace::Constant);
    // Buffers: slots 0..=30.
    assert_eq!(slot_spec(buffer_at(konst(30, 1)), dev, 0), Ok(()));
    assert_eq!(
        slot_spec(buffer_at(konst(0, 31)), AddrSpace::Thread, 0),
        Ok(())
    );
    assert!(past(slot_spec(buffer_at(konst(31, 1)), dev, 0)));
    assert!(past(slot_spec(
        buffer_at(konst(1, 31)),
        AddrSpace::Thread,
        0
    )));
    // A buffer table whose count a function constant gives.
    let fc_count = |index: u32| {
        move |g| match buffer(0, 0).role {
            ParamRole::Buffer(b) => ParamRole::Buffer(BufferArg {
                location: Slot {
                    index: AbiInt::Const(index),
                    count: AbiInt::Global(g),
                },
                ..b
            }),
            other => other,
        }
    };
    assert_eq!(slot_spec(fc_count(1), AddrSpace::Thread, 30), Ok(()));
    assert!(past(slot_spec(fc_count(1), AddrSpace::Thread, 31)));
    // Textures: slots 0..=127.
    assert_eq!(slot_spec(texture_at(konst(127, 1)), dev, 0), Ok(()));
    assert_eq!(
        slot_spec(texture_at(konst(0, 128)), AddrSpace::Thread, 0),
        Ok(())
    );
    assert!(past(slot_spec(texture_at(konst(128, 1)), dev, 0)));
    assert!(past(slot_spec(
        texture_at(konst(1, 128)),
        AddrSpace::Thread,
        0
    )));
    // Acceleration structures and function tables bind at a buffer index: slots 0..=30.
    assert_eq!(slot_spec(object_at(konst(30, 1)), dev, 0), Ok(()));
    assert!(past(slot_spec(object_at(konst(31, 1)), dev, 0)));
    assert_eq!(slot_spec(table_at(konst(30, 1)), dev, 0), Ok(()));
    assert!(past(slot_spec(table_at(konst(31, 1)), dev, 0)));
    // Threadgroup memory: slots 0..=30 of its own table.
    let tg = AddrSpace::Threadgroup;
    assert_eq!(slot_spec(threadgroup_at(konst(30, 1)), tg, 0), Ok(()));
    assert!(past(slot_spec(threadgroup_at(konst(31, 1)), tg, 0)));
    // Samplers: slots 0..=15.
    assert_eq!(slot_spec(sampler_at(konst(15, 1)), cst, 0), Ok(()));
    assert!(past(slot_spec(sampler_at(konst(16, 1)), cst, 0)));
    assert!(past(slot_spec(
        sampler_at(konst(0, 17)),
        AddrSpace::Thread,
        0
    )));
}

/// A site of the `matmul2d` lowering calls a `matmul2d` ABI declaration or a tensor intrinsic, so
/// in a module that declares neither, the expansion and the refusal of what is left read no
/// instruction: each costs one step per function, however long the entry. A module that declares
/// one is read whole.
#[test]
fn the_matmul2d_passes_skip_a_module_without_their_declarations() {
    let work = |adds: usize, declared: Option<&str>| -> u64 {
        let mut m = ModuleBuilder::new();
        let void = m.void();
        let i32t = m.int(32);
        let fty = m.fn_ty(void, &[i32t]);
        if let Some(name) = declared {
            let d = m.declare(name, fty);
            let section = m.sym("air.externally_defined");
            m.func_mut(d).unwrap().section = Some(Section::Other(section));
        }
        let mut f = m.define("k", fty);
        let mut v = arg(0);
        for _ in 0..adds {
            v = f.bin(BinOp::Add, v, arg(0));
        }
        f.ret(None);
        let k = f.finish();
        let mut parts = m.finish().unwrap().into_parts();
        let body = parts.funcs.get(k).and_then(|f| f.body.clone()).unwrap();
        let ep = kernel(k, Vec::new());
        let mut meter = meter();
        let (body, lanes) = matmul2d::expand(body, &mut parts, &ep, &mut meter).unwrap();
        assert!(lanes.is_none());
        matmul2d::refuse_left(&body, &parts, &ep.stage, &ep.name, &mut meter).unwrap();
        meter.work()
    };
    let functions = 2;
    assert_eq!(work(10, Some("other")), functions * 2);
    assert_eq!(work(1000, Some("other")), functions * 2);
    for name in [
        "__tensorops_impl_matmul2d_op_cooperative_tensor_init",
        "air.get_extent_private_tensor.i32",
    ] {
        assert!(work(1000, Some(name)) > 2000, "{name}");
    }
}
