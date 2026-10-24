//! Devirtualization tests over IR built with the typed test builder: calls through visible-function
//! tables and through constant function addresses.

use std::collections::BTreeMap;

use super::*;
use crate::abi::{
    AbiInt, Access, AirAbi, ArgTypeInfo, BufferArg, CompileOptions, EntryPoint, IndirectArg,
    Member, MemberKind, ModuleLimits, ObjectArg, ParamBinding, ParamRole, Slot, Stage, StructInfo,
    StructRef, UserKey, VisibleFunction,
};
use crate::budget::{Budget, Meter};
use crate::intrinsics::{AirIntrinsic, FunctionTable};
use crate::ir::build::{FunctionBuilder, ModuleBuilder};
use crate::ir::print::print;
use crate::ir::Value;
use crate::ir::{AddrSpace, Align, BinOp, Call, CastOp, Const, FastMath, IntPredicate, TailKind};

fn meter() -> Meter {
    Meter::new(&Budget::default())
}

fn abi() -> AirAbi {
    AirAbi {
        air_version: None,
        language: None,
        compile: CompileOptions::default(),
        limits: ModuleLimits::default(),
        entries: Vec::new(),
        function_constants: Vec::new(),
        static_samplers: Vec::new(),
        visible_function_refs: Vec::new(),
        visible_functions: Vec::new(),
        core_image_kernels: Vec::new(),
        linkage_signature: None,
        structs: Vec::new(),
        producer: None,
    }
}

fn slot(index: u32) -> Slot {
    Slot {
        index: AbiInt::Const(index),
        count: AbiInt::Const(1),
    }
}

fn binding(param: u32, role: ParamRole) -> ParamBinding {
    ParamBinding {
        param,
        role,
        unused: false,
        predicate: None,
        ty: ArgTypeInfo::default(),
        name: None,
    }
}

fn out(param: u32) -> ParamBinding {
    binding(
        param,
        ParamRole::Buffer(BufferArg {
            location: slot(0),
            access: Access::ReadWrite,
            space: AddrSpace::Device,
            buffer_size: None,
            layout: None,
            raster_order_group: None,
        }),
    )
}

fn table(param: u32) -> ParamBinding {
    table_at(param, 1)
}

fn table_at(param: u32, index: u32) -> ParamBinding {
    binding(
        param,
        ParamRole::FunctionTable(ObjectArg {
            location: slot(index),
            access: Some(Access::Read),
        }),
    )
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

fn arg(i: u32) -> Value {
    Value::Arg(crate::ir::ArgIdx::new(i))
}

fn request() -> SpecRequest {
    SpecRequest {
        call_stack_depth: 1,
        entry: "k".into(),
        constants: BTreeMap::new(),
    }
}

/// `air.get_function_pointer_visible_function_table` declared in `m`.
fn lookup_decl(m: &mut ModuleBuilder) -> FuncId {
    let thread = m.ptr(AddrSpace::Thread);
    let dev = m.ptr(AddrSpace::Device);
    let i32t = m.int(32);
    let ty = m.fn_ty(thread, &[dev, i32t]);
    m.declare("air.get_function_pointer_visible_function_table", ty)
}

fn lookup(f: &mut FunctionBuilder<'_>, decl: FuncId, table: Value, index: Value) -> Value {
    let thread = f.module().ptr(AddrSpace::Thread);
    let op = AirIntrinsic::FunctionTable(FunctionTable::GetFunctionPointer);
    call(f, thread, Callee::Air { decl, op }, &[table, index])
}

/// The kernel `k(device i32* out, table t, i32 tid)`: `*out = t[tid](tid)`, the call behind
/// `br false` when `dead`.
fn table_kernel(dead: bool) -> (Module, AirAbi) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let fty = m.fn_ty(i32t, &[i32t]);
    let decl = lookup_decl(&mut m);
    let no = m.const_int(i1, 0);
    let kty = m.fn_ty(void, &[dev, dev, i32t]);
    let mut k = m.define("k", kty);
    if dead {
        let (then, done) = (k.block(), k.block());
        k.cond_br(no, then, done);
        k.switch_to(then);
        let fp = lookup(&mut k, decl, arg(1), arg(2));
        let callee = Callee::Indirect {
            callee: fp,
            fn_ty: fty,
        };
        let r = call(&mut k, i32t, callee, &[arg(2)]);
        store(&mut k, r, arg(0));
        k.br(done);
        k.switch_to(done);
    } else {
        let fp = lookup(&mut k, decl, arg(1), arg(2));
        let callee = Callee::Indirect {
            callee: fp,
            fn_ty: fty,
        };
        let r = call(&mut k, i32t, callee, &[arg(2)]);
        store(&mut k, r, arg(0));
    }
    k.ret(None);
    let k = k.finish();
    let module = m.finish().unwrap();
    let mut abi = abi();
    abi.entries = vec![kernel(k, vec![out(0), table(1)])];
    (module, abi)
}

/// A linked module defining the visible `inc: i32 (i32)` (`x + 1`), `dbl: i32 (i32)` (`x * 2`),
/// and `neg: float (float)`, each a function a table can hold (words 2, 3, 4).
fn candidates() -> LinkedModule {
    let mut m = ModuleBuilder::new();
    let i32t = m.int(32);
    let f32t = m.float();
    let ity = m.fn_ty(i32t, &[i32t]);
    let fty = m.fn_ty(f32t, &[f32t]);
    let one = m.const_int(i32t, 1);
    let two = m.const_int(i32t, 2);
    let mut f = m.define("inc", ity);
    let v = f.bin(BinOp::Add, arg(0), one);
    f.ret(Some(v));
    let inc = f.finish();
    let mut f = m.define("dbl", ity);
    let v = f.bin(BinOp::Mul, arg(0), two);
    f.ret(Some(v));
    let dbl = f.finish();
    let mut f = m.define("neg", fty);
    f.ret(Some(arg(0)));
    let neg = f.finish();
    let module = m.finish().unwrap();
    let mut abi = abi();
    abi.visible_functions = [(inc, "inc"), (dbl, "dbl"), (neg, "neg")]
        .into_iter()
        .map(|(func, name)| VisibleFunction {
            func,
            params: Vec::new(),
            outputs: Vec::new(),
            stitching: None,
            name: DiagName::new(name),
        })
        .collect();
    LinkedModule {
        functions: ["inc", "dbl", "neg"]
            .into_iter()
            .zip(2..)
            .map(|(name, word)| LinkedName {
                name: name.as_bytes().into(),
                constants: Ok(BTreeMap::new()),
                handle: Some(word),
            })
            .collect(),
        code: Ok(LinkedCode { module, abi }),
    }
}

/// Devirtualization charges exactly the body it emits: the join of a table call that returns is a
/// PHI whose incoming list is charged once, by `Body::push_inst` as the PHI is pushed. The first
/// pass interns what the rewrite adds to the module (the word type, the default's zero), so the
/// second, over the same body, charges only its body.
#[test]
fn a_devirtualized_body_is_charged_exactly_its_footprint() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let ity = m.fn_ty(i32t, &[i32t]);
    let one = m.const_int(i32t, 1);
    let two = m.const_int(i32t, 2);
    let mut f = m.define("inc", ity);
    let v = f.bin(BinOp::Add, arg(0), one);
    f.ret(Some(v));
    let inc = f.finish();
    let mut f = m.define("dbl", ity);
    let v = f.bin(BinOp::Mul, arg(0), two);
    f.ret(Some(v));
    let dbl = f.finish();
    let decl = lookup_decl(&mut m);
    let kty = m.fn_ty(void, &[dev, dev, i32t]);
    let mut k = m.define("k", kty);
    let fp = lookup(&mut k, decl, arg(1), arg(2));
    let callee = Callee::Indirect {
        callee: fp,
        fn_ty: ity,
    };
    let r = call(&mut k, i32t, callee, &[arg(2)]);
    store(&mut k, r, arg(0));
    k.ret(None);
    let k = k.finish();
    let mut parts = m.finish().unwrap().into_parts();
    let body = parts
        .funcs
        .get(k)
        .and_then(|f| f.body.clone())
        .expect("k is defined");
    let candidates = [(2, "inc", inc), (3, "dbl", dbl)].map(|(word, name, f)| link::Candidate {
        word,
        name: name.as_bytes().into(),
        def: Ok(f),
    });
    let tables = [devirt::TableParam { arg: 1 }];
    let params = [dev, dev, i32t];
    let cx = devirt::Cx {
        tables: &tables,
        candidates: &candidates,
        linked: &[],
        params: &params,
        groups: &BTreeMap::new(),
        enters: &BTreeMap::new(),
        words: &devirt::FunctionWords::default(),
        fcs: &std::collections::HashMap::new(),
        name: &DiagName::new("k"),
    };
    let mut stubs = link::Stubs::default();
    let first = devirt::pass(&body, &mut parts, &mut stubs, &cx, &mut meter())
        .unwrap()
        .expect("the table call is rewritten");
    let mut ledger = meter();
    let pass = devirt::pass(&body, &mut parts, &mut stubs, &cx, &mut ledger)
        .unwrap()
        .expect("the table call is rewritten");
    assert_eq!(pass.body, first.body);
    let joins: Vec<usize> = pass
        .body
        .insts
        .values()
        .filter_map(|inst| match &inst.kind {
            InstKind::Phi { incoming, .. } => Some(incoming.len()),
            _ => None,
        })
        .collect();
    assert_eq!(joins, [3], "the join takes both arms and the default");
    assert_eq!(ledger.ledger(), pass.body.footprint() as u64);
}

fn refusal(r: Result<Entry, Error>) -> Unsupported {
    match r {
        Err(Error::Unsupported(u)) => u,
        Err(other) => panic!("{other}"),
        Ok(_) => panic!("specialized"),
    }
}

/// A call through a table is a switch over the candidates of its type, each inlined, with a
/// default; the table is read as a buffer of words.
#[test]
fn a_table_call_is_a_switch_over_the_candidates_of_its_type() {
    let (module, abi) = table_kernel(false);
    let e = specialize_linked(module, &abi, &request(), &[candidates()], &mut meter())
        .unwrap_or_else(|e| panic!("{e}"));
    let text = print(&e.module);
    let switch = text
        .lines()
        .find(|l| l.contains("switch "))
        .unwrap_or_else(|| panic!("no switch:\n{text}"));
    assert!(
        switch.contains(" 2: bb") && switch.contains(" 3: bb"),
        "{switch}"
    );
    assert!(
        !switch.contains(" 4: bb"),
        "another type is no case: {switch}"
    );
    assert!(!text.contains("call "), "every call is inlined:\n{text}");
    assert!(
        text.contains("add i32") && text.contains("mul i32"),
        "{text}"
    );
    assert!(matches!(
        e.ep.params.get(1).map(|p| &p.role),
        Some(ParamRole::Buffer(BufferArg {
            access: Access::Read,
            space: AddrSpace::Device,
            ..
        }))
    ));
}

/// A table call no candidate of its type answers is the empty switch, not a refusal: an entry
/// holds a handle, a handle exists only for a linked function, so with nothing linked the call is
/// unreachable and only the shader's own guards on the table run, as in Metal.
#[test]
fn a_table_call_without_candidates_is_the_empty_switch() {
    let (module, abi) = table_kernel(false);
    let e = specialize_linked(module, &abi, &request(), &[], &mut meter())
        .unwrap_or_else(|e| panic!("{e}"));
    let text = print(&e.module);
    let switch = text
        .lines()
        .find(|l| l.contains("switch "))
        .unwrap_or_else(|| panic!("no switch:\n{text}"));
    assert!(switch.trim().ends_with("[ ]"), "no case: {switch}");
    assert!(
        text.contains("store i32 zeroinitializer"),
        "the default answers: {text}"
    );
    let (module, abi) = table_kernel(true);
    assert!(specialize_linked(module, &abi, &request(), &[], &mut meter()).is_ok());
}

/// A candidate whose module cannot be imported is refused with its reason at a surviving call
/// only.
#[test]
fn an_unimportable_candidate_is_refused_at_a_surviving_call() {
    let broken = || LinkedModule {
        functions: vec![LinkedName {
            name: b"inc".as_slice().into(),
            constants: Ok(BTreeMap::new()),
            handle: Some(2),
        }],
        code: Err(Unsupported::not_implemented(Feature::Linking).into()),
    };
    let (module, abi) = table_kernel(false);
    let u = refusal(specialize_linked(
        module,
        &abi,
        &request(),
        &[broken()],
        &mut meter(),
    ));
    assert_eq!(u.feature, Feature::Linking);
    assert!(u.location.is_some());
    let (module, abi) = table_kernel(true);
    assert!(specialize_linked(module, &abi, &request(), &[broken()], &mut meter()).is_ok());
}

/// A candidate that is not imported but whose module decoded is a case only of calls of its
/// type: `neg: float (float)` unimportable leaves an `i32 (i32)` call alone, while `dbl: i32
/// (i32)` unimportable is refused with its reason at that call.
#[test]
fn an_unimportable_candidate_of_another_type_is_no_case() {
    let with_refused = |name: &str| {
        let mut linked = candidates();
        for f in &mut linked.functions {
            if &*f.name == name.as_bytes() {
                f.constants = Err(Unsupported::not_implemented(Feature::Linking).into());
            }
        }
        let (module, abi) = table_kernel(false);
        specialize_linked(module, &abi, &request(), &[linked], &mut meter())
    };
    let e = with_refused("neg").unwrap_or_else(|e| panic!("{e}"));
    let text = print(&e.module);
    assert!(
        text.contains("add i32") && text.contains("mul i32"),
        "{text}"
    );
    let u = refusal(with_refused("dbl"));
    assert_eq!(u.feature, Feature::Linking);
    assert!(u.location.is_some());
}

/// A candidate that calls through the table it is given calls itself: recursion, which the entry
/// runs on an explicit call stack, the candidate left in no call.
#[test]
fn a_candidate_calling_through_its_table_recurses_on_a_call_stack() {
    // Candidates: `rec: i32 (i32, ptr addrspace(1))` returning `t[0](x, t)`.
    let mut m = ModuleBuilder::new();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let rty = m.fn_ty(i32t, &[i32t, dev]);
    let decl = lookup_decl(&mut m);
    let zero = m.const_int(i32t, 0);
    let mut f = m.define("rec", rty);
    let fp = lookup(&mut f, decl, arg(1), zero);
    let r = call(
        &mut f,
        i32t,
        Callee::Indirect {
            callee: fp,
            fn_ty: rty,
        },
        &[arg(0), arg(1)],
    );
    f.ret(Some(r));
    let rec = f.finish();
    let lmodule = m.finish().unwrap();
    let mut labi = abi();
    labi.visible_functions = vec![VisibleFunction {
        func: rec,
        params: Vec::new(),
        outputs: Vec::new(),
        stitching: None,
        name: DiagName::new("rec"),
    }];
    let linked = LinkedModule {
        functions: vec![LinkedName {
            name: b"rec".as_slice().into(),
            constants: Ok(BTreeMap::new()),
            handle: Some(2),
        }],
        code: Ok(LinkedCode {
            module: lmodule,
            abi: labi,
        }),
    };
    // The kernel: `*out = t[tid](tid, t)`.
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let rty = m.fn_ty(i32t, &[i32t, dev]);
    let decl = lookup_decl(&mut m);
    let kty = m.fn_ty(void, &[dev, dev, i32t]);
    let mut k = m.define("k", kty);
    let fp = lookup(&mut k, decl, arg(1), arg(2));
    let r = call(
        &mut k,
        i32t,
        Callee::Indirect {
            callee: fp,
            fn_ty: rty,
        },
        &[arg(2), arg(1)],
    );
    store(&mut k, r, arg(0));
    k.ret(None);
    let kf = k.finish();
    let module = m.finish().unwrap();
    let mut abi = abi();
    abi.entries = vec![kernel(kf, vec![out(0), table(1)])];
    let e = specialize_linked(module, &abi, &request(), &[linked], &mut meter())
        .unwrap_or_else(|e| panic!("{e}"));
    let text = print(&e.module);
    assert_eq!(e.module.parts().funcs.len(), 1, "{text}");
    // Metal's byte model at depth 1: one linked function's frame, so one frame.
    assert!(text.contains("= type [1 x "), "{text}");
}

/// The kernel `k(device i32* out, table t, i32 tid)` storing `t[tid]` to `out`, behind `br false`
/// when `dead`.
fn store_kernel(dead: bool) -> (Module, AirAbi) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let decl = lookup_decl(&mut m);
    let no = m.const_int(i1, 0);
    let kty = m.fn_ty(void, &[dev, dev, i32t]);
    let mut k = m.define("k", kty);
    let done = k.block();
    if dead {
        let then = k.block();
        k.cond_br(no, then, done);
        k.switch_to(then);
    }
    let fp = lookup(&mut k, decl, arg(1), arg(2));
    store(&mut k, fp, arg(0));
    k.br(done);
    k.switch_to(done);
    k.ret(None);
    let kf = k.finish();
    let module = m.finish().unwrap();
    let mut abi = abi();
    abi.entries = vec![kernel(kf, vec![out(0), table(1)])];
    (module, abi)
}

/// A function pointer stored to memory is a dynamic call the translation does not implement:
/// refused at the store when it survives SCCP (at the refused call just before it, after the
/// four instructions of the word's load), and nothing when constants remove it.
#[test]
fn a_stored_function_pointer_is_a_dynamic_call_only_if_it_survives() {
    let (module, abi) = store_kernel(false);
    let u = refusal(specialize_linked(
        module,
        &abi,
        &request(),
        &[candidates()],
        &mut meter(),
    ));
    assert_eq!(u.feature, Feature::DynamicCall);
    assert_eq!(
        u.location,
        Some(Location::function(DiagName::new("k"), Some(0), Some(4)))
    );
    let (module, abi) = store_kernel(true);
    let e = specialize_linked(module, &abi, &request(), &[candidates()], &mut meter())
        .unwrap_or_else(|e| panic!("{e}"));
    let text = print(&e.module);
    assert!(!text.contains("store"), "{text}");
}

/// The kernel `k(device i32* out, table t, i32 tid)`: `*out = p(tid)` for `p` a `phi` of `t[tid]`
/// and a function pointer loaded from `out`, behind `br false` when `dead`.
fn phi_kernel(dead: bool) -> (Module, AirAbi) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let thread = m.ptr(AddrSpace::Thread);
    let fty = m.fn_ty(i32t, &[i32t]);
    let decl = lookup_decl(&mut m);
    let (no, zero) = (m.const_int(i1, 0), m.const_int(i32t, 0));
    let kty = m.fn_ty(void, &[dev, dev, i32t]);
    let mut k = m.define("k", kty);
    let (other, join, done) = (k.block(), k.block(), k.block());
    let here = if dead {
        let start = k.block();
        k.cond_br(no, start, done);
        k.switch_to(start);
        start
    } else {
        BlockId::new(0)
    };
    let fp = lookup(&mut k, decl, arg(1), arg(2));
    let c = k.push(
        i1,
        InstKind::ICmp {
            pred: IntPredicate::Eq,
            a: arg(2),
            b: zero,
        },
    );
    k.cond_br(c, other, join);
    k.switch_to(other);
    let loaded = k.push(
        thread,
        InstKind::Load {
            ptr: arg(0),
            align: Align::from_bytes(8),
            volatile: false,
            atomic: None,
            range: None,
        },
    );
    k.br(join);
    k.switch_to(join);
    let p = k.push(
        thread,
        InstKind::Phi {
            incoming: Box::new([(fp, here), (loaded, other)]),
            fmf: FastMath::NONE,
        },
    );
    let callee = Callee::Indirect {
        callee: p,
        fn_ty: fty,
    };
    let r = call(&mut k, i32t, callee, &[arg(2)]);
    store(&mut k, r, arg(0));
    k.br(done);
    k.switch_to(done);
    k.ret(None);
    let kf = k.finish();
    let module = m.finish().unwrap();
    let mut abi = abi();
    abi.entries = vec![kernel(kf, vec![out(0), table(1)])];
    (module, abi)
}

/// A `phi` of a table word and a pointer from memory is no word: the call through it is a
/// dynamic call when it survives (refused first after the block's `phi`), and nothing when
/// constants remove it.
#[test]
fn a_phi_of_a_word_and_a_loaded_pointer_is_a_dynamic_call_only_if_it_survives() {
    let (module, abi) = phi_kernel(false);
    let u = refusal(specialize_linked(
        module,
        &abi,
        &request(),
        &[candidates()],
        &mut meter(),
    ));
    assert_eq!(u.feature, Feature::DynamicCall);
    let (module, abi) = phi_kernel(true);
    assert!(specialize_linked(module, &abi, &request(), &[candidates()], &mut meter()).is_ok());
}

/// The kernel `k(device i32* out, table a, table b, i32 tid)`: `*out = f(g, tid)` for
/// `f = a[0]` of type `i32 (ptr, i32)` and `g = b[tid]`, looked up in that order or (`swapped`)
/// the other.
fn argument_kernel(swapped: bool) -> (Module, AirAbi) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let thread = m.ptr(AddrSpace::Thread);
    let aty = m.fn_ty(i32t, &[thread, i32t]);
    let decl = lookup_decl(&mut m);
    let zero = m.const_int(i32t, 0);
    let kty = m.fn_ty(void, &[dev, dev, dev, i32t]);
    let mut k = m.define("k", kty);
    let (f, g) = if swapped {
        let g = lookup(&mut k, decl, arg(2), arg(3));
        (lookup(&mut k, decl, arg(1), zero), g)
    } else {
        let f = lookup(&mut k, decl, arg(1), zero);
        (f, lookup(&mut k, decl, arg(2), arg(3)))
    };
    let callee = Callee::Indirect {
        callee: f,
        fn_ty: aty,
    };
    let r = call(&mut k, i32t, callee, &[g, arg(3)]);
    store(&mut k, r, arg(0));
    k.ret(None);
    let kf = k.finish();
    let module = m.finish().unwrap();
    let mut abi = abi();
    abi.entries = vec![kernel(kf, vec![out(0), table_at(1, 1), table_at(2, 2)])];
    (module, abi)
}

/// A linked module defining the visible `is_set: i32 (ptr, i32)`, which compares its function
/// pointer with null, and `inc: i32 (i32)` (words 2 and 3).
fn pointer_candidates() -> LinkedModule {
    let mut m = ModuleBuilder::new();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let thread = m.ptr(AddrSpace::Thread);
    let sty = m.fn_ty(i32t, &[thread, i32t]);
    let ity = m.fn_ty(i32t, &[i32t]);
    let null = m.constant(thread, Const::Null);
    let (one, two) = (m.const_int(i32t, 1), m.const_int(i32t, 2));
    let mut f = m.define("is_set", sty);
    let c = f.push(
        i1,
        InstKind::ICmp {
            pred: IntPredicate::Eq,
            a: arg(0),
            b: Value::Const(null),
        },
    );
    let v = f.push(
        i32t,
        InstKind::Select {
            c,
            t: one,
            f: two,
            fmf: FastMath::NONE,
        },
    );
    f.ret(Some(v));
    let is_set = f.finish();
    let mut f = m.define("inc", ity);
    let v = f.bin(BinOp::Add, arg(0), one);
    f.ret(Some(v));
    let inc = f.finish();
    let module = m.finish().unwrap();
    let mut abi = abi();
    abi.visible_functions = [(is_set, "is_set"), (inc, "inc")]
        .into_iter()
        .map(|(func, name)| VisibleFunction {
            func,
            params: Vec::new(),
            outputs: Vec::new(),
            stitching: None,
            name: DiagName::new(name),
        })
        .collect();
    LinkedModule {
        functions: ["is_set", "inc"]
            .into_iter()
            .zip(2..)
            .map(|(name, word)| LinkedName {
                name: name.as_bytes().into(),
                constants: Ok(BTreeMap::new()),
                handle: Some(word),
            })
            .collect(),
        code: Ok(LinkedCode { module, abi }),
    }
}

/// A function pointer passed as an argument, even to a call through a table, is a dynamic call
/// (refused at that call), whichever of the two lookups comes first.
#[test]
fn a_function_pointer_argument_is_a_dynamic_call_in_either_order() {
    let refused = |swapped| {
        let (module, abi) = argument_kernel(swapped);
        refusal(specialize_linked(
            module,
            &abi,
            &request(),
            &[pointer_candidates()],
            &mut meter(),
        ))
    };
    let (u, v) = (refused(false), refused(true));
    for u in [&u, &v] {
        assert_eq!(u.feature, Feature::DynamicCall);
    }
    // At the refused call before the table call, after the two loads (four instructions for
    // `b[tid]`, two for `a[0]` once SCCP folds its constant index).
    let at = Some(Location::function(DiagName::new("k"), Some(0), Some(6)));
    assert_eq!((u.location.clone(), v.location.clone()), (at.clone(), at));
}

/// `size()` and `is_null` alone read words 1 and 0: the table becomes a buffer, and no candidate
/// is needed.
#[test]
fn table_queries_read_words() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let sty = m.fn_ty(i32t, &[dev]);
    let nty = m.fn_ty(i1, &[dev]);
    let size = m.declare("air.get_size_visible_function_table", sty);
    let null = m.declare("air.is_null_visible_function_table", nty);
    let kty = m.fn_ty(void, &[dev, dev]);
    let mut k = m.define("k", kty);
    let op = AirIntrinsic::FunctionTable;
    let s = call(
        &mut k,
        i32t,
        Callee::Air {
            decl: size,
            op: op(FunctionTable::Size),
        },
        &[arg(1)],
    );
    let n = call(
        &mut k,
        i1,
        Callee::Air {
            decl: null,
            op: op(FunctionTable::IsNull),
        },
        &[arg(1)],
    );
    let wide = k.push(
        i32t,
        InstKind::Cast {
            op: crate::ir::CastOp::ZExt,
            v: n,
        },
    );
    let v = k.bin(BinOp::Add, s, wide);
    store(&mut k, v, arg(0));
    k.ret(None);
    let kf = k.finish();
    let module = m.finish().unwrap();
    let mut abi = abi();
    abi.entries = vec![kernel(kf, vec![out(0), table(1)])];
    let e = specialize_linked(module, &abi, &request(), &[], &mut meter())
        .unwrap_or_else(|e| panic!("{e}"));
    let text = print(&e.module);
    assert!(!text.contains("visible_function_table"), "{text}");
    assert_eq!(text.matches("load i32").count(), 2, "{text}");
    assert!(matches!(
        e.ep.params.get(1).map(|p| &p.role),
        Some(ParamRole::Buffer(_))
    ));
}

/// The kernel `k(device i32* out, i32 tid)`: `*out = h(&target, tid)`, where `h(fp, x)` returns
/// `fp(x)`; `target` is `f: i32 (i32)`, which with `recurse` returns `h(&f, x)` itself.
fn pointer_kernel(recurse: bool) -> (Module, AirAbi) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let thread = m.ptr(AddrSpace::Thread);
    let fty = m.fn_ty(i32t, &[i32t]);
    let hty = m.fn_ty(i32t, &[thread, i32t]);
    let one = m.const_int(i32t, 1);
    let mut h = m.define("h", hty);
    let r = call(
        &mut h,
        i32t,
        Callee::Indirect {
            callee: arg(0),
            fn_ty: fty,
        },
        &[arg(1)],
    );
    h.ret(Some(r));
    let h = h.finish();
    let mut f = m.define("f", fty);
    let fid = f.id();
    let v = if recurse {
        let faddr = f.module().constant(thread, Const::FuncAddr(fid));
        call(
            &mut f,
            i32t,
            Callee::Direct(h),
            &[Value::Const(faddr), arg(0)],
        )
    } else {
        f.bin(BinOp::Add, arg(0), one)
    };
    f.ret(Some(v));
    let fid = f.finish();
    let faddr = m.constant(thread, Const::FuncAddr(fid));
    let kty = m.fn_ty(void, &[dev, i32t]);
    let mut k = m.define("k", kty);
    let r = call(
        &mut k,
        i32t,
        Callee::Direct(h),
        &[Value::Const(faddr), arg(1)],
    );
    store(&mut k, r, arg(0));
    k.ret(None);
    let kf = k.finish();
    let module = m.finish().unwrap();
    let mut abi = abi();
    abi.entries = vec![kernel(kf, vec![out(0)])];
    (module, abi)
}

/// A call through a function's constant address at its own type, as a helper taking a function
/// pointer has once inlined, is a direct call and is inlined in turn. One that recurses through
/// its own address, passed to the helper as a value, recurses on a call stack: the helper's call
/// through its parameter may call `f`, whose address is taken, so `f` and the helper are a cycle;
/// the parameter is `f`'s address at every call, so each copy of the helper calls `f` itself.
/// Neither is linked, so the helper does not fold into `f`: two frames of 16 bytes at depth 1, both
/// of which Metal's model holds.
#[test]
fn a_call_through_a_constant_address_is_direct() {
    let (module, abi) = pointer_kernel(false);
    let e = specialize(module, &abi, &request(), &mut meter()).unwrap_or_else(|e| panic!("{e}"));
    let text = print(&e.module);
    assert!(!text.contains("call "), "{text}");
    assert!(text.contains("add i32"), "{text}");
    let (module, abi) = pointer_kernel(true);
    let e = specialize(module, &abi, &request(), &mut meter()).unwrap_or_else(|e| panic!("{e}"));
    let text = print(&e.module);
    assert_eq!(e.module.parts().funcs.len(), 1, "{text}");
    assert!(!text.contains("call "), "{text}");
    assert!(text.contains("= type [2 x "), "{text}");
}

/// The functions `a: i32 (i32)` (`x + 1`) and `b: i32 (i32)` (`x << 1`) of a kernel's own
/// module, neither visible (no table holds them), and their addresses.
fn chosen_functions(m: &mut ModuleBuilder) -> (Value, Value) {
    let i32t = m.int(32);
    let thread = m.ptr(AddrSpace::Thread);
    let fty = m.fn_ty(i32t, &[i32t]);
    let one = m.const_int(i32t, 1);
    let mut a = m.define("a", fty);
    let v = a.bin(BinOp::Add, arg(0), one);
    a.ret(Some(v));
    let a = a.finish();
    let mut b = m.define("b", fty);
    let v = b.bin(BinOp::Shl, arg(0), one);
    b.ret(Some(v));
    let b = b.finish();
    (
        Value::Const(m.constant(thread, Const::FuncAddr(a))),
        Value::Const(m.constant(thread, Const::FuncAddr(b))),
    )
}

/// `tid & 1 != 0` in `k`.
fn odd(k: &mut FunctionBuilder<'_>, tid: Value) -> Value {
    let i1 = k.module().int(1);
    let i32t = k.module().int(32);
    let (zero, one) = (k.module().const_int(i32t, 0), k.module().const_int(i32t, 1));
    let low = k.bin(BinOp::And, tid, one);
    k.push(
        i1,
        InstKind::ICmp {
            pred: IntPredicate::Ne,
            a: low,
            b: zero,
        },
    )
}

fn select(k: &mut FunctionBuilder<'_>, c: Value, t: Value, f: Value) -> Value {
    let thread = k.module().ptr(AddrSpace::Thread);
    k.push(
        thread,
        InstKind::Select {
            c,
            t,
            f,
            fmf: FastMath::NONE,
        },
    )
}

/// `kernel k(device i32* out, i32 tid)` with `body` computing `*out`.
fn chosen_kernel(
    body: impl FnOnce(&mut FunctionBuilder<'_>, (Value, Value)) -> Value,
) -> (Module, AirAbi) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let funcs = chosen_functions(&mut m);
    let kty = m.fn_ty(void, &[dev, i32t]);
    let mut k = m.define("k", kty);
    let r = body(&mut k, funcs);
    store(&mut k, r, arg(0));
    k.ret(None);
    let kf = k.finish();
    let module = m.finish().unwrap();
    let mut abi = abi();
    abi.entries = vec![kernel(kf, vec![out(0)])];
    (module, abi)
}

/// A call through `fp`, of type `i32 (i32)`, with `tid`.
fn call_through(k: &mut FunctionBuilder<'_>, fp: Value, tid: Value) -> Value {
    let i32t = k.module().int(32);
    let fty = k.module().fn_ty(i32t, &[i32t]);
    call(
        k,
        i32t,
        Callee::Indirect {
            callee: fp,
            fn_ty: fty,
        },
        &[tid],
    )
}

/// A call through a pointer chosen at run time among functions' addresses (a `select` of two
/// constant addresses, neither a table's entry) calls the function the pointer holds: each
/// address is carried as its function's word, and the call is a switch over the two functions,
/// each inlined (measured Metal's answers). It was the compute function-pointer row's
/// tripwire (`runtime::families::features`), refused as `DynamicCall` before.
#[test]
fn a_pointer_chosen_at_run_time_calls_the_function_it_holds() {
    let (module, abi) = chosen_kernel(|k, (a, b)| {
        let c = odd(k, arg(1));
        let fp = select(k, c, a, b);
        call_through(k, fp, arg(1))
    });
    let e = specialize(module, &abi, &request(), &mut meter()).unwrap_or_else(|e| panic!("{e}"));
    let text = print(&e.module);
    assert!(!text.contains("call "), "{text}");
    assert!(text.contains("add i32"), "{text}");
    assert!(text.contains("shl i32"), "{text}");
}

/// A function pointer stored to a thread's memory and loaded back is its word: the store writes
/// the word's 8 bytes and the load reads them back, so a call through the loaded pointer is a
/// switch over the functions whose address is taken at its type.
#[test]
fn a_function_pointer_kept_in_a_thread_s_memory_is_its_word() {
    let (module, abi) = chosen_kernel(|k, (a, b)| {
        let thread = k.module().ptr(AddrSpace::Thread);
        let i32t = k.module().int(32);
        let i64t = k.module().int(64);
        let pair = k.module().array(thread, 2);
        let one = k.module().const_int(i32t, 1);
        let (zero64, one64) = (k.module().const_int(i64t, 0), k.module().const_int(i64t, 1));
        let slots = k.push(
            thread,
            InstKind::Alloca {
                ty: pair,
                count: one,
                align: Align::from_bytes(8),
            },
        );
        let at = |k: &mut FunctionBuilder<'_>, i: Value| {
            k.push(
                thread,
                InstKind::Gep {
                    src_ty: pair,
                    base: slots,
                    indices: Box::new([zero64, i]),
                    inbounds: true,
                },
            )
        };
        let (first, second) = (at(k, zero64), at(k, one64));
        store(k, a, first);
        store(k, b, second);
        let low = k.bin(BinOp::And, arg(1), one);
        let wide = k.push(
            i64t,
            InstKind::Cast {
                op: CastOp::ZExt,
                v: low,
            },
        );
        let slot = at(k, wide);
        let fp = k.push(
            thread,
            InstKind::Load {
                ptr: slot,
                align: Align::from_bytes(8),
                volatile: false,
                atomic: None,
                range: None,
            },
        );
        call_through(k, fp, arg(1))
    });
    let e = specialize(module, &abi, &request(), &mut meter()).unwrap_or_else(|e| panic!("{e}"));
    let text = print(&e.module);
    assert!(!text.contains("call "), "{text}");
    assert!(text.contains("store i64"), "{text}");
    assert!(text.contains("load i64"), "{text}");
    assert!(
        text.contains("add i32") && text.contains("shl i32"),
        "{text}"
    );
}

/// Function pointers compared for equality compare their words: a chosen pointer with a
/// function's address leaves no pointer comparison.
#[test]
fn function_pointers_compare_as_their_words() {
    let (module, abi) = chosen_kernel(|k, (a, b)| {
        let i1 = k.module().int(1);
        let i32t = k.module().int(32);
        let c = odd(k, arg(1));
        let fp = select(k, c, a, b);
        let same = k.push(
            i1,
            InstKind::ICmp {
                pred: IntPredicate::Eq,
                a: fp,
                b: a,
            },
        );
        k.push(
            i32t,
            InstKind::Cast {
                op: CastOp::ZExt,
                v: same,
            },
        )
    });
    let e = specialize(module, &abi, &request(), &mut meter()).unwrap_or_else(|e| panic!("{e}"));
    let text = print(&e.module);
    assert!(!text.contains("icmp eq ptr"), "{text}");
    assert!(!text.contains("call "), "{text}");
}

/// The address of a visible function no linked function resolves is refused as a call to it
/// is, `NeedsInput(LinkedFunction)`, where it survives, even when nothing calls through it:
/// Metal refuses to make such a pipeline ("unresolved visible function reference").
#[test]
fn an_unresolved_reference_s_address_needs_its_linked_function() {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let thread = m.ptr(AddrSpace::Thread);
    let fty = m.fn_ty(i32t, &[i32t]);
    let (a, _) = chosen_functions(&mut m);
    let stub = m.declare("f.MTL_VISIBLE_FN_REF", fty);
    let f = Value::Const(m.constant(thread, Const::FuncAddr(stub)));
    let kty = m.fn_ty(void, &[dev, i32t]);
    let mut k = m.define("k", kty);
    let c = odd(&mut k, arg(1));
    let fp = select(&mut k, c, a, f);
    let same = k.push(
        i1,
        InstKind::ICmp {
            pred: IntPredicate::Eq,
            a: fp,
            b: a,
        },
    );
    let r = k.push(
        i32t,
        InstKind::Cast {
            op: CastOp::ZExt,
            v: same,
        },
    );
    store(&mut k, r, arg(0));
    k.ret(None);
    let kf = k.finish();
    let module = m.finish().unwrap();
    let mut abi = abi();
    abi.entries = vec![kernel(kf, vec![out(0)])];
    abi.visible_function_refs = vec![crate::abi::VisibleFunctionRef {
        target: crate::abi::VisibleTarget::Function(stub),
        symbol: crate::abi::VisibleSymbol(b"f".as_slice().into()),
    }];
    match specialize(module, &abi, &request(), &mut meter()) {
        Err(Error::NeedsInput(MissingInput::LinkedFunction { symbol })) => {
            assert_eq!(symbol, DiagName::new("f"));
        }
        Err(e) => panic!("{e}"),
        Ok(_) => panic!("specialized"),
    }
}

/// A pointer loaded from a thread's memory when no function of the call's type has its address
/// taken and no table can hold one holds no function of the stage: the call through it is
/// `DynamicCall` (Metal's answer is no defined one).
#[test]
fn a_loaded_pointer_no_function_can_answer_is_a_dynamic_call() {
    let (module, abi) = chosen_kernel(|k, _| {
        let thread = k.module().ptr(AddrSpace::Thread);
        let i32t = k.module().int(32);
        let one = k.module().const_int(i32t, 1);
        let null = Value::Const(k.module().constant(thread, Const::Null));
        let slot = k.push(
            thread,
            InstKind::Alloca {
                ty: thread,
                count: one,
                align: Align::from_bytes(8),
            },
        );
        store(k, null, slot);
        let fp = k.push(
            thread,
            InstKind::Load {
                ptr: slot,
                align: Align::from_bytes(8),
                volatile: false,
                atomic: None,
                range: None,
            },
        );
        call_through(k, fp, arg(1))
    });
    let u = refusal(specialize(module, &abi, &request(), &mut meter()));
    assert_eq!(u.feature, Feature::DynamicCall);
}

/// `kernel k(device i32* out, i32 tid)`: for an odd `tid`, `*out` is a call with `tid` through
/// `mode == 1 ? &a : &b` ([`chosen_functions`]), `mode` function constant 3. The call is past the
/// entry block, so the entry block's own refusal does not decide an unsupplied `mode` before
/// inlining, as in a kernel of the corpus that chooses among 162 functions by its constants, behind
/// a null check.
fn constant_chosen_kernel() -> (Module, AirAbi, GlobalId) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let mode = super::tests::fc_global(&mut m, "mode", i32t);
    let mode_addr = m.global_addr(mode);
    let one = m.const_int(i32t, 1);
    let (a, b) = chosen_functions(&mut m);
    let kty = m.fn_ty(void, &[dev, i32t]);
    let mut k = m.define("k", kty);
    let (then, exit) = (k.block(), k.block());
    let v = k.push(
        i32t,
        InstKind::Load {
            ptr: mode_addr,
            align: Align::from_bytes(4),
            volatile: false,
            atomic: None,
            range: None,
        },
    );
    let is = k.push(
        i1,
        InstKind::ICmp {
            pred: IntPredicate::Eq,
            a: v,
            b: one,
        },
    );
    let fp = select(&mut k, is, a, b);
    let c = odd(&mut k, arg(1));
    k.cond_br(c, then, exit);
    k.switch_to(then);
    let r = call_through(&mut k, fp, arg(1));
    store(&mut k, r, arg(0));
    k.br(exit);
    k.switch_to(exit);
    k.ret(None);
    let kf = k.finish();
    let module = m.finish().unwrap();
    let mut abi = abi();
    abi.entries = vec![kernel(kf, vec![out(0)])];
    abi.function_constants = vec![super::tests::fc_decl(3, mode, i32t)];
    (module, abi, mode)
}

/// A call through a pointer function constants choose is decided before it becomes a switch: with
/// the constant supplied, a direct call to the function chosen, the only one
/// inlined; unsupplied, the call is left as it was before item 27, so SCCP refuses the constant
/// where the call stands and no switch inlines every function the pointer might be. The corpus's
/// kernel choosing among 162 functions so inlined all of them, 2.5 times its memory, to refuse the
/// same constant.
#[test]
fn a_pointer_function_constants_choose_is_decided_before_its_switch() {
    for supplied in [true, false] {
        let (module, abi, mode) = constant_chosen_kernel();
        let k = abi.entries[0].func;
        let mut parts = module.into_parts();
        let mut m = meter();
        if supplied {
            // As `fc::bind` binds a supplied constant: its value is the global's initializer.
            let i32t = parts.types.intern(Type::Int(32), &mut m).unwrap();
            let one = crate::ir::Constant {
                ty: i32t,
                value: Const::Int(1),
            };
            let one = parts.consts.intern(one, &mut m).unwrap();
            let g = parts.globals.get_mut(mode).expect("mode is a global");
            g.init = Some(one);
            g.externally_initialized = false;
        }
        let fcs = std::collections::HashMap::from([(
            mode,
            sccp::FcState {
                index: 3,
                supplied,
                decl: true,
            },
        )]);
        let funcs = |name: &[u8]| {
            parts
                .funcs
                .iter()
                .find(|(_, f)| parts.symbols.get(f.name) == Some(name))
                .map(|(id, _)| id)
                .expect("defined")
        };
        let (a, b) = (funcs(b"a"), funcs(b"b"));
        let addressed = BTreeSet::from([a, b]);
        let words = devirt::FunctionWords::new(&parts, &[], &addressed, 2, &mut m).unwrap();
        let params = match parts.funcs.get(k).and_then(|f| parts.types.get(f.ty)) {
            Some(Type::Function { params, .. }) => params.to_vec(),
            _ => panic!("k is a function"),
        };
        let body = parts
            .funcs
            .get(k)
            .and_then(|f| f.body.clone())
            .expect("k is defined");
        let cx = devirt::Cx {
            tables: &[],
            candidates: &[],
            linked: &[],
            params: &params,
            groups: &BTreeMap::new(),
            enters: &BTreeMap::new(),
            words: &words,
            fcs: &fcs,
            name: &DiagName::new("k"),
        };
        let mut stubs = link::Stubs::default();
        let pass = devirt::pass(&body, &mut parts, &mut stubs, &cx, &mut m).unwrap();
        if !supplied {
            assert!(
                pass.is_none(),
                "an unsupplied constant's call is left as it was"
            );
            continue;
        }
        let pass = pass.expect("the supplied constant's call is decided");
        assert!(pass.calls);
        let mut switches = 0;
        let mut direct = Vec::new();
        for inst in pass.body.insts.values() {
            match &inst.kind {
                InstKind::Switch { .. } => switches += 1,
                InstKind::Call(c) => match c.callee {
                    Callee::Direct(f) => direct.push(f),
                    _ => panic!("a call that is not direct"),
                },
                _ => {}
            }
        }
        assert_eq!(
            switches, 0,
            "no switch over the functions the pointer might be"
        );
        assert_eq!(direct, vec![a], "the call is the chosen function's");
    }
    // Through the whole specialization: unsupplied, the constant is refused, as before item 27;
    // supplied as 1, the call is `a`'s (`x + 1`), and `b`'s `x << 1` is not inlined.
    let (module, abi, _) = constant_chosen_kernel();
    let refused = specialize(module, &abi, &request(), &mut meter()).map(|_| ());
    assert!(
        matches!(
            refused,
            Err(Error::NeedsInput(MissingInput::FunctionConstant {
                index: 3
            }))
        ),
        "{refused:?}"
    );
    let (module, abi, _) = constant_chosen_kernel();
    let mut req = request();
    req.constants.insert(
        3,
        ConstValue {
            ty: crate::metal::DataType::Int,
            bytes: 1u32.to_le_bytes().into(),
        },
    );
    let e = specialize(module, &abi, &req, &mut meter()).unwrap_or_else(|e| panic!("{e}"));
    let text = print(&e.module);
    assert!(text.contains("add i32"), "{text}");
    assert!(
        !text.contains("shl i32") && !text.contains("call "),
        "{text}"
    );
}

/// Inlining a candidate at every table call is charged like any inlining: over the cap it is
/// `LimitExceeded(InlinedInstructions)`, decided before the body is built.
#[test]
fn inlining_candidates_at_every_table_call_is_capped() {
    // A candidate of 2^16 instructions, called through the table at 40 sites.
    let mut m = ModuleBuilder::new();
    let i32t = m.int(32);
    let ity = m.fn_ty(i32t, &[i32t]);
    let one = m.const_int(i32t, 1);
    let mut f = m.define("big", ity);
    let mut v = arg(0);
    for _ in 0..(1u32 << 16) {
        v = f.bin(BinOp::Add, v, one);
    }
    f.ret(Some(v));
    let big = f.finish();
    let lmodule = m.finish().unwrap();
    let mut labi = abi();
    labi.visible_functions = vec![VisibleFunction {
        func: big,
        params: Vec::new(),
        outputs: Vec::new(),
        stitching: None,
        name: DiagName::new("big"),
    }];
    let linked = LinkedModule {
        functions: vec![LinkedName {
            name: b"big".as_slice().into(),
            constants: Ok(BTreeMap::new()),
            handle: Some(2),
        }],
        code: Ok(LinkedCode {
            module: lmodule,
            abi: labi,
        }),
    };
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let ity = m.fn_ty(i32t, &[i32t]);
    let decl = lookup_decl(&mut m);
    let kty = m.fn_ty(void, &[dev, dev, i32t]);
    let mut k = m.define("k", kty);
    let mut r = arg(2);
    for _ in 0..40 {
        let fp = lookup(&mut k, decl, arg(1), arg(2));
        let callee = Callee::Indirect {
            callee: fp,
            fn_ty: ity,
        };
        r = call(&mut k, i32t, callee, &[r]);
    }
    store(&mut k, r, arg(0));
    k.ret(None);
    let kf = k.finish();
    let module = m.finish().unwrap();
    let mut abi = abi();
    abi.entries = vec![kernel(kf, vec![out(0), table(1)])];
    assert_eq!(
        specialize_linked(module, &abi, &request(), &[linked], &mut meter()).map(|_| ()),
        Err(Error::LimitExceeded(
            crate::error::Limit::InlinedInstructions
        ))
    );
}

/// How [`argument_buffer_kernel`] uses the tables it loads.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TableUse {
    /// `t[tid](tid)`, `size(t)`, and `is_null(t)` of the member `t`.
    Calls,
    /// The same of `tid == 0 ? t : u`, a `select` of the members `t` and `u`, or of `t` and the
    /// null table (`null`).
    Select { null: bool },
    /// `size(t)` of the member `t`, and `t` stored to `out`.
    Stored,
    /// `size(t)` of the member `t`, and a store to the address `t` (`inttoptr` of its id is no
    /// address: the store's address is the table), behind `br false` when `dead`.
    Escaped { dead: bool },
}

/// The kernel `k(device i32* out, device Tables* args, i32 tid)` whose argument buffer holds the
/// table members `t` (offset 0) and `u` (offset 8), loaded from `args`; each result is stored to
/// `out`.
fn argument_buffer_kernel(how: TableUse) -> (Module, AirAbi) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i8t = m.int(8);
    let i32t = m.int(32);
    let i64t = m.int(64);
    let dev = m.ptr(AddrSpace::Device);
    let fty = m.fn_ty(i32t, &[i32t]);
    let decl = lookup_decl(&mut m);
    let size_ty = m.fn_ty(i32t, &[dev]);
    let size = m.declare("air.get_size_visible_function_table", size_ty);
    let null_ty = m.fn_ty(i1, &[dev]);
    let is_null = m.declare("air.is_null_visible_function_table", null_ty);
    let (no, zero, eight) = (
        m.const_int(i1, 0),
        m.const_int(i32t, 0),
        m.const_int(i64t, 8),
    );
    let null = Value::Const(m.constant(dev, Const::Null));
    let kty = m.fn_ty(void, &[dev, dev, i32t]);
    let mut k = m.define("k", kty);
    let load = |k: &mut FunctionBuilder<'_>, ptr| {
        k.push(
            dev,
            InstKind::Load {
                ptr,
                align: Align::from_bytes(8),
                volatile: false,
                atomic: None,
                range: None,
            },
        )
    };
    let done = k.block();
    if how == (TableUse::Escaped { dead: true }) {
        let then = k.block();
        k.cond_br(no, then, done);
        k.switch_to(then);
    }
    let t = load(&mut k, arg(1));
    let table = match how {
        TableUse::Calls | TableUse::Stored | TableUse::Escaped { .. } => t,
        TableUse::Select { null: other_null } => {
            let other = match other_null {
                true => null,
                false => {
                    let at = k.push(
                        dev,
                        InstKind::Gep {
                            src_ty: i8t,
                            base: arg(1),
                            indices: Box::new([eight]),
                            inbounds: true,
                        },
                    );
                    load(&mut k, at)
                }
            };
            let c = k.push(
                i1,
                InstKind::ICmp {
                    pred: IntPredicate::Eq,
                    a: arg(2),
                    b: zero,
                },
            );
            k.push(
                dev,
                InstKind::Select {
                    c,
                    t: other,
                    f: t,
                    fmf: FastMath::NONE,
                },
            )
        }
    };
    let size_of = |k: &mut FunctionBuilder<'_>, table| {
        let op = AirIntrinsic::FunctionTable(FunctionTable::Size);
        let n = call(k, i32t, Callee::Air { decl: size, op }, &[table]);
        store(k, n, arg(0));
    };
    match how {
        TableUse::Stored => {
            size_of(&mut k, table);
            store(&mut k, table, arg(0));
        }
        TableUse::Escaped { .. } => {
            size_of(&mut k, table);
            store(&mut k, arg(2), table);
        }
        TableUse::Calls | TableUse::Select { .. } => {
            let fp = lookup(&mut k, decl, table, arg(2));
            let callee = Callee::Indirect {
                callee: fp,
                fn_ty: fty,
            };
            let r = call(&mut k, i32t, callee, &[arg(2)]);
            store(&mut k, r, arg(0));
            size_of(&mut k, table);
            let op = AirIntrinsic::FunctionTable(FunctionTable::IsNull);
            let b = call(&mut k, i1, Callee::Air { decl: is_null, op }, &[table]);
            let w = k.push(
                i32t,
                InstKind::Cast {
                    op: crate::ir::CastOp::ZExt,
                    v: b,
                },
            );
            store(&mut k, w, arg(0));
        }
    }
    k.br(done);
    k.switch_to(done);
    k.ret(None);
    let kf = k.finish();
    let module = m.finish().unwrap();
    let mut abi = abi();
    let member = |offset: u32, id: u32| Member {
        offset,
        size: 8,
        array_len: 0,
        layout: None,
        kind: MemberKind::Indirect(IndirectArg {
            ordinal: id,
            role: ParamRole::FunctionTable(ObjectArg {
                location: slot(id),
                access: Some(Access::Read),
            }),
            ty: ArgTypeInfo::default(),
            name: None,
        }),
        render_target: None,
        raster_order_group: None,
        type_name: DiagName::new("visible_function_table"),
        name: DiagName::new("t"),
        key: UserKey(b"t".as_slice().into()),
    };
    abi.structs = vec![StructInfo {
        members: vec![member(0, 0), member(8, 1)],
    }];
    let args = binding(
        1,
        ParamRole::ArgumentBuffer(BufferArg {
            location: slot(1),
            access: Access::Read,
            space: AddrSpace::Device,
            buffer_size: Some(16),
            layout: Some(StructRef(0)),
            raster_order_group: None,
        }),
    );
    abi.entries = vec![kernel(kf, vec![out(0), args])];
    (module, abi)
}

/// The calls of `text` to the table word read ([`FunctionTable::Word`]).
fn word_reads(text: &str) -> usize {
    text.lines()
        .filter(|l| l.contains("call ") && l.contains("FunctionTable(Word)"))
        .count()
}

/// A table loaded from an argument buffer is carried as its `i64` resource id: its load reads an
/// `i64`, and its size, null test, and entry each read a word of the table by that id, the entry's
/// word dispatching over the candidates of the call's type as a table parameter's does. The
/// argument buffer stays an argument buffer.
#[test]
fn a_table_loaded_from_memory_reads_its_words_by_its_id() {
    let (module, abi) = argument_buffer_kernel(TableUse::Calls);
    assert!(abi.reaches_function_tables(&abi.entries[0]));
    let e = specialize_linked(module, &abi, &request(), &[candidates()], &mut meter())
        .unwrap_or_else(|e| panic!("{e}"));
    let text = print(&e.module);
    assert!(text.contains("load i64"), "{text}");
    assert_eq!(word_reads(&text), 3, "{text}");
    let switch = text
        .lines()
        .find(|l| l.contains("switch "))
        .unwrap_or_else(|| panic!("no switch:\n{text}"));
    assert!(
        switch.contains(" 2: bb") && switch.contains(" 3: bb") && !switch.contains(" 4: bb"),
        "{switch}"
    );
    assert!(matches!(
        e.ep.params.get(1).map(|p| &p.role),
        Some(ParamRole::ArgumentBuffer(_))
    ));
}

/// A `select` of two loaded tables, or of a loaded table and the null table, is carried as a
/// `select` of their ids (the null table's id 0), whose words are read once per operation.
#[test]
fn a_select_of_loaded_tables_selects_their_ids() {
    for null in [false, true] {
        let (module, abi) = argument_buffer_kernel(TableUse::Select { null });
        let e = specialize_linked(module, &abi, &request(), &[candidates()], &mut meter())
            .unwrap_or_else(|e| panic!("{null}: {e}"));
        let text = print(&e.module);
        assert!(
            text.lines()
                .any(|l| l.contains("select ") && l.contains("i64")),
            "{null}: {text}"
        );
        assert_eq!(word_reads(&text), 3, "{null}: {text}");
    }
}

/// A call through a loaded table that no candidate of its type answers is the empty switch too,
/// the same answer a table parameter's call gets.
#[test]
fn a_loaded_table_call_without_candidates_is_the_empty_switch() {
    let (module, abi) = argument_buffer_kernel(TableUse::Calls);
    let e = specialize_linked(module, &abi, &request(), &[], &mut meter())
        .unwrap_or_else(|e| panic!("{e}"));
    let text = print(&e.module);
    let switch = text
        .lines()
        .find(|l| l.contains("switch "))
        .unwrap_or_else(|| panic!("no switch:\n{text}"));
    assert!(switch.trim().ends_with("[ ]"), "no case: {switch}");
    assert!(
        text.contains("store i32 zeroinitializer"),
        "the default answers: {text}"
    );
}

/// A loaded table a table operation reads and a store copies into memory: the store stores its
/// `i64` id, the 8 bytes Metal stores.
#[test]
fn a_stored_table_stores_its_id() {
    let (module, abi) = argument_buffer_kernel(TableUse::Stored);
    let e = specialize_linked(module, &abi, &request(), &[candidates()], &mut meter())
        .unwrap_or_else(|e| panic!("{e}"));
    let text = print(&e.module);
    assert_eq!(word_reads(&text), 1, "{text}");
    let id = text
        .lines()
        .find_map(|l| l.trim().strip_suffix(" = load i64, %a1, align 8"))
        .unwrap_or_else(|| panic!("no id load:\n{text}"));
    assert!(
        text.lines()
            .any(|l| l.trim() == format!("store {id}, %a0, align 4")),
        "{text}"
    );
}

/// A loaded table used as no table operation, a store, or a carrier uses it (here the address
/// of a store) is a dynamic call the translation does not implement when the use survives, and
/// nothing when constants remove it.
#[test]
fn an_escaped_table_is_a_dynamic_call_only_if_it_survives() {
    let (module, abi) = argument_buffer_kernel(TableUse::Escaped { dead: false });
    let u = refusal(specialize_linked(
        module,
        &abi,
        &request(),
        &[candidates()],
        &mut meter(),
    ));
    assert_eq!(u.feature, Feature::DynamicCall);
    let (module, abi) = argument_buffer_kernel(TableUse::Escaped { dead: true });
    assert!(specialize_linked(module, &abi, &request(), &[candidates()], &mut meter()).is_ok());
}
