//! Linking tests over IR built with the typed test builder: a kernel calls visible functions
//! through reference stubs, and linked modules define them.

use std::collections::BTreeMap;

use super::*;
use crate::abi::{
    AbiInt, Access, AirAbi, ArgTypeInfo, BufferArg, CompileOptions, EntryPoint, ModuleLimits,
    ParamBinding, ParamRole, Slot, Stage, VisibleFunction, VisibleFunctionRef, VisibleSymbol,
    VisibleTarget,
};
use crate::budget::{Budget, Meter};
use crate::error::LinkProblem;
use crate::ir::build::{FunctionBuilder, ModuleBuilder};
use crate::ir::print::print;
use crate::ir::{
    AddrSpace, Align, BinOp, Call, Const, ConstId, FastMath, Global, Linkage, Section, TailKind,
    Value,
};
use crate::metal::DataType;

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

fn reference(stub: FuncId, symbol: &str) -> VisibleFunctionRef {
    VisibleFunctionRef {
        target: VisibleTarget::Function(stub),
        symbol: VisibleSymbol(symbol.as_bytes().into()),
    }
}

fn visible(func: FuncId, name: &str) -> VisibleFunction {
    VisibleFunction {
        func,
        params: Vec::new(),
        outputs: Vec::new(),
        stitching: None,
        name: DiagName::new(name),
    }
}

fn call(f: &mut FunctionBuilder<'_>, ty: TypeId, callee: FuncId, args: &[Value]) -> Value {
    f.push(
        ty,
        InstKind::Call(Box::new(Call {
            callee: Callee::Direct(callee),
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

/// `kernel k(device i32* out, i32 tid)`: `*out = f(tid)` through the stub of `f`, an `i32
/// (i32)` visible function it references. With `dead`, the call sits behind `br false`.
fn caller(dead: bool) -> (Module, AirAbi) {
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let fty = m.fn_ty(i32t, &[i32t]);
    let stub = m.declare("f.MTL_VISIBLE_FN_REF", fty);
    let kty = m.fn_ty(void, &[dev, i32t]);
    let no = m.const_int(i1, 0);
    let mut k = m.define("k", kty);
    if dead {
        let then = k.block();
        let done = k.block();
        k.cond_br(no, then, done);
        k.switch_to(then);
        let v = call(&mut k, i32t, stub, &[arg(1)]);
        store(&mut k, v, arg(0));
        k.br(done);
        k.switch_to(done);
    } else {
        let v = call(&mut k, i32t, stub, &[arg(1)]);
        store(&mut k, v, arg(0));
    }
    k.ret(None);
    let k = k.finish();
    let module = m.finish().unwrap();
    let mut abi = abi();
    abi.entries = vec![EntryPoint {
        func: k,
        stage: Stage::Kernel {
            max_total_threads: None,
        },
        params: vec![ParamBinding {
            param: 0,
            role: ParamRole::Buffer(BufferArg {
                location: Slot {
                    index: AbiInt::Const(0),
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
        }],
        outputs: Vec::new(),
        name: DiagName::new("k"),
        user_annotation: None,
    }];
    abi.visible_function_refs = vec![reference(stub, "f")];
    (module, abi)
}

/// A linked module whose visible function `name` is `i32 (i32 x)`: `x * 3 + table[x & 3]`, with
/// `table` a constant global of the module; it also calls, through its own stubs, each of
/// `calls` (an `i32 (i32)` visible function of another module) on its result, in order.
fn callee(name: &str, calls: &[&str]) -> LinkedModule {
    let mut m = ModuleBuilder::new();
    let i32t = m.int(32);
    let fty = m.fn_ty(i32t, &[i32t]);
    let arr = m.array(i32t, 4);
    let words: Vec<ConstId> = [5u64, 6, 7, 8]
        .iter()
        .map(|&v| m.constant(i32t, Const::Int(v)))
        .collect();
    let init = m.constant(arr, Const::Aggregate(words.into()));
    let table_name = m.sym("table");
    let table = m.global(Global {
        name: table_name,
        value_ty: arr,
        space: AddrSpace::Constant,
        constant: true,
        init: Some(init),
        externally_initialized: false,
        visibility: Default::default(),
        unnamed_addr: Default::default(),
        linkage: Linkage::Internal,
        align: None,
        section: None,
    });
    let stubs: Vec<(FuncId, &str)> = calls
        .iter()
        .map(|&c| (m.declare(&format!("{c}.MTL_VISIBLE_FN_REF"), fty), c))
        .collect();
    let three = m.const_int(i32t, 3);
    let table = m.global_addr(table);
    let zero64 = {
        let i64t = m.int(64);
        m.const_int(i64t, 0)
    };
    let mut f = m.define(name, fty);
    let x3 = f.bin(BinOp::Mul, arg(0), three);
    let i = f.bin(BinOp::And, arg(0), three);
    let i64t = f.module().int(64);
    let idx = f.push(
        i64t,
        InstKind::Cast {
            op: crate::ir::CastOp::ZExt,
            v: i,
        },
    );
    let ptr = f.module().ptr(AddrSpace::Constant);
    let at = f.push(
        ptr,
        InstKind::Gep {
            src_ty: arr,
            base: table,
            indices: vec![zero64, idx].into(),
            inbounds: true,
        },
    );
    let word = f.push(
        i32t,
        InstKind::Load {
            ptr: at,
            align: Align::from_bytes(4),
            volatile: false,
            atomic: None,
            range: None,
        },
    );
    let mut v = f.bin(BinOp::Add, x3, word);
    for &(stub, _) in &stubs {
        v = call(&mut f, i32t, stub, &[v]);
    }
    f.ret(Some(v));
    let func = f.finish();
    let module = m.finish().unwrap();
    let mut abi = abi();
    abi.visible_functions = vec![visible(func, name)];
    abi.visible_function_refs = stubs.iter().map(|&(s, c)| reference(s, c)).collect();
    linked(name, module, abi)
}

/// The linked module `module` (with `abi`) linking its function `name` with no constants.
fn linked(name: &str, module: Module, abi: AirAbi) -> LinkedModule {
    LinkedModule {
        functions: vec![LinkedName {
            name: name.as_bytes().into(),
            function: name.as_bytes().into(),
            specialized: false,
            constants: Ok(BTreeMap::new()),
            handle: None,
            resolves_references: true,
        }],
        code: Ok(LinkedCode { module, abi }),
    }
}

/// The decoded module of `l`.
fn code(l: &mut LinkedModule) -> &mut LinkedCode {
    l.code.as_mut().expect("a decoded linked module")
}

fn request() -> SpecRequest {
    SpecRequest {
        call_stack_depth: 1,
        entry: "k".into(),
        constants: BTreeMap::new(),
    }
}

fn link(linked: &[LinkedModule]) -> Result<Entry, Error> {
    let (module, abi) = caller(false);
    specialize_linked(module, &abi, &request(), linked, &mut meter())
}

fn text(e: &Entry) -> String {
    print(&e.module)
}

/// The kernel's one function after linking: no call survives, and the linked body (its multiply,
/// its table load from the imported global) is inlined.
#[test]
fn a_referenced_linked_function_is_imported_and_inlined() {
    let e = link(&[callee("f", &[])]).unwrap();
    let t = text(&e);
    let defined = e.module.parts().funcs.values().filter(|f| f.body.is_some());
    assert_eq!(defined.count(), 1, "{t}");
    assert!(!t.contains("call "), "{t}");
    assert!(t.contains("mul i32"), "{t}");
    // The imported table, with its initializer.
    assert!(t.contains("[5, 6, 7, 8]"), "{t}");
    assert!(
        t.contains("\"table\" = internal addrspace(2) constant"),
        "{t}"
    );
}

/// A linked function's own references are resolved against the same linked functions.
#[test]
fn references_of_linked_functions_are_resolved_in_turn() {
    let e = link(&[callee("f", &["g"]), callee("g", &[])]).unwrap();
    let t = text(&e);
    assert!(!t.contains("call "), "{t}");
    assert_eq!(t.matches("mul i32").count(), 2, "{t}");
}

/// Linked functions that call each other are a call cycle, which the entry runs on an explicit
/// call stack.
#[test]
fn a_cycle_through_linked_functions_recurses_on_a_call_stack() {
    let e = link(&[callee("f", &["g"]), callee("g", &["f"])]).unwrap_or_else(|e| panic!("{e}"));
    let t = text(&e);
    assert_eq!(e.module.parts().funcs.len(), 1, "{t}");
    // Metal's byte model at depth 1: the two linked functions' 16-byte frames, so two frames,
    // the root's activation and one call.
    assert!(t.contains("= type [2 x "), "{t}");
}

/// A surviving call to a reference no linked function resolves needs that function, by name.
#[test]
fn an_unresolved_reference_needs_the_linked_function() {
    let want = Error::NeedsInput(MissingInput::LinkedFunction {
        symbol: DiagName::new("f"),
    });
    assert_eq!(link(&[]).unwrap_err(), want);
    assert_eq!(link(&[callee("other", &[])]).unwrap_err(), want);
    // Through a linked function: its reference to `g` has no linked function.
    let nested = Error::NeedsInput(MissingInput::LinkedFunction {
        symbol: DiagName::new("g"),
    });
    assert_eq!(link(&[callee("f", &["g"])]).unwrap_err(), nested);
}

/// A function only `binaryFunctions` lists resolves no reference, the entry's or a linked
/// function's, as Metal's creation fails over a reference only such a function names
/// ("unresolved visible function reference", a probe on an M3 Max under macOS 27.0.1,
/// `binary_function_linking.m`); beside a function of its name that resolves references, the
/// reference resolves to that one, with no ambiguity.
#[test]
fn a_function_only_binary_functions_lists_resolves_no_reference() {
    let binary = |mut l: LinkedModule| {
        for f in &mut l.functions {
            f.resolves_references = false;
            f.handle = Some(2);
        }
        l
    };
    let missing = |symbol: &str| {
        Error::NeedsInput(MissingInput::LinkedFunction {
            symbol: DiagName::new(symbol),
        })
    };
    assert_eq!(link(&[binary(callee("f", &[]))]).unwrap_err(), missing("f"));
    assert_eq!(
        link(&[callee("f", &["g"]), binary(callee("g", &[]))]).unwrap_err(),
        missing("g")
    );
    // Another `f`, `x ^ 7`, beside the one that resolves references (`x * 3 + table[x & 3]`).
    let mut m = ModuleBuilder::new();
    let i32t = m.int(32);
    let fty = m.fn_ty(i32t, &[i32t]);
    let seven = m.const_int(i32t, 7);
    let mut f = m.define("f", fty);
    let v = f.bin(BinOp::Xor, arg(0), seven);
    f.ret(Some(v));
    let func = f.finish();
    let mut other_abi = abi();
    other_abi.visible_functions = vec![visible(func, "f")];
    let other = linked("f", m.finish().unwrap(), other_abi);
    for listed in [
        [binary(other.clone()), callee("f", &[])],
        [callee("f", &[]), binary(other)],
    ] {
        let t = text(&link(&listed).unwrap_or_else(|e| panic!("{e}")));
        assert!(!t.contains("call "), "{t}");
        assert!(t.contains("mul i32"), "{t}");
        assert!(!t.contains("xor i32"), "{t}");
    }
}

/// A call that SCCP deletes needs nothing linked, and supplying functions nothing references
/// changes nothing.
#[test]
fn a_dead_reference_needs_nothing_and_unreferenced_functions_change_nothing() {
    let (module, abi) = caller(true);
    let alone = specialize(module.clone(), &abi, &request(), &mut meter()).unwrap();
    let extra = [callee("f", &[]), callee("g", &[])];
    let with = specialize_linked(module, &abi, &request(), &extra, &mut meter()).unwrap();
    assert_eq!(text(&alone), text(&with));
}

/// Why a linked function of the referenced name cannot resolve it, each only for a surviving
/// call: another function type, a function its module does not export, two functions of the name,
/// another data layout.
#[test]
fn a_linked_function_that_cannot_resolve_the_reference_is_a_typed_descriptor_error() {
    let problem = |name: &str, problem| {
        Error::InvalidRequest(RequestError::LinkedFunction {
            name: DiagName::new(name),
            problem,
        })
    };
    // `f` as `i32 (i32, i32)`.
    let mut m = ModuleBuilder::new();
    let i32t = m.int(32);
    let fty = m.fn_ty(i32t, &[i32t, i32t]);
    let mut f = m.define("f", fty);
    f.ret(Some(arg(1)));
    let func = f.finish();
    let mut two = callee("f", &[]);
    code(&mut two).module = m.finish().unwrap();
    code(&mut two).abi.visible_functions = vec![visible(func, "f")];
    assert_eq!(
        link(&[two]).unwrap_err(),
        problem("f", LinkProblem::Signature)
    );

    let mut hidden = callee("f", &[]);
    code(&mut hidden).abi.visible_functions.clear();
    assert_eq!(
        link(&[hidden]).unwrap_err(),
        problem("f", LinkProblem::NotVisible)
    );

    assert_eq!(
        link(&[callee("f", &[]), callee("f", &[])]).unwrap_err(),
        problem("f", LinkProblem::Ambiguous)
    );

    let mut other = callee("f", &[]);
    let mut parts = code(&mut other).module.clone().into_parts();
    parts.data_layout.stack_align_bits = Some(1024);
    code(&mut other).module = Module::new(parts, &mut meter()).unwrap();
    assert_eq!(
        link(&[other]).unwrap_err(),
        problem("f", LinkProblem::DataLayout)
    );

    // Constants for a function that declares none.
    let mut valued = callee("f", &[]);
    let values = valued.functions[0].constants.as_mut().unwrap();
    values.insert(
        2,
        ConstValue {
            ty: DataType::Int,
            bytes: 1u32.to_le_bytes().into(),
        },
    );
    assert_eq!(
        link(&[valued]).unwrap_err(),
        Error::InvalidRequest(RequestError::FunctionConstantIndex { index: 2 })
    );
}

/// A linked module's linkage signature is read as the entry module's: a stitched module (the form
/// Metal's stitcher writes) and a linked-constants module that declares no constant link like any
/// other module, their own code imported. A linked-constants module that still declares a
/// constant has no reading: refused at the surviving call, naming the linked function, where the
/// same module with a stitching signature links (its function `free` reaches no constant).
#[test]
fn a_linked_module_s_signature_is_read_as_the_entry_s() {
    use crate::abi::{LinkageSignature, LinkageSignatureKind};
    let signed = |code: &mut LinkedCode, kind| {
        code.abi.linkage_signature = Some(LinkageSignature { kind, operands: 2 });
    };
    for kind in [
        LinkageSignatureKind::StitchingModules,
        LinkageSignatureKind::LinkedConstants,
    ] {
        let mut l = callee("f", &[]);
        signed(code(&mut l), kind);
        let e = link(&[l]).unwrap_or_else(|e| panic!("{kind:?}: {e}"));
        let t = text(&e);
        assert!(!t.contains("call "), "{kind:?}: {t}");
        assert!(t.contains("mul i32"), "{kind:?}: {t}");
    }
    let constants = |kind| {
        let mut code = constant_callee();
        signed(&mut code, kind);
        link(&[LinkedModule {
            functions: vec![LinkedName {
                name: b"f".as_slice().into(),
                function: b"free".as_slice().into(),
                specialized: true,
                constants: Ok(BTreeMap::new()),
                handle: None,
                resolves_references: true,
            }],
            code: Ok(code),
        }])
    };
    let u = match constants(LinkageSignatureKind::LinkedConstants) {
        Err(Error::Unsupported(u)) => u,
        other => panic!("{other:?}"),
    };
    assert_eq!(u.feature, Feature::Linking, "{u}");
    assert_eq!(u.symbol, Some(DiagName::new("f")), "{u}");
    let at = Location::function(DiagName::new("k"), Some(0), Some(0));
    assert_eq!(u.location, Some(at), "{u}");
    let stitched = constants(LinkageSignatureKind::StitchingModules);
    let t = text(&stitched.unwrap_or_else(|e| panic!("{e}")));
    assert!(t.contains("add i32 %a1, i32 5"), "{t}");
}

/// The visible functions of a linked module that declares the `i32` function constant `c`
/// (index 0), each `i32 (i32 x)`, as Apple's compiler writes such a module: its static
/// initializer copies `c` into the internal global `c.copy`, which the functions read.
/// `reads`: `x * c`; `guarded`: `is_function_constant_defined(c) ? x * c : x + 1`; `free`:
/// `x + 5`.
fn constant_callee() -> LinkedCode {
    let mut m = ModuleBuilder::new();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let cptr = m.ptr(AddrSpace::Constant);
    let fty = m.fn_ty(i32t, &[i32t]);
    let c = super::tests::fc_global(&mut m, "c", i32t);
    let undef = m.constant(i32t, Const::Undef);
    let copy_name = m.sym("c.copy");
    let copy = m.global(Global {
        name: copy_name,
        value_ty: i32t,
        space: AddrSpace::Constant,
        constant: false,
        init: Some(undef),
        externally_initialized: false,
        visibility: Default::default(),
        unnamed_addr: Default::default(),
        linkage: Linkage::Internal,
        align: None,
        section: None,
    });
    let (c, copy) = (m.global_addr(c), m.global_addr(copy));
    let is_def_ty = m.fn_ty(i1, &[cptr]);
    let is_def = m.declare("air.is_function_constant_defined", is_def_ty);
    let (one, five) = (m.const_int(i32t, 1), m.const_int(i32t, 5));
    let load = |f: &mut FunctionBuilder<'_>, ptr: Value| {
        f.push(
            i32t,
            InstKind::Load {
                ptr,
                align: Align::from_bytes(4),
                volatile: false,
                atomic: None,
                range: None,
            },
        )
    };
    let void = m.void();
    let init_ty = m.fn_ty(void, &[]);
    let mut init = m.define("init", init_ty);
    let v = load(&mut init, c);
    store(&mut init, v, copy);
    init.ret(None);
    let init = init.finish();
    let init = m.func_mut(init).unwrap();
    init.section = Some(Section::StaticInit);
    init.linkage = Linkage::Internal;

    let mut reads = m.define("reads", fty);
    let v = load(&mut reads, copy);
    let r = reads.bin(BinOp::Mul, arg(0), v);
    reads.ret(Some(r));
    let reads = reads.finish();

    let mut guarded = m.define("guarded", fty);
    let defined = guarded.push(
        i1,
        InstKind::Call(Box::new(Call {
            callee: Callee::Air {
                decl: is_def,
                op: crate::intrinsics::AirIntrinsic::FunctionConstant(
                    crate::intrinsics::FunctionConstant::IsDefined,
                ),
            },
            args: vec![c].into(),
            fmf: FastMath::NONE,
            tail: TailKind::None,
            attrs: None,
            range: None,
        })),
    );
    let v = load(&mut guarded, copy);
    let t = guarded.bin(BinOp::Mul, arg(0), v);
    let f = guarded.bin(BinOp::Add, arg(0), one);
    let r = guarded.push(
        i32t,
        InstKind::Select {
            c: defined,
            t,
            f,
            fmf: FastMath::NONE,
        },
    );
    guarded.ret(Some(r));
    let guarded = guarded.finish();

    let mut free = m.define("free", fty);
    let r = free.bin(BinOp::Add, arg(0), five);
    free.ret(Some(r));
    let free = free.finish();

    let module = m.finish().unwrap();
    let mut abi = abi();
    abi.visible_functions = vec![
        visible(reads, "reads"),
        visible(guarded, "guarded"),
        visible(free, "free"),
    ];
    abi.function_constants = vec![super::tests::fc_decl(0, GlobalId::new(0), i32t)];
    LinkedCode { module, abi }
}

/// The caller's reference `f` linked to `function` of [`constant_callee`], made specialized or
/// not, with `c`'s value where `c` gives one.
fn constant_link(function: &str, specialized: bool, c: Option<u32>) -> Result<Entry, Error> {
    let constants = c
        .map(|v| {
            let value = ConstValue {
                ty: DataType::Int,
                bytes: v.to_le_bytes().into(),
            };
            BTreeMap::from([(0, value)])
        })
        .unwrap_or_default();
    link(&[LinkedModule {
        functions: vec![LinkedName {
            name: b"f".as_slice().into(),
            function: function.as_bytes().into(),
            specialized,
            constants: Ok(constants),
            handle: None,
            resolves_references: true,
        }],
        code: Ok(constant_callee()),
    }])
}

/// A specialized linked function is specialized with its own values before it is linked, as
/// Metal specializes the `MTLFunction`: its module's static initializer copies the value, and the
/// import computes with it, `is_function_constant_defined` true.
#[test]
fn a_specialized_linked_function_computes_with_its_constant_values() {
    for function in ["reads", "guarded"] {
        let e = constant_link(function, true, Some(7)).unwrap_or_else(|e| panic!("{e}"));
        let t = text(&e);
        assert!(!t.contains("call "), "{function}: {t}");
        assert!(!t.contains("load "), "{function}: {t}");
        assert!(t.contains("mul i32 %a1, i32 7"), "{function}: {t}");
        assert!(!t.contains("add i32"), "{function}: {t}");
    }
}

/// A specialized linked function made without a value for the constant links, as Metal makes
/// that pipeline: the constant is undefined (`is_function_constant_defined` false) and reads as
/// an entry's undefined constants read, as zero.
#[test]
fn a_specialized_linked_function_without_a_value_reads_its_constant_as_undefined() {
    let reads = text(&constant_link("reads", true, None).unwrap_or_else(|e| panic!("{e}")));
    assert!(
        reads.contains("mul i32 %a1, i32 zeroinitializer"),
        "{reads}"
    );
    assert!(!reads.contains("load "), "{reads}");
    let guarded = text(&constant_link("guarded", true, None).unwrap_or_else(|e| panic!("{e}")));
    assert!(guarded.contains("add i32 %a1, i32 1"), "{guarded}");
    assert!(!guarded.contains("mul i32"), "{guarded}");
}

/// An unspecialized linked function (`newFunctionWithName:`) that reaches a function constant
/// refuses the link at the surviving call, guarded by `is_function_constant_defined` or not, as
/// Metal fails the creation ("cannot lower module with unresolved function constants"); one that
/// reaches none links, and a dead call needs nothing.
#[test]
fn an_unspecialized_linked_function_that_reaches_a_constant_is_refused() {
    let unresolved = Error::InvalidRequest(RequestError::LinkedFunction {
        name: DiagName::new("f"),
        problem: LinkProblem::UnresolvedConstant { index: 0 },
    });
    for function in ["reads", "guarded"] {
        assert_eq!(
            constant_link(function, false, None).map(|_| ()),
            Err(unresolved.clone()),
            "{function}"
        );
    }
    let free = text(&constant_link("free", false, None).unwrap_or_else(|e| panic!("{e}")));
    assert!(free.contains("add i32 %a1, i32 5"), "{free}");
    let (module, abi) = caller(true);
    let dead = LinkedModule {
        functions: vec![LinkedName {
            name: b"f".as_slice().into(),
            function: b"reads".as_slice().into(),
            specialized: false,
            constants: Ok(BTreeMap::new()),
            handle: None,
            resolves_references: true,
        }],
        code: Ok(constant_callee()),
    };
    assert!(specialize_linked(module, &abi, &request(), &[dead], &mut meter()).is_ok());
}

/// A linked function links under its name (a specialized name, where it has one) and is defined
/// by its function of another name: a reference to the function's own name it does not resolve.
#[test]
fn a_linked_function_links_under_its_name_not_its_function_s() {
    let e = constant_link("free", true, None).unwrap_or_else(|e| panic!("{e}"));
    assert!(text(&e).contains("add i32 %a1, i32 5"));
    let own = LinkedModule {
        functions: vec![LinkedName {
            name: b"other".as_slice().into(),
            function: b"f".as_slice().into(),
            specialized: true,
            constants: Ok(BTreeMap::new()),
            handle: None,
            resolves_references: true,
        }],
        ..callee("f", &[])
    };
    assert_eq!(
        link(&[own]).unwrap_err(),
        Error::NeedsInput(MissingInput::LinkedFunction {
            symbol: DiagName::new("f"),
        })
    );
}

/// A linked module's static initializer runs before the import, with no function constant too:
/// what it stores is the global's value. Here `c` is an ordinary constant global holding 9.
#[test]
fn a_linked_module_s_static_initializer_runs_before_the_import() {
    let mut code = constant_callee();
    code.abi.function_constants.clear();
    let mut parts = code.module.into_parts();
    let i32t = parts.types.lookup(&Type::Int(32)).unwrap();
    let nine = parts
        .consts
        .intern(
            crate::ir::Constant {
                ty: i32t,
                value: Const::Int(9),
            },
            &mut meter(),
        )
        .unwrap();
    let c = parts.globals.get_mut(GlobalId::new(0)).unwrap();
    c.init = Some(nine);
    c.externally_initialized = false;
    c.section = None;
    code.module = Module::new(parts, &mut meter()).unwrap();
    let e = link(&[LinkedModule {
        functions: vec![LinkedName {
            name: b"f".as_slice().into(),
            function: b"reads".as_slice().into(),
            specialized: false,
            constants: Ok(BTreeMap::new()),
            handle: None,
            resolves_references: true,
        }],
        code: Ok(code),
    }])
    .unwrap_or_else(|e| panic!("{e}"));
    assert!(text(&e).contains("mul i32 %a1, i32 9"), "{}", text(&e));
}

/// A call to a declaration no reference names is still a function to link from a dynamic
/// library, whatever is linked.
#[test]
fn a_call_to_another_declaration_is_still_linking() {
    let (module, mut abi) = caller(false);
    abi.visible_function_refs.clear();
    let u = match specialize_linked(module, &abi, &request(), &[callee("f", &[])], &mut meter()) {
        Err(Error::Unsupported(u)) => u,
        other => panic!("{other:?}"),
    };
    assert_eq!(u.feature, Feature::Linking);
    assert_eq!(u.symbol, Some(DiagName::new("f.MTL_VISIBLE_FN_REF")));
}

/// An identified struct `name` of two `i32` fields.
fn pair_ty(m: &mut ModuleBuilder, name: &str) -> TypeId {
    let i32t = m.int(32);
    let name = m.sym(name);
    m.ty(Type::Struct {
        fields: vec![i32t, i32t].into(),
        packed: false,
        name: Some(name),
    })
}

/// The Metal compiler names one struct `%struct.T.153` in the caller and `%struct.T` in the
/// module that defines the visible function: linking maps types as LLVM's IR linker does, by
/// body, so the reference resolves and the definition's struct is the caller's.
#[test]
fn a_struct_of_another_name_and_the_same_body_is_the_same_type() {
    // Caller: `*out = f(tid).0 + f(tid).1`, `f` returning `"pair.153"`.
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let pair = pair_ty(&mut m, "pair.153");
    let fty = m.fn_ty(pair, &[i32t]);
    let stub = m.declare("f.MTL_VISIBLE_FN_REF", fty);
    let kty = m.fn_ty(void, &[dev, i32t]);
    let mut k = m.define("k", kty);
    let v = call(&mut k, pair, stub, &[arg(1)]);
    let a = k.push(
        i32t,
        InstKind::ExtractValue {
            agg: v,
            indices: vec![0].into(),
        },
    );
    let b = k.push(
        i32t,
        InstKind::ExtractValue {
            agg: v,
            indices: vec![1].into(),
        },
    );
    let s = k.bin(BinOp::Add, a, b);
    store(&mut k, s, arg(0));
    k.ret(None);
    k.finish();
    let (_, mut abi) = caller(false);
    abi.visible_function_refs = vec![reference(stub, "f")];
    let module = m.finish().unwrap();

    // Linked: `f(x) = {x * 3, x}` returning `"pair"`.
    let mut m = ModuleBuilder::new();
    let i32t = m.int(32);
    let pair = pair_ty(&mut m, "pair");
    let fty = m.fn_ty(pair, &[i32t]);
    let three = m.const_int(i32t, 3);
    let undef = Value::Const(m.constant(pair, Const::Undef));
    let mut f = m.define("f", fty);
    let x3 = f.bin(BinOp::Mul, arg(0), three);
    let p = f.push(
        pair,
        InstKind::InsertValue {
            agg: undef,
            val: x3,
            indices: vec![0].into(),
        },
    );
    let p = f.push(
        pair,
        InstKind::InsertValue {
            agg: p,
            val: arg(0),
            indices: vec![1].into(),
        },
    );
    f.ret(Some(p));
    let func = f.finish();
    let mut labi = abi_with_visible(func, "f");
    labi.visible_function_refs.clear();
    let lf = linked("f", m.finish().unwrap(), labi);
    let e = specialize_linked(module, &abi, &request(), &[lf], &mut meter()).unwrap();
    let t = text(&e);
    assert!(!t.contains("call "), "{t}");
    assert!(t.contains("mul i32"), "{t}");
    assert!(t.contains("; \"pair.153\""), "{t}");
    assert!(!t.contains("; \"pair\""), "{t}");
}

/// The ABI of a linked module exporting `func` as `name`.
fn abi_with_visible(func: FuncId, name: &str) -> AirAbi {
    let mut a = abi();
    a.visible_functions = vec![visible(func, name)];
    a
}

/// A definition imported for one reference is not converted for another reference of an
/// isomorphic but different type: here the caller has two structs of one body, `f` resolves
/// against the caller's second, and `g`'s own reference to `f` maps to its first. Refused as not
/// implemented, naming `f`.
#[test]
fn a_second_reference_of_an_isomorphic_other_type_is_not_implemented() {
    // Caller: `*out = g(f(tid).0)`, `f` returning `"b"`, with `"a"` of the same body before it.
    let mut m = ModuleBuilder::new();
    let void = m.void();
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    pair_ty(&mut m, "a");
    let b = pair_ty(&mut m, "b");
    let fty = m.fn_ty(b, &[i32t]);
    let gty = m.fn_ty(i32t, &[i32t]);
    let fstub = m.declare("f.MTL_VISIBLE_FN_REF", fty);
    let gstub = m.declare("g.MTL_VISIBLE_FN_REF", gty);
    let kty = m.fn_ty(void, &[dev, i32t]);
    let mut k = m.define("k", kty);
    let v = call(&mut k, b, fstub, &[arg(1)]);
    let x = k.push(
        i32t,
        InstKind::ExtractValue {
            agg: v,
            indices: vec![0].into(),
        },
    );
    let w = call(&mut k, i32t, gstub, &[x]);
    store(&mut k, w, arg(0));
    k.ret(None);
    let kf = k.finish();
    let (_, mut abi) = caller(false);
    abi.entries[0].func = kf;
    abi.visible_function_refs = vec![reference(fstub, "f"), reference(gstub, "g")];
    let module = m.finish().unwrap();

    // `f(x) = {x, x}` returning `"d"`.
    let mut m = ModuleBuilder::new();
    let i32t = m.int(32);
    let d = pair_ty(&mut m, "d");
    let fty = m.fn_ty(d, &[i32t]);
    let undef = Value::Const(m.constant(d, Const::Undef));
    let mut f = m.define("f", fty);
    let p = f.push(
        d,
        InstKind::InsertValue {
            agg: undef,
            val: arg(0),
            indices: vec![0].into(),
        },
    );
    let p = f.push(
        d,
        InstKind::InsertValue {
            agg: p,
            val: arg(0),
            indices: vec![1].into(),
        },
    );
    f.ret(Some(p));
    let func = f.finish();
    let lf = linked("f", m.finish().unwrap(), abi_with_visible(func, "f"));

    // `g(x) = f(x).0` through its own reference to `f`, returning `"c"`.
    let mut m = ModuleBuilder::new();
    let i32t = m.int(32);
    let c = pair_ty(&mut m, "c");
    let fty = m.fn_ty(c, &[i32t]);
    let gty = m.fn_ty(i32t, &[i32t]);
    let fstub = m.declare("f.MTL_VISIBLE_FN_REF", fty);
    let mut g = m.define("g", gty);
    let v = call(&mut g, c, fstub, &[arg(0)]);
    let x = g.push(
        i32t,
        InstKind::ExtractValue {
            agg: v,
            indices: vec![0].into(),
        },
    );
    g.ret(Some(x));
    let func = g.finish();
    let mut gabi = abi_with_visible(func, "g");
    gabi.visible_function_refs = vec![reference(fstub, "f")];
    let lg = linked("g", m.finish().unwrap(), gabi);

    let u = match specialize_linked(module, &abi, &request(), &[lf, lg], &mut meter()) {
        Err(Error::Unsupported(u)) => u,
        other => panic!("{other:?}"),
    };
    assert_eq!(u.feature, Feature::Linking, "{u}");
    assert_eq!(u.symbol, Some(DiagName::new("f")), "{u}");
}
