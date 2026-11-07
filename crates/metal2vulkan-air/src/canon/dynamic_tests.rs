//! Dynamic-library linking tests over IR built with the typed test builder: a kernel imports
//! functions and variables through its `air.dyld_flat_table`, and the stage's dynamic libraries
//! define them. The cases follow the probes' measured Metal answers (an M3 Max and an M4 agree):
//! the search order, a library's own calls binding to its own definitions, a variable shared by
//! every reference, the refusals of what is missing or not loaded.

use std::collections::BTreeMap;

use super::*;
use crate::abi::{
    AbiInt, Access, AirAbi, ArgTypeInfo, BufferArg, CompileOptions, EntryPoint, ModuleLimits,
    ParamBinding, ParamRole, Slot, Stage,
};
use crate::budget::{Budget, Meter};
use crate::ir::build::{FunctionBuilder, ModuleBuilder};
use crate::ir::print::print;
use crate::ir::{
    Align, BinOp, Call, Const, ConstId, FastMath, Global, Linkage, TailKind, Visibility,
};

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

/// One `i32 (i32 x)` function of a test module: `x * mul`, plus each of `calls` applied to `x`,
/// plus each variable of `reads` (an `i32` in the constant space).
#[derive(Clone, Copy)]
struct Def<'a> {
    name: &'a str,
    mul: u64,
    calls: &'a [&'a str],
    reads: &'a [&'a str],
    linkage: Linkage,
    visibility: Visibility,
}

fn def<'a>(name: &'a str, mul: u64) -> Def<'a> {
    Def {
        name,
        mul,
        calls: &[],
        reads: &[],
        linkage: Linkage::External,
        visibility: Visibility::Default,
    }
}

/// The body every test function shares: `x * mul + Σ calls(x) + Σ reads`, its callees and
/// variables found by name (`funcs`, `vars`) or else declared and listed as imports.
struct Names {
    funcs: BTreeMap<String, FuncId>,
    vars: BTreeMap<String, GlobalId>,
    imports: Vec<Item>,
}

impl Names {
    fn func(&mut self, m: &mut ModuleBuilder, name: &str) -> FuncId {
        if let Some(&f) = self.funcs.get(name) {
            return f;
        }
        let i32t = m.int(32);
        let fty = m.fn_ty(i32t, &[i32t]);
        let f = m.declare(name, fty);
        self.funcs.insert(name.into(), f);
        self.imports.push(Item::Func(f));
        f
    }

    fn var(&mut self, m: &mut ModuleBuilder, name: &str) -> GlobalId {
        if let Some(&g) = self.vars.get(name) {
            return g;
        }
        let i32t = m.int(32);
        let g = variable(m, name, i32t, None);
        self.vars.insert(name.into(), g);
        self.imports.push(Item::Global(g));
        g
    }
}

/// A constant-space variable `name` of `ty`: a definition (external linkage) with `init`, or a
/// declaration.
fn variable(m: &mut ModuleBuilder, name: &str, ty: TypeId, init: Option<ConstId>) -> GlobalId {
    let name = m.sym(name);
    m.global(Global {
        name,
        value_ty: ty,
        space: AddrSpace::Constant,
        constant: init.is_some(),
        init,
        externally_initialized: false,
        linkage: Linkage::External,
        visibility: Visibility::Default,
        unnamed_addr: Default::default(),
        align: None,
        section: None,
    })
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

/// `x * mul + Σ calls(x) + Σ reads` in `f`, whose argument `x` is `x`.
fn compute(f: &mut FunctionBuilder<'_>, names: &mut Names, x: Value, d: &Def<'_>) -> Value {
    let i32t = f.module().int(32);
    let mul = f.module().const_int(i32t, d.mul);
    let mut v = f.bin(BinOp::Mul, x, mul);
    for c in d.calls {
        let callee = names.func(f.module(), c);
        let r = call(f, i32t, callee, &[x]);
        v = f.bin(BinOp::Add, v, r);
    }
    for r in d.reads {
        let g = names.var(f.module(), r);
        let at = f.module().global_addr(g);
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
        v = f.bin(BinOp::Add, v, word);
    }
    v
}

/// The flat table of `imports`: a struct of their addresses.
fn flat_table(m: &mut ModuleBuilder, imports: &[Item]) {
    if imports.is_empty() {
        return;
    }
    let mut tys = Vec::new();
    let mut words = Vec::new();
    for item in imports {
        let (ty, value) = match *item {
            Item::Func(f) => (m.ptr(AddrSpace::Thread), Const::FuncAddr(f)),
            Item::Global(g) => (m.ptr(AddrSpace::Constant), Const::GlobalAddr(g)),
        };
        tys.push(ty);
        words.push(m.constant(ty, value));
    }
    let table = m.struct_ty(&tys);
    let init = m.constant(table, Const::Aggregate(words.into()));
    let name = m.sym("air.dyld_flat_table");
    m.global(Global {
        name,
        value_ty: table,
        space: AddrSpace::Thread,
        constant: false,
        init: Some(init),
        externally_initialized: false,
        linkage: Linkage::External,
        visibility: Visibility::Hidden,
        unnamed_addr: Default::default(),
        align: None,
        section: None,
    });
}

/// A module of `defs` (in order: each calls only those before it, or imports) and the defined
/// variables `vars` (`i32` values), with its flat table.
fn module_of(m: &mut ModuleBuilder, defs: &[Def<'_>], vars: &[(&str, u64)]) -> Names {
    let i32t = m.int(32);
    let mut names = Names {
        funcs: BTreeMap::new(),
        vars: BTreeMap::new(),
        imports: Vec::new(),
    };
    for (name, value) in vars {
        let init = m.constant(i32t, Const::Int(*value));
        let g = variable(m, name, i32t, Some(init));
        names.vars.insert((*name).into(), g);
    }
    let fty = m.fn_ty(i32t, &[i32t]);
    for d in defs {
        let mut f = m.define(d.name, fty);
        let x = f.arg(0);
        let v = compute(&mut f, &mut names, x, d);
        f.ret(Some(v));
        let id = f.finish();
        if let Some(func) = m.func_mut(id) {
            func.linkage = d.linkage;
            func.visibility = d.visibility;
        }
        names.funcs.insert(d.name.into(), id);
    }
    names
}

/// A dynamic library of install name `name` holding `defs` and `vars`.
fn library(name: &str, defs: &[Def<'_>], vars: &[(&str, u64)]) -> DynamicModule {
    let mut m = ModuleBuilder::new();
    let names = module_of(&mut m, defs, vars);
    flat_table(&mut m, &names.imports);
    DynamicModule {
        install_name: name.as_bytes().into(),
        code: Ok(LinkedCode {
            module: m.finish().unwrap(),
            abi: abi(),
        }),
    }
}

/// `kernel k(device i32* out, i32 tid)`: `*out = tid * 1 + Σ calls(tid) + Σ reads`, its callees
/// and variables imports (or among `defs`, the module's own). With `dead`, the store sits behind
/// `br false`.
fn client(kernel: Def<'_>, defs: &[Def<'_>], dead: bool) -> (Module, AirAbi) {
    let mut m = ModuleBuilder::new();
    let mut names = module_of(&mut m, defs, &[]);
    let void = m.void();
    let i1 = m.int(1);
    let i32t = m.int(32);
    let dev = m.ptr(AddrSpace::Device);
    let kty = m.fn_ty(void, &[dev, i32t]);
    let no = m.const_int(i1, 0);
    let mut k = m.define("k", kty);
    let (out, tid) = (k.arg(0), k.arg(1));
    if dead {
        let then = k.block();
        let done = k.block();
        k.cond_br(no, then, done);
        k.switch_to(then);
        let v = compute(&mut k, &mut names, tid, &kernel);
        store(&mut k, v, out);
        k.br(done);
        k.switch_to(done);
    } else {
        let v = compute(&mut k, &mut names, tid, &kernel);
        store(&mut k, v, out);
    }
    k.ret(None);
    let k = k.finish();
    flat_table(&mut m, &names.imports);
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
    (module, abi)
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

fn request() -> SpecRequest {
    SpecRequest {
        call_stack_depth: 1,
        open_cl_mode: false,
        entry: "k".into(),
        constants: BTreeMap::new(),
    }
}

/// The kernel calling (and reading) what `kernel` names, specialized against `libraries`.
fn specialize_with(
    (module, abi): (Module, AirAbi),
    libraries: Vec<DynamicModule>,
    unloaded: Option<Error>,
) -> Result<Entry, Error> {
    let links = DynamicLinks {
        libraries,
        unloaded,
    };
    specialize_dynamic(module, &abi, &request(), (&[], &links), &mut meter())
}

fn kernel<'a>(calls: &'a [&'a str], reads: &'a [&'a str]) -> Def<'a> {
    Def {
        calls,
        reads,
        ..def("k", 1)
    }
}

fn missing(symbol: &str) -> Error {
    Error::NeedsInput(MissingInput::DynamicSymbol {
        symbol: DiagName::new(symbol),
    })
}

/// The entry's text, with no call left.
fn inlined(e: &Entry) -> String {
    let t = print(&e.module);
    assert!(!t.contains("call "), "{t}");
    t
}

/// An imported function a library exports is imported and inlined, its body the library's.
#[test]
fn an_import_a_library_exports_is_inlined() {
    let e = specialize_with(
        client(kernel(&["f"], &[]), &[], false),
        vec![library("libA", &[def("f", 1007)], &[])],
        None,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let t = inlined(&e);
    assert!(t.contains("1007"), "{t}");
}

/// A surviving use of an import no library exports needs it, by symbol; with an install name not
/// loaded, Metal fails the creation whatever the other libraries export.
#[test]
fn an_import_no_library_exports_needs_the_symbol() {
    let refused = |libraries, unloaded| {
        specialize_with(client(kernel(&["f"], &[]), &[], false), libraries, unloaded).unwrap_err()
    };
    assert_eq!(refused(Vec::new(), None), missing("f"));
    assert_eq!(
        refused(vec![library("libA", &[def("g", 3)], &[])], None),
        missing("f")
    );
    // A definition of internal linkage is not exported.
    let internal = Def {
        linkage: Linkage::Internal,
        ..def("f", 3)
    };
    assert_eq!(
        refused(vec![library("libA", &[internal], &[])], None),
        missing("f")
    );
    let unloaded = Error::NeedsInput(MissingInput::DynamicLibrary {
        install_name: DiagName::new("libB"),
    });
    assert_eq!(
        refused(
            vec![library("libA", &[def("f", 3)], &[])],
            Some(unloaded.clone())
        ),
        unloaded
    );
}

/// An import whose use SCCP deletes needs nothing: loading is lazy, so even an install name not
/// loaded leaves it fine.
#[test]
fn a_dead_import_needs_nothing() {
    let unloaded = Error::NeedsInput(MissingInput::DynamicLibrary {
        install_name: DiagName::new("libB"),
    });
    for (libraries, unloaded) in [(Vec::new(), None), (Vec::new(), Some(unloaded))] {
        specialize_with(
            client(kernel(&["f"], &["v"]), &[], true),
            libraries,
            unloaded,
        )
        .unwrap_or_else(|e| panic!("{e}"));
    }
}

/// The first library in the search order that exports a symbol supplies it.
#[test]
fn the_first_library_exporting_the_symbol_wins() {
    let e = specialize_with(
        client(kernel(&["f"], &[]), &[], false),
        vec![
            library("libP", &[def("g", 5)], &[]),
            library("libA", &[def("f", 1007)], &[]),
            library("libB", &[def("f", 1011)], &[]),
        ],
        None,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let t = inlined(&e);
    assert!(t.contains("1007") && !t.contains("1011"), "{t}");
}

/// A library function's call to a function its own library defines binds to that definition,
/// while the module's import of the same symbol is searched for (probe: a preloaded library's
/// `g` supplies the kernel's `g`, not `f`'s).
#[test]
fn a_library_call_to_its_own_definition_binds_internally() {
    let f = Def {
        calls: &["g"],
        ..def("f", 1007)
    };
    let e = specialize_with(
        client(kernel(&["f", "g"], &[]), &[], false),
        vec![
            library("libP", &[def("g", 2003)], &[]),
            library("libA", &[def("g", 2001), f], &[]),
        ],
        None,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let t = inlined(&e);
    // `f` from libA calls libA's `g`; the kernel's `g` is libP's.
    for constant in ["1007", "2001", "2003"] {
        assert!(t.contains(constant), "{constant}: {t}");
    }
}

/// A library function's own imports are searched for in the same order: libC's `f` imports `h`,
/// which libB exports. The module's own definitions supply no import (probe: the client's `h` does
/// not answer `f`'s).
#[test]
fn a_library_import_is_searched_for_in_turn() {
    let f = Def {
        calls: &["h"],
        ..def("f", 1007)
    };
    let own_h = def("h", 3001);
    let e = specialize_with(
        client(
            Def {
                calls: &["f", "h"],
                ..def("k", 1)
            },
            &[own_h],
            false,
        ),
        vec![
            library("libC", &[f], &[]),
            library("libB", &[def("h", 3003)], &[]),
        ],
        None,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let t = inlined(&e);
    for constant in ["1007", "3001", "3003"] {
        assert!(t.contains(constant), "{constant}: {t}");
    }
    // Without libB, `f`'s import is missing.
    let refused = specialize_with(
        client(kernel(&["f"], &[]), &[], false),
        vec![library("libC", &[f], &[])],
        None,
    )
    .unwrap_err();
    assert_eq!(refused, missing("h"));
}

/// An imported variable takes the library's value (probe: `extern constant float dyl_k` reads
/// 7.5); one no library exports needs its symbol at a surviving use.
#[test]
fn an_imported_variable_takes_the_library_definition() {
    let e = specialize_with(
        client(kernel(&[], &["v"]), &[], false),
        vec![library("libV", &[], &[("v", 4242)])],
        None,
    )
    .unwrap_or_else(|e| panic!("{e}"));
    let t = inlined(&e);
    assert!(t.contains("4242"), "{t}");
    let refused =
        specialize_with(client(kernel(&[], &["v"]), &[], false), Vec::new(), None).unwrap_err();
    assert_eq!(refused, missing("v"));
}

/// A variable is one object for every reference: the module's import of it and the library's own
/// function reading it resolve to one global, the module's declaration made the definition.
#[test]
fn an_imported_variable_is_one_object() {
    let get = Def {
        reads: &["v"],
        ..def("get", 1007)
    };
    let (module, abi) = client(kernel(&["get"], &["v"]), &[], false);
    let links = DynamicLinks {
        libraries: vec![library("libV", &[get], &[("v", 4242)])],
        unloaded: None,
    };
    let (parts, stubs, _, _) = link::link(
        module.into_parts(),
        &abi,
        (&[], &links),
        false,
        &mut meter(),
    )
    .unwrap();
    let named: Vec<&Global> = parts
        .globals
        .values()
        .filter(|g| parts.symbols.get(g.name) == Some(b"v".as_slice()))
        .collect();
    assert_eq!(named.len(), 1);
    assert!(named[0].init.is_some());
    assert!(stubs.imports_variables());
}

/// What the search cannot answer is refused, each only at a surviving use: a library whose
/// module does not decode before the one exporting it, a hidden definition (whose export Metal's
/// answer is not measured for), a definition of another kind or type, or a library linking does
/// not import from (function constants).
#[test]
fn what_the_search_cannot_answer_is_refused() {
    let refused = |libraries: Vec<DynamicModule>| {
        specialize_with(client(kernel(&["f"], &[]), &[], false), libraries, None).unwrap_err()
    };
    let bad: Error = Unsupported::not_implemented(Feature::Linking).into();
    let undecoded = DynamicModule {
        install_name: b"libX".as_slice().into(),
        code: Err(bad.clone()),
    };
    assert_eq!(
        refused(vec![undecoded, library("libA", &[def("f", 3)], &[])]),
        bad
    );
    let linking = |e: Error, key: &'static str| match e {
        Error::Unsupported(u) => {
            assert_eq!(u.feature, Feature::Linking, "{u}");
            assert_eq!(u.symbol, Some(DiagName::new("f")), "{u}");
            assert_eq!(
                u.location,
                Some(Location::metadata(
                    Some(DiagName::new("f")),
                    None,
                    Some(key)
                )),
                "{u}"
            );
        }
        other => panic!("{other}"),
    };
    let hidden = Def {
        visibility: Visibility::Hidden,
        ..def("f", 3)
    };
    linking(
        refused(vec![
            library("libA", &[hidden], &[]),
            library("libB", &[def("f", 5)], &[]),
        ]),
        "visibility",
    );
    linking(refused(vec![library("libA", &[], &[("f", 1)])]), "type");
    let mut constants = library("libA", &[def("f", 3)], &[]);
    if let Ok(code) = constants.code.as_mut() {
        let i32t = code.module.types().lookup(&Type::Int(32)).unwrap();
        code.abi.function_constants = vec![crate::abi::FunctionConstantDecl {
            index: 0,
            global: GlobalId::new(0),
            ty: i32t,
            flag: None,
            data_type: None,
            name: DiagName::new("c"),
            type_name: DiagName::new("int"),
        }];
    }
    linking(refused(vec![constants]), "air.function_constants");
}

/// A declaration no flat table names is no import: a call to it is still `Linking`, whatever
/// the libraries export.
#[test]
fn a_declaration_the_flat_table_does_not_name_is_still_linking() {
    let (module, abi) = client(kernel(&["f"], &[]), &[], false);
    let mut parts = module.into_parts();
    let table = parts
        .globals
        .values_mut()
        .find(|g| parts.symbols.get(g.name) == Some(b"air.dyld_flat_table".as_slice()))
        .unwrap();
    table.name = parts.symbols.intern(b"other_table", &mut meter()).unwrap();
    let module = Module::new(parts, &mut meter()).unwrap();
    match specialize_with(
        (module, abi),
        vec![library("libA", &[def("f", 3)], &[])],
        None,
    ) {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::Linking, "{u}");
            assert_eq!(u.symbol, Some(DiagName::new("f")), "{u}");
        }
        other => panic!("{other:?}"),
    }
}
