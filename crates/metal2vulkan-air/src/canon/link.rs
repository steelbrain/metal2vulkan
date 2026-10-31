//! Static linking of visible-function references, and the linking of dynamic-library imports.
//!
//! A module calls a `[[visible]]` function it does not define through a stub declaration, which
//! `!air.visible_function_references` maps to the function's name, the Metal linker's symbol. A
//! pipeline supplies the functions it links (`MTLLinkedFunctions`) as [`LinkedModule`]s. This step,
//! run right after entry selection, resolves every stub whose name one linked function has: that
//! function is imported into the module with everything it reaches (its helpers, globals,
//! constants, types, and declarations; its own stubs are resolved in turn, against the same linked
//! functions), and every direct call to the stub is redirected to the imported definition, which
//! inlining then takes like any other callee. A call cycle through linked functions is inlining's
//! `Recursion`. Types map as LLVM's IR linker maps them: the definition's type pairs with the
//! stub's structurally, identified structs by body whatever their names (the Metal compiler numbers
//! a struct `%struct.T.153` in one module and `%struct.T` in another), and any other identified
//! struct is the module's first struct of the same body.
//!
//! Nothing here refuses: a call to a stub may be dead once constants are bound, and Metal links
//! what the specialized function calls. Each stub that is not resolved keeps a [`Stub`] state,
//! and the post-check refuses a call to it that survives SCCP: `NeedsInput(LinkedFunction)` when
//! no linked function has its name, and otherwise the reason the named function cannot resolve
//! it (another function type, a function that is not visible, two linked functions of that name,
//! another data layout, or a module feature linking does not implement yet: function constants,
//! a static initializer, a linkage signature, or a global it only declares and does not import).
//!
//! When the entry takes a visible-function table, every function the pipeline can place in one (a
//! [`LinkedName`] with a table word) is imported the same way, after the stubs, as a [`Candidate`]
//! of the calls through tables that `devirt` rewrites; an import that cannot be made is kept as the
//! candidate's refusal ([`Unimported`]), with the function's type when its module decoded, raised
//! only at a surviving call that may call it.
//!
//! A module imports a symbol from a dynamic library (`MTLDynamicLibrary`) through a declaration its
//! `air.dyld_flat_table` names, the AIR linker's list of the module's imports: a function, or a
//! variable (a global it only declares). The stage's dynamic libraries ([`DynamicLinks`]) are
//! searched in order, as Metal searches them (probes, an M3 Max and an M4): one order for every
//! import, the module's and each imported library function's alike, the first library that exports
//! the symbol (defines it with a linkage other than internal or private) supplying it. Its
//! definition is imported as a linked function is, with everything it reaches, so a library
//! function's calls to the functions its own library defines stay bound to them (Metal searches for
//! none of those), and the imports it brings are searched for in turn. A function import's calls
//! are redirected to the definition; a variable import's declaration becomes the definition, so
//! every reference to the symbol, the library's own included, is one object. The module's own
//! definitions supply no import: Metal does not export them. When an install name could not be
//! loaded ([`DynamicLinks::unloaded`]), nothing is searched and every import's surviving use is
//! refused with that error, as Metal fails such a creation whatever the other libraries export; an
//! import no library exports is refused at a surviving use, `NeedsInput(DynamicSymbol)`. A
//! declaration no flat table names is no import: a call to it is still `Linking`, not implemented.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};

use crate::abi::{AirAbi, SamplerState, StaticSampler, VisibleSymbol};
use crate::budget::{vec_bytes, Meter};
use crate::error::{DiagName, Limit, Location};
use crate::error::{Error, Feature, LinkProblem, MissingInput, RequestError, Unsupported};
use crate::ir::{Arena, Callee, Const, ConstId, ConstTable, DataLayout, FuncId, Function, Global};
use crate::ir::{GlobalId, Idx, InstKind, Linkage, Module, ModuleParts, Section, Sym, Symbols};
use crate::ir::{Type, TypeId, TypeTable, Visibility};

use super::fc;
use super::fold::Folder;
use super::renumber::{bodies, inst_refs, isomorphism, Bodies, Renumber, Syms};
use super::rewrite::{filled, invariant, scoped};
use super::sccp::{self, FcState};
use super::static_init::{self, Unevaluable};
use super::ConstValue;

/// One module the pipeline links functions from, as canonicalization takes it: the functions
/// linked from it and the module decoded once for all of them.
#[derive(Clone, Debug)]
pub struct LinkedModule {
    /// The functions the pipeline links from this module.
    pub functions: Vec<LinkedName>,
    /// The decoded module, or why its bitcode does not decode: refused only at a surviving call
    /// to one of `functions` (a module nothing calls may be anything).
    pub code: Result<LinkedCode, Error>,
}

/// One function the pipeline links: the name it links it under (its Metal function name, which a
/// visible-function reference names), the name of its definition in its module, and the
/// function-constant values it was specialized with by index, or why those values do not apply to
/// its module (refused only at a surviving call).
#[derive(Clone, Debug)]
pub struct LinkedName {
    /// The Metal function name it links under, which a visible-function reference names: its
    /// specialized name where it was made with one (`MTLFunctionDescriptor.specializedName`), else
    /// `function`.
    pub name: Box<[u8]>,
    /// The name of its definition in its module, a visible function of the module.
    pub function: Box<[u8]>,
    /// The function was made specialized (`newFunctionWithName:constantValues:`, or
    /// `newFunctionWithDescriptor:` with or without values), as against `newFunctionWithName:`:
    /// a constant of its module without a value is then undefined rather than unresolved. A
    /// function with constant values is specialized whatever this says.
    pub specialized: bool,
    /// The function-constant values it was specialized with, by index, or why they do not apply
    /// to its module.
    pub constants: Result<BTreeMap<u32, ConstValue>, Error>,
    /// The word of a table entry holding this function
    /// (the translator's `FunctionHandle`), when the pipeline can place it in
    /// a visible-function table (it is in `functions` or `binary_functions`).
    pub handle: Option<u32>,
}

/// A function a call through a visible-function table may call: its table word, its name
/// (diagnostics), and its imported definition, or why it cannot be imported (raised only at a
/// surviving call that may call it).
#[derive(Clone, Debug)]
pub(super) struct Candidate {
    pub word: u32,
    pub name: Box<[u8]>,
    pub def: Result<FuncId, Unimported>,
}

/// Why a [`Candidate`] was not imported, and which calls it could answer.
#[derive(Clone, Debug)]
pub(super) struct Unimported {
    pub error: Error,
    /// The function's type in its own module (that module's index among the linked modules),
    /// when the module decoded and has a function of the candidate's name: only a call of that
    /// type (up to struct identity) may call it. `None` when its type is unknown, so any call may.
    pub ty: Option<(usize, TypeId)>,
}

/// A decoded linked module and its ABI.
#[derive(Clone, Debug)]
pub struct LinkedCode {
    /// The decoded, verified module.
    pub module: Module,
    /// Its metadata ABI, cross-checked against `module`.
    pub abi: AirAbi,
}

/// One dynamic library a pipeline stage links (`MTLDynamicLibrary`): its install name and its
/// decoded module.
#[derive(Clone, Debug)]
pub struct DynamicModule {
    /// The install name it was loaded under, for diagnostics.
    pub install_name: Box<[u8]>,
    /// The decoded module, or why its bitcode does not decode: refused at a surviving use of an
    /// import whose search reaches it (whether it exports the symbol is unknown).
    pub code: Result<LinkedCode, Error>,
}

/// The dynamic libraries a pipeline stage's imports resolve against.
#[derive(Clone, Debug, Default)]
pub struct DynamicLinks {
    /// The libraries in the order a symbol is searched for: the stage's preloaded libraries,
    /// then those its install names load.
    pub libraries: Vec<DynamicModule>,
    /// Why an install name the stage's libraries name was not loaded, when one was not: every
    /// import's surviving use is refused with it, and nothing is searched.
    pub unloaded: Option<Error>,
}

impl DynamicLinks {
    /// Whether an import can be searched for: some library is listed and every install name
    /// loaded.
    fn searchable(&self) -> bool {
        !self.libraries.is_empty() && self.unloaded.is_none()
    }
}

/// What linking made of one stub declaration.
#[derive(Clone, Debug)]
pub(super) enum Stub {
    /// Its calls now call this imported definition.
    Resolved(FuncId),
    /// No linked function has the name it references.
    Missing(VisibleSymbol),
    /// The linked function of its name cannot resolve it: a surviving call is refused with this.
    Refused(Error),
}

/// A function or a variable of a module.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Item {
    Func(FuncId),
    Global(GlobalId),
}

/// The state of every stub declaration of the linked module, by function, and of its
/// dynamic-library imports.
#[derive(Debug, Default)]
pub(super) struct Stubs {
    of: BTreeMap<FuncId, Stub>,
    /// Every dynamic-library import of the linked module: a declaration a flat table names.
    imports: BTreeSet<Item>,
    /// The function imports a library resolved: each one's imported definition.
    dynamic: BTreeMap<FuncId, FuncId>,
    /// The imports whose search refused: a surviving use is refused with this.
    refused: BTreeMap<Item, Error>,
    /// Why an install name was not loaded, when one was not (`DynamicLinks::unloaded`).
    unloaded: Option<Error>,
}

impl Stubs {
    /// The state of `f`, if it is a stub declaration.
    pub fn get(&self, f: FuncId) -> Option<&Stub> {
        self.of.get(&f)
    }

    /// The function a call to `f` calls: the definition a stub or a function import of that id
    /// was resolved to, or `f`.
    pub fn target(&self, f: FuncId) -> FuncId {
        match self.of.get(&f) {
            Some(Stub::Resolved(def)) => *def,
            _ => self.dynamic.get(&f).copied().unwrap_or(f),
        }
    }

    /// Whether some import is a variable: only then can a use of a declared one survive.
    pub fn imports_variables(&self) -> bool {
        self.imports.iter().any(|i| matches!(i, Item::Global(_)))
    }

    /// Why a surviving use of the declaration `item`, of symbol `symbol`, is refused when it is
    /// a dynamic-library import: its search's refusal, the install name not loaded, or that no
    /// library exports it. `None` when it is no import.
    pub fn import_refusal(&self, item: Item, symbol: &[u8]) -> Option<Error> {
        if !self.imports.contains(&item) {
            return None;
        }
        if let Some(e) = self.refused.get(&item) {
            return Some(e.clone());
        }
        Some(match &self.unloaded {
            Some(e) => e.clone(),
            None => Error::NeedsInput(MissingInput::DynamicSymbol {
                symbol: diag(symbol),
            }),
        })
    }

    /// The linked definitions the stubs resolved to.
    pub fn resolved(&self) -> impl Iterator<Item = FuncId> + '_ {
        self.of.values().filter_map(|s| match s {
            Stub::Resolved(f) => Some(*f),
            Stub::Missing(_) | Stub::Refused(_) => None,
        })
    }

    /// Record the declaration `f` as one whose surviving call is refused with `e` (a candidate a
    /// table call may call that cannot be imported, or a table call nothing can answer).
    pub fn refuse(&mut self, f: FuncId, e: Error, m: &mut Meter) -> Result<(), Error> {
        m.charge(std::mem::size_of::<(FuncId, Stub)>())?;
        self.of.insert(f, Stub::Refused(e));
        Ok(())
    }
}

fn diag(bytes: &[u8]) -> DiagName {
    DiagName::new(String::from_utf8_lossy(bytes))
}

fn link_problem(name: &[u8], problem: LinkProblem) -> Error {
    Error::InvalidRequest(RequestError::LinkedFunction {
        name: diag(name),
        problem,
    })
}

/// A module feature linking does not implement yet, for the linked function `name`, at its
/// metadata `key`.
pub(super) fn not_implemented(name: &[u8], key: &'static str) -> Error {
    Unsupported::not_implemented(Feature::Linking)
        .naming(diag(name))
        .at(Location::metadata(Some(diag(name)), None, Some(key)))
        .into()
}

/// The name of the global listing a module's dynamic-library imports: AIR linker ABI.
const FLAT_TABLE: &[u8] = b"air.dyld_flat_table";

/// Whether `module` imports symbols from dynamic libraries: it has the `air.dyld_flat_table` the
/// AIR linker writes into a module it links against them. A module without one links no dynamic
/// library.
pub fn imports_dynamically(module: &Module) -> bool {
    let parts = module.parts();
    parts
        .globals
        .values()
        .any(|g| parts.symbols.get(g.name) == Some(FLAT_TABLE))
}

/// The declarations `parts` imports from dynamic libraries: those its `air.dyld_flat_table` names,
/// sorted.
fn imports(parts: &ModuleParts, m: &mut Meter) -> Result<Vec<Item>, Error> {
    let mut table = None;
    for g in parts.globals.values() {
        m.tick(1)?;
        if parts.symbols.get(g.name) == Some(FLAT_TABLE) {
            table = g.init;
            break;
        }
    }
    let Some(init) = table else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    let mut seen = BTreeSet::new();
    let mut stack = vec![init];
    let mut bytes = 0usize;
    let walked: Result<(), Error> = (|| {
        while let Some(c) = stack.pop() {
            m.tick(1)?;
            if !seen.insert(c) {
                continue;
            }
            let k = parts
                .consts
                .get(c)
                .ok_or_else(|| invariant("constant out of range"))?;
            let step = std::mem::size_of::<ConstId>() * (k.value.children().len() + 3);
            m.charge(step)?;
            bytes += step;
            match k.value {
                Const::FuncAddr(f) if parts.funcs.get(f).is_some_and(|f| f.body.is_none()) => {
                    m.charge(std::mem::size_of::<Item>())?;
                    out.push(Item::Func(f));
                }
                Const::GlobalAddr(g) if parts.globals.get(g).is_some_and(|g| g.init.is_none()) => {
                    m.charge(std::mem::size_of::<Item>())?;
                    out.push(Item::Global(g));
                }
                _ => stack.extend_from_slice(k.value.children()),
            }
        }
        Ok(())
    })();
    m.release(bytes);
    walked?;
    out.sort();
    out.dedup();
    Ok(out)
}

/// What linking made of a module: its parts, the state of each stub, the constexpr samplers of
/// the imported globals, and the candidates of its calls through visible-function tables.
pub(super) type Linked = (ModuleParts, Stubs, LinkedSamplers, Vec<Candidate>);

/// The constexpr samplers linking brought: one group per import that brought any, in the order of
/// the imports, each in its module's order (`!air.sampler_states`), with where it came from; then
/// the states of each function the pipeline can place in a table that nothing imported (no stub
/// names it and the entry reaches no table), in its module's order, and whether such a function's
/// are unknown (it could not be planned). A pipeline state numbers the samplers of the functions it
/// links, a function a table can hold called or not.
#[derive(Clone, Debug, Default)]
pub(super) struct LinkedSamplers {
    pub groups: Vec<(Vec<StaticSampler>, SamplerOrigin)>,
    pub unimported: Vec<Vec<SamplerState>>,
    pub unknown: bool,
}

/// What brought a group of linked constexpr samplers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SamplerOrigin {
    /// A linked function the pipeline can place in a table (of `functions` or
    /// `binary_functions`): Metal numbers its constexpr samplers called or not.
    Table,
    /// A linked function of `privateFunctions`: Metal numbers its constexpr samplers as far as
    /// the stage reaches them, as the stage's own.
    Private,
    /// A dynamic library's symbol.
    Library,
}

/// Resolve the stubs of `parts` (whose ABI is `abi`) against `linked`, importing each linked
/// function a stub names, and with `tables` (the entry can call through a visible-function
/// table: one it takes, or one an argument buffer it takes holds) every function a table can hold
/// as a [`Candidate`]; then its dynamic-library imports against `dynamic`. A module without any
/// is returned unchanged.
pub(super) fn link(
    parts: ModuleParts,
    abi: &AirAbi,
    (linked, dynamic): (&[LinkedModule], &DynamicLinks),
    tables: bool,
    m: &mut Meter,
) -> Result<Linked, Error> {
    let mut queue = VecDeque::new();
    for (f, symbol) in abi.visible_function_stubs() {
        m.tick(1)?;
        let declared = parts.funcs.get(f).is_some_and(|f| f.body.is_none());
        if declared {
            m.charge(std::mem::size_of::<(FuncId, VisibleSymbol)>() + symbol.0.len())?;
            queue.push_back((f, symbol.clone()));
        }
    }
    let candidates = tables
        && linked
            .iter()
            .any(|l| l.functions.iter().any(|f| f.handle.is_some()));
    let client = imports(&parts, m)?;
    if queue.is_empty() && !candidates && (client.is_empty() || !dynamic.searchable()) {
        let stubs = Stubs {
            imports: client.into_iter().collect(),
            unloaded: dynamic.unloaded.clone(),
            ..Stubs::default()
        };
        let (unimported, unknown) = scoped(m, |m, held| {
            unimported_samplers(linked, |_| false, &parts.data_layout, m, held)
        })?;
        let samplers = LinkedSamplers {
            groups: Vec::new(),
            unimported,
            unknown,
        };
        return Ok((parts, stubs, samplers, Vec::new()));
    }
    scoped(m, |m, held| {
        link_in(
            parts,
            queue,
            (linked, dynamic),
            (candidates, client),
            m,
            held,
        )
    })
}

/// The destination tables an import moves through.
struct Tables {
    symbols: Symbols,
    types: TypeTable,
    consts: ConstTable,
    /// The destination's identified structs by body, built at the first import.
    bodies: Option<Bodies>,
}

fn link_in(
    parts: ModuleParts,
    mut queue: VecDeque<(FuncId, VisibleSymbol)>,
    (linked, dynamic): (&[LinkedModule], &DynamicLinks),
    (mut candidates_left, client): (bool, Vec<Item>),
    m: &mut Meter,
    held: &mut usize,
) -> Result<Linked, Error> {
    let ModuleParts {
        symbols,
        types,
        consts,
        mut globals,
        mut funcs,
        metadata,
        named_md,
        triple,
        source_filename,
        data_layout,
    } = parts;
    let mut tables = Tables {
        symbols,
        types,
        consts,
        bodies: None,
    };
    let mut stubs = Stubs {
        unloaded: dynamic.unloaded.clone(),
        ..Stubs::default()
    };
    let mut libs = Libraries::new(dynamic, m, held)?;
    for item in client {
        let sym = match item {
            Item::Func(f) => funcs.get(f).map(|f| f.name),
            Item::Global(g) => globals.get(g).map(|g| g.name),
        };
        let name = sym
            .and_then(|s| tables.symbols.get(s))
            .ok_or_else(|| invariant("an import without its name"))?;
        libs.declare(item, name, &mut stubs, m, held)?;
    }
    // The constexpr samplers of the imported globals, in the module's ids, one group per import.
    let mut samplers = Vec::new();
    // Per linked function (module, function): its imported definition, or why it cannot be
    // imported.
    let mut imported: BTreeMap<(usize, usize), Result<FuncId, Error>> = BTreeMap::new();
    let mut candidates = Vec::new();
    loop {
        let Some((stub, symbol)) = queue.pop_front() else {
            // The stubs are resolved: import the table candidates (once, after the stubs, so a
            // function a stub also names is imported at the stub's type), then resolve the stubs
            // they bring. Then search the dynamic libraries for each import, which can bring
            // stubs and imports in turn.
            if !candidates_left {
                let Some(item) = libs.queue.pop_front() else {
                    break;
                };
                let brought = Brought {
                    layout: &data_layout,
                    samplers: &mut samplers,
                    queue: &mut queue,
                    libs: &mut libs,
                    stubs: &mut stubs,
                    table: false,
                };
                tables = resolve(item, tables, (&mut globals, &mut funcs), brought, m, held)?;
                continue;
            }
            candidates_left = false;
            for (i, l) in linked.iter().enumerate() {
                for (j, f) in l.functions.iter().enumerate() {
                    m.tick(1)?;
                    let Some(word) = f.handle else {
                        continue;
                    };
                    let def = match imported.get(&(i, j)).cloned() {
                        Some(done) => done,
                        None => {
                            let brought = Brought {
                                layout: &data_layout,
                                samplers: &mut samplers,
                                queue: &mut queue,
                                libs: &mut libs,
                                stubs: &mut stubs,
                                table: true,
                            };
                            let (done, t) = import_linked(
                                (l, f, None),
                                tables,
                                (&mut globals, &mut funcs),
                                brought,
                                m,
                                held,
                            )?;
                            tables = t;
                            m.charge(
                                std::mem::size_of::<((usize, usize), Result<FuncId, Error>)>(),
                            )?;
                            imported.insert((i, j), done.clone());
                            done
                        }
                    };
                    let def = match def {
                        Ok(def) => Ok(def),
                        Err(error) => Err(Unimported {
                            error,
                            ty: source_type(l, &f.function, m)?.map(|t| (i, t)),
                        }),
                    };
                    m.charge(std::mem::size_of::<Candidate>() + f.name.len())?;
                    candidates.push(Candidate {
                        word,
                        name: f.name.clone(),
                        def,
                    });
                }
            }
            continue;
        };
        m.tick(1)?;
        if stubs.of.contains_key(&stub) {
            continue;
        }
        let mut named = linked.iter().enumerate().flat_map(|(i, l)| {
            l.functions
                .iter()
                .enumerate()
                .filter(|(_, f)| *f.name == *symbol.0)
                .map(move |(j, f)| ((i, j), l, f))
        });
        let state = match (named.next(), named.next()) {
            (None, _) => Stub::Missing(symbol),
            (Some(_), Some(_)) => Stub::Refused(link_problem(&symbol.0, LinkProblem::Ambiguous)),
            (Some((at, l, f)), None) => {
                let def = match imported.get(&at).cloned() {
                    Some(done) => done,
                    None => {
                        let want = funcs
                            .get(stub)
                            .map(|f| f.ty)
                            .ok_or_else(|| invariant("a stub out of range"))?;
                        let brought = Brought {
                            layout: &data_layout,
                            samplers: &mut samplers,
                            queue: &mut queue,
                            libs: &mut libs,
                            stubs: &mut stubs,
                            table: f.handle.is_some(),
                        };
                        let (done, t) = import_linked(
                            (l, f, Some(want)),
                            tables,
                            (&mut globals, &mut funcs),
                            brought,
                            m,
                            held,
                        )?;
                        tables = t;
                        m.charge(std::mem::size_of::<((usize, usize), Result<FuncId, Error>)>())?;
                        imported.insert(at, done.clone());
                        done
                    }
                };
                match def {
                    Err(refusal) => Stub::Refused(refusal),
                    Ok(def) => {
                        let ty = |f: FuncId| {
                            funcs
                                .get(f)
                                .map(|f| f.ty)
                                .ok_or_else(|| invariant("a linked function out of range"))
                        };
                        let (want, got) = (ty(stub)?, ty(def)?);
                        let types = (&tables.types, &tables.types);
                        if want == got {
                            Stub::Resolved(def)
                        } else if isomorphism(types, (got, want), |_| None, m)?.is_some() {
                            // Another reference to a definition already imported for a reference
                            // of a different but isomorphic type (two identified structs of one
                            // body): the call would need the value converted between them.
                            Stub::Refused(not_implemented(&symbol.0, "type"))
                        } else {
                            Stub::Refused(link_problem(&symbol.0, LinkProblem::Signature))
                        }
                    }
                }
            }
        };
        m.charge(std::mem::size_of::<(FuncId, Stub)>())?;
        stubs.of.insert(stub, state);
    }
    redirect(&mut funcs, &stubs, m)?;
    let (unimported, unknown) = unimported_samplers(
        linked,
        |at| imported.contains_key(&at),
        &data_layout,
        m,
        held,
    )?;
    let Tables {
        symbols,
        types,
        consts,
        ..
    } = tables;
    Ok((
        ModuleParts {
            symbols,
            types,
            consts,
            globals,
            funcs,
            metadata,
            named_md,
            triple,
            source_filename,
            data_layout,
        },
        stubs,
        LinkedSamplers {
            groups: samplers,
            unimported,
            unknown,
        },
        candidates,
    ))
}

/// The constexpr sampler states of each function of `linked` the pipeline can place in a table
/// (it has a table word) that `imported` (by module and function index) says was not imported,
/// where its module has constexpr samplers: those it reaches, in its module's order, as an import
/// of it would bring them (a function that brings none is left out); and whether such a function
/// could not be planned (its import would be refused), so its samplers are unknown
/// ([`LinkedSamplers::unimported`], [`LinkedSamplers::unknown`]). `layout` is the destination's.
fn unimported_samplers(
    linked: &[LinkedModule],
    imported: impl Fn((usize, usize)) -> bool,
    layout: &DataLayout,
    m: &mut Meter,
    held: &mut usize,
) -> Result<(Vec<Vec<SamplerState>>, bool), Error> {
    let (mut groups, mut unknown) = (Vec::new(), false);
    for (i, l) in linked.iter().enumerate() {
        let Ok(code) = &l.code else {
            continue;
        };
        if code.abi.static_samplers.is_empty() {
            continue;
        }
        let mut src_imports = None;
        for (j, f) in l.functions.iter().enumerate() {
            m.tick(1)?;
            if f.handle.is_none() || imported((i, j)) {
                continue;
            }
            let src_imports = match &src_imports {
                Some(items) => items,
                None => {
                    let items = imports(code.module.parts(), m)?;
                    *held += items.len() * std::mem::size_of::<Item>();
                    src_imports.insert(items)
                }
            };
            let (source, plan) = match plan(code, f, src_imports, layout, m, held)? {
                Ok(planned) => planned,
                Err(_) => {
                    unknown = true;
                    continue;
                }
            };
            source.release(m);
            let mut states = Vec::new();
            for s in &code.abi.static_samplers {
                m.tick(1)?;
                if plan.keep_g.get(s.global.index()).copied().unwrap_or(false) {
                    m.charge(std::mem::size_of::<SamplerState>())?;
                    *held += std::mem::size_of::<SamplerState>();
                    states.push(s.state);
                }
            }
            if !states.is_empty() {
                groups.push(states);
            }
        }
    }
    Ok((groups, unknown))
}

/// The type of the function `name` of `l`'s module, when the module decoded and has one.
fn source_type(l: &LinkedModule, name: &[u8], m: &mut Meter) -> Result<Option<TypeId>, Error> {
    let Ok(code) = &l.code else {
        return Ok(None);
    };
    for f in code.module.parts().funcs.values() {
        m.tick(1)?;
        if code.module.symbols().get(f.name) == Some(name) {
            return Ok(Some(f.ty));
        }
    }
    Ok(None)
}

/// Where an import records what it brings besides the definition.
struct Brought<'s, 'a> {
    /// The destination's data layout, which a linked module must share.
    layout: &'s DataLayout,
    /// The constexpr samplers of the imported globals, one group per import that brought any
    /// (and what brought it).
    samplers: &'s mut Vec<(Vec<StaticSampler>, SamplerOrigin)>,
    /// The stubs still to resolve.
    queue: &'s mut VecDeque<(FuncId, VisibleSymbol)>,
    /// The dynamic libraries' search, and the imports still to search for.
    libs: &'s mut Libraries<'a>,
    /// The stubs' and the imports' states.
    stubs: &'s mut Stubs,
    /// The import is of a linked function the pipeline can place in a table.
    table: bool,
}

/// Import the linked function `f` of `l` (at the type `want` a stub references it with, if any)
/// with what it reaches, queueing its own stubs and imports: its definition, or why it cannot be
/// imported.
fn import_linked(
    (l, f, want): (&LinkedModule, &LinkedName, Option<TypeId>),
    tables: Tables,
    dest: (&mut Arena<GlobalId, Global>, &mut Arena<FuncId, Function>),
    brought: Brought<'_, '_>,
    m: &mut Meter,
    held: &mut usize,
) -> Result<(Result<FuncId, Error>, Tables), Error> {
    let code = match &l.code {
        Err(e) => return Ok((Err(e.clone()), tables)),
        Ok(code) => code,
    };
    let src_imports = imports(code.module.parts(), m)?;
    *held += src_imports.len() * std::mem::size_of::<Item>();
    let (source, plan) = match plan(code, f, &src_imports, brought.layout, m, held)? {
        Err(refusal) => return Ok((Err(refusal), tables)),
        Ok(planned) => planned,
    };
    let brought = bring(
        (source.code(), &plan, want, None, &src_imports),
        tables,
        dest,
        brought,
        m,
        held,
    );
    source.release(m);
    let (root, t) = brought?;
    match root {
        Item::Func(def) => Ok((Ok(def), t)),
        Item::Global(_) => Err(invariant("a linked function imported as a variable")),
    }
}

/// What one import brings in: the source module's root (a linked function, or a dynamic
/// library's definition), and which of the module's globals and functions it reaches.
struct Plan {
    root: Item,
    keep_g: Vec<bool>,
    keep_f: Vec<bool>,
}

/// The code an import of a linked function reads: its module as decoded, or a copy of it
/// specialized with the function's values ([`specialize`]), with the ledger bytes the copy holds.
enum Source<'c> {
    Module(&'c LinkedCode),
    Specialized(Box<LinkedCode>, usize),
}

impl Source<'_> {
    fn code(&self) -> &LinkedCode {
        match self {
            Source::Module(code) => code,
            Source::Specialized(code, _) => code,
        }
    }

    /// Drop the source, returning a specialized copy's bytes to the ledger.
    fn release(self, m: &mut Meter) {
        if let Source::Specialized(code, bytes) = self {
            drop(code);
            m.release(bytes);
        }
    }
}

/// `e` as a linked function's refusal (`Ok(Err)`), raised only at a surviving call; a budget stop,
/// a cancellation, an attempt-wide cap or an internal error returns at once.
fn refusal<T>(e: Error) -> Result<Result<T, Error>, Error> {
    match e {
        Error::Budget(_)
        | Error::Cancelled
        | Error::Internal(_)
        | Error::LimitExceeded(Limit::WorkUnits | Limit::LedgerBytes) => Err(e),
        e => Ok(Err(e)),
    }
}

/// Decide the import of `f` from `l` (whose dynamic-library imports are `imports`): the code it
/// imports from and what it brings, `Ok(Err(refusal))` when it cannot be linked (the stub's
/// state), an outer error for a budget stop. Its definition is the visible function of `l` named
/// `f.function`, whatever name it links under. A module that declares function constants or
/// defines a static initializer is specialized for `f` first ([`specialize`]). The destination
/// ids are assigned by [`import`].
fn plan<'c>(
    l: &'c LinkedCode,
    f: &LinkedName,
    imports: &[Item],
    layout: &DataLayout,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Result<(Source<'c>, Plan), Error>, Error> {
    let name = &*f.name;
    if l.abi.linkage_signature.is_some() {
        return Ok(Err(not_implemented(name, "air.signature")));
    }
    let constants = match &f.constants {
        Ok(c) => c,
        Err(e) => return Ok(Err(e.clone())),
    };
    if l.module.data_layout() != layout {
        return Ok(Err(link_problem(name, LinkProblem::DataLayout)));
    }
    let visible = l.abi.visible_functions.iter().find(|v| {
        l.module
            .func(v.func)
            .and_then(|f| l.module.symbols().get(f.name))
            == Some(&*f.function)
    });
    let Some(func) = visible.map(|v| v.func) else {
        return Ok(Err(link_problem(name, LinkProblem::NotVisible)));
    };
    if l.module.func(func).is_none_or(|f| f.body.is_none()) {
        return Ok(Err(link_problem(name, LinkProblem::NotVisible)));
    }
    if l.abi.function_constants.is_empty() && !static_initializer(l) {
        if let Some(&index) = constants.keys().next() {
            // The module declares no constant, so no index is declared.
            return Ok(Err(Error::InvalidRequest(
                RequestError::FunctionConstantIndex { index },
            )));
        }
        let plan = reach(l.module.parts(), Item::Func(func), imports, m, held)?;
        return Ok(plan.map(|plan| (Source::Module(l), plan)));
    }
    specialize((l, f, constants, func), imports, m, held)
}

/// Whether `l` defines a static initializer.
fn static_initializer(l: &LinkedCode) -> bool {
    l.module
        .parts()
        .funcs
        .values()
        .any(|f| f.section == Some(Section::StaticInit) && f.body.is_some())
}

/// The import of `f`, the visible function `func` of `l`, from a copy of `l` specialized as Metal
/// specializes an `MTLFunction` before it links it: `constants` bound to the module's function
/// constants and its static initializer run over them, which is where the compiler copies each
/// constant into the global its functions read. Measured on an Apple M3 Max under macOS 27.0.1
/// (`linked_function_specialized_name.m`), a visible function reading `[[function_constant(0)]]`
/// through that copy, one compute pipeline linking it per shape:
///
/// - made specialized ([`LinkedName::specialized`]: `newFunctionWithName:constantValues:` or
///   `newFunctionWithDescriptor:`), with the value: the pipeline computes with it; without it
///   (empty values, or a descriptor with none): the pipeline is made, and the constant's reads are
///   undefined (the kernel's store of the result did not land);
/// - made with `newFunctionWithName:`: the creation fails, "cannot lower module with unresolved
///   function constants", even where only an `is_function_constant_defined` guard reaches the
///   constant; a function that reaches no constant links.
///
/// So a specialized function's copy has what the values decide folded (SCCP over each function
/// the import reaches), a constant without a value reading as the entry's undefined constants
/// read; an unspecialized function's copy folds nothing, and a constant the import reaches
/// without a value, or a global the static initializer computed from one, refuses it
/// ([`LinkProblem::UnresolvedConstant`], the lowest index). The copy's ids are the module's, so
/// the plan's marks name the module's globals and functions too.
fn specialize<'c>(
    (l, f, constants, func): (&LinkedCode, &LinkedName, &BTreeMap<u32, ConstValue>, FuncId),
    imports: &[Item],
    m: &mut Meter,
    held: &mut usize,
) -> Result<Result<(Source<'c>, Plan), Error>, Error> {
    let (pending, mut fcs) = match fc::check(&l.module, &l.abi, constants, m) {
        Ok(checked) => checked,
        Err(e) => return refusal(e),
    };
    let mut charged = copy_bytes(l.module.parts());
    m.charge(charged)?;
    let made = specialized_copy(
        (l, f, &pending, &mut fcs),
        func,
        imports,
        &mut charged,
        m,
        held,
    );
    m.release(vec_bytes(&pending) + fcs.len() * std::mem::size_of::<(GlobalId, FcState)>() * 2);
    match made {
        Ok(Ok((code, plan))) => Ok(Ok((Source::Specialized(Box::new(code), charged), plan))),
        Ok(Err(refused)) => {
            m.release(charged);
            Ok(Err(refused))
        }
        Err(e) => {
            m.release(charged);
            Err(e)
        }
    }
}

/// [`specialize`]'s copy of `l` and the plan of `f`'s import from it. The bodies SCCP builds are
/// added to `charged`.
fn specialized_copy(
    (l, f, pending, fcs): (
        &LinkedCode,
        &LinkedName,
        &[fc::Pending<'_>],
        &mut HashMap<GlobalId, FcState>,
    ),
    func: FuncId,
    imports: &[Item],
    charged: &mut usize,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Result<(LinkedCode, Plan), Error>, Error> {
    let mut parts = l.module.parts().clone();
    if let Err(e) = fc::bind(
        &parts.types,
        &mut parts.consts,
        &mut parts.globals,
        pending,
        m,
    ) {
        return refusal(e);
    }
    let evaluated = match static_init::evaluate(&mut parts, fcs, Unevaluable::Refuse, m) {
        Ok(evaluated) => evaluated,
        Err(e) => return refusal(e),
    };
    for (g, index) in evaluated.derived {
        m.charge(std::mem::size_of::<(GlobalId, FcState)>() * 2)?;
        fcs.insert(
            g,
            FcState {
                index,
                supplied: false,
                decl: false,
            },
        );
    }
    let root = Item::Func(func);
    let plan = match reach(&parts, root, imports, m, held)? {
        Ok(plan) => plan,
        Err(refused) => return Ok(Err(refused)),
    };
    // Values come only with a specialized function.
    let plan = if f.specialized || !pending.is_empty() {
        for (k, &kept) in plan.keep_f.iter().enumerate() {
            m.tick(1)?;
            if !kept {
                continue;
            }
            let id = FuncId::from_index(k).ok_or_else(|| invariant("ids"))?;
            let ModuleParts {
                types,
                consts,
                globals,
                funcs,
                ..
            } = &mut parts;
            let Some(body) = funcs.get(id).and_then(|f| f.body.as_ref()) else {
                continue;
            };
            let mut cx = sccp::Context {
                folder: Folder {
                    types,
                    consts,
                    held: 0,
                },
                globals,
                fcs,
                name: diag(&f.name),
            };
            let solved = match sccp::run(body, &mut cx, m) {
                Ok(solved) => solved,
                Err(e) => return refusal(e),
            };
            *charged += solved.body.footprint();
            if let Some(func) = funcs.get_mut(id) {
                func.body = Some(solved.body);
            }
        }
        match reach(&parts, root, imports, m, held)? {
            Ok(plan) => plan,
            Err(refused) => return Ok(Err(refused)),
        }
    } else {
        let unresolved = plan
            .keep_g
            .iter()
            .enumerate()
            .filter(|&(_, &kept)| kept)
            .filter_map(|(k, _)| GlobalId::from_index(k).and_then(|g| fcs.get(&g)))
            .filter(|state| !state.supplied)
            .map(|state| state.index)
            .min();
        if let Some(index) = unresolved {
            return Ok(Err(link_problem(
                &f.name,
                LinkProblem::UnresolvedConstant { index },
            )));
        }
        plan
    };
    let module = match Module::new(parts, m) {
        Ok(module) => module,
        Err(e) => return refusal(e),
    };
    let abi = l.abi.clone();
    Ok(Ok((LinkedCode { module, abi }, plan)))
}

/// The ledger bytes a copy of `parts` holds: its function bodies
/// ([`crate::ir::Body::footprint`]), its arenas, and its interned constants, types and strings,
/// each stored twice (in its arena and as its lookup key).
fn copy_bytes(parts: &ModuleParts) -> usize {
    let bodies = parts
        .funcs
        .values()
        .filter_map(|f| f.body.as_ref())
        .fold(0usize, |n, b| n.saturating_add(b.footprint()));
    let consts = parts.consts.iter().fold(0usize, |n, (_, c)| {
        n.saturating_add(
            std::mem::size_of::<crate::ir::Constant>()
                + std::mem::size_of::<ConstId>() * c.value.children().len(),
        )
    });
    let strings = (0..parts.symbols.len())
        .filter_map(Sym::from_index)
        .filter_map(|s| parts.symbols.get(s))
        .fold(0usize, |n, b| n.saturating_add(b.len()));
    let types = parts
        .types
        .len()
        .saturating_mul(std::mem::size_of::<Type>() * 2 + 1);
    bodies
        .saturating_add(parts.funcs.capacity_bytes())
        .saturating_add(parts.globals.capacity_bytes())
        .saturating_add(parts.metadata.capacity_bytes())
        .saturating_add(consts.saturating_mul(2))
        .saturating_add(strings.saturating_mul(2))
        .saturating_add(types)
}

/// Mark what `root` of the module `parts` reaches: the functions it calls or takes the address of,
/// transitively, and the constants and globals their bodies and initializers name. A global the
/// module only declares and does not import (`imports`) is refused (`Ok(Err)`): it is defined
/// nowhere linking reads.
fn reach(
    parts: &ModuleParts,
    root: Item,
    imports: &[Item],
    m: &mut Meter,
    held: &mut usize,
) -> Result<Result<Plan, Error>, Error> {
    let mut keep_f = filled(m, held, parts.funcs.len(), false)?;
    let mut keep_g = filled(m, held, parts.globals.len(), false)?;
    let mut keep_c = filled(m, held, parts.consts.len(), false)?;
    let mut fstack = Vec::new();
    let mut cstack: Vec<ConstId> = Vec::new();
    match root {
        Item::Func(f) => {
            let k = keep_f
                .get_mut(f.index())
                .ok_or_else(|| invariant("the root out of range"))?;
            *k = true;
            fstack.push(f);
        }
        Item::Global(g) => {
            let k = keep_g
                .get_mut(g.index())
                .ok_or_else(|| invariant("the root out of range"))?;
            *k = true;
            cstack.extend(parts.globals.get(g).and_then(|g| g.init));
        }
    }
    loop {
        while let Some(c) = cstack.pop() {
            m.tick(1)?;
            let slot = keep_c
                .get_mut(c.index())
                .ok_or_else(|| invariant("constant out of range"))?;
            if *slot {
                continue;
            }
            *slot = true;
            let k = parts
                .consts
                .get(c)
                .ok_or_else(|| invariant("constant out of range"))?;
            let bytes = std::mem::size_of_val(k.value.children());
            m.charge(bytes)?;
            *held += bytes;
            cstack.extend_from_slice(k.value.children());
            match k.value {
                Const::GlobalAddr(g) => {
                    let kg = keep_g
                        .get_mut(g.index())
                        .ok_or_else(|| invariant("global out of range"))?;
                    if !*kg {
                        *kg = true;
                        match parts.globals.get(g).map(|g| (g.init, g.name)) {
                            Some((Some(init), _)) => cstack.push(init),
                            // A dynamic-library import: the search resolves it.
                            Some((None, _)) if imports.binary_search(&Item::Global(g)).is_ok() => {}
                            // Any other global the module only declares is defined nowhere
                            // linking reads.
                            Some((None, gname)) => {
                                let gname = parts.symbols.get(gname).unwrap_or(b"?");
                                return Ok(Err(not_implemented(gname, "global")));
                            }
                            None => return Err(invariant("global out of range")),
                        }
                    }
                }
                Const::FuncAddr(g) => {
                    if let Some(k) = keep_f.get_mut(g.index()) {
                        if !*k {
                            *k = true;
                            fstack.push(g);
                        }
                    }
                }
                _ => {}
            }
        }
        let Some(f) = fstack.pop() else {
            break;
        };
        m.tick(1)?;
        let body = parts.funcs.get(f).and_then(|f| f.body.as_ref());
        for inst in body.into_iter().flat_map(|b| b.insts.values()) {
            m.tick(1)?;
            inst_refs(
                &inst.kind,
                inst.ty,
                |c| cstack.push(c),
                |g| {
                    if let Some(k) = keep_f.get_mut(g.index()) {
                        if !*k {
                            *k = true;
                            fstack.push(g);
                        }
                    }
                },
                |_| {},
            );
        }
    }
    Ok(Ok(Plan {
        root,
        keep_g,
        keep_f,
    }))
}

/// Where an import puts one item it keeps.
#[derive(Clone, Copy, Debug)]
enum Place<I> {
    /// Appended at the destination's end.
    New,
    /// Already in the destination at this id: not imported again.
    Existing(I),
    /// Replaces this destination declaration of the same symbol: a variable's definition.
    Into(I),
}

/// Where an import puts what it keeps, by source id, and the variables it brings that are their
/// symbol's definition for every import (the search finds them) with no destination global yet.
struct Places {
    globals: Vec<Place<GlobalId>>,
    funcs: Vec<Place<FuncId>>,
    found: Vec<(GlobalId, Box<[u8]>)>,
}

/// Where the import of `plan` from `code` (the dynamic library `library`, when it is one, whose
/// imports are `imports`) puts each item it keeps: an item of that library already imported is not
/// imported again; a variable import's declaration is its symbol's destination global, when there
/// is one; and a variable the library exports that the search finds for its symbol becomes that
/// destination global (it replaces the symbol's declaration), so the library's own references and
/// every import of the symbol are one object.
fn places(
    (code, plan, library, imports): (&LinkedCode, &Plan, Option<usize>, &[Item]),
    (globals, types): (&Arena<GlobalId, Global>, &TypeTable),
    libs: &mut Libraries<'_>,
    layout: &DataLayout,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Places, Error> {
    let parts = code.module.parts();
    let mut gp = filled(m, held, plan.keep_g.len(), Place::New)?;
    let mut fp = filled(m, held, plan.keep_f.len(), Place::New)?;
    let mut found = Vec::new();
    let name_of = |sym| {
        parts
            .symbols
            .get(sym)
            .ok_or_else(|| invariant("a symbol out of range"))
    };
    if let Some(i) = library {
        for (k, place) in fp.iter_mut().enumerate() {
            m.tick(1)?;
            if plan.keep_f.get(k) != Some(&true) {
                continue;
            }
            let f = FuncId::from_index(k).ok_or_else(|| invariant("ids"))?;
            if let Some(Item::Func(copy)) = libs.copies.get(&(i, Item::Func(f))) {
                *place = Place::Existing(*copy);
            }
        }
        for (k, place) in gp.iter_mut().enumerate() {
            m.tick(1)?;
            if plan.keep_g.get(k) != Some(&true) {
                continue;
            }
            let g = GlobalId::from_index(k).ok_or_else(|| invariant("ids"))?;
            if let Some(Item::Global(copy)) = libs.copies.get(&(i, Item::Global(g))) {
                *place = Place::Existing(*copy);
            }
        }
    }
    for item in imports {
        m.tick(1)?;
        let Item::Global(g) = *item else {
            continue;
        };
        let (Some(true), Some(place @ Place::New)) =
            (plan.keep_g.get(g.index()).copied(), gp.get_mut(g.index()))
        else {
            continue;
        };
        let global = parts
            .globals
            .get(g)
            .ok_or_else(|| invariant("an import out of range"))?;
        if let Some(&v) = libs.vars.get(name_of(global.name)?) {
            *place = Place::Existing(v);
        }
    }
    let Some(i) = library else {
        return Ok(Places {
            globals: gp,
            funcs: fp,
            found,
        });
    };
    for (k, place) in gp.iter_mut().enumerate() {
        m.tick(1)?;
        if plan.keep_g.get(k) != Some(&true) || !matches!(place, Place::New) {
            continue;
        }
        let g = GlobalId::from_index(k).ok_or_else(|| invariant("ids"))?;
        let global = parts
            .globals
            .get(g)
            .ok_or_else(|| invariant("global out of range"))?;
        if global.init.is_none() || !exported(global.linkage) {
            continue;
        }
        let name = name_of(global.name)?;
        let Found::At(j, Item::Global(d)) = libs.search(name, layout, m, held)? else {
            continue;
        };
        if (j, d) != (i, g) {
            continue;
        }
        match libs.vars.get(name) {
            None => {
                m.charge(std::mem::size_of::<(GlobalId, Box<[u8]>)>() + name.len())?;
                *held += std::mem::size_of::<(GlobalId, Box<[u8]>)>() + name.len();
                found.push((g, name.into()));
            }
            Some(&v) => match globals.get(v) {
                // The symbol's declaration, of this type: the definition replaces it.
                Some(decl) if decl.init.is_none() => {
                    let same = isomorphism(
                        (&parts.types, types),
                        (global.value_ty, decl.value_ty),
                        |_| None,
                        m,
                    )?;
                    // Of another type, the library's references keep their own copy, and the
                    // declaration's search refuses it.
                    if same.is_some() {
                        *place = Place::Into(v);
                    }
                }
                _ => return Err(invariant("a variable's definition imported twice")),
            },
        }
    }
    Ok(Places {
        globals: gp,
        funcs: fp,
        found,
    })
}

/// Import `plan` of `code` (from the dynamic library `library`, when it is one, whose imports are
/// `imports`), its root at the type `want` a reference has: the root's copy. The imports it
/// brings are recorded and searched for in turn, and a linked function's stubs are queued.
fn bring(
    (code, plan, want, library, imports): (
        &LinkedCode,
        &Plan,
        Option<TypeId>,
        Option<usize>,
        &[Item],
    ),
    tables: Tables,
    (globals, funcs): (&mut Arena<GlobalId, Global>, &mut Arena<FuncId, Function>),
    brought: Brought<'_, '_>,
    m: &mut Meter,
    held: &mut usize,
) -> Result<(Item, Tables), Error> {
    let Brought {
        layout,
        samplers,
        queue,
        libs,
        stubs,
        table,
    } = brought;
    let places = places(
        (code, plan, library, imports),
        (globals, &tables.types),
        libs,
        layout,
        m,
        held,
    )?;
    let parts = code.module.parts();
    // The root's type is the reference's when they are isomorphic (the check that follows the
    // import compares them), and a variable's definition takes its declaration's type.
    let mut pairs = Vec::new();
    if let Some(want) = want {
        let root = match plan.root {
            Item::Func(f) => parts.funcs.get(f).map(|f| f.ty),
            Item::Global(g) => parts.globals.get(g).map(|g| g.value_ty),
        };
        pairs.push((
            root.ok_or_else(|| invariant("the root out of range"))?,
            want,
        ));
    }
    for (k, place) in places.globals.iter().enumerate() {
        if let Place::Into(d) = place {
            let g = GlobalId::from_index(k).ok_or_else(|| invariant("ids"))?;
            let src = parts.globals.get(g).map(|g| g.value_ty);
            let dst = globals.get(*d).map(|g| g.value_ty);
            let (Some(src), Some(dst)) = (src, dst) else {
                return Err(invariant("a placed variable out of range"));
            };
            m.charge(std::mem::size_of::<(TypeId, TypeId)>())?;
            *held += std::mem::size_of::<(TypeId, TypeId)>();
            pairs.push((src, dst));
        }
    }
    let (t, gmap, fmap, imported_samplers) = import(
        (code, plan, &pairs),
        &places,
        tables,
        (globals, funcs),
        m,
        held,
    )?;
    if !imported_samplers.is_empty() {
        let origin = match (library, table) {
            (Some(_), _) => SamplerOrigin::Library,
            (None, true) => SamplerOrigin::Table,
            (None, false) => SamplerOrigin::Private,
        };
        samplers.push((imported_samplers, origin));
    }
    let is_new = |place: Option<&Place<GlobalId>>| !matches!(place, Some(Place::Existing(_)));
    if let Some(i) = library {
        let copied = std::mem::size_of::<((usize, Item), Item)>() * 2;
        for (k, d) in gmap.iter().enumerate() {
            m.tick(1)?;
            let (Some(d), true) = (d, is_new(places.globals.get(k))) else {
                continue;
            };
            let g = GlobalId::from_index(k).ok_or_else(|| invariant("ids"))?;
            m.charge(copied)?;
            *held += copied;
            libs.copies.insert((i, Item::Global(g)), Item::Global(*d));
        }
        for (k, d) in fmap.iter().enumerate() {
            m.tick(1)?;
            let (Some(d), Some(Place::New)) = (d, places.funcs.get(k)) else {
                continue;
            };
            let f = FuncId::from_index(k).ok_or_else(|| invariant("ids"))?;
            m.charge(copied)?;
            *held += copied;
            libs.copies.insert((i, Item::Func(f)), Item::Func(*d));
        }
    }
    for (g, name) in places.found {
        if let Some(Some(d)) = gmap.get(g.index()) {
            m.charge(std::mem::size_of::<(Box<[u8]>, GlobalId)>() * 2)?;
            *held += std::mem::size_of::<(Box<[u8]>, GlobalId)>() * 2;
            libs.vars.insert(name, *d);
        }
    }
    // The imports it brings: each one it appended is searched for in turn.
    for &item in imports {
        m.tick(1)?;
        let (new, sym) = match item {
            Item::Func(f) => {
                let new = match places.funcs.get(f.index()) {
                    Some(Place::New) => fmap.get(f.index()).copied().flatten(),
                    _ => None,
                };
                (new.map(Item::Func), parts.funcs.get(f).map(|f| f.name))
            }
            Item::Global(g) => {
                let new = match places.globals.get(g.index()) {
                    Some(Place::New) => gmap.get(g.index()).copied().flatten(),
                    _ => None,
                };
                (new.map(Item::Global), parts.globals.get(g).map(|g| g.name))
            }
        };
        let Some(new) = new else {
            continue;
        };
        let name = sym
            .and_then(|s| parts.symbols.get(s))
            .ok_or_else(|| invariant("an import without its name"))?;
        libs.declare(new, name, stubs, m, held)?;
    }
    // A linked function's own stubs, as imported. (A dynamic library's are not resolved
    // against the linked functions: what Metal makes of them is not measured, so a call to
    // one stays a declaration's, refused at a surviving call.)
    if library.is_none() {
        for (f, s) in code.abi.visible_function_stubs() {
            if let Some(new) = fmap.get(f.index()).copied().flatten() {
                m.charge(std::mem::size_of::<(FuncId, VisibleSymbol)>())?;
                queue.push_back((new, s.clone()));
            }
        }
    }
    let root = match plan.root {
        Item::Func(f) => fmap.get(f.index()).copied().flatten().map(Item::Func),
        Item::Global(g) => gmap.get(g.index()).copied().flatten().map(Item::Global),
    };
    Ok((
        root.ok_or_else(|| invariant("the root was not imported"))?,
        t,
    ))
}

/// Dense ids from `base` for the kept entries of `keep` that `places` appends, in order; the
/// others keep their place's id.
fn number<I: Idx>(
    keep: &[bool],
    places: &[Place<I>],
    base: usize,
) -> Result<Vec<Option<I>>, Error> {
    let mut next = base;
    let mut out = Vec::with_capacity(keep.len());
    for (k, &kept) in keep.iter().enumerate() {
        out.push(match (kept, places.get(k)) {
            (false, _) => None,
            (true, Some(Place::Existing(id) | Place::Into(id))) => Some(*id),
            (true, _) => {
                let id = I::from_index(next).ok_or_else(|| invariant("ids"))?;
                next = next.checked_add(1).ok_or_else(|| invariant("ids"))?;
                Some(id)
            }
        });
    }
    Ok(out)
}

/// Destination ids by source id.
type Maps = (Vec<Option<GlobalId>>, Vec<Option<FuncId>>);

/// Import what `plan` marks of `l` into the destination, each item where `places` puts it: the
/// appended globals and functions in their linked-module order. `pairs` maps source types to the
/// destination types they take (a reference's, a declaration's) before anything is renumbered.
/// Returns the tables, the destination id of each global and function of `l` that the import
/// kept, and the constexpr samplers of the globals it brought.
fn import(
    (l, plan, pairs): (&LinkedCode, &Plan, &[(TypeId, TypeId)]),
    places: &Places,
    tables: Tables,
    (globals, funcs): (&mut Arena<GlobalId, Global>, &mut Arena<FuncId, Function>),
    m: &mut Meter,
    held: &mut usize,
) -> Result<
    (
        Tables,
        Vec<Option<GlobalId>>,
        Vec<Option<FuncId>>,
        Vec<StaticSampler>,
    ),
    Error,
> {
    let bytes = plan.keep_g.len() * std::mem::size_of::<Option<GlobalId>>()
        + plan.keep_f.len() * std::mem::size_of::<Option<FuncId>>();
    m.charge(bytes)?;
    *held += bytes;
    let (gmap, fmap): Maps = (
        number(&plan.keep_g, &places.globals, globals.len())?,
        number(&plan.keep_f, &places.funcs, funcs.len())?,
    );
    // The module's facts about the globals it brings: its constexpr samplers, which the memory
    // analysis recognizes by global. Its other per-global facts are refused by `plan` (function
    // constants) or name no global a function reaches (entry predicates).
    let mut samplers = Vec::new();
    for s in &l.abi.static_samplers {
        m.tick(1)?;
        if matches!(
            places.globals.get(s.global.index()),
            Some(Place::Existing(_))
        ) {
            continue;
        }
        if let Some(global) = gmap.get(s.global.index()).copied().flatten() {
            m.charge(std::mem::size_of::<StaticSampler>())?;
            samplers.push(StaticSampler {
                global,
                state: s.state,
            });
        }
    }
    let parts = l.module.parts();
    let Tables {
        symbols,
        types,
        consts,
        bodies: known,
    } = tables;
    let known = match known {
        Some(known) => known,
        None => bodies(&types, m, held)?,
    };
    let syms = Syms::import(&parts.symbols, symbols, m, held)?;
    let mut r = Renumber::new(
        (&parts.types, &parts.consts),
        (types, consts),
        (&gmap, &fmap),
        syms,
        m,
        held,
    )?
    .merging(known);
    // A pair that is not isomorphic maps nothing: the definition imports at its own type, and
    // the check that follows the import refuses the reference.
    for &(src, dst) in pairs {
        r.pair(src, dst, m)?;
    }
    for (g, global) in parts.globals.iter() {
        m.tick(1)?;
        let Some(want) = gmap.get(g.index()).copied().flatten() else {
            continue;
        };
        match places.globals.get(g.index()) {
            Some(Place::Existing(_)) => {}
            Some(Place::Into(d)) => {
                let out = r.global(global, m)?;
                let slot = globals
                    .get_mut(*d)
                    .ok_or_else(|| invariant("a placed variable out of range"))?;
                *slot = out;
            }
            _ => {
                let out = r.global(global, m)?;
                if globals.push(out, m)? != want {
                    return Err(invariant("an imported global's id"));
                }
            }
        }
    }
    for (f, func) in parts.funcs.iter() {
        m.tick(1)?;
        let Some(want) = fmap.get(f.index()).copied().flatten() else {
            continue;
        };
        match places.funcs.get(f.index()) {
            Some(Place::Existing(_)) => {}
            Some(Place::Into(_)) => return Err(invariant("a function placed into another")),
            _ => {
                let out = r.function(func, m)?;
                if funcs.push(out, m)? != want {
                    return Err(invariant("an imported function's id"));
                }
            }
        }
    }
    let Renumber {
        types,
        consts,
        syms,
        bodies,
        ..
    } = r;
    let Syms::Import { into: symbols, .. } = syms else {
        return Err(invariant("an import without its symbol table"));
    };
    Ok((
        Tables {
            symbols,
            types,
            consts,
            bodies,
        },
        gmap,
        fmap,
        samplers,
    ))
}

/// Whether a definition of linkage `linkage` is exported: a library supplies it to an import.
fn exported(linkage: Linkage) -> bool {
    !matches!(
        linkage,
        Linkage::Internal | Linkage::Private | Linkage::AvailableExternally
    )
}

/// What one dynamic library exports, by symbol (with its visibility), and the metadata key of a
/// module feature linking does not implement that keeps its definitions from being imported.
struct Exports<'a> {
    by_name: BTreeMap<&'a [u8], (Item, Visibility)>,
    blocked: Option<&'static str>,
}

/// What the search for an import's symbol found.
enum Found {
    /// The first library that exports it, and its definition there.
    At(usize, Item),
    /// A library the search reached cannot answer: its module does not decode, it cannot be
    /// imported from, or its definition is hidden (which Metal's answer is not measured).
    Refused(Error),
    /// No library exports it.
    Nowhere,
}

/// The dynamic-library side of one link: the search, what it imported, and the imports still to
/// search for.
struct Libraries<'a> {
    links: &'a DynamicLinks,
    /// Each library's exports, indexed at its first search.
    exports: Vec<Option<Exports<'a>>>,
    /// The imports still to search for, when the libraries are searched.
    queue: VecDeque<Item>,
    /// Each library item imported so far, by (library, its id in the library): its copy.
    copies: BTreeMap<(usize, Item), Item>,
    /// The destination global standing for each variable import's symbol: its declaration until
    /// the definition the search finds replaces it, or that definition.
    vars: BTreeMap<Box<[u8]>, GlobalId>,
}

impl<'a> Libraries<'a> {
    fn new(links: &'a DynamicLinks, m: &mut Meter, held: &mut usize) -> Result<Self, Error> {
        let n = links.libraries.len();
        let bytes = n * std::mem::size_of::<Option<Exports<'a>>>();
        m.charge(bytes)?;
        *held += bytes;
        Ok(Libraries {
            links,
            exports: (0..n).map(|_| None).collect(),
            queue: VecDeque::new(),
            copies: BTreeMap::new(),
            vars: BTreeMap::new(),
        })
    }

    /// The decoded module of library `i`, which a search found a definition in.
    fn code(&self, i: usize) -> Result<&'a LinkedCode, Error> {
        let links: &'a DynamicLinks = self.links;
        links
            .libraries
            .get(i)
            .and_then(|l| l.code.as_ref().ok())
            .ok_or_else(|| invariant("a found library without its module"))
    }

    /// Record the destination declaration `item`, of symbol `name`, as an import: searched for
    /// in turn when the libraries are searched, and for a variable the global its symbol's
    /// references are, when the symbol has none yet.
    fn declare(
        &mut self,
        item: Item,
        name: &[u8],
        stubs: &mut Stubs,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<(), Error> {
        m.charge(std::mem::size_of::<Item>() * 2)?;
        stubs.imports.insert(item);
        if self.links.searchable() {
            m.charge(std::mem::size_of::<Item>())?;
            *held += std::mem::size_of::<Item>();
            self.queue.push_back(item);
        }
        if let Item::Global(g) = item {
            if !self.vars.contains_key(name) {
                let bytes = std::mem::size_of::<(Box<[u8]>, GlobalId)>() * 2 + name.len();
                m.charge(bytes)?;
                *held += bytes;
                self.vars.insert(name.into(), g);
            }
        }
        Ok(())
    }

    /// Search the libraries in order for the first that exports `name`.
    fn search(
        &mut self,
        name: &[u8],
        layout: &DataLayout,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<Found, Error> {
        let links: &'a DynamicLinks = self.links;
        for (i, lib) in links.libraries.iter().enumerate() {
            m.tick(1)?;
            let code = match &lib.code {
                Ok(code) => code,
                Err(e) => return Ok(Found::Refused(e.clone())),
            };
            let slot = self
                .exports
                .get_mut(i)
                .ok_or_else(|| invariant("a library out of range"))?;
            let exports = match slot {
                Some(exports) => exports,
                None => slot.insert(exports(code, layout, m, held)?),
            };
            let Some(&(item, visibility)) = exports.by_name.get(name) else {
                continue;
            };
            return Ok(match (visibility, exports.blocked) {
                (Visibility::Hidden, _) => Found::Refused(not_implemented(name, "visibility")),
                (_, Some(key)) => Found::Refused(not_implemented(name, key)),
                _ => Found::At(i, item),
            });
        }
        Ok(Found::Nowhere)
    }
}

/// Index what `code` exports, and whether its definitions can be imported into a module of
/// data layout `layout`.
fn exports<'a>(
    code: &'a LinkedCode,
    layout: &DataLayout,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Exports<'a>, Error> {
    let parts = code.module.parts();
    let entry = std::mem::size_of::<(&[u8], (Item, Visibility))>() * 2;
    let mut by_name = BTreeMap::new();
    for (id, f) in parts.funcs.iter() {
        m.tick(1)?;
        if f.body.is_none() || !exported(f.linkage) {
            continue;
        }
        if let Some(name) = parts.symbols.get(f.name) {
            m.charge(entry)?;
            *held += entry;
            by_name.insert(name, (Item::Func(id), f.visibility));
        }
    }
    for (id, g) in parts.globals.iter() {
        m.tick(1)?;
        if g.init.is_none() || !exported(g.linkage) {
            continue;
        }
        if let Some(name) = parts.symbols.get(g.name) {
            m.charge(entry)?;
            *held += entry;
            by_name.insert(name, (Item::Global(id), g.visibility));
        }
    }
    // A dynamic library has nowhere to bind a function constant: `newDynamicLibrary:` takes no
    // constant values. Measured on an Apple M3 Max under macOS 27.0.1, a library source declaring
    // `[[function_constant(0)]]`: `newDynamicLibrary:` refuses outright, "cannot lower module with
    // unresolved function constants", when the code it exports reads the constant, and succeeds
    // when nothing reads it. So the reading shape cannot reach this at all, and what blocking the
    // exports of a module that declares any constant costs is the other shape: a library Metal
    // makes, whose exports read no constant. The import's reach set (`reach`) is what would
    // separate them. What a pipeline importing from such a library then does is unmeasured: the
    // probe got the library made but not a symbol resolved through it.
    let blocked = if code.abi.linkage_signature.is_some() {
        Some("air.signature")
    } else if !code.abi.function_constants.is_empty() {
        Some("air.function_constants")
    } else if code.module.data_layout() != layout {
        Some("datalayout")
    } else if static_initializer(code) {
        Some("air.static_init")
    } else {
        None
    };
    Ok(Exports { by_name, blocked })
}

/// Search the dynamic libraries for the import `item` and resolve it: a function's calls are
/// redirected to the definition found, a variable's declaration becomes it. A search that refuses,
/// or a definition of another kind or type, is recorded as the import's refusal; one that finds
/// nothing leaves no state.
fn resolve(
    item: Item,
    tables: Tables,
    (globals, funcs): (&mut Arena<GlobalId, Global>, &mut Arena<FuncId, Function>),
    brought: Brought<'_, '_>,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Tables, Error> {
    let Brought {
        layout,
        samplers,
        queue,
        libs,
        stubs,
        ..
    } = brought;
    // The declaration's symbol and type; a variable its definition already replaced is done.
    let declared = match item {
        Item::Func(d) => funcs.get(d).map(|f| (f.name, f.ty)),
        Item::Global(d) => match globals.get(d) {
            Some(g) if g.init.is_some() => return Ok(tables),
            g => g.map(|g| (g.name, g.value_ty)),
        },
    };
    let (sym, ty) = declared.ok_or_else(|| invariant("an import out of range"))?;
    let name: Box<[u8]> = tables
        .symbols
        .get(sym)
        .ok_or_else(|| invariant("an import without its name"))?
        .into();
    m.charge(name.len())?;
    *held += name.len();
    let (i, def) = match libs.search(&name, layout, m, held)? {
        Found::Nowhere => return Ok(tables),
        Found::Refused(e) => return refuse_import(stubs, item, e, tables, m),
        Found::At(i, def) => (i, def),
    };
    let copied = libs.copies.get(&(i, def)).copied();
    match (item, def) {
        (Item::Func(d), Item::Func(_)) => {
            let (copy, tables) = match copied {
                Some(Item::Func(copy)) => (copy, tables),
                Some(Item::Global(_)) => return Err(invariant("a function's copy is a variable")),
                None => {
                    let code = libs.code(i)?;
                    let src_imports = imports(code.module.parts(), m)?;
                    *held += src_imports.len() * std::mem::size_of::<Item>();
                    let plan = match reach(code.module.parts(), def, &src_imports, m, held)? {
                        Ok(plan) => plan,
                        Err(e) => return refuse_import(stubs, item, e, tables, m),
                    };
                    let brought = Brought {
                        layout,
                        samplers: &mut *samplers,
                        queue: &mut *queue,
                        libs: &mut *libs,
                        stubs: &mut *stubs,
                        table: false,
                    };
                    let (root, t) = bring(
                        (code, &plan, Some(ty), Some(i), &src_imports),
                        tables,
                        (&mut *globals, &mut *funcs),
                        brought,
                        m,
                        held,
                    )?;
                    match root {
                        Item::Func(copy) => (copy, t),
                        Item::Global(_) => {
                            return Err(invariant("a function imported as a variable"))
                        }
                    }
                }
            };
            let got = funcs
                .get(copy)
                .map(|f| f.ty)
                .ok_or_else(|| invariant("an imported function out of range"))?;
            if got != ty {
                return refuse_import(stubs, item, not_implemented(&name, "type"), tables, m);
            }
            m.charge(std::mem::size_of::<(FuncId, FuncId)>() * 2)?;
            stubs.dynamic.insert(d, copy);
            Ok(tables)
        }
        (Item::Global(d), Item::Global(g)) => {
            // Imported already for a declaration of another type (`places`).
            if copied.is_some() {
                return refuse_import(stubs, item, not_implemented(&name, "type"), tables, m);
            }
            let code = libs.code(i)?;
            let parts = code.module.parts();
            let def_ty = parts
                .globals
                .get(g)
                .map(|g| g.value_ty)
                .ok_or_else(|| invariant("a found variable out of range"))?;
            let types = (&parts.types, &tables.types);
            if isomorphism(types, (def_ty, ty), |_| None, m)?.is_none() {
                return refuse_import(stubs, item, not_implemented(&name, "type"), tables, m);
            }
            let src_imports = imports(parts, m)?;
            *held += src_imports.len() * std::mem::size_of::<Item>();
            let plan = match reach(code.module.parts(), def, &src_imports, m, held)? {
                Ok(plan) => plan,
                Err(e) => return refuse_import(stubs, item, e, tables, m),
            };
            let brought = Brought {
                layout,
                samplers,
                queue,
                libs,
                stubs,
                table: false,
            };
            let (root, t) = bring(
                (code, &plan, None, Some(i), &src_imports),
                tables,
                (globals, funcs),
                brought,
                m,
                held,
            )?;
            if root != Item::Global(d) {
                return Err(invariant(
                    "a variable's definition did not replace its declaration",
                ));
            }
            Ok(t)
        }
        _ => refuse_import(stubs, item, not_implemented(&name, "type"), tables, m),
    }
}

/// Record `e` as the refusal of a surviving use of the import `item`.
fn refuse_import(
    stubs: &mut Stubs,
    item: Item,
    e: Error,
    tables: Tables,
    m: &mut Meter,
) -> Result<Tables, Error> {
    m.charge(std::mem::size_of::<(Item, Error)>())?;
    stubs.refused.insert(item, e);
    Ok(tables)
}

/// Redirect every direct call to a resolved stub or function import to its definition.
fn redirect(
    funcs: &mut Arena<FuncId, Function>,
    stubs: &Stubs,
    m: &mut Meter,
) -> Result<(), Error> {
    let resolves = |s: &Stub| matches!(s, Stub::Resolved(_));
    if !stubs.of.values().any(resolves) && stubs.dynamic.is_empty() {
        return Ok(());
    }
    for func in funcs.values_mut() {
        let Some(body) = func.body.as_mut() else {
            continue;
        };
        for inst in body.insts.values_mut() {
            m.tick(1)?;
            let InstKind::Call(call) = &mut inst.kind else {
                continue;
            };
            if let Callee::Direct(s) = call.callee {
                call.callee = Callee::Direct(stubs.target(s));
            }
        }
    }
    Ok(())
}
