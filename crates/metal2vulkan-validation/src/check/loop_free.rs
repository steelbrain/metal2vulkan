//! Mechanical loop-freedom of the modules one case executes, over decoded AIR.
//!
//! Every executed module is decoded once through the product's typed AIR input layer (the
//! `metal2vulkan-air` crate) from its bitcode: the retained `blob_b64` of a
//! corpus source or library module (the bytes the Metal oracle loads), or, for a row that retains
//! none (a committed public fixture or a synthetic test row), the `llvm-as` assembly of exactly the
//! text the case's `air_sha256` names. The classifier never reads sanitized text itself.
//!
//! The input is the closed set of modules one pipeline executes: the entry module plus every
//! linked module the executor loads. That set is mechanically loop-free when, starting from the
//! case's entry function (and every function the runtime may run on its own: static and
//! function-constant initializer sections, `llvm.global_ctors`),
//!
//! - every reachable function's CFG has no cycle among its blocks reachable from its entry block,
//!   and
//! - the call graph over the reachable functions is acyclic, where
//!   - a direct call to a defined function is an edge to it; a direct call to a declaration is
//!     an edge to every function of that name the other executed modules define, or, for a
//!     stub the module's `!air.visible_function_references` names, to every executed definition
//!     of the linked function it stands for,
//!   - an indirect call of the entry module whose callee is closed (a function address, or a
//!     `select` tree of function addresses and null that nothing but other such selects, null
//!     tests, and indirect calls uses; [`crate::call_targets`]) is an edge to each of its targets;
//!     any other indirect call is an edge to every possible target: every function defined in a
//!     linked module and every entry-module function whose address escapes (an instruction operand
//!     or a global initializer takes it other than as a leaf of closed callees; metadata references
//!     are not address uses),
//!   - a call to an `air.intersect*` intrinsic is an edge to every function defined in a linked
//!     intersection-function module, which the device may invoke from it,
//!   - any other `air.*` / `llvm.*` intrinsic is a leaf, as is the post-tessellation control-point
//!     fetch an entry's `air.patch_control_point_input` names (Metal synthesizes its body from the
//!     vertex descriptor), and so is a call of MetalPerformancePrimitives' `matmul2d` ABI whose
//!     constant descriptor fixes its work (Metal supplies its body at pipeline creation;
//!     `matmul2d`), and
//!   - a reachable call to anything else (a declaration no executed module defines) counts as a
//!     cycle.
//!
//! The typed decoder is exhaustive: a terminator, instruction, or record it does not model is a
//! decode refusal, which classifies as [`CycleEvidence::Undecoded`]. The two ABI facts above come
//! from the module's decoded [`AirAbi`]: its visible-function-reference stubs and its entries'
//! control-point fetches. They only *resolve* calls, so a module whose metadata ABI does not decode
//! is still classified, with neither: its calls to those declarations stay unresolved. Likewise a
//! reference through a global names no callable stub. Every gap therefore over-refuses; none can
//! run a looping case on a GPU.

use crate::call_targets::{const_funcs, ClosedCalls};
use metal2vulkan::Budget;
use metal2vulkan_air::abi::AirAbi;
use metal2vulkan_air::bitcode::PointeeTable;
use metal2vulkan_air::budget::Meter;
use metal2vulkan_air::graph::{Cfg, Graph};
use metal2vulkan_air::ir::{Callee, FuncId, InstKind, Module, Section};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

/// Mechanical loop-freedom of the modules one case executes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LoopClass {
    /// Proven acyclic by the rules in the module documentation.
    LoopFree,
    /// Not proven acyclic; the evidence names why.
    Cyclic(CycleEvidence),
}

impl LoopClass {
    pub fn is_loop_free(&self) -> bool {
        matches!(self, Self::LoopFree)
    }

    /// The evidence, when the class is cyclic.
    pub fn evidence(&self) -> Option<&CycleEvidence> {
        match self {
            Self::LoopFree => None,
            Self::Cyclic(evidence) => Some(evidence),
        }
    }
}

/// Why a module set is not mechanically loop-free. Each variant names the function or module it
/// was found in so a refusal can be traced to the AIR without re-running the classifier.
#[derive(
    Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum CycleEvidence {
    /// A cycle among the reachable blocks of one reachable function.
    CfgCycle { function: String },
    /// A call-graph cycle (recursion, direct, mutual, or through an over-approximated indirect
    /// or intrinsic edge); `function` is on the cycle.
    CallCycle { function: String },
    /// A reachable call to a function neither defined in the executed modules nor an intrinsic.
    UnresolvedCall { function: String, callee: String },
    /// The entry module defines no function named as the case's entry.
    MissingEntry { entry: String },
    /// An executed module whose bitcode could not be obtained or decoded.
    Undecoded { module: String, detail: String },
}

impl CycleEvidence {
    /// Stable short label for counting evidence in a census.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::CfgCycle { .. } => "cfg_cycle",
            Self::CallCycle { .. } => "call_cycle",
            Self::UnresolvedCall { .. } => "unresolved_call",
            Self::MissingEntry { .. } => "missing_entry",
            Self::Undecoded { .. } => "undecoded",
        }
    }
}

impl fmt::Display for CycleEvidence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CfgCycle { function } => write!(formatter, "CFG cycle in @{function}"),
            Self::CallCycle { function } => {
                write!(formatter, "call-graph cycle through @{function}")
            }
            Self::UnresolvedCall { function, callee } => write!(
                formatter,
                "@{function} calls @{callee}, which no executed module defines"
            ),
            Self::MissingEntry { entry } => {
                write!(formatter, "the entry module defines no @{entry}")
            }
            Self::Undecoded { module, detail } => {
                write!(formatter, "module {module} does not decode: {detail}")
            }
        }
    }
}

/// How the executor loads a module into the pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ModuleRole {
    /// The stage entry's own module. Its functions are indirect-call targets only when their
    /// address is taken.
    Entry,
    /// A linked visible-function module (direct reference or table entry). Every function it
    /// defines is a possible indirect-call target.
    Linked,
    /// A linked intersection-function module. Its functions are possible indirect-call targets
    /// and may also be invoked by the device from an `air.intersect*` intrinsic.
    Intersection,
}

/// Where one executed module's bitcode comes from.
#[derive(Clone, Copy, Debug)]
pub struct ModuleSource<'a> {
    /// The row's label, for evidence.
    pub label: &'a str,
    /// The row's `.ll` text, assembled only when the row retains no bitcode.
    pub air_ll: &'a str,
    /// The row's retained bitcode, base64.
    pub blob_b64: Option<&'a str>,
}

/// One executed module, decoded once, with its metadata ABI when that decodes.
pub struct DecodedModule {
    module: Module,
    /// The ABI decode's refusal when the module's metadata ABI does not decode: the module then
    /// resolves no stub and no control-point fetch (module documentation).
    abi: Result<AirAbi, String>,
}

impl DecodedModule {
    /// Decode and verify `bitcode` (a raw or wrapped AIR bitcode module) into the typed IR, then
    /// its metadata ABI (`air::abi::decode`), under the standard per-attempt budget. An IR
    /// refusal is [`CycleEvidence::Undecoded`]; an ABI refusal only withholds the ABI's call
    /// resolutions. The ABI is decoded without the typed-pointer pointee cross-checks, which
    /// concern argument layouts and carry no call-graph fact.
    pub fn from_bitcode(label: &str, bitcode: &[u8]) -> Result<DecodedModule, CycleEvidence> {
        let undecoded = |error: metal2vulkan_air::error::Error| CycleEvidence::Undecoded {
            module: label.to_string(),
            detail: metal2vulkan::error::Error::from(error).to_string(),
        };
        let mut meter = Meter::new(&Budget::standard());
        let (_, stream) =
            metal2vulkan_air::container::bitcode_stream(bitcode).map_err(undecoded)?;
        let tables = metal2vulkan_air::bitcode::decode_module(
            stream,
            metal2vulkan_air::bitcode::DecodeOptions::default(),
            &mut meter,
        )
        .map_err(undecoded)?;
        let module = Module::new(tables.parts, &mut meter).map_err(undecoded)?;
        let abi = metal2vulkan_air::abi::decode(&module, &PointeeTable::default(), &mut meter)
            .map_err(|error| error.to_string());
        Ok(DecodedModule { module, abi })
    }

    /// The typed IR.
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// The metadata ABI, or why it does not decode.
    pub fn abi(&self) -> Result<&AirAbi, &str> {
        self.abi.as_ref().map_err(String::as_str)
    }

    /// Obtain and decode one executed module's bitcode; see the module documentation.
    pub fn load(source: ModuleSource<'_>) -> Result<DecodedModule, CycleEvidence> {
        let undecoded = |detail: String| CycleEvidence::Undecoded {
            module: source.label.to_string(),
            detail,
        };
        let bitcode = match source.blob_b64 {
            Some(blob) => {
                use base64::Engine as _;
                base64::engine::general_purpose::STANDARD
                    .decode(blob)
                    .map_err(|error| undecoded(format!("retained blob is not base64: {error}")))?
            }
            None => assemble(source.air_ll).map_err(undecoded)?,
        };
        Self::from_bitcode(source.label, &bitcode)
    }
}

/// `llvm-as` of `air_ll` (opaque-pointer bitcode), in a scratch directory removed on return. The
/// checker's source for a row that retains no bitcode; tests use it to author real blobs.
pub fn assemble(air_ll: &str) -> Result<Vec<u8>, String> {
    use metal2vulkan_devtools::{llvm_assemble, AssembleError};
    let scratch = crate::ScratchDir::new("loop-free-assemble")?;
    let path = scratch.path().join("module.ll");
    std::fs::write(&path, air_ll).map_err(|error| format!("write {}: {error}", path.display()))?;
    llvm_assemble(&path).map_err(|error| match error {
        AssembleError::Missing(why) => {
            format!("the row retains no bitcode and llvm-as is unavailable: {why}")
        }
        AssembleError::Rejected(why) => format!("llvm-as rejected the row's text: {why}"),
    })
}

/// Whether the Metal oracle's fuel transform ([`crate::oracle_fuel`]) can bound every cycle of a
/// module set. The transform instruments the entry module only, and threads its fuel state through
/// extra parameters, so it covers exactly the CFG cycles of entry-module functions that nothing but
/// direct calls from the case entry reaches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FuelScope {
    /// No cycle, or only CFG cycles the transform instruments.
    Covered,
    /// A cycle the transform cannot bound: a call cycle, an unresolved call, an undecoded module,
    /// or a CFG cycle in a linked module or in a function reached other than by direct calls from
    /// the entry (`CfgCycle` names that function).
    Uncovered(CycleEvidence),
}

impl FuelScope {
    pub fn is_covered(&self) -> bool {
        matches!(self, Self::Covered)
    }

    /// The evidence, when the scope is uncovered.
    pub fn evidence(&self) -> Option<&CycleEvidence> {
        match self {
            Self::Covered => None,
            Self::Uncovered(evidence) => Some(evidence),
        }
    }
}

/// The mechanical class of a module set and what oracle fuel covers of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Classification {
    pub class: LoopClass,
    pub fuel: FuelScope,
}

impl Classification {
    fn refused(evidence: CycleEvidence) -> Classification {
        Classification {
            class: LoopClass::Cyclic(evidence.clone()),
            fuel: FuelScope::Uncovered(evidence),
        }
    }
}

/// Load and classify the closed set of modules one pipeline executes, starting at `entry` in the
/// [`ModuleRole::Entry`] module(s).
pub fn classify_sources<'a>(
    entry: &str,
    modules: impl IntoIterator<Item = (ModuleRole, ModuleSource<'a>)>,
) -> LoopClass {
    classify_sources_with_fuel(entry, modules).class
}

/// [`classify_sources`] together with the oracle's [`FuelScope`].
pub fn classify_sources_with_fuel<'a>(
    entry: &str,
    modules: impl IntoIterator<Item = (ModuleRole, ModuleSource<'a>)>,
) -> Classification {
    decode_and_classify(entry, modules).0
}

/// [`classify_sources_with_fuel`], also handing back the modules decoded, the entry module (which
/// the caller lists first) first, so a caller that reads their typed facts decodes each once. The
/// modules are `Err` with the entry's evidence when the entry did not decode; a later module that
/// does not decode refuses the classification but not the entry, and it and the modules after it
/// are not handed back.
pub fn decode_and_classify<'a>(
    entry: &str,
    modules: impl IntoIterator<Item = (ModuleRole, ModuleSource<'a>)>,
) -> (
    Classification,
    Result<Vec<(ModuleRole, DecodedModule)>, CycleEvidence>,
) {
    classify_loaded(entry, Vec::new(), modules)
}

/// [`decode_and_classify`] with the entry module given as `bitcode`, the module an executor
/// compiles itself in place of the row the checker assembled (`label` names it in evidence), and
/// `linked`, the other modules the pipeline executes. The decoded entry comes back as in
/// [`decode_and_classify`], so the caller can read the compiled module's own typed facts.
pub fn classify_compiled_with_fuel<'a>(
    entry: &str,
    label: &str,
    bitcode: &[u8],
    linked: impl IntoIterator<Item = (ModuleRole, ModuleSource<'a>)>,
) -> (Classification, Result<DecodedModule, CycleEvidence>) {
    match DecodedModule::from_bitcode(label, bitcode) {
        Ok(module) => {
            let (classification, modules) =
                classify_loaded(entry, vec![(ModuleRole::Entry, module)], linked);
            (
                classification,
                modules.map(|modules| {
                    modules
                        .into_iter()
                        .next()
                        .map(|(_, module)| module)
                        .expect("the compiled entry module comes first")
                }),
            )
        }
        Err(evidence) => (Classification::refused(evidence.clone()), Err(evidence)),
    }
}

/// Load `modules` after the already `decoded` ones, then classify them all, handing back the
/// modules decoded (see [`decode_and_classify`]).
fn classify_loaded<'a>(
    entry: &str,
    mut decoded: Vec<(ModuleRole, DecodedModule)>,
    modules: impl IntoIterator<Item = (ModuleRole, ModuleSource<'a>)>,
) -> (
    Classification,
    Result<Vec<(ModuleRole, DecodedModule)>, CycleEvidence>,
) {
    let mut refused = None;
    for (role, source) in modules {
        match DecodedModule::load(source) {
            Ok(module) => decoded.push((role, module)),
            Err(evidence) if decoded.is_empty() => {
                return (Classification::refused(evidence.clone()), Err(evidence));
            }
            Err(evidence) => {
                refused = Some(evidence);
                break;
            }
        }
    }
    let classification = match refused {
        Some(evidence) => Classification::refused(evidence),
        None => {
            let borrowed = decoded
                .iter()
                .map(|(role, module)| (*role, module))
                .collect::<Vec<_>>();
            classify_with_fuel(entry, &borrowed)
        }
    };
    if decoded.is_empty() {
        let evidence = CycleEvidence::MissingEntry {
            entry: entry.to_string(),
        };
        return (classification, Err(evidence));
    }
    (classification, Ok(decoded))
}

/// Classify one text-only entry module with nothing linked (assembled with `llvm-as`); see
/// [`classify_sources`].
pub fn classify_text_module(entry: &str, air_ll: &str) -> LoopClass {
    classify_sources(
        entry,
        [(
            ModuleRole::Entry,
            ModuleSource {
                label: "<text module>",
                air_ll,
                blob_b64: None,
            },
        )],
    )
}

/// Classify decoded modules; see the module documentation.
pub fn classify(entry: &str, modules: &[(ModuleRole, &DecodedModule)]) -> LoopClass {
    classify_with_fuel(entry, modules).class
}

/// [`classify`] together with the oracle's [`FuelScope`].
pub fn classify_with_fuel(entry: &str, modules: &[(ModuleRole, &DecodedModule)]) -> Classification {
    match CallGraph::build(entry, modules) {
        Ok(graph) => graph.classify(modules),
        Err(evidence) => Classification::refused(evidence),
    }
}

/// One defined function of one executed module.
#[derive(Clone, Copy)]
struct Node {
    module: usize,
    func: FuncId,
}

/// The call graph over every defined function, with a virtual root (node 0) whose successors are
/// the functions the runtime starts.
struct CallGraph {
    nodes: Vec<Node>,
    edges: Vec<Vec<u32>>,
    /// Per edge of `edges`: whether it is a direct call to a function the caller's own module
    /// defines (the only edge the oracle's fuel state can be threaded along).
    local: Vec<Vec<bool>>,
    /// The case-entry functions among the root's successors.
    entries: BTreeSet<u32>,
    /// Every possible target of an indirect call whose callee is not closed: a caller outside
    /// this graph's direct calls (a runtime function table, a linked module) may reach it without
    /// the oracle's fuel state.
    indirect: BTreeSet<u32>,
    /// The targets of each closed indirect call: the oracle passes its fuel state to every one of
    /// them or to none (their prototypes stay one type).
    closed: Vec<Vec<u32>>,
    /// The first call per node that resolves to nothing, as (callee name).
    unresolved: Vec<Option<String>>,
}

impl Graph for CallGraph {
    type Node = u32;

    fn node_count(&self) -> usize {
        self.edges.len()
    }

    fn entry(&self) -> u32 {
        0
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.edges
            .get(node as usize)
            .into_iter()
            .flat_map(|edges| edges.iter().copied())
    }
}

const ROOT: u32 = 0;

/// Intrinsic family through which the device may invoke linked intersection functions.
const INTERSECTION_INTRINSIC_PREFIX: &[u8] = b"air.intersect";

/// The LLVM ABI global listing module constructors the runtime runs itself.
const GLOBAL_CTORS: &[u8] = b"llvm.global_ctors";

impl CallGraph {
    fn build(entry: &str, modules: &[(ModuleRole, &DecodedModule)]) -> Result<Self, CycleEvidence> {
        let mut graph = CallGraph {
            nodes: vec![Node {
                module: usize::MAX,
                func: FuncId::new(0),
            }],
            edges: vec![Vec::new()],
            local: vec![Vec::new()],
            entries: BTreeSet::new(),
            indirect: BTreeSet::new(),
            closed: Vec::new(),
            unresolved: vec![None],
        };
        let mut index = HashMap::<(usize, FuncId), u32>::new();
        let mut by_name = BTreeMap::<&[u8], Vec<u32>>::new();
        for (position, (_, decoded)) in modules.iter().enumerate() {
            let module = &decoded.module;
            for (func, function) in module.parts().funcs.iter() {
                if function.body.is_none() {
                    continue;
                }
                let node = graph.nodes.len() as u32;
                graph.nodes.push(Node {
                    module: position,
                    func,
                });
                graph.edges.push(Vec::new());
                graph.local.push(Vec::new());
                graph.unresolved.push(None);
                index.insert((position, func), node);
                by_name
                    .entry(module.symbols().get(function.name).unwrap_or_default())
                    .or_default()
                    .push(node);
            }
        }

        let mut indirect_targets = BTreeSet::new();
        let mut intersection_functions = BTreeSet::new();
        let mut closed_calls = modules
            .iter()
            .map(|_| ClosedCalls::default())
            .collect::<Vec<_>>();
        for (position, (role, decoded)) in modules.iter().enumerate() {
            let defined = graph
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| node.module == position)
                .map(|(node, _)| node as u32);
            match role {
                ModuleRole::Entry => {
                    let closed = crate::call_targets::closed_calls(&decoded.module);
                    indirect_targets.extend(
                        defined.filter(|&node| {
                            closed.escaped.contains(&graph.nodes[node as usize].func)
                        }),
                    );
                    closed_calls[position] = closed;
                }
                ModuleRole::Linked => indirect_targets.extend(defined),
                ModuleRole::Intersection => {
                    let defined = defined.collect::<Vec<_>>();
                    indirect_targets.extend(defined.iter().copied());
                    intersection_functions.extend(defined);
                }
            }
        }

        // Roots: the case entry, then everything the runtime runs by itself.
        let mut roots = Vec::new();
        for (position, (role, decoded)) in modules.iter().enumerate() {
            let module = &decoded.module;
            if *role == ModuleRole::Entry {
                roots.extend(module.parts().funcs.iter().filter_map(|(func, function)| {
                    (module.symbols().get(function.name) == Some(entry.as_bytes()))
                        .then(|| index.get(&(position, func)).copied())
                        .flatten()
                }));
            }
        }
        if roots.is_empty() {
            return Err(CycleEvidence::MissingEntry {
                entry: entry.to_string(),
            });
        }
        graph.entries = roots.iter().copied().collect();
        for (position, (_, decoded)) in modules.iter().enumerate() {
            let module = &decoded.module;
            for (func, function) in module.parts().funcs.iter() {
                if matches!(
                    function.section,
                    Some(Section::StaticInit | Section::FcInitializer)
                ) {
                    roots.extend(index.get(&(position, func)).copied());
                }
            }
            for global in module.parts().globals.values() {
                if module.symbols().get(global.name) == Some(GLOBAL_CTORS) {
                    let mut funcs = BTreeSet::new();
                    if let Some(init) = global.init {
                        const_funcs(module, init, &mut funcs, &mut HashSet::new());
                    }
                    roots.extend(
                        funcs
                            .into_iter()
                            .filter_map(|func| index.get(&(position, func)).copied()),
                    );
                }
            }
        }
        graph.local[ROOT as usize] = vec![false; roots.len()];
        graph.edges[ROOT as usize] = roots;

        let facts = modules
            .iter()
            .map(|(_, decoded)| {
                let abi = decoded.abi.as_ref().ok();
                let stubs = abi
                    .into_iter()
                    .flat_map(AirAbi::visible_function_stubs)
                    .map(|(stub, symbol)| (stub, &*symbol.0))
                    .collect::<HashMap<FuncId, &[u8]>>();
                let fetches = abi
                    .into_iter()
                    .flat_map(AirAbi::control_point_fetches)
                    .collect::<HashSet<_>>();
                (stubs, fetches)
            })
            .collect::<Vec<_>>();
        for node in 1..graph.nodes.len() {
            let Node {
                module: position,
                func,
            } = graph.nodes[node];
            let module = &modules[position].1.module;
            let (stubs, runtime_fetches) = &facts[position];
            let Some(body) = module
                .func(func)
                .and_then(|function| function.body.as_ref())
            else {
                continue;
            };
            let mut edges = Vec::new();
            let mut local = Vec::new();
            let mut unresolved = None;
            for (at, inst) in body.insts.iter() {
                let InstKind::Call(call) = &inst.kind else {
                    continue;
                };
                match &call.callee {
                    Callee::Direct(callee) => {
                        if let Some(&target) = index.get(&(position, *callee)) {
                            edges.push(target);
                            local.push(true);
                            continue;
                        }
                        if runtime_fetches.contains(callee) {
                            continue;
                        }
                        // A `matmul2d` ABI call whose constant descriptor bounds its work is a
                        // leaf, as an intrinsic is.
                        if module.func(*callee).is_some_and(|decl| {
                            decl.body.is_none()
                                && matmul2d::bounded_leaf(module, body, at, call, decl)
                        }) {
                            continue;
                        }
                        let name = match stubs.get(callee) {
                            Some(&symbol) => symbol,
                            None => module
                                .func(*callee)
                                .and_then(|function| module.symbols().get(function.name))
                                .unwrap_or_default(),
                        };
                        match by_name.get(name) {
                            Some(targets) => {
                                edges.extend(targets.iter().copied());
                                local.resize(edges.len(), false);
                            }
                            None => {
                                unresolved.get_or_insert_with(|| {
                                    String::from_utf8_lossy(name).into_owned()
                                });
                            }
                        }
                    }
                    Callee::Indirect { .. } => match closed_calls[position].targets(func, at) {
                        Some(targets) => {
                            let targets = targets
                                .iter()
                                .filter_map(|&target| index.get(&(position, target)).copied())
                                .collect::<Vec<_>>();
                            edges.extend(targets.iter().copied());
                            local.resize(edges.len(), true);
                            graph.closed.push(targets);
                        }
                        None => {
                            edges.extend(indirect_targets.iter().copied());
                            local.resize(edges.len(), false);
                        }
                    },
                    Callee::UnknownIntrinsic(decl) => {
                        let name = module
                            .func(*decl)
                            .and_then(|function| module.symbols().get(function.name))
                            .unwrap_or_default();
                        if name.starts_with(INTERSECTION_INTRINSIC_PREFIX) {
                            edges.extend(intersection_functions.iter().copied());
                            local.resize(edges.len(), false);
                        }
                    }
                    Callee::Air { .. } | Callee::Llvm { .. } | Callee::Unbound { .. } => {}
                }
            }
            graph.edges[node] = edges;
            graph.local[node] = local;
            graph.unresolved[node] = unresolved;
        }
        graph.indirect = indirect_targets;
        Ok(graph)
    }

    /// Check every reachable function (in DFS preorder from the root) for a CFG cycle and an
    /// unresolved call, then the call graph for a cycle. The class is the first evidence in that
    /// order; the fuel scope needs the whole picture, so every reachable function is examined.
    fn classify(&self, modules: &[(ModuleRole, &DecodedModule)]) -> Classification {
        let mut meter = Meter::new(&Budget::standard());
        let calls = match Cfg::build(self, &mut meter) {
            Ok(calls) => calls,
            Err(error) => return Classification::refused(self.internal(&error.into())),
        };
        let mut first = None;
        let mut unresolved = None;
        let mut cyclic = vec![false; self.nodes.len()];
        for node in calls.preorder().filter(|&node| node != ROOT) {
            let Node { module, func } = self.nodes[node as usize];
            if let Some(body) = modules[module]
                .1
                .module
                .func(func)
                .and_then(|function| function.body.as_ref())
            {
                let cfg = match Cfg::build(body, &mut meter) {
                    Ok(cfg) => cfg,
                    Err(error) => {
                        return Classification::refused(
                            first.unwrap_or_else(|| self.internal(&error.into())),
                        )
                    }
                };
                cyclic[node as usize] = cfg.preorder().any(|block| {
                    cfg.successors(block)
                        .any(|next| cfg.is_retreating(block, next))
                });
                meter.release(cfg.ledger_bytes());
                if cyclic[node as usize] && first.is_none() {
                    first = Some(CycleEvidence::CfgCycle {
                        function: self.name(modules, node),
                    });
                }
            }
            if let Some(callee) = &self.unresolved[node as usize] {
                let evidence = CycleEvidence::UnresolvedCall {
                    function: self.name(modules, node),
                    callee: callee.clone(),
                };
                unresolved.get_or_insert_with(|| evidence.clone());
                first.get_or_insert(evidence);
            }
        }
        let head = calls.preorder().find_map(|node| {
            calls
                .successors(node)
                .find(|&next| calls.is_retreating(node, next))
        });
        let call_cycle = head.map(|node| CycleEvidence::CallCycle {
            function: self.name(modules, node),
        });
        let class = match first.or_else(|| call_cycle.clone()) {
            None => LoopClass::LoopFree,
            Some(evidence) => LoopClass::Cyclic(evidence),
        };
        let fuel = match unresolved.or(call_cycle) {
            Some(evidence) => FuelScope::Uncovered(evidence),
            None => self.fuel_scope(&calls, &cyclic, modules),
        };
        meter.release(calls.ledger_bytes());
        Classification { class, fuel }
    }

    /// With the call graph acyclic and resolved: every function that reaches a CFG cycle, and
    /// every other target of a closed indirect call one of whose targets does, must sit in the
    /// entry module and be reached only by local calls (direct, or indirect with a closed callee)
    /// from the case entry (the entry itself only from the root), so the fuel state can be
    /// threaded to it.
    fn fuel_scope(
        &self,
        calls: &Cfg<u32>,
        cyclic: &[bool],
        modules: &[(ModuleRole, &DecodedModule)],
    ) -> FuelScope {
        // Edges fuel cannot be threaded along: from the root to anything but the case entry,
        // between functions anything but a local call within one module (never into the entry),
        // and any call reaching a function whose address escapes.
        let mut foreign = vec![false; self.nodes.len()];
        for node in calls.preorder() {
            for (index, &target) in self.edges[node as usize].iter().enumerate() {
                let threadable = if node == ROOT {
                    self.entries.contains(&target)
                } else {
                    self.local[node as usize][index] && !self.entries.contains(&target)
                };
                foreign[target as usize] |= !threadable;
            }
        }
        for &node in &self.indirect {
            foreign[node as usize] = true;
        }
        // A cyclic function each node reaches, callees first (reverse RPO of an acyclic graph).
        // A closed indirect call passes the state to all of its targets or none: when one of them
        // reaches a cycle, every one takes the state, as does everything that calls it, and must
        // be threadable as that one is. Such a target counts as reaching the cycle, until nothing
        // changes.
        let order = calls.rpo().collect::<Vec<_>>();
        let mut witness = vec![None::<u32>; self.nodes.len()];
        let mut takes_state = vec![None::<u32>; self.nodes.len()];
        loop {
            for &node in order.iter().rev().filter(|&&node| node != ROOT) {
                witness[node as usize] = if cyclic[node as usize] {
                    Some(node)
                } else {
                    takes_state[node as usize].or_else(|| {
                        calls
                            .successors(node)
                            .find_map(|next| witness[next as usize])
                    })
                };
            }
            let mut changed = false;
            for targets in &self.closed {
                let Some(cycle) = targets.iter().find_map(|&target| witness[target as usize])
                else {
                    continue;
                };
                for &target in targets {
                    if witness[target as usize].is_none() {
                        takes_state[target as usize] = Some(cycle);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        for &node in order.iter().filter(|&&node| node != ROOT) {
            let Some(cycle) = witness[node as usize] else {
                continue;
            };
            let module = self.nodes[node as usize].module;
            if foreign[node as usize] || modules[module].0 != ModuleRole::Entry {
                return FuelScope::Uncovered(CycleEvidence::CfgCycle {
                    function: self.name(modules, cycle),
                });
            }
        }
        FuelScope::Covered
    }

    fn name(&self, modules: &[(ModuleRole, &DecodedModule)], node: u32) -> String {
        let Node { module, func } = self.nodes[node as usize];
        let decoded = &modules[module].1.module;
        let name = decoded
            .func(func)
            .and_then(|function| decoded.symbols().get(function.name))
            .unwrap_or_default();
        String::from_utf8_lossy(name).into_owned()
    }

    fn internal(&self, error: &metal2vulkan::error::Error) -> CycleEvidence {
        CycleEvidence::Undecoded {
            module: "<call graph>".into(),
            detail: error.to_string(),
        }
    }
}

mod matmul2d;
#[cfg(test)]
mod tests;
