//! The function constants one function of an unlinked module reads: what the metallib linker writes
//! into a function's `CNST` tag, and Metal answers as the function's `functionConstantsDictionary`.
//!
//! AIR declares function constants per module (`air.function_constants`), and a module's static
//! initializers copy them into internal globals (`constant int k = c;` at program scope) and
//! compute the predicate globals an argument's `[[function_constant(...)]]` names
//! (`air.function_constant` in the entry's metadata). A function reads a constant when its code
//! or its entry metadata names the constant's global or a global the static initializers compute
//! from it; it asks whether the constant is defined when it calls
//! `air.is_function_constant_defined` on it. Measured against every function of the public
//! reflection fixtures' metallibs (`validation/fixtures/public/reflection/metallib-tags.txt`),
//! the `CNST` tag lists exactly the constants the function reads, each required unless the
//! function asks about it.
//!
//! What the static initializers compute from which constant is a dependency analysis, not an
//! evaluation (no constant has a value here): each value depends on the constants its operands
//! depend on, a load on what was stored into the globals its address may point into, and a call
//! on what its callee returns. A shape whose dependencies the analysis cannot state exactly (a
//! pointer loaded from memory and followed, a pointer stored into memory that may point
//! anywhere, an address it cannot follow, an access other than a plain load or store, a call the
//! static-initializer interpreter does not run) makes the answer unknown for a function that
//! names a global the static initializers write; so does a function address the function's code
//! takes, whose callee's reads Metal's answer was never measured for.
//!
//! Branches follow Metal's answers for the fixtures `reflection/constant_initializers`,
//! `constant_loops` and `constant_branches` (measured on an M3 Max and an M4). A merge's (`phi`)
//! value depends on its incoming values and on the conditions of the branches that decide which
//! edge control arrives along (an if/else's reader lists its condition with its arms' constants). A
//! loop of a static initializer itself adds the constants every value in it depends on to every
//! function of the module, one that names no global the initializers write too (its trip count, the
//! value its body folds, and the constant a loop of a literal trip count folds); a constant
//! combined with the loop's result after it is listed only by the functions that read its global,
//! and a loop in a function the initializer calls adds nothing beyond its result's dependencies.
//! Unmeasured, and so unknown for every function: a store, a call, or a return that a branch
//! decides, a merge a branch decides outside a loop (nested), a merge or a loop that depends on a
//! definedness query, and a shape the analysis does not follow beside a branch. A definedness query
//! a global stores is listed, optional, by the functions that read the global, as one the function
//! asks itself.
//!
//! A pointer the static initializers store into a global depends on the constants that chose it,
//! and a function that names the global reads those (fixture `reflection/constant_pointers`: a
//! program-scope pointer a constant selects between two rows of a literal table answers that
//! constant, measured on an M3 Max and an M4). What the function then reads through the pointer
//! is the pointed-into globals' contents, which Metal's answer was measured for only where they
//! hold nothing a constant decides: the answer is unknown where a global the pointer may point
//! into (or one its initializer addresses) is a declared constant, a global the static
//! initializers write, one without a known initializer, or holds a function's address.

use std::collections::{BTreeSet, HashMap};

use crate::abi::AirAbi;
use crate::budget::Meter;
use crate::error::Error;
use crate::graph::{Cfg, PostDomTree, Sccs};
use crate::intrinsics::{AirIntrinsic, FunctionConstant, LlvmIntrinsic};
use crate::ir::{ArgIdx, BlockId, Body, Callee, Const, ConstId, FuncId, GlobalId, InstId};
use crate::ir::{InstKind, Md, MdId, Module, Section, Type, TypeId, Value};

/// A function constant a function reads (see [`constants_read`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConstantRead {
    /// The constant's index (`[[function_constant(index)]]`).
    pub index: u32,
    /// Whether the function asks whether the constant is defined
    /// (`air.is_function_constant_defined`), which makes it optional: Metal answers `required`
    /// for exactly the constants it does not ask about.
    pub asked: bool,
}

/// The function constants function `func` of `module` (whose ABI is `abi`) reads, by index: the
/// constants its code, its callees' code, and its entry metadata name, directly or through a
/// global the static initializers compute from them (module docs). `None` when that is unknown:
/// a static initializer the dependency analysis does not follow, or a function address the
/// function's code takes. A module that declares no constant answers none without analysis.
pub fn constants_read(
    module: &Module,
    abi: &AirAbi,
    func: FuncId,
    m: &mut Meter,
) -> Result<Option<Vec<ConstantRead>>, Error> {
    if abi.function_constants.is_empty() {
        return Ok(Some(Vec::new()));
    }
    m.charge(abi.function_constants.len() * std::mem::size_of::<(GlobalId, u32)>() * 2)?;
    let fcs: HashMap<GlobalId, u32> = abi
        .function_constants
        .iter()
        .map(|c| (c.global, c.index))
        .collect();
    let Some(mut named) = Named::of(module, &fcs, func, m)? else {
        return Ok(None);
    };
    // Only a global a static initializer can write carries a constant it does not name: a
    // variable (`constant` globals, the constants' own among them, are never written).
    let written = named
        .globals
        .iter()
        .any(|g| !fcs.contains_key(g) && module.global(*g).is_some_and(|g| !g.constant));
    // A static initializer that branches adds the constants its branches and merges depend on
    // to every function of the module, one that names no global it writes too (module docs),
    // so the analysis runs for every function wherever one branches.
    let (memory, everywhere) = match written || Flow::branches(module, m)? {
        true => match Flow::run(module, &fcs, m)? {
            Initializers::Known { memory, everywhere } => (memory, everywhere),
            Initializers::Unknown if !written => (HashMap::new(), Deps::default()),
            Initializers::Unknown | Initializers::Unstated => return Ok(None),
        },
        false => (HashMap::new(), Deps::default()),
    };
    // A definedness query a branch or a merge depends on: what Metal answers is unmeasured.
    if !everywhere.asked.is_empty() {
        return Ok(None);
    }
    let mut read = everywhere.read;
    for g in &named.globals {
        if let Some(&index) = fcs.get(g) {
            read.insert(index);
        }
        if let Some(held) = memory.get(g) {
            // A pointer the static initializers stored: known only where what it points into
            // holds nothing a constant decides (module docs).
            match &held.points {
                Points::Into(pointees) if inert(module, &fcs, &memory, pointees, m)? => {}
                Points::Into(_) | Points::Anywhere => return Ok(None),
            }
            read.extend(held.deps.read.iter().copied());
            // A global stored from a definedness query: a function that reads it lists the
            // constant, optional, as one that asks about it itself does (fixture
            // `reflection/constant_initializers`).
            read.extend(held.deps.asked.iter().copied());
            named.asked.extend(held.deps.asked.iter().copied());
        }
    }
    m.charge(read.len() * std::mem::size_of::<ConstantRead>())?;
    Ok(Some(
        read.into_iter()
            .map(|index| ConstantRead {
                index,
                asked: named.asked.contains(&index),
            })
            .collect(),
    ))
}

/// Whether the globals `pointees` hold nothing a constant decides: none is a declared constant
/// (`fcs`), a global the static initializers write (`memory`), one without a known initializer,
/// or one whose initializer holds a function's address, and the same holds for every global
/// their initializers address.
fn inert(
    module: &Module,
    fcs: &HashMap<GlobalId, u32>,
    memory: &HashMap<GlobalId, Fact>,
    pointees: &BTreeSet<GlobalId>,
    m: &mut Meter,
) -> Result<bool, Error> {
    let mut globals: Vec<GlobalId> = pointees.iter().copied().collect();
    let mut seen: BTreeSet<GlobalId> = BTreeSet::new();
    let mut consts: Vec<ConstId> = Vec::new();
    while let Some(g) = globals.pop() {
        m.tick(1)?;
        if !seen.insert(g) {
            continue;
        }
        if fcs.contains_key(&g) || memory.contains_key(&g) {
            return Ok(false);
        }
        let Some(init) = module
            .global(g)
            .filter(|g| !g.externally_initialized)
            .and_then(|g| g.init)
        else {
            return Ok(false);
        };
        consts.push(init);
        while let Some(c) = consts.pop() {
            m.tick(1)?;
            let Some(k) = module.consts().get(c) else {
                continue;
            };
            match k.value {
                Const::GlobalAddr(h) => globals.push(h),
                Const::FuncAddr(_) => return Ok(false),
                _ => consts.extend_from_slice(k.value.children()),
            }
        }
    }
    Ok(true)
}

/// What a function names: the globals its code (and its callees') and its entry metadata refer
/// to, and the constants it asks about.
struct Named {
    globals: BTreeSet<GlobalId>,
    asked: BTreeSet<u32>,
}

impl Named {
    /// What `func` names, or `None` when its code takes a defined function's address or asks
    /// about a global that is no declared constant.
    fn of(
        module: &Module,
        fcs: &HashMap<GlobalId, u32>,
        func: FuncId,
        m: &mut Meter,
    ) -> Result<Option<Named>, Error> {
        let mut walk = Walk {
            module,
            func,
            funcs: vec![func],
            seen_funcs: BTreeSet::from([func]),
            seen_consts: BTreeSet::new(),
            globals: BTreeSet::new(),
            asked: BTreeSet::new(),
            unknown: false,
        };
        // The entry's metadata nodes: every named list's node that begins with its address. A
        // function its metadata names (a post-tessellation entry's control-point function) is
        // part of it.
        let mut nodes: Vec<MdId> = Vec::new();
        for list in module.named_md() {
            for &node in list.ops.iter() {
                m.tick(1)?;
                if walk.names_func(node) {
                    nodes.push(node);
                }
            }
        }
        let mut seen_nodes = BTreeSet::new();
        while let Some(node) = nodes.pop() {
            m.tick(1)?;
            if !seen_nodes.insert(node) {
                continue;
            }
            match module.md(node) {
                Some(Md::Node { ops, .. }) => nodes.extend(ops.iter().flatten().copied()),
                Some(Md::Value(c)) => walk.constant(*c, false, m)?,
                _ => {}
            }
        }
        let mut consts = Vec::new();
        while let Some(f) = walk.funcs.pop() {
            let Some(body) = module.func(f).and_then(|f| f.body.as_ref()) else {
                continue;
            };
            for inst in body.insts.values() {
                m.tick(1)?;
                consts.clear();
                inst.kind.for_each_value(|v| {
                    if let Value::Const(c) = v {
                        consts.push(c);
                    }
                });
                for &c in &consts {
                    walk.constant(c, true, m)?;
                }
                let InstKind::Call(call) = &inst.kind else {
                    continue;
                };
                match call.callee {
                    Callee::Direct(callee) => walk.function(callee),
                    Callee::Air {
                        op: AirIntrinsic::FunctionConstant(FunctionConstant::IsDefined),
                        ..
                    } => {
                        let index = call
                            .args
                            .first()
                            .and_then(|&v| global_of(module, v))
                            .and_then(|g| fcs.get(&g));
                        match index {
                            Some(&index) => {
                                walk.asked.insert(index);
                            }
                            None => walk.unknown = true,
                        }
                    }
                    _ => {}
                }
            }
        }
        if walk.unknown {
            return Ok(None);
        }
        let Walk { globals, asked, .. } = walk;
        Ok(Some(Named { globals, asked }))
    }
}

/// [`Named::of`]'s walk state.
struct Walk<'a> {
    module: &'a Module,
    func: FuncId,
    funcs: Vec<FuncId>,
    seen_funcs: BTreeSet<FuncId>,
    seen_consts: BTreeSet<ConstId>,
    globals: BTreeSet<GlobalId>,
    asked: BTreeSet<u32>,
    /// The answer is unknown (see [`Named::of`]).
    unknown: bool,
}

impl Walk<'_> {
    /// Whether metadata node `node` begins with the function's address.
    fn names_func(&self, node: MdId) -> bool {
        let Some(Md::Node { ops, .. }) = self.module.md(node) else {
            return false;
        };
        let Some(Some(first)) = ops.first() else {
            return false;
        };
        let Some(Md::Value(c)) = self.module.md(*first) else {
            return false;
        };
        matches!(
            self.module.consts().get(*c).map(|k| &k.value),
            Some(Const::FuncAddr(f)) if *f == self.func
        )
    }

    /// A defined function the walk reaches.
    fn function(&mut self, f: FuncId) {
        let defined = self.module.func(f).is_some_and(|f| f.body.is_some());
        if defined && self.seen_funcs.insert(f) {
            self.funcs.push(f);
        }
    }

    /// The globals constant `c` names, through its operands and the named globals' initializers.
    /// A defined function's address in code (`in_code`) makes the answer unknown; one in the
    /// entry's metadata is part of the function.
    fn constant(&mut self, c: ConstId, in_code: bool, m: &mut Meter) -> Result<(), Error> {
        let mut stack = vec![c];
        while let Some(c) = stack.pop() {
            m.tick(1)?;
            if !self.seen_consts.insert(c) {
                continue;
            }
            let Some(k) = self.module.consts().get(c) else {
                continue;
            };
            stack.extend_from_slice(k.value.children());
            match k.value {
                Const::GlobalAddr(g) => {
                    if self.globals.insert(g) {
                        if let Some(init) = self.module.global(g).and_then(|g| g.init) {
                            stack.push(init);
                        }
                    }
                }
                Const::FuncAddr(f) if f != self.func => {
                    let defined = self.module.func(f).is_some_and(|f| f.body.is_some());
                    if defined && in_code {
                        self.unknown = true;
                    }
                    self.function(f);
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// The global value `v` addresses, when it is a constant global address.
fn global_of(module: &Module, v: Value) -> Option<GlobalId> {
    let Value::Const(c) = v else {
        return None;
    };
    match module.consts().get(c).map(|k| &k.value) {
        Some(Const::GlobalAddr(g)) => Some(*g),
        _ => None,
    }
}

/// Whether a value of type `ty` holds a pointer (is one, or a vector or aggregate with one).
fn holds_pointer(module: &Module, ty: TypeId, m: &mut Meter) -> Result<bool, Error> {
    let mut stack = vec![ty];
    while let Some(ty) = stack.pop() {
        m.tick(1)?;
        match module.types().get(ty) {
            Some(Type::Ptr { .. }) => return Ok(true),
            Some(Type::Vector { elem, .. } | Type::Array { elem, .. }) => stack.push(*elem),
            Some(Type::Struct { fields, .. }) => stack.extend_from_slice(fields),
            _ => {}
        }
    }
    Ok(false)
}

/// The constants a value depends on: those it is computed from (`read`), and those whose
/// definedness it is (`asked`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Deps {
    read: BTreeSet<u32>,
    asked: BTreeSet<u32>,
}

impl Deps {
    /// Add `other`'s constants; the number added.
    fn absorb(&mut self, other: &Deps) -> usize {
        let before = self.read.len() + self.asked.len();
        self.read.extend(other.read.iter().copied());
        self.asked.extend(other.asked.iter().copied());
        self.read.len() + self.asked.len() - before
    }
}

/// Where a pointer value may point: into these globals, or anywhere.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Points {
    Into(BTreeSet<GlobalId>),
    Anywhere,
}

impl Default for Points {
    fn default() -> Points {
        Points::Into(BTreeSet::new())
    }
}

/// What the analysis knows of one value.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Fact {
    deps: Deps,
    points: Points,
}

impl Fact {
    /// Add `other`'s facts; the number of elements added (an `Anywhere` counts one).
    fn absorb(&mut self, other: &Fact) -> usize {
        let mut added = self.deps.absorb(&other.deps);
        match (&mut self.points, &other.points) {
            (Points::Anywhere, _) => {}
            (points, Points::Anywhere) => {
                *points = Points::Anywhere;
                added += 1;
            }
            (Points::Into(a), Points::Into(b)) => {
                let before = a.len();
                a.extend(b.iter().copied());
                added += a.len() - before;
            }
        }
        added
    }
}

/// The bytes charged for each element a fact gains (a set node's share).
const ELEMENT_BYTES: usize = 32;

/// What the static initializers' analysis states ([`Flow::run`]).
enum Initializers {
    /// What each global holds once the static initializers ran (`memory`), and the constants
    /// every function of the module reads (`everywhere`, module docs).
    Known {
        memory: HashMap<GlobalId, Fact>,
        everywhere: Deps,
    },
    /// A shape the analysis does not follow: what the globals the static initializers write
    /// hold is unknown.
    Unknown,
    /// A shape the analysis does not follow beside a branch, which may hide what a branch or a
    /// merge depends on: the constants every function reads are unknown.
    Unstated,
}

/// A shape one pass of the analysis does not follow ([`Flow::function`]).
struct Unfollowed;

/// The control facts of one function the analysis runs over, per block ([`BlockShape`]).
struct Shape {
    blocks: Vec<BlockShape>,
}

/// One block's control facts.
#[derive(Clone, Default)]
struct BlockShape {
    preds: Vec<BlockId>,
    /// Whether the block lies on a cycle.
    cyclic: bool,
    /// Whether the block ends in a branch with two or more different successors.
    branches: bool,
    /// The blocks whose conditional branch decides whether this one runs (its control
    /// dependence: a branch with two different successors, one of which the block
    /// post-dominates, where the block does not post-dominate the branch).
    deciders: BTreeSet<BlockId>,
}

impl Shape {
    /// The control facts of `body`; `None` when a block reachable from the entry reaches no
    /// return (its post-dominators, and so what decides it, are not stated).
    fn of(body: &Body, m: &mut Meter) -> Result<Option<Shape>, Error> {
        m.charge(body.blocks.len() * std::mem::size_of::<BlockShape>())?;
        let cfg = Cfg::build(body, m)?;
        let shape = Shape::with_cfg(body, &cfg, m);
        m.release(cfg.ledger_bytes());
        shape
    }

    fn with_cfg(body: &Body, cfg: &Cfg<BlockId>, m: &mut Meter) -> Result<Option<Shape>, Error> {
        let pdom = PostDomTree::compute(cfg, m)?;
        let sccs = match Sccs::compute(body, m) {
            Ok(sccs) => sccs,
            Err(e) => {
                m.release(pdom.ledger_bytes());
                return Err(e);
            }
        };
        let shape = Shape::from_graphs(body, cfg, &pdom, &sccs, m);
        m.release(pdom.ledger_bytes() + sccs.ledger_bytes());
        shape
    }

    fn from_graphs(
        body: &Body,
        cfg: &Cfg<BlockId>,
        pdom: &PostDomTree<BlockId>,
        sccs: &Sccs<BlockId>,
        m: &mut Meter,
    ) -> Result<Option<Shape>, Error> {
        let mut shape = Shape {
            blocks: vec![BlockShape::default(); body.blocks.len()],
        };
        for c in (0..sccs.len()).filter_map(|c| u32::try_from(c).ok()) {
            m.tick(1)?;
            if sccs.is_cyclic(c) {
                for b in sccs.members(c) {
                    if let Some(at) = shape.blocks.get_mut(ix(b)) {
                        at.cyclic = true;
                    }
                }
            }
        }
        let mut succs: BTreeSet<BlockId> = BTreeSet::new();
        for y in body.blocks.ids() {
            m.tick(1)?;
            if !cfg.is_reachable(y) {
                continue;
            }
            if !pdom.is_reachable(y) {
                return Ok(None);
            }
            succs.clear();
            succs.extend(body.successors(y));
            for &s in &succs {
                if let Some(at) = shape.blocks.get_mut(ix(s)) {
                    at.preds.push(y);
                }
            }
            if succs.len() < 2 {
                continue;
            }
            if let Some(at) = shape.blocks.get_mut(ix(y)) {
                at.branches = true;
            }
            // Every block from a successor up the post-dominator tree to the branch's own
            // post-dominator runs as the branch decides.
            let stop = pdom.ipdom(y);
            for &s in &succs {
                let mut at = Some(s);
                while let Some(x) = at {
                    m.tick(1)?;
                    if Some(x) == stop {
                        break;
                    }
                    if let Some(block) = shape.blocks.get_mut(ix(x)) {
                        if block.deciders.insert(y) {
                            m.charge(ELEMENT_BYTES)?;
                        }
                    }
                    at = pdom.ipdom(x);
                }
            }
        }
        Ok(Some(shape))
    }

    /// Whether a branch decides whether `block` runs.
    fn decided(&self, block: BlockId) -> bool {
        self.blocks
            .get(ix(block))
            .is_some_and(|b| !b.deciders.is_empty())
    }

    /// Whether `block` lies on a cycle.
    fn cyclic(&self, block: BlockId) -> bool {
        self.blocks.get(ix(block)).is_some_and(|b| b.cyclic)
    }

    /// The blocks whose branch decides which incoming edge of `block` control arrives along:
    /// each predecessor that branches, and the blocks that decide a predecessor.
    fn gates(&self, block: BlockId, m: &mut Meter) -> Result<BTreeSet<BlockId>, Error> {
        let mut gates = BTreeSet::new();
        let preds = self.blocks.get(ix(block)).map_or(&[][..], |b| &b.preds);
        for (p, at) in preds
            .iter()
            .filter_map(|&p| Some((p, self.blocks.get(ix(p))?)))
        {
            m.tick(1)?;
            if at.branches {
                gates.insert(p);
            }
            gates.extend(at.deciders.iter().copied());
        }
        m.charge(gates.len() * ELEMENT_BYTES)?;
        Ok(gates)
    }
}

/// A block's dense index.
fn ix(block: BlockId) -> usize {
    block.raw() as usize
}

/// The dependency analysis of the static initializers (module docs): a fixed point over the
/// `air.static_init` functions and every function they call, flow-insensitive.
struct Flow<'a> {
    module: &'a Module,
    fcs: &'a HashMap<GlobalId, u32>,
    values: HashMap<(FuncId, InstId), Fact>,
    args: HashMap<(FuncId, ArgIdx), Fact>,
    returns: HashMap<FuncId, Fact>,
    /// What each global holds once the static initializers ran: the constants stored into it,
    /// and where the pointers stored into it point.
    memory: HashMap<GlobalId, Fact>,
    /// The constants a branch's condition or a merge's (`phi`) value depends on: every function
    /// of the module reads them (module docs).
    everywhere: Deps,
    changed: bool,
}

impl<'a> Flow<'a> {
    /// The `air.static_init` functions and every function they call, and whether any of them
    /// branches (a conditional branch or a switch); `None` for a call to a declaration, which the
    /// static-initializer interpreter does not run either.
    fn functions(module: &Module, m: &mut Meter) -> Result<Option<(Vec<FuncId>, bool)>, Error> {
        let mut funcs: Vec<FuncId> = module
            .parts()
            .funcs
            .iter()
            .filter(|(_, f)| f.section == Some(Section::StaticInit) && f.body.is_some())
            .map(|(id, _)| id)
            .collect();
        let mut seen: BTreeSet<FuncId> = funcs.iter().copied().collect();
        let mut branches = false;
        let mut next = 0;
        while let Some(&f) = funcs.get(next) {
            next += 1;
            let Some(body) = module.func(f).and_then(|f| f.body.as_ref()) else {
                continue;
            };
            for inst in body.insts.values() {
                m.tick(1)?;
                let call = match &inst.kind {
                    InstKind::CondBr { .. } | InstKind::Switch { .. } => {
                        branches = true;
                        continue;
                    }
                    InstKind::Call(call) => call,
                    _ => continue,
                };
                if let Callee::Direct(g) = call.callee {
                    if !module.func(g).is_some_and(|g| g.body.is_some()) {
                        return Ok(None);
                    }
                    if seen.insert(g) {
                        funcs.push(g);
                    }
                }
            }
        }
        Ok(Some((funcs, branches)))
    }

    /// Whether a static initializer (or a function one calls) branches, or calls a declaration,
    /// so the analysis must run before any function's answer is known.
    fn branches(module: &Module, m: &mut Meter) -> Result<bool, Error> {
        Ok(Flow::functions(module, m)?.is_none_or(|(_, branches)| branches))
    }

    /// What each global holds once the static initializers ran ([`Initializers`]).
    fn run(
        module: &'a Module,
        fcs: &'a HashMap<GlobalId, u32>,
        m: &mut Meter,
    ) -> Result<Initializers, Error> {
        let Some((funcs, branches)) = Flow::functions(module, m)? else {
            return Ok(Initializers::Unknown);
        };
        let mut shapes = HashMap::new();
        for &f in &funcs {
            let Some(body) = module.func(f).and_then(|f| f.body.as_ref()) else {
                continue;
            };
            match Shape::of(body, m)? {
                Some(shape) => {
                    shapes.insert(f, shape);
                }
                // A block that reaches no return: what it decides is not followed.
                None if branches => return Ok(Initializers::Unstated),
                None => return Ok(Initializers::Unknown),
            }
        }
        let mut flow = Flow {
            module,
            fcs,
            values: HashMap::new(),
            args: HashMap::new(),
            returns: HashMap::new(),
            memory: HashMap::new(),
            everywhere: Deps::default(),
            changed: true,
        };
        // Every round either adds an element to a fact or ends: the facts are finite sets.
        while flow.changed {
            flow.changed = false;
            for &f in &funcs {
                let Some(shape) = shapes.get(&f) else {
                    continue;
                };
                match flow.function(f, shape, m)? {
                    None => {}
                    // Beside a branch, a shape the analysis does not follow may hide what a
                    // branch or a merge depends on.
                    Some(Unfollowed) if branches => return Ok(Initializers::Unstated),
                    Some(Unfollowed) => return Ok(Initializers::Unknown),
                }
            }
        }
        Ok(Initializers::Known {
            memory: flow.memory,
            everywhere: flow.everywhere,
        })
    }

    /// One pass over function `f`'s instructions, whose control facts are `shape`; `Some` when a
    /// shape stops the analysis.
    fn function(
        &mut self,
        f: FuncId,
        shape: &Shape,
        m: &mut Meter,
    ) -> Result<Option<Unfollowed>, Error> {
        let module = self.module;
        let Some(func) = module.func(f) else {
            return Ok(None);
        };
        let Some(body) = func.body.as_ref() else {
            return Ok(None);
        };
        // A static initializer's loop: every function of the module reads the constants its
        // values depend on (module docs); a loop of a function it calls adds none.
        let init = func.section == Some(Section::StaticInit);
        let mut operands = Vec::new();
        for (b, id, inst) in body
            .blocks
            .ids()
            .flat_map(|b| body.block_insts(b).map(move |(id, inst)| (b, id, inst)))
        {
            m.tick(1)?;
            let decided = shape.decided(b);
            let looped = init && shape.cyclic(b);
            let fact = match &inst.kind {
                InstKind::Load {
                    ptr,
                    volatile,
                    atomic,
                    ..
                } => {
                    if *volatile || atomic.is_some() {
                        return Ok(Some(Unfollowed));
                    }
                    let p = self.operand(f, *ptr, m)?;
                    let Points::Into(globals) = &p.points else {
                        return Ok(Some(Unfollowed));
                    };
                    let mut deps = p.deps.clone();
                    for g in globals {
                        if let Some(&index) = self.fcs.get(g) {
                            deps.read.insert(index);
                        }
                        if let Some(held) = self.memory.get(g) {
                            deps.absorb(&held.deps);
                        }
                    }
                    // A pointer loaded from memory: where it points is not followed.
                    let points = match holds_pointer(module, inst.ty, m)? {
                        true => Points::Anywhere,
                        false => Points::default(),
                    };
                    Fact { deps, points }
                }
                InstKind::Store {
                    val,
                    ptr,
                    volatile,
                    atomic,
                    ..
                } => {
                    // A store a branch decides: what Metal answers is unmeasured.
                    if *volatile || atomic.is_some() || decided {
                        return Ok(Some(Unfollowed));
                    }
                    let p = self.operand(f, *ptr, m)?;
                    let v = self.operand(f, *val, m)?;
                    let Points::Into(globals) = &p.points else {
                        return Ok(Some(Unfollowed));
                    };
                    // A pointer stored into memory is recorded with where it points (a later load
                    // of it is still not followed, above); one that may point anywhere is not.
                    if v.points == Points::Anywhere {
                        return Ok(Some(Unfollowed));
                    }
                    let mut stored = v;
                    stored.deps.absorb(&p.deps);
                    for g in globals {
                        if self.fcs.contains_key(g) {
                            return Ok(Some(Unfollowed));
                        }
                        let slot = self.memory.entry(*g).or_default();
                        let added = slot.absorb(&stored);
                        m.charge(added * ELEMENT_BYTES)?;
                        self.changed |= added > 0;
                    }
                    continue;
                }
                InstKind::Alloca { .. } => Fact {
                    deps: Deps::default(),
                    points: Points::Anywhere,
                },
                InstKind::AtomicRmw { .. } | InstKind::CmpXchg { .. } | InstKind::Fence(_) => {
                    return Ok(Some(Unfollowed));
                }
                InstKind::Call(call) => match call.callee {
                    // A call a branch decides: what its effects add is unmeasured.
                    Callee::Direct(_) if decided => return Ok(Some(Unfollowed)),
                    Callee::Direct(g) => {
                        for (i, &a) in (0u32..).zip(call.args.iter()) {
                            let fact = self.operand(f, a, m)?;
                            let slot = self.args.entry((g, ArgIdx::new(i))).or_default();
                            let added = slot.absorb(&fact);
                            m.charge(added * ELEMENT_BYTES)?;
                            self.changed |= added > 0;
                        }
                        self.returns.get(&g).cloned().unwrap_or_default()
                    }
                    Callee::Air {
                        op: AirIntrinsic::FunctionConstant(FunctionConstant::IsDefined),
                        ..
                    } => {
                        let index = call
                            .args
                            .first()
                            .and_then(|&v| global_of(module, v))
                            .and_then(|g| self.fcs.get(&g));
                        let Some(&index) = index else {
                            return Ok(Some(Unfollowed));
                        };
                        Fact {
                            deps: Deps {
                                read: BTreeSet::new(),
                                asked: BTreeSet::from([index]),
                            },
                            points: Points::default(),
                        }
                    }
                    // The pure calls the static-initializer interpreter runs: a predicate's
                    // normalization, a conversion, an integer minimum, maximum or count.
                    Callee::Air {
                        op:
                            AirIntrinsic::FunctionConstant(FunctionConstant::NormalizePredicate(_))
                            | AirIntrinsic::Convert(_),
                        ..
                    }
                    | Callee::Llvm {
                        op: LlvmIntrinsic::Int { .. } | LlvmIntrinsic::NarrowInt { .. },
                        ..
                    } => self.union(f, &call.args, m)?,
                    _ => return Ok(Some(Unfollowed)),
                },
                InstKind::Ret(v) => {
                    // A return a branch decides: the value's dependence on it is not followed.
                    if v.is_some() && decided {
                        return Ok(Some(Unfollowed));
                    }
                    if let Some(v) = v {
                        let fact = self.operand(f, *v, m)?;
                        let slot = self.returns.entry(f).or_default();
                        let added = slot.absorb(&fact);
                        m.charge(added * ELEMENT_BYTES)?;
                        self.changed |= added > 0;
                    }
                    continue;
                }
                // A branch's condition: what it decides takes its constants (a merge's value,
                // below); in a static initializer's loop, every function does.
                InstKind::CondBr { c: v, .. } | InstKind::Switch { v, .. } => {
                    if looped {
                        let deps = self.operand(f, *v, m)?.deps;
                        self.spread(&deps, m)?;
                    }
                    continue;
                }
                // A merge's value depends on its incoming values and on the conditions of the
                // branches that decide which edge arrives (module docs). A merge a branch decides
                // itself, outside a loop, is unmeasured.
                InstKind::Phi { incoming, .. } => {
                    if decided && !shape.cyclic(b) {
                        return Ok(Some(Unfollowed));
                    }
                    operands.clear();
                    operands.extend(incoming.iter().map(|&(v, _)| v));
                    for y in shape.gates(b, m)? {
                        match body.terminator(y).map(|t| &t.kind) {
                            Some(InstKind::CondBr { c, .. }) => operands.push(*c),
                            Some(InstKind::Switch { v, .. }) => operands.push(*v),
                            _ => {}
                        }
                    }
                    let fact = self.union(f, &operands, m)?;
                    // A merge a definedness query decides: unmeasured.
                    if !fact.deps.asked.is_empty() {
                        return Ok(Some(Unfollowed));
                    }
                    fact
                }
                InstKind::Br { .. } | InstKind::Unreachable => continue,
                kind => {
                    operands.clear();
                    kind.for_each_value(|v| operands.push(v));
                    self.union(f, &operands, m)?
                }
            };
            if looped {
                self.spread(&fact.deps, m)?;
            }
            let slot = self.values.entry((f, id)).or_default();
            let added = slot.absorb(&fact);
            m.charge(added * ELEMENT_BYTES)?;
            self.changed |= added > 0;
        }
        Ok(None)
    }

    /// Add `deps` to the constants every function reads.
    fn spread(&mut self, deps: &Deps, m: &mut Meter) -> Result<(), Error> {
        let added = self.everywhere.absorb(deps);
        m.charge(added * ELEMENT_BYTES)?;
        self.changed |= added > 0;
        Ok(())
    }

    /// The union of `values`' facts.
    fn union(&self, f: FuncId, values: &[Value], m: &mut Meter) -> Result<Fact, Error> {
        let mut fact = Fact::default();
        for &v in values {
            fact.absorb(&self.operand(f, v, m)?);
        }
        Ok(fact)
    }

    /// What is known of value `v` of function `f`.
    fn operand(&self, f: FuncId, v: Value, m: &mut Meter) -> Result<Fact, Error> {
        Ok(match v {
            Value::Inst(i) => self.values.get(&(f, i)).cloned().unwrap_or_default(),
            Value::Arg(a) => self.args.get(&(f, a)).cloned().unwrap_or_default(),
            // A constant depends on no constant, and points into the globals it names.
            Value::Const(c) => {
                let mut globals = BTreeSet::new();
                let mut stack = vec![c];
                while let Some(c) = stack.pop() {
                    m.tick(1)?;
                    let Some(k) = self.module.consts().get(c) else {
                        continue;
                    };
                    match k.value {
                        Const::GlobalAddr(g) => {
                            globals.insert(g);
                        }
                        Const::Expr(_) => stack.extend_from_slice(k.value.children()),
                        _ => {}
                    }
                }
                Fact {
                    deps: Deps::default(),
                    points: Points::Into(globals),
                }
            }
        })
    }
}
