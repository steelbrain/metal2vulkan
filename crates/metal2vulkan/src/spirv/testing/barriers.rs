//! Workgroup barriers of an emitted module checked against the module's own uniformity: a test and
//! sweep check, never on a product path.
//!
//! Vulkan defines an `OpControlBarrier` with `Workgroup` execution scope only where every
//! invocation of the workgroup executes the same dynamic instance of it. [`workgroup_barriers`]
//! runs a workgroup-uniformity analysis of the entry function as emitted, independent of the
//! lowering's own analysis (`lower::simd::uniform`), and reports every workgroup barrier in a
//! block the analysis does not prove every invocation executes together.
//!
//! **Values.** A value is divergent when invocations of one workgroup may hold different values
//! for it. The sources are:
//!
//! - the per-invocation builtins (every `Input` builtin but the workgroup's id, count, and size,
//!   and the subgroup size and count) and every other `Input` or `Output` variable;
//! - `OpUndef`, atomics (an `OpAtomicLoad` too, but see below), subgroup operations, derivatives,
//!   helper-invocation tests, and ray queries;
//! - loads of `Workgroup` memory, of a pointer whose root the check cannot name, and of a
//!   `Function` or `Private` variable that some store makes divergent: a store of a divergent
//!   value, through a divergent pointer, or in a divergent block (a variable whose pointer a call
//!   receives counts as stored there);
//! - a call of a function that reads any of those, or receives a pointer whose memory it reads;
//! - a phi with a divergent incoming value or entered from a divergent block.
//!
//! Every other instruction is divergent exactly when one of its operands is. Loads of buffer,
//! uniform, push-constant, and image memory at a uniform address are uniform: SPIR-V no longer
//! says which buffer memory Metal declared `constant`, so the check takes buffer memory as fixed
//! during the dispatch, a weaker proof than the lowering's (which takes only `constant` memory as
//! fixed) but one that never contradicts it.
//!
//! **Barrier intervals.** A load of `Workgroup` memory at a uniform address in a uniform block is
//! uniform when the block executes a workgroup barrier with `WorkgroupMemory` semantics before it
//! and after it and nothing between the two writes workgroup memory: every invocation passes the
//! first barrier with every earlier write visible, and none writes before every invocation has
//! passed the second, so all of them read the same value. The predicated walk's loop vote
//! is such a load.
//!
//! **Uniform by construction.** A `Function` variable the emitter records in
//! `TypedModule::uniform_vars` (the `cont` flag of a predicated loop that iterates together, which
//! the lowering's analysis proved) is uniform.
//!
//! **Blocks.** A block is divergent when it is control dependent (on the post-dominator tree)
//! on a conditional branch or switch whose condition is divergent or whose block is divergent; a
//! reachable block from which no exit is reachable is divergent. An early `OpReturn` in a
//! divergent construct thus makes everything after the construct divergent. A block that ends in
//! `OpUnreachable` is outside the graph: a defined execution never reaches one, so an edge into
//! it is never taken and it is no exit, and a construct whose only other way out is
//! `OpUnreachable` (a `switch` default Metal's compiler marks so) leaves what follows it uniform.
//!
//! The analysis iterates to a fixed point; every flag only turns divergent.

use std::collections::HashMap;

use crate::budget::{Budget, Meter};
use crate::error::{Error, InternalError, InternalKind};
use crate::graph::{Cfg, Graph, PostDomTree};
use crate::spirv::ids::{BlockId, FuncId, GlobalId, Value, VarId};
use crate::spirv::ir::{
    GlobalKind, Inst, Scope, StorageClass, Terminator, Type, TypedModule, ValueOrigin,
};

/// What [`workgroup_barriers`] found.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BarrierReport {
    /// Workgroup barriers in reachable blocks of the entry function.
    pub barriers: u32,
    /// Those in a block the analysis does not prove workgroup uniform: (block, instruction
    /// index in it).
    pub nonuniform: Vec<(BlockId, usize)>,
    /// Workgroup barriers in other functions, which the check does not analyze (a call of one
    /// in a divergent block is not reported either).
    pub in_callees: u32,
}

fn invariant(what: &str) -> Error {
    Error::Internal(InternalError::new(
        InternalKind::Invariant,
        what.to_string(),
    ))
}

/// The builtins every invocation of a workgroup reads alike.
fn uniform_builtin(b: crate::spv::BuiltIn) -> bool {
    use crate::spv::BuiltIn as B;
    matches!(
        b,
        B::NumWorkgroups | B::WorkgroupSize | B::WorkgroupId | B::SubgroupSize | B::NumSubgroups
    )
}

/// Whether `inst` is a workgroup barrier with `WorkgroupMemory` semantics (a barrier interval's
/// bound).
fn interval_barrier(inst: &Inst) -> bool {
    matches!(inst, Inst::ControlBarrier { exec, sem, .. }
        if exec.scope == Scope::Workgroup && sem.sem.workgroup_memory)
}

/// Whether `inst` is an `OpControlBarrier` with `Workgroup` execution scope.
fn workgroup_barrier(inst: &Inst) -> bool {
    matches!(inst, Inst::ControlBarrier { exec, .. } if exec.scope == Scope::Workgroup)
}

/// The variable a pointer reaches through access chains.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Root {
    Global(GlobalId),
    Var(VarId),
    /// A function parameter.
    Param,
    /// Some other pointer (a loaded or converted one), of storage class `class`.
    Other(Option<StorageClass>),
}

/// The CFG of one function over its blocks' layout positions.
struct FnGraph<'a> {
    m: &'a TypedModule,
    blocks: &'a [BlockId],
    pos: &'a HashMap<BlockId, u32>,
}

fn targets(t: &Terminator) -> Vec<BlockId> {
    match t {
        Terminator::Branch(b) => vec![*b],
        Terminator::CondBranch { t, f, .. } => vec![*t, *f],
        Terminator::Switch { default, cases, .. } => std::iter::once(*default)
            .chain(cases.iter().map(|(_, b)| *b))
            .collect(),
        _ => Vec::new(),
    }
}

impl Graph for FnGraph<'_> {
    type Node = u32;

    fn node_count(&self) -> usize {
        self.blocks.len()
    }

    fn entry(&self) -> u32 {
        0
    }

    /// A block's targets but those that end in `OpUnreachable`: a defined execution never
    /// reaches one, so no edge into it is taken and it is no exit.
    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        let term = self
            .blocks
            .get(node as usize)
            .and_then(|b| self.m.blocks().get(b.index()))
            .map(|b| targets(&b.term))
            .unwrap_or_default();
        term.into_iter()
            .filter(|t| {
                self.m
                    .blocks()
                    .get(t.index())
                    .is_none_or(|b| !matches!(b.term, Terminator::Unreachable))
            })
            .filter_map(|t| self.pos.get(&t).copied())
    }
}

/// Per block, by instruction index: the interval between two `WorkgroupMemory` workgroup
/// barriers that holds the instruction and writes no workgroup memory, if any.
fn quiet_intervals(m: &TypedModule, insts: &[crate::spirv::ir::InstNode]) -> Vec<bool> {
    // Segment `s` is the stretch after the `s`-th interval barrier.
    let mut seg = Vec::with_capacity(insts.len());
    let mut writes: Vec<bool> = vec![false];
    let mut s = 0usize;
    for n in insts {
        if interval_barrier(&n.inst) {
            s += 1;
            writes.push(false);
            seg.push(usize::MAX);
            continue;
        }
        if writes_workgroup(m, &n.inst) {
            if let Some(w) = writes.get_mut(s) {
                *w = true;
            }
        }
        seg.push(s);
    }
    let closed = s;
    seg.into_iter()
        .map(|s| s != usize::MAX && s > 0 && s < closed && !writes.get(s).copied().unwrap_or(true))
        .collect()
}

/// Whether `inst` may write workgroup memory (a call may).
fn writes_workgroup(m: &TypedModule, inst: &Inst) -> bool {
    let wg = |p: Value| {
        !matches!(
            pointer_class(m, p),
            Some(
                StorageClass::Function
                    | StorageClass::Private
                    | StorageClass::StorageBuffer
                    | StorageClass::PhysicalStorageBuffer
                    | StorageClass::Uniform
                    | StorageClass::PushConstant
                    | StorageClass::Image
                    | StorageClass::Output
            )
        )
    };
    match inst {
        Inst::Store { ptr, .. }
        | Inst::AtomicStore { ptr, .. }
        | Inst::Atomic { ptr, .. }
        | Inst::AtomicCompareExchange { ptr, .. } => wg(*ptr),
        Inst::Call { .. } => true,
        _ => false,
    }
}

/// The storage class of pointer value `p`'s type.
fn pointer_class(m: &TypedModule, p: Value) -> Option<StorageClass> {
    let ty = m.value(p).ok()?.ty;
    match m.types().get(ty).ok()? {
        Type::Pointer { class, .. } => Some(*class),
        _ => None,
    }
}

/// The analysis state of one module.
struct Analysis<'a> {
    m: &'a TypedModule,
    /// By value: the instruction that defines it, as (function-local block position, index), in
    /// the entry function.
    defs: HashMap<Value, (u32, usize)>,
    entry_blocks: &'a [BlockId],
    div_val: Vec<bool>,
    div_var: Vec<bool>,
    div_global: Vec<bool>,
    div_block: Vec<bool>,
    /// By function: it reads per-invocation state ([`Analysis::impure`]).
    impure: Vec<bool>,
}

impl Analysis<'_> {
    fn val(&self, v: Value) -> bool {
        self.div_val.get(v.index()).copied().unwrap_or(true)
    }

    fn root(&self, mut p: Value) -> Root {
        // Access chains nest at most as deep as the types; a bound keeps a malformed module finite.
        for _ in 0..64 {
            let Ok(def) = self.m.value(p) else {
                return Root::Other(None);
            };
            match def.origin {
                ValueOrigin::Global(g) => return Root::Global(g),
                ValueOrigin::Var(v) => return Root::Var(v),
                ValueOrigin::Param { .. } => return Root::Param,
                ValueOrigin::Inst(_) => {}
                _ => return Root::Other(pointer_class(self.m, p)),
            }
            let base = self
                .defs
                .get(&p)
                .and_then(|&(b, i)| self.inst_at(b, i))
                .and_then(|n| match &n.inst {
                    Inst::AccessChain { base, .. } => Some(*base),
                    _ => None,
                });
            match base {
                Some(b) => p = b,
                None => return Root::Other(pointer_class(self.m, p)),
            }
        }
        Root::Other(None)
    }

    fn inst_at(&self, b: u32, i: usize) -> Option<&crate::spirv::ir::InstNode> {
        let id = self.entry_blocks.get(b as usize)?;
        self.m.blocks().get(id.index())?.insts.get(i)
    }

    /// Whether memory at `root` may hold different values for different invocations (loaded at
    /// a uniform address, outside a quiet barrier interval).
    fn memory_divergent(&self, root: Root) -> bool {
        match root {
            Root::Var(v) => self.div_var.get(v.index()).copied().unwrap_or(true),
            Root::Global(g) => {
                let Some(gv) = self.m.globals().get(g.index()) else {
                    return true;
                };
                match (gv.class, &gv.kind) {
                    (StorageClass::Input, GlobalKind::BuiltIn { builtin, .. }) => {
                        !uniform_builtin(*builtin)
                    }
                    (StorageClass::Private, _) => {
                        self.div_global.get(g.index()).copied().unwrap_or(true)
                    }
                    (
                        StorageClass::StorageBuffer
                        | StorageClass::Uniform
                        | StorageClass::PushConstant
                        | StorageClass::UniformConstant
                        | StorageClass::PhysicalStorageBuffer
                        | StorageClass::Image,
                        _,
                    ) => false,
                    _ => true,
                }
            }
            Root::Param => true,
            Root::Other(class) => !matches!(
                class,
                Some(
                    StorageClass::StorageBuffer
                        | StorageClass::PhysicalStorageBuffer
                        | StorageClass::Uniform
                        | StorageClass::PushConstant
                        | StorageClass::UniformConstant
                        | StorageClass::Image
                )
            ),
        }
    }

    fn is_workgroup(&self, root: Root) -> bool {
        match root {
            Root::Global(g) => self
                .m
                .globals()
                .get(g.index())
                .is_some_and(|gv| gv.class == StorageClass::Workgroup),
            _ => false,
        }
    }

    /// A load of pointer `ptr` at block `b`, instruction `i` (`quiet`: in a quiet barrier
    /// interval).
    fn load(&self, ptr: Value, b: u32, quiet: bool) -> bool {
        let root = self.root(ptr);
        if self.val(ptr) {
            return true;
        }
        if self.is_workgroup(root) {
            let block = self.div_block.get(b as usize).copied().unwrap_or(true);
            return block || !quiet;
        }
        self.memory_divergent(root)
    }

    /// Whether the result of `inst` (block `b`, quiet interval `quiet`) is divergent.
    fn eval(&self, inst: &Inst, b: u32, quiet: bool) -> bool {
        match inst {
            Inst::Load { ptr, .. } => self.load(*ptr, b, quiet),
            Inst::AtomicLoad { ptr, .. } => {
                let root = self.root(*ptr);
                !self.is_workgroup(root) || self.load(*ptr, b, quiet)
            }
            Inst::Undef
            | Inst::Atomic { .. }
            | Inst::AtomicCompareExchange { .. }
            | Inst::Derivative { .. }
            | Inst::Group { .. }
            | Inst::ImageQueryLod { .. }
            | Inst::RayQuery(_) => true,
            Inst::Call { func, args } => {
                self.impure.get(func.index()).copied().unwrap_or(true)
                    || args.iter().any(|a| self.val(*a))
            }
            other => {
                let mut any = false;
                other.for_each_value(|v| any |= self.val(v));
                any
            }
        }
    }

    /// Mark the variable at `root` divergent; returns whether it changed.
    fn taint(&mut self, root: Root) -> bool {
        let slot = match root {
            Root::Var(v) => {
                if self.m.uniform_vars().contains(&v) {
                    return false;
                }
                self.div_var.get_mut(v.index())
            }
            Root::Global(g) => match self.m.globals().get(g.index()) {
                Some(gv) if gv.class == StorageClass::Private => self.div_global.get_mut(g.index()),
                _ => None,
            },
            _ => None,
        };
        match slot {
            Some(s) if !*s => {
                *s = true;
                true
            }
            _ => false,
        }
    }
}

/// Which functions read per-invocation state: a per-invocation builtin, `Input`, `Output`,
/// `Workgroup`, or `Private` memory, or memory through a parameter or an unnamed pointer; an
/// atomic, `OpUndef`, subgroup operation, derivative, helper test, or ray query; or a call of
/// such a function. Also counts the workgroup barriers outside `entry`.
fn impure_functions(m: &TypedModule, entry: FuncId) -> (Vec<bool>, u32) {
    let n = m.functions().len();
    let mut impure = vec![false; n];
    let mut barriers = 0u32;
    let reads = |p: Value| -> bool {
        // Follow access chains inside `f`.
        let mut v = p;
        for _ in 0..64 {
            let Ok(def) = m.value(v) else { return true };
            match def.origin {
                ValueOrigin::Global(g) => {
                    let Some(gv) = m.globals().get(g.index()) else {
                        return true;
                    };
                    return match (gv.class, &gv.kind) {
                        (StorageClass::Input, GlobalKind::BuiltIn { builtin, .. }) => {
                            !uniform_builtin(*builtin)
                        }
                        (
                            StorageClass::StorageBuffer
                            | StorageClass::Uniform
                            | StorageClass::PushConstant
                            | StorageClass::UniformConstant
                            | StorageClass::PhysicalStorageBuffer
                            | StorageClass::Image,
                            _,
                        ) => false,
                        _ => true,
                    };
                }
                // The function's own variables hold what it computed.
                ValueOrigin::Var(_) => return false,
                ValueOrigin::Inst(b) => {
                    let base = m
                        .blocks()
                        .get(b.index())
                        .and_then(|blk| blk.insts.iter().find(|n| n.result == Some(v)))
                        .and_then(|n| match &n.inst {
                            Inst::AccessChain { base, .. } => Some(*base),
                            _ => None,
                        });
                    match base {
                        Some(x) => v = x,
                        None => {
                            return !matches!(
                                pointer_class(m, v),
                                Some(
                                    StorageClass::StorageBuffer
                                        | StorageClass::PhysicalStorageBuffer
                                        | StorageClass::Uniform
                                        | StorageClass::PushConstant
                                        | StorageClass::UniformConstant
                                        | StorageClass::Image
                                )
                            )
                        }
                    }
                }
                _ => return true,
            }
        }
        true
    };
    // The call graph is acyclic (the verifier checks it); iterate to a fixed point.
    let mut changed = true;
    while changed {
        changed = false;
        for (fi, f) in m.functions().iter().enumerate() {
            if fi == entry.index() || impure.get(fi).copied().unwrap_or(true) {
                continue;
            }
            let mut bad = false;
            for b in &f.blocks {
                let Some(blk) = m.blocks().get(b.index()) else {
                    continue;
                };
                for n in &blk.insts {
                    bad |= match &n.inst {
                        Inst::Load { ptr, .. } => reads(*ptr),
                        Inst::AtomicLoad { .. }
                        | Inst::Atomic { .. }
                        | Inst::AtomicCompareExchange { .. }
                        | Inst::Undef
                        | Inst::Group { .. }
                        | Inst::Derivative { .. }
                        | Inst::ImageQueryLod { .. }
                        | Inst::RayQuery(_) => true,
                        Inst::Call { func, .. } => {
                            impure.get(func.index()).copied().unwrap_or(true)
                        }
                        _ => false,
                    };
                }
            }
            if bad {
                if let Some(x) = impure.get_mut(fi) {
                    *x = true;
                }
                changed = true;
            }
        }
    }
    for (fi, f) in m.functions().iter().enumerate() {
        if fi == entry.index() {
            continue;
        }
        for b in &f.blocks {
            if let Some(blk) = m.blocks().get(b.index()) {
                barriers += blk
                    .insts
                    .iter()
                    .filter(|n| workgroup_barrier(&n.inst))
                    .count() as u32;
            }
        }
    }
    (impure, barriers)
}

/// Check every workgroup barrier of `m`'s entry function against the module's own uniformity
/// (see the module documentation).
pub fn workgroup_barriers(m: &TypedModule) -> Result<BarrierReport, Error> {
    workgroup_barriers_of(m, m.entry().func)
}

/// [`workgroup_barriers`] of function `entry` of `m`, taken as called by every invocation of the
/// workgroup together with no argument: an object or mesh function's body, which its entry wrapper
/// calls between its own barriers (`lower::mesh::wrap_entry`). Its `in_callees` counts the
/// workgroup barriers of every other function, the wrapper's among them.
pub fn workgroup_barriers_of(m: &TypedModule, entry: FuncId) -> Result<BarrierReport, Error> {
    let f = m
        .functions()
        .get(entry.index())
        .ok_or_else(|| invariant("no entry function"))?;
    let pos: HashMap<BlockId, u32> = f
        .blocks
        .iter()
        .enumerate()
        .map(|(i, b)| (*b, i as u32))
        .collect();
    let mut meter = Meter::new(&Budget::default());
    let graph = FnGraph {
        m,
        blocks: &f.blocks,
        pos: &pos,
    };
    let cfg = Cfg::build(&graph, &mut meter)?;
    let pdom = PostDomTree::compute(&cfg, &mut meter)?;
    let (impure, in_callees) = impure_functions(m, entry);
    let mut defs = HashMap::new();
    let mut quiet: Vec<Vec<bool>> = Vec::with_capacity(f.blocks.len());
    for (bp, b) in f.blocks.iter().enumerate() {
        let blk = m
            .blocks()
            .get(b.index())
            .ok_or_else(|| invariant("a block out of range"))?;
        for (i, n) in blk.insts.iter().enumerate() {
            if let Some(r) = n.result {
                defs.insert(r, (bp as u32, i));
            }
        }
        quiet.push(quiet_intervals(m, &blk.insts));
    }
    let mut a = Analysis {
        m,
        defs,
        entry_blocks: &f.blocks,
        div_val: vec![false; m.values().len()],
        div_var: vec![false; m.vars().len()],
        div_global: vec![false; m.globals().len()],
        div_block: vec![false; f.blocks.len()],
        impure,
    };
    // Undefined values are divergent wherever they are defined; so are the blocks no exit is
    // reachable from.
    for (i, d) in m.values().iter().enumerate() {
        if matches!(d.origin, ValueOrigin::Param { .. }) {
            if let Some(x) = a.div_val.get_mut(i) {
                *x = true;
            }
        }
    }
    for bp in 0..f.blocks.len() as u32 {
        if cfg.is_reachable(bp) && !pdom.is_reachable(bp) {
            if let Some(x) = a.div_block.get_mut(bp as usize) {
                *x = true;
            }
        }
    }
    let mut marked = vec![false; f.blocks.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for (bp, b) in f.blocks.iter().enumerate() {
            let Some(blk) = m.blocks().get(b.index()) else {
                continue;
            };
            let bpu = bp as u32;
            if !cfg.is_reachable(bpu) {
                continue;
            }
            for phi in &blk.phis {
                let d = phi.incoming.iter().any(|(v, from)| {
                    a.val(*v)
                        || pos
                            .get(from)
                            .and_then(|p| a.div_block.get(*p as usize))
                            .copied()
                            .unwrap_or(true)
                });
                if d && !a.val(phi.result) {
                    if let Some(x) = a.div_val.get_mut(phi.result.index()) {
                        *x = true;
                    }
                    changed = true;
                }
            }
            let q = quiet.get(bp).map(Vec::as_slice).unwrap_or(&[]);
            let block_div = a.div_block.get(bp).copied().unwrap_or(true);
            for (i, n) in blk.insts.iter().enumerate() {
                if let Some(r) = n.result {
                    if !a.val(r) && a.eval(&n.inst, bpu, q.get(i).copied().unwrap_or(false)) {
                        if let Some(x) = a.div_val.get_mut(r.index()) {
                            *x = true;
                        }
                        changed = true;
                    }
                }
                match &n.inst {
                    Inst::Store { ptr, value, .. } => {
                        if block_div || a.val(*ptr) || a.val(*value) {
                            let root = a.root(*ptr);
                            changed |= a.taint(root);
                        }
                    }
                    Inst::Call { func, args } => {
                        let bad = block_div
                            || a.impure.get(func.index()).copied().unwrap_or(true)
                            || args.iter().any(|x| a.val(*x));
                        for x in args.iter() {
                            if pointer_class(m, *x).is_some() && bad {
                                let root = a.root(*x);
                                changed |= a.taint(root);
                            }
                        }
                    }
                    _ => {}
                }
            }
            // A divergent branch, or any branch of a divergent block, makes the blocks control
            // dependent on it divergent.
            let branch_div = match &blk.term {
                Terminator::CondBranch { cond, .. } => Some(a.val(*cond)),
                Terminator::Switch { sel, .. } => Some(a.val(*sel)),
                _ => None,
            };
            let Some(d) = branch_div else { continue };
            if !(d || block_div) || marked.get(bp).copied().unwrap_or(true) {
                continue;
            }
            if let Some(x) = marked.get_mut(bp) {
                *x = true;
            }
            let stop = pdom.ipdom(bpu);
            for t in targets(&blk.term) {
                let mut x = pos.get(&t).copied();
                while let Some(xb) = x {
                    if Some(xb) == stop {
                        break;
                    }
                    if let Some(s) = a.div_block.get_mut(xb as usize) {
                        if !*s {
                            *s = true;
                            changed = true;
                        }
                    }
                    x = pdom.ipdom(xb);
                }
            }
        }
    }
    let mut report = BarrierReport {
        in_callees,
        ..BarrierReport::default()
    };
    for (bp, b) in f.blocks.iter().enumerate() {
        if !cfg.is_reachable(bp as u32) {
            continue;
        }
        let Some(blk) = m.blocks().get(b.index()) else {
            continue;
        };
        for (i, n) in blk.insts.iter().enumerate() {
            if !workgroup_barrier(&n.inst) {
                continue;
            }
            report.barriers += 1;
            if a.div_block.get(bp).copied().unwrap_or(true) {
                report.nonuniform.push((*b, i));
            }
        }
    }
    meter.release(pdom.ledger_bytes());
    meter.release(cfg.ledger_bytes());
    Ok(report)
}
