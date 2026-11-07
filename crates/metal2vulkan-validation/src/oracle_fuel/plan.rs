//! What to instrument: the functions that need the fuel state, their loops, and which loops are
//! barrier loops with which budget. Read-only over the verified module.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use metal2vulkan_air::budget::Meter;
use metal2vulkan_air::graph::{Cfg, LoopForest, LoopId};
use metal2vulkan_air::intrinsics::{AirIntrinsic, Sync};
use metal2vulkan_air::ir::{
    BlockId, Body, Callee, FuncId, InstId, InstKind, Module, Section, TailKind,
};

use super::counted::{self, COUNTED_LOOP_LIMIT};
use super::{FuelError, FuelUnsupported, LoopSelection, ORACLE_NEST_LIMIT};
use crate::call_targets::{address_taken, closed_calls, const_funcs, ClosedCalls};

/// The budget of a barrier loop of barrier depth `depth` (as amended): `max(2, ⌊F_b^(1/d)⌋)`,
/// computed exactly in integers. `depth` is at least 1 for a barrier loop; 0 is treated as 1.
pub(super) fn barrier_budget(barrier_fuel: u32, depth: u32) -> u32 {
    let d = depth.max(1);
    // The largest k with k^d ≤ F_b, by bisection over [0, F_b].
    let fits = |k: u64| -> bool {
        let mut acc: u64 = 1;
        for _ in 0..d {
            acc = acc.saturating_mul(k);
            if acc > u64::from(barrier_fuel) {
                return false;
            }
        }
        true
    };
    let (mut lo, mut hi) = (0u64, u64::from(barrier_fuel));
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    u32::try_from(lo).unwrap_or(u32::MAX).max(2)
}

/// The instrumentation plan of one module for one entry.
pub(super) struct Plan {
    pub entry: FuncId,
    /// Every function that takes the fuel state (the entry included), in module order.
    pub funcs: BTreeMap<FuncId, FnPlan>,
    /// The loops of the functions the entry reaches that [`LoopSelection::Oracle`] leaves
    /// unchecked as counted, and the largest product of header visits along a chain of them (0
    /// when there is none; `Exemptions`).
    pub exempt_loops: u32,
    pub exempt_nest: u64,
}

/// One function that takes the fuel state.
pub(super) struct FnPlan {
    pub loops: Vec<LoopPlan>,
    /// Calls to other functions that take the fuel state: their arguments grow, and the caller
    /// tests `forced` after them.
    pub fueled_calls: BTreeSet<InstId>,
}

/// One loop of a [`FnPlan`].
pub(super) struct LoopPlan {
    pub header: BlockId,
    /// The blocks of the loop with an edge to the header (ascending, deduplicated). A check is
    /// spliced on each of these edges.
    pub latches: Vec<BlockId>,
    /// `Some` for a barrier loop.
    pub barrier: Option<BarrierLoop>,
}

/// A barrier loop's counter.
pub(super) struct BarrierLoop {
    /// `max(2, ⌊F_b^(1/d)⌋)`.
    pub budget: u32,
    /// The blocks outside the loop with an edge to its header (ascending, deduplicated): the
    /// counter is reset on each of these edges.
    pub preheaders: Vec<BlockId>,
}

/// How a call can affect the plan.
#[derive(Clone, PartialEq, Eq)]
enum CallKind {
    /// A call that runs one of these functions this module defines: a direct call (one), or an
    /// indirect call whose callee is closed (any of its targets; [`crate::call_targets`]).
    Local(Box<[FuncId]>),
    /// A barrier, or a call the transform cannot see into (an indirect call whose callee is not
    /// closed, an external declaration, an unmodeled intrinsic): it may reach a barrier.
    Barrier,
    /// A modeled intrinsic other than a barrier.
    Leaf,
}

impl CallKind {
    /// The functions the call may run that this module defines.
    fn targets(&self) -> &[FuncId] {
        match self {
            CallKind::Local(targets) => targets,
            CallKind::Barrier | CallKind::Leaf => &[],
        }
    }
}

/// The kind of the call `call` in `f`, calling `callee`.
fn call_kind(
    module: &Module,
    closed: &ClosedCalls,
    f: FuncId,
    call: InstId,
    callee: &Callee,
) -> CallKind {
    match callee {
        Callee::Direct(g) => match module.func(*g) {
            Some(function) if function.body.is_some() => CallKind::Local(Box::new([*g])),
            _ => CallKind::Barrier,
        },
        Callee::Air {
            op: AirIntrinsic::Sync(Sync::Barrier { .. }),
            ..
        } => CallKind::Barrier,
        Callee::Air { .. } | Callee::Llvm { .. } => CallKind::Leaf,
        Callee::Indirect { .. } => match closed.targets(f, call) {
            Some(targets) => CallKind::Local(targets.iter().copied().collect()),
            None => CallKind::Barrier,
        },
        Callee::UnknownIntrinsic(_) | Callee::Unbound { .. } => CallKind::Barrier,
    }
}

/// Per defined, reachable function: its calls and loops.
struct FnFacts {
    /// `(call, its block, kind)` for every call in layout order.
    calls: Vec<(InstId, BlockId, CallKind)>,
    forest: LoopForest<BlockId>,
    /// Header -> predecessors (every block, reachable or not, once per block).
    preds: BTreeMap<BlockId, BTreeSet<BlockId>>,
}

pub(super) fn name(module: &Module, f: FuncId) -> String {
    module
        .func(f)
        .and_then(|function| module.symbols().get(function.name))
        .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
        .unwrap_or_default()
}

fn internal(e: impl Into<metal2vulkan::error::Error>) -> FuelError {
    FuelError::Internal(e.into().to_string())
}

pub(super) fn plan(
    module: &Module,
    entry: &str,
    barrier_fuel: u32,
    selection: LoopSelection,
    meter: &mut Meter,
) -> Result<Plan, FuelError> {
    let entry_id = module
        .parts()
        .funcs
        .iter()
        .find(|(_, f)| f.body.is_some() && module.symbols().get(f.name) == Some(entry.as_bytes()))
        .map(|(id, _)| id)
        .ok_or_else(|| FuelUnsupported::MissingEntry {
            entry: entry.to_string(),
        })?;

    // Reachable functions over local calls (direct, and indirect with a closed callee), in
    // reverse postorder (callers first), and the absence of a call cycle among them.
    let closed = closed_calls(module);
    let order = reverse_postorder(module, &closed, entry_id)?;
    let mut facts = BTreeMap::new();
    for &f in &order {
        facts.insert(f, fn_facts(module, &closed, f, meter)?);
    }
    let mut reaches_barrier = BTreeSet::new();
    for &f in order.iter().rev() {
        if facts[&f].calls.iter().any(|(_, _, k)| match k {
            CallKind::Barrier => true,
            CallKind::Local(targets) => targets.iter().any(|g| reaches_barrier.contains(g)),
            CallKind::Leaf => false,
        }) {
            reaches_barrier.insert(f);
        }
    }

    let depth = BarrierDepth::compute(&order, &facts, &reaches_barrier);
    let exempt = match selection {
        LoopSelection::Oracle => Exemptions::compute(module, &order, &facts, ORACLE_NEST_LIMIT),
        _ => Exemptions::none(),
    };
    // The loops that get a check, per function (every one but the oracle's exempt counted nests,
    // or a compile probe's selection), indexed by loop.
    let counted = |f: FuncId, l: LoopId| {
        let body = module.func(f).and_then(|function| function.body.as_ref());
        let fact = &facts[&f];
        body.and_then(|body| {
            counted::header_visits(
                module,
                body,
                &fact.forest,
                &fact.preds,
                l,
                COUNTED_LOOP_LIMIT,
            )
        })
        .is_some()
    };
    let selected: BTreeMap<FuncId, Vec<bool>> = order
        .iter()
        .map(|&f| {
            let picks = facts[&f]
                .forest
                .loops()
                .map(|l| match selection {
                    LoopSelection::Oracle => !exempt.is_exempt(f, l),
                    LoopSelection::All => true,
                    LoopSelection::Entry => f == entry_id,
                    LoopSelection::Callees => f != entry_id,
                    LoopSelection::Barrier => depth.is_barrier(f, l),
                    LoopSelection::NonBarrier => !depth.is_barrier(f, l),
                    LoopSelection::Counted => counted(f, l),
                    LoopSelection::Uncounted => !counted(f, l),
                })
                .collect();
            (f, picks)
        })
        .collect();
    let checked = |f: FuncId, l: LoopId| {
        selected
            .get(&f)
            .and_then(|picks| picks.get(l.index()))
            .copied()
            .unwrap_or(false)
    };

    // Who needs the state: a function with a loop or a call to one that needs it (callees
    // first). A closed indirect call passes the state to whichever target it runs, so when one of
    // its targets needs the state every target takes it (their prototypes stay one type); that
    // can reach further callers, so the pass repeats until nothing changes.
    let mut needs = BTreeSet::new();
    loop {
        let before = needs.len();
        for &f in order.iter().rev() {
            let fact = &facts[&f];
            let calls_needing = fact
                .calls
                .iter()
                .any(|(_, _, k)| k.targets().iter().any(|g| needs.contains(g)));
            if fact.forest.loops().any(|l| checked(f, l)) || calls_needing {
                needs.insert(f);
            }
        }
        for fact in facts.values() {
            for (_, _, k) in &fact.calls {
                if k.targets().iter().any(|g| needs.contains(g)) {
                    needs.extend(k.targets().iter().copied());
                }
            }
        }
        if needs.len() == before {
            break;
        }
    }
    // Everything that needs the state must be reached only by local calls from the entry: its
    // address escapes nowhere a call the transform cannot see could take it.
    let taken = address_taken(module);
    let started = runtime_started(module);
    if taken.contains(&entry_id) || called(module).contains(&entry_id) {
        return Err(FuelUnsupported::EntryCalled {
            entry: entry.to_string(),
        }
        .into());
    }
    for &f in &needs {
        if f != entry_id && (closed.escaped.contains(&f) || started.contains(&f)) {
            return Err(FuelUnsupported::UnthreadableLoop {
                function: name(module, f),
            }
            .into());
        }
    }
    // A loop the entry cannot reach but something else may run is not bounded either, and a
    // function outside the plan cannot pass the state to one inside it.
    for (f, function) in module.parts().funcs.iter() {
        let Some(body) = &function.body else { continue };
        if !needs.contains(&f) {
            for (at, inst) in body.insts.iter() {
                if let InstKind::Call(call) = &inst.kind {
                    let kind = call_kind(module, &closed, f, at, &call.callee);
                    if let Some(&g) = kind.targets().iter().find(|g| needs.contains(g)) {
                        return Err(FuelUnsupported::UnthreadableLoop {
                            function: name(module, g),
                        }
                        .into());
                    }
                }
            }
        }
        if needs.contains(&f) || !(taken.contains(&f) || started.contains(&f)) {
            continue;
        }
        let cfg = Cfg::build(body, meter).map_err(internal)?;
        let cyclic = cfg
            .preorder()
            .any(|b| cfg.successors(b).any(|s| cfg.is_retreating(b, s)));
        meter.release(cfg.ledger_bytes());
        if cyclic {
            return Err(FuelUnsupported::UnthreadableLoop {
                function: name(module, f),
            }
            .into());
        }
    }

    let mut funcs = BTreeMap::new();
    for &f in &order {
        if !needs.contains(&f) {
            continue;
        }
        let fact = &facts[&f];
        let mut loops = Vec::new();
        for l in fact.forest.loops().filter(|&l| checked(f, l)) {
            let header = fact.forest.header(l);
            let preds = fact.preds.get(&header).cloned().unwrap_or_default();
            let (latches, preheaders): (Vec<_>, Vec<_>) =
                preds.into_iter().partition(|&p| fact.forest.contains(l, p));
            let barrier = if depth.is_barrier(f, l) {
                if !fact.forest.is_reducible(l) {
                    return Err(FuelUnsupported::IrreducibleBarrierLoop {
                        function: name(module, f),
                    }
                    .into());
                }
                Some(BarrierLoop {
                    budget: barrier_budget(barrier_fuel, depth.depth(f, l)),
                    preheaders,
                })
            } else {
                None
            };
            loops.push(LoopPlan {
                header,
                latches,
                barrier,
            });
        }
        // A `musttail` call requires its caller's prototype to match its callee's, and `f`'s
        // grows (by the status buffer in the entry, by the state and the status buffer
        // elsewhere). One that would pass the state also promises the callee touches none of the
        // caller's allocas, which the state is, and must return before the caller's forced test.
        if fact
            .calls
            .iter()
            .any(|&(call, _, _)| tail(module, f, call) == Some(TailKind::MustTail))
        {
            return Err(FuelUnsupported::MustTailCall {
                function: name(module, f),
            }
            .into());
        }
        let fueled_calls = fact
            .calls
            .iter()
            .filter(|(_, _, k)| k.targets().iter().any(|g| needs.contains(g)))
            .map(|&(i, _, _)| i)
            .collect();
        funcs.insert(
            f,
            FnPlan {
                loops,
                fueled_calls,
            },
        );
    }
    for fact in facts.values() {
        meter.release(fact.forest.ledger_bytes());
    }
    Ok(Plan {
        entry: entry_id,
        funcs,
        exempt_loops: exempt.count(),
        exempt_nest: exempt.largest_nest(entry_id),
    })
}

/// The counted loops [`LoopSelection::Oracle`] leaves unchecked.
///
/// Callees first, and innermost loops first within a function, a loop is exempt when it is
/// counted (`counted::header_visits`) and its header visits times the largest product of the
/// exempt loops it holds is at most the limit. What a loop holds is the loops nested in it in its
/// function and, through every call in its blocks, the chains each target runs from its top. A
/// checked loop between two exempt ones counts 1: its first pass takes no back edge, so it draws
/// nothing. Every chain of nested exempt loops, across calls, therefore visits its innermost
/// header at most the limit times per entry of its outermost loop. Deciding inner loops first
/// leaves the check on the outer loop of a nest past the limit, where it is passed least often.
struct Exemptions {
    /// Per function, per loop: whether it is exempt.
    exempt: BTreeMap<FuncId, Vec<bool>>,
    /// Per function: the largest product along a chain of exempt loops a call to it runs.
    nest: BTreeMap<FuncId, u64>,
}

impl Exemptions {
    /// No loop exempt: every selection but the oracle's.
    fn none() -> Exemptions {
        Exemptions {
            exempt: BTreeMap::new(),
            nest: BTreeMap::new(),
        }
    }

    fn compute(
        module: &Module,
        order: &[FuncId],
        facts: &BTreeMap<FuncId, FnFacts>,
        limit: u64,
    ) -> Exemptions {
        let mut exempt = BTreeMap::new();
        let mut nest = BTreeMap::<FuncId, u64>::new();
        for &f in order.iter().rev() {
            let fact = &facts[&f];
            let forest = &fact.forest;
            let body = module.func(f).and_then(|function| function.body.as_ref());
            // The largest exempt product each loop holds, and the function's top holds.
            let mut held = vec![1u64; forest.len()];
            let mut top = 1u64;
            for (_, b, k) in &fact.calls {
                let chain = k
                    .targets()
                    .iter()
                    .map(|g| nest.get(g).copied().unwrap_or(1))
                    .max()
                    .unwrap_or(1);
                match forest.innermost(*b) {
                    Some(l) => held[l.index()] = held[l.index()].max(chain),
                    None => top = top.max(chain),
                }
            }
            let mut picks = vec![false; forest.len()];
            for l in forest.innermost_first() {
                let inner = held[l.index()];
                let product = body
                    .and_then(|body| {
                        counted::header_visits(module, body, forest, &fact.preds, l, limit)
                    })
                    .map(|visits| visits.saturating_mul(inner))
                    .filter(|&product| product <= limit);
                picks[l.index()] = product.is_some();
                let through = product.unwrap_or(inner);
                match forest.parent(l) {
                    Some(p) => held[p.index()] = held[p.index()].max(through),
                    None => top = top.max(through),
                }
            }
            exempt.insert(f, picks);
            nest.insert(f, top);
        }
        Exemptions { exempt, nest }
    }

    fn is_exempt(&self, f: FuncId, l: LoopId) -> bool {
        self.exempt
            .get(&f)
            .and_then(|picks| picks.get(l.index()))
            .copied()
            .unwrap_or(false)
    }

    /// How many loops are exempt.
    fn count(&self) -> u32 {
        let n = self.exempt.values().flatten().filter(|&&e| e).count();
        u32::try_from(n).unwrap_or(u32::MAX)
    }

    /// The largest product of header visits along a chain of exempt loops the entry runs (0 when
    /// none is exempt).
    fn largest_nest(&self, entry: FuncId) -> u64 {
        if self.count() == 0 {
            0
        } else {
            self.nest.get(&entry).copied().unwrap_or(0)
        }
    }
}

/// The tail marker of the call `call` in `f`.
fn tail(module: &Module, f: FuncId, call: InstId) -> Option<TailKind> {
    let body = module.func(f)?.body.as_ref()?;
    match &body.insts.get(call)?.kind {
        InstKind::Call(call) => Some(call.tail),
        _ => None,
    }
}

/// Defined functions reachable from `entry` over local calls, callers before callees; a cycle
/// is refused.
fn reverse_postorder(
    module: &Module,
    closed: &ClosedCalls,
    entry: FuncId,
) -> Result<Vec<FuncId>, FuelError> {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        Open,
        Done,
    }
    let callees = |f: FuncId| -> Vec<FuncId> {
        let mut out = Vec::new();
        if let Some(body) = module.func(f).and_then(|func| func.body.as_ref()) {
            for (at, inst) in body.insts.iter() {
                if let InstKind::Call(call) = &inst.kind {
                    let kind = call_kind(module, closed, f, at, &call.callee);
                    out.extend(kind.targets().iter().copied());
                }
            }
        }
        out
    };
    let mut marks = BTreeMap::<FuncId, Mark>::new();
    let mut post = Vec::new();
    let mut stack = vec![(entry, callees(entry), 0usize)];
    marks.insert(entry, Mark::Open);
    while let Some((f, succs, next)) = stack.last_mut() {
        if let Some(&g) = succs.get(*next) {
            *next += 1;
            match marks.get(&g) {
                Some(Mark::Open) => {
                    return Err(FuelUnsupported::CallCycle {
                        function: name(module, g),
                    }
                    .into())
                }
                Some(Mark::Done) => {}
                None => {
                    marks.insert(g, Mark::Open);
                    let s = callees(g);
                    stack.push((g, s, 0));
                }
            }
        } else {
            let f = *f;
            marks.insert(f, Mark::Done);
            post.push(f);
            stack.pop();
        }
    }
    post.reverse();
    Ok(post)
}

fn fn_facts(
    module: &Module,
    closed: &ClosedCalls,
    f: FuncId,
    meter: &mut Meter,
) -> Result<FnFacts, FuelError> {
    let body: &Body = module
        .func(f)
        .and_then(|func| func.body.as_ref())
        .ok_or_else(|| FuelError::Internal(format!("@{} has no body", name(module, f))))?;
    let mut calls = Vec::new();
    let mut preds = BTreeMap::<BlockId, BTreeSet<BlockId>>::new();
    for (b, block) in body.blocks.iter() {
        for i in block.inst_ids() {
            if let Some(InstKind::Call(call)) = body.insts.get(i).map(|inst| &inst.kind) {
                calls.push((i, b, call_kind(module, closed, f, i, &call.callee)));
            }
        }
        for s in body.successors(b) {
            preds.entry(s).or_default().insert(b);
        }
    }
    let cfg = Cfg::build(body, meter).map_err(internal)?;
    let forest = LoopForest::compute(&cfg, meter).map_err(internal)?;
    meter.release(cfg.ledger_bytes());
    Ok(FnFacts {
        calls,
        forest,
        preds,
    })
}

/// Barrier loops and their depth: the longest chain of nested barrier loops through a loop, across
/// calls.
struct BarrierDepth {
    /// Per function, per loop: whether it is a barrier loop.
    barrier: BTreeMap<FuncId, Vec<bool>>,
    /// Per function, per loop: barrier loops enclosing it in its function, itself included.
    enclosing: BTreeMap<FuncId, Vec<u32>>,
    /// Per function, per loop: the longest barrier chain strictly below it.
    below: BTreeMap<FuncId, Vec<u32>>,
    /// Per function: the longest barrier chain enclosing any call to it.
    above: BTreeMap<FuncId, u32>,
}

impl BarrierDepth {
    fn compute(
        order: &[FuncId],
        facts: &BTreeMap<FuncId, FnFacts>,
        reaches_barrier: &BTreeSet<FuncId>,
    ) -> BarrierDepth {
        let mut barrier = BTreeMap::new();
        let mut enclosing = BTreeMap::new();
        for &f in order {
            let fact = &facts[&f];
            let forest = &fact.forest;
            let n = forest.len();
            let mut is_barrier = vec![false; n];
            for (_, b, k) in &fact.calls {
                let b = *b;
                let reaches = match k {
                    CallKind::Barrier => true,
                    CallKind::Local(targets) => targets.iter().any(|g| reaches_barrier.contains(g)),
                    CallKind::Leaf => false,
                };
                if !reaches {
                    continue;
                }
                // Every loop containing the call's block reaches a barrier.
                let mut l = forest.innermost(b);
                while let Some(id) = l {
                    is_barrier[id.index()] = true;
                    l = forest.parent(id);
                }
            }
            let mut enc = vec![0u32; n];
            for l in forest.loops() {
                let parent = forest.parent(l).map_or(0, |p| enc[p.index()]);
                enc[l.index()] = parent + u32::from(is_barrier[l.index()]);
            }
            barrier.insert(f, is_barrier);
            enclosing.insert(f, enc);
        }

        // Longest chains below, callees first.
        let mut below = BTreeMap::<FuncId, Vec<u32>>::new();
        let mut below_fn = BTreeMap::<FuncId, u32>::new();
        for &f in order.iter().rev() {
            let fact = &facts[&f];
            let forest = &fact.forest;
            let is_barrier = &barrier[&f];
            let mut below_loop = vec![0u32; forest.len()];
            let mut top = 0u32;
            let call_chain = |k: &CallKind| {
                k.targets()
                    .iter()
                    .map(|g| below_fn.get(g).copied().unwrap_or(0))
                    .max()
                    .unwrap_or(0)
            };
            for (_, b, k) in &fact.calls {
                let b = *b;
                let chain = call_chain(k);
                match forest.innermost(b) {
                    Some(l) => below_loop[l.index()] = below_loop[l.index()].max(chain),
                    None => top = top.max(chain),
                }
            }
            for l in forest.innermost_first() {
                let through = u32::from(is_barrier[l.index()]) + below_loop[l.index()];
                match forest.parent(l) {
                    Some(p) => below_loop[p.index()] = below_loop[p.index()].max(through),
                    None => top = top.max(through),
                }
            }
            below.insert(f, below_loop);
            below_fn.insert(f, top);
        }

        // Longest chains above, callers first.
        let mut above = BTreeMap::<FuncId, u32>::new();
        if let Some(&entry) = order.first() {
            above.insert(entry, 0);
        }
        for &f in order {
            let fact = &facts[&f];
            let here = above.get(&f).copied().unwrap_or(0);
            for (_, b, k) in &fact.calls {
                let enc = fact
                    .forest
                    .innermost(*b)
                    .map_or(0, |l| enclosing[&f][l.index()]);
                for &g in k.targets() {
                    let slot = above.entry(g).or_insert(0);
                    *slot = (*slot).max(here + enc);
                }
            }
        }
        BarrierDepth {
            barrier,
            enclosing,
            below,
            above,
        }
    }

    fn is_barrier(&self, f: FuncId, l: LoopId) -> bool {
        self.barrier
            .get(&f)
            .and_then(|v| v.get(l.index()))
            .copied()
            .unwrap_or(false)
    }

    fn depth(&self, f: FuncId, l: LoopId) -> u32 {
        let at = |m: &BTreeMap<FuncId, Vec<u32>>| {
            m.get(&f)
                .and_then(|v| v.get(l.index()))
                .copied()
                .unwrap_or(0)
        };
        self.above.get(&f).copied().unwrap_or(0) + at(&self.enclosing) + at(&self.below)
    }
}

/// Every function some call instruction calls directly.
fn called(module: &Module) -> BTreeSet<FuncId> {
    let mut out = BTreeSet::new();
    for function in module.parts().funcs.values() {
        let Some(body) = &function.body else { continue };
        for inst in body.insts.values() {
            if let InstKind::Call(call) = &inst.kind {
                if let Callee::Direct(f) = call.callee {
                    out.insert(f);
                }
            }
        }
    }
    out
}

/// Functions the runtime runs itself: initializer sections and `llvm.global_ctors`.
fn runtime_started(module: &Module) -> BTreeSet<FuncId> {
    let mut out = BTreeSet::new();
    for (f, function) in module.parts().funcs.iter() {
        if matches!(
            function.section,
            Some(Section::StaticInit | Section::FcInitializer)
        ) {
            out.insert(f);
        }
    }
    let mut seen = HashSet::new();
    for global in module.parts().globals.values() {
        if module.symbols().get(global.name) == Some(b"llvm.global_ctors".as_slice()) {
            if let Some(init) = global.init {
                const_funcs(module, init, &mut out, &mut seen);
            }
        }
    }
    out
}
