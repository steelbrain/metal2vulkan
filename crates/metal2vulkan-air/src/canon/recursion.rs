//! Recursion through visible functions as an explicit call stack.
//!
//! SPIR-V has no recursion. Metal runs a kernel whose visible functions call each other
//! recursively, through visible-function references or through visible-function tables, up to a
//! reach set by the compute pipeline's `maxCallStackDepth` `k` (measured on an M3 Max and an M4).
//! Each recursive part of the entry's call graph becomes a loop over an explicit stack of frames in
//! the thread's memory, in three steps:
//!
//! 1. **Preparation** ([`prepare`], before inlining). The call graph of the defined functions
//!    the entry reaches has an edge for each direct call, each call through a function's constant
//!    address at its own type, and, for each other call through a pointer of function type `T`,
//!    one to each function a table can hold whose type is `T` (a [`Candidate`]) and one to each
//!    function of type `T` whose address a reached body takes as a value (a function pointer passed
//!    as a value may hold it). Its cyclic components, unless the entry is in one, are the recursive
//!    parts. A component's *helpers*, its functions no pipeline links, fold into its linked
//!    functions when there are both, each helper is called in it only directly, and the helpers
//!    call each other in no cycle ([`folds`]): its members are then the linked functions, whose
//!    bodies take the helpers inlined, as Metal inlines a function that is not visible. Otherwise
//!    each of its functions is a member. Each member gets an *enter* marker, a declaration of its
//!    type that every direct call to it, and every call through its constant address, now calls
//!    (devirtualization too, `devirt::Cx::enters`). Each component gets a *dispatch* marker per
//!    type of its members a table can hold, `ret (i32 word, params...)`: devirtualization calls
//!    it for those members' words, one switch case block for all of them, with the word first
//!    (`devirt::Cx::groups`). Inlining takes no marker, so the rounds of inlining and
//!    devirtualization end.
//! 2. **Regions** ([`Recursion::islands`], after the rounds). A marker call outside its
//!    component's copies is a *root*. It gets a *region*: a copy of each member's body is
//!    appended to the entry's body, unreached, as a *head* block (a head marker call, then a
//!    *parameter* marker call per parameter the copy does not take as a value) and the member's
//!    blocks, each `ret` a *return* marker call and `unreachable`. A parameter that holds one
//!    value in every activation takes it ([`Recursion::fixed`]): the root's argument, when every
//!    recursive call passes the parameter on unchanged and the argument is defined outside every
//!    copy, so devirtualization reads a table the root passes as the root's; or a constant every
//!    call that reaches it passes, so a function pointer passed as a value is that function's
//!    constant address, which devirtualization calls through the function's enter marker. The
//!    root call now calls the region's *root* marker. Inlining and devirtualization run over the
//!    copies, and a root they reach there (a call into another component) gets its own region,
//!    until no root is new.
//! 3. **Assembly** ([`Recursion::assemble`]). Each region becomes a machine over a stack of `N`
//!    frames, `[N x {i32, slots...}]` (`N` at least 1), and a depth, both allocated in the entry
//!    block:
//!    - The root stores depth 0 and branches to the head of the member it calls (a dispatch: a
//!      switch on the word over the heads); its activation is frame 0's. A head's parameter
//!      markers become PHIs.
//!    - A recursive call (an enter or dispatch marker call in one of the region's copies) checks
//!      for room: with depth `d` and `d + 1 < N`, it stores its call site's number in frame
//!      `d + 1`, stores depth `d + 1`, and branches to the callee's head; with no room it calls
//!      nothing and yields zero. Its result is a PHI at its continuation.
//!    - A return branches to its return type's block. At depth 0 it leaves the machine to the
//!      root's continuation; otherwise it reads the call site's number from the frame, stores
//!      depth `d - 1`, and switches to that call site's continuation.
//!    - A copy's value live across one of its recursive calls (live into a continuation, its
//!      own call's result aside) is stored to its slot of the current frame after its definition
//!      and loaded at each use. A copy's `alloca` is a slot of the frame, and an address computed
//!      from one, or from a value defined outside the copy, by `getelementptr`, `bitcast`, or
//!      `addrspacecast` is computed again at each use, so neither is stored.
//!
//! **Capacity** (Metal's model, measured on an M4 under macOS 27.0.1 with the probes' shapes: a
//! 16-byte frame is refused at 16359, a 272-byte one at 963, a 288-byte pair at 909, a 32-byte
//! non-recursive chain at 8180). Metal reserves `k` times the frames of the pipeline's linked
//! functions, each rounded up to 16 bytes ([`reserved`]), and a recursion holds as many activations
//! as those bytes hold of its own frames: one function's holds `k`, its root's and `k - 1` calls,
//! whatever its frame, whether it recurses directly, through a table, or through a pointer passed
//! to a helper, and whether the kernel calls it or the helper (a function whose depth 0 answers a
//! nonzero word answers depth `k - 1` at `k`, depth 0 at `k` 1, so the root's activation is one of
//! the `k`); three 16-byte functions' hold `3k` (correct to 3 frames at `k`
//! 1). The *linked frames* `L` are each member's frame (its call site's number and its copy's
//! slots, as Metal's: a call's return and what it keeps across the call, packed largest alignment
//! first) and, for each other linked function, 16 bytes when it calls a visible function or through
//! a pointer and 0 for a leaf. A component's stack has `N = ceil(k L / s)` frames ([`frames`]), `s`
//! the smallest of its members' reserved frames: a lone function's `k`, and at least `k` always, so
//! at `k` 1 a lone function's root runs and calls nothing. Past `N` a call is not made and yields
//! zero, where Metal's answer is not a defined one (a recorded near-equivalent, API.md §10). `k` 0
//! is taken as 1. Plain Metal and its API validation answer a lone function's depths below `k`,
//! which the lowering does, and depth `k` too where the deepest call answers zero, which a call
//! past the stack yields; Metal's shader validation reaches depth `k` and refuses past 908 for the
//! 16-byte frame. Metal's backend keeps less in some frames than AIR keeps across the call, so a
//! component of unequal frames reaches further than Metal's, never less, on the shapes measured.
//!
//! **Creation.** Metal refuses a pipeline whose reserved stack, `k L`, passes [`CALL_STACK_BYTES`]
//! ("Compute pipeline exceeds available stack space"): the refusal is
//! `LimitExceeded(CallStackBytes)`, also for linked functions that do not recurse. The stacks are
//! the thread's Function storage, allocas like its others: selection holds the entry's private
//! memory to what the device holds correctly (`ProfileFields::function_storage`; a vertex or
//! fragment function on a device not measured to Metal's 192 KiB, noted), and refuses a recursion
//! past it as `Unsupported{Recursion}` needing more, since past it a driver can answer wrongly or
//! hang with no error.

use std::collections::{BTreeMap, BTreeSet};

use crate::budget::{Meter, CALL_STACK_BYTES};
use crate::error::{Error, Feature, Limit, Unsupported};
use crate::ir::{AddrSpace, BinOp, Block, BlockId, Body, Callee, CastOp, Const, ConstId};
use crate::ir::{Constant, FastMath, FuncId, Idx, Inst, InstId, InstKind, IntFlags};
use crate::ir::{IntPredicate, ModuleParts, Sym, Type, TypeId, TypeLayouts, Value};

use super::CallStack;
use crate::graph::{Graph, Sccs};

use super::devirt::{declare, FunctionWords};
use super::inline;
use super::link::{Candidate, Stub, Stubs};
use super::rewrite::{filled, invariant, scoped};

/// The frames of a stack at `maxCallStackDepth` `k`, the activations it holds: Metal's stack, `k`
/// times the pipeline's linked frames `linked`, holds that many of the component's `smallest`
/// reserved frame, and at least `k` (see the module docs).
pub(super) fn frames(k: u32, linked: u64, smallest: u64) -> u64 {
    let k = u64::from(k.max(1));
    k.saturating_mul(linked).div_ceil(smallest.max(1)).max(k)
}

/// Metal's stack at `maxCallStackDepth` `k` for the linked frames `linked`, refused past
/// [`CALL_STACK_BYTES`] as Metal refuses the pipeline.
fn metal_stack(k: u32, linked: u64) -> Result<u64, Error> {
    let bytes = u64::from(k.max(1)).saturating_mul(linked);
    if bytes > CALL_STACK_BYTES {
        return Err(Error::LimitExceeded(Limit::CallStackBytes));
    }
    Ok(bytes)
}

/// The frame Metal is modelled to reserve for one of `frame` bytes: rounded up to 16, at least 16.
pub(super) fn reserved(frame: u64) -> u64 {
    frame.max(1).div_ceil(16).saturating_mul(16)
}

/// What [`prepare`] found and made, and the regions made since.
pub(super) struct Recursion {
    k: u32,
    /// The linked frames of the linked functions outside every component (see the module docs).
    others: u64,
    comps: Vec<Component>,
    markers: BTreeMap<FuncId, Marker>,
    regions: Vec<Region>,
    /// Each member a table can hold, with its component's dispatch marker of its type
    /// (`devirt::Cx::groups`).
    groups: BTreeMap<FuncId, FuncId>,
    /// Each member, with its enter marker (`devirt::Cx::enters`).
    enters: BTreeMap<FuncId, FuncId>,
}

/// One recursive part of the call graph.
struct Component {
    /// Its functions, ascending.
    members: Vec<FuncId>,
    /// Its members' recursive calls ([`recursive_calls`]).
    calls: Vec<RecursiveCall>,
    /// Per function type, the words of its members a table can hold, with the member.
    words: BTreeMap<TypeId, Vec<(u32, FuncId)>>,
}

/// A call in a member's body that may call a member of its component: its caller, the members
/// it may call, and its arguments to them (the word of a dispatch left out).
struct RecursiveCall {
    caller: FuncId,
    callees: Vec<FuncId>,
    args: Vec<Value>,
}

/// What a parameter of a member holds over a region's activations ([`Recursion::fixed`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Held {
    /// Nothing known yet.
    Unknown,
    /// The same value in every activation: a constant, or a value defined outside the copies.
    Value(Value),
    /// Values that may differ.
    Varies,
}

impl Held {
    fn meet(self, other: Held) -> Held {
        match (self, other) {
            (Held::Unknown, x) | (x, Held::Unknown) => x,
            (Held::Value(a), Held::Value(b)) if a == b => Held::Value(a),
            _ => Held::Varies,
        }
    }
}

/// What a marker declaration stands for.
#[derive(Clone, Copy, Debug)]
enum Marker {
    /// A direct call to `member` of component `comp`.
    Enter { comp: usize, member: FuncId },
    /// A call through a table of function type `ty` to a member of component `comp`, the word
    /// first.
    Dispatch { comp: usize, ty: TypeId },
    /// The root call of a region.
    Root { region: usize },
    /// The head of `member`'s copy in a region.
    Head { region: usize, member: FuncId },
    /// Parameter `index` of `member`'s copy in a region, not taken from the root.
    Param {
        region: usize,
        member: FuncId,
        index: u32,
    },
    /// A return of `member`'s copy in a region, with its value when it has one.
    Ret { region: usize, member: FuncId },
}

/// One root's machine.
struct Region {
    comp: usize,
    /// What the root called: [`Marker::Enter`] or [`Marker::Dispatch`].
    root: Marker,
}

/// The call graph of the defined functions the entry reaches.
struct CallGraph {
    funcs: Vec<FuncId>,
    /// Per node, its callees.
    edges: Vec<Vec<u32>>,
    entry: u32,
}

impl Graph for CallGraph {
    type Node = u32;

    fn node_count(&self) -> usize {
        self.funcs.len()
    }

    fn entry(&self) -> u32 {
        self.entry
    }

    fn successors(&self, node: u32) -> impl Iterator<Item = u32> + '_ {
        self.edges.get(node as usize).into_iter().flatten().copied()
    }
}

/// The function type of defined or declared function `f`.
fn fn_ty(parts: &ModuleParts, f: FuncId) -> Result<TypeId, Error> {
    parts
        .funcs
        .get(f)
        .map(|f| f.ty)
        .ok_or_else(|| invariant("a function out of range"))
}

/// The return type and parameter types of function type `ty`.
fn signature(parts: &ModuleParts, ty: TypeId) -> Result<(TypeId, Vec<TypeId>), Error> {
    match parts.types.get(ty) {
        Some(Type::Function { ret, params, .. }) => Ok((*ret, params.to_vec())),
        _ => Err(invariant("a function's type is not a function type")),
    }
}

/// The defined function a call through constant `callee` at type `ty` calls (a stub's linked
/// definition), as devirtualization makes it direct.
fn constant_callee(
    parts: &ModuleParts,
    stubs: &Stubs,
    callee: Value,
    ty: TypeId,
) -> Option<FuncId> {
    let Value::Const(c) = callee else {
        return None;
    };
    let Some(Const::FuncAddr(f)) = parts.consts.get(c).map(|k| &k.value) else {
        return None;
    };
    let target = match stubs.get(*f) {
        Some(Stub::Resolved(def)) => *def,
        _ => *f,
    };
    let func = parts.funcs.get(target)?;
    (func.body.is_some() && func.ty == ty).then_some(target)
}

/// The node of defined function `h` in `g`, made when it has none (`node` holds each function's).
fn graph_node(
    g: &mut CallGraph,
    node: &mut [Option<u32>],
    h: FuncId,
    m: &mut Meter,
    held: &mut usize,
) -> Result<u32, Error> {
    if let Some(n) = node.get(h.index()).copied().flatten() {
        return Ok(n);
    }
    let n = u32::try_from(g.funcs.len()).map_err(|_| invariant("call graph"))?;
    if let Some(slot) = node.get_mut(h.index()) {
        *slot = Some(n);
    }
    m.charge(std::mem::size_of::<(FuncId, Vec<u32>)>())?;
    *held += std::mem::size_of::<(FuncId, Vec<u32>)>();
    g.funcs.push(h);
    g.edges.push(Vec::new());
    Ok(n)
}

/// Push to `found` each defined function whose address constant `c` holds, itself or among its
/// elements and operands (a stub's linked definition for a stub's), and each stub declaration no
/// linked function resolves whose address it holds (its address has a word too), reading each
/// constant once (`seen`). Each push is charged to `held`.
fn addressed(
    parts: &ModuleParts,
    stubs: &Stubs,
    c: ConstId,
    (seen, found): (&mut BTreeSet<ConstId>, &mut Vec<FuncId>),
    m: &mut Meter,
    held: &mut usize,
) -> Result<(), Error> {
    let mut work = vec![c];
    while let Some(c) = work.pop() {
        m.tick(1)?;
        if !seen.insert(c) {
            continue;
        }
        m.charge(std::mem::size_of::<ConstId>() * 3)?;
        *held += std::mem::size_of::<ConstId>() * 3;
        let Some(k) = parts.consts.get(c) else {
            continue;
        };
        if let Const::FuncAddr(f) = k.value {
            let def = match stubs.get(f) {
                Some(Stub::Resolved(def)) => *def,
                _ => f,
            };
            if parts.funcs.get(def).is_some_and(|d| d.body.is_some()) || stubs.get(def).is_some() {
                m.charge(std::mem::size_of::<FuncId>())?;
                *held += std::mem::size_of::<FuncId>();
                found.push(def);
            }
        }
        work.extend(k.value.children().iter().copied());
    }
    Ok(())
}

/// What [`prepare`] found: the recursion, if any, and the words of the functions whose address
/// is taken.
pub(super) struct Prepared {
    /// The recursive parts of the call graph and their markers: `None` when there is no
    /// recursion outside the entry's own cycle (which inlining refuses as before).
    pub recursion: Option<Recursion>,
    /// The word each function a call through a pointer may call is carried as.
    pub words: FunctionWords,
}

/// Find the recursive parts of the call graph `entry` reaches and make their markers (step 1 of
/// the module docs), and give each function whose address a reached body takes a word
/// ([`FunctionWords`], the stage's own words starting at `first_own`).
pub(super) fn prepare(
    parts: &mut ModuleParts,
    stubs: &Stubs,
    entry: FuncId,
    (candidates, first_own): (&[Candidate], u32),
    k: u32,
    m: &mut Meter,
) -> Result<Prepared, Error> {
    scoped(m, |m, held| {
        prepare_in(parts, stubs, entry, (candidates, first_own), k, m, held)
    })
}

fn prepare_in(
    parts: &mut ModuleParts,
    stubs: &Stubs,
    entry: FuncId,
    (candidates, first_own): (&[Candidate], u32),
    k: u32,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Prepared, Error> {
    // The functions a table can hold, defined, by type.
    let mut held_by: BTreeMap<TypeId, Vec<(u32, FuncId)>> = BTreeMap::new();
    for c in candidates {
        m.tick(1)?;
        if let Ok(g) = c.def {
            if parts.funcs.get(g).is_some_and(|f| f.body.is_some()) {
                m.charge(std::mem::size_of::<(u32, FuncId)>() * 2)?;
                *held += std::mem::size_of::<(u32, FuncId)>() * 2;
                held_by
                    .entry(fn_ty(parts, g)?)
                    .or_default()
                    .push((c.word, g));
            }
        }
    }
    // The call graph, by a worklist from the entry. A call through a pointer that is no constant
    // may call each function of its type whose address a reached body takes as a value
    // (`taken`), so such a function is a node once a call of its type is, and its edges are
    // added once every body is read.
    let mut node: Vec<Option<u32>> = filled(m, held, parts.funcs.len(), None)?;
    let mut g = CallGraph {
        funcs: vec![entry],
        edges: vec![Vec::new()],
        entry: 0,
    };
    if let Some(slot) = node.get_mut(entry.index()) {
        *slot = Some(0);
    }
    let mut taken: BTreeMap<TypeId, Vec<FuncId>> = BTreeMap::new();
    let mut taken_any: BTreeSet<FuncId> = BTreeSet::new();
    // The stubs no linked function resolves whose address a reached body takes: no node, but a
    // word.
    let mut taken_stubs: BTreeSet<FuncId> = BTreeSet::new();
    let mut indirect: BTreeSet<TypeId> = BTreeSet::new();
    let mut through: Vec<(u32, TypeId)> = Vec::new();
    let mut seen_consts: BTreeSet<ConstId> = BTreeSet::new();
    let mut next = 0usize;
    while let Some(&f) = g.funcs.get(next) {
        let at = u32::try_from(next).map_err(|_| invariant("call graph"))?;
        let mut out = Vec::new();
        // The functions whose address the body takes as a value (a call's callee is no value).
        let mut found = Vec::new();
        // The taken functions a call through a pointer may now reach, nodes with no edge yet.
        let mut targets = Vec::new();
        let body = parts
            .funcs
            .get(f)
            .and_then(|f| f.body.as_ref())
            .ok_or_else(|| invariant("a reached function without a body"))?;
        for inst in body.insts.values() {
            m.tick(1)?;
            let mut failed = None;
            let mut operand = |v: Value| {
                if let (Value::Const(c), None) = (v, &failed) {
                    let at = (&mut seen_consts, &mut found);
                    if let Err(e) = addressed(parts, stubs, c, at, m, held) {
                        failed = Some(e);
                    }
                }
            };
            match &inst.kind {
                InstKind::Call(call) => call.args.iter().copied().for_each(&mut operand),
                kind => kind.for_each_value(&mut operand),
            }
            if let Some(e) = failed {
                return Err(e);
            }
            let InstKind::Call(call) = &inst.kind else {
                continue;
            };
            match call.callee {
                Callee::Direct(h) if parts.funcs.get(h).is_some_and(|h| h.body.is_some()) => {
                    out.push(h);
                }
                Callee::Indirect { callee, fn_ty } => {
                    match constant_callee(parts, stubs, callee, fn_ty) {
                        Some(h) => out.push(h),
                        None => {
                            for &(_, h) in held_by.get(&fn_ty).into_iter().flatten() {
                                out.push(h);
                            }
                            if indirect.insert(fn_ty) {
                                targets.extend(taken.get(&fn_ty).into_iter().flatten().copied());
                            }
                            m.charge(std::mem::size_of::<(u32, TypeId)>() * 2)?;
                            *held += std::mem::size_of::<(u32, TypeId)>() * 2;
                            through.push((at, fn_ty));
                        }
                    }
                }
                _ => {}
            }
        }
        for h in found {
            if parts.funcs.get(h).is_none_or(|f| f.body.is_none()) {
                m.charge(std::mem::size_of::<FuncId>() * 2)?;
                *held += std::mem::size_of::<FuncId>() * 2;
                taken_stubs.insert(h);
                continue;
            }
            if !taken_any.insert(h) {
                continue;
            }
            let ty = fn_ty(parts, h)?;
            m.charge(std::mem::size_of::<FuncId>() * 3)?;
            *held += std::mem::size_of::<FuncId>() * 3;
            taken.entry(ty).or_default().push(h);
            if indirect.contains(&ty) {
                targets.push(h);
            }
        }
        let mut edges = Vec::with_capacity(out.len());
        m.charge(out.len() * std::mem::size_of::<u32>())?;
        *held += out.len() * std::mem::size_of::<u32>();
        for h in out {
            edges.push(graph_node(&mut g, &mut node, h, m, held)?);
        }
        if let Some(slot) = g.edges.get_mut(next) {
            *slot = edges;
        }
        for h in targets {
            graph_node(&mut g, &mut node, h, m, held)?;
        }
        next += 1;
    }
    // Each call through a pointer that is no constant may call each function of its type whose
    // address is taken.
    for (at, ty) in through {
        for &h in taken.get(&ty).into_iter().flatten() {
            m.tick(1)?;
            let n = node
                .get(h.index())
                .copied()
                .flatten()
                .ok_or_else(|| invariant("a taken function without a node"))?;
            m.charge(std::mem::size_of::<u32>())?;
            *held += std::mem::size_of::<u32>();
            if let Some(edges) = g.edges.get_mut(at as usize) {
                edges.push(n);
            }
        }
    }
    // The linked functions: the table candidates and the stubs' definitions.
    let linked: BTreeSet<FuncId> = candidates
        .iter()
        .filter_map(|c| c.def.as_ref().ok().copied())
        .chain(stubs.resolved())
        .collect();
    m.charge(linked.len() * std::mem::size_of::<FuncId>() * 2)?;
    *held += linked.len() * std::mem::size_of::<FuncId>() * 2;
    let addressed: BTreeSet<FuncId> = taken_any.union(&taken_stubs).copied().collect();
    m.charge(addressed.len() * std::mem::size_of::<FuncId>() * 2)?;
    *held += addressed.len() * std::mem::size_of::<FuncId>() * 2;
    let words = FunctionWords::new(parts, candidates, &addressed, first_own, m)?;
    let sccs = Sccs::compute(&g, m)?;
    let ledger = sccs.ledger_bytes();
    let reached = Reached {
        parts,
        stubs,
        linked: &linked,
        taken: &taken_any,
    };
    let found = components(&g, &sccs, (&words, &*parts), &reached, m, held);
    m.release(ledger);
    let (comps, folding) = found?;
    let others = other_linked_frames(parts, stubs, &linked, &comps, m, held)?;
    if comps.is_empty() {
        // Metal reserves the linked functions' frames whether or not they recurse.
        metal_stack(k, others)?;
        return Ok(Prepared {
            recursion: None,
            words,
        });
    }
    let mut r = Recursion {
        k,
        others,
        comps,
        markers: BTreeMap::new(),
        regions: Vec::new(),
        groups: BTreeMap::new(),
        enters: BTreeMap::new(),
    };
    // The markers: an enter marker per member, a dispatch marker per component and type a table
    // can hold.
    let mut enter = BTreeMap::new();
    for (c, comp) in r.comps.iter().enumerate() {
        for &member in &comp.members {
            let ty = fn_ty(parts, member)?;
            let marker = declare(parts, (b"recursion enter", ty), m)?;
            r.markers.insert(marker, Marker::Enter { comp: c, member });
            enter.insert(member, marker);
        }
        for (&ty, words) in &comp.words {
            let (ret, params) = signature(parts, ty)?;
            let i32t = parts.types.intern(Type::Int(32), m)?;
            let mut with_word = Vec::with_capacity(params.len() + 1);
            with_word.push(i32t);
            with_word.extend(params);
            let dty = parts.types.intern(
                Type::Function {
                    ret,
                    params: with_word.into_boxed_slice(),
                    vararg: false,
                },
                m,
            )?;
            let marker = declare(parts, (b"recursion dispatch", dty), m)?;
            r.markers.insert(marker, Marker::Dispatch { comp: c, ty });
            for &(_, member) in words {
                r.groups.insert(member, marker);
            }
        }
        m.charge(comp.members.len() * std::mem::size_of::<(FuncId, FuncId, Marker)>() * 3)?;
        *held += comp.members.len() * std::mem::size_of::<(FuncId, FuncId, Marker)>() * 3;
    }
    // Every direct call to a member, and every call through its constant address, calls its
    // enter marker.
    redirect(parts, stubs, &g.funcs, &enter, m)?;
    // A member of a component whose helpers fold takes them inlined: they are called only
    // directly in it, and in no cycle among themselves, so inlining ends at the enter markers.
    // A call through a member's constant address that inlining exposes (the helper's call
    // through its parameter) calls its enter marker in turn.
    for &f in &folding {
        let inlined = inline::inline(
            &parts.funcs,
            (&parts.types, &parts.symbols),
            &mut parts.consts,
            f,
            m,
        )?;
        if let Some(body) = inlined {
            let func = parts
                .funcs
                .get_mut(f)
                .ok_or_else(|| invariant("a member out of range"))?;
            if let Some(replaced) = func.body.replace(body) {
                replaced.release(m);
            }
        }
    }
    redirect(parts, stubs, &folding, &enter, m)?;
    r.enters = enter;
    for c in 0..r.comps.len() {
        let calls = recursive_calls(&r, parts, c, &taken, m, held)?;
        if let Some(comp) = r.comps.get_mut(c) {
            comp.calls = calls;
        }
    }
    Ok(Prepared {
        recursion: Some(r),
        words,
    })
}

/// What [`components`] reads of the module beyond the call graph.
struct Reached<'a> {
    parts: &'a ModuleParts,
    stubs: &'a Stubs,
    /// The linked functions: the table candidates and the stubs' definitions.
    linked: &'a BTreeSet<FuncId>,
    /// The functions whose address a reached body takes as a value.
    taken: &'a BTreeSet<FuncId>,
}

/// The cyclic components of `g` the entry is in none of, and the members whose helpers fold
/// into them ([`folds`]). A component's words are its members' that have one ([`FunctionWords`]:
/// a table can hold it, or its address is taken), by type.
fn components(
    g: &CallGraph,
    sccs: &Sccs<u32>,
    (words, parts): (&FunctionWords, &ModuleParts),
    reached: &Reached<'_>,
    m: &mut Meter,
    held: &mut usize,
) -> Result<(Vec<Component>, Vec<FuncId>), Error> {
    let mut comps = Vec::new();
    let mut folding = Vec::new();
    for c in 0..u32::try_from(sccs.len()).map_err(|_| invariant("components"))? {
        m.tick(1)?;
        if !sccs.is_cyclic(c) || sccs.members(c).any(|n| n == g.entry) {
            continue;
        }
        let nodes: BTreeSet<u32> = sccs.members(c).collect();
        m.charge(nodes.len() * std::mem::size_of::<(u32, FuncId)>() * 3)?;
        *held += nodes.len() * std::mem::size_of::<(u32, FuncId)>() * 3;
        let mut members = Vec::new();
        for &n in &nodes {
            members.push(*g.funcs.get(n as usize).ok_or_else(|| invariant("node"))?);
        }
        members.sort_unstable();
        if folds(g, &nodes, reached, m, held)? {
            members.retain(|f| reached.linked.contains(f));
            folding.extend(members.iter().copied());
        }
        comps.push(Component {
            members,
            calls: Vec::new(),
            words: BTreeMap::new(),
        });
    }
    for comp in &mut comps {
        for &f in &comp.members {
            m.tick(1)?;
            let Some(w) = words.word(f) else {
                continue;
            };
            m.charge(std::mem::size_of::<(TypeId, (u32, FuncId))>() * 2)?;
            *held += std::mem::size_of::<(TypeId, (u32, FuncId))>() * 2;
            comp.words.entry(fn_ty(parts, f)?).or_default().push((w, f));
        }
        // In word order: a table's candidates in the order the pipeline numbers them.
        for words in comp.words.values_mut() {
            words.sort_unstable();
        }
    }
    Ok((comps, folding))
}

/// Whether the helpers of the cyclic component of `nodes`, its functions no pipeline links, fold
/// into its linked functions, which are then its members: there are both, each helper is called in
/// the component only directly (its address is not taken as a value, and no call through a constant
/// address reaches it), and the helpers call each other in no cycle. Metal inlines a function that
/// is not visible into its caller (`apply` into the `prec` that passes its own address to it), so a
/// recursion through one has the linked functions' frames alone.
fn folds(
    g: &CallGraph,
    nodes: &BTreeSet<u32>,
    reached: &Reached<'_>,
    m: &mut Meter,
    held: &mut usize,
) -> Result<bool, Error> {
    let func = |n: u32| {
        g.funcs
            .get(n as usize)
            .copied()
            .ok_or_else(|| invariant("node"))
    };
    let mut helpers = BTreeSet::new();
    let mut helper_funcs = BTreeSet::new();
    let mut linked = false;
    for &n in nodes {
        let f = func(n)?;
        if reached.linked.contains(&f) {
            linked = true;
        } else if reached.taken.contains(&f) {
            return Ok(false);
        } else {
            m.charge(std::mem::size_of::<(u32, FuncId)>() * 3)?;
            *held += std::mem::size_of::<(u32, FuncId)>() * 3;
            helpers.insert(n);
            helper_funcs.insert(f);
        }
    }
    if !linked || helpers.is_empty() {
        return Ok(false);
    }
    let parts = reached.parts;
    for &n in nodes {
        let f = func(n)?;
        let body = parts
            .funcs
            .get(f)
            .and_then(|f| f.body.as_ref())
            .ok_or_else(|| invariant("a member without a body"))?;
        for inst in body.insts.values() {
            m.tick(1)?;
            let InstKind::Call(call) = &inst.kind else {
                continue;
            };
            let Callee::Indirect { callee, fn_ty } = call.callee else {
                continue;
            };
            let h = constant_callee(parts, reached.stubs, callee, fn_ty);
            if h.is_some_and(|h| helper_funcs.contains(&h)) {
                return Ok(false);
            }
        }
    }
    acyclic(g, &helpers, m, held)
}

/// Whether the calls among `nodes` of `g` form no cycle (a depth-first search over them).
fn acyclic(
    g: &CallGraph,
    nodes: &BTreeSet<u32>,
    m: &mut Meter,
    held: &mut usize,
) -> Result<bool, Error> {
    // Absent = unvisited, false = on the search's path, true = done.
    let mut done: BTreeMap<u32, bool> = BTreeMap::new();
    let mut path: Vec<(u32, usize)> = Vec::new();
    for &start in nodes {
        if done.contains_key(&start) {
            continue;
        }
        m.charge(std::mem::size_of::<(u32, bool, u32, usize)>() * 3)?;
        *held += std::mem::size_of::<(u32, bool, u32, usize)>() * 3;
        done.insert(start, false);
        path.push((start, 0));
        while let Some(&(n, i)) = path.last() {
            m.tick(1)?;
            let Some(&s) = g.edges.get(n as usize).and_then(|e| e.get(i)) else {
                done.insert(n, true);
                path.pop();
                continue;
            };
            if let Some(top) = path.last_mut() {
                top.1 += 1;
            }
            if !nodes.contains(&s) {
                continue;
            }
            match done.get(&s) {
                Some(false) => return Ok(false),
                Some(true) => {}
                None => {
                    m.charge(std::mem::size_of::<(u32, bool, u32, usize)>() * 3)?;
                    *held += std::mem::size_of::<(u32, bool, u32, usize)>() * 3;
                    done.insert(s, false);
                    path.push((s, 0));
                }
            }
        }
    }
    Ok(true)
}

/// Make every direct call to a member, and every call through its constant address, in the
/// bodies of `funcs` call the member's enter marker (`enter`).
fn redirect(
    parts: &mut ModuleParts,
    stubs: &Stubs,
    funcs: &[FuncId],
    enter: &BTreeMap<FuncId, FuncId>,
    m: &mut Meter,
) -> Result<(), Error> {
    for &f in funcs {
        let Some(body) = parts.funcs.get(f).and_then(|f| f.body.as_ref()) else {
            continue;
        };
        let mut redirect = Vec::new();
        for (i, inst) in body.insts.iter() {
            m.tick(1)?;
            let InstKind::Call(call) = &inst.kind else {
                continue;
            };
            let target = match call.callee {
                Callee::Direct(h) => Some(h),
                Callee::Indirect { callee, fn_ty } => constant_callee(parts, stubs, callee, fn_ty),
                _ => None,
            };
            if let Some(&marker) = target.and_then(|h| enter.get(&h)) {
                redirect.push((i, marker));
            }
        }
        let body = parts
            .funcs
            .get_mut(f)
            .and_then(|f| f.body.as_mut())
            .ok_or_else(|| invariant("a reached function without a body"))?;
        for (i, marker) in redirect {
            if let Some(InstKind::Call(call)) = body.insts.get_mut(i).map(|i| &mut i.kind) {
                call.callee = Callee::Direct(marker);
            }
        }
    }
    Ok(())
}

/// The linked frames of the `linked` functions (the table candidates and the stubs'
/// definitions) in none of `comps`: 16 bytes for one that calls a visible function or through a
/// pointer, in its body or the bodies of the functions it calls directly (which Metal inlines),
/// and 0 for a leaf.
fn other_linked_frames(
    parts: &ModuleParts,
    stubs: &Stubs,
    linked: &BTreeSet<FuncId>,
    comps: &[Component],
    m: &mut Meter,
    held: &mut usize,
) -> Result<u64, Error> {
    let mut bytes = 0u64;
    for &f in linked {
        if comps.iter().any(|c| c.members.binary_search(&f).is_ok()) {
            continue;
        }
        if calls_out(parts, stubs, linked, f, m, held)? {
            bytes = bytes.saturating_add(16);
        }
    }
    Ok(bytes)
}

/// Whether defined function `f`, or a function it calls directly that is not `linked` (which
/// Metal inlines), calls a visible function (a stub, or a `linked` function) or through a
/// pointer.
fn calls_out(
    parts: &ModuleParts,
    stubs: &Stubs,
    linked: &BTreeSet<FuncId>,
    f: FuncId,
    m: &mut Meter,
    held: &mut usize,
) -> Result<bool, Error> {
    let mut seen = BTreeSet::from([f]);
    let mut work = vec![f];
    while let Some(g) = work.pop() {
        let Some(body) = parts.funcs.get(g).and_then(|g| g.body.as_ref()) else {
            continue;
        };
        for inst in body.insts.values() {
            m.tick(1)?;
            let InstKind::Call(call) = &inst.kind else {
                continue;
            };
            match call.callee {
                Callee::Indirect { .. } => return Ok(true),
                Callee::Direct(h) if stubs.get(h).is_some() || linked.contains(&h) => {
                    return Ok(true)
                }
                Callee::Direct(h)
                    if parts.funcs.get(h).is_some_and(|h| h.body.is_some()) && seen.insert(h) =>
                {
                    m.charge(std::mem::size_of::<FuncId>() * 2)?;
                    *held += std::mem::size_of::<FuncId>() * 2;
                    work.push(h);
                }
                _ => {}
            }
        }
    }
    Ok(false)
}

/// `v` with no-op `bitcast`s (to their operand's own type) looked through, in `f`'s body.
fn through_noop(parts: &ModuleParts, f: FuncId, body: &Body, v: Value) -> Value {
    let mut v = v;
    for _ in 0..body.insts.len() {
        let Value::Inst(i) = v else { return v };
        let Some(inst) = body.insts.get(i) else {
            return v;
        };
        let InstKind::Cast {
            op: CastOp::Bitcast,
            v: from,
        } = inst.kind
        else {
            return v;
        };
        let from_ty = match from {
            Value::Arg(a) => parts
                .funcs
                .get(f)
                .and_then(|f| match parts.types.get(f.ty) {
                    Some(Type::Function { params, .. }) => params.get(a.index()).copied(),
                    _ => None,
                }),
            Value::Inst(j) => body.insts.get(j).map(|j| j.ty),
            Value::Const(c) => parts.consts.ty(c),
        };
        if from_ty != Some(inst.ty) {
            return v;
        }
        v = from;
    }
    v
}

/// The recursive calls of member bodies of component `c`: each call to a member's enter marker,
/// and each call through a pointer of a type some of its members have, which may call each of
/// them a table can hold or whose address is `taken`.
fn recursive_calls(
    r: &Recursion,
    parts: &ModuleParts,
    c: usize,
    taken: &BTreeMap<TypeId, Vec<FuncId>>,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Vec<RecursiveCall>, Error> {
    let comp = r.comps.get(c).ok_or_else(|| invariant("component"))?;
    let mut out = Vec::new();
    for &g in &comp.members {
        let body = parts
            .funcs
            .get(g)
            .and_then(|f| f.body.as_ref())
            .ok_or_else(|| invariant("a member without a body"))?;
        for inst in body.insts.values() {
            m.tick(1)?;
            let InstKind::Call(call) = &inst.kind else {
                continue;
            };
            let callees: Vec<FuncId> = match call.callee {
                Callee::Direct(f) => match r.markers.get(&f) {
                    Some(&Marker::Enter { comp, member }) if comp == c => vec![member],
                    _ => continue,
                },
                Callee::Indirect { fn_ty, .. } => {
                    let words = comp.words.get(&fn_ty).into_iter().flatten().map(|w| w.1);
                    let pointed = taken
                        .get(&fn_ty)
                        .into_iter()
                        .flatten()
                        .copied()
                        .filter(|f| comp.members.binary_search(f).is_ok());
                    let set: BTreeSet<FuncId> = words.chain(pointed).collect();
                    if set.is_empty() {
                        continue;
                    }
                    set.into_iter().collect()
                }
                _ => continue,
            };
            let args: Vec<Value> = call
                .args
                .iter()
                .map(|&a| through_noop(parts, g, body, a))
                .collect();
            let bytes = std::mem::size_of::<Value>() * args.len()
                + std::mem::size_of::<FuncId>() * callees.len()
                + std::mem::size_of::<RecursiveCall>();
            m.charge(bytes)?;
            *held += bytes;
            out.push(RecursiveCall {
                caller: g,
                callees,
                args,
            });
        }
    }
    Ok(out)
}

impl Recursion {
    /// The dispatch markers devirtualization calls for the members a table can hold.
    pub fn groups(&self) -> &BTreeMap<FuncId, FuncId> {
        &self.groups
    }

    /// The enter marker devirtualization calls for a call through a member's constant address.
    pub fn enters(&self) -> &BTreeMap<FuncId, FuncId> {
        &self.enters
    }

    /// The entry's call stack once a region has one: the `maxCallStackDepth` the stacks are
    /// sized for, at least 1.
    pub fn call_stack(&self) -> Option<CallStack> {
        (!self.regions.is_empty()).then_some(CallStack {
            depth: self.k.max(1),
        })
    }

    /// The marker `kind` calls directly, if it calls one.
    fn marker(&self, kind: &InstKind) -> Option<(FuncId, Marker)> {
        let InstKind::Call(call) = kind else {
            return None;
        };
        let Callee::Direct(f) = call.callee else {
            return None;
        };
        self.markers.get(&f).map(|&k| (f, k))
    }

    /// Per block of `body`, the region and member whose copy holds it: the blocks each head
    /// reaches. Copies are unreached and reach no other copy until assembly.
    fn owners(
        &self,
        body: &Body,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<Vec<Option<(usize, FuncId)>>, Error> {
        let mut owner = filled(m, held, body.blocks.len(), None)?;
        let mut stack = Vec::new();
        for (b, block) in body.blocks.iter() {
            m.tick(1)?;
            let Some(first) = body.insts.get(block.start) else {
                continue;
            };
            let Some((_, Marker::Head { region, member })) = self.marker(&first.kind) else {
                continue;
            };
            stack.push(b);
            while let Some(x) = stack.pop() {
                match owner.get(x.index()).copied().flatten() {
                    Some(o) if o == (region, member) => continue,
                    Some(_) => return Err(invariant("two copies reach one block")),
                    None => {}
                }
                if let Some(slot) = owner.get_mut(x.index()) {
                    *slot = Some((region, member));
                }
                for s in body.successors(x) {
                    m.tick(1)?;
                    stack.push(s);
                }
            }
        }
        Ok(owner)
    }

    /// Give each root of `body` a region and append its copies (step 2 of the module docs):
    /// `None` when no root is new.
    pub fn islands(
        &mut self,
        body: &Body,
        parts: &mut ModuleParts,
        (entry_params, emap): (&[TypeId], &[ConstId]),
        m: &mut Meter,
    ) -> Result<Option<Body>, Error> {
        scoped(m, |m, held| {
            self.islands_in(body, parts, (entry_params, emap), m, held)
        })
    }

    fn islands_in(
        &mut self,
        body: &Body,
        parts: &mut ModuleParts,
        (entry_params, emap): (&[TypeId], &[ConstId]),
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<Option<Body>, Error> {
        let owner = self.owners(body, m, held)?;
        let mut roots = Vec::new();
        for (b, block) in body.blocks.iter() {
            for i in block.inst_ids() {
                m.tick(1)?;
                let Some(inst) = body.insts.get(i) else {
                    continue;
                };
                let comp = match self.marker(&inst.kind) {
                    Some((_, Marker::Enter { comp, .. } | Marker::Dispatch { comp, .. })) => comp,
                    _ => continue,
                };
                let inside = owner
                    .get(b.index())
                    .copied()
                    .flatten()
                    .and_then(|(region, _)| self.regions.get(region))
                    .is_some_and(|r| r.comp == comp);
                if !inside {
                    roots.push(i);
                }
            }
        }
        if roots.is_empty() {
            return Ok(None);
        }
        // The body, then each root's copies.
        let mut out = Body::default();
        for inst in body.insts.values() {
            out.push_inst(inst.clone(), m)?;
        }
        for block in body.blocks.values() {
            out.blocks.push(*block, m)?;
        }
        for i in roots {
            let (marker, kind) = self
                .marker(&body.insts.get(i).ok_or_else(|| invariant("root"))?.kind)
                .ok_or_else(|| invariant("a root that is no marker call"))?;
            let comp = match kind {
                Marker::Enter { comp, .. } | Marker::Dispatch { comp, .. } => comp,
                _ => return Err(invariant("a root that is no recursive call")),
            };
            let region = self.regions.len();
            self.regions.push(Region { comp, root: kind });
            let ty = fn_ty(parts, marker)?;
            let root = declare(parts, (b"recursion root", ty), m)?;
            self.markers.insert(root, Marker::Root { region });
            let args: Vec<Value> = match out.insts.get(i).map(|i| &i.kind) {
                Some(InstKind::Call(call)) => match kind {
                    Marker::Dispatch { .. } => call.args.iter().skip(1).copied().collect(),
                    _ => call.args.to_vec(),
                },
                _ => return Err(invariant("a root that is no call")),
            };
            if let Some(InstKind::Call(call)) = out.insts.get_mut(i).map(|i| &mut i.kind) {
                call.callee = Callee::Direct(root);
            }
            // A root argument defined outside every copy holds in every activation.
            let available = |v: Value| match v {
                Value::Inst(x) => {
                    let b = block_of(body, x);
                    b.is_some_and(|b| owner.get(b.index()).copied().flatten().is_none())
                }
                Value::Arg(_) | Value::Const(_) => true,
            };
            let c = self.comps.get(comp).ok_or_else(|| invariant("component"))?;
            let members = c.members.clone();
            let entered = match kind {
                Marker::Dispatch { ty, .. } => c
                    .words
                    .get(&ty)
                    .into_iter()
                    .flatten()
                    .map(|w| w.1)
                    .collect(),
                Marker::Enter { member, .. } => vec![member],
                _ => return Err(invariant("a root that is no recursive call")),
            };
            let root = (
                entered.as_slice(),
                args.as_slice(),
                &available as &dyn Fn(Value) -> bool,
            );
            let fixed = self.fixed(parts, comp, root, emap, m, held)?;
            for member in members {
                self.copy(&mut out, parts, (region, member), (&fixed, entry_params), m)?;
            }
        }
        Ok(Some(out))
    }

    /// The parameters of component `comp`'s members that hold one value in every activation of
    /// a region whose root enters the members `entered` with `args` (`available` says which of
    /// them every activation can read), with that value: one of the root's, or a constant every
    /// call that reaches the parameter passes. It is the greatest fixpoint over the component's
    /// recursive calls: a call passes a constant (as the entry's erased body has it, through
    /// `emap`), its caller's own parameter, which holds what that one does, or anything else, which
    /// varies.
    fn fixed(
        &self,
        parts: &ModuleParts,
        comp: usize,
        (entered, args, available): (&[FuncId], &[Value], &dyn Fn(Value) -> bool),
        emap: &[ConstId],
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<BTreeMap<(FuncId, u32), Value>, Error> {
        let c = self.comps.get(comp).ok_or_else(|| invariant("component"))?;
        let arity = |f: FuncId| -> Result<u32, Error> {
            let (_, params) = signature(parts, fn_ty(parts, f)?)?;
            u32::try_from(params.len()).map_err(|_| invariant("params"))
        };
        let mut state: BTreeMap<(FuncId, u32), Held> = BTreeMap::new();
        for &f in &c.members {
            for i in 0..arity(f)? {
                state.insert((f, i), Held::Unknown);
            }
        }
        let bytes = state.len() * std::mem::size_of::<((FuncId, u32), Held)>() * 2;
        m.charge(bytes)?;
        *held += bytes;
        for &f in entered {
            for i in 0..arity(f)? {
                m.tick(1)?;
                let given = match args.get(i as usize) {
                    Some(&a) if available(a) => Held::Value(a),
                    _ => Held::Varies,
                };
                if let Some(h) = state.get_mut(&(f, i)) {
                    *h = h.meet(given);
                }
            }
        }
        loop {
            let mut changed = false;
            for call in &c.calls {
                for &f in &call.callees {
                    for i in 0..arity(f)? {
                        m.tick(1)?;
                        let passed = match call.args.get(i as usize) {
                            Some(&Value::Const(k)) => {
                                Held::Value(Value::Const(emap.get(k.index()).copied().unwrap_or(k)))
                            }
                            Some(&Value::Arg(a)) => state
                                .get(&(call.caller, a.raw()))
                                .copied()
                                .unwrap_or(Held::Varies),
                            _ => Held::Varies,
                        };
                        if let Some(h) = state.get_mut(&(f, i)) {
                            let met = h.meet(passed);
                            if met != *h {
                                *h = met;
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        Ok(state
            .into_iter()
            .filter_map(|(at, h)| match h {
                Held::Value(v) => Some((at, v)),
                Held::Unknown | Held::Varies => None,
            })
            .collect())
    }

    /// Append to `out` the copy of `member` in `region`, which takes each parameter `fixed`
    /// gives ([`Recursion::fixed`]) as that value.
    fn copy(
        &mut self,
        out: &mut Body,
        parts: &mut ModuleParts,
        (region, member): (usize, FuncId),
        (fixed, entry_params): (&BTreeMap<(FuncId, u32), Value>, &[TypeId]),
        m: &mut Meter,
    ) -> Result<(), Error> {
        let ty = fn_ty(parts, member)?;
        let (ret, params) = signature(parts, ty)?;
        let void = parts.types.intern(Type::Void, m)?;
        let unit = parts.types.intern(
            Type::Function {
                ret: void,
                params: Box::new([]),
                vararg: false,
            },
            m,
        )?;
        let head = declare(parts, (b"recursion head", unit), m)?;
        self.markers.insert(head, Marker::Head { region, member });
        let ret_ty = parts.types.intern(
            Type::Function {
                ret: void,
                params: if ret == void {
                    Box::new([])
                } else {
                    Box::new([ret])
                },
                vararg: false,
            },
            m,
        )?;
        let ret_marker = declare(parts, (b"recursion return", ret_ty), m)?;
        self.markers
            .insert(ret_marker, Marker::Ret { region, member });
        // The value of each parameter that holds one in every activation.
        let mut given = Vec::with_capacity(params.len());
        for (j, &pty) in params.iter().enumerate() {
            let index = u32::try_from(j).map_err(|_| invariant("params"))?;
            given.push(
                fixed
                    .get(&(member, index))
                    .copied()
                    .filter(|&a| value_ty(out, parts, entry_params, a) == Some(pty)),
            );
        }
        // The head block: the head marker, each parameter the root does not give, then a branch
        // to the copy's entry.
        let first_inst = u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?;
        let first_block = u32::try_from(out.blocks.len()).map_err(|_| invariant("ids"))?;
        let call = call_of;
        out.push_inst(
            Inst {
                ty: void,
                kind: call(head, Box::new([])),
            },
            m,
        )?;
        let mut param_values = Vec::with_capacity(params.len());
        for (j, (&pty, given)) in params.iter().zip(given).enumerate() {
            let index = u32::try_from(j).map_err(|_| invariant("params"))?;
            match given {
                Some(a) => param_values.push(a),
                None => {
                    let pty_fn = parts.types.intern(
                        Type::Function {
                            ret: pty,
                            params: Box::new([]),
                            vararg: false,
                        },
                        m,
                    )?;
                    let marker = declare(parts, (b"recursion parameter", pty_fn), m)?;
                    self.markers.insert(
                        marker,
                        Marker::Param {
                            region,
                            member,
                            index,
                        },
                    );
                    let v = out.push_inst(
                        Inst {
                            ty: pty,
                            kind: call(marker, Box::new([])),
                        },
                        m,
                    )?;
                    param_values.push(Value::Inst(v));
                }
            }
        }
        let src = parts
            .funcs
            .get(member)
            .and_then(|f| f.body.as_ref())
            .ok_or_else(|| invariant("a member without a body"))?;
        let block0 = first_block + 1;
        out.push_inst(
            Inst {
                ty: void,
                kind: InstKind::Br {
                    dest: BlockId::new(block0),
                    hints: None,
                },
            },
            m,
        )?;
        let head_end = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
        out.blocks.push(
            Block {
                start: InstId::new(first_inst),
                end: head_end,
                name: None,
            },
            m,
        )?;
        // The member's blocks. Each `ret` is two instructions, so ids are laid out first.
        let mut new_id = Vec::with_capacity(src.insts.len());
        let mut next = u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?;
        for inst in src.insts.values() {
            m.tick(1)?;
            new_id.push(next);
            next += if matches!(inst.kind, InstKind::Ret(_)) {
                2
            } else {
                1
            };
        }
        m.charge(new_id.len() * std::mem::size_of::<u32>())?;
        let map = |v: Value| -> Result<Value, Error> {
            Ok(match v {
                Value::Arg(a) => *param_values
                    .get(a.index())
                    .ok_or_else(|| invariant("a parameter out of range"))?,
                Value::Inst(i) => Value::Inst(InstId::new(
                    *new_id
                        .get(i.index())
                        .ok_or_else(|| invariant("an instruction out of range"))?,
                )),
                Value::Const(_) => v,
            })
        };
        let shift = |b: BlockId| BlockId::new(b.raw() + block0);
        for block in src.blocks.values() {
            let start = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
            for i in block.inst_ids() {
                let inst = src.insts.get(i).ok_or_else(|| invariant("instruction"))?;
                let mut kind = inst.kind.clone();
                let mut failed = None;
                kind.for_each_value_mut(|v| match map(*v) {
                    Ok(x) => *v = x,
                    Err(e) => failed = Some(e),
                });
                if let Some(e) = failed {
                    return Err(e);
                }
                match &mut kind {
                    InstKind::Phi { incoming, .. } => {
                        for (_, b) in incoming.iter_mut() {
                            *b = shift(*b);
                        }
                    }
                    InstKind::Br { dest, .. } => *dest = shift(*dest),
                    InstKind::CondBr { t, f, .. } => {
                        *t = shift(*t);
                        *f = shift(*f);
                    }
                    InstKind::Switch { default, cases, .. } => {
                        *default = shift(*default);
                        for (_, b) in cases.iter_mut() {
                            *b = shift(*b);
                        }
                    }
                    _ => {}
                }
                if let InstKind::Ret(v) = kind {
                    let args: Box<[Value]> = v.into_iter().collect();
                    out.push_inst(
                        Inst {
                            ty: void,
                            kind: call(ret_marker, args),
                        },
                        m,
                    )?;
                    out.push_inst(
                        Inst {
                            ty: void,
                            kind: InstKind::Unreachable,
                        },
                        m,
                    )?;
                } else {
                    out.push_inst(Inst { ty: inst.ty, kind }, m)?;
                }
            }
            let end = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
            out.blocks.push(
                Block {
                    start,
                    end,
                    name: block.name,
                },
                m,
            )?;
        }
        m.release(new_id.len() * std::mem::size_of::<u32>());
        Ok(())
    }
}

/// The type of value `v` of `body`, whose arguments are of types `params`.
fn value_ty(body: &Body, parts: &ModuleParts, params: &[TypeId], v: Value) -> Option<TypeId> {
    match v {
        Value::Inst(i) => body.insts.get(i).map(|i| i.ty),
        Value::Const(c) => parts.consts.ty(c),
        Value::Arg(a) => params.get(a.index()).copied(),
    }
}

/// The block of `body` holding instruction `i` (blocks tile the arena in order).
fn block_of(body: &Body, i: InstId) -> Option<BlockId> {
    let mut lo = 0usize;
    let mut hi = body.blocks.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let b = BlockId::from_index(mid)?;
        let block = body.blocks.get(b)?;
        if i < block.start {
            hi = mid;
        } else if i >= block.end {
            lo = mid + 1;
        } else {
            return Some(b);
        }
    }
    None
}

/// What assembly keeps of one region: its frame and the places its copies' markers name.
struct Machine {
    /// The activations its stack holds ([`frames`]), at least 1.
    frames: u64,
    /// The frame's fields after the call site's number: the copies' allocas and stored values.
    fields: Vec<TypeId>,
    /// `[frames x {i32, fields...}]`.
    stack_ty: TypeId,
    /// Each member's head block.
    heads: BTreeMap<FuncId, BlockId>,
    /// Each parameter marker call, by member and position.
    params: BTreeMap<(FuncId, u32), InstId>,
    /// The recursive call sites, numbered by position.
    calls: Vec<InstId>,
}

/// How a copy's value is reached at a use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reach {
    /// Stored to frame field `field` after its definition, loaded at each use.
    Slot(u32),
    /// An `alloca` of the copy: frame field `field`'s address.
    Alloca(u32),
    /// An address computed again at each use from an `alloca` of the copy or a value defined
    /// outside it.
    Remat,
}

/// A body under construction: the old body's instructions under their own ids, then new ones,
/// and each block's instruction ids in order.
struct Draft {
    insts: Vec<Inst>,
    blocks: Vec<Vec<u32>>,
    names: Vec<Option<Sym>>,
}

impl Draft {
    fn block(&mut self, m: &mut Meter, held: &mut usize) -> Result<BlockId, Error> {
        let b = BlockId::from_index(self.blocks.len()).ok_or_else(|| invariant("block ids"))?;
        let bytes = std::mem::size_of::<(Vec<u32>, Option<Sym>)>();
        m.charge(bytes)?;
        *held += bytes;
        self.blocks.push(Vec::new());
        self.names.push(None);
        Ok(b)
    }

    fn place(&mut self, b: BlockId, id: u32, m: &mut Meter, held: &mut usize) -> Result<(), Error> {
        m.charge(std::mem::size_of::<u32>())?;
        *held += std::mem::size_of::<u32>();
        self.blocks
            .get_mut(b.index())
            .ok_or_else(|| invariant("draft block"))?
            .push(id);
        Ok(())
    }

    fn push(
        &mut self,
        b: BlockId,
        (ty, kind): (TypeId, InstKind),
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<Value, Error> {
        let id = u32::try_from(self.insts.len()).map_err(|_| invariant("instruction ids"))?;
        let bytes = std::mem::size_of::<Inst>() + kind.heap_bytes();
        m.charge(bytes)?;
        *held += bytes;
        self.insts.push(Inst { ty, kind });
        self.place(b, id, m, held)?;
        Ok(Value::Inst(InstId::new(id)))
    }
}

/// The scalar types and constants assembly emits with.
struct Words {
    void: TypeId,
    i1: TypeId,
    i32: TypeId,
    ptr: TypeId,
    zero: Value,
    one: Value,
}

impl Words {
    fn new(parts: &mut ModuleParts, m: &mut Meter) -> Result<Words, Error> {
        let void = parts.types.intern(Type::Void, m)?;
        let i1 = parts.types.intern(Type::Int(1), m)?;
        let i32 = parts.types.intern(Type::Int(32), m)?;
        let ptr = parts.types.intern(
            Type::Ptr {
                space: AddrSpace::Thread,
            },
            m,
        )?;
        let zero = int(parts, i32, 0, m)?;
        let one = int(parts, i32, 1, m)?;
        Ok(Words {
            void,
            i1,
            i32,
            ptr,
            zero,
            one,
        })
    }
}

/// The integer constant `v` of type `ty`.
fn int(parts: &mut ModuleParts, ty: TypeId, v: u64, m: &mut Meter) -> Result<Value, Error> {
    let c = parts.consts.intern(
        Constant {
            ty,
            value: Const::Int(v),
        },
        m,
    )?;
    Ok(Value::Const(c))
}

/// The zero of type `ty` (the result of a call past the stack's capacity).
fn zero_of(parts: &mut ModuleParts, ty: TypeId, m: &mut Meter) -> Result<Value, Error> {
    let c = parts.consts.intern(
        Constant {
            ty,
            value: Const::Null,
        },
        m,
    )?;
    Ok(Value::Const(c))
}

/// The bytes of a frame of a call site's number (an `i32`) and the slots `slots`, packed as
/// Metal's frame is: the largest alignment first, so that only the end is padded, to the largest
/// alignment. A `[3 x [2 x <4 x i32>]]` array kept beside a word is 112 bytes, as Metal reserves
/// it, where a struct of the slots in their order is 128.
fn frame_bytes(parts: &ModuleParts, slots: &[TypeId], m: &mut Meter) -> Result<u64, Error> {
    let before = m.ledger();
    let layouts = TypeLayouts::new(&parts.data_layout, &parts.types, m);
    let bytes = layouts.and_then(|l| {
        let mut fields = Vec::with_capacity(slots.len() + 1);
        fields.push((4u64, 4u64));
        for &t in slots {
            let a = l
                .abi_align(t)
                .ok_or_else(|| invariant("a frame slot without a size"))?;
            let size = l
                .alloc_size(t)
                .ok_or_else(|| invariant("a frame slot without a size"))?;
            fields.push((a.max(1), size));
        }
        fields.sort_by_key(|f| std::cmp::Reverse(f.0));
        let (mut end, mut align) = (0u64, 1u64);
        for (a, size) in fields {
            end = end.div_ceil(a).saturating_mul(a).saturating_add(size);
            align = align.max(a);
        }
        Ok(end.div_ceil(align).saturating_mul(align))
    });
    let spent = usize::try_from(m.ledger().saturating_sub(before)).unwrap_or(usize::MAX);
    m.release(spent);
    bytes
}

/// A `call` of `callee` with `args`.
fn call_of(callee: FuncId, args: Box<[Value]>) -> InstKind {
    InstKind::Call(Box::new(crate::ir::Call {
        callee: Callee::Direct(callee),
        args,
        fmf: FastMath::NONE,
        tail: Default::default(),
        attrs: None,
        range: None,
    }))
}

/// A dense bit set.
#[derive(Clone)]
struct Bits(Vec<u64>);

impl Bits {
    fn new(n: usize) -> Bits {
        Bits(vec![0; n.div_ceil(64)])
    }

    fn set(&mut self, i: u32) {
        if let Some(w) = self.0.get_mut((i / 64) as usize) {
            *w |= 1 << (i % 64);
        }
    }

    fn ones(&self) -> impl Iterator<Item = u32> + '_ {
        self.0.iter().enumerate().flat_map(|(k, &w)| {
            (0..64u32)
                .filter(move |b| w & (1 << b) != 0)
                .map(move |b| k as u32 * 64 + b)
        })
    }
}

impl Recursion {
    /// Make each region its machine (step 3 of the module docs). `body` is returned when there
    /// is no region.
    pub fn assemble(
        &self,
        body: Body,
        parts: &mut ModuleParts,
        m: &mut Meter,
    ) -> Result<Body, Error> {
        if self.regions.is_empty() {
            return Ok(body);
        }
        let out = scoped(m, |m, held| self.assemble_in(&body, parts, m, held));
        body.release(m);
        out
    }

    fn assemble_in(
        &self,
        body: &Body,
        parts: &mut ModuleParts,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<Body, Error> {
        let owner = self.owners(body, m, held)?;
        let mut inst_block: Vec<BlockId> = filled(m, held, body.insts.len(), BlockId::new(0))?;
        for (b, block) in body.blocks.iter() {
            for i in block.inst_ids() {
                if let Some(slot) = inst_block.get_mut(i.index()) {
                    *slot = b;
                }
            }
        }
        let island = |i: InstId| {
            inst_block
                .get(i.index())
                .and_then(|b| owner.get(b.index()).copied().flatten())
        };
        // The places each region's markers name.
        let mut machines: Vec<Machine> = Vec::with_capacity(self.regions.len());
        for _ in &self.regions {
            machines.push(Machine {
                frames: 0,
                fields: Vec::new(),
                stack_ty: TypeId::new(0),
                heads: BTreeMap::new(),
                params: BTreeMap::new(),
                calls: Vec::new(),
            });
        }
        for (b, block) in body.blocks.iter() {
            for i in block.inst_ids() {
                m.tick(1)?;
                let Some(inst) = body.insts.get(i) else {
                    continue;
                };
                match self.marker(&inst.kind).map(|(_, k)| k) {
                    Some(Marker::Head { region, member }) => {
                        if let Some(mc) = machines.get_mut(region) {
                            mc.heads.insert(member, b);
                        }
                    }
                    Some(Marker::Param {
                        region,
                        member,
                        index,
                    }) => {
                        if let Some(mc) = machines.get_mut(region) {
                            mc.params.insert((member, index), i);
                        }
                    }
                    Some(Marker::Enter { comp, .. } | Marker::Dispatch { comp, .. }) => {
                        let (region, _) = island(i).ok_or_else(|| invariant("a root left"))?;
                        let mc = machines
                            .get_mut(region)
                            .ok_or_else(|| invariant("region"))?;
                        if self.regions.get(region).map(|r| r.comp) != Some(comp) {
                            return Err(invariant("a root left"));
                        }
                        mc.calls.push(i);
                    }
                    _ => {}
                }
            }
        }
        // How each copy's values are reached, and each region's frame.
        let mut reach: BTreeMap<InstId, Reach> = BTreeMap::new();
        let mut copies: BTreeMap<(usize, FuncId), Vec<BlockId>> = BTreeMap::new();
        for (b, o) in owner.iter().enumerate() {
            if let Some(o) = o {
                let b = BlockId::from_index(b).ok_or_else(|| invariant("block"))?;
                copies.entry(*o).or_default().push(b);
            }
        }
        // Each member's own frame, Metal's for it: its call site's number and its copy's slots,
        // reserved (the largest over its copies).
        let mut own: BTreeMap<FuncId, u64> = BTreeMap::new();
        for (&(region, member), blocks) in &copies {
            let mc = machines
                .get_mut(region)
                .ok_or_else(|| invariant("region"))?;
            let start = mc.fields.len();
            self.analyze(body, parts, blocks, mc, &mut reach, &island, m, held)?;
            let slots = mc.fields.get(start..).ok_or_else(|| invariant("fields"))?;
            let bytes = reserved(frame_bytes(parts, slots, m)?);
            let r = own.entry(member).or_default();
            *r = (*r).max(bytes);
        }
        let linked = own.values().fold(self.others, |s, &b| s.saturating_add(b));
        metal_stack(self.k, linked)?;
        let w = Words::new(parts, m)?;
        for (mc, region) in machines.iter_mut().zip(&self.regions) {
            let mut fields = Vec::with_capacity(mc.fields.len() + 1);
            fields.push(w.i32);
            fields.extend(mc.fields.iter().copied());
            let frame = parts.types.intern(
                Type::Struct {
                    fields: fields.into_boxed_slice(),
                    packed: false,
                    name: None,
                },
                m,
            )?;
            let smallest = self
                .comps
                .get(region.comp)
                .ok_or_else(|| invariant("component"))?
                .members
                .iter()
                .filter_map(|f| own.get(f).copied())
                .min()
                .ok_or_else(|| invariant("a component without a copy"))?;
            mc.frames = frames(self.k, linked, smallest);
            mc.stack_ty = parts.types.intern(
                Type::Array {
                    elem: frame,
                    len: mc.frames,
                },
                m,
            )?;
        }
        self.emit(body, parts, &machines, &reach, &owner, &w, m, held)
    }
}

impl Recursion {
    /// The liveness of one copy, the blocks `blocks` of `mc`'s region: its allocas, the
    /// addresses computed again at each use, and the values a recursive call separates from a
    /// use, which `reach` gets as frame fields of `mc`.
    fn analyze(
        &self,
        body: &Body,
        parts: &mut ModuleParts,
        blocks: &[BlockId],
        mc: &mut Machine,
        reach: &mut BTreeMap<InstId, Reach>,
        island: &dyn Fn(InstId) -> Option<(usize, FuncId)>,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<(), Error> {
        let first = blocks.first().ok_or_else(|| invariant("an empty copy"))?;
        let copy = body
            .blocks
            .get(*first)
            .and_then(|b| island(b.start))
            .ok_or_else(|| invariant("a copy's block"))?;
        let mine = |v: Value| match v {
            Value::Inst(x) if island(x) == Some(copy) => Some(x),
            _ => None,
        };
        let calls: BTreeSet<InstId> = mc.calls.iter().copied().collect();
        // Allocas and addresses: a chain of `getelementptr`, `bitcast`, and `addrspacecast`
        // pointers down to an alloca of the copy or a value outside it.
        let mut remat: BTreeMap<InstId, bool> = BTreeMap::new();
        for &b in blocks {
            for x in body.blocks.get(b).into_iter().flat_map(|b| b.inst_ids()) {
                m.tick(1)?;
                if remat.contains_key(&x) {
                    continue;
                }
                let mut chain = Vec::new();
                let mut at = Value::Inst(x);
                let found = loop {
                    let Some(c) = mine(at) else { break true };
                    if let Some(&r) = remat.get(&c) {
                        break r;
                    }
                    let inst = body.insts.get(c).ok_or_else(|| invariant("instruction"))?;
                    chain.push(c);
                    let ptr = matches!(parts.types.get(inst.ty), Some(Type::Ptr { .. }));
                    match &inst.kind {
                        InstKind::Alloca { .. } => break true,
                        InstKind::Gep { base, .. } if ptr => at = *base,
                        InstKind::Cast {
                            op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                            v,
                        } if ptr => at = *v,
                        _ => break false,
                    }
                    if chain.len() > body.insts.len() {
                        return Err(invariant("an address chain without an end"));
                    }
                };
                m.charge(chain.len() * std::mem::size_of::<(InstId, bool)>() * 2)?;
                *held += chain.len() * std::mem::size_of::<(InstId, bool)>() * 2;
                for c in chain {
                    remat.insert(c, found);
                }
            }
        }
        // The copy's values a use may need: its other non-`void` results, densely numbered.
        let mut index: BTreeMap<InstId, u32> = BTreeMap::new();
        let mut values: Vec<InstId> = Vec::new();
        for &b in blocks {
            for x in body.blocks.get(b).into_iter().flat_map(|b| b.inst_ids()) {
                let inst = body.insts.get(x).ok_or_else(|| invariant("instruction"))?;
                if let InstKind::Alloca { ty, count, .. } = &inst.kind {
                    let n = match count {
                        Value::Const(c) => match parts.consts.get(*c).map(|k| &k.value) {
                            Some(Const::Int(n)) => *n,
                            _ => return Err(dynamic_alloca()),
                        },
                        _ => return Err(dynamic_alloca()),
                    };
                    let field = if n == 1 {
                        *ty
                    } else {
                        parts.types.intern(Type::Array { elem: *ty, len: n }, m)?
                    };
                    mc.fields.push(field);
                    let f = u32::try_from(mc.fields.len()).map_err(|_| invariant("fields"))?;
                    reach.insert(x, Reach::Alloca(f));
                    continue;
                }
                if remat.get(&x) == Some(&true) {
                    reach.insert(x, Reach::Remat);
                    continue;
                }
                if matches!(parts.types.get(inst.ty), Some(Type::Void)) {
                    continue;
                }
                let n = u32::try_from(values.len()).map_err(|_| invariant("values"))?;
                index.insert(x, n);
                values.push(x);
            }
        }
        let bytes = values.len() * std::mem::size_of::<(InstId, u32)>() * 3;
        m.charge(bytes)?;
        *held += bytes;
        // The values a use at an operand needs: the value, or an address's indices.
        let effective = |v: Value, out: &mut Vec<u32>| {
            let mut v = v;
            for _ in 0..=body.insts.len() {
                let Some(x) = mine(v) else { return };
                if let Some(&n) = index.get(&x) {
                    out.push(n);
                    return;
                }
                match body.insts.get(x).map(|i| &i.kind) {
                    Some(InstKind::Gep { base, indices, .. }) => {
                        for &i in indices.iter() {
                            if let Some(n) = mine(i).and_then(|i| index.get(&i)) {
                                out.push(*n);
                            }
                        }
                        v = *base;
                    }
                    Some(InstKind::Cast { v: from, .. }) => v = *from,
                    _ => return,
                }
            }
        };
        // Segments: each block cut after each recursive call.
        struct Seg {
            block: BlockId,
            uses: Vec<u32>,
            defs: Vec<u32>,
            /// Ends with a recursive call: its one successor is the next segment.
            call: bool,
        }
        let mut segs: Vec<Seg> = Vec::new();
        let mut first_seg: BTreeMap<BlockId, usize> = BTreeMap::new();
        let mut last_seg: BTreeMap<BlockId, usize> = BTreeMap::new();
        let mut def_seg: Vec<usize> = filled(m, held, values.len(), usize::MAX)?;
        let mut phi_uses: Vec<(BlockId, Value)> = Vec::new();
        for &b in blocks {
            let block = body.blocks.get(b).ok_or_else(|| invariant("block"))?;
            first_seg.insert(b, segs.len());
            segs.push(Seg {
                block: b,
                uses: Vec::new(),
                defs: Vec::new(),
                call: false,
            });
            let mut carried: Option<InstId> = None;
            for x in block.inst_ids() {
                m.tick(1)?;
                let inst = body.insts.get(x).ok_or_else(|| invariant("instruction"))?;
                let s = segs.len() - 1;
                if let Some(c) = carried.take() {
                    if let Some(&n) = index.get(&c) {
                        if let Some(d) = def_seg.get_mut(n as usize) {
                            *d = s;
                        }
                        if let Some(seg) = segs.get_mut(s) {
                            seg.defs.push(n);
                        }
                    }
                }
                let mut used = Vec::new();
                if let InstKind::Phi { incoming, .. } = &inst.kind {
                    for &(v, p) in incoming.iter() {
                        phi_uses.push((p, v));
                    }
                } else {
                    inst.kind.for_each_value(|v| effective(v, &mut used));
                }
                let seg = segs.get_mut(s).ok_or_else(|| invariant("segment"))?;
                seg.uses.extend(used);
                if calls.contains(&x) {
                    seg.call = true;
                    carried = Some(x);
                    segs.push(Seg {
                        block: b,
                        uses: Vec::new(),
                        defs: Vec::new(),
                        call: false,
                    });
                } else if let Some(&n) = index.get(&x) {
                    seg.defs.push(n);
                    if let Some(d) = def_seg.get_mut(n as usize) {
                        *d = s;
                    }
                }
            }
            if carried.is_some() {
                // A block's terminator follows each of its calls.
                return Err(invariant("a recursive call ends its block"));
            }
            last_seg.insert(b, segs.len() - 1);
        }
        for (p, v) in phi_uses {
            let mut used = Vec::new();
            effective(v, &mut used);
            let s = *last_seg
                .get(&p)
                .ok_or_else(|| invariant("a phi's predecessor"))?;
            if let Some(seg) = segs.get_mut(s) {
                seg.uses.extend(used);
            }
        }
        // A use in its value's own segment follows the definition.
        for (s, seg) in segs.iter_mut().enumerate() {
            seg.uses.retain(|&n| def_seg.get(n as usize) != Some(&s));
            seg.uses.sort_unstable();
            seg.uses.dedup();
        }
        let succs: Vec<Vec<usize>> = segs
            .iter()
            .enumerate()
            .map(|(s, seg)| {
                if seg.call {
                    vec![s + 1]
                } else {
                    body.successors(seg.block)
                        .filter_map(|b| first_seg.get(&b).copied())
                        .collect()
                }
            })
            .collect();
        let words = values.len().div_ceil(64);
        let bytes = segs.len().saturating_mul(words * 8 + 64);
        m.charge(bytes)?;
        *held += bytes;
        let mut live: Vec<Bits> = vec![Bits::new(values.len()); segs.len()];
        loop {
            let mut changed = false;
            for s in (0..segs.len()).rev() {
                m.tick(1)?;
                let mut out = Bits::new(values.len());
                for &t in succs.get(s).into_iter().flatten() {
                    if let Some(l) = live.get(t) {
                        for (o, w) in out.0.iter_mut().zip(&l.0) {
                            *o |= w;
                        }
                    }
                }
                let seg = segs.get(s).ok_or_else(|| invariant("segment"))?;
                for &n in &seg.defs {
                    if let Some(w) = out.0.get_mut((n / 64) as usize) {
                        *w &= !(1 << (n % 64));
                    }
                }
                for &n in &seg.uses {
                    out.set(n);
                }
                if let Some(l) = live.get_mut(s) {
                    if l.0 != out.0 {
                        *l = out;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        // The values live into a continuation: stored in the frame.
        let mut stored = Bits::new(values.len());
        for (s, seg) in segs.iter().enumerate() {
            if seg.call {
                if let Some(l) = live.get(s + 1) {
                    for (o, w) in stored.0.iter_mut().zip(&l.0) {
                        *o |= w;
                    }
                }
            }
        }
        for n in stored.ones() {
            let x = *values.get(n as usize).ok_or_else(|| invariant("value"))?;
            let ty = body
                .insts
                .get(x)
                .map(|i| i.ty)
                .ok_or_else(|| invariant("value"))?;
            mc.fields.push(ty);
            let f = u32::try_from(mc.fields.len()).map_err(|_| invariant("fields"))?;
            reach.insert(x, Reach::Slot(f));
        }
        Ok(())
    }
}

/// The refusal of an `alloca` of a dynamic count in a recursive function's copy.
fn dynamic_alloca() -> Error {
    Unsupported::not_implemented(Feature::Recursion).into()
}

/// A region's return blocks for one return type.
#[derive(Clone, Copy)]
struct Rets {
    /// Every return of the type branches here; its PHI (for a non-`void` type) is the value.
    ret: BlockId,
    value: Option<Value>,
    /// At depth above 0: the call site's continuation, by its number.
    pop: BlockId,
    /// The pop's switch, whose cases are filled last.
    switch: u32,
    /// At depth 0: the root's continuation, filled last.
    exit: BlockId,
}

/// The assembly's emission state.
struct Asm<'a> {
    rec: &'a Recursion,
    body: &'a Body,
    machines: &'a [Machine],
    reach: &'a BTreeMap<InstId, Reach>,
    owner: &'a [Option<(usize, FuncId)>],
    w: &'a Words,
    d: Draft,
    /// Old values the machine replaced.
    replace: BTreeMap<u32, Value>,
    /// Per old block, the draft block holding its terminator.
    tail: Vec<BlockId>,
    /// Per region, its depth and its stack.
    depth: Vec<Value>,
    stack: Vec<Value>,
    unreach: Vec<BlockId>,
    rets: BTreeMap<(usize, TypeId), Rets>,
    /// Each parameter marker's PHI's incoming edges, and each return PHI's.
    params: BTreeMap<InstId, Vec<(Value, BlockId)>>,
    returns: BTreeMap<u32, Vec<(Value, BlockId)>>,
    /// Each pop switch's cases.
    cases: BTreeMap<u32, Vec<(u64, BlockId)>>,
    /// Each region's root continuation and its type.
    roots: BTreeMap<usize, (BlockId, TypeId)>,
    /// Per old block, the PHI operands of its successors it computes before its terminator:
    /// `(phi, incoming position, value)`.
    edges: BTreeMap<BlockId, Vec<(InstId, usize, Value)>>,
}

impl Recursion {
    fn emit(
        &self,
        body: &Body,
        parts: &mut ModuleParts,
        machines: &[Machine],
        reach: &BTreeMap<InstId, Reach>,
        owner: &[Option<(usize, FuncId)>],
        w: &Words,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<Body, Error> {
        let mut insts = Vec::new();
        m.reserve(&mut insts, body.insts.len())?;
        *held += body.insts.len() * std::mem::size_of::<Inst>();
        for inst in body.insts.values() {
            m.charge(inst.kind.heap_bytes())?;
            *held += inst.kind.heap_bytes();
            insts.push(inst.clone());
        }
        let mut a = Asm {
            rec: self,
            body,
            machines,
            reach,
            owner,
            w,
            d: Draft {
                insts,
                blocks: Vec::new(),
                names: Vec::new(),
            },
            replace: BTreeMap::new(),
            tail: Vec::new(),
            depth: Vec::new(),
            stack: Vec::new(),
            unreach: Vec::new(),
            rets: BTreeMap::new(),
            params: BTreeMap::new(),
            returns: BTreeMap::new(),
            cases: BTreeMap::new(),
            roots: BTreeMap::new(),
            edges: BTreeMap::new(),
        };
        for (b, block) in body.blocks.iter() {
            let id = a.d.block(m, held)?;
            if let Some(n) = a.d.names.get_mut(id.index()) {
                *n = block.name;
            }
            a.tail.push(b);
        }
        // Each region's depth and stack, allocated first in the entry block, and its blocks.
        let mut allocas = Vec::new();
        for (r, mc) in machines.iter().enumerate() {
            let one = w.one;
            let depth = a.make(
                (
                    w.ptr,
                    InstKind::Alloca {
                        ty: w.i32,
                        count: one,
                        align: None,
                    },
                ),
                m,
                held,
            )?;
            let stack = a.make(
                (
                    w.ptr,
                    InstKind::Alloca {
                        ty: mc.stack_ty,
                        count: one,
                        align: None,
                    },
                ),
                m,
                held,
            )?;
            allocas.push(depth);
            allocas.push(stack);
            a.depth.push(Value::Inst(InstId::new(depth)));
            a.stack.push(Value::Inst(InstId::new(stack)));
            let unreach = a.d.block(m, held)?;
            a.d.push(unreach, (w.void, InstKind::Unreachable), m, held)?;
            a.unreach.push(unreach);
            let comp = self
                .regions
                .get(r)
                .and_then(|g| self.comps.get(g.comp))
                .ok_or_else(|| invariant("component"))?;
            let mut types = BTreeSet::new();
            for &member in &comp.members {
                types.insert(signature(parts, fn_ty(parts, member)?)?.0);
            }
            for ty in types {
                let rets = a.rets_block(parts, r, ty, m, held)?;
                a.rets.insert((r, ty), rets);
            }
        }
        // Each copy's PHI operands that its predecessor computes.
        for (b, block) in body.blocks.iter() {
            if owner.get(b.index()).copied().flatten().is_none() {
                continue;
            }
            for x in block.inst_ids() {
                let Some(InstKind::Phi { incoming, .. }) = body.insts.get(x).map(|i| &i.kind)
                else {
                    continue;
                };
                for (k, &(v, p)) in incoming.iter().enumerate() {
                    if a.needs(v, owner.get(b.index()).copied().flatten()) {
                        m.charge(std::mem::size_of::<(InstId, usize, Value)>())?;
                        *held += std::mem::size_of::<(InstId, usize, Value)>();
                        a.edges.entry(p).or_default().push((x, k, v));
                    }
                }
            }
        }
        for (b, _) in body.blocks.iter() {
            a.block(parts, b, &allocas, m, held)?;
        }
        a.finish(m, held)
    }
}

impl Asm<'_> {
    /// A new instruction placed in no block yet.
    fn make(
        &mut self,
        (ty, kind): (TypeId, InstKind),
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<u32, Error> {
        let id = u32::try_from(self.d.insts.len()).map_err(|_| invariant("instruction ids"))?;
        let bytes = std::mem::size_of::<Inst>() + kind.heap_bytes();
        m.charge(bytes)?;
        *held += bytes;
        self.d.insts.push(Inst { ty, kind });
        Ok(id)
    }

    fn push(
        &mut self,
        b: BlockId,
        ty: TypeId,
        kind: InstKind,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<Value, Error> {
        self.d.push(b, (ty, kind), m, held)
    }

    /// Region `r`'s return blocks for type `ty`.
    fn rets_block(
        &mut self,
        parts: &mut ModuleParts,
        r: usize,
        ty: TypeId,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<Rets, Error> {
        let w = self.w;
        let ret = self.d.block(m, held)?;
        let pop = self.d.block(m, held)?;
        let exit = self.d.block(m, held)?;
        let value = if ty == w.void {
            None
        } else {
            let phi = self.push(
                ret,
                ty,
                InstKind::Phi {
                    incoming: Box::new([]),
                    fmf: FastMath::NONE,
                },
                m,
                held,
            )?;
            Some(phi)
        };
        let depth = *self.depth.get(r).ok_or_else(|| invariant("region"))?;
        let stack = *self.stack.get(r).ok_or_else(|| invariant("region"))?;
        let d = self.push(ret, w.i32, load(depth), m, held)?;
        let top = self.push(
            ret,
            w.i1,
            InstKind::ICmp {
                pred: IntPredicate::Eq,
                a: d,
                b: w.zero,
            },
            m,
            held,
        )?;
        self.push(
            ret,
            w.void,
            InstKind::CondBr {
                c: top,
                t: exit,
                f: pop,
                hints: None,
            },
            m,
            held,
        )?;
        let at = self.field(parts, (r, stack, d), 0, pop, m, held)?;
        let id = self.push(pop, w.i32, load(at), m, held)?;
        let below = self.push(
            pop,
            w.i32,
            InstKind::BinOp {
                op: BinOp::Sub,
                flags: IntFlags::NONE,
                fmf: FastMath::NONE,
                a: d,
                b: w.one,
            },
            m,
            held,
        )?;
        self.push(pop, w.void, store(below, depth), m, held)?;
        let unreach = *self.unreach.get(r).ok_or_else(|| invariant("region"))?;
        let switch = self.push(
            pop,
            w.void,
            InstKind::Switch {
                v: id,
                default: unreach,
                cases: Box::new([]),
                hints: None,
            },
            m,
            held,
        )?;
        let Value::Inst(switch) = switch else {
            return Err(invariant("a switch that is no instruction"));
        };
        Ok(Rets {
            ret,
            value,
            pop,
            switch: switch.raw(),
            exit,
        })
    }

    /// The address of field `f` of frame `d` of region `r`'s stack, computed in `b`.
    fn field(
        &mut self,
        parts: &mut ModuleParts,
        (r, stack, d): (usize, Value, Value),
        f: u32,
        b: BlockId,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<Value, Error> {
        let ty = self
            .machines
            .get(r)
            .map(|mc| mc.stack_ty)
            .ok_or_else(|| invariant("region"))?;
        let f = int(parts, self.w.i32, u64::from(f), m)?;
        self.push(
            b,
            self.w.ptr,
            InstKind::Gep {
                src_ty: ty,
                base: stack,
                indices: Box::new([self.w.zero, d, f]),
                inbounds: true,
            },
            m,
            held,
        )
    }

    /// The address of field `f` of the current frame of region `r`, computed in `b`.
    fn current(
        &mut self,
        parts: &mut ModuleParts,
        r: usize,
        f: u32,
        b: BlockId,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<Value, Error> {
        let depth = *self.depth.get(r).ok_or_else(|| invariant("region"))?;
        let stack = *self.stack.get(r).ok_or_else(|| invariant("region"))?;
        let d = self.push(b, self.w.i32, load(depth), m, held)?;
        self.field(parts, (r, stack, d), f, b, m, held)
    }

    /// Whether operand `v` of an instruction of copy `copy` is reached otherwise than as itself.
    fn needs(&self, v: Value, copy: Option<(usize, FuncId)>) -> bool {
        match v {
            Value::Inst(x) => copy.is_some() && self.reach.contains_key(&x),
            _ => false,
        }
    }

    /// Operand `v` of an instruction of copy `copy`, as it is reached in block `b`.
    fn fix(
        &mut self,
        parts: &mut ModuleParts,
        v: Value,
        copy: Option<(usize, FuncId)>,
        b: BlockId,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<Value, Error> {
        let (Value::Inst(x), Some((r, _))) = (v, copy) else {
            return Ok(v);
        };
        let body = self.body;
        match self.reach.get(&x).copied() {
            None => Ok(v),
            Some(Reach::Slot(f)) => {
                let ty = body
                    .insts
                    .get(x)
                    .map(|i| i.ty)
                    .ok_or_else(|| invariant("value"))?;
                let at = self.current(parts, r, f, b, m, held)?;
                self.push(b, ty, load(at), m, held)
            }
            Some(Reach::Alloca(f)) => self.current(parts, r, f, b, m, held),
            Some(Reach::Remat) => {
                // The chain down to an alloca or a value outside the copy, then up again.
                let mut chain = vec![x];
                let mut at = x;
                let bottom = loop {
                    let base = match body.insts.get(at).map(|i| &i.kind) {
                        Some(InstKind::Gep { base, .. }) => *base,
                        Some(InstKind::Cast { v, .. }) => *v,
                        _ => return Err(invariant("an address that is no chain")),
                    };
                    match base {
                        Value::Inst(y) => match self.reach.get(&y).copied() {
                            Some(Reach::Remat) => {
                                chain.push(y);
                                at = y;
                            }
                            Some(Reach::Alloca(f)) => {
                                break self.current(parts, r, f, b, m, held)?;
                            }
                            _ => break base,
                        },
                        _ => break base,
                    }
                    if chain.len() > body.insts.len() {
                        return Err(invariant("an address chain without an end"));
                    }
                };
                let mut v = bottom;
                for &y in chain.iter().rev() {
                    let inst = body.insts.get(y).ok_or_else(|| invariant("value"))?;
                    let ty = inst.ty;
                    let kind = match &inst.kind {
                        InstKind::Gep {
                            src_ty,
                            indices,
                            inbounds,
                            ..
                        } => {
                            let (src_ty, inbounds) = (*src_ty, *inbounds);
                            let indices = indices.to_vec();
                            let mut fixed = Vec::with_capacity(indices.len());
                            for i in indices {
                                fixed.push(self.fix(parts, i, copy, b, m, held)?);
                            }
                            InstKind::Gep {
                                src_ty,
                                base: v,
                                indices: fixed.into_boxed_slice(),
                                inbounds,
                            }
                        }
                        InstKind::Cast { op, .. } => InstKind::Cast { op: *op, v },
                        _ => return Err(invariant("an address that is no chain")),
                    };
                    v = self.push(b, ty, kind, m, held)?;
                }
                Ok(v)
            }
        }
    }

    /// Store value `v` (instruction `x` of copy `copy`) to its frame field, in block `b`.
    fn keep(
        &mut self,
        parts: &mut ModuleParts,
        (x, v): (InstId, Value),
        copy: Option<(usize, FuncId)>,
        b: BlockId,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<(), Error> {
        if let (Some(Reach::Slot(f)), Some((r, _))) = (self.reach.get(&x).copied(), copy) {
            let at = self.current(parts, r, f, b, m, held)?;
            self.push(b, self.w.void, store(v, at), m, held)?;
        }
        Ok(())
    }

    /// Emit old block `b`.
    fn block(
        &mut self,
        parts: &mut ModuleParts,
        b: BlockId,
        allocas: &[u32],
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<(), Error> {
        let body = self.body;
        let block = *body.blocks.get(b).ok_or_else(|| invariant("block"))?;
        let copy = self.owner.get(b.index()).copied().flatten();
        let mut cur = b;
        if b.index() == 0 {
            for &x in allocas {
                self.d.place(cur, x, m, held)?;
            }
        }
        // Stores of PHI results wait for the block's other PHIs.
        let mut pending: Vec<(InstId, Value)> = Vec::new();
        let ids: Vec<InstId> = block.inst_ids().collect();
        let mut k = 0usize;
        while let Some(&x) = ids.get(k) {
            k += 1;
            m.tick(1)?;
            let inst = body.insts.get(x).ok_or_else(|| invariant("instruction"))?;
            let marker = self.rec.marker(&inst.kind).map(|(_, kind)| kind);
            let phi_like = matches!(inst.kind, InstKind::Phi { .. })
                || matches!(marker, Some(Marker::Head { .. } | Marker::Param { .. }));
            if !phi_like {
                for (y, v) in pending.drain(..) {
                    self.keep(parts, (y, v), copy, cur, m, held)?;
                }
            }
            match marker {
                Some(Marker::Head { .. }) => {}
                Some(Marker::Param { .. }) => {
                    let phi = self.push(
                        cur,
                        inst.ty,
                        InstKind::Phi {
                            incoming: Box::new([]),
                            fmf: FastMath::NONE,
                        },
                        m,
                        held,
                    )?;
                    self.replace.insert(x.raw(), phi);
                    self.params.entry(x).or_default();
                    pending.push((x, phi));
                }
                Some(Marker::Ret { region, member }) => {
                    let InstKind::Call(call) = &inst.kind else {
                        return Err(invariant("a return that is no call"));
                    };
                    let arg = call.args.first().copied();
                    let ret = signature(parts, fn_ty(parts, member)?)?.0;
                    let rets = *self
                        .rets
                        .get(&(region, ret))
                        .ok_or_else(|| invariant("a return type"))?;
                    let v = match arg {
                        Some(v) => Some(self.fix(parts, v, copy, cur, m, held)?),
                        None => None,
                    };
                    self.edge_values(parts, b, copy, cur, m, held)?;
                    self.push(
                        cur,
                        self.w.void,
                        InstKind::Br {
                            dest: rets.ret,
                            hints: None,
                        },
                        m,
                        held,
                    )?;
                    if let (Some(v), Some(Value::Inst(phi))) = (v, rets.value) {
                        self.returns.entry(phi.raw()).or_default().push((v, cur));
                    }
                    let rest: Vec<&InstKind> = ids
                        .get(k..)
                        .into_iter()
                        .flatten()
                        .filter_map(|&y| body.insts.get(y).map(|i| &i.kind))
                        .collect();
                    if !matches!(rest.as_slice(), [InstKind::Unreachable]) {
                        return Err(invariant("a return not followed by unreachable"));
                    }
                    break;
                }
                Some(Marker::Enter { .. } | Marker::Dispatch { .. }) => {
                    cur = self.recursive_call(parts, x, copy, cur, m, held)?;
                }
                Some(Marker::Root { region }) => {
                    cur = self.root_call(parts, x, region, copy, cur, m, held)?;
                }
                None => {
                    if copy.is_some()
                        && matches!(self.reach.get(&x), Some(Reach::Alloca(_) | Reach::Remat))
                    {
                        continue;
                    }
                    if let InstKind::Phi { .. } = inst.kind {
                        // Its operands' fixes are computed in its predecessors (`edges`).
                        self.d.place(cur, x.raw(), m, held)?;
                        pending.push((x, Value::Inst(x)));
                        continue;
                    }
                    let mut kind = inst.kind.clone();
                    let mut operands = Vec::new();
                    kind.for_each_value(|v| operands.push(v));
                    let mut fixed = Vec::with_capacity(operands.len());
                    for v in operands {
                        fixed.push(self.fix(parts, v, copy, cur, m, held)?);
                    }
                    let mut it = fixed.into_iter();
                    kind.for_each_value_mut(|v| {
                        if let Some(f) = it.next() {
                            *v = f;
                        }
                    });
                    if kind.is_terminator() {
                        self.edge_values(parts, b, copy, cur, m, held)?;
                    }
                    if let Some(slot) = self.d.insts.get_mut(x.index()) {
                        slot.kind = kind;
                    }
                    self.d.place(cur, x.raw(), m, held)?;
                    if !matches!(parts.types.get(inst.ty), Some(Type::Void)) {
                        self.keep(parts, (x, Value::Inst(x)), copy, cur, m, held)?;
                    }
                }
            }
        }
        for (y, v) in pending {
            self.keep(parts, (y, v), copy, cur, m, held)?;
        }
        if let Some(t) = self.tail.get_mut(b.index()) {
            *t = cur;
        }
        Ok(())
    }

    /// Compute, in `cur`, the operands old block `b`'s successors' PHIs take from it.
    fn edge_values(
        &mut self,
        parts: &mut ModuleParts,
        b: BlockId,
        copy: Option<(usize, FuncId)>,
        cur: BlockId,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<(), Error> {
        let Some(list) = self.edges.remove(&b) else {
            return Ok(());
        };
        for (phi, k, v) in list {
            let fixed = self.fix(parts, v, copy, cur, m, held)?;
            if let Some(InstKind::Phi { incoming, .. }) =
                self.d.insts.get_mut(phi.index()).map(|i| &mut i.kind)
            {
                if let Some(slot) = incoming.get_mut(k) {
                    slot.0 = fixed;
                }
            }
        }
        Ok(())
    }

    /// The heads a call of `marker` (a member, or a dispatch's members by word) enters in region
    /// `r`: each with its word (0 for a member) and the position of its first argument.
    fn targets(&self, r: usize, marker: Marker) -> Result<(Vec<(u64, FuncId)>, usize), Error> {
        let comp = self
            .rec
            .regions
            .get(r)
            .and_then(|g| self.rec.comps.get(g.comp))
            .ok_or_else(|| invariant("component"))?;
        Ok(match marker {
            Marker::Enter { member, .. } => (vec![(0, member)], 0),
            Marker::Dispatch { ty, .. } => (
                comp.words
                    .get(&ty)
                    .into_iter()
                    .flatten()
                    .map(|&(w, f)| (u64::from(w), f))
                    .collect(),
                1,
            ),
            _ => return Err(invariant("a call that is no recursive call")),
        })
    }

    /// Branch from `b` into region `r`'s heads for a call of `marker` with `args`.
    fn enter(
        &mut self,
        r: usize,
        marker: Marker,
        args: &[Value],
        b: BlockId,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<(), Error> {
        let (targets, skip) = self.targets(r, marker)?;
        let machines = self.machines;
        let mc = machines.get(r).ok_or_else(|| invariant("region"))?;
        let head = |f: FuncId| {
            mc.heads
                .get(&f)
                .copied()
                .ok_or_else(|| invariant("a member without a head"))
        };
        let kind = match marker {
            Marker::Enter { member, .. } => InstKind::Br {
                dest: head(member)?,
                hints: None,
            },
            _ => {
                let word = *args
                    .first()
                    .ok_or_else(|| invariant("a dispatch without a word"))?;
                let mut cases = Vec::with_capacity(targets.len());
                for &(w, f) in &targets {
                    cases.push((w, head(f)?));
                }
                InstKind::Switch {
                    v: word,
                    default: *self.unreach.get(r).ok_or_else(|| invariant("region"))?,
                    cases: cases.into_boxed_slice(),
                    hints: None,
                }
            }
        };
        // One incoming edge per case to a member's head, for each parameter it does not take
        // from the root.
        let mut edges = Vec::new();
        for &(_, f) in &targets {
            for (&(_, j), &param) in mc.params.range((f, 0)..=(f, u32::MAX)) {
                let v = *args
                    .get(j as usize + skip)
                    .ok_or_else(|| invariant("an argument out of range"))?;
                edges.push((param, v));
            }
        }
        for (param, v) in edges {
            m.charge(std::mem::size_of::<(Value, BlockId)>())?;
            *held += std::mem::size_of::<(Value, BlockId)>();
            self.params.entry(param).or_default().push((v, b));
        }
        self.push(b, self.w.void, kind, m, held)?;
        Ok(())
    }

    /// The recursive call `x` of copy `copy`, which `cur` ends at: its push, its no-room path,
    /// and its continuation, which this returns.
    fn recursive_call(
        &mut self,
        parts: &mut ModuleParts,
        x: InstId,
        copy: Option<(usize, FuncId)>,
        cur: BlockId,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<BlockId, Error> {
        let (r, _) = copy.ok_or_else(|| invariant("a recursive call outside a copy"))?;
        let body = self.body;
        let inst = body.insts.get(x).ok_or_else(|| invariant("instruction"))?;
        let (_, marker) = self
            .rec
            .marker(&inst.kind)
            .ok_or_else(|| invariant("a recursive call that is no marker call"))?;
        let InstKind::Call(call) = &inst.kind else {
            return Err(invariant("a recursive call that is no call"));
        };
        let ty = inst.ty;
        let raw: Vec<Value> = call.args.to_vec();
        let mut args = Vec::with_capacity(raw.len());
        for v in raw {
            args.push(self.fix(parts, v, copy, cur, m, held)?);
        }
        let machines = self.machines;
        let mc = machines.get(r).ok_or_else(|| invariant("region"))?;
        let number = mc
            .calls
            .iter()
            .position(|&c| c == x)
            .ok_or_else(|| invariant("an unnumbered call"))?;
        let frames = mc.frames;
        let w = self.w;
        let depth = *self.depth.get(r).ok_or_else(|| invariant("region"))?;
        let stack = *self.stack.get(r).ok_or_else(|| invariant("region"))?;
        let d = self.push(cur, w.i32, load(depth), m, held)?;
        let next = self.push(
            cur,
            w.i32,
            InstKind::BinOp {
                op: BinOp::Add,
                flags: IntFlags::NONE,
                fmf: FastMath::NONE,
                a: d,
                b: w.one,
            },
            m,
            held,
        )?;
        let n = int(parts, w.i32, frames, m)?;
        let room = self.push(
            cur,
            w.i1,
            InstKind::ICmp {
                pred: IntPredicate::Ult,
                a: next,
                b: n,
            },
            m,
            held,
        )?;
        let push = self.d.block(m, held)?;
        let over = self.d.block(m, held)?;
        let cont = self.d.block(m, held)?;
        self.push(
            cur,
            w.void,
            InstKind::CondBr {
                c: room,
                t: push,
                f: over,
                hints: None,
            },
            m,
            held,
        )?;
        let at = self.field(parts, (r, stack, next), 0, push, m, held)?;
        let id = int(parts, w.i32, number as u64, m)?;
        self.push(push, w.void, store(id, at), m, held)?;
        self.push(push, w.void, store(next, depth), m, held)?;
        self.enter(r, marker, &args, push, m, held)?;
        self.push(
            over,
            w.void,
            InstKind::Br {
                dest: cont,
                hints: None,
            },
            m,
            held,
        )?;
        let rets = *self
            .rets
            .get(&(r, ty))
            .ok_or_else(|| invariant("a call's return type"))?;
        if let Some(value) = rets.value {
            let zero = zero_of(parts, ty, m)?;
            let phi = self.push(
                cont,
                ty,
                InstKind::Phi {
                    incoming: Box::new([(value, rets.pop), (zero, over)]),
                    fmf: FastMath::NONE,
                },
                m,
                held,
            )?;
            self.replace.insert(x.raw(), phi);
            self.keep(parts, (x, phi), copy, cont, m, held)?;
        }
        self.cases
            .entry(rets.switch)
            .or_default()
            .push((number as u64, cont));
        Ok(cont)
    }

    /// The root call `x` of region `r` (in `cur`, inside copy `copy` or the entry's own code):
    /// it enters the machine at depth 0, and its continuation, which this returns, takes the
    /// root's return. Its activation is the stack's first frame, which every stack has.
    fn root_call(
        &mut self,
        parts: &mut ModuleParts,
        x: InstId,
        r: usize,
        copy: Option<(usize, FuncId)>,
        cur: BlockId,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<BlockId, Error> {
        let body = self.body;
        let inst = body.insts.get(x).ok_or_else(|| invariant("instruction"))?;
        let InstKind::Call(call) = &inst.kind else {
            return Err(invariant("a root that is no call"));
        };
        let ty = inst.ty;
        let raw: Vec<Value> = call.args.to_vec();
        let mut args = Vec::with_capacity(raw.len());
        for v in raw {
            args.push(self.fix(parts, v, copy, cur, m, held)?);
        }
        let root = self
            .rec
            .regions
            .get(r)
            .map(|g| g.root)
            .ok_or_else(|| invariant("region"))?;
        let depth = *self.depth.get(r).ok_or_else(|| invariant("region"))?;
        self.push(cur, self.w.void, store(self.w.zero, depth), m, held)?;
        self.enter(r, root, &args, cur, m, held)?;
        let cont = self.d.block(m, held)?;
        let rets = *self
            .rets
            .get(&(r, ty))
            .ok_or_else(|| invariant("a root's return type"))?;
        if let Some(value) = rets.value {
            let phi = self.push(
                cont,
                ty,
                InstKind::Phi {
                    incoming: Box::new([(value, rets.exit)]),
                    fmf: FastMath::NONE,
                },
                m,
                held,
            )?;
            self.replace.insert(x.raw(), phi);
            self.keep(parts, (x, phi), copy, cont, m, held)?;
        }
        self.roots.insert(r, (cont, ty));
        Ok(cont)
    }

    /// Fill what the blocks left open, and lay the draft out as a body.
    fn finish(self, m: &mut Meter, held: &mut usize) -> Result<Body, Error> {
        let Asm {
            body,
            w,
            mut d,
            replace,
            tail,
            rets,
            params,
            returns,
            cases,
            roots,
            ..
        } = self;
        // Each return type's exit: to the root's continuation for the root's type.
        for (&(r, ty), rets) in &rets {
            let kind = match roots.get(&r) {
                Some(&(cont, root_ty)) if root_ty == ty => InstKind::Br {
                    dest: cont,
                    hints: None,
                },
                _ => InstKind::Unreachable,
            };
            d.push(rets.exit, (w.void, kind), m, held)?;
        }
        for (switch, list) in cases {
            if let Some(InstKind::Switch { cases, .. }) =
                d.insts.get_mut(switch as usize).map(|i| &mut i.kind)
            {
                *cases = list.into_boxed_slice();
            }
        }
        for (phi, list) in returns {
            if let Some(InstKind::Phi { incoming, .. }) =
                d.insts.get_mut(phi as usize).map(|i| &mut i.kind)
            {
                *incoming = list.into_boxed_slice();
            }
        }
        for (param, list) in params {
            let Some(&Value::Inst(phi)) = replace.get(&param.raw()) else {
                return Err(invariant("a parameter without a PHI"));
            };
            if let Some(InstKind::Phi { incoming, .. }) =
                d.insts.get_mut(phi.index()).map(|i| &mut i.kind)
            {
                *incoming = list.into_boxed_slice();
            }
        }
        // The old PHIs name the blocks that now end their predecessors.
        for inst in d.insts.iter_mut().take(body.insts.len()) {
            if let InstKind::Phi { incoming, .. } = &mut inst.kind {
                for (_, b) in incoming.iter_mut() {
                    if let Some(&t) = tail.get(b.index()) {
                        *b = t;
                    }
                }
            }
        }
        // Lay out: the final id of each placed instruction.
        let mut final_id: Vec<u32> = filled(m, held, d.insts.len(), u32::MAX)?;
        let mut next = 0u32;
        for block in &d.blocks {
            for &p in block {
                if let Some(slot) = final_id.get_mut(p as usize) {
                    *slot = next;
                    next += 1;
                }
            }
        }
        let replace = &replace;
        let resolve = |v: Value| -> Result<Value, Error> {
            let mut v = v;
            for _ in 0..=replace.len() {
                match v {
                    Value::Inst(i) => match replace.get(&i.raw()) {
                        Some(&r) => v = r,
                        None => {
                            let f = final_id
                                .get(i.index())
                                .copied()
                                .filter(|&f| f != u32::MAX)
                                .ok_or_else(|| invariant("a use of an instruction not placed"))?;
                            return Ok(Value::Inst(InstId::new(f)));
                        }
                    },
                    _ => return Ok(v),
                }
            }
            Err(invariant("replacements form a cycle"))
        };
        let mut out = Body::default();
        for (block, name) in d.blocks.iter().zip(&d.names) {
            let start = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
            for &p in block {
                let inst = d.insts.get(p as usize).ok_or_else(|| invariant("draft"))?;
                let mut kind = inst.kind.clone();
                let mut failed = None;
                kind.for_each_value_mut(|v| match resolve(*v) {
                    Ok(r) => *v = r,
                    Err(e) => failed = Some(e),
                });
                if let Some(e) = failed {
                    return Err(e);
                }
                out.push_inst(Inst { ty: inst.ty, kind }, m)?;
            }
            let end = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
            out.blocks.push(
                Block {
                    start,
                    end,
                    name: *name,
                },
                m,
            )?;
        }
        Ok(out)
    }
}

/// A `load` of the instruction's type from `ptr`.
fn load(ptr: Value) -> InstKind {
    InstKind::Load {
        ptr,
        align: None,
        volatile: false,
        atomic: None,
        range: None,
    }
}

/// A `store` of `val` to `ptr`.
fn store(val: Value, ptr: Value) -> InstKind {
    InstKind::Store {
        val,
        ptr,
        align: None,
        volatile: false,
        atomic: None,
    }
}
