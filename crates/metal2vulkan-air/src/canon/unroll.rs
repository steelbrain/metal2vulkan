//! Full unrolling of the loops whose trip count the bound constants decided.
//!
//! After SCCP, a loop whose counter starts at a constant and steps by constants runs the same
//! number of times on every run, and the values computed from its counter (an element of a
//! function-constant vector the counter indexes, a field of a constant global at the counter, a
//! shift by it) are constants in each iteration. When one of them decides a branch or a switch in
//! the loop's body, the loop is unrolled: its iterations become copies of its blocks in sequence,
//! each with its header's PHIs replaced by that iteration's values, so the SCCP that follows folds
//! what each copy decides, and deletes the paths no copy takes. A loop qualifies when:
//!
//! - it is reducible, with one preheader edge into its header and one back edge, from its latch;
//! - it is left only from its header or its latch, each a conditional branch with one way out;
//! - its exits are decided by integer arithmetic, comparisons, selects, element and member
//!   accesses and constant-global reads ([`super::const_loads::constant_field_load`]) on its
//!   header's PHIs and constants, evaluated iteration by iteration until one exit is taken, within
//!   [`TRIPS`] iterations and [`SLICE`] evaluated instructions an iteration (no float arithmetic or
//!   comparison is evaluated: those answers are the device's);
//! - a conditional branch or a switch of its body (not the exits of the loop or of a loop nested
//!   in it) tests a value computed that way from the PHIs of the loop's headers and of its nested
//!   loops' headers: unrolling decides it ([`profitable`]);
//! - its latch carries no `llvm.loop.unroll.disable`;
//! - its copies hold at most [`UNROLLED`] instructions, and the entry's unrolling adds at most
//!   [`GROWTH`] instructions over all rounds.
//!
//! Only the outermost qualifying loops are unrolled in a round; a loop nested in one is copied
//! with it, and the next round, after SCCP, takes each copy whose counter now decides its body.
//! There are at most [`ROUNDS`] rounds. The body all rounds leave, folded, replaces the entry's
//! only where it is no larger ([`live_size`]): where what the copies decide removes less than the
//! copies add (a loop whose counter picks among a few cheap arms of a body it runs many times),
//! the loops stay, as the module is what the driver compiles. Unrolling changes no operation: every instruction is copied
//! as it is, with its flags, and runs as many times as before, so what the program computes is the
//! same. The ceilings bound only how large the copies may grow, which bounds the translation's
//! time and memory; a loop past them stays a loop and computes the same answers.
//!
//! A value of the loop used after it is the last iteration's copy: the iteration that takes the
//! exit. An exit block keeps the PHI entry of the edge that iteration takes, and a block only another
//! exit reached is left unreachable, for SCCP to delete.

use crate::budget::Meter;
use crate::error::Error;
use crate::graph::{Cfg, LoopForest, LoopId};
use crate::ir::{Arena, Block, BlockId, Body, CastOp, Const, ConstId, Constant, Global};
use crate::ir::{GlobalId, Idx, Inst, InstId, InstKind, ModuleParts, Unroll, Value};

use super::const_loads;
use super::fold::Folder;
use super::rewrite::{filled, invariant, scoped};

/// The most iterations a loop is unrolled for.
pub(super) const TRIPS: u32 = 64;
/// The most instructions one loop's copies hold together.
pub(super) const UNROLLED: u64 = 8192;
/// The most instructions the unrolling of one entry adds, over all its rounds.
pub(super) const GROWTH: u64 = 1 << 15;
/// The most rounds of unrolling (and of the SCCP after each): the depth of a nest unrolled.
pub(super) const ROUNDS: u32 = 4;
/// The most instructions evaluated to decide one iteration's exits.
pub(super) const SLICE: u32 = 256;

/// No entry in a dense table.
const NONE: u32 = u32::MAX;

/// The size of `body` once nothing unobserved is left in it: every instruction that is not
/// [`super::sccp::is_pure`] (a store, a call, an atomic, a terminator), and every pure one such an
/// instruction reads, directly or through others, but for unconditional branches and PHIs of one
/// entry, which the copies of an unrolled loop leave where its blocks join. A pure value nothing
/// of those reads (a counter SCCP folded, an address no load uses any more) is not counted, as
/// the deletion of unobserved code removes it later.
pub(super) fn live_size(body: &Body, m: &mut Meter) -> Result<u64, Error> {
    scoped(m, |m, held| {
        let mut live = filled(m, held, body.insts.len(), false)?;
        let mut work: Vec<InstId> = Vec::new();
        for b in body.blocks.ids() {
            for (i, inst) in body.block_insts(b) {
                m.tick(1)?;
                if !super::sccp::is_pure(&inst.kind) {
                    mark(&mut work, &mut live, i, m)?;
                }
            }
        }
        let mut n = 0u64;
        let mut used: Vec<InstId> = Vec::new();
        while let Some(i) = work.pop() {
            m.tick(1)?;
            let inst = body
                .insts
                .get(i)
                .ok_or_else(|| invariant("instruction out of range"))?;
            // A jump and a PHI of one entry are no work of their own: they go where blocks merge.
            let copy = match &inst.kind {
                InstKind::Br { .. } => true,
                InstKind::Phi { incoming, .. } => incoming.len() == 1,
                _ => false,
            };
            n += u64::from(!copy);
            let mut count = 0;
            inst.kind.for_each_value(|_| count += 1);
            used.clear();
            m.grow(&mut used, count)?;
            inst.kind.for_each_value(|v| {
                if let Value::Inst(u) = v {
                    used.push(u);
                }
            });
            for &u in &used {
                mark(&mut work, &mut live, u, m)?;
            }
        }
        *held = held
            .saturating_add((work.capacity() + used.capacity()) * std::mem::size_of::<InstId>());
        Ok(n)
    })
}

/// Mark instruction `i` of [`live_size`] live and queue it, once.
fn mark(work: &mut Vec<InstId>, live: &mut [bool], i: InstId, m: &mut Meter) -> Result<(), Error> {
    match live.get_mut(i.index()) {
        Some(seen) if !*seen => {
            *seen = true;
            m.grow(work, 1)?;
            work.push(i);
            Ok(())
        }
        Some(_) => Ok(()),
        None => Err(invariant("instruction out of range")),
    }
}

/// One loop to unroll.
struct Plan {
    header: BlockId,
    latch: BlockId,
    /// The loop's blocks: the header first, then the others in layout order.
    order: Vec<BlockId>,
    /// The header's PHIs, each with its preheader and latch entries' values.
    phis: Vec<(InstId, Value, Value)>,
    /// The way into the loop the header's branch takes when it does not leave (a header that
    /// exits).
    stay: Option<BlockId>,
    /// How many times the header runs.
    trips: u32,
    /// The exit the last iteration takes: from the header or the latch, to a block outside.
    exit: (BlockId, BlockId),
    /// Instructions in one copy (the header's PHIs not counted), and in the last one, which is the
    /// header alone when the header exits.
    copy_insts: u32,
    last_insts: u32,
}

impl Plan {
    /// Whether the last copy is the header alone: the last iteration leaves from a header that is
    /// not the latch.
    fn header_only_last(&self) -> bool {
        self.exit.0 == self.header && self.latch != self.header
    }
}

/// One round of unrolling over `body` (module docs): the unrolled body, or `None` when no loop
/// qualifies. `grown` counts the instructions the entry's rounds added so far.
pub(super) fn round(
    body: &Body,
    parts: &mut ModuleParts,
    grown: &mut u64,
    m: &mut Meter,
) -> Result<Option<Body>, Error> {
    if *grown >= GROWTH || !has_back_edge(body) {
        return Ok(None);
    }
    let cfg = Cfg::build(body, m)?;
    let r = LoopForest::compute(&cfg, m).and_then(|forest| {
        let r = scoped(m, |m, held| {
            unroll_in(body, parts, (&cfg, &forest), grown, m, held)
        });
        m.release(forest.ledger_bytes());
        r
    });
    m.release(cfg.ledger_bytes());
    r
}

/// Whether some branch of `body` goes to its own block or an earlier one in layout: a body with
/// none has no loop, and needs no graph.
fn has_back_edge(body: &Body) -> bool {
    body.blocks
        .ids()
        .any(|b| body.successors(b).any(|s| s.raw() <= b.raw()))
}

/// What the analysis of one round reads.
struct Cx<'a> {
    body: &'a Body,
    cfg: &'a Cfg<BlockId>,
    forest: &'a LoopForest<BlockId>,
    block_of: Vec<BlockId>,
}

fn unroll_in(
    body: &Body,
    parts: &mut ModuleParts,
    (cfg, forest): (&Cfg<BlockId>, &LoopForest<BlockId>),
    grown: &mut u64,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Option<Body>, Error> {
    let mut block_of = filled(m, held, body.insts.len(), BlockId::new(0))?;
    for (b, block) in body.blocks.iter() {
        m.tick(1)?;
        for i in block.inst_ids() {
            if let Some(slot) = block_of.get_mut(i.index()) {
                *slot = b;
            }
        }
    }
    let cx = Cx {
        body,
        cfg,
        forest,
        block_of,
    };
    let mut inside = filled(m, held, body.insts.len(), false)?;
    let mut plans: Vec<Plan> = Vec::new();
    // Loops come outer before inner: one nested in a planned loop is copied with it.
    let mut planned: Vec<LoopId> = Vec::new();
    for l in forest.loops() {
        m.tick(1)?;
        if planned.iter().any(|&p| forest.loop_contains(p, l)) {
            continue;
        }
        let (types, consts, globals) = (&parts.types, &mut parts.consts, &parts.globals);
        let mut folder = Folder {
            types,
            consts,
            held: 0,
        };
        let plan = plan(&cx, l, (&mut folder, globals), &mut inside, *grown, m)?;
        if let Some(plan) = plan {
            let added =
                u64::from(plan.trips - 1) * u64::from(plan.copy_insts) + u64::from(plan.last_insts);
            *grown = grown.saturating_add(added.saturating_sub(u64::from(plan.copy_insts)));
            m.grow(&mut plans, 1)?;
            *held = held.saturating_add(std::mem::size_of::<Plan>());
            plans.push(plan);
            planned.push(l);
        }
    }
    if plans.is_empty() {
        return Ok(None);
    }
    build(&cx, &plans, &mut parts.consts, m, held).map(Some)
}

/// The plan of loop `l`, when it qualifies (module docs). `inside` is all false on entry and on
/// return.
fn plan(
    cx: &Cx<'_>,
    l: LoopId,
    (folder, globals): (&mut Folder<'_>, &Arena<GlobalId, Global>),
    inside: &mut [bool],
    grown: u64,
    m: &mut Meter,
) -> Result<Option<Plan>, Error> {
    let (body, forest) = (cx.body, cx.forest);
    let header = forest.header(l);
    if !forest.is_reducible(l) || header.index() == 0 {
        return Ok(None);
    }
    // One preheader edge and one back edge into the header.
    let (mut pre, mut latch) = (None, None);
    for p in cx.cfg.predecessors(header) {
        let slot = if forest.contains(l, p) {
            &mut latch
        } else {
            &mut pre
        };
        if slot.replace(p).is_some() {
            return Ok(None);
        }
    }
    let (Some(pre), Some(latch)) = (pre, latch) else {
        return Ok(None);
    };
    let edges_to = |b: BlockId| body.successors(b).filter(|&s| s == header).count();
    if edges_to(pre) != 1 || edges_to(latch) != 1 {
        return Ok(None);
    }
    let Some(term) = body.terminator(latch) else {
        return Ok(None);
    };
    if term
        .kind
        .loop_hints()
        .is_some_and(|h| h.unroll == Unroll::Disable)
    {
        return Ok(None);
    }
    // The size, and the exits: only from the header or the latch.
    let mut size = 0u64;
    for b in forest.blocks(l) {
        m.tick(1)?;
        let block = body
            .blocks
            .get(b)
            .ok_or_else(|| invariant("loop block out of range"))?;
        size += block.len() as u64;
        if size > UNROLLED {
            return Ok(None);
        }
        if b != header && b != latch && body.successors(b).any(|s| !forest.contains(l, s)) {
            return Ok(None);
        }
    }
    // A conditional branch with one way in the loop and one out: its condition, its way in, its
    // way out, and the condition's value that leaves.
    let exit_of = |b: BlockId| -> Option<Option<(Value, BlockId, BlockId, bool)>> {
        let term = body.terminator(b)?;
        if !body.successors(b).any(|s| !forest.contains(l, s)) {
            return Some(None);
        }
        match term.kind {
            InstKind::CondBr { c, t, f, .. } => {
                match (forest.contains(l, t), forest.contains(l, f)) {
                    (true, false) => Some(Some((c, t, f, false))),
                    (false, true) => Some(Some((c, f, t, true))),
                    _ => None,
                }
            }
            _ => None,
        }
    };
    let Some(header_exit) = exit_of(header) else {
        return Ok(None);
    };
    let Some(latch_exit) = exit_of(latch) else {
        return Ok(None);
    };
    if header_exit.is_none() && latch_exit.is_none() {
        return Ok(None);
    }
    if latch != header && latch_exit.is_some_and(|(_, way_in, _, _)| way_in != header) {
        return Ok(None);
    }
    if latch == header && header_exit.is_some_and(|(_, way_in, _, _)| way_in != header) {
        return Ok(None);
    }
    // The header's PHIs, each with exactly its two entries.
    let mut phis = Vec::new();
    for (i, inst) in body.block_insts(header) {
        let InstKind::Phi { incoming, .. } = &inst.kind else {
            break;
        };
        let (mut from_pre, mut from_latch) = (None, None);
        for &(v, p) in incoming.iter() {
            let slot = if p == pre {
                &mut from_pre
            } else if p == latch {
                &mut from_latch
            } else {
                return Ok(None);
            };
            if slot.replace(v).is_some() {
                return Ok(None);
            }
        }
        let (Some(a), Some(b)) = (from_pre, from_latch) else {
            return Ok(None);
        };
        m.grow(&mut phis, 1)?;
        phis.push((i, a, b));
    }
    let profitable = scoped(m, |m, _| profitable(cx, l, folder.types, inside, m));
    if !profitable? {
        m.release(phis.capacity() * std::mem::size_of::<(InstId, Value, Value)>());
        return Ok(None);
    }
    // The iterations, until one leaves.
    let mut vals: Vec<Option<ConstId>> = phis
        .iter()
        .map(|&(_, pre, _)| match pre {
            Value::Const(c) => Some(c),
            _ => None,
        })
        .collect();
    let mut sim = Sim {
        cx,
        l,
        phis: &phis,
        memo: Vec::new(),
        steps: 0,
    };
    let mut decided = None;
    for k in 1..=TRIPS {
        m.tick(1)?;
        sim.memo.clear();
        sim.steps = 0;
        let leaves = |sim: &mut Sim<'_>,
                      folder: &mut Folder<'_>,
                      vals: &[Option<ConstId>],
                      exit: Option<(Value, BlockId, BlockId, bool)>,
                      m: &mut Meter|
         -> Result<Option<Option<BlockId>>, Error> {
            let Some((c, _, out, when)) = exit else {
                return Ok(Some(None));
            };
            Ok(sim
                .eval(c, vals, (folder, globals), m)?
                .and_then(|x| folder.int(x))
                .map(|x| ((x != 0) == when).then_some(out)))
        };
        let Some(out) = leaves(&mut sim, folder, &vals, header_exit, m)? else {
            break;
        };
        if let Some(out) = out {
            decided = Some((k, (header, out)));
            break;
        }
        if latch != header {
            let Some(out) = leaves(&mut sim, folder, &vals, latch_exit, m)? else {
                break;
            };
            if let Some(out) = out {
                decided = Some((k, (latch, out)));
                break;
            }
        }
        let mut next = Vec::with_capacity(vals.len());
        for &(_, _, back) in &phis {
            next.push(sim.eval(back, &vals, (folder, globals), m)?);
        }
        vals = next;
    }
    let Some((trips, exit)) = decided else {
        return Ok(None);
    };
    let nphis = u32::try_from(phis.len()).map_err(|_| invariant("header PHIs"))?;
    let copy_insts = u32::try_from(size)
        .map_err(|_| invariant("loop size"))?
        .saturating_sub(nphis);
    let header_len = body.blocks.get(header).map_or(0, Block::len);
    let header_insts = u32::try_from(header_len)
        .map_err(|_| invariant("header size"))?
        .saturating_sub(nphis);
    let mut order = vec![header];
    order.extend(forest.blocks(l).filter(|&b| b != header));
    order.sort_by_key(|&b| (b != header, b.raw()));
    let plan = Plan {
        header,
        latch,
        order,
        phis,
        stay: header_exit.map(|(_, way_in, _, _)| way_in),
        trips,
        exit,
        copy_insts,
        last_insts: copy_insts,
    };
    let last_insts = if plan.header_only_last() {
        header_insts
    } else {
        copy_insts
    };
    let unrolled = u64::from(trips - 1) * u64::from(copy_insts) + u64::from(last_insts);
    if unrolled > UNROLLED || grown.saturating_add(unrolled.saturating_sub(size)) > GROWTH {
        return Ok(None);
    }
    m.charge(plan.order.capacity() * std::mem::size_of::<BlockId>())?;
    Ok(Some(Plan { last_insts, ..plan }))
}

/// Whether a conditional branch or switch of loop `l`'s body tests a value that is constant in
/// each iteration once the loop and the loops nested in it are unrolled: computed by integer
/// operations, element and member accesses and constant-global reads from constants and the PHIs
/// of their headers (those PHIs included when all their entries are such values). The exits of
/// `l` and of its nested loops are the loops' own control, not counted. `inside` is all false on
/// entry and on return.
fn profitable(
    cx: &Cx<'_>,
    l: LoopId,
    types: &crate::ir::TypeTable,
    inside: &mut [bool],
    m: &mut Meter,
) -> Result<bool, Error> {
    let (body, forest) = (cx.body, cx.forest);
    // The candidates, optimistically all in; then each whose operand is not drops out, until none
    // does.
    for b in forest.blocks(l) {
        let nested_header = forest
            .loop_of_header(b)
            .is_some_and(|h| forest.loop_contains(l, h));
        for (i, inst) in body.block_insts(b) {
            let ints = const_loads::integers_only(types, inst.ty);
            let candidate = match &inst.kind {
                InstKind::Phi { .. } => nested_header && ints,
                InstKind::BinOp { op, .. } => !op.is_float() && ints,
                InstKind::ICmp { .. } => true,
                InstKind::Cast {
                    op: CastOp::Trunc | CastOp::ZExt | CastOp::SExt,
                    ..
                } => ints,
                InstKind::Select { .. }
                | InstKind::Freeze(_)
                | InstKind::ExtractElement { .. }
                | InstKind::InsertElement { .. }
                | InstKind::ShuffleVector { .. }
                | InstKind::ExtractValue { .. }
                | InstKind::InsertValue { .. } => ints,
                InstKind::Gep { .. } => true,
                InstKind::Load {
                    volatile: false,
                    atomic: None,
                    ..
                } => ints,
                _ => false,
            };
            if let Some(slot) = inside.get_mut(i.index()) {
                *slot = candidate;
            }
        }
    }
    let known = |inside: &[bool], v: Value| match v {
        Value::Const(_) => true,
        Value::Arg(_) => false,
        Value::Inst(j) => inside.get(j.index()).copied().unwrap_or(false),
    };
    let mut changed = true;
    let mut passes = 0;
    while changed && passes < 16 {
        changed = false;
        passes += 1;
        for b in forest.blocks(l) {
            m.tick(1)?;
            for (i, inst) in body.block_insts(b) {
                if !inside.get(i.index()).copied().unwrap_or(false) {
                    continue;
                }
                let mut all = true;
                match &inst.kind {
                    InstKind::Phi { incoming, .. } => {
                        all = incoming.iter().all(|&(v, _)| known(inside, v));
                    }
                    kind => kind.for_each_value(|v| all &= known(inside, v)),
                }
                if !all {
                    if let Some(slot) = inside.get_mut(i.index()) {
                        *slot = false;
                    }
                    changed = true;
                }
            }
        }
    }
    let mut decides = false;
    if !changed {
        for b in forest.blocks(l) {
            let Some(term) = body.terminator(b) else {
                continue;
            };
            let c = match term.kind {
                InstKind::CondBr { c, .. } => c,
                InstKind::Switch { v, .. } => v,
                _ => continue,
            };
            // A loop's own exit: the innermost loop of the block is left by it.
            let own = forest
                .innermost(b)
                .is_some_and(|inner| body.successors(b).any(|s| !forest.contains(inner, s)));
            if !own && matches!(c, Value::Inst(_)) && known(inside, c) {
                decides = true;
                break;
            }
        }
    }
    for b in forest.blocks(l) {
        for i in body.blocks.get(b).into_iter().flat_map(Block::inst_ids) {
            if let Some(slot) = inside.get_mut(i.index()) {
                *slot = false;
            }
        }
    }
    Ok(decides)
}

/// The evaluation of one iteration's exits.
struct Sim<'a> {
    cx: &'a Cx<'a>,
    l: LoopId,
    phis: &'a [(InstId, Value, Value)],
    /// The values this iteration computed so far.
    memo: Vec<(InstId, Option<ConstId>)>,
    steps: u32,
}

impl Sim<'_> {
    /// The value of `v` in the iteration whose header PHIs hold `vals`: `None` when it is not a
    /// constant this evaluation computes (module docs).
    fn eval(
        &mut self,
        v: Value,
        vals: &[Option<ConstId>],
        (f, globals): (&mut Folder<'_>, &Arena<GlobalId, Global>),
        m: &mut Meter,
    ) -> Result<Option<ConstId>, Error> {
        let i = match v {
            Value::Const(c) => return Ok(Some(c)),
            Value::Arg(_) => return Ok(None),
            Value::Inst(i) => i,
        };
        if let Some(k) = self.phis.iter().position(|&(p, _, _)| p == i) {
            return Ok(vals.get(k).copied().flatten());
        }
        if let Some(&(_, c)) = self.memo.iter().find(|&&(j, _)| j == i) {
            return Ok(c);
        }
        let b = self.cx.block_of.get(i.index()).copied();
        if b.and_then(|b| self.cx.forest.innermost(b)) != Some(self.l) {
            return Ok(None);
        }
        self.steps += 1;
        if self.steps > SLICE {
            return Ok(None);
        }
        m.tick(1)?;
        let inst = self
            .cx
            .body
            .insts
            .get(i)
            .ok_or_else(|| invariant("instruction out of range"))?;
        let ty = inst.ty;
        let mut ops = Vec::new();
        match &inst.kind {
            InstKind::Phi { .. } => {}
            kind => kind.for_each_value(|v| ops.push(v)),
        }
        let mut cs = Vec::with_capacity(ops.len());
        for v in ops {
            match self.eval(v, vals, (&mut *f, globals), m)? {
                Some(c) => cs.push(c),
                None => {
                    self.memo.push((i, None));
                    return Ok(None);
                }
            }
        }
        let r = match (&inst.kind, cs.as_slice()) {
            (InstKind::BinOp { op, .. }, &[a, b]) if !op.is_float() => f.binop(*op, ty, a, b, m)?,
            (InstKind::ICmp { pred, .. }, &[a, b]) => f.icmp(*pred, ty, a, b, m)?,
            (
                InstKind::Cast {
                    op: op @ (CastOp::Trunc | CastOp::ZExt | CastOp::SExt),
                    ..
                },
                &[x],
            ) => f.cast(*op, x, ty, m)?,
            (InstKind::Select { .. }, &[c, t, e]) => match f.int(c) {
                Some(0) => Some(e),
                Some(_) => Some(t),
                None => None,
            },
            (InstKind::Freeze(_), &[x]) => f.freeze(x, m)?,
            (InstKind::ExtractElement { .. }, &[v, x]) => f.extract_element(v, x, m)?,
            (InstKind::InsertElement { .. }, &[v, e, x]) => f.insert_element(ty, v, e, x, m)?,
            (InstKind::ExtractValue { indices, .. }, &[a]) => f.extract_value(a, indices, m)?,
            (InstKind::InsertValue { indices, .. }, &[a, v]) => f.insert_value(a, v, indices, m)?,
            (InstKind::ShuffleVector { mask, .. }, &[a, b]) => f.shuffle(ty, a, b, mask, m)?,
            (
                InstKind::Gep {
                    src_ty, inbounds, ..
                },
                ops,
            ) => const_loads::constant_field_address(f, globals, (*src_ty, *inbounds, ty), ops, m)?,
            (
                InstKind::Load {
                    volatile: false,
                    atomic: None,
                    ..
                },
                &[p],
            ) => const_loads::constant_field_load(f, globals, p, ty, m)?,
            _ => None,
        };
        // An undefined value decides nothing here.
        let r = r.filter(|&c| {
            !matches!(
                f.consts.get(c).map(|k| &k.value),
                Some(Const::Undef | Const::Poison)
            )
        });
        self.memo.push((i, r));
        Ok(r)
    }
}

/// The body with every planned loop unrolled (module docs).
fn build(
    cx: &Cx<'_>,
    plans: &[Plan],
    consts: &mut crate::ir::ConstTable,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Body, Error> {
    let body = cx.body;
    let nblocks = body.blocks.len();
    // Per old block: its plan, its place in its plan's copies, or its new id (an outside block).
    let mut owner = filled(m, held, nblocks, NONE)?;
    let mut local_block = filled(m, held, nblocks, NONE)?;
    let mut local = filled(m, held, body.insts.len(), NONE)?;
    let mut phi_slot = filled(m, held, body.insts.len(), NONE)?;
    for (p, plan) in plans.iter().enumerate() {
        let p = u32::try_from(p).map_err(|_| invariant("plans"))?;
        let mut n = 0u32;
        for (k, &b) in plan.order.iter().enumerate() {
            m.tick(1)?;
            *owner
                .get_mut(b.index())
                .ok_or_else(|| invariant("block out of range"))? = p;
            *local_block
                .get_mut(b.index())
                .ok_or_else(|| invariant("block out of range"))? =
                u32::try_from(k).map_err(|_| invariant("blocks"))?;
            for (i, _) in body.block_insts(b) {
                if let Some(h) = plan.phis.iter().position(|&(q, _, _)| q == i) {
                    *at_mut(&mut phi_slot, i.index())? =
                        u32::try_from(h).map_err(|_| invariant("PHIs"))?;
                    continue;
                }
                *at_mut(&mut local, i.index())? = n;
                n += 1;
            }
        }
        if n != plan.copy_insts {
            return Err(invariant("a loop copy is not its measured size"));
        }
    }
    // The emission order: every outside block in layout order, each loop's copies where its first
    // block in layout was. The new ids follow it.
    let mut outside_block = filled(m, held, nblocks, NONE)?;
    let mut outside_inst = filled(m, held, nblocks, NONE)?;
    let mut copy_block = Vec::new();
    let mut copy_inst = Vec::new();
    for plan in plans {
        let n = plan.trips as usize;
        let mut b = Vec::new();
        let mut i = Vec::new();
        m.reserve(&mut b, n)?;
        m.reserve(&mut i, n)?;
        *held = held.saturating_add(2 * n * std::mem::size_of::<u32>());
        b.resize(n, 0u32);
        i.resize(n, 0u32);
        copy_block.push(b);
        copy_inst.push(i);
    }
    let mut seq: Vec<(u32, u32, BlockId)> = Vec::new();
    let (mut nb, mut ni) = (0u32, 0u32);
    let mut emitted = vec![false; plans.len()];
    for (b, block) in body.blocks.iter() {
        m.tick(1)?;
        let p = at(&owner, b.index())?;
        if p == NONE {
            *at_mut(&mut outside_block, b.index())? = nb;
            *at_mut(&mut outside_inst, b.index())? = ni;
            m.grow(&mut seq, 1)?;
            seq.push((NONE, 0, b));
            nb += 1;
            ni += u32::try_from(block.len()).map_err(|_| invariant("block size"))?;
            continue;
        }
        let pi = p as usize;
        let done = at_mut(&mut emitted, pi)?;
        if *done {
            continue;
        }
        *done = true;
        let plan = plans.get(pi).ok_or_else(|| invariant("plans"))?;
        for k in 1..=plan.trips {
            let copy = (k - 1) as usize;
            *at_mut(
                copy_block.get_mut(pi).ok_or_else(|| invariant("plans"))?,
                copy,
            )? = nb;
            *at_mut(
                copy_inst.get_mut(pi).ok_or_else(|| invariant("plans"))?,
                copy,
            )? = ni;
            let last = k == plan.trips && plan.header_only_last();
            let blocks: &[BlockId] = if last {
                plan.order
                    .get(..1)
                    .ok_or_else(|| invariant("a loop of no block"))?
            } else {
                &plan.order
            };
            m.grow(&mut seq, blocks.len())?;
            for &ob in blocks {
                seq.push((p, k, ob));
            }
            nb += u32::try_from(blocks.len()).map_err(|_| invariant("blocks"))?;
            ni += if last {
                plan.last_insts
            } else {
                plan.copy_insts
            };
        }
    }
    *held = held.saturating_add(seq.capacity() * std::mem::size_of::<(u32, u32, BlockId)>());
    let map = Map {
        body,
        plans,
        block_of: &cx.block_of,
        owner: &owner,
        local: &local,
        local_block: &local_block,
        phi_slot: &phi_slot,
        outside_block: &outside_block,
        outside_inst: &outside_inst,
        copy_block: &copy_block,
        copy_inst: &copy_inst,
    };
    // Each copy's header PHIs: the first iteration's from the preheader, each later one's the
    // previous copy's latch values.
    let mut subs: Vec<Vec<Value>> = Vec::with_capacity(plans.len());
    for (p, plan) in plans.iter().enumerate() {
        let mut s = Vec::new();
        let n = plan.trips as usize * plan.phis.len();
        m.reserve(&mut s, n)?;
        *held = held.saturating_add(n * std::mem::size_of::<Value>());
        for k in 1..=plan.trips {
            for &(_, pre, back) in &plan.phis {
                let v = if k == 1 {
                    map.value(None, pre, &subs, &s, consts, m)?
                } else {
                    map.value(Some((p as u32, k - 1)), back, &subs, &s, consts, m)?
                };
                s.push(v);
            }
        }
        subs.push(s);
    }
    let mut out = Body::with_capacity(ni as usize, nb as usize, m)?;
    for &(p, k, b) in &seq {
        m.tick(1)?;
        let block = *body
            .blocks
            .get(b)
            .ok_or_else(|| invariant("block out of range"))?;
        let start = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
        let ctx = (p != NONE).then_some((p, k));
        for (i, inst) in body.block_insts(b) {
            if ctx.is_some() && at(&phi_slot, i.index())? != NONE {
                continue;
            }
            let kind = map.inst(ctx, b, inst, &subs, consts, m)?;
            out.push_inst(Inst { ty: inst.ty, kind }, m)?;
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
    if out.insts.len() != ni as usize || out.blocks.len() != nb as usize {
        return Err(invariant("the unrolled body is not its measured size"));
    }
    Ok(out)
}

/// Entry `i` of a dense table.
fn at<T: Copy>(table: &[T], i: usize) -> Result<T, Error> {
    table
        .get(i)
        .copied()
        .ok_or_else(|| invariant("a table index out of range"))
}

/// Entry `i` of a dense table, to write.
fn at_mut<T>(table: &mut [T], i: usize) -> Result<&mut T, Error> {
    table
        .get_mut(i)
        .ok_or_else(|| invariant("a table index out of range"))
}

/// The first new id of copy `k` (1-based) of plan `q` in `table`.
fn copy(table: &[Vec<u32>], q: u32, k: u32) -> Result<u32, Error> {
    let copies = table
        .get(q as usize)
        .ok_or_else(|| invariant("a plan out of range"))?;
    at(copies, (k as usize).wrapping_sub(1))
}

/// The renaming of the old body's values and blocks into the unrolled one.
struct Map<'a> {
    body: &'a Body,
    plans: &'a [Plan],
    block_of: &'a [BlockId],
    owner: &'a [u32],
    local: &'a [u32],
    local_block: &'a [u32],
    phi_slot: &'a [u32],
    outside_block: &'a [u32],
    outside_inst: &'a [u32],
    copy_block: &'a [Vec<u32>],
    copy_inst: &'a [Vec<u32>],
}

impl Map<'_> {
    /// Plan `q`.
    fn plan(&self, q: u32) -> Result<&Plan, Error> {
        self.plans
            .get(q as usize)
            .ok_or_else(|| invariant("a plan out of range"))
    }

    /// The new value of `v` used in copy `ctx` (`(plan, iteration)`, 1-based), or outside every
    /// copy (`None`). A loop's value used outside it is its last copy's; one the last copy does not
    /// hold (it is the header alone) is used only where nothing reaches, and is `poison`. `subs`
    /// holds the finished plans' PHI values, `cur` the current plan's so far.
    fn value(
        &self,
        ctx: Option<(u32, u32)>,
        v: Value,
        subs: &[Vec<Value>],
        cur: &[Value],
        consts: &mut crate::ir::ConstTable,
        m: &mut Meter,
    ) -> Result<Value, Error> {
        let Value::Inst(i) = v else {
            return Ok(v);
        };
        let b = *self
            .block_of
            .get(i.index())
            .ok_or_else(|| invariant("instruction out of range"))?;
        let q = at(self.owner, b.index())?;
        if q == NONE {
            let start = self
                .body
                .blocks
                .get(b)
                .ok_or_else(|| invariant("block out of range"))?
                .start;
            return Ok(Value::Inst(InstId::new(
                at(self.outside_inst, b.index())? + (i.raw() - start.raw()),
            )));
        }
        let plan = self.plan(q)?;
        let k = match ctx {
            Some((p, k)) if p == q => k,
            _ => plan.trips,
        };
        let h = at(self.phi_slot, i.index())?;
        if h != NONE {
            let slot = (k as usize - 1) * plan.phis.len() + h as usize;
            let s = subs.get(q as usize).map_or(cur, Vec::as_slice);
            return s
                .get(slot)
                .copied()
                .ok_or_else(|| invariant("a PHI value used before it is made"));
        }
        if k == plan.trips && plan.header_only_last() && b != plan.header {
            let ty = self
                .body
                .insts
                .get(i)
                .ok_or_else(|| invariant("instruction out of range"))?
                .ty;
            let c = consts.intern(
                Constant {
                    ty,
                    value: Const::Poison,
                },
                m,
            )?;
            return Ok(Value::Const(c));
        }
        Ok(Value::Inst(InstId::new(
            copy(self.copy_inst, q, k)? + at(self.local, i.index())?,
        )))
    }

    /// The new id of block `b` reached from copy `ctx`, or from outside every copy: a loop's header
    /// from outside is its first copy's.
    fn block(&self, ctx: Option<(u32, u32)>, b: BlockId) -> Result<BlockId, Error> {
        let q = at(self.owner, b.index())?;
        if q == NONE {
            return Ok(BlockId::new(at(self.outside_block, b.index())?));
        }
        let k = match ctx {
            Some((p, k)) if p == q => k,
            _ if b == self.plan(q)?.header => 1,
            _ => return Err(invariant("a branch into a loop past its header")),
        };
        Ok(BlockId::new(
            copy(self.copy_block, q, k)? + at(self.local_block, b.index())?,
        ))
    }

    /// Instruction `inst` of block `b`, in copy `ctx` or outside every copy, in the new body.
    fn inst(
        &self,
        ctx: Option<(u32, u32)>,
        b: BlockId,
        inst: &Inst,
        subs: &[Vec<Value>],
        consts: &mut crate::ir::ConstTable,
        m: &mut Meter,
    ) -> Result<InstKind, Error> {
        // A copy's header and latch branch to the next copy, or out in the last.
        if let (Some((p, k)), true) = (ctx, inst.kind.is_terminator()) {
            let plan = self.plan(p)?;
            let last = k == plan.trips;
            let out = |dest| InstKind::Br { dest, hints: None };
            if b == plan.latch && (!last || plan.exit.0 == b) {
                return Ok(out(if last {
                    self.block(None, plan.exit.1)?
                } else {
                    self.block(Some((p, k + 1)), plan.header)?
                }));
            }
            if b == plan.header {
                if let Some(stay) = plan.stay {
                    return Ok(out(if last && plan.exit.0 == b {
                        self.block(None, plan.exit.1)?
                    } else {
                        self.block(ctx, stay)?
                    }));
                }
            }
        }
        let mut kind = inst.kind.clone();
        if let InstKind::Phi { incoming, .. } = &mut kind {
            let mut kept = Vec::with_capacity(incoming.len());
            for &(v, pred) in incoming.iter() {
                let q = at(self.owner, pred.index())?;
                let from = match (ctx, q) {
                    (_, NONE) => None,
                    (Some((p, k)), q) if p == q => Some((p, k)),
                    // An exit block's entry from a loop: only the edge the last iteration takes.
                    (None, q) => {
                        let plan = self.plan(q)?;
                        if plan.exit != (pred, b) {
                            continue;
                        }
                        Some((q, plan.trips))
                    }
                    _ => return Err(invariant("a PHI entry from another loop's copy")),
                };
                let v = self.value(from, v, subs, &[], consts, m)?;
                kept.push((v, self.block(from, pred)?));
            }
            *incoming = kept.into_boxed_slice();
            return Ok(kind);
        }
        let mut ops = Vec::new();
        kind.for_each_value(|v| ops.push(v));
        let mut mapped = Vec::with_capacity(ops.len());
        for v in ops {
            mapped.push(self.value(ctx, v, subs, &[], consts, m)?);
        }
        let mut next = mapped.into_iter();
        kind.for_each_value_mut(|v| {
            if let Some(n) = next.next() {
                *v = n;
            }
        });
        match &mut kind {
            InstKind::Br { dest, .. } => *dest = self.block(ctx, *dest)?,
            InstKind::CondBr { t, f, .. } => {
                *t = self.block(ctx, *t)?;
                *f = self.block(ctx, *f)?;
            }
            InstKind::Switch { default, cases, .. } => {
                *default = self.block(ctx, *default)?;
                for (_, d) in cases.iter_mut() {
                    *d = self.block(ctx, *d)?;
                }
            }
            _ => {}
        }
        Ok(kind)
    }
}
