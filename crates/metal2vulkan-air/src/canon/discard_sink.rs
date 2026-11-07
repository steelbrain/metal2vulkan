//! A simdgroup sum before a discard, where Metal's compiler moves it past the discard.
//!
//! Metal leaves out of a simdgroup operation before a `discard_fragment` the quads the discard
//! drops whole when its compiler moves the operation past the discard. Measured on Apple M4
//! (macOS 27.0.1, Metal 382.5.3, AGXCompilerCore 360.34.5) and Apple M3 Max (macOS 26.7, Metal
//! 373.7, AGXCompilerCore 353.14) alike, at MSL 2.2 and 2.3, from the AIR `metalfe-32023.921.6`
//! makes (and the owned fixture `fragment_discard_hoist`): a `simd_sum`
//! before an `if (c) discard_fragment();` counts only the lanes that survive the discard when
//! nothing between them reads its result or has a side effect, the guard's block holds the discard
//! alone, and the paths rejoin where the sum's block leads, whatever the condition reads (a
//! constant or device buffer, one the function writes later, a texture sample or read, a loop's
//! result, a call's). It counts the dropped quads too when the condition reads the sum, or when a
//! device store or an atomic lies between. Where arithmetic reads the sum between (a call's,
//! inlined), Metal counted them in the one shape measured, and where the guarded discard is nested
//! in a further branch, Metal counted them in one shape and moved the sum in another: neither is
//! known in general.
//!
//! `expand` moves such a sum to the start of the block where the discard's paths rejoin, so the
//! fragment lowering meets it after the discard, where its whole-quad exit gives Metal's lanes.
//! `placement` states, for any non-quad simdgroup operation before a discard, whether Metal's
//! answer is known: moved, counted (a measured reason keeps it ahead of the discard), or unknown,
//! which the lowering records as `PrecisionNote::QuadsCountedBeforeDiscard`.

use std::collections::{BTreeMap, BTreeSet};

use crate::budget::Meter;
use crate::error::Error;
use crate::graph::{Cfg, DomTree, PostDomTree};
use crate::intrinsics::{AirIntrinsic, Fragment, GroupOp, GroupScope, Sync};
use crate::ir::{BlockId, Body, Callee, InstId, InstKind, Value};

use super::rewrite::{apply, scoped, BodyEdit};

/// What Metal does with a non-quad simdgroup operation before a discard (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// No discard follows on the operation's post-dominator chain or in a region it opens.
    NoDiscard,
    /// Metal's compiler moves it past the discard: placed at the start of this block.
    Moved(BlockId),
    /// Metal counts the dropped quads too, for a measured reason: the discard's condition reads
    /// the operation, or a side effect lies between them.
    Counted,
    /// No measurement covers the shape.
    Unknown,
}

/// The structural facts `placement` reads, computed once per body.
///
/// The dominator and post-dominator trees own their answers, so the control-flow graph they were
/// built from is released as soon as they exist: nothing here reads it again.
pub struct Facts {
    dom: DomTree<BlockId>,
    pdom: PostDomTree<BlockId>,
    /// The blocks holding a `discard_fragment` call.
    discards: BTreeSet<BlockId>,
}

impl Facts {
    /// The facts of `body`, or `None` when no block of it calls `discard_fragment`. The placement
    /// answers where a discard lies relative to an operation, so a body with no discard needs no
    /// graph: the blocks are scanned first and nothing is built when the scan finds none.
    pub fn of(body: &Body, m: &mut Meter) -> Result<Option<Facts>, Error> {
        let discards: BTreeSet<BlockId> = body
            .blocks
            .iter()
            .filter(|&(b, _)| body.block_insts(b).any(|(_, i)| is_discard(&i.kind)))
            .map(|(b, _)| b)
            .collect();
        if discards.is_empty() {
            return Ok(None);
        }
        let (dom, pdom) = {
            let cfg = Cfg::build(body, m)?;
            let built = trees(&cfg, m);
            m.release(cfg.ledger_bytes());
            built?
        };
        Ok(Some(Facts {
            dom,
            pdom,
            discards,
        }))
    }

    /// Return the facts' ledger charge.
    pub fn release(self, m: &mut Meter) {
        m.release(self.dom.ledger_bytes() + self.pdom.ledger_bytes());
    }
}

/// The dominator and post-dominator trees of `cfg`. On error nothing stays charged for them.
fn trees(
    cfg: &Cfg<BlockId>,
    m: &mut Meter,
) -> Result<(DomTree<BlockId>, PostDomTree<BlockId>), Error> {
    let dom = DomTree::compute(cfg, m)?;
    match PostDomTree::compute(cfg, m) {
        Ok(pdom) => Ok((dom, pdom)),
        Err(e) => {
            m.release(dom.ledger_bytes());
            Err(e)
        }
    }
}

/// Whether `kind` calls `discard_fragment`.
fn is_discard(kind: &InstKind) -> bool {
    matches!(kind, InstKind::Call(c) if matches!(
        c.callee,
        Callee::Air { op: AirIntrinsic::Fragment(Fragment::Discard), .. }
    ))
}

/// Whether `kind` is a simdgroup sum, the one operation measured. The other reductions, the
/// ballots, shuffles and scans are unmeasured: before a discard each is `Placement::Unknown`.
fn is_sum(kind: &InstKind) -> bool {
    matches!(kind, InstKind::Call(c) if matches!(
        c.callee,
        Callee::Air { op: AirIntrinsic::Group(g), .. }
            if g.scope == GroupScope::Simdgroup && g.op == GroupOp::Sum
    ))
}

/// Whether `kind` is a non-quad simdgroup operation: one whose lanes a discard may change.
pub fn is_non_quad(kind: &InstKind) -> bool {
    match kind {
        InstKind::Call(c) => match &c.callee {
            Callee::Air { op, .. } => match op {
                AirIntrinsic::Group(g) => g.scope == GroupScope::Simdgroup,
                AirIntrinsic::SimdMatrix(_) | AirIntrinsic::Sync(Sync::Barrier { .. }) => true,
                AirIntrinsic::AsyncCopy(_) => true,
                _ => false,
            },
            Callee::Llvm { op, .. } => matches!(op, crate::intrinsics::LlvmIntrinsic::Agx(_)),
            _ => true,
        },
        _ => false,
    }
}

/// Whether `kind` has an effect a move past it could reorder: a store, an atomic, a fence, a
/// call other than an `air.`/`llvm.` operation without one, or a texture write or atomic.
fn has_side_effect(kind: &InstKind) -> bool {
    match kind {
        InstKind::Store { .. }
        | InstKind::AtomicRmw { .. }
        | InstKind::CmpXchg { .. }
        | InstKind::Fence(_) => true,
        InstKind::Call(c) => match &c.callee {
            Callee::Air { op, .. } => match op {
                AirIntrinsic::Atomic(_) | AirIntrinsic::AsyncCopy(_) => true,
                AirIntrinsic::Coherent(k) => k.kind == crate::intrinsics::AccessKind::Store,
                AirIntrinsic::Texture(t) => matches!(
                    t.op,
                    crate::intrinsics::TexOp::Atomic(_)
                        | crate::intrinsics::TexOp::Write
                        | crate::intrinsics::TexOp::WriteImageblockSlice
                ),
                AirIntrinsic::Sync(_) => true,
                _ => false,
            },
            Callee::Llvm { op, .. } => matches!(
                op,
                crate::intrinsics::LlvmIntrinsic::Memset { .. }
                    | crate::intrinsics::LlvmIntrinsic::Memcpy { .. }
                    | crate::intrinsics::LlvmIntrinsic::Memmove { .. }
            ),
            Callee::Direct(_)
            | Callee::Indirect { .. }
            | Callee::UnknownIntrinsic(_)
            | Callee::Unbound { .. } => true,
        },
        _ => false,
    }
}

/// Whether `kind` reads value `v`.
fn uses(kind: &InstKind, v: Value) -> bool {
    let mut found = false;
    kind.for_each_value(|x| found |= x == v);
    found
}

/// The blocks strictly between `from` and `to`: reachable from `from`'s successors without
/// passing `to`, `from` and `to` excluded, and whether such a path returns to `from` (`from` lies
/// on a cycle that avoids `to`).
fn between(body: &Body, from: BlockId, to: BlockId) -> (Vec<BlockId>, bool) {
    let mut seen = BTreeSet::new();
    let mut work: Vec<BlockId> = body.successors(from).collect();
    let mut out = Vec::new();
    let mut returns = false;
    while let Some(b) = work.pop() {
        if b == from {
            returns = true;
            continue;
        }
        if b == to || !seen.insert(b) {
            continue;
        }
        work.extend(body.successors(b));
        out.push(b);
    }
    (out, returns)
}

/// Whether a path from block `b` returns to it without passing `avoid`.
fn returns_avoiding(body: &Body, b: BlockId, avoid: BlockId) -> bool {
    b != avoid && between(body, b, avoid).1
}

/// What lies between an operation and the point a walk reached.
#[derive(Default)]
struct Between {
    /// Something reads the operation's result.
    used: bool,
    /// Something has a side effect.
    effect: bool,
    /// Another non-quad simdgroup operation.
    other: bool,
    /// A discard outside the measured shapes (an unconditional one, say).
    discard: bool,
}

impl Between {
    /// Account for an instruction of kind `kind`, the operation's result being `result`.
    fn look(&mut self, kind: &InstKind, result: Value) {
        self.used |= uses(kind, result);
        self.effect |= has_side_effect(kind);
        self.other |= is_non_quad(kind);
        self.discard |= is_discard(kind);
    }

    /// Account for every instruction of block `b`.
    fn look_at(&mut self, body: &Body, b: BlockId, result: Value) {
        for (_, i) in body.block_insts(b) {
            self.look(&i.kind, result);
        }
    }
}

/// The block holding instruction `id`.
fn block_of(body: &Body, id: InstId) -> Option<BlockId> {
    body.blocks
        .iter()
        .find(|(_, k)| k.start.raw() <= id.raw() && id.raw() < k.end.raw())
        .map(|(b, _)| b)
}

/// Where Metal places the non-quad simdgroup operation `op` (in block `home`) relative to the
/// first discard after it (the module docs).
///
/// The walk follows `op`'s post-dominator chain, gathering what lies between, to the first
/// region holding a discard: the discard guarded by the branch that opens the region
/// (`if (c) discard_fragment();`, the guard's block holding the discard alone), or that shape
/// nested in a further branch. Any other discard on the way is `Unknown`.
pub fn placement(f: &Facts, body: &Body, op: InstId, home: BlockId) -> Placement {
    let Some(inst) = body.insts.get(op) else {
        return Placement::Unknown;
    };
    let result = Value::Inst(op);
    let mut seen = Between::default();
    for (_, i) in body
        .block_insts(home)
        .skip_while(|&(id, _)| id != op)
        .skip(1)
    {
        seen.look(&i.kind, result);
    }
    let mut p = home;
    loop {
        if seen.discard {
            return Placement::Unknown;
        }
        let Some(next) = f.pdom.ipdom(p) else {
            return Placement::NoDiscard;
        };
        let (region, _) = between(body, p, next);
        let discards: Vec<BlockId> = region
            .iter()
            .copied()
            .filter(|b| f.discards.contains(b))
            .collect();
        if discards.is_empty() {
            // No discard before `next`, whose instructions lie between the operation and the
            // first discard too.
            for &b in &region {
                seen.look_at(body, b, result);
            }
            seen.look_at(body, next, result);
            p = next;
            continue;
        }
        for &b in region.iter().filter(|b| !discards.contains(b)) {
            seen.look_at(body, b, result);
        }
        if !is_sum(&inst.kind) || seen.discard {
            return Placement::Unknown;
        }
        let [s] = discards[..] else {
            return Placement::Unknown;
        };
        // The branch guarding the discard: `p`'s own (the measured shape), or one nested in a
        // further branch of the region.
        let guard = if guarded_discard(f, body, p, next) == Some(s) {
            Some((p, false))
        } else {
            region
                .iter()
                .copied()
                .find(|&q| {
                    f.pdom
                        .ipdom(q)
                        .is_some_and(|merge| guarded_discard(f, body, q, merge) == Some(s))
                })
                .map(|q| (q, true))
        };
        let Some((guard, nested)) = guard else {
            return Placement::Unknown;
        };
        // The discard's condition reads the operation: Metal computes it ahead of the discard,
        // in every lane. A side effect between them keeps it there too. A nested discard was
        // measured both ways.
        if condition_uses(body, guard, result) || seen.effect {
            return Placement::Counted;
        }
        if nested || seen.used || seen.other {
            return Placement::Unknown;
        }
        // The lanes that run the operation are the lanes that reach `next`, but for those the
        // discard drops, and each runs both as often: `next` post-dominates `home`, `home`
        // dominates `next`, and no cycle holds one without the other (a loop wholly between them
        // is passed through). Its operands are defined at `next` on every path, and every use
        // follows.
        let mut defined = true;
        inst.kind.for_each_value(|v| {
            if let Value::Inst(d) = v {
                defined &= block_of(body, d).is_some_and(|b| f.dom.dominates(b, next));
            }
        });
        if !f.dom.dominates(home, next)
            || returns_avoiding(body, home, next)
            || returns_avoiding(body, next, home)
            || !defined
            || !uses_follow(f, body, op, next)
        {
            return Placement::Unknown;
        }
        return Placement::Moved(next);
    }
}

/// The block holding the discard that block `p` guards, when `p` ends in `if (c) discard;`: a
/// conditional branch to `next` and to a block that holds the discard alone and branches only to
/// `next`.
fn guarded_discard(f: &Facts, body: &Body, p: BlockId, next: BlockId) -> Option<BlockId> {
    let (_, last) = body.block_insts(p).last()?;
    let InstKind::CondBr { t, f: e, .. } = last.kind else {
        return None;
    };
    let s = match (t == next, e == next) {
        (true, false) => e,
        (false, true) => t,
        _ => return None,
    };
    let only_next = body.successors(s).all(|x| x == next) && body.successors(s).count() == 1;
    let mut insts = body.block_insts(s);
    let alone = insts.next().is_some_and(|(_, i)| is_discard(&i.kind))
        && insts.next().is_some_and(|(_, i)| i.kind.is_terminator())
        && insts.next().is_none();
    (f.discards.contains(&s) && only_next && alone).then_some(s)
}

/// Whether the branch condition ending block `p` reads `v`, directly or through the pure
/// instructions computing it.
fn condition_uses(body: &Body, p: BlockId, v: Value) -> bool {
    let Some((_, last)) = body.block_insts(p).last() else {
        return false;
    };
    let InstKind::CondBr { c, .. } = last.kind else {
        return false;
    };
    let mut work = vec![c];
    let mut seen = BTreeSet::new();
    while let Some(x) = work.pop() {
        if x == v {
            return true;
        }
        let Value::Inst(i) = x else {
            continue;
        };
        if !seen.insert(i) {
            continue;
        }
        if let Some(n) = body.insts.get(i) {
            n.kind.for_each_value(|o| work.push(o));
        }
    }
    false
}

/// Whether every use of `op` lies where the start of block `to` dominates it: in `to` past its
/// PHIs, or in a block `to` dominates (a PHI's use at the end of its incoming block).
fn uses_follow(f: &Facts, body: &Body, op: InstId, to: BlockId) -> bool {
    let v = Value::Inst(op);
    body.blocks.iter().all(|(b, _)| {
        body.block_insts(b).all(|(_, i)| match &i.kind {
            InstKind::Phi { incoming, .. } => incoming
                .iter()
                .all(|&(x, from)| x != v || f.dom.dominates(to, from)),
            k => !uses(k, v) || f.dom.dominates(to, b),
        })
    })
}

/// Move each simdgroup reduction of a fragment entry's `body` that Metal's compiler moves past a
/// discard (the module docs), repeating while one moves (past a later discard), at most once per
/// discard.
pub(super) fn expand(body: Body, m: &mut Meter) -> Result<Body, Error> {
    let mut body = body;
    let mut rounds = 0usize;
    loop {
        let Some(facts) = Facts::of(&body, m)? else {
            return Ok(body);
        };
        if rounds > facts.discards.len() {
            facts.release(m);
            return Ok(body);
        }
        let mut moves: BTreeMap<InstId, BlockId> = BTreeMap::new();
        for (b, _) in body.blocks.iter() {
            for (id, i) in body.block_insts(b) {
                m.tick(1)?;
                if !is_sum(&i.kind) {
                    continue;
                }
                if let Placement::Moved(to) = placement(&facts, &body, id, b) {
                    moves.insert(id, to);
                }
            }
        }
        facts.release(m);
        if moves.is_empty() {
            return Ok(body);
        }
        let next = scoped(m, |m, held| {
            let mut edit = BodyEdit::keep_all(&body, m, held)?;
            edit.moved = moves;
            apply(&body, &edit, &|c| c, m)
        })?;
        body.release(m);
        body = next;
        rounds += 1;
    }
}
