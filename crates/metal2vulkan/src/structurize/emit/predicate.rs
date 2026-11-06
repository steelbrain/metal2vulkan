//! P8's predicated walk: emit a structured tree so that every block holding a workgroup
//! synchronization runs in workgroup-uniform control flow, whatever the divergence of the source.
//!
//! A `bool` `Function` variable `live` holds, for each invocation, whether it executes the
//! current point of the source program. It is `true` at entry, or the emitter's entry flag where
//! it has one (`CodeEmitter::entry_live`: the invocations that run the body at all, the others
//! walking it as if they had returned at entry). Every construct that holds a synchronization
//! (**hot**) is walked by every invocation of the workgroup, in order:
//!
//! - an `If`, `Switch`, `Dispatch`, or `GuardPending` runs its arms one after another, each with
//!   `live` set to the invocations that take it, and merges `live` as the union of the arms' ends;
//! - a `Loop` iterates while any invocation of the workgroup is live in it (a workgroup vote,
//!   [`CodeEmitter::vote`], so its exit is uniform); `Break` and `Continue` clear `live` and set
//!   the loop's per-invocation `brk`/`cont` flags, the next iteration's `live` is `cont`, and after
//!   the loop `live` is `brk`;
//! - `Return` and `Unreachable` clear `live` (a returned invocation never becomes live again);
//! - a block runs through [`CodeEmitter::block_live`]: its instructions only where `live` holds,
//!   its synchronizations everywhere with `live` as the lane's active flag (every lane is active
//!   in a block `uniform::Uniformity` proves uniform, where every invocation is live or none is).
//!
//! Every condition and selector is combined with `live` by `OpSelect` (never `OpLogicalAnd`), so
//! a value a non-live invocation never computed does not reach a branch.
//!
//! A construct that holds no synchronization runs as it is: guarded by `if live` where some
//! present invocation may not be live, and, when it can leave itself (`Return`, or a `Break` or
//! `Continue` of an enclosing loop), with real divergent branches and the same flags instead of
//! real exits, since a real exit would skip a later synchronization.
//!
//! **Uniform positions.** The walk tracks statically whether every invocation of the workgroup
//! executes the current point (`all`) and whether every invocation executing it is live
//! (`live`). A position is uniform (both hold) at entry without an entry flag (with one, the
//! entry is not: some invocation may not be live), in the arms of a real branch taken from
//! one, in the body of a loop that iterates together, and after a construct entered from one
//! that no invocation leaves early (every invocation live before it is live after it). It never
//! becomes uniform at a block because `uniform::Uniformity` proves the block uniform: that means
//! every invocation executes it or none does, and the walk also emits the blocks that no
//! invocation executes (an arm nobody takes, the rest of an iteration everybody left). In a
//! uniform position a construct on a uniform condition is a real branch, a loop whose first block
//! is uniform iterates together (its exit test reads `cont`, which is uniform, rather than
//! voting), and `Return`, `Break`, and `Continue` are real exits (after setting the loop's flag).
//!
//! The walk opens no more constructs than the plain walk per tree level; its recursion follows
//! the construct nesting, which is capped at half of `Limit::ConstructNesting` for it.

use std::cell::Cell;

use super::{emit_loop, list, CodeEmitter, Cx};
use crate::air::ir::BlockId;
use crate::budget::CONSTRUCT_NESTING;
use crate::error::{Error, Limit};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopCtx, LoopFn, LoopSpec};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, ControlRule};
use crate::spirv::ids::{Value, VarId};
use crate::spirv::ir::{BinOp, ConstBits, Constant, MemAccess, Type, UnOp};

use super::super::tree::{Stmt, StmtId, StructTree};

/// What the walk knows statically about the invocations at a point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct St {
    /// Every invocation of the workgroup executes this point (a synchronization may be emitted).
    all: bool,
    /// Every invocation executing this point is live (`live` need not be read).
    live: bool,
}

const UNIFORM: St = St {
    all: true,
    live: true,
};

/// The per-statement facts of the tree, by statement index.
pub(super) struct Facts {
    /// The statement holds a block with a synchronization.
    hot: Vec<bool>,
    /// The statement holds a `Break` or `Continue` of a loop around it.
    leaves: Vec<bool>,
    /// The statement holds a `Return` or `Unreachable`.
    ends: Vec<bool>,
}

impl Facts {
    fn get(v: &[bool], id: StmtId) -> bool {
        v.get(id.index()).copied().unwrap_or(true)
    }

    fn hot(&self, id: StmtId) -> bool {
        Self::get(&self.hot, id)
    }

    /// The statement can leave itself other than by falling through.
    fn escapes(&self, id: StmtId) -> bool {
        Self::get(&self.leaves, id) || Self::get(&self.ends, id)
    }

    fn ends(&self, id: StmtId) -> bool {
        Self::get(&self.ends, id)
    }

    /// Heap bytes (charged by [`facts`]).
    fn bytes(&self) -> usize {
        self.hot.len() + self.leaves.len() + self.ends.len()
    }
}

/// The facts of `tree` (children precede parents in the arena), and its construct nesting depth,
/// refused above half of `Limit::ConstructNesting`.
fn facts<E: CodeEmitter>(cx: &mut Cx<'_, E>) -> Result<Facts, Error> {
    let tree = cx.tree;
    let n = tree.len();
    let bytes = n.saturating_mul(3 + 4);
    cx.meter.charge(bytes)?;
    let r = (|| {
        let mut f = Facts {
            hot: vec![false; n],
            leaves: vec![false; n],
            ends: vec![false; n],
        };
        for (id, st) in tree.stmts() {
            cx.meter.tick(1)?;
            let any = |v: &[bool]| st.children().any(|c| Facts::get(v, c));
            let hot = match st {
                Stmt::Code(b) => cx.c.syncs(*b),
                _ => any(&f.hot),
            };
            let leaves = match st {
                Stmt::Break | Stmt::Continue => true,
                Stmt::Loop { .. } => false,
                _ => any(&f.leaves),
            };
            let ends = match st {
                Stmt::Return(_) | Stmt::Unreachable => true,
                _ => any(&f.ends),
            };
            let i = id.index();
            for (v, x) in [
                (&mut f.hot, hot),
                (&mut f.leaves, leaves),
                (&mut f.ends, ends),
            ] {
                if let Some(slot) = v.get_mut(i) {
                    *slot = x;
                }
            }
        }
        // Construct nesting, parents first.
        let mut depth = vec![0u32; n];
        for (id, st) in tree.stmts().rev() {
            cx.meter.tick(1)?;
            let d = depth.get(id.index()).copied().unwrap_or(0);
            let inner = d.saturating_add(u32::from(super::is_construct(st)));
            if u64::from(inner) > CONSTRUCT_NESTING / 2 {
                return Err(Error::LimitExceeded(Limit::ConstructNesting));
            }
            for c in st.children() {
                if let Some(x) = depth.get_mut(c.index()) {
                    *x = inner;
                }
            }
        }
        Ok(f)
    })();
    match r {
        Ok(f) => {
            // The depth table is gone; the facts stay charged until the walk ends.
            cx.meter.release(bytes - f.bytes());
            Ok(f)
        }
        Err(e) => {
            cx.meter.release(bytes);
            Err(e)
        }
    }
}

/// The flags and exits of the innermost walked loop.
#[derive(Clone, Copy, Debug)]
struct Flags<'l> {
    brk: VarId,
    cont: VarId,
    /// The loop's real exits, taken by a `Break` or `Continue` in a uniform position.
    lc: &'l LoopCtx,
}

/// What the walk carries besides the emission state.
struct W<'f> {
    facts: &'f Facts,
    live: VarId,
    bool_ty: crate::spirv::ids::TypeId,
}

impl W<'_> {
    fn konst(&self, fb: &mut FunctionBuilder<'_>, v: bool) -> BResult<Value> {
        let c = fb.module().constant(Constant::Scalar {
            ty: self.bool_ty,
            bits: ConstBits(u64::from(v)),
        })?;
        fb.module().const_value(c)
    }

    fn load(&self, fb: &mut FunctionBuilder<'_>, v: VarId) -> BResult<Value> {
        let p = fb.var_ptr(v)?;
        fb.load(p, MemAccess::default())
    }

    fn store(&self, fb: &mut FunctionBuilder<'_>, v: VarId, x: Value) -> BResult<()> {
        let p = fb.var_ptr(v)?;
        fb.store(p, x, MemAccess::default())
    }

    /// The live flag at a point in state `st`: the constant `true` where every present
    /// invocation is live.
    fn live(&self, fb: &mut FunctionBuilder<'_>, st: St) -> BResult<Value> {
        if st.live {
            self.konst(fb, true)
        } else {
            self.load(fb, self.live)
        }
    }

    /// `l ? x : false`: `x` where `l` holds, never reading a value `l` does not vouch for.
    fn and(&self, fb: &mut FunctionBuilder<'_>, l: Value, x: Value) -> BResult<Value> {
        let f = self.konst(fb, false)?;
        fb.select(l, x, f)
    }

    /// Store `x` to variable `var` where `live` holds (every present invocation when `st.live`).
    fn store_live(
        &self,
        fb: &mut FunctionBuilder<'_>,
        st: St,
        ptr: Value,
        x: Value,
    ) -> BResult<()> {
        if st.live {
            return fb.store(ptr, x, MemAccess::default());
        }
        let l = self.load(fb, self.live)?;
        let old = fb.load(ptr, MemAccess::default())?;
        let y = fb.select(l, x, old)?;
        fb.store(ptr, y, MemAccess::default())
    }
}

/// Walk the tree of `cx` predicated from its root; on success no block is open.
pub(super) fn walk_root<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
) -> Result<(), Error> {
    let facts = facts(cx)?;
    let r = (|| -> BResult<()> {
        let bool_ty = fb.module().ty(Type::Bool)?;
        let live = fb.local_var(bool_ty, None)?;
        let w = W {
            facts: &facts,
            live,
            bool_ty,
        };
        // Every invocation is live at entry, or those the emitter's entry flag says run the body.
        let (first, st) = match cx.c.entry_live() {
            Some(l) => (
                l,
                St {
                    all: true,
                    live: false,
                },
            ),
            None => (w.konst(fb, true)?, UNIFORM),
        };
        w.store(fb, live, first)?;
        let end = walk(cx, fb, &w, cx.tree.root(), st, None)?;
        if end.is_some() && fb.current_block().is_some() {
            // Every invocation that returned in a divergent position ends here.
            cx.c.on_return(None, fb)?;
        }
        cx.poll(fb)
    })();
    cx.meter.release(facts.bytes());
    r.map_err(Error::from)
}

/// The first original block statement `id` runs, through leading sequence items.
fn first_code(tree: &StructTree, mut id: StmtId) -> Option<BlockId> {
    loop {
        match tree.stmt(id).ok()? {
            Stmt::Seq(items) => id = *items.first()?,
            Stmt::Code(b) => return Some(*b),
            _ => return None,
        }
    }
}

/// The state after joining paths that end in `a` and `b` (`None`: no path continues).
fn meet(a: Option<St>, b: Option<St>) -> Option<St> {
    match (a, b) {
        (Some(x), Some(y)) => Some(St {
            all: x.all && y.all,
            live: x.live && y.live,
        }),
        (x, None) => x,
        (None, y) => y,
    }
}

/// Walk statement `id` (sequences with an explicit stack) from state `st`; `flags` are the
/// innermost walked loop's. Returns the state after it, or `None` when no block is open (every
/// invocation left by a real return or unreachable).
fn walk<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    w: &W<'_>,
    id: StmtId,
    st: St,
    flags: Option<Flags<'_>>,
) -> BResult<Option<St>> {
    let root = [id];
    let mut frames: Vec<&[StmtId]> = vec![&root];
    let tree = cx.tree;
    let mut st = Some(st);
    while let Some(frame) = frames.last_mut() {
        let Some((&id, rest)) = frame.split_first() else {
            frames.pop();
            continue;
        };
        *frame = rest;
        let next = super::frame_next(&frames);
        let s = tree
            .stmt(id)
            .map_err(|_| BuildError::Control(ControlRule::NoCurrentBlock))?;
        let Some(now) = st else {
            // Every path ended: only statements that emit nothing may follow.
            match s {
                Stmt::Unreachable => continue,
                Stmt::Seq(items) if items.is_empty() => continue,
                _ => return Err(BuildError::Control(ControlRule::NoCurrentBlock)),
            }
        };
        st = match s {
            Stmt::Seq(items) => {
                frames.push(items);
                Some(now)
            }
            Stmt::Dispatch { arms, .. } if arms.len() == 1 => {
                // A one-arm dispatch always takes its arm.
                frames.push(arms);
                Some(now)
            }
            _ => {
                let r = stmt(cx, fb, w, id, s, next, now, flags)?;
                if !matches!(s, Stmt::Code(_)) {
                    // Values bound inside the statement's constructs do not dominate what follows.
                    cx.c.forget();
                }
                r
            }
        };
    }
    Ok(st)
}

/// One statement that is not a sequence.
#[inline(never)]
fn stmt<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    w: &W<'_>,
    id: StmtId,
    s: &Stmt,
    next: Option<StmtId>,
    st: St,
    flags: Option<Flags<'_>>,
) -> BResult<Option<St>> {
    let (hot, escapes) = (w.facts.hot(id), w.facts.escapes(id));
    match s {
        Stmt::Code(b) => code(cx, fb, w, *b, st).map(Some),
        Stmt::SetSel { sel, value } => {
            let ptr = cx.sel_ptr(fb, *sel)?;
            let k = cx.konst(fb, *value)?;
            w.store_live(fb, st, ptr, k)?;
            Ok(Some(st))
        }
        Stmt::SetPending(label) => {
            let ptr = cx.pending_ptr(fb)?;
            let k = cx.konst(fb, label.pending_value())?;
            w.store_live(fb, st, ptr, k)?;
            Ok(Some(st))
        }
        Stmt::ClearPending(label) => {
            let (eq, p) = cx.pending_is(fb, label.pending_value())?;
            let zero = cx.konst(fb, 0)?;
            let cleared = fb.select(eq, zero, p)?;
            let ptr = cx.pending_ptr(fb)?;
            w.store_live(fb, st, ptr, cleared)?;
            Ok(Some(st))
        }
        Stmt::Break | Stmt::Continue => {
            let f = flags.ok_or(BuildError::Control(ControlRule::NotInnermostLoop))?;
            let brk = matches!(s, Stmt::Break);
            let var = if brk { f.brk } else { f.cont };
            let ptr = fb.var_ptr(var)?;
            let t = w.konst(fb, true)?;
            w.store_live(fb, st, ptr, t)?;
            if st == UNIFORM {
                // Every invocation leaves together: a real exit, so nothing after it in this
                // iteration runs (a later block the analysis proves uniform is not reached).
                if brk {
                    f.lc.break_(fb)?;
                } else {
                    f.lc.continue_(fb)?;
                }
                return Ok(None);
            }
            let no = w.konst(fb, false)?;
            w.store(fb, w.live, no)?;
            Ok(Some(St { live: false, ..st }))
        }
        Stmt::Return(v) if st == UNIFORM => {
            cx.c.on_return(*v, fb)?;
            match fb.current_block() {
                Some(open) => Err(BuildError::Control(ControlRule::UnterminatedBlock(open))),
                None => Ok(None),
            }
        }
        Stmt::Unreachable if st.live => {
            fb.unreachable()?;
            Ok(None)
        }
        Stmt::Return(None) | Stmt::Unreachable => {
            let no = w.konst(fb, false)?;
            w.store(fb, w.live, no)?;
            Ok(Some(St { live: false, ..st }))
        }
        // A kernel returns no value; a value returned in a divergent position has nowhere to go.
        Stmt::Return(Some(_)) => Err(BuildError::Control(ControlRule::ReturnType)),
        _ if !hot && !escapes => {
            // Runs as it is, where live. A loop without an exit leaves no block open where every
            // present invocation runs it.
            plain(cx, fb, w, id, s, next, st)?;
            Ok(fb.current_block().map(|_| st))
        }
        Stmt::If {
            cond,
            then,
            otherwise,
        } => {
            let arms = [*then, *otherwise];
            if st == UNIFORM && cx.c.uniform_cond(*cond) {
                let c = cx.c.cond(*cond, fb)?;
                return real_if(cx, fb, w, c, arms, flags);
            }
            let c = cx.c.cond(*cond, fb)?;
            let nc = fb.unary(UnOp::LogicalNot, c)?;
            branches(cx, fb, w, id, st, &[(c, *then), (nc, *otherwise)], flags)
        }
        Stmt::GuardPending(body) => {
            let (none, _) = cx.pending_is(fb, 0)?;
            let some = fb.unary(UnOp::LogicalNot, none)?;
            branches_opt(
                cx,
                fb,
                w,
                id,
                st,
                &[(none, Some(*body)), (some, None)],
                flags,
            )
        }
        Stmt::Switch {
            sel,
            default,
            cases,
        } => {
            let x = cx.c.selector(*sel, fb)?;
            if st == UNIFORM && cx.c.uniform_cond(*sel) {
                return real_switch(cx, fb, w, x, cases, *default, flags);
            }
            let lits: Vec<(Vec<u64>, StmtId)> =
                cases.iter().map(|(l, a)| (l.to_vec(), *a)).collect();
            let arms = matches(cx, fb, x, &lits, *default)?;
            branches(cx, fb, w, id, st, &arms, flags)
        }
        Stmt::Dispatch { sel, arms } => {
            let ptr = cx.sel_ptr(fb, *sel)?;
            let x = fb.load(ptr, MemAccess::default())?;
            let Some((default, cases)) = arms.split_last() else {
                return Err(BuildError::Control(ControlRule::CaseLiterals));
            };
            let lits: Vec<(Vec<u64>, StmtId)> = cases
                .iter()
                .enumerate()
                .map(|(i, a)| (vec![i as u64], *a))
                .collect();
            let arms = matches(cx, fb, x, &lits, *default)?;
            branches(cx, fb, w, id, st, &arms, flags)
        }
        Stmt::Loop { body } => predicated_loop(cx, fb, w, id, *body, next, st),
        Stmt::Seq(_) | Stmt::Block { .. } | Stmt::Jump(_) => {
            Err(BuildError::Control(ControlRule::NoCurrentBlock))
        }
    }
}

/// Block `b` from state `st`.
fn code<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    w: &W<'_>,
    b: BlockId,
    st: St,
) -> BResult<St> {
    let syncs = cx.c.syncs(b);
    if syncs && !st.all {
        // A synchronization outside workgroup-uniform control flow: the walk never gets here.
        return Err(BuildError::Control(ControlRule::NoCurrentBlock));
    }
    if st.live {
        cx.c.block(b, fb)?;
    } else {
        let l = w.load(fb, w.live)?;
        cx.c.block_live(b, Some(l), fb)?;
    }
    cx.poll(fb)?;
    Ok(st)
}

/// A statement without synchronization or escape, run as the plain walk emits it, guarded by
/// `live` where some present invocation may not be live.
fn plain<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    w: &W<'_>,
    id: StmtId,
    s: &Stmt,
    next: Option<StmtId>,
    st: St,
) -> BResult<()> {
    if let Stmt::Code(b) = s {
        return code(cx, fb, w, *b, st).map(|_| ());
    }
    let run = move |fb: &mut FunctionBuilder<'_>, cx: &mut Cx<'_, E>| -> BResult<()> {
        match cx.tree.stmt(id) {
            // A loop takes the exit selector of the dispatch after it (P5).
            Ok(Stmt::Loop { body }) => emit_loop(cx, fb, *body, next),
            _ => list(cx, fb, id, None),
        }
    };
    if st.live {
        return run(fb, cx);
    }
    let l = w.load(fb, w.live)?;
    fb.if_(
        cx,
        l,
        Box::new(move |fb, cx| {
            run(fb, cx)?;
            Ok(if fb.current_block().is_some() {
                ArmEnd::FallThrough(Vec::new())
            } else {
                ArmEnd::Exits
            })
        }),
        None,
    )?;
    Ok(())
}

/// A walked arm of a real construct, recording its end state in `end`.
fn arm<'a, 'e, E: CodeEmitter>(
    w: &'a W<'a>,
    id: StmtId,
    st: St,
    flags: Option<Flags<'a>>,
    end: &'a Cell<Option<St>>,
) -> ArmFn<'a, Cx<'e, E>> {
    Box::new(move |fb, cx| {
        let e = walk(cx, fb, w, id, st, flags)?;
        end.set(e);
        Ok(if fb.current_block().is_some() {
            ArmEnd::FallThrough(Vec::new())
        } else {
            ArmEnd::Exits
        })
    })
}

/// A real `if` on a workgroup-uniform condition in a uniform position: its arms are uniform.
fn real_if<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    w: &W<'_>,
    c: Value,
    [then, otherwise]: [StmtId; 2],
    flags: Option<Flags<'_>>,
) -> BResult<Option<St>> {
    let (te, oe) = (super::is_empty(cx, then)?, super::is_empty(cx, otherwise)?);
    if te && oe {
        return Ok(Some(UNIFORM));
    }
    let (t_end, o_end) = (Cell::new(None), Cell::new(Some(UNIFORM)));
    let f = if oe {
        None
    } else {
        Some(arm(w, otherwise, UNIFORM, flags, &o_end))
    };
    fb.if_(cx, c, arm(w, then, UNIFORM, flags, &t_end), f)?;
    Ok(fb.current_block().and(meet(t_end.get(), o_end.get())))
}

/// A real `switch` on a workgroup-uniform selector in a uniform position.
fn real_switch<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    w: &W<'_>,
    x: Value,
    cases: &[(Box<[u64]>, StmtId)],
    default: StmtId,
    flags: Option<Flags<'_>>,
) -> BResult<Option<St>> {
    let ends: Vec<Cell<Option<St>>> = (0..=cases.len()).map(|_| Cell::new(None)).collect();
    let mut arms = Vec::with_capacity(cases.len());
    for ((lits, a), end) in cases.iter().zip(&ends) {
        let lits = lits
            .iter()
            .map(|&l| u32::try_from(l))
            .collect::<Result<Vec<u32>, _>>()
            .map_err(|_| BuildError::Control(ControlRule::CaseLiterals))?;
        arms.push((lits, arm(w, *a, UNIFORM, flags, end)));
    }
    let d_end = ends
        .last()
        .ok_or(BuildError::Control(ControlRule::CaseLiterals))?;
    fb.switch_(cx, x, arms, arm(w, default, UNIFORM, flags, d_end))?;
    let end = ends.iter().fold(None, |acc, e| meet(acc, e.get()));
    Ok(fb.current_block().and(end))
}

/// For a selector `x` and the case literals `cases` (each with its arm) and `default`, each
/// arm with the `bool` that selects it.
fn matches<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    x: Value,
    cases: &[(Vec<u64>, StmtId)],
    default: StmtId,
) -> BResult<Vec<(Value, StmtId)>> {
    let mut out = Vec::with_capacity(cases.len() + 1);
    let mut any: Option<Value> = None;
    for (lits, a) in cases {
        let mut hit: Option<Value> = None;
        for &l in lits {
            let l = u32::try_from(l).map_err(|_| BuildError::Control(ControlRule::CaseLiterals))?;
            let k = cx.konst(fb, l)?;
            let eq = fb.binary(BinOp::IEqual, x, k)?;
            hit = Some(match hit {
                Some(h) => fb.binary(BinOp::LogicalOr, h, eq)?,
                None => eq,
            });
        }
        let hit = hit.ok_or(BuildError::Control(ControlRule::CaseLiterals))?;
        any = Some(match any {
            Some(h) => fb.binary(BinOp::LogicalOr, h, hit)?,
            None => hit,
        });
        out.push((hit, *a));
    }
    let none = match any {
        Some(h) => fb.unary(UnOp::LogicalNot, h)?,
        None => {
            let b = fb.module().ty(Type::Bool)?;
            let c = fb.module().constant(Constant::Scalar {
                ty: b,
                bits: ConstBits(1),
            })?;
            fb.module().const_value(c)?
        }
    };
    out.push((none, default));
    Ok(out)
}

fn branches<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    w: &W<'_>,
    id: StmtId,
    st: St,
    arms: &[(Value, StmtId)],
    flags: Option<Flags<'_>>,
) -> BResult<Option<St>> {
    let arms: Vec<(Value, Option<StmtId>)> = arms.iter().map(|(c, a)| (*c, Some(*a))).collect();
    branches_opt(cx, fb, w, id, st, &arms, flags)
}

/// A branch of statement `id` over `arms` (each taken where its condition holds, the conditions
/// disjoint and exhaustive for live invocations; an arm `None` runs nothing), from state `st`
/// in a position that is not uniform or on a divergent condition.
///
/// A statement with a synchronization runs every arm, one after another, with `live` set to the
/// invocations that take it; `live` is then the union of the arms' ends. One without runs each
/// arm as a real branch for the live invocations that take it.
fn branches_opt<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    w: &W<'_>,
    id: StmtId,
    st: St,
    arms: &[(Value, Option<StmtId>)],
    flags: Option<Flags<'_>>,
) -> BResult<Option<St>> {
    let l0 = w.live(fb, st)?;
    // Every arm's condition for the invocations live here, computed before any arm runs.
    let mut on = Vec::with_capacity(arms.len());
    for (c, a) in arms {
        let x = if st.live { *c } else { w.and(fb, l0, *c)? };
        let empty = match a {
            Some(a) => super::is_empty(cx, *a)?,
            None => true,
        };
        on.push((x, a.filter(|_| !empty)));
    }
    let after = St {
        all: st.all,
        live: st.live && !w.facts.escapes(id),
    };
    if !w.facts.hot(id) {
        // Real divergent branches: every invocation in an arm is live.
        let local = St {
            all: false,
            live: true,
        };
        for (x, a) in on {
            let Some(a) = a else { continue };
            let end = Cell::new(None);
            fb.if_(cx, x, arm(w, a, local, flags, &end), None)?;
        }
        return Ok(Some(after));
    }
    // Sequential arms, every invocation walking each.
    let arm_st = St {
        all: st.all,
        live: false,
    };
    let mut ends: Vec<Value> = Vec::with_capacity(on.len());
    for (x, a) in on {
        match a {
            Some(a) => {
                w.store(fb, w.live, x)?;
                if walk(cx, fb, w, a, arm_st, flags)?.is_none() {
                    return Ok(None);
                }
                ends.push(w.load(fb, w.live)?);
            }
            // An empty arm ends as it starts.
            None => ends.push(x),
        }
    }
    let mut u = w.konst(fb, false)?;
    for e in ends {
        u = fb.binary(BinOp::LogicalOr, u, e)?;
    }
    w.store(fb, w.live, u)?;
    Ok(Some(after))
}

/// A loop from state `st`, walked with per-invocation `brk`/`cont` flags: while any invocation
/// continues, `live` is `cont`. A loop with a synchronization is iterated by every invocation
/// until none is live in it (a workgroup vote), or, when it starts in a uniform position at a
/// uniform block, while the uniform `live` holds; one without lets each invocation leave when it
/// is no longer live.
fn predicated_loop<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    w: &W<'_>,
    id: StmtId,
    body: StmtId,
    next: Option<StmtId>,
    st: St,
) -> BResult<Option<St>> {
    let hot = w.facts.hot(id);
    let together =
        st == UNIFORM && first_code(cx.tree, body).is_some_and(|b| cx.c.uniform_block(b));
    let (vote, body_st) = match (hot, together) {
        (true, true) => (false, UNIFORM),
        (true, false) => (
            true,
            St {
                all: st.all,
                live: false,
            },
        ),
        (false, _) => (
            false,
            St {
                all: false,
                live: true,
            },
        ),
    };
    let exit_sel = match next.map(|n| cx.tree.stmt(n)) {
        Some(Ok(Stmt::Dispatch { sel, arms })) if arms.len() > 1 => Some(
            cx.vars
                .selector(*sel)
                .ok_or(BuildError::Control(ControlRule::ExitSelector))?,
        ),
        _ => None,
    };
    let (brk, cont) = (
        fb.local_var(w.bool_ty, None)?,
        fb.local_var(w.bool_ty, None)?,
    );
    // Every invocation runs the loop's first block together (`Uniformity`), so each reaches the
    // exit test with the same `cont` (the record for checks; no word changes).
    #[cfg(test)]
    if hot && together {
        fb.note_uniform_var(cont);
    }
    let l0 = w.live(fb, st)?;
    w.store(fb, cont, l0)?;
    let no = w.konst(fb, false)?;
    w.store(fb, brk, no)?;
    let lf: LoopFn<'_, Cx<'_, E>> = Box::new(move |fb, cx, lc| {
        let f = Flags { brk, cont, lc };
        let c = w.load(fb, f.cont)?;
        w.store(fb, w.live, c)?;
        let no = w.konst(fb, false)?;
        w.store(fb, f.cont, no)?;
        let go = if vote { cx.c.vote(c, fb)? } else { c };
        let stop = fb.unary(UnOp::LogicalNot, go)?;
        fb.if_(
            cx,
            stop,
            Box::new(|fb, _cx| {
                lc.break_(fb)?;
                Ok(ArmEnd::Exits)
            }),
            None,
        )?;
        walk(cx, fb, w, body, body_st, Some(f))?;
        Ok(())
    });
    fb.loop_(
        cx,
        LoopSpec {
            exit_sel,
            dont_unroll: false,
        },
        lf,
    )?;
    if fb.current_block().is_none() {
        return Ok(None);
    }
    let b = w.load(fb, brk)?;
    w.store(fb, w.live, b)?;
    Ok(Some(St {
        all: st.all,
        live: st.live && !w.facts.ends(id),
    }))
}
