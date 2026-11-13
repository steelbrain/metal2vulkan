//! P7: the forward jumps with one `pending` variable.
//!
//! SPIR-V has no forward multi-level label, so every `Block(Y, S)` / `Jump(Y)` pair becomes one of:
//!
//! - **Tail jump.** After the jump, every enclosing statement up to `Block(Y)` has nothing left:
//!   the jump is the last statement of each enclosing `Seq`, possibly inside `If`/`Switch`/
//!   `Dispatch` arms (whose merges fall through), never inside a `Loop`. It is dropped; falling
//!   out of the arms reaches `code(Y)`.
//! - **Non-tail jump.** `SetPending(Y)`, then fall out. Every `Seq` position after a statement
//!   that may finish with a label pending becomes `GuardPending(rest)`
//!   (`if pending == NONE { rest }`), and `Block(Y, S)` becomes `Seq[S, ClearPending(Y)]`.
//!   `code(Y)` itself is guarded exactly when a label other than `Y` may still be pending after
//!   the clear, which falls out of the same rule one `Seq` up.
//!
//! "May finish with a label pending" is decided with block depths: a statement at `Block`
//! nesting depth `D` may finish with a label pending exactly when it contains a non-tail jump to
//! a `Block` at depth `< D` (a `Block` inside the statement is cleared before it finishes). So
//! each statement needs only the minimum target depth of the non-tail jumps below it, one
//! bottom-up pass. Jumps never cross a `Loop` (a loop body leaves only through `Break`), so a
//! `Loop` contributes nothing, and `pending` is always `NONE` at `Break`, `Continue`, `Return`,
//! and loop entry: between a `SetPending(Y)` and its `ClearPending(Y)` every statement is skipped
//! by a guard.
//!
//! Three linear passes over the P6 arena (parents before children to hand down the tail label and
//! depth; children before parents for the minimum; children before parents to rebuild), and the
//! output adds at most one `GuardPending` and one `Seq` per `Seq` position and one `ClearPending`
//! per label.

use super::cfg::{at, at_mut, invariant, with_scratch, Scratch};
use super::loops::NONE;
use super::tree::{Stmt, StmtId, StructTree};
use crate::budget::Meter;
use crate::error::Error;

/// P7's result.
#[derive(Debug)]
pub struct Pending {
    pub tree: StructTree,
    /// Whether some jump is not a tail jump (the entry needs a `pending` variable).
    pub uses_pending: bool,
}

/// P7 over P6's tree, which uses labels `0..labels`. `src` is left as it was; on error nothing
/// this pass charged stays on the ledger.
pub(super) fn place(src: &StructTree, labels: u32, meter: &mut Meter) -> Result<Pending, Error> {
    let capacity = src.len().saturating_add(src.len() / 4).saturating_add(16);
    let mut out = StructTree::with_capacity(meter, capacity)?;
    let lowered = with_scratch(meter, |meter, scratch| {
        lower(src, labels as usize, &mut out, meter, scratch)
    });
    match lowered {
        Ok(uses_pending) => Ok(Pending {
            tree: out,
            uses_pending,
        }),
        Err(e) => {
            meter.release(out.ledger_bytes());
            Err(e)
        }
    }
}

/// Per-statement facts handed from parent to child.
struct Down {
    reached: Vec<bool>,
    /// The label that falling out of the statement reaches with nothing in between (`NONE`).
    tail: Vec<u32>,
    /// The number of enclosing `Block`s.
    depth: Vec<u32>,
}

impl Down {
    fn visit(&mut self, parent: StmtId, c: StmtId, tail: u32, depth: u32) -> Result<(), Error> {
        if c >= parent {
            return Err(invariant(format!(
                "{parent:?} refers to the later statement {c:?}"
            )));
        }
        let reached = at_mut(&mut self.reached, c.index())?;
        if *reached {
            return Err(invariant(format!("{c:?} has two parents")));
        }
        *reached = true;
        *at_mut(&mut self.tail, c.index())? = tail;
        *at_mut(&mut self.depth, c.index())? = depth;
        Ok(())
    }
}

fn lower(
    src: &StructTree,
    labels: usize,
    out: &mut StructTree,
    meter: &mut Meter,
    scratch: &mut Scratch,
) -> Result<bool, Error> {
    let n = src.len();
    let mut down = Down {
        reached: scratch.filled(meter, n, false)?,
        tail: scratch.filled(meter, n, NONE)?,
        depth: scratch.filled(meter, n, 0u32)?,
    };
    let mut label_depth = scratch.filled(meter, labels, NONE)?;
    let mut used = scratch.filled(meter, labels, false)?;
    let mut non_tail = scratch.filled(meter, n, false)?;
    *at_mut(&mut down.reached, src.root().index())? = true;

    // 1. Parents before children: tail labels, depths, which jumps are tail jumps.
    for (id, stmt) in src.stmts().rev() {
        meter.tick(1)?;
        if !*at(&down.reached, id.index())? {
            continue;
        }
        let (tail, depth) = (*at(&down.tail, id.index())?, *at(&down.depth, id.index())?);
        match stmt {
            Stmt::Seq(items) => {
                for (i, &c) in items.iter().enumerate() {
                    let last = i + 1 == items.len();
                    down.visit(id, c, if last { tail } else { NONE }, depth)?;
                }
            }
            Stmt::If { .. } | Stmt::Switch { .. } | Stmt::Dispatch { .. } => {
                for c in stmt.children() {
                    down.visit(id, c, tail, depth)?;
                }
            }
            Stmt::Loop { body, .. } => down.visit(id, *body, NONE, depth)?,
            Stmt::Block { label, body } => {
                let slot = at_mut(&mut label_depth, label.index())?;
                if *slot != NONE {
                    return Err(invariant(format!("{label:?} labels two blocks")));
                }
                *slot = depth;
                let inner = depth
                    .checked_add(1)
                    .ok_or_else(|| invariant("block nesting exceeds u32"))?;
                down.visit(id, *body, label.index() as u32, inner)?;
            }
            Stmt::Jump(label) => {
                if tail != label.index() as u32 {
                    *at_mut(&mut non_tail, id.index())? = true;
                    *at_mut(&mut used, label.index())? = true;
                }
            }
            Stmt::SetPending(_) | Stmt::ClearPending(_) | Stmt::GuardPending(_) => {
                return Err(invariant("P7 input already holds pending statements"));
            }
            Stmt::Code(_)
            | Stmt::SetSel { .. }
            | Stmt::Break
            | Stmt::Continue
            | Stmt::Return(_)
            | Stmt::Unreachable => {}
        }
    }

    // 2. Children before parents: the smallest block depth a non-tail jump below targets.
    let mut min_target = scratch.filled(meter, n, NONE)?;
    for (id, stmt) in src.stmts() {
        meter.tick(1)?;
        if !*at(&down.reached, id.index())? {
            continue;
        }
        let m = match stmt {
            Stmt::Jump(label) if *at(&non_tail, id.index())? => {
                let d = *at(&label_depth, label.index())?;
                if d == NONE {
                    return Err(invariant(format!("jump to {label:?} outside its block")));
                }
                d
            }
            Stmt::Loop { .. } => NONE,
            _ => {
                let mut m = NONE;
                for c in stmt.children() {
                    m = m.min(*at(&min_target, c.index())?);
                }
                m
            }
        };
        *at_mut(&mut min_target, id.index())? = m;
    }

    // 3. Children before parents: rebuild. `None` is an empty statement not yet materialized.
    let mut new_of: Vec<Option<StmtId>> = scratch.filled(meter, n, None)?;
    let mut acc: Vec<StmtId> = Vec::new();
    for (id, stmt) in src.stmts() {
        meter.tick(1)?;
        if !*at(&down.reached, id.index())? {
            continue;
        }
        let depth = *at(&down.depth, id.index())?;
        let new = match stmt {
            Stmt::Seq(items) => {
                // Built back to front: `acc` holds the rest of the sequence, reversed.
                acc.clear();
                for &c in items.iter().rev() {
                    meter.tick(1)?;
                    let Some(nc) = *at(&new_of, c.index())? else {
                        continue;
                    };
                    if *at(&min_target, c.index())? < depth && !acc.is_empty() {
                        acc.reverse();
                        let rest = seq_of(out, meter, &acc)?;
                        let guard = out.push(meter, Stmt::GuardPending(rest))?;
                        acc.clear();
                        scratch.push(meter, &mut acc, guard)?;
                    }
                    scratch.push(meter, &mut acc, nc)?;
                }
                acc.reverse();
                match acc.as_slice() {
                    [] => None,
                    [only] => Some(*only),
                    items => Some(out.seq(meter, items)?),
                }
            }
            Stmt::Block { label, body } => {
                let body = *at(&new_of, body.index())?;
                if *at(&used, label.index())? {
                    let clear = out.push(meter, Stmt::ClearPending(*label))?;
                    Some(match body {
                        Some(b) => out.seq(meter, &[b, clear])?,
                        None => clear,
                    })
                } else {
                    body
                }
            }
            Stmt::Jump(label) => {
                if *at(&non_tail, id.index())? {
                    Some(out.push(meter, Stmt::SetPending(*label))?)
                } else {
                    None
                }
            }
            Stmt::If {
                cond,
                then,
                otherwise,
            } => {
                let then = materialize(out, meter, &new_of, *then)?;
                let otherwise = materialize(out, meter, &new_of, *otherwise)?;
                Some(out.push(
                    meter,
                    Stmt::If {
                        cond: *cond,
                        then,
                        otherwise,
                    },
                )?)
            }
            Stmt::Switch {
                sel,
                default,
                cases,
            } => {
                let default = materialize(out, meter, &new_of, *default)?;
                acc.clear();
                for (_, arm) in cases.iter() {
                    let arm = materialize(out, meter, &new_of, *arm)?;
                    scratch.push(meter, &mut acc, arm)?;
                }
                Some(out.switch(meter, *sel, default, cases, &acc)?)
            }
            Stmt::Dispatch { sel, arms } => {
                acc.clear();
                for &arm in arms.iter() {
                    let arm = materialize(out, meter, &new_of, arm)?;
                    scratch.push(meter, &mut acc, arm)?;
                }
                Some(out.dispatch(meter, *sel, &acc)?)
            }
            Stmt::Loop { body, unrolling } => {
                let body = materialize(out, meter, &new_of, *body)?;
                let unrolling = *unrolling;
                Some(out.push(meter, Stmt::Loop { body, unrolling })?)
            }
            leaf @ (Stmt::Code(_)
            | Stmt::SetSel { .. }
            | Stmt::Break
            | Stmt::Continue
            | Stmt::Return(_)
            | Stmt::Unreachable) => Some(out.push(meter, leaf.clone())?),
            Stmt::SetPending(_) | Stmt::ClearPending(_) | Stmt::GuardPending(_) => {
                return Err(invariant("P7 input already holds pending statements"));
            }
        };
        *at_mut(&mut new_of, id.index())? = new;
    }
    let root = materialize(out, meter, &new_of, src.root())?;
    out.set_root(root)?;
    Ok(used.iter().any(|&u| u))
}

/// `items` as one statement (a single item stands for itself).
fn seq_of(out: &mut StructTree, meter: &mut Meter, items: &[StmtId]) -> Result<StmtId, Error> {
    match items {
        [only] => Ok(*only),
        items => out.seq(meter, items),
    }
}

/// The rebuilt statement for `old`, or a fresh empty `Seq` where it lowered to nothing.
fn materialize(
    out: &mut StructTree,
    meter: &mut Meter,
    new_of: &[Option<StmtId>],
    old: StmtId,
) -> Result<StmtId, Error> {
    match *at(new_of, old.index())? {
        Some(s) => Ok(s),
        None => out.seq(meter, &[]),
    }
}
