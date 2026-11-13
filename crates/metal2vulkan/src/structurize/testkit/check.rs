//! The tree-level structural checker: the facts about a [`Structured`]
//! that the lowering relies on to produce valid structured SPIR-V.
//!
//! - The arena is a tree rooted at the root: every statement has exactly one parent, none is
//!   orphaned, and children precede parents.
//! - No duplication: every expected original block is in exactly one `Code`, and no other block
//!   is.
//! - P7 is complete: no `Block` or `Jump` is left; `SetPending`, `ClearPending`, and
//!   `GuardPending` appear only with a `pending` variable, which no `SetSel` or `Dispatch` uses.
//! - `Break`/`Continue` only inside a `Loop`; every selector index is in range; `Dispatch` has
//!   arms; switch cases have literals, distinct across the switch.
//! - Construct exits: no loop body and not the function body completes normally (every path ends
//!   in `Break`, `Continue`, `Return`, or `Unreachable`); a `Loop` completes only through a
//!   `Break` for it.
//!
//! Selector values and pending discipline are dynamic properties; [`super::interp::run_tree`]
//! checks them on every run.

use std::collections::BTreeSet;

use crate::air::ir::BlockId;
use crate::structurize::tree::{Stmt, StmtId, Structured};

/// Check the structural rules on `s`, whose original blocks must be exactly `originals`.
pub fn check_structured(s: &Structured, originals: &BTreeSet<BlockId>) -> Result<(), String> {
    let tree = &s.tree;
    let n = tree.len();
    let sel_ok = |index: usize, what: &str| -> Result<(), String> {
        if index >= s.selectors as usize {
            return Err(format!("{what} selector {index} out of range"));
        }
        if s.pending.is_some_and(|p| p.index() == index) {
            return Err(format!("{what} uses the pending variable"));
        }
        Ok(())
    };
    if let Some(p) = s.pending {
        if p.index() >= s.selectors as usize {
            return Err("pending selector out of range".into());
        }
    }

    // Parents before children: reachability, single parent, loop depth.
    let mut reached = vec![false; n];
    let mut loop_depth = vec![0u32; n];
    reached[tree.root().index()] = true;
    let mut blocks = BTreeSet::new();
    for (id, st) in tree.stmts().rev() {
        if !reached[id.index()] {
            return Err(format!("{id:?} is unreachable from the root"));
        }
        let depth = loop_depth[id.index()];
        let inner = depth + u32::from(matches!(st, Stmt::Loop { .. }));
        for c in st.children() {
            if c >= id {
                return Err(format!("{id:?} refers to the later {c:?}"));
            }
            if reached[c.index()] {
                return Err(format!("{c:?} has two parents"));
            }
            reached[c.index()] = true;
            loop_depth[c.index()] = inner;
        }
        match st {
            Stmt::Code(b) => {
                if !blocks.insert(*b) {
                    return Err(format!("{b:?} placed twice"));
                }
            }
            Stmt::SetSel { sel, .. } => sel_ok(sel.index(), "SetSel")?,
            Stmt::Dispatch { sel, arms } => {
                sel_ok(sel.index(), "Dispatch")?;
                if arms.is_empty() {
                    return Err(format!("{id:?}: dispatch without arms"));
                }
            }
            Stmt::Switch { cases, .. } => {
                let mut seen = BTreeSet::new();
                for (lits, _) in cases.iter() {
                    if lits.is_empty() {
                        return Err(format!("{id:?}: switch case without literals"));
                    }
                    for l in lits.iter() {
                        if !seen.insert(*l) {
                            return Err(format!("{id:?}: switch literal {l} twice"));
                        }
                    }
                }
            }
            Stmt::Break | Stmt::Continue if depth == 0 => {
                return Err(format!("{id:?}: break/continue outside a loop"));
            }
            Stmt::Block { .. } | Stmt::Jump(_) => {
                return Err(format!("{id:?}: P6 statement {st:?} left after P7"));
            }
            Stmt::SetPending(_) | Stmt::ClearPending(_) | Stmt::GuardPending(_)
                if s.pending.is_none() =>
            {
                return Err(format!("{id:?}: {st:?} without a pending variable"));
            }
            _ => {}
        }
    }
    if &blocks != originals {
        let missing: Vec<_> = originals.difference(&blocks).collect();
        let extra: Vec<_> = blocks.difference(originals).collect();
        return Err(format!(
            "original blocks differ: missing {missing:?}, unexpected {extra:?}"
        ));
    }

    // Children before parents: whether a statement can complete normally, and whether it holds a
    // `Break` for the nearest enclosing loop.
    let mut completes = vec![false; n];
    let mut breaks = vec![false; n];
    for (id, st) in tree.stmts() {
        let kids = || st.children().map(StmtId::index);
        let (c, b) = match st {
            Stmt::Seq(_) => (kids().all(|k| completes[k]), kids().any(|k| breaks[k])),
            Stmt::If { .. } | Stmt::Switch { .. } | Stmt::Dispatch { .. } => {
                (kids().any(|k| completes[k]), kids().any(|k| breaks[k]))
            }
            Stmt::Loop { body, .. } => {
                if completes[body.index()] {
                    return Err(format!("{id:?}: the loop body can fall out"));
                }
                (breaks[body.index()], false)
            }
            Stmt::Break => (false, true),
            Stmt::Continue | Stmt::Return(_) | Stmt::Unreachable => (false, false),
            Stmt::GuardPending(body) => (true, breaks[body.index()]),
            _ => (true, kids().any(|k| breaks[k])),
        };
        completes[id.index()] = c;
        breaks[id.index()] = b;
    }
    if completes[tree.root().index()] {
        return Err("the function body can fall off its end".into());
    }
    Ok(())
}
