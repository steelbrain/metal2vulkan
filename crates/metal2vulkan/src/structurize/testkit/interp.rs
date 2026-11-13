//! A CFG interpreter for trace equivalence.
//!
//! The state is one `u64`, seeded per run ([`Hash`]). Executing original block `b` appends `b`
//! to the trace and mixes `b` into the state. A condition, switch selector, or return value named
//! by value `v` is a hash of the current state and `v`. Synthetic nodes of a working graph change
//! only selector variables, never the state, so equal traces mean equal decisions.

use super::hash::Hash;
use super::view::{TestCfg, TestKind};
use crate::air::ir::{BlockId, ValueId};
use crate::structurize::cfg::{Content, Node, StructCfg, Term};
use crate::structurize::tree::{Label, Stmt, StmtId, StructTree};

/// How a run ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum End {
    /// Returned, with the return value's hash (`None` for `ret void`).
    Returned(Option<u64>),
    /// Reached `unreachable`.
    Unreachable,
    /// Executed the event limit of original blocks without ending.
    Cutoff,
}

/// The first original blocks a run executed, and how it ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    pub trace: Vec<BlockId>,
    pub end: End,
}

struct State {
    h: Hash,
    trace: Vec<BlockId>,
    limit: usize,
}

impl State {
    fn new(seed: u64, limit: usize) -> State {
        State {
            h: Hash::seeded(seed),
            trace: Vec::new(),
            limit,
        }
    }

    /// Execute original block `b`; false once the event limit is reached.
    fn exec(&mut self, b: BlockId) -> bool {
        if self.trace.len() == self.limit {
            return false;
        }
        self.trace.push(b);
        self.h.exec(b);
        true
    }

    fn value(&self, v: ValueId) -> u64 {
        self.h.value(v)
    }

    fn cond(&self, v: ValueId) -> bool {
        self.h.cond(v)
    }

    fn selector(&self, v: ValueId) -> u64 {
        self.h.selector(v)
    }

    fn finish(self, end: End) -> Run {
        Run {
            trace: self.trace,
            end,
        }
    }
}

/// Run the original graph from `seed` for at most `events` blocks.
pub fn run_test_cfg(g: &TestCfg, seed: u64, events: usize) -> Run {
    let mut s = State::new(seed, events);
    let mut b = BlockId::new(0);
    loop {
        if !s.exec(b) {
            return s.finish(End::Cutoff);
        }
        let block = &g.blocks[b.raw() as usize];
        let v = ValueId::new(b.raw());
        b = match &block.kind {
            TestKind::Br => block.succ[0],
            TestKind::CondBr => block.succ[if s.cond(v) { 0 } else { 1 }],
            TestKind::Switch(literals) => {
                let sel = s.selector(v);
                match literals.iter().position(|&l| l == sel) {
                    Some(i) => block.succ[i + 1],
                    None => block.succ[0],
                }
            }
            TestKind::Return => return s.finish(End::Returned(None)),
            TestKind::ReturnValue => {
                let r = s.value(v);
                return s.finish(End::Returned(Some(r)));
            }
            TestKind::Unreachable => return s.finish(End::Unreachable),
        };
    }
}

/// Run a working graph from `seed` for at most `events` original blocks. Fails when a selector is
/// read before it is written or out of range, or when more than `steps` nodes run without the
/// run ending or reaching the event limit (synthetic nodes looping on their own).
pub fn run_struct_cfg(
    sc: &StructCfg,
    seed: u64,
    events: usize,
    steps: usize,
) -> Result<Run, String> {
    let mut s = State::new(seed, events);
    let mut sels: Vec<Option<u32>> = vec![None; sc.selector_count()];
    let mut n = sc.entry();
    for _ in 0..steps {
        let block = sc.block(n).map_err(|e| e.to_string())?;
        match block.content {
            Content::Code(b) => {
                if !s.exec(b) {
                    return Ok(s.finish(End::Cutoff));
                }
            }
            Content::SetSel { sel, value } => sels[sel.index()] = Some(value),
            Content::Empty => {}
        }
        n = match &block.term {
            Term::Br(t) => *t,
            Term::CondBr { cond, t, f } => {
                if s.cond(*cond) {
                    *t
                } else {
                    *f
                }
            }
            Term::Switch {
                sel,
                default,
                cases,
            } => {
                let v = s.selector(*sel);
                cases
                    .iter()
                    .find(|(lits, _)| lits.contains(&v))
                    .map_or(*default, |(_, t)| *t)
            }
            Term::Dispatch { sel, targets } => {
                let v = sels[sel.index()]
                    .ok_or_else(|| format!("{n:?} reads selector {sel:?} before it is set"))?;
                *targets
                    .get(v as usize)
                    .ok_or_else(|| format!("{n:?}: selector value {v} out of range"))?
            }
            Term::Return(None) => return Ok(s.finish(End::Returned(None))),
            Term::Return(Some(v)) => {
                let r = s.value(*v);
                return Ok(s.finish(End::Returned(Some(r))));
            }
            Term::Unreachable => return Ok(s.finish(End::Unreachable)),
        };
    }
    Err(format!(
        "{steps} steps without ending after {} original blocks (stuck at {n:?})",
        s.trace.len()
    ))
}

/// Check that `sc` executes the same original blocks as `g` for `seeds` seeds, `events` blocks
/// each. The step limit allows every node of `sc` between two original blocks.
pub fn check_trace_equivalence(
    g: &TestCfg,
    sc: &StructCfg,
    seeds: std::ops::Range<u64>,
    events: usize,
) -> Result<(), String> {
    let steps = (events + 1).saturating_mul(sc.len() + 1);
    for seed in seeds {
        let expected = run_test_cfg(g, seed, events);
        let actual = run_struct_cfg(sc, seed, events, steps)?;
        if actual != expected {
            return Err(format!(
                "seed {seed}: expected {expected:?}, got {actual:?}"
            ));
        }
    }
    Ok(())
}

/// Nodes a run of `sc` visits, for diagnostics.
pub fn describe(sc: &StructCfg) -> String {
    sc.blocks()
        .map(|(n, b): (Node, _)| format!("{n:?}: {:?} -> {:?}", b.content, b.term))
        .collect::<Vec<_>>()
        .join("\n")
}

/// A frame of the tree interpreter's control stack.
enum Frame<'t> {
    /// The rest of a `Seq`.
    Seq(&'t [StmtId]),
    /// A running `Loop`; `Continue` restarts `body`.
    Loop(StmtId),
    /// A running `Block`; `Jump(label)` resumes after it.
    Block(Label),
}

/// Run a structured tree (P6 or P7 output) from `seed` for at most `events` original blocks.
///
/// Fails when a selector is read before it is written or out of range, when `Break`/`Continue`
/// run outside a loop or a `Jump` outside its block, when control falls out of a loop body or off
/// the end of the function, when `Break`, `Continue`, `Return`, or a loop entry runs while a
/// forward jump is pending (the P7 discipline), or when `steps` statements run without the run
/// ending or reaching the event limit. `pending` starts at `NONE`, as P8 initializes it.
pub fn run_tree(
    tree: &StructTree,
    selectors: usize,
    seed: u64,
    events: usize,
    steps: usize,
) -> Result<Run, String> {
    let stmt = |id: StmtId| tree.stmt(id).map_err(|e| e.to_string());
    let mut s = State::new(seed, events);
    let mut sels: Vec<Option<u32>> = vec![None; selectors];
    let mut pending = 0u32;
    let mut frames: Vec<Frame<'_>> = Vec::new();
    let mut next = Some(tree.root());
    for _ in 0..steps {
        let current = next;
        next = None;
        let Some(id) = current else {
            // Fall out of the innermost open statement.
            match frames.last_mut() {
                None => return Err("control fell off the end of the function".into()),
                Some(Frame::Seq(rest)) => match rest.split_first() {
                    Some((first, tail)) => {
                        next = Some(*first);
                        *rest = tail;
                    }
                    None => {
                        frames.pop();
                    }
                },
                Some(Frame::Loop(_)) => return Err("control fell out of a loop body".into()),
                Some(Frame::Block(_)) => {
                    frames.pop();
                }
            }
            continue;
        };
        let quiet = |what: &str, pending: u32| {
            if pending == 0 {
                Ok(())
            } else {
                Err(format!("{what} while jump {pending} is pending"))
            }
        };
        match stmt(id)? {
            Stmt::Seq(items) => frames.push(Frame::Seq(items)),
            Stmt::Code(b) => {
                if !s.exec(*b) {
                    return Ok(s.finish(End::Cutoff));
                }
            }
            Stmt::SetSel { sel, value } => {
                *sels
                    .get_mut(sel.index())
                    .ok_or_else(|| format!("{sel:?} out of range"))? = Some(*value);
            }
            Stmt::If {
                cond,
                then,
                otherwise,
            } => next = Some(if s.cond(*cond) { *then } else { *otherwise }),
            Stmt::Switch {
                sel,
                default,
                cases,
            } => {
                let v = s.selector(*sel);
                next = Some(
                    cases
                        .iter()
                        .find(|(lits, _)| lits.contains(&v))
                        .map_or(*default, |(_, t)| *t),
                );
            }
            Stmt::Dispatch { sel, arms } => {
                let v = sels
                    .get(sel.index())
                    .copied()
                    .flatten()
                    .ok_or_else(|| format!("{id:?} reads {sel:?} before it is set"))?;
                next = Some(
                    *arms
                        .get(v as usize)
                        .ok_or_else(|| format!("{id:?}: {sel:?} = {v} out of range"))?,
                );
            }
            Stmt::Loop { body, .. } => {
                quiet("loop entry", pending)?;
                frames.push(Frame::Loop(*body));
                next = Some(*body);
            }
            Stmt::Break | Stmt::Continue => {
                quiet("break/continue", pending)?;
                loop {
                    match frames.last() {
                        None => return Err(format!("{id:?}: break/continue outside a loop")),
                        Some(Frame::Loop(body)) => {
                            if matches!(stmt(id)?, Stmt::Continue) {
                                next = Some(*body);
                            } else {
                                frames.pop();
                            }
                            break;
                        }
                        Some(_) => {
                            frames.pop();
                        }
                    }
                }
            }
            Stmt::Block { label, body } => {
                frames.push(Frame::Block(*label));
                next = Some(*body);
            }
            Stmt::Jump(label) => loop {
                match frames.pop() {
                    None | Some(Frame::Loop(_)) => {
                        return Err(format!("{id:?}: jump to {label:?} outside its block"))
                    }
                    Some(Frame::Block(l)) if l == *label => break,
                    Some(_) => {}
                }
            },
            Stmt::SetPending(label) => pending = label.pending_value(),
            Stmt::ClearPending(label) => {
                if pending == label.pending_value() {
                    pending = 0;
                }
            }
            Stmt::GuardPending(body) => {
                if pending == 0 {
                    next = Some(*body);
                }
            }
            Stmt::Return(v) => {
                quiet("return", pending)?;
                let r = v.map(|v| s.value(v));
                return Ok(s.finish(End::Returned(r)));
            }
            Stmt::Unreachable => return Ok(s.finish(End::Unreachable)),
        }
    }
    Err(format!(
        "{steps} statements without ending after {} original blocks",
        s.trace.len()
    ))
}

/// Check that the tree executes the same original blocks as `g` for `seeds` seeds, `events`
/// blocks each. The step limit allows every statement of the tree between two original blocks.
pub fn check_tree_equivalence(
    g: &TestCfg,
    tree: &StructTree,
    selectors: usize,
    seeds: std::ops::Range<u64>,
    events: usize,
) -> Result<(), String> {
    let steps = (events + 1).saturating_mul(2 * tree.len() + 2);
    for seed in seeds {
        let expected = run_test_cfg(g, seed, events);
        let actual = run_tree(tree, selectors, seed, events, steps)?;
        if actual != expected {
            return Err(format!(
                "seed {seed}: expected {expected:?}, got {actual:?}"
            ));
        }
    }
    Ok(())
}

/// Every statement of a tree, one per line, for diagnostics.
pub fn describe_tree(tree: &StructTree) -> String {
    let mut out = format!("root {:?}\n", tree.root());
    for (id, stmt) in tree.stmts() {
        out.push_str(&format!("{id:?}: {stmt:?}\n"));
    }
    out
}
