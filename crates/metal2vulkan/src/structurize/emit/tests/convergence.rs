//! Convergence equivalence at tree level.
//!
//! 32 lanes run one structurized function. Each lane is a deterministic single-thread program
//! (the testkit scheme: executing block `b` mixes `b` into the lane's hash; conditions and
//! selectors are read from it), seeded per lane, so lanes diverge.
//!
//! - **Source, ideal tangle (C5).** Each lane runs the original CFG alone, and executions are
//!   keyed by block and loop iterations ([`tangle`](crate::structurize::testkit::tangle), shared
//!   with the SPIR-V-level check in `spirv::interp` tests).
//! - **Output, maximal reconvergence.** The lanes run the P7 tree as structured SIMT, the SPIR-V
//!   rule under `MaximallyReconverges` for the constructs P8 emits one-to-one: an `If`, `Switch`,
//!   `Dispatch`, or `GuardPending` runs each arm with the lanes that take it and rejoins every lane
//!   that falls out at its merge; a `Loop` runs each iteration with every lane that continued,
//!   and rejoins every lane that broke at its merge; lanes that return leave. Each execution of
//!   `Code(B)` runs with a set of lanes.
//!
//! The property: every lane executes the same original blocks as it does alone, ends the same
//! way, and for every block the multiset of lane sets its executions ran with equals the multiset
//! of its ideal tangles. It is checked for every block, tangled or not (tangled instructions
//! observe exactly this set).
//!
//! Irreducible inputs are skipped: P1's dispatch loops have no source-level iteration to compare
//! with, and a tangled irreducible region is refused (`IrreducibleConvergent`).

use std::collections::BTreeMap;

use crate::air::ir::{BlockId, ValueId};
use crate::budget::{Budget, Meter};
use crate::structurize::testkit::generate::{self, Rng};
use crate::structurize::testkit::hash::Hash;
use crate::structurize::testkit::interp::End;
use crate::structurize::testkit::tangle::{self, sorted, Tangles};
use crate::structurize::testkit::TestCfg;
use crate::structurize::tree::{Stmt, StmtId, Structured};
use crate::structurize::{prepare, structurize};

const LANES: usize = 32;
/// Original blocks each lane may execute.
const EVENTS: usize = 40;

/// One lane's single-thread state.
#[derive(Clone, Debug)]
struct Lane {
    h: Hash,
    trace: Vec<BlockId>,
    end: Option<End>,
    sels: Vec<Option<u32>>,
    pending: u32,
}

impl Lane {
    fn new(seed: u64, selectors: usize) -> Lane {
        Lane {
            h: Hash::seeded(seed),
            trace: Vec::new(),
            end: None,
            sels: vec![None; selectors],
            pending: 0,
        }
    }

    /// Execute `b`; false (and the lane ends) once the event limit is reached.
    fn exec(&mut self, b: BlockId) -> bool {
        if self.trace.len() == EVENTS {
            self.end = Some(End::Cutoff);
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
}

/// Lanes leaving a statement: falling out, breaking, continuing (the rest ended).
#[derive(Default)]
struct Exits {
    fall: Vec<usize>,
    brk: Vec<usize>,
    cont: Vec<usize>,
}

impl Exits {
    fn absorb(&mut self, other: Exits) {
        self.fall.extend(other.fall);
        self.brk.extend(other.brk);
        self.cont.extend(other.cont);
    }
}

/// The structured SIMT machine over a P7 tree.
struct Simt<'a> {
    s: &'a Structured,
    lanes: Vec<Lane>,
    tangles: Tangles,
    steps: usize,
}

impl Simt<'_> {
    fn stmt(&self, id: StmtId) -> &Stmt {
        self.s.tree.stmt(id).expect("statement")
    }

    /// Run `id` with `mask` (sorted lanes).
    fn run(&mut self, id: StmtId, mask: Vec<usize>) -> Exits {
        self.steps += 1;
        assert!(self.steps < 1_000_000, "SIMT run did not end");
        if mask.is_empty() {
            return Exits::default();
        }
        match self.stmt(id).clone() {
            Stmt::Seq(items) => {
                let mut out = Exits::default();
                let mut cur = mask;
                for item in items.iter() {
                    let e = self.run(*item, cur);
                    cur = e.fall;
                    out.brk.extend(e.brk);
                    out.cont.extend(e.cont);
                }
                out.fall = cur;
                out
            }
            Stmt::Code(b) => {
                let ran: Vec<usize> = mask
                    .into_iter()
                    .filter(|&l| self.lanes[l].exec(b))
                    .collect();
                if !ran.is_empty() {
                    self.tangles
                        .entry(b)
                        .or_default()
                        .push(ran.iter().copied().collect());
                }
                Exits {
                    fall: ran,
                    ..Exits::default()
                }
            }
            Stmt::SetSel { sel, value } => {
                for &l in &mask {
                    self.lanes[l].sels[sel.index()] = Some(value);
                }
                Exits {
                    fall: mask,
                    ..Exits::default()
                }
            }
            Stmt::If {
                cond,
                then,
                otherwise,
            } => {
                let (t, f): (Vec<usize>, Vec<usize>) =
                    mask.into_iter().partition(|&l| self.lanes[l].cond(cond));
                self.arms(vec![(then, t), (otherwise, f)])
            }
            Stmt::Switch {
                sel,
                default,
                cases,
            } => {
                let mut arms: Vec<(StmtId, Vec<usize>)> = cases
                    .iter()
                    .map(|(_, a)| (*a, Vec::new()))
                    .chain(std::iter::once((default, Vec::new())))
                    .collect();
                for l in mask {
                    let v = self.lanes[l].selector(sel);
                    let i = cases
                        .iter()
                        .position(|(lits, _)| lits.contains(&v))
                        .unwrap_or(cases.len());
                    arms[i].1.push(l);
                }
                self.arms(arms)
            }
            Stmt::Dispatch { sel, arms } => {
                let mut split: Vec<(StmtId, Vec<usize>)> =
                    arms.iter().map(|a| (*a, Vec::new())).collect();
                for l in mask {
                    let v = self.lanes[l].sels[sel.index()].expect("selector set") as usize;
                    split[v].1.push(l);
                }
                self.arms(split)
            }
            Stmt::GuardPending(body) => {
                let (run, skip): (Vec<usize>, Vec<usize>) =
                    mask.into_iter().partition(|&l| self.lanes[l].pending == 0);
                let mut e = self.run(body, run);
                e.fall.extend(skip);
                e.fall.sort_unstable();
                e
            }
            Stmt::Loop { body, .. } => {
                let mut active = mask;
                let mut exits = Vec::new();
                while !active.is_empty() {
                    let e = self.run(body, active);
                    assert!(e.fall.is_empty(), "control fell out of a loop body");
                    exits.extend(e.brk);
                    active = e.cont;
                    active.sort_unstable();
                }
                exits.sort_unstable();
                Exits {
                    fall: exits,
                    ..Exits::default()
                }
            }
            Stmt::Break => Exits {
                brk: mask,
                ..Exits::default()
            },
            Stmt::Continue => Exits {
                cont: mask,
                ..Exits::default()
            },
            Stmt::SetPending(label) => {
                for &l in &mask {
                    self.lanes[l].pending = label.pending_value();
                }
                Exits {
                    fall: mask,
                    ..Exits::default()
                }
            }
            Stmt::ClearPending(label) => {
                for &l in &mask {
                    if self.lanes[l].pending == label.pending_value() {
                        self.lanes[l].pending = 0;
                    }
                }
                Exits {
                    fall: mask,
                    ..Exits::default()
                }
            }
            Stmt::Return(v) => {
                for l in mask {
                    let r = v.map(|v| self.lanes[l].value(v));
                    self.lanes[l].end = Some(End::Returned(r));
                }
                Exits::default()
            }
            Stmt::Unreachable => {
                for l in mask {
                    self.lanes[l].end = Some(End::Unreachable);
                }
                Exits::default()
            }
            Stmt::Block { .. } | Stmt::Jump(_) => panic!("P6 statement in a P7 tree"),
        }
    }

    /// Run each arm with its lanes; every lane that falls out rejoins at the merge.
    fn arms(&mut self, arms: Vec<(StmtId, Vec<usize>)>) -> Exits {
        let mut out = Exits::default();
        for (arm, lanes) in arms {
            let e = self.run(arm, lanes);
            out.absorb(e);
        }
        out.fall.sort_unstable();
        out
    }
}

/// Check convergence equivalence of `g` for lane seeds `base * 32 + lane`. Returns false when
/// `g` is skipped (irreducible, or refused).
fn check(g: &TestCfg, bases: std::ops::Range<u64>) -> bool {
    let mut m = Meter::new(&Budget::default());
    let Ok(p) = prepare(g, &mut m) else {
        return false;
    };
    if !tangle::has_source_iterations(&p) {
        p.release(&mut m);
        return false;
    }
    let s = structurize(g, &mut m).expect("structurize after prepare");
    for base in bases {
        let seeds: Vec<u64> = (0..LANES as u64).map(|l| base * LANES as u64 + l).collect();
        let want = tangle::ideal(g, &p, &seeds, EVENTS).expect("a reducible graph");

        let mut simt = Simt {
            s: &s,
            lanes: seeds
                .iter()
                .map(|&seed| Lane::new(seed, s.selectors as usize))
                .collect(),
            tangles: BTreeMap::new(),
            steps: 0,
        };
        let e = simt.run(s.tree.root(), (0..LANES).collect());
        assert!(
            e.fall.is_empty() && e.brk.is_empty() && e.cont.is_empty(),
            "lanes left the function body without returning: {g:?}"
        );
        for (l, (a, b)) in want.runs.iter().zip(&simt.lanes).enumerate() {
            assert_eq!(
                (&a.trace, Some(&a.end)),
                (&b.trace, b.end.as_ref()),
                "lane {l} (seed {}) runs differently under SIMT: {g:?}",
                seeds[l]
            );
        }
        assert_eq!(
            sorted(simt.tangles),
            want.tangles,
            "tangles differ for base {base}: {g:?}\n{}",
            crate::structurize::testkit::interp::describe_tree(&s.tree)
        );
    }
    s.release(&mut m);
    p.release(&mut m);
    assert_eq!(m.ledger(), 0);
    true
}

/// Check every canonical graph of `n` blocks over `choices`; returns how many were checked.
fn exhaustive(n: u32, choices: &[Vec<u32>]) -> usize {
    let mut checked = 0;
    generate::canonical(n, choices, |g| {
        checked += usize::from(check(g, 0..1));
    });
    checked
}

#[test]
fn exhaustive_small_graphs_converge_like_the_source() {
    let mut checked = 0;
    for n in 1..=4 {
        checked += exhaustive(n, &generate::out_degree_two_choices(n));
    }
    for n in 1..=3 {
        checked += exhaustive(n, &generate::switch_choices(n));
    }
    assert!(checked > 1000, "only {checked} reducible graphs checked");
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "five-block and four-block switch graphs: runs in release (G-rel)"
)]
fn exhaustive_larger_graphs_converge_like_the_source() {
    let checked = exhaustive(5, &generate::out_degree_two_choices(5))
        + exhaustive(4, &generate::switch_choices(4));
    assert!(checked > 20_000, "only {checked} reducible graphs checked");
}

#[test]
fn generated_programs_converge_like_the_source() {
    let mut checked = 0;
    for seed in 0..400 {
        checked += usize::from(check(
            &generate::structured(seed, 2 + (seed % 4) as u32, 0),
            0..3,
        ));
        checked += usize::from(check(
            &generate::structured(seed ^ 0x6d75_6c74, 3 + (seed % 3) as u32, 0),
            0..3,
        ));
    }
    let mut rng = Rng::new(0xc0_4e46);
    for _ in 0..600 {
        let n = 2 + rng.below(30) as u32;
        checked += usize::from(check(
            &generate::dag_with_back_edges(n, rng.next_u64()),
            0..2,
        ));
    }
    for k in 1..=10 {
        checked += usize::from(check(&generate::break_tail_ladder(k), 0..4));
    }
    assert!(checked > 1000, "only {checked} reducible graphs checked");
}

#[test]
fn named_convergence_shapes() {
    // while (h) { if (p) { tail; break; } latch } exit: tail (3) runs after the loop, once with
    // every lane that broke in any iteration; the exit block (5) runs once with everyone.
    let g = TestCfg::from_successors(&[vec![1], vec![2, 5], vec![3, 4], vec![5], vec![1], vec![]]);
    assert!(check(&g, 0..8));
    // An early return before a join, and a non-tail jump through `pending`.
    let g = TestCfg::from_successors(&[vec![1, 3], vec![2, 3], vec![], vec![]]);
    assert!(check(&g, 0..8));
    let g = TestCfg::from_successors(&[vec![1, 4], vec![2, 3], vec![5], vec![4], vec![5], vec![]]);
    assert!(check(&g, 0..8));
    // A multi-level break (exit ladder).
    let g = TestCfg::from_successors(&[
        vec![1],
        vec![2, 8],
        vec![3, 6],
        vec![4, 5],
        vec![8],
        vec![2],
        vec![7],
        vec![1],
        vec![],
    ]);
    assert!(check(&g, 0..8));
}
