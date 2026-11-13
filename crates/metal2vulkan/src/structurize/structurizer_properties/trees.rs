//! P4–P7 properties: tree-level trace equivalence after P6 and after P7, the structural
//! checker, the output size bound with its measured constant, named shapes (pending use, exit
//! ladders, switches), the demotion plan, refusals, ledger balance, determinism, and linear work.

use std::collections::BTreeSet;
use std::ops::Range;

use super::{check_ok, meter, EVENTS};
use crate::air::ir::{BlockId, ValueId};
use crate::budget::{Budget, Meter};
use crate::error::{Error, Limit};
use crate::graph::Cfg;
use crate::spirv::ids::{DemoteTy, ScalarTy};
use crate::structurize::normalize::Normalizer;
use crate::structurize::testkit::check::check_structured;
use crate::structurize::testkit::generate::{self, Rng};
use crate::structurize::testkit::interp::{check_tree_equivalence, describe_tree};
use crate::structurize::testkit::TestCfg;
use crate::structurize::tree::{Stmt, Structured};
use crate::structurize::{pending, prepare, regions, structure_prepared, structurize, Prepared};

/// The reachable original blocks of `g`.
fn originals(g: &TestCfg) -> BTreeSet<BlockId> {
    let mut m = meter();
    let cfg = Cfg::build(g, &mut m).unwrap();
    (0..g.blocks.len() as u32)
        .map(BlockId::new)
        .filter(|&b| cfg.is_reachable(b))
        .collect()
}

/// Run and check P5–P7 stage by stage on a prepared `g`, then the whole of P4–P7. The ledger
/// ends where it started.
pub(super) fn check_stages(g: &TestCfg, prepared: &Prepared, seeds: Range<u64>, m: &mut Meter) {
    let start = m.ledger();
    let (sc, loops) = (&prepared.cfg, &prepared.loops);
    let mut norm = Normalizer::new(loops, sc.len(), sc.selector_count() as u32, m).unwrap();
    let p6 = regions::structure(sc, loops, &[], &mut norm, m)
        .unwrap_or_else(|e| panic!("P6 refused {e:?} for {g:?}"));
    let selectors = norm.selectors() as usize;
    check_tree_equivalence(g, &p6.tree, selectors, seeds.clone(), EVENTS).unwrap_or_else(|e| {
        panic!(
            "P6 changed the traces: {e}\n{g:?}\n{}",
            describe_tree(&p6.tree)
        )
    });
    let p7 = pending::place(&p6.tree, p6.labels, m).unwrap();
    check_tree_equivalence(g, &p7.tree, selectors, seeds.clone(), EVENTS).unwrap_or_else(|e| {
        panic!(
            "P7 changed the traces: {e}\n{g:?}\nP6:\n{}P7:\n{}",
            describe_tree(&p6.tree),
            describe_tree(&p7.tree)
        )
    });

    let s = structure_prepared(g, prepared, m).unwrap();
    assert_eq!(
        s.tree, p7.tree,
        "structurize differs from the staged passes"
    );
    check_structured(&s, &originals(g))
        .unwrap_or_else(|e| panic!("{e}\n{g:?}\n{}", describe_tree(&s.tree)));
    assert_eq!(s.pending.is_some(), p7.uses_pending);
    assert!(s.tree.lowered_blocks() <= prepared.output_cap);
    s.release(m);
    m.release(p7.tree.ledger_bytes());
    m.release(p6.tree.ledger_bytes());
    norm.release(m);
    assert_eq!(m.ledger(), start, "P4–P7 left ledger bytes behind");
}

/// Structurize `succ` (checking every stage) and return the result.
fn structured(succ: &[Vec<u32>]) -> Structured {
    let g = TestCfg::from_successors(succ);
    check_ok(&g, 0..16);
    let mut m = meter();
    structurize(&g, &mut m).unwrap()
}

/// How many statements of the result match `f`.
fn count(s: &Structured, f: impl Fn(&Stmt) -> bool) -> usize {
    s.tree.stmts().filter(|(_, st)| f(st)).count()
}

#[test]
fn short_circuit_joins_are_tail_jumps() {
    // if (a && b) { X } Y: 0 -> {1, 3}; 1 -> {X2, 3}; X2 -> 3; Y3 ret.
    let s = structured(&[vec![1, 3], vec![2, 3], vec![3], vec![]]);
    assert_eq!(s.pending, None);
    assert_eq!(count(&s, |st| matches!(st, Stmt::GuardPending(_))), 0);
    assert_eq!(count(&s, |st| matches!(st, Stmt::If { .. })), 2);
}

#[test]
fn a_jump_past_another_merge_goes_through_pending() {
    // 0 -> {1, 4}; 1 -> {2, 3}; 2 -> 5; 3 -> 4; 4 -> 5; 5 ret. The jump 2 -> 5 leaves the block
    // labeled 4 with code(4) still to run: a non-tail jump, and code(4) is guarded.
    let s = structured(&[vec![1, 4], vec![2, 3], vec![5], vec![4], vec![5], vec![]]);
    assert!(s.pending.is_some());
    assert_eq!(count(&s, |st| matches!(st, Stmt::SetPending(_))), 1);
    assert_eq!(count(&s, |st| matches!(st, Stmt::ClearPending(_))), 1);
    assert_eq!(count(&s, |st| matches!(st, Stmt::GuardPending(_))), 1);
    assert_eq!(s.selectors, 1);
}

#[test]
fn a_break_tail_loop_has_one_exit_dispatch() {
    // while (h) { if (p) { tail; break; } latch } exit: the tail runs after the loop, so the loop
    // has two exit targets (the tail and the exit) and one selector dispatches them.
    let s = structured(&[vec![1], vec![2, 5], vec![3, 4], vec![5], vec![1], vec![]]);
    assert_eq!(count(&s, |st| matches!(st, Stmt::Loop { .. })), 1);
    assert_eq!(count(&s, |st| matches!(st, Stmt::Break)), 2);
    assert_eq!(count(&s, |st| matches!(st, Stmt::Continue)), 1);
    assert_eq!(count(&s, |st| matches!(st, Stmt::Dispatch { .. })), 1);
    assert_eq!(count(&s, |st| matches!(st, Stmt::SetSel { .. })), 2);
    assert_eq!(s.selectors, 1);
}

#[test]
fn a_returning_tail_is_dispatched_after_the_loop() {
    // while (h) { if (p) { tail; return; } latch } post; ret: two exit targets, one selector.
    let s = structured(&[
        vec![1],
        vec![2, 5],
        vec![3, 4],
        vec![],
        vec![1],
        vec![6],
        vec![],
    ]);
    assert_eq!(count(&s, |st| matches!(st, Stmt::Dispatch { .. })), 1);
    assert_eq!(count(&s, |st| matches!(st, Stmt::SetSel { .. })), 2);
    assert_eq!(s.selectors, 1);
}

#[test]
fn a_multi_level_break_is_an_exit_ladder() {
    // outer: while (ho) { inner: while (hi) { if (p) { T; break outer; } latch } post; }  E
    let s = structured(&[
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
    assert_eq!(count(&s, |st| matches!(st, Stmt::Loop { .. })), 2);
    // T runs after both loops: the inner loop dispatches post or a break of the outer loop, and the
    // outer loop dispatches T or E.
    assert_eq!(count(&s, |st| matches!(st, Stmt::Dispatch { .. })), 2);
    assert_eq!(s.pending, None);
}

#[test]
fn switches_with_shared_targets_and_default_merges() {
    // 0: switch { default: 3, 0: 1, 1: 2, 2: 1, 3: 3 }; 1 -> 3; 2 -> {1, 3}; 3 ret.
    let s = structured(&[vec![3, 1, 2, 1, 3], vec![3], vec![1, 3], vec![]]);
    let switches: Vec<_> = s
        .tree
        .stmts()
        .filter_map(|(_, st)| match st {
            Stmt::Switch { cases, .. } => Some(cases.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(switches.len(), 1);
    let literals: Vec<Vec<u64>> = switches[0].iter().map(|(l, _)| l.to_vec()).collect();
    assert_eq!(literals, [vec![0, 2], vec![1]]);
}

#[test]
fn an_infinite_loop_has_an_unreachable_merge() {
    // 0 -> 1; 1 -> {2, 3}; 2 -> 1; 3 -> 1: no exit.
    let s = structured(&[vec![1], vec![2, 3], vec![1], vec![1]]);
    assert_eq!(count(&s, |st| matches!(st, Stmt::Loop { .. })), 1);
    assert_eq!(count(&s, |st| matches!(st, Stmt::Unreachable)), 1);
    assert_eq!(count(&s, |st| matches!(st, Stmt::Break)), 0);
}

#[test]
fn irreducible_dispatch_survives_structuring() {
    for k in 1..=4 {
        let g = generate::irreducible_ladder(k);
        check_ok(&g, 0..8);
        let mut m = meter();
        let s = structurize(&g, &mut m).unwrap();
        assert!(count(&s, |st| matches!(st, Stmt::Dispatch { .. })) >= k as usize);
        s.release(&mut m);
        assert_eq!(m.ledger(), 0);
    }
}

#[test]
fn the_demotion_plan_lists_every_crossing_value_and_phi() {
    let mut g = TestCfg::from_successors(&[vec![1, 2], vec![3], vec![3], vec![]]);
    let int = DemoteTy::Scalar(ScalarTy::Int { bits: 32 });
    let pair = DemoteTy::Composite(vec![int.clone(), DemoteTy::FatPtr { wide: true }].into());
    g.cross = vec![
        (ValueId::new(7), int.clone()),
        (ValueId::new(3), pair.clone()),
    ];
    g.phis = vec![(ValueId::new(9), DemoteTy::ArenaOffset)];
    let mut m = meter();
    let s = structurize(&g, &mut m).unwrap();
    let demoted: Vec<_> = s.demoted.iter().map(|d| (d.value, d.ty.clone())).collect();
    assert_eq!(demoted, g.cross);
    let phis: Vec<_> = s.phis.iter().map(|p| (p.phi, p.ty.clone())).collect();
    assert_eq!(phis, g.phis);
    s.release(&mut m);
    assert_eq!(m.ledger(), 0);

    // A value listed twice breaks the view contract.
    g.phis.push((ValueId::new(9), int));
    let mut m = meter();
    let err = structurize(&g, &mut m).unwrap_err();
    assert!(matches!(err, Error::Internal(_)), "{err:?}");
    assert_eq!(m.ledger(), 0);
}

#[test]
fn the_output_cap_is_checked_before_emission() {
    let g = generate::break_tail_ladder(4);
    let mut m = meter();
    let mut prepared = prepare(&g, &mut m).unwrap();
    let held = m.ledger();
    prepared.output_cap = 8;
    assert_eq!(
        structure_prepared(&g, &prepared, &mut m).unwrap_err(),
        Error::LimitExceeded(Limit::OutputBlocks)
    );
    assert_eq!(m.ledger(), held, "a refused structuring kept ledger bytes");
    prepared.release(&mut m);
    assert_eq!(m.ledger(), 0);
}

/// Loops whose bodies are switches, with shared targets, cases to the default (merge), and
/// breaks/continues from the arms: `k` loops in sequence.
fn switch_loops(k: u32, seed: u64) -> TestCfg {
    let mut rng = Rng::new(seed);
    let mut succ = Vec::new();
    for i in 0..k {
        // h -> {s, exit}; s: switch over {a, b, latch, exit}; a -> latch; b -> {a, exit};
        // latch -> h.
        let h = 5 * i;
        let (s, a, b, latch, exit) = (h + 1, h + 2, h + 3, h + 4, h + 5);
        succ.push(vec![s, exit]);
        let mut arms = vec![latch, a, b, a, exit, h];
        let drop = rng.below(arms.len() as u64 - 2) as usize + 2;
        arms.truncate(drop.max(3));
        succ.push(arms);
        succ.push(vec![latch]);
        succ.push(if rng.chance(8) {
            vec![a, exit]
        } else {
            vec![h]
        });
        succ.push(vec![h]);
    }
    succ.push(vec![]);
    TestCfg::from_successors(&succ)
}

#[test]
fn switch_heavy_loops() {
    for seed in 0..200 {
        check_ok(&switch_loops(1 + (seed % 4) as u32, seed), 0..6);
    }
}

#[test]
fn multi_exit_programs() {
    // Structured programs with many escapes: multi-level breaks and continues.
    for seed in 0..300 {
        let g = generate::structured(seed ^ 0x6d75_6c74, 3 + (seed % 3) as u32, 0);
        check_ok(&g, 0..6);
    }
}

/// `lowered_blocks / (n + e)` of the pruned input, and whether the cap held.
fn size_ratio(g: &TestCfg) -> f64 {
    let mut m = meter();
    let cfg = Cfg::build(g, &mut m).unwrap();
    let n = cfg.reachable_count() as f64;
    let e: usize = (0..g.blocks.len() as u32)
        .map(BlockId::new)
        .filter(|&b| cfg.is_reachable(b))
        .map(|b| cfg.successors(b).len())
        .sum();
    let s = structurize(g, &mut m).unwrap();
    let ratio = (s.tree.lowered_blocks() - 1) as f64 / (n + e as f64);
    s.release(&mut m);
    ratio
}

#[test]
fn output_size_is_linear_with_a_small_constant() {
    let mut worst = 0f64;
    let mut rng = Rng::new(0x51ce);
    for seed in 0..300 {
        worst = worst.max(size_ratio(&generate::structured(
            seed,
            2 + (seed % 4) as u32,
            0,
        )));
        worst = worst.max(size_ratio(&switch_loops(1 + (seed % 5) as u32, seed)));
        let g = generate::dag_with_back_edges(2 + rng.below(40) as u32, rng.next_u64());
        let mut m = meter();
        if let Ok(s) = structurize(&g, &mut m) {
            s.release(&mut m);
            worst = worst.max(size_ratio(&g));
        }
    }
    for k in [1, 10, 100] {
        worst = worst.max(size_ratio(&generate::break_tail_ladder(k)));
        worst = worst.max(size_ratio(&generate::irreducible_ladder(k)));
    }
    // Reported constant (output linear in n + e); the cap allows 4. Measured when
    // this test was written: 1.75 over these families.
    eprintln!("structurizer output: at most {worst:.3} lowered blocks per (n + e)");
    assert!(worst <= 2.0, "lowered blocks per (n + e) reached {worst}");
}

#[test]
fn structuring_is_deterministic() {
    for seed in 0..50 {
        let g = generate::structured(seed, 4, (seed % 3) as u32);
        let run = || {
            let mut m = meter();
            structurize(&g, &mut m)
        };
        match (run(), run()) {
            (Ok(a), Ok(b)) => assert_eq!(a, b),
            (Err(a), Err(b)) => assert_eq!(a, b),
            (a, b) => panic!("runs differ: {a:?} / {b:?}"),
        }
    }
}

/// `structurize` under a sweep of ledger and work caps: every refusal leaves the ledger empty.
fn check_release_on_error(g: &TestCfg) {
    let budget = Budget::default();
    let mut full = meter();
    let s = structurize(g, &mut full).unwrap();
    let (peak, work) = (full.ledger_peak(), full.work());
    s.release(&mut full);
    assert_eq!(full.ledger(), 0);
    let mut refusals = 0;
    let steps = 61;
    for i in 0..steps {
        for mut m in [
            Meter::with_caps(&budget, u64::MAX, peak * i / steps),
            Meter::with_caps(&budget, work * i / steps, u64::MAX),
        ] {
            match structurize(g, &mut m) {
                Ok(s) => s.release(&mut m),
                Err(Error::LimitExceeded(Limit::LedgerBytes | Limit::WorkUnits)) => refusals += 1,
                Err(other) => panic!("unexpected error {other:?}"),
            }
            assert_eq!(m.ledger(), 0, "a refused structurize kept ledger bytes");
        }
    }
    assert!(
        refusals > steps,
        "the sweep refused too rarely ({refusals})"
    );
}

#[test]
fn structurize_refusals_leave_the_ledger_balanced() {
    check_release_on_error(&generate::structured(7, 5, 0));
    check_release_on_error(&generate::structured(11, 4, 2));
    check_release_on_error(&generate::break_tail_ladder(20));
    check_release_on_error(&switch_loops(6, 3));
    check_release_on_error(&TestCfg::from_successors(&[
        vec![1, 4],
        vec![2, 3],
        vec![5],
        vec![4],
        vec![5],
        vec![],
    ]));
}

/// Work units of `structurize` on `g`.
fn work(g: &TestCfg) -> u64 {
    let mut m = meter();
    structurize(g, &mut m).unwrap().release(&mut m);
    m.work()
}

#[test]
fn structurize_work_grows_linearly() {
    type Family = fn(u32) -> TestCfg;
    let families: [(&str, Family, u32); 4] = [
        ("break tails", generate::break_tail_ladder, 4_000),
        ("irreducible", generate::irreducible_ladder, 4_000),
        ("switch loops", |k| switch_loops(k, 5), 4_000),
        ("pending chain", pending_chain, 4_000),
    ];
    for (name, family, k) in families {
        let (ws, wl) = (work(&family(k)), work(&family(2 * k)));
        assert!(
            wl <= 2 * ws + ws / 5,
            "{name}: work {ws} -> {wl} when the graph doubles"
        );
    }
}

/// `k` copies in sequence of the non-tail jump shape `0 -> {1, 4}; 1 -> {2, 3}; 2 -> 5; 3 -> 4;
/// 4 -> 5`, each `5` starting the next copy.
fn pending_chain(k: u32) -> TestCfg {
    let mut succ = Vec::new();
    for i in 0..k {
        let b = 5 * i;
        succ.push(vec![b + 1, b + 4]);
        succ.push(vec![b + 2, b + 3]);
        succ.push(vec![b + 5]);
        succ.push(vec![b + 4]);
        succ.push(vec![b + 5]);
    }
    succ.push(vec![]);
    TestCfg::from_successors(&succ)
}

#[test]
fn deep_trees_do_not_recurse() {
    // A 200k-block chain, a 100k-deep `if` nest, and a 2000-deep loop nest, on the default test
    // thread stack.
    let chain: Vec<Vec<u32>> = (0..200_000u32)
        .map(|i| if i + 1 < 200_000 { vec![i + 1] } else { vec![] })
        .collect();
    let nest: Vec<Vec<u32>> = (0..100_000u32)
        .map(|i| {
            if i + 1 < 100_000 {
                vec![i + 1, 100_000]
            } else {
                vec![100_000]
            }
        })
        .chain(std::iter::once(vec![]))
        .collect();
    let depth = 2_000u32;
    let mut loops = Vec::new();
    for i in 0..depth {
        let latch = 2 * i + 1;
        let inner = if i + 1 < depth { 2 * (i + 1) } else { latch };
        loops.push(vec![inner, latch]);
        loops.push(if i == 0 {
            vec![0, 2 * depth]
        } else {
            vec![2 * i, 2 * i - 1]
        });
    }
    loops.push(vec![]);
    for succ in [chain, nest, loops] {
        let g = TestCfg::from_successors(&succ);
        let mut m = meter();
        let s = structurize(&g, &mut m).unwrap();
        check_structured(&s, &originals(&g)).unwrap();
        s.release(&mut m);
        assert_eq!(m.ledger(), 0);
    }
}

#[test]
fn structured_results_are_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<Structured>();
}
