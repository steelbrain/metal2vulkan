//! P8/P9 properties: every generated CFG structurizes,
//! emits through the builder, and passes the owned verifier and `spirv-val` (Vulkan 1.2 and 1.3)
//! before and after single-store promotion; promotion removes exactly the variables whose store
//! dominates every load; loops carry their exit selector; deep nests are refused before the
//! builder recurses past the cap; emission is deterministic.
//!
//! Many functions share one module (each a callee of the entry), so one `spirv-val` run covers a
//! whole batch of graphs.
//!
//! **Gate and soak** (the split). Every graph of at most five blocks with out-degree at most two
//! and of at most three blocks with switches is emitted in every run. G-rel emits deterministic
//! hash samples of the larger families (`generate::in_sample`): one in 64 of the 710,556 six-block
//! graphs, one in 8 of the 111,807 four-block switch graphs, and one in 4,096 of the 15,794,698
//! five-block ones, each with every check below. The ignored `soak_` tests emit every six-block
//! graph, every four-block switch graph, and one five-block switch graph in 256:
//!
//! ```text
//! cargo test -p metal2vulkan --all-features --release --lib -- --ignored structurize::emit::tests::soak_
//! ```

mod convergence;

use std::collections::BTreeMap;

use super::{emit, CodeEmitter, StructVars};
use crate::air::ir::{BlockId, ValueId};
use crate::budget::{Budget, Meter, CONSTRUCT_NESTING};
use crate::device::profile::DeviceProfile;
use crate::error::{Error, Limit};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, FuncId, ScalarTy, TypeId, Value, VarId};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ExecMode, ExecModel, Inst, MemAccess, StorageClass, Type,
    TypedModule, UnOp,
};
use crate::spirv::testing::{check_valid, committed_profiles, spirv_val, TargetEnv};
use crate::structurize::promote::{promote_single_store, PromoteReport};
use crate::structurize::testkit::generate::{self, Rng};
use crate::structurize::testkit::TestCfg;
use crate::structurize::tree::Stmt;
use crate::structurize::{prepare, structurize, CfgView};

fn meter() -> Meter {
    Meter::new(&Budget::default())
}

/// Phi `ValueId`s are `PHI_BASE + block`, demoted values `ValueId(block)`.
const PHI_BASE: u32 = 1 << 20;

/// A view of `g` in which every block's value crosses blocks and every join has a phi.
fn with_values(g: &TestCfg) -> TestCfg {
    let mut g = g.clone();
    let int = DemoteTy::Scalar(ScalarTy::Int { bits: 32 });
    let mut preds = vec![0u32; g.blocks.len()];
    for b in &g.blocks {
        for s in &b.succ {
            preds[s.raw() as usize] += 1;
        }
    }
    g.cross = (0..g.blocks.len() as u32)
        .map(|b| (ValueId::new(b), int.clone()))
        .collect();
    g.phis = (0..g.blocks.len() as u32)
        .filter(|&b| preds[b as usize] >= 2)
        .map(|b| (ValueId::new(PHI_BASE + b), int.clone()))
        .collect();
    g
}

/// Module-wide test state: the `Private` hash and return slots.
struct Globals {
    u32_: TypeId,
    state: Value,
    ret: Value,
}

/// A [`CodeEmitter`] for a [`TestCfg`]: each block mixes its id and the demoted values of its
/// predecessors into a `Private` hash, stores its own value, writes its successors' phi
/// variables, and runs one tangled `OpGroupNonUniformElect` when marked tangled. Conditions and
/// selectors are read from the hash.
struct TestEmitter<'a> {
    g: &'a TestCfg,
    preds: Vec<Vec<u32>>,
    glob: &'a mut Globals,
    vars: Option<StructVars>,
}

impl<'a> TestEmitter<'a> {
    fn new(g: &'a TestCfg, glob: &'a mut Globals) -> TestEmitter<'a> {
        let mut preds = vec![Vec::new(); g.blocks.len()];
        for (i, b) in g.blocks.iter().enumerate() {
            for s in &b.succ {
                preds[s.raw() as usize].push(i as u32);
            }
        }
        TestEmitter {
            g,
            preds,
            glob,
            vars: None,
        }
    }

    fn konst(&self, fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
        let c = fb.module().constant(Constant::Scalar {
            ty: self.glob.u32_,
            bits: ConstBits(u64::from(v)),
        })?;
        fb.module().const_value(c)
    }
}

impl CodeEmitter for TestEmitter<'_> {
    fn var_type(&mut self, ty: &DemoteTy, _m: &mut ModuleBuilder) -> BResult<TypeId> {
        assert_eq!(ty, &DemoteTy::Scalar(ScalarTy::Int { bits: 32 }));
        Ok(self.glob.u32_)
    }

    fn bind(&mut self, vars: &StructVars) -> BResult<()> {
        self.vars = Some(vars.clone());
        Ok(())
    }

    fn block(&mut self, b: BlockId, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
        let vars = self.vars.as_ref().expect("bound");
        let loads: Vec<VarId> = self.preds[b.raw() as usize]
            .iter()
            .filter_map(|&p| vars.demoted_var(ValueId::new(p)))
            .chain(vars.phi_var(ValueId::new(PHI_BASE + b.raw())))
            .collect();
        let own = vars.demoted_var(ValueId::new(b.raw()));
        let phi_stores: Vec<VarId> = self.g.blocks[b.raw() as usize]
            .succ
            .iter()
            .filter_map(|s| vars.phi_var(ValueId::new(PHI_BASE + s.raw())))
            .collect();
        let a = MemAccess::default();
        let mut x = fb.load(self.glob.state, a)?;
        for var in loads {
            let ptr = fb.var_ptr(var)?;
            let v = fb.load(ptr, a)?;
            x = fb.binary(BinOp::BitwiseXor, x, v)?;
        }
        let k = self.konst(fb, 0x9e37_79b1)?;
        x = fb.binary(BinOp::IMul, x, k)?;
        let id = self.konst(fb, b.raw() + 1)?;
        x = fb.binary(BinOp::IAdd, x, id)?;
        if self.g.blocks[b.raw() as usize].tangled {
            let e = fb.group_elect()?;
            let (one, zero) = (self.konst(fb, 1)?, self.konst(fb, 0)?);
            let bit = fb.select(e, one, zero)?;
            x = fb.binary(BinOp::BitwiseXor, x, bit)?;
        }
        fb.store(self.glob.state, x, a)?;
        for var in own.into_iter().chain(phi_stores) {
            let ptr = fb.var_ptr(var)?;
            fb.store(ptr, x, a)?;
        }
        Ok(())
    }

    fn cond(&mut self, _v: ValueId, fb: &mut FunctionBuilder<'_>) -> BResult<Value> {
        let x = fb.load(self.glob.state, MemAccess::default())?;
        let one = self.konst(fb, 1)?;
        let bit = fb.binary(BinOp::BitwiseAnd, x, one)?;
        fb.binary(BinOp::IEqual, bit, one)
    }

    fn selector(&mut self, _v: ValueId, fb: &mut FunctionBuilder<'_>) -> BResult<Value> {
        let x = fb.load(self.glob.state, MemAccess::default())?;
        let range = self.konst(fb, generate_range())?;
        fb.binary(BinOp::UMod, x, range)
    }

    fn on_return(&mut self, v: Option<ValueId>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
        if v.is_some() {
            let x = fb.load(self.glob.state, MemAccess::default())?;
            fb.store(self.glob.ret, x, MemAccess::default())?;
        }
        fb.ret(None)
    }
}

fn generate_range() -> u32 {
    crate::structurize::testkit::SELECTOR_RANGE as u32
}

/// One emitted function: the structurizer's statement counts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct FnFacts {
    loops: usize,
    exit_dispatches: usize,
}

/// A module holding one function per graph `structurize` accepts (the others are skipped), and
/// the module before promotion.
struct Batch {
    before: TypedModule,
    after: TypedModule,
    funcs: Vec<(usize, FuncId, FnFacts)>,
    report: PromoteReport,
}

/// Structurize, emit, finalize, and promote `graphs` (each given a value per block and a phi per
/// join), one function per graph.
fn build_batch(graphs: &[TestCfg]) -> Batch {
    let mut m = meter();
    let mut b = ModuleBuilder::new();
    let void = b.ty(Type::Void).unwrap();
    let u32_ = b
        .ty(Type::Int {
            width: 32,
            signed: false,
        })
        .unwrap();
    let state = b.plain_global(StorageClass::Private, u32_, None).unwrap();
    let ret = b.plain_global(StorageClass::Private, u32_, None).unwrap();
    let mut glob = Globals {
        u32_,
        state: b.global_value(state).unwrap(),
        ret: b.global_value(ret).unwrap(),
    };
    let mut funcs = Vec::new();
    for (i, g) in graphs.iter().enumerate() {
        let g = with_values(g);
        let s = match structurize(&g, &mut m) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let loops = s
            .tree
            .stmts()
            .filter(|(_, st)| matches!(st, Stmt::Loop { .. }))
            .count();
        // Exit selectors are numbered after P1's entry selectors (P5), so every multi-arm
        // dispatch on one is a loop's exit dispatch.
        let entry_selectors = {
            let p = prepare(&g, &mut m).unwrap();
            let n = p.cfg.selector_count();
            p.release(&mut m);
            n
        };
        let exit_dispatches = s
            .tree
            .stmts()
            .filter(|(_, st)| {
                matches!(st, Stmt::Dispatch { sel, arms }
                    if arms.len() > 1 && sel.index() >= entry_selectors)
            })
            .count();
        let mut f = b.function(void).unwrap();
        let mut e = TestEmitter::new(&g, &mut glob);
        emit(&s, &mut f, &mut e, &mut m)
            .unwrap_or_else(|err| panic!("P8 refused {err:?} for {g:?}"));
        let id = f.finish().unwrap();
        s.release(&mut m);
        funcs.push((
            i,
            id,
            FnFacts {
                loops,
                exit_dispatches,
            },
        ));
    }
    let mut entry = b.function(void).unwrap();
    for (_, id, _) in &funcs {
        entry.call(*id, &[]).unwrap();
    }
    entry.ret(None).unwrap();
    let entry = entry.finish().unwrap();
    b.entry(
        ExecModel::GLCompute,
        entry,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )
    .unwrap();
    let before = b.finalize().unwrap();
    let mut after = before.clone();
    let report = promote_single_store(&mut after, &mut m).unwrap();
    assert_eq!(m.ledger(), 0, "P8/P9 left ledger bytes behind");
    Batch {
        before,
        after,
        funcs,
        report,
    }
}

/// The profile the batches are validated on: Vulkan 1.3 (so L3 runs `spirv-val` for 1.2 and
/// 1.3) with compute subgroups.
fn profile() -> DeviceProfile {
    committed_profiles()
        .unwrap()
        .into_iter()
        .find(|(name, _)| name == "runtime-floor")
        .expect("committed runtime-floor profile")
        .1
}

/// How much of L3 a batch runs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Depth {
    /// The owned verifier and `spirv-val` (1.2 and 1.3) before and after P9.
    Full,
    /// The owned verifier before and after P9, `spirv-val` after P9 only (the release families,
    /// where `spirv-val` dominates the run time).
    Shipped,
}

/// Structurize, emit, and promote `graphs`, and validate the module before and after P9 (owned
/// verifier with the profile rules, requirements, `spirv-val` 1.2 and 1.3). Returns the batch.
fn check_batch(graphs: &[TestCfg], profile: &DeviceProfile, depth: Depth) -> Batch {
    assert!(
        TargetEnv::for_profile(profile).contains(&TargetEnv::Vulkan1_3),
        "the batch profile must cover Vulkan 1.3"
    );
    let batch = build_batch(graphs);
    if depth == Depth::Shipped {
        crate::spirv::verify::check(&batch.before, profile, &mut meter())
            .unwrap_or_else(|e| panic!("P8 output rejected ({} graphs): {e}", graphs.len()));
    }
    let full = [(&batch.before, "P8"), (&batch.after, "P9")];
    let checked = if depth == Depth::Full {
        &full[..]
    } else {
        &full[1..]
    };
    for (m, when) in checked {
        let words = check_valid((*m).clone(), profile)
            .unwrap_or_else(|e| panic!("{when} output rejected ({} graphs): {e}", graphs.len()));
        assert!(!words.is_empty());
    }
    for (_, id, facts) in &batch.funcs {
        assert_eq!(
            loops_with_exit_sel(&batch.after, *id),
            facts.exit_dispatches,
            "every multi-exit loop carries its exit selector"
        );
        assert_eq!(loops_of(&batch.after, *id), facts.loops);
    }
    batch
}

fn loops_of(m: &TypedModule, f: FuncId) -> usize {
    m.loops()
        .iter()
        .filter(|l| m.blocks()[l.header.index()].func == f)
        .count()
}

fn loops_with_exit_sel(m: &TypedModule, f: FuncId) -> usize {
    m.loops()
        .iter()
        .filter(|l| m.blocks()[l.header.index()].func == f && l.exit_sel.is_some())
        .count()
}

/// Check `graphs` in batches of `size`.
fn check_all(graphs: impl IntoIterator<Item = TestCfg>, size: usize, depth: Depth) -> usize {
    let profile = profile();
    let mut batch = Vec::with_capacity(size);
    let mut count = 0;
    for g in graphs {
        batch.push(g);
        if batch.len() == size {
            count += check_batch(&batch, &profile, depth).funcs.len();
            batch.clear();
        }
    }
    if !batch.is_empty() {
        count += check_batch(&batch, &profile, depth).funcs.len();
    }
    count
}

fn canonical(n: u32, choices: Vec<Vec<u32>>) -> Vec<TestCfg> {
    let mut out = Vec::new();
    generate::canonical(n, &choices, |g| out.push(g.clone()));
    out
}

#[test]
fn exhaustive_out_degree_two_up_to_five_blocks_emit_valid_spirv() {
    let graphs: Vec<TestCfg> = (1..=5)
        .flat_map(|n| canonical(n, generate::out_degree_two_choices(n)))
        .collect();
    assert_eq!(graphs.len(), 2 + 8 + 91 + 1419 + 28752);
    assert_eq!(check_all(graphs, 4000, Depth::Full), 30272);
}

#[test]
fn exhaustive_switches_up_to_three_blocks_emit_valid_spirv() {
    let graphs: Vec<TestCfg> = (1..=3)
        .flat_map(|n| canonical(n, generate::switch_choices(n)))
        .collect();
    let n = graphs.len();
    assert_eq!(check_all(graphs, 4000, Depth::Full), n);
}

/// One in this many of the 710,556 six-block out-degree-two graphs is emitted in G-rel (a hash
/// sample, `generate::in_sample`, as `structurizer_properties` samples);
/// `soak_exhaustive_out_degree_two_six_blocks_emit_valid_spirv` emits all of them.
const SIX_BLOCK_EMIT_SAMPLE: u64 = 64;
/// One in this many four-block switch graphs is emitted in G-rel; the soak emits all of them.
const FOUR_BLOCK_SWITCH_EMIT_SAMPLE: u64 = 8;
/// One in this many of the 15,794,698 five-block switch graphs is emitted in G-rel.
const FIVE_BLOCK_SWITCH_EMIT_SAMPLE: u64 = 4096;
/// One in this many five-block switch graphs is emitted by the switch soak.
const FIVE_BLOCK_SWITCH_SOAK_SAMPLE: u64 = 256;

/// Emit and check, in batches of `size`, the canonical graphs of `n` blocks over `choices` whose
/// enumeration index is in the one-in-`every` hash sample (`every = 1`: all of them), streamed so
/// that no family is held at once. Returns how many were enumerated and how many checked.
fn check_sampled(
    n: u32,
    choices: &[Vec<u32>],
    every: u64,
    size: usize,
    depth: Depth,
) -> (u64, u64) {
    let profile = profile();
    let mut batch = Vec::with_capacity(size);
    let mut checked = 0u64;
    let total = generate::canonical_sampled(
        n,
        choices,
        |i| generate::in_sample(i, every),
        |g| {
            batch.push(g.clone());
            if batch.len() == size {
                checked += check_batch(&batch, &profile, depth).funcs.len() as u64;
                batch.clear();
            }
        },
    );
    if !batch.is_empty() {
        checked += check_batch(&batch, &profile, depth).funcs.len() as u64;
    }
    (total, checked)
}

/// A one-in-`every` sample of `total` graphs holds `checked`, within 10% of `total / every`.
fn assert_sample_size(total: u64, checked: u64, every: u64) {
    assert!(
        checked * every > total * 9 / 10 && checked * every < total * 11 / 10,
        "sampled {checked} of {total} at one in {every}"
    );
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "six-block graphs: runs in release (G-rel: cargo test --release -- structurize::emit)"
)]
fn sampled_out_degree_two_six_blocks_emit_valid_spirv() {
    // All 710,556 graphs passed when this was written (release, 406 s at one thread); G-rel emits a
    // hash sample (the split), and the ignored soak all of them.
    let (total, checked) = check_sampled(
        6,
        &generate::out_degree_two_choices(6),
        SIX_BLOCK_EMIT_SAMPLE,
        8000,
        Depth::Shipped,
    );
    assert_eq!(total, 710_556);
    eprintln!("six-block graphs: {total}, emitted {checked}");
    assert_sample_size(total, checked, SIX_BLOCK_EMIT_SAMPLE);
}

#[test]
#[ignore = "soak: all 710,556 six-block graphs (cargo test -p metal2vulkan --all-features --release --lib -- --ignored structurize::emit::tests::soak_)"]
fn soak_exhaustive_out_degree_two_six_blocks_emit_valid_spirv() {
    let (total, checked) = check_sampled(
        6,
        &generate::out_degree_two_choices(6),
        1,
        8000,
        Depth::Shipped,
    );
    assert_eq!((total, checked), (710_556, 710_556));
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "switch graphs on four and five blocks: runs in release (G-rel)"
)]
fn sampled_switches_four_and_five_blocks_emit_valid_spirv() {
    let (total, checked) = check_sampled(
        4,
        &generate::switch_choices(4),
        FOUR_BLOCK_SWITCH_EMIT_SAMPLE,
        8000,
        Depth::Shipped,
    );
    assert_eq!(total, 111_807);
    eprintln!("four-block switch graphs: {total}, emitted {checked}");
    assert_sample_size(total, checked, FOUR_BLOCK_SWITCH_EMIT_SAMPLE);
    let (total, checked) = check_sampled(
        5,
        &generate::switch_choices(5),
        FIVE_BLOCK_SWITCH_EMIT_SAMPLE,
        8000,
        Depth::Shipped,
    );
    assert_eq!(total, 15_794_698);
    eprintln!("five-block switch graphs: {total}, emitted {checked}");
    assert_sample_size(total, checked, FIVE_BLOCK_SWITCH_EMIT_SAMPLE);
}

/// Every four-block switch graph, and a one-in-256 hash sample of the five-block ones (P0-P7 run
/// on all of them in the `structurizer_properties` soak; this adds P8/P9 and L3).
#[test]
#[ignore = "soak: every four-block switch graph and one in 256 five-block ones (cargo test -p metal2vulkan --all-features --release --lib -- --ignored structurize::emit::tests::soak_)"]
fn soak_switches_four_blocks_and_sampled_five_emit_valid_spirv() {
    let (total, checked) = check_sampled(4, &generate::switch_choices(4), 1, 8000, Depth::Shipped);
    assert_eq!((total, checked), (111_807, 111_807));
    let (total, checked) = check_sampled(
        5,
        &generate::switch_choices(5),
        FIVE_BLOCK_SWITCH_SOAK_SAMPLE,
        8000,
        Depth::Shipped,
    );
    assert_eq!(total, 15_794_698);
    eprintln!("five-block switch graphs: {total}, emitted {checked}");
    assert_sample_size(total, checked, FIVE_BLOCK_SWITCH_SOAK_SAMPLE);
}

#[test]
fn generated_families_emit_valid_spirv() {
    let mut graphs = Vec::new();
    for seed in 0..400 {
        graphs.push(generate::structured(seed, 2 + (seed % 4) as u32, 0));
        graphs.push(generate::structured(
            seed ^ 0x6d75_6c74,
            3 + (seed % 3) as u32,
            0,
        ));
        graphs.push(generate::structured(
            seed,
            3 + (seed % 3) as u32,
            1 + (seed % 3) as u32,
        ));
    }
    let mut rng = Rng::new(0x5eed_12b0);
    for _ in 0..1500 {
        let n = 2 + rng.below(40) as u32;
        graphs.push(generate::dag_with_back_edges(n, rng.next_u64()));
    }
    for k in 1..=12 {
        graphs.push(generate::irreducible_ladder(k));
        graphs.push(generate::break_tail_ladder(k));
    }
    for k in 1..=8 {
        graphs.push(generate::nested_irreducible(k));
    }
    // Tangled blocks: every reducible structured program with tangled markers everywhere.
    for seed in 0..200 {
        let g = generate::structured(seed, 2 + (seed % 4) as u32, 0);
        let all: Vec<u32> = (0..g.blocks.len() as u32).collect();
        graphs.push(g.with_tangled(&all));
    }
    let total = graphs.len();
    let emitted = check_all(graphs, 1000, Depth::Full);
    // Only depth refusals and tangled irreducible regions are skipped.
    assert!(emitted * 10 >= total * 9, "{emitted} of {total} emitted");
}

#[test]
fn every_committed_profile_accepts_an_untangled_batch() {
    let graphs: Vec<TestCfg> = (0..60)
        .map(|seed| generate::structured(seed, 3, (seed % 2) as u32))
        .collect();
    let batch = build_batch(&graphs);
    for (name, profile) in committed_profiles().unwrap() {
        check_valid(batch.after.clone(), &profile)
            .unwrap_or_else(|e| panic!("{name} rejected the P9 output: {e}"));
    }
}

/// Count the instructions of `f` matching `pred`.
fn count_insts(m: &TypedModule, f: FuncId, pred: impl Fn(&Inst) -> bool) -> usize {
    m.functions()[f.index()]
        .blocks
        .iter()
        .flat_map(|b| m.blocks()[b.index()].insts.iter())
        .filter(|n| pred(&n.inst))
        .count()
}

#[test]
fn promotion_removes_dominated_single_store_variables_only() {
    // A diamond: 0 -> {1, 2} -> 3. Block values: 0 dominates every use of its value (in 1, 2),
    // so it is promoted; 1 and 2 are single-store values loaded in 3, which they do not
    // dominate, so they stay; the phi of 3 has two stores and stays.
    let g = TestCfg::from_successors(&[vec![1, 2], vec![3], vec![3], vec![]]);
    let batch = build_batch(&[g]);
    let (_, f, _) = &batch.funcs[0];
    let report = batch.report;
    let before = &batch.before.functions()[f.index()];
    let after = &batch.after.functions()[f.index()];
    assert_eq!(
        report.promoted as usize,
        before.vars.len() - after.vars.len()
    );
    assert_eq!(report.kept as usize, after.vars.len());
    // Variables of block values 0 and 3 (3's value is stored and never loaded) are promoted.
    assert_eq!(report.promoted, 2, "{report:?}");
    let copies = count_insts(&batch.after, *f, |i| {
        matches!(
            i,
            Inst::Unary {
                op: UnOp::CopyObject,
                ..
            }
        )
    });
    assert_eq!(copies, 2, "the loads of block 0's value in blocks 1 and 2");
    check_valid(batch.after.clone(), &profile()).unwrap();

    // A loop: a value stored in the body and loaded by the next iteration's header is not
    // dominated at the load (the load precedes the store in the same block or the header
    // dominates the body), so it stays.
    let g = TestCfg::from_successors(&[vec![1], vec![2, 3], vec![1], vec![]]);
    let batch = build_batch(&[g]);
    let (_, f, _) = &batch.funcs[0];
    let m = &batch.after;
    for v in &m.functions()[f.index()].vars {
        let ptr = m.vars()[v.index()].value;
        let stores = count_insts(
            m,
            *f,
            |i| matches!(i, Inst::Store { ptr: p, .. } if *p == ptr),
        );
        let loads = count_insts(
            m,
            *f,
            |i| matches!(i, Inst::Load { ptr: p, .. } if *p == ptr),
        );
        assert!(
            stores != 1 || loads > 0,
            "a single-store variable without loads survived"
        );
    }
    check_valid(batch.after.clone(), &profile()).unwrap();
}

#[test]
fn promotion_is_idempotent_and_deterministic() {
    let graphs: Vec<TestCfg> = (0..40)
        .map(|seed| generate::structured(seed, 4, (seed % 3) as u32))
        .collect();
    let (a, b) = (build_batch(&graphs), build_batch(&graphs));
    let words = |m: &TypedModule| {
        crate::spirv::serialize::serialize(m.clone())
            .map(|r| r.words)
            .unwrap()
    };
    assert_eq!(
        words(&a.after),
        words(&b.after),
        "P8/P9 are not deterministic"
    );
    let mut again = a.after.clone();
    let mut m = meter();
    let r = promote_single_store(&mut again, &mut m).unwrap();
    assert_eq!(r.promoted, 0, "a second P9 still promoted something");
    assert_eq!(m.ledger(), 0);
    assert_eq!(words(&again), words(&a.after));
}

/// `k` nested `if`s: block `i` branches to `i + 1` or to the common join `k`.
fn if_nest(k: u32) -> TestCfg {
    let succ: Vec<Vec<u32>> = (0..k)
        .map(|i| if i + 1 < k { vec![i + 1, k] } else { vec![k] })
        .chain(std::iter::once(vec![]))
        .collect();
    TestCfg::from_successors(&succ)
}

/// The construct nesting depth P8 recurses to for `g`.
fn nesting(g: &TestCfg) -> u64 {
    let mut m = meter();
    let s = structurize(g, &mut m).unwrap();
    let mut depth = vec![0u64; s.tree.len()];
    let mut max = 0;
    for (id, st) in s.tree.stmts().rev() {
        let inner = depth[id.index()] + u64::from(super::is_construct(st));
        max = max.max(inner);
        for c in st.children() {
            depth[c.index()] = inner;
        }
    }
    s.release(&mut m);
    max
}

/// Emit `g` into a fresh function of a throwaway module.
fn emit_one(g: &TestCfg) -> Result<StructVars, Error> {
    let g = with_values(g);
    let mut m = meter();
    let s = structurize(&g, &mut m).unwrap();
    let mut b = ModuleBuilder::new();
    let void = b.ty(Type::Void).unwrap();
    let u32_ = b
        .ty(Type::Int {
            width: 32,
            signed: false,
        })
        .unwrap();
    let state = b.plain_global(StorageClass::Private, u32_, None).unwrap();
    let mut glob = Globals {
        u32_,
        state: b.global_value(state).unwrap(),
        ret: b.global_value(state).unwrap(),
    };
    let mut f = b.function(void).unwrap();
    let mut e = TestEmitter::new(&g, &mut glob);
    let r = emit(&s, &mut f, &mut e, &mut m);
    s.release(&mut m);
    assert_eq!(m.ledger(), 0);
    r
}

#[test]
fn nesting_at_the_cap_emits_on_a_2_mib_stack_and_beyond_it_is_refused() {
    // Find the nest whose depth is exactly the cap (one `If` per level plus the root).
    let k = (1..=CONSTRUCT_NESTING as u32 + 8)
        .find(|&k| nesting(&if_nest(k)) == CONSTRUCT_NESTING)
        .expect("an if nest at the cap");
    // At the cap, a 2 MiB thread stack suffices, debug build included (about 6.8 KiB per level
    // in debug, 1 KiB in release, when the cap was set).
    std::thread::Builder::new()
        .stack_size(2 << 20)
        .spawn(move || emit_one(&if_nest(k)).map(|_| ()))
        .unwrap()
        .join()
        .expect("emission at the cap overflowed a 2 MiB stack")
        .unwrap();
    // One level more is refused before the builder is touched.
    assert_eq!(nesting(&if_nest(k + 1)), CONSTRUCT_NESTING + 1);
    assert_eq!(
        emit_one(&if_nest(k + 1)).unwrap_err(),
        Error::LimitExceeded(Limit::ConstructNesting)
    );
}

/// The walk charges the builder's growth after each block (the ledger peak covers every byte
/// the walk added) and returns it when it ends; under a ledger too small for the module, the
/// walk is refused with `LedgerBytes` while it grows, not after.
#[test]
fn the_walk_charges_the_builder_growth_as_it_emits() {
    let chain: Vec<Vec<u32>> = (0..2_000u32)
        .map(|i| if i + 1 < 2_000 { vec![i + 1] } else { vec![] })
        .collect();
    let g = with_values(&TestCfg::from_successors(&chain));
    let walk = |m: &mut Meter| -> (Result<StructVars, Error>, usize, usize, u64) {
        let s = structurize(&g, m).unwrap();
        let held = m.ledger();
        let mut b = ModuleBuilder::new();
        let void = b.ty(Type::Void).unwrap();
        let u32_ = b
            .ty(Type::Int {
                width: 32,
                signed: false,
            })
            .unwrap();
        let state = b.plain_global(StorageClass::Private, u32_, None).unwrap();
        let mut glob = Globals {
            u32_,
            state: b.global_value(state).unwrap(),
            ret: b.global_value(state).unwrap(),
        };
        let before = b.footprint();
        let r = {
            let mut f = b.function(void).unwrap();
            let mut e = TestEmitter::new(&g, &mut glob);
            emit(&s, &mut f, &mut e, m)
        };
        s.release(m);
        (r, before, b.footprint(), held)
    };
    let mut m = meter();
    let (r, before, after, held) = walk(&mut m);
    r.unwrap();
    assert_eq!(m.ledger(), 0, "the walk returns what it charged");
    let grown = (after - before) as u64;
    assert!(grown > 2_000 * 8, "the builder grew by {grown} bytes");
    assert!(m.ledger_peak() >= grown, "{} < {grown}", m.ledger_peak());

    // With room for the structurizer but not the module, the walk is refused mid-emission.
    let structurizer_peak = {
        let mut alone = meter();
        structurize(&g, &mut alone).unwrap().release(&mut alone);
        alone.ledger_peak()
    };
    let mut tight = Meter::with_caps(
        &Budget::default(),
        crate::budget::WORK_UNITS,
        structurizer_peak.max(held + grown / 2),
    );
    let (r, _, after, _) = walk(&mut tight);
    assert_eq!(r.unwrap_err(), Error::LimitExceeded(Limit::LedgerBytes));
    assert!(
        after - before < grown as usize,
        "refused before the whole module was built"
    );
    assert_eq!(tight.ledger(), 0);
}

#[test]
fn a_deep_sequence_and_a_deep_loop_nest_emit() {
    // Sequences do not recurse: a 20k-block chain emits; so does a loop nest near the cap.
    let chain: Vec<Vec<u32>> = (0..20_000u32)
        .map(|i| if i + 1 < 20_000 { vec![i + 1] } else { vec![] })
        .collect();
    emit_one(&TestCfg::from_successors(&chain)).unwrap();
    let depth = 100u32;
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
    let g = TestCfg::from_successors(&loops);
    assert!(nesting(&g) <= CONSTRUCT_NESTING);
    let batch = build_batch(&[g]);
    assert_eq!(loops_of(&batch.after, batch.funcs[0].1), depth as usize);
    check_valid(batch.after, &profile()).unwrap();
}

#[test]
fn wide_switch_literals_are_not_implemented() {
    let mut g = TestCfg::from_successors(&[vec![1, 2], vec![], vec![]]);
    g.blocks[0].kind = crate::structurize::testkit::TestKind::Switch(vec![1 << 40]);
    let err = emit_one(&g).unwrap_err();
    let Error::Unsupported(u) = err else {
        panic!("expected Unsupported, got {err:?}");
    };
    assert_eq!(u.feature, crate::error::Feature::WideSwitchLiteral);
    assert_eq!(u.needs, crate::error::Needs::NotImplemented);
}

#[test]
fn pending_starts_at_zero_in_the_entry_block() {
    // The non-tail jump shape (P7 uses `pending`): the entry block stores 0 into it first.
    let g = TestCfg::from_successors(&[vec![1, 4], vec![2, 3], vec![5], vec![4], vec![5], vec![]]);
    let batch = build_batch(&[g]);
    let (_, f, _) = &batch.funcs[0];
    let m = &batch.before;
    let func = &m.functions()[f.index()];
    let entry = &m.blocks()[func.blocks[0].index()];
    let Some(Inst::Store { ptr, value, .. }) = entry.insts.first().map(|n| &n.inst) else {
        panic!("the entry block does not start with the pending store");
    };
    let pending_ptrs: Vec<Value> = func
        .vars
        .iter()
        .map(|v| m.vars()[v.index()].value)
        .collect();
    assert!(pending_ptrs.contains(ptr));
    let zero = m
        .consts()
        .iter()
        .find(|(_, c)| {
            matches!(
                &c.kind,
                crate::spirv::ir::ConstKind::Plain(Constant::Scalar { bits: ConstBits(0), ty })
                    if *ty == m.vars()[func.vars[0].index()].pointee
            )
        })
        .map(|(id, _)| m.const_value(id).unwrap())
        .expect("a u32 zero");
    assert_eq!(*value, zero);
    // The same module validates under Vulkan 1.2 without a profile's relaxations too.
    let r = crate::spirv::serialize::serialize(batch.after.clone()).unwrap();
    let words =
        crate::spirv::serialize::relocate(&r, &crate::spirv::testing::link_dense(&r)).unwrap();
    spirv_val(&words, TargetEnv::Vulkan1_2).unwrap();
}

#[test]
fn view_lists_are_what_the_emitter_binds() {
    // `with_values` gives every block a demoted value and every join a phi; P8 allocates one
    // variable each, plus the selectors, and binds them all.
    let g = with_values(&TestCfg::from_successors(&[
        vec![1, 2],
        vec![3],
        vec![3],
        vec![],
    ]));
    let vars = emit_one(&g).unwrap();
    let demoted: BTreeMap<ValueId, VarId> = vars.demoted.iter().copied().collect();
    assert_eq!(demoted.len(), g.cross_block_values().len());
    assert_eq!(vars.phis.len(), g.phis().len());
    for (v, var) in &vars.demoted {
        assert_eq!(vars.demoted_var(*v), Some(*var));
    }
    for (v, var) in &vars.phis {
        assert_eq!(vars.phi_var(*v), Some(*var));
    }
    assert_eq!(vars.pending, None);
}
