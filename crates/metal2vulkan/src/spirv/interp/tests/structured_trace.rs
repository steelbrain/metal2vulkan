//! SPIR-V-level trace equivalence and convergence equivalence of the
//! structurizer's emitted modules: every generated CFG is structurized (P0–P7), emitted through the
//! builder (P8), and run on the interpreter before and after single-store promotion (P9); each
//! invocation's block trace, end, and return value equal the testkit interpreter's run of the
//! source CFG on the same seed.
//!
//! **Convergence** ([`Scheme::CONVERGENCE`]): 32 invocations, one subgroup, run each function
//! under the interpreter's maximal reconvergence (the module declares `MaximallyReconvergesKHR`),
//! and every block execution records `OpGroupNonUniformBallot(true)`, the lanes executing it
//! together. Per original block, the multiset of those lane sets equals the multiset of its
//! ideal tangles ([`tangle`]), computed from each lane's run of the source alone. Irreducible
//! graphs have no source-level iterations and are not compared.
//!
//! The emitted blocks compute the testkit's decision scheme in `u64` SPIR-V: executing original
//! block `b` appends `b` to the invocation's trace and mixes it into a `Private` state; every
//! condition, selector, and return value is the testkit's hash of that state. Each run records
//! at most [`EVENTS`] blocks and then returns (the testkit's prefix cutoff), so graphs that loop
//! forever on some seeds are testable. Demoted values and phis go through P8's variables: every
//! block's value crosses blocks and every join has a phi. A join checks that its phi variable
//! holds the value its predecessor stored (the state before the join), and each block records
//! the value its own demoted variable held from its previous execution, which the reference
//! predicts; both fail if a variable is dropped, aliased, or promoted wrongly.
//!
//! **Predication** ([`Scheme::PREDICATED`]): the tangled blocks of a void graph are workgroup
//! synchronizations and the function is emitted by P8's predicated walk. 32 invocations run in four
//! 8-lane subgroups; a tangled block records, instead of a subgroup ballot, the invocations live in
//! it, gathered through workgroup memory between workgroup barriers (as the emulated simdgroup tier
//! exchanges lanes), and the walk's loop votes are the same workgroup gather. Each invocation's
//! trace and end equal the source run, the live sets of every tangled block equal its ideal
//! tangles, and the interpreter traps any barrier the invocations reach at different points.

use super::*;
use crate::air::ir::{BlockId as AirBlock, ValueId};
use crate::budget::{Budget, Meter};
use crate::error::{Error, Feature};
use crate::spirv::ids::{DemoteTy, FuncId, ScalarTy, VarId};
use crate::spirv::ir::AtomicOp;
use crate::spirv::testing::{check_valid, committed_profiles};
use crate::structurize::emit::{emit, CodeEmitter, Poll, StructVars};
use crate::structurize::promote::promote_single_store;
use crate::structurize::testkit::generate::{self, Rng};
use crate::structurize::testkit::hash::{key, Hash, EXEC_MUL, MIX};
use crate::structurize::testkit::interp::{run_test_cfg, End};
use crate::structurize::testkit::tangle::{self, Tangles};
use crate::structurize::testkit::{TestCfg, TestKind, SELECTOR_RANGE};
use crate::structurize::{prepare, structurize};

/// Original blocks each invocation may execute.
const EVENTS: u32 = 24;
/// Words before the trace: count, end, return value (low, high), phi mismatches.
const HEAD: u32 = 5;
/// Words per event: the block, its own variable's previous value, and the ballot.
const EVENT_WORDS: u32 = 3;
/// Words per (graph, invocation): the head, then the events.
const STRIDE: u32 = HEAD + EVENT_WORDS * EVENTS;
/// Phi `ValueId`s are `PHI_BASE + block`, demoted values `ValueId(block)`.
const PHI_BASE: u32 = 1 << 20;

const END_RETURN: u32 = 1;
const END_RETURN_VALUE: u32 = 2;
const END_UNREACHABLE: u32 = 3;
const END_CUTOFF: u32 = 4;

/// How the modules run: invocations (seeds) per graph, all in one 32-lane subgroup, and whether
/// each block execution records its ballot.
#[derive(Clone, Copy, Debug)]
struct Scheme {
    lanes: u32,
    ballots: bool,
    /// The walk is predicated: tangled blocks synchronize the workgroup.
    predicated: bool,
    /// Predicated only: every invocation runs on its graph's first seed, so every condition is
    /// workgroup uniform, and the emitter reports as uniform the conditions of the blocks this
    /// holds for and the blocks [`claimed_uniformity`] derives from them (otherwise nothing is
    /// reported uniform).
    uniform: Option<fn(u32) -> bool>,
}

impl Scheme {
    /// Trace equivalence.
    const TRACE: Scheme = Scheme {
        lanes: 8,
        ballots: false,
        predicated: false,
        uniform: None,
    };
    /// Trace and convergence equivalence at the 32-lane simdgroup width.
    const CONVERGENCE: Scheme = Scheme {
        lanes: 32,
        ballots: true,
        predicated: false,
        uniform: None,
    };
    /// Trace equivalence of the predicated walk, and the live sets of its synchronizations
    /// against the ideal tangles, at 32 invocations in 8-lane subgroups.
    const PREDICATED: Scheme = Scheme {
        lanes: 32,
        ballots: true,
        predicated: true,
        uniform: None,
    };

    /// [`Scheme::PREDICATED`] with uniform runs whose conditions of the blocks `claim` holds for
    /// are reported uniform: the walk takes real branches, real loop exits, and real returns
    /// there, and predicates the rest.
    const fn predicated_uniform(claim: fn(u32) -> bool) -> Scheme {
        Scheme {
            uniform: Some(claim),
            ..Scheme::PREDICATED
        }
    }

    /// The seed of invocation `lane` of graph `j`.
    fn seed(self, j: u32, lane: u32) -> u64 {
        seed(j, if self.uniform.is_some() { 0 } else { lane })
    }
}

/// The blocks `Uniformity` would prove workgroup uniform in `g` if exactly the conditions of the
/// blocks `claim` holds for were uniform: a block is divergent when it is control dependent (on the
/// post-dominator tree) on a branch whose condition is not claimed or whose block is divergent.
fn claimed_uniformity(g: &TestCfg, claim: impl Fn(u32) -> bool) -> Vec<bool> {
    let n = g.blocks.len();
    // Post-dominator sets (every block reaches a return: `every_block_returns`).
    let mut pdom: Vec<Vec<bool>> = vec![vec![true; n]; n];
    let mut changed = true;
    while changed {
        changed = false;
        for (b, blk) in g.blocks.iter().enumerate() {
            let mut next = vec![blk.succ.is_empty(); n];
            if !blk.succ.is_empty() {
                next = vec![true; n];
                for s in &blk.succ {
                    for (x, p) in pdom[s.raw() as usize].iter().enumerate() {
                        next[x] = next[x] && *p;
                    }
                }
            }
            next[b] = true;
            if next != pdom[b] {
                pdom[b] = next;
                changed = true;
            }
        }
    }
    // `x` is control dependent on branch `p` when it post-dominates a successor of `p` and does
    // not strictly post-dominate `p`.
    let dependent = |x: usize, p: usize| {
        g.blocks[p]
            .succ
            .iter()
            .any(|s| pdom[s.raw() as usize][x] && (x == p || !pdom[p][x]))
    };
    let mut divergent = vec![false; n];
    let mut changed = true;
    while changed {
        changed = false;
        for p in 0..n {
            let mut succ: Vec<u32> = g.blocks[p].succ.iter().map(|s| s.raw()).collect();
            succ.sort_unstable();
            succ.dedup();
            if succ.len() < 2 || (claim(p as u32) && !divergent[p]) {
                continue;
            }
            for (x, d) in divergent.iter_mut().enumerate() {
                if !*d && dependent(x, p) {
                    *d = true;
                    changed = true;
                }
            }
        }
    }
    divergent.iter().map(|d| !d).collect()
}

/// Module-wide state shared by the emitted functions.
struct Glob {
    u32_: TypeId,
    u64_: TypeId,
    /// `Private` pointers: the `u64` hash, the `u32` event count, and the `u32` word offset of
    /// the current (graph, invocation) record.
    state: Value,
    count: Value,
    base: Value,
    buf: GlobalId,
    /// `Private` `u32`: the invocation's index in the workgroup.
    lane: Value,
    /// A `Workgroup` `u32` the predicated scheme gathers live invocations in.
    shared: Value,
}

fn k32(fb: &mut FunctionBuilder<'_>, g: &Glob, v: u32) -> BResult<Value> {
    let c = fb.module().constant(Constant::Scalar {
        ty: g.u32_,
        bits: ConstBits(u64::from(v)),
    })?;
    fb.module().const_value(c)
}

fn k64(fb: &mut FunctionBuilder<'_>, g: &Glob, v: u64) -> BResult<Value> {
    let c = fb.module().constant(Constant::Scalar {
        ty: g.u64_,
        bits: ConstBits(v),
    })?;
    fb.module().const_value(c)
}

/// The testkit's `mix` in SPIR-V.
fn mix_ir(fb: &mut FunctionBuilder<'_>, g: &Glob, z: Value) -> BResult<Value> {
    let c0 = k64(fb, g, MIX[0])?;
    let mut z = fb.binary(BinOp::IAdd, z, c0)?;
    for (shift, mul) in [(30, MIX[1]), (27, MIX[2])] {
        let s = k64(fb, g, shift)?;
        let hi = fb.binary(BinOp::ShiftRightLogical, z, s)?;
        let x = fb.binary(BinOp::BitwiseXor, z, hi)?;
        let m = k64(fb, g, mul)?;
        z = fb.binary(BinOp::IMul, x, m)?;
    }
    let s = k64(fb, g, 31)?;
    let hi = fb.binary(BinOp::ShiftRightLogical, z, s)?;
    fb.binary(BinOp::BitwiseXor, z, hi)
}

/// The testkit's hash of the current state and value `v`.
fn value_ir(fb: &mut FunctionBuilder<'_>, g: &Glob, v: ValueId) -> BResult<Value> {
    let h = fb.load(g.state, MemAccess::default())?;
    let k = k64(fb, g, key(v))?;
    let x = fb.binary(BinOp::BitwiseXor, h, k)?;
    mix_ir(fb, g, x)
}

/// A pointer to word `off` (a `u32` value) of the current record.
fn word(fb: &mut FunctionBuilder<'_>, g: &Glob, off: Value) -> BResult<Value> {
    let base = fb.load(g.base, MemAccess::default())?;
    let i = fb.binary(BinOp::IAdd, base, off)?;
    let zero = k32(fb, g, 0)?;
    let b = fb.module().global_value(g.buf)?;
    fb.access_chain(b, &[zero, i])
}

fn put(fb: &mut FunctionBuilder<'_>, g: &Glob, off: u32, v: Value) -> BResult<()> {
    let o = k32(fb, g, off)?;
    let p = word(fb, g, o)?;
    fb.store(p, v, MemAccess::default())
}

/// The low and high words of a `u64`.
fn halves(fb: &mut FunctionBuilder<'_>, g: &Glob, v: Value) -> BResult<(Value, Value)> {
    let lo = fb.convert(ConvOp::UConvert, v, g.u32_)?;
    let s = k64(fb, g, 32)?;
    let hi = fb.binary(BinOp::ShiftRightLogical, v, s)?;
    let hi = fb.convert(ConvOp::UConvert, hi, g.u32_)?;
    Ok((lo, hi))
}

/// `if cond { end = code; return }`, leaving the merge open.
fn end_if(fb: &mut FunctionBuilder<'_>, g: &Glob, cond: Value, code: u32) -> BResult<()> {
    let arm: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let c = k32(fb, g, code)?;
        put(fb, g, 1, c)?;
        fb.ret(None)?;
        Ok(ArmEnd::Exits)
    });
    fb.if_(&mut (), cond, arm, None)?;
    Ok(())
}

/// The invocations of the workgroup where `pred` holds, as a lane mask: every invocation
/// executes it together (workgroup barriers), as the emulated tier's vote does.
fn gather(fb: &mut FunctionBuilder<'_>, g: &Glob, pred: Value) -> BResult<Value> {
    let relaxed = Semantics::RELAXED;
    let zero = k32(fb, g, 0)?;
    fb.atomic_store(g.shared, Scope::Workgroup, relaxed, zero)?;
    super::wg_barrier(fb)?;
    let one = k32(fb, g, 1)?;
    let lane = fb.load(g.lane, MemAccess::default())?;
    let bit = fb.binary(BinOp::ShiftLeftLogical, one, lane)?;
    let bit = fb.select(pred, bit, zero)?;
    fb.atomic(AtomicOp::Or, g.shared, Scope::Workgroup, relaxed, bit)?;
    super::wg_barrier(fb)?;
    let x = fb.atomic_load(g.shared, Scope::Workgroup, relaxed)?;
    super::wg_barrier(fb)?;
    Ok(x)
}

/// A [`CodeEmitter`] running the testkit's decision scheme and recording the trace (and each
/// execution's ballot when `ballots`). With `predicated`, the tangled blocks are
/// synchronizations of the predicated walk, and their ballots are the live invocations.
struct TraceEmitter<'a> {
    g: &'a TestCfg,
    glob: &'a Glob,
    vars: Option<StructVars>,
    ballots: bool,
    predicated: bool,
    /// By block: the block is reported workgroup uniform, and its condition or selector.
    uniform_blocks: Vec<bool>,
    uniform_conds: Vec<bool>,
}

impl<'a> TraceEmitter<'a> {
    /// The emitter of graph `g` (its values added) under `scheme`.
    fn new(g: &'a TestCfg, glob: &'a Glob, scheme: Scheme) -> TraceEmitter<'a> {
        let (uniform_blocks, uniform_conds) = match scheme.uniform {
            Some(claim) => (
                claimed_uniformity(g, claim),
                (0..g.blocks.len() as u32).map(claim).collect(),
            ),
            None => (Vec::new(), Vec::new()),
        };
        TraceEmitter {
            g,
            glob,
            vars: None,
            ballots: scheme.ballots,
            predicated: scheme.predicated,
            uniform_blocks,
            uniform_conds,
        }
    }

    fn vars(&self) -> &StructVars {
        self.vars
            .as_ref()
            .expect("P8 binds the variables before any block")
    }

    fn tangled(&self, b: AirBlock) -> bool {
        self.g
            .blocks
            .get(b.raw() as usize)
            .is_some_and(|x| x.tangled)
    }

    /// The predicated scheme's ballot of block `b`, executed by every invocation: the invocations
    /// live in it (`live`, else all) when it synchronizes, else none.
    fn live_ballot(
        &self,
        b: AirBlock,
        live: Option<Value>,
        fb: &mut FunctionBuilder<'_>,
    ) -> BResult<Value> {
        let g = self.glob;
        if !self.tangled(b) {
            return k32(fb, g, 0);
        }
        let pred = match live {
            Some(l) => l,
            None => {
                let z = k32(fb, g, 0)?;
                fb.binary(BinOp::IEqual, z, z)?
            }
        };
        gather(fb, g, pred)
    }

    /// Execute block `b` for this invocation: record it (with `ballot` when `ballots`), mix it
    /// into the state, and store the variables. A predicated run keeps executing past
    /// [`EVENTS`], writing its last event slot, so every invocation reaches every
    /// synchronization; such a graph is not compared.
    fn record(
        &mut self,
        b: AirBlock,
        ballot: Option<Value>,
        fb: &mut FunctionBuilder<'_>,
    ) -> BResult<()> {
        let g = self.glob;
        let a = MemAccess::default();
        let cnt = fb.load(g.count, a)?;
        let h = fb.load(g.state, a)?;
        // Record `b` and its own variable's value from its previous execution.
        let at = if self.predicated {
            let limit = k32(fb, g, EVENTS)?;
            let inside = fb.binary(BinOp::ULessThan, cnt, limit)?;
            let last = k32(fb, g, EVENTS - 1)?;
            fb.select(inside, cnt, last)?
        } else {
            cnt
        };
        let per = k32(fb, g, EVENT_WORDS)?;
        let slot = fb.binary(BinOp::IMul, at, per)?;
        let head = k32(fb, g, HEAD)?;
        let slot = fb.binary(BinOp::IAdd, slot, head)?;
        if let Some(ballot) = ballot {
            let two = k32(fb, g, 2)?;
            let slot2 = fb.binary(BinOp::IAdd, slot, two)?;
            let p = word(fb, g, slot2)?;
            fb.store(p, ballot, a)?;
        }
        let p = word(fb, g, slot)?;
        let id = k32(fb, g, b.raw())?;
        fb.store(p, id, a)?;
        let own = self.vars().demoted_var(ValueId::new(b.raw()));
        let prev = match own {
            Some(var) => {
                let ptr = fb.var_ptr(var)?;
                let v = fb.load(ptr, a)?;
                fb.convert(ConvOp::UConvert, v, g.u32_)?
            }
            None => k32(fb, g, 0)?,
        };
        let one = k32(fb, g, 1)?;
        let slot1 = fb.binary(BinOp::IAdd, slot, one)?;
        let p = word(fb, g, slot1)?;
        fb.store(p, prev, a)?;
        let cnt1 = fb.binary(BinOp::IAdd, cnt, one)?;
        fb.store(g.count, cnt1, a)?;

        // A join's phi holds what its predecessor stored: the state before this block.
        if let Some(var) = self.vars().phi_var(ValueId::new(PHI_BASE + b.raw())) {
            let ptr = fb.var_ptr(var)?;
            let phi = fb.load(ptr, a)?;
            let differs = fb.binary(BinOp::INotEqual, phi, h)?;
            let zero = k32(fb, g, 0)?;
            let later = fb.binary(BinOp::INotEqual, cnt, zero)?;
            let bad = fb.binary(BinOp::LogicalAnd, differs, later)?;
            let add = fb.select(bad, one, zero)?;
            let four = k32(fb, g, 4)?;
            let p = word(fb, g, four)?;
            let n = fb.load(p, a)?;
            let n = fb.binary(BinOp::IAdd, n, add)?;
            fb.store(p, n, a)?;
        }

        // Execute: mix `b` into the state.
        let k = k64(fb, g, u64::from(b.raw()).wrapping_mul(EXEC_MUL))?;
        let x = fb.binary(BinOp::BitwiseXor, h, k)?;
        let h1 = mix_ir(fb, g, x)?;
        fb.store(g.state, h1, a)?;
        let block = &self.g.blocks[b.raw() as usize];
        let stores: Vec<VarId> = own
            .into_iter()
            .chain(
                block
                    .succ
                    .iter()
                    .filter_map(|s| self.vars().phi_var(ValueId::new(PHI_BASE + s.raw()))),
            )
            .collect();
        for var in stores {
            let ptr = fb.var_ptr(var)?;
            fb.store(ptr, h1, a)?;
        }
        if block.kind == TestKind::Unreachable && !self.predicated {
            // Reaching `unreachable` is an observable end; the `OpUnreachable` P8 emits after
            // this block is never executed.
            let t = fb.binary(BinOp::IEqual, cnt1, cnt1)?;
            end_if(fb, g, t, END_UNREACHABLE)?;
        }
        Ok(())
    }
}

impl CodeEmitter for TraceEmitter<'_> {
    fn var_type(&mut self, ty: &DemoteTy, _m: &mut ModuleBuilder) -> BResult<TypeId> {
        assert_eq!(ty, &DemoteTy::Scalar(ScalarTy::Int { bits: 64 }));
        Ok(self.glob.u64_)
    }

    fn bind(&mut self, vars: &StructVars) -> BResult<()> {
        self.vars = Some(vars.clone());
        Ok(())
    }

    fn block(
        &mut self,
        b: AirBlock,
        fb: &mut FunctionBuilder<'_>,
        _poll: &mut Poll<'_>,
    ) -> BResult<()> {
        let g = self.glob;
        if self.predicated {
            let ballot = self.live_ballot(b, None, fb)?;
            return self.record(b, Some(ballot), fb);
        }
        // The prefix cutoff: the testkit stops before executing an event past the limit.
        let cnt = fb.load(g.count, MemAccess::default())?;
        let limit = k32(fb, g, EVENTS)?;
        let full = fb.binary(BinOp::IEqual, cnt, limit)?;
        end_if(fb, g, full, END_CUTOFF)?;
        let ballot = if self.ballots {
            // The lanes executing this block together (their low word: one 32-lane subgroup).
            let t = fb.binary(BinOp::IEqual, cnt, cnt)?;
            let ballot = fb.group_ballot(t)?;
            Some(fb.composite_extract(ballot, &[0])?)
        } else {
            None
        };
        self.record(b, ballot, fb)
    }

    fn predicated(&self) -> bool {
        self.predicated
    }

    fn syncs(&self, b: AirBlock) -> bool {
        self.predicated && self.tangled(b)
    }

    fn uniform_block(&self, b: AirBlock) -> bool {
        self.uniform_blocks
            .get(b.raw() as usize)
            .copied()
            .unwrap_or(false)
    }

    fn uniform_cond(&self, v: ValueId) -> bool {
        self.uniform_conds
            .get(v.raw() as usize)
            .copied()
            .unwrap_or(false)
    }

    fn block_live(
        &mut self,
        b: AirBlock,
        live: Option<Value>,
        fb: &mut FunctionBuilder<'_>,
        poll: &mut Poll<'_>,
    ) -> BResult<()> {
        let Some(l) = live else {
            return self.block(b, fb, poll);
        };
        let ballot = self.live_ballot(b, Some(l), fb)?;
        fb.if_(
            self,
            l,
            Box::new(move |fb, e: &mut Self| {
                e.record(b, Some(ballot), fb)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            }),
            None,
        )?;
        Ok(())
    }

    fn vote(&mut self, live: Value, fb: &mut FunctionBuilder<'_>) -> BResult<Value> {
        let g = self.glob;
        let x = gather(fb, g, live)?;
        let zero = k32(fb, g, 0)?;
        fb.binary(BinOp::INotEqual, x, zero)
    }

    fn cond(&mut self, v: ValueId, fb: &mut FunctionBuilder<'_>) -> BResult<Value> {
        let g = self.glob;
        let r = value_ir(fb, g, v)?;
        let one = k64(fb, g, 1)?;
        let bit = fb.binary(BinOp::BitwiseAnd, r, one)?;
        fb.binary(BinOp::IEqual, bit, one)
    }

    fn selector(&mut self, v: ValueId, fb: &mut FunctionBuilder<'_>) -> BResult<Value> {
        let g = self.glob;
        let r = value_ir(fb, g, v)?;
        let range = k64(fb, g, SELECTOR_RANGE)?;
        let s = fb.binary(BinOp::UMod, r, range)?;
        fb.convert(ConvOp::UConvert, s, g.u32_)
    }

    fn on_return(&mut self, v: Option<ValueId>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
        let g = self.glob;
        let code = match v {
            Some(v) => {
                let r = value_ir(fb, g, v)?;
                let (lo, hi) = halves(fb, g, r)?;
                put(fb, g, 2, lo)?;
                put(fb, g, 3, hi)?;
                END_RETURN_VALUE
            }
            None => END_RETURN,
        };
        let c = k32(fb, g, code)?;
        put(fb, g, 1, c)?;
        fb.ret(None)
    }
}

/// `g` with a `u64` value per block crossing blocks and a phi per join.
fn with_values(g: &TestCfg) -> TestCfg {
    let mut g = g.clone();
    let int = DemoteTy::Scalar(ScalarTy::Int { bits: 64 });
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

/// The seed of invocation `lane` of graph `j`.
fn seed(j: u32, lane: u32) -> u64 {
    u64::from(j) * 0x1_0000 + u64::from(lane) + 0x7ace
}

/// What one module holds: the modules before and after P9, the indices of the graphs
/// `structurize` accepted (one function each), and each refused graph's index with its error.
struct Built {
    before: TypedModule,
    after: TypedModule,
    ran: Vec<usize>,
    refused: Vec<(usize, Error)>,
}

/// One module holding a function per graph `structurize` accepts, before and after P9. The entry
/// runs every function once per invocation, each on its own seed and record.
fn build(graphs: &[TestCfg], scheme: Scheme) -> Built {
    let mut meter = Meter::new(&Budget::default());
    let mut b = ModuleBuilder::new();
    let glob = new_glob(&mut b);
    let u64_ = glob.u64_;
    let void = b.ty(Type::Void).expect("void");
    let mut funcs: Vec<(usize, FuncId)> = Vec::new();
    let mut refused = Vec::new();
    for (i, g) in graphs.iter().enumerate() {
        let g = with_values(g);
        let s = match structurize(&g, &mut meter) {
            Ok(s) => s,
            Err(e) => {
                refused.push((i, e));
                continue;
            }
        };
        let mut f = b.function(void).unwrap();
        let mut e = TraceEmitter::new(&g, &glob, scheme);
        emit(&s, &mut f, &mut e, &mut meter)
            .unwrap_or_else(|err| panic!("P8 refused {err:?} for {g:?}"));
        funcs.push((i, f.finish().unwrap()));
        s.release(&mut meter);
    }
    let mut f = b.function(void).unwrap();
    let gid = gid(&mut f);
    let gid64 = f.convert(ConvOp::UConvert, gid, u64_).unwrap();
    let a = MemAccess::default();
    // One workgroup: the global index is the lane.
    f.store(glob.lane, gid, a).unwrap();
    for (j, (_, id)) in funcs.iter().enumerate() {
        let j = j as u32;
        let s0 = k64(&mut f, &glob, seed(j, 0)).unwrap();
        let s = if scheme.uniform.is_some() {
            s0
        } else {
            f.binary(BinOp::IAdd, gid64, s0).unwrap()
        };
        let h = mix_ir(&mut f, &glob, s).unwrap();
        f.store(glob.state, h, a).unwrap();
        let zero = k32(&mut f, &glob, 0).unwrap();
        f.store(glob.count, zero, a).unwrap();
        let first = k32(&mut f, &glob, j * scheme.lanes).unwrap();
        let rec = f.binary(BinOp::IAdd, first, gid).unwrap();
        let stride = k32(&mut f, &glob, STRIDE).unwrap();
        let rec = f.binary(BinOp::IMul, rec, stride).unwrap();
        f.store(glob.base, rec, a).unwrap();
        f.call(*id, &[]).unwrap();
        let n = f.load(glob.count, a).unwrap();
        put(&mut f, &glob, 0, n).unwrap();
    }
    f.ret(None).unwrap();
    let entry = f.finish().unwrap();
    let mut modes = vec![ExecMode::LocalSize {
        x: scheme.lanes,
        y: 1,
        z: 1,
    }];
    if scheme.ballots && !scheme.predicated {
        modes.push(ExecMode::MaximallyReconverges);
    }
    b.entry(ExecModel::GLCompute, entry, modes).unwrap();
    let before = b.finalize().unwrap();
    let mut after = before.clone();
    promote_single_store(&mut after, &mut meter).unwrap();
    assert_eq!(meter.ledger(), 0, "P0-P9 left ledger bytes behind");
    Built {
        before,
        after,
        ran: funcs.into_iter().map(|(i, _)| i).collect(),
        refused,
    }
}

/// The module-wide state of a trace module, declared in `b`.
fn new_glob(b: &mut ModuleBuilder) -> Glob {
    let buf = buffer(b, 0);
    let u32_ = u32_ty(b);
    let u64_ = b
        .ty(Type::Int {
            width: 64,
            signed: false,
        })
        .expect("u64");
    let state = b.plain_global(StorageClass::Private, u64_, None).unwrap();
    let count = b.plain_global(StorageClass::Private, u32_, None).unwrap();
    let base = b.plain_global(StorageClass::Private, u32_, None).unwrap();
    let lane = b.plain_global(StorageClass::Private, u32_, None).unwrap();
    let shared = b.plain_global(StorageClass::Workgroup, u32_, None).unwrap();
    Glob {
        u32_,
        u64_,
        state: b.global_value(state).unwrap(),
        count: b.global_value(count).unwrap(),
        base: b.global_value(base).unwrap(),
        buf,
        lane: b.global_value(lane).unwrap(),
        shared: b.global_value(shared).unwrap(),
    }
}

/// What the testkit predicts for one record: the words the emitted code writes.
fn expected(g: &TestCfg, seed: u64) -> Vec<u32> {
    let run = run_test_cfg(g, seed, EVENTS as usize);
    let mut w = vec![0u32; STRIDE as usize];
    w[0] = run.trace.len() as u32;
    let (code, ret) = match run.end {
        End::Returned(None) => (END_RETURN, 0),
        End::Returned(Some(r)) => (END_RETURN_VALUE, r),
        End::Unreachable => (END_UNREACHABLE, 0),
        End::Cutoff => (END_CUTOFF, 0),
    };
    w[1] = code;
    w[2] = ret as u32;
    w[3] = (ret >> 32) as u32;
    // Replay the hash to predict each block's previous value (the testkit's state scheme).
    let mut h = Hash::seeded(seed);
    let mut last: Vec<Option<u64>> = vec![None; g.blocks.len()];
    for (e, b) in run.trace.iter().enumerate() {
        let at = (HEAD as usize) + (EVENT_WORDS as usize) * e;
        w[at] = b.raw();
        w[at + 1] = last[b.raw() as usize].map_or(0, |v| v as u32);
        h.exec(*b);
        last[b.raw() as usize] = Some(h.raw());
    }
    w
}

/// A record's first-execution slots hold whatever the variable held before any store; only a
/// block's later executions are compared.
fn mask_first_executions(g: &TestCfg, rec: &mut [u32]) {
    let n = rec[0].min(EVENTS) as usize;
    let mut seen = vec![false; g.blocks.len()];
    for e in 0..n {
        let at = (HEAD as usize) + (EVENT_WORDS as usize) * e;
        let b = rec[at] as usize;
        if b < seen.len() && !seen[b] {
            seen[b] = true;
            rec[at + 1] = 0;
        }
    }
}

/// How a set of graphs fared: how many ran like the source, how many of those converged like it
/// (with ballots) and in how many block executions the lanes had diverged (ran with a strict
/// subset of the lanes), and each refused graph's index with the structurizer's error.
#[derive(Debug, Default)]
struct Checked {
    ran: usize,
    converged: usize,
    diverged: usize,
    refused: Vec<(usize, Error)>,
}

/// The per-block multisets of lane sets the ballots of `lanes` records (from word `at` on)
/// name, for the blocks `keep` holds for. Each execution is recorded once by every lane in it,
/// which checks that each lane is in its own ballot.
fn recorded_tangles(w: &[u32], at: usize, lanes: u32, keep: &dyn Fn(u32) -> bool) -> Tangles {
    let mut seen: std::collections::BTreeMap<(u32, u32), u32> = Default::default();
    for lane in 0..lanes {
        let rec = &w[at + (lane * STRIDE) as usize..][..STRIDE as usize];
        for e in 0..rec[0].min(EVENTS) as usize {
            let ev = &rec[(HEAD as usize) + (EVENT_WORDS as usize) * e..];
            if !keep(ev[0]) {
                continue;
            }
            assert_ne!(
                ev[2] & (1 << lane),
                0,
                "lane {lane} is not in its own ballot"
            );
            *seen.entry((ev[0], ev[2])).or_default() += 1;
        }
    }
    let mut out = Tangles::new();
    for ((block, ballot), n) in seen {
        let size = ballot.count_ones();
        assert_eq!(
            n % size,
            0,
            "block {block}: ballot {ballot:#x} recorded {n} times"
        );
        let set: std::collections::BTreeSet<usize> =
            (0..32).filter(|l| ballot & (1 << l) != 0).collect();
        let sets = out.entry(AirBlock::new(block)).or_default();
        sets.extend(std::iter::repeat_n(set, (n / size) as usize));
    }
    tangle::sorted(out)
}

/// Structurize, emit, and promote `graphs`, validate both modules (L3), run each on the
/// interpreter, and compare every record with the testkit; with ballots, compare every reducible
/// graph's recorded tangles with its ideal tangles. Predicated, only the tangled blocks' tangles
/// are compared, the modules run in 8-lane subgroups in both scheduling orders, and a graph some
/// invocation of which runs past [`EVENTS`] runs (every barrier checked) but is not compared.
fn check_batch(graphs: &[TestCfg], scheme: Scheme) -> Checked {
    let Built {
        before,
        after,
        ran,
        refused,
    } = build(graphs, scheme);
    let lanes = scheme.lanes;
    let mut diverged = 0;
    // The ideal tangles of each function that has them.
    let ideal: Vec<Option<Tangles>> = if scheme.ballots {
        ran.iter()
            .enumerate()
            .map(|(j, gi)| {
                let g = &graphs[*gi];
                let mut m = Meter::new(&Budget::default());
                let p = prepare(g, &mut m).expect("prepare after structurize");
                let seeds: Vec<u64> = (0..lanes).map(|l| scheme.seed(j as u32, l)).collect();
                let mut t = tangle::ideal(g, &p, &seeds, EVENTS as usize).map(|i| i.tangles);
                p.release(&mut m);
                if let (Some(t), true) = (&mut t, scheme.predicated) {
                    t.retain(|b, _| g.blocks[b.raw() as usize].tangled);
                }
                t
            })
            .collect()
    } else {
        Vec::new()
    };
    // Predicated graphs some invocation of which reaches the prefix cutoff.
    let cut: Vec<bool> = ran
        .iter()
        .enumerate()
        .map(|(j, gi)| {
            scheme.predicated
                && (0..lanes).any(|l| {
                    run_test_cfg(&graphs[*gi], scheme.seed(j as u32, l), EVENTS as usize).end
                        == End::Cutoff
                })
        })
        .collect();
    let configs: Vec<Config> = if scheme.predicated {
        [Order::Forward, Order::Reverse]
            .into_iter()
            .map(|order| Config {
                step_limit: 1 << 30,
                subgroup: SubgroupSize::W8,
                order,
                ..Config::default()
            })
            .collect()
    } else {
        vec![Config {
            step_limit: 1 << 34,
            ..Config::default()
        }]
    };
    let profile = committed_profiles()
        .expect("committed profiles")
        .into_iter()
        .find(|(name, _)| name == "lavapipe")
        .expect("committed lavapipe profile")
        .1;
    let tangled = |g: &TestCfg, b: u32| g.blocks.get(b as usize).is_some_and(|x| x.tangled);
    for (m, when) in [(&before, "P8"), (&after, "P9")] {
        check_valid(m.clone(), &profile)
            .unwrap_or_else(|e| panic!("{when} output rejected ({} graphs): {e}", graphs.len()));
        let words_n = ran.len() * (lanes * STRIDE) as usize;
        for (k, cfg) in configs.iter().enumerate() {
            let out = run(m, cfg, buffers(&[words_n]))
                .unwrap_or_else(|e| panic!("{when} {:?}: {e}", cfg.order));
            let w = words(&out.buffers[0]);
            for (j, gi) in ran.iter().enumerate() {
                if cut[j] {
                    continue;
                }
                let g = &graphs[*gi];
                let first = (j as u32 * lanes * STRIDE) as usize;
                if let Some(Some(want)) = ideal.get(j) {
                    let keep = |b: u32| !scheme.predicated || tangled(g, b);
                    assert_eq!(
                        &recorded_tangles(&w, first, lanes, &keep),
                        want,
                        "{when} {:?}: graph {:?} (tangled {:?}): the tangles differ from the \
                         ideal ones",
                        cfg.order,
                        g.successor_lists(),
                        (0..g.blocks.len() as u32)
                            .filter(|&b| tangled(g, b))
                            .collect::<Vec<_>>()
                    );
                    if when == "P8" && k == 0 {
                        diverged += want
                            .values()
                            .flatten()
                            .filter(|t| t.len() < lanes as usize)
                            .count();
                    }
                }
                for lane in 0..lanes {
                    let at = first + (lane * STRIDE) as usize;
                    let mut got = w[at..at + STRIDE as usize].to_vec();
                    // The ballots were compared above; a lane's own run does not predict them.
                    for e in 0..EVENTS as usize {
                        got[(HEAD as usize) + (EVENT_WORDS as usize) * e + 2] = 0;
                    }
                    let mut want = expected(g, scheme.seed(j as u32, lane));
                    mask_first_executions(g, &mut got);
                    mask_first_executions(g, &mut want);
                    assert_eq!(
                        got,
                        want,
                        "{when} {:?}: graph {:?}, lane {lane}: the emitted module's record \
                         differs from the testkit's",
                        cfg.order,
                        g.successor_lists()
                    );
                }
            }
        }
    }
    Checked {
        ran: ran.len(),
        converged: ideal
            .iter()
            .zip(&cut)
            .filter(|(t, cut)| t.is_some() && !**cut)
            .count(),
        diverged,
        refused,
    }
}

/// [`check_batch`] over `graphs` in modules of at most `size` functions; refused indices are into
/// `graphs`.
fn check_all(graphs: &[TestCfg], size: usize, scheme: Scheme) -> Checked {
    let mut all = Checked::default();
    for (k, batch) in graphs.chunks(size).enumerate() {
        let c = check_batch(batch, scheme);
        all.ran += c.ran;
        all.converged += c.converged;
        all.diverged += c.diverged;
        all.refused
            .extend(c.refused.into_iter().map(|(i, e)| (k * size + i, e)));
    }
    all
}

/// [`check_all`], requiring every graph to run.
fn check_all_run(graphs: &[TestCfg], size: usize, scheme: Scheme) -> Checked {
    let c = check_all(graphs, size, scheme);
    assert!(c.refused.is_empty(), "refused: {:?}", c.refused);
    assert_eq!(c.ran, graphs.len());
    c
}

fn canonical(n: u32, choices: Vec<Vec<u32>>) -> Vec<TestCfg> {
    let mut out = Vec::new();
    generate::canonical(n, &choices, |g| out.push(g.clone()));
    out
}

#[test]
fn exhaustive_small_graphs_run_like_the_source() {
    let graphs: Vec<TestCfg> = (1..=3)
        .flat_map(|n| canonical(n, generate::out_degree_two_choices(n)))
        .chain((1..=3).flat_map(|n| canonical(n, generate::switch_choices(n))))
        .collect();
    check_all_run(&graphs, 200, Scheme::TRACE);
}

#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "four-block graphs: runs in release (G-rel: cargo test --release -- structured_trace)"
)]
fn exhaustive_four_block_graphs_run_like_the_source() {
    let graphs = canonical(4, generate::out_degree_two_choices(4));
    check_all_run(&graphs, 400, Scheme::TRACE);
}

/// The generated families, each graph named by the generator call that made it.
fn generated_families() -> Vec<(String, TestCfg)> {
    let mut named: Vec<(String, TestCfg)> = Vec::new();
    for seed in 0..60 {
        let depth = 2 + (seed % 4) as u32;
        named.push((
            format!("structured({seed}, {depth}, 0)"),
            generate::structured(seed, depth, 0),
        ));
        let (depth, crossings) = (3 + (seed % 3) as u32, 1 + (seed % 3) as u32);
        named.push((
            format!("structured({seed}, {depth}, {crossings})"),
            generate::structured(seed, depth, crossings),
        ));
    }
    let mut rng = Rng::new(0x7ace_12b0);
    for _ in 0..120 {
        let n = 2 + rng.below(24) as u32;
        let seed = rng.next_u64();
        named.push((
            format!("dag_with_back_edges({n}, {seed:#x})"),
            generate::dag_with_back_edges(n, seed),
        ));
    }
    for k in 1..=6 {
        named.push((
            format!("irreducible_ladder({k})"),
            generate::irreducible_ladder(k),
        ));
        named.push((
            format!("break_tail_ladder({k})"),
            generate::break_tail_ladder(k),
        ));
        named.push((
            format!("nested_irreducible({k})"),
            generate::nested_irreducible(k),
        ));
    }
    named
}

/// Check `named` under `scheme`: the refused set must be exactly empty (every family nests
/// irreducible loops within the depth cap and tangles nothing), listed by generator call if not.
fn check_named(named: &[(String, TestCfg)], size: usize, scheme: Scheme) -> Checked {
    let graphs: Vec<TestCfg> = named.iter().map(|(_, g)| g.clone()).collect();
    let c = check_all(&graphs, size, scheme);
    let refused: Vec<(&str, &Error)> = c
        .refused
        .iter()
        .map(|(i, e)| (named[*i].0.as_str(), e))
        .collect();
    assert!(refused.is_empty(), "structurize refused {refused:?}");
    assert_eq!(c.ran, graphs.len());
    c
}

#[test]
fn generated_families_run_like_the_source() {
    check_named(&generated_families(), 100, Scheme::TRACE);
}

/// Reducible shapes where lanes diverge and must reconverge: a break tail that runs after the
/// loop, an early return before a join, a non-tail jump through `pending`, a multi-level break
/// (exit ladder), a switch in a loop with shared targets, and nested loops.
fn convergence_shapes() -> Vec<TestCfg> {
    [
        vec![vec![1], vec![2, 5], vec![3, 4], vec![5], vec![1], vec![]],
        vec![vec![1, 3], vec![2, 3], vec![], vec![]],
        vec![vec![1, 4], vec![2, 3], vec![5], vec![4], vec![5], vec![]],
        vec![
            vec![1],
            vec![2, 8],
            vec![3, 6],
            vec![4, 5],
            vec![8],
            vec![2],
            vec![7],
            vec![1],
            vec![],
        ],
        vec![
            vec![1, 5],
            vec![4, 2, 3, 2, 5],
            vec![4],
            vec![2, 5],
            vec![0],
            vec![],
        ],
        vec![vec![1], vec![2, 5], vec![3, 4], vec![3, 1], vec![1], vec![]],
    ]
    .iter()
    .map(|succ| TestCfg::from_successors(succ))
    .collect()
}

/// Control §7.3d at the SPIR-V level: 32 lanes of one subgroup run the emitted functions under
/// maximal reconvergence, and every block's executions run with exactly its ideal tangles (named
/// shapes, and every graph of at most three blocks with out-degree at most two).
#[test]
fn small_graphs_converge_like_the_source_at_32_lanes() {
    let graphs: Vec<TestCfg> = convergence_shapes()
        .into_iter()
        .chain((1..=3).flat_map(|n| canonical(n, generate::out_degree_two_choices(n))))
        .collect();
    let c = check_all_run(&graphs, 200, Scheme::CONVERGENCE);
    eprintln!(
        "{} graphs, {} compared, {} diverged executions",
        graphs.len(),
        c.converged,
        c.diverged
    );
    assert!(
        c.converged >= 90 && c.diverged >= 3000,
        "only {} reducible graphs compared, {} diverged executions",
        c.converged,
        c.diverged
    );
}

/// [`small_graphs_converge_like_the_source_at_32_lanes`] over the generated families and a
/// one-in-four hash sample of the graphs of four blocks with out-degree at most two.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "32-lane runs of the generated families: runs in release (G-rel)"
)]
fn generated_families_converge_like_the_source_at_32_lanes() {
    let c = check_named(&generated_families(), 100, Scheme::CONVERGENCE);
    eprintln!(
        "generated: {} compared, {} diverged executions",
        c.converged, c.diverged
    );
    assert!(
        c.converged >= 200 && c.diverged >= 6000,
        "only {} reducible generated graphs compared, {} diverged executions",
        c.converged,
        c.diverged
    );
    let mut graphs = Vec::new();
    generate::canonical_sampled(
        4,
        &generate::out_degree_two_choices(4),
        |i| generate::in_sample(i, 4),
        |g| graphs.push(g.clone()),
    );
    let c = check_all_run(&graphs, 400, Scheme::CONVERGENCE);
    eprintln!(
        "four blocks: {} graphs, {} compared, {} diverged executions",
        graphs.len(),
        c.converged,
        c.diverged
    );
    assert!(
        c.converged > 250 && c.diverged > 5000,
        "only {} reducible four-block graphs compared, {} diverged executions",
        c.converged,
        c.diverged
    );
}

#[test]
fn unreachable_and_void_returns_end_like_the_source() {
    // 0 -> {1, 2}; 1 returns void; 2 -> {0, 3}; 3 is unreachable.
    let mut g = TestCfg::from_successors(&[vec![1, 2], vec![], vec![0, 3], vec![]]);
    g.blocks[1].kind = TestKind::Return;
    g.blocks[3].kind = TestKind::Unreachable;
    let ends: Vec<u32> = (0..Scheme::TRACE.lanes)
        .map(|l| expected(&g, seed(0, l))[1])
        .collect();
    assert!(
        ends.contains(&END_RETURN) && ends.contains(&END_UNREACHABLE),
        "the seeds must reach both ends: {ends:?}"
    );
    check_all_run(&[g.clone()], 1, Scheme::TRACE);
    check_all_run(&[g], 1, Scheme::CONVERGENCE);
}

/// The kernel form of `g` for the predicated walk: every return is `ret void`, an `unreachable`
/// block returns, and the blocks `tangled` holds for synchronize the workgroup.
fn kernel_form(g: &TestCfg, tangled: impl Fn(u32) -> bool) -> TestCfg {
    let mut g = g.clone();
    for (b, x) in g.blocks.iter_mut().enumerate() {
        if matches!(x.kind, TestKind::ReturnValue | TestKind::Unreachable) {
            x.kind = TestKind::Return;
        }
        x.tangled = tangled(b as u32);
    }
    g
}

/// Whether a return is reachable from every block of `g`. A predicated run has no prefix cutoff
/// (every invocation reaches every synchronization), so a cycle nothing leaves would never end.
fn every_block_returns(g: &TestCfg) -> bool {
    let n = g.blocks.len();
    let mut returns: Vec<bool> = g.blocks.iter().map(|b| b.succ.is_empty()).collect();
    let mut changed = true;
    while changed {
        changed = false;
        for (i, b) in g.blocks.iter().enumerate() {
            if !returns[i] && b.succ.iter().any(|s| returns[s.raw() as usize]) {
                returns[i] = true;
                changed = true;
            }
        }
    }
    (0..n).all(|i| returns[i])
}

/// Each graph from every block of which a return is reachable, with every block tangled and with
/// the even blocks tangled.
fn kernel_forms(graphs: &[TestCfg]) -> Vec<TestCfg> {
    graphs
        .iter()
        .filter(|g| every_block_returns(g))
        .flat_map(|g| [kernel_form(g, |_| true), kernel_form(g, |b| b % 2 == 0)])
        .collect()
}

/// [`check_all`] under a predicated `scheme`, where the only refusal allowed is a tangled
/// irreducible region; returns the checked counts and the number of graphs refused.
fn check_predicated(graphs: &[TestCfg], size: usize, scheme: Scheme) -> (Checked, usize) {
    let c = check_all(graphs, size, scheme);
    for (i, e) in &c.refused {
        assert!(
            matches!(e, Error::Unsupported(u) if u.feature == Feature::IrreducibleConvergent),
            "graph {:?}: {e:?}",
            graphs[*i].successor_lists()
        );
    }
    let refused = c.refused.len();
    assert_eq!(c.ran + refused, graphs.len());
    (c, refused)
}

/// P8's predicated walk runs every tangled block (a synchronization) in workgroup-uniform control
/// flow, so the interpreter's workgroup barriers never trap, while each invocation runs like the
/// source and every tangled block's live invocations are its ideal tangles (named shapes, every
/// graph of at most three blocks with out-degree at most two, and every switch of at most two
/// blocks, each fully and alternately tangled; 32 invocations in four 8-lane subgroups, both
/// scheduling orders).
#[test]
fn small_graphs_run_predicated_like_the_source() {
    let graphs: Vec<TestCfg> = convergence_shapes()
        .into_iter()
        .chain((1..=3).flat_map(|n| canonical(n, generate::out_degree_two_choices(n))))
        .chain((1..=2).flat_map(|n| canonical(n, generate::switch_choices(n))))
        .collect();
    let forms = kernel_forms(&graphs);
    let (c, refused) = check_predicated(&forms, 200, Scheme::PREDICATED);
    eprintln!(
        "{} forms, {refused} refused, {} compared, {} diverged executions",
        forms.len(),
        c.converged,
        c.diverged
    );
    assert!(
        c.converged >= 50 && c.diverged >= 300,
        "only {} graphs compared, {} diverged executions",
        c.converged,
        c.diverged
    );
}

/// The uniform schemes: every condition reported uniform, and the conditions of the odd blocks.
const UNIFORM_CLAIMS: [fn(u32) -> bool; 2] = [|_| true, |b| b % 2 == 1];

/// [`check_predicated`] of `forms` under each of [`UNIFORM_CLAIMS`]; returns the graphs compared
/// and the forms whose reported uniformity takes a real branch or exit somewhere.
fn check_uniform_claims(forms: &[TestCfg], size: usize) -> usize {
    let mut compared = 0;
    for claim in UNIFORM_CLAIMS {
        let (c, _) = check_predicated(forms, size, Scheme::predicated_uniform(claim));
        compared += c.converged;
    }
    compared
}

/// Where every invocation executes a construct together, the predicated walk takes real branches
/// and real `break`, `continue`, and `return` exits, and predicates only the constructs whose
/// conditions are not known to be uniform. Uniform runs (every invocation on one seed) report the
/// conditions of every block, or of the odd blocks, as uniform, and the blocks the uniformity
/// analysis derives from them: each invocation still runs like the source, and every tangled block
/// runs with all invocations live.
#[test]
fn small_graphs_run_predicated_with_uniform_facts() {
    let graphs: Vec<TestCfg> = convergence_shapes()
        .into_iter()
        .chain((1..=3).flat_map(|n| canonical(n, generate::out_degree_two_choices(n))))
        .chain((1..=2).flat_map(|n| canonical(n, generate::switch_choices(n))))
        .collect();
    let forms = kernel_forms(&graphs);
    let compared = check_uniform_claims(&forms, 200);
    eprintln!("{} forms, {compared} compared", forms.len());
    assert!(compared >= 100, "only {compared} graphs compared");
}

/// [`small_graphs_run_predicated_like_the_source`] over the switches of three blocks and the
/// generated families.
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "32-lane predicated runs of switches and the generated families: runs in release (G-rel)"
)]
fn generated_families_run_predicated_like_the_source() {
    let graphs: Vec<TestCfg> = canonical(3, generate::switch_choices(3))
        .into_iter()
        .chain(generated_families().into_iter().map(|(_, g)| g))
        .collect();
    let forms = kernel_forms(&graphs);
    let (c, refused) = check_predicated(&forms, 100, Scheme::PREDICATED);
    eprintln!(
        "generated: {} forms, {refused} refused, {} compared, {} diverged executions",
        forms.len(),
        c.converged,
        c.diverged
    );
    let uniform = check_uniform_claims(&forms, 100);
    eprintln!("generated, uniform facts: {uniform} compared");
    assert!(
        uniform >= 800,
        "only {uniform} generated graphs compared with uniform facts"
    );
    assert!(
        c.converged >= 400 && c.diverged >= 3000,
        "only {} generated graphs compared, {} diverged executions",
        c.converged,
        c.diverged
    );
}

/// `k` nested `if`s: block `i < k` branches to block `i + 1` or to its merge `k + 1 + i`, block
/// `k` (the innermost) and each merge fall through to the enclosing merge, and the last block
/// returns.
fn nested_ifs(k: u32) -> TestCfg {
    let mut succ: Vec<Vec<u32>> = (0..k).map(|i| vec![i + 1, k + 1 + i]).collect();
    succ.push(vec![2 * k]);
    succ.extend((0..k).map(|i| vec![if i == 0 { 2 * k + 1 } else { k + i }]));
    succ.push(vec![]);
    TestCfg::from_successors(&succ)
}

/// The predicated walk recurses along the construct nesting and refuses it above half of
/// `Limit::ConstructNesting` (`LimitExceeded`), leaving the builder untouched; the plain walk
/// takes the same tree.
#[test]
fn the_predicated_walk_refuses_nesting_above_half_the_cap() {
    use crate::budget::CONSTRUCT_NESTING;
    use crate::error::Limit;
    let half = u32::try_from(CONSTRUCT_NESTING / 2).expect("cap");
    for (k, refused) in [(half, false), (half + 1, true)] {
        let g = with_values(&kernel_form(&nested_ifs(k), |_| true));
        assert!(every_block_returns(&g));
        for scheme in [Scheme::PREDICATED, Scheme::TRACE] {
            let mut meter = Meter::new(&Budget::default());
            let s = structurize(&g, &mut meter).expect("structurize");
            let mut b = ModuleBuilder::new();
            let glob = new_glob(&mut b);
            let void = b.ty(Type::Void).expect("void");
            let mut f = b.function(void).unwrap();
            let mut e = TraceEmitter::new(&g, &glob, scheme);
            let r = emit(&s, &mut f, &mut e, &mut meter);
            s.release(&mut meter);
            match r {
                Err(Error::LimitExceeded(Limit::ConstructNesting)) => {
                    assert!(refused && scheme.predicated, "depth {k} {scheme:?}");
                }
                Ok(_) => assert!(!refused || !scheme.predicated, "depth {k} {scheme:?}"),
                Err(err) => panic!("depth {k} {scheme:?}: {err:?}"),
            }
        }
    }
}
