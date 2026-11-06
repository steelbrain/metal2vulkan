//! Emulated simdgroups in divergent control flow: the plan of a kernel whose emulated sites are not
//! all in workgroup-uniform blocks.
//!
//! The emulated tier (`workgroup`) exchanges lanes between workgroup barriers, which every
//! invocation of the workgroup must execute together. A kernel with an emulated site outside
//! workgroup-uniform control flow is emitted **predicated** (`structurize::emit::predicate`):
//! every invocation, including those that returned early or took another arm, walks each
//! construct that holds a workgroup synchronization, carrying a per-invocation `live` flag, so
//! every synchronization executes in workgroup-uniform control flow; an invocation that is not
//! live executes no other instruction there and takes part in each exchange as an inactive lane.
//!
//! This module plans it while selecting ([`plan`]):
//!
//! - **Synchronizations** ([`is_sync`]): every simdgroup or quad operation, matrix and AGX
//!   multiply-add, and control barrier (a simdgroup barrier is a workgroup barrier in the
//!   emulated tier; a threadgroup barrier after an early return must be reached by the returned
//!   invocations too, or the barriers of the live ones would pair with the returned ones' next
//!   exchange). A block that holds one is a sync block.
//! - **Uniform facts** for the walk: the blocks `uniform::Uniformity` proves workgroup uniform
//!   (every invocation is live there, so the walk emits them as they are), and the branch
//!   conditions and switch selectors it proves uniform (a construct that branches on one in a
//!   workgroup-uniform position is emitted as a real branch).
//! - **Demotions**: in a divergent block the walk guards every instruction but the
//!   synchronizations with `if live { … }`, so a value defined before a synchronization and used
//!   at or after it, and a branch condition or switch selector that the walk reads after the
//!   block, crosses the guard's merge and is demoted to a `Function` variable like a
//!   block-crossing value.
//! - **Multiply-adds**: a matrix or AGX multiply-add in a divergent block is a synchronization like
//!   any other; its exchange publishes the live flag, and a cell of a lane that is not live reads
//!   as zero, which is Metal's answer for a partly live simdgroup's matrix multiply-add
//!   (`workgroup::Emu::read_cell`), with Metal's `+0` sign. The AGX form shares the rule; its Metal
//!   reading is unmeasured. Nothing is refused. On the subgroup tiers the same multiply-add ballots
//!   its live lanes instead (`super::partly_live`).
//!
//! **Workgroup barriers in every tier**. A threadgroup barrier that part of the workgroup skips
//! (after a divergent early return, under a thread-dependent condition, or in a loop with a
//! per-thread trip count) is a workgroup barrier in non-uniform control flow, which Vulkan leaves
//! undefined (an Intel GPU hangs on it). Every kernel, object function and mesh function with a
//! workgroup barrier in a block `Uniformity` does not prove uniform is walked predicated, whatever
//! its tier ([`barriers`] outside the emulated tier, `workgroup::check` in it, which only a kernel
//! takes), so every invocation reaches each such barrier, the returned ones inactive: the remaining
//! invocations synchronize among themselves, Apple's reading.
//!
//! **Partial object threadgroups**. An object function's partial threadgroup
//! (`lower::mesh::plan_partial_groups`) runs at the whole local size, and only its first threads
//! are Metal's. One whose body holds a workgroup barrier anywhere is walked predicated
//! ([`Predication::partial_groups`]) from `live` set to whether the invocation is one of its
//! threadgroup's threads (`dispatch::KernelDecl::live`, which answers every invocation of a
//! threadgroup that is not partial), so the invocations past them walk to each barrier as
//! invocations that returned at entry, executing nothing else of the body: no store, load or
//! atomic of theirs is made. Outside the emulated tier only the workgroup barriers are synchronizations
//! ([`SyncSet::WorkgroupBarriers`]): a subgroup operation runs where `live` holds, so it sees the
//! live invocations that execute it, its tangle in the source under maximal reconvergence. A loop
//! vote there uses a workgroup word of its own (`RuntimeFact::PredicationVote`), declared when a
//! synchronization lies in a loop.

use std::collections::BTreeMap;

use crate::air::ir::{BlockId, Idx, InstKind, Value, ValueId};
use crate::budget::Meter;
use crate::device::subgroup::Tier;
use crate::error::{Error, Location};
use crate::graph::LoopForest;
use crate::lower::select::SelectCx;
use crate::lower::strategy::{FamilyStrategy, Strategy};
use crate::lower::sync::barrier::{self, BarrierStrategy};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::Stage;
use crate::spirv::ids::DemoteTy;

use super::agx::AgxStrategy;
use super::matrix::MatrixStrategy;
use super::uniform::Uniformity;
use super::SimdStrategy;

/// The instructions the predicated walk runs in every invocation of the workgroup: its
/// synchronizations.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SyncSet {
    /// The emulated tier ([`is_sync`]): every emulated group operation, matrix and AGX
    /// multiply-add, and control barrier, each of which crosses workgroup barriers there.
    #[default]
    Emulated,
    /// Every other tier: the control barriers emitted at `Workgroup` scope
    /// (`barrier::is_workgroup_barrier`, which knows the function's `stage` and `tier`). A
    /// subgroup or quad operation, a multiply-add, and a subgroup-scope barrier run where `live`
    /// holds, like any other instruction.
    WorkgroupBarriers { stage: Stage, tier: Tier },
}

impl SyncSet {
    /// Whether strategy `s` is one of the set's synchronizations.
    pub fn holds(self, s: &Strategy) -> bool {
        match self {
            SyncSet::Emulated => is_sync(s),
            SyncSet::WorkgroupBarriers { stage, tier } => {
                barrier::is_workgroup_barrier(s, stage, tier)
            }
        }
    }
}

/// What the predicated walk of a kernel needs (`SimdPlan::predication`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Predication {
    /// By AIR block index: every invocation of the workgroup executes the block together.
    pub uniform: Vec<bool>,
    /// By AIR block index: the block holds a workgroup synchronization (of [`Predication::set`]).
    pub syncs: Vec<bool>,
    /// The branch conditions and switch selectors every invocation evaluates alike, sorted.
    pub uniform_conds: Vec<ValueId>,
    /// Which instructions are synchronizations.
    pub set: SyncSet,
    /// Outside the emulated tier: the module declares a workgroup word for the loop votes
    /// (`RuntimeFact::PredicationVote`), because a synchronization lies in a loop. The emulated
    /// tier votes in its scratch's first word.
    pub vote_word: bool,
    /// An object function's partial threadgroups (`MeshPlan::partial_groups`): the walk starts
    /// with `live` set only where the invocation is one of its threadgroup's threads
    /// (`dispatch::emit_entry`), rather than everywhere.
    pub partial_groups: bool,
}

impl Predication {
    /// Whether strategy `s` is a synchronization of this walk.
    pub fn is_sync(&self, s: &Strategy) -> bool {
        self.set.holds(s)
    }

    /// Whether AIR block `b` is workgroup uniform.
    pub fn is_uniform(&self, b: BlockId) -> bool {
        self.uniform.get(b.raw() as usize).copied().unwrap_or(false)
    }

    /// Whether AIR block `b` holds a workgroup synchronization.
    pub fn syncs(&self, b: BlockId) -> bool {
        self.syncs.get(b.raw() as usize).copied().unwrap_or(false)
    }

    /// Whether branch condition or switch selector `v` is workgroup uniform.
    pub fn uniform_cond(&self, v: ValueId) -> bool {
        self.uniform_conds.binary_search(&v).is_ok()
    }
}

/// Whether strategy `s` synchronizes the workgroup in the emulated tier: it exchanges lanes, reads
/// the simdgroup's active lanes, or is a control barrier.
pub fn is_sync(s: &Strategy) -> bool {
    matches!(
        s,
        Strategy::Barrier(BarrierStrategy::Barrier(_))
            | Strategy::Simd(
                SimdStrategy::Group(_)
                    | SimdStrategy::Matrix(MatrixStrategy::Mac(_))
                    | SimdStrategy::Agx(AgxStrategy::MatMad(_))
            )
    )
}

/// Plan the predicated walk of the kernel of `cx` with synchronizations `set`: the uniform and sync
/// blocks, the uniform conditions, and the values the guards demote, added to `cross` (sorted,
/// block-crossing values). The vote word is left to the caller.
pub fn plan(
    cx: &SelectCx<'_>,
    strategy: &[Option<Strategy>],
    u: &Uniformity,
    set: SyncSet,
    cross: &mut Vec<(ValueId, DemoteTy)>,
    m: &mut Meter,
) -> Result<Predication, Error> {
    let body = cx.body;
    let values = cx.values;
    let nb = body.blocks.len();
    let strategy_of = |id| {
        values
            .inst(id)
            .and_then(|v| strategy.get(v.index()))
            .and_then(Option::as_ref)
    };
    // The per-block flags, and each instruction's block and position in it.
    let n = body.insts.len();
    let bytes = nb.saturating_mul(2).saturating_add(n.saturating_mul(8));
    m.charge(bytes)?;
    let r = (|| {
        let mut p = Predication {
            uniform: vec![false; nb],
            syncs: vec![false; nb],
            uniform_conds: Vec::new(),
            set,
            vote_word: false,
            partial_groups: false,
        };
        let mut home: Vec<(u32, u32)> = vec![(u32::MAX, 0); n];
        for (b, block) in body.blocks.iter() {
            let uniform = u.is_uniform(b);
            if let Some(x) = p.uniform.get_mut(b.raw() as usize) {
                *x = uniform;
            }
            for (i, id) in block.inst_ids().enumerate() {
                m.tick(1)?;
                if let Some(h) = home.get_mut(id.index()) {
                    *h = (b.raw(), u32::try_from(i).unwrap_or(u32::MAX));
                }
                let Some(s) = strategy_of(id) else { continue };
                if !set.holds(s) {
                    continue;
                }
                if let Some(x) = p.syncs.get_mut(b.raw() as usize) {
                    *x = true;
                }
            }
            match body.terminator(b).map(|t| &t.kind) {
                Some(InstKind::CondBr { c, .. }) if u.is_uniform_value(cx.entry, *c) => {
                    p.uniform_conds.extend(values.id(*c));
                }
                Some(InstKind::Switch { v, .. }) if u.is_uniform_value(cx.entry, *v) => {
                    p.uniform_conds.extend(values.id(*v));
                }
                _ => {}
            }
        }
        p.uniform_conds.sort_unstable();
        p.uniform_conds.dedup();
        let extra = demotions(cx, strategy, set, &home, m)?;
        if !extra.is_empty() {
            let mut all: BTreeMap<ValueId, DemoteTy> = cross.drain(..).collect();
            all.extend(extra);
            cross.extend(all);
        }
        Ok(p)
    })();
    m.release(bytes);
    r
}

/// The bytes of the workgroup word a predicated kernel outside the emulated tier votes in
/// (`RuntimeFact::PredicationVote`).
pub const VOTE_BYTES: u32 = 4;

/// Plan the predicated walk of a kernel, object or mesh function outside the emulated tier: when
/// some control barrier emitted at `Workgroup` scope stands in a block `Uniformity` does not prove
/// workgroup uniform, every invocation must walk to it. `given` is the body's uniformity when the
/// caller already holds it (`simd::partly_live`, which keeps its ledger bytes); otherwise it is
/// computed here, only for a function with such a barrier, and released. A function whose
/// workgroup barriers are all in uniform blocks keeps `predication: None` and is emitted as before,
/// but for an object function of partial threadgroups (`MeshPlan::partial_groups`): one with any
/// workgroup barrier is walked predicated from its threadgroup's threads
/// ([`Predication::partial_groups`]).
///
/// A synchronization in a loop makes the walk vote (`structurize::emit::CodeEmitter::vote`),
/// which needs one workgroup word beside the function's threadgroup arena: the function depends on
/// `RuntimeFact::PredicationVote`, and its static arena with that word must fit its stage's
/// workgroup memory (`mesh::workgroup_limit`, else `ThreadgroupMemorySize`); an object function's
/// arena fact counts the word (`RuntimeFact::ObjectThreadgroupArena`). An object function whose
/// barriers are uniform and whose word does not fit runs no partial threadgroup instead
/// (`MeshPlan::partial_groups` cleared, so the binder refuses such a draw).
pub fn barriers(
    cx: &mut SelectCx<'_>,
    strategy: &[Option<Strategy>],
    cross: &mut Vec<(ValueId, DemoteTy)>,
    tier: Tier,
    given: Option<&Uniformity>,
    m: &mut Meter,
) -> Result<(), Error> {
    if !matches!(cx.stage, Stage::Kernel | Stage::Object | Stage::Mesh) || tier == Tier::Emulated {
        return Ok(());
    }
    let set = SyncSet::WorkgroupBarriers {
        stage: cx.stage,
        tier,
    };
    if !strategy.iter().flatten().any(|s| set.holds(s)) {
        return Ok(());
    }
    // A partial object threadgroup's invocations past its threads walk to every barrier.
    let partial = cx.stage == Stage::Object && cx.families.mesh.partial_groups;
    let computed;
    let u = match given {
        Some(u) => u,
        None => {
            computed = Uniformity::of(cx.entry, cx.body, m)?;
            &computed
        }
    };
    let r = (|| -> Result<Option<Predication>, Error> {
        let divergent = cx.body.blocks.iter().any(|(b, block)| {
            !u.is_uniform(b)
                && block.inst_ids().any(|id| {
                    cx.values
                        .inst(id)
                        .and_then(|v| strategy.get(v.index()))
                        .and_then(Option::as_ref)
                        .is_some_and(|s| set.holds(s))
                })
        });
        if !divergent && !partial {
            return Ok(None);
        }
        let mut p = plan(cx, strategy, u, set, cross, m)?;
        p.vote_word = sync_in_loop(cx, &p, m)?;
        p.partial_groups = partial;
        Ok(Some((p, divergent)))
    })();
    if given.is_none() {
        m.release(u.ledger_bytes());
    }
    let Some((p, divergent)) = r? else {
        return Ok(());
    };
    if p.vote_word {
        // The arena the module declares (`access::local::use_arena` checked its static part) and
        // its typed threadgroup objects.
        let arena = crate::lower::access::local::static_workgroup_bytes(cx);
        let required =
            super::workgroup::arena_bytes(arena, 0).saturating_add(u64::from(VOTE_BYTES));
        match crate::lower::mesh::workgroup_limit(cx).check(required) {
            Ok(()) => {}
            // Uniform barriers need the walk only for partial threadgroups, which then stay
            // refused.
            Err(Error::Unsupported(_)) if !divergent => {
                cx.families.mesh.partial_groups = false;
                return Ok(());
            }
            Err(e) => return Err(e),
        }
        cx.facts.insert(RuntimeFact::PredicationVote);
        vote_in_object_arena(cx);
    }
    cx.families.simd.predication = Some(p);
    Ok(())
}

/// An object function's arena fact with the vote word in its static part, which the binder checks
/// beside the encoder's lengths (`RuntimeFact::ObjectThreadgroupArena`).
fn vote_in_object_arena(cx: &mut SelectCx<'_>) {
    let arena = cx.facts.iter().find_map(|f| match *f {
        RuntimeFact::ObjectThreadgroupArena {
            static_bytes,
            payload,
        } => Some((static_bytes, payload)),
        _ => None,
    });
    let Some((static_bytes, payload)) = arena else {
        return;
    };
    cx.facts.remove(&RuntimeFact::ObjectThreadgroupArena {
        static_bytes,
        payload,
    });
    cx.facts.insert(RuntimeFact::ObjectThreadgroupArena {
        static_bytes: static_bytes.saturating_add(VOTE_BYTES),
        payload,
    });
}

/// Whether a sync block of `p` lies in a loop of the kernel's CFG, so the walk may vote: it votes
/// only in a loop that holds a synchronization.
fn sync_in_loop(cx: &SelectCx<'_>, p: &Predication, m: &mut Meter) -> Result<bool, Error> {
    if !p.syncs.iter().any(|&s| s) {
        return Ok(false);
    }
    let loops = LoopForest::compute(&cx.entry.cfg, m)?;
    let any = cx
        .body
        .blocks
        .iter()
        .any(|(b, _)| p.syncs(b) && loops.innermost(b).is_some());
    m.release(loops.ledger_bytes());
    Ok(any)
}

/// The values a guard's merge separates from a use in the same block: defined by an instruction
/// that is not a synchronization, and used by the block's terminator (the walk branches after
/// the block's last guard and parallel copy), or, unless a phi, at or after a later
/// synchronization of the block (the synchronization included) or by the parallel copy into a
/// successor's phi after one. Every block counts, the workgroup-uniform ones included: the walk
/// guards a uniform block wherever it cannot tell statically that every invocation is live (an
/// arm of a `Dispatch`, or the rest of an iteration after a divergent `break`), since the block
/// may be one that no invocation executes. A use is what `FamilyPlans::reads` and
/// `FamilyPlans::incoming_reads` say an instruction or a copy reads (the value a forwarding
/// `extractvalue` forwards; a member-wise phi's members), the hooks the walk reloads through.
fn demotions(
    cx: &SelectCx<'_>,
    strategy: &[Option<Strategy>],
    set: SyncSet,
    home: &[(u32, u32)],
    m: &mut Meter,
) -> Result<BTreeMap<ValueId, DemoteTy>, Error> {
    let body = cx.body;
    let values = cx.values;
    let families = &cx.families;
    let sync_at = |id| {
        values
            .inst(id)
            .and_then(|v| strategy.get(v.index()))
            .and_then(Option::as_ref)
            .is_some_and(|s| set.holds(s))
    };
    let mut out = BTreeMap::new();
    for (b, block) in body.blocks.iter() {
        // The position of each synchronization of the block, in order.
        let syncs: Vec<u32> = block
            .inst_ids()
            .enumerate()
            .filter(|(_, id)| sync_at(*id))
            .map(|(i, _)| u32::try_from(i).unwrap_or(u32::MAX))
            .collect();
        let end = u32::try_from(block.len()).unwrap_or(u32::MAX);
        // Whether a synchronization lies in `(def, use]`.
        let between = |def: u32, at: u32| syncs.iter().any(|&s| def < s && s <= at);
        let mut uses: Vec<(Value, u32, bool)> = Vec::new();
        for (i, id) in block.inst_ids().enumerate() {
            m.tick(1)?;
            let Some(inst) = body.insts.get(id) else {
                continue;
            };
            let at = u32::try_from(i).unwrap_or(u32::MAX);
            let term = matches!(inst.kind, InstKind::CondBr { .. } | InstKind::Switch { .. });
            // A phi reads its incoming values in their blocks, below.
            families.reads(body, id, inst, |v| uses.push((v, at, term)));
        }
        // The parallel copies into the successors' phis, after the block's last instruction.
        for s in body.successors(b) {
            for (sid, inst) in body.block_insts(s) {
                let InstKind::Phi { incoming, .. } = &inst.kind else {
                    continue;
                };
                for (v, from) in incoming.iter() {
                    if *from == b {
                        families.incoming_reads(body, values, (sid, *v, b), |r| {
                            uses.push((r, end, false))
                        });
                    }
                }
            }
        }
        for (v, at, term) in uses {
            let Value::Inst(d) = v else { continue };
            let Some(&(home_block, def)) = home.get(d.index()) else {
                continue;
            };
            // A synchronization's result is defined outside the guards, and a phi is read from
            // its incoming variable, which nothing writes inside its block before the parallel
            // copy at its end; the terminator is read after that copy, so a phi it reads is
            // demoted (its demoted variable is written at the block's top).
            let phi = matches!(
                body.insts.get(d).map(|i| &i.kind),
                Some(InstKind::Phi { .. })
            );
            if home_block != b.raw() || sync_at(d) || (phi && !term) {
                continue;
            }
            if !(term || between(def, at)) {
                continue;
            }
            let Some(id) = values.inst(d) else {
                return Err(cx.invariant("a use out of the value range"));
            };
            let valueless = strategy
                .get(id.index())
                .and_then(Option::as_ref)
                .is_some_and(|s| !s.yields_value());
            if valueless || out.contains_key(&id) {
                continue;
            }
            let Some(def_inst) = body.insts.get(d) else {
                continue;
            };
            let ty = cx.lowered_repr(v, def_inst.ty).map_err(|e| match e {
                Error::Unsupported(u) if u.location.is_none() => Error::Unsupported(u.at(
                    Location::function(cx.entry.ep.name.clone(), Some(b.raw()), Some(d.raw())),
                )),
                other => other,
            })?;
            out.insert(id, ty);
        }
    }
    Ok(out)
}

/// The SPIR-V words the predicated walk adds to a kernel's estimate: per block (the walk may guard
/// a workgroup-uniform one too, as for `demotions`) its guards, the loads and stores of the live
/// flag and of the per-loop flags, the sequential arms of its terminator's construct, and a
/// workgroup vote; per operand use the reload of a demoted value; per synchronization its sanitized
/// operands and the reads of the published active flags (`workgroup::predicated_extra_words`).
pub fn extra_words(
    body: &crate::air::ir::Body,
    values: crate::air::ir::ValueIndex,
    strategy: &[Option<Strategy>],
    set: SyncSet,
) -> u64 {
    const BLOCK: u64 = 512;
    const USE: u64 = 12;
    const SYNC: u64 = 96;
    let mut words = 0u64;
    for (_, block) in body.blocks.iter() {
        words = words.saturating_add(BLOCK);
        for id in block.inst_ids() {
            let Some(inst) = body.insts.get(id) else {
                continue;
            };
            let mut uses = 0u64;
            inst.kind.for_each_value(|_| uses += 1);
            words = words.saturating_add(USE.saturating_mul(uses));
            let s = values
                .inst(id)
                .and_then(|v| strategy.get(v.index()))
                .and_then(Option::as_ref);
            if let Some(s) = s.filter(|s| set.holds(s)) {
                words = words
                    .saturating_add(SYNC)
                    .saturating_add(u64::from(super::workgroup::predicated_extra_words(s)));
            }
        }
    }
    words
}
