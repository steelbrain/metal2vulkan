//! `discard_fragment`: which form each discard site takes, from the language version, forward
//! reachability from the site, and the profile.
//!
//! Every site's form is decided once the whole function is selected ([`plan`], called from
//! `fragment::finish`), because it depends on what the instructions after the site are:
//!
//! - **MSL ≥ 2.3** (demote semantics): [`DiscardForm::Demote`] where the profile has
//!   `shaderDemoteToHelperInvocation`, else [`DiscardForm::Predicate`], whatever follows the site:
//!   Metal keeps the discarded thread running as a helper, so a loop after the site runs under
//!   Metal too, like any other loop. A non-quad subgroup operation after a predicated site is
//!   refused (below).
//! - **MSL < 2.3**, whose specification says a discard ends the thread: Metal does not end it.
//!   An M4 runs a discarded thread of MSL 2.2 exactly as one of MSL 2.3 (macOS 27.0.1, offline and
//!   run-time compiles, with and without the debug layer): its device
//!   stores and atomics do not land, and it runs a later loop to the end, joins `quad_shuffle`
//!   inside and after it, and stays a lane of `simd_sum` and `simd_ballot`. So by what is
//!   forward-reachable from the site:
//!   - no tangled instruction: [`DiscardForm::Terminate`] (`OpTerminateInvocation` where the
//!     profile has it, else `OpKill`): nothing after the site can tell it from a helper;
//!   - otherwise demote or predication as at MSL 2.3.
//! - A module that does not state its language version cannot say which semantics apply: a discard
//!   there is `Unsupported{Lowering(Fragment), NotImplemented}`.
//!
//! Non-quad subgroup operations are the simdgroup-scope group operations, simdgroup matrices and
//! asynchronous copies, the AGX lane intrinsics, and barriers.
//!
//! Predication keeps a discarded invocation running: a `Private bool discarded` is set at the site
//! and the entry's return kills when it is set, so outputs are dropped. Every store, atomic, image
//! write and copy a predicated site reaches runs only where `discarded` is clear ([`guarded`]): the
//! emitter puts it in `if !discarded`, and an atomic's result is zero for a discarded invocation,
//! whose atomics Metal's helper semantics do not land (an M4 drops a discarded thread's device
//! stores and atomics), so no family's emitter takes a guard of its own. An asynchronous copy,
//! which its simdgroup's threads make together, is no side effect of one thread to guard: a
//! predicated site reaching one keeps the rules for a non-quad subgroup operation below. A
//! predicated site from which a non-quad subgroup operation is reachable is a device gap instead,
//! `Unsupported{Lowering(Fragment)}` needing `shaderDemoteToHelperInvocation`: the predicated
//! invocation is not a helper, so it is a live lane there in every quad, while an M4 counts a
//! discarded thread only while its quad keeps a live thread, and drops a quad whose every thread is
//! discarded from the simdgroup. Demote, which the profile lacks there, maps Metal's helper to
//! Vulkan's, as for every other helper lane, and is the form every profile with it takes;
//! `MaximallyReconvergesKHR`, which an entry with a subgroup operation declares where the profile
//! has it (`lower::simd`), keeps such a helper a lane of later subgroup operations. Demote needs no guard: Vulkan helper invocations have no memory side effects and
//! their outputs are discarded.
//!
//! **A quad of helpers alone**. Metal drops a quad whose every thread is discarded from the
//! simdgroup. Vulkan leaves that to the implementation: one whose four invocations are all helpers
//! "may" be terminated, and under `RequireFullQuadsKHR` lavapipe keeps such a quad a lane of a
//! later `simd_sum`, where without it it dropped the quad. So before each non-quad subgroup
//! operation a site reaches, a quad whose four invocations are each discarded or a helper exits, as
//! Metal drops it: predicated sites test their predicate and the `HelperInvocation` input, demoting
//! ones `OpIsHelperInvocationEXT`, which a demoted invocation answers too. The exits are placed
//! where every such operation runs in a block every invocation executes together, outside an
//! interlock, on a profile with fragment quad operations; elsewhere a demoting site leaves the quad
//! to the device.

use std::collections::{BTreeMap, BTreeSet};

use crate::air::abi::{LanguageVersion, Version};
use crate::air::canon::discard_sink::{placement, Facts, Placement};
use crate::air::intrinsics::{AirIntrinsic, GroupScope, LlvmIntrinsic, Sync, TexOp};
use crate::air::ir::{AddrSpace, BlockId, Body, Callee, Inst, InstId, InstKind, Type, Value};
use crate::budget::Meter;
use crate::device::requirements::VkFeature;
use crate::device::subgroup::OpBits;
use crate::error::{Capability, Error, Feature, LowerFamily, Unsupported};
use crate::lower::select::SelectCx;
use crate::lower::simd::uniform::Uniformity;

/// How one discard site is lowered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DiscardForm {
    /// `if true { OpTerminateInvocation | OpKill }`: the invocation ends here.
    Terminate,
    /// `OpDemoteToHelperInvocation`.
    Demote,
    /// `discarded ← true`; the entry's return kills a discarded invocation.
    Predicate,
}

/// The first language version with demote semantics for `discard_fragment` (MSL 2.3).
const DEMOTE_SEMANTICS: Version = Version {
    major: 2,
    minor: 3,
    patch: 0,
};

/// What is forward-reachable from a point, as bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Reach(u8);

impl Reach {
    const TANGLED: u8 = 1;
    const NON_QUAD: u8 = 2;
    const SIDE_EFFECT: u8 = 4;

    fn has(self, bit: u8) -> bool {
        self.0 & bit != 0
    }

    fn with(self, o: Reach) -> Reach {
        Reach(self.0 | o.0)
    }
}

fn not_lowered() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::Fragment)).into()
}

/// Whether `v` is a pointer into `Thread` memory (private to the invocation, so a store to it is
/// no side effect of a discarded invocation).
fn is_thread_ptr(cx: &SelectCx<'_>, v: Value) -> bool {
    cx.value_ty(v)
        .and_then(|t| cx.entry.module.types().get(t))
        .is_some_and(|t| {
            matches!(
                t,
                Type::Ptr {
                    space: AddrSpace::Thread
                }
            )
        })
}

/// What instruction `id` itself contributes.
fn contribution(cx: &SelectCx<'_>, id: InstId, inst: &Inst) -> Reach {
    let mut r = 0u8;
    if cx.values.inst(id).is_some_and(|v| cx.tangled.contains(&v)) {
        r |= Reach::TANGLED;
    }
    r |= match &inst.kind {
        InstKind::Store { ptr, .. } if !is_thread_ptr(cx, *ptr) => Reach::SIDE_EFFECT,
        InstKind::AtomicRmw { .. } | InstKind::CmpXchg { .. } => Reach::SIDE_EFFECT,
        InstKind::Call(call) => match &call.callee {
            Callee::Air { op, .. } => match op {
                AirIntrinsic::Atomic(_) => Reach::SIDE_EFFECT,
                AirIntrinsic::Coherent(c) => match c.kind {
                    crate::air::intrinsics::AccessKind::Store => Reach::SIDE_EFFECT,
                    crate::air::intrinsics::AccessKind::Load => 0,
                },
                AirIntrinsic::Texture(t) => match t.op {
                    TexOp::Atomic(_) => Reach::SIDE_EFFECT,
                    TexOp::Write | TexOp::WriteImageblockSlice => Reach::SIDE_EFFECT,
                    _ => 0,
                },
                AirIntrinsic::Group(g) if g.scope == GroupScope::Simdgroup => Reach::NON_QUAD,
                AirIntrinsic::SimdMatrix(_) | AirIntrinsic::Sync(Sync::Barrier { .. }) => {
                    Reach::NON_QUAD
                }
                AirIntrinsic::AsyncCopy(_) => Reach::NON_QUAD | Reach::SIDE_EFFECT,
                _ => 0,
            },
            Callee::Llvm { op, .. } => match op {
                LlvmIntrinsic::Memset { dst, .. }
                | LlvmIntrinsic::Memcpy { dst, .. }
                | LlvmIntrinsic::Memmove { dst, .. }
                    if *dst != AddrSpace::Thread =>
                {
                    Reach::SIDE_EFFECT
                }
                LlvmIntrinsic::Agx(_) => Reach::NON_QUAD,
                _ => 0,
            },
            // Refused by `select` before this runs; counted as everything to stay conservative.
            Callee::Direct(_)
            | Callee::Indirect { .. }
            | Callee::UnknownIntrinsic(_)
            | Callee::Unbound { .. } => Reach::TANGLED | Reach::SIDE_EFFECT | Reach::NON_QUAD,
        },
        _ => 0,
    };
    Reach(r)
}

/// Whether instruction `id` is tangled (a derivative, an implicit-LOD sample, a quad or subgroup
/// operation, a barrier) or a non-quad subgroup operation: what tells the invocations of a quad
/// apart from each other (the loop over the covered samples). Valid once every instruction is
/// selected.
pub(super) fn quad_operation(cx: &SelectCx<'_>, id: InstId, inst: &Inst) -> bool {
    let r = contribution(cx, id, inst);
    r.has(Reach::TANGLED) || r.has(Reach::NON_QUAD)
}

/// Forward reachability over the whole function, computed once: what is reachable from each
/// block's entry.
struct Reachability {
    entry: Vec<Reach>,
}

impl Reachability {
    fn new(cx: &SelectCx<'_>, body: &Body) -> Reachability {
        let n = body.blocks.len();
        let mut own = vec![Reach::default(); n];
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (b, block) in body.blocks.iter() {
            let i = b.raw() as usize;
            let mut r = Reach::default();
            for id in block.inst_ids() {
                if let Some(inst) = body.insts.get(id) {
                    r = r.with(contribution(cx, id, inst));
                }
            }
            for s in body.successors(b) {
                if let Some(p) = preds.get_mut(s.raw() as usize) {
                    p.push(i);
                }
            }
            if let Some(o) = own.get_mut(i) {
                *o = r;
            }
        }
        // Backward fixpoint: each block's bits only grow, so every block re-enters the worklist
        // at most once per bit.
        let mut entry = own;
        let mut work: Vec<usize> = (0..n).collect();
        while let Some(i) = work.pop() {
            let here = entry.get(i).copied().unwrap_or_default();
            for &p in preds.get(i).map(Vec::as_slice).unwrap_or_default() {
                if let Some(e) = entry.get_mut(p) {
                    let grown = e.with(here);
                    if grown != *e {
                        *e = grown;
                        work.push(p);
                    }
                }
            }
        }
        Reachability { entry }
    }

    /// What is reachable after the instruction `site` of block `b`.
    fn after(&self, cx: &SelectCx<'_>, body: &Body, b: BlockId, site: InstId) -> Reach {
        let mut r = Reach::default();
        for (id, inst) in body.block_insts(b) {
            if id.raw() > site.raw() {
                r = r.with(contribution(cx, id, inst));
            }
        }
        for s in body.successors(b) {
            r = r.with(
                self.entry
                    .get(s.raw() as usize)
                    .copied()
                    .unwrap_or_default(),
            );
        }
        r
    }
}

/// The non-quad subgroup operations before which a whole quad of discarded invocations and
/// helpers exits: of the operations some site of `sites` (each with its block) reaches, in the
/// site's block after it and in every block forward-reachable from it, the first one after each
/// site and after each block's start. A later one with no site between follows an exit already, and
/// no invocation's predicate has changed since.
fn non_quad_after(cx: &SelectCx<'_>, body: &Body, sites: &[(InstId, BlockId)]) -> BTreeSet<InstId> {
    let is_site: BTreeSet<InstId> = sites.iter().map(|&(s, _)| s).collect();
    let mut ops = BTreeSet::new();
    let mut seen = vec![false; body.blocks.len()];
    let mut work = Vec::new();
    for &(site, b) in sites {
        let after = body.block_insts(b).filter(|(id, _)| id.raw() > site.raw());
        first_of_each_run(cx, &is_site, after, &mut ops);
        work.extend(body.successors(b));
    }
    while let Some(b) = work.pop() {
        match seen.get_mut(b.raw() as usize) {
            Some(s) if !*s => *s = true,
            _ => continue,
        }
        first_of_each_run(cx, &is_site, body.block_insts(b), &mut ops);
        work.extend(body.successors(b));
    }
    ops
}

/// Add to `ops` the first non-quad subgroup operation of each run of `insts` that a site of
/// `is_site` or the start of `insts` opens.
fn first_of_each_run<'b>(
    cx: &SelectCx<'_>,
    is_site: &BTreeSet<InstId>,
    insts: impl Iterator<Item = (InstId, &'b Inst)>,
    ops: &mut BTreeSet<InstId>,
) {
    let mut open = true;
    for (id, inst) in insts {
        if is_site.contains(&id) {
            open = true;
        } else if open && contribution(cx, id, inst).has(Reach::NON_QUAD) {
            ops.insert(id);
            open = false;
        }
    }
}

/// The block of each instruction in `ids`.
fn blocks_of(body: &Body, ids: impl Iterator<Item = InstId>) -> Option<Vec<(InstId, BlockId)>> {
    // Blocks tile the instruction arena in order, so an instruction's block is the last one
    // starting at or before it: one pass over the blocks, then a binary search per instruction.
    let starts: Vec<(InstId, BlockId)> = body.blocks.iter().map(|(b, k)| (k.start, b)).collect();
    ids.map(|id| {
        let i = starts.partition_point(|(start, _)| start.raw() <= id.raw());
        let (_, b) = *starts.get(i.checked_sub(1)?)?;
        body.blocks
            .get(b)
            .is_some_and(|k| id.raw() < k.end.raw())
            .then_some((id, b))
    })
    .collect()
}

/// The planned discards of one function: each site's form, the non-quad subgroup operations
/// before which a quad whose every invocation is discarded or a helper exits (under predication,
/// or where the sites demote), and whether a non-quad subgroup operation runs before a site on a
/// path to it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct DiscardPlan {
    pub forms: BTreeMap<InstId, DiscardForm>,
    pub quad_exits: BTreeSet<InstId>,
    pub counted_before: bool,
    /// The side effects a predicated site reaches, each run only where the invocation was not
    /// discarded ([`guarded`]).
    pub guarded: BTreeSet<InstId>,
}

/// The stores, atomics, image writes and copies outside `Thread` memory that some site of `sites`
/// (each with its block) reaches: after the site in its block, or in a block forward-reachable
/// from it. `None` where one of them is also a non-quad subgroup operation (an asynchronous copy,
/// which the simdgroup's threads make together, so no one thread's guard can skip it).
fn guarded(
    cx: &SelectCx<'_>,
    body: &Body,
    sites: &[(InstId, BlockId)],
    m: &mut Meter,
) -> Result<Option<BTreeSet<InstId>>, Error> {
    let mut out = BTreeSet::new();
    let mut seen = vec![false; body.blocks.len()];
    let mut work = Vec::new();
    let mut collective = false;
    let mut take = |id: InstId, inst: &Inst, out: &mut BTreeSet<InstId>| {
        let r = contribution(cx, id, inst);
        if r.has(Reach::SIDE_EFFECT) {
            collective |= r.has(Reach::NON_QUAD);
            out.insert(id);
        }
    };
    for &(site, b) in sites {
        for (id, inst) in body.block_insts(b) {
            if id.raw() > site.raw() {
                take(id, inst, &mut out);
            }
        }
        work.extend(body.successors(b));
    }
    while let Some(b) = work.pop() {
        match seen.get_mut(b.raw() as usize) {
            Some(s) if !*s => *s = true,
            _ => continue,
        }
        let block = body
            .blocks
            .get(b)
            .ok_or_else(|| cx.invariant("a successor block"))?;
        m.tick(block.end.raw().saturating_sub(block.start.raw()))?;
        for (id, inst) in body.block_insts(b) {
            take(id, inst, &mut out);
        }
        work.extend(body.successors(b));
    }
    Ok((!collective).then_some(out))
}

/// The non-quad subgroup operations that run before some site of `sites` (each with its block)
/// on a path to it, each with its block: earlier in the site's block, or in a block from which the
/// site's block is reachable (itself included where a path leaves it and returns).
fn non_quad_before(
    cx: &SelectCx<'_>,
    body: &Body,
    sites: &[(InstId, BlockId)],
) -> Vec<(InstId, BlockId)> {
    let n = body.blocks.len();
    let mut preds: Vec<Vec<BlockId>> = vec![Vec::new(); n];
    for (b, _) in body.blocks.iter() {
        for s in body.successors(b) {
            if let Some(p) = preds.get_mut(s.raw() as usize) {
                p.push(b);
            }
        }
    }
    let non_quad = |id: InstId, inst: &Inst| contribution(cx, id, inst).has(Reach::NON_QUAD);
    let mut found = BTreeSet::new();
    let mut seen = vec![false; n];
    let mut work = Vec::new();
    for &(site, b) in sites {
        found.extend(
            body.block_insts(b)
                .filter(|&(id, inst)| id.raw() < site.raw() && non_quad(id, inst))
                .map(|(id, _)| (id, b)),
        );
        work.extend(preds.get(b.raw() as usize).into_iter().flatten().copied());
    }
    while let Some(b) = work.pop() {
        match seen.get_mut(b.raw() as usize) {
            Some(s) if !*s => *s = true,
            _ => continue,
        }
        found.extend(
            body.block_insts(b)
                .filter(|&(id, inst)| non_quad(id, inst))
                .map(|(id, _)| (id, b)),
        );
        work.extend(preds.get(b.raw() as usize).into_iter().flatten().copied());
    }
    found.into_iter().collect()
}

/// Whether some operation of `before` (each with its block) runs before a discard where Metal's
/// answer is not known: its lanes include the quads the discard drops, which Metal's leave out
/// where its compiler moves the operation past the discard
/// (`PrecisionNote::QuadsCountedBeforeDiscard`). An operation canonicalization moved is past the
/// discard; one that Metal counts the dropped quads in for a measured reason
/// (`Placement::Counted`) answers as Metal's.
fn counted_before(body: &Body, before: &[(InstId, BlockId)], m: &mut Meter) -> Result<bool, Error> {
    if before.is_empty() {
        return Ok(false);
    }
    let Some(facts) = Facts::of(body, m)? else {
        // No block of the body discards, so none of `before` runs before one.
        return Ok(false);
    };
    let unknown = before
        .iter()
        .any(|&(op, b)| placement(&facts, body, op, b) != Placement::Counted);
    facts.release(m);
    Ok(unknown)
}

/// Whether whole-quad exits can be placed before the non-quad subgroup operations `ops` the
/// discard sites reach, Metal's helper lanes: the function runs outside an interlock, the profile
/// has quad operations in the fragment stage, and every one of `ops` runs in a block every
/// invocation executes together, where each quad's invocations are all present. Predicating the
/// sites with them also needs nothing the predicate would have to guard to follow a site; demoting
/// ones need no guard.
fn quad_exits_hold(
    cx: &SelectCx<'_>,
    interlocked: bool,
    ops: &BTreeSet<InstId>,
    m: &mut Meter,
) -> Result<bool, Error> {
    let fragment = &cx.profile.fields().subgroup.fragment;
    if ops.is_empty() || interlocked || !fragment.supported || !fragment.ops.contains(OpBits::QUAD)
    {
        return Ok(false);
    }
    let Some(placed) = blocks_of(cx.body, ops.iter().copied()) else {
        return Err(cx.invariant("a subgroup operation outside every block"));
    };
    let u = Uniformity::of(cx.entry, cx.body, m)?;
    let held = placed.iter().all(|&(_, b)| u.is_uniform(b));
    m.release(u.ledger_bytes());
    Ok(held)
}

/// Decide the form of every discard site in `sites`, and where whole quads exit. `interlocked`: the
/// function runs inside an interlock.
pub(super) fn plan(
    cx: &SelectCx<'_>,
    sites: &[InstId],
    interlocked: bool,
    m: &mut Meter,
) -> Result<DiscardPlan, Error> {
    let mut planned = DiscardPlan::default();
    if sites.is_empty() {
        return Ok(planned);
    }
    let Some(LanguageVersion(version)) = cx.entry.language else {
        return Err(not_lowered());
    };
    let demote_semantics = version >= DEMOTE_SEMANTICS;
    let helper = if cx.profile.fields().fragment.demote_to_helper {
        DiscardForm::Demote
    } else {
        DiscardForm::Predicate
    };
    let body = cx.body;
    let Some(placed) = blocks_of(body, sites.iter().copied()) else {
        return Err(cx.invariant("a discard site outside every block"));
    };
    planned.counted_before = counted_before(body, &non_quad_before(cx, body, &placed), m)?;
    let reach = Reachability::new(cx, body);
    let after: Vec<Reach> = placed
        .iter()
        .map(|&(site, b)| reach.after(cx, body, b, site))
        .collect();
    // A site from which a non-quad subgroup operation is reachable, under helper semantics:
    // Metal keeps a discarded thread a lane there while its quad keeps a live thread, and drops
    // a quad whose every thread is discarded. Predicating every site of the function keeps the
    // discarded invocation a full participant on every device, and the exit before each such
    // operation drops a quad of discarded invocations and helpers alone, as Metal does.
    let tangled_non_quad = after
        .iter()
        .any(|r| r.has(Reach::NON_QUAD) && (demote_semantics || r.has(Reach::TANGLED)));
    if tangled_non_quad {
        let ops = non_quad_after(cx, body, &placed);
        if quad_exits_hold(cx, interlocked, &ops, m)? {
            // Where a side effect follows a site, the sites demote where the profile can, which
            // needs no guard; else they are predicated and each side effect is guarded, where no
            // asynchronous copy is among them.
            let effects = after.iter().any(|r| r.has(Reach::SIDE_EFFECT));
            let guards = if !effects {
                Some(BTreeSet::new())
            } else if helper == DiscardForm::Predicate {
                guarded(cx, body, &placed, m)?
            } else {
                None
            };
            if let Some(guards) = guards {
                planned.forms = sites.iter().map(|&s| (s, DiscardForm::Predicate)).collect();
                planned.quad_exits = ops;
                planned.guarded = guards;
                return Ok(planned);
            }
            // A side effect follows a site, so the sites demote: the exits test the helper bit,
            // since the device may keep a quad of helpers alone a lane of the operation (the
            // module docs).
            if helper == DiscardForm::Demote {
                planned.quad_exits = ops;
            }
        }
    }
    for (&(site, _), after) in placed.iter().zip(&after) {
        let form = if demote_semantics || after.has(Reach::TANGLED) {
            helper
        } else {
            DiscardForm::Terminate
        };
        if form == DiscardForm::Predicate && after.has(Reach::NON_QUAD) {
            // A predicated invocation stays a live lane of every later simdgroup operation,
            // where Metal drops a quad whose every thread is discarded, and no whole-quad exit
            // could be placed. Demote, the form a profile with it takes, keeps the thread a helper:
            // the profile lacks it.
            return Err(Unsupported::needs_any_of(
                Feature::Lowering(LowerFamily::Fragment),
                vec![vec![Capability::Feature(
                    VkFeature::ShaderDemoteToHelperInvocation,
                )]],
            )
            .into());
        }
        planned.forms.insert(site, form);
    }
    // Every side effect a predicated site reaches runs only where the invocation was not
    // discarded. No such site reaches a non-quad subgroup operation here, so no asynchronous
    // copy either.
    let predicated: Vec<(InstId, BlockId)> = placed
        .iter()
        .copied()
        .filter(|(site, _)| planned.forms.get(site) == Some(&DiscardForm::Predicate))
        .collect();
    if !predicated.is_empty() {
        let Some(guards) = guarded(cx, body, &predicated, m)? else {
            return Err(cx.invariant("a predicated site reaches an asynchronous copy"));
        };
        planned.guarded = guards;
    }
    Ok(planned)
}
