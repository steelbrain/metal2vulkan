//! The emulated simdgroup tier (`Tier::Emulated`): 32-lane simdgroups of a kernel whose device
//! cannot run them on subgroups, realized through workgroup memory.
//!
//! **Layout.** Simdgroup `k` is the linear threads `[32k, 32k + 32)` of the threadgroup (Metal
//! lane `lin & 31`, as the dispatch builtins derive it from the linear local index `lin`), and a
//! quad the threads `[4q, 4q + 4)`. The Vulkan subgroup plays no part: no subgroup instruction is
//! emitted, so the tier needs no subgroup feature and no subgroup size.
//!
//! **Exchange.** An operation that reads other lanes publishes the values it moves: each lane
//! stores its words at `scratch[lin · W + i]`, a workgroup barrier makes them visible, each lane
//! reads the words of the lanes it needs, and a second workgroup barrier keeps the next exchange
//! from overwriting them before every lane has read. `W` is the widest exchange of the function
//! (words per lane: one per lane of each value it moves; Metal moves no 64-bit value between
//! lanes, so a 64-bit form is not lowered), and the scratch is a `u32` array of `W · ⌈N / 32⌉ · 32`
//! words, `N = sx·sy·sz` the threadgroup's threads (the local-size specialization constants), so
//! every lane index a simdgroup of the threadgroup names is inside it. Loads and stores carry the
//! memory model's coherence and the barriers its semantics (`NonPrivatePointer` and
//! `MakeAvailable`/`MakeVisible` under the Vulkan model), as the threadgroup arena's do.
//!
//! **Exactness.** A workgroup barrier is defined only where every invocation of the workgroup
//! executes it, so every operation that exchanges or synchronizes lanes must run in
//! workgroup-uniform control flow. In a block [`super::uniform`] proves uniform every thread of
//! the threadgroup executes the operation, so the active lanes of a simdgroup are exactly its
//! lanes below `N` (the last simdgroup of a threadgroup whose size is not a multiple of 32 is
//! partially occupied, as in Metal). A kernel with an operation in any other block is emitted
//! predicated ([`super::divergent`]): every invocation executes the operation, and the invocations
//! that do not execute it in the source publish an inactive flag beside their words
//! (`EmitCx::active`), so the active lanes are the lanes below `N` whose flag is set; a source lane
//! past `N` reads zero bits, and so does one below `N` that is not active (a multiply-add's cell
//! reads zero), reductions and prefix sums fold only active lanes (in the tree forms an inactive
//! lane takes part with the operation's identity, `-0` for a float sum, and in a float min or max,
//! which is idempotent, with the first active lane's operand instead, so no NaN reaches `NMin` or
//! `NMax`), and votes, `is_first`, `broadcast_first`, and the quad's first lane and mask read the
//! active lanes' word. [`vote`] decides whether any invocation of the workgroup is still in a
//! predicated loop. In workgroup-uniform blocks:
//!
//! - shuffles and broadcasts read their source lane's words; a source outside the threadgroup (a
//!   lane of a partial simdgroup past `N`, whose value Metal leaves undefined) reads all-zero
//!   bits, Metal's answer ([`Emu::read_lane`], which amends the calling-lane rule), so no read uses
//!   words no lane wrote; relative shuffles keep Metal's rule of reading the calling lane where the
//!   source leaves the simdgroup or quad;
//! - simdgroup reductions combine the lanes below `N` in a tree of five exchange rounds
//!   (`tree_fold`): exact for integers, a narrow float sum rounded after each addition, and a float
//!   sum's association order the tree's (`ReductionOrder` is recorded as in every tier); prefix
//!   sums take five rounds too (`tree_scan`; a float exclusive sum shifts the operands one lane
//!   first), exact for integers and associated as the rounds for floats;
//! - votes and ballots OR each lane's bit into the simdgroup's word (bit `i` is Metal lane `i`)
//!   with workgroup atomics (`Emu::gather_bits`), and a quad's votes into its first lane's word;
//!   `is_first` is lane 0 (always below `N`), and `quad_active_threads_mask` the quad's lanes below
//!   `N`;
//! - the simdgroup-matrix and AGX multiply-adds publish their four `f32` cells once and read
//!   them for every step (`matrix::tile_mad`); a lane past `N` takes part with zero cells, as
//!   does, in a divergent block, a lane that is not live ([`Emu::read_cell`], which amends this
//!   item's calling-lane rule for the multiply-adds);
//! - a simdgroup barrier is a workgroup barrier with the Metal flags' memories, which orders a
//!   superset of the simdgroup's threads; a simdgroup-scope fence orders at workgroup scope for
//!   the same reason (`sync::barrier`).
//!
//! **Memory.** The scratch takes `4 · W · 32` bytes per simdgroup beside the kernel's own
//! threadgroup memory, the arena the module declares ([`arena_bytes`]: whole words and at least
//! `MIN_ARENA_BYTES`). [`check`] refuses a kernel whose static arena and one simdgroup's scratch
//! exceed `maxComputeSharedMemorySize` (`ThreadgroupMemorySize`), and the function depends on
//! `RuntimeFact::SimdgroupEmulation{bytes_per_simdgroup, arena}`: the planner checks every
//! dispatch region's total (the arena with the `[[threadgroup(n)]]` lengths, and the scratch of
//! its simdgroups); the pipeline's `max_total_threads_per_threadgroup` is at most the threads
//! whose scratch fits beside the static arena, and its `static_threadgroup_memory_length` is the
//! kernel's own, as Metal reports it, so a dispatch whose threadgroup memory lengths fit beside
//! that length but not beside the scratch of its simdgroups is refused (the device's gap).
//!
//! **Reconvergence**. The emulated operations exchange through workgroup memory between workgroup
//! barriers in workgroup-uniform control flow (predicated where the source diverges), so no result
//! depends on how subgroups reconverge: an entry whose only tangled instructions are emulated ones
//! declares no reconvergence mode and records `Reconvergence::Guaranteed`.

use crate::air::intrinsics::{GroupOp, GroupScope, MemFlags};
use crate::air::ir::ValueId;
use crate::budget::Meter;
use crate::error::{Error, Feature, LowerFamily, Unsupported};
use crate::lower::arith::legal::{Form, Held};
use crate::lower::arith::{narrow, ops, wide};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::strategy::Strategy;
use crate::lower::sync::barrier::{self, BarrierStrategy};
use crate::lower::sync::memory_model;
use crate::memory::arena::MIN_ARENA_BYTES;
use crate::memory::leaf::Coherence;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{SpecKey, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, Dangling};
use crate::spirv::ids::{ConstId, DemoteTy, GlobalId, TypeId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, GlslOp, GroupArith, MemAccess, MemoryOrder, Scope,
    Semantics, SpecOp, StorageClass, Type,
};

use super::agx::AgxStrategy;
use super::divergent;
use super::group::{
    bin_k, fill_source, k, minus_zero_unless, nth, post, pre, quad_tree, zero_cell, Exchange, Lane,
    SIMD_WIDTH,
};
use super::matrix::MatrixStrategy;
use super::uniform::Uniformity;
use super::{GroupPlan, ReconvergenceMode, SimdStrategy};

fn dangling() -> BuildError {
    BuildError::Dangling(Dangling::Value(SValue::new(u32::MAX)))
}

/// The words one lane publishes for a value of form `f`: one per lane (a lane of 32 bits or
/// fewer, a `bool` included).
fn form_words(f: Form) -> u32 {
    u32::from(f.lanes)
}

/// The form of the values group operation `g` moves between lanes, if it moves any: a vote or a
/// ballot moves its predicate; `is_first` and the quad's active mask move nothing (lane 0 and the
/// quad's first lane are known, and the active lanes are those below the thread count).
fn moved(g: &GroupPlan) -> Option<Form> {
    use GroupOp as G;
    match g.op {
        G::IsFirst | G::ActiveThreadsMask => None,
        G::All | G::Any | G::Ballot => Some(Form {
            held: Held::Bool,
            lanes: 1,
        }),
        _ => Some(g.form),
    }
}

/// The words one lane publishes for group operation `g` (a shuffle-and-fill publishes its data
/// and its fill).
fn group_words(g: &GroupPlan) -> u32 {
    let per = moved(g).map_or(0, form_words);
    match g.op {
        GroupOp::ShuffleAndFillUp | GroupOp::ShuffleAndFillDown => 2 * per,
        _ => per,
    }
}

/// The words the matrix and AGX multiply-adds publish: two `a` and two `b` cells, `f32` each.
const TILE_WORDS: u32 = 4;

/// The words one lane publishes for strategy `s`, when `s` exchanges values between lanes.
fn exchange_words(s: &Strategy) -> Option<u32> {
    match s {
        Strategy::Simd(SimdStrategy::Group(g)) => Some(group_words(g)),
        Strategy::Simd(
            SimdStrategy::Matrix(MatrixStrategy::Mac(_))
            | SimdStrategy::Agx(AgxStrategy::MatMad(_)),
        ) => Some(TILE_WORDS),
        _ => None,
    }
}

/// The words one lane publishes for strategy `s` in a divergent block of a predicated kernel: its
/// values and its active flag, which every group operation reads (a vote's lanes, a shuffle's
/// source, the first active lane) and a multiply-add's cell reads ([`Emu::read_cell`]). A barrier
/// publishes nothing.
fn predicated_words(s: &Strategy) -> Option<u32> {
    match s {
        Strategy::Simd(SimdStrategy::Group(g)) => Some(group_words(g) + 1),
        Strategy::Simd(
            SimdStrategy::Matrix(MatrixStrategy::Mac(_))
            | SimdStrategy::Agx(AgxStrategy::MatMad(_)),
        ) => Some(TILE_WORDS + 1),
        _ => exchange_words(s),
    }
}

/// Whether strategy `s` exchanges values or synchronizes between the lanes of a simdgroup, so
/// that the emulated tier emits workgroup barriers for it.
fn emulated_site(s: &Strategy) -> bool {
    match s {
        Strategy::Barrier(BarrierStrategy::Barrier(b)) => b.scope == Scope::Subgroup,
        _ => exchange_words(s).is_some(),
    }
}

/// The SPIR-V words one scratch read of a value of `w` words emits: the source clamp, per word its
/// address, access chain, and load, and the value's reassembly.
fn read_words(w: u32) -> u32 {
    48 + 40 * w
}

/// The SPIR-V words of one of [`tree_fold`]'s or [`tree_scan`]'s rounds on a value of `v` words:
/// the published words, the two barriers, the source lane, its read, the lane-order selects of a
/// float min or max, the combining step, and the select that keeps the partial where the partner
/// holds no contribution.
fn tree_round_words(v: u32, g: &GroupPlan) -> u32 {
    48 * v + read_words(v) + 128 + narrow::round_words(g.form)
}

/// The SPIR-V words of [`Emu::gather_bits`] of `conds` words and the closing barrier: the clears,
/// three barriers, the lane's bit, and per word its select, atomic `Or`, and load.
fn gather_words(conds: u32) -> u32 {
    96 + 96 * conds
}

/// The SPIR-V words the emulated tier adds to strategy `s`'s own estimate: the stores of the
/// published words and the two barriers, and per read of another lane's value the scratch read with
/// its combining step; a simdgroup reduction is five tree rounds, a prefix sum or product six (a
/// float's exclusive sum and an exclusive product shift first), and a vote or ballot one
/// `Emu::gather_bits`. The subgroup instructions the estimate counted are not emitted, so this
/// over-counts.
pub fn extra_words(s: &Strategy) -> u32 {
    let Some(w) = exchange_words(s) else {
        return 0;
    };
    let publish = 48 * w + 48;
    let reads = match s {
        Strategy::Simd(SimdStrategy::Group(g)) => {
            use GroupOp as G;
            let v = moved(g).map_or(0, form_words);
            let step = read_words(v) + 64 + narrow::round_words(g.form);
            let rounds = SIMD_WIDTH.trailing_zeros();
            match (g.scope, g.op) {
                (GroupScope::Quad, G::IsFirst | G::ActiveThreadsMask)
                | (GroupScope::Simdgroup, G::IsFirst) => 64,
                (GroupScope::Simdgroup, G::All | G::Any | G::Ballot | G::ActiveThreadsMask)
                | (GroupScope::Quad, G::All | G::Any | G::Ballot) => gather_words(2),
                (
                    GroupScope::Simdgroup,
                    G::PrefixInclusiveSum
                    | G::PrefixExclusiveSum
                    | G::PrefixInclusiveProduct
                    | G::PrefixExclusiveProduct,
                ) => (rounds + 1) * tree_round_words(v, g),
                (
                    GroupScope::Simdgroup,
                    G::Sum | G::Product | G::Min | G::Max | G::And | G::Or | G::Xor,
                ) => rounds * tree_round_words(v, g),
                (
                    GroupScope::Quad,
                    G::Sum
                    | G::Product
                    | G::Min
                    | G::Max
                    | G::And
                    | G::Or
                    | G::Xor
                    | G::PrefixInclusiveSum
                    | G::PrefixExclusiveSum
                    | G::PrefixInclusiveProduct
                    | G::PrefixExclusiveProduct,
                ) => 4 * step,
                (_, G::ShuffleAndFillUp | G::ShuffleAndFillDown) => 2 * step,
                _ => step,
            }
        }
        _ => TILE_READS * (read_words(1) + CELL_SELECT_WORDS),
    };
    publish + reads
}

/// The SPIR-V words the emulated tier adds to strategy `s` in a divergent block of a predicated
/// kernel: besides [`extra_words`], the published active flag and the active-lane select of each
/// value, the active-lane word of `is_first` and `broadcast_first` and of the votes, and, for a
/// quad's other operations, which read its lanes one by one, the read of each lane's flag; for a
/// multiply-add, per cell read the source lane's flag.
pub fn predicated_extra_words(s: &Strategy) -> u32 {
    let base = extra_words(s);
    match s {
        Strategy::Simd(SimdStrategy::Group(g)) => {
            let votes = matches!(g.op, GroupOp::All | GroupOp::Any);
            let flags = if g.scope == GroupScope::Quad && !votes {
                SIMD_WIDTH * (read_words(1) + 64)
            } else {
                gather_words(1)
            };
            base.saturating_mul(2)
                .saturating_add(flags)
                .saturating_add(96)
        }
        Strategy::Simd(
            SimdStrategy::Matrix(MatrixStrategy::Mac(_))
            | SimdStrategy::Agx(AgxStrategy::MatMad(_)),
        ) => base.saturating_add(TILE_READS * read_words(1)),
        _ => base,
    }
}

/// The cell reads of a multiply-add ([`extra_words`]): per step `k` of the (at most) eight, the
/// `a` cell and two `b` cells, each with the select of a zero cell (`CELL_SELECT_WORDS`).
const TILE_READS: u32 = 8 * 3;

/// The SPIR-V words of the select of a zero cell where a multiply-add's source lane does not
/// take part, and its constant ([`Emu::read_cell`]).
const CELL_SELECT_WORDS: u32 = 16;

/// The scratch bytes one simdgroup of a function publishing `words` per lane takes
/// (`4 · words · 32`).
pub fn bytes_per_simdgroup(words: u32) -> u64 {
    4 * u64::from(words) * u64::from(SIMD_WIDTH)
}

/// The threadgroup memory beside the scratch when the module's arena, whose static part is
/// `arena` (`None`: no arena is declared), holds `regions` more bytes of specialized
/// `[[threadgroup(n)]]` regions: the arena's `u32` array, whole words and at least
/// `MIN_ARENA_BYTES` (`memory::arena`).
pub fn arena_bytes(arena: Option<u32>, regions: u64) -> u64 {
    match arena {
        Some(s) => (u64::from(s).saturating_add(regions))
            .next_multiple_of(4)
            .max(MIN_ARENA_BYTES),
        None => regions,
    }
}

/// Complete an emulated kernel's selection. A kernel whose emulated sites and threadgroup barriers
/// are all in workgroup-uniform control flow exchanges; any other is planned predicated
/// (`divergent::plan`, which adds the values its guards separate to the block-crossing values
/// `cross`; a matrix or AGX multiply-add in a divergent block publishes its live flag, and reads a
/// lane that is not active as zero cells). The static arena with one simdgroup's scratch must fit
/// `maxComputeSharedMemorySize` (else `ThreadgroupMemorySize`). A function that exchanges values
/// sets the plan's exchange width and depends on `RuntimeFact::SimdgroupEmulation`. An entry whose
/// only tangled instructions are emulated sites and threadgroup barriers the lockstep pass placed
/// keeps no reconvergence mode. `given` is the body's uniformity when the caller already holds it
/// (`simd::partly_live`, which keeps its ledger bytes); otherwise it is computed and released here.
pub fn check(
    cx: &mut SelectCx<'_>,
    strategy: &[Option<Strategy>],
    cross: &mut Vec<(ValueId, DemoteTy)>,
    given: Option<&Uniformity>,
    m: &mut Meter,
) -> Result<(), Error> {
    let computed;
    let u = match given {
        Some(u) => u,
        None => {
            computed = Uniformity::of(cx.entry, cx.body, m)?;
            &computed
        }
    };
    let predication = match divergent_site(cx, strategy, u) {
        Some(_) => {
            divergent::plan(cx, strategy, u, divergent::SyncSet::Emulated, cross, m).map(Some)
        }
        None => Ok(None),
    };
    if given.is_none() {
        m.release(u.ledger_bytes());
    }
    let predication = predication?;
    // Metal moves no 64-bit value between lanes, and AIR has no `double` group operation, so no
    // exchange of a 64-bit form is lowered.
    let wide = strategy.iter().flatten().any(|s| match s {
        Strategy::Simd(SimdStrategy::Group(g)) => moved(g).is_some_and(|f| f.container() == 64),
        _ => false,
    });
    if wide {
        return Err(Unsupported::not_implemented(Feature::Lowering(LowerFamily::Simd)).into());
    }
    // An emulated site's result does not depend on subgroup reconvergence, nor does a threadgroup
    // barrier the lockstep pass placed where the threadgroup runs together.
    let other_tangled = cx.tangled.iter().any(|&v| {
        crate::lower::sync::barrier::reconverges(&cx.families.barrier, v)
            && !strategy
                .get(v.index())
                .and_then(Option::as_ref)
                .is_some_and(emulated_site)
    });
    if !other_tangled {
        cx.families.simd.mode = ReconvergenceMode::None;
        cx.families.simd.implementation_defined = false;
    }
    // The widest exchange: a site of a divergent block also publishes its lane's active flag (in
    // a uniform block every invocation is live or none is, so every lane is active and no flag
    // is published), and a predicated kernel's workgroup votes use the first word of the
    // scratch.
    let mut words = 0u32;
    for (b, block) in cx.body.blocks.iter() {
        let divergent = predication.as_ref().is_some_and(|p| !p.is_uniform(b));
        for id in block.inst_ids() {
            let s = cx
                .values
                .inst(id)
                .and_then(|v| strategy.get(v.index()))
                .and_then(Option::as_ref);
            let w = match s {
                Some(s) if divergent => predicated_words(s),
                Some(s) => exchange_words(s),
                None => None,
            };
            words = words.max(w.unwrap_or(0));
        }
    }
    if predication.is_some() {
        words = words.max(1);
    }
    cx.families.simd.exchange_words = words;
    cx.families.simd.predication = predication;
    if words == 0 {
        return Ok(());
    }
    let per_simdgroup = bytes_per_simdgroup(words);
    // The arena the module declares (`access::local::use_arena` checked its static part) and its
    // typed threadgroup objects.
    let arena = crate::lower::access::local::static_workgroup_bytes(cx);
    let required = arena_bytes(arena, 0).saturating_add(per_simdgroup);
    crate::lower::mesh::workgroup_limit(cx).check(required)?;
    cx.facts.insert(RuntimeFact::SimdgroupEmulation {
        bytes_per_simdgroup: u32::try_from(per_simdgroup).unwrap_or(u32::MAX),
        arena,
    });
    Ok(())
}

/// The first synchronization of the tier (`divergent::is_sync`: an emulated site, or a
/// threadgroup barrier) outside workgroup-uniform control flow, as `(block, instruction)`.
fn divergent_site(
    cx: &SelectCx<'_>,
    strategy: &[Option<Strategy>],
    u: &Uniformity,
) -> Option<(u32, u32)> {
    for (b, block) in cx.body.blocks.iter() {
        if u.is_uniform(b) {
            continue;
        }
        for id in block.inst_ids() {
            let site = cx
                .values
                .inst(id)
                .and_then(|v| strategy.get(v.index()))
                .and_then(Option::as_ref)
                .is_some_and(divergent::is_sync);
            if site {
                return Some((b.raw(), id.raw()));
            }
        }
    }
    None
}

/// The emulated tier's module-scope objects: the threadgroup's thread count and the scratch.
#[derive(Clone, Copy, Debug)]
pub struct Declared {
    /// `sx·sy·sz`, an `OpSpecConstantOp`.
    pub threads: ConstId,
    /// The scratch array, when the function exchanges values.
    pub scratch: Option<GlobalId>,
}

fn u32_ty(mb: &mut ModuleBuilder) -> BResult<TypeId> {
    mb.ty(Type::Int {
        width: 32,
        signed: false,
    })
}

fn konst(mb: &mut ModuleBuilder, v: u32) -> BResult<ConstId> {
    let ty = u32_ty(mb)?;
    mb.constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(v)),
    })
}

/// Declare the thread count and, for a function that exchanges `words` per lane, the scratch
/// in `Workgroup` memory: one slot per lane of `(N + 31) & ~31`, a `u32` or, for more than one
/// word, a `u32[words]`, so a word's pointer is one access chain at the lane and the constant
/// word with no slot arithmetic. The local-size constants are the dispatch family's for the `stage`
/// (`SpecKey::dispatch_keys`, default 1).
pub fn declare(mb: &mut ModuleBuilder, stage: Stage, words: u32) -> BResult<Declared> {
    let u32_ = u32_ty(mb)?;
    let [sx, sy, sz] = SpecKey::dispatch_keys(stage)
        .local
        .map(|k| mb.spec(k, u32_, ConstBits(1)));
    let xy = mb.spec_op(SpecOp::IMul, u32_, &[sx?, sy?])?;
    let threads = mb.spec_op(SpecOp::IMul, u32_, &[xy, sz?])?;
    if words == 0 {
        return Ok(Declared {
            threads,
            scratch: None,
        });
    }
    let up = konst(mb, SIMD_WIDTH - 1)?;
    let mask = konst(mb, !(SIMD_WIDTH - 1))?;
    let padded = mb.spec_op(SpecOp::IAdd, u32_, &[threads, up])?;
    let padded = mb.spec_op(SpecOp::BitwiseAnd, u32_, &[padded, mask])?;
    let slot = if words == 1 {
        u32_
    } else {
        let w = konst(mb, words)?;
        mb.ty(Type::Array {
            elem: u32_,
            len: w,
            stride: None,
        })?
    };
    let ty = mb.ty(Type::Array {
        elem: slot,
        len: padded,
        stride: None,
    })?;
    let scratch = mb.plain_global(StorageClass::Workgroup, ty, None)?;
    Ok(Declared {
        threads,
        scratch: Some(scratch),
    })
}

/// The emulated tier's exchange state in the open block.
#[derive(Clone, Copy, Debug)]
pub struct Emu {
    /// The threadgroup's threads.
    n: SValue,
    scratch: Option<GlobalId>,
    /// Words per lane.
    stride: u32,
    coherence: Coherence,
    sem: Semantics,
    /// In a divergent block of a predicated kernel, the calling lane's live flag, which every
    /// exchange publishes after its values: a lane whose flag is clear is inactive. `None`: every
    /// lane inside the threadgroup is active.
    active: Option<SValue>,
}

/// Words published by [`Emu::publish`]: each value's first word and type, and the index in
/// `layout` of the published active flag, if any.
pub struct Published {
    layout: Vec<(u32, TypeId)>,
    active: Option<usize>,
}

/// The exchange state of the plan's emulated tier.
pub(super) fn emu(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<Emu> {
    let Some(d) = cx.decl.families.simd.emu else {
        return Err(cx.fail("the emulated simdgroup tier is not declared"));
    };
    let n = fb.module().const_value(d.threads)?;
    let flags = MemFlags {
        threadgroup: true,
        ..MemFlags::default()
    };
    Ok(Emu {
        n,
        scratch: d.scratch,
        stride: cx.plan.families.simd.exchange_words,
        coherence: memory_model::coherence(cx.plan),
        sem: barrier::semantics(
            flags,
            MemoryOrder::AcquireRelease,
            barrier::memory_model(cx.plan),
        ),
        active: cx.active,
    })
}

/// Whether `live` holds for some invocation of the workgroup: every invocation clears the vote
/// word, a workgroup barrier, each sets it where `live` holds, a barrier, each reads it, and a
/// barrier keeps the next exchange or vote from writing it before every invocation has read. The
/// word is the scratch's first in the emulated tier, and the kernel's own vote word elsewhere
/// (`simd::Declared::vote`); it is only reached by atomics, which the barriers order. The result is
/// workgroup uniform; the caller is in workgroup-uniform control flow.
pub fn vote(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, live: SValue) -> BResult<SValue> {
    if let Some(word) = cx.decl.families.simd.vote {
        let flags = MemFlags {
            threadgroup: true,
            ..MemFlags::default()
        };
        let sem = barrier::semantics(
            flags,
            MemoryOrder::AcquireRelease,
            barrier::memory_model(cx.plan),
        );
        let zero = k(fb, 0)?;
        let one = k(fb, 1)?;
        let p = fb.module().global_value(word)?;
        return vote_at(fb, p, sem, live, (zero, one));
    }
    let e = emu(cx, fb)?;
    let zero = k(fb, 0)?;
    let one = k(fb, 1)?;
    let p = e.word_ptr(fb, zero, 0)?;
    vote_at(fb, p, e.sem, live, (zero, one))
}

/// [`vote`] in the word at `p`, with workgroup barriers of semantics `sem`.
fn vote_at(
    fb: &mut FunctionBuilder<'_>,
    p: SValue,
    sem: Semantics,
    live: SValue,
    (zero, one): (SValue, SValue),
) -> BResult<SValue> {
    let relaxed = Semantics::RELAXED;
    fb.atomic_store(p, Scope::Workgroup, relaxed, zero)?;
    fb.control_barrier(Scope::Workgroup, Scope::Workgroup, sem)?;
    let bit = fb.select(live, one, zero)?;
    fb.atomic(
        crate::spirv::ir::AtomicOp::Or,
        p,
        Scope::Workgroup,
        relaxed,
        bit,
    )?;
    fb.control_barrier(Scope::Workgroup, Scope::Workgroup, sem)?;
    let x = fb.atomic_load(p, Scope::Workgroup, relaxed)?;
    fb.control_barrier(Scope::Workgroup, Scope::Workgroup, sem)?;
    fb.binary(BinOp::INotEqual, x, zero)
}

/// Declare the workgroup word a predicated kernel outside the emulated tier votes in.
pub fn declare_vote(mb: &mut ModuleBuilder) -> BResult<GlobalId> {
    let u32_ = u32_ty(mb)?;
    mb.plain_global(StorageClass::Workgroup, u32_, None)
}

impl Emu {
    /// A workgroup barrier over the scratch.
    pub fn barrier(&self, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
        fb.control_barrier(Scope::Workgroup, Scope::Workgroup, self.sem)
    }

    /// A pointer to word `i` of lane `lane`'s slot: the slot's element `i` ([`declare`]).
    fn word_ptr(&self, fb: &mut FunctionBuilder<'_>, lane: SValue, i: u32) -> BResult<SValue> {
        let Some(g) = self.scratch else {
            return Err(dangling());
        };
        if i >= self.stride {
            return Err(dangling());
        }
        let base = fb.module().global_value(g)?;
        if self.stride == 1 {
            fb.access_chain(base, &[lane])
        } else {
            let i = k(fb, i)?;
            fb.access_chain(base, &[lane, i])
        }
    }

    /// Store the words of `vals` in order at the calling lane's slot from word 0 (no active
    /// flag, no barrier).
    fn put(&self, fb: &mut FunctionBuilder<'_>, lin: SValue, vals: &[SValue]) -> BResult<()> {
        let access = self.coherence.access(fb, MemAccess::default(), true)?;
        let mut at = 0u32;
        for v in vals {
            for w in to_words(fb, *v)? {
                let p = self.word_ptr(fb, lin, at)?;
                fb.store(p, w, access)?;
                at += 1;
            }
        }
        Ok(())
    }

    /// The simdgroup's word of each of `conds`: bit `j` of word `c` is set where Metal lane `j`
    /// (of the lanes that execute this) holds `conds[c]`. Every lane stores `vals` at its slot
    /// from word 0 and atomically clears the `conds.len()` words after them, a workgroup barrier,
    /// each lane atomically ORs its bit into those words of the simdgroup's first lane `base`, a
    /// barrier, and each lane loads them. OR is commutative, so every lane of the simdgroup reads
    /// the same words whatever the order of the atomics. The caller may read `vals` from any lane
    /// of the threadgroup that stored them and closes the exchange with [`Emu::barrier`].
    fn gather_bits(
        &self,
        fb: &mut FunctionBuilder<'_>,
        ln: &Lane,
        base: SValue,
        vals: &[SValue],
        conds: &[SValue],
    ) -> BResult<Vec<SValue>> {
        let lin = ln.l;
        let access = self.coherence.access(fb, MemAccess::default(), true)?;
        let mut at = 0u32;
        for v in vals {
            for w in to_words(fb, *v)? {
                let p = self.word_ptr(fb, lin, at)?;
                fb.store(p, w, access)?;
                at += 1;
            }
        }
        let zero = k(fb, 0)?;
        let relaxed = Semantics::RELAXED;
        for c in 0..conds.len() as u32 {
            let p = self.word_ptr(fb, lin, at + c)?;
            fb.atomic_store(p, Scope::Workgroup, relaxed, zero)?;
        }
        self.barrier(fb)?;
        let m = ln.metal(fb)?;
        let one = k(fb, 1)?;
        let bit = fb.binary(BinOp::ShiftLeftLogical, one, m)?;
        let mut ptrs = Vec::with_capacity(conds.len());
        for (c, cond) in conds.iter().enumerate() {
            let b = fb.select(*cond, bit, zero)?;
            let p = self.word_ptr(fb, base, at + c as u32)?;
            fb.atomic(
                crate::spirv::ir::AtomicOp::Or,
                p,
                Scope::Workgroup,
                relaxed,
                b,
            )?;
            ptrs.push(p);
        }
        self.barrier(fb)?;
        let mut words = Vec::with_capacity(ptrs.len());
        for p in ptrs {
            words.push(fb.atomic_load(p, Scope::Workgroup, relaxed)?);
        }
        Ok(words)
    }

    /// The value of type `ty` lane `lane` stored from word 0 of its slot.
    fn get(&self, fb: &mut FunctionBuilder<'_>, lane: SValue, ty: TypeId) -> BResult<SValue> {
        self.load_value(fb, lane, 0, ty)
    }

    /// Store the calling lane's `vals` (and its active flag, in a divergent block) to its slot,
    /// then a workgroup barrier.
    pub fn publish(
        &self,
        fb: &mut FunctionBuilder<'_>,
        lin: SValue,
        vals: &[SValue],
    ) -> BResult<Published> {
        let mut layout = Vec::with_capacity(vals.len() + 1);
        let mut at = 0u32;
        let access = self.coherence.access(fb, MemAccess::default(), true)?;
        let flag = self.active.map(|a| (vals.len(), a));
        for v in vals.iter().copied().chain(flag.map(|(_, a)| a)) {
            let v = &v;
            let ty = fb.value_ty(*v)?;
            let words = to_words(fb, *v)?;
            layout.push((at, ty));
            for w in words {
                if at >= self.stride {
                    return Err(dangling());
                }
                let p = self.word_ptr(fb, lin, at)?;
                fb.store(p, w, access)?;
                at += 1;
            }
        }
        self.barrier(fb)?;
        Ok(Published {
            layout,
            active: flag.map(|(i, _)| i),
        })
    }

    /// Value `i` of `p` as lane `src` published it; a lane outside the threadgroup or inactive
    /// reads the calling lane's. The reductions read through it and keep only active lanes'
    /// values; a shuffle reads through [`Emu::read_lane`].
    pub fn read(
        &self,
        fb: &mut FunctionBuilder<'_>,
        lin: SValue,
        p: &Published,
        i: usize,
        src: SValue,
    ) -> BResult<SValue> {
        self.read_on(fb, lin, p, i, src).map(|(x, _)| x)
    }

    /// Value `i` of `p` as a shuffle reads it from lane `src`. A lane past the thread count does
    /// not exist, and reads all-zero bits ([`zero_cell`]): Metal's answer, deterministic across
    /// runs, inputs, and threadgroup shapes, where it leaves the value undefined. That amends the
    /// calling-lane rule; it is measured for every move. A lane below the thread count that is not
    /// active in a divergent block reads all-zero bits too: Metal's answer for every move, whether
    /// the value was computed before the branch or inside it (amending the calling-lane
    /// refinement).
    pub fn read_lane(
        &self,
        fb: &mut FunctionBuilder<'_>,
        lin: SValue,
        p: &Published,
        i: usize,
        src: SValue,
    ) -> BResult<SValue> {
        let (first, ty) = p.layout.get(i).copied().ok_or_else(dangling)?;
        let (at, on) = self.source(fb, lin, p, src)?;
        let x = self.load_value(fb, at, first, ty)?;
        let zero = zero_cell(fb, x)?;
        fb.select(on, x, zero)
    }

    /// A multiply-add's cell `i` of `p` as lane `src` published it (`matrix::tile_mad`). A lane
    /// that is not active takes part with a zero cell: one outside the threadgroup, and, in a
    /// divergent block, one that is not live. Metal leaves a partial or partly live simdgroup's
    /// product undefined; its answer, deterministic across runs, inputs, and dispatch shapes, is
    /// the product with those lanes' cells zero, whatever they hold (measured for the matrix form;
    /// the AGX form shares the rule unmeasured). The zero is `+0`, Metal's sign. That amends the
    /// calling-lane rule, which [`Emu::read`] keeps, for the multiply-adds.
    pub fn read_cell(
        &self,
        fb: &mut FunctionBuilder<'_>,
        lin: SValue,
        p: &Published,
        i: usize,
        src: SValue,
    ) -> BResult<SValue> {
        let (x, on) = self.read_on(fb, lin, p, i, src)?;
        let zero = zero_cell(fb, x)?;
        fb.select(on, x, zero)
    }

    /// Value `i` of `p` as lane `src` published it (the calling lane's where `src` is outside the
    /// threadgroup or inactive), and whether `src` is an active lane of the threadgroup.
    fn read_on(
        &self,
        fb: &mut FunctionBuilder<'_>,
        lin: SValue,
        p: &Published,
        i: usize,
        src: SValue,
    ) -> BResult<(SValue, SValue)> {
        let (first, ty) = p.layout.get(i).copied().ok_or_else(dangling)?;
        let (src, on) = self.source(fb, lin, p, src)?;
        let x = self.load_value(fb, src, first, ty)?;
        Ok((x, on))
    }

    /// Whether lane `src` is an active lane of the threadgroup, and the lane to read in its
    /// place: `src` when it is, else the calling lane.
    fn source(
        &self,
        fb: &mut FunctionBuilder<'_>,
        lin: SValue,
        p: &Published,
        src: SValue,
    ) -> BResult<(SValue, SValue)> {
        let inside = fb.binary(BinOp::ULessThan, src, self.n)?;
        let at = fb.select(inside, src, lin)?;
        let Some(ai) = p.active else {
            return Ok((at, inside));
        };
        let (first, ty) = p.layout.get(ai).copied().ok_or_else(dangling)?;
        let a = self.load_value(fb, at, first, ty)?;
        let no = bool_k(fb, false)?;
        let on = fb.select(inside, a, no)?;
        let at = fb.select(on, src, lin)?;
        Ok((at, on))
    }

    /// The value of type `ty` whose words start at word `first` of lane `lane`'s slot.
    fn load_value(
        &self,
        fb: &mut FunctionBuilder<'_>,
        lane: SValue,
        first: u32,
        ty: TypeId,
    ) -> BResult<SValue> {
        let access = self.coherence.access(fb, MemAccess::default(), false)?;
        let count = word_count(fb, ty)?;
        let mut words = Vec::with_capacity(count as usize);
        for j in 0..count {
            let ptr = self.word_ptr(fb, lane, first + j)?;
            words.push(fb.load(ptr, access)?);
        }
        from_words(fb, ty, &words)
    }

    /// Whether lane `src` is an active lane of the threadgroup: inside it, and, in a divergent
    /// block, with its published active flag set.
    fn on(
        &self,
        fb: &mut FunctionBuilder<'_>,
        lin: SValue,
        p: &Published,
        src: SValue,
    ) -> BResult<SValue> {
        self.source(fb, lin, p, src).map(|(_, on)| on)
    }

    /// Whether lane `src` is inside the threadgroup (active, in uniform control flow).
    fn exists(&self, fb: &mut FunctionBuilder<'_>, src: SValue) -> BResult<SValue> {
        fb.binary(BinOp::ULessThan, src, self.n)
    }
}

fn bool_k(fb: &mut FunctionBuilder<'_>, v: bool) -> BResult<SValue> {
    let ty = fb.module().ty(Type::Bool)?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(v)),
    })?;
    fb.module().const_value(c)
}

fn u32_value_ty(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(Type::Int {
        width: 32,
        signed: false,
    })
}

/// The `u32` words of scalar or vector `v` (each component in order, a `bool` as 0 or 1).
fn to_words(fb: &mut FunctionBuilder<'_>, v: SValue) -> BResult<Vec<SValue>> {
    let ty = fb.value_ty(v)?;
    match fb.module().types().get(ty)?.clone() {
        Type::Vector { elem, len } => {
            let mut out = Vec::new();
            for i in 0..u32::from(len) {
                let c = fb.composite_extract(v, &[i])?;
                out.extend(scalar_words(fb, c, elem)?);
            }
            Ok(out)
        }
        _ => scalar_words(fb, v, ty),
    }
}

fn scalar_words(fb: &mut FunctionBuilder<'_>, v: SValue, ty: TypeId) -> BResult<Vec<SValue>> {
    let u32_ = u32_value_ty(fb)?;
    match fb.module().types().get(ty)?.clone() {
        Type::Int { width: 32, .. } | Type::Float { width: 32 } => {
            if ty == u32_ {
                Ok(vec![v])
            } else {
                Ok(vec![fb.convert(ConvOp::Bitcast, v, u32_)?])
            }
        }
        Type::Bool => {
            let (one, zero) = (k(fb, 1)?, k(fb, 0)?);
            Ok(vec![fb.select(v, one, zero)?])
        }
        _ => Err(dangling()),
    }
}

/// The words of a value of type `ty`: one per component.
fn word_count(fb: &mut FunctionBuilder<'_>, ty: TypeId) -> BResult<u32> {
    match fb.module().types().get(ty)?.clone() {
        Type::Vector { len, .. } => Ok(u32::from(len)),
        _ => Ok(1),
    }
}

/// The value of type `ty` whose words are `words` ([`to_words`]'s inverse).
fn from_words(fb: &mut FunctionBuilder<'_>, ty: TypeId, words: &[SValue]) -> BResult<SValue> {
    match fb.module().types().get(ty)?.clone() {
        Type::Vector { elem, len } => {
            let mut parts = Vec::with_capacity(usize::from(len));
            for i in 0..usize::from(len) {
                let w = words.get(i).copied().ok_or_else(dangling)?;
                parts.push(scalar_from_word(fb, elem, w)?);
            }
            fb.composite_construct(ty, &parts)
        }
        _ => scalar_from_word(fb, ty, nth(words, 0)?),
    }
}

fn scalar_from_word(fb: &mut FunctionBuilder<'_>, ty: TypeId, x: SValue) -> BResult<SValue> {
    let u32_ = u32_value_ty(fb)?;
    match fb.module().types().get(ty)?.clone() {
        Type::Int { width: 32, .. } | Type::Float { width: 32 } if ty == u32_ => Ok(x),
        Type::Int { width: 32, .. } | Type::Float { width: 32 } => {
            fb.convert(ConvOp::Bitcast, x, ty)
        }
        Type::Bool => {
            let zero = k(fb, 0)?;
            fb.binary(BinOp::INotEqual, x, zero)
        }
        _ => Err(dangling()),
    }
}

/// `v` published, read from lane `src`, and the exchange closed.
fn one(fb: &mut FunctionBuilder<'_>, ln: &Lane, v: SValue, src: SValue) -> BResult<SValue> {
    let ex = ln.exchange(fb, &[v])?;
    let x = ln.read(fb, &ex, 0, src)?;
    ln.close(fb, ex)?;
    Ok(x)
}

/// `v` published, read by a shuffle from lane `src` ([`Emu::read_lane`]: a lane past the thread
/// count reads all-zero bits), and the exchange closed.
fn shuffled(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    e: &Emu,
    v: SValue,
    src: SValue,
) -> BResult<SValue> {
    let ex = ln.exchange(fb, &[v])?;
    let x = e.read_lane(fb, ln.l, published(&ex)?, 0, src)?;
    ln.close(fb, ex)?;
    Ok(x)
}

/// A shuffle-and-fill of `g`'s scope (`group::fill_source`): the data and the fill published, each
/// read from the source lane (one past the thread count reads all-zero bits), and the data taken
/// where the source is inside the segment.
fn shift_and_fill(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    e: &Emu,
    g: &GroupPlan,
    a: &[SValue],
) -> BResult<SValue> {
    let (data, fill, d, modulo) = (nth(a, 0)?, nth(a, 1)?, nth(a, 2)?, nth(a, 3)?);
    let up = g.op == GroupOp::ShuffleAndFillUp;
    let (src, inside) = fill_source(fb, ln, g, up, d, modulo)?;
    let ex = ln.exchange(fb, &[data, fill])?;
    let p = published(&ex)?;
    let x = e.read_lane(fb, ln.l, p, 0, src)?;
    let y = e.read_lane(fb, ln.l, p, 1, src)?;
    ln.close(fb, ex)?;
    fb.select(inside, x, y)
}

/// A quad prefix sum or product: the quad's lanes read one by one in lane order, Metal's order
/// (measured), each taken where it lies below the calling lane (or is it, inclusive), exists, and
/// in a divergent block is active. The empty answer is the operation's
/// [`identity`]: `-0` for a float sum and 1 for a product, Metal's for the first active lane.
fn quad_scan(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    e: &Emu,
    g: &GroupPlan,
    v: SValue,
    qbase: SValue,
    q: SValue,
) -> BResult<SValue> {
    let inclusive = !super::group::exclusive(g.op);
    let ex = ln.exchange(fb, &[v])?;
    let mut acc = identity(fb, g)?;
    let mut some = None;
    for j in 0..4 {
        let src = bin_k(fb, BinOp::IAdd, qbase, j)?;
        let at = k(fb, j)?;
        let below = if inclusive {
            fb.binary(BinOp::ULessThanEqual, at, q)?
        } else {
            fb.binary(BinOp::ULessThan, at, q)?
        };
        let (y, on) = match e.active {
            None => {
                let on = e.exists(fb, src)?;
                (ln.read(fb, &ex, 0, src)?, on)
            }
            Some(_) => e.read_on(fb, ln.l, published(&ex)?, 0, src)?,
        };
        let take = fb.binary(BinOp::LogicalAnd, below, on)?;
        let c = combine(fb, g, acc, y)?;
        acc = fb.select(take, c, acc)?;
        some = Some(match some {
            Some(s) => fb.binary(BinOp::LogicalOr, s, take)?,
            None => take,
        });
    }
    ln.close(fb, ex)?;
    if let (Some(GroupArith::FAdd), false, Some(some)) = (g.arith, inclusive, some) {
        acc = minus_zero_unless(fb, g.form, some, acc)?;
    }
    post(fb, g, acc)
}

/// `a op b` for the combining operation of a reduction.
pub(super) fn combine(
    fb: &mut FunctionBuilder<'_>,
    g: &GroupPlan,
    a: SValue,
    b: SValue,
) -> BResult<SValue> {
    use GroupArith as A;
    match g.arith.ok_or_else(dangling)? {
        A::IAdd => fb.binary(BinOp::IAdd, a, b),
        A::FAdd => {
            let s = fb.binary(BinOp::FAdd, a, b)?;
            narrow::round(fb, g.form, s)
        }
        A::IMul => fb.binary(BinOp::IMul, a, b),
        A::FMul => {
            let p = fb.binary(BinOp::FMul, a, b)?;
            narrow::round(fb, g.form, p)
        }
        A::SMin => fb.glsl(GlslOp::SMin, &[a, b]),
        A::UMin => fb.glsl(GlslOp::UMin, &[a, b]),
        A::FMin => fb.glsl(GlslOp::NMin, &[a, b]),
        A::SMax => fb.glsl(GlslOp::SMax, &[a, b]),
        A::UMax => fb.glsl(GlslOp::UMax, &[a, b]),
        A::FMax => fb.glsl(GlslOp::NMax, &[a, b]),
        A::BitwiseAnd => fb.binary(BinOp::BitwiseAnd, a, b),
        A::BitwiseOr => fb.binary(BinOp::BitwiseOr, a, b),
        A::BitwiseXor => fb.binary(BinOp::BitwiseXor, a, b),
        A::LogicalAnd => fb.binary(BinOp::LogicalAnd, a, b),
        A::LogicalOr => fb.binary(BinOp::LogicalOr, a, b),
        A::LogicalXor => fb.binary(BinOp::LogicalNotEqual, a, b),
    }
}

/// The scratch words an exchange of the emulated tier published.
fn published(ex: &Exchange) -> BResult<&Published> {
    match ex {
        Exchange::Workgroup(p) => Ok(p),
        Exchange::Subgroup(_) => Err(dangling()),
    }
}

/// A quad reduction: the quad's four lanes, each taking part where it exists and in a divergent
/// block is active, combined in Metal's order ([`quad_tree`]).
fn quad_fold(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    e: &Emu,
    g: &GroupPlan,
    v: SValue,
    qbase: SValue,
) -> BResult<SValue> {
    let x = pre(fb, g, v)?;
    let ex = ln.exchange(fb, &[x])?;
    let mut lanes = Vec::with_capacity(4);
    for j in 0..4 {
        let src = bin_k(fb, BinOp::IAdd, qbase, j)?;
        lanes.push(match e.active {
            None => {
                let on = e.exists(fb, src)?;
                (ln.read(fb, &ex, 0, src)?, on)
            }
            Some(_) => e.read_on(fb, ln.l, published(&ex)?, 0, src)?,
        });
    }
    ln.close(fb, ex)?;
    let r = quad_tree(fb, g, &lanes)?;
    post(fb, g, r)
}

/// The simdgroup's 32-bit word of `pred` (bit `j` is Metal lane `j`, set for an active lane
/// whose `pred` holds), and with `act` the word of its active lanes (in a divergent block; `None`
/// otherwise): each lane ORs its bit into the simdgroup's words ([`Emu::gather_bits`]), so the word
/// costs a few instructions instead of a read per lane.
fn word(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    e: &Emu,
    base: SValue,
    pred: SValue,
    act: bool,
) -> BResult<(SValue, Option<SValue>)> {
    let (set, live) = match e.active {
        Some(a) => {
            let no = bool_k(fb, false)?;
            (fb.select(a, pred, no)?, Some(a))
        }
        None => (pred, None),
    };
    let conds: Vec<SValue> = match live {
        Some(a) if act => vec![set, a],
        _ => vec![set],
    };
    let words = e.gather_bits(fb, ln, base, &[], &conds)?;
    e.barrier(fb)?;
    let w = nth(&words, 0)?;
    Ok((w, (conds.len() == 2).then(|| nth(&words, 1)).transpose()?))
}

/// The identity of a reduction's combining operation, of type `ty` (the value `pre` gives): `-0`
/// for a float sum, 1 for a product, and the neutral integer otherwise. A float min or max has none: a NaN is one
/// of `NMin` and `NMax` only under `SignedZeroInfNanPreserve`, without which a device may assume
/// that no operand is a NaN, so their rounds skip an absent partial instead. `-0` is neither a NaN
/// nor an infinity, and `-0 + x` is `x` for every `x` where signed zeros are preserved; where they
/// are not, the device may give any zero sum either sign, the operands' own included, so the
/// identity changes no result the device's arithmetic does not.
fn identity(fb: &mut FunctionBuilder<'_>, g: &GroupPlan) -> BResult<SValue> {
    use GroupArith as A;
    let f = g.form;
    match g.arith.ok_or_else(dangling)? {
        A::FAdd => ops::fsplat(fb, f, -0.0),
        A::FMul => ops::fsplat(fb, f, 1.0),
        A::IMul => ops::splat(fb, f, 1),
        A::IAdd | A::UMax | A::BitwiseOr | A::BitwiseXor => ops::splat(fb, f, 0),
        A::UMin | A::BitwiseAnd => ops::splat(fb, f, u64::from(u32::MAX)),
        A::SMin => ops::splat(fb, f, 0x7FFF_FFFF),
        A::SMax => ops::splat(fb, f, 0x8000_0000),
        _ => Err(dangling()),
    }
}

/// The distances of the simdgroup's tree rounds: 1, 2, 4, 8, and 16.
fn distances() -> impl Iterator<Item = u32> {
    (0..SIMD_WIDTH.trailing_zeros()).map(|r| 1u32 << r)
}

/// A simdgroup reduction as a tree of five exchange rounds. In round `d` (1, 2, 4, 8, 16) each
/// lane publishes its partial, reads the partial of the first lane of its sibling block of `d`
/// lanes (`(lin ^ d) & !(d - 1)`), and combines the lower block's partial with the upper one's
/// where the sibling block holds a contributing lane, else keeps its own. After round `d` every
/// lane holds the reduction of the contributing lanes of its aligned block of `2d` lanes, the
/// same value in each of them, so after five rounds every lane holds the simdgroup's.
///
/// In a workgroup-uniform block the lanes below the thread count contribute, and a block holds
/// one exactly when its first lane is below it. In a divergent block an inactive lane takes part
/// with the operation's [`identity`], except in a float min or max, which has none: there an
/// inactive lane takes part with the operand of the simdgroup's first active lane, which one
/// exchange gathers (`Emu::gather_bits`). `min` and `max` are idempotent, so the copies change no
/// result: the rounds reduce the active lanes' operands and copies of one of them, whose minimum or
/// maximum is the active lanes', no value but an active lane's reaches `NMin` or `NMax` (so no NaN
/// the kernel did not compute), and a `±0` result is the zero of an active lane. Which zero, when
/// the active lanes hold both, is the device's: without `SPV_KHR_float_controls2`, `GLSL.std.450`
/// does not say which operand `NMin` and `NMax` return for equal zeros, so a device that returns
/// the second can answer `+0` where the per-round flags answered `-0` (Metal's choice is not
/// measured). A lane with no active lane in its simdgroup is not live and keeps its own operand.
/// Integer operations are associative and commutative, so the result is exact; a float sum's
/// association order is the tree's, which `ReductionOrder` (recorded in every tier) allows; a float
/// min or max combines its operands in lane order, so a `±0` tie resolves the same way in every
/// lane.
fn tree_fold(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    e: &Emu,
    g: &GroupPlan,
    v: SValue,
    base: SValue,
) -> BResult<SValue> {
    let lin = ln.l;
    let x = pre(fb, g, v)?;
    let ty = fb.value_ty(x)?;
    let ordered = matches!(g.arith, Some(GroupArith::FMin | GroupArith::FMax));
    let zero = k(fb, 0)?;
    let mut acc = match e.active {
        Some(a) if ordered => {
            let words = e.gather_bits(fb, ln, base, &[x], &[a])?;
            let act = nth(&words, 0)?;
            let first = fb.glsl(GlslOp::FindILsb, &[act])?;
            let first = bin_k(fb, BinOp::BitwiseAnd, first, SIMD_WIDTH - 1)?;
            let at = fb.binary(BinOp::BitwiseOr, base, first)?;
            let some = fb.binary(BinOp::INotEqual, act, zero)?;
            let src = fb.select(some, at, lin)?;
            let y = e.get(fb, src, ty)?;
            e.barrier(fb)?;
            fb.select(a, x, y)?
        }
        Some(a) => {
            let id = identity(fb, g)?;
            fb.select(a, x, id)?
        }
        None => x,
    };
    for d in distances() {
        e.put(fb, lin, &[acc])?;
        e.barrier(fb)?;
        let sib = bin_k(fb, BinOp::BitwiseXor, lin, d)?;
        let sib = if d > 1 {
            bin_k(fb, BinOp::BitwiseAnd, sib, !(d - 1))?
        } else {
            sib
        };
        let there = e.exists(fb, sib)?;
        let src = fb.select(there, sib, lin)?;
        let y = e.get(fb, src, ty)?;
        e.barrier(fb)?;
        let r = if ordered {
            let bit = bin_k(fb, BinOp::BitwiseAnd, lin, d)?;
            let upper = fb.binary(BinOp::INotEqual, bit, zero)?;
            let lo = fb.select(upper, y, acc)?;
            let hi = fb.select(upper, acc, y)?;
            combine(fb, g, lo, hi)?
        } else {
            combine(fb, g, acc, y)?
        };
        acc = fb.select(there, r, acc)?;
    }
    post(fb, g, acc)
}

/// A prefix sum or product as five exchange rounds (Hillis and Steele). Every lane starts with
/// its operand, or the operation's [`identity`] where it is not active (a divergent block): 0, or
/// `-0` for a float sum, and 1 for a product. In round `d` (1, 2, 4, 8, 16) each lane publishes its
/// partial and combines the partial of the lane `d` below it where that lane is in its simdgroup
/// (Metal lane `m >= d`), else keeps its own. After round `d` a lane holds the sum or product of
/// the lanes of its simdgroup in `(m - 2d, m]`, so after five rounds the inclusive one.
///
/// The exclusive integer sum subtracts the lane's own operand; integer addition wraps, so both
/// are exact. A float's exclusive sum and every exclusive product instead shift the operands up
/// one lane first (lane `m` starts with lane `m - 1`'s operand, and lane 0 with none, which no
/// round reads), so nothing is subtracted or divided, and are the empty answer where no active
/// lane lies below the caller: `+0` for a float sum, the identity of Vulkan's scans, and 1 for a
/// product, Metal's (measured); no active lane lies below the caller at every lane below it in a
/// workgroup-uniform block, or by the active lanes' word ([`Emu::gather_bits`]) in a divergent one.
/// Integer products wrap, so they are exact. A float sum's or product's association order is the
/// rounds' (`ReductionOrder`, recorded as in every tier), a narrow one rounded after each step.
fn tree_scan(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    e: &Emu,
    g: &GroupPlan,
    v: SValue,
    base: SValue,
    m: SValue,
) -> BResult<SValue> {
    let lin = ln.l;
    let float = g.form.is_float();
    let exclusive = super::group::exclusive(g.op);
    let product = matches!(g.arith, Some(GroupArith::IMul | GroupArith::FMul));
    let shifted = exclusive && (float || product);
    let x = match e.active {
        Some(a) => {
            let id = identity(fb, g)?;
            fb.select(a, v, id)?
        }
        None => v,
    };
    let ty = fb.value_ty(x)?;
    // The partial of the lane `d` below the caller, and whether that lane is in the simdgroup
    // from Metal lane `d + skip` on.
    let below = |fb: &mut FunctionBuilder<'_>,
                 acc: SValue,
                 d: u32,
                 skip: u32|
     -> BResult<(SValue, SValue)> {
        e.put(fb, lin, &[acc])?;
        e.barrier(fb)?;
        let need = k(fb, d + skip)?;
        let ok = fb.binary(BinOp::UGreaterThanEqual, m, need)?;
        let dv = k(fb, d)?;
        let from = fb.binary(BinOp::ISub, lin, dv)?;
        let src = fb.select(ok, from, lin)?;
        let y = e.get(fb, src, ty)?;
        e.barrier(fb)?;
        Ok((y, ok))
    };
    // After an exclusive scan's shift, lane `m` holds lane `m - 1`'s operand, so the lane `d`
    // below holds one only from Metal lane `d + 1` on.
    let skip = u32::from(shifted);
    let mut acc = x;
    if skip == 1 {
        acc = below(fb, x, 1, 0)?.0;
    }
    for d in distances() {
        let (y, ok) = below(fb, acc, d, skip)?;
        let r = combine(fb, g, y, acc)?;
        acc = fb.select(ok, r, acc)?;
    }
    if exclusive {
        acc = if shifted {
            let zero = k(fb, 0)?;
            let some = match e.active {
                Some(a) => {
                    let words = e.gather_bits(fb, ln, base, &[], &[a])?;
                    e.barrier(fb)?;
                    let one = k(fb, 1)?;
                    let bit = fb.binary(BinOp::ShiftLeftLogical, one, m)?;
                    let lower = fb.binary(BinOp::ISub, bit, one)?;
                    let act = fb.binary(BinOp::BitwiseAnd, nth(&words, 0)?, lower)?;
                    fb.binary(BinOp::INotEqual, act, zero)?
                }
                None => fb.binary(BinOp::INotEqual, m, zero)?,
            };
            let empty = match (product, float) {
                (true, true) => ops::fsplat(fb, g.form, 1.0)?,
                (true, false) => ops::splat(fb, g.form, 1)?,
                _ => ops::fsplat(fb, g.form, 0.0)?,
            };
            fb.select(some, acc, empty)?
        } else {
            fb.binary(BinOp::ISub, acc, x)?
        };
    }
    post(fb, g, acc)
}

/// In a divergent block, the word of the active lanes of the `span` (at most 32) lanes from `first`
/// (bit `j` is lane `first + j`), from the active flags `ex` published.
fn active_bits(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    e: &Emu,
    ex: &Exchange,
    first: SValue,
    span: u32,
) -> BResult<SValue> {
    let p = published(ex)?;
    let zero = k(fb, 0)?;
    let mut a = zero;
    for j in 0..span {
        let src = bin_k(fb, BinOp::IAdd, first, j)?;
        let on = e.on(fb, ln.l, p, src)?;
        let bit = k(fb, 1 << j)?;
        let bit = fb.select(on, bit, zero)?;
        a = fb.binary(BinOp::BitwiseOr, a, bit)?;
    }
    Ok(a)
}

/// The mask of the lanes of a group of `span` (at most 32) lanes from `first` below the thread
/// count: `(1 << min(N - first, span)) - 1`.
fn present(fb: &mut FunctionBuilder<'_>, e: &Emu, first: SValue, span: u32) -> BResult<SValue> {
    let left = fb.binary(BinOp::ISub, e.n, first)?;
    let span_v = k(fb, span)?;
    let full = fb.binary(BinOp::UGreaterThanEqual, left, span_v)?;
    let zero = k(fb, 0)?;
    let shift = fb.select(full, zero, left)?;
    let one = k(fb, 1)?;
    let partial = fb.binary(BinOp::ShiftLeftLogical, one, shift)?;
    let partial = fb.binary(BinOp::ISub, partial, one)?;
    let all = k(
        fb,
        if span >= 32 {
            u32::MAX
        } else {
            (1 << span) - 1
        },
    )?;
    fb.select(full, all, partial)
}

/// A simdgroup operation in the emulated tier (see the module documentation).
pub(super) fn simd_op(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    e: &Emu,
    g: &GroupPlan,
    a: &[SValue],
) -> BResult<SValue> {
    use GroupOp as G;
    let v = || nth(a, 0);
    let last = SIMD_WIDTH - 1;
    let lin = ln.l;
    let m = ln.metal(fb)?;
    let zero = k(fb, 0)?;
    let base = ln.absolute(fb, zero)?;
    match g.op {
        G::Broadcast | G::Shuffle => {
            let i = bin_k(fb, BinOp::BitwiseAnd, nth(a, 1)?, last)?;
            let src = fb.binary(BinOp::BitwiseOr, base, i)?;
            shuffled(fb, ln, e, v()?, src)
        }
        G::ShuffleXor => {
            let mask = bin_k(fb, BinOp::BitwiseAnd, nth(a, 1)?, last)?;
            let src = fb.binary(BinOp::BitwiseXor, lin, mask)?;
            shuffled(fb, ln, e, v()?, src)
        }
        // The source is at or below the calling lane, which exists: never past the thread
        // count.
        G::ShuffleUp => {
            let d = nth(a, 1)?;
            let ok = fb.binary(BinOp::ULessThanEqual, d, m)?;
            let from = fb.binary(BinOp::ISub, lin, d)?;
            let src = fb.select(ok, from, lin)?;
            one(fb, ln, v()?, src)
        }
        G::ShuffleDown => {
            let d = nth(a, 1)?;
            let t = fb.binary(BinOp::IAdd, m, d)?;
            let w = k(fb, SIMD_WIDTH)?;
            let ok = fb.binary(BinOp::ULessThan, t, w)?;
            let from = fb.binary(BinOp::IAdd, lin, d)?;
            let src = fb.select(ok, from, lin)?;
            shuffled(fb, ln, e, v()?, src)
        }
        G::ShuffleRotateDown => {
            let t = fb.binary(BinOp::IAdd, m, nth(a, 1)?)?;
            let t = bin_k(fb, BinOp::BitwiseAnd, t, last)?;
            let src = fb.binary(BinOp::BitwiseOr, base, t)?;
            shuffled(fb, ln, e, v()?, src)
        }
        G::ShuffleRotateUp => {
            let t = fb.binary(BinOp::ISub, m, nth(a, 1)?)?;
            let t = bin_k(fb, BinOp::BitwiseAnd, t, last)?;
            let src = fb.binary(BinOp::BitwiseOr, base, t)?;
            shuffled(fb, ln, e, v()?, src)
        }
        G::ShuffleAndFillUp | G::ShuffleAndFillDown => shift_and_fill(fb, ln, e, g, a),
        // In a divergent block the first active lane (none: the calling lane, which is not
        // live); else every lane of the simdgroup below the thread count is active and lane 0 is
        // the first.
        G::BroadcastFirst => match e.active {
            Some(a) => {
                let x = v()?;
                let ty = fb.value_ty(x)?;
                let words = e.gather_bits(fb, ln, base, &[x], &[a])?;
                let act = nth(&words, 0)?;
                // No active lane: the calling lane, which is then not live, reads its own.
                let first = fb.glsl(GlslOp::FindILsb, &[act])?;
                let first = bin_k(fb, BinOp::BitwiseAnd, first, last)?;
                let at = fb.binary(BinOp::BitwiseOr, base, first)?;
                let some = fb.binary(BinOp::INotEqual, act, zero)?;
                let src = fb.select(some, at, lin)?;
                let x = e.get(fb, src, ty)?;
                e.barrier(fb)?;
                Ok(x)
            }
            None => one(fb, ln, v()?, base),
        },
        G::IsFirst => match e.active {
            Some(a) => {
                let words = e.gather_bits(fb, ln, base, &[], &[a])?;
                e.barrier(fb)?;
                let first = fb.glsl(GlslOp::FindILsb, &[nth(&words, 0)?])?;
                fb.binary(BinOp::IEqual, first, m)
            }
            None => fb.binary(BinOp::IEqual, lin, base),
        },
        G::PrefixInclusiveSum
        | G::PrefixExclusiveSum
        | G::PrefixInclusiveProduct
        | G::PrefixExclusiveProduct => tree_scan(fb, ln, e, g, v()?, base, m),
        G::Sum | G::Product | G::Min | G::Max | G::And | G::Or | G::Xor => {
            tree_fold(fb, ln, e, g, v()?, base)
        }
        G::All | G::Any => {
            let all = g.op == G::All;
            let (w, act) = word(fb, ln, e, base, v()?, all)?;
            if all {
                let act = match act {
                    Some(a) => a,
                    None => present(fb, e, base, SIMD_WIDTH)?,
                };
                fb.binary(BinOp::IEqual, w, act)
            } else {
                fb.binary(BinOp::INotEqual, w, zero)
            }
        }
        G::Ballot => {
            let (w, _) = word(fb, ln, e, base, v()?, false)?;
            wide::zext_word(fb, g.form, w)
        }
        // The ballot of `true`: in a divergent block the active lanes; else every lane of the
        // simdgroup below the thread count.
        G::ActiveThreadsMask => {
            let w = match e.active {
                Some(_) => {
                    let yes = bool_k(fb, true)?;
                    word(fb, ln, e, base, yes, false)?.0
                }
                None => present(fb, e, base, SIMD_WIDTH)?,
            };
            wide::zext_word(fb, g.form, w)
        }
    }
}

/// A quad operation in the emulated tier: quads are the aligned four linear threads.
pub(super) fn quad_op(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    e: &Emu,
    g: &GroupPlan,
    a: &[SValue],
) -> BResult<SValue> {
    use GroupOp as G;
    let v = || nth(a, 0);
    let lin = ln.l;
    let q = bin_k(fb, BinOp::BitwiseAnd, lin, 3)?;
    let qbase = bin_k(fb, BinOp::BitwiseAnd, lin, !3)?;
    // The lane operand: the selected constant, or the operand's low two bits.
    let operand = |fb: &mut FunctionBuilder<'_>| -> BResult<SValue> {
        match g.konst {
            Some(i) => k(fb, i),
            None => bin_k(fb, BinOp::BitwiseAnd, nth(a, 1)?, 3),
        }
    };
    match g.op {
        G::ShuffleXor if g.konst == Some(0) => v(),
        G::Broadcast | G::Shuffle => {
            let i = operand(fb)?;
            let src = fb.binary(BinOp::BitwiseOr, qbase, i)?;
            shuffled(fb, ln, e, v()?, src)
        }
        G::ShuffleXor => {
            let mask = operand(fb)?;
            let src = fb.binary(BinOp::BitwiseXor, lin, mask)?;
            shuffled(fb, ln, e, v()?, src)
        }
        // At or below the calling lane: never past the thread count.
        G::ShuffleUp => {
            let d = nth(a, 1)?;
            let ok = fb.binary(BinOp::ULessThanEqual, d, q)?;
            let from = fb.binary(BinOp::ISub, lin, d)?;
            let src = fb.select(ok, from, lin)?;
            one(fb, ln, v()?, src)
        }
        G::ShuffleDown => {
            let d = nth(a, 1)?;
            let t = fb.binary(BinOp::IAdd, q, d)?;
            let four = k(fb, 4)?;
            let ok = fb.binary(BinOp::ULessThan, t, four)?;
            let from = fb.binary(BinOp::IAdd, lin, d)?;
            let src = fb.select(ok, from, lin)?;
            shuffled(fb, ln, e, v()?, src)
        }
        G::ShuffleRotateDown => {
            let t = fb.binary(BinOp::IAdd, q, nth(a, 1)?)?;
            let t = bin_k(fb, BinOp::BitwiseAnd, t, 3)?;
            let src = fb.binary(BinOp::BitwiseOr, qbase, t)?;
            shuffled(fb, ln, e, v()?, src)
        }
        G::ShuffleRotateUp => {
            let t = fb.binary(BinOp::ISub, q, nth(a, 1)?)?;
            let t = bin_k(fb, BinOp::BitwiseAnd, t, 3)?;
            let src = fb.binary(BinOp::BitwiseOr, qbase, t)?;
            shuffled(fb, ln, e, v()?, src)
        }
        G::ShuffleAndFillUp | G::ShuffleAndFillDown => shift_and_fill(fb, ln, e, g, a),
        // In a divergent block the quad's first active lane by the published flags (none: the
        // calling lane, which is then not live, reads its own); else the quad's first lane, which
        // is below the thread count, as the calling lane is.
        G::BroadcastFirst => match e.active {
            Some(_) => {
                let ex = ln.exchange(fb, &[v()?])?;
                let act = active_bits(fb, ln, e, &ex, qbase, 4)?;
                let first = fb.glsl(GlslOp::FindILsb, &[act])?;
                let at = fb.binary(BinOp::IAdd, qbase, first)?;
                let zero = k(fb, 0)?;
                let some = fb.binary(BinOp::INotEqual, act, zero)?;
                let src = fb.select(some, at, lin)?;
                let x = ln.read(fb, &ex, 0, src)?;
                ln.close(fb, ex)?;
                Ok(x)
            }
            None => one(fb, ln, v()?, qbase),
        },
        G::PrefixInclusiveSum
        | G::PrefixExclusiveSum
        | G::PrefixInclusiveProduct
        | G::PrefixExclusiveProduct => quad_scan(fb, ln, e, g, v()?, qbase, q),
        G::Sum | G::Product | G::Min | G::Max | G::And | G::Or | G::Xor => {
            quad_fold(fb, ln, e, g, v()?, qbase)
        }
        // The quad's lanes OR their bits into the word of its first lane: only they reach that
        // word, at their Metal lanes' bits, so the quad's bits are its lanes' and a vote reads one
        // word instead of each lane.
        G::All | G::Any => {
            let all = g.op == G::All;
            let (w, act) = word(fb, ln, e, qbase, v()?, all)?;
            if all {
                let act = match act {
                    Some(a) => a,
                    None => {
                        let present = present(fb, e, qbase, 4)?;
                        let at = bin_k(fb, BinOp::BitwiseAnd, lin, SIMD_WIDTH - 4)?;
                        fb.binary(BinOp::ShiftLeftLogical, present, at)?
                    }
                };
                fb.binary(BinOp::IEqual, w, act)
            } else {
                let zero = k(fb, 0)?;
                fb.binary(BinOp::INotEqual, w, zero)
            }
        }
        // In a divergent block the quad's active lanes are published.
        G::IsFirst | G::ActiveThreadsMask if e.active.is_some() => {
            let ex = ln.exchange(fb, &[])?;
            let act = active_bits(fb, ln, e, &ex, qbase, 4)?;
            ln.close(fb, ex)?;
            if g.op == G::ActiveThreadsMask {
                return Ok(act);
            }
            let first = fb.glsl(GlslOp::FindILsb, &[act])?;
            fb.binary(BinOp::IEqual, first, q)
        }
        G::IsFirst => fb.binary(BinOp::IEqual, lin, qbase),
        G::ActiveThreadsMask => present(fb, e, qbase, 4),
        // The quad's four bits of the simdgroup's ballot word (inactive and absent lanes give
        // 0): bit `j` is quad lane `j`.
        G::Ballot => {
            let zero = k(fb, 0)?;
            let base = ln.absolute(fb, zero)?;
            let (w, _) = word(fb, ln, e, base, v()?, false)?;
            let at = bin_k(fb, BinOp::BitwiseAnd, lin, 28)?;
            let bits = fb.binary(BinOp::ShiftRightLogical, w, at)?;
            bin_k(fb, BinOp::BitwiseAnd, bits, 15)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::arith::legal::{FloatFmt, Held};

    /// A lane publishes one word per lane of each value it moves: the predicate of a vote or
    /// ballot, nothing for `is_first` and the quad's active mask, the data and the fill of a
    /// shuffle-and-fill; the multiply-adds publish four cells.
    #[test]
    fn exchanged_words_follow_the_moved_values() {
        let plan = |op, held, lanes| GroupPlan {
            scope: GroupScope::Simdgroup,
            op,
            form: Form { held, lanes },
            arith: None,
            konst: None,
            quad_instructions: true,
        };
        let half4 = Held::Float(FloatFmt::Half);
        let u32_ = Held::Int {
            bits: 32,
            container: 32,
        };
        assert_eq!(group_words(&plan(GroupOp::Shuffle, half4, 4)), 4);
        assert_eq!(group_words(&plan(GroupOp::ShuffleAndFillDown, u32_, 2)), 4);
        assert_eq!(group_words(&plan(GroupOp::All, Held::Bool, 1)), 1);
        let ballot = Held::Int {
            bits: 64,
            container: 64,
        };
        assert_eq!(group_words(&plan(GroupOp::Ballot, ballot, 1)), 1);
        assert_eq!(group_words(&plan(GroupOp::IsFirst, Held::Bool, 1)), 0);
        assert_eq!(bytes_per_simdgroup(TILE_WORDS), 512);
    }
}
