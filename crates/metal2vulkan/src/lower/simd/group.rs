//! Emission of the simd strategies per tier, their operation classes, and their word bounds. See
//! the module documentation of `lower::simd`.

use crate::air::intrinsics::{GroupOp, GroupScope};
use crate::air::ir::{InstId, InstKind, Value};
use crate::device::subgroup::{OpBits, Tier};
use crate::lower::arith::legal::{FloatFmt, Form, Held};
use crate::lower::arith::{narrow, ops};
use crate::lower::emit::EmitCx;
use crate::spirv::builder::ops::group::GroupMode;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::{BResult, BuildError, Dangling};
use crate::spirv::ids::{GlobalId, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, GlslOp, GroupArith, MemAccess, QuadDirection, ShuffleOp,
    Type, UnOp,
};

use super::workgroup::{Emu, Published};
use super::{GroupPlan, SimdStrategy};

/// The lanes of a Metal simdgroup.
pub const SIMD_WIDTH: u32 = 32;

fn dangling() -> BuildError {
    BuildError::Dangling(Dangling::Value(SValue::new(u32::MAX)))
}

/// Whether `op` is a reduction, prefix sum or prefix product.
fn combines(op: GroupOp) -> bool {
    matches!(
        op,
        GroupOp::Sum
            | GroupOp::Product
            | GroupOp::Min
            | GroupOp::Max
            | GroupOp::And
            | GroupOp::Or
            | GroupOp::Xor
    ) || scan(op)
}

/// Whether `op` is a prefix sum or prefix product.
fn scan(op: GroupOp) -> bool {
    matches!(
        op,
        GroupOp::PrefixInclusiveSum
            | GroupOp::PrefixExclusiveSum
            | GroupOp::PrefixInclusiveProduct
            | GroupOp::PrefixExclusiveProduct
    )
}

/// Whether `op` is an exclusive prefix sum or product.
pub(super) fn exclusive(op: GroupOp) -> bool {
    matches!(
        op,
        GroupOp::PrefixExclusiveSum | GroupOp::PrefixExclusiveProduct
    )
}

/// Whether `g` is an operation this module emits with the form it was selected for: a `bool`
/// vote or `is_first`, a simdgroup's 64-bit ballot and active mask, a quad's 32-bit-container
/// (`ushort`) ballot and active mask, and a combining operation for every reduction and prefix
/// sum. Every operation exists in both scopes (the quad's members as Apple's compiler emits them).
pub(super) fn check(g: &GroupPlan) -> Result<(), ()> {
    use GroupOp as G;
    let f = g.form;
    let form = match (g.scope, g.op) {
        (_, G::All | G::Any | G::IsFirst) => f.is_bool() && f.lanes == 1,
        (GroupScope::Simdgroup, G::Ballot | G::ActiveThreadsMask) => {
            (f.is_int() || f.is_wide()) && f.container() == 64 && f.lanes == 1
        }
        (GroupScope::Quad, G::Ballot | G::ActiveThreadsMask) => {
            f.is_int() && f.container() == 32 && f.lanes == 1
        }
        (_, op) if combines(op) => g.arith.is_some(),
        _ => true,
    };
    if form {
        Ok(())
    } else {
        Err(())
    }
}

/// The operation classes `g` needs in the `Direct` and `PartialFragment` tiers (the simdgroup is
/// the subgroup).
pub fn direct_ops(g: &GroupPlan) -> OpBits {
    use GroupOp as G;
    match (g.scope, g.op) {
        (GroupScope::Quad, _) => quad_ops(g),
        (_, G::BroadcastFirst | G::Ballot | G::ActiveThreadsMask) => OpBits::BALLOT,
        (_, G::All | G::Any) => OpBits::VOTE,
        (_, G::IsFirst) => OpBits::BASIC,
        (_, op) if combines(op) => OpBits::ARITHMETIC,
        _ => OpBits::SHUFFLE,
    }
}

/// The operation classes `g` needs in the `Cluster32` tier (the simdgroup is a 32-lane cluster of
/// the subgroup).
pub fn cluster_ops(g: &GroupPlan) -> OpBits {
    use GroupOp as G;
    match (g.scope, g.op) {
        (GroupScope::Quad, _) => quad_ops(g),
        (_, G::BroadcastFirst) => OpBits::BALLOT.union(OpBits::SHUFFLE),
        (_, G::All | G::Any | G::Ballot | G::IsFirst | G::ActiveThreadsMask) => OpBits::BALLOT,
        (_, op) if scan(op) => OpBits::BALLOT.union(OpBits::SHUFFLE),
        (_, op) if combines(op) => OpBits::CLUSTERED,
        _ => OpBits::SHUFFLE,
    }
}

/// Quads are the same in every tier: aligned groups of four subgroup lanes. A reduction, a
/// prefix sum or product and the first active lane's broadcast read the quad's lanes by shuffles
/// under the ballot of its active lanes, in Metal's order, and a vote, `is_first`, the ballot and
/// the active mask are the quad's bits of the ballot; a fragment's are quad instructions alone
/// ([`quad_votes`], [`quad_lanes`]). A broadcast or shuffle-xor by a constant lane or mask is
/// Vulkan's quad class where the stage takes it, and a shuffle elsewhere.
fn quad_ops(g: &GroupPlan) -> OpBits {
    use GroupOp as G;
    match (g.op, g.konst) {
        (G::ShuffleXor, Some(0)) => OpBits::NONE,
        (G::Broadcast | G::ShuffleXor, Some(_)) if g.quad_instructions => OpBits::QUAD,
        _ if g.helpers && !reads_lanes(g) => OpBits::QUAD,
        (G::BroadcastFirst, _) => OpBits::BALLOT.union(OpBits::SHUFFLE),
        (op, _) if combines(op) => OpBits::BALLOT.union(OpBits::SHUFFLE),
        (G::All | G::Any | G::Ballot | G::IsFirst | G::ActiveThreadsMask, _) => OpBits::BALLOT,
        _ => OpBits::SHUFFLE,
    }
}

/// Whether `g` reads another lane's value by that lane's index: a shuffle, shuffle-xor, relative
/// shuffle, rotation, broadcast by lane or fill, in either scope (not a quad shuffle-xor by the
/// constant 0, which is the value itself). Such a read names a lane that may not take part, so
/// where its group may be partly live it reads through the ballot of the live lanes
/// ([`Lane::read_from`], `super::partly_live`). The other operations read only active lanes: the
/// first active lane's broadcast, the reductions, prefix sums, votes and ballots.
pub fn reads_lanes(g: &GroupPlan) -> bool {
    use GroupOp as G;
    match (g.op, g.scope, g.konst) {
        (G::ShuffleXor, GroupScope::Quad, Some(0)) => false,
        (
            G::Broadcast
            | G::Shuffle
            | G::ShuffleXor
            | G::ShuffleUp
            | G::ShuffleDown
            | G::ShuffleRotateDown
            | G::ShuffleRotateUp
            | G::ShuffleAndFillUp
            | G::ShuffleAndFillDown,
            _,
            _,
        ) => true,
        _ => false,
    }
}

/// The census name of a group strategy. A float sum or product and a float prefix sum or product,
/// whose association order is the device's (`PrecisionNote::ReductionOrder`), name their float
/// format, so the census states the reductions a comparison class tolerates.
pub(super) fn name(g: &GroupPlan) -> &'static str {
    use GroupOp as G;
    let fadd = match g.arith {
        Some(GroupArith::FAdd) => g.form.float_fmt(),
        _ => None,
    };
    let fmul = match g.arith {
        Some(GroupArith::FMul) => g.form.float_fmt(),
        _ => None,
    };
    match (g.scope, g.op) {
        (GroupScope::Simdgroup, G::Broadcast | G::Shuffle | G::ShuffleXor) => "simd.shuffle",
        (
            GroupScope::Simdgroup,
            G::ShuffleUp | G::ShuffleDown | G::ShuffleRotateDown | G::ShuffleRotateUp,
        ) => "simd.shuffle_relative",
        (GroupScope::Simdgroup, G::ShuffleAndFillUp | G::ShuffleAndFillDown) => "simd.shuffle_fill",
        (GroupScope::Simdgroup, G::BroadcastFirst) => "simd.broadcast_first",
        (GroupScope::Simdgroup, G::PrefixInclusiveSum | G::PrefixExclusiveSum) => match fadd {
            Some(FloatFmt::Half) => "simd.scan.fadd.half",
            Some(FloatFmt::BFloat) => "simd.scan.fadd.bfloat",
            Some(FloatFmt::F32) => "simd.scan.fadd.float",
            Some(FloatFmt::F64) => "simd.scan.fadd.double",
            None => "simd.scan",
        },
        (GroupScope::Simdgroup, G::PrefixInclusiveProduct | G::PrefixExclusiveProduct) => {
            match fmul {
                Some(FloatFmt::Half) => "simd.scan.fmul.half",
                Some(FloatFmt::BFloat) => "simd.scan.fmul.bfloat",
                Some(FloatFmt::F32) => "simd.scan.fmul.float",
                Some(FloatFmt::F64) => "simd.scan.fmul.double",
                None => "simd.scan",
            }
        }
        (GroupScope::Simdgroup, G::Product) => match fmul {
            Some(FloatFmt::Half) => "simd.reduce.fmul.half",
            Some(FloatFmt::BFloat) => "simd.reduce.fmul.bfloat",
            Some(FloatFmt::F32) => "simd.reduce.fmul.float",
            Some(FloatFmt::F64) => "simd.reduce.fmul.double",
            None => "simd.reduce",
        },
        (GroupScope::Simdgroup, G::All | G::Any) => "simd.vote",
        (GroupScope::Simdgroup, G::Ballot | G::ActiveThreadsMask) => "simd.ballot",
        (GroupScope::Simdgroup, G::IsFirst) => "simd.is_first",
        (GroupScope::Simdgroup, _) => match fadd {
            Some(FloatFmt::Half) => "simd.reduce.fadd.half",
            Some(FloatFmt::BFloat) => "simd.reduce.fadd.bfloat",
            Some(FloatFmt::F32) => "simd.reduce.fadd.float",
            Some(FloatFmt::F64) => "simd.reduce.fadd.double",
            None => "simd.reduce",
        },
        (GroupScope::Quad, G::Broadcast | G::ShuffleXor) if g.konst.is_some() => "quad.fixed",
        // In Metal's measured order in every tier: no format to name.
        (GroupScope::Quad, op) if scan(op) => "quad.scan",
        (GroupScope::Quad, op) if combines(op) => "quad.reduce",
        (GroupScope::Quad, G::BroadcastFirst) => "quad.broadcast_first",
        (GroupScope::Quad, G::ShuffleAndFillUp | G::ShuffleAndFillDown) => "quad.shuffle_fill",
        (GroupScope::Quad, G::All | G::Any) => "quad.vote",
        (GroupScope::Quad, G::Ballot | G::IsFirst | G::ActiveThreadsMask) => "quad.ballot",
        (GroupScope::Quad, _) => "quad.shuffle",
    }
}

/// An upper bound on the words one use of `g` emits in any tier: instructions at 8 words each with
/// their share of new constants and types, the narrow roundings of a float sum, and the 32 steps of
/// the `Cluster32` prefix sum.
pub(super) fn words(g: &GroupPlan) -> u32 {
    use GroupOp as G;
    let round = narrow::round_words(g.form);
    let lanes = u32::from(g.form.lanes);
    // The lane load, the ballot types, and the splats of a vector form.
    let frame = 48 + 16 * lanes;
    frame
        + match g.op {
            op if scan(op) => 2048 + 32 * round,
            G::ShuffleAndFillUp | G::ShuffleAndFillDown => 160,
            // A quad's four reads with whether each lane takes part (by its ballot bits, or a
            // fragment's quad votes), three combinations and the result's rounding
            // ([`quad_tree`]).
            op if combines(op) && g.scope == GroupScope::Quad => 512 + 4 * round,
            op if combines(op) => 64 + round,
            // A fragment's quad's four votes and their bits, or four reads and their selects
            // ([`quad_votes`], [`quad_lanes`]).
            G::Ballot | G::IsFirst | G::ActiveThreadsMask | G::BroadcastFirst
                if g.helpers && g.scope == GroupScope::Quad =>
            {
                512
            }
            _ => 128,
        }
}

/// The lane of the calling invocation, in the tier's simdgroup layout.
pub(super) struct Lane {
    /// The calling lane's index among the lanes it exchanges values with:
    /// `SubgroupLocalInvocationId`, or in the emulated tier the linear local index.
    pub(super) l: SValue,
    /// The simdgroup is the aligned 32 lanes of `l` (`Cluster32`, and the emulated tier) rather
    /// than the whole subgroup.
    cluster: bool,
    /// The emulated tier's workgroup exchange.
    pub(super) emu: Option<Emu>,
    /// The `SubgroupSize` builtin on the `PartialFragment` tier when the function has a partly
    /// live lane read ([`Lane::live_lanes`]): there the subgroup may be narrower than the 32 Metal
    /// lanes a read names.
    size: Option<GlobalId>,
}

/// The lanes executing a partly live lane read ([`Lane::live_lanes`]): the ballot of `true`, and
/// on the `PartialFragment` tier the subgroup's size, below which a source lane must be to be
/// named by the ballot's bit extract or by a shuffle.
#[derive(Clone, Copy)]
pub(super) struct Live {
    ballot: SValue,
    size: Option<SValue>,
}

/// Values published for [`Lane::read`]: the values themselves on a subgroup, their scratch words
/// in the emulated tier.
pub(super) enum Exchange {
    Subgroup(Vec<SValue>),
    Workgroup(Published),
}

pub(super) fn k(fb: &mut FunctionBuilder<'_>, n: u32) -> BResult<SValue> {
    let c = fb.u32_const(n)?;
    fb.module().const_value(c)
}

pub(super) fn bin_k(fb: &mut FunctionBuilder<'_>, op: BinOp, a: SValue, n: u32) -> BResult<SValue> {
    let n = k(fb, n)?;
    fb.binary(op, a, n)
}

fn true_(fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
    let b = fb.module().ty(Type::Bool)?;
    let c = fb.module().constant(Constant::Scalar {
        ty: b,
        bits: ConstBits(1),
    })?;
    fb.module().const_value(c)
}

pub(super) fn shuffle(fb: &mut FunctionBuilder<'_>, v: SValue, src: SValue) -> BResult<SValue> {
    fb.group_shuffle(ShuffleOp::Shuffle, v, src)
}

impl Lane {
    /// The Metal lane: `L & 31` in a cluster, `L` otherwise.
    pub(super) fn metal(&self, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        if self.cluster {
            bin_k(fb, BinOp::BitwiseAnd, self.l, SIMD_WIDTH - 1)
        } else {
            Ok(self.l)
        }
    }

    /// The subgroup lane of Metal lane `m` (`m < 32`) of the calling lane's simdgroup.
    pub(super) fn absolute(&self, fb: &mut FunctionBuilder<'_>, m: SValue) -> BResult<SValue> {
        if self.cluster {
            let base = bin_k(fb, BinOp::BitwiseAnd, self.l, !(SIMD_WIDTH - 1))?;
            fb.binary(BinOp::BitwiseOr, base, m)
        } else {
            Ok(m)
        }
    }

    /// The 32-bit ballot word of `pred` that covers the calling lane's simdgroup (bit `i` is
    /// Metal lane `i`): word `L >> 5` of the subgroup ballot in a cluster, word 0 otherwise.
    fn word(&self, fb: &mut FunctionBuilder<'_>, pred: SValue) -> BResult<SValue> {
        let b = fb.group_ballot(pred)?;
        if self.cluster {
            let i = bin_k(fb, BinOp::ShiftRightLogical, self.l, 5)?;
            fb.vector_extract_dynamic(b, i)
        } else {
            fb.composite_extract(b, &[0])
        }
    }

    /// The active lanes of the calling lane's simdgroup.
    fn active(&self, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        let t = true_(fb)?;
        self.word(fb, t)
    }

    /// The calling lane's quad's four bits of `word`, a word of its simdgroup (bit `i` Metal lane
    /// `i`): bit `j` is quad lane `j`. Quads never straddle a word.
    fn quad_bits(&self, fb: &mut FunctionBuilder<'_>, word: SValue) -> BResult<SValue> {
        let at = bin_k(fb, BinOp::BitwiseAnd, self.l, 28)?;
        let bits = fb.binary(BinOp::ShiftRightLogical, word, at)?;
        bin_k(fb, BinOp::BitwiseAnd, bits, 15)
    }

    /// Publish `vals` for the calling lane's simdgroup to [`Lane::read`]: on a subgroup a
    /// shuffle reads them where they are; in the emulated tier they are stored to the scratch,
    /// followed by a workgroup barrier.
    pub(super) fn exchange(
        &self,
        fb: &mut FunctionBuilder<'_>,
        vals: &[SValue],
    ) -> BResult<Exchange> {
        match &self.emu {
            Some(e) => e.publish(fb, self.l, vals).map(Exchange::Workgroup),
            None => Ok(Exchange::Subgroup(vals.to_vec())),
        }
    }

    /// Value `i` of `ex` as lane `src` of the lane space published it.
    pub(super) fn read(
        &self,
        fb: &mut FunctionBuilder<'_>,
        ex: &Exchange,
        i: usize,
        src: SValue,
    ) -> BResult<SValue> {
        match (ex, &self.emu) {
            (Exchange::Subgroup(vals), None) => shuffle(fb, nth(vals, i)?, src),
            (Exchange::Workgroup(p), Some(e)) => e.read(fb, self.l, p, i, src),
            _ => Err(dangling()),
        }
    }

    /// The lanes of the subgroup that execute the open block: the ballot of `true`
    /// (`OpGroupNonUniformBallot`, a `uvec4` whose bit `i` is subgroup lane `i`), in which only
    /// the active invocations' bits are set, and on the `PartialFragment` tier the subgroup's
    /// size.
    pub(super) fn live_lanes(&self, fb: &mut FunctionBuilder<'_>) -> BResult<Live> {
        let t = true_(fb)?;
        let ballot = fb.group_ballot(t)?;
        let size = match self.size {
            Some(var) => {
                let p = fb.module().global_value(var)?;
                Some(fb.load(p, MemAccess::default())?)
            }
            None => None,
        };
        Ok(Live { ballot, size })
    }

    /// The lane a multiply-add (`matrix::tile_mad`) or a partly live group operation
    /// ([`Lane::read_from`]) reads for lane `src`. On a subgroup with `live` (its
    /// [`Lane::live_lanes`]) `src` takes part exactly when its ballot bit is set, and a `src` that
    /// takes no part is read through the calling lane, which does, so no shuffle names an
    /// invocation outside its tangle: SPIR-V makes such a shuffle's result poison, and a branch or
    /// an index that consumes poison is undefined behavior. On the `PartialFragment` tier a Metal
    /// lane at or above the subgroup's size takes no part, and the bit extract reads the calling
    /// lane's bit instead, so no index reaches the size (the bit extract and the shuffle are
    /// poison there too). What a lane reads in place of one that takes no part is unspecified:
    /// MSL leaves such a read undefined (section 6.10.2), and the value is not promised. Without
    /// `live` (a block that every lane of a whole subgroup executes), and in the emulated tier,
    /// whose exchange carries its own active flags (`workgroup::Emu::read`), `src` is read as it
    /// is.
    pub(super) fn cell_source(
        &self,
        fb: &mut FunctionBuilder<'_>,
        live: Option<Live>,
        src: SValue,
    ) -> BResult<SValue> {
        match live {
            Some(live) if self.emu.is_none() => {
                let on = match live.size {
                    Some(n) => {
                        let inside = fb.binary(BinOp::ULessThan, src, n)?;
                        let idx = fb.select(inside, src, self.l)?;
                        let bit = fb.group_ballot_bit_extract(live.ballot, idx)?;
                        fb.binary(BinOp::LogicalAnd, inside, bit)?
                    }
                    None => fb.group_ballot_bit_extract(live.ballot, src)?,
                };
                fb.select(on, src, self.l)
            }
            _ => Ok(src),
        }
    }

    /// End an exchange: in the emulated tier a workgroup barrier, so no lane publishes over the
    /// scratch before every lane has read it.
    pub(super) fn close(&self, fb: &mut FunctionBuilder<'_>, ex: Exchange) -> BResult<()> {
        match (ex, &self.emu) {
            (Exchange::Workgroup(_), Some(e)) => e.barrier(fb),
            _ => Ok(()),
        }
    }

    /// Read `v` from the lane `from` when `ok`, else from the calling lane (Metal keeps the
    /// lane's own value where the source is outside its simdgroup or quad), through `live` as
    /// [`Lane::read_from`] reads.
    fn read_or_own(
        &self,
        fb: &mut FunctionBuilder<'_>,
        live: Option<Live>,
        v: SValue,
        ok: SValue,
        from: SValue,
    ) -> BResult<SValue> {
        let src = fb.select(ok, from, self.l)?;
        self.read_from(fb, live, v, src)
    }

    /// Value `v` of lane `src`. With `live` (a partly live group operation's
    /// [`Lane::live_lanes`], `super::partly_live`), a lane that does not take part is read through
    /// the calling lane ([`Lane::cell_source`]), so no shuffle names an invocation outside its
    /// tangle; without it, the plain shuffle.
    fn read_from(
        &self,
        fb: &mut FunctionBuilder<'_>,
        live: Option<Live>,
        v: SValue,
        src: SValue,
    ) -> BResult<SValue> {
        let at = self.cell_source(fb, live, src)?;
        shuffle(fb, v, at)
    }
}

/// The operand values of call `at`.
pub(super) fn args(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    at: InstId,
) -> BResult<Vec<SValue>> {
    let Some(inst) = cx.body.insts.get(at) else {
        return Err(cx.fail("a simd strategy names a missing instruction"));
    };
    let InstKind::Call(call) = &inst.kind else {
        return Err(cx.fail("a simd strategy on a non-call"));
    };
    let vals: Vec<Value> = call.args.to_vec();
    let mut out = Vec::with_capacity(vals.len());
    for v in vals {
        let Some(id) = cx.plan.values.id(v) else {
            return Err(cx.fail("a simd operand out of the value range"));
        };
        let x = cx.operand(fb, id)?;
        // In a guarded block of a predicated kernel a lane that is not live never computed its
        // operands: it takes part with null ones, so every published word and every lane index is a
        // defined value.
        let x = match cx.live {
            Some(live) => {
                let ty = fb.value_ty(x)?;
                let null = fb.module().constant(Constant::Null { ty })?;
                let null = fb.module().const_value(null)?;
                fb.select(live, x, null)?
            }
            None => x,
        };
        out.push(x);
    }
    Ok(out)
}

pub(super) fn nth(a: &[SValue], i: usize) -> BResult<SValue> {
    a.get(i).copied().ok_or_else(dangling)
}

/// The calling invocation's lane in the plan's tier (`Direct`, `PartialFragment`, `Cluster32`,
/// or `Emulated`), loaded in the open block.
pub(super) fn lane(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<Lane> {
    let (cluster, emulated) = match cx.plan.simd.1 {
        Tier::Direct | Tier::PartialFragment => (false, false),
        Tier::Cluster32 => (true, false),
        Tier::Emulated => (true, true),
        _ => return Err(cx.fail("a simdgroup operation below the subgroup tiers")),
    };
    let Some(var) = cx.decl.families.simd.lane else {
        return Err(cx.fail("the simdgroup lane is not declared"));
    };
    let p = fb.module().global_value(var)?;
    let l = fb.load(p, MemAccess::default())?;
    let emu = if emulated {
        Some(super::workgroup::emu(cx, fb)?)
    } else {
        None
    };
    let size = cx.decl.families.simd.size;
    Ok(Lane {
        l,
        cluster,
        emu,
        size,
    })
}

/// Emit strategy `s` of instruction `at` and bind its result.
pub(super) fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    at: InstId,
    s: &SimdStrategy,
) -> BResult<()> {
    let Some(result) = cx.plan.values.inst(at) else {
        return Err(cx.fail("a simd instruction out of the value range"));
    };
    let x = match s {
        SimdStrategy::Width(form) => ops::splat(fb, *form, u64::from(SIMD_WIDTH))?,
        SimdStrategy::Assumed => true_(fb)?,
        SimdStrategy::VectorVote { all } => {
            let a = args(cx, fb, at)?;
            let v = nth(&a, 0)?;
            fb.unary(if *all { UnOp::All } else { UnOp::Any }, v)?
        }
        SimdStrategy::Group(g) => {
            let a = args(cx, fb, at)?;
            let lane = lane(cx, fb)?;
            // A lane read where the group may be partly live reads through its live lanes; the
            // emulated tier's exchange carries its own active flags.
            let live = if lane.emu.is_none() && super::partly_live_at(cx, at) {
                Some(lane.live_lanes(fb)?)
            } else {
                None
            };
            match (g.scope, &lane.emu) {
                (GroupScope::Simdgroup, Some(e)) => super::workgroup::simd_op(fb, &lane, e, g, &a)?,
                (GroupScope::Quad, Some(e)) => super::workgroup::quad_op(fb, &lane, e, g, &a)?,
                (GroupScope::Simdgroup, None) => simd_op(fb, &lane, live, g, &a)?,
                (GroupScope::Quad, None) => quad_op(fb, &lane, live, g, &a)?,
            }
        }
        SimdStrategy::Agx(_) | SimdStrategy::Matrix(_) => {
            return Err(cx.fail("an AGX or matrix strategy emitted as a group operation"))
        }
    };
    cx.define(result, x);
    Ok(())
}

/// A reduction's operand: a signed narrow integer sign-extended to its container.
pub(super) fn pre(fb: &mut FunctionBuilder<'_>, g: &GroupPlan, v: SValue) -> BResult<SValue> {
    match g.arith {
        Some(GroupArith::SMin | GroupArith::SMax) => ops::sext(fb, g.form, v),
        _ => Ok(v),
    }
}

/// A reduction's result in the held form: an integer masked to its width, a narrow float sum or
/// product rounded to its format.
pub(super) fn post(fb: &mut FunctionBuilder<'_>, g: &GroupPlan, r: SValue) -> BResult<SValue> {
    match g.form.held {
        Held::Int { .. } => ops::mask(fb, g.form, r),
        Held::Float(_) if matches!(g.arith, Some(GroupArith::FAdd | GroupArith::FMul)) => {
            narrow::round(fb, g.form, r)
        }
        _ => Ok(r),
    }
}

fn reduce(
    fb: &mut FunctionBuilder<'_>,
    g: &GroupPlan,
    v: SValue,
    mode: GroupMode,
) -> BResult<SValue> {
    let op = g.arith.ok_or_else(dangling)?;
    let x = pre(fb, g, v)?;
    let r = fb.group_arith(op, mode, x)?;
    post(fb, g, r)
}

/// A simdgroup operation; `live` is a partly live lane read's [`Lane::live_lanes`]
/// (`super::partly_live`).
fn simd_op(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    live: Option<Live>,
    g: &GroupPlan,
    a: &[SValue],
) -> BResult<SValue> {
    use GroupOp as G;
    // The value operand (absent for `is_first` and the active mask).
    let v = || nth(a, 0);
    let last = SIMD_WIDTH - 1;
    match g.op {
        G::Broadcast | G::Shuffle => {
            let i = bin_k(fb, BinOp::BitwiseAnd, nth(a, 1)?, last)?;
            let src = ln.absolute(fb, i)?;
            ln.read_from(fb, live, v()?, src)
        }
        // A mask below 32 keeps the exchange inside the 32-lane cluster; a partly live one reads
        // the lane it names through the live lanes.
        G::ShuffleXor => {
            let m = bin_k(fb, BinOp::BitwiseAnd, nth(a, 1)?, last)?;
            match live {
                Some(_) => {
                    let src = fb.binary(BinOp::BitwiseXor, ln.l, m)?;
                    ln.read_from(fb, live, v()?, src)
                }
                None => fb.group_shuffle(ShuffleOp::Xor, v()?, m),
            }
        }
        G::ShuffleUp => {
            let d = nth(a, 1)?;
            let m = ln.metal(fb)?;
            let ok = fb.binary(BinOp::ULessThanEqual, d, m)?;
            let from = fb.binary(BinOp::ISub, ln.l, d)?;
            ln.read_or_own(fb, live, v()?, ok, from)
        }
        G::ShuffleDown => {
            let d = nth(a, 1)?;
            let m = ln.metal(fb)?;
            let t = fb.binary(BinOp::IAdd, m, d)?;
            let w = k(fb, SIMD_WIDTH)?;
            let ok = fb.binary(BinOp::ULessThan, t, w)?;
            let from = fb.binary(BinOp::IAdd, ln.l, d)?;
            ln.read_or_own(fb, live, v()?, ok, from)
        }
        G::ShuffleRotateDown => {
            let m = ln.metal(fb)?;
            let t = fb.binary(BinOp::IAdd, m, nth(a, 1)?)?;
            let t = bin_k(fb, BinOp::BitwiseAnd, t, last)?;
            let src = ln.absolute(fb, t)?;
            ln.read_from(fb, live, v()?, src)
        }
        // Rotate up by `d` is rotate down by the width less `d`: the lane `(m - d) mod 32`, which
        // the wrapping subtraction's low five bits are.
        G::ShuffleRotateUp => {
            let m = ln.metal(fb)?;
            let t = fb.binary(BinOp::ISub, m, nth(a, 1)?)?;
            let t = bin_k(fb, BinOp::BitwiseAnd, t, last)?;
            let src = ln.absolute(fb, t)?;
            ln.read_from(fb, live, v()?, src)
        }
        G::ShuffleAndFillUp | G::ShuffleAndFillDown => fill(fb, ln, live, g, a),
        G::BroadcastFirst if ln.cluster => {
            let act = ln.active(fb)?;
            let first = fb.glsl(GlslOp::FindILsb, &[act])?;
            let src = ln.absolute(fb, first)?;
            shuffle(fb, v()?, src)
        }
        G::BroadcastFirst => fb.group_broadcast_first(v()?),
        G::PrefixInclusiveSum
        | G::PrefixExclusiveSum
        | G::PrefixInclusiveProduct
        | G::PrefixExclusiveProduct => {
            if ln.cluster {
                cluster_scan(fb, ln, g, v()?)
            } else if exclusive(g.op) {
                reduce(fb, g, v()?, GroupMode::ExclusiveScan)
            } else {
                reduce(fb, g, v()?, GroupMode::InclusiveScan)
            }
        }
        G::Sum | G::Product | G::Min | G::Max | G::And | G::Or | G::Xor => {
            let mode = if ln.cluster {
                GroupMode::Clustered(SIMD_WIDTH)
            } else {
                GroupMode::Reduce
            };
            reduce(fb, g, v()?, mode)
        }
        G::All | G::Any if ln.cluster => {
            let w = ln.word(fb, v()?)?;
            if g.op == G::All {
                let act = ln.active(fb)?;
                fb.binary(BinOp::IEqual, w, act)
            } else {
                let zero = k(fb, 0)?;
                fb.binary(BinOp::INotEqual, w, zero)
            }
        }
        G::All | G::Any => fb.group_vote(g.op == G::All, v()?),
        // Metal's `simd_vote`: bit `i` is Metal lane `i`, zero-extended to the 64-bit result.
        G::Ballot => {
            let w = ln.word(fb, v()?)?;
            crate::lower::arith::wide::zext_word(fb, g.form, w)
        }
        G::IsFirst if ln.cluster => {
            let act = ln.active(fb)?;
            let first = fb.glsl(GlslOp::FindILsb, &[act])?;
            let m = ln.metal(fb)?;
            fb.binary(BinOp::IEqual, first, m)
        }
        G::IsFirst => fb.group_elect(),
        // Metal's `simd_active_threads_mask`: the ballot of `true`, bit `i` Metal lane `i`.
        G::ActiveThreadsMask => {
            let w = ln.active(fb)?;
            crate::lower::arith::wide::zext_word(fb, g.form, w)
        }
    }
}

/// `{simd,quad}_shuffle_and_fill_{up,down}(data, fill, delta, modulo)`: within each segment of
/// the simdgroup ([`fill_source`] gives its width from the modulo), lane `l` of the segment reads
/// `data[l ∓ delta]` when that is inside the segment, else `fill` at the position wrapped around
/// the segment (`l - delta + width` up, `l + delta - width` down). Up compares against the
/// unreduced `delta`, so a `delta` of the whole segment reads only `fill`; the reduced `delta`
/// only keeps the wrap arithmetic from underflowing (old-engine authored cases
/// `shift-a-whole-simd-cluster-out-through-the-fill`, `filter_error_map_SIMD`).
///
/// A partly live fill (`live`) reads both values of a source that does not take part through the
/// calling lane ([`Lane::cell_source`]): MSL leaves the values undefined, and they are not
/// promised.
fn fill(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    live: Option<Live>,
    g: &GroupPlan,
    a: &[SValue],
) -> BResult<SValue> {
    let (data, fill, d, modulo) = (nth(a, 0)?, nth(a, 1)?, nth(a, 2)?, nth(a, 3)?);
    let up = g.op == GroupOp::ShuffleAndFillUp;
    let (src, inside) = fill_source(fb, ln, g, up, d, modulo)?;
    let at = ln.cell_source(fb, live, src)?;
    let x = shuffle(fb, data, at)?;
    let y = shuffle(fb, fill, at)?;
    fb.select(inside, x, y)
}

/// The segment width Metal gives a shuffle-and-fill whose modulo is the constant `m`, in either
/// scope (measured on the M3 Max and the M4): for `m` up to 32 the next power of two, at least 2;
/// past 32 the power of two whose exponent is that one's modulo 8, an exponent of 0, 6 or 7 giving
/// 2 (513 to 1024 fill in fours, 33 to 512 in pairs). The modulo is a `ushort`.
pub(super) fn constant_fill_width(m: u32) -> u32 {
    let below = (m.wrapping_sub(1) & 0xFFFF) as u16;
    let e = (u16::BITS - below.leading_zeros()) % 8;
    if (1..=5).contains(&e) {
        1 << e
    } else {
        2
    }
}

/// The source lane of the shuffle-and-fill `g` by `d` with `modulo`, and whether the lane reads
/// the data (not the fill) there ([`fill`]). Metal answers the modulo by whether the AIR gives it
/// as a constant (measured on the M3 Max and the M4 for every delta below 9):
/// - a constant (`g.konst`) fills the simdgroup in segments of [`constant_fill_width`] lanes, in
///   either scope, so a quad's fill by a constant 5 to 32 reads past the quad;
/// - a runtime modulo moves the lanes only when it is a power of two from 2 to the group's width
///   (2 and 4 for a quad, up to 32 for a simdgroup), and otherwise leaves every lane its own data.
///
/// Segments are aligned in the simdgroup, so a quad's of 2 and 4 are within the quad.
pub(super) fn fill_source(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    g: &GroupPlan,
    up: bool,
    d: SValue,
    modulo: SValue,
) -> BResult<(SValue, SValue)> {
    let (modulo, moves) = match g.konst {
        Some(m) => (k(fb, constant_fill_width(m))?, None),
        None => {
            let width = match g.scope {
                GroupScope::Simdgroup => SIMD_WIDTH,
                GroupScope::Quad => 4,
            };
            let zero = k(fb, 0)?;
            let one = k(fb, 1)?;
            let below = fb.binary(BinOp::ISub, modulo, one)?;
            let bits = fb.binary(BinOp::BitwiseAnd, modulo, below)?;
            let power = fb.binary(BinOp::IEqual, bits, zero)?;
            let low = bin_k(fb, BinOp::UGreaterThanEqual, modulo, 2)?;
            let high = bin_k(fb, BinOp::ULessThanEqual, modulo, width)?;
            let within = fb.binary(BinOp::LogicalAnd, low, high)?;
            let moves = fb.binary(BinOp::LogicalAnd, power, within)?;
            // A modulo that moves nothing divides as 1: its lanes keep their own data below.
            (fb.select(moves, modulo, one)?, Some(moves))
        }
    };
    let m = ln.metal(fb)?;
    let l = fb.binary(BinOp::UMod, m, modulo)?;
    let base = fb.binary(BinOp::ISub, ln.l, l)?;
    let (inside, wrapped) = if up {
        let dm = fb.binary(BinOp::UMod, d, modulo)?;
        let inside = fb.binary(BinOp::UGreaterThanEqual, l, d)?;
        let lifted = fb.binary(BinOp::IAdd, l, modulo)?;
        let lifted = fb.binary(BinOp::ISub, lifted, dm)?;
        (inside, fb.binary(BinOp::UMod, lifted, modulo)?)
    } else {
        let s = fb.binary(BinOp::IAdd, l, d)?;
        let inside = fb.binary(BinOp::ULessThan, s, modulo)?;
        (inside, fb.binary(BinOp::UMod, s, modulo)?)
    };
    let src = fb.binary(BinOp::IAdd, base, wrapped)?;
    let Some(moves) = moves else {
        return Ok((src, inside));
    };
    let src = fb.select(moves, src, ln.l)?;
    let own = true_(fb)?;
    let inside = fb.select(moves, inside, own)?;
    Ok((src, inside))
}

/// A fragment's quad holds helpers, which take part in Metal's quad operations (`full_quads`).
/// Vulkan runs its quad instructions over a quad's helpers, and its other group operations, a
/// ballot and a shuffle among them, over them only where the entry maximally reconverges (section
/// 9.31, "Helper Invocations"), and AMD's open Linux driver leaves the helpers out of a ballot even
/// there. So a fragment reads its quad through quad instructions alone: quad lane `j` takes part
/// when `OpGroupNonUniformQuadAnyKHR` of "the quad lane is `j`" (and of `pred`, where given)
/// holds.
fn quad_votes(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    pred: Option<SValue>,
) -> BResult<Vec<SValue>> {
    let q = bin_k(fb, BinOp::BitwiseAnd, ln.l, 3)?;
    (0..4)
        .map(|j| {
            let me = bin_k(fb, BinOp::IEqual, q, j)?;
            let me = match pred {
                Some(p) => fb.binary(BinOp::LogicalAnd, me, p)?,
                None => me,
            };
            fb.group_quad_vote(false, me)
        })
        .collect()
}

/// The quad's four bits of a fragment's quad `votes` ([`quad_votes`]): bit `j` is quad lane `j`'s.
fn quad_word(fb: &mut FunctionBuilder<'_>, votes: &[SValue]) -> BResult<SValue> {
    let zero = k(fb, 0)?;
    let mut word = zero;
    for (j, &on) in (0..).zip(votes) {
        let bit = k(fb, 1 << j)?;
        let bit = fb.select(on, bit, zero)?;
        word = fb.binary(BinOp::BitwiseOr, word, bit)?;
    }
    Ok(word)
}

/// A fragment's quad's four lanes of `x` ([`quad_votes`]): for each quad lane `j`, its value,
/// `OpGroupNonUniformQuadBroadcast` from the constant lane `j`, and whether it takes part. A lane
/// that takes no part is read as poison, which no caller's select takes (SPIR-V's `OpSelect`
/// keeps an operand it does not select from its result).
fn quad_lanes(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    x: SValue,
) -> BResult<Vec<(SValue, SValue)>> {
    let q = bin_k(fb, BinOp::BitwiseAnd, ln.l, 3)?;
    (0..4)
        .map(|j| {
            let me = bin_k(fb, BinOp::IEqual, q, j)?;
            let on = fb.group_quad_vote(false, me)?;
            let i = k(fb, j)?;
            Ok((fb.group_quad_broadcast(x, i)?, on))
        })
        .collect()
}

/// A quad reduction in every subgroup tier: the quad's four lanes, each with whether it takes
/// part, combined in Metal's order ([`quad_tree`]). A fragment's lanes are its [`quad_lanes`]:
/// every lane folds the same four values in the same order, bit for bit. Elsewhere lane `j` takes
/// part when its bit of the ballot of the active lanes is set, and its value is the shuffle from
/// it, or from the calling lane when it takes no part.
fn quad_reduce(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    g: &GroupPlan,
    v: SValue,
) -> BResult<SValue> {
    let x = pre(fb, g, v)?;
    let lanes = if g.helpers {
        quad_lanes(fb, ln, x)?
    } else {
        let word = ln.active(fb)?;
        let act = ln.quad_bits(fb, word)?;
        let base = bin_k(fb, BinOp::BitwiseAnd, ln.l, !3)?;
        let zero = k(fb, 0)?;
        let mut lanes = Vec::with_capacity(4);
        for j in 0..4 {
            let bit = bin_k(fb, BinOp::ShiftRightLogical, act, j)?;
            let bit = bin_k(fb, BinOp::BitwiseAnd, bit, 1)?;
            let on = fb.binary(BinOp::INotEqual, bit, zero)?;
            let from = bin_k(fb, BinOp::BitwiseOr, base, j)?;
            lanes.push((ln.read_or_own(fb, None, x, on, from)?, on));
        }
        lanes
    };
    let r = quad_tree(fb, g, &lanes)?;
    post(fb, g, r)
}

/// Metal's quad reduction over the quad's four lanes, each a value and whether it takes part
/// (measured): `(x0 op x1) op (x2 op x3)`, a lane that takes no part counting as the operation's
/// identity, so a pair of one such lane is its other value. A float sum or product is therefore
/// Metal's, bit for bit.
pub(super) fn quad_tree(
    fb: &mut FunctionBuilder<'_>,
    g: &GroupPlan,
    lanes: &[(SValue, SValue)],
) -> BResult<SValue> {
    let pair = |fb: &mut FunctionBuilder<'_>,
                (a, on_a): (SValue, SValue),
                (b, on_b): (SValue, SValue)|
     -> BResult<(SValue, SValue)> {
        let c = super::workgroup::combine(fb, g, a, b)?;
        let with_b = fb.select(on_a, c, b)?;
        let r = fb.select(on_b, with_b, a)?;
        Ok((r, fb.binary(BinOp::LogicalOr, on_a, on_b)?))
    };
    let (x0, x1, x2, x3) = (
        nth_lane(lanes, 0)?,
        nth_lane(lanes, 1)?,
        nth_lane(lanes, 2)?,
        nth_lane(lanes, 3)?,
    );
    let low = pair(fb, x0, x1)?;
    let high = pair(fb, x2, x3)?;
    Ok(pair(fb, low, high)?.0)
}

fn nth_lane(lanes: &[(SValue, SValue)], i: usize) -> BResult<(SValue, SValue)> {
    lanes.get(i).copied().ok_or_else(dangling)
}

/// The lanes a [`lane_scan`] reads: the word of the group's active lanes (bit `j` is the group's
/// lane `j`), the calling lane's index in the group, how it reads them, and the group's width.
struct Span {
    act: SValue,
    m: SValue,
    reads: LaneReads,
    width: u32,
}

/// How a [`lane_scan`] reads the group's lane `j`: by a shuffle from the group's first lane
/// `base` plus `j`, or the calling lane where `j` takes no part; or, in a fragment's quad, as its
/// [`quad_lanes`] value.
enum LaneReads {
    Shuffle { base: SValue },
    Quad(Vec<SValue>),
}

/// A `Cluster32` prefix sum or product: [`lane_scan`] over the 32-lane cluster. An exclusive float
/// sum with no contribution is `+0`, the identity of Vulkan's scans.
fn cluster_scan(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    g: &GroupPlan,
    v: SValue,
) -> BResult<SValue> {
    let act = ln.active(fb)?;
    let m = ln.metal(fb)?;
    let zero = k(fb, 0)?;
    let base = ln.absolute(fb, zero)?;
    let span = Span {
        act,
        m,
        reads: LaneReads::Shuffle { base },
        width: SIMD_WIDTH,
    };
    lane_scan(fb, ln, g, v, span, true)
}

/// A quad prefix sum or product in every subgroup tier: [`lane_scan`] over the quad, Metal's order
/// (measured, in lane order, the exclusive sum of the first active lane `-0` and its product 1, a
/// `half` rounded at every step). A fragment's quad is its [`quad_lanes`], their votes the active
/// lanes; elsewhere the active lanes are the quad's bits of the ballot.
fn quad_scan(fb: &mut FunctionBuilder<'_>, ln: &Lane, g: &GroupPlan, v: SValue) -> BResult<SValue> {
    let lanes = if g.helpers {
        Some(quad_lanes(fb, ln, v)?)
    } else {
        None
    };
    let act = match &lanes {
        Some(lanes) => {
            let votes: Vec<SValue> = lanes.iter().map(|&(_, on)| on).collect();
            quad_word(fb, &votes)?
        }
        None => {
            let word = ln.active(fb)?;
            ln.quad_bits(fb, word)?
        }
    };
    let m = bin_k(fb, BinOp::BitwiseAnd, ln.l, 3)?;
    let reads = match lanes {
        Some(lanes) => LaneReads::Quad(lanes.into_iter().map(|(x, _)| x).collect()),
        None => LaneReads::Shuffle {
            base: bin_k(fb, BinOp::BitwiseAnd, ln.l, !3)?,
        },
    };
    let span = Span {
        act,
        m,
        reads,
        width: 4,
    };
    lane_scan(fb, ln, g, v, span, false)
}

/// A prefix sum or product over a group of lanes read one by one: the sum or product, in
/// increasing lane order, of the values of the group's active lanes below the calling lane (and
/// the calling lane itself when inclusive). Each step reads one lane when it is included and the
/// calling lane otherwise, so no step reads an inactive lane (a fragment's quad step takes its
/// lane's [`quad_lanes`] value, which the step's select keeps only where the lane is included),
/// and inactive lanes contribute nothing whatever the divergence. The empty answer is the
/// operation's identity: `-0` for a float sum, or `+0` where `plus_zero` (the cluster's), and 1
/// for a product, Metal's answer for the first active lane (measured).
fn lane_scan(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    g: &GroupPlan,
    v: SValue,
    span: Span,
    plus_zero: bool,
) -> BResult<SValue> {
    let inclusive = !exclusive(g.op);
    let float = g.form.is_float();
    let product = matches!(g.arith, Some(GroupArith::IMul | GroupArith::FMul));
    let add = match (product, float) {
        (false, false) => BinOp::IAdd,
        (false, true) => BinOp::FAdd,
        (true, false) => BinOp::IMul,
        (true, true) => BinOp::FMul,
    };
    let one = k(fb, 1)?;
    let own = fb.binary(BinOp::ShiftLeftLogical, one, span.m)?;
    let below = fb.binary(BinOp::ISub, own, one)?;
    let lanes = if inclusive {
        fb.binary(BinOp::BitwiseOr, below, own)?
    } else {
        below
    };
    let include = fb.binary(BinOp::BitwiseAnd, lanes, span.act)?;
    let zero = k(fb, 0)?;
    // `-0` is the identity of a float sum (`-0 + x == x` for every `x`, `+0` included), and 1 of a
    // product.
    let mut acc = match (product, float) {
        (false, true) => ops::fsplat(fb, g.form, -0.0)?,
        (false, false) => ops::splat(fb, g.form, 0)?,
        (true, true) => ops::fsplat(fb, g.form, 1.0)?,
        (true, false) => ops::splat(fb, g.form, 1)?,
    };
    for j in 0..span.width {
        let bit = bin_k(fb, BinOp::ShiftRightLogical, include, j)?;
        let bit = bin_k(fb, BinOp::BitwiseAnd, bit, 1)?;
        let on = fb.binary(BinOp::INotEqual, bit, zero)?;
        let x = match &span.reads {
            LaneReads::Shuffle { base } => {
                let from = bin_k(fb, BinOp::BitwiseOr, *base, j)?;
                ln.read_or_own(fb, None, v, on, from)?
            }
            LaneReads::Quad(values) => nth(values, j as usize)?,
        };
        let sum = fb.binary(add, acc, x)?;
        let sum = if float {
            narrow::round(fb, g.form, sum)?
        } else {
            sum
        };
        acc = fb.select(on, sum, acc)?;
    }
    if float && !inclusive && !product {
        let some = fb.binary(BinOp::INotEqual, include, zero)?;
        acc = if plus_zero {
            let pz = ops::fsplat(fb, g.form, 0.0)?;
            fb.select(some, acc, pz)?
        } else {
            minus_zero_unless(fb, g.form, some, acc)?
        };
    }
    post(fb, g, acc)
}

/// `acc` where `some` holds, else the float form `f`'s `-0`, the empty exclusive quad sum, selected
/// through the value's bits: a device that compiles the module with fast math may fold a `-0`
/// constant to `+0` (MoltenVK's did on the M4), and an integer select keeps the bits.
pub(super) fn minus_zero_unless(
    fb: &mut FunctionBuilder<'_>,
    f: Form,
    some: SValue,
    acc: SValue,
) -> BResult<SValue> {
    let width = match f.held {
        Held::Float(FloatFmt::F64) => 64,
        _ => 32,
    };
    let int = Form {
        held: Held::Int {
            bits: width,
            container: width,
        },
        lanes: f.lanes,
    };
    let ity = ops::ty(fb, int)?;
    let fty = fb.value_ty(acc)?;
    let bits = fb.convert(ConvOp::Bitcast, acc, ity)?;
    let minus = ops::splat(fb, int, 1u64 << (width - 1))?;
    let bits = fb.select(some, bits, minus)?;
    fb.convert(ConvOp::Bitcast, bits, fty)
}

/// A quad operation; `live` is a partly live lane read's [`Lane::live_lanes`]
/// (`super::partly_live`), which reads its lane by a shuffle through the live lanes in place of a
/// quad instruction.
fn quad_op(
    fb: &mut FunctionBuilder<'_>,
    ln: &Lane,
    live: Option<Live>,
    g: &GroupPlan,
    a: &[SValue],
) -> BResult<SValue> {
    use GroupOp as G;
    // The value operand (absent for `is_first` and the active mask).
    let v = || nth(a, 0);
    let quad_lane = |fb: &mut FunctionBuilder<'_>| bin_k(fb, BinOp::BitwiseAnd, ln.l, 3);
    let quad_base = |fb: &mut FunctionBuilder<'_>| bin_k(fb, BinOp::BitwiseAnd, ln.l, !3);
    // The quad lane a broadcast reads or the mask a shuffle-xor applies: the constant one, or
    // the operand's low two bits.
    let operand = |fb: &mut FunctionBuilder<'_>| match g.konst {
        Some(c) => k(fb, c),
        None => bin_k(fb, BinOp::BitwiseAnd, nth(a, 1)?, 3),
    };
    let instructions = g.quad_instructions && live.is_none();
    match (g.op, g.konst) {
        (G::Broadcast, Some(i)) if instructions => {
            let i = k(fb, i)?;
            fb.group_quad_broadcast(v()?, i)
        }
        (G::ShuffleXor, Some(0)) => v(),
        (G::ShuffleXor, Some(x)) if instructions => {
            let dir = match x {
                1 => QuadDirection::Horizontal,
                2 => QuadDirection::Vertical,
                _ => QuadDirection::Diagonal,
            };
            fb.group_quad_swap(v()?, dir)
        }
        // A lane-varying lane or mask, a constant one where the stage takes no quad
        // instructions, or a partly live read: the shuffle from the lane the quad instruction
        // reads, quad lane `i` of the caller's quad or the caller's lane xor the mask (which stays
        // in its quad).
        (G::Broadcast | G::Shuffle, _) => {
            let i = operand(fb)?;
            let base = quad_base(fb)?;
            let src = fb.binary(BinOp::BitwiseOr, base, i)?;
            ln.read_from(fb, live, v()?, src)
        }
        (G::ShuffleXor, _) => {
            let m = operand(fb)?;
            match live {
                Some(_) => {
                    let src = fb.binary(BinOp::BitwiseXor, ln.l, m)?;
                    ln.read_from(fb, live, v()?, src)
                }
                None => fb.group_shuffle(ShuffleOp::Xor, v()?, m),
            }
        }
        (G::ShuffleUp, _) => {
            let d = nth(a, 1)?;
            let q = quad_lane(fb)?;
            let ok = fb.binary(BinOp::ULessThanEqual, d, q)?;
            let from = fb.binary(BinOp::ISub, ln.l, d)?;
            ln.read_or_own(fb, live, v()?, ok, from)
        }
        (G::ShuffleDown, _) => {
            let d = nth(a, 1)?;
            let q = quad_lane(fb)?;
            let t = fb.binary(BinOp::IAdd, q, d)?;
            let four = k(fb, 4)?;
            let ok = fb.binary(BinOp::ULessThan, t, four)?;
            let from = fb.binary(BinOp::IAdd, ln.l, d)?;
            ln.read_or_own(fb, live, v()?, ok, from)
        }
        (G::ShuffleRotateDown, _) => {
            let q = quad_lane(fb)?;
            let t = fb.binary(BinOp::IAdd, q, nth(a, 1)?)?;
            let t = bin_k(fb, BinOp::BitwiseAnd, t, 3)?;
            let base = quad_base(fb)?;
            let src = fb.binary(BinOp::BitwiseOr, base, t)?;
            ln.read_from(fb, live, v()?, src)
        }
        // Rotate up by `d` is the quad lane `(q - d) mod 4`, the wrapping subtraction's low bits.
        (G::ShuffleRotateUp, _) => {
            let q = quad_lane(fb)?;
            let t = fb.binary(BinOp::ISub, q, nth(a, 1)?)?;
            let t = bin_k(fb, BinOp::BitwiseAnd, t, 3)?;
            let base = quad_base(fb)?;
            let src = fb.binary(BinOp::BitwiseOr, base, t)?;
            ln.read_from(fb, live, v()?, src)
        }
        (G::ShuffleAndFillUp | G::ShuffleAndFillDown, _) => fill(fb, ln, live, g, a),
        // A fragment's quad's first lane that takes part ([`quad_lanes`]), the calling lane at the
        // latest.
        (G::BroadcastFirst, _) if g.helpers => {
            let lanes = quad_lanes(fb, ln, v()?)?;
            let mut first = v()?;
            for &(x, on) in lanes.iter().rev() {
                first = fb.select(on, x, first)?;
            }
            Ok(first)
        }
        // The quad's first active lane, by the ballot of `true` (the calling lane is one).
        (G::BroadcastFirst, _) => {
            let word = ln.active(fb)?;
            let act = ln.quad_bits(fb, word)?;
            let first = fb.glsl(GlslOp::FindILsb, &[act])?;
            let base = quad_base(fb)?;
            let src = fb.binary(BinOp::BitwiseOr, base, first)?;
            shuffle(fb, v()?, src)
        }
        (
            G::PrefixInclusiveSum
            | G::PrefixExclusiveSum
            | G::PrefixInclusiveProduct
            | G::PrefixExclusiveProduct,
            _,
        ) => quad_scan(fb, ln, g, v()?),
        (G::Sum | G::Product | G::Min | G::Max | G::And | G::Or | G::Xor, _) => {
            quad_reduce(fb, ln, g, v()?)
        }
        // Over the active lanes: every one's predicate holds, or one's does. A fragment's quad
        // holds helpers, which take part (`full_quads`): there the quad vote, which Vulkan runs
        // over them whatever the entry's reconvergence (section 9.31, "Helper Invocations").
        (G::All | G::Any, _) if g.helpers => fb.group_quad_vote(g.op == G::All, v()?),
        (G::All | G::Any, _) => {
            let word = ln.word(fb, v()?)?;
            let held = ln.quad_bits(fb, word)?;
            if g.op == G::Any {
                let zero = k(fb, 0)?;
                return fb.binary(BinOp::INotEqual, held, zero);
            }
            let word = ln.active(fb)?;
            let act = ln.quad_bits(fb, word)?;
            fb.binary(BinOp::IEqual, held, act)
        }
        (G::Ballot | G::IsFirst | G::ActiveThreadsMask, _) => {
            // The quad's four bits of the lanes that take part (and whose predicate holds, for a
            // ballot): a fragment's quad votes ([`quad_votes`]), elsewhere the quad's bits of the
            // simdgroup's ballot word (of the predicate for a ballot, of `true` otherwise;
            // inactive lanes give 0), as quads never straddle a word. Bit `j` is quad lane `j`,
            // as Metal's `quad_vote` numbers them.
            let pred = if g.op == G::Ballot { Some(v()?) } else { None };
            let bits = if g.helpers {
                let votes = quad_votes(fb, ln, pred)?;
                quad_word(fb, &votes)?
            } else {
                let word = match pred {
                    Some(p) => ln.word(fb, p)?,
                    None => ln.active(fb)?,
                };
                ln.quad_bits(fb, word)?
            };
            if g.op != G::IsFirst {
                return Ok(bits);
            }
            let first = fb.glsl(GlslOp::FindILsb, &[bits])?;
            let q = quad_lane(fb)?;
            fb.binary(BinOp::IEqual, first, q)
        }
    }
}
