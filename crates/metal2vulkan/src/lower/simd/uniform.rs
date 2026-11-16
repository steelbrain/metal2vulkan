//! Workgroup uniformity of a kernel's blocks: which blocks every invocation of a workgroup executes
//! together, the same number of times.
//!
//! The emulated simdgroup tier (`Tier::Emulated`) moves lanes through workgroup memory between
//! two workgroup barriers, which Vulkan defines only when every invocation of the workgroup
//! executes the same dynamic instance of the barrier. An emulated operation in a block this
//! analysis proves uniform is emitted as it is; a kernel with one in any other block is emitted
//! predicated (`super::divergent`), which reads the uniform blocks and the uniform branch
//! conditions and switch selectors ([`Uniformity::is_uniform_value`]) to keep real branches where
//! every invocation takes them together.
//!
//! **Values.** A value is *divergent* when invocations of one workgroup may hold different values
//! for it; everything else is uniform. The sources of divergence are:
//!
//! - the per-invocation dispatch builtins (positions and indices in the grid, threadgroup,
//!   simdgroup, and quadgroup), every stage-input role, and any parameter without a binding;
//!   buffers, textures, samplers, and the other resource roles are one object per dispatch, and
//!   the per-threadgroup builtins (threadgroup position and counts, sizes) are uniform;
//! - `undef` and `poison` constants (each use may read a different value), and a vector shuffle
//!   with a `poison` lane;
//! - a load, unless it reads `constant` memory through a uniform pointer (Metal's `constant`
//!   address space is not written while a dispatch runs, so every invocation reads the same
//!   value); device, threadgroup, and thread memory may be written by other invocations or hold
//!   per-invocation data;
//! - allocas, atomics, and calls, except the per-invocation arithmetic families (`air` math,
//!   conversions, packing, `bool` vector votes, the simdgroup width, and the `llvm` numeric
//!   intrinsics), whose result is divergent only when an operand is;
//! - a phi with a divergent incoming value, or entered from a divergent block.
//!
//! Every other instruction is divergent exactly when one of its operands is.
//!
//! **Lanes.** A vector of at most [`LANES`] lanes is tracked lane by lane: a lane-wise operation
//! (an arithmetic operation, comparison, cast, `select`, `freeze` or negation whose operands and
//! result have its lane count, a scalar operand standing for every lane) is divergent in a lane
//! exactly when an operand is there, an element extracted at a constant index is divergent when
//! its lane is, an insertion at a constant index replaces its lane's, and a shuffle with no
//! `poison` lane moves each lane's. A value is divergent when any of its lanes is.
//!
//! **Unit axes** ([`Uniformity::with_unit_axes`]). In a threadgroup of one thread along an axis,
//! every thread's `thread_position_in_threadgroup` there is 0 and its `thread_position_in_grid`
//! there is its threadgroup's `threadgroup_position_in_grid`, so among a workgroup's or a
//! simdgroup's invocations both are uniform on that axis: the analysis of a dispatch whose
//! threadgroups have such axes takes those lanes of the two builtins (the first, a scalar one's)
//! as uniform.
//!
//! **Blocks.** A block is divergent when some invocations of a workgroup may execute it while
//! others do not, or a different number of times: it is control dependent on a branch or switch
//! whose condition is divergent, or on one in a divergent block (control dependence is read off
//! the post-dominator tree: the blocks from a successor up to, not including, the branch's
//! immediate post-dominator). A reachable block from which no exit is reachable has no
//! post-dominator information and is divergent. Loops need no special case: a loop whose exit
//! condition is divergent makes its latch, header, and every header phi divergent, so a value
//! that depends on the iteration count is divergent where the loop is left.
//!
//! **Simdgroups** ([`Among::Simdgroup`]). The same analysis among the threads of one simdgroup: its
//! index in the threadgroup is uniform, and so is the result of a simdgroup reduction, vote,
//! ballot, active mask, or broadcast of the first active lane or of a uniform lane, which every
//! active thread of one execution receives alike, in a block every thread of the simdgroup runs
//! together. In a divergent block threads may hold the results of different executions (some
//! threads run it, or a loop's threads leave it after different iterations and each keeps its last
//! result), so there the result is divergent; it turns divergent when its block does.
//!
//! **The barrier contract** ([`Uniformity::under_barriers`]). Metal requires every thread of a
//! barrier's scope to execute it, the same number of times; a program that does not is undefined.
//! Taking that contract, a block holding a barrier whose execution scope covers the invocations
//! related (a threadgroup barrier; for a simdgroup, a simdgroup barrier too) is never divergent,
//! and neither is a block in no loop that dominates such a block in no loop and that the barrier
//! block post-dominates: each runs at most once per thread, and exactly when the other does.
//! Divergence still flows through such a block's own branch condition and phis.
//!
//! **Primitives** ([`Among::Primitive`]). The same analysis among the invocations of a vertex
//! function that write the vertices of one primitive of a draw: those vertices belong to one
//! instance, one patch and one amplification, so `[[instance_id]]`, `[[base_vertex]]`,
//! `[[base_instance]]`, `[[amplification_id]]`, `[[amplification_count]]` and `[[patch_id]]` are
//! uniform, and so is each vertex attribute its caller names (one fetched per instance or at a
//! constant step); `[[vertex_id]]`, `[[position_in_patch]]` and every other stage input are
//! divergent. A simdgroup's collective operations are divergent there, as for a workgroup: the
//! vertices of one primitive may run in different simdgroups.
//!
//! **Instances** ([`Uniformity::per_instance`]). The same values among the threads of a workgroup
//! that execute one dynamic instance of an instruction: Vulkan's dynamically uniform, which a
//! loop's divergent exit does not end inside the loop. A loop's threads that are still in it run
//! its iterations together, so a header phi is divergent only through a divergent incoming
//! value: a counter with a uniform start and a uniform step is uniform in every iteration, even
//! where threads leave the loop after different iterations. Block divergence reaches values
//! through two rules instead of the phi's divergent incoming block:
//!
//! - a phi at a **join** of a divergent branch is divergent: a block that paths from two of the
//!   branch's successors reach without passing the branch again (up to its immediate
//!   post-dominator, which may be one); a block that only one successor's paths reach is not, so
//!   a loop header that only the back edge's paths reach is not;
//! - a value defined in a loop that holds a divergent branch, which an instruction outside the
//!   loop uses, is divergent: threads may leave the loop after different iterations, each with
//!   its own last value. The value is divergent inside the loop too, which only loses precision.
//!
//! The analysis is one worklist over instructions and blocks, each of which turns divergent at
//! most once; its memory is charged to the meter.

use crate::air::abi::{Builtin, ParamRole};
use crate::air::intrinsics::{AirIntrinsic, GroupOp, GroupScope, LlvmIntrinsic, Sync, ThreadScope};
use crate::air::ir::{
    AddrSpace, BlockId, Body, Callee, Const, ConstId, Idx, Inst, InstId, InstKind, Type, TypeId,
    Value,
};
use crate::air::Entry;
use crate::budget::Meter;
use crate::error::Error;
use crate::graph::{LoopForest, PostDomTree};

/// The workgroup uniformity of a kernel's blocks and values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Uniformity {
    /// By block index: some invocations of a workgroup may execute the block without the others.
    divergent: Vec<bool>,
    /// By instruction index: invocations of one workgroup may hold different results.
    values: Vec<bool>,
    /// By parameter index: invocations of one workgroup may hold different arguments.
    args: Vec<bool>,
}

impl Uniformity {
    /// Whether every invocation of a workgroup executes block `b` the same number of times.
    pub fn is_uniform(&self, b: BlockId) -> bool {
        !self
            .divergent
            .get(b.raw() as usize)
            .copied()
            .unwrap_or(true)
    }

    /// Whether every invocation of a workgroup that evaluates `v` of `e`'s entry body holds the
    /// same value (an `undef`/`poison` constant does not).
    pub fn is_uniform_value(&self, e: &Entry, v: Value) -> bool {
        match v {
            Value::Arg(a) => !self.args.get(a.index()).copied().unwrap_or(true),
            Value::Inst(i) => !self.values.get(i.index()).copied().unwrap_or(true),
            Value::Const(c) => !const_divergent(e, c),
        }
    }

    /// Bytes this structure keeps (charged by [`Uniformity::of`]; the caller releases them).
    pub fn ledger_bytes(&self) -> usize {
        self.divergent.len() + self.values.len() + self.args.len()
    }
}

/// `undef`/`poison` anywhere in constant `c`: each use may read another value. Constants are
/// acyclic; the depth is bounded by the type nesting.
fn const_divergent(e: &Entry, c: ConstId) -> bool {
    let consts = e.module.consts();
    let mut stack = vec![c];
    while let Some(c) = stack.pop() {
        match consts.get(c).map(|k| &k.value) {
            Some(Const::Undef | Const::Poison) | None => return true,
            Some(v) => stack.extend(v.children().iter().copied()),
        }
    }
    false
}

/// The invocations a [`Uniformity`] relates: those of one workgroup, those of one simdgroup
/// (the 32 consecutive linear threads `[32k, 32k + 32)`), or a vertex function's that write the
/// vertices of one primitive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Among {
    Workgroup,
    Simdgroup,
    Primitive,
}

/// The most lanes of a vector the analysis tracks lane by lane (the module docs, "Lanes"); a
/// value's lanes are the bits of a mask, and a value of no tracked lanes (a scalar, an aggregate,
/// a wider vector) is divergent in [`ALL`] of them or in none.
pub const LANES: u32 = 8;

/// Every lane of a value divergent.
const ALL: u8 = u8::MAX;

/// The axes of a threadgroup of one thread along each: bit 0 `x`, bit 1 `y`, bit 2 `z`.
pub type UnitAxes = u8;

/// The lanes of the dispatch builtin `b`, a parameter of `lanes` tracked lanes (`None` a scalar),
/// that are divergent among the invocations `among` relates in threadgroups of one thread along
/// the axes `unit` (the module docs, "Unit axes"); [`ALL`] where every lane may be.
fn builtin_lanes(b: &Builtin, among: Among, lanes: Option<u32>, unit: UnitAxes) -> u8 {
    if !per_invocation(b, among) {
        return 0;
    }
    let positional = matches!(
        b,
        Builtin::ThreadPositionInThreadgroup | Builtin::ThreadPositionInGrid
    );
    if !positional || unit == 0 || among == Among::Primitive {
        return ALL;
    }
    match lanes {
        // A scalar position is its `x`.
        None => match unit & 1 {
            0 => ALL,
            _ => 0,
        },
        Some(n) => lane_bits(n) & !unit,
    }
}

/// The mask of a value of `n` tracked lanes.
fn lane_bits(n: u32) -> u8 {
    match n {
        0 => 0,
        n if n >= 8 => ALL,
        n => (1u8 << n) - 1,
    }
}

/// Whether the dispatch builtin `b` may differ between the invocations `among` relates.
fn per_invocation(b: &Builtin, among: Among) -> bool {
    if among == Among::Simdgroup && *b == Builtin::SimdgroupIndexInThreadgroup {
        return false;
    }
    // The vertices of one primitive share their instance, patch and amplification.
    if among == Among::Primitive {
        return !matches!(
            b,
            Builtin::InstanceId
                | Builtin::BaseVertex
                | Builtin::BaseInstance
                | Builtin::AmplificationId
                | Builtin::AmplificationCount
                | Builtin::PatchId
        );
    }
    !matches!(
        b,
        Builtin::ThreadgroupPositionInGrid
            | Builtin::ThreadsPerThreadgroup
            | Builtin::ThreadsPerGrid
            | Builtin::ThreadgroupsPerGrid
            | Builtin::ThreadsPerSimdgroup
            | Builtin::SimdgroupsPerThreadgroup
            | Builtin::QuadgroupsPerThreadgroup
            | Builtin::DispatchThreadsPerThreadgroup
            | Builtin::DispatchSimdgroupsPerThreadgroup
    )
}

/// Whether the parameter role `r` may differ between the invocations `among` relates.
fn divergent_role(r: &ParamRole, among: Among) -> bool {
    match r {
        ParamRole::Builtin(b) => per_invocation(b, among),
        ParamRole::Buffer(_)
        | ParamRole::ArgumentBuffer(_)
        | ParamRole::Threadgroup(_)
        | ParamRole::Texture(_)
        | ParamRole::Sampler { .. }
        | ParamRole::AccelerationStructure { .. }
        | ParamRole::FunctionTable(_)
        | ParamRole::IntersectionTable(_)
        | ParamRole::IndirectConstant { .. }
        | ParamRole::CommandBuffer(_) => false,
        _ => true,
    }
}

/// Whether a call's result is one value for every active thread of a simdgroup whatever its
/// operands (a reduction, vote, ballot, the active mask, or a broadcast of the first active lane);
/// a broadcast of a lane is when its lane operand is uniform.
fn simdgroup_collective(callee: &Callee) -> Option<bool> {
    let Callee::Air {
        op: AirIntrinsic::Group(g),
        ..
    } = callee
    else {
        return None;
    };
    if g.scope != GroupScope::Simdgroup {
        return None;
    }
    match g.op {
        GroupOp::BroadcastFirst
        | GroupOp::Sum
        | GroupOp::Product
        | GroupOp::Min
        | GroupOp::Max
        | GroupOp::And
        | GroupOp::Or
        | GroupOp::Xor
        | GroupOp::All
        | GroupOp::Any
        | GroupOp::Ballot
        | GroupOp::ActiveThreadsMask => Some(false),
        GroupOp::Broadcast => Some(true),
        _ => None,
    }
}

/// Whether a call's result depends only on its operands (so it is divergent only when one is).
fn per_invocation_arithmetic(callee: &Callee) -> bool {
    match callee {
        Callee::Air { op, .. } => matches!(
            op,
            AirIntrinsic::Math(_)
                | AirIntrinsic::Convert(_)
                | AirIntrinsic::Pack(_)
                | AirIntrinsic::BoolReduce(_)
                | AirIntrinsic::SimdgroupSize(_)
        ),
        Callee::Llvm { op, .. } => matches!(
            op,
            LlvmIntrinsic::Float { .. }
                | LlvmIntrinsic::Int { .. }
                | LlvmIntrinsic::NarrowInt { .. }
                | LlvmIntrinsic::Abs { .. }
                | LlvmIntrinsic::Cttz { .. }
                | LlvmIntrinsic::Ctlz { .. }
        ),
        _ => false,
    }
}

/// What a run reads the loop forest for.
#[derive(Clone, Copy)]
enum Mode<'a> {
    /// Whether invocations execute each block together, without the barrier contract.
    Executed,
    /// The same, taking the barrier contract over these loops.
    UnderBarriers(&'a LoopForest<BlockId>),
    /// The values per dynamic instance over these loops (the module docs, "Instances").
    PerInstance(&'a LoopForest<BlockId>),
}

/// No label ([`Instance::label`]).
const UNLABELED: u32 = u32::MAX;

/// The per-instance rules' state (the module docs, "Instances").
struct Instance<'a> {
    loops: &'a LoopForest<BlockId>,
    /// By block: a join of a divergent branch, whose phis are divergent.
    joins: Vec<bool>,
    /// By loop: it holds a divergent branch, so its threads may leave it after different
    /// iterations.
    apart: Vec<bool>,
    /// By instruction: divergent because an instruction outside a loop held apart uses it.
    forced: Vec<bool>,
    /// By block, for the branch being walked: the successor whose paths reach the block, or the
    /// block itself where paths from two meet ([`UNLABELED`] for none yet).
    label: Vec<u32>,
    /// By block: the walk that last placed it in its region ([`Instance::walk`]).
    stamp: Vec<u32>,
    /// The current walk, from 1.
    walk: u32,
    /// The current walk's region, in reverse postorder.
    region: Vec<BlockId>,
}

/// The most passes a branch's join walk takes over its region before it takes every block of
/// two or more predecessors there as a join (each pass that changes nothing ends the walk; a
/// reducible region settles in three).
const JOIN_PASSES: u32 = 8;

/// One item of the worklist.
#[derive(Clone, Copy)]
enum Item {
    /// Re-evaluate the instruction.
    Inst(InstId),
    /// The block turned divergent.
    Block(BlockId),
}

struct State<'a> {
    e: &'a Entry,
    body: &'a Body,
    among: Among,
    /// By parameter and by instruction: the divergent lanes (the module docs, "Lanes").
    args: Vec<u8>,
    insts: Vec<u8>,
    blocks: Vec<bool>,
    /// By block: uniform by the barrier contract, never divergent (empty without the contract).
    pinned: Vec<bool>,
    /// The block of each instruction.
    home: Vec<u32>,
    /// The users of each instruction, as a CSR table: `users[start[i]..start[i + 1]]`.
    start: Vec<u32>,
    users: Vec<InstId>,
    /// Blocks whose branch region was already marked.
    marked: Vec<bool>,
    work: Vec<Item>,
    /// The per-instance rules' state, in that mode.
    instance: Option<Instance<'a>>,
}

impl State<'_> {
    fn value_divergent(&self, v: Value) -> bool {
        self.lanes_of(v) != 0
    }

    /// The divergent lanes of `v`.
    fn lanes_of(&self, v: Value) -> u8 {
        match v {
            Value::Arg(a) => self.args.get(a.index()).copied().unwrap_or(ALL),
            Value::Const(c) => match const_divergent(self.e, c) {
                true => ALL,
                false => 0,
            },
            Value::Inst(i) => self.insts.get(i.index()).copied().unwrap_or(ALL),
        }
    }

    /// The tracked lanes of a value of type `ty`: `None` for any other type (the module docs).
    fn tracked(&self, ty: TypeId) -> Option<u32> {
        match self.e.module.types().get(ty)? {
            Type::Vector { len, .. } if *len <= LANES => Some(*len),
            _ => None,
        }
    }

    /// The tracked lanes of `v`'s type.
    fn tracked_value(&self, v: Value) -> Option<u32> {
        self.tracked(self.e.module.value_ty(self.e.func, v)?)
    }

    /// The constant index `v`, where it is one below [`LANES`].
    fn lane_index(&self, v: Value) -> Option<u32> {
        let Value::Const(c) = v else {
            return None;
        };
        match self.e.module.consts().get(c).map(|k| &k.value) {
            Some(Const::Int(k)) if *k < u64::from(LANES) => u32::try_from(*k).ok(),
            Some(Const::Null) => Some(0),
            _ => None,
        }
    }

    /// The divergent lanes of a lane-wise operation's operand `v` in a result of `n` lanes: its
    /// own where it has `n` tracked lanes, every lane or none for a scalar, `None` for any other.
    fn operand_lanes(&self, v: Value, n: u32) -> Option<u8> {
        match self.tracked_value(v) {
            Some(k) if k == n => Some(self.lanes_of(v)),
            Some(_) => None,
            None => {
                let scalar = self.e.module.value_ty(self.e.func, v).is_some_and(|ty| {
                    matches!(
                        self.e.module.types().get(ty),
                        Some(
                            Type::Int(_)
                                | Type::Half
                                | Type::BFloat
                                | Type::Float
                                | Type::Double
                                | Type::Ptr { .. }
                        )
                    )
                });
                scalar.then(|| match self.value_divergent(v) {
                    true => lane_bits(n),
                    false => 0,
                })
            }
        }
    }

    /// The divergent lanes of instruction `i`, of `n` tracked lanes, where it moves or combines
    /// lanes (the module docs, "Lanes"); `None` where it does not, or an operand is not tracked.
    fn eval_lanes(&self, kind: &InstKind, n: u32) -> Option<u8> {
        let all = lane_bits(n);
        match kind {
            InstKind::InsertElement { vec, elt, idx } => {
                let k = self.lane_index(*idx)?;
                let vec = self.operand_lanes(*vec, n)?;
                let elt = match self.value_divergent(*elt) {
                    true => 1u8 << k,
                    false => 0,
                };
                Some((vec & !(1u8 << k) | elt) & all)
            }
            InstKind::ShuffleVector { a, b, mask } => {
                let na = self.tracked_value(*a)?;
                let (la, lb) = (self.lanes_of(*a), self.lanes_of(*b));
                let mut out = 0u8;
                for (j, lane) in mask.iter().enumerate() {
                    let src = (*lane)?;
                    let bit = match src < na {
                        true => la >> src & 1,
                        false => lb >> (src - na) & 1,
                    };
                    out |= bit << j;
                }
                Some(out & all)
            }
            InstKind::BinOp { a, b, .. }
            | InstKind::ICmp { a, b, .. }
            | InstKind::FCmp { a, b, .. } => {
                Some((self.operand_lanes(*a, n)? | self.operand_lanes(*b, n)?) & all)
            }
            InstKind::Select { c, t, f, .. } => Some(
                (self.operand_lanes(*c, n)?
                    | self.operand_lanes(*t, n)?
                    | self.operand_lanes(*f, n)?)
                    & all,
            ),
            InstKind::Cast { v, .. } | InstKind::FNeg { v, .. } | InstKind::Freeze(v) => {
                Some(self.operand_lanes(*v, n)? & all)
            }
            _ => None,
        }
    }

    fn block_divergent(&self, b: BlockId) -> bool {
        self.blocks.get(b.raw() as usize).copied().unwrap_or(true)
    }

    /// The block of instruction `i`.
    fn home_block(&self, i: InstId) -> BlockId {
        BlockId::new(self.home.get(i.index()).copied().unwrap_or(u32::MAX))
    }

    fn any_operand(&self, kind: &InstKind) -> bool {
        let mut any = false;
        kind.for_each_value(|v| any |= self.value_divergent(v));
        any
    }

    fn ptr_space(&self, v: Value) -> Option<AddrSpace> {
        let ty = self.e.module.value_ty(self.e.func, v)?;
        match self.e.module.types().get(ty)? {
            Type::Ptr { space } => Some(*space),
            _ => None,
        }
    }

    /// The divergent lanes of instruction `i` under the current facts.
    fn eval(&self, i: InstId) -> u8 {
        let Some(inst) = self.body.insts.get(i) else {
            return ALL;
        };
        let forced = |x: &Instance<'_>| x.forced.get(i.index()).copied().unwrap_or(true);
        if self.instance.as_ref().is_some_and(forced) {
            return ALL;
        }
        if let Some(n) = self.tracked(inst.ty) {
            if let Some(lanes) = self.eval_lanes(&inst.kind, n) {
                return lanes;
            }
        }
        // An element extracted at a constant index is its lane's.
        if let InstKind::ExtractElement { vec, idx } = &inst.kind {
            if let (Some(k), Some(_)) = (self.lane_index(*idx), self.tracked_value(*vec)) {
                return match self.lanes_of(*vec) >> k & 1 {
                    0 => 0,
                    _ => ALL,
                };
            }
        }
        match self.eval_whole(i, inst) {
            true => ALL,
            false => 0,
        }
    }

    /// Whether instruction `i`, `inst`, is divergent as a whole under the current facts.
    fn eval_whole(&self, i: InstId, inst: &Inst) -> bool {
        match &inst.kind {
            InstKind::Phi { incoming, .. } => match &self.instance {
                Some(x) => {
                    let home = self.home_block(i);
                    x.joins.get(home.index()).copied().unwrap_or(true)
                        || incoming.iter().any(|(v, _)| self.value_divergent(*v))
                }
                None => incoming
                    .iter()
                    .any(|(v, from)| self.value_divergent(*v) || self.block_divergent(*from)),
            },
            InstKind::Load {
                ptr,
                volatile,
                atomic,
                ..
            } => {
                *volatile
                    || atomic.is_some()
                    || self.ptr_space(*ptr) != Some(AddrSpace::Constant)
                    || self.value_divergent(*ptr)
            }
            InstKind::Alloca { .. } | InstKind::AtomicRmw { .. } | InstKind::CmpXchg { .. } => true,
            InstKind::ShuffleVector { mask, .. } if mask.iter().any(Option::is_none) => true,
            InstKind::Call(call) => match simdgroup_collective(&call.callee) {
                Some(by_lane) if self.among == Among::Simdgroup => {
                    self.block_divergent(self.home_block(i))
                        || (by_lane && call.args.get(1).is_none_or(|&l| self.value_divergent(l)))
                }
                _ => !per_invocation_arithmetic(&call.callee) || self.any_operand(&inst.kind),
            },
            kind => self.any_operand(kind),
        }
    }

    fn set_block(&mut self, b: BlockId) {
        if self.pinned.get(b.raw() as usize).copied().unwrap_or(false) {
            return;
        }
        if let Some(d) = self.blocks.get_mut(b.raw() as usize) {
            if !*d {
                *d = true;
                self.work.push(Item::Block(b));
            }
        }
    }

    /// Mark the blocks control dependent on the branch that ends block `a` divergent.
    fn mark_region(
        &mut self,
        a: BlockId,
        pdom: &PostDomTree<BlockId>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        match self.marked.get_mut(a.raw() as usize) {
            Some(done) if !*done => *done = true,
            _ => return Ok(()),
        }
        let stop = pdom.ipdom(a);
        let succs: Vec<BlockId> = self.body.successors(a).collect();
        for s in succs {
            let mut x = Some(s);
            while let Some(b) = x {
                if Some(b) == stop {
                    break;
                }
                m.tick(1)?;
                self.set_block(b);
                x = pdom.ipdom(b);
            }
        }
        Ok(())
    }

    fn users(&self, i: InstId) -> &[InstId] {
        let at = |k: usize| self.start.get(k).copied().unwrap_or(0) as usize;
        let (lo, hi) = (at(i.index()), at(i.index() + 1));
        self.users.get(lo..hi).unwrap_or(&[])
    }

    /// Under the per-instance rules, what the branch that ends block `d` turning divergent does
    /// (the module docs, "Instances"): the phis at its joins turn divergent, and so do the values
    /// that instructions outside a loop around it use.
    fn split(
        &mut self,
        d: BlockId,
        pdom: &PostDomTree<BlockId>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        if self.instance.is_none() {
            return Ok(());
        }
        self.mark_joins(d, pdom, m)?;
        self.mark_apart(d, m)
    }

    /// Mark the joins of the divergent branch that ends block `d`, and queue their phis. Each
    /// block of the region (what `d`'s successors reach without passing `d`, up to `d`'s immediate
    /// post-dominator) takes the label of the successor whose paths reach it, or its own where
    /// two labels meet there, which makes it a join; passes in reverse postorder repeat until no
    /// label changes, at most [`JOIN_PASSES`] times.
    fn mark_joins(
        &mut self,
        d: BlockId,
        pdom: &PostDomTree<BlockId>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let (body, cfg) = (self.body, &self.e.cfg);
        let Some(x) = self.instance.as_mut() else {
            return Ok(());
        };
        let stop = pdom.ipdom(d);
        x.walk = x.walk.wrapping_add(1);
        if x.walk == 0 {
            x.stamp.iter_mut().for_each(|s| *s = 0);
            x.walk = 1;
        }
        let walk = x.walk;
        let mut succs: Vec<BlockId> = body.successors(d).filter(|&s| s != d).collect();
        succs.sort_unstable();
        succs.dedup();
        x.region.clear();
        let mut stack = succs.clone();
        while let Some(b) = stack.pop() {
            m.tick(1)?;
            let (Some(stamp), Some(label)) =
                (x.stamp.get_mut(b.index()), x.label.get_mut(b.index()))
            else {
                continue;
            };
            if b == d || *stamp == walk {
                continue;
            }
            *stamp = walk;
            *label = UNLABELED;
            x.region.push(b);
            if Some(b) != stop {
                stack.extend(body.successors(b));
            }
        }
        x.region
            .sort_unstable_by_key(|b| cfg.rpo_number(*b).unwrap_or(u32::MAX));
        let mut joined = Vec::new();
        let mut settled = false;
        for _ in 0..JOIN_PASSES {
            let mut changed = false;
            for k in 0..x.region.len() {
                m.tick(1)?;
                let Some(&b) = x.region.get(k) else {
                    continue;
                };
                let mut seen = UNLABELED;
                let mut join = false;
                let mut see = |l: u32| match seen {
                    UNLABELED => seen = l,
                    s if s != l => join = true,
                    _ => {}
                };
                if succs.binary_search(&b).is_ok() {
                    see(b.raw());
                }
                for p in cfg.predecessors(b) {
                    let i = p.index();
                    if p != d && x.stamp.get(i) == Some(&walk) {
                        match x.label.get(i).copied().unwrap_or(UNLABELED) {
                            UNLABELED => {}
                            l => see(l),
                        }
                    }
                }
                let new = match join {
                    true => b.raw(),
                    false => seen,
                };
                if x.label.get(b.index()) != Some(&new) {
                    if let Some(l) = x.label.get_mut(b.index()) {
                        *l = new;
                    }
                    changed = true;
                }
                if join {
                    if let Some(j) = x.joins.get_mut(b.index()).filter(|j| !**j) {
                        *j = true;
                        joined.push(b);
                    }
                }
            }
            if !changed {
                settled = true;
                break;
            }
        }
        if !settled {
            // An unsettled region: every block where paths may meet is a join.
            for k in 0..x.region.len() {
                let Some(&b) = x.region.get(k) else {
                    continue;
                };
                if cfg.predecessors(b).len() > 1 {
                    if let Some(j) = x.joins.get_mut(b.index()).filter(|j| !**j) {
                        *j = true;
                        joined.push(b);
                    }
                }
            }
        }
        for b in joined {
            let Some(block) = body.blocks.get(b) else {
                continue;
            };
            for id in block.inst_ids() {
                match body.insts.get(id).map(|i| &i.kind) {
                    Some(InstKind::Phi { .. }) => self.work.push(Item::Inst(id)),
                    _ => break,
                }
            }
        }
        Ok(())
    }

    /// Hold apart every loop around block `d`, whose branch turned divergent, not already held
    /// apart (an enclosing one is when an inner one is): force divergent each value defined in
    /// such a loop that an instruction outside it uses, and queue it.
    fn mark_apart(&mut self, d: BlockId, m: &mut Meter) -> Result<(), Error> {
        let body = self.body;
        let mut chain = Vec::new();
        let mut forced = Vec::new();
        if let Some(x) = &self.instance {
            let mut l = x.loops.innermost(d);
            let mut inner = None;
            while let Some(lp) = l {
                if x.apart.get(lp.index()).copied().unwrap_or(true) {
                    break;
                }
                chain.push(lp);
                for b in x.loops.blocks(lp) {
                    // The inner loop's values were read: a use outside this loop is outside it.
                    if inner.is_some_and(|i| x.loops.contains(i, b)) {
                        continue;
                    }
                    let Some(block) = body.blocks.get(b) else {
                        continue;
                    };
                    for id in block.inst_ids() {
                        m.tick(1)?;
                        let leaves = self
                            .users(id)
                            .iter()
                            .any(|&u| !x.loops.contains(lp, self.home_block(u)));
                        if leaves && !x.forced.get(id.index()).copied().unwrap_or(true) {
                            forced.push(id);
                        }
                    }
                }
                inner = Some(lp);
                l = x.loops.parent(lp);
            }
        }
        if let Some(x) = self.instance.as_mut() {
            for lp in chain {
                if let Some(a) = x.apart.get_mut(lp.index()) {
                    *a = true;
                }
            }
            for &id in &forced {
                if let Some(f) = x.forced.get_mut(id.index()) {
                    *f = true;
                }
            }
        }
        self.work.extend(forced.into_iter().map(Item::Inst));
        Ok(())
    }
}

impl Uniformity {
    /// The uniformity of the blocks of `e`'s entry body. The returned structure stays charged to
    /// `m` ([`Uniformity::ledger_bytes`]); the analysis' own tables are released.
    pub fn of(e: &Entry, body: &Body, m: &mut Meter) -> Result<Uniformity, Error> {
        Uniformity::analyze(e, body, Among::Workgroup, Mode::Executed, &[], 0, m)
    }

    /// [`of`](Self::of) for a dispatch whose threadgroups have one thread along the axes `unit`
    /// (the module docs, "Unit axes").
    pub fn with_unit_axes(
        e: &Entry,
        body: &Body,
        unit: UnitAxes,
        m: &mut Meter,
    ) -> Result<Uniformity, Error> {
        Uniformity::analyze(e, body, Among::Workgroup, Mode::Executed, &[], unit, m)
    }

    /// The uniformity of the blocks and values of vertex function `e`'s entry body among the
    /// invocations that write the vertices of one primitive ([`Among::Primitive`]), with the
    /// parameters `uniform` (vertex attributes fetched per instance or at a constant step) taken
    /// as uniform.
    pub fn among_primitive(
        e: &Entry,
        body: &Body,
        uniform: &[u32],
        m: &mut Meter,
    ) -> Result<Uniformity, Error> {
        Uniformity::analyze(e, body, Among::Primitive, Mode::Executed, uniform, 0, m)
    }

    /// The uniformity of the blocks of `e`'s entry body among the invocations `among` relates,
    /// taking the barrier contract (see the module documentation; `loops` is the loop forest of
    /// `e`'s CFG). For a simdgroup, its index in the threadgroup and the results of its
    /// collective operations (reductions, votes, ballots, the active mask, and broadcasts of a
    /// uniform lane) are uniform too.
    pub fn under_barriers(
        e: &Entry,
        body: &Body,
        among: Among,
        loops: &LoopForest<BlockId>,
        m: &mut Meter,
    ) -> Result<Uniformity, Error> {
        Uniformity::analyze(e, body, among, Mode::UnderBarriers(loops), &[], 0, m)
    }

    /// The uniformity of the values of `e`'s entry body among the invocations of one workgroup
    /// that execute the same dynamic instance of an instruction (the module docs, "Instances";
    /// `loops` is the loop forest of `e`'s CFG). Its block facts are [`of`](Self::of)'s.
    pub fn per_instance(
        e: &Entry,
        body: &Body,
        loops: &LoopForest<BlockId>,
        m: &mut Meter,
    ) -> Result<Uniformity, Error> {
        Uniformity::analyze(
            e,
            body,
            Among::Workgroup,
            Mode::PerInstance(loops),
            &[],
            0,
            m,
        )
    }

    fn analyze(
        e: &Entry,
        body: &Body,
        among: Among,
        mode: Mode<'_>,
        uniform: &[u32],
        unit: UnitAxes,
        m: &mut Meter,
    ) -> Result<Uniformity, Error> {
        let n = body.insts.len();
        let nb = body.blocks.len();
        // The users table, the per-instruction flags, homes, and worklist items, and the
        // per-block flags and worklist items (each instruction and block turns divergent once; a
        // simdgroup collective is queued once more when its block does; the users of a tracked
        // vector are queued once more per lane it turns divergent in after its first).
        let types = e.module.types();
        let vector = |d: InstId| {
            body.insts.get(d).is_some_and(|inst| {
                matches!(types.get(inst.ty), Some(Type::Vector { len, .. }) if *len <= LANES)
            })
        };
        let (mut uses, mut vector_uses) = (0usize, 0usize);
        for inst in body.insts.values() {
            inst.kind.for_each_value(|v| {
                if let Value::Inst(d) = v {
                    uses += 1;
                    vector_uses += usize::from(vector(d));
                }
            });
        }
        let scratch = n
            .saturating_mul(4 + 4 + 3 * 16)
            .saturating_add(uses.saturating_mul(4))
            .saturating_add(vector_uses.saturating_mul(16 * (LANES as usize - 1)))
            .saturating_add(nb.saturating_mul(2 + 16))
            .saturating_add(match mode {
                // The joins, labels, stamps and region by block, the forced flags and their
                // worklist items by instruction, a walk's stack by edge, the loops' flags.
                Mode::PerInstance(loops) => nb
                    .saturating_mul(1 + 4 + 4 + 4)
                    .saturating_add(n.saturating_mul(1 + 16))
                    .saturating_add(
                        (body.blocks.iter())
                            .map(|(b, _)| body.successors(b).count())
                            .sum::<usize>()
                            .saturating_mul(4),
                    )
                    .saturating_add(loops.len()),
                _ => 0,
            });
        // What the result keeps: the block, instruction, and argument flags.
        let kept = nb.saturating_add(n).saturating_add(e.ep.params.len());
        m.charge(scratch.saturating_add(kept))?;
        let result = PostDomTree::compute(&e.cfg, m)
            .map_err(Error::from)
            .and_then(|pdom| {
                let r = State::new(e, body, among, mode, uniform, unit, m)
                    .and_then(|st| match mode {
                        Mode::UnderBarriers(loops) => st.pin(&pdom, loops, m),
                        _ => Ok(st),
                    })
                    .and_then(|st| st.run(&pdom, m));
                m.release(pdom.ledger_bytes());
                r
            });
        m.release(scratch);
        match result {
            Ok(u) => {
                // The argument flags are by parameter, which may differ from the bound count.
                m.release(kept);
                m.charge(u.ledger_bytes())?;
                Ok(u)
            }
            Err(e) => {
                m.release(kept);
                Err(e)
            }
        }
    }
}

impl<'a> State<'a> {
    fn new(
        e: &'a Entry,
        body: &'a Body,
        among: Among,
        mode: Mode<'a>,
        uniform: &[u32],
        unit: UnitAxes,
        m: &mut Meter,
    ) -> Result<State<'a>, Error> {
        let n = body.insts.len();
        let nb = body.blocks.len();
        let params: Vec<TypeId> = e
            .module
            .func(e.func)
            .and_then(|f| match e.module.types().get(f.ty) {
                Some(Type::Function { params, .. }) => Some(params.to_vec()),
                _ => None,
            })
            .unwrap_or_default();
        let mut args = vec![ALL; params.len()];
        for b in &e.ep.params {
            let lanes =
                (params.get(b.param as usize)).and_then(|&ty| match e.module.types().get(ty) {
                    Some(Type::Vector { len, .. }) if *len <= LANES => Some(*len),
                    _ => None,
                });
            if let Some(a) = args.get_mut(b.param as usize) {
                *a = match &b.role {
                    ParamRole::Builtin(builtin) => builtin_lanes(builtin, among, lanes, unit),
                    role => match divergent_role(role, among) {
                        true => ALL,
                        false => 0,
                    },
                };
            }
        }
        for &p in uniform {
            if let Some(a) = args.get_mut(p as usize) {
                *a = 0;
            }
        }
        let mut home = vec![u32::MAX; n];
        for (b, block) in body.blocks.iter() {
            for id in block.inst_ids() {
                if let Some(h) = home.get_mut(id.index()) {
                    *h = b.raw();
                }
            }
        }
        // Users, counted then placed.
        let mut start = vec![0u32; n + 1];
        for inst in body.insts.values() {
            m.tick(1)?;
            inst.kind.for_each_value(|v| {
                if let Value::Inst(d) = v {
                    if let Some(c) = start.get_mut(d.index() + 1) {
                        *c += 1;
                    }
                }
            });
        }
        for k in 1..start.len() {
            let prev = start.get(k - 1).copied().unwrap_or(0);
            if let Some(c) = start.get_mut(k) {
                *c += prev;
            }
        }
        let mut fill = start.clone();
        let mut users = vec![InstId::new(0); start.last().copied().unwrap_or(0) as usize];
        for (id, inst) in body.insts.iter() {
            inst.kind.for_each_value(|v| {
                if let Value::Inst(d) = v {
                    if let Some(at) = fill.get_mut(d.index()) {
                        if let Some(slot) = users.get_mut(*at as usize) {
                            *slot = id;
                        }
                        *at += 1;
                    }
                }
            });
        }
        Ok(State {
            e,
            body,
            among,
            args,
            insts: vec![0; n],
            blocks: vec![false; nb],
            pinned: Vec::new(),
            home,
            start,
            users,
            marked: vec![false; nb],
            work: Vec::new(),
            instance: match mode {
                Mode::PerInstance(loops) => Some(Instance {
                    loops,
                    joins: vec![false; nb],
                    apart: vec![false; loops.len()],
                    forced: vec![false; n],
                    label: vec![UNLABELED; nb],
                    stamp: vec![0; nb],
                    walk: 0,
                    region: Vec::with_capacity(nb),
                }),
                _ => None,
            },
        })
    }

    /// Pin the blocks the barrier contract makes uniform (see the module documentation).
    fn pin(
        mut self,
        pdom: &PostDomTree<BlockId>,
        loops: &LoopForest<BlockId>,
        m: &mut Meter,
    ) -> Result<State<'a>, Error> {
        let body = self.body;
        let nb = body.blocks.len();
        let mut pinned = vec![false; nb];
        let mut barriers = Vec::new();
        for (b, block) in body.blocks.iter() {
            for id in block.inst_ids() {
                m.tick(1)?;
                let Some(InstKind::Call(call)) = body.insts.get(id).map(|i| &i.kind) else {
                    continue;
                };
                let Callee::Air {
                    op: AirIntrinsic::Sync(Sync::Barrier { scope, .. }),
                    ..
                } = &call.callee
                else {
                    continue;
                };
                let covers = match scope {
                    ThreadScope::Threadgroup => true,
                    ThreadScope::Simdgroup => self.among == Among::Simdgroup,
                    _ => false,
                };
                if let Some(p) = pinned.get_mut(b.index()).filter(|p| covers && !**p) {
                    *p = true;
                    m.grow(&mut barriers, 1)?;
                    barriers.push(b);
                }
            }
        }
        // The blocks in no loop that run exactly when a barrier block in no loop does: up its
        // dominator chain while it post-dominates them (a block already pinned was walked from).
        for &p in &barriers {
            if loops.innermost(p).is_some() {
                continue;
            }
            let mut x = self.e.dom.idom(p);
            while let Some(xb) = x {
                m.tick(1)?;
                let done = pinned.get(xb.index()).copied().unwrap_or(true);
                if done || loops.innermost(xb).is_some() || !pdom.post_dominates(p, xb) {
                    break;
                }
                if let Some(slot) = pinned.get_mut(xb.index()) {
                    *slot = true;
                }
                x = self.e.dom.idom(xb);
            }
        }
        m.release(crate::budget::vec_bytes(&barriers));
        self.pinned = pinned;
        Ok(self)
    }

    fn run(mut self, pdom: &PostDomTree<BlockId>, m: &mut Meter) -> Result<Uniformity, Error> {
        let body = self.body;
        // A reachable block that reaches no exit has no post-dominator information.
        for (b, _) in body.blocks.iter() {
            if self.e.cfg.is_reachable(b) && !pdom.is_reachable(b) {
                self.set_block(b);
            }
        }
        for (id, _) in body.insts.iter().rev() {
            self.work.push(Item::Inst(id));
        }
        while let Some(item) = self.work.pop() {
            m.tick(1)?;
            match item {
                Item::Inst(i) => {
                    let Some(inst) = body.insts.get(i) else {
                        continue;
                    };
                    let branch =
                        matches!(inst.kind, InstKind::CondBr { .. } | InstKind::Switch { .. });
                    // A value's divergent lanes only grow: a scalar's once, a vector's at most
                    // once per lane.
                    let old = self.insts.get(i.index()).copied().unwrap_or(ALL);
                    if old == ALL {
                        continue;
                    }
                    let new = old | self.eval(i);
                    if new == old {
                        continue;
                    }
                    if let Some(d) = self.insts.get_mut(i.index()) {
                        *d = new;
                    }
                    if branch && old == 0 {
                        let a = BlockId::new(self.home.get(i.index()).copied().unwrap_or(0));
                        self.mark_region(a, pdom, m)?;
                        self.split(a, pdom, m)?;
                    }
                    let us: Vec<InstId> = self.users(i).to_vec();
                    self.work.extend(us.into_iter().map(Item::Inst));
                }
                Item::Block(b) => {
                    // Its branch is divergent wherever it leads, and its successors' phis are
                    // entered from a divergent block.
                    self.mark_region(b, pdom, m)?;
                    // Its simdgroup collectives' threads may hold different executions' results.
                    if self.among == Among::Simdgroup {
                        if let Some(block) = body.blocks.get(b) {
                            for id in block.inst_ids() {
                                m.tick(1)?;
                                let collective = matches!(
                                    body.insts.get(id).map(|i| &i.kind),
                                    Some(InstKind::Call(c)) if simdgroup_collective(&c.callee).is_some()
                                );
                                if collective {
                                    self.work.push(Item::Inst(id));
                                }
                            }
                        }
                    }
                    let succs: Vec<BlockId> = body.successors(b).collect();
                    for s in succs {
                        let Some(block) = body.blocks.get(s) else {
                            continue;
                        };
                        for id in block.inst_ids() {
                            match body.insts.get(id).map(|i| &i.kind) {
                                Some(InstKind::Phi { .. }) => self.work.push(Item::Inst(id)),
                                _ => break,
                            }
                        }
                    }
                }
            }
        }
        Ok(Uniformity {
            divergent: self.blocks,
            values: self.insts.iter().map(|&l| l != 0).collect(),
            args: self.args.iter().map(|&l| l != 0).collect(),
        })
    }
}

#[cfg(test)]
mod tests;
