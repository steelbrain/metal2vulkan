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
//! **Active lanes** ([`Uniformity::among_active_lanes`]). A value that may differ between the
//! active lanes of one execution of its instruction in a simdgroup, which is what the memory a
//! read reaches is chosen by (`memory::plan`'s lane reads): the analysis among a simdgroup
//! with no control divergence. No block turns divergent, so a phi is divergent only through its
//! incoming values: the lanes that run a loop's header together hold its counter alike, whatever
//! branch some lanes left by before the loop. A load is divergent only through its pointer (or as
//! a volatile or atomic one), in any address space: the lanes that run it together read the same
//! bytes at a uniform address; a thread's own memory is reached only through an alloca, which is
//! divergent. What this misses errs toward uniform: a phi joining uniform values after a divergent
//! branch, and a value a loop left on a per-lane condition computes, read after it.
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
    /// Among the active lanes (the module docs, "Active lanes"): no control divergence, and a
    /// load divergent only through its pointer.
    active: bool,
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
            InstKind::Phi { incoming, .. } => incoming
                .iter()
                .any(|(v, from)| self.value_divergent(*v) || self.block_divergent(*from)),
            InstKind::Load {
                ptr,
                volatile,
                atomic,
                ..
            } => {
                *volatile
                    || atomic.is_some()
                    || (!self.active && self.ptr_space(*ptr) != Some(AddrSpace::Constant))
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
}

impl Uniformity {
    /// The uniformity of the blocks of `e`'s entry body. The returned structure stays charged to
    /// `m` ([`Uniformity::ledger_bytes`]); the analysis' own tables are released.
    pub fn of(e: &Entry, body: &Body, m: &mut Meter) -> Result<Uniformity, Error> {
        Uniformity::analyze(e, body, Among::Workgroup, Flow::Control(None), &[], 0, m)
    }

    /// The values of `e`'s entry body that may differ between the active lanes of one execution
    /// of their instruction in a simdgroup (the module docs, "Active lanes"). Its blocks are all
    /// uniform: control divergence is not followed. The returned structure stays charged to `m`.
    pub fn among_active_lanes(e: &Entry, body: &Body, m: &mut Meter) -> Result<Uniformity, Error> {
        Uniformity::analyze(e, body, Among::Simdgroup, Flow::Data, &[], 0, m)
    }

    /// [`of`](Self::of) for a dispatch whose threadgroups have one thread along the axes `unit`
    /// (the module docs, "Unit axes").
    pub fn with_unit_axes(
        e: &Entry,
        body: &Body,
        unit: UnitAxes,
        m: &mut Meter,
    ) -> Result<Uniformity, Error> {
        Uniformity::analyze(e, body, Among::Workgroup, Flow::Control(None), &[], unit, m)
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
        Uniformity::analyze(
            e,
            body,
            Among::Primitive,
            Flow::Control(None),
            uniform,
            0,
            m,
        )
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
        Uniformity::analyze(e, body, among, Flow::Control(Some(loops)), &[], 0, m)
    }

    fn analyze(
        e: &Entry,
        body: &Body,
        among: Among,
        flow: Flow<'_>,
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
            .saturating_add(nb.saturating_mul(2 + 16));
        // What the result keeps: the block, instruction, and argument flags.
        let kept = nb.saturating_add(n).saturating_add(e.ep.params.len());
        m.charge(scratch.saturating_add(kept))?;
        let result = match flow {
            Flow::Data => {
                State::new(e, body, among, true, uniform, unit, m).and_then(|st| st.run(None, m))
            }
            Flow::Control(contract) => PostDomTree::compute(&e.cfg, m)
                .map_err(Error::from)
                .and_then(|pdom| {
                    let r = State::new(e, body, among, false, uniform, unit, m)
                        .and_then(|st| match contract {
                            Some(loops) => st.pin(&pdom, loops, m),
                            None => Ok(st),
                        })
                        .and_then(|st| st.run(Some(&pdom), m));
                    m.release(pdom.ledger_bytes());
                    r
                }),
        };
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

/// What divergence an analysis follows.
#[derive(Clone, Copy)]
enum Flow<'l> {
    /// Values and control: a block control dependent on a divergent branch is divergent, and so
    /// is a phi entered from one; with the barrier contract when the loop forest is given.
    Control(Option<&'l LoopForest<BlockId>>),
    /// Values alone, among the active lanes (the module docs, "Active lanes").
    Data,
}

impl<'a> State<'a> {
    fn new(
        e: &'a Entry,
        body: &'a Body,
        among: Among,
        active: bool,
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
            active,
            args,
            insts: vec![0; n],
            blocks: vec![false; nb],
            pinned: Vec::new(),
            home,
            start,
            users,
            marked: vec![false; nb],
            work: Vec::new(),
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

    /// Run the worklist; without `pdom` no control divergence is followed (no block turns
    /// divergent).
    fn run(
        mut self,
        pdom: Option<&PostDomTree<BlockId>>,
        m: &mut Meter,
    ) -> Result<Uniformity, Error> {
        let body = self.body;
        // A reachable block that reaches no exit has no post-dominator information.
        if let Some(pdom) = pdom {
            for (b, _) in body.blocks.iter() {
                if self.e.cfg.is_reachable(b) && !pdom.is_reachable(b) {
                    self.set_block(b);
                }
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
                    if let (true, 0, Some(pdom)) = (branch, old, pdom) {
                        let a = BlockId::new(self.home.get(i.index()).copied().unwrap_or(0));
                        self.mark_region(a, pdom, m)?;
                    }
                    let us: Vec<InstId> = self.users(i).to_vec();
                    self.work.extend(us.into_iter().map(Item::Inst));
                }
                Item::Block(b) => {
                    // Its branch is divergent wherever it leads, and its successors' phis are
                    // entered from a divergent block. (A block turns divergent only where
                    // control divergence is followed.)
                    if let Some(pdom) = pdom {
                        self.mark_region(b, pdom, m)?;
                    }
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
