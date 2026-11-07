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
//! The analysis is one worklist over instructions and blocks, each of which turns divergent at
//! most once; its memory is charged to the meter.

use crate::air::abi::{Builtin, ParamRole};
use crate::air::intrinsics::{AirIntrinsic, GroupOp, GroupScope, LlvmIntrinsic, Sync, ThreadScope};
use crate::air::ir::{
    AddrSpace, BlockId, Body, Callee, Const, ConstId, Idx, InstId, InstKind, Type, Value,
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
    args: Vec<bool>,
    insts: Vec<bool>,
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
        match v {
            Value::Arg(a) => self.args.get(a.index()).copied().unwrap_or(true),
            Value::Const(c) => const_divergent(self.e, c),
            Value::Inst(i) => self.insts.get(i.index()).copied().unwrap_or(true),
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

    /// Whether instruction `i` is divergent under the current facts.
    fn eval(&self, i: InstId) -> bool {
        let Some(inst) = self.body.insts.get(i) else {
            return true;
        };
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
}

impl Uniformity {
    /// The uniformity of the blocks of `e`'s entry body. The returned structure stays charged to
    /// `m` ([`Uniformity::ledger_bytes`]); the analysis' own tables are released.
    pub fn of(e: &Entry, body: &Body, m: &mut Meter) -> Result<Uniformity, Error> {
        Uniformity::analyze(e, body, Among::Workgroup, None, &[], m)
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
        Uniformity::analyze(e, body, Among::Primitive, None, uniform, m)
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
        Uniformity::analyze(e, body, among, Some(loops), &[], m)
    }

    fn analyze(
        e: &Entry,
        body: &Body,
        among: Among,
        contract: Option<&LoopForest<BlockId>>,
        uniform: &[u32],
        m: &mut Meter,
    ) -> Result<Uniformity, Error> {
        let n = body.insts.len();
        let nb = body.blocks.len();
        // The users table, the per-instruction flags, homes, and worklist items, and the
        // per-block flags and worklist items (each instruction and block turns divergent once; a
        // simdgroup collective is queued once more when its block does).
        let mut uses = 0usize;
        for inst in body.insts.values() {
            inst.kind
                .for_each_value(|v| uses += usize::from(matches!(v, Value::Inst(_))));
        }
        let scratch = n
            .saturating_mul(4 + 4 + 3 * 16)
            .saturating_add(uses.saturating_mul(4))
            .saturating_add(nb.saturating_mul(2 + 16));
        // What the result keeps: the block, instruction, and argument flags.
        let kept = nb.saturating_add(n).saturating_add(e.ep.params.len());
        m.charge(scratch.saturating_add(kept))?;
        let result = PostDomTree::compute(&e.cfg, m)
            .map_err(Error::from)
            .and_then(|pdom| {
                let r = State::new(e, body, among, uniform, m)
                    .and_then(|st| match contract {
                        Some(loops) => st.pin(&pdom, loops, m),
                        None => Ok(st),
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
        uniform: &[u32],
        m: &mut Meter,
    ) -> Result<State<'a>, Error> {
        let n = body.insts.len();
        let nb = body.blocks.len();
        let nargs = e
            .module
            .func(e.func)
            .and_then(|f| match e.module.types().get(f.ty) {
                Some(Type::Function { params, .. }) => Some(params.len()),
                _ => None,
            })
            .unwrap_or(0);
        let mut args = vec![true; nargs];
        for b in &e.ep.params {
            if let Some(a) = args.get_mut(b.param as usize) {
                *a = divergent_role(&b.role, among);
            }
        }
        for &p in uniform {
            if let Some(a) = args.get_mut(p as usize) {
                *a = false;
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
            insts: vec![false; n],
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
                    let already = self.insts.get(i.index()).copied().unwrap_or(true);
                    if already || !self.eval(i) {
                        continue;
                    }
                    if let Some(d) = self.insts.get_mut(i.index()) {
                        *d = true;
                    }
                    if branch {
                        let a = BlockId::new(self.home.get(i.index()).copied().unwrap_or(0));
                        self.mark_region(a, pdom, m)?;
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
            values: self.insts,
            args: self.args,
        })
    }
}

#[cfg(test)]
mod tests;
