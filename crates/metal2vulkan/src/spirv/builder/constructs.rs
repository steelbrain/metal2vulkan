//! Builder-closed constructs: `if_`, `switch_`, `switch_at_use`, `loop_`, `demote`/`kill`.
//!
//! These openers are the only way to create structured control flow. Each construct is entered
//! and merged inside one call, and its arms and bodies are closures run in place, so constructs
//! nest by closure scope and every rule of SPIR-V structured control flow holds by construction:
//!
//! - A selection header ends in `OpSelectionMerge M` and a conditional branch or switch to fresh
//!   arm blocks; each arm that falls through branches to `M`. Values that fall through merge in
//!   builder-owned `OpPhi`s at `M` (one per value, same order); arms that exit contribute no
//!   incoming edge. A merge that no arm reaches is `OpUnreachable`, and the construct leaves no
//!   block open.
//! - A loop has the canonical shape: the open block becomes the preheader and branches to an empty
//!   header `Hs = OpLoopMerge E C; OpBranch S0`, the body starts at `S0`, a body that falls off its
//!   end continues, the single continue block is `C = OpBranch Hs`, and `E` is a fresh merge that
//!   is `OpUnreachable` when no `break_` reaches it. Loops yield no SSA values: loop-carried and
//!   loop-out values live in `Function` variables. Every loop records a [`LoopMeta`].
//! - A switch of more literals than one `OpSwitch` may carry ([`SWITCH_PAIRS`]) is built as a
//!   switch on the chunk of arms the selector names, then a switch on the chunk index whose
//!   cases dispatch on the selector within the chunk (`split`): exactly the same arm runs for every
//!   selector value, each `OpSwitch` within the limit.
//! - A tangled instruction may sit in a source construct, including a source switch, but never
//!   inside a switch-at-use: the builder refuses it there as it is appended and `finalize` refuses
//!   a call from a switch-at-use to a function that holds one.
//! - Blocks are laid out when entered, so the layout is the construct preorder (header, contents,
//!   continue, merge) and each block follows its dominators.
//!
//! Shared state is threaded through a context parameter `ctx: &mut C` rather than captured, so
//! several arms can each use it mutably. An error returned from an arm or body aborts the function:
//! the builder is not rolled back, so the function cannot be finished.

use std::collections::BTreeSet;

use crate::spirv::builder::function::{operand, FunctionBuilder};
use crate::spirv::builder::module::BlockDraft;
use crate::spirv::error::{BResult, BuildError, ControlRule, Dangling, EntryRule, OperandRule};
use crate::spirv::ids::{BlockId, FuncId, TypeId, Value, VarId};
use crate::spirv::ir::{
    Block, EntryPoint, ExecModel, Function, Inst, LoopMeta, Merge, Phi, Shape, Terminator, Type,
    Unrolling, ValueOrigin,
};

/// An arm of `if_`/`switch_`: runs with the arm's first block open and says how it ends.
pub type ArmFn<'a, C> = Box<dyn FnOnce(&mut FunctionBuilder<'_>, &mut C) -> BResult<ArmEnd> + 'a>;

/// A loop body: runs with the body's first block open. Falling off the end continues the loop.
pub type LoopFn<'a, C> =
    Box<dyn FnOnce(&mut FunctionBuilder<'_>, &mut C, &LoopCtx) -> BResult<()> + 'a>;

/// How a construct arm ends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArmEnd {
    /// The arm's open block branches to the merge; these values are merged by `OpPhi`.
    FallThrough(Vec<Value>),
    /// The arm terminated its last block (return, kill, unreachable, break, continue).
    Exits,
}

/// The merged values of a selection: one `OpPhi` result per fall-through value, in order. Empty
/// when the arms carry no values or every arm exits; in the latter case no block is open after
/// the construct.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Merged(pub Vec<Value>);

/// What a loop is for, beyond its shape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoopSpec {
    /// The structurizer's multi-exit dispatch variable (a `u32`/`i32` `Function` variable).
    pub exit_sel: Option<VarId>,
    /// What the loop asks of the driver's unrolling (`LoopControl`).
    pub unrolling: Unrolling,
}

/// The exits of one open loop, handed to its body. `break_` and `continue_` are valid only while
/// this loop is the innermost open loop of its function.
#[derive(Debug)]
pub struct LoopCtx {
    func: FuncId,
    header: BlockId,
    merge: BlockId,
    cont: BlockId,
}

impl LoopCtx {
    /// Terminate the open block with a branch to the loop's merge.
    pub fn break_(&self, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
        fb.loop_exit(self, true)
    }

    /// Terminate the open block with a branch to the loop's continue block.
    pub fn continue_(&self, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
        fb.loop_exit(self, false)
    }
}

/// The terminator `kill` emits; `select` picks it from the profile.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KillForm {
    /// `OpKill`.
    Kill,
    /// `OpTerminateInvocation` (`SPV_KHR_terminate_invocation`).
    TerminateInvocation,
}

/// Per-function construct state: the open loops and switches (innermost last).
#[derive(Debug, Default)]
pub(super) struct ConstructState {
    loops: Vec<OpenLoop>,
    switches: Vec<OpenSwitch>,
}

impl ConstructState {
    /// Whether an open switch is a switch-at-use, so no tangled instruction may be added.
    pub(super) fn in_switch_at_use(&self) -> bool {
        self.switches.iter().any(|s| s.at_use)
    }
}

#[derive(Clone, Copy, Debug)]
struct OpenSwitch {
    at_use: bool,
}

#[derive(Clone, Copy, Debug)]
struct OpenLoop {
    header: BlockId,
    breaks: u32,
}

/// A fall-through arm: its values and the block that branches to the merge.
type Fall = (Vec<Value>, BlockId);

/// The most `(literal, label)` pairs one `OpSwitch` may carry: SPIR-V's universal limit (the
/// specification's "Universal Limits", which `spirv-val` enforces). A switch with more literals
/// is split into switches within it (`split`).
pub const SWITCH_PAIRS: usize = 16_383;

/// A switch's kind: a switch-at-use or a source switch, and the most `(literal, label)` pairs one
/// of its `OpSwitch`es may carry ([`SWITCH_PAIRS`]; the tests also use smaller limits, to check the
/// split on small switches).
#[derive(Clone, Copy, Debug)]
pub(in crate::spirv) struct SwitchKind {
    pub(in crate::spirv) at_use: bool,
    pub(in crate::spirv) limit: usize,
}

/// What a switch case runs: an arm, or (in a split switch) a chunk of the original cases,
/// dispatched by a nested switch on the original selector `sel`.
enum CaseBody<'a, C> {
    Arm(ArmFn<'a, C>),
    Chunk {
        sel: Value,
        cases: Vec<(Vec<u32>, CaseBody<'a, C>)>,
    },
}

impl FunctionBuilder<'_> {
    /// A structured selection on bool `c`: arm `t` when true, arm `f` (or nothing) when false.
    pub fn if_<C>(
        &mut self,
        ctx: &mut C,
        c: Value,
        t: ArmFn<'_, C>,
        f: Option<ArmFn<'_, C>>,
    ) -> BResult<Merged> {
        const OP: crate::spv::Op = crate::spv::Op::BranchConditional;
        let header = self.open_construct()?;
        match self.num(c)? {
            Some(n) if n.is_bool() && n.is_scalar() => {}
            _ => return Err(operand(OP, OperandRule::Category(0))),
        }
        let func = self.id();
        let tb = self.module().new_block(func)?;
        let fb = match f {
            Some(_) => Some(self.module().new_block(func)?),
            None => None,
        };
        let m = self.module().new_block(func)?;
        self.terminate(
            Some(Merge::Selection { merge: m }),
            Terminator::CondBranch {
                cond: c,
                t: tb,
                f: fb.unwrap_or(m),
            },
        )?;
        let mut falls = Vec::with_capacity(2);
        falls.extend(self.arm(ctx, tb, m, t)?);
        match (f, fb) {
            (Some(f), Some(b)) => falls.extend(self.arm(ctx, b, m, f)?),
            // The header's false edge reaches the merge directly, carrying no values.
            _ => falls.push((Vec::new(), header)),
        }
        self.merge_arms(m, falls)
    }

    /// A structured switch on 32-bit integer `sel`: each case's literals select its arm, other
    /// values the default arm. Arms never fall into one another.
    ///
    /// This is a source switch: one that the source program's control flow already has (a source
    /// `switch`, or the structurizer's exit dispatch), so a tangled instruction inside it keeps its
    /// source position and uniformity. Lowering's per-use dispatch over resource candidates uses
    /// [`switch_at_use`](Self::switch_at_use) instead.
    pub fn switch_<C>(
        &mut self,
        ctx: &mut C,
        sel: Value,
        cases: Vec<(Vec<u32>, ArmFn<'_, C>)>,
        default: ArmFn<'_, C>,
    ) -> BResult<Merged> {
        self.switch_impl(ctx, sel, cases, default, false)
    }

    /// A switch-at-use: lowering's dispatch over the candidates of a multi-candidate texture or
    /// sampler use, shaped as [`switch_`](Self::switch_). The switch has no source counterpart, so
    /// it must not contain a tangled instruction: appending one inside its arms (at any construct
    /// depth) fails with `ControlRule::TangledInSwitchAtUse`, and `finalize` refuses a call from
    /// its arms to a function that holds one. Tangled uses take the binding-array or
    /// hoisted-gradient forms instead.
    pub fn switch_at_use<C>(
        &mut self,
        ctx: &mut C,
        sel: Value,
        cases: Vec<(Vec<u32>, ArmFn<'_, C>)>,
        default: ArmFn<'_, C>,
    ) -> BResult<Merged> {
        self.switch_impl(ctx, sel, cases, default, true)
    }

    fn switch_impl<C>(
        &mut self,
        ctx: &mut C,
        sel: Value,
        cases: Vec<(Vec<u32>, ArmFn<'_, C>)>,
        default: ArmFn<'_, C>,
        at_use: bool,
    ) -> BResult<Merged> {
        let kind = SwitchKind {
            at_use,
            limit: SWITCH_PAIRS,
        };
        self.switch_within(ctx, sel, cases, default, kind)
    }

    /// [`switch_impl`](Self::switch_impl) of `kind`: a switch of more literals than
    /// `kind.limit` is split ([`split`]).
    pub(in crate::spirv) fn switch_within<C>(
        &mut self,
        ctx: &mut C,
        sel: Value,
        cases: Vec<(Vec<u32>, ArmFn<'_, C>)>,
        default: ArmFn<'_, C>,
        kind: SwitchKind,
    ) -> BResult<Merged> {
        let cases = cases
            .into_iter()
            .map(|(lits, arm)| (lits, CaseBody::Arm(arm)))
            .collect();
        self.switch_bodies(ctx, sel, cases, default, kind)
    }

    /// [`switch_within`](Self::switch_within) over case bodies.
    fn switch_bodies<C>(
        &mut self,
        ctx: &mut C,
        sel: Value,
        cases: Vec<(Vec<u32>, CaseBody<'_, C>)>,
        default: ArmFn<'_, C>,
        kind: SwitchKind,
    ) -> BResult<Merged> {
        let header = self.switch_header(sel)?;
        let mut literals = 0usize;
        {
            let mut seen = BTreeSet::new();
            for (lits, _) in &cases {
                if lits.is_empty() || !lits.iter().all(|l| seen.insert(*l)) {
                    return Err(BuildError::Control(ControlRule::CaseLiterals));
                }
                literals += lits.len();
            }
        }
        if literals > kind.limit {
            return self.split_switch(ctx, sel, cases, default, kind);
        }
        self.switch_construct(ctx, (header, sel), cases, default, kind)
    }

    /// The open block, about to become the header of a switch on `sel`, a 32-bit integer.
    fn switch_header(&self, sel: Value) -> BResult<BlockId> {
        const OP: crate::spv::Op = crate::spv::Op::Switch;
        let header = self.open_construct()?;
        match self.num(sel)? {
            Some(n) if n.is_int() && n.is_scalar() => {
                if n.width != 32 {
                    return Err(operand(OP, OperandRule::Width(0)));
                }
            }
            _ => return Err(operand(OP, OperandRule::Category(0))),
        }
        Ok(header)
    }

    /// The switch on `sel` from `header` over `cases`, whose literals are distinct and at most
    /// `kind.limit` (a chunk's nested switch has the same kind).
    fn switch_construct<C>(
        &mut self,
        ctx: &mut C,
        (header, sel): (BlockId, Value),
        cases: Vec<(Vec<u32>, CaseBody<'_, C>)>,
        default: ArmFn<'_, C>,
        kind: SwitchKind,
    ) -> BResult<Merged> {
        let func = self.id();
        let mut arms = Vec::with_capacity(cases.len());
        let mut targets = Vec::with_capacity(cases.iter().map(|(l, _)| l.len()).sum());
        for (lits, arm) in cases {
            let b = self.module().new_block(func)?;
            targets.extend(lits.iter().map(|l| (u64::from(*l), b)));
            arms.push((b, arm));
        }
        let db = self.module().new_block(func)?;
        let m = self.module().new_block(func)?;
        self.terminate(
            Some(Merge::Selection { merge: m }),
            Terminator::Switch {
                sel,
                default: db,
                cases: targets,
            },
        )?;
        if kind.at_use {
            self.module().switches_at_use.push(header);
        }
        self.cs.switches.push(OpenSwitch {
            at_use: kind.at_use,
        });
        let r = self.switch_arms(ctx, m, arms, (db, default), kind);
        self.cs.switches.pop();
        let falls = r?;
        self.merge_arms(m, falls)
    }

    /// Run a switch's case arms, then its default arm, each from its fresh block.
    fn switch_arms<'a, C>(
        &mut self,
        ctx: &mut C,
        m: BlockId,
        arms: Vec<(BlockId, CaseBody<'a, C>)>,
        default: (BlockId, ArmFn<'a, C>),
        kind: SwitchKind,
    ) -> BResult<Vec<Fall>> {
        let mut falls = Vec::with_capacity(arms.len() + 1);
        for (b, body) in arms {
            let fall = match body {
                CaseBody::Arm(arm) => self.arm(ctx, b, m, arm)?,
                CaseBody::Chunk { sel, cases } => self.chunk(ctx, (b, m), sel, cases, kind)?,
            };
            falls.extend(fall);
        }
        let (db, default) = default;
        falls.extend(self.arm(ctx, db, m, default)?);
        Ok(falls)
    }

    /// A structured loop of canonical shape. The open block becomes the preheader; the body runs
    /// with the body's first block open and exits through its [`LoopCtx`], `ret`, or a terminator.
    /// After the loop the merge block is open, unless no `break_` reaches it.
    pub fn loop_<C>(&mut self, ctx: &mut C, spec: LoopSpec, body: LoopFn<'_, C>) -> BResult<()> {
        let preheader = self.open_construct()?;
        if let Some(v) = spec.exit_sel {
            self.check_exit_sel(v)?;
        }
        let func = self.id();
        let header = self.module().new_block(func)?;
        let first = self.module().new_block(func)?;
        let cont = self.module().new_block(func)?;
        let merge = self.module().new_block(func)?;
        self.terminate(None, Terminator::Branch(header))?;
        self.enter(header)?;
        self.terminate(Some(Merge::Loop { merge, cont }), Terminator::Branch(first))?;
        self.module().loops.push(LoopMeta {
            header,
            merge,
            cont,
            preheader,
            exit_sel: spec.exit_sel,
            unrolling: spec.unrolling,
        });
        self.enter(first)?;
        self.cs.loops.push(OpenLoop { header, breaks: 0 });
        let lc = LoopCtx {
            func,
            header,
            merge,
            cont,
        };
        let r = body(self, ctx, &lc);
        let frame = self.cs.loops.pop();
        r?;
        let frame = frame
            .filter(|f| f.header == header)
            .ok_or(BuildError::Control(ControlRule::NotInnermostLoop))?;
        if self.current_block().is_some() {
            self.terminate(None, Terminator::Branch(cont))?;
        }
        self.enter(cont)?;
        self.terminate(None, Terminator::Branch(header))?;
        self.enter(merge)?;
        if frame.breaks == 0 {
            self.terminate(None, Terminator::Unreachable)?;
        }
        Ok(())
    }

    /// `OpDemoteToHelperInvocation` (fragment only).
    pub fn demote(&mut self) -> BResult<()> {
        self.emit_void(Inst::DemoteToHelperInvocation)
    }

    /// Terminate the open block with `OpKill` or `OpTerminateInvocation` (fragment only). When the
    /// entry declares an interlock mode, only the entry function may kill; `finalize` checks both
    /// once the entry point is known.
    pub fn kill(&mut self, form: KillForm) -> BResult<()> {
        let term = match form {
            KillForm::Kill => Terminator::Kill,
            KillForm::TerminateInvocation => Terminator::TerminateInvocation,
        };
        self.terminate(None, term)
    }

    // ---- Plumbing ----

    /// The open block, which is about to become a construct header.
    fn open_construct(&self) -> BResult<BlockId> {
        let b = self
            .current_block()
            .ok_or(BuildError::Control(ControlRule::NoCurrentBlock))?;
        Ok(b)
    }

    /// `exit_sel` must be a 32-bit integer `Function` variable of this function: a variable of
    /// another function is `ExitSelector`; an id that names no variable is dangling.
    fn check_exit_sel(&self, v: VarId) -> BResult<()> {
        let var = self
            .mb()
            .vars
            .get(v.index())
            .ok_or(BuildError::Dangling(Dangling::Var(v)))?;
        if var.func != self.id() {
            return Err(BuildError::Control(ControlRule::ExitSelector));
        }
        match self.mb().types().get(var.pointee)? {
            Type::Int { width: 32, .. } => Ok(()),
            _ => Err(BuildError::Control(ControlRule::ExitSelector)),
        }
    }

    /// Run arm `body` from fresh block `entry`; a fall-through branches to `merge`.
    fn arm<C>(
        &mut self,
        ctx: &mut C,
        entry: BlockId,
        merge: BlockId,
        body: ArmFn<'_, C>,
    ) -> BResult<Option<Fall>> {
        self.enter(entry)?;
        match (body(self, ctx)?, self.current_block()) {
            (ArmEnd::FallThrough(vals), Some(end)) => {
                self.terminate(None, Terminator::Branch(merge))?;
                Ok(Some((vals, end)))
            }
            (ArmEnd::Exits, None) => Ok(None),
            _ => Err(BuildError::Control(ControlRule::ArmEnd)),
        }
    }

    /// Open merge block `m` and merge the fall-through arms' values by `OpPhi`.
    fn merge_arms(&mut self, m: BlockId, falls: Vec<Fall>) -> BResult<Merged> {
        let Some((first, _)) = falls.first() else {
            self.enter(m)?;
            self.terminate(None, Terminator::Unreachable)?;
            return Ok(Merged(Vec::new()));
        };
        let tys = first
            .iter()
            .map(|v| self.value_ty(*v))
            .collect::<BResult<Vec<TypeId>>>()?;
        for (vals, _) in &falls {
            if vals.len() != tys.len() {
                return Err(BuildError::Control(ControlRule::ArmValues));
            }
            for (v, ty) in vals.iter().zip(&tys) {
                if self.value_ty(*v)? != *ty {
                    return Err(BuildError::Control(ControlRule::ArmValues));
                }
            }
        }
        for ty in &tys {
            let f = self.mb().types().facts(*ty)?;
            if !matches!(f.shape, Shape::Plain | Shape::Implicit | Shape::Explicit)
                || f.runtime_sized
            {
                return Err(BuildError::Control(ControlRule::PhiType));
            }
        }
        self.enter(m)?;
        let mut out = Vec::with_capacity(tys.len());
        for (i, ty) in tys.into_iter().enumerate() {
            let result = self.module().new_value(ty, ValueOrigin::Phi(m))?;
            let incoming = falls
                .iter()
                .map(|(vals, b)| {
                    vals.get(i)
                        .map(|v| (*v, *b))
                        .ok_or(BuildError::Control(ControlRule::ArmValues))
                })
                .collect::<BResult<Vec<_>>>()?;
            let incoming_cap = incoming.capacity();
            let phis = &mut self.block_mut(m)?.phis;
            let before = phis.capacity();
            phis.push(Phi { result, incoming });
            let after = phis.capacity();
            let module = self.module();
            module.count_block_growth::<Phi>(before, after);
            module.count_block_growth::<(Value, BlockId)>(0, incoming_cap);
            out.push(result);
        }
        Ok(Merged(out))
    }

    /// `break_` (`brk`) or `continue_` of loop `lc`.
    fn loop_exit(&mut self, lc: &LoopCtx, brk: bool) -> BResult<()> {
        let innermost = matches!(self.cs.loops.last(), Some(top) if top.header == lc.header);
        if lc.func != self.id() || !innermost {
            return Err(BuildError::Control(ControlRule::NotInnermostLoop));
        }
        let target = if brk { lc.merge } else { lc.cont };
        self.terminate(None, Terminator::Branch(target))?;
        if brk {
            if let Some(top) = self.cs.loops.last_mut() {
                top.breaks = top.breaks.saturating_add(1);
            }
        }
        Ok(())
    }
}

/// `OpKill`/`OpTerminateInvocation` need the fragment model, and only the entry function may
/// kill when the entry declares an interlock mode. Called by `finalize`.
pub(super) fn check_kills(blocks: &[BlockDraft], entry: &EntryPoint) -> BResult<()> {
    let interlock = entry.modes.iter().any(|m| m.is_interlock());
    for b in blocks {
        let op = match b.term {
            Some(Terminator::Kill) => crate::spv::Op::Kill,
            Some(Terminator::TerminateInvocation) => crate::spv::Op::TerminateInvocation,
            _ => continue,
        };
        if entry.model != ExecModel::Fragment {
            return Err(BuildError::Entry(EntryRule::InstructionForModel(op)));
        }
        if interlock && b.func != entry.func {
            return Err(BuildError::Control(ControlRule::KillOutsideEntry(b.func)));
        }
    }
    Ok(())
}

/// No tangled instruction sits inside a switch-at-use, directly or through a call. The builder
/// refuses direct ones as they are appended; this also covers calls. The arms of a switch are the
/// blocks laid out strictly between its header and its merge (preorder layout). Called by
/// `finalize`.
pub(super) fn check_switches_at_use(
    functions: &[Function],
    blocks: &[Block],
    headers: &[BlockId],
) -> BResult<()> {
    if headers.is_empty() {
        return Ok(());
    }
    let block = |b: BlockId| {
        blocks
            .get(b.index())
            .ok_or(BuildError::Dangling(Dangling::Block(b)))
    };
    let direct = |b: &Block| b.insts.iter().any(|n| n.inst.is_tangled());
    let callees = |b: &Block| -> Vec<FuncId> {
        b.insts
            .iter()
            .filter_map(|n| match &n.inst {
                Inst::Call { func, .. } => Some(*func),
                _ => None,
            })
            .collect()
    };

    // Functions that transitively execute a tangled instruction (a monotone fixpoint over calls).
    let mut tangled_fn = vec![false; functions.len()];
    let mut changed = true;
    while changed {
        changed = false;
        for (fi, f) in functions.iter().enumerate() {
            if tangled_fn.get(fi).copied().unwrap_or(true) {
                continue;
            }
            let mut t = false;
            for b in &f.blocks {
                let b = block(*b)?;
                t |= direct(b)
                    || callees(b)
                        .iter()
                        .any(|g| tangled_fn.get(g.index()).copied().unwrap_or(false));
            }
            if t {
                if let Some(x) = tangled_fn.get_mut(fi) {
                    *x = true;
                }
                changed = true;
            }
        }
    }

    for h in headers {
        let hb = block(*h)?;
        let Some(Merge::Selection { merge }) = hb.merge else {
            return Err(BuildError::Dangling(Dangling::Block(*h)));
        };
        let f = functions
            .get(hb.func.index())
            .ok_or(BuildError::Dangling(Dangling::Func(hb.func)))?;
        let at = |b: BlockId| {
            f.blocks
                .iter()
                .position(|x| *x == b)
                .ok_or(BuildError::Dangling(Dangling::Block(b)))
        };
        let (start, end) = (at(*h)?, at(merge)?);
        for b in f.blocks.get(start + 1..end).unwrap_or(&[]) {
            let b = block(*b)?;
            if direct(b) {
                return Err(BuildError::Control(ControlRule::TangledInSwitchAtUse));
            }
            if let Some(g) = callees(b)
                .into_iter()
                .find(|g| tangled_fn.get(g.index()).copied().unwrap_or(false))
            {
                return Err(BuildError::Control(ControlRule::TangledCallInSwitchAtUse(
                    g,
                )));
            }
        }
    }
    Ok(())
}

mod split;
#[cfg(test)]
mod tests;
