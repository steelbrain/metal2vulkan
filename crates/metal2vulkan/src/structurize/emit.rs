//! P8: emit the structured tree through the typed builder's constructs.
//!
//! Every tree statement maps to exactly one builder operation, so the structured-CFG rules hold by
//! construction of the builder and no block is ever duplicated:
//!
//! | Tree | Builder |
//! |---|---|
//! | `Seq` | the items in order, in the open block |
//! | `Code(b)` | [`CodeEmitter::block`] (the block's instructions, demotion loads/stores included) |
//! | `If(c, a, b)` | `if_` on [`CodeEmitter::cond`]; an empty `b` is the header's edge to the merge, both empty drop the `If` |
//! | `Switch` | `switch_` on [`CodeEmitter::selector`], one arm per target, the default arm last |
//! | `Dispatch(sel, arms)` | `switch_` on the loaded selector: case `i` is `arms[i]`, the last arm is the default (a single arm is inlined) |
//! | `Loop(body)` | `loop_`, whose `exit_sel` is the selector of the `Dispatch` that follows the loop |
//! | `Break` / `Continue` | `break_` / `continue_` of the innermost loop |
//! | `SetSel` / `SetPending` | a store of the constant |
//! | `ClearPending(Y)` | `pending = (pending == Y) ? 0 : pending`, branch-free |
//! | `GuardPending(s)` | `if_(pending == 0, s)` |
//! | `Return(v)` | [`CodeEmitter::on_return`] |
//! | `Unreachable` | `OpUnreachable` |
//!
//! Before anything is emitted, one metered pass over the arena checks what the builder cannot:
//! the tree is P7 output (no `Block`/`Jump`), selector indices are in range, switch literals fit
//! the 32-bit selector (wider ones are `Unsupported{NotImplemented}`), and the construct nesting
//! depth is within `Limit::ConstructNesting`. The builder nests constructs by closure scope, so
//! emission recurses once per nested construct; that cap bounds the stack (a `Seq` inside a `Seq`
//! is walked with an explicit stack and costs nothing).
//!
//! Variables: each structurizer selector `SelVar(i)`, `i < selectors`, is a `u32` `Function`
//! variable, `pending` among them. `pending` is set to `0` (no jump pending) in the entry block
//! before the first statement. Each demoted value and each phi gets a `Function` variable of the
//! type [`CodeEmitter::var_type`] gives its `DemoteTy`; the emitter receives them all through
//! [`CodeEmitter::bind`] and emits the loads and stores itself.

use crate::air::ir::{BlockId, ValueId};
use crate::budget::{Growth, Meter, CONSTRUCT_NESTING};
use crate::error::{Error, Feature, InternalError, InternalKind, Limit, Unsupported};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopCtx, LoopFn, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, ControlRule};
use crate::spirv::ids::{DemoteTy, TypeId, Value, VarId};
use crate::spirv::ir::{BinOp, ConstBits, Constant, MemAccess, Type};

use super::cfg::SelVar;
use super::tree::{Stmt, StmtId, StructTree, Structured};

/// What P8 needs from lowering. `lower::Emitter` implements it from the `Plan`; the value
/// environment is the emitter's own.
pub trait CodeEmitter {
    /// The SPIR-V type of a `Function` variable holding a value demoted as `ty` (the Plan's
    /// lowered representation).
    fn var_type(&mut self, ty: &DemoteTy, m: &mut ModuleBuilder) -> BResult<TypeId>;
    /// Receive the variables P8 allocated, before any block is emitted.
    fn bind(&mut self, vars: &StructVars) -> BResult<()>;
    /// Emit original block `b` without its terminator into the open block: its instructions, the
    /// loads of the demoted values it uses and phi variables it reads (once, at its top), the
    /// stores of the demoted values it defines, and the parallel copy into the incoming variables
    /// of its successors' phis. It may open builder-closed constructs and must leave a block open.
    fn block(&mut self, b: BlockId, fb: &mut FunctionBuilder<'_>) -> BResult<()>;
    /// The `bool` value of branch condition `v`, in the open block.
    fn cond(&mut self, v: ValueId, fb: &mut FunctionBuilder<'_>) -> BResult<Value>;
    /// A 32-bit integer for switch selector `v`, in the open block: for every case literal `l` of
    /// the switch, it equals `l` exactly when `v` does, and it equals no literal otherwise.
    fn selector(&mut self, v: ValueId, fb: &mut FunctionBuilder<'_>) -> BResult<Value>;
    /// Terminate the open block with the entry's return of `v` (predication's `if discarded
    /// { kill }` lives here).
    fn on_return(&mut self, v: Option<ValueId>, fb: &mut FunctionBuilder<'_>) -> BResult<()>;

    // The predicated walk ([`predicate`]). An emitter that does not predicate keeps the defaults,
    // and P8 walks its tree plainly.

    /// Whether the function is walked predicated: some block holds a workgroup synchronization
    /// outside workgroup-uniform control flow.
    fn predicated(&self) -> bool {
        false
    }

    /// The per-invocation `bool` the predicated walk starts `live` from, computed in the
    /// function's entry block: whether the invocation runs the body at all (an object function's
    /// partial threadgroup runs only its first invocations). `None`: every invocation does.
    fn entry_live(&self) -> Option<Value> {
        None
    }

    /// Whether block `b` holds a workgroup synchronization, which every invocation of the
    /// workgroup must execute together.
    fn syncs(&self, _b: BlockId) -> bool {
        false
    }

    /// Whether every invocation of the workgroup executes block `b` together.
    fn uniform_block(&self, _b: BlockId) -> bool {
        false
    }

    /// Whether branch condition or switch selector `v` has one value across the workgroup.
    fn uniform_cond(&self, _v: ValueId) -> bool {
        false
    }

    /// Emit block `b` like [`CodeEmitter::block`] where some invocations executing it may not be
    /// live (`live`, the loaded per-invocation flag): every instruction but a synchronization
    /// runs only where `live` holds, and each synchronization runs everywhere with `live` as the
    /// invocation's active flag (in a [`CodeEmitter::uniform_block`], where every invocation is
    /// live or none is, every invocation takes part as active). It leaves a block open.
    fn block_live(
        &mut self,
        b: BlockId,
        _live: Option<Value>,
        fb: &mut FunctionBuilder<'_>,
    ) -> BResult<()> {
        self.block(b, fb)
    }

    /// Whether `live` holds for some invocation of the workgroup: a workgroup-uniform `bool`,
    /// computed with workgroup barriers in workgroup-uniform control flow.
    fn vote(&mut self, _live: Value, _fb: &mut FunctionBuilder<'_>) -> BResult<Value> {
        Err(BuildError::Control(ControlRule::NoCurrentBlock))
    }

    /// Forget the values bound since the last block opened: the walk continues after a
    /// construct, where they do not dominate.
    fn forget(&mut self) {}
}

/// The `Function` variables P8 allocates for a [`Structured`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructVars {
    /// `SelVar(i)` is `selectors[i]` (`u32`).
    pub selectors: Vec<VarId>,
    /// The `pending` variable, one of `selectors`.
    pub pending: Option<VarId>,
    /// Each demoted value's variable, in the `Structured`'s order.
    pub demoted: Vec<(ValueId, VarId)>,
    /// Each phi's incoming variable, in the `Structured`'s order.
    pub phis: Vec<(ValueId, VarId)>,
    demoted_index: Vec<(ValueId, VarId)>,
    phi_index: Vec<(ValueId, VarId)>,
}

impl StructVars {
    /// The variable of demoted value `v`.
    pub fn demoted_var(&self, v: ValueId) -> Option<VarId> {
        lookup(&self.demoted_index, v)
    }

    /// The incoming variable of phi `v`.
    pub fn phi_var(&self, v: ValueId) -> Option<VarId> {
        lookup(&self.phi_index, v)
    }

    /// The variable of selector `sel`.
    pub fn selector(&self, sel: SelVar) -> Option<VarId> {
        self.selectors.get(sel.index()).copied()
    }
}

fn lookup(index: &[(ValueId, VarId)], v: ValueId) -> Option<VarId> {
    index
        .binary_search_by_key(&v, |(k, _)| *k)
        .ok()
        .and_then(|i| index.get(i))
        .map(|(_, var)| *var)
}

fn invariant(detail: impl Into<String>) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// Whether `s` opens a builder construct when emitted (a nesting level of the recursion).
fn is_construct(s: &Stmt) -> bool {
    match s {
        Stmt::If { .. } | Stmt::Switch { .. } | Stmt::Loop { .. } | Stmt::GuardPending(_) => true,
        Stmt::Dispatch { arms, .. } => arms.len() > 1,
        _ => false,
    }
}

/// The metered pre-pass: P7 output, selector ranges, literal widths, construct nesting depth.
fn check_tree(s: &Structured, meter: &mut Meter) -> Result<(), Error> {
    let tree = &s.tree;
    let sel_ok = |sel: &SelVar| {
        if sel.index() < s.selectors as usize {
            Ok(())
        } else {
            Err(invariant(format!("selector {sel:?} out of range")))
        }
    };
    if let Some(p) = &s.pending {
        sel_ok(p)?;
    }
    let bytes = tree.len().saturating_mul(std::mem::size_of::<u32>());
    meter.charge(bytes)?;
    let result = (|| {
        let mut depth = vec![0u32; tree.len()];
        for (id, st) in tree.stmts().rev() {
            meter.tick(1)?;
            match st {
                Stmt::Block { .. } | Stmt::Jump(_) => {
                    return Err(invariant(format!("{id:?}: P6 statement in a P7 tree")));
                }
                Stmt::SetSel { sel, .. } | Stmt::Dispatch { sel, .. } => sel_ok(sel)?,
                Stmt::SetPending(_) | Stmt::ClearPending(_) | Stmt::GuardPending(_)
                    if s.pending.is_none() =>
                {
                    return Err(invariant(format!(
                        "{id:?}: pending statement without pending"
                    )));
                }
                Stmt::Switch { cases, .. }
                    if cases
                        .iter()
                        .any(|(lits, _)| lits.iter().any(|&l| l > u64::from(u32::MAX))) =>
                {
                    return Err(Error::Unsupported(Unsupported::not_implemented(
                        Feature::WideSwitchLiteral,
                    )));
                }
                _ => {}
            }
            let d = depth.get(id.index()).copied().unwrap_or(0);
            let inner = d.saturating_add(u32::from(is_construct(st)));
            if u64::from(inner) > CONSTRUCT_NESTING {
                return Err(Error::LimitExceeded(Limit::ConstructNesting));
            }
            for c in st.children() {
                *depth
                    .get_mut(c.index())
                    .ok_or_else(|| invariant(format!("{c:?} out of range")))? = inner;
            }
        }
        Ok(())
    })();
    meter.release(bytes);
    result
}

/// P8: emit `s` into `fb`, whose entry block is open and empty, through `c`. On success no block
/// is open; the caller finishes the function. The builder's growth is charged to `meter` after
/// each block's code, so an oversized module is refused while it grows, and returned when the
/// walk ends: the caller owns the builder and charges what it holds. A refusal from the pre-pass
/// (`Limit:: ConstructNesting`, `Unsupported`) leaves `fb` untouched; a builder error aborts the
/// function (the builder is not rolled back).
pub fn emit(
    s: &Structured,
    fb: &mut FunctionBuilder<'_>,
    c: &mut impl CodeEmitter,
    meter: &mut Meter,
) -> Result<StructVars, Error> {
    check_tree(s, meter)?;
    let vars = allocate(s, fb, c)?;
    c.bind(&vars)?;
    let u32_ty = u32_type(fb.module())?;
    // The builder's growth during this walk is charged after each block's code and returned when
    // the walk ends: the builder is the caller's, who charges what it holds.
    let start = fb.module().footprint();
    let growth = Growth::from_charged(start);
    let mut cx = Cx {
        tree: &s.tree,
        c,
        vars: &vars,
        u32_ty,
        meter,
        growth,
        fault: None,
    };
    if let Some(p) = vars.pending {
        let zero = cx.konst(fb, 0)?;
        let ptr = fb.var_ptr(p)?;
        fb.store(ptr, zero, MemAccess::default())?;
    }
    let walked = if cx.c.predicated() {
        predicate::walk_root(&mut cx, fb)
    } else {
        list(&mut cx, fb, s.tree.root(), None)
            .and_then(|()| cx.poll(fb))
            .map_err(Error::from)
    };
    cx.meter.release(cx.growth.charged() - start);
    if let Some(fault) = cx.fault {
        return Err(fault);
    }
    walked?;
    if let Some(open) = fb.current_block() {
        return Err(BuildError::Control(ControlRule::UnterminatedBlock(open)).into());
    }
    Ok(vars)
}

fn u32_type(m: &mut ModuleBuilder) -> BResult<TypeId> {
    m.ty(Type::Int {
        width: 32,
        signed: false,
    })
}

/// Allocate every structurizer variable in the entry block.
fn allocate(
    s: &Structured,
    fb: &mut FunctionBuilder<'_>,
    c: &mut impl CodeEmitter,
) -> BResult<StructVars> {
    let u32_ty = u32_type(fb.module())?;
    let selectors = (0..s.selectors)
        .map(|_| fb.local_var(u32_ty, None))
        .collect::<BResult<Vec<_>>>()?;
    let pending = match s.pending {
        Some(p) => Some(
            *selectors
                .get(p.index())
                .ok_or(BuildError::Control(ControlRule::ExitSelector))?,
        ),
        None => None,
    };
    let mut var = |value: ValueId, ty: &DemoteTy| -> BResult<(ValueId, VarId)> {
        let ty = c.var_type(ty, fb.module())?;
        Ok((value, fb.local_var(ty, None)?))
    };
    let demoted = s
        .demoted
        .iter()
        .map(|d| var(d.value, &d.ty))
        .collect::<BResult<Vec<_>>>()?;
    let phis = s
        .phis
        .iter()
        .map(|p| var(p.phi, &p.ty))
        .collect::<BResult<Vec<_>>>()?;
    let sorted = |list: &[(ValueId, VarId)]| {
        let mut v = list.to_vec();
        v.sort_unstable_by_key(|(k, _)| *k);
        v
    };
    Ok(StructVars {
        demoted_index: sorted(&demoted),
        phi_index: sorted(&phis),
        selectors,
        pending,
        demoted,
        phis,
    })
}

/// Emission state threaded through the construct arms.
struct Cx<'e, E> {
    tree: &'e StructTree,
    c: &'e mut E,
    vars: &'e StructVars,
    u32_ty: TypeId,
    meter: &'e mut Meter,
    /// The builder bytes charged so far.
    growth: Growth,
    /// A meter refusal met inside the walk, which returns a builder error; `emit` reports this
    /// instead.
    fault: Option<Error>,
}

impl<E> Cx<'_, E> {
    /// Charge the builder's growth since the last poll to the meter.
    fn poll(&mut self, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
        let footprint = fb.module().footprint();
        self.growth.charge_to(footprint, self.meter).map_err(|e| {
            self.fault.get_or_insert(e.into());
            BuildError::Control(ControlRule::NoCurrentBlock)
        })
    }

    fn stmt(&self, id: StmtId) -> BResult<&Stmt> {
        self.tree
            .stmt(id)
            .map_err(|_| BuildError::Control(ControlRule::NoCurrentBlock))
    }

    fn konst(&self, fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
        let c = fb.module().constant(Constant::Scalar {
            ty: self.u32_ty,
            bits: ConstBits(u64::from(v)),
        })?;
        fb.module().const_value(c)
    }

    fn sel_ptr(&self, fb: &FunctionBuilder<'_>, sel: SelVar) -> BResult<Value> {
        let var = self
            .vars
            .selector(sel)
            .ok_or(BuildError::Control(ControlRule::ExitSelector))?;
        fb.var_ptr(var)
    }

    fn pending_ptr(&self, fb: &FunctionBuilder<'_>) -> BResult<Value> {
        let var = self
            .vars
            .pending
            .ok_or(BuildError::Control(ControlRule::ExitSelector))?;
        fb.var_ptr(var)
    }

    /// `pending == value` in the open block.
    fn pending_is(&self, fb: &mut FunctionBuilder<'_>, value: u32) -> BResult<(Value, Value)> {
        let ptr = self.pending_ptr(fb)?;
        let p = fb.load(ptr, MemAccess::default())?;
        let k = self.konst(fb, value)?;
        Ok((fb.binary(BinOp::IEqual, p, k)?, p))
    }
}

/// Whether `id` emits nothing (an empty `Seq`, possibly nested).
fn is_empty<E>(cx: &Cx<'_, E>, id: StmtId) -> BResult<bool> {
    let mut stack = vec![id];
    while let Some(s) = stack.pop() {
        match cx.stmt(s)? {
            Stmt::Seq(items) => stack.extend(items.iter().copied()),
            _ => return Ok(false),
        }
    }
    Ok(true)
}

/// An arm of a construct: emit `id`, then fall through if a block is still open.
fn arm<'a, 'e, E: CodeEmitter>(id: StmtId, lc: Option<&'a LoopCtx>) -> ArmFn<'a, Cx<'e, E>> {
    Box::new(move |fb, cx| {
        list(cx, fb, id, lc)?;
        Ok(if fb.current_block().is_some() {
            ArmEnd::FallThrough(Vec::new())
        } else {
            ArmEnd::Exits
        })
    })
}

/// Emit statement `id` (a sequence walked with an explicit stack) into the open block. `lc` is the
/// innermost enclosing loop.
///
/// This is the one frame on the recursion path that every nesting level pays for, so it only
/// walks the sequence; each statement is emitted by an out-of-line function, and only the
/// construct openers recurse (back into `list`, through the builder).
fn list<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    id: StmtId,
    lc: Option<&LoopCtx>,
) -> BResult<()> {
    let root = [id];
    let mut frames: Vec<&[StmtId]> = vec![&root];
    // Borrow the arena for the whole walk; `cx` stays usable for the arms.
    let tree = cx.tree;
    while let Some(frame) = frames.last_mut() {
        let Some((&id, rest)) = frame.split_first() else {
            frames.pop();
            continue;
        };
        *frame = rest;
        let next = frame_next(&frames);
        let st = tree
            .stmt(id)
            .map_err(|_| BuildError::Control(ControlRule::NoCurrentBlock))?;
        match st {
            Stmt::Seq(items) => frames.push(items),
            Stmt::Dispatch { arms, .. } if arms.len() == 1 && fb.current_block().is_some() => {
                // A one-arm dispatch always takes its arm: inline it.
                frames.push(arms);
            }
            _ if fb.current_block().is_none() => {
                // Every path before this statement ended (return, break, a loop without a
                // break): only statements that emit nothing may follow.
                if !matches!(st, Stmt::Unreachable) {
                    return Err(BuildError::Control(ControlRule::NoCurrentBlock));
                }
            }
            Stmt::If { .. } => emit_if(cx, fb, st, lc)?,
            Stmt::Switch { .. } => emit_switch(cx, fb, st, lc)?,
            Stmt::Dispatch { .. } => emit_dispatch(cx, fb, st, lc)?,
            Stmt::Loop { body } => emit_loop(cx, fb, *body, next)?,
            Stmt::GuardPending(body) => emit_guard(cx, fb, *body, lc)?,
            _ => simple(cx, fb, st, lc)?,
        }
    }
    Ok(())
}

/// A statement that opens no construct.
#[inline(never)]
fn simple<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    st: &Stmt,
    lc: Option<&LoopCtx>,
) -> BResult<()> {
    match st {
        Stmt::Code(b) => {
            cx.c.block(*b, fb)?;
            cx.poll(fb)?;
        }
        Stmt::SetSel { sel, value } => {
            let ptr = cx.sel_ptr(fb, *sel)?;
            let k = cx.konst(fb, *value)?;
            fb.store(ptr, k, MemAccess::default())?;
        }
        Stmt::Break => lc
            .ok_or(BuildError::Control(ControlRule::NotInnermostLoop))?
            .break_(fb)?,
        Stmt::Continue => lc
            .ok_or(BuildError::Control(ControlRule::NotInnermostLoop))?
            .continue_(fb)?,
        Stmt::SetPending(label) => {
            let ptr = cx.pending_ptr(fb)?;
            let k = cx.konst(fb, label.pending_value())?;
            fb.store(ptr, k, MemAccess::default())?;
        }
        Stmt::ClearPending(label) => {
            let (eq, p) = cx.pending_is(fb, label.pending_value())?;
            let zero = cx.konst(fb, 0)?;
            let cleared = fb.select(eq, zero, p)?;
            let ptr = cx.pending_ptr(fb)?;
            fb.store(ptr, cleared, MemAccess::default())?;
        }
        Stmt::Return(v) => {
            cx.c.on_return(*v, fb)?;
            if let Some(open) = fb.current_block() {
                return Err(BuildError::Control(ControlRule::UnterminatedBlock(open)));
            }
        }
        Stmt::Unreachable => fb.unreachable()?,
        // Constructs and sequences are `list`'s; `Block`/`Jump` never reach P8 (the pre-pass
        // refuses them).
        _ => return Err(BuildError::Control(ControlRule::NoCurrentBlock)),
    }
    Ok(())
}

#[inline(never)]
fn emit_if<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    st: &Stmt,
    lc: Option<&LoopCtx>,
) -> BResult<()> {
    let Stmt::If {
        cond,
        then,
        otherwise,
    } = st
    else {
        return Err(BuildError::Control(ControlRule::NoCurrentBlock));
    };
    let (te, oe) = (is_empty(cx, *then)?, is_empty(cx, *otherwise)?);
    if te && oe {
        return Ok(());
    }
    let c = cx.c.cond(*cond, fb)?;
    let f = if oe { None } else { Some(arm(*otherwise, lc)) };
    fb.if_(cx, c, arm(*then, lc), f)?;
    Ok(())
}

#[inline(never)]
fn emit_switch<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    st: &Stmt,
    lc: Option<&LoopCtx>,
) -> BResult<()> {
    let Stmt::Switch {
        sel,
        default,
        cases,
    } = st
    else {
        return Err(BuildError::Control(ControlRule::NoCurrentBlock));
    };
    let s = cx.c.selector(*sel, fb)?;
    let arms = cases
        .iter()
        .map(|(lits, a)| {
            let lits = lits
                .iter()
                .map(|&l| u32::try_from(l))
                .collect::<Result<Vec<u32>, _>>()
                .map_err(|_| BuildError::Control(ControlRule::CaseLiterals))?;
            Ok((lits, arm(*a, lc)))
        })
        .collect::<BResult<Vec<_>>>()?;
    fb.switch_(cx, s, arms, arm(*default, lc))?;
    Ok(())
}

#[inline(never)]
fn emit_dispatch<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    st: &Stmt,
    lc: Option<&LoopCtx>,
) -> BResult<()> {
    let Stmt::Dispatch { sel, arms } = st else {
        return Err(BuildError::Control(ControlRule::NoCurrentBlock));
    };
    let Some((default, cases)) = arms.split_last() else {
        return Err(BuildError::Control(ControlRule::CaseLiterals));
    };
    let ptr = cx.sel_ptr(fb, *sel)?;
    let s = fb.load(ptr, MemAccess::default())?;
    let cases = cases
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let lit =
                u32::try_from(i).map_err(|_| BuildError::Control(ControlRule::CaseLiterals))?;
            Ok((vec![lit], arm(*a, lc)))
        })
        .collect::<BResult<Vec<_>>>()?;
    fb.switch_(cx, s, cases, arm(*default, lc))?;
    Ok(())
}

/// A loop; `next` is the statement after it, whose exit dispatch names the loop's exit selector
/// (P5, P6 place it right after the loop).
#[inline(never)]
fn emit_loop<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    body: StmtId,
    next: Option<StmtId>,
) -> BResult<()> {
    let exit_sel = match next.map(|n| cx.tree.stmt(n)) {
        Some(Ok(Stmt::Dispatch { sel, arms })) if arms.len() > 1 => Some(
            cx.vars
                .selector(*sel)
                .ok_or(BuildError::Control(ControlRule::ExitSelector))?,
        ),
        _ => None,
    };
    let f: LoopFn<'_, Cx<'_, E>> = Box::new(move |fb, cx, lc| list(cx, fb, body, Some(lc)));
    fb.loop_(
        cx,
        LoopSpec {
            exit_sel,
            dont_unroll: false,
        },
        f,
    )
}

#[inline(never)]
fn emit_guard<E: CodeEmitter>(
    cx: &mut Cx<'_, E>,
    fb: &mut FunctionBuilder<'_>,
    body: StmtId,
    lc: Option<&LoopCtx>,
) -> BResult<()> {
    if is_empty(cx, body)? {
        return Ok(());
    }
    let (none, _) = cx.pending_is(fb, 0)?;
    fb.if_(cx, none, arm(body, lc), None)?;
    Ok(())
}

/// The next statement of the innermost open sequence.
fn frame_next(frames: &[&[StmtId]]) -> Option<StmtId> {
    frames.last().and_then(|f| f.first()).copied()
}

mod predicate;

#[cfg(test)]
mod tests;
