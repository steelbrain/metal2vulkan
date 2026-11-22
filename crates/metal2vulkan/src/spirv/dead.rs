//! Unused pure instructions leave the emitted module, IR→IR like P9.
//!
//! The lowering computes some values that no instruction reads in the end: the exact `f32` of a
//! narrow load whose every store wrote the loaded format bits instead, a lane of a composite that a
//! later lowering rebuilt, or a partial an emulated exchange never needed. A value is *dead* when
//! nothing reads it: no instruction, phi, terminator, or tangle entry, and no dead value's reader.
//! Its instruction is removed when it is [`removable`]: it computes a result from its operands and
//! nothing else (arithmetic, conversions, composites, `GLSL.std.450`, access chains, array lengths,
//! and plain loads), so running it or not changes no other value, no memory, and no other
//! invocation. A store, atomic, barrier, image write, subgroup operation, derivative,
//! implicitly-derived sample, interpolation, volatile load, or load that makes memory visible is
//! never removed, whatever reads its result.
//!
//! A call is removed the same way when its function is pure ([`pure_functions`]): read from the
//! function's body, which has no loop, returns on every path, and holds only removable instructions
//! and calls of pure functions, so a call computes its result and nothing else. A function that no
//! call left in the module reaches from the entry point leaves after the sweep, with its body, and
//! what only it read leaves with it ([`TypedModule::remove_dead`]). No function is named: a routine
//! the lowering defines once per module (a fast division, a precise one, a `half` fused
//! multiply-add) leaves when every division that called it was dead.
//!
//! A constant nothing reads leaves too: its declaration is dropped when it is [`removable_const`]
//! and no live instruction, phi, terminator, live composite constant, or variable initializer reads
//! it, and no instruction, array type, or execution mode names it as a constant operand. Such
//! constants are what the removed instructions read, and the constants a lowering made for a
//! sequence it then did not need.
//!
//! Work is linear in the module's instructions and declarations (each value is removed at most
//! once, each removal walks its own operands, and each function's body is read once for its
//! purity and once for what it calls), metered on the per-attempt [`Meter`] with every table
//! charged to its ledger.

use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind};
use crate::spirv::ids::{BlockId, ConstId, FuncId, TypeId, Value};
use crate::spirv::ir::{
    ConstDef, ConstKind, Constant, Decl, GlslOp, Inst, InstAt, Merge, Terminator, Type, TypedModule,
};

fn invariant(detail: impl Into<String>) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// Whether `inst` may be removed when nothing reads its result: it computes the result from its
/// operands and has no other effect.
pub fn removable(inst: &Inst) -> bool {
    match inst {
        Inst::Unary { .. }
        | Inst::Binary { .. }
        | Inst::Select { .. }
        | Inst::BitFieldInsert { .. }
        | Inst::BitFieldExtract { .. }
        | Inst::Convert { .. }
        | Inst::CompositeConstruct { .. }
        | Inst::CompositeExtract { .. }
        | Inst::CompositeInsert { .. }
        | Inst::VectorExtractDynamic { .. }
        | Inst::VectorInsertDynamic { .. }
        | Inst::VectorShuffle { .. }
        | Inst::Undef
        | Inst::AccessChain { .. }
        | Inst::ArrayLength { .. } => true,
        Inst::Glsl { op, .. } => !matches!(
            op,
            GlslOp::InterpolateAtCentroid
                | GlslOp::InterpolateAtSample
                | GlslOp::InterpolateAtOffset
        ),
        Inst::Load { access, .. } => !access.volatile && access.make_visible.is_none(),
        _ => false,
    }
}

/// Whether `inst` may be removed when nothing reads its result, given which functions are pure
/// (`pure`, indexed by function, from [`pure_functions`]): [`removable`] names it, or it calls a
/// pure function.
pub fn removable_with(inst: &Inst, pure: &[bool]) -> bool {
    match inst {
        Inst::Call { func, .. } => pure.get(func.index()).copied().unwrap_or(false),
        inst => removable(inst),
    }
}

/// Which functions of `m` are pure, indexed by function: a call of one computes its result from its
/// arguments and nothing else, so it may be removed when nothing reads it ([`removable_with`]).
/// Read from the body alone, with no marking and no names: no block of it declares a loop (so,
/// structured, it has no cycle and ends), every block ends in a branch, a switch, or a return, and
/// every instruction is one [`removable`] names or a call of a pure function. Vulkan forbids
/// recursion; a function found on its own call path is not pure. Each body is read once; the work
/// stack holds at most one entry per function.
pub fn pure_functions(m: &TypedModule) -> Vec<bool> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum State {
        Unread,
        Reading,
        Pure,
        Impure,
    }
    let n = m.functions().len();
    let mut state = vec![State::Unread; n];
    // (function, position in its blocks, instruction index): where its reading stopped.
    let mut stack: Vec<(usize, usize, usize)> = Vec::new();
    for root in 0..n {
        if state.get(root) != Some(&State::Unread) {
            continue;
        }
        if let Some(s) = state.get_mut(root) {
            *s = State::Reading;
        }
        stack.push((root, 0, 0));
        while let Some(&(f, pos, index)) = stack.last() {
            let mut verdict = None;
            let mut next = None;
            let blocks = m
                .functions()
                .get(f)
                .map_or(&[][..], |x| x.blocks.as_slice());
            match blocks.get(pos).and_then(|b| m.blocks().get(b.index())) {
                None if pos < blocks.len() => verdict = Some(State::Impure),
                None => verdict = Some(State::Pure),
                Some(blk) => {
                    let ends = matches!(
                        blk.term,
                        Terminator::Branch(_)
                            | Terminator::CondBranch { .. }
                            | Terminator::Switch { .. }
                            | Terminator::Return
                            | Terminator::ReturnValue(_)
                    );
                    if index == 0 && (matches!(blk.merge, Some(Merge::Loop { .. })) || !ends) {
                        verdict = Some(State::Impure);
                    }
                    let mut i = index;
                    while verdict.is_none() {
                        let Some(node) = blk.insts.get(i) else {
                            next = Some((pos + 1, 0));
                            break;
                        };
                        match &node.inst {
                            Inst::Call { func, .. } => match state.get(func.index()) {
                                Some(State::Pure) => i += 1,
                                Some(State::Unread) => {
                                    next = Some((pos, i));
                                    break;
                                }
                                _ => verdict = Some(State::Impure),
                            },
                            inst if removable(inst) => i += 1,
                            _ => verdict = Some(State::Impure),
                        }
                    }
                }
            }
            if let Some(v) = verdict {
                if let Some(s) = state.get_mut(f) {
                    *s = v;
                }
                stack.pop();
                continue;
            }
            let Some((pos, index)) = next else { continue };
            if let Some(top) = stack.last_mut() {
                *top = (f, pos, index);
            }
            // A call of an unread function: read the callee first, then come back to the call.
            let callee = m
                .functions()
                .get(f)
                .and_then(|x| x.blocks.get(pos))
                .and_then(|b| m.blocks().get(b.index()))
                .and_then(|b| b.insts.get(index))
                .and_then(|node| match node.inst {
                    Inst::Call { func, .. } => Some(func.index()),
                    _ => None,
                });
            if let Some(g) = callee.filter(|g| state.get(*g) == Some(&State::Unread)) {
                if let Some(s) = state.get_mut(g) {
                    *s = State::Reading;
                }
                stack.push((g, 0, 0));
            }
        }
    }
    state.into_iter().map(|s| s == State::Pure).collect()
}

/// Whether constant `def` may leave the module when nothing reads it: a plain scalar, composite, or
/// null. Specialization constants and the workgroup size always stay. A plain constant is read only
/// where [`remove_dead_keeping`] counts its readers: an SSA operand, a composite's or
/// specialization operation's part, a variable's initializer, an instruction's constant operand
/// ([`Inst::for_each_const`]), or an array type's length.
pub fn removable_const(def: &ConstDef) -> bool {
    matches!(def.kind, ConstKind::Plain(_))
}

/// What [`remove_dead_keeping`] removed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeadReport {
    /// Instructions removed.
    pub insts: u32,
    /// Phis removed.
    pub phis: u32,
    /// Constant declarations removed.
    pub consts: u32,
    /// Vector and function type declarations removed.
    pub types: u32,
    /// Instructions replaced by copies of offered values.
    pub copies: u32,
    /// Functions removed because no call left in the module reaches them.
    pub funcs: u32,
}

/// Charge `n` elements of `T` to the ledger.
fn charge<T>(meter: &mut Meter, ledger: &mut usize, n: usize) -> Result<(), Error> {
    let bytes = n.saturating_mul(std::mem::size_of::<T>());
    meter.charge(bytes)?;
    *ledger = ledger.saturating_add(bytes);
    Ok(())
}

/// Where a removable result is defined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Def {
    None,
    Inst(InstAt),
    Phi {
        block: BlockId,
        index: usize,
    },
    Const(ConstId),
    /// A removable instruction that becomes a copy of an offered value ([`Offer`]).
    Copy {
        at: InstAt,
        to: Value,
    },
}

/// Remove every dead removable instruction, dead phi, and dead removable constant of `m`, then
/// re-derive its interface and requirements. On error `m` is unchanged.
#[cfg(test)]
pub fn remove_dead(m: &mut TypedModule, meter: &mut Meter) -> Result<DeadReport, Error> {
    remove_dead_keeping(m, &[], meter)
}

/// Remove every dead removable instruction, dead phi, and dead removable constant of `m`, then
/// re-derive its interface and requirements, after weighing `offers`: each group's instructions
/// become copies of their offered values when what that frees outweighs what it keeps alive, in
/// words. On error `m` is unchanged.
pub fn remove_dead_keeping(
    m: &mut TypedModule,
    offers: &[Offer],
    meter: &mut Meter,
) -> Result<DeadReport, Error> {
    let mut ledger = 0usize;
    let r = plan(m, offers, meter, &mut ledger);
    meter.release(ledger);
    let Dead {
        insts,
        phis,
        consts,
        types,
        copies,
        funcs,
    } = r?;
    let report = DeadReport {
        insts: u32::try_from(insts.len()).unwrap_or(u32::MAX),
        phis: u32::try_from(phis.len()).unwrap_or(u32::MAX),
        consts: u32::try_from(consts.len()).unwrap_or(u32::MAX),
        types: u32::try_from(types.len()).unwrap_or(u32::MAX),
        copies: u32::try_from(copies.len()).unwrap_or(u32::MAX),
        funcs: u32::try_from(funcs.len()).unwrap_or(u32::MAX),
    };
    if report != DeadReport::default() {
        m.remove_dead(&insts, &phis, &consts, &types, &copies, &funcs)?;
    }
    Ok(report)
}

/// An equal value the lowering offers for a [`removable`] instruction's result: `to` holds what
/// `from` holds wherever `from` is, and is defined before `from`'s instruction in its block or in a
/// block that dominates it, so `from`'s instruction may become `OpCopyObject` of `to`. The lowering
/// offers what it can compute more than one way when only the module as a whole tells which way is
/// smaller; [`remove_dead_keeping`] decides. The offers of one `group` are taken or declined
/// together (they share what they keep alive).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Offer {
    pub group: u32,
    pub from: Value,
    pub to: Value,
}

/// The dead instructions, phis (block, index), and constants of a module, the instructions
/// that become copies of offered values, and the functions no call reaches. The instructions,
/// phis, and copies are all in functions that stay.
struct Dead {
    insts: Vec<InstAt>,
    phis: Vec<(BlockId, usize)>,
    /// Sorted.
    consts: Vec<ConstId>,
    types: Vec<TypeId>,
    copies: Vec<(InstAt, Value)>,
    funcs: Vec<FuncId>,
}

/// The dead instructions, phis, and constants of `m`, and the copies the offers weighed into
/// (read-only).
fn plan(
    m: &TypedModule,
    offers: &[Offer],
    meter: &mut Meter,
    ledger: &mut usize,
) -> Result<Dead, Error> {
    // Which functions are pure: read once, one state and at most one stack entry each.
    let nf = m.functions().len();
    charge::<(bool, [usize; 4])>(meter, ledger, nf)?;
    meter.tick(u32::try_from(m.blocks().len()).unwrap_or(u32::MAX))?;
    let pure = pure_functions(m);
    let n = m.values().len();
    charge::<u32>(meter, ledger, n)?;
    let mut uses = vec![0u32; n];
    charge::<Def>(meter, ledger, n)?;
    let mut defs = vec![Def::None; n];
    let count = |uses: &mut Vec<u32>, v: Value| {
        if let Some(u) = uses.get_mut(v.index()) {
            *u = u.saturating_add(1);
        }
    };
    for (bi, blk) in m.blocks().iter().enumerate() {
        let block = BlockId::new(u32::try_from(bi).map_err(|_| invariant("block count"))?);
        meter.tick(
            u32::try_from(blk.insts.len().saturating_add(blk.phis.len()))
                .unwrap_or(u32::MAX)
                .saturating_add(1),
        )?;
        for (index, phi) in blk.phis.iter().enumerate() {
            for (v, _) in &phi.incoming {
                count(&mut uses, *v);
            }
            if let Some(d) = defs.get_mut(phi.result.index()) {
                *d = Def::Phi { block, index };
            }
        }
        for (index, node) in blk.insts.iter().enumerate() {
            node.inst.for_each_value(|v| count(&mut uses, v));
            // A constant operand is read for as long as the module lasts: no removable
            // instruction carries one, and a removal releases only SSA operands.
            let mut named = Ok(());
            node.inst.for_each_const(|c| match m.const_value(c) {
                Ok(v) => count(&mut uses, v),
                Err(e) => named = Err(e),
            });
            named?;
            if let Some(r) = node.result.filter(|_| removable_with(&node.inst, &pure)) {
                if let Some(d) = defs.get_mut(r.index()) {
                    *d = Def::Inst(InstAt { block, index });
                }
            }
        }
        blk.term.for_each_value(|v| count(&mut uses, v));
    }
    // Module-level readers of constants: a declared composite's parts (released when the
    // composite leaves), and variable initializers.
    meter.tick(u32::try_from(m.decls().len()).unwrap_or(u32::MAX))?;
    for d in m.decls() {
        let Decl::Const(c) = d else { continue };
        let def = m.consts().get(*c)?;
        match &def.kind {
            ConstKind::Plain(Constant::Composite { parts, .. }) => {
                for p in parts.iter() {
                    count(&mut uses, m.const_value(*p)?);
                }
            }
            ConstKind::WorkgroupSize { parts } => {
                for p in parts {
                    count(&mut uses, m.const_value(*p)?);
                }
            }
            ConstKind::SpecOp { operands, .. } => {
                for p in operands.iter() {
                    count(&mut uses, m.const_value(*p)?);
                }
            }
            ConstKind::Plain(_) | ConstKind::Spec { .. } => {}
        }
        if removable_const(def) {
            let v = m.const_value(*c)?;
            if let Some(d) = defs.get_mut(v.index()) {
                *d = Def::Const(*c);
            }
        }
    }
    let inits = m.globals().iter().filter_map(|g| g.init);
    for c in inits.chain(m.vars().iter().filter_map(|v| v.init)) {
        count(&mut uses, m.const_value(c)?);
    }
    // Array lengths (of every interned type: a type stays interned when its declaration leaves).
    meter.tick(u32::try_from(m.types().len()).unwrap_or(u32::MAX))?;
    for (_, t) in m.types().iter() {
        if let Type::Array { len, .. } = t {
            count(&mut uses, m.const_value(*len)?);
        }
    }
    charge::<bool>(meter, ledger, n)?;
    let live = defs.iter().map(|d| *d != Def::None).collect();
    let mut sim = Sim {
        m,
        uses,
        defs,
        live,
        log: Vec::new(),
        logging: false,
        stack: Vec::new(),
        steps: 0,
    };
    // Everything nothing reads, transitively.
    for i in 0..n {
        let v = Value::new(u32::try_from(i).map_err(|_| invariant("value count"))?);
        if sim.uses.get(i).copied() == Some(0) && sim.live(v) {
            sim.kill(v, meter)?;
        }
    }
    // The functions no call that stays reaches leave, and so does what only they read.
    charge::<bool>(meter, ledger, nf)?;
    let gone = sim.unreached(meter, ledger)?;
    sim.release(&gone, meter)?;
    let copies = sim.weigh(offers, meter, ledger)?;
    let in_gone = |b: BlockId| {
        m.blocks()
            .get(b.index())
            .is_some_and(|blk| gone.get(blk.func.index()).copied().unwrap_or(false))
    };
    let mut insts = Vec::new();
    let mut phis = Vec::new();
    let mut consts = Vec::new();
    for (i, d) in sim.defs.iter().enumerate() {
        if sim.live.get(i).copied().unwrap_or(true) {
            continue;
        }
        meter.tick(1)?;
        match *d {
            Def::None => {}
            Def::Inst(at) | Def::Copy { at, .. } => {
                if !in_gone(at.block) {
                    charge::<InstAt>(meter, ledger, 1)?;
                    insts.push(at);
                }
            }
            Def::Phi { block, index } => {
                if !in_gone(block) {
                    charge::<(BlockId, usize)>(meter, ledger, 1)?;
                    phis.push((block, index));
                }
            }
            Def::Const(c) => {
                charge::<ConstId>(meter, ledger, 1)?;
                consts.push(c);
            }
        }
    }
    consts.sort_unstable();
    let types = unread_types(&sim, &consts, &gone, meter, ledger)?;
    let mut funcs = Vec::new();
    for (fi, g) in gone.iter().enumerate() {
        if *g {
            charge::<FuncId>(meter, ledger, 1)?;
            funcs.push(FuncId::new(
                u32::try_from(fi).map_err(|_| invariant("function count"))?,
            ));
        }
    }
    Ok(Dead {
        insts,
        phis,
        consts,
        types,
        copies,
        funcs,
    })
}

/// The declared vector and function types nothing that stays in the module refers to: no kept
/// instruction or phi result, signature or parameter of a function that stays, variable of one,
/// global, push member, kept constant, declared type other than a vector or a function type (each
/// of which stays), or vector or function type that is itself referred to. A vector type is
/// declared for the values of some sequence, and a function type for a function's signature; when
/// they all leave, it does.
fn unread_types(
    sim: &Sim<'_>,
    dead_consts: &[ConstId],
    gone: &[bool],
    meter: &mut Meter,
    ledger: &mut usize,
) -> Result<Vec<TypeId>, Error> {
    let m = sim.m;
    let n = m.types().len();
    charge::<bool>(meter, ledger, n)?;
    let mut used = vec![false; n];
    let mut stack: Vec<TypeId> = Vec::new();
    let mut mark = |t: TypeId, stack: &mut Vec<TypeId>| {
        if let Some(u) = used.get_mut(t.index()) {
            if !*u {
                *u = true;
                stack.push(t);
            }
        }
    };
    let kept = |v: Value| sim.def(v) == Def::None || sim.live(v);
    let stays = |f: FuncId| !gone.get(f.index()).copied().unwrap_or(false);
    for blk in m.blocks().iter().filter(|b| stays(b.func)) {
        meter.tick(u32::try_from(blk.insts.len()).unwrap_or(u32::MAX))?;
        for phi in &blk.phis {
            if kept(phi.result) {
                mark(m.value(phi.result)?.ty, &mut stack);
            }
        }
        for node in &blk.insts {
            if let Some(r) = node.result.filter(|r| kept(*r)) {
                mark(m.value(r)?.ty, &mut stack);
            }
        }
    }
    for (fi, f) in m.functions().iter().enumerate() {
        if gone.get(fi).copied().unwrap_or(false) {
            continue;
        }
        mark(f.ret, &mut stack);
        mark(f.fn_ty, &mut stack);
        for p in &f.params {
            mark(m.value(*p)?.ty, &mut stack);
        }
    }
    for g in m.globals() {
        mark(g.ptr_ty, &mut stack);
        mark(g.pointee, &mut stack);
    }
    for v in m.vars().iter().filter(|v| stays(v.func)) {
        mark(v.ptr_ty, &mut stack);
        mark(v.pointee, &mut stack);
    }
    for p in m.push_members() {
        mark(p.ty, &mut stack);
    }
    for d in m.decls() {
        match d {
            Decl::Const(c) if dead_consts.binary_search(c).is_err() => {
                mark(m.consts().get(*c)?.ty, &mut stack);
            }
            // Only vector and function types leave, so every other declared type stays, with
            // what it refers to.
            Decl::Type(t) if !leaves(m.types().get(*t)?) => {
                mark(*t, &mut stack);
            }
            _ => {}
        }
    }
    charge::<TypeId>(meter, ledger, n)?;
    while let Some(t) = stack.pop() {
        meter.tick(1)?;
        match m.types().get(t)? {
            Type::Vector { elem, .. }
            | Type::Array { elem, .. }
            | Type::RuntimeArray { elem, .. } => mark(*elem, &mut stack),
            Type::Matrix { col, .. } => mark(*col, &mut stack),
            Type::Struct { members, .. } => {
                for x in members.iter() {
                    mark(*x, &mut stack);
                }
            }
            Type::Pointer { pointee, .. } => mark(*pointee, &mut stack),
            Type::Function { ret, params } => {
                mark(*ret, &mut stack);
                for x in params.iter() {
                    mark(*x, &mut stack);
                }
            }
            Type::Image(img) => mark(img.sampled_type, &mut stack),
            Type::SampledImage { image } => mark(*image, &mut stack),
            _ => {}
        }
    }
    let mut out = Vec::new();
    for d in m.decls() {
        if let Decl::Type(t) = d {
            if !used.get(t.index()).copied().unwrap_or(true) && leaves(m.types().get(*t)?) {
                charge::<TypeId>(meter, ledger, 1)?;
                out.push(*t);
            }
        }
    }
    Ok(out)
}

/// Whether a declared type of kind `t` may leave when nothing that stays refers to it: a vector
/// or a function type.
fn leaves(t: &Type) -> bool {
    matches!(t, Type::Vector { .. } | Type::Function { .. })
}

/// The most removal or revival steps one offer group's trial may take: a bound on the work of
/// weighing, whatever the module; a group whose trial would take more is declined.
const MAX_TRIAL_STEPS: u32 = 1 << 14;

/// One change a trial made, to undo it.
enum Undo {
    Uses(Value, u32),
    Live(Value, bool),
    Def(Value, Def),
}

/// The use counts, definitions, and liveness of a module's values while dead code is found and
/// offers are weighed.
struct Sim<'m> {
    m: &'m TypedModule,
    uses: Vec<u32>,
    defs: Vec<Def>,
    /// Whether each value with a definition is still read (directly or through live values).
    live: Vec<bool>,
    log: Vec<Undo>,
    logging: bool,
    stack: Vec<Value>,
    /// Steps of the trial in progress.
    steps: u32,
}

impl Sim<'_> {
    fn def(&self, v: Value) -> Def {
        self.defs.get(v.index()).copied().unwrap_or(Def::None)
    }

    /// Whether `v` has a removable definition that is live.
    fn live(&self, v: Value) -> bool {
        self.def(v) != Def::None && self.live.get(v.index()).copied().unwrap_or(false)
    }

    fn set_live(&mut self, v: Value, to: bool) {
        if let Some(l) = self.live.get_mut(v.index()) {
            if self.logging {
                self.log.push(Undo::Live(v, *l));
            }
            *l = to;
        }
    }

    fn add_use(&mut self, v: Value, up: bool) -> u32 {
        match self.uses.get_mut(v.index()) {
            Some(u) => {
                if self.logging {
                    self.log.push(Undo::Uses(v, *u));
                }
                *u = if up {
                    u.saturating_add(1)
                } else {
                    u.saturating_sub(1)
                };
                *u
            }
            None => 1,
        }
    }

    /// Call `f` on every value definition `d` reads.
    fn operands(&self, d: Def, mut f: impl FnMut(Value)) -> Result<(), Error> {
        let m = self.m;
        match d {
            Def::None => {}
            Def::Inst(at) => m
                .blocks()
                .get(at.block.index())
                .and_then(|b| b.insts.get(at.index))
                .ok_or_else(|| invariant("a dead instruction out of range"))?
                .inst
                .for_each_value(f),
            Def::Copy { to, .. } => f(to),
            Def::Phi { block, index } => {
                let phi = m
                    .blocks()
                    .get(block.index())
                    .and_then(|b| b.phis.get(index))
                    .ok_or_else(|| invariant("a dead phi out of range"))?;
                for (o, _) in &phi.incoming {
                    f(*o);
                }
            }
            Def::Const(c) => {
                if let ConstKind::Plain(Constant::Composite { parts, .. }) =
                    &m.consts().get(c)?.kind
                {
                    for p in parts.iter() {
                        f(m.const_value(*p)?);
                    }
                }
            }
        }
        Ok(())
    }

    /// The words definition `d` takes in the module, as the serializer writes it (operands that
    /// are literals counted as their words).
    fn words(&self, d: Def) -> Result<u32, Error> {
        let mut n = 0u32;
        self.operands(d, |_| n += 1)?;
        let extra = match d {
            Def::Inst(at) => match self
                .m
                .blocks()
                .get(at.block.index())
                .and_then(|b| b.insts.get(at.index))
                .map(|x| &x.inst)
            {
                Some(Inst::CompositeExtract { indices, .. })
                | Some(Inst::CompositeInsert { indices, .. }) => indices.len(),
                Some(Inst::VectorShuffle { components, .. }) => components.len(),
                // The extended instruction set and the instruction number.
                Some(Inst::Glsl { .. }) => 2,
                // The function.
                Some(Inst::Call { .. }) => 1,
                _ => 0,
            },
            // A phi's incoming pairs: each value with its predecessor block.
            Def::Phi { .. } => n as usize,
            // A scalar's literal (64-bit scalars are rare enough to count as one word).
            Def::Const(c) => match self.m.consts().get(c)?.kind {
                ConstKind::Plain(Constant::Scalar { .. }) => 1,
                _ => 0,
            },
            Def::Copy { .. } | Def::None => 0,
        };
        Ok(3u32
            .saturating_add(n)
            .saturating_add(u32::try_from(extra).unwrap_or(u32::MAX)))
    }

    fn step(&mut self, meter: &mut Meter) -> Result<bool, Error> {
        meter.tick(1)?;
        self.steps = self.steps.saturating_add(1);
        Ok(!self.logging || self.steps <= MAX_TRIAL_STEPS)
    }

    /// Remove `v` if nothing reads it, then whatever only it read; the words removed.
    fn kill(&mut self, v: Value, meter: &mut Meter) -> Result<Option<u64>, Error> {
        let mut words = 0u64;
        self.stack.push(v);
        while let Some(v) = self.stack.pop() {
            if !self.step(meter)? {
                self.stack.clear();
                return Ok(None);
            }
            if !self.live(v) || self.uses.get(v.index()).copied().unwrap_or(1) != 0 {
                continue;
            }
            let d = self.def(v);
            self.set_live(v, false);
            words = words.saturating_add(u64::from(self.words(d)?));
            let mut reads = Vec::new();
            self.operands(d, |o| reads.push(o))?;
            for o in reads {
                if self.add_use(o, false) == 0 && self.live(o) {
                    self.stack.push(o);
                }
            }
        }
        Ok(Some(words))
    }

    /// Which functions no call that stays reaches from the entry point, indexed by function: a
    /// call stays unless it is a dead removable one, and only the calls of reached functions
    /// count.
    fn unreached(&self, meter: &mut Meter, ledger: &mut usize) -> Result<Vec<bool>, Error> {
        let m = self.m;
        let nf = m.functions().len();
        let mut reached = vec![false; nf];
        charge::<FuncId>(meter, ledger, nf)?;
        let mut stack = vec![m.entry().func];
        if let Some(r) = reached.get_mut(m.entry().func.index()) {
            *r = true;
        }
        while let Some(f) = stack.pop() {
            let blocks = m
                .functions()
                .get(f.index())
                .ok_or_else(|| invariant("a reached function out of range"))?;
            for b in &blocks.blocks {
                let blk = m
                    .blocks()
                    .get(b.index())
                    .ok_or_else(|| invariant("a function's block out of range"))?;
                meter.tick(u32::try_from(blk.insts.len()).unwrap_or(u32::MAX))?;
                for node in &blk.insts {
                    let Inst::Call { func, .. } = node.inst else {
                        continue;
                    };
                    let stays = node
                        .result
                        .is_none_or(|r| self.def(r) == Def::None || self.live(r));
                    if let Some(r) = reached.get_mut(func.index()).filter(|r| stays && !**r) {
                        *r = true;
                        stack.push(func);
                    }
                }
            }
        }
        Ok(reached.into_iter().map(|r| !r).collect())
    }

    /// Release what the bodies of the functions `gone` read, as their removal does: each value a
    /// live instruction, phi, or terminator of theirs reads loses that reader, each constant
    /// operand and variable initializer of theirs loses one too, and whatever nothing reads then
    /// is removed. Their own removable values are marked removed as they are released, so each
    /// reader is released once (by [`Sim::kill`] or here). Only constants are read across
    /// functions, so what leaves outside them is constants.
    fn release(&mut self, gone: &[bool], meter: &mut Meter) -> Result<(), Error> {
        let m = self.m;
        if !gone.contains(&true) {
            return Ok(());
        }
        let mut reads = Vec::new();
        for (fi, f) in m.functions().iter().enumerate() {
            if !gone.get(fi).copied().unwrap_or(false) {
                continue;
            }
            for b in &f.blocks {
                let blk = m
                    .blocks()
                    .get(b.index())
                    .ok_or_else(|| invariant("a function's block out of range"))?;
                meter.tick(
                    u32::try_from(blk.insts.len().saturating_add(blk.phis.len()))
                        .unwrap_or(u32::MAX)
                        .saturating_add(1),
                )?;
                for phi in &blk.phis {
                    if self.live(phi.result) {
                        self.set_live(phi.result, false);
                        reads.extend(phi.incoming.iter().map(|(v, _)| *v));
                    }
                }
                for node in &blk.insts {
                    let own = node.result.filter(|r| self.def(*r) != Def::None);
                    match own {
                        Some(r) if !self.live(r) => {}
                        Some(r) => {
                            self.set_live(r, false);
                            node.inst.for_each_value(|v| reads.push(v));
                        }
                        None => node.inst.for_each_value(|v| reads.push(v)),
                    }
                    let mut named = Ok(());
                    node.inst.for_each_const(|c| match m.const_value(c) {
                        Ok(v) => reads.push(v),
                        Err(e) => named = Err(e),
                    });
                    named?;
                }
                blk.term.for_each_value(|v| reads.push(v));
                self.drop_reads(&mut reads, meter)?;
            }
            for v in &f.vars {
                let var = m
                    .vars()
                    .get(v.index())
                    .ok_or_else(|| invariant("a function's variable out of range"))?;
                if let Some(c) = var.init {
                    reads.push(m.const_value(c)?);
                }
            }
            self.drop_reads(&mut reads, meter)?;
        }
        Ok(())
    }

    /// Take one reader from each value of `reads` (emptied), removing each one nothing reads.
    fn drop_reads(&mut self, reads: &mut Vec<Value>, meter: &mut Meter) -> Result<(), Error> {
        while let Some(o) = reads.pop() {
            if self.add_use(o, false) == 0 && self.live(o) {
                self.kill(o, meter)?;
            }
        }
        Ok(())
    }

    /// Make dead `v` live again, with whatever dead value it reads; the words revived.
    fn revive(&mut self, v: Value, meter: &mut Meter) -> Result<Option<u64>, Error> {
        let mut words = 0u64;
        self.stack.push(v);
        while let Some(v) = self.stack.pop() {
            if !self.step(meter)? {
                self.stack.clear();
                return Ok(None);
            }
            if self.def(v) == Def::None || self.live(v) {
                continue;
            }
            let d = self.def(v);
            self.set_live(v, true);
            words = words.saturating_add(u64::from(self.words(d)?));
            let mut reads = Vec::new();
            self.operands(d, |o| reads.push(o))?;
            for o in reads {
                self.add_use(o, true);
                if self.def(o) != Def::None && !self.live(o) {
                    self.stack.push(o);
                }
            }
        }
        Ok(Some(words))
    }

    /// Weigh each offer group, latest first: its offers' instructions become copies of the
    /// offered values when the words that frees exceed the words the offered values revive. The
    /// copies kept, by instruction.
    fn weigh(
        &mut self,
        offers: &[Offer],
        meter: &mut Meter,
        ledger: &mut usize,
    ) -> Result<Vec<(InstAt, Value)>, Error> {
        let mut order: Vec<&Offer> = offers.iter().collect();
        charge::<&Offer>(meter, ledger, order.len())?;
        order.sort_by_key(|o| std::cmp::Reverse(o.group));
        let mut kept: Vec<(Value, InstAt, Value)> = Vec::new();
        // The undo log and the work stack, charged as they grow.
        let mut held = 0usize;
        let mut i = 0;
        while let Some(first) = order.get(i) {
            let group = first.group;
            let n = order
                .get(i..)
                .unwrap_or(&[])
                .iter()
                .take_while(|o| o.group == group)
                .count();
            let members: Vec<Offer> = order
                .get(i..i + n)
                .unwrap_or(&[])
                .iter()
                .map(|o| **o)
                .collect();
            i += n;
            self.log.clear();
            self.logging = true;
            self.steps = 0;

            let mut gain: i64 = 0;
            let mut made = Vec::new();
            let mut declined = false;
            for o in members {
                let Def::Inst(at) = self.def(o.from) else {
                    continue;
                };
                // Only an instruction `removable` names becomes a copy, never a call.
                let copyable = self
                    .m
                    .blocks()
                    .get(at.block.index())
                    .and_then(|b| b.insts.get(at.index))
                    .is_some_and(|node| removable(&node.inst));
                if !copyable || !self.live(o.from) || o.from == o.to {
                    continue;
                }
                let before = self.words(Def::Inst(at))?;
                // `from` becomes a copy of `to`: `to` gains a reader, `from`'s operands lose one.
                self.add_use(o.to, true);
                let revived = if self.def(o.to) != Def::None && !self.live(o.to) {
                    self.revive(o.to, meter)?
                } else {
                    Some(0)
                };
                let mut reads = Vec::new();
                self.operands(Def::Inst(at), |x| reads.push(x))?;
                if let Some(d) = self.defs.get_mut(o.from.index()) {
                    self.log.push(Undo::Def(o.from, *d));
                    *d = Def::Copy { at, to: o.to };
                }
                let mut freed = Some(0u64);
                for x in reads {
                    if self.add_use(x, false) == 0 && self.live(x) {
                        freed = match (freed, self.kill(x, meter)?) {
                            (Some(a), Some(b)) => Some(a.saturating_add(b)),
                            _ => None,
                        };
                    }
                }
                let (Some(revived), Some(freed)) = (revived, freed) else {
                    declined = true;
                    break;
                };
                let copy = self.words(Def::Copy { at, to: o.to })?;
                gain = gain
                    .saturating_add(i64::try_from(freed).unwrap_or(i64::MAX))
                    .saturating_add(i64::from(before))
                    .saturating_sub(i64::try_from(revived).unwrap_or(i64::MAX))
                    .saturating_sub(i64::from(copy));
                made.push((o.from, at, o.to));
            }
            self.logging = false;
            if declined || gain <= 0 {
                while let Some(u) = self.log.pop() {
                    match u {
                        Undo::Uses(v, x) => {
                            if let Some(u) = self.uses.get_mut(v.index()) {
                                *u = x;
                            }
                        }
                        Undo::Live(v, x) => {
                            if let Some(l) = self.live.get_mut(v.index()) {
                                *l = x;
                            }
                        }
                        Undo::Def(v, d) => {
                            if let Some(x) = self.defs.get_mut(v.index()) {
                                *x = d;
                            }
                        }
                    }
                }
            } else {
                charge::<(Value, InstAt, Value)>(meter, ledger, made.len())?;
                kept.extend(made);
            }
            let now = self
                .log
                .capacity()
                .saturating_mul(std::mem::size_of::<Undo>())
                .saturating_add(
                    self.stack
                        .capacity()
                        .saturating_mul(std::mem::size_of::<Value>()),
                );
            if now > held {
                charge::<u8>(meter, ledger, now - held)?;
                held = now;
            }
            self.log.clear();
        }
        charge::<(InstAt, Value)>(meter, ledger, kept.len())?;
        Ok(kept
            .into_iter()
            .filter(|(from, _, _)| self.live(*from))
            .map(|(_, at, to)| (at, to))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
    use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
    use crate::spirv::error::BResult;
    use crate::spirv::ids::GlobalId;
    use crate::spirv::interp::{run, Buffer, Config, Resources};
    use crate::spirv::ir::{
        natural_layout, BinOp, Contraction, ExecMode, ExecModel, LayoutRule, MemAccess,
        MemoryOrder, Scope, Semantics, StorageClass,
    };
    use crate::spirv::testing::{assert_valid, committed_profiles};

    fn slot(index: u32) -> ResourceSlot {
        ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        }
    }

    fn buffer(b: &mut ModuleBuilder) -> BResult<GlobalId> {
        let u32_ = b.ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let rta = b.ty(Type::RuntimeArray {
            elem: u32_,
            stride: Some(4),
        })?;
        let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
        let block = b.ty(Type::Struct {
            members: [rta].into(),
            layout: Some(layout),
        })?;
        b.resource(slot(0), StorageClass::StorageBuffer, block)
    }

    fn k(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
        let c = fb.u32_const(v)?;
        fb.module().const_value(c)
    }

    fn word(fb: &mut FunctionBuilder<'_>, buf: GlobalId, i: u32) -> BResult<Value> {
        let base = fb.module().global_value(buf)?;
        let (zero, i) = (k(fb, 0)?, k(fb, i)?);
        fb.access_chain(base, &[zero, i])
    }

    /// A kernel that loads word 0, keeps a dead chain of arithmetic on it, an unused `undef`,
    /// and an unused volatile load of word 2, and stores `w0 + 5` to word 1.
    fn kernel() -> BResult<TypedModule> {
        let mut b = ModuleBuilder::new();
        let buf = buffer(&mut b)?;
        let void = b.ty(Type::Void)?;
        let u32_ = b.ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let mut fb = b.function(void)?;
        let p0 = word(&mut fb, buf, 0)?;
        let w0 = fb.load(p0, MemAccess::default())?;
        let one = k(&mut fb, 1)?;
        let dead = fb.binary(BinOp::IAdd, w0, one)?;
        let two = k(&mut fb, 2)?;
        fb.binary(BinOp::IMul, dead, two)?;
        fb.undef(u32_)?;
        let p2 = word(&mut fb, buf, 2)?;
        fb.load(
            p2,
            MemAccess {
                volatile: true,
                ..MemAccess::default()
            },
        )?;
        let five = k(&mut fb, 5)?;
        let r = fb.binary(BinOp::IAdd, w0, five)?;
        let p1 = word(&mut fb, buf, 1)?;
        fb.store(p1, r, MemAccess::default())?;
        fb.ret(None)?;
        let id = fb.finish()?;
        b.entry(
            ExecModel::GLCompute,
            id,
            vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
        )?;
        b.finalize()
    }

    fn insts(m: &TypedModule) -> usize {
        m.blocks().iter().map(|b| b.insts.len()).sum()
    }

    /// The dead arithmetic chain (two instructions) and the unused `undef` leave; the volatile load
    /// (and the access chain it reads) stays; the result is valid and computes what it computed.
    #[test]
    fn dead_pure_instructions_leave_and_effects_stay() {
        let mut m = kernel().expect("build");
        let before = insts(&m);
        let report =
            remove_dead(&mut m, &mut Meter::new(&crate::budget::Budget::default())).expect("dce");
        assert_eq!(report.insts, 3, "{report:?}");
        assert_eq!(insts(&m), before - 3);
        assert!(m
            .blocks()
            .iter()
            .flat_map(|b| &b.insts)
            .any(|n| matches!(n.inst, Inst::Load { access, .. } if access.volatile)));
        let profile = committed_profiles()
            .expect("profiles")
            .into_iter()
            .find(|(s, _)| s == "moltenvk-m3")
            .expect("moltenvk-m3")
            .1;
        assert_valid(m.clone(), &profile);
        let res = Resources {
            buffers: vec![Buffer {
                slot: Some(slot(0)),
                address: None,
                bytes: [7u32, 0, 9].iter().flat_map(|w| w.to_le_bytes()).collect(),
            }],
            push: Vec::new(),
            spec: Default::default(),
        };
        let out = run(&m, &Config::default(), res).expect("run");
        let bytes = out.buffers.first().expect("buffer");
        assert_eq!(bytes.get(4..8), Some(&12u32.to_le_bytes()[..]));
        // A second pass finds nothing.
        let again =
            remove_dead(&mut m, &mut Meter::new(&crate::budget::Budget::default())).expect("dce");
        assert_eq!(again, DeadReport::default());
    }

    /// A kernel whose live code reads a `float` constant `a` (also a part of an unused vector
    /// constant) and a private variable initialized with constant vector `init`, and whose dead
    /// code reads `float` constant `b`; it also declares an unused null vector, an unused
    /// integer, an array type whose length `len` nothing else reads, and a barrier whose scope
    /// and semantics are constant operands. Stores `a + w0` (as bits) to word 1 and the
    /// variable's lane 1 to word 2.
    fn constants_kernel() -> BResult<(TypedModule, [Value; 7])> {
        let mut b = ModuleBuilder::new();
        let buf = buffer(&mut b)?;
        let void = b.ty(Type::Void)?;
        let u32_ = b.ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let f32_ = b.ty(Type::Float { width: 32 })?;
        let v2 = b.ty(Type::Vector { elem: f32_, len: 2 })?;
        let scalar = |b: &mut ModuleBuilder, x: f32| {
            b.constant(Constant::Scalar {
                ty: f32_,
                bits: crate::spirv::ir::ConstBits(u64::from(x.to_bits())),
            })
        };
        let (ka, kb, kc) = (
            scalar(&mut b, 1.5)?,
            scalar(&mut b, 2.5)?,
            scalar(&mut b, 3.5)?,
        );
        let unused_vec = b.constant(Constant::Composite {
            ty: v2,
            parts: [ka, kb].into(),
        })?;
        let init = b.constant(Constant::Composite {
            ty: v2,
            parts: [kc, kc].into(),
        })?;
        let null = b.constant(Constant::Null { ty: v2 })?;
        let int = b.constant(Constant::Scalar {
            ty: u32_,
            bits: crate::spirv::ir::ConstBits(77),
        })?;
        let len = b.constant(Constant::Scalar {
            ty: u32_,
            bits: crate::spirv::ir::ConstBits(5),
        })?;
        b.ty(Type::Array {
            elem: u32_,
            len,
            stride: None,
        })?;
        let var = b.plain_global(StorageClass::Private, v2, Some(init))?;
        let values = [
            b.const_value(ka)?,
            b.const_value(kb)?,
            b.const_value(unused_vec)?,
            b.const_value(init)?,
            b.const_value(null)?,
            b.const_value(int)?,
            b.const_value(len)?,
        ];
        let mut fb = b.function(void)?;
        let p0 = word(&mut fb, buf, 0)?;
        let w0 = fb.load(p0, MemAccess::default())?;
        let x = fb.convert(crate::spirv::ir::ConvOp::Bitcast, w0, f32_)?;
        let (a, kb) = (values[0], values[1]);
        let live = fb.binary(BinOp::FAdd, x, a)?;
        fb.binary(BinOp::FMul, x, kb)?;
        let bits = fb.convert(crate::spirv::ir::ConvOp::Bitcast, live, u32_)?;
        let p1 = word(&mut fb, buf, 1)?;
        fb.store(p1, bits, MemAccess::default())?;
        let pv = fb.module().global_value(var)?;
        let v = fb.load(pv, MemAccess::default())?;
        let lane = fb.composite_extract(v, &[1])?;
        let bits = fb.convert(crate::spirv::ir::ConvOp::Bitcast, lane, u32_)?;
        let p2 = word(&mut fb, buf, 2)?;
        fb.store(p2, bits, MemAccess::default())?;
        let sem = Semantics {
            order: MemoryOrder::AcquireRelease,
            workgroup_memory: true,
            ..Semantics::RELAXED
        };
        fb.control_barrier(Scope::Workgroup, Scope::Workgroup, sem)?;
        fb.ret(None)?;
        let id = fb.finish()?;
        b.entry(
            ExecModel::GLCompute,
            id,
            vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
        )?;
        Ok((b.finalize()?, values))
    }

    /// The `float` constant only dead code read, the unused vector constant, the unused null, and
    /// the unused integer leave the declarations; the `float` constant live code reads, the
    /// initializer and its part, the array length, and the barrier's scope and semantics stay; and
    /// the module is valid and computes what it computed.
    #[test]
    fn dead_constants_leave_and_read_ones_stay() {
        let (mut m, [a, b, unused_vec, init, null, int, len]) = constants_kernel().expect("build");
        let declared = |m: &TypedModule, v: Value| {
            m.decls()
                .iter()
                .any(|d| matches!(d, Decl::Const(c) if m.const_value(*c).ok() == Some(v)))
        };
        let report =
            remove_dead(&mut m, &mut Meter::new(&crate::budget::Budget::default())).expect("dce");
        assert_eq!((report.insts, report.consts), (1, 4), "{report:?}");
        for v in [b, unused_vec, null, int] {
            assert!(!declared(&m, v), "{v:?} stays declared");
        }
        let mut named = vec![a, init, len];
        for blk in m.blocks() {
            for node in &blk.insts {
                node.inst
                    .for_each_const(|c| named.extend(m.const_value(c).ok()));
            }
        }
        assert_eq!(named.len(), 6, "the barrier names three constants");
        for v in named {
            assert!(declared(&m, v), "{v:?} left");
        }
        let profile = committed_profiles()
            .expect("profiles")
            .into_iter()
            .find(|(s, _)| s == "moltenvk-m3")
            .expect("moltenvk-m3")
            .1;
        assert_valid(m.clone(), &profile);
        let res = Resources {
            buffers: vec![Buffer {
                slot: Some(slot(0)),
                address: None,
                bytes: [2.0f32.to_bits(), 0, 0]
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect(),
            }],
            push: Vec::new(),
            spec: Default::default(),
        };
        let out = run(&m, &Config::default(), res).expect("run");
        let bytes = out.buffers.first().expect("buffer");
        assert_eq!(bytes.get(4..8), Some(&3.5f32.to_bits().to_le_bytes()[..]));
        assert_eq!(bytes.get(8..12), Some(&3.5f32.to_bits().to_le_bytes()[..]));
        crate::spirv::serialize::serialize(m).expect("serialize");
    }

    /// A kernel that computes `w0` two ways, `w0 | 0` (one instruction) and `w0 + 1 + 2 - 3`
    /// (three), offers the one emitted first for the other, and stores the other to word 1;
    /// with `take` the short one comes first. A dead `uvec2` of `w0` declares the only `uvec2`,
    /// and a pointer type no value has points to the only `uvec3`.
    fn offers_kernel(take: bool) -> BResult<(TypedModule, Vec<Offer>)> {
        let mut b = ModuleBuilder::new();
        let buf = buffer(&mut b)?;
        let void = b.ty(Type::Void)?;
        let u32_ = b.ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let mut fb = b.function(void)?;
        let p0 = word(&mut fb, buf, 0)?;
        let w0 = fb.load(p0, MemAccess::default())?;
        let (zero, one, two, three) = (
            k(&mut fb, 0)?,
            k(&mut fb, 1)?,
            k(&mut fb, 2)?,
            k(&mut fb, 3)?,
        );
        let short = |fb: &mut FunctionBuilder<'_>| fb.binary(BinOp::BitwiseOr, w0, zero);
        let long = |fb: &mut FunctionBuilder<'_>| -> BResult<Value> {
            let a = fb.binary(BinOp::IAdd, w0, one)?;
            let a = fb.binary(BinOp::IAdd, a, two)?;
            fb.binary(BinOp::ISub, a, three)
        };
        let (to, from) = if take {
            (short(&mut fb)?, long(&mut fb)?)
        } else {
            (long(&mut fb)?, short(&mut fb)?)
        };
        let group = fb.offer_group();
        fb.offer(group, from, to);
        let u2 = fb.module().ty(Type::Vector { elem: u32_, len: 2 })?;
        fb.composite_construct(u2, &[w0, w0])?;
        // A declared pointer type no value has keeps the vector type it points to.
        let u3 = fb.module().ty(Type::Vector { elem: u32_, len: 3 })?;
        fb.module().ty(Type::Pointer {
            class: StorageClass::Private,
            pointee: u3,
        })?;
        let p1 = word(&mut fb, buf, 1)?;
        fb.store(p1, from, MemAccess::default())?;
        fb.ret(None)?;
        let id = fb.finish()?;
        b.entry(
            ExecModel::GLCompute,
            id,
            vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
        )?;
        let offers = b.offers().to_vec();
        Ok((b.finalize()?, offers))
    }

    /// An offer is taken when what it frees outweighs what it keeps alive (the long chain becomes a
    /// copy of the short value) and declined otherwise (the short value stays, the long chain
    /// leaves); either way the module is valid, stores `w0`, and serializes, the vector type only
    /// dead code used leaves while the one a declared pointer type names stays, and a second pass
    /// changes nothing.
    #[test]
    fn offers_are_weighed_in_words() {
        for take in [true, false] {
            let (mut m, offers) = offers_kernel(take).expect("build");
            let before = insts(&m);
            let mut meter = Meter::new(&crate::budget::Budget::default());
            let report = remove_dead_keeping(&mut m, &offers, &mut meter).expect("dce");
            // Taken: the long chain's two inner instructions leave and its last one is a copy;
            // declined: the long chain leaves. The dead vector leaves both times.
            let (copies, removed) = if take { (1, 3) } else { (0, 4) };
            assert_eq!(
                (report.copies, report.insts),
                (copies, removed),
                "{take}: {report:?}"
            );
            assert_eq!(insts(&m), before - removed as usize);
            assert_eq!(report.types, 1, "{take}: {report:?}");
            let profile = committed_profiles()
                .expect("profiles")
                .into_iter()
                .find(|(s, _)| s == "moltenvk-m3")
                .expect("moltenvk-m3")
                .1;
            assert_valid(m.clone(), &profile);
            let res = Resources {
                buffers: vec![Buffer {
                    slot: Some(slot(0)),
                    address: None,
                    bytes: [41u32, 0].iter().flat_map(|w| w.to_le_bytes()).collect(),
                }],
                push: Vec::new(),
                spec: Default::default(),
            };
            let out = run(&m, &Config::default(), res).expect("run");
            let bytes = out.buffers.first().expect("buffer");
            assert_eq!(bytes.get(4..8), Some(&41u32.to_le_bytes()[..]), "{take}");
            crate::spirv::serialize::serialize(m.clone()).expect("serialize");
            let mut meter = Meter::new(&crate::budget::Budget::default());
            let again = remove_dead_keeping(&mut m, &[], &mut meter).expect("dce");
            assert_eq!(again, DeadReport::default(), "{take}");
        }
    }

    /// Only instructions that compute a result from their operands are removable.
    #[test]
    fn effects_are_never_removable() {
        let ptr = Value::new(0);
        assert!(!removable(&Inst::Store {
            ptr,
            value: ptr,
            access: MemAccess::default()
        }));
        assert!(!removable(&Inst::Load {
            ptr,
            access: MemAccess {
                volatile: true,
                ..MemAccess::default()
            }
        }));
        assert!(removable(&Inst::Load {
            ptr,
            access: MemAccess::default()
        }));
        assert!(!removable(&Inst::Glsl {
            op: GlslOp::InterpolateAtCentroid,
            args: [ptr].into(),
            contract: Contraction::Allowed,
        }));
        assert!(!removable(&Inst::Call {
            func: crate::spirv::ids::FuncId::new(0),
            args: [].into()
        }));
    }

    /// A float constant `x` in the module of `fb`, as a value.
    fn kf(fb: &mut FunctionBuilder<'_>, x: f32) -> BResult<Value> {
        let f32_ = fb.module().ty(Type::Float { width: 32 })?;
        let c = fb.module().constant(Constant::Scalar {
            ty: f32_,
            bits: crate::spirv::ir::ConstBits(u64::from(x.to_bits())),
        })?;
        fb.module().const_value(c)
    }

    /// A kernel over two pure routines, `r(x) = x * 2` and `s(x, y) = r(x) + y + 3`: it calls
    /// `s` on word 0's float and never reads the result, and stores `r(word 0)` to word 1 when
    /// `live_r` (else word 0 itself).
    fn routines_kernel(live_r: bool) -> BResult<TypedModule> {
        let mut b = ModuleBuilder::new();
        let buf = buffer(&mut b)?;
        let void = b.ty(Type::Void)?;
        let u32_ = b.ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let f32_ = b.ty(Type::Float { width: 32 })?;
        let mut fb = b.function(f32_)?;
        let x = fb.param(f32_)?;
        let two = kf(&mut fb, 2.0)?;
        let y = fb.binary(BinOp::FMul, x, two)?;
        fb.ret(Some(y))?;
        let r = fb.finish()?;
        let mut fb = b.function(f32_)?;
        let x = fb.param(f32_)?;
        let y = fb.param(f32_)?;
        let rx = fb.call(r, &[x])?.expect("a float");
        let sum = fb.binary(BinOp::FAdd, rx, y)?;
        let three = kf(&mut fb, 3.0)?;
        let sum = fb.binary(BinOp::FAdd, sum, three)?;
        fb.ret(Some(sum))?;
        let s = fb.finish()?;
        let mut fb = b.function(void)?;
        let p0 = word(&mut fb, buf, 0)?;
        let w0 = fb.load(p0, MemAccess::default())?;
        let f = fb.convert(crate::spirv::ir::ConvOp::Bitcast, w0, f32_)?;
        fb.call(s, &[f, f])?;
        let out = if live_r {
            let v = fb.call(r, &[f])?.expect("a float");
            fb.convert(crate::spirv::ir::ConvOp::Bitcast, v, u32_)?
        } else {
            w0
        };
        let p1 = word(&mut fb, buf, 1)?;
        fb.store(p1, out, MemAccess::default())?;
        fb.ret(None)?;
        let id = fb.finish()?;
        b.entry(
            ExecModel::GLCompute,
            id,
            vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
        )?;
        b.finalize()
    }

    /// A call of a pure function that nothing reads leaves, and its function leaves once no call
    /// that stays reaches it, with the constants and the function type only it read. With `r` still
    /// called, `s` leaves with `3.0` and its two-parameter function type; with every call of `r`
    /// inside `s`, both leave with `2.0` and both function types. The module stays valid, computes
    /// what it computed, and a second pass finds nothing.
    #[test]
    fn a_dead_call_of_a_pure_routine_leaves_with_its_routine() {
        // Without `r`'s call, the call of `s` also left the bitcast it read with nothing to read.
        for (live_r, insts, funcs, consts, types) in [(true, 1, 1, 1, 1), (false, 2, 2, 2, 2)] {
            let mut m = routines_kernel(live_r).expect("build");
            assert_eq!(pure_functions(&m), [true, true, false]);
            let report = remove_dead(&mut m, &mut Meter::new(&crate::budget::Budget::default()))
                .expect("dce");
            let want = DeadReport {
                insts,
                consts,
                types,
                funcs,
                ..DeadReport::default()
            };
            assert_eq!(report, want, "live_r {live_r}");
            assert_eq!(m.functions().len(), 3 - funcs as usize);
            assert_eq!(m.entry().func.index(), m.functions().len() - 1);
            let calls = m
                .blocks()
                .iter()
                .flat_map(|b| &b.insts)
                .filter(|n| matches!(n.inst, Inst::Call { .. }))
                .count();
            assert_eq!(calls, usize::from(live_r));
            let profile = committed_profiles()
                .expect("profiles")
                .into_iter()
                .find(|(s, _)| s == "moltenvk-m3")
                .expect("moltenvk-m3")
                .1;
            assert_valid(m.clone(), &profile);
            let res = Resources {
                buffers: vec![Buffer {
                    slot: Some(slot(0)),
                    address: None,
                    bytes: [1.5f32.to_bits(), 0]
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect(),
                }],
                push: Vec::new(),
                spec: Default::default(),
            };
            let out = run(&m, &Config::default(), res).expect("run");
            let bytes = out.buffers.first().expect("buffer");
            let want = if live_r { 3.0f32 } else { 1.5 };
            assert_eq!(bytes.get(4..8), Some(&want.to_bits().to_le_bytes()[..]));
            crate::spirv::serialize::serialize(m.clone()).expect("serialize");
            let again = remove_dead(&mut m, &mut Meter::new(&crate::budget::Budget::default()))
                .expect("dce");
            assert_eq!(again, DeadReport::default());
        }
    }

    /// A call of a function that stores stays though nothing reads its result, and so does a call
    /// of a function that calls it; neither function is pure, so both stay and the store still
    /// runs.
    #[test]
    fn a_dead_call_of_an_impure_function_stays() {
        let mut b = ModuleBuilder::new();
        let buf = buffer(&mut b).expect("buffer");
        let void = b.ty(Type::Void).expect("void");
        let u32_ = b
            .ty(Type::Int {
                width: 32,
                signed: false,
            })
            .expect("u32");
        let build = |b: &mut ModuleBuilder| -> BResult<()> {
            let mut fb = b.function(u32_)?;
            let x = fb.param(u32_)?;
            let p2 = word(&mut fb, buf, 2)?;
            fb.store(p2, x, MemAccess::default())?;
            let one = k(&mut fb, 1)?;
            let y = fb.binary(BinOp::IAdd, x, one)?;
            fb.ret(Some(y))?;
            let w = fb.finish()?;
            let mut fb = b.function(u32_)?;
            let x = fb.param(u32_)?;
            let y = fb.call(w, &[x])?.expect("a word");
            fb.ret(Some(y))?;
            let p = fb.finish()?;
            let mut fb = b.function(void)?;
            let p0 = word(&mut fb, buf, 0)?;
            let w0 = fb.load(p0, MemAccess::default())?;
            fb.call(p, &[w0])?;
            fb.ret(None)?;
            let id = fb.finish()?;
            b.entry(
                ExecModel::GLCompute,
                id,
                vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
            )?;
            Ok(())
        };
        build(&mut b).expect("build");
        let mut m = b.finalize().expect("finalize");
        assert_eq!(pure_functions(&m), [false, false, false]);
        let report =
            remove_dead(&mut m, &mut Meter::new(&crate::budget::Budget::default())).expect("dce");
        assert_eq!(report, DeadReport::default());
        assert_eq!(m.functions().len(), 3);
        let res = Resources {
            buffers: vec![Buffer {
                slot: Some(slot(0)),
                address: None,
                bytes: [7u32, 0, 0].iter().flat_map(|w| w.to_le_bytes()).collect(),
            }],
            push: Vec::new(),
            spec: Default::default(),
        };
        let out = run(&m, &Config::default(), res).expect("run");
        let bytes = out.buffers.first().expect("buffer");
        assert_eq!(bytes.get(8..12), Some(&7u32.to_le_bytes()[..]));
    }
}
