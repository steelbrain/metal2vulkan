//! Inlining.
//!
//! Every direct call to a defined function is inlined into the entry, transitively, so the entry
//! becomes the module's one defined function. SPIR-V forbids recursion, so a call-graph cycle is
//! `Unsupported{Recursion}`.
//!
//! **Size first, then one pass.** A memoized pre-pass over the call DAG computes, for every
//! reachable function `f`, exactly how many instructions and blocks its inlined body has:
//!
//! ```text
//! insts(f)  = own(f)    + Σ_calls (insts(g) + phi(call))     phi = 1 when the call has a value
//! blocks(f) = blocks(f) + Σ_calls (blocks(g) + 1)                   and g has a `ret`
//! ```
//!
//! The entry's count is checked against `Limit::InlinedInstructions` before anything
//! proportional to it is allocated. Because the counts are exact, each old instruction and block
//! of each function has a fixed offset in its inlined body, and the new id of any value is known
//! before it is emitted (a use may precede its definition in layout order). The body is then
//! emitted in one pass over an explicit frame stack (no host recursion, however deep the call
//! graph):
//!
//! - A call to a defined `g` ends its block's segment with `br` to `g`'s inlined entry; `g`'s
//!   blocks follow; a **continuation** block then takes the rest of the caller's block.
//! - A `ret v` of `g` becomes `br continuation`; when the call has a value and `g` returns, the
//!   continuation starts with `phi [v, ret block]...` (one PHI per call, even with one `ret`),
//!   which replaces the call's uses. A call with a value whose callee never returns has an
//!   unreachable continuation, and its uses read `undef`.
//! - Arguments replace the callee's parameters. A PHI names its predecessor by the last segment
//!   of the old block, where the old terminator now lives.
//! - Allocas stay where the callee had them.

use crate::budget::Meter;
use crate::error::{DiagName, Location};
use crate::error::{Error, Feature, Limit, Unsupported};
use crate::ir::Value;
use crate::ir::{Arena, Block, BlockId, Body, Callee, Const, ConstTable, Constant, FuncId};
use crate::ir::{Function, Idx, Inst, InstId, InstKind, Symbols, Type, TypeId, TypeTable};

use super::rewrite::{filled, invariant, scoped};

/// Inline every direct call to a defined function into `entry`'s body. `None` when the body
/// calls no defined function. A refused call is located in the function that makes it and names
/// its callee (`symbols`).
pub(super) fn inline(
    funcs: &Arena<FuncId, Function>,
    (types, symbols): (&TypeTable, &Symbols),
    consts: &mut ConstTable,
    entry: FuncId,
    m: &mut Meter,
) -> Result<Option<Body>, Error> {
    scoped(m, |m, held| {
        let Some(sizes) = Sizes::measure(funcs, (types, symbols), entry, m, held)? else {
            return Ok(None);
        };
        let insts = sizes.insts(entry)?;
        m.cap(Limit::InlinedInstructions, insts)?;
        let tables = Tables::build(funcs, types, &sizes, m, held)?;
        // The counts are exact: the inlined body holds no arena slack.
        let out = Body::with_capacity(usize_of(insts)?, usize_of(sizes.blocks(entry)?)?, m)?;
        let e = Emit {
            funcs,
            types,
            consts,
            sizes: &sizes,
            tables: &tables,
            out,
            open: None,
        };
        e.run(entry, m, held).map(Some)
    })
}

/// The body of a defined function, or `None`.
fn body_of(funcs: &Arena<FuncId, Function>, f: FuncId) -> Option<&Body> {
    funcs.get(f).and_then(|f| f.body.as_ref())
}

/// The defined function `kind` calls directly, if any: a call the inliner replaces.
pub(super) fn inlined_callee(funcs: &Arena<FuncId, Function>, kind: &InstKind) -> Option<FuncId> {
    match kind {
        InstKind::Call(call) => match call.callee {
            Callee::Direct(g) if body_of(funcs, g).is_some() => Some(g),
            _ => None,
        },
        _ => None,
    }
}

/// The inlined sizes of every function the entry reaches (the pre-pass).
struct Sizes {
    /// Per function: inlined instructions and blocks, own `ret` count; `None` if unreached.
    of: Vec<Option<Size>>,
}

#[derive(Clone, Copy, Debug)]
struct Size {
    insts: u64,
    blocks: u64,
    rets: u32,
}

impl Sizes {
    /// Measure the call DAG under `entry`, refusing a cycle. `None` when the entry calls no
    /// defined function.
    fn measure(
        funcs: &Arena<FuncId, Function>,
        (types, symbols): (&TypeTable, &Symbols),
        entry: FuncId,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<Option<Sizes>, Error> {
        let body = body_of(funcs, entry).ok_or_else(|| invariant("the entry has no body"))?;
        if !body
            .insts
            .values()
            .any(|i| inlined_callee(funcs, &i.kind).is_some())
        {
            return Ok(None);
        }
        // 0 = unvisited, 1 = on the DFS stack, 2 = measured.
        let mut state = filled(m, held, funcs.len(), 0u8)?;
        let mut of: Vec<Option<Size>> = filled(m, held, funcs.len(), None)?;
        let mut stack: Vec<(FuncId, u32)> = vec![(entry, 0)];
        if let Some(s) = state.get_mut(entry.index()) {
            *s = 1;
        }
        while let Some(&mut (f, ref mut cursor)) = stack.last_mut() {
            m.tick(1)?;
            let body = body_of(funcs, f).ok_or_else(|| invariant("a callee without a body"))?;
            let mut next = None;
            while let Some(inst) = body.insts.get(InstId::new(*cursor)) {
                let at = InstId::new(*cursor);
                *cursor += 1;
                let Some(g) = inlined_callee(funcs, &inst.kind) else {
                    continue;
                };
                // A refusal of this call: at it in `f`, naming `g`.
                let refuse = |feature| -> Error {
                    let name = |f: FuncId| {
                        let bytes = funcs.get(f).and_then(|f| symbols.get(f.name));
                        DiagName::new(String::from_utf8_lossy(bytes.unwrap_or(b"?")))
                    };
                    let block = body
                        .blocks
                        .iter()
                        .find(|(_, b)| b.start <= at && at < b.end)
                        .map(|(b, _)| b.raw());
                    Unsupported::not_implemented(feature)
                        .naming(name(g))
                        .at(Location::function(name(f), block, Some(at.raw())))
                        .into()
                };
                if !takes(funcs, types, g, &inst.kind) {
                    return Err(refuse(Feature::DirectCall));
                }
                match state.get(g.index()).copied() {
                    Some(0) => {
                        next = Some(g);
                        break;
                    }
                    Some(1) => return Err(refuse(Feature::Recursion)),
                    _ => {}
                }
            }
            if let Some(g) = next {
                if let Some(s) = state.get_mut(g.index()) {
                    *s = 1;
                }
                let bytes = std::mem::size_of::<(FuncId, u32)>();
                m.charge(bytes)?;
                *held = held.saturating_add(bytes);
                stack.push((g, 0));
                continue;
            }
            // Every callee of `f` is measured.
            let mut size = Size {
                insts: 0,
                blocks: body.blocks.len() as u64,
                rets: 0,
            };
            for inst in body.insts.values() {
                m.tick(1)?;
                size.insts = size.insts.saturating_add(1);
                if matches!(inst.kind, InstKind::Ret(_)) {
                    size.rets = size.rets.saturating_add(1);
                }
                if let Some(g) = inlined_callee(funcs, &inst.kind) {
                    let c = of
                        .get(g.index())
                        .copied()
                        .flatten()
                        .ok_or_else(|| invariant("a callee measured after its caller"))?;
                    let phi = u64::from(has_phi(types, inst.ty, c.rets));
                    size.insts = size.insts.saturating_add(c.insts).saturating_add(phi);
                    size.blocks = size.blocks.saturating_add(c.blocks).saturating_add(1);
                }
            }
            if let Some(slot) = of.get_mut(f.index()) {
                *slot = Some(size);
            }
            if let Some(s) = state.get_mut(f.index()) {
                *s = 2;
            }
            stack.pop();
        }
        Ok(Some(Sizes { of }))
    }

    fn get(&self, f: FuncId) -> Result<Size, Error> {
        self.of
            .get(f.index())
            .copied()
            .flatten()
            .ok_or_else(|| invariant("an unmeasured function"))
    }

    fn insts(&self, f: FuncId) -> Result<u64, Error> {
        self.get(f).map(|s| s.insts)
    }

    fn blocks(&self, f: FuncId) -> Result<u64, Error> {
        self.get(f).map(|s| s.blocks)
    }
}

/// Whether the inliner can take a call to `g`: not variadic, no `byval` parameter or argument (a
/// callee's private copy of its argument, which inlining would have to materialize).
fn takes(funcs: &Arena<FuncId, Function>, types: &TypeTable, g: FuncId, call: &InstKind) -> bool {
    let Some(f) = funcs.get(g) else {
        return false;
    };
    let vararg = matches!(types.get(f.ty), Some(Type::Function { vararg: true, .. }));
    let byval_arg = match call {
        InstKind::Call(c) => c
            .attrs
            .as_ref()
            .is_some_and(|a| a.params.iter().any(|p| p.byval.is_some())),
        _ => false,
    };
    !(vararg || byval_arg || f.params.iter().any(|p| p.attrs.byval.is_some()))
}

/// Whether a call of type `ty` to a callee with `rets` returns gets a continuation PHI.
fn has_phi(types: &TypeTable, ty: TypeId, rets: u32) -> bool {
    rets > 0 && !matches!(types.get(ty), Some(Type::Void))
}

/// Per reached function: where each old instruction and block lands in its inlined body.
struct Tables {
    of: Vec<Option<Table>>,
}

struct Table {
    /// Per instruction: its offset (a call's: its `br`).
    inst: Vec<u32>,
    /// Per block: the offset of its first segment and of its last (where its terminator lands).
    first: Vec<u32>,
    last: Vec<u32>,
}

impl Tables {
    fn build(
        funcs: &Arena<FuncId, Function>,
        types: &TypeTable,
        sizes: &Sizes,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<Tables, Error> {
        let mut of: Vec<Option<Table>> = Vec::new();
        m.reserve(&mut of, sizes.of.len())?;
        *held = held.saturating_add(sizes.of.len() * std::mem::size_of::<Option<Table>>());
        for (index, size) in sizes.of.iter().enumerate() {
            let (Some(_), Some(f)) = (size, FuncId::from_index(index)) else {
                of.push(None);
                continue;
            };
            let body = body_of(funcs, f).ok_or_else(|| invariant("a callee without a body"))?;
            let mut inst = filled(m, held, body.insts.len(), 0u32)?;
            let mut first = filled(m, held, body.blocks.len(), 0u32)?;
            let mut last = filled(m, held, body.blocks.len(), 0u32)?;
            let (mut ni, mut nb) = (0u64, 0u64);
            for (b, block) in body.blocks.iter() {
                m.tick(1)?;
                if let Some(slot) = first.get_mut(b.index()) {
                    *slot = u32_of(nb)?;
                }
                for i in block.inst_ids() {
                    if let Some(slot) = inst.get_mut(i.index()) {
                        *slot = u32_of(ni)?;
                    }
                    ni += 1;
                    let kind = &body
                        .insts
                        .get(i)
                        .ok_or_else(|| invariant("instruction out of range"))?
                        .kind;
                    if let Some(g) = inlined_callee(funcs, kind) {
                        let c = sizes.get(g)?;
                        let ty = body.insts.get(i).map(|i| i.ty);
                        let phi = ty.is_some_and(|t| has_phi(types, t, c.rets));
                        ni = ni + c.insts + u64::from(phi);
                        nb = nb + c.blocks + 1;
                    }
                }
                if let Some(slot) = last.get_mut(b.index()) {
                    *slot = u32_of(nb)?;
                }
                nb += 1;
            }
            of.push(Some(Table { inst, first, last }));
        }
        Ok(Tables { of })
    }

    fn get(&self, f: FuncId) -> Result<&Table, Error> {
        self.of
            .get(f.index())
            .and_then(Option::as_ref)
            .ok_or_else(|| invariant("a function without offsets"))
    }
}

fn u32_of(v: u64) -> Result<u32, Error> {
    u32::try_from(v).map_err(|_| Error::LimitExceeded(Limit::InlinedInstructions))
}

fn usize_of(v: u64) -> Result<usize, Error> {
    usize::try_from(v).map_err(|_| Error::LimitExceeded(Limit::InlinedInstructions))
}

/// A function being emitted: its offsets in the output and where it is.
struct Frame<'b> {
    body: &'b Body,
    table: &'b Table,
    /// The output ids of its first instruction and block.
    ib: u32,
    bb: u32,
    /// What its parameters are in the output.
    args: Vec<Value>,
    /// Where its `ret` branches (a callee's continuation); `None` for the entry.
    cont: Option<BlockId>,
    /// The next old block, and the next old instruction in it.
    block: u32,
    at: InstId,
    /// Set while a callee runs: the continuation's PHI type, and its incoming entries once the
    /// callee is done.
    resume: Option<Resume>,
}

struct Resume {
    phi: Option<TypeId>,
    incoming: Vec<(Value, BlockId)>,
    /// The continuation's output block id.
    cont: u32,
}

struct Emit<'a> {
    funcs: &'a Arena<FuncId, Function>,
    types: &'a TypeTable,
    consts: &'a mut ConstTable,
    sizes: &'a Sizes,
    tables: &'a Tables,
    out: Body,
    /// The open segment: its first instruction and its name.
    open: Option<(InstId, Option<crate::ir::Sym>)>,
}

impl<'a> Emit<'a> {
    fn frame(
        &self,
        f: FuncId,
        ib: u32,
        bb: u32,
        args: Vec<Value>,
        cont: Option<BlockId>,
    ) -> Result<Frame<'a>, Error> {
        let body = body_of(self.funcs, f).ok_or_else(|| invariant("a callee without a body"))?;
        let at = body
            .blocks
            .get(BlockId::new(0))
            .ok_or_else(|| invariant("a body without blocks"))?
            .start;
        Ok(Frame {
            body,
            table: self.tables.get(f)?,
            ib,
            bb,
            args,
            cont,
            block: 0,
            at,
            resume: None,
        })
    }

    fn next_inst(&self) -> Result<u32, Error> {
        u32::try_from(self.out.insts.len()).map_err(|_| invariant("instruction ids"))
    }

    fn next_block(&self) -> Result<u32, Error> {
        u32::try_from(self.out.blocks.len()).map_err(|_| invariant("block ids"))
    }

    fn close(&mut self, m: &mut Meter) -> Result<(), Error> {
        let (start, name) = self
            .open
            .take()
            .ok_or_else(|| invariant("no open segment"))?;
        let end = InstId::new(self.next_inst()?);
        self.out.blocks.push(Block { start, end, name }, m)?;
        Ok(())
    }

    fn push(&mut self, ty: TypeId, kind: InstKind, m: &mut Meter) -> Result<(), Error> {
        self.out.push_inst(Inst { ty, kind }, m)?;
        Ok(())
    }

    /// The output value of `v`, an operand in `fr`.
    fn map(&mut self, fr: &Frame<'_>, v: Value, m: &mut Meter) -> Result<Value, Error> {
        Ok(match v {
            Value::Arg(a) => fr
                .args
                .get(a.index())
                .copied()
                .ok_or_else(|| invariant("argument out of range"))?,
            Value::Const(c) => Value::Const(c),
            Value::Inst(j) => {
                let inst = fr
                    .body
                    .insts
                    .get(j)
                    .ok_or_else(|| invariant("instruction out of range"))?;
                let off = fr
                    .table
                    .inst
                    .get(j.index())
                    .copied()
                    .ok_or_else(|| invariant("instruction out of range"))?;
                let id = fr.ib.checked_add(off).ok_or_else(|| invariant("ids"))?;
                match inlined_callee(self.funcs, &inst.kind) {
                    None => Value::Inst(InstId::new(id)),
                    Some(g) => {
                        let c = self.sizes.get(g)?;
                        if has_phi(self.types, inst.ty, c.rets) {
                            let phi = u64::from(id) + 1 + c.insts;
                            Value::Inst(InstId::new(u32_of(phi)?))
                        } else {
                            // The callee never returns: the continuation is unreachable.
                            let undef = Constant {
                                ty: inst.ty,
                                value: Const::Undef,
                            };
                            Value::Const(self.consts.intern(undef, m)?)
                        }
                    }
                }
            }
        })
    }

    fn target(fr: &Frame<'_>, b: BlockId, last: bool) -> Result<BlockId, Error> {
        let offs = if last {
            &fr.table.last
        } else {
            &fr.table.first
        };
        let off = offs
            .get(b.index())
            .copied()
            .ok_or_else(|| invariant("block out of range"))?;
        Ok(BlockId::new(
            fr.bb.checked_add(off).ok_or_else(|| invariant("ids"))?,
        ))
    }

    /// `inst` of `fr`, in the output.
    fn rewrite(&mut self, fr: &Frame<'_>, inst: &Inst, m: &mut Meter) -> Result<InstKind, Error> {
        let mut kind = match (&inst.kind, fr.cont) {
            (InstKind::Ret(_), Some(dest)) => return Ok(InstKind::Br { dest, hints: None }),
            (k, _) => k.clone(),
        };
        let mut ops = Vec::new();
        kind.for_each_value(|v| ops.push(v));
        let mut mapped = Vec::with_capacity(ops.len());
        for v in ops {
            mapped.push(self.map(fr, v, m)?);
        }
        let mut next = mapped.into_iter();
        kind.for_each_value_mut(|v| {
            if let Some(n) = next.next() {
                *v = n;
            }
        });
        match &mut kind {
            InstKind::Br { dest, .. } => *dest = Self::target(fr, *dest, false)?,
            InstKind::CondBr { t, f, .. } => {
                *t = Self::target(fr, *t, false)?;
                *f = Self::target(fr, *f, false)?;
            }
            InstKind::Switch { default, cases, .. } => {
                *default = Self::target(fr, *default, false)?;
                for (_, b) in cases.iter_mut() {
                    *b = Self::target(fr, *b, false)?;
                }
            }
            InstKind::Phi { incoming, .. } => {
                for (_, p) in incoming.iter_mut() {
                    *p = Self::target(fr, *p, true)?;
                }
            }
            _ => {}
        }
        Ok(kind)
    }

    /// Emit `entry`'s inlined body. Each frame's argument list is scratch, returned with the
    /// pass's (`held`); a continuation PHI's incoming list is its operand list, charged once, by
    /// [`Body::push_inst`] as the PHI is pushed.
    fn run(mut self, entry: FuncId, m: &mut Meter, held: &mut usize) -> Result<Body, Error> {
        let void = self
            .types
            .lookup(&Type::Void)
            .ok_or_else(|| invariant("a body without a void type"))?;
        let params = match self.funcs.get(entry).and_then(|f| self.types.get(f.ty)) {
            Some(Type::Function { params, .. }) => params.len(),
            _ => return Err(invariant("the entry's type is not a function")),
        };
        let mut args = Vec::new();
        m.reserve(&mut args, params)?;
        *held = held.saturating_add(params.saturating_mul(std::mem::size_of::<Value>()));
        for k in 0..params {
            let k = u32::try_from(k).map_err(|_| invariant("parameters"))?;
            args.push(Value::Arg(crate::ir::ArgIdx::new(k)));
        }
        let mut stack = vec![self.frame(entry, 0, 0, args, None)?];
        loop {
            m.tick(1)?;
            let Some(top) = stack.last_mut() else {
                break;
            };
            let body = top.body;
            let Some(block) = body.blocks.get(BlockId::new(top.block)).copied() else {
                // The frame is done. A callee's caller resumes in its continuation.
                let done = stack.pop().ok_or_else(|| invariant("frame stack"))?;
                if let Some(caller) = stack.last_mut() {
                    let r = caller
                        .resume
                        .as_mut()
                        .ok_or_else(|| invariant("a callee without a call"))?;
                    if r.phi.is_some() {
                        for (b, blk) in done.body.blocks.iter() {
                            let Some(term) = blk.last().and_then(|t| done.body.insts.get(t)) else {
                                continue;
                            };
                            if let InstKind::Ret(Some(v)) = term.kind {
                                let v = self.map(&done, v, m)?;
                                let from = Self::target(&done, b, true)?;
                                r.incoming.push((v, from));
                            }
                        }
                    }
                }
                continue;
            };
            if self.open.is_none() {
                let name = match top.resume.take() {
                    Some(r) => {
                        if r.cont != self.next_block()? {
                            return Err(invariant("inlined block offsets disagree"));
                        }
                        let start = InstId::new(self.next_inst()?);
                        self.open = Some((start, None));
                        if let Some(ty) = r.phi {
                            let kind = InstKind::Phi {
                                incoming: r.incoming.into_boxed_slice(),
                                fmf: crate::ir::FastMath::NONE,
                            };
                            self.push(ty, kind, m)?;
                        }
                        None
                    }
                    None => {
                        let expect = Self::target(top, BlockId::new(top.block), false)?;
                        if expect.raw() != self.next_block()? {
                            return Err(invariant("inlined block offsets disagree"));
                        }
                        Some(block.name)
                    }
                };
                if let Some(name) = name {
                    self.open = Some((InstId::new(self.next_inst()?), name));
                }
            }
            let mut call = None;
            while top.at < block.end {
                let i = top.at;
                let inst = body
                    .insts
                    .get(i)
                    .ok_or_else(|| invariant("instruction out of range"))?;
                top.at = InstId::new(i.raw() + 1);
                if let Some(g) = inlined_callee(self.funcs, &inst.kind) {
                    call = Some((g, inst));
                    break;
                }
                let off = top
                    .table
                    .inst
                    .get(i.index())
                    .copied()
                    .ok_or_else(|| invariant("instruction out of range"))?;
                if top.ib.checked_add(off) != Some(self.next_inst()?) {
                    return Err(invariant("inlined instruction offsets disagree"));
                }
                let kind = self.rewrite(top, inst, m)?;
                self.push(inst.ty, kind, m)?;
            }
            let Some((g, inst)) = call else {
                // The block's terminator is out.
                self.close(m)?;
                top.block += 1;
                if let Some(b) = body.blocks.get(BlockId::new(top.block)) {
                    top.at = b.start;
                }
                continue;
            };
            // A call to `g`: branch into its inlined body, which follows this segment.
            let callee_bb = self.next_block()? + 1;
            self.push(
                void,
                InstKind::Br {
                    dest: BlockId::new(callee_bb),
                    hints: None,
                },
                m,
            )?;
            self.close(m)?;
            let c = self.sizes.get(g)?;
            let callee_ib = self.next_inst()?;
            let cont = u64::from(callee_bb) + c.blocks;
            let InstKind::Call(call) = &inst.kind else {
                return Err(invariant("a call that is not a call"));
            };
            let mut args = Vec::new();
            m.reserve(&mut args, call.args.len())?;
            *held = held.saturating_add(std::mem::size_of_val::<[Value]>(&call.args));
            for &a in call.args.iter() {
                args.push(self.map(top, a, m)?);
            }
            let cont = u32_of(cont)?;
            top.resume = Some(Resume {
                phi: has_phi(self.types, inst.ty, c.rets).then_some(inst.ty),
                incoming: Vec::new(),
                cont,
            });
            let callee = self.frame(g, callee_ib, callee_bb, args, Some(BlockId::new(cont)))?;
            stack.push(callee);
        }
        let expect = self.sizes.insts(entry)?;
        if self.out.insts.len() as u64 != expect {
            return Err(invariant("the inlined body is not the measured size"));
        }
        Ok(self.out)
    }
}
