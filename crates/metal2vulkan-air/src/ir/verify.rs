//! The verified [`Module`] and its construction invariant.
//!
//! Guest bytes are untrusted, so a module exists only once [`Module::new`] has checked its parts:
//! every id is in range, constants and instructions obey LLVM's typing rules, globals and
//! function signatures are consistent, blocks tile their instruction arena with PHIs first and
//! exactly one terminator last, the entry block has no predecessors, PHIs have one entry per
//! predecessor edge, every use is dominated by its definition, and calls match the callee's
//! function type. Downstream passes assume this and do not re-check.
//!
//! # Dominance
//!
//! SSA dominance is checked over the dominator tree of `graph/`, the single implementation, built
//! per function body. Where a use happens:
//!
//! - a non-PHI operand is used at its instruction: the definition must come earlier in the same
//!   block or lie in a block that strictly dominates it;
//! - a PHI incoming value is used at the end of its incoming block: the definition must lie in a
//!   block that dominates that block (its own block included, so a loop-carried PHI may name a
//!   value of the latch, or of its own block on a self-loop).
//!
//! A use in a block unreachable from the entry is exempt, as in LLVM: no execution reaches it, a
//! valid LLVM module may contain such a use (even `%x = add i32 %x, 1`), and canonicalization
//! deletes unreachable blocks. A definition in an unreachable block dominates no reachable use. The
//! CFG and dominator tree are charged to the meter's ledger before they are reserved and returned
//! to it when the function is verified.

use super::attrs::ParamAttrs;
use super::constant::{Const, ConstExpr, Constant};
use super::id::{BlockId, ConstId, FuncId, GlobalId, Idx, InstId, Sym, TypeId};
use super::inst::{BinOp, Callee, CastOp, FastMath, IntFlags, Ordering, Value};
use super::inst::{Inst, InstKind};
use super::metadata::{Md, NamedMd};
use super::module::{Body, Function, Global, Linkage, ModuleParts, Section};
use super::symbols::Symbols;
use super::types::{AddrSpace, Type, TypeTable};
use super::{ConstTable, DataLayout};
use crate::budget::Meter;
use crate::error::Error;
use crate::error::{DiagName, InputError, InputErrorKind, Location, VerifyError};
use crate::graph::{Cfg, DomTree};

/// A verified AIR module. Immutable: passes consume it with [`Module::into_parts`] and build a
/// new one through [`Module::new`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Module {
    parts: ModuleParts,
}

impl Module {
    /// Verify `parts` into a module.
    pub fn new(parts: ModuleParts, meter: &mut Meter) -> Result<Module, Error> {
        Verifier::new(&parts).module(None, meter)?;
        Ok(Module { parts })
    }

    /// [`Module::new`], also returning the control-flow graph and dominator tree of `keep`'s
    /// body, or `None` where `keep` has none.
    ///
    /// The dominance check builds exactly those two structures for every body it verifies and
    /// returns them to the ledger when the body passes, so a caller that needs the entry's pair
    /// next ([`crate::canon::specialize_dynamic`]) would compute the same graph and the same tree
    /// over the same body a second time. This hands the pair over instead, with its ledger charge
    /// still held, which is what the caller's own `Cfg::build` and `DomTree::compute` would have
    /// charged.
    pub(crate) fn new_with_graphs(
        parts: ModuleParts,
        keep: FuncId,
        meter: &mut Meter,
    ) -> Result<(Module, Option<(Cfg<BlockId>, DomTree<BlockId>)>), Error> {
        let kept = Verifier::new(&parts).module(Some(keep), meter)?;
        Ok((Module { parts }, kept.map(|g| (g.cfg, g.dom))))
    }

    /// Take the module apart for a rewriting pass.
    pub fn into_parts(self) -> ModuleParts {
        self.parts
    }

    /// Read-only access to every part.
    pub fn parts(&self) -> &ModuleParts {
        &self.parts
    }

    /// The string interner.
    pub fn symbols(&self) -> &Symbols {
        &self.parts.symbols
    }

    /// The type interner.
    pub fn types(&self) -> &TypeTable {
        &self.parts.types
    }

    /// The constant interner.
    pub fn consts(&self) -> &ConstTable {
        &self.parts.consts
    }

    /// The target data layout.
    pub fn data_layout(&self) -> &DataLayout {
        &self.parts.data_layout
    }

    /// The global at `id`; `None` when out of range.
    pub fn global(&self, id: GlobalId) -> Option<&Global> {
        self.parts.globals.get(id)
    }

    /// The function at `id`; `None` when out of range.
    pub fn func(&self, id: FuncId) -> Option<&Function> {
        self.parts.funcs.get(id)
    }

    /// The named metadata lists, in bitcode order.
    pub fn named_md(&self) -> &[NamedMd] {
        &self.parts.named_md
    }

    /// The metadata node at `id`; `None` when out of range.
    pub fn md(&self, id: super::MdId) -> Option<&Md> {
        self.parts.metadata.get(id)
    }

    /// The type of `v` as an operand of `func`.
    pub fn value_ty(&self, func: FuncId, v: Value) -> Option<TypeId> {
        let f = self.func(func)?;
        match v {
            Value::Arg(a) => fn_params(&self.parts.types, f.ty)?.get(a.index()).copied(),
            Value::Inst(i) => f.body.as_ref()?.insts.get(i).map(|i| i.ty),
            Value::Const(c) => self.parts.consts.ty(c),
        }
    }
}

fn verify_error(kind: VerifyError, at: Location) -> Error {
    Error::InvalidInput(InputError::new(InputErrorKind::Verify(kind), at))
}

fn fn_params(types: &TypeTable, ty: TypeId) -> Option<&[TypeId]> {
    match types.get(ty)? {
        Type::Function { params, .. } => Some(params),
        _ => None,
    }
}

/// The (return, params, vararg) of a function type.
fn fn_sig(types: &TypeTable, ty: TypeId) -> Option<(TypeId, &[TypeId], bool)> {
    match types.get(ty)? {
        Type::Function {
            ret,
            params,
            vararg,
        } => Some((*ret, params, *vararg)),
        _ => None,
    }
}

struct Verifier<'a> {
    p: &'a ModuleParts,
}

/// The graphs the dominance check of one body builds, kept for the caller that asked for them
/// ([`Module::new_with_graphs`]); their ledger charge is still held.
struct BodyGraphs {
    cfg: Cfg<BlockId>,
    dom: DomTree<BlockId>,
}

impl BodyGraphs {
    /// Return what the pair holds to `meter`'s ledger.
    fn release(self, meter: &mut Meter) {
        meter.release(self.dom.ledger_bytes());
        meter.release(self.cfg.ledger_bytes());
    }
}

/// Typing context for one function body: where an error is reported and how operands type.
struct FnCtx<'a> {
    name: DiagName,
    params: &'a [TypeId],
    ret: TypeId,
    body: &'a Body,
    /// The block of each instruction.
    block_of: Vec<BlockId>,
    block: Option<BlockId>,
    inst: Option<InstId>,
}

impl FnCtx<'_> {
    fn at(&self) -> Location {
        Location::function(
            self.name.clone(),
            self.block.map(|b| b.raw()),
            self.inst.map(|i| i.raw()),
        )
    }

    fn err(&self, kind: VerifyError) -> Error {
        verify_error(kind, self.at())
    }
}

impl<'a> Verifier<'a> {
    fn new(p: &'a ModuleParts) -> Verifier<'a> {
        Verifier { p }
    }

    fn types(&self) -> &'a TypeTable {
        &self.p.types
    }

    fn ty(&self, id: TypeId) -> Option<&'a Type> {
        self.p.types.get(id)
    }

    fn is_void(&self, id: TypeId) -> bool {
        matches!(self.ty(id), Some(Type::Void))
    }

    fn is_i1(&self, id: TypeId) -> bool {
        matches!(self.ty(id), Some(Type::Int(1)))
    }

    fn diag(&self, sym: Sym) -> DiagName {
        DiagName::new(String::from_utf8_lossy(
            self.p.symbols.get(sym).unwrap_or(b"?"),
        ))
    }

    fn space(&self, raw: u32) -> Option<AddrSpace> {
        AddrSpace::from_raw(raw).ok()
    }

    /// Verify every part of the module, keeping the graphs the dominance check builds for `keep`'s
    /// body ([`Module::new_with_graphs`]).
    fn module(&self, keep: Option<FuncId>, meter: &mut Meter) -> Result<Option<BodyGraphs>, Error> {
        let module_err = |k| verify_error(k, Location::module());
        let syms = &self.p.symbols;
        let sym_ok = |s: Sym| syms.contains(s);
        if !self
            .p
            .triple
            .into_iter()
            .chain(self.p.source_filename)
            .all(sym_ok)
        {
            return Err(module_err(VerifyError::BadValueRef));
        }
        for (_, ty) in self.types().iter() {
            let name = match ty {
                Type::Struct { name, .. } | Type::OpaqueStruct { name } => *name,
                _ => None,
            };
            if !name.is_none_or(sym_ok) {
                return Err(module_err(VerifyError::MalformedType));
            }
        }
        for (_, c) in self.p.consts.iter() {
            meter.tick(1)?;
            if !self.constant(c) {
                return Err(module_err(VerifyError::MalformedConstant));
            }
        }
        for (_, g) in self.p.globals.iter() {
            meter.tick(1)?;
            if !self.global(g) {
                return Err(module_err(VerifyError::MalformedGlobal));
            }
        }
        for (_, md) in self.p.metadata.iter() {
            meter.tick(1)?;
            let ok = match md {
                Md::String(s) => sym_ok(*s),
                Md::Value(c) => self.p.consts.contains(*c),
                Md::Node { ops, .. } => ops.iter().flatten().all(|&m| self.p.metadata.contains(m)),
                Md::Debug => true,
            };
            if !ok {
                return Err(module_err(VerifyError::MalformedMetadata));
            }
        }
        for n in &self.p.named_md {
            if !sym_ok(n.name) || !n.ops.iter().all(|&m| self.p.metadata.contains(m)) {
                return Err(module_err(VerifyError::MalformedMetadata));
            }
        }
        let mut kept: Option<BodyGraphs> = None;
        for (id, f) in self.p.funcs.iter() {
            let step = meter
                .tick(1)
                .and_then(|()| self.function(f, keep == Some(id), meter));
            match step {
                Ok(Some(graphs)) => kept = Some(graphs),
                Ok(None) => {}
                // A later function's refusal gives the kept pair's bytes back.
                Err(e) => {
                    if let Some(graphs) = kept {
                        graphs.release(meter);
                    }
                    return Err(e);
                }
            }
        }
        Ok(kept)
    }

    fn section_ok(&self, s: Option<Section>) -> bool {
        match s {
            Some(Section::Other(sym)) => self.p.symbols.contains(sym),
            Some(Section::FcInitializer | Section::StaticInit) | None => true,
        }
    }

    fn param_attrs_ok(&self, a: &ParamAttrs) -> bool {
        a.types().all(|t| self.types().is_sized(t)) && a.syms().all(|s| self.p.symbols.contains(s))
    }

    fn global(&self, g: &Global) -> bool {
        let declaration_linkage = matches!(g.linkage, Linkage::External | Linkage::ExternalWeak);
        self.p.symbols.contains(g.name)
            && self.types().is_sized(g.value_ty)
            && self.section_ok(g.section)
            && match g.init {
                Some(c) => {
                    self.p.consts.ty(c) == Some(g.value_ty)
                        && self.p.consts.inline_asm(Value::Const(c)).is_none()
                }
                None => declaration_linkage,
            }
    }

    fn constant(&self, c: &Constant) -> bool {
        let Some(ty) = self.ty(c.ty) else {
            return false;
        };
        // Inline assembly is no LLVM constant: it is never another constant's operand.
        let asm_child = c
            .value
            .children()
            .iter()
            .any(|&k| self.p.consts.inline_asm(Value::Const(k)).is_some());
        if asm_child {
            return false;
        }
        let child_ty = |id: ConstId| self.p.consts.ty(id);
        match &c.value {
            Const::Int(bits) => match ty {
                Type::Int(w) => fits(*bits, u32::from(*w)),
                _ => false,
            },
            Const::Float(bits) => ty.float_bits().is_some_and(|w| fits(*bits, w)),
            Const::Null | Const::Undef | Const::Poison => ty.is_sized(),
            Const::Aggregate(elems) => {
                let tys: Option<Vec<TypeId>> = elems.iter().map(|&e| child_ty(e)).collect();
                let Some(tys) = tys else {
                    return false;
                };
                match ty {
                    Type::Struct { fields, .. } => **fields == *tys,
                    Type::Array { elem, len } => {
                        u64::try_from(tys.len()) == Ok(*len) && tys.iter().all(|t| t == elem)
                    }
                    Type::Vector { elem, len } => {
                        u32::try_from(tys.len()) == Ok(*len) && tys.iter().all(|t| t == elem)
                    }
                    _ => false,
                }
            }
            Const::GlobalAddr(g) => self
                .p
                .globals
                .get(*g)
                .is_some_and(|g| *ty == Type::Ptr { space: g.space }),
            Const::FuncAddr(f) => {
                self.p.funcs.contains(*f)
                    && self
                        .space(self.p.data_layout.program_space)
                        .is_some_and(|space| *ty == Type::Ptr { space })
            }
            // LLVM types an inline assembly expression `ptr` in address space 0.
            Const::InlineAsm(_) => self
                .space(0)
                .is_some_and(|space| *ty == Type::Ptr { space }),
            Const::Expr(ConstExpr::Cast { op, ops: [v] }) => {
                child_ty(*v).is_some_and(|src| self.cast_ok(*op, src, c.ty))
            }
            Const::Expr(ConstExpr::BinOp {
                op,
                flags,
                ops: [a, b],
            }) => match (child_ty(*a), child_ty(*b)) {
                (Some(a), Some(b)) => self.binop_ok(*op, *flags, FastMath::NONE, a, b, c.ty),
                _ => false,
            },
            Const::Expr(ConstExpr::Gep {
                src_ty,
                ops,
                inbounds: _,
            }) => {
                let Some((&base, indices)) = ops.split_first() else {
                    return false;
                };
                let Some(base_ty) = child_ty(base) else {
                    return false;
                };
                let idx: Option<Vec<(TypeId, Option<u64>)>> = indices
                    .iter()
                    .map(|&i| {
                        let k = self.p.consts.get(i)?;
                        Some((k.ty, const_int(&k.value)))
                    })
                    .collect();
                idx.is_some_and(|idx| self.gep_ok(*src_ty, base_ty, &idx, c.ty))
            }
        }
    }

    fn function(
        &self,
        f: &'a Function,
        keep: bool,
        meter: &mut Meter,
    ) -> Result<Option<BodyGraphs>, Error> {
        let name = self.diag(f.name);
        let fn_err = |k| verify_error(k, Location::function(name.clone(), None, None));
        let Some((ret, params, _)) = fn_sig(self.types(), f.ty) else {
            return Err(fn_err(VerifyError::MalformedFunction));
        };
        let declaration_linkage = matches!(f.linkage, Linkage::External | Linkage::ExternalWeak);
        let attrs_ok = f.params.len() == params.len()
            && f.params.iter().all(|p| {
                self.param_attrs_ok(&p.attrs) && p.name.is_none_or(|s| self.p.symbols.contains(s))
            })
            && self.param_attrs_ok(&f.ret_attrs)
            && f.attrs
                .strings
                .iter()
                .all(|&(k, v)| self.p.symbols.contains(k) && self.p.symbols.contains(v));
        if !self.p.symbols.contains(f.name)
            || !attrs_ok
            || !self.section_ok(f.section)
            || (f.body.is_none() && !declaration_linkage)
        {
            return Err(fn_err(VerifyError::MalformedFunction));
        }
        let Some(body) = &f.body else {
            return Ok(None);
        };
        // Everything charged here is returned once the body is checked, pass or fail.
        let mut block_of = Vec::new();
        meter.reserve(&mut block_of, body.insts.len())?;
        let block_of_bytes = body
            .insts
            .len()
            .saturating_mul(std::mem::size_of::<BlockId>());
        let mut cx = FnCtx {
            name,
            params,
            ret,
            body,
            block_of,
            block: None,
            inst: None,
        };
        let checked = self.layout(&mut cx, meter).and_then(|()| {
            self.branch_targets(&mut cx, meter)?;
            let cfg = Cfg::build(body, meter)?;
            let dom = match DomTree::compute(&cfg, meter) {
                Ok(dom) => dom,
                Err(e) => {
                    meter.release(cfg.ledger_bytes());
                    return Err(e);
                }
            };
            let graphs = BodyGraphs { cfg, dom };
            match self.instructions(&mut cx, &graphs.cfg, &graphs.dom, meter) {
                // The caller asked for this body's pair, so it stays charged and goes to it.
                Ok(()) if keep => Ok(Some(graphs)),
                Ok(()) => {
                    graphs.release(meter);
                    Ok(None)
                }
                Err(e) => {
                    graphs.release(meter);
                    Err(e)
                }
            }
        });
        meter.release(block_of_bytes);
        checked
    }

    /// The per-instruction checks, in layout order: operand references, typing, then dominance.
    fn instructions(
        &self,
        cx: &mut FnCtx<'a>,
        cfg: &Cfg<BlockId>,
        dom: &DomTree<BlockId>,
        meter: &mut Meter,
    ) -> Result<(), Error> {
        let body = cx.body;
        for (b, block) in body.blocks.iter() {
            cx.block = Some(b);
            for i in block.inst_ids() {
                meter.tick(1)?;
                cx.inst = Some(i);
                let inst = body
                    .insts
                    .get(i)
                    .ok_or_else(|| cx.err(VerifyError::BadValueRef))?;
                self.operands(cx, inst)?;
                self.inst(cx, inst, cfg)?;
                self.dominance(cx, b, i, inst, dom, meter)?;
            }
        }
        Ok(())
    }

    /// Blocks tile the arena; PHIs lead; exactly one terminator ends each block.
    fn layout(&self, cx: &mut FnCtx<'a>, meter: &mut Meter) -> Result<(), Error> {
        let body = cx.body;
        if body.blocks.is_empty() {
            return Err(cx.err(VerifyError::MissingTerminator));
        }
        let mut next = 0u32;
        for (b, block) in body.blocks.iter() {
            cx.block = Some(b);
            meter.tick(1)?;
            if block.start.raw() != next || block.end.raw() <= block.start.raw() {
                return Err(cx.err(VerifyError::MissingTerminator));
            }
            if block.name.is_some_and(|s| !self.p.symbols.contains(s)) {
                return Err(cx.err(VerifyError::BadBlockRef));
            }
            next = block.end.raw();
            let mut phis_done = false;
            for i in block.inst_ids() {
                cx.inst = Some(i);
                let inst = body
                    .insts
                    .get(i)
                    .ok_or_else(|| cx.err(VerifyError::MissingTerminator))?;
                let last = Some(i) == block.last();
                if inst.kind.is_terminator() != last {
                    return Err(cx.err(if last {
                        VerifyError::MissingTerminator
                    } else {
                        VerifyError::TerminatorNotLast
                    }));
                }
                let is_phi = matches!(inst.kind, InstKind::Phi { .. });
                if is_phi && phis_done {
                    return Err(cx.err(VerifyError::PhiPredecessors));
                }
                phis_done |= !is_phi;
                cx.block_of.push(b);
            }
        }
        cx.block = None;
        cx.inst = None;
        if next as usize != body.insts.len() {
            return Err(cx.err(VerifyError::MissingTerminator));
        }
        Ok(())
    }

    /// Every branch target is a block of the body, and no edge enters the entry block. After
    /// this the body satisfies the `graph::Graph` contract.
    fn branch_targets(&self, cx: &mut FnCtx<'a>, meter: &mut Meter) -> Result<(), Error> {
        let body = cx.body;
        let entry = BlockId::new(0);
        for (b, block) in body.blocks.iter() {
            cx.block = Some(b);
            cx.inst = block.last();
            for s in body.successors(b) {
                meter.tick(1)?;
                if !body.blocks.contains(s) {
                    return Err(cx.err(VerifyError::BadBlockRef));
                }
                if s == entry {
                    cx.block = Some(entry);
                    cx.inst = None;
                    return Err(cx.err(VerifyError::EntryBlockHasPredecessors));
                }
            }
        }
        cx.block = None;
        cx.inst = None;
        Ok(())
    }

    /// Every operand exists and is a non-void value.
    fn operands(&self, cx: &FnCtx<'a>, inst: &Inst) -> Result<(), Error> {
        let mut ok = true;
        // Inline assembly is only ever a call's callee (LLVM: "Cannot take the address of an
        // inline asm!").
        let mut asm_uses = 0usize;
        let asm_callee = matches!(&inst.kind, InstKind::Call(call)
            if matches!(call.callee, Callee::Indirect { callee, .. }
                if self.p.consts.inline_asm(callee).is_some()));
        inst.kind.for_each_value(|v| {
            asm_uses += usize::from(self.p.consts.inline_asm(v).is_some());
            ok &= match v {
                Value::Arg(a) => a.index() < cx.params.len(),
                Value::Const(c) => self.p.consts.contains(c),
                Value::Inst(i) => cx
                    .body
                    .insts
                    .get(i)
                    .is_some_and(|def| !self.is_void(def.ty)),
            };
        });
        if !ok || asm_uses > usize::from(asm_callee) {
            return Err(cx.err(VerifyError::BadValueRef));
        }
        if let InstKind::Call(call) = &inst.kind {
            if let Some(f) = call.callee.func() {
                if !self.p.funcs.contains(f) {
                    return Err(cx.err(VerifyError::BadValueRef));
                }
            }
        }
        Ok(())
    }

    /// Every instruction operand of `inst` (at `at` in `block`) is available where it is used;
    /// see the module docs for the rules and the unreachable-block exemption. Runs after the
    /// operand and typing checks, so every operand and incoming block is in range.
    fn dominance(
        &self,
        cx: &FnCtx<'a>,
        block: BlockId,
        at: InstId,
        inst: &Inst,
        dom: &DomTree<BlockId>,
        meter: &mut Meter,
    ) -> Result<(), Error> {
        // `use_at` is the using instruction, or `None` for the end of `use_block` (a PHI edge).
        let available = |def: InstId, use_block: BlockId, use_at: Option<InstId>| {
            if !dom.is_reachable(use_block) {
                return true;
            }
            match cx.block_of.get(def.index()) {
                Some(&def_block) if def_block == use_block => use_at.is_none_or(|u| def < u),
                Some(&def_block) => dom.dominates(def_block, use_block),
                None => false,
            }
        };
        let mut uses = 0u32;
        let mut ok = true;
        match &inst.kind {
            InstKind::Phi { incoming, .. } => {
                for &(v, from) in incoming.iter() {
                    uses = uses.saturating_add(1);
                    if let Value::Inst(def) = v {
                        ok &= available(def, from, None);
                    }
                }
            }
            kind => kind.for_each_value(|v| {
                uses = uses.saturating_add(1);
                if let Value::Inst(def) = v {
                    ok &= available(def, block, Some(at));
                }
            }),
        }
        meter.tick(uses)?;
        if ok {
            Ok(())
        } else {
            Err(cx.err(VerifyError::UseNotDominated))
        }
    }

    fn value_ty(&self, cx: &FnCtx<'a>, v: Value) -> Option<TypeId> {
        match v {
            Value::Arg(a) => cx.params.get(a.index()).copied(),
            Value::Inst(i) => cx.body.insts.get(i).map(|i| i.ty),
            Value::Const(c) => self.p.consts.ty(c),
        }
    }

    fn inst(&self, cx: &FnCtx<'a>, inst: &Inst, cfg: &Cfg<BlockId>) -> Result<(), Error> {
        let vt = |v: Value| {
            self.value_ty(cx, v)
                .ok_or_else(|| cx.err(VerifyError::BadValueRef))
        };
        let operand = |ok: bool| {
            if ok {
                Ok(())
            } else {
                Err(cx.err(VerifyError::OperandType))
            }
        };
        let result = |ok: bool| {
            if ok {
                Ok(())
            } else {
                Err(cx.err(VerifyError::ResultType))
            }
        };
        let types = self.types();
        let res = inst.ty;
        if !types.contains(res) {
            return Err(cx.err(VerifyError::ResultType));
        }
        let void = self.is_void(res);
        let sized = types.is_sized(res);
        match &inst.kind {
            InstKind::BinOp {
                op,
                flags,
                fmf,
                a,
                b,
            } => {
                let (a, b) = (vt(*a)?, vt(*b)?);
                operand(self.binop_ok(*op, *flags, *fmf, a, b, a))?;
                result(res == a)
            }
            InstKind::FNeg { v, .. } => {
                let v = vt(*v)?;
                operand(types.float_bits(v).is_some())?;
                result(res == v)
            }
            InstKind::Cast { op, v } => {
                let v = vt(*v)?;
                operand(self.cast_ok(*op, v, res))
            }
            InstKind::ICmp { a, b, .. } => {
                let (a, b) = (vt(*a)?, vt(*b)?);
                operand(a == b && (types.int_bits(a).is_some() || types.ptr_space(a).is_some()))?;
                result(self.is_bool_like(res, types.lanes(a)))
            }
            InstKind::FCmp { a, b, .. } => {
                let (a, b) = (vt(*a)?, vt(*b)?);
                operand(a == b && types.float_bits(a).is_some())?;
                result(self.is_bool_like(res, types.lanes(a)))
            }
            InstKind::Select { c, t, f, fmf } => {
                let (c, t, f) = (vt(*c)?, vt(*t)?, vt(*f)?);
                operand(t == f && sized)?;
                operand(
                    self.is_i1(c)
                        || (self.is_bool_like(c, types.lanes(t)) && types.lanes(t).is_some()),
                )?;
                operand(fmf.is_empty() || types.float_bits(t).is_some())?;
                result(res == t)
            }
            InstKind::Freeze(v) => {
                let v = vt(*v)?;
                result(res == v && sized)
            }
            InstKind::ExtractElement { vec, idx } => {
                let (vec, idx) = (vt(*vec)?, vt(*idx)?);
                operand(types.lanes(vec).is_some() && self.is_int_scalar(idx))?;
                result(res == types.scalar(vec))
            }
            InstKind::InsertElement { vec, elt, idx } => {
                let (vec, elt, idx) = (vt(*vec)?, vt(*elt)?, vt(*idx)?);
                operand(
                    types.lanes(vec).is_some()
                        && elt == types.scalar(vec)
                        && self.is_int_scalar(idx),
                )?;
                result(res == vec)
            }
            InstKind::ShuffleVector { a, b, mask } => {
                let (a, b) = (vt(*a)?, vt(*b)?);
                let Some(n) = types.lanes(a) else {
                    return operand(false);
                };
                operand(
                    a == b
                        && !mask.is_empty()
                        && mask
                            .iter()
                            .flatten()
                            .all(|&m| u64::from(m) < 2 * u64::from(n)),
                )?;
                result(
                    types.scalar(res) == types.scalar(a)
                        && types.lanes(res).map(|l| l as usize) == Some(mask.len()),
                )
            }
            InstKind::ExtractValue { agg, indices } => {
                let agg = vt(*agg)?;
                let leaf = self.aggregate_leaf(agg, indices);
                operand(leaf.is_some())?;
                result(leaf == Some(res))
            }
            InstKind::InsertValue { agg, val, indices } => {
                let (agg, val) = (vt(*agg)?, vt(*val)?);
                operand(self.aggregate_leaf(agg, indices) == Some(val))?;
                result(res == agg)
            }
            InstKind::Alloca { ty, count, .. } => {
                let count = vt(*count)?;
                operand(types.is_sized(*ty) && self.is_int_scalar(count))?;
                let space = self.space(self.p.data_layout.alloca_space);
                result(space.is_some_and(|space| self.ty(res) == Some(&Type::Ptr { space })))
            }
            InstKind::Load { ptr, atomic, .. } => {
                let ptr = vt(*ptr)?;
                operand(self.is_ptr_scalar(ptr))?;
                operand(
                    atomic.is_none_or(|a| {
                        !matches!(a.ordering, Ordering::Release | Ordering::AcqRel)
                    }),
                )?;
                result(sized)
            }
            InstKind::Store {
                val, ptr, atomic, ..
            } => {
                let (val, ptr) = (vt(*val)?, vt(*ptr)?);
                operand(self.is_ptr_scalar(ptr) && types.is_sized(val))?;
                operand(
                    atomic.is_none_or(|a| {
                        !matches!(a.ordering, Ordering::Acquire | Ordering::AcqRel)
                    }),
                )?;
                result(void)
            }
            InstKind::Gep {
                src_ty,
                base,
                indices,
                ..
            } => {
                let base = vt(*base)?;
                let idx: Result<Vec<(TypeId, Option<u64>)>, Error> = indices
                    .iter()
                    .map(|&i| {
                        let ty = vt(i)?;
                        let value = match i {
                            Value::Const(c) => {
                                self.p.consts.get(c).and_then(|k| const_int(&k.value))
                            }
                            Value::Arg(_) | Value::Inst(_) => None,
                        };
                        Ok((ty, value))
                    })
                    .collect();
                operand(self.gep_ok(*src_ty, base, &idx?, res))
            }
            InstKind::AtomicRmw {
                op,
                ptr,
                val,
                atomic,
                ..
            } => {
                let (ptr, val) = (vt(*ptr)?, vt(*val)?);
                let value_ok = match self.ty(val) {
                    Some(t) if op.is_float() => t.is_float(),
                    Some(Type::Int(_)) => true,
                    Some(t) => {
                        *op == super::AtomicRmwOp::Xchg
                            && (t.is_float() || matches!(t, Type::Ptr { .. }))
                    }
                    None => false,
                };
                operand(
                    self.is_ptr_scalar(ptr) && value_ok && atomic.ordering != Ordering::Unordered,
                )?;
                result(res == val)
            }
            InstKind::CmpXchg {
                ptr,
                cmp,
                new,
                success,
                failure,
                ..
            } => {
                let (ptr, cmp, new) = (vt(*ptr)?, vt(*cmp)?, vt(*new)?);
                operand(
                    self.is_ptr_scalar(ptr)
                        && cmp == new
                        && matches!(self.ty(cmp), Some(Type::Int(_) | Type::Ptr { .. }))
                        && success.ordering != Ordering::Unordered
                        && !matches!(
                            failure,
                            Ordering::Unordered | Ordering::Release | Ordering::AcqRel
                        ),
                )?;
                result(match self.ty(res) {
                    Some(Type::Struct {
                        fields,
                        packed: false,
                        name: None,
                    }) => matches!(**fields, [t, b] if t == cmp && self.is_i1(b)),
                    _ => false,
                })
            }
            InstKind::Fence(a) => {
                operand(matches!(
                    a.ordering,
                    Ordering::Acquire | Ordering::Release | Ordering::AcqRel | Ordering::SeqCst
                ))?;
                result(void)
            }
            InstKind::Call(call) => {
                let fn_ty = match &call.callee {
                    Callee::Indirect { callee, fn_ty } => {
                        operand(self.is_ptr_scalar(vt(*callee)?))?;
                        *fn_ty
                    }
                    Callee::Direct(f) => self.callee_ty(cx, *f, false)?,
                    Callee::Air { decl, .. }
                    | Callee::Llvm { decl, .. }
                    | Callee::UnknownIntrinsic(decl) => self.callee_ty(cx, *decl, true)?,
                };
                let Some((ret, params, vararg)) = fn_sig(types, fn_ty) else {
                    return operand(false);
                };
                let arity_ok = if vararg {
                    call.args.len() >= params.len()
                } else {
                    call.args.len() == params.len()
                };
                operand(arity_ok)?;
                for (&a, &p) in call.args.iter().zip(params) {
                    operand(vt(a)? == p)?;
                }
                for a in call.args.iter().skip(params.len()) {
                    operand(types.is_sized(vt(*a)?))?;
                }
                if let Some(attrs) = &call.attrs {
                    operand(
                        attrs.params.len() <= call.args.len()
                            && attrs.params.iter().all(|p| self.param_attrs_ok(p))
                            && self.param_attrs_ok(&attrs.ret)
                            && attrs.func.strings.iter().all(|&(k, v)| {
                                self.p.symbols.contains(k) && self.p.symbols.contains(v)
                            }),
                    )?;
                }
                result(res == ret)
            }
            InstKind::Phi { incoming, fmf } => {
                result(sized)?;
                operand(fmf.is_empty() || types.float_bits(res).is_some())?;
                for &(v, _) in incoming.iter() {
                    operand(vt(v)? == res)?;
                }
                if incoming.iter().any(|&(_, b)| !cx.body.blocks.contains(b)) {
                    return Err(cx.err(VerifyError::BadBlockRef));
                }
                let mut blocks: Vec<BlockId> = incoming.iter().map(|&(_, b)| b).collect();
                blocks.sort_unstable();
                let mut entries: Vec<(BlockId, Value)> =
                    incoming.iter().map(|&(v, b)| (b, v)).collect();
                entries.sort_unstable();
                let consistent = entries.windows(2).all(|w| match w {
                    [(b0, v0), (b1, v1)] => b0 != b1 || v0 == v1,
                    _ => true,
                });
                // `Cfg` lists predecessors by ascending source block, once per parallel edge.
                let preds = cx.block.map(|b| cfg.predecessors(b));
                if !preds.is_some_and(|p| p.eq(blocks.iter().copied())) || !consistent {
                    return Err(cx.err(VerifyError::PhiPredecessors));
                }
                Ok(())
            }
            InstKind::Ret(v) => {
                result(void)?;
                match v {
                    None => operand(self.is_void(cx.ret)),
                    Some(v) => operand(vt(*v)? == cx.ret && !self.is_void(cx.ret)),
                }
            }
            InstKind::Br { .. } | InstKind::Unreachable => result(void),
            InstKind::CondBr { c, .. } => {
                operand(self.is_i1(vt(*c)?))?;
                result(void)
            }
            InstKind::Switch { v, cases, .. } => {
                let v = vt(*v)?;
                let Some(Type::Int(w)) = self.ty(v) else {
                    return operand(false);
                };
                let mut values: Vec<u64> = cases.iter().map(|&(c, _)| c).collect();
                values.sort_unstable();
                operand(
                    values.iter().all(|&c| fits(c, u32::from(*w)))
                        && values.windows(2).all(|w| matches!(w, [a, b] if a != b)),
                )?;
                result(void)
            }
        }
    }

    /// The type of a callee function, which must be a declaration for intrinsic forms.
    fn callee_ty(&self, cx: &FnCtx<'a>, f: FuncId, intrinsic: bool) -> Result<TypeId, Error> {
        let func = self
            .p
            .funcs
            .get(f)
            .ok_or_else(|| cx.err(VerifyError::BadValueRef))?;
        if intrinsic && func.body.is_some() {
            return Err(cx.err(VerifyError::OperandType));
        }
        Ok(func.ty)
    }

    /// `res` is `i1`, or `<n x i1>` when `lanes` is `Some(n)`.
    fn is_bool_like(&self, res: TypeId, lanes: Option<u32>) -> bool {
        match lanes {
            None => self.is_i1(res),
            Some(n) => self.types().lanes(res) == Some(n) && self.is_i1(self.types().scalar(res)),
        }
    }

    fn is_int_scalar(&self, t: TypeId) -> bool {
        matches!(self.ty(t), Some(Type::Int(_)))
    }

    fn is_ptr_scalar(&self, t: TypeId) -> bool {
        matches!(self.ty(t), Some(Type::Ptr { .. }))
    }

    fn binop_ok(
        &self,
        op: BinOp,
        flags: IntFlags,
        fmf: FastMath,
        a: TypeId,
        b: TypeId,
        res: TypeId,
    ) -> bool {
        let types = self.types();
        let kind_ok = if op.is_float() {
            types.float_bits(a).is_some()
        } else {
            types.int_bits(a).is_some() && fmf.is_empty()
        };
        a == b && a == res && kind_ok && op.allows(flags)
    }

    fn cast_ok(&self, op: CastOp, src: TypeId, dst: TypeId) -> bool {
        let t = self.types();
        let (Some(s), Some(d)) = (t.get(src), t.get(dst)) else {
            return false;
        };
        if s.is_aggregate() || d.is_aggregate() || !s.is_sized() || !d.is_sized() {
            return false;
        }
        let same_lanes = t.lanes(src) == t.lanes(dst);
        let (si, di) = (t.int_bits(src), t.int_bits(dst));
        let (sf, df) = (t.float_bits(src), t.float_bits(dst));
        let (sp, dp) = (t.ptr_space(src), t.ptr_space(dst));
        match op {
            CastOp::Trunc => same_lanes && matches!((si, di), (Some(s), Some(d)) if d < s),
            CastOp::ZExt | CastOp::SExt => {
                same_lanes && matches!((si, di), (Some(s), Some(d)) if d > s)
            }
            CastOp::FPTrunc => same_lanes && matches!((sf, df), (Some(s), Some(d)) if d < s),
            CastOp::FPExt => same_lanes && matches!((sf, df), (Some(s), Some(d)) if d > s),
            CastOp::FPToUI | CastOp::FPToSI => same_lanes && sf.is_some() && di.is_some(),
            CastOp::UIToFP | CastOp::SIToFP => same_lanes && si.is_some() && df.is_some(),
            CastOp::PtrToInt => same_lanes && sp.is_some() && di.is_some(),
            CastOp::IntToPtr => same_lanes && si.is_some() && dp.is_some(),
            CastOp::AddrSpaceCast => same_lanes && matches!((sp, dp), (Some(s), Some(d)) if s != d),
            CastOp::Bitcast => match (sp, dp) {
                (Some(s), Some(d)) => same_lanes && s == d,
                (None, None) => {
                    let bits = |elem: Option<u32>, lanes: Option<u32>| {
                        elem.map(|b| u64::from(b) * u64::from(lanes.unwrap_or(1)))
                    };
                    let sb = bits(si.map(u32::from).or(sf), t.lanes(src));
                    let db = bits(di.map(u32::from).or(df), t.lanes(dst));
                    sb.is_some() && sb == db
                }
                _ => false,
            },
        }
    }

    /// A scalar `getelementptr`: the base is a pointer, every index an integer, struct indices
    /// constant `i32`s in range, and the result a pointer in the base's space.
    fn gep_ok(
        &self,
        src_ty: TypeId,
        base: TypeId,
        indices: &[(TypeId, Option<u64>)],
        res: TypeId,
    ) -> bool {
        let types = self.types();
        let Some(Type::Ptr { space }) = self.ty(base) else {
            return false;
        };
        if !types.is_sized(src_ty) || self.ty(res) != Some(&Type::Ptr { space: *space }) {
            return false;
        }
        let Some((first, rest)) = indices.split_first() else {
            return true;
        };
        if !self.is_int_scalar(first.0) {
            return false;
        }
        let mut cur = src_ty;
        for &(ity, value) in rest {
            cur = match self.ty(cur) {
                Some(Type::Struct { fields, .. }) => {
                    let field = match (self.ty(ity), value) {
                        (Some(Type::Int(32)), Some(v)) => usize::try_from(v).ok(),
                        _ => None,
                    };
                    match field.and_then(|f| fields.get(f)) {
                        Some(&f) => f,
                        None => return false,
                    }
                }
                Some(Type::Array { elem, .. } | Type::Vector { elem, .. })
                    if self.is_int_scalar(ity) =>
                {
                    *elem
                }
                _ => return false,
            };
        }
        true
    }

    /// The member type `indices` select inside aggregate `agg`.
    fn aggregate_leaf(&self, agg: TypeId, indices: &[u32]) -> Option<TypeId> {
        if indices.is_empty() {
            return None;
        }
        let mut cur = agg;
        for &i in indices {
            cur = match self.ty(cur)? {
                Type::Struct { fields, .. } => *fields.get(i as usize)?,
                Type::Array { elem, len } if u64::from(i) < *len => *elem,
                _ => return None,
            };
        }
        Some(cur)
    }
}

/// Whether `bits` is a canonical zero-extended value of `width` bits.
fn fits(bits: u64, width: u32) -> bool {
    width >= 64 || bits >> width == 0
}

fn const_int(c: &Const) -> Option<u64> {
    match c {
        Const::Int(v) => Some(*v),
        _ => None,
    }
}
