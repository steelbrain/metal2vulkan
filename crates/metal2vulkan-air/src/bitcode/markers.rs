//! Marker declarations no use survives.
//!
//! Decode drops every call to a marker (`llvm.lifetime.*`, `llvm.assume`,
//! `llvm.experimental.noalias.scope.decl`: [`DeclClass::Drop`]), so a marker declaration whose
//! every use was such a call is left with none. What remains of it is how an LLVM major spells the
//! marker, not the program: LLVM 22 dropped `lifetime.start`'s size operand, so the declaration's
//! type depends on the major that assembled the module, and LLVM's reader re-mangles a
//! typed-pointer marker (`.p0i8` to `.p0`) into a new declaration after the module's others, so
//! its position depends on the reader that wrote the text. The calls' operands are the major's too:
//! LLVM 21 writes `lifetime.start`'s size (`i64 16`), so a size no kept instruction uses is a
//! constant only under LLVM 21. [`drop_unused`] removes each such declaration, with its address
//! constant, every constant only the dropped calls named (with the constants only those were built
//! of), and the types only the removed used, and renumbers the functions, constants, and types that
//! remain in their old order. A marker declaration with any other use (an operand, a constant, an
//! initializer, or metadata naming its address) is kept as written, and so is a constant anything
//! kept names; nothing else is removed. Removal does not undo order: a type or constant LLVM's
//! writer reached first through a marker (LLVM 21 numbers `i64` with `lifetime.start`'s type) keeps
//! that place, where LLVM 22 numbers it at its first other use, so a module whose only earlier
//! reach of a type is a marker still decodes to two numberings: type/constant numbering differs
//! between LLVM versions when a marker declaration's type mentions a type first used later. A
//! fixture may be unable to drop the marker, because the alloca it brackets is the shape the
//! fixture pins (an AIR 1.x `atomic_compare_exchange_weak`'s expected-value pointer, `air.frexp`'s
//! `thread I*` exponent), so the golden-printout test compares such a module with its type and
//! constant ids renumbered off the pinned major and to the byte under it
//! (`metal2vulkan/src/tests/air_input.rs`). A decode runs it only when asked
//! (`DecodeOptions::drop_unused_markers`): it costs a pass over every instruction and a rebuilt
//! module, and translation does not need it, because entry pruning keeps only what the entry
//! reaches.

use super::types::grow;
use super::ModuleTables;
use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind};
use crate::intrinsics::DeclClass;
use crate::ir::{
    Arena, Callee, Const, ConstExpr, ConstId, ConstTable, Constant, FuncId, Idx, Inst, InstKind,
    Md, ModuleParts, ParamAttrs, Type, TypeId, TypeTable, Value,
};

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// `n` cleared flags, charged to the ledger.
fn flags(m: &mut Meter, n: usize) -> Result<Vec<bool>, Error> {
    let mut v = Vec::new();
    m.reserve(&mut v, n)?;
    v.resize(n, false);
    Ok(v)
}

fn put(v: &mut [bool], i: usize, to: bool) {
    if let Some(slot) = v.get_mut(i) {
        *slot = to;
    }
}

fn get(v: &[bool], i: usize) -> bool {
    v.get(i).copied().unwrap_or(false)
}

/// The new id of each old one: the removed are `None`, the others numbered in their old order.
fn ranks<I: Idx>(removed: &[bool], m: &mut Meter) -> Result<Vec<Option<I>>, Error> {
    let mut out = Vec::new();
    m.reserve(&mut out, removed.len())?;
    let mut next = 0usize;
    for &r in removed {
        if r {
            out.push(None);
        } else {
            out.push(Some(
                I::from_index(next).ok_or_else(|| invariant("renumbered id out of range"))?,
            ));
            next += 1;
        }
    }
    Ok(out)
}

/// Every type an instruction names directly.
fn inst_types(inst: &Inst, mut f: impl FnMut(TypeId)) {
    f(inst.ty);
    match &inst.kind {
        InstKind::Alloca { ty, .. } => f(*ty),
        InstKind::Gep { src_ty, .. } => f(*src_ty),
        InstKind::Call(call) => {
            if let Callee::Indirect { fn_ty, .. } = call.callee {
                f(fn_ty);
            }
            if let Some(a) = &call.attrs {
                a.ret.types().for_each(&mut f);
                a.params.iter().flat_map(ParamAttrs::types).for_each(&mut f);
            }
        }
        _ => {}
    }
}

/// Remove the marker declarations of `intrinsics` (every `air.*`/`llvm.*` declaration with its
/// class) that nothing in `t` uses, and the constants of `marker_consts` (the operands of the
/// dropped marker calls) that nothing kept names, renumbering what remains in order. `t` is
/// returned unchanged when there is none.
pub(super) fn drop_unused(
    t: ModuleTables,
    intrinsics: &[(FuncId, DeclClass)],
    marker_consts: &[ConstId],
    m: &mut Meter,
) -> Result<ModuleTables, Error> {
    match Plan::new(&t.parts, intrinsics, marker_consts, m)? {
        None => Ok(t),
        Some(plan) => plan.apply(t, m),
    }
}

/// The renumbering of one module: `None` for what is removed.
struct Plan {
    funcs: Vec<Option<FuncId>>,
    consts: Vec<Option<ConstId>>,
    types: Vec<Option<TypeId>>,
    /// Whether any type is removed (otherwise the type table is kept as it is).
    types_change: bool,
}

impl Plan {
    fn new(
        p: &ModuleParts,
        intrinsics: &[(FuncId, DeclClass)],
        marker_consts: &[ConstId],
        m: &mut Meter,
    ) -> Result<Option<Plan>, Error> {
        let mut dropped = flags(m, p.funcs.len())?;
        let mut any = false;
        for &(f, class) in intrinsics {
            m.tick(1)?;
            if matches!(class, DeclClass::Drop(_))
                && p.funcs.get(f).is_some_and(|f| f.body.is_none())
            {
                put(&mut dropped, f.index(), true);
                any = true;
            }
        }
        if !any && marker_consts.is_empty() {
            return Ok(None);
        }

        // ---- Candidates: the constants a dropped call named and those they are built of. A
        // global's or function's address is a module value, there whatever uses it: never one. ----
        let removable = |c: ConstId| {
            p.consts
                .get(c)
                .is_some_and(|k| !matches!(k.value, Const::FuncAddr(_) | Const::GlobalAddr(_)))
        };
        let mut candidate = flags(m, p.consts.len())?;
        let mut work = Vec::new();
        for &c in marker_consts {
            m.tick(1)?;
            if !get(&candidate, c.index()) && removable(c) {
                put(&mut candidate, c.index(), true);
                grow(m, &mut work)?;
                work.push(c);
            }
        }
        while let Some(c) = work.pop() {
            m.tick(1)?;
            for &k in p.consts.get(c).map_or(&[][..], |c| c.value.children()) {
                if !get(&candidate, k.index()) && removable(k) {
                    put(&mut candidate, k.index(), true);
                    grow(m, &mut work)?;
                    work.push(k);
                }
            }
        }

        // ---- Uses: calls, and every constant an operand, initializer, metadata node, or other
        // kept constant names. ----
        let mut named = flags(m, p.consts.len())?;
        for f in p.funcs.values() {
            let Some(body) = &f.body else {
                continue;
            };
            for inst in body.insts.values() {
                m.tick(1)?;
                inst.kind.for_each_value(|v| {
                    if let Value::Const(c) = v {
                        put(&mut named, c.index(), true);
                    }
                });
                if let InstKind::Call(call) = &inst.kind {
                    if let Some(g) = call.callee.func() {
                        put(&mut dropped, g.index(), false);
                    }
                }
            }
        }
        for g in p.globals.values() {
            if let Some(c) = g.init {
                put(&mut named, c.index(), true);
            }
        }
        for md in p.metadata.values() {
            if let Md::Value(c) = md {
                put(&mut named, c.index(), true);
            }
        }
        // A constant names its children; a candidate only once something kept names it.
        for (id, c) in p.consts.iter() {
            m.tick(1)?;
            if !get(&candidate, id.index()) {
                for &k in c.value.children() {
                    put(&mut named, k.index(), true);
                }
            }
        }
        for (id, _) in p.consts.iter() {
            if get(&candidate, id.index()) && get(&named, id.index()) {
                grow(m, &mut work)?;
                work.push(id);
            }
        }
        while let Some(c) = work.pop() {
            m.tick(1)?;
            for &k in p.consts.get(c).map_or(&[][..], |c| c.value.children()) {
                if !get(&named, k.index()) {
                    put(&mut named, k.index(), true);
                    if get(&candidate, k.index()) {
                        grow(m, &mut work)?;
                        work.push(k);
                    }
                }
            }
        }
        for (id, c) in p.consts.iter() {
            if let Const::FuncAddr(f) = c.value {
                if get(&named, id.index()) {
                    put(&mut dropped, f.index(), false);
                }
            }
        }
        let mut gone_c = flags(m, p.consts.len())?;
        let mut any_gone = false;
        for (id, c) in p.consts.iter() {
            let gone = match c.value {
                Const::FuncAddr(f) => get(&dropped, f.index()),
                _ => get(&candidate, id.index()) && !get(&named, id.index()),
            };
            put(&mut gone_c, id.index(), gone);
            any_gone |= gone;
        }
        if !any_gone && !dropped.iter().any(|&d| d) {
            return Ok(None);
        }

        // ---- Types: those the removed name and nothing kept does. ----
        let mut kept = flags(m, p.types.len())?;
        let mut dead = flags(m, p.types.len())?;
        for (id, c) in p.consts.iter() {
            m.tick(1)?;
            let to = if get(&gone_c, id.index()) {
                &mut dead
            } else {
                &mut kept
            };
            put(to, c.ty.index(), true);
            if let Const::Expr(ConstExpr::Gep { src_ty, .. }) = &c.value {
                put(to, src_ty.index(), true);
            }
        }
        for g in p.globals.values() {
            put(&mut kept, g.value_ty.index(), true);
        }
        for (f, func) in p.funcs.iter() {
            m.tick(1)?;
            let to = if get(&dropped, f.index()) {
                &mut dead
            } else {
                &mut kept
            };
            put(to, func.ty.index(), true);
            for t in func
                .ret_attrs
                .types()
                .chain(func.params.iter().flat_map(|q| q.attrs.types()))
            {
                put(to, t.index(), true);
            }
            if let Some(body) = &func.body {
                for inst in body.insts.values() {
                    m.tick(1)?;
                    inst_types(inst, |t| put(to, t.index(), true));
                }
            }
        }
        // Children precede their parents, so one pass from the last type closes both sets.
        for (id, ty) in p.types.iter().rev() {
            m.tick(1)?;
            let (k, d) = (get(&kept, id.index()), get(&dead, id.index()));
            if k || d {
                for c in ty.children() {
                    if k {
                        put(&mut kept, c.index(), true);
                    }
                    if d {
                        put(&mut dead, c.index(), true);
                    }
                }
            }
        }
        let mut gone_t = flags(m, p.types.len())?;
        let mut types_change = false;
        for (i, slot) in gone_t.iter_mut().enumerate() {
            *slot = get(&dead, i) && !get(&kept, i);
            types_change |= *slot;
        }
        Ok(Some(Plan {
            funcs: ranks(&dropped, m)?,
            consts: ranks(&gone_c, m)?,
            types: ranks(&gone_t, m)?,
            types_change,
        }))
    }

    fn ty(&self, t: TypeId) -> Result<TypeId, Error> {
        self.types
            .get(t.index())
            .copied()
            .flatten()
            .ok_or_else(|| invariant("a kept use of a removed type"))
    }

    fn konst(&self, c: ConstId) -> Result<ConstId, Error> {
        self.consts
            .get(c.index())
            .copied()
            .flatten()
            .ok_or_else(|| invariant("a kept use of a removed constant"))
    }

    fn func(&self, f: FuncId) -> Result<FuncId, Error> {
        self.funcs
            .get(f.index())
            .copied()
            .flatten()
            .ok_or_else(|| invariant("a kept use of a removed function"))
    }

    fn param_attrs(&self, a: &mut ParamAttrs) -> Result<(), Error> {
        if let Some(t) = &mut a.byval {
            *t = self.ty(*t)?;
        }
        if let Some(t) = &mut a.sret {
            *t = self.ty(*t)?;
        }
        Ok(())
    }

    fn inst(&self, inst: &mut Inst) -> Result<(), Error> {
        inst.ty = self.ty(inst.ty)?;
        let mut failed = None;
        inst.kind.for_each_value_mut(|v| {
            if let Value::Const(c) = v {
                match self.konst(*c) {
                    Ok(n) => *c = n,
                    Err(e) => {
                        failed.get_or_insert(e);
                    }
                }
            }
        });
        if let Some(e) = failed {
            return Err(e);
        }
        match &mut inst.kind {
            InstKind::Alloca { ty, .. } => *ty = self.ty(*ty)?,
            InstKind::Gep { src_ty, .. } => *src_ty = self.ty(*src_ty)?,
            InstKind::Call(call) => {
                match &mut call.callee {
                    Callee::Direct(f)
                    | Callee::UnknownIntrinsic(f)
                    | Callee::Unbound { decl: f, .. }
                    | Callee::Air { decl: f, .. }
                    | Callee::Llvm { decl: f, .. } => *f = self.func(*f)?,
                    Callee::Indirect { fn_ty, .. } => *fn_ty = self.ty(*fn_ty)?,
                }
                if let Some(a) = &mut call.attrs {
                    self.param_attrs(&mut a.ret)?;
                    for p in a.params.iter_mut() {
                        self.param_attrs(p)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// The type a kept type becomes: its children renumbered.
    fn type_of(&self, ty: &Type) -> Result<Type, Error> {
        Ok(match ty {
            Type::Vector { elem, len } => Type::Vector {
                elem: self.ty(*elem)?,
                len: *len,
            },
            Type::Array { elem, len } => Type::Array {
                elem: self.ty(*elem)?,
                len: *len,
            },
            Type::Struct {
                fields,
                packed,
                name,
            } => Type::Struct {
                fields: fields
                    .iter()
                    .map(|&f| self.ty(f))
                    .collect::<Result<_, _>>()?,
                packed: *packed,
                name: *name,
            },
            Type::Function {
                ret,
                params,
                vararg,
            } => Type::Function {
                ret: self.ty(*ret)?,
                params: params
                    .iter()
                    .map(|&p| self.ty(p))
                    .collect::<Result<_, _>>()?,
                vararg: *vararg,
            },
            other => {
                if !other.children().is_empty() {
                    return Err(invariant("a type with children not renumbered"));
                }
                other.clone()
            }
        })
    }

    /// The constant a kept constant becomes.
    fn constant_of(&self, c: &Constant) -> Result<Constant, Error> {
        let value = match &c.value {
            Const::Aggregate(elems) => Const::Aggregate(
                elems
                    .iter()
                    .map(|&e| self.konst(e))
                    .collect::<Result<_, _>>()?,
            ),
            Const::FuncAddr(f) => Const::FuncAddr(self.func(*f)?),
            Const::Expr(ConstExpr::Cast { op, ops: [v] }) => Const::Expr(ConstExpr::Cast {
                op: *op,
                ops: [self.konst(*v)?],
            }),
            Const::Expr(ConstExpr::BinOp {
                op,
                flags,
                ops: [a, b],
            }) => Const::Expr(ConstExpr::BinOp {
                op: *op,
                flags: *flags,
                ops: [self.konst(*a)?, self.konst(*b)?],
            }),
            Const::Expr(ConstExpr::Gep {
                src_ty,
                inbounds,
                ops,
            }) => Const::Expr(ConstExpr::Gep {
                src_ty: self.ty(*src_ty)?,
                inbounds: *inbounds,
                ops: ops
                    .iter()
                    .map(|&o| self.konst(o))
                    .collect::<Result<_, _>>()?,
            }),
            Const::Int(_)
            | Const::Float(_)
            | Const::Null
            | Const::Undef
            | Const::Poison
            | Const::GlobalAddr(_)
            | Const::InlineAsm(_) => c.value.clone(),
        };
        Ok(Constant {
            ty: self.ty(c.ty)?,
            value,
        })
    }

    fn apply(self, t: ModuleTables, m: &mut Meter) -> Result<ModuleTables, Error> {
        let ModuleTables {
            parts,
            types: bc_types,
            mut values,
            attr_lists,
            md_kinds,
            sync_scopes,
            mut defined,
            function_blocks,
            mut fn_entries,
            enum_attrs,
        } = t;
        let ModuleParts {
            symbols,
            types,
            consts,
            mut globals,
            funcs,
            mut metadata,
            named_md,
            triple,
            source_filename,
            data_layout,
        } = parts;

        let types = if self.types_change {
            let mut out = TypeTable::new();
            for (id, ty) in types.iter() {
                m.tick(1)?;
                let Some(want) = self.types.get(id.index()).copied().flatten() else {
                    continue;
                };
                if out.intern(self.type_of(ty)?, m)? != want {
                    return Err(invariant("a type renumbered out of order"));
                }
            }
            out
        } else {
            types
        };
        let mut new_consts = ConstTable::new();
        for (id, c) in consts.iter() {
            m.tick(1)?;
            let Some(want) = self.consts.get(id.index()).copied().flatten() else {
                continue;
            };
            if new_consts.intern(self.constant_of(c)?, m)? != want {
                return Err(invariant("a constant renumbered out of order"));
            }
        }
        for g in globals.values_mut() {
            g.value_ty = self.ty(g.value_ty)?;
            if let Some(c) = &mut g.init {
                *c = self.konst(*c)?;
            }
        }
        let mut new_funcs = Arena::new();
        for (i, mut f) in funcs.into_vec().into_iter().enumerate() {
            m.tick(1)?;
            if self.funcs.get(i).copied().flatten().is_none() {
                continue;
            }
            f.ty = self.ty(f.ty)?;
            self.param_attrs(&mut f.ret_attrs)?;
            for p in f.params.iter_mut() {
                self.param_attrs(&mut p.attrs)?;
            }
            if let Some(body) = &mut f.body {
                for inst in body.insts.values_mut() {
                    m.tick(1)?;
                    self.inst(inst)?;
                }
            }
            new_funcs.push(f, m)?;
        }
        for md in metadata.values_mut() {
            if let Md::Value(c) = md {
                *c = self.konst(*c)?;
            }
        }
        for v in values.iter_mut() {
            v.id =
                v.id.and_then(|c| self.consts.get(c.index()).copied().flatten());
        }
        for f in defined.iter_mut() {
            *f = self.func(*f)?;
        }
        for (f, _) in fn_entries.iter_mut() {
            *f = self.func(*f)?;
        }
        let bc_types = bc_types.renumbered(|t| self.types.get(t.index()).copied().flatten());
        Ok(ModuleTables {
            parts: ModuleParts {
                symbols,
                types,
                consts: new_consts,
                globals,
                funcs: new_funcs,
                metadata,
                named_md,
                triple,
                source_filename,
                data_layout,
            },
            types: bc_types,
            values,
            attr_lists,
            md_kinds,
            sync_scopes,
            defined,
            function_blocks,
            fn_entries,
            enum_attrs: enum_attrs
                .renumbered_funcs(|f| self.funcs.get(f.index()).copied().flatten()),
        })
    }
}
