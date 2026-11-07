//! Rebuilding one module's functions and globals into other tables (pruning, linking).
//!
//! [`Renumber`] maps the types, constants, globals, functions, and symbols that the source parts
//! name into destination tables: fresh ones when pruning renumbers a module densely
//! ([`Syms::Same`]: the pruned module keeps its symbol table), or another module's when linking
//! imports a linked function into it ([`Syms::Import`]: each symbol is interned into the
//! destination's table on first use). Types and constants are interned children first on first
//! use, so the destination's ids follow the order of the walk that asks for them, and one fixed
//! walk gives one result. Globals and functions are mapped through tables the caller decides
//! (`gmap`, `fmap`).
//!
//! Linking maps types as LLVM's IR linker does: [`Renumber::pair`] first pairs a linked
//! definition's type with the destination type of the reference it resolves, structurally
//! (identified structs by body, not name), and with [`Renumber::merging`] any other identified
//! struct of the source becomes the destination's first identified struct of the same body. Struct
//! names stay diagnostics; no decision branches on them.

use std::collections::HashMap;

use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::{AgxOp, LlvmIntrinsic, Template};
use crate::ir::{Arena, Block, Body, Call, CallAttrs, Callee, Const, ConstExpr, ConstId};
use crate::ir::{ConstTable, Constant, FnAttrs, FuncId, Function, Global, GlobalId, Idx};
use crate::ir::{Inst, InstKind, ParamAttrs, Section, Sym, Symbols, Type, TypeId};
use crate::ir::{TypeTable, Value};

use super::rewrite::{filled, invariant};

/// Call `f` on every constant, function, and type an instruction names directly.
pub(super) fn inst_refs(
    kind: &InstKind,
    ty: TypeId,
    mut on_const: impl FnMut(ConstId),
    mut on_func: impl FnMut(FuncId),
    mut on_type: impl FnMut(TypeId),
) {
    on_type(ty);
    kind.for_each_value(|v| {
        if let Value::Const(c) = v {
            on_const(c);
        }
    });
    match kind {
        InstKind::Alloca { ty, .. } => on_type(*ty),
        InstKind::Gep { src_ty, .. } => on_type(*src_ty),
        InstKind::Call(call) => {
            match call.callee {
                Callee::Indirect { fn_ty, .. } => on_type(fn_ty),
                ref c => {
                    if let Some(f) = c.func() {
                        on_func(f);
                    }
                }
            }
            if let Some(a) = &call.attrs {
                a.ret.types().for_each(&mut on_type);
                a.params
                    .iter()
                    .flat_map(ParamAttrs::types)
                    .for_each(&mut on_type);
            }
        }
        _ => {}
    }
}

/// How symbols map from the source parts to the destination.
pub(super) enum Syms<'a> {
    /// The destination keeps the source's symbol table (pruning).
    Same,
    /// Every symbol is interned into `into` on first use (linking a module into another).
    Import {
        from: &'a Symbols,
        into: Symbols,
        map: Vec<Option<Sym>>,
    },
}

impl<'a> Syms<'a> {
    /// Symbols of `from` interned into `into`, with the map charged to `held`.
    pub fn import(
        from: &'a Symbols,
        into: Symbols,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<Syms<'a>, Error> {
        Ok(Syms::Import {
            from,
            into,
            map: filled(m, held, from.len(), None)?,
        })
    }

    /// The destination symbol of `s`.
    fn map(&mut self, s: Sym, m: &mut Meter) -> Result<Sym, Error> {
        match self {
            Syms::Same => Ok(s),
            Syms::Import { from, into, map } => {
                let slot = map
                    .get_mut(s.index())
                    .ok_or_else(|| invariant("symbol out of range"))?;
                if let Some(done) = *slot {
                    return Ok(done);
                }
                let bytes = from
                    .get(s)
                    .ok_or_else(|| invariant("symbol out of range"))?;
                let new = into.intern(bytes, m)?;
                *slot = Some(new);
                Ok(new)
            }
        }
    }
}

/// The destination's identified structs by body (fields and packedness): the first of each.
pub(super) type Bodies = HashMap<(Box<[TypeId]>, bool), TypeId>;

/// The identified structs of `types` by body, the lowest id of each body, charged to `held`.
pub(super) fn bodies(types: &TypeTable, m: &mut Meter, held: &mut usize) -> Result<Bodies, Error> {
    let mut out = Bodies::new();
    for (id, ty) in types.iter() {
        m.tick(1)?;
        if let Type::Struct {
            fields,
            packed,
            name: Some(_),
        } = ty
        {
            if let std::collections::hash_map::Entry::Vacant(e) =
                out.entry((fields.clone(), *packed))
            {
                let bytes = body_bytes(fields);
                m.charge(bytes)?;
                *held = held.saturating_add(bytes);
                e.insert(id);
            }
        }
    }
    Ok(out)
}

/// Whether `a` of `from` and `b` of `to` have the same shape, with identified structs matched by
/// body whatever their names, given the images `known` already fixes: the map of each type `a`
/// contains to its counterpart in `b`, or `None` when they differ or a contained type would need
/// two images.
pub(super) fn isomorphism(
    (from, to): (&TypeTable, &TypeTable),
    (a, b): (TypeId, TypeId),
    known: impl Fn(TypeId) -> Option<TypeId>,
    m: &mut Meter,
) -> Result<Option<HashMap<TypeId, TypeId>>, Error> {
    let mut map: HashMap<TypeId, TypeId> = HashMap::new();
    let mut stack = vec![(a, b)];
    while let Some((s, d)) = stack.pop() {
        m.tick(1)?;
        if let Some(image) = known(s).or_else(|| map.get(&s).copied()) {
            if image != d {
                return Ok(None);
            }
            continue;
        }
        let (Some(st), Some(dt)) = (from.get(s), to.get(d)) else {
            return Err(invariant("type out of range"));
        };
        let pairs: Vec<(TypeId, TypeId)> = match (st, dt) {
            (
                Type::Struct {
                    fields: x,
                    packed: px,
                    name: nx,
                },
                Type::Struct {
                    fields: y,
                    packed: py,
                    name: ny,
                },
            ) if px == py && x.len() == y.len() && nx.is_some() == ny.is_some() => {
                x.iter().copied().zip(y.iter().copied()).collect()
            }
            (Type::OpaqueStruct { .. }, Type::OpaqueStruct { .. }) => Vec::new(),
            (Type::Vector { elem: x, len: lx }, Type::Vector { elem: y, len: ly }) if lx == ly => {
                vec![(*x, *y)]
            }
            (Type::Array { elem: x, len: lx }, Type::Array { elem: y, len: ly }) if lx == ly => {
                vec![(*x, *y)]
            }
            (
                Type::Function {
                    ret: rx,
                    params: x,
                    vararg: vx,
                },
                Type::Function {
                    ret: ry,
                    params: y,
                    vararg: vy,
                },
            ) if vx == vy && x.len() == y.len() => std::iter::once((*rx, *ry))
                .chain(x.iter().copied().zip(y.iter().copied()))
                .collect(),
            (x, y) if x.children().is_empty() && x == y => Vec::new(),
            _ => return Ok(None),
        };
        m.charge(std::mem::size_of::<(TypeId, TypeId)>().saturating_mul(pairs.len() + 1))?;
        map.insert(s, d);
        stack.extend(pairs);
    }
    Ok(Some(map))
}

fn body_bytes(fields: &[TypeId]) -> usize {
    std::mem::size_of::<((Box<[TypeId]>, bool), TypeId)>()
        .saturating_add(std::mem::size_of_val(fields))
}

/// Rebuilds types, constants, globals, and functions of one module into destination tables.
pub(super) struct Renumber<'a> {
    old_types: &'a TypeTable,
    old_consts: &'a ConstTable,
    /// The destination tables.
    pub types: TypeTable,
    pub consts: ConstTable,
    tmap: Vec<Option<TypeId>>,
    cmap: Vec<Option<ConstId>>,
    gmap: &'a [Option<GlobalId>],
    fmap: &'a [Option<FuncId>],
    pub syms: Syms<'a>,
    /// When merging (linking), the destination's identified structs by body.
    pub bodies: Option<Bodies>,
}

impl<'a> Renumber<'a> {
    /// A renumbering of the source's types and constants (`from`) into `into` (fresh tables or
    /// another module's), mapping globals and functions through `maps` and symbols through
    /// `syms`. The memo tables are charged to `held`.
    pub fn new(
        (old_types, old_consts): (&'a TypeTable, &'a ConstTable),
        (types, consts): (TypeTable, ConstTable),
        (gmap, fmap): (&'a [Option<GlobalId>], &'a [Option<FuncId>]),
        syms: Syms<'a>,
        m: &mut Meter,
        held: &mut usize,
    ) -> Result<Renumber<'a>, Error> {
        Ok(Renumber {
            old_types,
            old_consts,
            types,
            consts,
            tmap: filled(m, held, old_types.len(), None)?,
            cmap: filled(m, held, old_consts.len(), None)?,
            gmap,
            fmap,
            syms,
            bodies: None,
        })
    }

    /// This renumbering, making each identified struct of the source the destination's struct
    /// of `bodies` with the same body when there is one (LLVM's IR linker does so), and adding
    /// the structs it interns to `bodies`.
    pub fn merging(self, bodies: Bodies) -> Renumber<'a> {
        Renumber {
            bodies: Some(bodies),
            ..self
        }
    }

    /// Map source type `src` to destination type `dst` if they are isomorphic ([`isomorphism`]):
    /// every type `src` contains is mapped to the corresponding part of `dst`, and later
    /// renumbering keeps these (LLVM's IR linker maps a declaration's type to its definition's
    /// so). `false` (nothing mapped) when they are not. Call it before renumbering anything that
    /// contains `src`.
    pub fn pair(&mut self, src: TypeId, dst: TypeId, m: &mut Meter) -> Result<bool, Error> {
        let tmap = &self.tmap;
        let known = |t: TypeId| tmap.get(t.index()).copied().flatten();
        let Some(pairs) = isomorphism((self.old_types, &self.types), (src, dst), known, m)? else {
            return Ok(false);
        };
        for (s, d) in pairs {
            if let Some(slot) = self.tmap.get_mut(s.index()) {
                *slot = Some(d);
            }
        }
        Ok(true)
    }

    fn sym(&mut self, s: Sym, m: &mut Meter) -> Result<Sym, Error> {
        self.syms.map(s, m)
    }

    /// The new id of old type `t`. Iterative: types nest as deep as the input makes them.
    pub fn ty(&mut self, t: TypeId, m: &mut Meter) -> Result<TypeId, Error> {
        let mut stack = vec![(t, false)];
        while let Some((cur, expanded)) = stack.pop() {
            m.tick(1)?;
            if self.tmap.get(cur.index()).copied().flatten().is_some() {
                continue;
            }
            let old = self
                .old_types
                .get(cur)
                .ok_or_else(|| invariant("type out of range"))?;
            if !expanded {
                stack.push((cur, true));
                stack.extend(old.children().into_iter().rev().map(|c| (c, false)));
                continue;
            }
            let name = match old {
                Type::Struct { name, .. } | Type::OpaqueStruct { name } => *name,
                _ => None,
            };
            let name = name.map(|s| self.sym(s, m)).transpose()?;
            let map = |c: TypeId| -> Result<TypeId, Error> {
                self.tmap
                    .get(c.index())
                    .copied()
                    .flatten()
                    .ok_or_else(|| invariant("type child not renumbered"))
            };
            let new = match old {
                Type::Vector { elem, len } => Type::Vector {
                    elem: map(*elem)?,
                    len: *len,
                },
                Type::Array { elem, len } => Type::Array {
                    elem: map(*elem)?,
                    len: *len,
                },
                Type::Struct { fields, packed, .. } => Type::Struct {
                    fields: fields.iter().map(|&f| map(f)).collect::<Result<_, _>>()?,
                    packed: *packed,
                    name,
                },
                Type::OpaqueStruct { .. } => Type::OpaqueStruct { name },
                Type::Function {
                    ret,
                    params,
                    vararg,
                } => Type::Function {
                    ret: map(*ret)?,
                    params: params.iter().map(|&p| map(p)).collect::<Result<_, _>>()?,
                    vararg: *vararg,
                },
                other => other.clone(),
            };
            let id = match (&new, &mut self.bodies) {
                (
                    Type::Struct {
                        fields,
                        packed,
                        name: Some(_),
                    },
                    Some(bodies),
                ) => match bodies.get(&(fields.clone(), *packed)) {
                    Some(&same) => same,
                    None => {
                        let key = (fields.clone(), *packed);
                        let bytes = body_bytes(fields);
                        let id = self.types.intern(new, m)?;
                        m.charge(bytes)?;
                        bodies.insert(key, id);
                        id
                    }
                },
                _ => self.types.intern(new, m)?,
            };
            if let Some(slot) = self.tmap.get_mut(cur.index()) {
                *slot = Some(id);
            }
        }
        self.tmap
            .get(t.index())
            .copied()
            .flatten()
            .ok_or_else(|| invariant("type not renumbered"))
    }

    /// The new id of old constant `c`, children first.
    pub fn constant(&mut self, c: ConstId, m: &mut Meter) -> Result<ConstId, Error> {
        let mut stack = vec![(c, false)];
        while let Some((cur, expanded)) = stack.pop() {
            m.tick(1)?;
            if self.cmap.get(cur.index()).copied().flatten().is_some() {
                continue;
            }
            let old = self
                .old_consts
                .get(cur)
                .ok_or_else(|| invariant("constant out of range"))?;
            if !expanded {
                stack.push((cur, true));
                stack.extend(old.value.children().iter().rev().map(|&k| (k, false)));
                continue;
            }
            let cmap = &self.cmap;
            let map = |k: ConstId| -> Result<ConstId, Error> {
                cmap.get(k.index())
                    .copied()
                    .flatten()
                    .ok_or_else(|| invariant("constant child not renumbered"))
            };
            let value = match &old.value {
                Const::Aggregate(elems) => {
                    Const::Aggregate(elems.iter().map(|&e| map(e)).collect::<Result<_, _>>()?)
                }
                Const::GlobalAddr(g) => Const::GlobalAddr(
                    self.gmap
                        .get(g.index())
                        .copied()
                        .flatten()
                        .ok_or_else(|| invariant("global not kept"))?,
                ),
                Const::FuncAddr(f) => Const::FuncAddr(
                    self.fmap
                        .get(f.index())
                        .copied()
                        .flatten()
                        .ok_or_else(|| invariant("function not kept"))?,
                ),
                Const::Expr(ConstExpr::Cast { op, ops: [v] }) => Const::Expr(ConstExpr::Cast {
                    op: *op,
                    ops: [map(*v)?],
                }),
                Const::Expr(ConstExpr::BinOp {
                    op,
                    flags,
                    ops: [a, b],
                }) => Const::Expr(ConstExpr::BinOp {
                    op: *op,
                    flags: *flags,
                    ops: [map(*a)?, map(*b)?],
                }),
                Const::Expr(ConstExpr::Gep {
                    src_ty,
                    inbounds,
                    ops,
                }) => {
                    let ops = ops.iter().map(|&o| map(o)).collect::<Result<_, _>>()?;
                    Const::Expr(ConstExpr::Gep {
                        src_ty: self.ty(*src_ty, m)?,
                        inbounds: *inbounds,
                        ops,
                    })
                }
                other => other.clone(),
            };
            let ty = self.ty(old.ty, m)?;
            let id = self.consts.intern(Constant { ty, value }, m)?;
            if let Some(slot) = self.cmap.get_mut(cur.index()) {
                *slot = Some(id);
            }
        }
        self.cmap
            .get(c.index())
            .copied()
            .flatten()
            .ok_or_else(|| invariant("constant not renumbered"))
    }

    fn func_id(&self, f: FuncId) -> Result<FuncId, Error> {
        self.fmap
            .get(f.index())
            .copied()
            .flatten()
            .ok_or_else(|| invariant("callee not kept"))
    }

    fn strings(
        &mut self,
        strings: &[(Sym, Sym)],
        m: &mut Meter,
    ) -> Result<Box<[(Sym, Sym)]>, Error> {
        let mut out = Vec::with_capacity(strings.len());
        for &(k, v) in strings {
            out.push((self.sym(k, m)?, self.sym(v, m)?));
        }
        Ok(out.into_boxed_slice())
    }

    fn fn_attrs(&mut self, a: &FnAttrs, m: &mut Meter) -> Result<FnAttrs, Error> {
        Ok(FnAttrs {
            strings: self.strings(&a.strings, m)?,
            ..a.clone()
        })
    }

    fn param_attrs(&mut self, a: &ParamAttrs, m: &mut Meter) -> Result<ParamAttrs, Error> {
        Ok(ParamAttrs {
            byval: a.byval.map(|t| self.ty(t, m)).transpose()?,
            sret: a.sret.map(|t| self.ty(t, m)).transpose()?,
            strings: self.strings(&a.strings, m)?,
            ..a.clone()
        })
    }

    fn section(&mut self, s: Option<Section>, m: &mut Meter) -> Result<Option<Section>, Error> {
        Ok(match s {
            Some(Section::Other(sym)) => Some(Section::Other(self.sym(sym, m)?)),
            other => other,
        })
    }

    /// Old global `g` in the destination.
    pub fn global(&mut self, g: &Global, m: &mut Meter) -> Result<Global, Error> {
        Ok(Global {
            name: self.sym(g.name, m)?,
            value_ty: self.ty(g.value_ty, m)?,
            init: g.init.map(|c| self.constant(c, m)).transpose()?,
            section: self.section(g.section, m)?,
            ..g.clone()
        })
    }

    /// Old function `f`, and its body, in the destination.
    pub fn function(&mut self, f: &Function, m: &mut Meter) -> Result<Function, Error> {
        let ty = self.ty(f.ty, m)?;
        let ret_attrs = self.param_attrs(&f.ret_attrs, m)?;
        let mut params = Vec::with_capacity(f.params.len());
        for p in f.params.iter() {
            let mut p = p.clone();
            p.attrs = self.param_attrs(&p.attrs, m)?;
            p.name = p.name.map(|s| self.sym(s, m)).transpose()?;
            params.push(p);
        }
        let body = f.body.as_ref().map(|b| self.body(b, m)).transpose()?;
        Ok(Function {
            name: self.sym(f.name, m)?,
            ty,
            linkage: f.linkage,
            visibility: f.visibility,
            unnamed_addr: f.unnamed_addr,
            cconv: f.cconv,
            attrs: self.fn_attrs(&f.attrs, m)?,
            ret_attrs,
            params: params.into_boxed_slice(),
            body,
            section: self.section(f.section, m)?,
            align: f.align,
        })
    }

    fn body(&mut self, b: &Body, m: &mut Meter) -> Result<Body, Error> {
        let mut blocks = Arena::new();
        for block in b.blocks.values() {
            let name = block.name.map(|s| self.sym(s, m)).transpose()?;
            blocks.push(Block { name, ..*block }, m)?;
        }
        let mut out = Body {
            blocks,
            insts: Arena::new(),
        };
        for inst in b.insts.values() {
            m.tick(1)?;
            let mut kind = inst.kind.clone();
            let mut failed = None;
            let mut consts = Vec::new();
            kind.for_each_value(|v| {
                if let Value::Const(c) = v {
                    consts.push(c);
                }
            });
            let mut mapped = Vec::with_capacity(consts.len());
            for c in consts {
                mapped.push(self.constant(c, m)?);
            }
            let mut next = mapped.into_iter();
            kind.for_each_value_mut(|v| {
                if let Value::Const(_) = v {
                    match next.next() {
                        Some(c) => *v = Value::Const(c),
                        None => {
                            failed.get_or_insert(invariant("operand count changed"));
                        }
                    }
                }
            });
            if let Some(e) = failed {
                return Err(e);
            }
            match &mut kind {
                InstKind::Alloca { ty, .. } => *ty = self.ty(*ty, m)?,
                InstKind::Gep { src_ty, .. } => *src_ty = self.ty(*src_ty, m)?,
                InstKind::Call(call) => self.call(call, m)?,
                _ => {}
            }
            let ty = self.ty(inst.ty, m)?;
            out.push_inst(Inst { ty, kind }, m)?;
        }
        Ok(out)
    }

    /// `op` with its symbol renumbered: an unmodeled AGX operation is the one that keeps its
    /// name, for diagnostics.
    fn llvm_op(&mut self, op: LlvmIntrinsic, m: &mut Meter) -> Result<LlvmIntrinsic, Error> {
        Ok(match op {
            LlvmIntrinsic::Agx(AgxOp::Unmodeled(s)) => {
                LlvmIntrinsic::Agx(AgxOp::Unmodeled(self.sym(s, m)?))
            }
            other => other,
        })
    }

    fn call(&mut self, call: &mut Call, m: &mut Meter) -> Result<(), Error> {
        call.callee = match call.callee {
            Callee::Direct(f) => Callee::Direct(self.func_id(f)?),
            Callee::Indirect { callee, fn_ty } => Callee::Indirect {
                callee,
                fn_ty: self.ty(fn_ty, m)?,
            },
            Callee::Air { decl, op } => Callee::Air {
                decl: self.func_id(decl)?,
                op,
            },
            Callee::Llvm { decl, op } => Callee::Llvm {
                decl: self.func_id(decl)?,
                op: self.llvm_op(op, m)?,
            },
            Callee::UnknownIntrinsic(f) => Callee::UnknownIntrinsic(self.func_id(f)?),
            Callee::Unbound { decl, template } => Callee::Unbound {
                decl: self.func_id(decl)?,
                template: match template {
                    Template::Llvm(op) => Template::Llvm(self.llvm_op(op, m)?),
                    air => air,
                },
            },
        };
        if let Some(a) = &call.attrs {
            let mut params = Vec::with_capacity(a.params.len());
            for p in a.params.iter() {
                params.push(self.param_attrs(p, m)?);
            }
            call.attrs = Some(Box::new(CallAttrs {
                func: self.fn_attrs(&a.func, m)?,
                ret: self.param_attrs(&a.ret, m)?,
                params: params.into_boxed_slice(),
            }));
        }
        Ok(())
    }
}
