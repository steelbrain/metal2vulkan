//! MSL 4.1's generic pointers, each given the one address space its pointers come from.
//!
//! A pointer of the generic space (`addrspace(8)`) is a device, thread or threadgroup pointer a
//! cast made generic: it keeps pointing into that space, and
//! `air.is_{device,thread,threadgroup}_pointer` asks which. Vulkan's logical addressing has no
//! pointer that may point into either of two storage classes, so each generic pointer of the
//! inlined entry becomes a pointer of the one space every pointer it may hold comes from.
//!
//! Generic pointers that must share a type share a space: a `getelementptr`, `bitcast`,
//! `addrspacecast` or `freeze` of a generic pointer and its operand, a `phi` or `select` and its
//! operands, the two sides of an `icmp`, the generic pointers stored into and loaded from one
//! slot of thread memory whose every access the translation sees, and a slot and the slot a copy
//! of its bytes moves it to ([`cells`]), and a generic
//! pointer an aggregate or a vector value holds (a *leaf*) and what it comes from ([`leaves`]).
//! The links are the union-find of [`plan()`]. Each such group has one space where every pointer
//! entering it does:
//!
//! - an `addrspacecast` from a pointer of a device, thread or threadgroup space brings in that
//!   space, and so does a constant expression of such a cast;
//! - a null, undefined or poison constant brings in none: it points nowhere;
//! - a generic pointer a call, an `inttoptr`, an atomic or a load from other memory gives, or a
//!   parameter, brings in a space only the run knows, and so does an aggregate a load or a call
//!   gives for each of its leaves.
//!
//! A group that pointers of two spaces enter, or one a space only the run knows, is
//! `Unsupported{GenericPointer(..)}` at the instruction where it does, naming how
//! ([`GenericPointer`]): no Vulkan pointer holds either. So is a generic pointer passed to a call
//! other than a query (the callee's type names the generic space) or returned, alone or in an
//! aggregate (`Call`).
//!
//! Each generic pointer then is its group's pointer type, and a group that no space enters a
//! thread pointer (it addresses nothing); an aggregate or a vector value's type is its type with
//! each leaf its group's pointer. A cast into the generic space is its operand. A cast out of it
//! is its operand where it names the group's space, and poison where it names another (an
//! `addrspacecast` of a pointer not of the space it names is poison). Every other instruction
//! keeps its operation, its generic constants those of its group's type (a store's, its value's
//! group's; an aggregate constant's, its leaves'). A query is false for another space; for the
//! group's own space it is true, or, where a null constant may reach the pointer (through thread
//! memory and aggregates too), whether the pointer is not null, as Metal answers no space for a
//! null generic pointer. That answer is the pointer's structure's where it decides it: a
//! variable's or a global's address, a cast of one or an in-bounds step from one is not null, a
//! null constant is, and a `select` between two such pointers is not null where its condition
//! picks the one that is not. Else it is a comparison with the null pointer of the group's space.
//! A generic pointer that only answered queries read goes, with what only it reads. Aggregate
//! types that hold generic pointers in memory (an `alloca`'s or a `getelementptr`'s source type)
//! keep them: they lay out bytes, and a pointer of every space is 64 bits.

use crate::budget::Meter;
use crate::error::{DiagName, Error, Feature, GenericPointer, Location, Unsupported};
use crate::intrinsics::AirIntrinsic;
use crate::ir::{AddrSpace, Body, Callee, CastOp, Const, ConstExpr, ConstId, Constant, FastMath};
use crate::ir::{FuncId, Idx, InstId, InstKind, IntPredicate, ModuleParts, Type, TypeId, Value};

use super::rewrite::{apply, filled, invariant, scoped, BodyEdit};

mod cells;
mod leaves;
mod plan;

use plan::{plan, Plan};

/// The most aggregate levels searched for a generic pointer in a type.
const DEPTH: u32 = 32;

/// The pointers a group of generic pointers may hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Space {
    /// None yet: only constants that point nowhere.
    Unseen,
    /// Pointers of this space.
    One(AddrSpace),
    /// Pointers of two spaces, or of a space only the run knows: how, and the instruction where
    /// it entered, where one did.
    Many(GenericPointer, Option<InstId>),
}

impl Space {
    /// The space of a group that both `self` and `other` enter, at instruction `at`.
    fn join(self, other: Space, at: Option<InstId>) -> Space {
        match (self, other) {
            (Space::Unseen, s) | (s, Space::Unseen) => s,
            (Space::One(a), Space::One(b)) if a == b => Space::One(a),
            (m @ Space::Many(..), _) | (_, m @ Space::Many(..)) => m,
            _ => Space::Many(GenericPointer::TwoSpaces, at),
        }
    }
}

/// A generic pointer refused for `how`, where no instruction locates it.
fn refused(how: GenericPointer) -> Error {
    Unsupported::not_implemented(Feature::GenericPointer(how)).into()
}

/// The entry's name, body and parameter types, and the module's tables.
struct Cx<'a> {
    name: &'a DiagName,
    body: &'a Body,
    parts: &'a ModuleParts,
    params: &'a [TypeId],
}

impl Cx<'_> {
    /// A generic pointer refused for `how`, at instruction `at` of the entry.
    fn refuse(&self, how: GenericPointer, at: InstId) -> Error {
        let block = self
            .body
            .blocks
            .iter()
            .find(|(_, b)| b.start <= at && at < b.end)
            .map(|(b, _)| b.raw());
        Unsupported::not_implemented(Feature::GenericPointer(how))
            .at(Location::function(self.name.clone(), block, Some(at.raw())))
            .into()
    }

    fn generic(&self, ty: TypeId) -> bool {
        matches!(
            self.parts.types.get(ty),
            Some(Type::Ptr {
                space: AddrSpace::Generic
            })
        )
    }

    fn ty(&self, v: Value) -> Option<TypeId> {
        match v {
            Value::Inst(i) => self.body.insts.get(i).map(|inst| inst.ty),
            Value::Const(c) => self.parts.consts.ty(c),
            Value::Arg(a) => self.params.get(a.index()).copied(),
        }
    }

    fn is_generic(&self, v: Value) -> bool {
        self.ty(v).is_some_and(|t| self.generic(t))
    }

    /// Whether `ty` holds a generic pointer below an aggregate or vector level.
    fn holds_generic(&self, ty: TypeId, depth: u32) -> Result<bool, Error> {
        if depth > DEPTH {
            return Err(refused(GenericPointer::Aggregate));
        }
        Ok(match self.parts.types.get(ty) {
            Some(Type::Vector { elem, .. } | Type::Array { elem, .. }) => {
                self.generic(*elem) || self.holds_generic(*elem, depth + 1)?
            }
            Some(Type::Struct { fields, .. }) => {
                let mut any = false;
                for &f in fields.iter() {
                    any |= self.generic(f) || self.holds_generic(f, depth + 1)?;
                }
                any
            }
            _ => false,
        })
    }

    /// The space a generic constant brings in, and whether it is null.
    fn constant(&self, c: ConstId, depth: u32) -> Result<(Space, bool), Error> {
        if depth > DEPTH {
            return Err(refused(GenericPointer::Integer));
        }
        let k = self
            .parts
            .consts
            .get(c)
            .ok_or_else(|| invariant("a constant out of range"))?;
        Ok(match &k.value {
            Const::Null => (Space::Unseen, true),
            Const::Undef | Const::Poison => (Space::Unseen, false),
            Const::Expr(ConstExpr::Cast {
                op: CastOp::AddrSpaceCast | CastOp::Bitcast,
                ops: [o],
            }) => {
                let ty = self
                    .parts
                    .consts
                    .ty(*o)
                    .ok_or_else(|| invariant("a constant"))?;
                match self.parts.types.get(ty) {
                    Some(Type::Ptr {
                        space: AddrSpace::Generic,
                    }) => self.constant(*o, depth + 1)?,
                    Some(Type::Ptr { space }) => (
                        Space::One(*space),
                        matches!(
                            self.parts.consts.get(*o).map(|k| &k.value),
                            Some(Const::Null)
                        ),
                    ),
                    _ => (Space::Many(GenericPointer::Integer, None), false),
                }
            }
            Const::Expr(ConstExpr::Gep { ops, .. }) => match ops.first() {
                Some(&base) => self.constant(base, depth + 1)?,
                None => (Space::Many(GenericPointer::Integer, None), false),
            },
            _ => (Space::Many(GenericPointer::Integer, None), false),
        })
    }

    /// Whether pointer `v` (of any space) is not null where its structure decides it: a variable's
    /// or a global's address, a cast of one, or an in-bounds step from one is not null; a null,
    /// undefined or poison constant points nowhere. `None` where only the run knows.
    fn nonnull(&self, v: Value, depth: u32) -> Option<bool> {
        if depth > DEPTH {
            return None;
        }
        match v {
            Value::Const(c) => self.nonnull_constant(c, depth),
            Value::Arg(_) => None,
            Value::Inst(i) => match &self.body.insts.get(i)?.kind {
                InstKind::Alloca { .. } => Some(true),
                InstKind::Gep {
                    base,
                    inbounds: true,
                    ..
                } => self.nonnull(*base, depth + 1),
                InstKind::Cast {
                    op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                    v,
                }
                | InstKind::Freeze(v) => self.nonnull(*v, depth + 1),
                _ => None,
            },
        }
    }

    /// [`Cx::nonnull`] of constant `c`.
    fn nonnull_constant(&self, c: ConstId, depth: u32) -> Option<bool> {
        if depth > DEPTH {
            return None;
        }
        match &self.parts.consts.get(c)?.value {
            Const::Null | Const::Undef | Const::Poison => Some(false),
            Const::GlobalAddr(_) => Some(true),
            Const::Expr(ConstExpr::Cast {
                op: CastOp::AddrSpaceCast | CastOp::Bitcast,
                ops: [o],
            }) => self.nonnull_constant(*o, depth + 1),
            Const::Expr(ConstExpr::Gep {
                inbounds: true,
                ops,
                ..
            }) => self.nonnull_constant(*ops.first()?, depth + 1),
            _ => None,
        }
    }

    /// Whether generic pointer `p` is not null, as far as its structure decides it: by itself, or
    /// as a `select` between two pointers it decides.
    fn not_null(&self, p: Value) -> NotNull {
        if let Some(b) = self.nonnull(p, 0) {
            return NotNull::Is(b);
        }
        if let Value::Inst(i) = p {
            if let Some(InstKind::Select { c, t, f, .. }) = self.body.insts.get(i).map(|x| &x.kind)
            {
                if let (Some(t), Some(f)) = (self.nonnull(*t, 0), self.nonnull(*f, 0)) {
                    return if t == f {
                        NotNull::Is(t)
                    } else {
                        NotNull::When { c: *c, t }
                    };
                }
            }
        }
        NotNull::Run
    }

    /// The space and nullness a generic operand brings in by itself: a constant's, `Many` for a
    /// parameter, nothing for an instruction (its group carries it).
    fn operand(&self, v: Value) -> Result<(Space, bool), Error> {
        match v {
            Value::Const(c) => self.constant(c, 0),
            Value::Arg(_) => Ok((Space::Many(GenericPointer::Call, None), false)),
            Value::Inst(_) => Ok((Space::Unseen, false)),
        }
    }
}

/// Whether a generic pointer is not null.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NotNull {
    /// Its structure decides it.
    Is(bool),
    /// A `select` on `c` decides it: `t` where `c` holds, the other answer where not.
    When { c: Value, t: bool },
    /// Only the run knows.
    Run,
}

/// `body` with every generic pointer a pointer of its group's space (module docs), and whether a
/// query folded (SCCP then folds what it decides). `func` is the entry whose body it is.
pub(super) fn resolve(
    body: Body,
    parts: &mut ModuleParts,
    func: FuncId,
    name: &DiagName,
    m: &mut Meter,
) -> Result<(Body, bool), Error> {
    let params: Vec<TypeId> = match parts.funcs.get(func).and_then(|f| parts.types.get(f.ty)) {
        Some(Type::Function { params, .. }) => params.to_vec(),
        _ => return Err(invariant("the entry's type is not a function")),
    };
    let planned = scoped(m, |m, held| {
        let cx = Cx {
            name,
            body: &body,
            parts,
            params: &params,
        };
        plan(&cx, m, held)
    })?;
    let Some(plan) = planned else {
        return Ok((body, false));
    };
    let edited = scoped(m, |m, held| {
        rewrite(&body, parts, (name, &params), &plan, m, held)
    })?;
    body.release(m);
    Ok((edited, plan.folds))
}

/// Whether `kind` is a query or a comparison of generic pointers, whose space is its pointer's,
/// or a store of a generic pointer, whose space is its value's.
fn shares(cx: &Cx<'_>, kind: &InstKind) -> bool {
    match kind {
        InstKind::Call(call) => matches!(
            call.callee,
            Callee::Air {
                op: AirIntrinsic::PointerSpace(_),
                ..
            }
        ),
        InstKind::ICmp { a, b, .. } => cx.is_generic(*a) && cx.is_generic(*b),
        InstKind::Store { val, .. } => cx.is_generic(*val),
        _ => false,
    }
}

/// The interned pointer types and constants the rewrite names.
struct Interned {
    ptrs: Vec<(AddrSpace, TypeId)>,
}

impl Interned {
    fn ptr(
        &mut self,
        parts: &mut ModuleParts,
        s: AddrSpace,
        m: &mut Meter,
    ) -> Result<TypeId, Error> {
        if let Some(&(_, t)) = self.ptrs.iter().find(|(x, _)| *x == s) {
            return Ok(t);
        }
        let t = parts.types.intern(Type::Ptr { space: s }, m)?;
        self.ptrs.push((s, t));
        Ok(t)
    }
}

/// Generic constant `c` as a constant of space `s`.
fn retyped(
    parts: &mut ModuleParts,
    interned: &mut Interned,
    c: ConstId,
    s: AddrSpace,
    depth: u32,
    m: &mut Meter,
) -> Result<ConstId, Error> {
    if depth > DEPTH {
        return Err(refused(GenericPointer::Integer));
    }
    let k = parts
        .consts
        .get(c)
        .cloned()
        .ok_or_else(|| invariant("a constant out of range"))?;
    let ty = interned.ptr(parts, s, m)?;
    let value = match k.value {
        v @ (Const::Null | Const::Undef | Const::Poison) => v,
        Const::Expr(ConstExpr::Cast {
            op: CastOp::AddrSpaceCast | CastOp::Bitcast,
            ops: [o],
        }) => {
            let oty = parts.consts.ty(o).ok_or_else(|| invariant("a constant"))?;
            return match parts.types.get(oty) {
                Some(Type::Ptr {
                    space: AddrSpace::Generic,
                }) => retyped(parts, interned, o, s, depth + 1, m),
                Some(Type::Ptr { space }) if *space == s => Ok(o),
                Some(Type::Ptr { .. }) => Err(refused(GenericPointer::TwoSpaces)),
                _ => Err(refused(GenericPointer::Integer)),
            };
        }
        Const::Expr(ConstExpr::Gep {
            src_ty,
            inbounds,
            ops,
        }) => {
            let (&base, rest) = ops
                .split_first()
                .ok_or_else(|| refused(GenericPointer::Integer))?;
            let base = retyped(parts, interned, base, s, depth + 1, m)?;
            let ops: Box<[ConstId]> = std::iter::once(base).chain(rest.iter().copied()).collect();
            Const::Expr(ConstExpr::Gep {
                src_ty,
                inbounds,
                ops,
            })
        }
        _ => return Err(refused(GenericPointer::Integer)),
    };
    parts.consts.intern(Constant { ty, value }, m)
}

/// The space a group of `space` takes: its own, or thread where none entered it.
fn taken(space: Space) -> AddrSpace {
    match space {
        Space::One(s) => s,
        _ => AddrSpace::Thread,
    }
}

fn rewrite(
    body: &Body,
    parts: &mut ModuleParts,
    (name, params): (&DiagName, &[TypeId]),
    plan: &Plan,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Body, Error> {
    let mut edit = BodyEdit::keep_all(body, m, held)?;
    let mut interned = Interned { ptrs: Vec::new() };
    let i1 = parts.types.intern(Type::Int(1), m)?;
    let boolean = |parts: &mut ModuleParts, b: bool, m: &mut Meter| {
        parts.consts.intern(
            Constant {
                ty: i1,
                value: Const::Int(u64::from(b)),
            },
            m,
        )
    };
    for (i, inst) in body.insts.iter() {
        m.tick(1)?;
        let generic_of = |parts: &ModuleParts, v: Value| -> bool {
            let ty = match v {
                Value::Inst(j) => body.insts.get(j).map(|x| x.ty),
                Value::Const(c) => parts.consts.ty(c),
                Value::Arg(a) => params.get(a.index()).copied(),
            };
            matches!(
                ty.and_then(|t| parts.types.get(t)),
                Some(Type::Ptr {
                    space: AddrSpace::Generic
                })
            )
        };
        let result_generic = matches!(
            parts.types.get(inst.ty),
            Some(Type::Ptr {
                space: AddrSpace::Generic
            })
        );
        let own = plan.space.get(i.index()).copied().unwrap_or(Space::Unseen);
        // An aggregate or a vector that holds generic pointers: its leaves' spaces.
        if let Some(leaves) = plan.leaves.as_ref() {
            if let Some(at) = leaves.of(i, parts, inst.ty) {
                let mut r = leaves::Retyped {
                    leaves,
                    interned: &mut interned,
                };
                leaves::rewrite(parts, &mut r, &plan.leaf_space, (i, inst), at, &mut edit, m)?;
                continue;
            }
        }
        // A cast into the generic space, or a no-op cast within it, is its operand.
        if let InstKind::Cast {
            op: CastOp::AddrSpaceCast | CastOp::Bitcast,
            v,
        } = inst.kind
        {
            let from_generic = generic_of(parts, v);
            if result_generic {
                let s = taken(own);
                let v = match v {
                    Value::Const(c) if from_generic => {
                        Value::Const(retyped(parts, &mut interned, c, s, 0, m)?)
                    }
                    other => other,
                };
                edit.replace(i, v);
                continue;
            }
            if from_generic {
                // A cast out of the generic space: its operand where it names the group's space,
                // else poison.
                let to = match parts.types.get(inst.ty) {
                    Some(Type::Ptr { space }) => *space,
                    _ => return Err(refused(GenericPointer::Integer)),
                };
                let group = match v {
                    Value::Inst(j) => plan.space.get(j.index()).copied().unwrap_or(Space::Unseen),
                    Value::Const(c) => {
                        Cx {
                            name,
                            body,
                            parts,
                            params,
                        }
                        .constant(c, 0)?
                        .0
                    }
                    Value::Arg(_) => return Err(refused(GenericPointer::Call)),
                };
                let value = match group {
                    // A pointer computed from no pointer, made a pointer of another space.
                    Space::Unseen if to != AddrSpace::Thread && matches!(v, Value::Inst(_)) => {
                        let poison = parts.consts.intern(
                            Constant {
                                ty: inst.ty,
                                value: Const::Poison,
                            },
                            m,
                        )?;
                        Value::Const(poison)
                    }
                    Space::One(s) if s != to => {
                        let poison = parts.consts.intern(
                            Constant {
                                ty: inst.ty,
                                value: Const::Poison,
                            },
                            m,
                        )?;
                        Value::Const(poison)
                    }
                    Space::Many(how, _) => return Err(refused(how)),
                    _ => match v {
                        Value::Const(c) => {
                            Value::Const(retyped(parts, &mut interned, c, to, 0, m)?)
                        }
                        other => other,
                    },
                };
                edit.replace(i, value);
                continue;
            }
        }
        // A query: false for another space; true, or not null, for its group's own.
        if let InstKind::Call(call) = &inst.kind {
            if let Callee::Air {
                op: AirIntrinsic::PointerSpace(q),
                ..
            } = call.callee
            {
                let p = *call
                    .args
                    .first()
                    .ok_or_else(|| invariant("a query's pointer"))?;
                let (s, nullable) = match p {
                    Value::Inst(j) => (
                        plan.space.get(j.index()).copied().unwrap_or(Space::Unseen),
                        plan.null.get(j.index()).copied().unwrap_or(true),
                    ),
                    Value::Const(c) => Cx {
                        name,
                        body,
                        parts,
                        params,
                    }
                    .constant(c, 0)?,
                    Value::Arg(_) => return Err(refused(GenericPointer::Call)),
                };
                let decided = Cx {
                    name,
                    body,
                    parts,
                    params,
                }
                .not_null(p);
                match (s, decided) {
                    (Space::One(s), NotNull::Is(b)) if s == q.space && nullable => {
                        edit.replace(i, Value::Const(boolean(parts, b, m)?));
                    }
                    (Space::One(s), NotNull::When { c, t: true }) if s == q.space && nullable => {
                        edit.replace(i, c);
                    }
                    (Space::One(s), NotNull::When { c, t: false }) if s == q.space && nullable => {
                        let (t, f) = (boolean(parts, false, m)?, boolean(parts, true, m)?);
                        edit.rewrite.insert(
                            i,
                            InstKind::Select {
                                c,
                                t: Value::Const(t),
                                f: Value::Const(f),
                                fmf: FastMath::NONE,
                            },
                        );
                    }
                    (Space::One(s), _) if s == q.space && nullable => {
                        let null = Value::Const(retyped_null(parts, &mut interned, s, m)?);
                        let a = match p {
                            Value::Const(c) => {
                                Value::Const(retyped(parts, &mut interned, c, s, 0, m)?)
                            }
                            other => other,
                        };
                        edit.rewrite.insert(
                            i,
                            InstKind::ICmp {
                                pred: IntPredicate::Ne,
                                a,
                                b: null,
                            },
                        );
                    }
                    (Space::One(s), _) => {
                        edit.replace(i, Value::Const(boolean(parts, s == q.space, m)?));
                    }
                    (Space::Unseen, _) => {
                        edit.replace(i, Value::Const(boolean(parts, false, m)?));
                    }
                    (Space::Many(how, _), _) => return Err(refused(how)),
                }
                continue;
            }
        }
        // Every other instruction: its generic result its group's type, its generic constants
        // that type's, and each aggregate constant holding generic pointers each the space its
        // leaf brings.
        let holds = |parts: &ModuleParts, v: Value| match (v, plan.leaves.as_ref()) {
            (Value::Const(c), Some(l)) => parts
                .consts
                .ty(c)
                .is_some_and(|t| !generic_of(parts, v) && l.counted(parts, t).unwrap_or(0) > 0),
            _ => false,
        };
        let mut constants = false;
        inst.kind.for_each_value(|v| {
            constants |= matches!(v, Value::Const(_)) && (generic_of(parts, v) || holds(parts, v));
        });
        let shared = result_generic || shares_in(parts, body, (name, params), &inst.kind);
        if constants {
            let mut kind = inst.kind.clone();
            let mut failed = None;
            let mut values: Vec<(Value, AddrSpace)> = Vec::new();
            kind.for_each_value(|v| {
                if let Value::Const(c) = v {
                    if generic_of(parts, v) {
                        let s = if shared {
                            taken(own)
                        } else {
                            match (Cx {
                                name,
                                body,
                                parts,
                                params,
                            })
                            .constant(c, 0)
                            {
                                Ok((Space::Many(how, _), _)) => {
                                    failed.get_or_insert(refused(how));
                                    AddrSpace::Thread
                                }
                                Ok((s, _)) => taken(s),
                                Err(e) => {
                                    failed.get_or_insert(e);
                                    AddrSpace::Thread
                                }
                            }
                        };
                        values.push((v, s));
                    }
                }
            });
            if let Some(e) = failed {
                return Err(e);
            }
            let mut mapped: Vec<(ConstId, ConstId)> = Vec::new();
            for (v, s) in values {
                if let Value::Const(c) = v {
                    if !mapped.iter().any(|(k, _)| *k == c) {
                        mapped.push((c, retyped(parts, &mut interned, c, s, 0, m)?));
                    }
                }
            }
            let mut aggregates: Vec<ConstId> = Vec::new();
            kind.for_each_value(|v| {
                if let (Value::Const(c), true) = (v, holds(parts, v)) {
                    aggregates.push(c);
                }
            });
            for c in aggregates {
                if mapped.iter().any(|(k, _)| *k == c) {
                    continue;
                }
                let (Some(leaves), Some(old)) = (plan.leaves.as_ref(), parts.consts.ty(c)) else {
                    continue;
                };
                let cx = Cx {
                    name,
                    body,
                    parts,
                    params,
                };
                let Some(spaces) = leaves::own_spaces(&cx, leaves, c)? else {
                    continue;
                };
                let mut r = leaves::Retyped {
                    leaves,
                    interned: &mut interned,
                };
                let t = r.ty(parts, old, &spaces, m)?;
                mapped.push((c, r.constant(parts, c, t, &spaces, m)?));
            }
            kind.for_each_value_mut(|v| {
                if let Value::Const(c) = *v {
                    if let Some(&(_, n)) = mapped.iter().find(|(k, _)| *k == c) {
                        *v = Value::Const(n);
                    }
                }
            });
            edit.rewrite.insert(i, kind);
        }
        if result_generic {
            let t = interned.ptr(parts, taken(own), m)?;
            edit.retype.insert(i, t);
        }
    }
    sweep(body, parts, &mut edit, m, held)?;
    apply(body, &edit, &|c| c, m)
}

/// Delete the generic pointers `edit` leaves without a use: those only answered queries read, and
/// what only those read. Each computes an address and nothing else (a `getelementptr`, a cast, a
/// `freeze`, a `phi` or a `select`).
fn sweep(
    body: &Body,
    parts: &ModuleParts,
    edit: &mut BodyEdit,
    m: &mut Meter,
    held: &mut usize,
) -> Result<(), Error> {
    let n = body.insts.len();
    let mut candidate: Vec<bool> = filled(m, held, n, false)?;
    for (i, inst) in body.insts.iter() {
        m.tick(1)?;
        let generic = matches!(
            parts.types.get(inst.ty),
            Some(Type::Ptr {
                space: AddrSpace::Generic
            })
        );
        let address = matches!(
            inst.kind,
            InstKind::Gep { .. }
                | InstKind::Cast {
                    op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                    ..
                }
                | InstKind::Freeze(_)
                | InstKind::Phi { .. }
                | InstKind::Select { .. }
        );
        let replaced = edit.replace.get(i.index()).is_some_and(Option::is_some);
        if let Some(slot) = candidate.get_mut(i.index()) {
            *slot = generic && address && !replaced;
        }
    }
    // Live: what a surviving instruction that is no candidate reads, and what a live candidate
    // reads; each candidate enters the stack once.
    let mut live: Vec<bool> = filled(m, held, n, false)?;
    let mut stack: Vec<InstId> = filled(m, held, n, InstId::new(0))?;
    stack.clear();
    let mut failed = None;
    let mut mark = |v: Value, live: &mut Vec<bool>, stack: &mut Vec<InstId>| {
        let mut v = v;
        for _ in 0..=n {
            let Value::Inst(j) = v else { return };
            match edit.replace.get(j.index()).copied().flatten() {
                Some(next) => v = next,
                None => {
                    if candidate.get(j.index()).copied().unwrap_or(false)
                        && !live.get(j.index()).copied().unwrap_or(true)
                    {
                        if let Some(slot) = live.get_mut(j.index()) {
                            *slot = true;
                        }
                        stack.push(j);
                    }
                    return;
                }
            }
        }
        failed.get_or_insert_with(|| invariant("instruction replacements form a cycle"));
    };
    for (i, inst) in body.insts.iter() {
        m.tick(1)?;
        let gone = edit.replace.get(i.index()).is_some_and(Option::is_some)
            || edit.delete.get(i.index()).copied().unwrap_or(false)
            || candidate.get(i.index()).copied().unwrap_or(false);
        if gone {
            continue;
        }
        let kind = edit.rewrite.get(&i).unwrap_or(&inst.kind);
        kind.for_each_value(|v| mark(v, &mut live, &mut stack));
    }
    while let Some(j) = stack.pop() {
        m.tick(1)?;
        let inst = body
            .insts
            .get(j)
            .ok_or_else(|| invariant("instruction out of range"))?;
        let kind = edit.rewrite.get(&j).unwrap_or(&inst.kind);
        kind.for_each_value(|v| mark(v, &mut live, &mut stack));
    }
    if let Some(e) = failed {
        return Err(e);
    }
    for k in 0..n {
        if candidate.get(k).copied().unwrap_or(false) && !live.get(k).copied().unwrap_or(true) {
            edit.delete(InstId::new(
                u32::try_from(k).map_err(|_| invariant("instruction ids past u32"))?,
            ));
        }
    }
    Ok(())
}

/// A null pointer of space `s`.
fn retyped_null(
    parts: &mut ModuleParts,
    interned: &mut Interned,
    s: AddrSpace,
    m: &mut Meter,
) -> Result<ConstId, Error> {
    let ty = interned.ptr(parts, s, m)?;
    parts.consts.intern(
        Constant {
            ty,
            value: Const::Null,
        },
        m,
    )
}

/// [`shares`] over the module's tables alone.
fn shares_in(
    parts: &ModuleParts,
    body: &Body,
    (name, params): (&DiagName, &[TypeId]),
    kind: &InstKind,
) -> bool {
    shares(
        &Cx {
            name,
            body,
            parts,
            params,
        },
        kind,
    )
}
