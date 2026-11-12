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
//! operands, and the two sides of an `icmp`. Each such group has one space where every pointer
//! entering it does:
//!
//! - an `addrspacecast` from a pointer of a device, thread or threadgroup space brings in that
//!   space, and so does a constant expression of such a cast;
//! - a null, undefined or poison constant brings in none: it points nowhere;
//! - a generic pointer a load, a call, an `inttoptr`, an atomic or an aggregate gives, or a
//!   parameter, brings in a space only the run knows.
//!
//! A group that pointers of two spaces enter, or one a space only the run knows, is
//! `Unsupported{AddressSpace(8)}`: no Vulkan pointer holds either. So is a generic pointer passed
//! to a call other than a query (the callee's type names the generic space), returned, or held in
//! an aggregate or a vector value.
//!
//! Each generic pointer then is its group's pointer type, and a group that no space enters a
//! thread pointer (it addresses nothing). A cast into the generic space is its operand. A cast
//! out of it is its operand where it names the group's space, and poison where it names another
//! (an `addrspacecast` of a pointer not of the space it names is poison). Every other instruction
//! keeps its operation, its generic constants those of its group's type. A query is false for
//! another space; for the group's own space it is true, or, where a null constant may reach the
//! pointer, whether the pointer is not null, as Metal answers no space for a null generic
//! pointer. Aggregate types that hold generic pointers (an `alloca`'s or a `getelementptr`'s
//! source type) keep them: they lay out bytes, and a pointer of every space is 64 bits.

use crate::budget::Meter;
use crate::error::{Error, Feature, Unsupported};
use crate::intrinsics::AirIntrinsic;
use crate::ir::{AddrSpace, Body, Callee, CastOp, Const, ConstExpr, ConstId, Constant, FuncId};
use crate::ir::{Idx, InstKind, IntPredicate, ModuleParts, Type, TypeId, Value};

use super::rewrite::{apply, filled, invariant, scoped, BodyEdit};

/// The most aggregate levels searched for a generic pointer in a type.
const DEPTH: u32 = 32;

/// The pointers a group of generic pointers may hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Space {
    /// None yet: only constants that point nowhere.
    Unseen,
    /// Pointers of this space.
    One(AddrSpace),
    /// Pointers of two spaces, or of a space only the run knows.
    Many,
}

impl Space {
    fn join(self, other: Space) -> Space {
        match (self, other) {
            (Space::Unseen, s) | (s, Space::Unseen) => s,
            (Space::One(a), Space::One(b)) if a == b => Space::One(a),
            _ => Space::Many,
        }
    }
}

fn refused() -> Error {
    Unsupported::not_implemented(Feature::AddressSpace(AddrSpace::Generic.raw())).into()
}

/// The entry's body and parameter types, and the module's tables.
struct Cx<'a> {
    body: &'a Body,
    parts: &'a ModuleParts,
    params: &'a [TypeId],
}

impl Cx<'_> {
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
            return Err(refused());
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
            return Err(refused());
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
                    _ => (Space::Many, false),
                }
            }
            Const::Expr(ConstExpr::Gep { ops, .. }) => match ops.first() {
                Some(&base) => self.constant(base, depth + 1)?,
                None => (Space::Many, false),
            },
            _ => (Space::Many, false),
        })
    }

    /// The space and nullness a generic operand brings in by itself: a constant's, `Many` for a
    /// parameter, nothing for an instruction (its group carries it).
    fn operand(&self, v: Value) -> Result<(Space, bool), Error> {
        match v {
            Value::Const(c) => self.constant(c, 0),
            Value::Arg(_) => Ok((Space::Many, false)),
            Value::Inst(_) => Ok((Space::Unseen, false)),
        }
    }
}

/// The generic operands of `kind` that share the result's group (or, for an `icmp`, each other's).
fn linked(kind: &InstKind) -> Vec<Value> {
    match kind {
        InstKind::Gep { base, .. } => vec![*base],
        InstKind::Cast {
            op: CastOp::Bitcast | CastOp::AddrSpaceCast,
            v,
        }
        | InstKind::Freeze(v) => vec![*v],
        InstKind::Phi { incoming, .. } => incoming.iter().map(|&(v, _)| v).collect(),
        InstKind::Select { t, f, .. } => vec![*t, *f],
        InstKind::ICmp { a, b, .. } => vec![*a, *b],
        _ => Vec::new(),
    }
}

/// The group's representative of `i` (path halving).
fn find(parent: &mut [u32], i: usize) -> usize {
    let mut x = i;
    while let Some(&p) = parent.get(x) {
        let p = p as usize;
        if p == x {
            break;
        }
        let grand = parent.get(p).copied().unwrap_or(p as u32);
        if let Some(slot) = parent.get_mut(x) {
            *slot = grand;
        }
        x = grand as usize;
    }
    x
}

fn union(parent: &mut [u32], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        if let Some(slot) = parent.get_mut(ra) {
            *slot = rb as u32;
        }
    }
}

/// `body` with every generic pointer a pointer of its group's space (module docs), and whether a
/// query folded (SCCP then folds what it decides). `func` is the entry whose body it is.
pub(super) fn resolve(
    body: Body,
    parts: &mut ModuleParts,
    func: FuncId,
    m: &mut Meter,
) -> Result<(Body, bool), Error> {
    let params: Vec<TypeId> = match parts.funcs.get(func).and_then(|f| parts.types.get(f.ty)) {
        Some(Type::Function { params, .. }) => params.to_vec(),
        _ => return Err(invariant("the entry's type is not a function")),
    };
    let planned = scoped(m, |m, held| {
        let cx = Cx {
            body: &body,
            parts,
            params: &params,
        };
        plan(&cx, m, held)
    })?;
    let Some(plan) = planned else {
        return Ok((body, false));
    };
    let edited = scoped(m, |m, held| rewrite(&body, parts, &params, &plan, m, held))?;
    body.release(m);
    Ok((edited, plan.folds))
}

/// Each generic instruction's group space and nullness; `None` when the body holds no generic
/// pointer.
struct Plan {
    /// Per instruction: its group's space (`Unseen` for a non-generic instruction).
    space: Vec<Space>,
    /// Per instruction: whether a null constant may reach it.
    null: Vec<bool>,
    /// Whether a query is answered.
    folds: bool,
}

fn plan(cx: &Cx<'_>, m: &mut Meter, held: &mut usize) -> Result<Option<Plan>, Error> {
    let n = cx.body.insts.len();
    // Whether anything is generic: an instruction's result or any operand.
    let mut any = false;
    for inst in cx.body.insts.values() {
        m.tick(1)?;
        any |= cx.generic(inst.ty);
        inst.kind.for_each_value(|v| any |= cx.is_generic(v));
        if any {
            break;
        }
    }
    if !any {
        return Ok(None);
    }
    let mut parent: Vec<u32> = filled(m, held, n, 0)?;
    for (k, slot) in parent.iter_mut().enumerate() {
        *slot = u32::try_from(k).map_err(|_| invariant("instruction ids past u32"))?;
    }
    let mut source: Vec<Space> = filled(m, held, n, Space::Unseen)?;
    let mut null: Vec<bool> = filled(m, held, n, false)?;
    let mut folds = false;
    for (i, inst) in cx.body.insts.iter() {
        m.tick(1)?;
        let generic = cx.generic(inst.ty);
        // Generic values held where no group reaches: an aggregate or a vector.
        if !generic && cx.holds_generic(inst.ty, 0)? {
            return Err(refused());
        }
        match &inst.kind {
            InstKind::Call(call) => {
                let query = matches!(
                    call.callee,
                    Callee::Air {
                        op: AirIntrinsic::PointerSpace(_),
                        ..
                    }
                );
                if call.args.iter().any(|&a| cx.is_generic(a)) && !query {
                    return Err(refused());
                }
                folds |= query;
                // A query's pointer is its own group's.
            }
            InstKind::Ret(Some(v)) if cx.is_generic(*v) => return Err(refused()),
            _ => {}
        }
        if !generic {
            // An `icmp` of generic pointers joins its sides' groups.
            if let InstKind::ICmp { a, b, .. } = inst.kind {
                if let (Value::Inst(x), Value::Inst(y)) = (a, b) {
                    if cx.is_generic(a) && cx.is_generic(b) {
                        union(&mut parent, x.index(), y.index());
                    }
                }
            }
            continue;
        }
        let (mut s, mut z) = (Space::Unseen, false);
        match &inst.kind {
            InstKind::Cast {
                op: CastOp::AddrSpaceCast,
                v,
            } if !cx.is_generic(*v) => match cx.ty(*v).and_then(|t| cx.parts.types.get(t)) {
                Some(Type::Ptr { space }) => {
                    s = Space::One(*space);
                    z = matches!(
                        v,
                        Value::Const(c) if matches!(cx.parts.consts.get(*c).map(|k| &k.value), Some(Const::Null))
                    );
                }
                _ => s = Space::Many,
            },
            InstKind::Gep { .. }
            | InstKind::Cast {
                op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                ..
            }
            | InstKind::Freeze(_)
            | InstKind::Phi { .. }
            | InstKind::Select { .. } => {
                for v in linked(&inst.kind) {
                    let (vs, vz) = cx.operand(v)?;
                    s = s.join(vs);
                    z |= vz;
                    if let Value::Inst(j) = v {
                        union(&mut parent, i.index(), j.index());
                    }
                }
            }
            // A load, a call, an `inttoptr`, an atomic or an aggregate's element.
            _ => s = Space::Many,
        }
        if let Some(slot) = source.get_mut(i.index()) {
            *slot = s;
        }
        if let Some(slot) = null.get_mut(i.index()) {
            *slot = z;
        }
    }
    // Each group's space: the join of what enters each of its members.
    let mut group: Vec<Space> = filled(m, held, n, Space::Unseen)?;
    for k in 0..n {
        m.tick(1)?;
        let r = find(&mut parent, k);
        let s = source.get(k).copied().unwrap_or(Space::Unseen);
        if let Some(slot) = group.get_mut(r) {
            *slot = slot.join(s);
        }
    }
    // An `icmp`'s constant sides enter its group too.
    for inst in cx.body.insts.values() {
        if let InstKind::ICmp { a, b, .. } = inst.kind {
            if !(cx.is_generic(a) && cx.is_generic(b)) {
                continue;
            }
            if let (Value::Inst(x), c @ Value::Const(_)) | (c @ Value::Const(_), Value::Inst(x)) =
                (a, b)
            {
                let r = find(&mut parent, x.index());
                let (s, _) = cx.operand(c)?;
                if let Some(slot) = group.get_mut(r) {
                    *slot = slot.join(s);
                }
            }
        }
    }
    let mut space: Vec<Space> = filled(m, held, n, Space::Unseen)?;
    for (i, inst) in cx.body.insts.iter() {
        m.tick(1)?;
        if !cx.generic(inst.ty) && !query_or_compare(cx, &inst.kind) {
            continue;
        }
        let r = match &inst.kind {
            // A query's or a comparison's group is its pointer's.
            InstKind::Call(call) if !cx.generic(inst.ty) => match call.args.first() {
                Some(Value::Inst(p)) => find(&mut parent, p.index()),
                _ => continue,
            },
            InstKind::ICmp { a, b, .. } if !cx.generic(inst.ty) => match (a, b) {
                (Value::Inst(p), _) | (_, Value::Inst(p)) => find(&mut parent, p.index()),
                _ => continue,
            },
            _ => find(&mut parent, i.index()),
        };
        let s = group.get(r).copied().unwrap_or(Space::Unseen);
        if s == Space::Many {
            return Err(refused());
        }
        if let Some(slot) = space.get_mut(i.index()) {
            *slot = s;
        }
    }
    // Nullness, forward to a fixed point: a null constant reaches what it enters and what that
    // reaches.
    let mut changed = true;
    let mut rounds = 0;
    while changed {
        changed = false;
        rounds += 1;
        if rounds > n + 1 {
            return Err(invariant("nullness does not settle"));
        }
        for (i, inst) in cx.body.insts.iter() {
            m.tick(1)?;
            if !cx.generic(inst.ty) || null.get(i.index()).copied().unwrap_or(true) {
                continue;
            }
            let reached = match &inst.kind {
                InstKind::Cast {
                    op: CastOp::AddrSpaceCast,
                    v,
                } if !cx.is_generic(*v) => false,
                kind => linked(kind).iter().any(|v| match v {
                    Value::Inst(j) => null.get(j.index()).copied().unwrap_or(false),
                    _ => false,
                }),
            };
            if reached {
                if let Some(slot) = null.get_mut(i.index()) {
                    *slot = true;
                }
                changed = true;
            }
        }
    }
    Ok(Some(Plan { space, null, folds }))
}

/// Whether `kind` is a query or a comparison of generic pointers, whose space is its pointer's.
fn query_or_compare(cx: &Cx<'_>, kind: &InstKind) -> bool {
    match kind {
        InstKind::Call(call) => matches!(
            call.callee,
            Callee::Air {
                op: AirIntrinsic::PointerSpace(_),
                ..
            }
        ),
        InstKind::ICmp { a, b, .. } => cx.is_generic(*a) && cx.is_generic(*b),
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
        return Err(refused());
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
                _ => Err(refused()),
            };
        }
        Const::Expr(ConstExpr::Gep {
            src_ty,
            inbounds,
            ops,
        }) => {
            let (&base, rest) = ops.split_first().ok_or_else(refused)?;
            let base = retyped(parts, interned, base, s, depth + 1, m)?;
            let ops: Box<[ConstId]> = std::iter::once(base).chain(rest.iter().copied()).collect();
            Const::Expr(ConstExpr::Gep {
                src_ty,
                inbounds,
                ops,
            })
        }
        _ => return Err(refused()),
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
    params: &[TypeId],
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
                    _ => return Err(refused()),
                };
                let group = match v {
                    Value::Inst(j) => plan.space.get(j.index()).copied().unwrap_or(Space::Unseen),
                    Value::Const(c) => {
                        Cx {
                            body,
                            parts,
                            params,
                        }
                        .constant(c, 0)?
                        .0
                    }
                    Value::Arg(_) => return Err(refused()),
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
                    Space::Many => return Err(refused()),
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
                        body,
                        parts,
                        params,
                    }
                    .constant(c, 0)?,
                    Value::Arg(_) => return Err(refused()),
                };
                match s {
                    Space::One(s) if s == q.space && nullable => {
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
                    Space::One(s) => {
                        edit.replace(i, Value::Const(boolean(parts, s == q.space, m)?));
                    }
                    Space::Unseen => edit.replace(i, Value::Const(boolean(parts, false, m)?)),
                    Space::Many => return Err(refused()),
                }
                continue;
            }
        }
        // Every other instruction: its generic result its group's type, its generic constants
        // that type's.
        let mut constants = false;
        inst.kind.for_each_value(|v| {
            constants |= matches!(v, Value::Const(_)) && generic_of(parts, v);
        });
        let shared = result_generic || query_or_compare_in(parts, body, params, &inst.kind);
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
                                body,
                                parts,
                                params,
                            })
                            .constant(c, 0)
                            {
                                Ok((s, _)) if s != Space::Many => taken(s),
                                Ok(_) => {
                                    failed.get_or_insert(refused());
                                    AddrSpace::Thread
                                }
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
    apply(body, &edit, &|c| c, m)
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

/// [`query_or_compare`] over the module's tables alone.
fn query_or_compare_in(
    parts: &ModuleParts,
    body: &Body,
    params: &[TypeId],
    kind: &InstKind,
) -> bool {
    query_or_compare(
        &Cx {
            body,
            parts,
            params,
        },
        kind,
    )
}
