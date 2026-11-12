//! Aggregate and vector values that hold generic pointers.
//!
//! Each generic pointer a struct, array or vector value holds is a *leaf*: a struct's leaves are
//! its fields' in order, and an array's or a vector's are its element type's, shared by every
//! element (an index only the run knows picks any of them). Each leaf of each instruction whose
//! value holds one is a union-find node, in the group of what it comes from:
//!
//! - an `insertvalue`'s leaves under its indices, the inserted value's; its others, its
//!   aggregate operand's (all of them where an index steps into an array, whose elements share
//!   leaves);
//! - an `extractvalue`'s leaves (or the generic pointer it extracts), its aggregate operand's
//!   under its indices;
//! - a `phi`'s, a `select`'s or a `freeze`'s leaves, its operands';
//! - a vector's one leaf through `insertelement`, `extractelement` and `shufflevector`.
//!
//! A constant operand brings in each leaf's space; an aggregate a load, a call or a parameter
//! gives brings in a space only the run knows. Each such value then holds pointers of its leaves'
//! spaces: its type is its type with each generic pointer that space's ([`Retyped`]), its
//! constants retyped alike.

use crate::budget::Meter;
use crate::error::{Error, GenericPointer};
use crate::ir::{AddrSpace, Const, ConstId, Constant, Idx, Inst, InstId, InstKind, ModuleParts};
use crate::ir::{Type, TypeId, Value};

use super::super::rewrite::{filled, invariant, BodyEdit};
use super::{refused, retyped, taken, Cx, Interned, Space, DEPTH};

/// No leaves.
const NONE: u32 = u32::MAX;

/// The leaves of the entry's instructions.
pub(super) struct Leaves {
    /// Per instruction: its first leaf, where its value is an aggregate or a vector that holds
    /// generic pointers; `NONE` else.
    first: Vec<u32>,
    /// Per type, the leaves it holds (`NONE` until counted).
    counts: Vec<u32>,
    /// How many leaves there are.
    pub total: usize,
}

impl Leaves {
    /// The leaves of the entry's instructions, `None` when no instruction's value or constant
    /// operand holds a generic pointer in an aggregate or a vector. Every type such a value or
    /// constant has is counted, with the types it holds.
    pub fn find(cx: &Cx<'_>, m: &mut Meter, held: &mut usize) -> Result<Option<Leaves>, Error> {
        let n = cx.body.insts.len();
        let mut leaves = Leaves {
            first: filled(m, held, n, NONE)?,
            counts: filled(m, held, cx.parts.types.len(), NONE)?,
            total: 0,
        };
        let mut total = 0u32;
        let mut any = false;
        let mut constants = Vec::new();
        for (i, inst) in cx.body.insts.iter() {
            m.tick(1)?;
            constants.clear();
            inst.kind.for_each_value(|v| {
                if let Some(t) = cx.ty(v).filter(|&t| {
                    matches!(v, Value::Const(_))
                        && !cx.generic(t)
                        && cx.parts.types.holds_pointer(t)
                }) {
                    constants.push(t);
                }
            });
            for &t in &constants {
                any |= leaves.count(cx.parts, t, 0)? > 0;
            }
            if cx.generic(inst.ty) || !cx.parts.types.holds_pointer(inst.ty) {
                continue;
            }
            let count = leaves.count(cx.parts, inst.ty, 0)?;
            if count == 0 {
                continue;
            }
            any = true;
            if let Some(f) = leaves.first.get_mut(i.index()) {
                *f = total;
            }
            total = total
                .checked_add(count)
                .ok_or_else(|| invariant("leaves past u32"))?;
        }
        leaves.total = total as usize;
        Ok(any.then_some(leaves))
    }

    /// The first leaf of instruction `i`'s value and how many it holds, where it holds any.
    pub fn of(&self, i: InstId, parts: &ModuleParts, ty: TypeId) -> Option<(u32, u32)> {
        let f = *self.first.get(i.index())?;
        if f == NONE {
            return None;
        }
        Some((f, self.counted(parts, ty)?))
    }

    /// The leaves type `ty` holds, counted already.
    pub fn counted(&self, parts: &ModuleParts, ty: TypeId) -> Option<u32> {
        match parts.types.get(ty) {
            Some(Type::Ptr {
                space: AddrSpace::Generic,
            }) => Some(1),
            _ => self.counts.get(ty.index()).copied().filter(|&c| c != NONE),
        }
    }

    /// The leaves type `ty` holds: a generic pointer is one.
    pub fn count(&mut self, parts: &ModuleParts, ty: TypeId, depth: u32) -> Result<u32, Error> {
        if depth > DEPTH {
            return Err(refused(GenericPointer::Aggregate));
        }
        if let Some(c) = self.counted(parts, ty) {
            return Ok(c);
        }
        let c = match parts.types.get(ty) {
            Some(Type::Vector { elem, .. } | Type::Array { elem, .. }) => {
                self.count(parts, *elem, depth + 1)?
            }
            Some(Type::Struct { fields, .. }) => {
                let mut sum = 0u32;
                for &f in fields.iter() {
                    sum = sum
                        .checked_add(self.count(parts, f, depth + 1)?)
                        .ok_or_else(|| invariant("leaves past u32"))?;
                }
                sum
            }
            _ => 0,
        };
        if let Some(slot) = self.counts.get_mut(ty.index()) {
            *slot = c;
        }
        Ok(c)
    }

    /// The leaves of type `ty` (counted) the value at `indices` holds, `lo..hi`, and whether
    /// every index steps into a struct (so the value is those leaves alone, not one of an
    /// array's elements that share them).
    pub fn range(
        &self,
        parts: &ModuleParts,
        ty: TypeId,
        indices: &[u32],
    ) -> Result<(u32, u32, bool), Error> {
        let count = |t: TypeId| {
            self.counted(parts, t)
                .ok_or_else(|| invariant("a type's leaves not counted"))
        };
        let (mut lo, mut ty, mut exact) = (0u32, ty, true);
        for &k in indices {
            match parts.types.get(ty) {
                Some(Type::Struct { fields, .. }) => {
                    let k = k as usize;
                    for &f in fields.iter().take(k) {
                        lo = lo
                            .checked_add(count(f)?)
                            .ok_or_else(|| invariant("leaves past u32"))?;
                    }
                    ty = *fields
                        .get(k)
                        .ok_or_else(|| invariant("an aggregate index out of range"))?;
                }
                Some(Type::Array { elem, .. } | Type::Vector { elem, .. }) => {
                    ty = *elem;
                    exact = false;
                }
                _ => return Err(invariant("an aggregate index into a scalar")),
            }
        }
        let hi = lo
            .checked_add(count(ty)?)
            .ok_or_else(|| invariant("leaves past u32"))?;
        Ok((lo, hi, exact))
    }
}

/// The space leaf `k` of constant `c` brings in, and whether it is null. `None` where the
/// constant holds fewer leaves.
pub(super) fn constant_leaf(
    cx: &Cx<'_>,
    leaves: &Leaves,
    c: ConstId,
    k: u32,
    depth: u32,
) -> Result<(Space, bool), Error> {
    if depth > DEPTH {
        return Err(refused(GenericPointer::Aggregate));
    }
    let ty = cx
        .parts
        .consts
        .ty(c)
        .ok_or_else(|| invariant("a constant out of range"))?;
    if cx.generic(ty) {
        return cx.constant(c, 0);
    }
    let value = &cx
        .parts
        .consts
        .get(c)
        .ok_or_else(|| invariant("a constant out of range"))?
        .value;
    Ok(match (value, cx.parts.types.get(ty)) {
        (Const::Null, _) => (Space::Unseen, true),
        (Const::Undef | Const::Poison, _) => (Space::Unseen, false),
        (Const::Aggregate(elems), Some(Type::Struct { fields, .. })) => {
            let mut lo = 0u32;
            for (&e, &f) in elems.iter().zip(fields.iter()) {
                let n = leaves.counted(cx.parts, f).unwrap_or(0);
                if k < lo + n {
                    return constant_leaf(cx, leaves, e, k - lo, depth + 1);
                }
                lo += n;
            }
            return Err(invariant("a constant leaf out of range"));
        }
        // Every element holds the shared leaves.
        (Const::Aggregate(elems), Some(Type::Array { .. } | Type::Vector { .. })) => {
            let (mut s, mut z) = (Space::Unseen, false);
            for &e in elems.iter() {
                let (es, ez) = constant_leaf(cx, leaves, e, k, depth + 1)?;
                s = s.join(es, None);
                z |= ez;
            }
            (s, z)
        }
        _ => (Space::Many(GenericPointer::Aggregate, None), false),
    })
}

/// Types and constants with their generic pointers made pointers of their leaves' spaces.
pub(super) struct Retyped<'a> {
    pub leaves: &'a Leaves,
    pub interned: &'a mut Interned,
}

impl Retyped<'_> {
    /// Type `ty` with its leaves' generic pointers of `spaces`, one a leaf.
    pub fn ty(
        &mut self,
        parts: &mut ModuleParts,
        ty: TypeId,
        spaces: &[AddrSpace],
        m: &mut Meter,
    ) -> Result<TypeId, Error> {
        let t = parts
            .types
            .get(ty)
            .cloned()
            .ok_or_else(|| invariant("a type out of range"))?;
        let made = match t {
            Type::Ptr {
                space: AddrSpace::Generic,
            } => {
                let s = *spaces.first().ok_or_else(|| invariant("a leaf's space"))?;
                return self.interned.ptr(parts, s, m);
            }
            Type::Struct {
                fields,
                packed,
                name,
            } => {
                let mut lo = 0usize;
                let mut out = Vec::new();
                m.reserve(&mut out, fields.len())?;
                for &f in fields.iter() {
                    let n = self.leaves.counted(parts, f).unwrap_or(0) as usize;
                    let part = spaces
                        .get(lo..lo + n)
                        .ok_or_else(|| invariant("a struct's leaves"))?;
                    out.push(if n == 0 {
                        f
                    } else {
                        self.ty(parts, f, part, m)?
                    });
                    lo += n;
                }
                m.release(out.capacity() * std::mem::size_of::<TypeId>());
                if out.iter().eq(fields.iter()) {
                    return Ok(ty);
                }
                Type::Struct {
                    fields: out.into_boxed_slice(),
                    packed,
                    name,
                }
            }
            Type::Array { elem, len } => Type::Array {
                elem: self.ty(parts, elem, spaces, m)?,
                len,
            },
            Type::Vector { elem, len } => Type::Vector {
                elem: self.ty(parts, elem, spaces, m)?,
                len,
            },
            _ => return Ok(ty),
        };
        parts.types.intern(made, m)
    }

    /// Constant `c` as a constant of type `ty`, the retyped form of its own, whose leaves are of
    /// `spaces`.
    pub fn constant(
        &mut self,
        parts: &mut ModuleParts,
        c: ConstId,
        ty: TypeId,
        spaces: &[AddrSpace],
        m: &mut Meter,
    ) -> Result<ConstId, Error> {
        let k = parts
            .consts
            .get(c)
            .cloned()
            .ok_or_else(|| invariant("a constant out of range"))?;
        if parts.consts.ty(c) == Some(ty) {
            return Ok(c);
        }
        let value = match k.value {
            v @ (Const::Null | Const::Undef | Const::Poison) => v,
            Const::Aggregate(elems) => {
                let fields: Vec<TypeId> = match parts.types.get(ty) {
                    Some(Type::Struct { fields, .. }) => fields.to_vec(),
                    Some(Type::Array { elem, .. } | Type::Vector { elem, .. }) => {
                        vec![*elem; elems.len()]
                    }
                    _ => return Err(invariant("an aggregate constant of a scalar type")),
                };
                let olds: Vec<TypeId> = match parts.types.get(k.ty) {
                    Some(Type::Struct { fields, .. }) => fields.to_vec(),
                    Some(Type::Array { elem, .. } | Type::Vector { elem, .. }) => {
                        vec![*elem; elems.len()]
                    }
                    _ => return Err(invariant("an aggregate constant of a scalar type")),
                };
                let shared = !matches!(parts.types.get(ty), Some(Type::Struct { .. }));
                let mut lo = 0usize;
                let mut out = Vec::with_capacity(elems.len());
                for ((&e, &f), &old) in elems.iter().zip(fields.iter()).zip(olds.iter()) {
                    let n = self.leaves.counted(parts, old).unwrap_or(0) as usize;
                    let part = if shared {
                        spaces
                    } else {
                        let p = spaces
                            .get(lo..lo + n)
                            .ok_or_else(|| invariant("a constant's leaves"))?;
                        lo += n;
                        p
                    };
                    out.push(if n == 0 {
                        e
                    } else if matches!(
                        parts.types.get(old),
                        Some(Type::Ptr {
                            space: AddrSpace::Generic
                        })
                    ) {
                        let s = *part.first().ok_or_else(|| invariant("a leaf's space"))?;
                        retyped(parts, self.interned, e, s, 0, m)?
                    } else {
                        self.constant(parts, e, f, part, m)?
                    });
                }
                Const::Aggregate(out.into_boxed_slice())
            }
            _ => return Err(refused(GenericPointer::Aggregate)),
        };
        parts.consts.intern(Constant { ty, value }, m)
    }

    /// Constant operand `c` of an instruction whose value is of type `ty` (retyped) and holds
    /// leaves of `spaces`: that type's constant.
    fn operand(
        &mut self,
        parts: &mut ModuleParts,
        v: &mut Value,
        ty: TypeId,
        spaces: &[AddrSpace],
        m: &mut Meter,
    ) -> Result<bool, Error> {
        let Value::Const(c) = *v else {
            return Ok(false);
        };
        *v = Value::Const(self.constant(parts, c, ty, spaces, m)?);
        Ok(true)
    }
}

/// The spaces of constant `c`'s leaves, each the space it brings itself (thread where none):
/// `None` where it holds none.
pub(super) fn own_spaces(
    cx: &Cx<'_>,
    leaves: &Leaves,
    c: ConstId,
) -> Result<Option<Vec<AddrSpace>>, Error> {
    let Some(ty) = cx.parts.consts.ty(c) else {
        return Err(invariant("a constant out of range"));
    };
    if cx.generic(ty) {
        return Ok(None);
    }
    let count = leaves.counted(cx.parts, ty).unwrap_or(0);
    if count == 0 {
        return Ok(None);
    }
    let mut out = Vec::with_capacity(count as usize);
    for k in 0..count {
        match constant_leaf(cx, leaves, c, k, 0)?.0 {
            Space::Many(how, _) => return Err(refused(how)),
            s => out.push(taken(s)),
        }
    }
    Ok(Some(out))
}

/// The rewrite of instruction `i`, whose value holds `count` leaves from `first`: its type with
/// each leaf's group's space, and its constant operands retyped alike.
pub(super) fn rewrite(
    parts: &mut ModuleParts,
    r: &mut Retyped<'_>,
    leaf_space: &[Space],
    (i, inst): (InstId, &Inst),
    (first, count): (u32, u32),
    edit: &mut BodyEdit,
    m: &mut Meter,
) -> Result<(), Error> {
    let spaces: Vec<AddrSpace> = leaf_space
        .get(first as usize..first as usize + count as usize)
        .ok_or_else(|| invariant("an instruction's leaves out of range"))?
        .iter()
        .map(|&s| taken(s))
        .collect();
    let ty = r.ty(parts, inst.ty, &spaces, m)?;
    let mut kind = inst.kind.clone();
    let mut changed = false;
    match &mut kind {
        InstKind::InsertValue { agg, val, indices } => {
            changed |= r.operand(parts, agg, ty, &spaces, m)?;
            if let Value::Const(c) = *val {
                let (lo, hi, _) = r.leaves.range(parts, inst.ty, indices)?;
                if lo < hi {
                    let part = spaces
                        .get(lo as usize..hi as usize)
                        .ok_or_else(|| invariant("an inserted value's leaves"))?;
                    let old = parts
                        .consts
                        .ty(c)
                        .ok_or_else(|| invariant("a constant out of range"))?;
                    let new = if matches!(
                        parts.types.get(old),
                        Some(Type::Ptr {
                            space: AddrSpace::Generic
                        })
                    ) {
                        let s = *part.first().ok_or_else(|| invariant("a leaf's space"))?;
                        retyped(parts, r.interned, c, s, 0, m)?
                    } else {
                        let t = r.ty(parts, old, part, m)?;
                        r.constant(parts, c, t, part, m)?
                    };
                    *val = Value::Const(new);
                    changed = true;
                }
            }
        }
        InstKind::InsertElement { vec, elt, .. } => {
            changed |= r.operand(parts, vec, ty, &spaces, m)?;
            if let Value::Const(c) = *elt {
                let s = *spaces.first().ok_or_else(|| invariant("a leaf's space"))?;
                *elt = Value::Const(retyped(parts, r.interned, c, s, 0, m)?);
                changed = true;
            }
        }
        InstKind::Phi { incoming, .. } => {
            for (v, _) in incoming.iter_mut() {
                changed |= r.operand(parts, v, ty, &spaces, m)?;
            }
        }
        InstKind::Select { t, f, .. } | InstKind::ShuffleVector { a: t, b: f, .. } => {
            changed |= r.operand(parts, t, ty, &spaces, m)?;
            changed |= r.operand(parts, f, ty, &spaces, m)?;
        }
        InstKind::Freeze(v) => changed |= r.operand(parts, v, ty, &spaces, m)?,
        _ => {}
    }
    if changed {
        edit.rewrite.insert(i, kind);
    }
    if ty != inst.ty {
        edit.retype.insert(i, ty);
    }
    Ok(())
}
