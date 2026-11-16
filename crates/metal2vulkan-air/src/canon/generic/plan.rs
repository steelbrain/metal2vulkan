//! The plan: each generic pointer's group, the space every pointer entering it brings, and
//! whether a null pointer may reach it.
//!
//! The union-find's nodes are the instructions, then the slots of thread memory generic pointers
//! pass through ([`cells`]), then the leaves of the aggregate and vector values that hold them
//! ([`leaves`]). Nullness flows along the same links, from a value to what takes it: an operand to
//! the address it derives, a stored value to its slot and the slot to its loads and to the slots
//! copies move it to, an inserted value
//! or an aggregate's leaf to the leaf that takes it.
//!
//! [`leaves`]: super::leaves

use crate::budget::Meter;
use crate::error::{Error, GenericPointer};
use crate::intrinsics::AirIntrinsic;
use crate::ir::{Callee, CastOp, Const, Idx, InstId, InstKind, Type, Value};

use super::super::rewrite::{filled, invariant};
use super::cells::{self, Cells};
use super::leaves::{constant_leaf, Leaves};
use super::{shares, Cx, Space};

/// Each generic instruction's group space and nullness, and each leaf's space.
pub(super) struct Plan {
    /// Per instruction: its group's space (`Unseen` for a non-generic instruction).
    pub space: Vec<Space>,
    /// Per instruction: whether a null constant may reach it.
    pub null: Vec<bool>,
    /// Whether a query is answered.
    pub folds: bool,
    /// The leaves of the aggregate and vector values that hold generic pointers.
    pub leaves: Option<Leaves>,
    /// Per leaf: its group's space.
    pub leaf_space: Vec<Space>,
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

/// The union-find, what enters each node, and the links nullness flows along.
struct Groups<'c, 'a> {
    cx: &'c Cx<'a>,
    /// Instructions, then slots, then leaves.
    n: usize,
    slots: usize,
    parent: Vec<u32>,
    source: Vec<Space>,
    null: Vec<bool>,
    /// Nullness flows from the first node to the second.
    flow: Vec<(u32, u32)>,
}

impl Groups<'_, '_> {
    fn slot(&self, node: u32) -> usize {
        self.n + node as usize
    }

    fn leaf(&self, first: u32, k: u32) -> usize {
        self.n + self.slots + first as usize + k as usize
    }

    /// The group's representative of `i` (path halving).
    fn find(&mut self, i: usize) -> usize {
        let mut x = i;
        while let Some(&p) = self.parent.get(x) {
            let p = p as usize;
            if p == x {
                break;
            }
            let grand = self.parent.get(p).copied().unwrap_or(p as u32);
            if let Some(slot) = self.parent.get_mut(x) {
                *slot = grand;
            }
            x = grand as usize;
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            if let Some(slot) = self.parent.get_mut(ra) {
                *slot = rb as u32;
            }
        }
    }

    /// Node `to` takes node `from`'s pointers: one group, and `from`'s nullness flows to `to`.
    fn takes(&mut self, to: usize, from: usize, m: &mut Meter) -> Result<(), Error> {
        self.union(to, from);
        let pair = (
            u32::try_from(from).map_err(|_| invariant("nodes past u32"))?,
            u32::try_from(to).map_err(|_| invariant("nodes past u32"))?,
        );
        m.reserve(&mut self.flow, 1)?;
        self.flow.push(pair);
        Ok(())
    }

    /// Space `s` enters node `at_node`, a null pointer with it where `z`.
    fn enter(&mut self, node: usize, s: Space, z: bool, at: InstId) {
        if let Some(x) = self.source.get_mut(node) {
            *x = x.join(s, Some(at));
        }
        if let Some(x) = self.null.get_mut(node) {
            *x |= z;
        }
    }

    /// Node `to` takes leaf `k` of operand `v` (a generic pointer operand is its own leaf 0), at
    /// instruction `at`: one group, and, where `flows`, the operand's nullness. A value that
    /// shares only its type (an `insertvalue`'s leaf its inserted value replaces, and its
    /// aggregate operand's) does not flow.
    fn operand(
        &mut self,
        to: usize,
        (v, k, flows): (Value, u32, bool),
        leaves: &mut Option<Leaves>,
        at: InstId,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let cx = self.cx;
        match v {
            Value::Inst(j) => {
                let ty = cx
                    .body
                    .insts
                    .get(j)
                    .map(|x| x.ty)
                    .ok_or_else(|| invariant("an operand out of range"))?;
                let from = if cx.generic(ty) {
                    j.index()
                } else {
                    let (first, count) = leaves
                        .as_ref()
                        .and_then(|l| l.of(j, cx.parts, ty))
                        .ok_or_else(|| invariant("an aggregate operand without leaves"))?;
                    if k >= count {
                        return Err(invariant("a leaf out of range"));
                    }
                    self.leaf(first, k)
                };
                if flows {
                    self.takes(to, from, m)
                } else {
                    self.union(to, from);
                    Ok(())
                }
            }
            Value::Const(c) => {
                let (s, z) = match leaves.as_mut() {
                    Some(l) => constant_leaf(cx, l, c, k, 0)?,
                    None => cx.constant(c, 0)?,
                };
                self.enter(to, s, z && flows, at);
                Ok(())
            }
            Value::Arg(_) => {
                self.enter(to, Space::Many(GenericPointer::Call, Some(at)), false, at);
                Ok(())
            }
        }
    }
}

pub(super) fn plan(cx: &Cx<'_>, m: &mut Meter, held: &mut usize) -> Result<Option<Plan>, Error> {
    let n = cx.body.insts.len();
    // Whether anything is generic: an instruction's result or any operand, or a generic pointer
    // an aggregate or a vector holds.
    let mut any = false;
    for inst in cx.body.insts.values() {
        m.tick(1)?;
        any |= cx.generic(inst.ty) || holds(cx, Some(inst.ty))?;
        // A type too deep to search is planned, and refused there.
        inst.kind.for_each_value(|v| {
            any |= cx.is_generic(v)
                || (matches!(v, Value::Const(_)) && holds(cx, cx.ty(v)).unwrap_or(true));
        });
        if any {
            break;
        }
    }
    if !any {
        return Ok(None);
    }
    let cells = cells::find(cx, m, held)?;
    let mut leaves = Leaves::find(cx, m, held)?;
    let slots = cells.as_ref().map_or(0, |c| c.slots);
    let total = leaves.as_ref().map_or(0, |l| l.total);
    let nodes = n + slots + total;
    let mut g = Groups {
        cx,
        n,
        slots,
        parent: filled(m, held, nodes, 0)?,
        source: filled(m, held, nodes, Space::Unseen)?,
        null: filled(m, held, nodes, false)?,
        flow: Vec::new(),
    };
    for (k, slot) in g.parent.iter_mut().enumerate() {
        *slot = u32::try_from(k).map_err(|_| invariant("nodes past u32"))?;
    }
    let reach = |c: &Option<Cells>, i: InstId| c.as_ref().and_then(|c| c.reach(i));
    let mut folds = false;
    for (i, inst) in cx.body.insts.iter() {
        m.tick(1)?;
        let generic = cx.generic(inst.ty);
        match &inst.kind {
            InstKind::Call(call) => {
                let query = matches!(
                    call.callee,
                    Callee::Air {
                        op: AirIntrinsic::PointerSpace(_),
                        ..
                    }
                );
                let mut crossing = false;
                for &a in call.args.iter() {
                    crossing |= cx.is_generic(a) || holds(cx, cx.ty(a))?;
                }
                if crossing && !query {
                    return Err(cx.refuse(GenericPointer::Call, i));
                }
                folds |= query;
                // A query's pointer is its own group's.
            }
            InstKind::Ret(Some(v)) if cx.is_generic(*v) || holds(cx, cx.ty(*v))? => {
                return Err(cx.refuse(GenericPointer::Call, i))
            }
            _ => {}
        }
        // An aggregate or a vector that holds generic pointers: its leaves.
        if let Some((first, count)) = leaves.as_ref().and_then(|l| l.of(i, cx.parts, inst.ty)) {
            aggregate(
                &mut g,
                &mut leaves,
                (i, &inst.kind, inst.ty),
                (first, count),
                m,
            )?;
            continue;
        }
        // A store of a generic pointer is in its value's group, and in its slot's where it
        // stores into a cell.
        if let InstKind::Store { val, .. } = inst.kind {
            if cx.is_generic(val) {
                g.operand(i.index(), (val, 0, true), &mut leaves, i, m)?;
                if let Some(Some(node)) = reach(&cells, i) {
                    let s = g.slot(node);
                    g.takes(s, i.index(), m)?;
                }
            }
        }
        if !generic {
            // An `icmp` of generic pointers joins its sides' groups.
            if let InstKind::ICmp { a, b, .. } = inst.kind {
                if let (Value::Inst(x), Value::Inst(y)) = (a, b) {
                    if cx.is_generic(a) && cx.is_generic(b) {
                        g.union(x.index(), y.index());
                    }
                }
            }
            continue;
        }
        match &inst.kind {
            InstKind::Cast {
                op: CastOp::AddrSpaceCast,
                v,
            } if !cx.is_generic(*v) => match cx.ty(*v).and_then(|t| cx.parts.types.get(t)) {
                Some(Type::Ptr { space }) => {
                    let z = matches!(
                        v,
                        Value::Const(c) if matches!(cx.parts.consts.get(*c).map(|k| &k.value), Some(Const::Null))
                    );
                    g.enter(i.index(), Space::One(*space), z, i);
                }
                _ => g.enter(
                    i.index(),
                    Space::Many(GenericPointer::Integer, Some(i)),
                    false,
                    i,
                ),
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
                    g.operand(i.index(), (v, 0, true), &mut leaves, i, m)?;
                }
            }
            // A load from a cell's slot is in the slot's group.
            InstKind::Load { .. } => match reach(&cells, i) {
                Some(Some(node)) => {
                    let s = g.slot(node);
                    g.takes(i.index(), s, m)?;
                }
                _ => g.enter(
                    i.index(),
                    Space::Many(GenericPointer::Memory, Some(i)),
                    false,
                    i,
                ),
            },
            // A generic pointer an aggregate or a vector holds is its leaf's.
            InstKind::ExtractValue { agg, indices } => {
                if matches!(agg, Value::Const(_)) {
                    return Err(cx.refuse(GenericPointer::Aggregate, i));
                }
                let ty = cx.ty(*agg).ok_or_else(|| invariant("an operand's type"))?;
                let (lo, _, _) = range(&mut leaves, cx, ty, indices)?;
                g.operand(i.index(), (*agg, lo, true), &mut leaves, i, m)?;
            }
            InstKind::ExtractElement { vec, .. } => {
                if matches!(vec, Value::Const(_)) {
                    return Err(cx.refuse(GenericPointer::Aggregate, i));
                }
                g.operand(i.index(), (*vec, 0, true), &mut leaves, i, m)?;
            }
            InstKind::AtomicRmw { .. } | InstKind::CmpXchg { .. } => g.enter(
                i.index(),
                Space::Many(GenericPointer::Memory, Some(i)),
                false,
                i,
            ),
            InstKind::Call(_) => g.enter(
                i.index(),
                Space::Many(GenericPointer::Call, Some(i)),
                false,
                i,
            ),
            // An `inttoptr`: no other operation makes a pointer.
            _ => g.enter(
                i.index(),
                Space::Many(GenericPointer::Integer, Some(i)),
                false,
                i,
            ),
        }
    }
    // An access at an unknown offset of a cell may be at any of its slots.
    for &(a, b) in cells.as_ref().map_or(&[][..], |c| &c.links[..]) {
        let (a, b) = (g.slot(a), g.slot(b));
        g.takes(a, b, m)?;
        g.takes(b, a, m)?;
    }
    // A slot a copy moves to takes the slot it moves from.
    for &(to, from) in cells.as_ref().map_or(&[][..], |c| &c.moves[..]) {
        let (to, from) = (g.slot(to), g.slot(from));
        g.takes(to, from, m)?;
    }
    // Each group's space: the join of what enters each of its members.
    let mut group: Vec<Space> = filled(m, held, nodes, Space::Unseen)?;
    for node in 0..nodes {
        m.tick(1)?;
        let r = g.find(node);
        let s = g.source.get(node).copied().unwrap_or(Space::Unseen);
        let at = (node < n)
            .then(|| u32::try_from(node).ok().map(InstId::new))
            .flatten();
        if let Some(slot) = group.get_mut(r) {
            *slot = slot.join(s, at);
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
                let r = g.find(x.index());
                let (s, _) = cx.operand(c)?;
                if let Some(slot) = group.get_mut(r) {
                    *slot = slot.join(s, Some(x));
                }
            }
        }
    }
    let mut space: Vec<Space> = filled(m, held, n, Space::Unseen)?;
    let mut leaf_space: Vec<Space> = filled(m, held, total, Space::Unseen)?;
    for (i, inst) in cx.body.insts.iter() {
        m.tick(1)?;
        if let Some((first, count)) = leaves.as_ref().and_then(|l| l.of(i, cx.parts, inst.ty)) {
            for k in 0..count {
                let r = g.find(g.leaf(first, k));
                let s = group.get(r).copied().unwrap_or(Space::Unseen);
                if let Space::Many(how, at) = s {
                    return Err(cx.refuse(how, at.unwrap_or(i)));
                }
                if let Some(slot) = leaf_space.get_mut(first as usize + k as usize) {
                    *slot = s;
                }
            }
            continue;
        }
        if !cx.generic(inst.ty) && !shares(cx, &inst.kind) {
            continue;
        }
        let r = match &inst.kind {
            // A query's or a comparison's group is its pointer's.
            InstKind::Call(call) if !cx.generic(inst.ty) => match call.args.first() {
                Some(Value::Inst(p)) => g.find(p.index()),
                _ => continue,
            },
            InstKind::ICmp { a, b, .. } if !cx.generic(inst.ty) => match (a, b) {
                (Value::Inst(p), _) | (_, Value::Inst(p)) => g.find(p.index()),
                _ => continue,
            },
            _ => g.find(i.index()),
        };
        let s = group.get(r).copied().unwrap_or(Space::Unseen);
        if let Space::Many(how, at) = s {
            return Err(cx.refuse(how, at.unwrap_or(i)));
        }
        if let Some(slot) = space.get_mut(i.index()) {
            *slot = s;
        }
    }
    // Nullness: a null constant reaches the node it enters and, along the flow, what takes it. A
    // load of a slot no store has written reads an undefined pointer, which may be taken as not
    // null.
    *held = held.saturating_add(g.flow.capacity() * std::mem::size_of::<(u32, u32)>());
    g.flow.sort_unstable();
    let mut stack: Vec<u32> = Vec::new();
    m.reserve(&mut stack, nodes)?;
    *held = held.saturating_add(nodes * std::mem::size_of::<u32>());
    for (k, &z) in g.null.iter().enumerate() {
        if z {
            stack.push(u32::try_from(k).map_err(|_| invariant("nodes past u32"))?);
        }
    }
    while let Some(from) = stack.pop() {
        m.tick(1)?;
        let start = g.flow.partition_point(|&(f, _)| f < from);
        for &(f, to) in g.flow.get(start..).unwrap_or(&[]) {
            if f != from {
                break;
            }
            if let Some(z) = g.null.get_mut(to as usize) {
                if !*z {
                    *z = true;
                    stack.push(to);
                }
            }
        }
    }
    g.null.truncate(n);
    Ok(Some(Plan {
        space,
        null: g.null,
        folds,
        leaves,
        leaf_space,
    }))
}

/// Whether type `ty` is an aggregate or a vector that holds a generic pointer.
fn holds(cx: &Cx<'_>, ty: Option<crate::ir::TypeId>) -> Result<bool, Error> {
    match ty {
        Some(t) if cx.parts.types.holds_pointer(t) && !cx.generic(t) => cx.holds_generic(t, 0),
        _ => Ok(false),
    }
}

/// [`Leaves::range`] of type `ty`, counted first.
fn range(
    leaves: &mut Option<Leaves>,
    cx: &Cx<'_>,
    ty: crate::ir::TypeId,
    indices: &[u32],
) -> Result<(u32, u32, bool), Error> {
    let l = leaves
        .as_mut()
        .ok_or_else(|| invariant("an aggregate without leaves"))?;
    l.count(cx.parts, ty, 0)?;
    l.range(cx.parts, ty, indices)
}

/// The links of instruction `i` (of `kind` and type `ty`), whose value is an aggregate or a
/// vector holding `count` leaves from `first`: each leaf takes what it comes from.
fn aggregate(
    g: &mut Groups<'_, '_>,
    leaves: &mut Option<Leaves>,
    (i, kind, ty): (InstId, &InstKind, crate::ir::TypeId),
    (first, count): (u32, u32),
    m: &mut Meter,
) -> Result<(), Error> {
    let cx = g.cx;
    match kind {
        InstKind::InsertValue { agg, val, indices } => {
            // Every leaf shares its aggregate operand's type; a leaf the inserted value replaces
            // takes no other pointer from it.
            let (lo, hi, exact) = range(leaves, cx, ty, indices)?;
            for k in 0..count {
                let to = g.leaf(first, k);
                let replaced = exact && (lo..hi).contains(&k);
                g.operand(to, (*agg, k, !replaced), leaves, i, m)?;
            }
            for k in lo..hi {
                let to = g.leaf(first, k);
                g.operand(to, (*val, k - lo, true), leaves, i, m)?;
            }
        }
        InstKind::ExtractValue { agg, indices } => {
            if matches!(agg, Value::Const(_)) {
                return Err(cx.refuse(GenericPointer::Aggregate, i));
            }
            let from_ty = cx.ty(*agg).ok_or_else(|| invariant("an operand's type"))?;
            let (lo, _, _) = range(leaves, cx, from_ty, indices)?;
            for k in 0..count {
                let to = g.leaf(first, k);
                g.operand(to, (*agg, lo + k, true), leaves, i, m)?;
            }
        }
        InstKind::Phi { .. }
        | InstKind::Select { .. }
        | InstKind::Freeze(_)
        | InstKind::ShuffleVector { .. } => {
            let operands = match kind {
                InstKind::ShuffleVector { a, b, .. } => vec![*a, *b],
                other => linked(other),
            };
            for v in operands {
                for k in 0..count {
                    let to = g.leaf(first, k);
                    g.operand(to, (v, k, true), leaves, i, m)?;
                }
            }
        }
        InstKind::InsertElement { vec, elt, .. } => {
            let to = g.leaf(first, 0);
            g.operand(to, (*vec, 0, true), leaves, i, m)?;
            g.operand(to, (*elt, 0, true), leaves, i, m)?;
        }
        _ => {
            let how = match kind {
                InstKind::Load { .. } => GenericPointer::Memory,
                InstKind::Call(_) => GenericPointer::Call,
                _ => GenericPointer::Aggregate,
            };
            for k in 0..count {
                let to = g.leaf(first, k);
                g.enter(to, Space::Many(how, Some(i)), false, i);
            }
        }
    }
    Ok(())
}
