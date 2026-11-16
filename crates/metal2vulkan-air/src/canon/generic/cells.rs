//! Thread memory that generic pointers pass through: an allocation whose every access is visible
//! holds, at each slot where a generic pointer is stored, the pointers stored there, so a generic
//! pointer loaded from the slot is in their group.
//!
//! A *cell* is an `alloca` that a generic pointer is loaded from or stored into. Its address
//! reaches each use through `getelementptr`s, bitcasts, `freeze`s and casts between the thread and
//! the generic space, at a byte offset from the cell's start that is a translation-time constant
//! or unknown. A generic access at a constant offset is at that offset's *slot*; one at an unknown
//! offset may be at any slot of its cell, and the cell's slots then share one group.
//!
//! An address also reaches its uses through thread memory: a thread pointer stored into a
//! *followed* slot and loaded back is the address stored ([`follow`]), as a lambda that captures
//! a struct by reference holds the struct's address, and its call operator, inlined, loads it
//! from the lambda's object to read the struct's fields. A slot is followed where every pointer
//! stored into it is the address of one `alloca` at one offset (null among them or not), no other
//! write, copy or read at another offset or width meets its bytes, and no use of its own
//! `alloca`'s address is other than an access or address arithmetic. Storing an address into a
//! followed slot is no use of it that leaves the translation's sight.
//!
//! A cell is *open*, and a generic pointer loaded from it has a space only the run knows
//! ([`GenericPointer::Memory`]), where a use of its address is other than a load's or a store's
//! address, a memory intrinsic's destination or source, a comparison or address arithmetic (the
//! address stored as a value into a slot not followed, cast to an integer or to another space,
//! passed to another call, merged by a `phi` or a `select`, or held in an aggregate), or where
//! bytes other than a generic pointer's may be read as one: two slots overlap, a write of
//! anything else (a store of another type, an atomic operation, a copy other than a slot copy or
//! a `memset`) overlaps a slot or writes bytes the translation cannot place, or a generic access
//! at an unknown offset meets any such write.
//!
//! A *slot copy* is a `memcpy` or `memmove` of a translation-time constant length from one
//! `alloca` at a constant offset to another, as a struct holding a generic pointer is copied
//! whole. It moves each slot of the source that the copied bytes hold whole to the destination,
//! at the same distance from the copy's start: the destination slot takes the source slot's
//! pointers, and a chain of copies moves a slot along it (a destination that held none becomes a
//! cell). It is a slot copy only where neither cell has a generic access at an unknown offset, no
//! slot of either holds bytes on both sides of either end of the copied bytes, every slot of the
//! destination inside them is a moved one, and the source is not open; its other bytes are then
//! data copied onto data. Any other copy writes its bytes, as above.
//!
//! [`GenericPointer::Memory`]: crate::error::GenericPointer::Memory

use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::LlvmIntrinsic;
use crate::ir::Value;
use crate::ir::{
    AddrSpace, Callee, CastOp, Const, Idx, InstId, InstKind, Layout, Type, TypeId, TypeLayouts,
};

use super::super::pieces::{gep_offset, int};
use super::super::rewrite::{filled, invariant};
use super::Cx;

/// No cell.
const NONE: u32 = u32::MAX;

/// Where an address lies: in the `alloca` it is an instruction of, at a byte offset from its
/// start (`None` where not a translation-time constant).
type Place = Option<(u32, Option<u64>)>;

/// A slot of an `alloca` that thread pointers pass through: the `alloca`'s index and the slot's
/// byte offset.
type Slot = (u32, u64);

/// The followed slots, sorted, each with the address every pointer stored there is
/// ([`follow`]).
type Followed = Vec<(Slot, (u32, Option<u64>))>;

/// The generic loads and stores of thread allocations, and the slots they reach.
pub(super) struct Cells {
    /// Each generic load or store of a cell, in instruction order, and the slot it reaches:
    /// `None` where its cell is open.
    reach: Vec<(InstId, Option<u32>)>,
    /// How many slots there are.
    pub slots: usize,
    /// Slots in one group: an access at an unknown offset and each slot of its cell.
    pub links: Vec<(u32, u32)>,
    /// The slots slot copies move: each destination slot, which takes the pointers of the
    /// source slot that follows it.
    pub moves: Vec<(u32, u32)>,
}

impl Cells {
    /// The slot generic load or store `i` reaches: `None` where `i` is no generic access of a
    /// cell, `Some(None)` where its cell is open.
    pub fn reach(&self, i: InstId) -> Option<Option<u32>> {
        self.reach
            .binary_search_by_key(&i, |&(j, _)| j)
            .ok()
            .and_then(|k| self.reach.get(k))
            .map(|&(_, s)| s)
    }
}

/// One cell's accesses: the generic accesses' offsets, and the writes of other bytes.
#[derive(Clone, Default)]
struct Cell {
    /// Whether some use of its address is no access.
    open: bool,
    /// The constant offsets of its generic accesses.
    slots: Vec<u64>,
    /// Whether a generic access lies at an unknown offset.
    anywhere: bool,
    /// The byte ranges other writes cover; `None` for one the translation cannot place.
    writes: Vec<Option<(u64, u64)>>,
}

/// The cells of the entry's body, `None` when no generic pointer is loaded from or stored into
/// an `alloca`.
pub(super) fn find(cx: &Cx<'_>, m: &mut Meter, held: &mut usize) -> Result<Option<Cells>, Error> {
    let mut accesses = false;
    for inst in cx.body.insts.values() {
        m.tick(1)?;
        accesses |= generic_access(cx, &inst.kind, inst.ty).is_some();
    }
    if !accesses {
        return Ok(None);
    }
    let layouts = TypeLayouts::new(&cx.parts.data_layout, &cx.parts.types, m)?;
    let found = followed_places(cx, &layouts, m, held)
        .and_then(|(place, followed)| cells(cx, &layouts, (&place, &followed), m, held));
    m.release(cx.parts.types.len() * std::mem::size_of::<Option<Layout>>());
    found
}

/// The address a generic load or store accesses.
fn generic_access(cx: &Cx<'_>, kind: &InstKind, ty: TypeId) -> Option<Value> {
    match kind {
        InstKind::Load { ptr, .. } if cx.generic(ty) => Some(*ptr),
        InstKind::Store { val, ptr, .. } if cx.is_generic(*val) => Some(*ptr),
        _ => None,
    }
}

/// Whether `ty` is a thread pointer.
fn thread(cx: &Cx<'_>, ty: TypeId) -> bool {
    matches!(
        cx.parts.types.get(ty),
        Some(Type::Ptr {
            space: AddrSpace::Thread
        })
    )
}

/// Whether `kind` stores a thread pointer.
fn pointer_store(cx: &Cx<'_>, kind: &InstKind) -> bool {
    matches!(kind, InstKind::Store { val, .. } if cx.ty(*val).is_some_and(|t| thread(cx, t)))
}

/// Whether an `addrspacecast` of `v` to `ty` goes between the thread and the generic space, so
/// its result points where `v` does.
fn thread_generic(cx: &Cx<'_>, v: Value, ty: TypeId) -> bool {
    let Some(from) = cx.ty(v) else {
        return false;
    };
    (thread(cx, from) && cx.generic(ty)) || (cx.generic(from) && thread(cx, ty))
}

/// Whether `v` is a null pointer constant.
fn null(cx: &Cx<'_>, v: Value) -> bool {
    match v {
        Value::Const(c) => matches!(cx.parts.consts.get(c).map(|k| &k.value), Some(Const::Null)),
        _ => false,
    }
}

/// Each instruction's [`Place`] and the slots followed, found together: the slots [`follow`]
/// takes from the places so far, then the places they lead to, until neither changes. A slot
/// followed once and not after is never followed again, so the rounds end; past their bound no
/// slot is followed.
fn followed_places(
    cx: &Cx<'_>,
    layouts: &TypeLayouts,
    m: &mut Meter,
    held: &mut usize,
) -> Result<(Vec<Place>, Followed), Error> {
    let stores = cx
        .body
        .insts
        .values()
        .filter(|inst| pointer_store(cx, &inst.kind))
        .count();
    let mut followed = Followed::new();
    let mut ruled_out: Vec<Slot> = Vec::new();
    for _ in 0..=2 * stores + 1 {
        let place = places(cx, layouts, &followed, m, held)?;
        let next = follow(cx, layouts, (&place, &followed, &ruled_out), m, held)?;
        if next == followed {
            give_back(m, held, next);
            give_back(m, held, ruled_out);
            return Ok((place, followed));
        }
        for &(slot, _) in &followed {
            if next.binary_search_by_key(&slot, |&(s, _)| s).is_err() {
                m.reserve(&mut ruled_out, 1)?;
                *held = held.saturating_add(std::mem::size_of::<Slot>());
                ruled_out.push(slot);
            }
        }
        ruled_out.sort_unstable();
        ruled_out.dedup();
        give_back(m, held, place);
        give_back(m, held, followed);
        followed = next;
    }
    give_back(m, held, followed);
    give_back(m, held, ruled_out);
    let none = Followed::new();
    Ok((places(cx, layouts, &none, m, held)?, none))
}

/// Give back what the meter charged for `v`: its elements, each charged as [`filled`] and a
/// one-element `reserve` charge them.
fn give_back<T>(m: &mut Meter, held: &mut usize, v: Vec<T>) {
    let bytes = v.len().saturating_mul(std::mem::size_of::<T>());
    m.release(bytes);
    *held = held.saturating_sub(bytes);
}

/// Each instruction's [`Place`]: an `alloca`'s start, what a `getelementptr`, a bitcast, a
/// `freeze` or a cast between the thread and the generic space derives from an address that has
/// one, and a thread pointer loaded from a slot of `followed`. Passes run in instruction order
/// until none changes, so an operand defined later is placed too.
fn places(
    cx: &Cx<'_>,
    layouts: &TypeLayouts,
    followed: &Followed,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Vec<Place>, Error> {
    let n = cx.body.insts.len();
    let mut place: Vec<Place> = filled(m, held, n, None)?;
    let mut changed = true;
    let mut rounds = 0;
    while changed {
        changed = false;
        rounds += 1;
        if rounds > n + 1 {
            return Err(invariant("address derivations do not settle"));
        }
        for (i, inst) in cx.body.insts.iter() {
            m.tick(1)?;
            let of = |v: Value| match v {
                Value::Inst(j) => place.get(j.index()).copied().flatten(),
                _ => None,
            };
            let p = match &inst.kind {
                InstKind::Alloca { .. } => {
                    let k = u32::try_from(i.index()).map_err(|_| invariant("ids past u32"))?;
                    Some((k, Some(0)))
                }
                InstKind::Gep {
                    src_ty,
                    base,
                    indices,
                    ..
                } => of(*base).map(|(a, at)| {
                    let step = gep_offset(cx.parts, layouts, *src_ty, indices);
                    (a, at.zip(step).and_then(|(x, y)| x.checked_add(y)))
                }),
                InstKind::Cast {
                    op: CastOp::Bitcast,
                    v,
                }
                | InstKind::Freeze(v) => of(*v),
                InstKind::Cast {
                    op: CastOp::AddrSpaceCast,
                    v,
                } if thread_generic(cx, *v, inst.ty) => of(*v),
                InstKind::Load { ptr, .. } if thread(cx, inst.ty) => match of(*ptr) {
                    Some((a, Some(at))) => followed
                        .binary_search_by_key(&(a, at), |&(s, _)| s)
                        .ok()
                        .and_then(|k| followed.get(k))
                        .map(|&(_, to)| to),
                    _ => None,
                },
                _ => None,
            };
            if let Some(slot) = place.get_mut(i.index()) {
                if *slot != p {
                    *slot = p;
                    changed = true;
                }
            }
        }
    }
    Ok(place)
}

/// One slot thread pointers may pass through: where the pointers stored into it point, while
/// they agree, and whether anything rules it out.
struct Candidate {
    slot: Slot,
    to: Option<(u32, Option<u64>)>,
    ok: bool,
}

/// The slots thread pointers pass through, under `place` and the slots `followed` so far (the
/// module docs): an `alloca`'s slot at a constant offset that a thread pointer is stored into,
/// whose stores all store a null or the address of one `alloca` at one offset, that no other
/// store, atomic operation, copy, `memset` or load at another offset or width meets (a load or
/// a copy of the bytes elsewhere reads them as no pointer the translation follows), that no
/// other such slot overlaps, of an `alloca` whose address no use but an access and address
/// arithmetic takes (a store of it into a slot of `followed` among them). No slot of
/// `ruled_out` is one.
fn follow(
    cx: &Cx<'_>,
    layouts: &TypeLayouts,
    (place, followed, ruled_out): (&[Place], &Followed, &[Slot]),
    m: &mut Meter,
    held: &mut usize,
) -> Result<Followed, Error> {
    let n = cx.body.insts.len();
    let at = |v: Value| match v {
        Value::Inst(j) => place.get(j.index()).copied().flatten(),
        _ => None,
    };
    let width = u64::from(
        cx.parts
            .data_layout
            .pointer(AddrSpace::Thread.raw())
            .size_bits
            .div_ceil(8),
    );
    let mut candidates: Vec<Candidate> = Vec::new();
    for inst in cx.body.insts.values() {
        m.tick(1)?;
        let InstKind::Store { val, ptr, .. } = &inst.kind else {
            continue;
        };
        if !cx.ty(*val).is_some_and(|t| thread(cx, t)) {
            continue;
        }
        let Some((a, Some(off))) = at(*ptr) else {
            continue;
        };
        if ruled_out.binary_search(&(a, off)).is_ok() {
            continue;
        }
        m.reserve(&mut candidates, 1)?;
        *held = held.saturating_add(std::mem::size_of::<Candidate>());
        let to = at(*val);
        let ok = to.is_some() || null(cx, *val);
        candidates.push(Candidate {
            slot: (a, off),
            to,
            ok,
        });
    }
    if candidates.is_empty() {
        return Ok(Followed::new());
    }
    // One candidate a slot: its stores' addresses agree, and a null joins any.
    candidates.sort_by_key(|c| c.slot);
    let stored = candidates
        .len()
        .saturating_mul(std::mem::size_of::<Candidate>());
    let mut merged: Vec<Candidate> = Vec::new();
    for c in candidates {
        match merged.last_mut() {
            Some(last) if last.slot == c.slot => {
                last.ok &= c.ok;
                match (last.to, c.to) {
                    (Some(x), Some(y)) if x != y => last.ok = false,
                    (None, y) => last.to = y,
                    _ => {}
                }
            }
            _ => {
                m.reserve(&mut merged, 1)?;
                *held = held.saturating_add(std::mem::size_of::<Candidate>());
                merged.push(c);
            }
        }
    }
    m.release(stored);
    *held = held.saturating_sub(stored);
    // Two slots of one `alloca` whose pointers overlap.
    for k in 1..merged.len() {
        let (before, after) = merged.split_at_mut(k);
        if let (Some(a), Some(b)) = (before.last_mut(), after.first_mut()) {
            if a.slot.0 == b.slot.0 && b.slot.1 - a.slot.1 < width {
                a.ok = false;
                b.ok = false;
            }
        }
    }
    // Rule out the candidates that `(a, off, len)` meets: every one of `a` where the offset or
    // the length is unknown.
    let meets = |merged: &mut [Candidate], (a, off, len): (u32, Option<u64>, Option<u64>)| {
        let lo = merged.partition_point(|c| c.slot.0 < a);
        let rest = merged.get_mut(lo..).unwrap_or_default();
        for c in rest.iter_mut().take_while(|c| c.slot.0 == a) {
            let hit = match off.zip(len) {
                Some((o, l)) => {
                    o < c.slot.1.saturating_add(width) && c.slot.1 < o.saturating_add(l)
                }
                None => true,
            };
            if hit {
                c.ok = false;
            }
        }
    };
    let size = |v: Value| cx.ty(v).and_then(|t| layouts.store_size(t));
    let mut open: Vec<bool> = filled(m, held, n, false)?;
    let exempt = |ptr: Value| {
        at(ptr).is_some_and(|(a, off)| {
            off.is_some_and(|o| followed.binary_search_by_key(&(a, o), |&(s, _)| s).is_ok())
        })
    };
    for inst in cx.body.insts.values() {
        m.tick(1)?;
        match &inst.kind {
            InstKind::Store { val, ptr, .. } => {
                let pointer = cx.ty(*val).is_some_and(|t| thread(cx, t));
                if let Some((a, off)) = at(*ptr) {
                    if !pointer || off.is_none() {
                        meets(&mut merged, (a, off, size(*val)));
                    }
                }
            }
            InstKind::Load { ptr, .. } => {
                if let Some((a, off)) = at(*ptr) {
                    let pointer = thread(cx, inst.ty) && off.is_some();
                    let exact =
                        pointer && off.is_some_and(|o| merged.iter().any(|c| c.slot == (a, o)));
                    if !exact {
                        meets(&mut merged, (a, off, layouts.store_size(inst.ty)));
                    }
                }
            }
            InstKind::AtomicRmw { ptr, val, .. } => {
                if let Some((a, off)) = at(*ptr) {
                    meets(&mut merged, (a, off, size(*val)));
                }
            }
            InstKind::CmpXchg { ptr, new, .. } => {
                if let Some((a, off)) = at(*ptr) {
                    meets(&mut merged, (a, off, size(*new)));
                }
            }
            InstKind::Call(call) => {
                if let (
                    Callee::Llvm {
                        op:
                            LlvmIntrinsic::Memset { .. }
                            | LlvmIntrinsic::Memcpy { .. }
                            | LlvmIntrinsic::Memmove { .. },
                        ..
                    },
                    [dst, src, len, ..],
                ) = (&call.callee, &*call.args)
                {
                    let len = int(cx.parts, *len);
                    for v in [*dst, *src] {
                        if let Some((a, off)) = at(v) {
                            meets(&mut merged, (a, off, len));
                        }
                    }
                }
            }
            _ => {}
        }
        uses(cx, layouts, (&inst.kind, inst.ty), &exempt, |u| {
            if let Use::Escape(v) = u {
                if let Some((a, _)) = at(v) {
                    if let Some(o) = open.get_mut(a as usize) {
                        *o = true;
                    }
                }
            }
        });
    }
    let mut next = Followed::new();
    for c in &merged {
        let closed = !open.get(c.slot.0 as usize).copied().unwrap_or(true);
        if let (true, true, Some(to)) = (c.ok, closed, c.to) {
            m.reserve(&mut next, 1)?;
            *held = held.saturating_add(std::mem::size_of::<(Slot, (u32, Option<u64>))>());
            next.push((c.slot, to));
        }
    }
    give_back(m, held, merged);
    give_back(m, held, open);
    Ok(next)
}

fn cells(
    cx: &Cx<'_>,
    layouts: &TypeLayouts,
    (place, followed): (&[Place], &Followed),
    m: &mut Meter,
    held: &mut usize,
) -> Result<Option<Cells>, Error> {
    let n = cx.body.insts.len();
    let at = |v: Value| match v {
        Value::Inst(j) => place.get(j.index()).copied().flatten(),
        _ => None,
    };
    // Each cell's index, by its `alloca`.
    let mut index: Vec<u32> = filled(m, held, n, NONE)?;
    let mut table: Vec<Cell> = Vec::new();
    let mut generic: Vec<(InstId, u32, Option<u64>)> = Vec::new();
    for (i, inst) in cx.body.insts.iter() {
        m.tick(1)?;
        let Some((a, off)) = generic_access(cx, &inst.kind, inst.ty).and_then(at) else {
            continue;
        };
        let slot = index
            .get_mut(a as usize)
            .ok_or_else(|| invariant("an alloca out of range"))?;
        if *slot == NONE {
            *slot = u32::try_from(table.len()).map_err(|_| invariant("cells past u32"))?;
            m.reserve(&mut table, 1)?;
            *held = held.saturating_add(std::mem::size_of::<Cell>());
            table.push(Cell::default());
        }
        m.reserve(&mut generic, 1)?;
        *held = held.saturating_add(std::mem::size_of::<(InstId, u32, Option<u64>)>());
        generic.push((i, *slot, off));
    }
    if generic.is_empty() {
        return Ok(None);
    }
    for &(_, c, off) in &generic {
        let cell = table
            .get_mut(c as usize)
            .ok_or_else(|| invariant("a cell out of range"))?;
        match off {
            Some(o) => {
                m.reserve(&mut cell.slots, 1)?;
                *held = held.saturating_add(std::mem::size_of::<u64>());
                cell.slots.push(o);
            }
            None => cell.anywhere = true,
        }
    }
    for cell in &mut table {
        m.tick(1)?;
        cell.slots.sort_unstable();
        cell.slots.dedup();
    }
    // A generic pointer's bytes.
    let width = u64::from(
        cx.parts
            .data_layout
            .pointer(AddrSpace::Generic.raw())
            .size_bits
            .div_ceil(8),
    );
    let mut copies = slot_copies(cx, &at, m, held)?;
    moved_slots(&copies, (&mut index, &mut table), width, m, held)?;
    for c in &mut copies {
        m.tick(1)?;
        c.moves = moves(c, (&index, &table), width);
    }
    // The cell an address is in, and its offset there.
    let cell_of = |v: Value| -> Option<(u32, Option<u64>)> {
        let (a, off) = at(v)?;
        let c = *index.get(a as usize)?;
        (c != NONE).then_some((c, off))
    };
    let exempt = |ptr: Value| {
        at(ptr).is_some_and(|(a, off)| {
            off.is_some_and(|o| followed.binary_search_by_key(&(a, o), |&(s, _)| s).is_ok())
        })
    };
    let mut failed = None;
    for (i, inst) in cx.body.insts.iter() {
        m.tick(1)?;
        if copies
            .binary_search_by_key(&i, |c| c.call)
            .is_ok_and(|k| copies.get(k).is_some_and(|c| c.moves))
        {
            continue;
        }
        uses(cx, layouts, (&inst.kind, inst.ty), &exempt, |u| {
            let (v, write) = match u {
                Use::Escape(v) => (v, None),
                Use::Write(v, len) => (v, Some(len)),
            };
            let Some((c, off)) = cell_of(v) else { return };
            let Some(cell) = table.get_mut(c as usize) else {
                return;
            };
            match write {
                None => cell.open = true,
                Some(len) => {
                    let range = off
                        .zip(len)
                        .and_then(|(lo, len)| Some((lo, lo.checked_add(len)?)));
                    if let Err(e) = m.reserve(&mut cell.writes, 1) {
                        failed.get_or_insert(e);
                        return;
                    }
                    *held = held.saturating_add(std::mem::size_of::<Option<(u64, u64)>>());
                    cell.writes.push(range);
                }
            }
        });
        if let Some(e) = failed.take() {
            return Err(e);
        }
    }
    for cell in &mut table {
        m.tick(1)?;
        cell.open |= !disciplined(cell, width);
    }
    // A copy from an open cell moves no slot: it writes bytes the translation cannot read as
    // pointers, which may open its destination, and so on along a chain.
    let mut changed = true;
    while changed {
        changed = false;
        for c in &mut copies {
            m.tick(1)?;
            let (s, d) = (cell_at(&index, c.src.0), cell_at(&index, c.dst.0));
            let src_open = s.and_then(|s| table.get(s)).is_some_and(|x| x.open);
            if !(c.moves && src_open) {
                continue;
            }
            c.moves = false;
            changed = true;
            if let Some(cell) = d.and_then(|d| table.get_mut(d)) {
                m.reserve(&mut cell.writes, 1)?;
                *held = held.saturating_add(std::mem::size_of::<Option<(u64, u64)>>());
                cell.writes
                    .push(c.dst.1.checked_add(c.len).map(|hi| (c.dst.1, hi)));
                cell.open |= !disciplined(cell, width);
            }
        }
    }
    // Slot nodes: each cell's distinct offsets, then one for an access at an unknown offset.
    let mut first: Vec<u32> = Vec::new();
    m.reserve(&mut first, table.len())?;
    *held = held.saturating_add(table.len() * std::mem::size_of::<u32>());
    let mut slots = 0u32;
    let mut links = Vec::new();
    for cell in &mut table {
        m.tick(1)?;
        first.push(slots);
        let count = u32::try_from(cell.slots.len() + usize::from(cell.anywhere))
            .map_err(|_| invariant("slots past u32"))?;
        if cell.anywhere && !cell.open {
            let any = slots + count - 1;
            for k in 0..count - 1 {
                m.reserve(&mut links, 1)?;
                *held = held.saturating_add(std::mem::size_of::<(u32, u32)>());
                links.push((slots + k, any));
            }
        }
        slots = slots
            .checked_add(count)
            .ok_or_else(|| invariant("slots past u32"))?;
    }
    let mut reach = Vec::new();
    m.reserve(&mut reach, generic.len())?;
    *held = held.saturating_add(generic.len() * std::mem::size_of::<(InstId, Option<u32>)>());
    for &(i, c, off) in &generic {
        let cell = table
            .get(c as usize)
            .ok_or_else(|| invariant("a cell out of range"))?;
        let base = *first
            .get(c as usize)
            .ok_or_else(|| invariant("a cell out of range"))?;
        let node = if cell.open {
            None
        } else {
            let k = match off {
                Some(o) => cell
                    .slots
                    .binary_search(&o)
                    .map_err(|_| invariant("a slot not found"))?,
                None => cell.slots.len(),
            };
            Some(base + u32::try_from(k).map_err(|_| invariant("slots past u32"))?)
        };
        reach.push((i, node));
    }
    // Each slot a copy moves, by node: the destination's takes the source's.
    let node = |a: u32, o: u64| -> Option<u32> {
        let c = cell_at(&index, a)?;
        let k = table.get(c)?.slots.binary_search(&o).ok()?;
        first.get(c)?.checked_add(u32::try_from(k).ok()?)
    };
    let mut moved = Vec::new();
    for c in copies.iter().filter(|c| c.moves) {
        let Some(src) = cell_at(&index, c.src.0).and_then(|s| table.get(s)) else {
            continue;
        };
        for &o in &src.slots {
            m.tick(1)?;
            let Some(t) = c.image(o, width) else { continue };
            let (to, from) = node(c.dst.0, t)
                .zip(node(c.src.0, o))
                .ok_or_else(|| invariant("a moved slot not found"))?;
            m.reserve(&mut moved, 1)?;
            *held = held.saturating_add(std::mem::size_of::<(u32, u32)>());
            moved.push((to, from));
        }
    }
    give_back(m, held, copies);
    Ok(Some(Cells {
        reach,
        slots: slots as usize,
        links,
        moves: moved,
    }))
}

/// A `memcpy` or `memmove` of a translation-time constant length from one `alloca` at a constant
/// offset to another: the call, the destination's `alloca` and offset, the source's, the length,
/// and whether it is a *slot copy*, which moves the source's slots to the destination, or writes
/// its bytes as any other copy does.
struct SlotCopy {
    call: InstId,
    dst: (u32, u64),
    src: (u32, u64),
    len: u64,
    moves: bool,
}

impl SlotCopy {
    /// Where the copy puts source slot `o`, where the copied bytes hold it whole.
    fn image(&self, o: u64, width: u64) -> Option<u64> {
        let (_, lo) = self.src;
        let hi = lo.checked_add(self.len)?;
        let end = o.checked_add(width)?;
        if o < lo || end > hi {
            return None;
        }
        self.dst.1.checked_add(o - lo)
    }
}

/// The cell of `alloca` `a`, if it is one.
fn cell_at(index: &[u32], a: u32) -> Option<usize> {
    let c = *index.get(a as usize)?;
    (c != NONE).then_some(c as usize)
}

/// The `memcpy`s and `memmove`s of a translation-time constant length from one `alloca` at a
/// constant offset to another, in instruction order.
fn slot_copies(
    cx: &Cx<'_>,
    at: &dyn Fn(Value) -> Place,
    m: &mut Meter,
    held: &mut usize,
) -> Result<Vec<SlotCopy>, Error> {
    let mut copies = Vec::new();
    for (i, inst) in cx.body.insts.iter() {
        m.tick(1)?;
        let InstKind::Call(call) = &inst.kind else {
            continue;
        };
        let (
            Callee::Llvm {
                op: LlvmIntrinsic::Memcpy { .. } | LlvmIntrinsic::Memmove { .. },
                ..
            },
            [dst, src, len, ..],
        ) = (&call.callee, &*call.args)
        else {
            continue;
        };
        let (Some((a, Some(d))), Some((b, Some(s))), Some(len)) =
            (at(*dst), at(*src), int(cx.parts, *len))
        else {
            continue;
        };
        if a == b {
            continue;
        }
        m.reserve(&mut copies, 1)?;
        *held = held.saturating_add(std::mem::size_of::<SlotCopy>());
        copies.push(SlotCopy {
            call: i,
            dst: (a, d),
            src: (b, s),
            len,
            moves: false,
        });
    }
    Ok(copies)
}

/// Give each copy's destination the slots of its source that the copied bytes hold whole, at
/// the same distance from the copy's start, until no copy adds one: a chain of copies moves a
/// slot along it. A destination that held no slot becomes a cell.
fn moved_slots(
    copies: &[SlotCopy],
    (index, table): (&mut [u32], &mut Vec<Cell>),
    width: u64,
    m: &mut Meter,
    held: &mut usize,
) -> Result<(), Error> {
    let mut changed = true;
    while changed {
        changed = false;
        for c in copies {
            m.tick(1)?;
            let Some(src) = cell_at(index, c.src.0).and_then(|s| table.get(s)) else {
                continue;
            };
            let mut moved: Vec<u64> = Vec::new();
            for &o in &src.slots {
                m.tick(1)?;
                if let Some(t) = c.image(o, width) {
                    m.reserve(&mut moved, 1)?;
                    *held = held.saturating_add(std::mem::size_of::<u64>());
                    moved.push(t);
                }
            }
            if moved.is_empty() {
                continue;
            }
            let slot = index
                .get_mut(c.dst.0 as usize)
                .ok_or_else(|| invariant("an alloca out of range"))?;
            if *slot == NONE {
                *slot = u32::try_from(table.len()).map_err(|_| invariant("cells past u32"))?;
                m.reserve(table, 1)?;
                *held = held.saturating_add(std::mem::size_of::<Cell>());
                table.push(Cell::default());
            }
            let dst = table
                .get_mut(*slot as usize)
                .ok_or_else(|| invariant("a cell out of range"))?;
            for &t in &moved {
                if let Err(k) = dst.slots.binary_search(&t) {
                    m.reserve(&mut dst.slots, 1)?;
                    *held = held.saturating_add(std::mem::size_of::<u64>());
                    dst.slots.insert(k, t);
                    changed = true;
                }
            }
            give_back(m, held, moved);
        }
    }
    Ok(())
}

/// Whether copy `c` is a slot copy: neither cell has a generic access at an unknown offset, no
/// slot of either holds bytes on both sides of either end of the copied bytes, and each slot of
/// the destination inside them is the image of a slot of the source, so every generic pointer the
/// destination's slots read there is one the source's slots held.
fn moves(c: &SlotCopy, (index, table): (&[u32], &[Cell]), width: u64) -> bool {
    let cell = |a: u32| cell_at(index, a).and_then(|k| table.get(k));
    let (Some(src), Some(dst)) = (cell(c.src.0), cell(c.dst.0)) else {
        return false;
    };
    if src.anywhere || dst.anywhere {
        return false;
    }
    if straddles(&src.slots, c.src.1, c.len, width) || straddles(&dst.slots, c.dst.1, c.len, width)
    {
        return false;
    }
    let (lo, hi) = (c.dst.1, c.dst.1.saturating_add(c.len));
    let from = dst.slots.partition_point(|&t| t < lo);
    dst.slots
        .get(from..)
        .unwrap_or_default()
        .iter()
        .take_while(|&&t| t < hi)
        .all(|&t| src.slots.binary_search(&(t - lo + c.src.1)).is_ok())
}

/// Whether a slot of `slots` (sorted) holds bytes on both sides of either end of the `len` bytes
/// at `lo`.
fn straddles(slots: &[u64], lo: u64, len: u64, width: u64) -> bool {
    let across = |x: u64| {
        let k = slots.partition_point(|&s| s.saturating_add(width) <= x);
        slots.get(k).is_some_and(|&s| s < x)
    };
    across(lo) || across(lo.saturating_add(len))
}

/// A use of an address other than a generic access, address arithmetic, a read or a comparison.
enum Use {
    /// The address leaves the uses the translation sees.
    Escape(Value),
    /// Bytes other than a generic pointer's are written at the address: so many, where a
    /// translation-time constant.
    Write(Value, Option<u64>),
}

/// The [`Use`]s `kind` (of result type `ty`) makes of its operands. A generic store writes a
/// generic pointer, and a load, a copy's source and a comparison only read; a `getelementptr`, a
/// bitcast, a `freeze` and a cast between the thread and the generic space derive the address
/// their result is ([`places`]). A store into an address `exempt` takes, a followed slot's, is no
/// escape of the address it stores ([`follow`]).
fn uses(
    cx: &Cx<'_>,
    layouts: &TypeLayouts,
    (kind, ty): (&InstKind, TypeId),
    exempt: &dyn Fn(Value) -> bool,
    mut on: impl FnMut(Use),
) {
    let size = |v: Value| cx.ty(v).and_then(|t| layouts.store_size(t));
    match kind {
        InstKind::Gep { .. }
        | InstKind::Cast {
            op: CastOp::Bitcast,
            ..
        }
        | InstKind::Freeze(_)
        | InstKind::Load { .. }
        | InstKind::ICmp { .. } => {}
        InstKind::Cast {
            op: CastOp::AddrSpaceCast,
            v,
        } if thread_generic(cx, *v, ty) => {}
        InstKind::Store { val, ptr, .. } => {
            if !exempt(*ptr) {
                on(Use::Escape(*val));
            }
            if !cx.is_generic(*val) {
                on(Use::Write(*ptr, size(*val)));
            }
        }
        InstKind::AtomicRmw { ptr, val, .. } => {
            on(Use::Escape(*val));
            on(Use::Write(*ptr, size(*val)));
        }
        InstKind::CmpXchg { ptr, cmp, new, .. } => {
            on(Use::Escape(*cmp));
            on(Use::Escape(*new));
            on(Use::Write(*ptr, size(*new)));
        }
        InstKind::Call(call) => match (&call.callee, &*call.args) {
            (
                Callee::Llvm {
                    op:
                        LlvmIntrinsic::Memset { .. }
                        | LlvmIntrinsic::Memcpy { .. }
                        | LlvmIntrinsic::Memmove { .. },
                    ..
                },
                [dst, _, len, ..],
            ) => on(Use::Write(*dst, int(cx.parts, *len))),
            _ => kind.for_each_value(|v| on(Use::Escape(v))),
        },
        kind => kind.for_each_value(|v| on(Use::Escape(v))),
    }
}

/// Whether only generic pointers' bytes are read as generic pointers from `cell`, its slots
/// sorted: no two slots overlap, no other write overlaps a slot or lies where the translation
/// cannot place it, and no write meets an access at an unknown offset.
fn disciplined(cell: &Cell, width: u64) -> bool {
    if cell.slots.windows(2).any(|w| match w {
        [a, b] => b - a < width,
        _ => false,
    }) {
        return false;
    }
    if cell.anywhere && !cell.writes.is_empty() {
        return false;
    }
    cell.writes.iter().all(|w| match *w {
        None => false,
        Some((lo, hi)) => {
            // The first slot that ends past `lo` starts at or after `hi`.
            let k = cell
                .slots
                .partition_point(|&s| s.saturating_add(width) <= lo);
            cell.slots.get(k).is_none_or(|&s| s >= hi)
        }
    })
}
