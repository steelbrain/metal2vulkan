//! Thread memory that generic pointers pass through: an allocation whose every access is visible
//! holds, at each slot where a generic pointer is stored, the pointers stored there, so a generic
//! pointer loaded from the slot is in their group.
//!
//! A *cell* is an `alloca` that a generic pointer is loaded from or stored into. Its address
//! reaches each use through `getelementptr`s, bitcasts and `freeze`s, at a byte offset from the
//! cell's start that is a translation-time constant or unknown. A generic access at a constant
//! offset is at that offset's *slot*; one at an unknown offset may be at any slot of its cell, and
//! the cell's slots then share one group.
//!
//! A cell is *open*, and a generic pointer loaded from it has a space only the run knows
//! ([`GenericPointer::Memory`]), where a use of its address is other than a load's or a store's
//! address, a memory intrinsic's destination or source, a comparison or address arithmetic (the
//! address stored as a value, cast to an integer or to another space, passed to another call,
//! merged by a `phi` or a `select`, or held in an aggregate), or where bytes other than a generic
//! pointer's may be read as one: two slots overlap, a write of anything else (a store of another
//! type, an atomic operation, a copy or a `memset`) overlaps a slot or writes bytes the
//! translation cannot place, or a generic access at an unknown offset meets any such write.
//!
//! [`GenericPointer::Memory`]: crate::error::GenericPointer::Memory

use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::LlvmIntrinsic;
use crate::ir::Value;
use crate::ir::{AddrSpace, Callee, CastOp, Idx, InstId, InstKind, Layout, TypeId, TypeLayouts};

use super::super::pieces::{gep_offset, int};
use super::super::rewrite::{filled, invariant};
use super::Cx;

/// No cell.
const NONE: u32 = u32::MAX;

/// Where an address lies: in the `alloca` it is an instruction of, at a byte offset from its
/// start (`None` where not a translation-time constant).
type Place = Option<(u32, Option<u64>)>;

/// The generic loads and stores of thread allocations, and the slots they reach.
pub(super) struct Cells {
    /// Each generic load or store of a cell, in instruction order, and the slot it reaches:
    /// `None` where its cell is open.
    reach: Vec<(InstId, Option<u32>)>,
    /// How many slots there are.
    pub slots: usize,
    /// Slots in one group: an access at an unknown offset and each slot of its cell.
    pub links: Vec<(u32, u32)>,
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
    let found =
        places(cx, &layouts, m, held).and_then(|place| cells(cx, &layouts, &place, m, held));
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

/// Each instruction's [`Place`]: an `alloca`'s start, and what a `getelementptr`, a bitcast or a
/// `freeze` derives from an address that has one. Passes run in instruction order until none
/// changes, so an operand defined later is placed too.
fn places(
    cx: &Cx<'_>,
    layouts: &TypeLayouts,
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

fn cells(
    cx: &Cx<'_>,
    layouts: &TypeLayouts,
    place: &[Place],
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
    // The cell an address is in, and its offset there.
    let cell_of = |v: Value| -> Option<(u32, Option<u64>)> {
        let (a, off) = at(v)?;
        let c = *index.get(a as usize)?;
        (c != NONE).then_some((c, off))
    };
    let mut failed = None;
    for inst in cx.body.insts.values() {
        m.tick(1)?;
        uses(cx, layouts, &inst.kind, |u| {
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
    // A generic pointer's bytes.
    let width = u64::from(
        cx.parts
            .data_layout
            .pointer(AddrSpace::Generic.raw())
            .size_bits
            .div_ceil(8),
    );
    // Slot nodes: each cell's distinct offsets, then one for an access at an unknown offset.
    let mut first: Vec<u32> = Vec::new();
    m.reserve(&mut first, table.len())?;
    *held = held.saturating_add(table.len() * std::mem::size_of::<u32>());
    let mut slots = 0u32;
    let mut links = Vec::new();
    for cell in &mut table {
        m.tick(1)?;
        cell.slots.sort_unstable();
        cell.slots.dedup();
        cell.open |= !disciplined(cell, width);
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
    Ok(Some(Cells {
        reach,
        slots: slots as usize,
        links,
    }))
}

/// A use of an address other than a generic access, address arithmetic, a read or a comparison.
enum Use {
    /// The address leaves the uses the translation sees.
    Escape(Value),
    /// Bytes other than a generic pointer's are written at the address: so many, where a
    /// translation-time constant.
    Write(Value, Option<u64>),
}

/// The [`Use`]s `kind` makes of its operands. A generic store writes a generic pointer, and a
/// load, a copy's source and a comparison only read; a `getelementptr`, a bitcast and a `freeze`
/// derive the address their result is ([`places`]).
fn uses(cx: &Cx<'_>, layouts: &TypeLayouts, kind: &InstKind, mut on: impl FnMut(Use)) {
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
        InstKind::Store { val, ptr, .. } => {
            on(Use::Escape(*val));
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
