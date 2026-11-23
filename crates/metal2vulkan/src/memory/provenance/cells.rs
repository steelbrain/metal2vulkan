//! Pointer cells: store-to-load forwarding through closed allocas.
//!
//! A pointer stored into an alloca and loaded back is, without this, a pointer loaded from
//! memory: `Dynamic`, resolved at run time through the resident set, which Profile F lacks and
//! which no handle, arena, or buffer-table pointer can take. When every access to the alloca is
//! visible, the loaded pointer is instead one of the pointers stored at its slot, so its fact is
//! the join of theirs, and the cell holds each pointer in its lowered representation
//! (`lower::access::cell`).
//!
//! **Slots.** A pointer access into a cell lands at its exact byte offset ([`Slot::At`]) or, when
//! the offset is not a translation-time constant, anywhere ([`Slot::Any`]). A load at `o` reads the
//! pointers stored at `o` and at `Any`; a load at `Any` reads every slot of the cell.
//!
//! **Closure.** The analysis first assumes every alloca forwards, runs the fixpoint, and then
//! verifies each alloca against the final facts ([`verify`]). An alloca is **open** (its pointer
//! loads stay `Dynamic`) when some instruction reaches it other than through address arithmetic
//! (`getelementptr`, casts, `phi`, `select`, `freeze`), a plain load or store address, a memory
//! intrinsic's operand (`memset`, `memcpy`, `memmove`: its bytes are a data access), a compare,
//! or a pointer stored into a forwarding cell: an atomic, any other call, `ptrtoint`, an
//! aggregate holding it that leaves SSA, a load or store of an aggregate that holds pointers, or
//! a store of its pointer into other memory. An aggregate stays a value when its every use, and
//! every use of each aggregate built on it, is an `extractvalue`, the aggregate operand of an
//! `insertvalue`, or an incoming value of a `phi` of such aggregates (a struct of `array_ref`s
//! passed by value to an inlined function): a pointer inserted into it reaches only the
//! extractions of it, which provenance reads as the pointer itself, so its accesses are theirs.
//! It is open too when its slots are not disciplined: two pointer slots overlap, one slot is
//! accessed as two pointer types, a pointer access at an unknown offset meets any data access, a
//! data access (a load or store of a non-pointer, a memory intrinsic's bytes) overlaps a pointer
//! slot, or a data access's bytes are not translation-time constants; unless it is a pool
//! (below). And every alloca is open when some
//! access goes through a thread-space pointer that may be `Dynamic` (it may alias any alloca). A
//! verification that opens allocas or makes pools reruns the fixpoint with them open or pooled; a
//! second one opens every alloca, so the analysis runs at most three rounds.
//!
//! **Pools**. An alloca reached only through the accesses above whose slots are not disciplined,
//! every pointer access of which is of a thread-space pointer, and every pointer it holds an arena
//! pointer (null, a stack allocation, or `Dynamic`; not a table parameter's), forwards as a pool
//! ([`CellFacts::pooled`]): its data may overlap its pointers, and its pointer accesses may be at
//! any offset (an array of structs indexed at run time, each holding a thread pointer into the
//! array). Each round verifies its pools again: one reached otherwise, or holding another pointer,
//! opens. A pointer loaded from a pool is any pointer whose bytes may be there, or bytes no pointer
//! wrote: the join of the pointers that leak, and null. A pointer leaks when it is stored into a
//! pool; when it is a pool's (data read from a pool may carry its bytes anywhere); when it is
//! stored into memory that does not forward it (data read back from there may carry it); and when
//! an instruction turns it into data or hands it to code no cell sees ([`escapes`]: `ptrtoint`, an
//! aggregate or vector built of it, a call other than an intrinsic's, an atomic's value, a return).
//! Every pool holds every leaked pointer, so a pointer moved as data (a byte copy, a `memcpy`, an
//! integer of its bits) between pools, or out of any memory into a pool, is among its loads'
//! pointers. A pointer that is `Dynamic` among them makes the loads `Dynamic`, and every access
//! through them then opens every alloca, as above. The lowered pool holds each thread pointer as
//! its `u32` arena offset (`lower::access::cell`); a load of bytes no pointer wrote reads whatever
//! they hold, an arena offset that Metal's pointer of those bytes would not have, both undefined to
//! dereference.
//!
//! **Slot copies**. A `memcpy` or `memmove` of a translation-time constant length between two
//! pointers into forwarding cells at constant offsets moves each source pointer slot that lies
//! wholly inside the copied bytes to the destination slot at the same distance from the copy's
//! start: the destination slot holds the source slot's pointers, and the copy's other bytes are
//! data copied onto data. Such a copy keeps both cells closed when no pointer access of either is
//! at an unknown offset, no slot straddles either end of the copied bytes, and every destination
//! slot inside them is a slot of every source cell at the same distance and of the same type
//! ([`settle_copies`]). The lowered copy moves the arena bytes that hold each slot's lowered
//! representation (`lower::access::copy`), which the destination's loads read as their own:
//! provenance joins the copied pointers into the destination slot's loads (their roots and
//! classes), so the representations agree as a store's and its loads' do. Any other copy is a
//! data access of its bytes, as above. This is MPS's `_MPSNDArrays` shape: a struct holding a
//! buffer-table pointer, copied whole into the struct each kernel call takes.
//!
//! **Pointer bits**. A `ptrtoint` of a pointer to an integer of the pointer's store size, every use
//! of which is the value of a plain store ([`bit_stores`]), moves the pointer's bits into memory
//! and nowhere else. When every one of those stores lands in forwarding cells, each is a store of
//! the pointer: the slot holds the pointer (its lowered representation), and the integer is never
//! observed ([`CellFacts::bits`]). Otherwise the stores are data and the `ptrtoint` reaches the
//! pointer's memory, as above. Likewise a plain integer load of a pointer's store size whose every
//! use is an `inttoptr` to a pointer of that size ([`word_loads`]) reads a pointer's bits and only
//! turns them back into the pointer: through a forwarding cell it is a load of the slot's pointer,
//! each `inttoptr` that pointer ([`CellFacts::words`]), as the round trip is through a handle
//! array. And a plain integer load of that size whose every use is the value of a plain store of it
//! ([`bit_copies`]) copies the bytes it reads to each store's address: between forwarding cells at
//! constant offsets it is a slot copy of 8 bytes, as a `memcpy` of them is.
//!
//! A load from a slot that no store reaches reads an undefined pointer: the null root (a valid
//! refinement), assigned once the facts are settled so a transiently empty slot never adds it.
//!
//! **Zero fills**. A `memset` of zero bytes of a translation-time constant length into forwarding
//! cells at a constant offset ([`zero_fill`]) writes null over every pointer slot it covers: zero
//! bytes read back as null in every lowered representation, so the fill is not a data access of its
//! bytes. The analysis is flow-insensitive: a load from a slot that a zero fill covers reads null
//! as well as every pointer stored there, and a load at an unknown offset reads null when its cell
//! has any zero bytes. A slot copy moves the zero bytes inside the copied bytes to the destination
//! at the same distance from the copy's start. Such a fill keeps its cells closed when it covers
//! each pointer slot wholly or not at all and, where some pointer access of the cell is at an
//! unknown offset, every such access lies at a multiple of the pointer's size (its offset's proven
//! alignment), as each end of the fill does, so the fill covers whole slots wherever the access
//! lands ([`disciplined`]); any other `memset` is a data access, as above. The lowered fill writes the
//! zero bytes as it would into any private memory.

use std::collections::{BTreeMap, BTreeSet};

use crate::air::intrinsics::LlvmIntrinsic;
use crate::air::ir::{
    AddrSpace, Body, Call, Callee, CastOp, Inst, InstId, InstKind, Type, TypeId, Value,
};
use crate::budget::{tree_bytes, Meter};
use crate::error::Error;

use super::{grow, map_entry_bytes, push_working, Cx, Extent, OffsetRange, PtrFact, Root};

/// Where a pointer access into a cell lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Slot {
    /// At this byte offset from the cell's start.
    At(i64),
    /// Anywhere in the cell.
    Any,
}

impl Slot {
    pub(crate) fn of(range: OffsetRange) -> Slot {
        range.exact().map_or(Slot::Any, Slot::At)
    }
}

/// The endpoints of a copy between forwarding cells ([`Cx::copy_endpoints`]).
#[derive(Clone)]
pub(crate) struct CopyEnds {
    /// The source cells and the source pointer's offset in them.
    pub(crate) src: Vec<InstId>,
    pub(crate) s: i64,
    /// The destination cells and the destination pointer's offset in them.
    pub(crate) dst: Vec<InstId>,
    pub(crate) d: i64,
}

impl CopyEnds {
    /// Where a pointer slot at source offset `o` lands in a destination cell when `len` bytes
    /// are copied: `None` unless the slot's eight bytes lie wholly inside the copied bytes.
    pub(crate) fn image(&self, o: i64, len: i64) -> Option<i64> {
        let end = self.s.checked_add(len)?;
        (o >= self.s && o.checked_add(POINTER_BYTES)? <= end)
            .then(|| o.checked_sub(self.s)?.checked_add(self.d))
            .flatten()
    }
}

/// The alloca roots of a forwarding pointer's root set.
fn allocas(roots: &[Root]) -> Vec<InstId> {
    roots
        .iter()
        .filter_map(|r| match r {
            Root::Alloca(a) => Some(*a),
            _ => None,
        })
        .collect()
}

/// Which allocas forward the pointers stored into them (the analysis's verdict).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CellFacts {
    /// No alloca forwards (the verification failed twice).
    pub all_open: bool,
    /// Allocas whose pointer loads are `Dynamic`.
    pub open: BTreeSet<InstId>,
    /// Forwarding allocas whose slots are not disciplined, each holding thread-space pointers only:
    /// a pointer loaded from one is one of the pool's pointers (module doc, **Pools**).
    pub pooled: BTreeSet<InstId>,
    /// The loads and stores of pointers that go through forwarding cells only: each moves the
    /// pointer's lowered representation (`lower::access::cell`).
    pub accesses: BTreeSet<InstId>,
    /// The slot group of each access in `accesses`, named by its first access: the stores and loads
    /// that meet at a cell slot share one group, joined through a pointer into several cells, an
    /// access at an unknown offset (the whole cell), and a slot copy (its source and destination
    /// slots). A group's loads read the bytes its stores write, so their encodings fix the one the
    /// group's stores write (`lower::access::cell`).
    pub groups: BTreeMap<InstId, InstId>,
    /// The `ptrtoint`s whose bits go only into forwarding cells, with their pointers: each use
    /// stores the integer into a forwarding cell, so the store is a store of the pointer (its
    /// lowered representation) and the integer is never observed.
    pub bits: BTreeMap<InstId, Value>,
    /// The integer loads from forwarding cells that only `inttoptr`s read, each with one of its
    /// `inttoptr`s: each is a load of the slot's pointer, held as that pointer's lowered
    /// representation, and each `inttoptr` a copy of it.
    pub words: BTreeMap<InstId, InstId>,
}

impl CellFacts {
    /// Whether a pointer with `roots` addresses only forwarding cells: every root but null is an
    /// alloca that is not open, and there is one.
    pub fn forwards(&self, roots: &[Root]) -> bool {
        if self.all_open {
            return false;
        }
        let mut any = false;
        for r in roots {
            match r {
                Root::Null => {}
                Root::Alloca(a) if !self.open.contains(a) => any = true,
                _ => return false,
            }
        }
        any
    }

    /// The slot group of cell access `access` ([`CellFacts::groups`]).
    pub fn group(&self, access: InstId) -> Option<InstId> {
        self.groups.get(&access).copied()
    }

    /// The bytes the verdict holds, as the ledger charges them.
    pub fn retained_bytes(&self) -> usize {
        (self.open.len() + self.pooled.len() + self.accesses.len())
            * map_entry_bytes::<InstId, ()>()
            + self.groups.len() * map_entry_bytes::<InstId, InstId>()
            + self.bits.len() * map_entry_bytes::<InstId, Value>()
            + self.words.len() * map_entry_bytes::<InstId, InstId>()
    }

    /// The pointer whose bits `v` is when `v` is a `ptrtoint` in [`CellFacts::bits`].
    pub fn bits_of(&self, v: Value) -> Option<Value> {
        match v {
            Value::Inst(c) => self.bits.get(&c).copied(),
            _ => None,
        }
    }
}

/// The union-find that forms the slot groups of the cell accesses ([`CellFacts::groups`]): each
/// root is its group's smallest access, so a group's name does not depend on the union order.
pub(crate) struct SlotGroups {
    /// The accesses, sorted.
    ids: Vec<InstId>,
    parent: Vec<u32>,
    /// Ledger bytes of `ids` and `parent`.
    bytes: usize,
}

impl SlotGroups {
    pub(crate) fn new(accesses: &BTreeSet<InstId>, m: &mut Meter) -> Result<SlotGroups, Error> {
        let n = accesses.len();
        let bytes = n.saturating_mul(std::mem::size_of::<InstId>() + std::mem::size_of::<u32>());
        m.charge(bytes)?;
        let parent = (0..n)
            .map(|i| u32::try_from(i).unwrap_or(u32::MAX))
            .collect();
        Ok(SlotGroups {
            ids: accesses.iter().copied().collect(),
            parent,
            bytes,
        })
    }

    fn find(&mut self, mut i: usize) -> usize {
        while let Some(&p) = self.parent.get(i) {
            let p = p as usize;
            if p == i {
                break;
            }
            let grand = self.parent.get(p).copied().unwrap_or(p as u32);
            if let Some(slot) = self.parent.get_mut(i) {
                *slot = grand;
            }
            i = p;
        }
        i
    }

    /// Put accesses `a` and `b` in one group (an id that is no access is ignored).
    pub(crate) fn union(&mut self, a: InstId, b: InstId) {
        let (Ok(i), Ok(j)) = (self.ids.binary_search(&a), self.ids.binary_search(&b)) else {
            return;
        };
        let (ri, rj) = (self.find(i), self.find(j));
        let (lo, hi) = (ri.min(rj), ri.max(rj));
        if let (Some(slot), Ok(lo)) = (self.parent.get_mut(hi), u32::try_from(lo)) {
            *slot = lo;
        }
    }

    /// Each access's group, named by its smallest access.
    pub(crate) fn finish(mut self, m: &mut Meter) -> Result<BTreeMap<InstId, InstId>, Error> {
        m.charge(
            self.ids
                .len()
                .saturating_mul(map_entry_bytes::<InstId, InstId>()),
        )?;
        let mut out = BTreeMap::new();
        for i in 0..self.ids.len() {
            m.tick(1)?;
            let root = self.find(i);
            if let (Some(&a), Some(&r)) = (self.ids.get(i), self.ids.get(root)) {
                out.insert(a, r);
            }
        }
        m.release(self.bytes);
        Ok(out)
    }
}

/// The ledger bytes of one slot's contents, of one reader, and of one cell's zero bytes.
const CONTENT_BYTES: usize = map_entry_bytes::<(InstId, Slot), PtrFact>();
const READER_BYTES: usize = map_entry_bytes::<(InstId, u32), ()>();
const ZERO_BYTES: usize = map_entry_bytes::<(InstId, i64, i64), ()>();

/// Forwarding's working state in one round of the analysis.
#[derive(Default)]
pub(crate) struct Cells {
    pub(crate) facts: CellFacts,
    /// Every store of a pointer value, `(value, address)`, found once.
    pub(crate) stores: Vec<(Value, Value)>,
    /// The pointer index and address of every load of a pointer, found once.
    pub(crate) loads: Vec<(u32, Value)>,
    /// Every `memcpy` or `memmove` of a translation-time constant length between two
    /// thread-space pointers, `(destination, source, length)`, found once.
    pub(crate) copies: Vec<(Value, Value, u64)>,
    /// Every store of a pointer's bits, `(store, ptrtoint, pointer, address)`, found once
    /// ([`bit_stores`]); its `(pointer, address)` is in `stores` too.
    pub(crate) bit_stores: Vec<(InstId, InstId, Value, Value)>,
    /// Every pointer-word load, by load: its address and the `inttoptr`s that read it, found once
    /// ([`word_loads`]); each `inttoptr`'s pointer index and the address are in `loads` too.
    pub(crate) word_loads: BTreeMap<InstId, (Value, Vec<InstId>)>,
    /// Every copy of pointer bits, `(load, store)` of each plain integer load of a pointer's store
    /// size whose every use is the value of a plain store, found once ([`bit_copies`]); each
    /// `(destination, source, 8)` is in `copies` too.
    pub(crate) bit_copies: Vec<(InstId, InstId)>,
    /// Every `memset` of zero bytes of a translation-time constant length into thread memory,
    /// `(address, length)`, found once ([`zero_fill`]).
    pub(crate) zero_fills: Vec<(Value, u64)>,
    /// Every thread-space pointer an instruction turns into data or hands to code no cell sees
    /// (`ptrtoint`, an aggregate or vector built of it, a call other than an intrinsic's, an
    /// atomic's value, a return), found once ([`escapes`]): data may carry it into a pool.
    pub(crate) escapes: Vec<Value>,
    /// The join of every pointer whose bytes data may carry into a pool (module doc, **Pools**);
    /// `None` while there is none.
    leaked: Option<PtrFact>,
    /// The join of the pointers stored at each slot of each cell.
    contents: BTreeMap<(InstId, Slot), PtrFact>,
    /// `(cell, lo, hi)`: the bytes `[lo, hi)` of a cell that a zero fill writes, at its own
    /// address or through slot copies.
    zeros: BTreeSet<(InstId, i64, i64)>,
    /// `(cell, load)`: the loads that read each cell, revisited when its contents grow.
    readers: BTreeSet<(InstId, u32)>,
    /// The worklist drained and a store pass found nothing new: a slot that holds nothing now is
    /// never written.
    pub(crate) settled: bool,
    /// Ledger bytes of `contents`, `readers`, and `zeros`.
    bytes: usize,
}

impl Cells {
    /// Forget one round's contents and readers.
    pub(crate) fn reset(&mut self, m: &mut Meter) {
        m.release(self.bytes);
        self.bytes = 0;
        self.contents.clear();
        self.readers.clear();
        self.zeros.clear();
        self.leaked = None;
        self.settled = false;
    }

    /// Add the zero bytes `[lo, hi)` to cell `a`; whether they are new.
    fn add_zeros(&mut self, a: InstId, lo: i64, hi: i64, m: &mut Meter) -> Result<bool, Error> {
        if lo >= hi || self.zeros.contains(&(a, lo, hi)) {
            return Ok(false);
        }
        self.charge(ZERO_BYTES, m)?;
        Ok(self.zeros.insert((a, lo, hi)))
    }

    /// Whether a pointer load at `slot` of cell `a` may read its zero bytes: some meet the slot's
    /// bytes, or the slot is unknown and the cell has some.
    fn zeroed(&self, a: InstId, slot: Slot, m: &mut Meter) -> Result<bool, Error> {
        for &(_, lo, hi) in self
            .zeros
            .range((a, i64::MIN, i64::MIN)..=(a, i64::MAX, i64::MAX))
        {
            m.tick(1)?;
            let meets = match slot {
                Slot::Any => true,
                Slot::At(o) => lo < o.saturating_add(POINTER_BYTES) && o < hi,
            };
            if meets {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn charge(&mut self, n: usize, m: &mut Meter) -> Result<(), Error> {
        m.charge(n)?;
        self.bytes = self.bytes.saturating_add(n);
        Ok(())
    }
}

impl Cx<'_> {
    /// The fact of the pointer that load `id` (pointer index `me`) reads through a pointer with
    /// fact `f`, when `f` addresses only forwarding cells: the join of the pointers stored at the
    /// slots it reads. `None` for any other load.
    pub(super) fn cell_load(
        &mut self,
        me: u32,
        f: PtrFact,
        m: &mut Meter,
    ) -> Result<Option<PtrFact>, Error> {
        let roots = self.sets.get(f.roots);
        if !self.cells.facts.forwards(roots) {
            return Ok(None);
        }
        let cells: Vec<InstId> = roots
            .iter()
            .filter_map(|r| match r {
                Root::Alloca(a) => Some(*a),
                _ => None,
            })
            .collect();
        let slot = Slot::of(f.range);
        let mut out = PtrFact::BOTTOM;
        let mut zeroed = false;
        for a in cells {
            if self.cells.readers.insert((a, me)) {
                self.cells.charge(READER_BYTES, m)?;
            }
            // A pool's load reads any of its pointers, or bytes no pointer wrote (module doc).
            let pooled = self.cells.facts.pooled.contains(&a);
            zeroed = zeroed || pooled || self.cells.zeroed(a, slot, m)?;
            let all = (a, Slot::At(i64::MIN))..=(a, Slot::Any);
            for (&(_, s), x) in self.cells.contents.range(all) {
                m.tick(1)?;
                if pooled || slot == Slot::Any || s == Slot::Any || s == slot {
                    out = PtrFact {
                        roots: self.sets.join(out.roots, x.roots, m)?,
                        range: out.range.join(x.range),
                        shape: out.shape.join(x.shape),
                        align: out.align.min(x.align),
                    };
                }
            }
        }
        // Zero bytes over the slot read back as null (module doc).
        if zeroed {
            let s = self.sets.single(Root::Null, m)?;
            out = self.join(out, PtrFact::at(s, 0), m)?;
        }
        if out == PtrFact::BOTTOM && self.cells.settled {
            let s = self.sets.single(Root::Null, m)?;
            out = PtrFact::at(s, 0);
        }
        Ok(Some(out))
    }

    /// Join every stored pointer into the slots its store writes, and add each zero fill's bytes
    /// to the cells it writes; the loads of every cell whose contents grew, to revisit.
    pub(super) fn store_pass(&mut self, m: &mut Meter) -> Result<Vec<u32>, Error> {
        let mut grew: Vec<InstId> = Vec::new();
        for i in 0..self.cells.stores.len() {
            m.tick(1)?;
            let Some(&(val, ptr)) = self.cells.stores.get(i) else {
                break;
            };
            let pf = self.get(ptr, m)?;
            let vf = self.get(val, m)?;
            if pf == PtrFact::BOTTOM || vf == PtrFact::BOTTOM {
                continue;
            }
            let roots = self.sets.get(pf.roots);
            if !self.cells.facts.forwards(roots) {
                // A thread-space pointer stored where no cell forwards it: data read back from
                // there may carry it into a pool.
                if !self.cells.facts.pooled.is_empty()
                    && self.space_of_value(val) == Some(AddrSpace::Thread)
                {
                    self.leak(vf, m)?;
                }
                continue;
            }
            let cells: Vec<InstId> = roots
                .iter()
                .filter_map(|r| match r {
                    Root::Alloca(a) => Some(*a),
                    _ => None,
                })
                .collect();
            let slot = Slot::of(pf.range);
            for a in cells {
                // A pool keeps its pointers whatever their slots.
                let slot = if self.cells.facts.pooled.contains(&a) {
                    Slot::Any
                } else {
                    slot
                };
                let old = self.cells.contents.get(&(a, slot)).copied();
                let new = self.join(old.unwrap_or(PtrFact::BOTTOM), vf, m)?;
                if Some(new) != old {
                    if old.is_none() {
                        self.cells.charge(CONTENT_BYTES, m)?;
                    }
                    self.cells.contents.insert((a, slot), new);
                    grew.push(a);
                }
            }
        }
        for i in 0..self.cells.zero_fills.len() {
            m.tick(1)?;
            let Some(&(ptr, len)) = self.cells.zero_fills.get(i) else {
                break;
            };
            let pf = self.get(ptr, m)?;
            let Some((cells, lo, hi)) = self.zero_bytes(pf, len) else {
                continue;
            };
            for a in cells {
                if self.cells.add_zeros(a, lo, hi, m)? {
                    grew.push(a);
                }
            }
        }
        self.copy_pass(&mut grew, m)?;
        self.pool_pass(&mut grew, m)?;
        grew.sort_unstable();
        grew.dedup();
        let mut out = Vec::new();
        for a in grew {
            for &(_, l) in self.cells.readers.range((a, 0)..=(a, u32::MAX)) {
                m.tick(1)?;
                out.push(l);
            }
        }
        Ok(out)
    }

    /// The cells a zero fill of `len` bytes through a pointer with fact `pf` writes, and its bytes
    /// in them, when the pointer addresses only forwarding cells at a translation-time constant
    /// offset; `None` otherwise (the fill is then a data access of its bytes, [`verify`]).
    fn zero_bytes(&self, pf: PtrFact, len: u64) -> Option<(Vec<InstId>, i64, i64)> {
        let Slot::At(lo) = Slot::of(pf.range) else {
            return None;
        };
        let hi = lo.checked_add(i64::try_from(len).ok()?)?;
        let roots = self.sets.get(pf.roots);
        (pf != PtrFact::BOTTOM && self.cells.facts.forwards(roots))
            .then(|| (allocas(roots), lo, hi))
    }

    /// The cells and exact offsets of a slot copy's endpoints: `(source cells, source offset,
    /// destination cells, destination offset)` when both pointers address only forwarding cells
    /// at translation-time constant offsets; `None` otherwise (the copy is then a data access of
    /// its bytes, [`verify`]).
    pub(super) fn copy_endpoints(
        &mut self,
        dst: Value,
        src: Value,
        m: &mut Meter,
    ) -> Result<Option<CopyEnds>, Error> {
        let (df, sf) = (self.get(dst, m)?, self.get(src, m)?);
        if df == PtrFact::BOTTOM || sf == PtrFact::BOTTOM {
            return Ok(None);
        }
        let (Slot::At(d), Slot::At(s)) = (Slot::of(df.range), Slot::of(sf.range)) else {
            return Ok(None);
        };
        let (droots, sroots) = (self.sets.get(df.roots), self.sets.get(sf.roots));
        if !self.cells.facts.forwards(droots) || !self.cells.facts.forwards(sroots) {
            return Ok(None);
        }
        Ok(Some(CopyEnds {
            src: allocas(sroots),
            s,
            dst: allocas(droots),
            d,
        }))
    }

    /// Carry the pointers a slot copy moves: the contents of each source slot that lies wholly
    /// inside the copied bytes join the destination slot at the same distance from the copy's
    /// start, and a source cell's contents at an unknown offset join the destination's; the
    /// source's zero bytes inside the copied bytes are the destination's at the same distance.
    /// Repeated until no content grows, so a chain of copies carries its pointers to its end;
    /// the cells whose contents grew are added to `grew`.
    fn copy_pass(&mut self, grew: &mut Vec<InstId>, m: &mut Meter) -> Result<(), Error> {
        loop {
            let mut changed = false;
            for i in 0..self.cells.copies.len() {
                m.tick(1)?;
                let Some(&(dst, src, len)) = self.cells.copies.get(i) else {
                    break;
                };
                let Some(ends) = self.copy_endpoints(dst, src, m)? else {
                    continue;
                };
                let Ok(len) = i64::try_from(len) else {
                    continue;
                };
                let mut moved: Vec<(Slot, PtrFact)> = Vec::new();
                let mut zeros: Vec<(i64, i64)> = Vec::new();
                let mut moved_bytes = 0usize;
                let result = self.move_contents(&ends, len, &mut moved, &mut moved_bytes, m);
                let result = result
                    .and_then(|()| self.move_zeros(&ends, len, &mut zeros, &mut moved_bytes, m));
                let result = result.and_then(|()| {
                    for &b in &ends.dst {
                        for &(slot, x) in &moved {
                            let old = self.cells.contents.get(&(b, slot)).copied();
                            let new = self.join(old.unwrap_or(PtrFact::BOTTOM), x, m)?;
                            if Some(new) != old {
                                if old.is_none() {
                                    self.cells.charge(CONTENT_BYTES, m)?;
                                }
                                self.cells.contents.insert((b, slot), new);
                                grow(grew, 1, m)?;
                                grew.push(b);
                                changed = true;
                            }
                        }
                        for &(lo, hi) in &zeros {
                            m.tick(1)?;
                            if self.cells.add_zeros(b, lo, hi, m)? {
                                grow(grew, 1, m)?;
                                grew.push(b);
                                changed = true;
                            }
                        }
                    }
                    Ok(())
                });
                m.release(moved_bytes);
                result?;
            }
            if !changed {
                return Ok(());
            }
        }
    }

    /// Join `f` into the pointers whose bytes data may carry into a pool.
    fn leak(&mut self, f: PtrFact, m: &mut Meter) -> Result<(), Error> {
        if f == PtrFact::BOTTOM {
            return Ok(());
        }
        self.cells.leaked = Some(match self.cells.leaked {
            Some(l) => self.join(l, f, m)?,
            None => f,
        });
        Ok(())
    }

    /// Carry the pointers data may move between pools (module doc, **Pools**): every pool's
    /// pointers, every escaping pointer, and every thread-space pointer stored where no cell
    /// forwards it (joined by the store pass) leak; every pool holds what leaks. The pools whose
    /// contents grew are added to `grew`. One work unit per pool, per pool's content and per
    /// escaping pointer.
    fn pool_pass(&mut self, grew: &mut Vec<InstId>, m: &mut Meter) -> Result<(), Error> {
        if self.cells.facts.pooled.is_empty() {
            return Ok(());
        }
        for i in 0..self.cells.escapes.len() {
            m.tick(1)?;
            let Some(&v) = self.cells.escapes.get(i) else {
                break;
            };
            let f = self.get(v, m)?;
            self.leak(f, m)?;
        }
        let mut next = self.next_pool(None);
        while let Some(a) = next {
            m.tick(1)?;
            let all = (a, Slot::At(i64::MIN))..=(a, Slot::Any);
            let mut held = PtrFact::BOTTOM;
            for (_, x) in self.cells.contents.range(all) {
                m.tick(1)?;
                held = PtrFact {
                    roots: self.sets.join(held.roots, x.roots, m)?,
                    range: held.range.join(x.range),
                    shape: held.shape.join(x.shape),
                    align: held.align.min(x.align),
                };
            }
            self.leak(held, m)?;
            next = self.next_pool(Some(a));
        }
        let Some(leaked) = self.cells.leaked else {
            return Ok(());
        };
        let mut next = self.next_pool(None);
        while let Some(a) = next {
            m.tick(1)?;
            let old = self.cells.contents.get(&(a, Slot::Any)).copied();
            let new = self.join(old.unwrap_or(PtrFact::BOTTOM), leaked, m)?;
            if Some(new) != old {
                if old.is_none() {
                    self.cells.charge(CONTENT_BYTES, m)?;
                }
                self.cells.contents.insert((a, Slot::Any), new);
                grow(grew, 1, m)?;
                grew.push(a);
            }
            next = self.next_pool(Some(a));
        }
        Ok(())
    }

    /// The first pool after `after` (the first of all for `None`) that is not open.
    fn next_pool(&self, after: Option<InstId>) -> Option<InstId> {
        use std::ops::Bound::{Excluded, Unbounded};
        let from = after.map_or(Unbounded, Excluded);
        self.cells
            .facts
            .pooled
            .range((from, Unbounded))
            .find(|a| !self.cells.facts.open.contains(a))
            .copied()
    }

    /// The contents a slot copy moves, each at its destination slot, into `moved` (a working
    /// vector whose charges `bytes` counts).
    fn move_contents(
        &self,
        ends: &CopyEnds,
        len: i64,
        moved: &mut Vec<(Slot, PtrFact)>,
        bytes: &mut usize,
        m: &mut Meter,
    ) -> Result<(), Error> {
        for &a in &ends.src {
            let all = (a, Slot::At(i64::MIN))..=(a, Slot::Any);
            for (&(_, slot), x) in self.cells.contents.range(all) {
                m.tick(1)?;
                let to = match slot {
                    Slot::Any => Slot::Any,
                    Slot::At(o) => match ends.image(o, len) {
                        Some(t) => Slot::At(t),
                        None => continue,
                    },
                };
                push_working(moved, (to, *x), bytes, m)?;
            }
        }
        Ok(())
    }

    /// The zero bytes a slot copy moves, each at its destination offsets, into `zeros` (a working
    /// vector whose charges `bytes` counts): each source zero range clipped to the copied bytes.
    fn move_zeros(
        &self,
        ends: &CopyEnds,
        len: i64,
        zeros: &mut Vec<(i64, i64)>,
        bytes: &mut usize,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let (Some(end), Some(shift)) = (ends.s.checked_add(len), ends.d.checked_sub(ends.s)) else {
            return Ok(());
        };
        for &a in &ends.src {
            let all = (a, i64::MIN, i64::MIN)..=(a, i64::MAX, i64::MAX);
            for &(_, lo, hi) in self.cells.zeros.range(all) {
                m.tick(1)?;
                let (lo, hi) = (lo.max(ends.s), hi.min(end));
                if let (true, Some(lo), Some(hi)) =
                    (lo < hi, lo.checked_add(shift), hi.checked_add(shift))
                {
                    push_working(zeros, (lo, hi), bytes, m)?;
                }
            }
        }
        Ok(())
    }

    /// The loads of pointers through forwarding cells whose fact is still empty, once the facts
    /// are settled: they read slots no store writes.
    pub(super) fn unwritten_loads(&mut self, m: &mut Meter) -> Result<Vec<u32>, Error> {
        let mut out = Vec::new();
        for i in 0..self.cells.loads.len() {
            m.tick(1)?;
            let Some(&(me, ptr)) = self.cells.loads.get(i) else {
                break;
            };
            if self.facts.get(me as usize) != Some(&PtrFact::BOTTOM) {
                continue;
            }
            let pf = self.get(ptr, m)?;
            if pf != PtrFact::BOTTOM && self.cells.facts.forwards(self.sets.get(pf.roots)) {
                out.push(me);
            }
        }
        Ok(out)
    }
}

/// What the verification learns about one alloca.
#[derive(Default)]
struct Use {
    /// Exact pointer slots, with the pointer type accessed there.
    slots: BTreeMap<i64, TypeId>,
    /// The pointer type of a pointer access at an unknown offset.
    wild: Option<TypeId>,
    /// Some pointer access at an unknown offset is not proven to lie at a multiple of the
    /// pointer's size.
    wild_unaligned: bool,
    /// The bytes of each data access, `[lo, hi)`, ordered, each once.
    data: BTreeSet<(i64, i64)>,
    /// Some data access has bytes that are not translation-time constants.
    data_unknown: bool,
    /// The bytes of each zero fill into the cell at a constant offset, `[lo, hi)`, ordered, each
    /// once: null over the pointer slots they cover (module doc).
    zero: BTreeSet<(i64, i64)>,
    /// Reached otherwise than through its disciplined accesses.
    open: bool,
}

/// The allocas that pointers are stored into or loaded from and that the final facts of this
/// round show are not closed (module doc), in order: every such alloca when some access goes
/// through a thread-space pointer that may be `Dynamic`; and the closed ones whose slots are not
/// disciplined that become pools. An alloca no pointer access reaches needs no verdict:
/// forwarding only changes pointer loads.
pub(super) fn verify(cx: &mut Cx<'_>, body: &Body, m: &mut Meter) -> Result<Verdict, Error> {
    m.release(
        cx.cells.facts.bits.len() * map_entry_bytes::<InstId, Value>()
            + cx.cells.facts.words.len() * map_entry_bytes::<InstId, InstId>(),
    );
    cx.cells.facts.bits.clear();
    cx.cells.facts.words.clear();
    if cx.cells.facts.all_open {
        return Ok(Verdict::default());
    }
    let bits = pointer_bits(cx, m)?;
    let words = pointer_words(cx, m)?;
    let copied = bit_copy_ends(cx, body, m)?;
    let held = held_insertions(cx, body, m)?;
    let result = verify_round(cx, body, &bits, &words, &copied, &held, m);
    m.release(held.len() * map_entry_bytes::<InstId, ()>());
    drop(held);
    let copied_bytes = copied.0.len() * map_entry_bytes::<InstId, Vec<CopyEnds>>()
        + copied.1.len() * map_entry_bytes::<InstId, InstId>();
    drop(copied);
    m.release(copied_bytes);
    let (by_store, by_cast) = bits;
    m.release(
        by_store.len() * map_entry_bytes::<InstId, InstId>()
            + by_cast.len() * map_entry_bytes::<InstId, (Value, bool)>(),
    );
    let failed = result?;
    for (c, (p, all)) in by_cast {
        if all {
            m.charge(map_entry_bytes::<InstId, Value>())?;
            cx.cells.facts.bits.insert(c, p);
        }
    }
    for (l, c) in words {
        cx.cells.facts.words.insert(l, c);
    }
    Ok(failed)
}

/// The copies of pointer bits that are slot copies under this round's facts: by load, the ends
/// of each of its stores' copies when every one of them is a slot copy; and each store's load.
type BitCopies = (BTreeMap<InstId, Vec<CopyEnds>>, BTreeMap<InstId, InstId>);

fn bit_copy_ends(cx: &mut Cx<'_>, body: &Body, m: &mut Meter) -> Result<BitCopies, Error> {
    let mut ends: BTreeMap<InstId, Option<Vec<CopyEnds>>> = BTreeMap::new();
    let mut stores = BTreeMap::new();
    let mut charged = 0usize;
    for i in 0..cx.cells.bit_copies.len() {
        m.tick(1)?;
        let Some(&(load, store)) = cx.cells.bit_copies.get(i) else {
            break;
        };
        let (dst, src) = match (body.insts.get(store), body.insts.get(load)) {
            (
                Some(Inst {
                    kind: InstKind::Store { ptr: d, .. },
                    ..
                }),
                Some(Inst {
                    kind: InstKind::Load { ptr: s, .. },
                    ..
                }),
            ) => (*d, *s),
            _ => return Err(super::invariant("a bit copy that is no load and store")),
        };
        m.charge(map_entry_bytes::<InstId, InstId>())?;
        stores.insert(store, load);
        let end = cx.copy_endpoints(dst, src, m)?;
        if !ends.contains_key(&load) {
            m.charge(map_entry_bytes::<InstId, Vec<CopyEnds>>())?;
            charged += map_entry_bytes::<InstId, Vec<CopyEnds>>();
        }
        let e = ends.entry(load).or_insert_with(|| Some(Vec::new()));
        match (e.as_mut(), end) {
            (Some(v), Some(c)) => {
                grow(v, 1, m)?;
                v.push(c);
            }
            _ => *e = None,
        }
    }
    let whole: BTreeMap<InstId, Vec<CopyEnds>> = ends
        .into_iter()
        .filter_map(|(l, e)| Some((l, e?)))
        .collect();
    m.release(charged.saturating_sub(whole.len() * map_entry_bytes::<InstId, Vec<CopyEnds>>()));
    Ok((whole, stores))
}

/// The pointer-word loads whose address forwards under this round's facts, each with its first
/// `inttoptr` (charged as [`CellFacts::words`] holds them).
fn pointer_words(cx: &mut Cx<'_>, m: &mut Meter) -> Result<BTreeMap<InstId, InstId>, Error> {
    use std::ops::Bound::{Excluded, Unbounded};
    let mut out = BTreeMap::new();
    // Walk the loads by key, reading each entry in place: `cx.get` needs the context mutably.
    let mut next = cx.cells.word_loads.keys().next().copied();
    while let Some(l) = next {
        m.tick(1)?;
        next = cx
            .cells
            .word_loads
            .range((Excluded(l), Unbounded))
            .next()
            .map(|(k, _)| *k);
        let Some((addr, Some(c))) = cx
            .cells
            .word_loads
            .get(&l)
            .map(|(a, cs)| (*a, cs.first().copied()))
        else {
            continue;
        };
        let f = cx.get(addr, m)?;
        if f != PtrFact::BOTTOM && cx.cells.facts.forwards(cx.sets.get(f.roots)) {
            m.charge(map_entry_bytes::<InstId, InstId>())?;
            out.insert(l, c);
        }
    }
    Ok(out)
}

/// The stores of pointer bits by store (each naming its `ptrtoint`), and each such `ptrtoint`
/// with its pointer and whether every store of it lands in forwarding cells, under this round's
/// facts.
type Bits = (BTreeMap<InstId, InstId>, BTreeMap<InstId, (Value, bool)>);

fn pointer_bits(cx: &mut Cx<'_>, m: &mut Meter) -> Result<Bits, Error> {
    let mut by_store = BTreeMap::new();
    let mut by_cast: BTreeMap<InstId, (Value, bool)> = BTreeMap::new();
    for i in 0..cx.cells.bit_stores.len() {
        m.tick(1)?;
        let Some(&(store, c, p, addr)) = cx.cells.bit_stores.get(i) else {
            break;
        };
        let f = cx.get(addr, m)?;
        let forwards = f != PtrFact::BOTTOM && cx.cells.facts.forwards(cx.sets.get(f.roots));
        m.charge(map_entry_bytes::<InstId, InstId>())?;
        by_store.insert(store, c);
        match by_cast.get_mut(&c) {
            Some(e) => e.1 &= forwards,
            None => {
                m.charge(map_entry_bytes::<InstId, (Value, bool)>())?;
                by_cast.insert(c, (p, forwards));
            }
        }
    }
    Ok((by_store, by_cast))
}

/// The `insertvalue`s whose aggregate stays a value (module doc): every use of it, and of each
/// aggregate built on it, is an `extractvalue`, the aggregate operand of an `insertvalue`, or an
/// incoming value of a `phi` of such aggregates. One work unit per instruction and per aggregate
/// that leaves; the set is charged to the ledger, which [`verify`] returns.
fn held_insertions(cx: &Cx<'_>, body: &Body, m: &mut Meter) -> Result<BTreeSet<InstId>, Error> {
    let types = cx.e.module.types();
    // An `insertvalue` or a `phi` of an aggregate that holds a pointer.
    let builds = |v: Value| match v {
        Value::Inst(c) => body.insts.get(c).is_some_and(|i| {
            matches!(i.kind, InstKind::InsertValue { .. } | InstKind::Phi { .. })
                && types.get(i.ty).is_some_and(Type::is_aggregate)
                && types.holds_pointer(i.ty)
        }),
        _ => false,
    };
    // The aggregates some use takes out of SSA, and those still to follow to what they are built on.
    let mut left: BTreeSet<InstId> = BTreeSet::new();
    let mut work: Vec<InstId> = Vec::new();
    let mut grown = 0usize;
    for (_, inst) in body.insts.iter() {
        m.tick(1)?;
        let keeps = |v: Value| match &inst.kind {
            InstKind::ExtractValue { agg, .. } | InstKind::InsertValue { agg, .. } => *agg == v,
            // A `phi` of an aggregate that holds a pointer is itself such an aggregate.
            InstKind::Phi { .. } => true,
            _ => false,
        };
        let mut found = Ok(());
        inst.kind.for_each_value(|v| {
            if found.is_err() || !builds(v) || keeps(v) {
                return;
            }
            if let Value::Inst(c) = v {
                found = leave(c, (&mut left, &mut work, &mut grown), m);
            }
        });
        found?;
    }
    // An aggregate built on one that leaves leaves with it.
    while let Some(c) = work.pop() {
        m.tick(1)?;
        let (agg, incoming) = match body.insts.get(c).map(|i| &i.kind) {
            Some(InstKind::InsertValue { agg, .. }) => (Some(*agg), &[][..]),
            Some(InstKind::Phi { incoming, .. }) => (None, &incoming[..]),
            _ => (None, &[][..]),
        };
        for v in agg.into_iter().chain(incoming.iter().map(|(v, _)| *v)) {
            if let (Value::Inst(a), true) = (v, builds(v)) {
                leave(a, (&mut left, &mut work, &mut grown), m)?;
            }
        }
    }
    m.release(grown);
    let mut held = BTreeSet::new();
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        if matches!(inst.kind, InstKind::InsertValue { .. })
            && builds(Value::Inst(id))
            && !left.contains(&id)
        {
            m.charge(map_entry_bytes::<InstId, ()>())?;
            held.insert(id);
        }
    }
    m.release(left.len() * map_entry_bytes::<InstId, ()>());
    Ok(held)
}

/// Record that aggregate `c` leaves SSA ([`held_insertions`]): new, it is charged to the ledger and
/// queued to follow.
fn leave(
    c: InstId,
    (left, work, grown): (&mut BTreeSet<InstId>, &mut Vec<InstId>, &mut usize),
    m: &mut Meter,
) -> Result<(), Error> {
    if left.insert(c) {
        m.charge(map_entry_bytes::<InstId, ()>())?;
        push_working(work, c, grown, m)?;
    }
    Ok(())
}

/// [`verify`]'s walk of one round, with the round's pointer bits. Its working tables are
/// returned to the ledger when it returns ([`Round::charged`]).
fn verify_round(
    cx: &mut Cx<'_>,
    body: &Body,
    bits: &Bits,
    words: &BTreeMap<InstId, InstId>,
    ends: &BitCopies,
    held: &BTreeSet<InstId>,
    m: &mut Meter,
) -> Result<Verdict, Error> {
    let mut round = Round::default();
    let r = round.walk(cx, body, (bits, words, ends, held), m);
    m.release(round.charged());
    r
}

/// The working tables of one [`verify_round`]: the cells' uses, the slot copies, and an operand
/// buffer, charged to the ledger as they grow.
#[derive(Default)]
struct Round {
    uses: BTreeMap<InstId, Use>,
    copies: Vec<(CopyEnds, i64)>,
    operands: Vec<Value>,
    /// The bytes `copies` and `operands` charged as they grew.
    grown: usize,
}

/// What a round reads besides the body: the pointer bits, the pointer words, the bit copies, and
/// the insertions into aggregates that stay values.
type RoundInputs<'r> = (
    &'r Bits,
    &'r BTreeMap<InstId, InstId>,
    &'r BitCopies,
    &'r BTreeSet<InstId>,
);

impl Round {
    /// The ledger bytes the tables hold: every cell's entry, exact slot, data range, and zero
    /// range as its insertion charged it (each is charged once, when it is new), and the vectors' growth.
    fn charged(&self) -> usize {
        let range = tree_bytes(1, std::mem::size_of::<(i64, i64)>());
        let contents: usize = self
            .uses
            .values()
            .map(|u| {
                u.slots
                    .len()
                    .saturating_mul(map_entry_bytes::<i64, TypeId>())
                    .saturating_add(u.data.len().saturating_mul(range))
                    .saturating_add(u.zero.len().saturating_mul(range))
            })
            .fold(0, usize::saturating_add);
        self.uses
            .len()
            .saturating_mul(map_entry_bytes::<InstId, Use>())
            .saturating_add(contents)
            .saturating_add(self.grown)
    }

    fn walk(
        &mut self,
        cx: &mut Cx<'_>,
        body: &Body,
        ((by_store, by_cast), words, (copy_ends, copy_stores), held): RoundInputs<'_>,
        m: &mut Meter,
    ) -> Result<Verdict, Error> {
        // The pointer a store of pointer bits stores, when it is a store of that pointer.
        let bits_stored = |store: InstId| {
            let c = by_store.get(&store)?;
            by_cast.get(c).filter(|e| e.1).map(|e| e.0)
        };
        let uses = &mut self.uses;
        let mut aliased = false;
        for (id, inst) in body.insts.iter() {
            m.tick(1)?;
            // A `memset`, `memcpy`, or `memmove` touches its pointers' bytes as data: a copy that
            // reaches a pointer slot moves or clobbers the pointer, so it opens the cell through the
            // slot discipline, and one that stays clear of the slots keeps it. A copy between
            // forwarding cells at constant offsets is instead a slot copy, settled below.
            if let InstKind::Call(call) = &inst.kind {
                if let Some((pointers, len)) = memory_intrinsic(cx, call) {
                    // A zero fill into forwarding cells at a constant offset writes null over the
                    // slots it covers, not data (module doc).
                    if note_zeros(cx, uses, call, m)? {
                        continue;
                    }
                    if let Some(copy) = slot_copy(cx, call, pointers, len, m)? {
                        for c in copy.0.src.iter().chain(&copy.0.dst) {
                            if !uses.contains_key(c) {
                                m.charge(map_entry_bytes::<InstId, Use>())?;
                                uses.insert(*c, Use::default());
                            }
                        }
                        push_working(&mut self.copies, copy, &mut self.grown, m)?;
                        continue;
                    }
                    for (i, &v) in call.args.iter().enumerate() {
                        if i < pointers {
                            note_data(cx, uses, &mut aliased, v, len, m)?;
                        } else {
                            open_roots(cx, uses, v, m)?;
                        }
                    }
                    continue;
                }
            }
            // A copy of pointer bits between forwarding cells is a slot copy of its 8 bytes,
            // settled with the others; its load and stores access no slot themselves.
            if let Some(ends) = copy_ends.get(&id) {
                for e in ends {
                    for c in e.src.iter().chain(&e.dst) {
                        if !uses.contains_key(c) {
                            m.charge(map_entry_bytes::<InstId, Use>())?;
                            uses.insert(*c, Use::default());
                        }
                    }
                    push_working(
                        &mut self.copies,
                        (e.clone(), POINTER_BYTES),
                        &mut self.grown,
                        m,
                    )?;
                }
                continue;
            }
            if copy_stores
                .get(&id)
                .is_some_and(|l| copy_ends.contains_key(l))
            {
                continue;
            }
            match &inst.kind {
                // A pointer-word load through forwarding cells is a load of the pointer its
                // `inttoptr`s are: an access of each of their types.
                InstKind::Load { ptr, .. } if words.contains_key(&id) => {
                    // Each `inttoptr` by its index, read in place (`note_access` needs the context).
                    let cast = |cx: &Cx<'_>, k: usize| {
                        cx.cells
                            .word_loads
                            .get(&id)
                            .and_then(|(_, cs)| cs.get(k))
                            .copied()
                    };
                    let mut k = 0;
                    while let Some(c) = cast(cx, k) {
                        k += 1;
                        let ty = body.insts.get(c).map(|i| i.ty).ok_or_else(|| {
                            super::invariant("a pointer word's inttoptr is missing")
                        })?;
                        note_access(cx, uses, &mut aliased, *ptr, ty, m)?;
                    }
                }
                InstKind::Load { ptr, .. } => {
                    note_access(cx, uses, &mut aliased, *ptr, inst.ty, m)?;
                }
                // A store of pointer bits into forwarding cells is a store of the pointer.
                InstKind::Store { ptr, .. } if bits_stored(id).is_some() => {
                    let p = bits_stored(id).unwrap_or(*ptr);
                    let ty = cx
                        .value_ty(p)
                        .ok_or_else(|| super::invariant("a stored pointer without a type"))?;
                    note_access(cx, uses, &mut aliased, *ptr, ty, m)?;
                }
                InstKind::Store { val, ptr, .. } => {
                    let ty = cx
                        .value_ty(*val)
                        .ok_or_else(|| super::invariant("a stored value without a type"))?;
                    note_access(cx, uses, &mut aliased, *ptr, ty, m)?;
                    if cx.space_of(ty).is_some() {
                        let pf = cx.get(*ptr, m)?;
                        if !cx.cells.facts.forwards(cx.sets.get(pf.roots)) {
                            open_roots(cx, uses, *val, m)?;
                        }
                    }
                }
                InstKind::Gep { .. }
                | InstKind::Phi { .. }
                | InstKind::Select { .. }
                | InstKind::Freeze(_)
                | InstKind::ICmp { .. }
                | InstKind::Cast {
                    op: CastOp::Bitcast,
                    ..
                } => {}
                // Pointer bits stored only into forwarding cells: the stores are the pointer's.
                InstKind::Cast {
                    op: CastOp::PtrToInt,
                    ..
                } if by_cast.get(&id).is_some_and(|e| e.1) => {}
                // A pointer inserted into an aggregate that stays a value reaches only the
                // extractions of it, which are the pointer itself (module doc).
                InstKind::InsertValue { .. } if held.contains(&id) => {}
                other => {
                    // Any other operand use reaches the memory it points to: an atomic, a call,
                    // `ptrtoint`, an aggregate.
                    let operands = &mut self.operands;
                    operands.clear();
                    let mut count = 0usize;
                    other.for_each_value(|_| count += 1);
                    super::grow_working(operands, count, &mut self.grown, m)?;
                    other.for_each_value(|v| operands.push(v));
                    for &v in operands.iter() {
                        if cx.space_of_value(v) == Some(AddrSpace::Thread) {
                            let f = cx.get(v, m)?;
                            if cx.sets.get(f.roots).contains(&Root::Dynamic)
                                && matches!(
                                    other,
                                    InstKind::AtomicRmw { .. }
                                        | InstKind::CmpXchg { .. }
                                        | InstKind::Call(_)
                                )
                            {
                                aliased = true;
                            }
                        }
                        open_roots(cx, uses, v, m)?;
                    }
                }
            }
        }
        settle_copies(uses, &self.copies, m)?;
        let mut out = Verdict::default();
        for (a, u) in uses.iter() {
            m.tick(1)?;
            if cx.cells.facts.open.contains(a) || !u.pointers() {
                continue;
            }
            if aliased {
                grow(&mut out.open, 1, m)?;
                out.open.push(*a);
                continue;
            }
            let pooled = cx.cells.facts.pooled.contains(a);
            if !pooled && disciplined(u, m)? {
                continue;
            }
            // A cell reached only through its accesses whose slots are not disciplined forwards
            // as a pool when every pointer it holds is an arena pointer of the thread space; a
            // pool that this round's facts show reached otherwise, or holding another pointer,
            // opens (module doc).
            let pool = !u.open && thread_only(cx, u, m)? && arena_only(cx, *a, m)?;
            if pool && !pooled {
                grow(&mut out.pooled, 1, m)?;
                out.pooled.push(*a);
            } else if !pool {
                grow(&mut out.open, 1, m)?;
                out.open.push(*a);
            }
        }
        Ok(out)
    }
}

/// What one round's verification finds ([`verify`]).
#[derive(Default)]
pub(super) struct Verdict {
    /// The allocas it opens.
    pub(super) open: Vec<InstId>,
    /// The allocas that forward as pools from the next round.
    pub(super) pooled: Vec<InstId>,
}

impl Verdict {
    /// The round's facts stand: nothing opens and no pool is new.
    pub(super) fn holds(&self) -> bool {
        self.open.is_empty() && self.pooled.is_empty()
    }

    /// The ledger bytes the two lists hold.
    pub(super) fn bytes(&self) -> usize {
        (self.open.capacity() + self.pooled.capacity())
            .saturating_mul(std::mem::size_of::<InstId>())
    }
}

/// Whether every pointer access of `u` is of a thread-space pointer: one work unit per slot.
fn thread_only(cx: &Cx<'_>, u: &Use, m: &mut Meter) -> Result<bool, Error> {
    for t in u.wild.iter().chain(u.slots.values()) {
        m.tick(1)?;
        if cx.space_of(*t) != Some(AddrSpace::Thread) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether every pointer cell `a` holds this round is an arena pointer of the thread space: null,
/// a stack allocation, or `Dynamic` (some arena offset). A table parameter, a global, or an
/// intrinsic's pointer has another representation. One work unit per slot's content and per root.
fn arena_only(cx: &Cx<'_>, a: InstId, m: &mut Meter) -> Result<bool, Error> {
    let all = (a, Slot::At(i64::MIN))..=(a, Slot::Any);
    for (_, x) in cx.cells.contents.range(all) {
        m.tick(1)?;
        for r in cx.sets.get(x.roots) {
            m.tick(1)?;
            if !matches!(r, Root::Null | Root::Alloca(_) | Root::Dynamic) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// The thread-space pointers in `body` that an instruction turns into data or hands to code no
/// cell sees ([`Cells::escapes`]): every thread-space operand of an instruction other than a load
/// or store (whose stored pointer the store pass follows), an address computation (`getelementptr`,
/// `bitcast`, `phi`, `select`, `freeze`), a compare, or a call of an `air.*` or `llvm.*` intrinsic
/// (whose pointer operands are memory it reads or writes). One work unit per instruction and per
/// operand.
pub(super) fn escapes(cx: &Cx<'_>, body: &Body, m: &mut Meter) -> Result<Vec<Value>, Error> {
    let mut out = Vec::new();
    for (_, inst) in body.insts.iter() {
        m.tick(1)?;
        match &inst.kind {
            InstKind::Load { .. }
            | InstKind::Store { .. }
            | InstKind::Gep { .. }
            | InstKind::Phi { .. }
            | InstKind::Select { .. }
            | InstKind::Freeze(_)
            | InstKind::ICmp { .. }
            | InstKind::Cast {
                op: CastOp::Bitcast,
                ..
            } => continue,
            InstKind::Call(call)
                if matches!(call.callee, Callee::Air { .. } | Callee::Llvm { .. }) =>
            {
                continue
            }
            _ => {}
        }
        let mut found = Ok(());
        inst.kind.for_each_value(|v| {
            if found.is_err() || cx.space_of_value(v) != Some(AddrSpace::Thread) {
                return;
            }
            found = m
                .tick(1)
                .map_err(Error::from)
                .and_then(|()| grow(&mut out, 1, m))
                .map(|()| out.push(v));
        });
        found?;
    }
    Ok(out)
}

/// The stores of pointer bits in `body`: each plain store whose value is a `ptrtoint` of a pointer
/// to an integer of the pointer's store size, every use of which is the value of a plain store;
/// `(store, ptrtoint, pointer, address)`.
pub(super) fn bit_stores(
    cx: &Cx<'_>,
    body: &Body,
    m: &mut Meter,
) -> Result<Vec<(InstId, InstId, Value, Value)>, Error> {
    let mut casts: BTreeMap<InstId, Value> = BTreeMap::new();
    let mut charged = 0usize;
    let mut out = Vec::new();
    let mut find = |charged: &mut usize| -> Result<(), Error> {
        for (id, inst) in body.insts.iter() {
            m.tick(1)?;
            let InstKind::Cast {
                op: CastOp::PtrToInt,
                v,
            } = &inst.kind
            else {
                continue;
            };
            let Some(pty) = cx.value_ty(*v).filter(|t| cx.space_of(*t).is_some()) else {
                continue;
            };
            let (Some(a), Some(b)) = (cx.layouts.store_size(pty), cx.layouts.store_size(inst.ty))
            else {
                continue;
            };
            if a == b {
                m.charge(map_entry_bytes::<InstId, Value>())?;
                *charged += map_entry_bytes::<InstId, Value>();
                casts.insert(id, *v);
            }
        }
        if casts.is_empty() {
            return Ok(());
        }
        // A use other than a plain store's value observes the integer.
        let mut observed: BTreeSet<InstId> = BTreeSet::new();
        for (id, inst) in body.insts.iter() {
            m.tick(1)?;
            if let InstKind::Store {
                val: Value::Inst(c),
                ptr,
                atomic: None,
                ..
            } = &inst.kind
            {
                if let Some(&p) = casts.get(c) {
                    grow(&mut out, 1, m)?;
                    out.push((id, *c, p, *ptr));
                    continue;
                }
            }
            let mut added = 0usize;
            inst.kind.for_each_value(|v| {
                if let Value::Inst(c) = v {
                    if casts.contains_key(&c) && observed.insert(c) {
                        added += 1;
                    }
                }
            });
            if added > 0 {
                let n = added.saturating_mul(map_entry_bytes::<InstId, ()>());
                m.charge(n)?;
                *charged += n;
            }
        }
        out.retain(|(_, c, _, _)| !observed.contains(c));
        Ok(())
    };
    let found = find(&mut charged);
    m.release(charged);
    found?;
    Ok(out)
}

/// The pointer-word loads of `body`: each plain integer load whose every use is an `inttoptr` that
/// round-trips it (`provenance::round_trip`: to a pointer of the load's store size), with its
/// address and those `inttoptr`s.
pub(super) fn word_loads(
    cx: &Cx<'_>,
    body: &Body,
    m: &mut Meter,
) -> Result<BTreeMap<InstId, (Value, Vec<InstId>)>, Error> {
    // The round trips, by the load they read.
    let mut out: BTreeMap<InstId, (Value, Vec<InstId>)> = BTreeMap::new();
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        let InstKind::Cast {
            op: CastOp::IntToPtr,
            v: Value::Inst(l),
        } = &inst.kind
        else {
            continue;
        };
        let Some(ptr) = super::round_trip(cx.e, cx.layouts, body, Value::Inst(*l), inst.ty) else {
            continue;
        };
        if !out.contains_key(l) {
            m.charge(map_entry_bytes::<InstId, (Value, Vec<InstId>)>())?;
        }
        let e = out.entry(*l).or_insert((ptr, Vec::new()));
        grow(&mut e.1, 1, m)?;
        e.1.push(id);
    }
    if out.is_empty() {
        return Ok(out);
    }
    // A load with any other use is an integer.
    let mut observed: BTreeSet<InstId> = BTreeSet::new();
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        if let InstKind::Cast {
            op: CastOp::IntToPtr,
            v: Value::Inst(l),
        } = &inst.kind
        {
            if out.get(l).is_some_and(|(_, cs)| cs.contains(&id)) {
                continue;
            }
        }
        let mut added = 0usize;
        inst.kind.for_each_value(|v| {
            if let Value::Inst(l) = v {
                if out.contains_key(&l) && observed.insert(l) {
                    added += 1;
                }
            }
        });
        m.charge(added.saturating_mul(map_entry_bytes::<InstId, ()>()))?;
    }
    m.release(observed.len() * map_entry_bytes::<InstId, ()>());
    for l in observed {
        if let Some((_, cs)) = out.remove(&l) {
            m.release(
                map_entry_bytes::<InstId, (Value, Vec<InstId>)>()
                    + cs.capacity() * std::mem::size_of::<InstId>(),
            );
        }
    }
    Ok(out)
}

/// The copies of pointer bits in `body`: `(load, store)` for each store whose value is a plain
/// integer load of a pointer's store size, every use of which is the value of a plain store.
pub(super) fn bit_copies(
    cx: &Cx<'_>,
    body: &Body,
    m: &mut Meter,
) -> Result<Vec<(InstId, InstId)>, Error> {
    let mut loads: BTreeSet<InstId> = BTreeSet::new();
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        let InstKind::Load {
            volatile: false,
            atomic: None,
            ..
        } = &inst.kind
        else {
            continue;
        };
        let int = matches!(
            cx.e.module.types().get(inst.ty),
            Some(crate::air::ir::Type::Int(_))
        );
        if int && cx.layouts.store_size(inst.ty) == Some(POINTER_BYTES as u64) {
            m.charge(map_entry_bytes::<InstId, ()>())?;
            loads.insert(id);
        }
    }
    let mut out = Vec::new();
    if loads.is_empty() {
        return Ok(out);
    }
    let mut observed: BTreeSet<InstId> = BTreeSet::new();
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        if let InstKind::Store {
            val: Value::Inst(l),
            volatile: false,
            atomic: None,
            ..
        } = &inst.kind
        {
            if loads.contains(l) {
                grow(&mut out, 1, m)?;
                out.push((*l, id));
                continue;
            }
        }
        let mut added = 0usize;
        inst.kind.for_each_value(|v| {
            if let Value::Inst(l) = v {
                if loads.contains(&l) && observed.insert(l) {
                    added += 1;
                }
            }
        });
        m.charge(added.saturating_mul(map_entry_bytes::<InstId, ()>()))?;
    }
    out.retain(|(l, _)| !observed.contains(l));
    m.release((loads.len() + observed.len()) * map_entry_bytes::<InstId, ()>());
    Ok(out)
}

/// The slot copy a `memcpy` or `memmove` call is, if it is one: a translation-time constant length
/// between two pointers into forwarding cells at constant offsets.
fn slot_copy(
    cx: &mut Cx<'_>,
    call: &Call,
    pointers: usize,
    len: Option<u64>,
    m: &mut Meter,
) -> Result<Option<(CopyEnds, i64)>, Error> {
    let (Some(len), 2, [dst, src, ..]) = (len, pointers, &call.args[..]) else {
        return Ok(None);
    };
    let Ok(len) = i64::try_from(len) else {
        return Ok(None);
    };
    if cx.space_of_value(*dst) != Some(AddrSpace::Thread)
        || cx.space_of_value(*src) != Some(AddrSpace::Thread)
    {
        return Ok(None);
    }
    Ok(cx.copy_endpoints(*dst, *src, m)?.map(|ends| (ends, len)))
}

/// Settle the slot copies. A copy moves each source pointer slot that lies wholly inside its bytes
/// to the destination slot at the same distance from the copy's start, so those slots become the
/// destination's too, with their pointer types (to a fixpoint, for chains of copies). A copy is
/// disciplined when neither side has a pointer access at an unknown offset, no pointer slot of
/// either side straddles the copied bytes' ends, and each destination slot inside them is a slot of
/// every source cell at the same distance, of the same type: the copy then moves pointers onto
/// pointer slots and data onto data. Any other copy opens the cells it touches through its bytes,
/// as the data access it then is: a cell that is open, or that a copy cannot keep closed, makes
/// each of its copies a data access of the other side's copied bytes, repeated until no more cells
/// open.
fn settle_copies(
    uses: &mut BTreeMap<InstId, Use>,
    copies: &[(CopyEnds, i64)],
    m: &mut Meter,
) -> Result<(), Error> {
    if copies.is_empty() {
        return Ok(());
    }
    // The destination slots each copy implies.
    loop {
        let mut changed = false;
        for (ends, len) in copies {
            m.tick(1)?;
            let mut moved: Vec<(i64, TypeId)> = Vec::new();
            let mut moved_bytes = 0usize;
            for a in &ends.src {
                let Some(u) = uses.get(a) else { continue };
                for (&o, &ty) in &u.slots {
                    m.tick(1)?;
                    if let Some(t) = ends.image(o, *len) {
                        push_working(&mut moved, (t, ty), &mut moved_bytes, m)?;
                    }
                }
            }
            let placed = (|| {
                for b in &ends.dst {
                    let Some(u) = uses.get_mut(b) else { continue };
                    for &(t, ty) in &moved {
                        m.tick(1)?;
                        match u.slots.get(&t) {
                            Some(&had) if had == ty => {}
                            Some(_) => u.open = true,
                            None => {
                                m.charge(map_entry_bytes::<i64, TypeId>())?;
                                u.slots.insert(t, ty);
                                changed = true;
                            }
                        }
                    }
                }
                Ok::<(), Error>(())
            })();
            m.release(moved_bytes);
            placed?;
        }
        if !changed {
            break;
        }
    }
    let mut kept = Kept::default();
    let settled = kept.settle(uses, copies, m);
    m.release(kept.charged);
    settled
}

/// Which copies are data accesses and which cells are disciplined, as [`settle_copies`] settles
/// them: the working tables, charged to the ledger as they grow.
#[derive(Default)]
struct Kept {
    /// Whether each copy is a data access of its bytes.
    data: Vec<bool>,
    /// Whether each cell of a copy is disciplined ([`disciplined`]): read once, then kept as data
    /// accesses are added to the cell.
    cells: BTreeMap<InstId, bool>,
    /// The ledger bytes the tables hold.
    charged: usize,
}

impl Kept {
    /// Mark the copies that do not keep their cells closed as data accesses, and add their
    /// bytes to both sides' data, until no more cells open.
    fn settle(
        &mut self,
        uses: &mut BTreeMap<InstId, Use>,
        copies: &[(CopyEnds, i64)],
        m: &mut Meter,
    ) -> Result<(), Error> {
        // Copies that keep their cells closed; the others are data accesses.
        for (ends, len) in copies {
            m.tick(1)?;
            let data = !copy_disciplined(uses, ends, *len, m)?;
            push_working(&mut self.data, data, &mut self.charged, m)?;
        }
        for (ends, _) in copies {
            for c in ends.src.iter().chain(&ends.dst) {
                m.tick(1)?;
                if self.cells.contains_key(c) {
                    continue;
                }
                let Some(u) = uses.get(c) else { continue };
                let kept = disciplined(u, m)?;
                m.charge(map_entry_bytes::<InstId, bool>())?;
                self.charged = self
                    .charged
                    .saturating_add(map_entry_bytes::<InstId, bool>());
                self.cells.insert(*c, kept);
            }
        }
        loop {
            let mut changed = false;
            for (i, (ends, len)) in copies.iter().enumerate() {
                m.tick(1)?;
                let mut broken = self.data.get(i).copied().unwrap_or(true);
                for c in ends.src.iter().chain(&ends.dst) {
                    if broken {
                        break;
                    }
                    m.tick(1)?;
                    broken = self.cells.get(c) == Some(&false);
                }
                if !broken {
                    continue;
                }
                if let Some(d) = self.data.get_mut(i) {
                    if !*d {
                        *d = true;
                        changed = true;
                    }
                }
                for (cells, at) in [(&ends.src, ends.s), (&ends.dst, ends.d)] {
                    let hi = at.saturating_add(*len);
                    for c in cells {
                        m.tick(1)?;
                        let Some(u) = uses.get_mut(c) else { continue };
                        if !note_range(u, (at, hi), m)? {
                            continue;
                        }
                        // One more data access leaves a disciplined cell disciplined when it
                        // keeps clear of every pointer slot, and of a wild access, which no data
                        // may meet ([`disciplined`]).
                        if let Some(k) = self.cells.get_mut(c) {
                            if *k && (u.wild.is_some() || !clear_of_slots(u, at, hi)) {
                                *k = false;
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                return Ok(());
            }
        }
    }
}

/// Whether one copy moves pointer slots onto pointer slots and data onto data ([`settle_copies`]):
/// one work unit per cell, per slot, and per destination slot it reads.
fn copy_disciplined(
    uses: &BTreeMap<InstId, Use>,
    ends: &CopyEnds,
    len: i64,
    m: &mut Meter,
) -> Result<bool, Error> {
    for c in ends.src.iter().chain(&ends.dst) {
        m.tick(1)?;
        if uses.get(c).is_none_or(|u| u.wild.is_some()) {
            return Ok(false);
        }
    }
    // No slot straddles either end of the copied bytes.
    let sides = [(&ends.src, ends.s), (&ends.dst, ends.d)];
    for (cells, at) in sides {
        let hi = at.saturating_add(len);
        for c in cells {
            let Some(u) = uses.get(c) else {
                return Ok(false);
            };
            for &o in u.slots.keys() {
                m.tick(1)?;
                let end = o.saturating_add(POINTER_BYTES);
                if !(end <= at || o >= hi || (o >= at && end <= hi)) {
                    return Ok(false);
                }
            }
        }
    }
    // Every destination slot inside the copied bytes is a slot of every source cell.
    let hi = ends.d.saturating_add(len);
    for b in &ends.dst {
        let Some(u) = uses.get(b) else {
            return Ok(false);
        };
        for (&q, &ty) in u.slots.range(ends.d..hi) {
            m.tick(1)?;
            let Some(pre) = q.checked_sub(ends.d).and_then(|x| x.checked_add(ends.s)) else {
                return Ok(false);
            };
            for a in &ends.src {
                m.tick(1)?;
                if uses.get(a).and_then(|a| a.slots.get(&pre)) != Some(&ty) {
                    return Ok(false);
                }
            }
        }
    }
    Ok(true)
}

impl Use {
    /// Some pointer is stored into or loaded from the alloca: only then does forwarding matter.
    fn pointers(&self) -> bool {
        self.wild.is_some() || !self.slots.is_empty()
    }
}

/// Whether one alloca's accesses keep its pointer slots apart from each other and from its data:
/// one work unit per slot and per data access it reads.
fn disciplined(u: &Use, m: &mut Meter) -> Result<bool, Error> {
    if u.open {
        return Ok(false);
    }
    if let Some(w) = u.wild {
        if !u.data.is_empty() || u.data_unknown {
            return Ok(false);
        }
        // A zero fill beside them covers whole slots: every access at an unknown offset lies at
        // a multiple of the pointer's size, and so does each end of every fill.
        let whole = |&(lo, hi): &(i64, i64)| {
            lo.rem_euclid(POINTER_BYTES) == 0 && hi.rem_euclid(POINTER_BYTES) == 0
        };
        if !u.zero.is_empty() && (u.wild_unaligned || !u.zero.iter().all(whole)) {
            return Ok(false);
        }
        for t in u.slots.values() {
            m.tick(1)?;
            if *t != w {
                return Ok(false);
            }
        }
        return fills_cover_slots(u, m);
    }
    if u.data_unknown {
        return Ok(false);
    }
    let mut last: Option<i64> = None;
    for &o in u.slots.keys() {
        m.tick(1)?;
        if last.is_some_and(|l| o.saturating_sub(l) < POINTER_BYTES) {
            return Ok(false);
        }
        last = Some(o);
    }
    // Every data access keeps clear of every pointer slot.
    for &(lo, hi) in &u.data {
        m.tick(1)?;
        if !clear_of_slots(u, lo, hi) {
            return Ok(false);
        }
    }
    fills_cover_slots(u, m)
}

/// Whether every zero fill of `u` covers each exact pointer slot wholly or not at all.
fn fills_cover_slots(u: &Use, m: &mut Meter) -> Result<bool, Error> {
    for &(lo, hi) in &u.zero {
        m.tick(1)?;
        if lo >= hi {
            continue;
        }
        for &o in u
            .slots
            .range(lo.saturating_sub(POINTER_BYTES - 1)..hi)
            .map(|(o, _)| o)
        {
            m.tick(1)?;
            if o < lo || o.saturating_add(POINTER_BYTES) > hi {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Whether data bytes `[lo, hi)` keep clear of every pointer slot of `u`: the slot nearest below
/// their end does.
fn clear_of_slots(u: &Use, lo: i64, hi: i64) -> bool {
    u.slots
        .range(..hi)
        .next_back()
        .is_none_or(|(&o, _)| o.saturating_add(POINTER_BYTES) <= lo)
}

/// Add data bytes `[lo, hi)` to `u`'s data accesses, charging a new entry to the ledger; whether
/// they were new.
fn note_range(u: &mut Use, range: (i64, i64), m: &mut Meter) -> Result<bool, Error> {
    if u.data.contains(&range) {
        return Ok(false);
    }
    m.charge(tree_bytes(1, std::mem::size_of::<(i64, i64)>()))?;
    Ok(u.data.insert(range))
}

/// The store size of a pointer (the AIR datalayout's `p:64`).
const POINTER_BYTES: i64 = 8;

/// Record an access of type `ty` through `ptr`.
fn note_access(
    cx: &mut Cx<'_>,
    uses: &mut BTreeMap<InstId, Use>,
    aliased: &mut bool,
    ptr: Value,
    ty: TypeId,
    m: &mut Meter,
) -> Result<(), Error> {
    if cx.space_of_value(ptr) != Some(AddrSpace::Thread) {
        return Ok(());
    }
    let f = cx.get(ptr, m)?;
    let roots = cx.sets.get(f.roots).to_vec();
    if roots.contains(&Root::Dynamic) {
        *aliased = true;
    }
    let pointer = cx.space_of(ty).is_some();
    let size = cx.layouts.store_size(ty);
    // A pointer moved through an address that does not forward is lost to every cell it reaches.
    let lost = pointer && !cx.cells.facts.forwards(&roots);
    for r in roots {
        let Root::Alloca(a) = r else { continue };
        if !uses.contains_key(&a) {
            m.charge(map_entry_bytes::<InstId, Use>())?;
        }
        let u = uses.entry(a).or_default();
        if lost {
            u.open = true;
        } else if pointer {
            if size != Some(POINTER_BYTES as u64) {
                u.open = true;
                continue;
            }
            match Slot::of(f.range) {
                Slot::At(o) => {
                    if !u.slots.contains_key(&o) {
                        m.charge(map_entry_bytes::<i64, TypeId>())?;
                    }
                    if u.slots.insert(o, ty).is_some_and(|t| t != ty) {
                        u.open = true;
                    }
                }
                Slot::Any => {
                    if u.wild.replace(ty).is_some_and(|t| t != ty) {
                        u.open = true;
                    }
                    u.wild_unaligned |= (1i64 << f.align) < POINTER_BYTES;
                }
            }
        } else if cx.e.module.types().holds_pointer(ty) {
            u.open = true;
        } else {
            let reach = size.map(super::Reach::at);
            match Extent::of(f.range, reach) {
                Extent::Empty => {}
                Extent::Span { lo, hi } => {
                    note_range(u, (lo, hi), m)?;
                }
                Extent::Unknown => u.data_unknown = true,
            }
        }
    }
    Ok(())
}

/// The copy `call` is when it is a `memcpy` or `memmove` of a translation-time constant length
/// between two thread-space pointers: `(destination, source, length)`.
pub(super) fn constant_copy(cx: &Cx<'_>, call: &Call) -> Option<(Value, Value, u64)> {
    let (2, Some(len)) = memory_intrinsic(cx, call)? else {
        return None;
    };
    let [dst, src, ..] = call.args[..] else {
        return None;
    };
    let thread = |v| cx.space_of_value(v) == Some(AddrSpace::Thread);
    (thread(dst) && thread(src)).then_some((dst, src, len))
}

/// The zero fill `call` is when it is a `memset` of zero bytes of a translation-time constant,
/// non-zero length into thread memory: `(address, length)` (module doc).
pub(super) fn zero_fill(cx: &Cx<'_>, call: &Call) -> Option<(Value, u64)> {
    let (1, Some(len)) = memory_intrinsic(cx, call)? else {
        return None;
    };
    let [dst, fill, ..] = call.args[..] else {
        return None;
    };
    let zero = cx.const_index(fill) == Some(0);
    (zero && len > 0 && cx.space_of_value(dst) == Some(AddrSpace::Thread)).then_some((dst, len))
}

/// Record the zero bytes `call` writes when it is a zero fill into forwarding cells at a
/// translation-time constant offset; whether it is.
fn note_zeros(
    cx: &mut Cx<'_>,
    uses: &mut BTreeMap<InstId, Use>,
    call: &Call,
    m: &mut Meter,
) -> Result<bool, Error> {
    let Some((ptr, len)) = zero_fill(cx, call) else {
        return Ok(false);
    };
    let f = cx.get(ptr, m)?;
    let Some((cells, lo, hi)) = cx.zero_bytes(f, len) else {
        return Ok(false);
    };
    for a in cells {
        if !uses.contains_key(&a) {
            m.charge(map_entry_bytes::<InstId, Use>())?;
        }
        let u = uses.entry(a).or_default();
        if !u.zero.contains(&(lo, hi)) {
            m.charge(tree_bytes(1, std::mem::size_of::<(i64, i64)>()))?;
            u.zero.insert((lo, hi));
        }
    }
    Ok(true)
}

/// The pointer operands (the first `n`) of a memory intrinsic and its byte count when that is a
/// translation-time constant; `None` for any other call.
fn memory_intrinsic(cx: &Cx<'_>, call: &Call) -> Option<(usize, Option<u64>)> {
    let n = match &call.callee {
        Callee::Llvm {
            op: LlvmIntrinsic::Memset { .. },
            ..
        } => 1,
        Callee::Llvm {
            op: LlvmIntrinsic::Memcpy { .. } | LlvmIntrinsic::Memmove { .. },
            ..
        } => 2,
        _ => return None,
    };
    let len = call
        .args
        .get(2)
        .and_then(|&l| cx.const_index(l))
        .and_then(|l| u64::try_from(l).ok());
    Some((n, len))
}

/// Record a data access of `len` bytes (unknown when `None`) through `ptr`.
fn note_data(
    cx: &mut Cx<'_>,
    uses: &mut BTreeMap<InstId, Use>,
    aliased: &mut bool,
    ptr: Value,
    len: Option<u64>,
    m: &mut Meter,
) -> Result<(), Error> {
    if cx.space_of_value(ptr) != Some(AddrSpace::Thread) {
        return Ok(());
    }
    let f = cx.get(ptr, m)?;
    let roots = cx.sets.get(f.roots).to_vec();
    if roots.contains(&Root::Dynamic) {
        *aliased = true;
    }
    for r in roots {
        let Root::Alloca(a) = r else { continue };
        if !uses.contains_key(&a) {
            m.charge(map_entry_bytes::<InstId, Use>())?;
        }
        let u = uses.entry(a).or_default();
        match Extent::of(f.range, len.map(super::Reach::at)) {
            Extent::Empty => {}
            Extent::Span { lo, hi } => {
                note_range(u, (lo, hi), m)?;
            }
            Extent::Unknown => u.data_unknown = true,
        }
    }
    Ok(())
}

/// Open every alloca pointer operand `v` may reach.
fn open_roots(
    cx: &mut Cx<'_>,
    uses: &mut BTreeMap<InstId, Use>,
    v: Value,
    m: &mut Meter,
) -> Result<(), Error> {
    if cx.space_of_value(v) != Some(AddrSpace::Thread) {
        return Ok(());
    }
    let f = cx.get(v, m)?;
    for r in cx.sets.get(f.roots).to_vec() {
        if let Root::Alloca(a) = r {
            if !uses.contains_key(&a) {
                m.charge(map_entry_bytes::<InstId, Use>())?;
            }
            uses.entry(a).or_default().open = true;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meter() -> Meter {
        crate::pipeline::testing::meter()
    }

    /// Cell `x` (instruction 0) holds a pointer slot at 0 and `data` data accesses of four bytes
    /// past it; cell `y` (instruction 1) is empty. `copies` copies of `x`'s first 8 bytes into
    /// `y` each move the slot onto `y`'s slot 0, so each keeps both cells closed. Settled, the
    /// work units charged.
    fn settle(copies: usize, data: i64) -> u64 {
        let (x, y) = (InstId::new(0), InstId::new(1));
        let ty = TypeId::new(7);
        let mut cell = Use::default();
        cell.slots.insert(0, ty);
        for i in 0..data {
            cell.data.insert((8 + 4 * i, 12 + 4 * i));
        }
        let mut uses = BTreeMap::from([(x, cell), (y, Use::default())]);
        let copy = CopyEnds {
            src: vec![x],
            s: 0,
            dst: vec![y],
            d: 0,
        };
        let copies = vec![(copy, 8); copies];
        let mut m = meter();
        settle_copies(&mut uses, &copies, &mut m).expect("settle");
        assert_eq!(
            m.ledger(),
            map_entry_bytes::<i64, TypeId>() as u64,
            "y's slot alone"
        );
        let (Some(xu), Some(yu)) = (uses.get(&x), uses.get(&y)) else {
            panic!("the cells");
        };
        assert_eq!(yu.slots.get(&0), Some(&ty), "the copy moved the slot");
        let mut check = meter();
        assert!(disciplined(xu, &mut check).expect("x") && disciplined(yu, &mut check).expect("y"));
        m.work()
    }

    /// Settling slot copies is metered by what it reads. A cell's data accesses are read once, so
    /// 1024 more of them in the copied cell cost at least 1024 units and at most a few per access
    /// whether 64 or 1024 copies read the cell. Previously the copies' checks re-read every slot
    /// and data access of both cells per copy and round, unticked: 1024 more data accesses cost no
    /// work unit at all.
    #[test]
    fn settling_slot_copies_reads_each_cell_s_data_once_and_charges_it() {
        for copies in [64, 1024] {
            let (few, many) = (settle(copies, 0), settle(copies, 1024));
            assert!(
                many >= few + 1024,
                "{copies} copies: {many} units against {few}"
            );
            assert!(
                many <= few + 2 * 1024,
                "{copies} copies: {many} units against {few}"
            );
        }
    }

    /// A data access already noted is noted once: the cell's accesses are an ordered set, and a
    /// copy that breaks adds its bytes to each side once, whatever the number of rounds or
    /// copies (each lookup a search, not a scan).
    #[test]
    fn a_broken_copy_adds_its_bytes_once() {
        let (x, y) = (InstId::new(0), InstId::new(1));
        let ty = TypeId::new(7);
        let mut cell = Use::default();
        cell.slots.insert(0, ty);
        let wild = Use {
            wild: Some(ty),
            ..Use::default()
        };
        let mut uses = BTreeMap::from([(x, cell), (y, wild)]);
        let copy = CopyEnds {
            src: vec![x],
            s: 16,
            dst: vec![y],
            d: 0,
        };
        let copies = vec![(copy, 8); 32];
        let mut m = meter();
        settle_copies(&mut uses, &copies, &mut m).expect("settle");
        let (Some(xu), Some(yu)) = (uses.get(&x), uses.get(&y)) else {
            panic!("the cells");
        };
        assert_eq!(xu.data.iter().copied().collect::<Vec<_>>(), vec![(16, 24)]);
        assert_eq!(yu.data.iter().copied().collect::<Vec<_>>(), vec![(0, 8)]);
        assert!(disciplined(xu, &mut m).expect("x"));
        assert!(
            !disciplined(yu, &mut m).expect("y"),
            "data met a wild access"
        );
    }
}
