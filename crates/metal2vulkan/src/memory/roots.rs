//! Dynamic roots: the root table, the resident-set search, and which roots are wide.
//!
//! **Wide roots.** A buffer pointer whose root set contains `Dynamic` (a pointer loaded from
//! memory or made from an integer) is a *wide* fat pointer `(root, at, host, size: u64)`, `at`
//! being the 64-bit Metal displacement from the allocation's start and `host` and `size` its root
//! entry's (`lower::access::dynamic`). Every value of a
//! phi, `select`, GEP, or cast has an operand whose root set shares a root with its own, so the
//! width is decided per **component**: two roots are connected when some buffer pointer's root
//! set holds both, and every root of a component that holds `Dynamic` is wide
//! ([`wide_roots`]). A pointer is wide exactly when one of its roots is (the null root joins no
//! component: every set may hold it). Operands and results of one pointer operation are
//! therefore always of one width, and a static slot reached by a wide pointer is only ever
//! reached by wide pointers, through the root table (`BufferForm::Psb`).
//!
//! **The root table** is read at the device address in `PushField::RootTableAddress`: 32-byte
//! entries `{host, guest, size, bind_offset}: u64`. Entry 0 is the null root (all fields 0:
//! `guest = 0` makes a miss round-trip through a later `ptrtoint`, `size = 0` puts every access
//! through it out of bounds). Entry 1 is the write-only sink (`host` its device address). Entry
//! `2 + n` is `[[buffer(n)]]`. The **resident set** (the bound arguments and
//! `BindState.used_resources`, overlaps coalesced into the outermost range) is the entries `R0 ..
//! R0 + N`, sorted by `guest` and disjoint but for indirect command buffers' records, which may
//! overlap (each found only at its start, `lower::access::dynamic::search_start`), each with
//! `bind_offset = 0`. The binder states `R0` and
//! `N` in the sink entry's otherwise unread `guest` and `size` fields ([`RESIDENT_FIRST`],
//! [`RESIDENT_COUNT`]); `N` is any count (At most 2¹⁶ before, a larger set
//! `BindError::ResidentSetTooLarge`). `R0` counts entries from the table's address modulo 2⁶⁴, so
//! the set may lie apart from the table, below it too (a consumer's set shared by many tables,
//! `RootTable::resident_fields`).
//!
//! **The search** ([`resolve`]) turns a guest address `i` into a wide pointer: a branch-free
//! binary search of [`SEARCH_STEPS`] unrolled steps for the last resident entry whose `guest ≤ i`,
//! then a loop of the same step while more than one candidate is left, which only a set of more
//! than 2^[`SEARCH_STEPS`] entries enters (one uniform test where every invocation searches one
//! table, no further load), then `i − guest ≤ size` (a one-past-the-end pointer stays in its
//! allocation). A hit is `(root, i − guest)`, with the entry's `host` and `size` for a wide pointer
//! ([`resolve_wide`]); a miss, or an empty set, is `(0, i)` (`(0, i, 0, 0)`). A module with
//! more than one search defines it once ([`define_resolve`]), as it does the handle-table search,
//! and every search is a call of it; a single search is emitted in place. It reads the set as an
//! array of typed entries from its first, `table + 32·R0` in 64-bit wrapping arithmetic, one access
//! chain per field at an index in `0 .. N`, inside the set whatever `R0` is: an index below an
//! array's start is outside it, where SPIR-V gives no address, so a set below the table is reached
//! by the integer arithmetic alone. With `N = 0` it reads the table's entry 0. The root it gives is
//! `R0` plus the index, modulo 2⁶⁴, whose entry is at `table + 32·root` as every root entry is
//! ([`entry_field`]): the entry the search read the wide pointer's `host` and `size` from.

use std::collections::{BTreeMap, BTreeSet};

use super::provenance::{map_entry_bytes, Root};
use super::union_find::UnionFind;
use crate::budget::Meter;
use crate::error::{Error, InternalError, InternalKind};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{FuncId, TypeId, Value};
use crate::spirv::ir::{natural_layout, BinOp, ConstBits, Constant, LayoutRule, MemAccess, Type};

/// Bytes of one root-table entry.
pub const ENTRY_BYTES: u64 = 32;
/// The null root.
pub const NULL_ROOT: u64 = 0;
/// The sink's root.
pub const SINK_ROOT: u64 = 1;
/// Byte offset of an entry's `host` field (the allocation's device address).
pub const HOST: u64 = 0;
/// Byte offset of an entry's `guest` field (the allocation's Metal address).
pub const GUEST: u64 = 8;
/// Byte offset of an entry's `size` field (the padded allocation size).
pub const SIZE: u64 = 16;
/// Byte offset of an entry's `bind_offset` field (parameter slots only; 0 in the resident set).
pub const BIND_OFFSET: u64 = 24;
/// The field of the sink entry that holds the resident set's first root `R0`.
pub const RESIDENT_FIRST: u64 = GUEST;
/// The field of the sink entry that holds the resident set's entry count `N`.
pub const RESIDENT_COUNT: u64 = SIZE;
/// Unrolled steps of the resident-set search: a set of up to 2^`SEARCH_STEPS` entries is
/// searched by them alone; a larger one, by the loop after them too (the module docs).
pub const SEARCH_STEPS: u32 = 16;

/// The wide roots (the module docs' rule): every root of a component, over the given buffer
/// pointer root sets, that holds `Dynamic`.
///
/// The components are a union-find over the roots the sets hold (`memory::union_find`: joined by
/// size, with path halving), so its work is near-linear in the roots it reads: one work unit per
/// set, per root read, and per step of every find. Its tables are charged to the ledger and
/// returned when it ends; the wide set it returns is not charged. Which roots are wide does not
/// depend on the order in which the sets are joined.
pub fn wide_roots<'s>(
    sets: impl Iterator<Item = &'s [Root]>,
    m: &mut Meter,
) -> Result<BTreeSet<Root>, Error> {
    let mut c = Components::default();
    let wide = c.wide(sets, m);
    m.release(c.charged.saturating_add(c.uf.charged()));
    wide
}

/// The components of [`wide_roots`]: each root's dense index in a union-find.
#[derive(Default)]
struct Components {
    index: BTreeMap<Root, u32>,
    uf: UnionFind,
    /// The ledger bytes the index holds.
    charged: usize,
}

impl Components {
    fn wide<'s>(
        &mut self,
        sets: impl Iterator<Item = &'s [Root]>,
        m: &mut Meter,
    ) -> Result<BTreeSet<Root>, Error> {
        let mut any_dynamic = false;
        for set in sets {
            m.tick(1)?;
            let mut first: Option<u32> = None;
            for r in set.iter().copied().filter(|r| *r != Root::Null) {
                m.tick(1)?;
                any_dynamic |= r == Root::Dynamic;
                let i = self.add(r, m)?;
                match first {
                    None => first = Some(i),
                    Some(f) => self.uf.union(f, i, m)?,
                }
            }
        }
        if !any_dynamic {
            return Ok(BTreeSet::new());
        }
        let dynamic = match self.index.get(&Root::Dynamic) {
            Some(&d) => self.uf.find(d, m)?,
            None => return Err(invariant("the dynamic root has no component")),
        };
        let mut wide = BTreeSet::new();
        for (r, &i) in &self.index {
            m.tick(1)?;
            if self.uf.find(i, m)? == dynamic {
                wide.insert(*r);
            }
        }
        Ok(wide)
    }

    /// The index of root `r`, which a root seen first starts as a component of its own.
    fn add(&mut self, r: Root, m: &mut Meter) -> Result<u32, Error> {
        if let Some(&i) = self.index.get(&r) {
            return Ok(i);
        }
        m.charge(map_entry_bytes::<Root, u32>())?;
        self.charged = self.charged.saturating_add(map_entry_bytes::<Root, u32>());
        let i = self.uf.push(m)?;
        self.index.insert(r, i);
        Ok(i)
    }
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

fn u64_ty(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(Type::Int {
        width: 64,
        signed: false,
    })
}

/// A `u64` constant.
pub fn k64(fb: &mut FunctionBuilder<'_>, v: u64) -> BResult<Value> {
    let ty = u64_ty(fb)?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(v),
    })?;
    fb.module().const_value(c)
}

/// The `u64` field at byte `field` of entry `root` (a `u64`) of the root table at `table`.
pub fn entry_field(
    fb: &mut FunctionBuilder<'_>,
    table: Value,
    root: Value,
    field: u64,
) -> BResult<Value> {
    let stride = k64(fb, ENTRY_BYTES)?;
    let at = fb.binary(BinOp::IMul, root, stride)?;
    let at = if field == 0 {
        at
    } else {
        let f = k64(fb, field)?;
        fb.binary(BinOp::IAdd, at, f)?
    };
    let addr = fb.binary(BinOp::IAdd, table, at)?;
    let u64_ = u64_ty(fb)?;
    let p = fb.device_place(addr, u64_)?;
    fb.load(
        p,
        MemAccess {
            aligned: Some(8),
            ..MemAccess::default()
        },
    )
}

/// The resident set a replay preprocess's resident root table at `table` names
/// (`bind::ReplayPreprocess::resident`, `bind::DrawReplayPreprocess::resident`): its first root
/// `R0` and count `N`, from the sink entry, which a consumer points anywhere
/// (`RootTable::resident_fields`), the set's address (`table + 32·R0`, modulo 2⁶⁴), and the
/// address of its entries' unpadded extents, a `u64` each, right after the set, then one zero
/// (`ResidentSet::replay_bytes`; the binder lays its own set out so).
#[derive(Clone, Copy)]
pub struct ResidentView {
    pub first: Value,
    pub count: Value,
    pub set: Value,
    pub extents: Value,
}

/// [`ResidentView`] of the resident root table at `table`.
pub fn resident_view(fb: &mut FunctionBuilder<'_>, table: Value) -> BResult<ResidentView> {
    let e = entries(fb, table)?;
    let sink = k64(fb, SINK_ROOT)?;
    let first = field_of(fb, e, sink, RESIDENT_FIRST)?;
    let count = field_of(fb, e, sink, RESIDENT_COUNT)?;
    let stride = k64(fb, ENTRY_BYTES)?;
    let skip = fb.binary(BinOp::IMul, first, stride)?;
    let set = fb.binary(BinOp::IAdd, table, skip)?;
    let bytes = fb.binary(BinOp::IMul, count, stride)?;
    let extents = fb.binary(BinOp::IAdd, set, bytes)?;
    Ok(ResidentView {
        first,
        count,
        set,
        extents,
    })
}

/// The unpadded extent of root `root` of `view`'s set, which a search of it found where `found`
/// holds; where it does not, the first entry's (or the zero after an empty set's), a read that
/// stays in the set's bytes and that the caller's test of `found` discards.
pub fn extent_of(
    fb: &mut FunctionBuilder<'_>,
    view: &ResidentView,
    root: Value,
    found: Value,
) -> BResult<Value> {
    let index = fb.binary(BinOp::ISub, root, view.first)?;
    let zero = k64(fb, 0)?;
    let index = fb.select(found, index, zero)?;
    let eight = k64(fb, 8)?;
    let rel = fb.binary(BinOp::IMul, index, eight)?;
    let addr = fb.binary(BinOp::IAdd, view.extents, rel)?;
    let u64_ = u64_ty(fb)?;
    let p = fb.device_place(addr, u64_)?;
    fb.load(
        p,
        MemAccess {
            aligned: Some(8),
            ..MemAccess::default()
        },
    )
}

/// Define the module's resident-set search: a function `fn(table: u64, guest: u64) -> wide`
/// (`wide` a `u64vec4`), the wide pointer `(root, at, host, size)` of `guest` through the root
/// table at `table` ([`resolve_wide`]). Every search of the module calls it, so its words are paid
/// once.
pub fn define_resolve(mb: &mut ModuleBuilder, wide: TypeId) -> BResult<FuncId> {
    let u64_ = mb.ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let mut fb = mb.function(wide)?;
    let table = fb.param(u64_)?;
    let guest = fb.param(u64_)?;
    let r = resolve_wide(&mut fb, table, guest)?;
    let fat = fb.composite_construct(wide, &r)?;
    fb.ret(Some(fat))?;
    fb.finish()
}

/// The root table at `table` as a pointer to an array of its 32-byte entries.
fn entries(fb: &mut FunctionBuilder<'_>, table: Value) -> BResult<Value> {
    let u64_ = u64_ty(fb)?;
    let fields = [u64_; (ENTRY_BYTES / 8) as usize];
    let layout = natural_layout(fb.module().types(), LayoutRule::Std430, false, &fields)?;
    let entry = fb.module().ty(Type::Struct {
        members: fields.into(),
        layout: Some(layout),
    })?;
    let array = fb.module().ty(Type::RuntimeArray {
        elem: entry,
        stride: Some(ENTRY_BYTES as u32),
    })?;
    fb.device_place(table, array)
}

/// The `u64` field at byte `field` of entry `root` (a `u64`) of [`entries`] `e`.
fn field_of(fb: &mut FunctionBuilder<'_>, e: Value, root: Value, field: u64) -> BResult<Value> {
    let member = fb.u32_const((field / 8) as u32)?;
    let member = fb.module().const_value(member)?;
    let p = fb.access_chain(e, &[root, member])?;
    fb.load(
        p,
        MemAccess {
            aligned: Some(8),
            ..MemAccess::default()
        },
    )
}

/// The wide pointer `[root, at, host, size]` (four `u64`s) of guest address `guest` (a `u64`)
/// through the resident set of the root table at `table` ([`resolve`]): a hit's `at` is `i −
/// guest` (a resident entry's `bind_offset` is 0) and `host` and `size` its entry's, read here
/// once; a miss is `[0, i, 0, 0]`, the null root's fields.
pub fn resolve_wide(
    fb: &mut FunctionBuilder<'_>,
    table: Value,
    guest: Value,
) -> BResult<[Value; 4]> {
    let s = search(fb, table, guest)?;
    let zero = k64(fb, 0)?;
    let host = field_of(fb, s.set, s.base, HOST)?;
    let host = fb.select(s.hit, host, zero)?;
    let size = fb.select(s.hit, s.size, zero)?;
    Ok([s.root, s.off, host, size])
}

/// The wide pointer `(root, offset)` (two `u64`s) of guest address `guest` (a `u64`) through the
/// resident set of the root table at `table` (the module docs' search). The search walks the
/// set's entries from its first, `R0` entries from the table; an empty set starts at the null
/// root, whose entry every step then reads (the length stays 0) and whose fields `(guest 0,
/// size 0)` give the miss `(0, i)` whether or not the final test hits.
pub fn resolve(
    fb: &mut FunctionBuilder<'_>,
    table: Value,
    guest: Value,
) -> BResult<(Value, Value)> {
    let s = search(fb, table, guest)?;
    Ok((s.root, s.off))
}

/// A finished search ([`search`]).
struct Found {
    /// The set's entries, from its first.
    set: Value,
    /// The last candidate's index in the set.
    base: Value,
    /// Its entry's `size`.
    size: Value,
    /// Whether it holds the address.
    hit: Value,
    root: Value,
    off: Value,
}

/// The search of [`resolve`].
fn search(fb: &mut FunctionBuilder<'_>, table: Value, guest: Value) -> BResult<Found> {
    let e = entries(fb, table)?;
    let sink = k64(fb, SINK_ROOT)?;
    let first = field_of(fb, e, sink, RESIDENT_FIRST)?;
    let count = field_of(fb, e, sink, RESIDENT_COUNT)?;
    let zero = k64(fb, 0)?;
    let valid = fb.binary(BinOp::INotEqual, count, zero)?;
    let start = fb.select(valid, first, zero)?;
    let stride = k64(fb, ENTRY_BYTES)?;
    let skip = fb.binary(BinOp::IMul, start, stride)?;
    let at = fb.binary(BinOp::IAdd, table, skip)?;
    let set = entries(fb, at)?;
    let mut base = zero;
    let mut len = count;
    for _ in 0..SEARCH_STEPS {
        (base, len) = search_step(fb, set, guest, base, len)?;
    }
    let base = search_rest(fb, set, guest, base, len)?;
    let g = field_of(fb, set, base, GUEST)?;
    let size = field_of(fb, set, base, SIZE)?;
    let rel = fb.binary(BinOp::ISub, guest, g)?;
    let above = fb.binary(BinOp::ULessThanEqual, g, guest)?;
    let within = fb.binary(BinOp::ULessThanEqual, rel, size)?;
    let hit = fb.binary(BinOp::LogicalAnd, above, within)?;
    let null = k64(fb, NULL_ROOT)?;
    let found = fb.binary(BinOp::IAdd, start, base)?;
    let root = fb.select(hit, found, null)?;
    let off = fb.select(hit, rel, guest)?;
    Ok(Found {
        set,
        base,
        size,
        hit,
        root,
        off,
    })
}

/// One step of the search: of the `len` candidates from `base` of the set `set`, the half from
/// the middle where its entry's Metal address is at most `guest`, else the half before it.
fn search_step(
    fb: &mut FunctionBuilder<'_>,
    set: Value,
    guest: Value,
    base: Value,
    len: Value,
) -> BResult<(Value, Value)> {
    let one = k64(fb, 1)?;
    let half = fb.binary(BinOp::ShiftRightLogical, len, one)?;
    let mid = fb.binary(BinOp::IAdd, base, half)?;
    let g = field_of(fb, set, mid, GUEST)?;
    let take = fb.binary(BinOp::ULessThanEqual, g, guest)?;
    let base = fb.select(take, mid, base)?;
    let len = fb.binary(BinOp::ISub, len, half)?;
    Ok((base, len))
}

/// The search's end past its unrolled steps: [`search_step`] while more than one of the `len`
/// candidates from `base` is left, which only a set of more than 2^[`SEARCH_STEPS`] entries
/// leaves; the last candidate. The loop carries both in `Function` variables.
fn search_rest(
    fb: &mut FunctionBuilder<'_>,
    set: Value,
    guest: Value,
    base: Value,
    len: Value,
) -> BResult<Value> {
    let u64_ = u64_ty(fb)?;
    let at = fb.local_var(u64_, None)?;
    let at = fb.var_ptr(at)?;
    let left = fb.local_var(u64_, None)?;
    let left = fb.var_ptr(left)?;
    fb.store(at, base, MemAccess::default())?;
    fb.store(left, len, MemAccess::default())?;
    let body: LoopFn<'_, ()> = Box::new(move |fb, ctx, lc| {
        let len = fb.load(left, MemAccess::default())?;
        let one = k64(fb, 1)?;
        let found = fb.binary(BinOp::ULessThanEqual, len, one)?;
        let exit: ArmFn<'_, ()> = Box::new(move |fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(ctx, found, exit, None)?;
        let base = fb.load(at, MemAccess::default())?;
        let (base, len) = search_step(fb, set, guest, base, len)?;
        fb.store(at, base, MemAccess::default())?;
        fb.store(left, len, MemAccess::default())
    });
    fb.loop_(&mut (), LoopSpec::default(), body)?;
    fb.load(at, MemAccess::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::Meter;

    fn meter() -> Meter {
        crate::pipeline::testing::meter()
    }

    /// Roots connected through shared root sets form one component; the component of `Dynamic`
    /// is wide, others are not, and null joins nothing.
    #[test]
    fn the_component_of_the_dynamic_root_is_wide() {
        let sets: Vec<Vec<Root>> = vec![
            vec![Root::Param(0), Root::Param(1)],
            vec![Root::Param(1), Root::Null],
            vec![Root::Dynamic, Root::Null, Root::Param(2)],
            vec![Root::Param(2), Root::Param(3)],
            vec![Root::Param(4)],
            vec![Root::Null, Root::Param(5)],
        ];
        let wide = wide_roots(sets.iter().map(|s| s.as_slice()), &mut meter()).expect("wide");
        let want: BTreeSet<Root> = [Root::Dynamic, Root::Param(2), Root::Param(3)].into();
        assert_eq!(wide, want);
    }

    #[test]
    fn without_a_dynamic_root_nothing_is_wide() {
        let sets: Vec<Vec<Root>> = vec![vec![Root::Param(0), Root::Param(1)], vec![Root::Null]];
        let wide = wide_roots(sets.iter().map(|s| s.as_slice()), &mut meter()).expect("wide");
        assert!(wide.is_empty());
    }

    /// Joining in any order links every set: a later set bridging two components pulls the
    /// dynamic one's roots into the other.
    #[test]
    fn a_bridging_set_joins_two_components() {
        let sets: Vec<Vec<Root>> = vec![
            vec![Root::Param(0), Root::Param(1)],
            vec![Root::Param(2), Root::Dynamic],
            vec![Root::Param(1), Root::Param(2)],
        ];
        let wide = wide_roots(sets.iter().map(|s| s.as_slice()), &mut meter()).expect("wide");
        assert_eq!(wide.len(), 4);
    }

    /// `n` parameter roots joined pairwise from the last pair to the first, `Dynamic` with the
    /// last: unioned without ranks, each pair hangs the component built so far under its first
    /// root, a chain `n` deep, and every later find walks it.
    fn reversed_chain(n: u32) -> Vec<Vec<Root>> {
        let mut sets = vec![vec![Root::Param(n), Root::Dynamic]];
        sets.extend(
            (0..n)
                .rev()
                .map(|i| vec![Root::Param(i), Root::Param(i + 1)]),
        );
        sets
    }

    /// The union-find is joined by size with path halving, so a chain of sets costs work
    /// near-linear in its roots, and every step is a work unit: 2^16 pairs take under 8 units a
    /// root, and twice as many roots under 2.1 times the units. Unioned without ranks or
    /// compression (the earlier code), the 2^16-pair chain is 2^16 deep and the final finds walk
    /// about 2^31 steps; it gave no result within 60 s in a test build.
    #[test]
    fn a_chain_of_root_sets_is_joined_in_near_linear_work() {
        let units = |n: u32| {
            let sets = reversed_chain(n);
            let mut m = meter();
            let wide = wide_roots(sets.iter().map(|s| s.as_slice()), &mut m).expect("wide");
            assert_eq!(
                wide.len(),
                n as usize + 2,
                "every root is in the dynamic component"
            );
            assert_eq!(m.ledger(), 0, "the union-find returns its tables");
            m.work()
        };
        let (small, large) = (units(1 << 16), units(1 << 17));
        assert!(small < 8 << 16, "{small} units for 2^16 pairs");
        assert!(
            large * 10 < small * 21,
            "{large} units for 2^17 pairs, {small} for 2^16"
        );
    }

    /// Every root read and every find step is a work unit: a cap below the chain's work stops it
    /// with `LimitExceeded(WorkUnits)`. The code previously ticked once per set and per root of the
    /// final pass (2^12 + 1 and 2^12 + 2 here), so this cap admitted it.
    #[test]
    fn the_union_find_is_stopped_by_the_work_cap() {
        let n = 1 << 12;
        let sets = reversed_chain(n);
        let budget = crate::budget::Budget::default();
        let mut m = Meter::with_caps(&budget, 3 * u64::from(n), crate::budget::LEDGER_BYTES);
        assert_eq!(
            wide_roots(sets.iter().map(|s| s.as_slice()), &mut m),
            Err(Error::LimitExceeded(crate::error::Limit::WorkUnits))
        );
    }
}
