//! `MemoryPlan`: slots, arenas, typed allocas, constant globals, the blob.
//!
//! [`analyze_memory`] runs the provenance analysis ([`super::provenance`]) over the specialized
//! entry and turns its facts into the plan every memory leaf reads:
//!
//! - **Representations**: the [`Repr`] of every pointer value, which `select` demotes it to. A
//!   buffer pointer is wide exactly when one of its roots is in the root component of `Dynamic`
//!   (`super::roots::wide_roots`), so a pointer and its operands share one width. On a profile with
//!   device addresses (Profile A), a buffer slot root whose pointer escapes (stored as a value, or
//!   `ptrtoint`) joins that component: its address may come back as a loaded pointer, and a wide
//!   pointer's guest address is exact. On Profile F it stays static, and its stored or converted
//!   address is `SlotGuestBase` plus its displacement. Such a slot's window is `Whole` when its own
//!   accesses give none.
//! - **Slots**: every buffer or argument-buffer parameter an instruction reaches, and every
//!   accessed element of a buffer table (an `array_ref` of buffers), with its access set, its
//!   proven [`Window`], and whether it needs its guest base (`PushField::SlotGuestBase`:
//!   `ptrtoint`, a stored pointer, or a cross-root compare). The slot's form and view kind are
//!   `select`'s; the plan only states the facts.
//! - **Threadgroup arena**: static threadgroup globals first, each at its own alignment, then the
//!   `[[threadgroup(n)]]` parameter regions, whose lengths are specialization inputs. The static
//!   size is exactly the sum of the allocations plus alignment padding: no sink word.
//! - **Typed threadgroup objects**: a kernel's static threadgroup global whose every access is a
//!   plain load or store of exactly the subobject at its offset shape, and whose pointers merge
//!   with no other root, escape or are stored, is a `Workgroup` variable of its own type out of the
//!   arena, its leaves in their held forms, where the kernel has no regions or imageblock, no
//!   threadgroup pointer of unknown root, and the held objects fit the device's limit
//!   (`Planner::typed_threadgroup`).
//! - **Private arena and typed allocas**: an alloca whose every access is a plain load or store of
//!   exactly the subobject at its offset shape, whose pointers never merge with another root or
//!   escape, and whose type holds no pointer, is typed, while the function's typed objects stay
//!   within `alloca::MAX_UNSHARED`; every other alloca has a place in the private arena. A
//!   pointer-holding alloca is a pointer cell: a closed one holds each stored pointer in its
//!   lowered representation, an open one a buffer pointer as its 8-byte guest address.
//! - **Constant globals**: a closed, type-consistent constant is a `Private` variable with its
//!   initializer, unless its type would take the function's typed objects past
//!   `alloca::MAX_UNSHARED`; any other one is placed in the **constant blob**, the byte image the
//!   artifact carries (`WriteSource::ConstantBlob`).
//! - **Needs**: dynamic roots, the handle classes that need the handle table, and whether a guest
//!   base is needed.
//!
//! Profile F (`BufferAccess::Floor`) has no device addresses, so a dereference of a pointer whose
//! root set contains `Dynamic` is refused here, naming what the profile lacks of Profile A's
//! `bufferDeviceAddress` and `shaderInt64`. Every other refusal of a represented construct belongs
//! to the family that lowers it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use super::abi_index::AbiIndex;
use super::provenance::{
    self, grow, imageblock_cell, CellFacts, Class, ConstForm, Extent, HandleClass, OffsetRange,
    PrivForm, Provenance, Repr, Root, RootSetId, RootSets, RootUse, Shape,
};
use super::window;
use crate::air::abi::{AbiInt, Access, ImageblockKind, ParamRole};
use crate::air::ir::{
    AddrSpace, Const, ConstId, GlobalId, InstId, InstKind, Type, TypeId, TypeLayouts, Value,
    ValueId, ValueIndex,
};
use crate::air::Entry;
use crate::budget::{tree_bytes, vec_bytes, Meter};
use crate::device::profile::BufferAccess;
use crate::device::DeviceProfile;
use crate::error::{Error, Feature, InternalError, InternalKind, Limit, LowerFamily, Unsupported};
use crate::pipeline::interface::{AccessSet, Window};
use crate::pipeline::sym::{SlotClass, SlotRef};

/// One buffer slot the entry reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotPlan {
    pub slot: SlotRef,
    /// The entry parameter bound to the slot.
    pub param: u32,
    /// For an element of a buffer-table parameter, the element: the slot is `first + element` of
    /// the table's argument-table range. `None` for a buffer parameter.
    pub element: Option<u32>,
    pub space: AddrSpace,
    /// The ABI's declared access.
    pub declared: Access,
    /// `air.arg_type_size`, when stated.
    pub arg_size: Option<u32>,
    /// `air.buffer_size` (a sized pointee), when stated.
    pub buffer_size: Option<u32>,
    /// An argument buffer (`air.indirect_buffer`).
    pub argument_buffer: bool,
    /// What the function's accesses do (an atomic reads and writes).
    pub access: AccessSet,
    /// What the function's accesses through a dynamic root may do to the slot: on a profile with
    /// device addresses, a slot whose address escapes may come back as a loaded or converted
    /// pointer, which every access of the dynamic root may be. It writes where some such access
    /// writes, and reads where some reads or writes: a store through a device address merges its
    /// partial words with two atomics, which read the word they modify, and whether one does is
    /// known only at its emission. Empty for a slot whose address does not escape. The stage
    /// interface reports it beside [`SlotPlan::access`]; lowering reads only `access`.
    pub wide: AccessSet,
    /// Some access is an atomic read-modify-write.
    pub atomic: bool,
    /// The proven window; `None` when the slot is never accessed.
    pub window: Option<Window>,
    /// The slot's guest base address is observed.
    pub guest_base: bool,
}

/// An object in an arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ArenaObject {
    Global(GlobalId),
    Alloca(InstId),
}

/// One statically sized object of an arena, at its byte offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ArenaEntry {
    pub object: ArenaObject,
    pub offset: u64,
    pub size: u64,
    pub align: u64,
}

/// A `[[threadgroup(n)]]` parameter region: its length is the pipeline's threadgroup memory
/// length for `index`, a specialization input, so it follows the static entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DynamicRegion {
    pub param: u32,
    pub index: u32,
    pub align: u64,
}

/// A kernel's explicit imageblock: `width · height` cells `cell_bytes` apart (the larger of
/// `air.imageblock_data_size` and the widest access through a cell pointer, rounded up to
/// `air.arg_type_align_size`, so every cell is as aligned as the first), the extent the compute
/// encoder sets, so it follows the `[[threadgroup(n)]]` regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImageblockRegion {
    pub param: u32,
    pub cell_bytes: u32,
}

/// The layout of one arena: exact sizes, no sink word.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArenaLayout {
    /// The static objects, in object order whatever order they were placed in.
    pub entries: Vec<ArenaEntry>,
    /// The end of the static objects (that of the last one placed).
    pub static_size: u64,
    /// The static objects lie where Metal places them (the threadgroup arena's): an integer address
    /// names a byte of them. Metal's placement of an arena's objects is otherwise unobservable.
    pub metal_layout: bool,
    pub dynamic: Vec<DynamicRegion>,
    /// The threadgroup arena's imageblock, last.
    pub imageblock: Option<ImageblockRegion>,
}

impl ArenaLayout {
    /// Place `object` of `size` bytes at the next offset aligned to `align`. The private arena
    /// places its objects in object order (instruction order), each strictly after the last
    /// placed, so [`ArenaLayout::entry`] is a search; placing one at or before the last is an
    /// invariant violation.
    fn place(
        &mut self,
        object: ArenaObject,
        size: u64,
        align: u64,
        m: &mut Meter,
    ) -> Result<u64, Error> {
        if self.entries.last().is_some_and(|e| e.object >= object) {
            return Err(invariant("an arena object placed out of order"));
        }
        self.push(object, size, align, m)
    }

    /// Place `object` of `size` bytes at the next offset aligned to `align`, in any order.
    fn push(
        &mut self,
        object: ArenaObject,
        size: u64,
        align: u64,
        m: &mut Meter,
    ) -> Result<u64, Error> {
        let offset = round_up(self.static_size, align).ok_or_else(too_large)?;
        let end = offset.checked_add(size).ok_or_else(too_large)?;
        grow(&mut self.entries, 1, m)?;
        self.static_size = end;
        self.entries.push(ArenaEntry {
            object,
            offset,
            size,
            align,
        });
        Ok(offset)
    }

    /// The arena of `objects` (object, size, alignment) placed in their order, each at the next
    /// offset aligned to its alignment from 0 (the threadgroup arena's, in Metal's order), its
    /// entries then sorted into object order so [`ArenaLayout::entry`] is a search (one sort; the
    /// offsets are unchanged). An object placed twice is an invariant violation.
    fn in_placement_order(
        objects: &[(ArenaObject, u64, u64)],
        m: &mut Meter,
    ) -> Result<ArenaLayout, Error> {
        let mut arena = ArenaLayout::default();
        for &(object, size, align) in objects {
            m.tick(1)?;
            arena.push(object, size, align, m)?;
        }
        arena.entries.sort_unstable_by_key(|e| e.object);
        if arena
            .entries
            .windows(2)
            .any(|w| matches!(w, [a, b] if a.object == b.object))
        {
            return Err(invariant("an arena object placed twice"));
        }
        Ok(arena)
    }

    /// The entry of `object`: a binary search, the entries being in object order
    /// ([`ArenaLayout::place`]).
    pub fn entry(&self, object: ArenaObject) -> Option<&ArenaEntry> {
        let at = self
            .entries
            .binary_search_by_key(&object, |e| e.object)
            .ok()?;
        self.entries.get(at)
    }
}

/// The least alignment of Metal's static threadgroup objects: a `uchar[3]` after a `ushort[5]`
/// at 0 lies at 12 (measured on an M3 Max and an M4).
const METAL_TG_ALIGN: u64 = 4;

/// The static threadgroup objects a constant or an instruction names ([`Planner::metal_order`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Named {
    None,
    One(GlobalId),
    /// Two or more.
    Many,
}

impl Named {
    /// What `self` and `other` name together.
    fn and(self, other: Named) -> Named {
        match (self, other) {
            (Named::None, x) | (x, Named::None) => x,
            (Named::One(a), Named::One(b)) if a == b => Named::One(a),
            _ => Named::Many,
        }
    }
}

/// A typed allocation: a `Function` variable of `count` elements of `ty`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AllocaPlan {
    pub inst: InstId,
    pub ty: TypeId,
    pub count: u64,
    /// Its simdgroup matrices hold only fragments, and it holds each as its lane's two elements
    /// (`alloca::TypedNode::Fragment`).
    pub fragments: bool,
}

/// A typed threadgroup object: a static threadgroup global held as a `Workgroup` variable of its
/// own type (`memory::alloca`'s typed shape, each leaf in its held form), out of the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SharedPlan {
    pub global: GlobalId,
    pub ty: TypeId,
    /// The variable's bytes of workgroup memory, an upper bound
    /// ([`super::alloca::TypedTy::held_bytes`] rounded up to 16).
    pub bytes: u64,
}

/// Where a program-scope constant lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstGlobalForm {
    /// A `Private` variable with its constant initializer, accessed by typed access chains.
    Private,
    /// At `offset` in the constant blob.
    Blob { offset: u64 },
}

/// One program-scope constant the entry reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConstGlobalPlan {
    pub global: GlobalId,
    pub form: ConstGlobalForm,
}

/// The constant blob: the initializers of every blob-form constant at their offsets. The bytes are
/// allocated once at their final length and shared, by selection and the artifact, without a copy.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConstantBlob {
    pub bytes: Arc<[u8]>,
}

/// What the function needs from the binder and the device.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemNeeds {
    /// Some buffer pointer may have a dynamic root (a resident-set lookup).
    pub dynamic_roots: bool,
    /// The classes of handles loaded from memory, which need the handle table.
    pub handle_table: Option<Vec<HandleClass>>,
}

/// The memory plan of one entry.
#[derive(Clone, Debug)]
pub struct MemoryPlan {
    pub values: ValueIndex,
    /// The datalayout of every AIR type (GEP strides, store sizes, and struct field offsets for
    /// the access leaves).
    pub layouts: TypeLayouts,
    /// By `ValueId`; `Some` exactly for pointer values (arguments, instructions, and the pointer
    /// constants instructions use).
    pub repr: Vec<Option<Repr>>,
    /// The interned root sets `repr` names.
    pub sets: RootSets,
    /// The offset range of every pointer value, sorted by value.
    pub offsets: Vec<(ValueId, OffsetRange)>,
    /// The device and constant pointer values whose offset is a constant plus one dynamic
    /// multiple of a stride (`Shape::Strided`: one dynamic element index), with that base and
    /// stride, sorted by value.
    pub strided: Vec<(ValueId, i64, u64)>,
    /// The root set of every pointer value, sorted by value (a representation names it only for
    /// buffer pointers and handles).
    pub root_sets: Vec<(ValueId, RootSetId)>,
    pub slots: Vec<SlotPlan>,
    /// Each slot's root (`Param` for a buffer parameter, `BufferElement` for an element of a
    /// buffer table) with its index in `slots`, sorted by root, so a root's slot is one search
    /// ([`MemoryPlan::slot_of_root`]).
    pub slot_roots: Vec<(Root, u32)>,
    pub tg_arena: ArenaLayout,
    /// The static threadgroup globals held as typed `Workgroup` objects, in global order, out of
    /// `tg_arena` ([`Planner::typed_threadgroup`]).
    pub typed_shared: Vec<SharedPlan>,
    /// The static part of Metal's own threadgroup layout, every static object at its place
    /// (Metal's `staticThreadgroupMemoryLength`): `tg_arena`'s own static part, or with typed
    /// objects the layout's that holds them too.
    pub tg_metal_static: u64,
    pub priv_arena: ArenaLayout,
    pub typed_allocas: Vec<AllocaPlan>,
    /// The bytes of private memory a thread holds where a driver backs it with scratch memory,
    /// which selection holds to the device's Function storage: the private arena (its pointers
    /// merge, escape, or reinterpret its bytes), and each typed alloca (a recursion's call stacks
    /// among them) and each constant held as a `Private` variable that some access indexes at run
    /// time, each at its allocation size. An object reached only at constant offsets counts
    /// nothing: a driver holds its parts in registers. Nothing is shared between objects whose
    /// lives do not overlap, so the sum is conservative.
    pub private_bytes: u64,
    pub const_globals: Vec<ConstGlobalPlan>,
    /// Each constant global with its index in `const_globals`, sorted by global
    /// ([`MemoryPlan::const_global`]).
    pub const_index: Vec<(GlobalId, u32)>,
    pub blob: Option<ConstantBlob>,
    pub needs: MemNeeds,
    /// Which allocas forward the pointers stored into them (pointer cells).
    pub cells: CellFacts,
    /// The entry's bindings by parameter and layouts' members by offset, indexed once.
    pub abi: AbiIndex,
    /// The `<64 x T>` instructions that are simdgroup matrices' fragments, sorted
    /// (`memory::fragment`: elements 2..63 zero or undefined; [`MemoryPlan::is_fragment`]).
    pub fragment_values: Vec<InstId>,
}

impl MemoryPlan {
    /// Whether `<64 x T>` instruction `id` is a simdgroup matrix's fragment: its elements 2..63
    /// are zero or undefined (`memory::fragment`).
    pub fn is_fragment(&self, id: InstId) -> bool {
        self.fragment_values.binary_search(&id).is_ok()
    }

    /// The representation of `v`, if it is a pointer value.
    pub fn repr(&self, v: Value) -> Option<Repr> {
        let id = self.values.id(v)?;
        self.repr.get(id.index()).copied().flatten()
    }

    /// The roots of a root set.
    pub fn roots(&self, id: RootSetId) -> &[Root] {
        self.sets.get(id)
    }

    /// The offset range of pointer value `v` from its root.
    pub fn offset(&self, v: Value) -> Option<OffsetRange> {
        let id = self.values.id(v)?;
        let at = self.offsets.binary_search_by_key(&id, |(v, _)| *v).ok()?;
        self.offsets.get(at).map(|(_, r)| *r)
    }

    /// `(base, stride)` when the offset of device or constant pointer value `v` is `base + k ·
    /// stride` for one dynamic element index `k` (`Shape::Strided`); `None` for any other shape.
    pub fn strided(&self, v: Value) -> Option<(i64, u64)> {
        let id = self.values.id(v)?;
        let at = self
            .strided
            .binary_search_by_key(&id, |(v, _, _)| *v)
            .ok()?;
        self.strided
            .get(at)
            .map(|(_, base, stride)| (*base, *stride))
    }

    /// The roots pointer value `v` may point into.
    pub fn roots_of(&self, v: Value) -> Option<&[Root]> {
        let id = self.values.id(v)?;
        let at = self.root_sets.binary_search_by_key(&id, |(v, _)| *v).ok()?;
        self.root_sets.get(at).map(|(_, r)| self.sets.get(*r))
    }

    /// The slot index of buffer parameter `param` (a buffer table has one slot per accessed
    /// element instead; see [`MemoryPlan::slot_of_root`]).
    pub fn slot_of_param(&self, param: u32) -> Option<usize> {
        self.slot_of_root(Root::Param(param))
    }

    /// The slot index of buffer root `r`: a buffer parameter, or an element of a buffer table.
    /// `None` for every other root. A search of [`MemoryPlan::slot_roots`].
    pub fn slot_of_root(&self, r: Root) -> Option<usize> {
        if !matches!(r, Root::Param(_) | Root::BufferElement { .. }) {
            return None;
        }
        let at = self.slot_roots.binary_search_by_key(&r, |(k, _)| *k).ok()?;
        self.slot_roots.get(at).map(|(_, i)| *i as usize)
    }

    /// The plan of constant global `g`, when an instruction reaches it: a search of
    /// [`MemoryPlan::const_index`].
    pub fn const_global(&self, g: GlobalId) -> Option<&ConstGlobalPlan> {
        let at = self
            .const_index
            .binary_search_by_key(&g, |(k, _)| *k)
            .ok()?;
        let (_, i) = self.const_index.get(at)?;
        self.const_globals.get(*i as usize)
    }

    /// Whether the entry declares static threadgroup objects, in the arena or typed.
    pub fn has_static_threadgroup_objects(&self) -> bool {
        !self.tg_arena.entries.is_empty() || !self.typed_shared.is_empty()
    }

    /// The bytes of workgroup memory the typed threadgroup objects take ([`SharedPlan::bytes`]).
    pub fn typed_shared_bytes(&self) -> u64 {
        self.typed_shared
            .iter()
            .fold(0u64, |sum, s| sum.saturating_add(s.bytes))
    }

    /// The bytes the plan's tables hold, as the ledger charges them.
    pub fn retained_bytes(&self) -> usize {
        self.repr.capacity() * std::mem::size_of::<Option<Repr>>()
            + self.offsets.capacity() * std::mem::size_of::<(ValueId, OffsetRange)>()
            + self.strided.capacity() * std::mem::size_of::<(ValueId, i64, u64)>()
            + self.root_sets.capacity() * std::mem::size_of::<(ValueId, RootSetId)>()
            + self.sets.retained_bytes()
            + self.slots.capacity() * std::mem::size_of::<SlotPlan>()
            + self.slot_roots.capacity() * std::mem::size_of::<(Root, u32)>()
            + self.const_index.capacity() * std::mem::size_of::<(GlobalId, u32)>()
            + (self.tg_arena.entries.capacity() + self.priv_arena.entries.capacity())
                * std::mem::size_of::<ArenaEntry>()
            + self.tg_arena.dynamic.capacity() * std::mem::size_of::<DynamicRegion>()
            + self.typed_allocas.capacity() * std::mem::size_of::<AllocaPlan>()
            + self.typed_shared.capacity() * std::mem::size_of::<SharedPlan>()
            + self.const_globals.capacity() * std::mem::size_of::<ConstGlobalPlan>()
            + self.blob.as_ref().map_or(0, |b| b.bytes.len())
            + self.layouts.retained_bytes()
            + self.cells.retained_bytes()
            + self.abi.retained_bytes()
    }

    /// `(slot index, exact byte offset)` when `v` is a buffer pointer whose only root is one
    /// slot, at a constant offset.
    #[cfg(test)]
    pub fn static_slot(&self, v: Value) -> Option<(usize, i64)> {
        let Some(Repr::BufPtr { roots, .. }) = self.repr(v) else {
            return None;
        };
        let [r] = self.roots(roots) else {
            return None;
        };
        Some((self.slot_of_root(*r)?, self.offset(v)?.exact()?))
    }
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

fn not_lowered() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::Access)).into()
}

fn too_large() -> Error {
    Unsupported::not_representable(Feature::ThreadgroupMemorySize).into()
}

/// The class of a handle-array parameter when `roots` is exactly that parameter.
fn handle_array_root(abi: &AbiIndex, roots: &[Root]) -> Option<HandleClass> {
    match roots {
        [Root::Param(p)] => abi.handle_array(*p),
        _ => None,
    }
}

/// Each slot's root with its index ([`MemoryPlan::slot_roots`]), sorted by root: of equal roots
/// the first slot is kept, which a scan of the slots would find. One work unit per slot.
fn slot_roots(slots: &[SlotPlan], m: &mut Meter) -> Result<Vec<(Root, u32)>, Error> {
    let mut out = Vec::new();
    m.reserve(&mut out, slots.len())?;
    for (i, s) in slots.iter().enumerate() {
        m.tick(1)?;
        let root = match s.element {
            None => Root::Param(s.param),
            Some(element) => Root::BufferElement {
                param: s.param,
                element,
            },
        };
        let i = u32::try_from(i).map_err(|_| invariant("more slots than u32"))?;
        out.push((root, i));
    }
    out.sort_by_key(|(r, _)| *r);
    out.dedup_by_key(|(r, _)| *r);
    Ok(out)
}

/// Each constant global with its index ([`MemoryPlan::const_index`]), sorted by global: of equal
/// globals the first is kept. One work unit per constant.
fn const_index(
    const_globals: &[ConstGlobalPlan],
    m: &mut Meter,
) -> Result<Vec<(GlobalId, u32)>, Error> {
    let mut out = Vec::new();
    m.reserve(&mut out, const_globals.len())?;
    for (i, c) in const_globals.iter().enumerate() {
        m.tick(1)?;
        let i = u32::try_from(i).map_err(|_| invariant("more constants than u32"))?;
        out.push((c.global, i));
    }
    out.sort_by_key(|(g, _)| *g);
    out.dedup_by_key(|(g, _)| *g);
    Ok(out)
}

/// Whether sorted root set `roots` holds the dynamic root.
fn has_dynamic(roots: &[Root]) -> bool {
    roots.binary_search(&Root::Dynamic).is_ok()
}

/// The parameter of a buffer table when `roots` is exactly that table.
fn buffer_table_root(abi: &AbiIndex, roots: &[Root]) -> Option<u32> {
    match roots {
        [Root::Param(p)] if abi.buffer_table(*p).is_some() => Some(*p),
        _ => None,
    }
}

/// The window of a slot whose accesses give `window`, given its uses `u`: a slot whose address
/// escapes is reached by the pointers recovered from it, which may reach any byte, so it is `Whole`
/// when its own accesses give none (its root entry carries its guest address).
fn escaped_window(window: Option<Window>, u: &RootUse) -> Option<Window> {
    match window {
        None if u.escapes => Some(Window::Whole),
        w => w,
    }
}

fn round_up(value: u64, align: u64) -> Option<u64> {
    let align = align.max(1);
    value.checked_next_multiple_of(align)
}

/// Analyze the memory of `e` for profile `p`.
///
/// Every table the analysis builds is charged to the ledger before it is reserved. The working
/// tables die with the analysis, so on return the ledger holds exactly the plan's
/// [`MemoryPlan::retained_bytes`], and after a refusal nothing.
pub fn analyze_memory(e: &Entry, p: &DeviceProfile, m: &mut Meter) -> Result<MemoryPlan, Error> {
    let before = m.ledger();
    let result = analyze(e, p, m);
    let spent = usize::try_from(m.ledger().saturating_sub(before)).unwrap_or(usize::MAX);
    m.release(spent);
    let plan = result?;
    m.charge(plan.retained_bytes())?;
    Ok(plan)
}

fn analyze(e: &Entry, p: &DeviceProfile, m: &mut Meter) -> Result<MemoryPlan, Error> {
    let layouts = TypeLayouts::new(&e.layout, e.module.types(), m)?;
    let prov = provenance::analyze(e, &layouts, m)?;
    if let (Some((b, id)), BufferAccess::Floor) = (prov.dynamic_deref, p.fields().buffer_access) {
        let err: Error =
            Unsupported::needs_any_of(Feature::DereferenceLoadedPointer, vec![p.addressed_needs()])
                .into();
        return Err(provenance::located(err, e, b, id));
    }
    let addressed = p.fields().buffer_access == BufferAccess::Addressed;
    Planner {
        e,
        layouts,
        prov,
        addressed,
        p,
    }
    .plan(m)
}

/// What [`Planner::reprs`] derives.
struct Reprs {
    repr: Vec<Option<Repr>>,
    offsets: Vec<(ValueId, OffsetRange)>,
    strided: Vec<(ValueId, i64, u64)>,
    root_sets: Vec<(ValueId, RootSetId)>,
    handle_table: Option<Vec<HandleClass>>,
}

/// How a typed object would hold a type ([`Planner::fit`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fit {
    /// Typed, within the function's room.
    Typed,
    /// Typed, but past the room: the object is held untyped.
    OverRoom,
    /// No typed object holds the type.
    Untyped,
}

struct Planner<'a> {
    e: &'a Entry,
    /// Moved into the plan: the access leaves read GEP strides and store sizes from it.
    layouts: TypeLayouts,
    prov: Provenance,
    /// The profile reaches memory by device address (Profile A): escaping roots are wide.
    addressed: bool,
    /// The profile, whose held forms decide which allocas a typed object can hold.
    p: &'a DeviceProfile,
}

/// Where an aggregate constant's parts lie in its bytes ([`Planner::parts_at`]).
#[derive(Clone, Copy)]
enum PartsAt<'a> {
    /// A struct's fields, at their offsets.
    Fields(&'a [TypeId], &'a [u64]),
    /// An array's elements or a vector's lanes, of one type, this many bytes apart.
    Strided(TypeId, u64),
}

impl PartsAt<'_> {
    /// Part `i`'s type and byte offset.
    fn part(self, i: usize) -> Option<(TypeId, u64)> {
        match self {
            PartsAt::Fields(fields, offs) => Some((*fields.get(i)?, *offs.get(i)?)),
            PartsAt::Strided(elem, stride) => {
                Some((elem, u64::try_from(i).ok()?.checked_mul(stride)?))
            }
        }
    }
}

impl Planner<'_> {
    fn uses(&self, r: Root) -> RootUse {
        self.prov.uses.get(&r).copied().unwrap_or_default()
    }

    /// What the accesses through the dynamic root may do to a buffer slot whose root has uses
    /// `u` ([`SlotPlan::wide`]): on a profile with device addresses, an escaping slot is in the
    /// dynamic root's reach, so it is written where the dynamic root is written, and read where it
    /// is read or written (a store through a device address may merge).
    fn wide_access(&self, u: &RootUse) -> AccessSet {
        if !(self.addressed && u.escapes) {
            return AccessSet::UNUSED;
        }
        let d = self.uses(Root::Dynamic);
        AccessSet {
            read: d.read || d.write,
            write: d.write,
        }
    }

    fn plan(self, m: &mut Meter) -> Result<MemoryPlan, Error> {
        self.check_tables()?;
        let dynamic = self.any_dynamic_buffer_root();
        let slots = self.slots(dynamic, m)?;
        // The unshared type size the function's typed objects may still declare.
        let mut room = super::alloca::MAX_UNSHARED;
        let (const_globals, blob) = self.const_globals(&mut room, m)?;
        let tg_arena = self.threadgroup_arena(&[], m)?;
        let (priv_arena, typed_allocas, fragment_values) = self.allocas(&mut room, m)?;
        let tg_metal_static = tg_arena.static_size;
        let (tg_arena, typed_shared) = self.typed_threadgroup(tg_arena, &mut room, m)?;
        let private_bytes = self.private_bytes(&priv_arena, &typed_allocas, &const_globals, m)?;
        let Reprs {
            repr,
            offsets,
            strided,
            root_sets,
            handle_table,
        } = self.reprs(&const_globals, &typed_allocas, &typed_shared, dynamic, m)?;
        let slot_roots = slot_roots(&slots, m)?;
        let const_index = const_index(&const_globals, m)?;
        let Planner { prov, layouts, .. } = self;
        Ok(MemoryPlan {
            values: prov.values,
            layouts,
            repr,
            sets: prov.sets.into_table(),
            offsets,
            strided,
            root_sets,
            slots,
            slot_roots,
            tg_arena,
            typed_shared,
            tg_metal_static,
            priv_arena,
            typed_allocas,
            private_bytes,
            const_globals,
            const_index,
            blob,
            needs: MemNeeds {
                dynamic_roots: dynamic,
                handle_table,
            },
            cells: prov.cells,
            abi: prov.abi,
            fragment_values,
        })
    }

    /// The bytes of private memory a thread holds where a driver backs it with scratch memory
    /// ([`MemoryPlan::private_bytes`]): the private arena's size, and the allocation size of each
    /// typed alloca and each `Private` constant some access indexes at run time. An alloca that
    /// holds simdgroup matrices as fragments holds each as its two elements
    /// ([`super::fragment::bytes`]).
    fn private_bytes(
        &self,
        arena: &ArenaLayout,
        typed: &[AllocaPlan],
        consts: &[ConstGlobalPlan],
        m: &mut Meter,
    ) -> Result<u64, Error> {
        let mut bytes = arena.static_size;
        for a in typed
            .iter()
            .filter(|a| self.uses(Root::Alloca(a.inst)).indexed)
        {
            let size = if a.fragments {
                super::fragment::bytes(self.e.module.types(), &self.layouts, a.ty, m)?
            } else {
                self.layouts.alloc_size(a.ty)
            };
            let size = size.ok_or_else(not_lowered)?;
            bytes = bytes.saturating_add(size.saturating_mul(a.count));
        }
        let private = |c: &&ConstGlobalPlan| c.form == ConstGlobalForm::Private;
        let indexed = |c: &&ConstGlobalPlan| self.uses(Root::Global(c.global)).indexed;
        for c in consts.iter().filter(private).filter(indexed) {
            let global = self
                .e
                .module
                .global(c.global)
                .ok_or_else(|| invariant("a constant global out of range"))?;
            let size = self
                .layouts
                .alloc_size(global.value_ty)
                .ok_or_else(not_lowered)?;
            bytes = bytes.saturating_add(size);
        }
        Ok(bytes)
    }

    /// The root set of every buffer (device or constant space, not handle class) pointer.
    fn buffer_root_sets(&self) -> impl Iterator<Item = RootSetId> + '_ {
        let types = self.e.module.types();
        self.prov
            .facts
            .iter()
            .zip(&self.prov.classes)
            .filter(move |((v, _), c)| {
                let space = self
                    .prov
                    .values
                    .value(*v)
                    .and_then(|v| self.e.module.value_ty(self.e.func, v))
                    .and_then(|t| match types.get(t) {
                        Some(Type::Ptr { space }) => Some(*space),
                        _ => None,
                    });
                !matches!(c, Class::Handle(_))
                    && matches!(space, Some(AddrSpace::Device | AddrSpace::Constant))
            })
            .map(|((_, f), _)| f.roots)
    }

    /// The distinct root sets of the buffer pointers ([`Planner::buffer_root_sets`]), each once,
    /// in the order first seen: one work unit per pointer, the record of the sets seen charged to
    /// the ledger while it lives. The vector's charge is added to `charged`, which the caller
    /// returns when it drops it.
    fn distinct_buffer_root_sets(
        &self,
        charged: &mut usize,
        m: &mut Meter,
    ) -> Result<Vec<RootSetId>, Error> {
        let n = self.prov.sets.len();
        m.charge(n)?;
        let mut seen = vec![false; n];
        let mut out = Vec::new();
        let result = (|| {
            for id in self.buffer_root_sets() {
                m.tick(1)?;
                if let Some(s) = seen.get_mut(id.index()) {
                    if !*s {
                        *s = true;
                        provenance::push_working(&mut out, id, charged, m)?;
                    }
                }
            }
            Ok(())
        })();
        m.release(n);
        result.map(|()| out)
    }

    /// Every buffer slot root (a parameter or a buffer-table element) whose pointer escapes,
    /// paired with the dynamic root: the sets that join it to the dynamic component, on a profile
    /// with device addresses only (Profile F has no root entries; its escaping roots stay static).
    /// A constant global's address has no guest address to escape as (the blob is no resident
    /// allocation), so it stays out and its stored pointer is refused. A working table charged to
    /// the ledger.
    fn escaping_roots(&self, m: &mut Meter) -> Result<Vec<[Root; 2]>, Error> {
        let mut out = Vec::new();
        if !self.addressed {
            return Ok(out);
        }
        for (r, u) in &self.prov.uses {
            if u.escapes && matches!(r, Root::Param(_) | Root::BufferElement { .. }) {
                grow(&mut out, 1, m)?;
                out.push([*r, Root::Dynamic]);
            }
        }
        Ok(out)
    }

    /// Whether some buffer (device or constant space, byte class) pointer has a dynamic root
    /// (a set is sorted, so each is one search).
    fn any_dynamic_buffer_root(&self) -> bool {
        self.buffer_root_sets()
            .any(|id| has_dynamic(self.prov.sets.get(id)))
    }

    fn slots(&self, dynamic: bool, m: &mut Meter) -> Result<Vec<SlotPlan>, Error> {
        let stage = super::static_root::stage_of(self.e);
        let mut slots = Vec::new();
        for b in &self.e.ep.params {
            let (arg, argument_buffer) = match &b.role {
                ParamRole::Buffer(arg) => (arg, false),
                ParamRole::ArgumentBuffer(arg) => (arg, true),
                _ => continue,
            };
            let u = self.uses(Root::Param(b.param));
            if !u.referenced {
                continue;
            }
            if !matches!(arg.space, AddrSpace::Device | AddrSpace::Constant) {
                return Err(invariant("a buffer role outside device and constant space"));
            }
            let slot = |index: u32| SlotRef {
                stage,
                class: SlotClass::Buffer,
                index,
            };
            if self.prov.abi.buffer_table(b.param).is_some() {
                // A buffer table: one slot per element an access reaches. The table itself is only
                // read (provenance refuses every other access).
                let (AbiInt::Const(first), AbiInt::Const(_)) =
                    (arg.location.index, arg.location.count)
                else {
                    return Err(not_lowered());
                };
                let range = Root::BufferElement {
                    param: b.param,
                    element: 0,
                }..=Root::BufferElement {
                    param: b.param,
                    element: u32::MAX,
                };
                for (r, eu) in self.prov.uses.range(range) {
                    let Root::BufferElement { element, .. } = *r else {
                        return Err(invariant("a table range holds another root"));
                    };
                    if !eu.referenced {
                        continue;
                    }
                    let index = first.checked_add(element).ok_or_else(not_lowered)?;
                    grow(&mut slots, 1, m)?;
                    slots.push(SlotPlan {
                        slot: slot(index),
                        param: b.param,
                        element: Some(element),
                        space: arg.space,
                        declared: arg.access,
                        // The metadata's sizes describe the table argument, not its elements.
                        arg_size: None,
                        buffer_size: None,
                        argument_buffer,
                        access: AccessSet {
                            read: eu.read,
                            write: eu.write,
                        },
                        wide: self.wide_access(eu),
                        atomic: eu.atomic,
                        window: escaped_window(window::window(eu.extent, None, dynamic), eu),
                        guest_base: eu.escapes || eu.cross_compare,
                    });
                }
                continue;
            }
            let (AbiInt::Const(index), AbiInt::Const(1)) = (arg.location.index, arg.location.count)
            else {
                return Err(not_lowered());
            };
            grow(&mut slots, 1, m)?;
            slots.push(SlotPlan {
                slot: slot(index),
                param: b.param,
                element: None,
                space: arg.space,
                declared: arg.access,
                arg_size: b.ty.size,
                buffer_size: arg.buffer_size,
                argument_buffer,
                access: AccessSet {
                    read: u.read,
                    write: u.write,
                },
                wide: self.wide_access(&u),
                atomic: u.atomic,
                window: escaped_window(window::window(u.extent, arg.buffer_size, dynamic), &u),
                guest_base: u.escapes || u.cross_compare,
            });
        }
        Ok(slots)
    }

    /// A buffer table is a pointer to the argument table's own entries: it is only read, through
    /// loads of whole entries (provenance refuses every other access). A table pointer that
    /// escapes, merges with another root, or is compared across roots has no model (the entries are
    /// in no arena). A threadgroup role passed as a table (an array of regions) has no arena model
    /// either.
    fn check_tables(&self) -> Result<(), Error> {
        for b in &self.e.ep.params {
            let u = self.uses(Root::Param(b.param));
            if !u.referenced {
                continue;
            }
            if self.prov.abi.buffer_table(b.param).is_some()
                && (u.escapes || u.merged || u.cross_compare)
            {
                return Err(not_lowered());
            }
            if let ParamRole::Threadgroup(arg) = &b.role {
                let space = self
                    .e
                    .module
                    .func(self.e.func)
                    .and_then(|f| match self.e.module.types().get(f.ty) {
                        Some(Type::Function { params, .. }) => params.get(b.param as usize),
                        _ => None,
                    })
                    .and_then(|&t| self.e.module.types().ptr_space(t));
                if space == Some(AddrSpace::Thread) || arg.location.count != AbiInt::Const(1) {
                    return Err(not_lowered());
                }
            }
        }
        Ok(())
    }

    fn is_static_sampler(&self, g: GlobalId) -> bool {
        self.e.static_samplers.iter().any(|(s, _)| *s == g)
    }

    /// The form of every program-scope constant an instruction reaches, and the blob.
    ///
    /// The blob is shared with the artifact, so it is allocated once, at its final length, after
    /// every constant is placed. Placing checks every blob-form constant's initializer for every
    /// refusal the writer can meet but the meter's ([`Planner::blob_nodes`], over the writer's part
    /// layout, [`Planner::parts_at`]) before it charges any constant's bytes or the work of writing
    /// it, so an initializer's refusal is met before the blob's bytes are charged, whichever
    /// constant is placed first, and a function refused that way allocates no blob. The meter's
    /// refusals can fall elsewhere: the check charges a work unit per distinct part, the writing
    /// charged one per path, so a refusal of the initializer can now come before a work refusal,
    /// and the writer's own charges can still meet the ledger cap after the blob is allocated.
    fn const_globals(
        &self,
        room: &mut u64,
        m: &mut Meter,
    ) -> Result<(Vec<ConstGlobalPlan>, Option<ConstantBlob>), Error> {
        let mut out = Vec::new();
        // Each blob-form constant's initializer, type, and offset, and the blob's length.
        let mut placed: Vec<(ConstId, TypeId, usize)> = Vec::new();
        let mut len = 0usize;
        self.place_consts(room, &mut out, &mut placed, &mut len, m)?;
        if placed.is_empty() {
            return Ok((out, None));
        }
        // One allocation of the `len` bytes placing charged: the iterator's length is exact.
        let mut bytes: Arc<[u8]> = std::iter::repeat_n(0u8, len).collect();
        let blob = Arc::get_mut(&mut bytes)
            .ok_or_else(|| invariant("the constant blob is shared before it is written"))?;
        for &(init, ty, at) in &placed {
            self.serialize(init, ty, blob, at, m)?;
        }
        Ok((out, Some(ConstantBlob { bytes })))
    }

    /// Decide the form of each program-scope constant into `out`, and place each blob-form one
    /// at its offset in `placed`, `len` the blob's length so far.
    ///
    /// Every blob-form constant's initializer is checked for the writer's refusals
    /// ([`Planner::blob_nodes`]) before any constant's bytes are charged, and only then are the
    /// bytes and the work of writing each charged, in placement order. So a refused initializer
    /// is refused as itself, before the ledger or the work meter sees the blob, wherever it is
    /// placed: the order of the globals, which follows the entry's reach, does not decide whether a
    /// function reports its refusal or the meter's.
    fn place_consts(
        &self,
        room: &mut u64,
        out: &mut Vec<ConstGlobalPlan>,
        placed: &mut Vec<(ConstId, TypeId, usize)>,
        len: &mut usize,
        m: &mut Meter,
    ) -> Result<(), Error> {
        // Each placed constant's end in the blob and the nodes writing it visits; the working
        // list's bytes are released once it is charged.
        let mut costs: Vec<(usize, u64)> = Vec::new();
        let mut working = 0usize;
        let placing = (|| -> Result<(), Error> {
            for (&root, u) in &self.prov.uses {
                let Root::Global(g) = root else { continue };
                let global = self
                    .e
                    .module
                    .global(g)
                    .ok_or_else(|| invariant("a global root out of range"))?;
                match global.space {
                    AddrSpace::Threadgroup => continue,
                    AddrSpace::Constant if self.is_static_sampler(g) => continue,
                    AddrSpace::Constant => {}
                    // No corpus row has device or thread-space globals; memory has no model.
                    _ => return Err(not_lowered()),
                }
                let Some(init) = global.init else {
                    return Err(not_lowered());
                };
                // A pointer stored into a pointer cell is read back without its definitions,
                // which the typed form's access chains are read from.
                let closed = !u.merged && !u.escapes && !u.stored && !u.cross_compare && !u.write;
                // A type no typed object holds stays `Private`: selection refuses it, as before,
                // which only sends a typed object past the room to the blob.
                let form =
                    if closed && u.typed && self.fit(global.value_ty, room, m)? != Fit::OverRoom {
                        ConstGlobalForm::Private
                    } else {
                        let align = self.global_align(g, global.value_ty)?;
                        let size = self
                            .layouts
                            .alloc_size(global.value_ty)
                            .ok_or_else(not_lowered)?;
                        let offset = round_up(*len as u64, align).ok_or_else(not_lowered)?;
                        let end = offset.checked_add(size).ok_or_else(not_lowered)?;
                        let end = usize::try_from(end).map_err(|_| not_lowered())?;
                        // The initializer's refusals; its bytes and its writing's work are charged
                        // once every initializer is checked.
                        let nodes = self.blob_nodes(init, global.value_ty, m)?;
                        let at = usize::try_from(offset).map_err(|_| not_lowered())?;
                        grow(placed, 1, m)?;
                        placed.push((init, global.value_ty, at));
                        provenance::push_working(&mut costs, (end, nodes), &mut working, m)?;
                        *len = end;
                        ConstGlobalForm::Blob { offset }
                    };
                grow(out, 1, m)?;
                out.push(ConstGlobalPlan { global: g, form });
            }
            // Each constant's bytes, then the work of writing it, in placement order.
            let mut charged = 0usize;
            for &(end, nodes) in &costs {
                m.charge(end.saturating_sub(charged))?;
                m.tick(u32::try_from(nodes).unwrap_or(u32::MAX))?;
                charged = end;
            }
            Ok(())
        })();
        m.release(working);
        placing
    }

    fn global_align(&self, g: GlobalId, ty: TypeId) -> Result<u64, Error> {
        let global = self
            .e
            .module
            .global(g)
            .ok_or_else(|| invariant("a global out of range"))?;
        let abi = self.layouts.abi_align(ty).ok_or_else(not_lowered)?;
        Ok(global.align.map_or(abi, |a| a.bytes().max(abi)))
    }

    /// The nodes [`Planner::serialize`] visits writing constant `c` of type `ty` (one per path:
    /// a part reached twice counts twice), or the first refusal it meets. The walk visits each
    /// distinct pair of a constant and a type once, a work unit each, in the writer's order (each
    /// node's own checks, then its parts, the last first), and lays out an aggregate's parts as the
    /// writer does ([`Planner::parts_at`]). A part already checked holds no refusal, so of the
    /// initializer's refusals the first met is the one the writer meets. The writer's bounds checks
    /// are invariants the part layout's own checks imply. The meter's refusals follow this walk's
    /// work units and charges, not the writer's.
    fn blob_nodes(&self, c: ConstId, ty: TypeId, m: &mut Meter) -> Result<u64, Error> {
        enum Step {
            Enter(ConstId, TypeId),
            Exit((ConstId, TypeId), Box<[(ConstId, TypeId)]>),
        }
        let escaping =
            || -> Error { Unsupported::not_implemented(Feature::ConstantGlobalEscaping).into() };
        let consts = self.e.module.consts();
        let entry = tree_bytes(1, std::mem::size_of::<((ConstId, TypeId), u64)>());
        let mut nodes: BTreeMap<(ConstId, TypeId), u64> = BTreeMap::new();
        let mut held = 0usize;
        let mut stack = Vec::new();
        grow(&mut stack, 1, m)?;
        stack.push(Step::Enter(c, ty));
        let walked = (|| -> Result<u64, Error> {
            while let Some(step) = stack.pop() {
                let (key, parts) = match step {
                    Step::Exit(key, parts) => (key, parts),
                    Step::Enter(c, ty) => {
                        if nodes.contains_key(&(c, ty)) {
                            continue;
                        }
                        m.tick(1)?;
                        let k = consts
                            .get(c)
                            .ok_or_else(|| invariant("a constant out of range"))?;
                        let parts: Box<[(ConstId, TypeId)]> = match &k.value {
                            Const::Null | Const::Undef | Const::Poison => Box::new([]),
                            Const::Int(_) | Const::Float(_) => {
                                let n = self.layouts.store_size(ty).ok_or_else(not_lowered)?;
                                usize::try_from(n).map_err(|_| not_lowered())?;
                                Box::new([])
                            }
                            Const::Aggregate(elems) => {
                                // The parts, held twice: by the node's exit and while entered.
                                let bytes = elems
                                    .len()
                                    .saturating_mul(2 * std::mem::size_of::<(ConstId, TypeId)>());
                                m.charge(bytes)?;
                                held = held.saturating_add(bytes);
                                let at = self.parts_at(ty, elems.len())?;
                                elems
                                    .iter()
                                    .enumerate()
                                    .map(|(i, &e)| at.part(i).map(|(t, _)| (e, t)))
                                    .collect::<Option<_>>()
                                    .ok_or_else(|| invariant("a constant's part out of range"))?
                            }
                            Const::GlobalAddr(_)
                            | Const::FuncAddr(_)
                            | Const::Expr(_)
                            | Const::InlineAsm(_) => return Err(escaping()),
                        };
                        grow(&mut stack, parts.len().saturating_add(1), m)?;
                        stack.push(Step::Exit((c, ty), parts.clone()));
                        stack.extend(parts.iter().map(|&(c, t)| Step::Enter(c, t)));
                        continue;
                    }
                };
                let below = parts.iter().try_fold(1u64, |n, part| {
                    nodes
                        .get(part)
                        .map(|&p| n.saturating_add(p))
                        .ok_or_else(|| invariant("a constant's part was not walked"))
                })?;
                m.charge(entry)?;
                held = held.saturating_add(entry);
                nodes.insert(key, below);
            }
            nodes
                .get(&(c, ty))
                .copied()
                .ok_or_else(|| invariant("a constant was not walked"))
        })();
        m.release(held.saturating_add(vec_bytes(&stack)));
        walked
    }

    /// Where the `n` parts of an aggregate constant of type `ty` lie in its bytes, as the AIR
    /// datalayout lays it out in memory: a struct's fields at their offsets, an array's elements at
    /// their allocation size apart, and a vector's lanes packed at their width, which must be whole
    /// bytes (a vector of narrower lanes packs bits, which the blob does not write). Placing a
    /// constant ([`Planner::blob_nodes`]) and writing it ([`Planner::serialize`]) both read this,
    /// so they meet the same refusals. A part count other than the type's, or a part that would
    /// pass the aggregate's bytes, is an invariant violation: no write passes the constant's own
    /// bytes.
    fn parts_at(&self, ty: TypeId, n: usize) -> Result<PartsAt<'_>, Error> {
        let types = self.e.module.types();
        let (at, count) = match types.get(ty) {
            Some(Type::Struct { fields, .. }) => {
                let offs = self.layouts.field_offsets(ty).ok_or_else(not_lowered)?;
                (PartsAt::Fields(fields, offs), fields.len().min(offs.len()))
            }
            Some(Type::Array { elem, len }) => {
                let stride = self.layouts.alloc_size(*elem).ok_or_else(not_lowered)?;
                let count = usize::try_from(*len).unwrap_or(usize::MAX);
                (PartsAt::Strided(*elem, stride), count)
            }
            Some(Type::Vector { elem, len }) => {
                let lane = self.layouts.store_size(*elem).ok_or_else(not_lowered)?;
                let bits = match types.get(*elem) {
                    Some(Type::Int(w)) => u64::from(*w),
                    _ => lane.saturating_mul(8),
                };
                if lane.checked_mul(8) != Some(bits) {
                    return Err(not_lowered());
                }
                (PartsAt::Strided(*elem, lane), *len as usize)
            }
            _ => return Err(not_lowered()),
        };
        if count != n {
            return Err(invariant("an aggregate constant of the wrong arity"));
        }
        // Each part within the aggregate's bytes; a part of no known size refuses where it is
        // written, as before.
        let total = self.layouts.store_size(ty).ok_or_else(not_lowered)?;
        let last = match at {
            PartsAt::Fields(..) => 0..n,
            PartsAt::Strided(..) => n.saturating_sub(1)..n,
        };
        for i in last {
            let (t, off) = at
                .part(i)
                .ok_or_else(|| invariant("a constant's part out of range"))?;
            if let Some(size) = self.layouts.store_size(t) {
                if off.checked_add(size).is_none_or(|end| end > total) {
                    return Err(invariant("a constant's part past its bytes"));
                }
            }
        }
        Ok(at)
    }

    /// Write constant `c` of type `ty` as its in-memory bytes at `at` of `out` (little endian,
    /// the AIR datalayout: [`Planner::parts_at`]), never past its own bytes. Pointers and
    /// constant expressions inside an initializer would need a relocation, which the blob does
    /// not have; placing the constant met every such refusal first ([`Planner::blob_nodes`])
    /// and charged the nodes this visits.
    fn serialize(
        &self,
        c: ConstId,
        ty: TypeId,
        out: &mut [u8],
        at: usize,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let escaping =
            || -> Error { Unsupported::not_implemented(Feature::ConstantGlobalEscaping).into() };
        let consts = self.e.module.consts();
        // The constant's own bytes: no write may pass them.
        let end = self
            .layouts
            .store_size(ty)
            .and_then(|n| usize::try_from(n).ok())
            .and_then(|n| at.checked_add(n))
            .ok_or_else(not_lowered)?;
        let mut stack = vec![(c, ty, at)];
        while let Some((c, ty, at)) = stack.pop() {
            let k = consts
                .get(c)
                .ok_or_else(|| invariant("a constant out of range"))?;
            match &k.value {
                Const::Null | Const::Undef | Const::Poison => {}
                Const::Int(v) | Const::Float(v) => {
                    let n = self.layouts.store_size(ty).ok_or_else(not_lowered)?;
                    let n = usize::try_from(n).map_err(|_| not_lowered())?;
                    let bytes = v.to_le_bytes();
                    let to = at
                        .checked_add(n)
                        .filter(|&to| to <= end)
                        .ok_or_else(|| invariant("a blob write past its constant"))?;
                    let dst = out
                        .get_mut(at..to)
                        .ok_or_else(|| invariant("a blob write out of range"))?;
                    for (i, d) in dst.iter_mut().enumerate() {
                        *d = bytes.get(i).copied().unwrap_or(0);
                    }
                }
                Const::Aggregate(elems) => {
                    let placed_bytes = elems
                        .len()
                        .saturating_mul(std::mem::size_of::<(TypeId, u64)>());
                    m.charge(placed_bytes)?;
                    let parts = self.parts_at(ty, elems.len())?;
                    let placed: Vec<(TypeId, u64)> = (0..elems.len())
                        .map(|i| parts.part(i))
                        .collect::<Option<_>>()
                        .ok_or_else(|| invariant("a constant's part out of range"))?;
                    grow(&mut stack, elems.len(), m)?;
                    for (&e, (t, off)) in elems.iter().zip(placed) {
                        let off = usize::try_from(off).map_err(|_| not_lowered())?;
                        stack.push((e, t, at.saturating_add(off)));
                    }
                    m.release(placed_bytes);
                }
                Const::GlobalAddr(_)
                | Const::FuncAddr(_)
                | Const::Expr(_)
                | Const::InlineAsm(_) => return Err(escaping()),
            }
        }
        Ok(())
    }

    /// The threadgroup arena: the static globals as Metal lays them out, then the parameter
    /// regions.
    ///
    /// Metal places a kernel's static threadgroup objects in the reverse of the order of their
    /// last references in the function, each at the next offset aligned to its alignment and to
    /// at least 4 bytes, from 0 ([`Planner::metal_order`]; measured on an M3 Max and an M4). The
    /// layout is otherwise unobservable, but an integer threadgroup address names a byte of it
    /// (`lower::access::local`), so the arena is that layout. The globals of `typed` (in global
    /// order) are held out of it ([`Planner::typed_threadgroup`]): the rest keep Metal's order.
    fn threadgroup_arena(&self, typed: &[SharedPlan], m: &mut Meter) -> Result<ArenaLayout, Error> {
        let (order, metal_layout) = self.metal_order(m)?;
        let mut objects = Vec::new();
        grow(&mut objects, order.len(), m)?;
        for g in order {
            if typed.binary_search_by_key(&g, |s| s.global).is_ok() {
                continue;
            }
            let global = self
                .e
                .module
                .global(g)
                .ok_or_else(|| invariant("a global out of range"))?;
            let size = self
                .layouts
                .alloc_size(global.value_ty)
                .ok_or_else(not_lowered)?;
            let align = self.global_align(g, global.value_ty)?.max(METAL_TG_ALIGN);
            objects.push((ArenaObject::Global(g), size, align));
        }
        let mut arena = ArenaLayout::in_placement_order(&objects, m)?;
        arena.metal_layout = metal_layout;
        for b in &self.e.ep.params {
            let ParamRole::Threadgroup(arg) = &b.role else {
                continue;
            };
            let AbiInt::Const(index) = arg.location.index else {
                return Err(not_lowered());
            };
            // Metal threadgroup lengths are multiples of 16 bytes; every region starts on one.
            grow(&mut arena.dynamic, 1, m)?;
            arena.dynamic.push(DynamicRegion {
                param: b.param,
                index,
                align: 16,
            });
        }
        arena.imageblock = self.imageblock();
        Ok(arena)
    }

    /// The static threadgroup globals held as typed objects, and the arena of the rest (`full`
    /// when none is typed).
    ///
    /// A global is a `Workgroup` variable of its own type, each leaf in its held form
    /// (`memory::alloca`: a `half` in an `f32` that is always a `half` value, a `float` as itself),
    /// where every access of it is a plain load or store of exactly the subobject at its offset
    /// shape, its pointers merge with no other root, escape, are stored or compared with another
    /// root's, and no atomic reaches it ([`RootUse`]). Its loads and stores are then those of the
    /// held value through an access chain, contained by clamping each dynamic index to its level
    /// (`lower::access::local`), with no conversion to and from a `half`'s bits and no sub-word
    /// merge of a word's other bytes, which the arena's two word atomics do: the value a load
    /// returns is the one the last store stored, as in Metal's memory, NaN bits included. A
    /// 32-bit array's element is one typed load or store whose clamped indices hoist out of a
    /// loop, where the arena checks each word's byte offset.
    ///
    /// Only a kernel's static objects, without `[[threadgroup(n)]]` regions or an imageblock: Metal
    /// reports its static length (`tg_metal_static`), and the encoder's lengths must fit beside
    /// it, which the binder checks against the reported length. A held object may take more bytes
    /// than Metal's (a `half`'s four), so an object is typed only while the function's workgroup
    /// memory stays within the device's limit with the whole arena counted, and only where the
    /// held object takes at most twice its own bytes (no 8-bit leaf held in four times them).
    /// Without a threadgroup pointer of no known object (`Root::Dynamic`: loaded, or an integer
    /// address), as such a pointer could address any byte of Metal's layout. The objects take
    /// their unshared type size from `room`, as typed allocas do.
    fn typed_threadgroup(
        &self,
        full: ArenaLayout,
        room: &mut u64,
        m: &mut Meter,
    ) -> Result<(ArenaLayout, Vec<SharedPlan>), Error> {
        if full.entries.is_empty() || !self.threadgroup_typable(m)? {
            return Ok((full, Vec::new()));
        }
        let limit = super::arena::workgroup_limit(
            crate::pipeline::sym::Stage::Kernel,
            &self.p.fields().limits,
        );
        let available = limit.available.saturating_sub(limit.wrapper);
        let mut total = full.static_size;
        let mut typed: Vec<SharedPlan> = Vec::new();
        let chosen = (|| -> Result<(), Error> {
            for e in &full.entries {
                m.tick(1)?;
                let ArenaObject::Global(g) = e.object else {
                    continue;
                };
                let ty = self
                    .e
                    .module
                    .global(g)
                    .ok_or_else(|| invariant("a global out of range"))?
                    .value_ty;
                let u = self.uses(Root::Global(g));
                let plain = u.dereferenced
                    && u.typed
                    && !(u.atomic || u.merged || u.escapes || u.stored || u.cross_compare);
                if !plain || self.e.module.types().holds_pointer(ty) {
                    continue;
                }
                let Some(shape) = super::alloca::typed_ty(self.e, self.p, ty, false, m)?.ok()
                else {
                    continue;
                };
                let unshared = shape.unshared();
                let held = shape.held_bytes(m);
                shape.release(m);
                let bytes = held?.checked_next_multiple_of(16).unwrap_or(u64::MAX);
                let twice = e.size.saturating_mul(2).checked_next_multiple_of(16);
                if twice.is_none_or(|t| bytes > t) {
                    continue;
                }
                let Some(grown) = total.checked_add(bytes).filter(|t| *t <= available) else {
                    continue;
                };
                let Some(left) = room.checked_sub(unshared) else {
                    continue;
                };
                *room = left;
                total = grown;
                grow(&mut typed, 1, m)?;
                typed.push(SharedPlan {
                    global: g,
                    ty,
                    bytes,
                });
            }
            Ok(())
        })();
        chosen?;
        if typed.is_empty() {
            return Ok((full, typed));
        }
        // The objects are in global order (the arena's entries are), as the lookups search them.
        let mut arena = self.threadgroup_arena(&typed, m)?;
        arena.metal_layout = false;
        Ok((arena, typed))
    }

    /// Whether the entry's static threadgroup globals may be typed
    /// ([`Planner::typed_threadgroup`]): a kernel without `[[threadgroup(n)]]` regions or an
    /// imageblock, none of whose threadgroup pointers (values or constants) may have the dynamic
    /// root.
    fn threadgroup_typable(&self, m: &mut Meter) -> Result<bool, Error> {
        if !matches!(self.e.ep.stage, crate::air::abi::Stage::Kernel { .. }) {
            return Ok(false);
        }
        let regions = self
            .e
            .ep
            .params
            .iter()
            .any(|b| matches!(b.role, ParamRole::Threadgroup(_) | ParamRole::Imageblock(_)));
        if regions {
            return Ok(false);
        }
        let types = self.e.module.types();
        let values = self.prov.values;
        let threadgroup = |v: Value| {
            self.e
                .module
                .value_ty(self.e.func, v)
                .and_then(|t| types.get(t))
                .is_some_and(|t| {
                    matches!(
                        t,
                        Type::Ptr {
                            space: AddrSpace::Threadgroup | AddrSpace::ThreadgroupImageblock
                        }
                    )
                })
        };
        let facts = self
            .prov
            .facts
            .iter()
            .filter_map(|(v, f)| Some((values.value(*v)?, f)))
            .chain(
                self.prov
                    .const_facts
                    .iter()
                    .map(|(c, f)| (Value::Const(*c), f)),
            );
        for (v, f) in facts {
            m.tick(1)?;
            if threadgroup(v) && has_dynamic(self.prov.sets.get(f.roots)) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The entry's static threadgroup objects in the order Metal places them, and whether that
    /// order is known to be Metal's.
    ///
    /// Metal places the objects in the reverse of the order of their last references in the
    /// function: walking the body backwards (blocks in layout order, each block's instructions in
    /// order, an instruction's operands and the constants they name), the first object met lies
    /// at 0, the next after it, and so on. Measured on an M3 Max and an M4 over kernels whose
    /// module order, first-reference order and last-reference order all differ, last references
    /// in either arm of a branch and in an atomic, and objects that a function constant removes.
    /// The order is known only when the body names every object and no instruction or constant
    /// names two of them (whose order no measurement fixes); otherwise the objects are placed in
    /// module order, which nothing observes, and no integer address reads them
    /// ([`ArenaLayout::metal_layout`]). A module declaring function constants is refused there
    /// too: Metal lays out the specialized function.
    fn metal_order(&self, m: &mut Meter) -> Result<(Vec<GlobalId>, bool), Error> {
        let globals = &self.e.module.parts().globals;
        let mut tg: Vec<GlobalId> = Vec::new();
        for (g, global) in globals.iter() {
            m.tick(1)?;
            if global.space == AddrSpace::Threadgroup {
                grow(&mut tg, 1, m)?;
                tg.push(g);
            }
        }
        if tg.is_empty() {
            return Ok((tg, true));
        }
        let body = self
            .e
            .module
            .func(self.e.func)
            .and_then(|f| f.body.as_ref())
            .ok_or_else(|| invariant("the entry function has no body"))?;
        // The threadgroup object each constant names, children before parents (`ConstTable::iter`).
        let consts = self.e.module.consts();
        let mut named: Vec<Named> = Vec::new();
        grow(&mut named, consts.len(), m)?;
        for (_, k) in consts.iter() {
            m.tick(1)?;
            let here = match &k.value {
                Const::GlobalAddr(g)
                    if globals.get(*g).map(|x| x.space) == Some(AddrSpace::Threadgroup) =>
                {
                    Named::One(*g)
                }
                v => v.children().iter().try_fold(Named::None, |acc, c| {
                    m.tick(1)?;
                    Ok::<_, Error>(
                        acc.and(named.get(c.raw() as usize).copied().unwrap_or(Named::None)),
                    )
                })?,
            };
            named.push(here);
        }
        // Each object's last reference: the index of the instruction that names it last.
        let mut last: BTreeMap<GlobalId, usize> = BTreeMap::new();
        let entry = tree_bytes(1, std::mem::size_of::<(GlobalId, usize)>());
        let mut charged = 0usize;
        let mut known = true;
        let walk = (|| -> Result<(), Error> {
            for (at, (_, inst)) in body.insts.iter().enumerate() {
                m.tick(1)?;
                let mut here = Named::None;
                inst.kind.for_each_value(|v| {
                    if let Value::Const(c) = v {
                        here =
                            here.and(named.get(c.raw() as usize).copied().unwrap_or(Named::None));
                    }
                });
                match here {
                    Named::None => {}
                    Named::One(g) => {
                        if last.insert(g, at).is_none() {
                            m.charge(entry)?;
                            charged += entry;
                        }
                    }
                    Named::Many => known = false,
                }
            }
            Ok(())
        })();
        let order = walk.and_then(|()| {
            known &= last.len() == tg.len();
            if known {
                let mut by_last: Vec<(usize, GlobalId)> = Vec::new();
                grow(&mut by_last, last.len(), m)?;
                by_last.extend(last.iter().map(|(&g, &at)| (at, g)));
                by_last.sort_unstable_by(|a, b| b.cmp(a));
                Ok(by_last.into_iter().map(|(_, g)| g).collect())
            } else {
                Ok(tg)
            }
        });
        m.release(charged);
        Ok((order?, known))
    }

    /// The threadgroup arena's imageblock region, when it has a model.
    ///
    /// A kernel's explicit imageblock parameter: exactly one explicit imageblock with an
    /// `air.imageblock_data_size` and a power-of-two `air.arg_type_align_size`, unaliased.
    /// `air.imageblock_data` names no imageblock, so a kernel with two has none;
    /// `lower::access::imageblock` refuses a use without one. The cell stride is the data size
    /// rounded up to the alignment (Metal's own sizes already are multiples): the layout is
    /// unobservable, and a stride that is not would misalign every other cell's fields. A cell
    /// pointer may reach past the data size (a `{half v}` cell stored as a `half2`, a slice of a
    /// `half2` from it), which Metal holds, each cell its own (its imageblock is 8 bytes a pixel
    /// for that layout, measured on the M3): the stride then covers the widest access through a
    /// cell pointer, [`Planner::imageblock_reach`].
    ///
    /// A tile function's implicit imageblock: with an explicit layout aliased onto it, that
    /// layout's cell, as above but a whole number of words (its members are the attachments'
    /// texels at their offsets); with only one implicit imageblock, its layout's members' extent
    /// rounded up to a word. No function observes the stride. `lower::access::tile` checks the layouts against each other and the
    /// attachments.
    fn imageblock(&self) -> Option<ImageblockRegion> {
        let of = |kind: ImageblockKind| {
            self.e.ep.params.iter().filter_map(move |b| match &b.role {
                ParamRole::Imageblock(arg) if arg.kind == kind => Some((b, arg)),
                _ => None,
            })
        };
        let mut explicit = of(ImageblockKind::Explicit);
        let mut implicit = of(ImageblockKind::Implicit);
        let (first_explicit, second_explicit) = (explicit.next(), explicit.next());
        let (first_implicit, second_implicit) = (implicit.next(), implicit.next());
        if second_explicit.is_some() || second_implicit.is_some() {
            return None;
        }
        match (first_explicit, first_implicit) {
            // An unaliased explicit imageblock beside an implicit one has no model.
            (Some((b, arg)), implicit) if arg.alias_implicit || implicit.is_none() => {
                let size = arg.data_size.filter(|n| *n > 0)?;
                let align = b.ty.align.filter(|a| a.is_power_of_two())?;
                // A tile function's cells are whole words, which its staging zeroes where no
                // attachment texel covers them (`lower::access::tile`).
                let align = if arg.alias_implicit {
                    align.max(4)
                } else {
                    align
                };
                let size = match arg.alias_implicit {
                    true => size,
                    false => size.max(self.imageblock_reach()?),
                };
                Some(ImageblockRegion {
                    param: b.param,
                    cell_bytes: size.checked_next_multiple_of(align)?,
                })
            }
            (None, Some((b, arg))) => {
                let layout = self.e.structs.get(usize::try_from(arg.layout?.0).ok()?)?;
                let mut end = 0u32;
                for m in &layout.members {
                    let n = m.size.checked_mul(m.array_len.max(1))?;
                    end = end.max(m.offset.checked_add(n)?);
                }
                Some(ImageblockRegion {
                    param: b.param,
                    cell_bytes: end.checked_next_multiple_of(4)?.max(4),
                })
            }
            _ => None,
        }
    }

    /// The end of the widest access through a kernel's cell pointers (`air.imageblock_data`):
    /// the highest byte past its cell's start that a load, a store, an atomic, or a slice write
    /// reaches through one, 0 for none. An access at an offset the analysis does not bound, or
    /// before the cell, adds nothing: `lower::access` and `lower::texture::slice` check each
    /// against the cell and refuse what it does not hold.
    fn imageblock_reach(&self) -> Option<u32> {
        let mut end = 0u32;
        for (r, u) in &self.prov.uses {
            if !imageblock_cell(self.e, *r) {
                continue;
            }
            if let Extent::Span { lo, hi } = u.extent {
                if lo >= 0 {
                    end = end.max(u32::try_from(hi).ok()?);
                }
            }
        }
        Some(end)
    }

    /// Whether `v` is an implicit imageblock parameter.
    fn implicit_imageblock_param(&self, v: Value) -> bool {
        let Value::Arg(a) = v else { return false };
        self.e.ep.params.iter().any(|b| {
            b.param == a.raw()
                && matches!(&b.role, ParamRole::Imageblock(arg) if arg.kind == ImageblockKind::Implicit)
        })
    }

    /// The entry's payload parameter (`air.payload`), if it has one.
    fn payload_param(&self) -> Option<u32> {
        self.e
            .ep
            .params
            .iter()
            .find(|b| matches!(b.role, ParamRole::Payload { .. }))
            .map(|b| b.param)
    }

    /// Whether the entry has an implicit imageblock or an explicit one aliased onto it: tile
    /// shading.
    fn tile_shading(&self) -> bool {
        self.e.ep.params.iter().any(|b| match &b.role {
            ParamRole::Imageblock(arg) => {
                arg.kind == ImageblockKind::Implicit || arg.alias_implicit
            }
            _ => false,
        })
    }

    /// How a typed object would hold type `ty`: its shape is built ([`super::alloca::typed_ty`], in
    /// work linear in its distinct types), and a shape whose unshared size fits in `room` takes it
    /// from the room. The shape is dropped and its ledger bytes returned; a budget stop is an
    /// error, never an untyped placement.
    fn fit(&self, ty: TypeId, room: &mut u64, m: &mut Meter) -> Result<Fit, Error> {
        Ok(match self.shape_size(ty, false, m)? {
            Some((size, _)) => match room.checked_sub(size) {
                Some(left) => {
                    *room = left;
                    Fit::Typed
                }
                None => Fit::OverRoom,
            },
            None => Fit::Untyped,
        })
    }

    /// The unshared size of the typed shape of `ty`, with its simdgroup matrices as fragments
    /// when `fragments`, and whether it holds any fragment; `None` when no typed object holds
    /// it. The shape is dropped and its ledger bytes returned.
    fn shape_size(
        &self,
        ty: TypeId,
        fragments: bool,
        m: &mut Meter,
    ) -> Result<Option<(u64, bool)>, Error> {
        Ok(
            match super::alloca::typed_ty(self.e, self.p, ty, fragments, m)? {
                Ok(shape) => {
                    let out = (shape.unshared(), shape.has_fragments());
                    shape.release(m);
                    Some(out)
                }
                Err(_) => None,
            },
        )
    }

    /// The allocas a typed object may hold, in instruction order: each one's unshared size with
    /// its simdgroup matrices as fragments, and whether it holds any. Those that hold matrices
    /// and fit `room` in instruction order, as every typed object before them does, are the
    /// candidates for holding fragments (`memory::fragment`). The vector's charge is added to
    /// `charged`.
    fn eligible(
        &self,
        body: &crate::air::ir::Body,
        room: u64,
        charged: &mut usize,
        m: &mut Meter,
    ) -> Result<(Vec<(InstId, u64, bool)>, BTreeSet<InstId>), Error> {
        let mut out = Vec::new();
        let mut candidates = BTreeSet::new();
        let mut room = room;
        for (id, inst) in body.insts.iter() {
            let InstKind::Alloca { ty, .. } = &inst.kind else {
                continue;
            };
            let u = self.uses(Root::Alloca(id));
            // A pointer-holding alloca is a pointer cell in the arena.
            let cell = self.e.module.types().holds_pointer(*ty);
            if !u.typed || u.merged || u.escapes || u.stored || u.cross_compare || cell {
                continue;
            }
            let Some((size, fragments)) = self.shape_size(*ty, true, m)? else {
                continue;
            };
            provenance::grow_working(&mut out, 1, charged, m)?;
            out.push((id, size, fragments));
            if let Some(left) = room.checked_sub(size) {
                room = left;
                if fragments {
                    m.charge(provenance::map_entry_bytes::<InstId, ()>())?;
                    *charged = charged.saturating_add(provenance::map_entry_bytes::<InstId, ()>());
                    candidates.insert(id);
                }
            }
        }
        Ok((out, candidates))
    }

    /// The private arena and the typed allocas, in instruction order. An alloca a typed object
    /// cannot hold, or one past `room`, has its place in the arena.
    ///
    /// An alloca that holds simdgroup matrices is typed only when they hold fragments
    /// (`memory::fragment`): the candidates are found in a first pass over the room, and the held
    /// ones take their room first. They fit together, since they fit among the first pass's typed
    /// objects; the other allocas then take the room left in instruction order. With them, the `<64
    /// x T>` instructions that are fragments.
    fn allocas(
        &self,
        room: &mut u64,
        m: &mut Meter,
    ) -> Result<(ArenaLayout, Vec<AllocaPlan>, Vec<InstId>), Error> {
        let body = self
            .e
            .module
            .func(self.e.func)
            .and_then(|f| f.body.as_ref())
            .ok_or_else(|| invariant("the entry function has no body"))?;
        let mut charged = 0;
        let placed = self.place_allocas(body, room, &mut charged, m);
        m.release(charged);
        placed
    }

    fn place_allocas(
        &self,
        body: &crate::air::ir::Body,
        room: &mut u64,
        charged: &mut usize,
        m: &mut Meter,
    ) -> Result<(ArenaLayout, Vec<AllocaPlan>, Vec<InstId>), Error> {
        let (eligible, candidates) = self.eligible(body, *room, charged, m)?;
        let super::fragment::Fragments { held, values } =
            super::fragment::held(self.e, &self.layouts, &self.prov, &candidates, m)?;
        for (id, size, _) in &eligible {
            if held.contains(id) {
                *room = room
                    .checked_sub(*size)
                    .ok_or_else(|| invariant("allocas holding fragments past the room"))?;
            }
        }
        let mut arena = ArenaLayout::default();
        let mut typed = Vec::new();
        let mut next = eligible.iter().peekable();
        for (id, inst) in body.insts.iter() {
            let InstKind::Alloca { ty, count, align } = &inst.kind else {
                continue;
            };
            let count = match count {
                Value::Const(c) => match self.e.module.consts().get(*c).map(|k| &k.value) {
                    Some(Const::Int(n)) => *n,
                    _ => return Err(not_lowered()),
                },
                _ => return Err(not_lowered()),
            };
            // A type with a member no typed object holds (a chunked vector) has a place in the
            // arena, whose leaves hold every such member; so does a simdgroup matrix's `<64 x
            // half>` storage that holds more than fragments.
            let shape = next.next_if(|(e, _, _)| *e == id).copied();
            let fragments = held.contains(&id);
            let typed_here = match shape {
                Some(_) if fragments => true,
                Some((_, size, false)) => match room.checked_sub(size) {
                    Some(left) => {
                        *room = left;
                        true
                    }
                    None => false,
                },
                _ => false,
            };
            if typed_here {
                grow(&mut typed, 1, m)?;
                typed.push(AllocaPlan {
                    inst: id,
                    ty: *ty,
                    count,
                    fragments,
                });
                continue;
            }
            // An arena past `u64` is past its `u32` offsets' bound, which selection checks when a
            // pointer into the arena is used; a ledger refusal stays.
            let size = self
                .layouts
                .alloc_size(*ty)
                .ok_or_else(not_lowered)?
                .checked_mul(count)
                .ok_or(Error::LimitExceeded(Limit::PrivateArenaBytes))?;
            let abi = self.layouts.abi_align(*ty).ok_or_else(not_lowered)?;
            let align = align.map_or(abi, |a| a.bytes().max(abi));
            arena
                .place(ArenaObject::Alloca(id), size, align, m)
                .map_err(|err| {
                    if err == too_large() {
                        Error::LimitExceeded(Limit::PrivateArenaBytes)
                    } else {
                        err
                    }
                })?;
        }
        Ok((arena, typed, values))
    }

    /// The representation of every pointer value, the offsets, and the handle-table classes.
    fn reprs(
        &self,
        const_globals: &[ConstGlobalPlan],
        typed_allocas: &[AllocaPlan],
        typed_shared: &[SharedPlan],
        dynamic: bool,
        m: &mut Meter,
    ) -> Result<Reprs, Error> {
        // The wide roots: the component of the dynamic root over the buffer pointers' root sets, so
        // a pointer and every operand it is computed from share one width. A buffer root whose
        // pointer escapes (stored as a value, or `ptrtoint`) joins it: its address can come back as
        // a loaded pointer, and a wide pointer's guest address is exact.
        //
        // Each distinct set is joined once, so the union-find reads the roots of the interned
        // sets, not of every pointer.
        let escaping = self.escaping_roots(m)?;
        let wide_roots = if dynamic || !escaping.is_empty() {
            let mut charged = 0;
            let distinct = self.distinct_buffer_root_sets(&mut charged, m);
            let wide = distinct.and_then(|distinct| {
                let sets = distinct.iter().map(|id| self.prov.sets.get(*id));
                let joined = escaping.iter().map(|pair| pair.as_slice());
                super::roots::wide_roots(sets.chain(joined), m)
            });
            m.release(charged);
            wide?
        } else {
            BTreeSet::new()
        };
        let values = self.prov.values;
        let mut repr: Vec<Option<Repr>> = Vec::new();
        m.reserve(&mut repr, values.len())?;
        repr.resize(values.len(), None);
        let mut offsets = Vec::new();
        m.reserve(
            &mut offsets,
            self.prov.facts.len() + self.prov.const_facts.len(),
        )?;
        let mut strided = Vec::new();
        let mut root_sets = Vec::new();
        m.reserve(
            &mut root_sets,
            self.prov.facts.len() + self.prov.const_facts.len(),
        )?;
        let mut handles = BTreeSet::new();
        let types = self.e.module.types();
        // The two lookup maps below (`InstId` and `GlobalId` keys are both `u32` newtypes).
        m.charge(
            (typed_allocas.len() + const_globals.len())
                .saturating_mul(provenance::map_entry_bytes::<InstId, u32>()),
        )?;
        let typed: BTreeMap<InstId, u32> = typed_allocas
            .iter()
            .enumerate()
            .filter_map(|(i, a)| Some((a.inst, u32::try_from(i).ok()?)))
            .collect();
        let private: BTreeMap<GlobalId, u32> = const_globals
            .iter()
            .enumerate()
            .filter(|(_, c)| c.form == ConstGlobalForm::Private)
            .filter_map(|(i, c)| Some((c.global, u32::try_from(i).ok()?)))
            .collect();
        // Walked in place: the facts, then the pointer constants (no n-sized copy).
        let entries = self
            .prov
            .facts
            .iter()
            .zip(self.prov.classes.iter().copied())
            .map(|((v, f), c)| (*v, *f, c))
            .chain(self.prov.const_facts.iter().filter_map(|(c, f)| {
                let v = values.id(Value::Const(*c))?;
                let class = match self.e.module.consts().get(*c).map(|k| &k.value) {
                    Some(Const::GlobalAddr(g)) if self.is_static_sampler(*g) => {
                        Class::Handle(HandleClass::Sampler)
                    }
                    _ => Class::Unknown,
                };
                Some((v, *f, class))
            }));
        for (v, f, class) in entries {
            m.tick(1)?;
            // An implicit imageblock's pointer parameter has no lowered form: a tile function
            // reaches the imageblock through its intrinsics, and any pointer computed from the
            // parameter is refused below.
            if values
                .value(v)
                .is_some_and(|v| self.implicit_imageblock_param(v))
            {
                continue;
            }
            let ty = values
                .value(v)
                .and_then(|v| self.e.module.value_ty(self.e.func, v))
                .ok_or_else(|| invariant("a pointer value without a type"))?;
            let Some(Type::Ptr { space }) = types.get(ty) else {
                return Err(invariant("a pointer fact on a non-pointer value"));
            };
            let roots = self.prov.sets.get(f.roots);
            let array = handle_array_root(&self.prov.abi, roots);
            let table = buffer_table_root(&self.prov.abi, roots);
            let r = match (class, *space) {
                (Class::Handle(class), _) => {
                    if has_dynamic(roots) {
                        handles.insert(class);
                    }
                    Repr::Handle {
                        class,
                        cands: f.roots,
                    }
                }
                (Class::Byte | Class::Unknown, _) if array.is_some() => Repr::HandleArray {
                    class: array.ok_or_else(|| invariant("a handle array without a class"))?,
                },
                (Class::Byte | Class::Unknown, AddrSpace::Thread) if table.is_some() => {
                    Repr::BufferTable {
                        param: table.ok_or_else(|| invariant("a buffer table without a param"))?,
                    }
                }
                (Class::Conflict, _) => {
                    return Err(Unsupported::not_representable(Feature::HandleAsPointer).into())
                }
                // An imageblock cell pointer is an offset into the threadgroup arena, where the
                // imageblock is the last region: only `air.imageblock_data` makes one. Any other
                // (an implicit imageblock's parameter) is tile shading's when the entry has an
                // implicit imageblock.
                (_, AddrSpace::ThreadgroupImageblock) => {
                    if !roots.iter().all(|r| imageblock_cell(self.e, *r)) {
                        let feature = if self.tile_shading() {
                            Feature::TileShading
                        } else {
                            Feature::Imageblocks
                        };
                        return Err(Unsupported::not_implemented(feature).into());
                    }
                    Repr::TgPtr
                }
                // A pointer into a typed threadgroup object has it as its one root: the object's
                // pointers merge with no other's (`Planner::typed_threadgroup`).
                (_, AddrSpace::Threadgroup) => match roots {
                    [Root::Global(g)] => match typed_shared.binary_search_by_key(g, |s| s.global) {
                        Ok(i) => Repr::TgTyped(
                            u32::try_from(i).map_err(|_| invariant("a typed object past u32"))?,
                        ),
                        Err(_) => Repr::TgPtr,
                    },
                    _ => Repr::TgPtr,
                },
                // An object or mesh function's payload: an offset into its payload arena, for a
                // pointer whose one root is the payload parameter; any other is not implemented. In
                // any other function, and as a mesh output that is no handle, the address space
                // keeps the refusal of an address space without established meaning.
                (_, AddrSpace::ObjectData)
                    if matches!(
                        self.e.ep.stage,
                        crate::air::abi::Stage::Object { .. } | crate::air::abi::Stage::Mesh { .. }
                    ) =>
                {
                    match (roots, self.payload_param()) {
                        ([Root::Param(p)], Some(payload)) if *p == payload => Repr::PayloadPtr,
                        _ => return Err(Unsupported::not_implemented(Feature::MeshShading).into()),
                    }
                }
                (_, space @ (AddrSpace::ObjectData | AddrSpace::MeshOutput)) => {
                    return Err(
                        Unsupported::not_implemented(Feature::AddressSpace(space.raw())).into(),
                    )
                }
                (_, AddrSpace::Thread) => match roots {
                    [Root::Alloca(a)] => match typed.get(a) {
                        Some(&i) => Repr::PrivPtr(PrivForm::Typed(i)),
                        None => Repr::PrivPtr(PrivForm::Arena),
                    },
                    _ => Repr::PrivPtr(PrivForm::Arena),
                },
                (_, AddrSpace::Device | AddrSpace::Constant) => {
                    let private = match roots {
                        [Root::Global(g)] => private.get(g).copied(),
                        _ => None,
                    };
                    match private {
                        Some(i) => Repr::ConstPtr(ConstForm::Private(i)),
                        // Every non-null root of a buffer pointer's set is in one component
                        // (`wide_roots` joined them), and a pointer constant's set holds at
                        // most one, so its first non-null root decides its width.
                        None => Repr::BufPtr {
                            roots: f.roots,
                            wide: roots
                                .iter()
                                .find(|r| **r != Root::Null)
                                .is_some_and(|r| wide_roots.contains(r)),
                        },
                    }
                }
            };
            if let Some(slot) = repr.get_mut(v.index()) {
                *slot = Some(r);
            }
            if let (Repr::BufPtr { .. }, Shape::Strided { base, strides }) = (r, f.shape) {
                let mut each = strides.iter();
                if let (Some(stride), None) = (each.next(), each.next()) {
                    grow(&mut strided, 1, m)?;
                    strided.push((v, base, stride));
                }
            }
            offsets.push((v, f.range));
            root_sets.push((v, f.roots));
        }
        offsets.sort_unstable_by_key(|(v, _)| *v);
        strided.sort_unstable_by_key(|(v, _, _)| *v);
        root_sets.sort_unstable_by_key(|(v, _)| *v);
        let handle_table = (!handles.is_empty()).then(|| handles.into_iter().collect());
        Ok(Reprs {
            repr,
            offsets,
            strided,
            root_sets,
            handle_table,
        })
    }
}

#[cfg(test)]
mod tests;
