//! The provenance lattice and `Repr`.
//!
//! One forward dataflow over the specialized entry's SSA assigns every pointer value a
//! [`PtrFact`]: the set of [`Root`]s it may point into, the range of its byte offset from that
//! root, and the offset's [`Shape`] (an exact constant, or a constant plus a dynamic multiple of
//! each of a few strides).
//! The lattice is finite in the root sets and widened in the offsets, so the fixpoint is reached
//! in a bounded number of visits. A second pass reads every use of a pointer: accesses (with
//! their byte extents and access types), escapes (`ptrtoint`, a pointer stored as a value),
//! cross-root compares, and the handle seeds. Handle classification is a union-find over the values
//! that `phi`, `select`, and `freeze` merge: a class is `Byte`, one [`HandleClass`], or a conflict,
//! which is `Unsupported{HandleAsPointer}`.
//!
//! A buffer table (an `array_ref` of buffers passed as a thread pointer) is read one element
//! pointer at a time; each element is its own static root ([`Root::BufferElement`]), bound to its
//! own argument-table slot.
//!
//! [`Repr`] is the lowered representation `select` demotes a pointer value to; every variant is
//! demotable (to a fat pointer, an arena offset, or a handle index), so the structurizer never
//! meets a non-demotable merge. Non-pointer values have no `Repr` here: their lowered
//! representation is the arithmetic family's.

use std::collections::{BTreeMap, HashMap};

use crate::air::abi::{AbiHandle, AbiInt, Member, MemberKind, ParamBinding, ParamRole};
use crate::air::intrinsics::{AirIntrinsic, HandleKind, Operand, ResultField, Signature};
use crate::air::ir::{
    AddrSpace, BlockId, Body, Callee, CastOp, Const, ConstExpr, ConstId, GlobalId, Inst, InstId,
    InstKind, Type, TypeId, TypeLayouts, Value, ValueIndex,
};
use crate::air::Entry;
use crate::budget::Meter;
use crate::error::{Error, Feature, InternalError, InternalKind, LowerFamily, Unsupported};
use crate::memory::abi_index::AbiIndex;
use crate::memory::alloca::{self, Strides};
use crate::memory::union_find::UnionFind;

mod cells;
mod objects;

pub use cells::CellFacts;
use cells::{Cells, Slot, SlotGroups};
use objects::{is_object, ObjectBuffer, ObjectLayouts};

// ---- Roots and root sets ----

/// Where a pointer value may point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Root {
    /// The null pointer (also `undef`/`poison` pointers, a valid refinement): root 0.
    Null,
    /// Entry parameter `n` (a buffer, argument buffer, threadgroup region, or resource handle;
    /// for a buffer table, the table itself).
    Param(u32),
    /// Element `element` of the buffer-table parameter `param` (`array_ref<device T*>`): the buffer
    /// bound to argument-table slot `first + element`, a static root like a parameter.
    BufferElement { param: u32, element: u32 },
    /// A module global (threadgroup, constant data, or a constexpr sampler).
    Global(GlobalId),
    /// A stack allocation.
    Alloca(InstId),
    /// A handle an intrinsic call produces (`get_read_sampler`, an async-copy event), or the
    /// opaque pointer result of an intrinsic of an out-of-scope family.
    Call(InstId),
    /// A pointer loaded from memory or made from an integer: the guest→root translation at run
    /// time. In the thread and threadgroup spaces, some arena offset.
    Dynamic,
}

/// An interned, sorted set of [`Root`]s. `RootSetId::EMPTY` is the lattice bottom.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RootSetId(u32);

impl RootSetId {
    /// The empty set.
    pub const EMPTY: RootSetId = RootSetId(0);

    /// The dense index.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// The root-set interner. Joins are memoized, so a fixpoint over many merges of the same sets
/// allocates each result once.
#[derive(Clone, Debug)]
pub struct RootSets {
    sets: Vec<Box<[Root]>>,
    index: HashMap<Box<[Root]>, RootSetId>,
    joins: HashMap<(RootSetId, RootSetId), RootSetId>,
}

/// Ledger bytes of one interned set of `n` roots: the set and its index key (two copies of the
/// roots), the set's slot in the table, and the index entry.
fn set_bytes(n: usize) -> usize {
    n.saturating_mul(2 * std::mem::size_of::<Root>())
        .saturating_add(std::mem::size_of::<Box<[Root]>>())
        .saturating_add(map_entry_bytes::<Box<[Root]>, RootSetId>())
}

/// Ledger bytes of one memoized join.
const JOIN_BYTES: usize = map_entry_bytes::<(RootSetId, RootSetId), RootSetId>();

impl RootSets {
    fn new() -> RootSets {
        let empty: Box<[Root]> = Box::new([]);
        let mut index = HashMap::new();
        index.insert(empty.clone(), RootSetId::EMPTY);
        RootSets {
            sets: vec![empty],
            index,
            joins: HashMap::new(),
        }
    }

    /// The roots of `id` (empty for an id out of range).
    pub fn get(&self, id: RootSetId) -> &[Root] {
        self.sets.get(id.index()).map_or(&[], |s| &**s)
    }

    /// The number of interned sets.
    pub(crate) fn len(&self) -> usize {
        self.sets.len()
    }

    /// The sets alone, without the interning index and the join memo (for a finished plan).
    pub fn into_table(self) -> RootSets {
        RootSets {
            sets: self.sets,
            index: HashMap::new(),
            joins: HashMap::new(),
        }
    }

    /// The bytes the interned sets hold.
    pub fn retained_bytes(&self) -> usize {
        self.sets
            .iter()
            .map(|s| s.len() * std::mem::size_of::<Root>() + std::mem::size_of::<Box<[Root]>>())
            .sum::<usize>()
            + self
                .index
                .keys()
                .map(|k| {
                    k.len() * std::mem::size_of::<Root>()
                        + map_entry_bytes::<Box<[Root]>, RootSetId>()
                })
                .sum::<usize>()
            + self.joins.len() * JOIN_BYTES
    }

    /// Intern a sorted, deduplicated set. Readers of a set rely on its order (a search for
    /// `Dynamic`, a table load's first and last elements), so a new set that is not strictly
    /// increasing is an invariant violation, not a set.
    fn intern(&mut self, roots: Vec<Root>, m: &mut Meter) -> Result<RootSetId, Error> {
        let key = roots.into_boxed_slice();
        if let Some(&id) = self.index.get(&key) {
            return Ok(id);
        }
        if !key.iter().zip(key.iter().skip(1)).all(|(a, b)| a < b) {
            return Err(invariant("a root set out of order"));
        }
        m.charge(set_bytes(key.len()))?;
        let id =
            RootSetId(u32::try_from(self.sets.len()).map_err(|_| invariant("too many root sets"))?);
        self.sets.push(key.clone());
        self.index.insert(key, id);
        Ok(id)
    }

    /// The set of one root.
    fn single(&mut self, r: Root, m: &mut Meter) -> Result<RootSetId, Error> {
        self.intern(vec![r], m)
    }

    /// The union of two sets.
    fn join(&mut self, a: RootSetId, b: RootSetId, m: &mut Meter) -> Result<RootSetId, Error> {
        if a == b || b == RootSetId::EMPTY {
            return Ok(a);
        }
        if a == RootSetId::EMPTY {
            return Ok(b);
        }
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(&id) = self.joins.get(&key) {
            return Ok(id);
        }
        let mut out: Vec<Root> = self.get(a).iter().chain(self.get(b)).copied().collect();
        out.sort_unstable();
        out.dedup();
        let id = self.intern(out, m)?;
        m.charge(JOIN_BYTES)?;
        self.joins.insert(key, id);
        Ok(id)
    }
}

// ---- Offsets ----

/// The range of a pointer's byte offset from its root, as a closed interval of constants, or
/// `Top` when some offset is not a translation-time constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OffsetRange {
    Empty,
    Span { lo: i64, hi: i64 },
    Top,
}

impl OffsetRange {
    fn join(self, o: OffsetRange) -> OffsetRange {
        match (self, o) {
            (OffsetRange::Empty, x) | (x, OffsetRange::Empty) => x,
            (OffsetRange::Top, _) | (_, OffsetRange::Top) => OffsetRange::Top,
            (OffsetRange::Span { lo: a, hi: b }, OffsetRange::Span { lo: c, hi: d }) => {
                OffsetRange::Span {
                    lo: a.min(c),
                    hi: b.max(d),
                }
            }
        }
    }

    /// The exact offset, when the range is one constant.
    pub fn exact(self) -> Option<i64> {
        match self {
            OffsetRange::Span { lo, hi } if lo == hi => Some(lo),
            _ => None,
        }
    }
}

/// The shape of a pointer's offset for typed access chains: a constant, or a constant plus one
/// dynamic multiple of each of its strides (one dynamic element index per nested array level).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Shape {
    Empty,
    Exact(i64),
    Strided { base: i64, strides: Strides },
    Top,
}

impl Shape {
    fn join(self, o: Shape) -> Shape {
        match (self, o) {
            (Shape::Empty, x) | (x, Shape::Empty) => x,
            (a, b) if a == b => a,
            _ => Shape::Top,
        }
    }
}

/// The analysis fact of one pointer value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PtrFact {
    pub roots: RootSetId,
    pub range: OffsetRange,
    pub shape: Shape,
}

impl PtrFact {
    const BOTTOM: PtrFact = PtrFact {
        roots: RootSetId::EMPTY,
        range: OffsetRange::Empty,
        shape: Shape::Empty,
    };

    fn at(roots: RootSetId, offset: i64) -> PtrFact {
        PtrFact {
            roots,
            range: OffsetRange::Span {
                lo: offset,
                hi: offset,
            },
            shape: Shape::Exact(offset),
        }
    }

    fn unknown(roots: RootSetId) -> PtrFact {
        PtrFact {
            roots,
            range: OffsetRange::Top,
            shape: Shape::Top,
        }
    }
}

/// How many times a value's offsets may change before they widen to `Top`.
const WIDEN_AFTER: u8 = 3;

// ---- Handles ----

/// The class of a resource handle. Texture shapes and accesses are the texture family's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HandleClass {
    Texture,
    Sampler,
    AccelerationStructure,
    FunctionTable,
    IntersectionTable,
    Command,
    ComputePipelineState,
    RenderPipelineState,
    /// A depth/stencil state an argument buffer holds, for a render command's setter.
    DepthStencilState,
    /// A simdgroup async-copy event.
    Event,
    /// An object function's mesh grid.
    MeshGrid,
    /// A mesh function's output.
    Mesh,
    /// A fragment function's `interpolant` input.
    Interpolant,
}

impl HandleClass {
    /// The class of an ABI role or argument-buffer member.
    pub fn of_abi(h: AbiHandle) -> HandleClass {
        match h {
            AbiHandle::Texture(_) => HandleClass::Texture,
            AbiHandle::Sampler => HandleClass::Sampler,
            AbiHandle::AccelerationStructure(_) => HandleClass::AccelerationStructure,
            AbiHandle::FunctionTable => HandleClass::FunctionTable,
            AbiHandle::IntersectionTable => HandleClass::IntersectionTable,
            AbiHandle::CommandBuffer => HandleClass::Command,
            AbiHandle::ComputePipelineState => HandleClass::ComputePipelineState,
            AbiHandle::RenderPipelineState => HandleClass::RenderPipelineState,
            AbiHandle::DepthStencilState => HandleClass::DepthStencilState,
            AbiHandle::MeshGridProperties => HandleClass::MeshGrid,
            AbiHandle::Mesh => HandleClass::Mesh,
            AbiHandle::Interpolant => HandleClass::Interpolant,
        }
    }

    /// The class of an intrinsic's typed handle position.
    pub fn of_kind(k: HandleKind) -> HandleClass {
        match k {
            HandleKind::Texture(_) => HandleClass::Texture,
            HandleKind::Sampler => HandleClass::Sampler,
            HandleKind::VisibleFunctionTable => HandleClass::FunctionTable,
            HandleKind::IntersectionFunctionTable => HandleClass::IntersectionTable,
            HandleKind::AccelerationStructure => HandleClass::AccelerationStructure,
            HandleKind::Command => HandleClass::Command,
            HandleKind::ComputePipelineState => HandleClass::ComputePipelineState,
            HandleKind::RenderPipelineState => HandleClass::RenderPipelineState,
            HandleKind::DepthStencilState => HandleClass::DepthStencilState,
            HandleKind::Event => HandleClass::Event,
            HandleKind::MeshGrid => HandleClass::MeshGrid,
            HandleKind::Mesh => HandleClass::Mesh,
            HandleKind::Interpolant => HandleClass::Interpolant,
        }
    }
}

/// A value's classification: unconstrained, bytes, one handle class, or a conflict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    Unknown,
    Byte,
    Handle(HandleClass),
    Conflict,
}

impl Class {
    /// The lattice join: equal classes agree, anything joined with `Unknown` is itself, and
    /// every other pair conflicts.
    pub fn join(self, o: Class) -> Class {
        match (self, o) {
            (Class::Unknown, x) | (x, Class::Unknown) => x,
            (a, b) if a == b => a,
            _ => Class::Conflict,
        }
    }
}

// ---- Repr ----

/// How a private (thread-space) pointer is represented.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrivForm {
    /// A byte offset into one typed allocation (`MemoryPlan::typed_allocas[index]`), accessed
    /// through typed access chains.
    Typed(u32),
    /// A byte offset into the private arena.
    Arena,
}

/// How a pointer to a program-scope constant is represented.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConstForm {
    /// A byte offset into one closed, type-consistent constant (`MemoryPlan::const_globals[index]`),
    /// a `Private` variable with a constant initializer.
    Private(u32),
}

/// The lowered representation of a pointer value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Repr {
    /// A fat pointer `(root u32, offset u32 | u64)` into the buffer roots of `roots`: parameter
    /// slots, the constant blob, null, and (with `wide`) dynamic roots.
    BufPtr {
        roots: RootSetId,
        wide: bool,
    },
    /// A `u32` byte offset into the threadgroup arena.
    TgPtr,
    /// A `u32` byte offset into one typed threadgroup object (`MemoryPlan::typed_shared[index]`),
    /// a `Workgroup` variable accessed through typed access chains.
    TgTyped(u32),
    PrivPtr(PrivForm),
    ConstPtr(ConstForm),
    /// A `u32` byte offset into a handle-array parameter (`array<texture2d<T>, N>`): each load
    /// through it reads one of the parameter's handles.
    HandleArray {
        class: HandleClass,
    },
    /// A `u32` byte offset into the buffer-table parameter `param` (`array_ref<device T*>`): each
    /// load through it reads the fat pointer of one of its elements.
    BufferTable {
        param: u32,
    },
    /// A `u32` handle index over the candidates `cands`.
    Handle {
        class: HandleClass,
        cands: RootSetId,
    },
    /// A `u32` byte offset into an object or mesh function's payload arena (`object_data`): the
    /// payload parameter's offset is 0.
    PayloadPtr,
}

// ---- Uses of roots ----

/// The byte extent a set of accesses covers relative to a root, or `Unknown` when some access
/// has a non-constant offset or length.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Extent {
    Empty,
    /// `[lo, hi)`.
    Span {
        lo: i64,
        hi: i64,
    },
    Unknown,
}

impl Extent {
    fn join(self, o: Extent) -> Extent {
        match (self, o) {
            (Extent::Empty, x) | (x, Extent::Empty) => x,
            (Extent::Unknown, _) | (_, Extent::Unknown) => Extent::Unknown,
            (Extent::Span { lo: a, hi: b }, Extent::Span { lo: c, hi: d }) => Extent::Span {
                lo: a.min(c),
                hi: b.max(d),
            },
        }
    }

    /// The extent of an access reaching `reach` from its pointer, at pointer offsets `range`.
    fn of(range: OffsetRange, reach: Option<Reach>) -> Extent {
        let reach =
            reach.and_then(|r| Some((i64::try_from(r.start).ok()?, i64::try_from(r.end).ok()?)));
        match (range, reach) {
            (OffsetRange::Empty, _) => Extent::Empty,
            (OffsetRange::Span { lo, hi }, Some((start, end))) => {
                match (lo.checked_add(start), hi.checked_add(end)) {
                    (Some(lo), Some(hi)) => Extent::Span { lo, hi },
                    _ => Extent::Unknown,
                }
            }
            _ => Extent::Unknown,
        }
    }
}

/// The bytes one access touches relative to its pointer, `[start, end)`: `[0, n)` for a plain
/// access of `n` bytes, the hull of the 64 cells for a simdgroup-matrix block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Reach {
    pub start: u64,
    pub end: u64,
}

impl Reach {
    /// The `n` bytes at the pointer.
    pub fn at(n: u64) -> Reach {
        Reach { start: 0, end: n }
    }
}

/// Everything the uses of one root reveal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RootUse {
    /// Some instruction operand reaches this root.
    pub referenced: bool,
    pub read: bool,
    pub write: bool,
    /// An atomic read-modify-write.
    pub atomic: bool,
    /// The union of every access's byte extent.
    pub extent: Extent,
    /// Every access is a plain load or store whose type is exactly the subobject at its offset
    /// shape, and no other use (memcpy, intrinsic) reaches it.
    pub typed: bool,
    /// A pointer into this root merges with another root.
    pub merged: bool,
    /// A pointer into this root becomes an integer (`ptrtoint`) or is stored as a value into
    /// memory that does not forward it.
    pub escapes: bool,
    /// A pointer into this root is stored into a forwarding pointer cell: its loads read it back as
    /// the pointer's lowered representation, not through its definitions.
    pub stored: bool,
    /// A pointer into this root is compared with a pointer that may have another root.
    pub cross_compare: bool,
    /// A pointer into this root is dereferenced (its root set reaches an access).
    pub dereferenced: bool,
    /// Some access reaches this root at an offset that is not one constant: an index the
    /// thread computes (a dynamic multiple of a stride, or an offset not known at all).
    pub indexed: bool,
}

impl Default for RootUse {
    fn default() -> RootUse {
        RootUse {
            referenced: false,
            read: false,
            write: false,
            atomic: false,
            extent: Extent::Empty,
            typed: true,
            merged: false,
            escapes: false,
            stored: false,
            cross_compare: false,
            dereferenced: false,
            indexed: false,
        }
    }
}

/// How an instruction touches memory through a pointer operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Touch {
    Read,
    Write,
    ReadWrite { atomic: bool },
}

/// The provenance analysis of one entry.
#[derive(Clone, Debug)]
pub struct Provenance {
    pub values: ValueIndex,
    pub sets: RootSets,
    /// The fact of every pointer-typed argument and instruction, sorted by value.
    pub facts: Vec<(crate::air::ir::ValueId, PtrFact)>,
    /// The fact of every pointer constant an instruction uses.
    pub const_facts: BTreeMap<ConstId, PtrFact>,
    /// The classification of every pointer-typed argument and instruction, sorted as `facts`.
    pub classes: Vec<Class>,
    /// Every root an instruction reaches, with its uses.
    pub uses: BTreeMap<Root, RootUse>,
    /// The first dereference of a pointer whose root set contains `Dynamic`, in the device or
    /// constant space: the instruction and its block.
    pub dynamic_deref: Option<(BlockId, InstId)>,
    /// Which allocas forward the pointers stored into them.
    pub cells: CellFacts,
    /// The entry's bindings and layouts, indexed once.
    pub abi: AbiIndex,
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// Make room for `need` more elements of `v`, charging the ledger for any growth before it is
/// reserved. Growth is at least the current capacity, so pushes stay amortized, and it is exact, so
/// the charge covers the new allocation.
pub(crate) fn grow<T>(v: &mut Vec<T>, need: usize, m: &mut Meter) -> Result<(), Error> {
    if v.capacity() - v.len() >= need {
        return Ok(());
    }
    let additional = need.max(v.capacity()).max(4);
    let bytes = additional.saturating_mul(std::mem::size_of::<T>());
    m.charge(bytes)?;
    v.try_reserve_exact(additional).map_err(|err| {
        Error::Internal(InternalError::new(
            InternalKind::AllocationFailed,
            format!("reserving {bytes} ledger-admitted bytes: {err}"),
        ))
    })
}

/// Make room for `need` more elements of the working vector `v` ([`grow`]), adding the charge to
/// `bytes`, which the caller releases once `v` is dropped.
pub(crate) fn grow_working<T>(
    v: &mut Vec<T>,
    need: usize,
    bytes: &mut usize,
    m: &mut Meter,
) -> Result<(), Error> {
    if v.capacity() - v.len() >= need {
        return Ok(());
    }
    let additional = need.max(v.capacity()).max(4);
    grow(v, need, m)?;
    *bytes = bytes.saturating_add(additional.saturating_mul(std::mem::size_of::<T>()));
    Ok(())
}

/// Push `x` onto the working vector `v`, charging its growth to `m` and adding the charge to
/// `bytes`, which the caller releases once `v` is dropped.
pub(crate) fn push_working<T>(
    v: &mut Vec<T>,
    x: T,
    bytes: &mut usize,
    m: &mut Meter,
) -> Result<(), Error> {
    if v.len() == v.capacity() {
        let before = v.capacity();
        grow(v, 1, m)?;
        let additional = before.max(1).max(4);
        *bytes = bytes.saturating_add(additional.saturating_mul(std::mem::size_of::<T>()));
    }
    v.push(x);
    Ok(())
}

/// The bytes charged for one entry of the per-root use table.
const USE_ENTRY_BYTES: usize = map_entry_bytes::<Root, RootUse>();

/// The flags a use of a pointer sets on every root it may point into ([`Walk::mark`]): only ever
/// set, never cleared, so a root set given them once needs them no more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Marks(u8);

impl Marks {
    /// An instruction operand reaches the root.
    const REFERENCED: Marks = Marks(1);
    /// Referenced, and its address escapes.
    const ESCAPES: Marks = Marks(1 | 2);
    /// Referenced, and stored into a forwarding cell.
    const STORED: Marks = Marks(1 | 4);
    /// Compared with a pointer that may have another root.
    const CROSS_COMPARE: Marks = Marks(8);

    fn apply(self, u: &mut RootUse) {
        u.referenced |= self.0 & 1 != 0;
        u.escapes |= self.0 & 2 != 0;
        u.stored |= self.0 & 4 != 0;
        u.cross_compare |= self.0 & 8 != 0;
    }
}

/// The bytes charged for one entry of the walk's record of the marks each root set was given.
const MARKED_ENTRY_BYTES: usize = map_entry_bytes::<RootSetId, u8>();

/// The use record of root `r`, charging the ledger before a new entry is inserted.
fn root_use<'u>(
    uses: &'u mut BTreeMap<Root, RootUse>,
    r: Root,
    m: &mut Meter,
) -> Result<&'u mut RootUse, Error> {
    if !uses.contains_key(&r) {
        m.charge(USE_ENTRY_BYTES)?;
    }
    Ok(uses.entry(r).or_default())
}

fn not_lowered() -> Error {
    Unsupported::not_implemented(Feature::Lowering(LowerFamily::Access)).into()
}

/// Constant integer index `v` of `e`'s module, sign-extended from its width; `None` when not
/// constant.
fn const_index(e: &Entry, v: Value) -> Option<i64> {
    let Value::Const(c) = v else { return None };
    let k = e.module.consts().get(c)?;
    match (&k.value, e.module.types().get(k.ty)?) {
        (Const::Int(u), Type::Int(w)) => {
            let w = u32::from(*w).clamp(1, 64);
            let shift = 64 - w;
            Some(((*u << shift) as i64) >> shift)
        }
        (Const::Null, Type::Int(_)) => Some(0),
        _ => None,
    }
}

/// The byte displacement of `getelementptr src_ty, _, idx..` in `e`'s module under `layouts`:
/// the sum of its constant indices' displacements (`None` once it leaves `i64`), and the stride
/// of each dynamic index (one per array level; an index over zero-sized elements adds none). A
/// struct index that is not constant, or an element the layouts do not size, is `Lowering(Access)`.
pub(crate) fn gep_displacement(
    e: &Entry,
    layouts: &TypeLayouts,
    src_ty: TypeId,
    idx: &[Value],
) -> Result<(Option<i64>, Vec<u64>), Error> {
    // The exact constant displacement; `None` once it leaves `i128` (several huge indices),
    // which makes the offset unknown (`Top`) like any displacement outside `i64`.
    let mut konst: Option<i128> = Some(0);
    let mut strides: Vec<u64> = Vec::new();
    let mut cur = src_ty;
    for (n, &i) in idx.iter().enumerate() {
        let scale = if n == 0 {
            layouts.alloc_size(src_ty)
        } else {
            match e.module.types().get(cur) {
                Some(Type::Struct { fields, .. }) => {
                    let k = const_index(e, i).ok_or_else(not_lowered)?;
                    let k = usize::try_from(k).map_err(|_| not_lowered())?;
                    let off = layouts
                        .field_offsets(cur)
                        .and_then(|o| o.get(k).copied())
                        .ok_or_else(not_lowered)?;
                    konst = konst.and_then(|c| c.checked_add(i128::from(off)));
                    cur = *fields.get(k).ok_or_else(not_lowered)?;
                    continue;
                }
                Some(Type::Array { elem, .. } | Type::Vector { elem, .. }) => {
                    cur = *elem;
                    layouts.alloc_size(*elem)
                }
                _ => None,
            }
        }
        .ok_or_else(not_lowered)?;
        match const_index(e, i) {
            Some(k) => {
                konst = konst.and_then(|c| {
                    i128::from(k)
                        .checked_mul(i128::from(scale))
                        .and_then(|d| c.checked_add(d))
                })
            }
            None if scale == 0 => {}
            None => strides.push(scale),
        }
    }
    Ok((konst.and_then(|c| i64::try_from(c).ok()), strides))
}

/// The dense index of a pointer-typed argument or instruction.
const NONE: u32 = u32::MAX;

/// The analysis state.
struct Cx<'a> {
    e: &'a Entry,
    layouts: &'a TypeLayouts,
    values: ValueIndex,
    /// By argument then instruction (constants excluded): the pointer index or `NONE`.
    slot: Vec<u32>,
    /// The value of each pointer index.
    ptr_values: Vec<crate::air::ir::ValueId>,
    facts: Vec<PtrFact>,
    changes: Vec<u8>,
    sets: RootSets,
    const_facts: BTreeMap<ConstId, PtrFact>,
    /// The pointer sources of every pointer-typed `extractvalue`, sorted by instruction.
    extracts: Vec<(InstId, Extract)>,
    /// Store-to-load forwarding through pointer cells.
    cells: Cells,
    /// The entry's bindings and layouts, indexed once.
    abi: AbiIndex,
}

impl Cx<'_> {
    /// The position in `extracts` of the pointer-typed `extractvalue` `id`.
    fn extract_at(&self, id: InstId) -> Option<usize> {
        self.extracts.binary_search_by_key(&id, |(i, _)| *i).ok()
    }

    /// Source `j` of the `extractvalue` at position `at`.
    fn extract_source(&self, at: usize, j: usize) -> Option<Value> {
        self.extracts
            .get(at)
            .and_then(|(_, x)| x.sources.get(j))
            .copied()
    }

    /// The intersect call `j` whose primitive-data pointer the `extractvalue` at position `at`
    /// may read.
    fn extract_intersection(&self, at: usize, j: usize) -> Option<InstId> {
        self.extracts
            .get(at)
            .and_then(|(_, x)| x.intersections.get(j))
            .copied()
    }
}

/// The resolved sources of the pointer-typed `extractvalue` `id` in `extracts`.
fn find_extract(extracts: &[(InstId, Extract)], id: InstId) -> Option<&Extract> {
    let at = extracts.binary_search_by_key(&id, |(i, _)| *i).ok()?;
    extracts.get(at).map(|(_, x)| x)
}

/// Where the pointer an `extractvalue` reads was inserted: pointer values, whether an
/// undefined or null aggregate or a load of an aggregate supplies it, and the intersect calls
/// whose primitive-data pointer it is.
#[derive(Clone, Debug, Default)]
struct Extract {
    sources: Vec<Value>,
    null: bool,
    dynamic: bool,
    intersections: Vec<InstId>,
}

/// The ledger bytes of one entry of a map from `K` to `V`: twice the entry, which covers B-tree
/// node slack and a hash table's load factor and control bytes.
pub(crate) const fn map_entry_bytes<K, V>() -> usize {
    2 * (std::mem::size_of::<K>() + std::mem::size_of::<V>() + 1)
}

/// The ledger bytes of one index path of `n` indices held in a map key.
fn path_bytes(n: usize) -> usize {
    n.saturating_mul(std::mem::size_of::<u32>())
}

/// One `insertvalue` chain, read once from its top: for each index path, the first insertion
/// below the top (its depth from the top and the inserted value), and the aggregate the chain
/// starts from.
struct Chain {
    inserts: BTreeMap<Box<[u32]>, (u32, Value)>,
    base: Value,
}

/// The `insertvalue` chains the `extractvalue` resolution has read, by top instruction. Every
/// extract of the same aggregate shares one visit of its chain (an inlined callee that returns a
/// struct inserts each field once and the caller extracts each field once), so resolution costs
/// the chain once plus a lookup per extract instead of the chain per extract. The table is a
/// working table: [`Chains::release`] returns its ledger bytes when resolution ends.
#[derive(Default)]
struct Chains {
    read: HashMap<InstId, Chain>,
    bytes: usize,
}

impl Chains {
    fn charge(&mut self, n: usize, m: &mut Meter) -> Result<(), Error> {
        m.charge(n)?;
        self.bytes = self.bytes.saturating_add(n);
        Ok(())
    }

    /// Release every byte the table was charged.
    fn release(self, m: &mut Meter) {
        m.release(self.bytes);
    }

    /// The chain whose top is `top`, read on first use.
    fn chain(
        &mut self,
        body: &crate::air::ir::Body,
        top: InstId,
        m: &mut Meter,
    ) -> Result<&Chain, Error> {
        if !self.read.contains_key(&top) {
            self.charge(map_entry_bytes::<InstId, Chain>(), m)?;
            let mut inserts = BTreeMap::new();
            let mut v = Value::Inst(top);
            let mut depth = 0u32;
            while let Value::Inst(id) = v {
                let Some(InstKind::InsertValue { agg, val, indices }) =
                    body.insts.get(id).map(|i| &i.kind)
                else {
                    break;
                };
                m.tick(1)?;
                if !inserts.contains_key(&**indices) {
                    self.charge(
                        map_entry_bytes::<Box<[u32]>, (u32, Value)>() + path_bytes(indices.len()),
                        m,
                    )?;
                    inserts.insert(Box::<[u32]>::from(&**indices), (depth, *val));
                }
                depth = depth.saturating_add(1);
                v = *agg;
            }
            self.read.insert(top, Chain { inserts, base: v });
        }
        self.read
            .get(&top)
            .ok_or_else(|| invariant("an insertvalue chain was not recorded"))
    }

    /// Where the member at non-empty `path` of the chain topped by `top` comes from: the first
    /// insertion (from the top) whose indices are a prefix of `path` supplies it (with the rest
    /// of the path), and without one the chain's base does. An insertion that only partly
    /// overwrites the member (its indices extend `path`) above that one has no model.
    fn resolve(
        &mut self,
        body: &crate::air::ir::Body,
        top: InstId,
        path: &[u32],
        m: &mut Meter,
    ) -> Result<(Value, Box<[u32]>), Error> {
        let chain = self.chain(body, top, m)?;
        let mut hit: Option<(u32, Value, usize)> = None;
        for n in 1..=path.len() {
            m.tick(1)?;
            let Some(&(depth, val)) = path.get(..n).and_then(|p| chain.inserts.get(p)) else {
                continue;
            };
            if hit.is_none_or(|(d, _, _)| depth < d) {
                hit = Some((depth, val, n));
            }
        }
        let range = (
            std::ops::Bound::Excluded(path),
            std::ops::Bound::<&[u32]>::Unbounded,
        );
        for (k, &(depth, _)) in chain.inserts.range::<[u32], _>(range) {
            m.tick(1)?;
            if !k.starts_with(path) {
                break;
            }
            if hit.is_none_or(|(d, _, _)| depth < d) {
                return Err(not_lowered());
            }
        }
        Ok(match hit {
            Some((_, val, n)) => (val, path.get(n..).unwrap_or(&[]).into()),
            None => (chain.base, path.into()),
        })
    }
}

/// Resolve the pointer at `path` of aggregate `agg` through `insertvalue`, `extractvalue`,
/// `phi`, `select`, `freeze`, and aggregate constants (inlined callees pass structs of handles
/// by value). Each `(value, path)` is visited once, so phi cycles terminate; `insertvalue`
/// chains are read once through `chains`. The visited set is released on return.
fn resolve_extract(
    e: &Entry,
    body: &crate::air::ir::Body,
    agg: Value,
    path: &[u32],
    chains: &mut Chains,
    m: &mut Meter,
) -> Result<Extract, Error> {
    let mut seen_bytes = 0usize;
    let result = resolve_walk(e, body, agg, path, chains, &mut seen_bytes, m);
    m.release(seen_bytes);
    result
}

fn resolve_walk(
    e: &Entry,
    body: &crate::air::ir::Body,
    agg: Value,
    path: &[u32],
    chains: &mut Chains,
    seen_bytes: &mut usize,
    m: &mut Meter,
) -> Result<Extract, Error> {
    let mut out = Extract::default();
    let mut seen: std::collections::HashSet<(Value, Box<[u32]>)> = std::collections::HashSet::new();
    let mut work: Vec<(Value, Box<[u32]>)> = vec![(agg, path.into())];
    while let Some((v, path)) = work.pop() {
        m.tick(1)?;
        if seen.contains(&(v, path.clone())) {
            continue;
        }
        let entry = map_entry_bytes::<(Value, Box<[u32]>), ()>() + path_bytes(path.len());
        m.charge(entry)?;
        *seen_bytes = seen_bytes.saturating_add(entry);
        seen.insert((v, path.clone()));
        if path.is_empty() {
            let ty = e.module.value_ty(e.func, v);
            if !matches!(
                ty.and_then(|t| e.module.types().get(t)),
                Some(Type::Ptr { .. })
            ) {
                return Err(not_lowered());
            }
            if let Value::Const(c) = v {
                if matches!(
                    e.module.consts().get(c).map(|k| &k.value),
                    Some(Const::Undef | Const::Poison | Const::Null)
                ) {
                    out.null = true;
                    continue;
                }
            }
            out.sources.push(v);
            continue;
        }
        match v {
            Value::Const(c) => match e.module.consts().get(c).map(|k| &k.value) {
                Some(Const::Undef | Const::Poison | Const::Null) => out.null = true,
                Some(Const::Aggregate(elems)) => {
                    let (&first, rest) = path.split_first().ok_or_else(not_lowered)?;
                    let elem = *elems.get(first as usize).ok_or_else(not_lowered)?;
                    work.push((Value::Const(elem), rest.into()));
                }
                _ => return Err(not_lowered()),
            },
            Value::Arg(_) => return Err(not_lowered()),
            Value::Inst(id) => {
                let inst = body
                    .insts
                    .get(id)
                    .ok_or_else(|| invariant("an aggregate out of range"))?;
                match &inst.kind {
                    InstKind::InsertValue { .. } => {
                        work.push(chains.resolve(body, id, &path, m)?);
                    }
                    InstKind::ExtractValue { agg, indices } => {
                        let full: Vec<u32> = indices.iter().chain(path.iter()).copied().collect();
                        work.push((*agg, full.into()));
                    }
                    InstKind::Phi { incoming, .. } => {
                        for (v, _) in incoming.iter() {
                            work.push((*v, path.clone()));
                        }
                    }
                    InstKind::Select { t, f, .. } => {
                        work.push((*t, path.clone()));
                        work.push((*f, path));
                    }
                    InstKind::Freeze(v) => work.push((*v, path)),
                    InstKind::Load { .. } => out.dynamic = true,
                    // An intersection's primitive-data pointer: a pointer into the acceleration
                    // structure, which no Vulkan structure holds. It is the call's opaque root
                    // (`Root::Call`, [`primitive_data_root`]); `lower::raytracing` owns every
                    // reader of it and refuses each that reads it.
                    InstKind::Call(call) => match &call.callee {
                        Callee::Air {
                            op: AirIntrinsic::Intersect(i),
                            ..
                        } if i
                            .tags
                            .result()
                            .index(ResultField::PrimitiveData)
                            .is_some_and(|k| matches!(&*path, [only] if *only == k)) =>
                        {
                            out.intersections.push(id);
                        }
                        _ => return Err(not_lowered()),
                    },
                    _ => return Err(not_lowered()),
                }
            }
        }
    }
    out.sources.sort_unstable();
    out.sources.dedup();
    out.intersections.sort_unstable();
    out.intersections.dedup();
    Ok(out)
}

impl<'a> Cx<'a> {
    fn ty(&self, t: TypeId) -> Option<&'a Type> {
        self.e.module.types().get(t)
    }

    fn space_of(&self, t: TypeId) -> Option<AddrSpace> {
        match self.ty(t)? {
            Type::Ptr { space } => Some(*space),
            _ => None,
        }
    }

    /// Refuse vectors of pointers: no corpus row needs them and memory has no model for them.
    fn check_ty(&self, t: TypeId) -> Result<(), Error> {
        match self.ty(t) {
            Some(Type::Vector { elem, .. }) if matches!(self.ty(*elem), Some(Type::Ptr { .. })) => {
                Err(not_lowered())
            }
            _ => Ok(()),
        }
    }

    fn value_ty(&self, v: Value) -> Option<TypeId> {
        self.e.module.value_ty(self.e.func, v)
    }

    /// The address space of pointer value `v` (`None` for a non-pointer).
    fn space_of_value(&self, v: Value) -> Option<AddrSpace> {
        self.value_ty(v).and_then(|t| self.space_of(t))
    }

    fn ptr_index(&self, v: Value) -> Option<usize> {
        let id = match v {
            Value::Arg(_) | Value::Inst(_) => self.values.id(v)?,
            Value::Const(_) => return None,
        };
        match self.slot.get(id.index()).copied() {
            Some(NONE) | None => None,
            Some(i) => Some(i as usize),
        }
    }

    /// The current fact of pointer operand `v` (bottom for one not yet computed).
    fn get(&mut self, v: Value, m: &mut Meter) -> Result<PtrFact, Error> {
        match v {
            Value::Const(c) => self.const_fact(c, m),
            _ => Ok(self
                .ptr_index(v)
                .and_then(|i| self.facts.get(i).copied())
                .unwrap_or(PtrFact::BOTTOM)),
        }
    }

    /// The fact of a pointer constant, computed once. The constant graph is acyclic with
    /// children first, so the recursion depth is the nesting of constant expressions, which
    /// is walked with an explicit stack.
    fn const_fact(&mut self, c: ConstId, m: &mut Meter) -> Result<PtrFact, Error> {
        if let Some(f) = self.const_facts.get(&c) {
            return Ok(*f);
        }
        let mut stack = vec![c];
        while let Some(&top) = stack.last() {
            m.tick(1)?;
            if self.const_facts.contains_key(&top) {
                stack.pop();
                continue;
            }
            let k = self
                .e
                .module
                .consts()
                .get(top)
                .ok_or_else(|| invariant("a constant out of range"))?;
            // Pointer operands of an expression must be computed first.
            let pending: Vec<ConstId> = match &k.value {
                Const::Expr(ConstExpr::Gep { ops, .. }) => {
                    ops.first().copied().into_iter().collect()
                }
                Const::Expr(ConstExpr::Cast {
                    op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                    ops,
                }) => ops.to_vec(),
                _ => Vec::new(),
            }
            .into_iter()
            .filter(|o| !self.const_facts.contains_key(o))
            .collect();
            if !pending.is_empty() {
                stack.extend(pending);
                continue;
            }
            let fact = match &k.value {
                Const::Null | Const::Undef | Const::Poison => {
                    let s = self.sets.single(Root::Null, m)?;
                    PtrFact::at(s, 0)
                }
                Const::GlobalAddr(g) => {
                    let s = self.sets.single(Root::Global(*g), m)?;
                    PtrFact::at(s, 0)
                }
                Const::Expr(ConstExpr::Cast {
                    op: CastOp::IntToPtr,
                    ..
                }) => {
                    let s = self.sets.single(Root::Dynamic, m)?;
                    PtrFact::unknown(s)
                }
                Const::Expr(ConstExpr::Cast {
                    op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                    ops: [o],
                }) => {
                    if !matches!(self.ty(k.ty), Some(Type::Ptr { .. }))
                        || self.space_of(k.ty)
                            != self.e.module.consts().ty(*o).and_then(|t| self.space_of(t))
                    {
                        return Err(not_lowered());
                    }
                    self.const_facts.get(o).copied().unwrap_or(PtrFact::BOTTOM)
                }
                Const::Expr(ConstExpr::Gep { src_ty, ops, .. }) => {
                    let (base, idx) = ops.split_first().ok_or_else(not_lowered)?;
                    let base = self
                        .const_facts
                        .get(base)
                        .copied()
                        .unwrap_or(PtrFact::BOTTOM);
                    let idx: Vec<Value> = idx.iter().map(|&i| Value::Const(i)).collect();
                    self.gep(base, *src_ty, &idx)?
                }
                _ => return Err(not_lowered()),
            };
            m.charge(map_entry_bytes::<ConstId, PtrFact>())?;
            self.const_facts.insert(top, fact);
            stack.pop();
        }
        self.const_facts
            .get(&c)
            .copied()
            .ok_or_else(|| invariant("a pointer constant without a fact"))
    }

    /// A constant integer index, sign-extended from its width; `None` when not constant.
    fn const_index(&self, v: Value) -> Option<i64> {
        const_index(self.e, v)
    }

    /// The fact of `getelementptr src_ty, base, idx..`.
    fn gep(&self, base: PtrFact, src_ty: TypeId, idx: &[Value]) -> Result<PtrFact, Error> {
        let (delta, strides) = gep_displacement(self.e, self.layouts, src_ty, idx)?;
        let range = match (base.range, delta, strides.is_empty()) {
            (OffsetRange::Empty, _, _) => OffsetRange::Empty,
            (OffsetRange::Span { lo, hi }, Some(d), true) => {
                match (lo.checked_add(d), hi.checked_add(d)) {
                    (Some(lo), Some(hi)) => OffsetRange::Span { lo, hi },
                    _ => OffsetRange::Top,
                }
            }
            _ => OffsetRange::Top,
        };
        // Each dynamic term adds its stride (one per nested array level).
        let added = strides
            .iter()
            .try_fold(None::<Strides>, |acc, s| match acc {
                None => Strides::one(*s).map(Some),
                Some(acc) => acc.with(*s).map(Some),
            });
        let shape = match (base.shape, delta, added) {
            (Shape::Empty, _, _) => Shape::Empty,
            (_, None, _) | (_, _, None) => Shape::Top,
            (Shape::Exact(c), Some(d), Some(None)) => {
                c.checked_add(d).map_or(Shape::Top, Shape::Exact)
            }
            (Shape::Exact(c), Some(d), Some(Some(strides))) => c
                .checked_add(d)
                .map_or(Shape::Top, |base| Shape::Strided { base, strides }),
            (Shape::Strided { base, strides }, Some(d), Some(more)) => {
                match (
                    base.checked_add(d),
                    more.map_or(Some(strides), |m| strides.union(m)),
                ) {
                    (Some(base), Some(strides)) => Shape::Strided { base, strides },
                    _ => Shape::Top,
                }
            }
            (Shape::Top, _, _) => Shape::Top,
        };
        Ok(PtrFact {
            roots: base.roots,
            range,
            shape,
        })
    }

    /// The transfer function of pointer-typed instruction `id`.
    fn transfer(&mut self, id: InstId, inst: &Inst, m: &mut Meter) -> Result<PtrFact, Error> {
        // An element of a handle-array parameter is one of that parameter's handles: the
        // candidate stays the parameter and the offset names the element.
        if let InstKind::Load { ptr, .. } = &inst.kind {
            let f = self.get(*ptr, m)?;
            // The operand is not reached yet (layout order may put a use before its
            // definition); the operand's flow edge revisits this load once it is, so a
            // handle-array element never joins a premature `Dynamic`.
            if f == PtrFact::BOTTOM {
                return Ok(f);
            }
            if let [Root::Param(p)] = self.sets.get(f.roots) {
                let p = *p;
                if self.abi.handle_array(p).is_some() {
                    return Ok(f);
                }
                if let Some(t) = self.abi.buffer_table(p) {
                    return self.table_load(p, t, f, inst.ty, m);
                }
            }
            // A pointer loaded from a forwarding cell is one of the pointers stored at its slot.
            if let Some(me) = self.ptr_index(Value::Inst(id)) {
                if let Some(x) = self.cell_load(me as u32, f, m)? {
                    return Ok(x);
                }
            }
        }
        // An integer round trip of a handle-array element is that element's handle, as a
        // pointer load of it is.
        if let InstKind::Cast {
            op: CastOp::IntToPtr,
            v,
        } = &inst.kind
        {
            let e = self.e;
            let body = e.module.func(e.func).and_then(|f| f.body.as_ref());
            if let Some(ptr) = body.and_then(|b| round_trip(e, self.layouts, b, *v, inst.ty)) {
                let f = self.get(ptr, m)?;
                // As for a load: revisited through the flow edge once the operand is reached.
                if f == PtrFact::BOTTOM {
                    return Ok(f);
                }
                if let [Root::Param(p)] = self.sets.get(f.roots) {
                    if self.abi.handle_array(*p).is_some() {
                        return Ok(f);
                    }
                }
                // A pointer word read from a forwarding cell is one of the pointers stored at
                // its slot, as a pointer load of it is.
                let word = matches!(v, Value::Inst(l) if self.cells.word_loads.contains_key(l));
                if let (true, Some(me)) = (word, self.ptr_index(Value::Inst(id))) {
                    if let Some(x) = self.cell_load(me as u32, f, m)? {
                        return Ok(x);
                    }
                }
            }
        }
        Ok(match &inst.kind {
            InstKind::Alloca { .. } => {
                let s = self.sets.single(Root::Alloca(id), m)?;
                PtrFact::at(s, 0)
            }
            InstKind::Gep {
                src_ty,
                base,
                indices,
                ..
            } => {
                let b = self.get(*base, m)?;
                self.gep(b, *src_ty, indices)?
            }
            InstKind::Phi { incoming, .. } => {
                let mut f = PtrFact::BOTTOM;
                for (v, _) in incoming.iter() {
                    let o = self.get(*v, m)?;
                    f = self.join(f, o, m)?;
                }
                f
            }
            InstKind::Select { t, f, .. } => {
                let a = self.get(*t, m)?;
                let b = self.get(*f, m)?;
                self.join(a, b, m)?
            }
            InstKind::Freeze(v) => self.get(*v, m)?,
            InstKind::ExtractValue { .. } => {
                let at = self.extract_at(id);
                let (null, dynamic) = at
                    .and_then(|a| self.extracts.get(a))
                    .map_or((false, false), |(_, x)| (x.null, x.dynamic));
                let mut f = PtrFact::BOTTOM;
                let mut j = 0;
                while let Some(v) = at.and_then(|a| self.extract_source(a, j)) {
                    let o = self.get(v, m)?;
                    f = self.join(f, o, m)?;
                    j += 1;
                }
                // An intersection's primitive-data pointer: the call's opaque root, at an
                // unknown offset.
                let mut j = 0;
                while let Some(c) = at.and_then(|a| self.extract_intersection(a, j)) {
                    let s = self.sets.single(Root::Call(c), m)?;
                    f = self.join(f, PtrFact::unknown(s), m)?;
                    j += 1;
                }
                if null {
                    let s = self.sets.single(Root::Null, m)?;
                    f = self.join(f, PtrFact::at(s, 0), m)?;
                }
                if dynamic {
                    let s = self.sets.single(Root::Dynamic, m)?;
                    f = self.join(f, PtrFact::unknown(s), m)?;
                }
                f
            }
            InstKind::Load { .. }
            | InstKind::Cast {
                op: CastOp::IntToPtr,
                ..
            } => {
                let s = self.sets.single(Root::Dynamic, m)?;
                PtrFact::unknown(s)
            }
            InstKind::Cast {
                op: CastOp::Bitcast,
                v,
            } if self.value_ty(*v).and_then(|t| self.space_of(t)) == self.space_of(inst.ty) => {
                self.get(*v, m)?
            }
            InstKind::Call(call) => {
                let handle = match &call.callee {
                    Callee::Air { op, .. } => op.signature().ret_handle(),
                    _ => None,
                };
                let null = matches!(
                    &call.callee,
                    Callee::Air {
                        op: crate::air::intrinsics::AirIntrinsic::Texture(t),
                        ..
                    } if t.op == crate::air::intrinsics::TexOp::GetNull
                );
                match handle {
                    Some(_) if null => {
                        let s = self.sets.single(Root::Null, m)?;
                        PtrFact::at(s, 0)
                    }
                    Some(_) => {
                        let s = self.sets.single(Root::Call(id), m)?;
                        PtrFact::at(s, 0)
                    }
                    // An imageblock cell: its own root, at the cell's first byte.
                    None if is_imageblock_data(&inst.kind) => {
                        let s = self.sets.single(Root::Call(id), m)?;
                        PtrFact::at(s, 0)
                    }
                    // Another intrinsic's pointer (a ray-tracing data pointer, a function-table
                    // entry): an opaque root. Every such family is refused by its own route in
                    // `select`, so no access through it lowers.
                    None if matches!(call.callee, Callee::Air { .. } | Callee::Llvm { .. }) => {
                        let s = self.sets.single(Root::Call(id), m)?;
                        PtrFact::unknown(s)
                    }
                    None => return Err(not_lowered()),
                }
            }
            // A pointer out of an aggregate or vector, a cast between spaces, or any other
            // producer has no model.
            _ => return Err(not_lowered()),
        })
    }

    /// The fact of a pointer loaded through buffer table `p` at offsets `f`: the elements whose
    /// pointer the load may read, each at offset 0. A constant offset names one element; a constant
    /// range, the elements it overlaps within the table; an unknown offset, every element. A table
    /// read outside the table is undefined in Metal: its elements are dropped from the set (the
    /// access lowering keeps the index inside the table), and a read wholly outside it is the null
    /// pointer.
    fn table_load(
        &mut self,
        p: u32,
        t: BufferTable,
        f: PtrFact,
        ty: TypeId,
        m: &mut Meter,
    ) -> Result<PtrFact, Error> {
        let AbiInt::Const(count) = t.count else {
            return Err(not_lowered());
        };
        // The element is a pointer of the table's space; any other load reads its bytes.
        if self.space_of(ty) != Some(t.space) {
            return Err(not_lowered());
        }
        let size = self
            .layouts
            .store_size(ty)
            .and_then(|s| i64::try_from(s).ok())
            .filter(|&s| s > 0)
            .ok_or_else(not_lowered)?;
        let last = i64::from(count) - 1;
        let (lo, hi) = match f.range {
            OffsetRange::Empty => return Ok(PtrFact::BOTTOM),
            OffsetRange::Span { lo, hi } => {
                // A constant offset inside an element reads parts of two pointers.
                if lo == hi && lo.rem_euclid(size) != 0 {
                    return Err(not_lowered());
                }
                (lo.div_euclid(size).max(0), hi.div_euclid(size).min(last))
            }
            OffsetRange::Top => (0, last),
        };
        if lo > hi {
            let s = self.sets.single(Root::Null, m)?;
            return Ok(PtrFact::at(s, 0));
        }
        let n = usize::try_from(hi - lo + 1).map_err(|_| not_lowered())?;
        let mut roots = Vec::new();
        grow(&mut roots, n, m)?;
        for k in lo..=hi {
            let element = u32::try_from(k).map_err(|_| invariant("a table element index"))?;
            roots.push(Root::BufferElement { param: p, element });
        }
        // The working list dies here; the interned set is charged on its own.
        let scratch = roots.capacity().saturating_mul(std::mem::size_of::<Root>());
        let s = self.sets.intern(roots, m)?;
        m.release(scratch);
        Ok(PtrFact::at(s, 0))
    }

    fn join(&mut self, a: PtrFact, b: PtrFact, m: &mut Meter) -> Result<PtrFact, Error> {
        Ok(PtrFact {
            roots: self.sets.join(a.roots, b.roots, m)?,
            range: a.range.join(b.range),
            shape: a.shape.join(b.shape),
        })
    }
}

/// The byte size of a value of type `t` in memory (its store size).
fn store_size(layouts: &TypeLayouts, t: TypeId) -> Option<u64> {
    layouts.store_size(t)
}

/// Whether an access of type `a` at offset shape `shape` into an object of type `obj` (an array
/// of the allocated type) is a typed access chain: an exact subobject, or one dynamic element index
/// over each of the array levels whose element sizes are the strides (`alloca::path`, which the
/// access family lowers with).
fn typed_access(
    e: &Entry,
    layouts: &TypeLayouts,
    obj: TypeId,
    count: u64,
    shape: Shape,
    a: TypeId,
) -> bool {
    alloca_shape(shape).is_some_and(|s| alloca::path(e, layouts, obj, count, s, a).is_some())
}

/// A pointer's offset shape as a typed access resolves it, `None` for no single shape.
fn alloca_shape(shape: Shape) -> Option<alloca::Shape> {
    match shape {
        Shape::Exact(c) => Some(alloca::Shape::Exact(c)),
        Shape::Strided { base, strides } => Some(alloca::Shape::Strided { base, strides }),
        Shape::Empty | Shape::Top => None,
    }
}

/// The handle class of the parameter entry binding `b` binds when it is an array of resource
/// handles (`array<texture2d<T>, N>` or `array_ref<texture2d<T>>`): an ABI handle role whose
/// parameter is a thread-space pointer to its handles (the ABI's role check admits a thread
/// pointer for a handle role only as such a table). Each load through it reads one handle. The IR
/// type decides, not the location count: an `array_ref` whose count a function constant makes 1
/// is still a pointer to one handle, and its parameter no handle. By parameter, the memory plan's
/// index holds the answer (`AbiIndex::handle_array`).
pub fn handle_array_of(e: &Entry, b: &ParamBinding) -> Option<HandleClass> {
    let p = b.param;
    let h = b.role.handle()?;
    let func = e.module.func(e.func)?;
    let ty = match e.module.types().get(func.ty)? {
        Type::Function { params, .. } => *params.get(p as usize)?,
        _ => return None,
    };
    (e.module.types().ptr_space(ty)? == AddrSpace::Thread).then(|| HandleClass::of_abi(h))
}

/// The pointer operand of the load that an `inttoptr` of `v` to a pointer of type `ty` round-trips:
/// `v` is a plain load of an integer as wide as that pointer, so the `inttoptr` is the pointer the
/// loaded bytes hold. Through a handle-array element it is that element's handle (`Cx::transfer`);
/// anywhere else the integer is an address like any other.
pub fn round_trip(
    e: &Entry,
    layouts: &TypeLayouts,
    body: &crate::air::ir::Body,
    v: Value,
    ty: TypeId,
) -> Option<Value> {
    let Value::Inst(l) = v else { return None };
    let load = body.insts.get(l)?;
    let InstKind::Load {
        ptr,
        volatile: false,
        atomic: None,
        ..
    } = &load.kind
    else {
        return None;
    };
    if !matches!(e.module.types().get(load.ty), Some(Type::Int(_))) {
        return None;
    }
    let size = layouts.store_size(load.ty)?;
    (layouts.store_size(ty)? == size).then_some(*ptr)
}

/// A buffer-table parameter: a buffer or argument-buffer role passed as an `array_ref`, whose LLVM
/// parameter is a thread-space pointer to `count` buffer pointers in `space`. Element `k` is the
/// buffer bound to argument-table slot `first + k`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferTable {
    pub first: AbiInt,
    pub count: AbiInt,
    /// The address space of the element pointers.
    pub space: AddrSpace,
    pub argument_buffer: bool,
}

/// The buffer table the parameter entry binding `b` binds is, if it is one: a buffer or
/// argument-buffer role whose parameter type is a thread-space pointer (the role check admits a
/// thread pointer for these roles only as an `array_ref`). By parameter, the memory plan's index
/// holds the answer (`AbiIndex::buffer_table`).
pub fn buffer_table_of(e: &Entry, b: &ParamBinding) -> Option<BufferTable> {
    let p = b.param;
    let (arg, argument_buffer) = match &b.role {
        ParamRole::Buffer(arg) => (arg, false),
        ParamRole::ArgumentBuffer(arg) => (arg, true),
        _ => return None,
    };
    let func = e.module.func(e.func)?;
    let ty = match e.module.types().get(func.ty)? {
        Type::Function { params, .. } => *params.get(p as usize)?,
        _ => return None,
    };
    (e.module.types().ptr_space(ty)? == AddrSpace::Thread).then_some(BufferTable {
        first: arg.location.index,
        count: arg.location.count,
        space: arg.space,
        argument_buffer,
    })
}

/// Whether root `r` is an imageblock cell: the pointer an `air.imageblock_data` call returns, at
/// offset 0 of its cell.
pub fn imageblock_cell(e: &Entry, r: Root) -> bool {
    let Root::Call(id) = r else { return false };
    e.module
        .func(e.func)
        .and_then(|f| f.body.as_ref())
        .and_then(|b| b.insts.get(id))
        .is_some_and(|inst| is_imageblock_data(&inst.kind))
}

/// Whether root `r` is an intersection's primitive-data pointer: the opaque root of the
/// `air.intersect` call whose result holds it. Nothing in Vulkan holds the data it points to, so
/// `lower::raytracing` refuses every reader of it but the moves.
pub fn primitive_data_root(e: &Entry, r: Root) -> bool {
    let Root::Call(id) = r else { return false };
    e.module
        .func(e.func)
        .and_then(|f| f.body.as_ref())
        .and_then(|b| b.insts.get(id))
        .is_some_and(|inst| {
            matches!(
                &inst.kind,
                InstKind::Call(call) if matches!(
                    call.callee,
                    Callee::Air {
                        op: AirIntrinsic::Intersect(_),
                        ..
                    }
                )
            )
        })
}

/// Whether `kind` is an `air.imageblock_data` call.
fn is_imageblock_data(kind: &InstKind) -> bool {
    let InstKind::Call(call) = kind else {
        return false;
    };
    matches!(
        call.callee,
        Callee::Air {
            op: AirIntrinsic::Imageblock(crate::air::intrinsics::Imageblock::Data),
            ..
        }
    )
}

/// The parameter whose buffer object root `r` is: `Param(p)` of a buffer or argument buffer
/// passed by pointer, or an element of a buffer table.
fn buffer_object_param(abi: &AbiIndex, r: Root) -> Option<u32> {
    match r {
        Root::Param(p) if abi.buffer_table(p).is_none() => Some(p),
        Root::BufferElement { param, .. } => Some(param),
        _ => None,
    }
}

/// The argument-buffer member a pointer of `fact` addresses exactly: one root, a parameter (or
/// buffer-table element) of an argument buffer with a layout, at a constant offset that starts one
/// of the member's elements within the layout ([`member_at`](AbiIndex::member_at)). A load of a
/// pointer there is the member's handle (`Walk::seed_loaded`).
fn exact_member<'e>(
    e: &'e Entry,
    abi: &AbiIndex,
    roots: &[Root],
    fact: PtrFact,
) -> Option<&'e Member> {
    exact_member_at(e, abi, roots, fact.range)
}

/// [`exact_member`] of a pointer whose roots are `roots` and whose offset range is `range`.
fn exact_member_at<'e>(
    e: &'e Entry,
    abi: &AbiIndex,
    roots: &[Root],
    range: OffsetRange,
) -> Option<&'e Member> {
    let (Some(off), [r]) = (range.exact(), roots) else {
        return None;
    };
    let p = buffer_object_param(abi, *r)?;
    let ParamRole::ArgumentBuffer(arg) = &abi.binding(e, p)?.role else {
        return None;
    };
    abi.member_at(&e.structs, arg.layout?, u64::try_from(off).ok()?)
}

/// The handle load of an indirect-command object: a plain load of `bytes` bytes (a pointer's)
/// through a pointer whose roots are `roots` and whose offset range is `range` reads exactly one
/// command-buffer or pipeline-state member's field at a translation-time constant offset in its
/// argument buffer's first element, or `None`. The member's field is its word: the command buffer's
/// resource ID (its records lie at `indirect_commands::records_address` of it), or the pipeline
/// template's host-registered token.
pub(crate) fn object_handle_load(
    e: &Entry,
    abi: &AbiIndex,
    roots: &[Root],
    range: OffsetRange,
    bytes: u64,
) -> Option<AbiHandle> {
    let mem = exact_member_at(e, abi, roots, range)?;
    let h = mem.handle().filter(|h| is_object(*h))?;
    (bytes == u64::from(member_element_size(mem))).then_some(h)
}

/// The byte size of one element of argument-buffer member `mem`, the stride between its
/// elements: its own size for a member array (`array_len` elements) or a single member, and one
/// handle's for an `array<texture…, N>` or `array<sampler, N>` member, which
/// `air.struct_type_info` records as one member of all `N` handles whose role's location count is
/// `N`.
pub(crate) fn member_element_size(mem: &crate::air::abi::Member) -> u32 {
    if mem.array_len >= 1 {
        return mem.size;
    }
    let count = match &mem.kind {
        MemberKind::Indirect(arg) => match &arg.role {
            ParamRole::Texture(t) => Some(t.location.count),
            ParamRole::Sampler { location } => Some(location.count),
            _ => None,
        },
        _ => None,
    };
    match count {
        Some(AbiInt::Const(n)) if n > 1 && mem.size.is_multiple_of(n) => mem.size / n,
        _ => mem.size,
    }
}

/// A union-find over pointer indices with a class per set: the shared metered union-find
/// (`memory::union_find`, joined by size, a work unit per find step), each set's class and label at
/// its representative.
struct Classes {
    uf: UnionFind,
    class: Vec<Class>,
    /// Each set's label: the index that was its representative when the union-find joined the
    /// second set under the first, which a conflict that a union meets reports.
    label: Vec<u32>,
    /// The first value a conflicting seed reached, for the refusal's location.
    conflict: Option<usize>,
    /// The ledger bytes `class` and `label` hold.
    charged: usize,
}

impl Classes {
    /// `n` indices, each a set of its own, of class `Unknown`.
    fn new(n: usize, m: &mut Meter) -> Result<Classes, Error> {
        let uf = UnionFind::with_len(n, m)?;
        let mut charged = 0;
        let mut class = Vec::new();
        grow_working(&mut class, n, &mut charged, m)?;
        class.resize(n, Class::Unknown);
        let mut label = Vec::new();
        grow_working(&mut label, n, &mut charged, m)?;
        label.extend(0..uf_len(n)?);
        Ok(Classes {
            uf,
            class,
            label,
            conflict: None,
            charged,
        })
    }

    /// The ledger bytes the tables hold, which `analyze` returns when it drops them.
    fn charged(&self) -> usize {
        self.charged.saturating_add(self.uf.charged())
    }

    /// The representative of `i`'s set.
    fn find(&mut self, i: usize, m: &mut Meter) -> Result<usize, Error> {
        Ok(self.uf.find(uf_len(i)?, m)? as usize)
    }

    /// The class of `i`'s set.
    fn class_of(&mut self, i: usize, m: &mut Meter) -> Result<Class, Error> {
        let r = self.find(i, m)?;
        Ok(self.class.get(r).copied().unwrap_or(Class::Unknown))
    }

    fn seed(&mut self, i: usize, c: Class, m: &mut Meter) -> Result<(), Error> {
        let r = self.find(i, m)?;
        if let Some(slot) = self.class.get_mut(r) {
            *slot = slot.join(c);
            if *slot == Class::Conflict && self.conflict.is_none() {
                self.conflict = Some(i);
            }
        }
        Ok(())
    }

    fn union(&mut self, a: usize, b: usize, m: &mut Meter) -> Result<(), Error> {
        let (ra, rb) = (self.find(a, m)?, self.find(b, m)?);
        if ra == rb {
            return Ok(());
        }
        let at = |v: &Vec<Class>, r: usize| v.get(r).copied().unwrap_or(Class::Unknown);
        let joined = at(&self.class, ra).join(at(&self.class, rb));
        let label = self
            .label
            .get(ra)
            .copied()
            .ok_or_else(|| invariant("a class index out of range"))?;
        let r = self.uf.join(uf_len(ra)?, uf_len(rb)?, m)? as usize;
        match (self.class.get_mut(r), self.label.get_mut(r)) {
            (Some(c), Some(l)) => {
                *c = joined;
                *l = label;
            }
            _ => return Err(invariant("a class index out of range")),
        }
        if joined == Class::Conflict && self.conflict.is_none() {
            self.conflict = Some(label as usize);
        }
        Ok(())
    }
}

/// A pointer index as the union-find's.
fn uf_len(i: usize) -> Result<u32, Error> {
    u32::try_from(i).map_err(|_| invariant("more pointer values than u32"))
}

/// Run the provenance analysis over `e`.
pub fn analyze(e: &Entry, layouts: &TypeLayouts, m: &mut Meter) -> Result<Provenance, Error> {
    let func = e
        .module
        .func(e.func)
        .ok_or_else(|| invariant("the entry function is missing"))?;
    let body = func
        .body
        .as_ref()
        .ok_or_else(|| invariant("the entry function has no body"))?;
    let values = ValueIndex::new(func.params.len(), body.insts.len(), e.module.consts().len())
        .ok_or_else(|| invariant("the entry's value count exceeds u32"))?;
    let local = func.params.len() + body.insts.len();
    let mut param_tys: Vec<TypeId> = Vec::new();
    match e.module.types().get(func.ty) {
        Some(Type::Function { params, .. }) => {
            m.reserve(&mut param_tys, params.len())?;
            param_tys.extend_from_slice(params);
        }
        _ => return Err(invariant("the entry's type is not a function type")),
    }

    // Number the pointer-typed arguments and instructions.
    let is_ptr = |t: TypeId| matches!(e.module.types().get(t), Some(Type::Ptr { .. }));
    let count = param_tys.iter().filter(|&&t| is_ptr(t)).count()
        + body.insts.values().filter(|i| is_ptr(i.ty)).count();
    let mut slot: Vec<u32> = Vec::new();
    m.reserve(&mut slot, local)?;
    let mut ptr_values: Vec<crate::air::ir::ValueId> = Vec::new();
    m.reserve(&mut ptr_values, count)?;
    let locals = param_tys
        .iter()
        .map(|&t| is_ptr(t))
        .chain(body.insts.values().map(|i| is_ptr(i.ty)));
    for (i, ptr) in locals.enumerate() {
        slot.push(if ptr {
            let id = crate::air::ir::ValueId::from_index(i)
                .ok_or_else(|| invariant("a value out of range"))?;
            ptr_values.push(id);
            (ptr_values.len() - 1) as u32
        } else {
            NONE
        });
    }
    let n = ptr_values.len();
    let mut facts = Vec::new();
    m.reserve(&mut facts, n)?;
    facts.resize(n, PtrFact::BOTTOM);
    let mut changes = Vec::new();
    m.reserve(&mut changes, n)?;
    changes.resize(n, 0u8);
    let mut cx = Cx {
        e,
        layouts,
        values,
        slot,
        ptr_values,
        facts,
        changes,
        sets: RootSets::new(),
        const_facts: BTreeMap::new(),
        extracts: Vec::new(),
        cells: Cells::default(),
        abi: AbiIndex::of(e, m)?,
    };
    for &t in &param_tys {
        cx.check_ty(t)?;
    }

    // Flow edges: pointer operand -> the pointer instructions that merge or offset it (CSR).
    let mut out_deg: Vec<u32> = Vec::new();
    m.reserve(&mut out_deg, n + 1)?;
    out_deg.resize(n + 1, 0);
    // Pointers read out of aggregates: their sources, found once (SSA structure is static),
    // pushed in instruction order so the table is sorted.
    let pointer_extract = |inst: &Inst| {
        matches!(inst.kind, InstKind::ExtractValue { .. })
            && matches!(e.module.types().get(inst.ty), Some(Type::Ptr { .. }))
    };
    let extract_count = body.insts.values().filter(|i| pointer_extract(i)).count();
    m.reserve(&mut cx.extracts, extract_count)?;
    let mut chains = Chains::default();
    let resolved = (|| {
        for (id, inst) in body.insts.iter() {
            if let InstKind::ExtractValue { agg, indices } = &inst.kind {
                if cx.space_of(inst.ty).is_some() {
                    let x =
                        resolve_extract(e, body, *agg, indices, &mut chains, m).map_err(|err| {
                            match block_of(body, id) {
                                Some(b) => located(err, e, b, id),
                                None => err,
                            }
                        })?;
                    cx.extracts.push((id, x));
                }
            }
        }
        Ok::<(), Error>(())
    })();
    chains.release(m);
    resolved?;
    let extracts = &cx.extracts;
    let flow = |id: InstId, inst: &Inst, f: &mut dyn FnMut(Value)| match &inst.kind {
        InstKind::ExtractValue { .. } => {
            if let Some(x) = find_extract(extracts, id) {
                x.sources.iter().for_each(|v| f(*v));
            }
        }
        InstKind::Gep { base, .. } => f(*base),
        // A load through a handle-array parameter reads its operand's fact (`transfer`), and so
        // does an integer round trip of one.
        InstKind::Load { ptr, .. } => f(*ptr),
        InstKind::Cast {
            op: CastOp::IntToPtr,
            v,
        } => {
            if let Some(ptr) = round_trip(e, layouts, body, *v, inst.ty) {
                f(ptr)
            }
        }
        InstKind::Phi { incoming, .. } => incoming.iter().for_each(|(v, _)| f(*v)),
        InstKind::Select { t, f: e, .. } => {
            f(*t);
            f(*e)
        }
        InstKind::Freeze(v)
        | InstKind::Cast {
            op: CastOp::Bitcast,
            v,
        } => f(*v),
        _ => {}
    };
    let at = |err: Error, id: InstId| match block_of(body, id) {
        Some(b) => located(err, e, b, id),
        None => err,
    };
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        cx.check_ty(inst.ty).map_err(|err| at(err, id))?;
        if cx.space_of(inst.ty).is_none() {
            continue;
        }
        flow(id, inst, &mut |v| {
            if let Some(src) = cx.ptr_index(v) {
                if let Some(d) = out_deg.get_mut(src) {
                    *d += 1;
                }
            }
        });
    }
    let mut start: Vec<u32> = Vec::new();
    m.reserve(&mut start, n + 1)?;
    let mut acc = 0u32;
    for d in &out_deg {
        start.push(acc);
        acc = acc.saturating_add(*d);
    }
    let mut users: Vec<u32> = Vec::new();
    m.reserve(&mut users, acc as usize)?;
    users.resize(acc as usize, 0);
    let mut fill: Vec<u32> = Vec::new();
    m.reserve(&mut fill, start.len())?;
    fill.extend_from_slice(&start);
    for (id, inst) in body.insts.iter() {
        if cx.space_of(inst.ty).is_none() {
            continue;
        }
        let Some(dst) = cx.ptr_index(Value::Inst(id)) else {
            continue;
        };
        flow(id, inst, &mut |v| {
            if let Some(src) = cx.ptr_index(v) {
                if let Some(at) = fill.get_mut(src) {
                    if let Some(u) = users.get_mut(*at as usize) {
                        *u = dst as u32;
                    }
                    *at += 1;
                }
            }
        });
    }
    drop(fill);

    // The pointer stores and loads forwarding reads, found once; a store of a pointer's bits stores
    // that pointer when its cells forward.
    let bit_stores = cells::bit_stores(&cx, body, m)?;
    grow(&mut cx.cells.stores, bit_stores.len(), m)?;
    for &(_, _, p, ptr) in &bit_stores {
        cx.cells.stores.push((p, ptr));
    }
    cx.cells.bit_stores = bit_stores;
    // And a pointer-word load is a load of the pointer its `inttoptr`s are.
    let word_loads = cells::word_loads(&cx, body, m)?;
    for (addr, casts) in word_loads.values() {
        for &c in casts {
            if let Some(me) = cx.ptr_index(Value::Inst(c)) {
                grow(&mut cx.cells.loads, 1, m)?;
                cx.cells.loads.push((me as u32, *addr));
            }
        }
    }
    cx.cells.word_loads = word_loads;
    // And a copy of pointer bits is a slot copy of their 8 bytes where its cells forward.
    let bit_copies = cells::bit_copies(&cx, body, m)?;
    grow(&mut cx.cells.copies, bit_copies.len(), m)?;
    for &(l, s) in &bit_copies {
        if let (Some(InstKind::Load { ptr: src, .. }), Some(InstKind::Store { ptr: dst, .. })) = (
            body.insts.get(l).map(|i| &i.kind),
            body.insts.get(s).map(|i| &i.kind),
        ) {
            cx.cells.copies.push((*dst, *src, 8));
        }
    }
    cx.cells.bit_copies = bit_copies;
    for (id, inst) in body.insts.iter() {
        match &inst.kind {
            InstKind::Store { val, ptr, .. } if cx.space_of_value(*val).is_some() => {
                grow(&mut cx.cells.stores, 1, m)?;
                cx.cells.stores.push((*val, *ptr));
            }
            InstKind::Load { ptr, .. } if cx.space_of(inst.ty).is_some() => {
                if let Some(me) = cx.ptr_index(Value::Inst(id)) {
                    grow(&mut cx.cells.loads, 1, m)?;
                    cx.cells.loads.push((me as u32, *ptr));
                }
            }
            InstKind::Call(call) => {
                if let Some(copy) = cells::constant_copy(&cx, call) {
                    grow(&mut cx.cells.copies, 1, m)?;
                    cx.cells.copies.push(copy);
                }
                // A zero fill writes null over the cell slots it covers.
                if let Some(fill) = cells::zero_fill(&cx, call) {
                    grow(&mut cx.cells.zero_fills, 1, m)?;
                    cx.cells.zero_fills.push(fill);
                }
            }
            _ => {}
        }
    }

    // The fixpoint, in rounds: every alloca forwards until the verification of a round's facts
    // opens it; a second failed verification opens every alloca.
    let flow = Flow {
        start: &start,
        users: &users,
    };
    let mut failures = 0u32;
    loop {
        fixpoint(&mut cx, &flow, &param_tys, body, func.params.len(), m).map_err(|(err, id)| {
            match id {
                Some(id) => at(err, id),
                None => err,
            }
        })?;
        let failed = cells::verify(&mut cx, body, m)?;
        if failed.is_empty() {
            break;
        }
        failures += 1;
        if failures >= 2 {
            cx.cells.facts.all_open = true;
        } else {
            for a in failed {
                m.charge(map_entry_bytes::<InstId, ()>())?;
                cx.cells.facts.open.insert(a);
            }
        }
    }
    cx.cells.reset(m);
    drop((users, start, out_deg));

    // Uses and classes, the classes' tables returned to the ledger once resolved.
    let mut classes = Classes::new(n, m)?;
    let result = classify(e, body, cx, &mut classes, m);
    m.release(classes.charged());
    result
}

/// [`analyze`]'s pass over every use of the pointers `cx` holds facts for: the uses, the handle
/// classes (in `classes`), and the merges.
fn classify(
    e: &Entry,
    body: &Body,
    mut cx: Cx<'_>,
    classes: &mut Classes,
    m: &mut Meter,
) -> Result<Provenance, Error> {
    let (values, n) = (cx.values, cx.ptr_values.len());
    let mut uses: BTreeMap<Root, RootUse> = BTreeMap::new();
    let mut dynamic_deref = None;

    // ABI seeds on parameters, and on the removed textures and samplers the body still reads.
    for b in e.ep.params.iter().chain(&e.absent_handles) {
        let v = Value::Arg(crate::air::ir::ArgIdx::new(b.param));
        let Some(p) = cx.ptr_index(v) else { continue };
        match (&b.role, b.role.handle()) {
            // A handle array is a pointer to its handles (bytes); loads through it are handles.
            (_, Some(_)) if cx.abi.handle_array(b.param).is_some() => {
                classes.seed(p, Class::Byte, m)?
            }
            (_, Some(h)) => classes.seed(p, Class::Handle(HandleClass::of_abi(h)), m)?,
            (
                ParamRole::Buffer(_) | ParamRole::ArgumentBuffer(_) | ParamRole::Threadgroup(_),
                None,
            ) => classes.seed(p, Class::Byte, m)?,
            _ => {}
        }
    }
    let static_sampler = |g: GlobalId| e.static_samplers.iter().any(|(s, _)| *s == g);
    // The argument buffers whose layout holds an indirect-command object, or its address; the
    // layout tables only when some argument buffer has a layout.
    let layouts = if e
        .ep
        .params
        .iter()
        .any(|b| matches!(&b.role, ParamRole::ArgumentBuffer(arg) if arg.layout.is_some()))
    {
        ObjectLayouts::of(&e.structs, m)?
    } else {
        ObjectLayouts::default()
    };
    let mut objects = Vec::new();
    for b in &e.ep.params {
        let table = cx.abi.buffer_table(b.param).is_some();
        if let Some(o) = ObjectBuffer::of(b, table, &e.structs, &layouts) {
            m.grow(&mut objects, 1)?;
            objects.push(o);
        }
    }

    let mut w = Walk {
        cx: &mut cx,
        classes,
        uses: &mut uses,
        dynamic_deref: &mut dynamic_deref,
        static_sampler: &static_sampler,
        slots: Vec::new(),
        objects,
        layouts,
        object_address: false,
        marked: HashMap::new(),
    };
    let mut operands = Vec::new();
    for (b, block) in body.blocks.iter() {
        for id in block.inst_ids() {
            m.tick(1)?;
            let inst = body
                .insts
                .get(id)
                .ok_or_else(|| invariant("an instruction out of range"))?;
            w.inst((b, id), inst, &mut operands, m)
                .map_err(|err| located(err, e, b, id))?;
        }
    }
    // A cell slot's loads are the pointers stored there: one class.
    w.join_slot_classes(m)?;
    m.release(w.marked.len().saturating_mul(MARKED_ENTRY_BYTES));
    // An object's bytes reached through a run-time address, which the rule at each access cannot
    // see: once an object-holding argument buffer's address escapes, or a read may load the address
    // of an object's memory, every run-time dereference is refused.
    let escaped = w.uses.iter().any(|(r, u)| {
        u.escapes
            && buffer_object_param(&w.cx.abi, *r).is_some_and(|p| {
                w.objects
                    .iter()
                    .any(|o| o.param == p && o.reaches_objects())
            })
    });
    if escaped || w.object_address {
        if let Some((b, id)) = *w.dynamic_deref {
            return Err(located(
                Unsupported::not_implemented(Feature::ArgumentBufferObjects).into(),
                e,
                b,
                id,
            ));
        }
    }
    // Merges: every root of a multi-root set is merged. Each set is walked once, however many
    // pointers name it, one work unit per pointer and per root.
    let sets = w.cx.sets.len();
    m.charge(sets)?;
    let mut seen = vec![false; sets];
    for f in w.cx.facts.iter() {
        m.tick(1)?;
        let roots = w.cx.sets.get(f.roots);
        match seen.get_mut(f.roots.index()) {
            Some(s) if !*s && roots.len() > 1 => *s = true,
            _ => continue,
        }
        for r in roots {
            m.tick(1)?;
            root_use(w.uses, *r, m)?.merged = true;
        }
    }
    m.release(sets);
    if let Some(i) = classes.conflict {
        let at = cx
            .ptr_values
            .get(i)
            .and_then(|v| values.value(*v))
            .and_then(|v| match v {
                Value::Inst(id) => Some(id),
                _ => None,
            });
        let err: Error = Unsupported::not_representable(Feature::HandleAsPointer).into();
        return Err(
            match at.and_then(|id| block_of(body, id).map(|b| (b, id))) {
                Some((b, id)) => located(err, e, b, id),
                None => err,
            },
        );
    }
    let mut resolved: Vec<Class> = Vec::new();
    m.reserve(&mut resolved, n)?;
    for i in 0..n {
        resolved.push(classes.class_of(i, m)?);
    }
    let mut facts = Vec::new();
    m.reserve(&mut facts, n)?;
    facts.extend(cx.ptr_values.iter().copied().zip(cx.facts.iter().copied()));
    Ok(Provenance {
        values,
        sets: cx.sets,
        facts,
        const_facts: cx.const_facts,
        classes: resolved,
        uses,
        dynamic_deref,
        cells: cx.cells.facts,
        abi: cx.abi,
    })
}

/// The flow edges of the fixpoint (CSR): pointer operand -> the pointer instructions that merge
/// or offset it.
struct Flow<'f> {
    start: &'f [u32],
    users: &'f [u32],
}

/// One round of the fixpoint from bottom: parameters are their own roots at offset 0, then a
/// worklist over pointer instructions, seeded in layout order, until no fact changes and a store
/// pass into the forwarding cells finds nothing new. A refusal names the instruction whose transfer
/// raised it.
fn fixpoint(
    cx: &mut Cx<'_>,
    flow: &Flow<'_>,
    param_tys: &[TypeId],
    body: &crate::air::ir::Body,
    first_inst: usize,
    m: &mut Meter,
) -> Result<(), (Error, Option<InstId>)> {
    let bare = |err: Error| (err, None);
    let metered = |err: crate::air::error::Error| (Error::from(err), None);
    cx.facts.fill(PtrFact::BOTTOM);
    cx.changes.fill(0);
    cx.cells.reset(m);
    for (i, _) in param_tys.iter().enumerate() {
        let v = Value::Arg(crate::air::ir::ArgIdx::new(i as u32));
        if let Some(p) = cx.ptr_index(v) {
            let s = cx.sets.single(Root::Param(i as u32), m).map_err(bare)?;
            if let Some(f) = cx.facts.get_mut(p) {
                *f = PtrFact::at(s, 0);
            }
        }
    }
    let n = cx.ptr_values.len();
    let mut queued: Vec<bool> = Vec::new();
    m.reserve(&mut queued, n).map_err(metered)?;
    queued.resize(n, false);
    // Each pointer index is queued at most once at a time, so `n` entries never grow.
    let mut work: std::collections::VecDeque<u32> = std::collections::VecDeque::new();
    let work_bytes = n.saturating_mul(std::mem::size_of::<u32>());
    m.charge(work_bytes).map_err(metered)?;
    work.try_reserve(n).map_err(|err| {
        bare(Error::Internal(InternalError::new(
            InternalKind::AllocationFailed,
            format!("reserving the provenance worklist: {err}"),
        )))
    })?;
    for (i, v) in cx.ptr_values.iter().enumerate() {
        if v.index() >= first_inst {
            work.push_back(i as u32);
            if let Some(q) = queued.get_mut(i) {
                *q = true;
            }
        }
    }
    let queue = |i: u32, work: &mut std::collections::VecDeque<u32>, queued: &mut Vec<bool>| {
        if let Some(q) = queued.get_mut(i as usize) {
            if !*q {
                *q = true;
                work.push_back(i);
            }
        }
    };
    loop {
        while let Some(i) = work.pop_front() {
            m.tick(1).map_err(metered)?;
            let i = i as usize;
            if let Some(q) = queued.get_mut(i) {
                *q = false;
            }
            let vid = *cx
                .ptr_values
                .get(i)
                .ok_or_else(|| bare(invariant("a pointer index")))?;
            let Some(Value::Inst(id)) = cx.values.value(vid) else {
                continue;
            };
            let inst = body
                .insts
                .get(id)
                .ok_or_else(|| bare(invariant("an instruction out of range")))?;
            let mut new = cx.transfer(id, inst, m).map_err(|err| (err, Some(id)))?;
            let old = cx.facts.get(i).copied().unwrap_or(PtrFact::BOTTOM);
            // Monotone: never lose what an earlier visit found.
            new = cx.join(old, new, m).map_err(bare)?;
            if new == old {
                continue;
            }
            let changes = cx.changes.get(i).copied().unwrap_or(0).saturating_add(1);
            if changes > WIDEN_AFTER {
                if new.range != OffsetRange::Empty {
                    new.range = OffsetRange::Top;
                }
                if new.shape != Shape::Empty {
                    new.shape = Shape::Top;
                }
            }
            if let Some(c) = cx.changes.get_mut(i) {
                *c = changes;
            }
            if let Some(f) = cx.facts.get_mut(i) {
                *f = new;
            }
            let (s, t) = (
                flow.start.get(i).copied().unwrap_or(0) as usize,
                flow.start.get(i + 1).copied().unwrap_or(0) as usize,
            );
            for &u in flow.users.get(s..t).unwrap_or(&[]) {
                queue(u, &mut work, &mut queued);
            }
        }
        // The worklist is drained: store the pointers into the cells they are stored into, and
        // revisit the loads of every cell whose contents grew.
        let grew = cx.store_pass(m).map_err(bare)?;
        if !grew.is_empty() {
            for l in grew {
                queue(l, &mut work, &mut queued);
            }
            continue;
        }
        if cx.cells.settled {
            break;
        }
        cx.cells.settled = true;
        let unwritten = cx.unwritten_loads(m).map_err(bare)?;
        if unwritten.is_empty() {
            break;
        }
        for l in unwritten {
            queue(l, &mut work, &mut queued);
        }
    }
    m.release(work_bytes);
    m.release(n.saturating_mul(std::mem::size_of::<bool>()));
    Ok(())
}

fn block_of(body: &crate::air::ir::Body, id: InstId) -> Option<BlockId> {
    body.blocks
        .iter()
        .find(|(_, blk)| blk.start <= id && id < blk.end)
        .map(|(b, _)| b)
}

/// Attach the instruction's location to an unlocated refusal.
pub fn located(err: Error, e: &Entry, b: BlockId, id: InstId) -> Error {
    match err {
        Error::Unsupported(u) if u.location.is_none() => Error::Unsupported(u.at(
            crate::error::Location::function(e.ep.name.clone(), Some(b.raw()), Some(id.raw())),
        )),
        other => other,
    }
}

/// The use walk: accesses, escapes, compares, and handle seeds.
struct Walk<'c, 'a> {
    cx: &'c mut Cx<'a>,
    classes: &'c mut Classes,
    uses: &'c mut BTreeMap<Root, RootUse>,
    dynamic_deref: &'c mut Option<(BlockId, InstId)>,
    static_sampler: &'c dyn Fn(GlobalId) -> bool,
    /// The pointers stored into and loaded from each forwarding cell slot: `(cell, slot, pointer,
    /// the store or load that moves it)`.
    slots: Vec<(InstId, Slot, Value, InstId)>,
    /// The argument buffers whose layout holds an indirect-command object, or the address of
    /// memory that does.
    objects: Vec<ObjectBuffer>,
    /// Which layouts hold an object, by value or behind a pointer member.
    layouts: ObjectLayouts,
    /// Some read may load the address of an object's memory: a pointer member whose pointee
    /// layout holds an object.
    object_address: bool,
    /// The marks each root set has been given ([`Walk::mark`]), charged per entry.
    marked: HashMap<RootSetId, u8>,
}

impl Walk<'_, '_> {
    fn is_ptr(&self, v: Value) -> bool {
        self.cx
            .value_ty(v)
            .and_then(|t| self.cx.space_of(t))
            .is_some()
    }

    /// Seed the class of pointer operand `v`; a constant operand seeds nothing (a null or
    /// undefined pointer is unconstrained), except that a conflicting constant is refused.
    fn seed(&mut self, v: Value, c: Class, m: &mut Meter) -> Result<(), Error> {
        match v {
            Value::Const(k) => {
                let own = self.const_class(k);
                if own.join(c) == Class::Conflict {
                    return Err(Unsupported::not_representable(Feature::HandleAsPointer).into());
                }
            }
            _ => {
                if let Some(i) = self.cx.ptr_index(v) {
                    self.classes.seed(i, c, m)?;
                }
            }
        }
        Ok(())
    }

    /// The class a pointer constant carries: a constexpr sampler's address is a sampler handle,
    /// another global's address is bytes, null and undefined pointers are unconstrained.
    fn const_class(&self, k: ConstId) -> Class {
        match self.cx.e.module.consts().get(k).map(|c| &c.value) {
            Some(Const::GlobalAddr(g)) if (self.static_sampler)(*g) => {
                Class::Handle(HandleClass::Sampler)
            }
            Some(Const::GlobalAddr(_) | Const::Expr(_)) => Class::Byte,
            _ => Class::Unknown,
        }
    }

    /// Merge the classes of `dst` and pointer operand `v`.
    fn merge(&mut self, dst: Value, v: Value, m: &mut Meter) -> Result<(), Error> {
        let Some(d) = self.cx.ptr_index(dst) else {
            return Ok(());
        };
        match v {
            Value::Const(k) => {
                let c = self.const_class(k);
                self.classes.seed(d, c, m)?;
            }
            _ => {
                if let Some(s) = self.cx.ptr_index(v) {
                    self.classes.union(d, s, m)?;
                }
            }
        }
        Ok(())
    }

    /// Give every root `v` may point into `marks`, one work unit per root. A root set already
    /// given them is not walked again, so a pointer many instructions use walks its roots once
    /// per kind of mark, not once per use.
    fn mark(&mut self, v: Value, m: &mut Meter, marks: Marks) -> Result<(), Error> {
        let fact = self.cx.get(v, m)?;
        let done = match self.marked.get(&fact.roots) {
            Some(d) if d & marks.0 == marks.0 => return Ok(()),
            Some(d) => *d,
            None => {
                m.charge(MARKED_ENTRY_BYTES)?;
                0
            }
        };
        let uses = &mut *self.uses;
        for r in self.cx.sets.get(fact.roots) {
            m.tick(1)?;
            marks.apply(root_use(uses, *r, m)?);
        }
        self.marked.insert(fact.roots, done | marks.0);
        Ok(())
    }

    /// An access through `ptr` reaching `reach` from it (typed as `plain` for a plain load or
    /// store); `None` when some byte it touches is not a constant displacement.
    fn access(
        &mut self,
        at: (BlockId, InstId),
        ptr: Value,
        touch: Touch,
        reach: Option<Reach>,
        plain: Option<TypeId>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        self.seed(ptr, Class::Byte, m)?;
        let fact = self.cx.get(ptr, m)?;
        let space = self.cx.value_ty(ptr).and_then(|t| self.cx.space_of(t));
        let extent = Extent::of(fact.range, reach);
        let roots = self.cx.sets.get(fact.roots).to_vec();
        if roots.contains(&Root::Dynamic)
            && matches!(space, Some(AddrSpace::Device | AddrSpace::Constant))
            && self.dynamic_deref.is_none()
        {
            *self.dynamic_deref = Some(at);
        }
        let single = roots.len() == 1;
        let e = self.cx.e;
        // The handle load: a plain pointer load of exactly one object member's field at a
        // constant offset, the load `seed_loaded` classes as that object's handle.
        let handle_load = touch == Touch::Read
            && plain.and_then(|a| self.cx.space_of(a)).is_some()
            && exact_member(e, &self.cx.abi, &roots, fact).is_some_and(|mem| {
                mem.handle().is_some_and(is_object)
                    && reach.is_some_and(|r| {
                        r.start == 0 && r.end == u64::from(member_element_size(mem))
                    })
            });
        for r in roots.iter().copied() {
            m.tick(1)?;
            // The bytes of an argument buffer's command-buffer or pipeline-state member are an
            // indirect-command object, which only its handle load reads.
            let object = buffer_object_param(&self.cx.abi, r)
                .and_then(|p| self.objects.iter().find(|o| o.param == p))
                .copied();
            if let Some(o) = object {
                if !handle_load && o.touches_object(&e.structs, &self.layouts, fact, reach, m)? {
                    return Err(Unsupported::not_implemented(Feature::ArgumentBufferObjects).into());
                }
                if touch != Touch::Write
                    && !self.object_address
                    && o.reads_object_address(&e.structs, &self.layouts, fact, reach, m)?
                {
                    self.object_address = true;
                }
            }
            // A buffer table is only read, one element pointer at a time; any other access (a
            // store, an atomic, an intrinsic, a load of its bytes as data) has no model.
            if let Root::Param(p) = r {
                if let Some(t) = self.cx.abi.buffer_table(p) {
                    let element = plain.and_then(|a| self.cx.space_of(a)) == Some(t.space);
                    if touch != Touch::Read || !element {
                        return Err(not_lowered());
                    }
                }
            }
            let typed = single
                && match (plain, r) {
                    (Some(a), Root::Alloca(al)) => self.alloca_typed(al, fact.shape, a),
                    (Some(a), Root::Global(g)) => self.global_typed(g, fact.shape, a),
                    _ => false,
                };
            let u = root_use(self.uses, r, m)?;
            u.referenced = true;
            u.dereferenced = true;
            match touch {
                Touch::Read => u.read = true,
                Touch::Write => u.write = true,
                Touch::ReadWrite { atomic } => {
                    u.read = true;
                    u.write = true;
                    u.atomic |= atomic;
                }
            }
            u.extent = u.extent.join(extent);
            u.typed &= typed;
            u.indexed |= !matches!(fact.shape, Shape::Exact(_));
        }
        Ok(())
    }

    fn alloca_typed(&self, al: InstId, shape: Shape, a: TypeId) -> bool {
        let e = self.cx.e;
        let Some(InstKind::Alloca { ty, count, .. }) = e
            .module
            .func(e.func)
            .and_then(|f| f.body.as_ref())
            .and_then(|b| b.insts.get(al))
            .map(|i| &i.kind)
        else {
            return false;
        };
        let Some(count) = self
            .cx
            .const_index(*count)
            .and_then(|c| u64::try_from(c).ok())
        else {
            return false;
        };
        // An access of a simdgroup matrix's held elements is typed too: the plan holds such an
        // alloca typed only when its matrices hold fragments (`memory::fragment`).
        typed_access(e, self.cx.layouts, *ty, count, shape, a)
            || alloca_shape(shape).is_some_and(|s| {
                alloca::fragment_path(e, self.cx.layouts, *ty, count, s, a).is_some()
            })
    }

    fn global_typed(&self, g: GlobalId, shape: Shape, a: TypeId) -> bool {
        let e = self.cx.e;
        let Some(global) = e.module.global(g) else {
            return false;
        };
        typed_access(e, self.cx.layouts, global.value_ty, 1, shape, a)
    }

    /// Record `v` as the pointer that instruction `id` stores or loads through `ptr` when `ptr`
    /// addresses only forwarding cells; whether it does.
    fn cell_slot(
        &mut self,
        id: InstId,
        ptr: Value,
        v: Value,
        m: &mut Meter,
    ) -> Result<bool, Error> {
        let f = self.cx.get(ptr, m)?;
        let roots = self.cx.sets.get(f.roots);
        if !self.cx.cells.facts.forwards(roots) {
            return Ok(false);
        }
        m.charge(map_entry_bytes::<InstId, ()>())?;
        self.cx.cells.facts.accesses.insert(id);
        let slot = Slot::of(f.range);
        let cells: Vec<InstId> = roots
            .iter()
            .filter_map(|r| match r {
                Root::Alloca(a) => Some(*a),
                _ => None,
            })
            .collect();
        grow(&mut self.slots, cells.len(), m)?;
        for a in cells {
            self.slots.push((a, slot, v, id));
        }
        Ok(true)
    }

    /// Join the classes of the pointers stored into and loaded from each cell slot: a load at an
    /// exact offset reads the pointers stored there, and an access at an unknown offset meets
    /// every slot of its cell. A slot copy moves a source slot's pointers to its destination slot,
    /// so they join that slot's too. The stores and loads that meet this way form the cell's slot
    /// groups ([`CellFacts::groups`]): one group's loads read the bytes its stores write.
    fn join_slot_classes(&mut self, m: &mut Meter) -> Result<(), Error> {
        self.copy_slots(m)?;
        self.slots.sort_unstable();
        let mut groups = SlotGroups::new(&self.cx.cells.facts.accesses, m)?;
        let len = self.slots.len();
        let at = |w: &Self, i: usize| w.slots.get(i).copied();
        let mut i = 0;
        while let Some((cell, _, _, _)) = at(self, i) {
            let mut j = i;
            let mut wild = false;
            while let Some((a, s, _, _)) = at(self, j) {
                if a != cell {
                    break;
                }
                wild |= s == Slot::Any;
                j += 1;
            }
            // Each slot's pointers join one class; with a wild access, the whole cell's do. The
            // representative is a value, not a constant (a constant only seeds), and values sort
            // before constants within a slot.
            let key = |s: Slot| if wild { Slot::Any } else { s };
            let mut k = i;
            while let Some((_, s, _, first)) = at(self, k).filter(|_| k < j) {
                m.tick(1)?;
                let mut e = k;
                let mut rep = None;
                while let Some((_, t, v, access)) = at(self, e).filter(|_| e < j) {
                    if key(t) != key(s) {
                        break;
                    }
                    if rep.is_none() && !matches!(v, Value::Const(_)) {
                        rep = Some(v);
                    }
                    groups.union(first, access);
                    e += 1;
                }
                if let Some(rep) = rep {
                    for x in k..e {
                        m.tick(1)?;
                        if let Some((_, _, v, _)) = at(self, x) {
                            self.join_class(rep, v, m)?;
                        }
                    }
                }
                k = e.max(k + 1);
            }
            i = j.max(i + 1).min(len);
        }
        self.cx.cells.facts.groups = groups.finish(m)?;
        Ok(())
    }

    /// Add to [`Walk::slots`] the pointers each slot copy between forwarding cells moves into its
    /// destination slots, to a fixpoint (a chain of copies carries them to its end).
    fn copy_slots(&mut self, m: &mut Meter) -> Result<(), Error> {
        if self.cx.cells.copies.is_empty() {
            return Ok(());
        }
        let entry = map_entry_bytes::<(InstId, Slot, Value, InstId), ()>();
        let mut seen: std::collections::BTreeSet<(InstId, Slot, Value, InstId)> =
            std::collections::BTreeSet::new();
        m.charge(entry.saturating_mul(self.slots.len()))?;
        let mut seen_bytes = entry.saturating_mul(self.slots.len());
        seen.extend(self.slots.iter().copied());
        let result = (|| loop {
            let mut added = false;
            for i in 0..self.cx.cells.copies.len() {
                m.tick(1)?;
                let Some(&(dst, src, len)) = self.cx.cells.copies.get(i) else {
                    break;
                };
                let Some(ends) = self.cx.copy_endpoints(dst, src, m)? else {
                    continue;
                };
                let Ok(len) = i64::try_from(len) else {
                    continue;
                };
                let mut moved: Vec<(InstId, Slot, Value, InstId)> = Vec::new();
                let mut moved_bytes = 0usize;
                let collected = (|| {
                    for &(a, slot, v, access) in &self.slots {
                        m.tick(1)?;
                        if !ends.src.contains(&a) {
                            continue;
                        }
                        let to = match slot {
                            Slot::Any => Slot::Any,
                            Slot::At(o) => match ends.image(o, len) {
                                Some(t) => Slot::At(t),
                                None => continue,
                            },
                        };
                        for &b in &ends.dst {
                            push_working(&mut moved, (b, to, v, access), &mut moved_bytes, m)?;
                        }
                    }
                    Ok::<(), Error>(())
                })();
                if let Err(e) = collected {
                    m.release(moved_bytes);
                    return Err(e);
                }
                for &e in &moved {
                    if seen.insert(e) {
                        m.charge(entry)?;
                        seen_bytes = seen_bytes.saturating_add(entry);
                        grow(&mut self.slots, 1, m)?;
                        self.slots.push(e);
                        added = true;
                    }
                }
                m.release(moved_bytes);
            }
            if !added {
                return Ok(());
            }
        })();
        m.release(seen_bytes);
        result
    }

    /// Join the classes of pointer values `a` and `b` (a constant seeds the other's class).
    fn join_class(&mut self, a: Value, b: Value, m: &mut Meter) -> Result<(), Error> {
        match (a, b) {
            (Value::Const(_), Value::Const(_)) => Ok(()),
            (Value::Const(k), v) | (v, Value::Const(k)) => {
                let c = self.const_class(k);
                self.seed(v, c, m)
            }
            (a, b) => {
                if let (Some(i), Some(j)) = (self.cx.ptr_index(a), self.cx.ptr_index(b)) {
                    self.classes.union(i, j, m)?;
                }
                Ok(())
            }
        }
    }

    fn escape(&mut self, v: Value, m: &mut Meter) -> Result<(), Error> {
        self.mark(v, m, Marks::ESCAPES)
    }

    /// Mark every root a pointer operand reaches as referenced.
    fn reference(&mut self, v: Value, m: &mut Meter) -> Result<(), Error> {
        if !self.is_ptr(v) {
            return Ok(());
        }
        self.mark(v, m, Marks::REFERENCED)
    }

    /// Walk instruction `id`; `operands` is a scratch buffer the caller reuses.
    fn inst(
        &mut self,
        (b, id): (BlockId, InstId),
        inst: &Inst,
        operands: &mut Vec<Value>,
        m: &mut Meter,
    ) -> Result<(), Error> {
        operands.clear();
        let mut count = 0usize;
        inst.kind.for_each_value(|_| count += 1);
        grow(operands, count, m)?;
        inst.kind.for_each_value(|v| operands.push(v));
        for &v in operands.iter() {
            self.reference(v, m)?;
        }
        let this = Value::Inst(id);
        let size = |cx: &Cx<'_>, v: Value| cx.value_ty(v).and_then(|t| store_size(cx.layouts, t));
        match &inst.kind {
            InstKind::Load { ptr, .. } => {
                let n = store_size(self.cx.layouts, inst.ty);
                // A pointer word from forwarding cells loads the pointer its `inttoptr`s are: each
                // joins the slot.
                let word = self.cx.cells.facts.words.get(&id).copied();
                let plain = word
                    .and_then(|c| self.cx.e.module.value_ty(self.cx.e.func, Value::Inst(c)))
                    .unwrap_or(inst.ty);
                self.access((b, id), *ptr, Touch::Read, n.map(Reach::at), Some(plain), m)?;
                if self.is_ptr(this) {
                    self.seed_loaded(id, *ptr, m)?;
                    self.cell_slot(id, *ptr, this, m)?;
                }
                // Each `inttoptr` by its index, read in place (`cell_slot` needs the walk).
                let mut k = 0;
                while let Some(c) = word.and_then(|_| {
                    self.cx
                        .cells
                        .word_loads
                        .get(&id)
                        .and_then(|(_, cs)| cs.get(k))
                        .copied()
                }) {
                    k += 1;
                    self.cell_slot(id, *ptr, Value::Inst(c), m)?;
                }
            }
            InstKind::Store { val, .. }
                if !self.is_ptr(*val)
                    && self
                        .cx
                        .value_ty(*val)
                        .is_some_and(|t| self.cx.e.module.types().holds_pointer(t)) =>
            {
                return Err(not_lowered());
            }
            InstKind::Store { val, ptr, .. } => {
                let n = size(self.cx, *val);
                // A store of pointer bits into forwarding cells stores the pointer.
                let val = self.cx.cells.facts.bits_of(*val).unwrap_or(*val);
                let ty = self.cx.value_ty(val);
                self.access((b, id), *ptr, Touch::Write, n.map(Reach::at), ty, m)?;
                if self.is_ptr(val) {
                    // A pointer stored into a forwarding cell is read back by its loads; into
                    // any other memory, its address escapes.
                    if self.cell_slot(id, *ptr, val, m)? {
                        self.mark(val, m, Marks::STORED)?;
                    } else {
                        self.escape(val, m)?;
                    }
                }
            }
            InstKind::AtomicRmw { ptr, val, .. } => {
                let n = size(self.cx, *val);
                self.access(
                    (b, id),
                    *ptr,
                    Touch::ReadWrite { atomic: true },
                    n.map(Reach::at),
                    None,
                    m,
                )?;
                if self.is_ptr(*val) {
                    self.escape(*val, m)?;
                }
            }
            InstKind::CmpXchg { ptr, cmp, new, .. } => {
                let n = size(self.cx, *new);
                self.access(
                    (b, id),
                    *ptr,
                    Touch::ReadWrite { atomic: true },
                    n.map(Reach::at),
                    None,
                    m,
                )?;
                for v in [*cmp, *new] {
                    if self.is_ptr(v) {
                        self.escape(v, m)?;
                    }
                }
            }
            InstKind::Gep { base, .. } => {
                self.seed(*base, Class::Byte, m)?;
                self.seed(this, Class::Byte, m)?;
            }
            // `ptrtoint` is no class seed: a buffer pointer becomes its guest address, a handle its
            // resource identity (the frontend copies a struct holding a texture or an acceleration
            // structure through an integer). The conversion is lowered, or refused, by the family
            // of the operand's representation. Pointer bits stored only into forwarding cells are
            // the pointer's stores, which the stores mark.
            InstKind::Cast {
                op: CastOp::PtrToInt,
                ..
            } if self.cx.cells.facts.bits.contains_key(&id) => {}
            InstKind::Cast {
                op: CastOp::PtrToInt,
                v,
            } => self.escape(*v, m)?,
            // An integer round trip of a handle-array element is its handle.
            InstKind::Cast {
                op: CastOp::IntToPtr,
                ..
            } => {
                let fact = self.cx.get(this, m)?;
                if let [Root::Param(p)] = self.cx.sets.get(fact.roots) {
                    if let Some(class) = self.cx.abi.handle_array(*p) {
                        self.seed(this, Class::Handle(class), m)?;
                    }
                }
            }
            InstKind::Cast {
                op: CastOp::Bitcast,
                v,
            } if self.is_ptr(this) => self.merge(this, *v, m)?,
            InstKind::ICmp { a, b: other, .. } if self.is_ptr(*a) => {
                let (fa, fb) = (self.cx.get(*a, m)?, self.cx.get(*other, m)?);
                let (ra, rb) = (self.cx.sets.get(fa.roots), self.cx.sets.get(fb.roots));
                let null = |r: &[Root]| r == [Root::Null];
                let same = ra.len() == 1 && ra == rb;
                if !same && !null(ra) && !null(rb) {
                    self.mark(*a, m, Marks::CROSS_COMPARE)?;
                    self.mark(*other, m, Marks::CROSS_COMPARE)?;
                }
            }
            InstKind::Phi { incoming, .. } if self.is_ptr(this) => {
                for (v, _) in incoming.iter() {
                    self.merge(this, *v, m)?;
                }
            }
            InstKind::Select { t, f, .. } if self.is_ptr(this) => {
                self.merge(this, *t, m)?;
                self.merge(this, *f, m)?;
            }
            InstKind::Freeze(v) if self.is_ptr(this) => self.merge(this, *v, m)?,
            InstKind::Alloca { count, .. } => {
                self.seed(this, Class::Byte, m)?;
                if self.cx.const_index(*count).is_none() {
                    return Err(not_lowered());
                }
            }
            InstKind::Call(call) => self.call((b, id), call, m)?,
            InstKind::ExtractValue { .. } if self.is_ptr(this) => {
                let at = self.cx.extract_at(id);
                let mut j = 0;
                while let Some(v) = at.and_then(|a| self.cx.extract_source(a, j)) {
                    self.merge(this, v, m)?;
                    j += 1;
                }
            }
            InstKind::InsertElement { elt: val, .. } if self.is_ptr(*val) => {
                return Err(not_lowered());
            }
            InstKind::Ret(Some(v))
                if self.is_ptr(*v)
                    || self
                        .cx
                        .value_ty(*v)
                        .is_some_and(|t| self.cx.e.module.types().holds_pointer(t)) =>
            {
                return Err(not_lowered())
            }
            _ => {}
        }
        Ok(())
    }

    /// A pointer loaded from memory: a handle when the load reads an argument-buffer member
    /// whose `air.struct_type_info` kind is a resource.
    fn seed_loaded(&mut self, id: InstId, ptr: Value, m: &mut Meter) -> Result<(), Error> {
        let fact = self.cx.get(ptr, m)?;
        let roots = self.cx.sets.get(fact.roots);
        let e = self.cx.e;
        if let [Root::Param(p)] = roots {
            if let Some(class) = self.cx.abi.handle_array(*p) {
                return self.seed(Value::Inst(id), Class::Handle(class), m);
            }
            // An element of a buffer table is a buffer.
            if self.cx.abi.buffer_table(*p).is_some() {
                return self.seed(Value::Inst(id), Class::Byte, m);
            }
        }
        if let Some(h) = exact_member(e, &self.cx.abi, roots, fact).and_then(|mem| mem.handle()) {
            self.seed(Value::Inst(id), Class::Handle(HandleClass::of_abi(h)), m)?;
        }
        Ok(())
    }

    fn call(
        &mut self,
        at: (BlockId, InstId),
        call: &crate::air::ir::Call,
        m: &mut Meter,
    ) -> Result<(), Error> {
        let id = at.1;
        // A post-tessellation entry's control-point fetch: its pointer operand is the entry's
        // control-point input (canonicalization checked it), which names Metal's control points,
        // not memory (`lower::tessellation`).
        if let Callee::Direct(f) = call.callee {
            if self.cx.e.ep.is_control_point_function(f) {
                return Ok(());
            }
        }
        let sig: Option<Signature> = match &call.callee {
            Callee::Air { op, .. } => Some(op.signature()),
            Callee::Llvm { op, .. } => op.signature(),
            _ => None,
        };
        let Some(sig) = sig else {
            if call.args.iter().any(|a| self.is_ptr(*a)) || self.is_ptr(Value::Inst(id)) {
                return Err(not_lowered());
            }
            return Ok(());
        };
        if let Some(h) = sig.ret_handle() {
            self.seed(Value::Inst(id), Class::Handle(HandleClass::of_kind(h)), m)?;
        }
        // An encoder's buffer setter, or its indexed draw, reads no byte of its buffer: it writes
        // the buffer's guest address into the command's record, so the pointer escapes.
        let records_address = matches!(
            call.callee,
            Callee::Air {
                op: AirIntrinsic::IndirectCommand(op),
                ..
            } if op.records_address()
        );
        let lens = self.touch_lens(call);
        for (i, &arg) in call.args.iter().enumerate() {
            if !self.is_ptr(arg) {
                continue;
            }
            match sig.param(i) {
                Some(Operand::Handle(h)) => {
                    self.seed(arg, Class::Handle(HandleClass::of_kind(h)), m)?;
                }
                Some(Operand::Ptr(_)) if records_address => {
                    self.seed(arg, Class::Byte, m)?;
                    self.escape(arg, m)?;
                }
                Some(Operand::Ptr(_)) => {
                    let (touch, reach) = lens
                        .get(i)
                        .copied()
                        .flatten()
                        .unwrap_or((Touch::ReadWrite { atomic: false }, None));
                    self.access(at, arg, touch, reach, None, m)?;
                }
                _ => self.escape(arg, m)?,
            }
        }
        Ok(())
    }

    /// For each operand of an intrinsic call, how its pointer is touched and the bytes it
    /// reaches, where the intrinsic's contract states them; `None` means read-write of unknown
    /// extent.
    fn touch_lens(&self, call: &crate::air::ir::Call) -> Vec<Option<(Touch, Option<Reach>)>> {
        use crate::air::intrinsics::{AccessKind, AirIntrinsic, AtomicOp, LlvmIntrinsic};
        let mut out = vec![None; call.args.len()];
        let konst = |v: Option<&Value>| {
            v.and_then(|v| self.cx.const_index(*v))
                .and_then(|n| u64::try_from(n).ok())
                .map(Reach::at)
        };
        let mut set = |i: usize, t: Touch, r: Option<Reach>| {
            if let Some(s) = out.get_mut(i) {
                *s = Some((t, r));
            }
        };
        match &call.callee {
            Callee::Air {
                op: AirIntrinsic::Atomic(a),
                ..
            } => {
                let n = Some(Reach::at(
                    u64::from(a.ty.elem.bits().div_ceil(8)) * u64::from(a.ty.lanes),
                ));
                match a.op {
                    AtomicOp::Load => set(0, Touch::Read, n),
                    AtomicOp::Store => set(0, Touch::Write, n),
                    AtomicOp::CmpXchgWeak => {
                        set(0, Touch::ReadWrite { atomic: true }, n);
                        set(1, Touch::ReadWrite { atomic: false }, n);
                    }
                    _ => set(0, Touch::ReadWrite { atomic: true }, n),
                }
            }
            Callee::Air {
                op: AirIntrinsic::Coherent(c),
                ..
            } => {
                let n = Some(Reach::at(
                    u64::from(c.ty.elem.bits().div_ceil(8)) * u64::from(c.ty.lanes),
                ));
                match c.kind {
                    AccessKind::Load => set(0, Touch::Read, n),
                    AccessKind::Store => set(1, Touch::Write, n),
                }
            }
            // A texture compare-exchange reads its comparator through `expected` and writes the
            // old texel back through it: the texel the name spells.
            Callee::Air {
                op: AirIntrinsic::Texture(t),
                ..
            } if t.op == crate::air::intrinsics::TexOp::Atomic(AtomicOp::CmpXchgWeak) => {
                if let (Some(i), Some(texel)) = (
                    t.layout().index(crate::air::intrinsics::TexRole::Expected),
                    t.texel,
                ) {
                    let n = u64::from(texel.ty.elem.bits().div_ceil(8)) * u64::from(texel.ty.lanes);
                    set(i, Touch::ReadWrite { atomic: false }, Some(Reach::at(n)));
                }
            }
            // A slice write reads cells of the imageblock its pointer names a field of: in each,
            // one element of the type its name spells, at the pointer's offset in its cell.
            Callee::Air {
                op: AirIntrinsic::Texture(t),
                ..
            } if t.op == crate::air::intrinsics::TexOp::WriteImageblockSlice => {
                if let Some(i) = t
                    .layout()
                    .index(crate::air::intrinsics::TexRole::Imageblock)
                {
                    let n = t
                        .texel
                        .map(|e| u64::from(e.ty.elem.bits().div_ceil(8)) * u64::from(e.ty.lanes));
                    set(i, Touch::Read, n.map(Reach::at));
                }
            }
            Callee::Llvm {
                op: LlvmIntrinsic::Memset { .. },
                ..
            } => set(0, Touch::Write, konst(call.args.get(2))),
            Callee::Llvm {
                op: LlvmIntrinsic::Memcpy { .. } | LlvmIntrinsic::Memmove { .. },
                ..
            } => {
                let n = konst(call.args.get(2));
                set(0, Touch::Write, n);
                set(1, Touch::Read, n);
            }
            // A block load reads, a block store writes, the hull of its 64 cells when both
            // descriptors are constants.
            Callee::Air {
                op: AirIntrinsic::SimdMatrix(op),
                ..
            } => {
                if let Some(b) = crate::lower::simd::matrix::block_touch(self.cx.e, call, op) {
                    let touch = if b.write { Touch::Write } else { Touch::Read };
                    set(b.ptr, touch, b.reach);
                }
            }
            _ => {}
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::testing::meter;

    /// The handle classes join by size through the metered union-find: a chain of unions from the
    /// last index to the first, which left the unranked union-find a path through every index, is a
    /// star, so joining and resolving `n` indices costs between 1 and 4 units an index, one per
    /// find step; the tables' charge is what the ledger holds.
    #[test]
    fn handle_classes_join_by_size_and_meter_every_find_step() {
        let n = 1usize << 16;
        let mut m = meter();
        let mut c = Classes::new(n, &mut m).expect("classes");
        for i in (0..n - 1).rev() {
            c.union(i, i + 1, &mut m).expect("union");
        }
        c.seed(n - 1, Class::Byte, &mut m).expect("seed");
        for i in 0..n {
            assert_eq!(c.class_of(i, &mut m).expect("class"), Class::Byte);
        }
        let units = m.work();
        assert!(
            (n as u64..4 * n as u64).contains(&units),
            "{units} units for {n} indices"
        );
        assert_eq!(m.ledger() as usize, c.charged());
        assert_eq!(c.conflict, None);
    }

    /// A conflict a seed meets is reported at the seeded index, and one a union meets at the
    /// index that was the first set's representative when the union-find joined the second
    /// under the first, whichever representative joining by size keeps.
    #[test]
    fn a_class_conflict_keeps_its_location_under_joining_by_size() {
        let mut m = meter();
        let mut c = Classes::new(9, &mut m).expect("classes");
        for (a, b) in [(5, 6), (5, 7), (4, 5)] {
            c.union(a, b, &mut m).expect("union");
        }
        // {4, 5, 6, 7}: joined by size under 5, labelled 4.
        assert_eq!(c.find(4, &mut m).expect("find"), 5);
        c.seed(6, Class::Byte, &mut m).expect("seed");
        c.seed(8, Class::Handle(HandleClass::Texture), &mut m)
            .expect("seed");
        assert_eq!(c.conflict, None);
        c.union(6, 8, &mut m).expect("union");
        assert_eq!(c.conflict, Some(4));

        let mut c = Classes::new(3, &mut m).expect("classes");
        c.union(0, 1, &mut m).expect("union");
        c.seed(0, Class::Byte, &mut m).expect("seed");
        c.seed(1, Class::Handle(HandleClass::Sampler), &mut m)
            .expect("seed");
        assert_eq!(c.conflict, Some(1));
    }
}
