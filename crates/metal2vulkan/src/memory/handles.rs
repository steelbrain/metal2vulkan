//! The handle domain.
//!
//! **Handle values.** Every texture or sampler value is a `u32` *handle index*
//! (`DemoteTy::HandleIndex`), lowered to an image or sampler object only where an operation uses
//! it, so a `select`, `phi`, `freeze`, or cast of handles is an integer operation. A handle whose
//! provenance (`memory::provenance`) is a finite set of static candidates is encoded by the
//! candidate itself ([`Candidate::handle`]):
//!
//! - [`NULL_HANDLE`] (0) is the null handle (`get_null_texture`, a null or undefined pointer
//!   constant);
//! - `1 + n` is `[[texture(n)]]` for a texture, or `[[sampler(n)]]` for a runtime sampler
//!   (Metal's sampler slots are below 16, so these are `1..=16`);
//! - [`CONSTEXPR_SAMPLER_BASE`]` + k` is the entry's `k`-th constexpr sampler.
//!
//! An element of a handle-array parameter (`array<texture2d<T>, N> [[texture(n)]]`, AIR's
//! thread-space pointer to its handles) is the slot `n + element`, so a loaded element is an
//! ordinary static candidate and needs no descriptor array.
//!
//! **Uses**. An operation whose texture or sampler has one candidate reads that slot's binding
//! directly. One with several candidates dispatches on the handle index at the use, one arm per
//! non-null candidate, the null handle and every other value taking the null arm (a read or sample
//! is 0, a write is dropped, `is_null` is true). That dispatch is a builder-closed `switch_at_use`,
//! which must not contain a tangled instruction; the rules for tangled uses are the sampler
//! family's.
//!
//! **Heap handles**. A texture handle loaded from an argument buffer is the guest's 8-byte
//! `gpuResourceID`. Its load calls the module's one search of the **handle table**
//! ([`define_lookup`], [`lookup`]), a read-only storage buffer (`Synthetic(HandleTable)`) of
//! 16-byte entries `{resource_id: u64, element: u32, flags: u32}` sorted by id, and yields the
//! resident resource's **element** in the set-1 heap: one runtime descriptor array per image class
//! (`ResourceKind::Heap`), indexed `NonUniform`. Element [`NEUTRAL_ELEMENT`] (0) holds each class's
//! neutral and is the value of a null handle and of every id the table misses, so the heap domain
//! has the static domain's null handle and needs no switch. The table's first entry is the null
//! entry `{0, 0, 0}` (Metal's nil resource id), so it is never empty and every entry the search
//! reads is inside it; it holds at most [`TABLE_CAPACITY`] entries, the reach of the
//! [`SEARCH_STEPS`]-step branch-free search. A value whose roots are `Dynamic` and possibly null is
//! a heap handle ([`Domain::Heap`]); mixing static candidates with heap handles is
//! `Unsupported{ArgumentBufferTextures, NotImplemented}`. [`HeapGroups`] joins the heap handles
//! that `phi`, `select`, `freeze`, and casts merge, so the texture family declares one heap class
//! per group from all of its uses, as it declares one image per slot.
//!
//! **The out-of-bounds alpha**. A texture entry's flags carry [`FLAG_OUT_OF_BOUNDS_ALPHA_ONE`] when
//! the resident texture's pixel format has no alpha, whose out-of-bounds read then returns alpha 1,
//! as Metal's read of such a texture bound to a slot does (no heap texture was measured). A module
//! whose shader-checked heap read observes the alpha calls the search's flagged form
//! ([`Lookup::Flagged`]): the handle is the element with that flag in bit 31
//! ([`HANDLE_ALPHA_ONE`]), which every heap access masks off ([`Lookup::element_mask`]) and the
//! read selects its out-of-bounds alpha by. An element is below [`TABLE_CAPACITY`], so the bit is
//! never an element's.
//!
//! **Heap samplers** (the sampler half). A sampler handle loaded from an argument buffer is the
//! guest's resource id of an `MTLSamplerState`, searched in the stage's **sampler table**
//! (`Synthetic(SamplerTable)`), the handle table's layout over the argument buffers' samplers: a
//! sampler's resource id is not a texture's, so the two never share a table. The search's entry
//! form ([`Lookup::Entry`]) yields the index of the entry it finds, or 0, the null entry, for a
//! miss, and that index is the handle. Each use reads the entry's element
//! ([`EntryField::Element`]), which indexes the set-1 sampler array of the sampling's use
//! (`HandleClass::Sampler`, whose element 0 is the neutral sampler), and its flags word
//! ([`EntryField::Flags`]), the sampler's [`SamplerWord`](crate::metal::sampler::SamplerWord): the
//! state the shader reads of a sampler it learns at run time (pixel coordinates, and with them the
//! exact fetch and the centre of a nearest pixel sampler's texel). A null entry's element and word
//! are 0: the neutral sampler, and the state of `MTLSamplerDescriptor`'s defaults.

use std::collections::{BTreeMap, BTreeSet};

use crate::air::abi::{AbiHandle, AbiInt, Access as AbiAccess, ParamBinding, ParamRole};
use crate::air::ir::{AddrSpace, Body, GlobalId, InstId, InstKind, Type, Value};
use crate::air::Entry;
use crate::error::{Error, Feature, InternalError, InternalKind, LowerFamily, Unsupported};
use crate::memory::leaf::k32;
use crate::memory::views::{self, View};
use crate::pipeline::sym::ImageAccess;
use crate::pipeline::sym::ResourceSlot;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{FuncId, GlobalId as SGlobalId, TypeId, Value as SValue};
use crate::spirv::ir::{
    natural_layout, BinOp, ConvOp, LayoutRule, MemAccess, StorageClass, Type as SType,
};

use super::abi_index::AbiIndex;
use super::plan::MemoryPlan;
use super::provenance::{
    map_entry_bytes, member_element_size, push_working, HandleClass, OffsetRange, Repr, Root,
};
use super::union_find::UnionFind;
use crate::budget::Meter;

/// The null handle.
pub const NULL_HANDLE: u32 = 0;

/// The handle index of the entry's first constexpr sampler: runtime sampler slots (below 16)
/// take `1..=16`.
pub const CONSTEXPR_SAMPLER_BASE: u32 = 17;

/// Metal's number of sampler slots per stage.
const SAMPLER_SLOTS: u32 = 16;

/// The handle index of the absent handle ([`Candidate::Absent`]): above every slot's and every
/// constexpr sampler's, so no other candidate takes it.
pub const ABSENT_HANDLE: u32 = u32::MAX;

/// One static candidate of a texture or sampler handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Candidate {
    /// The null handle.
    Null,
    /// `[[texture(n)]]`.
    Texture(u32),
    /// The runtime sampler `[[sampler(n)]]`.
    Sampler(u32),
    /// The entry's `ordinal`-th constexpr sampler (`Entry::static_samplers`), the global `global`.
    Constexpr { ordinal: u32, global: GlobalId },
    /// A texture or sampler a function constant removed (`Entry::absent_handles`): Metal gives a
    /// read through it no value, so each operation on it is the absent read (the command buffer's
    /// execution status marked, the result zero), and a write or fence of it is dropped, as Metal
    /// drops one.
    Absent,
}

impl Candidate {
    /// The handle index that names this candidate.
    pub fn handle(self) -> u32 {
        match self {
            Candidate::Null => NULL_HANDLE,
            Candidate::Texture(n) | Candidate::Sampler(n) => n.saturating_add(1),
            Candidate::Constexpr { ordinal, .. } => CONSTEXPR_SAMPLER_BASE.saturating_add(ordinal),
            Candidate::Absent => ABSENT_HANDLE,
        }
    }
}

fn not_implemented(f: Feature) -> Error {
    Unsupported::not_implemented(f).into()
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// The constant slot range of a texture or sampler parameter `p`: `(class, first, count)`, its
/// binding found through the memory plan's index.
pub fn param_slots(e: &Entry, abi: &AbiIndex, p: u32) -> Option<(HandleClass, u32, u32)> {
    binding_slots(abi.binding(e, p)?)
}

/// The constant slot range of the texture or sampler parameter binding `b` binds:
/// `(class, first, count)`.
pub fn binding_slots(b: &ParamBinding) -> Option<(HandleClass, u32, u32)> {
    let (class, loc) = match &b.role {
        ParamRole::Texture(t) => (HandleClass::Texture, t.location),
        ParamRole::Sampler { location } => (HandleClass::Sampler, *location),
        _ => return None,
    };
    match (loc.index, loc.count) {
        (AbiInt::Const(first), AbiInt::Const(count)) => Some((class, first, count)),
        _ => None,
    }
}

/// The candidate of slot `n` of class `class`.
fn slot_candidate(class: HandleClass, n: u32) -> Result<Candidate, Error> {
    match class {
        HandleClass::Texture => Ok(Candidate::Texture(n)),
        HandleClass::Sampler if n < SAMPLER_SLOTS => Ok(Candidate::Sampler(n)),
        HandleClass::Sampler => Err(invariant("a sampler slot beyond Metal's sampler table")),
        _ => Err(not_implemented(Feature::HandleAsPointer)),
    }
}

/// The candidate of the constexpr sampler at global `g`, if it is one.
pub fn constexpr_candidate(e: &Entry, g: GlobalId) -> Option<Candidate> {
    let ordinal = e.static_samplers.iter().position(|(s, _)| *s == g)?;
    Some(Candidate::Constexpr {
        ordinal: u32::try_from(ordinal).ok()?,
        global: g,
    })
}

/// The elements `lo..=hi` of a handle array of `count` elements of `size` bytes that a pointer
/// at byte offsets `range` may name, or `None` when it names none (a read wholly outside the
/// array, which Metal leaves undefined).
pub fn array_elements(range: OffsetRange, size: u64, count: u32) -> Option<(u32, u32)> {
    if count == 0 || size == 0 {
        return None;
    }
    let last = i64::from(count - 1);
    let size = i64::try_from(size).ok()?;
    let (lo, hi) = match range {
        OffsetRange::Span { lo, hi } => (lo.div_euclid(size), hi.div_euclid(size)),
        OffsetRange::Top | OffsetRange::Empty => (0, last),
    };
    let (lo, hi) = (lo.max(0), hi.min(last));
    if lo > hi {
        return None;
    }
    Some((u32::try_from(lo).ok()?, u32::try_from(hi).ok()?))
}

/// The byte size of one handle (an AIR pointer) of value `v`'s type.
pub fn handle_size(e: &Entry, mp: &MemoryPlan, v: Value) -> Option<u64> {
    let ty = e.module.value_ty(e.func, v)?;
    match e.module.types().get(ty)? {
        Type::Ptr { .. } => mp.layouts.store_size(ty),
        _ => None,
    }
}

/// The static candidates of texture or sampler handle `v` of class `class`, sorted, or the
/// refusal of a handle with no static candidate set: a dynamic handle (loaded from an argument
/// buffer: a heap handle, which [`domain`] tells apart), the read sampler (`get_read_sampler`,
/// which has no value and samples nothing, only a texture `read` uses it, and that never asks for
/// its candidates), or a handle of another class.
///
/// The walk of an instruction's root set is metered: a work unit per root and per element of a
/// handle array it names, each element's candidate charged to the ledger before it is placed and
/// the working set returned when the walk ends. A parameter or a constant has one candidate and no
/// walk ([`value_candidates`]).
pub fn candidates(
    e: &Entry,
    mp: &MemoryPlan,
    v: Value,
    class: HandleClass,
    m: &mut Meter,
) -> Result<Vec<Candidate>, Error> {
    if let Some(r) = value_candidates(e, &mp.abi, v, class) {
        return r;
    }
    let Some(Repr::Handle { class: c, cands }) = mp.repr(v) else {
        return Err(not_implemented(Feature::Lowering(LowerFamily::Access)));
    };
    if c != class {
        return Err(not_implemented(Feature::HandleAsPointer));
    }
    let mut placed = Placed::default();
    let r = walk_roots(e, mp, (v, mp.roots(cands)), class, &mut placed, m);
    m.release(placed.charged);
    r.map(|()| placed.set.into_iter().collect())
}

/// The candidates [`candidates`]' walk has placed, and the ledger bytes they hold.
#[derive(Default)]
struct Placed {
    set: BTreeSet<Candidate>,
    charged: usize,
}

impl Placed {
    /// Place `c`, charging its entry first.
    fn place(&mut self, c: Candidate, m: &mut Meter) -> Result<(), Error> {
        let bytes = crate::budget::tree_bytes(1, std::mem::size_of::<Candidate>());
        m.charge(bytes)?;
        self.charged = self.charged.saturating_add(bytes);
        self.set.insert(c);
        Ok(())
    }
}

/// The candidates of a handle parameter or constant `v` (`None` for an instruction's value),
/// which [`candidates`] reads without a walk.
pub fn value_candidates(
    e: &Entry,
    abi: &AbiIndex,
    v: Value,
    class: HandleClass,
) -> Option<Result<Vec<Candidate>, Error>> {
    // A handle parameter itself (its provenance is the parameter).
    if let Value::Arg(a) = v {
        // A removed one is the absent handle.
        if let Some(c) = abi.absent(a.raw()) {
            return Some(if c != class {
                Err(not_implemented(Feature::HandleAsPointer))
            } else if abi.handle_array(a.raw()).is_some() {
                Err(not_implemented(Feature::Lowering(LowerFamily::Texture)))
            } else {
                Ok(vec![Candidate::Absent])
            });
        }
        return Some(match param_slots(e, abi, a.raw()) {
            Some((c, first, 1)) if c == class && abi.handle_array(a.raw()).is_none() => {
                slot_candidate(class, first).map(|c| vec![c])
            }
            _ => Err(not_implemented(Feature::Lowering(LowerFamily::Texture))),
        });
    }
    if let Value::Const(c) = v {
        use crate::air::ir::Const;
        return Some(match e.module.consts().get(c).map(|k| &k.value) {
            Some(Const::Null | Const::Undef | Const::Poison) => Ok(vec![Candidate::Null]),
            Some(Const::GlobalAddr(g)) if class == HandleClass::Sampler => {
                constexpr_candidate(e, *g)
                    .map(|c| vec![c])
                    .ok_or_else(|| not_implemented(Feature::Lowering(LowerFamily::Sampler)))
            }
            _ => Err(not_implemented(Feature::Lowering(LowerFamily::Texture))),
        });
    }
    None
}

/// [`candidates`]' walk of instruction `v`'s roots `roots` into `out`.
fn walk_roots(
    e: &Entry,
    mp: &MemoryPlan,
    (v, roots): (Value, &[Root]),
    class: HandleClass,
    out: &mut Placed,
    m: &mut Meter,
) -> Result<(), Error> {
    for r in roots {
        m.tick(1)?;
        match r {
            Root::Null => out.place(Candidate::Null, m)?,
            // A removed texture or sampler, or any element of a removed table: the absent handle.
            Root::Param(p) if mp.abi.absent(*p).is_some() => {
                if mp.abi.absent(*p) != Some(class) {
                    return Err(not_implemented(Feature::HandleAsPointer));
                }
                out.place(Candidate::Absent, m)?;
            }
            Root::Param(p) => {
                let Some((pc, first, count)) = param_slots(e, &mp.abi, *p) else {
                    return Err(not_implemented(Feature::Lowering(LowerFamily::Texture)));
                };
                if pc != class {
                    return Err(not_implemented(Feature::HandleAsPointer));
                }
                if mp.abi.handle_array(*p).is_none() {
                    out.place(slot_candidate(class, first)?, m)?;
                    continue;
                }
                // An element of a handle array: when the handle's only root is the array, its
                // offset names the elements it may be; otherwise any element.
                let range = match roots {
                    [_] => mp.offset(v).unwrap_or(OffsetRange::Top),
                    _ => OffsetRange::Top,
                };
                let size = handle_size(e, mp, v).ok_or_else(|| invariant("a handle's size"))?;
                match array_elements(range, size, count) {
                    Some((lo, hi)) => {
                        for k in lo..=hi {
                            m.tick(1)?;
                            let n = first
                                .checked_add(k)
                                .ok_or_else(|| invariant("a handle-array slot beyond u32"))?;
                            out.place(slot_candidate(class, n)?, m)?;
                        }
                    }
                    None => out.place(Candidate::Null, m)?,
                }
            }
            Root::Global(g) => match constexpr_candidate(e, *g) {
                Some(c) if class == HandleClass::Sampler => out.place(c, m)?,
                _ => return Err(not_implemented(Feature::HandleAsPointer)),
            },
            // A handle loaded from an argument buffer or from memory: the handle table and heap,
            // not lowered yet.
            Root::Dynamic => return Err(not_implemented(Feature::ArgumentBufferTextures)),
            // The read sampler has no value and samples nothing: a sample through it, and a merge
            // or copy of it, are refused here.
            Root::Call(_) => return Err(not_implemented(Feature::Lowering(LowerFamily::Sampler))),
            Root::BufferElement { .. } | Root::Alloca(_) => {
                return Err(not_implemented(Feature::HandleAsPointer))
            }
        }
    }
    Ok(())
}

/// The heap element of the null handle and of every resource id the handle table misses: each
/// heap array's element 0 is its class's neutral.
pub const NEUTRAL_ELEMENT: u32 = 0;
/// Bytes of one handle-table entry.
pub const TABLE_ENTRY_BYTES: u32 = 16;
/// Words of one handle-table entry: the resource id's low and high words, the element, the flags.
const ENTRY_WORDS: u32 = 4;
/// Word of an entry holding the resource id's low 32 bits.
pub const ENTRY_ID_LO: u32 = 0;
/// Word of an entry holding the resource id's high 32 bits.
pub const ENTRY_ID_HI: u32 = 1;
/// Word of an entry holding the heap element.
pub const ENTRY_ELEMENT: u32 = 2;
/// Word of an entry holding the flags: for a sampler, its
/// [`SamplerWord`](crate::metal::sampler::SamplerWord); for a texture,
/// [`FLAG_OUT_OF_BOUNDS_ALPHA_ONE`], [`FLAG_RED_BLUE`], and the index of its texel class in
/// [`FLAG_TEXEL_CLASS`], every other bit 0.
pub const ENTRY_FLAGS: u32 = 3;
/// The flag of a texture entry whose pixel format has no alpha
/// (`metal::format::out_of_bounds_alpha_one`): Metal's out-of-bounds read of such a texture
/// bound to a slot returns alpha 1 (MSL 4.1 section 2.9.1), which a read through the table takes
/// too; no heap texture was measured.
pub const FLAG_OUT_OF_BOUNDS_ALPHA_ONE: u32 = 1 << 1;
/// The flag of a texture entry whose pixel format a storage view may hold through its RGBA-order
/// twin, red and blue exchanged (`metal::format::red_blue_exchanged`: `BGRA8Unorm`,
/// `BGRA8Unorm_sRGB`, `BGR10A2Unorm`), which tells such a member from the red-first format of its
/// texel class (`RGBA8Unorm`, `RGBA8Unorm_sRGB`, `RGB10A2Unorm`): a storage access through a heap
/// handle exchanges the channels of a member its storage array views so
/// (`bind::heap::storage_conversions`). A handle carries it beside its class
/// ([`HANDLE_RED_BLUE`]).
pub const FLAG_RED_BLUE: u32 = 1 << 2;
/// The field of a texture entry's flags holding the index of its texel class in
/// `metal::texel_format::heap_classes`: the class a slot bound to the texture would state in its
/// `TexelFormat` specialization, `OTHER` (index 0) for a swizzled texture, by which a function
/// converts or reads the texture's texels where the module reads it (a `half` conversion on Metal's
/// own sampler, a fetched gather, a NaN texel's rule, a raw-held form's emulation). A heap handle
/// carries it in place, in a module whose search carries it ([`Lookup::Flagged`]'s `class`), so
/// [`HANDLE_TEXEL_CLASS`] is this field.
pub const FLAG_TEXEL_CLASS: u32 = 0x1fff << 16;
/// The field of a heap handle that carries its entry's [`FLAG_TEXEL_CLASS`], in place, in a
/// module whose search carries it ([`Lookup::Flagged`]'s `class`).
pub const HANDLE_TEXEL_CLASS: u32 = FLAG_TEXEL_CLASS;
/// The bit of a heap handle that carries its entry's [`FLAG_OUT_OF_BOUNDS_ALPHA_ONE`], in a
/// module whose search carries it ([`Lookup::Flagged`]'s `alpha`).
pub const HANDLE_ALPHA_ONE: u32 = 1 << 31;
/// The bit of a heap handle that carries its entry's [`FLAG_RED_BLUE`], in a module whose search
/// carries the texel class ([`Lookup::Flagged`]'s `class`).
pub const HANDLE_RED_BLUE: u32 = 1 << 30;
// Every element is below the table's capacity, so no carried bit is an element's, and the
// carried fields do not overlap.
const _: () = assert!(TABLE_CAPACITY <= 1 << HANDLE_TEXEL_CLASS.trailing_zeros());
const _: () = assert!(HANDLE_TEXEL_CLASS & HANDLE_ALPHA_ONE == 0);
const _: () = assert!((HANDLE_TEXEL_CLASS | HANDLE_ALPHA_ONE) & HANDLE_RED_BLUE == 0);
const _: () = assert!(FLAG_TEXEL_CLASS & FLAG_OUT_OF_BOUNDS_ALPHA_ONE == 0);
const _: () = assert!((FLAG_TEXEL_CLASS | FLAG_OUT_OF_BOUNDS_ALPHA_ONE) & FLAG_RED_BLUE == 0);

/// Unrolled steps of the handle-table search.
pub const SEARCH_STEPS: u32 = 16;
/// The most entries (the null entry included) the search covers.
pub const TABLE_CAPACITY: u32 = 1 << SEARCH_STEPS;

/// How a texture or sampler value's handle index is encoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Domain {
    /// A finite set of static candidates, sorted ([`Candidate::handle`]).
    Static(Vec<Candidate>),
    /// Loaded from an argument buffer: a set-1 heap element, [`NEUTRAL_ELEMENT`] for null.
    Heap,
}

/// What selection reads of a handle's [`Domain`] wherever the candidates themselves are not
/// needed: its encoding, and its one candidate when it has exactly one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainKind {
    /// [`Domain::Static`], with its candidate when it is the only one.
    Static(Option<Candidate>),
    /// [`Domain::Heap`].
    Heap,
}

impl Domain {
    /// The domain's [`DomainKind`].
    pub fn kind(&self) -> DomainKind {
        match self {
            Domain::Static(c) => DomainKind::Static(match c.as_slice() {
                [one] => Some(*one),
                _ => None,
            }),
            Domain::Heap => DomainKind::Heap,
        }
    }
}

/// Whether every root of `roots` is `Dynamic` or null, with at least one `Dynamic`.
fn heap_roots(roots: &[Root]) -> bool {
    // Sets are interned sorted.
    roots.binary_search(&Root::Dynamic).is_ok()
        && roots
            .iter()
            .all(|r| matches!(r, Root::Dynamic | Root::Null))
}

/// The domain of texture or sampler handle `v` of class `class`: a heap handle, or the static
/// candidates of [`candidates`] (whose refusals it keeps). A handle that mixes static candidates
/// with a heap handle is not implemented.
///
/// Metered as [`candidates`]: a work unit for the search for a dynamic root (the set is sorted),
/// then [`candidates`]' walk.
pub fn domain(
    e: &Entry,
    mp: &MemoryPlan,
    v: Value,
    class: HandleClass,
    m: &mut Meter,
) -> Result<Domain, Error> {
    if let (Value::Inst(_), Some(Repr::Handle { class: c, cands })) = (v, mp.repr(v)) {
        let roots = mp.roots(cands);
        m.tick(1)?;
        if roots.binary_search(&Root::Dynamic).is_ok() {
            if c != class {
                return Err(not_implemented(Feature::HandleAsPointer));
            }
            if !matches!(class, HandleClass::Texture | HandleClass::Sampler) || !heap_roots(roots) {
                return Err(not_implemented(Feature::ArgumentBufferTextures));
            }
            return Ok(Domain::Heap);
        }
    }
    candidates(e, mp, v, class, m).map(Domain::Static)
}

/// The class of `v` when it is a heap texture or sampler handle (the value of an instruction
/// whose every root is `Dynamic` or null, with a `Dynamic` one).
pub fn heap_handle(mp: &MemoryPlan, v: Value) -> Option<HandleClass> {
    match (v, mp.repr(v)?) {
        (Value::Inst(_), Repr::Handle { class, cands })
            if matches!(class, HandleClass::Texture | HandleClass::Sampler)
                && heap_roots(mp.roots(cands)) =>
        {
            Some(class)
        }
        _ => None,
    }
}

/// Whether `v` is a heap texture handle (the value of an instruction).
fn is_heap(mp: &MemoryPlan, v: Value) -> bool {
    matches!(
        (v, mp.repr(v)),
        (Value::Inst(_), Some(Repr::Handle { class: HandleClass::Texture, cands }))
            if heap_roots(mp.roots(cands))
    )
}

/// Whether `v` is a null handle: a null, `undef`, or `poison` pointer constant, or a handle whose
/// only root is null (`get_null_texture`).
fn is_null_handle(e: &Entry, mp: &MemoryPlan, v: Value) -> bool {
    use crate::air::ir::Const;
    match v {
        Value::Const(c) => matches!(
            e.module.consts().get(c).map(|k| &k.value),
            Some(Const::Null | Const::Undef | Const::Poison)
        ),
        _ => matches!(
            mp.repr(v),
            Some(Repr::Handle { cands, .. }) if mp.roots(cands).iter().all(|r| *r == Root::Null)
        ),
    }
}

/// The argument-buffer texture member a heap handle load reads: the argument buffer's parameter,
/// the member's position (`(layout, index)` in `Entry::structs`, so loads of one member, or of
/// elements of one member array, compare equal), and its heap array access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LoadedMember {
    param: u32,
    member: (usize, usize),
    /// The index of the last member the load may read: `member.1` but for a run of adjacent
    /// members of one access (`member_run`).
    last: usize,
    access: MemberHandle,
}

/// What an argument-buffer handle member holds: a texture, by its heap array access, or a
/// sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemberHandle {
    Texture(ImageAccess),
    Sampler,
}

/// The heap array access of a texture handle loaded through buffer pointer `ptr` with alignment
/// `align` (bytes): that of the argument-buffer member the load reads (`sample` and `read`
/// members are sampled images, `write` and `read_write` members storage images, as for
/// `[[texture(n)]]` parameters). The member is known when every offset the load may read lies in
/// it, when the offset is one dynamic element index over an array the member lies in
/// (`strided_member`: a member array, or the array of argument buffers the parameter points to), or
/// when the offsets span a run of adjacent single-handle members of one access that the load's
/// alignment keeps it at the start of (`member_run`, a `select` of two members' addresses), in the
/// argument buffer an entry parameter binds or in one nested in it (`within`). A handle loaded from
/// thread or threadgroup memory (a handle cell) is `Lowering(Access)`; one from a buffer whose
/// member is not known (a pointer loaded from memory that no nested-argument-buffer member holds,
/// an address merged by a `phi` or `select`, or offsets spanning other members) is not implemented.
pub fn member_access(
    e: &Entry,
    mp: &MemoryPlan,
    body: &Body,
    ptr: Value,
    align: u64,
) -> Result<ImageAccess, Error> {
    match loaded_member(e, mp, body, ptr, align)?.access {
        MemberHandle::Texture(access) => Ok(access),
        MemberHandle::Sampler => Err(not_implemented(Feature::ArgumentBufferTextures)),
    }
}

/// The pointer and the alignment (bytes) of the argument-buffer member load that defines handle
/// instruction `inst` of `body`: a load of the handle, or an `inttoptr` of a plain integer load
/// as wide as the handle (`provenance::round_trip`), whose integer is the member's resource id,
/// so that the `inttoptr` is the handle a load of the member reads (the frontend's form of a
/// `select` between two members). `None` for any other instruction.
pub fn member_load(
    e: &Entry,
    mp: &MemoryPlan,
    body: &Body,
    inst: &crate::air::ir::Inst,
) -> Option<(Value, u64)> {
    let bytes = |a: Option<crate::air::ir::Align>| a.map_or(1, |a| a.bytes());
    match &inst.kind {
        InstKind::Load { ptr, align, .. } => Some((*ptr, bytes(*align))),
        InstKind::Cast {
            op: crate::air::ir::CastOp::IntToPtr,
            v,
        } => {
            let ptr = super::provenance::round_trip(e, &mp.layouts, body, *v, inst.ty)?;
            let Value::Inst(l) = v else { return None };
            match &body.insts.get(*l)?.kind {
                InstKind::Load { align, .. } => Some((ptr, bytes(*align))),
                _ => None,
            }
        }
        _ => None,
    }
}

/// [`member_access`], with the member's identity.
fn loaded_member(
    e: &Entry,
    mp: &MemoryPlan,
    body: &Body,
    ptr: Value,
    align: u64,
) -> Result<LoadedMember, Error> {
    let space = e
        .module
        .value_ty(e.func, ptr)
        .and_then(|t| e.module.types().get(t))
        .and_then(|t| match t {
            Type::Ptr { space } => Some(*space),
            _ => None,
        });
    if !matches!(space, Some(AddrSpace::Device | AddrSpace::Constant)) {
        return Err(not_implemented(Feature::Lowering(LowerFamily::Access)));
    }
    let at = located(e, mp, body, ptr, align, 0)?;
    Ok(LoadedMember {
        param: at.param,
        member: at.member,
        last: at.last,
        access: handle_of(at.mem).ok_or_else(unknown_member)?,
    })
}

/// The refusal of a load whose argument-buffer member is not known.
fn unknown_member() -> Error {
    not_implemented(Feature::ArgumentBufferTextures)
}

/// The argument-buffer member a load through buffer pointer `ptr` with alignment `align` reads:
/// the outermost argument buffer's parameter, the member's place (`(layout, index)` in
/// `Entry::structs`), the last member of a run (`member_run`), and the member itself.
struct Located<'s> {
    param: u32,
    member: (usize, usize),
    last: usize,
    mem: &'s crate::air::abi::Member,
}

/// Where in an argument buffer a pointer points: the outermost argument buffer's parameter, the
/// layout of the argument buffer it points into, the size of one such argument buffer (for an
/// index over an array of them), and the pointer's offsets from that argument buffer's start.
struct Within {
    param: u32,
    layout: crate::air::abi::StructRef,
    element: Option<u32>,
    offsets: Offsets,
}

/// The offsets a pointer may have from the start of its argument buffer.
#[derive(Clone, Copy)]
enum Offsets {
    /// Every offset in `lo..=hi`.
    Span { lo: i64, hi: i64 },
    /// `base + k · stride` for one dynamic element index `k`.
    Strided { base: i64, stride: u64 },
}

/// The member a load through `ptr` reads ([`Located`]), `depth` argument buffers inside the
/// parameter's. The member is known when every offset the load may read lies in it, when the
/// offset is one dynamic element index over an array the member lies in (`strided_member`), or
/// when the offsets span a run of adjacent single-handle members of one access that the load's
/// alignment keeps it at the start of (`member_run`).
fn located<'e>(
    e: &'e Entry,
    mp: &MemoryPlan,
    body: &Body,
    ptr: Value,
    align: u64,
    depth: u64,
) -> Result<Located<'e>, Error> {
    let Within {
        param,
        layout,
        element,
        offsets,
    } = within(e, mp, body, ptr, depth)?;
    let member = match offsets {
        Offsets::Span { lo, hi } => {
            let (Ok(lo), Ok(hi)) = (u64::try_from(lo), u64::try_from(hi)) else {
                return Err(unknown_member());
            };
            let first = (mp.abi)
                .member_place_at(&e.structs, layout, lo)
                .ok_or_else(unknown_member)?;
            let last = (mp.abi)
                .member_place_at(&e.structs, layout, hi)
                .ok_or_else(unknown_member)?;
            if first.0 != last.0 {
                let run = member_run(&e.structs, (first, last), (lo, hi), align)
                    .ok_or_else(unknown_member)?;
                return Ok(Located {
                    param,
                    member: first.0,
                    last: run,
                    mem: first.1,
                });
            }
            first
        }
        Offsets::Strided { base, stride } => {
            strided_member((&mp.abi, &e.structs), layout, element, base, stride)
                .ok_or_else(unknown_member)?
        }
    };
    Ok(Located {
        param,
        member: member.0,
        last: member.0 .1,
        mem: member.1,
    })
}

/// Where buffer pointer `ptr` points ([`Within`]), `depth` argument buffers inside the
/// parameter's:
/// - into the argument buffer an entry parameter binds (a static root: the parameter, or an
///   element of its buffer table), at the offsets the memory plan knows;
/// - or into a **nested argument buffer**: `ptr` is the address loaded from an argument-buffer
///   member whose role is itself an argument buffer with a layout (`constant Sub *sub [[id(n)]]`,
///   its struct's `air.struct_type_info`), displaced by `getelementptr`s (constant indices, and at
///   most one dynamic index) and pointer casts. Its member is found in the nested layout as an
///   entry parameter's is in its own, and a member of a buffer nested in that one by the same step,
///   at most `LAYOUT_NESTING` deep. A pointer the load reaches any other way (a `phi` or `select`
///   of addresses, a pointer from memory that no such member holds) has no known member.
fn within(
    e: &Entry,
    mp: &MemoryPlan,
    body: &Body,
    ptr: Value,
    depth: u64,
) -> Result<Within, Error> {
    use crate::air::abi::{IndirectArg, MemberKind};
    let Some(Repr::BufPtr { roots, wide }) = mp.repr(ptr) else {
        return Err(unknown_member());
    };
    match (mp.roots(roots), wide) {
        ([Root::Param(param)] | [Root::BufferElement { param, .. }], false) => {
            let binding = mp.abi.binding(e, *param);
            let Some(ParamRole::ArgumentBuffer(arg)) = binding.map(|b| &b.role) else {
                return Err(unknown_member());
            };
            let layout = arg.layout.ok_or_else(unknown_member)?;
            let offsets = match (mp.offset(ptr), mp.strided(ptr)) {
                (Some(OffsetRange::Span { lo, hi }), _) => Offsets::Span { lo, hi },
                (_, Some((base, stride))) => Offsets::Strided { base, stride },
                _ => return Err(unknown_member()),
            };
            Ok(Within {
                param: *param,
                layout,
                // The size of one argument buffer the parameter points to (`air.arg_type_size`).
                element: binding.and_then(|b| b.ty.size),
                offsets,
            })
        }
        ([Root::Dynamic], _) if depth < crate::budget::LAYOUT_NESTING => {
            let (from, from_align, offsets) =
                nested_address(e, &mp.layouts, body, ptr).ok_or_else(unknown_member)?;
            let outer = located(e, mp, body, from, from_align, depth + 1)?;
            let MemberKind::Indirect(IndirectArg {
                role: ParamRole::ArgumentBuffer(arg),
                ty,
                ..
            }) = &outer.mem.kind
            else {
                return Err(unknown_member());
            };
            let layout = arg.layout.ok_or_else(unknown_member)?;
            if outer.last != outer.member.1 {
                return Err(unknown_member());
            }
            Ok(Within {
                param: outer.param,
                layout,
                element: ty.size,
                offsets,
            })
        }
        _ => Err(unknown_member()),
    }
}

/// The most `getelementptr`s and casts [`nested_address`] follows back to a load.
const NESTED_ADDRESS_STEPS: usize = 64;

/// The load that read pointer `ptr`'s base address from memory, as `(the load's address, its
/// alignment in bytes, ptr's offsets from the base)`: `ptr` is that load's result displaced by
/// `getelementptr`s whose indices are constant but for at most one dynamic index over one
/// stride, and by pointer casts. `None` for any other definition, or past
/// [`NESTED_ADDRESS_STEPS`] steps.
fn nested_address(
    e: &Entry,
    layouts: &crate::air::ir::TypeLayouts,
    body: &Body,
    ptr: Value,
) -> Option<(Value, u64, Offsets)> {
    use crate::air::ir::CastOp;
    let mut at = ptr;
    let mut konst: i64 = 0;
    let mut strides: Vec<u64> = Vec::new();
    for _ in 0..NESTED_ADDRESS_STEPS {
        let Value::Inst(id) = at else { return None };
        match &body.insts.get(id)?.kind {
            InstKind::Gep {
                src_ty,
                base,
                indices,
                ..
            } => {
                let (delta, more) =
                    super::provenance::gep_displacement(e, layouts, *src_ty, indices).ok()?;
                konst = konst.checked_add(delta?)?;
                strides.extend(more);
                at = *base;
            }
            InstKind::Cast {
                op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                v,
            } => at = *v,
            InstKind::Load {
                ptr: from,
                align,
                atomic: None,
                ..
            } => {
                let offsets = match strides.as_slice() {
                    [] => Offsets::Span {
                        lo: konst,
                        hi: konst,
                    },
                    [stride] => Offsets::Strided {
                        base: konst,
                        stride: *stride,
                    },
                    _ => return None,
                };
                return Some((*from, align.map_or(1, |a| a.bytes()), offsets));
            }
            _ => return None,
        }
    }
    None
}

/// What argument-buffer member `m` holds, if it is a texture or a sampler handle.
fn handle_of(m: &crate::air::abi::Member) -> Option<MemberHandle> {
    match m.handle()? {
        AbiHandle::Texture(AbiAccess::Sample | AbiAccess::Read) => {
            Some(MemberHandle::Texture(ImageAccess::Sampled))
        }
        AbiHandle::Texture(AbiAccess::Write | AbiAccess::ReadWrite) => {
            Some(MemberHandle::Texture(ImageAccess::Storage))
        }
        AbiHandle::Sampler => Some(MemberHandle::Sampler),
        _ => None,
    }
}

/// The index of the last member of the run a load reads whose offsets span `lo..=hi` from member
/// `first` to member `last` (the frontend's load through a `select` of two members' addresses):
/// `first`, `last`, and every member between them in their one layout are single handles (no array)
/// of one kind and access, each starting where the one before it ends, and the load's alignment
/// `align` is a multiple of their size, which `lo` (the run's start) is too, so every offset the
/// load can have starts one of them. Any of them is the handle a load at that offset reads, all of
/// one access, so the load is that access's heap handle. `None` for any other span.
fn member_run(
    structs: &[crate::air::abi::StructInfo],
    (first, last): (
        ((usize, usize), &crate::air::abi::Member),
        ((usize, usize), &crate::air::abi::Member),
    ),
    (lo, hi): (u64, u64),
    align: u64,
) -> Option<usize> {
    let ((layout, i), (other, j)) = (first.0, last.0);
    if layout != other || j <= i {
        return None;
    }
    let members = &structs.get(layout)?.members;
    let size = u64::from(first.1.size);
    let kind = handle_of(first.1)?;
    if size == 0 || !align.is_multiple_of(size) || !lo.is_multiple_of(size) {
        return None;
    }
    let mut end = u64::from(first.1.offset);
    for m in members.get(i..=j)? {
        let single = m.array_len == 0 && u64::from(m.size) == size;
        if !single || handle_of(m) != Some(kind) || u64::from(m.offset) != end {
            return None;
        }
        end += size;
    }
    // `hi` is the run's last member's start, as `lo` is its first's.
    (hi - lo == u64::from(last.1.offset) - u64::from(first.1.offset)).then_some(j)
}

/// That a sampler handle loaded through buffer pointer `ptr` with alignment `align` reads an
/// argument buffer's sampler member, known as for [`member_access`]; any other load is refused as
/// there.
pub fn sampler_member(
    e: &Entry,
    mp: &MemoryPlan,
    body: &Body,
    ptr: Value,
    align: u64,
) -> Result<(), Error> {
    match loaded_member(e, mp, body, ptr, align)?.access {
        MemberHandle::Sampler => Ok(()),
        MemberHandle::Texture(_) => Err(not_implemented(Feature::ArgumentBufferTextures)),
    }
}

/// The argument-buffer member a load at the offsets `base + k · stride` reads, for one dynamic
/// element index `k`, in layout `layout` of argument buffers `element` bytes apart: the member at
/// `base` (reduced into the first argument buffer when `stride` is `element`, an array of
/// argument buffers), provided the index walks an array that holds it, whose element is `stride`
/// bytes: the array of argument buffers, or a member array on the path to it (a nested struct
/// array included). Every in-bounds element of that array holds the member at the same place, and
/// an index outside the array is undefined in Metal (MSL's arrays follow C++'s), so the member is
/// the one every defined load reads; an undefined one still reads a resource id through the
/// argument buffer's own contained slot and finds a valid element or the neutral one. `None` when
/// no such array exists or `base` names no member. The member comes with its place in `structs`
/// (`member_place_at`).
fn strided_member<'s>(
    (abi, structs): (&AbiIndex, &'s [crate::air::abi::StructInfo]),
    layout: crate::air::abi::StructRef,
    element: Option<u32>,
    base: i64,
    stride: u64,
) -> Option<((usize, usize), &'s crate::air::abi::Member)> {
    use crate::air::abi::MemberKind;
    if stride == 0 {
        return None;
    }
    let mut walked = false;
    let mut off = base;
    if let Some(size) = element.filter(|s| u64::from(*s) == stride) {
        walked = true;
        off = base.rem_euclid(i64::from(size));
    }
    let mut off = u64::try_from(off).ok()?;
    let mut s = layout;
    for _ in 0..crate::budget::LAYOUT_NESTING {
        let info = structs.get(s.0 as usize)?;
        let index = abi.member_index(s, off)?;
        let mem = info.members.get(index)?;
        // A member array, or one `array<T, N>` member of `N` handles, whose element the index
        // may walk.
        let element = u64::from(member_element_size(mem));
        if (mem.array_len >= 1 || element != u64::from(mem.size)) && element == stride {
            walked = true;
        }
        let within = off - u64::from(mem.offset);
        let within = if mem.size > 0 {
            within % u64::from(mem.size)
        } else {
            within
        };
        match (mem.layout, &mem.kind) {
            (Some(inner), MemberKind::Plain | MemberKind::IndirectStruct { .. }) => {
                s = inner;
                off = within;
            }
            _ => {
                return (element > 0 && within.is_multiple_of(element) && walked)
                    .then_some(((s.0 as usize, index), mem))
            }
        }
    }
    None
}

/// The heap texture handles of a function, grouped by what `phi`, `select`, `freeze`, and casts
/// merge, and by the argument-buffer member their loads read: every handle of a group reaches one
/// heap array, whose class the texture family derives from all of the group's uses (as one
/// `[[texture(n)]]` slot's declaration comes from all of its uses). Loads of one member (or of
/// elements of one member array) name textures of its one declared type, so a read or write of the
/// member anywhere in the function types every handle loaded from it (the declared type itself is
/// only the MSL type name, which never decides anything). Each group has the access of the
/// argument-buffer members its loads read.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct HeapGroups {
    /// The group of each instruction defining a heap texture handle.
    group: BTreeMap<InstId, u32>,
    /// Each group's heap array access.
    access: Vec<ImageAccess>,
}

impl HeapGroups {
    /// Group the heap texture handles of `body`. A load (or the integer round trip of one,
    /// `member_load`) whose member access is not known, a merge
    /// of a heap handle with anything but heap and null handles, an `extractvalue` that no one
    /// `insertvalue` supplies, another instruction defining a heap handle, and a group whose
    /// members disagree on their access are refused.
    ///
    /// Metered: one work unit per instruction of the body, per heap handle, per value a handle
    /// merges, and per step of every find of the union-find ([`UnionFind`]); the working tables are
    /// charged to the ledger and returned, and the groups, which the plan keeps, stay charged.
    pub fn analyze(
        e: &Entry,
        body: &Body,
        mp: &MemoryPlan,
        m: &mut Meter,
    ) -> Result<HeapGroups, Error> {
        let mut work = Working::default();
        let groups = work.group(e, body, mp, m);
        m.release(work.charged.saturating_add(work.uf.charged()));
        groups
    }

    /// The group of the heap handle instruction `at` defines.
    pub fn group(&self, at: InstId) -> Option<u32> {
        self.group.get(&at).copied()
    }

    /// The heap array access of group `g`.
    pub fn access(&self, g: u32) -> Option<ImageAccess> {
        self.access.get(g as usize).copied()
    }

    /// The number of groups.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.access.len()
    }
}

/// The working tables of [`HeapGroups::analyze`], charged to the ledger as they grow.
#[derive(Default)]
struct Working {
    /// The union-find index of each heap handle, in instruction order.
    index: BTreeMap<InstId, u32>,
    uf: UnionFind,
    /// The access each handle's load reads.
    seed: Vec<Option<ImageAccess>>,
    /// The pairs of handles a merge or a load of one member joins.
    links: Vec<(u32, u32)>,
    /// The first load of each member, which every later load of it joins.
    first_load: BTreeMap<(u32, (usize, usize)), u32>,
    /// Each group's number, by its representative.
    number: BTreeMap<u32, u32>,
    /// The ledger bytes the tables hold, besides the union-find's.
    charged: usize,
}

impl Working {
    fn charge(&mut self, bytes: usize, m: &mut Meter) -> Result<(), Error> {
        m.charge(bytes)?;
        self.charged = self.charged.saturating_add(bytes);
        Ok(())
    }

    fn group(
        &mut self,
        e: &Entry,
        body: &Body,
        mp: &MemoryPlan,
        m: &mut Meter,
    ) -> Result<HeapGroups, Error> {
        for (id, _) in body.insts.iter() {
            m.tick(1)?;
            if is_heap(mp, Value::Inst(id)) {
                self.charge(map_entry_bytes::<InstId, u32>(), m)?;
                let i = self.uf.push(m)?;
                push_working(&mut self.seed, None, &mut self.charged, m)?;
                self.index.insert(id, i);
            }
        }
        let mut at = self.index.keys().next().copied();
        while let Some(id) = at {
            m.tick(1)?;
            let me = *self
                .index
                .get(&id)
                .ok_or_else(|| invariant("a heap handle without an index"))?;
            let inst = body
                .insts
                .get(id)
                .ok_or_else(|| invariant("a heap handle outside the body"))?;
            // A load of a member, or the integer round trip of one.
            match (member_load(e, mp, body, inst), &inst.kind) {
                (Some((ptr, align)), _) => {
                    let member = loaded_member(e, mp, body, ptr, align)?;
                    let MemberHandle::Texture(access) = member.access else {
                        return Err(not_implemented(Feature::ArgumentBufferTextures));
                    };
                    if let Some(s) = self.seed.get_mut(me as usize) {
                        *s = Some(access);
                    }
                    // Joined with every member it may read (one, or a run of them).
                    for index in member.member.1..=member.last {
                        m.tick(1)?;
                        let key = (member.param, (member.member.0, index));
                        match self.first_load.get(&key) {
                            Some(&o) => {
                                push_working(&mut self.links, (me, o), &mut self.charged, m)?
                            }
                            None => {
                                self.charge(map_entry_bytes::<(u32, (usize, usize)), u32>(), m)?;
                                self.first_load.insert(key, me);
                            }
                        }
                    }
                }
                (None, InstKind::Phi { incoming, .. }) => {
                    for (v, _) in incoming.iter() {
                        self.link(e, mp, me, *v, m)?;
                    }
                }
                (None, InstKind::Select { t, f, .. }) => {
                    self.link(e, mp, me, *t, m)?;
                    self.link(e, mp, me, *f, m)?;
                }
                (None, InstKind::Freeze(v) | InstKind::Cast { v, .. }) => {
                    self.link(e, mp, me, *v, m)?
                }
                // A handle read out of an aggregate is the value inserted there.
                (None, InstKind::ExtractValue { agg, indices }) => {
                    match crate::lower::arith::aggregate::source(body, *agg, indices) {
                        Some(v) => self.link(e, mp, me, v, m)?,
                        None => return Err(not_implemented(Feature::ArgumentBufferTextures)),
                    }
                }
                _ => return Err(not_implemented(Feature::ArgumentBufferTextures)),
            }
            at = self
                .index
                .range((std::ops::Bound::Excluded(id), std::ops::Bound::Unbounded))
                .next()
                .map(|(k, _)| *k);
        }
        let mut k = 0;
        while let Some(&(a, b)) = self.links.get(k) {
            m.tick(1)?;
            self.uf.union(a, b, m)?;
            k += 1;
        }
        // Number the groups densely in the order of their first handle, and join their accesses.
        let mut access: Vec<Option<ImageAccess>> = Vec::new();
        let mut group = BTreeMap::new();
        let mut retained = 0usize;
        let mut at = self.index.iter().next().map(|(k, v)| (*k, *v));
        while let Some((id, me)) = at {
            m.tick(1)?;
            let root = self.uf.find(me, m)?;
            let g = match self.number.get(&root) {
                Some(&g) => g,
                None => {
                    let g = u32::try_from(self.number.len())
                        .map_err(|_| invariant("more heap handle groups than u32"))?;
                    self.charge(map_entry_bytes::<u32, u32>(), m)?;
                    self.number.insert(root, g);
                    g
                }
            };
            if g as usize == access.len() {
                push_working(&mut access, None, &mut retained, m)?;
            }
            let slot = access
                .get_mut(g as usize)
                .ok_or_else(|| invariant("a heap group out of range"))?;
            match (*slot, self.seed.get(me as usize).copied().flatten()) {
                (Some(a), Some(b)) if a != b => {
                    return Err(not_implemented(Feature::ArgumentBufferTextures))
                }
                (None, b) => *slot = b,
                _ => {}
            }
            m.charge(map_entry_bytes::<InstId, u32>())?;
            group.insert(id, g);
            at = self
                .index
                .range((std::ops::Bound::Excluded(id), std::ops::Bound::Unbounded))
                .next()
                .map(|(k, v)| (*k, *v));
        }
        // A group with no load merges only null handles, which the domain calls static.
        let access = access
            .into_iter()
            .map(|a| a.ok_or_else(|| invariant("a heap handle group without a load")))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(HeapGroups { group, access })
    }

    /// Join heap handle `me` with operand `v` of its merge: another heap handle, or a null handle
    /// (which joins nothing). Any other operand is refused.
    fn link(
        &mut self,
        e: &Entry,
        mp: &MemoryPlan,
        me: u32,
        v: Value,
        m: &mut Meter,
    ) -> Result<(), Error> {
        m.tick(1)?;
        if let Value::Inst(d) = v {
            if let Some(&o) = self.index.get(&d) {
                return push_working(&mut self.links, (me, o), &mut self.charged, m);
            }
        }
        if is_null_handle(e, mp, v) {
            return Ok(());
        }
        Err(not_implemented(Feature::ArgumentBufferTextures))
    }
}

/// The handle table as a module declares it: the block's layout is the binder's 16-byte entries
/// either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Table {
    global: SGlobalId,
    u32_ty: TypeId,
    /// The `u64` type when the table is an entry array `{ Entry entries[]; }`, `Entry { u64 id;
    /// u32 element; u32 flags; }` (a profile with `shaderInt64`), whose search compares each id
    /// in one instruction; `None` for a word view `{ u32 words[]; }`, which compares an id's two
    /// words.
    u64_ty: Option<TypeId>,
}

/// Declare the handle table at binding symbol `sym`: an entry array when `int64` (the profile
/// has `shaderInt64`), else a word storage view.
pub fn declare_table(mb: &mut ModuleBuilder, sym: ResourceSlot, int64: bool) -> BResult<Table> {
    if !int64 {
        let v = views::declare_at(mb, sym, views::ViewKind::Storage)?;
        return Ok(Table {
            global: v.global,
            u32_ty: v.u32_ty,
            u64_ty: None,
        });
    }
    let int = |width| SType::Int {
        width,
        signed: false,
    };
    let (u32_ty, u64_ty) = (mb.ty(int(32))?, mb.ty(int(64))?);
    let fields = [u64_ty, u32_ty, u32_ty];
    let layout = natural_layout(mb.types(), LayoutRule::Std430, false, &fields)?;
    let entry = mb.ty(SType::Struct {
        members: fields.into(),
        layout: Some(layout),
    })?;
    let entries = mb.ty(SType::RuntimeArray {
        elem: entry,
        stride: Some(TABLE_ENTRY_BYTES),
    })?;
    let layout = natural_layout(mb.types(), LayoutRule::Std430, true, &[entries])?;
    let block = mb.ty(SType::Struct {
        members: [entries].into(),
        layout: Some(layout),
    })?;
    let global = mb.resource(sym, StorageClass::StorageBuffer, block)?;
    Ok(Table {
        global,
        u32_ty,
        u64_ty: Some(u64_ty),
    })
}

/// `a ≤ b` for two 64-bit values held as `(low, high)` word pairs.
fn le64(fb: &mut FunctionBuilder<'_>, a: (SValue, SValue), b: (SValue, SValue)) -> BResult<SValue> {
    let hi_lt = fb.binary(BinOp::ULessThan, a.1, b.1)?;
    let hi_eq = fb.binary(BinOp::IEqual, a.1, b.1)?;
    let lo_le = fb.binary(BinOp::ULessThanEqual, a.0, b.0)?;
    let tie = fb.binary(BinOp::LogicalAnd, hi_eq, lo_le)?;
    fb.binary(BinOp::LogicalOr, hi_lt, tie)
}

/// Word `field` of entry `k` (a `u32`) of the word view `table`.
fn entry_word(fb: &mut FunctionBuilder<'_>, table: View, k: SValue, field: u32) -> BResult<SValue> {
    let stride = k32(fb, ENTRY_WORDS)?;
    let at = fb.binary(BinOp::IMul, k, stride)?;
    let at = if field == 0 {
        at
    } else {
        let f = k32(fb, field)?;
        fb.binary(BinOp::IAdd, at, f)?
    };
    views::read_word(fb, table, at)
}

/// Member `member` of entry `k` of the entry array `t` (0 the id, 1 the element, 2 the flags).
fn entry_member(fb: &mut FunctionBuilder<'_>, t: Table, k: SValue, member: u32) -> BResult<SValue> {
    let base = fb.module().global_value(t.global)?;
    let zero = k32(fb, 0)?;
    let m = k32(fb, member)?;
    let p = fb.access_chain(base, &[zero, k, m])?;
    fb.load(p, MemAccess::default())
}

/// What the handle-table search yields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lookup {
    /// The entry's element.
    Element,
    /// The entry's element with flags of its entry beside it: where `alpha`, its
    /// [`FLAG_OUT_OF_BOUNDS_ALPHA_ONE`] in [`HANDLE_ALPHA_ONE`], the handle of a module whose
    /// shader-checked heap read observes the alpha; where `class`, its [`FLAG_TEXEL_CLASS`] in
    /// [`HANDLE_TEXEL_CLASS`], the handle of a module that reads a heap texture's texel class;
    /// where `red_blue`, its [`FLAG_RED_BLUE`] in [`HANDLE_RED_BLUE`], the handle of a module whose
    /// storage access converts a member by it (`bind::heap::storage_conversions`). Every heap
    /// access masks them off ([`Lookup::element_mask`]).
    Flagged {
        alpha: bool,
        class: bool,
        red_blue: bool,
    },
    /// The index of the entry found, 0 (the null entry) for a miss: the handle of an argument
    /// buffer's sampler, whose uses read the entry's element and word ([`entry_field`]).
    Entry,
}

impl Lookup {
    /// The entry flags this form carries into the handle, each with the left shift that moves it
    /// there.
    fn carried(self) -> Vec<(u32, u32)> {
        let Lookup::Flagged {
            alpha,
            class,
            red_blue,
        } = self
        else {
            return Vec::new();
        };
        let alpha = alpha.then_some((
            FLAG_OUT_OF_BOUNDS_ALPHA_ONE,
            HANDLE_ALPHA_ONE.trailing_zeros() - FLAG_OUT_OF_BOUNDS_ALPHA_ONE.trailing_zeros(),
        ));
        let class = class.then_some((FLAG_TEXEL_CLASS, 0));
        let red_blue = red_blue.then_some((
            FLAG_RED_BLUE,
            HANDLE_RED_BLUE.trailing_zeros() - FLAG_RED_BLUE.trailing_zeros(),
        ));
        alpha.into_iter().chain(class).chain(red_blue).collect()
    }

    /// The bits of a handle this form yields that are its element, where it carries flags beside
    /// it: every bit but the carried ones. An element is below [`TABLE_CAPACITY`], so no carried
    /// bit is ever an element's.
    pub fn element_mask(self) -> Option<u32> {
        let carried = self
            .carried()
            .into_iter()
            .fold(0, |bits, (flags, shift)| bits | flags << shift);
        (carried != 0).then_some(!carried)
    }
}

/// A field of a handle-table entry that a use reads by the entry's index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryField {
    /// The heap element.
    Element,
    /// The flags: a sampler entry's [`SamplerWord`](crate::metal::sampler::SamplerWord).
    Flags,
}

/// An upper estimate of [`entry_field`]'s words: over a word view, the index's scaling and offset,
/// the access chain, and the load; over an entry array, the access chain and the load; with their
/// constants.
pub const ENTRY_FIELD_WORDS: u32 = 32;

/// Field `field` of entry `k` of the handle table `t`, a [`Lookup::Entry`] handle: the null
/// entry's (element 0, word 0) for a miss. `k` is below the table's length, the search's answer.
pub fn entry_field(
    fb: &mut FunctionBuilder<'_>,
    t: Table,
    k: SValue,
    field: EntryField,
) -> BResult<SValue> {
    let (member, word) = match field {
        EntryField::Element => (1, ENTRY_ELEMENT),
        EntryField::Flags => (2, ENTRY_FLAGS),
    };
    match t.u64_ty {
        Some(_) => entry_member(fb, t, k, member),
        None => {
            let words = View {
                global: t.global,
                kind: views::ViewKind::Storage,
                u32_ty: t.u32_ty,
            };
            entry_word(fb, words, k, word)
        }
    }
}

/// The index of the last entry of the handle table `t` (its largest id's), from its length
/// (`OpArrayLength`): the null entry's, 0, where the table holds no other.
pub fn last_entry(fb: &mut FunctionBuilder<'_>, t: Table) -> BResult<SValue> {
    let base_ptr = fb.module().global_value(t.global)?;
    let len = fb.array_length(base_ptr, 0)?;
    let count = match t.u64_ty {
        Some(_) => len,
        None => {
            let two = k32(fb, 2)?;
            fb.binary(BinOp::ShiftRightLogical, len, two)?
        }
    };
    let one = k32(fb, 1)?;
    fb.binary(BinOp::ISub, count, one)
}

/// The heap element of the resource whose 64-bit id is `(lo, hi)`, through the handle table `t`
/// (the module docs' layout): a branch-free binary search of [`SEARCH_STEPS`] steps for the last
/// entry whose id is at most the key, then the entry's element if its id is the key, else
/// [`NEUTRAL_ELEMENT`]. Entry 0 is the null entry (id 0), so the search always finds an entry,
/// and every entry it reads is below the table's length (`OpArrayLength`). An entry array
/// compares each id as one `u64`; a word view compares its two words. [`Lookup::Flagged`] also
/// moves the found entry's flags it carries into their handle bits (a miss is
/// [`NEUTRAL_ELEMENT`] without them), and [`Lookup::Entry`] yields the found entry's index in
/// place of its element, 0 for a miss.
pub fn lookup(
    fb: &mut FunctionBuilder<'_>,
    t: Table,
    lo: SValue,
    hi: SValue,
    form: Lookup,
) -> BResult<SValue> {
    let base_ptr = fb.module().global_value(t.global)?;
    let len = fb.array_length(base_ptr, 0)?;
    let one = k32(fb, 1)?;
    let zero = k32(fb, 0)?;
    let words = View {
        global: t.global,
        kind: views::ViewKind::Storage,
        u32_ty: t.u32_ty,
    };
    // The key, and the id of an entry, in the table's form.
    let (key, count) = match t.u64_ty {
        Some(u64_ty) => {
            let l = fb.convert(ConvOp::UConvert, lo, u64_ty)?;
            let h = fb.convert(ConvOp::UConvert, hi, u64_ty)?;
            let s = k32(fb, 32)?;
            let h = fb.binary(BinOp::ShiftLeftLogical, h, s)?;
            (fb.binary(BinOp::BitwiseOr, h, l)?, len)
        }
        None => {
            let two = k32(fb, 2)?;
            (lo, fb.binary(BinOp::ShiftRightLogical, len, two)?)
        }
    };
    let at_most_key = |fb: &mut FunctionBuilder<'_>, k: SValue| -> BResult<SValue> {
        match t.u64_ty {
            Some(_) => {
                let id = entry_member(fb, t, k, 0)?;
                fb.binary(BinOp::ULessThanEqual, id, key)
            }
            None => {
                let id_lo = entry_word(fb, words, k, ENTRY_ID_LO)?;
                let id_hi = entry_word(fb, words, k, ENTRY_ID_HI)?;
                le64(fb, (id_lo, id_hi), (lo, hi))
            }
        }
    };
    let mut base = zero;
    let mut len = count;
    for _ in 0..SEARCH_STEPS {
        let half = fb.binary(BinOp::ShiftRightLogical, len, one)?;
        let mid = fb.binary(BinOp::IAdd, base, half)?;
        let take = at_most_key(fb, mid)?;
        base = fb.select(take, mid, base)?;
        len = fb.binary(BinOp::ISub, len, half)?;
    }
    let hit = match t.u64_ty {
        Some(_) => {
            let id = entry_member(fb, t, base, 0)?;
            fb.binary(BinOp::IEqual, id, key)?
        }
        None => {
            let id_lo = entry_word(fb, words, base, ENTRY_ID_LO)?;
            let id_hi = entry_word(fb, words, base, ENTRY_ID_HI)?;
            let eq_lo = fb.binary(BinOp::IEqual, id_lo, lo)?;
            let eq_hi = fb.binary(BinOp::IEqual, id_hi, hi)?;
            fb.binary(BinOp::LogicalAnd, eq_lo, eq_hi)?
        }
    };
    if form == Lookup::Entry {
        // The null entry, index 0, for a miss.
        return fb.select(hit, base, zero);
    }
    let mut element = entry_field(fb, t, base, EntryField::Element)?;
    let carried = form.carried();
    if !carried.is_empty() {
        let flags = entry_field(fb, t, base, EntryField::Flags)?;
        for (carried_flags, shift) in carried {
            let mask = k32(fb, carried_flags)?;
            let carried = fb.binary(BinOp::BitwiseAnd, flags, mask)?;
            let shift = k32(fb, shift)?;
            let carried = fb.binary(BinOp::ShiftLeftLogical, carried, shift)?;
            element = fb.binary(BinOp::BitwiseOr, element, carried)?;
        }
    }
    let neutral = k32(fb, NEUTRAL_ELEMENT)?;
    fb.select(hit, element, neutral)
}

/// An upper estimate of [`define_lookup`]'s words, counted once per module: per search step, over
/// an entry array, the id's access chain and load, the compare, and the step's arithmetic, or, over
/// a word view, two word reads (about 20 words each) and a five-instruction compare; then the
/// entry's id and element, and the function's own instructions and types. The flagged form also
/// reads the entry's flags (a word read) and moves each carried field (three instructions and two
/// constants); the entry form reads no element.
pub fn lookup_routine_words(int64: bool, form: Lookup) -> u32 {
    let step = if int64 { 48 } else { 96 };
    let carried = 48 * form.carried().len() as u32;
    SEARCH_STEPS * step + 160 + carried
}

/// Define the module's handle-table search: a function `fn(lo: u32, hi: u32) -> u32`, the heap
/// handle of the resource id `(lo, hi)` ([`lookup`] over `t`, in `form`). Every heap handle load
/// of the module calls it, so the search's words are paid once, not per load (as the shared wide
/// division).
pub fn define_lookup(mb: &mut ModuleBuilder, t: Table, form: Lookup) -> BResult<FuncId> {
    let mut fb = mb.function(t.u32_ty)?;
    let lo = fb.param(t.u32_ty)?;
    let hi = fb.param(t.u32_ty)?;
    let handle = lookup(&mut fb, t, lo, hi, form)?;
    fb.ret(Some(handle))?;
    fb.finish()
}

/// The search [`lookup`] emits, on the host: the element of `id` in `table`, a slice of
/// `(resource_id, element)` entries sorted by id whose first entry is the null entry. The
/// emitter's reference model, which every table a binder builds must satisfy.
pub fn lookup_host(table: &[(u64, u32)], id: u64) -> u32 {
    let (mut base, mut len) = (0usize, table.len());
    for _ in 0..SEARCH_STEPS {
        let half = len >> 1;
        let mid = base + half;
        if table.get(mid).is_some_and(|(k, _)| *k <= id) {
            base = mid;
        }
        len -= half;
    }
    match table.get(base) {
        Some((k, e)) if *k == id => *e,
        _ => NEUTRAL_ELEMENT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_indices_keep_the_classes_apart() {
        assert_eq!(Candidate::Null.handle(), NULL_HANDLE);
        assert_eq!(Candidate::Texture(0).handle(), 1);
        assert_eq!(Candidate::Texture(127).handle(), 128);
        assert_eq!(Candidate::Sampler(15).handle(), 16);
        let g = GlobalId::new(3);
        assert_eq!(
            Candidate::Constexpr {
                ordinal: 0,
                global: g
            }
            .handle(),
            CONSTEXPR_SAMPLER_BASE
        );
        // No runtime sampler slot shares a constexpr sampler's index.
        assert!(Candidate::Sampler(SAMPLER_SLOTS - 1).handle() < CONSTEXPR_SAMPLER_BASE);
    }

    /// `air.struct_type_info` members for [`strided_member`]'s tests.
    mod layout {
        use crate::air::abi::{
            AbiInt, Access, ArgTypeInfo, IndirectArg, Member, MemberKind, ParamRole, Slot,
            StructInfo, StructRef, TextureArg,
        };
        use crate::air::error::DiagName;

        fn member(offset: u32, size: u32, array_len: u32, kind: MemberKind) -> Member {
            Member {
                offset,
                size,
                array_len,
                layout: None,
                kind,
                render_target: None,
                raster_order_group: None,
                type_name: DiagName::new("t"),
                name: DiagName::new("m"),
                key: crate::air::abi::UserKey(b"m".as_slice().into()),
            }
        }

        /// A texture member of `count` handles (an `array<texture2d, count>` when above 1).
        pub fn texture(offset: u32, size: u32, array_len: u32, count: u32) -> Member {
            member(
                offset,
                size,
                array_len,
                MemberKind::Indirect(IndirectArg {
                    ordinal: 0,
                    role: ParamRole::Texture(TextureArg {
                        location: Slot {
                            index: AbiInt::Const(0),
                            count: AbiInt::Const(count),
                        },
                        access: Access::Sample,
                        raster_order_group: None,
                    }),
                    ty: ArgTypeInfo {
                        size: None,
                        align: None,
                        name: None,
                    },
                    name: None,
                }),
            )
        }

        pub fn plain(offset: u32, size: u32) -> Member {
            member(offset, size, 0, MemberKind::Plain)
        }

        /// A member array of `array_len` structs of layout `inner`, each `size` bytes.
        pub fn structs(offset: u32, size: u32, array_len: u32, inner: u32) -> Member {
            Member {
                layout: Some(StructRef(inner)),
                ..member(offset, size, array_len, MemberKind::Plain)
            }
        }

        pub fn info(members: Vec<Member>) -> StructInfo {
            StructInfo { members }
        }
    }

    /// One dynamic index names a member exactly when it walks an array holding it: a member
    /// array, an `array<texture2d, N>` member, a nested struct array, or the array of argument
    /// buffers; an index that walks none, or a base between elements, names none.
    #[test]
    fn a_strided_load_names_the_member_its_array_holds() {
        use crate::air::abi::StructRef;
        use layout::*;
        // 0: { tex a[4] (0..32); tex b<4> (32..64, one array<> member); int x (64);
        //      Inner c[2] (72..104) } with Inner = 1: { int y (0); tex t (8) }.
        let structs = vec![
            info(vec![
                texture(0, 8, 4, 1),
                texture(32, 32, 0, 4),
                plain(64, 4),
                structs(72, 16, 2, 1),
            ]),
            info(vec![plain(0, 4), texture(8, 8, 0, 1)]),
        ];
        let root = StructRef(0);
        let abi =
            AbiIndex::layouts(&structs, &mut crate::pipeline::testing::meter()).expect("index");
        let at = |element: Option<u32>, base: i64, stride: u64| {
            strided_member((&abi, &structs), root, element, base, stride).map(|(_, m)| m.offset)
        };
        // An element of the member array `a`, from its first element or a later one.
        assert_eq!(at(None, 0, 8), Some(0));
        assert_eq!(at(None, 16, 8), Some(0));
        // An element of the `array<>` member `b`.
        assert_eq!(at(None, 40, 8), Some(32));
        // The texture of an element of the struct array `c`.
        assert_eq!(at(None, 80, 16), Some(8));
        // The array of argument buffers (104 bytes each): member `a`, from a later element too.
        assert_eq!(at(Some(104), 8, 104), Some(0));
        assert_eq!(at(Some(104), 104 + 40, 104), Some(32));
        // A stride no array holding the member has, a base inside an element, a plain member.
        assert_eq!(at(None, 0, 16), None);
        assert_eq!(at(None, 4, 8), None);
        assert_eq!(at(None, 64, 8), None);
        assert_eq!(at(Some(104), 8, 208), None);
        assert_eq!(at(None, 0, 0), None);
    }

    #[test]
    fn array_elements_follow_the_offset_range() {
        let span = |lo, hi| OffsetRange::Span { lo, hi };
        assert_eq!(array_elements(span(8, 8), 8, 4), Some((1, 1)));
        assert_eq!(array_elements(span(0, 24), 8, 4), Some((0, 3)));
        assert_eq!(array_elements(OffsetRange::Top, 8, 4), Some((0, 3)));
        // Clipped to the array; wholly outside is none.
        assert_eq!(array_elements(span(-8, 40), 8, 4), Some((0, 3)));
        assert_eq!(array_elements(span(32, 64), 8, 4), None);
        assert_eq!(array_elements(span(-16, -8), 8, 4), None);
        assert_eq!(array_elements(span(0, 0), 8, 0), None);
    }

    /// A handle table of `n` entries: the null entry, then ids spread over both words, strictly
    /// increasing, each with its own element.
    fn table(n: usize) -> Vec<(u64, u32)> {
        (0..n as u64)
            .map(|i| match i {
                0 => (0, NEUTRAL_ELEMENT),
                _ => (i * 0x0000_0100_0000_0005 + 7, 1000 + i as u32),
            })
            .collect()
    }

    /// Present ids, the null id, ids between and around entries, and the top of the range.
    fn keys(t: &[(u64, u32)]) -> Vec<u64> {
        let mut k = vec![0, 1, u64::MAX];
        for (id, _) in t {
            k.extend([*id, id.wrapping_sub(1), id.wrapping_add(1)]);
        }
        k
    }

    /// The element of `id` by the library's search.
    fn reference(t: &[(u64, u32)], id: u64) -> u32 {
        t.binary_search_by_key(&id, |(k, _)| *k)
            .map_or(NEUTRAL_ELEMENT, |i| t[i].1)
    }

    /// The host model finds every entry of sorted tables up to [`TABLE_CAPACITY`] entries in
    /// [`SEARCH_STEPS`] steps, and maps every absent id (and the null id) to the neutral element.
    #[test]
    fn the_search_finds_every_entry_up_to_the_capacity() {
        assert_eq!(1u64 << SEARCH_STEPS, u64::from(TABLE_CAPACITY));
        for n in [1, 2, 3, 4, 5, 16, 17, 100, 4097, TABLE_CAPACITY as usize] {
            let t = table(n);
            for id in keys(&t) {
                assert_eq!(lookup_host(&t, id), reference(&t, id), "n {n} id {id:#x}");
            }
        }
    }

    const KEYS: u32 = 8;

    /// A kernel that looks up the [`KEYS`] ids (low, high word pairs) at the start of the
    /// buffer at slot 1 in the handle table at slot 0 by calling [`define_lookup`]'s search in
    /// `form`, and writes each element after them. The table is an entry array when `int64`, else
    /// a word view.
    fn lookup_module(int64: bool, form: Lookup) -> BResult<crate::spirv::ir::TypedModule> {
        use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
        use crate::spirv::builder::ModuleBuilder;
        use crate::spirv::ir::{ExecMode, ExecModel, MemAccess, Type};
        let sym = |index| ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        };
        let mut b = ModuleBuilder::new();
        let t = declare_table(&mut b, sym(0), int64)?;
        let io = views::declare_at(&mut b, sym(1), views::ViewKind::Storage)?;
        let search = define_lookup(&mut b, t, form)?;
        let void = b.ty(Type::Void)?;
        let mut f = b.function(void)?;
        for i in 0..KEYS {
            let at_lo = k32(&mut f, 2 * i)?;
            let at_hi = k32(&mut f, 2 * i + 1)?;
            let lo = views::read_word(&mut f, io, at_lo)?;
            let hi = views::read_word(&mut f, io, at_hi)?;
            let e = f
                .call(search, &[lo, hi])?
                .expect("the search returns the element");
            let at = k32(&mut f, 2 * KEYS + i)?;
            let p = views::word_ptr(&mut f, io, at)?.expect("a storage view is writable");
            f.store(p, e, MemAccess::default())?;
        }
        f.ret(None)?;
        let id = f.finish()?;
        b.entry(
            ExecModel::GLCompute,
            id,
            vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
        )?;
        b.finalize()
    }

    /// The emitted search is the host model: over the CPU interpreter, every key of tables of
    /// several sizes up to [`TABLE_CAPACITY`] entries (the null id, present ids, absent ids between
    /// and beyond them) gets [`lookup_host`]'s element, through an entry array (`u64` ids) and a
    /// word view. [`Lookup::Flagged`] also carries a found entry's [`FLAG_OUT_OF_BOUNDS_ALPHA_ONE`]
    /// in [`HANDLE_ALPHA_ONE`] where it carries the alpha, its texel class where it carries that,
    /// and its [`FLAG_RED_BLUE`] in [`HANDLE_RED_BLUE`] where it carries that, carrying no other
    /// bit of the entry, and [`Lookup::Entry`] yields the index
    /// of the last entry of the key's id, 0 for a miss.
    #[test]
    fn the_emitted_search_matches_the_host_model() {
        for int64 in [true, false] {
            let flagged = |alpha, class, red_blue| Lookup::Flagged {
                alpha,
                class,
                red_blue,
            };
            for form in [
                Lookup::Element,
                flagged(true, false, false),
                flagged(false, true, false),
                flagged(false, false, true),
                flagged(true, true, false),
                flagged(true, true, true),
                Lookup::Entry,
            ] {
                emitted_search_matches_the_host_model(int64, form);
            }
        }
    }

    /// Each flagged form's element mask clears exactly the bits it carries.
    #[test]
    fn a_flagged_handle_masks_off_the_bits_it_carries() {
        let flagged = |alpha, class, red_blue| {
            Lookup::Flagged {
                alpha,
                class,
                red_blue,
            }
            .element_mask()
        };
        assert_eq!(Lookup::Element.element_mask(), None);
        assert_eq!(Lookup::Entry.element_mask(), None);
        assert_eq!(flagged(false, false, false), None);
        assert_eq!(flagged(true, false, false), Some(!HANDLE_ALPHA_ONE));
        assert_eq!(HANDLE_TEXEL_CLASS, 0x1fff << 16);
        assert_eq!(flagged(false, true, false), Some(!HANDLE_TEXEL_CLASS));
        assert_eq!(flagged(false, false, true), Some(!HANDLE_RED_BLUE));
        assert_eq!(
            flagged(true, true, true),
            Some(!(HANDLE_ALPHA_ONE | HANDLE_TEXEL_CLASS | HANDLE_RED_BLUE)),
            "every carried field is masked off, and the element's 16 bits stay"
        );
    }

    /// The flags of the test tables' entry of `element`: none on the null entry, else its low six
    /// bits and a class index of it, so each carried flag occurs set and clear, beside bits no
    /// form carries.
    fn flags(element: u32) -> u32 {
        if element == NEUTRAL_ELEMENT {
            return 0;
        }
        (element % 64) | ((element.wrapping_mul(2_654_435_761) >> 7) & FLAG_TEXEL_CLASS)
    }

    fn emitted_search_matches_the_host_model(int64: bool, form: Lookup) {
        use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
        use crate::spirv::interp::{run, Buffer, Config, Resources};
        let m = lookup_module(int64, form).expect("module");
        let sym = |index| ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        };
        for n in [1, 2, 3, 5, 17, 100, TABLE_CAPACITY as usize] {
            let t = table(n);
            let bytes: Vec<u8> = t
                .iter()
                .flat_map(|(id, e)| {
                    [*id as u32, (*id >> 32) as u32, *e, flags(*e)]
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                })
                .collect();
            assert_eq!(bytes.len(), n * TABLE_ENTRY_BYTES as usize);
            let all = keys(&t);
            // The interesting keys of the large table: its ends, its middle, and absent ones.
            let all: Vec<u64> = if n > 100 {
                let mid = t[n / 2].0;
                let last = t[n - 1].0;
                vec![0, 1, t[1].0, mid, mid + 1, last, last + 1, u64::MAX]
            } else {
                all
            };
            for chunk in all.chunks(KEYS as usize) {
                let mut io: Vec<u8> = Vec::new();
                for i in 0..KEYS as usize {
                    let id = chunk.get(i).copied().unwrap_or(0);
                    io.extend((id as u32).to_le_bytes());
                    io.extend(((id >> 32) as u32).to_le_bytes());
                }
                io.extend(vec![0xAA; 4 * KEYS as usize]);
                let res = Resources {
                    buffers: vec![
                        Buffer {
                            slot: Some(sym(0)),
                            address: None,
                            bytes: bytes.clone(),
                        },
                        Buffer {
                            slot: Some(sym(1)),
                            address: None,
                            bytes: io,
                        },
                    ],
                    push: Vec::new(),
                    spec: Default::default(),
                };
                let out = run(&m, &Config::default(), res).expect("run");
                let io = &out.buffers[1];
                for (i, id) in chunk.iter().enumerate() {
                    let at = 4 * (2 * KEYS as usize + i);
                    let got = u32::from_le_bytes(io[at..at + 4].try_into().expect("word"));
                    let e = lookup_host(&t, *id);
                    let carried = |on: bool, flag: u32, bit: u32| match on {
                        true if flags(e) & flag != 0 => bit,
                        _ => 0,
                    };
                    let want = match form {
                        Lookup::Element => e,
                        Lookup::Flagged {
                            alpha,
                            class,
                            red_blue,
                        } => {
                            e | carried(alpha, FLAG_OUT_OF_BOUNDS_ALPHA_ONE, HANDLE_ALPHA_ONE)
                                | carried(red_blue, FLAG_RED_BLUE, HANDLE_RED_BLUE)
                                | match class {
                                    true => flags(e) & FLAG_TEXEL_CLASS,
                                    false => 0,
                                }
                        }
                        Lookup::Entry => {
                            t.iter().rposition(|(k, _)| k == id).map_or(0, |i| i as u32)
                        }
                    };
                    assert_eq!(got, want, "int64 {int64} {form:?} n {n} id {id:#x}");
                }
            }
        }
    }
}
