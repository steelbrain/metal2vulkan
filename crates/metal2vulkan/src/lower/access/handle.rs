//! Texture and sampler handle values.
//!
//! A texture or sampler value with a static candidate set (`memory::handles`) is its `u32`
//! handle index (`DemoteTy::HandleIndex`): a handle parameter is its slot's constant index, the
//! null handle 0, and `select`, `phi`, `freeze`, and casts of handles are integer operations.
//! Operations that use a handle resolve it where they are selected (the texture and sampler
//! families, [`texture_target`] and [`sampler_target`]).
//!
//! **Handle arrays** (`array<texture2d<T>, N> [[texture(n)]]`, `Repr::HandleArray`): the
//! parameter and every GEP, cast, `freeze`, or `select` of it are `u32` byte offsets into the
//! array (the buffer-table form); a load of an element yields the element's handle index, `1 + n +
//! k`. As for a buffer table, the element `k = offset / size` of a dynamic offset is **clamped**
//! into the elements the load may read (`memory::handles::array_elements`): Metal leaves an index
//! outside the array undefined, and the clamp keeps it on a bound slot. The division is exact for
//! an aligned load, and a load with a smaller alignment (which would read parts of two handles) is
//! not implemented. The parameter's IR type (a thread pointer) makes it an array, not its location
//! count: an `array_ref` of one handle is an array too.
//!
//! **Handle words**. A plain load of an integer as wide as a handle through a handle-array element
//! pointer, whose every use is an `inttoptr` that provenance made that element's handle (the
//! frontend's integer copy of a handle, `memory::provenance::round_trip`), is the element's handle
//! index ([`plan_words`]), and each `inttoptr` a copy of it. The integer, the guest's bits of the
//! handle, is never observed: a load with any other use stays refused. The same holds for a plain
//! integer load of an argument-buffer member's resource id whose every use is an `inttoptr` to a
//! heap texture or sampler (the frontend's form of a `select` between two members, `cond ? ab.a :
//! ab.b`): the load is the member's heap lookup ([`HandleStrategy::Lookup`]), as a pointer load of
//! the member is, and each `inttoptr` a copy.
//!
//! **Heap handles** (`memory::handles`). A texture handle loaded from an argument buffer is its
//! set-1 heap element: the load reads the member's 8-byte guest resource id as two words through
//! the argument buffer's slot (an ordinary contained leaf access) and searches the stage's handle
//! table for it ([`HandleStrategy::Lookup`], `access.handle.lookup`) by calling the module's one
//! search function ([`HandleTable::lookup`]); a miss is the neutral element 0, the null handle.
//! Merges of heap handles are the integer operations above; a texture operation on one indexes its
//! group's heap array ([`heap_target`], the texture family). The function then reads the handle
//! table, one more storage descriptor of the stage. Where a shader-checked heap read observes its
//! alpha, the search also carries the entry's out-of-bounds-alpha flag in the handle's bit 31,
//! where a heap access reads its texel class the class's index in bits 16 to 28, and where a heap
//! storage access converts by its red-blue order that flag in bit 30
//! (`memory::handles::Lookup::Flagged`); where a heap read through a sampler may hand back its
//! border texels, the search carries the entry's index in bits 16 to 31 instead, and every use
//! reads the entry's flags it needs from the table (`memory::handles::Lookup::Indexed`).
//!
//! **Heap samplers** (the sampler half). A sampler handle loaded from an argument buffer is
//! searched the same way in the stage's sampler table ([`HandleStrategy::SamplerLookup`],
//! `access.handle.sampler_lookup`), the handle table's layout over the argument buffers' samplers,
//! by the module's one search of it, in its entry form: the handle is the index of the sampler's
//! table entry, 0 (the null entry) for a miss (`memory::handles::Lookup::Entry`). The sampler
//! family reads the entry's element, which indexes the sampler array of each sampling's use, and
//! its state word (`metal::sampler::SamplerWord`) where the sampling reads the sampler's state.
//!
//! Every other use of a handle value here (a store, a compare, `ptrtoint`, a load through a
//! handle) is refused. A static handle parked in a forwarding private cell is its index there
//! (`access::cell`); one loaded from any other thread or threadgroup memory stays
//! `Lowering(Access)`, and a heap handle merged with static candidates, or one loaded from a
//! buffer whose member is not known, is `ArgumentBufferTextures` (`memory::handles::domain`,
//! `member_access`).
//!
//! The read sampler (`get_read_sampler`) is no handle value: it has no handle index, so a merge or
//! copy of it is refused here (`Lowering(Sampler)`). Its strategy yields no value, so it is never
//! carried between blocks, and the texture `read` that uses it, in any block, ignores it.

use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::air::ir::{AddrSpace, CastOp, Inst, InstId, InstKind, Type, Value, ValueId};
use crate::budget::{tree_bytes, Meter};
use crate::error::{Error, Feature, LowerFamily, Unsupported};
use crate::lower::arith::{legal, Form, Held};
use crate::lower::emit::EmitCx;
use crate::lower::select::{DescriptorCounts, Plan, SelectCx};
use crate::memory::handles::{self, Candidate, Domain, DomainKind, NULL_HANDLE};
use crate::memory::leaf::{k32, Leaf, Path};
use crate::memory::provenance::{HandleClass, OffsetRange, Repr};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, SyntheticBinding};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, FuncId, Value as SValue};
use crate::spirv::ir::BinOp;

use super::dynamic::WideAccess;
use super::emit::index_u32;
use super::{is_ptr_ty, operand, refuse, Access, AccessPlan, BufferReach, GepTerms};

/// The binding of `v` when it is a texture or sampler parameter (or a table of them) a function
/// constant removed that the body still reads (`Entry::absent_handles`).
pub fn absent_param(e: &crate::air::Entry, v: Value) -> Option<&crate::air::abi::ParamBinding> {
    let Value::Arg(a) = v else { return None };
    e.absent_handles.iter().find(|b| b.param == a.raw())
}

/// The value of removed texture or sampler parameter `b` ([`absent_param`]): the absent handle
/// (`memory::handles::ABSENT_HANDLE`), or offset 0 into a removed table, whose every element
/// loads as the absent handle.
pub fn absent_param_value(
    cx: &EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: &crate::air::abi::ParamBinding,
) -> BResult<SValue> {
    let v = match crate::memory::provenance::handle_array_of(cx.entry, b) {
        Some(_) => 0,
        None => handles::ABSENT_HANDLE,
    };
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

/// A handle operand: an instruction or parameter value, or a constant handle (null).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HandleOperand {
    Value(ValueId),
    Const(u32),
}

impl HandleOperand {
    /// The `u32` handle index in the open block.
    pub fn emit(self, cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
        match self {
            HandleOperand::Value(v) => cx.operand(fb, v),
            HandleOperand::Const(k) => k32(fb, k),
        }
    }
}

/// The element a handle-array load yields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Element {
    /// A constant handle: one element's, or null for a read wholly outside the array.
    Handle(u32),
    /// The clamped element of a dynamic offset.
    Dynamic {
        /// The array pointer.
        ptr: ValueId,
        /// A handle's size in bytes.
        size: u32,
        /// The handle index of element `lo`, and the element range the load may read.
        handle_lo: u32,
        lo: u32,
        hi: u32,
    },
}

/// The handle strategies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HandleStrategy {
    /// `getelementptr` on a handle-array offset: `base + konst + Σ index · stride` (wrapping).
    ArrayGep {
        base: ValueId,
        konst: u32,
        terms: GepTerms,
    },
    /// A copy of a handle or handle-array offset (`bitcast`, `addrspacecast`, `freeze`).
    Copy { of: HandleOperand },
    /// `select c, t, f` of two handles or two offsets into one handle array.
    Select {
        c: ValueId,
        t: HandleOperand,
        f: HandleOperand,
    },
    /// A load of an element of a handle array.
    Load(Element),
    /// A texture handle loaded from an argument buffer: the 8-byte resource id (two words) the
    /// access reads, searched in the handle table.
    Lookup(Box<IdLoad>),
    /// A sampler handle loaded from an argument buffer: the 8-byte resource id the access reads,
    /// searched in the sampler table.
    SamplerLookup(Box<IdLoad>),
}

/// Where a heap handle's resource id is read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IdLoad {
    /// Through the argument buffer an entry parameter binds: a contained leaf access of its slot.
    Static(Access),
    /// Through a nested argument buffer's device address: a wide leaf access, contained as every
    /// access through a loaded pointer is.
    Wide(WideAccess),
}

impl IdLoad {
    /// What the id's load reaches in the finished plan `slots`: a static access's targets, or a
    /// device address.
    pub fn reach(&self, slots: &AccessPlan) -> BufferReach {
        match self {
            IdLoad::Static(a) => a.reach(slots),
            IdLoad::Wide(_) => BufferReach::ADDRESSES,
        }
    }

    /// An upper estimate of the SPIR-V words of the id's load.
    fn words(&self) -> u32 {
        match self {
            IdLoad::Static(a) => a.words(),
            IdLoad::Wide(a) => a.words(),
        }
    }

    /// The static access, whose slots a volatile load orders (`access::note_ordered`).
    pub(super) fn static_access(&self) -> Option<&Access> {
        match self {
            IdLoad::Static(a) => Some(a),
            IdLoad::Wide(_) => None,
        }
    }

    /// The same load, volatile.
    pub(super) fn volatile(self) -> IdLoad {
        match self {
            IdLoad::Static(a) => IdLoad::Static(Access {
                volatile: true,
                ..a
            }),
            IdLoad::Wide(a) => IdLoad::Wide(WideAccess {
                volatile: true,
                ..a
            }),
        }
    }
}

impl HandleStrategy {
    /// The census name (`access.handle.<op>`) in the finished plan `slots`: a lookup qualified by
    /// the reach of the id's load ([`BufferReach`]).
    pub fn name(&self, slots: &AccessPlan) -> Cow<'static, str> {
        match self {
            HandleStrategy::ArrayGep { .. } => "access.handle.array_gep".into(),
            HandleStrategy::Copy { .. } => "access.handle.copy".into(),
            HandleStrategy::Select { .. } => "access.handle.select".into(),
            HandleStrategy::Load(Element::Handle(_)) => "access.handle.array_load".into(),
            HandleStrategy::Load(Element::Dynamic { .. }) => {
                "access.handle.array_load.clamped".into()
            }
            HandleStrategy::Lookup(a) => a.reach(slots).qualify("access.handle.lookup"),
            HandleStrategy::SamplerLookup(a) => {
                a.reach(slots).qualify("access.handle.sampler_lookup")
            }
        }
    }

    /// An upper estimate of the SPIR-V words one use emits.
    pub fn words(&self) -> u32 {
        match self {
            HandleStrategy::ArrayGep { terms, .. } => {
                12u32.saturating_add(16u32.saturating_mul(terms.len() as u32))
            }
            HandleStrategy::Copy { .. } => 2,
            HandleStrategy::Select { .. } => 6,
            HandleStrategy::Load(Element::Handle(_)) => 4,
            HandleStrategy::Load(Element::Dynamic { .. }) => 40,
            // The id's load, its two words, and the call; the search is the module's routine,
            // counted once ([`routine_words`]).
            HandleStrategy::Lookup(a) | HandleStrategy::SamplerLookup(a) => {
                LOOKUP_CALL_WORDS.saturating_add(a.words())
            }
        }
    }
}

/// Whether this module lowers handles of `class` (textures and samplers; the other classes are
/// refused by their families' routes).
fn lowered_class(class: HandleClass) -> bool {
    matches!(class, HandleClass::Texture | HandleClass::Sampler)
}

/// The class of handle value `v`, if it is a texture or sampler handle.
fn handle_class(cx: &SelectCx<'_>, v: Value) -> Option<HandleClass> {
    match cx.memory.repr(v)? {
        Repr::Handle { class, .. } if lowered_class(class) => Some(class),
        _ => None,
    }
}

/// The class of texture- or sampler-array pointer `v`.
fn array_class(cx: &SelectCx<'_>, v: Value) -> Option<HandleClass> {
    match cx.memory.repr(v)? {
        Repr::HandleArray { class } if lowered_class(class) => Some(class),
        _ => None,
    }
}

/// Whether `v` is a pointer constant that is the null handle (null, `undef`, or `poison`).
fn null_constant(cx: &SelectCx<'_>, v: Value) -> bool {
    use crate::air::ir::Const;
    let Value::Const(c) = v else { return false };
    matches!(
        cx.entry.module.consts().get(c).map(|k| &k.value),
        Some(Const::Null | Const::Undef | Const::Poison)
    )
}

/// The domain of handle value `v` of class `class` (texture or sampler only): whether its
/// static candidates are one, or a heap handle ([`domain_kind`]).
fn handle_domain(cx: &SelectCx<'_>, v: Value, class: HandleClass) -> Result<DomainKind, Error> {
    if !lowered_class(class) {
        return Err(refuse());
    }
    domain_kind(cx, v, class)
}

/// The [`DomainKind`] of `handles::domain` for value `v` and class `class`, without a walk of
/// `v`'s roots: a parameter's or constant's from its one candidate, an instruction's from the table
/// selection read once per handle key before any instruction (`root_sets`).
pub(crate) fn domain_kind(
    cx: &SelectCx<'_>,
    v: Value,
    class: HandleClass,
) -> Result<DomainKind, Error> {
    if let Some(r) = handles::value_candidates(cx.entry, &cx.memory.abi, v, class) {
        return r.map(|c| Domain::Static(c).kind());
    }
    match cx.memory.repr(v) {
        // `handles::domain`'s answer for a handle of another class, dynamic root or not.
        Some(Repr::Handle { class: c, .. }) if c != class => {
            Err(Unsupported::not_implemented(Feature::HandleAsPointer).into())
        }
        Some(Repr::Handle { cands, .. }) => cx
            .root_sets
            .handle(cx.entry, cx.memory, v, class, cands)
            .cloned()
            .unwrap_or_else(|| Err(cx.invariant("a handle's root set was not read"))),
        _ => Err(Unsupported::not_implemented(Feature::Lowering(LowerFamily::Access)).into()),
    }
}

/// The static candidates of handle value `v` of class `class` (texture or sampler only): a walk
/// of its roots, metered (`handles::candidates`).
fn static_candidates(
    cx: &SelectCx<'_>,
    v: Value,
    class: HandleClass,
    m: &mut Meter,
) -> Result<Vec<Candidate>, Error> {
    if !lowered_class(class) {
        return Err(refuse());
    }
    handles::candidates(cx.entry, cx.memory, v, class, m)
}

/// Handle operand `v` of a handle-valued instruction of class `class`.
fn handle_operand(cx: &SelectCx<'_>, v: Value, class: HandleClass) -> Result<HandleOperand, Error> {
    if null_constant(cx, v) {
        return Ok(HandleOperand::Const(NULL_HANDLE));
    }
    if matches!(v, Value::Const(_)) {
        // A constexpr sampler's address.
        if !lowered_class(class) {
            return Err(refuse());
        }
        let cands = handles::value_candidates(cx.entry, &cx.memory.abi, v, class)
            .unwrap_or_else(|| Err(cx.invariant("a constant handle without a candidate")))?;
        return match cands.as_slice() {
            [c] => Ok(HandleOperand::Const(c.handle())),
            _ => Err(refuse()),
        };
    }
    if cx.param(v).is_none() && handle_class(cx, v) != Some(class) {
        return Err(refuse());
    }
    handle_domain(cx, v, class)?;
    Ok(HandleOperand::Value(operand(cx, v)?))
}

/// The lowered representation of a handle (a `u32` handle index) or a handle-array offset (a
/// `u32` byte offset), `None` for any other value.
pub fn lowered_pointer(cx: &SelectCx<'_>, v: Value) -> Result<Option<DemoteTy>, Error> {
    if array_class(cx, v).is_some() {
        return Ok(Some(DemoteTy::ArenaOffset));
    }
    let Some(class) = handle_class(cx, v) else {
        return Ok(None);
    };
    handle_domain(cx, v, class)?;
    // A phi's incoming constant would need a pointer constant's materialization, which the
    // emitter's operands do not provide.
    if super::phis::incoming(cx, v)?.is_some_and(|i| i.constant) {
        return Err(refuse());
    }
    Ok(Some(DemoteTy::HandleIndex))
}

/// The element a load of a value of AIR type `ty` (a handle, or a handle word's integer)
/// through handle-array pointer `ptr` of `class` yields.
fn element(
    cx: &SelectCx<'_>,
    ptr: Value,
    ty: crate::air::ir::TypeId,
    class: HandleClass,
    align: Option<crate::air::ir::Align>,
) -> Result<Element, Error> {
    let Some([crate::memory::provenance::Root::Param(p)]) = cx.memory.roots_of(ptr) else {
        return Err(cx.invariant("a handle-array load with a root outside its array"));
    };
    // Every element of a removed table is the absent handle.
    if let Some(c) = cx.memory.abi.absent(*p) {
        if c != class {
            return Err(Unsupported::not_implemented(Feature::HandleAsPointer).into());
        }
        return Ok(Element::Handle(handles::ABSENT_HANDLE));
    }
    let (pc, first, count) =
        handles::param_slots(cx.entry, &cx.memory.abi, *p).ok_or_else(refuse)?;
    if pc != class {
        return Err(Unsupported::not_implemented(Feature::HandleAsPointer).into());
    }
    let size = cx
        .memory
        .layouts
        .store_size(ty)
        .and_then(|s| u32::try_from(s).ok())
        .filter(|s| *s > 0)
        .ok_or_else(refuse)?;
    let range = cx.memory.offset(ptr).unwrap_or(OffsetRange::Top);
    let Some((lo, hi)) = handles::array_elements(range, u64::from(size), count) else {
        return Ok(Element::Handle(NULL_HANDLE));
    };
    let handle = |k: u32| -> Result<u32, Error> {
        let n = first.checked_add(k).ok_or_else(refuse)?;
        let c = match class {
            HandleClass::Texture => Candidate::Texture(n),
            HandleClass::Sampler => Candidate::Sampler(n),
            _ => return Err(refuse()),
        };
        Ok(c.handle())
    };
    if lo == hi {
        return Ok(Element::Handle(handle(lo)?));
    }
    // The division by the handle size is exact only for an aligned load.
    if align.map_or(1, |a| a.bytes()) < u64::from(size) {
        return Err(refuse());
    }
    Ok(Element::Dynamic {
        ptr: operand(cx, ptr)?,
        size,
        handle_lo: handle(lo)?,
        lo,
        hi,
    })
}

/// What a planned handle word ([`plan_words`]) is the integer of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandleWord {
    /// An element of a handle array of this class.
    Element(HandleClass),
    /// An argument-buffer member's resource id, a heap handle of this class.
    Member(HandleClass),
}

/// Plan the function's handle words before any instruction is selected: each plain load of an
/// integer through a handle-array element pointer whose every use is an `inttoptr` round trip
/// (`memory::provenance::round_trip`) that provenance made the element's handle, and each plain
/// integer load through a buffer pointer whose every use is an `inttoptr` round trip to a heap
/// texture or sampler of one class (the member it reads is the selection's,
/// `memory::handles::member_load`). The integer (the guest's bits of the handle) is never
/// observed, so the load is the element's handle index, or the member's heap lookup, as a
/// pointer load of it is, and each `inttoptr` a copy of it. A load with any other use stays
/// refused. One pass finds the candidates and one reads every use.
pub fn plan_words(cx: &mut SelectCx<'_>, m: &mut Meter) -> Result<(), Error> {
    let body = cx.body;
    let entry = tree_bytes(1, std::mem::size_of::<(InstId, Option<HandleWord>)>());
    // A member word's class is the first `inttoptr`'s; `None` before one is read.
    let mut words: BTreeMap<InstId, Option<HandleWord>> = BTreeMap::new();
    let mut charged = 0usize;
    let plan = || -> Result<BTreeMap<InstId, HandleWord>, Error> {
        for (id, inst) in body.insts.iter() {
            m.tick(1)?;
            let InstKind::Load {
                ptr,
                volatile: false,
                atomic: None,
                ..
            } = &inst.kind
            else {
                continue;
            };
            if is_ptr_ty(cx, inst.ty) {
                continue;
            }
            let word = match (array_class(cx, *ptr), cx.memory.repr(*ptr)) {
                (Some(class), _) => Some(HandleWord::Element(class)),
                (None, Some(Repr::BufPtr { wide: false, .. })) => None,
                _ => continue,
            };
            m.charge(entry)?;
            charged += entry;
            words.insert(id, word);
        }
        if words.is_empty() {
            return Ok(BTreeMap::new());
        }
        for (id, inst) in body.insts.iter() {
            m.tick(1)?;
            // The candidate this instruction round-trips to its handle, if it does.
            let kept = match &inst.kind {
                InstKind::Cast {
                    op: CastOp::IntToPtr,
                    v: Value::Inst(l),
                } => match words.get(l) {
                    Some(word) => {
                        let here = Value::Inst(id);
                        let ptr = crate::memory::provenance::round_trip(
                            cx.entry,
                            &cx.memory.layouts,
                            body,
                            Value::Inst(*l),
                            inst.ty,
                        );
                        let class = handle_class(cx, here);
                        match (ptr, word) {
                            (Some(ptr), Some(HandleWord::Element(c))) => {
                                let same = cx.memory.roots_of(here).is_some()
                                    && cx.memory.roots_of(here) == cx.memory.roots_of(ptr);
                                (same && class == Some(*c)).then_some(*l)
                            }
                            (Some(_), member) => {
                                let heap = handles::heap_handle(cx.memory, here);
                                let agrees = match member {
                                    Some(HandleWord::Member(c)) => heap == Some(*c),
                                    _ => heap.is_some(),
                                };
                                match (agrees, heap) {
                                    (true, Some(c)) => {
                                        words.insert(*l, Some(HandleWord::Member(c)));
                                        Some(*l)
                                    }
                                    _ => None,
                                }
                            }
                            (None, _) => None,
                        }
                    }
                    None => None,
                },
                _ => None,
            };
            inst.kind.for_each_value(|v| {
                if let Value::Inst(l) = v {
                    if kept != Some(l) {
                        words.remove(&l);
                    }
                }
            });
        }
        // A candidate no `inttoptr` read has no class: it is no word.
        Ok(words
            .into_iter()
            .filter_map(|(id, word)| Some((id, word?)))
            .collect())
    };
    let planned = plan();
    m.release(charged);
    cx.families.access.handle_words = planned?;
    Ok(())
}

/// Whether `v` is a planned handle word ([`plan_words`]): held as its handle index.
pub fn is_word(plan: &super::AccessPlan, v: Value) -> bool {
    matches!(v, Value::Inst(l) if plan.handle_words.contains_key(&l))
}

/// The class of the heap handle planned word `v` is the resource id of ([`HandleWord::Member`]).
fn member_word(plan: &super::AccessPlan, v: Value) -> Option<HandleClass> {
    match v {
        Value::Inst(l) => match plan.handle_words.get(&l) {
            Some(HandleWord::Member(class)) => Some(*class),
            _ => None,
        },
        _ => None,
    }
}

/// Select the handle strategy of `inst`, or `None` when it involves no handle or handle array.
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    m: &mut Meter,
) -> Result<Option<HandleStrategy>, Error> {
    let here = Value::Inst(at);
    let result = handle_class(cx, here);
    let array = array_class(cx, here);
    Ok(Some(match &inst.kind {
        InstKind::Gep {
            src_ty,
            base,
            indices,
            ..
        } if array.is_some() => {
            if array_class(cx, *base) != array {
                return Err(refuse());
            }
            let base = operand(cx, *base)?;
            let (konst, terms) = super::gep_offset(cx, *src_ty, indices)?;
            HandleStrategy::ArrayGep { base, konst, terms }
        }
        InstKind::Cast {
            op: CastOp::Bitcast | CastOp::AddrSpaceCast,
            v,
        }
        | InstKind::Freeze(v)
            if is_ptr_ty(cx, inst.ty) && (result.is_some() || array.is_some()) =>
        {
            let of = match (result, array) {
                (Some(class), _) => handle_operand(cx, *v, class)?,
                (None, class) if array_class(cx, *v) == class => {
                    HandleOperand::Value(operand(cx, *v)?)
                }
                _ => return Err(refuse()),
            };
            HandleStrategy::Copy { of }
        }
        InstKind::Select { c, t, f, .. } if result.is_some() || array.is_some() => {
            let c = match c {
                Value::Const(_) => return Err(refuse()),
                c => operand(cx, *c)?,
            };
            let (t, f) = match (result, array) {
                (Some(class), _) => (
                    handle_operand(cx, *t, class)?,
                    handle_operand(cx, *f, class)?,
                ),
                (None, class) if array_class(cx, *t) == class && array_class(cx, *f) == class => (
                    HandleOperand::Value(operand(cx, *t)?),
                    HandleOperand::Value(operand(cx, *f)?),
                ),
                _ => return Err(refuse()),
            };
            HandleStrategy::Select { c, t, f }
        }
        InstKind::Load {
            ptr,
            align,
            volatile: false,
            atomic: None,
            ..
        } if array_class(cx, *ptr).is_some() => {
            let class = array_class(cx, *ptr).ok_or_else(refuse)?;
            // A handle, or a handle word's integer.
            if result != Some(class) && !is_word(&cx.families.access, here) {
                return Err(refuse());
            }
            HandleStrategy::Load(element(cx, *ptr, inst.ty, class, *align)?)
        }
        // An argument-buffer member's resource id loaded as an integer whose every use is an
        // `inttoptr` to its heap handle: the member's lookup.
        InstKind::Load {
            ptr,
            align,
            volatile: false,
            atomic: None,
            ..
        } if member_word(&cx.families.access, here).is_some() => {
            match member_word(&cx.families.access, here) {
                Some(HandleClass::Texture) => {
                    HandleStrategy::Lookup(Box::new(lookup(cx, *ptr, *align, Table::Handles, m)?))
                }
                Some(HandleClass::Sampler) => HandleStrategy::SamplerLookup(Box::new(lookup(
                    cx,
                    *ptr,
                    *align,
                    Table::Samplers,
                    m,
                )?)),
                _ => return Err(refuse()),
            }
        }
        // The `inttoptr` of a handle word is its handle.
        InstKind::Cast {
            op: CastOp::IntToPtr,
            v,
        } if result.is_some() && is_word(&cx.families.access, *v) => HandleStrategy::Copy {
            of: HandleOperand::Value(operand(cx, *v)?),
        },
        InstKind::Load {
            ptr,
            align,
            volatile: false,
            atomic: None,
            ..
        } if result == Some(HandleClass::Texture)
            && handle_domain(cx, here, HandleClass::Texture)? == DomainKind::Heap =>
        {
            HandleStrategy::Lookup(Box::new(lookup(cx, *ptr, *align, Table::Handles, m)?))
        }
        InstKind::Load {
            ptr,
            align,
            volatile: false,
            atomic: None,
            ..
        } if result == Some(HandleClass::Sampler)
            && handle_domain(cx, here, HandleClass::Sampler)? == DomainKind::Heap =>
        {
            HandleStrategy::SamplerLookup(Box::new(lookup(cx, *ptr, *align, Table::Samplers, m)?))
        }
        InstKind::Load { ptr, .. } if result.is_some() => {
            // A handle loaded from memory otherwise: an atomic load of an argument-buffer
            // member, or a handle kept in a private or threadgroup object, a pointer cell. A
            // volatile load is selected as its plain twin (`access::volatile_form`) and never
            // reaches this arm.
            let space = cx
                .value_ty(*ptr)
                .and_then(|t| cx.entry.module.types().get(t))
                .and_then(|t| match t {
                    Type::Ptr { space } => Some(*space),
                    _ => None,
                });
            return Err(match space {
                Some(AddrSpace::Device | AddrSpace::Constant) => {
                    Unsupported::not_implemented(Feature::ArgumentBufferTextures).into()
                }
                _ => refuse(),
            });
        }
        _ => {
            // Any other access-family instruction on a handle has no model.
            let mut touches = result.is_some() || array.is_some();
            inst.kind.for_each_value(|v| {
                touches |= handle_class(cx, v).is_some() || array_class(cx, v).is_some()
            });
            if touches {
                return Err(refuse());
            }
            return Ok(None);
        }
    }))
}

/// Which table a heap handle's resource id is searched in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Table {
    /// The handle table: a texture's.
    Handles,
    /// The sampler table: a sampler's.
    Samplers,
}

/// The resource-id load of a heap texture or sampler handle through buffer pointer `ptr` (the
/// argument buffer's member): two words, contained like any leaf access of the slot. The function
/// then reads the handle table or the sampler table (`table`), one more storage descriptor.
fn lookup(
    cx: &mut SelectCx<'_>,
    ptr: Value,
    align: Option<crate::air::ir::Align>,
    table: Table,
    m: &mut Meter,
) -> Result<IdLoad, Error> {
    let bytes = align.map_or(1, |a| a.bytes());
    match table {
        Table::Handles => {
            handles::member_access(cx.entry, cx.memory, cx.body, ptr, bytes).map(|_| ())?
        }
        Table::Samplers => handles::sampler_member(cx.entry, cx.memory, cx.body, ptr, bytes)?,
    }
    let leaf = Leaf {
        form: Form {
            held: Held::Int {
                bits: 32,
                container: 32,
            },
            lanes: 2,
        },
        bytes: 8,
        path: if align.map_or(1, |a| a.bytes()) >= 4 {
            Path::Aligned
        } else {
            Path::Bytes
        },
    };
    let access = match cx.memory.repr(ptr) {
        // A nested argument buffer's member: its address was loaded from memory.
        Some(Repr::BufPtr { wide: true, .. }) => IdLoad::Wide(WideAccess {
            ptr: super::dynamic::plan_deref(cx, ptr)?,
            value: None,
            leaf,
            coherence: None,
            volatile: false,
        }),
        _ => {
            let (targets, null, in_blob) = super::buffer_roots(cx, ptr, m)?;
            if in_blob {
                return Err(refuse());
            }
            IdLoad::Static(super::access_with_leaf(
                cx, ptr, None, leaf, targets, null, None,
            )?)
        }
    };
    let ids = if legal::holds_wide(cx.profile) {
        HandleIds::Words
    } else {
        HandleIds::U64
    };
    let planned = match table {
        Table::Handles => &mut cx.families.access.handle_table,
        Table::Samplers => &mut cx.families.access.sampler_table,
    };
    if planned.is_none() {
        *planned = Some(ids);
        cx.descriptors = cx.descriptors.plus(&DescriptorCounts {
            storage_buffers: 1,
            ..DescriptorCounts::default()
        });
    }
    Ok(access)
}

/// The words of a heap handle load besides its id's access: the id's two words and the search's
/// call (`OpCompositeExtract` of 5 words each, `OpFunctionCall` of 6).
const LOOKUP_CALL_WORDS: u32 = 24;

/// The words of the handle-table and sampler-table searches a function's heap handle loads call,
/// with the census name of the strategy that calls each: counted once per module, not per load.
pub fn routine_words(plan: &AccessPlan) -> impl Iterator<Item = (&'static str, u64)> {
    let form = plan.lookup();
    let handles = plan.handle_table.map(move |ids| {
        let words = handles::lookup_routine_words(ids == HandleIds::U64, form);
        ("access.handle.lookup", u64::from(words))
    });
    let samplers = plan.sampler_table.map(|ids| {
        let words = handles::lookup_routine_words(ids == HandleIds::U64, handles::Lookup::Entry);
        ("access.handle.sampler_lookup", u64::from(words))
    });
    handles.into_iter().chain(samplers)
}

/// How the handle-table search reads an entry's resource id: as one `u64` on a profile with
/// `shaderInt64` (the table declared as an entry array), else as two words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandleIds {
    U64,
    Words,
}

/// The stage's handle table and the module's search of it.
#[derive(Clone, Copy, Debug)]
pub struct HandleTable {
    /// `fn(lo: u32, hi: u32) -> u32`, the heap handle of a resource id
    /// ([`handles::define_lookup`]): its element, with the entry's flags the plan's search carries;
    /// for the sampler table, the index of its entry.
    pub lookup: FuncId,
    /// The table, whose entries a sampler handle's uses read ([`handles::entry_field`]).
    pub table: handles::Table,
}

/// The binding symbol of the handle table of `stage`.
pub fn table_symbol(stage: crate::pipeline::sym::Stage) -> ResourceSlot {
    ResourceSlot {
        stage,
        kind: ResourceKind::Synthetic(SyntheticBinding::HandleTable),
        index: 0,
    }
}

/// The binding symbol of the sampler table of `stage`.
pub fn sampler_table_symbol(stage: crate::pipeline::sym::Stage) -> ResourceSlot {
    ResourceSlot {
        stage,
        kind: ResourceKind::Synthetic(SyntheticBinding::SamplerTable),
        index: 0,
    }
}

/// Declare the stage's handle table (an entry array, or a word view on a profile without
/// `shaderInt64`) and the module's search of it, when the plan reads it.
pub fn declare_table(mb: &mut ModuleBuilder, plan: &Plan) -> BResult<Option<HandleTable>> {
    let Some(ids) = plan.families.access.handle_table else {
        return Ok(None);
    };
    let table = handles::declare_table(mb, table_symbol(plan.stage), ids == HandleIds::U64)?;
    let lookup = handles::define_lookup(mb, table, plan.families.access.lookup())?;
    Ok(Some(HandleTable { lookup, table }))
}

/// Declare the stage's sampler table and the module's search of it in its entry form, when the
/// plan reads it.
pub fn declare_sampler_table(mb: &mut ModuleBuilder, plan: &Plan) -> BResult<Option<HandleTable>> {
    let Some(ids) = plan.families.access.sampler_table else {
        return Ok(None);
    };
    let sym = sampler_table_symbol(plan.stage);
    let table = handles::declare_table(mb, sym, ids == HandleIds::U64)?;
    let lookup = handles::define_lookup(mb, table, handles::Lookup::Entry)?;
    Ok(Some(HandleTable { lookup, table }))
}

/// Emit handle strategy `s`, defining value `vid`.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    vid: ValueId,
    s: &HandleStrategy,
) -> BResult<()> {
    let r = match s {
        HandleStrategy::ArrayGep { base, konst, terms } => {
            let mut off = cx.operand(fb, *base)?;
            if *konst != 0 {
                let c = k32(fb, *konst)?;
                off = fb.binary(BinOp::IAdd, off, c)?;
            }
            for (v, stride, form) in terms {
                let x = cx.operand(fb, *v)?;
                let mut x = index_u32(fb, x, *form)?;
                if *stride != 1 {
                    let s = k32(fb, *stride)?;
                    x = fb.binary(BinOp::IMul, x, s)?;
                }
                off = fb.binary(BinOp::IAdd, off, x)?;
            }
            off
        }
        HandleStrategy::Copy { of } => of.emit(cx, fb)?,
        HandleStrategy::Select { c, t, f } => {
            let c = cx.operand(fb, *c)?;
            let t = t.emit(cx, fb)?;
            let f = f.emit(cx, fb)?;
            fb.select(c, t, f)?
        }
        HandleStrategy::Load(Element::Handle(h)) => k32(fb, *h)?,
        HandleStrategy::Load(Element::Dynamic {
            ptr,
            size,
            handle_lo,
            lo,
            hi,
        }) => {
            let off = cx.operand(fb, *ptr)?;
            let size = k32(fb, *size)?;
            let k = fb.binary(BinOp::UDiv, off, size)?;
            let (lo, hi) = (k32(fb, *lo)?, k32(fb, *hi)?);
            let below = fb.binary(BinOp::ULessThan, k, lo)?;
            let k = fb.select(below, lo, k)?;
            let above = fb.binary(BinOp::ULessThan, hi, k)?;
            let k = fb.select(above, hi, k)?;
            // Element `k` is handle `handle_lo + (k - lo)`.
            let delta = fb.binary(BinOp::ISub, k, lo)?;
            let base = k32(fb, *handle_lo)?;
            fb.binary(BinOp::IAdd, base, delta)?
        }
        HandleStrategy::Lookup(a) | HandleStrategy::SamplerLookup(a) => {
            let id = match a.as_ref() {
                IdLoad::Static(a) => {
                    let p = cx.operand(fb, a.ptr)?;
                    let m = fb.composite_extract(p, &[1])?;
                    super::emit::access_at(cx, fb, a, p, m, None)?
                }
                IdLoad::Wide(a) => super::dynamic::wide_access(cx, fb, a, None)?,
            };
            let Some(id) = id else {
                return Err(cx.fail("a resource-id load without a value"));
            };
            let table = match s {
                HandleStrategy::SamplerLookup(_) => cx.decl.families.access.sampler_table,
                _ => cx.decl.families.access.handle_table,
            };
            let Some(table) = table else {
                return Err(cx.fail("a handle lookup without its table"));
            };
            let lo = fb.composite_extract(id, &[0])?;
            let hi = fb.composite_extract(id, &[1])?;
            match fb.call(table.lookup, &[lo, hi])? {
                Some(element) => element,
                None => return Err(cx.fail("a handle lookup without a result")),
            }
        }
    };
    cx.define(vid, r);
    Ok(())
}

/// How an operation reaches its texture or sampler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HandleTarget {
    /// One candidate: its binding directly (a null candidate alone is `Switch` with no arm).
    One(Candidate),
    /// Several candidates: a switch-at-use on handle `handle`, one arm per non-null candidate
    /// (sorted), the null handle and every other value taking the null arm.
    Switch {
        handle: ValueId,
        cands: Vec<Candidate>,
    },
    /// The constant null handle (a `null`, `undef`, or `poison` operand; an absent sampler
    /// argument read only beside its texture reads as `null`): the operation is its null arm.
    Null,
}

impl HandleTarget {
    /// Whether the use is not one static binding: a switch-at-use, or the null arm alone.
    pub fn switched(&self) -> bool {
        matches!(self, HandleTarget::Switch { .. } | HandleTarget::Null)
    }
}

/// How a use reaches handle `v` of class `class`. The candidates are walked for the use, metered
/// (`handles::candidates`), and a switch's are charged to the ledger for the rest of the
/// attempt, as the plan holds its strategies until emission ends.
fn target(
    cx: &SelectCx<'_>,
    v: Value,
    class: HandleClass,
    m: &mut Meter,
) -> Result<HandleTarget, Error> {
    if null_constant(cx, v) {
        if !lowered_class(class) {
            return Err(refuse());
        }
        return Ok(HandleTarget::Null);
    }
    let cands = static_candidates(cx, v, class, m)?;
    match cands.as_slice() {
        [c] if *c != Candidate::Null => Ok(HandleTarget::One(*c)),
        _ => {
            if matches!(v, Value::Const(_)) {
                // A constant that is neither null nor one candidate (a constexpr sampler's
                // address takes `One` above).
                return Err(refuse());
            }
            let handle = operand(cx, v)?;
            let cands: Vec<Candidate> = cands
                .into_iter()
                .filter(|c| *c != Candidate::Null)
                .collect();
            m.charge(cands.len().saturating_mul(std::mem::size_of::<Candidate>()))?;
            Ok(HandleTarget::Switch { handle, cands })
        }
    }
}

/// The handle value of texture operand `v` when it is a heap handle: a set-1 heap element, which
/// the texture family indexes its group's heap array with. `None` for a handle with static
/// candidates ([`texture_target`]).
pub fn heap_target(cx: &SelectCx<'_>, v: Value) -> Result<Option<ValueId>, Error> {
    if !matches!(v, Value::Inst(_)) {
        return Ok(None);
    }
    match domain_kind(cx, v, HandleClass::Texture) {
        Ok(DomainKind::Heap) => Ok(Some(operand(cx, v)?)),
        // A static handle, or a refusal `texture_target` reports with the texture family's
        // feature.
        Ok(DomainKind::Static(_)) | Err(_) => Ok(None),
    }
}

/// The handle value of sampler operand `v` when it is a heap handle: an argument buffer's sampler,
/// its element in the set-1 sampler arrays with its pixel-coordinate flag. `None` for a sampler
/// with static candidates ([`sampler_target`]).
pub fn heap_sampler_target(cx: &SelectCx<'_>, v: Value) -> Result<Option<ValueId>, Error> {
    if !matches!(v, Value::Inst(_)) {
        return Ok(None);
    }
    match domain_kind(cx, v, HandleClass::Sampler) {
        Ok(DomainKind::Heap) => Ok(Some(operand(cx, v)?)),
        // A static handle, or a refusal `sampler_target` reports with the sampler family's
        // feature.
        Ok(DomainKind::Static(_)) | Err(_) => Ok(None),
    }
}

/// How a texture operation reaches texture operand `v` (its candidates walked for the use,
/// metered).
pub fn texture_target(cx: &SelectCx<'_>, v: Value, m: &mut Meter) -> Result<HandleTarget, Error> {
    target(cx, v, HandleClass::Texture, m).map_err(|e| match e {
        Error::Unsupported(u) if u.feature == Feature::Lowering(LowerFamily::Access) => {
            Unsupported::not_implemented(Feature::Lowering(LowerFamily::Texture)).into()
        }
        e => e,
    })
}

/// How a sampling operation reaches sampler operand `v` (its candidates walked for the use,
/// metered).
pub fn sampler_target(cx: &SelectCx<'_>, v: Value, m: &mut Meter) -> Result<HandleTarget, Error> {
    target(cx, v, HandleClass::Sampler, m).map_err(|e| match e {
        Error::Unsupported(u) if u.feature == Feature::Lowering(LowerFamily::Access) => {
            Unsupported::not_implemented(Feature::Lowering(LowerFamily::Sampler)).into()
        }
        e => e,
    })
}
