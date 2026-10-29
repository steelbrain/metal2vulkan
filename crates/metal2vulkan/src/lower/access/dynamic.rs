//! Dynamic buffer roots and stored pointers.
//!
//! **Wide pointers.** A buffer pointer whose root set meets the component of the dynamic root
//! (`memory::roots`) is the wide fat pointer `u64vec4(root, at, host, size)`: `root` indexes the
//! root table (0 null or a resident-set miss, `2 + n` for `[[buffer(n)]]`, `R0 + k` for resident
//! entry `k`), `at` is the Metal displacement from the allocation's start (the entry's
//! `bind_offset` plus the displacement from it) in wrapping 64-bit arithmetic (exact, unlike a
//! static pointer's `u32` offset), and `host` and `size` are the entry's, read once where the
//! pointer is formed, so no access reads the root table: a driver that cannot prove a store
//! leaves the table alone would read it again after every store. A parameter is `(2 + n,
//! bind_offset, host, size)` of its entry, read in the function's first block ([`emit_entry`]); a
//! GEP adds its 64-bit datalayout offset to `at` (`access::gep_offset_wide`, reading indices
//! through the index-extension rule, and extending each by its [`IndexForm`]); casts and `freeze`
//! copy the vector and `select` selects it. The root table is read-only to the dispatch: the binder
//! writes it before submission and no lowering stores to it, so a field read where the pointer is
//! formed is the field an access would read.
//!
//! **Where dynamic roots come from.** A load of a device or constant pointer from memory reads
//! its 8-byte guest address (stored pointers are guest addresses): from a buffer through a static
//! or a wide pointer, or from a **pointer cell** in the threadgroup arena or an open one in the
//! private arena (an alloca holding a pointer is an arena object, never typed; a closed private
//! cell forwards its pointers instead, `access::cell`); and `inttoptr` takes one from an `i64`. The
//! resident-set search (`memory::roots::resolve_wide`) turns the address into `(root, i − guest,
//! host, size)` of the entry that holds it, a miss into `(0, i, 0, 0)` (the null root's fields).
//!
//! **Stored and converted pointers**. A buffer pointer stored as a value (into a buffer, a wide
//! pointer's allocation, or a pointer cell) is its guest address, an 8-byte leaf, and `ptrtoint` is
//! the same address truncated to the integer's width: `guest + at` of its root entry, 0 for the
//! null constant. The memory plan puts every buffer root whose pointer escapes
//! this way in the dynamic component (`memory::plan`), so such a pointer is wide, its 64-bit offset
//! exact, and its slot's root entry written (a slot that is only stored has the window `Whole`:
//! pointers recovered from its address may reach any byte). Profile F has no root entries, so its
//! escaping roots stay static and a stored or converted static pointer is `SlotGuestBase` plus its
//! displacement (`access::pointer`). A pointer into the constant blob has no guest address (the
//! blob is no resident allocation) and is refused.
//!
//! **Accesses** through a wide pointer compute from the fields it carries (no switch: every
//! candidate, static or resident, is one code path): the leaf's words at `host + (at & ~3)`, and
//! the sink's address, read once in the function's first block. Containment
//! is always on unless the profile has `Containment::UnsafeOff` (robustness does not cover device
//! addresses): the normative check `size ≥ n ∧ at ≤ size − n` in `u64`, the access at `select(inb,
//! address, sink)`, and a load's value `select(inb, v, 0)`; an out-of-bounds store lands in the
//! write-only sink. Under `GuestIsolation::On` an access through root 0 (a null pointer or a
//! resident-set miss) is counted: word 4 of the `ExecutionStatus` buffer
//! (`StatusWord::ResidentMisses`) is incremented and `StatusBits::RESIDENT_MISS` set, both relaxed
//! `Device` atomics, the buffer reached by `PushField::ExecutionStatusAddress` (a function with
//! device-address slots carries the status by address). Every static slot a wide access may reach
//! becomes `BufferForm::Psb` (`access::finish_slots`), so the binder writes its root entry.
//!
//! **Compares.** `icmp` of wide pointers, or of a wide pointer and the null constant, compares
//! their Metal addresses: `guest + at` of each pointer's root entry, the null root's fields being
//! 0 (so a miss `(0, i, 0, 0)` is address `i`, and null is 0). A compare of a wide
//! pointer with a static one is `access::pointer`'s.
//!
//! **Null constants** (`null`, `undef`, `poison`) are the null root: `(0, 0, 0, 0)` as a `select` side
//! or a phi's incoming value.
//!
//! **Coherent accesses** (`air.{load,store}.{device,system}_coherent`) through a wide pointer are
//! the wide leaf access with `NonPrivatePointer` and `MakePointerVisible` (a load) or
//! `MakePointerAvailable` (a store) at `Device` scope, selecting the Vulkan memory model, as
//! through a static-root pointer.
//!
//! **Atomics** through a wide pointer are `access::atomic`'s, at the device address
//! `atomic_address` computes from the root entry, contained and counted like a leaf access.
//!
//! **Memory intrinsics** (`access::copy`) resolve a wide endpoint once (`wide_base`: the root
//! entry's fields, and the miss counted when the length is nonzero) and access each unit at its
//! displacement from it (`wide_at`), contained like a leaf access. A `memmove` that may alias
//! compares the endpoints' Metal addresses: a wide pointer's `guest_address`, and a static-root
//! pointer's `static_guest_address` (`SlotGuestBase` plus its displacement).
//!
//! A volatile access (a leaf, or a pointer's guest address) is the same access whose words carry
//! `Volatile` (`access::volatile_form`).
//!
//! Not lowered yet (`Unsupported{Lowering(Access), NotImplemented}`): pointers stored in a typed
//! object or a constant global, a wide root set holding the constant blob, and aggregate
//! accesses. Profile F refuses every wide pointer (`DereferenceLoadedPointer`, or
//! `PointerToInteger` in a function without a loaded pointer, naming what it lacks of
//! `bufferDeviceAddress` and `shaderInt64`).

use std::borrow::Cow;
use std::collections::BTreeSet;

use crate::air::abi::{AbiInt, ParamBinding, ParamRole};
use crate::air::intrinsics::{AccessKind, AirIntrinsic, Coherent as CoherentAccess};
use crate::air::ir::{
    AddrSpace, Align, Call, Callee, CastOp, Inst, InstId, InstKind, IntPredicate, Type, Value,
    ValueId,
};
use crate::bind::status::{StatusBits, StatusWord};
use crate::budget::Meter;
use crate::device::profile::{BufferAccess, Containment, GuestIsolation};
use crate::device::requirements::VkFeature;
use crate::error::{Capability, Error, Feature, LowerFamily, Unsupported};
use crate::lower::arith::{Form, Held};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::contain;
use crate::memory::leaf::{self, k32, Coherence, Leaf, Path, Units, WordOps};
use crate::memory::provenance::{Repr, RootSetId};
use crate::memory::roots::{self, k64};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{PushField, Stage};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, TypeId, Value as SValue};
use crate::spirv::ir::{AtomicOp, BinOp, ConvOp, Scope, Semantics, Type as SType};

use super::local::{self, Home, LocalAccess};
use super::psb::PsbMem;
use super::{
    is_ptr_ty, not_lowered, operand, refuse, slot_root, static_roots, Access, AccessPlan,
    BufferReach, IndexForm, SideReach, Target, WideGepTerms,
};

/// The access family's dynamic-root plan.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DynamicPlan {
    /// Memory-plan slots a wide access may reach: each is reached through the root table
    /// (`BufferForm::Psb`).
    pub slots: BTreeSet<usize>,
    /// Buffer parameters whose value is a wide pointer.
    pub params: BTreeSet<u32>,
    /// The function reads the root table for a wide pointer (a search or a wide access).
    pub used: bool,
    /// The resident-set searches of the function (a loaded pointer or `inttoptr` each): the
    /// module defines the search once when there are several ([`roots::define_resolve`]).
    pub resolves: u32,
    /// Accesses through root 0 are counted in the `ExecutionStatus` buffer
    /// (`GuestIsolation::On`).
    pub misses: bool,
}

/// A leaf access through a wide pointer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WideAccess {
    pub ptr: ValueId,
    /// The stored value (stores only).
    pub value: Option<ValueId>,
    pub leaf: Leaf,
    /// The coherence of a `*_coherent` access; `None` for the function's own.
    pub coherence: Option<Coherence>,
    /// A volatile access: every word it loads or stores carries `Volatile`.
    pub volatile: bool,
}

/// Where a loaded pointer's guest address is read from, or a stored one's is written to (its
/// 8-byte leaf).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// Through a static-root pointer (a descriptor view or a static device-address slot).
    Static(Access),
    /// Through a wide pointer.
    Wide(WideAccess),
    /// A pointer cell in the private or threadgroup arena.
    Arena(LocalAccess),
}

impl Source {
    pub(super) fn words(&self) -> u32 {
        match self {
            Source::Static(a) => a.words(),
            Source::Wide(a) => a.leaf.cost().saturating_add(ACCESS_WORDS),
            Source::Arena(a) => a.leaf.cost().saturating_add(96),
        }
    }

    /// What the access reaches in the finished plan `slots`: a static-root pointer's targets, a
    /// wide pointer's device address, or an arena cell's arena.
    pub fn reach(&self, slots: &AccessPlan) -> SideReach {
        match self {
            Source::Static(a) => SideReach::Buffer(a.reach(slots)),
            Source::Wide(_) => SideReach::Buffer(BufferReach::ADDRESSES),
            Source::Arena(_) => SideReach::Arena,
        }
    }

    /// The same access, volatile.
    fn volatile(self) -> Source {
        match self {
            Source::Static(a) => Source::Static(Access {
                volatile: true,
                ..a
            }),
            Source::Wide(a) => Source::Wide(WideAccess {
                volatile: true,
                ..a
            }),
            Source::Arena(a) => Source::Arena(LocalAccess {
                volatile: true,
                ..a
            }),
        }
    }
}

/// The dynamic-root strategies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DynStrategy {
    /// `getelementptr` on a wide pointer: `offset + konst + Σ index · stride` (wrapping `u64`).
    Gep {
        base: ValueId,
        konst: u64,
        terms: WideGepTerms,
    },
    /// `bitcast`, `addrspacecast`, or `freeze`: the operand itself.
    Copy { of: ValueId },
    /// `select c, t, f` (a side may be the null constant).
    Select { c: ValueId, t: Side, f: Side },
    /// A pointer loaded from memory: its guest address, then the resident-set search.
    LoadPointer(Source),
    /// A buffer pointer stored as a value: its guest address, an 8-byte leaf.
    StorePointer { to: Source, value: Side },
    /// `ptrtoint` of a buffer pointer to an `i<bits>` (at most 64): its guest address, truncated
    /// to the width.
    ToInt { of: Side, bits: u8 },
    /// `inttoptr` of an `i64`: the resident-set search.
    FromInt { of: ValueId },
    /// A leaf load through a wide pointer.
    Load(WideAccess),
    /// A leaf store through a wide pointer.
    Store(WideAccess),
    /// `icmp` of wide pointers (or the null constant): their Metal addresses compared.
    Compare {
        pred: IntPredicate,
        a: Side,
        b: Side,
    },
}

/// An operand of a wide pointer compare, `select`, store, or conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Ptr(ValueId),
    /// The null pointer constant (Metal address 0; the wide pointer `(0, 0, 0, 0)`).
    Null,
}

/// SPIR-V words of the module's resident-set search ([`roots::resolve`], once per module): per
/// step the halving, the entry's access chain and load, a compare, and a select, about 40 words,
/// plus the table's and the set's conversions, the hit test, and a function's frame; and the
/// loop that ends a set past the unrolled steps: one more step, its two variables' stores and
/// loads, and the loop's and its exit's blocks.
pub const SEARCH_WORDS: u32 = 44 * (roots::SEARCH_STEPS + 1) + 216 + 96;
/// SPIR-V words of one use of the search: the root table's address and the call.
const CALL_WORDS: u32 = 24;
/// SPIR-V words of one wide access before its leaf: the pointer's fields, the check, the sink
/// selects, and the miss counter.
const ACCESS_WORDS: u32 = 320;
/// SPIR-V words of one wide pointer's Metal address: an entry read and an addition.
const ADDRESS_WORDS: u32 = 72;
/// SPIR-V words of one [`WideBase`]: the pointer's fields, the sink's address, and the miss
/// counter.
pub const BASE_WORDS: u32 = 256;

impl WideAccess {
    /// An upper estimate of the SPIR-V words of the access.
    pub(super) fn words(&self) -> u32 {
        self.leaf.cost().saturating_add(ACCESS_WORDS)
    }
}

impl DynStrategy {
    /// The volatile form of a load or store (`access::volatile_form`): the word loads and stores of
    /// its leaf (a value's, or a pointer's guest address) carry `Volatile`; the refusal of any
    /// other strategy.
    pub(super) fn volatile(self) -> Result<DynStrategy, Error> {
        Ok(match self {
            DynStrategy::Load(a) => DynStrategy::Load(WideAccess {
                volatile: true,
                ..a
            }),
            DynStrategy::Store(a) => DynStrategy::Store(WideAccess {
                volatile: true,
                ..a
            }),
            DynStrategy::LoadPointer(s) => DynStrategy::LoadPointer(s.volatile()),
            DynStrategy::StorePointer { to, value } => DynStrategy::StorePointer {
                to: to.volatile(),
                value,
            },
            _ => return Err(refuse()),
        })
    }

    /// The census name (`access.dynamic.<op>`) in the finished plan `slots`: a pointer loaded
    /// from or stored through a static-root pointer is qualified by the reach of its targets
    /// ([`BufferReach`]); through a wide pointer or an arena cell it is named apart.
    pub fn name(&self, slots: &AccessPlan) -> Cow<'static, str> {
        match self {
            DynStrategy::Gep { .. } => "access.dynamic.gep".into(),
            DynStrategy::Copy { .. } => "access.dynamic.pointer_copy".into(),
            DynStrategy::Select { .. } => "access.dynamic.pointer_select".into(),
            DynStrategy::LoadPointer(s) => match s {
                Source::Static(a) => a.reach(slots).qualify("access.dynamic.load_pointer"),
                Source::Wide(_) => "access.dynamic.load_pointer.wide".into(),
                Source::Arena(_) => "access.dynamic.load_pointer.arena".into(),
            },
            DynStrategy::StorePointer { to, .. } => match to {
                Source::Static(a) => a.reach(slots).qualify("access.dynamic.store_pointer"),
                Source::Wide(_) => "access.dynamic.store_pointer.wide".into(),
                Source::Arena(_) => "access.dynamic.store_pointer.arena".into(),
            },
            DynStrategy::ToInt { .. } => "access.dynamic.ptrtoint".into(),
            DynStrategy::FromInt { .. } => "access.dynamic.inttoptr".into(),
            DynStrategy::Load(_) => "access.dynamic.load".into(),
            DynStrategy::Store(_) => "access.dynamic.store".into(),
            DynStrategy::Compare { .. } => "access.dynamic.compare".into(),
        }
    }

    /// An upper estimate of the SPIR-V words one use emits.
    pub fn words(&self) -> u32 {
        match self {
            DynStrategy::Gep { terms, .. } => {
                16u32.saturating_add(20u32.saturating_mul(terms.len() as u32))
            }
            DynStrategy::Copy { .. } => 2,
            DynStrategy::Select { .. } => 6,
            DynStrategy::LoadPointer(s) => s.words().saturating_add(CALL_WORDS),
            DynStrategy::StorePointer { to, .. } => to.words().saturating_add(ADDRESS_WORDS),
            DynStrategy::ToInt { .. } => ADDRESS_WORDS,
            DynStrategy::FromInt { .. } => CALL_WORDS,
            DynStrategy::Load(a) | DynStrategy::Store(a) => {
                a.leaf.cost().saturating_add(ACCESS_WORDS)
            }
            DynStrategy::Compare { .. } => 2 * ADDRESS_WORDS + 8,
        }
    }
}

/// The root set of `v` when it is a wide pointer.
/// Whether `v` is a wide pointer.
pub(super) fn is_wide(cx: &SelectCx<'_>, v: Value) -> bool {
    wide(cx, v).is_some()
}

fn wide(cx: &SelectCx<'_>, v: Value) -> Option<RootSetId> {
    match cx.memory.repr(v)? {
        Repr::BufPtr { roots, wide: true } => Some(roots),
        _ => None,
    }
}

/// Profile F has no device addresses: every wide pointer is refused there, naming what the
/// profile lacks of Profile A: as a loaded pointer's dereference when the function has a dynamic
/// root, else as a stored or converted pointer (whose root joined the dynamic component).
fn require_addresses(cx: &SelectCx<'_>) -> Result<(), Error> {
    if cx.profile.fields().buffer_access != BufferAccess::Addressed {
        let feature = if cx.memory.needs.dynamic_roots {
            Feature::DereferenceLoadedPointer
        } else {
            Feature::PointerToInteger
        };
        return Err(Unsupported::needs_any_of(feature, vec![cx.profile.addressed_needs()]).into());
    }
    Ok(())
}

/// The wide operand `v`: refused unless it is a wide pointer value (a constant pointer is not
/// materialized).
fn wide_operand(cx: &SelectCx<'_>, v: Value) -> Result<ValueId, Error> {
    if wide(cx, v).is_none() {
        return Err(refuse());
    }
    operand(cx, v)
}

/// The function reads the root table for a wide pointer.
fn use_table(cx: &mut SelectCx<'_>) {
    cx.families.access.dynamic.used = true;
    cx.push.insert(PushField::RootTableAddress);
}

/// Plan a read of wide pointer `v`'s root entry: its static slots go through the root table, so
/// the binder writes their entries. The slots are its root set's (`root_sets::SetFacts`), each
/// with a window; a root that is neither null, dynamic, nor an accessed buffer slot (the constant
/// blob, an arena, a slot the function never accesses and therefore has no root entry for) is
/// refused. A set's slots join the plan's once, at its first such read.
pub(super) fn plan_entry(cx: &mut SelectCx<'_>, v: Value) -> Result<ValueId, Error> {
    require_addresses(cx)?;
    let roots = wide(cx, v).ok_or_else(refuse)?;
    let Some(set) = cx.root_sets.get_mut(roots) else {
        return Err(cx.invariant("a wide pointer's root set was not read"));
    };
    let slots = set.wide_slots.as_deref().ok_or_else(refuse)?;
    if !set.merged {
        cx.families
            .access
            .dynamic
            .slots
            .extend(slots.iter().copied());
        set.merged = true;
    }
    use_table(cx);
    operand(cx, v)
}

/// Plan a dereference of wide pointer `v` ([`plan_entry`]); under isolation an access through
/// root 0 is counted, which needs the stage's store feature for the counter's atomics.
pub(super) fn plan_deref(cx: &mut SelectCx<'_>, v: Value) -> Result<ValueId, Error> {
    plan_entry(cx, v)?;
    plan_misses(cx)?;
    operand(cx, v)
}

/// Under isolation, plan the counting of an access through root 0, which needs the stage's store
/// feature for the counter's atomics.
fn plan_misses(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    if cx.profile.fields().isolation == GuestIsolation::On {
        let enabled = &cx.profile.fields().enabled.features;
        if let Some(f) = stage_stores(cx.stage).find(|f| !enabled.contains(f)) {
            return Err(Unsupported::needs(
                Feature::DereferenceLoadedPointer,
                Capability::Feature(f),
            )
            .into());
        }
        cx.families.access.dynamic.misses = true;
        cx.push.insert(PushField::ExecutionStatusAddress);
        cx.facts.insert(RuntimeFact::ExecutionStatus);
    }
    Ok(())
}

/// The feature the `ExecutionStatus` counter's atomics, and any other store of the product's
/// (a vertex function's capture records), need in `stage` besides what any module needs: stage
/// stores and atomics outside compute.
pub(crate) fn stage_stores(stage: Stage) -> impl Iterator<Item = VkFeature> {
    match stage {
        Stage::Vertex => Some(VkFeature::VertexPipelineStoresAndAtomics),
        Stage::Fragment => Some(VkFeature::FragmentStoresAndAtomics),
        // Vulkan's stage store features name no task or mesh stage: their stores need none.
        Stage::Kernel | Stage::Object | Stage::Mesh => None,
    }
    .into_iter()
}

/// The lowered representation of a wide pointer value `v` (`None` for any other value). A phi's
/// incoming null constant is the null wide pointer (`pointer::phi_incoming`); a phi whose incoming
/// value is another constant or not wide is refused.
pub fn lowered_pointer(cx: &SelectCx<'_>, v: Value) -> Result<Option<DemoteTy>, Error> {
    if wide(cx, v).is_none() {
        return Ok(None);
    }
    require_addresses(cx)?;
    if super::phis::incoming(cx, v)?.is_some_and(|i| i.not_wide) {
        return Err(refuse());
    }
    Ok(Some(DemoteTy::FatPtr { wide: true }))
}

/// The SPIR-V type of a wide pointer: `u64vec4(root, at, host, size)`.
pub fn wide_type(mb: &mut crate::spirv::builder::ModuleBuilder) -> BResult<TypeId> {
    let u = mb.ty(SType::Int {
        width: 64,
        signed: false,
    })?;
    mb.ty(SType::Vector { elem: u, len: 4 })
}

/// Whether buffer parameter `b`'s value is a wide pointer.
pub fn is_wide_param(cx: &SelectCx<'_>, b: &ParamBinding) -> bool {
    wide(cx, Value::Arg(crate::air::ir::ArgIdx::new(b.param))).is_some()
}

/// Plan a buffer parameter whose value is a wide pointer.
pub fn select_param(cx: &mut SelectCx<'_>, b: &ParamBinding) -> Result<(), Error> {
    require_addresses(cx)?;
    cx.families.access.dynamic.params.insert(b.param);
    Ok(())
}

/// The value of a wide buffer parameter, `[2 + n, bind_offset, host, size]` of its root entry
/// (read where the value is first needed: [`emit_entry`] holds it from the function's first
/// block), or `None` when `b` is not one.
pub fn param_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: &ParamBinding,
) -> BResult<Option<SValue>> {
    if !cx.plan.families.access.dynamic.params.contains(&b.param) {
        return Ok(None);
    }
    let index = match &b.role {
        ParamRole::Buffer(arg) | ParamRole::ArgumentBuffer(arg) => match arg.location.index {
            AbiInt::Const(i) => i,
            _ => return Err(cx.fail(format!("parameter {} is no buffer slot", b.param))),
        },
        _ => return Err(cx.fail(format!("parameter {} is no buffer", b.param))),
    };
    let root = k64(fb, u64::from(slot_root(index)))?;
    let table = cx.push_field(fb, PushField::RootTableAddress)?;
    let at = roots::entry_field(fb, table, root, roots::BIND_OFFSET)?;
    let host = roots::entry_field(fb, table, root, roots::HOST)?;
    let size = roots::entry_field(fb, table, root, roots::SIZE)?;
    Ok(Some(fat(fb, [root, at, host, size])?))
}

/// The work before the body (the entry hook): in the function's first block, which dominates
/// every block, read the sink's address and each wide buffer parameter's root entry once, so no
/// access reads the root table again. The table is read-only to the dispatch: the binder writes
/// it before submission and no lowering stores to it.
pub fn emit_entry(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
    let dynamic = &cx.plan.families.access.dynamic;
    if !dynamic.used {
        return Ok(());
    }
    let params: Vec<u32> = dynamic.params.iter().copied().collect();
    if cx.plan.families.access.psb_contained {
        cx.sink_host(fb)?;
    }
    for p in params {
        let Some(v) = cx
            .plan
            .values
            .id(crate::air::ir::Value::Arg(crate::air::ir::ArgIdx::new(p)))
        else {
            continue;
        };
        let x = cx.operand(fb, v)?;
        cx.hold(v, x);
    }
    Ok(())
}

/// The sink's address (`host` of the sink's root entry), read in the open block.
pub(crate) fn read_sink_host(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
    let table = cx.push_field(fb, PushField::RootTableAddress)?;
    let sink = k64(fb, roots::SINK_ROOT)?;
    roots::entry_field(fb, table, sink, roots::HOST)
}

/// The 8-byte leaf of a stored pointer (its guest address, a `u64`).
fn pointer_leaf(
    cx: &SelectCx<'_>,
    ty: crate::air::ir::TypeId,
    align: Option<Align>,
) -> Result<Leaf, Error> {
    let bytes = cx.memory.layouts.store_size(ty).ok_or_else(refuse)?;
    if bytes != 8 {
        return Err(refuse());
    }
    let align = align.map_or(1, |a| a.bytes());
    Ok(Leaf {
        form: Form {
            held: Held::Int {
                bits: 64,
                container: 64,
            },
            lanes: 1,
        },
        bytes: 8,
        path: if align >= 4 {
            Path::Aligned
        } else {
            Path::Bytes
        },
    })
}

/// A static-root load (or, `write`, store) of `leaf` through `ptr` (the access family's
/// `Access`).
fn static_access(
    cx: &mut SelectCx<'_>,
    ptr: Value,
    leaf: Leaf,
    write: bool,
    m: &mut Meter,
) -> Result<Access, Error> {
    let (mut targets, null) = static_roots(cx, ptr, m)?;
    let check = contain::encoding(cx.profile);
    for t in &mut targets {
        *t = Target { check, ..*t };
        let plan = cx
            .memory
            .slots
            .get(t.slot)
            .ok_or_else(|| cx.invariant("a target names a missing slot"))?;
        if write && super::view_kind(cx.profile, plan) != super::ViewKind::Storage {
            return Err(cx.invariant("a pointer store through a uniform view"));
        }
        cx.push.insert(PushField::SlotBindOffset(plan.slot));
    }
    Ok(Access {
        ptr: operand(cx, ptr)?,
        value: None,
        leaf,
        targets,
        null,
        coherence: None,
        volatile: false,
        blob: None,
    })
}

/// A stored value operand: an instruction or parameter value, or a constant the arith family
/// materializes.
fn stored_value(cx: &SelectCx<'_>, v: Value) -> Result<ValueId, Error> {
    if matches!(v, Value::Const(_)) {
        if cx.constant(v).is_none() {
            return Err(not_lowered(LowerFamily::Arith));
        }
        return cx
            .values
            .id(v)
            .ok_or_else(|| cx.invariant("a stored value out of range"));
    }
    operand(cx, v)
}

/// Select the dynamic-root strategy of `inst`, or `None` when it involves no wide pointer (or
/// is an operation another family selects, such as an atomic).
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    m: &mut Meter,
) -> Result<Option<DynStrategy>, Error> {
    let here = Value::Inst(at);
    let result_wide = wide(cx, here).is_some();
    if let InstKind::Call(call) = &inst.kind {
        if let Callee::Air {
            op: AirIntrinsic::Coherent(c),
            ..
        } = &call.callee
        {
            return coherent(cx, inst, call, c);
        }
    }
    Ok(Some(match &inst.kind {
        InstKind::Gep {
            src_ty,
            base,
            indices,
            ..
        } if result_wide => {
            require_addresses(cx)?;
            let base = wide_operand(cx, *base)?;
            let (konst, terms) = super::gep_offset_wide(cx, *src_ty, indices)?;
            DynStrategy::Gep { base, konst, terms }
        }
        InstKind::Cast {
            op: CastOp::Bitcast | CastOp::AddrSpaceCast,
            v,
        }
        | InstKind::Freeze(v)
            if result_wide && is_ptr_ty(cx, inst.ty) =>
        {
            require_addresses(cx)?;
            DynStrategy::Copy {
                of: wide_operand(cx, *v)?,
            }
        }
        InstKind::Select { c, t, f, .. } if result_wide && is_ptr_ty(cx, inst.ty) => {
            require_addresses(cx)?;
            let c = match c {
                Value::Const(_) => return Err(refuse()),
                c => operand(cx, *c)?,
            };
            let side = |v: Value| -> Result<Side, Error> {
                if super::pointer::null_constant(cx, v) {
                    return Ok(Side::Null);
                }
                wide_operand(cx, v).map(Side::Ptr)
            };
            DynStrategy::Select {
                c,
                t: side(*t)?,
                f: side(*f)?,
            }
        }
        InstKind::Cast {
            op: CastOp::IntToPtr,
            v,
        } if result_wide => {
            require_addresses(cx)?;
            if !matches!(
                cx.value_ty(*v).and_then(|t| cx.entry.module.types().get(t)),
                Some(Type::Int(64))
            ) {
                return Err(refuse());
            }
            use_table(cx);
            cx.families.access.dynamic.resolves += 1;
            DynStrategy::FromInt {
                of: operand(cx, *v)?,
            }
        }
        InstKind::Load {
            ptr,
            align,
            volatile: false,
            atomic: None,
            ..
        } if result_wide => {
            require_addresses(cx)?;
            let leaf = pointer_leaf(cx, inst.ty, *align)?;
            use_table(cx);
            cx.families.access.dynamic.resolves += 1;
            let source = match cx.memory.repr(*ptr) {
                Some(Repr::BufPtr { wide: false, .. }) => {
                    Source::Static(static_access(cx, *ptr, leaf, false, m)?)
                }
                Some(Repr::BufPtr { wide: true, .. }) => Source::Wide(WideAccess {
                    ptr: plan_deref(cx, *ptr)?,
                    value: None,
                    leaf,
                    coherence: None,
                    volatile: false,
                }),
                // A pointer cell in threadgroup or private memory.
                _ => Source::Arena(cell(cx, *ptr, leaf, false)?),
            };
            DynStrategy::LoadPointer(source)
        }
        InstKind::Store {
            val,
            ptr,
            align,
            volatile: false,
            atomic: None,
        } if is_buffer_ptr(cx, *val) => {
            let value = stored(cx, *val)?;
            let ty = cx
                .value_ty(*val)
                .ok_or_else(|| cx.invariant("a stored pointer without a type"))?;
            let leaf = pointer_leaf(cx, ty, *align)?;
            let to = match cx.memory.repr(*ptr) {
                Some(Repr::BufPtr { wide: false, .. }) => {
                    Source::Static(static_access(cx, *ptr, leaf, true, m)?)
                }
                Some(Repr::BufPtr { wide: true, .. }) => Source::Wide(WideAccess {
                    ptr: plan_deref(cx, *ptr)?,
                    value: None,
                    leaf,
                    coherence: None,
                    volatile: false,
                }),
                _ => Source::Arena(cell(cx, *ptr, leaf, true)?),
            };
            DynStrategy::StorePointer { to, value }
        }
        InstKind::Cast {
            op: CastOp::PtrToInt,
            v,
        } if is_buffer_ptr(cx, *v) => {
            let bits = match cx.entry.module.types().get(inst.ty) {
                Some(Type::Int(b @ 1..=64)) => *b,
                _ => return Err(refuse()),
            };
            DynStrategy::ToInt {
                of: stored(cx, *v)?,
                bits,
            }
        }
        InstKind::Load {
            ptr,
            align,
            volatile: false,
            atomic: None,
            ..
        } if wide(cx, *ptr).is_some() && !is_ptr_ty(cx, inst.ty) => {
            let leaf = leaf::leaf_of(
                cx.entry,
                cx.profile,
                &cx.memory.layouts,
                inst.ty,
                align_of(*align),
            )?;
            DynStrategy::Load(WideAccess {
                ptr: plan_deref(cx, *ptr)?,
                value: None,
                leaf,
                coherence: None,
                volatile: false,
            })
        }
        InstKind::Store {
            val,
            ptr,
            align,
            volatile: false,
            atomic: None,
        } if wide(cx, *ptr).is_some() => {
            let ty = cx
                .value_ty(*val)
                .ok_or_else(|| cx.invariant("a stored value without a type"))?;
            if is_ptr_ty(cx, ty) {
                return Err(refuse());
            }
            let leaf = leaf::leaf_of(
                cx.entry,
                cx.profile,
                &cx.memory.layouts,
                ty,
                align_of(*align),
            )?;
            let value = stored_value(cx, *val)?;
            DynStrategy::Store(WideAccess {
                ptr: plan_deref(cx, *ptr)?,
                value: Some(value),
                leaf,
                coherence: None,
                volatile: false,
            })
        }
        InstKind::ICmp { pred, a, b } if wide(cx, *a).is_some() || wide(cx, *b).is_some() => {
            let mut side = |v: Value| -> Result<Side, Error> {
                if is_null(cx, v) {
                    return Ok(Side::Null);
                }
                plan_entry(cx, v).map(Side::Ptr)
            };
            DynStrategy::Compare {
                pred: *pred,
                a: side(*a)?,
                b: side(*b)?,
            }
        }
        _ => return Ok(None),
    }))
}

/// A `air.{load,store}.{device,system}_coherent` access through a wide pointer: the wide leaf
/// access with the coherent memory operands at `Device` scope, which selects the Vulkan memory
/// model, as for a static-root pointer (`select_coherent`); `None` for another pointer.
fn coherent(
    cx: &mut SelectCx<'_>,
    inst: &Inst,
    call: &Call,
    c: &CoherentAccess,
) -> Result<Option<DynStrategy>, Error> {
    let arg = |i: usize| -> Result<Value, Error> {
        call.args
            .get(i)
            .copied()
            .ok_or_else(|| cx.invariant("a coherent access without its operands"))
    };
    let (ptr, value) = match c.kind {
        AccessKind::Load => (arg(0)?, None),
        AccessKind::Store => (arg(1)?, Some(arg(0)?)),
    };
    if wide(cx, ptr).is_none() {
        return Ok(None);
    }
    if !matches!(c.space, AddrSpace::Device | AddrSpace::Constant) {
        return Err(refuse());
    }
    let ty = match value {
        None => inst.ty,
        Some(v) => cx
            .value_ty(v)
            .ok_or_else(|| cx.invariant("a coherent store's value without a type"))?,
    };
    // The intrinsic states no alignment: the value's natural (ABI) one.
    let align = cx.memory.layouts.abi_align(ty).unwrap_or(1);
    let leaf = leaf::leaf_of(cx.entry, cx.profile, &cx.memory.layouts, ty, align)?;
    let value = match value {
        Some(v) => Some(stored_value(cx, v)?),
        None => None,
    };
    let a = WideAccess {
        ptr: plan_deref(cx, ptr)?,
        value,
        leaf,
        coherence: Some(Coherence::Device),
        volatile: c.volatile,
    };
    crate::lower::sync::memory_model::require_vulkan(cx);
    Ok(Some(match c.kind {
        AccessKind::Load => DynStrategy::Load(a),
        AccessKind::Store => DynStrategy::Store(a),
    }))
}

/// A memory intrinsic's endpoint `v` when it is a wide pointer: its allocation, reached through the
/// root entry as a leaf access's (the access's leaf is replaced per unit); `None` for another
/// pointer.
pub(super) fn memory_endpoint(
    cx: &mut SelectCx<'_>,
    v: Value,
) -> Result<Option<WideAccess>, Error> {
    if wide(cx, v).is_none() {
        return Ok(None);
    }
    Ok(Some(WideAccess {
        ptr: plan_deref(cx, v)?,
        value: None,
        leaf: leaf::Leaf {
            form: Form {
                held: Held::Int {
                    bits: 8,
                    container: 32,
                },
                lanes: 1,
            },
            bytes: 1,
            path: Path::OneWord,
        },
        coherence: None,
        volatile: false,
    }))
}

/// Whether `v` is a device or constant pointer (a buffer pointer, whatever its roots).
fn is_buffer_ptr(cx: &SelectCx<'_>, v: Value) -> bool {
    matches!(
        cx.value_ty(v).and_then(|t| cx.entry.module.types().get(t)),
        Some(Type::Ptr {
            space: AddrSpace::Device | AddrSpace::Constant
        })
    )
}

/// A buffer pointer stored or converted to an integer: the null constant, or a wide pointer (every
/// root whose pointer escapes is in the dynamic component, so its guest address is exact from its
/// root entry). A pointer into the constant blob has no guest address, and any other operand is
/// refused.
fn stored(cx: &mut SelectCx<'_>, v: Value) -> Result<Side, Error> {
    require_addresses(cx)?;
    if is_null(cx, v) {
        return Ok(Side::Null);
    }
    plan_entry(cx, v).map(Side::Ptr)
}

/// The 8-byte access of pointer cell `ptr` (a pointer into an arena), a store of one where `write`:
/// its leaf holds a guest address.
fn cell(cx: &mut SelectCx<'_>, ptr: Value, leaf: Leaf, write: bool) -> Result<LocalAccess, Error> {
    let Some(home @ Home::Arena(arena)) = local::home_of(cx, ptr) else {
        return Err(refuse());
    };
    local::use_arena(cx, arena)?;
    if write {
        local::writable(cx, arena)?;
    }
    Ok(LocalAccess {
        arena,
        ptr: local::local_ptr(cx, ptr, home)?,
        value: None,
        leaf,
        contained: cx.profile.fields().containment != Containment::UnsafeOff,
        volatile: false,
    })
}

/// Whether `v` is a null pointer constant (`null`, `undef`, or `poison`).
fn is_null(cx: &SelectCx<'_>, v: Value) -> bool {
    super::pointer::null_constant(cx, v)
}

fn align_of(a: Option<Align>) -> u64 {
    a.map_or(1, |a| a.bytes())
}

// ---- Emission ----

fn u64_ty(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(SType::Int {
        width: 64,
        signed: false,
    })
}

/// The index operand `x` (held in `form`) as its 64-bit value.
fn index_u64(fb: &mut FunctionBuilder<'_>, x: SValue, form: IndexForm) -> BResult<SValue> {
    let u64_ = u64_ty(fb)?;
    match form {
        IndexForm::Wide => Ok(x),
        IndexForm::WordZero => fb.convert(ConvOp::UConvert, x, u64_),
        IndexForm::Word => fb.convert(ConvOp::SConvert, x, u64_),
        IndexForm::Narrow { bits } => {
            let s = k32(fb, 32 - u32::from(bits))?;
            let up = fb.binary(BinOp::ShiftLeftLogical, x, s)?;
            let v = fb.binary(BinOp::ShiftRightArithmetic, up, s)?;
            fb.convert(ConvOp::SConvert, v, u64_)
        }
        IndexForm::Bool => {
            let ones = k64(fb, u64::MAX)?;
            let zero = k64(fb, 0)?;
            fb.select(x, ones, zero)
        }
        IndexForm::BoolZero => {
            let one = k64(fb, 1)?;
            let zero = k64(fb, 0)?;
            fb.select(x, one, zero)
        }
    }
}

/// The wide pointer `[root, at, host, size]`.
fn fat(fb: &mut FunctionBuilder<'_>, parts: [SValue; 4]) -> BResult<SValue> {
    let ty = wide_type(fb.module())?;
    fb.composite_construct(ty, &parts)
}

/// The fields of wide pointer value `p`: `[root, at, host, size]`.
fn fields(fb: &mut FunctionBuilder<'_>, p: SValue) -> BResult<[SValue; 4]> {
    Ok([
        fb.composite_extract(p, &[0])?,
        fb.composite_extract(p, &[1])?,
        fb.composite_extract(p, &[2])?,
        fb.composite_extract(p, &[3])?,
    ])
}

/// Count an access through root 0 when `live` holds (always without it): `ResidentMisses += 1`,
/// `RESIDENT_MISS` set.
fn count_miss(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    root: SValue,
    live: Option<SValue>,
) -> BResult<()> {
    let status = cx.push_field(fb, PushField::ExecutionStatusAddress)?;
    let null = k64(fb, roots::NULL_ROOT)?;
    let miss = fb.binary(BinOp::IEqual, root, null)?;
    let miss = match live {
        Some(l) => fb.binary(BinOp::LogicalAnd, miss, l)?,
        None => miss,
    };
    let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let u32_ = fb.module().ty(SType::Int {
            width: 32,
            signed: false,
        })?;
        let word = |fb: &mut FunctionBuilder<'_>, w: StatusWord| -> BResult<SValue> {
            let off = k64(fb, u64::from(w.byte_offset()))?;
            let a = fb.binary(BinOp::IAdd, status, off)?;
            fb.device_place(a, u32_)
        };
        let count = word(fb, StatusWord::ResidentMisses)?;
        let one = k32(fb, 1)?;
        fb.atomic(
            AtomicOp::IAdd,
            count,
            Scope::Device,
            Semantics::RELAXED,
            one,
        )?;
        let bits = word(fb, StatusWord::Status)?;
        let bit = k32(fb, StatusBits::RESIDENT_MISS.0)?;
        fb.atomic(AtomicOp::Or, bits, Scope::Device, Semantics::RELAXED, bit)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), miss, then, None)?;
    Ok(())
}

/// The access `a` through its wide pointer: the loaded value, or the stored value's store.
pub(super) fn wide_access(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    a: &WideAccess,
    stored: Option<SValue>,
) -> BResult<Option<SValue>> {
    let p = cx.operand(fb, a.ptr)?;
    let b = wide_base(cx, fb, p, a, None)?;
    wide_at(fb, b, None, &a.leaf, stored)
}

/// A wide pointer's allocation in the open block: the root entry's fields the pointer carries and
/// the containment bounds, for the leaf accesses at displacements from it (a memory intrinsic's
/// units).
#[derive(Clone, Copy, Debug)]
pub struct WideBase {
    /// The entry's `host` (a `u64`).
    host: SValue,
    /// The pointer's `at` (a `u64`).
    at: SValue,
    /// The entry's `size` and the sink's address, under containment.
    bounds: Option<(SValue, SValue)>,
    ops: WordOps,
    /// The narrow units the profile stores and loads directly (`psb::narrow_units`).
    units: Units,
}

/// Resolve wide pointer value `p` (a `u64vec4`) for the accesses of `a`: with its coherence (a
/// `*_coherent` access's, or the function's own) and volatility. Under isolation an access
/// through root 0 is counted here, once, when `live` holds (always without it).
pub(super) fn wide_base(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    p: SValue,
    a: &WideAccess,
    live: Option<SValue>,
) -> BResult<WideBase> {
    base_with(cx, fb, p, (a.coherence, a.volatile), live)
}

/// [`wide_base`] with the coherence (`None`: the function's own) and volatility `ops`.
fn base_with(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    p: SValue,
    ops: (Option<Coherence>, bool),
    live: Option<SValue>,
) -> BResult<WideBase> {
    let [root, at, host, size] = fields(fb, p)?;
    if cx.plan.families.access.dynamic.misses {
        count_miss(cx, fb, root, live)?;
    }
    let bounds = if cx.plan.families.access.psb_contained {
        Some((size, cx.sink_host(fb)?))
    } else {
        None
    };
    Ok(WideBase {
        host,
        at,
        bounds,
        units: cx.plan.families.access.psb_units,
        ops: WordOps {
            coherence: ops
                .0
                .unwrap_or_else(|| crate::lower::sync::memory_model::coherence(cx.plan)),
            volatile: ops.1,
        },
    })
}

/// Load (or store `stored`) `leaf` at `d` bytes (a `u64`; `None` for 0) past `b`: the words at
/// `host + (a & ~3)` for `a = at + d`, contained: the normative check in `u64`, the access at
/// `select(inb, address, sink)`, and a load's value `select(inb, v, 0)`; an out-of-bounds store
/// lands in the write-only sink.
pub(super) fn wide_at(
    fb: &mut FunctionBuilder<'_>,
    b: WideBase,
    d: Option<SValue>,
    leaf: &Leaf,
    stored: Option<SValue>,
) -> BResult<Option<SValue>> {
    let at = match d {
        Some(d) => fb.binary(BinOp::IAdd, b.at, d)?,
        None => b.at,
    };
    let three = k64(fb, 3)?;
    let low = fb.binary(BinOp::BitwiseAnd, at, three)?;
    let aligned = fb.binary(BinOp::ISub, at, low)?;
    let base = fb.binary(BinOp::IAdd, b.host, aligned)?;
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let low = fb.convert(ConvOp::UConvert, low, u32_)?;
    let (base, low, inb) = match b.bounds {
        Some((size, sink)) => {
            let inb = contain::in_bounds(fb, at, size, u64::from(leaf.bytes))?;
            let base = fb.select(inb, base, sink)?;
            let zero = k32(fb, 0)?;
            let low = fb.select(inb, low, zero)?;
            (base, low, Some(inb))
        }
        None => (base, low, None),
    };
    let mut mem = PsbMem::at(base).with(b.ops).with_units(b.units);
    match stored {
        None => {
            let v = leaf::load(fb, &mut mem, leaf, low)?;
            Ok(Some(match inb {
                Some(inb) => {
                    let z = leaf::zero(fb, leaf)?;
                    fb.select(inb, v, z)?
                }
                None => v,
            }))
        }
        Some(v) => {
            // An out-of-bounds store writes the sink, which no shader reads observably.
            leaf::store(fb, &mut mem, leaf, low, v)?;
            Ok(None)
        }
    }
}

/// Where an atomic through a wide pointer operates.
#[derive(Clone, Copy, Debug)]
pub(super) struct AtomicAddress {
    /// The device address (a `u64`) of the atomic's word: its object, or a narrow atomic's
    /// containing `u32` word.
    pub(super) addr: SValue,
    /// A narrow atomic's byte offset (a `u32`, 0 to 3) within its word.
    pub(super) byte: Option<SValue>,
    /// The containment condition (`None` without containment).
    pub(super) inb: Option<SValue>,
}

/// Where an atomic of `n` bytes (1, 2, 4, or 8) through wide pointer `ptr` operates: `host +
/// (at & ~(w - 1))` of the pointer's fields for the word size `w =
/// max(n, 4)`, naturally aligned as Metal atomic objects are, and out of bounds the sink's
/// address (the in-bounds test is on the `n` bytes, whose containing word the padded size also
/// covers). An access through root 0 is counted first under isolation, as a leaf access's is.
pub(super) fn atomic_address(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    ptr: ValueId,
    n: u32,
) -> BResult<AtomicAddress> {
    let p = cx.operand(fb, ptr)?;
    let [root, at, host, size] = fields(fb, p)?;
    if cx.plan.families.access.dynamic.misses {
        count_miss(cx, fb, root, None)?;
    }
    let word = u64::from(n.max(4));
    let mask = k64(fb, !(word - 1))?;
    let aligned = fb.binary(BinOp::BitwiseAnd, at, mask)?;
    let addr = fb.binary(BinOp::IAdd, host, aligned)?;
    let byte = if n < 4 {
        let three = k64(fb, 3)?;
        let low = fb.binary(BinOp::BitwiseAnd, at, three)?;
        let u32_ = fb.module().ty(SType::Int {
            width: 32,
            signed: false,
        })?;
        Some(fb.convert(ConvOp::UConvert, low, u32_)?)
    } else {
        None
    };
    if !cx.plan.families.access.psb_contained {
        return Ok(AtomicAddress {
            addr,
            byte,
            inb: None,
        });
    }
    let inb = contain::in_bounds(fb, at, size, u64::from(n))?;
    let sink = cx.sink_host(fb)?;
    let addr = fb.select(inb, addr, sink)?;
    Ok(AtomicAddress {
        addr,
        byte,
        inb: Some(inb),
    })
}

/// The `u64` `a` truncated to `i<bits>`, held as its value family holds it: an `i1` as a `bool`
/// (its low bit), a wider integer zero-extended in its container (a `u32` up to 32 bits, else a
/// `u64`).
fn truncated(fb: &mut FunctionBuilder<'_>, a: SValue, bits: u8) -> BResult<SValue> {
    match bits {
        1 => {
            let one = k64(fb, 1)?;
            let low = fb.binary(BinOp::BitwiseAnd, a, one)?;
            let zero = k64(fb, 0)?;
            fb.binary(BinOp::INotEqual, low, zero)
        }
        64 => Ok(a),
        33..=63 => {
            let m = k64(fb, (1u64 << bits) - 1)?;
            fb.binary(BinOp::BitwiseAnd, a, m)
        }
        _ => {
            let u32_ = fb.module().ty(SType::Int {
                width: 32,
                signed: false,
            })?;
            let w = fb.convert(ConvOp::UConvert, a, u32_)?;
            if bits == 32 {
                return Ok(w);
            }
            let m = k32(fb, (1u32 << bits) - 1)?;
            fb.binary(BinOp::BitwiseAnd, w, m)
        }
    }
}

/// The wide pointer of `side`: the operand, or `(0, 0, 0, 0)` for the null constant.
fn wide_side(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, side: Side) -> BResult<SValue> {
    match side {
        Side::Ptr(p) => cx.operand(fb, p),
        Side::Null => {
            let zero = k64(fb, 0)?;
            fat(fb, [zero; 4])
        }
    }
}

/// The Metal address of wide pointer `side` ([`guest_address`]; 0 for the null constant).
fn metal_address(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, side: Side) -> BResult<SValue> {
    let p = match side {
        Side::Null => return k64(fb, 0),
        Side::Ptr(p) => cx.operand(fb, p)?,
    };
    guest_address(cx, fb, p)
}

/// The Metal (guest) address of wide pointer value `p` (a `u64vec4`): `guest + at` of its root
/// entry (the null root's fields are 0, so a miss `[0, i, 0, 0]` is `i`).
pub(super) fn guest_address(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    p: SValue,
) -> BResult<SValue> {
    let root = fb.composite_extract(p, &[0])?;
    let at = fb.composite_extract(p, &[1])?;
    let table = cx.push_field(fb, PushField::RootTableAddress)?;
    let guest = roots::entry_field(fb, table, root, roots::GUEST)?;
    fb.binary(BinOp::IAdd, guest, at)
}

/// The Metal (guest) address of static-root fat pointer `fat` whose candidates are `targets`: for
/// the candidate its root names, `SlotGuestBase(slot)` (the bound buffer's `gpu_address + offset`)
/// plus the 64-bit displacement its `u32` offset stands for in the slot's window
/// (`psb::displacement`); 0 through the null root. Exact for every offset a Metal-defined access of
/// the window reaches.
pub(super) fn static_guest_address(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    targets: &[Target],
    fat: SValue,
) -> BResult<SValue> {
    let root = fb.composite_extract(fat, &[0])?;
    let m = fb.composite_extract(fat, &[1])?;
    let mut acc = k64(fb, 0)?;
    for t in targets {
        let Some(choice) = cx.plan.slots.get(t.slot).copied().flatten() else {
            return Err(cx.fail("a guest address names an unplanned slot"));
        };
        let start = match choice.window {
            crate::pipeline::interface::Window::Bounded { start, .. } => start,
            crate::pipeline::interface::Window::Whole => 0,
        };
        let base = cx.push_field(fb, PushField::SlotGuestBase(choice.slot))?;
        let base = super::pointer::slot_guest_base_u64(fb, base)?;
        let disp = super::psb::displacement(fb, m, start)?;
        let guest = fb.binary(BinOp::IAdd, base, disp)?;
        let r = k32(fb, t.root)?;
        let here = fb.binary(BinOp::IEqual, root, r)?;
        acc = fb.select(here, guest, acc)?;
    }
    Ok(acc)
}

/// Plan the Metal addresses of the static-root candidates `targets` ([`static_guest_address`]):
/// their `SlotGuestBase` push fields, summed with displacements in `u64` arithmetic that needs
/// `shaderInt64` (named in the refusal of `feature` otherwise).
pub(super) fn plan_static_guest(
    cx: &mut SelectCx<'_>,
    targets: &[Target],
    feature: Feature,
) -> Result<(), Error> {
    if !cx.profile.fields().leaf.int64 {
        return Err(Unsupported::needs(
            feature,
            Capability::Feature(crate::device::requirements::VkFeature::ShaderInt64),
        )
        .into());
    }
    for t in targets {
        let slot = cx
            .memory
            .slots
            .get(t.slot)
            .ok_or_else(|| cx.invariant("a guest address names a missing slot"))?
            .slot;
        cx.push.insert(PushField::SlotGuestBase(slot));
    }
    Ok(())
}

/// The wide pointer of guest address `guest` (a `u64`): a call of the module's resident-set
/// search ([`roots::define_resolve`]), or the search in place when it is the module's only one.
fn resolve(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, guest: SValue) -> BResult<SValue> {
    let table = cx.push_field(fb, PushField::RootTableAddress)?;
    match cx.decl.families.access.resolve {
        Some(search) => fb
            .call(search, &[table, guest])?
            .ok_or_else(|| cx.fail("a resident-set search without a result")),
        None => {
            let r = roots::resolve_wide(fb, table, guest)?;
            fat(fb, r)
        }
    }
}

/// Whether the module defines the resident-set search as a function: a function with more than
/// one search calls it, one with a single search emits it in place (a function's frame and call
/// would only add words).
pub fn defines_search(plan: &DynamicPlan) -> bool {
    plan.resolves > 1
}

/// The module's routines the dynamic strategies use, charged once to the census: the resident-set
/// search, `access.dynamic.resolve`, whether called or in place.
pub fn routine_words(plan: &DynamicPlan) -> impl Iterator<Item = (&'static str, u64)> {
    (plan.resolves > 0)
        .then_some(("access.dynamic.resolve", u64::from(SEARCH_WORDS)))
        .into_iter()
}

/// Emit dynamic strategy `s`, defining value `vid`.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    vid: ValueId,
    s: &DynStrategy,
) -> BResult<()> {
    let r = match s {
        DynStrategy::Gep { base, konst, terms } => {
            let p = cx.operand(fb, *base)?;
            let [root, mut off, host, size] = fields(fb, p)?;
            if *konst != 0 {
                let c = k64(fb, *konst)?;
                off = fb.binary(BinOp::IAdd, off, c)?;
            }
            for (v, stride, form) in terms {
                let x = cx.operand(fb, *v)?;
                let mut x = index_u64(fb, x, *form)?;
                if *stride != 1 {
                    let s = k64(fb, *stride)?;
                    x = fb.binary(BinOp::IMul, x, s)?;
                }
                off = fb.binary(BinOp::IAdd, off, x)?;
            }
            fat(fb, [root, off, host, size])?
        }
        DynStrategy::Copy { of } => cx.operand(fb, *of)?,
        DynStrategy::Select { c, t, f } => {
            let c = cx.operand(fb, *c)?;
            let t = wide_side(cx, fb, *t)?;
            let f = wide_side(cx, fb, *f)?;
            fb.select(c, t, f)?
        }
        DynStrategy::FromInt { of } => {
            let x = cx.operand(fb, *of)?;
            resolve(cx, fb, x)?
        }
        DynStrategy::LoadPointer(Source::Static(a)) => {
            let p = cx.operand(fb, a.ptr)?;
            let m = fb.composite_extract(p, &[1])?;
            let Some(guest) = super::emit::access_at(cx, fb, a, p, m, None)? else {
                return Err(cx.fail("a pointer load without a value"));
            };
            resolve(cx, fb, guest)?
        }
        DynStrategy::LoadPointer(Source::Wide(a)) => {
            let Some(guest) = wide_access(cx, fb, a, None)? else {
                return Err(cx.fail("a pointer load without a value"));
            };
            resolve(cx, fb, guest)?
        }
        DynStrategy::LoadPointer(Source::Arena(a)) => {
            let Some(guest) = local::emit_access(cx, fb, a, None)? else {
                return Err(cx.fail("a pointer load without a value"));
            };
            resolve(cx, fb, guest)?
        }
        DynStrategy::StorePointer { to, value } => {
            let guest = metal_address(cx, fb, *value)?;
            match to {
                Source::Static(a) => {
                    let p = cx.operand(fb, a.ptr)?;
                    let m = fb.composite_extract(p, &[1])?;
                    super::emit::access_at(cx, fb, a, p, m, Some(guest))?;
                }
                Source::Wide(a) => {
                    wide_access(cx, fb, a, Some(guest))?;
                }
                Source::Arena(a) => {
                    local::emit_access(cx, fb, a, Some(guest))?;
                }
            }
            return Ok(());
        }
        DynStrategy::ToInt { of, bits } => {
            let a = metal_address(cx, fb, *of)?;
            truncated(fb, a, *bits)?
        }
        DynStrategy::Load(a) => match wide_access(cx, fb, a, None)? {
            Some(v) => v,
            None => return Err(cx.fail("a load without a value")),
        },
        DynStrategy::Store(a) => {
            let Some(val) = a.value else {
                return Err(cx.fail("a store without a value"));
            };
            let v = cx.operand(fb, val)?;
            wide_access(cx, fb, a, Some(v))?;
            return Ok(());
        }
        DynStrategy::Compare { pred, a, b } => {
            let a = metal_address(cx, fb, *a)?;
            let b = metal_address(cx, fb, *b)?;
            let op = match pred {
                IntPredicate::Eq => BinOp::IEqual,
                IntPredicate::Ne => BinOp::INotEqual,
                IntPredicate::Ugt => BinOp::UGreaterThan,
                IntPredicate::Uge => BinOp::UGreaterThanEqual,
                IntPredicate::Ult => BinOp::ULessThan,
                IntPredicate::Ule => BinOp::ULessThanEqual,
                IntPredicate::Sgt => BinOp::SGreaterThan,
                IntPredicate::Sge => BinOp::SGreaterThanEqual,
                IntPredicate::Slt => BinOp::SLessThan,
                IntPredicate::Sle => BinOp::SLessThanEqual,
            };
            fb.binary(op, a, b)?
        }
    };
    cx.define(vid, r);
    Ok(())
}

/// Plan a load of the 8-byte word at buffer pointer `ptr` (of AIR type `ty`, aligned `align`) as
/// data, a `u64`: through a static root (a descriptor view or a static device-address slot), or
/// a wide pointer's root entry. An indirect-command object's handle load reads its field this
/// way (`access::encode`).
pub(super) fn plan_word(
    cx: &mut SelectCx<'_>,
    ptr: Value,
    ty: crate::air::ir::TypeId,
    align: Option<Align>,
    m: &mut Meter,
) -> Result<Source, Error> {
    let leaf = pointer_leaf(cx, ty, align)?;
    Ok(match cx.memory.repr(ptr) {
        Some(Repr::BufPtr { wide: false, .. }) => {
            Source::Static(static_access(cx, ptr, leaf, false, m)?)
        }
        Some(Repr::BufPtr { wide: true, .. }) => Source::Wide(WideAccess {
            ptr: plan_deref(cx, ptr)?,
            value: None,
            leaf,
            coherence: None,
            volatile: false,
        }),
        _ => return Err(refuse()),
    })
}

/// Plan a resident-set search outside a pointer value (an encoder's record buffer): the function
/// reads the root table, searches it once more, and under isolation counts an access through
/// root 0. The caller checked the profile's device addresses and the stage's store feature.
pub(super) fn plan_search(cx: &mut SelectCx<'_>) -> Result<(), Error> {
    require_addresses(cx)?;
    use_table(cx);
    cx.families.access.dynamic.resolves += 1;
    plan_misses(cx)
}

/// Plan the guest address of buffer pointer `v` written as data: the null constant, or a wide
/// pointer (every root whose pointer escapes is wide on a profile with device addresses).
pub(super) fn plan_address(cx: &mut SelectCx<'_>, v: Value) -> Result<Side, Error> {
    stored(cx, v)
}

/// The 8-byte word `s` reads ([`plan_word`]), a `u64`.
pub(super) fn emit_word(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    s: &Source,
) -> BResult<SValue> {
    let word = match s {
        Source::Static(a) => {
            let p = cx.operand(fb, a.ptr)?;
            let m = fb.composite_extract(p, &[1])?;
            super::emit::access_at(cx, fb, a, p, m, None)?
        }
        Source::Wide(a) => wide_access(cx, fb, a, None)?,
        Source::Arena(a) => local::emit_access(cx, fb, a, None)?,
    };
    word.ok_or_else(|| cx.fail("an object word without a value"))
}

/// The wide pointer of the resident allocation that starts at guest address `guest` (a `u64`):
/// the resident-set search ([`resolve`]), its hit kept only at the allocation's start, any other
/// address a miss, `(0, guest)`. Indirect command buffers' records are searched so: records longer
/// than the distance between two IDs' starts run past the next IDs' starts
/// (`indirect_commands::records_address`), and an ID whose own records are not resident then
/// misses rather than reaching into another's.
pub(super) fn search_start(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    guest: SValue,
) -> BResult<SValue> {
    let p = resolve(cx, fb, guest)?;
    let [root, off, host, size] = fields(fb, p)?;
    let zero = k64(fb, 0)?;
    let at_start = fb.binary(BinOp::IEqual, off, zero)?;
    let root = fb.select(at_start, root, zero)?;
    let off = fb.select(at_start, off, guest)?;
    let host = fb.select(at_start, host, zero)?;
    let size = fb.select(at_start, size, zero)?;
    fat(fb, [root, off, host, size])
}

/// The allocation of wide pointer `p` resolved for plain accesses in the function's coherence,
/// an access through root 0 counted under isolation ([`wide_base`]).
pub(super) fn plain_base(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    p: SValue,
) -> BResult<WideBase> {
    base_with(cx, fb, p, (None, false), None)
}

/// A `PhysicalStorageBuffer` pointer to the `u32` at `d` bytes (a `u64`, 4-aligned past a
/// 4-aligned base) past `b`, contained: out of bounds, the sink's first word.
pub(super) fn word_place(fb: &mut FunctionBuilder<'_>, b: WideBase, d: SValue) -> BResult<SValue> {
    let at = fb.binary(BinOp::IAdd, b.at, d)?;
    let mut addr = fb.binary(BinOp::IAdd, b.host, at)?;
    if let Some((size, sink)) = b.bounds {
        let inb = contain::in_bounds(fb, at, size, 4)?;
        addr = fb.select(inb, addr, sink)?;
    }
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    fb.device_place(addr, u32_)
}

/// The Metal address of `side` ([`plan_address`]): a wide pointer's guest address, or 0.
pub(super) fn address_of(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    side: Side,
) -> BResult<SValue> {
    metal_address(cx, fb, side)
}
