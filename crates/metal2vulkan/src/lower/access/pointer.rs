//! Pointer values.
//!
//! **Null constants.** `null`, `undef`, and `poison` pointers are the null root (provenance's
//! `Root::Null`, a valid refinement of the undefined ones). As a buffer pointer's operand they
//! are the null fat pointer of the value's width, `uvec2(0, 0)` or `u64vec2(0, 0)`
//! ([`null_value`]): the side of a `select` and a phi's incoming value, which the phi's copy on
//! its incoming edge materializes ([`phi_incoming`]), a GEP's base, and an access's pointer
//! ([`null_operand`]). Canonicalization rewrites every read of a gated-off buffer argument to
//! such a constant, so a null base is common. A dereference through the null root, which Metal
//! leaves undefined, keeps the access family's rules: a pointer that can only be null loads 0 and
//! drops its stores, one whose other candidates are slots is contained in them (one slot:
//! dereferenced at the null pointer's offset; several: the switch-at-use's default arm), and a wide
//! one is out of bounds.
//!
//! **Compares** of device and constant pointers ([`PointerValue::CompareOffsets`],
//! [`PointerValue::CompareAddresses`]) compare Metal addresses:
//!
//! - Two pointers into one object whose address never enters the result compare their offsets:
//!   two static fat pointers whose root sets are the same single slot (no null), two pointers
//!   into the constant blob, or two into one typed constant global (`memory::alloca`). Equality
//!   compares the `u32` offsets; an unsigned order compares them relative to the start of the
//!   2³² bytes both displacements lie in (the proven offset range's start when both ranges fit
//!   one such span, else the slot's span below). The blob and a typed constant have no
//!   Metal address the function can read, so a signed order on them, and any compare between
//!   them and another object, stays refused.
//! - Every other compare of static fat pointers, wide pointers, and null compares addresses. A
//!   static fat pointer's address is its candidate slot's `SlotGuestBase` (the bound buffer's
//!   `gpu_address + offset`; 0 when the slot is unbound, which is what Metal passes for a `nil`
//!   buffer) plus the 64-bit displacement its `u32` offset stands for, and through the null root
//!   that displacement alone. The displacement is the unique value congruent to the offset in the
//!   2³² bytes starting at the pointer's proven offset range when that range fits them (exact by
//!   the analysis), else in the slot's span: the binding's first 2³² bytes when the slot's window
//!   lies in them, which hold every pointer an access reaches and every pointer MSL lets the
//!   function form from the argument (C++ pointer arithmetic stays within `[b, b + size]`, and a
//!   descriptor slot's range is at most `maxStorageBufferRange`), else the 2³² bytes from the
//!   window's start (every pointer an access reaches). A slot whose address only a compare observes
//!   is a `Whole` window. A wide pointer's address is its root entry's `guest + bind_offset +
//!   offset`. With `shaderInt64` the address is a `u64`; without it, two `u32` words compared high
//!   word first, and `SlotGuestBase` is placed as a `uvec2` on every profile (the same eight
//!   bytes).
//!
//! **Stored and converted pointers on Profile F** ([`PointerValue::ToInt`],
//! [`PointerValue::StorePointer`]). Without device addresses an escaping slot root stays static
//! (`memory::plan`), so `ptrtoint` of a static fat pointer (or of the null constant) is its
//! Metal address as above, truncated to the integer's width (a `bool` its low bit), and a stored
//! one is that address as an 8-byte leaf through a static-root destination (stored pointers are
//! guest addresses). On Profile A such a pointer is wide.
//!
//! **Handles of the other families.** A command buffer, pipeline state, acceleration structure, or
//! function table loaded from an argument buffer, or any access-family instruction on one, is
//! refused as its family is when it is a parameter (`lower::select::param_family`):
//! `ArgumentBufferObjects`, `RayTracing`, or `DynamicCall`, all `NotImplemented`.

use std::borrow::Cow;

use crate::air::ir::{
    Align, CastOp, Const, Inst, InstId, InstKind, IntPredicate, Type, TypeId, Value, ValueId,
};
use crate::budget::Meter;
use crate::device::profile::BufferAccess;
use crate::error::{Error, Feature, Unsupported};
use crate::lower::arith::wide::{self, Pair};
use crate::lower::arith::{Form, Held};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::leaf::{k32, Leaf, Path};
use crate::memory::provenance::{HandleClass, OffsetRange, Repr};
use crate::memory::roots::k64;
use crate::pipeline::interface::Window;
use crate::pipeline::sym::{PushField, SlotRef};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, Value as SValue};
use crate::spirv::ir::{BinOp, ConstBits, Constant, ConvOp, Type as SType};

use super::emit::fat_part;
use super::local::{self, Home, LocalPtr};
use super::{blob, buffer_roots, dynamic, operand, psb, refuse, statics, Access, AccessPlan};

/// Bytes a `u32` offset tells apart.
const SPAN: i128 = 1 << 32;

/// Where a compare operand's Metal address comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Addr {
    /// The null pointer constant (`null`, `undef`, or `poison`): address 0.
    Null,
    /// A static-root fat pointer.
    Static(StaticAddr),
    /// A wide pointer: `guest + bind_offset + offset` of its root entry.
    Wide(ValueId),
}

/// A static-root fat pointer's address: per candidate, `SlotGuestBase` plus its displacement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaticAddr {
    pub ptr: ValueId,
    /// The candidate slots: (memory-plan slot, root number).
    pub targets: Vec<(usize, u32)>,
    /// The pointer may be null (root 0), whose address is its displacement alone.
    pub null: bool,
    /// The start of the 2³² bytes the displacement lies in when the proven offset range fixes
    /// one; else each candidate slot's span (`window_start`; 0 through the null root).
    pub start: Option<i64>,
}

/// An offset operand of a same-object compare.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Offset {
    /// A static fat pointer (its offset part).
    Fat(ValueId),
    /// A blob or typed-constant offset.
    Local(LocalPtr),
}

/// The pointer-value strategies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PointerValue {
    /// `icmp` of two pointers into one object, by their offsets (`start` for an unsigned
    /// order; `slot` names the slot whose span applies when `start` is `None`).
    CompareOffsets {
        pred: IntPredicate,
        a: Offset,
        b: Offset,
        start: Option<i64>,
        slot: Option<usize>,
    },
    /// `icmp` of Metal addresses, in two words without `shaderInt64` (`words`).
    CompareAddresses {
        pred: IntPredicate,
        a: Addr,
        b: Addr,
        words: bool,
    },
    /// `ptrtoint` of a static fat pointer or the null constant to `i<bits>` (Profile F).
    ToInt { of: Addr, bits: u8, words: bool },
    /// A static fat pointer or the null constant stored as its 8-byte Metal address through a
    /// static-root destination (Profile F).
    StorePointer {
        to: Box<Access>,
        value: Addr,
        words: bool,
    },
}

/// SPIR-V words of one candidate's address: the guest base, the displacement, the sum, and the
/// root select (twice that in two words).
const CANDIDATE_WORDS: u32 = 48;

impl Addr {
    fn words(&self, two: bool) -> u32 {
        let per = if two { 2 } else { 1 };
        match self {
            Addr::Null => 4,
            Addr::Static(s) => (s.targets.len() as u32 + u32::from(s.null))
                .saturating_add(1)
                .saturating_mul(CANDIDATE_WORDS * per),
            Addr::Wide(_) => 80,
        }
    }
}

impl PointerValue {
    /// The volatile form of a pointer store (`access::volatile_form`): the words of the stored
    /// Metal address carry `Volatile`; the refusal of any other strategy.
    pub(super) fn volatile(self) -> Result<PointerValue, Error> {
        match self {
            PointerValue::StorePointer { to, value, words } => Ok(PointerValue::StorePointer {
                to: Box::new(Access {
                    volatile: true,
                    ..*to
                }),
                value,
                words,
            }),
            _ => Err(refuse()),
        }
    }

    /// The census name (`access.pointer.<op>`) in the finished plan `slots`: a pointer store
    /// qualified by the reach of its destination ([`BufferReach`](super::BufferReach)).
    pub fn name(&self, slots: &AccessPlan) -> Cow<'static, str> {
        match self {
            PointerValue::CompareOffsets { .. } => "access.pointer.compare_offsets".into(),
            PointerValue::CompareAddresses { words: false, .. } => {
                "access.pointer.compare_addresses".into()
            }
            PointerValue::CompareAddresses { words: true, .. } => {
                "access.pointer.compare_addresses.words".into()
            }
            PointerValue::ToInt { .. } => "access.pointer.ptrtoint".into(),
            PointerValue::StorePointer { to, .. } => {
                to.reach(slots).qualify("access.pointer.store_pointer")
            }
        }
    }

    /// An upper estimate of the SPIR-V words one use emits.
    pub fn words(&self) -> u32 {
        match self {
            PointerValue::CompareOffsets { .. } => 24,
            PointerValue::CompareAddresses { a, b, words, .. } => a
                .words(*words)
                .saturating_add(b.words(*words))
                .saturating_add(if *words { 48 } else { 8 }),
            PointerValue::ToInt { of, words, .. } => of.words(*words).saturating_add(16),
            PointerValue::StorePointer { to, value, words } => to
                .words()
                .saturating_add(value.words(*words))
                .saturating_add(8),
        }
    }
}

/// Whether `v` is a device or constant pointer constant that is the null root: `null`, `undef`,
/// or `poison`. A threadgroup or private null is not: the arena families hold those pointers as
/// offsets with no null value, so such a constant operand stays refused there.
pub(super) fn null_constant(cx: &SelectCx<'_>, v: Value) -> bool {
    let Value::Const(c) = v else { return false };
    matches!(
        cx.entry.module.consts().get(c).map(|k| &k.value),
        Some(Const::Null | Const::Undef | Const::Poison)
    ) && buffer_space(cx, v)
}

/// The feature a handle of `class` is refused as, when no lowering here owns it: as its family
/// is for a parameter of that role (`lower::select::param_family`).
fn unlowered(class: HandleClass) -> Option<Feature> {
    match class {
        HandleClass::Texture | HandleClass::Sampler | HandleClass::Event => None,
        HandleClass::MeshGrid | HandleClass::Mesh => Some(Feature::MeshShading),
        HandleClass::AccelerationStructure => Some(Feature::RayTracing),
        HandleClass::FunctionTable => Some(Feature::DynamicCall),
        HandleClass::IntersectionTable => Some(Feature::IntersectionFunctionTable),
        HandleClass::Command
        | HandleClass::ComputePipelineState
        | HandleClass::RenderPipelineState => Some(Feature::ArgumentBufferObjects),
        // Read for a render command's setter alone, whose state no record holds.
        HandleClass::DepthStencilState => Some(Feature::IndirectRenderState),
    }
}

/// The feature of value `v`'s handle class, when it is a handle no lowering here owns.
fn unlowered_handle(cx: &SelectCx<'_>, v: Value) -> Option<Feature> {
    match cx.memory.repr(v)? {
        Repr::Handle { class, .. } | Repr::HandleArray { class } => unlowered(class),
        _ => None,
    }
}

/// Whether `v` is a device or constant pointer.
fn buffer_space(cx: &SelectCx<'_>, v: Value) -> bool {
    matches!(
        cx.value_ty(v).and_then(|t| cx.entry.module.types().get(t)),
        Some(Type::Ptr {
            space: crate::air::ir::AddrSpace::Device | crate::air::ir::AddrSpace::Constant
        })
    )
}

/// The start of the 2³² bytes pointer `v`'s proven offset range lies in, when it fits them.
fn range_start(cx: &SelectCx<'_>, v: Value) -> Option<i64> {
    match cx.memory.offset(v)? {
        OffsetRange::Span { lo, hi } if i128::from(hi) - i128::from(lo) < SPAN => Some(lo),
        _ => None,
    }
}

/// The start of the 2³² bytes both `a`'s and `b`'s proven offset ranges lie in, if any.
fn common_start(cx: &SelectCx<'_>, a: Value, b: Value) -> Option<i64> {
    match (cx.memory.offset(a)?, cx.memory.offset(b)?) {
        (OffsetRange::Span { lo: l1, hi: h1 }, OffsetRange::Span { lo: l2, hi: h2 }) => {
            let (lo, hi) = (l1.min(l2), h1.max(h2));
            (i128::from(hi) - i128::from(lo) < SPAN).then_some(lo)
        }
        _ => None,
    }
}

/// A compare operand, classified.
enum Kind {
    Null,
    Wide,
    Fat,
    Blob,
    Constant(u32),
}

fn kind(cx: &SelectCx<'_>, v: Value) -> Result<Kind, Error> {
    if null_constant(cx, v) {
        return Ok(Kind::Null);
    }
    match cx.memory.repr(v) {
        Some(Repr::BufPtr { wide: true, .. }) => Ok(Kind::Wide),
        Some(Repr::BufPtr { wide: false, .. }) => match blob::is_blob(cx, v)? {
            true => Ok(Kind::Blob),
            false => Ok(Kind::Fat),
        },
        _ => match local::home_of(cx, v) {
            Some(Home::Constant(k)) => Ok(Kind::Constant(k)),
            _ => Err(refuse()),
        },
    }
}

fn unsigned_or_equality(pred: IntPredicate) -> bool {
    !matches!(
        pred,
        IntPredicate::Sgt | IntPredicate::Sge | IntPredicate::Slt | IntPredicate::Sle
    )
}

/// The single candidate slot of static fat pointer `v` when it has exactly one and may be
/// neither null nor in the blob.
fn single_slot(cx: &SelectCx<'_>, v: Value) -> Result<Option<usize>, Error> {
    let s = statics(cx, v)?;
    Ok(match &*s.targets {
        [t] if !s.null && !s.in_blob => Some(t.slot),
        _ => None,
    })
}

/// The address of compare operand `v`: its slots' guest bases become push fields, and a slot
/// whose address only a compare observes gets a window (`access::finish_slots`).
fn address(cx: &mut SelectCx<'_>, v: Value, k: &Kind, m: &mut Meter) -> Result<Addr, Error> {
    match k {
        Kind::Null => Ok(Addr::Null),
        Kind::Wide => Ok(Addr::Wide(dynamic::plan_entry(cx, v)?)),
        Kind::Fat => {
            let (targets, null, in_blob) = buffer_roots(cx, v, m)?;
            // The blob has no Metal address the function can read.
            if in_blob {
                return Err(refuse());
            }
            let mut out = Vec::with_capacity(targets.len());
            for t in &targets {
                let slot = cx
                    .memory
                    .slots
                    .get(t.slot)
                    .ok_or_else(|| cx.invariant("a compared pointer names a missing slot"))?
                    .slot;
                cx.push.insert(PushField::SlotGuestBase(slot));
                cx.families.access.observed.insert(t.slot);
                out.push((t.slot, t.root));
            }
            Ok(Addr::Static(StaticAddr {
                ptr: operand(cx, v)?,
                targets: out,
                null,
                start: range_start(cx, v),
            }))
        }
        Kind::Blob | Kind::Constant(_) => Err(refuse()),
    }
}

/// Select `icmp pred a, b` of device or constant pointers, or `None` for a compare another
/// module selects: of thread or threadgroup pointers (the arena family's), and of wide pointers
/// with each other or with null (`dynamic`).
fn compare(
    cx: &mut SelectCx<'_>,
    pred: IntPredicate,
    a: Value,
    b: Value,
    m: &mut Meter,
) -> Result<Option<PointerValue>, Error> {
    if !buffer_space(cx, a) {
        return Ok(None);
    }
    let (ka, kb) = (kind(cx, a)?, kind(cx, b)?);
    Ok(Some(match (&ka, &kb) {
        (Kind::Wide, Kind::Wide | Kind::Null) | (Kind::Null, Kind::Wide) => return Ok(None),
        (Kind::Blob, Kind::Blob) if unsigned_or_equality(pred) => PointerValue::CompareOffsets {
            pred,
            a: Offset::Local(blob::blob_ptr(cx, a)?),
            b: Offset::Local(blob::blob_ptr(cx, b)?),
            start: Some(0),
            slot: None,
        },
        (Kind::Constant(i), Kind::Constant(j)) if i == j && unsigned_or_equality(pred) => {
            let home = Home::Constant(*i);
            PointerValue::CompareOffsets {
                pred,
                a: Offset::Local(local::local_ptr(cx, a, home)?),
                b: Offset::Local(local::local_ptr(cx, b, home)?),
                start: Some(0),
                slot: None,
            }
        }
        (Kind::Fat, Kind::Fat)
            if unsigned_or_equality(pred)
                && single_slot(cx, a)?.is_some()
                && single_slot(cx, a)? == single_slot(cx, b)? =>
        {
            PointerValue::CompareOffsets {
                pred,
                a: Offset::Fat(operand(cx, a)?),
                b: Offset::Fat(operand(cx, b)?),
                start: common_start(cx, a, b),
                slot: single_slot(cx, a)?,
            }
        }
        (Kind::Null | Kind::Fat | Kind::Wide, Kind::Null | Kind::Fat | Kind::Wide) => {
            let words = !cx.profile.fields().leaf.int64;
            let x = address(cx, a, &ka, m)?;
            let y = address(cx, b, &kb, m)?;
            if words && (matches!(x, Addr::Wide(_)) || matches!(y, Addr::Wide(_))) {
                // A wide pointer implies Profile A, which implies `shaderInt64`.
                return Err(cx.invariant("a wide pointer on a profile without shaderInt64"));
            }
            PointerValue::CompareAddresses {
                pred,
                a: x,
                b: y,
                words,
            }
        }
        _ => return Err(refuse()),
    }))
}

/// Whether stored or converted pointer `v` takes the static form here: a static fat pointer
/// (which on Profile A never escapes), or the null constant on a profile without device addresses.
/// Every other pointer is `dynamic`'s (or refused there).
fn static_escape(cx: &SelectCx<'_>, v: Value) -> Result<Option<Kind>, Error> {
    if !buffer_space(cx, v) {
        return Ok(None);
    }
    let addressed = cx.profile.fields().buffer_access == BufferAccess::Addressed;
    Ok(match kind(cx, v) {
        Ok(Kind::Fat) => Some(Kind::Fat),
        Ok(Kind::Null) if !addressed => Some(Kind::Null),
        _ => None,
    })
}

/// The 8-byte leaf of a stored Metal address of AIR pointer type `ty` (a `u64`, or two words
/// without `shaderInt64`).
fn address_leaf(cx: &SelectCx<'_>, ty: TypeId, align: Option<Align>) -> Result<Leaf, Error> {
    if cx.memory.layouts.store_size(ty) != Some(8) {
        return Err(refuse());
    }
    let held = if cx.profile.fields().leaf.int64 {
        Held::Int {
            bits: 64,
            container: 64,
        }
    } else {
        Held::Wide { bits: 64 }
    };
    Ok(Leaf {
        form: Form { held, lanes: 1 },
        bytes: 8,
        path: if align.map_or(1, |a| a.bytes()) >= 4 {
            Path::Aligned
        } else {
            Path::Bytes
        },
    })
}

/// A store of pointer `val` (taking the static form) through `ptr`: a leaf store of its
/// address through a static-root destination. A destination in the blob, an arena, or behind a
/// wide pointer is refused (a wide one needs device addresses, which make `val` wide too).
fn store_pointer(
    cx: &mut SelectCx<'_>,
    val: Value,
    k: Kind,
    ptr: Value,
    align: Option<Align>,
    m: &mut Meter,
) -> Result<PointerValue, Error> {
    let ty = cx
        .value_ty(val)
        .ok_or_else(|| cx.invariant("a stored pointer without a type"))?;
    let leaf = address_leaf(cx, ty, align)?;
    let (targets, null, in_blob) = buffer_roots(cx, ptr, m)?;
    if in_blob {
        return Err(refuse());
    }
    for t in &targets {
        let plan = cx
            .memory
            .slots
            .get(t.slot)
            .ok_or_else(|| cx.invariant("a target names a missing slot"))?;
        if super::view_kind(cx.profile, plan) != super::ViewKind::Storage {
            return Err(cx.invariant("a pointer store through a uniform view"));
        }
    }
    let to = super::access_with_leaf(cx, ptr, None, leaf, targets, null, None)?;
    let words = !cx.profile.fields().leaf.int64;
    Ok(PointerValue::StorePointer {
        to: Box::new(to),
        value: address(cx, val, &k, m)?,
        words,
    })
}

/// Select the pointer-value strategy of `inst`, or `None` when another module selects it. An
/// instruction on a handle of another family is refused as that family.
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    m: &mut Meter,
) -> Result<Option<PointerValue>, Error> {
    let mut family = unlowered_handle(cx, Value::Inst(at));
    inst.kind.for_each_value(|v| {
        if family.is_none() {
            family = unlowered_handle(cx, v);
        }
    });
    if let Some(f) = family {
        return Err(Unsupported::not_implemented(f).into());
    }
    match &inst.kind {
        InstKind::ICmp { pred, a, b } => compare(cx, *pred, *a, *b, m),
        InstKind::Cast {
            op: CastOp::PtrToInt,
            v,
        } => {
            let Some(k) = static_escape(cx, *v)? else {
                return Ok(None);
            };
            let bits = match cx.entry.module.types().get(inst.ty) {
                Some(Type::Int(b @ 1..=64)) => *b,
                _ => return Err(refuse()),
            };
            let words = !cx.profile.fields().leaf.int64;
            Ok(Some(PointerValue::ToInt {
                of: address(cx, *v, &k, m)?,
                bits,
                words,
            }))
        }
        InstKind::Store {
            val,
            ptr,
            align,
            volatile: false,
            atomic: None,
        } => match static_escape(cx, *val)? {
            Some(k) => store_pointer(cx, *val, k, *ptr, *align, m).map(Some),
            None => Ok(None),
        },
        _ => Ok(None),
    }
}

// ---- Emission ----

/// The null pointer in representation `ty` (a narrow or wide fat pointer), or `None` for
/// another representation.
pub fn null_value(fb: &mut FunctionBuilder<'_>, ty: &DemoteTy) -> BResult<Option<SValue>> {
    // A narrow fat pointer `(root, offset)`; a wide one `(root, at, host, size)`.
    let (width, len) = match ty {
        DemoteTy::FatPtr { wide: false } => (32, 2),
        DemoteTy::FatPtr { wide: true } => (64, 4),
        _ => return Ok(None),
    };
    let elem = fb.module().ty(SType::Int {
        width,
        signed: false,
    })?;
    let vec = fb.module().ty(SType::Vector { elem, len })?;
    let zero = fb.module().constant(Constant::Scalar {
        ty: elem,
        bits: ConstBits(0),
    })?;
    let c = fb.module().constant(Constant::Composite {
        ty: vec,
        parts: vec![zero; len as usize].into(),
    })?;
    Ok(Some(fb.module().const_value(c)?))
}

/// Whether AIR value `v` is a device or constant pointer constant that is the null root (the
/// emitter's view of [`null_constant`]).
fn null_source(cx: &EmitCx<'_>, v: ValueId) -> bool {
    let Some(Value::Const(c)) = cx.plan.values.value(v) else {
        return false;
    };
    let Some(k) = cx.entry.module.consts().get(c) else {
        return false;
    };
    matches!(k.value, Const::Null | Const::Undef | Const::Poison)
        && matches!(
            cx.entry.module.types().get(k.ty),
            Some(Type::Ptr {
                space: crate::air::ir::AddrSpace::Device | crate::air::ir::AddrSpace::Constant
            })
        )
}

/// The value of AIR value `v` when it is a null device or constant pointer constant used as a
/// plain operand: the null narrow fat pointer `uvec2(0, 0)`, the only representation such an
/// operand takes (a null root joins no wide component, and the wide forms' null sides are
/// explicit). `None` for any other value.
pub fn null_operand(
    cx: &EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    v: ValueId,
) -> BResult<Option<SValue>> {
    if !null_source(cx, v) {
        return Ok(None);
    }
    null_value(fb, &DemoteTy::FatPtr { wide: false })
}

/// The value phi `phi`'s copy on an incoming edge stores for incoming value `src`: a null
/// pointer constant in the phi's representation, else `src`'s value.
pub fn phi_incoming(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    phi: ValueId,
    src: ValueId,
) -> BResult<SValue> {
    if null_source(cx, src) {
        let ty = cx
            .plan
            .phis
            .binary_search_by_key(&phi, |(v, _)| *v)
            .ok()
            .and_then(|i| cx.plan.phis.get(i))
            .map(|(_, t)| t.clone());
        if let Some(ty) = ty {
            if let Some(x) = null_value(fb, &ty)? {
                return Ok(x);
            }
        }
    }
    cx.operand(fb, src)
}

/// A 64-bit Metal address: a `u64`, or its two words without `shaderInt64`.
#[derive(Clone, Copy, Debug)]
enum Address {
    U64(SValue),
    Words(Pair),
}

fn zero(fb: &mut FunctionBuilder<'_>, words: bool) -> BResult<Address> {
    Ok(if words {
        let z = k32(fb, 0)?;
        Address::Words(Pair { lo: z, hi: z })
    } else {
        Address::U64(k64(fb, 0)?)
    })
}

fn mismatch(cx: &mut EmitCx<'_>) -> crate::spirv::error::BuildError {
    cx.fail("a Metal address in two forms")
}

fn add(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    a: Address,
    b: Address,
) -> BResult<Address> {
    match (a, b) {
        (Address::U64(x), Address::U64(y)) => Ok(Address::U64(fb.binary(BinOp::IAdd, x, y)?)),
        (Address::Words(x), Address::Words(y)) => Ok(Address::Words(wide::add(fb, x, y)?)),
        _ => Err(mismatch(cx)),
    }
}

fn pick(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    c: SValue,
    a: Address,
    b: Address,
) -> BResult<Address> {
    match (a, b) {
        (Address::U64(x), Address::U64(y)) => Ok(Address::U64(fb.select(c, x, y)?)),
        (Address::Words(x), Address::Words(y)) => Ok(Address::Words(wide::sel(fb, c, x, y)?)),
        _ => Err(mismatch(cx)),
    }
}

/// The displacement `start + zext((m − start) mod 2³²)` the `u32` offset `m` stands for.
fn displacement(
    fb: &mut FunctionBuilder<'_>,
    m: SValue,
    start: i64,
    words: bool,
) -> BResult<Address> {
    if !words {
        return Ok(Address::U64(psb::displacement(fb, m, start)?));
    }
    let z = k32(fb, 0)?;
    if start == 0 {
        return Ok(Address::Words(Pair { lo: m, hi: z }));
    }
    // The two's-complement bits of `start` are its value modulo 2³² and 2⁶⁴.
    let low = k32(fb, start as u32)?;
    let rel = fb.binary(BinOp::ISub, m, low)?;
    let high = k32(fb, ((start as u64) >> 32) as u32)?;
    Ok(Address::Words(wide::add(
        fb,
        Pair { lo: low, hi: high },
        Pair { lo: rel, hi: z },
    )?))
}

/// `SlotGuestBase(slot)` (a `uvec2` push field) in the address form.
fn guest_base(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    slot: SlotRef,
    words: bool,
) -> BResult<Address> {
    let v = cx.push_field(fb, PushField::SlotGuestBase(slot))?;
    if words {
        return Ok(Address::Words(wide::split(fb, v)?));
    }
    Ok(Address::U64(slot_guest_base_u64(fb, v)?))
}

/// A `SlotGuestBase` field's value (a `uvec2`, low word first) as a `u64` (needs `shaderInt64`).
pub(super) fn slot_guest_base_u64(fb: &mut FunctionBuilder<'_>, v: SValue) -> BResult<SValue> {
    let u64_ = fb.module().ty(SType::Int {
        width: 64,
        signed: false,
    })?;
    fb.convert(ConvOp::Bitcast, v, u64_)
}

/// The start of the 2³² bytes a pointer into plan slot `slot` whose offset range the analysis
/// does not bound is taken to lie in. 0 when the slot's window lies in the binding's first 2³²
/// bytes (`Whole`, or `Bounded` within them): that span holds every pointer an access of the
/// window reaches and every pointer an MSL function can form from the argument (C++ pointer
/// arithmetic stays within `[b, b + size]`, and a descriptor slot's bound range is at most
/// `maxStorageBufferRange`). Else the window's start (every pointer an access of the window
/// reaches).
fn window_start(cx: &mut EmitCx<'_>, slot: usize) -> BResult<(SlotRef, i64)> {
    let Some(choice) = cx.plan.slots.get(slot).copied().flatten() else {
        return Err(cx.fail("a pointer address names an unplanned slot"));
    };
    let start = match choice.window {
        Window::Bounded { start, len }
            if start < 0 || i128::from(start) + i128::from(len) > SPAN =>
        {
            start
        }
        Window::Bounded { .. } | Window::Whole => 0,
    };
    Ok((choice.slot, start))
}

/// The Metal address of `a` in the open block.
fn emit_address(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    a: &Addr,
    words: bool,
) -> BResult<Address> {
    match a {
        Addr::Null => zero(fb, words),
        Addr::Wide(p) => {
            if words {
                return Err(cx.fail("a wide pointer's address in two words"));
            }
            let p = cx.operand(fb, *p)?;
            Ok(Address::U64(dynamic::guest_address(cx, fb, p)?))
        }
        Addr::Static(s) => {
            let fat = cx.operand(fb, s.ptr)?;
            let root = fat_part(fb, fat, 0)?;
            let m = fat_part(fb, fat, 1)?;
            let mut acc = if s.null {
                Some(displacement(fb, m, s.start.unwrap_or(0), words)?)
            } else {
                None
            };
            for &(slot, r) in &s.targets {
                let (slot, window) = window_start(cx, slot)?;
                let base = guest_base(cx, fb, slot, words)?;
                let d = displacement(fb, m, s.start.unwrap_or(window), words)?;
                let here = add(cx, fb, base, d)?;
                acc = Some(match acc {
                    None => here,
                    Some(prev) => {
                        let rk = k32(fb, r)?;
                        let is = fb.binary(BinOp::IEqual, root, rk)?;
                        pick(cx, fb, is, here, prev)?
                    }
                });
            }
            match acc {
                Some(x) => Ok(x),
                None => Err(cx.fail("a static address with no candidate")),
            }
        }
    }
}

/// The SPIR-V comparison of `pred` on scalars.
pub(super) fn predicate_op(pred: IntPredicate) -> BinOp {
    match pred {
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
    }
}

fn offset_value(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, o: Offset) -> BResult<SValue> {
    match o {
        Offset::Fat(v) => {
            let p = cx.operand(fb, v)?;
            fat_part(fb, p, 1)
        }
        Offset::Local(p) => blob::ptr_value(cx, fb, p),
    }
}

/// Address `a` truncated to `i<bits>`, held as its value family holds it: an `i1` as a `bool` (the
/// low bit), up to 32 bits zero-extended in a `u32`, a wider integer masked in a `u64` or, in two
/// words, as `Held::Wide`.
fn truncated(fb: &mut FunctionBuilder<'_>, a: Address, bits: u8) -> BResult<SValue> {
    let low = match a {
        Address::U64(x) if bits > 32 => {
            if bits == 64 {
                return Ok(x);
            }
            let m = k64(fb, (1u64 << bits) - 1)?;
            return fb.binary(BinOp::BitwiseAnd, x, m);
        }
        Address::Words(p) if bits > 32 => {
            let f = wide::wide_form(bits, 1);
            let p = wide::mask_top(fb, f, p)?;
            return wide::join(fb, f, p);
        }
        Address::U64(x) => {
            let u32_ = fb.module().ty(SType::Int {
                width: 32,
                signed: false,
            })?;
            fb.convert(ConvOp::UConvert, x, u32_)?
        }
        Address::Words(p) => p.lo,
    };
    match bits {
        1 => {
            let one = k32(fb, 1)?;
            let b = fb.binary(BinOp::BitwiseAnd, low, one)?;
            let z = k32(fb, 0)?;
            fb.binary(BinOp::INotEqual, b, z)
        }
        32 => Ok(low),
        _ => {
            let m = k32(fb, (1u32 << bits) - 1)?;
            fb.binary(BinOp::BitwiseAnd, low, m)
        }
    }
}

/// Address `a` as the register value of an 8-byte address leaf.
fn address_value(fb: &mut FunctionBuilder<'_>, a: Address) -> BResult<SValue> {
    match a {
        Address::U64(x) => Ok(x),
        Address::Words(p) => wide::join(fb, wide::wide_form(64, 1), p),
    }
}

/// Emit pointer-value strategy `s`, defining value `vid`.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    vid: ValueId,
    s: &PointerValue,
) -> BResult<()> {
    let r = match s {
        PointerValue::CompareOffsets {
            pred,
            a,
            b,
            start,
            slot,
        } => {
            let mut x = offset_value(cx, fb, *a)?;
            let mut y = offset_value(cx, fb, *b)?;
            if !matches!(pred, IntPredicate::Eq | IntPredicate::Ne) {
                let start = match (start, slot) {
                    (Some(s), _) => *s,
                    (None, Some(slot)) => window_start(cx, *slot)?.1,
                    (None, None) => 0,
                };
                if start as u32 != 0 {
                    let k = k32(fb, start as u32)?;
                    x = fb.binary(BinOp::ISub, x, k)?;
                    y = fb.binary(BinOp::ISub, y, k)?;
                }
            }
            fb.binary(predicate_op(*pred), x, y)?
        }
        PointerValue::CompareAddresses { pred, a, b, words } => {
            let x = emit_address(cx, fb, a, *words)?;
            let y = emit_address(cx, fb, b, *words)?;
            match (x, y) {
                (Address::U64(x), Address::U64(y)) => fb.binary(predicate_op(*pred), x, y)?,
                (Address::Words(x), Address::Words(y)) => {
                    let f = wide::wide_form(64, 1);
                    let x = wide::join(fb, f, x)?;
                    let y = wide::join(fb, f, y)?;
                    wide::int_compare(fb, *pred, f, x, y)?
                }
                _ => return Err(mismatch(cx)),
            }
        }
        PointerValue::ToInt { of, bits, words } => {
            let a = emit_address(cx, fb, of, *words)?;
            truncated(fb, a, *bits)?
        }
        PointerValue::StorePointer { to, value, words } => {
            let a = emit_address(cx, fb, value, *words)?;
            let v = address_value(fb, a)?;
            let p = cx.operand(fb, to.ptr)?;
            let m = fat_part(fb, p, 1)?;
            super::emit::access_at(cx, fb, to, p, m, Some(v))?;
            return Ok(());
        }
    };
    cx.define(vid, r);
    Ok(())
}
