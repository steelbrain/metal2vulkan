//! `lower::access`: loads, stores, address arithmetic, atomics, and memory intrinsics.
//!
//! **Pointer values**. A buffer pointer whose root set holds only static parameter slots (and
//! possibly null) is the fat pointer `uvec2(root, offset)`: `root` is `2 + n` for `[[buffer(n)]]`
//! (0 is null) and `offset` the `u32` Metal byte offset from the slot's bind offset, wrapping
//! (static roots need no `Int64`). A GEP is integer arithmetic on the offset with strides
//! from the AIR datalayout, and reads an `i64` index that is a zero or sign extension of a narrower
//! value as that value ([`index`]); `bitcast`, `addrspacecast`, and `freeze` copy the pointer,
//! `select` selects it, and a phi or a block-crossing pointer is demoted as the `uvec2`
//! ([`lowered_pointer`], [`spirv_type`]).
//!
//! **Accesses** ([`Access`]). A non-atomic load or store of a scalar or 2–4-lane vector is one
//! leaf (`memory::leaf`) through the slot's descriptor view (`memory::views`): the
//! descriptor-relative byte offset is the slot's `SlotBindOffset` residual plus the pointer's
//! offset. A pointer with several candidate slots is dispatched by a switch-at-use on its root; the
//! default arm is the null root, whose load is 0 and whose store is dropped. Containment is the
//! profile's encoding (`memory::contain`): none with robust descriptors, else the normative check
//! against the storage view's bound size (`OpArrayLength`, the padded range).
//!
//! **Slots** ([`finish_slots`]). An accessed slot is a descriptor view: with robust descriptors (or
//! containment off), a uniform view for a read-only `constant` slot whose `Bounded` window's
//! descriptor range fits `maxUniformBufferRange` (`u32[]` with `uniformBufferStandardLayout`, else
//! `uvec4[]`); a storage view otherwise. The in-shader encoding never uses a uniform view: its
//! declared size is the window's range, while the binder may bind a shorter range (the binder
//! clamps it to the allocation), and a shader cannot query a uniform range. A storage slot whose
//! range is not provably within `maxStorageBufferRange` declares `RuntimeFact::BufferRangeLimit`.
//! On Profile F an exceeded per-stage storage-buffer limit views read-only slots as uniform buffers
//! instead: sized to the window where it bounds the range, else to `maxUniformBufferRange`, with
//! the range (`UniformViewRange`) and, for a `device` slot, the absence of aliasing writes
//! (`UniformViewAliasing`) left to the binder. On Profile A a `BufferRangeLimit` variant
//! (`LowerInputs.psb_slots`) and the per-stage descriptor-limit fallback move storage slots to
//! their device addresses (`BufferForm::Psb`, the PSB leaf of `psb`, reached through the root
//! table), which declare `RuntimeFact::DeviceAddresses`. The pulled vertex fetch's source buffers
//! are slots under the same rules ([`pulled`]): one sharing a read-only buffer argument's Metal
//! index is that argument's slot, widened to a `Whole` storage slot; any other is a read-only
//! `Whole` storage slot of its own.
//!
//! **Census names**. A slot's form is decided after its accesses are selected, so a strategy's
//! census name reads the finished plan ([`AccessPlan::check`], [`BufferReach`]): an access that
//! reaches a device-address slot or a wide pointer's allocation is named apart from a descriptor
//! access of the same strategy (`.psb`, or `.psb_and_view` when it may reach both), and `.checked`
//! names the check its targets take as emitted, a device-address slot's in-shader one whatever the
//! profile's encoding. A strategy whose side may be an arena or buffer memory names that side's
//! [`SideReach`] (`.arena`), and a memory intrinsic names each side apart (`.to_<reach>`,
//! `.from_<reach>`, `copy::MemOp::name`).
//!
//! **Arenas** ([`local`]): threadgroup and private-arena pointers are `u32` byte offsets into their
//! arena (`memory::arena`); typed allocas and constant `Private` globals are typed objects with
//! clamped dynamic indices (`memory::alloca`). A kernel's explicit imageblock is the threadgroup
//! arena's last region, its cells reached through `air.imageblock_data` ([`imageblock`]).
//!
//! **The `ArgumentTable`** ([`table`]). Push fields that do not fit the stage's push window are
//! read from the stage's `ArgumentTable` uniform buffer, which this family declares and counts as
//! one more uniform descriptor in the per-stage limits.
//!
//! **Atomics, memory intrinsics, and the constant blob**: `air` and LLVM memory atomics on buffer
//! slots, wide pointers, and the threadgroup arena, typed (`u64`, native `f32` add) ones through
//! device addresses ([`atomic`]); `memcpy`/`memmove`/`memset` between buffers, wide pointers,
//! arenas, and the blob ([`copy`]); pointers into the pipeline constant blob and loads through them
//! ([`blob`]).
//!
//! **Buffer tables** ([`buffer_table`]): table pointers are `u32` offsets into the table; a load of
//! an element pointer is the element's static fat pointer, a dynamic element index clamped into the
//! elements the load may read.
//!
//! **Dynamic roots** ([`dynamic`]): pointers loaded from memory (buffers and pointer cells that do
//! not forward) or made from integers, pointers whose address escapes, and every pointer sharing a
//! root component with them, are wide fat pointers `u64vec2(root, offset)` resolved through the
//! root table's resident set and accessed by device address, contained, with root-0 accesses
//! counted under isolation; a stored or converted pointer is its guest address.
//!
//! **Handles** ([`handle`]): texture and sampler values with static candidates are `u32` handle
//! indices (`memory::handles`); a `select`, `phi`, `freeze`, or cast of them is integer arithmetic,
//! and a handle-array parameter is a byte offset whose element loads yield the element slot's
//! handle, clamped into the elements the load may read. A texture handle loaded from an argument
//! buffer is its set-1 heap element, found in the stage's handle table.
//!
//! **Pointer values** ([`mod@pointer`]): `null`, `undef`, and `poison` pointers are the null root
//! (a `select` side, a phi's incoming value, a GEP base, an access's pointer); compares of device
//! and constant pointers compare offsets within one object and Metal addresses (`SlotGuestBase`
//! plus displacement, or a wide pointer's root entry) otherwise; on Profile F a stored or converted
//! static pointer is its Metal address; an instruction on a handle of a family lowered elsewhere is
//! refused as that family.
//!
//! **Volatile accesses**. A volatile load or store is selected as the same instruction without
//! `volatile` and then takes its volatile form (`volatile_form`): every word load and store it
//! composes carries SPIR-V's `Volatile` (performed, once, never combined), in the function's memory
//! model, whose coherence it keeps. Loads of read-only memories (uniform views, the constant blob,
//! typed constant objects) stay plain, and a handle-array or buffer-table element load, which reads
//! no memory at the access, is refused volatile.
//!
//! Not lowered yet (`Unsupported{Lowering(Access), NotImplemented}`): coherent accesses of
//! threadgroup memory, 64-bit threadgroup atomics, compares involving the constant blob or a
//! typed constant and another object, and aggregate accesses.

pub mod absent;
pub mod atomic;
pub mod blob;
pub mod buffer_table;
pub mod cell;
pub mod copy;
pub mod dynamic;
mod emit;
pub mod encode;
pub mod function_table;
pub mod handle;
pub mod imageblock;
pub mod index;
pub mod local;
pub(crate) mod phis;
pub mod pointer;
mod psb;
pub mod pulled;
pub(crate) mod root_sets;
pub mod table;
pub mod tile;
pub(crate) mod tile_memory;

pub use emit::emit;

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use crate::air::abi::{AbiInt, ParamBinding, ParamRole};
use crate::air::ir::{AddrSpace, CastOp, Const, Inst, InstId, InstKind, Type, Value, ValueId};
use crate::budget::Meter;
use crate::device::profile::{BufferAccess, Containment, Limits};
use crate::device::requirements::VkLimit;
use crate::device::DeviceProfile;
use crate::error::{Capability, Error, Feature, LowerFamily, Unsupported};
use crate::lower::emit::EmitCx;
use crate::lower::select::{DescriptorCounts, FamilyPlans, Plan, SelectCx, SlotChoice};
use crate::lower::strategy::FamilyStrategy;
use crate::memory::contain::{self, Check};
use crate::memory::leaf::{self, Coherence, Leaf, Units};
use crate::memory::plan::SlotPlan;
use crate::memory::provenance::Repr;
use crate::memory::views::{self, View, ViewKind};
use crate::pipeline::facts::{RuntimeFact, UniformViewChoice};
use crate::pipeline::interface::{
    AccessSet, BufferForm, DescriptorView, PushLoc, PushPlacement, Window,
};
use crate::pipeline::sym::{PushField, SlotRef};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, TypeId, Value as SValue};
use crate::spirv::ir::{ConstBits, Constant, ExecMode, Type as SType};

/// The root number of the null pointer.
pub const NULL_ROOT: u32 = 0;

/// The root number of the constant blob in a static fat pointer whose root set mixes blob-form
/// constant globals with parameter slots: above every slot's `2 + n`.
pub const BLOB_ROOT: u32 = u32::MAX;

/// The root number of `[[buffer(n)]]`.
pub fn slot_root(index: u32) -> u32 {
    index.saturating_add(2)
}

/// The access strategies.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AccessStrategy {
    /// A static-root fat pointer computed from other pointers.
    Pointer(PtrOp),
    /// A leaf load.
    Load(Access),
    /// A leaf store.
    Store(Access),
    /// An arena pointer or access (threadgroup and private arenas).
    Local(local::LocalStrategy),
    /// A memory atomic.
    Atomic(atomic::AtomicAccess),
    /// A pointer into the constant blob, or a load through one.
    Blob(blob::BlobStrategy),
    /// `llvm.memcpy`, `llvm.memmove`, or `llvm.memset`.
    Memory(copy::MemOp),
    /// A buffer-table pointer or element-pointer load.
    Table(buffer_table::TableStrategy),
    /// A wide (dynamic-root) pointer or an access through one.
    Dynamic(dynamic::DynStrategy),
    /// A texture or sampler handle value, or a handle-array offset.
    Handle(handle::HandleStrategy),
    /// A pointer compare, or a static pointer converted or stored on Profile F.
    Value(pointer::PointerValue),
    /// A pointer stored into or loaded from a forwarding pointer cell.
    Cell(cell::CellStrategy),
    /// A kernel imageblock's cell pointer or extent.
    Imageblock(imageblock::ImageblockStrategy),
    /// An indirect-command object's word, or an encoder operation's record writes.
    Encode(encode::EncodeStrategy),
    /// A word of a visible-function table read by its resource id.
    FunctionTable(function_table::TableWord),
    /// A read of an argument the function constants removed.
    AbsentRead(absent::AbsentRead),
}

/// An operand of a static fat pointer `select`: a fat pointer, or a pointer into the constant
/// blob (its offset) that the `select` places under [`BLOB_ROOT`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PtrSide {
    Fat(ValueId),
    Blob(local::LocalPtr),
    /// The null pointer constant (`null`, `undef`, or `poison`): `uvec2(0, 0)`.
    Null,
}

/// The operand `v` of a fat-pointer `select`.
fn select_side(cx: &mut SelectCx<'_>, v: Value) -> Result<PtrSide, Error> {
    if pointer::null_constant(cx, v) {
        return Ok(PtrSide::Null);
    }
    if blob::is_blob(cx, v)? {
        blob::use_blob(cx)?;
        return Ok(PtrSide::Blob(blob::blob_ptr(cx, v)?));
    }
    statics(cx, v)?;
    Ok(PtrSide::Fat(operand(cx, v)?))
}

/// A pointer-valued instruction on static-root fat pointers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PtrOp {
    Gep(Gep),
    /// `bitcast`, `addrspacecast`, or `freeze`: the operand itself.
    Copy {
        of: ValueId,
    },
    /// `select c, t, f`.
    Select {
        c: ValueId,
        t: PtrSide,
        f: PtrSide,
    },
}

/// `getelementptr`: `offset + konst + Σ index · stride`, in wrapping `u32` arithmetic (the
/// low 32 bits of LLVM's wrapping 64-bit GEP arithmetic).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gep {
    pub base: ValueId,
    /// The constant part, mod 2^32.
    pub konst: u32,
    /// The dynamic indices with their byte strides (mod 2^32).
    pub terms: Vec<(ValueId, u32, IndexForm)>,
}

/// How a GEP index operand (held in the arithmetic family's form) becomes the index value mod
/// 2^32: its sign extension for an index of the operand's own type, or the extension an index
/// extension applies ([`index`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IndexForm {
    /// `i1`, sign-extended: `select(x, -1, 0)`.
    Bool,
    /// `i1`, zero-extended (`zext i1 to i64`): `select(x, 1, 0)`.
    BoolZero,
    /// `i2..i31` zero-extended in a `u32`: shifted up and arithmetically back.
    Narrow { bits: u8 },
    /// `i32`, sign-extended (a plain `i32` index, or `sext i32 to i64`).
    Word,
    /// `i2..i32` zero-extended (`zext` to `i64`): held zero-extended in a `u32`. The same word as
    /// [`IndexForm::Word`] modulo 2³²; only a 64-bit offset (a wide pointer) tells them apart.
    WordZero,
    /// `i33..i64` in a `u64`: its low word.
    Wide,
}

/// The dynamic terms of a GEP offset: `(index, byte stride mod 2^32, index form)`.
pub type GepTerms = Vec<(ValueId, u32, IndexForm)>;

/// One candidate slot of an access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Target {
    /// The slot, as an index into the memory plan's slots.
    pub slot: usize,
    /// Its root number (the switch-at-use literal).
    pub root: u32,
    /// Its containment check.
    pub check: Check,
}

/// A leaf load or store through a static-root pointer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Access {
    pub ptr: ValueId,
    /// The stored value (stores only).
    pub value: Option<ValueId>,
    pub leaf: Leaf,
    /// The candidate slots, by root.
    pub targets: Vec<Target>,
    /// The pointer may be null (the switch default).
    pub null: bool,
    /// The coherence of a `*_coherent` access; `None` for the function's own.
    pub coherence: Option<Coherence>,
    /// A volatile access: every word it loads or stores carries `Volatile`.
    pub volatile: bool,
    /// The pointer may point into the constant blob (a load only): the blob's containment check,
    /// read under [`BLOB_ROOT`] in the switch-at-use.
    pub blob: Option<Check>,
}

impl Access {
    fn words(&self) -> u32 {
        let per = self
            .targets
            .iter()
            .map(|t| {
                // The residual or root-table reads, and the containment check (the PSB form,
                // which a slot may take after selection, costs the most).
                leaf_words(&self.leaf)
                    + match t.check {
                        Check::None => 24,
                        Check::InShader => 64,
                    }
                    + 160
            })
            .fold(0u32, u32::saturating_add);
        let blob = match self.blob {
            Some(_) => leaf_words(&self.leaf).saturating_add(96),
            None => 0,
        };
        per.saturating_add(blob)
            .saturating_add(16u32.saturating_mul(self.targets.len() as u32 + 2))
    }
}

fn leaf_words(l: &Leaf) -> u32 {
    l.cost()
}

impl Access {
    /// The buffer memory the access reaches in the finished plan `slots`: its targets' slots, and
    /// the constant blob's view when the pointer may point there.
    pub fn reach(&self, slots: &AccessPlan) -> BufferReach {
        let blob = if self.blob.is_some() {
            BufferReach::VIEWS
        } else {
            BufferReach::NONE
        };
        slots.reach(&self.targets).union(blob)
    }

    /// Whether some target is checked in the shader as `slots` emits it ([`AccessPlan::check`]).
    fn checked(&self, slots: &AccessPlan) -> bool {
        self.targets
            .iter()
            .any(|t| slots.check(t) == Check::InShader)
    }

    /// The census name of a load or store named `base` (`access.load`, `access.store`):
    /// `.switch` for several candidate slots, then the reach of its targets
    /// ([`BufferReach::qualifier`]), then `.checked` when one of them is checked in the shader.
    fn name(&self, base: &'static str, slots: &AccessPlan) -> Cow<'static, str> {
        let switch = if self.targets.len() > 1 {
            ".switch"
        } else {
            ""
        };
        let checked = if self.checked(slots) { ".checked" } else { "" };
        match (switch, self.reach(slots).qualifier(), checked) {
            ("", "", "") => base.into(),
            (switch, reach, checked) => format!("{base}{switch}{reach}{checked}").into(),
        }
    }
}

/// How a strategy's accesses reach buffer memory: through descriptor views, which the profile's
/// containment encoding contains (no check with robust descriptors, the in-shader check otherwise),
/// by device address (a `BufferForm::Psb` slot, or a wide pointer's root entry), which is checked
/// in the shader whatever the profile (unless containment is off), or both. Arenas and typed
/// objects are neither: their containment is their own ([`SideReach`]).
///
/// The census names a strategy that may reach either by its reach ([`BufferReach::qualify`]), so
/// that one name never covers two leaf encodings: a run of a descriptor access is no evidence for
/// a device-address access of the same strategy, nor the reverse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct BufferReach {
    /// Some access is through a descriptor view.
    pub views: bool,
    /// Some access is by device address.
    pub addresses: bool,
}

impl BufferReach {
    /// No buffer memory.
    pub const NONE: BufferReach = BufferReach {
        views: false,
        addresses: false,
    };
    /// Descriptor views only.
    pub const VIEWS: BufferReach = BufferReach {
        views: true,
        addresses: false,
    };
    /// Device addresses only.
    pub const ADDRESSES: BufferReach = BufferReach {
        views: false,
        addresses: true,
    };

    /// Both reaches.
    pub fn union(self, other: BufferReach) -> BufferReach {
        BufferReach {
            views: self.views || other.views,
            addresses: self.addresses || other.addresses,
        }
    }

    /// The census qualifier of the reach: none without a device address, `.psb` with device
    /// addresses only, `.psb_and_view` with both.
    pub fn qualifier(self) -> &'static str {
        match (self.views, self.addresses) {
            (_, false) => "",
            (false, true) => ".psb",
            (true, true) => ".psb_and_view",
        }
    }

    /// The qualifier without its dot (`psb`, `psb_and_view`), or none for descriptor views alone.
    pub fn token(self) -> Option<&'static str> {
        self.qualifier().strip_prefix('.')
    }

    /// The census name `base` qualified by the reach ([`BufferReach::qualifier`]).
    pub fn qualify(self, base: &'static str) -> Cow<'static, str> {
        match self.qualifier() {
            "" => base.into(),
            q => format!("{base}{q}").into(),
        }
    }
}

/// What one side of an access reaches when it may be an arena or buffer memory (a memory
/// intrinsic's endpoint, a simdgroup block, a pointer word): buffer memory ([`BufferReach`]), or
/// the threadgroup or private arena, whose units are contained in the shader in every leg (unless
/// containment is off; a unit at a constant offset inside the arena's translation-time size needs
/// no check), whatever the profile's buffer encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SideReach {
    /// Buffer memory: descriptor views, device addresses, or both.
    Buffer(BufferReach),
    /// An arena.
    Arena,
}

impl SideReach {
    /// The census token of the side: none for descriptor views alone (or no memory), else the
    /// buffer reach's ([`BufferReach::token`]) or `arena`.
    pub fn token(self) -> Option<&'static str> {
        match self {
            SideReach::Buffer(b) => b.token(),
            SideReach::Arena => Some("arena"),
        }
    }

    /// The census name `base` of a strategy with one side, qualified by the side's token
    /// (`base.<token>`).
    pub fn qualify(self, base: &'static str) -> Cow<'static, str> {
        match self.token() {
            None => base.into(),
            Some(t) => format!("{base}.{t}").into(),
        }
    }
}

impl AccessPlan {
    /// Whether target `t`'s slot is reached by its device address (`finish_slots`).
    pub fn by_address(&self, t: &Target) -> bool {
        matches!(
            self.slots.get(t.slot).copied().flatten(),
            Some(SlotLowering::Psb { .. })
        )
    }

    /// The containment check target `t` takes as emitted (`emit::prepare`): its view's, the
    /// profile's encoding (`t.check`), or for a device-address slot the in-shader check, which
    /// robust buffer access does not replace, unless containment is off (`psb_contained`).
    pub fn check(&self, t: &Target) -> Check {
        match (self.by_address(t), self.psb_contained) {
            (false, _) => t.check,
            (true, true) => Check::InShader,
            (true, false) => Check::None,
        }
    }

    /// The reach of static-root `targets` ([`BufferReach`]).
    pub fn reach(&self, targets: &[Target]) -> BufferReach {
        targets
            .iter()
            .map(|t| {
                if self.by_address(t) {
                    BufferReach::ADDRESSES
                } else {
                    BufferReach::VIEWS
                }
            })
            .fold(BufferReach::NONE, BufferReach::union)
    }

    /// The form `finish_slots` gave pulled source buffer `slot` ([`pulled`]): its accessed slot's,
    /// or its own.
    pub fn source_lowering(&self, slot: SlotRef) -> Option<SlotLowering> {
        match self.pulled.iter().find(|p| p.slot == slot)?.reach {
            pulled::Reach::Shared { slot } => self.slots.get(slot).copied().flatten(),
            pulled::Reach::Own(l) => Some(l),
        }
    }

    /// The reach of the pulled reads of source buffer `slot` ([`AccessPlan::source_lowering`]).
    pub fn source_reach(&self, slot: SlotRef) -> BufferReach {
        match self.source_lowering(slot) {
            Some(SlotLowering::Psb { .. }) => BufferReach::ADDRESSES,
            Some(SlotLowering::View(_)) => BufferReach::VIEWS,
            None => BufferReach::NONE,
        }
    }
}

impl FamilyStrategy for AccessStrategy {
    fn name(&self, plans: &FamilyPlans) -> Cow<'static, str> {
        let slots = &plans.access;
        match self {
            AccessStrategy::Pointer(PtrOp::Gep(_)) => "access.gep".into(),
            AccessStrategy::Pointer(PtrOp::Copy { .. }) => "access.pointer_copy".into(),
            AccessStrategy::Pointer(PtrOp::Select { .. }) => "access.pointer_select".into(),
            AccessStrategy::Load(a) => a.name("access.load", slots),
            AccessStrategy::Store(a) => a.name("access.store", slots),
            AccessStrategy::Local(l) => l.name().into(),
            AccessStrategy::Atomic(a) => a.name(slots),
            AccessStrategy::Blob(b) => b.name().into(),
            AccessStrategy::Memory(m) => m.name(slots),
            AccessStrategy::Table(t) => t.name().into(),
            AccessStrategy::Dynamic(d) => d.name(slots),
            AccessStrategy::Handle(h) => h.name(slots),
            AccessStrategy::Value(p) => p.name(slots),
            AccessStrategy::Cell(c) => c.name().into(),
            AccessStrategy::Imageblock(i) => i.name().into(),
            AccessStrategy::Encode(e) => e.name(slots),
            AccessStrategy::FunctionTable(t) => t.name(),
            AccessStrategy::AbsentRead(a) => a.name().into(),
        }
    }

    fn words(&self) -> u32 {
        match self {
            AccessStrategy::Pointer(PtrOp::Gep(g)) => {
                12u32.saturating_add(12u32.saturating_mul(g.terms.len() as u32))
            }
            AccessStrategy::Pointer(PtrOp::Copy { .. }) => 2,
            AccessStrategy::Pointer(PtrOp::Select { .. }) => 6,
            AccessStrategy::Load(a) | AccessStrategy::Store(a) => a.words(),
            AccessStrategy::Local(l) => l.words(),
            AccessStrategy::Atomic(a) => a.words(),
            AccessStrategy::Blob(b) => b.words(),
            AccessStrategy::Memory(m) => m.words(),
            AccessStrategy::Table(t) => t.words(),
            AccessStrategy::Dynamic(d) => d.words(),
            AccessStrategy::Handle(h) => h.words(),
            AccessStrategy::Value(p) => p.words(),
            AccessStrategy::Cell(c) => c.words(),
            AccessStrategy::Imageblock(i) => i.words(),
            AccessStrategy::Encode(e) => e.words(),
            AccessStrategy::FunctionTable(t) => t.words(),
            AccessStrategy::AbsentRead(a) => a.words(),
        }
    }
}

pub(crate) fn not_lowered(family: LowerFamily) -> Error {
    Unsupported::not_implemented(Feature::Lowering(family)).into()
}

fn refuse() -> Error {
    not_lowered(LowerFamily::Access)
}

/// The candidate slots and null-ness of static-root buffer pointer `v`, or the refusal of any
/// other pointer, including one that may point into the constant blob ([`buffer_roots`]).
fn static_roots(cx: &SelectCx<'_>, v: Value, m: &mut Meter) -> Result<(Vec<Target>, bool), Error> {
    let s = statics(cx, v)?;
    if s.in_blob {
        return Err(refuse());
    }
    Ok((held(s, m)?, s.null))
}

/// The candidate slots, null-ness, and blob-ness of static-root buffer pointer `v`: whether its
/// root set holds a blob-form constant global, besides parameter slots and null. Every buffer root
/// (a parameter or an element of a buffer table) maps to its slot through
/// `MemoryPlan::slot_of_root`; any other root is refused.
fn buffer_roots(
    cx: &SelectCx<'_>,
    v: Value,
    m: &mut Meter,
) -> Result<(Vec<Target>, bool, bool), Error> {
    let s = statics(cx, v)?;
    Ok((held(s, m)?, s.null, s.in_blob))
}

/// What [`buffer_roots`] reads, without copying the candidates: the set's facts, which
/// `root_sets::plan` read once for the function. A pointer whose candidates are only checked, not
/// held, reads them here at no cost per root.
fn statics<'c>(cx: &'c SelectCx<'_>, v: Value) -> Result<&'c root_sets::Statics, Error> {
    let Some(Repr::BufPtr { roots, wide: false }) = cx.memory.repr(v) else {
        return Err(refuse());
    };
    let set = cx
        .root_sets
        .get(roots)
        .ok_or_else(|| cx.invariant("a buffer pointer's root set was not read"))?;
    set.statics.as_ref().map_err(|r| r.error(cx))
}

/// The candidates of `s`, as an access holds them: one work unit per candidate, which also
/// stands for the walks of them the access makes while it is selected (placing them, noting their
/// units, and noting an ordered access's slots), and their bytes charged to the ledger for the
/// rest of the attempt, as the plan holds its strategies until emission ends.
fn held(s: &root_sets::Statics, m: &mut Meter) -> Result<Vec<Target>, Error> {
    let n = s.targets.len();
    m.tick(u32::try_from(n).unwrap_or(u32::MAX))?;
    m.charge(n.saturating_mul(std::mem::size_of::<Target>()))?;
    Ok(s.targets.to_vec())
}

/// The lowered representation of pointer value `v`: the fat pointer of a static-root buffer
/// pointer. A phi's incoming null constant is the null fat pointer, which the copy on its edge
/// materializes (`pointer::phi_incoming`); any other constant incoming pointer, and a blob offset
/// (which would need a conversion on its edge), is refused.
pub fn lowered_pointer(cx: &SelectCx<'_>, v: Value) -> Result<DemoteTy, Error> {
    if let Some(d) = encode::lowered_pointer(cx, v) {
        return Ok(d);
    }
    if let Some(d) = handle::lowered_pointer(cx, v)? {
        return Ok(d);
    }
    if let Some(d) = dynamic::lowered_pointer(cx, v)? {
        return Ok(d);
    }
    if let Some(d) = local::lowered_pointer(cx, v)? {
        return Ok(d);
    }
    if let Some(d) = blob::lowered_pointer(cx, v)? {
        return Ok(d);
    }
    if let Some(d) = buffer_table::lowered_pointer(cx, v) {
        return Ok(d);
    }
    statics(cx, v)?;
    // A blob offset merging into a fat pointer would need a conversion on its edge (the phi's
    // incoming values, read once per phi).
    if let Some(e) = phis::incoming(cx, v)?.and_then(|i| phis::fat_refusal(cx, i)) {
        return Err(e);
    }
    Ok(DemoteTy::FatPtr { wide: false })
}

/// The SPIR-V type of a pointer representation `ty`: `uvec2` for a narrow fat pointer, `u32`
/// for an arena offset or a handle index, `u64vec2` for a wide fat pointer, `None` for the others.
pub fn spirv_type(mb: &mut ModuleBuilder, ty: &DemoteTy) -> BResult<Option<TypeId>> {
    match ty {
        DemoteTy::FatPtr { wide: true } => Ok(Some(dynamic::wide_type(mb)?)),
        DemoteTy::ArenaOffset | DemoteTy::HandleIndex => Ok(Some(mb.ty(SType::Int {
            width: 32,
            signed: false,
        })?)),
        DemoteTy::FatPtr { wide: false } => {
            let u = mb.ty(SType::Int {
                width: 32,
                signed: false,
            })?;
            Ok(Some(mb.ty(SType::Vector { elem: u, len: 2 })?))
        }
        _ => Ok(None),
    }
}

/// How an accessed slot is reached.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SlotLowering {
    /// Through its descriptor view.
    View(ViewKind),
    /// Through its device address (`BufferForm::Psb`). `start` is the start of the slot's window
    /// relative to the bind offset (0 for `Window::Whole`): the `u32` pointer offset is recovered
    /// as the unique 64-bit displacement in `[start, start + 2³²)` congruent to it (`psb`).
    Psb { start: i64 },
}

/// The access family's part of the per-function plan: how each accessed slot is reached, by
/// memory-plan slot index (the plan's `slots` carry the forms).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AccessPlan {
    pub slots: Vec<Option<SlotLowering>>,
    /// Device-address accesses are contained (robust buffer access does not cover them, so only
    /// `Containment::UnsafeOff` turns this off).
    pub psb_contained: bool,
    /// The narrow units device-address accesses load and store directly (`psb::narrow_units` of
    /// the profile): a sub-word store there is one narrow store instead of two word atomics.
    pub psb_units: Units,
    /// The arenas the function uses.
    pub local: local::LocalPlan,
    /// The function's index extensions ([`index`]): planned before any instruction is selected
    /// ([`index::plan_extensions`]).
    pub index_extensions: BTreeSet<InstId>,
    /// The function's index words ([`index`]): planned with the index extensions.
    pub index_words: BTreeSet<InstId>,
    /// The function's handle words ([`handle`]): integer loads of a handle-array element or of an
    /// argument-buffer member's resource id that only `inttoptr`s back to the handle read, held as
    /// the handle index.
    pub handle_words: BTreeMap<InstId, handle::HandleWord>,
    /// The encodings the loads of each pointer-cell slot group read ([`cell`]), by group: planned
    /// before any store is selected ([`cell::plan_reads`]).
    pub cell_reads: BTreeMap<InstId, cell::Reads>,
    /// The `ptrtoint`s whose bits go only into forwarding cells (`CellFacts::bits`) and whose
    /// pointer a cell holds in its own encoding: each emits nothing, and its stores store its
    /// pointer ([`cell`]). A wide pointer's bits take the integer path.
    pub cell_bits: BTreeSet<InstId>,
    /// The pointer-word loads from forwarding cells (`CellFacts::words`) whose pointer a cell holds
    /// in its own encoding: each loads its `inttoptr`s' pointer ([`cell`]). A wide pointer's word
    /// is loaded as data.
    pub cell_words: BTreeSet<InstId>,
    /// The push placement spills into the stage's `ArgumentTable` uniform buffer, whose descriptor
    /// `finish_slots` counted and whose fields this family reads ([`table`]).
    pub argument_table: bool,
    /// The constant blob the function reads ([`blob`]): the artifact carries its bytes.
    pub blob: Option<crate::memory::blob::BlobUse>,
    /// Wide pointers and the slots they reach ([`dynamic`]).
    pub dynamic: dynamic::DynamicPlan,
    /// Slots a typed atomic (`u64`, or a native `f32` add) reaches: reached by device address,
    /// the only typed alias of a slot's words ([`atomic`]).
    pub typed_slots: BTreeSet<usize>,
    /// The pulled vertex fetch's source buffers, by Metal index ([`pulled`]).
    pub pulled: Vec<pulled::PulledSlot>,
    /// Some texture handle is loaded from an argument buffer: the function reads the stage's
    /// handle table ([`handle`]), one more storage descriptor, and its search reads the ids as
    /// this.
    pub handle_table: Option<handle::HandleIds>,
    /// Some sampler handle is loaded from an argument buffer: the function reads the stage's
    /// sampler table ([`handle`]), one more storage descriptor, and its search reads the ids as
    /// this.
    pub sampler_table: Option<handle::HandleIds>,
    /// Memory-plan slots whose Metal address (`SlotGuestBase`) a pointer compare reads
    /// ([`mod@pointer`]): a slot no access bounds gets the window `Whole`, so the binder plans it
    /// and writes the field.
    pub observed: BTreeSet<usize>,
    /// Memory-plan slots a volatile, coherent, or atomic access reaches: what they read may change
    /// while the dispatch or draw runs, and an atomic needs a storage view's word pointer, so the
    /// per-stage storage-buffer fallback never views them as uniform buffers.
    pub ordered: BTreeSet<usize>,
    /// The units ([`Units`]) the static-root accesses of each slot read or write whole, by
    /// memory-plan slot index: narrow units whose storage access the profile has, and `uvec4`
    /// units. The slot gets a view of each that its final view kind holds
    /// (`views::UnitViews::declare`).
    pub units: BTreeMap<usize, Units>,
    /// A tile function's attachments and the ones it stages ([`tile`]).
    pub tile: Option<tile::TilePlan>,
    /// The handle loads of indirect-command objects whose every use is an encoder operand
    /// ([`encode::plan_objects`]): each is its field's word, a `u64`.
    pub objects: BTreeSet<InstId>,
    /// The typed loads and stores whose access chain takes its defining `getelementptr`'s own
    /// index operands ([`local`]'s line-up rule), each with those operands: the access reads them
    /// where it stands, beside its pointer (`FamilyPlans::reads`).
    pub typed_indices: BTreeMap<InstId, Vec<Value>>,
    /// The kernel is a device-replay variant (`LoweringInputs::replay_bindings`): every accessed
    /// slot is a device-address slot, and a slot's Metal address is its root entry's `guest +
    /// bind_offset` ([`replay_guest_base`]), never a `SlotGuestBase` push field.
    pub replay: bool,
    /// A heap read the shader checks observes its texel's alpha (`lower::texture`): the
    /// handle-table search carries the alpha (`memory::handles::Lookup::Flagged`), so every heap
    /// handle carries its entry's out-of-bounds-alpha flag beside its element.
    pub heap_alpha: bool,
    /// Some visible-function table is read by its resource id ([`function_table`]): the function
    /// reads the stage's function-table directory, one more storage descriptor, whose search reads
    /// the ids as this states, and the heap's function-table array, or, on a device without the
    /// array's features, the stage's function-table snapshot, one more again.
    pub function_tables: Option<function_table::TableReads>,
    /// A heap texture's access reads its texel class: the handle-table search carries the class's index
    /// (`memory::handles::Lookup::Flagged`), so every heap handle carries it beside its element.
    pub heap_class: bool,
    /// A heap texture's float write the function makes tests its texel class's index
    /// (`lower::texture::emit`): the handle-table search carries the index, as under
    /// [`heap_class`](Self::heap_class), but the module needs no table of the classes.
    pub heap_class_index: bool,
    /// The relaxed atomic loads an exit of a loop holding them depends on ([`atomic::spin`]): each
    /// acquires its memory. Planned before any instruction is selected.
    pub spins: BTreeSet<InstId>,
    /// The stage binds the command buffer's `ExecutionStatus` buffer
    /// (`SyntheticBinding::ExecutionStatus`), one more storage descriptor, on a device that does
    /// not address buffers, and why ([`absent::StatusBinding`]).
    pub status_binding: Option<absent::StatusBinding>,
}

impl AccessPlan {
    /// Whether the handle-table search carries each heap handle's texel class index.
    pub fn carries_class(&self) -> bool {
        self.heap_class || self.heap_class_index
    }

    /// What the function's handle-table search yields.
    pub fn lookup(&self) -> crate::memory::handles::Lookup {
        if self.heap_alpha || self.carries_class() {
            crate::memory::handles::Lookup::Flagged {
                alpha: self.heap_alpha,
                class: self.carries_class(),
            }
        } else {
            crate::memory::handles::Lookup::Element
        }
    }
}

/// The access family's module-scope objects.
#[derive(Clone, Debug, Default)]
pub struct Declared {
    /// The view of each accessed slot, by memory-plan slot index.
    pub views: Vec<Option<View>>,
    /// The arenas.
    pub local: local::LocalDecl,
    /// The stage's `ArgumentTable`, when the placement spilled fields there.
    pub table: Option<table::Table>,
    /// The constant blob's storage view, when the function reads the blob.
    pub blob: Option<View>,
    /// The storage view of each pulled source buffer with a descriptor slot of its own, by Metal
    /// index ([`pulled`]).
    pub pulled: Vec<(u32, View)>,
    /// The stage's handle table and its search, when a texture handle is loaded from an
    /// argument buffer.
    pub handle_table: Option<handle::HandleTable>,
    /// The stage's sampler table and its search, when a sampler handle is loaded from an
    /// argument buffer.
    pub sampler_table: Option<handle::HandleTable>,
    /// The unit views of each accessed slot, by memory-plan slot index.
    pub units: Vec<views::UnitViews>,
    /// The module's resident-set search, when a function searches it more than once
    /// ([`dynamic`]).
    pub resolve: Option<crate::spirv::ids::FuncId>,
    /// A tile function's attachment images and render area ([`tile`]).
    pub tile: tile::TileDecl,
    /// The stage's function-table directory and search and the heap's function-table array,
    /// when a table is read by its resource id.
    pub function_tables: Option<function_table::FunctionTables>,
    /// The stage's view of the command buffer's `ExecutionStatus` buffer, where it binds the
    /// buffer ([`absent::declare`]).
    pub status: Option<View>,
    /// The `RangeAtBindOffset` specialization of each slot read through a uniform view, by
    /// memory-plan slot index: where it holds, an access reads the view at the Metal offset alone
    /// ([`range_at_bind_offset`]).
    pub at_bind: Vec<Option<SValue>>,
}

/// Slot `slot`'s `RangeAtBindOffset` specialization: a `bool`, false by default, fixed at pipeline
/// creation, true where the binder starts the slot's descriptor range at its bind offset, so its
/// `SlotBindOffset` residual is 0 and an access reads the Metal offset alone: a constant for a
/// constant member, which a driver can push ahead of the shader.
fn range_at_bind_offset(mb: &mut ModuleBuilder, slot: SlotRef) -> BResult<SValue> {
    let b = mb.ty(SType::Bool)?;
    let c = mb.spec(
        crate::pipeline::sym::SpecKey::RangeAtBindOffset(slot),
        b,
        ConstBits(0),
    )?;
    mb.const_value(c)
}

/// The module's routines the access family's strategies call, each charged once to the census: the
/// handle-table and function-table-directory searches and the resident-set search.
pub fn routine_words(plan: &AccessPlan) -> impl Iterator<Item = (&'static str, u64)> + '_ {
    handle::routine_words(plan)
        .chain(function_table::routine_words(plan.function_tables))
        .chain(dynamic::routine_words(&plan.dynamic))
}

/// Plan a used resource parameter (buffer, argument buffer, threadgroup, indirect constant).
/// Buffer and argument-buffer parameters are pointer values (their fat pointer); the other roles
/// are lowered elsewhere.
pub fn select_param(cx: &mut SelectCx<'_>, b: &ParamBinding) -> Result<(), Error> {
    match b.role {
        // A buffer table's value is an offset into the table; its elements are planned where their
        // pointers are loaded.
        ParamRole::Buffer(_) | ParamRole::ArgumentBuffer(_)
            if buffer_table::is_table_param(cx.entry, b) =>
        {
            Ok(())
        }
        ParamRole::Buffer(_) | ParamRole::ArgumentBuffer(_) if dynamic::is_wide_param(cx, b) => {
            dynamic::select_param(cx, b)
        }
        ParamRole::Buffer(_) | ParamRole::ArgumentBuffer(_) => {
            if cx.memory.slot_of_param(b.param).is_none() {
                // A parameter no access reaches has no slot; its value is never needed, except
                // by a pointer use that is not an access, which is refused where it is selected.
                return Ok(());
            }
            match statics(cx, Value::Arg(crate::air::ir::ArgIdx::new(b.param)))? {
                s if s.in_blob => Err(refuse()),
                _ => Ok(()),
            }
        }
        // A tile function's is its tile's memory, which the pass keeps. An object or mesh
        // function's payload is its own arena.
        ParamRole::Threadgroup(_) | ParamRole::Payload { .. } => local::select_param(cx, b),
        // The imageblock's value (a struct holding its opaque base) has no lowered form: a kernel
        // reaches the imageblock only through its intrinsics.
        ParamRole::Imageblock(_) => Err(Unsupported::not_implemented(Feature::Imageblocks).into()),
        _ => Err(refuse()),
    }
}

/// The value of a resource parameter `b`: the fat pointer `uvec2(2 + n, 0)` of `[[buffer(n)]]`.
pub fn param_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: &ParamBinding,
) -> BResult<SValue> {
    if let ParamRole::Threadgroup(_) | ParamRole::Payload { .. } = &b.role {
        return local::param_value(cx, fb, b);
    }
    if buffer_table::is_table_param(cx.entry, b) {
        return buffer_table::param_value(fb);
    }
    if let Some(v) = dynamic::param_value(cx, fb, b)? {
        return Ok(v);
    }
    let index = match &b.role {
        ParamRole::Buffer(arg) | ParamRole::ArgumentBuffer(arg) => match arg.location.index {
            AbiInt::Const(i) => Some(i),
            _ => None,
        },
        _ => None,
    };
    let Some(index) = index else {
        return Err(cx.fail(format!("parameter {} is no buffer slot", b.param)));
    };
    let u = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let v2 = fb.module().ty(SType::Vector { elem: u, len: 2 })?;
    let root = fb.module().constant(Constant::Scalar {
        ty: u,
        bits: ConstBits(u64::from(slot_root(index))),
    })?;
    let zero = fb.module().constant(Constant::Scalar {
        ty: u,
        bits: ConstBits(0),
    })?;
    let c = fb.module().constant(Constant::Composite {
        ty: v2,
        parts: [root, zero].into(),
    })?;
    fb.module().const_value(c)
}

/// The view kind of a slot on profile `p`: a uniform view for a read-only `constant` slot with a
/// `Bounded` window whose descriptor range fits `maxUniformBufferRange`, when the profile's
/// containment encoding emits no check; a storage view otherwise.
///
/// Under the in-shader encoding a uniform view could not be contained: the bound range may be
/// shorter than the view's declared size (the binder clamps it to the padded allocation), a uniform
/// block has no `OpArrayLength`, and the proven window bounds Metal-defined accesses only. A
/// storage view's check reads the size actually bound.
pub fn view_kind(p: &DeviceProfile, s: &SlotPlan) -> ViewKind {
    let limits = &p.fields().limits;
    let uniform = match s.window {
        Some(window)
            if s.space == AddrSpace::Constant
                && !s.access.write
                && !s.atomic
                && contain::encoding(p) == Check::None =>
        {
            uniform_bytes(window, limits)
        }
        _ => None,
    };
    match uniform {
        Some(bytes) => uniform_view(p, bytes),
        None => ViewKind::Storage,
    }
}

/// The uniform view of `bytes` bytes (a multiple of 16) on `p`: `u32[]` with
/// `uniformBufferStandardLayout`, `uvec4[]` without.
fn uniform_view(p: &DeviceProfile, bytes: u32) -> ViewKind {
    if p.fields().descriptors.uniform_buffer_standard_layout {
        ViewKind::UniformWords { words: bytes / 4 }
    } else {
        ViewKind::UniformVec4 { vecs: bytes / 16 }
    }
}

/// Whether a uniform view's descriptor range is provably within `maxUniformBufferRange`: a
/// `Bounded` window's range is at most `uniform_range(len, A)`, with `A` the descriptor base
/// alignment `max(minUniformBufferOffsetAlignment, 16)`.
fn uniform_range_fits(window: Window, limits: &Limits) -> bool {
    uniform_bytes(window, limits).is_some()
}

/// The uniform range of a `Bounded` window that provably fits `maxUniformBufferRange`.
fn uniform_bytes(window: Window, limits: &Limits) -> Option<u32> {
    match window {
        Window::Bounded { len, .. } => {
            let align = views::base_alignment(limits.min_uniform_buffer_offset_alignment);
            views::uniform_range(len, align)
                .filter(|r| *r <= u64::from(limits.max_uniform_buffer_range))
                .and_then(|r| u32::try_from(r).ok())
        }
        _ => None,
    }
}

/// The uniform view the per-stage storage-buffer fallback gives a read-only slot with `window`:
/// sized to the window's range where that provably fits `maxUniformBufferRange`, else the largest
/// whole number of 16-byte vectors within it, which covers every range instantiation accepts
/// (`BindError::DescriptorRangeExceeded` above the limit).
fn fallback_view(p: &DeviceProfile, window: Window) -> ViewKind {
    let limits = &p.fields().limits;
    let bytes = uniform_bytes(window, limits).unwrap_or(limits.max_uniform_buffer_range & !15);
    uniform_view(p, bytes)
}

/// The fallback's order among read-only storage slots, lowest first: a window whose uniform range
/// provably fits, then a `constant` slot, then a `device` slot, which also needs instantiation's
/// alias check. Within a rank the highest Metal index moves first, as the Profile A fallback does.
fn fallback_rank(plan: &SlotPlan, window: Window, limits: &Limits) -> u8 {
    if uniform_range_fits(window, limits) {
        0
    } else if plan.space == AddrSpace::Constant {
        1
    } else {
        2
    }
}

/// Whether a storage slot's descriptor range is provably within `maxStorageBufferRange`: a
/// `Bounded` window's range is at most `round_up(len + align - 1, 16)`, with `align` the descriptor
/// base alignment `max(minStorageBufferOffsetAlignment, 16)`.
fn storage_range_fits(window: Window, limits: &Limits) -> bool {
    match window {
        Window::Bounded { len, .. } => {
            let align = views::base_alignment(limits.min_storage_buffer_offset_alignment);
            views::uniform_range(len, align)
                .is_some_and(|r| r <= u64::from(limits.max_storage_buffer_range))
        }
        _ => false,
    }
}

fn descriptor_view(kind: ViewKind) -> DescriptorView {
    match kind {
        ViewKind::Storage => DescriptorView::Storage,
        ViewKind::UniformWords { .. } => DescriptorView::UniformWords,
        ViewKind::UniformVec4 { .. } => DescriptorView::UniformVec4,
    }
}

/// Choose the form of every accessed slot and of every pulled vertex source buffer, and add their
/// descriptors to `cx.descriptors`. Returns the form of each accessed slot, by memory-plan slot
/// index (`None` for slots the function does not access).
///
/// A slot is a descriptor view unless a `BufferRangeLimit` variant (`LowerInputs.psb_slots`)
/// moves it to its device address, or, on Profile A, the per-stage descriptor limits are exceeded:
/// then the highest-index storage slots move to device addresses, one at a time, until the
/// storage-buffer and resource counts fit (a deterministic per-function choice). Every
/// device-address slot declares `DeviceAddresses` and reads the root table
/// (`PushField::RootTableAddress`) instead of its bind residual.
///
/// On Profile F with robust descriptors (or containment off), an exceeded per-stage
/// storage-buffer limit moves read-only storage slots to uniform views instead, one at a time in
/// `fallback_rank` order, while a uniform buffer is left. Every candidate whose window does not
/// bound its range declares `UniformViewRange` whether or not the fallback ran (so a plan can ask
/// for the variant that keeps them all storage views), each slot the compile keeps a storage view
/// declares it `Kept`, and a viewed `device` slot declares `UniformViewAliasing`.
///
/// A pulled source buffer ([`pulled`]) shares the accessed slot of its Metal index, widened to a
/// `Whole` storage slot that the fetch reads too (a window outside the first 2³² bytes above the
/// bind offset is refused), or is a read-only `Whole` storage slot of its own; either takes part
/// in the rules above like any storage slot.
///
/// Each step of either fallback moves one storage view, so the descriptors and push fields of the
/// slots' forms are kept in a census the step updates, and the order of the slots it moves is
/// sorted once: the fallbacks take one pass over the slots, metered, where they recounted every
/// slot per slot moved before. The working tables are charged while they run.
pub fn finish_slots(
    cx: &mut SelectCx<'_>,
    m: &mut Meter,
) -> Result<Vec<Option<SlotChoice>>, Error> {
    let work = cx.memory.slots.len().saturating_mul(SLOT_WORK_BYTES);
    m.charge(work)?;
    let result = place_slots(cx, m);
    m.release(work);
    result
}

/// The ledger bytes `place_slots` works in per memory-plan slot: its form and window, whether
/// the fallback viewed it, and its places in the two fallbacks' orders.
const SLOT_WORK_BYTES: usize = std::mem::size_of::<Option<SlotLowering>>()
    + std::mem::size_of::<Option<Window>>()
    + 1
    + std::mem::size_of::<(usize, u8, u32)>()
    + std::mem::size_of::<(u32, Moved)>();

/// The descriptors and push fields of the slots' forms while [`finish_slots`] moves them: the
/// storage and uniform views, and the push fields with each device-address slot's
/// `SlotBindOffset` replaced by the root table. A move updates it in place.
struct Census {
    storage: u32,
    uniform: u32,
    fields: BTreeSet<PushField>,
}

impl Census {
    /// The census of `forms` over the function's push fields `push`; `dynamic` when a wide
    /// pointer reads the root table.
    fn new(
        push: &BTreeSet<PushField>,
        dynamic: bool,
        forms: impl Iterator<Item = (SlotRef, SlotLowering)>,
        m: &mut Meter,
    ) -> Result<Census, Error> {
        let mut c = Census {
            storage: 0,
            uniform: 0,
            fields: push.clone(),
        };
        let mut psb = dynamic;
        for (slot, l) in forms {
            m.tick(1)?;
            match l {
                SlotLowering::View(ViewKind::Storage) => c.storage = c.storage.saturating_add(1),
                SlotLowering::View(_) => c.uniform = c.uniform.saturating_add(1),
                SlotLowering::Psb { .. } => {
                    c.fields.remove(&PushField::SlotBindOffset(slot));
                    psb = true;
                }
            }
        }
        if psb {
            c.fields.insert(PushField::RootTableAddress);
        }
        Ok(c)
    }

    /// The slots' descriptors in `stage`: their views, and the `ArgumentTable` when the push
    /// fields spill.
    fn counts(&self, stage: crate::pipeline::sym::Stage) -> DescriptorCounts {
        DescriptorCounts {
            uniform_buffers: u32::from(table::spills(stage, &self.fields))
                .saturating_add(self.uniform),
            storage_buffers: self.storage,
            ..DescriptorCounts::default()
        }
    }

    /// A storage view became a view of `kind`.
    fn viewed(&mut self, kind: ViewKind) {
        self.storage = self.storage.saturating_sub(1);
        match kind {
            ViewKind::Storage => self.storage = self.storage.saturating_add(1),
            ViewKind::UniformWords { .. } | ViewKind::UniformVec4 { .. } => {
                self.uniform = self.uniform.saturating_add(1)
            }
        }
    }

    /// A storage view of `slot` became its device address.
    fn addressed(&mut self, slot: SlotRef) {
        self.storage = self.storage.saturating_sub(1);
        self.fields.remove(&PushField::SlotBindOffset(slot));
        self.fields.insert(PushField::RootTableAddress);
    }
}

/// [`finish_slots`]'s placement, in its charged working tables.
fn place_slots(cx: &mut SelectCx<'_>, m: &mut Meter) -> Result<Vec<Option<SlotChoice>>, Error> {
    let p = cx.profile;
    let limits = &p.fields().limits;
    let addressed = p.fields().buffer_access == BufferAccess::Addressed;
    let range_variant = || -> Error {
        // Profile F honors `BufferRangeLimit` at plan time; it has no device address.
        Unsupported::needs_any_of(Feature::BufferRange, vec![p.addressed_needs()]).into()
    };
    let memory = cx.memory;
    let stage = cx.stage;
    let sources: Vec<SlotRef> = cx
        .families
        .stage_io
        .buffers
        .iter()
        .map(|&b| crate::lower::stage_io::fetch::slot(stage, b))
        .collect();
    let replay = cx.inputs.replay_bindings;
    if replay || cx.inputs.flat_grid || cx.inputs.folded_groups {
        plan_replay(cx, &sources)?;
    }
    let shared_slot = |s: SlotRef| {
        memory
            .slots
            .iter()
            .position(|m| m.slot == s && m.window.is_some())
    };
    let mut lowering: Vec<Option<SlotLowering>> = Vec::with_capacity(memory.slots.len());
    let mut windows: Vec<Option<Window>> = Vec::with_capacity(memory.slots.len());
    for (i, plan) in memory.slots.iter().enumerate() {
        m.tick(1)?;
        // A slot only a compare reaches is planned for its guest base.
        let observed = cx
            .families
            .access
            .observed
            .contains(&i)
            .then_some(Window::Whole);
        let Some(window) = plan.window.or(observed) else {
            lowering.push(None);
            windows.push(None);
            continue;
        };
        let shared = sources.contains(&plan.slot);
        let window = if shared {
            pulled::share(window)?
        } else {
            window
        };
        windows.push(Some(window));
        // A slot a wide pointer may reach is read through its root entry; a slot a typed atomic
        // reaches needs a typed pointer, which only its device address gives; a replay variant
        // reads every slot through its root entry, which the replay's preprocess writes from the
        // replayed command's record.
        if replay
            || cx.families.access.dynamic.slots.contains(&i)
            || cx.families.access.typed_slots.contains(&i)
        {
            lowering.push(Some(psb_lowering(window)?));
            continue;
        }
        // The fetch's unbounded read needs a storage view.
        let kind = if shared {
            ViewKind::Storage
        } else {
            view_kind(p, plan)
        };
        let mut l = SlotLowering::View(kind);
        if kind == ViewKind::Storage {
            if !storage_range_fits(window, limits) {
                cx.facts
                    .insert(RuntimeFact::BufferRangeLimit { slot: plan.slot });
            }
            if cx.inputs.psb_slots.contains(&plan.slot) {
                if !addressed {
                    return Err(range_variant());
                }
                l = psb_lowering(window)?;
            }
        }
        lowering.push(Some(l));
    }
    // Source buffers no accessed slot shares: read-only `Whole` storage slots of their own, whose
    // range is never provably within `maxStorageBufferRange`.
    let mut own: Vec<(SlotRef, SlotLowering)> = Vec::new();
    for &s in &sources {
        if shared_slot(s).is_some() {
            continue;
        }
        cx.facts.insert(RuntimeFact::BufferRangeLimit { slot: s });
        let l = if cx.inputs.psb_slots.contains(&s) {
            if !addressed {
                return Err(range_variant());
            }
            psb_lowering(Window::Whole)?
        } else {
            SlotLowering::View(ViewKind::Storage)
        };
        own.push((s, l));
    }
    // The push fields as they will be once the slot forms are final: a device-address slot reads
    // the root table instead of its residual. The `ArgumentTable` is one more uniform buffer when
    // they spill (the synthetic bindings count against the limits).
    let forms = memory
        .slots
        .iter()
        .zip(&lowering)
        .filter_map(|(plan, l)| l.map(|l| (plan.slot, l)))
        .chain(own.iter().copied());
    let mut census = Census::new(&cx.push, cx.families.access.dynamic.used, forms, m)?;
    // Profile F's per-stage storage-buffer fallback: with robust descriptors (or containment off),
    // while the storage-buffer count exceeds its limit and a uniform buffer is left, the first
    // read-only storage slot in `fallback_rank` order becomes a uniform view. Slots the variant
    // keeps (`LowerInputs::storage_view_slots`), a pulled fetch's source, and slots a volatile,
    // coherent, or atomic access reaches stay storage views.
    let inputs = cx.inputs;
    let fallback = !addressed && contain::encoding(p) == Check::None;
    let candidates: Vec<(usize, u8)> = if fallback {
        memory
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, plan)| {
                let window = windows.get(i).copied().flatten()?;
                let storage = lowering.get(i).copied().flatten()
                    == Some(SlotLowering::View(ViewKind::Storage));
                let read_only = !plan.access.write && !plan.atomic;
                let eligible = storage
                    && read_only
                    && !sources.contains(&plan.slot)
                    && !cx.families.access.ordered.contains(&i);
                eligible.then(|| (i, fallback_rank(plan, window, limits)))
            })
            .collect()
    } else {
        Vec::new()
    };
    // The order the fallback views the candidates in: by rank, the highest Metal index first. The
    // sort is stable, so equal keys keep memory-plan order, the first of which the fallback
    // takes, as the search for the least key per step once did.
    let mut order: Vec<(usize, u8, u32)> = Vec::new();
    for &(i, rank) in &candidates {
        m.tick(1)?;
        if let Some(plan) = memory.slots.get(i) {
            if !inputs.storage_view_slots.contains(&plan.slot) {
                order.push((i, rank, plan.slot.index));
            }
        }
    }
    order.sort_by_key(|&(_, rank, index)| (rank, std::cmp::Reverse(index)));
    let mut viewed = vec![false; memory.slots.len()];
    for (i, _, _) in order {
        m.tick(1)?;
        let total = cx.descriptors.plus(&census.counts(stage));
        if total.storage_buffers <= limits.max_per_stage_descriptor_storage_buffers
            || total.uniform_buffers >= limits.max_per_stage_descriptor_uniform_buffers
        {
            break;
        }
        let Some(window) = windows.get(i).copied().flatten() else {
            break;
        };
        let kind = fallback_view(p, window);
        if let Some(l) = lowering.get_mut(i) {
            *l = Some(SlotLowering::View(kind));
            census.viewed(kind);
        }
        if let Some(v) = viewed.get_mut(i) {
            *v = true;
        }
    }
    // The fallback's facts, on every compile where it can run: each candidate whose window does
    // not bound its uniform range states what this compile made of it (`Viewed`, or `Spare`
    // whether or not the fallback ran), so a plan can name every slot a variant must keep a
    // storage view, a variant that pulls a vertex layout included (it adds a storage buffer); a
    // viewed `device` slot needs the alias check. Every slot the compile keeps states `Kept`,
    // candidate or not, so the plan of the variant restates its key.
    for &(i, _) in &candidates {
        let (Some(plan), Some(window)) = (memory.slots.get(i), windows.get(i).copied().flatten())
        else {
            continue;
        };
        let slot = plan.slot;
        if inputs.storage_view_slots.contains(&slot) {
            continue;
        }
        let is_viewed = viewed.get(i).copied().unwrap_or(false);
        if is_viewed {
            // The slot is no storage view any more. A slot nothing reads (only compared) cannot
            // observe a write, so it needs no alias check.
            cx.facts.remove(&RuntimeFact::BufferRangeLimit { slot });
            if plan.space != AddrSpace::Constant && plan.access.read {
                cx.facts.insert(RuntimeFact::UniformViewAliasing { slot });
            }
        }
        if !uniform_range_fits(window, limits) {
            let choice = if is_viewed {
                UniformViewChoice::Viewed
            } else {
                UniformViewChoice::Spare
            };
            cx.facts
                .insert(RuntimeFact::UniformViewRange { slot, choice });
        }
    }
    for plan in &memory.slots {
        if inputs.storage_view_slots.contains(&plan.slot) {
            cx.facts.insert(RuntimeFact::UniformViewRange {
                slot: plan.slot,
                choice: UniformViewChoice::Kept,
            });
        }
    }
    if addressed {
        // The storage views the step may move, the highest Metal index first: the accessed slots
        // (by their chosen windows) in memory-plan order, then the source buffers of their own.
        // A stable sort read from its end takes the last of equal indices, as the search for the
        // greatest index per step once did.
        let storage = SlotLowering::View(ViewKind::Storage);
        let mut order: Vec<(u32, Moved)> = Vec::new();
        for (i, (plan, (l, w))) in memory
            .slots
            .iter()
            .zip(lowering.iter().zip(&windows))
            .enumerate()
        {
            m.tick(1)?;
            if let (Some(l), Some(w)) = (l, w) {
                if *l == storage {
                    order.push((plan.slot.index, Moved::Accessed(i, *w)));
                }
            }
        }
        for (k, (s, l)) in own.iter().enumerate() {
            m.tick(1)?;
            if *l == storage {
                order.push((s.index, Moved::Own(k)));
            }
        }
        order.sort_by_key(|(index, _)| *index);
        loop {
            m.tick(1)?;
            let total = cx.descriptors.plus(&census.counts(stage));
            let storage_bound = matches!(
                total.exceeded(limits),
                Some((
                    VkLimit::MaxPerStageDescriptorStorageBuffers | VkLimit::MaxPerStageResources,
                    _,
                    _
                ))
            );
            if !storage_bound {
                break;
            }
            let Some((_, moved)) = order.pop() else {
                break;
            };
            match moved {
                Moved::Accessed(i, window) => {
                    let psb = psb_lowering(window)?;
                    if let (Some(l), Some(plan)) = (lowering.get_mut(i), memory.slots.get(i)) {
                        *l = Some(psb);
                        census.addressed(plan.slot);
                    }
                }
                Moved::Own(k) => {
                    let psb = psb_lowering(Window::Whole)?;
                    if let Some((s, l)) = own.get_mut(k) {
                        *l = psb;
                        census.addressed(*s);
                    }
                }
            }
        }
    }
    let mut slots = Vec::with_capacity(memory.slots.len());
    let addressed_slot = |cx: &mut SelectCx<'_>, slot: SlotRef| {
        cx.facts.insert(RuntimeFact::DeviceAddresses { slot });
        cx.push.remove(&PushField::SlotBindOffset(slot));
        cx.push.insert(PushField::RootTableAddress);
    };
    for (plan, (l, w)) in memory.slots.iter().zip(lowering.iter().zip(&windows)) {
        let (Some(l), Some(window)) = (l, w) else {
            slots.push(None);
            continue;
        };
        let form = match l {
            SlotLowering::View(kind) => BufferForm::Descriptor(descriptor_view(*kind)),
            SlotLowering::Psb { .. } => {
                addressed_slot(cx, plan.slot);
                BufferForm::Psb
            }
        };
        // The slot's reported access is its own accesses' and what the accesses through the
        // dynamic root may do to it (`SlotPlan::wide`).
        let access = AccessSet {
            read: plan.access.read || plan.wide.read,
            write: plan.access.write || plan.wide.write,
        };
        slots.push(Some(SlotChoice {
            slot: plan.slot,
            form,
            // A shared slot is the function's and the fetch's (`pulled::shared_access`).
            access: if sources.contains(&plan.slot) {
                pulled::shared_access(access)
            } else {
                access
            },
            window: *window,
        }));
    }
    for (slot, l) in &own {
        if let SlotLowering::Psb { .. } = l {
            addressed_slot(cx, *slot);
        }
    }
    let counts = census.counts(stage);
    cx.descriptors = cx.descriptors.plus(&counts);
    cx.families.access.argument_table = table::spills(stage, &cx.push);
    let mut sourced: Vec<pulled::PulledSlot> = own
        .iter()
        .map(|&(s, l)| pulled::PulledSlot {
            slot: s,
            reach: pulled::Reach::Own(l),
        })
        .collect();
    for &s in &sources {
        if let Some(slot) = shared_slot(s) {
            sourced.push(pulled::PulledSlot {
                slot: s,
                reach: pulled::Reach::Shared { slot },
            });
        }
    }
    sourced.sort_by_key(|p| p.slot.index);
    cx.families.access.pulled = sourced;
    cx.families.access.slots = lowering;
    cx.families.access.psb_contained = p.fields().containment != Containment::UnsafeOff;
    cx.families.access.psb_units = psb::narrow_units(p);
    Ok(slots)
}

/// The checks and push fields of a device-replay variant (`LoweringInputs::replay_bindings`), a
/// flat variant (`LoweringInputs::flat_grid`) or a folded variant
/// (`LoweringInputs::folded_groups`), before its slots' forms are chosen: a kernel with neither a
/// stage-input descriptor or fetch (`sources`) nor a tile function's attachments, or for a replay
/// variant also a vertex function with no fetch or a fragment function, or for a folded variant
/// also an object function (whose draw's pieces fold its grid, `bind::mesh_arguments`), on a
/// profile with device addresses. A replay variant reads every `SlotGuestBase` the selection asked for from its
/// slot's root entry instead ([`replay_guest_base`]), so the push block holds the root table's
/// address in its place; and a kernel that reads a grid builtin reads its `DispatchRegion` from the
/// region record the replay's preprocess writes per command and region ([`replay_region`]), so the
/// push block holds that record's address in its place. A flat variant reads the record's
/// `threads_per_grid` whatever builtins it reads, from a replay's record or, bound as any other
/// kernel, from the record a preprocess writes for its dispatch beside the group counts; and it
/// runs no emulated simdgroups: its threads return early past the grid, which no workgroup
/// barrier may follow. A folded variant reads its group counts and every grid builtin's region
/// from the record a preprocess writes for its dispatch, whatever builtins it reads; its
/// workgroups past Metal's last threadgroup return early as wholes, so it runs emulated
/// simdgroups too.
fn plan_replay(cx: &mut SelectCx<'_>, sources: &[SlotRef]) -> Result<(), Error> {
    let p = cx.profile;
    if p.fields().buffer_access != BufferAccess::Addressed {
        return Err(Unsupported::needs_any_of(
            Feature::GpuEncodedIndirectCommands,
            vec![p.addressed_needs()],
        )
        .into());
    }
    // A replayed command is a compute dispatch with no stage-in region and no tile pass, or a
    // draw whose vertex function fetches no attribute: its buffers are the record's, read through
    // the slot table, which no fetch of a vertex descriptor reads.
    use crate::pipeline::sym::Stage;
    let drawn = cx.inputs.replay_bindings && matches!(cx.stage, Stage::Vertex | Stage::Fragment);
    // A folded object function: its draw's pieces fold Metal's object grid
    // (`bind::mesh_arguments`).
    let pieces = cx.inputs.folded_groups && cx.stage == Stage::Object;
    let fetches = cx
        .inputs
        .vertex_fetch
        .as_ref()
        .is_some_and(|f| !f.attributes.is_empty());
    if (cx.stage != Stage::Kernel && !drawn && !pieces)
        || fetches
        || !sources.is_empty()
        || cx.inputs.stage_input.is_some()
        || cx.inputs.tile_attachments.is_some()
        || cx.families.access.tile.is_some()
    {
        return Err(Unsupported::not_implemented(Feature::GpuEncodedIndirectCommands).into());
    }
    if cx.inputs.flat_grid && cx.inputs.emulated_simdgroups {
        return Err(Unsupported::not_implemented(Feature::GpuEncodedIndirectCommands).into());
    }
    // A folded variant reads its group counts from the record whatever builtins it reads.
    if cx.push.remove(&PushField::DispatchRegion) || cx.inputs.flat_grid || cx.inputs.folded_groups
    {
        cx.push.insert(PushField::ReplayRegionAddress);
    }
    if !cx.inputs.replay_bindings {
        return Ok(());
    }
    let guest_bases: Vec<PushField> = cx
        .push
        .iter()
        .filter(|f| matches!(f, PushField::SlotGuestBase(_)))
        .copied()
        .collect();
    for f in guest_bases {
        cx.push.remove(&f);
        cx.push.insert(PushField::RootTableAddress);
    }
    cx.families.access.replay = true;
    // A replayed draw's plans state the variant.
    if drawn {
        cx.facts.insert(RuntimeFact::ReplayBindings);
    }
    Ok(())
}

/// The `DispatchRegion` value (`array<uvec4, 2>`, the layout) of a device-replay variant's
/// dispatch: the 32-byte region record at `ReplayRegionAddress`, which the replay's preprocess
/// wrote for the dispatch's command and region (its first threadgroup, the command's packed
/// threadgroup size, and the command's grid in threads); or a flat variant's, the record a
/// preprocess wrote for its dispatch, of which the variant reads the grid. Routed here by
/// `EmitCx::push_field` for the `DispatchRegion` of either.
pub fn replay_region(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<SValue> {
    let address = cx.push_field(fb, PushField::ReplayRegionAddress)?;
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let uvec4 = fb.module().ty(SType::Vector { elem: u32_, len: 4 })?;
    let two = fb.module().constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(2),
    })?;
    let region = fb.module().ty(SType::Array {
        elem: uvec4,
        len: two,
        stride: Some(16),
    })?;
    let place = fb.device_place(address, region)?;
    fb.load(
        place,
        crate::spirv::ir::MemAccess {
            aligned: Some(16),
            ..crate::spirv::ir::MemAccess::default()
        },
    )
}

/// The Metal address of `slot` in a device-replay variant, in `SlotGuestBase`'s shape (a `uvec2`,
/// low word first): its root entry's `guest + bind_offset`, which is the bound buffer's
/// `gpu_address + offset` the field would hold, 0 for an unbound slot (a zero entry), as the field
/// is. Routed here by `EmitCx::push_field` for a `SlotGuestBase` of a replay variant.
pub fn replay_guest_base(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    slot: SlotRef,
) -> BResult<SValue> {
    let table = cx.push_field(fb, PushField::RootTableAddress)?;
    let root = crate::memory::roots::k64(fb, u64::from(slot_root(slot.index)))?;
    let guest = crate::memory::roots::entry_field(fb, table, root, crate::memory::roots::GUEST)?;
    let bind =
        crate::memory::roots::entry_field(fb, table, root, crate::memory::roots::BIND_OFFSET)?;
    let base = fb.binary(crate::spirv::ir::BinOp::IAdd, guest, bind)?;
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let uvec2 = fb.module().ty(SType::Vector { elem: u32_, len: 2 })?;
    fb.convert(crate::spirv::ir::ConvOp::Bitcast, base, uvec2)
}

/// A storage slot the per-stage limit fallback moves to its device address.
#[derive(Clone, Copy)]
enum Moved {
    /// Memory-plan slot `.0`, with its chosen window.
    Accessed(usize, Window),
    /// The `.0`th source buffer of its own.
    Own(usize),
}

/// The device-address lowering of a slot with `window`: the window's start fixes how the `u32`
/// pointer offset maps back to a 64-bit displacement from the bind offset, which is exact for every
/// window of at most 2³² bytes. A `Bounded` window longer than that holds constant offsets the
/// `u32` offset cannot tell apart; it needs a wide static pointer, which is not implemented.
fn psb_lowering(window: Window) -> Result<SlotLowering, Error> {
    match window {
        Window::Whole => Ok(SlotLowering::Psb { start: 0 }),
        Window::Bounded { start, len } if len <= 1u64 << 32 => Ok(SlotLowering::Psb { start }),
        Window::Bounded { .. } => Err(refuse()),
    }
}

/// Accept the push placement: every field placed in the `ArgumentTable` is read from the table's
/// uniform buffer ([`table`]), whose descriptor [`finish_slots`] counted from the same field set,
/// and the table fits `maxUniformBufferRange`.
pub fn accept_placement(cx: &SelectCx<'_>, placement: &PushPlacement) -> Result<(), Error> {
    let spilled = placement
        .fields
        .iter()
        .any(|f| matches!(f.loc, PushLoc::ArgumentTable { .. }));
    if spilled != cx.families.access.argument_table {
        return Err(
            cx.invariant("the push placement disagrees with the counted ArgumentTable descriptor")
        );
    }
    let range = cx.profile.fields().limits.max_uniform_buffer_range;
    if placement.argument_table_size > range {
        return Err(Unsupported::needs(
            Feature::DescriptorCount,
            Capability::Limit {
                limit: VkLimit::MaxUniformBufferRange,
                required: u64::from(placement.argument_table_size),
                available: u64::from(range),
            },
        )
        .into());
    }
    Ok(())
}

/// The value of push field `field` placed in the stage's `ArgumentTable` (routed here by
/// `EmitCx::push_field` for every field the push block does not hold).
pub fn table_field(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    field: PushField,
) -> BResult<SValue> {
    let read = match &cx.decl.families.access.table {
        Some(t) => table::read(fb, t, field)?,
        None => None,
    };
    match read {
        Some(v) => Ok(v),
        None => Err(cx.fail(format!(
            "a strategy reads the unplaced push field {field:?}"
        ))),
    }
}

/// Declare the access family's module-scope objects: the view of every accessed slot.
pub fn declare(
    mb: &mut ModuleBuilder,
    plan: &Plan,
    _modes: &mut Vec<ExecMode>,
) -> BResult<Declared> {
    let mut d = Declared {
        local: local::declare(mb, plan)?,
        table: table::declare(mb, &plan.push)?,
        blob: match plan.families.access.blob {
            Some(_) => Some(crate::memory::blob::declare(mb, plan.stage)?),
            None => None,
        },
        pulled: pulled::declare(mb, &plan.families.access)?,
        handle_table: handle::declare_table(mb, plan)?,
        sampler_table: handle::declare_sampler_table(mb, plan)?,
        resolve: match dynamic::defines_search(&plan.families.access.dynamic) {
            true => {
                let wide = dynamic::wide_type(mb)?;
                Some(crate::memory::roots::define_resolve(mb, wide)?)
            }
            false => None,
        },
        tile: tile::declare(mb, &plan.families.access)?,
        function_tables: function_table::declare(mb, plan)?,
        status: absent::declare(mb, plan.stage, &plan.families.access)?,
        ..Declared::default()
    };
    for (i, s) in plan.slots.iter().enumerate() {
        let lowering = plan.families.access.slots.get(i).copied().flatten();
        let view = match (s, lowering) {
            (Some(c), Some(SlotLowering::View(kind))) => Some(views::declare(mb, c.slot, kind)?),
            _ => None,
        };
        d.views.push(view);
        d.at_bind.push(match (s, view.map(|v| v.kind)) {
            (Some(c), Some(ViewKind::UniformWords { .. } | ViewKind::UniformVec4 { .. })) => {
                Some(range_at_bind_offset(mb, c.slot)?)
            }
            _ => None,
        });
        let units = plan
            .families
            .access
            .units
            .get(&i)
            .copied()
            .unwrap_or(Units::NONE);
        d.units.push(match (s, view) {
            (Some(c), Some(view)) => views::UnitViews::declare(mb, c.slot, view, units)?,
            _ => views::UnitViews::NONE,
        });
    }
    Ok(d)
}

/// `v` as a sign-extended integer constant.
fn const_index(cx: &SelectCx<'_>, v: Value) -> Option<i64> {
    let Value::Const(c) = v else { return None };
    let k = cx.entry.module.consts().get(c)?;
    match (&k.value, cx.entry.module.types().get(k.ty)?) {
        (Const::Int(u), Type::Int(w)) => {
            let w = u32::from(*w).clamp(1, 64);
            let shift = 64 - w;
            Some(((*u << shift) as i64) >> shift)
        }
        (Const::Null, Type::Int(_)) => Some(0),
        _ => None,
    }
}

/// The value id of an instruction or parameter operand, or of a null device or constant pointer
/// constant (`null`, `undef`, or `poison`: the null fat pointer `uvec2(0, 0)`,
/// `pointer::null_operand`); any other constant is refused.
fn operand(cx: &SelectCx<'_>, v: Value) -> Result<ValueId, Error> {
    if matches!(v, Value::Const(_)) && !pointer::null_constant(cx, v) {
        return Err(refuse());
    }
    cx.values
        .id(v)
        .ok_or_else(|| cx.invariant("an operand out of the value range"))
}

fn index_form(cx: &SelectCx<'_>, v: Value) -> Result<IndexForm, Error> {
    let ty = cx
        .value_ty(v)
        .and_then(|t| cx.entry.module.types().get(t))
        .ok_or_else(|| cx.invariant("a GEP index without a type"))?;
    Ok(match ty {
        Type::Int(1) => IndexForm::Bool,
        Type::Int(bits @ 2..=31) => IndexForm::Narrow { bits: *bits },
        Type::Int(32) => IndexForm::Word,
        Type::Int(33..=64) => IndexForm::Wide,
        _ => return Err(refuse()),
    })
}

fn select_gep(
    cx: &mut SelectCx<'_>,
    at: InstId,
    src_ty: crate::air::ir::TypeId,
    base: Value,
    indices: &[Value],
) -> Result<Gep, Error> {
    statics(cx, Value::Inst(at))?;
    statics(cx, base)?;
    let base = operand(cx, base)?;
    let (konst, terms) = gep_offset(cx, src_ty, indices)?;
    Ok(Gep { base, konst, terms })
}

/// The offset a GEP adds to its base: its constant part and its dynamic terms, from the AIR
/// datalayout, shared by buffer and arena pointers. The offset is Metal's wrapping 64-bit sum
/// taken modulo 2^32: the low words of [`gep_offset_wide`].
fn gep_offset(
    cx: &mut SelectCx<'_>,
    src_ty: crate::air::ir::TypeId,
    indices: &[Value],
) -> Result<(u32, GepTerms), Error> {
    let (konst, terms) = gep_offset_wide(cx, src_ty, indices)?;
    Ok((
        konst as u32,
        terms
            .into_iter()
            .map(|(v, stride, form)| (v, stride as u32, form))
            .collect(),
    ))
}

/// The dynamic terms of a 64-bit GEP offset: `(index, byte stride mod 2^64, index form)`.
pub type WideGepTerms = Vec<(ValueId, u64, IndexForm)>;

/// The offset a GEP adds to its base as Metal's wrapping 64-bit sum (a wide pointer's offset): its
/// constant part and its dynamic terms, from the AIR datalayout. Indices are read through the
/// index-extension rule.
fn gep_offset_wide(
    cx: &mut SelectCx<'_>,
    src_ty: crate::air::ir::TypeId,
    indices: &[Value],
) -> Result<(u64, WideGepTerms), Error> {
    let (konst, terms) = gep_offset_indexed(cx, src_ty, indices)?;
    Ok((
        konst,
        terms
            .into_iter()
            .map(|t| (t.index, t.stride, t.form))
            .collect(),
    ))
}

/// What a GEP's dynamic index steps over: the elements of its pointer (its first index), the
/// `len` elements of an array, or a vector's lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Indexed {
    Pointer,
    Array(u64),
    Vector,
}

/// A dynamic term of a GEP offset ([`gep_offset_indexed`]): the GEP's index operand, that index
/// read through the index-extension rule, its byte stride mod 2^64, and what it steps over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IndexedTerm {
    operand: Value,
    index: ValueId,
    form: IndexForm,
    stride: u64,
    over: Indexed,
}

/// [`gep_offset_wide`]'s constant part and dynamic terms, each with what its index steps over, in
/// the GEP's index order.
fn gep_offset_indexed(
    cx: &mut SelectCx<'_>,
    src_ty: crate::air::ir::TypeId,
    indices: &[Value],
) -> Result<(u64, Vec<IndexedTerm>), Error> {
    let layouts = &cx.memory.layouts;
    let types = cx.entry.module.types();
    let mut dynamic = Vec::new();
    let mut konst: u64 = 0;
    let mut terms = Vec::new();
    let mut cur = src_ty;
    for (n, &i) in indices.iter().enumerate() {
        let (scale, over) = if n == 0 {
            (layouts.alloc_size(src_ty), Indexed::Pointer)
        } else {
            match types.get(cur) {
                Some(Type::Struct { fields, .. }) => {
                    let k = const_index(cx, i).ok_or_else(refuse)?;
                    let k = usize::try_from(k).map_err(|_| refuse())?;
                    let off = layouts
                        .field_offsets(cur)
                        .and_then(|o| o.get(k).copied())
                        .ok_or_else(refuse)?;
                    konst = konst.wrapping_add(off);
                    cur = *fields.get(k).ok_or_else(refuse)?;
                    continue;
                }
                Some(Type::Array { elem, len }) => {
                    cur = *elem;
                    (layouts.alloc_size(*elem), Indexed::Array(*len))
                }
                Some(Type::Vector { elem, .. }) => {
                    cur = *elem;
                    (layouts.alloc_size(*elem), Indexed::Vector)
                }
                _ => (None, Indexed::Vector),
            }
        };
        let scale = scale.ok_or_else(refuse)?;
        match const_index(cx, i) {
            Some(k) => konst = konst.wrapping_add((k as u64).wrapping_mul(scale)),
            None if scale == 0 => {}
            None => dynamic.push((i, scale, over)),
        }
    }
    for (i, stride, over) in dynamic {
        // An index extension is read as its narrow operand; any other index as itself.
        let (v, form) = match index::narrowed(cx, i) {
            Some(n) => n,
            None => (i, index_form(cx, i)?),
        };
        terms.push(IndexedTerm {
            operand: i,
            index: operand(cx, v)?,
            form,
            stride,
            over,
        });
    }
    Ok((konst, terms))
}

fn select_access(
    cx: &mut SelectCx<'_>,
    ptr: Value,
    value: Option<Value>,
    ty: crate::air::ir::TypeId,
    align: Option<crate::air::ir::Align>,
    m: &mut Meter,
) -> Result<Access, Error> {
    let (targets, null, blob_check) = blob_roots(cx, ptr, value.is_some(), m)?;
    let align = align.map_or(1, |a| a.bytes());
    let leaf = leaf::leaf_of(cx.entry, cx.profile, &cx.memory.layouts, ty, align)?;
    access_with_leaf(cx, ptr, value, leaf, targets, null, blob_check)
}

/// The candidate targets of static-root pointer `ptr` ([`buffer_roots`]), accessed as a write
/// when `write`, and the constant blob's containment check when `ptr` may point into the blob,
/// which the function then reads. The blob is constant: a write through such a pointer is
/// refused. Loads, stores, and memory intrinsics' endpoints (`copy::endpoint`) share it.
fn blob_roots(
    cx: &mut SelectCx<'_>,
    ptr: Value,
    write: bool,
    m: &mut Meter,
) -> Result<(Vec<Target>, bool, Option<Check>), Error> {
    let (targets, null, in_blob) = buffer_roots(cx, ptr, m)?;
    if in_blob && write {
        return Err(refuse());
    }
    let blob_check = if in_blob {
        blob::use_blob(cx)?;
        Some(contain::encoding(cx.profile))
    } else {
        None
    };
    Ok((targets, null, blob_check))
}

/// Give each of `targets` the profile's containment check and read its slot's `SlotBindOffset`;
/// a written target is a storage view. Every static-root access places its targets here
/// ([`access_with_leaf`], `copy::endpoint`).
fn place_targets(cx: &mut SelectCx<'_>, targets: &mut [Target], write: bool) -> Result<(), Error> {
    let check = contain::encoding(cx.profile);
    for t in targets {
        t.check = check;
        let plan = cx
            .memory
            .slots
            .get(t.slot)
            .ok_or_else(|| cx.invariant("a target names a missing slot"))?;
        if write && view_kind(cx.profile, plan) != ViewKind::Storage {
            return Err(cx.invariant("a write through a uniform view"));
        }
        cx.push.insert(PushField::SlotBindOffset(plan.slot));
    }
    Ok(())
}

/// Give each slot of `targets` the unit views that read or write `leaf` (a store when `write`)
/// in whole units ([`leaf::leaf_units`]): a storage slot the narrow views the profile can access,
/// and any slot a `uvec4` view.
fn note_units(
    cx: &mut SelectCx<'_>,
    targets: &[Target],
    leaf: &Leaf,
    write: bool,
) -> Result<(), Error> {
    let units = leaf::leaf_units(leaf, write).intersect(Units::storage_on(cx.profile));
    if units.is_empty() {
        return Ok(());
    }
    for t in targets {
        let plan = cx
            .memory
            .slots
            .get(t.slot)
            .ok_or_else(|| cx.invariant("a target names a missing slot"))?;
        let units = match view_kind(cx.profile, plan) {
            ViewKind::Storage => units,
            // A uniform view has no narrow view.
            ViewKind::UniformWords { .. } | ViewKind::UniformVec4 { .. } => Units {
                vec4: units.vec4,
                ..Units::NONE
            },
        };
        if units.is_empty() {
            continue;
        }
        let entry = cx
            .families
            .access
            .units
            .entry(t.slot)
            .or_insert(Units::NONE);
        *entry = entry.union(units);
    }
    Ok(())
}

/// The leaf access `leaf` through static-root pointer `ptr` with candidate `targets` (from
/// [`buffer_roots`]), placed by [`place_targets`].
fn access_with_leaf(
    cx: &mut SelectCx<'_>,
    ptr: Value,
    value: Option<Value>,
    leaf: Leaf,
    mut targets: Vec<Target>,
    null: bool,
    blob_check: Option<Check>,
) -> Result<Access, Error> {
    place_targets(cx, &mut targets, value.is_some())?;
    note_units(cx, &targets, &leaf, value.is_some())?;
    let ptr = operand(cx, ptr)?;
    let value = match value {
        None => None,
        Some(v) => {
            // A stored constant must be one the arith family materializes; any other value is
            // an instruction result or parameter its own family selected.
            if matches!(v, Value::Const(_)) {
                if cx.constant(v).is_none() {
                    return Err(not_lowered(LowerFamily::Arith));
                }
                Some(
                    cx.values
                        .id(v)
                        .ok_or_else(|| cx.invariant("a stored value out of range"))?,
                )
            } else {
                Some(operand(cx, v)?)
            }
        }
    };
    Ok(Access {
        ptr,
        value,
        leaf,
        targets,
        null,
        coherence: None,
        volatile: false,
        blob: blob_check,
    })
}

/// A `air.{load,store}.{device,system}_coherent` access through a static-root buffer pointer: the
/// plain leaf access with `NonPrivatePointer` and `MakePointerVisible` (a load) or
/// `MakePointerAvailable` (a store) at `Device` scope, which selects the Vulkan memory model, on a
/// profile with that model and its device scope; elsewhere the plain access through `Coherent`
/// resources under `GLSL450` (`sync::memory_model::coherent_access`). System coherence is `Device`
/// scope, the widest of one logical device (the reading of Metal's all-devices scope). A coherent
/// access of another memory is not implemented.
fn select_coherent(
    cx: &mut SelectCx<'_>,
    inst: &Inst,
    call: &crate::air::ir::Call,
    c: &crate::air::intrinsics::Coherent,
    m: &mut Meter,
) -> Result<AccessStrategy, Error> {
    use crate::air::intrinsics::AccessKind;
    let arg = |i: usize| -> Result<Value, Error> {
        call.args
            .get(i)
            .copied()
            .ok_or_else(|| cx.invariant("a coherent access without its operands"))
    };
    let (ptr, value, ty) = match c.kind {
        AccessKind::Load => (arg(0)?, None, inst.ty),
        AccessKind::Store => {
            let v = arg(0)?;
            let ty = cx
                .value_ty(v)
                .ok_or_else(|| cx.invariant("a coherent store's value without a type"))?;
            (arg(1)?, Some(v), ty)
        }
    };
    if !matches!(c.space, AddrSpace::Device | AddrSpace::Constant) {
        return Err(refuse());
    }
    // The intrinsic states no alignment: the value's natural (ABI) one.
    let align = cx
        .memory
        .layouts
        .abi_align(ty)
        .and_then(crate::air::ir::Align::from_bytes);
    let mut a = select_access(cx, ptr, value, ty, align, m)?;
    a.coherence = Some(crate::lower::sync::memory_model::coherent_access(cx));
    a.volatile = c.volatile;
    Ok(match c.kind {
        AccessKind::Load => AccessStrategy::Load(a),
        AccessKind::Store => AccessStrategy::Store(a),
    })
}

fn is_ptr_ty(cx: &SelectCx<'_>, ty: crate::air::ir::TypeId) -> bool {
    matches!(cx.entry.module.types().get(ty), Some(Type::Ptr { .. }))
}

/// Select the strategy of access instruction `inst`, metering any walk it takes in `m`.
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    m: &mut Meter,
) -> Result<AccessStrategy, Error> {
    if let Some(plain) = non_volatile(inst) {
        let s = volatile_form(select(cx, at, &plain, m)?)?;
        note_ordered(cx, &s);
        return Ok(s);
    }
    if let Some(s) = function_table::select(cx, inst)? {
        return Ok(AccessStrategy::FunctionTable(s));
    }
    if let Some(s) = absent::select(cx, at, inst)? {
        return Ok(AccessStrategy::AbsentRead(s));
    }
    if let Some(s) = imageblock::select(cx, inst)? {
        return Ok(AccessStrategy::Imageblock(s));
    }
    if let Some(s) = encode::select(cx, at, inst, m)? {
        return Ok(AccessStrategy::Encode(s));
    }
    if let Some(s) = cell::select(cx, at, inst)? {
        return Ok(AccessStrategy::Cell(s));
    }
    if let Some(s) = handle::select(cx, at, inst, m)? {
        return Ok(AccessStrategy::Handle(s));
    }
    if let Some(s) = pointer::select(cx, at, inst, m)? {
        return Ok(AccessStrategy::Value(s));
    }
    if let Some(s) = dynamic::select(cx, at, inst, m)? {
        return Ok(AccessStrategy::Dynamic(s));
    }
    if let Some(s) = local::select(cx, at, inst, m)? {
        return Ok(AccessStrategy::Local(s));
    }
    if let Some(s) = blob::select(cx, at, inst)? {
        return Ok(AccessStrategy::Blob(s));
    }
    if let Some(a) = atomic::select(cx, at, inst, m)? {
        let s = AccessStrategy::Atomic(a);
        note_ordered(cx, &s);
        return Ok(s);
    }
    if let InstKind::Call(call) = &inst.kind {
        if let crate::air::ir::Callee::Air {
            op: crate::air::intrinsics::AirIntrinsic::Coherent(c),
            ..
        } = &call.callee
        {
            let s = select_coherent(cx, inst, call, c, m)?;
            note_ordered(cx, &s);
            return Ok(s);
        }
    }
    if let Some(op) = copy::select(cx, at, inst, m)? {
        let s = AccessStrategy::Memory(op);
        note_ordered(cx, &s);
        return Ok(s);
    }
    if let Some(s) = buffer_table::select(cx, at, inst)? {
        return Ok(AccessStrategy::Table(s));
    }
    match &inst.kind {
        InstKind::Load {
            ptr,
            align,
            volatile: false,
            atomic: None,
            ..
        } if !is_ptr_ty(cx, inst.ty) => Ok(AccessStrategy::Load(select_access(
            cx, *ptr, None, inst.ty, *align, m,
        )?)),
        InstKind::Store {
            val,
            ptr,
            align,
            volatile: false,
            atomic: None,
        } => {
            let ty = cx
                .value_ty(*val)
                .ok_or_else(|| cx.invariant("a stored value without a type"))?;
            if is_ptr_ty(cx, ty) {
                return Err(refuse());
            }
            Ok(AccessStrategy::Store(select_access(
                cx,
                *ptr,
                Some(*val),
                ty,
                *align,
                m,
            )?))
        }
        InstKind::Gep {
            src_ty,
            base,
            indices,
            ..
        } => Ok(AccessStrategy::Pointer(PtrOp::Gep(select_gep(
            cx, at, *src_ty, *base, indices,
        )?))),
        InstKind::Cast {
            op: CastOp::Bitcast | CastOp::AddrSpaceCast,
            v,
        }
        | InstKind::Freeze(v)
            if is_ptr_ty(cx, inst.ty) =>
        {
            statics(cx, Value::Inst(at))?;
            statics(cx, *v)?;
            Ok(AccessStrategy::Pointer(PtrOp::Copy {
                of: operand(cx, *v)?,
            }))
        }
        InstKind::Select { c, t, f, .. } if is_ptr_ty(cx, inst.ty) => {
            statics(cx, Value::Inst(at))?;
            let t = select_side(cx, *t)?;
            let f = select_side(cx, *f)?;
            let c = match c {
                Value::Const(_) => return Err(refuse()),
                c => operand(cx, *c)?,
            };
            Ok(AccessStrategy::Pointer(PtrOp::Select { c, t, f }))
        }
        _ => Err(refuse()),
    }
}

/// Record the static slots a volatile, coherent, or atomic access `s` reaches
/// (`AccessPlan::ordered`). An atomic's words are reached through a storage-view pointer, which a
/// uniform view has none of, and an atomic load reads words other invocations may write.
fn note_ordered(cx: &mut SelectCx<'_>, s: &AccessStrategy) {
    if let AccessStrategy::Atomic(atomic::AtomicAccess {
        mem: atomic::AtomicMem::Buffer { targets, .. },
        ..
    }) = s
    {
        let slots: Vec<usize> = targets.iter().map(|t| t.slot).collect();
        cx.families.access.ordered.extend(slots);
        return;
    }
    let ordered = |a: &Access| a.volatile || a.coherence.is_some();
    let accesses: Vec<&Access> = match s {
        AccessStrategy::Load(a) | AccessStrategy::Store(a) => vec![a],
        AccessStrategy::Handle(
            handle::HandleStrategy::Lookup(a) | handle::HandleStrategy::SamplerLookup(a),
        ) => a.static_access().into_iter().collect(),
        AccessStrategy::Memory(m) => {
            let mut v = Vec::new();
            if let copy::Endpoint::Buffer(a) = &m.dst {
                v.push(a);
            }
            if let copy::Source::Copy {
                from: copy::Endpoint::Buffer(a),
                ..
            } = &m.src
            {
                v.push(a);
            }
            v
        }
        _ => Vec::new(),
    };
    for a in accesses.into_iter().filter(|a| ordered(a)) {
        cx.families
            .access
            .ordered
            .extend(a.targets.iter().map(|t| t.slot));
    }
}

/// The same instruction as the volatile non-atomic load or store `inst`, without `volatile`;
/// `None` for any other instruction (an atomic one is the atomic family's whatever its flag).
fn non_volatile(inst: &Inst) -> Option<Inst> {
    if !matches!(
        inst.kind,
        InstKind::Load {
            volatile: true,
            atomic: None,
            ..
        } | InstKind::Store {
            volatile: true,
            atomic: None,
            ..
        }
    ) {
        return None;
    }
    let mut plain = inst.clone();
    if let InstKind::Load { volatile, .. } | InstKind::Store { volatile, .. } = &mut plain.kind {
        *volatile = false;
    }
    Some(plain)
}

/// The volatile form of strategy `s`, selected for a volatile load or store as if it were plain:
/// the same accesses, every word of which carries `Volatile` (a heap texture handle's resource-id
/// words included). A load of read-only memory (the constant blob, a typed constant object) is the
/// plain load: its words cannot change while the dispatch runs. Any other strategy (a handle-array
/// element or buffer-table element load, which reads no memory at the access) is refused.
fn volatile_form(s: AccessStrategy) -> Result<AccessStrategy, Error> {
    Ok(match s {
        AccessStrategy::Load(a) => AccessStrategy::Load(Access {
            volatile: true,
            ..a
        }),
        AccessStrategy::Store(a) => AccessStrategy::Store(Access {
            volatile: true,
            ..a
        }),
        AccessStrategy::Local(l) => AccessStrategy::Local(l.volatile()?),
        AccessStrategy::Dynamic(d) => AccessStrategy::Dynamic(d.volatile()?),
        AccessStrategy::Cell(c) => AccessStrategy::Cell(c.volatile()),
        AccessStrategy::Value(p) => AccessStrategy::Value(p.volatile()?),
        AccessStrategy::Blob(b @ blob::BlobStrategy::Load(_)) => AccessStrategy::Blob(b),
        AccessStrategy::Handle(handle::HandleStrategy::Lookup(a)) => {
            AccessStrategy::Handle(handle::HandleStrategy::Lookup(Box::new(a.volatile())))
        }
        AccessStrategy::Handle(handle::HandleStrategy::SamplerLookup(a)) => AccessStrategy::Handle(
            handle::HandleStrategy::SamplerLookup(Box::new(a.volatile())),
        ),
        _ => return Err(refuse()),
    })
}
