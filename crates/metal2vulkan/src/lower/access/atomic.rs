//! Memory atomics: `air.atomic.{global,local,private}.*` and LLVM `atomicrmw`, `cmpxchg`, and
//! atomic `load`/`store`.
//!
//! **Where.** The memory is the pointer's (memory analysis's representation, not the name): a
//! static-root buffer pointer reaches buffer slots (a descriptor view or a device-address slot,
//! several candidates dispatched by a switch-at-use on the root, like a plain access); a wide
//! (dynamic-root) pointer reaches the allocation its root entry names, at the entry's device
//! address (one code path for every candidate, as for a wide leaf access); a threadgroup-arena
//! offset reaches the arena. An `air.atomic.global` names device memory and `air.atomic.local`
//! threadgroup memory; a pointer of the other kind is refused. Every one of these memories is a
//! `u32` word array, so a 32-bit atomic is one SPIR-V atomic on the word `offset >> 2` (Metal
//! atomic objects are naturally aligned; an LLVM atomic whose alignment is below its size is
//! refused). Typed objects never reach an atomic: a typed alloca or constant `Private` global has
//! no intrinsic or atomic use by construction, and an LLVM atomic through a private pointer is
//! refused.
//!
//! **Thread memory**. MSL 4.1's `air.atomic.private` names a `thread` object, a word of the private
//! arena (its alloca has an intrinsic use, so it is never a typed object). Only its own thread ever
//! reaches that word: no other thread can read what it writes, so none can synchronize with it, and
//! its order and `mem_flags` order nothing another thread observes (Apple's compiler passes
//! `thread_scope_thread`). So it is the plain read-modify-write of the arena leaf: the load, the
//! operation (the integer operations, `xchg`, `f32` `add` and `sub`), the store, and the old value;
//! a compare-exchange stores the desired value when the old one equals the comparator and the old
//! one otherwise, and writes the old one back through `expected` as every compare-exchange does.
//!
//! **Operations.** `add`, `sub`, `min`/`max` (signed or unsigned by the name's sign token, or
//! LLVM's `min`/`max`/`umin`/`umax`), `and`, `or`, `xor`, and `xchg` are `OpAtomic{IAdd, ISub,
//! SMin/UMin, SMax/UMax, And, Or, Xor, Exchange}`; `load` and `store` are
//! `OpAtomicLoad`/`OpAtomicStore`. `air`'s `cmpxchg.weak` reads the comparator through its
//! `expected` thread pointer (a private-arena leaf), is `OpAtomicCompareExchange`, writes the old
//! value back through `expected` (on success it equals the comparator, so the write is C11's
//! failure write-back or a no-op), and returns the old value. LLVM's `cmpxchg` (weak or strong:
//! SPIR-V's exchange is strong, which is a valid weak one) returns `{old, old == comparator}`.
//! `f32` `xchg`, `load`, and `store` move the bit pattern.
//!
//! **Typed atomics through device addresses**. A descriptor view is one `u32` array and the arena
//! one `u32` array, so an atomic on a `u64` or an `f32` word needs a typed alias only a device
//! address gives: a `PhysicalStorageBuffer` pointer to the type at the word's address. Such an
//! atomic moves every slot it reaches to its device address (`AccessPlan::typed_slots`, read by
//! `finish_slots`), on profiles with device addresses; through a wide pointer it is at a device
//! address already. Every device-address atomic takes its pointer from one function (`device_ptr`).
//! A 64-bit atomic is then one `u64` SPIR-V atomic (`Int64Atomics`); `f32` `add` and `sub` (the add
//! of the negated operand, which IEEE 754 defines as the same operation) are `OpAtomicFAddEXT`
//! where the profile has `shaderBufferFloat32AtomicAdd`, instead of an execution-bounded loop that
//! contention could exhaust.
//!
//! **Narrow atomics** (`narrow`): an LLVM atomic on `i8` or `i16` operates on its field of the
//! containing `u32` word, in any of these memories: `load`, `and`, `or`, and `xor` are one word
//! atomic each, every other operation a compare-and-swap loop on the word.
//!
//! **Compare-and-swap loops.** Any other operation SPIR-V has no word instruction for is a
//! compare-and-swap loop on the word ([`CasOp`]): `f32` `add` and `sub` without the native path,
//! `f32` `min` and `max` (always: `NMin`/`NMax`, which ignore a NaN operand as LLVM's `fmin`/`fmax`
//! do, while `OpAtomicFMinEXT` leaves NaN operands to the device), and LLVM's `nand`,
//! `uinc_wrap`, and `udec_wrap` (on `u32` or `u64` words). The loop is lock-free (a failed exchange
//! means another invocation's succeeded) but not statically bounded; it opens a builder loop.
//!
//! **Scope and order.** The scope is the memory's: `Device` for buffers, `Workgroup` for the
//! arena (an LLVM `syncscope("singlethread")` atomic takes the same scope, which is only stronger,
//! as is an MSL 4.1 device atomic of a narrower `thread_scope`). A relaxed order (LLVM `monotonic`
//! or `unordered`) has no semantics; another order names memories at the Vulkan order (`seq_cst` is
//! `AcquireRelease`; a load keeps only the acquire part and a store only the release part), with
//! `MakeAvailable`/`MakeVisible` under the Vulkan memory model (`sync::barrier::semantics`). The
//! memories are the call's Metal `mem_flags` when the AIR 2.9 form carries them (`UniformMemory`
//! for `mem_device`, `WorkgroupMemory` for `mem_threadgroup` and `mem_threadgroup_imageblock` in a
//! kernel, `ImageMemory` for `mem_texture`; a fragment function's threadgroup memory is its tile,
//! device memory in the pass's buffer, so there those two order `UniformMemory`, and nothing where
//! the function has no tile; a vertex function has no threadgroup memory, and only an object or
//! mesh function has `mem_object_data`'s, its payload, so those name nothing elsewhere), and the
//! atomic's own memory (`UniformMemory` or `WorkgroupMemory`) for an AIR 2.8 or LLVM atomic. A
//! `Device`-scope atomic is reported to the memory-model family. Metal's `volatile` qualifier and
//! LLVM's `volatile` add nothing: an atomic is never elided or merged. A relaxed load that an exit
//! of a loop holding it depends on (a spin's, [`spin`]) acquires its own memory: with robust buffer
//! access, Mesa's RADV and ANV never showed a relaxed one a store that arrived after its first
//! read.
//!
//! **Containment** is the plain access's (`contain::atomic_encoding`): robust descriptors emit no
//! check, except on a driver whose robustness leaves buffer atomics uncontained; an in-shader check
//! guards a view's atomic with `if_(inb)` (its result is 0 out of bounds, where Metal's is
//! unmeasured); a device-address atomic out of bounds (a static slot's or a wide pointer's, checked
//! against the root entry's size) operates on the write-only sink and yields 0; an arena atomic
//! sits in `if_(inb)` against the arena's exact size. A wide pointer's atomic through root 0 (null
//! or a resident-set miss) is counted under isolation, as a wide leaf access is. A wide atomic's
//! address is naturally aligned (`(bind_offset + offset) & ~(n - 1)`), as Metal atomic objects are.
//!
//! **Refused.** 64-bit atomics need `shaderBufferInt64Atomics` / `shaderSharedInt64Atomics`
//! (`Unsupported{Int64Atomics}`) and, on a buffer, device addresses (the profile's addressed
//! needs); a 64-bit threadgroup atomic needs a `u64` view of the arena, which is not declared
//! (`Unsupported{Lowering(Access), NotImplemented}`), as are atomics of other types (pointers,
//! `half`), under-aligned LLVM atomics, and LLVM atomics through private pointers.

use std::borrow::Cow;

use crate::air::intrinsics::{
    AirIntrinsic, Atomic as AirAtomic, AtomicOp as AirOp, AtomicSpace, Elem, MemFlags,
    MemoryOrder as AirOrder, Sign,
};
use crate::air::ir::{
    Align, AtomicRmwOp, Callee, Inst, InstId, InstKind, Ordering, Type, TypeId, Value, ValueId,
};
use crate::budget::Meter;
use crate::device::profile::BufferAccess;
use crate::device::profile::Containment;
use crate::device::requirements::VkFeature;
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::lower::sync::{barrier, memory_model};
use crate::memory::arena::{Arena, ArenaMem};
use crate::memory::contain::{self, Check};
use crate::memory::leaf::{k32, WordOps};
use crate::memory::provenance::{PrivForm, Repr};
use crate::memory::views::{self, ViewKind};
use crate::pipeline::sym::{PushField, Stage};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{TypeId as STypeId, Value as SValue};
use crate::spirv::ir::{
    AtomicOp, BinOp, ConstBits, Constant, ConvOp, GlslOp, MemAccess, MemoryOrder, Scope, Semantics,
    Type as SType, UnOp,
};

use super::emit::{prepare, Mem, Prepared, ViewMem};
use super::local::{self, LocalAccess, LocalPtr};
use super::{not_lowered, refuse, static_roots, view_kind, AccessPlan, BufferReach, Target};
use crate::error::LowerFamily;

mod narrow;
pub mod spin;

/// The read-modify-write a compare-and-swap loop applies to the old value `o` and the operand
/// `x`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CasOp {
    /// `f32`: `o + x`.
    FAdd,
    /// `f32`: `o - x`.
    FSub,
    /// `f32`: `NMin(o, x)`.
    FMin,
    /// `f32`: `NMax(o, x)`.
    FMax,
    /// `!(o & x)` (LLVM `nand`).
    Nand,
    /// `o >= x ? 0 : o + 1` (LLVM `uinc_wrap`).
    UIncWrap,
    /// `o == 0 || o > x ? x : o - 1` (LLVM `udec_wrap`).
    UDecWrap,
}

impl CasOp {
    fn is_float(self) -> bool {
        matches!(self, CasOp::FAdd | CasOp::FSub | CasOp::FMin | CasOp::FMax)
    }
}

/// What an atomic does to its word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomicKind {
    /// One read-modify-write instruction (integer operations, and `Exchange` of either type).
    Rmw(AtomicOp),
    Load,
    Store,
    /// A compare-exchange (integers).
    CompareExchange,
    /// A read-modify-write as a compare-and-swap loop.
    Cas(CasOp),
    /// `f32` `add` (or `sub`, as the add of the negated operand) as `OpAtomicFAddEXT` through a
    /// typed device address.
    FloatAdd {
        negate: bool,
    },
}

/// The memory an atomic operates on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AtomicMem {
    /// Static-root buffer slots.
    Buffer {
        ptr: ValueId,
        targets: Vec<Target>,
        null: bool,
    },
    /// The threadgroup arena.
    Threadgroup { ptr: LocalPtr, contained: bool },
    /// A wide (dynamic-root) buffer pointer: the device address its root entry names.
    Wide { ptr: ValueId, contained: bool },
    /// A word of the private arena (`air.atomic.private`): its leaf, read and written by its own
    /// thread alone.
    Private(LocalAccess),
}

/// The comparator of a compare-exchange, and the shape of its result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Compare {
    /// `air`'s `cmpxchg.weak`: read through the `expected` thread pointer, the old value written
    /// back through it; the result is the old value.
    Expected(LocalAccess),
    /// LLVM `cmpxchg`: an operand; the result is `{old, old == comparator}`.
    Value(ValueId),
}

/// One memory atomic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtomicAccess {
    pub mem: AtomicMem,
    pub kind: AtomicKind,
    /// The operand (every kind but `Load`; the desired value of a compare-exchange).
    pub value: Option<ValueId>,
    /// The value is an `f32`, moved as its bit pattern.
    pub float: bool,
    /// The value is 64 bits wide (a `u64` word through a typed device address).
    pub wide: bool,
    /// The value is an `i8` or `i16` (its width in bits), a field of its containing `u32` word
    /// (`narrow`).
    pub narrow: Option<u8>,
    /// The comparator of a compare-exchange.
    pub compare: Option<Compare>,
    /// The success ordering, and the failure ordering of a compare-exchange.
    pub order: MemoryOrder,
    pub failure: MemoryOrder,
    /// The memories the orderings apply to: an AIR 2.9 atomic's `mem_flags` without the memories
    /// the stage lacks, or `None` for the atomic's own memory.
    pub flags: Option<MemFlags>,
    /// A relaxed load an exit of a loop holding it depends on ([`spin`]): it acquires its own
    /// memory.
    pub spin: bool,
}

impl AtomicAccess {
    /// The census name (`access.atomic.<memory>.<kind>[<reach>][.checked]`) in the finished plan
    /// `slots`: `.checked` when an access is checked in the shader as emitted (a buffer target by
    /// [`AccessPlan::check`]: its view's encoding, or a device-address slot's own check), and a
    /// buffer atomic qualified by the reach of its targets ([`BufferReach`]).
    pub fn name(&self, slots: &AccessPlan) -> Cow<'static, str> {
        let (mem, reach, checked) = match &self.mem {
            AtomicMem::Buffer { targets, .. } => (
                "buffer",
                slots.reach(targets),
                targets.iter().any(|t| slots.check(t) == Check::InShader),
            ),
            AtomicMem::Threadgroup { contained, .. } => {
                ("threadgroup", BufferReach::NONE, *contained)
            }
            AtomicMem::Wide { contained, .. } => ("wide", BufferReach::NONE, *contained),
            AtomicMem::Private(a) => ("private", BufferReach::NONE, a.contained),
        };
        let kind = match self.kind {
            _ if self.narrow.is_some() && narrow::loops(self.kind) => "narrow_cas",
            _ if self.narrow.is_some() => "narrow",
            AtomicKind::Rmw(_) => "rmw",
            AtomicKind::Load if self.spin => "spin_load",
            AtomicKind::Load => "load",
            AtomicKind::Store => "store",
            AtomicKind::CompareExchange => "cmpxchg",
            AtomicKind::Cas(op) if op.is_float() => "float_cas",
            AtomicKind::Cas(_) => "cas",
            AtomicKind::FloatAdd { .. } => "float_add",
        };
        let checked = if checked { ".checked" } else { "" };
        format!("access.atomic.{mem}.{kind}{}{checked}", reach.qualifier()).into()
    }

    /// An upper estimate of the SPIR-V words one use emits: the pointer and containment of each
    /// candidate (as a plain access's), the atomic, a loop, and the write-back.
    pub fn words(&self) -> u32 {
        let op: u32 = match self.kind {
            // The field's shift and mask, and a loop around the word's exchange.
            _ if self.narrow.is_some() && self.opens_loop() => 256,
            _ if self.narrow.is_some() => 48,
            AtomicKind::Cas(_) => 176,
            AtomicKind::CompareExchange => 56,
            _ => 24,
        };
        let compare = match &self.compare {
            Some(Compare::Expected(e)) => e.leaf.cost().saturating_mul(2).saturating_add(160),
            Some(Compare::Value(_)) => 16,
            None => 0,
        };
        let per = match &self.mem {
            AtomicMem::Buffer { targets, .. } => {
                (targets.len() as u32).saturating_mul(op.saturating_add(240))
                    + 16u32.saturating_mul(targets.len() as u32 + 1)
            }
            AtomicMem::Threadgroup { .. } => op.saturating_add(96),
            // The root entry's reads, the check, the sink, and the miss counter.
            AtomicMem::Wide { .. } => op.saturating_add(320),
            // The leaf's load and store around the operation.
            AtomicMem::Private(a) => a.leaf.cost().saturating_mul(2).saturating_add(op),
        };
        per.saturating_add(compare).saturating_add(16)
    }

    /// Whether the atomic opens a builder loop (a thread atomic never does: its operation is
    /// applied once).
    pub fn opens_loop(&self) -> bool {
        match self.narrow {
            _ if matches!(self.mem, AtomicMem::Private(_)) => false,
            Some(_) => narrow::loops(self.kind),
            None => matches!(self.kind, AtomicKind::Cas(_)),
        }
    }
}

fn order(o: AirOrder) -> MemoryOrder {
    match o {
        AirOrder::Relaxed => MemoryOrder::Relaxed,
        AirOrder::Acquire => MemoryOrder::Acquire,
        AirOrder::Release => MemoryOrder::Release,
        AirOrder::AcqRel | AirOrder::SeqCst => MemoryOrder::AcquireRelease,
    }
}

/// An LLVM ordering at the Vulkan order (`unordered` and `monotonic` are relaxed).
fn llvm_order(o: Ordering) -> MemoryOrder {
    match o {
        Ordering::Unordered | Ordering::Monotonic => MemoryOrder::Relaxed,
        Ordering::Acquire => MemoryOrder::Acquire,
        Ordering::Release => MemoryOrder::Release,
        Ordering::AcqRel | Ordering::SeqCst => MemoryOrder::AcquireRelease,
    }
}

/// The operation of an integer read-modify-write (or any `xchg`).
fn int_op(op: AirOp, sign: Option<Sign>) -> Option<AtomicOp> {
    let signed = sign == Some(Sign::Signed);
    Some(match op {
        AirOp::Add => AtomicOp::IAdd,
        AirOp::Sub => AtomicOp::ISub,
        AirOp::Min if signed => AtomicOp::SMin,
        AirOp::Min => AtomicOp::UMin,
        AirOp::Max if signed => AtomicOp::SMax,
        AirOp::Max => AtomicOp::UMax,
        AirOp::And => AtomicOp::And,
        AirOp::Or => AtomicOp::Or,
        AirOp::Xor => AtomicOp::Xor,
        AirOp::Xchg => AtomicOp::Exchange,
        AirOp::Load | AirOp::Store | AirOp::CmpXchgWeak => return None,
    })
}

fn air_kind(a: &AirAtomic) -> Result<AtomicKind, Error> {
    let float = a.ty.elem == Elem::F32;
    Ok(match a.op {
        AirOp::Load => AtomicKind::Load,
        AirOp::Store => AtomicKind::Store,
        AirOp::CmpXchgWeak => AtomicKind::CompareExchange,
        AirOp::Add if float => AtomicKind::Cas(CasOp::FAdd),
        AirOp::Sub if float => AtomicKind::Cas(CasOp::FSub),
        AirOp::Min if float => AtomicKind::Cas(CasOp::FMin),
        AirOp::Max if float => AtomicKind::Cas(CasOp::FMax),
        op => AtomicKind::Rmw(int_op(op, a.sign).ok_or_else(refuse)?),
    })
}

/// The kind of an LLVM `atomicrmw` on a value of type `elem`; `None` when the operator does not
/// apply to it (the verifier rejects such IR, so this is refused, not an invariant).
fn rmw_kind(op: AtomicRmwOp, elem: Elem) -> Option<AtomicKind> {
    let float = elem == Elem::F32;
    if op.is_float() != float && op != AtomicRmwOp::Xchg {
        return None;
    }
    Some(match op {
        AtomicRmwOp::Xchg => AtomicKind::Rmw(AtomicOp::Exchange),
        AtomicRmwOp::Add => AtomicKind::Rmw(AtomicOp::IAdd),
        AtomicRmwOp::Sub => AtomicKind::Rmw(AtomicOp::ISub),
        AtomicRmwOp::And => AtomicKind::Rmw(AtomicOp::And),
        AtomicRmwOp::Or => AtomicKind::Rmw(AtomicOp::Or),
        AtomicRmwOp::Xor => AtomicKind::Rmw(AtomicOp::Xor),
        AtomicRmwOp::Max => AtomicKind::Rmw(AtomicOp::SMax),
        AtomicRmwOp::Min => AtomicKind::Rmw(AtomicOp::SMin),
        AtomicRmwOp::UMax => AtomicKind::Rmw(AtomicOp::UMax),
        AtomicRmwOp::UMin => AtomicKind::Rmw(AtomicOp::UMin),
        AtomicRmwOp::Nand => AtomicKind::Cas(CasOp::Nand),
        AtomicRmwOp::UIncWrap => AtomicKind::Cas(CasOp::UIncWrap),
        AtomicRmwOp::UDecWrap => AtomicKind::Cas(CasOp::UDecWrap),
        AtomicRmwOp::FAdd => AtomicKind::Cas(CasOp::FAdd),
        AtomicRmwOp::FSub => AtomicKind::Cas(CasOp::FSub),
        // LLVM's `fmax`/`fmin` are `maxnum`/`minnum`: a NaN operand yields the other.
        AtomicRmwOp::FMax => AtomicKind::Cas(CasOp::FMax),
        AtomicRmwOp::FMin => AtomicKind::Cas(CasOp::FMin),
    })
}

/// The atomic element of AIR type `ty`: `i8`, `i16`, `i32`, `i64`, or `f32`; `None` for any
/// other.
fn elem_of(cx: &SelectCx<'_>, ty: TypeId) -> Option<Elem> {
    match cx.entry.module.types().get(ty)? {
        Type::Int(8) => Some(Elem::I8),
        Type::Int(16) => Some(Elem::I16),
        Type::Int(32) => Some(Elem::I32),
        Type::Int(64) => Some(Elem::I64),
        Type::Float => Some(Elem::F32),
        _ => None,
    }
}

/// An LLVM atomic on `e` with alignment `align`, which must be at least its size: an
/// under-aligned atomic is legal IR (LLVM lowers it to a library call), but every memory here
/// holds naturally aligned words, and a narrow field must not straddle two of them.
fn natural(align: Option<Align>, e: Elem) -> Result<Elem, Error> {
    let size = match e {
        Elem::I8 => 1,
        Elem::I16 => 2,
        Elem::I64 => 8,
        _ => 4,
    };
    match align {
        Some(a) if a.bytes() < size => Err(refuse()),
        _ => Ok(e),
    }
}

/// The value id of operand `v`: an instruction result or parameter, or a constant the arith
/// family materializes.
fn value_id(cx: &SelectCx<'_>, v: Value) -> Result<ValueId, Error> {
    if matches!(v, Value::Const(_)) && cx.constant(v).is_none() {
        return Err(not_lowered(LowerFamily::Arith));
    }
    cx.values
        .id(v)
        .ok_or_else(|| cx.invariant("an atomic operand out of the value range"))
}

/// Which memory an atomic names: `air`'s space, or none for an LLVM atomic (its pointer's).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Named {
    Air(AtomicSpace),
    Llvm,
}

/// The comparator operand of a compare-exchange, before selection.
#[derive(Clone, Copy)]
enum CompareOperand {
    Expected(Value),
    Value(Value),
}

/// A decoded atomic, before its memory is resolved.
struct Request {
    named: Named,
    ptr: Value,
    kind: AtomicKind,
    elem: Elem,
    value: Option<Value>,
    compare: Option<CompareOperand>,
    order: MemoryOrder,
    failure: MemoryOrder,
    /// An AIR 2.9 atomic's `mem_flags`; `None` for an AIR 2.8 or LLVM atomic.
    flags: Option<MemFlags>,
}

/// The request of `inst`, or `None` when it is no memory atomic.
fn request(cx: &SelectCx<'_>, inst: &Inst) -> Result<Option<Request>, Error> {
    let elem = |cx: &SelectCx<'_>, ty: TypeId| elem_of(cx, ty).ok_or_else(refuse);
    let ty_of = |cx: &SelectCx<'_>, v: Value| {
        cx.value_ty(v)
            .ok_or_else(|| cx.invariant("an atomic operand without a type"))
    };
    Ok(Some(match &inst.kind {
        InstKind::Call(call) => {
            let Callee::Air {
                op: AirIntrinsic::Atomic(a),
                ..
            } = &call.callee
            else {
                return Ok(None);
            };
            if a.ty.lanes != 1 || !matches!(a.ty.elem, Elem::I32 | Elem::I64 | Elem::F32) {
                return Err(refuse());
            }
            let args: &[Value] = &call.args;
            let arg = |i: usize| -> Result<Value, Error> {
                args.get(i)
                    .copied()
                    .ok_or_else(|| cx.invariant("an atomic without its operands"))
            };
            let kind = air_kind(a)?;
            let (value, compare) = match kind {
                AtomicKind::Load => (None, None),
                AtomicKind::CompareExchange => {
                    (Some(arg(2)?), Some(CompareOperand::Expected(arg(1)?)))
                }
                _ => (Some(arg(1)?), None),
            };
            Request {
                named: Named::Air(a.space),
                ptr: arg(0)?,
                kind,
                elem: a.ty.elem,
                value,
                compare,
                order: order(a.order),
                failure: a.failure.map_or(MemoryOrder::Relaxed, order),
                flags: a.flags,
            }
        }
        InstKind::AtomicRmw {
            op,
            ptr,
            val,
            atomic,
            align,
            ..
        } => {
            let e = natural(*align, elem(cx, ty_of(cx, *val)?)?)?;
            Request {
                named: Named::Llvm,
                ptr: *ptr,
                kind: rmw_kind(*op, e).ok_or_else(refuse)?,
                elem: e,
                value: Some(*val),
                compare: None,
                order: llvm_order(atomic.ordering),
                failure: MemoryOrder::Relaxed,
                flags: None,
            }
        }
        InstKind::CmpXchg {
            ptr,
            cmp,
            new,
            success,
            failure,
            align,
            ..
        } => {
            let e = natural(*align, elem(cx, ty_of(cx, *cmp)?)?)?;
            if e == Elem::F32 {
                return Err(refuse());
            }
            Request {
                named: Named::Llvm,
                ptr: *ptr,
                kind: AtomicKind::CompareExchange,
                elem: e,
                value: Some(*new),
                compare: Some(CompareOperand::Value(*cmp)),
                order: llvm_order(success.ordering),
                failure: llvm_order(*failure),
                flags: None,
            }
        }
        InstKind::Load {
            ptr,
            align,
            atomic: Some(a),
            ..
        } => Request {
            named: Named::Llvm,
            ptr: *ptr,
            kind: AtomicKind::Load,
            elem: natural(*align, elem(cx, inst.ty)?)?,
            value: None,
            compare: None,
            order: llvm_order(a.ordering),
            failure: MemoryOrder::Relaxed,
            flags: None,
        },
        InstKind::Store {
            val,
            ptr,
            align,
            atomic: Some(a),
            ..
        } => Request {
            named: Named::Llvm,
            ptr: *ptr,
            kind: AtomicKind::Store,
            elem: natural(*align, elem(cx, ty_of(cx, *val)?)?)?,
            value: Some(*val),
            compare: None,
            order: llvm_order(a.ordering),
            failure: MemoryOrder::Relaxed,
            flags: None,
        },
        _ => return Ok(None),
    }))
}

/// Select the atomic strategy of `inst`, or `None` when it is no memory atomic.
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    m: &mut Meter,
) -> Result<Option<AtomicAccess>, Error> {
    let Some(r) = request(cx, inst)? else {
        return Ok(None);
    };
    let (threadgroup, through_root) = match (cx.memory.repr(r.ptr), r.named) {
        (Some(Repr::TgPtr), Named::Air(AtomicSpace::Local) | Named::Llvm) => (true, false),
        (Some(Repr::BufPtr { wide, .. }), Named::Air(AtomicSpace::Global) | Named::Llvm) => {
            (false, wide)
        }
        (Some(Repr::PrivPtr(PrivForm::Arena)), Named::Air(AtomicSpace::Private)) => {
            return private(cx, inst, r).map(Some);
        }
        _ => return Err(refuse()),
    };
    let addressed = cx.profile.fields().buffer_access == BufferAccess::Addressed;
    let wide = r.elem == Elem::I64;
    let narrow = match r.elem {
        Elem::I8 => Some(8),
        Elem::I16 => Some(16),
        _ => None,
    };
    if wide {
        let leaf = &cx.profile.fields().leaf;
        let (have, feature) = if threadgroup {
            (
                leaf.shared_int64_atomics,
                VkFeature::ShaderSharedInt64Atomics,
            )
        } else {
            (
                leaf.buffer_int64_atomics,
                VkFeature::ShaderBufferInt64Atomics,
            )
        };
        if !have {
            return Err(
                Unsupported::needs(Feature::Int64Atomics, Capability::Feature(feature)).into(),
            );
        }
        if threadgroup {
            // The arena is a `u32` word array; a `u64` view of it is not declared.
            return Err(refuse());
        }
        if !addressed {
            // A buffer's `u64` word is reached through its device address only.
            return Err(Unsupported::needs_any_of(
                Feature::Int64Atomics,
                vec![cx.profile.addressed_needs()],
            )
            .into());
        }
    }
    // `f32` add and sub are native through a typed device address where the profile has the
    // buffer float-add feature; otherwise, and for the arena, a compare-and-swap loop.
    let kind = match r.kind {
        AtomicKind::Cas(op @ (CasOp::FAdd | CasOp::FSub))
            if !threadgroup && addressed && cx.profile.fields().leaf.buffer_float32_atomic_add =>
        {
            AtomicKind::FloatAdd {
                negate: op == CasOp::FSub,
            }
        }
        k => k,
    };
    let typed = wide || matches!(kind, AtomicKind::FloatAdd { .. });
    let value = match r.value {
        Some(v) => Some(value_id(cx, v)?),
        None => None,
    };
    let compare = match r.compare {
        None => None,
        Some(CompareOperand::Value(v)) => Some(Compare::Value(value_id(cx, v)?)),
        Some(CompareOperand::Expected(p)) => {
            let ty = cx
                .value_ty(
                    r.value
                        .ok_or_else(|| cx.invariant("a compare-exchange without value"))?,
                )
                .ok_or_else(|| cx.invariant("a compare-exchange value without a type"))?;
            Some(Compare::Expected(local::access(
                cx,
                Arena::Private,
                p,
                None,
                ty,
                None,
            )?))
        }
    };
    let mem = if threadgroup {
        local::use_arena(cx, Arena::Threadgroup)?;
        super::tile_memory::written(cx)?;
        AtomicMem::Threadgroup {
            ptr: local::access_ptr(
                cx,
                r.ptr,
                Arena::Threadgroup,
                u64::from(r.elem.bits().div_ceil(8)),
            )?,
            contained: cx.profile.fields().containment != Containment::UnsafeOff,
        }
    } else if through_root {
        // A wide pointer reaches every candidate through its root entry's device address, so a
        // typed atomic needs no slot moved.
        memory_model::note_device_scope(cx);
        AtomicMem::Wide {
            ptr: super::dynamic::plan_deref(cx, r.ptr)?,
            contained: cx.profile.fields().containment != Containment::UnsafeOff,
        }
    } else {
        let (mut targets, null) = static_roots(cx, r.ptr, m)?;
        let check = contain::atomic_encoding(cx.profile);
        for t in &mut targets {
            t.check = check;
            let plan = cx
                .memory
                .slots
                .get(t.slot)
                .ok_or_else(|| cx.invariant("an atomic target names a missing slot"))?;
            if view_kind(cx.profile, plan) != ViewKind::Storage {
                return Err(cx.invariant("an atomic through a uniform view"));
            }
            cx.push.insert(PushField::SlotBindOffset(plan.slot));
            if typed {
                // A typed atomic's slot is reached by device address (`finish_slots`).
                cx.families.access.typed_slots.insert(t.slot);
            }
        }
        memory_model::note_device_scope(cx);
        AtomicMem::Buffer {
            ptr: super::operand(cx, r.ptr)?,
            targets,
            null,
        }
    };
    let spin = kind == AtomicKind::Load && r.order == MemoryOrder::Relaxed && spin::is_spin(cx, at);
    Ok(Some(AtomicAccess {
        mem,
        kind,
        value,
        float: r.elem == Elem::F32,
        wide,
        narrow,
        compare,
        order: r.order,
        failure: r.failure,
        spin,
        // Threadgroup memory is a kernel's, an object function's, a mesh function's, and a
        // fragment function's tile (device memory in the pass's buffer, which emission orders as
        // such); a vertex function has none. The payload `mem_object_data` names is an object or
        // mesh function's.
        flags: r.flags.map(|f| match cx.stage {
            Stage::Object | Stage::Mesh => f,
            Stage::Kernel | Stage::Fragment => MemFlags {
                object_data: false,
                ..f
            },
            Stage::Vertex => MemFlags {
                threadgroup: false,
                threadgroup_imageblock: false,
                object_data: false,
                ..f
            },
        }),
    }))
}

/// The atomic `r` (instruction `inst`) on a word of the private arena: the leaf of its value type,
/// its operand, and the comparator a compare-exchange reads through `expected`.
fn private(cx: &mut SelectCx<'_>, inst: &Inst, r: Request) -> Result<AtomicAccess, Error> {
    if !matches!(r.elem, Elem::I32 | Elem::F32) {
        return Err(cx.invariant("a thread atomic of another type than i32 or f32"));
    }
    // The value's type: the operand's, or the result's for a load.
    let ty = match r.value {
        Some(v) => cx
            .value_ty(v)
            .ok_or_else(|| cx.invariant("an atomic operand without a type"))?,
        None => inst.ty,
    };
    let align = Align::from_bytes(4).ok_or_else(|| cx.invariant("a word's alignment"))?;
    let word = local::access(cx, Arena::Private, r.ptr, None, ty, Some(align))?;
    let value = match r.value {
        Some(v) => Some(value_id(cx, v)?),
        None => None,
    };
    let compare = match r.compare {
        None => None,
        Some(CompareOperand::Value(_)) => {
            return Err(cx.invariant("a thread atomic with an LLVM comparator"))
        }
        Some(CompareOperand::Expected(p)) => Some(Compare::Expected(local::access(
            cx,
            Arena::Private,
            p,
            None,
            ty,
            None,
        )?)),
    };
    Ok(AtomicAccess {
        mem: AtomicMem::Private(word),
        kind: r.kind,
        value,
        float: r.elem == Elem::F32,
        wide: false,
        narrow: None,
        compare,
        order: r.order,
        failure: r.failure,
        flags: r.flags,
        // Thread memory: no other invocation stores to it.
        spin: false,
    })
}

/// The new value of a thread atomic's word (word bit patterns of type `word`) for the old value
/// `old`, the operand `x`, and a compare-exchange's comparator `c`.
fn private_new(
    fb: &mut FunctionBuilder<'_>,
    kind: AtomicKind,
    word: STypeId,
    (old, x, c): (SValue, SValue, Option<SValue>),
) -> BResult<SValue> {
    Ok(match kind {
        AtomicKind::Rmw(AtomicOp::IAdd) => fb.binary(BinOp::IAdd, old, x)?,
        AtomicKind::Rmw(AtomicOp::ISub) => fb.binary(BinOp::ISub, old, x)?,
        AtomicKind::Rmw(AtomicOp::And) => fb.binary(BinOp::BitwiseAnd, old, x)?,
        AtomicKind::Rmw(AtomicOp::Or) => fb.binary(BinOp::BitwiseOr, old, x)?,
        AtomicKind::Rmw(AtomicOp::Xor) => fb.binary(BinOp::BitwiseXor, old, x)?,
        AtomicKind::Rmw(AtomicOp::SMin) => fb.glsl(GlslOp::SMin, &[old, x])?,
        AtomicKind::Rmw(AtomicOp::UMin) => fb.glsl(GlslOp::UMin, &[old, x])?,
        AtomicKind::Rmw(AtomicOp::SMax) => fb.glsl(GlslOp::SMax, &[old, x])?,
        AtomicKind::Rmw(AtomicOp::UMax) => fb.glsl(GlslOp::UMax, &[old, x])?,
        AtomicKind::Rmw(AtomicOp::Exchange) | AtomicKind::Store => x,
        AtomicKind::CompareExchange => {
            let equal = fb.binary(BinOp::IEqual, old, c.ok_or_else(dangling)?)?;
            fb.select(equal, x, old)?
        }
        AtomicKind::Cas(op) => cas_new(fb, op, word, old, x)?,
        AtomicKind::Rmw(_) | AtomicKind::Load | AtomicKind::FloatAdd { .. } => {
            return Err(dangling())
        }
    })
}

/// A thread atomic on leaf `leaf`: the old value as the word's bit pattern (`None` for a store).
fn on_private(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    leaf: &LocalAccess,
    o: Op,
    float: bool,
) -> BResult<Option<SValue>> {
    let typed = |fb: &mut FunctionBuilder<'_>, bits: SValue| -> BResult<SValue> {
        if float {
            let f = f32_ty(fb)?;
            fb.convert(ConvOp::Bitcast, bits, f)
        } else {
            Ok(bits)
        }
    };
    if o.kind == AtomicKind::Store {
        let v = typed(fb, o.value.ok_or_else(dangling)?)?;
        local::emit_access(cx, fb, leaf, Some(v))?;
        return Ok(None);
    }
    let loaded = local::emit_access(cx, fb, leaf, None)?.ok_or_else(dangling)?;
    let old = if float {
        fb.convert(ConvOp::Bitcast, loaded, o.word)?
    } else {
        loaded
    };
    if o.kind == AtomicKind::Load {
        return Ok(Some(old));
    }
    let x = o.value.ok_or_else(dangling)?;
    let new = private_new(fb, o.kind, o.word, (old, x, o.comparator))?;
    let new = typed(fb, new)?;
    local::emit_access(cx, fb, leaf, Some(new))?;
    Ok(Some(old))
}

/// Everything one atomic instruction needs once its word pointer is known.
#[derive(Clone, Copy, Debug)]
struct Op {
    kind: AtomicKind,
    scope: Scope,
    sem: Semantics,
    failure: Semantics,
    /// The word type.
    word: STypeId,
    /// The pointee of a typed device-address atomic (`u64` or `f32`), `None` for a word.
    typed: Option<STypeId>,
    /// A narrow atomic's width in bits: it operates on a field of the word.
    narrow: Option<u8>,
    /// The operand as the word's bit pattern.
    value: Option<SValue>,
    /// The comparator (compare-exchange).
    comparator: Option<SValue>,
    /// Acquire on the atomic's own memory: a compare-and-swap loop's read of the word, which the
    /// loop's exit depends on (a spin read, [`spin`]).
    reload: Semantics,
}

impl Op {
    fn yields(&self) -> bool {
        self.kind != AtomicKind::Store
    }
}

fn u32_ty(fb: &mut FunctionBuilder<'_>) -> BResult<STypeId> {
    fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })
}

fn f32_ty(fb: &mut FunctionBuilder<'_>) -> BResult<STypeId> {
    fb.module().ty(SType::Float { width: 32 })
}

/// The constant `v` of integer type `ty`.
fn konst(fb: &mut FunctionBuilder<'_>, ty: STypeId, v: u64) -> BResult<SValue> {
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(v),
    })?;
    fb.module().const_value(c)
}

/// A missing operand: an internal invariant failure, reported as a dangling value.
fn dangling() -> crate::spirv::error::BuildError {
    crate::spirv::error::BuildError::Dangling(crate::spirv::error::Dangling::Value(SValue::new(
        u32::MAX,
    )))
}

/// The semantics of a load (acquire at most) and of a store (release at most).
fn load_sem(s: Semantics) -> Semantics {
    match s.order {
        MemoryOrder::Relaxed | MemoryOrder::Release => Semantics::RELAXED,
        _ => Semantics {
            order: MemoryOrder::Acquire,
            make_available: false,
            ..s
        },
    }
}

fn store_sem(s: Semantics) -> Semantics {
    match s.order {
        MemoryOrder::Relaxed | MemoryOrder::Acquire => Semantics::RELAXED,
        _ => Semantics {
            order: MemoryOrder::Release,
            make_visible: false,
            ..s
        },
    }
}

/// The success and failure semantics of a compare-exchange. SPIR-V's failure semantics are
/// relaxed or acquire and never stronger than the success's; LLVM's `cmpxchg` and Metal's
/// `atomic_compare_exchange_weak_explicit` may fail acquiring where the success does not (a
/// `release` or `relaxed` success with an `acquire` failure), so that success is raised to
/// acquire as well. A stronger order only narrows the outcomes, and the failure keeps its
/// acquire.
fn compare_exchange_sems(success: Semantics, failure: Semantics) -> (Semantics, Semantics) {
    let acquires =
        |s: Semantics| matches!(s.order, MemoryOrder::Acquire | MemoryOrder::AcquireRelease);
    let failure = if acquires(failure) {
        load_sem(failure)
    } else {
        Semantics::RELAXED
    };
    let success = match success.order {
        _ if !acquires(failure) || acquires(success) => success,
        MemoryOrder::Release => Semantics {
            order: MemoryOrder::AcquireRelease,
            make_visible: failure.make_visible,
            ..success
        },
        _ => failure,
    };
    (success, failure)
}

/// `new = op(old, x)` on word bit patterns of type `word`.
fn cas_new(
    fb: &mut FunctionBuilder<'_>,
    op: CasOp,
    word: STypeId,
    old: SValue,
    x: SValue,
) -> BResult<SValue> {
    if op.is_float() {
        let f = f32_ty(fb)?;
        let a = fb.convert(ConvOp::Bitcast, old, f)?;
        let b = fb.convert(ConvOp::Bitcast, x, f)?;
        let r = match op {
            CasOp::FAdd => fb.binary(BinOp::FAdd, a, b)?,
            CasOp::FSub => fb.binary(BinOp::FSub, a, b)?,
            CasOp::FMin => fb.glsl(GlslOp::NMin, &[a, b])?,
            _ => fb.glsl(GlslOp::NMax, &[a, b])?,
        };
        return fb.convert(ConvOp::Bitcast, r, word);
    }
    let zero = konst(fb, word, 0)?;
    let one = konst(fb, word, 1)?;
    match op {
        CasOp::Nand => {
            let a = fb.binary(BinOp::BitwiseAnd, old, x)?;
            fb.unary(UnOp::Not, a)
        }
        CasOp::UIncWrap => {
            let wrap = fb.binary(BinOp::UGreaterThanEqual, old, x)?;
            let inc = fb.binary(BinOp::IAdd, old, one)?;
            fb.select(wrap, zero, inc)
        }
        _ => {
            let is_zero = fb.binary(BinOp::IEqual, old, zero)?;
            let above = fb.binary(BinOp::UGreaterThan, old, x)?;
            let wrap = fb.binary(BinOp::LogicalOr, is_zero, above)?;
            let dec = fb.binary(BinOp::ISub, old, one)?;
            fb.select(wrap, x, dec)
        }
    }
}

/// The compare-and-swap loop of `op` on word pointer `p`: the old value. The loop's exit depends
/// on its read of the word, which acquires (`Op::reload`).
fn cas_loop(fb: &mut FunctionBuilder<'_>, p: SValue, op: CasOp, o: Op) -> BResult<SValue> {
    let zero = fb.module().constant(Constant::Scalar {
        ty: o.word,
        bits: ConstBits(0),
    })?;
    // The old value leaves the loop through a variable.
    let var = fb.local_var(o.word, Some(zero))?;
    let out = fb.var_ptr(var)?;
    let x = o.value.ok_or_else(dangling)?;
    let body: LoopFn<'_, ()> = Box::new(move |fb, ctx, lc| {
        let old = fb.atomic_load(p, o.scope, o.reload)?;
        let new = cas_new(fb, op, o.word, old, x)?;
        let seen = fb.atomic_compare_exchange(p, o.scope, o.sem, Semantics::RELAXED, new, old)?;
        let done = fb.binary(BinOp::IEqual, seen, old)?;
        let exit: ArmFn<'_, ()> = Box::new(move |fb, _| {
            fb.store(out, seen, MemAccess::default())?;
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(ctx, done, exit, None)?;
        Ok(())
    });
    fb.loop_(&mut (), LoopSpec::default(), body)?;
    fb.load(out, MemAccess::default())
}

/// Where an atomic operates: a pointer to its word, and a narrow atomic's field's bit offset in
/// it (a `u32`).
#[derive(Clone, Copy, Debug)]
struct At {
    p: SValue,
    shift: Option<SValue>,
}

/// Apply `o` at `at`: the old value (`None` for a store).
fn apply(fb: &mut FunctionBuilder<'_>, at: At, o: Op) -> BResult<Option<SValue>> {
    let p = at.p;
    if let Some(bits) = o.narrow {
        let shift = at.shift.ok_or_else(dangling)?;
        return narrow::apply(fb, p, shift, bits, o);
    }
    Ok(Some(match o.kind {
        AtomicKind::Rmw(op) => fb.atomic(op, p, o.scope, o.sem, o.value.ok_or_else(dangling)?)?,
        AtomicKind::Load => fb.atomic_load(p, o.scope, load_sem(o.sem))?,
        AtomicKind::Store => {
            let v = o.value.ok_or_else(dangling)?;
            fb.atomic_store(p, o.scope, store_sem(o.sem), v)?;
            return Ok(None);
        }
        AtomicKind::CompareExchange => {
            let (success, failure) = compare_exchange_sems(o.sem, o.failure);
            fb.atomic_compare_exchange(
                p,
                o.scope,
                success,
                failure,
                o.value.ok_or_else(dangling)?,
                o.comparator.ok_or_else(dangling)?,
            )?
        }
        AtomicKind::Cas(op) => cas_loop(fb, p, op, o)?,
        AtomicKind::FloatAdd { negate } => {
            let f = f32_ty(fb)?;
            let x = o.value.ok_or_else(dangling)?;
            let mut x = fb.convert(ConvOp::Bitcast, x, f)?;
            if negate {
                x = fb.unary(UnOp::FNegate, x)?;
            }
            let old = fb.atomic(AtomicOp::FAdd, p, o.scope, o.sem, x)?;
            fb.convert(ConvOp::Bitcast, old, o.word)?
        }
    }))
}

/// Apply `o` at `at` only when `inb` holds: 0 (or nothing) out of bounds.
fn guarded(fb: &mut FunctionBuilder<'_>, inb: SValue, at: At, o: Op) -> BResult<Option<SValue>> {
    let then: ArmFn<'_, ()> =
        Box::new(move |fb, _| Ok(ArmEnd::FallThrough(apply(fb, at, o)?.into_iter().collect())));
    if !o.yields() {
        fb.if_(&mut (), inb, then, None)?;
        return Ok(None);
    }
    let otherwise: ArmFn<'_, ()> =
        Box::new(move |fb, _| Ok(ArmEnd::FallThrough(vec![konst(fb, o.word, 0)?])));
    let merged = fb.if_(&mut (), inb, then, Some(otherwise))?;
    Ok(merged.0.first().copied())
}

/// The atomic on one prepared buffer target.
fn on_target(fb: &mut FunctionBuilder<'_>, p: Prepared, o: Op) -> BResult<Option<SValue>> {
    let two = k32(fb, 2)?;
    let word = fb.binary(BinOp::ShiftRightLogical, p.off, two)?;
    let ptr = match (p.mem, o.typed) {
        (Mem::View(ViewMem(view, ..)), None) => {
            views::word_ptr(fb, view, word)?.ok_or_else(dangling)?
        }
        (Mem::Psb(m), typed) => {
            let pointee = match typed {
                Some(t) => t,
                None => u32_ty(fb)?,
            };
            let addr = m.word_address(fb, word)?;
            device_ptr(fb, addr, pointee)?
        }
        // Selection moved every slot of a typed atomic to its device address.
        (Mem::View(_), Some(_)) => return Err(dangling()),
    };
    let at = At {
        p: ptr,
        shift: field_shift(fb, o, p.off)?,
    };
    match (p.inb, p.guard_store) {
        (Some(inb), true) => guarded(fb, inb, at, o),
        // A device address out of bounds already points at the sink.
        (inb, _) => at_address(fb, at, inb, o),
    }
}

/// A narrow atomic's field's bit offset for byte offset `off` (a `u32`) of the access in its
/// word memory; `None` for a word atomic.
fn field_shift(fb: &mut FunctionBuilder<'_>, o: Op, off: SValue) -> BResult<Option<SValue>> {
    match o.narrow {
        Some(_) => narrow::shift(fb, off).map(Some),
        None => Ok(None),
    }
}

/// Apply `o` at device pointer `at`, which points at the sink when `inb` fails: 0 out of bounds.
fn at_address(
    fb: &mut FunctionBuilder<'_>,
    at: At,
    inb: Option<SValue>,
    o: Op,
) -> BResult<Option<SValue>> {
    let r = apply(fb, at, o)?;
    match (r, inb) {
        (Some(r), Some(inb)) => {
            let zero = konst(fb, o.word, 0)?;
            fb.select(inb, r, zero).map(Some)
        }
        (r, _) => Ok(r),
    }
}

/// The pointer an atomic operates on at device address `addr` (a `u64`), to a `pointee`: a place
/// (the member of a one-member block at `addr`, which MoltenVK's SPIRV-Cross can take the address
/// of, unlike a bare converted pointer). Every atomic this family emits through a device address
/// takes its pointer here.
fn device_ptr(fb: &mut FunctionBuilder<'_>, addr: SValue, pointee: STypeId) -> BResult<SValue> {
    fb.device_place(addr, pointee)
}

/// Emit atomic `a`, defining value `vid` with its result.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    vid: ValueId,
    a: &AtomicAccess,
) -> BResult<()> {
    let word = if a.wide {
        fb.module().ty(SType::Int {
            width: 64,
            signed: false,
        })?
    } else {
        u32_ty(fb)?
    };
    let typed = match a.kind {
        _ if a.wide => Some(word),
        AtomicKind::FloatAdd { .. } => Some(f32_ty(fb)?),
        _ => None,
    };
    let bytes = match (a.wide, a.narrow) {
        (true, _) => 8,
        (false, Some(bits)) => u32::from(bits / 8),
        (false, None) => 4,
    };
    let bits = |fb: &mut FunctionBuilder<'_>, x: SValue| -> BResult<SValue> {
        if a.float {
            fb.convert(ConvOp::Bitcast, x, word)
        } else {
            Ok(x)
        }
    };
    let value = match a.value {
        Some(v) => {
            let x = cx.operand(fb, v)?;
            Some(bits(fb, x)?)
        }
        None => None,
    };
    let comparator = match &a.compare {
        Some(Compare::Expected(e)) => match local::emit_access(cx, fb, e, None)? {
            Some(x) => Some(x),
            None => return Err(cx.fail("a compare-exchange without its comparator")),
        },
        Some(Compare::Value(v)) => Some(cx.operand(fb, *v)?),
        None => None,
    };
    let model = barrier::memory_model(cx.plan);
    // A fragment function's threadgroup memory is its tile's in the pass's buffer, a storage
    // buffer the whole device reaches: its atomics are `Device`-scope, and order it as device
    // memory.
    let tile_base = match a.mem {
        AtomicMem::Threadgroup { .. } => local::decl_of(cx, fb, Arena::Threadgroup)?.tile_base,
        _ => None,
    };
    let (own, scope) = match a.mem {
        AtomicMem::Threadgroup { .. } if tile_base.is_some() => (
            MemFlags {
                device: true,
                ..MemFlags::default()
            },
            Scope::Device,
        ),
        AtomicMem::Buffer { .. } => (
            MemFlags {
                device: true,
                ..MemFlags::default()
            },
            Scope::Device,
        ),
        AtomicMem::Threadgroup { .. } => (
            MemFlags {
                threadgroup: true,
                ..MemFlags::default()
            },
            Scope::Workgroup,
        ),
        // Thread memory orders nothing another thread observes.
        AtomicMem::Private(_) => (MemFlags::default(), Scope::Invocation),
        AtomicMem::Wide { .. } => (
            MemFlags {
                device: true,
                ..MemFlags::default()
            },
            Scope::Device,
        ),
    };
    // An AIR 2.9 atomic orders the memories its `mem_flags` name; any other its own. A fragment
    // function's threadgroup memory is its tile, device memory, whichever memory the atomic itself
    // accesses; a fragment function without a tile has no threadgroup memory for the flag to order.
    // A spin read acquires its own.
    let tile = cx
        .decl
        .families
        .access
        .local
        .tg
        .as_ref()
        .is_some_and(|t| t.fragment.is_some());
    let (flags, order) = match a.flags {
        _ if a.spin => (own, MemoryOrder::Acquire),
        Some(f)
            if cx.plan.stage == Stage::Fragment && (f.threadgroup || f.threadgroup_imageblock) =>
        {
            (
                MemFlags {
                    threadgroup: false,
                    threadgroup_imageblock: false,
                    device: f.device || tile,
                    ..f
                },
                a.order,
            )
        }
        Some(f) => (f, a.order),
        None => (own, a.order),
    };
    let o = Op {
        kind: a.kind,
        scope,
        sem: barrier::semantics(flags, order, model),
        failure: barrier::semantics(flags, a.failure, model),
        word,
        typed,
        narrow: a.narrow,
        value,
        comparator,
        reload: barrier::semantics(own, MemoryOrder::Acquire, model),
    };
    let result = match &a.mem {
        AtomicMem::Private(leaf) => on_private(cx, fb, leaf, o, a.float)?,
        AtomicMem::Threadgroup { ptr, contained } => {
            let decl = local::decl_of(cx, fb, Arena::Threadgroup)?;
            let off = match ptr {
                LocalPtr::Value(v) => cx.operand(fb, *v)?,
                LocalPtr::Offset(c) => k32(fb, *c)?,
            };
            let two = k32(fb, 2)?;
            let w = fb.binary(BinOp::ShiftRightLogical, off, two)?;
            let at = At {
                p: ArenaMem(decl, WordOps::default()).ptr(fb, w)?,
                shift: field_shift(fb, o, off)?,
            };
            let inb = match *contained {
                true => {
                    let size = fb.module().const_value(decl.size)?;
                    Some(contain::in_bounds(fb, off, size, u64::from(bytes))?)
                }
                false => None,
            };
            // A word of a tile's cell the pass does not keep reads as zero and keeps no write
            // (`access::tile::KeptCells`).
            let kept = match super::tile::KeptCells::of(cx, fb)? {
                Some(cells) => Some(cells.kept(fb, w)?),
                None => None,
            };
            let guard = match (inb, kept) {
                (Some(inb), Some(kept)) => Some(fb.binary(BinOp::LogicalAnd, inb, kept)?),
                (inb, kept) => inb.or(kept),
            };
            match guard {
                Some(guard) => guarded(fb, guard, at, o)?,
                None => apply(fb, at, o)?,
            }
        }
        AtomicMem::Wide { ptr, .. } => {
            let w = super::dynamic::atomic_address(cx, fb, *ptr, bytes)?;
            let at = At {
                p: device_ptr(fb, w.addr, typed.unwrap_or(word))?,
                shift: match w.byte {
                    Some(b) => field_shift(fb, o, b)?,
                    None => None,
                },
            };
            at_address(fb, at, w.inb, o)?
        }
        AtomicMem::Buffer { ptr, targets, .. } => {
            let fat = cx.operand(fb, *ptr)?;
            let m = fb.composite_extract(fat, &[1])?;
            let mut prepared = Vec::with_capacity(targets.len());
            for t in targets {
                let prep = prepare(cx, fb, t, m, bytes)?;
                if typed.is_some() && !matches!(prep.mem, Mem::Psb(_)) {
                    return Err(cx.fail("a typed atomic through a descriptor view"));
                }
                prepared.push((t.root, prep));
            }
            match prepared.as_slice() {
                // Only null: Metal leaves the dereference undefined; nothing is touched.
                [] => o.yields().then(|| konst(fb, word, 0)).transpose()?,
                [(_, one)] => on_target(fb, *one, o)?,
                many => {
                    let root = fb.composite_extract(fat, &[0])?;
                    let mut cases: Vec<(Vec<u32>, ArmFn<'_, ()>)> = Vec::with_capacity(many.len());
                    for (r, prep) in many.iter().copied() {
                        let arm: ArmFn<'_, ()> = Box::new(move |fb, _| {
                            Ok(ArmEnd::FallThrough(
                                on_target(fb, prep, o)?.into_iter().collect(),
                            ))
                        });
                        cases.push((vec![r], arm));
                    }
                    let default: ArmFn<'_, ()> = Box::new(move |fb, _| {
                        Ok(ArmEnd::FallThrough(if o.yields() {
                            vec![konst(fb, word, 0)?]
                        } else {
                            Vec::new()
                        }))
                    });
                    let merged = fb.switch_at_use(&mut (), root, cases, default)?;
                    merged.0.first().copied()
                }
            }
        }
    };
    let Some(old) = result else {
        return Ok(());
    };
    match (&a.compare, comparator) {
        (Some(Compare::Expected(e)), _) => {
            local::emit_access(cx, fb, e, Some(old))?;
        }
        (Some(Compare::Value(_)), Some(c)) => {
            // LLVM `cmpxchg` yields `{old, old == comparator}` (the arith family's aggregate).
            let equal = fb.binary(BinOp::IEqual, old, c)?;
            let b = fb.module().ty(SType::Bool)?;
            let pair = fb.module().ty(SType::Struct {
                members: [word, b].into(),
                layout: None,
            })?;
            let r = fb.composite_construct(pair, &[old, equal])?;
            cx.define(vid, r);
            return Ok(());
        }
        _ => {}
    }
    let r = if a.float {
        let f = f32_ty(fb)?;
        fb.convert(ConvOp::Bitcast, old, f)?
    } else {
        old
    };
    cx.define(vid, r);
    Ok(())
}
