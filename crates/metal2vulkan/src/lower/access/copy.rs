//! `llvm.memcpy`, `llvm.memmove`, and `llvm.memset`.
//!
//! **Endpoints.** Each pointer operand is resolved like a plain access's: static-root buffer slots
//! (descriptor views or device addresses; several candidates dispatched by a switch-at-use per
//! leaf; a source's candidates may include the constant blob), a wide (dynamic-root) pointer's
//! allocation (its root entry read once, each unit contained against it), the threadgroup or
//! private arena, or the constant blob (a source only). A typed object never reaches a memory
//! intrinsic: an alloca or constant global with one is placed in the private arena or the blob by
//! construction, so meeting one is an internal invariant failure.
//!
//! **Leaves.** The length is split into units of the widest leaf both endpoints' alignments
//! allow: 4-byte words when both are word-aligned, else single bytes; a word copy's last
//! `len mod 4` bytes are byte leaves. An endpoint is word-aligned when its call-site `align` is at
//! least 4, or when it is an arena or blob pointer whose offset is a translation-time constant
//! multiple of 4 (an object's placed offset displaced by constant GEPs; both memories start at a
//! word). A `memcpy` or `memmove` of a constant multiple of 16 bytes with a static buffer endpoint,
//! every buffer endpoint's call-site `align` at least 16 and its offset proven a multiple of 16
//! (`MemoryPlan::proven_align`: the call-site `align` alone is no proof, `memory::leaf`'s
//! placement), and every other endpoint word-aligned, moves 16-byte `uvec4` units instead: a
//! buffer slot with a `uvec4` view moves each unit as one element of it, a device address as one
//! access (`memory::leaf`'s whole words), and every other memory as its four words. Every unit is
//! one leaf load and store
//! (`memory::leaf`), each contained exactly as a plain access of that leaf, so a byte store never
//! disturbs a neighbour (the sub-word atomics); an arena unit at a constant offset that ends inside
//! the arena's translation-time size has no check, which could only pass. `memset` stores the byte
//! pattern (`byte · 0x01010101` for a word).
//!
//! **Loops.** Up to [`UNROLL_UNITS`] units of a constant length are unrolled, a 16-byte unit
//! counting as the four words it moves ([`unroll_units`]). A longer copy or fill is a counted loop
//! over the units, whose trip count is a translation-time constant (statically bounded; it opens a
//! builder loop), followed by the unrolled tail. A length that is not a constant (the dynamic size)
//! is a counted loop over `len / unit` units, followed by the tail bytes, each under `if (b < len
//! mod 4)`. No assumption is made about an `i64` length's high word: between wide endpoints only (a
//! `memcpy` or `memmove` whose both endpoints are wide, or a wide `memset`), whose allocations'
//! root entries have `u64` sizes, a `u64` length is counted exactly, with a `u64` counter and
//! displacements. Every other endpoint is displaced in `u32` (static slots, arenas, the blob: the
//! offset space), so a length of 2³² or more reaches past every byte it can address: the call is
//! out of bounds, and it counts as `u32::MAX` bytes, each unit contained. That covers every `i64`
//! held as two words without `shaderInt64`: such a profile has no device addresses
//! (`BufferAccess::Floor`), so no wide endpoint, and every window is at most
//! `maxStorageBufferRange` bytes.
//!
//! **Overlap.** `memcpy` operands may not overlap (LLVM). A `memmove` of at most
//! [`UNROLL_UNITS`] constant units loads every unit before storing any, so overlap is exact. A
//! longer or dynamic `memmove` ([`Overlap`]) between different memories (a buffer and an arena or
//! the blob, or the two arenas) cannot overlap and copies forward; within one arena, or within one
//! buffer slot (both endpoints' only candidate), it compares the two offsets from the memory's
//! start and copies forward when the destination is not above the source and backward otherwise,
//! which is exact for any overlap. Buffer endpoints with different or several candidate slots may
//! alias through the binding (one buffer bound at two slots), and a wide endpoint may reach any
//! buffer: those compare the endpoints' Metal addresses instead (`Overlap::Addressed`; a static
//! candidate's is its `SlotGuestBase` plus its displacement, a wide pointer's its root entry's
//! `guest + bind_offset + offset`), which needs `shaderInt64` for the `u64` addresses
//! (`Unsupported{PointerCompare}` naming it otherwise).
//!
//! **Volatile** intrinsics lower as the others, every unit accessed exactly once and in order, with
//! every word load and store of a unit carrying `Volatile` (a volatile load or store's operand; the
//! read-only blob stays plain). `volatile` orders nothing between invocations (only atomics,
//! barriers, and coherent accesses order memory across them).

use std::borrow::Cow;

use crate::air::intrinsics::LlvmIntrinsic;
use crate::air::ir::{Callee, CastOp, Const, Inst, InstId, InstKind, Value, ValueId};
use crate::budget::Meter;
use crate::error::{Error, LowerFamily};
use crate::lower::arith::{Form, Held};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::arena::Arena;
use crate::memory::leaf::{self, k32, u32_op, Leaf, Path};
use crate::memory::plan::ArenaObject;
use crate::pipeline::interface::Window;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ids::{DemoteTy, ScalarTy};
use crate::spirv::ir::{BinOp, ConstBits, Constant, ConvOp, MemAccess, Type as SType};

use super::blob;
use super::dynamic::{self, WideAccess, WideBase};
use super::emit::access_at;
use super::local::{self, Home, LocalAccess, LocalPtr};
use super::psb;
use super::{
    const_index, not_lowered, operand, refuse, Access, AccessPlan, BufferReach, SideReach,
};

/// The most units a copy or fill unrolls (the proposed threshold); a longer one is a counted loop.
pub const UNROLL_UNITS: u32 = 256;

/// The most `unit`-byte units a copy or fill unrolls: [`UNROLL_UNITS`], a 16-byte unit counting as
/// the four words it moves, so a copy unrolls exactly when its word copy would.
pub fn unroll_units(unit: u32) -> u32 {
    UNROLL_UNITS / (unit / 4).max(1)
}

/// Where one side of a memory intrinsic lives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Endpoint {
    /// Static-root buffer slots; the access's leaf is replaced per unit.
    Buffer(Access),
    /// An arena; the access's leaf is replaced per unit.
    Arena(LocalAccess),
    /// The constant blob (a source only).
    Blob { ptr: LocalPtr, checked: bool },
    /// A wide (dynamic-root) pointer's allocation; the access's leaf is replaced per unit.
    Wide(WideAccess),
}

impl Endpoint {
    /// An upper estimate of the SPIR-V words of one access of the endpoint whose leaf costs
    /// `per` words: per candidate slot with its containment and switch arm for buffers, the
    /// address, the check, and the sink selects for a wide pointer, and the leaf with its offset
    /// for an arena or the blob.
    pub fn access_words(&self, per: u32) -> u32 {
        match self {
            Endpoint::Buffer(a) => (a.targets.len() as u32)
                .max(1)
                .saturating_mul(per.saturating_add(160))
                .saturating_add(32),
            Endpoint::Arena(_) | Endpoint::Blob { .. } => per.saturating_add(32),
            Endpoint::Wide(_) => per.saturating_add(48),
        }
    }

    /// The same endpoint, its accesses volatile when `volatile` (the read-only blob stays plain).
    fn with_volatile(self, volatile: bool) -> Endpoint {
        match self {
            Endpoint::Buffer(a) => Endpoint::Buffer(Access { volatile, ..a }),
            Endpoint::Arena(a) => Endpoint::Arena(LocalAccess { volatile, ..a }),
            Endpoint::Wide(a) => Endpoint::Wide(WideAccess { volatile, ..a }),
            blob @ Endpoint::Blob { .. } => blob,
        }
    }

    /// What the endpoint reaches in the finished plan `slots`: its targets' slots, a wide pointer's
    /// device address, the constant blob's view, or its arena.
    pub fn reach(&self, slots: &AccessPlan) -> SideReach {
        match self {
            Endpoint::Buffer(a) => SideReach::Buffer(a.reach(slots)),
            Endpoint::Wide(_) => SideReach::Buffer(BufferReach::ADDRESSES),
            Endpoint::Blob { .. } => SideReach::Buffer(BufferReach::VIEWS),
            Endpoint::Arena(_) => SideReach::Arena,
        }
    }

    /// The SPIR-V words [`base`] emits once for the endpoint: a wide pointer's root-entry reads
    /// and miss counter (0 for the others).
    pub fn base_words(&self) -> u32 {
        match self {
            Endpoint::Wide(_) => dynamic::BASE_WORDS,
            Endpoint::Buffer(_) | Endpoint::Arena(_) | Endpoint::Blob { .. } => 0,
        }
    }
}

/// How a copy handles overlapping operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Overlap {
    /// None is possible (`memcpy`, or a `memmove` between different memories): forward.
    Disjoint,
    /// A short constant `memmove`: every unit is loaded before any is stored.
    Buffered,
    /// A `memmove` within one memory: forward when the destination's offset from `start` (the
    /// memory's first byte as a Metal offset, wrapping) is not above the source's, else
    /// backward.
    Directed { start: u32 },
    /// A `memmove` between buffer endpoints that may be one allocation through different roots
    /// (one buffer bound at two slots, or a wide pointer): as `Directed`, comparing the
    /// endpoints' Metal addresses.
    Addressed,
}

/// What a memory intrinsic writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// Bytes read from another endpoint.
    Copy { from: Endpoint, overlap: Overlap },
    /// The byte `value` (`memset`), an operand or a constant.
    Fill { value: Option<ValueId>, konst: u8 },
}

/// The length of a memory intrinsic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Length {
    /// `count` whole units, then `tail` bytes (below the unit).
    Const { count: u32, tail: u32 },
    /// The operand `len`, an `i32` (held in a `u32`) or an `i64` (`wide`, held in a `u64`, or as
    /// two `u32` words without `shaderInt64`).
    Dynamic { len: ValueId, wide: bool },
}

/// A memory intrinsic.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemOp {
    pub dst: Endpoint,
    pub src: Source,
    /// Bytes per unit: 16 (a constant-length copy only), 4, or 1.
    pub unit: u32,
    pub length: Length,
}

impl MemOp {
    /// Whether every endpoint is wide (a `memset`'s destination, or both of a copy's): the op
    /// reaches only allocations whose root entries have `u64` sizes.
    pub fn only_wide(&self) -> bool {
        matches!(self.dst, Endpoint::Wide(_))
            && matches!(
                &self.src,
                Source::Fill { .. }
                    | Source::Copy {
                        from: Endpoint::Wide(_),
                        ..
                    }
            )
    }

    /// Whether the units are a counted loop.
    pub fn looped(&self) -> bool {
        match self.length {
            Length::Const { count, .. } => count > unroll_units(self.unit),
            Length::Dynamic { .. } => true,
        }
    }

    /// The census name in the finished plan `slots`: the operation's ([`MemOp::op_name`]), then
    /// what the destination reaches (`.to_<token>`) and what a copy's source reaches
    /// (`.from_<token>`), each unless descriptor views alone ([`SideReach::token`]): each side's
    /// units take that side's check, so a name that merged them would cover a copy and its reverse.
    pub fn name(&self, slots: &AccessPlan) -> Cow<'static, str> {
        let to = self.dst.reach(slots).token();
        let from = match &self.src {
            Source::Copy { from, .. } => from.reach(slots).token(),
            Source::Fill { .. } => None,
        };
        let base = self.op_name();
        match (to, from) {
            (None, None) => base.into(),
            (to, from) => {
                let mut name = base.to_owned();
                for (side, token) in [(".to_", to), (".from_", from)] {
                    if let Some(t) = token {
                        name.push_str(side);
                        name.push_str(t);
                    }
                }
                name.into()
            }
        }
    }

    /// The operation's name (`access.memcpy`/`memmove`/`memset`, `.loop` when counted,
    /// `.dynamic` for a length that is not a constant).
    pub fn op_name(&self) -> &'static str {
        let dynamic = matches!(self.length, Length::Dynamic { .. });
        match (&self.src, self.looped(), dynamic) {
            (Source::Copy { overlap, .. }, _, true) => match overlap {
                Overlap::Directed { .. } | Overlap::Addressed => "access.memmove.dynamic",
                _ => "access.memcpy.dynamic",
            },
            (Source::Copy { overlap, .. }, looped, false) => match (overlap, looped) {
                (Overlap::Buffered, _) => "access.memmove",
                (Overlap::Directed { .. } | Overlap::Addressed, _) => "access.memmove.loop",
                (Overlap::Disjoint, false) => "access.memcpy",
                (Overlap::Disjoint, true) => "access.memcpy.loop",
            },
            (Source::Fill { .. }, _, true) => "access.memset.dynamic",
            (Source::Fill { .. }, false, false) => "access.memset",
            (Source::Fill { .. }, true, false) => "access.memset.loop",
        }
    }

    /// An upper estimate of the SPIR-V words one use emits.
    pub fn words(&self) -> u32 {
        let side =
            |e: &Endpoint, leaf: &Leaf| -> u32 { e.access_words(leaf.cost().saturating_add(64)) };
        // A wide endpoint reads its root entry once; a directed copy between addressed
        // endpoints computes their Metal addresses.
        let addressed = matches!(
            &self.src,
            Source::Copy {
                overlap: Overlap::Addressed,
                ..
            }
        );
        let once = |e: &Endpoint| -> u32 {
            let address = match e {
                Endpoint::Buffer(a) if addressed => {
                    48u32.saturating_mul(a.targets.len() as u32 + 1)
                }
                Endpoint::Wide(_) if addressed => 72,
                _ => 0,
            };
            e.base_words().saturating_add(address)
        };
        let bases = once(&self.dst).saturating_add(match &self.src {
            Source::Copy { from, .. } => once(from),
            Source::Fill { .. } => 0,
        });
        let unit_words = |leaf: &Leaf| -> u32 {
            let src = match &self.src {
                Source::Copy { from, .. } => side(from, leaf),
                Source::Fill { .. } => 8,
            };
            src.saturating_add(side(&self.dst, leaf))
        };
        // A directed copy emits its units twice (forward and backward).
        let copies: u32 = match &self.src {
            Source::Copy {
                overlap: Overlap::Directed { .. } | Overlap::Addressed,
                ..
            } => 2,
            _ => 1,
        };
        let (units, tail) = match self.length {
            Length::Const { count, tail } if count <= unroll_units(self.unit) => (
                count.saturating_mul(unit_words(&unit_leaf(self.unit))),
                tail,
            ),
            Length::Const { tail, .. } => {
                (unit_words(&unit_leaf(self.unit)).saturating_add(96), tail)
            }
            Length::Dynamic { .. } => (unit_words(&unit_leaf(self.unit)).saturating_add(112), 3),
        };
        units
            .saturating_add(tail.saturating_mul(unit_words(&unit_leaf(1)).saturating_add(16)))
            .saturating_mul(copies)
            .saturating_add(32)
            .saturating_add(bases)
    }
}

/// The leaf of an `n`-byte unit (16, 4, or 1): four `u32` words moved as one `uvec4` where the
/// memory has a `uvec4` view, else an unsigned integer of `8n` bits in a `u32`.
fn unit_leaf(n: u32) -> Leaf {
    if n == 16 {
        return Leaf {
            form: Form {
                held: Held::Int {
                    bits: 32,
                    container: 32,
                },
                lanes: 4,
            },
            bytes: 16,
            path: Path::Vec4,
        };
    }
    Leaf {
        form: Form {
            held: Held::Int {
                bits: (8 * n).min(32) as u8,
                container: 32,
            },
            lanes: 1,
        },
        bytes: n,
        path: if n >= 4 { Path::Aligned } else { Path::OneWord },
    }
}

/// The call-site `align` of argument `i` in bytes (1 when unstated).
pub fn arg_align(call: &crate::air::ir::Call, i: usize) -> u64 {
    call.attrs
        .as_ref()
        .and_then(|a| a.params.get(i))
        .and_then(|p| p.align)
        .map_or(1, |a| a.bytes())
}

/// Resolve pointer `v` as a memory intrinsic's endpoint, written when `write`. The simd family
/// reaches the lanes of an AGX emask access through the same endpoints.
pub fn endpoint(
    cx: &mut SelectCx<'_>,
    v: Value,
    write: bool,
    m: &mut Meter,
) -> Result<Endpoint, Error> {
    let leaf = unit_leaf(1);
    if blob::is_blob(cx, v)? {
        if write {
            return Err(refuse());
        }
        blob::use_blob(cx)?;
        return Ok(Endpoint::Blob {
            ptr: blob::blob_ptr(cx, v)?,
            checked: true,
        });
    }
    match local::home_of(cx, v) {
        Some(home @ Home::Arena(arena)) => {
            local::use_arena(cx, arena)?;
            local::composed(cx, arena);
            if write {
                local::writable(cx, arena)?;
            }
            let ptr = match const_arena_offset(cx, v, arena, 0) {
                Some(o) => LocalPtr::Offset(o),
                None => local::local_ptr(cx, v, home)?,
            };
            return Ok(Endpoint::Arena(LocalAccess {
                arena,
                ptr,
                value: None,
                leaf,
                contained: local::contained(cx, arena),
                volatile: false,
            }));
        }
        Some(_) => {
            return Err(cx.invariant("a typed object reached by a memory intrinsic"));
        }
        None => {}
    }
    // A source may also point into the constant blob (its `BLOB_ROOT` candidate); the blob is
    // constant, so a destination may not. The targets are placed as a load's or a store's are.
    let (mut targets, null, blob) = super::blob_roots(cx, v, write, m)?;
    // A copy's chunks take their own leaves when it is emitted, words among them. A chunk's
    // offset is the endpoint's plus a constant where the copy is unrolled; a looping copy's
    // chunks are noted apart (`select`).
    let fixed = super::fixed_offset(cx, v);
    super::place_targets(cx, &mut targets, (write, fixed), 4)?;
    Ok(Endpoint::Buffer(Access {
        ptr: operand(cx, v)?,
        value: None,
        leaf,
        targets,
        null,
        coherence: None,
        volatile: false,
        blob,
    }))
}

/// A memory intrinsic's endpoint `v`: a wide pointer's allocation, else [`endpoint`]'s.
fn intrinsic_endpoint(
    cx: &mut SelectCx<'_>,
    v: Value,
    write: bool,
    m: &mut Meter,
) -> Result<Endpoint, Error> {
    match dynamic::memory_endpoint(cx, v)? {
        Some(a) => Ok(Endpoint::Wide(a)),
        None => endpoint(cx, v, write, m),
    }
}

/// The most steps [`const_arena_offset`] follows from a pointer back to its object.
const OFFSET_WALK: u32 = 32;

/// The offset of arena pointer `v` in `arena` when it is known at translation time: an arena
/// object's address (an alloca or a global the memory plan placed there), displaced by casts,
/// `freeze`, and GEPs whose indices are all constants, in wrapping `u32` arithmetic as the arena
/// family computes them. `None` for any other pointer, or past [`OFFSET_WALK`] steps.
fn const_arena_offset(cx: &mut SelectCx<'_>, v: Value, arena: Arena, steps: u32) -> Option<u32> {
    if steps >= OFFSET_WALK {
        return None;
    }
    let object = match v {
        Value::Inst(i) => match cx.body.insts.get(i).map(|x| &x.kind) {
            Some(InstKind::Alloca { .. }) => ArenaObject::Alloca(i),
            Some(InstKind::Cast {
                op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                v,
            })
            | Some(InstKind::Freeze(v)) => {
                let v = *v;
                return const_arena_offset(cx, v, arena, steps + 1);
            }
            Some(InstKind::Gep {
                src_ty,
                base,
                indices,
                ..
            }) => {
                if indices.iter().any(|i| const_index(cx, *i).is_none()) {
                    return None;
                }
                let (src_ty, base, indices) = (*src_ty, *base, indices.clone());
                let b = const_arena_offset(cx, base, arena, steps + 1)?;
                // The GEP's own selection refuses a GEP the layouts cannot place.
                let (konst, _) = super::gep_offset(cx, src_ty, &indices).ok()?;
                return Some(b.wrapping_add(konst));
            }
            _ => return None,
        },
        Value::Const(c) => match cx.entry.module.consts().get(c).map(|k| &k.value) {
            Some(Const::GlobalAddr(g)) => ArenaObject::Global(*g),
            _ => return None,
        },
        _ => return None,
    };
    let layout = match arena {
        Arena::Private => &cx.memory.priv_arena,
        Arena::Threadgroup => &cx.memory.tg_arena,
        // The payload holds no object of the plan's.
        Arena::Payload => return None,
    };
    layout
        .entry(object)
        .and_then(|e| u32::try_from(e.offset).ok())
}

/// Select the strategy of `inst`, or `None` when it is no memory intrinsic.
pub fn select(
    cx: &mut SelectCx<'_>,
    _at: InstId,
    inst: &Inst,
    m: &mut Meter,
) -> Result<Option<MemOp>, Error> {
    let InstKind::Call(call) = &inst.kind else {
        return Ok(None);
    };
    let Callee::Llvm { op, .. } = &call.callee else {
        return Ok(None);
    };
    // A volatile intrinsic lowers as the others, its units' words `Volatile` (module doc).
    let (fill, overlap, volatile) = match *op {
        LlvmIntrinsic::Memset { volatile, .. } => (true, false, volatile),
        LlvmIntrinsic::Memcpy { volatile, .. } => (false, false, volatile),
        LlvmIntrinsic::Memmove { volatile, .. } => (false, true, volatile),
        _ => return Ok(None),
    };
    let args: &[Value] = &call.args;
    let arg = |cx: &SelectCx<'_>, i: usize| -> Result<Value, Error> {
        args.get(i)
            .copied()
            .ok_or_else(|| cx.invariant("a memory intrinsic without its operands"))
    };
    let n = arg(cx, 2)?;
    // A constant length is split into units once the endpoints' alignment is known.
    let span = match const_index(cx, n) {
        // A constant length beyond the `u32` offset space reaches no Metal allocation's end.
        Some(len) => Span::Bytes(u32::try_from(len).map_err(|_| refuse())?),
        None if matches!(n, Value::Const(_)) => return Err(not_lowered(LowerFamily::Arith)),
        None => {
            let ty = cx
                .value_ty(n)
                .ok_or_else(|| cx.invariant("a memory intrinsic length without a type"))?;
            let wide = match cx.lowered_repr(n, ty)? {
                DemoteTy::Scalar(ScalarTy::Int { bits: 32 }) => false,
                DemoteTy::Scalar(ScalarTy::Int { bits: 64 }) => true,
                // An `i64` held as two words without `shaderInt64`.
                DemoteTy::Vector {
                    elem: ScalarTy::Int { bits: 32 },
                    len: 2,
                } => true,
                _ => return Err(refuse()),
            };
            Span::Dynamic(Length::Dynamic {
                len: operand(cx, n)?,
                wide,
            })
        }
    };
    let d = arg(cx, 0)?;
    let dst = intrinsic_endpoint(cx, d, true, m)?.with_volatile(volatile);
    let from = if fill {
        None
    } else {
        Some(intrinsic_endpoint(cx, arg(cx, 1)?, false, m)?.with_volatile(volatile))
    };
    let unit = unit_of(
        (
            &dst,
            arg_align(call, 0),
            cx.memory.proven_align(cx.entry, d),
        ),
        from.as_ref()
            .map(|f| -> Result<_, Error> {
                let s = arg(cx, 1)?;
                Ok((f, arg_align(call, 1), cx.memory.proven_align(cx.entry, s)))
            })
            .transpose()?,
        &span,
    );
    let length = match span {
        Span::Bytes(len) => Length::Const {
            count: len / unit,
            tail: len % unit,
        },
        Span::Dynamic(length) => length,
    };
    // The leaves the copy moves: its units, and single bytes for a tail below the unit.
    let tail = match length {
        Length::Const { tail, .. } => tail > 0,
        Length::Dynamic { .. } => unit > 1,
    };
    for leaf in [Some(unit_leaf(unit)), tail.then(|| unit_leaf(1))]
        .into_iter()
        .flatten()
    {
        for (e, write) in [(Some(&dst), true), (from.as_ref(), false)] {
            if let Some(Endpoint::Buffer(a)) = e {
                super::note_units(cx, &a.targets, &leaf, write)?;
            }
        }
    }
    let buffered = matches!(length, Length::Const { count, .. } if count <= unroll_units(unit));
    // A copy that loops reaches its buffers at offsets that are not translation-time constants.
    if !buffered {
        for (e, write) in [(Some(&dst), true), (from.as_ref(), false)] {
            if let Some(Endpoint::Buffer(a)) = e {
                for t in &a.targets {
                    if let Some(plan) = cx.memory.slots.get(t.slot) {
                        let slot = plan.slot;
                        super::note_variable(cx, slot, false, !write);
                    }
                }
            }
        }
    }
    let src = if fill {
        let v = arg(cx, 1)?;
        match const_index(cx, v) {
            Some(b) => Source::Fill {
                value: None,
                konst: b as u8,
            },
            None if matches!(v, Value::Const(_)) => {
                return Err(not_lowered(LowerFamily::Arith));
            }
            None => Source::Fill {
                value: Some(operand(cx, v)?),
                konst: 0,
            },
        }
    } else {
        let from = from.ok_or_else(|| cx.invariant("a copy without its source"))?;
        let overlap = match (overlap, buffered) {
            (false, _) => Overlap::Disjoint,
            (true, true) => Overlap::Buffered,
            (true, false) => directed(cx, &dst, &from)?,
        };
        Source::Copy { from, overlap }
    };
    Ok(Some(MemOp {
        dst,
        src,
        unit,
        length,
    }))
}

/// A memory intrinsic's length before its unit is chosen.
enum Span {
    /// A constant number of bytes.
    Bytes(u32),
    /// A length that is not a constant.
    Dynamic(Length),
}

/// The bytes of each unit of a copy or fill to `dst` from `from` (each with its call-site
/// `align` and the alignment its offset is proven to have) of length `span` (module doc). A copy
/// of a constant nonzero multiple of 16 bytes with a static buffer endpoint, every buffer
/// endpoint's call-site `align` at least 16 and its offset proven a multiple of 16, and every
/// other endpoint word-aligned, moves 16: each unit's descriptor offset is a multiple of 16, so a
/// `uvec4` view element holds exactly its bytes, and every other memory moves a 16-byte unit as
/// its four words, which needs only word alignment. Otherwise a unit is a word when both endpoints
/// are word-aligned, else a byte.
fn unit_of(dst: (&Endpoint, u64, u64), from: Option<(&Endpoint, u64, u64)>, span: &Span) -> u32 {
    let ends = || std::iter::once(dst).chain(from);
    let vec4 = from.is_some()
        && matches!(*span, Span::Bytes(len) if len > 0 && len % 16 == 0)
        && ends().any(|(e, _, _)| matches!(e, Endpoint::Buffer(_)))
        && ends().all(|(e, call, proven)| match e {
            Endpoint::Buffer(_) => call >= 16 && proven >= 16,
            _ => known_align(e, call) >= 4,
        });
    if vec4 {
        16
    } else if ends().all(|(e, call, _)| known_align(e, call) >= 4) {
        4
    } else {
        1
    }
}

/// The alignment of endpoint `e`'s bytes in its word memory, at most 4 (the widest unit): the
/// call-site `align`, or for an arena or blob pointer at a constant offset that offset's own
/// alignment, whichever is larger (both memories start at a word). A buffer or wide endpoint's
/// offset from its memory's start is not known at translation time, so only the call site speaks
/// for it.
fn known_align(e: &Endpoint, call: u64) -> u64 {
    let konst = match e {
        Endpoint::Arena(LocalAccess {
            ptr: LocalPtr::Offset(o),
            ..
        })
        | Endpoint::Blob {
            ptr: LocalPtr::Offset(o),
            ..
        } => Some(*o),
        Endpoint::Arena(_) | Endpoint::Blob { .. } | Endpoint::Buffer(_) | Endpoint::Wide(_) => {
            None
        }
    };
    let at = konst.map_or(1, |o| match o % 4 {
        0 => 4,
        2 => 2,
        _ => 1,
    });
    call.max(at).min(4)
}

/// How a long or dynamic `memmove` from `from` to `dst` handles overlap (module doc).
fn directed(cx: &mut SelectCx<'_>, dst: &Endpoint, from: &Endpoint) -> Result<Overlap, Error> {
    Ok(match (dst, from) {
        (Endpoint::Arena(a), Endpoint::Arena(b)) if a.arena == b.arena => {
            Overlap::Directed { start: 0 }
        }
        (Endpoint::Buffer(a), Endpoint::Buffer(b)) => {
            match (a.targets.as_slice(), b.targets.as_slice()) {
                ([x], [y]) if x.slot == y.slot => {
                    let plan = cx
                        .memory
                        .slots
                        .get(x.slot)
                        .ok_or_else(|| cx.invariant("a memmove names a missing slot"))?;
                    let start = match plan.window {
                        Some(Window::Bounded { start, .. }) => start as u32,
                        _ => 0,
                    };
                    Overlap::Directed { start }
                }
                // Different or several candidate slots may be one buffer bound twice.
                _ => addressed(cx, &[a, b])?,
            }
        }
        (Endpoint::Wide(_), Endpoint::Wide(_)) => Overlap::Addressed,
        (Endpoint::Wide(_), Endpoint::Buffer(a)) | (Endpoint::Buffer(a), Endpoint::Wide(_)) => {
            addressed(cx, &[a])?
        }
        // Different memories never overlap.
        _ => Overlap::Disjoint,
    })
}

/// A `memmove` between buffer endpoints that may alias, the static ones `statics`: their Metal
/// addresses are compared (`Overlap::Addressed`), which reads each candidate slot's
/// `SlotGuestBase`.
fn addressed(cx: &mut SelectCx<'_>, statics: &[&Access]) -> Result<Overlap, Error> {
    for a in statics {
        dynamic::plan_static_guest(cx, &a.targets, crate::error::Feature::PointerCompare)?;
    }
    Ok(Overlap::Addressed)
}

/// An endpoint with its base resolved in the open block.
#[derive(Clone, Copy)]
pub enum Base<'a> {
    Buffer {
        a: &'a Access,
        fat: SValue,
        m: SValue,
    },
    Arena {
        a: &'a LocalAccess,
        off: SValue,
    },
    Blob {
        off: SValue,
        checked: bool,
    },
    /// A wide pointer `p` and its resolved allocation.
    Wide {
        p: SValue,
        b: WideBase,
    },
}

pub fn base<'a>(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    e: &'a Endpoint,
) -> BResult<Base<'a>> {
    base_live(cx, fb, e, None)
}

/// [`base`] of an intrinsic that accesses memory only when `live` holds (a dynamic length's
/// nonzero one): a wide endpoint through root 0 counts a miss only then.
fn base_live<'a>(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    e: &'a Endpoint,
    live: Option<SValue>,
) -> BResult<Base<'a>> {
    Ok(match e {
        Endpoint::Buffer(a) => {
            let fat = cx.operand(fb, a.ptr)?;
            let m = super::emit::fat_part(fb, fat, 1)?;
            Base::Buffer { a, fat, m }
        }
        Endpoint::Arena(a) => Base::Arena {
            a,
            off: match a.ptr {
                LocalPtr::Value(v) => cx.operand(fb, v)?,
                LocalPtr::Offset(o) => k32(fb, o)?,
            },
        },
        Endpoint::Blob { ptr, checked } => Base::Blob {
            off: blob::ptr_value(cx, fb, *ptr)?,
            checked: *checked,
        },
        Endpoint::Wide(a) => {
            let p = cx.operand(fb, a.ptr)?;
            Base::Wide {
                p,
                b: dynamic::wide_base(cx, fb, p, a, live)?,
            }
        }
    })
}

/// `d` as a `u64` displacement from a wide base (`None` for 0).
fn wide_delta(fb: &mut FunctionBuilder<'_>, d: Delta) -> BResult<Option<SValue>> {
    Ok(match d {
        Delta::Const(0) => None,
        Delta::Const(c) => Some(psb::k64(fb, u64::from(c))?),
        Delta::Value(v) => {
            let u64_ = u64_ty(fb)?;
            Some(fb.convert(ConvOp::UConvert, v, u64_)?)
        }
        Delta::Wide(v) => Some(v),
    })
}

/// The `u64` type.
fn u64_ty(fb: &mut FunctionBuilder<'_>) -> BResult<crate::spirv::ids::TypeId> {
    fb.module().ty(SType::Int {
        width: 64,
        signed: false,
    })
}

/// `x + d`, known at translation time when both are. A `u64` displacement is a wide base's only.
fn plus(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, x: SValue, d: Delta) -> BResult<SValue> {
    let d = match d {
        Delta::Const(c) => k32(fb, c)?,
        Delta::Value(v) => v,
        Delta::Wide(_) => return Err(cx.fail("a 64-bit displacement from a 32-bit endpoint")),
    };
    u32_op(fb, BinOp::IAdd, x, d)
}

/// The bytes of `arena` every invocation has, known at translation time: the private arena's
/// size, the payload's, or the threadgroup arena's when no specialized region follows its static
/// objects.
fn arena_bytes(cx: &EmitCx<'_>, arena: Arena) -> Option<u64> {
    let local = &cx.plan.families.access.local;
    match arena {
        Arena::Private => local.private,
        Arena::Payload => local.payload,
        Arena::Threadgroup => local
            .tg
            .as_ref()
            .filter(|t| t.regions.is_empty())
            .map(|t| t.static_size),
    }
}

/// `a` with `leaf`, and without its in-shader check when the access provably lies inside the
/// arena: a constant offset whose `leaf.bytes` end within the arena's known size (the check could
/// only pass).
fn arena_access(
    cx: &EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    a: &LocalAccess,
    off: SValue,
    leaf: Leaf,
) -> BResult<LocalAccess> {
    let inside = match (leaf::known(fb, off)?, arena_bytes(cx, a.arena)) {
        (Some(o), Some(size)) => u64::from(o) + u64::from(leaf.bytes) <= size,
        _ => false,
    };
    Ok(LocalAccess {
        leaf,
        contained: a.contained && !inside,
        ..a.clone()
    })
}

/// A byte displacement from an endpoint's base.
#[derive(Clone, Copy, Debug)]
pub enum Delta {
    Const(u32),
    Value(SValue),
    /// A `u64` value: a displacement from a wide base only.
    Wide(SValue),
}

/// Load `leaf` at `b + d`: the value (contained).
pub fn read(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: Base<'_>,
    d: Delta,
    leaf: Leaf,
) -> BResult<SValue> {
    match b {
        Base::Buffer { a, fat, m } => {
            let a = Access { leaf, ..a.clone() };
            let m = plus(cx, fb, m, d)?;
            match access_at(cx, fb, &a, fat, m, None)? {
                Some(v) => Ok(v),
                None => Err(cx.fail("a buffer load without a value")),
            }
        }
        Base::Arena { a, off } => {
            let off = plus(cx, fb, off, d)?;
            let a = arena_access(cx, fb, a, off, leaf)?;
            match local::emit_access_at(cx, fb, &a, off, None)? {
                Some(v) => Ok(v),
                None => Err(cx.fail("an arena load without a value")),
            }
        }
        Base::Blob { off, checked } => {
            let off = plus(cx, fb, off, d)?;
            blob::load_at(cx, fb, &leaf, off, checked)
        }
        Base::Wide { b, .. } => {
            let d = wide_delta(fb, d)?;
            match dynamic::wide_at(fb, b, d, &leaf, None)? {
                Some(v) => Ok(v),
                None => Err(cx.fail("a wide load without a value")),
            }
        }
    }
}

/// Store `v` as `leaf` at `b + d` (contained).
pub fn write(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: Base<'_>,
    d: Delta,
    leaf: Leaf,
    v: SValue,
) -> BResult<()> {
    match b {
        Base::Buffer { a, fat, m } => {
            let a = Access { leaf, ..a.clone() };
            let m = plus(cx, fb, m, d)?;
            access_at(cx, fb, &a, fat, m, Some(v)).map(|_| ())
        }
        Base::Arena { a, off } => {
            let off = plus(cx, fb, off, d)?;
            let a = arena_access(cx, fb, a, off, leaf)?;
            local::emit_access_at(cx, fb, &a, off, Some(v)).map(|_| ())
        }
        Base::Blob { .. } => Err(cx.fail("a memory intrinsic writes the constant blob")),
        Base::Wide { b, .. } => {
            let d = wide_delta(fb, d)?;
            dynamic::wide_at(fb, b, d, &leaf, Some(v)).map(|_| ())
        }
    }
}

/// What one unit stores: a loaded value's source or the fill pattern of the unit's width.
#[derive(Clone, Copy)]
enum Feed<'a> {
    Base(Base<'a>),
    /// The fill byte, a `u32` in `0..=255`.
    Fill(SValue),
}

/// The fill pattern of an `n`-byte unit from fill byte `byte` (a constant for a constant byte).
fn pattern(fb: &mut FunctionBuilder<'_>, byte: SValue, n: u32) -> BResult<SValue> {
    if n == 1 {
        return Ok(byte);
    }
    let splat = k32(fb, 0x0101_0101)?;
    u32_op(fb, BinOp::IMul, byte, splat)
}

/// Copy or fill one unit of `leaf` at displacement `d`.
fn unit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    dst: Base<'_>,
    from: Feed<'_>,
    d: Delta,
    leaf: Leaf,
) -> BResult<()> {
    let v = match from {
        Feed::Base(b) => read(cx, fb, b, d, leaf)?,
        Feed::Fill(byte) => pattern(fb, byte, leaf.bytes)?,
    };
    write(cx, fb, dst, d, leaf, v)
}

/// The two sides of a copy or fill.
#[derive(Clone, Copy)]
struct Ends<'a> {
    dst: Base<'a>,
    from: Feed<'a>,
}

/// A number of units or tail bytes: a constant, a `u32` value, or (between wide endpoints only) a
/// `u64` value.
#[derive(Clone, Copy, Debug)]
enum Count {
    Const(u32),
    Value(SValue),
    Wide(SValue),
}

impl Count {
    /// The count and whether it is a `u64`.
    fn value(self, fb: &mut FunctionBuilder<'_>) -> BResult<(SValue, bool)> {
        match self {
            Count::Const(c) => Ok((k32(fb, c)?, false)),
            Count::Value(v) => Ok((v, false)),
            Count::Wide(v) => Ok((v, true)),
        }
    }
}

/// A constant of `wide` (`u64`) or `u32` type.
fn konst(fb: &mut FunctionBuilder<'_>, wide: bool, v: u32) -> BResult<SValue> {
    if wide {
        psb::k64(fb, u64::from(v))
    } else {
        k32(fb, v)
    }
}

/// The units and the tail bytes after them of `op`'s length, in the open block. A dynamic `i64`
/// length is exact between wide endpoints (a `u64` unit count); elsewhere one of 2³² or more (a
/// nonzero high word) counts as `u32::MAX` bytes (module doc).
fn span(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, op: &MemOp) -> BResult<(Count, Count)> {
    let (len, wide) = match op.length {
        Length::Const { count, tail } => return Ok((Count::Const(count), Count::Const(tail))),
        Length::Dynamic { len, wide } => (cx.operand(fb, len)?, wide),
    };
    // A dynamic length moves words or bytes ([`select`]).
    if op.unit > 4 {
        return Err(cx.fail("a dynamic length in units wider than a word"));
    }
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    if wide && !cx.plan.families.arith.wide && op.only_wide() {
        // A `u64` length between wide endpoints, counted exactly.
        if op.unit == 1 {
            return Ok((Count::Wide(len), Count::Const(0)));
        }
        let two = psb::k64(fb, 2)?;
        let units = fb.binary(BinOp::ShiftRightLogical, len, two)?;
        let low = fb.convert(ConvOp::UConvert, len, u32_)?;
        let three = k32(fb, 3)?;
        let tail = fb.binary(BinOp::BitwiseAnd, low, three)?;
        return Ok((Count::Wide(units), Count::Value(tail)));
    }
    let len = if wide {
        // A `u64`, or the two words `(low, high)` of a profile without `shaderInt64`.
        let (low, over) = if cx.plan.families.arith.wide {
            let low = fb.composite_extract(len, &[0])?;
            let high = fb.composite_extract(len, &[1])?;
            let zero = k32(fb, 0)?;
            (low, fb.binary(BinOp::INotEqual, high, zero)?)
        } else {
            let low = fb.convert(ConvOp::UConvert, len, u32_)?;
            let k = psb::k64(fb, 32)?;
            let high = fb.binary(BinOp::ShiftRightLogical, len, k)?;
            let zero = psb::k64(fb, 0)?;
            (low, fb.binary(BinOp::INotEqual, high, zero)?)
        };
        let max = k32(fb, u32::MAX)?;
        fb.select(over, max, low)?
    } else {
        len
    };
    if op.unit == 1 {
        return Ok((Count::Value(len), Count::Const(0)));
    }
    let two = k32(fb, 2)?;
    let units = fb.binary(BinOp::ShiftRightLogical, len, two)?;
    let three = k32(fb, 3)?;
    let tail = fb.binary(BinOp::BitwiseAnd, len, three)?;
    Ok((Count::Value(units), Count::Value(tail)))
}

/// Copy or fill `n` units of `leaf` (`step` bytes each) from displacement 0, first to last or
/// (`backward`) last to first: unrolled up to [`UNROLL_UNITS`] constant units, else a counted
/// loop (a builder loop).
fn units<'c>(
    cx: &mut EmitCx<'c>,
    fb: &mut FunctionBuilder<'_>,
    e: Ends<'_>,
    leaf: Leaf,
    step: u32,
    n: Count,
    backward: bool,
) -> BResult<()> {
    if let Count::Const(n) = n {
        if n <= unroll_units(step) {
            for i in 0..n {
                let k = if backward { n - 1 - i } else { i };
                unit(cx, fb, e.dst, e.from, Delta::Const(k * step), leaf)?;
            }
            return Ok(());
        }
    }
    // A `u64` count (between wide endpoints) runs a `u64` counter and displacement.
    let (n, wide) = n.value(fb)?;
    let ty = if wide {
        u64_ty(fb)?
    } else {
        fb.module().ty(SType::Int {
            width: 32,
            signed: false,
        })?
    };
    let zero = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(0),
    })?;
    let var = fb.local_var(ty, Some(zero))?;
    let counter = fb.var_ptr(var)?;
    let body: LoopFn<'_, EmitCx<'c>> = Box::new(move |fb, cx, lc| {
        let k = fb.load(counter, MemAccess::default())?;
        let done = fb.binary(BinOp::UGreaterThanEqual, k, n)?;
        let exit: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(cx, done, exit, None)?;
        let one = konst(fb, wide, 1)?;
        let next = fb.binary(BinOp::IAdd, k, one)?;
        fb.store(counter, next, MemAccess::default())?;
        let index = if backward {
            let last = fb.binary(BinOp::ISub, n, one)?;
            fb.binary(BinOp::ISub, last, k)?
        } else {
            k
        };
        let s = konst(fb, wide, step)?;
        let d = fb.binary(BinOp::IMul, index, s)?;
        let d = if wide {
            Delta::Wide(d)
        } else {
            Delta::Value(d)
        };
        unit(cx, fb, e.dst, e.from, d, leaf)
    });
    fb.loop_(cx, LoopSpec::default(), body)?;
    Ok(())
}

/// Copy or fill the `tail` bytes at displacement `at`, first to last or (`backward`) last to
/// first. A dynamic tail (below 4) moves byte `b` under `if (b < tail)`.
fn tail_bytes<'c>(
    cx: &mut EmitCx<'c>,
    fb: &mut FunctionBuilder<'_>,
    e: Ends<'_>,
    at: Count,
    tail: Count,
    backward: bool,
) -> BResult<()> {
    let byte = unit_leaf(1);
    let bytes: Vec<u32> = match tail {
        Count::Const(t) => (0..t).collect(),
        Count::Value(_) => (0..3).collect(),
        // The tail is below the unit, a `u32` even of a `u64` length ([`span`]).
        Count::Wide(_) => return Err(cx.fail("a 64-bit tail byte count")),
    };
    let order: Vec<u32> = if backward {
        bytes.into_iter().rev().collect()
    } else {
        bytes
    };
    for b in order {
        let d = match at {
            Count::Const(c) => Delta::Const(c.wrapping_add(b)),
            Count::Value(v) => {
                let k = k32(fb, b)?;
                Delta::Value(fb.binary(BinOp::IAdd, v, k)?)
            }
            Count::Wide(v) => {
                let k = psb::k64(fb, u64::from(b))?;
                Delta::Wide(fb.binary(BinOp::IAdd, v, k)?)
            }
        };
        match tail {
            Count::Const(_) | Count::Wide(_) => unit(cx, fb, e.dst, e.from, d, byte)?,
            Count::Value(t) => {
                let k = k32(fb, b)?;
                let some = fb.binary(BinOp::ULessThan, k, t)?;
                let arm: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, cx| {
                    unit(cx, fb, e.dst, e.from, d, byte)?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                });
                fb.if_(cx, some, arm, None)?;
            }
        }
    }
    Ok(())
}

/// Copy or fill the whole span, forward or backward (the tail's bytes are the highest).
fn span_copy<'c>(
    cx: &mut EmitCx<'c>,
    fb: &mut FunctionBuilder<'_>,
    e: Ends<'_>,
    step: u32,
    n: Count,
    tail: Count,
    backward: bool,
) -> BResult<()> {
    let at = match n {
        Count::Const(c) => Count::Const(c.wrapping_mul(step)),
        Count::Value(v) => {
            let s = k32(fb, step)?;
            Count::Value(fb.binary(BinOp::IMul, v, s)?)
        }
        Count::Wide(v) => {
            let s = psb::k64(fb, u64::from(step))?;
            Count::Wide(fb.binary(BinOp::IMul, v, s)?)
        }
    };
    let leaf = unit_leaf(step);
    if backward {
        tail_bytes(cx, fb, e, at, tail, true)?;
        units(cx, fb, e, leaf, step, n, true)
    } else {
        units(cx, fb, e, leaf, step, n, false)?;
        tail_bytes(cx, fb, e, at, tail, false)
    }
}

/// Where a directed `memmove` compares endpoint `b`: its offset from `start` in its memory
/// (`Directed`, a `u32`), or its Metal address (`Addressed`, a `u64`).
fn position(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: Base<'_>,
    overlap: Overlap,
) -> BResult<SValue> {
    match (overlap, b) {
        (Overlap::Directed { start }, Base::Buffer { m: off, .. })
        | (Overlap::Directed { start }, Base::Arena { off, .. } | Base::Blob { off, .. }) => {
            let k = k32(fb, start)?;
            fb.binary(BinOp::ISub, off, k)
        }
        (Overlap::Addressed, Base::Buffer { a, fat, .. }) => {
            dynamic::static_guest_address(cx, fb, &a.targets, fat)
        }
        (Overlap::Addressed, Base::Wide { p, .. }) => dynamic::guest_address(cx, fb, p),
        _ => Err(cx.fail("a directed memmove between endpoints it cannot compare")),
    }
}

/// Emit memory intrinsic `op`.
pub fn emit<'c>(
    cx: &mut EmitCx<'c>,
    fb: &mut FunctionBuilder<'_>,
    _vid: ValueId,
    op: &MemOp,
) -> BResult<()> {
    if let Length::Const { count: 0, tail: 0 } = op.length {
        return Ok(());
    }
    let (n, tail) = span(cx, fb, op)?;
    // A dynamic length may be 0: a wide endpoint then accesses nothing (and counts no miss).
    let wide = matches!(op.dst, Endpoint::Wide(_))
        || matches!(
            &op.src,
            Source::Copy {
                from: Endpoint::Wide(_),
                ..
            }
        );
    let nonzero = |fb: &mut FunctionBuilder<'_>, v: SValue, wide: bool| -> BResult<SValue> {
        let zero = konst(fb, wide, 0)?;
        fb.binary(BinOp::INotEqual, v, zero)
    };
    let live = match (wide, n, tail) {
        (true, Count::Value(n), Count::Const(0)) => Some(nonzero(fb, n, false)?),
        (true, Count::Value(n), Count::Value(t)) => {
            let bytes = fb.binary(BinOp::IAdd, n, t)?;
            Some(nonzero(fb, bytes, false)?)
        }
        // A `u64` unit count: live when it or the tail is nonzero.
        (true, Count::Wide(n), Count::Const(0)) => Some(nonzero(fb, n, true)?),
        (true, Count::Wide(n), Count::Value(t)) => {
            let units = nonzero(fb, n, true)?;
            let bytes = nonzero(fb, t, false)?;
            Some(fb.binary(BinOp::LogicalOr, units, bytes)?)
        }
        _ => None,
    };
    let dst = base_live(cx, fb, &op.dst, live)?;
    let (from, overlap) = match &op.src {
        Source::Copy { from, overlap } => (Feed::Base(base_live(cx, fb, from, live)?), *overlap),
        Source::Fill { value: Some(v), .. } => (Feed::Fill(cx.operand(fb, *v)?), Overlap::Disjoint),
        Source::Fill { value: None, konst } => {
            (Feed::Fill(k32(fb, u32::from(*konst))?), Overlap::Disjoint)
        }
    };
    match overlap {
        Overlap::Disjoint => span_copy(cx, fb, Ends { dst, from }, op.unit, n, tail, false),
        Overlap::Buffered => {
            // A short `memmove`: every unit is loaded before any is stored.
            let (Feed::Base(src), Length::Const { count, tail }) = (from, op.length) else {
                return Err(cx.fail("a buffered memmove without a source or a constant length"));
            };
            let wide = unit_leaf(op.unit);
            let byte = unit_leaf(1);
            let tail_at = count.saturating_mul(op.unit);
            let mut loaded = Vec::with_capacity((count + tail) as usize);
            for k in 0..count {
                let d = Delta::Const(k * op.unit);
                loaded.push((d, wide, read(cx, fb, src, d, wide)?));
            }
            for b in 0..tail {
                let d = Delta::Const(tail_at + b);
                loaded.push((d, byte, read(cx, fb, src, d, byte)?));
            }
            for (d, leaf, v) in loaded {
                write(cx, fb, dst, d, leaf, v)?;
            }
            Ok(())
        }
        Overlap::Directed { .. } | Overlap::Addressed => {
            let Feed::Base(src) = from else {
                return Err(cx.fail("a memmove without a source"));
            };
            // Forward when the destination is not above the source, else backward: each unit
            // is read before any overlapping unit is written.
            let d = position(cx, fb, dst, overlap)?;
            let s = position(cx, fb, src, overlap)?;
            let forward = fb.binary(BinOp::ULessThanEqual, d, s)?;
            let (step, e) = (op.unit, Ends { dst, from });
            let then: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, cx| {
                span_copy(cx, fb, e, step, n, tail, false)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            let otherwise: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, cx| {
                span_copy(cx, fb, e, step, n, tail, true)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            fb.if_(cx, forward, then, Some(otherwise))?;
            Ok(())
        }
    }
}
