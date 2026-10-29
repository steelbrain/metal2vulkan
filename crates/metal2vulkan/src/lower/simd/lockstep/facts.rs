//! The threadgroup-memory and device-memory accesses of a kernel body and what its IR states
//! about their addresses (device memory): the objects a pointer may point into, the bytes of one
//! object it may start at, a canonical spelling of the address, its offset into the array row its
//! last non-constant index steps through, whether the address is injective in the thread of a
//! simdgroup, and the unsigned comparisons the branches dominating the access settled. Value bounds
//! come from constants, the thread-index builtins, exact arithmetic, and loop counters' exit tests.
//!
//! Every fact is sound in the direction the detector reads it: a pair of accesses is only ever
//! cleared of a conflict by a fact that holds on every execution, and a fact the walk cannot
//! establish (an unknown object, a non-constant offset, an address whose dependence on the thread
//! is not a chain of exact steps) leaves the pair conflicting.

use std::collections::{BTreeMap, BTreeSet};

use crate::air::abi::{Builtin, ParamRole};
use crate::air::intrinsics::{
    AirIntrinsic, AtomicOp, Imageblock, LlvmIntrinsic, MemoryOrder, Sign, SimdMatrix, Sync, TexOp,
    ThreadScope,
};
use crate::air::ir::{
    AddrSpace, AtomicRmwOp, BinOp, BlockId, Body, Callee, CastOp, Const, ConstExpr, ConstId,
    GlobalId, Idx, InstId, InstKind, IntFlags, Type, TypeId, TypeLayouts, Value,
};
use crate::air::Entry;
use crate::budget::{tree_bytes, Meter};
use crate::error::Error;

use super::super::uniform::Uniformity;

/// The most objects a pointer's root set names before it is read as any object.
const MAX_ROOTS: usize = 8;
/// The longest chain of address or index steps the walks follow before giving up (a fact they do
/// not reach is unknown, which keeps the pair conflicting).
const MAX_CHAIN: u32 = 64;

/// The memory one pass of the analysis orders: threadgroup memory, whose objects never overlap, or
/// device memory, whose buffers the analysis takes as distinct objects too (see [`Root`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Space {
    /// `threadgroup` and `threadgroup_imageblock` memory.
    Threadgroup,
    /// `device` memory: buffers (`constant` memory is never written).
    Device,
}

/// An object of the memory a pass orders: a global, a buffer parameter, or (in device memory) the
/// buffer a pointer loaded from memory names, by the canonical spelling of the address it is
/// loaded from, so two loads of one argument-buffer member name one object. Distinct objects are
/// taken never to overlap. In threadgroup memory that holds (Metal allocates each object apart);
/// in device memory it is the analysis's reading: two bindings of one buffer, or a buffer and a
/// pointer loaded from memory into it, may overlap, and a pair reaching one byte through two
/// such objects is not ordered.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Root {
    Global(GlobalId),
    Param(u32),
    Loaded(Value),
}

/// A read-modify-write whose order against another of the same class does not change memory:
/// both commute, and neither result is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Update {
    Add,
    And,
    Or,
    Xor,
    SMin,
    SMax,
    UMin,
    UMax,
}

/// One threadgroup-memory access of the body.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Access {
    pub at: InstId,
    pub block: BlockId,
    /// The instruction's index in its block.
    pub pos: u32,
    /// The address; `None` for an operation on threadgroup memory without a pointer operand
    /// (an imageblock access), which may touch any of it.
    pub ptr: Option<Value>,
    pub read: bool,
    pub write: bool,
    /// The bytes accessed from `ptr`, when the instruction states them.
    pub size: Option<u64>,
    /// A commutative update whose result is unused.
    pub update: Option<Update>,
    /// The value a plain store writes, and the alignment it states.
    pub stored: Option<(Value, u64)>,
}

/// How the anchors of two threads of one simdgroup relate at one dynamic instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Spread {
    /// Both threads hold one anchor.
    Same,
    /// The threads' anchors differ by `min` to `max` bytes, either way, or, where `same`, are one
    /// anchor.
    Apart { min: u64, max: u64, same: bool },
}

/// How a GEP index extends its core to the pointer's index width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Ext {
    /// The core has the index width.
    None,
    Zero,
    Sign,
}

/// An address as a constant byte offset from its anchor: the address its last non-constant
/// step computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Anchor {
    /// The anchor's spelling: equal ids are equal anchors at one dynamic instance.
    pub id: u32,
    pub off: i64,
    pub spread: Option<Spread>,
    /// For a GEP anchor whose core is extended and whose base and earlier indices are uniform in
    /// the simdgroup.
    pub group: Option<Group>,
}

/// A GEP anchor's shape without its extended core, which orders the addresses of that shape
/// the way it orders their cores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Group {
    /// The spelling of the GEP without its core: equal ids differ only in the core, and the
    /// prefix they share is one address in every thread of the simdgroup.
    pub id: u32,
    pub core: Value,
    /// The bytes per unit of the core.
    pub stride: u64,
    /// Whether the core is sign-extended (its order is signed); else it is zero-extended.
    pub signed: bool,
}

/// An address's anchor, the blocks computing it (all of them, and those of its group), and
/// its row.
struct Anchored {
    anchor: Anchor,
    homes: Vec<BlockId>,
    group_homes: Vec<BlockId>,
    row: Option<Row>,
}

/// What is known about an access's address.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Addr {
    /// The objects it may point into; `None`: any.
    pub roots: Option<Vec<Root>>,
    /// One object and the byte offsets into it the address may take, `lo..=hi` (one offset
    /// when they are equal).
    pub span: Option<Span>,
    /// Its anchor and offset.
    pub anchor: Option<Anchor>,
    /// The blocks that compute the values the anchor is made of (none for arguments and
    /// constants, computed once).
    pub homes: Vec<BlockId>,
    /// The block that computes the value the access stores, if an instruction does.
    pub stored_home: Option<BlockId>,
    /// Its offset into the array its last non-constant index steps through.
    pub row: Option<Row>,
    /// The blocks computing the values the anchor's group is made of (its base and the indices
    /// before its core).
    pub group_homes: Vec<BlockId>,
    /// When the anchor's core is a sum `x + y` that does not wrap: the operands it is at least
    /// in its group's order, with their blocks.
    pub sum: Vec<(Value, Option<BlockId>)>,
    /// Facts `(x, y, signed)`, `x < y` (signed or unsigned) wherever the access runs: the
    /// conditions of the branch edges that dominate its block.
    pub below: Vec<(Value, Value, bool)>,
    /// The equalities that leave one thread of a simdgroup at most to run the access
    /// ([`Lone`]): the conditions of the branch edges that dominate its block.
    pub lone: Vec<Lone>,
    /// For a plain store: the blocks of the instructions its address is computed from through
    /// arithmetic alone ([`AddrFacts::leaves`]); `None` for another access, or where the walk
    /// does not end.
    pub leaves: Option<Vec<BlockId>>,
}

/// A fact `x == c` wherever an access runs, where `x` differs between any two threads of a
/// simdgroup (as an anchor's index may, [`AddrFacts::inj_int`]) and `c` is one value among them
/// (`if (tid == 0)`): of the threads of a simdgroup that hold one value of each, one at most
/// runs the access. `homes` are the blocks computing `x` and `c` (none for an argument or a
/// constant); `lockstep` holds them to the dynamic instances of a pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Lone {
    pub x: Value,
    pub c: Value,
    pub homes: [Option<BlockId>; 2],
}

/// The facts of the branch edges that dominate a block ([`AddrFacts::guards`]).
#[derive(Clone, Debug, Default)]
struct Guards {
    below: Vec<(Value, Value, bool)>,
    lone: Vec<Lone>,
}

/// An address as `base + size * q + o`, where `size` is the size of the array the address's
/// last non-constant GEP index steps through, every other step of the address is a multiple
/// of that size (`q` is any integer), and `o = stride * index + off` is the offset into the
/// array, in `lo..=hi`. Two addresses with one row spelling whose offsets stay inside the
/// array touch one byte only if their offsets do: differing `q`s put them in different arrays.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Row {
    /// The spelling: the canonical base, the array's size, the index's stride, and its core and
    /// extension.
    pub id: u32,
    pub size: u64,
    pub off: i64,
    pub lo: i64,
    pub hi: i64,
    /// How `stride * index` differs between two threads of a simdgroup.
    pub spread: Option<Spread>,
    /// The blocks computing the base and the core.
    pub homes: Vec<BlockId>,
}

/// The byte offsets `lo..=hi` into one object an address may take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Span {
    pub root: Root,
    pub lo: i64,
    pub hi: i64,
}

/// The spelling of an anchor: a value, or a GEP's source type, canonical base, the indices
/// before its last non-constant one, and that index's core and extension
/// ([`AddrFacts::affine`]; the constant it adds is part of the offset).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum AnchorKey {
    Value(Value),
    Gep(TypeId, Value, Box<[Value]>, Value, Ext),
    /// A [`Row`]: the canonical base, the array's size, the bytes per unit of the core, the
    /// core, and its extension.
    Row(Value, u64, u64, Value, Ext),
    /// An anchor's group: a GEP's source type, canonical base, and the indices before its core.
    Group(TypeId, Value, Box<[Value]>),
}

/// The accesses of a body, by block and position.
pub(super) struct Accesses {
    pub list: Vec<Access>,
    /// By block: the positions of the barriers that order threadgroup memory, ascending.
    pub barriers: Vec<Vec<u32>>,
    /// By block: the index of the first instruction after the PHIs, and of the terminator.
    pub first: Vec<u32>,
    pub term: Vec<u32>,
    /// By block: the range of `list` holding the block's accesses.
    pub range: Vec<(u32, u32)>,
    /// Some access writes.
    pub writes: bool,
}

/// Whether AIR call `op` takes its device-space pointers as resource handles rather than byte
/// addresses: texture, sampler, function-table, acceleration-structure, intersection, and
/// indirect-command operations.
fn handle_call(op: &AirIntrinsic) -> bool {
    matches!(
        op,
        AirIntrinsic::Texture(_)
            | AirIntrinsic::Sampler(_)
            | AirIntrinsic::FunctionTable(_)
            | AirIntrinsic::AccelerationStructure(_)
            | AirIntrinsic::Intersect(_)
            | AirIntrinsic::IndirectCommand(_)
    )
}

/// Whether `v` is a pointer into the memory of `space`.
fn space_ptr(e: &Entry, v: Value, space: Space) -> bool {
    let of_space = |at: &AddrSpace| match space {
        Space::Threadgroup => matches!(
            at,
            AddrSpace::Threadgroup | AddrSpace::ThreadgroupImageblock
        ),
        Space::Device => *at == AddrSpace::Device,
    };
    e.module
        .value_ty(e.func, v)
        .and_then(|t| e.module.types().get(t))
        .is_some_and(|t| matches!(t, Type::Ptr { space: at } if of_space(at)))
}

fn rmw_update(op: AtomicRmwOp) -> Option<Update> {
    Some(match op {
        AtomicRmwOp::Add | AtomicRmwOp::Sub => Update::Add,
        AtomicRmwOp::And => Update::And,
        AtomicRmwOp::Or => Update::Or,
        AtomicRmwOp::Xor => Update::Xor,
        AtomicRmwOp::Min => Update::SMin,
        AtomicRmwOp::Max => Update::SMax,
        AtomicRmwOp::UMin => Update::UMin,
        AtomicRmwOp::UMax => Update::UMax,
        _ => return None,
    })
}

fn air_update(a: &crate::air::intrinsics::Atomic) -> Option<Update> {
    if !a.ty.elem.is_int() {
        return None;
    }
    Some(match (a.op, a.sign) {
        (AtomicOp::Add | AtomicOp::Sub, _) => Update::Add,
        (AtomicOp::And, _) => Update::And,
        (AtomicOp::Or, _) => Update::Or,
        (AtomicOp::Xor, _) => Update::Xor,
        (AtomicOp::Min, Some(Sign::Signed)) => Update::SMin,
        (AtomicOp::Max, Some(Sign::Signed)) => Update::SMax,
        (AtomicOp::Min, Some(Sign::Unsigned)) => Update::UMin,
        (AtomicOp::Max, Some(Sign::Unsigned)) => Update::UMax,
        _ => return None,
    })
}

/// Whether call operation `op` is a barrier that orders the memory of `space` among the threads
/// of a simdgroup: it names that memory, both releases and acquires it (`seq_cst`, as every AIR
/// 2.8 barrier is, or `acq_rel`), and its memory scope holds the simdgroup. MSL 4.1's `relaxed`,
/// `acquire`, and `release` barriers do not order a write before one with a read after it, and
/// a `thread_scope_thread` barrier orders no other thread's accesses. For device memory a barrier
/// that names threadgroup memory counts too: every thread of its scope reaches it before any
/// passes, so the device accesses on either side are not ordered by lockstep, and what the
/// barrier's flags promise for them is the program's own reading.
fn ordering_barrier(op: &AirIntrinsic, space: Space) -> bool {
    match op {
        AirIntrinsic::Sync(s @ Sync::Barrier { flags, memory, .. }) => {
            let named = flags.threadgroup || flags.threadgroup_imageblock;
            (named || (space == Space::Device && flags.device))
                && matches!(
                    s.barrier_order(),
                    Some(MemoryOrder::SeqCst | MemoryOrder::AcqRel)
                )
                && *memory != ThreadScope::Thread
        }
        _ => false,
    }
}

impl Accesses {
    /// The accesses, barriers, and block shapes of `body`.
    pub fn of(
        e: &Entry,
        body: &Body,
        layouts: &TypeLayouts,
        space: Space,
        m: &mut Meter,
    ) -> Result<Accesses, Error> {
        let in_space = |v: Value| space_ptr(e, v, space);
        let n = body.insts.len();
        let nb = body.blocks.len();
        let mut uses = Vec::new();
        m.reserve(&mut uses, n)?;
        uses.resize(n, 0u32);
        for inst in body.insts.values() {
            m.tick(1)?;
            inst.kind.for_each_value(|v| {
                if let Value::Inst(d) = v {
                    if let Some(c) = uses.get_mut(d.index()) {
                        *c = c.saturating_add(1);
                    }
                }
            });
        }
        let unused = |i: InstId| uses.get(i.index()).copied() == Some(0);
        // A memory intrinsic's constant length.
        let len = |v: Option<&Value>| match v {
            Some(Value::Const(c)) => const_int(e, *c).and_then(|n| u64::try_from(n).ok()),
            _ => None,
        };
        let size_of = |ty: Option<TypeId>| ty.and_then(|t| layouts.store_size(t));
        let ty_of = |v: Value| e.module.value_ty(e.func, v);
        let mut acc = Accesses {
            list: Vec::new(),
            barriers: Vec::new(),
            first: Vec::new(),
            term: Vec::new(),
            range: Vec::new(),
            writes: false,
        };
        m.reserve(&mut acc.barriers, nb)?;
        m.reserve(&mut acc.first, nb)?;
        m.reserve(&mut acc.term, nb)?;
        m.reserve(&mut acc.range, nb)?;
        for (b, block) in body.blocks.iter() {
            m.tick(1)?;
            let start = u32::try_from(acc.list.len()).unwrap_or(u32::MAX);
            let mut bars = Vec::new();
            let mut first = None;
            for (pos, (at, inst)) in body.block_insts(b).enumerate() {
                let pos = u32::try_from(pos).unwrap_or(u32::MAX);
                if first.is_none() && !matches!(inst.kind, InstKind::Phi { .. }) {
                    first = Some(pos);
                }
                let mut push = |a: Access, m: &mut Meter| -> Result<(), Error> {
                    acc.writes |= a.write;
                    m.grow(&mut acc.list, 1)?;
                    acc.list.push(a);
                    Ok(())
                };
                let access = |ptr: Value, read: bool, write: bool, size: Option<u64>| Access {
                    at,
                    block: b,
                    pos,
                    ptr: Some(ptr),
                    read,
                    write,
                    size,
                    update: None,
                    stored: None,
                };
                match &inst.kind {
                    InstKind::Load { ptr, .. } if in_space(*ptr) => {
                        push(access(*ptr, true, false, size_of(Some(inst.ty))), m)?;
                    }
                    InstKind::Store {
                        val, ptr, align, ..
                    } if in_space(*ptr) => {
                        let stored = Some((*val, align.map_or(1, |a| a.bytes())));
                        push(
                            Access {
                                stored,
                                ..access(*ptr, false, true, size_of(ty_of(*val)))
                            },
                            m,
                        )?;
                    }
                    InstKind::AtomicRmw { op, ptr, val, .. } if in_space(*ptr) => {
                        let update = rmw_update(*op).filter(|_| unused(at));
                        push(
                            Access {
                                update,
                                ..access(*ptr, true, true, size_of(ty_of(*val)))
                            },
                            m,
                        )?;
                    }
                    InstKind::CmpXchg { ptr, cmp, .. } if in_space(*ptr) => {
                        push(access(*ptr, true, true, size_of(ty_of(*cmp))), m)?;
                    }
                    InstKind::Call(call) => {
                        if let Callee::Air { op, .. } = &call.callee {
                            if ordering_barrier(op, space) {
                                m.grow(&mut bars, 1)?;
                                bars.push(pos);
                                continue;
                            }
                            // A texture, sampler, function table, acceleration structure,
                            // intersector, or indirect command buffer is a handle in the device
                            // address space, not a byte address: its calls access no buffer
                            // word (texture memory is not ordered).
                            if space == Space::Device && handle_call(op) {
                                continue;
                            }
                            // The implicit imageblock's accesses name a pixel, not an address:
                            // any byte of it. `imageblock_data` only computes the address its
                            // loads and stores use.
                            let implicit = match op {
                                _ if space != Space::Threadgroup => None,
                                AirIntrinsic::Imageblock(Imageblock::Load(_)) => Some(false),
                                AirIntrinsic::Imageblock(Imageblock::Store(_)) => Some(true),
                                _ => None,
                            };
                            if let Some(write) = implicit {
                                push(
                                    Access {
                                        at,
                                        block: b,
                                        pos,
                                        ptr: None,
                                        read: !write,
                                        write,
                                        size: None,
                                        update: None,
                                        stored: None,
                                    },
                                    m,
                                )?;
                                continue;
                            }
                        }
                        for (k, &arg) in call.args.iter().enumerate() {
                            if !in_space(arg) {
                                continue;
                            }
                            let a = match &call.callee {
                                Callee::Air {
                                    op: AirIntrinsic::Atomic(a),
                                    ..
                                } if k == 0 => {
                                    let size = Some(
                                        u64::from(a.ty.elem.bits().div_ceil(8))
                                            .saturating_mul(u64::from(a.ty.lanes.max(1))),
                                    );
                                    match a.op {
                                        AtomicOp::Load => access(arg, true, false, size),
                                        AtomicOp::Store => access(arg, false, true, size),
                                        _ => Access {
                                            update: air_update(a)
                                                .filter(|_| !a.fetch || unused(at)),
                                            ..access(arg, true, true, size)
                                        },
                                    }
                                }
                                Callee::Air {
                                    op: AirIntrinsic::SimdMatrix(SimdMatrix::Load8x8 { .. }),
                                    ..
                                } => access(arg, true, false, None),
                                // An imageblock slice written to a texture is read.
                                Callee::Air {
                                    op: AirIntrinsic::Texture(t),
                                    ..
                                } if t.op == TexOp::WriteImageblockSlice => {
                                    access(arg, true, false, None)
                                }
                                Callee::Air {
                                    op: AirIntrinsic::SimdMatrix(SimdMatrix::Store8x8 { .. }),
                                    ..
                                } => access(arg, false, true, None),
                                Callee::Llvm {
                                    op: LlvmIntrinsic::Memset { .. },
                                    ..
                                } if k == 0 => access(arg, false, true, len(call.args.get(2))),
                                Callee::Llvm {
                                    op: LlvmIntrinsic::Memcpy { .. } | LlvmIntrinsic::Memmove { .. },
                                    ..
                                } if k <= 1 => access(arg, k == 1, k == 0, len(call.args.get(2))),
                                _ => access(arg, true, true, None),
                            };
                            push(a, m)?;
                        }
                    }
                    _ => {}
                }
            }
            let end = u32::try_from(acc.list.len()).unwrap_or(u32::MAX);
            acc.range.push((start, end));
            acc.barriers.push(bars);
            let len = u32::try_from(block.len()).unwrap_or(u32::MAX);
            acc.first.push(first.unwrap_or(0));
            acc.term.push(len.saturating_sub(1));
        }
        m.release(crate::budget::vec_bytes(&uses));
        Ok(acc)
    }

    /// The accesses of block `b`, in order.
    pub fn in_block(&self, b: BlockId) -> &[Access] {
        self.range
            .get(b.index())
            .and_then(|&(s, e)| self.list.get(s as usize..e as usize))
            .unwrap_or(&[])
    }

    /// The first ordering barrier of block `b` after position `pos` (`None`: none).
    pub fn barrier_after(&self, b: BlockId, pos: u32) -> Option<u32> {
        self.barriers
            .get(b.index())
            .and_then(|v| v.iter().copied().find(|&p| p > pos))
    }

    /// The first ordering barrier of block `b`.
    pub fn first_barrier(&self, b: BlockId) -> Option<u32> {
        self.barriers
            .get(b.index())
            .and_then(|v| v.first().copied())
    }

    /// Whether block `b` holds an ordering barrier.
    pub fn has_barrier(&self, b: BlockId) -> bool {
        self.first_barrier(b).is_some()
    }
}

/// The value of integer constant `c`, sign-extended from its width.
fn const_int(e: &Entry, c: ConstId) -> Option<i64> {
    let k = e.module.consts().get(c)?;
    let Const::Int(v) = k.value else {
        return None;
    };
    let Some(Type::Int(w)) = e.module.types().get(k.ty) else {
        return None;
    };
    let w = u32::from(*w);
    if w == 0 || w > 64 {
        return None;
    }
    let shift = 64 - w;
    Some(((v << shift) as i64) >> shift)
}

/// The width of integer type `ty`.
fn int_width(e: &Entry, ty: Option<TypeId>) -> Option<u32> {
    match e.module.types().get(ty?)? {
        Type::Int(w) => Some(u32::from(*w)),
        _ => None,
    }
}

/// Distinct threads of one simdgroup hold values whose difference, read as a signed number of
/// the value's width, has a magnitude in `[step, spread]` (`spread` below half the width's
/// range), or, where `same`, is 0: two threads may then hold one value, and the fact orders
/// accesses apart only where one value keeps them apart too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Inj {
    step: u64,
    spread: u64,
    same: bool,
}

impl Inj {
    /// The fact scaled by `c` in a `w`-bit domain, if no difference wraps.
    fn scaled(self, c: u64, w: u32) -> Option<Inj> {
        let spread = self.spread.checked_mul(c)?;
        (c != 0 && w >= 2 && spread < 1u64 << (w.min(64) - 1)).then_some(Inj {
            step: self.step.checked_mul(c)?,
            spread,
            same: self.same,
        })
    }

    /// The fact of the value extended from `w1` bits to more: the difference may gain or lose
    /// one `2^w1`.
    fn extended(self, w1: u32) -> Option<Inj> {
        if w1 == 0 || w1 >= 64 || self.spread >= 1u64 << (w1 - 1) {
            return None;
        }
        Some(Inj {
            step: self.step,
            spread: (1u64 << w1) - self.step,
            same: self.same,
        })
    }

    /// The fact of the value truncated to `w2` bits.
    fn truncated(self, w2: u32) -> Option<Inj> {
        ((2..=64).contains(&w2) && self.spread < 1u64 << (w2 - 1)).then_some(self)
    }

    /// The fact of a value known to lie in `[lo, hi]` (unsigned, below half its width's range),
    /// whose differences are at most `hi - lo` and survive an extension unchanged.
    fn within(self, (lo, hi): (u64, u64)) -> Option<Inj> {
        let spread = self.spread.min(hi.checked_sub(lo)?);
        (spread >= self.step).then_some(Inj {
            step: self.step,
            spread,
            same: self.same,
        })
    }
}

/// What [`AddrFacts::counter`] knows of a loop counter: its range, its start and step, and
/// whether the back edge is never taken (the counter is then its start).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Counter {
    lo: u64,
    hi: u64,
    start: Value,
    step: u64,
    once: bool,
    /// Every value is the start plus a multiple of the step: no step wraps, or the step
    /// divides the width's range.
    residue: bool,
}

/// The address facts of a body's accesses.
pub(super) struct AddrFacts<'a> {
    e: &'a Entry,
    body: &'a Body,
    layouts: &'a TypeLayouts,
    /// Uniformity among the threads of one simdgroup: a uniform value is one value in both
    /// threads of a pair.
    u: &'a Uniformity,
    /// By instruction: its block.
    home: Vec<u32>,
    /// By instruction: its canonical spelling, once computed.
    canon: Vec<Option<Value>>,
    /// The first GEP of each spelling.
    geps: BTreeMap<(TypeId, Value, Box<[Value]>), InstId>,
    /// The id of each anchor spelling.
    anchors: BTreeMap<AnchorKey, u32>,
    /// By block: [`AddrFacts::guards`], once computed.
    guards: BTreeMap<u32, Guards>,
    /// By PHI or select: the objects it may point into ([`AddrFacts::roots`]), once walked.
    merges: BTreeMap<InstId, Option<Vec<Root>>>,
}

impl<'a> AddrFacts<'a> {
    pub fn new(
        e: &'a Entry,
        body: &'a Body,
        layouts: &'a TypeLayouts,
        u: &'a Uniformity,
        m: &mut Meter,
    ) -> Result<AddrFacts<'a>, Error> {
        let n = body.insts.len();
        let mut home = Vec::new();
        m.reserve(&mut home, n)?;
        home.resize(n, u32::MAX);
        for (b, block) in body.blocks.iter() {
            for id in block.inst_ids() {
                if let Some(h) = home.get_mut(id.index()) {
                    *h = b.raw();
                }
            }
        }
        let mut canon = Vec::new();
        m.reserve(&mut canon, n)?;
        canon.resize(n, None);
        Ok(AddrFacts {
            e,
            body,
            layouts,
            u,
            home,
            canon,
            geps: BTreeMap::new(),
            anchors: BTreeMap::new(),
            guards: BTreeMap::new(),
            merges: BTreeMap::new(),
        })
    }

    fn kind(&self, v: Value) -> Option<&'a InstKind> {
        match v {
            Value::Inst(i) => self.body.insts.get(i).map(|i| &i.kind),
            _ => None,
        }
    }

    fn uniform(&self, v: Value) -> bool {
        self.u.is_uniform_value(self.e, v)
    }

    /// The facts of the address of `access`.
    pub fn of(&mut self, access: &Access, m: &mut Meter) -> Result<Addr, Error> {
        let stored_home = access.stored.and_then(|(v, _)| self.home_of(v));
        let Some(ptr) = access.ptr else {
            return Ok(Addr {
                stored_home,
                ..Addr::default()
            });
        };
        let (anchor, homes, group_homes, row) = match self.anchor(ptr, m)? {
            Some(found) => (
                Some(found.anchor),
                found.homes,
                found.group_homes,
                found.row,
            ),
            None => (None, Vec::new(), Vec::new(), None),
        };
        let mut sum = Vec::new();
        if let Some(g) = anchor.and_then(|a| a.group) {
            for x in self.summands(g.core, g.signed, m)?.into_iter().flatten() {
                m.grow(&mut sum, 1)?;
                sum.push((x, self.home_of(x)));
            }
        }
        let Guards { below, lone } = self.guards(access.block, m)?;
        let leaves = if access.stored.is_some() && !access.read {
            self.leaves(ptr, m)?
        } else {
            None
        };
        Ok(Addr {
            roots: self.roots(ptr, m)?,
            span: self.span(ptr, m)?,
            anchor,
            homes,
            stored_home,
            row,
            group_homes,
            sum,
            below,
            lone,
            leaves,
        })
    }

    /// The blocks of the instructions address `ptr` is computed from through arithmetic alone
    /// (GEPs, casts, integer and bitwise operations, comparisons, selects, and freezes): every
    /// other instruction it reaches is a leaf, and so are arguments and constants, which have no
    /// block. Where every leaf keeps one value while the access runs, so does the address. `None`
    /// when the walk reaches more than [`MAX_CHAIN`] instructions.
    fn leaves(&self, ptr: Value, m: &mut Meter) -> Result<Option<Vec<BlockId>>, Error> {
        let mut out: Vec<BlockId> = Vec::new();
        let mut seen: BTreeSet<InstId> = BTreeSet::new();
        let mut stack: Vec<Value> = Vec::new();
        m.grow(&mut stack, 1)?;
        stack.push(ptr);
        let mut charged = 0usize;
        let found = loop {
            let Some(v) = stack.pop() else {
                break true;
            };
            m.tick(1)?;
            let Value::Inst(i) = v else {
                continue;
            };
            if !seen.insert(i) {
                continue;
            }
            if seen.len() > MAX_CHAIN as usize {
                break false;
            }
            let entry = tree_bytes(1, std::mem::size_of::<InstId>());
            m.charge(entry)?;
            charged = charged.saturating_add(entry);
            match self.kind(v) {
                Some(
                    kind @ (InstKind::Gep { .. }
                    | InstKind::Cast { .. }
                    | InstKind::BinOp { .. }
                    | InstKind::ICmp { .. }
                    | InstKind::Select { .. }
                    | InstKind::Freeze(_)),
                ) => {
                    let mut n = 0usize;
                    kind.for_each_value(|_| n += 1);
                    m.grow(&mut stack, n)?;
                    kind.for_each_value(|x| stack.push(x));
                }
                _ => {
                    if let Some(h) = self.home_of(v).filter(|h| !out.contains(h)) {
                        m.grow(&mut out, 1)?;
                        out.push(h);
                    }
                }
            }
        };
        m.release(charged);
        m.release(crate::budget::vec_bytes(&stack));
        Ok(found.then_some(out))
    }

    /// The operands of `core` it is at least, in the signed order when `signed` and else the
    /// unsigned one, when it is `x + y` (an addition, or a disjoint `or`) that does not wrap in
    /// that order: its flags say so, or the operands' ranges do (a disjoint `or` wraps in
    /// neither). Without unsigned wrap the sum is at least both operands; without signed wrap,
    /// at least each operand whose partner is not negative.
    fn summands(
        &self,
        core: Value,
        signed: bool,
        m: &mut Meter,
    ) -> Result<[Option<Value>; 2], Error> {
        let Some(InstKind::BinOp {
            op: op @ (BinOp::Add | BinOp::Or),
            flags,
            a,
            b,
            ..
        }) = self.kind(core)
        else {
            return Ok([None; 2]);
        };
        if *op == BinOp::Or && !flags.contains(IntFlags::DISJOINT) {
            return Ok([None; 2]);
        }
        let Some(w) = self.width(core).filter(|&w| (1..=64).contains(&w)) else {
            return Ok([None; 2]);
        };
        let (ra, rb) = (self.range(*a, 0, m)?, self.range(*b, 0, m)?);
        // The largest value the order puts no sign on.
        let top = if signed {
            u64::MAX.checked_shr(65 - w).unwrap_or(0)
        } else {
            u64::MAX >> (64 - w)
        };
        let fits = match (ra, rb) {
            (Some((_, ha)), Some((_, hb))) => u128::from(ha) + u128::from(hb) <= u128::from(top),
            _ => false,
        };
        let flag = if signed { IntFlags::NSW } else { IntFlags::NUW };
        if !(*op == BinOp::Or || flags.contains(flag) || fits) {
            return Ok([None; 2]);
        }
        if !signed {
            return Ok([Some(*a), Some(*b)]);
        }
        let nonneg = |r: Option<(u64, u64)>| r.is_some_and(|(_, hi)| hi <= top);
        Ok([nonneg(rb).then_some(*a), nonneg(ra).then_some(*b)])
    }

    /// The facts of the branch edges that dominate block `x0`, wherever it runs: walking up its
    /// dominators, each block entered only from a conditional branch (on two distinct targets)
    /// runs only where that branch's integer comparison chose it. Its operands keep their values
    /// there: they are computed before the branch, and nothing reaches the block's dominated
    /// region from their definitions without passing the block again. A comparison that held
    /// `x < y`, signed or unsigned, gives `below`; one that held `x == c`, `x` differing between
    /// the threads of a simdgroup and `c` uniform among them, gives `lone`.
    fn guards(&mut self, x0: BlockId, m: &mut Meter) -> Result<Guards, Error> {
        let charge = |g: &Guards, m: &mut Meter| {
            m.charge(crate::budget::vec_bytes(&g.below))?;
            m.charge(crate::budget::vec_bytes(&g.lone))
        };
        if let Some(found) = self.guards.get(&x0.raw()) {
            let out = found.clone();
            charge(&out, m)?;
            return Ok(out);
        }
        use crate::air::ir::IntPredicate as P;
        let mut out = Guards::default();
        let mut d = Some(x0);
        for _ in 0..MAX_CHAIN {
            m.tick(1)?;
            let Some(x) = d else {
                break;
            };
            d = self.e.dom.idom(x);
            let mut preds = self.e.cfg.predecessors(x);
            let (Some(p), None) = (preds.next(), preds.next()) else {
                continue;
            };
            let Some(InstKind::CondBr { c, t, f, .. }) = self.body.terminator(p).map(|i| &i.kind)
            else {
                continue;
            };
            if t == f {
                continue;
            }
            let Some(InstKind::ICmp { pred, a, b }) = self.kind(*c) else {
                continue;
            };
            let held = if *t == x { *pred } else { inverse(*pred) };
            let fact = match held {
                P::Ult => (*a, *b, false),
                P::Ugt => (*b, *a, false),
                P::Slt => (*a, *b, true),
                P::Sgt => (*b, *a, true),
                P::Eq => {
                    let (a, b) = (*a, *b);
                    let apart = |i: Option<Inj>| i.is_some_and(|i| !i.same);
                    let lone = if self.uniform(b) && apart(self.inj_int(a, 0, m)?) {
                        Some((a, b))
                    } else if self.uniform(a) && apart(self.inj_int(b, 0, m)?) {
                        Some((b, a))
                    } else {
                        None
                    };
                    if let Some((x, c)) = lone {
                        m.grow(&mut out.lone, 1)?;
                        out.lone.push(Lone {
                            x,
                            c,
                            homes: [self.home_of(x), self.home_of(c)],
                        });
                    }
                    continue;
                }
                _ => continue,
            };
            m.grow(&mut out.below, 1)?;
            out.below.push(fact);
        }
        m.charge(tree_bytes(1, std::mem::size_of::<(u32, Guards)>()))?;
        charge(&out, m)?;
        self.guards.insert(x0.raw(), out.clone());
        Ok(out)
    }

    /// The block of instruction `v`.
    fn home_of(&self, v: Value) -> Option<BlockId> {
        match v {
            Value::Inst(i) => self
                .home
                .get(i.index())
                .copied()
                .filter(|&h| h != u32::MAX)
                .map(BlockId::new),
            _ => None,
        }
    }

    /// The block whose value fixes how core `core` differs between threads at any two
    /// dynamic instances: its own, or for a varying loop counter whose threads' values differ
    /// by their starts' difference modulo the step ([`AddrFacts::inj_int`]), its start's.
    fn core_home(&self, core: Value, m: &mut Meter) -> Result<Option<BlockId>, Error> {
        if let (Some(InstKind::Phi { incoming, .. }), Some(w), false) = (
            self.kind(core),
            self.width(core).filter(|&w| (2..=64).contains(&w)),
            self.uniform(core),
        ) {
            if let Some(k) = self.counter(core, incoming, w, 0, m)? {
                let steady = !k.once
                    && k.residue
                    && self
                        .inj_int(k.start, 0, m)?
                        .is_some_and(|i| i.spread < k.step);
                if steady {
                    return Ok(self.home_of(k.start));
                }
            }
        }
        Ok(self.home_of(core))
    }

    fn intern(&mut self, key: AnchorKey, m: &mut Meter) -> Result<u32, Error> {
        if let Some(&id) = self.anchors.get(&key) {
            return Ok(id);
        }
        let extra = match &key {
            AnchorKey::Gep(_, _, idx, _, _) | AnchorKey::Group(_, _, idx) => {
                std::mem::size_of_val::<[Value]>(idx)
            }
            AnchorKey::Value(_) | AnchorKey::Row(..) => 0,
        };
        m.charge(tree_bytes(
            1,
            std::mem::size_of::<(AnchorKey, u32)>() + extra,
        ))?;
        let id = u32::try_from(self.anchors.len()).unwrap_or(u32::MAX);
        self.anchors.insert(key, id);
        Ok(id)
    }

    /// The anchor of `ptr` and the blocks computing the values it is made of: constant GEP
    /// steps and casts are folded into the offset up to the last GEP with a non-constant index,
    /// whose constant trailing indices are too, and so is the constant that index adds to its
    /// core ([`AddrFacts::affine`]); any other address is its own anchor.
    fn anchor(&mut self, ptr: Value, m: &mut Meter) -> Result<Option<Anchored>, Error> {
        let mut off = 0i64;
        let mut v = ptr;
        let mut idx = Vec::new();
        for _ in 0..MAX_CHAIN {
            m.tick(1)?;
            match self.kind(v) {
                Some(InstKind::Cast {
                    op: CastOp::Bitcast,
                    v: base,
                }) => v = *base,
                Some(InstKind::Gep {
                    src_ty,
                    base,
                    indices,
                    ..
                }) => {
                    let konst = |k: &Value| match k {
                        Value::Const(c) => const_int(self.e, *c),
                        _ => None,
                    };
                    let split = indices.iter().rposition(|k| konst(k).is_none());
                    idx.clear();
                    for k in indices.get(split.map_or(0, |p| p + 1)..).unwrap_or(&[]) {
                        let Some(k) = konst(k) else {
                            return Ok(None);
                        };
                        m.grow(&mut idx, 1)?;
                        idx.push(k);
                    }
                    let Some(p) = split else {
                        // Every index is constant: a constant step from the base.
                        match self
                            .gep_offset(*src_ty, &idx)
                            .and_then(|s| off.checked_add(s))
                        {
                            Some(o) => off = o,
                            None => return Ok(None),
                        }
                        v = *base;
                        continue;
                    };
                    let (Some(bits), Some(stride), Some(&k)) = (
                        self.index_bits(v),
                        self.stride_at(*src_ty, indices, p),
                        indices.get(p),
                    ) else {
                        return Ok(None);
                    };
                    let (core, ext, c) = self.affine(k, bits, m)?;
                    let shifted = self
                        .type_after(*src_ty, indices, p)
                        .and_then(|ty| self.steps(ty, &idx))
                        .and_then(|s| off.checked_add(s))
                        .zip(i64::try_from(stride).ok().and_then(|st| c.checked_mul(st)))
                        .and_then(|(o, d)| o.checked_add(d));
                    let Some(off) = shifted else {
                        return Ok(None);
                    };
                    let row = self.row((*src_ty, *base, indices), p, (k, core, ext), off, m)?;
                    let prefix = indices.get(..=p).unwrap_or(&[]);
                    let spread = self.gep_spread(*src_ty, *base, prefix, bits, 0, m)?;
                    let cbase = self.canon(*base, 0, m)?;
                    let before = indices.get(..p).unwrap_or(&[]);
                    m.charge(std::mem::size_of_val::<[Value]>(before))?;
                    let key = AnchorKey::Gep(*src_ty, cbase, before.into(), core, ext);
                    let id = self.intern(key, m)?;
                    // A group compares cores from one address prefix, which must then be one
                    // value in every thread of the simdgroup.
                    let shared = self.uniform(*base) && before.iter().all(|&x| self.uniform(x));
                    let group = match ext {
                        Ext::Zero | Ext::Sign if shared => {
                            m.charge(std::mem::size_of_val::<[Value]>(before))?;
                            let g = AnchorKey::Group(*src_ty, cbase, before.into());
                            Some(Group {
                                id: self.intern(g, m)?,
                                core,
                                stride,
                                signed: ext == Ext::Sign,
                            })
                        }
                        _ => None,
                    };
                    let mut group_homes = Vec::new();
                    for x in before.iter().copied().chain([*base]) {
                        if let Some(h) = self.home_of(x).filter(|h| !group_homes.contains(h)) {
                            m.grow(&mut group_homes, 1)?;
                            group_homes.push(h);
                        }
                    }
                    let mut homes = Vec::new();
                    m.reserve(&mut homes, group_homes.len() + 1)?;
                    homes.extend(group_homes.iter().copied());
                    if let Some(h) = self.core_home(core, m)?.filter(|h| !homes.contains(h)) {
                        homes.push(h);
                    }
                    let anchor = Anchor {
                        id,
                        off,
                        spread,
                        group,
                    };
                    return Ok(Some(Anchored {
                        anchor,
                        homes,
                        group_homes,
                        row,
                    }));
                }
                _ => {
                    let c = self.canon(v, 0, m)?;
                    let spread = self.ptr_spread(v, 0, m)?;
                    let id = self.intern(AnchorKey::Value(c), m)?;
                    let homes: Vec<BlockId> = self.home_of(v).into_iter().collect();
                    m.charge(crate::budget::vec_bytes(&homes))?;
                    let anchor = Anchor {
                        id,
                        off,
                        spread,
                        group: None,
                    };
                    return Ok(Some(Anchored {
                        anchor,
                        homes,
                        group_homes: Vec::new(),
                        row: None,
                    }));
                }
            }
        }
        Ok(None)
    }

    /// The [`Row`] of a GEP over `src_ty` from `base` by `indices` whose last non-constant
    /// index is `at`, that index `k` being `ext(core)` plus a constant, and `off` the constant
    /// bytes the address adds to `stride * ext(core)` past the GEP's earlier steps: index `at`
    /// must step through an array, every earlier non-constant index by a multiple of its size
    /// (their constant steps fold into `off` modulo it), and the core must be bounded. The base
    /// must be one value in every thread of the simdgroup: the offsets into the array are
    /// compared from it.
    fn row(
        &mut self,
        (src_ty, base, indices): (TypeId, Value, &[Value]),
        at: usize,
        (k, core, ext): (Value, Value, Ext),
        off: i64,
        m: &mut Meter,
    ) -> Result<Option<Row>, Error> {
        let types = self.e.module.types();
        let Some(prev) = at.checked_sub(1) else {
            return Ok(None);
        };
        if !self.uniform(base) {
            return Ok(None);
        }
        let size = self
            .type_after(src_ty, indices, prev)
            .and_then(|ty| match types.get(ty) {
                Some(Type::Array { .. } | Type::Vector { .. }) => self.layouts.alloc_size(ty),
                _ => None,
            });
        let (Some(size), Some(stride), Some(bits)) = (
            size.filter(|&s| s > 0 && s < 1 << 40),
            self.stride_at(src_ty, indices, at),
            self.index_bits(base),
        ) else {
            return Ok(None);
        };
        // The earlier steps: constant ones summed, the others multiples of the array's size.
        let mut fixed = 0i128;
        let mut ty = src_ty;
        for (i, &x) in indices.get(..at).unwrap_or(&[]).iter().enumerate() {
            m.tick(1)?;
            let konst = match x {
                Value::Const(c) => const_int(self.e, c),
                _ => None,
            };
            let (step, next) = if i == 0 {
                (self.layouts.alloc_size(ty), ty)
            } else {
                match types.get(ty) {
                    Some(Type::Struct { fields, .. }) => {
                        let Some(f) = konst.and_then(|f| usize::try_from(f).ok()) else {
                            return Ok(None);
                        };
                        let (Some(&o), Some(&field)) = (
                            self.layouts.field_offsets(ty).and_then(|o| o.get(f)),
                            fields.get(f),
                        ) else {
                            return Ok(None);
                        };
                        fixed += i128::from(o);
                        ty = field;
                        continue;
                    }
                    Some(Type::Array { elem, .. } | Type::Vector { elem, .. }) => {
                        (self.layouts.alloc_size(*elem), *elem)
                    }
                    _ => return Ok(None),
                }
            };
            let Some(step) = step else {
                return Ok(None);
            };
            match konst {
                Some(c) => fixed += i128::from(c) * i128::from(step),
                None if step % size == 0 => {}
                None => return Ok(None),
            }
            ty = next;
        }
        // The core's values as the index reads them: not negative where it is extended by sign
        // or already has the index's width.
        let Some(w1) = self.width(core) else {
            return Ok(None);
        };
        let Some((lo, hi)) = self
            .range(core, 0, m)?
            .filter(|&(_, hi)| ext == Ext::Zero || hi < 1u64 << (w1 - 1).min(62))
        else {
            return Ok(None);
        };
        let off = i128::from(off) + fixed.rem_euclid(i128::from(size));
        let stride_bytes = stride;
        let (stride, lo, hi) = (i128::from(stride), i128::from(lo), i128::from(hi));
        let (Ok(o), Ok(lo), Ok(hi)) = (
            i64::try_from(off),
            i64::try_from(stride * lo + off),
            i64::try_from(stride * hi + off),
        ) else {
            return Ok(None);
        };
        let spread = if self.uniform(core) {
            Some(Spread::Same)
        } else {
            self.index_inj(k, bits, 0, m)?
                .and_then(|i| i.scaled(u64::try_from(stride).ok()?, bits))
                .map(|i| Spread::Apart {
                    min: i.step,
                    max: i.spread,
                    same: i.same,
                })
        };
        let cbase = self.canon(base, 0, m)?;
        let id = self.intern(AnchorKey::Row(cbase, size, stride_bytes, core, ext), m)?;
        let mut homes = Vec::new();
        for h in [self.home_of(base), self.core_home(core, m)?] {
            if let Some(h) = h.filter(|h| !homes.contains(h)) {
                m.grow(&mut homes, 1)?;
                homes.push(h);
            }
        }
        Ok(Some(Row {
            id,
            size,
            off: o,
            lo,
            hi,
            spread,
            homes,
        }))
    }

    /// The bytes index `at` of a GEP over `src_ty` with `indices` steps by: the source type's
    /// size for the first index, an array or vector element's for a later one.
    fn stride_at(&self, src_ty: TypeId, indices: &[Value], at: usize) -> Option<u64> {
        match at {
            0 => self.layouts.alloc_size(src_ty),
            _ => {
                let ty = self.type_after(src_ty, indices, at - 1)?;
                match self.e.module.types().get(ty)? {
                    Type::Array { elem, .. } | Type::Vector { elem, .. } => {
                        self.layouts.alloc_size(*elem)
                    }
                    _ => None,
                }
            }
        }
    }

    /// GEP index `k`, for pointers that index with `bits` bits, as `ext(core) + c` exactly: an
    /// addition or subtraction of a constant, or a disjoint `or` with one, is peeled in the index
    /// width (where address arithmetic wraps alike), and inside an extension from a narrower
    /// width (the GEP's own sign extension of a narrower index included) where it does not wrap
    /// there ([`AddrFacts::peel_narrow`]).
    fn affine(&self, k: Value, bits: u32, m: &mut Meter) -> Result<(Value, Ext, i64), Error> {
        let mut v = self.settle(k, m)?;
        let mut c = 0i64;
        for _ in 0..MAX_CHAIN {
            m.tick(1)?;
            let Some(w) = self.width(v) else {
                break;
            };
            if w < bits {
                let (z, d) = self.peel_narrow(v, true, m)?;
                return Ok(match c.checked_add(d) {
                    Some(c) => (self.settle(z, m)?, Ext::Sign, c),
                    None => (v, Ext::Sign, c),
                });
            }
            if w > bits {
                break;
            }
            if let Some((x, d, _)) = self.plus_const(v) {
                match c.checked_add(d) {
                    Some(n) => {
                        c = n;
                        v = self.settle(x, m)?;
                        continue;
                    }
                    None => break,
                }
            }
            if let Some(InstKind::Cast {
                op: op @ (CastOp::ZExt | CastOp::SExt),
                v: y,
            }) = self.kind(v)
            {
                let signed = *op == CastOp::SExt;
                let y = self.settle(*y, m)?;
                let (z, d) = self.peel_narrow(y, signed, m)?;
                let ext = if signed { Ext::Sign } else { Ext::Zero };
                return Ok(match c.checked_add(d) {
                    Some(c) => (self.settle(z, m)?, ext, c),
                    None => (y, ext, c),
                });
            }
            break;
        }
        Ok((v, Ext::None, c))
    }

    /// `v` as `x + d` for a constant `d`, with the operation's flags: an addition, a subtraction
    /// of a constant, or a disjoint `or`.
    fn plus_const(&self, v: Value) -> Option<(Value, i64, IntFlags)> {
        let konst = |x: Value| match x {
            Value::Const(c) => const_int(self.e, c),
            _ => None,
        };
        match self.kind(v)? {
            InstKind::BinOp {
                op: op @ (BinOp::Add | BinOp::Or),
                flags,
                a,
                b,
                ..
            } if *op == BinOp::Add || flags.contains(IntFlags::DISJOINT) => {
                match (konst(*a), konst(*b)) {
                    (_, Some(d)) => Some((*a, d, *flags)),
                    (Some(d), None) => Some((*b, d, *flags)),
                    _ => None,
                }
            }
            InstKind::BinOp {
                op: BinOp::Sub,
                flags,
                a,
                b,
                ..
            } => konst(*b)
                .and_then(|d| d.checked_neg())
                .map(|d| (*a, d, *flags)),
            _ => None,
        }
    }

    /// Narrow value `y` as `z + d` exactly as an unsigned (or, when `signed`, a signed) number:
    /// constants peeled while the step cannot wrap in `y`'s width, as its flags state (`nuw`
    /// adding a non-negative constant or subtracting one, `nsw`, a disjoint `or`) or as the
    /// range of its other operand shows.
    fn peel_narrow(&self, y: Value, signed: bool, m: &mut Meter) -> Result<(Value, i64), Error> {
        let mut v = y;
        let mut d = 0i64;
        for _ in 0..MAX_CHAIN {
            m.tick(1)?;
            let (Some(w), Some((x, c, flags))) = (self.width(v), self.plus_const(v)) else {
                break;
            };
            let is_sub = matches!(self.kind(v), Some(InstKind::BinOp { op: BinOp::Sub, .. }));
            // A disjoint `or` adds without a carry: exact as a signed number, and as an unsigned
            // one when the constant is not negative.
            let stated = if signed {
                flags.contains(IntFlags::NSW) || flags.contains(IntFlags::DISJOINT)
            } else {
                (flags.contains(IntFlags::NUW) || flags.contains(IntFlags::DISJOINT))
                    && if is_sub { c <= 0 } else { c >= 0 }
            };
            if !(stated || self.no_wrap(x, c, w, signed, m)?) {
                break;
            }
            match d.checked_add(c) {
                Some(n) => {
                    d = n;
                    v = x;
                }
                None => break,
            }
        }
        Ok((v, d))
    }

    /// Whether `x + c` stays in the `w`-bit range (unsigned, or signed when `signed`) for
    /// every value of `x`'s range.
    fn no_wrap(
        &self,
        x: Value,
        c: i64,
        w: u32,
        signed: bool,
        m: &mut Meter,
    ) -> Result<bool, Error> {
        let Some((lo, hi)) = self.range(x, 0, m)? else {
            return Ok(false);
        };
        if !(1..=64).contains(&w) {
            return Ok(false);
        }
        let (lo, hi, c) = (i128::from(lo), i128::from(hi), i128::from(c));
        let (min, max) = if signed {
            if hi >= 1i128 << (w - 1) {
                return Ok(false);
            }
            (-(1i128 << (w - 1)), (1i128 << (w - 1)) - 1)
        } else {
            (0, (1i128 << w) - 1)
        };
        Ok(lo + c >= min && hi + c <= max)
    }

    /// The width of integer value `v`.
    fn width(&self, v: Value) -> Option<u32> {
        int_width(self.e, self.e.module.value_ty(self.e.func, v))
    }

    /// The unsigned range of integer `v` in its width, where the IR bounds it: a constant, a
    /// thread-index builtin (a threadgroup has at most 1024 threads and 32 simdgroups, a
    /// simdgroup 32 threads), a loop counter ([`AddrFacts::counter`]), and exact steps from one
    /// (extensions, a truncation that keeps it, a mask, a remainder or right shift by a
    /// constant, and an addition, subtraction, scaling, or left shift by a constant that does
    /// not wrap).
    fn range(&self, v: Value, depth: u32, m: &mut Meter) -> Result<Option<(u64, u64)>, Error> {
        if depth >= MAX_CHAIN {
            return Ok(None);
        }
        m.tick(1)?;
        let Some(w) = self.width(v).filter(|&w| (1..=64).contains(&w)) else {
            return Ok(None);
        };
        let top = u64::MAX >> (64 - w);
        let fit = |r: (u64, u64)| (r.1 <= top).then_some(r);
        let unsigned = |x: Value| match x {
            Value::Const(c) => const_int(self.e, c).map(|k| (k as u64) & top),
            _ => None,
        };
        Ok(match v {
            Value::Const(_) => unsigned(v).map(|k| (k, k)),
            Value::Arg(a) => self
                .e
                .ep
                .params
                .iter()
                .find(|p| p.param == a.raw())
                .and_then(|p| match &p.role {
                    ParamRole::Builtin(
                        Builtin::ThreadIndexInSimdgroup | Builtin::SimdgroupIndexInThreadgroup,
                    ) => Some((0, 31)),
                    ParamRole::Builtin(
                        Builtin::ThreadIndexInThreadgroup | Builtin::ThreadPositionInThreadgroup,
                    ) => Some((0, 1023)),
                    _ => None,
                })
                .and_then(fit),
            Value::Inst(_) => match self.kind(v) {
                Some(InstKind::Cast {
                    op: CastOp::ZExt,
                    v: x,
                })
                | Some(InstKind::Freeze(x)) => self.range(*x, depth + 1, m)?,
                Some(InstKind::Cast {
                    op: CastOp::SExt,
                    v: x,
                }) => {
                    let w1 = self.width(*x).unwrap_or(0);
                    self.range(*x, depth + 1, m)?
                        .filter(|r| w1 >= 1 && r.1 < 1u64 << (w1 - 1))
                }
                Some(InstKind::Cast {
                    op: CastOp::Trunc,
                    v: x,
                }) => self.range(*x, depth + 1, m)?.and_then(fit),
                Some(InstKind::Phi { incoming, .. }) => self
                    .counter(v, incoming, w, depth, m)?
                    .map(|k| (k.lo, k.hi))
                    .and_then(fit),
                Some(InstKind::BinOp {
                    op, flags, a, b, ..
                }) => {
                    let (ka, kb) = (unsigned(*a), unsigned(*b));
                    let (x, k) = match (ka, kb) {
                        (_, Some(k)) => (*a, k),
                        (Some(k), None)
                            if !matches!(
                                op,
                                BinOp::Sub | BinOp::URem | BinOp::LShr | BinOp::Shl
                            ) =>
                        {
                            (*b, k)
                        }
                        _ => return Ok(None),
                    };
                    match op {
                        BinOp::And => {
                            let hi = self.range(x, depth + 1, m)?.map_or(k, |r| r.1.min(k));
                            Some((0, hi))
                        }
                        BinOp::URem if k > 0 => {
                            let hi = self
                                .range(x, depth + 1, m)?
                                .map_or(k - 1, |r| r.1.min(k - 1));
                            Some((0, hi))
                        }
                        BinOp::LShr if k < u64::from(w) => Some(
                            self.range(x, depth + 1, m)?
                                .map_or((0, top >> k), |r| (r.0 >> k, r.1 >> k)),
                        ),
                        BinOp::Add => self
                            .range(x, depth + 1, m)?
                            .and_then(|r| Some((r.0.checked_add(k)?, r.1.checked_add(k)?)))
                            .and_then(fit),
                        BinOp::Or if flags.contains(IntFlags::DISJOINT) => self
                            .range(x, depth + 1, m)?
                            .and_then(|r| Some((r.0.checked_add(k)?, r.1.checked_add(k)?)))
                            .and_then(fit),
                        BinOp::Sub => self
                            .range(x, depth + 1, m)?
                            .and_then(|r| Some((r.0.checked_sub(k)?, r.1.checked_sub(k)?))),
                        BinOp::Mul => self
                            .range(x, depth + 1, m)?
                            .and_then(|r| Some((r.0.checked_mul(k)?, r.1.checked_mul(k)?)))
                            .and_then(fit),
                        BinOp::Shl if k < u64::from(w) => self
                            .range(x, depth + 1, m)?
                            .and_then(|r| {
                                Some((r.0.checked_mul(1 << k)?, r.1.checked_mul(1 << k)?))
                            })
                            .and_then(fit),
                        _ => None,
                    }
                }
                _ => None,
            },
        })
    }

    /// Loop counter `phi` (a `w`-bit PHI with `incoming`): one incoming value is the start, the
    /// other `phi + c` for a constant `c > 0` from a latch whose conditional branch back to the
    /// PHI's block tests the PHI or that next value against a constant `n` in the latch itself.
    /// A tested value the branch lets through is
    ///
    /// - below `n` (or at most `n`) unsigned;
    /// - below `n` (or at most `n`) signed, `n` not negative, where the start lies below half
    ///   the width's range and the largest tested value plus `c` stays there too (so by
    ///   induction no value is negative);
    /// - unequal to `n`, stepping by one from a start below `n`, so it reaches `n` before any
    ///   value past it.
    ///
    /// The PHI holds the start, or a next value the branch let through (testing the next
    /// value), or one more step from a PHI it let through (testing the PHI). When no start lets
    /// the branch through, the back edge is never taken and the PHI is its start. The start must
    /// be computed in a block that strictly dominates the PHI's (or be an argument or constant).
    fn counter(
        &self,
        phi: Value,
        incoming: &[(Value, BlockId)],
        w: u32,
        depth: u32,
        m: &mut Meter,
    ) -> Result<Option<Counter>, Error> {
        let top = u64::MAX >> (64 - w);
        let half = 1u64 << (w - 1);
        let [(x0, b0), (x1, b1)] = incoming else {
            return Ok(None);
        };
        let step = |x: Value| {
            self.plus_const(x)
                .filter(|&(base, c, _)| base == phi && c > 0)
                .and_then(|(_, c, _)| u64::try_from(c).ok())
        };
        let ((start, from), (next, latch, c)) = match (step(*x0), step(*x1)) {
            (None, Some(c)) => ((*x0, *b0), (*x1, *b1, c)),
            (Some(c), None) => ((*x1, *b1), (*x0, *b0, c)),
            _ => return Ok(None),
        };
        let (Some(header), true) = (self.home_of(phi), from != latch) else {
            return Ok(None);
        };
        // The start is computed before the loop is entered: wherever the counter is live, the
        // start holds the value the counter started from.
        if self
            .home_of(start)
            .is_some_and(|d| !self.e.dom.strictly_dominates(d, header))
        {
            return Ok(None);
        }
        let Some(InstKind::CondBr { c: cond, t, f, .. }) =
            self.body.terminator(latch).map(|i| &i.kind)
        else {
            return Ok(None);
        };
        if (*t == header) == (*f == header) || self.home_of(*cond) != Some(latch) {
            return Ok(None);
        }
        let Some(InstKind::ICmp { pred, a, b }) = self.kind(*cond) else {
            return Ok(None);
        };
        let konst = |x: Value| match x {
            Value::Const(k) => const_int(self.e, k).map(|k| (k as u64) & top),
            _ => None,
        };
        let tested = |x: Value| x == next || x == phi;
        let (pred, n, x) = match (konst(*a), konst(*b)) {
            (None, Some(n)) if tested(*a) => (*pred, n, *a),
            (Some(n), None) if tested(*b) => (swapped(*pred), n, *b),
            _ => return Ok(None),
        };
        let stay = if *t == header { pred } else { inverse(pred) };
        let Some((lo, hi)) = self.range(start, depth + 1, m)? else {
            return Ok(None);
        };
        use crate::air::ir::IntPredicate as P;
        // The largest tested value the branch lets through (`None`: none).
        let bound = match stay {
            P::Ult => n.checked_sub(1),
            P::Ule => Some(n),
            P::Slt | P::Sle if n < half && hi < half => {
                let b = if stay == P::Slt {
                    n.checked_sub(1)
                } else {
                    Some(n)
                };
                if b.unwrap_or(0)
                    .max(hi)
                    .checked_add(c)
                    .is_none_or(|x| x >= half)
                {
                    return Ok(None);
                }
                b
            }
            P::Ne if c == 1 && hi < n => n.checked_sub(1),
            _ => return Ok(None),
        };
        // No step past the largest value wraps.
        let exact = bound
            .unwrap_or(0)
            .max(hi)
            .checked_add(c)
            .is_some_and(|x| x <= top);
        let once = match bound {
            None => true,
            Some(b) if x == phi => b < lo,
            Some(b) => exact && lo.checked_add(c).is_some_and(|x| x > b),
        };
        if once {
            return Ok(Some(Counter {
                lo,
                hi,
                start,
                step: c,
                once: true,
                residue: true,
            }));
        }
        let Some(b) = bound else {
            return Ok(None);
        };
        let hi = if x == phi {
            if !exact {
                return Ok(None);
            }
            b.saturating_add(c).max(hi)
        } else {
            b.max(hi)
        };
        Ok(Some(Counter {
            lo: if exact { lo } else { 0 },
            hi,
            start,
            step: c,
            once: false,
            residue: exact || c.is_power_of_two(),
        }))
    }

    /// The start of `v` when it is a loop counter whose back edge is never taken
    /// ([`AddrFacts::counter`]), followed through such counters; else `v`.
    fn settle(&self, mut v: Value, m: &mut Meter) -> Result<Value, Error> {
        for _ in 0..MAX_CHAIN {
            m.tick(1)?;
            let (Some(InstKind::Phi { incoming, .. }), Some(w)) = (
                self.kind(v),
                self.width(v).filter(|&w| (1..=64).contains(&w)),
            ) else {
                break;
            };
            match self.counter(v, incoming, w, 0, m)? {
                Some(Counter {
                    start, once: true, ..
                }) => v = start,
                _ => break,
            }
        }
        Ok(v)
    }

    /// The object a constant address points into.
    fn const_root(&self, mut c: ConstId, m: &mut Meter) -> Result<Option<Root>, Error> {
        for _ in 0..MAX_CHAIN {
            m.tick(1)?;
            match self.e.module.consts().get(c).map(|k| &k.value) {
                Some(Const::GlobalAddr(g)) => return Ok(Some(Root::Global(*g))),
                Some(Const::Expr(ConstExpr::Gep { ops, .. })) => match ops.first() {
                    Some(&base) => c = base,
                    None => return Ok(None),
                },
                Some(Const::Expr(ConstExpr::Cast {
                    op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                    ops: [base],
                })) => c = *base,
                _ => return Ok(None),
            }
        }
        Ok(None)
    }

    /// The objects `ptr` may point into: every object its address steps, PHIs, and selects lead
    /// back to; `None` when one leads to anything else (a loaded or computed address, null).
    /// The answer for the PHI or select the address steps from is kept, so the reads through
    /// one pointer merge walk its incoming values once, however many there are (the contract, which
    /// the device pass's reads of every buffer access must keep).
    fn roots(&mut self, ptr: Value, m: &mut Meter) -> Result<Option<Vec<Root>>, Error> {
        let mut base = ptr;
        for _ in 0..MAX_CHAIN {
            m.tick(1)?;
            match self.kind(base) {
                Some(InstKind::Gep { base: b, .. }) => base = *b,
                Some(InstKind::Cast {
                    op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                    v,
                })
                | Some(InstKind::Freeze(v)) => base = *v,
                _ => break,
            }
        }
        let merge = match (base, self.kind(base)) {
            (Value::Inst(i), Some(InstKind::Phi { .. } | InstKind::Select { .. })) => i,
            _ => return self.walk_roots(ptr, m),
        };
        if let Some(found) = self.merges.get(&merge) {
            return Ok(found.clone());
        }
        let found = self.walk_roots(base, m)?;
        m.charge(
            tree_bytes(1, std::mem::size_of::<(InstId, Option<Vec<Root>>)>())
                + found.as_ref().map_or(0, crate::budget::vec_bytes),
        )?;
        self.merges.insert(merge, found.clone());
        Ok(found)
    }

    /// [`AddrFacts::roots`] by walking every step from `ptr`; a merge whose answer is kept is
    /// read from it.
    fn walk_roots(&mut self, ptr: Value, m: &mut Meter) -> Result<Option<Vec<Root>>, Error> {
        let mut out = Vec::new();
        let mut stack = vec![ptr];
        let mut seen = BTreeSet::new();
        let mut charged = 0usize;
        let result = 'walk: loop {
            let Some(v) = stack.pop() else {
                break Some(());
            };
            m.tick(1)?;
            if let Value::Inst(i) = v {
                if let Some(kept) = self.merges.get(&i) {
                    let Some(kept) = kept else {
                        break None;
                    };
                    for &r in kept {
                        if !out.contains(&r) {
                            if out.len() == MAX_ROOTS {
                                break 'walk None;
                            }
                            out.push(r);
                        }
                    }
                    continue;
                }
            }
            let found = match v {
                Value::Arg(a) => Some(Root::Param(a.raw())),
                Value::Const(c) => match self.const_root(c, m)? {
                    Some(r) => Some(r),
                    None => break None,
                },
                Value::Inst(i) => {
                    if !seen.insert(i) {
                        continue;
                    }
                    let entry = tree_bytes(1, std::mem::size_of::<InstId>());
                    m.charge(entry)?;
                    charged = charged.saturating_add(entry);
                    // A device pointer loaded from memory: the object its source names.
                    if let Some(InstKind::Load { ptr: src, .. }) = self.kind(v) {
                        if space_ptr(self.e, v, Space::Device) {
                            let at = self.canon(*src, 0, m)?;
                            if !out.contains(&Root::Loaded(at)) {
                                if out.len() == MAX_ROOTS {
                                    break None;
                                }
                                out.push(Root::Loaded(at));
                            }
                            continue;
                        }
                    }
                    match self.kind(v) {
                        Some(InstKind::Gep { base, .. }) => stack.push(*base),
                        Some(InstKind::Cast {
                            op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                            v,
                        })
                        | Some(InstKind::Freeze(v)) => stack.push(*v),
                        Some(InstKind::Phi { incoming, .. }) => {
                            stack.extend(incoming.iter().map(|&(v, _)| v))
                        }
                        Some(InstKind::Select { t, f, .. }) => {
                            stack.push(*t);
                            stack.push(*f);
                        }
                        _ => break None,
                    }
                    None
                }
            };
            if let Some(r) = found {
                if !out.contains(&r) {
                    if out.len() == MAX_ROOTS {
                        break None;
                    }
                    out.push(r);
                }
            }
        };
        m.release(charged);
        Ok(result.map(|()| {
            out.sort_unstable();
            out
        }))
    }

    /// The constant byte offset of the GEP over `src_ty` with constant indices `idx`.
    fn gep_offset(&self, src_ty: TypeId, idx: &[i64]) -> Option<i64> {
        let Some((&first, rest)) = idx.split_first() else {
            return Some(0);
        };
        let size = i64::try_from(self.layouts.alloc_size(src_ty)?).ok()?;
        first
            .checked_mul(size)?
            .checked_add(self.steps(src_ty, rest)?)
    }

    /// The constant byte offset of stepping into `ty` by constant indices `idx` (a GEP's indices
    /// after its first).
    fn steps(&self, mut ty: TypeId, idx: &[i64]) -> Option<i64> {
        let types = self.e.module.types();
        let mut off = 0i64;
        for &k in idx {
            match types.get(ty)? {
                Type::Struct { fields, .. } => {
                    let f = usize::try_from(k).ok()?;
                    let at = i64::try_from(*self.layouts.field_offsets(ty)?.get(f)?).ok()?;
                    off = off.checked_add(at)?;
                    ty = *fields.get(f)?;
                }
                Type::Array { elem, .. } | Type::Vector { elem, .. } => {
                    let size = i64::try_from(self.layouts.alloc_size(*elem)?).ok()?;
                    off = off.checked_add(k.checked_mul(size)?)?;
                    ty = *elem;
                }
                _ => return None,
            }
        }
        Some(off)
    }

    /// The type index `at` of a GEP over `src_ty` with `indices` reaches: `src_ty` for the first
    /// index, then a step into it per index, through constant struct indices.
    fn type_after(&self, src_ty: TypeId, indices: &[Value], at: usize) -> Option<TypeId> {
        let types = self.e.module.types();
        let mut ty = src_ty;
        for k in indices.get(1..=at)? {
            ty = match types.get(ty)? {
                Type::Struct { fields, .. } => {
                    let Value::Const(c) = k else {
                        return None;
                    };
                    *fields.get(usize::try_from(const_int(self.e, *c)?).ok()?)?
                }
                Type::Array { elem, .. } | Type::Vector { elem, .. } => *elem,
                _ => return None,
            };
        }
        Some(ty)
    }

    /// One object and the byte offsets into it `ptr` may take, when every step of the address
    /// is a GEP whose indices are constants or bounded ([`AddrFacts::range`]), or a cast.
    /// Offsets outside `[0, 2^31)` are left unknown (no threadgroup object is that large, and
    /// the pointer's own width may wrap them).
    fn span(&self, ptr: Value, m: &mut Meter) -> Result<Option<Span>, Error> {
        let (mut lo, mut hi) = (0i64, 0i64);
        let mut v = ptr;
        let mut idx = Vec::new();
        for _ in 0..MAX_CHAIN {
            m.tick(1)?;
            let step = match v {
                Value::Arg(a) => return Ok(within(Root::Param(a.raw()), lo, hi)),
                Value::Const(c) => match self.e.module.consts().get(c).map(|k| &k.value) {
                    Some(Const::GlobalAddr(g)) => return Ok(within(Root::Global(*g), lo, hi)),
                    Some(Const::Expr(ConstExpr::Gep { src_ty, ops, .. })) => {
                        let Some((&base, rest)) = ops.split_first() else {
                            return Ok(None);
                        };
                        idx.clear();
                        for &k in rest {
                            let Some(k) = const_int(self.e, k) else {
                                return Ok(None);
                            };
                            m.grow(&mut idx, 1)?;
                            idx.push(k);
                        }
                        v = Value::Const(base);
                        self.gep_offset(*src_ty, &idx).map(|o| (o, o))
                    }
                    Some(Const::Expr(ConstExpr::Cast {
                        op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                        ops: [base],
                    })) => {
                        v = Value::Const(*base);
                        Some((0, 0))
                    }
                    _ => return Ok(None),
                },
                Value::Inst(_) => match self.kind(v) {
                    Some(InstKind::Gep {
                        src_ty,
                        base,
                        indices,
                        ..
                    }) => {
                        let Some(bits) = self.index_bits(v) else {
                            return Ok(None);
                        };
                        let step = self.gep_span(*src_ty, indices, bits, m)?;
                        v = *base;
                        step
                    }
                    Some(InstKind::Cast {
                        op: CastOp::Bitcast | CastOp::AddrSpaceCast,
                        v: base,
                    }) => {
                        v = *base;
                        Some((0, 0))
                    }
                    _ => return Ok(None),
                },
            };
            match step.and_then(|(a, b)| Some((lo.checked_add(a)?, hi.checked_add(b)?))) {
                Some((a, b)) => (lo, hi) = (a, b),
                None => return Ok(None),
            }
        }
        m.release(crate::budget::vec_bytes(&idx));
        Ok(None)
    }

    /// The byte offsets `lo..=hi` a GEP over `src_ty` by `indices` (with `bits`-bit pointer
    /// indices) adds to its base: per index, its stride times every value its range allows. An
    /// index is read as the signed number the GEP extends, so its range must lie below half its
    /// width's; a struct index must be constant.
    fn gep_span(
        &self,
        src_ty: TypeId,
        indices: &[Value],
        bits: u32,
        m: &mut Meter,
    ) -> Result<Option<(i64, i64)>, Error> {
        let types = self.e.module.types();
        let (mut lo, mut hi) = (0i64, 0i64);
        let mut ty = src_ty;
        for (at, &k) in indices.iter().enumerate() {
            m.tick(1)?;
            let konst = match k {
                Value::Const(c) => const_int(self.e, c),
                _ => None,
            };
            let (stride, next) = if at == 0 {
                (self.layouts.alloc_size(ty), Some(ty))
            } else {
                match types.get(ty) {
                    Some(Type::Struct { fields, .. }) => {
                        let Some(f) = konst.and_then(|f| usize::try_from(f).ok()) else {
                            return Ok(None);
                        };
                        let at = self
                            .layouts
                            .field_offsets(ty)
                            .and_then(|o| o.get(f).copied())
                            .and_then(|o| i64::try_from(o).ok());
                        match (at, fields.get(f)) {
                            (Some(o), Some(&field)) => {
                                (lo, hi) = match lo.checked_add(o).zip(hi.checked_add(o)) {
                                    Some(r) => r,
                                    None => return Ok(None),
                                };
                                ty = field;
                                continue;
                            }
                            _ => return Ok(None),
                        }
                    }
                    Some(Type::Array { elem, .. } | Type::Vector { elem, .. }) => {
                        (self.layouts.alloc_size(*elem), Some(*elem))
                    }
                    _ => return Ok(None),
                }
            };
            let (Some(stride), Some(next)) = (stride.and_then(|s| i64::try_from(s).ok()), next)
            else {
                return Ok(None);
            };
            ty = next;
            let range = match konst {
                Some(c) => Some((c, c)),
                None => self.index_range(k, bits, m)?,
            };
            let step = range.and_then(|(a, b)| {
                let (a, b) = (a.checked_mul(stride)?, b.checked_mul(stride)?);
                Some((a.min(b), a.max(b)))
            });
            match step.and_then(|(a, b)| Some((lo.checked_add(a)?, hi.checked_add(b)?))) {
                Some(r) => (lo, hi) = r,
                None => return Ok(None),
            }
        }
        Ok(Some((lo, hi)))
    }

    /// The values GEP index `k` may take as the signed number the GEP reads (extended or
    /// truncated to `bits`), when its range lies below half its width's.
    fn index_range(&self, k: Value, bits: u32, m: &mut Meter) -> Result<Option<(i64, i64)>, Error> {
        let Some(w) = self.width(k).filter(|&w| w >= 2 && w <= bits) else {
            return Ok(None);
        };
        Ok(self
            .range(k, 0, m)?
            .filter(|&(_, hi)| hi < 1u64 << (w - 1).min(62))
            .and_then(|(lo, hi)| Some((i64::try_from(lo).ok()?, i64::try_from(hi).ok()?))))
    }

    /// The canonical spelling of `v`: casts that keep the address are looked through, and GEPs
    /// with the same source type, canonical base, and index values are one spelling.
    fn canon(&mut self, v: Value, depth: u32, m: &mut Meter) -> Result<Value, Error> {
        let Value::Inst(i) = v else {
            return Ok(v);
        };
        if let Some(c) = self.canon.get(i.index()).copied().flatten() {
            return Ok(c);
        }
        m.tick(1)?;
        let c = match self.kind(v) {
            Some(InstKind::Cast {
                op: CastOp::Bitcast,
                v: base,
            }) if depth < MAX_CHAIN => self.canon(*base, depth + 1, m)?,
            Some(InstKind::Gep {
                src_ty,
                base,
                indices,
                ..
            }) if depth < MAX_CHAIN => {
                let base = self.canon(*base, depth + 1, m)?;
                let key = (*src_ty, base, indices.clone());
                match self.geps.get(&key) {
                    Some(&rep) => Value::Inst(rep),
                    None => {
                        let bytes = tree_bytes(
                            1,
                            std::mem::size_of::<((TypeId, Value, Box<[Value]>), InstId)>()
                                + std::mem::size_of_val::<[Value]>(indices),
                        );
                        m.charge(bytes)?;
                        self.geps.insert(key, i);
                        v
                    }
                }
            }
            _ => v,
        };
        if let Some(slot) = self.canon.get_mut(i.index()) {
            *slot = Some(c);
        }
        Ok(c)
    }

    /// The injectivity of integer `v` in the thread of a simdgroup.
    fn inj_int(&self, v: Value, depth: u32, m: &mut Meter) -> Result<Option<Inj>, Error> {
        if depth >= MAX_CHAIN {
            return Ok(None);
        }
        m.tick(1)?;
        let width = int_width(self.e, self.e.module.value_ty(self.e.func, v));
        let Some(w) = width else {
            return Ok(None);
        };
        Ok(match v {
            Value::Arg(a) => self.builtin_inj(a.raw()).and_then(|i| i.truncated(w)),
            Value::Const(_) => None,
            Value::Inst(_) => match self.kind(v) {
                // A component of a position vector.
                Some(InstKind::ExtractElement {
                    vec: Value::Arg(a),
                    idx: Value::Const(_),
                }) => self
                    .builtin_inj(a.raw())
                    .filter(|i| i.same)
                    .and_then(|i| i.truncated(w)),
                Some(InstKind::BinOp {
                    op: op @ (BinOp::Add | BinOp::Sub | BinOp::Or),
                    flags,
                    a,
                    b,
                    ..
                }) if *op != BinOp::Or || flags.contains(IntFlags::DISJOINT) => {
                    if self.uniform(*b) {
                        self.inj_int(*a, depth + 1, m)?
                    } else if self.uniform(*a) {
                        self.inj_int(*b, depth + 1, m)?
                    } else {
                        None
                    }
                }
                Some(InstKind::BinOp {
                    op: op @ (BinOp::Mul | BinOp::Shl),
                    a,
                    b,
                    ..
                }) => {
                    let konst = |x: Value| match x {
                        Value::Const(c) => const_int(self.e, c),
                        _ => None,
                    };
                    let (inner, c) = match (op, konst(*a), konst(*b)) {
                        (BinOp::Mul, _, Some(c)) => (*a, c.unsigned_abs()),
                        (BinOp::Mul, Some(c), None) => (*b, c.unsigned_abs()),
                        (BinOp::Shl, _, Some(k)) if (0..63).contains(&k) && k < i64::from(w) => {
                            (*a, 1u64 << k)
                        }
                        _ => return Ok(None),
                    };
                    self.inj_int(inner, depth + 1, m)?
                        .and_then(|i| i.scaled(c, w))
                }
                Some(InstKind::Cast {
                    op: op @ (CastOp::ZExt | CastOp::SExt),
                    v: inner,
                }) => {
                    let w1 = int_width(self.e, self.e.module.value_ty(self.e.func, *inner));
                    match w1 {
                        Some(w1) if w1 < w => {
                            let ext = if *op == CastOp::ZExt {
                                Ext::Zero
                            } else {
                                Ext::Sign
                            };
                            match self.inj_int(*inner, depth + 1, m)? {
                                Some(i) => self.extend(i, *inner, w1, ext, m)?,
                                None => None,
                            }
                        }
                        _ => None,
                    }
                }
                Some(InstKind::Cast {
                    op: CastOp::Trunc,
                    v: inner,
                }) => self
                    .inj_int(*inner, depth + 1, m)?
                    .and_then(|i| i.truncated(w)),
                // A remainder modulo `2^k` of values closer than `2^k`: their difference, or
                // `2^k` less it, which is never 0.
                Some(InstKind::BinOp {
                    op: BinOp::And,
                    a,
                    b: Value::Const(c),
                    ..
                }) => {
                    let modulus = const_int(self.e, *c)
                        .and_then(|k| u64::try_from(k).ok())
                        .and_then(|k| k.checked_add(1))
                        .filter(|&k| k.is_power_of_two() && k < 1u64 << (w.min(64) - 1));
                    match (modulus, self.inj_int(*a, depth + 1, m)?) {
                        (Some(k), Some(i)) if i.spread < k => Some(Inj {
                            step: i.step.min(k - i.spread),
                            spread: i.spread.max(k - i.step),
                            same: i.same,
                        }),
                        _ => None,
                    }
                }
                Some(InstKind::Freeze(inner)) => self.inj_int(*inner, depth + 1, m)?,
                Some(InstKind::Phi { incoming, .. }) => match self.counter(v, incoming, w, 0, m)? {
                    Some(k) if k.once => self.inj_int(k.start, depth + 1, m)?,
                    // Every value is the start plus a multiple of the step, so two threads'
                    // values differ by their starts' difference modulo the step: never 0 where
                    // the starts are closer than the step, at whatever iterations.
                    Some(k) if k.residue && k.hi - k.lo < 1u64 << (w.min(64) - 1) => self
                        .inj_int(k.start, depth + 1, m)?
                        .filter(|i| i.spread < k.step)
                        .map(|i| Inj {
                            step: i.step.min(k.step - i.spread),
                            spread: k.hi - k.lo,
                            same: i.same,
                        })
                        .filter(|i| i.step <= i.spread),
                    _ => None,
                },
                _ => None,
            },
        })
    }

    /// How parameter `param`'s values differ between two threads of a simdgroup, where it is a
    /// thread's index or position builtin. A simdgroup is 32 consecutive linear threads of one
    /// threadgroup, so both indices differ by 1 to 31 between its threads. Each component of a
    /// position, in the grid or in the threadgroup, is the threadgroup's origin, one for the
    /// simdgroup, plus the thread's position in the threadgroup, below 1024: two threads' differ by
    /// at most 1023, and may be equal (in a threadgroup narrower than 32, threads of a simdgroup
    /// share an `x`).
    fn builtin_inj(&self, param: u32) -> Option<Inj> {
        let role = &self.e.ep.params.iter().find(|p| p.param == param)?.role;
        match role {
            ParamRole::Builtin(
                Builtin::ThreadIndexInSimdgroup | Builtin::ThreadIndexInThreadgroup,
            ) => Some(Inj {
                step: 1,
                spread: 31,
                same: false,
            }),
            ParamRole::Builtin(
                Builtin::ThreadPositionInGrid | Builtin::ThreadPositionInThreadgroup,
            ) => Some(Inj {
                step: 1,
                spread: 1023,
                same: true,
            }),
            _ => None,
        }
    }

    /// The pointer index width of address `v`'s space.
    fn index_bits(&self, v: Value) -> Option<u32> {
        match self
            .e
            .module
            .value_ty(self.e.func, v)
            .and_then(|t| self.e.module.types().get(t))
        {
            Some(Type::Ptr { space }) => Some(self.e.layout.pointer(space.raw()).index_bits),
            _ => None,
        }
    }

    /// How address `v` differs between two threads of a simdgroup.
    fn ptr_spread(&self, v: Value, depth: u32, m: &mut Meter) -> Result<Option<Spread>, Error> {
        if self.uniform(v) {
            return Ok(Some(Spread::Same));
        }
        if depth >= MAX_CHAIN {
            return Ok(None);
        }
        m.tick(1)?;
        match self.kind(v) {
            Some(InstKind::Cast {
                op: CastOp::Bitcast,
                v: base,
            }) => self.ptr_spread(*base, depth + 1, m),
            Some(InstKind::Gep {
                src_ty,
                base,
                indices,
                ..
            }) => match self.index_bits(v) {
                Some(bits) => self.gep_spread(*src_ty, *base, indices, bits, depth + 1, m),
                None => Ok(None),
            },
            _ => Ok(None),
        }
    }

    /// How a GEP over `src_ty` from `base` by `indices` differs between two threads of a
    /// simdgroup: as its base when every index is uniform, and by its stride times the index
    /// when one index is injective and the base uniform.
    fn gep_spread(
        &self,
        src_ty: TypeId,
        base: Value,
        indices: &[Value],
        index_bits: u32,
        depth: u32,
        m: &mut Meter,
    ) -> Result<Option<Spread>, Error> {
        let mut varying = indices
            .iter()
            .enumerate()
            .filter(|&(_, &k)| !self.uniform(k));
        match (varying.next(), varying.next()) {
            (None, _) => self.ptr_spread(base, depth + 1, m),
            (Some((at, &k)), None) if self.uniform(base) => {
                let Some(stride) = self.stride_at(src_ty, indices, at) else {
                    return Ok(None);
                };
                Ok(self
                    .index_inj(k, index_bits, depth + 1, m)?
                    .and_then(|i| i.scaled(stride, index_bits))
                    .map(|i| Spread::Apart {
                        min: i.step,
                        max: i.spread,
                        same: i.same,
                    }))
            }
            _ => Ok(None),
        }
    }

    /// The injectivity of GEP index `k` in the index width `bits`: its core's
    /// ([`AddrFacts::affine`]; the constant it adds moves both threads alike), extended exactly
    /// where the core's range keeps every difference, else as [`Inj::extended`] allows.
    fn index_inj(
        &self,
        k: Value,
        bits: u32,
        depth: u32,
        m: &mut Meter,
    ) -> Result<Option<Inj>, Error> {
        let (core, ext, _) = self.affine(k, bits, m)?;
        let Some(w1) = self.width(core) else {
            return Ok(None);
        };
        let Some(inj) = self.inj_int(core, depth, m)? else {
            return Ok(None);
        };
        if ext == Ext::None {
            return Ok(if w1 > bits {
                inj.truncated(bits)
            } else {
                Some(inj)
            });
        }
        self.extend(inj, core, w1, ext, m)
    }

    /// Fact `inj` of `w1`-bit value `v` after extension `ext`: exact where `v`'s range keeps
    /// every difference (both values in `[lo, hi]` differ by at most `hi - lo`; below half the
    /// narrow width's range, that difference is the one the narrow width reads, and the
    /// extension keeps it), else as [`Inj::extended`] allows.
    fn extend(
        &self,
        inj: Inj,
        v: Value,
        w1: u32,
        ext: Ext,
        m: &mut Meter,
    ) -> Result<Option<Inj>, Error> {
        let half = 1u64 << w1.saturating_sub(1).min(63);
        let exact = self
            .range(v, 0, m)?
            .filter(|&(lo, hi)| (ext != Ext::Sign || hi < half) && hi - lo < half);
        Ok(match exact {
            Some(r) => inj.within(r),
            None => inj.extended(w1),
        })
    }
}

/// `p` with its operands swapped: `a p b` is `b swapped(p) a`.
fn swapped(p: crate::air::ir::IntPredicate) -> crate::air::ir::IntPredicate {
    use crate::air::ir::IntPredicate as P;
    match p {
        P::Eq | P::Ne => p,
        P::Ugt => P::Ult,
        P::Uge => P::Ule,
        P::Ult => P::Ugt,
        P::Ule => P::Uge,
        P::Sgt => P::Slt,
        P::Sge => P::Sle,
        P::Slt => P::Sgt,
        P::Sle => P::Sge,
    }
}

/// The negation of `p`.
fn inverse(p: crate::air::ir::IntPredicate) -> crate::air::ir::IntPredicate {
    use crate::air::ir::IntPredicate as P;
    match p {
        P::Eq => P::Ne,
        P::Ne => P::Eq,
        P::Ugt => P::Ule,
        P::Uge => P::Ult,
        P::Ult => P::Uge,
        P::Ule => P::Ugt,
        P::Sgt => P::Sle,
        P::Sge => P::Slt,
        P::Slt => P::Sge,
        P::Sle => P::Sgt,
    }
}

/// The span `lo..=hi` into `root` when its offsets are ones a threadgroup object can hold.
fn within(root: Root, lo: i64, hi: i64) -> Option<Span> {
    let fits = |o: i64| (0..1i64 << 31).contains(&o);
    (fits(lo) && fits(hi)).then_some(Span { root, lo, hi })
}
