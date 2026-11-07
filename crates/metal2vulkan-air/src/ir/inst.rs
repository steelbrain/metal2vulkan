//! The closed instruction set.
//!
//! Every instruction carries its result type ([`Inst::ty`]; `void` for stores, fences,
//! terminators, and void calls). Floating-point operations keep their fast-math flags; the
//! lowering owns the precision contract. Metadata attachments survive only as typed
//! [`LoopHints`] on terminators and [`RangeMd`] on loads and calls; every other kind is dropped
//! at decode.

use super::attrs::CallAttrs;
use super::id::{ArgIdx, BlockId, ConstId, FuncId, InstId, TypeId};
use crate::intrinsics::{AirIntrinsic, LlvmIntrinsic, Template};

/// An SSA operand. Globals and functions are operands through [`super::Const::GlobalAddr`] /
/// [`super::Const::FuncAddr`] constants, so every value has exactly one spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Value {
    /// A parameter of the enclosing function.
    Arg(ArgIdx),
    /// A non-void instruction of the enclosing function.
    Inst(InstId),
    /// A module constant.
    Const(ConstId),
}

/// One instruction: its result type and its operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Inst {
    /// The result type; `void` when the instruction has no result.
    pub ty: TypeId,
    /// The operation and its operands.
    pub kind: InstKind,
}

/// The operation of an [`Inst`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum InstKind {
    // ---- Arithmetic ----
    /// An integer or floating-point binary operation, `a op b`, on scalars or vectors.
    BinOp {
        /// The operator.
        op: BinOp,
        /// The integer wrap/exactness flags; none on a floating-point operator.
        flags: IntFlags,
        /// The fast-math flags; none on an integer operator.
        fmf: FastMath,
        /// The left operand.
        a: Value,
        /// The right operand.
        b: Value,
    },
    /// `fneg`: the floating-point negation of `v`.
    FNeg {
        /// The operand negated.
        v: Value,
        /// The fast-math flags.
        fmf: FastMath,
    },
    /// A cast of `v` to the instruction's type.
    Cast {
        /// The cast operator.
        op: CastOp,
        /// The value cast.
        v: Value,
    },
    /// `icmp`: an integer or pointer comparison yielding `i1`, or a vector of `i1` per lane.
    ICmp {
        /// The comparison predicate.
        pred: IntPredicate,
        /// The left operand.
        a: Value,
        /// The right operand.
        b: Value,
    },
    /// `fcmp`: a floating-point comparison yielding `i1`, or a vector of `i1` per lane.
    FCmp {
        /// The comparison predicate.
        pred: FloatPredicate,
        /// The fast-math flags.
        fmf: FastMath,
        /// The left operand.
        a: Value,
        /// The right operand.
        b: Value,
    },
    /// `select`: `t` where `c` is true, else `f`; a vector condition selects per lane.
    Select {
        /// The condition: `i1`, or a vector of `i1` with the operands' lane count.
        c: Value,
        /// The value chosen where `c` is true.
        t: Value,
        /// The value chosen where `c` is false.
        f: Value,
        /// The fast-math flags; none unless the operands are floating-point.
        fmf: FastMath,
    },
    /// `freeze`: the operand with any `undef` or `poison` replaced by an arbitrary fixed value.
    Freeze(Value),

    // ---- Vectors ----
    /// `extractelement`: lane `idx` of `vec`.
    ExtractElement {
        /// The vector read.
        vec: Value,
        /// The lane index, an integer.
        idx: Value,
    },
    /// `insertelement`: `vec` with lane `idx` replaced by `elt`.
    InsertElement {
        /// The vector the result is built from.
        vec: Value,
        /// The new lane value.
        elt: Value,
        /// The lane index, an integer.
        idx: Value,
    },
    /// `None` mask lanes are `poison`.
    ShuffleVector {
        /// The first source vector; mask indices below its lane count select from it.
        a: Value,
        /// The second source vector; mask indices from `a`'s lane count up select from it.
        b: Value,
        /// One entry per result lane: an index into the lanes of `a` followed by those of `b`.
        mask: Box<[Option<u32>]>,
    },

    // ---- Aggregates ----
    /// `extractvalue`: the member of `agg` that `indices` name.
    ExtractValue {
        /// The struct or array read.
        agg: Value,
        /// The constant member indices, outermost first.
        indices: Box<[u32]>,
    },
    /// `insertvalue`: `agg` with the member that `indices` name replaced by `val`.
    InsertValue {
        /// The struct or array the result is built from.
        agg: Value,
        /// The new member value.
        val: Value,
        /// The constant member indices, outermost first.
        indices: Box<[u32]>,
    },

    // ---- Memory ----
    /// Stack allocation of `count` elements of `ty`.
    Alloca {
        /// The element type allocated.
        ty: TypeId,
        /// The number of elements, an integer.
        count: Value,
        /// The requested alignment; `None` when the record gives none.
        align: Option<Align>,
    },
    /// A load of the instruction's type (the authoritative access type) from `ptr`.
    Load {
        /// The address read.
        ptr: Value,
        /// The alignment the access assumes; `None` when the record gives none.
        align: Option<Align>,
        /// `volatile`: the access must be kept, once, in order with other volatile accesses.
        volatile: bool,
        /// The ordering and scope of a `load atomic`; `None` for a plain load.
        atomic: Option<Atomic>,
        /// A `!range` attachment on the loaded value.
        range: Option<RangeMd>,
    },
    /// `store` of `val` to `ptr`.
    Store {
        /// The value stored.
        val: Value,
        /// The address written.
        ptr: Value,
        /// The alignment the access assumes; `None` when the record gives none.
        align: Option<Align>,
        /// `volatile`: the access must be kept, once, in order with other volatile accesses.
        volatile: bool,
        /// The ordering and scope of a `store atomic`; `None` for a plain store.
        atomic: Option<Atomic>,
    },
    /// `getelementptr` over the explicit source element type.
    Gep {
        /// The source element type: the first index steps over it, the rest index into it.
        src_ty: TypeId,
        /// The base pointer.
        base: Value,
        /// The integer indices, in order.
        indices: Box<[Value]>,
        /// `inbounds`: the address stays within the object `base` points into (the decoder drops
        /// `nusw` and `nuw`).
        inbounds: bool,
    },
    /// `atomicrmw`: atomically replace `*ptr` with `*ptr op val`; the result is the old value.
    AtomicRmw {
        /// The operator.
        op: AtomicRmwOp,
        /// The address updated.
        ptr: Value,
        /// The operand combined with the old value.
        val: Value,
        /// The ordering and scope.
        atomic: Atomic,
        /// `volatile`: the access must be kept, once, in order with other volatile accesses.
        volatile: bool,
        /// The alignment the access assumes; `None` when the record gives none.
        align: Option<Align>,
    },
    /// Result type is `{ T, i1 }`.
    CmpXchg {
        /// The address updated.
        ptr: Value,
        /// The value `*ptr` is compared with.
        cmp: Value,
        /// The value stored when `*ptr` equals `cmp`.
        new: Value,
        /// The ordering and scope when the exchange happens.
        success: Atomic,
        /// The ordering when it does not; the scope is `success`'s.
        failure: Ordering,
        /// `weak`: the exchange may fail even when `*ptr` equals `cmp`.
        weak: bool,
        /// `volatile`: the access must be kept, once, in order with other volatile accesses.
        volatile: bool,
        /// The alignment the access assumes; `None` when the record gives none.
        align: Option<Align>,
    },
    /// `fence` with this ordering and scope.
    Fence(Atomic),

    // ---- Calls ----
    /// `call`, with its payload boxed.
    Call(Box<Call>),

    // ---- Other ----
    /// `phi`: the value of the incoming edge control arrived along.
    Phi {
        /// One `(value, predecessor block)` pair per incoming edge.
        incoming: Box<[(Value, BlockId)]>,
        /// The fast-math flags; none unless the result is floating-point.
        fmf: FastMath,
    },

    // ---- Terminators ----
    /// `ret`, with the returned value unless the function returns `void`.
    Ret(Option<Value>),
    /// An unconditional `br`.
    Br {
        /// The block branched to.
        dest: BlockId,
        /// The hints from a `!llvm.loop` attachment.
        hints: Option<LoopHints>,
    },
    /// A conditional `br`: to `t` when `c` is true, else to `f`.
    CondBr {
        /// The `i1` condition.
        c: Value,
        /// The block taken when `c` is true.
        t: BlockId,
        /// The block taken when `c` is false.
        f: BlockId,
        /// The hints from a `!llvm.loop` attachment.
        hints: Option<LoopHints>,
    },
    /// Case values are zero-extended bit patterns of `v`'s integer type.
    Switch {
        /// The integer switched on.
        v: Value,
        /// The block taken when no case matches.
        default: BlockId,
        /// One `(case value, block)` pair per case, in record order.
        cases: Box<[(u64, BlockId)]>,
        /// The hints from a `!llvm.loop` attachment.
        hints: Option<LoopHints>,
    },
    /// `unreachable`.
    Unreachable,
}

/// Every [`InstKind`] discriminant, for exhaustiveness tests and census tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Opcode {
    /// A binary operation (`InstKind::BinOp`).
    BinOp,
    /// `fneg` (`InstKind::FNeg`).
    FNeg,
    /// A cast (`InstKind::Cast`).
    Cast,
    /// `icmp` (`InstKind::ICmp`).
    ICmp,
    /// `fcmp` (`InstKind::FCmp`).
    FCmp,
    /// `select` (`InstKind::Select`).
    Select,
    /// `freeze` (`InstKind::Freeze`).
    Freeze,
    /// `extractelement` (`InstKind::ExtractElement`).
    ExtractElement,
    /// `insertelement` (`InstKind::InsertElement`).
    InsertElement,
    /// `shufflevector` (`InstKind::ShuffleVector`).
    ShuffleVector,
    /// `extractvalue` (`InstKind::ExtractValue`).
    ExtractValue,
    /// `insertvalue` (`InstKind::InsertValue`).
    InsertValue,
    /// `alloca` (`InstKind::Alloca`).
    Alloca,
    /// `load` (`InstKind::Load`).
    Load,
    /// `store` (`InstKind::Store`).
    Store,
    /// `getelementptr` (`InstKind::Gep`).
    Gep,
    /// `atomicrmw` (`InstKind::AtomicRmw`).
    AtomicRmw,
    /// `cmpxchg` (`InstKind::CmpXchg`).
    CmpXchg,
    /// `fence` (`InstKind::Fence`).
    Fence,
    /// `call` (`InstKind::Call`).
    Call,
    /// `phi` (`InstKind::Phi`).
    Phi,
    /// `ret` (`InstKind::Ret`).
    Ret,
    /// An unconditional `br` (`InstKind::Br`).
    Br,
    /// A conditional `br` (`InstKind::CondBr`).
    CondBr,
    /// `switch` (`InstKind::Switch`).
    Switch,
    /// `unreachable` (`InstKind::Unreachable`).
    Unreachable,
}

impl Opcode {
    /// Every opcode.
    #[cfg(test)]
    pub const ALL: [Opcode; 26] = [
        Opcode::BinOp,
        Opcode::FNeg,
        Opcode::Cast,
        Opcode::ICmp,
        Opcode::FCmp,
        Opcode::Select,
        Opcode::Freeze,
        Opcode::ExtractElement,
        Opcode::InsertElement,
        Opcode::ShuffleVector,
        Opcode::ExtractValue,
        Opcode::InsertValue,
        Opcode::Alloca,
        Opcode::Load,
        Opcode::Store,
        Opcode::Gep,
        Opcode::AtomicRmw,
        Opcode::CmpXchg,
        Opcode::Fence,
        Opcode::Call,
        Opcode::Phi,
        Opcode::Ret,
        Opcode::Br,
        Opcode::CondBr,
        Opcode::Switch,
        Opcode::Unreachable,
    ];

    /// Whether instructions of this opcode end a block.
    pub fn is_terminator(self) -> bool {
        matches!(
            self,
            Opcode::Ret | Opcode::Br | Opcode::CondBr | Opcode::Switch | Opcode::Unreachable
        )
    }
}

impl InstKind {
    /// The discriminant of this instruction.
    pub fn opcode(&self) -> Opcode {
        match self {
            InstKind::BinOp { .. } => Opcode::BinOp,
            InstKind::FNeg { .. } => Opcode::FNeg,
            InstKind::Cast { .. } => Opcode::Cast,
            InstKind::ICmp { .. } => Opcode::ICmp,
            InstKind::FCmp { .. } => Opcode::FCmp,
            InstKind::Select { .. } => Opcode::Select,
            InstKind::Freeze(_) => Opcode::Freeze,
            InstKind::ExtractElement { .. } => Opcode::ExtractElement,
            InstKind::InsertElement { .. } => Opcode::InsertElement,
            InstKind::ShuffleVector { .. } => Opcode::ShuffleVector,
            InstKind::ExtractValue { .. } => Opcode::ExtractValue,
            InstKind::InsertValue { .. } => Opcode::InsertValue,
            InstKind::Alloca { .. } => Opcode::Alloca,
            InstKind::Load { .. } => Opcode::Load,
            InstKind::Store { .. } => Opcode::Store,
            InstKind::Gep { .. } => Opcode::Gep,
            InstKind::AtomicRmw { .. } => Opcode::AtomicRmw,
            InstKind::CmpXchg { .. } => Opcode::CmpXchg,
            InstKind::Fence(_) => Opcode::Fence,
            InstKind::Call(_) => Opcode::Call,
            InstKind::Phi { .. } => Opcode::Phi,
            InstKind::Ret(_) => Opcode::Ret,
            InstKind::Br { .. } => Opcode::Br,
            InstKind::CondBr { .. } => Opcode::CondBr,
            InstKind::Switch { .. } => Opcode::Switch,
            InstKind::Unreachable => Opcode::Unreachable,
        }
    }

    /// Whether this instruction ends a block.
    pub fn is_terminator(&self) -> bool {
        self.opcode().is_terminator()
    }

    /// The heap bytes the operation holds outside its arena slot: a call's boxed payload with its
    /// arguments, attributes, and range, and every boxed operand list (mask, indices, incoming
    /// edges, cases, ranges). It is what the allocator holds for the operation beyond
    /// `size_of::<Inst>()`, and what a body is charged when the instruction enters it
    /// ([`super::Body::push_inst`]).
    pub fn heap_bytes(&self) -> usize {
        match self {
            InstKind::ShuffleVector { mask, .. } => std::mem::size_of_val::<[Option<u32>]>(mask),
            InstKind::ExtractValue { indices, .. } | InstKind::InsertValue { indices, .. } => {
                std::mem::size_of_val::<[u32]>(indices)
            }
            InstKind::Load { range, .. } => range.as_ref().map_or(0, RangeMd::heap_bytes),
            InstKind::Gep { indices, .. } => std::mem::size_of_val::<[Value]>(indices),
            InstKind::Call(call) => std::mem::size_of::<Call>().saturating_add(call.heap_bytes()),
            InstKind::Phi { incoming, .. } => std::mem::size_of_val::<[(Value, BlockId)]>(incoming),
            InstKind::Switch { cases, .. } => std::mem::size_of_val::<[(u64, BlockId)]>(cases),
            InstKind::BinOp { .. }
            | InstKind::FNeg { .. }
            | InstKind::Cast { .. }
            | InstKind::ICmp { .. }
            | InstKind::FCmp { .. }
            | InstKind::Select { .. }
            | InstKind::Freeze(_)
            | InstKind::ExtractElement { .. }
            | InstKind::InsertElement { .. }
            | InstKind::Alloca { .. }
            | InstKind::Store { .. }
            | InstKind::AtomicRmw { .. }
            | InstKind::CmpXchg { .. }
            | InstKind::Fence(_)
            | InstKind::Ret(_)
            | InstKind::Br { .. }
            | InstKind::CondBr { .. }
            | InstKind::Unreachable => 0,
        }
    }

    /// Call `f` on every value operand, in operand order. PHI incoming values are included;
    /// block operands are not (see [`InstKind::successors`]).
    pub fn for_each_value(&self, mut f: impl FnMut(Value)) {
        match self {
            InstKind::BinOp { a, b, .. }
            | InstKind::ICmp { a, b, .. }
            | InstKind::FCmp { a, b, .. }
            | InstKind::ShuffleVector { a, b, .. } => {
                f(*a);
                f(*b);
            }
            InstKind::FNeg { v, .. } | InstKind::Cast { v, .. } | InstKind::Freeze(v) => f(*v),
            InstKind::Select { c, t, f: e, .. } => {
                f(*c);
                f(*t);
                f(*e);
            }
            InstKind::ExtractElement { vec, idx } => {
                f(*vec);
                f(*idx);
            }
            InstKind::InsertElement { vec, elt, idx } => {
                f(*vec);
                f(*elt);
                f(*idx);
            }
            InstKind::ExtractValue { agg, .. } => f(*agg),
            InstKind::InsertValue { agg, val, .. } => {
                f(*agg);
                f(*val);
            }
            InstKind::Alloca { count, .. } => f(*count),
            InstKind::Load { ptr, .. } => f(*ptr),
            InstKind::Store { val, ptr, .. } => {
                f(*val);
                f(*ptr);
            }
            InstKind::Gep { base, indices, .. } => {
                f(*base);
                indices.iter().copied().for_each(f);
            }
            InstKind::AtomicRmw { ptr, val, .. } => {
                f(*ptr);
                f(*val);
            }
            InstKind::CmpXchg { ptr, cmp, new, .. } => {
                f(*ptr);
                f(*cmp);
                f(*new);
            }
            InstKind::Call(call) => {
                if let Callee::Indirect { callee, .. } = call.callee {
                    f(callee);
                }
                call.args.iter().copied().for_each(f);
            }
            InstKind::Phi { incoming, .. } => incoming.iter().for_each(|&(v, _)| f(v)),
            InstKind::Ret(v) => v.iter().copied().for_each(f),
            InstKind::CondBr { c, .. } => f(*c),
            InstKind::Switch { v, .. } => f(*v),
            InstKind::Fence(_) | InstKind::Br { .. } | InstKind::Unreachable => {}
        }
    }

    /// Call `f` on every value operand, mutably, in the order of [`InstKind::for_each_value`].
    pub fn for_each_value_mut(&mut self, mut f: impl FnMut(&mut Value)) {
        match self {
            InstKind::BinOp { a, b, .. }
            | InstKind::ICmp { a, b, .. }
            | InstKind::FCmp { a, b, .. }
            | InstKind::ShuffleVector { a, b, .. } => {
                f(a);
                f(b);
            }
            InstKind::FNeg { v, .. } | InstKind::Cast { v, .. } | InstKind::Freeze(v) => f(v),
            InstKind::Select { c, t, f: e, .. } => {
                f(c);
                f(t);
                f(e);
            }
            InstKind::ExtractElement { vec, idx } => {
                f(vec);
                f(idx);
            }
            InstKind::InsertElement { vec, elt, idx } => {
                f(vec);
                f(elt);
                f(idx);
            }
            InstKind::ExtractValue { agg, .. } => f(agg),
            InstKind::InsertValue { agg, val, .. } => {
                f(agg);
                f(val);
            }
            InstKind::Alloca { count, .. } => f(count),
            InstKind::Load { ptr, .. } => f(ptr),
            InstKind::Store { val, ptr, .. } => {
                f(val);
                f(ptr);
            }
            InstKind::Gep { base, indices, .. } => {
                f(base);
                indices.iter_mut().for_each(f);
            }
            InstKind::AtomicRmw { ptr, val, .. } => {
                f(ptr);
                f(val);
            }
            InstKind::CmpXchg { ptr, cmp, new, .. } => {
                f(ptr);
                f(cmp);
                f(new);
            }
            InstKind::Call(call) => {
                if let Callee::Indirect { callee, .. } = &mut call.callee {
                    f(callee);
                }
                call.args.iter_mut().for_each(f);
            }
            InstKind::Phi { incoming, .. } => incoming.iter_mut().for_each(|(v, _)| f(v)),
            InstKind::Ret(v) => v.iter_mut().for_each(f),
            InstKind::CondBr { c, .. } => f(c),
            InstKind::Switch { v, .. } => f(v),
            InstKind::Fence(_) | InstKind::Br { .. } | InstKind::Unreachable => {}
        }
    }

    /// The successor blocks of a terminator, in operand order (duplicates kept: a switch with
    /// two cases to one block has two edges). Empty for non-terminators.
    /// Allocation-free: the CFG of a body is walked through this on every build.
    pub fn successors(&self) -> impl Iterator<Item = BlockId> + '_ {
        let (first, second, cases): (Option<BlockId>, Option<BlockId>, &[(u64, BlockId)]) =
            match self {
                InstKind::Br { dest, .. } => (Some(*dest), None, &[]),
                InstKind::CondBr { t, f, .. } => (Some(*t), Some(*f), &[]),
                InstKind::Switch { default, cases, .. } => (Some(*default), None, cases),
                _ => (None, None, &[]),
            };
        first
            .into_iter()
            .chain(second)
            .chain(cases.iter().map(|&(_, b)| b))
    }

    /// The loop hints attached to a terminator.
    pub fn loop_hints(&self) -> Option<&LoopHints> {
        match self {
            InstKind::Br { hints, .. }
            | InstKind::CondBr { hints, .. }
            | InstKind::Switch { hints, .. } => hints.as_ref(),
            _ => None,
        }
    }
}

/// A call instruction's payload.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Call {
    /// What the call calls.
    pub callee: Callee,
    /// The arguments, in parameter order.
    pub args: Box<[Value]>,
    /// The fast-math flags.
    pub fmf: FastMath,
    /// The `tail`/`musttail`/`notail` marker.
    pub tail: TailKind,
    /// Call-site attributes, when the record carries any.
    pub attrs: Option<Box<CallAttrs>>,
    /// A `!range` attachment on the result.
    pub range: Option<RangeMd>,
}

impl Call {
    /// The heap bytes the call holds beyond its own box: its arguments, its boxed attributes with
    /// their lists, and its range. Every field is named, so a new one must be counted or shown to
    /// be held inline (`no_heap`).
    pub fn heap_bytes(&self) -> usize {
        let Call {
            callee,
            args,
            fmf,
            tail,
            attrs,
            range,
        } = self;
        let attrs = attrs.as_ref().map_or(0, |a| {
            std::mem::size_of::<CallAttrs>().saturating_add(a.heap_bytes())
        });
        callee
            .heap_bytes()
            .saturating_add(no_heap(fmf))
            .saturating_add(no_heap(tail))
            .saturating_add(std::mem::size_of_val::<[Value]>(args))
            .saturating_add(attrs)
            .saturating_add(range.as_ref().map_or(0, RangeMd::heap_bytes))
    }
}

/// The heap bytes of a field held inline: none. Only a `Copy` type is accepted, and a `Copy`
/// type owns no heap memory, so a `heap_bytes` that names every field of its struct, counting each
/// owned list and passing each other field here, fails to compile when a field it does not count
/// can own an allocation.
pub(super) fn no_heap<T: Copy>(_: &T) -> usize {
    0
}

impl Callee {
    /// The heap bytes the callee holds: none, as every form's fields are held inline.
    fn heap_bytes(&self) -> usize {
        match self {
            Callee::Direct(f) | Callee::UnknownIntrinsic(f) => no_heap(f),
            Callee::Indirect { callee, fn_ty } => no_heap(callee).saturating_add(no_heap(fn_ty)),
            Callee::Air { decl, op } => no_heap(decl).saturating_add(no_heap(op)),
            Callee::Llvm { decl, op } => no_heap(decl).saturating_add(no_heap(op)),
            Callee::Unbound { decl, template } => no_heap(decl).saturating_add(no_heap(template)),
        }
    }
}

/// What a call calls. Every call to a declaration named `air.*` or `llvm.*` is decoded once
/// into [`Callee::Air`] / [`Callee::Llvm`], kept as [`Callee::Unbound`] until its immediates are
/// constants, or kept as [`Callee::UnknownIntrinsic`] until after function-constant
/// specialization; afterwards no code inspects intrinsic names. The declaration is kept on every
/// form so the verifier checks arguments against its type.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Callee {
    /// A defined or declared (non-intrinsic) function.
    Direct(FuncId),
    /// A call through a pointer, with the call's explicit function type.
    Indirect {
        /// The function pointer called.
        callee: Value,
        /// The call's function type.
        fn_ty: TypeId,
    },
    /// An `air.*` intrinsic in the closed set.
    Air {
        /// The intrinsic's declaration.
        decl: FuncId,
        /// The decoded intrinsic.
        op: AirIntrinsic,
    },
    /// An `llvm.*` intrinsic in the closed set.
    Llvm {
        /// The intrinsic's declaration.
        decl: FuncId,
        /// The decoded intrinsic.
        op: LlvmIntrinsic,
    },
    /// An `air.*`/`llvm.*` declaration outside the closed set.
    UnknownIntrinsic(FuncId),
    /// An intrinsic in the closed set whose immediate operands are not all integer constants in
    /// the function that makes the call: an operand computed from the function's parameters, as
    /// Metal's compiler leaves OpenCL C's `barrier(flags)` out of line, passing its flags to
    /// `air.wg.barrier`. Specialization's constant propagation reads the immediates where
    /// inlining gives the parameters each call site's arguments, making the call
    /// [`Callee::Air`] or [`Callee::Llvm`]; one whose immediates are still not constant there is
    /// refused (`IntrinsicError::ImmediateNotConstant`).
    Unbound {
        /// The intrinsic's declaration.
        decl: FuncId,
        /// The declaration's template, its immediates not read in.
        template: Template,
    },
}

impl Callee {
    /// The called function, for every form except indirect calls.
    pub fn func(&self) -> Option<FuncId> {
        match self {
            Callee::Direct(f)
            | Callee::Air { decl: f, .. }
            | Callee::Llvm { decl: f, .. }
            | Callee::UnknownIntrinsic(f)
            | Callee::Unbound { decl: f, .. } => Some(*f),
            Callee::Indirect { .. } => None,
        }
    }
}

/// Binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BinOp {
    /// `add`: integer addition.
    Add,
    /// `sub`: integer subtraction.
    Sub,
    /// `mul`: integer multiplication.
    Mul,
    /// `udiv`: unsigned integer division.
    UDiv,
    /// `sdiv`: signed integer division.
    SDiv,
    /// `urem`: unsigned integer remainder.
    URem,
    /// `srem`: signed integer remainder, with the dividend's sign.
    SRem,
    /// `shl`: shift left.
    Shl,
    /// `lshr`: logical shift right, filling with zeros.
    LShr,
    /// `ashr`: arithmetic shift right, filling with the sign bit.
    AShr,
    /// `and`: bitwise and.
    And,
    /// `or`: bitwise or.
    Or,
    /// `xor`: bitwise exclusive or.
    Xor,
    /// `fadd`: floating-point addition.
    FAdd,
    /// `fsub`: floating-point subtraction.
    FSub,
    /// `fmul`: floating-point multiplication.
    FMul,
    /// `fdiv`: floating-point division.
    FDiv,
    /// `frem`: floating-point remainder, with the dividend's sign.
    FRem,
}

impl BinOp {
    /// Every operator.
    pub const ALL: [BinOp; 18] = [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::UDiv,
        BinOp::SDiv,
        BinOp::URem,
        BinOp::SRem,
        BinOp::Shl,
        BinOp::LShr,
        BinOp::AShr,
        BinOp::And,
        BinOp::Or,
        BinOp::Xor,
        BinOp::FAdd,
        BinOp::FSub,
        BinOp::FMul,
        BinOp::FDiv,
        BinOp::FRem,
    ];

    /// Whether this operator takes floating-point operands.
    pub fn is_float(self) -> bool {
        matches!(
            self,
            BinOp::FAdd | BinOp::FSub | BinOp::FMul | BinOp::FDiv | BinOp::FRem
        )
    }

    /// The LLVM mnemonic, for the printer.
    pub fn mnemonic(self) -> &'static str {
        match self {
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::Mul => "mul",
            BinOp::UDiv => "udiv",
            BinOp::SDiv => "sdiv",
            BinOp::URem => "urem",
            BinOp::SRem => "srem",
            BinOp::Shl => "shl",
            BinOp::LShr => "lshr",
            BinOp::AShr => "ashr",
            BinOp::And => "and",
            BinOp::Or => "or",
            BinOp::Xor => "xor",
            BinOp::FAdd => "fadd",
            BinOp::FSub => "fsub",
            BinOp::FMul => "fmul",
            BinOp::FDiv => "fdiv",
            BinOp::FRem => "frem",
        }
    }

    /// Whether `flags` are permitted on this operator (LLVM's flag placement rules).
    pub fn allows(self, flags: IntFlags) -> bool {
        let allowed = match self {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl => IntFlags::NUW.union(IntFlags::NSW),
            BinOp::UDiv | BinOp::SDiv | BinOp::LShr | BinOp::AShr => IntFlags::EXACT,
            BinOp::Or => IntFlags::DISJOINT,
            _ => IntFlags::NONE,
        };
        allowed.contains(flags)
    }
}

/// Cast operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CastOp {
    /// `trunc`: integer truncation to a narrower type.
    Trunc,
    /// `zext`: integer zero extension.
    ZExt,
    /// `sext`: integer sign extension.
    SExt,
    /// `fptrunc`: floating-point narrowing.
    FPTrunc,
    /// `fpext`: floating-point widening.
    FPExt,
    /// `fptoui`: floating-point to unsigned integer.
    FPToUI,
    /// `fptosi`: floating-point to signed integer.
    FPToSI,
    /// `uitofp`: unsigned integer to floating-point.
    UIToFP,
    /// `sitofp`: signed integer to floating-point.
    SIToFP,
    /// `ptrtoint`: pointer to integer.
    PtrToInt,
    /// `inttoptr`: integer to pointer.
    IntToPtr,
    /// `bitcast`: the same bits as another type of equal size.
    Bitcast,
    /// `addrspacecast`: a pointer to another address space.
    AddrSpaceCast,
}

impl CastOp {
    /// Every operator.
    #[cfg(test)]
    pub const ALL: [CastOp; 13] = [
        CastOp::Trunc,
        CastOp::ZExt,
        CastOp::SExt,
        CastOp::FPTrunc,
        CastOp::FPExt,
        CastOp::FPToUI,
        CastOp::FPToSI,
        CastOp::UIToFP,
        CastOp::SIToFP,
        CastOp::PtrToInt,
        CastOp::IntToPtr,
        CastOp::Bitcast,
        CastOp::AddrSpaceCast,
    ];

    /// The LLVM mnemonic, for the printer.
    pub fn mnemonic(self) -> &'static str {
        match self {
            CastOp::Trunc => "trunc",
            CastOp::ZExt => "zext",
            CastOp::SExt => "sext",
            CastOp::FPTrunc => "fptrunc",
            CastOp::FPExt => "fpext",
            CastOp::FPToUI => "fptoui",
            CastOp::FPToSI => "fptosi",
            CastOp::UIToFP => "uitofp",
            CastOp::SIToFP => "sitofp",
            CastOp::PtrToInt => "ptrtoint",
            CastOp::IntToPtr => "inttoptr",
            CastOp::Bitcast => "bitcast",
            CastOp::AddrSpaceCast => "addrspacecast",
        }
    }
}

/// Integer comparison predicates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IntPredicate {
    /// `eq`: equal.
    Eq,
    /// `ne`: not equal.
    Ne,
    /// `ugt`: unsigned greater than.
    Ugt,
    /// `uge`: unsigned greater than or equal.
    Uge,
    /// `ult`: unsigned less than.
    Ult,
    /// `ule`: unsigned less than or equal.
    Ule,
    /// `sgt`: signed greater than.
    Sgt,
    /// `sge`: signed greater than or equal.
    Sge,
    /// `slt`: signed less than.
    Slt,
    /// `sle`: signed less than or equal.
    Sle,
}

impl IntPredicate {
    /// Every predicate.
    pub const ALL: [IntPredicate; 10] = [
        IntPredicate::Eq,
        IntPredicate::Ne,
        IntPredicate::Ugt,
        IntPredicate::Uge,
        IntPredicate::Ult,
        IntPredicate::Ule,
        IntPredicate::Sgt,
        IntPredicate::Sge,
        IntPredicate::Slt,
        IntPredicate::Sle,
    ];

    /// The LLVM mnemonic, for the printer.
    pub fn mnemonic(self) -> &'static str {
        match self {
            IntPredicate::Eq => "eq",
            IntPredicate::Ne => "ne",
            IntPredicate::Ugt => "ugt",
            IntPredicate::Uge => "uge",
            IntPredicate::Ult => "ult",
            IntPredicate::Ule => "ule",
            IntPredicate::Sgt => "sgt",
            IntPredicate::Sge => "sge",
            IntPredicate::Slt => "slt",
            IntPredicate::Sle => "sle",
        }
    }
}

/// Floating-point comparison predicates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FloatPredicate {
    /// `false`: always false.
    False,
    /// `oeq`: ordered and equal.
    Oeq,
    /// `ogt`: ordered and greater than.
    Ogt,
    /// `oge`: ordered and greater than or equal.
    Oge,
    /// `olt`: ordered and less than.
    Olt,
    /// `ole`: ordered and less than or equal.
    Ole,
    /// `one`: ordered and not equal.
    One,
    /// `ord`: ordered (neither operand is a NaN).
    Ord,
    /// `uno`: unordered (either operand is a NaN).
    Uno,
    /// `ueq`: unordered or equal.
    Ueq,
    /// `ugt`: unordered or greater than.
    Ugt,
    /// `uge`: unordered or greater than or equal.
    Uge,
    /// `ult`: unordered or less than.
    Ult,
    /// `ule`: unordered or less than or equal.
    Ule,
    /// `une`: unordered or not equal.
    Une,
    /// `true`: always true.
    True,
}

impl FloatPredicate {
    /// Every predicate.
    pub const ALL: [FloatPredicate; 16] = [
        FloatPredicate::False,
        FloatPredicate::Oeq,
        FloatPredicate::Ogt,
        FloatPredicate::Oge,
        FloatPredicate::Olt,
        FloatPredicate::Ole,
        FloatPredicate::One,
        FloatPredicate::Ord,
        FloatPredicate::Uno,
        FloatPredicate::Ueq,
        FloatPredicate::Ugt,
        FloatPredicate::Uge,
        FloatPredicate::Ult,
        FloatPredicate::Ule,
        FloatPredicate::Une,
        FloatPredicate::True,
    ];

    /// The LLVM mnemonic, for the printer.
    pub fn mnemonic(self) -> &'static str {
        match self {
            FloatPredicate::False => "false",
            FloatPredicate::Oeq => "oeq",
            FloatPredicate::Ogt => "ogt",
            FloatPredicate::Oge => "oge",
            FloatPredicate::Olt => "olt",
            FloatPredicate::Ole => "ole",
            FloatPredicate::One => "one",
            FloatPredicate::Ord => "ord",
            FloatPredicate::Uno => "uno",
            FloatPredicate::Ueq => "ueq",
            FloatPredicate::Ugt => "ugt",
            FloatPredicate::Uge => "uge",
            FloatPredicate::Ult => "ult",
            FloatPredicate::Ule => "ule",
            FloatPredicate::Une => "une",
            FloatPredicate::True => "true",
        }
    }
}

/// Atomic memory orderings (`NotAtomic` is the absence of an [`Atomic`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Ordering {
    /// `unordered`: no ordering; only freedom from torn values.
    Unordered,
    /// `monotonic`: one total order of the accesses to each address (C++ `relaxed`).
    Monotonic,
    /// `acquire`.
    Acquire,
    /// `release`.
    Release,
    /// `acq_rel`: both `acquire` and `release`.
    AcqRel,
    /// `seq_cst`: `acq_rel` plus one total order of every `seq_cst` operation.
    SeqCst,
}

impl Ordering {
    /// Every ordering.
    #[cfg(test)]
    pub const ALL: [Ordering; 6] = [
        Ordering::Unordered,
        Ordering::Monotonic,
        Ordering::Acquire,
        Ordering::Release,
        Ordering::AcqRel,
        Ordering::SeqCst,
    ];

    /// The LLVM mnemonic, for the printer.
    pub fn mnemonic(self) -> &'static str {
        match self {
            Ordering::Unordered => "unordered",
            Ordering::Monotonic => "monotonic",
            Ordering::Acquire => "acquire",
            Ordering::Release => "release",
            Ordering::AcqRel => "acq_rel",
            Ordering::SeqCst => "seq_cst",
        }
    }
}

/// Synchronization scopes. Only these two are observed in AIR; any other scope name is refused
/// at decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SyncScope {
    /// `syncscope("singlethread")`: synchronizes only with the same thread.
    SingleThread,
    /// The default scope: synchronizes with every other thread.
    System,
}

impl SyncScope {
    /// Every scope.
    #[cfg(test)]
    pub const ALL: [SyncScope; 2] = [SyncScope::SingleThread, SyncScope::System];
}

/// The ordering and scope of an atomic access or fence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Atomic {
    /// The memory ordering.
    pub ordering: Ordering,
    /// The synchronization scope.
    pub scope: SyncScope,
}

/// `atomicrmw` operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AtomicRmwOp {
    /// `xchg`: store `val`.
    Xchg,
    /// `add`: integer addition.
    Add,
    /// `sub`: integer subtraction.
    Sub,
    /// `and`: bitwise and.
    And,
    /// `nand`: the complement of the bitwise and.
    Nand,
    /// `or`: bitwise or.
    Or,
    /// `xor`: bitwise exclusive or.
    Xor,
    /// `max`: signed integer maximum.
    Max,
    /// `min`: signed integer minimum.
    Min,
    /// `umax`: unsigned integer maximum.
    UMax,
    /// `umin`: unsigned integer minimum.
    UMin,
    /// `fadd`: floating-point addition.
    FAdd,
    /// `fsub`: floating-point subtraction.
    FSub,
    /// `fmax`: floating-point maximum, as `llvm.maxnum`.
    FMax,
    /// `fmin`: floating-point minimum, as `llvm.minnum`.
    FMin,
    /// `uinc_wrap`: increment, wrapping to 0 once the old value is at least `val`.
    UIncWrap,
    /// `udec_wrap`: decrement, wrapping to `val` when the old value is 0 or above `val`.
    UDecWrap,
}

impl AtomicRmwOp {
    /// Every operator.
    #[cfg(test)]
    pub const ALL: [AtomicRmwOp; 17] = [
        AtomicRmwOp::Xchg,
        AtomicRmwOp::Add,
        AtomicRmwOp::Sub,
        AtomicRmwOp::And,
        AtomicRmwOp::Nand,
        AtomicRmwOp::Or,
        AtomicRmwOp::Xor,
        AtomicRmwOp::Max,
        AtomicRmwOp::Min,
        AtomicRmwOp::UMax,
        AtomicRmwOp::UMin,
        AtomicRmwOp::FAdd,
        AtomicRmwOp::FSub,
        AtomicRmwOp::FMax,
        AtomicRmwOp::FMin,
        AtomicRmwOp::UIncWrap,
        AtomicRmwOp::UDecWrap,
    ];

    /// Whether this operator takes a floating-point operand.
    pub fn is_float(self) -> bool {
        matches!(
            self,
            AtomicRmwOp::FAdd | AtomicRmwOp::FSub | AtomicRmwOp::FMax | AtomicRmwOp::FMin
        )
    }

    /// The LLVM mnemonic, for the printer.
    pub fn mnemonic(self) -> &'static str {
        match self {
            AtomicRmwOp::Xchg => "xchg",
            AtomicRmwOp::Add => "add",
            AtomicRmwOp::Sub => "sub",
            AtomicRmwOp::And => "and",
            AtomicRmwOp::Nand => "nand",
            AtomicRmwOp::Or => "or",
            AtomicRmwOp::Xor => "xor",
            AtomicRmwOp::Max => "max",
            AtomicRmwOp::Min => "min",
            AtomicRmwOp::UMax => "umax",
            AtomicRmwOp::UMin => "umin",
            AtomicRmwOp::FAdd => "fadd",
            AtomicRmwOp::FSub => "fsub",
            AtomicRmwOp::FMax => "fmax",
            AtomicRmwOp::FMin => "fmin",
            AtomicRmwOp::UIncWrap => "uinc_wrap",
            AtomicRmwOp::UDecWrap => "udec_wrap",
        }
    }
}

/// Call tail markers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TailKind {
    /// No marker.
    #[default]
    None,
    /// `tail`: the callee does not access the caller's stack, so the call may be a tail call.
    Tail,
    /// `musttail`: the call must be a tail call.
    MustTail,
    /// `notail`: the call must not be a tail call.
    NoTail,
}

/// Calling conventions. Only C and fast are accepted; any other is refused at decode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CallConv {
    /// The C calling convention (`ccc`), LLVM's default.
    #[default]
    C,
    /// `fastcc`.
    Fast,
}

/// A power-of-two alignment in bytes, stored as its base-2 logarithm (at most 2^32, as in LLVM).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Align(u8);

impl Align {
    /// The largest alignment LLVM represents.
    pub const MAX_LOG2: u8 = 32;

    /// The alignment of `bytes`, if it is a power of two no larger than 2^32.
    pub fn from_bytes(bytes: u64) -> Option<Align> {
        if bytes.is_power_of_two() && bytes.trailing_zeros() <= u32::from(Align::MAX_LOG2) {
            u8::try_from(bytes.trailing_zeros()).ok().map(Align)
        } else {
            None
        }
    }

    /// The alignment 2^`log2`, if `log2 <= 32`.
    pub fn from_log2(log2: u8) -> Option<Align> {
        (log2 <= Align::MAX_LOG2).then_some(Align(log2))
    }

    /// The alignment in bytes.
    pub fn bytes(self) -> u64 {
        1u64 << self.0
    }

    /// The base-2 logarithm.
    #[cfg(test)]
    pub fn log2(self) -> u8 {
        self.0
    }
}

macro_rules! flag_set {
    (
        $(#[$doc:meta])* $name:ident { $($(#[$fdoc:meta])* $flag:ident = $bit:expr),* $(,)? }
        $(#[$all:meta])* ALL
    ) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub struct $name(u8);

        impl $name {
            /// No flags.
            pub const NONE: $name = $name(0);
            $($(#[$fdoc])* pub const $flag: $name = $name($bit);)*
            $(#[$all])*
            pub const ALL: &'static [($name, &'static str)] =
                &[$(($name::$flag, stringify!($flag))),*];

            /// Both sets.
            pub const fn union(self, other: $name) -> $name {
                $name(self.0 | other.0)
            }

            /// Whether every flag of `other` is set.
            pub const fn contains(self, other: $name) -> bool {
                self.0 & other.0 == other.0
            }

            /// The raw bits.
            #[cfg(test)]
            pub const fn bits(self) -> u8 {
                self.0
            }
        }
    };
}

flag_set!(
    /// Integer wrap/exactness flags.
    IntFlags {
        /// No unsigned wrap.
        NUW = 1,
        /// No signed wrap.
        NSW = 2,
        /// Exact division or shift.
        EXACT = 4,
        /// `or` operands share no set bit.
        DISJOINT = 8,
    }
    /// Every flag, with its mnemonic (the IR printer's).
    ALL
);

flag_set!(
    /// Fast-math flags.
    FastMath {
        /// Reassociation.
        REASSOC = 1,
        /// No NaNs.
        NNAN = 2,
        /// No infinities.
        NINF = 4,
        /// No signed zeros.
        NSZ = 8,
        /// Allow reciprocal.
        ARCP = 16,
        /// Allow contraction.
        CONTRACT = 32,
        /// Approximate functions.
        AFN = 64,
    }
    /// Every flag, with its mnemonic.
    ALL
);

impl FastMath {
    /// Whether no flag is set.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Loop hints decoded from a `!llvm.loop` attachment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LoopHints {
    /// The `llvm.loop.unroll.*` request.
    pub unroll: Unroll,
    /// `llvm.loop.mustprogress`.
    pub must_progress: bool,
}

/// An unroll request from `llvm.loop.unroll.*`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Unroll {
    /// No unroll request.
    #[default]
    Default,
    /// `llvm.loop.unroll.disable`.
    Disable,
    /// `llvm.loop.unroll.enable`.
    Enable,
    /// `llvm.loop.unroll.full`.
    Full,
    /// `llvm.loop.unroll.count`: unroll by this factor.
    Count(u32),
}

/// A `!range` attachment: half-open `[lo, hi)` pairs of zero-extended bit patterns.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RangeMd {
    /// The `[lo, hi)` pairs, in attachment order.
    pub ranges: Box<[(u64, u64)]>,
}

impl RangeMd {
    /// The heap bytes of the pairs.
    pub fn heap_bytes(&self) -> usize {
        let RangeMd { ranges } = self;
        std::mem::size_of_val::<[(u64, u64)]>(ranges)
    }
}
