//! The closed `llvm.*` set: kept operations, the markers decode drops, and the Apple GPU family
//! `llvm.agx<N>.*`.
//!
//! An `llvm.*` name is a base name followed by its overload suffix. The overload types are read
//! from the declaration and the suffix must spell them (LLVM's own reader refuses a misspelled
//! intrinsic). Pointer overloads are spelled `p<space>` (opaque) or `p<space><pointee>` (typed);
//! only the space is checked, pointees being decode-only.

use super::sig::{ImmKind, Immediates, Operand, Signature};
use super::ty::{Elem, NumTy};
use crate::error::IntrinsicError;
use crate::ir::{AddrSpace, Sym, Type, TypeId, TypeTable};

/// A kept floating-point `llvm.*` operation, `T (T × n)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LlvmFloatOp {
    /// `llvm.fmuladd`: `a * b + c`, fused or not as the backend chooses (LangRef).
    FMulAdd,
    /// `llvm.fma`: `a * b + c`, rounded once.
    Fma,
    /// `llvm.fabs`: the absolute value.
    FAbs,
    /// `llvm.minnum`: IEEE `minNum`, the lesser operand, or the other one where one is a NaN.
    MinNum,
    /// `llvm.maxnum`: IEEE `maxNum`, the greater operand, or the other one where one is a NaN.
    MaxNum,
    /// `llvm.exp2`: libm's `exp2` (LangRef), which Apple's math library defines `air.exp2` as.
    Exp2,
    /// `llvm.log2`: libm's `log2` (LangRef), which Apple's math library defines `air.log2` as.
    Log2,
    /// `llvm.sqrt`: the IEEE square root (LangRef), which Apple's math library defines
    /// `air.sqrt` as.
    Sqrt,
    /// `llvm.copysign`: the magnitude of the first operand with the sign bit of the second
    /// (LangRef), exact for every operand, NaNs and zeros included.
    CopySign,
}

/// A kept integer `llvm.*` operation, `T (T × n)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LlvmIntOp {
    /// `llvm.smax`: the signed maximum.
    SMax,
    /// `llvm.smin`: the signed minimum.
    SMin,
    /// `llvm.umax`: the unsigned maximum.
    UMax,
    /// `llvm.umin`: the unsigned minimum.
    UMin,
    /// `llvm.fshl`: funnel shift left, the high half of `a:b` shifted left by `c` modulo the width.
    Fshl,
    /// `llvm.fshr`: funnel shift right, the low half of `a:b` shifted right by `c` modulo the
    /// width.
    Fshr,
    /// `llvm.bswap`: the bytes in reverse order (a width that is a multiple of 16 bits).
    Bswap,
    /// `llvm.bitreverse`: the bits in reverse order.
    BitReverse,
    /// `llvm.ctpop`: the number of set bits.
    Ctpop,
    /// `llvm.uadd.sat`: unsigned saturating addition.
    UAddSat,
    /// `llvm.usub.sat`: unsigned saturating subtraction.
    USubSat,
    /// `llvm.sadd.sat`: signed saturating addition.
    SAddSat,
    /// `llvm.ssub.sat`: signed saturating subtraction.
    SSubSat,
}

/// The memory an AGX emask access names: `global` (device) or `local` (threadgroup).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AgxSpace {
    /// `global`: device memory.
    Global,
    /// `local`: threadgroup memory.
    Local,
}

impl AgxSpace {
    /// The pointer operand's address space.
    pub const fn space(self) -> AddrSpace {
        match self {
            AgxSpace::Global => AddrSpace::Device,
            AgxSpace::Local => AddrSpace::Threadgroup,
        }
    }
}

/// An Apple GPU (`llvm.agx<N>.*`) operation.
///
/// The modeled operations are those whose contract was measured on device (the old engine's
/// `agx_emask` and `matrix` lowerings): the `agx3` lane-masked accesses and their mask builder
/// (lane `i` of an emask access is active exactly when bit `i` of the first `i16` mask is set; the
/// second `i16` is a constant of the shape; the third, the element stride, has no effect), the
/// `agx2` cluster number, and the `agx2` distributed-register matrix multiply-adds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AgxOp {
    /// `llvm.agx2.cluster.num`: `i32 ()`, the physical GPU cluster the threadgroup runs on.
    ClusterNum,
    /// `llvm.agx2.{f16,f32}matmad{4x4,8x8}.v2{f16,f32}`: `<2 x T> (<2 x T> a, <2 x T> b,
    /// <2 x T> c)`, `c + a·b` over the 8x8 tile (or over each of its four 4x4 quadrants) whose
    /// cells are distributed two per lane over the 32 lanes of the simdgroup. `elem` is `F16`
    /// or `F32`; `dim` is 4 or 8.
    MatMad {
        /// The element: `F16` or `F32`.
        elem: Elem,
        /// The tile dimension: 4 (each 4x4 quadrant) or 8.
        dim: u8,
    },
    /// `llvm.agx3.edgecheck`: `i16 (i32 base, i32 low, i32 count)`, the four-lane activity mask.
    EdgeCheck,
    /// `llvm.agx3.load.with.emask.<space>.<T>`: `T (ptr, i16 mask, i16 shape, i16 stride)`.
    LoadEmask {
        /// The memory the name spells: `global` (device) or `local` (threadgroup).
        space: AgxSpace,
        /// The loaded type: a 2- or 4-lane vector of `i8`, `i16`, or `i32`.
        ty: NumTy,
    },
    /// `llvm.agx3.store.with.emask.<space>.<T>`: `void (ptr, T, i16 mask, i16 shape, i16
    /// stride)`.
    StoreEmask {
        /// The memory the name spells: `global` (device) or `local` (threadgroup).
        space: AgxSpace,
        /// The stored type: a 2- or 4-lane vector of `i8`, `i16`, or `i32`.
        ty: NumTy,
    },
    /// No exact contract is modeled (the `wide` forms, one-lane vectors, and every other AGX
    /// operation); lowering refuses it. The name is kept for diagnostics.
    Unmodeled(Sym),
}

impl AgxOp {
    /// Parse the name after `llvm.agx<N>.` for generation `gen`.
    fn parse(gen: &str, rest: &str) -> Option<AgxOp> {
        match gen {
            "2" => AgxOp::parse2(rest),
            "3" => AgxOp::parse3(rest),
            _ => None,
        }
    }

    /// Parse an `agx2` name: the cluster number, and the matrix multiply-adds, whose overload
    /// suffix must spell their element (`v2f16` or `v2f32`).
    fn parse2(rest: &str) -> Option<AgxOp> {
        if rest == "cluster.num" {
            return Some(AgxOp::ClusterNum);
        }
        let (op, suffix) = rest.split_once('.')?;
        let (elem, dim) = match op {
            "f16matmad4x4" => (Elem::F16, 4),
            "f16matmad8x8" => (Elem::F16, 8),
            "f32matmad4x4" => (Elem::F32, 4),
            "f32matmad8x8" => (Elem::F32, 8),
            _ => return None,
        };
        (NumTy::parse(suffix)? == NumTy { elem, lanes: 2 }).then_some(AgxOp::MatMad { elem, dim })
    }

    /// Parse an `agx3` name: the edge check and the emask accesses.
    fn parse3(rest: &str) -> Option<AgxOp> {
        if rest == "edgecheck" {
            return Some(AgxOp::EdgeCheck);
        }
        let (load, rest) = if let Some(r) = rest.strip_prefix("load.with.emask.") {
            (true, r)
        } else {
            (false, rest.strip_prefix("store.with.emask.")?)
        };
        let (space, ty) = rest.split_once('.')?;
        let space = match space {
            "global" => AgxSpace::Global,
            "local" => AgxSpace::Local,
            _ => return None,
        };
        let ty = NumTy::parse(ty)?;
        let ok = matches!(ty.elem, Elem::I8 | Elem::I16 | Elem::I32) && matches!(ty.lanes, 2 | 4);
        ok.then_some(if load {
            AgxOp::LoadEmask { space, ty }
        } else {
            AgxOp::StoreEmask { space, ty }
        })
    }

    fn signature(&self) -> Option<Signature> {
        let i16s = Operand::Num(NumTy::scalar(Elem::I16));
        let i32s = Operand::Num(NumTy::scalar(Elem::I32));
        Some(match *self {
            AgxOp::ClusterNum => Signature::new(i32s, &[]),
            AgxOp::MatMad { elem, .. } => {
                let v2 = Operand::Num(NumTy { elem, lanes: 2 });
                Signature::new(v2, &[v2, v2, v2])
            }
            AgxOp::EdgeCheck => Signature::new(i16s, &[i32s, i32s, i32s]),
            AgxOp::LoadEmask { space, ty } => Signature::new(
                Operand::Num(ty),
                &[Operand::Ptr(space.space()), i16s, i16s, i16s],
            ),
            AgxOp::StoreEmask { space, ty } => Signature::new(
                Operand::Void,
                &[
                    Operand::Ptr(space.space()),
                    Operand::Num(ty),
                    i16s,
                    i16s,
                    i16s,
                ],
            ),
            AgxOp::Unmodeled(_) => return None,
        })
    }
}

/// A decoded `llvm.*` intrinsic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LlvmIntrinsic {
    /// A kept floating-point operation.
    Float {
        /// The operation.
        op: LlvmFloatOp,
        /// The overload type: the result and every operand.
        ty: NumTy,
    },
    /// A kept integer operation.
    Int {
        /// The operation.
        op: LlvmIntOp,
        /// The overload type: the result and every operand.
        ty: NumTy,
    },
    /// `llvm.abs`, with its `is_int_min_poison` immediate.
    Abs {
        /// The integer overload type.
        ty: NumTy,
        /// The `is_int_min_poison` immediate: whether the result for the minimum integer is poison.
        int_min_poison: bool,
    },
    /// `llvm.cttz`, with its `is_zero_poison` immediate.
    Cttz {
        /// The integer overload type.
        ty: NumTy,
        /// The `is_zero_poison` immediate: whether the result for a zero operand is poison.
        zero_poison: bool,
    },
    /// `llvm.ctlz`, with its `is_zero_poison` immediate.
    Ctlz {
        /// The integer overload type.
        ty: NumTy,
        /// The `is_zero_poison` immediate: whether the result for a zero operand is poison.
        zero_poison: bool,
    },
    /// `llvm.memset(dst, i8, len, i1 volatile)`.
    Memset {
        /// The destination pointer's address space.
        dst: AddrSpace,
        /// The length operand's type: `I32` or `I64`.
        len: Elem,
        /// The `isvolatile` immediate.
        volatile: bool,
    },
    /// `llvm.memcpy(dst, src, len, i1 volatile)`.
    Memcpy {
        /// The destination pointer's address space.
        dst: AddrSpace,
        /// The source pointer's address space.
        src: AddrSpace,
        /// The length operand's type: `I32` or `I64`.
        len: Elem,
        /// The `isvolatile` immediate.
        volatile: bool,
    },
    /// `llvm.memmove(dst, src, len, i1 volatile)`.
    Memmove {
        /// The destination pointer's address space.
        dst: AddrSpace,
        /// The source pointer's address space.
        src: AddrSpace,
        /// The length operand's type: `I32` or `I64`.
        len: Elem,
        /// The `isvolatile` immediate.
        volatile: bool,
    },
    /// An Apple GPU operation, `llvm.agx<N>.*`.
    Agx(AgxOp),
}

/// The markers decode drops: semantic no-ops when their facts are not exploited (dropping a
/// lifetime marker only extends a lifetime), and the AGX yield, whose only effect is a delay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dropped {
    /// `llvm.lifetime.start`: the start of an object's lifetime.
    LifetimeStart,
    /// `llvm.lifetime.end`: the end of an object's lifetime.
    LifetimeEnd,
    /// `llvm.assume`: an optimizer hint that its `i1` operand is true.
    Assume,
    /// `llvm.experimental.noalias.scope.decl`: a noalias scope declaration for alias analysis.
    NoAliasScopeDecl,
    /// `llvm.agx3.yield`: `void (i16 n)`, the calling thread pausing for a time that grows with
    /// `n` (the backoff of a spin loop; the refinement). Measured on an M3 Max and an M4: it reads
    /// and writes nothing (every word of a kernel that yields between its arithmetic, stores and
    /// loads is the word without the yields), a decoupled-lookback chain across threadgroups gives
    /// the same words with and without it, and one call costs from under 0.1 µs at `n = 0` to about
    /// 0.7 ms at `n = 65535` (10 to 12 ns per unit of a large `n`). A delay is no answer a program
    /// can read, so decode drops the call.
    AgxYield,
}

/// What a base name decodes to, before its overloads are read.
#[derive(Clone, Copy, Debug)]
pub(super) enum Base {
    Float(LlvmFloatOp),
    Int(LlvmIntOp),
    Abs,
    Cttz,
    Ctlz,
    Memset,
    Memcpy,
    Memmove,
    Drop(Dropped),
}

/// The name of [`Dropped::AgxYield`] after `llvm.`; it has no overload suffix.
const AGX_YIELD: &str = "agx3.yield";

/// Every base name.
pub(super) const BASES: &[(&str, Base)] = &[
    ("fmuladd", Base::Float(LlvmFloatOp::FMulAdd)),
    ("fma", Base::Float(LlvmFloatOp::Fma)),
    ("fabs", Base::Float(LlvmFloatOp::FAbs)),
    ("minnum", Base::Float(LlvmFloatOp::MinNum)),
    ("maxnum", Base::Float(LlvmFloatOp::MaxNum)),
    ("exp2", Base::Float(LlvmFloatOp::Exp2)),
    ("log2", Base::Float(LlvmFloatOp::Log2)),
    ("sqrt", Base::Float(LlvmFloatOp::Sqrt)),
    ("copysign", Base::Float(LlvmFloatOp::CopySign)),
    ("smax", Base::Int(LlvmIntOp::SMax)),
    ("smin", Base::Int(LlvmIntOp::SMin)),
    ("umax", Base::Int(LlvmIntOp::UMax)),
    ("umin", Base::Int(LlvmIntOp::UMin)),
    ("fshl", Base::Int(LlvmIntOp::Fshl)),
    ("fshr", Base::Int(LlvmIntOp::Fshr)),
    ("bswap", Base::Int(LlvmIntOp::Bswap)),
    ("bitreverse", Base::Int(LlvmIntOp::BitReverse)),
    ("ctpop", Base::Int(LlvmIntOp::Ctpop)),
    ("uadd.sat", Base::Int(LlvmIntOp::UAddSat)),
    ("usub.sat", Base::Int(LlvmIntOp::USubSat)),
    ("sadd.sat", Base::Int(LlvmIntOp::SAddSat)),
    ("ssub.sat", Base::Int(LlvmIntOp::SSubSat)),
    ("abs", Base::Abs),
    ("cttz", Base::Cttz),
    ("ctlz", Base::Ctlz),
    ("memset", Base::Memset),
    ("memcpy", Base::Memcpy),
    ("memmove", Base::Memmove),
    ("lifetime.start", Base::Drop(Dropped::LifetimeStart)),
    ("lifetime.end", Base::Drop(Dropped::LifetimeEnd)),
    ("assume", Base::Drop(Dropped::Assume)),
    (
        "experimental.noalias.scope.decl",
        Base::Drop(Dropped::NoAliasScopeDecl),
    ),
];

/// The marker family an `llvm.*` symbol name belongs to when decode drops its calls, judged from
/// the name alone (the same base-name table decode uses). Tools that compare decoded IR with a
/// textual disassembly use this to discount the calls decode removes.
pub fn dropped_family(symbol: &str) -> Option<Dropped> {
    let rest = symbol.strip_prefix("llvm.")?;
    if rest == AGX_YIELD {
        return Some(Dropped::AgxYield);
    }
    BASES
        .iter()
        .filter_map(|&(name, base)| {
            let r = rest.strip_prefix(name)?;
            (r.is_empty() || r.starts_with('.')).then_some((name.len(), base))
        })
        .max_by_key(|&(len, _)| len)
        .and_then(|(_, base)| match base {
            Base::Drop(dropped) => Some(dropped),
            _ => None,
        })
}

impl LlvmFloatOp {
    fn arity(self) -> usize {
        match self {
            LlvmFloatOp::FAbs | LlvmFloatOp::Exp2 | LlvmFloatOp::Log2 | LlvmFloatOp::Sqrt => 1,
            LlvmFloatOp::MinNum | LlvmFloatOp::MaxNum | LlvmFloatOp::CopySign => 2,
            LlvmFloatOp::FMulAdd | LlvmFloatOp::Fma => 3,
        }
    }
}

impl LlvmIntOp {
    fn arity(self) -> usize {
        match self {
            LlvmIntOp::Bswap | LlvmIntOp::BitReverse | LlvmIntOp::Ctpop => 1,
            LlvmIntOp::Fshl | LlvmIntOp::Fshr => 3,
            _ => 2,
        }
    }
}

/// A classified `llvm.*` declaration.
#[derive(Clone, Copy, Debug)]
pub(super) enum Parsed {
    Kept(LlvmIntrinsic),
    Dropped(Dropped),
}

/// Why a name in the `llvm.` namespace did not decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Fail {
    /// Outside the closed set (a base or overload type the model does not name).
    Unknown,
    /// A known base whose declaration or suffix disagrees with it.
    Mismatch,
}

fn check(ok: bool) -> Result<(), Fail> {
    if ok {
        Ok(())
    } else {
        Err(Fail::Mismatch)
    }
}

/// Parse `rest` (the name after `llvm.`) against its declaration type.
pub(super) fn parse(
    rest: &str,
    sym: Sym,
    types: &TypeTable,
    fn_ty: TypeId,
) -> Result<Parsed, Fail> {
    let (base, suffix) = if rest == AGX_YIELD {
        (Base::Drop(Dropped::AgxYield), None)
    } else if let Some(after) = rest.strip_prefix("agx") {
        let n = after.bytes().take_while(u8::is_ascii_digit).count();
        if let (Some(gen), Some(op)) = (
            after.get(..n),
            after.get(n..).and_then(|r| r.strip_prefix('.')),
        ) {
            if n > 0 {
                let op = AgxOp::parse(gen, op).unwrap_or(AgxOp::Unmodeled(sym));
                return Ok(Parsed::Kept(LlvmIntrinsic::Agx(op)));
            }
        }
        return Err(Fail::Unknown);
    } else {
        // The longest base that is the whole name or followed by a `.`.
        BASES
            .iter()
            .filter_map(|&(name, base)| {
                let r = rest.strip_prefix(name)?;
                match r.strip_prefix('.') {
                    Some(s) => Some((name.len(), base, Some(s))),
                    None => r.is_empty().then_some((name.len(), base, None)),
                }
            })
            .max_by_key(|&(len, _, _)| len)
            .map(|(_, base, suffix)| (base, suffix))
            .ok_or(Fail::Unknown)?
    };
    let Some(Type::Function {
        ret,
        params,
        vararg: false,
    }) = types.get(fn_ty)
    else {
        return Err(Fail::Mismatch);
    };
    let param = |i: usize| params.get(i).copied().ok_or(Fail::Mismatch);
    // An overload type the model does not name is outside the closed set, not a mismatch.
    let num = |id: TypeId| NumTy::of(types, id).ok_or(Fail::Unknown);
    let space = |id: TypeId| match types.get(id) {
        Some(Type::Ptr { space }) => Ok(*space),
        _ => Err(Fail::Mismatch),
    };
    let token = |i: usize| suffix.and_then(|s| s.split('.').nth(i));
    let count = |n: usize| check(suffix.map_or(0, |s| s.split('.').count()) == n);
    let spells = |i: usize, ty: NumTy| check(token(i).and_then(NumTy::parse) == Some(ty));
    let spells_ptr = |i: usize, want: AddrSpace| {
        let t = token(i)
            .and_then(|t| t.strip_prefix('p'))
            .unwrap_or_default();
        let n = t.bytes().take_while(u8::is_ascii_digit).count();
        let got = t.get(..n).and_then(|d| d.parse::<u32>().ok());
        check(got == Some(want.raw()))
    };
    let op = match base {
        Base::Float(op) => {
            let ty = num(*ret)?;
            check(ty.elem.is_float() && params.len() == op.arity())?;
            count(1)?;
            spells(0, ty)?;
            LlvmIntrinsic::Float { op, ty }
        }
        Base::Int(op) => {
            let ty = num(*ret)?;
            let bswap_ok = op != LlvmIntOp::Bswap || ty.elem.bits() % 16 == 0;
            check(ty.elem.is_int() && bswap_ok && params.len() == op.arity())?;
            count(1)?;
            spells(0, ty)?;
            LlvmIntrinsic::Int { op, ty }
        }
        Base::Abs | Base::Cttz | Base::Ctlz => {
            let ty = num(*ret)?;
            check(ty.elem.is_int())?;
            count(1)?;
            spells(0, ty)?;
            match base {
                Base::Abs => LlvmIntrinsic::Abs {
                    ty,
                    int_min_poison: false,
                },
                Base::Cttz => LlvmIntrinsic::Cttz {
                    ty,
                    zero_poison: false,
                },
                _ => LlvmIntrinsic::Ctlz {
                    ty,
                    zero_poison: false,
                },
            }
        }
        Base::Memset => {
            let (dst, len) = (space(param(0)?)?, len_elem(types, param(2)?)?);
            count(2)?;
            spells_ptr(0, dst)?;
            spells(1, NumTy::scalar(len))?;
            LlvmIntrinsic::Memset {
                dst,
                len,
                volatile: false,
            }
        }
        Base::Memcpy | Base::Memmove => {
            let (dst, src) = (space(param(0)?)?, space(param(1)?)?);
            let len = len_elem(types, param(2)?)?;
            count(3)?;
            spells_ptr(0, dst)?;
            spells_ptr(1, src)?;
            spells(2, NumTy::scalar(len))?;
            if matches!(base, Base::Memcpy) {
                LlvmIntrinsic::Memcpy {
                    dst,
                    src,
                    len,
                    volatile: false,
                }
            } else {
                LlvmIntrinsic::Memmove {
                    dst,
                    src,
                    len,
                    volatile: false,
                }
            }
        }
        Base::Drop(d) => {
            match d {
                // `(ptr)` (LLVM 21+) or `(i64 size, ptr)`, overloaded on the pointer.
                Dropped::LifetimeStart | Dropped::LifetimeEnd => {
                    let ptr = match params.len() {
                        1 => param(0)?,
                        2 if NumTy::of(types, param(0)?) == Some(NumTy::scalar(Elem::I64)) => {
                            param(1)?
                        }
                        _ => return Err(Fail::Mismatch),
                    };
                    count(1)?;
                    spells_ptr(0, space(ptr)?)?;
                }
                Dropped::Assume => {
                    count(0)?;
                    check(
                        params.len() == 1
                            && NumTy::of(types, param(0)?) == Some(NumTy::scalar(Elem::I1)),
                    )?;
                }
                Dropped::NoAliasScopeDecl => {
                    count(0)?;
                    check(
                        params.len() == 1 && matches!(types.get(param(0)?), Some(Type::Metadata)),
                    )?;
                }
                // `void (i16)`: the operand sets only the delay.
                Dropped::AgxYield => {
                    count(0)?;
                    check(
                        params.len() == 1
                            && NumTy::of(types, param(0)?) == Some(NumTy::scalar(Elem::I16)),
                    )?;
                }
            }
            check(matches!(types.get(*ret), Some(Type::Void)))?;
            return Ok(Parsed::Dropped(d));
        }
    };
    Ok(Parsed::Kept(op))
}

/// The length operand of a memory intrinsic: `i32` or `i64`; another integer width is outside
/// the closed set.
fn len_elem(types: &TypeTable, id: TypeId) -> Result<Elem, Fail> {
    match types.get(id) {
        Some(Type::Int(32)) => Ok(Elem::I32),
        Some(Type::Int(64)) => Ok(Elem::I64),
        Some(Type::Int(_)) => Err(Fail::Unknown),
        _ => Err(Fail::Mismatch),
    }
}

impl LlvmIntrinsic {
    /// The typed signature; `None` for an AGX operation without a modeled contract.
    pub fn signature(&self) -> Option<Signature> {
        let flag = Operand::Imm(ImmKind::Bool);
        let mem = |dst: AddrSpace, src: Option<AddrSpace>, len: Elem| {
            let second = match src {
                Some(s) => Operand::Ptr(s),
                None => Operand::Num(NumTy::scalar(Elem::I8)),
            };
            Signature::new(
                Operand::Void,
                &[
                    Operand::Ptr(dst),
                    second,
                    Operand::Num(NumTy::scalar(len)),
                    flag,
                ],
            )
        };
        Some(match *self {
            LlvmIntrinsic::Float { op, ty } => same(ty, op.arity()),
            LlvmIntrinsic::Int { op, ty } => same(ty, op.arity()),
            LlvmIntrinsic::Abs { ty, .. }
            | LlvmIntrinsic::Cttz { ty, .. }
            | LlvmIntrinsic::Ctlz { ty, .. } => {
                Signature::new(Operand::Num(ty), &[Operand::Num(ty), flag])
            }
            LlvmIntrinsic::Memset { dst, len, .. } => mem(dst, None, len),
            LlvmIntrinsic::Memcpy { dst, src, len, .. }
            | LlvmIntrinsic::Memmove { dst, src, len, .. } => mem(dst, Some(src), len),
            LlvmIntrinsic::Agx(op) => return op.signature(),
        })
    }

    /// Read the call's immediates into the variant.
    pub(super) fn bind(self, imm: &Immediates<'_>) -> Result<LlvmIntrinsic, IntrinsicError> {
        Ok(match self {
            LlvmIntrinsic::Abs { ty, .. } => LlvmIntrinsic::Abs {
                ty,
                int_min_poison: imm.flag(1)?,
            },
            LlvmIntrinsic::Cttz { ty, .. } => LlvmIntrinsic::Cttz {
                ty,
                zero_poison: imm.flag(1)?,
            },
            LlvmIntrinsic::Ctlz { ty, .. } => LlvmIntrinsic::Ctlz {
                ty,
                zero_poison: imm.flag(1)?,
            },
            LlvmIntrinsic::Memset { dst, len, .. } => LlvmIntrinsic::Memset {
                dst,
                len,
                volatile: imm.flag(3)?,
            },
            LlvmIntrinsic::Memcpy { dst, src, len, .. } => LlvmIntrinsic::Memcpy {
                dst,
                src,
                len,
                volatile: imm.flag(3)?,
            },
            LlvmIntrinsic::Memmove { dst, src, len, .. } => LlvmIntrinsic::Memmove {
                dst,
                src,
                len,
                volatile: imm.flag(3)?,
            },
            other => other,
        })
    }
}

/// `T (T × n)`.
fn same(ty: NumTy, n: usize) -> Signature {
    let t = Operand::Num(ty);
    Signature::new(t, [t, t, t].get(..n).unwrap_or(&[]))
}
