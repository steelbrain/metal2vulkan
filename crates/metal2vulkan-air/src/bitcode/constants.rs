//! `CONSTANTS` blocks.
//!
//! A constants block appends values to the value table in record order. Aggregates and constant
//! expressions may refer to constants later in the same block (LLVM's value enumerator orders the
//! pool by type and frequency, not by dependency), so records are read into a raw list first and
//! then interned into the IR's [`ConstTable`] in dependency order, iteratively. A reference to a
//! value that is neither an earlier constant nor one of this block is `BadValueRef`, as is a
//! cycle. `DATA`, `STRING`, and `CSTRING` expand to aggregates of scalar constants, and so does an
//! `INTEGER` or `FLOAT` record of a vector type, a splat of its one value (LLVM 23 writes every
//! non-zero floating-point vector splat so, where older writers used `DATA`), so either spelling
//! decodes to the same constant. A scalar integer or float `NULL` is its zero value. The same
//! decoder serves the module-level block and a function's local block.

use super::codes::*;
use super::stream::{signed_vbr, Stream};
use super::types::{grow, BitcodeTypes};
use crate::budget::Meter;
use crate::error::{BitcodeError, InputError, InputErrorKind, Location};
use crate::error::{ConstantRecord, Error, Feature};
use crate::ir::{
    BinOp, CastOp, Const, ConstExpr, ConstId, ConstTable, Constant, InlineAsm, IntFlags, Type,
    TypeId, TypeTable,
};

/// One decoded constant: its id and the bitcode type index it was declared with.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DecodedConst {
    pub id: ConstId,
    pub bc_type: u32,
}

/// What the constant decoder reads and writes.
pub(crate) struct ConstCx<'c> {
    pub bc_types: &'c BitcodeTypes,
    pub types: &'c TypeTable,
    pub consts: &'c mut ConstTable,
}

/// A constant record whose operands are value ids (absolute), not yet interned.
#[derive(Clone, Debug)]
enum Raw {
    /// No value operands.
    Ready(Constant),
    Aggregate {
        ty: TypeId,
        elems: Box<[u64]>,
    },
    Cast {
        ty: TypeId,
        op: CastOp,
        v: u64,
    },
    BinOp {
        ty: TypeId,
        op: BinOp,
        flags: IntFlags,
        a: u64,
        b: u64,
    },
    Gep {
        ty: TypeId,
        src_ty: TypeId,
        inbounds: bool,
        ops: Box<[u64]>,
    },
}

impl Raw {
    fn for_each_operand(&self, mut f: impl FnMut(u64) -> Result<(), Error>) -> Result<(), Error> {
        match self {
            Raw::Ready(_) => Ok(()),
            Raw::Aggregate { elems: ops, .. } | Raw::Gep { ops, .. } => {
                ops.iter().try_for_each(|&v| f(v))
            }
            Raw::Cast { v, .. } => f(*v),
            Raw::BinOp { a, b, .. } => {
                f(*a)?;
                f(*b)
            }
        }
    }
}

/// Decode a `CONSTANTS` block that was just entered. Its first constant gets value id `base`;
/// `prior(id)` resolves ids below `base` that name constants (`None` for any other value).
/// Returns the block's constants in record order.
pub(crate) fn decode(
    s: &mut Stream<'_>,
    cx: &mut ConstCx<'_>,
    base: u64,
    prior: &dyn Fn(u64) -> Option<ConstId>,
    m: &mut Meter,
) -> Result<Vec<DecodedConst>, Error> {
    const B: u32 = CONSTANTS_BLOCK;
    let mut raw: Vec<(Raw, u32)> = Vec::new();
    let mut cur: Option<(u32, TypeId)> = None;
    let bc_types = cx.bc_types;
    let types = cx.types;
    s.records(m, B, |s, code, _, m| {
        if code == CST_SETTYPE {
            let idx = s.op32(B, code, 0)?;
            let ty = bc_types
                .ty(u64::from(idx))
                .ok_or_else(|| s.err(BitcodeError::BadType))?;
            cur = Some((idx, ty));
            return Ok(());
        }
        let (bc, ty) = cur.ok_or_else(|| s.malformed(B, code))?;
        let entry = record(s, cx.consts, bc_types, types, code, (bc, ty), m)?;
        grow(m, &mut raw)?;
        raw.push((entry, bc));
        Ok(())
    })?;
    let at = s.location();
    let ids = resolve(&raw, cx.consts, base, prior, m, &at)?;
    let mut out = Vec::new();
    m.reserve(&mut out, raw.len())?;
    for ((_, bc_type), id) in raw.iter().zip(ids) {
        out.push(DecodedConst {
            id,
            bc_type: *bc_type,
        });
    }
    Ok(out)
}

/// Decode one constant record of type `ty` (bitcode type index `bc`; not `SETTYPE`).
fn record(
    s: &Stream<'_>,
    consts: &mut ConstTable,
    bc_types: &BitcodeTypes,
    types: &TypeTable,
    code: u32,
    (bc, ty): (u32, TypeId),
    m: &mut Meter,
) -> Result<Raw, Error> {
    const B: u32 = CONSTANTS_BLOCK;
    let bad = || s.malformed(B, code);
    let ready = |value| Ok(Raw::Ready(Constant { ty, value }));
    match code {
        // LLVM writes every zero as `NULL`; a scalar integer or float zero is canonicalized to
        // its value so each scalar has one spelling (`Null` stays for pointers, vectors, and
        // aggregates).
        CST_NULL => ready(match types.get(ty) {
            Some(Type::Int(_)) => Const::Int(0),
            Some(t) if t.float_bits().is_some() => Const::Float(0),
            _ => Const::Null,
        }),
        CST_UNDEF => ready(Const::Undef),
        CST_POISON => ready(Const::Poison),
        CST_INTEGER => {
            let bits = types.int_bits(ty).ok_or_else(bad)?;
            let v = signed_vbr(s.op(B, code, 0)?) as u64;
            splat(consts, types, ty, Const::Int(v & mask(u32::from(bits))), m)
        }
        // LLVM writes an integer of more than 64 bits as WIDE_INTEGER and every other as INTEGER;
        // the type table refuses the wider types (`Feature::Type`), so a WIDE_INTEGER here is of a
        // type of at most 64 bits, which LLVM never writes.
        CST_WIDE_INTEGER => Err(bad()),
        CST_FLOAT => {
            let width = types.float_bits(ty).ok_or_else(bad)?;
            let v = s.op(B, code, 0)?;
            if v & !mask(width) != 0 {
                return Err(bad());
            }
            splat(consts, types, ty, Const::Float(v), m)
        }
        CST_AGGREGATE => Ok(Raw::Aggregate {
            ty,
            elems: copy(s.ops(), m)?,
        }),
        CST_STRING | CST_CSTRING | CST_DATA => {
            let elem = match types.get(ty) {
                Some(Type::Array { elem, .. } | Type::Vector { elem, .. }) => *elem,
                _ => return Err(bad()),
            };
            let scalar = |v: u64| -> Option<Const> {
                match types.get(elem)? {
                    Type::Int(w) if v & !mask(u32::from(*w)) == 0 => Some(Const::Int(v)),
                    t if code == CST_DATA && t.float_bits().is_some_and(|w| v & !mask(w) == 0) => {
                        Some(Const::Float(v))
                    }
                    _ => None,
                }
            };
            let nul = if code == CST_CSTRING { Some(0) } else { None };
            let ops = s.ops();
            let mut elems = Vec::new();
            m.reserve(&mut elems, ops.len() + usize::from(nul.is_some()))?;
            for &v in ops.iter().chain(nul.iter()) {
                m.tick(1)?;
                let value = scalar(v).ok_or_else(bad)?;
                elems.push(consts.intern(Constant { ty: elem, value }, m)?);
            }
            ready(Const::Aggregate(elems.into_boxed_slice()))
        }
        CST_CE_CAST => {
            let op = match s.op(B, code, 0)? {
                0 => CastOp::Trunc,
                1 => CastOp::ZExt,
                2 => CastOp::SExt,
                3 => CastOp::FPToUI,
                4 => CastOp::FPToSI,
                5 => CastOp::UIToFP,
                6 => CastOp::SIToFP,
                7 => CastOp::FPTrunc,
                8 => CastOp::FPExt,
                9 => CastOp::PtrToInt,
                10 => CastOp::IntToPtr,
                11 => CastOp::Bitcast,
                12 => CastOp::AddrSpaceCast,
                // CAST_PTRTOADDR: an LLVM cast the IR has no form for.
                13 => {
                    return Err(s.unsupported(Feature::Constant {
                        record: ConstantRecord::PtrToAddr,
                    }))
                }
                _ => return Err(bad()),
            };
            Ok(Raw::Cast {
                ty,
                op,
                v: s.op(B, code, 2)?,
            })
        }
        CST_CE_BINOP => {
            let float = types.float_bits(ty).is_some();
            let op = binop(s.op(B, code, 0)?, float).ok_or_else(bad)?;
            let raw_flags = s.ops().get(3).copied().unwrap_or(0);
            let flags = int_flags(op, raw_flags).ok_or_else(bad)?;
            Ok(Raw::BinOp {
                ty,
                op,
                flags,
                a: s.op(B, code, 1)?,
                b: s.op(B, code, 2)?,
            })
        }
        CST_CE_GEP_OLD | CST_CE_INBOUNDS_GEP | CST_CE_GEP_WITH_INRANGE_INDEX_OLD | CST_CE_GEP => {
            let ops = s.ops();
            // [src_ty?, flags?, n x (type, value)]
            let (src, inbounds, pairs) = match code {
                CST_CE_GEP_OLD | CST_CE_INBOUNDS_GEP if ops.len() % 2 == 1 => {
                    (Some(s.op(B, code, 0)?), code == CST_CE_INBOUNDS_GEP, 1)
                }
                CST_CE_GEP_OLD | CST_CE_INBOUNDS_GEP => (None, code == CST_CE_INBOUNDS_GEP, 0),
                _ => (Some(s.op(B, code, 0)?), s.op(B, code, 1)? & 1 != 0, 2),
            };
            let pairs = ops.get(pairs..).unwrap_or(&[]);
            if pairs.is_empty() || pairs.len() % 2 != 0 {
                return Err(bad());
            }
            let src_bc = match src {
                Some(t) => t,
                // The untyped spelling indexes through the base pointer's pointee.
                None => pairs
                    .first()
                    .and_then(|&base_ty| bc_types.pointee(base_ty))
                    .map(u64::from)
                    .ok_or_else(bad)?,
            };
            let src_ty = bc_types
                .ty(src_bc)
                .ok_or_else(|| s.err(BitcodeError::BadType))?;
            let mut vals = Vec::new();
            m.reserve(&mut vals, pairs.len() / 2)?;
            vals.extend(pairs.as_chunks::<2>().0.iter().map(|[_, v]| *v));
            Ok(Raw::Gep {
                ty,
                src_ty,
                inbounds,
                ops: vals.into_boxed_slice(),
            })
        }
        CST_INLINEASM_OLD | CST_INLINEASM_OLD2 | CST_INLINEASM_OLD3 | CST_INLINEASM => {
            inline_asm(s, bc_types, types, code, (bc, ty), m)
        }
        c => match cst_unsupported(c) {
            Some(record) => Err(s.unsupported(Feature::Constant { record })),
            None => Err(s.err(BitcodeError::UnknownRecord { block: B, code })),
        },
    }
}

/// An inline assembly record of type `ty` (bitcode index `bc`), in any of LLVM's four
/// spellings: `[fnty?, flags, asmlen, asm..., conslen, cons...]`, one byte per operand. `INLINEASM`
/// names its function type; the older three take it from the pointee of their typed pointer.
/// The flags are `sideeffect` (bit 0), `alignstack` (1), `inteldialect` (2, from `INLINEASM_OLD2`)
/// and `unwind` (3, from `INLINEASM_OLD3`); a bit the spelling does not define is malformed, as is
/// a type other than a pointer, a function type that is none, or a byte operand above 255.
fn inline_asm(
    s: &Stream<'_>,
    bc_types: &BitcodeTypes,
    types: &TypeTable,
    code: u32,
    (bc, ty): (u32, TypeId),
    m: &mut Meter,
) -> Result<Raw, Error> {
    let bad = || s.malformed(CONSTANTS_BLOCK, code);
    let ops = s.ops();
    let (fn_bc, rest) = match code {
        CST_INLINEASM => {
            let (&f, rest) = ops.split_first().ok_or_else(bad)?;
            (f, rest)
        }
        _ => (
            u64::from(bc_types.pointee(u64::from(bc)).ok_or_else(bad)?),
            ops,
        ),
    };
    let is_fn = bc_types
        .ty(fn_bc)
        .is_some_and(|f| matches!(types.get(f), Some(Type::Function { .. })));
    if !is_fn || !matches!(types.get(ty), Some(Type::Ptr { .. })) {
        return Err(bad());
    }
    let (&flags, rest) = rest.split_first().ok_or_else(bad)?;
    let defined: u64 = match code {
        CST_INLINEASM_OLD => 0b11,
        CST_INLINEASM_OLD2 => 0b111,
        _ => 0b1111,
    };
    if flags & !defined != 0 {
        return Err(bad());
    }
    let (template, rest) = asm_bytes(rest, &bad, m)?;
    let (constraints, rest) = asm_bytes(rest, &bad, m)?;
    if !rest.is_empty() {
        return Err(bad());
    }
    let asm = InlineAsm {
        template,
        constraints,
        side_effects: flags & 1 != 0,
        align_stack: flags & 2 != 0,
        intel_dialect: flags & 4 != 0,
        can_throw: flags & 8 != 0,
    };
    m.charge(std::mem::size_of::<InlineAsm>())?;
    Ok(Raw::Ready(Constant {
        ty,
        value: Const::InlineAsm(Box::new(asm)),
    }))
}

/// A length-prefixed string of one byte per operand at the front of `ops`, and the operands after
/// it; `bad()` when the length runs past the record or a byte is above 255.
fn asm_bytes<'o>(
    ops: &'o [u64],
    bad: &dyn Fn() -> Error,
    m: &mut Meter,
) -> Result<(Box<[u8]>, &'o [u64]), Error> {
    let (&len, rest) = ops.split_first().ok_or_else(bad)?;
    let len = usize::try_from(len)
        .ok()
        .filter(|&n| n <= rest.len())
        .ok_or_else(bad)?;
    let (text, rest) = rest.split_at(len);
    let mut out = Vec::new();
    m.reserve(&mut out, len)?;
    for &b in text {
        out.push(u8::try_from(b).map_err(|_| bad())?);
    }
    Ok((out.into_boxed_slice(), rest))
}

/// The scalar constant `value` of type `ty`, or, for a vector `ty`, the aggregate holding
/// `value` in every lane (the vector `INTEGER`/`FLOAT` splat spelling).
fn splat(
    consts: &mut ConstTable,
    types: &TypeTable,
    ty: TypeId,
    value: Const,
    m: &mut Meter,
) -> Result<Raw, Error> {
    let Some(lanes) = types.lanes(ty) else {
        return Ok(Raw::Ready(Constant { ty, value }));
    };
    let lane = consts.intern(
        Constant {
            ty: types.scalar(ty),
            value,
        },
        m,
    )?;
    m.tick(lanes)?;
    let lanes = usize::try_from(lanes).unwrap_or(usize::MAX);
    let mut elems = Vec::new();
    m.reserve(&mut elems, lanes)?;
    elems.resize(lanes, lane);
    Ok(Raw::Ready(Constant {
        ty,
        value: Const::Aggregate(elems.into_boxed_slice()),
    }))
}

/// Intern the raw constants in dependency order (iterative post-order).
fn resolve(
    raw: &[(Raw, u32)],
    consts: &mut ConstTable,
    base: u64,
    prior: &dyn Fn(u64) -> Option<ConstId>,
    m: &mut Meter,
    at: &Location,
) -> Result<Vec<ConstId>, Error> {
    let n = raw.len();
    let bad = || bad_value(at.clone());
    let mut state: Vec<Option<ConstId>> = Vec::new();
    m.reserve(&mut state, n)?;
    state.resize(n, None);
    let mut open: Vec<bool> = Vec::new();
    m.reserve(&mut open, n)?;
    open.resize(n, false);
    let local = |id: u64| -> Option<usize> {
        id.checked_sub(base)
            .and_then(|i| usize::try_from(i).ok())
            .filter(|&i| i < n)
    };
    let mut stack: Vec<(usize, bool)> = Vec::new();
    for root in 0..n {
        if state.get(root).copied().flatten().is_some() {
            continue;
        }
        grow(m, &mut stack)?;
        stack.push((root, false));
        while let Some((i, expanded)) = stack.pop() {
            m.tick(1)?;
            let (r, _) = raw.get(i).ok_or_else(bad)?;
            if expanded {
                let get = |v: u64| -> Result<ConstId, Error> {
                    match local(v) {
                        Some(l) => state.get(l).copied().flatten().ok_or_else(bad),
                        None if v < base => prior(v).ok_or_else(bad),
                        None => Err(bad()),
                    }
                };
                let c = build(r, get, m)?;
                let id = consts.intern(c, m)?;
                if let Some(slot) = state.get_mut(i) {
                    *slot = Some(id);
                }
                continue;
            }
            if state.get(i).copied().flatten().is_some() {
                continue;
            }
            if open.get(i).copied().unwrap_or(true) {
                return Err(bad());
            }
            if let Some(o) = open.get_mut(i) {
                *o = true;
            }
            grow(m, &mut stack)?;
            stack.push((i, true));
            r.for_each_operand(|v| {
                match local(v) {
                    Some(l) if state.get(l).copied().flatten().is_none() => {
                        grow(m, &mut stack)?;
                        stack.push((l, false));
                    }
                    Some(_) => {}
                    None if v < base && prior(v).is_some() => {}
                    None => return Err(bad()),
                }
                Ok(())
            })?;
        }
    }
    let mut ids = Vec::new();
    m.reserve(&mut ids, n)?;
    for c in state {
        ids.push(c.ok_or_else(bad)?);
    }
    Ok(ids)
}

/// The constant for a raw record whose operands `get` resolves.
fn build(
    r: &Raw,
    get: impl Fn(u64) -> Result<ConstId, Error>,
    m: &mut Meter,
) -> Result<Constant, Error> {
    let many = |ops: &[u64], m: &mut Meter| -> Result<Box<[ConstId]>, Error> {
        let mut out = Vec::new();
        m.reserve(&mut out, ops.len())?;
        for &v in ops {
            out.push(get(v)?);
        }
        Ok(out.into_boxed_slice())
    };
    Ok(match r {
        Raw::Ready(c) => c.clone(),
        Raw::Aggregate { ty, elems } => Constant {
            ty: *ty,
            value: Const::Aggregate(many(elems, m)?),
        },
        Raw::Cast { ty, op, v } => Constant {
            ty: *ty,
            value: Const::Expr(ConstExpr::Cast {
                op: *op,
                ops: [get(*v)?],
            }),
        },
        Raw::BinOp {
            ty,
            op,
            flags,
            a,
            b,
        } => Constant {
            ty: *ty,
            value: Const::Expr(ConstExpr::BinOp {
                op: *op,
                flags: *flags,
                ops: [get(*a)?, get(*b)?],
            }),
        },
        Raw::Gep {
            ty,
            src_ty,
            inbounds,
            ops,
        } => Constant {
            ty: *ty,
            value: Const::Expr(ConstExpr::Gep {
                src_ty: *src_ty,
                inbounds: *inbounds,
                ops: many(ops, m)?,
            }),
        },
    })
}

/// The binary operator for bitcode opcode `code` on integer or floating-point operands.
pub(crate) fn binop(code: u64, float: bool) -> Option<BinOp> {
    Some(match (code, float) {
        (0, false) => BinOp::Add,
        (1, false) => BinOp::Sub,
        (2, false) => BinOp::Mul,
        (3, false) => BinOp::UDiv,
        (4, false) => BinOp::SDiv,
        (5, false) => BinOp::URem,
        (6, false) => BinOp::SRem,
        (7, false) => BinOp::Shl,
        (8, false) => BinOp::LShr,
        (9, false) => BinOp::AShr,
        (10, false) => BinOp::And,
        (11, false) => BinOp::Or,
        (12, false) => BinOp::Xor,
        (0, true) => BinOp::FAdd,
        (1, true) => BinOp::FSub,
        (2, true) => BinOp::FMul,
        (4, true) => BinOp::FDiv,
        (6, true) => BinOp::FRem,
        _ => return None,
    })
}

/// The wrap/exact/disjoint flags of an integer binary operator's optional flags operand.
pub(crate) fn int_flags(op: BinOp, raw: u64) -> Option<IntFlags> {
    let bit = |n: u32, f: IntFlags| {
        if raw & (1 << n) != 0 {
            f
        } else {
            IntFlags::NONE
        }
    };
    let flags = match op {
        // OBO_NO_UNSIGNED_WRAP = 0, OBO_NO_SIGNED_WRAP = 1.
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl => {
            bit(0, IntFlags::NUW).union(bit(1, IntFlags::NSW))
        }
        // PEO_EXACT = 0.
        BinOp::UDiv | BinOp::SDiv | BinOp::LShr | BinOp::AShr => bit(0, IntFlags::EXACT),
        // PDI_DISJOINT = 0.
        BinOp::Or => bit(0, IntFlags::DISJOINT),
        _ => IntFlags::NONE,
    };
    // Unknown flag bits are refused rather than dropped.
    let known = match op {
        BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Shl => 3,
        BinOp::UDiv | BinOp::SDiv | BinOp::LShr | BinOp::AShr | BinOp::Or => 1,
        _ => 0,
    };
    (raw & !known == 0).then_some(flags)
}

/// The low `bits` bits set.
pub(crate) fn mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn copy(ops: &[u64], m: &mut Meter) -> Result<Box<[u64]>, Error> {
    let mut out = Vec::new();
    m.reserve(&mut out, ops.len())?;
    out.extend_from_slice(ops);
    Ok(out.into_boxed_slice())
}

fn bad_value(at: Location) -> Error {
    InputError::new(InputErrorKind::Bitcode(BitcodeError::BadValueRef), at).into()
}
