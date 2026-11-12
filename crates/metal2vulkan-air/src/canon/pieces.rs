//! A body rebuilt with some memory copies replaced by the typed moves they are.
//!
//! A `memcpy` of a texture table's handles ([`super::table_copy`]) moves bytes that lowering
//! reads as pointers. Each such call is replaced, in its place, by [`Piece`]s: a typed load from
//! the source and a store of it into the destination, each at a constant byte distance from the
//! call's own pointer operands.

use crate::budget::Meter;
use crate::error::Error;
use crate::ir::{AddrSpace, Align, Block, Body, CastOp, Const, ConstId, Constant, FuncId};
use crate::ir::{Idx, Inst, InstId, InstKind, ModuleParts, Type, TypeId, TypeLayouts, Value};

use super::rewrite::invariant;

/// One typed move replacing part of a copy: a load of `ty` from the source at byte distance
/// `at`, stored into the destination at `at`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Piece {
    pub at: u64,
    pub ty: TypeId,
}

impl Piece {
    /// The instructions the piece emits: its load and store, and an address for each at a
    /// distance other than 0.
    fn emitted(&self) -> u64 {
        2 + 2 * u64::from(self.at != 0)
    }
}

/// A copy's replacement: its source and destination operands, and its pieces in order.
#[derive(Clone, Debug)]
pub(super) struct Replacement {
    pub src: Value,
    pub dst: Value,
    pub pieces: Vec<Piece>,
}

impl Replacement {
    fn emitted(&self) -> u64 {
        self.pieces
            .iter()
            .map(Piece::emitted)
            .fold(0, u64::saturating_add)
    }

    /// The bytes the ledger holds for the replacement's pieces.
    pub(super) fn bytes(&self) -> usize {
        self.pieces.capacity() * std::mem::size_of::<Piece>()
    }
}

/// The types and constants the pieces' addresses are built from: a byte, a thread pointer, and
/// each distance's `i64` constant, sorted by distance.
struct Vocab<'a> {
    pub i8t: TypeId,
    pub thread: TypeId,
    pub offsets: &'a [(u64, ConstId)],
}

/// `body` with each call of `replaced` (sorted by instruction) replaced by its pieces, in its
/// place. A replaced instruction is a call that returns nothing, so nothing uses it.
fn rebuild(
    body: &Body,
    replaced: &[(InstId, Replacement)],
    v: &Vocab<'_>,
    m: &mut Meter,
) -> Result<Body, Error> {
    let replacement = |i: InstId| {
        replaced
            .binary_search_by_key(&i, |(c, _)| *c)
            .ok()
            .and_then(|k| replaced.get(k))
            .map(|(_, r)| r)
    };
    // Each old instruction's id in the output: a replaced call's is its last piece's.
    let mut result = Vec::new();
    m.reserve(&mut result, body.insts.len())?;
    let mut next = 0u64;
    for (i, _) in body.insts.iter() {
        m.tick(1)?;
        let n = replacement(i).map_or(1, Replacement::emitted);
        if n == 0 {
            return Err(invariant("a replacement without pieces"));
        }
        next = next
            .checked_add(n)
            .ok_or_else(|| invariant("instruction ids"))?;
        let last = u32::try_from(next - 1).map_err(|_| invariant("instruction ids"))?;
        result.push(InstId::new(last));
    }
    let map = |val: Value| -> Result<Value, Error> {
        match val {
            Value::Inst(i) => result
                .get(i.index())
                .map(|&n| Value::Inst(n))
                .ok_or_else(|| invariant("an operand out of range")),
            other => Ok(other),
        }
    };
    let mut out = Body::default();
    for (_, block) in body.blocks.iter() {
        m.tick(1)?;
        let start = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
        for i in block.inst_ids() {
            let inst = body
                .insts
                .get(i)
                .ok_or_else(|| invariant("instruction out of range"))?;
            if let Some(r) = replacement(i) {
                if !matches!(inst.kind, InstKind::Call(_)) {
                    return Err(invariant("a replaced instruction that is no call"));
                }
                let ends = (map(r.src)?, map(r.dst)?);
                for p in &r.pieces {
                    m.tick(1)?;
                    emit(&mut out, inst.ty, p, ends, v, m)?;
                }
            } else {
                let mut kind = inst.kind.clone();
                let mut failed = None;
                kind.for_each_value_mut(|val| match map(*val) {
                    Ok(n) => *val = n,
                    Err(e) => {
                        failed.get_or_insert(e);
                    }
                });
                if let Some(e) = failed {
                    return Err(e);
                }
                out.push_inst(Inst { ty: inst.ty, kind }, m)?;
            }
            if result.get(i.index()).map(|r| r.index() + 1) != Some(out.insts.len()) {
                return Err(invariant("a replacement's length differs from its measure"));
            }
        }
        let end = InstId::new(u32::try_from(out.insts.len()).map_err(|_| invariant("ids"))?);
        out.blocks.push(
            Block {
                start,
                end,
                name: block.name,
            },
            m,
        )?;
    }
    m.release(result.len().saturating_mul(std::mem::size_of::<InstId>()));
    Ok(out)
}

/// Emit piece `p` of a copy into `out`, `(src, dst)` its pointer operands in the output and
/// `void` its type. Its accesses assume no alignment: the copy's operands promise none.
fn emit(
    out: &mut Body,
    void: TypeId,
    p: &Piece,
    (src, dst): (Value, Value),
    v: &Vocab<'_>,
    m: &mut Meter,
) -> Result<(), Error> {
    let byte = Align::from_bytes(1);
    let at = p.at;
    let address = |out: &mut Body, base: Value, m: &mut Meter| -> Result<Value, Error> {
        if at == 0 {
            return Ok(base);
        }
        let off = v
            .offsets
            .binary_search_by_key(&at, |(a, _)| *a)
            .ok()
            .and_then(|k| v.offsets.get(k))
            .map(|(_, c)| *c)
            .ok_or_else(|| invariant("a piece's distance"))?;
        out.push_inst(
            Inst {
                ty: v.thread,
                kind: InstKind::Gep {
                    src_ty: v.i8t,
                    base,
                    indices: Box::new([Value::Const(off)]),
                    inbounds: true,
                },
            },
            m,
        )
        .map(Value::Inst)
    };
    let from = address(out, src, m)?;
    let value = out.push_inst(
        Inst {
            ty: p.ty,
            kind: InstKind::Load {
                ptr: from,
                align: byte,
                volatile: false,
                atomic: None,
                range: None,
            },
        },
        m,
    )?;
    let to = address(out, dst, m)?;
    out.push_inst(
        Inst {
            ty: void,
            kind: InstKind::Store {
                val: Value::Inst(value),
                ptr: to,
                align: byte,
                volatile: false,
                atomic: None,
            },
        },
        m,
    )?;
    Ok(())
}

/// Replace each of `replaced` (sorted by instruction) in the body of entry `func` by its pieces
/// ([`rebuild`]), interning the constants their addresses step by.
pub(super) fn replace(
    parts: &mut ModuleParts,
    func: FuncId,
    replaced: &[(InstId, Replacement)],
    m: &mut Meter,
) -> Result<(), Error> {
    if replaced.is_empty() {
        return Ok(());
    }
    let i8t = parts.types.intern(Type::Int(8), m)?;
    let i64t = parts.types.intern(Type::Int(64), m)?;
    let thread = parts.types.intern(
        Type::Ptr {
            space: AddrSpace::Thread,
        },
        m,
    )?;
    let mut ats: Vec<u64> = Vec::new();
    for (_, r) in replaced {
        for p in &r.pieces {
            m.tick(1)?;
            if p.at != 0 {
                m.reserve(&mut ats, 1)?;
                ats.push(p.at);
            }
        }
    }
    ats.sort_unstable();
    ats.dedup();
    let mut offsets: Vec<(u64, ConstId)> = Vec::new();
    m.reserve(&mut offsets, ats.len())?;
    for &at in &ats {
        let c = parts.consts.intern(
            Constant {
                ty: i64t,
                value: Const::Int(at),
            },
            m,
        )?;
        offsets.push((at, c));
    }
    let body = parts
        .funcs
        .get(func)
        .and_then(|f| f.body.as_ref())
        .ok_or_else(|| invariant("the entry has no body"))?;
    let v = Vocab {
        i8t,
        thread,
        offsets: &offsets,
    };
    let rebuilt = rebuild(body, replaced, &v, m);
    m.release(
        ats.capacity() * std::mem::size_of::<u64>()
            + offsets.capacity() * std::mem::size_of::<(u64, ConstId)>(),
    );
    let rebuilt = rebuilt?;
    let f = parts
        .funcs
        .get_mut(func)
        .ok_or_else(|| invariant("entry out of range"))?;
    if let Some(old) = f.body.replace(rebuilt) {
        old.release(m);
    }
    Ok(())
}

/// The base `v` addresses and its byte offset from it: through `getelementptr`s at constant,
/// non-negative indices and bitcasts, at most [`STEPS`] of them, to the first value that is
/// neither. `None` when one of those steps is not constant.
pub(super) fn offset_from(
    body: &Body,
    parts: &ModuleParts,
    layouts: &TypeLayouts,
    v: Value,
) -> Option<(Value, u64)> {
    let mut at = 0u64;
    let mut v = v;
    for _ in 0..STEPS {
        let Value::Inst(i) = v else {
            return Some((v, at));
        };
        match &body.insts.get(i)?.kind {
            InstKind::Gep {
                src_ty,
                base,
                indices,
                ..
            } => {
                at = at.checked_add(gep_offset(parts, layouts, *src_ty, indices)?)?;
                v = *base;
            }
            InstKind::Cast {
                op: CastOp::Bitcast,
                v: inner,
            } => v = *inner,
            _ => return Some((v, at)),
        }
    }
    None
}

/// The address steps [`offset_from`] follows.
const STEPS: usize = 64;

/// The byte offset a `getelementptr` over `src_ty` at constant, non-negative `indices` adds.
pub(super) fn gep_offset(
    parts: &ModuleParts,
    layouts: &TypeLayouts,
    src_ty: TypeId,
    indices: &[Value],
) -> Option<u64> {
    let (first, rest) = indices.split_first()?;
    let mut at = int(parts, *first)?.checked_mul(layouts.alloc_size(src_ty)?)?;
    let mut ty = src_ty;
    for &x in rest {
        let x = int(parts, x)?;
        let k = usize::try_from(x).ok()?;
        let (step, inner) = match parts.types.get(ty)? {
            Type::Struct { fields, .. } => (*layouts.field_offsets(ty)?.get(k)?, *fields.get(k)?),
            Type::Array { elem, .. } | Type::Vector { elem, .. } => {
                (x.checked_mul(layouts.alloc_size(*elem)?)?, *elem)
            }
            _ => return None,
        };
        at = at.checked_add(step)?;
        ty = inner;
    }
    Some(at)
}

/// The value of integer constant `v` when it is one and not negative.
pub(super) fn int(parts: &ModuleParts, v: Value) -> Option<u64> {
    let Value::Const(c) = v else {
        return None;
    };
    let bits = parts.types.int_bits(parts.consts.ty(c)?)?;
    let x = match parts.consts.get(c)?.value {
        Const::Int(x) => x,
        Const::Null => 0,
        _ => return None,
    };
    // A negative value (its sign bit set) is no offset, length, or flag here; an `i1` is a flag.
    ((x >> (u32::from(bits.clamp(1, 64)) - 1)) & 1 == 0 || bits == 1).then_some(x)
}
