//! Aggregate loads and stores as the accesses of their members.
//!
//! An LLVM `load` or `store` may move a whole array or struct (`load [4 x <4 x float>], ptr
//! addrspace(2) %p`): Metal's own linker emits one where it passes a field of a buffer's struct to
//! a function by value. Lowering reads and writes buffer and threadgroup memory a scalar or vector
//! at a time. After SCCP, [`expand`] replaces each such access, in its place, by its members' (the
//! scalars and vectors the type holds, in layout order): a `getelementptr i8` at the member's
//! constant byte distance from the access's pointer (none at distance 0), and a load chained into
//! the aggregate by `insertvalue`, starting from `undef`; a store's members are each an
//! `extractvalue` and a store. A member's alignment is the access's, lowered to the largest power
//! of two dividing its distance; an access that states none leaves its members' unstated (each
//! member's ABI alignment, which its offset in the layout honours).
//!
//! What is split: a non-volatile, non-atomic access in `device`, `constant` or `threadgroup`
//! memory of an array or struct that holds no pointer, with at least one and at most
//! [`MAX_MEMBERS`] members, nested at most [`MAX_DEPTH`] deep. The rest stays whole: `thread`
//! memory's allocations the memory planner takes whole, a pointer member (a handle or a resource)
//! is lowering's to follow, and lowering holds no larger aggregate, so it refuses the access.

use crate::budget::Meter;
use crate::error::Error;
use crate::ir::Value;
use crate::ir::{AddrSpace, Align, Block, Body, Const, ConstId, Constant, FuncId, Idx, Inst};
use crate::ir::{InstId, InstKind, Layout, ModuleParts, Type, TypeId, TypeLayouts, TypeTable};

use super::rewrite::invariant;

/// The most members an access is split into.
pub(super) const MAX_MEMBERS: usize = 256;
/// The deepest member an access is split to, counted in `insertvalue` indices.
pub(super) const MAX_DEPTH: usize = 8;

/// One scalar or vector of an aggregate: its byte distance from the aggregate's start, its type,
/// and the indices that name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Member {
    at: u64,
    ty: TypeId,
    path: [u32; MAX_DEPTH],
    depth: u8,
}

impl Member {
    fn indices(&self) -> Box<[u32]> {
        self.path
            .get(..usize::from(self.depth))
            .unwrap_or_default()
            .into()
    }
}

/// An access [`expand`] splits: a load (`val` none) or a store of `val` through `ptr`, of type
/// `ty`, in `members` (a range of the pass's member list).
#[derive(Clone, Copy, Debug)]
struct Split {
    ptr: Value,
    val: Option<Value>,
    ty: TypeId,
    ptr_ty: TypeId,
    align: Option<Align>,
    members: (usize, usize),
}

impl Split {
    /// The instructions the split access becomes: per member a load and an `insertvalue` (a
    /// store: an `extractvalue` and a store), and an address at a distance other than 0.
    fn emitted(&self, members: &[Member]) -> u64 {
        members
            .get(self.members.0..self.members.1)
            .unwrap_or_default()
            .iter()
            .map(|p| 2 + u64::from(p.at != 0))
            .sum()
    }
}

/// Append `ty`'s members at distance `at` and indices `path` to `out`; `false` when `ty` is not
/// split (it holds a pointer or a type with no layout, or is too large or too deep).
fn walk(
    (types, layouts): (&TypeTable, &TypeLayouts),
    ty: TypeId,
    at: u64,
    path: ([u32; MAX_DEPTH], usize),
    out: &mut Vec<Member>,
    m: &mut Meter,
) -> Result<bool, Error> {
    m.tick(1)?;
    let (indices, depth) = path;
    let within = |k: u64, step: u64| -> Option<(u64, [u32; MAX_DEPTH])> {
        let mut p = indices;
        *p.get_mut(depth)? = u32::try_from(k).ok()?;
        Some((at.checked_add(step)?, p))
    };
    match types.get(ty) {
        Some(Type::Array { elem, len }) => {
            if depth == MAX_DEPTH || *len > MAX_MEMBERS as u64 {
                return Ok(false);
            }
            let Some(size) = layouts.alloc_size(*elem) else {
                return Ok(false);
            };
            for k in 0..*len {
                let Some((step, p)) = k.checked_mul(size).and_then(|s| within(k, s)) else {
                    return Ok(false);
                };
                if !walk((types, layouts), *elem, step, (p, depth + 1), out, m)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Some(Type::Struct { fields, .. }) => {
            let Some(offsets) = layouts.field_offsets(ty) else {
                return Ok(false);
            };
            if depth == MAX_DEPTH || fields.len() != offsets.len() {
                return Ok(false);
            }
            for (k, (f, off)) in fields.iter().zip(offsets.iter()).enumerate() {
                let Some((step, p)) = within(k as u64, *off) else {
                    return Ok(false);
                };
                if !walk((types, layouts), *f, step, (p, depth + 1), out, m)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Some(Type::Vector { elem, .. }) if !matches!(types.get(*elem), Some(Type::Ptr { .. })) => {
            push(out, ty, at, path, m)
        }
        Some(Type::Int(_) | Type::Half | Type::BFloat | Type::Float | Type::Double) => {
            push(out, ty, at, path, m)
        }
        _ => Ok(false),
    }
}

/// Push the member `ty` at `at`, `false` when the aggregate has [`MAX_MEMBERS`] already.
fn push(
    out: &mut Vec<Member>,
    ty: TypeId,
    at: u64,
    (path, depth): ([u32; MAX_DEPTH], usize),
    m: &mut Meter,
) -> Result<bool, Error> {
    if out.len() >= MAX_MEMBERS {
        return Ok(false);
    }
    m.reserve(out, 1)?;
    out.push(Member {
        at,
        ty,
        path,
        depth: u8::try_from(depth).map_err(|_| invariant("member depth"))?,
    });
    Ok(true)
}

/// The members of aggregate `ty`, appended to `out`: the range they take, or `None` when `ty` is
/// not split (`out` left as it was).
fn members_of(
    tl: (&TypeTable, &TypeLayouts),
    ty: TypeId,
    out: &mut Vec<Member>,
    m: &mut Meter,
) -> Result<Option<(usize, usize)>, Error> {
    if !matches!(tl.0.get(ty), Some(Type::Array { .. } | Type::Struct { .. })) {
        return Ok(None);
    }
    let start = out.len();
    if walk(tl, ty, 0, ([0; MAX_DEPTH], 0), out, m)? && out.len() > start {
        return Ok(Some((start, out.len())));
    }
    out.truncate(start);
    Ok(None)
}

/// The access `inst` is when [`expand`] splits it.
fn split_of(
    inst: &Inst,
    ty_of: &dyn Fn(Value) -> Option<TypeId>,
    tl: (&TypeTable, &TypeLayouts),
    members: &mut Vec<Member>,
    m: &mut Meter,
) -> Result<Option<Split>, Error> {
    let (ptr, val, ty, align) = match &inst.kind {
        InstKind::Load {
            ptr,
            align,
            volatile: false,
            atomic: None,
            ..
        } => (*ptr, None, inst.ty, *align),
        InstKind::Store {
            val,
            ptr,
            align,
            volatile: false,
            atomic: None,
        } => match ty_of(*val) {
            Some(ty) => (*ptr, Some(*val), ty, *align),
            None => return Ok(None),
        },
        _ => return Ok(None),
    };
    let Some(ptr_ty) = ty_of(ptr) else {
        return Ok(None);
    };
    let split = matches!(
        tl.0.get(ptr_ty),
        Some(Type::Ptr {
            space: AddrSpace::Device | AddrSpace::Constant | AddrSpace::Threadgroup,
        })
    );
    if !split {
        return Ok(None);
    }
    Ok(members_of(tl, ty, members, m)?.map(|members| Split {
        ptr,
        val,
        ty,
        ptr_ty,
        align,
        members,
    }))
}

/// The types and constants the split accesses are built from.
struct Vocab<'a> {
    i8t: TypeId,
    void: TypeId,
    /// Each distance's `i64` constant, sorted by distance.
    offsets: &'a [(u64, ConstId)],
    /// Each split load's type's `undef`, sorted by type.
    undefs: &'a [(TypeId, ConstId)],
}

/// Emit split access `s` into `out`, its operands `(ptr, val)` mapped to the output.
fn emit(
    out: &mut Body,
    s: &Split,
    members: &[Member],
    (ptr, val): (Value, Option<Value>),
    v: &Vocab<'_>,
    m: &mut Meter,
) -> Result<(), Error> {
    let members = members
        .get(s.members.0..s.members.1)
        .ok_or_else(|| invariant("a split's members"))?;
    let mut agg = match val {
        Some(_) => None,
        None => Some(Value::Const(
            v.undefs
                .binary_search_by_key(&s.ty, |(t, _)| *t)
                .ok()
                .and_then(|k| v.undefs.get(k))
                .map(|(_, c)| *c)
                .ok_or_else(|| invariant("a split load's undef"))?,
        )),
    };
    for p in members {
        m.tick(1)?;
        let align = s.align.and_then(|a| {
            let dividing = if p.at == 0 {
                a.bytes()
            } else {
                a.bytes().min(1 << p.at.trailing_zeros())
            };
            Align::from_bytes(dividing)
        });
        let part = match val {
            Some(val) => Some(out.push_inst(
                Inst {
                    ty: p.ty,
                    kind: InstKind::ExtractValue {
                        agg: val,
                        indices: p.indices(),
                    },
                },
                m,
            )?),
            None => None,
        };
        let address = if p.at == 0 {
            ptr
        } else {
            let off = v
                .offsets
                .binary_search_by_key(&p.at, |(a, _)| *a)
                .ok()
                .and_then(|k| v.offsets.get(k))
                .map(|(_, c)| *c)
                .ok_or_else(|| invariant("a member's distance"))?;
            Value::Inst(out.push_inst(
                Inst {
                    ty: s.ptr_ty,
                    kind: InstKind::Gep {
                        src_ty: v.i8t,
                        base: ptr,
                        indices: Box::new([Value::Const(off)]),
                        inbounds: true,
                    },
                },
                m,
            )?)
        };
        match (part, agg) {
            (Some(part), _) => {
                out.push_inst(
                    Inst {
                        ty: v.void,
                        kind: InstKind::Store {
                            val: Value::Inst(part),
                            ptr: address,
                            align,
                            volatile: false,
                            atomic: None,
                        },
                    },
                    m,
                )?;
            }
            (None, Some(whole)) => {
                let loaded = out.push_inst(
                    Inst {
                        ty: p.ty,
                        kind: InstKind::Load {
                            ptr: address,
                            align,
                            volatile: false,
                            atomic: None,
                            range: None,
                        },
                    },
                    m,
                )?;
                agg = Some(Value::Inst(out.push_inst(
                    Inst {
                        ty: s.ty,
                        kind: InstKind::InsertValue {
                            agg: whole,
                            val: Value::Inst(loaded),
                            indices: p.indices(),
                        },
                    },
                    m,
                )?));
            }
            (None, None) => return Err(invariant("a split load without its aggregate")),
        }
    }
    Ok(())
}

/// Split every aggregate access of `body`, the body of entry `func` (the module docs).
pub(super) fn expand(
    body: Body,
    parts: &mut ModuleParts,
    func: FuncId,
    m: &mut Meter,
) -> Result<Body, Error> {
    let params: Vec<TypeId> = match parts.funcs.get(func).and_then(|f| parts.types.get(f.ty)) {
        Some(Type::Function { params, .. }) => params.to_vec(),
        _ => return Err(invariant("the entry's type is not a function")),
    };
    let ty_of = |v: Value| match v {
        Value::Inst(i) => body.insts.get(i).map(|n| n.ty),
        Value::Arg(n) => params.get(n.index()).copied(),
        Value::Const(k) => parts.consts.ty(k),
    };
    let aggregate = |t: Option<TypeId>| {
        matches!(
            t.and_then(|t| parts.types.get(t)),
            Some(Type::Array { .. } | Type::Struct { .. })
        )
    };
    let any = body.insts.iter().any(|(_, i)| match &i.kind {
        InstKind::Load { .. } => aggregate(Some(i.ty)),
        InstKind::Store { val, .. } => aggregate(ty_of(*val)),
        _ => false,
    });
    if !any {
        return Ok(body);
    }
    let layouts = TypeLayouts::new(&parts.data_layout, &parts.types, m)?;
    let layout_bytes = parts.types.len() * std::mem::size_of::<Option<Layout>>();
    let mut members = Vec::new();
    let mut splits: Vec<(InstId, Split)> = Vec::new();
    for (i, inst) in body.insts.iter() {
        m.tick(1)?;
        if let Some(s) = split_of(inst, &ty_of, (&parts.types, &layouts), &mut members, m)? {
            m.reserve(&mut splits, 1)?;
            splits.push((i, s));
        }
    }
    let charged = members.capacity() * std::mem::size_of::<Member>()
        + splits.capacity() * std::mem::size_of::<(InstId, Split)>()
        + layout_bytes;
    let out = if splits.is_empty() {
        Ok(body)
    } else {
        rebuild(body, parts, &splits, &members, m)
    };
    m.release(charged);
    out
}

/// `body` with each access of `splits` (sorted by instruction) replaced by its members' accesses,
/// in its place, interning the constants they use.
fn rebuild(
    body: Body,
    parts: &mut ModuleParts,
    splits: &[(InstId, Split)],
    members: &[Member],
    m: &mut Meter,
) -> Result<Body, Error> {
    let i8t = parts.types.intern(Type::Int(8), m)?;
    let i64t = parts.types.intern(Type::Int(64), m)?;
    let void = parts.types.intern(Type::Void, m)?;
    let mut ats: Vec<u64> = Vec::new();
    let mut tys: Vec<TypeId> = Vec::new();
    for (_, s) in splits {
        if s.val.is_none() {
            m.reserve(&mut tys, 1)?;
            tys.push(s.ty);
        }
        for p in members.get(s.members.0..s.members.1).unwrap_or_default() {
            m.tick(1)?;
            if p.at != 0 {
                m.reserve(&mut ats, 1)?;
                ats.push(p.at);
            }
        }
    }
    ats.sort_unstable();
    ats.dedup();
    tys.sort_unstable();
    tys.dedup();
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
    let mut undefs: Vec<(TypeId, ConstId)> = Vec::new();
    m.reserve(&mut undefs, tys.len())?;
    for &ty in &tys {
        let c = parts.consts.intern(
            Constant {
                ty,
                value: Const::Undef,
            },
            m,
        )?;
        undefs.push((ty, c));
    }
    let v = Vocab {
        i8t,
        void,
        offsets: &offsets,
        undefs: &undefs,
    };
    let built = rebuild_with(&body, splits, members, &v, m);
    m.release(
        ats.capacity() * std::mem::size_of::<u64>()
            + tys.capacity() * std::mem::size_of::<TypeId>()
            + offsets.capacity() * std::mem::size_of::<(u64, ConstId)>()
            + undefs.capacity() * std::mem::size_of::<(TypeId, ConstId)>(),
    );
    let built = built?;
    // The body the splits replaced is freed.
    body.release(m);
    Ok(built)
}

/// [`rebuild`] over the interned vocabulary `v`.
fn rebuild_with(
    body: &Body,
    splits: &[(InstId, Split)],
    members: &[Member],
    v: &Vocab<'_>,
    m: &mut Meter,
) -> Result<Body, Error> {
    let split = |i: InstId| {
        splits
            .binary_search_by_key(&i, |(c, _)| *c)
            .ok()
            .and_then(|k| splits.get(k))
            .map(|(_, s)| s)
    };
    // Each old instruction's id in the output: a split access's is its expansion's last.
    let mut result = Vec::new();
    m.reserve(&mut result, body.insts.len())?;
    let mut next = 0u64;
    for (i, _) in body.insts.iter() {
        m.tick(1)?;
        let n = split(i).map_or(1, |s| s.emitted(members));
        if n == 0 {
            return Err(invariant("a split without members"));
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
            if let Some(s) = split(i) {
                let operands = (map(s.ptr)?, s.val.map(map).transpose()?);
                emit(&mut out, s, members, operands, v, m)?;
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
                return Err(invariant("a split's length differs from its measure"));
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
