//! Aggregates that hold pointers: `insertvalue` and `extractvalue` on a struct or array with a
//! pointer member, such as the `struct { texture2d<T> t; sampler s; }` that MSL passes by value
//! (`insertvalue %struct.TextureFetcher poison, ptr addrspace(1) %tex, 0, 0`).
//!
//! Such an aggregate is **never a SPIR-V value**. A pointer's lowered representation belongs to
//! the pointer's family and depends on its provenance, not on its type (a texture handle index, a
//! buffer root and offset, ...), so no one SPIR-V type holds the aggregate. Instead the memory
//! analysis resolves every pointer an `extractvalue` reads through the `insertvalue` chain that
//! built its aggregate (`memory::provenance`), and the families use the resolved roots:
//!
//! - an `insertvalue` whose result holds a pointer, and an `extractvalue` whose result is such an
//!   aggregate, emit nothing and define no value ([`ArithStrategy::PointerAggregate`]);
//! - an `extractvalue` of a texture or sampler handle with one static candidate is that
//!   candidate's constant handle index ([`ArithStrategy::Handle`]); a texture or sampler
//!   operation on it binds the candidate directly, as it would the parameter;
//! - an `extractvalue` of a pointer member that a null (device or constant) or undefined pointer
//!   constant supplies, or that an insertion chain's undefined (or null) base supplies because no
//!   insertion does, is the null pointer in the result's own representation
//!   ([`ArithStrategy::Null`]), as such a member of a member-wise phi is (below): the null fat
//!   pointer, the null handle index, or an arena offset of 0 for an undefined one;
//! - any other `extractvalue` whose member one `insertvalue` of the chain supplies wholly (a heap
//!   handle loaded from an argument buffer, a handle with several candidates, a buffer, arena,
//!   or buffer-table pointer, a non-pointer member) is that inserted value itself
//!   ([`ArithStrategy::Forward`]), in the representation its own family gave it. The value is
//!   read where the `extractvalue` stands, so it is carried there like any operand
//!   ([`forwarded`], which `select` and `emit` consult for the block-crossing values). A pointer's
//!   value is its provenance's encoding (a root and offset, an arena or table offset, a handle
//!   index), which provenance gives the extraction from its one source (`memory::provenance`
//!   joins the sources of a pointer-typed `extractvalue`), so a forwarded pointer and its source
//!   have one [`Repr`] by construction; the forward checks it.
//!
//! **Member-wise phis**. A `phi` of such an aggregate (MPS's `_MPSKernelInOut` rebuilt in two
//! branches and merged) is the one exception: it is a SPIR-V composite of its members'
//! representations ([`phi_type`]), a pointer member's the one every value that can reach it has
//! (through insertion chains and other member-wise phis; a null or undefined pointer takes any),
//! every other member the arith family's. On each edge the copy into the phi assembles its
//! incoming aggregate member by member from the values its insertion chain inserted
//! ([`incoming`], [`Build`]), and an `extractvalue` whose member comes from the phi is a
//! composite extract of the phi's value ([`ArithStrategy::Member`]). A pointer member is to the
//! phi what an incoming pointer is to a pointer phi: provenance joins every value that can reach
//! it into each extraction of it (`memory::provenance` resolves extractions through phis), so the
//! extraction's representation is the member's and each value's encoding reads back under it.
//!
//! Every other use of such an aggregate (a `select`, `freeze`, store, or return of it) is refused
//! by its own family, which finds no lowered representation for it. Not implemented
//! (`Unsupported{Lowering(Arith), NotImplemented}`): an `extractvalue` that no one insertion or
//! member-wise phi supplies wholly (from a `select`, load, or parameter aggregate, or a member an
//! inner insertion overwrites in part), one of a handle of a family other than textures and
//! samplers, which that family lowers, and a member-wise phi whose pointer member two reaching
//! values give different representations, or whose incoming aggregate has a member no value
//! supplies in the member's representation.

use std::collections::{BTreeMap, HashSet};

use crate::air::ir::{
    AddrSpace, BlockId, Body, Const, ConstId, Inst, InstId, InstKind, Type, TypeId, Value, ValueId,
};
use crate::budget::Meter;
use crate::error::Error;
use crate::lower::arith::legal::{self, not_lowered, LoweredConst};
use crate::lower::arith::{ArithPlan, ArithStrategy};
use crate::lower::emit::EmitCx;
use crate::lower::raytracing;
use crate::lower::select::SelectCx;
use crate::memory::handles;
use crate::memory::leaf::k32;
use crate::memory::provenance::{HandleClass, Repr};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, Value as SValue};

/// Whether `inst` is an `insertvalue` or `extractvalue` on an aggregate that holds a pointer.
pub fn involved(cx: &SelectCx<'_>, inst: &Inst) -> bool {
    let agg = match &inst.kind {
        InstKind::InsertValue { agg, .. } | InstKind::ExtractValue { agg, .. } => *agg,
        _ => return false,
    };
    cx.value_ty(agg)
        .is_some_and(|t| cx.entry.module.types().holds_pointer(t))
}

/// The most `insertvalue`/`extractvalue` steps [`source`] follows for one `extractvalue`; a
/// longer chain is not implemented (a bound on selection work, never a semantic limit).
const MAX_STEPS: usize = 4096;

/// Select the strategy of `insertvalue` or `extractvalue` `inst` (at `at`) on an aggregate that
/// holds a pointer ([`involved`]). A forwarding `extractvalue` is recorded in the plan
/// ([`forwarded`]).
pub fn select(cx: &mut SelectCx<'_>, at: InstId, inst: &Inst) -> Result<ArithStrategy, Error> {
    let InstKind::ExtractValue { agg, indices } = &inst.kind else {
        return Ok(ArithStrategy::PointerAggregate);
    };
    let is_ptr = matches!(cx.entry.module.types().get(inst.ty), Some(Type::Ptr { .. }));
    if !is_ptr && cx.entry.module.types().holds_pointer(inst.ty) {
        return Ok(ArithStrategy::PointerAggregate);
    }
    if is_ptr {
        if let Some(k) = static_handle(cx, at)? {
            return Ok(ArithStrategy::Handle(k));
        }
    }
    let Some(src) = source(cx.body, *agg, indices) else {
        return member_extract(cx, at, inst, *agg, indices);
    };
    // A null or undefined pointer constant is the null pointer in the result's representation,
    // as it is a member-wise phi's member.
    if is_ptr && null_constant(cx, src, inst.ty) {
        return null_member(cx, at, inst.ty);
    }
    // The inserted value must be one emission produces, in the representation the result has.
    let here = cx.lowered_repr(Value::Inst(at), inst.ty)?;
    let there = match src {
        Value::Const(_) if cx.constant(src).is_none() => return Err(not_lowered()),
        _ => cx.lowered_repr(src, inst.ty)?,
    };
    if here != there {
        return Err(not_lowered());
    }
    // A pointer's value is its provenance's encoding (a root and offset, an arena or table
    // offset, a handle index): the inserted pointer's value is the result's only when both have
    // one representation, which provenance gives an extraction whose one source is that pointer.
    if is_ptr && cx.memory.repr(Value::Inst(at)) != cx.memory.repr(src) {
        return Err(not_lowered());
    }
    let id = cx
        .values
        .id(src)
        .ok_or_else(|| cx.invariant("a forwarded value out of range"))?;
    cx.families.arith.forwards.insert(at, src);
    Ok(ArithStrategy::Forward(id))
}

/// The value a forwarding `extractvalue` `at` reads in place of its aggregate operand, if it is
/// one ([`ArithStrategy::Forward`]): `select` and `emit` carry that value to the
/// `extractvalue`'s block as they carry an operand.
pub fn forwarded(plan: &ArithPlan, at: InstId) -> Option<Value> {
    plan.forwards.get(&at).copied()
}

/// The constant handle index of `extractvalue` `at` when it reads a texture or sampler handle
/// whose provenance is one static candidate; `None` for a heap handle or several candidates.
fn static_handle(cx: &SelectCx<'_>, at: InstId) -> Result<Option<u32>, Error> {
    let v = Value::Inst(at);
    let class = match cx.memory.repr(v) {
        Some(Repr::Handle { class, .. })
            if matches!(class, HandleClass::Texture | HandleClass::Sampler) =>
        {
            class
        }
        // A handle of another family is that family's to lower, and a pointer has a
        // representation whenever provenance reached it.
        Some(Repr::Handle { .. }) | None => return Err(not_lowered()),
        // Every other pointer (a buffer, arena, or table pointer) is forwarded.
        Some(_) => return Ok(None),
    };
    Ok(
        match crate::lower::access::handle::domain_kind(cx, v, class)? {
            handles::DomainKind::Static(one) => one.map(|c| c.handle()),
            handles::DomainKind::Heap => None,
        },
    )
}

/// The value that supplies the member at `path` of aggregate `agg` wholly: the latest
/// `insertvalue` of the chain whose indices are `path` or a prefix of it (the rest of the path
/// then inside its inserted value), seen through `extractvalue`s of aggregates. `None` when no
/// insertion supplies it (the chain's base does), when a later insertion overwrites only part of
/// it, or past `MAX_STEPS`.
pub fn source(body: &Body, agg: Value, path: &[u32]) -> Option<Value> {
    let mut agg = agg;
    let mut path: Vec<u32> = path.to_vec();
    for _ in 0..MAX_STEPS {
        let Value::Inst(id) = agg else { return None };
        match &body.insts.get(id)?.kind {
            InstKind::InsertValue {
                agg: base,
                val,
                indices,
            } => {
                if path.starts_with(indices) {
                    if indices.len() == path.len() {
                        return Some(*val);
                    }
                    path.drain(..indices.len());
                    agg = *val;
                } else if indices.starts_with(&path) {
                    return None;
                } else {
                    agg = *base;
                }
            }
            InstKind::ExtractValue {
                agg: inner,
                indices,
            } => {
                let mut full = indices.to_vec();
                full.extend_from_slice(&path);
                path = full;
                agg = *inner;
            }
            _ => return None,
        }
    }
    None
}

// ---- Member-wise phis ----

/// Where the member at a path of an aggregate comes from ([`member`]).
#[derive(Clone, Debug, PartialEq, Eq)]
enum Member {
    /// A value with a lowered value of its own holds the member wholly.
    Value(Value),
    /// A constant holds the member wholly.
    Const(ConstId),
    /// Member `path` of the member-wise phi `phi`.
    Phi { phi: InstId, path: Vec<u32> },
    /// A member of an undefined (`undef`, `poison`) aggregate or insertion chain base.
    Undef,
    /// A member of a null (all-zero) aggregate constant.
    Zero,
    /// An aggregate member that holds a pointer and that insertions (or an aggregate constant)
    /// supply member by member.
    Parts,
}

/// Whether AIR type `t` is an aggregate that holds a pointer (and so has no arith value).
fn pointer_aggregate(cx: &SelectCx<'_>, t: TypeId) -> bool {
    let types = cx.entry.module.types();
    !matches!(types.get(t), Some(Type::Ptr { .. })) && types.holds_pointer(t)
}

/// Whether instruction `id` is a member-wise phi: a `phi` of an aggregate that holds a pointer.
fn is_member_phi(cx: &SelectCx<'_>, id: InstId) -> bool {
    cx.body
        .insts
        .get(id)
        .is_some_and(|i| matches!(i.kind, InstKind::Phi { .. }) && pointer_aggregate(cx, i.ty))
}

/// Where the member at `path` of aggregate `agg` comes from: through `insertvalue` and
/// `extractvalue` chains to the value, constant, or member-wise phi that holds it wholly, or
/// `Parts` when insertions below it supply it member by member. `None` for an aggregate with no
/// model (a load, `select`, or parameter of one), or past `MAX_STEPS`.
fn member(cx: &SelectCx<'_>, agg: Value, path: &[u32]) -> Option<Member> {
    let mut agg = agg;
    let mut path: Vec<u32> = path.to_vec();
    for _ in 0..MAX_STEPS {
        let whole = path.is_empty() && !pointer_aggregate(cx, cx.value_ty(agg)?);
        match agg {
            Value::Const(c) if whole => return Some(Member::Const(c)),
            Value::Const(c) => match &cx.entry.module.consts().get(c)?.value {
                Const::Undef | Const::Poison => return Some(Member::Undef),
                Const::Null => return Some(Member::Zero),
                Const::Aggregate(elems) => {
                    let Some((&first, rest)) = path.split_first() else {
                        return Some(Member::Parts);
                    };
                    agg = Value::Const(*elems.get(first as usize)?);
                    path = rest.to_vec();
                }
                _ => return None,
            },
            _ if whole => return Some(Member::Value(agg)),
            Value::Arg(_) => return None,
            Value::Inst(id) => match &cx.body.insts.get(id)?.kind {
                InstKind::InsertValue {
                    agg: base,
                    val,
                    indices,
                } => {
                    if !path.is_empty() && path.starts_with(indices) {
                        path.drain(..indices.len());
                        agg = *val;
                    } else if indices.starts_with(&path) {
                        return Some(Member::Parts);
                    } else {
                        agg = *base;
                    }
                }
                InstKind::ExtractValue {
                    agg: inner,
                    indices,
                } => {
                    let mut full = indices.to_vec();
                    full.extend_from_slice(&path);
                    path = full;
                    agg = *inner;
                }
                InstKind::Phi { .. } if is_member_phi(cx, id) => {
                    return Some(Member::Phi { phi: id, path })
                }
                _ => return None,
            },
        }
    }
    None
}

/// The member type of AIR aggregate type `t` at index `i`.
fn member_ty(cx: &SelectCx<'_>, t: TypeId, i: u32) -> Option<TypeId> {
    match cx.entry.module.types().get(t)? {
        Type::Struct { fields, .. } => fields.get(i as usize).copied(),
        Type::Array { elem, len } if u64::from(i) < *len => Some(*elem),
        _ => None,
    }
}

/// The number of members of AIR aggregate type `t` (a struct's fields, an array's elements).
fn member_count(cx: &SelectCx<'_>, t: TypeId) -> Option<u32> {
    match cx.entry.module.types().get(t)? {
        Type::Struct { fields, .. } => u32::try_from(fields.len()).ok(),
        Type::Array { len, .. } => u32::try_from(*len).ok(),
        _ => None,
    }
}

/// The member of composite representation `d` at `path`.
fn at_path<'d>(d: &'d DemoteTy, path: &[u32]) -> Option<&'d DemoteTy> {
    let mut d = d;
    for &i in path {
        let DemoteTy::Composite(parts) = d else {
            return None;
        };
        d = parts.get(i as usize)?;
    }
    Some(d)
}

/// The composite representation of member-wise phi `phi`: per member of its AIR type, a pointer's
/// representation (the one every value reaching it has), a member that holds no pointer the arith
/// family's, and an aggregate that holds one a composite of its members. At most
/// [`legal::MAX_SCALARS`] leaves and [`legal::MAX_DEPTH`] levels, as an arith aggregate.
pub fn phi_type(cx: &SelectCx<'_>, phi: InstId, m: &mut Meter) -> Result<DemoteTy, Error> {
    let ty = cx
        .body
        .insts
        .get(phi)
        .map(|i| i.ty)
        .ok_or_else(|| cx.invariant("a member-wise phi out of range"))?;
    let mut leaves = 0u64;
    member_type(cx, phi, &mut Vec::new(), ty, &mut leaves, m)
}

fn member_type(
    cx: &SelectCx<'_>,
    phi: InstId,
    path: &mut Vec<u32>,
    ty: TypeId,
    leaves: &mut u64,
    m: &mut Meter,
) -> Result<DemoteTy, Error> {
    m.tick(1)?;
    if path.len() > legal::MAX_DEPTH as usize {
        return Err(not_lowered());
    }
    if matches!(cx.entry.module.types().get(ty), Some(Type::Ptr { .. })) {
        *leaves = leaves.saturating_add(1);
        return pointer_member(cx, phi, path, ty, m);
    }
    if !cx.entry.module.types().holds_pointer(ty) {
        let d = super::lowered_type(cx.entry, cx.profile, ty)?;
        // A member `bool` vector is held as `u32` lanes (`legal::as_member`).
        let d = if path.is_empty() {
            d
        } else {
            legal::as_member(d)
        };
        *leaves = leaves.saturating_add(u64::from(super::leaves(&d)));
        return Ok(d);
    }
    let n = member_count(cx, ty).ok_or_else(not_lowered)?;
    if u64::from(n).saturating_add(*leaves) > legal::MAX_SCALARS {
        return Err(not_lowered());
    }
    let mut parts = Vec::with_capacity(n as usize);
    for i in 0..n {
        let t = member_ty(cx, ty, i).ok_or_else(not_lowered)?;
        path.push(i);
        let d = member_type(cx, phi, path, t, leaves, m);
        path.pop();
        parts.push(d?);
    }
    if *leaves > legal::MAX_SCALARS {
        return Err(not_lowered());
    }
    Ok(DemoteTy::Composite(parts.into()))
}

/// The representation of pointer member `path` (of AIR type `ty`) of member-wise phi `phi`: the
/// one every value that can reach it has, through the incoming aggregates' insertion chains and
/// the member-wise phis they come from. A null or undefined pointer takes any representation
/// ([`Build::Zero`]); a member only they reach, and two representations, are not implemented.
/// An intersection's primitive-data pointer has no value: the member carries zeros for it, which
/// nothing reads (its extraction is the ray tracing family's, which reads none), and a member only
/// primitive-data and null pointers reach is a narrow fat pointer that is always zero.
fn pointer_member(
    cx: &SelectCx<'_>,
    phi: InstId,
    path: &[u32],
    ty: TypeId,
    m: &mut Meter,
) -> Result<DemoteTy, Error> {
    let mut seen: HashSet<(InstId, Vec<u32>)> = HashSet::new();
    let mut work: Vec<(InstId, Vec<u32>)> = vec![(phi, path.to_vec())];
    let mut found: Option<DemoteTy> = None;
    let mut valueless = false;
    let mut steps = 0usize;
    while let Some((p, path)) = work.pop() {
        m.tick(1)?;
        steps += 1;
        if steps > MAX_STEPS {
            return Err(not_lowered());
        }
        if !seen.insert((p, path.clone())) {
            continue;
        }
        let Some(InstKind::Phi { incoming, .. }) = cx.body.insts.get(p).map(|i| &i.kind) else {
            return Err(cx.invariant("a member-wise phi that is no phi"));
        };
        for (v, _) in incoming.iter() {
            match member(cx, *v, &path) {
                Some(Member::Value(x)) if raytracing::is_primitive_data(cx, x) => valueless = true,
                Some(Member::Value(x)) => {
                    let d = cx.lowered_repr(x, ty)?;
                    if found.as_ref().is_some_and(|f| *f != d) {
                        return Err(not_lowered());
                    }
                    found = Some(d);
                }
                Some(Member::Phi { phi: q, path: r }) => work.push((q, r)),
                Some(Member::Const(_) | Member::Undef | Member::Zero) => {}
                Some(Member::Parts) | None => return Err(not_lowered()),
            }
        }
    }
    found
        .or(valueless.then_some(DemoteTy::FatPtr { wide: false }))
        .ok_or_else(not_lowered)
}

/// Plan every member-wise phi of the body before its instructions are selected (so a use in any
/// block reads the phi's representation): its composite type, or nothing when that is not
/// implemented (the phi's own selection then refuses it, [`phi_type`]).
pub fn plan_phis(cx: &mut SelectCx<'_>, m: &mut Meter) -> Result<(), Error> {
    let body = cx.body;
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        if !matches!(inst.kind, InstKind::Phi { .. }) || !pointer_aggregate(cx, inst.ty) {
            continue;
        }
        match phi_type(cx, id, m) {
            Ok(d) => {
                cx.families.arith.phis.insert(id, d);
            }
            Err(Error::Unsupported(_)) => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// The composite representation of `v` when it is a planned member-wise phi.
pub fn phi_repr(plan: &ArithPlan, v: Value) -> Option<&DemoteTy> {
    let Value::Inst(id) = v else { return None };
    plan.phis.get(&id)
}

/// How a member-wise phi's value on one edge is assembled: the incoming aggregate's members, each
/// in the phi member's representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Build {
    /// A value that holds the member wholly, in the member's representation.
    Value(ValueId),
    /// A `bool` vector of `lanes` lanes that holds the member wholly, as the `u32` lanes the
    /// member holds it in (`legal::as_member`).
    Bools { value: ValueId, lanes: u8 },
    /// Member `path` of the member-wise phi `phi`'s value.
    Extract { phi: ValueId, path: Box<[u32]> },
    /// A constant member.
    Const(LoweredConst),
    /// A null or undefined member: the representation's zero (the null fat pointer, the null
    /// handle, arena offset 0 for an undefined arena pointer, the arith zero).
    Zero(DemoteTy),
    /// An aggregate member assembled from its members.
    Parts { ty: DemoteTy, parts: Box<[Build]> },
}

impl Build {
    /// Every value the assembly reads (on the edge's block end, as a copy reads its operand).
    pub fn reads(&self, out: &mut Vec<ValueId>) {
        match self {
            Build::Value(v) | Build::Bools { value: v, .. } | Build::Extract { phi: v, .. } => {
                out.push(*v)
            }
            Build::Const(_) | Build::Zero(_) => {}
            Build::Parts { parts, .. } => parts.iter().for_each(|p| p.reads(out)),
        }
    }
}

/// The assembly of every planned member-wise phi's incoming aggregate on each edge, once every
/// instruction is selected (`produced` says whether a value has a lowered value): a refusal
/// names the phi.
pub fn plan_incoming(
    cx: &SelectCx<'_>,
    produced: &dyn Fn(Value) -> bool,
    m: &mut Meter,
) -> Result<BTreeMap<(InstId, BlockId), Build>, (Error, InstId)> {
    let mut out = BTreeMap::new();
    for (&phi, d) in &cx.families.arith.phis {
        let at = |e: Error| (e, phi);
        let Some(inst) = cx.body.insts.get(phi) else {
            return Err(at(cx.invariant("a member-wise phi out of range")));
        };
        let InstKind::Phi { incoming, .. } = &inst.kind else {
            return Err(at(cx.invariant("a member-wise phi that is no phi")));
        };
        for (v, from) in incoming.iter() {
            m.tick(1).map_err(|e| at(e.into()))?;
            if out.contains_key(&(phi, *from)) {
                continue;
            }
            let b = build(cx, produced, *v, &mut Vec::new(), inst.ty, d, m).map_err(at)?;
            out.insert((phi, *from), b);
        }
    }
    Ok(out)
}

/// The assembly of member `path` (of AIR type `ty`, representation `want`) of aggregate `v`.
fn build(
    cx: &SelectCx<'_>,
    produced: &dyn Fn(Value) -> bool,
    v: Value,
    path: &mut Vec<u32>,
    ty: TypeId,
    want: &DemoteTy,
    m: &mut Meter,
) -> Result<Build, Error> {
    m.tick(1)?;
    let is_ptr = matches!(cx.entry.module.types().get(ty), Some(Type::Ptr { .. }));
    Ok(match member(cx, v, path).ok_or_else(not_lowered)? {
        // An intersection's primitive-data pointer has no value; nothing reads the member.
        Member::Value(x) if is_ptr && raytracing::is_primitive_data(cx, x) => zero(want)?,
        Member::Value(x) => {
            if !produced(x) {
                return Err(not_lowered());
            }
            let have = cx.lowered_repr(x, ty)?;
            let id = cx.values.id(x).ok_or_else(not_lowered)?;
            match legal::bool_lanes(cx.entry.module.types(), ty) {
                _ if have == *want => Build::Value(id),
                Some(lanes) if legal::as_member(have) == *want => Build::Bools { value: id, lanes },
                _ => return Err(not_lowered()),
            }
        }
        Member::Phi { phi, path: p } => {
            let whole = cx.families.arith.phis.get(&phi).ok_or_else(not_lowered)?;
            if at_path(whole, &p) != Some(want) {
                return Err(not_lowered());
            }
            let phi = cx
                .values
                .id(Value::Inst(phi))
                .ok_or_else(|| cx.invariant("a member-wise phi out of the value range"))?;
            Build::Extract {
                phi,
                path: p.into(),
            }
        }
        Member::Const(c) if is_ptr => match cx.entry.module.consts().get(c).map(|k| &k.value) {
            Some(Const::Undef | Const::Poison) => zero(want)?,
            Some(Const::Null) if null_pointer(cx, ty) => zero(want)?,
            _ => return Err(not_lowered()),
        },
        Member::Const(c) => Build::Const(member_constant(cx, c, want).ok_or_else(not_lowered)?),
        Member::Undef => zero(want)?,
        Member::Zero if !is_ptr || null_pointer(cx, ty) => zero(want)?,
        Member::Zero => return Err(not_lowered()),
        Member::Parts => {
            let DemoteTy::Composite(wants) = want else {
                return Err(not_lowered());
            };
            if member_count(cx, ty) != u32::try_from(wants.len()).ok() {
                return Err(not_lowered());
            }
            let mut parts = Vec::with_capacity(wants.len());
            for (i, w) in wants.iter().enumerate() {
                let i = u32::try_from(i).map_err(|_| not_lowered())?;
                let t = member_ty(cx, ty, i).ok_or_else(not_lowered)?;
                path.push(i);
                let b = build(cx, produced, v, path, t, w, m);
                path.pop();
                parts.push(b?);
            }
            Build::Parts {
                ty: want.clone(),
                parts: parts.into(),
            }
        }
    })
}

/// Constant `c` as a member of representation `want`: as it is, widened, or a `bool` vector as
/// the member's `u32` lanes (`legal::as_member`).
fn member_constant(cx: &SelectCx<'_>, c: ConstId, want: &DemoteTy) -> Option<LoweredConst> {
    super::constant_as(cx.entry, Value::Const(c), want).or_else(|| {
        let k = legal::const_as_member(super::constant(cx.entry, Value::Const(c))?);
        (k.demote() == *want).then_some(k)
    })
}

/// The null pointer as the extraction `at` of AIR pointer type `ty`: in the extraction's own
/// representation, which must be a pointer's.
fn null_member(cx: &SelectCx<'_>, at: InstId, ty: TypeId) -> Result<ArithStrategy, Error> {
    match cx.lowered_repr(Value::Inst(at), ty)? {
        d @ (DemoteTy::FatPtr { .. }
        | DemoteTy::Address
        | DemoteTy::HandleIndex
        | DemoteTy::ArenaOffset) => Ok(ArithStrategy::Null(d)),
        _ => Err(not_lowered()),
    }
}

/// Whether `v` is a pointer constant of AIR pointer type `ty` that is the null pointer in every
/// representation: `undef` or `poison` (any value refines it), or `null` where it has a
/// representation ([`null_pointer`]).
fn null_constant(cx: &SelectCx<'_>, v: Value, ty: TypeId) -> bool {
    let Value::Const(c) = v else {
        return false;
    };
    match cx.entry.module.consts().get(c).map(|k| &k.value) {
        Some(Const::Undef | Const::Poison) => true,
        Some(Const::Null) => null_pointer(cx, ty),
        _ => false,
    }
}

/// Whether the null pointer of AIR pointer type `ty` has a representation: a device or constant
/// pointer's null root. A threadgroup or private pointer has none (its arena offsets have no null
/// value).
fn null_pointer(cx: &SelectCx<'_>, ty: TypeId) -> bool {
    matches!(
        cx.entry.module.types().get(ty),
        Some(Type::Ptr {
            space: AddrSpace::Device | AddrSpace::Constant
        })
    )
}

/// The zero of representation `d` as a [`Build`], where it is one: every representation but a
/// composite with no zero of its own.
fn zero(d: &DemoteTy) -> Result<Build, Error> {
    Ok(Build::Zero(d.clone()))
}

/// The values the assembly of member-wise phi `phi`'s value on the edge from `from` reads, when
/// `phi` is one ([`plan_incoming`]).
pub fn incoming(plan: &ArithPlan, phi: InstId, from: BlockId) -> Option<&Build> {
    plan.incoming.get(&(phi, from))
}

/// Select the extraction `at` of member `path` of aggregate `agg`, when no insertion supplies
/// the member wholly. A member that comes from a member-wise phi is a composite extract of the
/// phi's value, in the member's representation, which must be the extraction's own (for a
/// pointer, provenance's representation of it). A pointer member of an undefined aggregate (an
/// insertion chain's `undef` or `poison` base), or a device or constant pointer member of a
/// null one, is the null pointer, as such a member of a member-wise phi is.
fn member_extract(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    agg: Value,
    path: &[u32],
) -> Result<ArithStrategy, Error> {
    let is_ptr = matches!(cx.entry.module.types().get(inst.ty), Some(Type::Ptr { .. }));
    let (phi, path) = match member(cx, agg, path) {
        Some(Member::Phi { phi, path }) => (phi, path),
        Some(Member::Undef) if is_ptr => return null_member(cx, at, inst.ty),
        Some(Member::Zero) if is_ptr && null_pointer(cx, inst.ty) => {
            return null_member(cx, at, inst.ty)
        }
        _ => return Err(not_lowered()),
    };
    let whole = cx.families.arith.phis.get(&phi).ok_or_else(not_lowered)?;
    let there = at_path(whole, &path).cloned().ok_or_else(not_lowered)?;
    let have = cx.lowered_repr(Value::Inst(at), inst.ty)?;
    // A member `bool` vector is held as `u32` lanes (`legal::as_member`).
    let bools = match legal::bool_lanes(cx.entry.module.types(), inst.ty) {
        _ if have == there => None,
        Some(lanes) if legal::as_member(have) == there => Some(lanes),
        _ => return Err(not_lowered()),
    };
    let of = cx
        .values
        .id(Value::Inst(phi))
        .ok_or_else(|| cx.invariant("a member-wise phi out of the value range"))?;
    cx.families.arith.forwards.insert(at, Value::Inst(phi));
    Ok(ArithStrategy::Member {
        of,
        path: path.into(),
        bools,
    })
}

/// Emit the value of member-wise phi `phi` on the edge from `from` in the open block (the edge's
/// block end), or `None` when `phi` is no member-wise phi.
pub fn emit_incoming(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    phi: InstId,
    from: BlockId,
) -> BResult<Option<SValue>> {
    let Some(b) = incoming(&cx.plan.families.arith, phi, from) else {
        return Ok(None);
    };
    let b = b.clone();
    emit_build(cx, fb, &b).map(Some)
}

fn emit_build(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, b: &Build) -> BResult<SValue> {
    match b {
        Build::Value(v) => cx.operand(fb, *v),
        Build::Bools { value, lanes } => {
            let x = cx.operand(fb, *value)?;
            super::ops::bools_to_member(fb, x, *lanes)
        }
        Build::Extract { phi, path } => {
            let x = cx.operand(fb, *phi)?;
            if path.is_empty() {
                return Ok(x);
            }
            fb.composite_extract(x, path)
        }
        Build::Const(k) => {
            let k = if cx.plan.families.arith.wide {
                legal::widen_const(k.clone())
            } else {
                k.clone()
            };
            match super::materialize(fb, &k)? {
                Some(x) => Ok(x),
                None => Err(cx.fail("a member constant arith does not materialize")),
            }
        }
        Build::Zero(d) => zero_value(cx, fb, d),
        Build::Parts { ty, parts } => {
            let mut xs = Vec::with_capacity(parts.len());
            for p in parts.iter() {
                xs.push(emit_build(cx, fb, p)?);
            }
            let Some(t) = legal::spirv_type(fb.module(), ty)? else {
                return Err(cx.fail("a member-wise composite without a SPIR-V type"));
            };
            fb.composite_construct(t, &xs)
        }
    }
}

/// The null pointer in pointer representation `d` ([`ArithStrategy::Null`]): the null fat
/// pointer, the null handle index, or arena offset 0.
pub fn null_value(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    d: &DemoteTy,
) -> BResult<SValue> {
    match d {
        DemoteTy::FatPtr { .. }
        | DemoteTy::Address
        | DemoteTy::HandleIndex
        | DemoteTy::ArenaOffset => zero_value(cx, fb, d),
        _ => Err(cx.fail("a null pointer in a representation of no pointer")),
    }
}

/// The zero of representation `d`: the null fat pointer, handle index or arena offset 0, the
/// arith zero, or a composite of its members' zeros.
fn zero_value(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, d: &DemoteTy) -> BResult<SValue> {
    match d {
        DemoteTy::FatPtr { .. } | DemoteTy::Address => {
            match crate::lower::access::pointer::null_value(fb, d)? {
                Some(x) => Ok(x),
                None => Err(cx.fail("a null fat pointer without a value")),
            }
        }
        DemoteTy::HandleIndex => k32(fb, handles::NULL_HANDLE),
        DemoteTy::ArenaOffset => k32(fb, 0),
        DemoteTy::Composite(parts)
            if parts.iter().any(|p| {
                matches!(
                    p,
                    DemoteTy::FatPtr { .. }
                        | DemoteTy::Address
                        | DemoteTy::HandleIndex
                        | DemoteTy::ArenaOffset
                        | DemoteTy::Composite(_)
                )
            }) =>
        {
            let mut xs = Vec::with_capacity(parts.len());
            for p in parts.iter() {
                xs.push(zero_value(cx, fb, p)?);
            }
            let Some(t) = legal::spirv_type(fb.module(), d)? else {
                return Err(cx.fail("a member-wise composite without a SPIR-V type"));
            };
            fb.composite_construct(t, &xs)
        }
        _ => {
            let k = legal::zero(d).ok_or_else(|| cx.fail("a zero arith does not hold"))?;
            match super::materialize(fb, &k)? {
                Some(x) => Ok(x),
                None => Err(cx.fail("a zero arith does not materialize")),
            }
        }
    }
}
