//! One invocation's instructions: arithmetic, conversions, composites, memory, and atomics.
//! Subgroup operations and barriers involve several invocations and live in `sched`/`group`.

use crate::spirv::ids::{FuncId, TypeId, Value};
use crate::spirv::ir::{
    AtomicOp, BinOp, Contraction, ConvOp, Inst, InstNode, MemAccess, Type, UnOp,
};

use super::float;
use super::glsl;
use super::layout;
use super::memory::{Mem, Shared};
use super::sched::{Lane, Prog};
use super::value::{mask, sext, Kind, Num, Ptr, Root, Val};
use super::{malformed, IResult, TrapKind, Unmodelled};

/// What the scheduler does after an instruction.
pub(super) enum Flow {
    Next,
    /// Enter `func` with `args`; the call's result (if any) is the instruction's result.
    Call {
        func: FuncId,
        args: Vec<Val>,
    },
}

/// Execute one instruction for one lane.
pub(super) fn exec(p: &Prog<'_>, l: &mut Lane, sh: &mut Shared, node: &InstNode) -> IResult<Flow> {
    let t = &p.t;
    let get = |l: &Lane, v: Value| l.get(p, v);
    let res_ty = || -> IResult<TypeId> {
        let r = node
            .result
            .ok_or_else(|| malformed("instruction without a result"))?;
        t.value_ty(r)
    };
    let v = match &node.inst {
        // Loading an image variable yields the image (`super::image`).
        Inst::Load { ptr, .. }
            if matches!(
                get(l, *ptr)?,
                Val::P(Ptr {
                    root: Root::Image(_),
                    ..
                })
            ) =>
        {
            get(l, *ptr)?
        }
        Inst::Load { ptr, access } => {
            let q = get(l, *ptr)?.ptr()?;
            check_align(q, access)?;
            let ty = res_ty()?;
            let mut mem = Mem {
                lane: &mut l.arena,
                sh,
            };
            layout::load(t, &mut mem, q, ty)?
        }
        Inst::Store { ptr, value, access } => {
            let q = get(l, *ptr)?.ptr()?;
            check_align(q, access)?;
            let val = get(l, *value)?;
            let ty = t.value_ty(*value)?;
            let mut mem = Mem {
                lane: &mut l.arena,
                sh,
            };
            layout::store(t, &mut mem, q, ty, &val)?;
            return Ok(Flow::Next);
        }
        Inst::Unary { op, a } => unary(p, *op, *a, &*l.get_ref(p, *a)?)?,
        // Every operation is rounded on its own: exactly `NoContraction`, and one of the results
        // an operation that allows contraction may give (a device may also fuse it).
        Inst::Binary {
            op,
            a,
            b,
            contract: Contraction::Allowed | Contraction::Forbidden,
        } => binary(p, *op, *a, &*l.get_ref(p, *a)?, &*l.get_ref(p, *b)?)?,
        Inst::Select { cond, t: tv, f } => {
            let c = l.get_ref(p, *cond)?;
            let (x, y) = (l.get_ref(p, *tv)?, l.get_ref(p, *f)?);
            match &*c {
                Val::S(c) => {
                    if *c != 0 {
                        x.into_owned()
                    } else {
                        y.into_owned()
                    }
                }
                Val::C(cs) => {
                    let (xs, ys) = (x.parts()?, y.parts()?);
                    let mut out = Vec::with_capacity(cs.len());
                    for ((c, a), b) in cs.iter().zip(xs).zip(ys) {
                        out.push(if c.s()? != 0 { a.clone() } else { b.clone() });
                    }
                    Val::C(out.into())
                }
                Val::P(_) => return Err(malformed("pointer condition")),
            }
        }
        // Correctly rounded (`float::fma`, one rounding).
        Inst::Fma { a, b, c } => {
            let w = p.t.num_of(*a)?.width;
            let (va, vb, vc) = (l.get_ref(p, *a)?, l.get_ref(p, *b)?, l.get_ref(p, *c)?);
            let (xs, ys, zs) = (va.comps()?, vb.comps()?, vc.comps()?);
            if xs.len() != ys.len() || xs.len() != zs.len() {
                return Err(malformed("component count mismatch"));
            }
            let mut out = Vec::with_capacity(xs.len());
            for ((x, y), z) in xs.into_iter().zip(ys).zip(zs) {
                out.push(float::fma(w, x, y, z)?);
            }
            Val::from_comps(out)
        }
        Inst::BitFieldInsert {
            base,
            insert,
            offset,
            count,
        } => {
            let (o, c) = field(get(l, *offset)?.s()?, get(l, *count)?.s()?)?;
            let m = if c == 0 { 0 } else { mask(c as u8) << o };
            map2(&*l.get_ref(p, *base)?, &*l.get_ref(p, *insert)?, |b, i| {
                Ok((b & !m) | ((i << o) & m))
            })?
        }
        Inst::BitFieldExtract {
            signed,
            base,
            offset,
            count,
        } => {
            let (o, c) = field(get(l, *offset)?.s()?, get(l, *count)?.s()?)?;
            let signed = *signed;
            map1(&*l.get_ref(p, *base)?, |b| {
                if c == 0 {
                    return Ok(0);
                }
                let x = (b >> o) & mask(c as u8);
                Ok(if signed {
                    (sext(x, c as u8) as u64) & mask(32)
                } else {
                    x
                })
            })?
        }
        Inst::Convert { op, a, .. } => convert(p, *op, *a, &*l.get_ref(p, *a)?, res_ty()?)?,
        Inst::CompositeConstruct { parts } => {
            let ty = res_ty()?;
            if matches!(t.get(ty)?, Type::Vector { .. }) {
                let mut comps = Vec::with_capacity(parts.len());
                for part in parts.iter() {
                    match &*l.get_ref(p, *part)? {
                        Val::S(x) => comps.push(*x),
                        v => comps.extend(v.comps()?),
                    }
                }
                Val::from_comps(comps)
            } else {
                let mut vals = Vec::with_capacity(parts.len());
                for part in parts.iter() {
                    vals.push(get(l, *part)?);
                }
                Val::C(vals.into())
            }
        }
        Inst::CompositeExtract { composite, indices } => {
            let whole = l.get_ref(p, *composite)?;
            let mut cur: &Val = &whole;
            for i in indices.iter() {
                cur = cur
                    .parts()?
                    .get(*i as usize)
                    .ok_or_else(|| malformed("extract index"))?;
            }
            cur.clone()
        }
        Inst::CompositeInsert {
            object,
            composite,
            indices,
        } => insert(get(l, *composite)?, indices, get(l, *object)?)?,
        Inst::VectorExtractDynamic { vector, index } => {
            let vec = l.get_ref(p, *vector)?;
            let k = dyn_index(p, *index, &*l.get_ref(p, *index)?, vec.parts()?.len())?;
            vec.parts()?
                .get(k)
                .cloned()
                .ok_or_else(|| malformed("dynamic extract"))?
        }
        Inst::VectorInsertDynamic {
            vector,
            component,
            index,
        } => {
            let vec = get(l, *vector)?;
            let k = dyn_index(p, *index, &get(l, *index)?, vec.parts()?.len())?;
            insert(vec, &[k as u32], get(l, *component)?)?
        }
        Inst::VectorShuffle { a, b, components } => {
            let mut all = l.get_ref(p, *a)?.comps()?;
            all.extend(l.get_ref(p, *b)?.comps()?);
            let mut out = Vec::with_capacity(components.len());
            for c in components.iter() {
                out.push(
                    *all.get(*c as usize)
                        .ok_or_else(|| malformed("shuffle component"))?,
                );
            }
            Val::from_comps(out)
        }
        Inst::Undef => t.zero(res_ty()?)?,
        // An element of a descriptor array of images (`sched`): the image bound to it.
        Inst::AccessChain { base, indices, .. }
            if matches!(
                get(l, *base)?,
                Val::P(Ptr {
                    root: Root::Image(_),
                    ..
                })
            ) =>
        {
            let Val::P(Ptr {
                root: Root::Image(k),
                ..
            }) = get(l, *base)?
            else {
                return Err(malformed("an image access chain without an image"));
            };
            let (_, ty) = t.pointee(t.value_ty(*base)?)?;
            let len = match t.get(ty)? {
                Type::Array { len, .. } => t.array_len(*len)?,
                Type::RuntimeArray { .. } => p
                    .image_array_len(k)
                    .ok_or_else(|| malformed("an image runtime array without a bound length"))?,
                _ => return Err(malformed("an access chain into an image")),
            };
            let [idx] = indices.as_ref() else {
                return Err(malformed("an image array access chain of several indices"));
            };
            let i = get(l, *idx)?.s()?;
            if i >= len {
                return Err(TrapKind::ImageOutOfBounds { image: k }.err());
            }
            Val::P(Ptr {
                root: Root::Image(k + i as u32),
                off: 0,
                ms: None,
            })
        }
        // An element of a descriptor array of buffers (`sched`): the buffer bound to it, then
        // the rest of the chain inside it.
        Inst::AccessChain { base, indices, .. }
            if matches!(
                get(l, *base)?,
                Val::P(Ptr {
                    root: Root::BufferArray(_),
                    ..
                })
            ) =>
        {
            let Val::P(Ptr {
                root: Root::BufferArray(k),
                ..
            }) = get(l, *base)?
            else {
                return Err(malformed(
                    "a buffer array access chain without a buffer array",
                ));
            };
            let (_, ty) = t.pointee(t.value_ty(*base)?)?;
            let elem = match t.get(ty)? {
                Type::Array { elem, .. } | Type::RuntimeArray { elem, .. } => *elem,
                _ => return Err(malformed("an access chain into a buffer array")),
            };
            let len = p
                .buffer_array_len(k)
                .ok_or_else(|| malformed("a buffer array without a bound length"))?;
            let Some((first, rest)) = indices.split_first() else {
                return Err(malformed("a buffer array access chain without an index"));
            };
            let i = sext(get(l, *first)?.s()?, t.num_of(*first)?.width);
            let e = u64::try_from(i)
                .ok()
                .filter(|e| *e < len)
                .ok_or(TrapKind::IndexOutOfRange { index: i, len }.err())?;
            let mut ty = elem;
            let mut cur = Ptr {
                root: Root::Buffer(k + e as u32),
                off: 0,
                ms: None,
            };
            for idx in rest.iter() {
                let n = t.num_of(*idx)?;
                let raw = get(l, *idx)?.s()?;
                let (ct, co, cms) = layout::child(t, ty, cur.ms, sext(raw, n.width))?;
                cur = Ptr {
                    ms: cms,
                    ..layout::offset(cur, co)?
                };
                ty = ct;
            }
            Val::P(cur)
        }
        Inst::AccessChain { base, indices, .. } => {
            let q = get(l, *base)?.ptr()?;
            let (_, mut ty) = t.pointee(t.value_ty(*base)?)?;
            let mut cur = q;
            for idx in indices.iter() {
                let n = t.num_of(*idx)?;
                let raw = get(l, *idx)?.s()?;
                // Indices are signed; a negative one is out of range.
                let i = sext(raw, n.width);
                let (ct, co, cms) = layout::child(t, ty, cur.ms, i)?;
                cur = Ptr {
                    ms: cms,
                    ..layout::offset(cur, co)?
                };
                ty = ct;
            }
            Val::P(cur)
        }
        Inst::ArrayLength { ptr, member } => {
            let q = get(l, *ptr)?.ptr()?;
            let (_, block) = t.pointee(t.value_ty(*ptr)?)?;
            let (rta, moff, _) = layout::child(t, block, None, i64::from(*member))?;
            let stride = match t.get(rta)? {
                Type::RuntimeArray {
                    stride: Some(s), ..
                } => u64::from(*s),
                _ => return Err(malformed("array length of a non-runtime array")),
            };
            let mut mem = Mem {
                lane: &mut l.arena,
                sh,
            };
            let (size, rel) = mem.extent(q.root, q.off)?;
            let start = rel.saturating_add(moff);
            Val::S(size.saturating_sub(start) / stride.max(1))
        }
        Inst::Call { func, args } => {
            let mut vals = Vec::with_capacity(args.len());
            for a in args.iter() {
                vals.push(get(l, *a)?);
            }
            return Ok(Flow::Call {
                func: *func,
                args: vals,
            });
        }
        Inst::AtomicLoad { ptr, .. } => {
            let q = get(l, *ptr)?.ptr()?;
            let ty = res_ty()?;
            let mut mem = Mem {
                lane: &mut l.arena,
                sh,
            };
            layout::load(t, &mut mem, q, ty)?
        }
        Inst::AtomicStore { ptr, value, .. } => {
            let q = get(l, *ptr)?.ptr()?;
            let val = get(l, *value)?;
            let ty = t.value_ty(*value)?;
            let mut mem = Mem {
                lane: &mut l.arena,
                sh,
            };
            layout::store(t, &mut mem, q, ty, &val)?;
            return Ok(Flow::Next);
        }
        Inst::Atomic { op, ptr, value, .. } => {
            let q = get(l, *ptr)?.ptr()?;
            let x = get(l, *value)?.s()?;
            let ty = res_ty()?;
            let n = t.num(ty)?;
            let full = mask(n.width);
            let len = usize::from(n.width / 8);
            let op = *op;
            let mut mem = Mem {
                lane: &mut l.arena,
                sh,
            };
            // Definedness is tracked per bit: `and` defines the bits it clears, `or` the bits it
            // sets, and an exchange every bit, so the two atomics of a sub-word store define
            // exactly the stored byte. Any other operation on an undefined bit is an uninitialized
            // read. The returned old value of `and`/`or` may hold undefined bits (the zero fill).
            let old = mem.atomic(q.root, q.off, len, |old, def| {
                let new = atomic_op(op, n, old, x)?;
                let defined = match op {
                    AtomicOp::And => (def | !x) & full,
                    AtomicOp::Or => (def | x) & full,
                    AtomicOp::Exchange => full,
                    _ if def == full => full,
                    _ => return Err(TrapKind::UninitializedRead { offset: q.off }.err()),
                };
                Ok((new, defined))
            })?;
            Val::S(old)
        }
        Inst::AtomicCompareExchange {
            ptr,
            value,
            comparator,
            ..
        } => {
            let q = get(l, *ptr)?.ptr()?;
            let (x, c) = (get(l, *value)?.s()?, get(l, *comparator)?.s()?);
            let ty = res_ty()?;
            rmw(p, &mut l.arena, sh, q, ty, |old| {
                Ok(if old == c { x } else { old })
            })?
        }
        Inst::MemoryBarrier { .. } => return Ok(Flow::Next),
        // Rounded on its own whatever `contract` says, as for `Binary`.
        Inst::Glsl { op, args, .. } => {
            let mut vals = Vec::with_capacity(args.len());
            for a in args.iter() {
                vals.push(get(l, *a)?);
            }
            glsl::eval(p, *op, args, &vals)?
        }
        Inst::Group { .. } | Inst::ControlBarrier { .. } | Inst::Derivative { .. } => {
            return Err(malformed(
                "a collective instruction reached lane evaluation",
            ))
        }
        Inst::SampledImage { .. } => return unmodelled(crate::spv::Op::SampledImage),
        // The image half of a loaded combined image sampler is its bound image (`sched`).
        Inst::Image(x) => match get(l, *x)? {
            v @ Val::P(Ptr {
                root: Root::Image(_),
                ..
            }) => v,
            _ => return unmodelled(crate::spv::Op::Image),
        },
        Inst::ImageSample { form, .. } => return unmodelled(form.to_spirv()),
        Inst::ImageGather { sparse: false, .. } => return unmodelled(crate::spv::Op::ImageGather),
        Inst::ImageGather { sparse: true, .. } => {
            return unmodelled(crate::spv::Op::ImageSparseGather)
        }
        Inst::ImageDrefGather { sparse: false, .. } => {
            return unmodelled(crate::spv::Op::ImageDrefGather)
        }
        Inst::ImageDrefGather { sparse: true, .. } => {
            return unmodelled(crate::spv::Op::ImageSparseDrefGather)
        }
        Inst::ImageFetch { .. }
        | Inst::ImageRead { .. }
        | Inst::ImageSparseTexelsResident(_)
        | Inst::ImageWrite { .. }
        | Inst::ImageTexelPointer { .. }
        | Inst::ImageQuery { .. } => match super::image::exec(p, l, sh, &node.inst)? {
            Some(Some(v)) => v,
            Some(None) => return Ok(Flow::Next),
            None => return Err(malformed("an image instruction outside the image model")),
        },
        Inst::ImageQueryLod { .. } => return unmodelled(crate::spv::Op::ImageQueryLod),
        Inst::BeginInterlock => return unmodelled(crate::spv::Op::BeginInvocationInterlockEXT),
        Inst::EndInterlock => return unmodelled(crate::spv::Op::EndInvocationInterlockEXT),
        Inst::DemoteToHelperInvocation => {
            return unmodelled(crate::spv::Op::DemoteToHelperInvocation)
        }
        Inst::IsHelperInvocation => return unmodelled(crate::spv::Op::IsHelperInvocationEXT),
        Inst::SetMeshOutputs { .. } => return unmodelled(crate::spv::Op::SetMeshOutputsEXT),
        Inst::EmitVertex => return unmodelled(crate::spv::Op::EmitVertex),
        Inst::EndPrimitive => return unmodelled(crate::spv::Op::EndPrimitive),
        // The interpreter has no acceleration structures: it cannot trace a ray.
        Inst::RayQuery(op) => return unmodelled(op.to_spirv()),
    };
    let r = node
        .result
        .ok_or_else(|| malformed("value without a result id"))?;
    l.set(p, r, v)?;
    Ok(Flow::Next)
}

fn unmodelled(op: crate::spv::Op) -> IResult<Flow> {
    Err(Unmodelled::Opcode(op).err())
}

/// A physical access must honour its `Aligned` operand.
fn check_align(q: Ptr, access: &MemAccess) -> IResult<()> {
    if let (Root::Physical, Some(a)) = (q.root, access.aligned) {
        if a != 0 && !q.off.is_multiple_of(u64::from(a)) {
            return Err(TrapKind::Misaligned {
                address: q.off,
                align: a,
            }
            .err());
        }
    }
    Ok(())
}

/// A read-modify-write of the scalar at `q`; returns the old value.
fn rmw(
    p: &Prog<'_>,
    arena: &mut Vec<super::memory::Region>,
    sh: &mut Shared,
    q: Ptr,
    ty: TypeId,
    f: impl FnOnce(u64) -> IResult<u64>,
) -> IResult<Val> {
    let mut mem = Mem { lane: arena, sh };
    let old = layout::load(&p.t, &mut mem, q, ty)?;
    let new = f(old.s()?)?;
    layout::store(&p.t, &mut mem, q, ty, &Val::S(new))?;
    Ok(old)
}

fn atomic_op(op: AtomicOp, n: Num, old: u64, x: u64) -> IResult<u64> {
    let w = n.width;
    Ok(match op {
        AtomicOp::Exchange => x,
        AtomicOp::IAdd => old.wrapping_add(x) & mask(w),
        AtomicOp::ISub => old.wrapping_sub(x) & mask(w),
        AtomicOp::SMin => pick(sext(old, w) <= sext(x, w), old, x),
        AtomicOp::SMax => pick(sext(old, w) >= sext(x, w), old, x),
        AtomicOp::UMin => old.min(x),
        AtomicOp::UMax => old.max(x),
        AtomicOp::And => old & x,
        AtomicOp::Or => old | x,
        AtomicOp::Xor => old ^ x,
        AtomicOp::FAdd => float::add(w, old, x)?,
    })
}

fn pick(c: bool, a: u64, b: u64) -> u64 {
    if c {
        a
    } else {
        b
    }
}

/// A bit field's (offset, count), trapping when it leaves the 32-bit base.
fn field(offset: u64, count: u64) -> IResult<(u32, u32)> {
    match (u32::try_from(offset), u32::try_from(count)) {
        (Ok(o), Ok(c)) if o.checked_add(c).is_some_and(|e| e <= 32) => Ok((o, c)),
        _ => Err(TrapKind::BitFieldOutOfRange.err()),
    }
}

/// A dynamic vector index, trapping when out of range.
fn dyn_index(p: &Prog<'_>, idx: Value, v: &Val, len: usize) -> IResult<usize> {
    let n = p.t.num_of(idx)?;
    let raw = v.s()?;
    let i = match n.kind {
        Kind::Int { signed: true } => sext(raw, n.width),
        _ => i64::try_from(raw).unwrap_or(i64::MAX),
    };
    match usize::try_from(i) {
        Ok(k) if k < len => Ok(k),
        _ => Err(TrapKind::IndexOutOfRange {
            index: i,
            len: len as u64,
        }
        .err()),
    }
}

/// `composite` with the part at `path` replaced by `object`.
fn insert(composite: Val, path: &[u32], object: Val) -> IResult<Val> {
    let Some((first, rest)) = path.split_first() else {
        return Ok(object);
    };
    let mut parts: Vec<Val> = composite.parts()?.to_vec();
    let slot = parts
        .get_mut(*first as usize)
        .ok_or_else(|| malformed("insert index"))?;
    *slot = insert(slot.clone(), rest, object)?;
    Ok(Val::C(parts.into()))
}

/// Apply `f` to each component of a scalar or vector. A scalar takes no allocation.
pub(super) fn map1(a: &Val, mut f: impl FnMut(u64) -> IResult<u64>) -> IResult<Val> {
    match a {
        Val::S(x) => Ok(Val::S(f(*x)?)),
        _ => {
            let parts = a.parts()?;
            let mut out = Vec::with_capacity(parts.len());
            for x in parts {
                out.push(Val::S(f(x.s()?)?));
            }
            Ok(Val::from_lanes(out))
        }
    }
}

/// Apply `f` to each pair of components. Two scalars take no allocation, two vectors only their
/// result's.
pub(super) fn map2(a: &Val, b: &Val, mut f: impl FnMut(u64, u64) -> IResult<u64>) -> IResult<Val> {
    match (a, b) {
        (Val::S(x), Val::S(y)) => return Ok(Val::S(f(*x, *y)?)),
        (Val::C(xs), Val::C(ys)) => {
            if xs.len() != ys.len() {
                return Err(malformed("component count mismatch"));
            }
            let mut out = Vec::with_capacity(xs.len());
            for (x, y) in xs.iter().zip(ys.iter()) {
                out.push(Val::S(f(x.s()?, y.s()?)?));
            }
            return Ok(Val::from_lanes(out));
        }
        _ => {}
    }
    let (xs, ys) = (a.comps()?, b.comps()?);
    if xs.len() != ys.len() {
        return Err(malformed("component count mismatch"));
    }
    let mut out = Vec::with_capacity(xs.len());
    for (x, y) in xs.into_iter().zip(ys) {
        out.push(f(x, y)?);
    }
    Ok(Val::from_comps(out))
}

fn unary(p: &Prog<'_>, op: UnOp, a: Value, va: &Val) -> IResult<Val> {
    let n = p.t.num_of(a).ok();
    let w = n.map_or(32, |n| n.width);
    match op {
        UnOp::SNegate => map1(va, |x| Ok(x.wrapping_neg() & mask(w))),
        UnOp::FNegate => map1(va, |x| Ok(x ^ (1u64 << (w - 1)))),
        UnOp::Not => map1(va, |x| Ok(!x & mask(w))),
        UnOp::LogicalNot => map1(va, |x| Ok(x ^ 1)),
        UnOp::BitReverse => map1(va, |x| Ok(u64::from((x as u32).reverse_bits()))),
        UnOp::BitCount => map1(va, |x| Ok(u64::from((x as u32).count_ones()))),
        UnOp::IsNan => map1(va, |x| Ok(u64::from(float::to_f64(x, w)?.is_nan()))),
        UnOp::IsInf => map1(va, |x| Ok(u64::from(float::to_f64(x, w)?.is_infinite()))),
        UnOp::Any => Ok(Val::S(u64::from(va.comps()?.iter().any(|c| *c != 0)))),
        UnOp::All => Ok(Val::S(u64::from(va.comps()?.iter().all(|c| *c != 0)))),
        UnOp::CopyObject => Ok(va.clone()),
        UnOp::QuantizeToF16 => map1(va, |x| {
            let h = float::f64_to_f16(float::to_f64(x, 32)?);
            // A value too small for a normal half becomes a zero. SPIR-V allows either `+0` or
            // `-0` whatever the input's sign; the interpreter models the stricter same-sign
            // choice.
            let h = if h & 0x7c00 == 0 { h & 0x8000 } else { h };
            float::from_f64(float::f16_to_f64(h), 32)
        }),
    }
}

/// Integer comparison of two components of width `w`.
fn icmp(op: BinOp, w: u8, a: u64, b: u64) -> bool {
    let (sa, sb) = (sext(a, w), sext(b, w));
    match op {
        BinOp::IEqual => a == b,
        BinOp::INotEqual => a != b,
        BinOp::UGreaterThan => a > b,
        BinOp::SGreaterThan => sa > sb,
        BinOp::UGreaterThanEqual => a >= b,
        BinOp::SGreaterThanEqual => sa >= sb,
        BinOp::ULessThan => a < b,
        BinOp::SLessThan => sa < sb,
        BinOp::ULessThanEqual => a <= b,
        BinOp::SLessThanEqual => sa <= sb,
        _ => false,
    }
}

fn fcmp(op: BinOp, x: f64, y: f64) -> bool {
    let unord = x.is_nan() || y.is_nan();
    let ord = |r: bool| !unord && r;
    let un = |r: bool| unord || r;
    match op {
        BinOp::FOrdEqual => ord(x == y),
        BinOp::FUnordEqual => un(x == y),
        BinOp::FOrdNotEqual => ord(x != y),
        BinOp::FUnordNotEqual => un(x != y),
        BinOp::FOrdLessThan => ord(x < y),
        BinOp::FUnordLessThan => un(x < y),
        BinOp::FOrdGreaterThan => ord(x > y),
        BinOp::FUnordGreaterThan => un(x > y),
        BinOp::FOrdLessThanEqual => ord(x <= y),
        BinOp::FUnordLessThanEqual => un(x <= y),
        BinOp::FOrdGreaterThanEqual => ord(x >= y),
        BinOp::FUnordGreaterThanEqual => un(x >= y),
        _ => false,
    }
}

/// An integer arithmetic or bitwise operation on components of width `w`.
pub(super) fn int_op(op: BinOp, w: u8, a: u64, b: u64) -> IResult<u64> {
    let m = mask(w);
    let (sa, sb) = (sext(a, w), sext(b, w));
    let min = sext(1u64 << (w - 1), w);
    let signed_ok = || -> IResult<()> {
        if sb == 0 {
            Err(TrapKind::DivideByZero.err())
        } else if sa == min && sb == -1 {
            Err(TrapKind::SignedOverflow.err())
        } else {
            Ok(())
        }
    };
    Ok(match op {
        BinOp::IAdd => a.wrapping_add(b) & m,
        BinOp::ISub => a.wrapping_sub(b) & m,
        BinOp::IMul => a.wrapping_mul(b) & m,
        BinOp::UDiv | BinOp::UMod => {
            if b == 0 {
                return Err(TrapKind::DivideByZero.err());
            }
            if op == BinOp::UDiv {
                a / b
            } else {
                a % b
            }
        }
        BinOp::SDiv => {
            signed_ok()?;
            (sa / sb) as u64 & m
        }
        BinOp::SRem => {
            signed_ok()?;
            (sa % sb) as u64 & m
        }
        BinOp::SMod => {
            signed_ok()?;
            let r = sa % sb;
            let r = if r != 0 && (r < 0) != (sb < 0) {
                r + sb
            } else {
                r
            };
            r as u64 & m
        }
        BinOp::BitwiseOr => a | b,
        BinOp::BitwiseXor => a ^ b,
        BinOp::BitwiseAnd => a & b,
        _ => return Err(malformed(format!("{op:?} is not an integer operation"))),
    })
}

/// A float arithmetic operation on components of width `w`.
pub(super) fn float_op(op: BinOp, w: u8, a: u64, b: u64) -> IResult<u64> {
    match op {
        BinOp::FAdd => float::add(w, a, b),
        BinOp::FSub => float::sub(w, a, b),
        BinOp::FMul => float::mul(w, a, b),
        BinOp::FDiv => float::bin(w, a, b, |x, y| x / y),
        // `%` on floats is the exact C `fmod`: the result has the sign of the dividend.
        BinOp::FRem => float::bin(w, a, b, |x, y| x % y),
        // The result has the sign of the divisor.
        _ => Err(malformed(format!("{op:?} is not a float operation"))),
    }
}

/// Dot product with per-operation rounding, in component order.
fn dot(w: u8, a: &[u64], b: &[u64]) -> IResult<u64> {
    let mut acc: Option<u64> = None;
    for (x, y) in a.iter().zip(b) {
        let prod = float::mul(w, *x, *y)?;
        acc = Some(match acc {
            None => prod,
            Some(s) => float::add(w, s, prod)?,
        });
    }
    acc.ok_or_else(|| malformed("empty dot product"))
}

fn binary(p: &Prog<'_>, op: BinOp, a: Value, va: &Val, vb: &Val) -> IResult<Val> {
    use BinOp as B;
    let bool_of = |r: bool| u64::from(r);
    match op {
        B::IAdd
        | B::ISub
        | B::IMul
        | B::UDiv
        | B::SDiv
        | B::UMod
        | B::SRem
        | B::SMod
        | B::BitwiseOr
        | B::BitwiseXor
        | B::BitwiseAnd => {
            let w = p.t.num_of(a)?.width;
            map2(va, vb, |x, y| int_op(op, w, x, y))
        }
        B::ShiftRightLogical | B::ShiftRightArithmetic | B::ShiftLeftLogical => {
            let w = p.t.num_of(a)?.width;
            map2(va, vb, |x, s| {
                if s >= u64::from(w) {
                    return Err(TrapKind::ShiftOutOfRange { shift: s, width: w }.err());
                }
                Ok(match op {
                    B::ShiftRightLogical => x >> s,
                    B::ShiftRightArithmetic => (sext(x, w) >> s) as u64 & mask(w),
                    _ => (x << s) & mask(w),
                })
            })
        }
        B::FAdd | B::FSub | B::FMul | B::FDiv | B::FRem => {
            let w = p.t.num_of(a)?.width;
            map2(va, vb, |x, y| float_op(op, w, x, y))
        }
        B::LogicalEqual => map2(va, vb, |x, y| Ok(bool_of(x == y))),
        B::LogicalNotEqual => map2(va, vb, |x, y| Ok(bool_of(x != y))),
        B::LogicalOr => map2(va, vb, |x, y| Ok(x | y)),
        B::LogicalAnd => map2(va, vb, |x, y| Ok(x & y)),
        B::IEqual
        | B::INotEqual
        | B::UGreaterThan
        | B::SGreaterThan
        | B::UGreaterThanEqual
        | B::SGreaterThanEqual
        | B::ULessThan
        | B::SLessThan
        | B::ULessThanEqual
        | B::SLessThanEqual => {
            let w = p.t.num_of(a)?.width;
            map2(va, vb, |x, y| Ok(bool_of(icmp(op, w, x, y))))
        }
        B::FOrdEqual
        | B::FUnordEqual
        | B::FOrdNotEqual
        | B::FUnordNotEqual
        | B::FOrdLessThan
        | B::FUnordLessThan
        | B::FOrdGreaterThan
        | B::FUnordGreaterThan
        | B::FOrdLessThanEqual
        | B::FUnordLessThanEqual
        | B::FOrdGreaterThanEqual
        | B::FUnordGreaterThanEqual => {
            let w = p.t.num_of(a)?.width;
            map2(va, vb, |x, y| {
                Ok(bool_of(fcmp(
                    op,
                    float::to_f64(x, w)?,
                    float::to_f64(y, w)?,
                )))
            })
        }
        B::IAddCarry | B::ISubBorrow | B::UMulExtended | B::SMulExtended => {
            let w = p.t.num_of(a)?.width;
            let (xs, ys) = (va.comps()?, vb.comps()?);
            let (mut lo, mut hi) = (Vec::new(), Vec::new());
            for (x, y) in xs.into_iter().zip(ys) {
                let (l, h) = match op {
                    B::IAddCarry => {
                        let s = u128::from(x) + u128::from(y);
                        (s as u64 & mask(w), u64::from(s >> w != 0))
                    }
                    B::ISubBorrow => ((x.wrapping_sub(y)) & mask(w), u64::from(x < y)),
                    B::UMulExtended => {
                        let pr = u128::from(x) * u128::from(y);
                        (pr as u64 & mask(w), (pr >> w) as u64 & mask(w))
                    }
                    _ => {
                        let pr = i128::from(sext(x, w)) * i128::from(sext(y, w));
                        (pr as u64 & mask(w), (pr >> w) as u64 & mask(w))
                    }
                };
                lo.push(l);
                hi.push(h);
            }
            Ok(Val::C(
                vec![Val::from_comps(lo), Val::from_comps(hi)].into(),
            ))
        }
        B::VectorTimesScalar => {
            let w = p.t.num_of(a)?.width;
            let s = vb.s()?;
            map1(va, |x| float::mul(w, x, s))
        }
        B::Dot => {
            let w = p.t.num_of(a)?.width;
            Ok(Val::S(dot(w, &va.comps()?, &vb.comps()?)?))
        }
    }
}

fn convert(p: &Prog<'_>, op: ConvOp, a: Value, va: &Val, res: TypeId) -> IResult<Val> {
    let t = &p.t;
    match op {
        ConvOp::ConvertUToPtr => Ok(Val::P(Ptr {
            root: Root::Physical,
            off: va.s()?,
            ms: None,
        })),
        ConvOp::Bitcast => bitcast(p, a, va, res),
        _ => {
            let (from, to) = (t.num_of(a)?, t.num(res)?);
            map1(va, |x| numeric(op, from, to, x))
        }
    }
}

/// A numeric conversion of one component.
fn numeric(op: ConvOp, from: Num, to: Num, x: u64) -> IResult<u64> {
    let (fw, tw) = (from.width, to.width);
    match op {
        ConvOp::UConvert => Ok(x & mask(tw)),
        ConvOp::SConvert => Ok(sext(x, fw) as u64 & mask(tw)),
        // An 8-bit float widens exactly: every value of either encoding is one of every wider
        // float.
        ConvOp::FConvert => match from.kind {
            Kind::Float8 { e5m2 } => float::from_f64(float::fp8_to_f64(x as u8, e5m2), tw),
            _ => float::from_f64(float::to_f64(x, fw)?, tw),
        },
        ConvOp::ConvertSToF => int_to_float(i128::from(sext(x, fw)), tw),
        ConvOp::ConvertUToF => int_to_float(i128::from(x), tw),
        ConvOp::ConvertFToU | ConvOp::ConvertFToS => {
            let v = float::to_f64(x, fw)?.trunc();
            let signed = op == ConvOp::ConvertFToS;
            let (lo, hi) = if signed {
                (
                    -(2f64.powi(i32::from(tw) - 1)),
                    2f64.powi(i32::from(tw) - 1),
                )
            } else {
                (0.0, 2f64.powi(i32::from(tw)))
            };
            // NaN fails both comparisons.
            if !(v >= lo && v < hi) {
                return Err(TrapKind::ConversionOutOfRange.err());
            }
            Ok(if signed {
                (v as i64) as u64 & mask(tw)
            } else {
                v as u64
            })
        }
        _ => Err(malformed(format!("{op:?} is not numeric"))),
    }
}

/// An integer rounded once to a `w`-bit float.
fn int_to_float(v: i128, w: u8) -> IResult<u64> {
    match w {
        32 => Ok(u64::from((v as f32).to_bits())),
        64 => Ok((v as f64).to_bits()),
        // Every integer that does not overflow a half is exact in f64.
        _ => float::from_f64(v as f64, w),
    }
}

/// Components `comps` of `fw` bits each, concatenated low first and cut into components of `tw`
/// bits, low first. Every SPIR-V numeric width is a multiple of 8 (a `bool` has no bit pattern
/// to cast), so the stream goes byte by byte.
fn rebits(comps: &[u64], fw: u8, tw: u8) -> IResult<Vec<u64>> {
    if fw == 0 || tw == 0 || !fw.is_multiple_of(8) || !tw.is_multiple_of(8) {
        return Err(malformed(format!("a bitcast between widths {fw} and {tw}")));
    }
    let (fb, tb) = (usize::from(fw / 8), usize::from(tw / 8));
    let mut bytes = Vec::with_capacity(comps.len() * fb);
    for c in comps {
        bytes.extend_from_slice(&c.to_le_bytes()[..fb]);
    }
    if !bytes.len().is_multiple_of(tb) {
        return Err(malformed("bitcast width mismatch"));
    }
    Ok(bytes
        .chunks(tb)
        .map(|c| {
            let mut w = [0u8; 8];
            w[..tb].copy_from_slice(c);
            u64::from_le_bytes(w)
        })
        .collect())
}

/// Reinterpret the bits of `va` as `res`: components are concatenated low first.
fn bitcast(p: &Prog<'_>, a: Value, va: &Val, res: TypeId) -> IResult<Val> {
    let t = &p.t;
    let to_ptr = matches!(t.get(res)?, Type::Pointer { .. });
    let (tw, n) = if to_ptr {
        (64u8, 1usize)
    } else {
        let nn = t.num(res)?;
        (nn.width, usize::from(nn.n))
    };
    let fw = match va {
        Val::P(_) => 64u8,
        _ => t.num(t.value_ty(a)?)?.width,
    };
    // A scalar of the result's width keeps its bits.
    if let (Val::S(x), false, 1) = (va, to_ptr, n) {
        if fw == tw {
            return Ok(Val::S(*x));
        }
    }
    let comps = match va {
        Val::P(q) => vec![q.off],
        _ => va.comps()?,
    };
    let out = rebits(&comps, fw, tw)?;
    if out.len() != n {
        return Err(malformed("bitcast width mismatch"));
    }
    if to_ptr {
        let off = *out.first().ok_or_else(|| malformed("bitcast to pointer"))?;
        return Ok(Val::P(Ptr {
            root: Root::Physical,
            off,
            ms: None,
        }));
    }
    Ok(Val::from_comps(out))
}
