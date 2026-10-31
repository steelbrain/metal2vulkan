//! Simdgroup asynchronous copies as plain AIR.
//!
//! MPS's matrix kernels copy tiles from device to threadgroup memory with
//! `air.simdgroup_async_copy_2d.p3i8.p1i8(size, align, dst, dst_epr, dst_stride, dst_tile, src,
//! src_epr, src_stride, src_tile, offset, clamp)`, which returns an event, beside
//! `air.get_null_simdgroup_event`, `air.is_null_simdgroup_event`, and
//! `air.wait_simdgroup_events(count, events)` ([`AsyncCopy`]). After inlining and before SCCP,
//! [`expand`] makes each such call of a kernel a call of a function built here, and inlines it:
//!
//! - **The copy** of `size`-byte elements (1, 2, or 4): the destination tile's element `(x, y)`,
//!   `x < dst_tile.x` and `y < dst_tile.y`, at byte `(y · dst_epr + x · dst_stride) · size` from
//!   `dst`, takes the source element `(sx, sy) = (x + offset.x, y + offset.y)` at byte
//!   `(sy · src_epr + sx · src_stride) · size` from `src` when `sx < src_tile.x` and
//!   `sy < src_tile.y` compared unsigned, and zero otherwise; nothing else is written. The lanes of
//!   the simdgroup that execute the call share the whole tile: of the `n` lanes (`simd_sum(1)`),
//!   the lane of rank `r` (`simd_prefix_exclusive_sum(1)`) copies columns `r, r + n, …` of every
//!   row, so the copy has completed when the call returns. Its event carries nothing but what
//!   `is_null` answers of it (below).
//! - **Events.** `is_null` of a copy's event answers `false` (an empty tile's too) and of the null
//!   event `true`, read from the event's own call. An event that went through memory, a PHI, or a
//!   select stays a call, which lowering refuses.
//! - **Waits** become `simdgroup_barrier(mem_flags::mem_threadgroup)`: every copy has completed,
//!   and the barrier makes each lane's part visible to the simdgroup. A wait then reads no event,
//!   so an event stored into thread memory that only waits read goes with the unobserved code.
//!
//! **Measured** (an M4 and the M3 Max, with and without Metal's validation layers; our own kernels
//! whose declared functions are renamed to the intrinsics): the threadgroup memory after the wait
//! of one-, two-, and four-byte copies over tiles from 0x4 to 64x16, element strides 1 to 3, a
//! destination pitch past its tile, offsets positive, negative, mixed, and past the source, source
//! tiles smaller than the destination's, empty, and of a negative extent (every column copied: the
//! test is unsigned), two simdgroups copying apart, simdgroups that copy nothing, and partial
//! simdgroups of 8 lanes and of 1 lane (each copies the whole tile); and every event's `is_null`.
//!
//! Not expanded (lowering refuses each call left, typed): a copy outside a kernel, from
//! threadgroup to device memory, of another element size, with a `clamp` operand other than 0
//! (clamping to the source's edge is not modeled), or whose size, alignment, or clamp is not a
//! literal constant.

use crate::abi::{EntryPoint, Stage};
use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::{AirIntrinsic, AsyncCopy};
use crate::ir::{
    AddrSpace, ArgIdx, BinOp, Body, Call, CallConv, Callee, CastOp, Const, FastMath, FnAttrs,
    FuncId, Function, InstId, InstKind, IntPredicate, Linkage, ModuleParts, Param, ParamAttrs,
    TailKind, Type, TypeId, UnnamedAddr, Value, Visibility,
};

use super::inline;
use super::out::Out;
use super::rewrite::invariant;

/// What one call becomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Lowered {
    /// A copy of `size`-byte elements whose accesses are aligned to `align` bytes.
    Copy { size: u8, align: u8 },
    /// The null event's value, the null pointer. A copy's event is the same value: only `is_null`
    /// reads an event, and it is answered from the event's call ([`Lowered::IsNull`]).
    Event,
    /// `is_null` of an event whose answer is `null`.
    IsNull { null: bool },
    /// A wait: a simdgroup barrier ordering threadgroup memory.
    Wait,
}

/// The copy's operands: `(size, align, dst, dst_epr, dst_stride, dst_tile, src, src_epr,
/// src_stride, src_tile, offset, clamp)`.
const COPY_ARGS: usize = 12;

/// The literal integer `v`, when it is one.
fn literal(parts: &ModuleParts, v: Value) -> Option<u64> {
    let Value::Const(c) = v else {
        return None;
    };
    match parts.consts.get(c)?.value {
        Const::Int(x) => Some(x),
        _ => None,
    }
}

/// The asynchronous-copy operation `kind` calls, with the call.
fn site(kind: &InstKind) -> Option<(AsyncCopy, &Call)> {
    let InstKind::Call(call) = kind else {
        return None;
    };
    match call.callee {
        Callee::Air {
            op: AirIntrinsic::AsyncCopy(a),
            ..
        } => Some((a, call)),
        _ => None,
    }
}

/// How the copy `call` is expanded, or `None` when it is not (module doc).
fn copy(parts: &ModuleParts, (dst, src): (AddrSpace, AddrSpace), call: &Call) -> Option<Lowered> {
    if (dst, src) != (AddrSpace::Threadgroup, AddrSpace::Device) || call.args.len() != COPY_ARGS {
        return None;
    }
    let size = literal(parts, *call.args.first()?)?;
    let align = literal(parts, *call.args.get(1)?)?;
    let clamp = literal(parts, *call.args.get(11)?)?;
    if !matches!(size, 1 | 2 | 4) || clamp != 0 {
        return None;
    }
    // Each element is at a multiple of its size from a pointer aligned to `align`.
    let align = if align.is_power_of_two() {
        align.min(size)
    } else {
        1
    };
    Some(Lowered::Copy {
        size: u8::try_from(size).ok()?,
        align: u8::try_from(align).ok()?,
    })
}

/// Whether `parts` declares an asynchronous-copy intrinsic. Every site calls one, so a module that
/// declares none has none, and [`expand`] need not read its instructions.
fn declares_sites(parts: &ModuleParts, m: &mut Meter) -> Result<bool, Error> {
    for f in parts.funcs.values() {
        m.tick(1)?;
        if f.body.is_some() {
            continue;
        }
        let air = parts
            .symbols
            .get(f.name)
            .and_then(|n| std::str::from_utf8(n).ok())
            .and_then(AirIntrinsic::parse);
        if matches!(air, Some(AirIntrinsic::AsyncCopy(_))) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Make each asynchronous-copy call of kernel `ep`'s inlined `body` that the module doc lists a
/// call of a function built for it, and inline the result.
pub(super) fn expand(
    body: Body,
    parts: &mut ModuleParts,
    ep: &EntryPoint,
    m: &mut Meter,
) -> Result<Body, Error> {
    if !matches!(ep.stage, Stage::Kernel { .. }) || !declares_sites(parts, m)? {
        return Ok(body);
    }
    // Each call's lowering, in instruction order.
    let mut plans: Vec<(InstId, Lowered)> = Vec::new();
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        let Some((op, call)) = site(&inst.kind) else {
            continue;
        };
        let lowered = match op {
            AsyncCopy::Copy2d { dst, src } => copy(parts, (dst, src), call),
            AsyncCopy::NullEvent => Some(Lowered::Event),
            AsyncCopy::IsNullEvent => {
                let event = call.args.first().and_then(|v| match v {
                    Value::Inst(e) => body.insts.get(*e).and_then(|e| site(&e.kind)),
                    _ => None,
                });
                match event {
                    Some((AsyncCopy::NullEvent, _)) => Some(Lowered::IsNull { null: true }),
                    Some((AsyncCopy::Copy2d { dst, src }, c)) => {
                        copy(parts, (dst, src), c).map(|_| Lowered::IsNull { null: false })
                    }
                    _ => None,
                }
            }
            AsyncCopy::WaitEvents => Some(Lowered::Wait),
        };
        if let Some(l) = lowered {
            m.reserve(&mut plans, 1)?;
            plans.push((id, l));
        }
    }
    if plans.is_empty() {
        m.release(plans.capacity() * std::mem::size_of::<(InstId, Lowered)>());
        return Ok(body);
    }
    // One function per lowering and callee type.
    let mut made: Vec<((Lowered, TypeId), FuncId)> = Vec::new();
    let mut calls = Vec::new();
    m.reserve(&mut calls, plans.len())?;
    for &(id, lowered) in &plans {
        let Some((_, call)) = body.insts.get(id).and_then(|i| site(&i.kind)) else {
            return Err(invariant("a planned asynchronous copy is not one"));
        };
        let Callee::Air { decl, .. } = call.callee else {
            return Err(invariant("a planned asynchronous copy is not an intrinsic"));
        };
        let ty = parts
            .funcs
            .get(decl)
            .map(|f| f.ty)
            .ok_or_else(|| invariant("a declaration out of range"))?;
        let f = match made.iter().find(|(k, _)| *k == (lowered, ty)) {
            Some(&(_, f)) => f,
            None => {
                let f = function(parts, lowered, ty, m)?;
                m.reserve(&mut made, 1)?;
                made.push(((lowered, ty), f));
                f
            }
        };
        calls.push((id, f));
    }
    m.release(made.capacity() * std::mem::size_of::<((Lowered, TypeId), FuncId)>());
    drop(made);
    m.release(plans.capacity() * std::mem::size_of::<(InstId, Lowered)>());
    drop(plans);

    // The calls rewritten in place to the functions built for them, then inlined.
    let mut body = body;
    for &(id, f) in &calls {
        m.tick(1)?;
        let inst = body
            .insts
            .get_mut(id)
            .ok_or_else(|| invariant("a planned call out of range"))?;
        let InstKind::Call(call) = &mut inst.kind else {
            return Err(invariant("a planned call is not a call"));
        };
        let lowered = Call {
            callee: Callee::Direct(f),
            args: call.args.clone(),
            fmf: FastMath::NONE,
            tail: TailKind::None,
            attrs: None,
            range: None,
        };
        m.charge(lowered.heap_bytes())?;
        m.release(call.heap_bytes());
        **call = lowered;
    }
    m.release(calls.capacity() * std::mem::size_of::<(InstId, FuncId)>());
    drop(calls);
    let func = ep.func;
    if let Some(replaced) = parts
        .funcs
        .get_mut(func)
        .ok_or_else(|| invariant("entry out of range"))?
        .body
        .replace(body)
    {
        replaced.release(m);
    }
    inline::inline(
        &parts.funcs,
        (&parts.types, &parts.symbols),
        &mut parts.consts,
        func,
        m,
    )?
    .ok_or_else(|| invariant("an expanded asynchronous copy was not inlined"))
}

/// The function `lowered` of the callee type `ty`: its parameters are the call's operands.
fn function(
    parts: &mut ModuleParts,
    lowered: Lowered,
    ty: TypeId,
    m: &mut Meter,
) -> Result<FuncId, Error> {
    let (ret, n) = match parts.types.get(ty) {
        Some(Type::Function {
            ret,
            params,
            vararg: false,
        }) => (*ret, params.len()),
        _ => return Err(invariant("an asynchronous copy's callee is not a function")),
    };
    let arg = |i: usize| -> Result<Value, Error> {
        u32::try_from(i)
            .map(|i| Value::Arg(ArgIdx::new(i)))
            .map_err(|_| invariant("parameters"))
    };
    let mut out = Out::new(parts);
    out.open(None)?;
    let result = match lowered {
        Lowered::Copy { size, align } => {
            if n != COPY_ARGS {
                return Err(invariant("an asynchronous copy's operands"));
            }
            let mut args = [Value::Arg(ArgIdx::new(0)); COPY_ARGS];
            for (i, a) in args.iter_mut().enumerate() {
                *a = arg(i)?;
            }
            copy_body(&mut out, &args, (size, align), m)?;
            Some(out.constant(ret, Const::Null, m)?)
        }
        Lowered::Event => Some(out.constant(ret, Const::Null, m)?),
        Lowered::IsNull { null } => Some(out.int(1, u64::from(null), m)?),
        Lowered::Wait => {
            simdgroup_barrier(&mut out, m)?;
            None
        }
    };
    out.ret(result, m)?;
    let Out { body, parts, .. } = out;
    let name = format!("__m2v.async_copy.{}", parts.funcs.len());
    let name = parts.symbols.intern(name.as_bytes(), m)?;
    let mut ps = Vec::new();
    m.reserve(&mut ps, n)?;
    ps.resize(n, Param::default());
    let f = Function {
        name,
        ty,
        linkage: Linkage::Internal,
        visibility: Visibility::Default,
        unnamed_addr: UnnamedAddr::Significant,
        cconv: CallConv::C,
        attrs: FnAttrs::default(),
        ret_attrs: ParamAttrs::default(),
        params: ps.into_boxed_slice(),
        body: Some(body),
        section: None,
        align: None,
    };
    parts.funcs.push(f, m)
}

/// `simdgroup_barrier(mem_flags::mem_threadgroup)`, in the form the module declares
/// `air.simdgroup.barrier` (an AIR 2.9 declaration's order operand is `seq_cst`), else AIR 2.8's.
fn simdgroup_barrier(out: &mut Out<'_>, m: &mut Meter) -> Result<(), Error> {
    const NAME: &str = "air.simdgroup.barrier";
    let void = out.ty(Type::Void, m)?;
    let i32t = out.int_ty(32, m)?;
    let ordered = out.parts.funcs.values().any(|f| {
        f.body.is_none()
            && out.parts.symbols.get(f.name) == Some(NAME.as_bytes())
            && matches!(
                out.parts.types.get(f.ty),
                Some(Type::Function { params, .. }) if params.len() == 3
            )
    });
    // `mem_flags::mem_threadgroup` (2), `memory_order_seq_cst` (5), simdgroup scope (4).
    let flags = out.k32(2, m)?;
    let scope = out.k32(4, m)?;
    if ordered {
        let order = out.k32(5, m)?;
        out.intrinsic(NAME, void, &[i32t; 3], &[flags, order, scope], m)?;
    } else {
        out.intrinsic(NAME, void, &[i32t; 2], &[flags, scope], m)?;
    }
    Ok(())
}

/// The copy of `size`-byte elements over the call's operands `a` ([`COPY_ARGS`]), each access
/// aligned to `align` bytes.
fn copy_body(
    out: &mut Out<'_>,
    a: &[Value; COPY_ARGS],
    (size, align): (u8, u8),
    m: &mut Meter,
) -> Result<(), Error> {
    let [_, _, dst, dst_epr, dst_stride, dst_tile, src, src_epr, src_stride, src_tile, offset, _] =
        *a;
    let i64t = out.int_ty(64, m)?;
    let i32t = out.int_ty(32, m)?;
    let bits = size.saturating_mul(8);
    let elem = out.int_ty(bits, m)?;
    let null = out.int(bits, 0, m)?;
    let lane = |out: &mut Out<'_>, i: u32, vec: Value, m: &mut Meter| -> Result<Value, Error> {
        let idx = out.k32(i, m)?;
        out.push(i64t, InstKind::ExtractElement { vec, idx }, m)
    };
    let (dtx, dty) = (lane(out, 0, dst_tile, m)?, lane(out, 1, dst_tile, m)?);
    let (stx, sty) = (lane(out, 0, src_tile, m)?, lane(out, 1, src_tile, m)?);
    let (ox, oy) = (lane(out, 0, offset, m)?, lane(out, 1, offset, m)?);
    // The calling lane's rank among the lanes executing the call, and their number.
    let one = out.k32(1, m)?;
    let rank = out.intrinsic(
        "air.simd_prefix_exclusive_sum.u.i32",
        i32t,
        &[i32t],
        &[one],
        m,
    )?;
    let lanes = out.intrinsic("air.simd_sum.u.i32", i32t, &[i32t], &[one], m)?;
    let rank = out.cast(CastOp::ZExt, rank, i64t, m)?;
    let lanes = out.cast(CastOp::ZExt, lanes, i64t, m)?;
    let zero = out.k64(0, m)?;
    let row_step = out.k64(1, m)?;
    let size = out.k64(u64::from(size), m)?;
    out.range_loop((i64t, zero, dty, row_step), &[], m, &mut |out, y, _, m| {
        out.range_loop((i64t, rank, dtx, lanes), &[], m, &mut |out, x, _, m| {
            // The source element and whether it lies inside the source tile, compared unsigned
            // (a coordinate below 0 is outside; a negative tile extent holds every coordinate).
            let sx = out.bin(i64t, BinOp::Add, x, ox, m)?;
            let sy = out.bin(i64t, BinOp::Add, y, oy, m)?;
            let in_x = out.icmp(IntPredicate::Ult, sx, stx, m)?;
            let in_y = out.icmp(IntPredicate::Ult, sy, sty, m)?;
            let i1 = out.int_ty(1, m)?;
            let inside = out.bin(i1, BinOp::And, in_x, in_y, m)?;
            let v = out.cond_value(inside, elem, null, m, &mut |out, m| {
                let at = element(out, (src_epr, src_stride), (sx, sy), size, m)?;
                let p = out.byte_gep(AddrSpace::Device, src, at, m)?;
                out.load(elem, p, u64::from(align), m)
            })?;
            let at = element(out, (dst_epr, dst_stride), (x, y), size, m)?;
            let p = out.byte_gep(AddrSpace::Threadgroup, dst, at, m)?;
            out.store(v, p, u64::from(align), m)?;
            Ok(Vec::new())
        })?;
        Ok(Vec::new())
    })?;
    Ok(())
}

/// The byte offset `(y · epr + x · stride) · size` of element `(x, y)`, in `i64`.
fn element(
    out: &mut Out<'_>,
    (epr, stride): (Value, Value),
    (x, y): (Value, Value),
    size: Value,
    m: &mut Meter,
) -> Result<Value, Error> {
    let i64t = out.int_ty(64, m)?;
    let row = out.bin(i64t, BinOp::Mul, y, epr, m)?;
    let col = out.bin(i64t, BinOp::Mul, x, stride, m)?;
    let index = out.bin(i64t, BinOp::Add, row, col, m)?;
    out.bin(i64t, BinOp::Mul, index, size, m)
}
