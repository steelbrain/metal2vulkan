//! Metal 4.1's block reads as reads.
//!
//! `block_read<W, H>` and `sparse_block_read<W, H>` of a 2D, 2D array, or 2D depth texture compile
//! to `air.block_read_{texture,depth}_2d[_array].<T>(tex, smp, coord, [layer], offset, lod,
//! access, out, status, size)` ([`TexOp::BlockRead`]). `size` is the block's `(W, H)`, `(2, 1)` or
//! `(1, 2)` (Metal takes no other). The block's texels go to `out` in row order and, where
//! `status` is not null (a sparse block read), their residency bytes to `status`: 1 where the
//! texel is not resident, else 0 (Metal's own header loads each byte as a value in `[0, 2)`).
//!
//! [`expand`] makes each such call a call of a function built here, and inlines it. Texel `k` is
//! the read `air.read_<same>(tex, smp, coord + offset + d_k, (0, 0), level, access)` (a depth
//! read's depth word 1), with `d_0 = (0, 0)` and `d_1` `(1, 0)` or `(0, 1)`, and `level` the call's
//! `lod` as a signed `int` clamped into the texture's levels, `clamp(lod, 0, levels - 1)`
//! (`get_num_mip_levels`). Its residency byte is bit 0 of that read's status word, which is set
//! where the read touched a non-resident region (`lower::texture::status`).
//!
//! **Measured** on the M3 Max (macOS 27.0.1, Apple metal 32023.922, `-std=metal4.1`) over 2D, 2D
//! array, depth, and `read_write` 2D textures of two levels, through the read sampler
//! (`block_read(coord, offset, lod)`), a constexpr `address::clamp_to_edge` sampler, and argument
//! samplers of every address mode, border color, and mip filter:
//!
//! - texel `k` of a block is the texel the read of the same sampler at `coord + offset + d_k`
//!   answers at level `clamp(lod, 0, levels - 1)`: past the texture through the read sampler zero;
//!   through a sampler, its address mode's (clamped to the edge, or the border color);
//! - a block read's level is clamped into the texture's (a `lod` of 2 or 5 over two levels reads
//!   level 1, one of `-1` level 0) where a read's past the last level is out of bounds, so the
//!   level clamps first and the read it becomes is never past the last level;
//! - the offset applies through the read sampler too, so it joins the coordinate and the read
//!   takes none;
//! - `sparse_block_read` of a texture that is not sparse answers resident for both texels, in and
//!   out of bounds, as a sparse read does.
//!
//! Not expanded (lowering refuses each call left, typed): a block whose size is not a literal
//! `(2, 1)` or `(1, 2)`.

use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::{AirIntrinsic, Elem, TexOp, TexRole, Texture};
use crate::ir::{
    AddrSpace, ArgIdx, BinOp, Body, Call, CallConv, Callee, Const, FnAttrs, FuncId, Function,
    InstId, InstKind, IntPredicate, Linkage, ModuleParts, Param, ParamAttrs, Type, TypeId,
    UnnamedAddr, Value, Visibility,
};

use super::inline;
use super::out::Out;
use super::rewrite::invariant;

/// The second texel of a block, beside the first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Second {
    /// `block_read<2, 1>`: the texel at `(x + 1, y)`.
    Right,
    /// `block_read<1, 2>`: the texel at `(x, y + 1)`.
    Below,
}

/// What one call becomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Lowered {
    /// The block read, its immediates bound.
    read: Texture,
    /// Where its second texel is.
    second: Second,
    /// Whether it writes residency bytes (its `status` is not the null pointer).
    status: bool,
    /// Its access-qualifier immediate, where its declaration states one.
    access: Option<u64>,
}

/// The block read `kind` calls, with the call.
fn site(kind: &InstKind) -> Option<(Texture, &Call)> {
    let InstKind::Call(call) = kind else {
        return None;
    };
    match call.callee {
        Callee::Air {
            op: AirIntrinsic::Texture(t),
            ..
        } if t.op == TexOp::BlockRead => Some((t, call)),
        _ => None,
    }
}

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

/// The literal `<2 x i32>` `v`, when it is one.
fn pair(parts: &ModuleParts, v: Value) -> Option<(u64, u64)> {
    let Value::Const(c) = v else {
        return None;
    };
    let lane = |c| literal(parts, Value::Const(c));
    match &parts.consts.get(c)?.value {
        Const::Aggregate(lanes) => match lanes[..] {
            [x, y] => Some((lane(x)?, lane(y)?)),
            _ => None,
        },
        _ => None,
    }
}

/// How the block read `t` of `call` is expanded, or `None` when it is not (module doc).
fn plan(parts: &ModuleParts, t: Texture, call: &Call) -> Option<Lowered> {
    let layout = t.layout();
    let size = *call.args.get(layout.index(TexRole::BlockSize)?)?;
    let second = match pair(parts, size)? {
        (2, 1) => Second::Right,
        (1, 2) => Second::Below,
        _ => return None,
    };
    let status = *call.args.get(layout.index(TexRole::BlockStatus)?)?;
    let null = matches!(
        status,
        Value::Const(c) if parts.consts.get(c).is_some_and(|k| k.value == Const::Null)
    );
    // A bound declaration's access is a literal: its immediate.
    let access = match layout.index(TexRole::Access) {
        Some(i) => Some(literal(parts, *call.args.get(i)?)?),
        None => None,
    };
    Some(Lowered {
        read: t,
        second,
        status: !null,
        access,
    })
}

/// Whether `parts` declares a block read. Every site calls one, so a module that declares none
/// has none, and [`expand`] need not read its instructions.
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
        if matches!(air, Some(AirIntrinsic::Texture(t)) if t.op == TexOp::BlockRead) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Make each block read of the entry `func`'s inlined `body` that the module doc lists a call of
/// a function built for it, and inline the result.
pub(super) fn expand(
    body: Body,
    parts: &mut ModuleParts,
    func: FuncId,
    m: &mut Meter,
) -> Result<Body, Error> {
    if !declares_sites(parts, m)? {
        return Ok(body);
    }
    // Each call's lowering and declaration, in instruction order.
    let mut plans: Vec<(InstId, Lowered, FuncId)> = Vec::new();
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        let Some((t, call)) = site(&inst.kind) else {
            continue;
        };
        let Callee::Air { decl, .. } = call.callee else {
            return Err(invariant("a block read is not an intrinsic"));
        };
        if let Some(l) = plan(parts, t, call) {
            m.reserve(&mut plans, 1)?;
            plans.push((id, l, decl));
        }
    }
    if plans.is_empty() {
        m.release(plans.capacity() * std::mem::size_of::<(InstId, Lowered, FuncId)>());
        return Ok(body);
    }
    // One function per lowering and declaration.
    let mut made: Vec<((Lowered, FuncId), FuncId)> = Vec::new();
    let mut calls = Vec::new();
    m.reserve(&mut calls, plans.len())?;
    for &(id, lowered, decl) in &plans {
        let f = match made.iter().find(|(k, _)| *k == (lowered, decl)) {
            Some(&(_, f)) => f,
            None => {
                let f = function(parts, lowered, decl, m)?;
                m.reserve(&mut made, 1)?;
                made.push(((lowered, decl), f));
                f
            }
        };
        calls.push((id, f));
    }
    m.release(made.capacity() * std::mem::size_of::<((Lowered, FuncId), FuncId)>());
    drop(made);
    m.release(plans.capacity() * std::mem::size_of::<(InstId, Lowered, FuncId)>());
    drop(plans);

    // The calls made calls of the functions built for them, then inlined.
    inline::inline_calls(
        body,
        parts,
        func,
        calls,
        "an expanded block read was not inlined",
        m,
    )
}

/// The function `lowered` of the block read declared by `decl`: its parameters are the call's
/// operands.
fn function(
    parts: &mut ModuleParts,
    lowered: Lowered,
    decl: FuncId,
    m: &mut Meter,
) -> Result<FuncId, Error> {
    let (ty, name) = parts
        .funcs
        .get(decl)
        .map(|f| (f.ty, f.name))
        .ok_or_else(|| invariant("a declaration out of range"))?;
    let name = parts
        .symbols
        .get(name)
        .and_then(|n| std::str::from_utf8(n).ok())
        .ok_or_else(|| invariant("a block read's name"))?;
    let read_name = match name.strip_prefix("air.block_read_") {
        Some(rest) => format!("air.read_{rest}"),
        None => return Err(invariant("a block read's name")),
    };
    let params: Vec<TypeId> = match parts.types.get(ty) {
        Some(Type::Function {
            params,
            vararg: false,
            ..
        }) => params.to_vec(),
        _ => return Err(invariant("a block read's callee is not a function")),
    };
    let block = lowered.read;
    let layout = block.layout();
    let at = |role| -> Result<(Value, TypeId), Error> {
        let i = layout
            .index(role)
            .ok_or_else(|| invariant("a block read's operand"))?;
        let ty = *params
            .get(i)
            .ok_or_else(|| invariant("a block read's parameters"))?;
        let i = u32::try_from(i).map_err(|_| invariant("parameters"))?;
        Ok((Value::Arg(ArgIdx::new(i)), ty))
    };
    let texel = block
        .texel
        .ok_or_else(|| invariant("a block read without its texel"))?;
    // The read each texel is: the block's texture, texel, and access, in the current form (a
    // block read has no other).
    let read = Texture {
        op: TexOp::Read,
        status: true,
        coord16: false,
        ..block
    };
    let mut out = Out::new(parts);
    out.open(None)?;
    let i32t = out.int_ty(32, m)?;
    let i8t = out.int_ty(8, m)?;
    let pair_ty = out.ty(Type::Vector { elem: i32t, len: 2 }, m)?;
    let elem = out.ty(
        match texel.ty.elem {
            Elem::F32 => Type::Float,
            Elem::F16 => Type::Half,
            Elem::I32 => Type::Int(32),
            Elem::I16 => Type::Int(16),
            _ => return Err(invariant("a block read's texel")),
        },
        m,
    )?;
    let texel_ty = match texel.ty.lanes {
        1 => elem,
        n => out.ty(Type::Vector { elem, len: n }, m)?,
    };
    let result_ty = out.ty(
        Type::Struct {
            fields: Box::new([texel_ty, i8t]),
            packed: false,
            name: None,
        },
        m,
    )?;
    // Each texel's bytes, and the alignment of the block's texels (a power of two).
    let bytes = u64::from(texel.ty.lanes) * if texel.ty.elem.bits() == 16 { 2 } else { 4 };
    let (tex, tex_ty) = at(TexRole::Texture)?;
    let (smp, smp_ty) = at(TexRole::Sampler)?;
    let (coord, _) = at(TexRole::Coord)?;
    let (offset, _) = at(TexRole::Offset)?;
    let (lod, _) = at(TexRole::Lod)?;
    let (texels, _) = at(TexRole::BlockTexels)?;
    let layer = match block.image.array {
        true => Some(at(TexRole::Layer)?.0),
        false => None,
    };
    // `clamp(lod, 0, levels - 1)`, `lod` a signed `int`.
    let kind = if block.image.depth {
        "depth"
    } else {
        "texture"
    };
    let array = if block.image.array { "_array" } else { "" };
    let levels = out.intrinsic(
        &format!("air.get_num_mip_levels_{kind}_2d{array}"),
        i32t,
        &[tex_ty],
        &[tex],
        m,
    )?;
    let zero = out.k32(0, m)?;
    let negative = out.icmp(IntPredicate::Slt, lod, zero, m)?;
    let low = out.select(i32t, negative, zero, lod, m)?;
    let top = out.i32k(BinOp::Sub, levels, 1, m)?;
    let past = out.icmp(IntPredicate::Ugt, low, top, m)?;
    let level = out.select(i32t, past, top, low, m)?;
    let first = out.bin(pair_ty, BinOp::Add, coord, offset, m)?;
    let (dx, dy) = match lowered.second {
        Second::Right => (1, 0),
        Second::Below => (0, 1),
    };
    let lane = |v| match v {
        Value::Const(c) => Ok(c),
        _ => Err(invariant("a constant is not one")),
    };
    let lanes = Box::new([lane(out.k32(dx, m)?)?, lane(out.k32(dy, m)?)?]);
    let delta = out.constant(pair_ty, Const::Aggregate(lanes), m)?;
    let second = out.bin(pair_ty, BinOp::Add, first, delta, m)?;
    let no_offset = out.constant(pair_ty, Const::Null, m)?;
    let depth_word = out.k32(1, m)?;
    let access = match lowered.access {
        Some(a) => Some(out.int(32, a, m)?),
        None => None,
    };
    let read_layout = read.layout();
    for (k, at_k) in [first, second].into_iter().enumerate() {
        let mut tys = Vec::new();
        let mut args = Vec::new();
        for (role, _) in read_layout.positions() {
            let (v, t) = match role {
                TexRole::Texture => (tex, tex_ty),
                TexRole::Sampler => (smp, smp_ty),
                TexRole::DepthWord => (depth_word, i32t),
                TexRole::Coord => (at_k, pair_ty),
                TexRole::Layer => (
                    layer.ok_or_else(|| invariant("a block read's layer"))?,
                    i32t,
                ),
                TexRole::Offset => (no_offset, pair_ty),
                TexRole::Lod => (level, i32t),
                TexRole::Access => (
                    access.ok_or_else(|| invariant("a block read's access"))?,
                    i32t,
                ),
                _ => return Err(invariant("a read's operand a block read lacks")),
            };
            tys.push(t);
            args.push(v);
        }
        let result = out.intrinsic(&read_name, result_ty, &tys, &args, m)?;
        let value = out.push(
            texel_ty,
            InstKind::ExtractValue {
                agg: result,
                indices: Box::new([0]),
            },
            m,
        )?;
        let step = u64::try_from(k).map_err(|_| invariant("texels"))?;
        let to = out.byte_gep_k(AddrSpace::Thread, texels, step * bytes, m)?;
        out.store(value, to, bytes, m)?;
        if lowered.status {
            let (status, _) = at(TexRole::BlockStatus)?;
            let word = out.push(
                i8t,
                InstKind::ExtractValue {
                    agg: result,
                    indices: Box::new([1]),
                },
                m,
            )?;
            let one = out.int(8, 1, m)?;
            let byte = out.bin(i8t, BinOp::And, word, one, m)?;
            let to = out.byte_gep_k(AddrSpace::Thread, status, step, m)?;
            out.store(byte, to, 1, m)?;
        }
    }
    out.ret(None, m)?;
    let Out { body, parts, .. } = out;
    let name = format!("__m2v.block_read.{}", parts.funcs.len());
    let name = parts.symbols.intern(name.as_bytes(), m)?;
    let mut ps = Vec::new();
    m.reserve(&mut ps, params.len())?;
    ps.resize(params.len(), Param::default());
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
