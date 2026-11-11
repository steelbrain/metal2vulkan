//! OpenCL mode's 1D images as the textures Apple's OpenCL binds for them.
//!
//! Apple's OpenCL compiler declares an `image1d_t` argument `texture1d` and an `image1d_array_t`
//! `texture1d_array`, and its OpenCL binds a 2D texture of height 1 for the first and a 2D array
//! texture of height 1 for the second. A pipeline of OpenCL mode reads those textures as the
//! kernel's 1D images: row 0, of the layer an array access names. Measured on an M3 Max (macOS
//! 27.0.1) and a Radeon Pro 5300M (macOS 26.7.1): the reflection states the two arguments
//! `MTLTextureType2D` and `MTLTextureType2DArray`; over 2D textures of height 1 bound there, a read
//! with and without a sampler, a sample at integer, unnormalized and normalized coordinates (a
//! linear filter at a border-clamped edge answers the texel, blending in no row of the border), a
//! write, `get_image_width` and `get_image_array_size` answer the 1D image's values. With 1D
//! textures bound, the M3 Max reads texel 0 at every coordinate.
//!
//! [`expand`] makes each such call of a 1D texture in a kernel of OpenCL mode a call of a function
//! built here, and inlines it: the same operation of a 2D texture, or a 2D array texture for a 1D
//! array, its coordinate `(x, 0)`, a sampled coordinate `(x, 0.5)` (the row's centre in
//! normalized and unnormalized coordinates alike), its offset `(o, 0)` or, where the 1D form has
//! none, `(0, 0)`. Lowering then declares the slot, and the binder binds it, as the 2D texture
//! Metal's OpenCL binds. Not rewritten: `get_num_mip_levels`, whose 1D answer (0) a 2D texture's is
//! not, and the image format queries, which read a hidden buffer whatever the shape
//! (`canon::opencl`); neither is OpenCL C's.

use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::{AirIntrinsic, ImageDim, ImageShape, Query, TexOp, TexRole, Texture};
use crate::ir::{
    ArgIdx, Body, Call, CallConv, Callee, Const, FnAttrs, FuncId, Function, InstId, InstKind,
    Linkage, ModuleParts, Param, ParamAttrs, Type, TypeId, UnnamedAddr, Value, Visibility,
};

use super::inline;
use super::out::Out;
use super::rewrite::invariant;

/// The 2D operation that reads 1D operation `t`'s texture as Metal's OpenCL binds it, where `t`
/// is one [`expand`] rewrites.
fn widened(t: Texture) -> Option<Texture> {
    if t.image.dim != ImageDim::D1 {
        return None;
    }
    match t.op {
        TexOp::Read
        | TexOp::Sample { .. }
        | TexOp::Write
        | TexOp::Query(Query::Width | Query::ArraySize) => Some(Texture {
            image: ImageShape {
                dim: ImageDim::D2,
                ..t.image
            },
            ..t
        }),
        _ => None,
    }
}

/// The 1D texture operation `kind` calls that [`expand`] rewrites, with its declaration and call.
fn site(kind: &InstKind) -> Option<(Texture, FuncId, &Call)> {
    let InstKind::Call(call) = kind else {
        return None;
    };
    match call.callee {
        Callee::Air {
            decl,
            op: AirIntrinsic::Texture(t),
        } if widened(t).is_some() => Some((t, decl, call)),
        _ => None,
    }
}

/// Whether `parts` declares a 1D texture operation [`expand`] rewrites. Every site calls one, so
/// a module that declares none has none, and [`expand`] need not read its instructions.
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
        if matches!(air, Some(AirIntrinsic::Texture(t)) if widened(t).is_some()) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Make each 1D texture operation of the entry `func`'s inlined `body` that the module doc lists
/// a call of a function built for it, and inline the result.
pub(super) fn expand(
    body: Body,
    parts: &mut ModuleParts,
    func: FuncId,
    m: &mut Meter,
) -> Result<Body, Error> {
    if !declares_sites(parts, m)? {
        return Ok(body);
    }
    // One function per operation and declaration, and each call's.
    let mut made: Vec<((Texture, FuncId), FuncId)> = Vec::new();
    let mut calls: Vec<(InstId, FuncId)> = Vec::new();
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        let Some((t, decl, call)) = site(&inst.kind) else {
            continue;
        };
        let f = match made.iter().find(|(k, _)| *k == (t, decl)) {
            Some(&(_, f)) => f,
            None => {
                let f = function(parts, (t, decl), &call.args, m)?;
                m.reserve(&mut made, 1)?;
                made.push(((t, decl), f));
                f
            }
        };
        m.reserve(&mut calls, 1)?;
        calls.push((id, f));
    }
    m.release(made.capacity() * std::mem::size_of::<((Texture, FuncId), FuncId)>());
    drop(made);
    if calls.is_empty() {
        m.release(calls.capacity() * std::mem::size_of::<(InstId, FuncId)>());
        return Ok(body);
    }
    inline::inline_calls(
        body,
        parts,
        func,
        calls,
        "a widened 1D texture operation was not inlined",
        m,
    )
}

/// The function of 1D operation `t` declared by `decl`, whose parameters are the call's operands:
/// it calls the 2D operation [`widened`] names. `args`, a call's operands, give the immediates
/// (`t` binds them: every call of `t` passes the same).
fn function(
    parts: &mut ModuleParts,
    (t, decl): (Texture, FuncId),
    args: &[Value],
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
        .ok_or_else(|| invariant("a 1D texture operation's name"))?;
    // The dimension token of the name: `texture_1d[_array]` to `texture_2d[_array]`.
    let wide_name = match name.split_once("_texture_1d") {
        Some((head, tail)) => format!("{head}_texture_2d{tail}"),
        None => return Err(invariant("a 1D texture operation's name")),
    };
    let (ret, params): (TypeId, Vec<TypeId>) = match parts.types.get(ty) {
        Some(Type::Function {
            ret,
            params,
            vararg: false,
        }) => (*ret, params.to_vec()),
        _ => return Err(invariant("a 1D texture operation is not a function")),
    };
    let wide = widened(t).ok_or_else(|| invariant("a 1D texture operation's 2D form"))?;
    let narrow = t.layout();
    let at = |role| -> Result<(usize, TypeId), Error> {
        let i = narrow
            .index(role)
            .ok_or_else(|| invariant("a 1D texture operation's operand"))?;
        let ty = *params
            .get(i)
            .ok_or_else(|| invariant("a 1D texture operation's parameters"))?;
        Ok((i, ty))
    };
    let param = |i: usize| -> Result<Value, Error> {
        let i = u32::try_from(i).map_err(|_| invariant("parameters"))?;
        Ok(Value::Arg(ArgIdx::new(i)))
    };
    let void = matches!(parts.types.get(ret), Some(Type::Void));
    let mut out = Out::new(parts);
    out.open(None)?;
    let mut tys = Vec::new();
    let mut wide_args = Vec::new();
    for (role, _) in wide.layout().positions() {
        let (v, ty) = match role {
            // The row's centre for a sampled coordinate, row 0 for an integer one.
            TexRole::Coord => {
                let (i, ty) = at(role)?;
                let float = matches!(out.parts.types.get(ty), Some(Type::Float));
                pair(&mut out, param(i)?, ty, float.then_some(0.5), m)?
            }
            TexRole::Offset => match narrow.index(role) {
                Some(_) => {
                    let (i, ty) = at(role)?;
                    pair(&mut out, param(i)?, ty, None, m)?
                }
                // A read's 2D offset, of its coordinate's lane type, which the 1D form has not.
                None => {
                    let (_, ty) = at(TexRole::Coord)?;
                    let pair_ty = out.ty(Type::Vector { elem: ty, len: 2 }, m)?;
                    (out.constant(pair_ty, Const::Null, m)?, pair_ty)
                }
            },
            // The immediates, which the 2D call must state as constants.
            TexRole::Access | TexRole::LodMode | TexRole::OffsetFlag | TexRole::Component => {
                let (i, ty) = at(role)?;
                let v = *args
                    .get(i)
                    .ok_or_else(|| invariant("a 1D texture operation's operands"))?;
                if !matches!(v, Value::Const(_)) {
                    return Err(invariant("a bound immediate is not a constant"));
                }
                (v, ty)
            }
            _ => {
                let (i, ty) = at(role)?;
                (param(i)?, ty)
            }
        };
        tys.push(ty);
        wide_args.push(v);
    }
    let result = out.intrinsic(&wide_name, ret, &tys, &wide_args, m)?;
    // The declaration decodes as the 2D operation meant, its immediates bound alike.
    let bound = match result {
        Value::Inst(id) => out.body.insts.get(id).map(|i| &i.kind),
        _ => None,
    };
    let meant =
        |c: &Call| matches!(c.callee, Callee::Air { op, .. } if op == AirIntrinsic::Texture(wide));
    if !matches!(bound, Some(InstKind::Call(c)) if meant(c)) {
        return Err(invariant("a 1D texture operation's 2D form"));
    }
    out.ret((!void).then_some(result), m)?;
    let Out { body, parts, .. } = out;
    let name = format!("__m2v.opencl_1d.{}", parts.funcs.len());
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

/// `v`, a lane of type `ty`, as lane 0 of a pair whose lane 1 is the `float` `y` where `Some`,
/// else zero: the pair and its type.
fn pair(
    out: &mut Out<'_>,
    v: Value,
    ty: TypeId,
    y: Option<f32>,
    m: &mut Meter,
) -> Result<(Value, TypeId), Error> {
    let pair_ty = out.ty(Type::Vector { elem: ty, len: 2 }, m)?;
    let base = match y {
        Some(y) => {
            let lane = |v| match v {
                Value::Const(c) => Ok(c),
                _ => Err(invariant("a constant is not one")),
            };
            let lanes = Box::new([lane(out.f32k(0.0, m)?)?, lane(out.f32k(y, m)?)?]);
            out.constant(pair_ty, Const::Aggregate(lanes), m)?
        }
        None => out.constant(pair_ty, Const::Null, m)?,
    };
    let idx = out.k32(0, m)?;
    let v = out.push(
        pair_ty,
        InstKind::InsertElement {
            vec: base,
            elt: v,
            idx,
        },
        m,
    )?;
    Ok((v, pair_ty))
}
