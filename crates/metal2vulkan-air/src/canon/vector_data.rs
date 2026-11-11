//! OpenCL C's vector data functions as their element accesses.
//!
//! `vload<n>`, `vstore<n>`, `vload_half[<n>]`, `vloada_half<n>`, `vstore_half[<n>][_<rounding>]`
//! and `vstorea_half<n>[_<rounding>]` reach AIR as calls of [`VectorData`] intrinsics. [`expand`]
//! makes each call a call of a function built here, and inlines it: lane `i` of the value is
//! element `offset * s + i` of the data `p` points at, `s` the function's stride
//! ([`VectorData::stride`]: `n`, but 4 for the aligned `half3` forms), read or written as the
//! data's own element: a `half` for the `half` forms, widened to `float` by a load and narrowed to
//! `half` by a store. An access asks no more alignment than its element's.
//!
//! **Measured** on an M3 Max (macOS 27.0.1), a Radeon Pro 5300M and a UHD Graphics 630 (macOS
//! 26.7.1), in OpenCL mode:
//!
//! - `vload3` of a `uint` pointer one element past a buffer's start reads the three elements that
//!   follow it at each offset (no 16-byte alignment), and `vstore3` writes three; so do `vload3`
//!   and `vstore3` of `uchar`s and `vload2` and `vstore2` of `short`s one element past the start;
//! - `vload_half3` steps three halves and `vloada_half3` four; `vstore_half3` writes three halves
//!   a step of three and `vstorea_half3` three a step of four, but on the Radeon Pro 5300M, whose
//!   `vstorea_half3` also writes the fourth (the M3 Max and the UHD Graphics 630 leave it, and so
//!   does the expansion);
//! - a store narrows each lane to `half` as the device converts, whatever rounding the name spells:
//!   the Radeon Pro 5300M and the UHD Graphics 630 store each of `vstore_half_rtz`, `_rtp` and
//!   `_rtn` (and their `vstore_half4` forms) as `vstore_half_rte`, word for word over 32 values
//!   among them ties, overflows past the largest `half`, subnormals and zeros, where the M3 Max
//!   rounds as named. What no Mac guarantees is no part of the store.

use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::{AirIntrinsic, Elem, VectorData};
use crate::ir::{
    ArgIdx, BinOp, Body, CallConv, Callee, CastOp, Const, FnAttrs, FuncId, Function, InstId,
    InstKind, Linkage, ModuleParts, Param, ParamAttrs, Type, TypeId, UnnamedAddr, Value,
    Visibility,
};

use super::inline;
use super::out::Out;
use super::rewrite::invariant;

/// The vector data function `kind` calls, with its declaration.
fn site(kind: &InstKind) -> Option<(VectorData, FuncId)> {
    let InstKind::Call(call) = kind else {
        return None;
    };
    match call.callee {
        Callee::Air {
            decl,
            op: AirIntrinsic::VectorData(v),
        } => Some((v, decl)),
        _ => None,
    }
}

/// Whether `parts` declares a vector data function. Every site calls one, so a module that
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
        if matches!(air, Some(AirIntrinsic::VectorData(_))) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Make each vector data call of the entry `func`'s inlined `body` a call of a function built for
/// it, and inline the result.
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
    let mut made: Vec<((VectorData, FuncId), FuncId)> = Vec::new();
    let mut calls: Vec<(InstId, FuncId)> = Vec::new();
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        let Some(key) = site(&inst.kind) else {
            continue;
        };
        let f = match made.iter().find(|(k, _)| *k == key) {
            Some(&(_, f)) => f,
            None => {
                let f = function(parts, key, m)?;
                m.reserve(&mut made, 1)?;
                made.push((key, f));
                f
            }
        };
        m.reserve(&mut calls, 1)?;
        calls.push((id, f));
    }
    m.release(made.capacity() * std::mem::size_of::<((VectorData, FuncId), FuncId)>());
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
        "an expanded vector data function was not inlined",
        m,
    )
}

/// The IR type of an element `e` of vector data.
fn element(e: Elem) -> Result<Type, Error> {
    Ok(match e {
        Elem::I8 | Elem::I16 | Elem::I32 | Elem::I64 => {
            Type::Int(u8::try_from(e.bits()).map_err(|_| invariant("an element's width"))?)
        }
        Elem::F16 => Type::Half,
        Elem::F32 => Type::Float,
        _ => return Err(invariant("a vector data element")),
    })
}

/// The function of vector data function `v` declared by `decl`, whose parameters are the call's
/// operands: its element accesses (module doc).
fn function(
    parts: &mut ModuleParts,
    (v, decl): (VectorData, FuncId),
    m: &mut Meter,
) -> Result<FuncId, Error> {
    let ty = parts
        .funcs
        .get(decl)
        .map(|f| f.ty)
        .ok_or_else(|| invariant("a declaration out of range"))?;
    let (ret, params): (TypeId, Vec<TypeId>) = match parts.types.get(ty) {
        Some(Type::Function {
            ret,
            params,
            vararg: false,
        }) => (*ret, params.to_vec()),
        _ => return Err(invariant("a vector data function is not a function")),
    };
    let arg = |i: u32| Value::Arg(ArgIdx::new(i));
    // A store's value comes first; then the offset and the pointer.
    let (value, offset, p) = if v.stores() {
        (Some(arg(0)), arg(1), arg(2))
    } else {
        (None, arg(0), arg(1))
    };
    let value_ty = match value {
        Some(_) => *params
            .first()
            .ok_or_else(|| invariant("a vector data store's parameters"))?,
        None => ret,
    };
    let lanes = v.ty.lanes;
    let widened = v.data() != v.ty.elem;
    let bytes = u64::from(v.data().bits() / 8);
    let mut out = Out::new(parts);
    out.open(None)?;
    let lane_ty = out.ty(element(v.ty.elem)?, m)?;
    let data_ty = out.ty(element(v.data())?, m)?;
    let i64t = out.int_ty(64, m)?;
    let ptr_ty = out.ptr_ty(v.space, m)?;
    let stride = out.k64(u64::from(v.stride()), m)?;
    let first = out.bin(i64t, BinOp::Mul, offset, stride, m)?;
    let mut loaded = match lanes {
        1 => None,
        _ => Some(out.constant(value_ty, Const::Undef, m)?),
    };
    for i in 0..lanes {
        let index = match i {
            0 => first,
            _ => {
                let k = out.k64(u64::from(i), m)?;
                out.bin(i64t, BinOp::Add, first, k, m)?
            }
        };
        let at = out.push(
            ptr_ty,
            InstKind::Gep {
                src_ty: data_ty,
                base: p,
                indices: Box::new([index]),
                inbounds: false,
            },
            m,
        )?;
        let lane = out.k32(i, m)?;
        match value {
            Some(value) => {
                let x = match lanes {
                    1 => value,
                    _ => out.push(
                        lane_ty,
                        InstKind::ExtractElement {
                            vec: value,
                            idx: lane,
                        },
                        m,
                    )?,
                };
                let x = if widened {
                    out.cast(CastOp::FPTrunc, x, data_ty, m)?
                } else {
                    x
                };
                out.store(x, at, bytes, m)?;
            }
            None => {
                let x = out.load(data_ty, at, bytes, m)?;
                let x = if widened {
                    out.cast(CastOp::FPExt, x, lane_ty, m)?
                } else {
                    x
                };
                loaded = Some(match loaded {
                    Some(vec) => out.push(
                        value_ty,
                        InstKind::InsertElement {
                            vec,
                            elt: x,
                            idx: lane,
                        },
                        m,
                    )?,
                    None => x,
                });
            }
        }
    }
    let result = match value {
        Some(_) => None,
        None => Some(loaded.ok_or_else(|| invariant("a vector data load of no lane"))?),
    };
    out.ret(result, m)?;
    let Out { body, parts, .. } = out;
    let name = format!("__m2v.vector_data.{}", parts.funcs.len());
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
