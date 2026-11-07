//! `byval` arguments made explicit before inlining.
//!
//! A `byval(T)` parameter is the callee's own copy of the `T` its argument points to, made at the
//! call. Apple's OpenCL compiler passes a struct argument so: the entry copies the value from its
//! constant buffer into an `alloca` and calls the kernel with that pointer `byval`. Each call
//! passing an argument `byval` gets the copy here, an `alloca T` and an `llvm.memcpy` of `T`'s
//! bytes from the argument just before the call, and passes the copy as a plain pointer; the
//! attribute then leaves the call and every function's parameters, so the inliner takes the call
//! as any other (`inline::takes`). An argument outside thread memory, or a `T` without a size,
//! leaves the module as it was, for the inliner to refuse.

use std::collections::BTreeMap;

use crate::budget::Meter;
use crate::error::Error;
use crate::intrinsics::{Elem, LlvmIntrinsic};
use crate::ir::{
    AddrSpace, Align, Block, Body, Call, Callee, Const, Constant, FastMath, FuncId, Inst, InstId,
    InstKind, ModuleParts, TailKind, Type, TypeId, TypeLayouts, Value,
};

use super::devirt::declare;
use super::rewrite::invariant;

/// The copy's declaration: `void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)`.
const MEMCPY: &[u8] = b"llvm.memcpy.p0.p0.i64";

/// What the copies of one `byval` type need: its alignment and its byte count as a constant.
#[derive(Clone, Copy)]
struct Copy {
    align: Option<Align>,
    len: Value,
}

/// The constants and types every copy shares.
struct Shared {
    /// By `byval` type.
    copies: BTreeMap<TypeId, Copy>,
    ptr: TypeId,
    void: TypeId,
    one: Value,
    no: Value,
    memcpy: FuncId,
}

/// Give each call of `parts` that passes an argument `byval` its copy, and drop the attribute from
/// the calls and from every function's parameters (see the module).
pub(super) fn explicit(parts: &mut ModuleParts, m: &mut Meter) -> Result<(), Error> {
    let mut types: Vec<TypeId> = Vec::new();
    for f in parts.funcs.values() {
        m.tick(1)?;
        for inst in f.body.iter().flat_map(|b| b.insts.values()) {
            // As many types as the module has, each listed once.
            for (_, t) in byvals(parts, &inst.kind) {
                if !types.contains(&t) {
                    types.push(t);
                }
            }
        }
    }
    if types.is_empty() {
        return Ok(());
    }
    if !every_argument_in_thread_memory(parts) {
        return Ok(());
    }
    // The layouts live only while the copies' sizes are read.
    let before = m.ledger();
    let layouts = TypeLayouts::new(&parts.data_layout, &parts.types, m)?;
    let sized: Option<Vec<(TypeId, u64, u64)>> = types
        .iter()
        .map(|&t| Some((t, layouts.alloc_size(t)?, layouts.abi_align(t)?)))
        .collect();
    drop(layouts);
    m.release(usize::try_from(m.ledger().saturating_sub(before)).unwrap_or(usize::MAX));
    let Some(sized) = sized else {
        return Ok(());
    };
    let i64t = parts.types.intern(Type::Int(64), m)?;
    let mut copies = BTreeMap::new();
    for (t, size, align) in sized {
        let len = intern(parts, i64t, Const::Int(size), m)?;
        copies.insert(
            t,
            Copy {
                align: Align::from_bytes(align),
                len,
            },
        );
    }
    let ptr = parts.types.intern(
        Type::Ptr {
            space: AddrSpace::Thread,
        },
        m,
    )?;
    let void = parts.types.intern(Type::Void, m)?;
    let i32t = parts.types.intern(Type::Int(32), m)?;
    let i1t = parts.types.intern(Type::Int(1), m)?;
    let one = intern(parts, i32t, Const::Int(1), m)?;
    let no = intern(parts, i1t, Const::Int(0), m)?;
    let memcpy_ty = parts.types.intern(
        Type::Function {
            ret: void,
            params: vec![ptr, ptr, i64t, i1t].into_boxed_slice(),
            vararg: false,
        },
        m,
    )?;
    let found = parts.funcs.iter().find(|(_, f)| {
        f.body.is_none() && f.ty == memcpy_ty && parts.symbols.get(f.name) == Some(MEMCPY)
    });
    let memcpy = match found {
        Some((id, _)) => id,
        None => declare(parts, (MEMCPY, memcpy_ty), m)?,
    };
    let shared = Shared {
        copies,
        ptr,
        void,
        one,
        no,
        memcpy,
    };
    let defined: Vec<FuncId> = parts
        .funcs
        .iter()
        .filter(|(_, f)| f.body.is_some())
        .map(|(id, _)| id)
        .collect();
    for f in defined {
        m.tick(1)?;
        let Some(body) = parts.funcs.get(f).and_then(|f| f.body.as_ref()) else {
            continue;
        };
        if !body
            .insts
            .values()
            .any(|i| byvals(parts, &i.kind).next().is_some())
        {
            continue;
        }
        let out = copied(parts, body, &shared, m)?;
        let replaced = parts
            .funcs
            .get_mut(f)
            .ok_or_else(|| invariant("a function out of range"))?
            .body
            .replace(out);
        if let Some(old) = replaced {
            old.release(m);
        }
    }
    for f in parts.funcs.values_mut() {
        for p in f.params.iter_mut() {
            p.attrs.byval = None;
        }
    }
    Ok(())
}

/// `c` of type `ty`, interned.
fn intern(
    parts: &mut ModuleParts,
    ty: TypeId,
    value: Const,
    m: &mut Meter,
) -> Result<Value, Error> {
    Ok(Value::Const(
        parts.consts.intern(Constant { ty, value }, m)?,
    ))
}

/// The arguments `kind` passes `byval`, if it is a call: each one's index and type, from the
/// call's attributes or its direct callee's parameters.
fn byvals<'p>(
    parts: &'p ModuleParts,
    kind: &'p InstKind,
) -> impl Iterator<Item = (usize, TypeId)> + 'p {
    let call = match kind {
        InstKind::Call(c) => Some(c.as_ref()),
        _ => None,
    };
    call.into_iter().flat_map(move |c| {
        let callee = match c.callee {
            Callee::Direct(g) => parts.funcs.get(g),
            _ => None,
        };
        (0..c.args.len()).filter_map(move |k| {
            let at_call = c
                .attrs
                .as_ref()
                .and_then(|a| a.params.get(k))
                .and_then(|p| p.byval);
            let at_callee = callee
                .and_then(|g| g.params.get(k))
                .and_then(|p| p.attrs.byval);
            at_call.or(at_callee).map(|t| (k, t))
        })
    })
}

/// Whether every argument passed `byval` is a pointer into thread memory, the one copy
/// [`MEMCPY`] makes.
fn every_argument_in_thread_memory(parts: &ModuleParts) -> bool {
    parts.funcs.values().all(|f| {
        let Some(body) = f.body.as_ref() else {
            return true;
        };
        let params: &[TypeId] = match parts.types.get(f.ty) {
            Some(Type::Function { params, .. }) => params,
            _ => &[],
        };
        body.insts.values().all(|inst| {
            let InstKind::Call(c) = &inst.kind else {
                return true;
            };
            byvals(parts, &inst.kind).all(|(k, _)| {
                let ty = match c.args.get(k) {
                    Some(Value::Inst(i)) => body.insts.get(*i).map(|i| i.ty),
                    Some(Value::Arg(a)) => params.get(a.index()).copied(),
                    Some(Value::Const(id)) => parts.consts.ty(*id),
                    None => None,
                };
                matches!(
                    ty.and_then(|t| parts.types.get(t)),
                    Some(Type::Ptr {
                        space: AddrSpace::Thread
                    })
                )
            })
        })
    })
}

/// `body` with each `byval` argument's copy made before its call (see the module).
fn copied(parts: &ModuleParts, body: &Body, shared: &Shared, m: &mut Meter) -> Result<Body, Error> {
    // Each old instruction's id in the output: a call's copies come before it.
    let mut result: Vec<InstId> = Vec::new();
    m.reserve(&mut result, body.insts.len())?;
    let mut next = 0u32;
    for (_, inst) in body.insts.iter() {
        let copies = u32::try_from(byvals(parts, &inst.kind).count())
            .map_err(|_| invariant("a call's arguments"))?;
        next = next
            .checked_add(2 * copies)
            .ok_or_else(|| invariant("instruction ids"))?;
        result.push(InstId::new(next));
        next = next
            .checked_add(1)
            .ok_or_else(|| invariant("instruction ids"))?;
    }
    let map = |v: Value| -> Result<Value, Error> {
        match v {
            Value::Inst(i) => result
                .get(i.index())
                .copied()
                .map(Value::Inst)
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
            let mut kind = inst.kind.clone();
            let mut failed = None;
            kind.for_each_value_mut(|v| match map(*v) {
                Ok(n) => *v = n,
                Err(e) => {
                    failed.get_or_insert(e);
                }
            });
            if let Some(e) = failed {
                return Err(e);
            }
            let passed: Vec<(usize, TypeId)> = byvals(parts, &inst.kind).collect();
            if let InstKind::Call(call) = &mut kind {
                for &(k, t) in &passed {
                    let copy = shared
                        .copies
                        .get(&t)
                        .copied()
                        .ok_or_else(|| invariant("a byval type without its copy"))?;
                    let source = call
                        .args
                        .get(k)
                        .copied()
                        .ok_or_else(|| invariant("a byval argument out of range"))?;
                    let alloca = out.push_inst(
                        Inst {
                            ty: shared.ptr,
                            kind: InstKind::Alloca {
                                ty: t,
                                count: shared.one,
                                align: copy.align,
                            },
                        },
                        m,
                    )?;
                    out.push_inst(
                        Inst {
                            ty: shared.void,
                            kind: InstKind::Call(Box::new(Call {
                                callee: Callee::Llvm {
                                    decl: shared.memcpy,
                                    op: LlvmIntrinsic::Memcpy {
                                        dst: AddrSpace::Thread,
                                        src: AddrSpace::Thread,
                                        len: Elem::I64,
                                        volatile: false,
                                    },
                                },
                                args: Box::new([Value::Inst(alloca), source, copy.len, shared.no]),
                                fmf: FastMath::NONE,
                                tail: TailKind::None,
                                attrs: None,
                                range: None,
                            })),
                        },
                        m,
                    )?;
                    if let Some(arg) = call.args.get_mut(k) {
                        *arg = Value::Inst(alloca);
                    }
                    if let Some(p) = call.attrs.as_mut().and_then(|a| a.params.get_mut(k)) {
                        p.byval = None;
                    }
                }
            }
            let id = out.push_inst(Inst { ty: inst.ty, kind }, m)?;
            if result.get(i.index()) != Some(&id) {
                return Err(invariant("a byval call's copies differ from their count"));
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
