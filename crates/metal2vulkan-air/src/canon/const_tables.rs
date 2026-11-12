//! Pointers loaded from a constant table at an index only the run knows.
//!
//! A program-scope `constant constexpr array<sampler, N>` is a constant global whose initializer
//! holds each element's sampler state address, and a sampling through `table[i]` loads the
//! element's pointer field through a `getelementptr` whose indices are constants but `i`.
//! [`super::const_loads`] reads such a load at constant indices only; at an index only the run
//! knows, the load stayed, and the memory plan models no pointer read back from memory, so the
//! module was refused (`Lowering(Access)`).
//!
//! This pass reads such a load as the choice among the table's constants it is: a non-volatile,
//! non-atomic load of a pointer whose address is a `getelementptr` instruction over a field of a
//! constant global (the global itself, or a field of it at constant indices, as `const_loads`
//! reads one), its first index 0 and every later index a constant but one, which indexes an array
//! of at most [`ELEMENTS`] elements. Each element's field must be a constant of the load's type,
//! or the load stays. The load becomes the `select` chain over that index: element `k` where the
//! index is `k`, for each `k` below the last, and the last element for every other index. Metal
//! leaves an index outside the array undefined; the last element keeps the load on one of the
//! table's, as a handle array's index is clamped to its elements (`memory::handles`). The sampler
//! family reads a `select` of sampler addresses as a sampler of several candidates, sampled by a
//! switch at its use. SCCP runs again over a body this pass changed, so a chain whose index it
//! then finds constant folds to that element.

use std::collections::BTreeSet;

use crate::budget::Meter;
use crate::error::Error;
use crate::ir::{Block, Body, Const, ConstId, Constant, FastMath, FuncId, Idx, Inst, InstId};
use crate::ir::{InstKind, IntPredicate, ModuleParts, Type, TypeId, Value};

use super::const_loads::{field_address, field_value, index, member_type};
use super::rewrite::invariant;

/// The most elements of a table this pass reads a load of: a longer table's load stays, refused
/// by the memory plan as before. Each element costs the load a compare and a `select`.
pub(super) const ELEMENTS: u64 = 64;

/// A table load's replacement: its address, the index it chooses by, and each element's constant
/// in order.
struct Chain {
    gep: InstId,
    index: Value,
    elems: Vec<ConstId>,
}

impl Chain {
    /// The instructions the chain emits: a compare and a `select` per element below the last.
    fn emitted(&self) -> usize {
        2 * self.elems.len().saturating_sub(1)
    }
}

/// Read every load of a pointer from a constant table at an index only the run knows (module
/// docs) as its `select` chain, and say whether there was one. Returns `body` itself when there
/// is none.
pub(super) fn fold(
    body: Body,
    parts: &mut ModuleParts,
    func: FuncId,
    m: &mut Meter,
) -> Result<(Body, bool), Error> {
    let mut found: Vec<(InstId, Chain)> = Vec::new();
    let r = find(&body, parts, func, &mut found, m).and_then(|()| {
        if found.is_empty() {
            return Ok(None);
        }
        rebuild(&body, parts, func, &found, m).map(Some)
    });
    let held = found
        .iter()
        .map(|(_, c)| c.elems.capacity() * std::mem::size_of::<ConstId>())
        .sum::<usize>()
        + found.capacity() * std::mem::size_of::<(InstId, Chain)>();
    m.release(held);
    match r? {
        Some(rebuilt) => {
            body.release(m);
            Ok((rebuilt, true))
        }
        None => Ok((body, false)),
    }
}

/// Each load of `body` that reads a pointer from a constant table at a dynamic index, with its
/// chain, in instruction order.
fn find(
    body: &Body,
    parts: &mut ModuleParts,
    func: FuncId,
    found: &mut Vec<(InstId, Chain)>,
    m: &mut Meter,
) -> Result<(), Error> {
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        if !matches!(parts.types.get(inst.ty), Some(Type::Ptr { .. })) {
            continue;
        }
        let InstKind::Load {
            ptr: Value::Inst(p),
            volatile: false,
            atomic: None,
            ..
        } = inst.kind
        else {
            continue;
        };
        let Some(InstKind::Gep {
            src_ty,
            base,
            indices,
            ..
        }) = body.insts.get(p).map(|g| &g.kind)
        else {
            continue;
        };
        let Some((g, path)) = field_address(body, parts, *base) else {
            continue;
        };
        let Some(global) = parts.globals.get(g) else {
            continue;
        };
        if member_type(&parts.types, global.value_ty, &path) != Some(*src_ty) {
            continue;
        }
        let Some((first, rest)) = indices.split_first() else {
            continue;
        };
        if !matches!(first, Value::Const(k) if index(&parts.types, &parts.consts, *k) == Some(0)) {
            continue;
        }
        let Some((at, len, dynamic)) = dynamic_index(parts, *src_ty, rest) else {
            continue;
        };
        if len > ELEMENTS
            || value_ty(body, parts, func, dynamic)
                .and_then(|t| parts.types.int_bits(t))
                .is_none()
        {
            continue;
        }
        let mut elems = Vec::new();
        m.reserve(&mut elems, len as usize)?;
        for k in 0..len {
            m.tick(1)?;
            let mut field = path.clone();
            for (j, x) in rest.iter().enumerate() {
                field.push(match (j == at, *x) {
                    (true, _) => k,
                    (false, Value::Const(c)) => {
                        index(&parts.types, &parts.consts, c).ok_or_else(|| invariant("index"))?
                    }
                    (false, _) => return Err(invariant("a second dynamic index")),
                });
            }
            let (types, consts, globals) = (&parts.types, &mut parts.consts, &parts.globals);
            match field_value((types, consts, globals), g, &field, inst.ty, m)? {
                Some(c) => elems.push(c),
                None => break,
            }
        }
        if elems.len() as u64 != len {
            m.release(elems.capacity() * std::mem::size_of::<ConstId>());
            continue;
        }
        m.reserve(found, 1)?;
        found.push((
            id,
            Chain {
                gep: p,
                index: dynamic,
                elems,
            },
        ));
    }
    Ok(())
}

/// The one index of `indices` (a `getelementptr`'s after its first, stepping from `src_ty`) that
/// is no constant, with its position and the length of the array it indexes, when every other
/// index is a constant that stays inside its aggregate. `None` otherwise.
fn dynamic_index(
    parts: &ModuleParts,
    src_ty: TypeId,
    indices: &[Value],
) -> Option<(usize, u64, Value)> {
    let mut t = src_ty;
    let mut found = None;
    for (j, x) in indices.iter().enumerate() {
        match *x {
            Value::Const(c) => {
                let i = index(&parts.types, &parts.consts, c)?;
                t = member_type(&parts.types, t, &[i])?;
            }
            v => {
                let Some(Type::Array { elem, len }) = parts.types.get(t) else {
                    return None;
                };
                if found.is_some() || *len == 0 {
                    return None;
                }
                found = Some((j, *len, v));
                t = *elem;
            }
        }
    }
    found
}

/// The type of `v` in the entry `func`'s `body`.
fn value_ty(body: &Body, parts: &ModuleParts, func: FuncId, v: Value) -> Option<TypeId> {
    match v {
        Value::Inst(i) => body.insts.get(i).map(|i| i.ty),
        Value::Const(c) => parts.consts.ty(c),
        Value::Arg(a) => {
            let f = parts.funcs.get(func)?;
            match parts.types.get(f.ty)? {
                Type::Function { params, .. } => params.get(a.index()).copied(),
                _ => None,
            }
        }
    }
}

/// `body` with each load of `found` (in instruction order) replaced, in its place, by its chain:
/// the compares of its index with each element's position below the last, then the `select`s
/// from the last element back, the first of them the load's value. A table of one element is
/// that element's constant. A load's `getelementptr` that nothing else uses is dropped: the
/// address of a table element at a dynamic index is what the memory plan refuses.
fn rebuild(
    body: &Body,
    parts: &mut ModuleParts,
    func: FuncId,
    found: &[(InstId, Chain)],
    m: &mut Meter,
) -> Result<Body, Error> {
    let chain = |i: InstId| {
        found
            .binary_search_by_key(&i, |(c, _)| *c)
            .ok()
            .and_then(|k| found.get(k))
            .map(|(_, c)| c)
    };
    let i1 = parts.types.intern(Type::Int(1), m)?;
    // The addresses only the chains' loads use.
    let mut dropped: BTreeSet<InstId> = found.iter().map(|(_, c)| c.gep).collect();
    m.charge(dropped.len().saturating_mul(SET_ENTRY))?;
    let held = dropped.len().saturating_mul(SET_ENTRY);
    for (i, inst) in body.insts.iter() {
        m.tick(1)?;
        if chain(i).is_some() {
            continue;
        }
        inst.kind.for_each_value(|v| {
            if let Value::Inst(u) = v {
                dropped.remove(&u);
            }
        });
    }
    // Each old instruction's value in the output: a chain's is its last `select`, or the
    // element's constant for a table of one; a dropped address has none.
    let mut result: Vec<Option<Value>> = Vec::new();
    m.reserve(&mut result, body.insts.len())?;
    let mut next = 0usize;
    for (i, _) in body.insts.iter() {
        m.tick(1)?;
        match chain(i) {
            Some(c) => match c.elems.as_slice() {
                [only] => result.push(Some(Value::Const(*only))),
                _ => {
                    next += c.emitted();
                    result.push(Some(Value::Inst(id(next - 1)?)));
                }
            },
            None if dropped.contains(&i) => result.push(None),
            None => {
                next += 1;
                result.push(Some(Value::Inst(id(next - 1)?)));
            }
        }
    }
    let map = |val: Value| -> Result<Value, Error> {
        match val {
            Value::Inst(i) => result
                .get(i.index())
                .copied()
                .flatten()
                .ok_or_else(|| invariant("an operand out of range")),
            other => Ok(other),
        }
    };
    let mut out = Body::default();
    for (_, block) in body.blocks.iter() {
        m.tick(1)?;
        let start = id(out.insts.len())?;
        for i in block.inst_ids() {
            let inst = body
                .insts
                .get(i)
                .ok_or_else(|| invariant("instruction out of range"))?;
            match chain(i) {
                Some(c) if c.elems.len() == 1 => {}
                None if dropped.contains(&i) => {}
                Some(c) => {
                    let index = map(c.index)?;
                    let ty = value_ty(body, parts, func, c.index)
                        .ok_or_else(|| invariant("a table index without a type"))?;
                    let Some((&last, rest)) = c.elems.split_last() else {
                        return Err(invariant("a table without elements"));
                    };
                    let mut compares = Vec::new();
                    m.reserve(&mut compares, rest.len())?;
                    for k in 0..rest.len() {
                        let at = parts.consts.intern(
                            Constant {
                                ty,
                                value: Const::Int(k as u64),
                            },
                            m,
                        )?;
                        let cmp = out.push_inst(
                            Inst {
                                ty: i1,
                                kind: InstKind::ICmp {
                                    pred: IntPredicate::Eq,
                                    a: index,
                                    b: Value::Const(at),
                                },
                            },
                            m,
                        )?;
                        compares.push(cmp);
                    }
                    let mut chosen = Value::Const(last);
                    for (&cmp, &elem) in compares.iter().zip(rest).rev() {
                        let s = out.push_inst(
                            Inst {
                                ty: inst.ty,
                                kind: InstKind::Select {
                                    c: Value::Inst(cmp),
                                    t: Value::Const(elem),
                                    f: chosen,
                                    fmf: FastMath::default(),
                                },
                            },
                            m,
                        )?;
                        chosen = Value::Inst(s);
                    }
                    m.release(compares.capacity() * std::mem::size_of::<InstId>());
                }
                None => {
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
            }
        }
        let end = id(out.insts.len())?;
        out.blocks.push(
            Block {
                start,
                end,
                name: block.name,
            },
            m,
        )?;
    }
    m.release(result.capacity() * std::mem::size_of::<Option<Value>>() + held);
    Ok(out)
}

/// The bytes the ledger holds for each entry of a set of instruction ids.
const SET_ENTRY: usize = 4 * std::mem::size_of::<InstId>();

fn id(n: usize) -> Result<InstId, Error> {
    u32::try_from(n)
        .map(InstId::new)
        .map_err(|_| invariant("instruction ids"))
}
