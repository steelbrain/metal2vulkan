//! Pointers loaded from fields of constant globals.
//!
//! A constant global whose initializer is known once function constants are bound and the static
//! initializers have run may hold pointers: a program-scope `constexpr sampler` taken by value is
//! a struct holding the address of its sampler state, and a function that picks one of two such
//! samplers by a function constant (`select i1 %c, ptr @a, ptr @b`) reads the picked one's field
//! through a `getelementptr`. SCCP folds the select, and a load of a whole global, but not a load
//! of a field. The memory plan models no pointer read back from memory and no global outside the
//! constant and threadgroup spaces, so such a module was refused, under the values that made the
//! select one global.
//!
//! This pass reads such a load as the initializer's constant at the field: a non-volatile,
//! non-atomic load of a pointer whose address is a field of a constant global at constant indices
//! (a `getelementptr` instruction or constant expression over the global's own value type, its
//! first index 0), where the field's type is the load's. A copy LLVM made through an integer (a
//! load of the field as an integer of the pointer's size, then `inttoptr` to the field's type)
//! reads the same constant at its `inttoptr`. The global is never written (it is `constant`, and
//! initialized by no one else), so every run reads that constant: the address of the sampler
//! state, which the sampler family reads as the sampler it names. Only pointers are read this
//! way here; SCCP reads the integer fields ([`constant_field_load`]). SCCP then runs again over a
//! body this pass changed, so what the constants decide (an aggregate of them, a phi of them, a
//! member of either) folds too.
//!
//! **SCCP's reads.** SCCP itself reads a field of a global in the constant address space whose
//! value is known, at constant indices, as the initializer's constant there, when the load's type
//! holds only integers (an integer, a vector of integers, or an array or struct of those):
//! [`constant_field_address`] makes a `getelementptr` whose base and indices SCCP found constant
//! the address constant it computes, and [`constant_field_load`] reads the field it names. The
//! constant address space is read-only to every function an entry runs: only a static initializer
//! writes it, and its evaluation folds every store into the global's initializer and marks it
//! `constant`. So no store or call of the entry can change the field, and every run reads the
//! initializer, whatever path the address took (a `select` between the global and a buffer that
//! SCCP decided, a copy of a record's members into a callee's parameters). A field holding a
//! floating-point value is not read: its constant would hand float comparisons to SCCP's folder,
//! and float answers are the device's. A pointer field is this pass's, as above.

use crate::budget::Meter;
use crate::error::Error;
use crate::ir::{AddrSpace, Arena, Body, CastOp, Const, ConstExpr, ConstId, ConstTable, Constant};
use crate::ir::{Global, GlobalId, InstId, InstKind, ModuleParts, Type, TypeId, TypeTable, Value};

use super::fold::Folder;
use super::rewrite::{apply, scoped, BodyEdit};

/// Read every load of a pointer from a field of a constant global in `body`, and every
/// `inttoptr` of such a field's integer copy, as the field's constant (module docs), and say
/// whether there was one. Returns `body` itself when there is none.
pub(super) fn fold(
    body: Body,
    parts: &mut ModuleParts,
    m: &mut Meter,
) -> Result<(Body, bool), Error> {
    let mut found = Vec::new();
    let r = find(&body, parts, &mut found, m).and_then(|()| {
        if found.is_empty() {
            return Ok(None);
        }
        scoped(m, |m, held| {
            let mut edit = BodyEdit::keep_all(&body, m, held)?;
            for &(id, c) in &found {
                edit.replace(id, Value::Const(c));
            }
            apply(&body, &edit, &|c| c, m)
        })
        .map(Some)
    });
    m.release(found.capacity() * std::mem::size_of::<(InstId, ConstId)>());
    match r? {
        Some(rebuilt) => {
            body.release(m);
            Ok((rebuilt, true))
        }
        None => Ok((body, false)),
    }
}

/// Each instruction of `body` that reads a pointer field of a constant global (module docs), with
/// the field's constant.
fn find(
    body: &Body,
    parts: &mut ModuleParts,
    found: &mut Vec<(InstId, ConstId)>,
    m: &mut Meter,
) -> Result<(), Error> {
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        let Some(space) = parts.types.ptr_space(inst.ty) else {
            continue;
        };
        if !matches!(parts.types.get(inst.ty), Some(Type::Ptr { .. })) {
            continue;
        }
        let ptr = match inst.kind {
            InstKind::Load {
                ptr,
                volatile: false,
                atomic: None,
                ..
            } => ptr,
            // `inttoptr` of an integer load of the pointer's size: a copy through an integer.
            InstKind::Cast {
                op: CastOp::IntToPtr,
                v: Value::Inst(l),
            } => match body.insts.get(l).map(|l| (&l.kind, l.ty)) {
                Some((
                    &InstKind::Load {
                        ptr,
                        volatile: false,
                        atomic: None,
                        ..
                    },
                    ty,
                )) if parts.types.int_bits(ty).map(u32::from)
                    == Some(parts.data_layout.pointer(space.raw()).size_bits)
                    && !matches!(parts.types.get(ty), Some(Type::Vector { .. })) =>
                {
                    ptr
                }
                _ => continue,
            },
            _ => continue,
        };
        let Some((g, path)) = field_address(body, parts, ptr) else {
            continue;
        };
        let (types, consts, globals) = (&parts.types, &mut parts.consts, &parts.globals);
        if let Some(c) = field_value((types, consts, globals), g, &path, inst.ty, m)? {
            m.reserve(found, 1)?;
            found.push((id, c));
        }
    }
    Ok(())
}

/// The constant global `v` addresses a field of, and the member indices of that field below the
/// global (empty for the global itself): `@g`, a `getelementptr` over the field's type at
/// constant indices whose first is 0 (an instruction of `body` or a constant expression), through
/// no-op pointer casts. `None` for any other address.
fn field_address(body: &Body, parts: &ModuleParts, v: Value) -> Option<(GlobalId, Vec<u64>)> {
    let (types, consts, globals) = (&parts.types, &parts.consts, &parts.globals);
    match v {
        Value::Inst(i) => match &body.insts.get(i)?.kind {
            InstKind::Gep {
                src_ty,
                base,
                indices,
                ..
            } => {
                let outer = field_address(body, parts, *base)?;
                let mut idx = Vec::with_capacity(indices.len());
                for x in indices.iter() {
                    let Value::Const(k) = *x else {
                        return None;
                    };
                    idx.push(k);
                }
                stepped(types, consts, globals, outer, *src_ty, &idx)
            }
            _ => None,
        },
        Value::Const(c) => const_field(types, consts, globals, c),
        Value::Arg(_) => None,
    }
}

/// The global constant address `c` points into, and the member indices of the field it names:
/// `@g`, a no-op pointer cast of such an address, or a `getelementptr` constant expression over
/// one ([`stepped`]). `None` for any other constant.
fn const_field(
    types: &TypeTable,
    consts: &ConstTable,
    globals: &Arena<GlobalId, Global>,
    c: ConstId,
) -> Option<(GlobalId, Vec<u64>)> {
    match &consts.get(c)?.value {
        Const::GlobalAddr(g) => Some((*g, Vec::new())),
        Const::Expr(ConstExpr::Cast {
            op: CastOp::Bitcast | CastOp::AddrSpaceCast,
            ops: [k],
        }) => const_field(types, consts, globals, *k),
        Const::Expr(ConstExpr::Gep { src_ty, ops, .. }) => {
            let (base, rest) = ops.split_first()?;
            let outer = const_field(types, consts, globals, *base)?;
            stepped(types, consts, globals, outer, *src_ty, rest)
        }
        _ => None,
    }
}

/// The field a `getelementptr` over `src_ty` at constant `indices` reaches from the field
/// `(g, path)`: `g` at `path` followed by every index after the first, when `src_ty` is the field's own
/// type and the first index is 0 (an address inside the field, not past it). `None` for a
/// negative index or one past its aggregate.
fn stepped(
    types: &TypeTable,
    consts: &ConstTable,
    globals: &Arena<GlobalId, Global>,
    (g, mut path): (GlobalId, Vec<u64>),
    src_ty: TypeId,
    indices: &[ConstId],
) -> Option<(GlobalId, Vec<u64>)> {
    let mut t = member_type(types, globals.get(g)?.value_ty, &path)?;
    if t != src_ty {
        return None;
    }
    let (first, rest) = indices.split_first()?;
    if index(types, consts, *first)? != 0 {
        return None;
    }
    for &k in rest {
        let x = index(types, consts, k)?;
        t = member_type(types, t, &[x])?;
        path.push(x);
    }
    Some((g, path))
}

/// The value of the constant `getelementptr` index `k`. A negative index (its sign bit set)
/// addresses no field.
fn index(types: &TypeTable, consts: &ConstTable, k: ConstId) -> Option<u64> {
    let bits = types.int_bits(consts.ty(k)?)?;
    let x = match consts.get(k)?.value {
        Const::Int(x) => x,
        Const::Null => 0,
        _ => return None,
    };
    ((x >> (u32::from(bits.clamp(1, 64)) - 1)) & 1 == 0).then_some(x)
}

/// The type of the member at indices `path` below type `t`: a struct's field, an array's element.
fn member_type(types: &TypeTable, mut t: TypeId, path: &[u64]) -> Option<TypeId> {
    for &i in path {
        t = match types.get(t)? {
            Type::Struct { fields, .. } => usize::try_from(i).ok().and_then(|i| fields.get(i))?,
            Type::Array { elem, len } if i < *len => elem,
            _ => return None,
        }
        .to_owned();
    }
    Some(t)
}

/// SCCP's address of a field ([module docs](self)): the `getelementptr` over `src_ty` whose base
/// and indices are the constants `ops`, typed `ty`, as a constant expression, when it names a
/// field of a global SCCP reads ([`sccp_readable`]). `None` for every other address.
pub(super) fn constant_field_address(
    folder: &mut Folder<'_>,
    globals: &Arena<GlobalId, Global>,
    (src_ty, inbounds, ty): (TypeId, bool, TypeId),
    ops: &[ConstId],
    m: &mut Meter,
) -> Result<Option<ConstId>, Error> {
    let Some((base, indices)) = ops.split_first() else {
        return Ok(None);
    };
    let (types, consts) = (folder.types, &*folder.consts);
    let Some(field) = const_field(types, consts, globals, *base) else {
        return Ok(None);
    };
    if !globals.get(field.0).is_some_and(sccp_readable) {
        return Ok(None);
    }
    if stepped(types, consts, globals, field, src_ty, indices).is_none() {
        return Ok(None);
    }
    m.tick(1)?;
    let value = Const::Expr(ConstExpr::Gep {
        src_ty,
        inbounds,
        ops: ops.into(),
    });
    folder.consts.intern(Constant { ty, value }, m).map(Some)
}

/// SCCP's read of a field ([module docs](self)): the load of type `ty` from the constant address
/// `ptr` as the initializer's constant at the field it names, when that is a field of a global
/// SCCP reads ([`sccp_readable`]) and `ty` holds only integers ([`integers_only`]). `None` for
/// every other load.
pub(super) fn constant_field_load(
    folder: &mut Folder<'_>,
    globals: &Arena<GlobalId, Global>,
    ptr: ConstId,
    ty: TypeId,
    m: &mut Meter,
) -> Result<Option<ConstId>, Error> {
    let types = folder.types;
    if !integers_only(types, ty) {
        return Ok(None);
    }
    let Some((g, path)) = const_field(types, folder.consts, globals, ptr) else {
        return Ok(None);
    };
    if !globals.get(g).is_some_and(sccp_readable) {
        return Ok(None);
    }
    match field_value((types, &mut *folder.consts, globals), g, &path, ty, m)? {
        // A member the initializer leaves undefined is no constant: SCCP would refuse a branch on
        // it, where the device reads some value.
        Some(c) if folder.fully_defined(c, m)? => Ok(Some(c)),
        _ => Ok(None),
    }
}

/// The most member types [`integers_only`] visits.
const MEMBERS: u32 = 1 << 12;

/// Whether SCCP reads a field of `global` as its initializer's: a global of the constant address
/// space, never written (`constant`), initialized by no one else, with an initializer.
fn sccp_readable(global: &Global) -> bool {
    global.space == AddrSpace::Constant
        && global.constant
        && !global.externally_initialized
        && global.init.is_some()
}

/// Whether a value of type `ty` holds only integers: an integer, a vector of integers, or an array
/// or struct of such (no float, pointer or opaque member). A type of more than [`MEMBERS`] members
/// to visit is taken as not, which only leaves its loads unread.
pub(super) fn integers_only(types: &TypeTable, ty: TypeId) -> bool {
    // A walk with its own stack: nothing caps how deeply aggregate types nest, and a struct's
    // fields may share a type, so the visits are counted.
    let mut stack = vec![ty];
    let mut visits = 0u32;
    while let Some(t) = stack.pop() {
        visits += 1;
        if visits > MEMBERS {
            return false;
        }
        match types.get(t) {
            Some(Type::Int(_)) => {}
            Some(Type::Vector { elem, .. } | Type::Array { elem, .. }) => stack.push(*elem),
            Some(Type::Struct { fields, .. }) => stack.extend_from_slice(fields),
            _ => return false,
        }
    }
    true
}

/// The constant at member indices `path` of constant global `g`'s initializer, when the member's
/// type is `ty`, or when its first leaf at byte 0 (its first member, recursively) is of type `ty`:
/// an access at an aggregate's own address reads its first member's bytes. `None` for a global
/// that may be written or is initialized from outside (a function constant left unsupplied), an
/// index past its aggregate, a part the initializer leaves `undef` or `poison`, and a member of
/// another type.
fn field_value(
    (types, consts, globals): (&TypeTable, &mut ConstTable, &Arena<GlobalId, Global>),
    g: GlobalId,
    path: &[u64],
    ty: TypeId,
    m: &mut Meter,
) -> Result<Option<ConstId>, Error> {
    let Some(global) = globals.get(g) else {
        return Ok(None);
    };
    let (true, false, Some(mut c)) = (global.constant, global.externally_initialized, global.init)
    else {
        return Ok(None);
    };
    let mut t = global.value_ty;
    let mut rest = path.iter();
    loop {
        m.tick(1)?;
        // The next member: the path's, then the first member while the type is not `ty`.
        let i = match rest.next() {
            Some(&i) => i,
            None if t == ty => break,
            None => 0,
        };
        let next = match types.get(t) {
            Some(Type::Struct { fields, .. }) => {
                usize::try_from(i).ok().and_then(|i| fields.get(i).copied())
            }
            Some(Type::Array { elem, len }) if i < *len => Some(*elem),
            _ => None,
        };
        let Some(next) = next else {
            return Ok(None);
        };
        c = match consts.get(c).map(|k| &k.value) {
            Some(Const::Aggregate(elems)) => {
                match usize::try_from(i).ok().and_then(|i| elems.get(i)) {
                    Some(&e) => e,
                    None => return Ok(None),
                }
            }
            Some(Const::Null) => consts.intern(
                Constant {
                    ty: next,
                    value: Const::Null,
                },
                m,
            )?,
            _ => return Ok(None),
        };
        t = next;
    }
    Ok(match consts.get(c).map(|k| &k.value) {
        Some(Const::Undef | Const::Poison) | None => None,
        Some(_) => Some(c),
    })
}
