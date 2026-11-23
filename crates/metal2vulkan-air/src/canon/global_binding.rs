//! Program-scope buffer bindings as buffer parameters.
//!
//! MSL 3.2 lets a program bind a buffer at program scope (`constant T * constant p
//! [[buffer(n)]];`), which every function may read. `xcrun metal` 32023.922 makes it a
//! constant-space global, externally initialized and `undef`, makes each read of it (`&p`
//! included) a load of the pointer it holds, and lists it in `!air.global_bindings`
//! ([`GlobalBinding`]). Metal fills it from the executing stage's own argument table: measured on
//! an Apple M3 Max under macOS 27.0.1 (m3-max-70268), a kernel's global reads and writes the buffer
//! bound at its index, a vertex and a fragment function that read one global each read their own
//! stage's buffer there, and two globals of one type at one index, or a global and a parameter of
//! one type, read the one buffer bound there. So `expand` makes each load of a bound buffer's
//! pointer a read of a parameter of the entry: its own parameter of the binding's role and type
//! where it takes one, else one `expand` adds, shared by the bindings of one role and type.
//! Lowering then binds an added parameter where a `[[buffer(n)]]` parameter binds.
//!
//! Three shapes stay refused, each where it is found: an added parameter whose slot another
//! parameter of the entry binds ([`shared_slot`]), and, through what pruning keeps, any read of a
//! binding that did not become a parameter ([`left`]): a texture's, a sampler's or threadgroup
//! memory's, or another read of a buffer's.

use crate::abi::{AbiInt, EntryPoint, GlobalBinding, ParamBinding, ParamRole, Slot};
use crate::budget::Meter;
use crate::error::{DiagName, Error, Feature, Location, Unsupported};
use crate::ir::{AddrSpace, ArgIdx, Body, Const, GlobalId, InstKind, ModuleParts, TypeId, Value};

use super::interface;
use super::rewrite::{append_params, apply, scoped, BodyEdit};

/// The space of the pointer a load of binding `b`'s global reads, where `expand` makes such a load
/// a parameter: `b` binds one buffer of the device or constant space at a stated slot.
fn buffer_space(b: &GlobalBinding) -> Option<AddrSpace> {
    match &b.role {
        ParamRole::Buffer(arg)
            if matches!(arg.space, AddrSpace::Device | AddrSpace::Constant)
                && matches!(
                    arg.location,
                    Slot {
                        index: AbiInt::Const(_),
                        count: AbiInt::Const(1)
                    }
                ) =>
        {
            Some(arg.space)
        }
        _ => None,
    }
}

/// The binding of `bindings` whose global the load `kind` of a value of type `ty` reads, when
/// `expand` makes the load a parameter.
fn bound<'b>(
    parts: &ModuleParts,
    bindings: &'b [GlobalBinding],
    kind: &InstKind,
    ty: TypeId,
) -> Option<&'b GlobalBinding> {
    let InstKind::Load {
        ptr: Value::Const(c),
        volatile: false,
        atomic: None,
        ..
    } = kind
    else {
        return None;
    };
    let Some(Const::GlobalAddr(g)) = parts.consts.get(*c).map(|k| &k.value) else {
        return None;
    };
    let b = bindings.iter().find(|b| b.global == *g)?;
    (parts.types.ptr_space(ty) == Some(buffer_space(b)?)).then_some(b)
}

/// Make each load of a bound buffer's pointer in the entry `ep`'s `body` a read of a parameter
/// (see the module), returning the new body and the parameters added for them (none, and `body`
/// unchanged, when it loads none).
pub(super) fn expand(
    body: Body,
    parts: &mut ModuleParts,
    ep: &EntryPoint,
    bindings: &[GlobalBinding],
    m: &mut Meter,
) -> Result<(Body, Vec<ParamBinding>), Error> {
    // The parameters to add, in the order the body first loads them: one per role and type.
    let mut added: Vec<(&GlobalBinding, TypeId)> = Vec::new();
    let mut loads = false;
    for (_, inst) in body.insts.iter() {
        m.tick(1)?;
        let Some(b) = bound(parts, bindings, &inst.kind, inst.ty) else {
            continue;
        };
        loads = true;
        if own(ep, b).is_none() && !added.iter().any(|(a, _)| alike(a, b)) {
            m.reserve(&mut added, 1)?;
            added.push((b, inst.ty));
        }
    }
    if !loads {
        return Ok((body, Vec::new()));
    }
    let first = if added.is_empty() {
        0
    } else {
        let tys: Vec<TypeId> = added.iter().map(|&(_, ty)| ty).collect();
        append_params(parts, ep.func, &tys, m)?
    };
    let param = |b: &GlobalBinding| -> Option<u32> {
        own(ep, b).or_else(|| {
            let at = added.iter().position(|(a, _)| alike(a, b))?;
            first.checked_add(u32::try_from(at).ok()?)
        })
    };
    let out = scoped(m, |m, held| {
        let mut edit = BodyEdit::keep_all(&body, m, held)?;
        for (i, inst) in body.insts.iter() {
            m.tick(1)?;
            if let Some(p) = bound(parts, bindings, &inst.kind, inst.ty).and_then(param) {
                edit.replace(i, Value::Arg(ArgIdx::new(p)));
            }
        }
        apply(&body, &edit, &|c| c, m)
    })?;
    body.release(m);
    let params = added
        .iter()
        .zip(first..)
        .map(|(&(b, _), param)| ParamBinding {
            param,
            role: b.role.clone(),
            unused: false,
            predicate: None,
            ty: b.ty.clone(),
            name: b.name.clone(),
        })
        .collect();
    Ok((out, params))
}

/// Whether bindings `a` and `b` bind one argument: one role and one type.
fn alike(a: &GlobalBinding, b: &GlobalBinding) -> bool {
    a.role == b.role && a.ty == b.ty
}

/// The parameter of `ep` that binding `b`'s reads read: one of its role and type that no predicate
/// gates and that the entry uses.
fn own(ep: &EntryPoint, b: &GlobalBinding) -> Option<u32> {
    ep.params
        .iter()
        .find(|p| p.role == b.role && p.ty == b.ty && p.predicate.is_none() && !p.unused)
        .map(|p| p.param)
}

/// Refuse the first of the `added` parameters whose buffer slot a present parameter of the entry
/// `params`, or an earlier added one, also binds: Metal reads the one buffer bound there through
/// both, which two parameters do not model.
pub(super) fn shared_slot(
    name: &DiagName,
    params: &[ParamBinding],
    added: &[ParamBinding],
) -> Result<(), Error> {
    for (i, p) in added.iter().enumerate() {
        let mut earlier = params.iter().chain(added.get(..i).unwrap_or_default());
        if earlier.any(|q| interface::overlap(p, q)) {
            return Err(refused(
                Feature::ProgramScopeSharedSlot,
                name,
                p.name.clone(),
            ));
        }
    }
    Ok(())
}

/// Refuse the first of the `bindings` whose global `kept` (what pruning keeps of the entry's
/// globals) still holds: a read of it did not become a parameter.
pub(super) fn left(
    name: &DiagName,
    bindings: &[GlobalBinding],
    kept: impl Fn(GlobalId) -> bool,
) -> Result<(), Error> {
    let Some(b) = bindings.iter().find(|b| kept(b.global)) else {
        return Ok(());
    };
    let feature = match buffer_space(b) {
        Some(_) => Feature::ProgramScopeRead,
        None => Feature::ProgramScopeResource,
    };
    Err(refused(feature, name, b.name.clone()))
}

/// `feature`, refused in entry `name` at the binding named `binding`.
fn refused(feature: Feature, name: &DiagName, binding: Option<DiagName>) -> Error {
    let refusal = Unsupported::not_implemented(feature).at(Location::metadata(
        Some(name.clone()),
        None,
        Some("air.global_bindings"),
    ));
    match binding {
        Some(b) => refusal.naming(b).into(),
        None => refusal.into(),
    }
}
