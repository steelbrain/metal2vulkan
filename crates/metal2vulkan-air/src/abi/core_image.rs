//! `!air.ci`: the functions a module exports as Core Image kernels.
//!
//! Each list entry is `!{ptr @fn, !outputs, !inputs}`:
//!
//! - `!outputs` holds at most one node, the return value: `!{!"air.ci_builtin", <attr>*}`, whose
//!   only attribute is `air.arg_type_name`;
//! - `!inputs` holds one node per LLVM parameter, `!{i32 <param>, !"<role>", <attr>*}`, where the
//!   role is `air.ci_builtin`, `air.ci_matrix`, `air.ci_sampler`, `air.ci_texture` (requires
//!   `air.location_count, i32 <n>` and one of `air.sample`, `air.read`, `air.write`,
//!   `air.read_write`), or `air.ci_pointer` (requires one of `air.read`, `air.write`,
//!   `air.read_write`, and `air.arg_type_size, i32 <n>` and `air.arg_type_align_size, i32 <n>`);
//! - every node may also carry `air.arg_type_name` and (inputs only) `air.arg_name`, both kept
//!   as [`DiagName`]s.
//!
//! Attributes may come in any order and each at most once. An unknown key, a key the role does
//! not define, a repeated or conflicting key, or a missing required key is an
//! [`AbiError`]. The parameter bindings, the output, each role's IR
//! type, and (for typed-pointer bitcode) a pointer's size against its pointee are cross-checked
//! against the function by `abi::check`.

use super::args::key_of;
use super::keys::{Key, BUFFER_ACCESS, TEXTURE_ACCESS};
use super::md::{kept_diag, node_ops, Ops, Site, Tok};
use super::{Access, CoreImageKernel, CoreImageOutput, CoreImageParam, CoreImageRole};
use crate::budget::Meter;
use crate::error::Error;
use crate::error::{AbiError, DiagName};
use crate::ir::{MdId, Module};

/// The named list, for error locations.
pub(super) const LIST: &str = "air.ci";

/// The attributes every input node may carry (an output takes only `air.arg_type_name`).
const NAMES: &[Key] = &[Key::ArgTypeName, Key::ArgName];
/// The attributes of `air.ci_texture`.
const TEXTURE: &[Key] = &[
    Key::LocationCount,
    Key::Sample,
    Key::Read,
    Key::Write,
    Key::ReadWrite,
];
/// The attributes of `air.ci_pointer`.
const POINTER: &[Key] = &[
    Key::Read,
    Key::Write,
    Key::ReadWrite,
    Key::ArgTypeSize,
    Key::ArgTypeAlignSize,
];

/// Decode one `!air.ci` entry.
pub(super) fn kernel(m: &Module, id: MdId, meter: &mut Meter) -> Result<CoreImageKernel, Error> {
    meter.tick(1)?;
    let module_site = Site::module();
    let key = Some(LIST);
    let mut n = Ops::node(m, id, module_site.clone())?;
    let func = match n.next(key)? {
        Tok::Func(f) => f,
        _ => return Err(module_site.err(AbiError::BadValue, key)),
    };
    let name = m
        .func(func)
        .and_then(|f| m.symbols().get(f.name))
        .ok_or_else(|| module_site.err(AbiError::BadValue, key))?;
    let name = kept_diag(name, meter)?;
    let site = Site {
        entry: Some(name.clone()),
        arg: None,
    };
    n.site = site.clone();
    let outputs_node = n.child(key)?;
    let inputs_node = n.child(key)?;
    n.finish()?;

    let output = match node_ops(m, outputs_node).ok_or_else(|| site.err(AbiError::BadValue, key))? {
        [] => None,
        [op] => {
            meter.tick(1)?;
            let op = op.ok_or_else(|| site.err(AbiError::BadValue, key))?;
            let mut o = Ops::node(m, op, site.clone())?;
            match key_of(&mut o)? {
                Key::CiBuiltin => {}
                other => return Err(site.err(AbiError::UnknownKey, Some(other.name()))),
            }
            let a = attrs(&mut o, &[&[Key::ArgTypeName]], meter)?;
            Some(CoreImageOutput {
                type_name: a.type_name,
            })
        }
        // A function returns at most one value.
        _ => return Err(site.err(AbiError::SignatureMismatch, key)),
    };

    let input_ids = node_ops(m, inputs_node).ok_or_else(|| site.err(AbiError::BadValue, key))?;
    let mut params = Vec::new();
    meter.reserve(&mut params, input_ids.len())?;
    for &op in input_ids {
        meter.tick(1)?;
        let op = op.ok_or_else(|| site.err(AbiError::BadValue, key))?;
        let mut i = Ops::node(m, op, site.clone())?;
        let param = i.u32(key)?;
        i.site.arg = Some(param);
        let role_key = key_of(&mut i)?;
        let allowed: &[&[Key]] = match role_key {
            Key::CiTexture => &[NAMES, TEXTURE],
            Key::CiPointer => &[NAMES, POINTER],
            Key::CiBuiltin | Key::CiMatrix | Key::CiSampler => &[NAMES],
            other => return Err(i.site.err(AbiError::UnknownKey, Some(other.name()))),
        };
        let a = attrs(&mut i, allowed, meter)?;
        let access = |expected: &'static str| {
            a.access
                .ok_or_else(|| i.site.err(AbiError::MissingKey, Some(expected)))
        };
        let required = |v: Option<u32>, k: Key| {
            v.ok_or_else(|| i.site.err(AbiError::MissingKey, Some(k.name())))
        };
        let role = match role_key {
            Key::CiTexture => CoreImageRole::Texture {
                location_count: required(a.location_count, Key::LocationCount)?,
                access: access(TEXTURE_ACCESS)?,
            },
            // `POINTER` has no `air.sample`, so the access is `air.read`, `air.write`, or
            // `air.read_write`.
            Key::CiPointer => CoreImageRole::Pointer {
                access: access(BUFFER_ACCESS)?,
                size: required(a.size, Key::ArgTypeSize)?,
                align: required(a.align, Key::ArgTypeAlignSize)?,
            },
            Key::CiMatrix => CoreImageRole::Matrix,
            Key::CiSampler => CoreImageRole::Sampler,
            // `air.ci_builtin`: every other key returned above.
            _ => CoreImageRole::Builtin,
        };
        params.push(CoreImageParam {
            param,
            role,
            type_name: a.type_name,
            name: a.name,
        });
    }
    Ok(CoreImageKernel {
        func,
        params,
        output,
        name,
    })
}

/// The attributes of one node, as read.
#[derive(Default)]
struct Attrs {
    type_name: Option<DiagName>,
    name: Option<DiagName>,
    access: Option<Access>,
    location_count: Option<u32>,
    size: Option<u32>,
    align: Option<u32>,
}

/// Read attributes until the node ends. A key outside `allowed` is `UnknownKey`; a repeated key,
/// or a second access key, is `BadValue`.
fn attrs(ops: &mut Ops<'_>, allowed: &[&[Key]], meter: &mut Meter) -> Result<Attrs, Error> {
    let mut a = Attrs::default();
    while !ops.is_done() {
        meter.tick(1)?;
        let k = key_of(ops)?;
        let name = Some(k.name());
        if !allowed.iter().any(|set| set.contains(&k)) {
            return Err(ops.site.err(AbiError::UnknownKey, name));
        }
        let repeated = match k {
            Key::ArgTypeName => a.type_name.replace(ops.diag(name, meter)?).is_some(),
            Key::ArgName => a.name.replace(ops.diag(name, meter)?).is_some(),
            Key::LocationCount => a.location_count.replace(ops.u32(name)?).is_some(),
            Key::ArgTypeSize => a.size.replace(ops.u32(name)?).is_some(),
            Key::ArgTypeAlignSize => a.align.replace(ops.u32(name)?).is_some(),
            Key::Sample => a.access.replace(Access::Sample).is_some(),
            Key::Read => a.access.replace(Access::Read).is_some(),
            Key::Write => a.access.replace(Access::Write).is_some(),
            Key::ReadWrite => a.access.replace(Access::ReadWrite).is_some(),
            _ => return Err(ops.site.err(AbiError::UnknownKey, name)),
        };
        if repeated {
            return Err(ops.site.err(AbiError::BadValue, name));
        }
    }
    Ok(a)
}
