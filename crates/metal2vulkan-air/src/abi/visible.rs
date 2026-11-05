//! `!air.visible`: the functions a module exports to the Metal linker.
//!
//! Each list entry is `!{ptr @fn, !outputs, !inputs[, !stitching]}`:
//!
//! - `!outputs` holds at most one `!{!"air.visible_output", <attr>*}` node, the return value;
//! - `!inputs` holds one node per LLVM parameter, `!{i32 <param>, !"air.visible_input",
//!   <attr>*}` or `!{i32 <param>, !"air.visible_padding"}`;
//! - the attributes are `air.arg_type_name` and `air.arg_name`, each at most once;
//! - `!stitching` (a `[[stitchable]]` function) is `!{!"air.stitching_info", <result>,
//!   <argument>*}`, where the result is `null` (a `void` function) or a stitching type, each
//!   argument is `!{!"air.stitching_argument", <stitching type>[, !"<name>"]}`, and a stitching
//!   type is `!{!"air.stitching_type", !<descriptor>, <helper>, <helper>, <helper>, <helper>}`
//!   with each helper a function or `null`.
//!
//! The recursive MSL type descriptor of a stitching type is only required to be a node: it
//! describes the value for a stitching graph, which no translation reads (the helper functions
//! are ordinary functions of the module). Everything else is strict: an unknown key, a missing
//! operand, or a repeated attribute is an [`AbiError`].

use super::keys::Key;
use super::md::{kept_diag, node_ops, Ops, Site, Tok};
use super::{
    Stitching, StitchingArgument, StitchingType, VisibleFunction, VisibleOutput, VisibleParam,
    VisibleParamKind,
};
use crate::budget::Meter;
use crate::error::Error;
use crate::error::{AbiError, DiagName};
use crate::ir::{MdId, Module};

/// Decode one `!air.visible` entry.
pub(super) fn function(m: &Module, id: MdId, meter: &mut Meter) -> Result<VisibleFunction, Error> {
    meter.tick(1)?;
    let module_site = Site::module();
    let key = Some("air.visible");
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
    let stitching = if n.is_done() {
        None
    } else {
        Some(stitching(m, n.child(key)?, &site, meter)?)
    };
    n.finish()?;

    let output_ids = node_ops(m, outputs_node).ok_or_else(|| site.err(AbiError::BadValue, key))?;
    let mut outputs = Vec::new();
    meter.reserve(&mut outputs, output_ids.len())?;
    for &op in output_ids {
        meter.tick(1)?;
        let op = op.ok_or_else(|| site.err(AbiError::BadValue, key))?;
        let mut o = Ops::node(m, op, site.clone())?;
        if role(&mut o)? != Key::VisibleOutput {
            return Err(site.err(AbiError::UnknownKey, key));
        }
        let (type_name, arg_name) = attrs(&mut o, meter)?;
        if arg_name.is_some() {
            return Err(site.err(AbiError::UnknownKey, Some(Key::ArgName.name())));
        }
        outputs.push(VisibleOutput { type_name });
    }

    let input_ids = node_ops(m, inputs_node).ok_or_else(|| site.err(AbiError::BadValue, key))?;
    let mut params = Vec::new();
    meter.reserve(&mut params, input_ids.len())?;
    for &op in input_ids {
        meter.tick(1)?;
        let op = op.ok_or_else(|| site.err(AbiError::BadValue, key))?;
        let mut i = Ops::node(m, op, site.clone())?;
        let param = i.u32(key)?;
        i.site.arg = Some(param);
        let kind = match role(&mut i)? {
            Key::VisibleInput => {
                let (type_name, name) = attrs(&mut i, meter)?;
                VisibleParamKind::Input { type_name, name }
            }
            Key::VisiblePadding => {
                i.finish()?;
                VisibleParamKind::Padding
            }
            other => return Err(i.site.err(AbiError::UnknownKey, Some(other.name()))),
        };
        params.push(VisibleParam { param, kind });
    }
    Ok(VisibleFunction {
        func,
        params,
        outputs,
        stitching,
        name,
    })
}

/// The role key of a visible input or output node.
fn role(ops: &mut Ops<'_>) -> Result<Key, Error> {
    let s = ops.string(None)?;
    Key::parse(s).ok_or_else(|| ops.site.err(AbiError::UnknownKey, None))
}

/// `air.arg_type_name` and `air.arg_name`, each at most once, until the node ends.
fn attrs(
    ops: &mut Ops<'_>,
    meter: &mut Meter,
) -> Result<(Option<DiagName>, Option<DiagName>), Error> {
    let (mut type_name, mut name) = (None, None);
    while !ops.is_done() {
        meter.tick(1)?;
        let k = role(ops)?;
        let slot = match k {
            Key::ArgTypeName => &mut type_name,
            Key::ArgName => &mut name,
            other => return Err(ops.site.err(AbiError::UnknownKey, Some(other.name()))),
        };
        if slot.is_some() {
            return Err(ops.site.err(AbiError::BadValue, Some(k.name())));
        }
        *slot = Some(ops.diag(Some(k.name()), meter)?);
    }
    Ok((type_name, name))
}

const STITCHING_INFO: &str = "air.stitching_info";
const STITCHING_ARGUMENT: &str = "air.stitching_argument";
const STITCHING_TYPE: &str = "air.stitching_type";

/// `!{!"air.stitching_info", null | <stitching type>, <argument>*}`.
fn stitching(m: &Module, id: MdId, site: &Site, meter: &mut Meter) -> Result<Stitching, Error> {
    let key = Some(STITCHING_INFO);
    let mut n = Ops::node(m, id, site.clone())?;
    tag(&mut n, STITCHING_INFO)?;
    let result = match n.next(key)? {
        Tok::Null => None,
        Tok::Node(ty) => Some(stitching_type(m, ty, site)?),
        _ => return Err(site.err(AbiError::BadValue, key)),
    };
    let mut arguments = Vec::new();
    meter.reserve(&mut arguments, n.len().saturating_sub(2))?;
    while !n.is_done() {
        meter.tick(1)?;
        let arg_key = Some(STITCHING_ARGUMENT);
        let mut a = Ops::node(m, n.child(key)?, site.clone())?;
        tag(&mut a, STITCHING_ARGUMENT)?;
        let ty = stitching_type(m, a.child(arg_key)?, site)?;
        let name = if a.is_done() {
            None
        } else {
            Some(a.diag(arg_key, meter)?)
        };
        a.finish()?;
        arguments.push(StitchingArgument { ty, name });
    }
    Ok(Stitching { result, arguments })
}

/// `!{!"air.stitching_type", !<descriptor>, <helper> x 4}`, each helper a function or `null`.
fn stitching_type(m: &Module, id: MdId, site: &Site) -> Result<StitchingType, Error> {
    let key = Some(STITCHING_TYPE);
    let mut n = Ops::node(m, id, site.clone())?;
    tag(&mut n, STITCHING_TYPE)?;
    // The MSL type descriptor: a node, not decoded (module documentation).
    n.child(key)?;
    let mut helpers = [None; 4];
    for helper in &mut helpers {
        *helper = match n.next(key)? {
            Tok::Func(f) => Some(f),
            Tok::Null => None,
            _ => return Err(site.err(AbiError::BadValue, key)),
        };
    }
    n.finish()?;
    Ok(StitchingType { helpers })
}

/// The node's leading tag string must be `want`.
fn tag(n: &mut Ops<'_>, want: &'static str) -> Result<(), Error> {
    if n.string(Some(want))? == want.as_bytes() {
        Ok(())
    } else {
        Err(n.site.err(AbiError::UnknownKey, Some(want)))
    }
}
