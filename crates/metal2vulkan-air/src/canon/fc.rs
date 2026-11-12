//! Function-constant binding.
//!
//! Each supplied value is checked against its declaration (the index must be declared; the
//! `MTLDataType` must name the initializer global's value type, and the bytes must be that type's
//! packed size) and becomes the initializer of the constant's global, which is then an ordinary
//! constant global that SCCP can load from. An unsupplied constant keeps its `undef` initializer
//! and stays externally initialized: it is **undefined**, and SCCP tracks every use.

use std::collections::{BTreeMap, HashMap};

use crate::abi::{self, AirAbi};
use crate::budget::Meter;
use crate::error::{Error, RequestError};
use crate::ir::{Arena, Const, ConstId, ConstTable, Constant, Global, GlobalId, Module};
use crate::ir::{Type, TypeId, TypeTable};
use crate::metal::DataType;

use super::sccp::{FcState, Undefined};
use super::ConstValue;

/// A supplied value checked against its declaration, not yet interned.
pub(super) struct Pending<'r> {
    pub index: u32,
    pub global: GlobalId,
    pub ty: TypeId,
    pub value: &'r ConstValue,
}

fn type_error(index: u32) -> Error {
    Error::InvalidRequest(RequestError::FunctionConstantType { index })
}

/// Check `values` against the declarations of `abi`: every index is declared and every
/// `MTLDataType` names its constant's stored type. Also returns the state SCCP needs for every
/// declared constant.
pub(super) fn check<'r>(
    module: &Module,
    abi: &AirAbi,
    values: &'r BTreeMap<u32, ConstValue>,
    m: &mut Meter,
) -> Result<(Vec<Pending<'r>>, HashMap<GlobalId, FcState>), Error> {
    let mut states = HashMap::new();
    for d in &abi.function_constants {
        m.tick(1)?;
        m.charge(std::mem::size_of::<(GlobalId, FcState)>() * 2)?;
        states.insert(
            d.global,
            FcState {
                index: Undefined::Constant(d.index),
                supplied: values.contains_key(&d.index),
                decl: true,
            },
        );
    }
    let mut pending = Vec::new();
    m.reserve(&mut pending, values.len())?;
    for (&index, value) in values {
        m.tick(1)?;
        let decl = abi
            .function_constants
            .iter()
            .find(|d| d.index == index)
            .ok_or(Error::InvalidRequest(RequestError::FunctionConstantIndex {
                index,
            }))?;
        if decl.data_type.is_some_and(|t| t != value.ty)
            || !abi::data_type_is(module, value.ty, decl.ty)
        {
            return Err(type_error(index));
        }
        pending.push(Pending {
            index,
            global: decl.global,
            ty: decl.ty,
            value,
        });
    }
    Ok((pending, states))
}

/// Intern each checked value and make it its global's initializer: the global becomes an
/// ordinary constant global. A byte count that does not match the type is a type error.
pub(super) fn bind(
    types: &TypeTable,
    consts: &mut ConstTable,
    globals: &mut Arena<GlobalId, Global>,
    pending: &[Pending<'_>],
    m: &mut Meter,
) -> Result<(), Error> {
    let mut values = Vec::new();
    m.reserve(&mut values, pending.len())?;
    for p in pending {
        let c = constant(types, consts, p.ty, p.value, m)?.ok_or(type_error(p.index))?;
        values.push((p.global, c));
    }
    for (g, c) in values {
        if let Some(g) = globals.get_mut(g) {
            g.init = Some(c);
            g.constant = true;
            g.externally_initialized = false;
        }
    }
    Ok(())
}

fn is_bool(ty: DataType) -> bool {
    matches!(
        ty,
        DataType::Bool | DataType::Bool2 | DataType::Bool3 | DataType::Bool4
    )
}

/// The constant of IR type `ty` (a scalar or vector the `MTLDataType` was checked against) whose
/// packed little-endian bytes are `value.bytes`; `None` when the byte count does not match. A
/// `bool` byte is normalized to 0 or 1, as MSL's `bool` stores it.
fn constant(
    types: &TypeTable,
    consts: &mut ConstTable,
    ty: TypeId,
    value: &ConstValue,
    m: &mut Meter,
) -> Result<Option<ConstId>, Error> {
    let (elem, lanes) = match types.get(ty) {
        Some(Type::Vector { elem, len }) => (*elem, Some(*len as usize)),
        _ => (ty, None),
    };
    let (width, float) = match types.get(elem) {
        Some(Type::Int(w)) => (usize::from(*w).div_ceil(8), false),
        Some(t) => match t.float_bits() {
            Some(bits) => (bits as usize / 8, true),
            None => return Ok(None),
        },
        None => return Ok(None),
    };
    let n = lanes.unwrap_or(1);
    if width == 0 || width > 8 || value.bytes.len() != width.saturating_mul(n) {
        return Ok(None);
    }
    let mut elems = Vec::new();
    m.reserve(&mut elems, n)?;
    for chunk in value.bytes.chunks_exact(width) {
        let mut word = [0u8; 8];
        if let Some(dst) = word.get_mut(..width) {
            dst.copy_from_slice(chunk);
        }
        let mut bits = u64::from_le_bytes(word);
        if is_bool(value.ty) {
            bits = u64::from(bits != 0);
        }
        let v = if float {
            Const::Float(bits)
        } else {
            Const::Int(bits)
        };
        elems.push(consts.intern(Constant { ty: elem, value: v }, m)?);
    }
    if lanes.is_none() {
        return Ok(elems.first().copied());
    }
    consts
        .intern(
            Constant {
                ty,
                value: Const::Aggregate(elems.into_boxed_slice()),
            },
            m,
        )
        .map(Some)
}
