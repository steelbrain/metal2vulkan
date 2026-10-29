//! The module code that declares and searches the tables of the extended-range `_sRGB` formats
//! (`super`'s docs), Metal's own decode steps and encode thresholds, which are the format's data
//! (`metal::format::extended_srgb`), and those of the padded `_sRGB` formats
//! (`metal::format::padded_srgb`).

use crate::metal::format::extended_srgb::{SRGB_DECODE_STEPS, SRGB_ENCODE_FIRST};
use crate::metal::format::padded_srgb::{
    R10X6_SRGB_DECODE_STEPS, R10X6_SRGB_ENCODE_FIRST, RG10X12_SRGB_WRITE_CODES,
};
use crate::metal::format::ExtendedRange;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{BinOp, ConstBits, Constant, MemAccess, StorageClass, Type};

/// The table's length: the offsets, padded to the search's power of two with entries whose
/// first index no encode index reaches.
pub(crate) const TABLE_LEN: u32 = 1024;

/// Declare the module's table of the `_sRGB` formats: a `Private` `u32[1024]` holding each
/// offset's decode step in its low half and its first encode index in its high half, and
/// `0xFFFF` above both past offset 639.
pub fn declare(mb: &mut ModuleBuilder) -> BResult<GlobalId> {
    let words = (0..TABLE_LEN as usize).map(|n| {
        match (SRGB_DECODE_STEPS.get(n), SRGB_ENCODE_FIRST.get(n)) {
            (Some(step), Some(first)) => u64::from(*step) | (u64::from(*first) << 16),
            _ => 0xFFFF_0000,
        }
    });
    declare_words(mb, words)
}

/// Declare the module's table of the padded `_sRGB` form `form`, a `Private` `u32[1024]`
/// indexed by code (`metal::format::padded_srgb`): `R10Unorm_X6_sRGB`'s holds each code's
/// decode step in its low half and its first encode index in its high half, as [`declare`]'s
/// does an offset's, so [`search`] encodes through it (`RGB10A2Unorm_sRGB`'s color channels
/// take the same); `RG10Unorm_X12_sRGB`'s holds the code a
/// write of each linear code stores. `None` for every other form, which has no such table.
pub fn declare_padded(mb: &mut ModuleBuilder, form: ExtendedRange) -> BResult<Option<GlobalId>> {
    let words: Vec<u64> = match form {
        ExtendedRange::R10X6Srgb | ExtendedRange::Rgb10A2Srgb => R10X6_SRGB_DECODE_STEPS
            .iter()
            .zip(&R10X6_SRGB_ENCODE_FIRST)
            .map(|(&step, &first)| u64::from(step) | (u64::from(first) << 16))
            .collect(),
        ExtendedRange::Rg10X12Srgb => RG10X12_SRGB_WRITE_CODES
            .iter()
            .map(|&c| u64::from(c))
            .collect(),
        _ => return Ok(None),
    };
    declare_words(mb, words).map(Some)
}

/// A `Private` `u32[TABLE_LEN]` initialized to `words`, in order.
fn declare_words(
    mb: &mut ModuleBuilder,
    words: impl IntoIterator<Item = u64>,
) -> BResult<GlobalId> {
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let len = mb.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(TABLE_LEN)),
    })?;
    let arr = mb.ty(Type::Array {
        elem: u32_,
        len,
        stride: None,
    })?;
    let mut parts = Vec::with_capacity(TABLE_LEN as usize);
    for word in words {
        parts.push(mb.constant(Constant::Scalar {
            ty: u32_,
            bits: ConstBits(word),
        })?);
    }
    let init = mb.constant(Constant::Composite {
        ty: arr,
        parts: parts.into(),
    })?;
    mb.plain_global(StorageClass::Private, arr, Some(init))
}

/// Word `i` (a `u32` scalar below [`TABLE_LEN`]) of `table`.
pub(crate) fn word(fb: &mut FunctionBuilder<'_>, table: GlobalId, i: Value) -> BResult<Value> {
    let base = fb.module().global_value(table)?;
    let p = fb.access_chain(base, &[i])?;
    fb.load(p, MemAccess::default())
}

/// The offset of encode index `i` (a `u32` scalar at most 16384): the largest offset whose first
/// index is at most `i`, ten dependent loads, branch-free. Over `R10Unorm_X6_sRGB`'s table
/// ([`declare_padded`]), the largest code whose first encode index is at most `i`.
pub fn search(fb: &mut FunctionBuilder<'_>, table: GlobalId, i: Value) -> BResult<Value> {
    let k = |fb: &mut FunctionBuilder<'_>, v: u32| -> BResult<Value> {
        let c = fb.u32_const(v)?;
        fb.module().const_value(c)
    };
    let k16 = k(fb, 16)?;
    let mut n = k(fb, 0)?;
    let mut step = TABLE_LEN / 2;
    while step > 0 {
        let s = k(fb, step)?;
        let j = fb.binary(BinOp::IAdd, n, s)?;
        let w = word(fb, table, j)?;
        let first = fb.binary(BinOp::ShiftRightLogical, w, k16)?;
        let at_least = fb.binary(BinOp::UGreaterThanEqual, i, first)?;
        n = fb.select(at_least, j, n)?;
        step /= 2;
    }
    Ok(n)
}
