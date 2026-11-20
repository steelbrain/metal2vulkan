//! A patch's tessellation factors read from Metal's factor buffer, as the control stage
//! (`pipeline::link::tessellation::control`, whose module docs state the addressing) reads them.
//! The binder sizes a draw's factor range by [`factor_bytes`], so this file is a runtime input
//! (`validation/build.rs`).

use crate::pipeline::interface::PatchDomain;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, Value};
use crate::spirv::ir::{BinOp, GlslOp, MemAccess};

/// Bytes of one patch's factors (`tessellationFactorStride`): Metal's
/// `MTLTriangleTessellationFactorsHalf` or `MTLQuadTessellationFactorsHalf`.
pub fn factor_bytes(domain: PatchDomain) -> u32 {
    match domain {
        PatchDomain::Triangle => 8,
        PatchDomain::Quad => 12,
    }
}

fn ku(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

/// Word `index` of `buffer` (a runtime array of `uint`: the factor, patch index or control-point
/// index buffer), or 0 when the bound range does not hold it.
pub fn word(fb: &mut FunctionBuilder<'_>, buffer: GlobalId, index: Value) -> BResult<Value> {
    let base = fb.module().global_value(buffer)?;
    let len = fb.array_length(base, 0)?;
    let inside = fb.binary(BinOp::ULessThan, index, len)?;
    // The binder binds at least 16 bytes, so `len - 1` is a word of the range.
    let one = ku(fb, 1)?;
    let last = fb.binary(BinOp::ISub, len, one)?;
    let at = fb.glsl(GlslOp::UMin, &[index, last])?;
    let zero = ku(fb, 0)?;
    let ptr = fb.access_chain(base, &[zero, at])?;
    let w = fb.load(ptr, MemAccess::default())?;
    fb.select(inside, w, zero)
}

/// The `count` factors' `half` bits (each a `uint`) from the even byte `address` of the factor
/// buffer `factors`: the words from the one that holds it, unpacked low half first, from that
/// word's high half where `address` is 2 past a multiple of 4; a word outside the bound range
/// reads 0.
pub fn factor_halves(
    fb: &mut FunctionBuilder<'_>,
    factors: GlobalId,
    address: Value,
    count: usize,
) -> BResult<Vec<Value>> {
    let two = ku(fb, 2)?;
    let first = fb.binary(BinOp::ShiftRightLogical, address, two)?;
    let mask = ku(fb, 0xffff)?;
    let sixteen = ku(fb, 16)?;
    let mut unpacked = Vec::with_capacity(count + 2);
    for k in 0..(count + 1).div_ceil(2) {
        let offset = ku(fb, k as u32)?;
        let index = fb.binary(BinOp::IAdd, first, offset)?;
        let w = word(fb, factors, index)?;
        unpacked.push(fb.binary(BinOp::BitwiseAnd, w, mask)?);
        unpacked.push(fb.binary(BinOp::ShiftRightLogical, w, sixteen)?);
    }
    let high = fb.binary(BinOp::BitwiseAnd, address, two)?;
    let zero = ku(fb, 0)?;
    let shifted = fb.binary(BinOp::INotEqual, high, zero)?;
    let mut halves = Vec::with_capacity(count);
    for (&low, &high) in unpacked.iter().zip(unpacked.iter().skip(1)).take(count) {
        halves.push(fb.select(shifted, high, low)?);
    }
    Ok(halves)
}
