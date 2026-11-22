//! What a merged draw's vertex and fragment variants read (`merged`): the header of a merged draw,
//! its commands' `VkDrawIndirectCommand`s (`merged::DRAW_ARGUMENT_BYTES`) and index entries
//! (`merged::DRAW_INDEX_ENTRY_BYTES`), and the SPIR-V helpers its reads are built from, reaching memory
//! by device address (`bufferDeviceAddress`, `shaderInt64`).

use crate::memory::roots;
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{BinOp, ConvOp, MemAccess};

mod merged;
pub(crate) use merged::{merged_fragment_table, merged_vertex, MergedVertex};

// ---- The module ----

/// The integer types the module uses.
#[derive(Clone, Copy)]
struct Types {
    u32_: TypeId,
    u64_: TypeId,
}

fn k32(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

fn k64(fb: &mut FunctionBuilder<'_>, v: u64) -> BResult<Value> {
    roots::k64(fb, v)
}

/// `base + offset` (`u64`s).
fn at(fb: &mut FunctionBuilder<'_>, base: Value, offset: u64) -> BResult<Value> {
    if offset == 0 {
        return Ok(base);
    }
    let d = k64(fb, offset)?;
    fb.binary(BinOp::IAdd, base, d)
}

fn aligned(n: u32) -> MemAccess {
    MemAccess {
        aligned: Some(n),
        ..MemAccess::default()
    }
}

/// The `u32` at device address `addr` (4-aligned).
fn load32(fb: &mut FunctionBuilder<'_>, t: Types, addr: Value) -> BResult<Value> {
    let p = fb.device_place(addr, t.u32_)?;
    fb.load(p, aligned(4))
}

/// The `u64` at device address `addr` (8-aligned).
fn load64(fb: &mut FunctionBuilder<'_>, t: Types, addr: Value) -> BResult<Value> {
    let p = fb.device_place(addr, t.u64_)?;
    fb.load(p, aligned(8))
}

/// `x` (a `u32`) as a `u64`.
fn wide(fb: &mut FunctionBuilder<'_>, t: Types, x: Value) -> BResult<Value> {
    fb.convert(ConvOp::UConvert, x, t.u64_)
}

/// A `u32` function variable initialized to `init`: its pointer.
fn counter(fb: &mut FunctionBuilder<'_>, t: Types, init: u32) -> BResult<Value> {
    let c = fb.u32_const(init)?;
    let var = fb.local_var(t.u32_, Some(c))?;
    fb.var_ptr(var)
}
