//! The stage's `ArgumentTable` uniform buffer.
//!
//! Push placement (`lower::push`) puts every field from the first one that does not fit the
//! stage's push window into the `ArgumentTable`, the uniform buffer bound at
//! `ResourceSlot{stage, Synthetic(ArgumentTable), 0}`, at offsets fixed by the same order. The
//! access family declares it as one `Uniform` block whose members are exactly the spilled
//! fields, each with the push shape's SPIR-V type at its placed offset (std140: the shapes lay
//! out identically under std140 and std430), and [`read`] loads a member exactly as
//! `EmitCx::push_field` loads a push member, so a family reading a field never knows where it was
//! placed. The table's descriptor counts in the per-stage limits (`access::finish_slots`).
//!
//! Under the identity domains no uniform buffer is declared: the spilled fields are a parameter
//! block in memory at the device address `PushField::ArgumentTableAddress` holds, which the
//! window always places. [`read_line`] loads it 16 bytes at a time, each line once where it
//! dominates (`EmitCx::block_line`), and [`field_of_lines`] puts each field together from its
//! lines' words.

use std::collections::BTreeSet;

use crate::lower::push::{self, FieldShape};
use crate::pipeline::interface::{PushLoc, PushPlacement};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::{BResult, BuildError, Dangling};
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, LayoutRule, MemAccess, MemberLayout, StorageClass,
    StructLayout, Type,
};

/// A declared `ArgumentTable`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Table {
    pub global: GlobalId,
    /// The spilled fields, in member order.
    pub fields: Vec<PushField>,
}

/// Whether `stage`'s placement of `fields` spills into the `ArgumentTable` (`push::spills`).
pub fn spills(stage: Stage, base: u32, fields: &BTreeSet<PushField>) -> bool {
    push::spills(stage, base, fields)
}

/// The binding symbol of `stage`'s `ArgumentTable`.
pub fn symbol(stage: Stage) -> ResourceSlot {
    ResourceSlot {
        stage,
        kind: ResourceKind::Synthetic(SyntheticBinding::ArgumentTable),
        index: 0,
    }
}

/// The SPIR-V type of a push field of shape `sh` (the same type the push block gives it).
pub fn shape_type(mb: &mut ModuleBuilder, sh: FieldShape) -> BResult<TypeId> {
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    match sh {
        FieldShape::U32 => Ok(u32_),
        // Only address fields are 64-bit; they are placed only where device addresses (and
        // `shaderInt64`) are.
        FieldShape::U64 => mb.ty(Type::Int {
            width: 64,
            signed: false,
        }),
        FieldShape::UVec2 => mb.ty(Type::Vector { elem: u32_, len: 2 }),
        FieldShape::UVec4Array(n) => {
            let v4 = mb.ty(Type::Vector { elem: u32_, len: 4 })?;
            let len = mb.constant(Constant::Scalar {
                ty: u32_,
                bits: ConstBits(u64::from(n)),
            })?;
            mb.ty(Type::Array {
                elem: v4,
                len,
                stride: Some(16),
            })
        }
    }
}

/// Declare the `ArgumentTable` of `placement`, or `None` when nothing spilled.
pub fn declare(mb: &mut ModuleBuilder, placement: &PushPlacement) -> BResult<Option<Table>> {
    let spilled: Vec<(PushField, u32)> = placement
        .fields
        .iter()
        .filter_map(|f| match f.loc {
            PushLoc::ArgumentTable { offset } => Some((f.field, offset)),
            PushLoc::Push { .. } => None,
        })
        .collect();
    if spilled.is_empty() {
        return Ok(None);
    }
    let mut members = Vec::with_capacity(spilled.len());
    let mut layout = Vec::with_capacity(spilled.len());
    for (field, offset) in &spilled {
        members.push(shape_type(mb, push::shape(*field))?);
        layout.push(MemberLayout {
            offset: *offset,
            matrix_stride: None,
        });
    }
    let block = mb.ty(Type::Struct {
        members: members.into(),
        layout: Some(StructLayout {
            rule: LayoutRule::Std140,
            block: true,
            non_writable: false,
            members: layout.into(),
        }),
    })?;
    let global = mb.resource(symbol(placement.stage), StorageClass::Uniform, block)?;
    Ok(Some(Table {
        global,
        fields: spilled.into_iter().map(|(f, _)| f).collect(),
    }))
}

/// The bytes of the parameter block one load reads: a line of 16 bytes at a multiple of 16. The
/// block's address is a multiple of 16 (the runtime's ring places every block so) and its size
/// is one (`PushPlacement::argument_table_size`), so every line a field touches lies in the block.
pub const LINE: u32 = 16;

/// The lines of the parameter block that field `field`, placed at byte `offset`, lies in: one for
/// a scalar or a 2-vector (each aligned to its size), `n` for an array of `n` 4-vectors.
pub fn lines(offset: u32, field: PushField) -> std::ops::RangeInclusive<u32> {
    let size = push::shape(field).size().max(1);
    offset / LINE..=(offset + size - 1) / LINE
}

/// Load line `line` of the parameter block at device address `block` (a `u64`): the identity
/// domains' `ArgumentTable`, which no descriptor binds. One `uvec4`, 16-aligned, through a place
/// whose member is `NonWritable`: nothing writes the block while the module runs (its consumer
/// writes it before the command that reads it), so a device may read it as read-only memory.
pub fn read_line(fb: &mut FunctionBuilder<'_>, block: Value, line: u32) -> BResult<Value> {
    let u32_ = fb.module().ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec4 = fb.module().ty(Type::Vector { elem: u32_, len: 4 })?;
    let u64_ = fb.module().ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let offset = fb.module().constant(Constant::Scalar {
        ty: u64_,
        bits: ConstBits(u64::from(line * LINE)),
    })?;
    let offset = fb.module().const_value(offset)?;
    let at = fb.binary(BinOp::IAdd, block, offset)?;
    let place = fb.device_place_read_only(at, uvec4)?;
    fb.load(
        place,
        MemAccess {
            aligned: Some(LINE),
            ..MemAccess::default()
        },
    )
}

/// Field `field`, placed at byte `offset` of the parameter block, from `lines`: each line of the
/// block it lies in ([`lines`]) with that line's words ([`read_line`]).
pub fn field_of_lines(
    fb: &mut FunctionBuilder<'_>,
    lines: &[(u32, Value)],
    offset: u32,
    field: PushField,
) -> BResult<Value> {
    /// The line holding byte `at`.
    fn line_of(lines: &[(u32, Value)], at: u32) -> BResult<Value> {
        lines
            .iter()
            .find(|(line, _)| *line == at / LINE)
            .map(|(_, v)| *v)
            .ok_or(BuildError::Dangling(Dangling::Value(Value::new(u32::MAX))))
    }
    /// The word at byte `at`.
    fn word(fb: &mut FunctionBuilder<'_>, lines: &[(u32, Value)], at: u32) -> BResult<Value> {
        let v = line_of(lines, at)?;
        fb.composite_extract(v, &[(at % LINE) / 4])
    }
    let sh = push::shape(field);
    let ty = shape_type(fb.module(), sh)?;
    match sh {
        FieldShape::U32 => word(fb, lines, offset),
        // 8-aligned: both words in one line.
        FieldShape::UVec2 | FieldShape::U64 => {
            let v = line_of(lines, offset)?;
            let w = (offset % LINE) / 4;
            let pair = fb.vector_shuffle(v, v, &[w, w + 1])?;
            match sh {
                FieldShape::U64 => fb.convert(ConvOp::Bitcast, pair, ty),
                _ => Ok(pair),
            }
        }
        FieldShape::UVec4Array(n) => {
            let parts = (0..n)
                .map(|k| line_of(lines, offset + k * LINE))
                .collect::<BResult<Vec<Value>>>()?;
            fb.composite_construct(ty, &parts)
        }
    }
}

/// Load `field` from `table` in the open block, or `None` when the table does not hold it.
pub fn read(
    fb: &mut FunctionBuilder<'_>,
    table: &Table,
    field: PushField,
) -> BResult<Option<Value>> {
    let Some(i) = table.fields.iter().position(|f| *f == field) else {
        return Ok(None);
    };
    let Ok(i) = u32::try_from(i) else {
        return Ok(None);
    };
    let base = fb.module().global_value(table.global)?;
    let index = fb.u32_const(i)?;
    let index = fb.module().const_value(index)?;
    let ptr = fb.access_chain(base, &[index])?;
    Ok(Some(fb.load(ptr, MemAccess::default())?))
}
