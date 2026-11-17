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
//! window always places, and [`read_block`] loads each at its placed offset.

use std::collections::BTreeSet;

use crate::device::profile::PushWindow;
use crate::lower::push::{self, FieldShape};
use crate::pipeline::interface::{PushLoc, PushPlacement};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, LayoutRule, MemAccess, MemberLayout, StorageClass, StructLayout,
    Type,
};

/// A declared `ArgumentTable`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Table {
    pub global: GlobalId,
    /// The spilled fields, in member order.
    pub fields: Vec<PushField>,
}

/// Whether `stage`'s placement of `fields` in push window `w` spills into the `ArgumentTable`
/// (`push::spills`).
pub fn spills(stage: Stage, w: PushWindow, fields: &BTreeSet<PushField>) -> bool {
    push::spills(stage, w, fields)
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

/// Load `field` from the parameter block at device address `block` (a `u64`), where it is placed
/// at byte `offset`: the identity domains' `ArgumentTable`, which no descriptor binds.
pub fn read_block(
    fb: &mut FunctionBuilder<'_>,
    block: Value,
    offset: u32,
    field: PushField,
) -> BResult<Value> {
    let sh = push::shape(field);
    let ty = shape_type(fb.module(), sh)?;
    let u64_ = fb.module().ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let offset = fb.module().constant(Constant::Scalar {
        ty: u64_,
        bits: ConstBits(u64::from(offset)),
    })?;
    let offset = fb.module().const_value(offset)?;
    let at = fb.binary(BinOp::IAdd, block, offset)?;
    let place = fb.device_place(at, ty)?;
    fb.load(
        place,
        MemAccess {
            aligned: Some(sh.align()),
            ..MemAccess::default()
        },
    )
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
