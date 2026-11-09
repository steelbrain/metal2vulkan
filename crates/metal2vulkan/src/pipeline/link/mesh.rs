//! The synthesized vertex stage of an emulated mesh function's pipeline.
//!
//! On a device without `VK_EXT_mesh_shader` a mesh function is a compute entry that writes each
//! of its threadgroups' vertices and primitives to the draw's records
//! ([`MeshRecords`], `SyntheticBinding::MeshRecords`; `lower::mesh` writes them). The pipeline
//! draws them through a vertex stage link adds after the mesh stage ([`vertex_module`]): the
//! draw is a non-indexed draw of `T · NP · k` vertices (`T` the draw's mesh threadgroups, `NP`
//! the mesh type's primitives, `k` a primitive's indices) as a list of the mesh's topology
//! ([`MeshRecords::topology`]), and vertex `v` is corner `c` of slot `s` of threadgroup `g`
//! (`v = (g · NP + s) · k + c`), whose records begin at the draw's first word (word
//! `RECORDS_BASE` of its work buffer, `SyntheticBinding::MeshWork`) plus `g` threadgroups'
//! words. It reads the slot's primitive record and the record of the vertex the corner's index
//! names, and writes:
//!
//! - `Position` from the vertex's position, and `PointSize` (a point mesh's, or a mesh's that
//!   writes it) from its point size, 1.0 where it writes none;
//! - `ClipDistance` from its clip distances, and `Layer` and `ViewportIndex` from the
//!   primitive's;
//! - each user varying from its record, the vertex's or, for primitive data and the raw indices,
//!   the primitive's at every corner: the fragment function reads those flat
//!   (`StageIoPlan::flat_primitive_inputs`), so whichever corner provokes, the primitive's value
//!   is read.
//!
//! A slot at or past the threadgroup's count, a slot whose primitive is culled, and a slot with
//! an index at or past `NV` draw nothing: every corner lies at [`FAR`], wholly outside the clip
//! volume, and reads vertex 0's record instead (Vulkan defines no index past the mesh's vertices
//! either). Vulkan's primitive order for the draw is the vertices' order: threadgroup by
//! threadgroup in their linear order, slot by slot, which is the order Metal draws a mesh draw's
//! primitives in.
//!
//! Under `VariantKey::mesh_last_first` (a draw under Apple's last provoking vertex on a device
//! that takes flat inputs from the first) vertex `v` reads corner `(c + k - 1) mod k` instead:
//! each primitive is drawn from Metal's last index first, a triangle `(i0, i1, i2)` as
//! `(i2, i0, i1)`, which keeps its winding, and a line `(i0, i1)` as `(i1, i0)`, so its flat
//! inputs are its last vertex's, as Metal's are at mode 1 (Metal probe: a mesh triangle's flat
//! input from its last index).
//!
//! The records do not hold the primitive id: a fragment function that reads `[[primitive_id]]`
//! behind an emulated mesh function is not implemented ([`records`]), since the vertex stage's
//! primitives are counted across the draw.

use crate::budget::Meter;
use crate::device::requirements::Requirements;
use crate::device::DeviceProfile;
use crate::error::{Error, Feature, InternalError, InternalKind, RuntimeInput, Unsupported};
pub use crate::lower::mesh::{RECORDS_SLOT, WORK_SLOT};
use crate::pipeline::artifact::{Reconvergence, SpecializationLayout, StageModule, StrategyCensus};
use crate::pipeline::interface::{MeshRecords, RecordField, RecordValue, StageInterface};
use crate::pipeline::stage_io::{IoNumeric, StageBuiltin};
use crate::pipeline::sym::Stage;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, ConvOp, ExecModel, IoDecor, IoDir, LayoutRule,
    MemAccess, StorageClass, Type, TypedModule,
};
use crate::spirv::serialize::{SymbolTable, ENTRY_NAME};
use crate::spv::BuiltIn as B;
use crate::vk;

/// Where a corner that draws nothing lies, in clip coordinates of `w` 1: `x` and `y` this far
/// past the clip volume, so every primitive of such corners lies wholly outside it, a point or a
/// wide line included.
pub const FAR: f32 = 65536.0;

/// The payload check of the mesh render pipeline of stages `ifaces`: its mesh function's payload
/// arena holds at least its object function's (`StageInterface::payload`,
/// `LoweringInputs::object_payload`), else `NeedsInput(ObjectPayload)`.
pub fn payload(ifaces: &[&StageInterface]) -> Result<(), Error> {
    let of = |stage: Stage| {
        ifaces
            .iter()
            .find(|i| i.stage == stage)
            .and_then(|i| i.payload)
    };
    match (of(Stage::Object), of(Stage::Mesh)) {
        (Some(object), Some(mesh)) if mesh < object => {
            Err(Error::NeedsInput(RuntimeInput::ObjectPayload))
        }
        _ => Ok(()),
    }
}

/// The records of the mesh render pipeline of stages `ifaces` whose mesh function is emulated
/// (`StageInterface::mesh_records`); `None` where its mesh function is not. A fragment function
/// that reads `[[primitive_id]]` behind an emulated mesh function is
/// `NotImplemented(MeshShading)` (the module docs).
pub fn records(ifaces: &[&StageInterface]) -> Result<Option<MeshRecords>, Error> {
    let Some(r) = ifaces.iter().find_map(|i| i.mesh_records.as_ref()) else {
        return Ok(None);
    };
    let reads_id = ifaces.iter().any(|i| {
        i.stage == Stage::Fragment
            && (i.io.builtins.contains(&StageBuiltin::PrimitiveId)
                || i.io
                    .inputs
                    .iter()
                    .any(|v| v.role == crate::pipeline::stage_io::IoRole::PrimitiveId))
    });
    if reads_id {
        return Err(Unsupported::not_implemented(Feature::MeshShading).into());
    }
    // An object threadgroup's slot holds the larger of the object and mesh functions' payloads
    // (Metal links payloads of different sizes).
    let payload = ifaces
        .iter()
        .filter_map(|i| i.emulated_payload)
        .max()
        .unwrap_or(0);
    let slot_words = crate::lower::mesh::work::GRID
        .checked_add(payload)
        .ok_or_else(|| {
            Error::Internal(InternalError::new(
                InternalKind::Invariant,
                "mesh records: a payload past the slot's word range".to_owned(),
            ))
        })?;
    Ok(Some(MeshRecords {
        slot_words,
        ..r.clone()
    }))
}

/// The synthesized vertex stage of records `r`, each primitive drawn from its last index first
/// where `last_first`, as a final module under link table `table`, with its module's
/// requirements: verified by the owned verifier, serialized, and relocated, as a compiled
/// function's module is.
pub fn synthesized_stage(
    r: &MeshRecords,
    last_first: bool,
    table: &impl SymbolTable,
    p: &DeviceProfile,
    m: &mut Meter,
) -> Result<(StageModule, Requirements), Error> {
    let module = vertex_module(r, last_first)?;
    let stats = crate::spirv::verify::verify(&module, p, m)?;
    let module = crate::spirv::serialize::serialize_for(module, p)?;
    let words = crate::spirv::serialize::relocate(&module, table)?;
    Ok((
        StageModule {
            stage: vk::ShaderStage::VERTEX,
            entry_point: ENTRY_NAME.to_owned(),
            spirv: words.into(),
            specialization: SpecializationLayout::default(),
            required_subgroup_size: None,
            stage_flags: vk::PipelineShaderStageCreateFlags::empty(),
            reconvergence: Reconvergence::Guaranteed,
            precision_notes: Vec::new(),
            strategies: StrategyCensus::default(),
            constant_blob: None,
            rounded_colors: Vec::new(),
            specialized_notes: Vec::new(),
            own_pixel_textures: Vec::new(),
            snapshot_textures: Vec::new(),
            snapshot_refusals: Vec::new(),
            level_reads: Vec::new(),
            private_bytes: 0,
            stats,
        },
        module.requirements,
    ))
}

fn ku(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

fn kf(fb: &mut FunctionBuilder<'_>, v: f32) -> BResult<Value> {
    let ty = fb.module().ty(Type::Float { width: 32 })?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(v.to_bits())),
    })?;
    fb.module().const_value(c)
}

/// Word `at` of the records.
fn word(fb: &mut FunctionBuilder<'_>, records: GlobalId, at: Value) -> BResult<Value> {
    let zero = ku(fb, 0)?;
    let base = fb.module().global_value(records)?;
    let ptr = fb.access_chain(base, &[zero, at])?;
    fb.load(ptr, MemAccess::default())
}

/// The scalar type of `numeric`.
fn scalar(mb: &mut ModuleBuilder, numeric: IoNumeric) -> BResult<TypeId> {
    match numeric {
        IoNumeric::Float => mb.ty(Type::Float { width: 32 }),
        IoNumeric::Sint => mb.ty(Type::Int {
            width: 32,
            signed: true,
        }),
        IoNumeric::Uint => mb.ty(Type::Int {
            width: 32,
            signed: false,
        }),
    }
}

/// The lanes of field `f` of the record at word `record`, each of its numeric type.
fn lanes(
    fb: &mut FunctionBuilder<'_>,
    records: GlobalId,
    record: Value,
    f: &RecordField,
) -> BResult<Vec<Value>> {
    let ty = scalar(fb.module(), f.numeric)?;
    let mut out = Vec::with_capacity(f.words as usize);
    for j in 0..f.words {
        let off = ku(fb, f.offset + j)?;
        let at = fb.binary(BinOp::IAdd, record, off)?;
        let w = word(fb, records, at)?;
        out.push(match f.numeric {
            IoNumeric::Uint => w,
            IoNumeric::Float | IoNumeric::Sint => fb.convert(ConvOp::Bitcast, w, ty)?,
        });
    }
    Ok(out)
}

/// What one field becomes: the output it is stored to and that output's type.
struct Out<'r> {
    field: &'r RecordField,
    global: GlobalId,
    ty: TypeId,
}

/// The synthesized vertex stage of records `r`, each primitive drawn from its last index first
/// where `last_first` (the module docs).
pub fn vertex_module(r: &MeshRecords, last_first: bool) -> BResult<TypedModule> {
    let mut mb = ModuleBuilder::new();
    let u32_ = scalar(&mut mb, IoNumeric::Uint)?;
    let f32_ = scalar(&mut mb, IoNumeric::Float)?;
    let vec4 = mb.ty(Type::Vector { elem: f32_, len: 4 })?;
    let void = mb.ty(Type::Void)?;
    let words = mb.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(mb.types(), LayoutRule::Std430, true, &[words])?;
    let block = mb.ty(Type::Struct {
        members: [words].into(),
        layout: Some(layout),
    })?;
    let records = mb.resource(RECORDS_SLOT, StorageClass::StorageBuffer, block)?;
    let work = mb.resource(WORK_SLOT, StorageClass::StorageBuffer, block)?;
    let vertex_index = mb.builtin(B::VertexIndex, IoDir::Input, u32_)?;
    let position = mb.builtin(B::Position, IoDir::Output, vec4)?;
    let find = |value: RecordValue| r.fields.iter().find(|f| f.value == value);
    let position_field = find(RecordValue::Position);
    let culled_field = find(RecordValue::Culled);
    let size_field = find(RecordValue::PointSize);
    // A point list needs its point sizes; a mesh that writes them has them in any topology.
    let point_size = if r.corners == 1 || size_field.is_some() {
        Some(mb.builtin(B::PointSize, IoDir::Output, f32_)?)
    } else {
        None
    };
    let mut outs = Vec::new();
    for f in &r.fields {
        let (global, ty) = match &f.value {
            RecordValue::Position | RecordValue::PointSize | RecordValue::Culled => continue,
            RecordValue::ClipDistance => {
                let len = mb.constant(Constant::Scalar {
                    ty: u32_,
                    bits: ConstBits(u64::from(f.words)),
                })?;
                let ty = mb.ty(Type::Array {
                    elem: f32_,
                    len,
                    stride: None,
                })?;
                (mb.builtin(B::ClipDistance, IoDir::Output, ty)?, ty)
            }
            RecordValue::Layer => (mb.builtin(B::Layer, IoDir::Output, u32_)?, u32_),
            RecordValue::ViewportIndex => {
                (mb.builtin(B::ViewportIndex, IoDir::Output, u32_)?, u32_)
            }
            RecordValue::Varying(key) => {
                let elem = scalar(&mut mb, f.numeric)?;
                let ty = match f.words {
                    1 => elem,
                    n => mb.ty(Type::Vector { elem, len: n as u8 })?,
                };
                (
                    mb.io(key.clone(), IoDir::Output, ty, IoDecor::default())?,
                    ty,
                )
            }
        };
        outs.push(Out {
            field: f,
            global,
            ty,
        });
    }
    let mut fb = mb.function(void)?;
    let p = fb.module().global_value(vertex_index)?;
    let v = fb.load(p, MemAccess::default())?;
    // `v = (g · NP + s) · k + c`.
    let per_group = ku(&mut fb, r.primitives.saturating_mul(r.corners))?;
    let k = ku(&mut fb, r.corners)?;
    let g = fb.binary(BinOp::UDiv, v, per_group)?;
    let rem = fb.binary(BinOp::UMod, v, per_group)?;
    let slot = fb.binary(BinOp::UDiv, rem, k)?;
    let mut corner = fb.binary(BinOp::UMod, rem, k)?;
    // Drawn from the last index first: corner `(c + k - 1) mod k`.
    if last_first && r.corners > 1 {
        let back = ku(&mut fb, r.corners - 1)?;
        let moved = fb.binary(BinOp::IAdd, corner, back)?;
        corner = fb.binary(BinOp::UMod, moved, k)?;
    }
    // The threadgroup's records, from the draw's first word: within a `u32`, since the binder
    // binds them whole.
    let at = ku(&mut fb, crate::lower::mesh::work::RECORDS_BASE)?;
    let base = word(&mut fb, work, at)?;
    let group_words = ku(&mut fb, r.group_words() as u32)?;
    let mine = fb.binary(BinOp::IMul, g, group_words)?;
    let first = fb.binary(BinOp::IAdd, base, mine)?;
    let count = word(&mut fb, records, first)?;
    let primitive_base = ku(&mut fb, r.primitive_base() as u32)?;
    let stride = ku(&mut fb, r.primitive_words)?;
    let into = fb.binary(BinOp::IMul, slot, stride)?;
    let primitives = fb.binary(BinOp::IAdd, first, primitive_base)?;
    let primitive = fb.binary(BinOp::IAdd, primitives, into)?;
    let at = fb.binary(BinOp::IAdd, primitive, corner)?;
    let index = word(&mut fb, records, at)?;
    let below = fb.binary(BinOp::ULessThan, slot, count)?;
    let nv = ku(&mut fb, r.vertices)?;
    let named = fb.binary(BinOp::ULessThan, index, nv)?;
    let mut drawn = fb.binary(BinOp::LogicalAnd, below, named)?;
    if let Some(f) = culled_field {
        let off = ku(&mut fb, f.offset)?;
        let at = fb.binary(BinOp::IAdd, primitive, off)?;
        let culled = word(&mut fb, records, at)?;
        let zero = ku(&mut fb, 0)?;
        let kept = fb.binary(BinOp::IEqual, culled, zero)?;
        drawn = fb.binary(BinOp::LogicalAnd, drawn, kept)?;
    }
    let zero = ku(&mut fb, 0)?;
    let index = fb.select(drawn, index, zero)?;
    let one = ku(&mut fb, 1)?;
    let vertex_stride = ku(&mut fb, r.vertex_words)?;
    let into = fb.binary(BinOp::IMul, index, vertex_stride)?;
    let vertices = fb.binary(BinOp::IAdd, first, one)?;
    let vertex = fb.binary(BinOp::IAdd, vertices, into)?;
    // The position, or `FAR` where the corner draws nothing.
    let written = match position_field {
        Some(f) => lanes(&mut fb, records, vertex, f)?,
        None => vec![
            kf(&mut fb, 0.0)?,
            kf(&mut fb, 0.0)?,
            kf(&mut fb, 0.0)?,
            kf(&mut fb, 1.0)?,
        ],
    };
    let far = [FAR, FAR, 0.0, 1.0];
    let mut xyzw = Vec::with_capacity(4);
    for (lane, away) in written.into_iter().zip(far) {
        let away = kf(&mut fb, away)?;
        xyzw.push(fb.select(drawn, lane, away)?);
    }
    let value = fb.composite_construct(vec4, &xyzw)?;
    let p = fb.module().global_value(position)?;
    fb.store(p, value, MemAccess::default())?;
    if let Some(g) = point_size {
        let written = match size_field {
            Some(f) => lanes(&mut fb, records, vertex, f)?,
            None => Vec::new(),
        };
        let size = match written.first() {
            Some(&size) => size,
            None => kf(&mut fb, 1.0)?,
        };
        let p = fb.module().global_value(g)?;
        fb.store(p, size, MemAccess::default())?;
    }
    for o in &outs {
        let record = if o.field.per_primitive {
            primitive
        } else {
            vertex
        };
        let parts = lanes(&mut fb, records, record, o.field)?;
        let value = match parts.as_slice() {
            [one] if !matches!(o.field.value, RecordValue::ClipDistance) => *one,
            _ => fb.composite_construct(o.ty, &parts)?,
        };
        let p = fb.module().global_value(o.global)?;
        fb.store(p, value, MemAccess::default())?;
    }
    fb.ret(None)?;
    let f = fb.finish()?;
    mb.entry(ExecModel::Vertex, f, Vec::new())?;
    mb.finalize()
}
