//! The synthesized stages of a tessellation pipeline.
//!
//! Metal has no tessellation control stage: a compute pass writes each patch's tessellation
//! factors to a buffer the render encoder binds (`setTessellationFactorBuffer:offset:
//! instanceStride:`), and the fixed-function tessellator reads them before the post-tessellation
//! vertex function runs once per generated vertex. Vulkan's tessellator takes its levels from a
//! tessellation control shader, so a tessellation pipeline has four stages:
//!
//! - the **vertex stage** ([`vertex_module`]) runs once per patch, each patch being the one
//!   control point of a `PATCH_LIST` patch: it writes `VertexIndex` (the patch's position, the
//!   draw's `patchStart` plus its index in the draw) and `InstanceIndex` (Metal's instance index,
//!   the base instance included) to the fixed locations
//!   `TessellationStage::PATCH_INDEX_LOCATION` and `TessellationStage::INSTANCE_LOCATION`;
//! - the **control stage** ([`control_module`]), one invocation per patch (`OutputVertices` 1),
//!   hands the evaluation stage Metal's patch index ([`patch_index`]) and the instance index as
//!   patch outputs at the same locations, and each of the patch's `N` control points' elements
//!   ([`elements`]) from `TessellationStage::CONTROL_POINT_LOCATION`, and writes the patch's
//!   tessellation levels from its factors ([`levels`]);
//! - the post-tessellation vertex function's evaluation stage, then the fragment stage.
//!
//! The control stage declares the pipeline's tessellation modes: the patch domain, the spacing
//! (Metal's `integer` and `pow2` partitions are `SpacingEqual`, `pow2` levels rounded up to a
//! power of two first; the fractional ones are Vulkan's fractional spacings), and the vertex
//! order. Under Vulkan's default upper-left domain origin, whose `TessCoord` is Metal's
//! `position_in_patch` lane for lane, Metal's clockwise output winding is `VertexOrderCcw` and its
//! counter-clockwise one `VertexOrderCw`: MoltenVK maps Vulkan's orders to Metal's windings
//! reversed (`mvkMTLWindingFromSpvExecutionMode`, with SPIRV-Cross passing `gl_TessCoord` through
//! unchanged for that origin). Measured: a consumer's device test,
//! `tessellation_states_tessellate_as_metals`, matches Metal's bytes (recorded on an Apple M3 Max)
//! for both windings, the clockwise default and the counter-clockwise one, on RADV (Radeon 780M,
//! RX 7900 XTX), NVIDIA Turing and MoltenVK; other drivers are not measured.
//!
//! **The patch index**. Metal's patch index (`[[patch_id]]`, the element of the patch's control
//! points and per-patch attributes) is the patch's position, or, for a draw with a patch index
//! buffer, the `uint` at the position in that buffer (`patchIndexBuffer[patchStart + i]`), while
//! the factors stay at the position (Metal probes on an M4). The control stage reads that word from
//! the patch index buffer (`SyntheticBinding::PatchIndices`, bound from an aligned base below its
//! offset, a multiple of 4, as the factor buffer is) where the pushed `PatchIndexBound` is nonzero;
//! a word outside the bound range reads 0. Metal's validation stops a direct draw there; without
//! it, Metal drew patch 0 there (probes 14b, 14d).
//!
//! **Control points**. Control point `k` of the patch whose index is `p` is element `p · N + k` of
//! its `PerPatchControlPoint` vertex buffers (wrapping `u32`), or, for a pipeline with a
//! control-point index type (`tessellationControlPointIndexType` `UInt16` or `UInt32`), the index
//! at that position in the draw's control-point index buffer
//! (`SyntheticBinding::ControlPointIndices`, bound from an aligned base below its offset, a
//! multiple of the index size; the pushed `ControlPointIndexOffset` is the residual): Metal probes
//! on an M4, with and without a patch index buffer, direct and indirect. The control stage hands
//! each element to the evaluation stage, which fetches the control point's attributes there
//! (`lower::tessellation`). An index outside the bound range reads 0. Metal's validation stops a
//! direct draw whose `(patchStart + patchCount) · N` indices pass the buffer; a patch index
//! buffer's patch indices are not checked against it, and an indirect draw is not checked (probe
//! 14e).
//!
//! **Factors.** A patch's `MTLTriangleTessellationFactorsHalf` (three edge factors and an inside
//! one, 8 bytes) or `MTLQuadTessellationFactorsHalf` (four and two, 12 bytes) sits at byte
//! `offset + position · size` (per-patch step functions) `+ instance · instanceStride`
//! (per-instance ones) of the factor buffer (`MTLTessellationFactorStepFunction`). Metal's
//! validation requires the offset and the instance stride to be multiples of 4, but plain Metal
//! reads an offset of 2 at that byte (Metal probe case `factor_offset_2` on an M4), so the stage
//! reads the `half`s from an even byte: the words from the one that holds it, each holding two
//! `half`s, low half first, the first `half` that word's high one where the address is 2 past a
//! multiple of 4. The binder refuses an odd offset or stride. The descriptor binds the buffer from
//! an aligned base below the bind offset, as a buffer slot's: the address adds the pushed residual.
//! A word outside the bound range reads 0 (contained), which culls. Edge `k` is Vulkan's outer
//! level `k` and inside factor `k` its inner level `k` (Metal's edge 0 is the edge `u = 0`, 1 `v =
//! 0`, 2 `u = 1` or `w = 0`, 3 `v = 1`, as Vulkan's outer levels; SPIRV-Cross maps them lane for
//! lane too). Then, as Metal states:
//!
//! 1. an edge factor that is negative, zero, or NaN culls the patch: every outer level is written
//!    0, and Vulkan discards a patch with an outer level at or below zero;
//! 2. with `tessellationFactorScaleEnabled`, every factor is multiplied by the encoder's scale,
//!    which Metal converts to `half` first (the binder pushes it so rounded, an infinite one as
//!    2^32, which takes every positive factor past the maximum level as infinity does, and
//!    refuses one that is not a positive `half`); the product of two `half`s is exact in
//!    `float`;
//! 3. every factor is clamped to its partition's range: `[1, max]` for `pow2` and `integer`,
//!    `[1, max - 1]` for `fractionalOdd`, `[2, max]` for `fractionalEven`, `max` the descriptor's
//!    `maxTessellationFactor` (at most the device's `maxTessellationGenerationLevel`, which link
//!    checks, so Vulkan's own clamp to that level changes nothing); a NaN inside factor takes the
//!    range's lower end and an infinite one its upper end;
//! 4. `pow2` rounds a level up to the next power of two, and the stage rounds a scaled `integer`
//!    level up to an integer itself; Vulkan rounds the others as Metal does (up to an integer,
//!    an odd, or an even integer).
//!
//! **Levels at `half` precision**. A driver that hands the levels to Metal's tessellator (MoltenVK:
//! SPIRV-Cross writes them to an `MTLTriangleTessellationFactorsHalf` or
//! `MTLQuadTessellationFactorsHalf` buffer, the only factor format Metal's tessellator reads)
//! rounds each to `half` before Metal rounds it up. Every level is a `half` but a scaled one:
//! the product of two `half`s is exact in `float`, not in `half`. So the stage rounds a scaled
//! `integer` level up itself, which is exact in `half` (at most 64) and which Vulkan's own
//! rounding leaves as it is: `0x47F3` by `0x3007`, 1.000445, is level 2 on every device, where
//! MoltenVK drew one segment from the unrounded product (its `half` is 1.0;
//! `patches::scale_product_unrounded`). A `pow2` level is a power of two already. A
//! scaled fractional level stays the product, which such a driver rounds to `half` (noted with
//! the scale, `PrecisionNote::TessellationFactorScale`).
//!
//! **No float operation sees a NaN or an infinity.** Without `SignedZeroInfNanPreserve` a Vulkan
//! implementation may assume that no operand or result of a float operation is a NaN or an
//! infinity, so neither the cull nor the clamp may rest on how a float comparison treats one.
//! Each factor is classified on its `half` bits with integer operations (a culling edge factor
//! is outside `[0x0001, 0x7c00]`; a finite positive one is in `[0x0001, 0x7bff]`; `+∞` is
//! `0x7c00`); only a finite positive factor becomes a `float` (its significand converted,
//! scaled by `Ldexp` and a power of two, exact whether it is normal or subnormal: every `half`
//! is a normal `float`), and the level of any other factor is a constant, its range's lower end
//! (NaN, zero, negative) or upper end (`+∞`). The pushed scale is a positive `half` or 2^32, so
//! the scaled product lies in `[2^-48, 2^48)`: every float the stage computes is finite.
//!
//! **The address form.** Under the identity domains (`Device::from_queries`' profile) the control
//! stage binds nothing: it reads the factors through their device address, the pushed
//! `PushField::TessellationFactorAddress` (the factor buffer's address plus the encoder's offset, a
//! `u64`), at the same bytes and with the same rules as above, from the word that holds the even
//! byte the address names, and no word past the one that holds the last factor's byte. Nothing
//! contains the read: as Metal's own, a read past the buffer reads whatever memory is there. The patch index is the patch's position: the form reads no patch
//! index buffer, and its consumer refuses a draw that names one. A control-point index type is not
//! implemented in it ([`super::control_plan`]). Its push fields ([`address_fields`]) are the
//! address and, where read, the instance stride and the scale, which the pipeline's interface
//! places as its vertex stage's (`PipelineInterface::push`).
//!
//! Noted on the pipeline's evaluation stage ([`Control::notes`]): the output winding's vertex
//! order, measured on four drivers (above) and not on the rest; and, read from Metal's
//! documentation but not observed: whether the fractional partitions'
//! levels take the scaled factor in `float` (Metal's `integer` levels take it unrounded, and a
//! product below `half`'s least subnormal does not cull); a subnormal edge factor and a NaN inside
//! factor under the fractional partitions (under `integer`, Metal's answer is this stage's); and
//! whether the per-instance step's instance index includes the base instance (taken to, as
//! `[[instance_id]]` does). These are A-track readings.

use crate::lower::tessellation::element_array;
use crate::lower::tessellation::factors::{self, factor_halves, word};
use crate::metal::{
    TessellationControlPointIndexType as IndexType, TessellationFactorStepFunction as Step,
    TessellationPartitionMode as Partition, Winding,
};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::interface::{PatchDomain, TessellationStage};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ConstBits, Constant, ConvOp, ExecMode, ExecModel, GlslOp, IoDir,
    LayoutRule, MemAccess, StorageClass, Type, TypedModule,
};

/// The factor buffer's binding symbol: a synthetic binding declared with the vertex stage's
/// symbols (the control stage belongs to Metal's vertex stage).
pub const FACTOR_SLOT: ResourceSlot = ResourceSlot {
    stage: Stage::Vertex,
    kind: ResourceKind::Synthetic(SyntheticBinding::TessellationFactors),
    index: 0,
};

/// The patch index buffer's binding symbol, declared with the vertex stage's symbols as the
/// factor buffer's is.
pub const PATCH_INDEX_SLOT: ResourceSlot = ResourceSlot {
    stage: Stage::Vertex,
    kind: ResourceKind::Synthetic(SyntheticBinding::PatchIndices),
    index: 0,
};

/// The control-point index buffer's binding symbol, declared with the vertex stage's symbols as
/// the factor buffer's is, by a pipeline with a control-point index type only.
pub const CONTROL_POINT_INDEX_SLOT: ResourceSlot = ResourceSlot {
    stage: Stage::Vertex,
    kind: ResourceKind::Synthetic(SyntheticBinding::ControlPointIndices),
    index: 0,
};

/// The control stage's push fields, in their order in its push range, 4 bytes each: the factor
/// buffer's residual bind offset and instance stride (`u32`), the scale (`float`), the patch
/// index buffer's residual bind offset, whether the draw binds one, and the control-point index
/// buffer's residual bind offset (`u32`, read by a pipeline with a control-point index type).
pub const PUSH_FIELDS: [PushField; 6] = [
    PushField::TessellationFactorOffset,
    PushField::TessellationFactorInstanceStride,
    PushField::TessellationFactorScale,
    PushField::PatchIndexOffset,
    PushField::PatchIndexBound,
    PushField::ControlPointIndexOffset,
];

/// Bytes of the control stage's push range.
pub const PUSH_BYTES: u32 = 24;

/// The address form's push fields (the module docs), for step function `step` and a scale
/// enabled or not: each with its byte offset from the stage's push offset, a multiple of 8. The
/// factors' address (`u64`), then, where the stage reads them, the instance stride (`u32`, a
/// per-instance step) and the scale (`float`).
pub fn address_fields(step: Step, scale_enabled: bool) -> Vec<(PushField, u32)> {
    let mut fields = vec![(PushField::TessellationFactorAddress, 0)];
    if per_instance(step) {
        fields.push((PushField::TessellationFactorInstanceStride, 8));
    }
    if scale_enabled {
        fields.push((PushField::TessellationFactorScale, 12));
    }
    fields
}

/// Whether step function `step` addresses a patch's factors by its instance.
fn per_instance(step: Step) -> bool {
    matches!(step, Step::PerInstance | Step::PerPatchAndPerInstance)
}

/// The size of the vertex stage's `VertexIndex`-carrying output array the control stage reads:
/// `gl_MaxPatchVertices`, the smallest `maxTessellationPatchSize` Vulkan allows.
const INPUT_ARRAY: u32 = 32;

/// What the control stage computes: the patch domain, the descriptor's tessellation state, and
/// where its push fields are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Control {
    pub domain: PatchDomain,
    pub partition: Partition,
    pub winding: Winding,
    /// `maxTessellationFactor`.
    pub max_factor: u32,
    pub step: Step,
    pub scale_enabled: bool,
    /// The byte offset of the first push field.
    pub push_offset: u32,
    /// The `N` of the function's `[[patch(_, N)]]` (0 when unstated: the evaluation stage then
    /// reads no control point, and the stage hands no element).
    pub control_points: u32,
    /// `tessellationControlPointIndexType`.
    pub index_type: IndexType,
    /// Whether the stage reads the factors through their address (the module docs' address
    /// form, under the identity domains) rather than through a bound factor buffer.
    pub addressed: bool,
}

impl Control {
    /// Edge and inside factors per patch.
    fn counts(&self) -> (usize, usize) {
        match self.domain {
            PatchDomain::Triangle => (3, 1),
            PatchDomain::Quad => (4, 2),
        }
    }

    /// Bytes and alignment of the stage's push range: [`PUSH_BYTES`] of words, or the address
    /// form's fields ([`address_fields`]), 8-aligned for the address.
    pub fn push_bytes(addressed: bool, step: Step, scale_enabled: bool) -> (u32, u32) {
        match addressed {
            false => (PUSH_BYTES, 4),
            true => {
                let end = (address_fields(step, scale_enabled).iter())
                    .map(|&(field, at)| at + crate::lower::push::shape(field).size())
                    .max()
                    .unwrap_or(0);
                (end, 8)
            }
        }
    }

    /// Bytes of one patch's factors (`tessellationFactorStride`).
    pub fn factor_bytes(domain: PatchDomain) -> u32 {
        factors::factor_bytes(domain)
    }

    /// The readings this control stage takes without a Metal observation on every driver, as
    /// precision notes (the module docs): the winding (measured on four drivers) and the edge
    /// cases always, the scale when it is enabled, the instance addressing under a per-instance
    /// step function, and a fractional partition's vertex places, the device tessellator's. Link
    /// notes them on the pipeline's evaluation stage ([`super::note_evaluation_stage`]).
    pub fn notes(&self) -> Vec<PrecisionNote> {
        let mut notes = vec![
            PrecisionNote::TessellationWinding,
            PrecisionNote::TessellationFactorEdgeCases,
        ];
        if self.scale_enabled {
            notes.push(PrecisionNote::TessellationFactorScale);
        }
        if per_instance(self.step) {
            notes.push(PrecisionNote::TessellationFactorInstance);
        }
        if matches!(
            self.partition,
            Partition::FractionalOdd | Partition::FractionalEven
        ) {
            notes.push(PrecisionNote::TessellationFractionalSpacing);
        }
        notes
    }

    /// The level of a factor `x` within the clamp range: `x`, rounded up to a power of two for
    /// `pow2` (whose range ends are powers of two, so each end is its own level).
    fn level(&self, x: f32) -> f32 {
        match self.partition {
            Partition::Pow2 => (x.ceil() as u32).next_power_of_two() as f32,
            _ => x,
        }
    }

    /// The clamp range of the partition (the module docs, item 3).
    fn range(&self) -> (f32, f32) {
        let max = self.max_factor as f32;
        match self.partition {
            Partition::FractionalOdd => (1.0, max - 1.0),
            Partition::FractionalEven => (2.0, max),
            _ => (1.0, max),
        }
    }
}

/// Where the control stage reads what a draw binds: the factor buffer, the patch index buffer,
/// and, for a pipeline with a control-point index type, the control-point index buffer (each a
/// `Block { uint[] }` storage buffer), and the push block with [`PUSH_FIELDS`] at member indices
/// `members`. In the address form no buffer is bound, and `members` holds the member of each of
/// [`address_fields`] at its field's place in [`PUSH_FIELDS`] (the address at the offset's).
#[derive(Clone, Copy, Debug)]
pub struct Sources {
    pub factors: Option<GlobalId>,
    pub patch_indices: Option<GlobalId>,
    pub control_point_indices: Option<GlobalId>,
    pub push: GlobalId,
    pub members: [u32; 6],
}

/// Declare the factor, patch index, and (for `c`'s control-point index type) control-point
/// index buffers, the push fields at `c.push_offset`, and the push block; in the address form, the
/// push fields of [`address_fields`] and the push block alone.
pub fn declare_sources(mb: &mut ModuleBuilder, c: &Control) -> BResult<Sources> {
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = mb.ty(Type::Float { width: 32 })?;
    if c.addressed {
        let u64_ = mb.ty(Type::Int {
            width: 64,
            signed: false,
        })?;
        let mut members = [0; 6];
        for (field, at) in address_fields(c.step, c.scale_enabled) {
            let (k, ty) = match field {
                PushField::TessellationFactorInstanceStride => (1, u32_),
                PushField::TessellationFactorScale => (2, f32_),
                _ => (0, u64_),
            };
            members[k] = mb.push_member(field, ty, c.push_offset + at)?.index;
        }
        let push = mb.push_global()?;
        return Ok(Sources {
            factors: None,
            patch_indices: None,
            control_point_indices: None,
            push,
            members,
        });
    }
    let words = mb.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(mb.types(), LayoutRule::Std430, true, &[words])?;
    let block = mb.ty(Type::Struct {
        members: [words].into(),
        layout: Some(layout),
    })?;
    let factors = mb.resource(FACTOR_SLOT, StorageClass::StorageBuffer, block)?;
    let patch_indices = mb.resource(PATCH_INDEX_SLOT, StorageClass::StorageBuffer, block)?;
    let control_point_indices = match c.index_type {
        IndexType::None => None,
        _ => Some(mb.resource(CONTROL_POINT_INDEX_SLOT, StorageClass::StorageBuffer, block)?),
    };
    let mut members = [0; 6];
    let types = [u32_, u32_, f32_, u32_, u32_, u32_];
    for (k, (field, ty)) in PUSH_FIELDS.iter().zip(types).enumerate() {
        let offset = c.push_offset + 4 * k as u32;
        members[k] = mb.push_member(*field, ty, offset)?.index;
    }
    let push = mb.push_global()?;
    Ok(Sources {
        factors: Some(factors),
        patch_indices: Some(patch_indices),
        control_point_indices,
        push,
        members,
    })
}

fn uint(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(Type::Int {
        width: 32,
        signed: false,
    })
}

fn float(fb: &mut FunctionBuilder<'_>) -> BResult<TypeId> {
    fb.module().ty(Type::Float { width: 32 })
}

fn ku(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

fn kf(fb: &mut FunctionBuilder<'_>, v: f32) -> BResult<Value> {
    let ty = float(fb)?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(v.to_bits())),
    })?;
    fb.module().const_value(c)
}

/// Push member `member` of the block `push`, loaded.
fn push_word(fb: &mut FunctionBuilder<'_>, push: GlobalId, member: u32) -> BResult<Value> {
    let base = fb.module().global_value(push)?;
    let index = ku(fb, member)?;
    let ptr = fb.access_chain(base, &[index])?;
    fb.load(ptr, MemAccess::default())
}

/// The `float` value of a finite positive `half` `h` (bits in `[0x0001, 0x7bff]`, a `uint`),
/// exactly: its significand (the implicit bit set for a normal one) converted and scaled by
/// `2^(max(e, 1) - 25)` as `Ldexp` by `max(e, 1)` and a multiply by `2^-25`, both exact since the
/// value is a normal `float`. Any `h` below `0x8000` decodes to a finite `float` (at most
/// `2047 · 2^6`), so no operation here can produce a NaN or an infinity.
fn half_value(fb: &mut FunctionBuilder<'_>, h: Value) -> BResult<Value> {
    let f32_ = float(fb)?;
    let ten = ku(fb, 10)?;
    let exp = fb.binary(BinOp::ShiftRightLogical, h, ten)?;
    let mask = ku(fb, 0x3ff)?;
    let mant = fb.binary(BinOp::BitwiseAnd, h, mask)?;
    let zero = ku(fb, 0)?;
    let subnormal = fb.binary(BinOp::IEqual, exp, zero)?;
    let implicit = ku(fb, 0x400)?;
    let normal = fb.binary(BinOp::BitwiseOr, mant, implicit)?;
    let significand = fb.select(subnormal, mant, normal)?;
    let one = ku(fb, 1)?;
    let exp = fb.select(subnormal, one, exp)?;
    let m = fb.convert(ConvOp::ConvertUToF, significand, f32_)?;
    let scaled = fb.glsl(GlslOp::Ldexp, &[m, exp])?;
    let ulp = kf(fb, f32::from_bits(0x3300_0000))?; // 2^-25
    fb.binary(BinOp::FMul, scaled, ulp)
}

/// `x` (finite) clamped to `[lo, hi]`: each bound by an ordered comparison and a select.
fn clamp(fb: &mut FunctionBuilder<'_>, x: Value, lo: f32, hi: f32) -> BResult<Value> {
    let lo = kf(fb, lo)?;
    let above = fb.binary(BinOp::FOrdGreaterThanEqual, x, lo)?;
    let x = fb.select(above, x, lo)?;
    let hi = kf(fb, hi)?;
    let below = fb.binary(BinOp::FOrdLessThanEqual, x, hi)?;
    fb.select(below, x, hi)
}

/// Whether the `half` bits `h` (a `uint` below `2^16`) are in `[first, last]`, decided on the
/// bits: `h - first <= last - first` without sign.
fn bits_in(fb: &mut FunctionBuilder<'_>, h: Value, first: u32, last: u32) -> BResult<Value> {
    let first_ = ku(fb, first)?;
    let from = fb.binary(BinOp::ISub, h, first_)?;
    let span = ku(fb, last - first)?;
    fb.binary(BinOp::ULessThanEqual, from, span)
}

/// `x` (in `[1, 64]`) rounded up to the next power of two: `1 << (msb(⌈x⌉ - 1) + 1)`, where the
/// most significant bit of 0 is -1 (`FindUMsb`), so 1 stays 1.
fn pow2(fb: &mut FunctionBuilder<'_>, x: Value) -> BResult<Value> {
    let u32_ = uint(fb)?;
    let f32_ = float(fb)?;
    let up = fb.glsl(GlslOp::Ceil, &[x])?;
    let n = fb.convert(ConvOp::ConvertFToU, up, u32_)?;
    let one = ku(fb, 1)?;
    let below = fb.binary(BinOp::ISub, n, one)?;
    let msb = fb.glsl(GlslOp::FindUMsb, &[below])?;
    let shift = fb.binary(BinOp::IAdd, msb, one)?;
    let p = fb.binary(BinOp::ShiftLeftLogical, one, shift)?;
    fb.convert(ConvOp::ConvertUToF, p, f32_)
}

/// Metal's patch index of the patch at `position` (a `uint`, the draw's `patchStart` plus the
/// patch's index in the draw): the patch index buffer's word at byte `residual + 4 · position`
/// of its bound range (wrapping `u32`; the pushed residual bind offset is a multiple of 4 modulo
/// `2^32`), 0 where the range does not hold it, when the pushed `PatchIndexBound` is nonzero;
/// `position` otherwise, and in the address form (the module docs).
pub fn patch_index(fb: &mut FunctionBuilder<'_>, src: &Sources, position: Value) -> BResult<Value> {
    let Some(patch_indices) = src.patch_indices else {
        return Ok(position);
    };
    let residual = push_word(fb, src.push, src.members[3])?;
    let four = ku(fb, 4)?;
    let bytes = fb.binary(BinOp::IMul, position, four)?;
    let address = fb.binary(BinOp::IAdd, residual, bytes)?;
    let two = ku(fb, 2)?;
    let at = fb.binary(BinOp::ShiftRightLogical, address, two)?;
    let indexed = word(fb, patch_indices, at)?;
    let bound = push_word(fb, src.push, src.members[4])?;
    let zero = ku(fb, 0)?;
    let bound = fb.binary(BinOp::INotEqual, bound, zero)?;
    fb.select(bound, indexed, position)
}

/// The elements of the `c.control_points` control points of the patch whose index is `patch`
/// (the module docs): control point `k`'s is `patch · N + k` (wrapping `u32`), or, with a
/// control-point index type, the index there in the control-point index buffer: the `uint` or
/// `ushort` at byte `residual + e · size` of its bound range (wrapping `u32`, the pushed residual
/// a multiple of the size modulo `2^32`), 0 where the range does not hold it.
pub fn elements(
    fb: &mut FunctionBuilder<'_>,
    c: &Control,
    src: &Sources,
    patch: Value,
) -> BResult<Vec<Value>> {
    let n = ku(fb, c.control_points)?;
    let first = fb.binary(BinOp::IMul, patch, n)?;
    let residual = match src.control_point_indices {
        Some(_) => Some(push_word(fb, src.push, src.members[5])?),
        None => None,
    };
    let mut out = Vec::with_capacity(c.control_points as usize);
    for k in 0..c.control_points {
        let k = ku(fb, k)?;
        let e = fb.binary(BinOp::IAdd, first, k)?;
        let (Some(buffer), Some(residual)) = (src.control_point_indices, residual) else {
            out.push(e);
            continue;
        };
        let size = match c.index_type {
            IndexType::UInt16 => 2,
            _ => 4,
        };
        let size_ = ku(fb, size)?;
        let bytes = fb.binary(BinOp::IMul, e, size_)?;
        let address = fb.binary(BinOp::IAdd, residual, bytes)?;
        let two = ku(fb, 2)?;
        let at = fb.binary(BinOp::ShiftRightLogical, address, two)?;
        let w = word(fb, buffer, at)?;
        out.push(if size == 2 {
            // The `ushort` in the word's low half at a multiple of 4, its high half otherwise.
            let half = fb.binary(BinOp::BitwiseAnd, address, two)?;
            let three = ku(fb, 3)?;
            let shift = fb.binary(BinOp::ShiftLeftLogical, half, three)?;
            let w = fb.binary(BinOp::ShiftRightLogical, w, shift)?;
            let mask = ku(fb, 0xffff)?;
            fb.binary(BinOp::BitwiseAnd, w, mask)?
        } else {
            w
        });
    }
    Ok(out)
}

/// The tessellation levels of the patch at `position` (the draw's `patchStart` plus the patch's
/// index in the draw, `uint`) of instance `instance` (Metal's index): the outer levels (edge
/// factors) and inner levels (inside factors) of `c.domain`, computed from the factors at `src`
/// as the module docs state.
pub fn levels(
    fb: &mut FunctionBuilder<'_>,
    c: &Control,
    src: &Sources,
    position: Value,
    instance: Value,
) -> BResult<(Vec<Value>, Vec<Value>)> {
    let (edges, insides) = c.counts();
    let halves = match src.factors {
        Some(factors) => {
            let mut address = push_word(fb, src.push, src.members[0])?;
            if matches!(c.step, Step::PerPatch | Step::PerPatchAndPerInstance) {
                let size = ku(fb, Control::factor_bytes(c.domain))?;
                let at = fb.binary(BinOp::IMul, position, size)?;
                address = fb.binary(BinOp::IAdd, address, at)?;
            }
            if per_instance(c.step) {
                let stride = push_word(fb, src.push, src.members[1])?;
                let at = fb.binary(BinOp::IMul, instance, stride)?;
                address = fb.binary(BinOp::IAdd, address, at)?;
            }
            factor_halves(fb, factors, address, edges + insides)?
        }
        None => {
            let address = factor_address(fb, c, src, position, instance)?;
            address_halves(fb, address, edges + insides)?
        }
    };
    // Each factor is classified on its bits (the module docs).
    // 1. An edge factor that is not above zero (negative, zero, or NaN: bits outside
    //    `[0x0001, 0x7c00]`) culls the patch.
    let mut culled = None;
    for &h in halves.iter().take(edges) {
        let one = ku(fb, 1)?;
        let from = fb.binary(BinOp::ISub, h, one)?;
        let span = ku(fb, 0x7c00 - 1)?;
        let not_above = fb.binary(BinOp::UGreaterThan, from, span)?;
        culled = Some(match culled {
            None => not_above,
            Some(c) => fb.binary(BinOp::LogicalOr, c, not_above)?,
        });
    }
    let scale = if c.scale_enabled {
        Some(push_word(fb, src.push, src.members[2])?)
    } else {
        None
    };
    let (lo, hi) = c.range();
    let low = kf(fb, c.level(lo))?;
    let high = kf(fb, c.level(hi))?;
    let one_bits = ku(fb, 0x3c00)?;
    let infinity = ku(fb, 0x7c00)?;
    let mut out = Vec::with_capacity(edges + insides);
    for h in halves.into_iter().take(edges + insides) {
        let finite = bits_in(fb, h, 0x0001, 0x7bff)?;
        let infinite = fb.binary(BinOp::IEqual, h, infinity)?;
        // Only a finite positive factor becomes a `float`; any other decodes as 1.0, unused.
        let h = fb.select(finite, h, one_bits)?;
        let f = half_value(fb, h)?;
        // 2. The scale (a `half`, so the product is exact in `float`).
        let f = match scale {
            Some(s) => fb.binary(BinOp::FMul, f, s)?,
            None => f,
        };
        // 3. The partition's range; 4. `pow2`'s rounding, and a scaled `integer` level's.
        let f = clamp(fb, f, lo, hi)?;
        let f = match c.partition {
            Partition::Pow2 => pow2(fb, f)?,
            Partition::Integer if scale.is_some() => fb.glsl(GlslOp::Ceil, &[f])?,
            _ => f,
        };
        // A NaN, zero, or negative factor takes the range's lower end, `+∞` its upper end.
        let other = fb.select(infinite, high, low)?;
        out.push(fb.select(finite, f, other)?);
    }
    let inner = out.split_off(edges);
    let zero = kf(fb, 0.0)?;
    let outer = match culled {
        Some(culled) => {
            let mut outer = Vec::with_capacity(edges);
            for f in out {
                outer.push(fb.select(culled, zero, f)?);
            }
            outer
        }
        None => out,
    };
    Ok((outer, inner))
}

/// The address form's address of the factors of the patch at `position` of instance `instance`:
/// the pushed address plus `position · size` (per-patch steps) and `instance · instanceStride`
/// (per-instance ones), the two terms summed as Metal's offset is, in `u32`.
fn factor_address(
    fb: &mut FunctionBuilder<'_>,
    c: &Control,
    src: &Sources,
    position: Value,
    instance: Value,
) -> BResult<Value> {
    let base = push_word(fb, src.push, src.members[0])?;
    let mut offset = None;
    if matches!(c.step, Step::PerPatch | Step::PerPatchAndPerInstance) {
        let size = ku(fb, Control::factor_bytes(c.domain))?;
        offset = Some(fb.binary(BinOp::IMul, position, size)?);
    }
    if per_instance(c.step) {
        let stride = push_word(fb, src.push, src.members[1])?;
        let at = fb.binary(BinOp::IMul, instance, stride)?;
        offset = Some(match offset {
            Some(o) => fb.binary(BinOp::IAdd, o, at)?,
            None => at,
        });
    }
    let Some(offset) = offset else {
        return Ok(base);
    };
    let u64_ = fb.module().ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let wide = fb.convert(ConvOp::UConvert, offset, u64_)?;
    fb.binary(BinOp::IAdd, base, wide)
}

/// The `count` factors' `half` bits (each a `uint`) at the even device address `address` (a
/// `u64`), as [`factor_halves`] reads them from a bound buffer: the words from the one that holds
/// the address to the one that holds the last factor's byte, each loaded through a read-only
/// `PhysicalStorageBuffer` pointer, unpacked low half first, from that word's high half where the
/// address is 2 past a multiple of 4.
fn address_halves(
    fb: &mut FunctionBuilder<'_>,
    address: Value,
    count: usize,
) -> BResult<Vec<Value>> {
    let u32_ = uint(fb)?;
    let u64_ = fb.module().ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let k64 = |fb: &mut FunctionBuilder<'_>, v: u64| -> BResult<Value> {
        let c = fb.module().constant(Constant::Scalar {
            ty: u64_,
            bits: ConstBits(v),
        })?;
        fb.module().const_value(c)
    };
    let aligned = k64(fb, !3)?;
    let first = fb.binary(BinOp::BitwiseAnd, address, aligned)?;
    let low = fb.convert(ConvOp::UConvert, address, u32_)?;
    let two = ku(fb, 2)?;
    let high = fb.binary(BinOp::BitwiseAnd, low, two)?;
    let zero = ku(fb, 0)?;
    let shifted = fb.binary(BinOp::INotEqual, high, zero)?;
    let mask = ku(fb, 0xffff)?;
    let sixteen = ku(fb, 16)?;
    let words = (count + 1).div_ceil(2);
    let mut unpacked = Vec::with_capacity(count + 2);
    for k in 0..words {
        let at = match k {
            0 => first,
            _ => {
                let offset = k64(fb, 4 * k as u64)?;
                let at = fb.binary(BinOp::IAdd, first, offset)?;
                // The word past the factors' last is read only where they start 2 past a
                // multiple of 4; elsewhere the first word is read again, so every word read
                // holds a factor's byte.
                match count % 2 == 0 && k == words - 1 {
                    true => fb.select(shifted, at, first)?,
                    false => at,
                }
            }
        };
        let ptr = fb.device_place_read_only(at, u32_)?;
        let w = fb.load(
            ptr,
            MemAccess {
                aligned: Some(4),
                ..MemAccess::default()
            },
        )?;
        unpacked.push(fb.binary(BinOp::BitwiseAnd, w, mask)?);
        unpacked.push(fb.binary(BinOp::ShiftRightLogical, w, sixteen)?);
    }
    let mut halves = Vec::with_capacity(count);
    for (&low, &high) in unpacked.iter().zip(unpacked.iter().skip(1)).take(count) {
        halves.push(fb.select(shifted, high, low)?);
    }
    Ok(halves)
}

/// The execution modes of the control stage: domain, spacing, vertex order, one output vertex.
fn modes(c: &Control) -> Vec<ExecMode> {
    vec![
        match c.domain {
            PatchDomain::Triangle => ExecMode::Triangles,
            PatchDomain::Quad => ExecMode::Quads,
        },
        match c.partition {
            Partition::FractionalOdd => ExecMode::SpacingFractionalOdd,
            Partition::FractionalEven => ExecMode::SpacingFractionalEven,
            _ => ExecMode::SpacingEqual,
        },
        match c.winding {
            Winding::CounterClockwise => ExecMode::VertexOrderCw,
            _ => ExecMode::VertexOrderCcw,
        },
        ExecMode::OutputVertices { n: 1 },
    ]
}

/// A `float[n]` array type.
fn float_array(mb: &mut ModuleBuilder, n: u32) -> BResult<TypeId> {
    let f32_ = mb.ty(Type::Float { width: 32 })?;
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let len = mb.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(n)),
    })?;
    mb.ty(Type::Array {
        elem: f32_,
        len,
        stride: None,
    })
}

/// The synthesized vertex stage: `VertexIndex` (the patch's position) and `InstanceIndex` to the
/// fixed locations the control stage reads (the module docs).
pub fn vertex_module() -> BResult<TypedModule> {
    let mut mb = ModuleBuilder::new();
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let void = mb.ty(Type::Void)?;
    let pairs = [
        (
            crate::spv::BuiltIn::VertexIndex,
            TessellationStage::PATCH_INDEX_LOCATION,
        ),
        (
            crate::spv::BuiltIn::InstanceIndex,
            TessellationStage::INSTANCE_LOCATION,
        ),
    ];
    let mut io = Vec::with_capacity(pairs.len());
    for (builtin, location) in pairs {
        let from = mb.builtin(builtin, IoDir::Input, u32_)?;
        let to = mb.fixed_io(location, IoDir::Output, u32_, false)?;
        io.push((from, to));
    }
    let mut fb = mb.function(void)?;
    for (from, to) in io {
        let src = fb.module().global_value(from)?;
        let v = fb.load(src, MemAccess::default())?;
        let dst = fb.module().global_value(to)?;
        fb.store(dst, v, MemAccess::default())?;
    }
    fb.ret(None)?;
    let f = fb.finish()?;
    mb.entry(ExecModel::Vertex, f, Vec::new())?;
    mb.finalize()
}

/// The synthesized tessellation control stage of `c` (the module docs).
pub fn control_module(c: &Control) -> BResult<TypedModule> {
    let mut mb = ModuleBuilder::new();
    let src = declare_sources(&mut mb, c)?;
    let u32_ = mb.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let void = mb.ty(Type::Void)?;
    let len = mb.constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(u64::from(INPUT_ARRAY)),
    })?;
    let per_vertex = mb.ty(Type::Array {
        elem: u32_,
        len,
        stride: None,
    })?;
    let mut io = |location: u32| -> BResult<(GlobalId, GlobalId)> {
        let from = mb.fixed_io(location, IoDir::Input, per_vertex, false)?;
        let to = mb.fixed_io(location, IoDir::Output, u32_, true)?;
        Ok((from, to))
    };
    let patch_io = io(TessellationStage::PATCH_INDEX_LOCATION)?;
    let instance_io = io(TessellationStage::INSTANCE_LOCATION)?;
    let elements_out = match c.control_points {
        0 => None,
        n => {
            let ty = element_array(&mut mb, n)?;
            let at = TessellationStage::CONTROL_POINT_LOCATION;
            Some(mb.fixed_io(at, IoDir::Output, ty, true)?)
        }
    };
    let outer_ty = float_array(&mut mb, 4)?;
    let inner_ty = float_array(&mut mb, 2)?;
    let outer = mb.builtin(crate::spv::BuiltIn::TessLevelOuter, IoDir::Output, outer_ty)?;
    let inner = mb.builtin(crate::spv::BuiltIn::TessLevelInner, IoDir::Output, inner_ty)?;
    let mut fb = mb.function(void)?;
    // Control point 0 of the patch's input is the patch: its position and instance index.
    let mut read = |from: GlobalId| -> BResult<Value> {
        let base = fb.module().global_value(from)?;
        let zero = ku(&mut fb, 0)?;
        let ptr = fb.access_chain(base, &[zero])?;
        fb.load(ptr, MemAccess::default())
    };
    let position = read(patch_io.0)?;
    let instance = read(instance_io.0)?;
    let patch = patch_index(&mut fb, &src, position)?;
    for (to, v) in [(patch_io.1, patch), (instance_io.1, instance)] {
        let dst = fb.module().global_value(to)?;
        fb.store(dst, v, MemAccess::default())?;
    }
    // Control point `k`'s element in component `k % 4` of element `k / 4`.
    if let Some(to) = elements_out {
        let base = fb.module().global_value(to)?;
        for (k, v) in elements(&mut fb, c, &src, patch)?.into_iter().enumerate() {
            let quad = ku(&mut fb, k as u32 / 4)?;
            let lane = ku(&mut fb, k as u32 % 4)?;
            let ptr = fb.access_chain(base, &[quad, lane])?;
            fb.store(ptr, v, MemAccess::default())?;
        }
    }
    let (outer_levels, inner_levels) = levels(&mut fb, c, &src, position, instance)?;
    for (array, values) in [(outer, outer_levels), (inner, inner_levels)] {
        let base = fb.module().global_value(array)?;
        for (k, v) in values.into_iter().enumerate() {
            let index = ku(&mut fb, k as u32)?;
            let ptr = fb.access_chain(base, &[index])?;
            fb.store(ptr, v, MemAccess::default())?;
        }
    }
    fb.ret(None)?;
    let f = fb.finish()?;
    mb.entry(ExecModel::TessellationControl, f, modes(c))?;
    mb.finalize()
}

#[cfg(test)]
mod tests;
