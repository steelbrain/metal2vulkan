//! GPU-encoded indexed draws under the identity domains: a function's `render_command` buffer
//! setters, indexed draw and reset write the records of an indirect command buffer
//! (`pipeline::IndirectDrawLayout`) through its `gpuResourceID`, the records' device address,
//! which an argument buffer's `command_buffer` member holds (its handle load is `encode`'s object
//! word, a `u64`).
//!
//! Each operation reads the header's capacity, slot counts and stride, and writes only where the
//! command's index is below the capacity and, for a buffer setter, the slot below its stage's
//! count; any other write, which Metal leaves undefined, writes nothing. Every write is a plain
//! `PhysicalStorageBuffer` store at `address + HEADER_BYTES + index * stride + field`:
//!
//! - `set_vertex_buffer` and `set_fragment_buffer` write the buffer's Metal address less 2^31 (the
//!   word an execution pushes as the slot's `SlotGuestBase`) at the slot's word. The attribute
//!   stride form is lowered only for the constant all ones, the pipeline's own stride, as MSL's
//!   plain setter spells it.
//! - `draw_indexed_primitives` of the constant triangle type writes the index token's data (the
//!   indices' address, `count` indices' bytes and the index type of the overload) and
//!   `VkDrawIndexedIndirectCommand` (`count`, the instances, first index 0, the base vertex and the
//!   base instance). Another primitive type is refused: an execution draws one topology.
//! - `reset` writes the index token at the header's reset index address with no bytes and a draw
//!   of nothing; the command's slot words stay.
//!
//! Every other indirect-command operation (compute commands, plain, patch and mesh draws,
//! pipeline and render state setters, barriers, copies) is
//! `Unsupported{GpuEncodedIndirectCommands, NotImplemented}`. A vertex or fragment function on a
//! device without its stage's store feature, and a device without device addresses, refuse every
//! operation, naming what they lack.

use std::borrow::Cow;

use crate::air::intrinsics::{AirIntrinsic, CommandStage, IndirectCommand};
use crate::air::ir::{Callee, Inst, InstKind, Value, ValueId};
use crate::budget::Meter;
use crate::device::profile::BufferAccess;
use crate::error::{
    Capability, DiagName, Error, Feature, InternalError, InternalKind, Unsupported,
};
use crate::lower::arith::LoweredConst;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::leaf::k32;
use crate::memory::provenance::Repr;
use crate::memory::roots::k64;
use crate::pipeline::IndirectDrawLayout as Records;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::Value as SValue;
use crate::spirv::ir::{BinOp, ConvOp, MemAccess, Type as SType};

use super::{buffer_roots, dynamic, operand, pointer, refuse, Target, SLOT_ADDRESS_BIAS};

/// One operation's writes into its command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Command {
    /// The command buffer's word: its records' device address.
    pub records: ValueId,
    /// The command's index (`i32`, read as unsigned).
    pub index: ValueId,
    pub write: Write,
}

/// What an operation writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Write {
    /// A buffer's address at `slot` (`i32`) of the vertex slots, or of the fragment slots.
    Buffer {
        fragment: bool,
        address: Address,
        slot: ValueId,
    },
    /// An indexed draw of triangles: `count` indices of 16 or 32 bits (`wide`) at `indices`.
    Draw {
        count: ValueId,
        indices: Address,
        wide: bool,
        instances: ValueId,
        base_vertex: ValueId,
        base_instance: ValueId,
    },
    /// A reset.
    Reset,
}

/// A buffer pointer's Metal address.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Address {
    /// The null pointer constant: 0.
    Null,
    /// A static-root fat pointer: its slot's address plus its displacement
    /// (`dynamic::static_guest_address`).
    Static { ptr: ValueId, targets: Vec<Target> },
    /// A wide pointer: the address it carries (`dynamic::guest_address`).
    Wide(ValueId),
}

/// SPIR-V words of an operation: the header's loads, the checks, its stores and the construct.
const WORDS: u32 = 400;

impl Command {
    /// The census name (`access.commands.<op>`).
    pub fn name(&self) -> Cow<'static, str> {
        match self.write {
            Write::Buffer {
                fragment: false, ..
            } => "access.commands.vertex_buffer".into(),
            Write::Buffer { fragment: true, .. } => "access.commands.fragment_buffer".into(),
            Write::Draw { .. } => "access.commands.draw_indexed".into(),
            Write::Reset => "access.commands.reset".into(),
        }
    }

    /// An upper estimate of the SPIR-V words one use emits.
    pub fn words(&self) -> u32 {
        WORDS
    }
}

/// The integer bits of constant `v`, zero-extended from its width; `None` for any other value.
fn constant_bits(cx: &SelectCx<'_>, v: Value) -> Option<u64> {
    match cx.constant(v)? {
        LoweredConst::Scalar { bits, .. } => Some(bits),
        _ => None,
    }
}

/// A non-pointer operand: an instruction or parameter value, or a constant the arith family
/// materializes.
fn value(cx: &SelectCx<'_>, v: Value) -> Result<ValueId, Error> {
    if matches!(v, Value::Const(_)) && cx.constant(v).is_none() {
        return Err(refuse());
    }
    cx.values
        .id(v)
        .ok_or_else(|| cx.invariant("an encoder operand out of the value range"))
}

/// The Metal address of buffer pointer `v`, planned.
fn address(cx: &mut SelectCx<'_>, v: Value, m: &mut Meter) -> Result<Address, Error> {
    if pointer::null_constant(cx, v) {
        return Ok(Address::Null);
    }
    match cx.memory.repr(v) {
        Some(Repr::BufPtr { wide: true, .. }) => Ok(Address::Wide(dynamic::plan_entry(cx, v)?)),
        Some(Repr::BufPtr { wide: false, .. }) => {
            let (targets, _, in_blob) = buffer_roots(cx, v, m)?;
            // A pointer into the constant blob has no Metal address.
            if in_blob {
                return Err(refuse());
            }
            dynamic::plan_static_guest(cx, &targets, Feature::GpuEncodedIndirectCommands)?;
            Ok(Address::Static {
                ptr: operand(cx, v)?,
                targets,
            })
        }
        _ => Err(refuse()),
    }
}

/// The refusal of every operation on a device that cannot write the records: one without device
/// addresses and `shaderInt64`, or a vertex or fragment function's without its stage's store
/// feature.
fn needs(cx: &SelectCx<'_>) -> Result<(), Error> {
    let f = cx.profile.fields();
    if f.buffer_access != BufferAccess::Addressed || !f.leaf.int64 {
        return Err(Unsupported::needs_any_of(
            Feature::GpuEncodedIndirectCommands,
            vec![cx.profile.addressed_needs()],
        )
        .into());
    }
    let enabled = &f.enabled.features;
    if let Some(feature) = dynamic::stage_stores(cx.stage).find(|f| !enabled.contains(f)) {
        return Err(Unsupported::needs(
            Feature::GpuEncodedIndirectCommands,
            Capability::Feature(feature),
        )
        .into());
    }
    Ok(())
}

/// Select the strategy of `inst` under the identity domains: an indirect-command operation's
/// writes, or its refusal; `None` for any other instruction, or on another profile.
pub fn select(cx: &mut SelectCx<'_>, inst: &Inst, m: &mut Meter) -> Result<Option<Command>, Error> {
    if !super::identity(cx.profile) {
        return Ok(None);
    }
    let InstKind::Call(call) = &inst.kind else {
        return Ok(None);
    };
    let Callee::Air {
        op: AirIntrinsic::IndirectCommand(op),
        ..
    } = call.callee
    else {
        return Ok(None);
    };
    let args = &call.args;
    let arg = |i: usize| -> Result<Value, Error> {
        args.get(i).copied().ok_or_else(|| {
            Error::Internal(InternalError::new(
                InternalKind::Invariant,
                "an encoder operation without its operands",
            ))
        })
    };
    // The command must be an object word: any other command handle is its family's refusal.
    let records = arg(0)?;
    if !matches!(records, Value::Inst(i) if cx.families.access.objects.contains(&i)) {
        return Err(Unsupported::not_implemented(Feature::ArgumentBufferObjects).into());
    }
    let unbuilt =
        || -> Error { Unsupported::not_implemented(Feature::GpuEncodedIndirectCommands).into() };
    let write = match op {
        IndirectCommand::SetBuffer { stage, strided, .. }
            if matches!(stage, CommandStage::Vertex | CommandStage::Fragment) =>
        {
            // The attribute stride form, lowered for the pipeline's own stride alone.
            if strided && constant_bits(cx, arg(3)?) != Some(u64::MAX) {
                return Err(unbuilt());
            }
            needs(cx)?;
            Write::Buffer {
                fragment: stage == CommandStage::Fragment,
                address: address(cx, arg(2)?, m)?,
                slot: value(cx, arg(if strided { 4 } else { 3 })?)?,
            }
        }
        IndirectCommand::DrawIndexedPrimitives { wide, .. } => {
            // `MTLPrimitiveTypeTriangle`: an execution draws a triangle list.
            if constant_bits(cx, arg(2)?) != Some(3) {
                return Err(
                    Unsupported::not_implemented(Feature::GpuEncodedIndirectCommands)
                        .naming(DiagName::new(
                            "a draw of another primitive type than triangles",
                        ))
                        .into(),
                );
            }
            needs(cx)?;
            Write::Draw {
                count: value(cx, arg(3)?)?,
                indices: address(cx, arg(4)?, m)?,
                wide,
                instances: value(cx, arg(5)?)?,
                base_vertex: value(cx, arg(6)?)?,
                base_instance: value(cx, arg(7)?)?,
            }
        }
        IndirectCommand::ResetRenderCommand => {
            needs(cx)?;
            Write::Reset
        }
        _ => return Err(unbuilt()),
    };
    Ok(Some(Command {
        records: operand(cx, records)?,
        index: value(cx, arg(1)?)?,
        write,
    }))
}

// ---- Emission ----

/// `x` (a `u32`) as a `u64`.
fn wide(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
    let u64_ = fb.module().ty(SType::Int {
        width: 64,
        signed: false,
    })?;
    fb.convert(ConvOp::UConvert, x, u64_)
}

/// The `u32` of the header at `records` (a `u64`) plus `at`.
fn header(fb: &mut FunctionBuilder<'_>, records: SValue, at: u32) -> BResult<SValue> {
    let at = k64(fb, u64::from(at))?;
    let p = fb.binary(BinOp::IAdd, records, at)?;
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let place = fb.device_place_read_only(p, u32_)?;
    fb.load(
        place,
        MemAccess {
            aligned: Some(4),
            ..MemAccess::default()
        },
    )
}

/// The Metal address of `a`, a `u64`.
fn emit_address(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, a: &Address) -> BResult<SValue> {
    match a {
        Address::Null => k64(fb, 0),
        Address::Wide(p) => {
            let p = cx.operand(fb, *p)?;
            dynamic::guest_address(cx, fb, p)
        }
        Address::Static { ptr, targets } => {
            let fat = cx.operand(fb, *ptr)?;
            dynamic::static_guest_address(cx, fb, targets, fat)
        }
    }
}

/// Emit command `c`'s writes, where its index is below the capacity (and its slot below its
/// stage's count).
pub fn emit(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>, c: &Command) -> BResult<()> {
    let records = cx.operand(fb, c.records)?;
    let capacity = header(fb, records, Records::CAPACITY_AT)?;
    let vertex_slots = header(fb, records, Records::VERTEX_SLOTS_AT)?;
    let fragment_slots = header(fb, records, Records::FRAGMENT_SLOTS_AT)?;
    let stride = header(fb, records, Records::STRIDE_AT)?;
    let index = cx.operand(fb, c.index)?;
    let mut valid = fb.binary(BinOp::ULessThan, index, capacity)?;
    // The command's first byte: the records past the header, `index` strides on.
    let i = wide(fb, index)?;
    let s = wide(fb, stride)?;
    let scaled = fb.binary(BinOp::IMul, i, s)?;
    let first = k64(fb, u64::from(Records::HEADER_BYTES))?;
    let past = fb.binary(BinOp::IAdd, records, first)?;
    let command = fb.binary(BinOp::IAdd, past, scaled)?;
    let eight = k32(fb, Records::SLOT_BYTES)?;
    let slots = fb.binary(BinOp::IAdd, vertex_slots, fragment_slots)?;
    let index_at = fb.binary(BinOp::IMul, slots, eight)?;
    let index_at = wide(fb, index_at)?;
    let token = fb.binary(BinOp::IAdd, command, index_at)?;
    // Each store: its address (a `u64`), its value, and its bytes (4 or 8).
    let mut stores: Vec<(SValue, SValue, u8)> = Vec::new();
    match &c.write {
        Write::Buffer {
            fragment,
            address,
            slot,
        } => {
            let slot = cx.operand(fb, *slot)?;
            let (count, first) = match fragment {
                false => (vertex_slots, None),
                true => (fragment_slots, Some(vertex_slots)),
            };
            let bound = fb.binary(BinOp::ULessThan, slot, count)?;
            valid = fb.binary(BinOp::LogicalAnd, valid, bound)?;
            let at = match first {
                None => slot,
                Some(v) => fb.binary(BinOp::IAdd, v, slot)?,
            };
            let at = fb.binary(BinOp::IMul, at, eight)?;
            let at = wide(fb, at)?;
            let to = fb.binary(BinOp::IAdd, command, at)?;
            let address = emit_address(cx, fb, address)?;
            let bias = k64(fb, SLOT_ADDRESS_BIAS)?;
            let word = fb.binary(BinOp::ISub, address, bias)?;
            stores.push((to, word, 8));
        }
        Write::Draw {
            count,
            indices,
            wide: wide_indices,
            instances,
            base_vertex,
            base_instance,
        } => {
            let count = cx.operand(fb, *count)?;
            let address = emit_address(cx, fb, indices)?;
            stores.push((token, address, 8));
            let (unit, ty) = match wide_indices {
                true => (4, Records::INDEX_UINT32),
                false => (2, Records::INDEX_UINT16),
            };
            let unit = k32(fb, unit)?;
            let bytes = fb.binary(BinOp::IMul, count, unit)?;
            let ty = k32(fb, ty)?;
            let zero = k32(fb, 0)?;
            let draw = [
                count,
                cx.operand(fb, *instances)?,
                zero,
                cx.operand(fb, *base_vertex)?,
                cx.operand(fb, *base_instance)?,
            ];
            let words = [(8, bytes), (12, ty)]
                .into_iter()
                .chain((draw.into_iter().enumerate()).map(|(word, x)| (20 + 4 * word as u32, x)));
            for (at, x) in words {
                let at = k64(fb, u64::from(at))?;
                let to = fb.binary(BinOp::IAdd, token, at)?;
                stores.push((to, x, 4));
            }
        }
        Write::Reset => {
            let at = k64(fb, u64::from(Records::RESET_INDEX_AT))?;
            let at = fb.binary(BinOp::IAdd, records, at)?;
            let u64_ = fb.module().ty(SType::Int {
                width: 64,
                signed: false,
            })?;
            let place = fb.device_place_read_only(at, u64_)?;
            let reset = fb.load(
                place,
                MemAccess {
                    aligned: Some(8),
                    ..MemAccess::default()
                },
            )?;
            stores.push((token, reset, 8));
            let ty = k32(fb, Records::INDEX_UINT32)?;
            let zero = k32(fb, 0)?;
            let words = [(8, zero), (12, ty)]
                .into_iter()
                .chain((0..5u32).map(|word| (20 + 4 * word, zero)));
            for (at, x) in words {
                let at = k64(fb, u64::from(at))?;
                let to = fb.binary(BinOp::IAdd, token, at)?;
                stores.push((to, x, 4));
            }
        }
    }
    let write: ArmFn<'_, ()> = Box::new(move |fb, _| {
        for (to, x, bytes) in stores {
            let ty = fb.module().ty(SType::Int {
                width: 8 * bytes,
                signed: false,
            })?;
            let place = fb.device_place(to, ty)?;
            fb.store(
                place,
                x,
                MemAccess {
                    aligned: Some(u32::from(bytes)),
                    ..MemAccess::default()
                },
            )?;
        }
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), valid, write, None)?;
    Ok(())
}
