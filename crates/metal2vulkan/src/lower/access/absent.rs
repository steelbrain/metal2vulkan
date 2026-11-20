//! Reads of an argument the pipeline's function constants removed.
//!
//! Specialization makes each observed read of a texture argument the constants removed (a read,
//! sample, gather, query, or atomic), or each sample through a removed sampler argument,
//! [`FunctionConstant::AbsentRead`] (`air::canon::interface`). Metal gives such a read no value:
//! its compiler dropped a store of one, made a fragment color computed from one all-zero, and,
//! beside a removed texture table, dropped a kernel's whole body, so no one translation is Metal's.
//! The read writes no status: it takes the zero of its result's type (a read's zeros and a
//! query's size of 0, which the nil texture, set 1's element 0, answers too) and fetches nothing,
//! a write of a removed texture is dropped, and the stage takes no `ExecutionStatus` buffer, push
//! field or fact for it.
//!
//! A load or atomic of threadgroup memory the constants removed is the same read
//! ([`FunctionConstant::AbsentMemory`]), and so is a texture operation on a removed texture or
//! sampler that lowering reaches through memory, an aggregate, a table, or a merge
//! (`texture::AbsentArgument`), which shares [`require`] and [`mark`].
//!
//! Where its reads mark, the stage states [`RuntimeFact::AbsentArgumentRead`] and
//! [`RuntimeFact::ExecutionStatus`]. It reaches the buffer by its device address
//! ([`PushField::ExecutionStatusAddress`]) where the device addresses buffers
//! (`bufferDeviceAddress` and `shaderInt64`), and through a storage-buffer binding of its own
//! (`SyntheticBinding::ExecutionStatus`, [`StatusBinding`]) where it does not, one more storage
//! descriptor of the stage: a stage past a per-stage limit with it is refused as `DescriptorCount`
//! (`lower::select`), naming the limit. A vertex or fragment function needs its stage's stores
//! and atomics; a device without them refuses the read as `FunctionConstantArgument`, naming what
//! it needs.

use crate::air::intrinsics::{AirIntrinsic, FunctionConstant};
use crate::air::ir::{Callee, Inst, InstId, InstKind, Type, Value, ValueId};
use crate::bind::status::{StatusBits, StatusWord};
use crate::device::profile::BufferAccess;
use crate::device::requirements::VkFeature;
use crate::error::Error;
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::leaf::k32;
use crate::memory::roots::k64;
use crate::memory::views::{self, View, ViewKind};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, SyntheticBinding};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::DemoteTy;
use crate::spirv::ir::{AtomicOp, BinOp, Constant, Scope, Semantics, Type as SType};

/// An absent read ([`FunctionConstant::AbsentRead`]): the status write where it marks, then zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AbsentRead {
    /// The lowered representation of the read's result; `None` for a read of no result.
    pub result: Option<DemoteTy>,
    /// Whether it sets the status bit ([`require`]).
    pub marks: bool,
}

impl AbsentRead {
    /// The census name.
    pub fn name(&self) -> &'static str {
        "access.absent_read"
    }

    /// An upper estimate of the SPIR-V words one read emits: where it marks, the push field's load
    /// and the word's address and pointer (or the binding's access chain), the bit, and the atomic
    /// `OR`; the zero is a constant.
    pub fn words(&self) -> u32 {
        if self.marks {
            32
        } else {
            0
        }
    }
}

/// Select the read of `inst` when it is a [`FunctionConstant::AbsentRead`] or
/// [`FunctionConstant::AbsentMemory`] call, `None` otherwise: the stage writes the
/// `ExecutionStatus` buffer ([`state_status`]).
pub fn select(cx: &mut SelectCx<'_>, at: InstId, inst: &Inst) -> Result<Option<AbsentRead>, Error> {
    let InstKind::Call(call) = &inst.kind else {
        return Ok(None);
    };
    let Callee::Air {
        op:
            AirIntrinsic::FunctionConstant(
                FunctionConstant::AbsentRead(_) | FunctionConstant::AbsentMemory(_),
            ),
        ..
    } = call.callee
    else {
        return Ok(None);
    };
    let marks = require(cx)?;
    let result = match cx.entry.module.types().get(inst.ty) {
        Some(Type::Void) => None,
        _ => Some(cx.lowered_repr(Value::Inst(at), inst.ty)?),
    };
    Ok(Some(AbsentRead { result, marks }))
}

/// What an absent read needs, stated for the stage, and whether it marks the status: nothing, and
/// it does not mark, an absent read reading as nil. The texture and sampler families' arms on the
/// absent handle need the same.
pub fn require(_cx: &mut SelectCx<'_>) -> Result<bool, Error> {
    Ok(false)
}

/// What the stage lacks to write the command buffer's `ExecutionStatus` words. Each refusal
/// names its own feature at its site ([`require`], `sync::barrier::require_seq_cst`), so the
/// refusal surface reads each one's condition (`scripts/refusal-surface`).
pub enum StatusShort {
    /// The stage's stores and atomics, which the device has not enabled.
    Stores(VkFeature),
}

/// What the stage lacks to write the command buffer's `ExecutionStatus` words, or `None` where
/// it can ([`state_status`] then states how it reaches the buffer).
pub fn status_short(cx: &SelectCx<'_>) -> Option<StatusShort> {
    let enabled = &cx.profile.fields().enabled.features;
    super::dynamic::stage_stores(cx.stage)
        .find(|f| !enabled.contains(f))
        .map(StatusShort::Stores)
}

/// Why a stage of a device that does not address buffers binds the command buffer's
/// `ExecutionStatus` buffer (`SyntheticBinding::ExecutionStatus`), one more storage descriptor of
/// the stage, which counts against its per-stage limits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum StatusBinding {
    /// Only an argument buffer's sampler's cubic test marks the buffer
    /// (`sampler::select::cubic_reach`): where the stage's descriptors pass a per-stage limit with
    /// it, [`finish`] gives it back, and the stage marks nothing, as a fragment stage does (the
    /// planner refuses a listed cubic sampler beside the stage).
    CubicMark,
    /// An absent read or a sequentially consistent fence writes the buffer: a stage past a
    /// per-stage limit with it is refused as `DescriptorCount`.
    Required,
}

/// State how the stage reaches the `ExecutionStatus` buffer, which [`status_short`] found it can
/// write, for `need`: its device address where the device addresses buffers, else its binding
/// ([`StatusBinding`]), counted once in the stage's descriptors.
pub fn state_status(cx: &mut SelectCx<'_>, need: StatusBinding) {
    cx.facts.insert(RuntimeFact::ExecutionStatus);
    if cx.profile.fields().buffer_access == BufferAccess::Addressed {
        cx.push.insert(PushField::ExecutionStatusAddress);
        return;
    }
    let access = &mut cx.families.access;
    if access.status_binding.is_none() {
        cx.descriptors.storage_buffers = cx.descriptors.storage_buffers.saturating_add(1);
    }
    access.status_binding = access.status_binding.max(Some(need));
}

/// Give back a binding of the `ExecutionStatus` buffer that only the cubic mark holds
/// ([`StatusBinding::CubicMark`]) where the stage's descriptors pass a per-stage limit with it:
/// the stage then marks nothing, and states no cubic reach, so the planner refuses a listed cubic
/// sampler beside it (`SamplerBicubicFilter`), as before the binding. Called once the slots'
/// forms are final, before the limits are checked (`lower::select`).
///
/// The mark is one the stage may need, not one it does: only a cubic sampler that a command
/// later lists in the argument buffer needs it, and that sampler then meets the planner's typed
/// refusal, so giving the binding back hides nothing. Refusing the stage instead would refuse
/// every stage near its limit that samples through an argument buffer, cubic sampler or not.
pub fn finish(cx: &mut SelectCx<'_>) {
    if cx.families.access.status_binding != Some(StatusBinding::CubicMark) {
        return;
    }
    if cx
        .descriptors
        .exceeded(&cx.profile.fields().limits)
        .is_none()
    {
        return;
    }
    cx.families.access.status_binding = None;
    cx.descriptors.storage_buffers = cx.descriptors.storage_buffers.saturating_sub(1);
    cx.families.sampler.cubic_reach = false;
    cx.facts.remove(&RuntimeFact::ExecutionStatus);
    cx.facts
        .remove(&RuntimeFact::CubicArgumentSampler { stage: cx.stage });
}

/// The binding symbol of `stage`'s `ExecutionStatus` buffer.
pub fn symbol(stage: Stage) -> ResourceSlot {
    ResourceSlot {
        stage,
        kind: ResourceKind::Synthetic(SyntheticBinding::ExecutionStatus),
        index: 0,
    }
}

/// Declare the stage's `ExecutionStatus` view, a storage word view at its binding symbol, where
/// it binds the buffer ([`StatusBinding`]).
pub fn declare(
    mb: &mut ModuleBuilder,
    stage: Stage,
    plan: &super::AccessPlan,
) -> BResult<Option<View>> {
    match plan.status_binding {
        Some(_) => Ok(Some(views::declare_at(
            mb,
            symbol(stage),
            ViewKind::Storage,
        )?)),
        None => Ok(None),
    }
}

/// Emit absent read `s`, whose result is `vid`: set the bit where it marks, and define the result
/// as zero.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    vid: ValueId,
    s: &AbsentRead,
) -> BResult<()> {
    if s.marks {
        mark(cx, fb)?;
    }
    let Some(d) = &s.result else {
        return Ok(());
    };
    let Some(ty) = crate::lower::arith::spirv_type(fb.module(), d)? else {
        return Err(cx.fail("an absent read without a lowered result type"));
    };
    let zero = fb.module().constant(Constant::Null { ty })?;
    let zero = fb.module().const_value(zero)?;
    cx.define(vid, zero);
    Ok(())
}

/// Set `StatusBits::ABSENT_ARGUMENT_READ` in the command buffer's `ExecutionStatus` word, where
/// the open block runs (a relaxed device-scope `OR`); [`require`] stated what it needs.
pub fn mark(cx: &mut EmitCx<'_>, fb: &mut FunctionBuilder<'_>) -> BResult<()> {
    mark_bits(cx, fb, StatusBits::ABSENT_ARGUMENT_READ)
}

/// Whether the stage can write the command buffer's `ExecutionStatus` word for the cubic mark
/// ([`require`]'s needs: the stage's stores and atomics), stating how it reaches the buffer where
/// it can ([`StatusBinding::CubicMark`]).
pub fn status_writable(cx: &mut SelectCx<'_>) -> bool {
    let writable = status_short(cx).is_none();
    if writable {
        state_status(cx, StatusBinding::CubicMark);
    }
    writable
}

/// Set `bits` in the command buffer's `ExecutionStatus` word, where the open block runs (a
/// relaxed device-scope `OR`); the stage stated how it reaches the buffer ([`require`],
/// [`status_writable`]).
pub fn mark_bits(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    bits: StatusBits,
) -> BResult<()> {
    let word = status_word(cx, fb, StatusWord::Status)?;
    let bit = k32(fb, bits.0)?;
    fb.atomic(AtomicOp::Or, word, Scope::Device, Semantics::RELAXED, bit)?;
    Ok(())
}

/// The place of the command buffer's `ExecutionStatus` word `w` (a `u32`): through the stage's
/// binding of the buffer where it has one ([`declare`]), else by the buffer's device address; the
/// stage stated which ([`state_status`]).
pub fn status_word(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    w: StatusWord,
) -> BResult<crate::spirv::ids::Value> {
    if let Some(view) = cx.decl.families.access.status {
        let word = k32(fb, w.byte_offset() / 4)?;
        return match views::word_ptr(fb, view, word)? {
            Some(ptr) => Ok(ptr),
            None => Err(cx.fail("an ExecutionStatus view that is not a storage view")),
        };
    }
    let status = cx.push_field(fb, PushField::ExecutionStatusAddress)?;
    let address = match w.byte_offset() {
        0 => status,
        at => {
            let off = k64(fb, u64::from(at))?;
            fb.binary(BinOp::IAdd, status, off)?
        }
    };
    let u32_ty = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    fb.device_place(address, u32_ty)
}
