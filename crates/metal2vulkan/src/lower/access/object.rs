//! Indirect-command objects: an argument buffer's command-buffer or pipeline-state member, and the
//! encoder operations that take one.
//!
//! **Object words.** An object's handle load (`memory::provenance::object_handle_load`) reads the
//! member's field, 8 bytes, as data ([`ObjectWord`], a `u64`): a command buffer's word is its
//! resource ID, Metal's `gpuResourceID`; a pipeline state's word is the template's token. A handle
//! load is admitted only when every use of its value is an encoder operand of its class
//! ([`plan_objects`]): the command operand of an encoder operation, the pipeline operand of
//! `air.set_pipeline_state_{compute,render}_command`, or a copy's source command buffer. Any other
//! use (an integer, a compare, a store, a merge) leaves the load refused
//! (`ArgumentBufferObjects`, `pointer::select`), so no other instruction observes the word.
//!
//! **Encoder operations** ([`select`]). This module lowers none. Under the identity domains the
//! operations `icb` lowers (a render command's buffer setters, its indexed triangle draw and a
//! reset) are selected there, before this module; every other encoder operation, and every one
//! under the resident-table domains, is `Unsupported{GpuEncodedIndirectCommands, NotImplemented}`,
//! and one whose command is not an object word is `Unsupported{ArgumentBufferObjects,
//! NotImplemented}`.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use crate::air::intrinsics::{AirIntrinsic, IndirectCommand};
use crate::air::ir::{Callee, Inst, InstId, InstKind, Value, ValueId};
use crate::budget::{tree_bytes, Meter};
use crate::error::{Error, Feature, Unsupported};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::provenance::{self, HandleClass, Repr};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, ScalarTy};

use super::dynamic::{self, Source};
use super::AccessPlan;

/// The handle load of an indirect-command object: its field's word, a `u64`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectWord(pub Source);

impl ObjectWord {
    /// The census name (`access.object.word`) in the finished plan `slots`, qualified by what the
    /// object's load reaches ([`super::SideReach`]).
    pub fn name(&self, slots: &AccessPlan) -> Cow<'static, str> {
        self.0.reach(slots).qualify("access.object.word")
    }

    /// An upper estimate of the SPIR-V words one use emits: a leaf access.
    pub fn words(&self) -> u32 {
        self.0.words()
    }
}

/// Whether `class` is an indirect-command object's.
fn object_class(class: HandleClass) -> bool {
    matches!(
        class,
        HandleClass::Command | HandleClass::ComputePipelineState | HandleClass::RenderPipelineState
    )
}

/// The encoder operation of call `call`, if it is one.
fn encoder(call: &crate::air::ir::Call) -> Option<IndirectCommand> {
    match call.callee {
        Callee::Air {
            op: AirIntrinsic::IndirectCommand(op),
            ..
        } => Some(op),
        _ => None,
    }
}

/// Whether operand `i` of encoder `op` may be an object word of `class`: the command operand,
/// the pipeline operand of a compute or render pipeline setter, or a copy's source command
/// buffer.
fn encoder_operand(op: IndirectCommand, i: usize, class: HandleClass) -> bool {
    match (i, class) {
        (0, HandleClass::Command) => true,
        (2, HandleClass::ComputePipelineState) => op == IndirectCommand::SetComputePipeline,
        (2, HandleClass::RenderPipelineState) => op == IndirectCommand::SetRenderPipeline,
        (2, HandleClass::Command) => op == IndirectCommand::CopyComputeCommand,
        _ => false,
    }
}

/// Plan the function's object words, before any instruction is selected: each handle load of an
/// indirect-command object (`provenance::object_handle_load`) whose every use is an encoder operand
/// of its class (`encoder_operand`).
pub fn plan_objects(cx: &mut SelectCx<'_>, m: &mut Meter) -> Result<(), Error> {
    let body = cx.body;
    let entry = tree_bytes(1, std::mem::size_of::<(InstId, HandleClass)>());
    let mut words: BTreeMap<InstId, HandleClass> = BTreeMap::new();
    for (id, inst) in body.insts.iter() {
        m.tick(1)?;
        let InstKind::Load {
            ptr,
            volatile: false,
            atomic: None,
            ..
        } = &inst.kind
        else {
            continue;
        };
        let Some(Repr::Handle { class, .. }) = cx.memory.repr(Value::Inst(id)) else {
            continue;
        };
        if !object_class(class) {
            continue;
        }
        let (Some(roots), Some(range), Some(bytes)) = (
            cx.memory.roots_of(*ptr),
            cx.memory.offset(*ptr),
            cx.memory.layouts.store_size(inst.ty),
        ) else {
            continue;
        };
        if provenance::object_handle_load(cx.entry, &cx.memory.abi, roots, range, bytes).is_some() {
            m.charge(entry)?;
            words.insert(id, class);
        }
    }
    if words.is_empty() {
        return Ok(());
    }
    for (_, inst) in body.insts.iter() {
        m.tick(1)?;
        match &inst.kind {
            InstKind::Call(call) if encoder(call).is_some() => {
                let op = encoder(call).ok_or_else(|| cx.invariant("an encoder vanished"))?;
                for (i, a) in call.args.iter().enumerate() {
                    if let Value::Inst(l) = a {
                        if words.get(l).is_some_and(|c| !encoder_operand(op, i, *c)) {
                            words.remove(l);
                        }
                    }
                }
            }
            kind => kind.for_each_value(|v| {
                if let Value::Inst(l) = v {
                    words.remove(&l);
                }
            }),
        }
    }
    cx.families.access.objects = words.into_keys().collect::<BTreeSet<_>>();
    Ok(())
}

/// The lowered representation of an object word ([`plan_objects`]): a `u64`; `None` for any
/// other value.
pub fn lowered_pointer(cx: &SelectCx<'_>, v: Value) -> Option<DemoteTy> {
    match v {
        Value::Inst(i) if cx.families.access.objects.contains(&i) => {
            Some(DemoteTy::Scalar(ScalarTy::Int { bits: 64 }))
        }
        _ => None,
    }
}

/// Select the strategy of `inst`: an object word's handle load, or the refusal of an encoder
/// operation (see the module docs); `None` for any other instruction.
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    m: &mut Meter,
) -> Result<Option<ObjectWord>, Error> {
    match &inst.kind {
        InstKind::Load {
            ptr,
            align,
            volatile: false,
            atomic: None,
            ..
        } if cx.families.access.objects.contains(&at) => Ok(Some(ObjectWord(dynamic::plan_word(
            cx, *ptr, inst.ty, *align, m,
        )?))),
        InstKind::Call(call) if encoder(call).is_some() => {
            let command = call.args.first();
            if !matches!(command, Some(Value::Inst(i)) if cx.families.access.objects.contains(i)) {
                return Err(Unsupported::not_implemented(Feature::ArgumentBufferObjects).into());
            }
            Err(Unsupported::not_implemented(Feature::GpuEncodedIndirectCommands).into())
        }
        _ => Ok(None),
    }
}

/// Emit object word `w`, defining value `vid`.
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    vid: ValueId,
    w: &ObjectWord,
) -> BResult<()> {
    let word = dynamic::emit_word(cx, fb, &w.0)?;
    cx.define(vid, word);
    Ok(())
}
