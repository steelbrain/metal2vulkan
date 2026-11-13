//! Whether a finished module is inert
//! ([`ModuleStats::inert`](crate::pipeline::artifact::ModuleStats::inert)):
//! every invocation of it ends, and none leaves anything behind its workgroup.
//!
//! An invocation ends where the module holds no loop: SPIR-V has no recursion (the verifier's
//! call-graph rule) and no backward edge but a loop's, so every path through every function is
//! finite. It leaves nothing behind where no instruction or terminator of the module, in any of its
//! functions, can:
//!
//! - store, write atomically or exchange through a pointer to memory other than `Function`,
//!   `Private` or `Workgroup` storage (an invocation's own and its workgroup's, which end with the
//!   workgroup), or through a pointer whose type is not one of those;
//! - write an image;
//! - emit an output of its stage, or change which of its outputs land: a mesh function's
//!   outputs, a geometry function's vertices and primitives, a task function's launch, a
//!   fragment's demotion, kill or termination;
//! - do anything the classification below does not name as reading or computing alone: an
//!   interlock and a ray query count as effects, so a module holding one is never inert.
//!
//! Every instruction and terminator form is named in an exhaustive `match`, so a form added to the
//! IR does not compile until it is classified here.

use crate::budget::Meter;
use crate::spirv::ids::Value;
use crate::spirv::ir::{Inst, Merge, StorageClass, Terminator, Type, TypedModule};

use super::VerifyError;

/// Whether `m` is inert (the module docs), metered one step a block and one an instruction.
pub(super) fn inert(m: &TypedModule, meter: &mut Meter) -> Result<bool, VerifyError> {
    if !m.loops.is_empty() {
        return Ok(false);
    }
    for block in &m.blocks {
        meter.tick(
            u32::try_from(block.insts.len())
                .unwrap_or(u32::MAX)
                .saturating_add(1),
        )?;
        if matches!(block.merge, Some(Merge::Loop { .. })) || !ends_inert(&block.term) {
            return Ok(false);
        }
        if !block.insts.iter().all(|node| leaves_nothing(m, &node.inst)) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether the terminator leaves the invocation's outputs and memory as they are.
fn ends_inert(term: &Terminator) -> bool {
    match term {
        Terminator::Branch(_)
        | Terminator::CondBranch { .. }
        | Terminator::Switch { .. }
        | Terminator::Return
        | Terminator::ReturnValue(_)
        | Terminator::Unreachable => true,
        Terminator::Kill | Terminator::TerminateInvocation | Terminator::EmitMeshTasks { .. } => {
            false
        }
    }
}

/// Whether the instruction writes nothing past the invocation's own and its workgroup's memory and
/// emits nothing.
fn leaves_nothing(m: &TypedModule, inst: &Inst) -> bool {
    match inst {
        Inst::Store { ptr, .. }
        | Inst::AtomicStore { ptr, .. }
        | Inst::Atomic { ptr, .. }
        | Inst::AtomicCompareExchange { ptr, .. } => local(m, *ptr),
        Inst::ImageWrite { .. }
        | Inst::BeginInterlock
        | Inst::EndInterlock
        | Inst::DemoteToHelperInvocation
        | Inst::SetMeshOutputs { .. }
        | Inst::EmitVertex
        | Inst::EndPrimitive
        | Inst::RayQuery(_) => false,
        Inst::Load { .. }
        | Inst::Unary { .. }
        | Inst::Binary { .. }
        | Inst::Select { .. }
        | Inst::BitFieldInsert { .. }
        | Inst::Fma { .. }
        | Inst::BitFieldExtract { .. }
        | Inst::Convert { .. }
        | Inst::CompositeConstruct { .. }
        | Inst::CompositeExtract { .. }
        | Inst::CompositeInsert { .. }
        | Inst::VectorExtractDynamic { .. }
        | Inst::VectorInsertDynamic { .. }
        | Inst::VectorShuffle { .. }
        | Inst::Undef
        | Inst::AccessChain { .. }
        | Inst::ArrayLength { .. }
        | Inst::Call { .. }
        | Inst::AtomicLoad { .. }
        | Inst::ImageTexelPointer { .. }
        | Inst::ControlBarrier { .. }
        | Inst::MemoryBarrier { .. }
        | Inst::SampledImage { .. }
        | Inst::Image(_)
        | Inst::ImageSample { .. }
        | Inst::ImageFetch { .. }
        | Inst::ImageGather { .. }
        | Inst::ImageDrefGather { .. }
        | Inst::ImageRead { .. }
        | Inst::ImageSparseTexelsResident(_)
        | Inst::ImageQuery { .. }
        | Inst::ImageQueryLod { .. }
        | Inst::Derivative { .. }
        | Inst::Group { .. }
        | Inst::Glsl { .. }
        | Inst::IsHelperInvocation => true,
    }
}

/// Whether `ptr` points into the invocation's own or its workgroup's memory. A pointer whose type
/// is not found is not.
fn local(m: &TypedModule, ptr: Value) -> bool {
    let Ok(def) = m.value(ptr) else {
        return false;
    };
    matches!(
        m.types().get(def.ty),
        Ok(Type::Pointer {
            class: StorageClass::Function | StorageClass::Private | StorageClass::Workgroup,
            ..
        })
    )
}
