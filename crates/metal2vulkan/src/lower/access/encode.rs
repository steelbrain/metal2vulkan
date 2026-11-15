//! GPU-encoded indirect commands: an encoder's operations write the host's command records
//! (`HostRecordsV1`, `crate::indirect_commands`) through device addresses.
//!
//! **Object words.** An argument buffer's command-buffer or pipeline-state member is an
//! indirect-command object. Its handle load (`memory::provenance::object_handle_load`) reads the
//! member's field, 8 bytes, as data ([`EncodeStrategy::Word`], a `u64`): a command buffer's word is
//! its resource ID, Metal's `gpuResourceID`, whose records lie at the guest address
//! `indirect_commands::records_address` of it, `2^63 + id * 2^31`; a pipeline state's word is the
//! template's host-registered token. A handle load is admitted only when every use of its value is
//! an encoder operand of its class ([`plan_objects`]): the command operand of an encoder operation,
//! or the pipeline operand of `air.set_pipeline_state_{compute,render}_command`. Any other use (an
//! integer, a compare, a store, a merge) leaves the load refused as before
//! (`ArgumentBufferObjects`, `pointer::select`), so no other instruction observes the word.
//!
//! **Operations** ([`EncodeStrategy::Command`]). The command word's records address is searched
//! in the resident set for the allocation that starts there (`dynamic::search_start`), and the
//! record buffer is accessed through its root entry, contained: an access through root 0 (an
//! address at which no resident allocation starts) is counted under isolation and lands in the
//! sink. Each operation reads the header's capacity, policy bits,
//! and binding counts; when the write is valid by the descriptor policy
//! (`indirect_commands::Policy`) it writes the command's fields, and otherwise it sets the header's
//! sticky fault flag (a relaxed atomic store of 1) and writes nothing, exactly as the old engine's
//! helpers do (`CommandEncoderOperation::helper`), so the host's `CommandBuffer::decode` fails the
//! run. A buffer setter writes its buffer's guest address (the pointer escapes, so it is wide) and,
//! for a kernel or vertex buffer, its attribute stride (a fragment, object or mesh buffer has
//! none; AIR 2.5's kernel or vertex setter takes none and writes the pipeline's own, all ones, as
//! the stride today's compiler spells MSL's plain setter with); a pipeline setter the pipeline
//! state's word; a dispatch its grid, local size, and action;
//! a draw its five arguments and action (on a buffer of mesh draws alone, nothing and no fault, as
//! a kernel's draw there draws nothing on Metal: `Policy::dropped_on`); a mesh draw its grid, its
//! object and mesh threadgroups' threads, and its action; an indexed draw its five arguments, its
//! index type (the overload's), its index buffer's guest address (wide, as a buffer setter's), and
//! its action; a patch draw its six arguments, its patch index and factor buffers' guest addresses
//! (0 for a null patch index buffer), its factor scale's `f32` bits, and its action; an indexed
//! patch draw a patch draw's fields and its control-point index buffer's guest address (0 for null)
//! where the record keeps it, under its own action; a barrier (of a compute or a render command)
//! its flag (set or cleared); a threadgroup-length setter the length at its binding, as set (a
//! multiple of 16 or not, as Metal keeps it); a stage-in region setter its six words; an imageblock
//! size setter its two `u16`s; and a reset (of a compute or render command) the record as
//! `CommandRecord::default` encodes it, in loops of `u64` stores over its command words, the
//! bindings below each stage's count in the header and the words past the bindings: a binding
//! past its count is zero in every record, as no write reaches one (`default_record`). A valid
//! operation (a copy too)
//! also stores 1 at its command's written flag past every record
//! (`indirect_commands::written_flag_offset`), contained as its fields are.
//!
//! **Copies** (`copy_command`). The source command buffer is an object word too, and its record
//! buffer is searched as the command's is. A copy is valid when the source index is below the
//! source's capacity and the two headers are of equal descriptors (policy bits and binding counts),
//! in which case it copies the whole record, `u64` by `u64` in a loop; or when the destination
//! inherits its pipeline and its buffers from a source that inherits neither and is otherwise of
//! the same descriptor (the destination binds no kernel buffer), in which case it copies every
//! `u64` but the pipeline's and the buffer bindings', the state the destination takes from the
//! encoder that executes it, as Metal's copy measured (M3 and M4). Any other copy's result depends
//! on Metal's private record layout (a copy between other descriptors, an index at either
//! capacity): it sets the destination's sticky fault and copies nothing.
//!
//! **Profile F** has no device addresses: every operation is `GpuEncodedIndirectCommands`, naming
//! what it lacks of `bufferDeviceAddress` and `shaderInt64`. A vertex or fragment function on a
//! device without the stage's store feature refuses an operation the same way, naming the
//! feature.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use crate::air::intrinsics::{AirIntrinsic, CommandStage, IndirectCommand};
use crate::air::ir::{AddrSpace, Callee, Inst, InstId, InstKind, Value, ValueId};
use crate::budget::{tree_bytes, Meter};
use crate::device::profile::BufferAccess;
use crate::device::requirements::VkFeature;
use crate::error::{Capability, Error, Feature, Unsupported};
use crate::indirect_commands::{
    layout, BufferStage, CommandEncoderOperation, Policy, BINDING_COUNT, DYNAMIC_STRIDE,
    INHERIT_BUFFERS, INHERIT_PIPELINE, RECORDS_ADDRESS_BASE, RECORDS_ADDRESS_SHIFT,
    WRITTEN_FLAG_BYTES,
};
use crate::lower::arith::{Form, Held};
use crate::lower::emit::EmitCx;
use crate::lower::select::SelectCx;
use crate::memory::leaf::{Leaf, Path};
use crate::memory::provenance::{self, HandleClass, Repr};
use crate::memory::roots::k64;
use crate::pipeline::sym::Stage;
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::FunctionBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ids::{DemoteTy, ScalarTy, Value as SValue};
use crate::spirv::ir::{
    BinOp, ConstBits, Constant, ConvOp, MemAccess, Scope, Semantics, Type as SType, UnOp,
};

use super::dynamic::{self, Side, Source};
use super::{operand, refuse, AccessPlan};

/// The encoder strategies.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodeStrategy {
    /// The handle load of an indirect-command object: its field's word, a `u64`.
    Word(Source),
    /// An encoder operation's record writes.
    Command(Box<Command>),
}

/// One encoder operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Command {
    /// The operation, in the host contract's terms (its policy and fields).
    pub op: CommandEncoderOperation,
    /// The command buffer's word: its resource ID, whose records lie at
    /// `indirect_commands::records_address` of it.
    pub command: ValueId,
    /// The command's index (`i32`).
    pub index: ValueId,
    pub fields: Fields,
}

/// What an operation writes into its command's record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fields {
    /// A buffer's guest address at `binding`, and for a kernel or vertex buffer its attribute
    /// stride (`i64`); a fragment, object or mesh buffer has none, nor has AIR 2.5's kernel or
    /// vertex setter.
    Buffer {
        address: Side,
        stride: Option<ValueId>,
        binding: ValueId,
    },
    /// The grid and local size (`<3 x i32>` each), and the action.
    Dispatch { grid: ValueId, local: ValueId },
    /// The pipeline state's word.
    Pipeline { word: ValueId },
    /// The barrier flag: 1 when `enabled`, else 0.
    Barrier { enabled: bool },
    /// A threadgroup length at `binding` (`i32` each).
    ThreadgroupLength { length: ValueId, binding: ValueId },
    /// The primitive type (written as Metal's encoding writes it: [`encoded_primitive_type`]),
    /// vertex start, vertex count, instance count, and base instance (`i32` each), and the action.
    Draw { arguments: [ValueId; 5] },
    /// The stage-in region's origin and size (`<3 x i32>` each).
    StageInRegion { origin: ValueId, size: ValueId },
    /// The imageblock's width and height (`<2 x i16>`, each lane held zero-extended in a `u32`).
    ImageblockSize { size: ValueId },
    /// The whole record, restored to its default.
    Reset,
    /// An indexed draw's primitive type (as [`Fields::Draw`]'s), index count, instance count,
    /// base vertex and base instance (`i32` each), its `MTLIndexType` (0 for 16-bit indices, 1
    /// for 32-bit ones), the index buffer's guest address, and the action.
    IndexedDraw {
        arguments: [ValueId; 5],
        index_type: u32,
        address: Side,
    },
    /// A patch draw's control points, patch start, patch count, instance count, base instance
    /// and factors' instance stride (`i32` each), its patch index buffer's guest address (0 for
    /// null) and its factor buffer's, its factor scale (`float`), and the action; an indexed
    /// patch draw's control-point index buffer's guest address too (0 for null).
    PatchDraw {
        arguments: [ValueId; 6],
        patch_indices: Side,
        control_point_indices: Option<Side>,
        factors: Side,
        scale: ValueId,
    },
    /// The record of command `source_index` (`i32`) of the source command buffer's word
    /// `source`.
    Copy {
        source: ValueId,
        source_index: ValueId,
    },
    /// A mesh draw's grid, and the threads of an object and of a mesh threadgroup (`<3 x i32>`
    /// each), and the action.
    MeshDraw {
        grid: ValueId,
        object: ValueId,
        mesh: ValueId,
    },
}

impl EncodeStrategy {
    /// The census name (`access.encode.<op>`) in the finished plan `slots`: a word load
    /// qualified by what the object's load reaches ([`super::SideReach`]).
    pub fn name(&self, slots: &AccessPlan) -> Cow<'static, str> {
        match self {
            EncodeStrategy::Word(s) => s.reach(slots).qualify("access.encode.word"),
            EncodeStrategy::Command(c) => match c.fields {
                Fields::Buffer { .. } => "access.encode.buffer",
                Fields::Dispatch { .. } => "access.encode.dispatch",
                Fields::Pipeline { .. } => "access.encode.pipeline",
                Fields::Barrier { .. } => "access.encode.barrier",
                Fields::ThreadgroupLength { .. } => "access.encode.threadgroup_length",
                Fields::Draw { .. } => "access.encode.draw",
                Fields::StageInRegion { .. } => "access.encode.stage_in_region",
                Fields::ImageblockSize { .. } => "access.encode.imageblock_size",
                Fields::Reset => "access.encode.reset",
                Fields::IndexedDraw { .. } => "access.encode.indexed_draw",
                Fields::PatchDraw {
                    control_point_indices: None,
                    ..
                } => "access.encode.patch_draw",
                Fields::PatchDraw {
                    control_point_indices: Some(_),
                    ..
                } => "access.encode.indexed_patch_draw",
                Fields::Copy { .. } => "access.encode.copy",
                Fields::MeshDraw { .. } => "access.encode.mesh_draw",
            }
            .into(),
        }
    }

    /// An upper estimate of the SPIR-V words one use emits: a word load is a leaf access; an
    /// operation is a search call, the record's root entry, three header loads and the policy test,
    /// and per field a contained store (a reset's loops: each its one store and the loop's own
    /// words, and its counts' header load and each stage's count, `RESET_WORDS`; a copy's: its
    /// load and store, the loop's words, and the source's search, header loads and tests,
    /// `COPY_WORDS`).
    pub fn words(&self) -> u32 {
        match self {
            EncodeStrategy::Word(s) => s.words(),
            EncodeStrategy::Command(c) => {
                let fields: u32 = match c.fields {
                    Fields::Buffer { stride, .. } => 1 + u32::from(stride.is_some()),
                    Fields::Dispatch { .. } => 7,
                    Fields::MeshDraw { .. } => 10,
                    Fields::Draw { .. } | Fields::StageInRegion { .. } => 6,
                    Fields::IndexedDraw { .. } => 8,
                    Fields::PatchDraw {
                        control_point_indices,
                        ..
                    } => 11 + u32::from(control_point_indices.is_some()),
                    Fields::Copy { .. } => 2,
                    Fields::Pipeline { .. } => 2,
                    Fields::Barrier { .. }
                    | Fields::ThreadgroupLength { .. }
                    | Fields::ImageblockSize { .. }
                    | Fields::Reset => 1,
                };
                let reset = match c.fields {
                    Fields::Reset => RESET_WORDS,
                    Fields::Copy { .. } => COPY_WORDS,
                    _ => 0,
                };
                COMMAND_WORDS
                    .saturating_add(FIELD_WORDS.saturating_mul(fields))
                    .saturating_add(reset)
            }
        }
    }
}

/// SPIR-V words of an operation before its fields: the search call, the root entry, the header
/// loads, the policy test, the written flag's address and store, the fault store, and the
/// selection.
const COMMAND_WORDS: u32 = dynamic::BASE_WORDS + 40 + 4 * FIELD_WORDS + 160;
/// SPIR-V words of one contained field access.
const FIELD_WORDS: u32 = 72;
/// SPIR-V words of a loop over a record's `u64`s beyond its store: the counter, the exit test, the
/// stride test and its select, the loop's blocks, and a compute reset's retained pipeline (its
/// load, and the tests and selects that keep it).
const LOOP_WORDS: u32 = 96 + FIELD_WORDS + 48;
/// SPIR-V words of a reset beyond its store: its loops (the command's words, the words past the
/// bindings, and a render command's four stages' bindings), the counts' header load, and each
/// stage's count (its shift, mask, clamp and bounds).
const RESET_WORDS: u32 = 6 * LOOP_WORDS + FIELD_WORDS + 4 * 40;
/// SPIR-V words of a copy beyond its load and store: the source's search call, root entry, four
/// header loads, and its pipeline and state loads, the descriptor and pipeline tests, and the loop
/// with its kept-state test and selection.
const COPY_WORDS: u32 = dynamic::BASE_WORDS + 24 + 6 * FIELD_WORDS + 156 + LOOP_WORDS + 64;

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

/// Select the encoder strategy of `inst`: an object word's handle load, or an encoder
/// operation; `None` for any other instruction.
pub fn select(
    cx: &mut SelectCx<'_>,
    at: InstId,
    inst: &Inst,
    m: &mut Meter,
) -> Result<Option<EncodeStrategy>, Error> {
    match &inst.kind {
        InstKind::Load {
            ptr,
            align,
            volatile: false,
            atomic: None,
            ..
        } if cx.families.access.objects.contains(&at) => Ok(Some(EncodeStrategy::Word(
            dynamic::plan_word(cx, *ptr, inst.ty, *align, m)?,
        ))),
        InstKind::Call(call) => match encoder(call) {
            Some(op) => select_command(cx, op, call).map(|c| Some(EncodeStrategy::Command(c))),
            None => Ok(None),
        },
        _ => Ok(None),
    }
}

/// The refusal of an encoder operation on a device that cannot run it.
fn needs(cx: &SelectCx<'_>) -> Result<(), Error> {
    let f = cx.profile.fields();
    if f.buffer_access != BufferAccess::Addressed {
        return Err(Unsupported::needs_any_of(
            Feature::GpuEncodedIndirectCommands,
            vec![cx.profile.addressed_needs()],
        )
        .into());
    }
    let store = match cx.stage {
        Stage::Vertex if !f.leaf.vertex_stores_and_atomics => {
            Some(VkFeature::VertexPipelineStoresAndAtomics)
        }
        Stage::Fragment if !f.leaf.fragment_stores_and_atomics => {
            Some(VkFeature::FragmentStoresAndAtomics)
        }
        _ => None,
    };
    if let Some(feature) = store {
        return Err(Unsupported::needs(
            Feature::GpuEncodedIndirectCommands,
            Capability::Feature(feature),
        )
        .into());
    }
    Ok(())
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

fn select_command(
    cx: &mut SelectCx<'_>,
    op: IndirectCommand,
    call: &crate::air::ir::Call,
) -> Result<Box<Command>, Error> {
    let arg = |i: usize| -> Result<Value, Error> {
        call.args.get(i).copied().ok_or_else(|| {
            Error::Internal(crate::error::InternalError::new(
                crate::error::InternalKind::Invariant,
                "an encoder operation without its operands",
            ))
        })
    };
    // The command must be an object word: any other command handle is its family's refusal.
    let command = arg(0)?;
    if !matches!(command, Value::Inst(i) if cx.families.access.objects.contains(&i)) {
        return Err(Unsupported::not_implemented(Feature::ArgumentBufferObjects).into());
    }
    needs(cx)?;
    dynamic::plan_search(cx)?;
    // Each operation in the host contract's terms (its policy), and what it writes.
    use CommandEncoderOperation as Op;
    let (operation, fields) = match op {
        IndirectCommand::SetBuffer {
            stage,
            space,
            strided,
        } => {
            // A kernel or vertex buffer's setter takes a stride before the binding; a fragment,
            // object or mesh buffer's takes none, nor does AIR 2.5's kernel or vertex setter,
            // MSL's plain setter, which writes the pipeline's own stride as today's spelling of it
            // (an `i64` of all ones) does.
            let fields = Fields::Buffer {
                address: dynamic::plan_address(cx, arg(2)?)?,
                stride: match strided {
                    true => Some(value(cx, arg(3)?)?),
                    false => None,
                },
                binding: value(cx, arg(if strided { 4 } else { 3 })?)?,
            };
            let stage = match stage {
                CommandStage::Kernel => BufferStage::Kernel,
                CommandStage::Vertex => BufferStage::Vertex,
                CommandStage::Fragment => BufferStage::Fragment,
                CommandStage::Object => BufferStage::Object,
                CommandStage::Mesh => BufferStage::Mesh,
            };
            let address_space = if space == AddrSpace::Constant { 2 } else { 1 };
            (
                Op::Buffer {
                    stage,
                    address_space,
                },
                fields,
            )
        }
        IndirectCommand::Dispatch { exact_threads } => (
            Op::Dispatch { exact_threads },
            Fields::Dispatch {
                grid: value(cx, arg(2)?)?,
                local: value(cx, arg(3)?)?,
            },
        ),
        IndirectCommand::SetComputePipeline | IndirectCommand::SetRenderPipeline => {
            let pipeline = arg(2)?;
            if !matches!(pipeline, Value::Inst(i) if cx.families.access.objects.contains(&i)) {
                return Err(Unsupported::not_implemented(Feature::ArgumentBufferObjects).into());
            }
            let word = operand(cx, pipeline)?;
            let render = op == IndirectCommand::SetRenderPipeline;
            (Op::Pipeline { render }, Fields::Pipeline { word })
        }
        IndirectCommand::ComputeBarrier { enabled } => (
            Op::Barrier {
                render: false,
                enabled,
            },
            Fields::Barrier { enabled },
        ),
        IndirectCommand::RenderBarrier { enabled } => (
            Op::Barrier {
                render: true,
                enabled,
            },
            Fields::Barrier { enabled },
        ),
        IndirectCommand::DrawMesh { threads } => (
            Op::DrawMesh { threads },
            Fields::MeshDraw {
                grid: value(cx, arg(2)?)?,
                object: value(cx, arg(3)?)?,
                mesh: value(cx, arg(4)?)?,
            },
        ),
        // An object threadgroup's memory length has no place in a render command's record yet:
        // the record, its replay and the runtime's execution hold none of it.
        IndirectCommand::SetObjectThreadgroupMemoryLength => {
            return Err(Unsupported::not_implemented(Feature::GpuEncodedIndirectCommands).into())
        }
        IndirectCommand::SetThreadgroupMemoryLength => (
            Op::ThreadgroupMemory,
            Fields::ThreadgroupLength {
                length: value(cx, arg(2)?)?,
                binding: value(cx, arg(3)?)?,
            },
        ),
        IndirectCommand::DrawPrimitives => (
            Op::Draw,
            Fields::Draw {
                arguments: [
                    value(cx, arg(2)?)?,
                    value(cx, arg(3)?)?,
                    value(cx, arg(4)?)?,
                    value(cx, arg(5)?)?,
                    value(cx, arg(6)?)?,
                ],
            },
        ),
        IndirectCommand::SetStageInRegion => (
            Op::StageInRegion,
            Fields::StageInRegion {
                origin: value(cx, arg(2)?)?,
                size: value(cx, arg(3)?)?,
            },
        ),
        IndirectCommand::SetImageblockSize => (
            Op::ImageblockSize,
            Fields::ImageblockSize {
                size: value(cx, arg(2)?)?,
            },
        ),
        IndirectCommand::ResetComputeCommand => (Op::Reset { render: false }, Fields::Reset),
        IndirectCommand::ResetRenderCommand => (Op::Reset { render: true }, Fields::Reset),
        // A render command's fixed-function state (cull mode, depth bias, depth clip mode,
        // winding, fill mode, depth/stencil state) has no place in the record: the record, its
        // replay and the runtime's execution hold none of it. Metal's Apple10 family alone runs
        // these setters; its compiler refuses them on an M4.
        IndirectCommand::SetRenderState(_) => {
            return Err(Unsupported::not_implemented(Feature::IndirectRenderState).into())
        }
        IndirectCommand::DrawPatches { .. } => (
            Op::DrawPatches,
            Fields::PatchDraw {
                arguments: [
                    value(cx, arg(2)?)?,
                    value(cx, arg(3)?)?,
                    value(cx, arg(4)?)?,
                    value(cx, arg(6)?)?,
                    value(cx, arg(7)?)?,
                    value(cx, arg(9)?)?,
                ],
                patch_indices: dynamic::plan_address(cx, arg(5)?)?,
                control_point_indices: None,
                factors: dynamic::plan_address(cx, arg(8)?)?,
                scale: value(cx, arg(10)?)?,
            },
        ),
        // A patch draw's operands, with the control-point index buffer after the patch indices.
        IndirectCommand::DrawIndexedPatches { .. } => (
            Op::DrawIndexedPatches,
            Fields::PatchDraw {
                arguments: [
                    value(cx, arg(2)?)?,
                    value(cx, arg(3)?)?,
                    value(cx, arg(4)?)?,
                    value(cx, arg(7)?)?,
                    value(cx, arg(8)?)?,
                    value(cx, arg(10)?)?,
                ],
                patch_indices: dynamic::plan_address(cx, arg(5)?)?,
                control_point_indices: Some(dynamic::plan_address(cx, arg(6)?)?),
                factors: dynamic::plan_address(cx, arg(9)?)?,
                scale: value(cx, arg(11)?)?,
            },
        ),
        IndirectCommand::DrawIndexedPrimitives { wide, .. } => (
            Op::DrawIndexed,
            Fields::IndexedDraw {
                arguments: [
                    value(cx, arg(2)?)?,
                    value(cx, arg(3)?)?,
                    value(cx, arg(5)?)?,
                    value(cx, arg(6)?)?,
                    value(cx, arg(7)?)?,
                ],
                index_type: u32::from(wide),
                address: dynamic::plan_address(cx, arg(4)?)?,
            },
        ),
        IndirectCommand::CopyComputeCommand => {
            // The source command buffer must be an object word too, and its record buffer is
            // searched as the command's is.
            let source = arg(2)?;
            if !matches!(source, Value::Inst(i) if cx.families.access.objects.contains(&i)) {
                return Err(Unsupported::not_implemented(Feature::ArgumentBufferObjects).into());
            }
            dynamic::plan_search(cx)?;
            (
                Op::Copy,
                Fields::Copy {
                    source: operand(cx, source)?,
                    source_index: value(cx, arg(3)?)?,
                },
            )
        }
    };
    Ok(Box::new(Command {
        op: operation,
        command: operand(cx, command)?,
        index: value(cx, arg(1)?)?,
        fields,
    }))
}

// ---- Emission ----

/// A `u32` leaf at a 4-aligned offset.
const WORD: Leaf = Leaf {
    form: Form {
        held: Held::Int {
            bits: 32,
            container: 32,
        },
        lanes: 1,
    },
    bytes: 4,
    path: Path::Aligned,
};

/// A `u64` leaf at an 8-aligned offset.
const DOUBLE: Leaf = Leaf {
    form: Form {
        held: Held::Int {
            bits: 64,
            container: 64,
        },
        lanes: 1,
    },
    bytes: 8,
    path: Path::Aligned,
};

fn k32(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<SValue> {
    crate::memory::leaf::k32(fb, v)
}

/// A draw's primitive type `operand` (a `u32`) as Metal's encoding writes it into the record: a
/// value past Metal's five (`MTLPrimitiveType`, 0 to 4) as 0, which draws points, as Metal's GPU
/// draws a kernel's draw of one (measured on the M3 Max, macOS 27.0.1: 5 to 12, 15, 16, 31, 63,
/// 255 to 257 and 65535, both draws). Its private types 5 to 9 are the CPU's alone.
fn encoded_primitive_type(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    operand: ValueId,
) -> BResult<SValue> {
    let x = cx.operand(fb, operand)?;
    let five = k32(fb, 5)?;
    let known = fb.binary(BinOp::ULessThan, x, five)?;
    let point = k32(fb, 0)?;
    fb.select(known, x, point)
}

/// The guest address the records of the command buffer whose word (its resource ID, a `u64`) is
/// `word` start at: `indirect_commands::records_address`, the ID shifted up
/// `RECORDS_ADDRESS_SHIFT` bits past `RECORDS_ADDRESS_BASE`. No records start at the null ID's.
fn records(fb: &mut FunctionBuilder<'_>, word: SValue) -> BResult<SValue> {
    let shift = k64(fb, u64::from(RECORDS_ADDRESS_SHIFT))?;
    let at = fb.binary(BinOp::ShiftLeftLogical, word, shift)?;
    let base = k64(fb, RECORDS_ADDRESS_BASE)?;
    fb.binary(BinOp::BitwiseOr, at, base)
}

/// `x` (a `u32`) as a `u64`.
fn wide(fb: &mut FunctionBuilder<'_>, x: SValue) -> BResult<SValue> {
    let u64_ = fb.module().ty(SType::Int {
        width: 64,
        signed: false,
    })?;
    fb.convert(ConvOp::UConvert, x, u64_)
}

/// `base + index * stride` in `u64`, for `index` a `u32`.
fn indexed(fb: &mut FunctionBuilder<'_>, base: u64, index: SValue, stride: u64) -> BResult<SValue> {
    let i = wide(fb, index)?;
    let s = k64(fb, stride)?;
    let scaled = fb.binary(BinOp::IMul, i, s)?;
    let b = k64(fb, base)?;
    fb.binary(BinOp::IAdd, b, scaled)
}

/// Emit encoder strategy `s`, defining value `vid` (an object word).
pub fn emit(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    vid: ValueId,
    s: &EncodeStrategy,
) -> BResult<()> {
    match s {
        EncodeStrategy::Word(source) => {
            let word = dynamic::emit_word(cx, fb, source)?;
            cx.define(vid, word);
            Ok(())
        }
        EncodeStrategy::Command(c) => emit_command(cx, fb, c),
    }
}

/// One field write: its displacement in the record buffer (a `u64`), its value, and its leaf.
type Write = (SValue, SValue, Leaf);

/// What a copy reads: the source's record buffer, its record's displacement in it (a `u64`),
/// whether the copy is one into a buffer that inherits (a `bool`), which keeps the destination's
/// pipeline and buffer bindings, and whether the destination keeps its pipeline and its state
/// word (a `bool`: into a buffer that inherits, or from a source that reports no pipeline).
#[derive(Clone, Copy)]
struct CopyFrom {
    base: dynamic::WideBase,
    record: SValue,
    into_inheriting: SValue,
    keeps_pipeline: SValue,
}

fn emit_command<'c>(cx: &mut EmitCx<'c>, fb: &mut FunctionBuilder<'_>, c: &Command) -> BResult<()> {
    let Policy {
        required,
        forbidden,
        any,
        binding,
        dynamic_stride,
        dropped_on,
    } = c.op.policy();
    let word = cx.operand(fb, c.command)?;
    let at = records(fb, word)?;
    let p = dynamic::search_start(cx, fb, at)?;
    let b = dynamic::plain_base(cx, fb, p)?;
    let capacity = header(cx, fb, b, layout::CAPACITY)?;
    let policy = header(cx, fb, b, layout::POLICY)?;
    let index = cx.operand(fb, c.index)?;
    let mut valid = fb.binary(BinOp::ULessThan, index, capacity)?;
    let mask = k32(fb, required | forbidden)?;
    let bits = fb.binary(BinOp::BitwiseAnd, policy, mask)?;
    let want = k32(fb, required)?;
    let allowed = fb.binary(BinOp::IEqual, bits, want)?;
    valid = fb.binary(BinOp::LogicalAnd, valid, allowed)?;
    // A draw needs one of the kinds whose records hold it; on a buffer of `dropped_on` kinds
    // that holds none, an otherwise valid draw is dropped without a fault.
    let mut dropped = None;
    if any != 0 {
        let kinds = k32(fb, any)?;
        let held = fb.binary(BinOp::BitwiseAnd, policy, kinds)?;
        let zero = k32(fb, 0)?;
        let holds = fb.binary(BinOp::INotEqual, held, zero)?;
        if dropped_on != 0 {
            let others = k32(fb, dropped_on)?;
            let other = fb.binary(BinOp::BitwiseAnd, policy, others)?;
            let other = fb.binary(BinOp::INotEqual, other, zero)?;
            let lacks = fb.unary(UnOp::LogicalNot, holds)?;
            let drop = fb.binary(BinOp::LogicalAnd, lacks, other)?;
            dropped = Some(fb.binary(BinOp::LogicalAnd, valid, drop)?);
        }
        valid = fb.binary(BinOp::LogicalAnd, valid, holds)?;
    }
    let record = indexed(fb, layout::RECORDS, index, layout::RECORD)?;
    let at = |fb: &mut FunctionBuilder<'_>, field: SValue| fb.binary(BinOp::IAdd, record, field);
    let fixed = |fb: &mut FunctionBuilder<'_>, field: u64| -> BResult<SValue> {
        let f = k64(fb, field)?;
        at(fb, f)
    };
    let mut writes: Vec<Write> = Vec::new();
    let mut reset = None;
    let mut copy = None;
    match &c.fields {
        Fields::Buffer {
            address,
            stride,
            binding: slot,
        } => {
            let CommandEncoderOperation::Buffer { stage, .. } = c.op else {
                return Err(cx.fail("a buffer setter of another operation"));
            };
            let slot = cx.operand(fb, *slot)?;
            let field = indexed(fb, layout::buffers(stage), slot, layout::BINDING)?;
            let field = at(fb, field)?;
            let address = dynamic::address_of(cx, fb, *address)?;
            writes.push((field, address, DOUBLE));
            let eight = k64(fb, layout::STRIDE)?;
            let to = fb.binary(BinOp::IAdd, field, eight)?;
            match stride {
                // The plain form writes the pipeline's stride, all ones, over a stride set
                // before too, as Metal's kernel-side setter does.
                None => {
                    let all = k64(fb, u64::MAX)?;
                    writes.push((to, all, DOUBLE));
                }
                Some(stride) => {
                    let stride = cx.operand(fb, *stride)?;
                    writes.push((to, stride, DOUBLE));
                    if dynamic_stride {
                        let dynamic = k32(fb, DYNAMIC_STRIDE)?;
                        let enabled = fb.binary(BinOp::BitwiseAnd, policy, dynamic)?;
                        let zero = k32(fb, 0)?;
                        let enabled = fb.binary(BinOp::INotEqual, enabled, zero)?;
                        let all = k64(fb, u64::MAX)?;
                        let fixed_stride = fb.binary(BinOp::IEqual, stride, all)?;
                        let ok = fb.binary(BinOp::LogicalOr, enabled, fixed_stride)?;
                        valid = fb.binary(BinOp::LogicalAnd, valid, ok)?;
                    }
                }
            }
            valid = bound(cx, fb, b, valid, binding, slot)?;
        }
        Fields::Dispatch { grid, local } => {
            let exact = matches!(
                c.op,
                CommandEncoderOperation::Dispatch {
                    exact_threads: true
                }
            );
            for (vector, first) in [(grid, 0u64), (local, 3)] {
                let v = cx.operand(fb, *vector)?;
                for component in 0..3u32 {
                    let x = fb.composite_extract(v, &[component])?;
                    let to = fixed(fb, layout::ARGUMENTS + 4 * (first + u64::from(component)))?;
                    writes.push((to, x, WORD));
                }
            }
            let to = fixed(fb, layout::ACTION)?;
            let tag = k32(fb, if exact { 2 } else { 1 })?;
            writes.push((to, tag, WORD));
        }
        Fields::Pipeline { word } => {
            let w = cx.operand(fb, *word)?;
            let to = fixed(fb, layout::PIPELINE)?;
            writes.push((to, w, DOUBLE));
            // A set pipeline is reported, it replaces one a reset retained, and the kernel wrote
            // it (`layout::KERNEL_PIPELINE`), where it names one: the state word flags no word
            // that holds none.
            let to = fixed(fb, layout::STATE)?;
            let none = k64(fb, 0)?;
            let named = fb.binary(BinOp::INotEqual, w, none)?;
            let kernel = k32(fb, layout::KERNEL_PIPELINE)?;
            let unflagged = k32(fb, 0)?;
            let state = fb.select(named, kernel, unflagged)?;
            writes.push((to, state, WORD));
        }
        Fields::Barrier { enabled } => {
            let to = fixed(fb, layout::BARRIER)?;
            let flag = k32(fb, u32::from(*enabled))?;
            writes.push((to, flag, WORD));
        }
        Fields::ThreadgroupLength {
            length: n,
            binding: slot,
        } => {
            let n = cx.operand(fb, *n)?;
            let slot = cx.operand(fb, *slot)?;
            let field = indexed(fb, layout::THREADGROUP, slot, 4)?;
            let to = at(fb, field)?;
            writes.push((to, n, WORD));
            valid = bound(cx, fb, b, valid, binding, slot)?;
        }
        Fields::Draw { arguments } => {
            for (i, a) in arguments.iter().enumerate() {
                let x = match i {
                    0 => encoded_primitive_type(cx, fb, *a)?,
                    _ => cx.operand(fb, *a)?,
                };
                let to = fixed(fb, layout::ARGUMENTS + 4 * i as u64)?;
                writes.push((to, x, WORD));
            }
            let to = fixed(fb, layout::ACTION)?;
            let tag = k32(fb, 3)?;
            writes.push((to, tag, WORD));
        }
        Fields::MeshDraw { grid, object, mesh } => {
            // The grid, then the object and the mesh threadgroups' threads, three words each
            // (`indirect_commands::layout::DRAW_ARGUMENT_WORDS`); the action is 7 for a draw of
            // threadgroups, 8 for one of threads.
            for (vector, first) in [(grid, 0u64), (object, 3), (mesh, 6)] {
                let v = cx.operand(fb, *vector)?;
                for component in 0..3u32 {
                    let x = fb.composite_extract(v, &[component])?;
                    let to = fixed(fb, layout::ARGUMENTS + 4 * (first + u64::from(component)))?;
                    writes.push((to, x, WORD));
                }
            }
            let threads = matches!(c.op, CommandEncoderOperation::DrawMesh { threads: true });
            let to = fixed(fb, layout::ACTION)?;
            let tag = k32(fb, if threads { 8 } else { 7 })?;
            writes.push((to, tag, WORD));
        }
        Fields::StageInRegion { origin, size } => {
            for (vector, first) in [(origin, 0u64), (size, 3)] {
                let v = cx.operand(fb, *vector)?;
                for component in 0..3u32 {
                    let x = fb.composite_extract(v, &[component])?;
                    let to = fixed(fb, layout::STAGE_IN + 4 * (first + u64::from(component)))?;
                    writes.push((to, x, WORD));
                }
            }
        }
        Fields::ImageblockSize { size } => {
            // Each lane is held zero-extended in a `u32`: the width is the low half of the word.
            let v = cx.operand(fb, *size)?;
            let width = fb.composite_extract(v, &[0])?;
            let height = fb.composite_extract(v, &[1])?;
            let sixteen = k32(fb, 16)?;
            let height = fb.binary(BinOp::ShiftLeftLogical, height, sixteen)?;
            let word = fb.binary(BinOp::BitwiseOr, width, height)?;
            let to = fixed(fb, layout::IMAGEBLOCK)?;
            writes.push((to, word, WORD));
        }
        Fields::Reset => {
            // A compute command's reset keeps the pipeline it executes with, reported cleared; a
            // render command's is unmeasured and clears it.
            let retain = matches!(c.op, CommandEncoderOperation::Reset { render: false });
            let render = matches!(c.op, CommandEncoderOperation::Reset { render: true });
            // The bindings it clears are those the descriptor counts (`default_record`).
            let counts = header(cx, fb, b, layout::COUNTS)?;
            reset = Some((record, retain, render, counts));
        }
        Fields::Copy {
            source,
            source_index,
        } => {
            let word = cx.operand(fb, *source)?;
            let at = records(fb, word)?;
            let p = dynamic::search_start(cx, fb, at)?;
            let base = dynamic::plain_base(cx, fb, p)?;
            let source_capacity = header(cx, fb, base, layout::CAPACITY)?;
            let source_policy = header(cx, fb, base, layout::POLICY)?;
            let source_counts = header(cx, fb, base, layout::COUNTS)?;
            let counts = header(cx, fb, b, layout::COUNTS)?;
            let source_index = cx.operand(fb, *source_index)?;
            let present = fb.binary(BinOp::ULessThan, source_index, source_capacity)?;
            valid = fb.binary(BinOp::LogicalAnd, valid, present)?;
            // Equal descriptors: the same policy bits and binding counts.
            let same_policy = fb.binary(BinOp::IEqual, source_policy, policy)?;
            let same_counts = fb.binary(BinOp::IEqual, source_counts, counts)?;
            let equal = fb.binary(BinOp::LogicalAnd, same_policy, same_counts)?;
            // Into a buffer that inherits both from one that inherits neither, of the same
            // descriptor otherwise, the destination binding no kernel buffer (count byte 0), at
            // an index other than 0: Metal's copy onto an inheriting buffer's command 0 runs it
            // with `threads_per_grid` and `threadgroups_per_grid` zero (M3 Max and M4), which the
            // destination's documented semantics do not explain.
            let zero = k32(fb, 0)?;
            let inherits = k32(fb, INHERIT_PIPELINE | INHERIT_BUFFERS)?;
            let source_inherits = fb.binary(BinOp::BitwiseAnd, source_policy, inherits)?;
            let owned = fb.binary(BinOp::IEqual, source_inherits, zero)?;
            let inheriting = fb.binary(BinOp::BitwiseOr, source_policy, inherits)?;
            let inheriting = fb.binary(BinOp::IEqual, policy, inheriting)?;
            let kernel = k32(fb, 0xff)?;
            let kernel_buffers = fb.binary(BinOp::BitwiseAnd, counts, kernel)?;
            let no_kernel_buffers = fb.binary(BinOp::IEqual, kernel_buffers, zero)?;
            let others = k32(fb, !0xff)?;
            let other_counts = fb.binary(BinOp::BitwiseAnd, counts, others)?;
            let source_others = fb.binary(BinOp::BitwiseAnd, source_counts, others)?;
            let same_others = fb.binary(BinOp::IEqual, other_counts, source_others)?;
            let not_first = fb.binary(BinOp::INotEqual, index, zero)?;
            let mut into_inheriting = fb.binary(BinOp::LogicalAnd, owned, inheriting)?;
            for test in [no_kernel_buffers, same_others, not_first] {
                into_inheriting = fb.binary(BinOp::LogicalAnd, into_inheriting, test)?;
            }
            let explained = fb.binary(BinOp::LogicalOr, equal, into_inheriting)?;
            valid = fb.binary(BinOp::LogicalAnd, valid, explained)?;
            // A source that reports no pipeline (none set, or one its reset retained) leaves the
            // destination's pipeline and its state word as they are.
            let source_record = indexed(fb, layout::RECORDS, source_index, layout::RECORD)?;
            let field = k64(fb, layout::PIPELINE)?;
            let at = fb.binary(BinOp::IAdd, source_record, field)?;
            let Some(token) = dynamic::wide_at(fb, base, Some(at), &DOUBLE, None)? else {
                return Err(cx.fail("a copied pipeline load without a value"));
            };
            let field = k64(fb, layout::STATE)?;
            let at = fb.binary(BinOp::IAdd, source_record, field)?;
            let Some(state) = dynamic::wide_at(fb, base, Some(at), &WORD, None)? else {
                return Err(cx.fail("a copied state load without a value"));
            };
            let none = k64(fb, 0)?;
            let named = fb.binary(BinOp::INotEqual, token, none)?;
            // Retained, it is unreported; a kernel's write of it is reported, and copied with its
            // state word (`layout::KERNEL_PIPELINE`).
            let retained = k32(fb, layout::RETAINED)?;
            let retained = fb.binary(BinOp::BitwiseAnd, state, retained)?;
            let unreported = fb.binary(BinOp::INotEqual, retained, zero)?;
            let unreported = fb.unary(UnOp::LogicalNot, unreported)?;
            let reports = fb.binary(BinOp::LogicalAnd, named, unreported)?;
            let reports = fb.unary(UnOp::LogicalNot, reports)?;
            let keeps_pipeline = fb.binary(BinOp::LogicalOr, into_inheriting, reports)?;
            copy = Some(CopyFrom {
                base,
                record: source_record,
                into_inheriting,
                keeps_pipeline,
            });
        }
        Fields::IndexedDraw {
            arguments: [primitive_type, count, instances, base_vertex, base_instance],
            index_type,
            address,
        } => {
            // Primitive type, index count, index type, instances, the address (a `u64` over two
            // words), base vertex, base instance (`indirect_commands::layout::DRAW_ARGUMENT_WORDS`).
            let operands = [
                (0, *primitive_type),
                (1, *count),
                (3, *instances),
                (6, *base_vertex),
                (7, *base_instance),
            ];
            for (word, operand) in operands {
                let x = match word {
                    0 => encoded_primitive_type(cx, fb, operand)?,
                    _ => cx.operand(fb, operand)?,
                };
                let to = fixed(fb, layout::ARGUMENTS + 4 * word)?;
                writes.push((to, x, WORD));
            }
            let index_type = k32(fb, *index_type)?;
            let to = fixed(fb, layout::ARGUMENTS + 8)?;
            writes.push((to, index_type, WORD));
            let address = dynamic::address_of(cx, fb, *address)?;
            let to = fixed(fb, layout::ARGUMENTS + 16)?;
            writes.push((to, address, DOUBLE));
            let to = fixed(fb, layout::ACTION)?;
            let tag = k32(fb, 4)?;
            writes.push((to, tag, WORD));
        }
        Fields::PatchDraw {
            arguments: [control_points, start, count, instances, base_instance, stride],
            patch_indices,
            control_point_indices,
            factors,
            scale,
        } => {
            // Control points, patch start, patch count, instances, the patch index buffer's and
            // the factor buffer's addresses (a `u64` over two words each), base instance, the
            // factors' instance stride, the factor scale
            // (`indirect_commands::layout::DRAW_ARGUMENT_WORDS`).
            let operands = [
                (0, *control_points),
                (1, *start),
                (2, *count),
                (3, *instances),
                (8, *base_instance),
                (9, *stride),
            ];
            for (word, operand) in operands {
                let x = cx.operand(fb, operand)?;
                let to = fixed(fb, layout::ARGUMENTS + 4 * word)?;
                writes.push((to, x, WORD));
            }
            for (side, word) in [(patch_indices, 4), (factors, 6)] {
                let address = dynamic::address_of(cx, fb, *side)?;
                let to = fixed(fb, layout::ARGUMENTS + 4 * word)?;
                writes.push((to, address, DOUBLE));
            }
            // The factor scale's `f32` bits.
            let scale = cx.operand(fb, *scale)?;
            let u32_ = fb.module().ty(SType::Int {
                width: 32,
                signed: false,
            })?;
            let bits = fb.convert(ConvOp::Bitcast, scale, u32_)?;
            let to = fixed(fb, layout::ARGUMENTS + 40)?;
            writes.push((to, bits, WORD));
            // An indexed patch draw's control-point index buffer takes the first two threadgroup
            // lengths' words (`indirect_commands::layout::CONTROL_POINT_INDICES`), and its action
            // is 6; a patch draw's is 5.
            let tag = match control_point_indices {
                Some(side) => {
                    let address = dynamic::address_of(cx, fb, *side)?;
                    let to = fixed(fb, layout::CONTROL_POINT_INDICES)?;
                    writes.push((to, address, DOUBLE));
                    6
                }
                None => 5,
            };
            let to = fixed(fb, layout::ACTION)?;
            let tag = k32(fb, tag)?;
            writes.push((to, tag, WORD));
        }
    }
    // The command's written flag past every record (`indirect_commands::written_flag_offset`),
    // set with its record.
    let past = indexed(fb, layout::RECORDS, capacity, layout::RECORD)?;
    let flag = indexed(fb, 0, index, WRITTEN_FLAG_BYTES as u64)?;
    let flag = fb.binary(BinOp::IAdd, past, flag)?;
    let one = k32(fb, 1)?;
    writes.push((flag, one, WORD));
    let fault_at = k64(fb, layout::FAULT)?;
    let fault = dynamic::word_place(fb, b, fault_at)?;
    let write: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, cx| {
        for (to, v, leaf) in writes {
            dynamic::wide_at(fb, b, Some(to), &leaf, Some(v))?;
        }
        if let Some((record, retain, render, counts)) = reset {
            default_record(cx, fb, b, (record, retain), (render, counts))?;
        }
        if let Some(from) = copy {
            copy_record(cx, fb, b, record, from)?;
        }
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    let faulted: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, cx| {
        match dropped {
            // A dropped draw sets no fault.
            Some(dropped) => {
                let faults = fb.unary(UnOp::LogicalNot, dropped)?;
                let store: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, _| {
                    fb.atomic_store(fault, Scope::Device, Semantics::RELAXED, one)?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                });
                fb.if_(cx, faults, store, None)?;
            }
            None => fb.atomic_store(fault, Scope::Device, Semantics::RELAXED, one)?,
        }
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(cx, valid, write, Some(faulted))?;
    Ok(())
}

/// The bindings' `u64`s of a record, from the first binding's address to past the last one's
/// stride.
const FIRST_BINDING: u32 = (layout::buffers(BufferStage::Kernel) / 8) as u32;
const PAST_BINDINGS: u32 = FIRST_BINDING + (layout::ALL_BINDINGS * layout::BINDING / 8) as u32;
/// The `u64`s of one binding: its address and its stride.
const BINDING_DOUBLES: u32 = (layout::BINDING / 8) as u32;
/// The record's pipeline `u64`, and the one its state word lies alone in.
const PIPELINE_DOUBLE: u32 = (layout::PIPELINE / 8) as u32;
const STATE_DOUBLE: u32 = (layout::STATE / 8) as u32;

/// The `u64`s of a record.
const DOUBLES: u32 = (layout::RECORD / 8) as u32;

/// Emit `each` for every `u64` of a record, its index `j` (a `u32`) counted in a loop.
fn each_double<'c, F>(cx: &mut EmitCx<'c>, fb: &mut FunctionBuilder<'_>, each: F) -> BResult<()>
where
    F: FnOnce(&mut EmitCx<'c>, &mut FunctionBuilder<'_>, SValue) -> BResult<()>,
{
    let from = k32(fb, 0)?;
    let to = k32(fb, DOUBLES)?;
    each_double_in(cx, fb, (from, to), each)
}

/// Emit `each` for the `u64`s of a record from index `from` to `to` (each a `u32`), its index `j`
/// counted in a loop.
fn each_double_in<'c, F>(
    cx: &mut EmitCx<'c>,
    fb: &mut FunctionBuilder<'_>,
    (from, to): (SValue, SValue),
    each: F,
) -> BResult<()>
where
    F: FnOnce(&mut EmitCx<'c>, &mut FunctionBuilder<'_>, SValue) -> BResult<()>,
{
    let u32_ = fb.module().ty(SType::Int {
        width: 32,
        signed: false,
    })?;
    let zero = fb.module().constant(Constant::Scalar {
        ty: u32_,
        bits: ConstBits(0),
    })?;
    let var = fb.local_var(u32_, Some(zero))?;
    let counter = fb.var_ptr(var)?;
    // The variable is initialized once, at the function's entry: a loop over a record in a loop
    // starts over.
    fb.store(counter, from, MemAccess::default())?;
    let body: LoopFn<'_, EmitCx<'c>> = Box::new(move |fb, cx, lc| {
        let j = fb.load(counter, MemAccess::default())?;
        let done = fb.binary(BinOp::UGreaterThanEqual, j, to)?;
        let exit: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(cx, done, exit, None)?;
        let one = k32(fb, 1)?;
        let next = fb.binary(BinOp::IAdd, j, one)?;
        fb.store(counter, next, MemAccess::default())?;
        each(cx, fb, j)
    });
    fb.loop_(cx, LoopSpec::default(), body)
}

/// Write the default record (`CommandRecord::default`'s encoding) at displacement `record` of the
/// record buffer `b`: zero in every `u64` of the record that a command may have set, a buffer
/// binding's address and stride too, as Metal's reset clears them, in loops: the command's words,
/// the bindings below each stage's count in the header's packed `counts` (`layout::COUNTS`: of a
/// `render` command's mesh, vertex, fragment and object stages, bytes 3, 1, 2 and 0; of a compute
/// command's kernel buffers, byte 0), and the words past the bindings (the threadgroup lengths, an
/// indexed patch draw's control-point indices, the imageblock size and the state word). A binding
/// past its stage's count is zero already, and is not written: the storage is made zero, an
/// encoder's write past the count faults and writes nothing, the CPU's setter refuses one, a
/// compute command has no other stage's setter, and a copy takes a source of the same descriptor.
/// So a reset writes the record's 64-byte head, 136-byte tail and 16 bytes a counted binding,
/// where it wrote all 2248 bytes. Where `retain`, a compute command's reset, the pipeline word
/// keeps the pipeline the command executes with, and the state word flags it retained when there
/// is one (`layout::RETAINED`), and keeps its flag of a kernel's write of it
/// (`layout::KERNEL_PIPELINE`): the retained pipeline is the one that write resolved, as the one
/// the command executes with on Metal.
fn default_record(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: dynamic::WideBase,
    (record, retain): (SValue, bool),
    (render, counts): (bool, SValue),
) -> BResult<()> {
    let retained = if retain {
        let field = k64(fb, layout::PIPELINE)?;
        let at = fb.binary(BinOp::IAdd, record, field)?;
        let Some(token) = dynamic::wide_at(fb, b, Some(at), &DOUBLE, None)? else {
            return Err(cx.fail("a reset's pipeline load without a value"));
        };
        let none = k64(fb, 0)?;
        let named = fb.binary(BinOp::INotEqual, token, none)?;
        // The state word lies alone in its `u64`, its bits in the low word.
        let field = k64(fb, layout::STATE)?;
        let at = fb.binary(BinOp::IAdd, record, field)?;
        let Some(held) = dynamic::wide_at(fb, b, Some(at), &DOUBLE, None)? else {
            return Err(cx.fail("a reset's state word load without a value"));
        };
        let kernel = k64(fb, u64::from(layout::KERNEL_PIPELINE))?;
        let kernel = fb.binary(BinOp::BitwiseAnd, held, kernel)?;
        let flag = k64(fb, u64::from(layout::RETAINED))?;
        let flag = fb.binary(BinOp::BitwiseOr, flag, kernel)?;
        let state = fb.select(named, flag, none)?;
        Some((token, state))
    } else {
        None
    };
    let clear = move |fb: &mut FunctionBuilder<'_>, j: SValue| -> BResult<()> {
        let mut v = k64(fb, 0)?;
        if let Some((token, state)) = retained {
            let pipeline = k32(fb, PIPELINE_DOUBLE)?;
            let pipeline = fb.binary(BinOp::IEqual, j, pipeline)?;
            v = fb.select(pipeline, token, v)?;
            let flags = k32(fb, STATE_DOUBLE)?;
            let flags = fb.binary(BinOp::IEqual, j, flags)?;
            v = fb.select(flags, state, v)?;
        }
        let to = indexed(fb, 0, j, 8)?;
        let to = fb.binary(BinOp::IAdd, record, to)?;
        dynamic::wide_at(fb, b, Some(to), &DOUBLE, Some(v))?;
        Ok(())
    };
    // The command's words, then the words past the bindings.
    for (from, to) in [(0, FIRST_BINDING), (PAST_BINDINGS, DOUBLES)] {
        let from = k32(fb, from)?;
        let to = k32(fb, to)?;
        each_double_in(cx, fb, (from, to), move |_, fb, j| clear(fb, j))?;
    }
    // Each stage's bindings below its count (at most `BINDING_COUNT`), two `u64`s a binding.
    let stages: &[(u32, u32)] = match render {
        true => &[(0, 3), (1, 1), (2, 2), (3, 0)],
        false => &[(0, 0)],
    };
    for &(stage, byte) in stages {
        let shift = k32(fb, 8 * byte)?;
        let count = fb.binary(BinOp::ShiftRightLogical, counts, shift)?;
        let mask = k32(fb, 0xff)?;
        let count = fb.binary(BinOp::BitwiseAnd, count, mask)?;
        let most = k32(fb, BINDING_COUNT as u32)?;
        let over = fb.binary(BinOp::UGreaterThan, count, most)?;
        let count = fb.select(over, most, count)?;
        let per = k32(fb, BINDING_DOUBLES)?;
        let doubles = fb.binary(BinOp::IMul, count, per)?;
        let first = FIRST_BINDING + stage * BINDING_COUNT as u32 * BINDING_DOUBLES;
        let from = k32(fb, first)?;
        let to = fb.binary(BinOp::IAdd, from, doubles)?;
        each_double_in(cx, fb, (from, to), move |_, fb, j| clear(fb, j))?;
    }
    Ok(())
}

/// Copy the source's record `from` onto the record at displacement `record` of the record buffer
/// `b`, `u64` by `u64` in a loop: every one but those the destination keeps: its pipeline's and
/// its state word's where `from.keeps_pipeline`, and its buffer bindings' for a copy into a buffer
/// that inherits.
fn copy_record<'c>(
    cx: &mut EmitCx<'c>,
    fb: &mut FunctionBuilder<'_>,
    b: dynamic::WideBase,
    record: SValue,
    from: CopyFrom,
) -> BResult<()> {
    each_double(cx, fb, move |cx, fb, j| {
        let pipeline = k32(fb, PIPELINE_DOUBLE)?;
        let pipeline = fb.binary(BinOp::IEqual, j, pipeline)?;
        let flags = k32(fb, STATE_DOUBLE)?;
        let flags = fb.binary(BinOp::IEqual, j, flags)?;
        let pipeline = fb.binary(BinOp::LogicalOr, pipeline, flags)?;
        let pipeline = fb.binary(BinOp::LogicalAnd, from.keeps_pipeline, pipeline)?;
        let first = k32(fb, FIRST_BINDING)?;
        let past = k32(fb, PAST_BINDINGS)?;
        let after = fb.binary(BinOp::UGreaterThanEqual, j, first)?;
        let below = fb.binary(BinOp::ULessThan, j, past)?;
        let binding = fb.binary(BinOp::LogicalAnd, after, below)?;
        let binding = fb.binary(BinOp::LogicalAnd, from.into_inheriting, binding)?;
        let kept = fb.binary(BinOp::LogicalOr, pipeline, binding)?;
        let taken = fb.unary(UnOp::LogicalNot, kept)?;
        let d = indexed(fb, 0, j, 8)?;
        let take: ArmFn<'_, EmitCx<'c>> = Box::new(move |fb, cx| {
            let at = fb.binary(BinOp::IAdd, from.record, d)?;
            let Some(v) = dynamic::wide_at(fb, from.base, Some(at), &DOUBLE, None)? else {
                return Err(cx.fail("a copied record load without a value"));
            };
            let to = fb.binary(BinOp::IAdd, record, d)?;
            dynamic::wide_at(fb, b, Some(to), &DOUBLE, Some(v))?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        fb.if_(cx, taken, take, None)?;
        Ok(())
    })
}

/// The header's `u32` at byte `at` of the record buffer `b`.
fn header(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: dynamic::WideBase,
    at: u64,
) -> BResult<SValue> {
    let d = k64(fb, at)?;
    match dynamic::wide_at(fb, b, Some(d), &WORD, None)? {
        Some(v) => Ok(v),
        None => Err(cx.fail("a record header load without a value")),
    }
}

/// `valid` and, when the operation's binding is bounded by a count, `slot` below it: byte
/// `count` of the header's packed counts.
fn bound(
    cx: &mut EmitCx<'_>,
    fb: &mut FunctionBuilder<'_>,
    b: dynamic::WideBase,
    valid: SValue,
    count: Option<u8>,
    slot: SValue,
) -> BResult<SValue> {
    let Some(n) = count else {
        return Ok(valid);
    };
    let counts = header(cx, fb, b, layout::COUNTS)?;
    let shift = k32(fb, u32::from(n) * 8)?;
    let shifted = fb.binary(BinOp::ShiftRightLogical, counts, shift)?;
    let byte = k32(fb, 255)?;
    let limit = fb.binary(BinOp::BitwiseAnd, shifted, byte)?;
    let ok = fb.binary(BinOp::ULessThan, slot, limit)?;
    fb.binary(BinOp::LogicalAnd, valid, ok)
}
