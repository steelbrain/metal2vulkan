//! The device replay of GPU-encoded draws: an execution of render commands a function wrote runs
//! on the GPU as one indirect draw per command, in the commands' order, without the records ever
//! returning to the host.
//!
//! - **What it runs.** Draws of primitives (`draw_primitives`) under the pipeline state the
//!   executing encoder set (the commands inherit it), each drawing with its own vertex and
//!   fragment buffers, or the encoder's where the commands inherit those. The state's vertex and
//!   fragment functions are its replay variant (`VariantKey::replay_bindings`): every buffer they
//!   read is the record's, through the slot table at their stage's `RootTableAddress`.
//! - **The primitive types** ([`DrawReplay::primitive_types`]). Vulkan fixes a pipeline's
//!   topology class, so the consumer records, for each command, one `vkCmdDrawIndirect` of one
//!   draw per primitive type the execution names (a type and its strip, of one class, under
//!   dynamic topology), in the order named; the preprocess writes the command's arguments into
//!   the slot of its own type and zero into the others.
//! - **The preprocess** ([`draw_replay_module`], one invocation per command, in workgroups of
//!   [`DrawReplayModule::LOCAL_SIZE`]) reads the command's record, checks it as the host's
//!   decode does (`CommandBuffer::decode` and `CommandDescriptor::validate_record`: the
//!   producer's sticky encoding fault, an action the descriptor does not hold, a barrier flag
//!   other than 0 or 1, a binding past its stage's count or bound with a dynamic stride the
//!   descriptor does not allow, a state word no record holds), writes the command's two **slot
//!   tables** (vertex, then fragment: the null entry, the sink entry naming the shared resident
//!   set, and entry `2 + n` the stage's buffer `n` resolved in the resident set as a compute
//!   replay's kernel buffer is, for each `n` below the slots its function reads,
//!   [`DrawReplay::vertex_slots`] and [`DrawReplay::fragment_slots`]), and its **arguments**:
//!   one `VkDrawIndirectCommand` per named
//!   type (vertex count, instance count, Metal's vertex start as the first vertex, and its base
//!   instance as the first instance, so `vertex_id`, `instance_id` and `base_instance` are the
//!   draw's own). A buffer whose address no resident allocation holds is unbound (its slot reads
//!   as a buffer never set), and counted as a resident miss ([`StatusBits::RESIDENT_MISS`], the
//!   [`StatusWord::ResidentMisses`] count), as the host's step leaves it unbound and counts it;
//!   the command still draws. A command's own buffer that no slot table holds (past the slots its
//!   stage's function reads, or of the object or mesh stage) is resolved for the count alone, as
//!   the host's step resolves every buffer below its stage's count.
//! - **A range read from a buffer** ([`DrawReplay::range_read`]: `executeCommandsInBuffer:
//!   indirectBuffer:indirectBufferOffset:`). The replay runs over every command of the buffer,
//!   merged into one draw, and each invocation first reads the `MTLIndirectCommandBufferExecutionRange`
//!   (`location`, then `length`, each a `u32`) in the parameters, where the consumer copies it
//!   before the preprocess runs ([`DrawReplayLayout::range_offset`]), as Metal's execution takes it: from
//!   `location` to `location + length`, clamped at the buffer's size. A command outside it draws
//!   nothing and reports nothing; where the location lies past the size, or `location + length`
//!   passes `2^32 - 1` (Metal's GPU then faults the command buffer, or hangs), no command draws and
//!   the replay reports `RANGE`, as the host's step does.
//! - **What it does not run** draws nothing and sets [`StatusBits::REPLAY_FAULT`] with its
//!   reasons ([`ReplayFaults`]) in the command buffer's `ExecutionStatus`: a record the host's
//!   decode refuses (`RECORD`, `ENCODING`), a draw of a primitive type the execution does not
//!   name (`ACTION`), a command that sets a pipeline while inheriting it (`PIPELINE`) or a
//!   buffer while inheriting buffers (`BUFFERS`), a set barrier, which the GPU cannot split the
//!   render pass at (`BARRIER`), and a command at or past the capacity the storage's header
//!   states (`RANGE`). An empty command draws nothing and reports nothing.
//! - **The scratch** is one consumer buffer with a device address and `STORAGE_BUFFER |
//!   INDIRECT_BUFFER` usage, of [`DrawReplayLayout::bytes`]: first what the consumer uploads, the
//!   parameters, the resident root table, each resident root's unpadded extent and the encoder's
//!   buffers in a record's layout (for inherited buffers), [`DrawReplayLayout::uploaded`] bytes
//!   that [`DrawReplayPreprocess`] states; then what the preprocess writes, the slot tables and
//!   the arguments.
//! - **Synchronization** (consumer): the producers' record writes and the scratch's upload before
//!   the preprocess (`COMPUTE_SHADER`, `SHADER_READ`); its `SHADER_WRITE` before the draws read
//!   the arguments (`DRAW_INDIRECT`, `INDIRECT_COMMAND_READ`) and the slot tables
//!   (`VERTEX_SHADER | FRAGMENT_SHADER`, `SHADER_READ`). The preprocess runs outside the render
//!   pass instance its draws are recorded in. A scratch reused for another replay is rewritten
//!   only after the last draw's reads.
//! - **Where it runs.** Profile A: the preprocess and the variants reach memory by device address
//!   (`bufferDeviceAddress`, `shaderInt64`). Elsewhere [`draw_replay_preprocess`] and
//!   [`draw_replay_module`] name what is missing (`GpuEncodedIndirectCommands`), and the host's
//!   step remains.

use std::sync::Arc;

use super::merged::{
    block_commands, corners_per_primitive, header as merged_header, MergedLayout, MERGED_CHUNK,
    MERGED_HEADER_BYTES,
};
use super::{
    at, count_loop, counter, flag, k32, k64, load32, load64, refused, report, section, store32,
    store64, wide, ReplayRefusal, Types, PARAMS_BYTES,
};
use crate::bind::instantiate::{PushHole, RootTable};
use crate::bind::resident::{replay_extents, resident_entries, stage_root_table};
use crate::bind::state::BindState;
use crate::bind::status::{ReplayFaults, StatusBits, StatusWord};
use crate::budget::{Budget, Meter};
use crate::device::profile::BufferAccess;
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{BindError, Error, Feature, InternalError, InternalKind, Unsupported};
use crate::indirect_commands::{
    layout, Bindings, BufferBinding, BufferStage, CommandRecord, Inheritance, StateSource,
    BINDING_COUNT, COMMAND_BUFFER_HEADER_BYTES, COMMAND_BYTES, DRAW, DYNAMIC_STRIDE,
};
use crate::memory::roots;
use crate::pipeline::layout::{PipelineLayout, PushConstantRange};
use crate::pipeline::sym::{PushField, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{
    AtomicOp, BinOp, ExecMode, ExecModel, IoDir, MemAccess, Scope, Semantics, Type, TypedModule,
    UnOp,
};
use crate::vk;

/// Bytes of one `VkDrawIndirectCommand`.
pub const DRAW_ARGUMENT_BYTES: u64 = 16;
/// The most primitive types one draw replay names: a list type and its strip.
pub const MAX_DRAW_REPLAY_TYPES: usize = 2;

/// The action tag of a record's draw of primitives (`CommandRecord::encode_into`).
const DRAW_ACTION: u32 = 3;

/// The parameters' words (`u32` each, from `resident - PARAMS_BYTES`).
const P_FIRST: u64 = 0;
const P_COUNT: u64 = 4;
const P_FLAGS: u64 = 8;
/// `tables - resident`: how far past the resident root table the slot tables start.
const P_TABLES_AT: u64 = 12;
/// `parent - resident`, `arguments - resident`.
const P_PARENT_AT: u64 = 16;
const P_ARGUMENTS_AT: u64 = 20;
/// The buffer slots each stage's slot table holds.
const P_VERTEX_SLOTS: u64 = 24;
const P_FRAGMENT_SLOTS: u64 = 28;
/// The primitive types named, then each (`MAX_DRAW_REPLAY_TYPES` words).
const P_TYPES: u64 = 32;
const P_TYPE: u64 = 36;
/// The range a replay that reads one reads (`location`, then `length`), which the consumer
/// copies there ([`DrawReplayLayout::range_offset`]).
const P_RANGE: u64 = 48;

/// The parameters' flag bit: the commands inherit the encoder's buffers.
const INHERIT_BUFFERS: u32 = 1;
/// The parameters' flag bit: the replay reads its range at `P_RANGE` ([`DrawReplay::range_read`]).
const RANGE_READ: u32 = 2;

/// What one device replay of GPU-encoded draws runs: commands `first .. first + count` of an
/// indirect command buffer of `capacity` commands, under the encoder's pipeline state, each
/// drawing with its own buffers or, where `inheritance.buffers` is the parent's, with `parent`'s
/// vertex and fragment buffers; one draw per command for each of `primitive_types`
/// (`MTLPrimitiveType` raw values: one type, or a list type and its strip). Each stage's slot
/// table holds its buffers below `vertex_slots` and `fragment_slots` (at most 32 each): the
/// slots past the highest one the stage's function reads need no entry.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrawReplay {
    pub capacity: u32,
    pub first: u32,
    pub count: u32,
    pub inheritance: Inheritance,
    pub parent: Bindings,
    pub primitive_types: Vec<u32>,
    pub vertex_slots: u32,
    pub fragment_slots: u32,
    /// Draw every command in one merged draw (`merged.rs`), its vertex function the merged
    /// variant (`VariantKey::merged_draws`), rather than one draw per command: for one list type
    /// or points (`primitive_types`), where no fragment function reads a command's buffers
    /// (`fragment_slots` 0).
    pub merged: bool,
    /// The commands run are those of the range the GPU reads as the preprocess runs (the module
    /// docs): `first` 0 and `count` the capacity, merged into one draw.
    pub range_read: bool,
}

impl DrawReplay {
    /// The commands `first .. first + count` of a buffer of `capacity` commands that inherit the
    /// encoder's pipeline state and draw with their own buffers (set `inheritance` and `parent`
    /// for commands that inherit the encoder's buffers too), drawing `primitive_types`, every
    /// slot of both stages in their tables (set `vertex_slots` and `fragment_slots` to the slots
    /// the functions read for smaller tables).
    pub fn new(capacity: u32, first: u32, count: u32, primitive_types: Vec<u32>) -> DrawReplay {
        DrawReplay {
            capacity,
            first,
            count,
            inheritance: Inheritance {
                pipeline: StateSource::Parent,
                buffers: StateSource::Command,
            },
            parent: Bindings::default(),
            primitive_types,
            vertex_slots: BINDING_COUNT as u32,
            fragment_slots: BINDING_COUNT as u32,
            merged: false,
            range_read: false,
        }
    }
}

/// Where each part of a draw replay's scratch lies, in bytes from its start.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DrawReplayLayout {
    /// Commands replayed.
    pub commands: u32,
    /// Primitive types named: the draws recorded per command.
    pub types: u32,
    /// The buffer slots each stage's slot table holds ([`DrawReplay::vertex_slots`],
    /// [`DrawReplay::fragment_slots`]).
    pub vertex_slots: u32,
    pub fragment_slots: u32,
    /// The resident root table ([`DrawReplayPreprocess::resident`]); the parameters
    /// ([`DrawReplayPreprocess::params`]) end here.
    pub resident_offset: u64,
    /// Each resident root's unpadded extent ([`DrawReplayPreprocess::extents`]): right after the
    /// resident root table, whose set ends with it.
    pub extents_offset: u64,
    /// The encoder's buffers ([`DrawReplayPreprocess::parent`]).
    pub parent_offset: u64,
    /// The bytes from the scratch's start the consumer uploads; the slot tables and the
    /// arguments, which the preprocess writes, follow.
    pub uploaded: u64,
    /// The slot tables: each command's vertex table, then its fragment table.
    pub tables_offset: u64,
    /// The arguments: `commands * types` `VkDrawIndirectCommand`s, which the preprocess writes.
    pub arguments_offset: u64,
    /// The scratch's size.
    pub bytes: u64,
    /// A merged replay's sections ([`DrawReplay::merged`]); `None` for draws per command.
    pub merged: Option<MergedLayout>,
}

impl DrawReplayLayout {
    /// The vertex function's slot table of the `command`th command of the range: its draws'
    /// vertex `RootTableAddress` holes take the scratch's address plus this.
    pub fn vertex_table_offset(&self, command: u32) -> u64 {
        self.tables_offset + u64::from(command) * self.tables_bytes()
    }

    /// The fragment function's slot table of the `command`th command.
    pub fn fragment_table_offset(&self, command: u32) -> u64 {
        self.vertex_table_offset(command) + table_bytes(self.vertex_slots)
    }

    /// Bytes of one command's two slot tables.
    fn tables_bytes(&self) -> u64 {
        table_bytes(self.vertex_slots) + table_bytes(self.fragment_slots)
    }

    /// The `VkDrawIndirectCommand` of the `command`th command for its `type_index`th primitive
    /// type.
    pub fn argument_offset(&self, command: u32, type_index: u32) -> u64 {
        self.arguments_offset
            + (u64::from(command) * u64::from(self.types) + u64::from(type_index))
                * DRAW_ARGUMENT_BYTES
    }

    /// Where the parameters start.
    pub fn params_offset(&self) -> u64 {
        self.resident_offset - PARAMS_BYTES
    }

    /// Where, in the parameters of a replay that reads its range ([`DrawReplay::range_read`]),
    /// the consumer copies the range's 8 bytes (`location`, then `length`) before the preprocess
    /// runs, from the scratch's start; the host uploads zero there.
    pub fn range_offset(&self) -> u64 {
        self.params_offset() + P_RANGE
    }
}

/// The preprocess of one draw replay: what the consumer uploads into the scratch (of
/// `layout.bytes`, with a device address) and how it records the preprocess
/// ([`draw_replay_module`]).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrawReplayPreprocess {
    pub layout: DrawReplayLayout,
    /// The preprocess's parameters, at `layout.params_offset()`.
    pub params: Vec<u8>,
    /// The resident root table, at `layout.resident_offset`: its sink entry's `host` field
    /// ([`RootTable::SINK_HOST_OFFSET`]) takes the sink's device address, as any root table's.
    pub resident: RootTable,
    /// One `u64` per entry of `resident`'s set, at `layout.extents_offset`, right after the set,
    /// then one zero: the unpadded extent a buffer resolves below, as a compute replay's, read
    /// after whatever set a consumer points the table at (`ReplayPreprocess::extents`).
    pub extents: Vec<u8>,
    /// The encoder's vertex and fragment buffers in a record's layout, at `layout.parent_offset`.
    pub parent: Vec<u8>,
    /// A merged replay's header, at its `layout.merged`'s `header_offset`
    /// ([`MERGED_HEADER_BYTES`]); empty for draws per command.
    pub header: Vec<u8>,
    /// The push bytes ([`DrawReplayModule::PUSH_BYTES`], `COMPUTE`, offset 0), zero at the
    /// `push_holes`.
    pub push_constants: Vec<u8>,
    /// `RootTableAddress`: the scratch's address plus `layout.resident_offset`;
    /// `ExecutionStatusAddress`: the command buffer's `ExecutionStatus`; `CommandRecordsAddress`:
    /// the record storage's (its header).
    pub push_holes: Vec<PushHole>,
    /// The preprocess's group count: one invocation per command, in workgroups of
    /// [`DrawReplayModule::LOCAL_SIZE`].
    pub group_count: [u32; 3],
}

/// The preprocess of `replay` for `state`'s resident set (its used resources and bound buffers, as
/// any draw's), on `device`. Its host work is bounded by the resident set; nothing of it is per
/// command. A device without device addresses is `Unsupported{GpuEncodedIndirectCommands}`
/// naming what it lacks; a replay it cannot run is [`BindError::InvalidReplay`]: a range past the
/// capacity, commands that name their own pipeline states, primitive types that are not one
/// type or a list type and its strip, or encoder buffers no record encodes.
pub fn draw_replay_preprocess(
    device: &Device,
    state: &BindState,
    replay: &DrawReplay,
) -> Result<DrawReplayPreprocess, BindError> {
    let profile = device.profile();
    if profile.fields().buffer_access != BufferAccess::Addressed {
        return Err(BindError::Unsupported(Unsupported::needs_any_of(
            Feature::GpuEncodedIndirectCommands,
            vec![profile.addressed_needs()],
        )));
    }
    let end = replay.first.checked_add(replay.count);
    if end.is_none_or(|end| end > replay.capacity) {
        return Err(refused(ReplayRefusal::RangeOutsideCapacity));
    }
    if replay.inheritance.pipeline != StateSource::Parent {
        return Err(refused(ReplayRefusal::OwnPipelines));
    }
    if !one_class(&replay.primitive_types) {
        return Err(refused(ReplayRefusal::DrawPrimitiveTypes));
    }
    let corners = match (replay.merged, replay.primitive_types.as_slice()) {
        (false, _) => None,
        (true, &[t]) if replay.fragment_slots == 0 => match corners_per_primitive(t) {
            Some(c) => Some(c),
            None => return Err(refused(ReplayRefusal::DrawPrimitiveTypes)),
        },
        (true, _) => return Err(refused(ReplayRefusal::DrawPrimitiveTypes)),
    };
    let whole = replay.first == 0 && replay.count == replay.capacity;
    if replay.range_read && (corners.is_none() || !whole) {
        return Err(refused(ReplayRefusal::RangeRead));
    }
    let most = BINDING_COUNT as u32;
    if replay.vertex_slots > most || replay.fragment_slots > most {
        return Err(refused(ReplayRefusal::NotReplayVariant));
    }
    // The encoder's vertex and fragment buffers, in a record's layout (an empty record).
    let mut record = CommandRecord::default();
    for stage in [BufferStage::Vertex, BufferStage::Fragment] {
        *record.bindings.buffers_mut(stage) = *replay.parent.buffers(stage);
    }
    let mut parent = vec![0u8; COMMAND_BYTES];
    record
        .encode_into(&mut parent)
        .map_err(|_| refused(ReplayRefusal::InvalidParent))?;

    let entries = resident_entries(state);
    let resident = stage_root_table(&[], 2, &entries);
    let extents = replay_extents(state, &entries);
    let commands = u64::from(replay.count);
    let types = replay.primitive_types.len() as u64;
    let tables = table_bytes(replay.vertex_slots) + table_bytes(replay.fragment_slots);
    let resident_offset = PARAMS_BYTES;
    // The extents right after the set, which the table holds from its root 2 to its end
    // (`memory::roots::ResidentView`).
    let extents_offset = resident_offset + resident.bytes.len() as u64;
    let parent_offset = section(extents_offset + extents.len() as u64);
    let header_offset = section(parent_offset + COMMAND_BYTES as u64);
    let uploaded = match corners {
        Some(_) => header_offset + MERGED_HEADER_BYTES,
        None => parent_offset + COMMAND_BYTES as u64,
    };
    let tables_offset = section(uploaded);
    let arguments_offset = section(tables_offset + commands * tables);
    let arguments_end = section(arguments_offset + commands * types * DRAW_ARGUMENT_BYTES);
    let merged = corners.map(|_| {
        let block_commands = block_commands(replay.count);
        let blocks = replay.count.div_ceil(block_commands);
        let prefix_offset = arguments_end;
        let sums_offset = section(prefix_offset + 8 * commands);
        let draws_offset = section(sums_offset + 8 * u64::from(blocks));
        MergedLayout {
            header_offset,
            prefix_offset,
            sums_offset,
            draws_offset,
            block_commands,
            blocks,
        }
    });
    let bytes = match &merged {
        Some(m) => section(m.draws_offset + 2 * DRAW_ARGUMENT_BYTES),
        None => arguments_end,
    };
    // Every offset a parameter states is relative to the resident root table; a range whose
    // tables or arguments lie past 2^32 bytes of it is refused.
    let as_param = |v: u64| {
        u32::try_from(v).map_err(|_| {
            refused(ReplayRefusal::TooManyCommands {
                count: replay.count,
            })
        })
    };
    let layout = DrawReplayLayout {
        commands: replay.count,
        types: types as u32,
        vertex_slots: replay.vertex_slots,
        fragment_slots: replay.fragment_slots,
        resident_offset,
        extents_offset,
        parent_offset,
        uploaded,
        tables_offset,
        arguments_offset,
        bytes,
        merged,
    };
    let header = match (&layout.merged, corners) {
        (Some(m), Some(c)) => merged_header(replay, &layout, m, (c, MERGED_CHUNK)),
        _ => Vec::new(),
    };
    let mut flags = if replay.inheritance.buffers == StateSource::Parent {
        INHERIT_BUFFERS
    } else {
        0
    };
    if replay.range_read {
        flags |= RANGE_READ;
    }
    let mut words = vec![
        replay.first,
        replay.count,
        flags,
        as_param(tables_offset - resident_offset)?,
        as_param(parent_offset - resident_offset)?,
        as_param(arguments_offset - resident_offset)?,
        replay.vertex_slots,
        replay.fragment_slots,
        types as u32,
    ];
    words.extend(&replay.primitive_types);
    let mut params = vec![0u8; PARAMS_BYTES as usize];
    for (w, out) in words.iter().zip(params.as_chunks_mut::<4>().0.iter_mut()) {
        out.copy_from_slice(&w.to_le_bytes());
    }
    let compute = vk::ShaderStageFlags::COMPUTE;
    Ok(DrawReplayPreprocess {
        layout,
        params,
        resident,
        extents,
        parent,
        header,
        push_constants: vec![0; DrawReplayModule::PUSH_BYTES as usize],
        push_holes: vec![
            PushHole::RootTableAddress {
                stages: compute,
                offset: DrawReplayModule::RESIDENT_PUSH,
            },
            PushHole::ExecutionStatusAddress {
                stages: compute,
                offset: DrawReplayModule::STATUS_PUSH,
            },
            PushHole::CommandRecordsAddress {
                stages: compute,
                offset: DrawReplayModule::RECORDS_PUSH,
            },
        ],
        group_count: [replay.count.div_ceil(DrawReplayModule::LOCAL_SIZE), 1, 1],
    })
}

/// Bytes of a slot table of `slots` buffer slots: the null and sink entries, then one per slot.
fn table_bytes(slots: u32) -> u64 {
    (2 + u64::from(slots)) * roots::ENTRY_BYTES
}

/// Whether `types` are one primitive type of Metal's five, or a list type and its strip (lines
/// or triangles), each once: draws one pipeline's topology class holds.
fn one_class(types: &[u32]) -> bool {
    match *types {
        [t] => t < 5,
        [a, b] => {
            let mut pair = [a, b];
            pair.sort_unstable();
            matches!(pair, [1, 2] | [3, 4])
        }
        _ => false,
    }
}

/// The crate-owned draw replay preprocess module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DrawReplayModule {
    /// SPIR-V words, entry point `main`, no descriptor.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (device addresses and `shaderInt64`).
    pub requirements: Requirements,
    /// The pipeline layout the consumer creates the preprocess's pipeline with: no descriptor
    /// set, and one `COMPUTE` push-constant range of [`DrawReplayModule::PUSH_BYTES`] at
    /// offset 0.
    pub layout: PipelineLayout,
}

impl DrawReplayModule {
    /// Bytes of the `COMPUTE` push-constant range at offset 0.
    pub const PUSH_BYTES: u32 = 24;
    /// The resident root table's device address (`u64`).
    pub const RESIDENT_PUSH: u32 = 0;
    /// The `ExecutionStatus` buffer's device address (`u64`).
    pub const STATUS_PUSH: u32 = 8;
    /// The record storage's device address (`u64`).
    pub const RECORDS_PUSH: u32 = 16;
    /// The invocations of one workgroup: each one command.
    pub const LOCAL_SIZE: u32 = 64;
}

/// The draw replay preprocess module of `device`: built with the typed builder, checked by the
/// owned verifier against the device's profile (requirements included), and serialized. A device
/// without device addresses is `Unsupported{GpuEncodedIndirectCommands}` naming what it lacks.
pub fn draw_replay_module(device: &Device) -> Result<DrawReplayModule, Error> {
    let profile = device.profile();
    if profile.fields().buffer_access != BufferAccess::Addressed {
        return Err(Unsupported::needs_any_of(
            Feature::GpuEncodedIndirectCommands,
            vec![profile.addressed_needs()],
        )
        .into());
    }
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the draw replay preprocess needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &no_binding)?;
    Ok(DrawReplayModule {
        spirv: words.into(),
        requirements: r.requirements,
        layout: PipelineLayout {
            sets: Vec::new(),
            push_constants: vec![PushConstantRange {
                stages: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: DrawReplayModule::PUSH_BYTES,
            }],
            push_descriptor_eligible: false,
        },
    })
}

/// The module declares no binding.
fn no_binding(_: &Symbol) -> Option<u32> {
    None
}

/// The root `guest` resolves to in the resident set the resident root table at `resident` names
/// (`view`), its offset there, and whether it lies inside the root's unpadded extent: a hit.
fn resolved(
    fb: &mut FunctionBuilder<'_>,
    (resident, view): (Value, &roots::ResidentView),
    guest: Value,
) -> BResult<(Value, Value, Value)> {
    let (root, offset) = roots::resolve(fb, resident, guest)?;
    let null = k64(fb, roots::NULL_ROOT)?;
    let found = fb.binary(BinOp::INotEqual, root, null)?;
    let extent = roots::extent_of(fb, view, root, found)?;
    let inside = fb.binary(BinOp::ULessThan, offset, extent)?;
    let hit = fb.binary(BinOp::LogicalAnd, found, inside)?;
    Ok((root, offset, hit))
}

/// The range a replay reads ([`DrawReplay::range_read`]) at `range_param` of the parameters,
/// over a buffer of `size` commands: where invocation `s`'s command lies
/// outside it, or the range is one Metal's GPU faults on, zero the command's `types` arguments
/// at `args` and return, the first invocation reporting the fault into `status`.
fn outside_range(
    fb: &mut FunctionBuilder<'_>,
    t: Types,
    (range_param, size, s): (Value, Value, Value),
    (args, types, status): (Value, Value, Value),
) -> BResult<()> {
    let location = load32(fb, t, range_param)?;
    let length_at = at(fb, range_param, 4)?;
    let length = load32(fb, t, length_at)?;
    let location = wide(fb, t, location)?;
    let length = wide(fb, t, length)?;
    let size = wide(fb, t, size)?;
    let s64 = wide(fb, t, s)?;
    let end = fb.binary(BinOp::IAdd, location, length)?;
    let past_size = fb.binary(BinOp::UGreaterThan, location, size)?;
    let most = k64(fb, u64::from(u32::MAX))?;
    let past_most = fb.binary(BinOp::UGreaterThan, end, most)?;
    let faulted = fb.binary(BinOp::LogicalOr, past_size, past_most)?;
    let below = fb.binary(BinOp::ULessThan, end, size)?;
    let end = fb.select(below, end, size)?;
    let before = fb.binary(BinOp::ULessThan, s64, location)?;
    let after = fb.binary(BinOp::UGreaterThanEqual, s64, end)?;
    let outside = fb.binary(BinOp::LogicalOr, before, after)?;
    let outside = fb.binary(BinOp::LogicalOr, outside, faulted)?;
    let leave: ArmFn<'_, ()> = Box::new(move |fb, _| {
        zero_draws(fb, t, args, types)?;
        let zero = k32(fb, 0)?;
        let first = fb.binary(BinOp::IEqual, s, zero)?;
        let reports = fb.binary(BinOp::LogicalAnd, faulted, first)?;
        let report_range: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let range = k32(fb, ReplayFaults::RANGE.0)?;
            report(fb, t, status, range)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        fb.if_(&mut (), reports, report_range, None)?;
        fb.ret(None)?;
        Ok(ArmEnd::Exits)
    });
    fb.if_(&mut (), outside, leave, None)?;
    Ok(())
}

/// Write zero into every argument of the command whose arguments start at `base`.
fn zero_draws(fb: &mut FunctionBuilder<'_>, t: Types, base: Value, types: Value) -> BResult<()> {
    count_loop(fb, t, types, move |fb, k| {
        let k64_ = wide(fb, t, k)?;
        let bytes = k64(fb, DRAW_ARGUMENT_BYTES)?;
        let rel = fb.binary(BinOp::IMul, k64_, bytes)?;
        let a = fb.binary(BinOp::IAdd, base, rel)?;
        let zero = k32(fb, 0)?;
        for i in 0..4 {
            let w = at(fb, a, 4 * i)?;
            store32(fb, t, w, zero)?;
        }
        Ok(())
    })
}

/// The module ([`draw_replay_module`]; the module docs).
pub fn typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let u64_ = b.ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let t = Types { u32_, u64_ };
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let m_resident = b.push_member(
        PushField::RootTableAddress,
        u64_,
        DrawReplayModule::RESIDENT_PUSH,
    )?;
    let m_status = b.push_member(
        PushField::ExecutionStatusAddress,
        u64_,
        DrawReplayModule::STATUS_PUSH,
    )?;
    let m_records = b.push_member(
        PushField::CommandRecordsAddress,
        u64_,
        DrawReplayModule::RECORDS_PUSH,
    )?;
    let push = b.push_global()?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let fb = &mut f;

    let push_value = fb.module().global_value(push)?;
    let field = |fb: &mut FunctionBuilder<'_>, index: u32| -> BResult<Value> {
        let i = k32(fb, index)?;
        let p = fb.access_chain(push_value, &[i])?;
        fb.load(p, MemAccess::default())
    };
    let resident = field(fb, m_resident.index)?;
    let status = field(fb, m_status.index)?;
    let records = field(fb, m_records.index)?;
    let id = fb.module().global_value(invocation)?;
    let id = fb.load(id, MemAccess::default())?;
    let s = fb.composite_extract(id, &[0])?;

    let below = k64(fb, PARAMS_BYTES)?;
    let params = fb.binary(BinOp::ISub, resident, below)?;
    let param = |fb: &mut FunctionBuilder<'_>, offset: u64| -> BResult<Value> {
        let a = at(fb, params, offset)?;
        load32(fb, t, a)
    };
    let first = param(fb, P_FIRST)?;
    let count = param(fb, P_COUNT)?;
    let flags = param(fb, P_FLAGS)?;
    let tables_at = param(fb, P_TABLES_AT)?;
    let parent_at = param(fb, P_PARENT_AT)?;
    let arguments_at = param(fb, P_ARGUMENTS_AT)?;
    let vertex_slots = param(fb, P_VERTEX_SLOTS)?;
    let fragment_slots = param(fb, P_FRAGMENT_SLOTS)?;
    let types = param(fb, P_TYPES)?;

    // An invocation past the range does nothing.
    let past = fb.binary(BinOp::UGreaterThanEqual, s, count)?;
    let leave: ArmFn<'_, ()> = Box::new(|fb, _| {
        fb.ret(None)?;
        Ok(ArmEnd::Exits)
    });
    fb.if_(&mut (), past, leave, None)?;

    let zero32 = k32(fb, 0)?;
    let one32 = k32(fb, 1)?;
    let zero64 = k64(fb, 0)?;

    // The command's arguments: `types` from `arguments + s * types * 16`.
    let s64 = wide(fb, t, s)?;
    let arguments_off = wide(fb, t, arguments_at)?;
    let arguments = fb.binary(BinOp::IAdd, resident, arguments_off)?;
    let types64 = wide(fb, t, types)?;
    let per_command = k64(fb, DRAW_ARGUMENT_BYTES)?;
    let per_command = fb.binary(BinOp::IMul, types64, per_command)?;
    let rel = fb.binary(BinOp::IMul, s64, per_command)?;
    let args = fb.binary(BinOp::IAdd, arguments, rel)?;

    // A replay that reads its range: a command outside it draws nothing, and a range Metal's GPU
    // faults on draws nothing at all, reported once.
    let range_bit = k32(fb, RANGE_READ)?;
    let range_bits = fb.binary(BinOp::BitwiseAnd, flags, range_bit)?;
    let reads_range = fb.binary(BinOp::INotEqual, range_bits, zero32)?;
    let range_param = at(fb, params, P_RANGE)?;
    let read_range: ArmFn<'_, ()> = Box::new(move |fb, _| {
        outside_range(fb, t, (range_param, count, s), (args, types, status))?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), reads_range, read_range, None)?;

    // A command at or past the capacity the storage's header states is not read.
    let capacity_at = at(fb, records, layout::CAPACITY)?;
    let capacity = load32(fb, t, capacity_at)?;
    let index32 = fb.binary(BinOp::IAdd, first, s)?;
    let outside = fb.binary(BinOp::UGreaterThanEqual, index32, capacity)?;
    let refuse: ArmFn<'_, ()> = Box::new(move |fb, _| {
        zero_draws(fb, t, args, types)?;
        let range = k32(fb, ReplayFaults::RANGE.0)?;
        report(fb, t, status, range)?;
        fb.ret(None)?;
        Ok(ArmEnd::Exits)
    });
    fb.if_(&mut (), outside, refuse, None)?;

    let fault = counter(fb, t, 0)?;
    let first64 = wide(fb, t, first)?;
    let index = fb.binary(BinOp::IAdd, first64, s64)?;
    let record_bytes = k64(fb, layout::RECORD)?;
    let rel = fb.binary(BinOp::IMul, index, record_bytes)?;
    let rec = fb.binary(BinOp::IAdd, records, rel)?;
    let rec = at(fb, rec, COMMAND_BUFFER_HEADER_BYTES as u64)?;

    // The producers' sticky encoding fault refuses every command.
    let fault_at = at(fb, records, layout::FAULT)?;
    let header_fault = load32(fb, t, fault_at)?;
    let encoding = fb.binary(BinOp::INotEqual, header_fault, zero32)?;
    flag(fb, fault, encoding, ReplayFaults::ENCODING.0)?;
    let policy_at = at(fb, records, layout::POLICY)?;
    let policy = load32(fb, t, policy_at)?;
    let counts_at = at(fb, records, layout::COUNTS)?;
    let counts = load32(fb, t, counts_at)?;

    // The action: empty, or a draw of primitives the descriptor holds; anything else is a record
    // the host's decode refuses (a render descriptor of draws of primitives holds no other).
    let action_at = at(fb, rec, layout::ACTION)?;
    let action = load32(fb, t, action_at)?;
    let empty = fb.binary(BinOp::IEqual, action, zero32)?;
    let draw_tag = k32(fb, DRAW_ACTION)?;
    let draw = fb.binary(BinOp::IEqual, action, draw_tag)?;
    let draw_bit = k32(fb, DRAW)?;
    let held_bits = fb.binary(BinOp::BitwiseAnd, policy, draw_bit)?;
    let draws_held = fb.binary(BinOp::INotEqual, held_bits, zero32)?;
    let draw = fb.binary(BinOp::LogicalAnd, draw, draws_held)?;
    let held = fb.binary(BinOp::LogicalOr, empty, draw)?;
    let mut malformed = fb.unary(UnOp::LogicalNot, held)?;
    let barrier_at = at(fb, rec, layout::BARRIER)?;
    let barrier = load32(fb, t, barrier_at)?;
    let bad_barrier = fb.binary(BinOp::UGreaterThan, barrier, one32)?;
    malformed = fb.binary(BinOp::LogicalOr, malformed, bad_barrier)?;
    let set_barrier = fb.binary(BinOp::IEqual, barrier, one32)?;
    flag(fb, fault, set_barrier, ReplayFaults::BARRIER.0)?;

    // The pipeline is the encoder's: a command that sets one refuses. A state word other than 0
    // is a compute command's retained pipeline, which no render record holds.
    let own_token_at = at(fb, rec, layout::PIPELINE)?;
    let own_token = load64(fb, t, own_token_at)?;
    let own_set = fb.binary(BinOp::INotEqual, own_token, zero64)?;
    flag(fb, fault, own_set, ReplayFaults::PIPELINE.0)?;
    let state_at = at(fb, rec, layout::STATE)?;
    let state = load32(fb, t, state_at)?;
    let bad_state = fb.binary(BinOp::INotEqual, state, zero32)?;
    malformed = fb.binary(BinOp::LogicalOr, malformed, bad_state)?;

    // Every binding the record holds, of each of its four stages (mesh, vertex, fragment,
    // object): one past its stage's count (the header's byte 3, 1, 2 or 0), or one bound with a
    // dynamic stride where the descriptor allows none or the stage takes none (fragment, object),
    // is a record the host's validation refuses; one set while the commands inherit buffers
    // refuses the command.
    let inherit_bit = k32(fb, INHERIT_BUFFERS)?;
    let inherit_bits = fb.binary(BinOp::BitwiseAnd, flags, inherit_bit)?;
    let inherit_buffers = fb.binary(BinOp::INotEqual, inherit_bits, zero32)?;
    let dynamic_bit = k32(fb, DYNAMIC_STRIDE)?;
    let dynamic_bits = fb.binary(BinOp::BitwiseAnd, policy, dynamic_bit)?;
    let dynamic_allowed = fb.binary(BinOp::INotEqual, dynamic_bits, zero32)?;
    let static_stride = k64(fb, u64::MAX)?;
    let bad_bindings = counter(fb, t, 0)?;
    let unresolved = counter(fb, t, 0)?;
    let view = roots::resident_view(fb, resident)?;
    let all = k32(fb, layout::ALL_BINDINGS as u32)?;
    count_loop(fb, t, all, |fb, n| {
        let n64 = wide(fb, t, n)?;
        let binding_bytes = k64(fb, layout::BINDING)?;
        let rel = fb.binary(BinOp::IMul, n64, binding_bytes)?;
        let first = k64(fb, layout::buffers(BufferStage::Kernel))?;
        let rel = fb.binary(BinOp::IAdd, rel, first)?;
        let own = fb.binary(BinOp::IAdd, rec, rel)?;
        let address = load64(fb, t, own)?;
        let stride_at = at(fb, own, layout::STRIDE)?;
        let stride = load64(fb, t, stride_at)?;
        let addressed = fb.binary(BinOp::INotEqual, address, zero64)?;
        let strided = fb.binary(BinOp::INotEqual, stride, zero64)?;
        let set = fb.binary(BinOp::LogicalOr, addressed, strided)?;
        let inherited = fb.binary(BinOp::LogicalAnd, inherit_buffers, set)?;
        flag(fb, fault, inherited, ReplayFaults::BUFFERS.0)?;
        let five = k32(fb, BINDING_COUNT.trailing_zeros())?;
        let stage = fb.binary(BinOp::ShiftRightLogical, n, five)?;
        let low = k32(fb, BINDING_COUNT as u32 - 1)?;
        let slot = fb.binary(BinOp::BitwiseAnd, n, low)?;
        // The header's byte of the stage's count: mesh 3, vertex 1, fragment 2, object 0.
        let mut byte = k32(fb, 0)?;
        for (stage_index, header_byte) in [(0u32, 3u32), (1, 1), (2, 2)] {
            let k = k32(fb, stage_index)?;
            let is = fb.binary(BinOp::IEqual, stage, k)?;
            let b = k32(fb, header_byte)?;
            byte = fb.select(is, b, byte)?;
        }
        let eight = k32(fb, 8)?;
        let shift = fb.binary(BinOp::IMul, byte, eight)?;
        let shifted = fb.binary(BinOp::ShiftRightLogical, counts, shift)?;
        let mask = k32(fb, 0xff)?;
        let stage_count = fb.binary(BinOp::BitwiseAnd, shifted, mask)?;
        let past = fb.binary(BinOp::UGreaterThanEqual, slot, stage_count)?;
        let past = fb.binary(BinOp::LogicalAnd, set, past)?;
        let dynamic = fb.binary(BinOp::INotEqual, stride, static_stride)?;
        let dynamic = fb.binary(BinOp::LogicalAnd, set, dynamic)?;
        let two = k32(fb, 2)?;
        let strideless = fb.binary(BinOp::UGreaterThanEqual, stage, two)?;
        let refused_stride = fb.unary(UnOp::LogicalNot, dynamic_allowed)?;
        let refused_stride = fb.binary(BinOp::LogicalOr, refused_stride, strideless)?;
        let bad_stride = fb.binary(BinOp::LogicalAnd, dynamic, refused_stride)?;
        let bad = fb.binary(BinOp::LogicalOr, past, bad_stride)?;
        flag(fb, bad_bindings, bad, 1)?;
        // A buffer of the command's own below its stage's count that no slot table holds (past
        // the slots its stage's function reads, or of another stage) is resolved too, so that its
        // miss counts as the host step's does.
        let first_address = k64(fb, BufferBinding::MIN_BUFFER_ADDRESS)?;
        let names = fb.binary(BinOp::UGreaterThanEqual, address, first_address)?;
        let below = fb.binary(BinOp::ULessThan, slot, stage_count)?;
        let mut held = [zero64; 2];
        for (h, (stage_index, slots)) in held
            .iter_mut()
            .zip([(1u32, vertex_slots), (2, fragment_slots)])
        {
            let k = k32(fb, stage_index)?;
            let is = fb.binary(BinOp::IEqual, stage, k)?;
            let inside = fb.binary(BinOp::ULessThan, slot, slots)?;
            *h = fb.binary(BinOp::LogicalAnd, is, inside)?;
        }
        let covered = fb.binary(BinOp::LogicalOr, held[0], held[1])?;
        let uncovered = fb.unary(UnOp::LogicalNot, covered)?;
        let own = fb.unary(UnOp::LogicalNot, inherit_buffers)?;
        let counted = fb.binary(BinOp::LogicalAnd, names, below)?;
        let counted = fb.binary(BinOp::LogicalAnd, counted, uncovered)?;
        let counted = fb.binary(BinOp::LogicalAnd, counted, own)?;
        let count: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let (_, _, hit) = resolved(fb, (resident, &view), address)?;
            let missed = fb.unary(UnOp::LogicalNot, hit)?;
            let one = k32(fb, 1)?;
            let zero = k32(fb, 0)?;
            let add = fb.select(missed, one, zero)?;
            let u = fb.load(unresolved, MemAccess::default())?;
            let u = fb.binary(BinOp::IAdd, u, add)?;
            fb.store(unresolved, u, MemAccess::default())?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        fb.if_(&mut (), counted, count, None)?;
        Ok(())
    })?;
    let bad = fb.load(bad_bindings, MemAccess::default())?;
    let bad = fb.binary(BinOp::INotEqual, bad, zero32)?;
    malformed = fb.binary(BinOp::LogicalOr, malformed, bad)?;
    flag(fb, fault, malformed, ReplayFaults::RECORD.0)?;

    // The draw's arguments, and whether its primitive type is one the execution names.
    let mut word = [zero32; 5];
    for (i, w) in word.iter_mut().enumerate() {
        let a = at(fb, rec, layout::ARGUMENTS + 4 * i as u64)?;
        *w = load32(fb, t, a)?;
    }
    let [primitive_type, vertex_start, vertex_count, instance_count, base_instance] = word;
    let named = counter(fb, t, 0)?;
    count_loop(fb, t, types, |fb, k| {
        let k64_ = wide(fb, t, k)?;
        let four = k64(fb, 4)?;
        let rel = fb.binary(BinOp::IMul, k64_, four)?;
        let at_type = at(fb, params, P_TYPE)?;
        let a = fb.binary(BinOp::IAdd, at_type, rel)?;
        let ty = load32(fb, t, a)?;
        let same = fb.binary(BinOp::IEqual, ty, primitive_type)?;
        flag(fb, named, same, 1)
    })?;
    let named = fb.load(named, MemAccess::default())?;
    let named = fb.binary(BinOp::INotEqual, named, zero32)?;
    let unnamed = fb.unary(UnOp::LogicalNot, named)?;
    let other_type = fb.binary(BinOp::LogicalAnd, draw, unnamed)?;
    flag(fb, fault, other_type, ReplayFaults::ACTION.0)?;

    // The slot tables, vertex then fragment: the null entry, then the sink entry naming the
    // shared resident set; each of the stage's buffers from its source (the record, or the
    // encoder's where the commands inherit buffers), resolved in the resident set.
    let tables_off = wide(fb, t, tables_at)?;
    let tables = fb.binary(BinOp::IAdd, resident, tables_off)?;
    let entry_bytes = k64(fb, roots::ENTRY_BYTES)?;
    let two = k64(fb, 2)?;
    let vertex_entries = wide(fb, t, vertex_slots)?;
    let vertex_entries = fb.binary(BinOp::IAdd, vertex_entries, two)?;
    let vertex_bytes = fb.binary(BinOp::IMul, vertex_entries, entry_bytes)?;
    let fragment_entries = wide(fb, t, fragment_slots)?;
    let fragment_entries = fb.binary(BinOp::IAdd, fragment_entries, two)?;
    let fragment_bytes = fb.binary(BinOp::IMul, fragment_entries, entry_bytes)?;
    let command_bytes = fb.binary(BinOp::IAdd, vertex_bytes, fragment_bytes)?;
    let table_rel = fb.binary(BinOp::IMul, s64, command_bytes)?;
    let vertex_table = fb.binary(BinOp::IAdd, tables, table_rel)?;
    let fragment_table = fb.binary(BinOp::IAdd, vertex_table, vertex_bytes)?;
    let parent_off = wide(fb, t, parent_at)?;
    let parent = fb.binary(BinOp::IAdd, resident, parent_off)?;
    let source = fb.select(inherit_buffers, parent, rec)?;
    let runs = fb.load(fault, MemAccess::default())?;
    let runs = fb.binary(BinOp::IEqual, runs, zero32)?;
    let runs = fb.binary(BinOp::LogicalAnd, runs, draw)?;
    // The misses of a command that draws: those of the buffers no slot table holds, and below
    // those of its tables'.
    let misses = counter(fb, t, 0)?;
    let unresolved = fb.load(unresolved, MemAccess::default())?;
    let unresolved = fb.select(runs, unresolved, zero32)?;
    fb.store(misses, unresolved, MemAccess::default())?;
    let sink_entry = roots::SINK_ROOT * roots::ENTRY_BYTES;
    let sink_at = at(fb, resident, sink_entry + roots::HOST)?;
    let sink = load64(fb, t, sink_at)?;
    let resident_count = view.count;
    for (stage, table, slots) in [
        (BufferStage::Vertex, vertex_table, vertex_slots),
        (BufferStage::Fragment, fragment_table, fragment_slots),
    ] {
        for field in [roots::HOST, roots::GUEST, roots::SIZE, roots::BIND_OFFSET] {
            let a = at(fb, table, roots::NULL_ROOT * roots::ENTRY_BYTES + field)?;
            store64(fb, t, a, zero64)?;
        }
        // `R0` of the slot table is the distance in entries from it to the resident set the
        // resident root table names, wherever it lies.
        let distance = fb.binary(BinOp::ISub, view.set, table)?;
        let entry_shift = k64(fb, roots::ENTRY_BYTES.trailing_zeros().into())?;
        let r0 = fb.binary(BinOp::ShiftRightLogical, distance, entry_shift)?;
        for (field, v) in [
            (roots::HOST, sink),
            (roots::RESIDENT_FIRST, r0),
            (roots::RESIDENT_COUNT, resident_count),
            (roots::BIND_OFFSET, zero64),
        ] {
            let a = at(fb, table, sink_entry + field)?;
            store64(fb, t, a, v)?;
        }
        count_loop(fb, t, slots, |fb, n| {
            let n64 = wide(fb, t, n)?;
            let binding_bytes = k64(fb, layout::BINDING)?;
            let rel = fb.binary(BinOp::IMul, n64, binding_bytes)?;
            let first = k64(fb, layout::buffers(stage))?;
            let rel = fb.binary(BinOp::IAdd, rel, first)?;
            let src = fb.binary(BinOp::IAdd, source, rel)?;
            let guest = load64(fb, t, src)?;
            // Below 2^32 an address names no buffer: a nil buffer's offset, or 0 unset.
            let first_address = k64(fb, BufferBinding::MIN_BUFFER_ADDRESS)?;
            let bound = fb.binary(BinOp::UGreaterThanEqual, guest, first_address)?;
            let live = fb.binary(BinOp::LogicalAnd, runs, bound)?;
            let (root, offset, hit) = resolved(fb, (resident, &view), guest)?;
            let size = roots::entry_field(fb, resident, root, roots::SIZE)?;
            let missed = fb.unary(UnOp::LogicalNot, hit)?;
            let miss = fb.binary(BinOp::LogicalAnd, live, missed)?;
            let one = k32(fb, 1)?;
            let zero = k32(fb, 0)?;
            let add = fb.select(miss, one, zero)?;
            let m = fb.load(misses, MemAccess::default())?;
            let m = fb.binary(BinOp::IAdd, m, add)?;
            fb.store(misses, m, MemAccess::default())?;
            let used = fb.binary(BinOp::LogicalAnd, live, hit)?;
            let host = roots::entry_field(fb, resident, root, roots::HOST)?;
            let base = roots::entry_field(fb, resident, root, roots::GUEST)?;
            let two = k64(fb, 2)?;
            let entry = fb.binary(BinOp::IAdd, n64, two)?;
            let entry_bytes = k64(fb, roots::ENTRY_BYTES)?;
            let entry = fb.binary(BinOp::IMul, entry, entry_bytes)?;
            let entry = fb.binary(BinOp::IAdd, table, entry)?;
            // An unbound slot holds a nil buffer's address word as its bind offset, so the
            // function sees the offset as the buffer's address, as Metal's does.
            let zero64 = k64(fb, 0)?;
            let nil_offset = fb.select(bound, zero64, guest)?;
            for (field, v, unused) in [
                (roots::HOST, host, zero64),
                (roots::GUEST, base, zero64),
                (roots::SIZE, size, zero64),
                (roots::BIND_OFFSET, offset, nil_offset),
            ] {
                let v = fb.select(used, v, unused)?;
                let a = at(fb, entry, field)?;
                store64(fb, t, a, v)?;
            }
            Ok(())
        })?;
    }
    // The misses, counted as a resident miss of the command buffer, as the host's step counts
    // an address no resident allocation holds.
    let missed = fb.load(misses, MemAccess::default())?;
    let any_miss = fb.binary(BinOp::INotEqual, missed, zero32)?;
    let counted: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let bit = k32(fb, StatusBits::RESIDENT_MISS.0)?;
        let word = at(fb, status, u64::from(StatusWord::Status.byte_offset()))?;
        let p = fb.device_place(word, t.u32_)?;
        fb.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bit)?;
        let word = at(
            fb,
            status,
            u64::from(StatusWord::ResidentMisses.byte_offset()),
        )?;
        let p = fb.device_place(word, t.u32_)?;
        fb.atomic(AtomicOp::IAdd, p, Scope::Device, Semantics::RELAXED, missed)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), any_miss, counted, None)?;

    // The arguments: the draw's in the slot of its type, zero in the others and for a command
    // that draws nothing.
    count_loop(fb, t, types, |fb, k| {
        let k64_ = wide(fb, t, k)?;
        let four = k64(fb, 4)?;
        let rel = fb.binary(BinOp::IMul, k64_, four)?;
        let at_type = at(fb, params, P_TYPE)?;
        let a = fb.binary(BinOp::IAdd, at_type, rel)?;
        let ty = load32(fb, t, a)?;
        let same = fb.binary(BinOp::IEqual, ty, primitive_type)?;
        let drawn = fb.binary(BinOp::LogicalAnd, runs, same)?;
        let bytes = k64(fb, DRAW_ARGUMENT_BYTES)?;
        let rel = fb.binary(BinOp::IMul, k64_, bytes)?;
        let out = fb.binary(BinOp::IAdd, args, rel)?;
        let zero = k32(fb, 0)?;
        for (i, v) in [vertex_count, instance_count, vertex_start, base_instance]
            .into_iter()
            .enumerate()
        {
            let v = fb.select(drawn, v, zero)?;
            let w = at(fb, out, 4 * i as u64)?;
            store32(fb, t, w, v)?;
        }
        Ok(())
    })?;

    // A refused command draws nothing (its arguments are zero above) and reports why.
    let faults = fb.load(fault, MemAccess::default())?;
    let faulted = fb.binary(BinOp::INotEqual, faults, zero32)?;
    let reported: ArmFn<'_, ()> = Box::new(move |fb, _| {
        report(fb, t, status, faults)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), faulted, reported, None)?;
    fb.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: DrawReplayModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

#[cfg(test)]
mod tests;
