//! The device replay of GPU-encoded compute commands on the core Vulkan 1.2 path: the records never
//! return to the host.
//!
//! A producer encodes compute commands into an indirect command buffer's record storage
//! (`HostRecordsV1`, `crate::indirect_commands`). The host replay decodes them after the producer
//! completes; the device replay runs them where they are, in three steps the consumer records after
//! the producer, all on the GPU:
//!
//! 1. **The preprocess** ([`replay_preprocess_module`], one invocation per replayed command):
//!    reads the command's record, resolves each of its kernel buffers' addresses in the resident
//!    set (as a loaded pointer's is, but only below the allocation's own length, as the host
//!    resolves a record's address), and writes the command's **slot table**: a root table whose
//!    entry `2 + n` is `[[buffer(n)]]`'s allocation (`host`, `guest`, `size`, and the address's
//!    offset in it as `bind_offset`), whose sink entry names the shared resident set, and whose
//!    entries the variant reads as it reads a bound buffer's. It splits the command's grid into
//!    **regions**: per axis the chunks of at most `maxComputeWorkGroupCount` whole threadgroups,
//!    then, for a thread dispatch whose grid is not a multiple of its threadgroup size, one edge
//!    threadgroup of the remaining threads, in the order (`x` outermost); and for each region it
//!    finds the registered variant ([`ReplayVariant`]: a template's pipeline token, threadgroup
//!    size, and threadgroup lengths) of the command's pipeline and lengths and the region's
//!    threadgroup size (the command's, or an edge's remainder), and writes the region's **record**
//!    (its first threadgroup, the command's packed threadgroup size, and the command's grid in
//!    threads: the `DispatchRegion` value) and one `VkDispatchIndirectCommand` per (command,
//!    region, variant): the region's group count for its variant, zero for every other variant, for
//!    a region the command does not have, for an empty command, and for a grid with a zero axis
//!    (which runs nothing in Metal, so it needs no variant and no size or count check).
//! 2. **The variants**: each registered variant is the template compiled with
//!    `LoweringInputs::replay_bindings` (every buffer slot read through its root entry, and a
//!    grid builtin's region from the dispatch's region record) for its threadgroup size and
//!    lengths, planned with those lengths and the resident set bound and no kernel buffer, and
//!    instantiated with
//!    [`BindingPlan::instantiate_replay`](crate::bind::BindingPlan::instantiate_replay).
//! 3. **The dispatches**: per command in range order, per region, per variant,
//!    `vkCmdDispatchIndirect` from the command's arguments ([`ReplayLayout::argument_offset`]),
//!    with the variant's push bytes whose `RootTableAddress` is the command's slot table
//!    ([`ReplayLayout::table_offset`]) and whose `ReplayRegionAddress` is the region's record
//!    ([`ReplayLayout::region_offset`]). A memory barrier separates commands: running them in
//!    order is one schedule of Metal's concurrent commands (a racy command set compares only
//!    under a class that admits any order). The regions of one command are independent threadgroups
//!    of one Metal dispatch and need no barrier between them.
//!
//! **Flat variants** ([`ReplayVariant::flat`]). A kernel that does not observe its threadgroup
//! partition registers one variant compiled with `LoweringInputs::flat_grid`, which every command
//! of its pipeline selects whatever its threadgroup size, lengths, and imageblock size. Such a
//! command is one region: its record's `region[1].xyz` is Metal's `threads_per_grid` (a thread
//! dispatch's grid, or a threadgroup dispatch's groups times its threadgroup size, each axis at
//! most `2^32 - 1`), and its group count spreads `⌈threads / N⌉` threadgroups of the variant's `N`
//! threads over `x`, then `y`, then `z` within the group-count limits, all in 64 bits
//! (`flat_dispatch`); the variant counts the grid's threads over them and returns past the last. A
//! command whose threadgroup has a zero axis or more threads than the state's
//! `maxTotalThreadsPerThreadgroup` runs nothing, as Metal's does, and reports nothing.
//!
//! **One indirect dispatch** (`flat`): a Metal dispatch whose grid and threadgroup size are GPU
//! data runs on a flat variant through [`flat_dispatch_module`], which derives the same flat
//! dispatch from the dispatch's arguments ([`FlatIndirect`]).
//!
//! **Nothing is guessed.** A command the closed variant set cannot run exactly dispatches nothing,
//! and the preprocess sets [`StatusBits::REPLAY_FAULT`] and its reasons ([`ReplayFaults`]) in the
//! command buffer's `ExecutionStatus`: the producer's sticky encoding fault, a malformed record, a
//! draw, no pipeline (or one set while inherited), a buffer set while inherited or bound with a
//! dynamic stride, a region no registered variant runs (an edge's remainder size included), more
//! regions than the execution records per command ([`ReplayExecution::regions`]), a threadgroup
//! grid above `2^32 - 1` threads, or a command at or past the capacity the record storage's header
//! states (whose record is not read: the execution's capacity must be the storage's). The
//! consumer reads the status after the command buffer ran. A kernel buffer whose address no
//! resident allocation holds is no fault: its slot is unbound (it reads as a buffer never set),
//! counted as a resident miss ([`StatusBits::RESIDENT_MISS`], the [`StatusWord::ResidentMisses`]
//! count), and the command runs, as the host's step leaves it unbound and counts it.
//!
//! **The scratch** is one consumer buffer with a device address and `STORAGE_BUFFER |
//! INDIRECT_BUFFER` usage, of [`ReplayLayout::bytes`]: the slot tables (the preprocess writes
//! them), the preprocess's parameters, the resident root table (a [`RootTable`] the consumer
//! uploads with its sink filled, as any root table), each resident root's unpadded extent, the
//! registered variants, the parent encoder's state in a record's layout (for inherited state),
//! the region records, and the arguments (the preprocess writes those two). [`ReplayPreprocess`]
//! states every byte the consumer uploads.
//!
//! **The preprocess module** (fixed per device: the group-count limits are its constants): one
//! `GLCompute` entry `main` of local size `(1, 1, 1)`, dispatched `(commands, 1, 1)`, no
//! descriptor, and [`ReplayPreprocessModule::PUSH_BYTES`] of `COMPUTE` push constants: the
//! resident root table's device address, the `ExecutionStatus` buffer's, and the record
//! storage's ([`ReplayPreprocess::push_holes`]); [`ReplayPreprocessModule::layout`] is that
//! pipeline layout. Its work per command is bounded whatever the range's length: one pass over the
//! record's 128 buffer bindings, the 32 kernel bindings' searches, and per region (at most
//! [`MAX_REPLAY_REGIONS`]) one pass over the at most 64 variants ([`MAX_REPLAY_VARIANTS`]) and its
//! arguments; no command reads another's record.
//!
//! **Synchronization** (consumer): the producer's record writes and the scratch upload before the
//! preprocess (`COMPUTE_SHADER`, `SHADER_READ`); its writes (`SHADER_WRITE`) before the dispatches
//! read the arguments (`DRAW_INDIRECT`, `INDIRECT_COMMAND_READ`) and the slot tables
//! (`COMPUTE_SHADER`, `SHADER_READ`); `SHADER_WRITE -> SHADER_READ | SHADER_WRITE` in
//! `COMPUTE_SHADER` between commands. Every resident allocation may be read and written by any
//! command: the records bind them, as `useResource` made them resident. A scratch reused for
//! another replay is rewritten only after the last dispatches' reads of its arguments and slot
//! tables (`DRAW_INDIRECT`, `INDIRECT_COMMAND_READ`; `COMPUTE_SHADER`, `SHADER_READ`).
//!
//! **The generated-commands path** ([`replay_preprocess_generated`]): where the consumer enabled
//! `VK_EXT_device_generated_commands` with compute pipeline binding, the same preprocess writes a
//! token stream (per command and region one sequence: the variant's execution-set index, its push
//! bytes with the command's slot table and the region's record, and the group count) and each
//! segment's sequence count, and one `vkCmdExecuteGeneratedCommandsEXT` per possible segment
//! replaces the dispatches per variant and the barrier between every command (`generated`).
//!
//! **Where it runs.** Profile A: the variants and the preprocess reach memory by device address
//! (`bufferDeviceAddress`, `shaderInt64`). Elsewhere [`replay_preprocess`],
//! [`replay_preprocess_module`], and the variants' compile name what is missing
//! (`GpuEncodedIndirectCommands`), and the host replay remains.

use std::num::NonZeroU64;
use std::sync::Arc;

use crate::bind::instantiate::{PushHole, RootTable};
use crate::bind::resident::{replay_extents, resident_entries, stage_root_table};
use crate::bind::state::BindState;
use crate::bind::status::{ReplayFaults, StatusBits, StatusWord};
use crate::budget::{Budget, Meter};
use crate::device::profile::BufferAccess;
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{BindError, Error, Feature, InternalError, InternalKind, Unsupported};
use crate::indirect_commands::THREADGROUP_BUFFER_ARGUMENT_COUNT_USIZE as THREADGROUP_COUNT;
use crate::indirect_commands::{
    layout, Bindings, BufferBinding, BufferStage, CommandRecord, Inheritance, StateSource,
    BINDING_COUNT, COMMAND_BUFFER_HEADER_BYTES, COMMAND_BYTES,
};
use crate::lower::dispatch::NOMINAL_AXIS_BITS;
use crate::memory::roots;
use crate::pipeline::layout::{PipelineLayout, PushConstantRange};
use crate::pipeline::sym::{PushField, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn, LoopFn, LoopSpec};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    AtomicOp, BinOp, ConstBits, Constant, ConvOp, ExecMode, ExecModel, IoDir, MemAccess, Scope,
    Semantics, Type, TypedModule, UnOp,
};
use crate::vk;

mod generated;
#[cfg(test)]
pub use generated::MAX_SEGMENT_COMMANDS;
pub use generated::{
    GeneratedLayout, GeneratedRefusal, GeneratedReplay, GeneratedVariant, NO_HOLE,
    TEMPLATE_HEAD_BYTES,
};
mod draws;
pub use draws::{
    draw_replay_module, draw_replay_preprocess, DrawReplay, DrawReplayLayout, DrawReplayModule,
    DrawReplayPreprocess, DRAW_ARGUMENT_BYTES, MAX_DRAW_REPLAY_TYPES,
};
mod flat;
pub use flat::{flat_dispatch_module, FlatArguments, FlatDispatchModule, FlatIndirect};
mod merged;
pub use merged::{
    draw_merge_module, DrawMergeModule, MergePhase, MergedLayout, MERGED_CHUNK,
    MERGED_HEADER_BYTES, MERGED_REST_FIRST,
};
pub(crate) use merged::{merged_vertex, MergedVertex};
mod predication;
mod rules;
pub use predication::{
    predication_step_module, PredicationCompare, PredicationOp, PredicationStep,
    PredicationStepForm, PredicationStepModule,
};
pub use rules::{
    draw_replay_draw, draw_replay_merges, draw_replay_primitive_type, draw_replay_slots,
    draw_replay_state, DrawReplayRule, DRAW_REPLAY_WRITES,
};

/// The most commands one device replay runs: Metal's indirect command buffer holds at most 16,384
/// commands.
pub const MAX_REPLAY_SLOTS: u32 = 16384;
/// The most registered variants one device replay selects among: each is one indirect dispatch
/// per command and region, so the dispatches recorded are `commands * regions * variants`.
pub const MAX_REPLAY_VARIANTS: u32 = 64;
/// The most regions one replayed command runs: the interior and the seven edges of a grid ragged on
/// every axis, or as many group-count chunks of a grid past the device's limit.
pub const MAX_REPLAY_REGIONS: u32 = 8;
/// The most indirect dispatches one device replay records, `commands * regions * variants`: 16,384
/// commands of 64 variants and one region each.
pub const MAX_REPLAY_DISPATCHES: u64 = 1 << 20;

/// Entries of a slot table: the null root, the sink, and one per kernel buffer binding.
const SLOT_TABLE_ENTRIES: u64 = 2 + BINDING_COUNT as u64;
/// Bytes of one slot table.
pub const SLOT_TABLE_BYTES: u64 = SLOT_TABLE_ENTRIES * roots::ENTRY_BYTES;
/// Bytes of the preprocess's parameters, just below the resident root table.
pub const PARAMS_BYTES: u64 = 96;
/// Bytes of one registered variant: its token (`u64`), threadgroup size (`u32` x 3, then a pad
/// word), threadgroup lengths (`u32` x 31), imageblock size (`u16` x 2, 0 for none), and a flat
/// variant's maximum threads per threadgroup (`u32`, 0 for none), padded to 8.
pub const VARIANT_BYTES: u64 = 160;
/// Bytes of one `VkDispatchIndirectCommand`.
pub const ARGUMENT_BYTES: u64 = 12;
/// Bytes of one region record: the `DispatchRegion` value (two `uvec4`).
pub const REGION_BYTES: u64 = 32;
/// The alignment of each scratch section.
const SECTION_ALIGN: u64 = 64;

/// The parameters' words (`u32` each, from `resident - PARAMS_BYTES`).
const P_FIRST: u64 = 0;
const P_SLOTS: u64 = 4;
const P_VARIANTS: u64 = 8;
const P_FLAGS: u64 = 12;
/// `resident - tables`: how far below the resident root table the slot tables start.
const P_TABLES_BELOW: u64 = 16;
/// `variants - resident`, `parent - resident`, `arguments - resident`.
const P_VARIANTS_AT: u64 = 20;
const P_PARENT_AT: u64 = 24;
const P_ARGUMENTS_AT: u64 = 28;
/// The regions recorded per command, and `regions - resident`.
const P_REGIONS: u64 = 32;
const P_REGIONS_AT: u64 = 36;
/// The generated-commands path: the sequence stride (0: K14c's arguments instead), the push token's
/// bytes, the most commands of one execution, and `templates - resident`, `counts - resident`,
/// `sequences - resident`.
const P_STRIDE: u64 = 40;
const P_PUSH_BYTES: u64 = 44;
const P_SEGMENT: u64 = 48;
const P_TEMPLATES_AT: u64 = 52;
const P_COUNTS_AT: u64 = 56;
const P_SEQUENCES_AT: u64 = 60;
/// The device's `max_threadgroup_memory_length`, which a command's lengths are clamped to before
/// they are rounded up to 16 bytes.
const P_MAX_LENGTH: u64 = 64;

/// The parameters' flag bits: the descriptor's inheritance.
const INHERIT_PIPELINE: u32 = 1 << 0;
const INHERIT_BUFFERS: u32 = 1 << 1;

/// A variant's fields.
const V_TOKEN: u64 = 0;
const V_LOCAL: u64 = 8;
const V_LENGTHS: u64 = 24;
/// The imageblock size (`u16` each, width low), 0 for a variant that reads none.
const V_IMAGEBLOCK: u64 = 148;
/// A flat variant's state's `maxTotalThreadsPerThreadgroup`, 0 for a variant of one threadgroup
/// size.
const V_FLAT: u64 = 152;

/// The most threads a threadgroup of Metal holds: a flat variant's state's
/// `maxTotalThreadsPerThreadgroup` is at most this.
const MAX_THREADGROUP_THREADS: u32 = 1024;

/// One registered variant of a device replay: the template of `pipeline` (its host-registered
/// token), compiled with `LoweringInputs::replay_bindings` for a threadgroup of
/// `threads_per_threadgroup` threads and the threadgroup lengths `threadgroup_memory_lengths` (a
/// length of 0 is an unset slot). A command selects it when its pipeline, threadgroup size, and
/// every threadgroup length are these: the command's lengths as the runtime's plan path takes them,
/// each clamped to the device's `max_threadgroup_memory_length` and rounded up to 16 bytes. A
/// kernel with an explicit imageblock is compiled for one imageblock size
/// (`SpecKey::ImageblockWidth`, `SpecKey::ImageblockHeight`), which its variant states
/// (`imageblock_size`, [`ReplayVariant::with_imageblock_size`]): a command selects it only when
/// its own size is that one, an unset size being 1 by 1. A variant of a kernel without one states
/// none, and every size selects it, as Metal ignores the size there. A flat variant
/// ([`ReplayVariant::flat`]) is selected by its pipeline alone.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReplayVariant {
    pub pipeline: NonZeroU64,
    pub threads_per_threadgroup: [u32; 3],
    pub threadgroup_memory_lengths: [u32; THREADGROUP_COUNT],
    /// The imageblock size (width, height) the variant was compiled for; `None` when its kernel
    /// has no explicit imageblock.
    pub imageblock_size: Option<[u16; 2]>,
    /// A flat variant's ([`ReplayVariant::flat`]) state's `maxTotalThreadsPerThreadgroup`;
    /// `None` for a variant of one threadgroup size.
    pub flat_max_total_threads: Option<u32>,
}

impl ReplayVariant {
    /// The variant of `pipeline` for `threads_per_threadgroup` and `threadgroup_memory_lengths`.
    pub fn new(
        pipeline: NonZeroU64,
        threads_per_threadgroup: [u32; 3],
        threadgroup_memory_lengths: [u32; THREADGROUP_COUNT],
    ) -> ReplayVariant {
        ReplayVariant {
            pipeline,
            threads_per_threadgroup,
            threadgroup_memory_lengths,
            imageblock_size: None,
            flat_max_total_threads: None,
        }
    }

    /// The flat variant of `pipeline`: its template compiled with `LoweringInputs::flat_grid` for a
    /// threadgroup of `threads_per_threadgroup` threads, for a state whose
    /// `maxTotalThreadsPerThreadgroup` is `max_total_threads` (1 to 1,024). Every command of the
    /// pipeline selects it, whatever its threadgroup size, lengths, and imageblock size (its kernel
    /// reads none of them), and no other variant of the pipeline may be registered beside it. A
    /// command whose threadgroup has a zero axis or more than `max_total_threads` threads runs
    /// nothing, as Metal's does; any other runs its grid in threadgroups of
    /// `threads_per_threadgroup` (the module's flat variants).
    pub fn flat(
        pipeline: NonZeroU64,
        threads_per_threadgroup: [u32; 3],
        max_total_threads: u32,
    ) -> ReplayVariant {
        ReplayVariant {
            pipeline,
            threads_per_threadgroup,
            threadgroup_memory_lengths: [0; THREADGROUP_COUNT],
            imageblock_size: None,
            flat_max_total_threads: Some(max_total_threads),
        }
    }

    /// This variant, compiled for the imageblock of `width` by `height` pixels.
    pub fn with_imageblock_size(self, [width, height]: [u16; 2]) -> ReplayVariant {
        ReplayVariant {
            imageblock_size: Some([width, height]),
            ..self
        }
    }
}

/// What one device replay runs: the commands `first .. first + count` of an indirect command buffer
/// of `capacity` commands whose descriptor selects `inheritance`, with the parent encoder's state
/// `parent` (its pipeline and kernel buffers, read where the commands inherit them).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayExecution {
    pub capacity: u32,
    pub first: u32,
    pub count: u32,
    pub inheritance: Inheritance,
    pub parent: Bindings,
    /// The regions the consumer records per command, 1 to 8 ([`MAX_REPLAY_REGIONS`]): a command of
    /// more regions dispatches nothing and reports [`ReplayFaults::REGIONS`]. One region runs every
    /// command whose grid is a multiple of its threadgroup size and within the device's group-count
    /// limit; eight run a grid ragged on every axis. The recorded dispatches, `count * regions *
    /// variants`, are at most 2^20 ([`MAX_REPLAY_DISPATCHES`]).
    pub regions: u32,
}

impl ReplayExecution {
    /// The commands `first .. first + count` of a buffer of `capacity` commands, inheriting
    /// nothing (set `inheritance` and `parent` for an execution that does), recording
    /// eight regions per command (`MAX_REPLAY_REGIONS`).
    pub fn new(capacity: u32, first: u32, count: u32) -> ReplayExecution {
        ReplayExecution {
            capacity,
            first,
            count,
            inheritance: Inheritance {
                pipeline: StateSource::Command,
                buffers: StateSource::Command,
            },
            parent: Bindings::default(),
            regions: MAX_REPLAY_REGIONS,
        }
    }
}

/// Why the binder sets up no device replay.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayRefusal {
    /// The range runs past the buffer's capacity.
    RangeOutsideCapacity,
    /// More than 16,384 commands ([`MAX_REPLAY_SLOTS`]), Metal's most.
    TooManyCommands { count: u32 },
    /// No variant, or more than 64 ([`MAX_REPLAY_VARIANTS`]).
    VariantCount { count: usize },
    /// Variant `index` has a zero threadgroup size on some axis, a threadgroup length that is
    /// not a multiple of 16, or an imageblock side of 0 (no command selects it), or repeats an
    /// earlier one; or it is a flat variant whose maximum is not 1 to 1,024 or that states a
    /// threadgroup length or an imageblock size, or one of a pipeline that another variant also
    /// names where either is flat.
    InvalidVariant { index: usize },
    /// The parent encoder's state has no record encoding: a kernel buffer's attribute stride is
    /// the all-bits-set value a record reserves for the static stride.
    InvalidParent,
    /// The plan is not a replay variant's: a buffer slot not read through its root entry or
    /// beyond a record's 32 kernel buffers, or a field a record cannot supply (a descriptor slot's
    /// residual, a slot's `SlotGuestBase`, the stage-in region).
    NotReplayVariant,
    /// The bind state binds a kernel buffer: a replayed command's record binds them.
    KernelBufferBound,
    /// No region per command, or more than 8 ([`MAX_REPLAY_REGIONS`]).
    RegionCount { count: u32 },
    /// More indirect dispatches than 2^20 ([`MAX_REPLAY_DISPATCHES`]):
    /// `commands * regions * variants`. Fewer regions per command record fewer.
    TooManyDispatches { dispatches: u64 },
    /// The generated-commands path cannot run the replay on this device; K14c's path
    /// ([`replay_preprocess`]) remains.
    GeneratedCommands { reason: GeneratedRefusal },
    /// A draw replay ([`draw_replay_preprocess`]) of commands that name their own pipeline
    /// states: it draws under the encoder's.
    OwnPipelines,
    /// A draw replay's primitive types are not one of Metal's five, or a list type and its strip:
    /// draws one pipeline's topology class holds ([`DrawReplay::primitive_types`]); or a merged
    /// replay's ([`DrawReplay::merged`]) are not one list type or points, or its fragment slot
    /// tables hold slots.
    DrawPrimitiveTypes,
    /// A draw replay that reads its range from a buffer ([`DrawReplay::range_read`]) is not
    /// merged into one draw, or does not run over every command of the buffer.
    RangeRead,
}

/// Where each part of a device replay's scratch lies, in bytes from its start.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReplayLayout {
    /// Commands replayed.
    pub slots: u32,
    /// Registered variants.
    pub variants: u32,
    /// Regions recorded per command ([`ReplayExecution::regions`]).
    pub regions: u32,
    /// The resident root table ([`ReplayPreprocess::resident`]); the parameters
    /// ([`ReplayPreprocess::params`]) end here.
    pub resident_offset: u64,
    /// Each resident root's unpadded extent ([`ReplayPreprocess::extents`]): right after the
    /// resident root table, whose set ends with it.
    pub extents_offset: u64,
    /// The registered variants ([`ReplayPreprocess::variants`]).
    pub variants_offset: u64,
    /// The parent's state ([`ReplayPreprocess::parent`]).
    pub parent_offset: u64,
    /// The region records: `slots * regions` of 32 bytes (`REGION_BYTES`), which the preprocess
    /// writes.
    pub regions_offset: u64,
    /// The arguments: `slots * regions * variants` `VkDispatchIndirectCommand`s (none on the
    /// generated-commands path).
    pub arguments_offset: u64,
    /// The scratch's size.
    pub bytes: u64,
    /// The generated-commands path's sections and token stream ([`replay_preprocess_generated`]);
    /// `None` on K14c's path.
    pub generated: Option<GeneratedLayout>,
}

impl ReplayLayout {
    /// The slot table of the `slot`th command of the range: the root table its dispatches read
    /// (their `RootTableAddress` holes take the scratch's address plus this).
    pub fn table_offset(&self, slot: u32) -> u64 {
        u64::from(slot) * SLOT_TABLE_BYTES
    }

    /// The record of region `region` of the `slot`th command: what the dispatches of that region
    /// read (their `ReplayRegionAddress` holes take the scratch's address plus this).
    pub fn region_offset(&self, slot: u32, region: u32) -> u64 {
        self.regions_offset
            + (u64::from(slot) * u64::from(self.regions) + u64::from(region)) * REGION_BYTES
    }

    /// The `VkDispatchIndirectCommand` of region `region` of the `slot`th command for the
    /// `variant`th variant.
    pub fn argument_offset(&self, slot: u32, region: u32, variant: u32) -> u64 {
        let per_slot = u64::from(self.regions) * u64::from(self.variants);
        self.arguments_offset
            + (u64::from(slot) * per_slot
                + u64::from(region) * u64::from(self.variants)
                + u64::from(variant))
                * ARGUMENT_BYTES
    }

    /// Where the parameters start.
    pub fn params_offset(&self) -> u64 {
        self.resident_offset - PARAMS_BYTES
    }
}

/// The preprocess of one device replay: what the consumer uploads into the scratch (of
/// `layout.bytes`, with a device address) and how it records the preprocess
/// ([`replay_preprocess_module`]).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayPreprocess {
    pub layout: ReplayLayout,
    /// The preprocess's parameters, at `layout.params_offset()`.
    pub params: Vec<u8>,
    /// The resident root table, at `layout.resident_offset`: its sink entry's `host` field
    /// ([`RootTable::SINK_HOST_OFFSET`]) takes the sink's device address, as any root table's.
    pub resident: RootTable,
    /// One `u64` per entry of `resident`'s set, at `layout.extents_offset`, right after the set,
    /// then one zero: how many bytes from the entry's guest address its allocations hold at their
    /// own lengths, without the padding. A kernel buffer resolves only below it, as the host
    /// resolves a record's address (`BufferBinding::resolve_buffer`). The preprocess reads them
    /// after the set the table names, wherever a consumer points it: one that points the table at
    /// a set uploaded later puts that set's extents after it
    /// ([`ResidentSet::replay_bytes`](crate::bind::ResidentSet::replay_bytes)).
    pub extents: Vec<u8>,
    /// The registered variants, at `layout.variants_offset`.
    pub variants: Vec<u8>,
    /// The parent encoder's pipeline and kernel buffers in a record's layout, at
    /// `layout.parent_offset`.
    pub parent: Vec<u8>,
    /// The push bytes ([`ReplayPreprocessModule::PUSH_BYTES`], `COMPUTE`, offset 0), zero at the
    /// `push_holes`.
    pub push_constants: Vec<u8>,
    /// `RootTableAddress`: the scratch's address plus `layout.resident_offset`;
    /// `ExecutionStatusAddress`: the command buffer's `ExecutionStatus`; `CommandRecordsAddress`:
    /// the record storage's (its header).
    pub push_holes: Vec<PushHole>,
    /// The preprocess's group count: one invocation per command.
    pub group_count: [u32; 3],
    /// The generated-commands path's templates and shared pipeline layout
    /// ([`replay_preprocess_generated`]); `None` on K14c's path.
    pub generated: Option<GeneratedReplay>,
}

fn refused(reason: ReplayRefusal) -> BindError {
    BindError::InvalidReplay { reason }
}

/// `v` rounded up to the section alignment.
fn section(v: u64) -> u64 {
    v.next_multiple_of(SECTION_ALIGN)
}

/// The preprocess of `execution` over `variants` for `state`'s resident set (its used resources and
/// bound buffers, as any dispatch's), on `device`. The variants' order is the arguments'
/// ([`ReplayLayout::argument_offset`]). Its host work is bounded by the at most 64 variants
/// ([`MAX_REPLAY_VARIANTS`]) and the resident set; nothing of it is per command. A device without
/// device addresses is `Unsupported{GpuEncodedIndirectCommands}` naming what it lacks; an execution
/// or variant set the replay cannot run is [`BindError::InvalidReplay`].
pub fn replay_preprocess(
    device: &Device,
    state: &BindState,
    execution: &ReplayExecution,
    variants: &[ReplayVariant],
) -> Result<ReplayPreprocess, BindError> {
    preprocess(device, state, execution, variants, None)
}

/// [`replay_preprocess`] on the generated-commands path (`generated`): `generated` holds each
/// registered variant's pipeline, plan, and instantiated commands, in `variants`' order. The
/// preprocess then writes each command's sequences and each possible segment's sequence count
/// ([`GeneratedLayout`]) instead of the arguments, and [`ReplayPreprocess::generated`] holds the
/// variants' templates and the pipeline layout they share. Besides [`replay_preprocess`]'s
/// refusals (the dispatch bound aside: the executions run `commands * regions` sequences), a
/// device or variant set the path cannot run is
/// [`ReplayRefusal::GeneratedCommands`], naming the limit the device's description states or the
/// variant that binds other descriptors; K14c's path remains for it.
pub fn replay_preprocess_generated(
    device: &Device,
    state: &BindState,
    execution: &ReplayExecution,
    variants: &[ReplayVariant],
    generated: &[GeneratedVariant<'_>],
) -> Result<ReplayPreprocess, BindError> {
    let plan = || generated::plan(device, execution.regions, variants, generated);
    preprocess(device, state, execution, variants, Some(&plan))
}

/// [`replay_preprocess`], on the generated-commands path `generated` plans where it is given
/// (after the checks both paths share).
fn preprocess(
    device: &Device,
    state: &BindState,
    execution: &ReplayExecution,
    variants: &[ReplayVariant],
    generated: Option<&dyn Fn() -> Result<generated::Planned, BindError>>,
) -> Result<ReplayPreprocess, BindError> {
    let profile = device.profile();
    if profile.fields().buffer_access != BufferAccess::Addressed {
        return Err(BindError::Unsupported(Unsupported::needs_any_of(
            Feature::GpuEncodedIndirectCommands,
            vec![profile.addressed_needs()],
        )));
    }
    let end = execution.first.checked_add(execution.count);
    if end.is_none_or(|end| end > execution.capacity) {
        return Err(refused(ReplayRefusal::RangeOutsideCapacity));
    }
    if execution.count > MAX_REPLAY_SLOTS {
        return Err(refused(ReplayRefusal::TooManyCommands {
            count: execution.count,
        }));
    }
    if variants.is_empty() || variants.len() > MAX_REPLAY_VARIANTS as usize {
        return Err(refused(ReplayRefusal::VariantCount {
            count: variants.len(),
        }));
    }
    if execution.regions == 0 || execution.regions > MAX_REPLAY_REGIONS {
        return Err(refused(ReplayRefusal::RegionCount {
            count: execution.regions,
        }));
    }
    // K14c's path records `commands * regions * variants` dispatches; the generated-commands
    // path writes no argument.
    let dispatches = if generated.is_some() {
        0
    } else {
        u64::from(execution.count) * u64::from(execution.regions) * variants.len() as u64
    };
    if dispatches > MAX_REPLAY_DISPATCHES {
        return Err(refused(ReplayRefusal::TooManyDispatches { dispatches }));
    }
    for (index, v) in variants.iter().enumerate() {
        let repeated = variants[..index].contains(v);
        let zero = v.threads_per_threadgroup.contains(&0);
        let lengths = v
            .threadgroup_memory_lengths
            .iter()
            .any(|l| !l.is_multiple_of(16));
        let imageblock = v.imageblock_size.is_some_and(|size| size.contains(&0));
        // A flat variant is its pipeline's only one, and reads no length or imageblock.
        let flat = v.flat_max_total_threads.is_some_and(|max| {
            !(1..=MAX_THREADGROUP_THREADS).contains(&max)
                || v.threadgroup_memory_lengths.iter().any(|&l| l != 0)
                || v.imageblock_size.is_some()
        });
        let shared = variants[..index].iter().any(|w| {
            w.pipeline == v.pipeline
                && (w.flat_max_total_threads.is_some() || v.flat_max_total_threads.is_some())
        });
        if repeated || zero || lengths || imageblock || flat || shared {
            return Err(refused(ReplayRefusal::InvalidVariant { index }));
        }
    }
    // The parent's pipeline and kernel buffers, in a record's layout (an empty record).
    let mut record = CommandRecord::default();
    record.bindings.pipeline = execution.parent.pipeline;
    *record.bindings.buffers_mut(BufferStage::Kernel) =
        *execution.parent.buffers(BufferStage::Kernel);
    let mut parent = vec![0u8; COMMAND_BYTES];
    record
        .encode_into(&mut parent)
        .map_err(|_| refused(ReplayRefusal::InvalidParent))?;

    let entries = resident_entries(state);
    let resident = stage_root_table(&[], 2, &entries);
    let extents = replay_extents(state, &entries);
    let slots = u64::from(execution.count);
    let count = variants.len() as u64;
    let regions = u64::from(execution.regions);
    let resident_offset = slots * SLOT_TABLE_BYTES + PARAMS_BYTES;
    // The extents right after the set, which the table holds from its root 2 to its end
    // (`memory::roots::ResidentView`).
    let extents_offset = resident_offset + resident.bytes.len() as u64;
    let variants_offset = section(extents_offset + extents.len() as u64);
    let parent_offset = section(variants_offset + count * VARIANT_BYTES);
    let regions_offset = section(parent_offset + COMMAND_BYTES as u64);
    let regions_end = regions_offset + slots * regions * REGION_BYTES;
    // The generated-commands path's templates, counts, and sequences: at most 64 templates and 2^17
    // sequences, each of at most 16 + 4 + 2^12 + 12 bytes.
    let planned = match generated {
        Some(plan) => Some(plan()?),
        None => None,
    };
    let (generated_layout, arguments_offset) = match &planned {
        Some(p) => {
            let templates_offset = section(regions_end);
            let template_bytes = p.replay.templates.len() as u64;
            let counts_offset = section(templates_offset + template_bytes);
            let sequences_offset = section(counts_offset + slots * 4);
            let end = sequences_offset + slots * regions * u64::from(p.stride);
            let g = GeneratedLayout {
                slots: execution.count,
                regions: execution.regions,
                stride: p.stride,
                push_bytes: p.push_bytes,
                segment_commands: p.segment_commands,
                templates_offset,
                counts_offset,
                sequences_offset,
            };
            (Some(g), section(end))
        }
        None => (None, section(regions_end)),
    };
    // At most 2^14 commands of 1,088 + 8 * 32 bytes, 2^20 dispatches of 12 bytes (or the
    // sequences above), and 2^16 resident entries of 32 + 8 bytes: far below 2^32, so every
    // offset fits a `u32` parameter.
    let bytes = section(arguments_offset + dispatches * ARGUMENT_BYTES);
    let layout = ReplayLayout {
        slots: execution.count,
        variants: count as u32,
        regions: execution.regions,
        resident_offset,
        extents_offset,
        variants_offset,
        parent_offset,
        regions_offset,
        arguments_offset,
        bytes,
        generated: generated_layout,
    };
    let mut flags = 0;
    if execution.inheritance.pipeline == StateSource::Parent {
        flags |= INHERIT_PIPELINE;
    }
    if execution.inheritance.buffers == StateSource::Parent {
        flags |= INHERIT_BUFFERS;
    }
    let words = [
        execution.first,
        execution.count,
        layout.variants,
        flags,
        (resident_offset - layout.table_offset(0)) as u32,
        (variants_offset - resident_offset) as u32,
        (parent_offset - resident_offset) as u32,
        (arguments_offset - resident_offset) as u32,
        execution.regions,
        (regions_offset - resident_offset) as u32,
        generated_layout.map_or(0, |g| g.stride),
        generated_layout.map_or(0, |g| g.push_bytes),
        generated_layout.map_or(0, |g| g.segment_commands),
        generated_layout.map_or(0, |g| (g.templates_offset - resident_offset) as u32),
        generated_layout.map_or(0, |g| (g.counts_offset - resident_offset) as u32),
        generated_layout.map_or(0, |g| (g.sequences_offset - resident_offset) as u32),
        device.metal_properties().max_threadgroup_memory_length,
    ];
    let mut params = vec![0u8; PARAMS_BYTES as usize];
    for (w, out) in words.iter().zip(params.as_chunks_mut::<4>().0.iter_mut()) {
        out.copy_from_slice(&w.to_le_bytes());
    }
    let mut table = vec![0u8; (count * VARIANT_BYTES) as usize];
    for (v, out) in variants.iter().zip(
        table
            .as_chunks_mut::<{ VARIANT_BYTES as usize }>()
            .0
            .iter_mut(),
    ) {
        let mut put = |at: u64, bytes: &[u8]| {
            out[at as usize..at as usize + bytes.len()].copy_from_slice(bytes);
        };
        put(V_TOKEN, &v.pipeline.get().to_le_bytes());
        for (i, x) in v.threads_per_threadgroup.iter().enumerate() {
            put(V_LOCAL + 4 * i as u64, &x.to_le_bytes());
        }
        for (i, x) in v.threadgroup_memory_lengths.iter().enumerate() {
            put(V_LENGTHS + 4 * i as u64, &x.to_le_bytes());
        }
        if let Some([width, height]) = v.imageblock_size {
            put(V_IMAGEBLOCK, &width.to_le_bytes());
            put(V_IMAGEBLOCK + 2, &height.to_le_bytes());
        }
        if let Some(max) = v.flat_max_total_threads {
            put(V_FLAT, &max.to_le_bytes());
        }
    }
    let compute = vk::ShaderStageFlags::COMPUTE;
    Ok(ReplayPreprocess {
        layout,
        params,
        resident,
        extents,
        variants: table,
        parent,
        push_constants: vec![0; ReplayPreprocessModule::PUSH_BYTES as usize],
        push_holes: vec![
            PushHole::RootTableAddress {
                stages: compute,
                offset: ReplayPreprocessModule::RESIDENT_PUSH,
            },
            PushHole::ExecutionStatusAddress {
                stages: compute,
                offset: ReplayPreprocessModule::STATUS_PUSH,
            },
            PushHole::CommandRecordsAddress {
                stages: compute,
                offset: ReplayPreprocessModule::RECORDS_PUSH,
            },
        ],
        group_count: [execution.count, 1, 1],
        generated: planned.map(|p| p.replay),
    })
}

/// The crate-owned preprocess module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayPreprocessModule {
    /// SPIR-V words, entry point `main`, no descriptor.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (device addresses and `shaderInt64`).
    pub requirements: Requirements,
    /// The group-count limits a replayed command's group count is checked against.
    pub limits: [u32; 3],
    /// The pipeline layout the consumer creates the preprocess's pipeline with: no descriptor
    /// set, and one `COMPUTE` push-constant range of [`ReplayPreprocessModule::PUSH_BYTES`] at
    /// offset 0.
    pub layout: PipelineLayout,
}

impl ReplayPreprocessModule {
    /// Bytes of the `COMPUTE` push-constant range at offset 0.
    pub const PUSH_BYTES: u32 = 24;
    /// The resident root table's device address (`u64`).
    pub const RESIDENT_PUSH: u32 = 0;
    /// The `ExecutionStatus` buffer's device address (`u64`).
    pub const STATUS_PUSH: u32 = 8;
    /// The record storage's device address (`u64`).
    pub const RECORDS_PUSH: u32 = 16;
}

/// The preprocess module of `device`: built with the typed builder, checked by the owned verifier
/// against the device's profile (requirements included), and serialized. A device without device
/// addresses is `Unsupported{GpuEncodedIndirectCommands}` naming what it lacks.
pub fn replay_preprocess_module(device: &Device) -> Result<ReplayPreprocessModule, Error> {
    let profile = device.profile();
    if profile.fields().buffer_access != BufferAccess::Addressed {
        return Err(Unsupported::needs_any_of(
            Feature::GpuEncodedIndirectCommands,
            vec![profile.addressed_needs()],
        )
        .into());
    }
    let limits = profile.fields().limits.max_compute_work_group_count;
    let m = typed_module(limits)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the replay preprocess needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &no_binding)?;
    Ok(ReplayPreprocessModule {
        spirv: words.into(),
        requirements: r.requirements,
        limits,
        layout: PipelineLayout {
            sets: Vec::new(),
            push_constants: vec![PushConstantRange {
                stages: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: ReplayPreprocessModule::PUSH_BYTES,
            }],
            push_descriptor_eligible: false,
        },
    })
}

/// The module declares no binding.
fn no_binding(_: &Symbol) -> Option<u32> {
    None
}

// ---- The module ----

/// The integer types the module uses.
#[derive(Clone, Copy)]
struct Types {
    u32_: TypeId,
    u64_: TypeId,
}

fn k32(fb: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = fb.u32_const(v)?;
    fb.module().const_value(c)
}

fn k64(fb: &mut FunctionBuilder<'_>, v: u64) -> BResult<Value> {
    roots::k64(fb, v)
}

/// `base + offset` (`u64`s).
fn at(fb: &mut FunctionBuilder<'_>, base: Value, offset: u64) -> BResult<Value> {
    if offset == 0 {
        return Ok(base);
    }
    let d = k64(fb, offset)?;
    fb.binary(BinOp::IAdd, base, d)
}

fn aligned(n: u32) -> MemAccess {
    MemAccess {
        aligned: Some(n),
        ..MemAccess::default()
    }
}

/// The `u32` at device address `addr` (4-aligned).
fn load32(fb: &mut FunctionBuilder<'_>, t: Types, addr: Value) -> BResult<Value> {
    let p = fb.device_place(addr, t.u32_)?;
    fb.load(p, aligned(4))
}

/// The `u64` at device address `addr` (8-aligned).
fn load64(fb: &mut FunctionBuilder<'_>, t: Types, addr: Value) -> BResult<Value> {
    let p = fb.device_place(addr, t.u64_)?;
    fb.load(p, aligned(8))
}

fn store32(fb: &mut FunctionBuilder<'_>, t: Types, addr: Value, v: Value) -> BResult<()> {
    let p = fb.device_place(addr, t.u32_)?;
    fb.store(p, v, aligned(4))
}

fn store64(fb: &mut FunctionBuilder<'_>, t: Types, addr: Value, v: Value) -> BResult<()> {
    let p = fb.device_place(addr, t.u64_)?;
    fb.store(p, v, aligned(8))
}

/// `x` (a `u32`) as a `u64`.
fn wide(fb: &mut FunctionBuilder<'_>, t: Types, x: Value) -> BResult<Value> {
    fb.convert(ConvOp::UConvert, x, t.u64_)
}

/// `*var |= c ? bits : 0` on the `u32` function variable `var`.
fn flag(fb: &mut FunctionBuilder<'_>, var: Value, c: Value, bits: u32) -> BResult<()> {
    let b = k32(fb, bits)?;
    let z = k32(fb, 0)?;
    let add = fb.select(c, b, z)?;
    let f = fb.load(var, MemAccess::default())?;
    let f = fb.binary(BinOp::BitwiseOr, f, add)?;
    fb.store(var, f, MemAccess::default())
}

/// A `u32` function variable initialized to `init`: its pointer.
fn counter(fb: &mut FunctionBuilder<'_>, t: Types, init: u32) -> BResult<Value> {
    let c = fb.u32_const(init)?;
    let var = fb.local_var(t.u32_, Some(c))?;
    fb.var_ptr(var)
}

/// The loop `for i in 0..n` (`n` a `u32` value) over `body(fb, i)`. Its index starts at 0 each
/// time the loop is entered, so a loop nested in another runs whole on every outer iteration (a
/// function variable is initialized once, at the function's entry).
fn count_loop<'a>(
    fb: &mut FunctionBuilder<'_>,
    t: Types,
    n: Value,
    body: impl FnOnce(&mut FunctionBuilder<'_>, Value) -> BResult<()> + 'a,
) -> BResult<()> {
    count_words(fb, t.u32_, n, body)
}

/// [`count_loop`] in a module of `u32`s alone (`u32_` its type), which declares no 64-bit type.
fn count_words<'a>(
    fb: &mut FunctionBuilder<'_>,
    u32_: TypeId,
    n: Value,
    body: impl FnOnce(&mut FunctionBuilder<'_>, Value) -> BResult<()> + 'a,
) -> BResult<()> {
    let start = fb.u32_const(0)?;
    let var = fb.local_var(u32_, Some(start))?;
    let index = fb.var_ptr(var)?;
    let zero = k32(fb, 0)?;
    fb.store(index, zero, MemAccess::default())?;
    let looped: LoopFn<'a, ()> = Box::new(move |fb, ctx, lc| {
        let i = fb.load(index, MemAccess::default())?;
        let done = fb.binary(BinOp::UGreaterThanEqual, i, n)?;
        let exit: ArmFn<'_, ()> = Box::new(move |fb, _| {
            lc.break_(fb)?;
            Ok(ArmEnd::Exits)
        });
        fb.if_(ctx, done, exit, None)?;
        let one = k32(fb, 1)?;
        let next = fb.binary(BinOp::IAdd, i, one)?;
        fb.store(index, next, MemAccess::default())?;
        body(fb, i)
    });
    fb.loop_(&mut (), LoopSpec::default(), looped)
}

/// Write zero into every argument of the command whose first argument (of `count`, one
/// `VkDispatchIndirectCommand` each: its regions times the variants) is at index
/// `first_argument` from `arguments`.
fn zero_arguments(
    fb: &mut FunctionBuilder<'_>,
    t: Types,
    (arguments, first_argument): (Value, Value),
    count: Value,
) -> BResult<()> {
    count_loop(fb, t, count, move |fb, v| {
        let v64 = wide(fb, t, v)?;
        let n = fb.binary(BinOp::IAdd, first_argument, v64)?;
        let bytes = k64(fb, ARGUMENT_BYTES)?;
        let rel = fb.binary(BinOp::IMul, n, bytes)?;
        let base = fb.binary(BinOp::IAdd, arguments, rel)?;
        let zero = k32(fb, 0)?;
        for i in 0..3 {
            let a = at(fb, base, 4 * i)?;
            store32(fb, t, a, zero)?;
        }
        Ok(())
    })
}

/// OR `REPLAY_FAULT` into the status word of the `ExecutionStatus` buffer at `status`, and
/// `faults` into its fault reasons.
fn report(fb: &mut FunctionBuilder<'_>, t: Types, status: Value, faults: Value) -> BResult<()> {
    let bits = k32(fb, StatusBits::REPLAY_FAULT.0)?;
    let word = at(fb, status, u64::from(StatusWord::Status.byte_offset()))?;
    let p = fb.device_place(word, t.u32_)?;
    fb.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, bits)?;
    let reasons = at(
        fb,
        status,
        u64::from(StatusWord::ReplayFaults.byte_offset()),
    )?;
    let p = fb.device_place(reasons, t.u32_)?;
    fb.atomic(AtomicOp::Or, p, Scope::Device, Semantics::RELAXED, faults)?;
    Ok(())
}

/// OR `RESIDENT_MISS` into the status word of the `ExecutionStatus` buffer at `status`, and add
/// `missed` (a `u32`) to its resident misses.
fn report_misses(
    fb: &mut FunctionBuilder<'_>,
    t: Types,
    status: Value,
    missed: Value,
) -> BResult<()> {
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
    Ok(())
}

/// Where the generated-commands path's sequences lie: the stream's address, the command's first
/// sequence (a `u64` index), the stride (a `u64`), and the dispatch token's offset in a sequence
/// (a `u64`).
#[derive(Clone, Copy)]
struct Stream {
    sequences: Value,
    first_sequence: Value,
    stride: Value,
    dispatch: Value,
}

impl Stream {
    /// The address of the command's sequence `r` (a `u32`).
    fn sequence(&self, fb: &mut FunctionBuilder<'_>, t: Types, r: Value) -> BResult<Value> {
        let r64 = wide(fb, t, r)?;
        let n = fb.binary(BinOp::IAdd, self.first_sequence, r64)?;
        let rel = fb.binary(BinOp::IMul, n, self.stride)?;
        fb.binary(BinOp::IAdd, self.sequences, rel)
    }
}

/// Write execution-set index 0 and a zero group count into each of the command's `regions`
/// sequences: they run nothing.
fn zero_sequences(
    fb: &mut FunctionBuilder<'_>,
    t: Types,
    stream: Stream,
    regions: Value,
) -> BResult<()> {
    count_loop(fb, t, regions, move |fb, r| {
        let seq = stream.sequence(fb, t, r)?;
        let zero = k32(fb, 0)?;
        store32(fb, t, seq, zero)?;
        let dispatch = fb.binary(BinOp::IAdd, seq, stream.dispatch)?;
        for i in 0..3 {
            let a = at(fb, dispatch, 4 * i)?;
            store32(fb, t, a, zero)?;
        }
        Ok(())
    })
}

/// Store the `u64` `v` at the 4-aligned device address `addr` as two `u32`s (little-endian).
fn store64_split(fb: &mut FunctionBuilder<'_>, t: Types, addr: Value, v: Value) -> BResult<()> {
    let low = fb.convert(ConvOp::UConvert, v, t.u32_)?;
    let shift = k64(fb, 32)?;
    let high = fb.binary(BinOp::ShiftRightLogical, v, shift)?;
    let high = fb.convert(ConvOp::UConvert, high, t.u32_)?;
    store32(fb, t, addr, low)?;
    let next = at(fb, addr, 4)?;
    store32(fb, t, next, high)
}

/// What a flat variant runs for one Metal dispatch ([`flat_dispatch`]).
#[derive(Clone, Copy)]
struct FlatDispatch {
    /// The dispatch runs: its grid has no zero axis, and its threadgroup no zero axis nor more
    /// threads than the state's maximum (what Metal runs nothing for).
    runs: Value,
    /// The dispatch cannot run (`GROUP_COUNT`): a threadgroup dispatch's grid above `2^32 - 1`
    /// threads on an axis, a grid of more than `2^64 - 1` threads, or one whose threadgroups of
    /// the variant's size no group count within the limits holds (or holds past `2^64 - 1`
    /// invocations).
    too_large: Value,
    /// Metal's `threads_per_grid`: the region record's `region[1].xyz`.
    threads: [Value; 3],
    /// The group count of the variant's dispatch.
    groups: [Value; 3],
}

/// `a * b` (two `u64`s, or two `u32`s with `zero` a `u32`) and whether it exceeds the type's
/// largest value: the double-width product's low and high words (`OpUMulExtended`), the product
/// exceeding it exactly where the high word is not `zero`. The crate's own modules test a
/// product's overflow in this form, and none divides the all-ones constant (a test checks every
/// one): NVIDIA's compiler (615.71.09) refuses a module that tests a `u64` product by dividing
/// (`b > (2^64 - 1) / a`) or a `u32` product by widening it to 64 bits and comparing it with
/// `2^32 - 1`, where those tests sit beside the preprocess's other arithmetic, and on an RTX 3090
/// one that tests a `u32` product by dividing (`b <= (2^32 - 1) / a`, the tessellation scan's
/// records count) ("NVVM compilation failed: 3"), and compiles this one.
pub(crate) fn checked_mul(
    fb: &mut FunctionBuilder<'_>,
    a: Value,
    b: Value,
    zero: Value,
) -> BResult<(Value, Value)> {
    let wide = fb.binary(BinOp::UMulExtended, a, b)?;
    let product = fb.composite_extract(wide, &[0])?;
    let high = fb.composite_extract(wide, &[1])?;
    let over = fb.binary(BinOp::INotEqual, high, zero)?;
    Ok((product, over))
}

/// The `bool` constant `v`.
fn boolean(fb: &mut FunctionBuilder<'_>, v: bool) -> BResult<Value> {
    let ty = fb.module().ty(Type::Bool)?;
    let c = fb.module().constant(Constant::Scalar {
        ty,
        bits: ConstBits(u64::from(v)),
    })?;
    fb.module().const_value(c)
}

/// `⌈n / d⌉` (`u64`s, `d` at least 1).
fn ceil_div64(fb: &mut FunctionBuilder<'_>, n: Value, d: Value) -> BResult<Value> {
    let zero = k64(fb, 0)?;
    let one = k64(fb, 1)?;
    let q = fb.binary(BinOp::UDiv, n, d)?;
    let r = fb.binary(BinOp::UMod, n, d)?;
    let partial = fb.binary(BinOp::INotEqual, r, zero)?;
    let extra = fb.select(partial, one, zero)?;
    fb.binary(BinOp::IAdd, q, extra)
}

/// The flat dispatch of a Metal dispatch of `grid` (threadgroups where `threadgroups`, threads
/// otherwise) in threadgroups of `local` threads, on a flat variant of `size` threads per
/// threadgroup (at least 1) for a state of at most `max_total` threads per threadgroup (1 to
/// 1,024), within the group-count `limits`. Everything that may pass `2^32 - 1` is computed in 64
/// bits and checked: Metal's grid in threads, its total, the `⌈total / size⌉` threadgroups spread
/// over `x` (up to its limit), then `y`, then `z`, and the invocations that group count dispatches,
/// whose index the variant computes. Reads nothing but its arguments, so any module that holds a
/// Metal dispatch's six words can derive its flat dispatch with it.
fn flat_dispatch(
    fb: &mut FunctionBuilder<'_>,
    t: Types,
    (threadgroups, grid, local): (Value, [Value; 3], [Value; 3]),
    (size, max_total): (Value, Value),
    limits: [u32; 3],
) -> BResult<FlatDispatch> {
    let zero32 = k32(fb, 0)?;
    let one32 = k32(fb, 1)?;
    let zero64 = k64(fb, 0)?;
    let one64 = k64(fb, 1)?;
    // The threadgroup: each axis 1 to the maximum (so the product of the clamped axes stays
    // below 2^30), and their product at most the maximum.
    let mut runs = boolean(fb, true)?;
    let mut volume = one32;
    for l in local {
        let empty = fb.binary(BinOp::IEqual, l, zero32)?;
        let above = fb.binary(BinOp::UGreaterThan, l, max_total)?;
        let bad = fb.binary(BinOp::LogicalOr, empty, above)?;
        let good = fb.unary(UnOp::LogicalNot, bad)?;
        runs = fb.binary(BinOp::LogicalAnd, runs, good)?;
        let axis = fb.select(good, l, one32)?;
        volume = fb.binary(BinOp::IMul, volume, axis)?;
    }
    let fits = fb.binary(BinOp::ULessThanEqual, volume, max_total)?;
    runs = fb.binary(BinOp::LogicalAnd, runs, fits)?;
    // Metal's grid in threads, each axis at most 2^32 - 1.
    let mut too_large = boolean(fb, false)?;
    let mut wide_threads = [one64; 3];
    let mut threads = [zero32; 3];
    for i in 0..3 {
        let empty = fb.binary(BinOp::IEqual, grid[i], zero32)?;
        let some = fb.unary(UnOp::LogicalNot, empty)?;
        runs = fb.binary(BinOp::LogicalAnd, runs, some)?;
        let g = wide(fb, t, grid[i])?;
        let l = wide(fb, t, local[i])?;
        let product = fb.binary(BinOp::IMul, g, l)?;
        wide_threads[i] = fb.select(threadgroups, product, g)?;
        let (_, over) = checked_mul(fb, grid[i], local[i], zero32)?;
        let beyond = fb.binary(BinOp::LogicalAnd, threadgroups, over)?;
        too_large = fb.binary(BinOp::LogicalOr, too_large, beyond)?;
        threads[i] = fb.convert(ConvOp::UConvert, wide_threads[i], t.u32_)?;
    }
    let [tx, ty, tz] = wide_threads;
    let (xy, over_xy) = checked_mul(fb, tx, ty, zero64)?;
    let (total, over_total) = checked_mul(fb, xy, tz, zero64)?;
    too_large = fb.binary(BinOp::LogicalOr, too_large, over_xy)?;
    too_large = fb.binary(BinOp::LogicalOr, too_large, over_total)?;
    // `⌈total / size⌉` threadgroups: as many along `x` as its limit allows, the rest along `y`
    // likewise, and the rest along `z`. A divisor of 0 (a grid that runs nothing) is taken as 1.
    let n = wide(fb, t, size)?;
    let none = fb.binary(BinOp::IEqual, n, zero64)?;
    let n = fb.select(none, one64, n)?;
    let mut left = ceil_div64(fb, total, n)?;
    let mut groups = [zero32; 3];
    let mut wide_groups = [one64; 3];
    for i in 0..3 {
        let limit = k64(fb, u64::from(limits[i].max(1)))?;
        let last = i == 2;
        let beyond = fb.binary(BinOp::UGreaterThan, left, limit)?;
        let along = if last {
            // What `z` cannot hold, no group count holds.
            too_large = fb.binary(BinOp::LogicalOr, too_large, beyond)?;
            left
        } else {
            fb.select(beyond, limit, left)?
        };
        let empty = fb.binary(BinOp::IEqual, along, zero64)?;
        let along = fb.select(empty, one64, along)?;
        wide_groups[i] = along;
        groups[i] = fb.convert(ConvOp::UConvert, along, t.u32_)?;
        if !last {
            left = ceil_div64(fb, left, along)?;
        }
    }
    // The invocations dispatched, whose last index the variant computes.
    let (xy, over_xy) = checked_mul(fb, wide_groups[0], wide_groups[1], zero64)?;
    let (xyz, over_xyz) = checked_mul(fb, xy, wide_groups[2], zero64)?;
    let (_, over_span) = checked_mul(fb, xyz, n, zero64)?;
    for over in [over_xy, over_xyz, over_span] {
        too_large = fb.binary(BinOp::LogicalOr, too_large, over)?;
    }
    Ok(FlatDispatch {
        runs,
        too_large,
        threads,
        groups,
    })
}

/// The preprocess module for `limits` as a typed module, before serialization (what
/// [`replay_preprocess_module`] serializes, and what the CPU interpreter runs). Its push fields
/// are `RootTableAddress` (the resident root table), `ExecutionStatusAddress`, and
/// `CommandRecordsAddress`, at [`ReplayPreprocessModule`]'s offsets.
pub fn typed_module(limits: [u32; 3]) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let u64_ = b.ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let bool_ = b.ty(Type::Bool)?;
    let t = Types { u32_, u64_ };
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let m_resident = b.push_member(
        PushField::RootTableAddress,
        u64_,
        ReplayPreprocessModule::RESIDENT_PUSH,
    )?;
    let m_status = b.push_member(
        PushField::ExecutionStatusAddress,
        u64_,
        ReplayPreprocessModule::STATUS_PUSH,
    )?;
    let m_records = b.push_member(
        PushField::CommandRecordsAddress,
        u64_,
        ReplayPreprocessModule::RECORDS_PUSH,
    )?;
    let push = b.push_global()?;
    let workgroup = b.builtin(crate::spv::BuiltIn::WorkgroupId, IoDir::Input, uvec3)?;
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
    let wg = fb.module().global_value(workgroup)?;
    let wg = fb.load(wg, MemAccess::default())?;
    let s = fb.composite_extract(wg, &[0])?;

    // The parameters, below the resident root table.
    let below = k64(fb, PARAMS_BYTES)?;
    let params = fb.binary(BinOp::ISub, resident, below)?;
    let param = |fb: &mut FunctionBuilder<'_>, offset: u64| -> BResult<Value> {
        let a = at(fb, params, offset)?;
        load32(fb, t, a)
    };
    let first = param(fb, P_FIRST)?;
    let slots = param(fb, P_SLOTS)?;
    let variants = param(fb, P_VARIANTS)?;
    let flags = param(fb, P_FLAGS)?;
    let tables_below = param(fb, P_TABLES_BELOW)?;
    let variants_at = param(fb, P_VARIANTS_AT)?;
    let parent_at = param(fb, P_PARENT_AT)?;
    let arguments_at = param(fb, P_ARGUMENTS_AT)?;
    let regions_n = param(fb, P_REGIONS)?;
    let regions_at = param(fb, P_REGIONS_AT)?;
    let stride = param(fb, P_STRIDE)?;
    let push_bytes = param(fb, P_PUSH_BYTES)?;
    let segment = param(fb, P_SEGMENT)?;
    let templates_at = param(fb, P_TEMPLATES_AT)?;
    let counts_at = param(fb, P_COUNTS_AT)?;
    let sequences_at = param(fb, P_SEQUENCES_AT)?;
    let max_length = param(fb, P_MAX_LENGTH)?;

    // An invocation past the range does nothing (the group count is the range's).
    let past = fb.binary(BinOp::UGreaterThanEqual, s, slots)?;
    let leave: ArmFn<'_, ()> = Box::new(|fb, _| {
        fb.ret(None)?;
        Ok(ArmEnd::Exits)
    });
    fb.if_(&mut (), past, leave, None)?;

    // The command's arguments: one per region and variant, from `s * regions * variants`.
    let s64 = wide(fb, t, s)?;
    let arguments_off = wide(fb, t, arguments_at)?;
    let arguments = fb.binary(BinOp::IAdd, resident, arguments_off)?;
    let variants64 = wide(fb, t, variants)?;
    let per_command = fb.binary(BinOp::IMul, regions_n, variants)?;
    let per_command64 = wide(fb, t, per_command)?;
    let first_argument = fb.binary(BinOp::IMul, s64, per_command64)?;

    // The generated-commands path (a nonzero stride): the command's sequences, one per region from
    // `s * regions`, and its execution's sequence count, instead of the arguments.
    let none32 = k32(fb, 0)?;
    let generated = fb.binary(BinOp::INotEqual, stride, none32)?;
    let sequences_off = wide(fb, t, sequences_at)?;
    let regions_wide = wide(fb, t, regions_n)?;
    let push_offset = k32(fb, GeneratedLayout::PUSH_OFFSET)?;
    let dispatch_rel = fb.binary(BinOp::IAdd, push_offset, push_bytes)?;
    let stream = Stream {
        sequences: fb.binary(BinOp::IAdd, resident, sequences_off)?,
        first_sequence: fb.binary(BinOp::IMul, s64, regions_wide)?,
        stride: wide(fb, t, stride)?,
        dispatch: wide(fb, t, dispatch_rel)?,
    };
    let counts_off = wide(fb, t, counts_at)?;
    let counts = fb.binary(BinOp::IAdd, resident, counts_off)?;
    let four64 = k64(fb, 4)?;
    let count_rel = fb.binary(BinOp::IMul, s64, four64)?;
    let sequence_count_at = fb.binary(BinOp::IAdd, counts, count_rel)?;
    let templates_off = wide(fb, t, templates_at)?;
    let templates = fb.binary(BinOp::IAdd, resident, templates_off)?;

    // A command whose index is not below the capacity the record storage's header states is not
    // read: it dispatches nothing and reports `RANGE` (the execution's capacity is not the
    // storage's). `first + s` is below the execution's capacity, a `u32`.
    let capacity_at = at(fb, records, layout::CAPACITY)?;
    let capacity = load32(fb, t, capacity_at)?;
    let index32 = fb.binary(BinOp::IAdd, first, s)?;
    let outside = fb.binary(BinOp::UGreaterThanEqual, index32, capacity)?;
    let refuse: ArmFn<'_, ()> = Box::new(move |fb, _| {
        // On the generated-commands path the command starts a segment of its own (the segment
        // before it ends at it, below), whose sequences run nothing.
        let sequences: ArmFn<'_, ()> = Box::new(move |fb, _| {
            zero_sequences(fb, t, stream, regions_n)?;
            store32(fb, t, sequence_count_at, regions_n)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        let arguments: ArmFn<'_, ()> = Box::new(move |fb, _| {
            zero_arguments(fb, t, (arguments, first_argument), per_command)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        fb.if_(&mut (), generated, sequences, Some(arguments))?;
        let range = k32(fb, ReplayFaults::RANGE.0)?;
        report(fb, t, status, range)?;
        fb.ret(None)?;
        Ok(ArmEnd::Exits)
    });
    fb.if_(&mut (), outside, refuse, None)?;

    let zero32 = k32(fb, 0)?;
    let one32 = k32(fb, 1)?;
    let two32 = k32(fb, 2)?;
    let three32 = k32(fb, 3)?;
    let zero64 = k64(fb, 0)?;
    let static_stride = k64(fb, u64::MAX)?;
    let no = fb.module().constant(Constant::Scalar {
        ty: bool_,
        bits: ConstBits(0),
    })?;
    let no = fb.module().const_value(no)?;
    let fault = counter(fb, t, 0)?;
    // The command's kernel buffers whose addresses no resident allocation holds.
    let misses = counter(fb, t, 0)?;

    // The record: `records + header + (first + s) * record bytes`.
    let first64 = wide(fb, t, first)?;
    let index = fb.binary(BinOp::IAdd, first64, s64)?;
    let stride = k64(fb, layout::RECORD)?;
    let rel = fb.binary(BinOp::IMul, index, stride)?;
    let rec = fb.binary(BinOp::IAdd, records, rel)?;
    let rec = at(fb, rec, COMMAND_BUFFER_HEADER_BYTES as u64)?;

    // The producer's sticky encoding fault refuses every command.
    let fault_at = at(fb, records, layout::FAULT)?;
    let header_fault = load32(fb, t, fault_at)?;
    let encoding = fb.binary(BinOp::INotEqual, header_fault, zero32)?;
    flag(fb, fault, encoding, ReplayFaults::ENCODING.0)?;

    // The action: empty, threadgroups, or threads run; a draw does not; anything else or a barrier
    // flag other than 0 or 1 is a record `CommandRecord::decode` rejects. No variant reads a
    // stage-in region (`ReplayRefusal::NotReplayVariant`), so a command's is ignored as Metal
    // ignores it for a kernel without `[[stage_in]]`; an imageblock size with one side 0 is
    // malformed.
    let action_at = at(fb, rec, layout::ACTION)?;
    let action = load32(fb, t, action_at)?;
    let empty = fb.binary(BinOp::IEqual, action, zero32)?;
    let groups_action = fb.binary(BinOp::IEqual, action, one32)?;
    let threads_action = fb.binary(BinOp::IEqual, action, two32)?;
    let draw = fb.binary(BinOp::IEqual, action, three32)?;
    let dispatch = fb.binary(BinOp::LogicalOr, groups_action, threads_action)?;
    let unknown = fb.binary(BinOp::UGreaterThan, action, three32)?;
    flag(fb, fault, draw, ReplayFaults::ACTION.0)?;
    let barrier_at = at(fb, rec, layout::BARRIER)?;
    let barrier = load32(fb, t, barrier_at)?;
    let mut malformed = fb.binary(BinOp::UGreaterThan, barrier, one32)?;
    malformed = fb.binary(BinOp::LogicalOr, malformed, unknown)?;
    // The imageblock size, width in the low half: unset (both 0) is 1 by 1.
    let imageblock_at = at(fb, rec, layout::IMAGEBLOCK)?;
    let imageblock_word = load32(fb, t, imageblock_at)?;
    let half = k32(fb, 0xffff)?;
    let sixteen = k32(fb, 16)?;
    let width = fb.binary(BinOp::BitwiseAnd, imageblock_word, half)?;
    let height = fb.binary(BinOp::ShiftRightLogical, imageblock_word, sixteen)?;
    let no_width = fb.binary(BinOp::IEqual, width, zero32)?;
    let no_height = fb.binary(BinOp::IEqual, height, zero32)?;
    let one_side = fb.binary(BinOp::LogicalNotEqual, no_width, no_height)?;
    malformed = fb.binary(BinOp::LogicalOr, malformed, one_side)?;
    let unset = fb.binary(BinOp::LogicalAnd, no_width, no_height)?;
    let one_by_one = k32(fb, 1 | 1 << 16)?;
    let imageblock = fb.select(unset, one_by_one, imageblock_word)?;
    // Each length as the plan path takes it: clamped to the device's maximum, then rounded up to
    // 16 bytes. The clamp keeps the sum below 2^32.
    let fifteen = k32(fb, 15)?;
    let aligned = k32(fb, !15)?;
    let mut lengths = Vec::with_capacity(THREADGROUP_COUNT);
    for i in 0..THREADGROUP_COUNT as u64 {
        let a = at(fb, rec, layout::THREADGROUP + 4 * i)?;
        let l = load32(fb, t, a)?;
        let over = fb.binary(BinOp::UGreaterThan, l, max_length)?;
        let l = fb.select(over, max_length, l)?;
        let l = fb.binary(BinOp::IAdd, l, fifteen)?;
        lengths.push(fb.binary(BinOp::BitwiseAnd, l, aligned)?);
    }
    flag(fb, fault, malformed, ReplayFaults::RECORD.0)?;
    let runs_any = fb.unary(UnOp::LogicalNot, empty)?;

    // The pipeline from its state source: a command that sets it while inheriting it, or a
    // nonempty command without one, is refused (`CommandBuffer::decode`,
    // `CommandRecord::resolve_bindings`).
    let pipeline_bit = k32(fb, INHERIT_PIPELINE)?;
    let buffers_bit = k32(fb, INHERIT_BUFFERS)?;
    let pf = fb.binary(BinOp::BitwiseAnd, flags, pipeline_bit)?;
    let inherit_pipeline = fb.binary(BinOp::INotEqual, pf, zero32)?;
    let bf = fb.binary(BinOp::BitwiseAnd, flags, buffers_bit)?;
    let inherit_buffers = fb.binary(BinOp::INotEqual, bf, zero32)?;
    let parent_off = wide(fb, t, parent_at)?;
    let parent = fb.binary(BinOp::IAdd, resident, parent_off)?;
    let own_token_at = at(fb, rec, layout::PIPELINE)?;
    let own_token = load64(fb, t, own_token_at)?;
    let parent_token_at = at(fb, parent, layout::PIPELINE)?;
    let parent_token = load64(fb, t, parent_token_at)?;
    let token = fb.select(inherit_pipeline, parent_token, own_token)?;
    let own_set = fb.binary(BinOp::INotEqual, own_token, zero64)?;
    let set_inherited = fb.binary(BinOp::LogicalAnd, inherit_pipeline, own_set)?;
    let none = fb.binary(BinOp::IEqual, token, zero64)?;
    let missing = fb.binary(BinOp::LogicalAnd, runs_any, none)?;
    let bad_pipeline = fb.binary(BinOp::LogicalOr, set_inherited, missing)?;
    flag(fb, fault, bad_pipeline, ReplayFaults::PIPELINE.0)?;
    // The pipeline word is the one the command executes with, a retained one included; a state word
    // other than 0 or `RETAINED`, or `RETAINED` beside no pipeline, is a record
    // `CommandRecord::decode` rejects.
    let state_at = at(fb, rec, layout::STATE)?;
    let state = load32(fb, t, state_at)?;
    let retained_bit = k32(fb, layout::RETAINED)?;
    let retained = fb.binary(BinOp::IEqual, state, retained_bit)?;
    let unnamed = fb.binary(BinOp::IEqual, own_token, zero64)?;
    let retained_none = fb.binary(BinOp::LogicalAnd, retained, unnamed)?;
    let flagged = fb.binary(BinOp::INotEqual, state, zero32)?;
    let unknown_state = fb.unary(UnOp::LogicalNot, retained)?;
    let unknown_state = fb.binary(BinOp::LogicalAnd, flagged, unknown_state)?;
    let bad_state = fb.binary(BinOp::LogicalOr, unknown_state, retained_none)?;
    flag(fb, fault, bad_state, ReplayFaults::RECORD.0)?;

    // A command that sets any buffer, of any stage, while inheriting buffers: a slot no setter
    // set holds an address and a stride of 0.
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
        let set = fb.binary(BinOp::LogicalAnd, inherit_buffers, set)?;
        flag(fb, fault, set, ReplayFaults::BUFFERS.0)
    })?;

    // The slot table: the null entry, then the sink entry naming the shared resident set.
    let below = wide(fb, t, tables_below)?;
    let tables = fb.binary(BinOp::ISub, resident, below)?;
    let table_bytes = k64(fb, SLOT_TABLE_BYTES)?;
    let table_rel = fb.binary(BinOp::IMul, s64, table_bytes)?;
    let table = fb.binary(BinOp::IAdd, tables, table_rel)?;
    for field in [roots::HOST, roots::GUEST, roots::SIZE, roots::BIND_OFFSET] {
        let a = at(fb, table, roots::NULL_ROOT * roots::ENTRY_BYTES + field)?;
        store64(fb, t, a, zero64)?;
    }
    let sink_entry = roots::SINK_ROOT * roots::ENTRY_BYTES;
    let sink_at = at(fb, resident, sink_entry + roots::HOST)?;
    let sink = load64(fb, t, sink_at)?;
    // `R0` of the slot table is the distance in entries from it to the resident set the resident
    // root table names, wherever it lies (`roots::ResidentView`), of its count.
    let view = roots::resident_view(fb, resident)?;
    let resident_count = view.count;
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

    // Each kernel buffer binding of the command's source, resolved in the resident set: its
    // allocation's entry with the address's offset in it as the bind offset, when the address is
    // below the entry's unpadded extent, so an allocation holds at least one byte there at its
    // own length (`BufferBinding::resolve_buffer` of 1 byte); zero when unbound, but for the
    // bind offset, which holds a nil buffer's address word, so the kernel sees the offset as
    // the buffer's address, as Metal's does. An address no allocation holds leaves its slot
    // unbound and counts a resident miss, as a replay of draws does. A dynamic stride of a bound
    // buffer is no replayed template's.
    let source = fb.select(inherit_buffers, parent, rec)?;
    let bindings = k32(fb, BINDING_COUNT as u32)?;
    count_loop(fb, t, bindings, |fb, n| {
        let n64 = wide(fb, t, n)?;
        let binding_bytes = k64(fb, layout::BINDING)?;
        let rel = fb.binary(BinOp::IMul, n64, binding_bytes)?;
        let kernel = k64(fb, layout::buffers(BufferStage::Kernel))?;
        let rel = fb.binary(BinOp::IAdd, rel, kernel)?;
        let src = fb.binary(BinOp::IAdd, source, rel)?;
        let guest = load64(fb, t, src)?;
        let stride_at = at(fb, src, layout::STRIDE)?;
        let stride = load64(fb, t, stride_at)?;
        // Below 2^32 an address names no buffer: a nil buffer's offset, or 0 unset.
        let first_address = k64(fb, BufferBinding::MIN_BUFFER_ADDRESS)?;
        let bound = fb.binary(BinOp::UGreaterThanEqual, guest, first_address)?;
        let dynamic = fb.binary(BinOp::INotEqual, stride, static_stride)?;
        let dynamic = fb.binary(BinOp::LogicalAnd, bound, dynamic)?;
        let dynamic = fb.binary(BinOp::LogicalAnd, runs_any, dynamic)?;
        flag(fb, fault, dynamic, ReplayFaults::BUFFERS.0)?;
        let live = fb.binary(BinOp::LogicalAnd, runs_any, bound)?;
        let (root, offset) = roots::resolve(fb, resident, guest)?;
        let null = k64(fb, roots::NULL_ROOT)?;
        let found = fb.binary(BinOp::INotEqual, root, null)?;
        let size = roots::entry_field(fb, resident, root, roots::SIZE)?;
        // The allocation's own extent, not its padded size: as the host's resolve.
        let extent = roots::extent_of(fb, &view, root, found)?;
        let inside = fb.binary(BinOp::ULessThan, offset, extent)?;
        let hit = fb.binary(BinOp::LogicalAnd, found, inside)?;
        let missed = fb.unary(UnOp::LogicalNot, hit)?;
        let miss = fb.binary(BinOp::LogicalAnd, live, missed)?;
        let one = k32(fb, 1)?;
        let add = fb.select(miss, one, zero32)?;
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

    // The command's threadgroup size and grid.
    let mut grid = [zero32; 3];
    let mut local = [zero32; 3];
    for i in 0..3 {
        let g = at(fb, rec, layout::ARGUMENTS + 4 * i as u64)?;
        grid[i] = load32(fb, t, g)?;
        let l = at(fb, rec, layout::ARGUMENTS + 4 * (3 + i) as u64)?;
        local[i] = load32(fb, t, l)?;
    }
    // A grid with a zero axis runs nothing, as in Metal and the host replay: it needs no variant,
    // and no size, count, or region check applies to it.
    let mut empty_grid = no;
    for g in grid {
        let zero_axis = fb.binary(BinOp::IEqual, g, zero32)?;
        empty_grid = fb.binary(BinOp::LogicalOr, empty_grid, zero_axis)?;
    }
    let nonempty = fb.unary(UnOp::LogicalNot, empty_grid)?;
    let runs_grid = fb.binary(BinOp::LogicalAnd, dispatch, nonempty)?;

    // A flat variant of the command's pipeline (registered as its only one): the command is one
    // region of its flat dispatch whatever its threadgroup size, lengths, and imageblock size, and
    // none of the regions below applies to it.
    let unset = k32(fb, u32::MAX)?;
    let variants_off = wide(fb, t, variants_at)?;
    let variants_base = fb.binary(BinOp::IAdd, resident, variants_off)?;
    let variant_entry = |fb: &mut FunctionBuilder<'_>, v: Value| -> BResult<Value> {
        let v64 = wide(fb, t, v)?;
        let bytes = k64(fb, VARIANT_BYTES)?;
        let rel = fb.binary(BinOp::IMul, v64, bytes)?;
        fb.binary(BinOp::IAdd, variants_base, rel)
    };
    let flat_selected = counter(fb, t, u32::MAX)?;
    count_loop(fb, t, variants, |fb, v| {
        let entry = variant_entry(fb, v)?;
        let token_at = at(fb, entry, V_TOKEN)?;
        let registered = load64(fb, t, token_at)?;
        let same = fb.binary(BinOp::IEqual, registered, token)?;
        let max_at = at(fb, entry, V_FLAT)?;
        let max = load32(fb, t, max_at)?;
        let flat = fb.binary(BinOp::INotEqual, max, zero32)?;
        let current = fb.load(flat_selected, MemAccess::default())?;
        let open = fb.binary(BinOp::IEqual, current, unset)?;
        let take = fb.binary(BinOp::LogicalAnd, same, flat)?;
        let take = fb.binary(BinOp::LogicalAnd, take, open)?;
        let next = fb.select(take, v, current)?;
        fb.store(flat_selected, next, MemAccess::default())
    })?;
    let flat_index = fb.load(flat_selected, MemAccess::default())?;
    let flat = fb.binary(BinOp::INotEqual, flat_index, unset)?;
    let fixed = fb.unary(UnOp::LogicalNot, flat)?;
    let read = fb.select(flat, flat_index, zero32)?;
    let entry = variant_entry(fb, read)?;
    let mut size = one32;
    for i in 0..3 {
        let a = at(fb, entry, V_LOCAL + 4 * i)?;
        let x = load32(fb, t, a)?;
        size = fb.binary(BinOp::IMul, size, x)?;
    }
    let max_at = at(fb, entry, V_FLAT)?;
    let max_total = load32(fb, t, max_at)?;
    let flat_grid = flat_dispatch(
        fb,
        t,
        (groups_action, grid, local),
        (size, max_total),
        limits,
    )?;
    let flat_runs = fb.binary(BinOp::LogicalAnd, flat, dispatch)?;
    let flat_runs = fb.binary(BinOp::LogicalAnd, flat_runs, flat_grid.runs)?;
    let flat_large = fb.binary(BinOp::LogicalAnd, flat_runs, flat_grid.too_large)?;
    flag(fb, fault, flat_large, ReplayFaults::GROUP_COUNT.0)?;

    // The regions, per axis: the whole threadgroups (the grid of a threadgroup dispatch, the grid
    // over the threadgroup size of a thread dispatch) in chunks of at most the device's group-count
    // limit, then a thread dispatch's remaining threads as one edge threadgroup. Each axis's piece
    // count is held to `regions + 1`, so their product (at most 729) is exact and tells whether the
    // command has more regions than the execution records. A threadgroup axis of 0 or above 1,024
    // (Metal's most, the packing) runs on no variant.
    let held = fb.binary(BinOp::IAdd, regions_n, one32)?;
    let most = k32(fb, 1 << NOMINAL_AXIS_BITS)?;
    let mut full = [zero32; 3];
    let mut rem = [zero32; 3];
    let mut chunks = [zero32; 3];
    let mut pieces = [zero32; 3];
    let mut threads_per_grid = [zero32; 3];
    let mut limit = [zero32; 3];
    let mut packed = zero32;
    let mut size_ok = fb.unary(UnOp::LogicalNot, no)?;
    let mut too_many_threads = no;
    for i in 0..3 {
        let is_zero = fb.binary(BinOp::IEqual, local[i], zero32)?;
        let above = fb.binary(BinOp::UGreaterThan, local[i], most)?;
        let bad = fb.binary(BinOp::LogicalOr, is_zero, above)?;
        let good = fb.unary(UnOp::LogicalNot, bad)?;
        size_ok = fb.binary(BinOp::LogicalAnd, size_ok, good)?;
        let divisor = fb.select(is_zero, one32, local[i])?;
        let quotient = fb.binary(BinOp::UDiv, grid[i], divisor)?;
        let remainder = fb.binary(BinOp::UMod, grid[i], divisor)?;
        full[i] = fb.select(threads_action, quotient, grid[i])?;
        rem[i] = fb.select(threads_action, remainder, zero32)?;
        limit[i] = k32(fb, limits[i].max(1))?;
        let whole = fb.binary(BinOp::UDiv, full[i], limit[i])?;
        let part = fb.binary(BinOp::UMod, full[i], limit[i])?;
        let partial = fb.binary(BinOp::INotEqual, part, zero32)?;
        let extra = fb.select(partial, one32, zero32)?;
        chunks[i] = fb.binary(BinOp::IAdd, whole, extra)?;
        let ragged = fb.binary(BinOp::INotEqual, rem[i], zero32)?;
        let edge = fb.select(ragged, one32, zero32)?;
        let n = fb.binary(BinOp::IAdd, chunks[i], edge)?;
        let over = fb.binary(BinOp::UGreaterThan, n, regions_n)?;
        pieces[i] = fb.select(over, held, n)?;
        // Metal's `threads_per_grid`: a thread dispatch's grid, or a threadgroup dispatch's
        // groups times its threadgroup size, which must fit 32 bits.
        let (product, beyond) = checked_mul(fb, grid[i], local[i], zero32)?;
        too_many_threads = fb.binary(BinOp::LogicalOr, too_many_threads, beyond)?;
        threads_per_grid[i] = fb.select(threads_action, grid[i], product)?;
        // The packing of the nominal size: `(L - 1) << (10 * axis)`.
        let minus = fb.binary(BinOp::ISub, local[i], one32)?;
        let shift = k32(fb, NOMINAL_AXIS_BITS * i as u32)?;
        let bits = fb.binary(BinOp::ShiftLeftLogical, minus, shift)?;
        packed = fb.binary(BinOp::BitwiseOr, packed, bits)?;
    }
    let yz = fb.binary(BinOp::IMul, pieces[1], pieces[2])?;
    let total = fb.binary(BinOp::IMul, pieces[0], yz)?;
    let too_many = fb.binary(BinOp::UGreaterThan, total, regions_n)?;
    let runs_fixed = fb.binary(BinOp::LogicalAnd, runs_grid, fixed)?;
    let bad_size = fb.unary(UnOp::LogicalNot, size_ok)?;
    let no_size = fb.binary(BinOp::LogicalAnd, runs_fixed, bad_size)?;
    flag(fb, fault, no_size, ReplayFaults::VARIANT.0)?;
    let wide_grid = fb.binary(BinOp::LogicalAnd, groups_action, too_many_threads)?;
    let wide_grid = fb.binary(BinOp::LogicalAnd, wide_grid, nonempty)?;
    let wide_grid = fb.binary(BinOp::LogicalAnd, wide_grid, fixed)?;
    flag(fb, fault, wide_grid, ReplayFaults::GROUP_COUNT.0)?;
    let over_regions = fb.binary(BinOp::LogicalAnd, runs_fixed, too_many)?;
    flag(fb, fault, over_regions, ReplayFaults::REGIONS.0)?;
    let fits = fb.unary(UnOp::LogicalNot, too_many)?;
    let splits = fb.binary(BinOp::LogicalAnd, runs_fixed, fits)?;
    let splits = fb.binary(BinOp::LogicalAnd, splits, size_ok)?;

    // Each region of the command: its record, its variant (the first registered one whose token,
    // threadgroup size, and threadgroup lengths are the command's pipeline, the region's size,
    // and the command's lengths; registration refuses repeats, so it is the only), and its
    // arguments, written as if the command were clean (a fault zeroes them below).
    let regions_off = wide(fb, t, regions_at)?;
    let regions_base = fb.binary(BinOp::IAdd, resident, regions_off)?;
    let regions64 = wide(fb, t, regions_n)?;
    let first_region = fb.binary(BinOp::IMul, s64, regions64)?;
    let mut pieces_at_least_one = [zero32; 3];
    for i in 0..3 {
        let none = fb.binary(BinOp::IEqual, pieces[i], zero32)?;
        pieces_at_least_one[i] = fb.select(none, one32, pieces[i])?;
    }
    count_loop(fb, t, regions_n, |fb, r| {
        // `r` in the order: `x` outermost, `z` innermost (an `x` index past its pieces is past
        // `total`, so inactive).
        let [_, py, pz] = pieces_at_least_one;
        let iz = fb.binary(BinOp::UMod, r, pz)?;
        let rest = fb.binary(BinOp::UDiv, r, pz)?;
        let iy = fb.binary(BinOp::UMod, rest, py)?;
        let ix = fb.binary(BinOp::UDiv, rest, py)?;
        let index = [ix, iy, iz];
        let mut base = [zero32; 3];
        let mut count = [zero32; 3];
        let mut size = [zero32; 3];
        for i in 0..3 {
            let edge = fb.binary(BinOp::UGreaterThanEqual, index[i], chunks[i])?;
            let chunk_base = fb.binary(BinOp::IMul, index[i], limit[i])?;
            base[i] = fb.select(edge, full[i], chunk_base)?;
            let left = fb.binary(BinOp::ISub, full[i], chunk_base)?;
            let capped = fb.binary(BinOp::ULessThan, left, limit[i])?;
            let chunk_count = fb.select(capped, left, limit[i])?;
            count[i] = fb.select(edge, one32, chunk_count)?;
            size[i] = fb.select(edge, rem[i], local[i])?;
        }
        let inside = fb.binary(BinOp::ULessThan, r, total)?;
        let active = fb.binary(BinOp::LogicalAnd, splits, inside)?;

        // Reset for each region: a function variable is initialized once.
        let selected = counter(fb, t, u32::MAX)?;
        fb.store(selected, unset, MemAccess::default())?;
        count_loop(fb, t, variants, |fb, v| {
            let entry = variant_entry(fb, v)?;
            let token_at = at(fb, entry, V_TOKEN)?;
            let registered = load64(fb, t, token_at)?;
            let mut same = fb.binary(BinOp::IEqual, registered, token)?;
            for (i, l) in size.iter().enumerate() {
                let a = at(fb, entry, V_LOCAL + 4 * i as u64)?;
                let x = load32(fb, t, a)?;
                let eq = fb.binary(BinOp::IEqual, x, *l)?;
                same = fb.binary(BinOp::LogicalAnd, same, eq)?;
            }
            for (i, l) in lengths.iter().enumerate() {
                let a = at(fb, entry, V_LENGTHS + 4 * i as u64)?;
                let x = load32(fb, t, a)?;
                let eq = fb.binary(BinOp::IEqual, x, *l)?;
                same = fb.binary(BinOp::LogicalAnd, same, eq)?;
            }
            // A variant that states no imageblock size (0) takes every command's.
            let a = at(fb, entry, V_IMAGEBLOCK)?;
            let x = load32(fb, t, a)?;
            let any = fb.binary(BinOp::IEqual, x, zero32)?;
            let eq = fb.binary(BinOp::IEqual, x, imageblock)?;
            let eq = fb.binary(BinOp::LogicalOr, any, eq)?;
            same = fb.binary(BinOp::LogicalAnd, same, eq)?;
            let current = fb.load(selected, MemAccess::default())?;
            let open = fb.binary(BinOp::IEqual, current, unset)?;
            let take = fb.binary(BinOp::LogicalAnd, same, open)?;
            let next = fb.select(take, v, current)?;
            fb.store(selected, next, MemAccess::default())
        })?;
        let matched = fb.load(selected, MemAccess::default())?;
        let unmatched = fb.binary(BinOp::IEqual, matched, unset)?;
        let missing = fb.binary(BinOp::LogicalAnd, active, unmatched)?;
        flag(fb, fault, missing, ReplayFaults::VARIANT.0)?;
        // A flat command runs its first region on its flat variant.
        let leading = fb.binary(BinOp::IEqual, r, zero32)?;
        let flat_go = fb.binary(BinOp::LogicalAnd, flat_runs, leading)?;
        let chosen = fb.select(flat, flat_index, matched)?;
        for (c, g) in count.iter_mut().zip(flat_grid.groups) {
            *c = fb.select(flat, g, *c)?;
        }

        // The region's record: the `DispatchRegion` value (a flat command's grid from its flat
        // dispatch, which its variant reads alone).
        let r64 = wide(fb, t, r)?;
        let n = fb.binary(BinOp::IAdd, first_region, r64)?;
        let bytes = k64(fb, REGION_BYTES)?;
        let rel = fb.binary(BinOp::IMul, n, bytes)?;
        let record = fb.binary(BinOp::IAdd, regions_base, rel)?;
        let mut grid_threads = threads_per_grid;
        for (x, f) in grid_threads.iter_mut().zip(flat_grid.threads) {
            *x = fb.select(flat, f, *x)?;
        }
        let words = [
            base[0],
            base[1],
            base[2],
            packed,
            grid_threads[0],
            grid_threads[1],
            grid_threads[2],
            zero32,
        ];
        for (i, w) in words.into_iter().enumerate() {
            let a = at(fb, record, 4 * i as u64)?;
            store32(fb, t, a, w)?;
        }

        // K14c's path: the region's arguments, its group count for its variant and zero for
        // every other. The generated-commands path: the region's sequence, its variant's index (0
        // where it runs nothing), the variant's push bytes with the command's slot table, the
        // region's record, and the status where the variant reads them, and its group count (or
        // zero), as K14c's path dispatches and pushes them.
        let found = fb.unary(UnOp::LogicalNot, unmatched)?;
        let go = fb.binary(BinOp::LogicalAnd, active, found)?;
        let go = fb.binary(BinOp::LogicalOr, go, flat_go)?;
        let region_argument = fb.binary(BinOp::IMul, n, variants64)?;
        let sequence: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let seq = stream.sequence(fb, t, r)?;
            let zero = k32(fb, 0)?;
            let v = fb.select(go, chosen, zero)?;
            store32(fb, t, seq, v)?;
            let v64 = wide(fb, t, v)?;
            let head = k32(fb, TEMPLATE_HEAD_BYTES as u32)?;
            let template_bytes = fb.binary(BinOp::IAdd, push_bytes, head)?;
            let template_bytes = wide(fb, t, template_bytes)?;
            let rel = fb.binary(BinOp::IMul, v64, template_bytes)?;
            let template = fb.binary(BinOp::IAdd, templates, rel)?;
            let body = at(fb, template, TEMPLATE_HEAD_BYTES)?;
            let push = at(fb, seq, u64::from(GeneratedLayout::PUSH_OFFSET))?;
            let two = k32(fb, 2)?;
            let words = fb.binary(BinOp::ShiftRightLogical, push_bytes, two)?;
            count_loop(fb, t, words, move |fb, w| {
                let w64 = wide(fb, t, w)?;
                let four = k64(fb, 4)?;
                let offset = fb.binary(BinOp::IMul, w64, four)?;
                let from = fb.binary(BinOp::IAdd, body, offset)?;
                let x = load32(fb, t, from)?;
                let to = fb.binary(BinOp::IAdd, push, offset)?;
                store32(fb, t, to, x)
            })?;
            for (k, address) in [(0, table), (1, record), (2, status)] {
                let hole_at = at(fb, template, 4 * k)?;
                let hole = load32(fb, t, hole_at)?;
                let absent = k32(fb, NO_HOLE)?;
                let present = fb.binary(BinOp::INotEqual, hole, absent)?;
                let fill: ArmFn<'_, ()> = Box::new(move |fb, _| {
                    let h64 = wide(fb, t, hole)?;
                    let to = fb.binary(BinOp::IAdd, push, h64)?;
                    store64_split(fb, t, to, address)?;
                    Ok(ArmEnd::FallThrough(Vec::new()))
                });
                fb.if_(&mut (), present, fill, None)?;
            }
            let dispatch = fb.binary(BinOp::IAdd, seq, stream.dispatch)?;
            for (i, g) in count.iter().enumerate() {
                let x = fb.select(go, *g, zero)?;
                let a = at(fb, dispatch, 4 * i as u64)?;
                store32(fb, t, a, x)?;
            }
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        let per_variant: ArmFn<'_, ()> = Box::new(move |fb, _| {
            count_loop(fb, t, variants, move |fb, v| {
                let v64 = wide(fb, t, v)?;
                let a = fb.binary(BinOp::IAdd, region_argument, v64)?;
                let bytes = k64(fb, ARGUMENT_BYTES)?;
                let rel = fb.binary(BinOp::IMul, a, bytes)?;
                let base = fb.binary(BinOp::IAdd, arguments, rel)?;
                let this = fb.binary(BinOp::IEqual, v, chosen)?;
                let this = fb.binary(BinOp::LogicalAnd, go, this)?;
                for (i, g) in count.iter().enumerate() {
                    let x = fb.select(this, *g, zero32)?;
                    let a = at(fb, base, 4 * i as u64)?;
                    store32(fb, t, a, x)?;
                }
                Ok(())
            })?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        fb.if_(&mut (), generated, sequence, Some(per_variant))?;
        Ok(())
    })?;

    // The generated-commands path's segments: a command starts one where its index in the range is
    // a multiple of the most commands of one execution or its record sets the barrier, and the
    // execution recorded there runs the commands up to the next start (or a command whose record
    // the storage does not hold, which starts its own), each command's regions; 0 elsewhere. At
    // most `segment - 1` later records' barrier words are read.
    let segments: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let zero = k32(fb, 0)?;
        let one = k32(fb, 1)?;
        let offset = fb.binary(BinOp::UMod, s, segment)?;
        let aligned = fb.binary(BinOp::IEqual, offset, zero)?;
        let barred = fb.binary(BinOp::INotEqual, barrier, zero)?;
        let start = fb.binary(BinOp::LogicalOr, aligned, barred)?;
        let run = counter(fb, t, 1)?;
        fb.store(run, one, MemAccess::default())?;
        let open = counter(fb, t, 1)?;
        fb.store(open, one, MemAccess::default())?;
        let later = fb.binary(BinOp::ISub, segment, one)?;
        count_loop(fb, t, later, move |fb, j| {
            let zero = k32(fb, 0)?;
            let one = k32(fb, 1)?;
            let k = fb.binary(BinOp::IAdd, j, one)?;
            let next = fb.binary(BinOp::IAdd, s, k)?;
            let inside = fb.binary(BinOp::ULessThan, next, slots)?;
            let m = fb.binary(BinOp::UMod, next, segment)?;
            let unaligned = fb.binary(BinOp::INotEqual, m, zero)?;
            let index = fb.binary(BinOp::IAdd, first, next)?;
            let stored = fb.binary(BinOp::ULessThan, index, capacity)?;
            let valid = fb.binary(BinOp::LogicalAnd, inside, unaligned)?;
            let valid = fb.binary(BinOp::LogicalAnd, valid, stored)?;
            // Its record where the storage holds it; this command's own otherwise (read, unused).
            let index64 = wide(fb, t, index)?;
            let record_bytes = k64(fb, layout::RECORD)?;
            let rel = fb.binary(BinOp::IMul, index64, record_bytes)?;
            let other = fb.binary(BinOp::IAdd, records, rel)?;
            let other = at(fb, other, COMMAND_BUFFER_HEADER_BYTES as u64)?;
            let read = fb.select(valid, other, rec)?;
            let barrier_at = at(fb, read, layout::BARRIER)?;
            let b = load32(fb, t, barrier_at)?;
            let clear = fb.binary(BinOp::IEqual, b, zero)?;
            let joins = fb.binary(BinOp::LogicalAnd, valid, clear)?;
            let o = fb.load(open, MemAccess::default())?;
            let still = fb.binary(BinOp::INotEqual, o, zero)?;
            let joins = fb.binary(BinOp::LogicalAnd, still, joins)?;
            let add = fb.select(joins, one, zero)?;
            let n = fb.load(run, MemAccess::default())?;
            let n = fb.binary(BinOp::IAdd, n, add)?;
            fb.store(run, n, MemAccess::default())?;
            fb.store(open, add, MemAccess::default())
        })?;
        let n = fb.load(run, MemAccess::default())?;
        let sequences = fb.binary(BinOp::IMul, n, regions_n)?;
        let c = fb.select(start, sequences, zero)?;
        store32(fb, t, sequence_count_at, c)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), generated, segments, None)?;

    // A refused command dispatches nothing and reports why.
    let faults = fb.load(fault, MemAccess::default())?;
    let faulted = fb.binary(BinOp::INotEqual, faults, zero32)?;
    let reported: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let sequences: ArmFn<'_, ()> = Box::new(move |fb, _| {
            zero_sequences(fb, t, stream, regions_n)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        let per_variant: ArmFn<'_, ()> = Box::new(move |fb, _| {
            zero_arguments(fb, t, (arguments, first_argument), per_command)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        fb.if_(&mut (), generated, sequences, Some(per_variant))?;
        report(fb, t, status, faults)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    // A command that runs counts its misses as a resident miss of the command buffer, as the
    // host's step counts an address no resident allocation holds.
    let counted: ArmFn<'_, ()> = Box::new(move |fb, _| {
        let missed = fb.load(misses, MemAccess::default())?;
        let any_miss = fb.binary(BinOp::INotEqual, missed, zero32)?;
        let count: ArmFn<'_, ()> = Box::new(move |fb, _| {
            report_misses(fb, t, status, missed)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        fb.if_(&mut (), any_miss, count, None)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    fb.if_(&mut (), faulted, reported, Some(counted))?;
    fb.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

#[cfg(test)]
mod tests;
