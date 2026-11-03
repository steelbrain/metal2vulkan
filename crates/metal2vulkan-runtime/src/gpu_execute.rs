//! Executing an indirect command buffer's compute commands on the GPU alone, where the GPU may
//! write their records (`executeCommandsInBuffer:withRange:`, the orchestrator's rulings of
//! 2026-10-06 and 2026-10-21).
//!
//! - **Totality, decided at commit.** An execution a host step would record (`indirect_execute.rs`:
//!   its buffer an encoder made resident for writing, or it lies inside a block of command
//!   predication) runs here where every record it may hold is covered: its range is given as
//!   such, not read from a buffer; a caller's check (`execute_commands_in_buffer_checked`) has
//!   nothing to see (`Execute::check_sees`), else the host step runs it before the commands reach
//!   the device; the buffer does not
//!   inherit the encoder's buffers; every pipeline state its records may name (the encoder's where
//!   the buffer inherits it, else every state made with `supportIndirectCommandBuffers` alive at
//!   the commit) compiles as a flat device-replay variant (`ComputePipelineState::replay_variant`:
//!   its kernel does not observe its threadgroup partition) whose plan binds no descriptor, and
//!   fetches no `[[stage_in]]` attribute, whose reach past 4 GiB the host step checks
//!   (`fetch_reach.rs`); no command of the range has a state the CPU set that the program released
//!   since, whose program the command runs while its record names the ID
//!   (`IndirectCommandBuffer::name_state`), and the variants of the states alive run the ID's
//!   holder instead; where the ID's holder as the GPU last could write the records is a state the
//!   program released since, whose program the buffer keeps (`released_programs.rs`), the ID names
//!   that program's variant, as Metal's record the GPU wrote runs it, unless the CPU set the state
//!   that took the ID since on a command of the range (`IndirectCommandBuffer::gpu_states`); and
//!   one of the two recordings below takes the variants. Each flat variant runs
//!   every grid and threadgroup size of its state, so the records' contents cannot leave a command
//!   uncovered. Otherwise the execution takes the host step (`indirect_execute.rs`, part 3a), which
//!   the device announces once (`Degradation::HostReplayedIndirectCommands`). Nothing is decided at
//!   run time, but for a checked execution whose check a route encoded since the commit gives
//!   something to see before its batch is submitted: its host step records it then
//!   (`Execute::outgrown`), from the states `commit` settled for it.
//! - **What runs.** One segment: the replay's preprocess (`bind::replay_preprocess_module`, an
//!   owned kernel) over the range, reading the records where the GPU and the CPU left them and
//!   writing each command's slot table, region record, and what the recording reads; a barrier;
//!   then the recording:
//!   - **generated commands** (`generated_execute.rs`), where the device enabled
//!     `VK_EXT_device_generated_commands` (`DeviceOptions::withhold_generated_commands` keeps it
//!     off) and the product sets the path up for the variants
//!     (`bind::replay_preprocess_generated`): per command one `vkCmdExecuteGeneratedCommandsEXT`
//!     of the sequences the preprocess wrote;
//!   - **indirect dispatches** (`dispatch_execute.rs`) on every other device, Vulkan 1.2's core
//!     (`bind::replay_preprocess`): per command of the range, in order, per variant its pipeline,
//!     its push bytes naming the command's slot table and region record, and one
//!     `vkCmdDispatchIndirect` of the group count the preprocess wrote for that command and
//!     variant: the command's own for the state its record names, zero for every other variant,
//!     and for a command that runs nothing. At most [`MAX_DISPATCHES`] (commands times variants)
//!     per execution, and at most [`DISPATCHES_PAST_UNORDERED`] commands times the variants past
//!     [`FREE_STATES_UNORDERED`] where no barrier goes between the commands
//!     ([`DISPATCHES_PAST_ORDERED`] past [`FREE_STATES_ORDERED`] where one does): past those the
//!     host step runs the execution, as it does it faster ([`dispatches_run`]).
//! - **Order.** In a serial encoder, a barrier between every two commands
//!   (`Barrier::between_dispatches`): the commands run in Metal's order, one after another, which
//!   orders them wherever their uses meet, as the records' uses are GPU data. In a concurrent
//!   encoder, Metal orders a command only after a barrier its record sets: where no route that
//!   wrote the buffer's records may set one (`IndirectCommandBuffer::command_writes`, no
//!   `CommandWrites::BARRIER`), the indirect dispatches go with no barrier between them, and else
//!   with one between every two. A route encoded after the commit can still run before the
//!   execution only where the execution's batch waits for an event its command buffer signals, so
//!   its batch is held until then ([`Rerecord`]): where the buffer's records may set a barrier by
//!   the time the batch is submitted, the commands are recorded again then, with the barriers, from
//!   what the commit found (no work on the GPU is waited for). The generated-commands path runs
//!   the commands of a segment its records leave without a barrier together, as Metal's concurrent
//!   commands may run.
//! - **Command predication.** An execution inside a block runs its preprocess by
//!   `vkCmdDispatchIndirect` from a gate the step before it writes (`predication.rs`,
//!   `Execute::preprocess_gate`): one invocation per command where the counter is 0, else none.
//!   Where the block skips, the preprocess writes nothing: every command's group counts (and
//!   sequence counts) stay the zeros the scratch was uploaded with, so no command runs, and nothing
//!   is reported, as Metal's skipped execution reports nothing.
//! - **Faults.** A record the preprocess cannot run (a header with the producers' fault, a record
//!   that does not decode, sets what its descriptor does not allow, or names no variant's state)
//!   runs nothing, and the command buffer's execution report states why, from the status words the
//!   preprocess wrote (`bind::ReplayFaults`), as the host step's report does.
//! - **What the commands reach.** The replay's resident set, which the preprocess resolves each
//!   record's buffer addresses in and the commands' kernels search for the pointers they load,
//!   holds the resources the command buffer's encoders made resident and the device's live
//!   buffers Metal reaches unnamed (`live_buffers.rs`, `BindState::resident_ranges`), as a directly
//!   encoded command's does: every buffer of its own, and a no-copy buffer or a heap's buffer once
//!   a command buffer's commands used it. Where the live buffers are past the search's capacity,
//!   the execution takes the host step.
//! - **Objects.** Per execution: the preprocess's scratch (a region of the command buffer's upload
//!   arena, `upload.rs`) and what the recording makes (`generated_execute.rs`,
//!   `dispatch_execute.rs`); the command buffer holds them until it completes. The execution record
//!   lists each variant once.

use super::argument_table::ArgumentTable;
use super::binding;
use super::blit::reads;
use super::command_buffer::{CommandBuffer, ReplayTable};
use super::degradation::CompiledPipeline;
use super::device::lock;
use super::dispatch_execute::{self, Template};
use super::encoder::{EncoderCore, RecordedSegment};
use super::error::{internal, Error};
use super::executed_commands::RecordsRead;
use super::execution::{ExecutedPipeline, Execution};
use super::fetch_reach;
use super::generated_execute;
use super::indirect_command_buffer::IndirectCommandBuffer;
use super::indirect_execute::{Execute, ExecutionRange};
use super::owned_kernel::OwnedModule;
use super::pipeline::{ComputePipelineState, Variant, VariantObjects};
use super::residency::Residency;
use super::resource::ResourceKey;
use super::tracker::{Barrier, Use};
use super::upload::Upload;
use super::Device;
use ash::vk;
use metal2vulkan::bind::{
    replay_preprocess, replay_preprocess_generated, BindState, BindingPlan, DispatchCommands,
    GeneratedVariant, NeutralResource, PredicationStep, PushHole, ReplayExecution,
    ReplayPreprocess, ReplayRefusal, ReplayVariant, ResidentRange, RootTable, StageTable,
};
use metal2vulkan::error::BindError;
use metal2vulkan::pipeline::{CommandWrites, Stage};
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

/// The threads of a flat variant's threadgroup: a wave on most devices, and within every
/// state's maximum but the smallest, which takes its maximum.
const FLAT_THREADS: u32 = 64;

/// The most indirect dispatches one execution records on the dispatches' path (commands times
/// variants): Metal's largest buffer's commands (16,384) of up to two states, or fewer commands of
/// more. Each costs the CPU one recorded dispatch and, on MoltenVK, a Metal dispatch the GPU runs
/// even at a zero group count (`dispatch_execute.rs`), so a range past it keeps the host step.
pub(super) const MAX_DISPATCHES: u64 = 1 << 15;

/// The variants (pipeline states the records may name) an execution the indirect dispatches run
/// may have, whatever its commands, where no barrier goes between them (a concurrent encoder's
/// whose records no route may set a barrier in, the module's order): with them every count of
/// commands ran faster on the GPU alone than in the host step. Each variant costs every command
/// one more indirect dispatch, which MoltenVK encodes as a Metal dispatch the GPU runs even at a
/// zero group count; past these, [`DISPATCHES_PAST_UNORDERED`] bounds those dispatches. Measured
/// with host_ab's icb workload on an M3 Max (macOS 27, MoltenVK 1.4.1), frame time on the GPU
/// alone over a host-step build's, interleaved runs: at 4,096 commands 0.53x with one variant,
/// 0.89x with four, 1.03x with five and 1.32x with eight; at 1,024 commands 0.88x with four,
/// 1.00x with five and 1.30x with eight; at 256 commands 0.90x and 0.96x with four, 1.06x and
/// 1.00x with five, 1.26x with eight.
pub(super) const FREE_STATES_UNORDERED: usize = 4;

/// The indirect dispatches past [`FREE_STATES_UNORDERED`] (the commands times the variants past
/// it) an execution whose commands go with no barrier between them may record on the GPU alone:
/// what the host step's round trip costs beside them, about 0.8 us of the GPU's time each against
/// 0.2 to 0.5 ms a frame. Measured as [`FREE_STATES_UNORDERED`] was: 256 commands of five variants
/// ran as fast on both paths, 4,096 of five and 1,024 of five 0.95x and 1.00x in the host step,
/// 256 of twelve 0.63x; 16 of twelve 1.28x and 16 and 64 of five 1.5x to 1.8x (slower) in the
/// host step.
pub(super) const DISPATCHES_PAST_UNORDERED: u64 = 256;

/// [`FREE_STATES_UNORDERED`] where a barrier goes between every two commands (a serial encoder's,
/// or a concurrent one's whose records may set one): MoltenVK ends its Metal compute encoder at
/// each barrier, which the host step's commands pay too, so a variant's dispatches weigh less
/// against it. Measured as [`FREE_STATES_UNORDERED`] was: at 256 commands 0.79x with one variant,
/// 0.90x with eight, 0.94x with twelve, 0.96x with fourteen and 1.06x with sixteen; at 64
/// commands 0.72x, 0.87x, 0.91x, 0.93x and 0.93x, and 1.02x with twenty-four; at 16 commands
/// 0.63x to 0.68x at every count up to sixteen, 0.85x with twenty-four. At 1,024 and 4,096
/// commands, where a Metal encoder a command bounds both paths, runs vary from 0.88x to 1.30x with
/// no trend in the count.
pub(super) const FREE_STATES_ORDERED: usize = 12;

/// [`DISPATCHES_PAST_UNORDERED`] where a barrier goes between every two commands, past
/// [`FREE_STATES_ORDERED`]: about 0.6 us of the GPU's time each against the host step's 0.3 ms a
/// frame. Measured as [`FREE_STATES_UNORDERED`] was: 256 commands of fourteen variants (512 past)
/// ran 0.96x and of sixteen 1.06x on the GPU alone, 64 of twenty-four (768 past) 1.02x, 16 of
/// twenty-four (192 past) 0.85x; 16, 64 and 256 commands of thirteen 1.51x, 1.15x and 1.04x
/// (slower) in the host step.
pub(super) const DISPATCHES_PAST_ORDERED: u64 = 512;

/// Whether the indirect dispatches run an execution of `commands` commands over `variants`
/// variants, with a barrier between every two commands where `ordered`: within
/// [`MAX_DISPATCHES`], and within its order's dispatches past its order's free variants. Both
/// paths run the commands exactly; the bounds choose the faster.
fn dispatches_run(commands: u32, variants: usize, ordered: bool) -> bool {
    let (free, past) = match ordered {
        true => (FREE_STATES_ORDERED, DISPATCHES_PAST_ORDERED),
        false => (FREE_STATES_UNORDERED, DISPATCHES_PAST_UNORDERED),
    };
    let commands = u64::from(commands);
    commands * variants.saturating_sub(free) as u64 <= past
        && commands * variants as u64 <= MAX_DISPATCHES
}

/// One variant of the execution: its compiled variant and plan, its replay commands, and its
/// registration.
pub(super) struct Prepared {
    pub(super) variant: Arc<Variant>,
    pub(super) plan: Arc<BindingPlan>,
    pub(super) commands: DispatchCommands,
    registered: ReplayVariant,
}

/// How the commands run once the preprocess wrote what they read. It keeps no copy of the
/// device's functions: the generated commands' are the device's, fetched as the commands are
/// recorded.
pub(super) enum Form {
    /// Through the device's generated-commands extension (`generated_execute.rs`).
    Generated,
    /// By indirect dispatches (`dispatch_execute.rs`), each variant's push template in the
    /// variants' order.
    Dispatches(Vec<Template>),
}

/// What [`total`] found: the variants, the replay's preprocess, the bind state they were
/// planned for, and the recording. Its resident set holds the live buffers as they lay then; the
/// command buffer's first submission points it at the set as it lies then, and holds the table
/// then current until it completes (`command_buffer/first_submission.rs`).
struct Total {
    prepared: Vec<Prepared>,
    pre: ReplayPreprocess,
    state: BindState,
    form: Form,
}

/// An execution `commit` recorded on the GPU alone by indirect dispatches with no barrier
/// between its commands, in a concurrent encoder, as no route that wrote its buffer's records
/// could set a barrier (the module's order): what `commit` found, kept until the batch is submitted
/// or the execution is recorded again with the barriers, where a route encoded since may set one.
pub(super) struct Rerecord {
    icb: IndirectCommandBuffer,
    total: Mutex<Option<Total>>,
}

impl Rerecord {
    /// Whether the commands must be recorded again, with a barrier between every two, before the
    /// batch is submitted: a route encoded since the commit may set a barrier in a record, and
    /// they were not recorded again yet.
    pub(super) fn due(&self) -> bool {
        self.icb.command_writes().may(CommandWrites::BARRIER) && lock(&self.total).is_some()
    }

    /// The recording of `execute` again into a segment of `command_buffer`, as `commit` found it,
    /// with a barrier between every two commands.
    pub(super) fn record(
        &self,
        command_buffer: &CommandBuffer,
        execute: &Execute,
    ) -> Result<RecordedSegment, Error> {
        let total = lock(&self.total)
            .take()
            .ok_or_else(|| internal("an execution recorded again twice"))?;
        let residency = command_buffer.residency()?.kernel();
        record_total(command_buffer, execute, &residency, &total, true)
    }
}

/// What the recordings share: the preprocess's scratch in the command buffer's arena (its device
/// address too), and the command buffer's status words' device address.
pub(super) struct Scratch {
    pub(super) upload: Upload,
    pub(super) address: u64,
    pub(super) status: u64,
}

/// The recording of `execute` on the GPU alone into a segment of `command_buffer`, where the
/// execution is total on its device (the module's rules), and, for one recorded with no barrier
/// between its commands, what its batch's submission needs to record it again with them; `None`
/// where it takes the host step. Refused as the device refuses an object it makes, or the arena an
/// upload.
pub(super) fn record(
    command_buffer: &CommandBuffer,
    execute: &Execute,
) -> Result<Option<(RecordedSegment, Option<Arc<Rerecord>>)>, Error> {
    // A checked execution's host step runs the caller's check first, where it may see something;
    // one inside a block of command predication runs here only from its preprocess's gate.
    if execute.check_sees() || (execute.predicated() && execute.preprocess_gate.is_none()) {
        return Ok(None);
    }
    let device = command_buffer.device();
    let residency = command_buffer.residency()?.kernel();
    // A serial encoder orders its commands wherever their uses meet; a concurrent one only after
    // a barrier a record sets.
    let ordered = execute.ordered || execute.icb.command_writes().may(CommandWrites::BARRIER);
    let Some(total) = total(device, execute, &residency, ordered)? else {
        return Ok(None);
    };
    // The GPU reads the records: the handlers learn the states they may name.
    command_buffer.report_executed(&execute.icb, RecordsRead::Gpu, &residency, || {
        let states = execute.states(device);
        let encode = states
            .iter()
            .any(ComputePipelineState::can_encode_indirect_commands);
        (Vec::new(), encode)
    });
    let barriers = match &total.form {
        Form::Dispatches(_) => ordered,
        Form::Generated => true,
    };
    let recorded = record_total(command_buffer, execute, &residency, &total, barriers)?;
    let rerecord = (!barriers).then(|| {
        Arc::new(Rerecord {
            icb: execute.icb.clone(),
            total: Mutex::new(Some(total)),
        })
    });
    Ok(Some((recorded, rerecord)))
}

/// `result`'s value; `None` where the binder refuses what the GPU-alone path asks for
/// (`Unsupported`, an invalid replay, or a threadgroup size that needs another variant): the
/// execution takes the host step. Any other error is the runtime's or the binder's.
fn refusal<T>(result: Result<T, BindError>) -> Result<Option<T>, Error> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(
            BindError::Unsupported(_)
            | BindError::InvalidReplay { .. }
            | BindError::NeedsVariant { .. },
        ) => Ok(None),
        Err(error) => Err(Error::Bind(error)),
    }
}

/// The variants, the replay's preprocess and the recording of `execute` on `device`, with
/// `residency`, its command buffer's, where every record it may hold is covered (the module's
/// rules), its commands with a barrier between every two where `ordered`; `None` where it is not.
fn total(
    device: &Device,
    execute: &Execute,
    residency: &Residency,
    ordered: bool,
) -> Result<Option<Total>, Error> {
    let ExecutionRange::Fixed(range) = &execute.range else {
        return Ok(None);
    };
    if execute.icb.descriptor().inherit_buffers || range.is_empty() {
        return Ok(None);
    }
    // A command whose state the CPU set and the program released since runs that state's
    // program while its record names the ID (`IndirectCommandBuffer::name_state`), which no
    // variant here runs: a state of the ID's live holder would.
    if execute.icb.sets_a_released_state(range.clone()) {
        return Ok(None);
    }
    let Ok(capacity) = u32::try_from(execute.icb.size()) else {
        return Ok(None);
    };
    // A replay binds no kernel buffer, and the command buffer's resident resources, every
    // encoder's (`residency.rs`). Its resident set holds the live buffers Metal reaches unnamed
    // besides (`live_buffers.rs`), listed apart from the used resources: a record's buffer address
    // and a pointer its kernel loads reach them, as Metal's do, of any count (Past 2^16 entries the
    // execution took the host step).
    let live: Vec<ResidentRange> = device
        .shared()
        .live_buffers
        .table(device)?
        .ranges()
        .collect();
    let mut state = BindState::default();
    state.stages.insert(Stage::Kernel, StageTable::default());
    state.used_resources = residency.used_resources();
    state.resident_ranges = live;
    let shape = state.shape();
    // Each state with the ID its records name it by: the encoder's where the buffer inherits it;
    // else the states alive, but where the records name the kept program of a state the program
    // released since the GPU last could write them (`IndirectCommandBuffer::gpu_states`).
    let states = match &execute.pipeline {
        Some(pipeline) => vec![(pipeline.gpu_resource_id().0, pipeline.clone())],
        None => match execute
            .icb
            .gpu_states(range.clone(), execute.states(device))
        {
            Some(states) => states,
            None => return Ok(None),
        },
    };
    // Where only the indirect dispatches could run it, none past their bounds, before a variant is
    // compiled for it.
    if device.shared().generated_commands.is_none()
        && !dispatches_run(range.len() as u32, states.len(), ordered)
    {
        return Ok(None);
    }
    let mut prepared = Vec::with_capacity(states.len());
    for (id, pipeline) in &states {
        match prepare(pipeline, *id, &state, &shape)? {
            Some(p) => prepared.push(p),
            None => return Ok(None),
        }
    }
    let mut execution = ReplayExecution::new(capacity, range.start, range.len() as u32);
    execution.inheritance = execute.icb.inheritance();
    execution.parent.pipeline = execute
        .pipeline
        .as_ref()
        .and_then(|pipeline| NonZeroU64::new(pipeline.gpu_resource_id().0));
    // A flat variant runs each command as one region.
    execution.regions = 1;
    let Some((pre, form)) = preprocess(device, &state, &execution, &prepared, ordered)? else {
        return Ok(None);
    };
    Ok(Some(Total {
        prepared,
        pre,
        state,
        form,
    }))
}

/// The replay's preprocess of `execution` over `prepared` for `state` on `device`, and the
/// recording that takes it: the generated commands where the device enabled them and the product
/// sets their path up, else the indirect dispatches, within their bounds for commands with a
/// barrier between every two where `ordered` ([`dispatches_run`]) and where each variant's push
/// bytes take the command's slot table and region record ([`Template::of`]). `None` where neither
/// takes it.
fn preprocess(
    device: &Device,
    state: &BindState,
    execution: &ReplayExecution,
    prepared: &[Prepared],
    ordered: bool,
) -> Result<Option<(ReplayPreprocess, Form)>, Error> {
    let registered: Vec<ReplayVariant> = prepared.iter().map(|p| p.registered).collect();
    if device.shared().generated_commands.is_some() {
        let generated: Vec<GeneratedVariant<'_>> = prepared
            .iter()
            .map(|p| GeneratedVariant::new(p.variant.pipeline(), &p.plan, &p.commands))
            .collect();
        let set_up = replay_preprocess_generated(
            device.compile_device(),
            state,
            execution,
            &registered,
            &generated,
        );
        match set_up {
            Ok(pre) => {
                if pre.generated.is_none() || pre.layout.generated.is_none() {
                    return Err(internal(
                        "a generated-commands replay without its sequences",
                    ));
                }
                return Ok(Some((pre, Form::Generated)));
            }
            // The product refuses the generated-commands path for these variants on this
            // device: the indirect dispatches remain.
            Err(BindError::InvalidReplay {
                reason: ReplayRefusal::GeneratedCommands { .. },
            }) => {}
            Err(error) => return refusal(Err(error)),
        }
    }
    if !dispatches_run(execution.count, prepared.len(), ordered) {
        return Ok(None);
    }
    let Some(templates) = prepared
        .iter()
        .map(Template::of)
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let Some(pre) = refusal(replay_preprocess(
        device.compile_device(),
        state,
        execution,
        &registered,
    ))?
    else {
        return Ok(None);
    };
    if pre.generated.is_some() || pre.layout.generated.is_some() {
        return Err(internal("a dispatches' replay with sequences"));
    }
    Ok(Some((pre, Form::Dispatches(templates))))
}

/// The flat replay variant of `pipeline`, which the records name by ID `id`, for `state` (of
/// `shape`), instantiated for its threadgroup size; `None` where it has none
/// (`ComputePipelineState::replay_variant`), its plan cannot run that size, or its kernel fetches
/// `[[stage_in]]` attributes.
fn prepare(
    pipeline: &ComputePipelineState,
    id: u64,
    state: &BindState,
    shape: &metal2vulkan::bind::BindShape,
) -> Result<Option<Prepared>, Error> {
    // A kernel that fetches `[[stage_in]]` attributes leaves the execution to the host step,
    // which checks each command's fetch past 4 GiB (`fetch_reach.rs`).
    if fetch_reach::fetches_stage_in(pipeline) {
        return Ok(None);
    }
    let Some((variant, plan)) = pipeline.replay_variant(shape)? else {
        return Ok(None);
    };
    // The variant runs threadgroups the device holds; a command runs nothing past Metal's limit
    // for the state alone, since the kernel does not observe Metal's threadgroups.
    let threads = [
        FLAT_THREADS
            .min(pipeline.pipeline_max_total_threads_per_threadgroup())
            .max(1),
        1,
        1,
    ];
    let max = pipeline.metal_max_total_threads_per_threadgroup();
    let Some(commands) = refusal(plan.instantiate_replay(state, threads))? else {
        return Ok(None);
    };
    let token = NonZeroU64::new(id)
        .ok_or_else(|| internal("an indirect command buffer's state without an ID"))?;
    Ok(Some(Prepared {
        variant,
        plan,
        commands,
        registered: ReplayVariant::flat(token, threads, max),
    }))
}

/// The scratch's initial bytes: the parameters, the resident root table with its sink entry's
/// `host` field at `sink`, the resident roots' extents, the variants, the parent's state and, on
/// the generated-commands path, the variants' templates, each at its layout offset; the slot
/// tables, region records, and arguments or sequences and counts are the preprocess's to write,
/// zero until it does.
fn scratch_bytes(pre: &ReplayPreprocess, sink: u64) -> Result<Vec<u8>, Error> {
    let layout = pre.layout;
    let len = usize::try_from(layout.bytes).map_err(|_| internal("a replay scratch too large"))?;
    let mut bytes = vec![0u8; len];
    let mut resident = pre.resident.bytes.clone();
    let at = RootTable::SINK_HOST_OFFSET as usize;
    resident
        .get_mut(at..at + 8)
        .ok_or_else(|| internal("a resident root table without a sink entry"))?
        .copy_from_slice(&sink.to_le_bytes());
    let templates = match (&pre.generated, &layout.generated) {
        (Some(replay), Some(generated)) => Some((generated.templates_offset, &replay.templates)),
        (None, None) => None,
        _ => return Err(internal("a replay of half the generated-commands path")),
    };
    for (offset, part) in [
        (layout.params_offset(), &pre.params),
        (layout.resident_offset, &resident),
        (layout.extents_offset, &pre.extents),
        (layout.variants_offset, &pre.variants),
        (layout.parent_offset, &pre.parent),
    ]
    .into_iter()
    .chain(templates)
    {
        let offset = offset as usize;
        bytes
            .get_mut(offset..offset + part.len())
            .ok_or_else(|| internal("a replay scratch section past the scratch"))?
            .copy_from_slice(part);
    }
    Ok(bytes)
}

/// Record the total execution `execute` (`total`, with `residency`, the command buffer's) into a
/// segment of `command_buffer` (the module's rules), indirect dispatches with a barrier between
/// every two commands where `barriers`.
fn record_total(
    command_buffer: &CommandBuffer,
    execute: &Execute,
    residency: &Residency,
    total: &Total,
    barriers: bool,
) -> Result<RecordedSegment, Error> {
    let device = command_buffer.device().clone();
    let Total {
        prepared,
        pre,
        state,
        form,
    } = total;
    let kernel = device.owned_kernel(OwnedModule::ReplayPreprocess)?;
    let no_address = || internal("a replay on a device without device addresses");
    let storage = execute.icb.storage();
    let records = storage.device_address().ok_or_else(no_address)?;
    let scratch = {
        let mut cb = command_buffer.state();
        let sink = cb
            .uploads
            .upload(&device, &[], u64::from(NeutralResource::SINK_BYTES))?
            .device_address()
            .ok_or_else(no_address)?;
        let bytes = scratch_bytes(pre, sink)?;
        // Its root tables start at device addresses that are multiples of 32 (`upload.rs`).
        let upload = cb
            .uploads
            .upload_aligned(&device, &bytes, pre.layout.bytes, 32)?;
        let status = cb.status_address(&device)?.ok_or_else(no_address)?;
        let address = upload.device_address().ok_or_else(no_address)?;
        Scratch {
            upload,
            address,
            status,
        }
    };
    let mut push = pre.push_constants.clone();
    for hole in &pre.push_holes {
        let (offset, address) = match *hole {
            PushHole::RootTableAddress { offset, .. } => {
                (offset, scratch.address + pre.layout.resident_offset)
            }
            PushHole::ExecutionStatusAddress { offset, .. } => (offset, scratch.status),
            PushHole::CommandRecordsAddress { offset, .. } => (offset, records),
            _ => {
                return Err(internal(
                    "a replay preprocess hole the runtime does not fill",
                ))
            }
        };
        push.get_mut(offset as usize..offset as usize + 8)
            .ok_or_else(|| internal("a replay preprocess hole past its push bytes"))?
            .copy_from_slice(&address.to_le_bytes());
    }
    // Inside a block of command predication the preprocess runs from its gate, made for one
    // invocation per command of the range (`Execute::preprocess_gate`), where the counter is 0.
    let gate = execute.preprocess_gate.clone();
    let uses = variant_uses(residency, prepared)?;
    let preprocess_uses = [reads(storage, 0..storage.length())];

    let mut core = EncoderCore::segment(command_buffer, execute.ordered)?;
    let compute = vk::PipelineStageFlags2::COMPUTE_SHADER;
    let group_count = pre.group_count;
    core.command(compute, &preprocess_uses, |raw, command_buffer| {
        // SAFETY: the segment's command buffer is recording; the kernel is the device's, and
        // its push bytes name the scratch, the status words and the records, which live until
        // the command buffer completes; the gate lies in the command buffer's arena, which has
        // indirect usage, and the step before the execution wrote it behind a barrier.
        unsafe {
            raw.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline,
            );
            raw.cmd_push_constants(
                command_buffer,
                kernel.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );
            match &gate {
                Some(gate) => raw.cmd_dispatch_indirect(
                    command_buffer,
                    gate.raw(),
                    gate.offset() + PredicationStep::GATE_OUT_OFFSET,
                ),
                None => {
                    let [x, y, z] = group_count;
                    raw.cmd_dispatch(command_buffer, x, y, z);
                }
            }
        }
    });
    // The preprocess's writes before the commands' preprocessing, indirect and shader reads; in a
    // serial encoder the commands need no barrier of the tracker's after the preprocess's read of
    // the records, which a command's resident write of the buffer would otherwise ask.
    core.barrier(Barrier::everything());
    core.uses().ordered_everything();
    let made = match form {
        Form::Generated => {
            let commands = device
                .shared()
                .generated_commands
                .as_ref()
                .ok_or_else(|| internal("generated commands on a device without them"))?;
            generated_execute::record(&mut core, (commands, pre, prepared), &scratch, &uses)?
        }
        Form::Dispatches(templates) => dispatch_execute::record(
            &mut core,
            (pre, prepared, templates.clone()),
            (&scratch, barriers),
            &uses,
        )?,
    };
    for (p, (objects, created_in)) in prepared.iter().zip(&made) {
        core.executed(|| {
            Execution::new(
                ExecutedPipeline::Compute(Arc::clone(p.variant.shared_pipeline())),
                Arc::new(state.shape()),
                vec![*created_in],
            )
        });
        announce(&device, p, objects);
    }
    // Its resident set as the live buffers and what its encoders made resident lie now: the
    // command buffer's first submission points its table at the set as it lies then.
    command_buffer.state().replay_table(ReplayTable {
        upload: scratch.upload.clone(),
        at: pre.layout.resident_offset,
        address: scratch.address + pre.layout.resident_offset,
    });
    core.retain(scratch.upload);
    let variants: Vec<Arc<Variant>> = prepared.iter().map(|p| Arc::clone(&p.variant)).collect();
    core.retain(variants);
    core.end_segment()
}

/// What the commands use: the resident resources each variant's plan reaches, resolved now, at
/// the commit, against `residency`, the command buffer's.
fn variant_uses(
    residency: &Residency,
    prepared: &[Prepared],
) -> Result<Vec<Use<ResourceKey>>, Error> {
    let empty = ArgumentTable::default();
    let stages = [(Stage::Kernel, &empty)];
    let tables = binding::Tables::new(&stages).with_residency(residency);
    let none = binding::Command {
        descriptor_bytes: &[],
        descriptor_ranges: &[],
        root_tables: Vec::new(),
        push_holes: &[],
        uses: &[],
        blobs: Vec::new(),
        mesh: None,
        tessellation: None,
    };
    let mut uses = Vec::new();
    for p in prepared {
        let command = binding::command_uses(
            "execute_commands_in_buffer",
            &p.plan,
            &none,
            &tables,
            None,
            None,
        )?;
        uses.extend(command.iter().cloned());
    }
    Ok(uses)
}

/// Announce on `device` what an execution that binds `p`'s variant for commands the GPU writes
/// announces, `objects` its layouts: the kinds its making announces at every occurrence, once an
/// execution, and what its bound textures and values degrade.
fn announce(device: &Device, p: &Prepared, objects: &VariantObjects) {
    if let [region] = p.commands.regions.as_slice() {
        objects.announce_bound(
            (
                std::slice::from_ref(&p.variant.pipeline().stage),
                CompiledPipeline::Kernel(p.variant.pipeline()),
            ),
            [(&region.specialization, None)],
        );
    }
    device.degrade_bound_formats(std::slice::from_ref(&p.variant.pipeline().stage), None);
    // The values the variant's plan pushes for the commands the GPU writes.
    for (n, region) in p.commands.regions.iter().enumerate() {
        if p.commands.regions[..n]
            .iter()
            .all(|earlier| earlier.specialization != region.specialization)
        {
            device.degrade_bound_values(
                std::slice::from_ref(&p.variant.pipeline().stage),
                &region.specialization,
                &|k| p.plan.specialized_value(k),
                None,
            );
        }
    }
    let stage = &p.variant.pipeline().stage;
    if stage
        .precision_notes
        .contains(&metal2vulkan::pipeline::PrecisionNote::ArgumentPixelSamplerScaling)
    {
        device.degrade_argument_pixel_samplers(
            std::slice::from_ref(stage),
            &device.argument_samplers(),
        );
    }
}
