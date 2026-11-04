//! Executing an indirect command buffer's compute commands in a compute encoder
//! (`executeCommandsInBuffer:withRange:`).
//!
//! - **When.** Metal's execution reads the records as it runs, so the CPU may write a command's
//!   record after the call. So the encoder records nothing for it at the call: it leaves an
//!   [`Execute`] as a segment ([`EncoderCore::defer`]), recorded into a `VkCommandBuffer` of its
//!   own ([`Execute::record`]), behind a barrier from everything before it and with one to
//!   everything after it, which goes between the encoder's two as its batch is submitted.
//!   `commit` records it from the CPU's records as they stand then, reading them a chunk of 256
//!   commands at a time and recording a chunk's before it reads the next, so a long range's
//!   records are recorded while the cache still holds them (`kept_commands::pieces`), but the
//!   range whole where its check, or the command buffer's executed-commands handlers, see every
//!   command before any is recorded; where the CPU writes the buffer's records again before the
//!   batch is ready to submit (an event wait holds it, [`Execute::rewritten`]), a host step
//!   records it again from the records as they stand then, waiting for no work on the GPU, and
//!   announcing nothing. A CPU write after the batch was submitted races the GPU, as on Metal.
//! - **Records the GPU may write**. A buffer a route that may write its records met
//!   (`IndirectCommandBuffer::gpu_written`, for good, `write_routes.rs`: its command buffer made it
//!   resident for writing and runs a function that may call an indirect-command operation, or a
//!   blit copied or reset commands into it) is written by functions as they run, so its
//!   records are known only once the work before the execution completed. Its segment starts a part
//!   of the encoder, and a batch, of its own. Where every record it may hold is covered, `commit`
//!   records it to run on the GPU alone (`gpu_execute.rs`); otherwise the scheduler submits the
//!   batch once every batch submitted before it on any queue completed and its host step recorded
//!   the segment from the records then (`scheduler`), which the device announces once
//!   (`Degradation::HostReplayedIndirectCommands`). So is a segment `commit` recorded from the
//!   CPU's records whose buffer such a route met after the commit ([`Unforeseen`]). The step
//!   compiles nothing: `commit`, or for such a segment the call that met the buffer (a later
//!   commit, a blit's, or a CPU setter's) before the scheduler sees the buffer written, settles on
//!   its own thread the
//!   variants every state a record may name needs, and the neutrals their plans bind
//!   ([`Execute::settle`]); the step makes only what a dispatch on the CPU makes (descriptor sets,
//!   upload chunks, views and samplers of bound textures) and the `VkPipeline` of a threadgroup
//!   size or threadgroup lengths a state has not dispatched before, from its compiled variant. A
//!   record names its pipeline state by its ID among the states made with
//!   `supportIndirectCommandBuffers` that lived at the commit (`ComputePipelineState::
//!   indirect_states`); an ID of none runs nothing, as Metal's undefined one may not. A record
//!   naming the ID of the state the CPU set on its command runs that state's program, though the
//!   program released the state since (`IndirectCommandBuffer::step_states`); one the GPU wrote,
//!   the state that held the ID as the GPU last could write the records, released since or not
//!   (`released_programs.rs`).
//! - **Each command** runs through the plan path as the encoder's dispatch would
//!   ([`record_dispatch`]): with its pipeline state, or the encoder's when the buffer inherits
//!   it; its kernel buffers, or the encoder's bound ones when the buffer inherits them; its
//!   threadgroup lengths, stage-in region (all zero: none set) and imageblock size (zero: 1 by
//!   1, as the encoder's until set); and the command buffer's resident resources, every
//!   encoder's, as the commit leaves them (`residency.rs`). A kernel buffer's attribute stride past
//!   32 bits, at an index a stage-in layout of `MTLBufferLayoutStrideDynamic` reads, is refused
//!   (`Unsupported{KernelStageIn, NotImplemented}`): Metal's answer is not measured. A kernel
//!   buffer set without a stride there, on a buffer whose descriptor supports dynamic attribute
//!   strides, is refused the runtime's own gap
//!   (`NotImplemented(RuntimeGap::IndirectCommandUnstridedBuffer)`, `unstrided_buffer`): plain
//!   Metal takes the command and completes it, and its fetch reads the buffer at stride 2^32 - 1,
//!   which the 32-bit fetch cannot place (the gap's doc). On a buffer without that support the
//!   records hold no stride word and Metal reads stride 0, which the plan path gives exactly, so
//!   the command is taken; and an inheriting buffer's command reads the encoder's buffer set
//!   without one at stride 0, as the encoder's own dispatch does. A command with no dispatch, no pipeline state, or a threadgroup
//!   that is empty or holds more threads than the state's `max_total_threads_per_threadgroup` runs
//!   nothing, as Metal's runs nothing, and its neighbours run (ruling a). A
//!   command whose dispatch the plan path refuses refuses the commit, with that error, or fails the
//!   command buffer where the host step records it (`CommandBufferError::Refused`, the same error
//!   typed in `CommandBuffer::refusal`).
//! - **Faults** (the host step's, as the device replay reports them, `bind::ReplayFaults`). A
//!   header that holds the producers' sticky encoding fault runs none of the commands
//!   (`ENCODING`); a record that does not decode, or sets what its descriptor does not allow,
//!   runs nothing (`RECORD`). The command buffer's execution report states them.
//! - **Buffers.** A command names each buffer by a Metal address, as Metal's does. The address
//!   resolves against the resources the command buffer's encoders made resident
//!   (`use_resource(s)`, `use_heap(s)`, before the execution or after it, in its encoder or
//!   another: Metal's residency is the command buffer's, `residency.rs`), then against the
//!   device's live buffers Metal reaches unnamed (`live_buffers.rs`): the buffer it lies in at the
//!   offset it lies at. One in none of them (a no-copy buffer or a heap's buffer no command
//!   buffer's commands used yet, or none) misses as a pointer loaded from memory does: its slot is
//!   unbound, so it reads zero and drops writes, and the command buffer's execution report counts
//!   the miss, one per slot.
//! - **Threadgroup lengths** are kept as set: one past the device's `max_threadgroup_memory_length`
//!   runs clamped to it (ruling c); one that is not a multiple of 16 bytes runs rounded up to one,
//!   as the plan rounds every length.
//! - **Checks.** An execution given a check (`execute_commands_in_buffer_checked`) calls it with
//!   each command it will record, before it records any, where what the buffer's records may hold
//!   leaves the check something to see ([`Execute::check_sees`]): `commit` with the CPU's records
//!   it read, a host step with the records as the GPU and the CPU left them. An execution `commit`
//!   recorded on the GPU alone (`gpu_execute.rs`), its check having nothing to see, whose check a
//!   route encoded since gives something to see takes the host step before its batch is submitted
//!   ([`Execute::outgrown`]).
//! - **Order.** Between commands, the segment orders as the encoder orders its dispatches (a
//!   serial encoder's where their uses meet; a concurrent one's not at all), and a command with
//!   its barrier set runs after every earlier command of the segment.

mod kept;

use super::argument_table::ArgumentTable;
use super::buffer::Buffer;
use super::command_buffer::CommandBuffer;
use super::compute::{
    prepare_variants, record_dispatch, validated_imageblock_size, DispatchTables,
};
use super::degradation::Degradation;
use super::device::lock;
use super::encoder::{EncoderCore, RecordedSegment};
use super::error::{internal, invalid, Error, RuntimeGap};
use super::executed_commands::{ExecutedCommand, RecordsRead};
use super::indirect_command_buffer::{
    IndirectCommandBuffer, IndirectCommandCheck, IndirectCommandCheckNeeds, Reader,
};
use super::kept_commands;
use super::pipeline::{BufferForms, ComputePipelineState, IndirectStates, Variant};
use super::predication::HostGate;
use super::residency::{whole_use, Residency};
use super::resource::ResourceKey;
use super::tracker::{Access, Barrier, EncoderUses, Summary, Use};
use super::upload::Upload;
use super::write_routes::replay_why;
use super::Device;
use ash::vk;
use metal2vulkan::bind::indirect_commands::{Action, BufferBinding, BufferStage, CommandRecord};
use metal2vulkan::bind::{Dispatch, ImageblockSize, ReplayFaults, StageInRegion};
use metal2vulkan::error::{BindError, Feature, Unsupported};
use metal2vulkan::pipeline::Stage;
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The call an execution's refusals name.
pub(super) const CALL: &str = "execute_commands_in_buffer";

/// The call an execution of a range read from a buffer names.
pub(super) const CALL_INDIRECT: &str = "execute_commands_in_buffer_indirect";

/// The bytes of an `MTLIndirectCommandBufferExecutionRange`: `location`, then `length`.
pub(super) const RANGE_BYTES: u64 = 8;

/// Refuse the execution range at `offset` of a buffer of `length` bytes where the runtime has no
/// exact answer (`execute_commands_in_buffer_indirect`). Metal's API validation layer stops
/// neither rule below and plain Metal refuses neither: the call always returns (an M3 Max under
/// macOS 27.0.1 with the layer off, `rangebuf.m`, `rangebuf2.m` and `rangebuf3.m`, one case per
/// process, on a compute encoder's dispatches and a render encoder's draws alike). Both are the
/// runtime's own gaps ([`RuntimeGap::ExecutionRangePastBuffer`],
/// [`RuntimeGap::ExecutionRangeOffFour`], which carry the measurements).
pub(super) fn check_range_buffer(length: u64, offset: u64) -> Result<(), Error> {
    let past = offset
        .checked_add(RANGE_BYTES)
        .is_none_or(|end| end > length);
    if past {
        return Err(Error::NotImplemented(
            RuntimeGap::ExecutionRangePastBuffer { offset, length },
        ));
    }
    if !offset.is_multiple_of(4) {
        return Err(Error::NotImplemented(RuntimeGap::ExecutionRangeOffFour {
            offset,
        }));
    }
    Ok(())
}

/// An execution of an indirect command buffer's commands, left to record later.
pub(super) struct Execute {
    pub(super) icb: IndirectCommandBuffer,
    /// The commands it runs.
    pub(super) range: ExecutionRange,
    /// The encoder's pipeline state, where the buffer inherits it.
    pub(super) pipeline: Option<ComputePipelineState>,
    /// The encoder's bound buffers where the buffer inherits them; else empty.
    pub(super) table: ArgumentTable,
    /// Whether the encoder runs its dispatches in encoded order.
    pub(super) ordered: bool,
    /// The check of the block of command predication it was encoded in, if any
    /// (`predication.rs`): a host step records it, and nothing where the counter skips.
    pub(super) gate: Option<HostGate>,
    /// Inside a block of command predication, for a range given as such
    /// ([`ExecutionRange::Fixed`]) that is not empty: the gate its preprocess runs from on the
    /// GPU-alone path (`gpu_execute.rs`), of one invocation per command of the range, which the
    /// step before it writes (`Predication::gates`).
    pub(super) preprocess_gate: Option<Upload>,
    /// The caller's check of the commands it records (`execute_commands_in_buffer_checked`), and
    /// what it looks for: `commit` or the host step calls it before it records them, where it may
    /// see something ([`check_sees`](Self::check_sees)), and then it keeps the execution off the
    /// GPU-alone path (`gpu_execute.rs`).
    pub(super) check: Option<(Mutex<IndirectCommandCheck>, IndirectCommandCheckNeeds)>,
    /// Where `commit` recorded it on the GPU alone with a check that had nothing to see then: the
    /// states its host step lets the records name, settled then ([`settle`](Self::settle)), for a
    /// route encoded since that gives the check something to see
    /// ([`outgrown`](Self::outgrown)). The step takes them.
    pub(super) unchecked: Mutex<Option<Vec<ComputePipelineState>>>,
    /// Where `commit` recorded it from the CPU's records, the count of the CPU's writes of them
    /// it read them after (`IndirectCommandBuffer::cpu_writes`); [`NOT_READ`] until then.
    pub(super) read_at: AtomicU64,
}

/// [`Execute::read_at`] of an execution `commit` did not record from the CPU's records.
pub(super) const NOT_READ: u64 = u64::MAX;

/// The commands an execution runs.
pub(super) enum ExecutionRange {
    /// `executeCommandsInBuffer:withRange:`'s, inside the buffer's size.
    Fixed(Range<u32>),
    /// `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`'s: a copy of the
    /// `MTLIndirectCommandBufferExecutionRange` (`location`, then `length`, each a `u32`), made
    /// where the execution is encoded behind a barrier from everything before
    /// (`indirect_dispatch::copy_arguments`), which the host step reads.
    Indirect(Buffer),
}

impl ExecutionRange {
    /// What a host step of an execution of `icb`'s commands over this range reads on the host of
    /// what the GPU may write, for the scheduler's order across queues
    /// (`CommandBufferShared::footprint`): the records' buffer, and the copy a range read from a
    /// buffer is read from, each read in every stage.
    pub(super) fn host_reads(&self, icb: &IndirectCommandBuffer) -> Vec<Use<ResourceKey>> {
        let read = Access::new(
            vk::PipelineStageFlags2::ALL_COMMANDS,
            vk::AccessFlags2::MEMORY_READ,
        );
        let mut reads = vec![whole_use(icb.storage(), read, Access::NONE)];
        if let ExecutionRange::Indirect(copy) = self {
            reads.push(whole_use(copy, read, Access::NONE));
        }
        reads
    }

    /// Whether the range is read from a buffer, which the GPU may write
    /// ([`ExecutionRange::Indirect`]): a host step records its execution, whoever wrote the
    /// records.
    pub(super) fn indirect(&self) -> bool {
        matches!(self, ExecutionRange::Indirect(_))
    }

    /// The commands it runs of a buffer of `size` commands: the range given, or the range the
    /// copy holds now as Metal's execution takes it (an M3 Max and an M4, for a compute encoder; a
    /// probe, the same chips, for a render encoder): from `location` to `location + length`,
    /// clamped at the buffer's size, so a location of 2 and a length of 5 in a buffer of 4 run
    /// commands 2 and 3, and a length of 0 or a location at the size runs nothing. Where the
    /// location lies past the size, or `location + length` passes `2^32 - 1` (Metal's GPU then
    /// faults the command buffer, or hangs), it runs nothing and reports `ReplayFaults::RANGE`.
    pub(super) fn commands(&self, size: u64) -> Result<(Range<u32>, ReplayFaults), Error> {
        let copy = match self {
            ExecutionRange::Fixed(range) => return Ok((range.clone(), ReplayFaults::default())),
            ExecutionRange::Indirect(copy) => copy,
        };
        let contents = copy
            .contents()
            .ok_or_else(|| internal("an execution range's copy without host bytes"))?;
        let mut bytes = [0u8; RANGE_BYTES as usize];
        // SAFETY: the copy is a `Shared` buffer of at least `RANGE_BYTES`, mapped; the GPU's
        // write to it completed before the host step (its batch ended with the barrier to the
        // host), and nothing writes it after.
        unsafe { std::ptr::copy_nonoverlapping(contents.as_ptr(), bytes.as_mut_ptr(), 8) };
        let [l0, l1, l2, l3, n0, n1, n2, n3] = bytes;
        let location = u64::from(u32::from_le_bytes([l0, l1, l2, l3]));
        let length = u64::from(u32::from_le_bytes([n0, n1, n2, n3]));
        if location > size || location + length > u64::from(u32::MAX) {
            return Ok((0..0, ReplayFaults::RANGE));
        }
        let end = (location + length).min(size);
        // Both lie within the size, which a buffer's `u32` command count holds.
        Ok((location as u32..end as u32, ReplayFaults::default()))
    }
}

/// What `commit` settled for an execution's host step ([`Execute::settle`]).
pub(super) struct Settled {
    /// The pipeline states its records may name.
    pub(super) states: Vec<ComputePipelineState>,
    /// The variants they need, held until the command buffer completes, so they stay live.
    pub(super) variants: Vec<Arc<Variant>>,
}

/// An execution `commit` recorded from the CPU's records. An encoder that makes its buffer resident
/// for writing before its batch is submitted settles its host step then, on that encoder's thread
/// and before the scheduler sees the buffer written (`IndirectCommandBuffer::written_by_gpu`), so
/// the step translates nothing.
pub(super) struct Unforeseen {
    execute: Arc<Execute>,
    /// The states made with `supportIndirectCommandBuffers` at the commit, held weakly: those its
    /// records may name (one made since names none); `None` where the buffer inherits the
    /// encoder's.
    listed: Option<IndirectStates>,
    /// Its command buffer's residency at the commit (`residency.rs`), without the live buffers.
    residency: Residency,
    /// What the settle made, once it ran.
    settled: Mutex<Option<Settled>>,
}

impl Unforeseen {
    /// The execution `execute` on `device`, as its commit leaves it, with its command buffer's
    /// `residency`.
    pub(super) fn new(
        device: &Device,
        execute: &Arc<Execute>,
        residency: Residency,
    ) -> Arc<Unforeseen> {
        let listed = execute
            .pipeline
            .is_none()
            .then(|| IndirectStates::list(device));
        Arc::new(Unforeseen {
            execute: Arc::clone(execute),
            listed,
            // The settle states variants of it alone: it holds no table of the live buffers.
            residency: residency.without_live(),
            settled: Mutex::new(None),
        })
    }

    /// Whether its host step was settled.
    pub(super) fn settled(&self) -> bool {
        lock(&self.settled).is_some()
    }

    /// Settle its host step on `device` ([`Execute::settle`]), unless it was: for the states
    /// listed at the commit that still live, or the encoder's.
    pub(super) fn settle(&self, device: &Device) {
        let mut settled = lock(&self.settled);
        if settled.is_none() {
            let states = self.live(device);
            *settled = Some(self.execute.settle(device, states, &self.residency));
        }
    }

    /// The states its host step lets the records name: those settled, else those listed at the
    /// commit that still live, or the encoder's.
    pub(super) fn states(&self, device: &Device) -> Vec<ComputePipelineState> {
        match &*lock(&self.settled) {
            Some(settled) => settled.states.clone(),
            None => self.live(device),
        }
    }

    fn live(&self, device: &Device) -> Vec<ComputePipelineState> {
        match &self.listed {
            Some(listed) => listed.live(),
            None => self.execute.states(device),
        }
    }
}

/// A command to run: its record, and the pipeline state it names (`None`: none, or the
/// encoder's where the buffer inherits it).
type Command = (CommandRecord, Option<ComputePipelineState>);

impl Execute {
    /// Record the commands into a segment of `command_buffer` (the module's rules): from the
    /// CPU's records as they stand now, or, for the host step, from the records as the GPU and
    /// the CPU left them, each naming its state among `states`.
    pub(super) fn record(
        &self,
        command_buffer: &CommandBuffer,
        states: Option<&[ComputePipelineState]>,
    ) -> Result<RecordedSegment, Error> {
        // Its host step inside a block of command predication that skips runs nothing.
        if states.is_some() && self.gate.as_ref().is_some_and(HostGate::skips) {
            return EncoderCore::segment(command_buffer, self.ordered)?.end_segment();
        }
        // Every encoder's resident resources, as the command buffer's are at the commit.
        let residency = command_buffer.residency()?.kernel();
        // Commands the CPU encoded, which their chunks' kept recordings may run (`kept.rs`).
        if states.is_none() {
            if let Some(keeping) = self.keeping(command_buffer, &residency) {
                return self.record_kept(command_buffer, &residency, keeping);
            }
        }
        let mut core = EncoderCore::segment(command_buffer, self.ordered)?;
        // The host step's commands take the large-buffer form, whose variants `settle` made.
        let forms = match states {
            Some(_) => BufferForms::Large,
            None => BufferForms::Bound,
        };
        let (range, mut faults) = self.range()?;
        // The caller's check sees each command the execution will record, before it records any,
        // where it may see something; so do the command buffer's executed-commands handlers,
        // which hear of the execution once. `commit` reads the rest a chunk at a time
        // (`kept_commands::pieces`), the host step the records as the GPU and the CPU left them.
        let check = self.check.as_ref().filter(|_| self.check_sees());
        let reports = command_buffer.reports_executed();
        let whole = states.is_some() || check.is_some() || reports;
        if states.is_none() {
            // Counted before the records are read: a write meanwhile records it again.
            self.read_at.store(self.icb.cpu_writes(), Ordering::Release);
        }
        let mut misses = 0u32;
        let forms = (&residency, forms);
        for piece in kept_commands::pieces(range, whole) {
            let start = piece.start;
            let Some(states) = states else {
                let (commands, released) = self.icb.records(piece, Reader::Commit)?;
                // The execution announces the commands it runs otherwise than Metal's may, as it
                // records them.
                self.icb.degrade_released(&released);
                if let Some((check, _)) = check {
                    let read = commands.iter().map(|(record, _)| record);
                    self.icb.check_read(check, start, read);
                }
                if reports {
                    let executed = commands.iter().map(Some);
                    let read = (RecordsRead::Commit, &residency);
                    self.report(command_buffer, read, start, executed);
                }
                self.dispatch(&mut core, &commands, forms, &mut misses)?;
                continue;
            };
            let (commands, written) = self.written_commands(piece, states, check)?;
            faults = ReplayFaults(faults.0 | written.0);
            if reports {
                let executed = commands.iter().map(Option::as_ref);
                let read = (RecordsRead::HostStep, &residency);
                self.report(command_buffer, read, start, executed);
            }
            self.dispatch(&mut core, commands.iter().flatten(), forms, &mut misses)?;
        }
        let mut segment = core.end_segment()?;
        segment.resident_misses = misses;
        segment.faults = faults;
        Ok(segment)
    }

    /// Hand the command buffer's executed-commands handlers what `executed` runs, commands from
    /// `start` (`None` where a record runs nothing for a fault), read at `read` under
    /// `residency`, the command buffer's (`executed_commands.rs`).
    fn report<'c>(
        &self,
        command_buffer: &CommandBuffer,
        (read, residency): (RecordsRead, &Residency),
        start: u32,
        executed: impl Iterator<Item = Option<&'c Command>>,
    ) {
        command_buffer.report_executed(&self.icb, read, residency, || {
            let commands = (u64::from(start)..)
                .zip(executed)
                .filter_map(|(index, command)| {
                    let (record, own) = command?;
                    let (pipeline, _) = self.runnable(record, own.as_ref())?;
                    Some(ExecutedCommand {
                        index,
                        can_encode_indirect_commands: pipeline.can_encode_indirect_commands(),
                    })
                })
                .collect();
            (commands, false)
        });
    }

    /// Record `commands`' dispatches through `core`, resolving against `residency`, the command
    /// buffer's, their variants in `forms` ([`record`](Self::record)'s), each resident-set miss
    /// counted in `misses`. A command's barrier orders it after the commands before it here.
    fn dispatch<'a>(
        &self,
        core: &mut EncoderCore,
        commands: impl IntoIterator<Item = &'a Command>,
        (residency, forms): (&Residency, BufferForms),
        misses: &mut u32,
    ) -> Result<(), Error> {
        let max_length = max_length(core);
        let mut ran = false;
        for (record, own) in commands {
            let Some((pipeline, dispatch)) = self.runnable(record, own.as_ref()) else {
                continue;
            };
            if record.barrier && ran {
                core.barrier(Barrier::between_dispatches());
            }
            let bindings = self.bindings(record, max_length, residency, misses);
            // A stride past 32 bits that a stage-in layout of `MTLBufferLayoutStrideDynamic`
            // reads: Metal's answer is not measured, and the fetch's `u32` displacement cannot
            // place it.
            if bindings
                .wide_strides
                .iter()
                .any(|&index| pipeline.reads_stage_in_stride(index))
            {
                return Err(Error::Bind(BindError::Unsupported(
                    Unsupported::not_implemented(Feature::KernelStageIn),
                )));
            }
            // A buffer set without a stride that such a layout reads, on a buffer that supports
            // dynamic attribute strides: Metal takes the command and its fetch reads at stride
            // 2^32 - 1, which the 32-bit fetch cannot place (`unstrided_buffer`). Without that
            // support Metal reads stride 0, which the plan path gives exactly.
            if let Some(refusal) = unstrided_buffer(
                self.icb.descriptor().support_dynamic_attribute_stride,
                bindings.unstrided.iter().copied(),
                |index| pipeline.reads_stage_in_stride(index),
            ) {
                return Err(refusal);
            }
            let tables = bindings.tables(self, pipeline, record, residency, forms)?;
            record_dispatch(core, &tables, CALL, dispatch)?;
            ran = true;
        }
        Ok(())
    }

    /// Settle, on the thread that commits or flags the buffer, what the host step needs to
    /// translate nothing, for `states`, those the records may name: each one's variants in the
    /// large-buffer form (`BufferForms::Large`: one whatever buffers a command binds, and its
    /// emulated-simdgroups twin where the kernel has one), and the device's neutrals their plans
    /// may bind. On a device without device addresses a variant still follows the bound
    /// buffers' lengths, so the step translates the variant of a shape it meets first. What
    /// cannot be settled now (a state the plan path refuses) is left to the step, which meets it
    /// as it is then.
    pub(super) fn settle(
        &self,
        device: &Device,
        states: Vec<ComputePipelineState>,
        residency: &Residency,
    ) -> Settled {
        let empty = ArgumentTable::default();
        let lengths = BTreeMap::new();
        let mut variants = Vec::new();
        for state in &states {
            let tables = DispatchTables {
                pipeline: state,
                table: if self.icb.descriptor().inherit_buffers {
                    &self.table
                } else {
                    &empty
                },
                threadgroup_lengths: &lengths,
                imageblock: ImageblockSize::new(1, 1),
                stage_in_region: None,
                residency,
                forms: BufferForms::Large,
            };
            variants.extend(prepare_variants(device, &tables).into_iter().flatten());
        }
        Settled { states, variants }
    }

    /// What its host step's recording may use that Metal orders, for the scheduler's order across
    /// queues (`CommandBufferShared::footprint`), with `residency`, its command buffer's: every
    /// resource made resident, with its usage (`Residency::bound_uses`), and what the step reads
    /// on the host ([`ExecutionRange::host_reads`]); and where the buffer inherits the encoder's
    /// buffers, what its commands reach of the encoder's bindings at the call
    /// (`ArgumentTable::bound_uses`). `None` where its commands may reach more that Metal orders:
    /// an inherited visible-function table; and inside a block of command predication, whose
    /// counter the step reads.
    pub(super) fn bound(&self, residency: &Residency) -> Option<Summary<ResourceKey>> {
        if self.gate.is_some() || self.preprocess_gate.is_some() {
            return None;
        }
        let inherited = match self.icb.descriptor().inherit_buffers {
            true => self.table.bound_uses()?,
            false => Vec::new(),
        };
        let mut uses = residency.bound_uses();
        uses.extend(self.range.host_reads(&self.icb));
        let aliases = &self.icb.device().shared().aliases;
        uses.extend(aliases.augment(&inherited).unwrap_or(inherited));
        let mut bound = EncoderUses::new(false);
        bound.record(vk::PipelineStageFlags2::ALL_COMMANDS, &uses, None);
        Some(bound.finish())
    }

    /// Whether it reads its range from a buffer, which the GPU may write
    /// ([`ExecutionRange::indirect`]): a host step records it, whoever wrote its records.
    pub(super) fn indirect(&self) -> bool {
        self.range.indirect()
    }

    /// Whether its check, if any, may see something among its commands, as its buffer's records
    /// may hold them now (`IndirectCommandCheckNeeds::sees`): commands under states that may
    /// encode indirect commands, where it looks for them (the encoder's state where the buffer
    /// inherits it, else those the records may name,
    /// `IndirectCommandBuffer::may_name_encoding_states`). A compute buffer's records hold no
    /// draw: a command of one runs no draw of points.
    pub(super) fn check_sees(&self) -> bool {
        let Some((_, needs)) = &self.check else {
            return false;
        };
        let needs = IndirectCommandCheckNeeds {
            point_draws: false,
            ..*needs
        };
        let encodes = || match &self.pipeline {
            Some(pipeline) => pipeline.can_encode_indirect_commands(),
            None => self.icb.may_name_encoding_states(),
        };
        let (points, encodes) = needs.sees(self.icb.command_writes(), encodes);
        points || encodes
    }

    /// Whether `commit` recorded it on the GPU alone with a check that had nothing to see, and a
    /// route encoded since gives the check something ([`check_sees`](Self::check_sees)): its host
    /// step records it instead, before its batch is submitted, calling the check.
    pub(super) fn outgrown(&self) -> bool {
        let unchecked = lock(&self.unchecked).is_some();
        unchecked && self.check_sees()
    }

    /// Where it is [`outgrown`](Self::outgrown), the states its host step lets the records name,
    /// taken: the step records it for good.
    pub(super) fn take_outgrown(&self) -> Option<Vec<ComputePipelineState>> {
        match self.outgrown() {
            true => lock(&self.unchecked).take(),
            false => None,
        }
    }

    /// Whether `commit` recorded it from the CPU's records and the CPU wrote the buffer's records
    /// since ([`read_at`](Self::read_at)): its host step records it again, from the records as
    /// they stand then, before its batch is submitted.
    pub(super) fn rewritten(&self) -> bool {
        let read_at = self.read_at.load(Ordering::Acquire);
        read_at != NOT_READ && self.icb.cpu_writes() != read_at
    }

    /// Whether it lies inside a block of command predication: a host step records it, once the
    /// steps before it left their counter (`predication.rs`), unless it runs on the GPU alone from
    /// its preprocess's gate (`gpu_execute.rs`).
    pub(super) fn predicated(&self) -> bool {
        self.gate.is_some()
    }

    /// The commands it runs ([`ExecutionRange::commands`]).
    fn range(&self) -> Result<(Range<u32>, ReplayFaults), Error> {
        self.range.commands(self.icb.size())
    }

    /// The pipeline states the records may name now: the encoder's where the buffer inherits
    /// it, else every state of `device` made with `supportIndirectCommandBuffers` alive now.
    pub(super) fn states(&self, device: &Device) -> Vec<ComputePipelineState> {
        match &self.pipeline {
            Some(pipeline) => vec![pipeline.clone()],
            None => ComputePipelineState::indirect_states(device),
        }
    }

    /// For the host step, the commands of `range`, the execution's, in order, `None` where a
    /// record runs nothing for a fault, and the faults: the records as the GPU and the CPU left
    /// them, each naming its state among `states`, each seen by `check` before any is recorded.
    fn written_commands(
        &self,
        range: Range<u32>,
        states: &[ComputePipelineState],
        check: Option<&(Mutex<IndirectCommandCheck>, IndirectCommandCheckNeeds)>,
    ) -> Result<(Vec<Option<Command>>, ReplayFaults), Error> {
        let written = self.icb.written_records(range.clone())?;
        if let Some((check, _)) = check {
            self.icb
                .check_recorded(check, range.start, &written.records);
        }
        // A record naming the ID of the state the CPU set runs its program, else the state of its
        // ID among `states` (`IndirectCommandBuffer::step_states`).
        let (named, released) = self.icb.step_states(range.start, &written.records, states);
        self.icb.degrade_released(&released);
        let commands = written
            .records
            .into_iter()
            .zip(named)
            .map(|(record, state)| record.map(|record| (record, state)))
            .collect();
        Ok((commands, written.faults))
    }

    /// The pipeline state and dispatch of `record`, which names `own`, where it runs on Metal:
    /// it dispatches, has a pipeline state, and its threadgroup holds at least one thread and no
    /// more than the state's `max_total_threads_per_threadgroup` (ruling a).
    fn runnable<'a>(
        &'a self,
        record: &CommandRecord,
        own: Option<&'a ComputePipelineState>,
    ) -> Option<(&'a ComputePipelineState, Dispatch)> {
        let dispatch = dispatch_of(record)?;
        let pipeline = self.pipeline.as_ref().or(own)?;
        runs(pipeline, &dispatch).then_some((pipeline, dispatch))
    }

    /// The command's own buffers, resolved against `residency`, the command buffer's (each miss
    /// counted in `misses`), and its threadgroup lengths, clamped to `max_length`.
    fn bindings(
        &self,
        record: &CommandRecord,
        max_length: u32,
        residency: &Residency,
        misses: &mut u32,
    ) -> Bindings {
        let descriptor = *self.icb.descriptor();
        let mut table = ArgumentTable::default();
        let mut wide_strides = Vec::new();
        let mut unstrided = Vec::new();
        if !descriptor.inherit_buffers {
            let bindings = record.bindings.buffers(BufferStage::Kernel);
            for (index, binding) in (0..descriptor.max_kernel_buffer_bind_count).zip(bindings) {
                let Some(binding) = binding else {
                    continue;
                };
                if !binding.names_buffer() {
                    degrade_nil_offset(self.icb.device(), binding, index);
                    continue;
                }
                let Some(bound) = residency.resolve(binding.address) else {
                    *misses = misses.wrapping_add(1);
                    continue;
                };
                table.buffers.insert(index, bound);
                // Only a stage-in layout of `MTLBufferLayoutStrideDynamic` reads the stride;
                // `record` refuses one past 32 bits there.
                match binding.attribute_stride.map(u32::try_from) {
                    Some(Ok(stride)) => {
                        table.attribute_strides.insert(index, stride);
                    }
                    Some(Err(_)) => wide_strides.push(index),
                    None => unstrided.push(index),
                }
            }
        }
        let lengths = (0..descriptor.max_kernel_threadgroup_memory_bind_count)
            .zip(record.threadgroup_memory_lengths)
            .filter(|&(_, length)| length != 0)
            .map(|(index, length)| (index, length.min(max_length)))
            .collect();
        Bindings {
            table,
            lengths,
            wide_strides,
            unstrided,
        }
    }
}

/// The refusal of an executed indirect command's buffers, where `unstrided` names the indices it
/// bound without an attribute stride and `reads_stride` the indices its pipeline state's layout of
/// `MTLBufferLayoutStrideDynamic` reads: the lowest index that is both, as
/// [`RuntimeGap::IndirectCommandUnstridedBuffer`], and `None` where no layout reads one. Plain
/// Metal takes such a command, so this is the runtime's own gap and not misuse (the gap's doc).
/// Both encoders' executions decide it so: a kernel buffer's stage-in layout here, a vertex
/// buffer's vertex-descriptor layout in `render/executions.rs`.
///
/// `dynamic_strides` is the indirect command buffer descriptor's
/// `support_dynamic_attribute_stride`. A buffer without it holds no stride word in its records at
/// all, so Metal's fetch reads stride 0, which the plan path gives exactly (`bind::dispatch` and
/// `bind::graphics` read a binding bound without a stride at stride 0): nothing is refused there,
/// whatever the layouts read.
pub(crate) fn unstrided_buffer(
    dynamic_strides: bool,
    unstrided: impl IntoIterator<Item = u32>,
    reads_stride: impl Fn(u32) -> bool,
) -> Option<Error> {
    if !dynamic_strides {
        return None;
    }
    unstrided
        .into_iter()
        .filter(|&index| reads_stride(index))
        .min()
        .map(|index| Error::NotImplemented(RuntimeGap::IndirectCommandUnstridedBuffer { index }))
}

/// Announce `Degradation::IndirectNilBufferOffset` for a command's `binding` at `index`, a nil
/// buffer's, where it was set at an offset whose low 32 bits are not 0: Metal's function sees
/// them as the buffer's address, the host replay's sees no buffer.
pub(super) fn degrade_nil_offset(device: &Device, binding: &BufferBinding, index: u32) {
    if binding.address == 0 {
        return;
    }
    device.degrade(Degradation::IndirectNilBufferOffset, || {
        (
            format!(
                "an indirect command's nil buffer at index {index}, set at offset {:#x}",
                binding.address
            ),
            "Metal's function sees the offset as the buffer's address; the execution, replayed \
             on the host, binds no buffer there"
                .to_owned(),
        )
    });
}

/// Announce, once per device, that an execution is replayed on the host (the module's records
/// the GPU may write, and ranges read from a buffer).
pub(super) fn degrade_host_replay(device: &Device, execute: &Execute) {
    // The buffer's records, which a route that may write them met, named (`write_routes.rs`).
    let why = if execute.icb.gpu_written() {
        None
    } else if execute.indirect() {
        Some(
            "an execution reads its range from a buffer \
             (executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:), which the GPU may \
             write",
        )
    } else if execute.predicated() {
        Some("an execution inside a block of command predication, whose condition is GPU memory")
    } else {
        // Recorded again for the CPU's writes since the commit (`Execute::rewritten`), waiting
        // for no work on the GPU.
        return;
    };
    device.degrade(Degradation::HostReplayedIndirectCommands, || {
        let why = match why {
            Some(why) => why.to_owned(),
            None => replay_why(&execute.icb, "the records an execution runs"),
        };
        (
            format!(
                "an indirect command buffer of {} commands",
                execute.icb.size()
            ),
            why,
        )
    });
}

/// A command's own buffers and threadgroup lengths ([`Execute::bindings`]).
struct Bindings {
    table: ArgumentTable,
    lengths: BTreeMap<u32, u32>,
    /// The kernel buffer indices bound with a stride past 32 bits.
    wide_strides: Vec<u32>,
    /// The kernel buffer indices bound without a stride.
    unstrided: Vec<u32>,
}

impl Bindings {
    /// The tables a command of `execute` with `record` dispatches `pipeline` with, its variant
    /// in `forms`: the encoder's buffers where the buffer inherits them, else its own. Refused as
    /// `setImageblockWidth:height:` refuses its imageblock size.
    fn tables<'a>(
        &'a self,
        execute: &'a Execute,
        pipeline: &'a ComputePipelineState,
        record: &CommandRecord,
        residency: &'a Residency,
        forms: BufferForms,
    ) -> Result<DispatchTables<'a>, Error> {
        let table = if execute.icb.descriptor().inherit_buffers {
            // A kernel that statically uses an inherited texture slot that holds a memoryless
            // texture is the runtime's gap: Metal's answer for an execution is not measured (an
            // encoder's own dispatch fails the command buffer, `ComputeCommandEncoder::
            // dispatch_threadgroups`).
            if (execute.table).uses_memoryless(pipeline.interface(), Stage::Kernel) {
                return Err(Error::NotImplemented(RuntimeGap::CommandReplay));
            }
            &execute.table
        } else {
            &self.table
        };
        Ok(DispatchTables {
            pipeline,
            table,
            threadgroup_lengths: &self.lengths,
            imageblock: imageblock(pipeline, record.imageblock_size)?,
            stage_in_region: stage_in_region(record.stage_in_region),
            residency,
            forms,
        })
    }
}

/// The device's `max_threadgroup_memory_length`, as a record's lengths are clamped to it.
fn max_length(core: &EncoderCore) -> u32 {
    let max_length = core.device().properties().max_threadgroup_memory_length;
    u32::try_from(max_length).unwrap_or(u32::MAX)
}

/// The dispatch `record` makes; `None` for a command that dispatches nothing.
fn dispatch_of(record: &CommandRecord) -> Option<Dispatch> {
    match record.action {
        Action::DispatchThreadgroups { grid, local_size } => Some(Dispatch::Threadgroups {
            groups: grid,
            threads_per_group: local_size,
        }),
        Action::DispatchThreads { grid, local_size } => Some(Dispatch::Threads {
            threads: grid,
            threads_per_group: local_size,
        }),
        Action::Empty
        | Action::DrawPrimitives { .. }
        | Action::DrawIndexedPrimitives { .. }
        | Action::DrawPatches { .. }
        | Action::DrawIndexedPatches { .. }
        | Action::DrawMeshThreadgroups { .. }
        | Action::DrawMeshThreads { .. } => None,
    }
}

/// Whether `dispatch` of `pipeline` runs on Metal: its threadgroup holds at least one thread and
/// no more than Metal's limit for the state (ruling a). One within it but past the state's
/// `max_total_threads_per_threadgroup`, where this device's limit is lower, runs on Metal: its
/// plan refuses it typed as the device's gap.
fn runs(pipeline: &ComputePipelineState, dispatch: &Dispatch) -> bool {
    let threads: u64 = dispatch
        .threads_per_group()
        .iter()
        .map(|&axis| u64::from(axis))
        .product();
    threads != 0 && threads <= u64::from(pipeline.metal_max_total_threads_per_threadgroup())
}

/// The imageblock size a command of `pipeline` runs with: 1 by 1 for a kernel without an
/// explicit imageblock or a record that sets none (both zero); else the size set, refused as
/// Metal's API validation refuses `setImageblockWidth:height:` (an indirect command's plain
/// answer is unmeasured, where the encoder's is measured).
fn imageblock(
    pipeline: &ComputePipelineState,
    [width, height]: [u16; 2],
) -> Result<ImageblockSize, Error> {
    if pipeline.imageblock_cell_bytes().is_none() || (width == 0 && height == 0) {
        return Ok(ImageblockSize::new(1, 1));
    }
    validated_imageblock_size(width.into(), height.into()).map_err(|problem| invalid(CALL, problem))
}

/// The stage-in region a record sets: origin, then size; `None` when all zero (none set).
fn stage_in_region([x, y, z, width, height, depth]: [u32; 6]) -> Option<StageInRegion> {
    let region = [x, y, z, width, height, depth];
    (region != [0; 6]).then(|| StageInRegion::new([x, y, z], [width, height, depth]))
}
