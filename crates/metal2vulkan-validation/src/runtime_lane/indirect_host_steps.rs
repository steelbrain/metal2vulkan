//! Executions of indirect command buffers whose compute commands a kernel writes
//! (`execute_commands_in_buffer` after `use_resource` with `WRITE`) on the device the cfg selects,
//! under the validation layer. The owned `icb_replay_encode` fixture's kernel writes the commands
//! (thread `tid` writes command `tid`: the `icb_replay_template` state, its 256-byte windows of
//! `params` and `out`, a 32-byte threadgroup length, 64 threads in groups of 8), and the commands
//! run as Metal runs them:
//!
//! - in the encoder that wrote them, a later encoder, a later command buffer, and on another
//!   queue after an event, the execution's command buffer committed before the writer's; the
//!   writer's state states that it can encode indirect commands, the template's that it cannot;
//! - for a buffer made resident for writing only after the execution's commit (a step `commit`
//!   did not foresee);
//! - in the large-buffer form (docs/API.md section 10), which within the windows reads and writes
//!   as the same commands written by the CPU do;
//! - a second execution makes no `VkPipeline`;
//! - a sticky encoding fault in the header runs no command, and the execution report says so;
//! - an execution reads its range from a buffer as it runs;
//! - a checked execution's host step checks each command before it records any, from the records
//!   it read once; `commit` checks the CPU's records it records the same way, where a state the
//!   records may name may encode, and never where none may;
//! - the blit encoder's copy, reset and optimize act on the commands as they run;
//! - a record the kernel wrote runs the state its ID named as it wrote it, after the program
//!   released it and after another state took its ID; where the state was released before the
//!   buffer could be written, nothing, announced;
//! - a kernel reaches the buffer made last with the ID it encodes through, placed or not, and
//!   none once that one is gone, the buffer at an ID a reservation leaves next, and an ID's
//!   records where an earlier ID's run past their start;
//! - a command buffer's executed-commands handler sees each execution where its records are read:
//!   in `commit` for the CPU's, in the host step (or on the GPU alone) for the kernel's, with each
//!   command's state's ability to encode indirect commands and the buffers made resident for
//!   writing. It sees the same on a device that keeps no execution record, as a consumer's keeps
//!   none;
//! - a no-copy buffer released while a later batch's host step resolves against a table of the
//!   live buffers that lists it keeps its pages until the command buffer completes, announced.
//!
//! One test reports, per device, how long a host step holds its command buffer and how late
//! another queue's command buffers committed meanwhile are submitted (a `STALL` line). Each
//! test's device announces the host replay once ([`HOST_STEPS`]).

use super::degrade_notices::{
    assert_made_once, counts, forced, on_device_with, shared_in_system_memory,
};
use super::generated_executions::flat_template;
use super::host_memory::{gpu_fill, no_copy, Mapping, GUEST_PAGE};
use super::indirect_executions::RELEASED_STATES;
use super::resource_ids::DELAYED_DEALLOCATORS as DELAYED;
use super::{buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::bind::{ArgumentDescriptor, ReplayFaults, StatusReport};
use metal2vulkan::metal::{DataType, IndirectCommandType, ResourceUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, RuntimeFact, StageSource};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitCommandEncoder, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue,
    ComputeCommandEncoder, ComputePipelineState, Deallocator, Degradation, Device, DeviceOptions,
    Error, EventRef, ExecutedCommands, ExecutedCommandsHandler, ExecutedPipeline, Execution,
    IdPlacement, IndirectCommand, IndirectCommandBuffer, IndirectCommandBufferDescriptor,
    IndirectCommandBufferProblem, IndirectCommandCheckNeeds, Invalid, PipelineOptions, Problem,
    RecordsRead, ResourceId, ResourceIdKind, ResourceOptions, RuntimeGap, Size,
};
use std::ptr::NonNull;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

/// The kind every test here takes: an execution of commands a kernel may write is replayed on
/// the host, announced once per device however many steps it takes.
pub(super) const HOST_STEPS: [Degradation; 1] = [Degradation::HostReplayedIndirectCommands];

/// The commands the kernel writes, one a thread.
pub(super) const COMMANDS: u64 = 4;

/// The words of one command's window of `params` and of `out`.
pub(super) const WINDOW: usize = 64;

/// The threads each command dispatches.
pub(super) const THREADS: usize = 64;

/// Run `test` on a validated device, then check that the device announced the host replay once
/// ([`HOST_STEPS`]) and, once it is dropped, that the layer reported nothing.
fn on_device(test: impl FnOnce(&Device)) {
    on_device_forcing(&[], test);
}

/// [`on_device`] of a test whose own steps force `own` too, each announced once beside
/// [`HOST_STEPS`] and the `Shared` buffers' system memory.
fn on_device_forcing(own: &[Degradation], test: impl FnOnce(&Device)) {
    on_device_forcing_with(|_| {}, own, test);
}

/// [`on_device_forcing`] on a device of the options `adjust` changes.
fn on_device_forcing_with(
    adjust: impl FnOnce(&mut DeviceOptions),
    own: &[Degradation],
    test: impl FnOnce(&Device),
) {
    on_device_with(adjust, |device, messages| {
        test(device);
        let kinds: Vec<Degradation> = HOST_STEPS.iter().chain(own).copied().collect();
        let shared = forced(&[(
            Degradation::SharedInSystemMemory,
            shared_in_system_memory(device),
        )]);
        assert_made_once(messages, &kinds, &shared);
    });
}

/// A state of owned fixture `stem`'s kernel, made with `supportIndirectCommandBuffers` as
/// `indirect` says.
pub(super) fn state(device: &Device, stem: &str, indirect: bool) -> ComputePipelineState {
    let bitcode = llvm_assemble(&engine_dir().join(format!("{stem}.ll"))).unwrap();
    let library = device.new_library_with_air(&bitcode).unwrap();
    let function = library.new_function_with_name(stem).unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = indirect;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Set a state that may write indirect commands (`icb_reset_encode`'s, made without
/// `supportIndirectCommandBuffers`, so no record names it) in an encoder of `command_buffer` of its
/// own that runs nothing: a buffer the command buffer makes resident for writing, before or after,
/// may be written by the GPU (`IndirectCommandBuffer::gpu_written`), and what its records may hold
/// (`IndirectCommandBuffer::command_writes`) gains the state's resets alone, which neither form of
/// an execution the GPU alone runs rules out, and no state it may name.
pub(super) fn encoding_state_set(command_buffer: &CommandBuffer) {
    let state = state(command_buffer.device(), "icb_reset_encode", false);
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.end_encoding().unwrap();
}

/// What a test runs: the writing kernel's state, the template's, a buffer of `COMMANDS`
/// commands of both dispatch kinds with `bind_count` kernel buffers, the writer's argument
/// buffer (the buffer's ID, then the template's), `params` and `out`.
pub(super) struct Setup {
    writer: ComputePipelineState,
    template: ComputePipelineState,
    icb: IndirectCommandBuffer,
    arguments: Buffer,
    params: Buffer,
    out: Buffer,
}

impl Setup {
    pub(super) fn new(device: &Device, bind_count: u32) -> Setup {
        Setup::with_template(
            device,
            bind_count,
            state(device, "icb_replay_template", true),
        )
    }

    /// [`Setup::new`] whose commands run `template`, a state made with
    /// `supportIndirectCommandBuffers` of a kernel of `icb_replay_template`'s buffers.
    pub(super) fn with_template(
        device: &Device,
        bind_count: u32,
        template: ComputePipelineState,
    ) -> Setup {
        let writer = state(device, "icb_replay_encode", false);
        let icb = commands_buffer(device, bind_count);
        let mut arguments = icb.gpu_resource_id().0.to_le_bytes().to_vec();
        arguments.extend(template.gpu_resource_id().0.to_le_bytes());
        let arguments = buffer_with(device, &arguments).unwrap();
        let params: Vec<u8> = (0..COMMANDS as usize * WINDOW)
            .map(|word| param(word / WINDOW, word % WINDOW))
            .flat_map(u32::to_le_bytes)
            .collect();
        let params = buffer_with(device, &params).unwrap();
        let out = buffer_with(device, &vec![0; COMMANDS as usize * WINDOW * 4]).unwrap();
        Setup {
            writer,
            template,
            icb,
            arguments,
            params,
            out,
        }
    }

    /// Encode the writing kernel on `encoder`: one thread a command.
    pub(super) fn write(&self, encoder: &mut ComputeCommandEncoder) {
        encoder.set_compute_pipeline_state(&self.writer).unwrap();
        encoder.set_buffer(Some(&self.arguments), 0, 0).unwrap();
        encoder.set_buffer(Some(&self.params), 0, 1).unwrap();
        encoder.set_buffer(Some(&self.out), 0, 2).unwrap();
        encoder
            .use_resource(self.icb.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        let commands = Size::new(COMMANDS, 1, 1);
        encoder.dispatch_threads(commands, commands).unwrap();
    }

    /// Encode the execution of every command on `encoder`, its windows resident.
    pub(super) fn execute(&self, encoder: &mut ComputeCommandEncoder) {
        encoder
            .use_resource(self.params.as_resource(), ResourceUsage::READ)
            .unwrap();
        encoder
            .use_resource(self.out.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        encoder
            .execute_commands_in_buffer(&self.icb, 0..COMMANDS)
            .unwrap();
    }

    /// Encode the execution of every command on `encoder`, its windows resident through no
    /// encoder: Metal's commands reach every live buffer of its own.
    pub(super) fn execute_naming_nothing(&self, encoder: &mut ComputeCommandEncoder) {
        encoder
            .execute_commands_in_buffer(&self.icb, 0..COMMANDS)
            .unwrap();
    }

    /// Encode the execution of the commands the range at `offset` of `range` names on `encoder`
    /// (`execute_commands_in_buffer_indirect`), its windows resident.
    fn execute_indirect(&self, encoder: &mut ComputeCommandEncoder, range: &Buffer, offset: u64) {
        encoder
            .use_resource(self.params.as_resource(), ResourceUsage::READ)
            .unwrap();
        encoder
            .use_resource(self.out.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        encoder
            .execute_commands_in_buffer_indirect(&self.icb, range, offset)
            .unwrap();
    }

    /// Write on the CPU the commands the kernel writes.
    fn write_on_the_cpu(&self) {
        for index in 0..COMMANDS {
            let command = self.icb.indirect_compute_command_at(index).unwrap();
            command.set_compute_pipeline_state(&self.template).unwrap();
            let offset = index * (WINDOW as u64) * 4;
            command
                .set_kernel_buffer(Some(&self.params), offset, 0)
                .unwrap();
            command
                .set_kernel_buffer(Some(&self.out), offset, 1)
                .unwrap();
            command.set_threadgroup_memory_length(32, 0).unwrap();
            let groups = Size::new(8, 1, 1);
            if index % 2 == 0 {
                command
                    .concurrent_dispatch_threadgroups(groups, groups)
                    .unwrap();
            } else {
                let threads = Size::new(THREADS as u64, 1, 1);
                command
                    .concurrent_dispatch_threads(threads, groups)
                    .unwrap();
            }
            command.set_barrier().unwrap();
        }
    }

    /// The setup with its template released, the writer's state in its place: the argument
    /// buffer still names the template's ID, which a state made after may take.
    fn releasing_template(self) -> Setup {
        Setup {
            template: self.writer.clone(),
            ..self
        }
    }

    /// `out`, which the commands write.
    pub(super) fn out(&self) -> &Buffer {
        &self.out
    }

    /// `out`, read back.
    pub(super) fn output(&self) -> Vec<u32> {
        words(&read_buffer(&self.out, 0, self.out.length()).unwrap())
    }
}

/// A buffer of `COMMANDS` commands of both dispatch kinds with `bind_count` kernel buffers and
/// one threadgroup length.
fn commands_buffer(device: &Device, bind_count: u32) -> IndirectCommandBuffer {
    commands_buffer_of(device, bind_count, COMMANDS)
}

/// [`commands_buffer`] of `count` commands.
fn commands_buffer_of(device: &Device, bind_count: u32, count: u64) -> IndirectCommandBuffer {
    try_commands_buffer_of(device, bind_count, count).unwrap()
}

/// [`commands_buffer_of`], or the device's refusal.
fn try_commands_buffer_of(
    device: &Device,
    bind_count: u32,
    count: u64,
) -> Result<IndirectCommandBuffer, Error> {
    let mut descriptor = IndirectCommandBufferDescriptor::new();
    descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH
        .union(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);
    descriptor.max_kernel_buffer_bind_count = bind_count;
    descriptor.max_kernel_threadgroup_memory_bind_count = 1;
    device.new_indirect_command_buffer(&descriptor, count, ResourceOptions::shared())
}

/// Word `word` of command `command`'s window of `params`.
pub(super) fn param(command: usize, word: usize) -> u32 {
    match word {
        0 => command as u32 + 2,
        1 => 100 * command as u32 + 7,
        _ => 0,
    }
}

/// What Metal computes: each command's thread `t` writes `params[0] * (t + 1) + params[1]` of
/// its window to word `t` of its window of `out`.
pub(super) fn expected() -> Vec<u32> {
    (0..COMMANDS as usize)
        .flat_map(|command| {
            (0..THREADS).map(move |t| param(command, 0) * (t as u32 + 1) + param(command, 1))
        })
        .collect()
}

/// [`expected`] where only the commands of `ran` run: zero in the other commands' windows.
fn expected_in(ran: std::ops::Range<usize>) -> Vec<u32> {
    expected()
        .into_iter()
        .enumerate()
        .map(|(word, value)| {
            if ran.contains(&(word / WINDOW)) {
                value
            } else {
                0
            }
        })
        .collect()
}

/// A buffer of 16 bytes: zero, then the execution range `location`, `length` at 8.
fn range_buffer(device: &Device, location: u32, length: u32) -> Buffer {
    let mut bytes = vec![0u8; 8];
    bytes.extend(location.to_le_bytes());
    bytes.extend(length.to_le_bytes());
    buffer_with(device, &bytes).unwrap()
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
        .collect()
}

/// Commit `command_buffer`, wait for it, and require it completed with a clean report.
pub(super) fn complete(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    wait(command_buffer);
}

/// Wait for committed `command_buffer`, and require it completed with a clean report.
pub(super) fn wait(command_buffer: &CommandBuffer) {
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(
        command_buffer.execution_report(),
        Some(StatusReport::default())
    );
}

/// The executions of the commands: the last `COMMANDS` of `command_buffer`'s.
fn command_runs(command_buffer: &CommandBuffer) -> Vec<Execution> {
    let executions = command_buffer.executions();
    let first = executions.len() - COMMANDS as usize;
    executions[first..].to_vec()
}

/// Whether `execution` reaches some buffer slot at its device address.
fn addressed(execution: &Execution) -> bool {
    match execution.pipeline() {
        ExecutedPipeline::Compute(pipeline) => pipeline
            .dependencies
            .iter()
            .any(|dependency| matches!(dependency.fact, RuntimeFact::DeviceAddresses { .. })),
        ExecutedPipeline::Render(_) => false,
    }
}

/// A device that asks for the host replay at every occurrence ([`HOST_STEPS`] in
/// `DeviceOptions::announce_each`) announces it at each execution, the host step's on the steps'
/// thread inside the step, before the batch is submitted: the debug assertion that
/// every-occurrence kinds are announced only in the program's call or a host step lets it pass
/// (tripped, it would end the steps' thread and the command buffer would never complete).
#[test]
fn a_host_step_announces_an_every_occurrence_kind_inside_the_step() {
    let each = |options: &mut DeviceOptions| options.announce_each = HOST_STEPS.into();
    on_device_with(each, |device, messages| {
        let queue = device.new_command_queue().unwrap();
        for run in 0..2 {
            let setup = Setup::new(device, 2);
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            setup.write(&mut encoder);
            setup.execute(&mut encoder);
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            assert_eq!(setup.output(), expected(), "run {run}");
        }
        let notices = counts(&messages.take_notices());
        let replays = notices.get(&HOST_STEPS[0]).copied().unwrap_or(0);
        assert!(replays >= 2, "announced at each execution: {notices:?}");
    });
}

/// The commands run where the kernel that wrote them was encoded before them: in its encoder,
/// in a later encoder of its command buffer, and in a later command buffer of its queue. The
/// writing kernel's state can encode indirect commands, as the buffer in its argument buffer
/// lets it; the commands' state, which takes none, cannot.
#[test]
fn commands_a_kernel_writes_run_after_it() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        // One encoder.
        let setup = Setup::new(device, 2);
        assert!(setup.writer.can_encode_indirect_commands());
        assert!(!setup.template.can_encode_indirect_commands());
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.write(&mut encoder);
        setup.execute(&mut encoder);
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(setup.output(), expected(), "in the encoder that wrote them");
        // Two encoders of one command buffer.
        let setup = Setup::new(device, 2);
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.write(&mut encoder);
        encoder.end_encoding().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.execute(&mut encoder);
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(setup.output(), expected(), "in a later encoder");
        // Two command buffers.
        let setup = Setup::new(device, 2);
        let writing = queue.command_buffer().unwrap();
        let mut encoder = writing.compute_command_encoder().unwrap();
        setup.write(&mut encoder);
        encoder.end_encoding().unwrap();
        let executing = queue.command_buffer().unwrap();
        let mut encoder = executing.compute_command_encoder().unwrap();
        setup.execute(&mut encoder);
        encoder.end_encoding().unwrap();
        writing.commit().unwrap();
        complete(&executing);
        wait(&writing);
        assert_eq!(setup.output(), expected(), "in a later command buffer");
    });
}

/// A checked execution (`execute_commands_in_buffer_checked`) of commands a kernel writes: its host
/// step calls the check once for each command it records, in order, with the command as
/// `IndirectCommandBuffer::commands` reads it (each a dispatch), before it records any: the
/// commands' windows of `out` are all still zero at every call. An execution of the CPU's records
/// in a buffer no encoder made resident for writing, which `commit` records, calls it there, on
/// the committing thread before `commit` returns, with each command, where the CPU set a state
/// that may encode on one of them (since replaced by the template); where no state the records
/// may name may, never. All run as their plain forms run. The commands the check reads answer
/// the runtime's unique identifiers of their states, which the device announces as the getter
/// does.
#[test]
fn a_checked_execution_checks_each_written_command_before_its_step_records_it() {
    on_device_forcing(&[Degradation::UnmeasuredUniqueIdentifier], |device| {
        let queue = device.new_command_queue().unwrap();
        let setup = Setup::new(device, 2);
        let seen: Arc<Mutex<Vec<(u64, bool, bool)>>> = Arc::default();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.write(&mut encoder);
        encoder
            .use_resource(setup.params.as_resource(), ResourceUsage::READ)
            .unwrap();
        encoder
            .use_resource(setup.out.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        let (out, log) = (setup.out.clone(), Arc::clone(&seen));
        encoder
            .execute_commands_in_buffer_checked(
                &setup.icb,
                0..COMMANDS,
                Box::new(move |index, command| {
                    let untouched = words(&read_buffer(&out, 0, out.length()).unwrap())
                        .iter()
                        .all(|word| *word == 0);
                    let dispatches = matches!(
                        command,
                        IndirectCommand::Compute(command) if command.dispatch.is_some()
                    );
                    log.lock().unwrap().push((index, dispatches, untouched));
                }),
                IndirectCommandCheckNeeds::ALL,
            )
            .unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(setup.output(), expected(), "the checked commands run");
        let every: Vec<(u64, bool, bool)> =
            (0..COMMANDS).map(|index| (index, true, true)).collect();
        assert_eq!(
            *seen.lock().unwrap(),
            every,
            "each command, before any runs"
        );
        // The CPU's records, recorded at the commit: checked there where a state they may name
        // may encode.
        let encoding = state(device, "icb_replay_encode", true);
        for encodes in [false, true] {
            let setup = Setup::new(device, 2);
            if encodes {
                let command = setup.icb.indirect_compute_command_at(0).unwrap();
                command.set_compute_pipeline_state(&encoding).unwrap();
            }
            setup.write_on_the_cpu();
            let seen: Arc<Mutex<Vec<(u64, std::thread::ThreadId)>>> = Arc::default();
            let log = Arc::clone(&seen);
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder
                .use_resource(setup.params.as_resource(), ResourceUsage::READ)
                .unwrap();
            encoder
                .use_resource(setup.out.as_resource(), ResourceUsage::WRITE)
                .unwrap();
            encoder
                .execute_commands_in_buffer_checked(
                    &setup.icb,
                    0..COMMANDS,
                    Box::new(move |index, _| {
                        log.lock()
                            .unwrap()
                            .push((index, std::thread::current().id()));
                    }),
                    IndirectCommandCheckNeeds::ALL,
                )
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            let at_commit = seen.lock().unwrap().clone();
            wait(&command_buffer);
            assert!(!setup.icb.gpu_written());
            assert_eq!(
                setup.output(),
                expected(),
                "the CPU's records run, encodes {encodes}"
            );
            let committing = std::thread::current().id();
            let every: Vec<(u64, std::thread::ThreadId)> = match encodes {
                true => (0..COMMANDS).map(|index| (index, committing)).collect(),
                false => Vec::new(),
            };
            assert_eq!(at_commit, every, "checked in `commit`, encodes {encodes}");
            assert_eq!(
                *seen.lock().unwrap(),
                every,
                "and only there, encodes {encodes}"
            );
        }
    });
}

/// A checked execution's host step reads each command's record once
/// (`WorkCounts::indirect_records_read`): its check reads the commands from the records the step
/// read, which are the commands `IndirectCommandBuffer::commands` reads once they ran.
#[test]
fn a_checked_host_step_reads_each_record_once() {
    on_device_forcing(&[Degradation::UnmeasuredUniqueIdentifier], |device| {
        let queue = device.new_command_queue().unwrap();
        let setup = Setup::new(device, 2);
        let seen: Arc<Mutex<Vec<String>>> = Arc::default();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.write(&mut encoder);
        encoder
            .use_resource(setup.params.as_resource(), ResourceUsage::READ)
            .unwrap();
        encoder
            .use_resource(setup.out.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        let log = Arc::clone(&seen);
        encoder
            .execute_commands_in_buffer_checked(
                &setup.icb,
                0..COMMANDS,
                Box::new(move |_, command| log.lock().unwrap().push(format!("{command:?}"))),
                IndirectCommandCheckNeeds::ALL,
            )
            .unwrap();
        encoder.end_encoding().unwrap();
        let before = device.work_counts().indirect_records_read;
        complete(&command_buffer);
        let read = device.work_counts().indirect_records_read - before;
        assert_eq!(read, COMMANDS, "each record read once");
        assert_eq!(setup.output(), expected(), "the checked commands run");
        let getter: Vec<String> = setup
            .icb
            .commands(0..COMMANDS)
            .unwrap()
            .iter()
            .map(|command| format!("{command:?}"))
            .collect();
        assert_eq!(*seen.lock().unwrap(), getter, "the getter's commands");
    });
}

/// The kernel writes the buffer whose ID its argument buffer holds now, as Metal's kernel does
/// (`kernel_cpu_reused`): a buffer made and gone leaves ID 1 free, the setup's buffer takes it
/// again, and the commands the kernel writes through ID 1, which the CPU wrote into its argument
/// buffer, run from it, while a buffer made before it, ID 2, resident for writing too, keeps fresh
/// commands.
#[test]
fn a_kernel_writes_the_buffer_that_holds_the_id_now() {
    on_device(|device| {
        let gone = commands_buffer(device, 2);
        let other = commands_buffer(device, 2);
        assert_eq!(
            [&gone, &other].map(IndirectCommandBuffer::gpu_resource_id),
            [ResourceId(1), ResourceId(2)]
        );
        drop(gone);
        let setup = Setup::new(device, 2);
        assert_eq!(setup.icb.gpu_resource_id(), ResourceId(1));
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder
            .use_resource(other.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        setup.write(&mut encoder);
        setup.execute(&mut encoder);
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(setup.output(), expected());
        let fresh = commands_buffer(device, 2);
        assert_eq!(
            format!("{:?}", other.commands(0..COMMANDS).unwrap()),
            format!("{:?}", fresh.commands(0..COMMANDS).unwrap())
        );
    });
}

/// A record a kernel wrote runs the state its ID named as the kernel wrote it, released since or
/// not, as Metal's runs it (measured on an M3 Max, `released_programs.rs`): where the program
/// releases the template after the kernel wrote the commands, they run the template's program,
/// and so they do after a state made after (the flat template of `generated_executions`) took its
/// ID, unannounced. Where the program released the template before any command buffer that may
/// write the buffer was encoded, the buffer keeps no program of it: Metal runs it, the commands
/// run nothing, announced (`Degradation::ReleasedCommandState`).
#[test]
fn commands_a_kernel_wrote_run_the_state_their_id_named_then() {
    for taken in [false, true] {
        on_device(|device| {
            let queue = device.new_command_queue().unwrap();
            let setup = Setup::new(device, 2);
            let id = setup.template.gpu_resource_id();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            setup.write(&mut encoder);
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            let setup = setup.releasing_template();
            let taker = taken.then(|| flat_template(device));
            if let Some(taker) = &taker {
                assert_eq!(
                    taker.gpu_resource_id(),
                    id,
                    "the flat template takes the ID"
                );
            }
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            setup.execute(&mut encoder);
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            assert_eq!(setup.output(), expected(), "taken {taken}");
        });
    }
    on_device_forcing(&RELEASED_STATES, |device| {
        let queue = device.new_command_queue().unwrap();
        let setup = Setup::new(device, 2).releasing_template();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.write(&mut encoder);
        setup.execute(&mut encoder);
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(
            setup.output(),
            expected_in(0..0),
            "the commands ran nothing"
        );
    });
}

/// A kernel encoding through an ID reaches the buffer made last with it, placed or not, as
/// Metal's does (`r_*`): a buffer placed at the setup's buffer's ID takes it,
/// so the kernel writes the placed buffer's commands and the setup's buffer keeps fresh ones,
/// which run nothing. Once the placed buffer is gone the ID reaches no buffer, though the setup's
/// still lives: the kernel's writes miss, which the execution report counts, and its commands
/// stay fresh.
#[test]
fn a_kernel_reaches_the_buffer_made_last_with_its_id() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let setup = Setup::new(device, 2);
        let id = setup.icb.gpu_resource_id();
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH
            .union(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);
        descriptor.max_kernel_buffer_bind_count = 2;
        descriptor.max_kernel_threadgroup_memory_bind_count = 1;
        descriptor.placement = IdPlacement::At(id);
        let placed = device
            .new_indirect_command_buffer(&descriptor, COMMANDS, ResourceOptions::shared())
            .unwrap();
        assert_eq!(placed.gpu_resource_id(), id);
        // Write through the ID with both resident for writing, then run each alone.
        let run = |write: bool, executed: &IndirectCommandBuffer| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&setup.out, 0..setup.out.length(), 0)
                .unwrap();
            blit.end_encoding().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            if write {
                encoder
                    .use_resource(placed.as_resource(), ResourceUsage::WRITE)
                    .unwrap();
                setup.write(&mut encoder);
            }
            encoder
                .use_resource(setup.params.as_resource(), ResourceUsage::READ)
                .unwrap();
            encoder
                .use_resource(setup.out.as_resource(), ResourceUsage::WRITE)
                .unwrap();
            encoder
                .execute_commands_in_buffer(executed, 0..COMMANDS)
                .unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            setup.output()
        };
        let none = vec![0; COMMANDS as usize * WINDOW];
        assert_eq!(run(true, &setup.icb), none, "the earlier buffer, written");
        assert_eq!(run(false, &placed), expected(), "the placed buffer");
        // The placed buffer gone, the ID reaches none: the kernel's writes miss, and the
        // execution report counts them (Metal leaves such an access undefined).
        drop(placed);
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.write(&mut encoder);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let report = command_buffer.execution_report().unwrap();
        assert!(report.resident_miss, "the writes miss: {report:?}");
        let fresh = commands_buffer(device, 2);
        assert_eq!(
            format!("{:?}", setup.icb.commands(0..COMMANDS).unwrap()),
            format!("{:?}", fresh.commands(0..COMMANDS).unwrap()),
            "the earlier buffer, written after the placed one is gone"
        );
    });
}

/// A kernel encoding through the ID a reservation leaves next reaches its buffer's records, and
/// the commands it writes run, as Metal's do (on an M3 Max, each buffer made past a reservation
/// of 2^23 + 5, 2^30 + 5 or 2^31 - 2 ran), up to the last ID, 2^31 - 1, whose records start 2^62
/// bytes below 2^64.
#[test]
fn a_kernel_reaches_the_records_of_a_buffer_past_a_reservation() {
    for reserved in [(1 << 23) + 5, (1 << 26) - 2, (1 << 26) + 5, (1 << 31) - 2] {
        on_device(|device| {
            device
                .reserve_resource_ids(
                    ResourceIdKind::IndirectCommandBuffer,
                    &[ResourceId(reserved)],
                )
                .unwrap();
            let setup = Setup::new(device, 2);
            assert_eq!(setup.icb.gpu_resource_id(), ResourceId(reserved + 1));
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            setup.write(&mut encoder);
            setup.execute(&mut encoder);
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            assert_eq!(setup.output(), expected(), "{reserved}");
        });
    }
}

/// A kernel reaches each ID's records at their start, however far an earlier ID's run, as Metal
/// makes and runs buffers of any length past a reservation (on an M3 Max, 2^22 commands at
/// 2^26 + 6): the records of ID r + 1, 955,300 commands (past 2^31 bytes), run past ID r + 2's
/// start. Through r + 2 the kernel writes r + 2's commands, which run, and r + 1's stay fresh;
/// with r + 2's buffer not resident, its writes miss, which the execution report counts, and
/// reach none of r + 1's records where r + 2's would start. A device whose buffers stop at 2 GiB
/// refuses the long buffer (`RuntimeGap::IndirectCommandBufferPastMaxBufferLength`).
#[test]
fn a_kernel_reaches_an_id_s_records_where_an_earlier_id_s_run_past_them() {
    on_device(|device| {
        let reserved = (1 << 26) + 5;
        device
            .reserve_resource_ids(
                ResourceIdKind::IndirectCommandBuffer,
                &[ResourceId(reserved)],
            )
            .unwrap();
        let count = 955_300;
        let queue = device.new_command_queue().unwrap();
        let long = match try_commands_buffer_of(device, 2, count) {
            Ok(long) => long,
            // A device whose buffers stop at 2 GiB (lavapipe) holds no records past 2^31 bytes,
            // which an ID's run past the next ID's start needs: refused, typed. The kernel
            // still writes the next ID's commands, which run.
            Err(Error::NotImplemented(RuntimeGap::IndirectCommandBufferPastMaxBufferLength {
                max_buffer_length,
                ..
            })) if max_buffer_length <= 1 << 31 => {
                let setup = Setup::new(device, 2);
                assert_eq!(setup.icb.gpu_resource_id(), ResourceId(reserved + 1));
                run_written(&queue, &setup);
                return;
            }
            Err(error) => panic!("{error:?}"),
        };
        assert_eq!(long.gpu_resource_id(), ResourceId(reserved + 1));
        let setup = Setup::new(device, 2);
        assert_eq!(setup.icb.gpu_resource_id(), ResourceId(reserved + 2));
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder
            .use_resource(long.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        setup.write(&mut encoder);
        setup.execute(&mut encoder);
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(setup.output(), expected(), "through r + 2");
        // The records about r + 2's start, 2^31 bytes into r + 1's, and r + 1's first: fresh.
        let fresh = format!(
            "{:?}",
            commands_buffer(device, 2).commands(0..1).unwrap()[0]
        );
        let first = (1 << 31) / 2248 - 2;
        let around = (first..first + 6).chain(0..COMMANDS);
        let fresh_all = |what: &str| {
            for index in around.clone() {
                let command = long.commands(index..index + 1).unwrap();
                assert_eq!(
                    format!("{:?}", command[0]),
                    fresh,
                    "{what}: command {index}"
                );
            }
        };
        fresh_all("written through r + 2");
        // r + 2's buffer not resident: the writes miss.
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder
            .use_resource(long.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        encoder.set_compute_pipeline_state(&setup.writer).unwrap();
        encoder.set_buffer(Some(&setup.arguments), 0, 0).unwrap();
        encoder.set_buffer(Some(&setup.params), 0, 1).unwrap();
        encoder.set_buffer(Some(&setup.out), 0, 2).unwrap();
        let commands = Size::new(COMMANDS, 1, 1);
        encoder.dispatch_threads(commands, commands).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let report = command_buffer.execution_report().unwrap();
        assert!(report.resident_miss, "the writes miss: {report:?}");
        fresh_all("with r + 2's records not resident");
    });
}

/// The commands run on another queue after an event the writer signals, the executing command
/// buffer committed first: its batch waits for the event, then for the writer's work to
/// complete. So does an execution committed before any encoder made the buffer resident for
/// writing (a step its commit did not foresee, which that encoder's `use_resource` settles), which
/// runs what the kernel wrote then, naming the states made with `supportIndirectCommandBuffers`
/// that lived at its commit.
#[test]
fn commands_a_kernel_writes_run_on_another_queue_after_its_event() {
    on_device(|device| {
        for foreseen in [true, false] {
            let setup = Setup::new(device, 2);
            let event = device.new_event().unwrap();
            let (writing, executing) = (
                device.new_command_queue().unwrap(),
                device.new_command_queue().unwrap(),
            );
            let writer = writing.command_buffer().unwrap();
            let encode_writer = |writer: &CommandBuffer| {
                let mut encoder = writer.compute_command_encoder().unwrap();
                setup.write(&mut encoder);
                encoder.end_encoding().unwrap();
                writer.encode_signal_event((&event).into(), 1).unwrap();
            };
            if foreseen {
                encode_writer(&writer);
            }
            let execution = executing.command_buffer().unwrap();
            execution.encode_wait_for_event((&event).into(), 1).unwrap();
            let mut encoder = execution.compute_command_encoder().unwrap();
            setup.execute(&mut encoder);
            encoder.end_encoding().unwrap();
            execution.commit().unwrap();
            if !foreseen {
                encode_writer(&writer);
            }
            complete(&writer);
            wait(&execution);
            assert_eq!(setup.output(), expected(), "foreseen: {foreseen}");
            // Recorded by the host step, in its form, whether or not the commit foresaw it.
            assert!(
                command_runs(&execution).iter().all(addressed),
                "the large-buffer form, foreseen: {foreseen}"
            );
        }
    });
}

/// The commands a kernel writes run in the large-buffer form (docs/API.md section 10): the
/// template's `out` slot, whose window is the whole buffer, at its device address, where the same
/// commands written by the CPU reach it through a descriptor; inside the windows both write what
/// Metal computes. A second execution of the written commands runs the same compiled variant and
/// `VkPipeline`s, made by the first.
#[test]
fn written_commands_run_in_the_large_buffer_form_and_compile_once() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let on_the_cpu = Setup::new(device, 2);
        on_the_cpu.write_on_the_cpu();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        on_the_cpu.execute(&mut encoder);
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(on_the_cpu.output(), expected(), "written on the CPU");
        assert!(command_runs(&command_buffer)
            .iter()
            .all(|run| !addressed(run)));

        let written = Setup::new(device, 2);
        let mut runs = Vec::new();
        for round in 0..2 {
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            written.write(&mut encoder);
            encoder.end_encoding().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            written.execute(&mut encoder);
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            assert_eq!(
                written.output(),
                expected(),
                "written by the kernel, round {round}"
            );
            runs.push(command_runs(&command_buffer));
        }
        assert!(runs[0].iter().all(addressed), "the large-buffer form");
        for (first, second) in runs[0].iter().zip(&runs[1]) {
            let (ExecutedPipeline::Compute(a), ExecutedPipeline::Compute(b)) =
                (first.pipeline(), second.pipeline())
            else {
                panic!("a command ran no compute pipeline");
            };
            assert!(std::sync::Arc::ptr_eq(a, b), "the second round's variant");
            assert_eq!(first.pipeline_creation(), second.pipeline_creation());
        }
    });
}

/// A write the buffer's descriptor does not allow (a kernel buffer past its bind count) sets the
/// header's sticky encoding fault: no command runs, and the execution report says why.
#[test]
fn an_encoding_fault_runs_no_command_and_is_reported() {
    on_device(|device| {
        let setup = Setup::new(device, 1);
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.write(&mut encoder);
        encoder.end_encoding().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.execute(&mut encoder);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        let report = command_buffer.execution_report().unwrap();
        assert!(report.replay_fault);
        assert_eq!(report.replay_faults, ReplayFaults::ENCODING);
        assert!(setup.output().iter().all(|&word| word == 0));
    });
}

/// An execution reads its range from a buffer as it runs (`execute_commands_in_buffer_indirect`)
/// and runs what Metal runs: the commands the CPU wrote from the location, clamped at the size;
/// nothing from a location at the size or a length of 0; and, for a location past the size or a
/// range past `2^32 - 1`, nothing, the execution report stating `RANGE`. A range a blit of the same
/// command buffer wrote before the execution names the commands a kernel wrote. An offset that is
/// not a multiple of 4, and a range past its buffer, are refused at the call.
#[test]
fn an_execution_reads_its_range_as_it_runs() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        for (location, length, ran) in [
            (1u32, 2u32, 1..3),
            (2, 5, 2..4),
            (4, 1, 4..4),
            (1, 0, 1..1),
            (0, u32::MAX, 0..4),
        ] {
            let setup = Setup::new(device, 2);
            setup.write_on_the_cpu();
            let range = range_buffer(device, location, length);
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            setup.execute_indirect(&mut encoder, &range, 8);
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            assert_eq!(setup.output(), expected_in(ran), "{location}, {length}");
        }
        for (location, length) in [(5u32, 2u32), (1, u32::MAX), (u32::MAX, u32::MAX)] {
            let setup = Setup::new(device, 2);
            setup.write_on_the_cpu();
            let range = range_buffer(device, location, length);
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            setup.execute_indirect(&mut encoder, &range, 8);
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            let report = command_buffer.execution_report().unwrap();
            assert!(report.replay_fault, "{location}, {length}");
            assert_eq!(report.replay_faults, ReplayFaults::RANGE);
            assert_eq!(setup.output(), expected_in(0..0), "{location}, {length}");
        }
        // The range a blit wrote, of the commands a kernel wrote.
        let setup = Setup::new(device, 2);
        let source = range_buffer(device, 1, 2);
        let range = range_buffer(device, 0, 0);
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.write(&mut encoder);
        encoder.end_encoding().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&source, 8, &range, 8, 8)
            .unwrap();
        blit.end_encoding().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.execute_indirect(&mut encoder, &range, 8);
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(setup.output(), expected_in(1..3), "a range a blit wrote");
        // Refusals, recording nothing.
        let setup = Setup::new(device, 2);
        let range = range_buffer(device, 1, 2);
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        // Plain Metal takes both, so neither is misuse: the runtime's own gaps.
        for offset in [2, 6] {
            assert_eq!(
                encoder
                    .execute_commands_in_buffer_indirect(&setup.icb, &range, offset)
                    .err(),
                Some(Error::NotImplemented(RuntimeGap::ExecutionRangeOffFour {
                    offset
                })),
                "offset {offset}"
            );
        }
        for (offset, length) in [(12, 16), (u64::MAX - 3, 16)] {
            assert_eq!(
                encoder
                    .execute_commands_in_buffer_indirect(&setup.icb, &range, offset)
                    .err(),
                Some(Error::NotImplemented(
                    RuntimeGap::ExecutionRangePastBuffer { offset, length }
                )),
                "offset {offset}"
            );
        }
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert!(command_buffer.executions().is_empty());
    });
}

/// The blit encoder's copy, reset and optimize act on the commands as Metal's do, as the command
/// buffer runs: commands 1 and 2 of CPU-written commands copied to the
/// same places of a fresh buffer run there, and its other commands nothing; all four copied, then
/// commands 1 and 2 reset by a blit, run 0 and 3; an optimize changes nothing. The copied
/// buffer's commands read back with the template's state, by its ID. A range past either
/// buffer is refused at the call. The copied buffer's executions are host steps. The commands
/// read back answer the template's identifier, made for indirect command buffers, which is not
/// measured to be Metal's number (`unmeasured-unique-identifier`).
#[test]
fn blits_copy_reset_and_optimize_commands_as_they_run() {
    on_device_forcing(&[Degradation::UnmeasuredUniqueIdentifier], |device| {
        let queue = device.new_command_queue().unwrap();
        let setup = Setup::new(device, 2);
        setup.write_on_the_cpu();
        let run = |destination: &IndirectCommandBuffer, blits: &dyn Fn(&mut BlitCommandEncoder)| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blits(&mut blit);
            blit.fill_buffer(&setup.out, 0..setup.out.length(), 0)
                .unwrap();
            blit.end_encoding().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder
                .use_resource(setup.params.as_resource(), ResourceUsage::READ)
                .unwrap();
            encoder
                .use_resource(setup.out.as_resource(), ResourceUsage::WRITE)
                .unwrap();
            encoder
                .execute_commands_in_buffer(destination, 0..COMMANDS)
                .unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            setup.output()
        };
        let copied = commands_buffer(device, 2);
        let out = run(&copied, &|blit| {
            blit.copy_indirect_command_buffer(&setup.icb, 1..3, &copied, 1)
                .unwrap();
        });
        assert_eq!(out, expected_in(1..3), "commands 1 and 2 copied");
        let commands = copied.commands(0..COMMANDS).unwrap();
        for (index, command) in commands.iter().enumerate() {
            let IndirectCommand::Compute(command) = command else {
                panic!("command {index} is not a compute command");
            };
            let named = command
                .pipeline
                .as_ref()
                .map(|state| state.gpu_resource_id());
            let expected = (1..3)
                .contains(&index)
                .then(|| setup.template.gpu_resource_id());
            assert_eq!(named, expected, "command {index}'s state");
        }
        let reset = commands_buffer(device, 2);
        let out = run(&reset, &|blit| {
            blit.copy_indirect_command_buffer(&setup.icb, 0..COMMANDS, &reset, 0)
                .unwrap();
            blit.reset_commands_in_buffer(&reset, 1..3).unwrap();
        });
        assert_eq!(
            out,
            expected()
                .into_iter()
                .enumerate()
                .map(|(word, value)| if (1..3).contains(&(word / WINDOW)) {
                    0
                } else {
                    value
                })
                .collect::<Vec<_>>(),
            "all copied, 1 and 2 reset"
        );
        let out = run(&setup.icb, &|blit| {
            blit.optimize_indirect_command_buffer(&setup.icb, 0..COMMANDS)
                .unwrap();
        });
        assert_eq!(out, expected(), "optimized");
        // Refusals, recording nothing.
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        let past = |location: u64, length: u64| {
            Some(Error::Invalid(Invalid::new(
                "copy_indirect_command_buffer",
                Problem::IndirectCommandBuffer(IndirectCommandBufferProblem::Range {
                    location,
                    length,
                    size: COMMANDS,
                }),
            )))
        };
        assert_eq!(
            blit.copy_indirect_command_buffer(&setup.icb, 2..5, &copied, 0)
                .err(),
            past(2, 3)
        );
        assert_eq!(
            blit.copy_indirect_command_buffer(&setup.icb, 0..2, &copied, 3)
                .err(),
            past(3, 2)
        );
        // A buffer of no command types is reset, optimized and copied to another as nothing, as
        // plain Metal's.
        let none = || {
            device
                .new_indirect_command_buffer(
                    &IndirectCommandBufferDescriptor::new(),
                    COMMANDS,
                    ResourceOptions::shared(),
                )
                .unwrap()
        };
        let (none, other) = (none(), none());
        blit.reset_commands_in_buffer(&none, 0..2).unwrap();
        blit.optimize_indirect_command_buffer(&none, 0..2).unwrap();
        blit.copy_indirect_command_buffer(&none, 0..2, &other, 0)
            .unwrap();
        blit.end_encoding().unwrap();
        complete(&command_buffer);
        // A buffer of no command types beside one of command types: the copy is taken, in either
        // direction, and moves nothing, as plain Metal's does. The destination of command types
        // holds every command, so a copy of the other buffer's records over them would leave it
        // running nothing; it runs them all.
        let kept = commands_buffer(device, 2);
        let out = run(&kept, &|blit| {
            blit.copy_indirect_command_buffer(&setup.icb, 0..COMMANDS, &kept, 0)
                .unwrap();
            blit.copy_indirect_command_buffer(&none, 0..COMMANDS, &kept, 0)
                .unwrap();
            blit.copy_indirect_command_buffer(&kept, 0..COMMANDS, &none, 0)
                .unwrap();
        });
        assert_eq!(
            out,
            expected(),
            "a copy beside no command types moved nothing"
        );
        // The range is still checked on both sides, each buffer's own size.
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        assert_eq!(
            blit.copy_indirect_command_buffer(&none, 0..COMMANDS + 1, &kept, 0)
                .err(),
            Some(Error::Invalid(Invalid::new(
                "copy_indirect_command_buffer",
                Problem::IndirectCommandBuffer(IndirectCommandBufferProblem::Range {
                    location: 0,
                    length: COMMANDS + 1,
                    size: COMMANDS,
                }),
            )))
        );
        blit.end_encoding().unwrap();
        complete(&command_buffer);
    });
}

/// An indirect command buffer of no command types encodes its own resource ID into an argument
/// buffer, as a buffer of command types does: its command types decide nothing about the word, and
/// the ID is its place among the device's indirect command buffers, so two buffers encoded in one
/// process take two IDs whatever their command types (plain Metal on an M3 Max, macOS 27.0.1:
/// `0100000000000000` then `0200000000000000` in the order they were made, either order, and the
/// same for two `Draw` buffers; `probes/icb_argument_encode_and_copy_destinations.m`).
#[test]
fn a_buffer_of_no_command_types_encodes_its_own_id() {
    // Not `on_device`: the test encodes and executes nothing, so no host step is announced.
    on_device_with(
        |_| {},
        |device, messages| {
            let made = |types: IndirectCommandType| {
                let mut descriptor = IndirectCommandBufferDescriptor::new();
                descriptor.command_types = types;
                device
                    .new_indirect_command_buffer(&descriptor, COMMANDS, ResourceOptions::shared())
                    .unwrap()
            };
            let none = made(IndirectCommandType::empty());
            let draw = made(IndirectCommandType::DRAW);
            // The ID is the place, not the descriptor: the buffer of no command types, made first,
            // holds the lower ID, and neither is 0.
            assert_eq!(
                [none.gpu_resource_id(), draw.gpu_resource_id()],
                [ResourceId(1), ResourceId(2)]
            );
            let mut encoder = device
                .new_argument_encoder_with_arguments(&[
                    ArgumentDescriptor::new(DataType::IndirectCommandBuffer, 0),
                    ArgumentDescriptor::new(DataType::IndirectCommandBuffer, 1),
                ])
                .unwrap();
            let arguments = device.new_buffer(64, ResourceOptions::shared()).unwrap();
            encoder.set_argument_buffer(Some(&arguments), 0).unwrap();
            let word = |at: usize| {
                let contents = arguments.contents().unwrap();
                // SAFETY: the buffer is shared and mapped, 64 bytes long, and nothing writes it now.
                let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr().add(at), 8) };
                u64::from_le_bytes(bytes.try_into().unwrap())
            };
            encoder.set_indirect_command_buffer(Some(&none), 0).unwrap();
            encoder.set_indirect_command_buffer(Some(&draw), 1).unwrap();
            assert_eq!([word(0), word(8)], [1, 2]);
            // The range form writes both, and `None` writes 0 at either index.
            encoder
                .set_indirect_command_buffers(&[Some(&draw), Some(&none)], 0)
                .unwrap();
            assert_eq!([word(0), word(8)], [2, 1]);
            encoder.set_indirect_command_buffer(None, 0).unwrap();
            assert_eq!([word(0), word(8)], [0, 1]);
            let shared = forced(&[(
                Degradation::SharedInSystemMemory,
                shared_in_system_memory(device),
            )]);
            assert_made_once(messages, &[], &shared);
        },
    );
}

/// How long a host step holds its command buffer, and how late another queue's command buffers
/// committed meanwhile are submitted: printed as a `STALL` line for the device.
#[test]
fn a_host_step_holds_back_only_its_own_command_buffer() {
    on_device(|device| {
        let setup = Setup::new(device, 2);
        let (stepping, other) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        let scratch = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        // Warm: the step's variants and pipelines are made once.
        let warm = Setup::new(device, 2);
        run_written(&stepping, &warm);
        let command_buffer = stepping.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.write(&mut encoder);
        encoder.end_encoding().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.execute(&mut encoder);
        encoder.end_encoding().unwrap();
        let started = Instant::now();
        command_buffer.commit().unwrap();
        let mut late = 0f64;
        let mut meanwhile = 0;
        while command_buffer.status() != CommandBufferStatus::Completed && started.elapsed() < WAIT
        {
            let small = other.command_buffer().unwrap();
            let mut blit = small.blit_command_encoder().unwrap();
            blit.fill_buffer(&scratch, 0..4096, meanwhile as u8)
                .unwrap();
            blit.end_encoding().unwrap();
            let committed = metal2vulkan_runtime::host_time();
            small.commit().unwrap();
            small.wait_until_scheduled().unwrap();
            late = late.max(small.kernel_start_time() - committed);
            meanwhile += 1;
        }
        wait(&command_buffer);
        assert_eq!(setup.output(), expected());
        let held = command_buffer.kernel_end_time() - command_buffer.kernel_start_time();
        println!(
            "STALL {}: the host step's command buffer went {:.3} ms from its first submission \
             to its last; {meanwhile} command buffers of another queue committed meanwhile were \
             submitted at most {:.3} ms after their commit",
            device.name(),
            held * 1e3,
            late * 1e3
        );
    });
}

/// Write and execute `setup`'s commands in one command buffer of `queue`, and check the output.
fn run_written(queue: &CommandQueue, setup: &Setup) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    setup.write(&mut encoder);
    encoder.end_encoding().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    setup.execute(&mut encoder);
    encoder.end_encoding().unwrap();
    complete(&command_buffer);
    assert_eq!(setup.output(), expected());
}

/// A no-copy buffer listed after a command buffer's first submission, and released while a later
/// batch's host step resolves against the live buffers' table that lists it, keeps its pages
/// until that command buffer completes: the commands the step records may reach them. Its
/// deallocator then runs in the command buffer's release, where its wait for that command buffer
/// returns at once, and the device announces the wait past Metal's release once
/// (`host-deallocator-delayed`; Metal's runs at the release, probe `dealloc-probe4`). The step's
/// own hold of the table was its last: the deallocator ran on the thread that submits, whose wait
/// for the command buffer never returned. The device withholds the GPU-alone form, so the
/// execution takes the host step; the executed-commands handler, called in the step, holds it
/// while the program releases the buffer.
#[test]
fn a_buffer_released_during_a_later_host_step_waits_for_its_command_buffer() {
    let withheld = |options: &mut DeviceOptions| options.withhold_generated_commands = true;
    on_device_forcing_with(withheld, &DELAYED, |device| {
        let (queue, other) = (
            device.new_command_queue().unwrap(),
            device.new_command_queue().unwrap(),
        );
        let setup = Setup::new(device, 2);
        let writing = queue.command_buffer().unwrap();
        let mut encoder = writing.compute_command_encoder().unwrap();
        setup.write(&mut encoder);
        encoder.end_encoding().unwrap();
        complete(&writing);
        let scratch = device.new_buffer(256, ResourceOptions::shared()).unwrap();
        let executing = queue.command_buffer().unwrap();
        let memory = Mapping::anonymous(GUEST_PAGE);
        let pointer = memory.pointer();
        let (sender, receiver) = mpsc::channel();
        let deallocator: Deallocator = {
            let executing = executing.clone();
            Box::new(move |_: NonNull<u8>, _: u64| {
                let status = executing.status();
                let waited = (
                    executing.wait_until_completed(),
                    executing.wait_until_completed_for(WAIT),
                );
                drop(executing);
                drop(memory);
                sender.send((status, waited)).unwrap();
            })
        };
        // SAFETY: the deallocator releases the mapping, after the buffer's last use.
        let buffer = unsafe {
            no_copy(
                device,
                pointer,
                GUEST_PAGE,
                ResourceOptions::default(),
                Some(deallocator),
            )
        }
        .unwrap();
        let (submitted, stepping) = (
            device.new_shared_event().unwrap(),
            device.new_shared_event().unwrap(),
        );
        let mut blit = executing.blit_command_encoder().unwrap();
        blit.fill_buffer(&scratch, 0..256, 1).unwrap();
        blit.end_encoding().unwrap();
        executing
            .encode_signal_event(EventRef::from(&submitted), 1)
            .unwrap();
        executing
            .encode_wait_for_event(EventRef::from(&stepping), 1)
            .unwrap();
        let mut encoder = executing.compute_command_encoder().unwrap();
        setup.execute(&mut encoder);
        encoder.end_encoding().unwrap();
        let (reading, read) = mpsc::channel::<()>();
        let (go, gone) = mpsc::channel::<()>();
        let held = Mutex::new(Some((reading, gone)));
        executing
            .add_executed_commands_handler(Arc::new(move |_: &ExecutedCommands| {
                if let Some((reading, gone)) = held.lock().unwrap().take() {
                    reading.send(()).unwrap();
                    gone.recv_timeout(WAIT).unwrap();
                }
            }))
            .unwrap();
        executing.commit().unwrap();
        assert!(
            submitted.wait_until_signaled_value(1, WAIT),
            "the first batch ran"
        );
        // Listed only now, after the first submission: the later batch's commands reach it.
        gpu_fill(&other, &buffer, 0..64, 0x2d);
        stepping.set_signaled_value(1);
        read.recv_timeout(WAIT)
            .expect("the host step read the records");
        let before = device.work_counts();
        drop(buffer);
        assert_eq!(
            device.work_counts().host_deallocators_delayed - before.host_deallocators_delayed,
            1,
            "the release waits for the step's table"
        );
        go.send(()).unwrap();
        let (status, waited) = receiver
            .recv_timeout(WAIT)
            .expect("the deallocator ran and returned");
        assert_eq!(status, CommandBufferStatus::Completed);
        assert_eq!(waited, (Ok(()), Ok(())));
        wait(&executing);
        assert_eq!(setup.output(), expected());
    });
}

/// A command buffer's executed-commands handler sees each execution where the runtime reads its
/// records, with each command that runs, its state's `can_encode_indirect_commands`, and the
/// indirect command buffers made resident for writing:
///
/// - the CPU's records, read in `commit` before it returns: command 0 names the writer's state
///   made with `supportIndirectCommandBuffers` (it can encode) over an empty grid, which runs
///   nothing, and the rest the template's (it cannot); none made resident for writing;
/// - the kernel's records, read after the commit in the host step, where each command runs the
///   template's state, or on the GPU alone, where every state the records may name is the
///   template's; the buffer made resident for writing.
///
/// Nothing is seen before the commit or more than once, and a handler added after the commit is
/// refused.
#[test]
fn the_executed_commands_handler_sees_each_execution_where_its_records_are_read() {
    on_device(|device| handler_sees_each_execution(device, true));
}

/// [`the_executed_commands_handler_sees_each_execution_where_its_records_are_read`] on a device
/// that keeps no execution record, as a consumer's device keeps none (`record_executions` off,
/// the default): the handler sees the same executions with the same facts, and the command
/// buffers list no execution.
#[test]
fn the_executed_commands_handler_sees_each_execution_without_execution_records() {
    let consumer = |options: &mut DeviceOptions| options.record_executions = false;
    on_device_forcing_with(consumer, &[], |device| {
        handler_sees_each_execution(device, false);
    });
}

/// The handler lane's steps on `device`, which keeps execution records as `records` says.
fn handler_sees_each_execution(device: &Device, records: bool) {
    {
        let queue = device.new_command_queue().unwrap();
        let seen: Arc<Mutex<Vec<ExecutedCommands>>> = Arc::default();
        let handler = || -> ExecutedCommandsHandler {
            let seen = Arc::clone(&seen);
            Arc::new(move |executed: &ExecutedCommands| {
                seen.lock().unwrap().push(executed.clone());
            })
        };
        let take = || {
            let mut held = seen.lock().unwrap();
            let read = held.clone();
            held.clear();
            read
        };
        let facts = |executed: &ExecutedCommands| {
            executed
                .commands
                .iter()
                .map(|command| (command.index, command.can_encode_indirect_commands))
                .collect::<Vec<_>>()
        };
        // The CPU's records. Everything of it goes before the next case: a command's state lives
        // while its record names it, and would be one the GPU's records may name.
        {
            let setup = Setup::new(device, 2);
            setup.write_on_the_cpu();
            let encoding = state(device, "icb_replay_encode", true);
            assert!(encoding.can_encode_indirect_commands());
            let command = setup.icb.indirect_compute_command_at(0).unwrap();
            command.set_compute_pipeline_state(&encoding).unwrap();
            command
                .concurrent_dispatch_threadgroups(Size::new(0, 1, 1), Size::new(8, 1, 1))
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            command_buffer
                .add_executed_commands_handler(handler())
                .unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            setup.execute(&mut encoder);
            encoder.end_encoding().unwrap();
            assert!(take().is_empty(), "seen before the commit");
            command_buffer.commit().unwrap();
            let read = take();
            assert_eq!(read.len(), 1, "{read:?}");
            assert_eq!(read[0].read, RecordsRead::Commit);
            assert!(read[0].buffer == setup.icb);
            let mut commands = vec![(0, true)];
            commands.extend((1..COMMANDS).map(|index| (index, false)));
            assert_eq!(facts(&read[0]), commands);
            assert!(read[0].writable.is_empty());
            wait(&command_buffer);
            assert!(take().is_empty(), "seen again");
            assert_eq!(!command_buffer.executions().is_empty(), records);
        }
        // The kernel's records.
        let setup = Setup::new(device, 2);
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer
            .add_executed_commands_handler(handler())
            .unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.write(&mut encoder);
        setup.execute(&mut encoder);
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(setup.output(), expected());
        let read = take();
        assert_eq!(read.len(), 1, "{read:?}");
        let executed = &read[0];
        match executed.read {
            RecordsRead::HostStep => {
                let template: Vec<_> = (0..COMMANDS).map(|index| (index, false)).collect();
                assert_eq!(facts(executed), template);
            }
            RecordsRead::Gpu => {
                assert!(executed.commands.is_empty());
                assert!(!executed.unread_states_can_encode);
            }
            RecordsRead::Commit => panic!("the kernel's records read at the commit"),
            other => panic!("{other:?}"),
        }
        assert!(executed.writable.len() == 1 && executed.writable[0] == setup.icb);
        assert_eq!(!command_buffer.executions().is_empty(), records);
        assert_eq!(
            command_buffer.add_executed_commands_handler(handler()),
            Err(Error::Invalid(Invalid::new(
                "add_executed_commands_handler",
                Problem::HandlerAfterCommit
            )))
        );
        eprintln!(
            "CBF-MARK {}: the executed-commands handler saw the CPU's records at the commit and \
             the kernel's records in the {:?} (execution records kept: {records})",
            device.name(),
            executed.read
        );
    }
}
