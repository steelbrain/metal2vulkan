//! Executions of indirect command buffers the GPU may write, on the GPU alone by indirect
//! dispatches (`runtime/dispatch_execute.rs`), on the device the cfg selects, under the validation
//! layer. Every test opens its device with `VK_EXT_device_generated_commands` withheld
//! (`DeviceOptions::withhold_generated_commands`), so the dispatches' recording runs on every
//! device, MoltenVK's among them. The records are the CPU's, of two states made with
//! `supportIndirectCommandBuffers` whose kernels do not observe their threadgroup partition
//! (`flat_sum` of `indirect_dispatches` and [`FLAT_MARK`]), and an encoder makes the buffer
//! resident for writing, so the GPU may write them: each execution is total, and runs with no
//! host step. Each runs what Metal runs:
//!
//! - each command its own state, group count and buffers, a command never written and a reset
//!   command nothing, in a serial and in a concurrent encoder, with commands times states indirect
//!   dispatches and nothing announced; the serial encoder's with a barrier between every two
//!   commands, the concurrent encoder's, whose records no route could set a barrier in, with none;
//! - a concurrent encoder's execution whose buffer's records the CPU sets a barrier in after the
//!   commit, before its batch is submitted (it waits for an event), recorded again with the
//!   barriers, still on the GPU alone;
//! - one whose command buffer waits for an event while a buffer over host pages it never uses is
//!   rebound, recorded once, its resident table pointed at the set as it lies at the first
//!   submission;
//! - a range its own commands alone;
//! - a record a blit copied naming a state the program released since, which runs that state's
//!   program, after another state took its ID too;
//! - with as many live states as the GPU-alone path takes in its encoder's order on the GPU alone,
//!   and with one more in the host step, which runs it faster;
//! - inside a block of command predication, nothing where the block skips and every command where
//!   it holds, from the preprocess's gate;
//! - an execution `commit` recorded from the CPU's records (no encoder made the buffer resident
//!   for writing) whose records the CPU writes after the commit, before the batch is submitted (it
//!   waits for an event), runs them as written then, recorded again from the records then, and
//!   announces nothing; without the write, the commit's recording runs;
//! - a checked execution whose check has nothing to see (no state the records may name encodes
//!   indirect commands) runs on the GPU alone, its check never called, and one whose range is read
//!   from a buffer keeps the host step, which does not call it either; where a state the CPU set
//!   on a command may encode, the host step of either calls it with each command before any runs,
//!   a check that looks for nothing still uncalled; where the CPU sets such a state after the
//!   commit, before the batch is submitted, the host step records the execution instead and calls
//!   the check. Each runs the same; the host step is announced once.

use super::degrade_notices::assert_made_once;
use super::generated_executions::flat_template;
use super::host_memory::Seen;
use super::indirect_host_steps::{self, HOST_STEPS};
use super::rebind_reach::{buffer_len, buffer_over, pages};
use super::rebinds::rebind;
use super::shared_kinds;
use super::{layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache, WAIT};
use crate::air_library::AirLibrary;
use crate::candidate::runtime_case::{buffer_with, read_buffer};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::StatusReport;
use metal2vulkan::metal::{DispatchType, IndirectCommandType, ResourceUsage};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, RuntimeFact, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, ComputeCommandEncoder, ComputePipelineState,
    Degradation, Device, EventRef, ExecutedPipeline, IndirectCommandBuffer,
    IndirectCommandBufferDescriptor, IndirectCommandCheck, IndirectCommandCheckNeeds,
    PipelineOptions, ResourceOptions, Size,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// `flat_mark(params, out, p)`: each thread adds `params[0] * 1000 + p.x` to `out[p.x]` by a
/// relaxed atomic, reading only `thread_position_in_grid`: it does not observe its threadgroup
/// partition.
const FLAT_MARK: &str = r#"define void @flat_mark(ptr addrspace(1) noundef "air-buffer-no-alias" %b0, ptr addrspace(1) noundef "air-buffer-no-alias" %b1, <3 x i32> noundef %p) {
entry:
  %base = load i32, ptr addrspace(1) %b0, align 4
  %px = extractelement <3 x i32> %p, i64 0
  %scaled = mul i32 %base, 1000
  %v = add i32 %scaled, %px
  %px64 = zext i32 %px to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %b1, i64 %px64
  %old = atomicrmw add ptr addrspace(1) %o, i32 %v monotonic, align 4
  ret void
}

!air.kernel = !{!0}
!air.version = !{!20}
!air.language_version = !{!21}
!0 = !{ptr @flat_mark, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b0"}
!4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"b1"}
!5 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"p"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The words of one command's window of `params` and of `out`.
const WINDOW: usize = 64;

/// The commands of the buffer.
const COMMANDS: u64 = 8;

/// What a command holds: nothing, or a dispatch of a state (`Sum`: `flat_sum`, `Mark`:
/// [`FLAT_MARK`]'s) of `threads` threads in all, as threadgroups or as threads.
#[derive(Clone, Copy)]
enum Command {
    /// Never written.
    Unwritten,
    /// Written, then reset.
    Reset,
    Sum(Grid),
    Mark(Grid),
}

/// A command's dispatch: `groups` threadgroups of `local` threads, or `threads` threads in
/// threadgroups of `local`.
#[derive(Clone, Copy)]
enum Grid {
    Groups { groups: u64, local: u64 },
    Threads { threads: u64, local: u64 },
}

impl Grid {
    /// The threads it runs.
    fn threads(self) -> usize {
        match self {
            Grid::Groups { groups, local } => (groups * local) as usize,
            Grid::Threads { threads, .. } => threads as usize,
        }
    }
}

/// The commands, by index: each state, a ragged grid among the threads dispatches, an unwritten
/// and a reset command.
const PLAN: [Command; COMMANDS as usize] = [
    Command::Sum(Grid::Groups {
        groups: 2,
        local: 8,
    }),
    Command::Mark(Grid::Threads {
        threads: 24,
        local: 8,
    }),
    Command::Unwritten,
    Command::Sum(Grid::Threads {
        threads: 5,
        local: 4,
    }),
    Command::Mark(Grid::Groups {
        groups: 1,
        local: 64,
    }),
    Command::Reset,
    Command::Mark(Grid::Threads {
        threads: 7,
        local: 2,
    }),
    Command::Sum(Grid::Groups {
        groups: 1,
        local: 1,
    }),
];

/// Word 0 of command `command`'s window of `params`.
fn base(command: usize) -> u32 {
    command as u32 + 2
}

/// What Metal computes into `out` where commands `ran` run once each: `flat_sum`'s thread `p` of
/// a grid of `t` adds `base + 3 p + 11 t + 13 + 17`, [`FLAT_MARK`]'s `base * 1000 + p`, each to
/// word `p` of its command's window, which starts at 0.
fn expected(ran: std::ops::Range<usize>) -> Vec<u32> {
    let mut out = vec![0u32; COMMANDS as usize * WINDOW];
    for command in ran {
        let window = &mut out[command * WINDOW..][..WINDOW];
        let b = base(command);
        match PLAN[command] {
            Command::Unwritten | Command::Reset => {}
            Command::Sum(grid) => {
                let t = grid.threads() as u32;
                for p in 0..t {
                    window[p as usize] += b + 3 * p + 11 * t + 30;
                }
            }
            Command::Mark(grid) => {
                for p in 0..grid.threads() as u32 {
                    window[p as usize] += b * 1000 + p;
                }
            }
        }
    }
    out
}

/// [`expected`] over a buffer of `size` commands: those past [`PLAN`]'s run nothing.
fn expected_of(size: u64, ran: std::ops::Range<usize>) -> Vec<u32> {
    let mut out = expected(ran);
    out.resize(size as usize * WINDOW, 0);
    out
}

/// The test's buffers and states: the indirect command buffer of [`PLAN`]'s commands, written by
/// the CPU, `params`, `out`, and both states.
struct Setup {
    icb: IndirectCommandBuffer,
    params: Buffer,
    out: Buffer,
    /// Held: the states made with `supportIndirectCommandBuffers` alive, which the records may
    /// name.
    states: [ComputePipelineState; 2],
}

impl Setup {
    fn new(device: &Device) -> Setup {
        Setup::sized(device, COMMANDS)
    }

    /// [`Setup::new`] of a buffer of `size` commands (at least [`COMMANDS`]), those past
    /// [`PLAN`]'s never written, with windows of `params` and `out` for each.
    fn sized(device: &Device, size: u64) -> Setup {
        let sum = flat_template(device);
        let mark = mark_state(device);
        let mut descriptor = IndirectCommandBufferDescriptor::new();
        descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH
            .union(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);
        descriptor.max_kernel_buffer_bind_count = 2;
        let icb = device
            .new_indirect_command_buffer(&descriptor, size, ResourceOptions::shared())
            .unwrap();
        let params: Vec<u8> = (0..size as usize * WINDOW)
            .map(|word| match word % WINDOW {
                0 => base(word / WINDOW),
                _ => 0,
            })
            .flat_map(u32::to_le_bytes)
            .collect();
        let params = buffer_with(device, &params).unwrap();
        let out = buffer_with(device, &vec![0; size as usize * WINDOW * 4]).unwrap();
        for (index, command) in PLAN.iter().enumerate() {
            let (state, grid) = match *command {
                Command::Unwritten => continue,
                Command::Reset => (
                    &sum,
                    Grid::Groups {
                        groups: 1,
                        local: 1,
                    },
                ),
                Command::Sum(grid) => (&sum, grid),
                Command::Mark(grid) => (&mark, grid),
            };
            let at = icb.indirect_compute_command_at(index as u64).unwrap();
            at.set_compute_pipeline_state(state).unwrap();
            let offset = (index * WINDOW * 4) as u64;
            at.set_kernel_buffer(Some(&params), offset, 0).unwrap();
            at.set_kernel_buffer(Some(&out), offset, 1).unwrap();
            match grid {
                Grid::Groups { groups, local } => at
                    .concurrent_dispatch_threadgroups(
                        Size::new(groups, 1, 1),
                        Size::new(local, 1, 1),
                    )
                    .unwrap(),
                Grid::Threads { threads, local } => at
                    .concurrent_dispatch_threads(Size::new(threads, 1, 1), Size::new(local, 1, 1))
                    .unwrap(),
            }
            if matches!(command, Command::Reset) {
                at.reset().unwrap();
            }
        }
        Setup {
            icb,
            params,
            out,
            states: [sum, mark],
        }
    }

    /// Make the buffer resident for writing in an encoder of `command_buffer` that writes
    /// nothing: from then on the GPU may write its records, so an execution reads them on the
    /// GPU or in a host step.
    fn writable(&self, command_buffer: &CommandBuffer) {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder
            .use_resource(self.icb.as_resource(), ResourceUsage::WRITE)
            .unwrap();
        encoder.end_encoding().unwrap();
        assert!(self.icb.gpu_written());
    }

    /// Make the commands' windows resident on `encoder`.
    fn resident(&self, encoder: &mut ComputeCommandEncoder) {
        encoder
            .use_resource(self.params.as_resource(), ResourceUsage::READ)
            .unwrap();
        encoder
            .use_resource(self.out.as_resource(), ResourceUsage::WRITE)
            .unwrap();
    }

    /// `out`, read back.
    fn output(&self) -> Vec<u32> {
        read_buffer(&self.out, 0, self.out.length())
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect()
    }
}

/// [`FLAT_MARK`]'s state on `device`, made with `supportIndirectCommandBuffers`.
fn mark_state(device: &Device) -> ComputePipelineState {
    mark_states(device, 1).pop().unwrap()
}

/// `count` states of [`FLAT_MARK`]'s on `device`, each its own, made with
/// `supportIndirectCommandBuffers` from one library.
fn mark_states(device: &Device, count: u64) -> Vec<ComputePipelineState> {
    let library = device
        .new_library_with_air(&llvm_assemble_text(FLAT_MARK).unwrap())
        .unwrap();
    let function = library.new_function_with_name("flat_mark").unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = true;
    (0..count)
        .map(|_| {
            device
                .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
                .unwrap()
        })
        .collect()
}

/// Run `test` on a validated device with `VK_EXT_device_generated_commands` withheld; then require
/// that the layer reported nothing.
fn on_device(test: impl FnOnce(&Device, &Messages)) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.withhold_generated_commands = true;
        })
        .unwrap();
        assert!(device.compile_device().generated_commands().is_none());
        test(&device, &messages);
    }
    layer_reported_nothing_at_all(&messages);
}

/// The indirect dispatches `device` replayed so far.
fn dispatches_indirect(device: &Device) -> u64 {
    device.work_counts().commands_recorded.cmd_dispatch_indirect
}

/// The pipeline barriers `device` replayed so far.
fn barriers(device: &Device) -> u64 {
    device.work_counts().commands_recorded.cmd_pipeline_barrier
}

/// Whether `command_buffer` ran a flat variant: the GPU-alone path's execution record, where the
/// host step's runs the variants its dispatches plan.
fn ran_flat(command_buffer: &CommandBuffer) -> bool {
    command_buffer
        .executions()
        .iter()
        .any(|execution| match execution.pipeline() {
            ExecutedPipeline::Compute(pipeline) => pipeline
                .dependencies
                .iter()
                .any(|dependency| dependency.fact == RuntimeFact::FlatGrid),
            ExecutedPipeline::Render(_) => false,
        })
}

/// Commit `command_buffer`, wait for it, and require it completed with a clean report.
fn complete(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(
        command_buffer.execution_report(),
        Some(StatusReport::default())
    );
}

/// Require that `device` announced `kinds` once each, and beside them only what the test's
/// `Shared` buffers force: nothing of its own for executions on the GPU alone.
fn announced(device: &Device, messages: &Messages, kinds: &[Degradation]) {
    assert_made_once(messages, kinds, &shared_kinds(device));
}

/// What the host step of [`PLAN`]'s commands announces: the step, and `split-dispatch` for the
/// ragged grid it dispatches as Vulkan does, whole workgroups (the flat variants on the GPU alone
/// take it exactly, announcing nothing).
const STEPPED: [Degradation; 2] = [HOST_STEPS[0], Degradation::SplitDispatch];

/// Every command runs its own state, group count and buffers, and the unwritten and reset ones
/// nothing, in a serial and in a concurrent encoder: on the GPU alone, one indirect dispatch per
/// command and state, the execution record listing the flat variants, nothing announced. The
/// serial encoder's commands take a barrier between every two, the concurrent encoder's none: no
/// route set a barrier in the buffer's records.
#[test]
fn each_command_runs_its_own_state_and_group_count() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let mut ordered = Vec::new();
        for dispatch_type in [DispatchType::Serial, DispatchType::Concurrent] {
            let setup = Setup::new(device);
            let command_buffer = queue.command_buffer().unwrap();
            setup.writable(&command_buffer);
            let before = dispatches_indirect(device);
            let barriers_before = barriers(device);
            let mut encoder = command_buffer
                .compute_command_encoder_with_dispatch_type(dispatch_type)
                .unwrap();
            setup.resident(&mut encoder);
            encoder
                .execute_commands_in_buffer(&setup.icb, 0..COMMANDS)
                .unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            let at = format!("{dispatch_type:?}");
            assert_eq!(setup.output(), expected(0..COMMANDS as usize), "{at}");
            assert!(ran_flat(&command_buffer), "{at}");
            let states = setup.states.len() as u64;
            assert_eq!(
                dispatches_indirect(device) - before,
                COMMANDS * states,
                "{at}: one indirect dispatch per command and state"
            );
            ordered.push(barriers(device) - barriers_before);
        }
        assert_eq!(
            ordered[0] - ordered[1],
            COMMANDS - 1,
            "a barrier between every two commands in the serial encoder alone: {ordered:?}"
        );
        announced(device, messages, &[]);
    });
}

/// A record the GPU wrote naming a state the program released since runs that state's program on
/// the GPU alone, as Metal's runs it (`released_programs.rs`): a blit copies a command of
/// [`FLAT_MARK`]'s state from a buffer of its own into the unwritten command 2 (the GPU writes
/// it), and the program releases the state and the source buffer; the execution runs it beside
/// the others, one indirect dispatch per command and state (the kept program's among them), with
/// nothing announced. So it does where a state made after (`flat_sum`'s) took the released
/// state's ID: the ID names the kept program, as the copy wrote it before the new state took it.
#[test]
fn a_copied_record_runs_its_released_state_s_program_on_the_gpu_alone() {
    for taken in [false, true] {
        on_device(|device, messages| {
            let queue = device.new_command_queue().unwrap();
            let setup = Setup::new(device);
            let threads = 8;
            let id = {
                let mark = mark_state(device);
                let mut descriptor = IndirectCommandBufferDescriptor::new();
                descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH
                    .union(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);
                descriptor.max_kernel_buffer_bind_count = 2;
                let source = device
                    .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
                    .unwrap();
                let at = source.indirect_compute_command_at(0).unwrap();
                at.set_compute_pipeline_state(&mark).unwrap();
                let offset = (2 * WINDOW * 4) as u64;
                at.set_kernel_buffer(Some(&setup.params), offset, 0)
                    .unwrap();
                at.set_kernel_buffer(Some(&setup.out), offset, 1).unwrap();
                at.concurrent_dispatch_threads(Size::new(threads, 1, 1), Size::new(threads, 1, 1))
                    .unwrap();
                let command_buffer = queue.command_buffer().unwrap();
                let mut blit = command_buffer.blit_command_encoder().unwrap();
                blit.copy_indirect_command_buffer(&source, 0..1, &setup.icb, 2)
                    .unwrap();
                blit.end_encoding().unwrap();
                complete(&command_buffer);
                mark.gpu_resource_id()
            };
            let taker = taken.then(|| flat_template(device));
            if let Some(taker) = &taker {
                assert_eq!(taker.gpu_resource_id(), id, "flat_sum's state takes the ID");
            }
            let before = dispatches_indirect(device);
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            setup.resident(&mut encoder);
            encoder
                .execute_commands_in_buffer(&setup.icb, 0..COMMANDS)
                .unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            let mut expected = expected(0..COMMANDS as usize);
            for p in 0..threads as usize {
                expected[2 * WINDOW + p] += base(2) * 1000 + p as u32;
            }
            assert_eq!(setup.output(), expected, "taken {taken}");
            assert!(ran_flat(&command_buffer), "taken {taken}: on the GPU alone");
            assert_eq!(
                dispatches_indirect(device) - before,
                COMMANDS * 3,
                "taken {taken}: one indirect dispatch per command and state"
            );
            announced(device, messages, &[]);
        });
    }
}

/// A concurrent encoder's execution recorded with no barrier between its commands is recorded
/// again with them where the CPU sets a barrier in one of the buffer's records after the commit,
/// before its batch is submitted (the batch waits for an event the CPU signals after the set): the
/// commands run on the GPU alone, with as many barriers as the serial encoder's, and write what
/// Metal computes. Without the set, none.
#[test]
fn a_barrier_set_after_the_commit_records_the_commands_again() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let mut taken = Vec::new();
        for set in [false, true] {
            let setup = Setup::new(device);
            let event = device.new_shared_event().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            setup.writable(&command_buffer);
            command_buffer
                .encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            let mut encoder = command_buffer
                .compute_command_encoder_with_dispatch_type(DispatchType::Concurrent)
                .unwrap();
            setup.resident(&mut encoder);
            encoder
                .execute_commands_in_buffer(&setup.icb, 0..COMMANDS)
                .unwrap();
            encoder.end_encoding().unwrap();
            let before = barriers(device);
            command_buffer.commit().unwrap();
            if set {
                let command = setup.icb.indirect_compute_command_at(3).unwrap();
                command.set_barrier().unwrap();
            }
            event.set_signaled_value(1);
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            assert_eq!(
                command_buffer.execution_report(),
                Some(StatusReport::default())
            );
            assert_eq!(setup.output(), expected(0..COMMANDS as usize), "set {set}");
            assert!(ran_flat(&command_buffer), "set {set}");
            taken.push(barriers(device) - before);
        }
        assert!(
            taken[1] >= taken[0] + COMMANDS - 1,
            "the commands recorded again with a barrier between every two: {taken:?}"
        );
        announced(device, messages, &[]);
    });
}

/// A rebind between the commit and the first submission (an event wait holds the command buffer)
/// of a buffer over host pages the commands never use moves the live buffers the preprocess
/// searches: the first submission points the execution's resident table at the set as it lies
/// then (`command_buffer/first_submission.rs`), nothing is recorded again, and it runs what it
/// runs without the rebind, the command buffer completing with a clean report (not
/// `EncodedUseRebound`) and announcing nothing.
#[test]
fn a_rebind_before_the_first_submission_reaches_the_execution_s_records_as_they_lie() {
    on_device(|device, messages| {
        let seen = Arc::new(Seen::default());
        let Some(unused) = buffer_over(device, pages(buffer_len(), 0, &[1]), &seen) else {
            return;
        };
        let queue = device.new_command_queue().unwrap();
        let setup = Setup::new(device);
        let event = device.new_shared_event().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        // The wait first: its first batch waits for the event too, so it is submitted after the
        // rebind.
        command_buffer
            .encode_wait_for_event(EventRef::from(&event), 1)
            .unwrap();
        setup.writable(&command_buffer);
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.resident(&mut encoder);
        encoder
            .execute_commands_in_buffer(&setup.icb, 0..COMMANDS)
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        let before = device.work_counts();
        let rebound = Arc::new(Seen::default());
        rebind(&unused, pages(buffer_len(), 0, &[2]), &rebound).unwrap();
        event.set_signaled_value(1);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let after = device.work_counts();
        assert_eq!(
            after.segments_recorded_again, before.segments_recorded_again,
            "recorded once"
        );
        assert_eq!(after.reencodes_refused, before.reencodes_refused);
        assert!(
            after.replay_tables_pointed > before.replay_tables_pointed,
            "its resident table pointed at the first submission"
        );
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert_eq!(
            command_buffer.execution_report(),
            Some(StatusReport::default())
        );
        assert_eq!(setup.output(), expected(0..COMMANDS as usize));
        assert!(ran_flat(&command_buffer), "on the GPU alone");
        announced(device, messages, &[]);
    });
}

/// The commands of the executions [`MOST_STATES`] bounds: [`PLAN`]'s and as many never written,
/// so both bounds fall below the replay's 64 variants (`bind::MAX_REPLAY_VARIANTS`).
const BOUNDED: u64 = 2 * COMMANDS;

/// The most live states an execution of [`BOUNDED`] commands runs with on the GPU alone, as
/// `runtime/gpu_execute.rs`'s `dispatches_run` bounds them: where its commands go with no barrier
/// between them (a concurrent encoder's, whose records no route could set a barrier in), four
/// free and 256 dispatches past them (`FREE_STATES_UNORDERED`, `DISPATCHES_PAST_UNORDERED`); where
/// they go with one (a serial encoder's), twelve and 512.
const MOST_STATES: [(DispatchType, u64); 2] = [
    (DispatchType::Concurrent, 4 + 256 / BOUNDED),
    (DispatchType::Serial, 12 + 512 / BOUNDED),
];

/// An execution whose records may name as many live states as the GPU-alone path takes for its
/// commands in its encoder's order runs on the GPU alone, one indirect dispatch per command and
/// state, announcing nothing; with one more state alive, which no record names, the host step
/// runs it, faster, with no indirect dispatch, announced once. Both write what Metal computes, in
/// a concurrent and in a serial encoder.
#[test]
fn an_execution_of_more_states_than_the_gpu_path_takes_runs_in_the_host_step() {
    for (dispatch_type, most) in MOST_STATES {
        on_device(|device, messages| {
            let queue = device.new_command_queue().unwrap();
            for states in [most, most + 1] {
                let setup = Setup::sized(device, BOUNDED);
                // Alive with `supportIndirectCommandBuffers` past the two the records name.
                let unnamed = mark_states(device, states - setup.states.len() as u64);
                let command_buffer = queue.command_buffer().unwrap();
                setup.writable(&command_buffer);
                let before = dispatches_indirect(device);
                let mut encoder = command_buffer
                    .compute_command_encoder_with_dispatch_type(dispatch_type)
                    .unwrap();
                setup.resident(&mut encoder);
                encoder
                    .execute_commands_in_buffer(&setup.icb, 0..BOUNDED)
                    .unwrap();
                encoder.end_encoding().unwrap();
                complete(&command_buffer);
                let at = format!("{dispatch_type:?} of {states} states");
                let mut want = expected(0..COMMANDS as usize);
                want.resize(BOUNDED as usize * WINDOW, 0);
                assert_eq!(setup.output(), want, "{at}");
                let alone = states <= most;
                assert_eq!(ran_flat(&command_buffer), alone, "{at}");
                let dispatched = dispatches_indirect(device) - before;
                assert_eq!(dispatched, if alone { BOUNDED * states } else { 0 }, "{at}");
                drop(unnamed);
            }
            announced(device, messages, &STEPPED);
        });
    }
}

/// A range runs its own commands alone, on the GPU alone, one indirect dispatch per command of the
/// range and state.
#[test]
fn a_range_runs_its_commands_alone() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let setup = Setup::new(device);
        let command_buffer = queue.command_buffer().unwrap();
        setup.writable(&command_buffer);
        let before = dispatches_indirect(device);
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.resident(&mut encoder);
        encoder
            .execute_commands_in_buffer(&setup.icb, 1..5)
            .unwrap();
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        let mut want = expected(1..5);
        // Command 0's window stays zero though the range ends past it.
        want[..WINDOW].fill(0);
        assert_eq!(setup.output(), want);
        assert!(ran_flat(&command_buffer));
        assert_eq!(dispatches_indirect(device) - before, 4 * 2);
        announced(device, messages, &[]);
    });
}

/// Inside a block of command predication an execution runs from its preprocess's gate, on the GPU
/// alone: nothing in a failing If's block, every command in its Else, and the same in a holding
/// If's block, whose Else runs nothing.
#[test]
fn predicated_executions_run_as_their_blocks_say() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        // Word 0 is 1: an If on `== 1` holds, one on `== 0` fails.
        let word = buffer_with(device, &1u32.to_le_bytes()).unwrap();
        for holds in [false, true] {
            let setup = Setup::new(device);
            let command_buffer = queue.command_buffer().unwrap();
            setup.writable(&command_buffer);
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            setup.resident(&mut encoder);
            encoder
                .encode_start_if(Some(&word), 0, 0, u32::from(holds))
                .unwrap();
            encoder
                .execute_commands_in_buffer(&setup.icb, 0..COMMANDS)
                .unwrap();
            encoder.encode_start_else().unwrap();
            encoder
                .execute_commands_in_buffer(&setup.icb, 0..COMMANDS)
                .unwrap();
            assert_eq!(encoder.encode_end_if(), Ok(true));
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            assert_eq!(
                setup.output(),
                expected(0..COMMANDS as usize),
                "holds {holds}: one of the two executions ran"
            );
            assert!(ran_flat(&command_buffer), "holds {holds}");
        }
        announced(device, messages, &[]);
    });
}

/// A state that may encode indirect commands (`icb_replay_encode`'s), made with
/// `supportIndirectCommandBuffers`: a command the CPU sets it on may name a state that encodes.
fn encoding_state(device: &Device) -> ComputePipelineState {
    indirect_host_steps::state(device, "icb_replay_encode", true)
}

/// A check that counts its calls, and the count.
fn counting() -> (IndirectCommandCheck, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let check: IndirectCommandCheck = Box::new(move |_, _| {
        counted.fetch_add(1, Ordering::SeqCst);
    });
    (check, calls)
}

/// An execution of the CPU's records, which `commit` records (no encoder made the buffer resident
/// for writing), runs the records as they stand when its batch is submitted, as Metal's execution
/// reads them as it runs: where the CPU resets command 0 after the commit, before the batch is
/// submitted (it waits for an event the test signals after the reset), a host step records the
/// execution again from the records then (each record read again), command 0 runs nothing and the
/// rest run; without the reset, the commit's recording runs, each record read once. Nothing is
/// announced but the split of a ragged grid the dispatches take.
#[test]
fn the_cpu_s_writes_after_the_commit_run_where_the_batch_waits() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        for write in [false, true] {
            let setup = Setup::new(device);
            let event = device.new_shared_event().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            command_buffer
                .encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            setup.resident(&mut encoder);
            encoder
                .execute_commands_in_buffer(&setup.icb, 0..COMMANDS)
                .unwrap();
            encoder.end_encoding().unwrap();
            let before = device.work_counts().indirect_records_read;
            command_buffer.commit().unwrap();
            if write {
                let command = setup.icb.indirect_compute_command_at(0).unwrap();
                command.reset().unwrap();
            }
            event.set_signaled_value(1);
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            assert_eq!(
                command_buffer.execution_report(),
                Some(StatusReport::default())
            );
            let read = device.work_counts().indirect_records_read - before;
            assert!(!setup.icb.gpu_written(), "write {write}");
            let ran = if write { 1 } else { 0 };
            assert_eq!(
                setup.output(),
                expected(ran..COMMANDS as usize),
                "write {write}"
            );
            let reads = if write { 2 * COMMANDS } else { COMMANDS };
            assert_eq!(read, reads, "write {write}: the records read");
        }
        announced(device, messages, &[Degradation::SplitDispatch]);
    });
}

/// A checked execution's check is called only where what the buffer's records may hold leaves it
/// something to see (`IndirectCommandCheckNeeds`): with the flat states alone, which encode no
/// indirect commands, a check that looks for every kind is never called, and a range given as
/// such runs on the GPU alone, while one read from a buffer keeps the host step; once the CPU set
/// a state that may encode on a command outside the range, the host step of either calls it with
/// each command of the range, and a check that looks for nothing is still never called. Each runs
/// what Metal runs; the host step is announced once.
#[test]
fn checked_executions_call_their_check_where_it_may_see_something() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let (all, none) = (
            IndirectCommandCheckNeeds::ALL,
            IndirectCommandCheckNeeds::new(false, false),
        );
        let mut encoding = None;
        for (encodes, needs, indirect) in [
            (false, all, false),
            (false, all, true),
            (true, all, false),
            (true, all, true),
            (true, none, false),
            (true, none, true),
        ] {
            let setup = Setup::sized(device, COMMANDS + 1);
            if encodes {
                // Made only now: with it alive, the GPU-alone path would run its variant too.
                let encoding = encoding.get_or_insert_with(|| encoding_state(device));
                let outside = setup.icb.indirect_compute_command_at(COMMANDS).unwrap();
                outside.set_compute_pipeline_state(encoding).unwrap();
            }
            let command_buffer = queue.command_buffer().unwrap();
            setup.writable(&command_buffer);
            let range = buffer_with(device, &[0, 0, 0, 0, COMMANDS as u8, 0, 0, 0]).unwrap();
            let (check, calls) = counting();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            setup.resident(&mut encoder);
            match indirect {
                false => encoder.execute_commands_in_buffer_checked(
                    &setup.icb,
                    0..COMMANDS,
                    check,
                    needs,
                ),
                true => encoder.execute_commands_in_buffer_indirect_checked(
                    &setup.icb, &range, 0, check, needs,
                ),
            }
            .unwrap();
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            let at = format!("encodes {encodes}, {needs:?}, indirect {indirect}");
            let want = expected_of(COMMANDS + 1, 0..COMMANDS as usize);
            assert_eq!(setup.output(), want, "{at}");
            let called = encodes && needs == all;
            let expected_calls = if called { COMMANDS as usize } else { 0 };
            assert_eq!(calls.load(Ordering::SeqCst), expected_calls, "{at}");
            if !encodes {
                assert_eq!(ran_flat(&command_buffer), !indirect, "{at}");
            } else if called {
                assert!(!ran_flat(&command_buffer), "{at}");
            }
        }
        // The check reads the commands, which answer the runtime's unique identifiers of their
        // states, announced as the getter announces them.
        let identifiers = [Degradation::UnmeasuredUniqueIdentifier];
        let kinds: Vec<Degradation> = STEPPED.iter().chain(&identifiers).copied().collect();
        announced(device, messages, &kinds);
    });
}

/// A checked execution `commit` recorded on the GPU alone, its check having nothing to see (the
/// flat states encode no indirect commands), takes the host step where the CPU sets a state that
/// may encode on a command after the commit, before the batch is submitted (it waits for an event
/// the test signals after the set): the step calls the check with each command of the range and
/// runs what Metal runs, in a serial and in a concurrent encoder (whose GPU-alone recording kept
/// what its batch's submission would record again). Without the set the check is never called and
/// the commands run on the GPU alone.
#[test]
fn a_check_given_something_to_see_after_the_commit_takes_the_host_step() {
    on_device(|device, messages| {
        let queue = device.new_command_queue().unwrap();
        let mut encoding = None;
        for (dispatch_type, set) in [
            (DispatchType::Serial, false),
            (DispatchType::Serial, true),
            (DispatchType::Concurrent, true),
        ] {
            let setup = Setup::sized(device, COMMANDS + 1);
            let event = device.new_shared_event().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            setup.writable(&command_buffer);
            command_buffer
                .encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            let mut encoder = command_buffer
                .compute_command_encoder_with_dispatch_type(dispatch_type)
                .unwrap();
            setup.resident(&mut encoder);
            let (check, calls) = counting();
            encoder
                .execute_commands_in_buffer_checked(
                    &setup.icb,
                    0..COMMANDS,
                    check,
                    IndirectCommandCheckNeeds::new(false, true),
                )
                .unwrap();
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            if set {
                // Made after the commit, which recorded the states alive then.
                let encoding = encoding.get_or_insert_with(|| encoding_state(device));
                let outside = setup.icb.indirect_compute_command_at(COMMANDS).unwrap();
                outside.set_compute_pipeline_state(encoding).unwrap();
            }
            event.set_signaled_value(1);
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            assert_eq!(
                command_buffer.execution_report(),
                Some(StatusReport::default())
            );
            let at = format!("{dispatch_type:?}, set {set}");
            let want = expected_of(COMMANDS + 1, 0..COMMANDS as usize);
            assert_eq!(setup.output(), want, "{at}");
            let expected_calls = if set { COMMANDS as usize } else { 0 };
            assert_eq!(calls.load(Ordering::SeqCst), expected_calls, "{at}");
            assert_eq!(ran_flat(&command_buffer), !set, "{at}");
        }
        let identifiers = [Degradation::UnmeasuredUniqueIdentifier];
        let kinds: Vec<Degradation> = STEPPED.iter().chain(&identifiers).copied().collect();
        announced(device, messages, &kinds);
    });
}
