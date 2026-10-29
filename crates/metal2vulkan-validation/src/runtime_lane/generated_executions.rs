//! Executions of indirect command buffers whose compute commands a kernel writes, on the GPU
//! alone, on the device the cfg selects, under the validation layer. The owned `icb_replay_encode`
//! fixture's kernel writes the commands as in `indirect_host_steps`, here of a state of the flat
//! kernel of `indirect_dispatches` ([`FLAT_SUM`]), which does not observe its threadgroup
//! partition: its flat replay variant binds no descriptor, so an execution is total where it is the
//! only state made with `supportIndirectCommandBuffers`. Each test opens its device as the runtime
//! does, `VK_EXT_device_generated_commands` enabled where the device has it (as every lane's
//! device), and again with it withheld (`DeviceOptions::withhold_generated_commands`), the indirect
//! dispatches' leg (`dispatch_executions`):
//!
//! - a total execution runs on the GPU alone, through the generated commands with the extension
//!   and by indirect dispatches without it: nothing is announced, and the execution record lists
//!   the flat variant once. Both write what Metal computes, in the encoder that wrote the commands
//!   and in a later command buffer;
//! - a state alive at the commit that observes its partition keeps the host step, which runs
//!   what Metal computes, and so does one whose kernel fetches a `[[stage_in]]` attribute
//!   (the host step checks each command's fetch past 4 GiB);
//! - a sticky encoding fault in the header runs no command, and the execution report says so;
//! - an execution whose commands' windows no encoder made resident writes what Metal computes
//!   on either path: both reach every live buffer of its own.
//!
//! A device that has no `VK_EXT_device_generated_commands` (or not what it requires) runs the
//! indirect dispatches either way, and prints a `NO-DGC` line.

use super::degrade_notices::assert_made_once;
use super::indirect_dispatches::FLAT_SUM;
use super::indirect_host_steps::{
    complete, param, state, wait, Setup, COMMANDS, HOST_STEPS, THREADS,
};
use super::shared_kinds;
use super::{layer_reported_nothing_at_all, validated_device_with, Messages, ScratchCache};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::ReplayFaults;
use metal2vulkan::metal::StepFunction;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, RuntimeFact, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{
    CommandBuffer, CommandBufferStatus, ComputePipelineState, Degradation, Device,
    ExecutedPipeline, PipelineOptions,
};

/// The flat kernel's state on `device`, made with `supportIndirectCommandBuffers`.
pub(super) fn flat_template(device: &Device) -> ComputePipelineState {
    let library = device
        .new_library_with_air(&llvm_assemble_text(FLAT_SUM).unwrap())
        .unwrap();
    let function = library.new_function_with_name("flat_sum").unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.support_indirect_command_buffers = true;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// What Metal computes: each command's thread `p` of its grid of `THREADS` adds
/// `params[0] + 3 p + 11 THREADS + 13 + 17` of its window to word `p` of its window of `out`,
/// which starts at 0.
fn expected() -> Vec<u32> {
    (0..COMMANDS as usize)
        .flat_map(|command| {
            (0..THREADS as u32).map(move |p| param(command, 0) + 3 * p + 11 * THREADS as u32 + 30)
        })
        .collect()
}

/// Run `test` on a validated device, `VK_EXT_device_generated_commands` withheld as `withheld`
/// says, with whether the device enabled it; then require that the layer reported nothing.
fn on_device(withheld: bool, test: impl FnOnce(&Device, &Messages, bool)) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.withhold_generated_commands = withheld;
        })
        .unwrap();
        let generated = device.compile_device().generated_commands().is_some();
        assert!(
            !(withheld && generated),
            "the extension was withheld, yet enabled"
        );
        if !withheld && !generated {
            println!(
                "NO-DGC {}: the device enables no VK_EXT_device_generated_commands; the \
                 generated-commands path did not run",
                device.name()
            );
        }
        test(&device, &messages, generated);
    }
    layer_reported_nothing_at_all(&messages);
}

/// Whether `command_buffer` ran the flat kernel's flat variant: the GPU-alone path's execution
/// record, where the host step's runs the variants its dispatches plan.
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

/// The generated-commands executions and the indirect dispatches `device` recorded so far.
fn recorded(device: &Device) -> (u64, u64) {
    let commands = device.work_counts().commands_recorded;
    (
        commands.cmd_execute_generated_commands,
        commands.cmd_dispatch_indirect,
    )
}

/// Require that the recording of an execution on the GPU alone that `generated` names ran since
/// `before` ([`recorded`]): generated-commands executions where the device enabled the
/// extension, else indirect dispatches and no generated-commands execution.
fn ran_recording(device: &Device, generated: bool, before: (u64, u64), at: &str) {
    let (executions, dispatches) = recorded(device);
    if generated {
        assert!(
            executions > before.0,
            "{at}: no generated-commands execution ran"
        );
    } else {
        assert_eq!(
            executions, before.0,
            "{at}: a generated-commands execution ran"
        );
        assert!(dispatches > before.1, "{at}: no indirect dispatch ran");
    }
}

/// Require that the host step was announced once where the GPU-alone path did not run, and never
/// where it did. Beside it, only what the test's `Shared` buffers force on `device` the first time
/// ([`shared_kinds`]); before, the GPU-alone path let any other kind pass.
fn announced(device: &Device, messages: &Messages, gpu: bool) {
    let kinds: &[Degradation] = match gpu {
        true => &[],
        false => &HOST_STEPS,
    };
    assert_made_once(messages, kinds, &shared_kinds(device));
}

/// A total execution runs on the GPU alone, through the generated commands where the device
/// enables the extension and by indirect dispatches otherwise; both write what Metal computes, in
/// the encoder that wrote the commands and in a later command buffer, the second round as the
/// first.
#[test]
fn a_total_execution_runs_on_the_gpu_alone() {
    for withheld in [false, true] {
        on_device(withheld, |device, messages, generated| {
            let queue = device.new_command_queue().unwrap();
            for round in 0..2 {
                // One encoder.
                let setup = Setup::with_template(device, 2, flat_template(device));
                let before = recorded(device);
                let command_buffer = queue.command_buffer().unwrap();
                let mut encoder = command_buffer.compute_command_encoder().unwrap();
                setup.write(&mut encoder);
                setup.execute(&mut encoder);
                encoder.end_encoding().unwrap();
                complete(&command_buffer);
                let at = format!("withheld {withheld}, round {round}");
                assert_eq!(setup.output(), expected(), "one encoder, {at}");
                assert!(ran_flat(&command_buffer), "one encoder, {at}");
                ran_recording(device, generated, before, &at);
                // Two command buffers, the execution's committed first.
                let setup = Setup::with_template(device, 2, flat_template(device));
                let before = recorded(device);
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
                assert_eq!(setup.output(), expected(), "two command buffers, {at}");
                assert!(ran_flat(&executing), "two command buffers, {at}");
                ran_recording(device, generated, before, &at);
            }
            announced(device, messages, true);
        });
    }
}

/// An execution whose commands' windows no encoder made resident runs as one that made them
/// resident: the GPU-alone path's resident set holds every live buffer of its own, which the
/// preprocess resolves the records' buffer addresses in, as the host step resolves them, and
/// Metal's commands reach. Both recordings write what Metal computes, nothing announced.
#[test]
fn an_execution_reaches_buffers_no_encoder_named() {
    for withheld in [false, true] {
        on_device(withheld, |device, messages, generated| {
            let setup = Setup::with_template(device, 2, flat_template(device));
            let before = recorded(device);
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            setup.write(&mut encoder);
            setup.execute_naming_nothing(&mut encoder);
            encoder.end_encoding().unwrap();
            complete(&command_buffer);
            assert_eq!(setup.output(), expected(), "withheld {withheld}");
            assert!(ran_flat(&command_buffer), "withheld {withheld}");
            ran_recording(device, generated, before, &format!("withheld {withheld}"));
            announced(device, messages, true);
        });
    }
}

/// A state made with `supportIndirectCommandBuffers` alive at the commit whose kernel observes
/// its threadgroup partition (`icb_replay_template`) leaves the execution not total: it takes the
/// host step, announced once, and writes what Metal computes.
#[test]
fn a_state_that_observes_its_partition_keeps_the_host_step() {
    on_device(false, |device, messages, _| {
        let observing = state(device, "icb_replay_template", true);
        let setup = Setup::with_template(device, 2, flat_template(device));
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.write(&mut encoder);
        setup.execute(&mut encoder);
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(setup.output(), expected());
        assert!(!ran_flat(&command_buffer));
        drop(observing);
        announced(device, messages, false);
    });
}

/// A state made with `supportIndirectCommandBuffers` alive at the commit whose kernel fetches a
/// `[[stage_in]]` attribute (`ob3_stage_input` at a constant step, which reads no stage-in
/// region) leaves the execution to the host step, announced once, which checks each command's
/// fetch past 4 GiB (`runtime/fetch_reach.rs`); it writes what Metal computes.
#[test]
fn a_state_that_fetches_stage_in_keeps_the_host_step() {
    on_device(false, |device, messages, _| {
        let fetching = super::stage_in_strides::stage_in_pipeline(device, StepFunction::Constant);
        let setup = Setup::with_template(device, 2, flat_template(device));
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        setup.write(&mut encoder);
        setup.execute(&mut encoder);
        encoder.end_encoding().unwrap();
        complete(&command_buffer);
        assert_eq!(setup.output(), expected());
        assert!(!ran_flat(&command_buffer));
        drop(fetching);
        announced(device, messages, false);
    });
}

/// A write the buffer's descriptor does not allow (a kernel buffer past its bind count) sets the
/// header's sticky encoding fault: no command runs, and the execution report says why, on either
/// path.
#[test]
fn an_encoding_fault_runs_no_command_on_either_path() {
    for withheld in [false, true] {
        on_device(withheld, |device, messages, generated| {
            let setup = Setup::with_template(device, 1, flat_template(device));
            let before = recorded(device);
            let queue = device.new_command_queue().unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            setup.write(&mut encoder);
            encoder.end_encoding().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            setup.execute(&mut encoder);
            encoder.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer
                .wait_until_completed_for(super::WAIT)
                .unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            let report = command_buffer.execution_report().unwrap();
            assert!(report.replay_fault, "withheld {withheld}");
            assert_eq!(report.replay_faults, ReplayFaults::ENCODING);
            assert!(setup.output().iter().all(|&word| word == 0));
            ran_recording(device, generated, before, &format!("withheld {withheld}"));
            announced(device, messages, true);
        });
    }
}
