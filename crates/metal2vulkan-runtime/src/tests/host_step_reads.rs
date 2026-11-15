//! The GPU's reads of what the program and the work before them wrote, on the paths an indirect
//! dispatch and an execution take: the flat path's preprocess reads a dispatch's arguments by
//! device address (`indirect_dispatch.rs`), the host step reads the copy the work before it made,
//! and an execution of commands a blit copied runs the copied records (`indirect_execute.rs`,
//! `gpu_execute.rs`), each with the outputs its commands write.
//!
//! - **Indirect dispatch.** `dispatch_threadgroups_indirect` of `[4, 1, 1]` threadgroups of 64
//!   threads stores `gid + 1` at each of the 256 threads' words: from arguments in a `Shared`
//!   buffer the CPU wrote (`new_buffer_with_bytes`), on the device as it is made and on one that
//!   imports no host pointer (`VK_EXT_external_memory_host` withheld); from arguments a blit copied
//!   into a `Private` buffer; and on a device without device addresses (`shaderInt64` withheld, so
//!   the compile layer refuses the flat and folded variants and the dispatch takes its host step),
//!   from arguments the CPU wrote and from arguments a blit copied earlier in the same command
//!   buffer.
//! - **Execution.** An indirect command buffer's dispatch, encoded by the CPU, copied by a blit
//!   into another buffer that the same command buffer executes next, writes `i` at each of its 64
//!   threads' words (`scale_and_offset` of zero input and scale).
//!
//! Each test prints what the device announced, how many host steps it recorded and where its
//! buffers' memory came from (`HOST-STEP-READS` lines).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The dispatch's kernel is
//! `inert_dispatch`'s `fill`, LLVM text (`llvm-as`); the execution's function is the owned
//! fixture `examples/runtime_examples.ll` (`llvm-as`).

use super::blit_holds::{options, Scratch, WAIT};
use super::device_recovery_live_table::{
    dispatched, dispatches, encode_command, indirect_state, WORDS,
};
use super::inert_dispatch::state;
use super::opencl_mode::{words, words_buffer};
use crate::{
    Buffer, CommandBufferStatus, Degradation, Device, MessageSink, ResourceOptions, ResourceRef,
    Size, WorkCounts,
};
use metal2vulkan::device::{VkExtension, VkFeature};
use metal2vulkan::metal::ResourceUsage;
use std::sync::{Arc, Mutex};

/// The words the dispatch tests read: past the 256 the dispatch writes.
const N: usize = 1024;

/// The degrade notices a device announced, each kind with its reason.
type Heard = Arc<Mutex<Vec<(Degradation, String)>>>;

/// A device of the cfg's kind, `features` and `extensions` withheld, its notices into the
/// answer's list.
fn device(
    scratch: &Scratch,
    features: &[VkFeature],
    extensions: &[VkExtension],
) -> (Device, Heard) {
    let heard: Heard = Arc::default();
    let hearing = Arc::clone(&heard);
    let mut options = options(&scratch.0);
    options.messages = Some(MessageSink::new(move |message| {
        if let Some(notice) = &message.degradation {
            let heard = (notice.degradation, notice.reason.clone());
            hearing.lock().unwrap().push(heard);
        }
    }));
    options.withheld_features.extend(features.iter().copied());
    options
        .withheld_extensions
        .extend(extensions.iter().copied());
    (Device::system_default_with(&options).unwrap(), heard)
}

/// Print what `device` announced, the host steps it recorded since `before` and where its
/// buffers' memory came from, under `what`.
fn report(what: &str, device: &Device, heard: &Heard, before: &WorkCounts) {
    let counts = device.work_counts();
    let steps = counts.host_steps_recorded - before.host_steps_recorded;
    eprintln!(
        "HOST-STEP-READS {what}: {steps} host steps recorded; device addresses {}; page-buffer \
         blocks {} made, {} plain; pointer imports {}",
        device.shared().device_address,
        counts.page_buffer_blocks_made,
        counts.page_buffer_blocks_plain,
        counts.host_pointer_imports_made,
    );
    for (kind, reason) in heard.lock().unwrap().iter() {
        eprintln!("HOST-STEP-READS {what}: announced {kind:?}: {reason}");
    }
}

/// `gid + 1` for each of the 256 threads of `[4, 1, 1]` threadgroups of 64, zero past them.
fn filled() -> Vec<u32> {
    (0..N as u32)
        .map(|i| if i < 256 { i + 1 } else { 0 })
        .collect()
}

/// `fill`'s indirect dispatch on `device` from the arguments in `arguments`, copied there by a
/// blit from `copied_from` first where it is given, in one command buffer: the words it stored.
fn dispatched_words(device: &Device, arguments: &Buffer, copied_from: Option<&Buffer>) -> Vec<u32> {
    let fill = state(device, "fill");
    let out = words_buffer(device, &[0; N]);
    let unused = words_buffer(device, &[0]);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    if let Some(source) = copied_from {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(source, 0, arguments, 0, 12)
            .unwrap();
        blit.end_encoding().unwrap();
    }
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&fill).unwrap();
    encoder.set_buffer(Some(&unused), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    encoder
        .dispatch_threadgroups_indirect(arguments, 0, Size::new(64, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None)
    );
    words(&out)
}

/// The dispatch from a `Shared` buffer the CPU wrote, on the device as it is made: the
/// consumer's case.
#[test]
fn an_indirect_dispatch_runs_the_arguments_the_cpu_wrote() {
    let scratch = Scratch::new("gpu-reads-shared");
    let (device, heard) = device(&scratch, &[], &[]);
    let before = device.work_counts();
    let written = words_buffer(&device, &[4, 1, 1]);
    let out = dispatched_words(&device, &written, None);
    report("shared arguments", &device, &heard, &before);
    assert_eq!(out, filled());
}

/// The dispatch from a `Shared` buffer the CPU wrote, on a device that imports no host pointer:
/// every buffer's memory is the driver's own.
#[test]
fn an_indirect_dispatch_runs_the_arguments_the_cpu_wrote_without_pointer_imports() {
    let scratch = Scratch::new("gpu-reads-unimported");
    let (device, heard) = device(&scratch, &[], &[VkExtension::ExtExternalMemoryHost]);
    let before = device.work_counts();
    let written = words_buffer(&device, &[4, 1, 1]);
    let out = dispatched_words(&device, &written, None);
    report(
        "shared arguments, no pointer imports",
        &device,
        &heard,
        &before,
    );
    assert_eq!(out, filled());
}

/// The dispatch from a `Private` buffer a blit filled in an earlier command buffer, on the device
/// as it is made.
#[test]
fn an_indirect_dispatch_runs_the_arguments_a_blit_wrote_into_private_memory() {
    let scratch = Scratch::new("gpu-reads-private");
    let (device, heard) = device(&scratch, &[], &[]);
    let before = device.work_counts();
    let written = words_buffer(&device, &[4, 1, 1]);
    let private = device.new_buffer(16, ResourceOptions::private()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&written, 0, &private, 0, 12)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let out = dispatched_words(&device, &private, None);
    report("private arguments", &device, &heard, &before);
    assert_eq!(out, filled());
}

/// The host step of an indirect dispatch runs the arguments the CPU wrote, and those a blit
/// copied before it in the same command buffer.
#[test]
fn an_indirect_dispatch_s_host_step_runs_its_arguments() {
    let scratch = Scratch::new("gpu-reads-host-step");
    let (device, heard) = device(&scratch, &[VkFeature::ShaderInt64], &[]);
    let before = device.work_counts();
    let written = words_buffer(&device, &[4, 1, 1]);
    let from_cpu = dispatched_words(&device, &written, None);
    report("shared arguments, host step", &device, &heard, &before);
    let steps = device.work_counts().host_steps_recorded - before.host_steps_recorded;
    let before = device.work_counts();
    let copied = words_buffer(&device, &[0, 0, 0]);
    let from_blit = dispatched_words(&device, &copied, Some(&written));
    report("copied arguments, host step", &device, &heard, &before);
    assert!(steps > 0, "the dispatch took its host step");
    assert_eq!(from_cpu, filled(), "the CPU's arguments");
    assert_eq!(from_blit, filled(), "the copied arguments");
}

/// An execution of a dispatch a blit copied into the buffer it executes, in the same command
/// buffer, runs the copied command.
#[test]
fn an_execution_of_copied_commands_runs_them() {
    let scratch = Scratch::new("gpu-reads-icb");
    let (device, heard) = device(&scratch, &[], &[]);
    let before = device.work_counts();
    let compute = indirect_state(&device);
    let (encoded, copy) = (dispatches(&device), dispatches(&device));
    let input = words_buffer(&device, &[0; WORDS as usize]);
    let scale = words_buffer(&device, &[0; WORDS as usize]);
    let output = words_buffer(&device, &[u32::MAX; WORDS as usize]);
    encode_command(&encoded, &compute, [&input, &output, &scale]);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_indirect_command_buffer(&encoded, 0..1, &copy, 0)
        .unwrap();
    blit.end_encoding().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    for (buffer, usage) in [
        (&input, ResourceUsage::READ),
        (&output, ResourceUsage::WRITE),
        (&scale, ResourceUsage::READ),
    ] {
        (encoder.use_resource(ResourceRef::Buffer(buffer), usage)).unwrap();
    }
    encoder.execute_commands_in_buffer(&copy, 0..1).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    report("copied commands", &device, &heard, &before);
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None)
    );
    let floats: Vec<f32> = words(&output).into_iter().map(f32::from_bits).collect();
    assert_eq!(floats, dispatched());
}
