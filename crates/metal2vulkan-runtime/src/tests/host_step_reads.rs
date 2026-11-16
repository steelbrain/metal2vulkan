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
//!   buffer; and there, from arguments in a no-copy buffer whose rebind before the command
//!   buffer's completion is refused (`RebindProblem::EncodedUses`), so the dispatch runs the
//!   arguments of the pages it was encoded over.
//! - **Execution.** An indirect command buffer's dispatch, encoded by the CPU, copied by a blit
//!   into another buffer that the same command buffer executes next, writes `i` at each of its 64
//!   threads' words (`scale_and_offset` of zero input and scale).
//!
//! - **Device pointers.** A direct dispatch of a kernel that loads a buffer's address from
//!   another buffer reads 256 words through it, and one that stores through it writes them
//!   (`useResource` makes the pointee resident): the translated kernel's own reads and writes by
//!   device address, with no indirect command.
//!
//! - **64-bit arithmetic.** A kernel's `mul_hi` of `ulong`s and `long`s, which the flat path's
//!   derivation of a dispatch computes too (`bind::replay`'s `checked_mul`), its product,
//!   division and remainder of `ulong`s, and its `mul_hi` of `uint`s and `int`s answer the exact
//!   values for small and large operands.
//!
//! Each test prints what the device announced, how many host steps it recorded and where its
//! buffers' memory came from (`HOST-STEP-READS` lines).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The dispatch's kernel is
//! `inert_dispatch`'s `fill`, LLVM text (`llvm-as`); the execution's function is the owned
//! fixture `examples/runtime_examples.ll` (`llvm-as`).

use super::blit_holds::{options, until, Scratch, WAIT};
use super::device_recovery_live_table::{
    dispatched, dispatches, encode_command, indirect_state, WORDS,
};
use super::inert_dispatch::state;
use super::opencl_mode::{words, words_buffer};
use super::rebind_behind_a_wait as rebind;
use crate::error::{invalid, Problem};
use crate::{
    Buffer, CommandBufferStatus, ComputePipelineState, Degradation, Device, Error, EventRef,
    MessageSink, PipelineOptions, RebindProblem, ResourceOptions, ResourceRef, Size, WorkCounts,
};
use metal2vulkan::device::{VkExtension, VkFeature};
use metal2vulkan::metal::ResourceUsage;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::sync::{Arc, Mutex};

/// Two kernels of buffers `a` (index 0, a device pointer) and `b` (index 1) and the thread's grid
/// position: `deref_load` stores `p[gid] + 1` at `b[gid]`, `p` the pointer `a` holds;
/// `deref_store` stores `gid + 1` at `p[gid]`.
const POINTER_KERNELS: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @deref_load(ptr addrspace(1) noundef %a, ptr addrspace(1) noundef %b, i32 noundef %gid) {\n\
entry:\n\
  %p = load ptr addrspace(1), ptr addrspace(1) %a, align 8\n\
  %gw = zext i32 %gid to i64\n\
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %gw\n\
  %x = load i32, ptr addrspace(1) %q, align 4\n\
  %v = add i32 %x, 1\n\
  %r = getelementptr inbounds i32, ptr addrspace(1) %b, i64 %gw\n\
  store i32 %v, ptr addrspace(1) %r, align 4\n\
  ret void\n\
}\n\n\
define void @deref_store(ptr addrspace(1) noundef %a, ptr addrspace(1) noundef %b, i32 noundef %gid) {\n\
entry:\n\
  %p = load ptr addrspace(1), ptr addrspace(1) %a, align 8\n\
  %gw = zext i32 %gid to i64\n\
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %gw\n\
  %v = add i32 %gid, 1\n\
  store i32 %v, ptr addrspace(1) %q, align 4\n\
  ret void\n\
}\n\n\
!llvm.module.flags = !{!0, !1}\n!air.kernel = !{!20, !21}\n!air.version = !{!8}\n!air.language_version = !{!9}\n\
!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n\
!1 = !{i32 7, !\"air.max_threadgroup_buffers\", i32 31}\n\
!3 = !{}\n!4 = !{!5, !6, !7}\n\
!5 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"ulong\", !\"air.arg_name\", !\"a\"}\n\
!6 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"b\"}\n\
!7 = !{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}\n\
!8 = !{i32 2, i32 8, i32 0}\n!9 = !{!\"Metal\", i32 4, i32 0, i32 0}\n\
!20 = !{ptr @deref_load, !3, !4}\n!21 = !{ptr @deref_store, !3, !4}\n";

/// The words the dispatch tests read: past the 256 the dispatch writes.
const N: usize = 1024;

/// A kernel of buffers `a` (index 0) and `b` (index 1) of `ulong`s and the thread's grid
/// position: from the pair `x = a[2 gid]`, `y = a[2 gid + 1]`, it stores at `b[6 gid]` on
/// `mul_hi(x, y)`, `x / y`, `x % y`, `mul_hi` of `x` and `y` as `long`s, the `mul_hi` of their
/// low words as `uint`s (low half) and as `int`s (high half), and `x * y` (the product's low
/// word).
const WIDE_KERNEL: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
declare i64 @air.mul_hi.u.i64(i64, i64)\ndeclare i64 @air.mul_hi.s.i64(i64, i64)\ndeclare i32 @air.mul_hi.u.i32(i32, i32)\ndeclare i32 @air.mul_hi.s.i32(i32, i32)\n\n\
define void @wide_math(ptr addrspace(1) noundef %a, ptr addrspace(1) noundef %b, i32 noundef %gid) {\n\
entry:\n\
  %g = zext i32 %gid to i64\n\
  %i = shl i64 %g, 1\n\
  %pa = getelementptr inbounds i64, ptr addrspace(1) %a, i64 %i\n\
  %x = load i64, ptr addrspace(1) %pa, align 8\n\
  %j = or i64 %i, 1\n\
  %pb = getelementptr inbounds i64, ptr addrspace(1) %a, i64 %j\n\
  %y = load i64, ptr addrspace(1) %pb, align 8\n\
  %hi = call i64 @air.mul_hi.u.i64(i64 %x, i64 %y)\n\
  %q = udiv i64 %x, %y\n\
  %r = urem i64 %x, %y\n\
  %shi = call i64 @air.mul_hi.s.i64(i64 %x, i64 %y)\n\
  %x32 = trunc i64 %x to i32\n\
  %y32 = trunc i64 %y to i32\n\
  %uhi32 = call i32 @air.mul_hi.u.i32(i32 %x32, i32 %y32)\n\
  %shi32 = call i32 @air.mul_hi.s.i32(i32 %x32, i32 %y32)\n\
  %uw = zext i32 %uhi32 to i64\n\
  %sw = zext i32 %shi32 to i64\n\
  %sw2 = shl i64 %sw, 32\n\
  %h32 = or i64 %uw, %sw2\n\
  %low = mul i64 %x, %y\n\
  %k = mul i64 %g, 6\n\
  %o0 = getelementptr inbounds i64, ptr addrspace(1) %b, i64 %k\n\
  store i64 %hi, ptr addrspace(1) %o0, align 8\n\
  %k1 = add i64 %k, 1\n\
  %o1 = getelementptr inbounds i64, ptr addrspace(1) %b, i64 %k1\n\
  store i64 %q, ptr addrspace(1) %o1, align 8\n\
  %k2 = add i64 %k, 2\n\
  %o2 = getelementptr inbounds i64, ptr addrspace(1) %b, i64 %k2\n\
  store i64 %r, ptr addrspace(1) %o2, align 8\n\
  %k3 = add i64 %k, 3\n\
  %o3 = getelementptr inbounds i64, ptr addrspace(1) %b, i64 %k3\n\
  store i64 %shi, ptr addrspace(1) %o3, align 8\n\
  %k4 = add i64 %k, 4\n\
  %o4 = getelementptr inbounds i64, ptr addrspace(1) %b, i64 %k4\n\
  store i64 %h32, ptr addrspace(1) %o4, align 8\n\
  %k5 = add i64 %k, 5\n\
  %o5 = getelementptr inbounds i64, ptr addrspace(1) %b, i64 %k5\n\
  store i64 %low, ptr addrspace(1) %o5, align 8\n\
  ret void\n\
}\n\n\
!llvm.module.flags = !{!0, !1}\n!air.kernel = !{!20}\n!air.version = !{!8}\n!air.language_version = !{!9}\n\
!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n\
!1 = !{i32 7, !\"air.max_threadgroup_buffers\", i32 31}\n\
!3 = !{}\n!4 = !{!5, !6, !7}\n\
!5 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"ulong\", !\"air.arg_name\", !\"a\"}\n\
!6 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"ulong\", !\"air.arg_name\", !\"b\"}\n\
!7 = !{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}\n\
!8 = !{i32 2, i32 8, i32 0}\n!9 = !{!\"Metal\", i32 4, i32 0, i32 0}\n\
!20 = !{ptr @wide_math, !3, !4}\n";

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
    eprintln!(
        "HOST-STEP-READS execution report: {:?}",
        command_buffer.execution_report()
    );
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

/// An indirect dispatch's host step resolves the buffers its encoder named as it records
/// (`Deferred::resolved_at_its_calls`), so the commit keeps their pages until the command buffer
/// completes, and a rebind of one meanwhile is refused (`Invalid`, `RebindProblem::EncodedUses`),
/// where Metal resolves the use as it submits the command buffer (a known gap,
/// `docs/FAST-PATHS.md`). On the device without device addresses, a dispatch of `[4, 1, 1]`
/// threadgroups from arguments in a no-copy buffer, behind a shared event only the host signals:
/// the rebind before the signal, over pages holding `[2, 1, 1]`, is refused and changes nothing;
/// the command buffer completes and the dispatch ran the arguments of the pages it was encoded
/// over, 256 threads; once it completed, the rebind is taken.
#[test]
fn a_rebind_while_an_indirect_dispatch_s_host_step_is_left_is_refused_until_completion() {
    // Declared before the device, so dropped after it.
    let pages = [rebind::Page::new(), rebind::Page::new()];
    let scratch = Scratch::new("gpu-reads-rebind");
    let (device, heard) = device(&scratch, &[VkFeature::ShaderInt64], &[]);
    let arguments = |groups: u32| -> Vec<u8> {
        [groups, 1, 1]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect()
    };
    pages[0].write(&arguments(4));
    pages[1].write(&arguments(2));
    // SAFETY: each page outlives every buffer over it (dropped after the device), and only the
    // buffer's commands use it while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            pages[0].memory(),
            rebind::BYTES as u64,
            ResourceOptions::shared(),
            None,
        )
    };
    let over = match made {
        Ok(buffer) => buffer,
        Err(Error::Unsupported(gap)) => {
            eprintln!("HOST-STEP-READS: the device imports no such pages: {gap:?}");
            return;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    let fill = state(&device, "fill");
    let out = words_buffer(&device, &[0; N]);
    let unused = words_buffer(&device, &[0]);
    // Its command buffer waits for an event only the host signals, below: the rebind comes
    // before its first submission.
    let gate = device.new_shared_event().unwrap();
    let before = device.work_counts();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    (command_buffer.encode_wait_for_event(EventRef::Shared(&gate), 1)).unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&fill).unwrap();
    encoder.set_buffer(Some(&unused), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    encoder
        .dispatch_threadgroups_indirect(&over, 0, Size::new(64, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    // SAFETY: as above.
    let refused = unsafe { over.rebind_bytes_no_copy(pages[1].memory(), None, None) };
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let after = device.work_counts();
    report("rebind refused, host step", &device, &heard, &before);
    println!(
        "HOST-STEP-READS rebind refused {}: {refused:?}; {:?} {:?}; host steps recorded {}",
        device.name(),
        command_buffer.status(),
        command_buffer.error(),
        after.host_steps_recorded - before.host_steps_recorded,
    );
    let encoded_uses = Problem::Rebind(RebindProblem::EncodedUses);
    assert_eq!(refused, Err(invalid("rebind_bytes_no_copy", encoded_uses)));
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None)
    );
    assert!(
        after.host_steps_recorded > before.host_steps_recorded,
        "the dispatch took its host step"
    );
    assert_eq!(words(&out), filled(), "the encoded pages' arguments");
    // The command buffer's hold on the pages goes as it completes.
    until("the rebind after the completion", || {
        // SAFETY: as above.
        unsafe { over.rebind_bytes_no_copy(pages[1].memory(), None, None) }.is_ok()
    });
    drop(over);
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

/// The state of kernel `name` of the LLVM text `kernels`.
fn text_state(device: &Device, kernels: &str, name: &str) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(kernels).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name(name)
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A direct dispatch of `name` over 256 threads, `a` holding `pointee`'s address and `b` bound,
/// `pointee` resident with `usage`.
fn through_pointer(
    device: &Device,
    name: &str,
    pointee: &Buffer,
    b: &Buffer,
    usage: ResourceUsage,
) {
    let state = text_state(device, POINTER_KERNELS, name);
    let address = pointee.gpu_address().0;
    let a = words_buffer(device, &[address as u32, (address >> 32) as u32]);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&a), 0, 0).unwrap();
    encoder.set_buffer(Some(b), 0, 1).unwrap();
    encoder
        .use_resource(ResourceRef::Buffer(pointee), usage)
        .unwrap();
    encoder
        .dispatch_threads(Size::new(256, 1, 1), Size::new(64, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None)
    );
}

/// A translated kernel's own load and store through a device pointer, in direct dispatches.
#[test]
fn a_kernel_reads_and_writes_through_a_device_pointer() {
    let scratch = Scratch::new("gpu-reads-pointer");
    let (device, heard) = device(&scratch, &[], &[]);
    let before = device.work_counts();
    let source: Vec<u32> = (0..N as u32).map(|i| 5 * i).collect();
    let pointee = words_buffer(&device, &source);
    let loaded = words_buffer(&device, &[0; N]);
    through_pointer(
        &device,
        "deref_load",
        &pointee,
        &loaded,
        ResourceUsage::READ,
    );
    let stored = words_buffer(&device, &[0; N]);
    let unused = words_buffer(&device, &[0]);
    through_pointer(
        &device,
        "deref_store",
        &stored,
        &unused,
        ResourceUsage::WRITE,
    );
    report("device pointer", &device, &heard, &before);
    let (loaded, stored) = (words(&loaded), words(&stored));
    eprintln!(
        "HOST-STEP-READS device pointer: loaded {:?}, stored {:?}",
        &loaded[..4],
        &stored[..4]
    );
    let expected_load: Vec<u32> = (0..N as u32)
        .map(|i| if i < 256 { 5 * i + 1 } else { 0 })
        .collect();
    assert_eq!(loaded, expected_load, "the load through the pointer");
    assert_eq!(stored, filled(), "the store through the pointer");
}

/// The kernel's `mul_hi`, division and remainder of `ulong`s answer the exact values.
#[test]
fn a_kernel_s_64_bit_multiply_high_and_division_are_exact() {
    let scratch = Scratch::new("gpu-reads-wide");
    let (device, heard) = device(&scratch, &[], &[]);
    let before = device.work_counts();
    let pairs: [(u64, u64); 8] = [
        (256, 1),
        (256, 64),
        (4, 64),
        (1 << 40, 1 << 40),
        (u64::MAX, u64::MAX),
        (u64::MAX, 3),
        (0x1234_5678_9abc_def0, 0xfedc_ba98_7654_3210),
        (1 << 32, 65535),
    ];
    let words_of = |longs: &[u64]| -> Vec<u32> {
        longs
            .iter()
            .flat_map(|&l| [l as u32, (l >> 32) as u32])
            .collect()
    };
    let inputs: Vec<u64> = pairs.iter().flat_map(|&(x, y)| [x, y]).collect();
    let a = words_buffer(&device, &words_of(&inputs));
    let b = words_buffer(&device, &[0; 2 * 6 * 8]);
    let state = text_state(&device, WIDE_KERNEL, "wide_math");
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&a), 0, 0).unwrap();
    encoder.set_buffer(Some(&b), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(8, 1, 1), Size::new(8, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    report("64-bit arithmetic", &device, &heard, &before);
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None)
    );
    let got = words(&b);
    let expected: Vec<u64> = pairs
        .iter()
        .flat_map(|&(x, y)| {
            let high = ((u128::from(x) * u128::from(y)) >> 64) as u64;
            let signed = ((i128::from(x as i64) * i128::from(y as i64)) >> 64) as u64;
            let (x32, y32) = (x as u32, y as u32);
            let uhi32 = ((u64::from(x32) * u64::from(y32)) >> 32) as u32;
            let shi32 = ((i64::from(x32 as i32) * i64::from(y32 as i32)) >> 32) as u32;
            [
                high,
                x / y,
                x % y,
                signed,
                u64::from(uhi32) | (u64::from(shi32) << 32),
                x.wrapping_mul(y),
            ]
        })
        .collect();
    let got: Vec<u64> = (got.as_chunks::<2>().0.iter())
        .map(|[lo, hi]| u64::from(*lo) | (u64::from(*hi) << 32))
        .collect();
    for (i, &(x, y)) in pairs.iter().enumerate() {
        eprintln!(
            "HOST-STEP-READS 64-bit arithmetic: {x:#x} {y:#x}: mul_hi {:#x} (exact {:#x}), \
             div {:#x} (exact {:#x}), rem {:#x} (exact {:#x}), signed mul_hi {:#x} (exact \
             {:#x}), 32-bit mul_hi signed:unsigned {:#x} (exact {:#x}), product {:#x} (exact \
             {:#x})",
            got[6 * i],
            expected[6 * i],
            got[6 * i + 1],
            expected[6 * i + 1],
            got[6 * i + 2],
            expected[6 * i + 2],
            got[6 * i + 3],
            expected[6 * i + 3],
            got[6 * i + 4],
            expected[6 * i + 4],
            got[6 * i + 5],
            expected[6 * i + 5],
        );
    }
    assert_eq!(got, expected);
}
