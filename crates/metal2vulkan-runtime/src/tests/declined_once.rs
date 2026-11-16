//! A variant a dispatch asks for in passing that the compile layer declines is asked of the
//! compile layer once a device. A kernel behind a threadgroup barrier observes its threadgroup
//! partition, so the compile layer refuses its flat variant
//! (`Unsupported{GpuEncodedIndirectCommands}`) and its indirect threadgroup dispatches run on its
//! folded variant. Every indirect dispatch plans anew (no kept plan of a direct dispatch serves it,
//! `compute.rs`'s `settle_live`), so each asks for the flat variant: a state's remembered refusal
//! answers its later dispatches (`ComputePipelineState::settle_flat`), and the device's memo of
//! compiled pipelines keeps the refusal under the variant's function key (`pipeline/memo.rs`), so
//! the first dispatch of each later state of the same function is answered with no translation.
//! Over [`STATES`] states of [`DISPATCHES`] dispatches each, the compile layer refuses one
//! translation and the pipeline creation log (`pipeline/creation_log.rs`,
//! `METAL2VULKAN_LOG_PIPELINES`) says `declined` once; every dispatch writes what Metal's does.
//!
//! A child process of this test binary ([`declined_child`]) runs with the log on, its lines and
//! the count of refused translations (`pipeline/sources.rs`'s `functions_refused`) on stderr; the
//! parent reads them.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). The kernel
//! is LLVM text written here (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, ResourceOptions, Size,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::process::Command;

/// The log's variable.
const LOG: &str = "METAL2VULKAN_LOG_PIPELINES";

/// What a line of the log starts with.
const LINE: &str = "metal2vulkan: pipeline creation: ";

/// What the child prints before the count of refused translations.
const REFUSED: &str = "declined-once: translations refused ";

/// The states of the one function.
const STATES: usize = 4;

/// Each state's indirect dispatches.
const DISPATCHES: usize = 4;

/// The call's threadgroup: one row.
const GROUP: u64 = 64;

/// A kernel `k` storing, after a threadgroup barrier, each thread's position in the grid at
/// `out[p]`.
const KERNEL: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @k(ptr addrspace(1) %out, i32 %p) {
  call void @air.wg.barrier(i32 1, i32 1)
  %i = zext i32 %p to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %p, ptr addrspace(1) %o, align 4
  ret void
}
declare void @air.wg.barrier(i32, i32)

!air.kernel = !{!0}
!air.version = !{!11}
!air.language_version = !{!12}
!0 = !{ptr @k, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"p"}
!11 = !{i32 2, i32 8, i32 0}
!12 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// A compute state of [`KERNEL`] on `device`, made from a library of its own.
fn state(device: &Device) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(KERNEL).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("k")
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Dispatch `state` `dispatches` times in one command buffer, each an indirect dispatch of one
/// threadgroup of [`GROUP`] threads read from arguments of its own, and check the threads' words.
fn dispatch(device: &Device, state: &ComputePipelineState, dispatches: usize) {
    let arguments: Vec<u8> = (0..3 * dispatches)
        .flat_map(|_| 1u32.to_le_bytes())
        .collect();
    let arguments = device
        .new_buffer_with_bytes(&arguments, ResourceOptions::shared())
        .unwrap();
    let out = device
        .new_buffer_with_bytes(&[0xff; 4 * GROUP as usize], ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    for at in 0..dispatches as u64 {
        encoder
            .dispatch_threadgroups_indirect(&arguments, 12 * at, Size::new(GROUP, 1, 1))
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's bytes, which the completed command buffer wrote.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 4 * GROUP as usize) };
    for (p, word) in bytes.as_chunks::<4>().0.iter().enumerate() {
        assert_eq!(u32::from_le_bytes(*word), p as u32, "thread {p}");
    }
}

/// [`STATES`] states of [`DISPATCHES`] indirect dispatches each ask the compile layer for the flat
/// variant once, and the log says so once (the module docs).
#[test]
fn a_declined_variant_is_asked_for_once_per_device() {
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "tests::declined_once::declined_child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(LOG, "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(output.status.success(), "the child failed: {stderr}");
    let declined = (stderr.lines())
        .filter(|line| line.starts_with(LINE) && line.contains("compute pipeline `k`"))
        .filter(|line| line.contains("declined") && line.contains("GpuEncodedIndirectCommands"))
        .count();
    let refused: u64 = (stderr.lines())
        .find_map(|line| line.strip_prefix(REFUSED))
        .unwrap_or_else(|| panic!("the child's count: {stderr}"))
        .parse()
        .unwrap();
    println!(
        "DECLINED-ONCE: {STATES} states of one function, {DISPATCHES} indirect dispatches each: \
         {refused} translation(s) refused, {declined} declined line(s)"
    );
    assert_eq!((refused, declined), (1, 1), "{stderr}");
}

/// The child of [`a_declined_variant_is_asked_for_once_per_device`], run as its own process with
/// the log on: [`STATES`] states of [`DISPATCHES`] indirect dispatches each, then the count of
/// translations the compile layer refused.
#[test]
#[ignore = "a child process of a_declined_variant_is_asked_for_once_per_device"]
fn declined_child() {
    let scratch = Scratch::new("declined-once");
    let device = device(&scratch.0);
    for _ in 0..STATES {
        let state = state(&device);
        dispatch(&device, &state, DISPATCHES);
    }
    let refused = crate::pipeline::functions_refused();
    eprintln!("{REFUSED}{refused}");
}
