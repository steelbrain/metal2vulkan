//! The free heap a large making's compile leaves is handed back (`pipeline/compile_memory.rs`): a
//! kernel of [`STEPS`] fused multiply-adds with [`WINDOW`] values live at once, past the registers,
//! made as three `VkPipeline`s (its state's, for a grid of whole threadgroups, and a ragged grid's
//! interior and edge regions), each a large making. glibc keeps what the driver's compile frees in
//! the compiling thread's arena; handed back after each, the process's resident set stays within
//! [`BOUND_MIB`] of what it was before the first, and the device counts the trims.
//!
//! Measured on an RX 7900 XTX (RADV, Mesa 26.2.6) with 200,000 steps and nothing handed back: 105
//! MiB before, 3.8 GiB after the first making, 7.3 GiB after the third (one run).
//!
//! A device test of glibc's allocator: compiled only under `m2v_gpu` on Linux with the GNU
//! environment. The module is LLVM text written here (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::opencl_mode::words_buffer;
use super::spares_bench::process_memory;
use super::uniform_bind_offsets::idle;
use crate::{CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, Size};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::fmt::Write;

/// The kernel's fused multiply-adds.
const STEPS: usize = 150_000;

/// The values each step's operand is behind: so many live at once.
const WINDOW: usize = 600;

/// The most the resident set may grow over the makings, in MiB.
const BOUND_MIB: u64 = 1024;

/// `heavy(device float *out [[buffer(0)]], uint gid [[thread_position_in_grid]], uint index
/// [[thread_index_in_threadgroup]])`: `out[gid] = f(index)`, `f` the chain of [`STEPS`] steps, each
/// step's operand [`WINDOW`] steps behind it.
fn kernel() -> String {
    let mut body = String::new();
    body.push_str("%at = zext i32 %gid to i64\n");
    body.push_str("%p = getelementptr inbounds float, ptr addrspace(1) %out, i64 %at\n");
    body.push_str("%x0 = uitofp i32 %index to float\n");
    let hex = |v: f32| format!("0x{:016X}", f64::from(v).to_bits());
    for i in 0..STEPS {
        let c = hex(1.0f32 + (i as f32 + 1.0) * 2f32.powi(-22));
        let (x, y) = match i < WINDOW {
            true => ("%x0".to_owned(), hex((i % 1024) as f32 * 0.25)),
            false => (format!("%x{i}"), format!("%x{}", i + 1 - WINDOW)),
        };
        let a = match i < WINDOW {
            true => format!("{x}, float {c}, float {y}"),
            false => format!("{x}, float {y}, float {c}"),
        };
        writeln!(body, "%x{} = call float @llvm.fma.f32(float {a})", i + 1).unwrap();
    }
    writeln!(body, "store float %x{STEPS}, ptr addrspace(1) %p, align 4").unwrap();
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
declare float @llvm.fma.f32(float, float, float)\n\n\
define void @heavy(ptr addrspace(1) noundef %out, i32 noundef %gid, i32 noundef %index) {{\n\
entry:\n{body}ret void\n}}\n\n\
!llvm.module.flags = !{{!0, !1}}\n!air.kernel = !{{!2}}\n!air.version = !{{!8}}\n!air.language_version = !{{!9}}\n\
!0 = !{{i32 7, !\"air.max_device_buffers\", i32 31}}\n\
!1 = !{{i32 7, !\"air.max_threadgroup_buffers\", i32 31}}\n\
!2 = !{{ptr @heavy, !3, !4}}\n!3 = !{{}}\n!4 = !{{!5, !6, !7}}\n\
!5 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"out\"}}\n\
!6 = !{{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}}\n\
!7 = !{{i32 2, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"index\"}}\n\
!8 = !{{i32 2, i32 8, i32 0}}\n!9 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

fn state(device: &Device) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(&kernel()).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("heavy")
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// Dispatch `n` threads in threadgroups of 128 and wait.
fn dispatch(device: &Device, state: &ComputePipelineState, n: u32) {
    let out = words_buffer(device, &[0; 256]);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .dispatch_threads(Size::new(u64::from(n), 1, 1), Size::new(128, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The process's resident set, in MiB.
fn resident_mib() -> u64 {
    process_memory()["vm-rss"] / 1024
}

#[test]
fn a_large_makings_free_heap_is_handed_back() {
    let scratch = Scratch::new("heap-trimmed");
    let device = device(&scratch.0);
    let before = resident_mib();
    let trims = || device.work_counts().heap_trims;
    let state = state(&device);
    dispatch(&device, &state, 128);
    idle(&device);
    let first = resident_mib();
    dispatch(&device, &state, 129);
    idle(&device);
    let second = resident_mib();
    println!(
        "HEAP-TRIMMED {}: resident {before} MiB before, {first} after the first making, {second} \
         after the second; {} trims",
        device.name(),
        trims()
    );
    assert!(trims() > 0, "no large making handed its free heap back");
    assert!(
        second <= before + BOUND_MIB,
        "the resident set grew from {before} MiB to {second} over three makings"
    );
}
