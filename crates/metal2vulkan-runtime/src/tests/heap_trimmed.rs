//! The free heap a large making's compile leaves is handed back (`pipeline/compile_memory.rs`):
//! a kernel of fused multiply-adds, each step's operand some values behind it ([`WIDE`]: a short
//! chain of many values live), made as two `VkPipeline`s of its partial-threadgroups variant, which runs a large kernel's whole and ragged grids alike
//! (`compute::partial_direct`), one per threadgroup size, each a large making. glibc keeps what
//! the driver's compile frees in the compiling thread's arena; handed back after each, the
//! process's resident set stays within [`BOUND_MIB`] of what it was before the first, and the
//! device counts the trims.
//!
//! Measured on an RX 7900 XTX (RADV, Mesa 26.2.6) with 200,000 steps, 600 live, and nothing handed
//! back: 105 MiB before, 3.8 GiB after the first making, 7.3 GiB after the third (one run).
//!
//! The sizes: some making must leave glibc more than the trim's floor free
//! (`compile_memory::FREE_FLOOR`, 256 MiB, 268 MB) on every device, and no driver's compile frees
//! alike. Each shape made twice in a process of its own, on an RTX 3090 (NVIDIA 620.70.06) and a
//! Radeon 780M (RADV, Mesa 26.2.6), the free heap after the first making (`mallinfo2`, what the
//! translation freed included) and the two makings' times:
//!
//! | steps, live | RTX 3090 | Radeon 780M |
//! |---|---|---|
//! | 40,000, 600 | 73 MB; 8.9 s, 8.5 s | 739 MB; 2.8 s, 1.8 s |
//! | 80,000, 600 | 189 MB; 28.6 s, 28.4 s | 1,506 MB; 6.1 s, 4.0 s |
//! | 160,000, 256 | 227 MB; 108.6 s, 94.8 s | 195 MB; 3.4 s, 1.8 s |
//! | 240,000, 192 | 303 MB; 109.1 s, 98.6 s | 214 MB; 5.1 s, 2.8 s |
//! | 320,000, 128 | 336 MB; 129.5 s, 124.0 s | 275 MB; 7.0 s, 3.7 s |
//! | 320,000, 64 | 334 MB; 60.5 s, 18.4 s | 277 MB; 7.0 s, 3.7 s |
//!
//! NVIDIA's compile frees about 0.7 KB a step whatever the values live, fastest with few; RADV's
//! leaves 200 to 280 MB with up to 256 live, at the floor or under it, and far more with 600.
//! Every shape NVIDIA's compile frees past the floor is past the stage bound no device is handed
//! (`Feature::StageSize`, an entry weight of 99,109; 320,000 steps weigh 320,038), and under it
//! NVIDIA's frees 189 MB at most measured (80,000 steps, 600 live). So the test makes the wide
//! chain alone, which RADV's and ANV's compiles free past the floor, and asks for a trim where its
//! making left more than the floor free. Each shape made once in a process of its own on the Iris
//! Xe of RPL-P (ANV, Mesa 26.2.6), the free heap after the making, its time and the process's peak
//! resident set (one run, the host shared):
//!
//! | steps, live | Iris Xe |
//! |---|---|
//! | 20,000, 600 | 794 MB; 145.7 s; 3.0 GiB |
//! | 40,000, 600 | not made in 240 s, where it was stopped (3.7 GiB) |
//! | 80,000, 600 | not made in 240 s, where it was stopped (5.4 GiB) |
//! | 96,000, 64 | 289 MB; 160.9 s; 7.2 GiB |
//!
//! So the wide chain takes 20,000 steps, whose compile alone frees past the floor on both: 794 MB
//! on the Iris Xe, three times the floor, and 346 MB on the Radeon 780M (made the same way, 1.6 s),
//! 1.3 times. A making may take minutes on the Iris Xe or a loaded host; each command buffer waits
//! up to [`MAKING_WAIT`].
//!
//! A device test of glibc's allocator: compiled only under `m2v_gpu` on Linux with the GNU
//! environment. The modules are LLVM text written here (`llvm-as`).

use super::blit_holds::{device, Scratch};
use super::opencl_mode::words_buffer;
use super::spares_bench::process_memory;
use super::uniform_bind_offsets::idle;
use crate::pipeline::compile_memory::heap_bytes;
use crate::pipeline::compile_memory::FREE_FLOOR;
use crate::{CommandBufferStatus, ComputePipelineState, Device, Error, PipelineOptions, Size};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::fmt::Write;
use std::time::{Duration, Instant};

/// The kernel's fused multiply-adds and the values each step's operand is behind (so many live
/// at once): a short chain of many, whose compile RADV's and ANV's drivers free past the trim's
/// floor (the module docs).
const WIDE: (usize, usize) = (20_000, 600);

/// The longest a command buffer of a large making waits.
const MAKING_WAIT: Duration = Duration::from_secs(300);

/// The most the resident set may grow over the makings, in MiB.
const BOUND_MIB: u64 = 1024;

/// `heavy(device float *out [[buffer(0)]], uint gid [[thread_position_in_grid]], uint index
/// [[thread_index_in_threadgroup]])`: `out[gid] = f(index)`, `f` the chain of `steps` steps, each
/// step's operand `window` steps behind it.
fn kernel(steps: usize, window: usize) -> String {
    let mut body = String::new();
    body.push_str("%at = zext i32 %gid to i64\n");
    body.push_str("%p = getelementptr inbounds float, ptr addrspace(1) %out, i64 %at\n");
    body.push_str("%x0 = uitofp i32 %index to float\n");
    let hex = |v: f32| format!("0x{:016X}", f64::from(v).to_bits());
    for i in 0..steps {
        let c = hex(1.0f32 + (i as f32 + 1.0) * 2f32.powi(-22));
        let (x, y) = match i < window {
            true => ("%x0".to_owned(), hex((i % 1024) as f32 * 0.25)),
            false => (format!("%x{i}"), format!("%x{}", i + 1 - window)),
        };
        let a = match i < window {
            true => format!("{x}, float {c}, float {y}"),
            false => format!("{x}, float {y}, float {c}"),
        };
        writeln!(body, "%x{} = call float @llvm.fma.f32(float {a})", i + 1).unwrap();
    }
    writeln!(body, "store float %x{steps}, ptr addrspace(1) %p, align 4").unwrap();
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

/// The state of `heavy` of `steps` steps, `window` live.
fn state(device: &Device, (steps, window): (usize, usize)) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(&kernel(steps, window)).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("heavy")
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap_or_else(|e| panic!("{steps} steps, {window} live: {e:?}"))
}

/// Dispatch `n` threads in threadgroups of `group` and wait.
fn dispatch(device: &Device, state: &ComputePipelineState, n: u32, group: u32) {
    let out = words_buffer(device, &[0; 256]);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .dispatch_threads(
            Size::new(u64::from(n), 1, 1),
            Size::new(u64::from(group), 1, 1),
        )
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(MAKING_WAIT)
        .unwrap();
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
    let (steps, window) = WIDE;
    let made = Instant::now();
    let state = state(&device, WIDE);
    dispatch(&device, &state, 128, 128);
    // What the first making left glibc free (`compile_memory::FREE_FLOOR`'s measure).
    let free = heap_bytes().map_or(0, |(_, free)| free);
    idle(&device);
    let first = resident_mib();
    dispatch(&device, &state, 129, 64);
    idle(&device);
    println!(
        "HEAP-TRIMMED {}: resident {before} MiB before; {steps} steps, {window} live: {free} bytes \
         free and {first} MiB resident after the first making, {} MiB after the second, {} \
         trims, {:.1} s",
        device.name(),
        resident_mib(),
        trims(),
        made.elapsed().as_secs_f64()
    );
    let after = resident_mib();
    // Only a making that left more than the floor free hands it back (the module docs).
    assert!(
        free <= FREE_FLOOR || trims() > 0,
        "a making left {free} bytes free, and none was handed back"
    );
    assert!(
        after <= before + BOUND_MIB,
        "the resident set grew from {before} MiB to {after} over the makings"
    );
}
