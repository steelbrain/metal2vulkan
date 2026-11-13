//! A direct `dispatchThreads:` of a ragged one-row grid in threadgroups of one row, of a kernel
//! that observes its threadgroup partition through its simdgroups alone, runs on the flat variant
//! of the call's threadgroups (`compute.rs`'s `flat_direct`, `VariantKey::call_threadgroups`): one
//! `VkPipeline`, where the split makes one for its interior and one for its edge, every thread's
//! lane and simdgroup what Metal's threadgroups give it, the last simdgroup's lanes past the grid
//! left out of its reduction as Metal's partial simdgroup leaves them. A grid of two rows, and a
//! kernel with a threadgroup barrier, run on the kernel's partial-threadgroups variant instead
//! (`compute.rs`'s `partial_direct`): one `VkPipeline` too. A kernel with a simdgroup barrier and
//! a reduction runs flat where its simdgroups are the device's subgroups (its tier asks for full
//! subgroups) and splits where they are not (lavapipe's 8-lane subgroups, which emulate them, and
//! which no partial-threadgroups variant runs).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernels are LLVM text written here
//! (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    Buffer, CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, ResourceOptions,
    Size,
};
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, FunctionDescriptor, LoweringInputs, StageSource,
};

/// The grid's threads along `x`: one threadgroup of [`GROUP`] and a ragged one of 36.
const THREADS: u64 = 100;
/// The call's threadgroup: one row of two simdgroups.
const GROUP: u64 = 64;

/// A kernel `k` storing, for each thread at `out[p]` (`p` its position in the grid), its index in
/// its simdgroup (`l`), or what `body` computes from it into `%v`, after `before`, with the
/// module's `declarations`.
fn kernel(before: &str, body: &str, declarations: &str) -> String {
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @k(ptr addrspace(1) %out, i32 %p, i32 %l) {{
{before}  {body}
  %i = zext i32 %p to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}}
{declarations}
!air.kernel = !{{!0}}
!air.version = !{{!11}}
!air.language_version = !{{!12}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"p"}}
!5 = !{{i32 2, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"l"}}
!11 = !{{i32 2, i32 8, i32 0}}
!12 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// Each thread's lane.
fn lanes() -> String {
    kernel("", "%v = add i32 %l, 0", "")
}

/// Each thread's lane, after a threadgroup barrier.
fn threadgroup_barrier() -> String {
    kernel(
        "  call void @air.wg.barrier(i32 1, i32 1)\n",
        "%v = add i32 %l, 0",
        "declare void @air.wg.barrier(i32, i32)\n",
    )
}

/// The sum of the lanes of each thread's simdgroup, after a simdgroup barrier.
fn simdgroup_sum() -> String {
    kernel(
        "  call void @air.simdgroup.barrier(i32 1, i32 4)\n",
        "%v = call i32 @air.simd_sum.u.i32(i32 %l)",
        "declare void @air.simdgroup.barrier(i32, i32)\ndeclare i32 @air.simd_sum.u.i32(i32)\n",
    )
}

/// What Metal's threadgroups of [`GROUP`] give thread `p` of a grid of [`THREADS`] along `x`: its
/// lane, and the sum of the lanes of its simdgroup, the last one's lanes those of the threads left.
fn metal(p: u64) -> (u32, u32) {
    let index = p % GROUP;
    let first = p - index % 32;
    let live = (THREADS - first).min(32);
    ((index % 32) as u32, (0..live).sum::<u64>() as u32)
}

/// The compute state of kernel `text` on `device`.
fn state(device: &Device, text: &str) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(text).unwrap();
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

/// Whether kernel `text`'s simdgroups are `device`'s subgroups: its tier asks for full subgroups
/// (`Direct`, `Cluster32`), so its simdgroup barrier is a subgroup one.
fn full_subgroups(device: &Device, text: &str) -> bool {
    let air = metal2vulkan_devtools::llvm_assemble_text(text).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = FunctionDescriptor::new(library.function("k").unwrap());
    device
        .compile_device()
        .compile_function(
            &function,
            &LoweringInputs::new(),
            &metal2vulkan::Budget::standard(),
        )
        .unwrap()
        .requires_full_subgroups()
}

/// Dispatch `state` over `rows` rows of [`THREADS`] in threadgroups of [`GROUP`], and the
/// `VkPipeline`s the dispatch made and the first row's words it wrote.
fn run(device: &Device, state: &ComputePipelineState, rows: u64) -> (u64, Vec<u32>) {
    let out: Buffer = device
        .new_buffer_with_bytes(&[0xff; 4 * THREADS as usize], ResourceOptions::shared())
        .unwrap();
    let made = device.work_counts().variant_pipelines_made;
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .dispatch_threads(Size::new(THREADS, rows, 1), Size::new(GROUP, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let made = device.work_counts().variant_pipelines_made - made;
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's bytes, which the completed command buffer wrote.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 4 * THREADS as usize) };
    let words = (bytes.as_chunks::<4>().0.iter())
        .map(|&word| u32::from_le_bytes(word))
        .collect();
    (made, words)
}

#[test]
fn a_ragged_row_of_a_simdgroup_kernel_runs_flat_in_the_call_s_threadgroups() {
    let scratch = Scratch::new("flat-simdgroups");
    let device = device(&scratch.0);
    let lane = |p: u64| metal(p).0;
    let sum = |p: u64| metal(p).1;
    let cases: [(&str, String, u64, fn(u64) -> u32); 4] = [
        ("lanes", lanes(), 1, lane),
        ("lanes over two rows", lanes(), 2, lane),
        ("threadgroup barrier", threadgroup_barrier(), 1, lane),
        ("simdgroup sum", simdgroup_sum(), 1, sum),
    ];
    let mut made = Vec::new();
    for (name, text, rows, expected) in &cases {
        let state = state(&device, text);
        let (pipelines, words) = run(&device, &state, *rows);
        for (p, &word) in words.iter().enumerate() {
            assert_eq!(word, expected(p as u64), "{name}: thread {p}");
        }
        made.push(pipelines);
    }
    let flat_sum = full_subgroups(&device, &simdgroup_sum());
    println!(
        "FLAT-SIMDGROUPS {}: VkPipelines made {made:?} for {:?}; the sum's simdgroups are \
         subgroups: {flat_sum}",
        device.name(),
        cases.each_ref().map(|case| case.0),
    );
    // One flat `VkPipeline`; the partial-threadgroups variant's one, twice; one where the
    // simdgroups are subgroups, else the split's interior and edge.
    let sum_made = match flat_sum {
        true => 1,
        false => 2,
    };
    assert_eq!(made, [1, 1, 1, sum_made]);
}
