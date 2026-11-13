//! A direct `dispatchThreads:` of a ragged grid of a kernel that observes its threadgroup
//! partition runs on the kernel's partial-threadgroups variant (`compute.rs`'s `partial_direct`,
//! `VariantKey::partial_threadgroups`): one `VkPipeline` of the call's threadgroup size for every
//! remainder, where the split makes its interior's and one per remainder's edge. Each grid's words
//! are the split's (the same kernel's state made to split,
//! `ComputePipelineState::split_ragged_grids`) and Metal's: the partial threadgroup's
//! `threads_per_threadgroup`, a neighbour's threadgroup word behind a threadgroup barrier, and,
//! where the kernel's simdgroups are the device's subgroups, its last simdgroup's sum over the
//! live lanes alone. Where they are not (lavapipe's 8-lane subgroups, which emulate them), that
//! kernel is refused the variant and splits.
//!
//! A kernel whose module holds `compile_memory::COUNTED_WORDS` SPIR-V words or more runs its whole
//! grids on the variant too, every thread live (`ComputePipelineState::whole_on_partial`): one
//! `VkPipeline` of the call's threadgroup size runs its whole grids of either dispatch form and its
//! ragged grid, with the plain pipeline's words and Metal's. A smaller kernel's whole grids take the
//! plain pipeline.
//!
//! Under `m2v_gpu`, a kernel heavy enough that its driver's compile is the cost: its ragged grids'
//! creation time on the variant and split, one run each.
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

/// The call's threadgroup: one row of two simdgroups.
const GROUP: u32 = 64;

/// The grids along `x`: remainders 36, 26, 13, 1 and 63 of [`GROUP`].
const GRIDS: [u32; 5] = [100, 90, 77, 65, 127];

/// The words the output holds: the widest grid's.
const WORDS: usize = 128;

/// `k`: `a[t] = 3t + 1; threadgroup barrier; out[pos] = n << 24 | a[(t + 1) mod n] << 12 |
/// simd_sum(t)`, `t` the thread's `thread_index_in_threadgroup` and `n` its threadgroup's
/// `threads_per_threadgroup`; without `simd` the sum's word is 0. `extra` is more of the body
/// before the store, which adds `%x` (0 where empty) to the word.
fn kernel(simd: bool, extra: &str) -> String {
    let (sum, declare) = match simd {
        true => (
            "  %s = call i32 @air.simd_sum.u.i32(i32 %t)\n",
            "declare i32 @air.simd_sum.u.i32(i32) #0\n",
        ),
        false => ("  %s = add i32 0, 0\n", ""),
    };
    let extra = match extra.is_empty() {
        true => "  %x = add i32 0, 0\n".to_owned(),
        false => extra.to_owned(),
    };
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@a = internal unnamed_addr addrspace(3) global [64 x i32] undef, align 4

declare float @llvm.fma.f32(float, float, float)

define void @k(ptr addrspace(1) noundef %out, i32 noundef %pos, i32 noundef %t, i32 noundef %n) {{
entry:
  %tz = zext i32 %t to i64
  %pa = getelementptr inbounds [64 x i32], ptr addrspace(3) @a, i64 0, i64 %tz
  %v0 = mul i32 %t, 3
  %v1 = add i32 %v0, 1
  store i32 %v1, ptr addrspace(3) %pa, align 4
  tail call void @air.wg.barrier(i32 2, i32 1) #0
  %t1 = add i32 %t, 1
  %u = urem i32 %t1, %n
  %uz = zext i32 %u to i64
  %pu = getelementptr inbounds [64 x i32], ptr addrspace(3) @a, i64 0, i64 %uz
  %got = load i32, ptr addrspace(3) %pu, align 4
{sum}{extra}  %hi = shl i32 %n, 24
  %mid = shl i32 %got, 12
  %hm = or i32 %hi, %mid
  %w0 = or i32 %hm, %s
  %w = add i32 %w0, %x
  %pz = zext i32 %pos to i64
  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %pz
  store i32 %w, ptr addrspace(1) %po, align 4
  ret void
}}

declare void @air.wg.barrier(i32, i32) #0
{declare}
attributes #0 = {{ convergent mustprogress nounwind willreturn }}

!air.kernel = !{{!0}}
!air.version = !{{!7}}
!air.language_version = !{{!8}}
!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5, !6}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"pos"}}
!5 = !{{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}}
!6 = !{{i32 3, !"air.threads_per_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"n"}}
!7 = !{{i32 2, i32 8, i32 0}}
!8 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// What Metal's `dispatchThreads` of `threads` threads in threadgroups of [`GROUP`] leaves for
/// [`kernel`] with no `extra`: a partial threadgroup's threads synchronize among themselves, and
/// its last simdgroup sums its live lanes alone.
fn metal(threads: u32, simd: bool) -> Vec<u32> {
    (0..threads)
        .map(|pos| {
            let n = (threads - pos / GROUP * GROUP).min(GROUP);
            let t = pos % GROUP;
            let got = ((t + 1) % n) * 3 + 1;
            let first = t - t % 32;
            let sum = match simd {
                true => (first..first + (n - first).min(32)).sum(),
                false => 0,
            };
            n << 24 | got << 12 | sum
        })
        .collect()
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
/// (`Direct`, `Cluster32`), which no emulated simdgroup does.
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

/// Dispatch `state` over `threads` threads in threadgroups of [`GROUP`] and wait: the
/// `VkPipeline`s the dispatch made and the words it wrote.
fn run(device: &Device, state: &ComputePipelineState, threads: u32) -> (u64, Vec<u32>) {
    run_as(device, state, threads, false)
}

/// [`run`], as a `dispatchThreadgroups:` of `threads / GROUP` threadgroups where `groups`.
fn run_as(
    device: &Device,
    state: &ComputePipelineState,
    threads: u32,
    groups: bool,
) -> (u64, Vec<u32>) {
    let out: Buffer = device
        .new_buffer_with_bytes(&[0xff; 4 * WORDS], ResourceOptions::shared())
        .unwrap();
    let made = device.work_counts().variant_pipelines_made;
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    let group = Size::new(u64::from(GROUP), 1, 1);
    let dispatched = match groups {
        true => encoder.dispatch_threadgroups(Size::new(u64::from(threads / GROUP), 1, 1), group),
        false => encoder.dispatch_threads(Size::new(u64::from(threads), 1, 1), group),
    };
    dispatched.unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let made = device.work_counts().variant_pipelines_made - made;
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's bytes, which the completed command buffer wrote.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 4 * WORDS) };
    let words = (bytes.as_chunks::<4>().0.iter())
        .map(|&word| u32::from_le_bytes(word))
        .collect();
    (made, words)
}

#[test]
fn ragged_grids_of_a_kernel_run_partial_threadgroups_on_one_pipeline() {
    let scratch = Scratch::new("partial-threadgroups");
    let device = device(&scratch.0);
    for simd in [false, true] {
        let text = kernel(simd, "");
        let partial = state(&device, &text);
        let split = state(&device, &text);
        split.split_ragged_grids();
        let (mut partial_made, mut split_made) = (0, 0);
        for threads in GRIDS {
            let want = metal(threads, simd);
            let (made, got) = run(&device, &partial, threads);
            partial_made += made;
            assert_eq!(
                &got[..want.len()],
                &want[..],
                "simd {simd}: {threads} partial"
            );
            let (made, got) = run(&device, &split, threads);
            split_made += made;
            assert_eq!(
                &got[..want.len()],
                &want[..],
                "simd {simd}: {threads} split"
            );
        }
        // A kernel whose simdgroups are emulated takes no partial-threadgroups variant.
        let runs = !simd || full_subgroups(&device, &text);
        println!(
            "PARTIAL-THREADGROUPS {}: simd {simd}: VkPipelines made over {} ragged grids: {} \
             partial, {} split; the variant runs: {runs}",
            device.name(),
            GRIDS.len(),
            partial_made,
            split_made,
        );
        // The split: its interior's and each remainder's edge; the variant: its one. Where the
        // variant does not run, both states split, and the device makes the split's `VkPipeline`s
        // once, for the state that dispatches first.
        let split = 1 + GRIDS.len() as u64;
        let want = match runs {
            true => (1, split),
            false => (split, 0),
        };
        assert_eq!((partial_made, split_made), want, "simd {simd}");
    }
}

/// A kernel heavy enough that its driver's compile is the cost of a pipeline: [`STEPS`] fused
/// multiply-adds of [`WINDOW`] values live at once, after the barrier.
#[cfg(m2v_gpu)]
const STEPS: usize = 40_000;

/// The values each step's operand is behind.
#[cfg(m2v_gpu)]
const WINDOW: usize = 400;

/// The fused multiply-adds of a large kernel's chain: enough that its module holds
/// `compile_memory::COUNTED_WORDS` SPIR-V words or more, which its test checks.
const LARGE_STEPS: usize = 8_000;

/// The heavy body: [`chain`] of [`STEPS`] steps, [`WINDOW`] values live at once.
#[cfg(m2v_gpu)]
fn heavy() -> String {
    chain(STEPS, WINDOW)
}

/// A body of `steps` fused multiply-adds, each step's operand `window` steps behind it: `%x` the
/// sign bit of the chain's last value, which is positive (every step's operands are), so the
/// words stay Metal's while no compiler can drop the chain.
fn chain(steps: usize, window: usize) -> String {
    use std::fmt::Write;
    let hex = |v: f32| format!("0x{:016X}", f64::from(v).to_bits());
    let mut body = String::from("  %f0 = uitofp i32 %t to float\n");
    for i in 0..steps {
        let c = hex(1.0f32 + (i as f32 + 1.0) * 2f32.powi(-22));
        let (x, y) = match i < window {
            true => ("%f0".to_owned(), hex((i % 1024) as f32 * 0.25)),
            false => (format!("%f{i}"), format!("%f{}", i + 1 - window)),
        };
        let a = match i < window {
            true => format!("{x}, float {c}, float {y}"),
            false => format!("{x}, float {y}, float {c}"),
        };
        writeln!(body, "  %f{} = call float @llvm.fma.f32(float {a})", i + 1).unwrap();
    }
    writeln!(body, "  %xb = bitcast float %f{steps} to i32").unwrap();
    body.push_str("  %x = lshr i32 %xb, 31\n");
    body
}

/// A large kernel's whole grids, `dispatchThreads:` of one and two threadgroups and a
/// `dispatchThreadgroups:` of two, run on its partial-threadgroups variant, every thread live, and
/// its ragged grid too: one `VkPipeline` of [`GROUP`], three whole dispatches on the variant
/// (`WorkCounts::whole_dispatches_on_partial_variants`), their words the plain pipeline's (the same
/// kernel's state made to split, `ComputePipelineState::split_ragged_grids`) and Metal's. The
/// smaller kernel's whole grids take the plain pipeline, a `VkPipeline` beside the variant's, which
/// its ragged grid takes.
#[test]
fn a_large_kernel_s_whole_grids_run_on_its_partial_variant() {
    let scratch = Scratch::new("partial-threadgroups-whole");
    let device = device(&scratch.0);
    // (threads, as a `dispatchThreadgroups:`).
    let dispatches = [(64, false), (128, false), (128, true), (100, false)];
    for large in [true, false] {
        let extra = match large {
            true => chain(LARGE_STEPS, 8),
            false => String::new(),
        };
        let text = kernel(false, &extra);
        let partial = state(&device, &text);
        assert_eq!(
            partial.whole_on_partial(false),
            large,
            "large {large}: the rule"
        );
        let plain = state(&device, &text);
        plain.split_ragged_grids();
        let counted = || device.work_counts().whole_dispatches_on_partial_variants;
        let before = counted();
        let mut made = 0;
        for (threads, groups) in dispatches {
            let want = metal(threads, false);
            let tag = format!("large {large}: {threads} threads, threadgroups {groups}");
            let (n, got) = run_as(&device, &partial, threads, groups);
            made += n;
            assert_eq!(&got[..want.len()], &want[..], "{tag}: the state");
            let (_, got) = run_as(&device, &plain, threads, groups);
            assert_eq!(&got[..want.len()], &want[..], "{tag}: the plain pipeline");
        }
        let whole = counted() - before;
        println!(
            "PARTIAL-THREADGROUPS-WHOLE {}: large {large}: VkPipelines made over {} dispatches: \
             {made}; whole dispatches on the variant: {whole}",
            device.name(),
            dispatches.len(),
        );
        let want = match large {
            true => (1, 3),
            false => (2, 0),
        };
        assert_eq!(
            (made, whole),
            want,
            "large {large}: made, whole on the variant"
        );
    }
}

#[test]
#[cfg(m2v_gpu)]
fn a_heavy_kernel_s_ragged_grids_pay_one_creation() {
    use std::time::Instant;
    let scratch = Scratch::new("partial-threadgroups-heavy");
    let device = device(&scratch.0);
    let text = kernel(false, &heavy());
    let mut took = Vec::new();
    for split in [false, true] {
        let state = state(&device, &text);
        if split {
            state.split_ragged_grids();
        }
        let start = Instant::now();
        let mut made = 0;
        for threads in GRIDS {
            let (n, got) = run(&device, &state, threads);
            made += n;
            let want = metal(threads, false);
            assert_eq!(&got[..want.len()], &want[..], "split {split}: {threads}");
        }
        took.push((made, start.elapsed()));
    }
    println!(
        "PARTIAL-THREADGROUPS-HEAVY {}: {} ragged grids of a kernel of {STEPS} steps: partial \
         {} VkPipelines in {:.2} s, split {} in {:.2} s",
        device.name(),
        GRIDS.len(),
        took[0].0,
        took[0].1.as_secs_f64(),
        took[1].0,
        took[1].1.as_secs_f64(),
    );
    assert_eq!(took[0].0, 1, "the variant's one VkPipeline");
    assert_eq!(took[1].0, 1 + GRIDS.len() as u64, "the split's");
}
