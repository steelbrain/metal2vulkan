//! A kernel's unit-axes variant (`pipeline/unit_axes.rs`) on the device the cfg selects.
//!
//! The kernel sums its simdgroup in a loop whose trip count reads `thread_position_in_grid.y`
//! (`(gid.y & 3) + 1` iterations of `simd_sum(t + i)`, `t` its thread index in its threadgroup),
//! dispatched in threadgroups of 16 threads: not whole 32-lane simdgroups, so its simdgroups are
//! emulated on every device, and walked predicated, since the loop's trip count is per thread to
//! the translator. In threadgroups of 8 x 1 x 2, one thread along `y`, every thread of a
//! threadgroup takes the same trip count, which the variant compiles in.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    Buffer, CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, ResourceOptions,
    Size, WorkCounts,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use std::time::Instant;

/// The threads of a threadgroup, and the threadgroups of a dispatch, along `y`.
const THREADS: u32 = 16;
const ROWS: u64 = 4;
/// The words the output holds: a row of [`THREADS`] per grid position along `y`, at most two
/// threads along `y` a threadgroup.
const WORDS: usize = (THREADS as usize) * 2 * ROWS as usize;
/// The most dispatches it takes the variant to be made and taken: the one that begins its
/// translation, the one that begins its region's making, the one that takes it, and one more.
const TRIES: usize = 4;

/// `kernel void unit_axes_y(device uint *out [[buffer(0)]], uint3 gid [[thread_position_in_grid]],
/// uint t [[thread_index_in_threadgroup]])` (the module docs), storing at `gid.y * 16 + t`.
const KERNEL: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @unit_axes_y(ptr addrspace(1) %out, <3 x i32> %gid, i32 %t) #0 {
entry:
  %g = extractelement <3 x i32> %gid, i64 1
  %k = and i32 %g, 3
  %trip = add i32 %k, 1
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i1, %loop ]
  %acc = phi i32 [ 0, %entry ], [ %acc1, %loop ]
  %v = add i32 %t, %i
  %s = tail call i32 @air.simd_sum.u.i32(i32 %v)
  %acc1 = add i32 %acc, %s
  %i1 = add i32 %i, 1
  %more = icmp ult i32 %i1, %trip
  br i1 %more, label %loop, label %done
done:
  %row = mul i32 %g, 16
  %at32 = add i32 %row, %t
  %at = zext i32 %at32 to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %at
  store i32 %acc1, ptr addrspace(1) %p, align 4
  ret void
}

declare i32 @air.simd_sum.u.i32(i32) #1

attributes #0 = { convergent mustprogress nounwind willreturn }
attributes #1 = { convergent mustprogress nounwind willreturn memory(none) }

!air.kernel = !{!0}
!air.version = !{!6}
!air.language_version = !{!7}

!0 = !{ptr @unit_axes_y, !1, !2}
!1 = !{}
!2 = !{!3, !4, !5}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!4 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
!5 = !{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!6 = !{i32 2, i32 8, i32 0}
!7 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The kernel's compute state on `device`.
fn state(device: &Device) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(KERNEL).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("unit_axes_y")
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The output's words after a dispatch of [`ROWS`] threadgroups of `group` along `y`, the
/// output filled with ones before it.
fn run(device: &Device, state: &ComputePipelineState, group: [u64; 3]) -> Vec<u32> {
    let out = device
        .new_buffer_with_bytes(&[0xff; 4 * WORDS], ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder
        .dispatch_threadgroups(
            Size::new(1, ROWS, 1),
            Size::new(group[0], group[1], group[2]),
        )
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

fn words(buffer: &Buffer) -> Vec<u32> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's bytes, which a completed command buffer wrote.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), 4 * WORDS) };
    (bytes.as_chunks::<4>().0.iter())
        .map(|&word| u32::from_le_bytes(word))
        .collect()
}

/// Metal's words for [`ROWS`] threadgroups of `group` along `y`: thread `t` of grid row `y`
/// runs `(y & 3) + 1` iterations, each adding the sum of `t' + i` over the threads `t'` of its
/// threadgroup (one simdgroup) still in the loop, stored at `16 y + t`; every other word ones.
fn expected(group: [u64; 3]) -> Vec<u32> {
    let [x, y, _] = group.map(|n| n as u32);
    let trip = |w: u32, t: u32| ((w * y + (t / x) % y) & 3) + 1;
    let mut words = vec![u32::MAX; WORDS];
    for w in 0..ROWS as u32 {
        for t in 0..THREADS {
            let acc = (0..trip(w, t))
                .map(|i| {
                    (0..THREADS)
                        .filter(|&u| trip(w, u) > i)
                        .map(|u| u + i)
                        .sum::<u32>()
                })
                .sum();
            let row = w * y + (t / x) % y;
            words[(row * THREADS + t) as usize] = acc;
        }
    }
    words
}

/// `device`'s work counts once its pipeline pool holds no piece, waiting at most [`WAIT`].
fn settled(device: &Device) -> WorkCounts {
    let until = Instant::now() + WAIT;
    loop {
        let counts = device.work_counts();
        if counts.pipeline_pool_busy == 0 {
            return counts;
        }
        assert!(Instant::now() < until, "the pipeline pool still busy");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// A dispatch of threadgroups with no unit axis (4 x 2 x 2) never begins the kernel's unit-axes
/// variant. The first dispatch of threadgroups of one thread along `y` (8 x 1 x 2) settles on the
/// plain variant and begins the variant on the pool, its dispatch waiting for none of it; once
/// its translation and its region's `VkPipeline` are made a later dispatch settles on it, and
/// every one of them leaves Metal's words, the plain variant's.
#[test]
fn a_unit_axis_dispatch_takes_its_variant_once_made() {
    let scratch = Scratch::new("unit-axes");
    let device = device(&scratch.0);
    let state = state(&device);

    // No unit axis: the plain variant, nothing begun.
    let plain = [4, 2, 2];
    assert_eq!(run(&device, &state, plain), expected(plain));
    let counts = settled(&device);
    assert_eq!(
        (counts.unit_axes_variants_begun, counts.unit_axes_meanwhile),
        (0, 0)
    );

    // Unit `y`: the plain variant while the variant is made, then the variant.
    let unit = [8, 1, 2];
    let want = expected(unit);
    let first = run(&device, &state, unit);
    assert_eq!(first, want, "the plain variant");
    let counts = settled(&device);
    assert_eq!(counts.unit_axes_variants_begun, 1);
    assert_eq!(counts.unit_axes_meanwhile, 1);
    assert_eq!(counts.unit_axes_dispatches, 0);
    let mut taken = None;
    for attempt in 1..TRIES {
        assert_eq!(run(&device, &state, unit), want, "dispatch {attempt}");
        let counts = settled(&device);
        if counts.unit_axes_dispatches == 1 {
            taken = Some(counts);
            break;
        }
    }
    let counts = taken.expect("the unit-axes variant taken");
    assert_eq!(counts.unit_axes_variants_begun, 1, "begun once");
    // The dispatches after it take what it planned, still the variant.
    assert_eq!(run(&device, &state, unit), want, "the variant again");
    assert_eq!(settled(&device).unit_axes_dispatches, 1);

    // No unit axis still takes the plain variant.
    assert_eq!(run(&device, &state, plain), expected(plain));
    assert_eq!(settled(&device).unit_axes_variants_begun, 1);
}
