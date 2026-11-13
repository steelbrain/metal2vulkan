//! A `float4` whose address is a multiple of 4 and not of 16 is read and written at that address,
//! whatever the `align 16` its AIR states, on the device: rows of 21 `float`s, one thread a row,
//! each written as four `float4`s and five `float`s, as Apple's compiler emits `device float4 *v =
//! (device float4 *)(out + 21 * p); v[i] = float4(p, 4i, 4i + 1, 4i + 2)` and `out[21 * p + c] =
//! c`. Row `p` starts at byte `84 p`, so 3 rows in 4 start 4, 8 or 12 bytes past a multiple of 16.
//! Metal's answer over 8 threads on an Apple M3 Max (`probes/vector_access_placement.m` in the
//! conformance crate) is every row exact: `p 0 1 2 p 4 5 6 p 8 9 10 p 12 13 14 16 17 18 19 20`.
//! The buffer is bound at offsets 0 and 4 (a `float` buffer may be bound at any multiple of 4).
//! Beside it, `float4`s at their own indices of a `float4` buffer, bound at 0 and 16, whose
//! offsets the translator proves multiples of 16.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The kernels are LLVM text written here
//! (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use crate::{
    Buffer, CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, ResourceOptions,
    Size,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};

/// The threads, one row each, in one threadgroup.
const THREADS: u64 = 8;
/// A row's `float`s.
const ROW: u64 = 21;

const HEADER: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

declare float @air.convert.f.f32.u.i32(i32)

!air.kernel = !{!9}
!air.version = !{!17}
!air.language_version = !{!18}
!9 = !{ptr @k, !10, !11}
!10 = !{}
!11 = !{!12, !13}
!13 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"p"}
!17 = !{i32 2, i32 8, i32 0}
!18 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The rows kernel, its loops kept as Apple's compiler emits them.
const ROWS: &str = r#"define void @k(ptr addrspace(1) noundef "air-buffer-no-alias" %0, i32 noundef %1) {
entry:
  %r = mul i32 %1, 21
  %rz = zext i32 %r to i64
  %row = getelementptr inbounds float, ptr addrspace(1) %0, i64 %rz
  %pf = tail call fast float @air.convert.f.f32.u.i32(i32 %1)
  %v0 = insertelement <4 x float> undef, float %pf, i64 0
  br label %vectors

vectors:
  %i = phi i32 [ 0, %entry ], [ %in, %vectors ]
  %f = shl nuw nsw i32 %i, 2
  %f1 = tail call fast float @air.convert.f.f32.u.i32(i32 %f)
  %v1 = insertelement <4 x float> %v0, float %f1, i64 1
  %g = or i32 %f, 1
  %g1 = tail call fast float @air.convert.f.f32.u.i32(i32 %g)
  %v2 = insertelement <4 x float> %v1, float %g1, i64 2
  %h = or i32 %f, 2
  %h1 = tail call fast float @air.convert.f.f32.u.i32(i32 %h)
  %v3 = insertelement <4 x float> %v2, float %h1, i64 3
  %iz = zext i32 %i to i64
  %at = getelementptr inbounds <4 x float>, ptr addrspace(1) %row, i64 %iz
  store <4 x float> %v3, ptr addrspace(1) %at, align 16
  %in = add nuw nsw i32 %i, 1
  %vdone = icmp eq i32 %in, 4
  br i1 %vdone, label %scalars, label %vectors

scalars:
  %c = phi i32 [ 16, %vectors ], [ %cn, %scalars ]
  %cf = tail call fast float @air.convert.f.f32.u.i32(i32 %c)
  %e = add i32 %c, %r
  %ez = zext i32 %e to i64
  %ep = getelementptr inbounds float, ptr addrspace(1) %0, i64 %ez
  store float %cf, ptr addrspace(1) %ep, align 4
  %cn = add nuw nsw i32 %c, 1
  %sdone = icmp eq i32 %cn, 21
  br i1 %sdone, label %exit, label %scalars

exit:
  ret void
}

!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
"#;

/// `v[p] = float4(p, 1, 2, 3)` over a `device float4 *`.
const GRID: &str = r#"define void @k(ptr addrspace(1) noundef "air-buffer-no-alias" %0, i32 noundef %1) {
entry:
  %pf = tail call fast float @air.convert.f.f32.u.i32(i32 %1)
  %v = insertelement <4 x float> <float undef, float 1.0, float 2.0, float 3.0>, float %pf, i64 0
  %iz = zext i32 %1 to i64
  %at = getelementptr inbounds <4 x float>, ptr addrspace(1) %0, i64 %iz
  store <4 x float> %v, ptr addrspace(1) %at, align 16
  ret void
}

!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"v"}
"#;

/// The compute state of kernel `body` on `device`.
fn state(device: &Device, body: &str) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(&format!("{HEADER}{body}")).unwrap();
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

/// Dispatch `state` over [`THREADS`] threads with a buffer of `len` bytes after `offset` bytes
/// bound at `offset`, and the `float`s it holds from there.
fn run(device: &Device, state: &ComputePipelineState, offset: u64, len: u64) -> Vec<f32> {
    let total = (offset + len) as usize;
    let out: Buffer = device
        .new_buffer_with_bytes(&vec![0xEE; total], ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), offset, 0).unwrap();
    encoder
        .dispatch_threads(Size::new(THREADS, 1, 1), Size::new(THREADS, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's bytes, which the completed command buffer wrote.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), total) };
    (bytes[offset as usize..].as_chunks::<4>().0.iter())
        .map(|&word| f32::from_le_bytes(word))
        .collect()
}

#[test]
fn a_float4_aligned_only_to_its_lanes_is_placed_at_its_address() {
    let scratch = Scratch::new("vector-placement");
    let device = device(&scratch.0);
    let metal: Vec<f32> = (0..THREADS)
        .flat_map(|p| {
            let p = p as f32;
            let mut row: Vec<f32> = (0..4)
                .flat_map(|i| {
                    let f = (4 * i) as f32;
                    [p, f, f + 1.0, f + 2.0]
                })
                .collect();
            row.extend((16..ROW).map(|c| c as f32));
            row
        })
        .collect();
    let rows = state(&device, ROWS);
    for offset in [0, 4] {
        let got = run(&device, &rows, offset, 4 * ROW * THREADS);
        assert_eq!(got, metal, "rows at offset {offset}");
    }
    let grid = state(&device, GRID);
    let want: Vec<f32> = (0..THREADS)
        .flat_map(|p| [p as f32, 1.0, 2.0, 3.0])
        .collect();
    for offset in [0, 16] {
        let got = run(&device, &grid, offset, 16 * THREADS);
        assert_eq!(got, want, "grid at offset {offset}");
    }
    println!(
        "VECTOR-PLACEMENT {}: the rows at offsets 0 and 4 and the grid at 0 and 16 are Metal's",
        device.name()
    );
}
