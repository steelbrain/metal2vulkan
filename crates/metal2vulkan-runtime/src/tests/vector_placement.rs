//! A `float4` whose address is a multiple of 4 and not of 16 is read and written at that address,
//! whatever the `align 16` its AIR states, on the device. Rows of 21 `float`s, one thread a row,
//! as Apple's compiler emits them (loops kept): row `p` starts at byte `84 p`, so 3 rows in 4 start
//! 4, 8 or 12 bytes past a multiple of 16.
//!
//! - [`STORES`]: `device float4 *v = (device float4 *)(out + 21 * p); v[i] = float4(p, 4i, 4i + 1,
//!   4i + 2)`, then `out[21 * p + c] = c` for `c` in 16..21.
//! - [`LOADS`]: `float4 x = ((device float4 *)(in + 21 * p))[i]`, each lane stored as a `float` at
//!   `out[21 * p + 4i + j]`, then `out[21 * p + c] = in[21 * p + c]`, over `in[j] = j`.
//!
//! Metal's answer over 8 threads on an Apple M3 Max and a Radeon Pro 5300M
//! (`probes/vector_access_placement.m` in the conformance crate) is every row exact: the stores'
//! row `p 0 1 2 p 4 5 6 p 8 9 10 p 12 13 14 16 17 18 19 20`, the loads' `in`. A placement at the
//! offset rounded down to 16 bytes instead puts row `p`'s vectors `(84 p mod 16) / 4` floats early:
//! the stores overwrite that many floats of the row before and leave as many of their own row's
//! floats 12 to 15 unwritten, and the loads read the row from that many floats early. The rows'
//! buffers are bound at offsets 0 and 4 (a `float` buffer may be bound at any multiple of 4).
//! Beside them, `float4`s at their own indices of a `float4` buffer ([`GRID`]), bound at 0 and 16,
//! whose offsets the translator proves multiples of 16.
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
/// The bytes of the rows.
const ROWS_BYTES: u64 = 4 * ROW * THREADS;

const HEADER: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

declare float @air.convert.f.f32.u.i32(i32)

!air.kernel = !{!9}
!air.version = !{!17}
!air.language_version = !{!18}
!9 = !{ptr @k, !10, !11}
!10 = !{}
!17 = !{i32 2, i32 8, i32 0}
!18 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// The stores kernel.
const STORES: &str = r#"define void @k(ptr addrspace(1) noundef "air-buffer-no-alias" %0, i32 noundef %1) {
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

!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!13 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"p"}
"#;

/// The loads kernel.
const LOADS: &str = r#"define void @k(ptr addrspace(1) noundef "air-buffer-no-alias" %0, ptr addrspace(1) noundef "air-buffer-no-alias" %1, i32 noundef %2) {
entry:
  %r = mul i32 %2, 21
  %rz = zext i32 %r to i64
  %row = getelementptr inbounds float, ptr addrspace(1) %0, i64 %rz
  br label %vectors

vectors:
  %i = phi i32 [ 0, %entry ], [ %in, %vectors ]
  %iz = zext i32 %i to i64
  %at = getelementptr inbounds <4 x float>, ptr addrspace(1) %row, i64 %iz
  %v = load <4 x float>, ptr addrspace(1) %at, align 16
  %f = shl nuw nsw i32 %i, 2
  %b = add i32 %f, %r
  %x0 = extractelement <4 x float> %v, i64 0
  %b0 = zext i32 %b to i64
  %p0 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %b0
  store float %x0, ptr addrspace(1) %p0, align 4
  %x1 = extractelement <4 x float> %v, i64 1
  %b1 = add i64 %b0, 1
  %p1 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %b1
  store float %x1, ptr addrspace(1) %p1, align 4
  %x2 = extractelement <4 x float> %v, i64 2
  %b2 = add i64 %b0, 2
  %p2 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %b2
  store float %x2, ptr addrspace(1) %p2, align 4
  %x3 = extractelement <4 x float> %v, i64 3
  %b3 = add i64 %b0, 3
  %p3 = getelementptr inbounds float, ptr addrspace(1) %1, i64 %b3
  store float %x3, ptr addrspace(1) %p3, align 4
  %in = add nuw nsw i32 %i, 1
  %vdone = icmp eq i32 %in, 4
  br i1 %vdone, label %scalars, label %vectors

scalars:
  %c = phi i32 [ 16, %vectors ], [ %cn, %scalars ]
  %e = add i32 %c, %r
  %ez = zext i32 %e to i64
  %ip = getelementptr inbounds float, ptr addrspace(1) %0, i64 %ez
  %y = load float, ptr addrspace(1) %ip, align 4
  %op = getelementptr inbounds float, ptr addrspace(1) %1, i64 %ez
  store float %y, ptr addrspace(1) %op, align 4
  %cn = add nuw nsw i32 %c, 1
  %sdone = icmp eq i32 %cn, 21
  br i1 %sdone, label %exit, label %scalars

exit:
  ret void
}

!11 = !{!12, !14, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!14 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!13 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"p"}
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

!11 = !{!12, !13}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"v"}
!13 = !{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"p"}
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

/// `bytes` after `offset` bytes of `0xEE`, as a shared buffer.
fn buffer(device: &Device, offset: u64, bytes: &[u8]) -> Buffer {
    let mut all = vec![0xEE; offset as usize];
    all.extend_from_slice(bytes);
    device
        .new_buffer_with_bytes(&all, ResourceOptions::shared())
        .unwrap()
}

/// Dispatch `state` over [`THREADS`] threads with `buffers` bound at `offset` at indices 0, 1, ...,
/// and the `float`s the last one holds from there.
fn run(device: &Device, state: &ComputePipelineState, offset: u64, buffers: &[Buffer]) -> Vec<f32> {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    for (index, buffer) in buffers.iter().enumerate() {
        encoder
            .set_buffer(Some(buffer), offset, index as u64)
            .unwrap();
    }
    encoder
        .dispatch_threads(Size::new(THREADS, 1, 1), Size::new(THREADS, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(command_buffer.error(), None);
    let out = buffers.last().unwrap();
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's bytes, which the completed command buffer wrote.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), out.length() as usize) };
    (bytes[offset as usize..].as_chunks::<4>().0.iter())
        .map(|&word| f32::from_le_bytes(word))
        .collect()
}

#[test]
fn a_float4_aligned_only_to_its_lanes_is_placed_at_its_address() {
    let scratch = Scratch::new("vector-placement");
    let device = device(&scratch.0);
    let stored: Vec<f32> = (0..THREADS)
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
    let input: Vec<f32> = (0..ROW * THREADS).map(|j| j as f32).collect();
    let input_bytes: Vec<u8> = input.iter().flat_map(|x| x.to_le_bytes()).collect();
    let stores = state(&device, STORES);
    let loads = state(&device, LOADS);
    for offset in [0, 4] {
        let out = buffer(&device, offset, &[0xEE; ROWS_BYTES as usize]);
        let got = run(&device, &stores, offset, &[out]);
        assert_eq!(got, stored, "the stores at offset {offset}");
        let from = buffer(&device, offset, &input_bytes);
        let out = buffer(&device, offset, &[0xEE; ROWS_BYTES as usize]);
        let got = run(&device, &loads, offset, &[from, out]);
        assert_eq!(got, input, "the loads at offset {offset}");
    }
    let grid = state(&device, GRID);
    let want: Vec<f32> = (0..THREADS)
        .flat_map(|p| [p as f32, 1.0, 2.0, 3.0])
        .collect();
    for offset in [0, 16] {
        let out = buffer(&device, offset, &[0xEE; 16 * THREADS as usize]);
        let got = run(&device, &grid, offset, &[out]);
        assert_eq!(got, want, "the grid at offset {offset}");
    }
    println!(
        "VECTOR-PLACEMENT {}: the stores and loads at offsets 0 and 4 and the grid at 0 and 16 are \
         Metal's",
        device.name()
    );
}
