//! Device-memory words the lanes of one simdgroup exchange, in Metal's order, through the runtime,
//! on the device the cfg selects, under the validation layer.
//!
//! Each kernel (written here from scratch) lets a lane read a buffer word another lane of its
//! simdgroup writes in the same barrier interval, with no barrier. Apple's M3 Max answers every
//! lane as its simdgroup's lockstep does (1,310,720 lanes per kernel, safe and fast math): an
//! instruction is done for every active lane before the next starts. The translation orders such
//! pairs with the barrier lockstep stood for, or with a subgroup barrier among the lanes that run
//! it together where the entry reconverges maximally:
//!
//! - `dl_raw1`, `dl_raw8`: `buf[t] = 3t + 1`, then `out[t] = buf[t ^ 1]` (`t ^ 8`): `3 (t ^ k) + 1`;
//! - `dl_war8`: `w = buf[t ^ 8]`, then `buf[t] = 100000 + t`, `out[t] = w`: the word before, `t ^ 8`;
//! - `dl_bcast`: lane 0 of each simdgroup stores `buf[32 s] = 7 s + 5` under a branch, then every
//!   lane reads `buf[32 s]`: `7 s + 5`;
//! - `dl_half`: the low 16 lanes of each simdgroup read `buf[t ^ 1]`, then store `buf[t]`, then
//!   write what they read to `out[t]`, in a block the high lanes skip: the word before, `t ^ 1`;
//!   the high lanes write nothing (`out` stays 0).
//!
//! `buf[i]` starts as `i`. Each kernel runs [`ROUNDS`] dispatches of [`N`] threads in threadgroups
//! of [`GROUP`], each from fresh buffers.

use super::tile_states::on_device_with;
use super::{at_plan, buffer_with, read_buffer, WAIT};
use crate::air_library::AirLibrary;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{CommandBufferStatus, Device, PipelineOptions, Size};

/// Threads a dispatch.
const N: u64 = 65536;
/// Threads a threadgroup.
const GROUP: u64 = 256;
/// Dispatches a kernel.
const ROUNDS: usize = 8;

/// The kernels.
const KERNELS: [&str; 5] = ["dl_raw1", "dl_raw8", "dl_war8", "dl_bcast", "dl_half"];

/// The module: `kernel(device uint *buf [[buffer(0)]], device uint *out [[buffer(1)]], uint t
/// [[thread_position_in_grid]], uint l [[thread_index_in_simdgroup]])`, each.
const MODULE: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @dl_raw1(ptr addrspace(1) %buf, ptr addrspace(1) %out, i32 %t, i32 %l) #0 {
  %i = zext i32 %t to i64
  %m = mul i32 %t, 3
  %v = add i32 %m, 1
  %p = getelementptr inbounds i32, ptr addrspace(1) %buf, i64 %i
  store i32 %v, ptr addrspace(1) %p, align 4
  %u = xor i32 %t, 1
  %j = zext i32 %u to i64
  %q = getelementptr inbounds i32, ptr addrspace(1) %buf, i64 %j
  %w = load i32, ptr addrspace(1) %q, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}

define void @dl_raw8(ptr addrspace(1) %buf, ptr addrspace(1) %out, i32 %t, i32 %l) #0 {
  %i = zext i32 %t to i64
  %m = mul i32 %t, 3
  %v = add i32 %m, 1
  %p = getelementptr inbounds i32, ptr addrspace(1) %buf, i64 %i
  store i32 %v, ptr addrspace(1) %p, align 4
  %u = xor i32 %t, 8
  %j = zext i32 %u to i64
  %q = getelementptr inbounds i32, ptr addrspace(1) %buf, i64 %j
  %w = load i32, ptr addrspace(1) %q, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}

define void @dl_war8(ptr addrspace(1) %buf, ptr addrspace(1) %out, i32 %t, i32 %l) #0 {
  %i = zext i32 %t to i64
  %u = xor i32 %t, 8
  %j = zext i32 %u to i64
  %q = getelementptr inbounds i32, ptr addrspace(1) %buf, i64 %j
  %w = load i32, ptr addrspace(1) %q, align 4
  %v = add i32 %t, 100000
  %p = getelementptr inbounds i32, ptr addrspace(1) %buf, i64 %i
  store i32 %v, ptr addrspace(1) %p, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}

define void @dl_bcast(ptr addrspace(1) %buf, ptr addrspace(1) %out, i32 %t, i32 %l) #0 {
entry:
  %s = lshr i32 %t, 5
  %b = shl i32 %s, 5
  %k = zext i32 %b to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %buf, i64 %k
  %first = icmp eq i32 %l, 0
  br i1 %first, label %write, label %read

write:
  %m = mul i32 %s, 7
  %v = add i32 %m, 5
  store i32 %v, ptr addrspace(1) %p, align 4
  br label %read

read:
  %w = load i32, ptr addrspace(1) %p, align 4
  %i = zext i32 %t to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %w, ptr addrspace(1) %o, align 4
  ret void
}

define void @dl_half(ptr addrspace(1) %buf, ptr addrspace(1) %out, i32 %t, i32 %l) #0 {
entry:
  %i = zext i32 %t to i64
  %low = icmp ult i32 %l, 16
  br i1 %low, label %arm, label %done

arm:
  %u = xor i32 %t, 1
  %j = zext i32 %u to i64
  %q = getelementptr inbounds i32, ptr addrspace(1) %buf, i64 %j
  %w = load i32, ptr addrspace(1) %q, align 4
  %v = add i32 %t, 1000000
  %p = getelementptr inbounds i32, ptr addrspace(1) %buf, i64 %i
  store i32 %v, ptr addrspace(1) %p, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %w, ptr addrspace(1) %o, align 4
  br label %done

done:
  ret void
}

attributes #0 = { convergent mustprogress nounwind willreturn }

!air.kernel = !{!0, !1, !2, !3, !4}
!air.version = !{!20}
!air.language_version = !{!21}

!0 = !{ptr @dl_raw1, !10, !11}
!1 = !{ptr @dl_raw8, !10, !11}
!2 = !{ptr @dl_war8, !10, !11}
!3 = !{ptr @dl_bcast, !10, !11}
!4 = !{ptr @dl_half, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"buf"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!15 = !{i32 3, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"l"}
!20 = !{i32 2, i32 8, i32 0}
!21 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// Metal's answer of `kernel` at thread `t` (the module docs).
fn metal(kernel: &str, t: u32) -> u32 {
    let lane = t % 32;
    match kernel {
        "dl_raw1" => 3 * (t ^ 1) + 1,
        "dl_raw8" => 3 * (t ^ 8) + 1,
        "dl_war8" => t ^ 8,
        "dl_bcast" => 7 * (t >> 5) + 5,
        "dl_half" if lane < 16 => t ^ 1,
        "dl_half" => 0,
        _ => unreachable!("a kernel of the module"),
    }
}

/// The module assembled.
fn bitcode() -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("m2v-device-lockstep-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("device_lockstep.ll");
    std::fs::write(&path, MODULE).unwrap();
    let bitcode = llvm_assemble(&path);
    let _ = std::fs::remove_dir_all(&dir);
    bitcode.unwrap()
}

/// The words of `out` after one dispatch of `kernel` on `device`.
fn run(device: &Device, bitcode: &[u8], kernel: &str) -> Vec<u32> {
    let library = device.new_library_with_air(bitcode).unwrap();
    let function = library.new_function_with_name(kernel).unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    let state = device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let initial: Vec<u8> = (0..N as u32).flat_map(|i| i.to_le_bytes()).collect();
    let buf = buffer_with(device, &initial).unwrap();
    let out = buffer_with(device, &vec![0; (N * 4) as usize]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&buf), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(N, 1, 1), Size::new(GROUP, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&out, 0, N * 4)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

/// Every lane reads Metal's answer: what its simdgroup's lockstep gives, in every round.
#[test]
fn device_words_a_simdgroup_exchanges_read_metal_s_answer() {
    on_device_with(at_plan, |device| {
        let name = device.name().to_owned();
        let bitcode = bitcode();
        let mut failed = Vec::new();
        for kernel in KERNELS {
            let mut wrong = 0usize;
            let mut first = Vec::new();
            for _ in 0..ROUNDS {
                for (t, got) in run(device, &bitcode, kernel).into_iter().enumerate() {
                    let want = metal(kernel, t as u32);
                    if got != want {
                        wrong += 1;
                        if first.len() < 4 {
                            first.push(format!("t={t} got {got}, Metal's {want}"));
                        }
                    }
                }
            }
            eprintln!(
                "{kernel} on {name}: {wrong} of {} lanes differ from Metal's {}",
                ROUNDS as u64 * N,
                first.join("; ")
            );
            if wrong > 0 {
                failed.push(format!("{kernel}: {wrong} lanes ({})", first.join("; ")));
            }
        }
        assert!(
            failed.is_empty(),
            "on {name}, lanes differ from Metal's:\n  {}",
            failed.join("\n  ")
        );
    });
}
