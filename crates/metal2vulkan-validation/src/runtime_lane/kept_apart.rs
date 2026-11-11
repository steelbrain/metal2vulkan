//! Float arithmetic in the AIR's order on the device the cfg selects, under the validation layer.
//!
//! The translation emits each float operation in the AIR's order and decorates it by its own
//! fast-math flags (`lower::arith`): one carrying both `contract` and `reassoc` is left undecorated
//! for the device's compiler to group and fuse as it chooses, and any other is `NoContraction`.
//! Vulkan requires a correctly rounded 32-bit `OpFAdd` and `OpFMul`, and forbids a `NoContraction`
//! operation to be fused or rearranged, so over 4096 inputs of normal operands each operation
//! kept apart must store the host's answer of the AIR's order exactly:
//!
//! - `strict_mul_add`, `a * b + c` with no flag, and `contract_mul_add`, the same under `contract`
//!   alone (the AIR lets it fuse but not regroup, which one decoration cannot say), each the
//!   product rounded and then the sum, never the fused multiply-add;
//! - `precise_sum`, `((d + c) + b) + a` with no flag, in that order.
//!
//! Half the multiply-adds' addends are `-RN(a * b)`, where the fused answer is the product's
//! rounding error and the AIR's is `0`, so a device that fused them would miss on those inputs.
//! The `fast` twins (`fast_mul_add`, `fast_sum`) are the device's own grouping: each prints how
//! many of its words are the AIR's order and how many the fused pair's, and a multiply-add must
//! store one of the two.
//!
//! A `half` chain, `(a + b) + c` of `a = 1` and `b = c = 2^-11`, is `1` rounded at each sum under
//! any rounding (`0x3F800000` widened) and `1 + 2^-10` held at `float` and rounded once
//! (`0x3F802000`). On a device without `shaderFloat16`, whose module rounds by the quantize
//! (`NativeHalfQuantize`), the `fast` chain stores the held sum under `HalfChainPolicy::AtExits`
//! (the default) and the rounded one under `HalfChainPolicy::EachOperation` (`exact_answers`); the
//! chain without flags the rounded one under both. Elsewhere each chain stores one of the two (a
//! driver that drops a conversion pair keeps the `float` sum), counted, or, where its stage records
//! `PrecisionNote::FastMathRewrites`, a sum between them, each rounding taken or skipped: on
//! MoltenVK the `fast` chain's 16-bit sums are left to Metal's fast compile, which may fold one's
//! conversions away (`1 + 2^-11`, `0x3F801000`, measured on an Apple M3 Max).

use super::{
    at_plan, buffer_with, layer_reported_nothing_at_all, read_buffer, validated_device_with,
    Messages, ScratchCache, WAIT,
};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::cache::VariantKey;
use metal2vulkan::device::HalfChainPolicy;
use metal2vulkan::library::Library;
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, FunctionDescriptor, PrecisionNote, Stage, StageSource,
};
use metal2vulkan::Budget;
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{CommandBufferStatus, Device, PipelineOptions, Size};

const MODULE: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @strict_mul_add(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %g) {
  %pi = getelementptr inbounds <4 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <4 x float>, ptr addrspace(1) %pi, align 16
  %a = extractelement <4 x float> %v, i64 0
  %b = extractelement <4 x float> %v, i64 1
  %c = extractelement <4 x float> %v, i64 2
  %p = fmul float %a, %b
  %r = fadd float %p, %c
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %r, ptr addrspace(1) %po, align 4
  ret void
}

define void @contract_mul_add(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %g) {
  %pi = getelementptr inbounds <4 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <4 x float>, ptr addrspace(1) %pi, align 16
  %a = extractelement <4 x float> %v, i64 0
  %b = extractelement <4 x float> %v, i64 1
  %c = extractelement <4 x float> %v, i64 2
  %p = fmul contract float %a, %b
  %r = fadd contract float %p, %c
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %r, ptr addrspace(1) %po, align 4
  ret void
}

define void @fast_mul_add(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %g) {
  %pi = getelementptr inbounds <4 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <4 x float>, ptr addrspace(1) %pi, align 16
  %a = extractelement <4 x float> %v, i64 0
  %b = extractelement <4 x float> %v, i64 1
  %c = extractelement <4 x float> %v, i64 2
  %p = fmul fast float %a, %b
  %r = fadd fast float %p, %c
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %r, ptr addrspace(1) %po, align 4
  ret void
}

define void @precise_sum(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %g) {
  %pi = getelementptr inbounds <4 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <4 x float>, ptr addrspace(1) %pi, align 16
  %a = extractelement <4 x float> %v, i64 0
  %b = extractelement <4 x float> %v, i64 1
  %c = extractelement <4 x float> %v, i64 2
  %d = extractelement <4 x float> %v, i64 3
  %s1 = fadd float %d, %c
  %s2 = fadd float %s1, %b
  %s3 = fadd float %s2, %a
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %s3, ptr addrspace(1) %po, align 4
  ret void
}

define void @fast_sum(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %g) {
  %pi = getelementptr inbounds <4 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <4 x float>, ptr addrspace(1) %pi, align 16
  %a = extractelement <4 x float> %v, i64 0
  %b = extractelement <4 x float> %v, i64 1
  %c = extractelement <4 x float> %v, i64 2
  %d = extractelement <4 x float> %v, i64 3
  %s1 = fadd fast float %d, %c
  %s2 = fadd fast float %s1, %b
  %s3 = fadd fast float %s2, %a
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %s3, ptr addrspace(1) %po, align 4
  ret void
}

!air.kernel = !{!20, !21, !22, !23, !24}
!air.version = !{!1}
!air.language_version = !{!2}

!1 = !{i32 2, i32 8, i32 0}
!2 = !{!"Metal", i32 4, i32 0, i32 0}
!3 = !{}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"in"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!7 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"g"}
!9 = !{!4, !5, !7}
!20 = !{ptr @strict_mul_add, !3, !9}
!21 = !{ptr @contract_mul_add, !3, !9}
!22 = !{ptr @fast_mul_add, !3, !9}
!23 = !{ptr @precise_sum, !3, !9}
!24 = !{ptr @fast_sum, !3, !9}
"#;

/// The `half` chains: `(a + b) + c` of the first three lanes narrowed to `half`, under `fast` and
/// with no flag, widened back to `float` and stored.
const HALF_MODULE: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

define void @fast_half_chain(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %g) {
  %pi = getelementptr inbounds <4 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <4 x float>, ptr addrspace(1) %pi, align 16
  %fa = extractelement <4 x float> %v, i64 0
  %fb = extractelement <4 x float> %v, i64 1
  %fc = extractelement <4 x float> %v, i64 2
  %a = fptrunc float %fa to half
  %b = fptrunc float %fb to half
  %c = fptrunc float %fc to half
  %s1 = fadd fast half %a, %b
  %s2 = fadd fast half %s1, %c
  %r = fpext half %s2 to float
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %r, ptr addrspace(1) %po, align 4
  ret void
}

define void @strict_half_chain(ptr addrspace(1) %in, ptr addrspace(1) %out, i32 %g) {
  %pi = getelementptr inbounds <4 x float>, ptr addrspace(1) %in, i32 %g
  %v = load <4 x float>, ptr addrspace(1) %pi, align 16
  %fa = extractelement <4 x float> %v, i64 0
  %fb = extractelement <4 x float> %v, i64 1
  %fc = extractelement <4 x float> %v, i64 2
  %a = fptrunc float %fa to half
  %b = fptrunc float %fb to half
  %c = fptrunc float %fc to half
  %s1 = fadd half %a, %b
  %s2 = fadd half %s1, %c
  %r = fpext half %s2 to float
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i32 %g
  store float %r, ptr addrspace(1) %po, align 4
  ret void
}

!air.kernel = !{!20, !21}
!air.version = !{!1}
!air.language_version = !{!2}

!1 = !{i32 2, i32 8, i32 0}
!2 = !{!"Metal", i32 4, i32 0, i32 0}
!3 = !{}
!4 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 16, !"air.arg_type_align_size", i32 16, !"air.arg_type_name", !"float4", !"air.arg_name", !"in"}
!5 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!7 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"g"}
!9 = !{!4, !5, !7}
!20 = !{ptr @fast_half_chain, !3, !9}
!21 = !{ptr @strict_half_chain, !3, !9}
"#;

/// The threads, one input each, in threadgroups of 64.
const THREADS: usize = 4096;

/// A fixed xorshift generator.
struct Words(u32);

impl Words {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }
}

/// The inputs `(a, b, c, d)`: normal operands of magnitude in `[2^-10, 2^10)`, either sign, with
/// random fractions; every other `c` is `-RN(a * b)`, so the AIR's multiply-add is `0` and the
/// fused one the product's rounding error. Every product, sum and rounding error stays normal.
fn inputs() -> Vec<[f32; 4]> {
    let mut w = Words(0x9E37_79B9);
    let mut normal = || {
        let x = w.next();
        let exponent = 127 - 10 + (x >> 24) % 20;
        f32::from_bits((x & 0x8000_0000) | (exponent << 23) | (w.next() & 0x7F_FFFF))
    };
    (0..THREADS)
        .map(|g| {
            let (a, b, c, d) = (normal(), normal(), normal(), normal());
            let c = if g % 2 == 0 { -(a * b) } else { c };
            [a, b, c, d]
        })
        .collect()
}

/// The words `kernel` stores on `device` over `input`, one a thread.
fn run(
    device: &Device,
    library: &metal2vulkan_runtime::Library,
    kernel: &str,
    input: &[[f32; 4]],
) -> Vec<u32> {
    let function = library.new_function_with_name(kernel).unwrap();
    let pipeline = device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .unwrap();
    let bytes: Vec<u8> = input
        .iter()
        .flat_map(|v| v.iter().flat_map(|f| f.to_le_bytes()))
        .collect();
    let input = buffer_with(device, &bytes).unwrap();
    let out = buffer_with(device, &vec![0xEE; 4 * THREADS]).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&pipeline).unwrap();
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    encoder
        .dispatch_threadgroups(Size::new(THREADS as u64 / 64, 1, 1), Size::new(64, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    read_buffer(&out, 0, 4 * THREADS as u64)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

/// The inputs whose word differs between `got` and `want`, with the two words, at most 24.
fn differing(got: &[u32], want: &[u32]) -> Vec<(usize, u32, u32)> {
    (got.iter().zip(want).enumerate())
        .filter(|(_, (x, y))| x != y)
        .map(|(g, (x, y))| (g, *x, *y))
        .take(24)
        .collect()
}

/// Each operation kept apart stores the AIR's order on the device, and a `fast` multiply-add the
/// AIR's or the fused one (module docs).
#[test]
fn arithmetic_kept_apart_stores_the_airs_order() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), at_plan).unwrap();
        let name = device.name().to_owned();
        let bitcode = llvm_assemble_text(MODULE).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let input = inputs();
        let unfused: Vec<u32> = input
            .iter()
            .map(|[a, b, c, _]| (a * b + c).to_bits())
            .collect();
        let fused: Vec<u32> = input
            .iter()
            .map(|[a, b, c, _]| a.mul_add(*b, *c).to_bits())
            .collect();
        let in_order: Vec<u32> = input
            .iter()
            .map(|[a, b, c, d]| (((d + c) + b) + a).to_bits())
            .collect();
        // The inputs tell the two multiply-adds apart.
        let apart = unfused.iter().zip(&fused).filter(|(u, f)| u != f).count();
        assert!(apart >= THREADS / 4, "{apart} inputs tell the fusion apart");
        let mut failures = Vec::new();
        for (kernel, want) in [
            ("strict_mul_add", &unfused),
            ("contract_mul_add", &unfused),
            ("precise_sum", &in_order),
        ] {
            let got = run(&device, &library, kernel, &input);
            let differ = differing(&got, want);
            eprintln!(
                "kept apart {kernel} on {name}: {} of {THREADS} words differ from the AIR's order",
                got.iter().zip(want.iter()).filter(|(g, w)| g != w).count()
            );
            if !differ.is_empty() {
                failures.push(format!(
                    "{kernel} on {name}: (input, device, host) {differ:x?}"
                ));
            }
        }
        let mul_add = run(&device, &library, "fast_mul_add", &input);
        let (mut as_written, mut as_fused, mut other) = (0, 0, Vec::new());
        for (g, word) in mul_add.iter().enumerate() {
            match (*word == unfused[g], *word == fused[g]) {
                (true, _) => as_written += 1,
                (false, true) => as_fused += 1,
                (false, false) => other.push((g, *word, unfused[g], fused[g])),
            }
        }
        let sum = run(&device, &library, "fast_sum", &input);
        let sum_in_order = sum.iter().zip(&in_order).filter(|(g, w)| g == w).count();
        eprintln!(
            "the device's own grouping on {name}: fast_mul_add {as_written} as written, {as_fused} \
             fused; fast_sum {sum_in_order} of {THREADS} in the AIR's order"
        );
        if !other.is_empty() {
            other.truncate(24);
            failures.push(format!(
                "fast_mul_add on {name}: neither as written nor fused (input, device, written, \
                 fused) {other:x?}"
            ));
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
    layer_reported_nothing_at_all(&messages);
}

/// A `half` chain without `shaderFloat16` rounds where its value leaves it by default and at each
/// operation under `exact_answers` (module docs).
#[test]
fn half_chains_without_float16_round_at_their_exits() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    {
        let bitcode = llvm_assemble_text(HALF_MODULE).unwrap();
        let tie = 2f32.powi(-11);
        let input = vec![[1.0, tie, tie, 0.0]; THREADS];
        let (rounded, held) = (1f32.to_bits(), (1.0 + 2f32.powi(-10)).to_bits());
        let mut failures = Vec::new();
        // The lane's devices take `exact_answers`; each here takes its own value.
        for (policy, exact) in [("AtExits", false), ("EachOperation", true)] {
            let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |o| {
                o.policy.half_chains = match exact {
                    true => HalfChainPolicy::EachOperation,
                    false => HalfChainPolicy::AtExits,
                };
                at_plan(o)
            })
            .unwrap();
            let name = device.name().to_owned();
            // Whether the compiled kernel records `note`.
            let records = |kernel: &str, note: PrecisionNote| {
                let library = Library::from_air(&bitcode).unwrap();
                let function = FunctionDescriptor::new(library.function(kernel).unwrap());
                let inputs = VariantKey::new().lowering_inputs(Stage::Kernel);
                device
                    .compile_device()
                    .compile_function(&function, &inputs, &Budget::standard())
                    .unwrap()
                    .precision_notes()
                    .contains(&note)
            };
            // The module rounds by the device's quantize: the device has no 16-bit floats.
            let quantizes = records("fast_half_chain", PrecisionNote::NativeHalfQuantize);
            let library = device.new_library_with_air(&bitcode).unwrap();
            for kernel in ["fast_half_chain", "strict_half_chain"] {
                // A chain the stage notes may skip any of its roundings: its sum lies between the
                // rounded one and the held one (both positive, so their bits order as values).
                let rewrites = records(kernel, PrecisionNote::FastMathRewrites);
                let skips = |w: u32| rewrites && (rounded..=held).contains(&w);
                let got = run(&device, &library, kernel, &input);
                let held_words = got.iter().filter(|w| **w == held).count();
                let rounded_words = got.iter().filter(|w| **w == rounded).count();
                let between = got
                    .iter()
                    .filter(|w| **w != held && **w != rounded && skips(**w))
                    .count();
                eprintln!(
                    "{kernel} under {policy} on {name} (quantize: {quantizes}, rewrites noted: \
                     {rewrites}): {held_words} held at float, {rounded_words} rounded at each \
                     sum, {between} between, of {THREADS}"
                );
                let want = match (quantizes, kernel == "fast_half_chain" && !exact) {
                    (true, true) => Some(held),
                    (true, false) => Some(rounded),
                    (false, _) => None,
                };
                let bad: Vec<(usize, u32)> = got
                    .iter()
                    .enumerate()
                    .filter(|(_, w)| match want {
                        Some(want) => **w != want,
                        None => **w != held && **w != rounded && !skips(**w),
                    })
                    .map(|(g, w)| (g, *w))
                    .take(24)
                    .collect();
                if !bad.is_empty() {
                    failures.push(format!(
                        "{kernel} under {policy} on {name}: want {want:x?} (held {held:x}, \
                         rounded {rounded:x}), got {bad:x?}"
                    ));
                }
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }
    layer_reported_nothing_at_all(&messages);
}
