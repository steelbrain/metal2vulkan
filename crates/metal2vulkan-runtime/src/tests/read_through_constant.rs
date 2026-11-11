//! A kernel buffer declared read-only in `device` space whose parameter is a `constant`-space
//! pointer, as Apple's compiler declares a by-value argument (`air.address_space 1`, a one-byte
//! `char`), runs as the same kernel declaring the `constant` buffer it reads, and computes what the
//! M3 Max (macOS 27.0.1) computes for it.
//!
//! The kernel is ours: `out[i] = p->rect + p->m[i & 3] * p->a + float4(p->n[i % 3], p->e)` over a
//! 160-byte `constant Params *p` (`float4 rect; float4x4 m; float3x3 n; float a, b, c, d, e;`)
//! set as bytes at index 0, `out` at index 1, over 8 threads, the 40 floats of `p` `k / 2 + 1`.
//! Metal's compiler wrote the kernel's `constant` node; the M3 Max made a pipeline of the kernel
//! with that node rewritten to the `device` `char` one and answered the same 32 floats as with the
//! node it wrote (`metal-as` of its typed text, both nodes).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The modules are LLVM text written here
//! (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use crate::{ComputePipelineState, Device, PipelineOptions, ResourceOptions, Size};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};

/// The node of the `Params` buffer as Metal's compiler writes it for `constant Params *p`.
const AS_CONSTANT: &str = "!\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 160, !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"Params\"";
/// The node as Apple's compiler writes it for a by-value argument: `device`, one `char`.
const AS_DEVICE_CHAR: &str = "!\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 1, !\"air.arg_type_align_size\", i32 1, !\"air.arg_type_name\", !\"char\"";

/// What the M3 Max answers, thread by thread.
const METAL: [f32; 32] = [
    63.0, 72.5, 82.0, 98.0, 99.0, 108.5, 118.0, 132.0, 135.0, 144.5, 154.0, 166.0, 165.0, 174.5,
    184.0, 200.0, 65.0, 74.5, 84.0, 98.0, 101.0, 110.5, 120.0, 132.0, 131.0, 140.5, 150.0, 166.0,
    167.0, 176.5, 186.0, 200.0,
];

/// The kernel `k`, its `Params` buffer's node holding `node`.
fn kernel(node: &str) -> String {
    format!(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @k(ptr addrspace(2) noundef readonly %0, ptr addrspace(1) noundef %1, i32 noundef %2) {{\n\
entry:\n\
  %rect = load <4 x float>, ptr addrspace(2) %0, align 16\n\
  %j = and i32 %2, 3\n\
  %j64 = zext i32 %j to i64\n\
  %jo = shl i64 %j64, 4\n\
  %mo = add i64 %jo, 16\n\
  %mp = getelementptr inbounds i8, ptr addrspace(2) %0, i64 %mo\n\
  %row = load <4 x float>, ptr addrspace(2) %mp, align 16\n\
  %ap = getelementptr inbounds i8, ptr addrspace(2) %0, i64 128\n\
  %a = load float, ptr addrspace(2) %ap, align 16\n\
  %r = urem i32 %2, 3\n\
  %r64 = zext i32 %r to i64\n\
  %ro = shl i64 %r64, 4\n\
  %no = add i64 %ro, 80\n\
  %np = getelementptr inbounds i8, ptr addrspace(2) %0, i64 %no\n\
  %n = load <3 x float>, ptr addrspace(2) %np, align 16\n\
  %ep = getelementptr inbounds i8, ptr addrspace(2) %0, i64 144\n\
  %e = load float, ptr addrspace(2) %ep, align 16\n\
  %a0 = insertelement <4 x float> poison, float %a, i64 0\n\
  %as = shufflevector <4 x float> %a0, <4 x float> poison, <4 x i32> zeroinitializer\n\
  %scaled = fmul <4 x float> %row, %as\n\
  %sum = fadd <4 x float> %rect, %scaled\n\
  %n4 = shufflevector <3 x float> %n, <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>\n\
  %ne = insertelement <4 x float> %n4, float %e, i64 3\n\
  %v = fadd <4 x float> %sum, %ne\n\
  %i64 = zext i32 %2 to i64\n\
  %op = getelementptr inbounds <4 x float>, ptr addrspace(1) %1, i64 %i64\n\
  store <4 x float> %v, ptr addrspace(1) %op, align 16\n\
  ret void\n\
}}\n\n\
!llvm.module.flags = !{{!0}}\n!air.kernel = !{{!1}}\n!air.version = !{{!7}}\n!air.language_version = !{{!8}}\n\
!0 = !{{i32 7, !\"air.max_device_buffers\", i32 31}}\n\
!1 = !{{ptr @k, !2, !3}}\n!2 = !{{}}\n!3 = !{{!4, !5, !6}}\n\
!4 = !{{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, {node}, !\"air.arg_name\", !\"p\"}}\n\
!5 = !{{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 16, !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"out\"}}\n\
!6 = !{{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"i\"}}\n\
!7 = !{{i32 2, i32 8, i32 0}}\n!8 = !{{!\"Metal\", i32 4, i32 0, i32 0}}\n"
    )
}

/// The state of `k` with its `Params` node `node`.
fn state(device: &Device, node: &str) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(&kernel(node)).unwrap();
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

/// The 32 floats `state` writes over 8 threads.
fn run(device: &Device, state: &ComputePipelineState) -> Vec<f32> {
    let params: Vec<u8> = (0..40)
        .flat_map(|k| (k as f32 / 2.0 + 1.0).to_le_bytes())
        .collect();
    let out = device
        .new_buffer_with_bytes(&[0; 128], ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_bytes(&params, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(8, 1, 1), Size::new(8, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    let contents = out.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), out.length() as usize) };
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect()
}

#[test]
fn a_read_only_device_buffer_read_through_a_constant_pointer_computes_what_metal_computes() {
    let scratch = Scratch::new("read-through-constant");
    let device = device(&scratch.0);
    for (form, node) in [("constant", AS_CONSTANT), ("device char", AS_DEVICE_CHAR)] {
        assert_eq!(run(&device, &state(&device, node)), METAL, "{form}");
    }
}
