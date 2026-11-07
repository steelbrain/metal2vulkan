//! Kernels in the shapes Metal's compiler writes for OpenCL C (`newLibraryWithSource` with
//! `sourceLanguage` 1, as the GL renderer's compute path builds an OpenCL program), each run and
//! compared with what Metal computes.
//!
//! - **A barrier out of line**: OpenCL C's `barrier(flags)` stays an internal function passing its
//!   flags on to `air.wg.barrier`, and `mem_fence(flags)` computes `air.atomic.fence`'s flags from
//!   them through `llvm.bitreverse.i2`. Both made and run on an M3 Max (macOS 27.0.1) and an Intel
//!   UHD 630 (26.7.1): `x[i] = 1`, the barrier, `x[i] += x[i ^ 1]` stores 2 in every word.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The modules are LLVM text written here
//! (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use crate::{Buffer, ComputePipelineState, Device, PipelineOptions, ResourceOptions, Size};

const HEAD: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\
declare i32 @air.get_global_id.i32(i32) #9\n\
attributes #9 = { nounwind readnone willreturn }\n\
!air.version = !{!90}\n\
!air.language_version = !{!91}\n\
!90 = !{i32 2, i32 8, i32 0}\n\
!91 = !{!\"Metal\", i32 3, i32 2, i32 0}\n";

/// The node of kernel argument 0, `x`, a read-write `int` buffer at index 0.
const X: &str = "!{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"int\", !\"air.arg_name\", !\"x\"}";

/// The words a dispatch runs over.
const WORDS: u64 = 64;

/// The state of kernel `name` of the module `text` (after [`HEAD`]).
fn state(device: &Device, text: &str, name: &str) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(&format!("{HEAD}{text}")).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name(name)
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// The words of `buffer`, which no command uses now.
fn words(buffer: &Buffer) -> Vec<u32> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command uses now.
    let bytes = unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) };
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// `state` run over [`WORDS`] threads in threadgroups of `group`, `x` holding `init` and `bind`
/// binding the rest; `x`'s words after.
fn run(
    device: &Device,
    state: &ComputePipelineState,
    init: &[u32],
    group: u64,
    bind: impl FnOnce(&mut crate::ComputeCommandEncoder),
) -> Vec<u32> {
    let bytes: Vec<u8> = init.iter().flat_map(|w| w.to_le_bytes()).collect();
    let x = device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&x), 0, 0).unwrap();
    bind(&mut encoder);
    encoder
        .dispatch_threads(Size::new(WORDS, 1, 1), Size::new(group, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    words(&x)
}

/// `x[i] = 1`, `@wait(3)`, `x[i] += x[i ^ 1]`, the barrier's flags (`mem_device |
/// mem_threadgroup`, which order the device words the threads exchange) `@wait`'s parameter.
const BARRIER: &str = "define void @fence(ptr addrspace(1) nocapture %x) #3 {\n\
  %id = tail call i32 @air.get_global_id.i32(i32 0)\n\
  %i = zext i32 %id to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %x, i64 %i\n\
  store i32 1, ptr addrspace(1) %p, align 4\n\
  tail call fastcc void @wait(i32 3)\n\
  %j = xor i64 %i, 1\n\
  %q = getelementptr inbounds i32, ptr addrspace(1) %x, i64 %j\n\
  %a = load i32, ptr addrspace(1) %q, align 4\n\
  %b = load i32, ptr addrspace(1) %p, align 4\n\
  %s = add i32 %a, %b\n\
  store i32 %s, ptr addrspace(1) %p, align 4\n\
  ret void\n}\n\
define internal fastcc void @wait(i32 %flags) #2 {\n\
  tail call void @air.wg.barrier(i32 %flags, i32 1) #1\n\
  ret void\n}\n\
declare void @air.wg.barrier(i32, i32) #1\n\
attributes #1 = { convergent nounwind willreturn }\n\
attributes #2 = { convergent noinline nounwind }\n\
attributes #3 = { convergent nounwind }\n\
!air.kernel = !{!0}\n\
!0 = !{ptr @fence, !1, !2}\n\
!1 = !{}\n\
!2 = !{!3}\n";

/// `x[i] = 1`, `@fence(2)` (its flags `bitreverse.i2` of the parameter's low bits, as OpenCL C's
/// `mem_fence(CLK_GLOBAL_MEM_FENCE)` becomes `mem_device`), `x[i] = 2`.
const FENCE: &str = "define void @store2(ptr addrspace(1) nocapture %x) #3 {\n\
  %id = tail call i32 @air.get_global_id.i32(i32 0)\n\
  %i = zext i32 %id to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %x, i64 %i\n\
  store i32 1, ptr addrspace(1) %p, align 4\n\
  tail call fastcc void @fence(i32 2)\n\
  store i32 2, ptr addrspace(1) %p, align 4\n\
  ret void\n}\n\
define internal fastcc void @fence(i32 %flags) #2 {\n\
  %t = trunc i32 %flags to i2\n\
  %r = tail call i2 @llvm.bitreverse.i2(i2 %t)\n\
  %m = zext i2 %r to i32\n\
  tail call void @air.atomic.fence(i32 %m, i32 3, i32 1) #1\n\
  ret void\n}\n\
declare void @air.atomic.fence(i32, i32, i32) #1\n\
declare i2 @llvm.bitreverse.i2(i2) #9\n\
attributes #1 = { nounwind willreturn }\n\
attributes #2 = { noinline nounwind }\n\
attributes #3 = { nounwind }\n\
!air.kernel = !{!0}\n\
!0 = !{ptr @store2, !1, !2}\n\
!1 = !{}\n\
!2 = !{!3}\n";

#[test]
fn a_barrier_and_a_fence_out_of_line_compute_what_metal_computes() {
    let scratch = Scratch::new("opencl-barrier");
    let device = device(&scratch.0);
    let start = vec![0u32; WORDS as usize];
    for (text, name) in [(BARRIER, "fence"), (FENCE, "store2")] {
        let text = format!("{text}!3 = {X}\n");
        let state = state(&device, &text, name);
        assert_eq!(
            run(&device, &state, &start, 32, |_| ()),
            vec![2; WORDS as usize],
            "{name}"
        );
    }
}
