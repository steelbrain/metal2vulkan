//! Kernels in the shapes Metal's compiler writes for OpenCL C (`newLibraryWithSource` with
//! `sourceLanguage` 1, as the GL renderer's compute path builds an OpenCL program), each run and
//! compared with what Metal computes.
//!
//! - **A barrier out of line**: OpenCL C's `barrier(flags)` stays an internal function passing its
//!   flags on to `air.wg.barrier`, and `mem_fence(flags)` computes `air.atomic.fence`'s flags from
//!   them through `llvm.bitreverse.i2`. Both made and run on an M3 Max (macOS 27.0.1) and an Intel
//!   UHD 630 (26.7.1): `x[i] = 1`, the barrier, `x[i] += x[i ^ 1]` stores 2 in every word.
//!
//! Under a device cfg, on every device of the cfg's kind. The modules are LLVM text written here;
//! `llvm-as` assembles them, and without it the test skips visibly.

use crate::tests::command::on_device::{each_device, finished};
use crate::tests::opencl_mode::{bytes_buffer, words};
use crate::{
    ComputePipelineDescriptor, ComputePipelineState, Device, DispatchType, FunctionDescriptor,
    Library, StageSource,
};
use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

/// The target, the work-item function, and the module's versions.
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
const WORDS: u32 = 64;

/// The library of the module `text` (after [`HEAD`]), or None where `llvm-as` is missing.
fn library(text: &str) -> Option<Library> {
    match llvm_assemble_text(&format!("{HEAD}{text}")) {
        Ok(air) => Some(Library::from_air(&air).expect("the library")),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(error) => panic!("the module: {error:?}"),
    }
}

/// The state of kernel `name` of `library`, a plain pipeline's.
fn state(device: &Device, library: &Library, name: &str) -> ComputePipelineState {
    let function = library.function(name).expect("the kernel");
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(function)));
    (device.new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers))
        .unwrap_or_else(|error| panic!("{name}: {error}"))
}

/// `state` run on `device` (named `name`) over [`WORDS`] threads in threadgroups of `group`, `x`
/// holding `init`; `x`'s words after.
fn run(
    name: &str,
    device: &Device,
    state: &ComputePipelineState,
    init: &[u32],
    group: u32,
) -> Vec<u32> {
    let bytes: Vec<u8> = init.iter().flat_map(|w| w.to_le_bytes()).collect();
    let x = bytes_buffer(device, &bytes);
    let queue = device.new_command_queue(1).expect("a queue");
    let command_buffer = queue.command_buffer().expect("a command buffer");
    let mut encoder =
        (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
    encoder
        .set_compute_pipeline_state(state)
        .expect("the state");
    encoder.set_buffer(Some(&x), 0, 0).expect("x");
    (encoder.dispatch_threads([WORDS, 1, 1], [group, 1, 1]))
        .unwrap_or_else(|error| panic!("{name}: the dispatch: {error}"));
    encoder.end_encoding();
    command_buffer.commit().expect("a commit");
    finished(name, &command_buffer);
    assert_eq!(command_buffer.error(), None, "{name}");
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
/// `mem_fence(CLK_GLOBAL_MEM_FENCE)` becomes `mem_device`, its order acq_rel (4), as Apple's
/// OpenCL compiler writes `mem_fence`: the probe opencl_operand_encodings, job m3-max-76799),
/// `x[i] = 2`.
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
  tail call void @air.atomic.fence(i32 %m, i32 4, i32 1) #1\n\
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
    let kernels = [(BARRIER, "fence"), (FENCE, "store2")];
    let Some(libraries) = kernels
        .iter()
        .map(|(text, _)| library(&format!("{text}!3 = {X}\n")))
        .collect::<Option<Vec<Library>>>()
    else {
        return;
    };
    each_device("opencl-barrier", |name, _, device| {
        let start = vec![0u32; WORDS as usize];
        for ((_, kernel), library) in kernels.iter().zip(&libraries) {
            let state = state(&device, library, kernel);
            assert_eq!(
                run(name, &device, &state, &start, 32),
                vec![2; WORDS as usize],
                "{name}: {kernel}"
            );
        }
        println!("{name}: the barrier and the fence out of line store 2 in every word");
    });
}
