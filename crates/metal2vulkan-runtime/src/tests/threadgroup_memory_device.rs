//! The device's `maxThreadgroupMemoryLength` is what a Mac of the class it presents answers, up to
//! its own `maxComputeSharedMemorySize`, and a kernel runs with all of it: one thread writes every
//! word of a threadgroup memory argument of exactly that length, then reads them back in reverse
//! order into a device buffer. On a RADV GPU presenting its own GPU's Mac (an AMD one) that is
//! 65536 bytes, where an Apple GPU's 32768 was answered before; on lavapipe its own 32768.
//!
//! The same device's answers of function pointers, dynamic libraries and rasterization rate maps
//! are printed and checked with it: Apple9's function pointer rows (exact where the device enables
//! the four descriptor-indexing features they need), its dynamic library rows (exact on every
//! device) and `supportsRasterizationRateMapWithLayerCount:` (yes for 1 and 2).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The module is LLVM text written here
//! (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::opencl_mode::{words, words_buffer};
use crate::{
    ComputePipelineState, Device, FamilyRow, FeatureRow, GpuFamily, MacClass, PipelineOptions,
    ResourceOptions, RowState, Size,
};
use metal2vulkan::device::VkFeature;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};

/// What each word of the threadgroup memory holds: its index, so marked.
const MARK: u32 = 0x5a5a_5a5a;

/// `fill(threadgroup uint *tg [[threadgroup(0)]], device uint *out [[buffer(0)]], constant uint
/// &n [[buffer(1)]])`: `tg[i] = i ^ MARK` for every `i < n`, then `out[j] = tg[n - 1 - j]`.
const KERNEL: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @fill(ptr addrspace(3) noundef %tg, ptr addrspace(1) noundef %out, ptr addrspace(2) noundef readonly %count) {\n\
entry:\n\
  %n = load i32, ptr addrspace(2) %count, align 4\n\
  %none = icmp eq i32 %n, 0\n\
  br i1 %none, label %done, label %write\n\
write:\n\
  %i = phi i32 [ 0, %entry ], [ %i.next, %write ]\n\
  %v = xor i32 %i, 1515870810\n\
  %i64 = zext i32 %i to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(3) %tg, i64 %i64\n\
  store i32 %v, ptr addrspace(3) %p, align 4\n\
  %i.next = add nuw i32 %i, 1\n\
  %more = icmp ult i32 %i.next, %n\n\
  br i1 %more, label %write, label %read\n\
read:\n\
  %j = phi i32 [ 0, %write ], [ %j.next, %read ]\n\
  %last = add i32 %n, -1\n\
  %k = sub i32 %last, %j\n\
  %k64 = zext i32 %k to i64\n\
  %q = getelementptr inbounds i32, ptr addrspace(3) %tg, i64 %k64\n\
  %w = load i32, ptr addrspace(3) %q, align 4\n\
  %j64 = zext i32 %j to i64\n\
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %j64\n\
  store i32 %w, ptr addrspace(1) %o, align 4\n\
  %j.next = add nuw i32 %j, 1\n\
  %again = icmp ult i32 %j.next, %n\n\
  br i1 %again, label %read, label %done\n\
done:\n\
  ret void\n\
}\n\n\
!llvm.module.flags = !{!0, !1}\n!air.kernel = !{!2}\n!air.version = !{!8}\n!air.language_version = !{!9}\n\
!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n\
!1 = !{i32 7, !\"air.max_threadgroup_buffers\", i32 31}\n\
!2 = !{ptr @fill, !3, !4}\n!3 = !{}\n!4 = !{!5, !6, !7}\n\
!5 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 3, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"tg\"}\n\
!6 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}\n\
!7 = !{i32 2, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"n\"}\n\
!8 = !{i32 2, i32 8, i32 0}\n!9 = !{!\"Metal\", i32 4, i32 0, i32 0}\n";

/// The state of `fill`.
fn state(device: &Device) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(KERNEL).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name("fill")
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The words `fill` writes with `length` bytes of threadgroup memory, in one threadgroup of one
/// thread.
fn run(device: &Device, state: &ComputePipelineState, length: u32) -> Vec<u32> {
    let n = length / 4;
    let out = device
        .new_buffer(u64::from(length), ResourceOptions::shared())
        .unwrap();
    let count = words_buffer(device, &[n]);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    encoder.set_buffer(Some(&count), 0, 1).unwrap();
    encoder.set_threadgroup_memory_length(length, 0).unwrap();
    let one = Size::new(1, 1, 1);
    encoder.dispatch_threadgroups(one, one).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    words(&out)
}

#[test]
fn a_kernel_takes_all_the_threadgroup_memory_its_mac_answers() {
    let scratch = Scratch::new("threadgroup-memory");
    let device = device(&scratch.0);
    let p = device.properties();
    let limit = u64::from(p.description.limits.max_compute_shared_memory_size);
    let mac = match p.mac_class() {
        MacClass::AppleSilicon => 32 << 10,
        MacClass::IntelOrAmd => 64 << 10,
    };
    let answer = p.max_threadgroup_memory_length;
    println!(
        "THREADGROUP-MEMORY {}: {:?}, maxComputeSharedMemorySize {limit}, \
         maxThreadgroupMemoryLength {answer}",
        p.name,
        p.mac_class()
    );
    assert_eq!(answer, limit.min(mac), "{}", p.name);

    let length = u32::try_from(answer).unwrap();
    let n = length / 4;
    let written = run(&device, &state(&device), length);
    let want: Vec<u32> = (0..n).rev().map(|i| i ^ MARK).collect();
    let wrong = written.iter().zip(&want).filter(|(a, b)| a != b).count();
    println!(
        "THREADGROUP-MEMORY {}: {n} words through {length} bytes, {wrong} wrong",
        p.name
    );
    assert_eq!(written.len(), want.len());
    assert_eq!(wrong, 0, "{}: first words {:x?}", p.name, &written[..8]);

    let apple9 = device.family_rows(GpuFamily::Apple9);
    let row = |r: FeatureRow| {
        apple9
            .iter()
            .find(|a| a.row == FamilyRow::Feature(r))
            .map(|a| a.state.clone())
    };
    let tables = [
        VkFeature::RuntimeDescriptorArray,
        VkFeature::DescriptorBindingPartiallyBound,
        VkFeature::ShaderStorageBufferArrayNonUniformIndexing,
        VkFeature::DescriptorBindingStorageBufferUpdateAfterBind,
    ]
    .into_iter()
    .all(|f| p.description.has(f));
    for r in [
        FeatureRow::FunctionPointersCompute,
        FeatureRow::FunctionPointersRender,
        FeatureRow::ComputeDynamicLibraries,
        FeatureRow::RenderDynamicLibraries,
    ] {
        let state = row(r);
        println!("THREADGROUP-MEMORY {}: Apple9 {r:?} {state:?}", p.name);
        let exact = match r {
            FeatureRow::FunctionPointersCompute | FeatureRow::FunctionPointersRender => tables,
            _ => true,
        };
        assert_eq!(state == Some(RowState::Exact), exact, "{} {r:?}", p.name);
    }
    for layers in [1, 2] {
        assert_eq!(
            p.supports_rasterization_rate_map_with_layer_count(layers),
            Ok(true),
            "{} {layers}",
            p.name
        );
    }
}
