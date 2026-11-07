//! OpenCL mode (`ComputePipelineDescriptor::open_cl_mode_enabled`): a kernel in the shape Apple's
//! OpenCL compiler writes (`__entry_<name>` calling the kernel, an internal function, which calls
//! the work-item functions through internal wrappers) compiles, reflects and records its mode as a
//! pipeline of OpenCL mode binds it, on every committed profile that compiles it.
//!
//! The reflection is the M3 Max's (macOS 27.0.1) for kernels of these shapes: the kernel's
//! arguments, each buffer 3 past its index and threadgroup memory at its own, then
//! `__global_offset_and_num_dims` (index 0, read-only `uint`, 4 bytes aligned to 4),
//! `__printf_buffer` (1, read-write `char`, 1 byte aligned to 1) and
//! `__image_order_data_type_table` (2, read-only `uint`), every binding used; an argument passed
//! by value is a read-only buffer of its own type, size and alignment, a struct's `Struct`. The
//! modules are LLVM text written here (`llvm-as`).

use crate::budget::Budget;
use crate::error::{DescriptorError, Error, Feature, LowerFamily};
use crate::library::Library;
use crate::lower::select::LowerInputs;
use crate::metal::{BindingAccess, BindingType, DataType};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::testing::AssembleError;
use crate::pipeline::{ComputePipelineDescriptor, FunctionDescriptor, StageSource};

use super::engine_argument_tables::device;

/// `__kernel void k(__global const float *a, __global float *c, __local float *tmp,
/// __constant uint *n)`: `tmp[l] = a[i]`, a barrier, `c[i] = tmp[l] + get_global_offset(0) +
/// get_work_dim() + n[0]` for `i = get_global_id(0)` and `l = get_local_id(0)`, in the shape the
/// OpenCL compiler writes it.
const MODULE: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v29-apple-macosx27.0.0\"\n\
define void @__entry_k(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(3) %2, ptr addrspace(2) %3) #0 {\n\
entry:\n\
  tail call fastcc void @k(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(3) %2, ptr addrspace(2) %3)\n\
  ret void\n}\n\
define internal fastcc void @k(ptr addrspace(1) %a, ptr addrspace(1) %c, ptr addrspace(3) %tmp, ptr addrspace(2) %n) #1 {\n\
entry:\n\
  %i = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %l = tail call fastcc i64 @_Z12get_local_idj(i32 0)\n\
  %pa = getelementptr inbounds float, ptr addrspace(1) %a, i64 %i\n\
  %va = load float, ptr addrspace(1) %pa, align 4\n\
  %pt = getelementptr inbounds float, ptr addrspace(3) %tmp, i64 %l\n\
  store float %va, ptr addrspace(3) %pt, align 4\n\
  tail call fastcc void @_Z7barrierj(i32 1)\n\
  %vt = load float, ptr addrspace(3) %pt, align 4\n\
  %o = tail call fastcc i64 @_Z17get_global_offsetj(i32 0)\n\
  %w = tail call fastcc i32 @_Z12get_work_dimv()\n\
  %vn = load i32, ptr addrspace(2) %n, align 4\n\
  %ow = trunc i64 %o to i32\n\
  %s1 = add i32 %ow, %w\n\
  %s2 = add i32 %s1, %vn\n\
  %f = uitofp i32 %s2 to float\n\
  %r = fadd float %vt, %f\n\
  %pc = getelementptr inbounds float, ptr addrspace(1) %c, i64 %i\n\
  store float %r, ptr addrspace(1) %pc, align 4\n\
  ret void\n}\n\
define internal fastcc i64 @_Z13get_global_idj(i32 %0) #2 {\n\
  %2 = tail call i32 @air.get_global_id.i32(i32 %0) #3\n\
  %3 = zext i32 %2 to i64\n\
  ret i64 %3\n}\n\
define internal fastcc i64 @_Z12get_local_idj(i32 %0) #2 {\n\
  %2 = tail call i32 @air.get_local_id.i32(i32 %0) #3\n\
  %3 = zext i32 %2 to i64\n\
  ret i64 %3\n}\n\
define internal fastcc i64 @_Z17get_global_offsetj(i32 %0) #2 {\n\
  %2 = tail call i32 @air.get_global_offset.i32(i32 %0) #3\n\
  %3 = zext i32 %2 to i64\n\
  ret i64 %3\n}\n\
define internal fastcc i32 @_Z12get_work_dimv() #2 {\n\
  %1 = tail call i32 @air.get_work_dim.i32() #3\n\
  ret i32 %1\n}\n\
define internal fastcc void @_Z7barrierj(i32 %0) #4 {\n\
  call void @air.wg.barrier(i32 %0, i32 5, i32 1) #5\n\
  ret void\n}\n\
declare i32 @air.get_global_id.i32(i32) #3\n\
declare i32 @air.get_local_id.i32(i32) #3\n\
declare i32 @air.get_global_offset.i32(i32) #3\n\
declare i32 @air.get_work_dim.i32() #3\n\
declare void @air.wg.barrier(i32, i32, i32) #5\n\
attributes #0 = { nounwind }\n\
attributes #1 = { convergent nounwind }\n\
attributes #2 = { nounwind optsize }\n\
attributes #3 = { nounwind readnone willreturn }\n\
attributes #4 = { convergent nounwind }\n\
attributes #5 = { convergent nounwind willreturn }\n\
!air.kernel = !{!0}\n\
!air.version = !{!7}\n\
!air.language_version = !{!8}\n\
!0 = !{ptr @__entry_k, !1, !2}\n\
!1 = !{}\n\
!2 = !{!3, !4, !5, !6}\n\
!3 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"a\"}\n\
!4 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"c\"}\n\
!5 = !{i32 2, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 3, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"tmp\"}\n\
!6 = !{i32 3, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"n\"}\n\
!7 = !{i32 2, i32 9, i32 0}\n\
!8 = !{!\"Metal\", i32 1, i32 1, i32 0}\n";

/// `__kernel void v(__global float *o, float k, struct S s)`, `struct S { float a; int b; }`:
/// `o[get_global_id(0)] = k + s.a`, each argument passed by value read from its constant pointer
/// as the OpenCL compiler writes it (the struct copied to a private `byval` copy).
const BY_VALUE: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v29-apple-macosx27.0.0\"\n\
%struct.S = type { float, i32 }\n\
define void @__entry_v(ptr addrspace(1) %0, ptr addrspace(2) %1, ptr addrspace(2) %2) #0 {\n\
entry:\n\
  %3 = alloca i64, align 8\n\
  %4 = load float, ptr addrspace(2) %1, align 4\n\
  %5 = load i64, ptr addrspace(2) %2, align 4\n\
  store i64 %5, ptr %3, align 8\n\
  tail call fastcc void @v(ptr addrspace(1) %0, float %4, ptr nonnull byval(%struct.S) %3)\n\
  ret void\n}\n\
define internal fastcc void @v(ptr addrspace(1) %o, float %k, ptr byval(%struct.S) align 4 %s) #1 {\n\
entry:\n\
  %i = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %pa = getelementptr inbounds %struct.S, ptr %s, i64 0, i32 0\n\
  %a = load float, ptr %pa, align 4\n\
  %r = fadd float %k, %a\n\
  %po = getelementptr inbounds float, ptr addrspace(1) %o, i64 %i\n\
  store float %r, ptr addrspace(1) %po, align 4\n\
  ret void\n}\n\
define internal fastcc i64 @_Z13get_global_idj(i32 %0) #2 {\n\
  %2 = tail call i32 @air.get_global_id.i32(i32 %0) #3\n\
  %3 = zext i32 %2 to i64\n\
  ret i64 %3\n}\n\
declare i32 @air.get_global_id.i32(i32) #3\n\
attributes #0 = { nounwind }\n\
attributes #1 = { convergent nounwind }\n\
attributes #2 = { nounwind optsize }\n\
attributes #3 = { nounwind readnone willreturn }\n\
!air.kernel = !{!0}\n\
!air.version = !{!7}\n\
!air.language_version = !{!8}\n\
!0 = !{ptr @__entry_v, !1, !2}\n\
!1 = !{}\n\
!2 = !{!3, !4, !5}\n\
!3 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"o\"}\n\
!4 = !{i32 1, !\"air.constant\", !\"air.location_index\", i32 1, i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"k\"}\n\
!5 = !{i32 2, !\"air.constant\", !\"air.location_index\", i32 2, i32 1, !\"air.struct_type_info\", !6, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"S\", !\"air.arg_name\", !\"s\"}\n\
!6 = !{i32 0, i32 4, i32 0, !\"float\", !\"a\", i32 4, i32 4, i32 0, !\"int\", !\"b\"}\n\
!7 = !{i32 2, i32 9, i32 0}\n\
!8 = !{!\"Metal\", i32 1, i32 1, i32 0}\n";

/// The kernel's function, or `None` (a skip) without `llvm-as`.
fn kernel() -> Option<FunctionDescriptor> {
    function(MODULE, "__entry_k")
}

/// Function `entry` of the module `text`, or `None` (a skip) without `llvm-as`.
fn function(text: &str, entry: &str) -> Option<FunctionDescriptor> {
    let bitcode = match metal2vulkan_devtools::llvm_assemble_text(text) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return None;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the module: {why}"),
    };
    let lib = Library::from_air(&bitcode).expect("library");
    Some(FunctionDescriptor::new(lib.function(entry).expect(entry)))
}

/// The compute descriptor of `kernel`, in OpenCL mode where `open_cl_mode`.
fn descriptor(kernel: StageSource, open_cl_mode: bool) -> ComputePipelineDescriptor {
    let mut d = ComputePipelineDescriptor::new(kernel);
    d.open_cl_mode_enabled = open_cl_mode;
    d
}

/// Each binding's name, type, index, access and use, and a buffer's data size, alignment and
/// data type.
type Row = (
    String,
    BindingType,
    u32,
    BindingAccess,
    bool,
    Option<(u64, u64, DataType)>,
);

fn rows(d: &ComputePipelineDescriptor) -> Vec<Row> {
    let r = d.reflection(&Budget::standard()).expect("reflection");
    r.bindings()
        .iter()
        .map(|b| {
            (
                b.name().to_owned(),
                b.binding_type(),
                b.index(),
                b.access(),
                b.is_used(),
                b.buffer().map(|x| {
                    (
                        x.buffer_data_size(),
                        x.buffer_alignment(),
                        x.buffer_data_type(),
                    )
                }),
            )
        })
        .collect()
}

#[test]
fn an_opencl_mode_pipeline_reflects_its_arguments_past_the_hidden_buffers() {
    let Some(k) = kernel() else {
        return;
    };
    let buffer = |name: &str, index, access, size, align, ty| {
        (
            name.to_owned(),
            BindingType::Buffer,
            index,
            access,
            true,
            Some((size, align, ty)),
        )
    };
    let tmp = (
        "tmp".to_owned(),
        BindingType::ThreadgroupMemory,
        0,
        BindingAccess::ReadWrite,
        true,
        None,
    );
    let float = DataType::Float;
    assert_eq!(
        rows(&descriptor(StageSource::Function(k.clone()), true)),
        [
            buffer("a", 3, BindingAccess::ReadOnly, 4, 4, float),
            buffer("c", 4, BindingAccess::ReadWrite, 4, 4, float),
            tmp.clone(),
            buffer("n", 5, BindingAccess::ReadOnly, 4, 4, DataType::UInt),
            buffer(
                "__global_offset_and_num_dims",
                0,
                BindingAccess::ReadOnly,
                4,
                4,
                DataType::UInt
            ),
            buffer(
                "__printf_buffer",
                1,
                BindingAccess::ReadWrite,
                1,
                1,
                DataType::Char
            ),
            buffer(
                "__image_order_data_type_table",
                2,
                BindingAccess::ReadOnly,
                4,
                4,
                DataType::UInt
            ),
        ]
    );
    // Without the mode the arguments are where their nodes put them, and no buffer is hidden.
    assert_eq!(
        rows(&descriptor(StageSource::Function(k.clone()), false)),
        [
            buffer("a", 0, BindingAccess::ReadOnly, 4, 4, float),
            buffer("c", 1, BindingAccess::ReadWrite, 4, 4, float),
            tmp,
            buffer("n", 2, BindingAccess::ReadOnly, 4, 4, DataType::UInt),
        ]
    );
    // Its slot writes name the same slots: the kernel's at their moved indices, then the hidden
    // buffers'.
    let writes = descriptor(StageSource::Function(k), true)
        .slot_writes(&Budget::standard())
        .expect("slot writes");
    let indices: Vec<_> = writes
        .kernel_slots()
        .iter()
        .map(|w| (w.name.clone(), w.index))
        .collect();
    assert_eq!(
        indices,
        [
            ("a".to_owned(), Some(3)),
            ("c".to_owned(), Some(4)),
            ("tmp".to_owned(), Some(0)),
            ("n".to_owned(), Some(5)),
            ("__global_offset_and_num_dims".to_owned(), Some(0)),
            ("__printf_buffer".to_owned(), Some(1)),
            ("__image_order_data_type_table".to_owned(), Some(2)),
        ]
    );
}

/// In OpenCL mode the kernel compiles, its key apart from the plain one's, and its artifact
/// records the mode; a compiled kernel serves only a pipeline of the mode it was compiled in
/// (`InvalidDescriptor(OpenClMode)`). Without the mode, `get_global_offset` and `get_work_dim`
/// have no value to read, and the compile refuses them, as Metal refuses a plain pipeline of the
/// kernel.
#[test]
fn an_opencl_mode_kernel_compiles_in_its_mode_alone() {
    let Some(k) = kernel() else {
        return;
    };
    let dev = device("moltenvk-m3");
    let b = Budget::standard();
    let cl = dev
        .compile_compute_pipeline(&descriptor(StageSource::Function(k.clone()), true), &b)
        .expect("an OpenCL-mode pipeline");
    let plain_key = dev.compute_pipeline_key(&descriptor(StageSource::Function(k.clone()), false));
    assert_ne!(cl.key, plain_key);
    let refused = dev
        .compile_compute_pipeline(&descriptor(StageSource::Function(k.clone()), false), &b)
        .map(|_| ());
    match refused {
        Err(Error::Unsupported(u)) => {
            assert_eq!(u.feature, Feature::Lowering(LowerFamily::Dispatch), "{u:?}");
        }
        other => panic!("{other:?}"),
    }
    let mut inputs = LowerInputs::new();
    inputs.open_cl_mode = true;
    let compiled = dev.compile_function(&k, &inputs, &b).expect("compiled");
    assert!(compiled.artifact().facts.contains(&RuntimeFact::OpenClMode));
    let pipeline = |open_cl_mode| {
        dev.compile_compute_pipeline(
            &descriptor(StageSource::Compiled(compiled.clone()), open_cl_mode),
            &b,
        )
        .map(|_| ())
    };
    assert_eq!(pipeline(true), Ok(()));
    assert_eq!(
        pipeline(false),
        Err(Error::InvalidDescriptor(DescriptorError::OpenClMode {
            compiled: true
        }))
    );
}

/// An argument passed by value reflects as the M3 Max reflects it in OpenCL mode: a read-only
/// buffer 3 past its index, of its own type, size and alignment, a struct's data type `Struct`
/// (size 8, alignment 4 for `struct { float; int; }`). The kernel compiles.
#[test]
fn an_opencl_mode_argument_by_value_reflects_as_a_constant_buffer_of_its_type() {
    let Some(v) = function(BY_VALUE, "__entry_v") else {
        return;
    };
    let rows = rows(&descriptor(StageSource::Function(v.clone()), true));
    let own: Vec<_> = rows[..3]
        .iter()
        .map(|(name, ty, index, access, used, buffer)| {
            (name.as_str(), *ty, *index, *access, *used, *buffer)
        })
        .collect();
    assert_eq!(
        own,
        [
            (
                "o",
                BindingType::Buffer,
                3,
                BindingAccess::ReadWrite,
                true,
                Some((4, 4, DataType::Float))
            ),
            (
                "k",
                BindingType::Buffer,
                4,
                BindingAccess::ReadOnly,
                true,
                Some((4, 4, DataType::Float))
            ),
            (
                "s",
                BindingType::Buffer,
                5,
                BindingAccess::ReadOnly,
                true,
                Some((8, 4, DataType::Struct))
            ),
        ]
    );
    assert_eq!(rows.len(), 6);
    device("moltenvk-m3")
        .compile_compute_pipeline(
            &descriptor(StageSource::Function(v), true),
            &Budget::standard(),
        )
        .expect("an OpenCL-mode pipeline");
}
