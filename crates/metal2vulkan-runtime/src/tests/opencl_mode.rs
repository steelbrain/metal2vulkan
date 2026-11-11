//! Compute pipelines of OpenCL mode (`ComputePipelineDescriptor::open_cl_mode_enabled`), each
//! kernel in the shape Apple's OpenCL compiler writes for OpenCL C (`__entry_<name>` calling the
//! kernel, an internal function, which calls the work-item functions through internal wrappers of
//! `air.get_*`), run with the three hidden buffers bound as Metal's OpenCL dispatcher binds them,
//! and compared with what the M3 Max (macOS 27.0.1) computes for the same kernels:
//!
//! - **add**: `c[i] = a[i] + b[i]` over 1024 threads, `a[i] = i`, `b[i] = 2i + 1`: `3i + 1`.
//! - **scale**: `b[i] *= k` for `i < n`, `k` and `n` passed by value, over 128 threads, `b[i] = i`:
//!   `k = 2.5, n = 100` set as bytes, then `k = 3, n = 50` in a buffer at offsets 16 and 32.
//! - **st**: `o[i] = s.a + s.b` for a struct passed by value, `s = {2.5, 4}`: 6.5.
//! - **wg**: `o[i] = get_local_size(0)` of `reqd_work_group_size(64, 1, 1)`: a pipeline of at most
//!   64 threads a threadgroup, which answers 64 in threadgroups of 64 and 32 in threadgroups of
//!   32.
//! - **lsum**: each threadgroup of 64 sums its `in[i] = i` through `__local` memory and a barrier:
//!   2016, 6112, 10208, 14304.
//! - **cmul**: `o[i] *= c[i & 3]` through a `__constant` pointer, `o[i] = i + 1`, `c = 1, 2, 3, 4`:
//!   1, 4, 9, 16, 5, 12, ...
//! - **items**: every work-item function, of the dimensions it names and of one passed by value,
//!   over a grid of 8 by 4 threads in threadgroups of 4 by 2 at offset (3, 5), of
//!   two dimensions. The M3 Max answers each word as OpenCL defines it, but for the store of
//!   `get_global_id(3) + get_local_size(4) + get_global_size(5)`, which it leaves unwritten; the
//!   crate stores OpenCL's 2.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. The modules are LLVM text written here
//! (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use crate::{Buffer, ComputePipelineState, Device, PipelineOptions, ResourceOptions, Size};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};

/// The target, the work-item functions' wrappers as the OpenCL compiler writes them, and the
/// module's versions.
const HEAD: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v29-apple-macosx27.0.0\"\n\
define internal fastcc i64 @_Z13get_global_idj(i32 %0) #2 {\n\
  %2 = tail call i32 @air.get_global_id.i32(i32 %0) #3\n\
  %3 = zext i32 %2 to i64\n\
  ret i64 %3\n}\n\
define internal fastcc i64 @_Z12get_local_idj(i32 %0) #2 {\n\
  %2 = tail call i32 @air.get_local_id.i32(i32 %0) #3\n\
  %3 = zext i32 %2 to i64\n\
  ret i64 %3\n}\n\
define internal fastcc i64 @_Z12get_group_idj(i32 %0) #2 {\n\
  %2 = tail call i32 @air.get_group_id.i32(i32 %0) #3\n\
  %3 = zext i32 %2 to i64\n\
  ret i64 %3\n}\n\
define internal fastcc i64 @_Z14get_local_sizej(i32 %0) #2 {\n\
  %2 = tail call i32 @air.get_local_size.i32(i32 %0) #3\n\
  %3 = zext i32 %2 to i64\n\
  ret i64 %3\n}\n\
define internal fastcc i64 @_Z15get_global_sizej(i32 %0) #2 {\n\
  %2 = tail call i32 @air.get_global_size.i32(i32 %0) #3\n\
  %3 = zext i32 %2 to i64\n\
  ret i64 %3\n}\n\
define internal fastcc i64 @_Z14get_num_groupsj(i32 %0) #2 {\n\
  %2 = tail call i32 @air.get_num_groups.i32(i32 %0) #3\n\
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
declare i32 @air.get_group_id.i32(i32) #3\n\
declare i32 @air.get_local_size.i32(i32) #3\n\
declare i32 @air.get_global_size.i32(i32) #3\n\
declare i32 @air.get_num_groups.i32(i32) #3\n\
declare i32 @air.get_global_offset.i32(i32) #3\n\
declare i32 @air.get_work_dim.i32() #3\n\
declare void @air.wg.barrier(i32, i32, i32) #5\n\
attributes #0 = { nounwind }\n\
attributes #1 = { convergent nounwind }\n\
attributes #2 = { nounwind optsize }\n\
attributes #3 = { nounwind readnone willreturn }\n\
attributes #4 = { convergent nounwind }\n\
attributes #5 = { convergent nounwind willreturn }\n\
!air.version = !{!90}\n\
!air.language_version = !{!91}\n\
!90 = !{i32 2, i32 9, i32 0}\n\
!91 = !{!\"Metal\", i32 1, i32 1, i32 0}\n";

/// The node of argument `n`, a buffer at `index` of address space `space` holding `ty`, `access`
/// one of `read`, `read_write`.
pub(super) fn buffer(n: u32, index: u32, access: &str, space: u32, ty: &str, name: &str) -> String {
    format!(
        "!{{i32 {n}, !\"air.buffer\", !\"air.location_index\", i32 {index}, i32 1, !\"air.{access}\", !\"air.address_space\", i32 {space}, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"{ty}\", !\"air.arg_name\", !\"{name}\"}}"
    )
}

/// The node of argument `n`, passed by value at `index`: `size` bytes of `ty`, its layout the node
/// `layout` names where `Some`.
fn constant(n: u32, index: u32, size: u32, ty: &str, name: &str, layout: Option<&str>) -> String {
    let layout = layout
        .map(|l| format!(" !\"air.struct_type_info\", {l},"))
        .unwrap_or_default();
    format!(
        "!{{i32 {n}, !\"air.constant\", !\"air.location_index\", i32 {index}, i32 1,{layout} !\"air.arg_type_size\", i32 {size}, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"{ty}\", !\"air.arg_name\", !\"{name}\"}}"
    )
}

/// The kernel list of `entry` with argument nodes `args`.
pub(super) fn kernel_list(entry: &str, args: &[String]) -> String {
    kernel_list_with(entry, args, "")
}

/// The kernel list of `entry` with argument nodes `args` and the trailing entry attributes
/// `attributes` (each `, !n`).
fn kernel_list_with(entry: &str, args: &[String], attributes: &str) -> String {
    let refs: Vec<String> = (0..args.len()).map(|i| format!("!{}", 3 + i)).collect();
    let mut out = format!(
        "!air.kernel = !{{!0}}\n!0 = !{{ptr @{entry}, !1, !2{attributes}}}\n!1 = !{{}}\n!2 = !{{{}}}\n",
        refs.join(", ")
    );
    for (i, a) in args.iter().enumerate() {
        out.push_str(&format!("!{} = {a}\n", 3 + i));
    }
    out
}

/// `__kernel void add(__global const float *a, __global const float *b, __global float *c)`.
fn add() -> String {
    let body = "define void @__entry_add(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2) #0 {\n\
entry:\n\
  tail call fastcc void @add(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2)\n\
  ret void\n}\n\
define internal fastcc void @add(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c) #1 {\n\
entry:\n\
  %i = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %pa = getelementptr inbounds float, ptr addrspace(1) %a, i64 %i\n\
  %va = load float, ptr addrspace(1) %pa, align 4\n\
  %pb = getelementptr inbounds float, ptr addrspace(1) %b, i64 %i\n\
  %vb = load float, ptr addrspace(1) %pb, align 4\n\
  %s = fadd float %va, %vb\n\
  %pc = getelementptr inbounds float, ptr addrspace(1) %c, i64 %i\n\
  store float %s, ptr addrspace(1) %pc, align 4\n\
  ret void\n}\n";
    let args = [
        buffer(0, 0, "read", 1, "float", "a"),
        buffer(1, 1, "read", 1, "float", "b"),
        buffer(2, 2, "read_write", 1, "float", "c"),
    ];
    format!("{body}{}", kernel_list("__entry_add", &args))
}

/// `__kernel void scale(__global float *b, float k, uint n)`: `if (i < n) b[i] *= k`.
fn scale() -> String {
    let body = "define void @__entry_scale(ptr addrspace(1) %0, ptr addrspace(2) %1, ptr addrspace(2) %2) #0 {\n\
entry:\n\
  %3 = load float, ptr addrspace(2) %1, align 4\n\
  %4 = load i32, ptr addrspace(2) %2, align 4\n\
  tail call fastcc void @scale(ptr addrspace(1) %0, float %3, i32 %4)\n\
  ret void\n}\n\
define internal fastcc void @scale(ptr addrspace(1) %b, float %k, i32 %n) #1 {\n\
entry:\n\
  %i = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %wide = zext i32 %n to i64\n\
  %in = icmp ult i64 %i, %wide\n\
  br i1 %in, label %then, label %end\n\
then:\n\
  %p = getelementptr inbounds float, ptr addrspace(1) %b, i64 %i\n\
  %v = load float, ptr addrspace(1) %p, align 4\n\
  %m = fmul float %v, %k\n\
  store float %m, ptr addrspace(1) %p, align 4\n\
  br label %end\n\
end:\n\
  ret void\n}\n";
    let args = [
        buffer(0, 0, "read_write", 1, "float", "b"),
        constant(1, 1, 4, "float", "k", None),
        constant(2, 2, 4, "uint", "n", None),
    ];
    format!("{body}{}", kernel_list("__entry_scale", &args))
}

/// `__kernel void st(__global float *o, struct S s)`, `struct S { float a; int b; }`: `o[i] = s.a +
/// s.b`, the entry copying the struct to the private `byval` copy the kernel reads.
fn st() -> String {
    let body = "%struct.S = type { float, i32 }\n\
define void @__entry_st(ptr addrspace(1) %0, ptr addrspace(2) %1) #0 {\n\
entry:\n\
  %2 = alloca i64, align 8\n\
  %3 = load i64, ptr addrspace(2) %1, align 4\n\
  store i64 %3, ptr %2, align 8\n\
  tail call fastcc void @st(ptr addrspace(1) %0, ptr nonnull byval(%struct.S) %2)\n\
  ret void\n}\n\
define internal fastcc void @st(ptr addrspace(1) %o, ptr byval(%struct.S) align 4 %s) #1 {\n\
entry:\n\
  %i = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %pa = getelementptr inbounds %struct.S, ptr %s, i64 0, i32 0\n\
  %a = load float, ptr %pa, align 4\n\
  %pb = getelementptr inbounds %struct.S, ptr %s, i64 0, i32 1\n\
  %b = load i32, ptr %pb, align 4\n\
  %fb = tail call float @air.convert.f.f32.s.i32(i32 %b) #3\n\
  %r = fadd float %a, %fb\n\
  %po = getelementptr inbounds float, ptr addrspace(1) %o, i64 %i\n\
  store float %r, ptr addrspace(1) %po, align 4\n\
  ret void\n}\n\
declare float @air.convert.f.f32.s.i32(i32) #3\n\
!50 = !{i32 0, i32 4, i32 0, !\"float\", !\"a\", i32 4, i32 4, i32 0, !\"int\", !\"b\"}\n";
    let args = [
        buffer(0, 0, "read_write", 1, "float", "o"),
        constant(1, 1, 8, "S", "s", Some("!50")),
    ];
    format!("{body}{}", kernel_list("__entry_st", &args))
}

/// `__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void wg(__global float *o)`:
/// `o[get_global_id(0)] = get_local_size(0)`.
fn wg() -> String {
    let body = "define void @__entry_wg(ptr addrspace(1) %0) #0 {\n\
entry:\n\
  tail call fastcc void @wg(ptr addrspace(1) %0)\n\
  ret void\n}\n\
define internal fastcc void @wg(ptr addrspace(1) %o) #1 {\n\
entry:\n\
  %i = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %n = tail call fastcc i64 @_Z14get_local_sizej(i32 0)\n\
  %f = uitofp i64 %n to float\n\
  %po = getelementptr inbounds float, ptr addrspace(1) %o, i64 %i\n\
  store float %f, ptr addrspace(1) %po, align 4\n\
  ret void\n}\n\
!51 = !{!\"air.work_group_size\", i32 64, i32 1, i32 1}\n";
    let args = [buffer(0, 0, "read_write", 1, "float", "o")];
    format!("{body}{}", kernel_list_with("__entry_wg", &args, ", !51"))
}

/// `__kernel void lsum(__global const float *in, __global float *out, __local float *tmp)`:
/// `tmp[l] = in[i]`, `barrier(CLK_LOCAL_MEM_FENCE)`, and thread 0 of each group stores the sum of
/// its group's `tmp` to `out[get_group_id(0)]`.
fn lsum() -> String {
    let body = "define void @__entry_lsum(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(3) %2) #0 {\n\
entry:\n\
  tail call fastcc void @lsum(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(3) %2)\n\
  ret void\n}\n\
define internal fastcc void @lsum(ptr addrspace(1) %in, ptr addrspace(1) %out, ptr addrspace(3) %tmp) #1 {\n\
entry:\n\
  %l = tail call fastcc i64 @_Z12get_local_idj(i32 0)\n\
  %g = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %pi = getelementptr inbounds float, ptr addrspace(1) %in, i64 %g\n\
  %v = load float, ptr addrspace(1) %pi, align 4\n\
  %pt = getelementptr inbounds float, ptr addrspace(3) %tmp, i64 %l\n\
  store float %v, ptr addrspace(3) %pt, align 4\n\
  tail call fastcc void @_Z7barrierj(i32 1)\n\
  %first = icmp eq i64 %l, 0\n\
  br i1 %first, label %loop, label %done\n\
loop:\n\
  %s = phi float [ %s1, %body ], [ 0.000000e+00, %entry ]\n\
  %j = phi i64 [ %j1, %body ], [ 0, %entry ]\n\
  %n = tail call fastcc i64 @_Z14get_local_sizej(i32 0)\n\
  %more = icmp ult i64 %j, %n\n\
  br i1 %more, label %body, label %exit\n\
body:\n\
  %pj = getelementptr inbounds float, ptr addrspace(3) %tmp, i64 %j\n\
  %vj = load float, ptr addrspace(3) %pj, align 4\n\
  %s1 = fadd float %s, %vj\n\
  %j1 = add nuw i64 %j, 1\n\
  br label %loop\n\
exit:\n\
  %grp = tail call fastcc i64 @_Z12get_group_idj(i32 0)\n\
  %po = getelementptr inbounds float, ptr addrspace(1) %out, i64 %grp\n\
  store float %s, ptr addrspace(1) %po, align 4\n\
  br label %done\n\
done:\n\
  ret void\n}\n";
    let args = [
        buffer(0, 0, "read", 1, "float", "in"),
        buffer(1, 1, "read_write", 1, "float", "out"),
        buffer(2, 0, "read_write", 3, "float", "tmp"),
    ];
    format!("{body}{}", kernel_list("__entry_lsum", &args))
}

/// `__kernel void cmul(__global float *o, __constant float *c)`: `o[i] *= c[i & 3]`.
fn cmul() -> String {
    let body = "define void @__entry_cmul(ptr addrspace(1) %0, ptr addrspace(2) %1) #0 {\n\
entry:\n\
  tail call fastcc void @cmul(ptr addrspace(1) %0, ptr addrspace(2) %1)\n\
  ret void\n}\n\
define internal fastcc void @cmul(ptr addrspace(1) %o, ptr addrspace(2) %c) #1 {\n\
entry:\n\
  %i = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %po = getelementptr inbounds float, ptr addrspace(1) %o, i64 %i\n\
  %vo = load float, ptr addrspace(1) %po, align 4\n\
  %k = and i64 %i, 3\n\
  %pc = getelementptr inbounds float, ptr addrspace(2) %c, i64 %k\n\
  %vc = load float, ptr addrspace(2) %pc, align 4\n\
  %m = fmul float %vo, %vc\n\
  store float %m, ptr addrspace(1) %po, align 4\n\
  ret void\n}\n";
    let args = [
        buffer(0, 0, "read_write", 1, "float", "o"),
        buffer(1, 1, "read", 2, "float", "c"),
    ];
    format!("{body}{}", kernel_list("__entry_cmul", &args))
}

/// `__kernel void items(__global uint *o, uint d)`: thread `i` (its linear index in the grid) stores eight words at `o[8i]`: `get_global_id(0)`,
/// `get_global_id(1)`, the local ids and group ids of dimensions 0 and 1 a byte each, the group
/// counts and local sizes so, the global sizes and `get_work_dim()` so, the sum of
/// `get_global_id`, `get_local_size`, `get_num_groups` and `get_global_offset` of `d`,
/// `get_global_id(3) + get_local_size(4) + get_global_size(5)`, and 7.
fn items() -> String {
    let body = "define void @__entry_items(ptr addrspace(1) %0, ptr addrspace(2) %1) #0 {\n\
entry:\n\
  %2 = load i32, ptr addrspace(2) %1, align 4\n\
  tail call fastcc void @items(ptr addrspace(1) %0, i32 %2)\n\
  ret void\n}\n\
define internal fastcc void @items(ptr addrspace(1) %o, i32 %d) #1 {\n\
entry:\n\
  %g1 = tail call fastcc i64 @_Z13get_global_idj(i32 1)\n\
  %f1 = tail call fastcc i64 @_Z17get_global_offsetj(i32 1)\n\
  %y = sub i64 %g1, %f1\n\
  %gs0 = tail call fastcc i64 @_Z15get_global_sizej(i32 0)\n\
  %row = mul i64 %y, %gs0\n\
  %g0 = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %f0 = tail call fastcc i64 @_Z17get_global_offsetj(i32 0)\n\
  %x = sub i64 %g0, %f0\n\
  %i = add i64 %row, %x\n\
  %base = shl i64 %i, 3\n\
  %w0 = trunc i64 %g0 to i32\n\
  %p0 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 %base\n\
  store i32 %w0, ptr addrspace(1) %p0, align 4\n\
  %w1 = trunc i64 %g1 to i32\n\
  %a1 = or i64 %base, 1\n\
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 %a1\n\
  store i32 %w1, ptr addrspace(1) %p1, align 4\n\
  %l0 = tail call fastcc i64 @_Z12get_local_idj(i32 0)\n\
  %l1 = tail call fastcc i64 @_Z12get_local_idj(i32 1)\n\
  %l1s = shl i64 %l1, 8\n\
  %q0 = or i64 %l1s, %l0\n\
  %r0 = tail call fastcc i64 @_Z12get_group_idj(i32 0)\n\
  %r0s = shl i64 %r0, 16\n\
  %q1 = or i64 %q0, %r0s\n\
  %r1 = tail call fastcc i64 @_Z12get_group_idj(i32 1)\n\
  %r1s = shl i64 %r1, 24\n\
  %q2 = or i64 %q1, %r1s\n\
  %w2 = trunc i64 %q2 to i32\n\
  %a2 = or i64 %base, 2\n\
  %p2 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 %a2\n\
  store i32 %w2, ptr addrspace(1) %p2, align 4\n\
  %n0 = tail call fastcc i64 @_Z14get_num_groupsj(i32 0)\n\
  %n1 = tail call fastcc i64 @_Z14get_num_groupsj(i32 1)\n\
  %n1s = shl i64 %n1, 8\n\
  %u0 = or i64 %n1s, %n0\n\
  %z0 = tail call fastcc i64 @_Z14get_local_sizej(i32 0)\n\
  %z0s = shl i64 %z0, 16\n\
  %u1 = or i64 %u0, %z0s\n\
  %z1 = tail call fastcc i64 @_Z14get_local_sizej(i32 1)\n\
  %z1s = shl i64 %z1, 24\n\
  %u2 = or i64 %u1, %z1s\n\
  %w3 = trunc i64 %u2 to i32\n\
  %a3 = or i64 %base, 3\n\
  %p3 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 %a3\n\
  store i32 %w3, ptr addrspace(1) %p3, align 4\n\
  %gs1 = tail call fastcc i64 @_Z15get_global_sizej(i32 1)\n\
  %gs1s = shl i64 %gs1, 8\n\
  %v0 = or i64 %gs1s, %gs0\n\
  %wd = tail call fastcc i32 @_Z12get_work_dimv()\n\
  %wds = shl i32 %wd, 16\n\
  %v0t = trunc i64 %v0 to i32\n\
  %w4 = or i32 %wds, %v0t\n\
  %a4 = or i64 %base, 4\n\
  %p4 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 %a4\n\
  store i32 %w4, ptr addrspace(1) %p4, align 4\n\
  %dg = tail call fastcc i64 @_Z13get_global_idj(i32 %d)\n\
  %dl = tail call fastcc i64 @_Z14get_local_sizej(i32 %d)\n\
  %s0 = add i64 %dl, %dg\n\
  %dn = tail call fastcc i64 @_Z14get_num_groupsj(i32 %d)\n\
  %s1 = add i64 %s0, %dn\n\
  %df = tail call fastcc i64 @_Z17get_global_offsetj(i32 %d)\n\
  %s2 = add i64 %s1, %df\n\
  %w5 = trunc i64 %s2 to i32\n\
  %a5 = or i64 %base, 5\n\
  %p5 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 %a5\n\
  store i32 %w5, ptr addrspace(1) %p5, align 4\n\
  %e0 = tail call fastcc i64 @_Z13get_global_idj(i32 3)\n\
  %e1 = tail call fastcc i64 @_Z14get_local_sizej(i32 4)\n\
  %t0 = add i64 %e1, %e0\n\
  %e2 = tail call fastcc i64 @_Z15get_global_sizej(i32 5)\n\
  %t1 = add i64 %t0, %e2\n\
  %w6 = trunc i64 %t1 to i32\n\
  %a6 = or i64 %base, 6\n\
  %p6 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 %a6\n\
  store i32 %w6, ptr addrspace(1) %p6, align 4\n\
  tail call fastcc void @_Z7barrierj(i32 2)\n\
  %a7 = or i64 %base, 7\n\
  %p7 = getelementptr inbounds i32, ptr addrspace(1) %o, i64 %a7\n\
  store i32 7, ptr addrspace(1) %p7, align 4\n\
  ret void\n}\n";
    let args = [
        buffer(0, 0, "read_write", 1, "uint", "o"),
        constant(1, 1, 4, "uint", "d", None),
    ];
    format!("{body}{}", kernel_list("__entry_items", &args))
}

/// The state of `entry` of the module `text` (after [`HEAD`]), of OpenCL mode.
pub(super) fn state(device: &Device, text: &str, entry: &str) -> ComputePipelineState {
    let air = metal2vulkan_devtools::llvm_assemble_text(&format!("{HEAD}{text}")).unwrap();
    let library = metal2vulkan::library::Library::from_air(&air).unwrap();
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name(entry)
        .unwrap();
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.open_cl_mode_enabled = true;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A shared buffer of `words`.
pub(super) fn words_buffer(device: &Device, words: &[u32]) -> Buffer {
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    device
        .new_buffer_with_bytes(&bytes, ResourceOptions::shared())
        .unwrap()
}

/// A shared buffer of `floats`.
pub(super) fn floats_buffer(device: &Device, floats: &[f32]) -> Buffer {
    words_buffer(
        device,
        &floats.iter().map(|f| f.to_bits()).collect::<Vec<_>>(),
    )
}

/// The words of `buffer`, which no command uses now.
pub(super) fn words(buffer: &Buffer) -> Vec<u32> {
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

/// The floats of `buffer`, which no command uses now.
pub(super) fn floats(buffer: &Buffer) -> Vec<f32> {
    words(buffer).into_iter().map(f32::from_bits).collect()
}

/// One of a kernel's own arguments, as bound.
#[derive(Clone, Copy)]
enum Arg<'a> {
    /// A buffer from a byte offset (`setBuffer:offset:atIndex:`).
    Buffer(&'a Buffer, u64),
    /// Bytes (`setBytes:length:atIndex:`).
    Bytes(&'a [u8]),
}

/// `buffer` from its start.
fn whole(buffer: &Buffer) -> Arg<'_> {
    Arg::Buffer(buffer, 0)
}

/// `state` dispatched over `grid` threads in threadgroups of `group`, buffer 0 holding the global
/// offset `offset` and the dimension count `dims`, buffers 1 and 2 bound as the dispatcher binds
/// them, `args` at 3 and up, and `local` bytes of threadgroup memory at index 0 where `Some`.
fn run(
    device: &Device,
    state: &ComputePipelineState,
    (grid, group): (Size, Size),
    (offset, dims): ([u32; 3], u32),
    args: &[Arg<'_>],
    local: Option<u32>,
) {
    let hidden = [
        words_buffer(device, &[offset[0], offset[1], offset[2], dims]),
        words_buffer(device, &[0; 16]),
        words_buffer(device, &[0; 16]),
    ];
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    for (index, b) in (0u32..).zip(&hidden) {
        encoder.set_buffer(Some(b), 0, index).unwrap();
    }
    for (index, arg) in (3u32..).zip(args) {
        match *arg {
            Arg::Buffer(b, offset) => encoder.set_buffer(Some(b), offset, index).unwrap(),
            Arg::Bytes(bytes) => encoder.set_bytes(bytes, index).unwrap(),
        }
    }
    if let Some(bytes) = local {
        encoder.set_threadgroup_memory_length(bytes, 0).unwrap();
    }
    encoder.dispatch_threads(grid, group).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
}

/// One dimension's grid.
fn line(threads: u64, group: u64) -> (Size, Size) {
    (Size::new(threads, 1, 1), Size::new(group, 1, 1))
}

#[test]
fn opencl_mode_kernels_compute_what_metal_computes() {
    let scratch = Scratch::new("opencl-mode");
    let device = device(&scratch.0);
    let one = ([0; 3], 1);

    const N: usize = 1024;
    let a = floats_buffer(&device, &(0..N).map(|i| i as f32).collect::<Vec<_>>());
    let b = floats_buffer(
        &device,
        &(0..N).map(|i| (2 * i + 1) as f32).collect::<Vec<_>>(),
    );
    let c = floats_buffer(&device, &[0.0; N]);
    let add_state = state(&device, &add(), "__entry_add");
    run(
        &device,
        &add_state,
        line(N as u64, 64),
        one,
        &[whole(&a), whole(&b), whole(&c)],
        None,
    );
    let sums: Vec<f32> = (0..N).map(|i| (3 * i + 1) as f32).collect();
    assert_eq!(floats(&c), sums, "add");

    let input = floats_buffer(&device, &(0..256).map(|i| i as f32).collect::<Vec<_>>());
    let out = floats_buffer(&device, &[0.0; 4]);
    let lsum_state = state(&device, &lsum(), "__entry_lsum");
    run(
        &device,
        &lsum_state,
        line(256, 64),
        one,
        &[whole(&input), whole(&out)],
        Some(64 * 4),
    );
    assert_eq!(floats(&out), [2016.0, 6112.0, 10208.0, 14304.0], "lsum");

    let o = floats_buffer(
        &device,
        &(0..64).map(|i| (i + 1) as f32).collect::<Vec<_>>(),
    );
    let k = floats_buffer(&device, &[1.0, 2.0, 3.0, 4.0]);
    let cmul_state = state(&device, &cmul(), "__entry_cmul");
    run(
        &device,
        &cmul_state,
        line(64, 32),
        one,
        &[whole(&o), whole(&k)],
        None,
    );
    let products: Vec<f32> = (0..64).map(|i| ((i + 1) * (i % 4 + 1)) as f32).collect();
    assert_eq!(floats(&o), products, "cmul");
    assert_eq!(floats(&o)[..6], [1.0, 4.0, 9.0, 16.0, 5.0, 12.0]);

    let ramp: Vec<f32> = (0..128).map(|i| i as f32).collect();
    let scaled = |k: f32, n: usize| -> Vec<f32> {
        (0..128)
            .map(|i| if i < n { i as f32 * k } else { i as f32 })
            .collect()
    };
    let scale_state = state(&device, &scale(), "__entry_scale");
    let b = floats_buffer(&device, &ramp);
    run(
        &device,
        &scale_state,
        line(128, 32),
        one,
        &[
            whole(&b),
            Arg::Bytes(&2.5f32.to_le_bytes()),
            Arg::Bytes(&100u32.to_le_bytes()),
        ],
        None,
    );
    let got = floats(&b);
    assert_eq!(got, scaled(2.5, 100), "scale, by bytes");
    assert_eq!(
        [got[1], got[99], got[100], got[127]],
        [2.5, 247.5, 100.0, 127.0]
    );
    let b = floats_buffer(&device, &ramp);
    let mut values = [0u32; 16];
    values[4] = 3.0f32.to_bits();
    values[8] = 50;
    let values = words_buffer(&device, &values);
    run(
        &device,
        &scale_state,
        line(128, 32),
        one,
        &[
            whole(&b),
            Arg::Buffer(&values, 16),
            Arg::Buffer(&values, 32),
        ],
        None,
    );
    let got = floats(&b);
    assert_eq!(got, scaled(3.0, 50), "scale, by buffers at offsets");
    assert_eq!([got[1], got[49], got[50]], [3.0, 147.0, 50.0]);

    let st_state = state(&device, &st(), "__entry_st");
    let o = floats_buffer(&device, &[0.0; 16]);
    let s: Vec<u8> = [2.5f32.to_bits(), 4]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    run(
        &device,
        &st_state,
        line(16, 16),
        one,
        &[whole(&o), Arg::Bytes(&s)],
        None,
    );
    assert_eq!(floats(&o), [6.5; 16], "st");

    let wg_state = state(&device, &wg(), "__entry_wg");
    assert_eq!(wg_state.max_total_threads_per_threadgroup(), 64);
    for group in [64u64, 32] {
        let o = floats_buffer(&device, &[0.0; 128]);
        run(
            &device,
            &wg_state,
            line(128, group),
            one,
            &[whole(&o)],
            None,
        );
        assert_eq!(floats(&o), [group as f32; 128], "wg in groups of {group}");
    }
}

/// The eight words thread `(x, y)` of [`items`] stores over the 8-by-4 grid in groups of 4 by 2 at
/// offset (3, 5), of two dimensions, reading dimension `d` (`dims`, `local`, `groups` and
/// `offset` OpenCL's 1, 1, 1 and 0 past dimension 1).
fn items_words(x: u32, y: u32, d: u32) -> [u32; 8] {
    let (offset, global, local) = ([3, 5], [8, 4], [4, 2]);
    let id = [x + offset[0], y + offset[1]];
    let lid = [x % local[0], y % local[1]];
    let group = [x / local[0], y / local[1]];
    let groups = [global[0] / local[0], global[1] / local[1]];
    let of = |v: [u32; 2], past: u32| v.get(d as usize).copied().unwrap_or(past);
    [
        id[0],
        id[1],
        lid[0] | lid[1] << 8 | group[0] << 16 | group[1] << 24,
        groups[0] | groups[1] << 8 | local[0] << 16 | local[1] << 24,
        global[0] | global[1] << 8 | 2 << 16,
        of(id, 0) + of(local, 1) + of(groups, 1) + of(offset, 0),
        2,
        7,
    ]
}

#[test]
fn opencl_mode_work_item_functions_read_the_dispatch() {
    let scratch = Scratch::new("opencl-mode-items");
    let device = device(&scratch.0);
    let items_state = state(&device, &items(), "__entry_items");
    // Dimension 1, as the M3 Max ran it, and dimension 5, past the grid's.
    for d in [1u32, 5] {
        let o = words_buffer(&device, &[0xEEEE_EEEE; 32 * 8]);
        run(
            &device,
            &items_state,
            (Size::new(8, 4, 1), Size::new(4, 2, 1)),
            ([3, 5, 0], 2),
            &[whole(&o), Arg::Bytes(&d.to_le_bytes())],
            None,
        );
        let got = words(&o);
        let want: Vec<u32> = (0..4)
            .flat_map(|y| (0..8).flat_map(move |x| items_words(x, y, d)))
            .collect();
        assert_eq!(got, want, "dimension {d}");
        if d == 1 {
            // The M3 Max's first thread, word 6 OpenCL's 2 where it leaves the word unwritten.
            assert_eq!(
                got[..8],
                [3, 5, 0, 0x0204_0202, 0x0002_0408, 0xe, 2, 7],
                "the first thread"
            );
        }
    }
}
