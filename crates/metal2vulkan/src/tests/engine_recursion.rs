//! Recursion through visible functions on an explicit call stack.
//!
//! The owned synthetic module [`MODULE`] holds, in the form Apple's compiler emits for the MSL its
//! comments quote (the probe kernels), a visible function that calls itself through its
//! visible-function reference (`drec`), three that recurse through entries of a
//! `visible_function_table` held in an argument buffer (`rec_sum`; `rec_even` and `rec_odd`, each
//! calling the other), and one whose 64-word private array lives across its recursive call
//! (`rec_big`), and one that recurses through a function pointer passed as a value (`prec` passes
//! its own address to the helper `apply`, which calls through it). Each kernel calls them with a
//! depth read per thread, and so do a vertex function (`v_rec`), a fragment function (`f_rec`),
//! and a tile function (`t_rec`) through `drec`. The pipeline links the visible functions from
//! the same module.
//!
//! - A kernel compiles under its pipeline's `maxCallStackDepth` `k`: direct recursion's stack
//!   holds `k` activations, the kernel's call and `k - 1` recursive calls, as plain Metal's does;
//!   recursion through a table of three 16-byte functions holds `3k` (our probes and the
//!   byte model). On the CPU interpreter every depth within them answers what Metal does, at `k` 1,
//!   2, 8, and 256. A call past the stack is not made and yields zero, so a deeper sum stops early
//!   and a deeper parity is zero (the recorded near-equivalent; Metal's own answer there is not a
//!   defined one).
//! - A depth whose stack passes Metal's (261728 bytes a thread) is
//!   `LimitExceeded(CallStackBytes)`, where Metal says "Compute pipeline exceeds available stack
//!   space"; stacks past the Function storage the device holds correctly (measured, or the floor
//!   of 16 KiB a thread on a device not measured) are `Unsupported{Recursion}` needing more; depth
//!   0 is depth 1.
//! - The kernels pass L3 on every committed profile that runs them, and record on each whose
//!   device was not measured that they rest on the floor.
//! - Recursion through a pointer passed as a value calls the function the pointer holds in every
//!   activation, on the same stack: `apply` folds into `prec`, so `prec` at depth `k`
//!   makes `k - 1` recursive calls and is refused past 16358, as plain Metal does, linked in
//!   `functions` or `private_functions` alike.
//! - A render pipeline's vertex and fragment functions, and a tile pipeline's tile function,
//!   compile under the pipeline's `maxVertexCallStackDepth`, `maxFragmentCallStackDepth`, and
//!   `maxCallStackDepth` as a kernel does under its own; a `Compiled` stage serves only a pipeline
//!   of its depth for that stage.
//!
//! `llvm-as` assembles the module; a missing tool skips visibly.

use super::engine_argument_tables::{bytes_at, capable_profiles, device, one_group, table_layout};
use crate::bind::{ArgumentFunctionTable, BindState, FunctionTableWords};
use crate::budget::{Budget, SPIRV_WORDS};
use crate::device::VkLimit;
use crate::error::{Capability, DescriptorError, Error, Feature, Limit, Unsupported};
use crate::library::{FunctionConstantValues, Library};
use crate::lower::access::function_table::{directory_symbol, heap_symbol};
use crate::lower::select::LowerInputs;
use crate::metal::PixelFormat;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::compile::{compile_bitcode_linked, EntryRef, LinkedSource};
use crate::pipeline::descriptor::ColorAttachmentDescriptor;
use crate::pipeline::descriptor::LinkedList;
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    link_and_validate, meter, profile, typed_module_linked_with, AssembleError,
};
use crate::pipeline::{
    ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor, FunctionHandle, LinkedFunction,
    LinkedFunctions, RenderPipelineDescriptor, RenderPipelineKeys, StageSource,
    TileRenderPipelineDescriptor,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::ir::TypedModule;
use crate::spirv::testing::committed_profiles;

/// `out[tid] = drec(depth[tid])`.
const DIRECT: &str = "k_direct";
/// `out[2 tid] = table[0](d, args)`, `out[2 tid + 1] = table[1](d, args)`.
const TABLE: &str = "rec_kernel";
/// `out[tid] = table[0](d, args)` for `rec_big`.
const BIG: &str = "big_kernel";
/// `out[tid] = apply(prec, depth[tid])`: `prec` recurses through the pointer `apply` calls.
const POINTER: &str = "k_pointer";
/// The functions [`TABLE`]'s pipeline links, in its table's order.
const SUMS: [&str; 3] = ["rec_sum", "rec_even", "rec_odd"];
/// The table's id.
const T: u64 = 0x10;

/// The owned synthetic module.
const MODULE: &str = r#"; Owned synthetic AIR, in the form Apple's compiler emits for the MSL
; each function's comment quotes.
source_filename = "uni_recursion.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

%struct.RecArgs = type { %"struct.metal::visible_function_table" }
%"struct.metal::visible_function_table" = type { ptr addrspace(1) }
%"struct.metal::_imageblock_base" = type { ptr addrspace(4) }

; [[visible]] uint drec(uint n) { return n == 0 ? 0u : n + drec(n - 1); }
define i32 @drec(i32 noundef %n) local_unnamed_addr #0 {
entry:
  %zero = icmp eq i32 %n, 0
  br i1 %zero, label %done, label %more
more:
  %m = add i32 %n, -1
  %r = tail call i32 @drec.MTL_VISIBLE_FN_REF(i32 noundef %m) #1
  %s = add i32 %r, %n
  br label %done
done:
  %v = phi i32 [ %s, %more ], [ 0, %entry ]
  ret i32 %v
}

declare i32 @drec.MTL_VISIBLE_FN_REF(i32) local_unnamed_addr section "air.externally_defined"

; kernel void k_direct(device uint *out [[buffer(0)]], device const uint *depth [[buffer(2)]],
;                      uint tid [[thread_position_in_grid]]) { out[tid] = drec(depth[tid]); }
define void @k_direct(ptr addrspace(1) noundef "air-buffer-no-alias" %out, ptr addrspace(1) noundef readonly "air-buffer-no-alias" %depth, i32 noundef %tid) local_unnamed_addr #0 {
  %i = zext i32 %tid to i64
  %dp = getelementptr inbounds i32, ptr addrspace(1) %depth, i64 %i
  %d = load i32, ptr addrspace(1) %dp, align 4
  %r = tail call i32 @drec.MTL_VISIBLE_FN_REF(i32 noundef %d) #1
  %op = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %r, ptr addrspace(1) %op, align 4
  ret void
}

; struct RecArgs; using rec_t = uint(uint, device const RecArgs &);
; struct RecArgs { visible_function_table<rec_t> table; };
; [[visible]] uint rec_sum(uint n, device const RecArgs &a) {
;   return n == 0 ? 0u : n + a.table[0](n - 1, a);
; }
define i32 @rec_sum(i32 noundef %n, ptr addrspace(1) noundef align 8 dereferenceable(8) %a) local_unnamed_addr #0 {
entry:
  %zero = icmp eq i32 %n, 0
  br i1 %zero, label %done, label %more
more:
  %tp = getelementptr inbounds %struct.RecArgs, ptr addrspace(1) %a, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(1) %tp, align 8
  %f = tail call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) readonly captures(none) %t, i32 0) #2
  %m = add i32 %n, -1
  %r = tail call i32 %f(i32 noundef %m, ptr addrspace(1) noundef align 8 dereferenceable(8) %a) #1
  %s = add i32 %r, %n
  br label %done
done:
  %v = phi i32 [ %s, %more ], [ 0, %entry ]
  ret i32 %v
}

; [[visible]] uint rec_even(uint n, device const RecArgs &a) {
;   return n == 0 ? 1u : a.table[2](n - 1, a);
; }
define i32 @rec_even(i32 noundef %n, ptr addrspace(1) noundef align 8 dereferenceable(8) %a) local_unnamed_addr #0 {
entry:
  %zero = icmp eq i32 %n, 0
  br i1 %zero, label %done, label %more
more:
  %m = add i32 %n, -1
  %tp = getelementptr inbounds %struct.RecArgs, ptr addrspace(1) %a, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(1) %tp, align 8
  %f = tail call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) readonly captures(none) %t, i32 2) #2
  %r = tail call i32 %f(i32 noundef %m, ptr addrspace(1) noundef align 8 dereferenceable(8) %a) #1
  br label %done
done:
  %v = phi i32 [ %r, %more ], [ 1, %entry ]
  ret i32 %v
}

; [[visible]] uint rec_odd(uint n, device const RecArgs &a) {
;   return n == 0 ? 0u : a.table[1](n - 1, a);
; }
define i32 @rec_odd(i32 noundef %n, ptr addrspace(1) noundef align 8 dereferenceable(8) %a) local_unnamed_addr #0 {
entry:
  %zero = icmp eq i32 %n, 0
  br i1 %zero, label %done, label %more
more:
  %m = add i32 %n, -1
  %tp = getelementptr inbounds %struct.RecArgs, ptr addrspace(1) %a, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(1) %tp, align 8
  %f = tail call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) readonly captures(none) %t, i32 1) #2
  %r = tail call i32 %f(i32 noundef %m, ptr addrspace(1) noundef align 8 dereferenceable(8) %a) #1
  br label %done
done:
  %v = phi i32 [ %r, %more ], [ 0, %entry ]
  ret i32 %v
}

; [[visible]] uint rec_big(uint n, device const RecArgs &a) {
;   uint w[64];
;   for (uint i = 0; i < 64; i++) w[i] = n * 64 + i;
;   uint s = n == 0 ? 0u : a.table[0](n - 1, a);
;   for (uint j = 0; j < 64; j++) s += w[(j * 7 + n) & 63] * (j + 1);
;   return s;
; }
define i32 @rec_big(i32 noundef %n, ptr addrspace(1) noundef align 8 dereferenceable(8) %a) local_unnamed_addr #0 {
entry:
  %w = alloca [64 x i32], align 4
  %base = shl i32 %n, 6
  br label %fill
fill:
  %i = phi i32 [ 0, %entry ], [ %i1, %fill ]
  %p = getelementptr inbounds [64 x i32], ptr %w, i32 0, i32 %i
  %x0 = add i32 %base, %i
  store i32 %x0, ptr %p, align 4
  %i1 = add nuw nsw i32 %i, 1
  %filled = icmp eq i32 %i1, 64
  br i1 %filled, label %test, label %fill
test:
  %zero = icmp eq i32 %n, 0
  br i1 %zero, label %sum, label %more
more:
  %tp = getelementptr inbounds %struct.RecArgs, ptr addrspace(1) %a, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(1) %tp, align 8
  %f = tail call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) readonly captures(none) %t, i32 0) #2
  %m = add i32 %n, -1
  %r = tail call i32 %f(i32 noundef %m, ptr addrspace(1) noundef align 8 dereferenceable(8) %a) #1
  br label %sum
sum:
  %s0 = phi i32 [ 0, %test ], [ %r, %more ]
  br label %read
read:
  %j = phi i32 [ 0, %sum ], [ %j1, %read ]
  %s = phi i32 [ %s0, %sum ], [ %s1, %read ]
  %j7 = mul i32 %j, 7
  %jn = add i32 %j7, %n
  %idx = and i32 %jn, 63
  %q = getelementptr inbounds [64 x i32], ptr %w, i32 0, i32 %idx
  %x = load i32, ptr %q, align 4
  %j1 = add nuw nsw i32 %j, 1
  %y = mul i32 %x, %j1
  %s1 = add i32 %s, %y
  %read_all = icmp eq i32 %j1, 64
  br i1 %read_all, label %exit, label %read
exit:
  ret i32 %s1
}

; kernel void rec_kernel(device const RecArgs &args [[buffer(1)]],
;                        device const uint *depth [[buffer(2)]], device uint *out [[buffer(0)]],
;                        uint tid [[thread_position_in_grid]]) {
;   uint d = depth[tid];
;   out[2 * tid] = args.table[0](d, args);
;   out[2 * tid + 1] = args.table[1](d, args);
; }
define void @rec_kernel(ptr addrspace(1) noundef align 8 dereferenceable(8) "air-buffer-no-alias" %args, ptr addrspace(1) noundef readonly "air-buffer-no-alias" %depth, ptr addrspace(1) noundef "air-buffer-no-alias" %out, i32 noundef %tid) local_unnamed_addr #0 {
  %i = zext i32 %tid to i64
  %dp = getelementptr inbounds i32, ptr addrspace(1) %depth, i64 %i
  %d = load i32, ptr addrspace(1) %dp, align 4
  %tp = getelementptr inbounds %struct.RecArgs, ptr addrspace(1) %args, i64 0, i32 0, i32 0
  %t0 = load ptr addrspace(1), ptr addrspace(1) %tp, align 8
  %f0 = tail call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) readonly captures(none) %t0, i32 0) #2
  %r0 = tail call i32 %f0(i32 noundef %d, ptr addrspace(1) noundef align 8 dereferenceable(8) %args) #1
  %w0 = shl i32 %tid, 1
  %j0 = zext i32 %w0 to i64
  %o0 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %j0
  store i32 %r0, ptr addrspace(1) %o0, align 4
  %t1 = load ptr addrspace(1), ptr addrspace(1) %tp, align 8
  %f1 = tail call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) readonly captures(none) %t1, i32 1) #2
  %r1 = tail call i32 %f1(i32 noundef %d, ptr addrspace(1) noundef align 8 dereferenceable(8) %args) #1
  %w1 = or i32 %w0, 1
  %j1 = zext i32 %w1 to i64
  %o1 = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %j1
  store i32 %r1, ptr addrspace(1) %o1, align 4
  ret void
}

; kernel void big_kernel(device const RecArgs &args [[buffer(1)]],
;                        device const uint *depth [[buffer(2)]], device uint *out [[buffer(0)]],
;                        uint tid [[thread_position_in_grid]]) {
;   out[tid] = args.table[0](depth[tid], args);
; }
define void @big_kernel(ptr addrspace(1) noundef align 8 dereferenceable(8) "air-buffer-no-alias" %args, ptr addrspace(1) noundef readonly "air-buffer-no-alias" %depth, ptr addrspace(1) noundef "air-buffer-no-alias" %out, i32 noundef %tid) local_unnamed_addr #0 {
  %i = zext i32 %tid to i64
  %dp = getelementptr inbounds i32, ptr addrspace(1) %depth, i64 %i
  %d = load i32, ptr addrspace(1) %dp, align 4
  %tp = getelementptr inbounds %struct.RecArgs, ptr addrspace(1) %args, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(1) %tp, align 8
  %f = tail call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) readonly captures(none) %t, i32 0) #2
  %r = tail call i32 %f(i32 noundef %d, ptr addrspace(1) noundef align 8 dereferenceable(8) %args) #1
  %op = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %r, ptr addrspace(1) %op, align 4
  ret void
}

; [[visible]] uint chain_a(uint n, device const RecArgs &a) { return 1000u + a.table[1](n + 1, a); }
define i32 @chain_a(i32 noundef %n, ptr addrspace(1) noundef align 8 dereferenceable(8) %a) local_unnamed_addr #0 {
  %tp = getelementptr inbounds %struct.RecArgs, ptr addrspace(1) %a, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(1) %tp, align 8
  %f = tail call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) readonly captures(none) %t, i32 1) #2
  %m = add i32 %n, 1
  %r = tail call i32 %f(i32 noundef %m, ptr addrspace(1) noundef align 8 dereferenceable(8) %a) #1
  %s = add i32 %r, 1000
  ret i32 %s
}

; [[visible]] uint chain_b(uint n, device const RecArgs &a) { return 100u + a.table[2](n + 1, a); }
define i32 @chain_b(i32 noundef %n, ptr addrspace(1) noundef align 8 dereferenceable(8) %a) local_unnamed_addr #0 {
  %tp = getelementptr inbounds %struct.RecArgs, ptr addrspace(1) %a, i64 0, i32 0, i32 0
  %t = load ptr addrspace(1), ptr addrspace(1) %tp, align 8
  %f = tail call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) readonly captures(none) %t, i32 2) #2
  %m = add i32 %n, 1
  %r = tail call i32 %f(i32 noundef %m, ptr addrspace(1) noundef align 8 dereferenceable(8) %a) #1
  %s = add i32 %r, 100
  ret i32 %s
}

; [[visible]] uint chain_c(uint n, device const RecArgs &a) { return n * 3u; }
define i32 @chain_c(i32 noundef %n, ptr addrspace(1) noundef align 8 dereferenceable(8) %a) local_unnamed_addr #0 {
  %r = mul i32 %n, 3
  ret i32 %r
}

; [[visible]] uint leaf(uint n) { return n * 3u; }
define i32 @leaf(i32 noundef %n) local_unnamed_addr #0 {
  %r = mul i32 %n, 3
  ret i32 %r
}

declare i32 @leaf.MTL_VISIBLE_FN_REF(i32) local_unnamed_addr section "air.externally_defined"

; [[visible]] uint leaf_caller(uint n, device const RecArgs &a) { return 7u + leaf(n); }
define i32 @leaf_caller(i32 noundef %n, ptr addrspace(1) noundef align 8 dereferenceable(8) %a) local_unnamed_addr #0 {
  %r = tail call i32 @leaf.MTL_VISIBLE_FN_REF(i32 noundef %n) #1
  %s = add i32 %r, 7
  ret i32 %s
}

; vertex float4 v_rec(uint vid [[vertex_id]]) { return float4(float(drec(vid)), 0, 0, 1); }
define <{ <4 x float> }> @v_rec(i32 noundef %vid) local_unnamed_addr #0 {
  %r = tail call i32 @drec.MTL_VISIBLE_FN_REF(i32 noundef %vid) #1
  %f = uitofp i32 %r to float
  %p = insertelement <4 x float> <float 0.0, float 0.0, float 0.0, float 1.0>, float %f, i64 0
  %o = insertvalue <{ <4 x float> }> undef, <4 x float> %p, 0
  ret <{ <4 x float> }> %o
}

; fragment float4 f_rec(float4 pos [[position]]) {
;   return float4(float(drec(uint(pos.x))), 0, 0, 1);
; }
define <4 x float> @f_rec(<4 x float> noundef %pos) local_unnamed_addr #0 {
  %x = extractelement <4 x float> %pos, i64 0
  %n = fptoui float %x to i32
  %r = tail call i32 @drec.MTL_VISIBLE_FN_REF(i32 noundef %n) #1
  %f = uitofp i32 %r to float
  %c = insertelement <4 x float> <float 0.0, float 0.0, float 0.0, float 1.0>, float %f, i64 0
  ret <4 x float> %c
}

; struct Px { float f [[color(0)]]; };
; kernel void t_rec(imageblock<Px, imageblock_layout_implicit> block,
;                   ushort2 tid [[thread_position_in_threadgroup]]) {
;   Px p = block.read(tid);
;   block.write(Px{float(drec(uint(p.f)))}, tid);
; }
define void @t_rec(%"struct.metal::_imageblock_base" %block, <2 x i16> noundef %tid) local_unnamed_addr #0 {
  %f = tail call float @air.load.implicit_imageblock.f32(i32 0, <2 x i16> %tid, i32 0, i16 0) #4
  %n = fptoui float %f to i32
  %r = tail call i32 @drec.MTL_VISIBLE_FN_REF(i32 noundef %n) #1
  %g = uitofp i32 %r to float
  tail call void @air.store.implicit_imageblock.f32(float %g, i32 0, <2 x i16> %tid, i32 0, i16 0) #5
  ret void
}

; using step_t = uint(uint);
; static uint apply(step_t *f, uint n) { return f(n); }
define internal fastcc i32 @apply(ptr noundef %f, i32 noundef %n) unnamed_addr #0 {
  %r = tail call i32 %f(i32 noundef %n) #1
  ret i32 %r
}

; [[visible]] uint prec(uint n) { return n == 0 ? 0u : n + apply(prec, n - 1); }
define i32 @prec(i32 noundef %n) local_unnamed_addr #0 {
entry:
  %zero = icmp eq i32 %n, 0
  br i1 %zero, label %done, label %more
more:
  %m = add i32 %n, -1
  %r = tail call fastcc i32 @apply(ptr noundef nonnull @prec.MTL_VISIBLE_FN_REF, i32 noundef %m)
  %s = add i32 %r, %n
  br label %done
done:
  %v = phi i32 [ %s, %more ], [ 0, %entry ]
  ret i32 %v
}

declare i32 @prec.MTL_VISIBLE_FN_REF(i32) local_unnamed_addr section "air.externally_defined"

; kernel void k_pointer(device uint *out [[buffer(0)]], device const uint *depth [[buffer(2)]],
;                       uint tid [[thread_position_in_grid]]) {
;   out[tid] = apply(prec, depth[tid]);
; }
define void @k_pointer(ptr addrspace(1) noundef "air-buffer-no-alias" %out, ptr addrspace(1) noundef readonly "air-buffer-no-alias" %depth, i32 noundef %tid) local_unnamed_addr #0 {
  %i = zext i32 %tid to i64
  %dp = getelementptr inbounds i32, ptr addrspace(1) %depth, i64 %i
  %d = load i32, ptr addrspace(1) %dp, align 4
  %r = tail call fastcc i32 @apply(ptr noundef nonnull @prec.MTL_VISIBLE_FN_REF, i32 noundef %d)
  %op = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %i
  store i32 %r, ptr addrspace(1) %op, align 4
  ret void
}

declare float @air.load.implicit_imageblock.f32(i32, <2 x i16>, i32, i16) local_unnamed_addr #6
declare void @air.store.implicit_imageblock.f32(float, i32, <2 x i16>, i32, i16) local_unnamed_addr #7

declare ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) readonly captures(none), i32) local_unnamed_addr #3

attributes #0 = { convergent mustprogress nounwind }
attributes #1 = { convergent nounwind }
attributes #2 = { nounwind willreturn memory(argmem: read) }
attributes #3 = { mustprogress nofree nounwind willreturn memory(argmem: read) }
attributes #4 = { nounwind willreturn memory(read) }
attributes #5 = { nounwind willreturn memory(write) }
attributes #6 = { mustprogress nofree nounwind willreturn memory(read) }
attributes #7 = { mustprogress nounwind willreturn memory(write) }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.visible = !{!9, !14, !15, !16, !17, !41, !42, !43, !44, !45, !62}
!air.kernel = !{!20, !25, !31, !57, !63}
!air.vertex = !{!47}
!air.fragment = !{!52}
!air.compile_options = !{!35, !36, !37}
!air.visible_function_references = !{!38, !46, !64}
!air.version = !{!39}
!air.language_version = !{!40}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 27, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{ptr @drec, !10, !12}
!10 = !{!11}
!11 = !{!"air.visible_output", !"air.arg_type_name", !"uint"}
!12 = !{!13}
!13 = !{i32 0, !"air.visible_input", !"air.arg_type_name", !"uint", !"air.arg_name", !"n"}
!14 = !{ptr @rec_sum, !10, !18}
!15 = !{ptr @rec_even, !10, !18}
!16 = !{ptr @rec_odd, !10, !18}
!17 = !{ptr @rec_big, !10, !18}
!18 = !{!13, !19}
!19 = !{i32 1, !"air.visible_input", !"air.arg_type_name", !"RecArgs", !"air.arg_name", !"a"}
!20 = !{ptr @k_direct, !21, !22}
!21 = !{}
!22 = !{!23, !24, !26}
!23 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!24 = !{i32 1, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"depth"}
!25 = !{ptr @rec_kernel, !21, !27}
!26 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!27 = !{!28, !32, !33, !34}
!28 = !{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 8, !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.struct_type_info", !29, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"RecArgs", !"air.arg_name", !"args"}
!29 = !{i32 0, i32 8, i32 0, !"visible_function_table", !"table", !"air.indirect_argument", !30}
!30 = !{i32 0, !"air.visible_function_table", !"air.location_index", i32 0, i32 1, !"air.read", !"air.arg_type_name", !"visible_function_table", !"air.arg_name", !"table"}
!31 = !{ptr @big_kernel, !21, !27}
!32 = !{i32 1, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"depth"}
!33 = !{i32 2, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}
!34 = !{i32 3, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!35 = !{!"air.compile.denorms_disable"}
!36 = !{!"air.compile.fast_math_enable"}
!37 = !{!"air.compile.framebuffer_fetch_enable"}
!38 = !{!"air.visible_function_reference", ptr @drec.MTL_VISIBLE_FN_REF, !"drec"}
!39 = !{i32 2, i32 8, i32 0}
!40 = !{!"Metal", i32 3, i32 1, i32 0}
!41 = !{ptr @chain_a, !10, !18}
!42 = !{ptr @chain_b, !10, !18}
!43 = !{ptr @chain_c, !10, !18}
!44 = !{ptr @leaf, !10, !12}
!45 = !{ptr @leaf_caller, !10, !18}
!46 = !{!"air.visible_function_reference", ptr @leaf.MTL_VISIBLE_FN_REF, !"leaf"}
!47 = !{ptr @v_rec, !48, !50}
!48 = !{!49}
!49 = !{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}
!50 = !{!51}
!51 = !{i32 0, !"air.vertex_id", !"air.arg_type_name", !"uint", !"air.arg_name", !"vid"}
!52 = !{ptr @f_rec, !53, !55}
!53 = !{!54}
!54 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!55 = !{!56}
!56 = !{i32 0, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"pos"}
!57 = !{ptr @t_rec, !21, !58}
!58 = !{!59, !61}
!59 = !{i32 0, !"air.imageblock", !"implicit", !"air.struct_type_info", !60, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"imageblock<Px, layout_implicit>", !"air.arg_name", !"block"}
!60 = !{i32 0, i32 4, i32 0, !"float", !"f", !"air.render_target", i32 0}
!61 = !{i32 1, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"ushort2", !"air.arg_name", !"tid"}
!62 = !{ptr @prec, !10, !12}
!63 = !{ptr @k_pointer, !21, !22}
!64 = !{!"air.visible_function_reference", ptr @prec.MTL_VISIBLE_FN_REF, !"prec"}
"#;

/// [`MODULE`] assembled.
fn module() -> Option<Vec<u8>> {
    match metal2vulkan_devtools::llvm_assemble_text(MODULE) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected the module: {why}"),
    }
}

/// The lowering inputs of a pipeline whose `maxCallStackDepth` is `k`.
pub(super) fn depth(k: u32) -> LowerInputs {
    LowerInputs {
        max_call_stack_depth: Some(k),
        ..LowerInputs::new()
    }
}

/// The functions `names` of `bitcode`, linked.
fn sources<'a>(
    bitcode: &'a [u8],
    names: &[&'a str],
    none: &'a FunctionConstantValues,
) -> Vec<LinkedSource<'a>> {
    names
        .iter()
        .map(|name| LinkedSource::new(name, bitcode, none))
        .collect()
}

/// The functions each kernel's pipeline links.
fn linked_names(entry: &str) -> &'static [&'static str] {
    match entry {
        DIRECT => &["drec"],
        TABLE => &SUMS,
        POINTER => &["prec"],
        _ => &["rec_big"],
    }
}

/// Kernel `entry` compiled at depth `k` on `p`, its functions linked.
fn compile(
    bitcode: &[u8],
    entry: &str,
    k: u32,
    p: &crate::device::DeviceProfile,
) -> Result<std::sync::Arc<crate::library::FunctionArtifact>, Error> {
    compile_linked(bitcode, entry, linked_names(entry), k, p)
}

/// Kernel `entry` compiled at depth `k` on `p`, the functions `names` linked.
pub(super) fn compile_linked(
    bitcode: &[u8],
    entry: &str,
    names: &[&str],
    k: u32,
    p: &crate::device::DeviceProfile,
) -> Result<std::sync::Arc<crate::library::FunctionArtifact>, Error> {
    let none = FunctionConstantValues::defaults();
    compile_bitcode_linked(
        bitcode,
        EntryRef::Name(entry),
        (&none, &sources(bitcode, names, &none)),
        p,
        &depth(k),
        &mut meter(),
        SPIRV_WORDS,
    )
}

/// The typed module of kernel `entry` at depth `k` on `p`.
fn typed(bitcode: &[u8], entry: &str, k: u32, p: &crate::device::DeviceProfile) -> TypedModule {
    let none = FunctionConstantValues::defaults();
    let linked = sources(bitcode, linked_names(entry), &none);
    typed_module_linked_with(bitcode, entry, p, &linked, &depth(k))
        .unwrap_or_else(|e| panic!("{entry} at depth {k}: {e}"))
        .1
}

/// The pipeline of kernel `entry` at depth `k` with its functions linked, through the public API,
/// and the handles of those functions in their linked order.
fn pipeline(bitcode: &[u8], entry: &str, k: u32) -> (ComputePipeline, Vec<FunctionHandle>) {
    let dev = device("moltenvk-m3");
    let lib = Library::from_air(bitcode).expect("library");
    let names = linked_names(entry);
    let linked = names.iter().fold(LinkedFunctions::new(), |l, name| {
        l.with_function(lib.function(name).expect("function"))
    });
    let handles = names
        .iter()
        .map(|name| {
            linked
                .function_handle(&LinkedFunction::new(lib.function(name).expect("function")))
                .expect("a handle")
        })
        .collect();
    let mut d = ComputePipelineDescriptor::new(StageSource::Function(
        FunctionDescriptor::new(lib.function(entry).expect("entry")).with_linked(linked),
    ));
    d.max_call_stack_depth = k;
    let pipe = dev
        .compile_compute_pipeline(&d, &Budget::standard())
        .unwrap_or_else(|e| panic!("{entry} at depth {k}: {e}"));
    (pipe, handles)
}

/// The directory bytes a dispatch of `pipe` gets with table [`T`] listed, and the bytes of
/// table [`T`] holding `handles` in order.
fn table_inputs(pipe: &ComputePipeline, handles: &[FunctionHandle]) -> (Vec<u8>, Vec<u8>) {
    let dev = device("moltenvk-m3");
    let (_, directory) = table_layout(pipe);
    let state = BindState {
        function_tables: vec![ArgumentFunctionTable::new(T, handles.len() as u32)],
        ..Default::default()
    };
    let plan = pipe.plan(&dev, &state.shape()).expect("plan");
    let cmds = plan
        .instantiate_dispatch(&state, &one_group())
        .expect("dispatch");
    let directory = bytes_at(&cmds.descriptor_bytes, directory).to_vec();
    let mut table = FunctionTableWords::new(handles.len() as u32);
    for (i, h) in handles.iter().enumerate() {
        assert!(table.set(i as u32, Some(*h)));
    }
    (directory, table.to_bytes())
}

/// Kernel buffer `index`.
pub(super) fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

/// A buffer bound at `slot` holding `bytes`.
pub(super) fn buffer(slot: ResourceSlot, bytes: Vec<u8>) -> Buffer {
    Buffer {
        slot: Some(slot),
        address: None,
        bytes,
    }
}

/// Run `m` with one thread per entry of `depths` (buffer 2), `out_words` output words a thread
/// (buffer 0), and, for a table kernel, `tables`: the directory's bytes and table [`T`]'s, with
/// [`T`]'s id at buffer 1. The words it writes.
fn dispatch(
    m: &TypedModule,
    depths: &[u32],
    out_words: usize,
    tables: Option<&(Vec<u8>, Vec<u8>)>,
) -> Vec<u32> {
    let n = depths.len();
    let mut buffers = vec![
        buffer(slot(0), vec![0; n * out_words * 4]),
        buffer(
            slot(2),
            depths.iter().flat_map(|d| d.to_le_bytes()).collect(),
        ),
    ];
    if let Some((directory, table)) = tables {
        buffers.extend([
            buffer(slot(1), T.to_le_bytes().to_vec()),
            buffer(directory_symbol(Stage::Kernel), directory.clone()),
            buffer(heap_symbol(Stage::Kernel), vec![0; 16]),
            buffer(heap_symbol(Stage::Kernel), table.clone()),
        ]);
    }
    let res = Resources {
        buffers,
        push: vec![0; 128],
        spec: Default::default(),
    };
    let cfg = Config {
        workgroups: [n as u32, 1, 1],
        ..Config::default()
    };
    let out = run(m, &cfg, res).unwrap_or_else(|e| panic!("{e:?}"));
    out.buffers[0]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// `n + (n - 1) + ...` over the activations `frames` frames hold, the kernel's call among them:
/// the sum Metal answers when `n` fits, which a call past the stack ends with zero.
fn sum(n: u32, frames: u32) -> u32 {
    (0..frames.min(n + 1)).fold(0u32, |s, i| s.wrapping_add(n - i))
}

/// `rec_even(n)` with `frames` frames: Metal's parity when the chain of `n + 1` calls fits, else
/// the zero the call past the stack yields.
fn even(n: u32, frames: u32) -> u32 {
    if n < frames {
        u32::from(n.is_multiple_of(2))
    } else {
        0
    }
}

/// `rec_big(n)` with `frames` frames.
fn big(n: u32, frames: u32) -> u32 {
    let w: Vec<u32> = (0..64).map(|i| (n << 6).wrapping_add(i)).collect();
    let s = if n == 0 || frames <= 1 {
        0
    } else {
        big(n - 1, frames - 1)
    };
    (0..64u32).fold(s, |s, j| {
        let x = w[(j.wrapping_mul(7).wrapping_add(n) & 63) as usize];
        s.wrapping_add(x.wrapping_mul(j + 1))
    })
}

/// The depths around a stack of `frames` frames: the deepest that fit and the shallowest that do
/// not, and a few at the bottom.
fn depths(frames: u32) -> Vec<u32> {
    let mut d: Vec<u32> = [0, 1, 2, 3]
        .into_iter()
        .chain([
            frames.saturating_sub(2),
            frames - 1,
            frames,
            frames + 1,
            frames + 4,
        ])
        .collect();
    d.sort_unstable();
    d.dedup();
    d
}

/// The activations Metal holds through a table of three 16-byte functions at the depths the tests
/// use, `3k` (our probes on an M3 Max and an M4: correct to 3 frames at `k` 1; their byte model):
/// the stack the lowering holds.
const TABLE_FRAMES: [(u32, u32); 4] = [(1, 3), (2, 6), (8, 24), (256, 768)];

/// Direct recursion makes `k - 1` recursive calls, as plain Metal does: its stack holds `k`
/// activations, the kernel's call among them, so every depth below `k` answers Metal's sum, and
/// depth `k` too, its last call's zero being the zero a call past the
/// stack yields; a deeper one stops early. At `k` 1 the kernel's call runs and makes none.
#[test]
fn direct_recursion_makes_one_call_fewer_than_its_depth() {
    let Some(bitcode) = module() else {
        return;
    };
    let p = profile("moltenvk-m3", Default::default()).expect("profile");
    for k in [1, 2, 8, 256] {
        let m = typed(&bitcode, DIRECT, k, &p);
        let ds = depths(k);
        let want: Vec<u32> = ds.iter().map(|&d| sum(d, k)).collect();
        assert_eq!(dispatch(&m, &ds, 1, None), want, "depth {k}: {ds:?}");
    }
}

/// Recursion through a table holds the frames Metal reached: within them each sum and parity is
/// Metal's, past them the sum stops and the parity is zero.
#[test]
fn table_recursion_reaches_the_frames_metal_reaches() {
    let Some(bitcode) = module() else {
        return;
    };
    let p = profile("moltenvk-m3", Default::default()).expect("profile");
    for (k, frames) in TABLE_FRAMES {
        let (pipe, handles) = pipeline(&bitcode, TABLE, k);
        let tables = table_inputs(&pipe, &handles);
        let m = typed(&bitcode, TABLE, k, &p);
        let ds = depths(frames);
        let want: Vec<u32> = ds
            .iter()
            .flat_map(|&d| [sum(d, frames), even(d, frames)])
            .collect();
        assert_eq!(
            dispatch(&m, &ds, 2, Some(&tables)),
            want,
            "depth {k}: {ds:?}"
        );
    }
}

/// A frame keeps its function's private array across the recursive call: at depth 8 every one of
/// the 8 activations the stack holds (one linked function: `k`) answers its own array's weighted
/// sum.
#[test]
fn a_recursive_frame_keeps_its_private_array() {
    let Some(bitcode) = module() else {
        return;
    };
    let p = profile("moltenvk-m3", Default::default()).expect("profile");
    let (pipe, handles) = pipeline(&bitcode, BIG, 8);
    let tables = table_inputs(&pipe, &handles);
    let m = typed(&bitcode, BIG, 8, &p);
    let ds: Vec<u32> = vec![0, 1, 2, 5, 6, 7, 8];
    let want: Vec<u32> = ds.iter().map(|&d| big(d, 8)).collect();
    assert_eq!(dispatch(&m, &ds, 1, Some(&tables)), want, "{ds:?}");
}

/// `Unsupported{Recursion}` needing `required` bytes of Function storage a thread, of which the
/// device holds `available`.
fn needs_storage(required: u64, available: u64) -> Result<(), Error> {
    Err(Error::Unsupported(Unsupported::needs_any_of(
        Feature::Recursion,
        vec![vec![Capability::Limit {
            limit: VkLimit::FunctionStorageBytesPerThread,
            required,
            available,
        }]],
    )))
}

/// The kernels at depth 8 pass L3 on every committed profile that reads tables from argument
/// buffers (the direct one on every profile), and compute there what they compute on the first.
/// Their stacks fit every device's Function storage, the floor of a device not measured among
/// them, which holds them with no note (`PrecisionNote::FunctionStorageAssumed` is past it).
#[test]
fn recursive_kernels_validate_and_agree_on_every_profile() {
    let Some(bitcode) = module() else {
        return;
    };
    let (pipe, handles) = pipeline(&bitcode, TABLE, 8);
    let tables = table_inputs(&pipe, &handles);
    let check = |stem: &str, entry: &str, p: &crate::device::DeviceProfile| {
        let a = compile(&bitcode, entry, 8, p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
        let noted = a.precision.contains(&PrecisionNote::FunctionStorageAssumed);
        assert!(!noted, "{stem} {entry}");
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        check(&stem, DIRECT, &p);
        check(&stem, POINTER, &p);
    }
    let ds = depths(24);
    let want: Vec<u32> = ds.iter().flat_map(|&d| [sum(d, 24), even(d, 24)]).collect();
    for (stem, p) in capable_profiles() {
        for entry in [TABLE, BIG] {
            check(&stem, entry, &p);
        }
        let m = typed(&bitcode, TABLE, 8, &p);
        assert_eq!(dispatch(&m, &ds, 2, Some(&tables)), want, "{stem}");
    }
}

/// A depth whose stack passes Metal's 261728 bytes a thread is refused as Metal refuses it:
/// Metal's stack is `k` times the linked functions' frames, each rounded up to 16 bytes. The
/// probes' refusals (an M4 under macOS 27.0.1): `drec` alone (16 bytes) at 16359; `rec_big` alone
/// (272) at 963; `rec_big` with `rec_sum` (288) at 909; the chain `chain_a`, `chain_b` (16 each)
/// and the leaf `chain_c` (0) at 8180; `prec` recursing through `apply`, which folds into it (16),
/// at 16359. The sums (48) are refused at 5453, and a non-recursive call through a visible-function
/// reference to a leaf (`leaf_caller`, 16) at 16359. One depth less is not.
#[test]
fn a_stack_past_metals_is_refused() {
    let Some(bitcode) = module() else {
        return;
    };
    let p = profile("moltenvk-m3", Default::default()).expect("profile");
    let refused = Err(Error::LimitExceeded(Limit::CallStackBytes));
    let cases: [(&str, &[&str], u32); 7] = [
        (DIRECT, &["drec"], 16359),
        (POINTER, &["prec"], 16359),
        (BIG, &["rec_big"], 963),
        (BIG, &["rec_big", "rec_sum"], 909),
        (BIG, &["chain_a", "chain_b", "chain_c"], 8180),
        (TABLE, &SUMS, 5453),
        (BIG, &["leaf_caller", "leaf"], 16359),
    ];
    for (entry, names, past) in cases {
        let r = compile_linked(&bitcode, entry, names, past, &p).map(|_| ());
        assert_eq!(r, refused, "{entry} {names:?} at {past}");
        let r = compile_linked(&bitcode, entry, names, past - 1, &p).map(|_| ());
        assert_ne!(r, refused, "{entry} {names:?} at {}", past - 1);
    }
}

/// Stacks past the device's Function storage are `Unsupported{Recursion}` needing more, counted
/// with the rest of the kernel's run-time-indexed private memory: a stack is indexed by its depth,
/// and the 4-byte variable each machine keeps its depth in is reached at a constant offset, so it
/// counts nothing. Every device holds the floor's 16 KiB a thread: on the RTX 5080's profile
/// `drec`'s `k` 8-byte frames fit to depth 2048; the table kernel's two stacks of `3k` 8-byte frames
/// fit at depth 341 and not at 342; the large frame's `k` of 264 bytes fit at depth 62 and not at
/// 63; on RDNA 3's `drec`'s frames fit to depth 2048 too.
#[test]
fn stacks_past_the_device_s_function_storage_are_refused() {
    let Some(bitcode) = module() else {
        return;
    };
    let (nv, radv, kib) = ("nvidia-rtx5080-laptop", "amd-rdna3-radv", 1024);
    let cases = [
        (nv, DIRECT, 2048, 2049 * 8, 16 * kib),
        (nv, TABLE, 341, 2 * 3 * 342 * 8, 16 * kib),
        (nv, BIG, 62, 63 * 264, 16 * kib),
        (radv, DIRECT, 2048, 2049 * 8, 16 * kib),
    ];
    let mut wrong = Vec::new();
    for (stem, entry, fits, required, available) in cases {
        let p = profile(stem, Default::default()).expect("profile");
        if let Err(e) = compile(&bitcode, entry, fits, &p) {
            wrong.push(format!("{stem} {entry} at {fits}: {e}"));
        }
        let r = compile(&bitcode, entry, fits + 1, &p).map(|_| ());
        if r != needs_storage(required, available) {
            wrong.push(format!("{stem} {entry} at {}: {r:?}", fits + 1));
        }
    }
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// Depth 0 is Metal's default, 1: the same answers, and through the public API the same
/// pipeline; a deeper one is another pipeline.
#[test]
fn depth_zero_is_depth_one() {
    let Some(bitcode) = module() else {
        return;
    };
    let p = profile("moltenvk-m3", Default::default()).expect("profile");
    let ds = depths(3);
    let one = typed(&bitcode, DIRECT, 1, &p);
    let zero = typed(&bitcode, DIRECT, 0, &p);
    assert_eq!(dispatch(&zero, &ds, 1, None), dispatch(&one, &ds, 1, None));
    let key = |k| pipeline(&bitcode, TABLE, k).0.key;
    assert_eq!(key(0), key(1));
    assert_ne!(key(1), key(8));
}

/// A compiled kernel's stack is sized for the depth it was compiled under
/// (`RuntimeFact::CallStackDepth`): it serves a pipeline of that depth, and one of another depth
/// is `InvalidDescriptor(MaxCallStackDepth)`, since its stack would hold other frames than the
/// pipeline's Metal kernel has.
#[test]
fn a_compiled_kernel_serves_only_its_depth() {
    let Some(bitcode) = module() else {
        return;
    };
    let dev = device("moltenvk-m3");
    let lib = Library::from_air(&bitcode).expect("library");
    let f = with_drec(&lib, DIRECT);
    let compiled = dev
        .compile_function(&f, &depth(8), &Budget::standard())
        .expect("compiled");
    let fact = RuntimeFact::CallStackDepth { depth: 8 };
    assert!(compiled.artifact().facts.contains(&fact));
    let pipeline = |k| {
        let mut d = ComputePipelineDescriptor::new(StageSource::Compiled(compiled.clone()));
        d.max_call_stack_depth = k;
        dev.compile_compute_pipeline(&d, &Budget::standard())
            .map(|_| ())
    };
    assert_eq!(pipeline(8), Ok(()));
    assert_eq!(pipeline(1), other_depth(8, 1));
    assert_eq!(pipeline(0), other_depth(8, 0));
}

/// The function `name` of `bitcode`'s library with `drec` linked.
fn with_drec(lib: &Library, name: &str) -> FunctionDescriptor {
    let linked = LinkedFunctions::new().with_function(lib.function("drec").expect("drec"));
    FunctionDescriptor::new(lib.function(name).expect("function")).with_linked(linked)
}

/// `InvalidDescriptor(MaxCallStackDepth)` for a stage compiled at `compiled` in a pipeline of
/// depth `given` for it.
fn other_depth(compiled: u32, given: u32) -> Result<(), Error> {
    Err(Error::InvalidDescriptor(
        DescriptorError::MaxCallStackDepth { compiled, given },
    ))
}

/// A render pipeline's vertex and fragment functions recurse on stacks of its
/// `maxVertexCallStackDepth` and `maxFragmentCallStackDepth`, each compiled as a kernel is under
/// its pipeline's depth: each stage's key inputs carry its depth past Metal's default, 1, so a
/// default descriptor (or depth 0) keeps its keys, and each stage records the depth its stack
/// holds. A `Compiled` stage serves only a pipeline of its depth for that stage.
#[test]
fn render_stages_recurse_under_their_pipeline_s_depths() {
    let Some(bitcode) = module() else {
        return;
    };
    let dev = device("moltenvk-m3");
    let lib = Library::from_air(&bitcode).expect("library");
    let (v, f) = (with_drec(&lib, "v_rec"), with_drec(&lib, "f_rec"));
    let descriptor = |vertex: StageSource, fragment: StageSource, (vk, fk): (u32, u32)| {
        let mut d = RenderPipelineDescriptor::new(vertex);
        d.fragment = Some(fragment);
        d.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
        d.max_vertex_call_stack_depth = vk;
        d.max_fragment_call_stack_depth = fk;
        d
    };
    let functions = |depths| {
        descriptor(
            StageSource::Function(v.clone()),
            StageSource::Function(f.clone()),
            depths,
        )
    };
    let b = Budget::standard();
    let keys = |depths| {
        dev.render_pipeline_key(&functions(depths), &b)
            .unwrap_or_else(|e| panic!("keys at {depths:?}: {e}"))
    };
    let depths = |k: &RenderPipelineKeys| {
        let of = |i: &Option<LowerInputs>| i.as_ref().and_then(|i| i.max_call_stack_depth);
        (of(&k.vertex_inputs), of(&k.fragment_inputs))
    };
    let (one, deep) = (keys((1, 1)), keys((8, 3)));
    assert_eq!(keys((0, 0)), one);
    assert_eq!(depths(&one), (None, None));
    assert_eq!(depths(&deep), (Some(8), Some(3)));
    assert_ne!(one.pipeline, deep.pipeline);
    assert_ne!(one.vertex, deep.vertex);
    assert_ne!(one.fragment, deep.fragment);
    dev.compile_render_pipeline(&functions((8, 3)), &b)
        .unwrap_or_else(|e| panic!("the pipeline at (8, 3): {e}"));
    let compile = |fd: &FunctionDescriptor, inputs: &Option<LowerInputs>| {
        let inputs = inputs.as_ref().expect("a Function stage's inputs");
        dev.compile_function(fd, inputs, &b)
            .unwrap_or_else(|e| panic!("a stage: {e}"))
    };
    let (cv, cf) = (
        compile(&v, &deep.vertex_inputs),
        compile(&f, &deep.fragment_inputs),
    );
    let fact = |depth| RuntimeFact::CallStackDepth { depth };
    assert!(cv.artifact().facts.contains(&fact(8)));
    assert!(cf.artifact().facts.contains(&fact(3)));
    for (stage, c) in [("vertex", &cv), ("fragment", &cf)] {
        link_and_validate(c.artifact(), dev.profile())
            .unwrap_or_else(|e| panic!("the {stage} stage: L3: {e}"));
    }
    let compiled = |depths| {
        let d = descriptor(
            StageSource::Compiled(cv.clone()),
            StageSource::Compiled(cf.clone()),
            depths,
        );
        dev.compile_render_pipeline(&d, &b).map(|_| ())
    };
    assert_eq!(compiled((8, 3)), Ok(()));
    assert_eq!(compiled((1, 3)), other_depth(8, 1));
    assert_eq!(compiled((8, 0)), other_depth(3, 0));
}

/// A tile function recurses on a stack of its tile pipeline's `maxCallStackDepth`, as a kernel does
/// on its compute pipeline's: the key carries the depth past Metal's default, 1, the function
/// records the depth its stack holds, and a `Compiled` one serves only a pipeline of its depth.
#[test]
fn a_tile_function_recurses_under_its_pipeline_s_depth() {
    let Some(bitcode) = module() else {
        return;
    };
    let dev = device("moltenvk-m3");
    let lib = Library::from_air(&bitcode).expect("library");
    let t = with_drec(&lib, "t_rec");
    let descriptor = |src: StageSource, k: u32| {
        let mut d = TileRenderPipelineDescriptor::new(src);
        d.color_attachment_formats[0] = Some(PixelFormat::R32Float);
        d.max_call_stack_depth = k;
        d
    };
    let key = |k| dev.tile_pipeline_key(&descriptor(StageSource::Function(t.clone()), k));
    assert_eq!(key(0), key(1));
    assert_ne!(key(1), key(8));
    let b = Budget::standard();
    dev.compile_tile_pipeline(&descriptor(StageSource::Function(t.clone()), 8), &b)
        .unwrap_or_else(|e| panic!("the tile pipeline at 8: {e}"));
    let inputs = LowerInputs {
        tile_attachments: Some([(0, PixelFormat::R32Float)].into()),
        max_call_stack_depth: Some(8),
        ..LowerInputs::new()
    };
    let compiled = dev
        .compile_function(&t, &inputs, &b)
        .unwrap_or_else(|e| panic!("the tile function at 8: {e}"));
    let fact = RuntimeFact::CallStackDepth { depth: 8 };
    assert!(compiled.artifact().facts.contains(&fact));
    link_and_validate(compiled.artifact(), dev.profile())
        .unwrap_or_else(|e| panic!("the tile function: L3: {e}"));
    let pipeline = |k| {
        let d = descriptor(StageSource::Compiled(compiled.clone()), k);
        dev.compile_tile_pipeline(&d, &b).map(|_| ())
    };
    assert_eq!(pipeline(8), Ok(()));
    assert_eq!(pipeline(1), other_depth(8, 1));
}

/// A recursion that closes through a function pointer passed as a value: `prec` passes its own
/// address to `apply`, which calls through it, and the kernel calls `apply` with `prec`'s address
/// too. The pointer is that constant at every call, so each `apply` calls `prec` itself on the
/// explicit call stack. `apply`, which no pipeline links, folds into `prec`, so at depth
/// `k` `prec` makes `k - 1` recursive calls, as plain Metal does and as direct
/// recursion does, whether the pipeline links `prec` in `functions`, where a table can hold it, or
/// in `private_functions`, where none can. A call past them yields zero.
#[test]
fn recursion_through_a_pointer_passed_as_a_value_makes_one_call_fewer_than_its_depth() {
    let Some(bitcode) = module() else {
        return;
    };
    let p = profile("moltenvk-m3", Default::default()).expect("profile");
    let none = FunctionConstantValues::defaults();
    for list in [LinkedList::Functions, LinkedList::PrivateFunctions] {
        let mut linked = LinkedSource::new("prec", &bitcode, &none);
        linked.key.list = list;
        for k in [1, 2, 8] {
            let m = typed_module_linked_with(&bitcode, POINTER, &p, &[linked], &depth(k))
                .unwrap_or_else(|e| panic!("{list:?} at depth {k}: {e}"))
                .1;
            let ds: Vec<u32> = (0..2 * k + 2).collect();
            let want: Vec<u32> = ds.iter().map(|&d| sum(d, k)).collect();
            assert_eq!(dispatch(&m, &ds, 1, None), want, "{list:?} at depth {k}");
        }
    }
}
