//! Lockstep made explicit.
//!
//! Apple GPUs run a simdgroup's 32 threads in lockstep, so a kernel may let one thread read a
//! threadgroup word another thread of its simdgroup overwrites one instruction later, with no
//! barrier between them, and still read the old word. Vulkan orders nothing there: lavapipe runs
//! a workgroup in chunks of invocations, each up to the next barrier, and the CPU interpreter
//! models that schedule with 8-lane subgroups that each run to their next barrier
//! (`SubgroupSize::W8`), in both orders. Each kernel below is written from scratch so that the
//! chunked schedule reads a word another chunk already overwrote; the translation inserts the
//! barrier lockstep stood for (`lower::simd::lockstep`), and every schedule then reads what
//! lockstep reads:
//!
//! - a conflict in workgroup-uniform control flow takes a threadgroup barrier and raises no
//!   simd class (`barrier.lockstep.threadgroup`);
//! - a conflict in one block of a divergent region takes a simdgroup barrier, a subgroup barrier
//!   on a 32-lane subgroup and a predicated workgroup barrier on emulated simdgroups
//!   (`barrier.lockstep.simdgroup`), also where only Metal's barrier contract makes the region's
//!   simdgroups run together;
//! - a conflict across the iterations of a loop whose only simdgroup-uniform position is the
//!   loop's header takes a simdgroup barrier there;
//! - a conflict no barrier position orders on a subgroup (an arm then a divergent loop) takes a
//!   simdgroup barrier that only emulated simdgroups make exact, and the kernel takes the
//!   `Emulated` tier (`barrier.lockstep.emulated.simdgroup`), except on a device that runs a
//!   simdgroup's 32 threads in lockstep itself (moltenvk-m3), where the kernel keeps its tier
//!   and requires that fact (`barrier.lockstep.device`).
//!
//! Accesses the IR proves disjoint between the threads of a simdgroup (each thread's own word,
//! two objects, two constant words, words further apart than a simdgroup spans, words a loop
//! counter's bounds keep apart, each thread's own word of whichever row an iteration picks, a
//! counter stepping by more than its threads' starts differ, a tree reduction's halves under
//! its guard) take nothing, and so do stores of one value every thread stores alike and accesses
//! one thread of a simdgroup at most runs (`if (t == 0)`). No GPU is used; `llvm-as` assembles the
//! kernels and a missing tool skips visibly.

use std::collections::BTreeMap;

use crate::device::requirements::VkProperty;
use crate::device::subgroup::Tier;
use crate::device::DevicePolicy;
use crate::lower::select::Plan;
use crate::pipeline::sym::{Axis, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, llvm_assemble, profile, typed_module, AssembleError, Scratch,
};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources, SubgroupSize};
use crate::spirv::ir::TypedModule;

const HEADER: &str = r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@tg = internal unnamed_addr addrspace(3) global [64 x i32] undef, align 4
@other = internal unnamed_addr addrspace(3) global [64 x i32] undef, align 4
@src = internal unnamed_addr addrspace(3) global [64 x i32] undef, align 4
@wide = internal unnamed_addr addrspace(3) global [128 x i32] undef, align 4
@tile = internal unnamed_addr addrspace(3) global [4 x [32 x i32]] undef, align 4
"#;

/// The metadata of kernel `name(device uint *out [[buffer(0)]], uint t
/// [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]])`.
fn footer(name: &str) -> String {
    format!(
        r#"
declare void @air.wg.barrier(i32, i32) #1
declare i32 @air.simd_sum.u.i32(i32) #1

attributes #0 = {{ convergent mustprogress nounwind willreturn }}
attributes #1 = {{ convergent mustprogress nounwind willreturn }}

!air.kernel = !{{!9}}
!air.version = !{{!18}}
!air.language_version = !{{!19}}

!9 = !{{ptr @{name}, !10, !11}}
!10 = !{{}}
!11 = !{{!12, !13, !14}}
!12 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!13 = !{{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}}
!14 = !{{i32 2, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"sg"}}
!18 = !{{i32 2, i32 8, i32 0}}
!19 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// `body` (the definition of kernel `name`) assembled with the module's header and metadata.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    assemble_text(name, &format!("{HEADER}\n{body}\n{}", footer(name)))
}

/// Module `text` (kernel `name`) assembled.
fn assemble_text(name: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-lockstep-{name}"));
    let dir = scratch.path();
    let path = dir.join("kernel.ll");
    std::fs::write(&path, text).expect("write");
    let bc = match llvm_assemble(&path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    };
    bc
}

/// Thread `t` stores `t`; after a barrier it reads the word of thread `t ^ 8` (in its simdgroup,
/// in another 8-lane chunk) and overwrites its own word in the same barrier interval. Lockstep
/// reads every word before any is overwritten: `out[t] = t ^ 8`.
const NEIGHBOR: &str = r#"
define void @neighbor(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 %1, ptr addrspace(3) %4, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %5 = xor i32 %1, 8
  %6 = zext i32 %5 to i64
  %7 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %6
  %8 = load i32, ptr addrspace(3) %7, align 4
  store i32 100, ptr addrspace(3) %4, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %9 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %8, ptr addrspace(1) %9, align 4
  ret void
}
"#;

/// The threads of simdgroup 0 alone rotate their words: thread `t` reads the word of thread
/// `(t + 1) & 31` and overwrites its own with twice it, in a block every thread of simdgroup 0
/// runs and no other thread does. Lockstep: `out[t] = 2 ((t + 1) & 31)` below 32, `t` above.
const ROTATE: &str = r#"
define void @rotate(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 %1, ptr addrspace(3) %4, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %5 = icmp eq i32 %sg, 0
  br i1 %5, label %walk, label %done

walk:
  %6 = add i32 %1, 1
  %7 = and i32 %6, 31
  %8 = zext i32 %7 to i64
  %9 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %8
  %10 = load i32, ptr addrspace(3) %9, align 4
  %11 = shl i32 %10, 1
  store i32 %11, ptr addrspace(3) %4, align 4
  br label %done

done:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %12 = load i32, ptr addrspace(3) %4, align 4
  %13 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %12, ptr addrspace(1) %13, align 4
  ret void
}
"#;

/// `rotate` behind a branch on a word of `out` (0, so every thread takes it): the analysis
/// cannot prove a loaded word uniform, but the block the branch enters runs exactly when the
/// barrier block after it does (it dominates that block, which post-dominates it, and neither is
/// in a loop), and Metal requires every thread to reach that barrier. So the block is uniform,
/// its branch on the simdgroup index is uniform within each simdgroup, and the rotation takes a
/// simdgroup barrier. Lockstep: as `rotate`.
const CONTRACT: &str = r#"
define void @contract(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 %1, ptr addrspace(3) %4, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %flag = load i32, ptr addrspace(1) %0, align 4
  %skip = icmp eq i32 %flag, 7
  br i1 %skip, label %exit, label %work

work:
  %5 = icmp eq i32 %sg, 0
  br i1 %5, label %walk, label %done

walk:
  %6 = add i32 %1, 1
  %7 = and i32 %6, 31
  %8 = zext i32 %7 to i64
  %9 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %8
  %10 = load i32, ptr addrspace(3) %9, align 4
  %11 = shl i32 %10, 1
  store i32 %11, ptr addrspace(3) %4, align 4
  br label %done

done:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %12 = load i32, ptr addrspace(3) %4, align 4
  %13 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %12, ptr addrspace(1) %13, align 4
  br label %exit

exit:
  ret void
}
"#;

/// The threads of simdgroup 0 alone run two iterations of: read the word of thread
/// `(t + 1) & 31`, overwrite their own with it plus one. Each iteration reads what the previous
/// one wrote, so the order holds within an iteration and across them. Lockstep:
/// `out[t] = ((t + 2) & 31) + 2` below 32, `t` above.
const CARRY: &str = r#"
define void @carry(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 %1, ptr addrspace(3) %4, align 4
  %5 = add i32 %1, 1
  %6 = and i32 %5, 31
  %7 = zext i32 %6 to i64
  %8 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %7
  tail call void @air.wg.barrier(i32 2, i32 1)
  %9 = icmp eq i32 %sg, 0
  br i1 %9, label %loop, label %done

loop:
  %10 = phi i32 [ 0, %2 ], [ %13, %loop ]
  %11 = load i32, ptr addrspace(3) %8, align 4
  %12 = add i32 %11, 1
  store i32 %12, ptr addrspace(3) %4, align 4
  %13 = add i32 %10, 1
  %14 = icmp ult i32 %13, 2
  br i1 %14, label %loop, label %done

done:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %15 = load i32, ptr addrspace(3) %4, align 4
  %16 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %15, ptr addrspace(1) %16, align 4
  ret void
}
"#;

/// The even threads of simdgroup 0 read the word of thread `(t + 2) & 31` in a branch arm; then
/// every thread of simdgroup 0 overwrites its own word twice in a loop the arm joins. Lockstep
/// runs the arm before the loop: `out[t] = (t + 2) & 31` for even `t` below 32, else 0.
const ARM_THEN_LOOP: &str = r#"
define void @arm_then_loop(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 %1, ptr addrspace(3) %4, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %5 = icmp ult i32 %1, 32
  br i1 %5, label %w, label %done

w:
  %6 = and i32 %1, 1
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %arm, label %loop

arm:
  %8 = add i32 %1, 2
  %9 = and i32 %8, 31
  %10 = zext i32 %9 to i64
  %11 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %10
  %12 = load i32, ptr addrspace(3) %11, align 4
  %13 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %12, ptr addrspace(1) %13, align 4
  br label %loop

loop:
  %14 = phi i32 [ 0, %w ], [ 0, %arm ], [ %16, %loop ]
  %15 = add i32 %14, 5
  store i32 %15, ptr addrspace(3) %4, align 4
  %16 = add i32 %14, 1
  %17 = icmp ult i32 %16, 2
  br i1 %17, label %loop, label %done

done:
  tail call void @air.wg.barrier(i32 2, i32 1)
  ret void
}
"#;

/// `arm_then_loop` whose loop also stores each iteration's value to `out[t + 32]` through a
/// pointer carried in a struct, as MSL carries the buffers it passes by value (MPS's
/// `_MPSKernelInOut`): the entry block inserts the pointer into a `{ ptr, i32 }`, and the loop
/// stores through the member it extracts. That `extractvalue` reads the inserted pointer where it
/// stands (a forwarding extract), from another block, in the block the predicated walk of the
/// forced `Emulated` tier splits at the lockstep barrier (right before the loop's threadgroup
/// store). Lockstep: as `arm_then_loop` below 32, and the last iteration's 6 from 32 on.
const FORWARD_CROSS: &str = r#"
define void @forward_cross(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 %1, ptr addrspace(3) %4, align 4
  %up = add i64 %3, 32
  %far = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %up
  %slot = insertvalue { ptr addrspace(1), i32 } undef, ptr addrspace(1) %far, 0
  tail call void @air.wg.barrier(i32 2, i32 1)
  %5 = icmp ult i32 %1, 32
  br i1 %5, label %w, label %done

w:
  %6 = and i32 %1, 1
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %arm, label %loop

arm:
  %8 = add i32 %1, 2
  %9 = and i32 %8, 31
  %10 = zext i32 %9 to i64
  %11 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %10
  %12 = load i32, ptr addrspace(3) %11, align 4
  %13 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %12, ptr addrspace(1) %13, align 4
  br label %loop

loop:
  %14 = phi i32 [ 0, %w ], [ 0, %arm ], [ %16, %loop ]
  %15 = add i32 %14, 5
  %to = extractvalue { ptr addrspace(1), i32 } %slot, 0
  store i32 %15, ptr addrspace(1) %to, align 4
  store i32 %15, ptr addrspace(3) %4, align 4
  %16 = add i32 %14, 1
  %17 = icmp ult i32 %16, 2
  br i1 %17, label %loop, label %done

done:
  tail call void @air.wg.barrier(i32 2, i32 1)
  ret void
}
"#;

/// `forward_cross` with the struct built in the loop before the lockstep barrier and the
/// pointer extracted after it, so the forwarded pointer crosses the barrier, one of the
/// predicated walk's synchronizations, within its block. Lockstep: as `forward_cross`.
const FORWARD_SYNC: &str = r#"
define void @forward_sync(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 %1, ptr addrspace(3) %4, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %5 = icmp ult i32 %1, 32
  br i1 %5, label %w, label %done

w:
  %6 = and i32 %1, 1
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %arm, label %loop

arm:
  %8 = add i32 %1, 2
  %9 = and i32 %8, 31
  %10 = zext i32 %9 to i64
  %11 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %10
  %12 = load i32, ptr addrspace(3) %11, align 4
  %13 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %12, ptr addrspace(1) %13, align 4
  br label %loop

loop:
  %14 = phi i32 [ 0, %w ], [ 0, %arm ], [ %16, %loop ]
  %15 = add i32 %14, 5
  %up = add i64 %3, 32
  %far = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %up
  %pair = insertvalue { ptr addrspace(1), i32 } undef, ptr addrspace(1) %far, 0
  store i32 %15, ptr addrspace(3) %4, align 4
  %to = extractvalue { ptr addrspace(1), i32 } %pair, 0
  store i32 %15, ptr addrspace(1) %to, align 4
  %16 = add i32 %14, 1
  %17 = icmp ult i32 %16, 2
  br i1 %17, label %loop, label %done

done:
  tail call void @air.wg.barrier(i32 2, i32 1)
  ret void
}
"#;

/// `arm_then_loop` whose threads store a value through a struct merged at the end: the entry block
/// inserts `out + t`, the loop inserts the same address computed in bytes before its lockstep
/// barrier, and a phi of the two (a member-wise phi) selects the pointer `done` stores through. The
/// loop's copy into that phi reads the loop's pointer after the barrier, one of the predicated
/// walk's synchronizations, within its block. The stored value is what the arm read, carried
/// through the loop. Lockstep: as `arm_then_loop` below 32, and `t` from 32 on.
const MEMBER_SYNC: &str = r#"
define void @member_sync(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 %1, ptr addrspace(3) %4, align 4
  %out = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  %own = insertvalue { ptr addrspace(1), i32 } undef, ptr addrspace(1) %out, 0
  tail call void @air.wg.barrier(i32 2, i32 1)
  %5 = icmp ult i32 %1, 32
  br i1 %5, label %w, label %done

w:
  %6 = and i32 %1, 1
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %arm, label %loop

arm:
  %8 = add i32 %1, 2
  %9 = and i32 %8, 31
  %10 = zext i32 %9 to i64
  %11 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %10
  %12 = load i32, ptr addrspace(3) %11, align 4
  br label %loop

loop:
  %13 = phi i32 [ 0, %w ], [ 0, %arm ], [ %16, %loop ]
  %14 = phi i32 [ 0, %w ], [ %12, %arm ], [ %14, %loop ]
  %15 = add i32 %13, 5
  %bytes = shl i64 %3, 2
  %here = getelementptr inbounds i8, ptr addrspace(1) %0, i64 %bytes
  %moved = insertvalue { ptr addrspace(1), i32 } undef, ptr addrspace(1) %here, 0
  store i32 %15, ptr addrspace(3) %4, align 4
  %16 = add i32 %13, 1
  %17 = icmp ult i32 %16, 2
  br i1 %17, label %loop, label %done

done:
  %dst = phi { ptr addrspace(1), i32 } [ %own, %2 ], [ %moved, %loop ]
  %res = phi i32 [ %1, %2 ], [ %14, %loop ]
  tail call void @air.wg.barrier(i32 2, i32 1)
  %p = extractvalue { ptr addrspace(1), i32 } %dst, 0
  store i32 %res, ptr addrspace(1) %p, align 4
  ret void
}
"#;

/// Each thread reads and rewrites its own word between barriers, reads two constant words of a
/// second object and writes a third, and moves the word of thread `t ^ 8` of a third object into
/// its own word: no two threads touch one byte in one interval, so nothing is inserted.
/// `out[t] = t + 1 + 7 + (t ^ 8)`.
const DISJOINT: &str = r#"
define void @disjoint(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 %1, ptr addrspace(3) %4, align 4
  %5 = getelementptr inbounds [64 x i32], ptr addrspace(3) @src, i64 0, i64 %3
  store i32 %1, ptr addrspace(3) %5, align 4
  %6 = icmp eq i32 %1, 0
  br i1 %6, label %seed, label %go

seed:
  store i32 3, ptr addrspace(3) getelementptr inbounds ([64 x i32], ptr addrspace(3) @other, i64 0, i64 40), align 4
  store i32 4, ptr addrspace(3) getelementptr inbounds ([64 x i32], ptr addrspace(3) @other, i64 0, i64 41), align 4
  br label %go

go:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %7 = load i32, ptr addrspace(3) %4, align 4
  %8 = add i32 %7, 1
  store i32 %8, ptr addrspace(3) %4, align 4
  %9 = load i32, ptr addrspace(3) %4, align 4
  %10 = load i32, ptr addrspace(3) getelementptr inbounds ([64 x i32], ptr addrspace(3) @other, i64 0, i64 40), align 4
  %11 = load i32, ptr addrspace(3) getelementptr inbounds ([64 x i32], ptr addrspace(3) @other, i64 0, i64 41), align 4
  %12 = add i32 %10, %11
  store i32 %12, ptr addrspace(3) getelementptr inbounds ([64 x i32], ptr addrspace(3) @other, i64 0, i64 42), align 4
  %13 = xor i32 %1, 8
  %14 = zext i32 %13 to i64
  %15 = getelementptr inbounds [64 x i32], ptr addrspace(3) @src, i64 0, i64 %14
  %16 = load i32, ptr addrspace(3) %15, align 4
  store i32 %16, ptr addrspace(3) %4, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %17 = add i32 %9, %12
  %18 = load i32, ptr addrspace(3) %4, align 4
  %19 = add i32 %17, %18
  %20 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %19, ptr addrspace(1) %20, align 4
  ret void
}
"#;

/// Thread `t` writes words `t` and `t + 64` of one object (the index of the second adds a
/// constant to the thread's index, which cannot wrap: a threadgroup has at most 1024 threads),
/// then reads word `t + 64` and overwrites word `t`: two threads of a simdgroup are at most 31
/// words apart, so they never touch one word. Every thread then stores 7 to one word, twice, and
/// reads it back: whichever thread's store lands last, the word holds 7. Nothing is inserted.
/// `out[t] = t + 64 + 7`.
const OFFSETS: &str = r#"
define void @offsets(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %hi = add i32 %1, 64
  %4 = zext i32 %hi to i64
  %5 = getelementptr inbounds [128 x i32], ptr addrspace(3) @wide, i64 0, i64 %3
  %6 = getelementptr inbounds [128 x i32], ptr addrspace(3) @wide, i64 0, i64 %4
  store i32 %1, ptr addrspace(3) %5, align 4
  store i32 %hi, ptr addrspace(3) %6, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %7 = load i32, ptr addrspace(3) %6, align 4
  store i32 %7, ptr addrspace(3) %5, align 4
  store i32 7, ptr addrspace(3) getelementptr inbounds ([64 x i32], ptr addrspace(3) @other, i64 0, i64 5), align 4
  store i32 7, ptr addrspace(3) getelementptr inbounds ([64 x i32], ptr addrspace(3) @other, i64 0, i64 5), align 4
  %8 = load i32, ptr addrspace(3) getelementptr inbounds ([64 x i32], ptr addrspace(3) @other, i64 0, i64 5), align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %9 = load i32, ptr addrspace(3) %5, align 4
  %10 = add i32 %9, %8
  %11 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %10, ptr addrspace(1) %11, align 4
  ret void
}
"#;

/// Thread `t` stores `t + 1` to word `t`; after a barrier, a counted loop reads words 0 to 2
/// (the counter's exit bound keeps them there) and thread `t` then writes word `t + 64`: the
/// bytes the loop may read lie before any the write may touch, so nothing is inserted.
/// `out[t] = 1 + 2 + 3`.
const FIELDS: &str = r#"
define void @fields(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds [128 x i32], ptr addrspace(3) @wide, i64 0, i64 %3
  %5 = add i32 %1, 1
  store i32 %5, ptr addrspace(3) %4, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %loop

loop:
  %a = phi i32 [ 0, %2 ], [ %next, %loop ]
  %s = phi i32 [ 0, %2 ], [ %sum, %loop ]
  %6 = zext i32 %a to i64
  %7 = getelementptr inbounds [128 x i32], ptr addrspace(3) @wide, i64 0, i64 %6
  %8 = load i32, ptr addrspace(3) %7, align 4
  %sum = add i32 %s, %8
  %next = add nuw nsw i32 %a, 1
  %9 = icmp eq i32 %next, 3
  br i1 %9, label %done, label %loop

done:
  %hi = add i32 %1, 64
  %10 = zext i32 %hi to i64
  %11 = getelementptr inbounds [128 x i32], ptr addrspace(3) @wide, i64 0, i64 %10
  store i32 %sum, ptr addrspace(3) %11, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %12 = load i32, ptr addrspace(3) %11, align 4
  %13 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %12, ptr addrspace(1) %13, align 4
  ret void
}
"#;

/// Each of four iterations writes thread `t`'s word `t & 31` of row `i` (through a loop that
/// starts at that word and steps by 32 while below 0, so it never repeats): across iterations
/// the row changes, but the offset into it stays each thread's own, so nothing is inserted.
/// `out[t] = t + 3`.
const ROWS: &str = r#"
define void @rows(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %lane = and i32 %1, 31
  br label %outer

outer:
  %i = phi i32 [ 0, %2 ], [ %inext, %latch ]
  %4 = zext i32 %i to i64
  %v = add i32 %1, %i
  br label %inner

inner:
  %j = phi i32 [ %lane, %outer ], [ %jnext, %inner ]
  %5 = sext i32 %j to i64
  %6 = getelementptr inbounds [4 x [32 x i32]], ptr addrspace(3) @tile, i64 0, i64 %4, i64 %5
  store i32 %v, ptr addrspace(3) %6, align 4
  %jnext = add nsw i32 %j, 32
  %7 = icmp slt i32 %j, 0
  br i1 %7, label %inner, label %latch

latch:
  %inext = add nuw nsw i32 %i, 1
  %8 = icmp eq i32 %inext, 4
  br i1 %8, label %end, label %outer

end:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %9 = getelementptr inbounds [4 x [32 x i32]], ptr addrspace(3) @tile, i64 0, i64 3, i64 %3
  %10 = load i32, ptr addrspace(3) %9, align 4
  %11 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %10, ptr addrspace(1) %11, align 4
  ret void
}
"#;

/// Thread `t` writes words `t`, `t + 32`, `t + 64`, and `t + 96` in a loop stepping by 32 from
/// `t`: every value the counter takes is its start plus a multiple of 32, and two threads of a
/// simdgroup start fewer than 32 apart, so at whatever iterations they never write one word and
/// nothing is inserted. `out[t] = t + 96`.
const STRIDED: &str = r#"
define void @strided(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = icmp ult i32 %1, 128
  br i1 %4, label %loop, label %done

loop:
  %i = phi i32 [ %1, %2 ], [ %next, %loop ]
  %5 = zext i32 %i to i64
  %6 = getelementptr inbounds [128 x i32], ptr addrspace(3) @wide, i64 0, i64 %5
  store i32 %i, ptr addrspace(3) %6, align 4
  %next = add nuw nsw i32 %i, 32
  %7 = icmp ult i32 %i, 96
  br i1 %7, label %loop, label %done

done:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %8 = add i32 %1, 96
  %9 = zext i32 %8 to i64
  %10 = getelementptr inbounds [128 x i32], ptr addrspace(3) @wide, i64 0, i64 %9
  %11 = load i32, ptr addrspace(3) %10, align 4
  %12 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %11, ptr addrspace(1) %12, align 4
  ret void
}
"#;

/// A tree reduction with a barrier per step: under `h > t`, thread `t` adds word `t + h` into
/// word `t`. The words written lie below `h` and the words read at or above it, so no pair
/// within a step meets and nothing is inserted. `out[t] = 0 + 1 + ... + 31`.
const GUARD: &str = r#"
define void @guard(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 %1, ptr addrspace(3) %4, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %loop

loop:
  %s = phi i32 [ 32, %2 ], [ %h, %latch ]
  %h = lshr i32 %s, 1
  %5 = icmp ugt i32 %h, %1
  br i1 %5, label %arm, label %latch

arm:
  %6 = load i32, ptr addrspace(3) %4, align 4
  %7 = add i32 %h, %1
  %8 = zext i32 %7 to i64
  %9 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %8
  %10 = load i32, ptr addrspace(3) %9, align 4
  %11 = add i32 %6, %10
  store i32 %11, ptr addrspace(3) %4, align 4
  br label %latch

latch:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %12 = icmp ult i32 %s, 4
  br i1 %12, label %done, label %loop

done:
  %13 = load i32, ptr addrspace(3) getelementptr inbounds ([64 x i32], ptr addrspace(3) @tg, i64 0, i64 0), align 4
  %14 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %13, ptr addrspace(1) %14, align 4
  ret void
}
"#;

/// [`GUARD`] in signed arithmetic, as `int` indices compile: halving by `sdiv`, the test
/// `h > t` signed, and the words `t` and `t + h` (without signed wrap) sign-extended. The
/// thread index is not negative, so `t + h` is at least `h` and the words read lie at or above
/// it. `out[t] = 0 + 1 + ... + 31`.
const SIGNED_GUARD: &str = r#"
define void @signed_guard(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = sext i32 %1 to i64
  %4 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 %1, ptr addrspace(3) %4, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  br label %loop

loop:
  %s = phi i32 [ 32, %2 ], [ %h, %latch ]
  %h = sdiv i32 %s, 2
  %5 = icmp sgt i32 %h, %1
  br i1 %5, label %arm, label %latch

arm:
  %6 = load i32, ptr addrspace(3) %4, align 4
  %7 = add nsw i32 %h, %1
  %8 = sext i32 %7 to i64
  %9 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %8
  %10 = load i32, ptr addrspace(3) %9, align 4
  %11 = add i32 %6, %10
  store i32 %11, ptr addrspace(3) %4, align 4
  br label %latch

latch:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %12 = icmp sgt i32 %s, 3
  br i1 %12, label %loop, label %done

done:
  %13 = load i32, ptr addrspace(3) getelementptr inbounds ([64 x i32], ptr addrspace(3) @tg, i64 0, i64 0), align 4
  %14 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %13, ptr addrspace(1) %14, align 4
  ret void
}
"#;

/// A tree reduction's halves from a base that differs between threads: lanes 0 to 15 index from
/// word 8 and lanes 16 to 31 from word 0, `t = lane & 15`, `h = sg + 8`. Under `t < h`, lane
/// `16 + t` reads word `t + 8`, which lane `t` writes: the guard keeps the cores apart, not the
/// addresses. Lockstep reads before writing: `out[l] = l + 16` below 8, `l - 8` from 16 to 23,
/// else 0.
const GROUP_BASE: &str = r#"
define void @group_base(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 %1, ptr addrspace(3) %4, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %h = add i32 %sg, 8
  %lane = and i32 %1, 31
  %hb = and i32 %lane, 16
  %hb2 = lshr i32 %hb, 1
  %boff = xor i32 %hb2, 8
  %bz = zext i32 %boff to i64
  %p = getelementptr inbounds i32, ptr addrspace(3) @tg, i64 %bz
  %t = and i32 %lane, 15
  %g = icmp ult i32 %t, %h
  %o = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  br i1 %g, label %arm, label %done

arm:
  %s = add nuw i32 %t, %h
  %sz = zext i32 %s to i64
  %ps = getelementptr inbounds i32, ptr addrspace(3) %p, i64 %sz
  %x = load i32, ptr addrspace(3) %ps, align 4
  %tz = zext i32 %t to i64
  %pt = getelementptr inbounds i32, ptr addrspace(3) %p, i64 %tz
  %y = add i32 %x, 1000
  store i32 %y, ptr addrspace(3) %pt, align 4
  store i32 %x, ptr addrspace(1) %o, align 4
  br label %done

done:
  ret void
}
"#;

/// A row from a base that differs between threads: `p = tg + lane` read as a `[4 x i32]` row by
/// the uniform index `u = sg & 1`, so lane `L` reads word `L + 1` and writes word `L`. The
/// offsets into the row are apart, the addresses are not. Lockstep: `out[l] = l + 1`.
const ROW_BASE: &str = r#"
define void @row_base(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 %1, ptr addrspace(3) %4, align 4
  %t32 = add i32 %1, 32
  %5 = zext i32 %t32 to i64
  %6 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %5
  store i32 %t32, ptr addrspace(3) %6, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %lane = and i32 %1, 31
  %lz = zext i32 %lane to i64
  %p = getelementptr inbounds i32, ptr addrspace(3) @tg, i64 %lz
  %u = and i32 %sg, 1
  %uz = zext i32 %u to i64
  %pa = getelementptr inbounds [4 x i32], ptr addrspace(3) %p, i64 0, i64 %uz
  %u1 = add nuw i32 %u, 1
  %u1z = zext i32 %u1 to i64
  %pb = getelementptr inbounds [4 x i32], ptr addrspace(3) %p, i64 0, i64 %u1z
  %x = load i32, ptr addrspace(3) %pb, align 4
  store i32 1000, ptr addrspace(3) %pa, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %x, ptr addrspace(1) %o, align 4
  ret void
}
"#;

/// Two rows of one size (64 bytes) and different strides, `[16 x i32]` and `[32 x i16]`, by one
/// uniform core `c = (sg + 1) % 3` (1 in simdgroup 0): the store covers bytes `[4c, 4c + 4)`,
/// the load bytes `[2c + 4, 2c + 6)`, which meet at `c = 1`. Lockstep reads word 1's high half
/// before any thread overwrites it: `out[l] = 0`.
const ROW_STRIDE: &str = r#"
define void @row_stride(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 %1, ptr addrspace(3) %4, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %w = add i32 %sg, 1
  %c = urem i32 %w, 3
  %cz = zext i32 %c to i64
  %pa = getelementptr inbounds [16 x i32], ptr addrspace(3) @tg, i64 0, i64 %cz
  %c2 = add nuw i32 %c, 2
  %c2z = zext i32 %c2 to i64
  %pb = getelementptr inbounds [32 x i16], ptr addrspace(3) @tg, i64 0, i64 %c2z
  %x = load i16, ptr addrspace(3) %pb, align 2
  store i32 458752, ptr addrspace(3) %pa, align 4
  %xz = zext i16 %x to i32
  %o = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %xz, ptr addrspace(1) %o, align 4
  ret void
}
"#;

/// A simdgroup sum inside a loop each lane leaves after its own number of iterations, read
/// after it: lane `L` keeps the sum of its last iteration, `32 - L` active lanes, so the
/// collective's result differs between lanes there. Lane `L` reads word `33 - L` and writes
/// word `32 - L`, which lane `L + 1` reads. Lockstep: `out[l] = 33 - l`.
const TEMPORAL: &str = r#"
define void @temporal(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 %1, ptr addrspace(3) %4, align 4
  %t32 = add i32 %1, 32
  %5 = zext i32 %t32 to i64
  %6 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %5
  store i32 %t32, ptr addrspace(3) %6, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %lane = and i32 %1, 31
  br label %loop

loop:
  %n = phi i32 [ 0, %2 ], [ %n1, %loop ]
  %s = tail call i32 @air.simd_sum.u.i32(i32 1)
  %n1 = add i32 %n, 1
  %c = icmp ule i32 %n1, %lane
  br i1 %c, label %loop, label %after

after:
  %sz = zext i32 %s to i64
  %i1 = add i64 %sz, 1
  %pa = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %i1
  %x = load i32, ptr addrspace(3) %pa, align 4
  %pb = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %sz
  store i32 1000, ptr addrspace(3) %pb, align 4
  %o = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %x, ptr addrspace(1) %o, align 4
  ret void
}
"#;

/// A loop whose bounds depend on the simdgroup index (uniform in a simdgroup, not in the
/// threadgroup); in each iteration `k` the even threads write `100 * k + t` to word
/// `(t + k) & 31`, in an arm only they run. Iteration `k + 1` overwrites words iteration `k`
/// wrote, and lockstep runs the iterations in order, so the last iteration of the word's
/// parity wins. The arm runs in no simdgroup-uniform position, so the barrier stands at the
/// loop's header, the one block every path from one iteration's write to the next crosses.
const HEADER_LOOP: &str = r#"
define void @header_loop(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 0, ptr addrspace(3) %4, align 4
  tail call void @air.wg.barrier(i32 2, i32 1)
  %end = add i32 %sg, 4
  %odd = and i32 %1, 1
  %even = icmp eq i32 %odd, 0
  br label %loop

loop:
  %k = phi i32 [ %sg, %2 ], [ %knext, %latch ]
  br i1 %even, label %arm, label %latch

arm:
  %5 = add i32 %1, %k
  %6 = and i32 %5, 31
  %7 = zext i32 %6 to i64
  %8 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %7
  %9 = mul i32 %k, 100
  %10 = add i32 %9, %1
  store i32 %10, ptr addrspace(3) %8, align 4
  br label %latch

latch:
  %knext = add i32 %k, 1
  %11 = icmp ult i32 %knext, %end
  br i1 %11, label %loop, label %done

done:
  tail call void @air.wg.barrier(i32 2, i32 1)
  %12 = load i32, ptr addrspace(3) %4, align 4
  %13 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %12, ptr addrspace(1) %13, align 4
  ret void
}
"#;

/// `neighbor` through device memory: thread `t` stores `t` to its own word of `out`, and after a
/// barrier that orders device memory reads the word of thread `t ^ 8` (in its simdgroup, in another
/// 8-lane chunk) and overwrites its own word with it in the same barrier interval. Lockstep reads
/// every word before any is overwritten: `out[t] = t ^ 8`.
const DEVICE_NEIGHBOR: &str = r#"
define void @device_neighbor(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %1, ptr addrspace(1) %4, align 4
  tail call void @air.wg.barrier(i32 1, i32 1)
  %5 = xor i32 %1, 8
  %6 = zext i32 %5 to i64
  %7 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %6
  %8 = load i32, ptr addrspace(1) %7, align 4
  store i32 %8, ptr addrspace(1) %4, align 4
  ret void
}
"#;

/// `rotate` through device memory: the threads of simdgroup 0 alone read the word of thread
/// `(t + 1) & 31` of `out` and overwrite their own with twice it, in a block every thread of
/// simdgroup 0 runs and no other thread does. Lockstep: `out[t] = 2 ((t + 1) & 31)` below 32, `t`
/// above.
const DEVICE_ROTATE: &str = r#"
define void @device_rotate(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %1, ptr addrspace(1) %4, align 4
  tail call void @air.wg.barrier(i32 1, i32 1)
  %5 = icmp eq i32 %sg, 0
  br i1 %5, label %walk, label %done

walk:
  %6 = add i32 %1, 1
  %7 = and i32 %6, 31
  %8 = zext i32 %7 to i64
  %9 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %8
  %10 = load i32, ptr addrspace(1) %9, align 4
  %11 = shl i32 %10, 1
  store i32 %11, ptr addrspace(1) %4, align 4
  br label %done

done:
  ret void
}
"#;

/// Each thread stores to its own word of `out`, reads it back and overwrites it, with no barrier:
/// no two threads touch one byte, so the device pass inserts nothing. `out[t] = t + 1`.
const DEVICE_OWN: &str = r#"
define void @device_own(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %1, ptr addrspace(1) %4, align 4
  %5 = load i32, ptr addrspace(1) %4, align 4
  %6 = add i32 %5, 1
  store i32 %6, ptr addrspace(1) %4, align 4
  ret void
}
"#;

/// `arms` through device memory, the shape Apple's front end gives the M3 probe's `arm` kernel: the
/// odd lanes store to their own word of `out` in one arm of a branch, and the even lanes read their
/// odd neighbour's word in the other and keep it in another object (the probe's second buffer; here
/// `@tg`). Program order does not order the two arms, and lockstep does not either: the M3 ran this
/// branch's true arm first (every even lane read the word as it was before), so a barrier ordering
/// the store first would answer what Metal does not.
const DEVICE_ARMS: &str = r#"
define void @device_arms(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  %5 = and i32 %1, 1
  %6 = icmp eq i32 %5, 0
  br i1 %6, label %even, label %odd

even:
  %7 = xor i32 %1, 1
  %8 = zext i32 %7 to i64
  %9 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %8
  %10 = load i32, ptr addrspace(1) %9, align 4
  %11 = getelementptr inbounds [64 x i32], ptr addrspace(3) @tg, i64 0, i64 %3
  store i32 %10, ptr addrspace(3) %11, align 4
  br label %done

odd:
  %12 = mul i32 %1, 3
  store i32 %12, ptr addrspace(1) %4, align 4
  br label %done

done:
  ret void
}
"#;

/// `half` through device memory: after a device barrier the low 16 threads of each simdgroup, in a
/// block no other thread runs, read the word of thread `t ^ 1` and overwrite their own with it.
/// Lockstep reads every word before any is overwritten: `out[t] = t ^ 1` where `t & 31 < 16`, `t`
/// elsewhere.
const DEVICE_HALF: &str = r#"
define void @device_half(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = zext i32 %1 to i64
  %4 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %3
  store i32 %1, ptr addrspace(1) %4, align 4
  tail call void @air.wg.barrier(i32 1, i32 1)
  %5 = and i32 %1, 31
  %6 = icmp ult i32 %5, 16
  br i1 %6, label %low, label %done

low:
  %7 = xor i32 %1, 1
  %8 = zext i32 %7 to i64
  %9 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %8
  %10 = load i32, ptr addrspace(1) %9, align 4
  store i32 %10, ptr addrspace(1) %4, align 4
  br label %done

done:
  ret void
}
"#;

/// A kernel that writes texel `(t, 0)` and then texel `(t ^ 1, 0)` of one texture: a texture is a
/// handle in the device address space, not a buffer word, so the device pass orders nothing
/// there.
const DEVICE_TEXTURE: &str = r#"
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)

define void @device_texture(ptr addrspace(1) %tex, i32 %t) #0 {
  %c = insertelement <2 x i32> zeroinitializer, i32 %t, i64 0
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %tex, <2 x i32> %c, <4 x float> zeroinitializer, i32 0, i32 2)
  %u = xor i32 %t, 1
  %d = insertelement <2 x i32> zeroinitializer, i32 %u, i64 0
  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %tex, <2 x i32> %d, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, i32 0, i32 2)
  ret void
}

attributes #0 = { convergent mustprogress nounwind willreturn }

!air.kernel = !{!0}
!air.version = !{!5}
!air.language_version = !{!6}

!0 = !{ptr @device_texture, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.write", !"air.arg_type_name", !"texture2d<float, write>", !"air.arg_name", !"tex"}
!4 = !{i32 1, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}
!5 = !{i32 2, i32 8, i32 0}
!6 = !{!"Metal", i32 4, i32 0, i32 0}
"#;

/// Thread 0 alone walks a chain through device memory, `out[k + 1] = out[k] + 1` for `k` in
/// `0..4`, behind `if (t == 0)`: one thread of a simdgroup at most runs the accesses, so no two
/// threads of one meet there, and the device pass inserts nothing. `out[t] = t` up to 4.
const DEVICE_LONE: &str = r#"
define void @device_lone(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %first, label %done

first:
  br label %loop

loop:
  %k = phi i32 [ 0, %first ], [ %n, %loop ]
  %4 = zext i32 %k to i64
  %5 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %4
  %6 = load i32, ptr addrspace(1) %5, align 4
  %7 = add i32 %6, 1
  %n = add nuw i32 %k, 1
  %8 = zext i32 %n to i64
  %9 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %8
  store i32 %7, ptr addrspace(1) %9, align 4
  %10 = icmp ult i32 %n, 4
  br i1 %10, label %loop, label %done

done:
  ret void
}
"#;

/// [`DEVICE_LONE`] behind `if ((t >> 1) == 0)`: threads 0 and 1 of simdgroup 0 both walk the
/// chain, so a thread's load may meet the other's store, and lockstep orders the pair.
const DEVICE_PAIR: &str = r#"
define void @device_pair(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %h = lshr i32 %1, 1
  %3 = icmp eq i32 %h, 0
  br i1 %3, label %first, label %done

first:
  br label %loop

loop:
  %k = phi i32 [ 0, %first ], [ %n, %loop ]
  %4 = zext i32 %k to i64
  %5 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %4
  %6 = load i32, ptr addrspace(1) %5, align 4
  %7 = add i32 %6, 1
  %n = add nuw i32 %k, 1
  %8 = zext i32 %n to i64
  %9 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %8
  store i32 %7, ptr addrspace(1) %9, align 4
  %10 = icmp ult i32 %n, 4
  br i1 %10, label %loop, label %done

done:
  ret void
}
"#;

/// Thread `i` takes the turn of iteration `i` of a uniform loop, behind `if (t == i)`: one thread
/// a turn, but another thread each turn, so the store of one turn and the load of the next meet
/// across the loop's iterations, and lockstep orders them. `out[t] = t` up to 4.
const DEVICE_TURNS: &str = r#"
define void @device_turns(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  br label %head

head:
  %i = phi i32 [ 0, %2 ], [ %n, %latch ]
  %3 = icmp eq i32 %1, %i
  br i1 %3, label %turn, label %latch

turn:
  %4 = zext i32 %i to i64
  %5 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %4
  %6 = load i32, ptr addrspace(1) %5, align 4
  %7 = add i32 %6, 1
  %m = add nuw i32 %i, 1
  %8 = zext i32 %m to i64
  %9 = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %8
  store i32 %7, ptr addrspace(1) %9, align 4
  br label %latch

latch:
  %n = add nuw i32 %i, 1
  %10 = icmp ult i32 %n, 4
  br i1 %10, label %head, label %done

done:
  ret void
}
"#;

/// A store loop over a word two threads of a simdgroup share: threads `2k` and `2k + 1` store
/// `4k + i` to word `k` in each of four iterations of a loop every thread runs alike, at one
/// address through the loop (`t >> 1`, which the analysis cannot tell apart between threads). Each
/// thread's last store is the last to its word whatever order the threads' iterations take, as
/// under lockstep, so the store against itself in a later iteration takes no barrier.
/// `out[k] = 4k + 3` for `k < 16`.
const DEVICE_SHARED_SLOT: &str = r#"
define void @device_shared_slot(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %h = lshr i32 %1, 1
  %x = zext i32 %h to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %x
  %b = shl i32 %h, 2
  br label %loop

loop:
  %i = phi i32 [ 0, %2 ], [ %n, %loop ]
  %v = add i32 %b, %i
  store i32 %v, ptr addrspace(1) %p, align 4
  %n = add nuw i32 %i, 1
  %c = icmp ult i32 %n, 4
  br i1 %c, label %loop, label %done

done:
  ret void
}
"#;

/// [`DEVICE_SHARED_SLOT`] whose loop also reads the word of the other pair (`(t >> 1) ^ 1`), which
/// another thread of the simdgroup stores in the same iteration: lockstep orders that read after
/// the iteration's stores and before the next, so the barrier stays.
const DEVICE_SHARED_SLOT_READ: &str = r#"
define void @device_shared_slot_read(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %h = lshr i32 %1, 1
  %x = zext i32 %h to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %x
  %o = xor i32 %h, 1
  %y = zext i32 %o to i64
  %q = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %y
  br label %loop

loop:
  %i = phi i32 [ 0, %2 ], [ %n, %loop ]
  %a = phi i32 [ 0, %2 ], [ %s, %loop ]
  %v = add i32 %a, %i
  store i32 %v, ptr addrspace(1) %p, align 4
  %r = load i32, ptr addrspace(1) %q, align 4
  %s = add i32 %a, %r
  %n = add nuw i32 %i, 1
  %c = icmp ult i32 %n, 4
  br i1 %c, label %loop, label %done

done:
  ret void
}
"#;

/// [`DEVICE_SHARED_SLOT`] storing to word `(t >> 1) + i`: the address moves with the iteration,
/// so a thread's store may land on the word another thread of its simdgroup stores in an earlier
/// iteration, and lockstep makes the later iteration's store the last; the barrier stays.
const DEVICE_MOVING_SLOT: &str = r#"
define void @device_moving_slot(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %h = lshr i32 %1, 1
  br label %loop

loop:
  %i = phi i32 [ 0, %2 ], [ %n, %loop ]
  %w = add i32 %h, %i
  %x = zext i32 %w to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %x
  store i32 %1, ptr addrspace(1) %p, align 4
  %n = add nuw i32 %i, 1
  %c = icmp ult i32 %n, 4
  br i1 %c, label %loop, label %done

done:
  ret void
}
"#;

/// [`DEVICE_SHARED_SLOT`] with a loop each thread leaves after `1 + (t & 1)` iterations: the two
/// threads of a word store it a different number of times, and lockstep makes the thread that
/// iterates longer store last; the barrier stays.
const DEVICE_SHARED_SLOT_DIVERGENT: &str = r#"
define void @device_shared_slot_divergent(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %h = lshr i32 %1, 1
  %x = zext i32 %h to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %x
  %odd = and i32 %1, 1
  %end = add nuw i32 %odd, 1
  br label %loop

loop:
  %i = phi i32 [ 0, %2 ], [ %n, %loop ]
  store i32 %1, ptr addrspace(1) %p, align 4
  %n = add nuw i32 %i, 1
  %c = icmp ult i32 %n, %end
  br i1 %c, label %loop, label %done

done:
  ret void
}
"#;

/// Each thread writes a record of two words at `2 gid` and `2 gid + 1`, `gid` its
/// `thread_position_in_grid`: two threads of a simdgroup hold equal positions (in a threadgroup
/// narrower than 32) or positions 1 to 1023 apart, so the two stores never meet between two
/// threads, and nothing is inserted.
const DEVICE_RECORD: &str = r#"
define void @device_record(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %m = mul i32 %1, 2
  %x = zext i32 %m to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %x
  store i32 %1, ptr addrspace(1) %p, align 4
  %q = getelementptr inbounds i32, ptr addrspace(1) %p, i64 1
  %v = add i32 %1, 1
  store i32 %v, ptr addrspace(1) %q, align 4
  ret void
}
"#;

/// [`DEVICE_RECORD`] storing both values to word `gid`: two threads of a simdgroup that share a
/// position store one word in two instructions, and lockstep makes the second last, so the
/// barrier stays.
const DEVICE_RECORD_SHARED: &str = r#"
define void @device_record_shared(ptr addrspace(1) %0, i32 %1, i32 %sg) #0 {
  %x = zext i32 %1 to i64
  %p = getelementptr inbounds i32, ptr addrspace(1) %0, i64 %x
  store i32 %1, ptr addrspace(1) %p, align 4
  %v = add i32 %1, 1
  store i32 %v, ptr addrspace(1) %p, align 4
  ret void
}
"#;

/// The words of `out` after one workgroup of `threads` threads ran `m`, in both scheduling
/// orders (which must agree) at subgroup width `lanes`.
fn run_words(plan: &Plan, m: &TypedModule, threads: u32, lanes: SubgroupSize) -> Vec<u32> {
    let mut spec = BTreeMap::new();
    for (a, v) in [Axis::X, Axis::Y, Axis::Z].into_iter().zip([threads, 1, 1]) {
        spec.insert(SpecKey::LocalSize(a), u64::from(v));
    }
    // The flattened tiers' Vulkan workgroup width.
    spec.insert(SpecKey::FlatLocalSize, u64::from(threads));
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0; threads as usize * 4],
        }],
        push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [1, 1, 1],
            subgroup: lanes,
            order,
            ..Config::default()
        };
        let out = run(m, &cfg, res.clone()).expect("run").buffers;
        outs.push(
            out[0]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<u32>>(),
        );
    }
    assert_eq!(outs[0], outs[1], "order independent");
    outs.swap_remove(0)
}

/// The census count of strategy `name` in `plan`.
fn uses(plan: &Plan, name: &str) -> u32 {
    plan.census().get(name).map_or(0, |&(n, _)| n)
}

/// Kernel `name` planned on `profile`, run at each width in `lanes` with `threads` threads, read
/// as lockstep reads (`expect`). Returns the plan.
fn check(
    name: &str,
    body: &str,
    profile_name: &str,
    threads: u32,
    lanes: &[SubgroupSize],
    expect: impl Fn(u32) -> u32,
) -> Option<Plan> {
    let bc = assemble(name, body)?;
    let p = profile(profile_name, DevicePolicy::default()).expect("profile");
    let (plan, m) = typed_module(&bc, name, &p).unwrap_or_else(|e| panic!("{name}: {e}"));
    let want: Vec<u32> = (0..threads).map(&expect).collect();
    for &w in lanes {
        assert_eq!(
            run_words(&plan, &m, threads, w),
            want,
            "{name} on {profile_name} at {w:?}"
        );
    }
    Some(plan)
}

const CHUNKED: [SubgroupSize; 3] = [SubgroupSize::W8, SubgroupSize::W16, SubgroupSize::W32];

#[test]
fn a_uniform_conflict_takes_a_threadgroup_barrier() {
    for profile_name in ["lavapipe", "moltenvk-m3", "vk12-floor"] {
        let Some(plan) = check("neighbor", NEIGHBOR, profile_name, 32, &CHUNKED, |t| t ^ 8) else {
            return;
        };
        assert_eq!(
            uses(&plan, "barrier.lockstep.threadgroup"),
            1,
            "{profile_name}"
        );
        assert_eq!(
            uses(&plan, "barrier.lockstep.simdgroup"),
            0,
            "{profile_name}"
        );
        assert_eq!(uses(&plan, "barrier.threadgroup"), 2, "{profile_name}");
        // A threadgroup barrier raises no simd class.
        assert_eq!(plan.simd.1, Tier::NoSubgroup, "{profile_name}");
    }
}

#[test]
fn a_conflict_in_one_simdgroup_takes_a_simdgroup_barrier() {
    let rotate = |t: u32| if t < 32 { 2 * ((t + 1) & 31) } else { t };
    let carry = |t: u32| if t < 32 { ((t + 2) & 31) + 2 } else { t };
    for (name, body, expect, barriers) in [
        ("rotate", ROTATE, &rotate as &dyn Fn(u32) -> u32, 1),
        ("contract", CONTRACT, &rotate, 1),
        // One barrier between the read and the write, one at the end of the iteration.
        ("carry", CARRY, &carry, 2),
    ] {
        // Emulated simdgroups: a predicated workgroup barrier, exact at every width.
        for profile_name in ["lavapipe", "vk12-floor"] {
            let Some(plan) = check(name, body, profile_name, 64, &CHUNKED, expect) else {
                return;
            };
            let at = format!("{name} on {profile_name}");
            assert_eq!(uses(&plan, "barrier.lockstep.simdgroup"), barriers, "{at}");
            assert_eq!(uses(&plan, "barrier.lockstep.threadgroup"), 0, "{at}");
            assert_eq!(plan.simd.1, Tier::Emulated, "{at}");
            assert!(plan.families.simd.predication.is_some(), "{at}");
        }
        // A 32-lane subgroup: a subgroup barrier where the whole simdgroup runs.
        let Some(plan) = check(name, body, "moltenvk-m3", 64, &[SubgroupSize::W32], expect) else {
            return;
        };
        assert_eq!(
            uses(&plan, "barrier.lockstep.simdgroup"),
            barriers,
            "{name}"
        );
        assert_eq!(plan.simd.1, Tier::Direct, "{name}");
    }
}

/// The arm's read, then the loop's writes: no subgroup barrier may stand between them. Emulated
/// simdgroups order the pair on lavapipe and vk12-floor; moltenvk-m3's device runs a simdgroup's
/// 32 threads in lockstep itself (`VkProperty::SimdgroupLockstep`), so there the kernel keeps its
/// own tier, no barrier stands, and the module requires the property. The CPU interpreter's 32-lane
/// subgroups run in lockstep as that device does.
#[test]
fn a_conflict_no_subgroup_barrier_orders_takes_emulated_simdgroups_or_the_devices_lockstep() {
    let expect = |t: u32| {
        if t < 32 && t.is_multiple_of(2) {
            (t + 2) & 31
        } else {
            0
        }
    };
    let requires = |profile_name: &str| {
        let bc = assemble("arm_then_loop", ARM_THEN_LOOP).expect("assembled once already");
        let p = profile(profile_name, DevicePolicy::default()).expect("profile");
        compile(&bc, "arm_then_loop", &p)
            .unwrap_or_else(|e| panic!("{profile_name}: {e}"))
            .requirements
            .properties
            .contains(&VkProperty::SimdgroupLockstep)
    };
    for profile_name in ["lavapipe", "vk12-floor"] {
        let Some(plan) = check(
            "arm_then_loop",
            ARM_THEN_LOOP,
            profile_name,
            64,
            &CHUNKED,
            expect,
        ) else {
            return;
        };
        // Named apart in the census: the kernel's lockstep order forces the tier.
        assert_eq!(
            uses(&plan, "barrier.lockstep.emulated.simdgroup"),
            1,
            "{profile_name}"
        );
        assert_eq!(
            uses(&plan, "barrier.lockstep.simdgroup"),
            0,
            "{profile_name}"
        );
        assert_eq!(uses(&plan, "barrier.lockstep.device"), 0, "{profile_name}");
        assert_eq!(plan.simd.1, Tier::Emulated, "{profile_name}");
        assert!(!requires(profile_name), "{profile_name}");
    }
    let Some(plan) = check(
        "arm_then_loop",
        ARM_THEN_LOOP,
        "moltenvk-m3",
        64,
        &[SubgroupSize::W32],
        expect,
    ) else {
        return;
    };
    assert_eq!(uses(&plan, "barrier.lockstep.device"), 1);
    assert!(!plan
        .census()
        .keys()
        .any(|k| k.starts_with("barrier.lockstep.") && k != "barrier.lockstep.device"));
    // The kernel's own tier: it holds no simdgroup operation.
    assert_eq!(plan.simd.1, Tier::NoSubgroup);
    assert!(requires("moltenvk-m3"));
}

/// The `Emulated` tier a lockstep barrier forces (item 4's last kind) walks the kernel
/// predicated, and that walk carries what an instruction reads as the plain walk and `select` do:
/// the pointer a forwarding `extractvalue` forwards, from another block into a block the walk
/// splits at a synchronization (`forward_cross`) and across the synchronization within its block
/// (`forward_sync`), and the members a member-wise phi's copy reads across it (`member_sync`).
/// Each failed to emit (`Invariant: ValueId(n) has no value in this block`) while
/// the walk reloaded and demoted an instruction's literal operands. On moltenvk-m3 the device's
/// lockstep orders the pair, so the kernels keep their tier and no walk is predicated.
#[test]
fn a_forced_predicated_walk_carries_what_each_instruction_reads() {
    let arm = |t: u32| {
        if t < 32 && t.is_multiple_of(2) {
            (t + 2) & 31
        } else {
            0
        }
    };
    let forward = |t: u32| if t < 32 { arm(t) } else { 6 };
    let member = |t: u32| if t < 32 { arm(t) } else { t };
    for (name, body, expect, extracts) in [
        (
            "forward_cross",
            FORWARD_CROSS,
            &forward as &dyn Fn(u32) -> u32,
            "arith.aggregate.forward",
        ),
        (
            "forward_sync",
            FORWARD_SYNC,
            &forward,
            "arith.aggregate.forward",
        ),
        (
            "member_sync",
            MEMBER_SYNC,
            &member,
            "arith.aggregate.member",
        ),
    ] {
        for profile_name in ["lavapipe", "vk12-floor"] {
            let Some(plan) = check(name, body, profile_name, 64, &CHUNKED, expect) else {
                return;
            };
            let at = format!("{name} on {profile_name}");
            assert_eq!(
                uses(&plan, "barrier.lockstep.emulated.simdgroup"),
                1,
                "{at}"
            );
            assert_eq!(plan.simd.1, Tier::Emulated, "{at}");
            assert!(plan.families.simd.predication.is_some(), "{at}");
            // The struct's pointer is read through the extract, not folded away.
            assert_eq!(uses(&plan, extracts), 1, "{at}");
        }
        let Some(plan) = check(name, body, "moltenvk-m3", 64, &[SubgroupSize::W32], expect) else {
            return;
        };
        assert_eq!(uses(&plan, "barrier.lockstep.device"), 1, "{name}");
        assert!(plan.families.simd.predication.is_none(), "{name}");
    }
}

#[test]
fn disjoint_accesses_take_no_barrier() {
    let disjoint = |t: u32| t + 1 + 7 + (t ^ 8);
    let offsets = |t: u32| t + 64 + 7;
    for (name, body, expect) in [
        ("disjoint", DISJOINT, &disjoint as &dyn Fn(u32) -> u32),
        ("offsets", OFFSETS, &offsets),
    ] {
        for profile_name in ["lavapipe", "moltenvk-m3"] {
            let Some(plan) = check(name, body, profile_name, 64, &CHUNKED, expect) else {
                return;
            };
            let census = plan.census();
            assert!(
                !census.keys().any(|k| k.starts_with("barrier.lockstep")),
                "{name} on {profile_name}: {census:?}"
            );
        }
    }
}

#[test]
fn bounded_rowed_strided_and_guarded_accesses_take_no_barrier() {
    let fields = |_: u32| 1u32 + 2 + 3;
    let rows = |t: u32| t + 3;
    let guard = |_: u32| (0..32u32).sum();
    let strided = |t: u32| t + 96;
    for (name, body, threads, expect) in [
        ("fields", FIELDS, 64, &fields as &dyn Fn(u32) -> u32),
        ("rows", ROWS, 32, &rows),
        ("strided", STRIDED, 32, &strided),
        ("guard", GUARD, 32, &guard),
        ("signed_guard", SIGNED_GUARD, 32, &guard),
    ] {
        for profile_name in ["lavapipe", "moltenvk-m3"] {
            let Some(plan) = check(name, body, profile_name, threads, &CHUNKED, expect) else {
                return;
            };
            let census = plan.census();
            assert!(
                !census.keys().any(|k| k.starts_with("barrier.lockstep")),
                "{name} on {profile_name}: {census:?}"
            );
        }
    }
}

/// Pairs whose indices look apart but whose addresses meet: a tree reduction's guard from a base
/// that differs between threads, a row from such a base, two rows of one size and different
/// strides, and a simdgroup sum read after a loop its lanes leave at different iterations. Each
/// keeps a barrier, so every schedule reads what lockstep reads; on moltenvk-m3 the device's own
/// lockstep, at 32 lanes, orders what no subgroup barrier may.
#[test]
fn indices_apart_from_a_shared_place_only_clear_their_pairs() {
    let group_base = |l: u32| {
        if l < 8 {
            l + 16
        } else if (16..24).contains(&l) {
            l - 8
        } else {
            0
        }
    };
    let row_base = |l: u32| l + 1;
    let row_stride = |_: u32| 0;
    let temporal = |l: u32| 33 - l;
    // The one lockstep barrier each kernel takes on lavapipe and vk12-floor, then on moltenvk-m3.
    // At 32 threads the interpreter's 32-lane schedule is lockstep and gives the answer either
    // way, so on moltenvk-m3 the barrier is what shows the pair is kept: `group_base`'s place
    // is ordered only within its simdgroup, which that device does itself.
    let tg = "barrier.lockstep.threadgroup";
    for (name, body, expect, elsewhere, on_m3) in [
        (
            "group_base",
            GROUP_BASE,
            &group_base as &dyn Fn(u32) -> u32,
            "barrier.lockstep.emulated.simdgroup",
            "barrier.lockstep.device",
        ),
        ("row_base", ROW_BASE, &row_base, tg, tg),
        ("row_stride", ROW_STRIDE, &row_stride, tg, tg),
        ("temporal", TEMPORAL, &temporal, tg, tg),
    ] {
        for (profile_name, lanes, want) in [
            ("moltenvk-m3", &[SubgroupSize::W32][..], on_m3),
            ("lavapipe", &CHUNKED[..], elsewhere),
            ("vk12-floor", &CHUNKED[..], elsewhere),
        ] {
            let Some(plan) = check(name, body, profile_name, 32, lanes, expect) else {
                return;
            };
            let lockstep: Vec<String> = plan
                .census()
                .into_iter()
                .filter(|(k, _)| k.starts_with("barrier.lockstep."))
                .map(|(k, (n, _))| format!("{k} {n}"))
                .collect();
            assert_eq!(lockstep, [format!("{want} 1")], "{name} on {profile_name}");
        }
    }
}

#[test]
fn a_conflict_across_iterations_takes_a_barrier_at_the_loop_header() {
    // The last iteration `k` of word `w`'s parity wrote it, from thread `(w - k) & 31`.
    let expect = |w: u32| {
        let k = if w.is_multiple_of(2) { 2 } else { 3 };
        100 * k + (w.wrapping_sub(k) & 31)
    };
    for (profile_name, lanes, tier) in [
        ("lavapipe", &CHUNKED[..], Tier::Emulated),
        ("moltenvk-m3", &[SubgroupSize::W32][..], Tier::Direct),
    ] {
        let Some(plan) = check("header_loop", HEADER_LOOP, profile_name, 32, lanes, expect) else {
            return;
        };
        assert_eq!(
            uses(&plan, "barrier.lockstep.simdgroup"),
            1,
            "{profile_name}"
        );
        assert_eq!(
            uses(&plan, "barrier.lockstep.emulated.simdgroup"),
            0,
            "{profile_name}"
        );
        assert_eq!(plan.simd.1, tier, "{profile_name}");
    }
}

/// The words of `out` (buffer 1) after one workgroup of `input.len()` threads ran `m` over
/// `input` (buffer 0), in both scheduling orders (which must agree) at subgroup width `lanes`.
fn run_over(plan: &Plan, m: &TypedModule, input: &[u32], lanes: SubgroupSize) -> Vec<u32> {
    let threads = input.len() as u32;
    let mut spec = BTreeMap::new();
    for (a, v) in [Axis::X, Axis::Y, Axis::Z].into_iter().zip([threads, 1, 1]) {
        spec.insert(SpecKey::LocalSize(a), u64::from(v));
    }
    spec.insert(SpecKey::FlatLocalSize, u64::from(threads));
    let slot = |index| {
        Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        })
    };
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: slot(0),
                address: None,
                bytes: input.iter().flat_map(|w| w.to_le_bytes()).collect(),
            },
            Buffer {
                slot: slot(1),
                address: None,
                bytes: vec![0xee; 8 * input.len()],
            },
        ],
        push: vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec,
    };
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [1, 1, 1],
            subgroup: lanes,
            order,
            ..Config::default()
        };
        let out = run(m, &cfg, res.clone()).expect("run").buffers;
        outs.push(
            out[1]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<u32>>(),
        );
    }
    assert_eq!(outs[0], outs[1], "order independent");
    outs.swap_remove(0)
}

/// Each simdgroup's threads run each iteration together: those still in the loop all read the
/// words as the iteration found them, then all write. `step` gives a thread's new word, or `None`
/// where it leaves the loop.
fn in_lockstep(words: &mut [u32], mut step: impl FnMut(usize, u32, &[u32]) -> Option<u32>) {
    let mut active = vec![true; words.len()];
    let mut iteration = 0;
    while active.iter().any(|&a| a) {
        let before = words.to_vec();
        for t in 0..words.len() {
            if active[t] {
                match step(t, iteration, &before) {
                    Some(word) => words[t] = word,
                    None => active[t] = false,
                }
            }
        }
        iteration += 1;
    }
}

/// The owned fixture `rtl_lockstep`, the runtime lane `driver_deviations`' lockstep gate: Apple's
/// compiler keeps each kernel's read of another thread's word and its write of its own in one loop
/// each thread leaves after its own count, so no barrier position is uniform in the simdgroup.
/// moltenvk-m3 relies on its device's lockstep there (a `barrier.lockstep.device` place,
/// `VkProperty::SimdgroupLockstep` required, its tier kept); lavapipe and vk12-floor take emulated
/// simdgroups. Each reads what lockstep reads, which the lane checks on the device
/// (`rtl_lockstep_divergent_loop`'s 64 threads run 1 to 5 iterations of `words[t] = words[t ^ 16] *
/// 3 + i + 1`; `rtl_lockstep_until`'s add the word 8 lanes on to their own while it is below a
/// limit, at most 8 times).
#[test]
fn the_lockstep_gate_fixture_needs_the_devices_lockstep_or_emulated_simdgroups() {
    let path = crate::pipeline::testing::engine_dir().join("rtl_lockstep.ll");
    let bc = match llvm_assemble(&path) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected rtl_lockstep: {why}"),
    };
    let counts: Vec<u32> = (0..64).map(|t| 1 + (t * 7 + 3) % 5).collect();
    let limits: Vec<u32> = (0..64).map(|t| 16 + (t * 37) % 97).collect();
    let divergent = {
        let mut words: Vec<u32> = (0..64).collect();
        in_lockstep(&mut words, |t, i, before| {
            (i < counts[t]).then(|| before[t ^ 16].wrapping_mul(3).wrapping_add(i + 1))
        });
        words.resize(128, 0xeeee_eeee);
        words
    };
    let until = {
        let mut words: Vec<u32> = (1..=64).collect();
        let mut steps = vec![0u32; 64];
        in_lockstep(&mut words, |t, _, before| {
            if before[t] >= limits[t] || steps[t] >= 8 {
                return None;
            }
            steps[t] += 1;
            Some(before[t].wrapping_add(before[((t + 8) & 31) | (t & 32)]))
        });
        words.extend(steps);
        words
    };
    for (kernel, input, want) in [
        ("rtl_lockstep_divergent_loop", &counts, divergent),
        ("rtl_lockstep_until", &limits, until),
    ] {
        for (profile_name, lanes) in [
            ("moltenvk-m3", &[SubgroupSize::W32][..]),
            ("lavapipe", &CHUNKED[..]),
            ("vk12-floor", &CHUNKED[..]),
        ] {
            let device = profile_name == "moltenvk-m3";
            let p = profile(profile_name, DevicePolicy::default()).expect("profile");
            let (plan, m) = typed_module(&bc, kernel, &p)
                .unwrap_or_else(|e| panic!("{kernel} on {profile_name}: {e}"));
            let at = format!("{kernel} on {profile_name}");
            assert_eq!(uses(&plan, "barrier.lockstep.device") > 0, device, "{at}");
            assert_eq!(
                uses(&plan, "barrier.lockstep.emulated.simdgroup") > 0,
                !device,
                "{at}"
            );
            if !device {
                assert_eq!(plan.simd.1, Tier::Emulated, "{at}");
            }
            let requires = compile(&bc, kernel, &p)
                .unwrap_or_else(|e| panic!("{at}: {e}"))
                .requirements
                .properties
                .contains(&VkProperty::SimdgroupLockstep);
            assert_eq!(requires, device, "{at}");
            for &w in lanes {
                assert_eq!(run_over(&plan, &m, input, w), want, "{at} at {w:?}");
            }
        }
    }
}

/// A device-memory pair only lockstep orders takes the barrier lockstep stood for, with
/// `mem_device`: in workgroup-uniform control flow a threadgroup barrier on every profile; in one
/// simdgroup's block a simdgroup barrier, emulated where the device has no 32-lane subgroup.
/// Without it the chunked schedule reads a word another chunk already overwrote.
#[test]
fn a_device_memory_conflict_takes_the_barrier_lockstep_stood_for() {
    for profile_name in ["lavapipe", "moltenvk-m3", "vk12-floor"] {
        let Some(plan) = check(
            "device_neighbor",
            DEVICE_NEIGHBOR,
            profile_name,
            32,
            &CHUNKED,
            |t| t ^ 8,
        ) else {
            return;
        };
        assert_eq!(
            uses(&plan, "barrier.lockstep.threadgroup"),
            1,
            "{profile_name}"
        );
        assert_eq!(plan.simd.1, Tier::NoSubgroup, "{profile_name}");
    }
    let rotate = |t: u32| if t < 32 { 2 * ((t + 1) & 31) } else { t };
    for profile_name in ["lavapipe", "vk12-floor"] {
        let Some(plan) = check(
            "device_rotate",
            DEVICE_ROTATE,
            profile_name,
            64,
            &CHUNKED,
            rotate,
        ) else {
            return;
        };
        assert_eq!(
            uses(&plan, "barrier.lockstep.simdgroup"),
            1,
            "{profile_name}"
        );
        assert_eq!(plan.simd.1, Tier::Emulated, "{profile_name}");
    }
    let Some(plan) = check(
        "device_rotate",
        DEVICE_ROTATE,
        "moltenvk-m3",
        64,
        &[SubgroupSize::W32],
        rotate,
    ) else {
        return;
    };
    assert_eq!(uses(&plan, "barrier.lockstep.simdgroup"), 1);
    assert_eq!(plan.simd.1, Tier::Direct);
}

/// Each thread's own device word takes no barrier.
#[test]
fn a_threads_own_device_word_takes_no_barrier() {
    for profile_name in ["lavapipe", "moltenvk-m3", "vk12-floor"] {
        let Some(plan) = check("device_own", DEVICE_OWN, profile_name, 32, &CHUNKED, |t| {
            t + 1
        }) else {
            return;
        };
        for name in [
            "barrier.lockstep.threadgroup",
            "barrier.lockstep.simdgroup",
            "barrier.lockstep.emulated.simdgroup",
            "barrier.lockstep.device",
        ] {
            assert_eq!(uses(&plan, name), 0, "{name} on {profile_name}");
        }
    }
}

/// The two arms of a branch take no device barrier between them: neither reaches the other, and
/// Metal's own answer is the arm order its GPU chose, not lockstep's program order.
#[test]
fn a_branchs_two_arms_take_no_device_barrier() {
    for profile_name in ["lavapipe", "moltenvk-m3", "vk12-floor"] {
        let Some(bc) = assemble("device_arms", DEVICE_ARMS) else {
            return;
        };
        let p = profile(profile_name, DevicePolicy::default()).expect("profile");
        let (plan, _) = typed_module(&bc, "device_arms", &p)
            .unwrap_or_else(|e| panic!("device_arms on {profile_name}: {e}"));
        for name in [
            "barrier.lockstep.threadgroup",
            "barrier.lockstep.simdgroup",
            "barrier.lockstep.emulated.simdgroup",
            "barrier.lockstep.device",
        ] {
            assert_eq!(uses(&plan, name), 0, "{name} on {profile_name}");
        }
    }
}

/// The lockstep census names of the inserted barriers.
const LOCKSTEP_NAMES: [&str; 4] = [
    "barrier.lockstep.threadgroup",
    "barrier.lockstep.simdgroup",
    "barrier.lockstep.emulated.simdgroup",
    "barrier.lockstep.device",
];

/// A pair in a block only some threads of each simdgroup run takes a subgroup barrier, and the
/// kernel keeps the `Direct` tier, where the entry reconverges maximally on 32-lane subgroups
/// (the RTX 5080's profile): the barrier orders the threads that run it together, those
/// Apple's simdgroup runs it for. On lavapipe's 8-lane subgroups it stays on emulated simdgroups,
/// and on MoltenVK the device's lockstep orders it.
#[test]
fn a_divergent_device_pair_takes_a_tangle_barrier_where_the_entry_reconverges_maximally() {
    let half = |t: u32| if t & 31 < 16 { t ^ 1 } else { t };
    let Some(plan) = check(
        "device_half",
        DEVICE_HALF,
        "nvidia-rtx5080-laptop",
        64,
        &[SubgroupSize::W32],
        half,
    ) else {
        return;
    };
    assert_eq!(uses(&plan, "barrier.lockstep.simdgroup"), 1);
    assert_eq!(uses(&plan, "barrier.lockstep.emulated.simdgroup"), 0);
    assert_eq!(plan.simd.1, Tier::Direct);
    let Some(plan) = check("device_half", DEVICE_HALF, "lavapipe", 64, &CHUNKED, half) else {
        return;
    };
    assert_eq!(uses(&plan, "barrier.lockstep.emulated.simdgroup"), 1);
    assert_eq!(plan.simd.1, Tier::Emulated);
    let Some(plan) = check(
        "device_half",
        DEVICE_HALF,
        "moltenvk-m3",
        64,
        &[SubgroupSize::W32],
        half,
    ) else {
        return;
    };
    assert_eq!(uses(&plan, "barrier.lockstep.device"), 1);
}

/// Texture calls take no device barrier: their pointer is the texture's handle.
#[test]
fn texture_calls_take_no_device_barrier() {
    let text = format!("{HEADER}\n{DEVICE_TEXTURE}");
    let Some(bc) = assemble_text("device_texture", &text) else {
        return;
    };
    for profile_name in ["lavapipe", "moltenvk-m3", "nvidia-rtx5080-laptop"] {
        let p = profile(profile_name, DevicePolicy::default()).expect("profile");
        let (plan, _) = typed_module(&bc, "device_texture", &p)
            .unwrap_or_else(|e| panic!("device_texture on {profile_name}: {e}"));
        for name in LOCKSTEP_NAMES {
            assert_eq!(uses(&plan, name), 0, "{name} on {profile_name}");
        }
    }
}

/// The profiles the one-thread guard is checked on: emulated simdgroups (lavapipe's 8-lane
/// subgroups, the floor), the tangle barrier (the RTX 5080's maximal reconvergence), and the
/// device's lockstep (MoltenVK).
const LONE_PROFILES: [&str; 4] = [
    "lavapipe",
    "vk12-floor",
    "nvidia-rtx5080-laptop",
    "moltenvk-m3",
];

/// The lanes `profile_name`'s kernels run on here: 32 where its subgroups are pinned 32 wide
/// or it runs a simdgroup in lockstep itself, else the chunked widths.
fn lanes_of(profile_name: &str) -> &'static [SubgroupSize] {
    match profile_name {
        "nvidia-rtx5080-laptop" | "moltenvk-m3" => &[SubgroupSize::W32],
        _ => &CHUNKED,
    }
}

/// Accesses one thread of a simdgroup at most runs take no barrier: behind `if (t == 0)` thread 0
/// alone walks its chain, so the kernel keeps no simd class and the `NoSubgroup` tier, where a
/// simdgroup barrier would have raised its class (on the RTX 5080, a pinned 32-lane subgroup whose
/// dispatch of fewer threads is a partial simdgroup).
#[test]
fn accesses_one_thread_of_a_simdgroup_runs_take_no_barrier() {
    let chain = |t: u32| if t <= 4 { t } else { 0 };
    for profile_name in LONE_PROFILES {
        let Some(plan) = check(
            "device_lone",
            DEVICE_LONE,
            profile_name,
            32,
            lanes_of(profile_name),
            chain,
        ) else {
            return;
        };
        for name in LOCKSTEP_NAMES {
            assert_eq!(uses(&plan, name), 0, "{name} on {profile_name}");
        }
        assert_eq!(plan.simd.1, Tier::NoSubgroup, "{profile_name}");
    }
}

/// A guard two threads of a simdgroup pass, and a turn another thread takes each iteration,
/// keep the barrier lockstep stood for (the limits): the one-thread fact needs a value that differs
/// between every two threads, and one value of each through the pair's iterations.
#[test]
fn a_guard_two_threads_pass_or_a_turn_per_iteration_keeps_its_barrier() {
    let chain = |t: u32| if t <= 4 { t } else { 0 };
    for (name, body) in [("device_pair", DEVICE_PAIR), ("device_turns", DEVICE_TURNS)] {
        for profile_name in LONE_PROFILES {
            let Some(plan) = check(name, body, profile_name, 32, lanes_of(profile_name), chain)
            else {
                return;
            };
            let barriers: u32 = LOCKSTEP_NAMES.iter().map(|n| uses(&plan, n)).sum();
            assert!(
                barriers >= 1,
                "{name} on {profile_name}: no lockstep barrier"
            );
        }
    }
}

/// The owned fixture `rfs_agx_yield`'s `agx_yield_lookback`, the runtime lanes' busy kernel:
/// its ticket, spin and publish run behind `if (lid == 0)` and use no simdgroup operation, so
/// it takes no lockstep barrier and the `NoSubgroup` tier on every profile here, and a dispatch
/// of one thread runs no emulated simdgroups (the first-use lane on the RTX 3080 saw it announce
/// `EmulatedSimdgroups` when a tangle barrier pinned it 32 lanes wide).
#[test]
fn the_busy_kernel_takes_no_lockstep_barrier() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine/rfs_agx_yield.ll");
    let bc = match llvm_assemble(&path) {
        Ok(b) => b,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected rfs_agx_yield: {why}"),
    };
    for profile_name in LONE_PROFILES {
        let p = profile(profile_name, DevicePolicy::default()).expect("profile");
        let (plan, _) = typed_module(&bc, "agx_yield_lookback", &p)
            .unwrap_or_else(|e| panic!("agx_yield_lookback on {profile_name}: {e}"));
        for name in LOCKSTEP_NAMES {
            assert_eq!(uses(&plan, name), 0, "{name} on {profile_name}");
        }
        assert_eq!(plan.simd.1, Tier::NoSubgroup, "{profile_name}");
    }
}

/// A store loop at an address each thread keeps through the loop, run alike by every thread of the
/// simdgroup, takes no barrier between its iterations, even where two threads share the word:
/// every schedule leaves a thread's last store there, as lockstep does.
#[test]
fn a_store_loop_at_one_address_takes_no_barrier() {
    let last = |t: u32| if t < 16 { 4 * t + 3 } else { 0 };
    for profile_name in LONE_PROFILES {
        let Some(plan) = check(
            "device_shared_slot",
            DEVICE_SHARED_SLOT,
            profile_name,
            32,
            lanes_of(profile_name),
            last,
        ) else {
            return;
        };
        for name in LOCKSTEP_NAMES {
            assert_eq!(uses(&plan, name), 0, "{name} on {profile_name}");
        }
    }
}

/// The store loop keeps its barrier where lockstep's order is observable: a read in the loop of a
/// word another thread stores, an address that moves with the iteration, or threads that leave the
/// loop after different numbers of iterations.
#[test]
fn a_store_loop_another_thread_reads_moves_or_leaves_early_keeps_its_barrier() {
    for (name, body) in [
        ("device_shared_slot_read", DEVICE_SHARED_SLOT_READ),
        ("device_moving_slot", DEVICE_MOVING_SLOT),
        ("device_shared_slot_divergent", DEVICE_SHARED_SLOT_DIVERGENT),
    ] {
        let Some(bc) = assemble(name, body) else {
            return;
        };
        for profile_name in LONE_PROFILES {
            let p = profile(profile_name, DevicePolicy::default()).expect("profile");
            let (plan, _) = typed_module(&bc, name, &p)
                .unwrap_or_else(|e| panic!("{name} on {profile_name}: {e}"));
            let barriers: u32 = LOCKSTEP_NAMES.iter().map(|n| uses(&plan, n)).sum();
            assert!(
                barriers >= 1,
                "{name} on {profile_name}: no lockstep barrier"
            );
        }
    }
}

/// The lockstep census of kernel `name` (`body`, whose `t` is its `thread_position_in_grid`) on
/// each of [`LONE_PROFILES`], summed.
fn grid_barriers(name: &str, body: &str) -> Option<Vec<(&'static str, u32)>> {
    let text = format!("{HEADER}\n{body}\n{}", footer(name)).replace(
        "air.thread_index_in_threadgroup",
        "air.thread_position_in_grid",
    );
    let bc = assemble_text(name, &text)?;
    let mut out = Vec::new();
    for profile_name in LONE_PROFILES {
        let p = profile(profile_name, DevicePolicy::default()).expect("profile");
        let (plan, _) =
            typed_module(&bc, name, &p).unwrap_or_else(|e| panic!("{name} on {profile_name}: {e}"));
        out.push((
            profile_name,
            LOCKSTEP_NAMES.iter().map(|n| uses(&plan, n)).sum(),
        ));
    }
    Some(out)
}

/// A record per grid position takes no barrier between its words: positions of two threads of a
/// simdgroup are equal or 1 to 1023 apart, and its words are apart either way. Two stores to one
/// word per position keep theirs: threads that share a position meet there.
#[test]
fn grid_position_records_take_no_barrier_unless_two_stores_share_a_word() {
    let Some(record) = grid_barriers("device_record", DEVICE_RECORD) else {
        return;
    };
    for (profile_name, n) in record {
        assert_eq!(n, 0, "device_record on {profile_name}");
    }
    let Some(shared) = grid_barriers("device_record_shared", DEVICE_RECORD_SHARED) else {
        return;
    };
    for (profile_name, n) in shared {
        assert!(
            n >= 1,
            "device_record_shared on {profile_name}: no lockstep barrier"
        );
    }
}
