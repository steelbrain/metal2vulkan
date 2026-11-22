//! Authored kernels (inline AIR text, for `llvm-as`) whose loop trips leave values that are read
//! after the loop, and what each thread of them writes: the product's CPU-interpreter tests and
//! validation's device tests both run them.
//!
//! Every kernel takes `%limits` (buffer 0, read: the case's two words), `%out` (buffer 1, one
//! word per thread) and `%t` (the thread's index in its threadgroup), and its module declares the
//! threadgroup arrays `@a` and `@b` of 128 words, the threadgroup barrier and `simd_sum` ([`module`]).
//!
//! - [`LOOP_LEAVER`]: a loop with a per-thread trip count and a threadgroup barrier in it (a
//!   predicated walk: the walk runs the later trips for a thread that has left), whose last
//!   trip's value each thread reads after the loop.
//! - [`TRIP_EXITS`]: a loop with three exits; the second exit reads a value its trip defined
//!   after the loop's first exit test.
//! - [`NESTED_EXITS`]: an outer loop with four exits and an inner loop with two, one of which
//!   leaves both loops; each exit reads values defined on its own path.

/// The module of kernel `name` whose body is `body`.
pub fn module(name: &str, body: &str) -> String {
    format!(
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v28-apple-macosx26.0.0"

@a = internal unnamed_addr addrspace(3) global [128 x i32] undef, align 4
@b = internal unnamed_addr addrspace(3) global [128 x i32] undef, align 4

define void @{name}(ptr addrspace(1) noundef readonly %limits, ptr addrspace(1) noundef %out, i32 noundef %t) #0 {{
{body}
}}

declare void @air.wg.barrier(i32, i32) #1
declare i32 @air.simd_sum.u.i32(i32) #1

attributes #0 = {{ convergent mustprogress nounwind willreturn "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "stack-protector-buffer-size"="8" }}
attributes #1 = {{ convergent mustprogress nounwind willreturn }}

!llvm.module.flags = !{{!0, !1, !2, !3, !4, !5, !6, !7, !8}}
!air.kernel = !{{!9}}
!air.compile_options = !{{!15, !16}}
!air.version = !{{!19}}
!air.language_version = !{{!20}}

!0 = !{{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 5]}}
!1 = !{{i32 1, !"wchar_size", i32 4}}
!2 = !{{i32 7, !"frame-pointer", i32 2}}
!3 = !{{i32 7, !"air.max_device_buffers", i32 31}}
!4 = !{{i32 7, !"air.max_constant_buffers", i32 31}}
!5 = !{{i32 7, !"air.max_threadgroup_buffers", i32 31}}
!6 = !{{i32 7, !"air.max_textures", i32 128}}
!7 = !{{i32 7, !"air.max_read_write_textures", i32 8}}
!8 = !{{i32 7, !"air.max_samplers", i32 16}}
!9 = !{{ptr @{name}, !10, !11}}
!10 = !{{}}
!11 = !{{!12, !13, !14}}
!12 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"limits"}}
!13 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!14 = !{{i32 2, !"air.thread_index_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"t"}}
!15 = !{{!"air.compile.denorms_disable"}}
!16 = !{{!"air.compile.fast_math_enable"}}
!19 = !{{i32 2, i32 8, i32 0}}
!20 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
    )
}

/// `i = 0; do { a[t] = 16t + i; barrier; x = a[t ^ 4]; barrier; v = (x + i) | 1 << 31; i++; }
/// while (i <= (t & 3)); out[t] = v;`: thread `t` leaves after `(t & 3) + 1` trips, and thread
/// `t ^ 4` runs as many, so it is in the same trip.
pub const LOOP_LEAVER: &str = r#"entry:
  %k = and i32 %t, 3
  %tz = zext i32 %t to i64
  %pa = getelementptr inbounds [128 x i32], ptr addrspace(3) @a, i64 0, i64 %tz
  %p = xor i32 %t, 4
  %pz = zext i32 %p to i64
  %pp = getelementptr inbounds [128 x i32], ptr addrspace(3) @a, i64 0, i64 %pz
  %w0 = shl i32 %t, 4
  br label %body
body:
  %i = phi i32 [ 0, %entry ], [ %i1, %body ]
  %w = add i32 %w0, %i
  store i32 %w, ptr addrspace(3) %pa, align 4
  tail call void @air.wg.barrier(i32 2, i32 1) #1
  %x = load i32, ptr addrspace(3) %pp, align 4
  tail call void @air.wg.barrier(i32 2, i32 1) #1
  %v0 = add i32 %x, %i
  %v = or i32 %v0, -2147483648
  %i1 = add i32 %i, 1
  %more = icmp ule i32 %i1, %k
  br i1 %more, label %body, label %done
done:
  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tz
  store i32 %v, ptr addrspace(1) %po, align 4
  ret void"#;

/// Thread `t`'s word of [`LOOP_LEAVER`], whose threadgroup holds thread `t ^ 4`.
pub fn loop_leaver(t: u32) -> u32 {
    (((t ^ 4) << 4) + 2 * (t & 3)) | 1 << 31
}

/// `for (i = 0; i < limits[0]; i++) { x = (7i + t) & 15; if (x == 3) { out[t] = 100i + x;
/// return; } if (x > 12) { out[t] = i + 50; return; } } out[t] = 999;`: `x` is defined after the
/// loop's first exit test and read after the loop by its second exit.
pub const TRIP_EXITS: &str = r#"entry:
  %n = load i32, ptr addrspace(1) %limits, align 4
  %tz = zext i32 %t to i64
  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tz
  br label %header
header:
  %i = phi i32 [ 0, %entry ], [ %i1, %latch ]
  %more = icmp ult i32 %i, %n
  br i1 %more, label %b1, label %none
b1:
  %i7 = mul i32 %i, 7
  %s = add i32 %i7, %t
  %x = and i32 %s, 15
  %hit = icmp eq i32 %x, 3
  br i1 %hit, label %found, label %b2
b2:
  %big = icmp ugt i32 %x, 12
  br i1 %big, label %large, label %latch
latch:
  %i1 = add i32 %i, 1
  br label %header
found:
  %ih = mul i32 %i, 100
  %r = add i32 %ih, %x
  store i32 %r, ptr addrspace(1) %po, align 4
  br label %done
large:
  %rl = add i32 %i, 50
  store i32 %rl, ptr addrspace(1) %po, align 4
  br label %done
none:
  store i32 999, ptr addrspace(1) %po, align 4
  br label %done
done:
  ret void"#;

/// Thread `t`'s word of [`TRIP_EXITS`] for `limits = [n, _]`.
pub fn trip_exits(n: u32, t: u32) -> u32 {
    for i in 0..n {
        let x = (7 * i + t) & 15;
        if x == 3 {
            return 100 * i + x;
        }
        if x > 12 {
            return i + 50;
        }
    }
    999
}

/// `for (i = 0; i < limits[0]; i++) { a = (5i + t) & 15; if (a == 1) { out[t] = 10000 + 16i + a;
/// return; } for (j = 0; j < limits[1]; j++) { b = (3j + a + t) & 31; if (b == 7) { out[t] =
/// 20000 + 256i + 8j + b; return; } if (b > 28) break; } c = a + i; if (c > 20) { out[t] = 30000 +
/// c; return; } } out[t] = 999;`: the outer loop's exits read `a`, then `b`, `i` and `j` (an exit
/// of both loops from the inner one), then `c`, each defined on its exit's own path.
pub const NESTED_EXITS: &str = r#"entry:
  %n = load i32, ptr addrspace(1) %limits, align 4
  %pm = getelementptr inbounds i32, ptr addrspace(1) %limits, i64 1
  %m = load i32, ptr addrspace(1) %pm, align 4
  %tz = zext i32 %t to i64
  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tz
  br label %oh
oh:
  %i = phi i32 [ 0, %entry ], [ %i1, %olatch ]
  %om = icmp ult i32 %i, %n
  br i1 %om, label %ob, label %none
ob:
  %i5 = mul i32 %i, 5
  %as = add i32 %i5, %t
  %a = and i32 %as, 15
  %ahit = icmp eq i32 %a, 1
  br i1 %ahit, label %exa, label %ih
ih:
  %j = phi i32 [ 0, %ob ], [ %j1, %ilatch ]
  %im = icmp ult i32 %j, %m
  br i1 %im, label %ib, label %after
ib:
  %j3 = mul i32 %j, 3
  %bs0 = add i32 %j3, %a
  %bs = add i32 %bs0, %t
  %b = and i32 %bs, 31
  %bhit = icmp eq i32 %b, 7
  br i1 %bhit, label %exb, label %ib2
ib2:
  %bbig = icmp ugt i32 %b, 28
  br i1 %bbig, label %after, label %ilatch
ilatch:
  %j1 = add i32 %j, 1
  br label %ih
after:
  %c = add i32 %a, %i
  %cbig = icmp ugt i32 %c, 20
  br i1 %cbig, label %exc, label %olatch
olatch:
  %i1 = add i32 %i, 1
  br label %oh
exa:
  %ia = shl i32 %i, 4
  %ra0 = add i32 %ia, %a
  %ra = add i32 %ra0, 10000
  store i32 %ra, ptr addrspace(1) %po, align 4
  br label %done
exb:
  %ib8 = shl i32 %i, 8
  %jb8 = shl i32 %j, 3
  %rb0 = add i32 %ib8, %jb8
  %rb1 = add i32 %rb0, %b
  %rb = add i32 %rb1, 20000
  store i32 %rb, ptr addrspace(1) %po, align 4
  br label %done
exc:
  %rc = add i32 %c, 30000
  store i32 %rc, ptr addrspace(1) %po, align 4
  br label %done
none:
  store i32 999, ptr addrspace(1) %po, align 4
  br label %done
done:
  ret void"#;

/// Thread `t`'s word of [`NESTED_EXITS`] for `limits = [n, m]`.
pub fn nested_exits(n: u32, m: u32, t: u32) -> u32 {
    for i in 0..n {
        let a = (5 * i + t) & 15;
        if a == 1 {
            return 10000 + 16 * i + a;
        }
        for j in 0..m {
            let b = (3 * j + a + t) & 31;
            if b == 7 {
                return 20000 + 256 * i + 8 * j + b;
            }
            if b > 28 {
                break;
            }
        }
        let c = a + i;
        if c > 20 {
            return 30000 + c;
        }
    }
    999
}

/// The trip bounds the loop-exit kernels are run with: `[n, m]`, `m` read by [`NESTED_EXITS`]
/// alone. They reach every exit of both kernels for some thread of 32.
pub const BOUNDS: [[u32; 2]; 6] = [[0, 0], [1, 0], [2, 5], [5, 3], [8, 8], [16, 2]];
