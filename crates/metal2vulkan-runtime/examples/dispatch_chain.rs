//! Not a consumer's example: what a chain of dependent compute dispatches costs the GPU, by how a
//! command buffer holds them, beside the same chain recorded straight through Vulkan on a
//! `VkDevice` of the example's own, made on the runtime device's physical device.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example dispatch_chain -- \
//!     [--dispatches N] [--groups G] [--threads T] [--runs R] [--only SHAPE,...] \
//!     [--binds 1|4|tex|ptr] [--host C] [--pages] [--heap] [--heap-churn K] [--heap-kept H] \
//!     [--live L]
//! ```
//!
//! Each run is one command buffer of `N` dispatches (200 by default) of `G` threadgroups of `T`
//! threads (4096 of 32 by default) over one `Private` buffer of `G * T` words, each thread adding
//! 1 to its own word: every dispatch reads what the one before wrote. The shapes:
//!
//! - **`encoders`**: `N` compute encoders of one dispatch each, as a program that opens an encoder
//!   for each dispatch records them;
//! - **`serial`**: one serial compute encoder of the `N` dispatches;
//! - **`concurrent`**: one concurrent encoder of them, with no memory barrier: nothing orders the
//!   dispatches, so this is the kernels' own time with no barrier between them (their words race,
//!   which is not checked);
//! - **`raw-barrier`**: one `VkCommandBuffer` of the `N` dispatches of the same work (a SPIR-V
//!   kernel written here, assembled by `spirv-as`), each after one `VkMemoryBarrier` from
//!   `COMPUTE_SHADER`'s `SHADER_WRITE` to `COMPUTE_SHADER`'s `SHADER_READ | SHADER_WRITE`, the
//!   least barrier Vulkan states for a read after a write between dispatches;
//! - **`raw-wide`**: the same, with the runtime's barrier between dispatches of an encoder
//!   (`Barrier::between_dispatches`: also `DRAW_INDIRECT` and its `INDIRECT_COMMAND_READ`, and
//!   `UNIFORM_READ`);
//! - **`raw-split`**: `raw-barrier` with each dispatch in a `VkCommandBuffer` of its own, after
//!   a `VkCommandBuffer` holding its barrier alone: `2N` command buffers in one `vkQueueSubmit`;
//! - **`raw-none`**: one `VkCommandBuffer` of the dispatches with no barrier.
//!
//! The runtime's shapes take the GPU's time from the command buffer's `gpu_start_time` to its
//! `gpu_end_time`, the raw ones from a timestamp before the first dispatch to one after the last.
//! Each line is the median over `R` runs (9 by default) after one not counted, per dispatch, with
//! the runtime's counts per command buffer that moved (`WorkCounts`, by name from their `Debug`
//! form), and those of the run not counted, the shape's first command buffer, apart: its first
//! recording of the kernel's dispatches, as a program's first command buffer of a kind is. The
//! words are read back after each chained shape: each must hold the dispatches run on it.
//!
//! ```text
//! CHAIN <shape> dispatches <N> groups <G> threads <T> gpu_us_per_dispatch <median> least <least>
//!     <device>
//! CHAIN-COUNTS <shape> <name>=<per command buffer> ...
//! CHAIN-FIRST <shape> <name>=<the first command buffer's> ...
//! ```
//!
//! With `--binds 4` the runtime's shapes run `bump_wide`, which also reads two `Private` buffers of
//! zeros at indices 1 and 2 and a word of 1 set as bytes at index 3 (`a[gid] += b[gid] + c[gid] +
//! k`), each dispatch binding its two zero buffers the other way round from the dispatch before it.
//! With `--binds tex` they run `bump_tex`, which samples two 16 by 16 `RGBA8Unorm` textures at
//! indices 0 and 1 through a sampler at 0, and reads a word of 1 set as bytes at buffer index 1
//! (`a[gid] += k`, plus 1000 where a texel's red is below -1e30, which none is): each dispatch
//! binds the pipeline state, the two textures (the other way round from the dispatch before it),
//! the sampler, the words and the bytes, the shape of Geekbench's compute workloads' encoders.
//! With `--binds ptr` they run `bump_ptr`, `bump_wide` with its first zero buffer read through an
//! address the dispatch sets as bytes at index 1 (a `Holder` of one pointer, as an argument buffer
//! holds it): a command that loads a pointer, which the device's resident set of its live buffers
//! resolves (`live_buffers.rs`).
//! With `--pages` the words and the zero buffers are `Shared` no-copy buffers over host pages
//! (`new_buffer_with_bytes_no_copy`), as a guest's buffers over its own RAM are on a device with
//! unified memory, which makes the device log the encoders' calls and hold the buffers its commands
//! name (`replay.rs`, `rebind.rs`). With `--heap` they are instead placed, 64 KiB apart, in one
//! `Placement` heap over host pages (`new_heap_with_bytes_no_copy`, then `new_buffer_at_offset`),
//! beside a no-copy buffer over those pages and 64 KiB more before them, as the consumer places a
//! guest's buffers in a heap over the guest's RAM, whose own buffer stands for the RAM: each use of
//! the heap's bytes takes a use of the RAM's key too (`host_aliases.rs`'s placements).
//!
//! **The host's time** (`--host C`): the runtime's shapes alone, each run `C` command buffers of the
//! `N` dispatches, every dispatch setting its pipeline state and buffers again as a program that
//! encodes a dispatch's whole state does, each command buffer committed with `commit_and_submit`
//! (as a consumer that submits on its own thread commits) and waited for before the next is made,
//! outside the times. Each call is timed on the calling thread; per dispatch, the means over the
//! `R` runs after one not counted:
//!
//! ```text
//! HOST <shape> dispatches <N> command_buffers <C> binds <B> ns_per_dispatch make <ns> open <ns>
//!     pipeline <ns> bind <ns> dispatch <ns> end <ns> encode <ns> commit <ns> thread_cpu <ns>
//!     least_encode_commit <ns>
//! HOST-COMMIT <shape> lowering <ns> recording <ns> pipeline_wait <ns> lock_wait <ns> pass <ns>
//!     submit <ns>
//! HOST-THREADS <shape> <group>=<ns> ...
//! ```
//!
//! `make` is `command_buffer`, `open` the encoders' opening, `pipeline` and `bind` the state's
//! calls, `end` `end_encoding`; `encode` is their sum and `commit` the `commit_and_submit` call;
//! `thread_cpu` the calling thread's CPU time over both (`CLOCK_THREAD_CPUTIME_ID`), and
//! `least_encode_commit` the least run's. `HOST-COMMIT` splits the commit call by the device's
//! commit phases (`WorkCounts::commit_lowering_nanos` and kin). `HOST-THREADS` is each group of the
//! process's threads' CPU time per dispatch over the counted runs, waits included, by the thread's
//! name (Linux only, from `/proc/self/task/*/schedstat`): `committer` is the calling thread (the
//! process's first), the driver's threads (a CPU device's execution among them) are `driver`.
//!
//! **Buffers in a heap a command buffer** (`--heap-churn K`, with `--host` and `--binds 4` or
//! `ptr`): each command buffer first makes `K` buffers of the words' length in an automatic
//! `Private` heap of the example's own, made once with room for them and zeroed, as a program that
//! takes a command buffer's scratch from a heap does; its dispatches take them as the two read-only
//! buffers in turn, and they are let go once it completed. `--heap-kept H` places `H` more buffers
//! of 256 bytes in that heap and `--live L` makes `L` more of the device's own, all living through
//! the runs, as a program's other buffers do. Per command buffer, the means over the counted runs:
//!
//! ```text
//! HOST-CB <shape> churn <K> kept <H> live <L> ns_per_command_buffer make <ns> place <ns>
//!     encode <ns> commit <ns> free <ns> <group>=<ns> ...
//! ```
//!
//! `make` is `command_buffer`, `place` the `K` buffers' making, `encode` the encoders' calls,
//! `commit` the `commit_and_submit` call and `free` the `K` buffers' release after completion,
//! each on the calling thread; then each group of threads' CPU time (as `HOST-THREADS`), the
//! scheduler's among them.
//!
//! The kernels are text here: LLVM IR assembled by `llvm-as`, SPIR-V by `spirv-as`, both from
//! `PATH`.

use ash::vk;
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{DispatchType, PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    Buffer, CommandBufferStatus, CommandQueue, ComputeCommandEncoder, ComputePipelineState, Device,
    DeviceOptions, Heap, HeapDescriptor, HeapType, HostMemory, PipelineOptions, ResourceOptions,
    SamplerState, Size, Texture, TextureDescriptor, WorkCounts,
};
use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};
use std::ptr::NonNull;
use std::time::{Duration, Instant};

/// The example's result: every runtime refusal is a typed `metal2vulkan_runtime::Error`.
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// The longest the example waits for a command buffer.
const WAIT: Duration = Duration::from_secs(60);

/// `bump`: `a[gid] += 1`, `a` a buffer of words at index 0.
const KERNEL: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @bump(ptr addrspace(1) noundef %a, i32 noundef %gid) {\n\
entry:\n\
  %gw = zext i32 %gid to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %a, i64 %gw\n\
  %x = load i32, ptr addrspace(1) %p, align 4\n\
  %v = add i32 %x, 1\n\
  store i32 %v, ptr addrspace(1) %p, align 4\n\
  ret void\n\
}\n\n\
!llvm.module.flags = !{!0, !1}\n!air.kernel = !{!20}\n!air.version = !{!8}\n!air.language_version = !{!9}\n\
!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n\
!1 = !{i32 7, !\"air.max_threadgroup_buffers\", i32 31}\n\
!3 = !{}\n!4 = !{!5, !7}\n\
!5 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"a\"}\n\
!7 = !{i32 1, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}\n\
!8 = !{i32 2, i32 8, i32 0}\n!9 = !{!\"Metal\", i32 4, i32 0, i32 0}\n\
!20 = !{ptr @bump, !3, !4}\n";

/// `bump_wide`: `a[gid] += b[gid] + c[gid] + *k`, `a` a buffer of words at index 0, `b` and `c`
/// read-only ones at 1 and 2, `k` a `constant` word at 3.
const KERNEL_WIDE: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @bump_wide(ptr addrspace(1) noundef %a, ptr addrspace(1) noundef readonly %b, ptr addrspace(1) noundef readonly %c, ptr addrspace(2) noundef %k, i32 noundef %gid) {\n\
entry:\n\
  %gw = zext i32 %gid to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %a, i64 %gw\n\
  %pb = getelementptr inbounds i32, ptr addrspace(1) %b, i64 %gw\n\
  %pc = getelementptr inbounds i32, ptr addrspace(1) %c, i64 %gw\n\
  %x = load i32, ptr addrspace(1) %p, align 4\n\
  %y = load i32, ptr addrspace(1) %pb, align 4\n\
  %z = load i32, ptr addrspace(1) %pc, align 4\n\
  %w = load i32, ptr addrspace(2) %k, align 4\n\
  %s1 = add i32 %x, %y\n\
  %s2 = add i32 %s1, %z\n\
  %v = add i32 %s2, %w\n\
  store i32 %v, ptr addrspace(1) %p, align 4\n\
  ret void\n\
}\n\n\
!llvm.module.flags = !{!0, !1}\n!air.kernel = !{!20}\n!air.version = !{!8}\n!air.language_version = !{!9}\n\
!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n\
!1 = !{i32 7, !\"air.max_threadgroup_buffers\", i32 31}\n\
!3 = !{}\n!4 = !{!5, !10, !11, !12, !7}\n\
!5 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"a\"}\n\
!10 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"b\"}\n\
!11 = !{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"c\"}\n\
!12 = !{i32 3, !\"air.buffer\", !\"air.buffer_size\", i32 4, !\"air.location_index\", i32 3, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"k\"}\n\
!7 = !{i32 4, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}\n\
!8 = !{i32 2, i32 8, i32 0}\n!9 = !{!\"Metal\", i32 4, i32 0, i32 0}\n\
!20 = !{ptr @bump_wide, !3, !4}\n";

/// `bump_ptr`: `bump_wide` with `b` read through the pointer `h` holds, `h` a `Holder` of one
/// pointer at index 1.
const KERNEL_PTR: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
%struct.Holder = type { ptr addrspace(1) }\n\n\
define void @bump_ptr(ptr addrspace(1) noundef %a, ptr addrspace(1) noundef readonly %h, ptr addrspace(1) noundef readonly %c, ptr addrspace(2) noundef %k, i32 noundef %gid) {\n\
entry:\n\
  %gw = zext i32 %gid to i64\n\
  %hp = getelementptr inbounds %struct.Holder, ptr addrspace(1) %h, i64 0, i32 0\n\
  %b = load ptr addrspace(1), ptr addrspace(1) %hp, align 8\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %a, i64 %gw\n\
  %pb = getelementptr inbounds i32, ptr addrspace(1) %b, i64 %gw\n\
  %pc = getelementptr inbounds i32, ptr addrspace(1) %c, i64 %gw\n\
  %x = load i32, ptr addrspace(1) %p, align 4\n\
  %y = load i32, ptr addrspace(1) %pb, align 4\n\
  %z = load i32, ptr addrspace(1) %pc, align 4\n\
  %w = load i32, ptr addrspace(2) %k, align 4\n\
  %s1 = add i32 %x, %y\n\
  %s2 = add i32 %s1, %z\n\
  %v = add i32 %s2, %w\n\
  store i32 %v, ptr addrspace(1) %p, align 4\n\
  ret void\n\
}\n\n\
!llvm.module.flags = !{!0, !1}\n!air.kernel = !{!20}\n!air.version = !{!8}\n!air.language_version = !{!9}\n\
!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n\
!1 = !{i32 7, !\"air.max_threadgroup_buffers\", i32 31}\n\
!3 = !{}\n!4 = !{!5, !10, !11, !12, !7}\n\
!5 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"a\"}\n\
!10 = !{i32 1, !\"air.indirect_buffer\", !\"air.buffer_size\", i32 8, !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.struct_type_info\", !13, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"Holder\", !\"air.arg_name\", !\"h\"}\n\
!13 = !{i32 0, i32 8, i32 0, !\"uint\", !\"p\", !\"air.indirect_argument\", !14}\n\
!14 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"p\"}\n\
!11 = !{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"c\"}\n\
!12 = !{i32 3, !\"air.buffer\", !\"air.buffer_size\", i32 4, !\"air.location_index\", i32 3, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"k\"}\n\
!7 = !{i32 4, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}\n\
!8 = !{i32 2, i32 8, i32 0}\n!9 = !{!\"Metal\", i32 4, i32 0, i32 0}\n\
!20 = !{ptr @bump_ptr, !3, !4}\n";

/// `bump_tex`: `a[gid] += k`, plus 1000 where a texel `t0` or `t1` samples through `s` at the
/// middle has its red below -1e30; `t0`, `t1` textures at 0 and 1, `s` a sampler at 0, `a` a
/// buffer of words at index 0, `k` a `constant` word at 1.
const KERNEL_TEX: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n\
target triple = \"air64_v28-apple-macosx26.0.0\"\n\n\
define void @bump_tex(ptr addrspace(1) %t0, ptr addrspace(1) %t1, ptr addrspace(2) %s, ptr addrspace(1) noundef %a, ptr addrspace(2) noundef %k, i32 noundef %gid) {\n\
entry:\n\
  %r0 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t0, ptr addrspace(2) %s, <2 x float> <float 0.5, float 0.5>, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)\n\
  %r1 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %t1, ptr addrspace(2) %s, <2 x float> <float 0.5, float 0.5>, i1 true, <2 x i32> zeroinitializer, i1 true, float 0.0, float 0.0, i32 0)\n\
  %v0 = extractvalue { <4 x float>, i8 } %r0, 0\n\
  %v1 = extractvalue { <4 x float>, i8 } %r1, 0\n\
  %x0 = extractelement <4 x float> %v0, i64 0\n\
  %x1 = extractelement <4 x float> %v1, i64 0\n\
  %x = fadd float %x0, %x1\n\
  %low = fcmp olt float %x, -1.0e30\n\
  %e = select i1 %low, i32 1000, i32 0\n\
  %gw = zext i32 %gid to i64\n\
  %p = getelementptr inbounds i32, ptr addrspace(1) %a, i64 %gw\n\
  %w = load i32, ptr addrspace(1) %p, align 4\n\
  %kv = load i32, ptr addrspace(2) %k, align 4\n\
  %s1 = add i32 %w, %kv\n\
  %v = add i32 %s1, %e\n\
  store i32 %v, ptr addrspace(1) %p, align 4\n\
  ret void\n\
}\n\n\
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)\n\n\
!llvm.module.flags = !{!0, !1}\n!air.kernel = !{!20}\n!air.version = !{!8}\n!air.language_version = !{!9}\n\
!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n\
!1 = !{i32 7, !\"air.max_threadgroup_buffers\", i32 31}\n\
!3 = !{}\n!4 = !{!10, !11, !12, !5, !13, !7}\n\
!10 = !{i32 0, !\"air.texture\", !\"air.location_index\", i32 0, i32 1, !\"air.sample\", !\"air.arg_type_name\", !\"texture2d<float, sample>\", !\"air.arg_name\", !\"t0\"}\n\
!11 = !{i32 1, !\"air.texture\", !\"air.location_index\", i32 1, i32 1, !\"air.sample\", !\"air.arg_type_name\", !\"texture2d<float, sample>\", !\"air.arg_name\", !\"t1\"}\n\
!12 = !{i32 2, !\"air.sampler\", !\"air.location_index\", i32 0, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}\n\
!5 = !{i32 3, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"a\"}\n\
!13 = !{i32 4, !\"air.buffer\", !\"air.buffer_size\", i32 4, !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"k\"}\n\
!7 = !{i32 5, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}\n\
!8 = !{i32 2, i32 8, i32 0}\n!9 = !{!\"Metal\", i32 4, i32 0, i32 0}\n\
!20 = !{ptr @bump_tex, !3, !4}\n";

/// The same work in SPIR-V, its threadgroup `{T}` threads wide: `words[gid.x] += 1` over a
/// storage buffer at set 0, binding 0.
const SPIRV: &str = "OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main \"main\" %gid %buf
OpExecutionMode %main LocalSize {T} 1 1
OpDecorate %gid BuiltIn GlobalInvocationId
OpDecorate %words ArrayStride 4
OpMemberDecorate %block 0 Offset 0
OpDecorate %block Block
OpDecorate %buf DescriptorSet 0
OpDecorate %buf Binding 0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%v3uint = OpTypeVector %uint 3
%in_v3 = OpTypePointer Input %v3uint
%gid = OpVariable %in_v3 Input
%words = OpTypeRuntimeArray %uint
%block = OpTypeStruct %words
%sb_block = OpTypePointer StorageBuffer %block
%buf = OpVariable %sb_block StorageBuffer
%sb_uint = OpTypePointer StorageBuffer %uint
%zero = OpConstant %uint 0
%one = OpConstant %uint 1
%main = OpFunction %void None %fn
%entry = OpLabel
%g = OpLoad %v3uint %gid
%x = OpCompositeExtract %uint %g 0
%p = OpAccessChain %sb_uint %buf %zero %x
%v = OpLoad %uint %p
%w = OpIAdd %uint %v %one
OpStore %p %w
OpReturn
OpFunctionEnd
";

/// What a dispatch binds (`--binds`), and so its kernel.
#[derive(Clone, Copy, PartialEq)]
enum Binds {
    /// `bump`'s words alone.
    One,
    /// `bump_wide`'s words, two buffers of zeros and bytes.
    Wide,
    /// `bump_tex`'s two textures, sampler, words and bytes.
    Textures,
    /// `bump_ptr`'s words, the address of a buffer of zeros, another buffer of zeros and bytes.
    Pointer,
}

impl Binds {
    fn name(self) -> &'static str {
        match self {
            Binds::One => "1",
            Binds::Wide => "4",
            Binds::Textures => "tex",
            Binds::Pointer => "ptr",
        }
    }
}

/// The run's shape: dispatches a command buffer, threadgroups a dispatch, threads a threadgroup,
/// timed runs, and the shapes to run.
struct Shape {
    dispatches: u32,
    groups: u32,
    threads: u32,
    runs: usize,
    only: Option<Vec<String>>,
    /// What a dispatch binds.
    binds: Binds,
    /// Command buffers a run of the host's time (`--host`), `None` for the GPU's.
    host: Option<u32>,
    /// Whether the words and zeros lie in no-copy buffers over host pages (`--pages`).
    pages: bool,
    /// Whether the words and zeros are placed in a heap over host pages beside a buffer over them
    /// (`--heap`).
    heap: bool,
    /// Buffers made in a heap of their own for each command buffer of the host's time
    /// (`--heap-churn`).
    churn: Option<u32>,
    /// Buffers of 256 bytes in that heap that live through the runs (`--heap-kept`).
    kept: u32,
    /// Buffers of 256 bytes of the device's own that live through the runs (`--live`).
    live: u32,
}

impl Shape {
    fn from_args() -> Result<Shape> {
        let mut shape = Shape {
            dispatches: 200,
            groups: 4096,
            threads: 32,
            runs: 9,
            only: None,
            binds: Binds::One,
            host: None,
            pages: false,
            heap: false,
            churn: None,
            kept: 0,
            live: 0,
        };
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            if flag == "--pages" || flag == "--heap" {
                shape.pages |= flag == "--pages";
                shape.heap |= flag == "--heap";
                continue;
            }
            let value = args.next().ok_or(format!("{flag} takes a value"))?;
            match flag.as_str() {
                "--dispatches" => shape.dispatches = value.parse()?,
                "--groups" => shape.groups = value.parse()?,
                "--threads" => shape.threads = value.parse()?,
                "--runs" => shape.runs = value.parse()?,
                "--only" => shape.only = Some(value.split(',').map(str::to_owned).collect()),
                "--binds" => {
                    shape.binds = match value.as_str() {
                        "1" => Binds::One,
                        "4" => Binds::Wide,
                        "tex" => Binds::Textures,
                        "ptr" => Binds::Pointer,
                        _ => return Err("--binds takes 1, 4, tex or ptr".into()),
                    }
                }
                "--host" => shape.host = Some(value.parse()?),
                "--heap-churn" => shape.churn = Some(value.parse()?),
                "--heap-kept" => shape.kept = value.parse()?,
                "--live" => shape.live = value.parse()?,
                _ => return Err(format!("unknown flag {flag}").into()),
            }
        }
        let zeros_read = matches!(shape.binds, Binds::Wide | Binds::Pointer);
        if shape.churn.is_some() && (shape.host.is_none() || !zeros_read) {
            return Err("--heap-churn takes --host and --binds 4 or ptr".into());
        }
        Ok(shape)
    }

    fn runs(&self, name: &str) -> bool {
        self.only
            .as_ref()
            .is_none_or(|only| only.iter().any(|shape| shape == name))
    }

    fn words(&self) -> u64 {
        u64::from(self.groups) * u64::from(self.threads)
    }
}

/// Run `tool` from `PATH` over `input` on its standard input: its standard output.
fn tool(tool: &str, args: &[&str], input: &str) -> Result<Vec<u8>> {
    let mut child = Command::new(tool)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("cannot run {tool} (put it on PATH): {error}"))?;
    child
        .stdin
        .take()
        .ok_or("no stdin")?
        .write_all(input.as_bytes())?;
    let output = child.wait_with_output()?;
    if !output.status.success() {
        return Err(format!("{tool}: {}", String::from_utf8_lossy(&output.stderr)).into());
    }
    Ok(output.stdout)
}

/// The state of the kernel `binds` binds for.
fn state(device: &Device, binds: Binds) -> Result<ComputePipelineState> {
    let (text, name) = match binds {
        Binds::One => (KERNEL, "bump"),
        Binds::Wide => (KERNEL_WIDE, "bump_wide"),
        Binds::Textures => (KERNEL_TEX, "bump_tex"),
        Binds::Pointer => (KERNEL_PTR, "bump_ptr"),
    };
    let bitcode = tool("llvm-as", &["-o", "-", "-"], text)?;
    let library = metal2vulkan::library::Library::from_air(&bitcode)?;
    let function = device
        .new_library_with_compile_library(library)
        .new_function_with_name(name)?;
    Ok(device.new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())?)
}

/// The counts of `counts` by name, read from their `Debug` form.
fn counts(counts: &WorkCounts) -> BTreeMap<String, u64> {
    let text = format!("{counts:?}");
    text.split([',', '{'])
        .filter_map(|part| {
            let (name, value) = part.split_once(':')?;
            Some((name.trim().to_owned(), value.trim().parse().ok()?))
        })
        .collect()
}

/// The samples of one shape: GPU microseconds per dispatch, the counts' sums over the runs
/// counted, and the counts of the run not counted, its first.
#[derive(Default)]
struct Samples {
    per_dispatch: Vec<f64>,
    counts: BTreeMap<String, u64>,
    first: BTreeMap<String, u64>,
}

/// What moved from `before` to `after`, added to `into`.
fn add_moved(into: &mut BTreeMap<String, u64>, before: &WorkCounts, after: &WorkCounts) {
    let before = counts(before);
    for (name, value) in counts(after) {
        let moved = value.saturating_sub(before.get(&name).copied().unwrap_or(0));
        if moved > 0 {
            *into.entry(name).or_default() += moved;
        }
    }
}

impl Samples {
    fn add_counts(&mut self, before: &WorkCounts, after: &WorkCounts) {
        add_moved(&mut self.counts, before, after);
    }

    fn print(mut self, name: &str, shape: &Shape, device: &str) {
        self.per_dispatch.sort_by(f64::total_cmp);
        println!(
            "CHAIN {name} dispatches {} groups {} threads {} gpu_us_per_dispatch {:.2} least {:.2} \
             {device}",
            shape.dispatches,
            shape.groups,
            shape.threads,
            self.per_dispatch[self.per_dispatch.len() / 2],
            self.per_dispatch[0],
        );
        if !self.counts.is_empty() {
            let runs = self.per_dispatch.len() as f64;
            let line: Vec<String> = (self.counts.iter())
                .map(|(name, sum)| format!("{name}={:.2}", *sum as f64 / runs))
                .collect();
            println!("CHAIN-COUNTS {name} {}", line.join(" "));
        }
        if !self.first.is_empty() {
            let line: Vec<String> = (self.first.iter())
                .map(|(name, value)| format!("{name}={value}"))
                .collect();
            println!("CHAIN-FIRST {name} {}", line.join(" "));
        }
    }
}

/// A call of the host's time, in [`Lap`]'s order.
#[derive(Clone, Copy)]
enum Call {
    Make,
    Open,
    Pipeline,
    Bind,
    Dispatch,
    End,
    Commit,
    Place,
    Free,
}

/// The calls' names, in [`Call`]'s order: the `HOST` line's, then the churned heap's
/// (`--heap-churn`).
const CALLS: [&str; 9] = [
    "make", "open", "pipeline", "bind", "dispatch", "end", "commit", "place", "free",
];

/// Nanoseconds of the calling thread's time by [`Call`], over a run.
#[derive(Default, Clone, Copy)]
struct Lap([u64; 9]);

impl Lap {
    /// Count the time since `at` to `call`, and start the next call's there.
    fn lap(&mut self, at: &mut Instant, call: Call) {
        let now = Instant::now();
        self.0[call as usize] += (now - *at).as_nanos() as u64;
        *at = now;
    }
}

/// The host's time of the counted runs: each call's sum, the calling thread's CPU over them, and
/// each run's encoding and commit.
#[derive(Default)]
struct HostSample {
    calls: [u64; 9],
    thread_cpu: u64,
    per_run: Vec<u64>,
}

impl HostSample {
    fn add(&mut self, lap: Lap) {
        for (sum, nanos) in self.calls.iter_mut().zip(lap.0) {
            *sum += nanos;
        }
        self.per_run.push(lap.0[..7].iter().sum());
    }

    fn print(&self, name: &str, shape: &Shape, command_buffers: u32, dispatches: f64) {
        let per = |nanos: u64| nanos as f64 / dispatches;
        let mut line = format!(
            "HOST {name} dispatches {} command_buffers {command_buffers} binds {} ns_per_dispatch",
            shape.dispatches,
            shape.binds.name()
        );
        for (call, nanos) in CALLS.iter().zip(self.calls).take(7) {
            if *call == "commit" {
                let encode: u64 = self.calls[..6].iter().sum();
                line.push_str(&format!(" encode {:.0}", per(encode)));
            }
            line.push_str(&format!(" {call} {:.0}", per(nanos)));
        }
        let least = self.per_run.iter().min().copied().unwrap_or(0);
        let runs = self.per_run.len().max(1) as f64;
        line.push_str(&format!(
            " thread_cpu {:.0} least_encode_commit {:.0}",
            per(self.thread_cpu),
            least as f64 * runs / dispatches
        ));
        println!("{line}");
    }

    /// The `HOST-CB` line of `--heap-churn`, `threads` each group's CPU time over the counted runs.
    fn print_per_command_buffer(
        &self,
        name: &str,
        shape: &Shape,
        command_buffers: u32,
        threads: &[(String, u64)],
    ) {
        let Some(churn) = shape.churn else {
            return;
        };
        let per =
            |nanos: u64| nanos as f64 / (command_buffers as usize * self.per_run.len()) as f64;
        let encode: u64 = self.calls[1..6].iter().sum();
        let mut line = format!(
            "HOST-CB {name} churn {churn} kept {} live {} ns_per_command_buffer make {:.0} place {:.0} \
             encode {:.0} commit {:.0} free {:.0}",
            shape.kept,
            shape.live,
            per(self.calls[Call::Make as usize]),
            per(self.calls[Call::Place as usize]),
            per(encode),
            per(self.calls[Call::Commit as usize]),
            per(self.calls[Call::Free as usize]),
        );
        for (group, nanos) in threads {
            line.push_str(&format!(" {group}={:.0}", per(*nanos)));
        }
        println!("{line}");
    }
}

/// The bytes a placed buffer takes in [`Placed`]'s heap, and the RAM's before the heap's.
const CHUNK: u64 = 0x1_0000;

/// The words and the zeros' heap (`--heap`): a `Placement` heap over host pages, beside a no-copy
/// buffer over its pages and [`CHUNK`] more before them, which stands for a guest's RAM. Both are
/// kept for the process's life, so no import outlives its pages.
struct Placed {
    heap: Heap,
    _ram: Buffer,
}

impl Placed {
    /// Room for three buffers of `bytes` bytes, each in chunks of its own.
    fn new(device: &Device, bytes: u64) -> Result<Placed> {
        let each = bytes.div_ceil(CHUNK) * CHUNK;
        let length = CHUNK + 3 * each;
        let pages = host_pages(length as usize)?;
        // SAFETY: page-aligned read-write memory of `length` bytes, never unmapped, which outlives
        // the buffer.
        let ram = unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::new(pages),
                length,
                ResourceOptions::shared(),
                None,
            )
        }?;
        // SAFETY: inside the mapping, past its first chunk.
        let at = NonNull::new(unsafe { pages.as_ptr().add(CHUNK as usize) }).ok_or("null")?;
        let descriptor =
            HeapDescriptor::new(HeapType::Placement, 3 * each, ResourceOptions::shared());
        // SAFETY: the mapping past its first chunk, which outlives the heap.
        let heap = unsafe {
            device.new_heap_with_bytes_no_copy(HostMemory::new(at), &descriptor, None, None)
        }?;
        Ok(Placed { heap, _ram: ram })
    }

    /// The `n`th buffer of `bytes` bytes placed in the heap.
    fn buffer(&self, n: u64, bytes: u64) -> Result<Buffer> {
        let each = bytes.div_ceil(CHUNK) * CHUNK;
        Ok(self
            .heap
            .new_buffer_at_offset(bytes, ResourceOptions::shared(), n * each)?)
    }
}

/// `bytes` bytes of fresh host pages, read-write, never unmapped: private, as RADV refuses to
/// import shared anonymous memory (`host_memory.rs`'s `import_refused`).
fn host_pages(bytes: usize) -> Result<NonNull<u8>> {
    // SAFETY: a fresh anonymous mapping, read-write, never unmapped.
    let pages = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            -1,
            0,
        )
    };
    if pages == libc::MAP_FAILED {
        return Err(format!("mmap: {}", std::io::Error::last_os_error()).into());
    }
    Ok(NonNull::new(pages.cast::<u8>()).ok_or("a null mapping")?)
}

/// This thread's CPU time so far, in nanoseconds.
fn thread_cpu_nanos() -> u64 {
    // SAFETY: a plain call with an out-parameter of the right type.
    let spec = unsafe {
        let mut spec: libc::timespec = std::mem::zeroed();
        libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut spec);
        spec
    };
    spec.tv_sec as u64 * 1_000_000_000 + spec.tv_nsec as u64
}

/// The group a thread named `comm` is counted in, the process's first thread (`main`) the
/// committer.
fn group_of(comm: &str, main: bool) -> &'static str {
    match comm {
        _ if main => "committer",
        c if c.starts_with("metal2vulkan-sc") => "scheduler",
        c if c.starts_with("metal2vulkan-qu") => "queue-waiters",
        c if c.starts_with("metal2vulkan") || c.starts_with("m2v-") => "other-m2v",
        _ => "driver",
    }
}

/// Nanoseconds of CPU by thread group, each of the process's threads by its name: Linux only,
/// none elsewhere.
fn cpu_by_group() -> Vec<(String, u64)> {
    let mut groups: Vec<(String, u64)> = Vec::new();
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return groups;
    };
    for task in tasks.flatten() {
        let path = task.path();
        let comm = std::fs::read_to_string(path.join("comm")).unwrap_or_default();
        let stat = std::fs::read_to_string(path.join("schedstat")).unwrap_or_default();
        let nanos: u64 = (stat.split_whitespace().next())
            .and_then(|n| n.parse().ok())
            .unwrap_or(0);
        let main = task.file_name().to_str() == Some(&std::process::id().to_string());
        let group = group_of(comm.trim(), main);
        match groups.iter_mut().find(|(n, _)| *n == group) {
            Some((_, total)) => *total += nanos,
            None => groups.push((group.to_owned(), nanos)),
        }
    }
    groups
}

/// The runtime's shapes over one buffer, with two of zeros for `bump_wide`, and two textures and
/// a sampler for `bump_tex`.
struct Ours<'a> {
    device: &'a Device,
    state: ComputePipelineState,
    words: Buffer,
    zeros: [Buffer; 2],
    textures: [Texture; 2],
    sampler: SamplerState,
    readback: Buffer,
    /// The heap each command buffer of the host's time makes its buffers in (`--heap-churn`).
    churn: Option<Heap>,
}

impl Ours<'_> {
    /// One command buffer of `shape`'s dispatches as `name` holds them, committed and waited for:
    /// its GPU time per dispatch in microseconds.
    fn run(&self, name: &str, shape: &Shape) -> Result<f64> {
        let queue = self.device.new_command_queue()?;
        let command_buffer = queue.command_buffer()?;
        let groups = Size::new(u64::from(shape.groups), 1, 1);
        let threads = Size::new(u64::from(shape.threads), 1, 1);
        let encoders = match name {
            "encoders" => shape.dispatches,
            _ => 1,
        };
        for _ in 0..encoders {
            let mut encoder = match name {
                "concurrent" => command_buffer
                    .compute_command_encoder_with_dispatch_type(DispatchType::Concurrent)?,
                _ => command_buffer.compute_command_encoder()?,
            };
            encoder.set_compute_pipeline_state(&self.state)?;
            for n in 0..shape.dispatches / encoders {
                self.bind(&mut encoder, shape.binds, n as usize, &[])?;
                encoder.dispatch_threadgroups(groups, threads)?;
            }
            encoder.end_encoding()?;
        }
        command_buffer.commit()?;
        command_buffer
            .wait_until_completed_for(WAIT)
            .map_err(|_| format!("a command buffer did not complete within {WAIT:?}"))?;
        if command_buffer.status() != CommandBufferStatus::Completed {
            return Err(format!("a command buffer ended {:?}", command_buffer.error()).into());
        }
        let gpu = command_buffer.gpu_end_time()? - command_buffer.gpu_start_time()?;
        Ok(gpu * 1e6 / f64::from(shape.dispatches))
    }

    /// Bind what dispatch `n` of an encoder binds: the words, and `bump_wide`'s read-only
    /// buffers (the next two of `churned`, the command buffer's buffers in its heap, where it has
    /// any) or `bump_tex`'s textures the other way round from dispatch `n - 1`, with their word
    /// of 1.
    fn bind(
        &self,
        encoder: &mut ComputeCommandEncoder,
        binds: Binds,
        n: usize,
        churned: &[Buffer],
    ) -> Result<()> {
        let one = 1u32.to_le_bytes();
        match binds {
            Binds::One => encoder.set_buffer(Some(&self.words), 0, 0)?,
            Binds::Wide | Binds::Pointer => {
                let (b, c) = match churned.len() {
                    0 => (&self.zeros[n % 2], &self.zeros[1 - n % 2]),
                    k => (&churned[(2 * n) % k], &churned[(2 * n + 1) % k]),
                };
                encoder.set_buffer(Some(&self.words), 0, 0)?;
                match binds {
                    Binds::Pointer => encoder.set_bytes(&b.gpu_address().0.to_le_bytes(), 1)?,
                    _ => encoder.set_buffer(Some(b), 0, 1)?,
                }
                encoder.set_buffer(Some(c), 0, 2)?;
                encoder.set_bytes(&one, 3)?;
            }
            Binds::Textures => {
                encoder.set_texture(Some(&self.textures[n % 2]), 0)?;
                encoder.set_texture(Some(&self.textures[1 - n % 2]), 1)?;
                encoder.set_sampler_state(Some(&self.sampler), 0)?;
                encoder.set_buffer(Some(&self.words), 0, 0)?;
                encoder.set_bytes(&one, 1)?;
            }
        }
        Ok(())
    }

    /// One run of the host's time: `command_buffers` command buffers of `shape`'s dispatches as
    /// `name` holds them on `queue`, each committed with `commit_and_submit` and waited for.
    fn run_host(
        &self,
        queue: &CommandQueue,
        name: &str,
        shape: &Shape,
        command_buffers: u32,
        host: &mut HostSample,
    ) -> Result<()> {
        let groups = Size::new(u64::from(shape.groups), 1, 1);
        let threads = Size::new(u64::from(shape.threads), 1, 1);
        let encoders = match name {
            "encoders" => shape.dispatches,
            _ => 1,
        };
        let mut lap = Lap::default();
        for _ in 0..command_buffers {
            let cpu = thread_cpu_nanos();
            let mut at = Instant::now();
            let command_buffer = queue.command_buffer()?;
            lap.lap(&mut at, Call::Make);
            let churned = self.churned(shape)?;
            lap.lap(&mut at, Call::Place);
            for _ in 0..encoders {
                let mut encoder = match name {
                    "concurrent" => command_buffer
                        .compute_command_encoder_with_dispatch_type(DispatchType::Concurrent)?,
                    _ => command_buffer.compute_command_encoder()?,
                };
                lap.lap(&mut at, Call::Open);
                for n in 0..shape.dispatches / encoders {
                    encoder.set_compute_pipeline_state(&self.state)?;
                    lap.lap(&mut at, Call::Pipeline);
                    self.bind(&mut encoder, shape.binds, n as usize, &churned)?;
                    lap.lap(&mut at, Call::Bind);
                    encoder.dispatch_threadgroups(groups, threads)?;
                    lap.lap(&mut at, Call::Dispatch);
                }
                encoder.end_encoding()?;
                lap.lap(&mut at, Call::End);
            }
            command_buffer.commit_and_submit()?;
            lap.lap(&mut at, Call::Commit);
            host.thread_cpu += thread_cpu_nanos() - cpu;
            command_buffer
                .wait_until_completed_for(WAIT)
                .map_err(|_| format!("a command buffer did not complete within {WAIT:?}"))?;
            if command_buffer.status() != CommandBufferStatus::Completed {
                return Err(format!("a command buffer ended {:?}", command_buffer.error()).into());
            }
            let mut at = Instant::now();
            drop(churned);
            lap.lap(&mut at, Call::Free);
        }
        host.add(lap);
        Ok(())
    }

    /// The buffers a command buffer of the host's time makes in its heap (`--heap-churn`): none
    /// without one.
    fn churned(&self, shape: &Shape) -> Result<Vec<Buffer>> {
        let (Some(heap), Some(count)) = (&self.churn, shape.churn) else {
            return Ok(Vec::new());
        };
        let bytes = shape.words() * 4;
        let placed: std::result::Result<Vec<Buffer>, _> = (0..count)
            .map(|_| heap.new_buffer(bytes, ResourceOptions::private()))
            .collect();
        Ok(placed?)
    }

    /// Clear the words, run the host's time of `shape` as `name` holds it, and check the words.
    fn host_shape(&self, name: &str, shape: &Shape, command_buffers: u32) -> Result<()> {
        self.fill(0)?;
        let queue = self.device.new_command_queue()?;
        let mut first = HostSample::default();
        self.run_host(&queue, name, shape, command_buffers, &mut first)?;
        let mut host = HostSample::default();
        let (before, threads) = (self.device.work_counts(), cpu_by_group());
        for _ in 0..shape.runs {
            self.run_host(&queue, name, shape, command_buffers, &mut host)?;
        }
        let (after, threads_after) = (self.device.work_counts(), cpu_by_group());
        if name != "concurrent" {
            let runs = shape.runs as u32 + 1;
            self.check(shape.dispatches * command_buffers * runs, name)?;
        }
        let dispatches = (shape.dispatches * command_buffers) as f64 * shape.runs as f64;
        host.print(name, shape, command_buffers, dispatches);
        let phases = [
            (
                "lowering",
                after.commit_lowering_nanos - before.commit_lowering_nanos,
            ),
            (
                "recording",
                after.commit_recording_nanos - before.commit_recording_nanos,
            ),
            (
                "pipeline_wait",
                after.commit_pipeline_wait_nanos - before.commit_pipeline_wait_nanos,
            ),
            (
                "lock_wait",
                after.commit_lock_wait_nanos - before.commit_lock_wait_nanos,
            ),
            ("pass", after.commit_pass_nanos - before.commit_pass_nanos),
            (
                "submit",
                after.commit_submit_nanos - before.commit_submit_nanos,
            ),
        ];
        let line: Vec<String> = (phases.iter())
            .map(|(phase, nanos)| format!("{phase} {:.0}", *nanos as f64 / dispatches))
            .collect();
        println!("HOST-COMMIT {name} {}", line.join(" "));
        let threads: Vec<(String, u64)> = (threads_after.iter())
            .map(|(group, nanos)| {
                let earlier = (threads.iter())
                    .find(|(g, _)| g == group)
                    .map_or(0, |(_, n)| *n);
                (group.clone(), nanos.saturating_sub(earlier))
            })
            .collect();
        let line: Vec<String> = (threads.iter())
            .map(|(group, nanos)| format!("{group}={:.0}", *nanos as f64 / dispatches))
            .collect();
        if !line.is_empty() {
            println!("HOST-THREADS {name} {}", line.join(" "));
        }
        host.print_per_command_buffer(name, shape, command_buffers, &threads);
        let mut moved = BTreeMap::new();
        add_moved(&mut moved, &before, &after);
        let line: Vec<String> = (moved.iter())
            .map(|(count, sum)| format!("{count}={:.2}", *sum as f64 / dispatches))
            .collect();
        println!("HOST-COUNTS {name} {}", line.join(" "));
        Ok(())
    }

    /// Clear the words, run `shape` as `name` holds it, and check what the chained shapes left.
    fn shape(&self, name: &str, shape: &Shape, device: &str) -> Result<()> {
        self.fill(0)?;
        let mut samples = Samples::default();
        for run in 0..=shape.runs {
            let before = self.device.work_counts();
            let per = self.run(name, shape)?;
            match run {
                0 => add_moved(&mut samples.first, &before, &self.device.work_counts()),
                _ => {
                    samples.per_dispatch.push(per);
                    samples.add_counts(&before, &self.device.work_counts());
                }
            }
        }
        if name != "concurrent" {
            self.check(shape.dispatches * (shape.runs as u32 + 1), name)?;
        }
        samples.print(name, shape, device);
        Ok(())
    }

    fn fill(&self, value: u8) -> Result<()> {
        let queue = self.device.new_command_queue()?;
        let command_buffer = queue.command_buffer()?;
        let mut blit = command_buffer.blit_command_encoder()?;
        blit.fill_buffer(&self.words, 0..self.words.length(), value)?;
        for zeros in &self.zeros {
            blit.fill_buffer(zeros, 0..zeros.length(), 0)?;
        }
        blit.end_encoding()?;
        command_buffer.commit()?;
        command_buffer
            .wait_until_completed_for(WAIT)
            .map_err(|_| "the fill did not complete")?;
        Ok(())
    }

    /// Copy the words back and check each is `want`.
    fn check(&self, want: u32, name: &str) -> Result<()> {
        let queue = self.device.new_command_queue()?;
        let command_buffer = queue.command_buffer()?;
        let mut blit = command_buffer.blit_command_encoder()?;
        blit.copy_from_buffer_to_buffer(&self.words, 0, &self.readback, 0, self.words.length())?;
        blit.end_encoding()?;
        command_buffer.commit()?;
        command_buffer
            .wait_until_completed_for(WAIT)
            .map_err(|_| "the read back did not complete")?;
        let contents = self.readback.contents().ok_or("no contents")?;
        for at in 0..(self.words.length() / 4) as usize {
            // SAFETY: `readback` is `Shared`, as long as `words`, and no command buffer writes it
            // while the CPU reads it.
            let word = unsafe { contents.as_ptr().cast::<u32>().add(at).read_volatile() };
            if word != want {
                return Err(format!("{name}: word {at} is {word}, not {want}").into());
            }
        }
        Ok(())
    }
}

/// The raw shapes' Vulkan objects, on a `VkDevice` of their own over the runtime device's
/// physical device, whose instance they borrow.
struct Raw<'a> {
    _runtime: &'a Device,
    vk: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    commands: Vec<vk::CommandBuffer>,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    descriptors: vk::DescriptorPool,
    set: vk::DescriptorSet,
    module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    buffers: [(vk::Buffer, vk::DeviceMemory); 2],
    bytes: u64,
    mapped: *const u32,
    queries: vk::QueryPool,
    fence: vk::Fence,
    period: f64,
}

/// The least barrier between dispatches, a read after a write.
fn least() -> vk::MemoryBarrier<'static> {
    vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
}

impl<'a> Raw<'a> {
    /// The objects, on a `VkDevice` made with one queue of the runtime device's primary family.
    fn new(device: &'a Device, shape: &Shape) -> Result<Raw<'a>> {
        // SAFETY: the handles are used while the runtime device lives (the result borrows it),
        // and none of the runtime's objects is destroyed.
        let handles = unsafe { device.raw() };
        let family = handles.queue_family;
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        // SAFETY: the instance's own physical device and a valid creation; the device is
        // destroyed in `drop`, before the instance (which the runtime device holds) goes.
        let vk = unsafe {
            handles.instance.create_device(
                handles.physical_device,
                &vk::DeviceCreateInfo::default().queue_create_infos(&queues),
                None,
            )
        }?;
        // SAFETY: the family and index were created.
        let queue = unsafe { vk.get_device_queue(family, 0) };
        let text = SPIRV.replace("{T}", &shape.threads.to_string());
        let bytes = tool(
            "spirv-as",
            &["--target-env", "vulkan1.2", "-o", "-", "-"],
            &text,
        )?;
        let code: Vec<u32> = (bytes.as_chunks::<4>().0.iter())
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let properties =
            // SAFETY: the instance's own physical device.
            unsafe { handles.instance.get_physical_device_properties(handles.physical_device) };
        let memory =
            // SAFETY: as above.
            unsafe { handles.instance.get_physical_device_memory_properties(handles.physical_device) };
        // SAFETY: every info below is valid and lives for its call; every object is of `vk`.
        // Objects made before a failure here are not destroyed: the example ends on it.
        unsafe {
            let make_buffer =
                |bytes: u64, usage, flags| -> Result<(vk::Buffer, vk::DeviceMemory)> {
                    let buffer = vk.create_buffer(
                        &vk::BufferCreateInfo::default().size(bytes).usage(usage),
                        None,
                    )?;
                    let requirements = vk.get_buffer_memory_requirements(buffer);
                    let index = (0..memory.memory_type_count)
                        .find(|&index| {
                            requirements.memory_type_bits & (1 << index) != 0
                                && memory.memory_types[index as usize]
                                    .property_flags
                                    .contains(flags)
                        })
                        .ok_or("no memory type for the buffer")?;
                    let allocation = vk.allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(requirements.size)
                            .memory_type_index(index),
                        None,
                    )?;
                    vk.bind_buffer_memory(buffer, allocation, 0)?;
                    Ok((buffer, allocation))
                };
            let bytes = shape.words() * 4;
            let words = make_buffer(
                bytes,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            let readback = make_buffer(
                bytes,
                vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?;
            let mapped = vk
                .map_memory(readback.1, 0, bytes, vk::MemoryMapFlags::empty())?
                .cast::<u32>();
            let binding = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)];
            let set_layout = vk.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
                None,
            )?;
            let layout = vk.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[set_layout]),
                None,
            )?;
            let module =
                vk.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let pipeline = vk
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(layout)],
                    None,
                )
                .map_err(|(_, error)| error)?[0];
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)];
            let descriptors = vk.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )?;
            let set = vk.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptors)
                    .set_layouts(&[set_layout]),
            )?[0];
            let info = [vk::DescriptorBufferInfo::default()
                .buffer(words.0)
                .range(vk::WHOLE_SIZE)];
            vk.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&info)],
                &[],
            );
            let pool = vk.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                None,
            )?;
            let commands = vk.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(2 * shape.dispatches + 2),
            )?;
            let queries = vk.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2),
                None,
            )?;
            let fence = vk.create_fence(&vk::FenceCreateInfo::default(), None)?;
            Ok(Raw {
                _runtime: device,
                vk,
                queue,
                pool,
                commands,
                layout,
                set_layout,
                descriptors,
                set,
                module,
                pipeline,
                buffers: [words, readback],
                bytes,
                mapped,
                queries,
                fence,
                period: f64::from(properties.limits.timestamp_period),
            })
        }
    }

    /// Submit `commands` and wait for them.
    ///
    /// # Safety
    ///
    /// Each is recorded and executable, and none is pending.
    unsafe fn submit(&self, commands: &[vk::CommandBuffer]) -> Result<()> {
        // SAFETY: the caller's contract; the queue is the caller's alone.
        unsafe {
            self.vk.queue_submit(
                self.queue,
                &[vk::SubmitInfo::default().command_buffers(commands)],
                self.fence,
            )?;
            self.vk
                .wait_for_fences(&[self.fence], true, WAIT.as_nanos() as u64)?;
            self.vk.reset_fences(&[self.fence])?;
        }
        Ok(())
    }

    /// Begin `command` for one submission.
    ///
    /// # Safety
    ///
    /// `command` is of the pool, not pending.
    unsafe fn begin(&self, command: vk::CommandBuffer) -> Result<()> {
        let info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the caller's contract.
        unsafe { self.vk.begin_command_buffer(command, &info) }?;
        Ok(())
    }

    /// Record a dispatch of the kernel into `command`.
    ///
    /// # Safety
    ///
    /// `command` is recording.
    unsafe fn dispatch(&self, command: vk::CommandBuffer, shape: &Shape) {
        // SAFETY: the caller's contract; the objects are live.
        unsafe {
            self.vk
                .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            self.vk.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.set],
                &[],
            );
            self.vk.cmd_dispatch(command, shape.groups, 1, 1);
        }
    }

    /// Record `barrier` from `src` stages to `dst` stages into `command`.
    ///
    /// # Safety
    ///
    /// `command` is recording.
    unsafe fn barrier(
        &self,
        command: vk::CommandBuffer,
        (src, dst, barrier): (
            vk::PipelineStageFlags,
            vk::PipelineStageFlags,
            vk::MemoryBarrier,
        ),
    ) {
        // SAFETY: the caller's contract.
        unsafe {
            self.vk.cmd_pipeline_barrier(
                command,
                src,
                dst,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    /// One run of `name`: its GPU time per dispatch in microseconds.
    fn run(&self, name: &str, shape: &Shape) -> Result<f64> {
        let compute = vk::PipelineStageFlags::COMPUTE_SHADER;
        let barrier = match name {
            "raw-wide" => Some((
                compute,
                compute | vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::SHADER_READ
                            | vk::AccessFlags::SHADER_WRITE
                            | vk::AccessFlags::UNIFORM_READ
                            | vk::AccessFlags::INDIRECT_COMMAND_READ,
                    ),
            )),
            "raw-none" => None,
            _ => Some((compute, compute, least())),
        };
        let split = name == "raw-split";
        // SAFETY: every command buffer is of the pool, none pending (each run waits for its
        // submission), and the pool is reset before they are recorded again.
        unsafe {
            self.vk
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            let first = self.commands[0];
            self.begin(first)?;
            self.vk.cmd_reset_query_pool(first, self.queries, 0, 2);
            self.vk.cmd_write_timestamp(
                first,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.queries,
                0,
            );
            let mut used = vec![first];
            let mut current = first;
            for n in 0..shape.dispatches as usize {
                if split {
                    self.vk.end_command_buffer(current)?;
                    current = self.commands[1 + 2 * n];
                    self.begin(current)?;
                    used.push(current);
                }
                if let (Some(barrier), true) = (barrier, n > 0) {
                    self.barrier(current, barrier);
                }
                if split {
                    self.vk.end_command_buffer(current)?;
                    current = self.commands[2 + 2 * n];
                    self.begin(current)?;
                    used.push(current);
                }
                self.dispatch(current, shape);
            }
            self.vk.cmd_write_timestamp(
                current,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.queries,
                1,
            );
            self.vk.end_command_buffer(current)?;
            self.submit(&used)?;
            let mut stamps = [0u64; 2];
            self.vk.get_query_pool_results(
                self.queries,
                0,
                &mut stamps,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            )?;
            let nanos = stamps[1].wrapping_sub(stamps[0]) as f64 * self.period;
            Ok(nanos / 1e3 / f64::from(shape.dispatches))
        }
    }

    /// Fill the words with `value`, or copy them back.
    fn transfer(&self, fill: Option<u32>) -> Result<()> {
        let command = self.commands[0];
        // SAFETY: as `run`.
        unsafe {
            self.vk
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            self.begin(command)?;
            match fill {
                Some(value) => {
                    self.vk
                        .cmd_fill_buffer(command, self.buffers[0].0, 0, vk::WHOLE_SIZE, value)
                }
                None => {
                    let to_copy = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
                    self.barrier(
                        command,
                        (
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::TRANSFER,
                            to_copy,
                        ),
                    );
                    let region = vk::BufferCopy::default().size(self.bytes);
                    self.vk.cmd_copy_buffer(
                        command,
                        self.buffers[0].0,
                        self.buffers[1].0,
                        &[region],
                    );
                    let to_host = vk::MemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::HOST_READ);
                    self.barrier(
                        command,
                        (
                            vk::PipelineStageFlags::TRANSFER,
                            vk::PipelineStageFlags::HOST,
                            to_host,
                        ),
                    );
                }
            }
            if fill.is_some() {
                let to_compute = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
                self.barrier(
                    command,
                    (
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        to_compute,
                    ),
                );
            }
            self.vk.end_command_buffer(command)?;
            self.submit(&[command])?;
        }
        Ok(())
    }
}

impl Drop for Raw<'_> {
    fn drop(&mut self) {
        // SAFETY: nothing is pending (each submission was waited for), and every object is this
        // example's own.
        unsafe {
            self.vk.destroy_fence(self.fence, None);
            self.vk.destroy_query_pool(self.queries, None);
            self.vk.destroy_command_pool(self.pool, None);
            self.vk.destroy_descriptor_pool(self.descriptors, None);
            self.vk.destroy_pipeline(self.pipeline, None);
            self.vk.destroy_shader_module(self.module, None);
            self.vk.destroy_pipeline_layout(self.layout, None);
            self.vk.destroy_descriptor_set_layout(self.set_layout, None);
            for (buffer, memory) in self.buffers {
                self.vk.destroy_buffer(buffer, None);
                self.vk.free_memory(memory, None);
            }
            self.vk.destroy_device(None);
        }
    }
}

/// An automatic `Private` heap with room for `count` buffers of `bytes` and `kept` of 256, its
/// bytes zeroed by a buffer over all of them that a blit fills and that goes after.
fn zeroed_heap(device: &Device, bytes: u64, count: u32, kept: u32) -> Result<Heap> {
    let need = device.heap_buffer_size_and_align(bytes, ResourceOptions::private());
    let small = device.heap_buffer_size_and_align(256, ResourceOptions::private());
    let size = need.size.next_multiple_of(need.align) * u64::from(count)
        + small.size.next_multiple_of(small.align) * u64::from(kept);
    let descriptor = HeapDescriptor::new(HeapType::Automatic, size, ResourceOptions::private());
    let heap = device.new_heap(&descriptor)?;
    let whole = heap.new_buffer(heap.size(), ResourceOptions::private())?;
    let queue = device.new_command_queue()?;
    let command_buffer = queue.command_buffer()?;
    let mut blit = command_buffer.blit_command_encoder()?;
    blit.fill_buffer(&whole, 0..whole.length(), 0)?;
    blit.end_encoding()?;
    command_buffer.commit()?;
    command_buffer
        .wait_until_completed_for(WAIT)
        .map_err(|_| "the heap's fill did not complete")?;
    Ok(heap)
}

fn main() -> Result<()> {
    let shape = Shape::from_args()?;
    let device = Device::system_default_with(&DeviceOptions::default())?;
    let name = device.name().to_string();
    println!("device: {name}");
    let bytes = shape.words() * 4;
    let placed = match shape.heap {
        true => Some(Placed::new(&device, bytes)?),
        false => None,
    };
    let mut place = 0;
    let mut buffer = || -> Result<Buffer> {
        if let Some(placed) = &placed {
            place += 1;
            return placed.buffer(place - 1, bytes);
        }
        if !shape.pages {
            return Ok(device.new_buffer(bytes, ResourceOptions::private())?);
        }
        // SAFETY: page-aligned read-write memory of `bytes` bytes, never unmapped, which outlives
        // the buffer.
        let buffer = unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::new(host_pages(bytes as usize)?),
                bytes,
                ResourceOptions::shared(),
                None,
            )
        }?;
        Ok(buffer)
    };
    let texture = || -> Result<Texture> {
        let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 16, 16, false);
        descriptor.usage = TextureUsage::SHADER_READ;
        descriptor.resource_options = ResourceOptions::private();
        Ok(device.new_texture(&descriptor)?)
    };
    let churn = match shape.churn {
        None => None,
        Some(count) => Some(zeroed_heap(&device, bytes, count, shape.kept)?),
    };
    // The heap's other buffers, listed among the device's live ones through the runs.
    let kept: std::result::Result<Vec<Buffer>, _> = (churn.iter())
        .flat_map(|heap| (0..shape.kept).map(|_| heap.new_buffer(256, ResourceOptions::private())))
        .collect();
    let _kept = kept?;
    // The program's other buffers, listed among the device's live ones through the runs.
    let live: std::result::Result<Vec<Buffer>, _> = (0..shape.live)
        .map(|_| device.new_buffer(256, ResourceOptions::private()))
        .collect();
    let _live = live?;
    let ours = Ours {
        device: &device,
        state: state(&device, shape.binds)?,
        words: buffer()?,
        zeros: [buffer()?, buffer()?],
        textures: [texture()?, texture()?],
        sampler: device.new_sampler_state(&SamplerDescriptor::new())?,
        readback: device.new_buffer(bytes, ResourceOptions::shared())?,
        churn,
    };
    for shape_name in ["encoders", "serial", "concurrent"] {
        match (shape.runs(shape_name), shape.host) {
            (false, _) => {}
            (true, Some(command_buffers)) => {
                ours.host_shape(shape_name, &shape, command_buffers)?
            }
            (true, None) => ours.shape(shape_name, &shape, &name)?,
        }
    }
    if shape.host.is_some() {
        return Ok(());
    }
    let raw = Raw::new(&device, &shape)?;
    for shape_name in ["raw-barrier", "raw-wide", "raw-split", "raw-none"] {
        if !shape.runs(shape_name) {
            continue;
        }
        raw.transfer(Some(0))?;
        let mut samples = Samples::default();
        for run in 0..=shape.runs {
            let per = raw.run(shape_name, &shape)?;
            if run > 0 {
                samples.per_dispatch.push(per);
            }
        }
        if shape_name != "raw-none" {
            raw.transfer(None)?;
            let want = shape.dispatches * (shape.runs as u32 + 1);
            for at in 0..shape.words() as usize {
                // SAFETY: the mapped read-back, as long as the words, written by the copy the
                // transfer waited for.
                let word = unsafe { raw.mapped.add(at).read_volatile() };
                if word != want {
                    return Err(format!("{shape_name}: word {at} is {word}, not {want}").into());
                }
            }
        }
        samples.print(shape_name, &shape, &name);
    }
    Ok(())
}
