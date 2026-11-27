//! Variants of the OpenCL C barrier and fence tests (`opencl_kernels`'s out-of-line barrier and
//! fence, `opencl_mode`'s work-item kernel's barrier) whose data crosses waves: threadgroups of 128
//! threads, more than one wave on every device, in which thread `t` of a threadgroup reads what
//! thread `t ^ 64` wrote, through threadgroup memory or device memory, synchronized by a barrier or
//! a fence of each flag alone and of both. Each kernel calls OpenCL C's `barrier(flags)` as Apple's
//! OpenCL compiler passes it on, `tail call fastcc void @_Z7barrierj(i32 noundef 1)`, to an
//! internal wrapper calling `air.wg.barrier` with the flags as they are, and `mem_fence(flags)` to
//! one that computes `air.atomic.fence`'s flags through `llvm.bitreverse.i2`:
//!
//! - **a barrier** of OpenCL's `CLK_LOCAL_MEM_FENCE` (1) alone where threadgroup memory carries the
//!   data, `CLK_GLOBAL_MEM_FENCE` (2) alone where device memory carries it, and both (3) for each;
//!   in a plain pipeline through the two-operand `air.wg.barrier(flags, 1)` (the out-of-line
//!   barrier's form), and in a pipeline of OpenCL mode through that form and through the
//!   three-operand `air.wg.barrier(flags, 5, 1)` (the work-item kernel's);
//! - **a fence** of each of the same flags in a plain pipeline: the data crosses through the fence,
//!   an execution barrier of no flags (`barrier(0)`, which no encoding reads as a memory flag) and
//!   the fence again, since a fence orders only its own thread's accesses.
//!
//! Every thread runs four rounds, as a tiled kernel's loop does: it writes its slot, synchronizes,
//! reads its partner's slot into a running value, and synchronizes again before the next round
//! overwrites the slot. The answer is computed here, exactly: thread `g` stores
//! `((w0 * 31 + w1) * 31 + w2) * 31 + w3` (wrapping), `wk = 4 * (g ^ 64) + 1000 + k`.
//!
//! Under a device cfg, on every device of the cfg's kind. The modules are LLVM text written here;
//! `llvm-as` assembles them, and without it the tests skip visibly.

use crate::tests::command::on_device::{each_device, finished};
use crate::tests::opencl_mode::{buffer, kernel_list, words, words_buffer};
use crate::{
    Buffer, ComputePipelineDescriptor, ComputePipelineState, Device, DispatchType,
    FunctionDescriptor, Library, StageSource,
};
use metal2vulkan_devtools::{llvm_assemble_text, AssembleError};

/// The target and data layout of every module.
const TARGET: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\n";

/// A plain pipeline's module head: the out-of-line barrier test's triple, versions and work-item
/// function.
const PLAIN: &str = "target triple = \"air64_v28-apple-macosx26.0.0\"\n\
declare i32 @air.get_global_id.i32(i32) #9\n\
attributes #9 = { nounwind readnone willreturn }\n\
!air.version = !{!90}\n\
!air.language_version = !{!91}\n\
!90 = !{i32 2, i32 8, i32 0}\n\
!91 = !{!\"Metal\", i32 3, i32 2, i32 0}\n";

/// A pipeline of OpenCL mode's module head: the work-item kernel's triple, versions and
/// `get_global_id` wrapper.
const OPENCL: &str = "target triple = \"air64_v29-apple-macosx27.0.0\"\n\
define internal fastcc i64 @_Z13get_global_idj(i32 %0) #6 {\n\
  %2 = tail call i32 @air.get_global_id.i32(i32 %0) #9\n\
  %3 = zext i32 %2 to i64\n\
  ret i64 %3\n}\n\
declare i32 @air.get_global_id.i32(i32) #9\n\
attributes #6 = { nounwind optsize }\n\
attributes #9 = { nounwind readnone willreturn }\n\
!air.version = !{!90}\n\
!air.language_version = !{!91}\n\
!90 = !{i32 2, i32 9, i32 0}\n\
!91 = !{!\"Metal\", i32 1, i32 1, i32 0}\n";

/// `barrier(flags)` passed on to the two-operand `air.wg.barrier`, out of line.
const BARRIER_TWO: &str = "define internal fastcc void @_Z7barrierj(i32 %0) #2 {\n\
  tail call void @air.wg.barrier(i32 %0, i32 1) #4\n\
  ret void\n}\n\
declare void @air.wg.barrier(i32, i32) #4\n";

/// `barrier(flags)` passed on to the three-operand `air.wg.barrier`, as the work-item kernel's
/// wrapper.
const BARRIER_THREE: &str = "define internal fastcc void @_Z7barrierj(i32 %0) #2 {\n\
  call void @air.wg.barrier(i32 %0, i32 5, i32 1) #4\n\
  ret void\n}\n\
declare void @air.wg.barrier(i32, i32, i32) #4\n";

/// `mem_fence(flags)`: `air.atomic.fence` of the flags' two bits reversed, out of line.
const FENCE: &str = "define internal fastcc void @_Z9mem_fencej(i32 %0) #2 {\n\
  %2 = trunc i32 %0 to i2\n\
  %3 = tail call i2 @llvm.bitreverse.i2(i2 %2)\n\
  %4 = zext i2 %3 to i32\n\
  tail call void @air.atomic.fence(i32 %4, i32 3, i32 1) #5\n\
  ret void\n}\n\
declare void @air.atomic.fence(i32, i32, i32) #5\n\
declare i2 @llvm.bitreverse.i2(i2) #9\n";

/// The attributes of the entry, the kernel, the wrappers and the intrinsics.
const ATTRIBUTES: &str = "attributes #0 = { nounwind }\n\
attributes #1 = { convergent nounwind }\n\
attributes #2 = { convergent noinline nounwind }\n\
attributes #4 = { convergent nounwind willreturn }\n\
attributes #5 = { nounwind willreturn }\n";

/// The threads of a threadgroup; thread `t` reads thread `t ^ 64`'s slot.
const GROUP: u32 = 128;
/// The threads of the grid: 256 threadgroups.
const THREADS: u32 = 256 * GROUP;
/// The rounds each thread runs.
const ROUNDS: u32 = 4;

/// Where a variant's kernel is made.
#[derive(Clone, Copy)]
enum Mode {
    /// A plain pipeline, the kernel its own entry.
    Plain,
    /// A pipeline of OpenCL mode, `__entry_xwave` calling the kernel.
    OpenCl,
}

/// What synchronizes a round's write and read.
#[derive(Clone, Copy)]
enum Sync {
    /// `barrier(flags)` through the two-operand `air.wg.barrier`.
    Barrier,
    /// `barrier(flags)` through the three-operand `air.wg.barrier`.
    ThreeOperandBarrier,
    /// `mem_fence(flags)`, `barrier(0)`, `mem_fence(flags)`.
    Fence,
}

/// What carries the data between the threads.
#[derive(Clone, Copy)]
enum Carrier {
    /// Threadgroup memory: a static `__local` array in a plain pipeline, a `__local` argument
    /// (threadgroup memory at index 0) in one of OpenCL mode.
    Threadgroup,
    /// Device memory: a `__global` buffer.
    Device,
}

/// One variant: its pipeline, its synchronization, the OpenCL flags it passes and its carrier.
#[derive(Clone, Copy)]
struct Variant {
    mode: Mode,
    sync: Sync,
    flags: u32,
    carrier: Carrier,
}

/// The calls that synchronize a round in `v`.
fn sync_calls(v: Variant) -> String {
    let flags = v.flags;
    match v.sync {
        Sync::Barrier | Sync::ThreeOperandBarrier => {
            format!("  tail call fastcc void @_Z7barrierj(i32 noundef {flags})\n")
        }
        Sync::Fence => format!(
            "  tail call fastcc void @_Z9mem_fencej(i32 noundef {flags})\n\
  tail call fastcc void @_Z7barrierj(i32 noundef 0)\n\
  tail call fastcc void @_Z9mem_fencej(i32 noundef {flags})\n"
        ),
    }
}

/// The module of `v`, its kernel `xwave`: thread `g` of the grid runs [`ROUNDS`] rounds over its
/// slot and its partner's (`g ^ 64`, of its own threadgroup), and stores its running value to
/// `out[g]`.
fn module(v: Variant) -> String {
    let (space, own, peer) = match (v.mode, v.carrier) {
        (Mode::Plain, Carrier::Threadgroup) => (
            3,
            "getelementptr inbounds [128 x i32], ptr addrspace(3) @tmp, i64 0, i64 %l",
            "getelementptr inbounds [128 x i32], ptr addrspace(3) @tmp, i64 0, i64 %pl",
        ),
        (Mode::OpenCl, Carrier::Threadgroup) => (
            3,
            "getelementptr inbounds i32, ptr addrspace(3) %c, i64 %l",
            "getelementptr inbounds i32, ptr addrspace(3) %c, i64 %pl",
        ),
        (_, Carrier::Device) => (
            1,
            "getelementptr inbounds i32, ptr addrspace(1) %c, i64 %g",
            "getelementptr inbounds i32, ptr addrspace(1) %c, i64 %pg",
        ),
    };
    let mut body = String::new();
    body.push_str(&format!(
        "  %l = and i64 %g, 127\n\
  %pl = xor i64 %l, 64\n\
  %pg = xor i64 %g, 64\n\
  %own = {own}\n\
  %peer = {peer}\n\
  %b = shl i32 %id, 2\n\
  %b0 = add i32 %b, 1000\n"
    ));
    let sync = sync_calls(v);
    for k in 0..ROUNDS {
        let before = match k {
            0 => "0".to_string(),
            _ => format!("%a{}", k - 1),
        };
        body.push_str(&format!(
            "  %v{k} = add i32 %b0, {k}\n\
  store i32 %v{k}, ptr addrspace({space}) %own, align 4\n\
{sync}\
  %w{k} = load i32, ptr addrspace({space}) %peer, align 4\n\
  %m{k} = mul i32 {before}, 31\n\
  %a{k} = add i32 %m{k}, %w{k}\n\
{sync}"
        ));
    }
    let last = ROUNDS - 1;
    body.push_str(&format!(
        "  %po = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g\n\
  store i32 %a{last}, ptr addrspace(1) %po, align 4\n\
  ret void\n}}\n"
    ));
    let wrappers = match v.sync {
        Sync::Barrier => BARRIER_TWO.to_string(),
        Sync::ThreeOperandBarrier => BARRIER_THREE.to_string(),
        Sync::Fence => format!("{BARRIER_TWO}{FENCE}"),
    };
    let out = buffer(0, 0, "read_write", 1, "uint", "out");
    match v.mode {
        Mode::Plain => {
            let (tmp, params, args) = match v.carrier {
                Carrier::Threadgroup => (
                    "@tmp = internal unnamed_addr addrspace(3) global [128 x i32] undef, align 4\n",
                    "ptr addrspace(1) nocapture %out",
                    vec![out],
                ),
                Carrier::Device => (
                    "",
                    "ptr addrspace(1) nocapture %out, ptr addrspace(1) nocapture %c",
                    vec![out, buffer(1, 1, "read_write", 1, "uint", "c")],
                ),
            };
            format!(
                "{TARGET}{PLAIN}{tmp}define void @xwave({params}) #1 {{\n\
  %id = tail call i32 @air.get_global_id.i32(i32 0)\n\
  %g = zext i32 %id to i64\n\
{body}{wrappers}{ATTRIBUTES}{}",
                kernel_list("xwave", &args)
            )
        }
        Mode::OpenCl => {
            let c = match v.carrier {
                Carrier::Threadgroup => buffer(1, 0, "read_write", 3, "uint", "c"),
                Carrier::Device => buffer(1, 1, "read_write", 1, "uint", "c"),
            };
            format!(
                "{TARGET}{OPENCL}define void @__entry_xwave(ptr addrspace(1) %0, ptr addrspace({space}) %1) #0 {{\n\
entry:\n\
  tail call fastcc void @xwave(ptr addrspace(1) %0, ptr addrspace({space}) %1)\n\
  ret void\n}}\n\
define internal fastcc void @xwave(ptr addrspace(1) %out, ptr addrspace({space}) %c) #1 {{\n\
entry:\n\
  %g = tail call fastcc i64 @_Z13get_global_idj(i32 0)\n\
  %id = trunc i64 %g to i32\n\
{body}{wrappers}{ATTRIBUTES}{}",
                kernel_list("__entry_xwave", &[out, c])
            )
        }
    }
}

/// What thread `g` stores: its running value over its partner's [`ROUNDS`] writes.
fn want(g: u32) -> u32 {
    let partner = g ^ 64;
    (0..ROUNDS).fold(0u32, |a, k| {
        a.wrapping_mul(31)
            .wrapping_add(4u32.wrapping_mul(partner).wrapping_add(1000 + k))
    })
}

/// The state of `v`'s kernel in `library`.
fn state(device: &Device, library: &Library, v: Variant) -> ComputePipelineState {
    let entry = match v.mode {
        Mode::Plain => "xwave",
        Mode::OpenCl => "__entry_xwave",
    };
    let function = library.function(entry).expect("the kernel");
    let mut descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(FunctionDescriptor::new(function)));
    descriptor.open_cl_mode_enabled = matches!(v.mode, Mode::OpenCl);
    (device.new_compute_pipeline_state(&descriptor, crate::tests::no_constexpr_samplers))
        .unwrap_or_else(|error| panic!("{entry}: {error}"))
}

/// `state` of `v` dispatched on `device` (named `name`) over [`THREADS`] threads in threadgroups of
/// [`GROUP`], `out` its output and `c` its device carrier where it has one: in a pipeline of
/// OpenCL mode the three hidden buffers bound as Metal's OpenCL dispatcher binds them (the first's
/// bytes at index 0, no offset and one dimension), the kernel's own from 3, and its `__local`
/// argument's bytes at threadgroup index 0; committed and finished without an error.
fn run(
    name: &str,
    device: &Device,
    state: &ComputePipelineState,
    v: Variant,
    out: &Buffer,
    c: &Buffer,
) {
    let queue = device.new_command_queue(1).expect("a queue");
    let command_buffer = queue.command_buffer().expect("a command buffer");
    let mut encoder =
        (command_buffer.compute_command_encoder(DispatchType::Serial)).expect("an encoder");
    encoder
        .set_compute_pipeline_state(state)
        .expect("the state");
    let hidden = match v.mode {
        Mode::Plain => Vec::new(),
        Mode::OpenCl => vec![
            words_buffer(device, &[0; 16]),
            words_buffer(device, &[0; 16]),
        ],
    };
    let first = match v.mode {
        Mode::Plain => 0,
        Mode::OpenCl => {
            let words: Vec<u8> = [0u32, 0, 0, 1]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect();
            encoder
                .set_bytes(&words, 0)
                .expect("the first hidden buffer");
            for (index, b) in (1u32..).zip(&hidden) {
                (encoder.set_buffer(Some(b), 0, index)).expect("a hidden buffer");
            }
            3
        }
    };
    encoder.set_buffer(Some(out), 0, first).expect("out");
    match v.carrier {
        Carrier::Device => encoder.set_buffer(Some(c), 0, first + 1).expect("c"),
        Carrier::Threadgroup if matches!(v.mode, Mode::OpenCl) => {
            (encoder.set_threadgroup_memory_length(4 * GROUP, 0)).expect("the __local bytes")
        }
        Carrier::Threadgroup => {}
    }
    (encoder.dispatch_threads([THREADS, 1, 1], [GROUP, 1, 1]))
        .unwrap_or_else(|error| panic!("{name}: the dispatch: {error}"));
    encoder.end_encoding();
    command_buffer.commit().expect("a commit");
    finished(name, &command_buffer);
    assert_eq!(command_buffer.error(), None, "{name}");
}

/// `v` on every device of the cfg's kind: every thread's value is [`want`]'s. A device whose
/// value differs anywhere fails with the count of threads that differ and the first few.
fn check(test: &str, v: Variant) {
    let library = match llvm_assemble_text(&module(v)) {
        Ok(air) => Library::from_air(&air).expect("the library"),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            return;
        }
        Err(error) => panic!("{test}: the module: {error:?}"),
    };
    each_device(test, |name, _, device| {
        let state = state(&device, &library, v);
        let sentinel = vec![0xeeee_eeee; THREADS as usize];
        let out = words_buffer(&device, &sentinel);
        let c = words_buffer(&device, &sentinel);
        run(name, &device, &state, v, &out, &c);
        let wrong: Vec<(u32, u32, u32)> = (0u32..)
            .zip(words(&out))
            .filter(|&(g, got)| got != want(g))
            .map(|(g, got)| (g, got, want(g)))
            .collect();
        assert!(
            wrong.is_empty(),
            "{name}: {} of {THREADS} threads differ; the first (thread, got, want): {:x?}",
            wrong.len(),
            &wrong[..wrong.len().min(8)]
        );
        println!("{name}: {THREADS} threads in groups of {GROUP} read their partners' writes");
    });
}

/// Each variant's test.
macro_rules! variants {
    ($($name:ident: $mode:ident, $sync:ident, $flags:literal, $carrier:ident;)*) => {
        $(
            #[test]
            fn $name() {
                let v = Variant {
                    mode: Mode::$mode,
                    sync: Sync::$sync,
                    flags: $flags,
                    carrier: Carrier::$carrier,
                };
                check(stringify!($name), v);
            }
        )*
    };
}

variants! {
    a_barrier_of_flags_1_orders_threadgroup_data_across_waves: Plain, Barrier, 1, Threadgroup;
    a_barrier_of_flags_2_orders_device_data_across_waves: Plain, Barrier, 2, Device;
    a_barrier_of_flags_3_orders_threadgroup_data_across_waves: Plain, Barrier, 3, Threadgroup;
    a_barrier_of_flags_3_orders_device_data_across_waves: Plain, Barrier, 3, Device;
    a_fence_of_flags_1_orders_threadgroup_data_across_waves: Plain, Fence, 1, Threadgroup;
    a_fence_of_flags_2_orders_device_data_across_waves: Plain, Fence, 2, Device;
    a_fence_of_flags_3_orders_threadgroup_data_across_waves: Plain, Fence, 3, Threadgroup;
    a_fence_of_flags_3_orders_device_data_across_waves: Plain, Fence, 3, Device;
    opencl_mode_barrier_of_flags_1_orders_threadgroup_data_across_waves: OpenCl, Barrier, 1, Threadgroup;
    opencl_mode_barrier_of_flags_2_orders_device_data_across_waves: OpenCl, Barrier, 2, Device;
    opencl_mode_barrier_of_flags_3_orders_threadgroup_data_across_waves: OpenCl, Barrier, 3, Threadgroup;
    opencl_mode_barrier_of_flags_3_orders_device_data_across_waves: OpenCl, Barrier, 3, Device;
    opencl_mode_three_operand_barrier_of_flags_1_orders_threadgroup_data_across_waves: OpenCl, ThreeOperandBarrier, 1, Threadgroup;
    opencl_mode_three_operand_barrier_of_flags_2_orders_device_data_across_waves: OpenCl, ThreeOperandBarrier, 2, Device;
    opencl_mode_three_operand_barrier_of_flags_3_orders_threadgroup_data_across_waves: OpenCl, ThreeOperandBarrier, 3, Threadgroup;
    opencl_mode_three_operand_barrier_of_flags_3_orders_device_data_across_waves: OpenCl, ThreeOperandBarrier, 3, Device;
}
