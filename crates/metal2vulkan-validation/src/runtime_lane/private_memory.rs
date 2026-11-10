//! How much Function storage a thread can hold on the device the cfg selects (wip-queue 0g step B):
//! the measurement the per-driver `VkLimit::FunctionStorageBytesPerThread` comes from. A recursive
//! kernel's call stack is Function storage, which a driver backs with scratch memory (lavapipe: the
//! worker thread's stack), and no Vulkan property states how much a thread holds. Past it a driver
//! can answer wrongly with no error: the Radeon 780M under RADV at 256 KiB a thread over 4096
//! threads.
//!
//! Each test compiles one kernel through the runtime whose thread holds a private array of `words`
//! 32-bit words (`alloca [words x i32]`). It writes the first word of each 64-byte line (`line ^
//! gid`), reads the lines back in an order no two threads share (line `(7919 j + gid) mod lines`
//! at step `j`), and writes the weighted sum `sum (j + 1) * word` to its output word: every line
//! of the array is touched, so threads whose storage overlaps answer wrongly. It makes the state,
//! runs it over one thread (the run that creates its `VkPipeline`), then 64, 4096 and 262144
//! threads (enough to fill every venue's GPU), each only after the one before answered correctly,
//! and prints one line per step: what it answered and its wall time; a step that did not answer
//! correctly ends the size with `STOP`. Nothing is asserted but the validation layer's silence: a
//! refusal or a wrong answer is the measurement. Each size is its own test, so a size that ends
//! the process (a worker thread's stack overflow on lavapipe) ends only that test. The measurement
//! runs take the sizes in ascending order, each alone and bounded; they stop at the first size with
//! a `STOP` or no end, so no larger size runs after a timeout or a lost device.

use super::at_plan;
use super::host_memory::on_device;
use super::host_memory::on_device_with;
use super::read_buffer;
use crate::air_library::AirLibrary;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble_text;
use metal2vulkan_runtime::{ComputePipelineState, Device, PipelineOptions, ResourceOptions, Size};
use std::time::{Duration, Instant};

/// The longest wait for one run's command buffer (the runs bound a whole size at 60 s).
const WAIT: Duration = Duration::from_secs(50);

/// The read order's stride: prime, so steps `0..lines` visit every line of the sizes measured.
const STRIDE: u32 = 7919;

/// The words of a 64-byte line.
const LINE: u32 = 16;

/// The kernel of a private array of `words` words (a multiple of [`LINE`]).
fn kernel(words: u32) -> String {
    let lines = words / LINE;
    format!(
        r#"; Owned synthetic kernel: a private array of {words} words a thread.
source_filename = "private_memory.metal"

define void @private_words(ptr addrspace(1) %out, i32 %gid) {{
entry:
  %a = alloca [{words} x i32], align 4
  br label %fill
fill:
  %i = phi i32 [ 0, %entry ], [ %i1, %fill ]
  %w = mul i32 %i, {LINE}
  %p = getelementptr inbounds [{words} x i32], ptr %a, i32 0, i32 %w
  %v = xor i32 %i, %gid
  store i32 %v, ptr %p, align 4
  %i1 = add nuw i32 %i, 1
  %filled = icmp eq i32 %i1, {lines}
  br i1 %filled, label %read, label %fill
read:
  %j = phi i32 [ 0, %fill ], [ %j1, %read ]
  %s = phi i32 [ 0, %fill ], [ %s1, %read ]
  %m = mul i32 %j, {STRIDE}
  %k = add i32 %m, %gid
  %line = urem i32 %k, {lines}
  %idx = mul i32 %line, {LINE}
  %q = getelementptr inbounds [{words} x i32], ptr %a, i32 0, i32 %idx
  %x = load i32, ptr %q, align 4
  %j1 = add nuw i32 %j, 1
  %y = mul i32 %x, %j1
  %s1 = add i32 %s, %y
  %done = icmp eq i32 %j1, {lines}
  br i1 %done, label %write, label %read
write:
  %g = zext i32 %gid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 %s1, ptr addrspace(1) %o, align 4
  ret void
}}

!air.kernel = !{{!0}}
!0 = !{{ptr @private_words, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
"#
    )
}

/// What thread `gid`'s output word holds.
fn expected(words: u32, gid: u32) -> u32 {
    let lines = words / LINE;
    (0..lines).fold(0u32, |s, j| {
        let line = (j.wrapping_mul(STRIDE).wrapping_add(gid)) % lines;
        s.wrapping_add((line ^ gid).wrapping_mul(j + 1))
    })
}

/// The pipeline state of the kernel of `words` words on `device`, or what refused it.
fn state(device: &Device, words: u32) -> Result<ComputePipelineState, String> {
    state_of(device, &kernel(words), "private_words")
}

/// The pipeline state of kernel `entry` of the module `text` on `device`, or what refused it.
fn state_of(device: &Device, text: &str, entry: &str) -> Result<ComputePipelineState, String> {
    let library = device
        .new_library_with_air(&llvm_assemble_text(text).unwrap())
        .map_err(|e| format!("library refused: {e}"))?;
    let function = library.new_function_with_name(entry).unwrap();
    device
        .new_compute_pipeline_state(
            &ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor())),
            &PipelineOptions::default(),
        )
        .map_err(|e| format!("state refused: {e}"))
}

/// Run `state` over `threads` threads: whether every checked thread's output word is what
/// `expected` says of its id, and what the run answered. The first run of a state also creates
/// its `VkPipeline`, so a refused creation is a refused dispatch or commit.
fn run(
    device: &Device,
    state: &ComputePipelineState,
    threads: u32,
    expected: &dyn Fn(u32) -> u32,
) -> (bool, String) {
    let out = device
        .new_buffer(u64::from(threads) * 4, ResourceOptions::shared())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    let group = threads.min(64);
    if let Err(e) = encoder.dispatch_threads(
        Size::new(threads.into(), 1, 1),
        Size::new(group.into(), 1, 1),
    ) {
        return (false, format!("dispatch refused: {e}"));
    }
    encoder.end_encoding().unwrap();
    if let Err(e) = command_buffer.commit() {
        return (false, format!("commit refused: {e}"));
    }
    if command_buffer.wait_until_completed_for(WAIT).is_err() {
        return (false, format!("no completion within {} s", WAIT.as_secs()));
    }
    if let Some(error) = command_buffer.error() {
        return (false, format!("command buffer error: {error:?}"));
    }
    let bytes = read_buffer(&out, 0, out.length()).unwrap();
    // Every thread of a 64-thread run, 64 spread over a wider one: the host's sums are O(words).
    let step = (threads / 64).max(1) as usize;
    let checked: Vec<u32> = (0..threads).step_by(step).collect();
    let wrong = checked
        .iter()
        .filter(|&&gid| {
            let at = gid as usize * 4;
            u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) != expected(gid)
        })
        .count();
    if wrong == 0 {
        (true, format!("correct ({} threads checked)", checked.len()))
    } else {
        (
            false,
            format!("WRONG in {wrong} of {} threads checked", checked.len()),
        )
    }
}

/// Measure `words` words a thread: the state's creation, a one-thread first run (which creates
/// the `VkPipeline`), then 64, 4096 and 262144 threads, each run only after the one before it
/// answered correctly. Each line names its step and wall time; the last line of a size that
/// stopped early says `STOP`.
fn measure(words: u32) {
    on_device(|device| {
        let line = |step: &str, answer: &str, start: Instant| {
            println!(
                "private-memory on {}: {} bytes a thread, {step}: {answer} ({} ms)",
                device.name(),
                u64::from(words) * 4,
                start.elapsed().as_millis()
            );
        };
        let start = Instant::now();
        let state = match state(device, words) {
            Ok(state) => {
                line("state", "made", start);
                state
            }
            Err(refused) => return line("state", &format!("STOP {refused}"), start),
        };
        for threads in [1, 64, 4096, 262_144] {
            let start = Instant::now();
            let (correct, answer) = run(device, &state, threads, &|gid| expected(words, gid));
            let step = format!("{threads} threads");
            if !correct {
                return line(&step, &format!("STOP {answer}"), start);
            }
            line(&step, &answer, start);
        }
    });
}

/// One ignored test per size, named by its bytes a thread.
macro_rules! sizes {
    ($($name:ident: $kib:expr;)*) => {$(
        #[test]
        #[ignore = "a measurement (a thread's Function storage): a run takes each size alone"]
        fn $name() {
            measure($kib * 256);
        }
    )*};
}

sizes! {
    kib_004: 4;
    kib_008: 8;
    kib_016: 16;
    kib_032: 32;
    kib_048: 48;
    kib_064: 64;
    kib_096: 96;
    kib_128: 128;
    kib_192: 192;
    kib_256: 256;
    kib_512: 512;
    mib_1: 1024;
}

/// The objects of [`held_kernel`] reached only at constant offsets, the words of each, and the
/// offsets (in words, [`HELD_POINTS`] of them) at which each is written and read back.
const HELD_OBJECTS: u32 = 32;
const HELD_WORDS: u32 = 1024;
const HELD_POINTS: u32 = 4;
const HELD_AT: [u32; HELD_POINTS as usize] = [0, 341, 682, 1023];

/// The words of [`held_kernel`]'s array indexed at run time: 16 KiB, under every measured
/// device's Function storage.
const INDEXED_WORDS: u32 = 4096;

/// The weight of object `k`'s `j`th constant offset.
fn held_weight(k: u32, j: u32) -> u32 {
    4 * k + j + 1
}

/// A kernel holding [`HELD_OBJECTS`] private arrays of [`HELD_WORDS`] words (128 KiB, four times
/// the 32 KiB of the Intel venue and more than twice lavapipe's 48 KiB) that it reaches only at
/// the constant offsets [`HELD_AT`], and one array of [`INDEXED_WORDS`] words it indexes at run
/// time as [`kernel`] does. It writes word `j` of object `k` as `(gid * w) ^ k` (`w` its
/// [`held_weight`]), fills and reads the indexed array in [`kernel`]'s order, then reads the
/// objects back in reverse, and writes the indexed array's sum plus `sum w * word` (the objects
/// count nothing toward the device's Function storage, the indexed array does).
fn held_kernel() -> String {
    let lines = INDEXED_WORDS / LINE;
    let mut allocas = String::new();
    let mut writes = String::new();
    let mut reads = String::new();
    let mut last = String::from("%s1");
    for k in 0..HELD_OBJECTS {
        allocas += &format!("  %o{k} = alloca [{HELD_WORDS} x i32], align 4\n");
        for (j, at) in (0..HELD_POINTS).zip(HELD_AT) {
            let w = held_weight(k, j);
            writes += &format!(
                "  %wp{k}_{j} = getelementptr inbounds [{HELD_WORDS} x i32], ptr %o{k}, i32 0, i32 {at}\n\
                 \x20 %wm{k}_{j} = mul i32 %gid, {w}\n\
                 \x20 %wv{k}_{j} = xor i32 %wm{k}_{j}, {k}\n\
                 \x20 store i32 %wv{k}_{j}, ptr %wp{k}_{j}, align 4\n"
            );
        }
    }
    for k in (0..HELD_OBJECTS).rev() {
        for (j, at) in (0..HELD_POINTS).zip(HELD_AT).rev() {
            let w = held_weight(k, j);
            reads += &format!(
                "  %rp{k}_{j} = getelementptr inbounds [{HELD_WORDS} x i32], ptr %o{k}, i32 0, i32 {at}\n\
                 \x20 %rv{k}_{j} = load i32, ptr %rp{k}_{j}, align 4\n\
                 \x20 %rm{k}_{j} = mul i32 %rv{k}_{j}, {w}\n\
                 \x20 %ra{k}_{j} = add i32 {last}, %rm{k}_{j}\n"
            );
            last = format!("%ra{k}_{j}");
        }
    }
    format!(
        r#"; Owned synthetic kernel: private arrays reached only at constant offsets, and one
; indexed at run time.
source_filename = "private_memory_held.metal"

define void @held_words(ptr addrspace(1) %out, i32 %gid) {{
entry:
{allocas}  %a = alloca [{INDEXED_WORDS} x i32], align 4
{writes}  br label %fill
fill:
  %i = phi i32 [ 0, %entry ], [ %i1, %fill ]
  %w = mul i32 %i, {LINE}
  %p = getelementptr inbounds [{INDEXED_WORDS} x i32], ptr %a, i32 0, i32 %w
  %v = xor i32 %i, %gid
  store i32 %v, ptr %p, align 4
  %i1 = add nuw i32 %i, 1
  %filled = icmp eq i32 %i1, {lines}
  br i1 %filled, label %read, label %fill
read:
  %j = phi i32 [ 0, %fill ], [ %j1, %read ]
  %s = phi i32 [ 0, %fill ], [ %s1, %read ]
  %m = mul i32 %j, {STRIDE}
  %k = add i32 %m, %gid
  %line = urem i32 %k, {lines}
  %idx = mul i32 %line, {LINE}
  %q = getelementptr inbounds [{INDEXED_WORDS} x i32], ptr %a, i32 0, i32 %idx
  %x = load i32, ptr %q, align 4
  %j1 = add nuw i32 %j, 1
  %y = mul i32 %x, %j1
  %s1 = add i32 %s, %y
  %done = icmp eq i32 %j1, {lines}
  br i1 %done, label %write, label %read
write:
{reads}  %g = zext i32 %gid to i64
  %o = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %g
  store i32 {last}, ptr addrspace(1) %o, align 4
  ret void
}}

!air.kernel = !{{!0}}
!0 = !{{ptr @held_words, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4}}
!3 = !{{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"uint", !"air.arg_name", !"out"}}
!4 = !{{i32 1, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"gid"}}
"#
    )
}

/// What thread `gid`'s output word of [`held_kernel`] holds.
fn held_expected(gid: u32) -> u32 {
    let held = (0..HELD_OBJECTS).fold(0u32, |s, k| {
        (0..HELD_POINTS).fold(s, |s, j| {
            let w = held_weight(k, j);
            s.wrapping_add((gid.wrapping_mul(w) ^ k).wrapping_mul(w))
        })
    });
    held.wrapping_add(expected(INDEXED_WORDS, gid))
}

/// Private arrays a kernel reaches only at constant offsets count nothing toward the device's
/// Function storage: a driver holds the parts it reaches in registers, so their declared size, past
/// the device's storage, does not reach its scratch memory. 128 KiB of them (four times the Intel
/// venue's 32 KiB, more than twice lavapipe's 48 KiB) with a 16 KiB array indexed at run time
/// compile on every venue and answer exactly over 1, 64, 4096 and 262144 threads.
#[test]
fn arrays_reached_at_constant_offsets_past_the_device_answer_exactly() {
    on_device_with(at_plan, |device| {
        let state = state_of(device, &held_kernel(), "held_words")
            .unwrap_or_else(|e| panic!("{}: {e}", device.name()));
        for threads in [1, 64, 4096, 262_144] {
            let (correct, answer) = run(device, &state, threads, &held_expected);
            assert!(
                correct,
                "{} over {threads} threads: {answer}",
                device.name()
            );
        }
    });
}

/// The control: an array indexed at run time past every measured device's Function storage
/// (256 KiB, above the 192 KiB of MoltenVK on Apple silicon) is refused, needing
/// `FunctionStorageBytesPerThread`.
#[test]
fn an_array_indexed_at_run_time_past_the_device_is_refused() {
    on_device(|device| {
        let refused = state(device, 256 * 256).err().unwrap_or_default();
        assert!(
            refused.contains("FunctionStorageBytesPerThread"),
            "{}: {refused:?}",
            device.name()
        );
    });
}
