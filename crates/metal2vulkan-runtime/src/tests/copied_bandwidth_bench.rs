//! The bandwidth of a `Shared` no-copy buffer that is a copied host buffer (`copied.rs`), as a
//! bandwidth test run where the device imports no such pages measures it: the gate refuses the
//! import, as an RTX 5080 refuses a guest's pages, so the GPU uses a copy that each commit copies
//! the CPU's writes into and each completion merges the GPU's writes back from. One command
//! buffer a pass, waited for, at 16, 64 and 256 MiB, in seven patterns:
//!
//! - `kernel-read`: `scale_and_offset` reads the buffer whole and writes a `Private` one;
//! - `kernel-write`: it reads a `Private` buffer and writes the buffer whole, each pass's scale
//!   another, so every float changes;
//! - `kernel-in-place`: it reads and writes the buffer whole in place (`x + i`);
//! - `blit-read`: a blit copies the buffer whole into a `Private` one;
//! - `blit-write`: a blit copies one of two `Private` buffers of other bytes over it whole;
//! - `cpu-then-read`: the CPU rewrites every float of the pages, then the kernel reads them;
//! - `both-write`: the CPU writes a word of every 16th page, then the kernel rewrites in place.
//!
//! Each runs over a copy with no log of its pages' writes, over the same with its comparisons on
//! up to eight threads (`copied-t8`, `copied::SPLIT_THREADS_FROM`), over one with a log the CPU's
//! writes are told to (`copied_log::TestLog`, as a guest RAM's dirty log), and, as the control,
//! over a `Shared` buffer of the device's own (host memory the GPU reaches across the bus).
//! Printed as the median of the passes after the first two (`COPIED-BW`): the buffer's bytes a
//! second over
//! the pass's wall time (the GB/s a test that times its command buffers reports), its commit and
//! the copy in in it, the GPU's time, the completion's merge back and its reads back of a base's
//! blocks, and the rest of the wait (the latency of the GPU's start and of the completion's
//! wake). The merge back and copy in are split by the phases of every 64th block they look at
//! (`copied::clock`): finding the block changed, reading the other side's, storing the bytes that
//! changed, the base's taking of it. Each copied row then checks the pages hold what the copy
//! does (a blit of it into the device's own memory). Ignored: run it in release, under the GPU
//! safety cfg,
//! `RUSTFLAGS='--cfg m2v_gpu' cargo test --release -p metal2vulkan-runtime --lib
//! copied_bandwidth_bench -- --ignored --nocapture`.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device that imports no host memory says so and passes.

use super::copied_log::TestLog;
use crate::copied::{self, clock};
use crate::{Buffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device};
use crate::{DeviceKind, DeviceOptions, HostMemory, ImportGate, PipelineOptions};
use crate::{ResourceOptions, Size, WorkCounts};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
/// The passes of each row, the first two of which are not counted.
const PASSES: usize = 12;
const WARM: usize = 2;
const LENGTHS: [usize; 3] = [16 << 20, 64 << 20, 256 << 20];
const PAGE: usize = 4096;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pattern {
    KernelRead,
    KernelWrite,
    KernelInPlace,
    BlitRead,
    BlitWrite,
    CpuThenRead,
    BothWrite,
}

const PATTERNS: [Pattern; 7] = [
    Pattern::KernelRead,
    Pattern::KernelWrite,
    Pattern::KernelInPlace,
    Pattern::BlitRead,
    Pattern::BlitWrite,
    Pattern::CpuThenRead,
    Pattern::BothWrite,
];

impl Pattern {
    fn name(self) -> &'static str {
        match self {
            Pattern::KernelRead => "kernel-read",
            Pattern::KernelWrite => "kernel-write",
            Pattern::KernelInPlace => "kernel-in-place",
            Pattern::BlitRead => "blit-read",
            Pattern::BlitWrite => "blit-write",
            Pattern::CpuThenRead => "cpu-then-read",
            Pattern::BothWrite => "both-write",
        }
    }
}

/// What a row's buffer is.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Subject {
    /// A copied host buffer over pages of the bench's own, logged or not, its comparisons on at
    /// most `threads` threads (`copied::SPLIT_THREADS_FROM`; 0: the runtime's own count).
    Copied { logged: bool, threads: usize },
    /// A `Shared` buffer of the device's own.
    Own,
}

impl Subject {
    fn name(self) -> &'static str {
        match self {
            Subject::Copied {
                logged: false,
                threads: 0,
            } => "copied",
            Subject::Copied {
                logged: false,
                threads: _,
            } => "copied-t8",
            Subject::Copied { logged: true, .. } => "copied-logged",
            Subject::Own => "own-shared",
        }
    }
}

/// A device of the cfg's kind; `None` where it imports no host memory.
fn device() -> Option<Device> {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    if device.shared().host_import.is_err() {
        eprintln!("COPIED-BW: the device imports no host memory here; nothing to test");
        return None;
    }
    Some(device)
}

/// Host memory of whole pages, freed when dropped (after every buffer over it).
struct Pages {
    pointer: NonNull<u8>,
    layout: Layout,
}

impl Pages {
    fn new(length: usize) -> Pages {
        let layout = Layout::from_size_align(length, 1 << 16).unwrap();
        // SAFETY: the layout's size is not zero.
        let pointer = NonNull::new(unsafe { std::alloc::alloc_zeroed(layout) }).unwrap();
        Pages { pointer, layout }
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: allocated by `Pages::new` with this layout; no buffer over it lives now.
        unsafe { std::alloc::dealloc(self.pointer.as_ptr(), self.layout) };
    }
}

/// Write `floats` floats at `at`, each `value(i)`.
fn fill(at: NonNull<u8>, floats: usize, value: impl Fn(usize) -> f32) {
    // SAFETY: `at` is `floats` floats of memory the bench owns, on 4 bytes, used by no running
    // command.
    let words = unsafe { std::slice::from_raw_parts_mut(at.as_ptr().cast::<f32>(), floats) };
    for (i, word) in words.iter_mut().enumerate() {
        *word = value(i);
    }
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// The clock's readings: whole copies in, merges back and reads back, then the phases of copies
/// in and of merges back.
fn clock_now() -> [u64; 11] {
    let mut now = [0u64; 11];
    for (which, whole) in clock::WHOLE.iter().enumerate() {
        now[which] = whole.load(Ordering::Relaxed);
    }
    for (of, phases) in clock::PHASES.iter().enumerate() {
        for (phase, spent) in phases.iter().enumerate() {
            now[3 + 4 * of + phase] = spent.load(Ordering::Relaxed);
        }
    }
    now
}

/// The buffers a size's rows share: two `Private` sources of other bytes, a `Private` target, a
/// `Shared` check buffer of the device's own.
struct Others {
    sources: [Buffer; 2],
    target: Buffer,
    check: Buffer,
}

impl Others {
    fn new(device: &Device, queue: &CommandQueue, length: usize) -> Others {
        let floats = length / 4;
        let private = || (device.new_buffer(length as u64, ResourceOptions::private())).unwrap();
        let sources = [private(), private()];
        let staging = device
            .new_buffer(length as u64, ResourceOptions::shared())
            .unwrap();
        for (k, source) in sources.iter().enumerate() {
            fill(staging.contents().unwrap(), floats, |i| {
                (i * (k + 1)) as f32
            });
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_buffer(&staging, 0, source, 0, length as u64)
                .unwrap();
            blit.end_encoding().unwrap();
            command_buffer.commit().unwrap();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
        }
        Others {
            sources,
            target: private(),
            check: device
                .new_buffer(length as u64, ResourceOptions::shared())
                .unwrap(),
        }
    }
}

/// Encode one pass of `pattern` over `buffer` into a new command buffer of `queue`.
fn encode(
    queue: &CommandQueue,
    state: &ComputePipelineState,
    (pattern, pass): (Pattern, usize),
    buffer: &Buffer,
    others: &Others,
) -> crate::CommandBuffer {
    let length = buffer.length();
    let command_buffer = queue.command_buffer().unwrap();
    let kernel = |input: &Buffer, output: &Buffer, scale: f32| {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(state).unwrap();
        encoder.set_buffer(Some(input), 0, 0).unwrap();
        encoder.set_buffer(Some(output), 0, 1).unwrap();
        encoder.set_bytes(&scale.to_le_bytes(), 2).unwrap();
        let threads = Size::new(length / 4, 1, 1);
        encoder
            .dispatch_threads(threads, Size::new(256, 1, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
    };
    let blit = |from: &Buffer, to: &Buffer| {
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        encoder
            .copy_from_buffer_to_buffer(from, 0, to, 0, length)
            .unwrap();
        encoder.end_encoding().unwrap();
    };
    match pattern {
        Pattern::KernelRead | Pattern::CpuThenRead => kernel(buffer, &others.target, 1.0),
        Pattern::KernelWrite => kernel(&others.sources[0], buffer, (pass + 1) as f32),
        Pattern::KernelInPlace | Pattern::BothWrite => kernel(buffer, buffer, 1.0),
        Pattern::BlitRead => blit(buffer, &others.target),
        Pattern::BlitWrite => blit(&others.sources[pass % 2], buffer),
    }
    command_buffer
}

/// The CPU's writes of `pattern`'s pass `pass` into the `length` bytes at `pages`, told to `log`.
fn cpu_writes(
    (pattern, pass): (Pattern, usize),
    pages: NonNull<u8>,
    length: usize,
    log: Option<&TestLog>,
) {
    match pattern {
        Pattern::CpuThenRead => {
            fill(pages, length / 4, |i| (i + pass) as f32);
            if let Some(log) = log {
                log.told(0..length as u64, 0);
            }
        }
        Pattern::BothWrite => {
            for page in (0..length / PAGE).step_by(16) {
                // SAFETY: a word of the bench's own pages, on 4 bytes.
                unsafe {
                    (pages.as_ptr().add(page * PAGE).cast::<u32>()).write_volatile(pass as u32 + 7)
                };
                if let Some(log) = log {
                    let at = (page * PAGE) as u64;
                    log.told(at..at + 4, 0);
                }
            }
        }
        _ => {}
    }
}

/// One row: `PASSES` passes of `pattern` over a buffer of `length` bytes made as `subject`.
fn row(
    device: &Device,
    queue: &CommandQueue,
    state: &ComputePipelineState,
    (pattern, subject, length): (Pattern, Subject, usize),
    others: &Others,
) {
    let floats = length / 4;
    let log = Arc::new(TestLog::default());
    let pages = Pages::new(length);
    let buffer = match subject {
        Subject::Copied { logged, .. } => {
            fill(pages.pointer, floats, |i| i as f32);
            let refuse: ImportGate = Arc::new(|_| false);
            // SAFETY: the pages outlive the buffer (dropped after it), and only its commands and
            // the bench's writes between command buffers use them.
            let memory = unsafe { HostMemory::new(pages.pointer) }.with_import_gate(refuse);
            let memory = match logged {
                true => memory.with_dirty_pages(Arc::clone(&log)),
                false => memory,
            };
            // SAFETY: as above.
            let buffer = unsafe {
                device.new_buffer_with_bytes_no_copy(
                    memory,
                    length as u64,
                    ResourceOptions::default(),
                    None,
                )
            }
            .unwrap();
            assert!(buffer.copied(), "the buffer is a copy");
            buffer
        }
        Subject::Own => {
            let buffer = device
                .new_buffer(length as u64, ResourceOptions::shared())
                .unwrap();
            fill(buffer.contents().unwrap(), floats, |i| i as f32);
            buffer
        }
    };
    let contents = buffer.contents().unwrap();
    let logged = matches!(subject, Subject::Copied { logged: true, .. }).then_some(&*log);
    if let Subject::Copied { threads, .. } = subject {
        copied::SPLIT_THREADS_FROM.store(threads, Ordering::Relaxed);
    }
    let (mut walls, mut commits, mut gpus, mut waits) = (vec![], vec![], vec![], vec![]);
    let (mut copy_ins, mut merges, mut reads_back) = (vec![], vec![], vec![]);
    let mut phases = [0u64; 8];
    let mut counted = (WorkCounts::default(), WorkCounts::default());
    for pass in 0..PASSES {
        cpu_writes((pattern, pass), contents, length, logged);
        let command_buffer = encode(queue, state, (pattern, pass), &buffer, others);
        if pass == WARM {
            counted.0 = device.work_counts();
        }
        let before = clock_now();
        let started = Instant::now();
        command_buffer.commit().unwrap();
        let committed = Instant::now();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        let ended = Instant::now();
        let after = clock_now();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        if pass < WARM {
            continue;
        }
        let ms = |nanos: u64| nanos as f64 / 1e6;
        let gpu = (command_buffer.gpu_end_time().unwrap_or(0.0))
            - command_buffer.gpu_start_time().unwrap_or(0.0);
        walls.push((ended - started).as_secs_f64() * 1e3);
        commits.push((committed - started).as_secs_f64() * 1e3);
        waits.push((ended - committed).as_secs_f64() * 1e3);
        gpus.push(gpu * 1e3);
        copy_ins.push(ms(after[0] - before[0]));
        merges.push(ms(after[1] - before[1]));
        reads_back.push(ms(after[2] - before[2]));
        for (phase, spent) in phases.iter_mut().enumerate() {
            *spent += after[3 + phase] - before[3 + phase];
        }
    }
    counted.1 = device.work_counts();
    let timed = (PASSES - WARM) as u64;
    let (wall, gpu, wait) = (median(walls), median(gpus), median(waits));
    let (copy_in, merge) = (median(copy_ins), median(merges));
    // Each comparison's phases as shares of its median, from the sampled blocks.
    let split = |which: &[u64], whole: f64| {
        let sum: u64 = which.iter().sum();
        let share = |at: usize| match sum {
            0 => 0.0,
            _ => whole * which[at] as f64 / sum as f64,
        };
        format!(
            "find {:.2}, other {:.2}, store {:.2}, base {:.2}",
            share(0),
            share(1),
            share(2),
            share(3)
        )
    };
    let (before, after) = (&counted.0, &counted.1);
    let per = |a: u64, b: u64| (b - a) / timed;
    eprintln!(
        "COPIED-BW {} {} {} MiB: {:.2} GB/s; a pass {wall:.2} ms: commit {:.2} (copy in {copy_in:.2}: \
         {}), GPU {gpu:.2} ({:.2} GB/s), merge back {merge:.2} ({}; reads back {:.2}), rest of the \
         wait {:.2}; a pass copied in {} KiB (skipped {} KiB), merged back {} KiB, read {} base \
         blocks ({} back), started {} threads",
        pattern.name(),
        subject.name(),
        length >> 20,
        length as f64 / 1e6 / wall,
        median(commits),
        split(&phases[..4], copy_in),
        length as f64 / 1e6 / gpu,
        split(&phases[4..], merge),
        median(reads_back),
        wait - gpu - merge,
        per(before.copied_copy_in_bytes, after.copied_copy_in_bytes) >> 10,
        per(
            before.copied_copy_in_bytes_skipped,
            after.copied_copy_in_bytes_skipped
        ) >> 10,
        per(before.copied_merge_back_bytes, after.copied_merge_back_bytes) >> 10,
        per(before.copied_base_blocks_read, after.copied_base_blocks_read),
        per(
            before.copied_base_blocks_read_back,
            after.copied_base_blocks_read_back
        ),
        per(before.copied_compare_threads, after.copied_compare_threads),
    );
    if let Subject::Copied { .. } = subject {
        // The pages hold what the copy does: a blit's commit copies in what the CPU wrote last,
        // and its read of the copy lands in the device's own memory.
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&buffer, 0, &others.check, 0, length as u64)
            .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        // SAFETY: both are `length` bytes no command uses now.
        let (copy, held) = unsafe {
            (
                std::slice::from_raw_parts(others.check.contents().unwrap().as_ptr(), length),
                std::slice::from_raw_parts(pages.pointer.as_ptr(), length),
            )
        };
        assert!(
            copy == held,
            "{} {}: the pages hold what the copy does",
            pattern.name(),
            subject.name()
        );
    }
    copied::SPLIT_THREADS_FROM.store(0, Ordering::Relaxed);
    drop(buffer);
    drop(pages);
}

/// Every pattern at every length over each subject.
#[test]
#[ignore = "a release-mode benchmark"]
fn copied_bandwidth_bench() {
    let Some(device) = device() else {
        return;
    };
    eprintln!("COPIED-BW on {}", device.name());
    let function = super::encoder_holds::library(&device, "examples/runtime_examples.ll")
        .new_function_with_name("scale_and_offset")
        .unwrap();
    let state = device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let subjects = [
        Subject::Copied {
            logged: false,
            threads: 0,
        },
        Subject::Copied {
            logged: false,
            threads: 8,
        },
        Subject::Copied {
            logged: true,
            threads: 0,
        },
        Subject::Own,
    ];
    for length in LENGTHS {
        let others = Others::new(&device, &queue, length);
        for pattern in PATTERNS {
            for subject in subjects {
                row(&device, &queue, &state, (pattern, subject, length), &others);
            }
        }
    }
}
