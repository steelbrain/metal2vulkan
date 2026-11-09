//! Submissions beside memory blocks being made while the host reclaims memory: what a
//! submission costs on a unified-memory device (RADV on a Radeon 780M) whose process makes
//! memory blocks while the kernel scans for free pages and compacts them, as it does on a host
//! whose builds left its memory fragmented.
//!
//! A block of a host-visible type lies in the kernel's GTT, whose pages its creation takes from
//! the kernel's page allocator, in high orders, under the reservation every submission of the
//! device's process also takes where the block is a VM-always-valid buffer object (RADV's
//! default): a creation that compacts memory for those pages holds every submission back.
//!
//! Each arm submits small command buffers for [`QUIET`] with nothing else running and for
//! [`PRESSED`] once the host's memory is fragmented, as a desktop's is after its builds: private
//! anonymous memory of all the host's free memory but [`FRAGMENT_SPARE_BYTES`], as far as the
//! bench's memory cgroup has room for it but [`CGROUP_SPARE_BYTES`] (a cgroup is charged for the
//! page cache it reads too, which its limit would reclaim in long runs, so the bench reads no
//! file), every page written, so that the free blocks of every order are taken, of which every
//! other page is then given back (`MADV_DONTNEED`), so that the free memory lies in single pages
//! between pages in use, `MemAvailable` staying high; beside the phase a thread maps private
//! anonymous memory advised for huge pages in pieces of [`PIECE_BYTES`], every page written, up to
//! [`ANON_HIGH_BYTES`] while the host's `MemAvailable` stays above [`FLOOR_BYTES`] and the cgroup
//! has room, then drops every other piece down to [`ANON_LOW_BYTES`], and again. Each command
//! buffer copies [`COPY_BYTES`] from a no-copy buffer over private anonymous memory (imported by
//! its pointer, a userptr on amdgpu) into a `Private` buffer and from that into another, and is
//! committed on the calling thread about every millisecond and waited for:
//! `CommandBuffer::commit_and_submit`, which submits on that thread unless the scheduler's thread
//! is submitting, timed, and the commit's turnaround to completion, timed too, so that a
//! submission held back on either thread shows. Beside the submissions, in both phases, a thread
//! (`reclaim-make`, so that a kernel trace tells it from the submitting one) makes memory as the
//! arm says, timing each making.
//!
//! The arms, one test each:
//! - **Made by the runtime**: a `Private` buffer of [`RUNTIME_BUFFER_BYTES`] every
//!   [`RUNTIME_EVERY`], at most [`RUNTIME_BUFFERS`] a phase: the runtime's pools grow, their
//!   blocks made ahead where it makes them ahead (`WorkCounts::buffer_blocks_made_ahead`).
//! - **System heap, 64 MiB** and **System heap, 4 MiB**: allocations of the device's first
//!   memory type that is host-visible and host-coherent but not device-local or cached (where a
//!   unified-memory device's `Private` blocks lie, `memory.rs`, `candidates`), of 64 MiB every
//!   [`RAW_EVERY`] or of 4 MiB sixteen times as often, [`RAW_BYTES`] a phase.
//! - **System heap, 64 MiB, exportable**: the system heap's 64 MiB allocations, each exportable
//!   as a dma-buf (`VkExportMemoryAllocateInfo`).
//! - **Carve-out, 64 MiB**: the same of the first device-local memory type that is not
//!   host-visible.
//! - **Made before**: [`HELD_BUFFERS`] `Private` buffers of [`RUNTIME_BUFFER_BYTES`] made before
//!   the quiet phase, and nothing made in the phases.
//! - **Exportable, held 32** and **Exportable, held 64**: that many of the system heap's 64 MiB
//!   allocations, each exportable, made before the quiet phase, and nothing made in the phases:
//!   what holding memory that is not private to the device's process costs each submission.
//! - **Nothing made**: the copies' buffers alone.
//!
//! Printed (`RECLAIM-SUBMIT`), per arm and phase: the submissions; for `commit_and_submit` and for
//! the turnaround each, those of 8 ms or more and of 17 ms or more, the median, 99th percentile and
//! slowest; the command buffers submitted, those the committing thread submitted
//! (`WorkCounts::batches_submitted_at_commit`) and the time spent in `vkQueueSubmit`
//! (`WorkCounts::batch_submit_nanos`); the makings, their median and slowest, and the runtime's
//! blocks made ahead and in place; the host's reclaim over the phase (`/proc/vmstat`: pages kswapd
//! and direct reclaim scanned, compaction stalls, allocation stalls, pages swapped out); the
//! memory the fragmenting took, the host's free memory and the cgroup's room it was sized from;
//! the free blocks of each order before and after the fragmenting and after the phase
//! (`/proc/buddyinfo`); and the slowest submissions', turnarounds' and makings' `CLOCK_MONOTONIC`
//! times, to line up with a kernel trace taken beside the run. Ignored: run it in release under
//! `--cfg m2v_gpu` on Linux, alone on its host (it presses the whole host's memory), one arm at a
//! time, in a cgroup with room for nearly all of the host's free memory: `cargo nextest run
//! --release -p metal2vulkan-runtime --lib --run-ignored all reclaim_submit_bench --no-capture`.

use super::host_residency::{device, Scratch};
use crate::{Buffer, CommandBufferStatus, CommandQueue, Device, HostMemory, ResourceOptions};
use ash::vk;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
const PAGE: u64 = 4096;
const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;
/// The runtime arm's buffers: two to a pooled block of 64 MiB.
const RUNTIME_BUFFER_BYTES: u64 = 32 * MIB;
const RUNTIME_BUFFERS: usize = 24;
const RUNTIME_EVERY: Duration = Duration::from_millis(400);
/// The raw arms' bytes a phase, and the gap between their 64 MiB allocations.
const RAW_BYTES: u64 = 768 * MIB;
const RAW_EVERY: Duration = Duration::from_millis(800);
/// The buffers made before the phases: 2 GiB, about what a consumer's guest holds in pooled
/// blocks at its desktop (32 to 35 blocks of 64 MiB).
const HELD_BUFFERS: u64 = 64;
/// The bytes each command buffer copies, twice.
const COPY_BYTES: u64 = 64 * 1024;
/// The no-copy buffer the copies read.
const NO_COPY_BYTES: u64 = 16 * MIB;
/// The phases' lengths.
const QUIET: Duration = Duration::from_secs(10);
const PRESSED: Duration = Duration::from_secs(20);
/// The gap between submissions.
const EVERY: Duration = Duration::from_millis(1);
/// The free memory the fragmenting mapping leaves: written, then given back page by page, every
/// other page kept.
const FRAGMENT_SPARE_BYTES: u64 = GIB;
/// The room the fragmenting mapping and the pressing thread leave in the bench's memory cgroup.
const CGROUP_SPARE_BYTES: u64 = 2 * GIB;
/// The anonymous memory the pressing thread holds between these bounds.
const ANON_HIGH_BYTES: u64 = 5 * GIB;
const ANON_LOW_BYTES: u64 = 2 * GIB;
const PIECE_BYTES: u64 = 256 * MIB;
/// The least `MemAvailable` at which the pressing thread maps another piece.
const FLOOR_BYTES: u64 = 3 * GIB / 2;
/// The slowest submissions and makings printed per phase.
const SLOWEST: usize = 8;

/// A private anonymous mapping, unmapped by the drop.
struct Anonymous {
    start: u64,
    length: u64,
}

impl Anonymous {
    /// A new mapping of `length` bytes, advised for huge pages where `huge` and against them
    /// otherwise (so that each of its pages is given back alone), every page written.
    fn new(length: u64, huge: bool) -> Anonymous {
        // SAFETY: a new private anonymous mapping.
        let start = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(start, libc::MAP_FAILED, "mmap");
        let advice = match huge {
            true => libc::MADV_HUGEPAGE,
            false => libc::MADV_NOHUGEPAGE,
        };
        // SAFETY: advice on the bench's own mapping; refused harmlessly without huge pages.
        unsafe { libc::madvise(start, length as usize, advice) };
        let start = start as u64;
        for page in 0..length / PAGE {
            // SAFETY: the mapping's own page, readable and writable, which no device uses yet.
            unsafe { *((start + page * PAGE) as *mut u8) = (page % 251) as u8 };
        }
        Anonymous { start, length }
    }
}

impl Drop for Anonymous {
    fn drop(&mut self) {
        // SAFETY: the bench's mapping, which nothing uses any more: a no-copy buffer over it goes
        // first.
        unsafe { libc::munmap(self.start as *mut libc::c_void, self.length as usize) };
    }
}

/// A `/proc/meminfo` field, in bytes.
fn meminfo(field: &str) -> u64 {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap();
    let line = text
        .lines()
        .find(|line| line.split(':').next() == Some(field))
        .unwrap();
    let kib: u64 = line.split_whitespace().nth(1).unwrap().parse().unwrap();
    kib * 1024
}

/// The bytes the bench's memory cgroup and each of its ancestors can still be charged before
/// their `memory.high` or `memory.max` (cgroup v2), the least of them; unbounded where none bounds
/// it or none is readable.
fn cgroup_room() -> u64 {
    let text = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    let Some(path) = text.lines().find_map(|line| line.strip_prefix("0::")) else {
        return u64::MAX;
    };
    let root = std::path::Path::new("/sys/fs/cgroup");
    let read = |dir: &std::path::Path, name: &str| {
        let text = std::fs::read_to_string(dir.join(name)).ok()?;
        text.trim().parse::<u64>().ok()
    };
    let own = root.join(path.trim_start_matches('/'));
    (own.ancestors())
        .take_while(|&dir| dir != root && dir.starts_with(root))
        .filter_map(|dir| {
            let current = read(dir, "memory.current")?;
            let bound = [read(dir, "memory.high"), read(dir, "memory.max")];
            Some(bound.into_iter().flatten().min()?.saturating_sub(current))
        })
        .min()
        .unwrap_or(u64::MAX)
}

/// The `/proc/vmstat` counts the bench prints, in its order.
const VMSTAT: [&str; 5] = [
    "pgscan_kswapd",
    "pgscan_direct",
    "compact_stall",
    "allocstall_normal",
    "pswpout",
];

fn vmstat() -> [u64; 5] {
    let text = std::fs::read_to_string("/proc/vmstat").unwrap();
    VMSTAT.map(|name| {
        (text.lines())
            .find_map(|line| {
                let (key, value) = line.split_once(' ')?;
                (key == name).then(|| value.trim().parse().unwrap())
            })
            .unwrap_or(0)
    })
}

/// The thread pressing the host's memory, stopped and joined by the drop.
struct Pressing {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Pressing {
    /// The host's memory fragmented (the module docs), what it took printed for arm `label`, then
    /// the thread started.
    fn start(label: &str) -> Pressing {
        let stop = Arc::new(AtomicBool::new(false));
        let (free, room) = (meminfo("MemFree"), cgroup_room());
        let length = (free.saturating_sub(FRAGMENT_SPARE_BYTES))
            .min(room.saturating_sub(CGROUP_SPARE_BYTES))
            / PAGE
            * PAGE;
        eprintln!(
            "RECLAIM-SUBMIT {label}: the fragmenting takes {} MiB of {} MiB free, the cgroup's \
             room {} MiB",
            length >> 20,
            free >> 20,
            room.min(u64::MAX >> 1) >> 20,
        );
        let fragments = Anonymous::new(length, false);
        for page in (0..length / PAGE).step_by(2) {
            // SAFETY: a page of the bench's own mapping, which nothing else uses: given back, it
            // reads as zeros if touched again.
            unsafe {
                libc::madvise(
                    (fragments.start + page * PAGE) as *mut libc::c_void,
                    PAGE as usize,
                    libc::MADV_DONTNEED,
                )
            };
        }
        let stopping = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("reclaim-press".to_owned())
            .spawn(move || press(&stopping, fragments))
            .unwrap();
        Pressing {
            stop,
            thread: Some(thread),
        }
    }
}

impl Drop for Pressing {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

/// The pressing thread's loop (the module docs), until `stop`, `fragments` kept until then.
fn press(stop: &AtomicBool, fragments: Anonymous) {
    let mut pieces: Vec<Anonymous> = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        while (pieces.len() as u64) * PIECE_BYTES < ANON_HIGH_BYTES
            && meminfo("MemAvailable") > FLOOR_BYTES + PIECE_BYTES
            && cgroup_room() > CGROUP_SPARE_BYTES + PIECE_BYTES
            && !stop.load(Ordering::Relaxed)
        {
            pieces.push(Anonymous::new(PIECE_BYTES, true));
        }
        // Every other piece dropped, so the free memory left is scattered between those kept.
        let mut index = 0;
        while (pieces.len() as u64) * PIECE_BYTES > ANON_LOW_BYTES && index < pieces.len() {
            pieces.remove(index);
            index += 1;
        }
    }
    drop(fragments);
}

/// A command buffer of `queue` copying [`COPY_BYTES`] from `source` into `middle` and from that
/// into `out`, committed on this thread and waited for: how long `commit_and_submit` took, and
/// how long from the commit until the command buffer was seen completed.
fn submit(
    queue: &CommandQueue,
    source: &Buffer,
    middle: &Buffer,
    out: &Buffer,
) -> (Duration, Duration) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    (encoder.copy_from_buffer_to_buffer(source, 0, middle, 0, COPY_BYTES)).unwrap();
    (encoder.copy_from_buffer_to_buffer(middle, 0, out, 0, COPY_BYTES)).unwrap();
    encoder.end_encoding().unwrap();
    let began = Instant::now();
    command_buffer.commit_and_submit().unwrap();
    let submitted = began.elapsed();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let completed = began.elapsed();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    (submitted, completed)
}

/// `CLOCK_MONOTONIC` now, in seconds.
fn monotonic() -> f64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a valid clock and a timespec to write.
    assert_eq!(
        unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut now) },
        0
    );
    now.tv_sec as f64 + now.tv_nsec as f64 * 1e-9
}

/// What an arm makes beside its submissions, in each phase.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Making {
    /// The runtime's `Private` buffers (the module docs).
    Runtime,
    /// Allocations of `bytes` of the memory type `index`, [`RAW_BYTES`] a phase, exportable as
    /// dma-bufs where `exportable`.
    Raw {
        index: u32,
        bytes: u64,
        exportable: bool,
    },
    Nothing,
}

/// What an arm made, freed at its end.
#[derive(Default)]
struct Made {
    buffers: Vec<Buffer>,
    raw: Vec<vk::DeviceMemory>,
}

/// Make as `making` says until `stop`: each making's time and `CLOCK_MONOTONIC` start.
fn make(
    device: &Device,
    making: Making,
    stop: &AtomicBool,
    made: &mut Made,
) -> Vec<(Duration, f64)> {
    let mut times = Vec::new();
    let (every, most) = match making {
        Making::Runtime => (RUNTIME_EVERY, RUNTIME_BUFFERS),
        Making::Raw { bytes, .. } => (
            RAW_EVERY.mul_f64(bytes as f64 / (64 * MIB) as f64),
            (RAW_BYTES / bytes) as usize,
        ),
        Making::Nothing => return times,
    };
    while !stop.load(Ordering::Relaxed) && times.len() < most {
        let at = monotonic();
        let began = Instant::now();
        match making {
            Making::Runtime => made.buffers.push(
                device
                    .new_buffer(RUNTIME_BUFFER_BYTES, ResourceOptions::private())
                    .unwrap(),
            ),
            Making::Raw {
                index,
                bytes,
                exportable,
            } => made.raw.push(allocate(device, index, bytes, exportable)),
            Making::Nothing => unreachable!(),
        }
        times.push((began.elapsed(), at));
        std::thread::sleep(every);
    }
    times
}

/// An allocation of `bytes` of `device`'s memory type `index`, exportable as a dma-buf where
/// `exportable`.
fn allocate(device: &Device, index: u32, bytes: u64, exportable: bool) -> vk::DeviceMemory {
    let mut export = vk::ExportMemoryAllocateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let mut info = vk::MemoryAllocateInfo::default()
        .allocation_size(bytes)
        .memory_type_index(index);
    if exportable {
        info = info.push_next(&mut export);
    }
    // SAFETY: the device is live, `info` valid and the type the device's.
    unsafe { crate::creation::allocate_memory(device.shared().vk(), &info) }.unwrap()
}

/// The median and slowest of `times`, in milliseconds, and the `count` slowest with their
/// `CLOCK_MONOTONIC` starts.
fn spread(times: &[(Duration, f64)], count: usize) -> (f64, f64, f64, String) {
    let ms = |took: Duration| took.as_secs_f64() * 1e3;
    if times.is_empty() {
        return (0.0, 0.0, 0.0, String::new());
    }
    let mut sorted: Vec<Duration> = times.iter().map(|&(took, _)| took).collect();
    sorted.sort();
    let at = |fraction: f64| ms(sorted[((sorted.len() - 1) as f64 * fraction) as usize]);
    let mut slowest = times.to_vec();
    slowest.sort_by_key(|&(took, _)| std::cmp::Reverse(took));
    let slowest: Vec<String> = (slowest.iter().take(count))
        .map(|&(took, at)| format!("{:.1} ms at {at:.3}", ms(took)))
        .collect();
    (
        at(0.5),
        at(0.99),
        ms(*sorted.last().unwrap()),
        slowest.join(", "),
    )
}

/// Submit for `length` beside `making` (the module docs), then print the phase `name` of
/// `arm`.
fn phase(
    device: &Device,
    (queue, source, middle, out): (&CommandQueue, &Buffer, &Buffer, &Buffer),
    (arm, name): (&str, &str),
    (length, making): (Duration, Making),
    made: &mut Made,
) {
    let before = (vmstat(), device.work_counts());
    let stop = AtomicBool::new(false);
    let (submits, turnarounds, makes) = std::thread::scope(|scope| {
        let maker = (std::thread::Builder::new().name("reclaim-make".to_owned()))
            .spawn_scoped(scope, || make(device, making, &stop, made))
            .unwrap();
        let began = Instant::now();
        let (mut submits, mut turnarounds) = (Vec::new(), Vec::new());
        while began.elapsed() < length {
            let at = monotonic();
            let (submitted, completed) = submit(queue, source, middle, out);
            submits.push((submitted, at));
            turnarounds.push((completed, at));
            std::thread::sleep(EVERY);
        }
        stop.store(true, Ordering::Relaxed);
        (submits, turnarounds, maker.join().unwrap())
    });
    let after = (vmstat(), device.work_counts());
    let over = |times: &[(Duration, f64)], bound: f64| {
        (times.iter())
            .filter(|&&(took, _)| took.as_secs_f64() * 1e3 >= bound)
            .count()
    };
    let reclaim: Vec<String> = (VMSTAT.iter().zip(before.0.iter().zip(after.0)))
        .map(|(name, (before, after))| format!("{name} {}", after - before))
        .collect();
    let (counts, earlier) = (&after.1, &before.1);
    let (median, p99, slowest, slow) = spread(&submits, SLOWEST);
    let (turn_median, turn_p99, turn_slowest, slow_turns) = spread(&turnarounds, SLOWEST);
    let (make_median, _, make_slowest, slow_makes) = spread(&makes, SLOWEST);
    let submitted = counts.command_buffers_submitted - earlier.command_buffers_submitted;
    let submit_nanos = counts.batch_submit_nanos - earlier.batch_submit_nanos;
    eprintln!(
        "RECLAIM-SUBMIT {} {arm}, {name}: {} submissions; commit_and_submit {} of 8 ms or more, \
         {} of 17 ms or more, ms median {median:.3}, p99 {p99:.3}, slowest {slowest:.3}; \
         turnaround {} of 8 ms or more, {} of 17 ms or more, ms median {turn_median:.3}, p99 \
         {turn_p99:.3}, slowest {turn_slowest:.3}; command buffers submitted {submitted}, by the \
         committing thread {}, in vkQueueSubmit {:.1} ms, {:.3} ms each; {} makings, ms median \
         {make_median:.1}, slowest {make_slowest:.1}; blocks made ahead {}, in place {}; \
         reclaim: {}; blocks live {}, empty {}; MemAvailable {} MiB",
        device.name(),
        submits.len(),
        over(&submits, 8.0),
        over(&submits, 17.0),
        over(&turnarounds, 8.0),
        over(&turnarounds, 17.0),
        counts.batches_submitted_at_commit - earlier.batches_submitted_at_commit,
        submit_nanos as f64 * 1e-6,
        submit_nanos as f64 * 1e-6 / submitted.max(1) as f64,
        makes.len(),
        counts.buffer_blocks_made_ahead - earlier.buffer_blocks_made_ahead,
        counts.buffer_blocks_made_in_place - earlier.buffer_blocks_made_in_place,
        reclaim.join(", "),
        counts.memory_blocks_live,
        counts.empty_memory_blocks_live,
        meminfo("MemAvailable") >> 20,
    );
    eprintln!("RECLAIM-SUBMIT {arm}, {name}: slowest submissions {slow}");
    eprintln!("RECLAIM-SUBMIT {arm}, {name}: slowest turnarounds {slow_turns}");
    eprintln!("RECLAIM-SUBMIT {arm}, {name}: slowest makings {slow_makes}");
}

/// The first memory type of `device` whose flags hold `with` and none of `without`.
fn memory_type(
    device: &Device,
    with: vk::MemoryPropertyFlags,
    without: vk::MemoryPropertyFlags,
) -> u32 {
    let memory = crate::device::lock(&device.shared().memory);
    (0..vk::MAX_MEMORY_TYPES as u32)
        .find(|&index| {
            (memory.type_flags(index))
                .is_some_and(|flags| flags.contains(with) && !flags.intersects(without))
        })
        .expect("a memory type")
}

/// The memory an arm makes: its system heap's or its carve-out's.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    Runtime,
    SystemHeap {
        bytes: u64,
        exportable: bool,
    },
    CarveOut,
    MadeBefore,
    /// That many exportable 64 MiB allocations of the system heap made before the phases.
    ExportableHeld(usize),
    Nothing,
}

/// Run the arm `label` (the module docs).
fn arm(label: &str, arm: Arm) {
    use vk::MemoryPropertyFlags as P;
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    if device.shared().host_import.is_err() {
        eprintln!("RECLAIM-SUBMIT: the device imports no host memory here: not run");
        return;
    }
    let queue = device.new_command_queue().unwrap();
    let private = |length| {
        device
            .new_buffer(length, ResourceOptions::private())
            .unwrap()
    };
    let mut made = Made::default();
    let system_heap = || {
        memory_type(
            &device,
            P::HOST_VISIBLE | P::HOST_COHERENT,
            P::DEVICE_LOCAL | P::HOST_CACHED | P::DEVICE_COHERENT_AMD | P::PROTECTED,
        )
    };
    let making = match arm {
        Arm::Runtime => Making::Runtime,
        Arm::SystemHeap { bytes, exportable } => Making::Raw {
            index: system_heap(),
            bytes,
            exportable,
        },
        Arm::CarveOut => Making::Raw {
            index: memory_type(
                &device,
                P::DEVICE_LOCAL,
                P::HOST_VISIBLE | P::DEVICE_COHERENT_AMD | P::PROTECTED,
            ),
            bytes: 64 * MIB,
            exportable: false,
        },
        Arm::MadeBefore => {
            made.buffers = (0..HELD_BUFFERS)
                .map(|_| private(RUNTIME_BUFFER_BYTES))
                .collect();
            Making::Nothing
        }
        Arm::ExportableHeld(count) => {
            let index = system_heap();
            made.raw = (0..count)
                .map(|_| allocate(&device, index, 64 * MIB, true))
                .collect();
            Making::Nothing
        }
        Arm::Nothing => Making::Nothing,
    };
    let pages = Anonymous::new(NO_COPY_BYTES, false);
    // SAFETY: the pages outlive the buffer, which drops first.
    let source = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(NonNull::new(pages.start as *mut u8).unwrap()),
            pages.length,
            ResourceOptions::default(),
            None,
        )
    }
    .unwrap();
    let (middle, out) = (private(COPY_BYTES), private(COPY_BYTES));
    let on = (&queue, &source, &middle, &out);
    phase(&device, on, (label, "quiet"), (QUIET, making), &mut made);
    let buddies = |when: &str| {
        let text = std::fs::read_to_string("/proc/buddyinfo").unwrap_or_default();
        for zone in text.lines().filter(|line| line.contains("Normal")) {
            eprintln!("RECLAIM-SUBMIT {label}: free blocks by order {when}: {zone}");
        }
    };
    buddies("before the fragmenting");
    let pressing = Pressing::start(label);
    buddies("after it");
    phase(
        &device,
        on,
        (label, "pressed"),
        (PRESSED, making),
        &mut made,
    );
    buddies("after the pressed phase");
    drop(pressing);
    drop((source, middle, out, made.buffers));
    for memory in made.raw {
        // SAFETY: nothing uses the memory.
        unsafe { crate::release::object(device.shared().vk(), memory) };
    }
    drop(pages);
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_runtime() {
    arm("made by the runtime", Arm::Runtime);
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_system_heap_64mib() {
    arm(
        "system heap, 64 MiB",
        Arm::SystemHeap {
            bytes: 64 * MIB,
            exportable: false,
        },
    );
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_system_heap_4mib() {
    arm(
        "system heap, 4 MiB",
        Arm::SystemHeap {
            bytes: 4 * MIB,
            exportable: false,
        },
    );
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_system_heap_64mib_exportable() {
    arm(
        "system heap, 64 MiB, exportable",
        Arm::SystemHeap {
            bytes: 64 * MIB,
            exportable: true,
        },
    );
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_carve_out_64mib() {
    arm("carve-out, 64 MiB", Arm::CarveOut);
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_made_before() {
    arm("made before", Arm::MadeBefore);
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_exportable_held_32() {
    arm("exportable, held 32", Arm::ExportableHeld(32));
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_exportable_held_64() {
    arm("exportable, held 64", Arm::ExportableHeld(64));
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_nothing_made() {
    arm("nothing made", Arm::Nothing);
}
