//! Submissions beside memory blocks being made while the host reclaims memory: what a
//! submission costs on a unified-memory device (RADV on a Radeon 780M) whose process makes
//! memory blocks while the kernel scans for free pages and compacts them, as it does on a host
//! whose builds left its page cache full.
//!
//! A block of a host-visible type lies in the kernel's GTT, whose pages its creation takes from
//! the kernel's page allocator, in high orders, under the reservation every submission of the
//! device's process also takes where the block is a VM-always-valid buffer object (RADV's
//! default): a creation that compacts memory for those pages holds every submission back.
//!
//! Each arm submits small command buffers for [`QUIET`] with nothing else running and for
//! [`PRESSED`] once the host's memory is fragmented, as a desktop's is after its builds: half of
//! it read into the page cache from a sparse file (holes, so nothing is written to disk), and
//! [`FRAGMENT_BYTES`] of private anonymous memory, every page written, of which every other page
//! is then given back (`MADV_DONTNEED`), so that much of the free memory lies in single pages
//! between pages in use, `MemAvailable` staying high; beside the phase a thread maps private
//! anonymous memory advised for huge pages in pieces of [`PIECE_BYTES`], every page written, up
//! to [`ANON_HIGH_BYTES`] while the host's `MemAvailable` stays above [`FLOOR_BYTES`], drops every
//! other piece down to [`ANON_LOW_BYTES`] and reads the file again. Each command buffer copies [`COPY_BYTES`] from a no-copy buffer over
//! private anonymous memory (imported by its pointer, a userptr on amdgpu) into a `Private`
//! buffer and from that into another, and is submitted on the calling thread
//! (`CommandBuffer::commit_and_submit`, timed) about every millisecond and waited for. Beside the
//! submissions, in both phases, a thread makes memory as the arm says, timing each making.
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
//! - **Nothing made**: the copies' buffers alone.
//!
//! Printed (`RECLAIM-SUBMIT`), per arm and phase: the submissions, those of 8 ms or more and of 17
//! ms or more, the median, 99th percentile and slowest `commit_and_submit`; the makings, their
//! median and slowest, and the runtime's blocks made ahead and in place; the host's reclaim over
//! the phase (`/proc/vmstat`: pages kswapd and direct reclaim scanned, compaction stalls,
//! allocation stalls, pages swapped out); and the slowest submissions' and makings'
//! `CLOCK_MONOTONIC` times, to line up with a kernel trace taken beside the run. Ignored: run it in
//! release under `--cfg m2v_gpu` on Linux, alone on its host (it presses the whole host's
//! memory), one arm at a time: `cargo nextest run --release -p metal2vulkan-runtime --lib
//! --run-ignored all reclaim_submit_bench --no-capture`.

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
/// The anonymous memory written and then given back page by page, every other page kept.
const FRAGMENT_BYTES: u64 = 6 * GIB;
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
    /// A new mapping of `length` bytes, advised for huge pages where `huge`, every page written.
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
        if huge {
            // SAFETY: advice on the bench's own mapping; refused harmlessly without huge pages.
            unsafe { libc::madvise(start, length as usize, libc::MADV_HUGEPAGE) };
        }
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

/// A sparse file of `length` bytes in the current directory (on the host's disk, not a `tmpfs`),
/// removed by the drop: reading it fills the page cache with its holes' zeros and writes nothing.
struct Heat(std::path::PathBuf);

impl Heat {
    fn new(length: u64) -> Heat {
        let path = std::env::current_dir()
            .unwrap()
            .join(format!(".reclaim-heat-{}", std::process::id()));
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(length).unwrap();
        Heat(path)
    }

    /// Read the whole file, through the page cache, unless `stop` is set first.
    fn read(&self, stop: &AtomicBool) {
        use std::io::Read;
        let mut file = std::fs::File::open(&self.0).unwrap();
        let mut chunk = vec![0u8; 4 * MIB as usize];
        while !stop.load(Ordering::Relaxed) {
            if file.read(&mut chunk).unwrap() == 0 {
                break;
            }
        }
    }
}

impl Drop for Heat {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
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
    /// The host's memory fragmented (the module docs), then the thread started.
    fn start() -> Pressing {
        let stop = Arc::new(AtomicBool::new(false));
        let heat = Heat::new(meminfo("MemTotal") / 2);
        heat.read(&stop);
        let fragments = Anonymous::new(FRAGMENT_BYTES, false);
        for page in (0..FRAGMENT_BYTES / PAGE).step_by(2) {
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
            .spawn(move || press(&stopping, &heat, fragments))
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
fn press(stop: &AtomicBool, heat: &Heat, fragments: Anonymous) {
    let mut pieces: Vec<Anonymous> = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        while (pieces.len() as u64) * PIECE_BYTES < ANON_HIGH_BYTES
            && meminfo("MemAvailable") > FLOOR_BYTES + PIECE_BYTES
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
        heat.read(stop);
    }
    drop(fragments);
}

/// A command buffer of `queue` copying [`COPY_BYTES`] from `source` into `middle` and from that
/// into `out`, submitted on this thread: how long `commit_and_submit` took. Waited for.
fn submit(queue: &CommandQueue, source: &Buffer, middle: &Buffer, out: &Buffer) -> Duration {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    (encoder.copy_from_buffer_to_buffer(source, 0, middle, 0, COPY_BYTES)).unwrap();
    (encoder.copy_from_buffer_to_buffer(middle, 0, out, 0, COPY_BYTES)).unwrap();
    encoder.end_encoding().unwrap();
    let began = Instant::now();
    command_buffer.commit_and_submit().unwrap();
    let took = began.elapsed();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    took
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
            } => {
                let mut export = vk::ExportMemoryAllocateInfo::default()
                    .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
                let mut info = vk::MemoryAllocateInfo::default()
                    .allocation_size(bytes)
                    .memory_type_index(index);
                if exportable {
                    info = info.push_next(&mut export);
                }
                // SAFETY: the device is live, `info` valid and the type the device's.
                let memory =
                    unsafe { crate::creation::allocate_memory(device.shared().vk(), &info) };
                made.raw.push(memory.unwrap());
            }
            Making::Nothing => unreachable!(),
        }
        times.push((began.elapsed(), at));
        std::thread::sleep(every);
    }
    times
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
    let (submits, makes) = std::thread::scope(|scope| {
        let maker = scope.spawn(|| make(device, making, &stop, made));
        let began = Instant::now();
        let mut submits: Vec<(Duration, f64)> = Vec::new();
        while began.elapsed() < length {
            let at = monotonic();
            submits.push((submit(queue, source, middle, out), at));
            std::thread::sleep(EVERY);
        }
        stop.store(true, Ordering::Relaxed);
        (submits, maker.join().unwrap())
    });
    let after = (vmstat(), device.work_counts());
    let over = |bound: f64| {
        (submits.iter())
            .filter(|&&(took, _)| took.as_secs_f64() * 1e3 >= bound)
            .count()
    };
    let reclaim: Vec<String> = (VMSTAT.iter().zip(before.0.iter().zip(after.0)))
        .map(|(name, (before, after))| format!("{name} {}", after - before))
        .collect();
    let (counts, earlier) = (&after.1, &before.1);
    let (median, p99, slowest, slow) = spread(&submits, SLOWEST);
    let (make_median, _, make_slowest, slow_makes) = spread(&makes, SLOWEST);
    eprintln!(
        "RECLAIM-SUBMIT {} {arm}, {name}: {} submissions, {} of 8 ms or more, {} of 17 ms or \
         more; commit_and_submit ms median {median:.3}, p99 {p99:.3}, slowest {slowest:.3}; {} \
         makings, ms median {make_median:.1}, slowest {make_slowest:.1}; blocks made ahead {}, \
         in place {}; reclaim: {}; blocks live {}, empty {}; MemAvailable {} MiB",
        device.name(),
        submits.len(),
        over(8.0),
        over(17.0),
        makes.len(),
        counts.buffer_blocks_made_ahead - earlier.buffer_blocks_made_ahead,
        counts.buffer_blocks_made_in_place - earlier.buffer_blocks_made_in_place,
        reclaim.join(", "),
        counts.memory_blocks_live,
        counts.empty_memory_blocks_live,
        meminfo("MemAvailable") >> 20,
    );
    eprintln!("RECLAIM-SUBMIT {arm}, {name}: slowest submissions {slow}");
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
    SystemHeap { bytes: u64, exportable: bool },
    CarveOut,
    MadeBefore,
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
    let making = match arm {
        Arm::Runtime => Making::Runtime,
        Arm::SystemHeap { bytes, exportable } => Making::Raw {
            index: memory_type(
                &device,
                P::HOST_VISIBLE | P::HOST_COHERENT,
                P::DEVICE_LOCAL | P::HOST_CACHED | P::DEVICE_COHERENT_AMD | P::PROTECTED,
            ),
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
    let pressing = Pressing::start();
    phase(
        &device,
        on,
        (label, "pressed"),
        (PRESSED, making),
        &mut made,
    );
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
fn reclaim_submit_bench_nothing_made() {
    arm("nothing made", Arm::Nothing);
}
