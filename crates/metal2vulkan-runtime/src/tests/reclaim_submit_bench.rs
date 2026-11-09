//! Submissions beside a device's held memory while the host reclaims memory: what a submission
//! costs on a unified-memory device (RADV on a Radeon 780M) whose process holds idle memory
//! blocks while the kernel scans for free pages and compacts them, as it does on a host whose
//! builds left its page cache full.
//!
//! Each arm makes its held memory, then submits small command buffers for [`QUIET`] with nothing
//! else running and for [`PRESSED`] beside a thread that presses the host's memory: it reads a
//! sparse file of [`HEAT_BYTES`] into the page cache (holes, so nothing is written to disk), then
//! maps private anonymous memory advised for huge pages in pieces of [`PIECE_BYTES`], every page
//! written, up to [`ANON_HIGH_BYTES`] while the host's `MemAvailable` stays above
//! [`FLOOR_BYTES`], drops every other piece down to [`ANON_LOW_BYTES`] and reads the file again,
//! until the arm ends. Each command buffer copies [`COPY_BYTES`] from a no-copy buffer over
//! private anonymous memory (imported by its pointer, a userptr on amdgpu) into a `Private`
//! buffer and from that into another, and is submitted on the calling thread
//! (`CommandBuffer::commit_and_submit`, timed) about every millisecond and waited for.
//!
//! The arms, one test each:
//! - **System heap, kept**: [`HELD_BUFFERS`] `Private` buffers of [`HELD_BUFFER_BYTES`], which a
//!   unified-memory device places in its system heap's types first (`memory.rs`, `candidates`),
//!   and [`EMPTY_BUFFERS`] more made and dropped, their blocks kept empty
//!   (`WorkCounts::empty_memory_blocks_live`).
//! - **System heap, given back**: the same with no budget for empty blocks, so those blocks are
//!   freed.
//! - **Carve-out**: the same bytes allocated from the device's first device-local memory type
//!   that is not host-visible, held and unused, beside the copies' buffers alone.
//! - **Nothing held**: the copies' buffers alone.
//! - **System heap, kept, no pointer import**: the first arm's memory, its copies from a
//!   `Private` buffer in place of the no-copy one.
//!
//! Printed (`RECLAIM-SUBMIT`), per arm and phase: the submissions, those of 8 ms or more and of 17
//! ms or more, the median, 99th percentile and slowest `commit_and_submit`; the host's reclaim
//! over the phase (`/proc/vmstat`: pages kswapd and direct reclaim scanned, compaction stalls,
//! allocation stalls, pages swapped out); the blocks the device holds; and the slowest
//! submissions' `CLOCK_MONOTONIC` times, to line up with a kernel trace taken beside the run.
//! Ignored: run it in release under `--cfg m2v_gpu` on Linux, alone on its host (it presses the
//! whole host's memory), one arm at a time: `cargo nextest run --release -p metal2vulkan-runtime
//! --lib --run-ignored all reclaim_submit_bench --no-capture`.

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
/// The held `Private` buffers and their size: 2 GiB, about what a consumer's guest holds in
/// pooled blocks at its desktop (32 to 35 blocks of 64 MiB).
const HELD_BUFFERS: u64 = 64;
const HELD_BUFFER_BYTES: u64 = 32 * MIB;
/// The buffers made and dropped so that their blocks are kept empty: 8 blocks of 64 MiB.
const EMPTY_BUFFERS: u64 = 16;
/// The bytes each command buffer copies, twice.
const COPY_BYTES: u64 = 64 * 1024;
/// The no-copy buffer the copies read.
const NO_COPY_BYTES: u64 = 16 * MIB;
/// The phases' lengths.
const QUIET: Duration = Duration::from_secs(5);
const PRESSED: Duration = Duration::from_secs(20);
/// The gap between submissions.
const EVERY: Duration = Duration::from_millis(1);
/// The page cache the pressing thread fills, from a sparse file.
const HEAT_BYTES: u64 = 8 * GIB;
/// The anonymous memory the pressing thread holds between these bounds.
const ANON_HIGH_BYTES: u64 = 14 * GIB;
const ANON_LOW_BYTES: u64 = 7 * GIB;
const PIECE_BYTES: u64 = 256 * MIB;
/// The least `MemAvailable` at which the pressing thread maps another piece.
const FLOOR_BYTES: u64 = 3 * GIB / 2;
/// The slowest submissions printed per phase.
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
    fn start() -> Pressing {
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = Arc::clone(&stop);
        let thread = std::thread::Builder::new()
            .name("reclaim-press".to_owned())
            .spawn(move || press(&stopping))
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

/// The pressing thread's loop (the module docs), until `stop`.
fn press(stop: &AtomicBool) {
    let heat = Heat::new(HEAT_BYTES);
    heat.read(stop);
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

/// Submit for `length` (the module docs), then print the phase `name` of `arm`.
fn phase(
    device: &Device,
    (queue, source, middle, out): (&CommandQueue, &Buffer, &Buffer, &Buffer),
    (arm, name): (&str, &str),
    length: Duration,
) {
    let before = vmstat();
    let began = Instant::now();
    let mut times: Vec<(Duration, f64)> = Vec::new();
    while began.elapsed() < length {
        let at = monotonic();
        times.push((submit(queue, source, middle, out), at));
        std::thread::sleep(EVERY);
    }
    let after = vmstat();
    let mut sorted: Vec<Duration> = times.iter().map(|&(took, _)| took).collect();
    sorted.sort();
    let ms = |took: Duration| took.as_secs_f64() * 1e3;
    let at = |fraction: f64| ms(sorted[((sorted.len() - 1) as f64 * fraction) as usize]);
    let over = |bound: f64| sorted.iter().filter(|&&took| ms(took) >= bound).count();
    let reclaim: Vec<String> = (VMSTAT.iter().zip(before.iter().zip(after)))
        .map(|(name, (before, after))| format!("{name} {}", after - before))
        .collect();
    let counts = device.work_counts();
    eprintln!(
        "RECLAIM-SUBMIT {} {arm}, {name}: {} submissions, {} of 8 ms or more, {} of 17 ms or \
         more; commit_and_submit ms median {:.3}, p99 {:.3}, slowest {:.3}; reclaim: {}; blocks \
         live {}, empty {}; MemAvailable {} MiB",
        device.name(),
        sorted.len(),
        over(8.0),
        over(17.0),
        at(0.5),
        at(0.99),
        ms(*sorted.last().unwrap()),
        reclaim.join(", "),
        counts.memory_blocks_live,
        counts.empty_memory_blocks_live,
        meminfo("MemAvailable") >> 20,
    );
    times.sort_by(|a, b| b.0.cmp(&a.0));
    let slowest: Vec<String> = (times.iter().take(SLOWEST))
        .map(|&(took, at)| format!("{:.1} ms at {at:.3}", ms(took)))
        .collect();
    eprintln!(
        "RECLAIM-SUBMIT {arm}, {name}: slowest {}",
        slowest.join(", ")
    );
}

/// What an arm holds beside its copies.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Held {
    /// [`HELD_BUFFERS`] `Private` buffers and [`EMPTY_BUFFERS`] made and dropped, their blocks
    /// kept empty where `kept`.
    SystemHeap {
        kept: bool,
    },
    /// The same bytes of the first device-local memory type that is not host-visible.
    CarveOut,
    Nothing,
}

/// The first device-local memory type of `device` that is not host-visible.
fn carve_out_type(device: &Device) -> u32 {
    let memory = crate::device::lock(&device.shared().memory);
    (0..vk::MAX_MEMORY_TYPES as u32)
        .find(|&index| {
            memory.type_flags(index).is_some_and(|flags| {
                flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    && !flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
            })
        })
        .expect("a device-local memory type")
}

/// Run the arm `label`: its memory held as `held`, its copies from a no-copy buffer where
/// `no_copy`.
fn arm(label: &str, held: Held, no_copy: bool) {
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
    let mut buffers = Vec::new();
    let mut raw = Vec::new();
    match held {
        Held::SystemHeap { kept } => {
            if !kept {
                crate::device::lock(&device.shared().memory).kept_budget_stand_in = Some(0);
            }
            buffers = (0..HELD_BUFFERS)
                .map(|_| private(HELD_BUFFER_BYTES))
                .collect();
            let dropped: Vec<Buffer> = (0..EMPTY_BUFFERS)
                .map(|_| private(HELD_BUFFER_BYTES))
                .collect();
            drop(dropped);
        }
        Held::CarveOut => {
            let vk_device = device.shared().vk();
            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(64 * MIB)
                .memory_type_index(carve_out_type(&device));
            let blocks = (HELD_BUFFERS + EMPTY_BUFFERS) * HELD_BUFFER_BYTES / (64 * MIB);
            raw = (0..blocks)
                // SAFETY: the device is live, `info` valid and the type the device's.
                .map(|_| unsafe { crate::creation::allocate_memory(vk_device, &info) }.unwrap())
                .collect();
        }
        Held::Nothing => {}
    }
    let pages = Anonymous::new(NO_COPY_BYTES, false);
    let source = match no_copy {
        // SAFETY: the pages outlive the buffer, which drops first.
        true => unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::new(NonNull::new(pages.start as *mut u8).unwrap()),
                pages.length,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap(),
        false => private(COPY_BYTES),
    };
    let (middle, out) = (private(COPY_BYTES), private(COPY_BYTES));
    let on = (&queue, &source, &middle, &out);
    phase(&device, on, (label, "quiet"), QUIET);
    let pressing = Pressing::start();
    phase(&device, on, (label, "pressed"), PRESSED);
    drop(pressing);
    drop((source, middle, out, buffers));
    let vk_device = device.shared().vk();
    for memory in raw {
        // SAFETY: nothing uses the memory.
        unsafe { crate::release::object(vk_device, memory) };
    }
    drop(pages);
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_system_heap_kept() {
    arm("system heap, kept", Held::SystemHeap { kept: true }, true);
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_system_heap_given_back() {
    arm(
        "system heap, given back",
        Held::SystemHeap { kept: false },
        true,
    );
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_carve_out() {
    arm("carve-out", Held::CarveOut, true);
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_nothing_held() {
    arm("nothing held", Held::Nothing, true);
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_system_heap_kept_no_pointer_import() {
    arm(
        "system heap, kept, no pointer import",
        Held::SystemHeap { kept: true },
        false,
    );
}
