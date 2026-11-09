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
//! - **Made by the runtime** and **Made by the runtime, settled**: a `Private` buffer of
//!   [`RUNTIME_BUFFER_BYTES`] every [`RUNTIME_EVERY`], at most [`RUNTIME_BUFFERS`] a phase: the
//!   runtime's pools grow, their blocks made ahead where it makes them ahead
//!   (`WorkCounts::buffer_blocks_made_ahead`), allocated exportable where the device's check of
//!   such blocks passed (`memory.rs`'s "Buffers in exportable blocks"); the second settles the
//!   device's in-place check before the phases, as a device of a driver whose check's record
//!   holds its verdicts knows them as it is made, where the first's check runs beside them, as a
//!   driver's first device's does.
//! - **System heap, 64 MiB**, **System heap, 16 MiB** and **System heap, 4 MiB**: allocations of
//!   the device's first memory type that is host-visible and host-coherent but not device-local or
//!   cached (where a unified-memory device's `Private` blocks lie, `memory.rs`, `candidates`), of
//!   64 MiB every [`RAW_EVERY`], or of 16 or 4 MiB as much more often as they are smaller,
//!   [`RAW_BYTES`] a phase.
//! - **System heap, 64 MiB, exportable**, **System heap, 64 MiB, opaque fd** and **System heap,
//!   64 MiB, both**: the system heap's 64 MiB allocations, each exportable
//!   (`VkExportMemoryAllocateInfo`) as a dma-buf, as an opaque file descriptor, or as both.
//! - **Carve-out, 64 MiB**: the same of the first device-local memory type that is not
//!   host-visible.
//! - **Made before**: [`HELD_BUFFERS`] `Private` buffers of [`RUNTIME_BUFFER_BYTES`] made before
//!   the quiet phase, and nothing made in the phases.
//! - **Exportable, held 32**, **Exportable, held 64**, **Opaque fd, held 64** and **Both, held
//!   64**: that many of the system heap's 64 MiB allocations, each exportable as a dma-buf, as an
//!   opaque file descriptor or as both, made before the quiet phase, and nothing made in the phases: what holding memory
//!   that is not private to the device's process costs each submission.
//! - **Nothing made**: the copies' buffers alone.
//!
//! Four arms ask instead whether memory allocated exportable holds one queue's work behind
//! another's, as memory that synchronizes implicitly would: **Two queues, private**, **Two
//! queues, exportable**, **Two queues, opaque fd** and **Two queues, both** each allocate
//! [`TWO_QUEUE_BYTES`] of the system heap, as a plain allocation or exportable as a dma-buf, an
//! opaque file descriptor or both, and bind three buffers in it, each made for those handle
//! types where the memory is exportable (`VUID-vkBindBufferMemory-memory-02726`). For
//! [`TWO_QUEUE_ROUNDS`] rounds, a second queue (one the device reserved for the caller, else a
//! compute queue it made beside its own) fills the first buffer [`LONG_FILLS`] times, and a
//! millisecond later a command queue's `VkQueue` (lent,
//! `CommandQueue::with_vk_queue`) copies [`COPY_BYTES`] from the second into the third, each
//! waited for by its fence: printed, the fills' and the copies' times from submission to their
//! fence, and the rounds whose copy completed while the fills ran. Copies held behind the fills
//! take as long as the fills' rest, and complete after them.
//!
//! **Export facts** prints, for a dma-buf and an opaque file descriptor, what the device answers
//! for the runtime's buffers (`vkGetPhysicalDeviceExternalBufferProperties`, of a buffer of the
//! runtime's usage with and without the device-address usage), for linear and
//! `DRM_FORMAT_MOD_LINEAR` images of [`FACT_FORMATS`] that are sampled and copied, and rendered
//! to, and for sampled images of [`PLANE_FORMATS`] (as `texture/ycbcr_planes.rs` makes them)
//! (`vkGetPhysicalDeviceImageFormatProperties2` with `VkPhysicalDeviceExternalImageFormatInfo`):
//! the result, whether such memory is exportable and only as a dedicated allocation, and the
//! handle types it is compatible with; then the device's in-place check's verdicts for each
//! class, access and row ends, once it settled.
//!
//! Printed (`RECLAIM-SUBMIT`), per arm and phase: the submissions; for `commit_and_submit` and for
//! the turnaround each, those of 8 ms or more and of 17 ms or more, the median, 99th percentile and
//! slowest; the command buffers submitted, those the committing thread submitted
//! (`WorkCounts::batches_submitted_at_commit`) and the time spent in `vkQueueSubmit`
//! (`WorkCounts::batch_submit_nanos`); the makings, their median and slowest, the runtime's
//! blocks made ahead and in place, its buffers' blocks made exportable and plain, and its buffers
//! laid in plain blocks while the check was pending or once it refused such blocks; the host's reclaim over the phase (`/proc/vmstat`: pages kswapd
//! and direct reclaim scanned, compaction stalls, allocation stalls, pages swapped out); the
//! memory the fragmenting took, the host's free memory and the cgroup's room it was sized from;
//! the free blocks of each order before and after the fragmenting and after the phase
//! (`/proc/buddyinfo`); and the slowest submissions', turnarounds' and makings' `CLOCK_MONOTONIC`
//! times, to line up with a kernel trace taken beside the run. Ignored: run it in release under
//! `--cfg m2v_gpu` on Linux, alone on its host (it presses the whole host's memory), one arm at a
//! time, in a cgroup with room for nearly all of the host's free memory: `cargo nextest run
//! --release -p metal2vulkan-runtime --lib --run-ignored all reclaim_submit_bench --no-capture`.

use super::host_residency::{device, Scratch};
use crate::{
    Buffer, CommandBufferStatus, CommandQueue, Device, DeviceKind, DeviceOptions, HostMemory,
    ResourceOptions, ShaderCacheOptions,
};
use ash::vk;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
/// The handle types an arm's memory is exportable as.
type Handle = vk::ExternalMemoryHandleTypeFlags;
const DMA_BUF: Handle = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
const OPAQUE_FD: Handle = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
const BOTH: Handle = Handle::from_raw(OPAQUE_FD.as_raw() | DMA_BUF.as_raw());
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
/// The formats the export facts ask of images.
const FACT_FORMATS: [vk::Format; 6] = [
    vk::Format::R8_UNORM,
    vk::Format::R8G8_UNORM,
    vk::Format::B8G8R8A8_UNORM,
    vk::Format::R8G8B8A8_UNORM,
    vk::Format::A2B10G10R10_UNORM_PACK32,
    vk::Format::R16G16B16A16_SFLOAT,
];
/// The formats of a two-plane 4:2:0 video surface's image (`texture/ycbcr_planes.rs`).
const PLANE_FORMATS: [vk::Format; 2] = [
    vk::Format::G8_B8R8_2PLANE_420_UNORM,
    vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
];
/// The two-queue arms' allocation, the bytes of its first buffer, which the second queue
/// fills, the fills in each of its command buffers, and the rounds.
const TWO_QUEUE_BYTES: u64 = 128 * MIB;
const LONG_BYTES: u64 = 96 * MIB;
const LONG_FILLS: u32 = 8;
const TWO_QUEUE_ROUNDS: usize = 100;

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
    /// `export`'s handle type where given.
    Raw {
        index: u32,
        bytes: u64,
        export: Option<Handle>,
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
                export,
            } => made.raw.push(allocate(device, index, bytes, export)),
            Making::Nothing => unreachable!(),
        }
        times.push((began.elapsed(), at));
        std::thread::sleep(every);
    }
    times
}

/// An allocation of `bytes` of `device`'s memory type `index`, exportable as `export`'s handle
/// type where given.
fn allocate(device: &Device, index: u32, bytes: u64, export: Option<Handle>) -> vk::DeviceMemory {
    let mut exportable =
        vk::ExportMemoryAllocateInfo::default().handle_types(export.unwrap_or_default());
    let mut info = vk::MemoryAllocateInfo::default()
        .allocation_size(bytes)
        .memory_type_index(index);
    if export.is_some() {
        info = info.push_next(&mut exportable);
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
         blocks of the runtime's buffers made exportable {}, made plain {}; buffers in plain \
         blocks awaiting the check {}, by its refusal {}; reclaim: {}; blocks live {}, empty {}; \
         MemAvailable {} MiB",
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
        counts.fd_buffer_blocks_made - earlier.fd_buffer_blocks_made,
        counts.fd_buffer_blocks_plain - earlier.fd_buffer_blocks_plain,
        counts.fd_blocks_awaited - earlier.fd_blocks_awaited,
        counts.fd_blocks_refused - earlier.fd_blocks_refused,
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
    /// The runtime's buffers, made with the device's in-place check settled first where `settled`.
    Runtime {
        settled: bool,
    },
    SystemHeap {
        bytes: u64,
        export: Option<Handle>,
    },
    CarveOut,
    MadeBefore,
    /// That many 64 MiB allocations of the system heap made before the phases, exportable as the
    /// handle type.
    ExportableHeld(usize, Handle),
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
        Arm::Runtime { settled } => {
            if settled {
                let known = device.settle_in_place_checks(WAIT);
                eprintln!("RECLAIM-SUBMIT {label}: the in-place check settled first: {known}");
            }
            Making::Runtime
        }
        Arm::SystemHeap { bytes, export } => Making::Raw {
            index: system_heap(),
            bytes,
            export,
        },
        Arm::CarveOut => Making::Raw {
            index: memory_type(
                &device,
                P::DEVICE_LOCAL,
                P::HOST_VISIBLE | P::DEVICE_COHERENT_AMD | P::PROTECTED,
            ),
            bytes: 64 * MIB,
            export: None,
        },
        Arm::MadeBefore => {
            made.buffers = (0..HELD_BUFFERS)
                .map(|_| private(RUNTIME_BUFFER_BYTES))
                .collect();
            Making::Nothing
        }
        Arm::ExportableHeld(count, handle) => {
            let index = system_heap();
            made.raw = (0..count)
                .map(|_| allocate(&device, index, 64 * MIB, Some(handle)))
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

/// Run the two-queue arm `label` (the module docs), its memory exportable as `export`'s handle
/// type where given.
fn two_queues(label: &str, export: Option<Handle>) {
    use vk::BufferUsageFlags as U;
    let scratch = Scratch::new();
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(scratch.0.clone()),
            ..ShaderCacheOptions::default()
        },
        reserved_queues: 1,
        extra_queues: vec![vk::QueueFlags::COMPUTE],
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    let queue = device.new_command_queue().unwrap();
    // SAFETY: the family is read; nothing is submitted.
    let lent_family = unsafe { queue.with_vk_queue(|_, _, family| family) };
    // SAFETY: the handles are used while the device lives.
    let raw = unsafe { device.raw() };
    // The second queue: one the device reserved for the caller, else a compute queue it made for
    // the caller beside its own (a Radeon 780M's graphics family has one queue, which the round
    // robin takes); the buffers it uses are the caller's, each used by one family alone.
    let reserved =
        (raw.reserved_queues.first()).map(|queue| ("reserved", queue.family, queue.queue));
    let extra =
        (raw.extra_queues.first()).map(|queue| ("extra compute", queue.family, queue.queue));
    let Some((kind, second_family, second)) = reserved.or(extra) else {
        eprintln!("RECLAIM-SUBMIT {label}: the device made no second queue: not run");
        return;
    };
    let vk_device = raw.device;
    let index = memory_type(
        &device,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_CACHED
            | vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD
            | vk::MemoryPropertyFlags::PROTECTED,
    );
    let memory = allocate(&device, index, TWO_QUEUE_BYTES, export);
    // The first buffer the second queue's family alone uses, the other two the lent queue's.
    let regions = [
        (0, LONG_BYTES, U::TRANSFER_DST),
        (LONG_BYTES, COPY_BYTES, U::TRANSFER_SRC),
        (LONG_BYTES + COPY_BYTES, COPY_BYTES, U::TRANSFER_DST),
    ];
    let buffers: Vec<vk::Buffer> = (regions.iter())
        .map(|&(offset, size, usage)| {
            let mut external = vk::ExternalMemoryBufferCreateInfo::default()
                .handle_types(export.unwrap_or_default());
            let mut info = vk::BufferCreateInfo::default().size(size).usage(usage);
            if export.is_some() {
                info = info.push_next(&mut external);
            }
            // SAFETY: the device is live and `info` valid; the buffer lies inside the memory, at
            // an offset of a multiple of 64 KiB, and names the memory's handle type where it has
            // one (`VUID-vkBindBufferMemory-memory-02726`).
            unsafe {
                let buffer = vk_device.create_buffer(&info, None).unwrap();
                vk_device
                    .bind_buffer_memory(buffer, memory, offset)
                    .unwrap();
                buffer
            }
        })
        .collect();
    // SAFETY: the pools, command buffers and fences are this device's, recorded once, each
    // submitted again only once its fence signaled, and destroyed once nothing runs.
    let (long_times, short_times, overlapped) = unsafe {
        let pool = |family| {
            let info = vk::CommandPoolCreateInfo::default().queue_family_index(family);
            vk_device.create_command_pool(&info, None).unwrap()
        };
        let pools = [pool(second_family), pool(lent_family)];
        let record = |pool, commands: &dyn Fn(vk::CommandBuffer)| {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffer =
                device.with_loader_word(|| vk_device.allocate_command_buffers(&info).unwrap())[0];
            let begin = vk::CommandBufferBeginInfo::default();
            vk_device
                .begin_command_buffer(command_buffer, &begin)
                .unwrap();
            commands(command_buffer);
            vk_device.end_command_buffer(command_buffer).unwrap();
            command_buffer
        };
        let long = record(pools[0], &|command_buffer| {
            for fill in 0..LONG_FILLS {
                vk_device.cmd_fill_buffer(command_buffer, buffers[0], 0, vk::WHOLE_SIZE, fill);
            }
        });
        let short = record(pools[1], &|command_buffer| {
            let region = vk::BufferCopy::default().size(COPY_BYTES);
            vk_device.cmd_copy_buffer(command_buffer, buffers[1], buffers[2], &[region]);
        });
        let fence = || (vk_device.create_fence(&vk::FenceCreateInfo::default(), None)).unwrap();
        let fences = [fence(), fence()];
        let wait =
            |fence| (vk_device.wait_for_fences(&[fence], true, WAIT.as_nanos() as u64)).unwrap();
        let (mut long_times, mut short_times, mut overlapped) = (Vec::new(), Vec::new(), 0);
        for _ in 0..TWO_QUEUE_ROUNDS {
            vk_device.reset_fences(&fences).unwrap();
            let long_buffers = [long];
            let submit = vk::SubmitInfo::default().command_buffers(&long_buffers);
            let (long_at, long_began) = (monotonic(), Instant::now());
            (vk_device.queue_submit(second, &[submit], fences[0])).unwrap();
            std::thread::sleep(Duration::from_millis(1));
            let short_buffers = [short];
            let submit = vk::SubmitInfo::default().command_buffers(&short_buffers);
            let (short_at, short_began) = (monotonic(), Instant::now());
            (queue.with_vk_queue(|lent, vk_queue, _| {
                lent.queue_submit(vk_queue, &[submit], fences[1])
            }))
            .unwrap();
            wait(fences[1]);
            short_times.push((short_began.elapsed(), short_at));
            if !vk_device.get_fence_status(fences[0]).unwrap() {
                overlapped += 1;
            }
            wait(fences[0]);
            long_times.push((long_began.elapsed(), long_at));
        }
        for fence in fences {
            vk_device.destroy_fence(fence, None);
        }
        for pool in pools {
            vk_device.destroy_command_pool(pool, None);
        }
        (long_times, short_times, overlapped)
    };
    let (long_median, _, long_slowest, _) = spread(&long_times, SLOWEST);
    let (median, p99, slowest, slow) = spread(&short_times, SLOWEST);
    eprintln!(
        "RECLAIM-SUBMIT {} {label}: {kind} queue of family {}, lent queue of family {}; {} \
         rounds; fills ms median {long_median:.3}, slowest {long_slowest:.3}; copies ms median \
         {median:.3}, p99 {p99:.3}, slowest {slowest:.3}; copies completed while the fills ran \
         {overlapped}",
        device.name(),
        second_family,
        lent_family,
        short_times.len(),
    );
    eprintln!("RECLAIM-SUBMIT {label}: slowest copies {slow}");
    for buffer in buffers {
        // SAFETY: nothing uses the buffer.
        unsafe { vk_device.destroy_buffer(buffer, None) };
    }
    // SAFETY: nothing uses the memory.
    unsafe { crate::release::object(device.shared().vk(), memory) };
}

/// The device's answers for memory exportable as the handle types (the module docs).
#[test]
#[ignore = "a measurement's facts: run on its host by name"]
fn reclaim_submit_bench_export_facts() {
    use vk::ExternalMemoryFeatureFlags as F;
    use vk::ImageUsageFlags as I;
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    // SAFETY: the handles are read while the device lives.
    let raw = unsafe { device.raw() };
    let (instance, physical) = (raw.instance, raw.physical_device);
    let features = |properties: vk::ExternalMemoryProperties| {
        let features = properties.external_memory_features;
        format!(
            "exportable {}, dedicated only {}, compatible {:#x}",
            features.contains(F::EXPORTABLE),
            features.contains(F::DEDICATED_ONLY),
            properties.compatible_handle_types.as_raw()
        )
    };
    for (name, handle) in [("dma-buf", DMA_BUF), ("opaque fd", OPAQUE_FD)] {
        for address in [false, true] {
            let info = vk::PhysicalDeviceExternalBufferInfo::default()
                .usage(crate::buffer::usage(address))
                .handle_type(handle);
            let mut properties = vk::ExternalBufferProperties::default();
            // SAFETY: the physical device is the instance's; `info` and `properties` are valid.
            unsafe {
                instance.get_physical_device_external_buffer_properties(
                    physical,
                    &info,
                    &mut properties,
                )
            };
            eprintln!(
                "RECLAIM-SUBMIT export facts, {name}, buffers with the device-address usage \
                 {address}: {}",
                features(properties.external_memory_properties)
            );
        }
        let copied = I::SAMPLED | I::TRANSFER_SRC | I::TRANSFER_DST;
        let images = (FACT_FORMATS.iter()).flat_map(|&format| {
            [copied, copied | I::COLOR_ATTACHMENT].map(|usage| (format, usage))
        });
        let planes = PLANE_FORMATS.map(|format| (format, I::SAMPLED));
        for (format, usage) in images.chain(planes) {
            for (tiling, tiling_name) in [
                (vk::ImageTiling::LINEAR, "linear"),
                (
                    vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT,
                    "DRM_FORMAT_MOD_LINEAR",
                ),
            ] {
                {
                    let mut external =
                        vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(handle);
                    let mut modifier = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
                        .drm_format_modifier(0)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE);
                    let mut info = vk::PhysicalDeviceImageFormatInfo2::default()
                        .format(format)
                        .ty(vk::ImageType::TYPE_2D)
                        .tiling(tiling)
                        .usage(usage)
                        .push_next(&mut external);
                    if tiling == vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT {
                        info = info.push_next(&mut modifier);
                    }
                    let mut answer = vk::ExternalImageFormatProperties::default();
                    let mut properties =
                        vk::ImageFormatProperties2::default().push_next(&mut answer);
                    // SAFETY: the physical device is the instance's; `info`, `properties` and
                    // their chains are valid.
                    let result = unsafe {
                        instance.get_physical_device_image_format_properties2(
                            physical,
                            &info,
                            &mut properties,
                        )
                    };
                    let answered = match result {
                        Ok(()) => features(answer.external_memory_properties),
                        Err(error) => format!("{error:?}"),
                    };
                    eprintln!(
                        "RECLAIM-SUBMIT export facts, {name}, {tiling_name} images of format {} \
                         and usage {:#x}: {answered}",
                        format.as_raw(),
                        usage.as_raw()
                    );
                }
            }
        }
    }
    let settled = device.settle_in_place_checks(WAIT);
    for class in crate::in_place::PagesClass::ALL {
        for access in crate::in_place::Access::ALL {
            let verdicts = (crate::in_place::RowEnds::ALL)
                .map(|rows| (rows, device.in_place_verdict(class, access, rows)));
            eprintln!(
                "RECLAIM-SUBMIT export facts, the in-place check (settled {settled}): {class:?} \
                 {access:?}: {verdicts:?}"
            );
        }
    }
}

#[test]
#[ignore = "a measurement of the GPU's queues: run alone on its host by name"]
fn reclaim_submit_bench_two_queues_private() {
    two_queues("two queues, private", None);
}

#[test]
#[ignore = "a measurement of the GPU's queues: run alone on its host by name"]
fn reclaim_submit_bench_two_queues_exportable() {
    two_queues("two queues, exportable", Some(DMA_BUF));
}

#[test]
#[ignore = "a measurement of the GPU's queues: run alone on its host by name"]
fn reclaim_submit_bench_two_queues_opaque_fd() {
    two_queues("two queues, opaque fd", Some(OPAQUE_FD));
}

#[test]
#[ignore = "a measurement of the GPU's queues: run alone on its host by name"]
fn reclaim_submit_bench_two_queues_both() {
    two_queues("two queues, both", Some(BOTH));
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_runtime() {
    arm("made by the runtime", Arm::Runtime { settled: false });
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_runtime_settled() {
    arm(
        "made by the runtime, settled",
        Arm::Runtime { settled: true },
    );
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_system_heap_64mib() {
    arm(
        "system heap, 64 MiB",
        Arm::SystemHeap {
            bytes: 64 * MIB,
            export: None,
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
            export: None,
        },
    );
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_system_heap_16mib() {
    arm(
        "system heap, 16 MiB",
        Arm::SystemHeap {
            bytes: 16 * MIB,
            export: None,
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
            export: Some(DMA_BUF),
        },
    );
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_system_heap_64mib_opaque_fd() {
    arm(
        "system heap, 64 MiB, opaque fd",
        Arm::SystemHeap {
            bytes: 64 * MIB,
            export: Some(OPAQUE_FD),
        },
    );
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_system_heap_64mib_both() {
    arm(
        "system heap, 64 MiB, both",
        Arm::SystemHeap {
            bytes: 64 * MIB,
            export: Some(BOTH),
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
    arm("exportable, held 32", Arm::ExportableHeld(32, DMA_BUF));
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_exportable_held_64() {
    arm("exportable, held 64", Arm::ExportableHeld(64, DMA_BUF));
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_opaque_fd_held_64() {
    arm("opaque fd, held 64", Arm::ExportableHeld(64, OPAQUE_FD));
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_both_held_64() {
    arm("both, held 64", Arm::ExportableHeld(64, BOTH));
}

#[test]
#[ignore = "a measurement that presses the host's memory: run alone on its host by name"]
fn reclaim_submit_bench_nothing_made() {
    arm("nothing made", Arm::Nothing);
}
