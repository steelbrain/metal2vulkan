//! What a guest's tile traffic costs over its RAM on a device where each listed import costs every
//! submission (RADV, the RAM with a source of dma-bufs: `host_ram.rs`'s `Plan::listed`): each
//! copy's time, and what the copies make resident of the guest's RAM.
//!
//! The RAM is an 8 GiB and then a 16 GiB `memfd` mapped shared, as QEMU's
//! `memory-backend-memfd,share=on` makes it, its source of dma-bufs udmabufs of its pages. The
//! guest's desktop wrote half of it ([`TOUCHED_PERCENT`]), in 2 MiB extents picked at random (a
//! model: a consumer measured 7.9 to 8.0 GiB of a 16 GiB guest resident at its desktop with views
//! alone); the rest of the file was never written. A scroll ([`STEPS`]) makes a tile of
//! [`TILE_PAGES`] of the pages the guest wrote at each step and copies it both ways, a device
//! cache's fill (`copy_from_host_ram`) and its download (`copy_to_host_ram`), each in a command
//! buffer of its own, committed and waited for; one of the visible tiles before it (the last
//! [`VISIBLE`]) is filled and downloaded again. A tile's pages lie as a surface's do in the
//! consumer's guests (the shape `tests/host_ram_copy_bench.rs` copies): about 75% of them the page
//! below the one before (a run of its own), 20% the one above (merged into it), the rest any
//! written page. No object lies over the RAM, so nothing but the copies imports its pages.
//!
//! Printed (`HOST-RAM-LISTED-BENCH`): per RAM, the chunks; every [`REPORT_EVERY`] steps, the bytes
//! of the file in memory (`mincore` over the RAM's mapping) against those the guest wrote, and
//! what whole-chunk imports would have made resident (every page of each chunk a copy touched:
//! a chunk's import pins all of it, and udmabuf allocates each page the guest never wrote); and
//! over the scroll, each way's median, 90th percentile and slowest copy: the encoding thread's
//! CPU time for the call, its wall time, the GPU's (`gpu_end_time - gpu_start_time`) and the
//! commit's return to the completion seen, and the dma-buf imports and chunk imports per copy.
//! Then a 5 MiB device cache's fill and download, [`ITERATIONS`] times each, the medians.
//!
//! Ignored: run it in release under `--cfg m2v_gpu` on Linux, on a host with `/dev/udmabuf`
//! (`cargo nextest run --release -p metal2vulkan-runtime --lib --run-ignored all
//! host_ram_listed_bench --no-capture`). It needs about 9 GiB of memory for the 16 GiB RAM.

use super::host_residency::{device, udmabuf, Scratch};
use crate::{
    Buffer, CommandBufferStatus, CommandQueue, Device, DmaBufChunk, DmaBufRequest, DmaBufSource,
    HostRam, KeepAlive, ResourceOptions,
};
use std::ops::Range;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
const GIB: u64 = 1 << 30;
const PAGE: u64 = 4096;
/// The extents the guest's desktop wrote or left, and the share it wrote.
const EXTENT: u64 = 2 << 20;
const TOUCHED_PERCENT: u64 = 50;
/// A tile's pages: about 7 MiB, as the consumer's views of a tile (1,774 to 1,824 pieces).
const TILE_PAGES: u64 = 1800;
/// The scroll's steps, the tiles visible at once, and how often the residency is printed.
const STEPS: usize = 240;
const VISIBLE: usize = 24;
const REPORT_EVERY: usize = 60;
/// A 5 MiB device cache's copies.
const CACHE_PAGES: u64 = 1280;
const ITERATIONS: usize = 40;

/// A small deterministic generator (xorshift).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// The RAM's file and mapping, unmapped and closed by the keep-alive.
struct Mapping {
    start: u64,
    length: u64,
    _fd: Arc<OwnedFd>,
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the bench's mapping, which no device imports any more: the keep-alive is dropped
        // after the last import over it is freed.
        unsafe { libc::munmap(self.start as *mut libc::c_void, self.length as usize) };
    }
}

/// A RAM of `length` bytes over a new `memfd`, its source udmabufs of its pages at the RAM's
/// addresses, as long as the host's udmabuf limit allows; its first byte.
fn ram(length: u64) -> (Arc<HostRam>, u64) {
    // SAFETY: a new anonymous file, owned at once, then sized.
    let fd = unsafe {
        let fd = libc::memfd_create(c"m2v-listed-bench".as_ptr(), libc::MFD_CLOEXEC);
        assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
        OwnedFd::from_raw_fd(fd)
    };
    // SAFETY: the descriptor is the bench's.
    assert_eq!(
        unsafe { libc::ftruncate(fd.as_raw_fd(), length as libc::off_t) },
        0
    );
    // SAFETY: a new shared mapping of the whole file.
    let start = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd.as_raw_fd(),
            0,
        )
    };
    assert_ne!(start, libc::MAP_FAILED, "mmap of the file");
    let start = start as u64;
    let fd = Arc::new(fd);
    let memfd = Arc::clone(&fd);
    let source: DmaBufSource = Arc::new(move |request: DmaBufRequest| {
        let first = request.start.as_ptr() as u64;
        let fd = udmabuf(&memfd, first - start, request.length)?;
        Ok(vec![DmaBufChunk {
            fd,
            length: request.length,
        }])
    });
    let keep: KeepAlive = Box::new(Mapping {
        start,
        length,
        _fd: fd,
    });
    let at = NonNull::new(start as *mut u8).unwrap();
    // SAFETY: the mapping is readable and writable until the keep-alive unmaps it.
    let ram = unsafe { HostRam::new(&[(at, length)], Some(keep)) }.unwrap();
    let largest = std::fs::read_to_string("/sys/module/udmabuf/parameters/size_limit_mb")
        .ok()
        .and_then(|limit| limit.trim().parse::<u64>().ok())
        .unwrap_or(64)
        .saturating_mul(1 << 20);
    let ram = ram.with_dma_buf(source).with_largest_dma_buf(largest);
    (Arc::new(ram), start)
}

/// The pages of the RAM from `start` (`pages` of them) the guest's desktop wrote: half its
/// extents, picked by `rng`, each page written once.
fn desktop(start: u64, pages: u64, rng: &mut Rng) -> Vec<bool> {
    let per_extent = EXTENT / PAGE;
    let mut written = vec![false; pages as usize];
    for extent in 0..pages / per_extent {
        if rng.next() % 100 >= TOUCHED_PERCENT {
            continue;
        }
        for page in extent * per_extent..(extent + 1) * per_extent {
            written[page as usize] = true;
            // SAFETY: the RAM's mapping, which no device uses yet.
            unsafe { *((start + page * PAGE) as *mut u8) = (page % 251) as u8 };
        }
    }
    written
}

/// A tile of `count` pages the guest wrote, none `taken` before, in the order its bytes take them
/// (the module docs), at the RAM's addresses from `start`; each taken.
fn tile(
    start: u64,
    written: &[bool],
    taken: &mut [bool],
    count: u64,
    rng: &mut Rng,
) -> Vec<Range<u64>> {
    let pages = written.len() as u64;
    let free = |page: u64, taken: &[bool]| written[page as usize] && !taken[page as usize];
    let any = |taken: &[bool], rng: &mut Rng| loop {
        let page = rng.next() % pages;
        if free(page, taken) {
            return page;
        }
    };
    let mut runs = Vec::with_capacity(count as usize);
    let mut at = any(taken, rng);
    for _ in 0..count {
        taken[at as usize] = true;
        runs.push(start + at * PAGE..start + (at + 1) * PAGE);
        let next = match rng.next() % 100 {
            0..75 => at.checked_sub(1),
            75..95 => Some(at + 1).filter(|&next| next < pages),
            _ => None,
        };
        at = match next {
            Some(next) if free(next, taken) => next,
            _ => any(taken, rng),
        };
    }
    runs
}

/// The bytes of the RAM's file in memory: `mincore` over its mapping from `start`.
fn resident_bytes(start: u64, length: u64) -> u64 {
    let mut held = vec![0u8; (length / PAGE) as usize];
    // SAFETY: the bench's mapping, mapped while the RAM lives; one byte a page.
    let answered = unsafe {
        libc::mincore(
            start as *mut libc::c_void,
            length as usize,
            held.as_mut_ptr(),
        )
    };
    assert_eq!(answered, 0, "mincore: {}", std::io::Error::last_os_error());
    held.iter().filter(|&&byte| byte & 1 != 0).count() as u64 * PAGE
}

/// The thread's CPU time.
fn thread_cpu() -> Duration {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a clock the call takes, into a value of the bench's.
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut now) };
    Duration::new(now.tv_sec as u64, now.tv_nsec as u32)
}

/// One way's copies: each copy's CPU time for the call, its wall time, the GPU's time, commit to
/// completion, and its dma-buf and chunk imports.
#[derive(Default)]
struct Way {
    cpu: Vec<Duration>,
    wall: Vec<Duration>,
    gpu: Vec<Duration>,
    completion: Vec<Duration>,
    dma_bufs: Vec<u64>,
    chunks: Vec<u64>,
}

impl Way {
    fn print(&mut self, device: &Device, what: &str) {
        let ms = |values: &mut Vec<Duration>| {
            values.sort();
            let at = |share: usize| values[(values.len() * share / 100).min(values.len() - 1)];
            format!(
                "median {:.3} ms, p90 {:.3} ms, slowest {:.3} ms",
                at(50).as_secs_f64() * 1e3,
                at(90).as_secs_f64() * 1e3,
                values[values.len() - 1].as_secs_f64() * 1e3
            )
        };
        let count = |values: &mut Vec<u64>| {
            values.sort_unstable();
            format!(
                "median {}, slowest {}",
                values[values.len() / 2],
                values[values.len() - 1]
            )
        };
        eprintln!(
            "HOST-RAM-LISTED-BENCH {}: {what} ({} copies): call cpu {}; call wall {}; gpu {}; \
             commit to completion {}; dma-buf imports {}; chunk imports {}",
            device.name(),
            self.cpu.len(),
            ms(&mut self.cpu),
            ms(&mut self.wall),
            ms(&mut self.gpu),
            ms(&mut self.completion),
            count(&mut self.dma_bufs),
            count(&mut self.chunks),
        );
    }
}

/// Copy `runs` of `ram` into `cache` (`gather`) or back, in a command buffer of its own,
/// committed and waited for; its figures into `way`.
fn copy(
    device: &Device,
    queue: &CommandQueue,
    (ram, runs): (&HostRam, &[Range<u64>]),
    cache: &Buffer,
    gather: bool,
    way: &mut Way,
) {
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    let (cpu, wall) = (thread_cpu(), Instant::now());
    match gather {
        true => encoder.copy_from_host_ram(ram, runs, cache, 0),
        false => encoder.copy_to_host_ram(cache, 0, ram, runs),
    }
    .unwrap();
    way.cpu.push(thread_cpu() - cpu);
    way.wall.push(wall.elapsed());
    encoder.end_encoding().unwrap();
    let committed = Instant::now();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    way.completion.push(committed.elapsed());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
    way.gpu.push(Duration::from_secs_f64(gpu.max(0.0)));
    let after = device.work_counts();
    way.dma_bufs
        .push(after.host_ram_dma_buf_imports - before.host_ram_dma_buf_imports);
    way.chunks
        .push(after.host_ram_chunks_imported - before.host_ram_chunks_imported);
}

/// The bytes whole-chunk imports would make resident: those the guest wrote (`written`, the
/// pages of the RAM from `start`), and every other page of each of `spans` (the RAM's chunks) a
/// copy touched (`used`).
fn with_chunks(start: u64, written: &[bool], spans: &[Range<u64>], used: &[bool]) -> u64 {
    let unwritten = (spans.iter().zip(used))
        .filter(|&(_, &used)| used)
        .map(|(span, _)| {
            let pages = (span.start - start) / PAGE..(span.end - start) / PAGE;
            written[pages.start as usize..pages.end as usize]
                .iter()
                .filter(|&&written| !written)
                .count() as u64
        });
    let written = written.iter().filter(|&&written| written).count() as u64;
    (written + unwritten.sum::<u64>()) * PAGE
}

/// One RAM of `length` bytes: the scroll, then the 5 MiB cache's copies (the module docs).
fn scroll(device: &Device, length: u64) {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d ^ length);
    let (ram, start) = ram(length);
    let written = desktop(start, length / PAGE, &mut rng);
    let written_bytes = written.iter().filter(|&&page| page).count() as u64 * PAGE;
    let used = device.use_host_ram(&ram).unwrap();
    let chunks = device.host_ram_chunks();
    let [(listed, spans)] = &chunks[..] else {
        panic!("one RAM: {chunks:?}");
    };
    eprintln!(
        "HOST-RAM-LISTED-BENCH {}: a {} GiB RAM, listed {listed}, {} chunks of {} MiB; the guest \
         wrote {} MiB of it, {} MiB of the file in memory",
        device.name(),
        length / GIB,
        spans.len(),
        (spans[0].end - spans[0].start) >> 20,
        written_bytes >> 20,
        resident_bytes(start, length) >> 20,
    );
    let queue = device.new_command_queue().unwrap();
    let cache = device
        .new_buffer(TILE_PAGES * PAGE, ResourceOptions::private())
        .unwrap();
    let mut taken = vec![false; written.len()];
    let mut chunk_used = vec![false; spans.len()];
    let mut tiles: Vec<Vec<Range<u64>>> = Vec::new();
    let (mut fill, mut download) = (Way::default(), Way::default());
    let began = device.work_counts();
    for step in 1..=STEPS {
        tiles.push(tile(start, &written, &mut taken, TILE_PAGES, &mut rng));
        let again =
            tiles.len().saturating_sub(VISIBLE) + (rng.next() as usize) % tiles.len().min(VISIBLE);
        for index in [tiles.len() - 1, again] {
            let runs = &tiles[index][..];
            for run in runs {
                let slot = spans.partition_point(|span| span.end <= run.start);
                chunk_used[slot] = true;
            }
            copy(device, &queue, (&*ram, runs), &cache, true, &mut fill);
            copy(device, &queue, (&*ram, runs), &cache, false, &mut download);
        }
        // A stop short of the job's memory: copies that pinned whole chunks would make the
        // guest's untouched RAM resident.
        let resident = resident_bytes(start, length);
        assert!(
            resident <= written_bytes + GIB,
            "step {step}: {} MiB of the file in memory, the guest wrote {} MiB",
            resident >> 20,
            written_bytes >> 20
        );
        if step % REPORT_EVERY == 0 {
            let counts = device.work_counts();
            eprintln!(
                "HOST-RAM-LISTED-BENCH {}: {} GiB, step {step}: {} MiB of the file in memory, the \
                 guest wrote {} MiB; whole-chunk imports would hold {} MiB ({} of {} chunks \
                 touched); {} chunks imported, {} live; {} run imports live",
                device.name(),
                length / GIB,
                resident_bytes(start, length) >> 20,
                written_bytes >> 20,
                with_chunks(start, &written, spans, &chunk_used) >> 20,
                chunk_used.iter().filter(|&&used| used).count(),
                spans.len(),
                counts.host_ram_chunks_imported - began.host_ram_chunks_imported,
                counts.host_ram_chunks_live,
                counts.run_imports_live,
            );
        }
    }
    let what = |way: &str| {
        format!(
            "{} GiB scroll, {way} of a {TILE_PAGES}-page tile",
            length / GIB
        )
    };
    fill.print(device, &what("fill"));
    download.print(device, &what("download"));
    // A 5 MiB device cache's copies, over the same tile each time.
    let runs = tile(start, &written, &mut taken, CACHE_PAGES, &mut rng);
    let (mut fill, mut download) = (Way::default(), Way::default());
    for _ in 0..ITERATIONS {
        copy(device, &queue, (&*ram, &runs[..]), &cache, true, &mut fill);
        copy(
            device,
            &queue,
            (&*ram, &runs[..]),
            &cache,
            false,
            &mut download,
        );
    }
    let what = |way: &str| format!("{} GiB, a 5 MiB cache's {way}", length / GIB);
    fill.print(device, &what("fill"));
    download.print(device, &what("download"));
    drop((cache, queue, used, ram));
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn listed_copies_over_8_and_16_gib_guests() {
    if std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/udmabuf")
        .is_err()
    {
        eprintln!("HOST-RAM-LISTED-BENCH: /dev/udmabuf cannot be opened: not run");
        return;
    }
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    if device.shared().host_import.is_err() {
        eprintln!("HOST-RAM-LISTED-BENCH: the device imports no host memory here: not run");
        return;
    }
    for length in [8 * GIB, 16 * GIB] {
        scroll(&device, length);
    }
}
