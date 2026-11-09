//! What a guest's tile traffic costs over its RAM on a device where each listed import costs every
//! submission (RADV, the RAM with a source of dma-bufs: `host_ram.rs`'s `Plan::listed`), the RAM
//! with a source of run lists (`HostRam::with_dma_buf_of_runs`): each copy's time, and what the
//! copies make resident of the guest's RAM.
//!
//! The RAM is an 8 GiB and then a 16 GiB `memfd`, sealed and mapped shared, as QEMU's
//! `memory-backend-memfd,share=on` makes it, its source of dma-bufs udmabufs of its pages and its
//! source of run lists udmabuf's lists of them (`UDMABUF_CREATE_LIST`, at most the host's
//! `list_limit` runs a list). The guest's desktop wrote half of it ([`TOUCHED_PERCENT`]), in 2 MiB
//! extents picked at random (a model: a consumer measured 7.9 to 8.0 GiB of a 16 GiB guest
//! resident at its desktop with views alone); the rest of the file was never written. A scroll
//! ([`STEPS`]) makes a tile of [`TILE_PAGES`] of the pages the guest wrote at each step and copies
//! it both ways, a device cache's fill (`copy_from_host_ram`) and its download
//! (`copy_to_host_ram`), each in a command buffer of its own, committed and waited for; then one
//! of the visible tiles (the last [`VISIBLE`]) is filled and downloaded again, the repeated
//! copies. A tile's pages lie as a surface's do in the consumer's guests (the shape
//! `tests/host_ram_copy_bench.rs` copies, [`Shape::Surface`]): about 75% of them the page below
//! the one before (a run of its own), 20% the one above (merged into it), the rest any written
//! page; then a shorter scroll ([`SCATTERED_STEPS`]) of tiles whose pages are each any written
//! page ([`Shape::Scattered`]), past the most sorted runs a consumer measured of a real tile
//! (1,477 of 1,800). No object lies over the RAM, so nothing but the copies imports its pages.
//!
//! Printed (`HOST-RAM-LISTED-BENCH`): per RAM, the chunks; every [`REPORT_EVERY`] steps, the bytes
//! of the file in memory (`mincore` over the RAM's mapping) against those the guest wrote, and
//! what whole-chunk imports would have made resident (every page of each chunk a copy touched:
//! a chunk's import pins all of it, and udmabuf allocates each page the guest never wrote); and
//! over each scroll, each way's median, 90th percentile and slowest copy, first and repeated: the
//! encoding thread's CPU time for the call, its wall time, the GPU's (`gpu_end_time -
//! gpu_start_time`) and the commit's return to the completion seen, and the sorted runs, the
//! dma-buf imports, the lists imported and the chunk imports per copy. Then a 5 MiB device
//! cache's fill and download over the same pages [`ITERATIONS`] times each, the first apart; and
//! first the same over a RAM of one 512 MiB chunk an object had imported, where a copy imports
//! nothing.
//!
//! Ignored: run it in release under `--cfg m2v_gpu` on Linux, on a host with `/dev/udmabuf`
//! (`cargo nextest run --release -p metal2vulkan-runtime --lib --run-ignored all
//! host_ram_listed_bench --no-capture`). It needs about 9 GiB of memory for the 16 GiB RAM.

use super::host_residency::{device, udmabuf, udmabuf_list, Scratch};
use crate::{
    Buffer, CommandBufferStatus, CommandQueue, Device, DmaBufChunk, DmaBufRequest,
    DmaBufRunsSource, DmaBufSource, HostMemory, HostRam, KeepAlive, ResourceOptions, RunResolver,
};
use std::ops::Range;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
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
/// The scroll's steps, the scattered scroll's, the tiles visible at once, and how often the
/// residency is printed.
const STEPS: usize = 240;
const SCATTERED_STEPS: usize = 60;
const VISIBLE: usize = 24;
const REPORT_EVERY: usize = 60;
/// A 5 MiB device cache's copies, and the RAM of one chunk they are copied over once imported.
const CACHE_PAGES: u64 = 1280;
const ITERATIONS: usize = 40;
const CHUNK_RAM: u64 = 512 << 20;

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

/// A udmabuf module parameter of this host (`/sys/module/udmabuf/parameters`), or `default`.
fn udmabuf_parameter(name: &str, default: u64) -> u64 {
    std::fs::read_to_string(format!("/sys/module/udmabuf/parameters/{name}"))
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

/// A RAM of `length` bytes over a new `memfd` sealed against growing and shrinking (as QEMU's
/// `memory-backend-memfd` seals it, and as udmabuf requires), its source udmabufs of its pages at
/// the RAM's addresses, as long as the host's udmabuf limit allows, and its source of run lists
/// udmabuf's lists, of at most the host's `list_limit` runs; its first byte.
pub(super) fn ram(length: u64) -> (Arc<HostRam>, u64) {
    ram_timed(length, Arc::default())
}

/// [`ram`], the nanoseconds its source of run lists spent making them added to `listing_nanos`.
pub(super) fn ram_timed(length: u64, listing_nanos: Arc<AtomicU64>) -> (Arc<HostRam>, u64) {
    // SAFETY: a new anonymous file, owned at once, then sized and sealed.
    let fd = unsafe {
        let fd = libc::memfd_create(
            c"m2v-listed-bench".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        );
        assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
        OwnedFd::from_raw_fd(fd)
    };
    // SAFETY: the descriptor is the bench's.
    unsafe {
        assert_eq!(libc::ftruncate(fd.as_raw_fd(), length as libc::off_t), 0);
        let seals = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
        assert_eq!(libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals), 0);
    }
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
    let memfd = Arc::clone(&fd);
    let listing: DmaBufRunsSource = Arc::new(move |requests: &[DmaBufRequest]| {
        let began = Instant::now();
        let runs: Vec<(u64, u64)> = (requests.iter())
            .map(|request| (request.start.as_ptr() as u64 - start, request.length))
            .collect();
        let fd = udmabuf_list(&memfd, &runs);
        let took = u64::try_from(began.elapsed().as_nanos()).unwrap_or(u64::MAX);
        listing_nanos.fetch_add(took, Ordering::Relaxed);
        let length = runs.iter().map(|&(_, length)| length).sum();
        Ok(DmaBufChunk { fd: fd?, length })
    });
    let keep: KeepAlive = Box::new(Mapping {
        start,
        length,
        _fd: fd,
    });
    let at = NonNull::new(start as *mut u8).unwrap();
    // SAFETY: the mapping is readable and writable until the keep-alive unmaps it.
    let ram = unsafe { HostRam::new(&[(at, length)], Some(keep)) }.unwrap();
    let largest = udmabuf_parameter("size_limit_mb", 64).saturating_mul(1 << 20);
    let most_runs = udmabuf_parameter("list_limit", 1024) as usize;
    let ram = (ram.with_dma_buf(source))
        .with_largest_dma_buf(largest)
        .with_dma_buf_of_runs(listing, most_runs);
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

/// How a tile's pages lie (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// As a surface's in the consumer's guests: about 75% of the pages the one below the page
    /// before, 20% the one above, the rest any written page.
    Surface,
    /// Each page any written page.
    Scattered,
}

/// A tile of `count` pages the guest wrote, none `taken` before, in the order its bytes take them
/// (the module docs), lying as `shape` says, at the RAM's addresses from `start`; each taken.
fn tile(
    start: u64,
    written: &[bool],
    taken: &mut [bool],
    (count, shape): (u64, Shape),
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
        let next = match (shape, rng.next() % 100) {
            (Shape::Scattered, _) => None,
            (Shape::Surface, 0..75) => at.checked_sub(1),
            (Shape::Surface, 75..95) => Some(at + 1).filter(|&next| next < pages),
            (Shape::Surface, _) => None,
        };
        at = match next {
            Some(next) if free(next, taken) => next,
            _ => any(taken, rng),
        };
    }
    runs
}

/// The runs a copy of `runs` lists over a chunk not imported: by address, those that meet one.
fn sorted_runs(runs: &[Range<u64>]) -> u64 {
    let mut sorted: Vec<(u64, u64)> = runs.iter().map(|run| (run.start, run.end)).collect();
    sorted.sort_unstable();
    let mut count = 0;
    let mut end: Option<u64> = None;
    for (start, run_end) in sorted {
        if end.is_none_or(|end| start > end) {
            count += 1;
        }
        end = Some(end.map_or(run_end, |end| end.max(run_end)));
    }
    count
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
pub(super) fn thread_cpu() -> Duration {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a clock the call takes, into a value of the bench's.
    unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut now) };
    Duration::new(now.tv_sec as u64, now.tv_nsec as u32)
}

/// One way's copies: each copy's CPU time for the call, its wall time, the GPU's time, commit to
/// completion, its sorted runs, and its dma-buf, list and chunk imports.
#[derive(Default)]
struct Way {
    cpu: Vec<Duration>,
    wall: Vec<Duration>,
    gpu: Vec<Duration>,
    completion: Vec<Duration>,
    runs: Vec<u64>,
    dma_bufs: Vec<u64>,
    lists: Vec<u64>,
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
             commit to completion {}; sorted runs {}; dma-buf imports {}; lists imported {}; \
             chunk imports {}",
            device.name(),
            self.cpu.len(),
            ms(&mut self.cpu),
            ms(&mut self.wall),
            ms(&mut self.gpu),
            ms(&mut self.completion),
            count(&mut self.runs),
            count(&mut self.dma_bufs),
            count(&mut self.lists),
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
    way.runs.push(sorted_runs(runs));
    way.dma_bufs
        .push(after.host_ram_dma_buf_imports - before.host_ram_dma_buf_imports);
    way.lists
        .push(after.host_ram_run_list_imports - before.host_ram_run_list_imports);
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

/// A scroll's copies: each way's first copy of a tile, and its repeated ones.
#[derive(Default)]
struct Scroll {
    fill: Way,
    download: Way,
    fill_again: Way,
    download_again: Way,
}

impl Scroll {
    fn print(&mut self, device: &Device, what: &str) {
        let ways = [
            (&mut self.fill, "fill"),
            (&mut self.download, "download"),
            (&mut self.fill_again, "repeated fill"),
            (&mut self.download_again, "repeated download"),
        ];
        for (way, name) in ways {
            way.print(device, &format!("{what}, {name}"));
        }
    }
}

/// The RAM a scroll copies over: from `start`, the pages the guest wrote, the chunks, and the
/// bytes it wrote; the pages tiles took and the chunks copies touched so far.
struct Guest<'a> {
    ram: &'a HostRam,
    start: u64,
    length: u64,
    written: &'a [bool],
    written_bytes: u64,
    spans: &'a [Range<u64>],
    taken: Vec<bool>,
    chunk_used: Vec<bool>,
}

impl Guest<'_> {
    /// `steps` steps of tiles lying as `shape` says, each copied both ways and one visible tile
    /// copied again (the module docs), over `cache` on `queue`.
    fn scroll(
        &mut self,
        (device, queue, cache): (&Device, &CommandQueue, &Buffer),
        (steps, shape): (usize, Shape),
        rng: &mut Rng,
    ) -> Scroll {
        let mut tiles: Vec<Vec<Range<u64>>> = Vec::new();
        let mut copies = Scroll::default();
        let began = device.work_counts();
        for step in 1..=steps {
            let made = tile(
                self.start,
                self.written,
                &mut self.taken,
                (TILE_PAGES, shape),
                rng,
            );
            tiles.push(made);
            let visible = tiles.len().min(VISIBLE);
            let again = tiles.len() - visible + (rng.next() as usize) % visible;
            for (index, repeated) in [(tiles.len() - 1, false), (again, true)] {
                let runs = &tiles[index][..];
                for run in runs {
                    let slot = self.spans.partition_point(|span| span.end <= run.start);
                    self.chunk_used[slot] = true;
                }
                let (fill, download) = match repeated {
                    false => (&mut copies.fill, &mut copies.download),
                    true => (&mut copies.fill_again, &mut copies.download_again),
                };
                copy(device, queue, (self.ram, runs), cache, true, fill);
                copy(device, queue, (self.ram, runs), cache, false, download);
            }
            // A stop short of the job's memory: copies that pinned whole chunks would make the
            // guest's untouched RAM resident.
            let resident = resident_bytes(self.start, self.length);
            assert!(
                resident <= self.written_bytes + GIB,
                "{shape:?} step {step}: {} MiB of the file in memory, the guest wrote {} MiB",
                resident >> 20,
                self.written_bytes >> 20
            );
            if step % REPORT_EVERY == 0 {
                let counts = device.work_counts();
                eprintln!(
                    "HOST-RAM-LISTED-BENCH {}: {} GiB, {shape:?} step {step}: {} MiB of the file \
                     in memory, the guest wrote {} MiB; whole-chunk imports would hold {} MiB ({} \
                     of {} chunks touched); {} chunks imported, {} live; {} lists imported; {} \
                     run imports live",
                    device.name(),
                    self.length / GIB,
                    resident >> 20,
                    self.written_bytes >> 20,
                    with_chunks(self.start, self.written, self.spans, &self.chunk_used) >> 20,
                    self.chunk_used.iter().filter(|&&used| used).count(),
                    self.spans.len(),
                    counts.host_ram_chunks_imported - began.host_ram_chunks_imported,
                    counts.host_ram_chunks_live,
                    counts.host_ram_run_list_imports - began.host_ram_run_list_imports,
                    counts.run_imports_live,
                );
            }
        }
        copies
    }
}

/// A 5 MiB device cache's copies over the same `runs`, [`ITERATIONS`] times each way: the first
/// of each way, and the repeated ones.
fn cache_copies(
    (device, queue, cache): (&Device, &CommandQueue, &Buffer),
    (ram, runs): (&HostRam, &[Range<u64>]),
) -> Scroll {
    let mut copies = Scroll::default();
    for iteration in 0..ITERATIONS {
        let (fill, download) = match iteration {
            0 => (&mut copies.fill, &mut copies.download),
            _ => (&mut copies.fill_again, &mut copies.download_again),
        };
        copy(device, queue, (ram, runs), cache, true, fill);
        copy(device, queue, (ram, runs), cache, false, download);
    }
    copies
}

/// One RAM of `length` bytes: the scrolls, then the 5 MiB cache's copies (the module docs).
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
    let mut guest = Guest {
        ram: &ram,
        start,
        length,
        written: &written,
        written_bytes,
        spans,
        taken: vec![false; written.len()],
        chunk_used: vec![false; spans.len()],
    };
    let on = (device, &queue, &cache);
    for (steps, shape) in [(STEPS, Shape::Surface), (SCATTERED_STEPS, Shape::Scattered)] {
        let mut copies = guest.scroll(on, (steps, shape), &mut rng);
        let what = format!(
            "{} GiB {shape:?} scroll of a {TILE_PAGES}-page tile",
            length / GIB
        );
        copies.print(device, &what);
    }
    // A 5 MiB device cache's copies, over the same pages each time.
    let runs = tile(
        start,
        &written,
        &mut guest.taken,
        (CACHE_PAGES, Shape::Surface),
        &mut rng,
    );
    let mut copies = cache_copies(on, (&*ram, &runs[..]));
    copies.print(device, &format!("{} GiB, a 5 MiB cache", length / GIB));
    eprintln!(
        "HOST-RAM-LISTED-BENCH {}: {} GiB at the end: {} MiB of the file in memory, the guest \
         wrote {} MiB",
        device.name(),
        length / GIB,
        resident_bytes(start, length) >> 20,
        written_bytes >> 20,
    );
    drop(guest);
    drop((cache, queue, used, ram));
}

/// The 5 MiB cache's copies over a RAM of one chunk ([`CHUNK_RAM`]) that an object over it had
/// imported: what each copy costs with no import of its own, as a copy over a RAM's imported
/// chunks does.
fn over_an_imported_chunk(device: &Device) {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let (ram, start) = ram(CHUNK_RAM);
    let written = vec![true; (CHUNK_RAM / PAGE) as usize];
    for page in 0..CHUNK_RAM / PAGE {
        // SAFETY: the RAM's mapping, which no device uses yet.
        unsafe { *((start + page * PAGE) as *mut u8) = (page % 251) as u8 };
    }
    let used = device.use_host_ram(&ram).unwrap();
    let before = device.work_counts();
    // A buffer of the RAM's first page over its own addresses: bound in its run's import, the
    // chunk asked for, which the device's thread imports.
    let runs: RunResolver = Arc::new(|at, length| Some(std::iter::once(at..at + length).collect()));
    // SAFETY: the RAM's own bytes, mapped until its keep-alive goes, after the buffer.
    let memory =
        unsafe { HostMemory::new(NonNull::new(start as *mut u8).unwrap()) }.with_runs(runs);
    // SAFETY: as above.
    let object = unsafe {
        device.new_buffer_with_bytes_no_copy(memory, PAGE, ResourceOptions::default(), None)
    }
    .unwrap();
    let began = Instant::now();
    while device.work_counts().host_ram_chunks_imported == before.host_ram_chunks_imported {
        assert!(
            began.elapsed() < WAIT,
            "the object's chunk was never imported"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let queue = device.new_command_queue().unwrap();
    let cache = device
        .new_buffer(CACHE_PAGES * PAGE, ResourceOptions::private())
        .unwrap();
    let mut taken = vec![false; written.len()];
    let tile = tile(
        start,
        &written,
        &mut taken,
        (CACHE_PAGES, Shape::Surface),
        &mut rng,
    );
    let mut copies = cache_copies((device, &queue, &cache), (&*ram, &tile[..]));
    copies.print(device, "a 5 MiB cache over an imported chunk");
    drop((cache, queue, object, used, ram));
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
    over_an_imported_chunk(&device);
    for length in [8 * GIB, 16 * GIB] {
        scroll(&device, length);
    }
}
