//! The first fill of a device cache over a fresh guest RAM: a gather of a 24 MiB cache whose
//! pages are thousands of runs scattered over five 64 MiB chunks nothing imported yet, by
//! `BlitCommandEncoder::copy_from_host_ram`, as a guest's first frame at login fills its caches;
//! then the same gather again (the chunks imported by then), and what each way of serving a first
//! fill costs on its own:
//!
//! - **Imports of the runs alone**: a no-copy buffer over each of the cache's runs of a second
//!   fresh mapping of the same layout, the runs less than a gap apart joined (0, 64 KiB, 1 MiB):
//!   their count, the bytes they pin and the time their making took.
//! - **Whole chunks**: the five chunks of a third fresh mapping imported by no-copy buffers, one
//!   after another, then on a thread each.
//! - **Through the CPU**: the runs copied by the CPU into a `Shared` buffer, and that buffer
//!   copied into the cache by the GPU.
//!
//! The RAM's pages are untouched but the cache's, which the bench writes first, as a guest's RAM
//! at login is. Each figure is the median of `ITERATIONS` runs where it repeats. Prints
//! `FIRST-FILL` lines. Ignored: run it in release under `--cfg m2v_gpu`,
//! `cargo test --release -p metal2vulkan-runtime --lib host_ram_first_fill_bench -- --ignored
//! --nocapture`.
//!
//! Beside it, new surfaces filled from a guest's RAM while others are
//! (`host_ram_contended_fill_bench`), and device caches released to fresh guest pages beside a
//! compositor (`host_ram_block_release_bench`).

use crate::{
    Buffer, CommandBufferStatus, CommandQueue, Device, DeviceKind, DeviceOptions, HostMemory,
    HostRam, KeepAlive, ResourceOptions, WorkCounts,
};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
const PAGE: u64 = 4096;
const CHUNK: u64 = 64 << 20;
/// A RAM of 16 chunks.
const RAM_BYTES: u64 = 16 * CHUNK;
/// The cache's bytes: 24 MiB.
const CACHE: u64 = 24 << 20;
/// The chunks the cache's pages lie in.
const CHUNKS: [u64; 5] = [2, 5, 6, 9, 13];
const ITERATIONS: usize = 10;

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

/// A fresh mapping of `RAM_BYTES` of a `memfd` (POSIX shared memory elsewhere), no page of it
/// touched, unmapped by its drop.
struct Fresh {
    start: u64,
    length: u64,
    fd: i32,
}

impl Fresh {
    fn new() -> Fresh {
        Fresh::of(RAM_BYTES)
    }

    /// A fresh mapping of `length` bytes.
    fn of(length: u64) -> Fresh {
        #[cfg(target_os = "linux")]
        // SAFETY: a name and flags the call takes.
        let fd = unsafe { libc::memfd_create(c"m2v-first-fill".as_ptr(), libc::MFD_CLOEXEC) };
        #[cfg(not(target_os = "linux"))]
        let fd = {
            let name = format!("/m2v-first-fill-{}", std::process::id());
            let name = std::ffi::CString::new(name).unwrap();
            // SAFETY: a name and flags the call takes; the name goes at once.
            unsafe {
                let fd = libc::shm_open(
                    name.as_ptr(),
                    libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
                    0o600,
                );
                libc::shm_unlink(name.as_ptr());
                fd
            }
        };
        assert!(fd >= 0);
        // SAFETY: the descriptor was just made.
        assert_eq!(unsafe { libc::ftruncate(fd, length as libc::off_t) }, 0);
        // SAFETY: a fresh shared mapping of the file.
        let start = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };
        assert_ne!(start, libc::MAP_FAILED);
        Fresh {
            start: start as u64,
            length,
            fd,
        }
    }
}

impl Drop for Fresh {
    fn drop(&mut self) {
        // SAFETY: the bench's mapping and descriptor, which nothing imports any more.
        unsafe {
            libc::munmap(self.start as *mut libc::c_void, self.length as usize);
            libc::close(self.fd);
        }
    }
}

/// The cache's runs, as offsets in the RAM, in the cache's order: runs of 1 to 4 pages at pages
/// of [`CHUNKS`] picked at random, apart, until they hold [`CACHE`].
fn cache_runs(seed: u64) -> Vec<Range<u64>> {
    let per_chunk = CHUNK / PAGE;
    let pages = CHUNKS.len() as u64 * per_chunk;
    let mut taken = vec![false; pages as usize];
    let mut rng = Rng(seed);
    let mut runs = Vec::new();
    let mut held = 0;
    while held < CACHE {
        let length = (1 + rng.next() % 4).min((CACHE - held) / PAGE);
        let at = rng.next() % (pages - length);
        if (at..at + length).any(|page| taken[page as usize]) {
            continue;
        }
        // Inside one chunk.
        if at / per_chunk != (at + length - 1) / per_chunk {
            continue;
        }
        for page in at..at + length {
            taken[page as usize] = true;
        }
        let chunk = CHUNKS[(at / per_chunk) as usize];
        let offset = chunk * CHUNK + at % per_chunk * PAGE;
        runs.push(offset..offset + length * PAGE);
        held += length * PAGE;
    }
    runs
}

/// Write the cache's pages of the mapping from `start`, as the guest's CPU does.
fn write_runs(start: u64, runs: &[Range<u64>]) {
    for run in runs {
        // SAFETY: bytes of the bench's mapping, which no command uses now.
        unsafe {
            std::ptr::write_bytes(
                (start + run.start) as *mut u8,
                (run.start / PAGE) as u8 | 1,
                (run.end - run.start) as usize,
            )
        };
    }
}

/// The median of `values`, in milliseconds.
fn median(values: &mut [Duration]) -> f64 {
    values.sort();
    values[values.len() / 2].as_secs_f64() * 1e3
}

/// One command buffer of one blit encoder `encode` fills, committed and waited for: the time the
/// call took, the GPU's time and commit to completion.
fn timed(
    queue: &CommandQueue,
    encode: impl FnOnce(&mut crate::BlitCommandEncoder),
) -> (Duration, Duration, Duration) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    let started = Instant::now();
    encode(&mut encoder);
    let call = started.elapsed();
    encoder.end_encoding().unwrap();
    let committed = Instant::now();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let completion = committed.elapsed();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
    (call, Duration::from_secs_f64(gpu.max(0.0)), completion)
}

/// What a fill's counts moved by.
fn moved(before: &WorkCounts, after: &WorkCounts) -> String {
    let ms = |a: u64, b: u64| (b - a) as f64 / 1e6;
    format!(
        "chunks imported {} (ahead {}) in {:.1} ms, copy import waits {} for {:.1} ms, run imports \
         {}, dispatches {}, regions {}",
        after.host_ram_chunks_imported - before.host_ram_chunks_imported,
        after.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead,
        ms(
            before.host_ram_chunk_import_nanos,
            after.host_ram_chunk_import_nanos
        ),
        after.host_ram_copy_import_waits - before.host_ram_copy_import_waits,
        ms(
            before.host_ram_copy_import_wait_nanos,
            after.host_ram_copy_import_wait_nanos
        ),
        after.run_imports_live as i64 - before.run_imports_live as i64,
        after.host_ram_copy_dispatches - before.host_ram_copy_dispatches,
        after.host_ram_copy_regions - before.host_ram_copy_regions,
    )
}

/// `runs` (by start) with those less than `gap` apart joined.
fn joined(runs: &[Range<u64>], gap: u64) -> Vec<Range<u64>> {
    let mut sorted = runs.to_vec();
    sorted.sort_by_key(|run| run.start);
    let mut out: Vec<Range<u64>> = Vec::new();
    for run in sorted {
        match out.last_mut() {
            Some(last) if run.start <= last.end + gap => last.end = last.end.max(run.end),
            _ => out.push(run),
        }
    }
    out
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn host_ram_first_fill_bench() {
    let options = DeviceOptions {
        kinds: vec![
            DeviceKind::Discrete,
            DeviceKind::Integrated,
            DeviceKind::Virtual,
        ],
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    if device.shared().host_import.is_err() {
        eprintln!("FIRST-FILL: the device imports no host memory here");
        return;
    }
    eprintln!("FIRST-FILL device: {}", device.name());
    let runs = cache_runs(0x2545_f491_4f6c_dd1d);
    eprintln!(
        "FIRST-FILL cache: {} runs, {} bytes, over chunks {CHUNKS:?} of a {} MiB RAM",
        runs.len(),
        runs.iter().map(|run| run.end - run.start).sum::<u64>(),
        RAM_BYTES >> 20,
    );
    let queue = device.new_command_queue().unwrap();
    let cache = device
        .new_buffer(CACHE, ResourceOptions::private())
        .unwrap();
    // The product's first fill over a fresh RAM, then the steady fills.
    let fresh = Fresh::new();
    write_runs(fresh.start, &runs);
    let at = NonNull::new(fresh.start as *mut u8).unwrap();
    let keep: KeepAlive = Box::new(());
    // SAFETY: the mapping is readable and writable until `fresh` goes, after the RAM's use.
    let ram = Arc::new(unsafe { HostRam::new(&[(at, RAM_BYTES)], Some(keep)) }.unwrap());
    let used = device.use_host_ram(&ram).unwrap();
    let host: Vec<Range<u64>> = (runs.iter())
        .map(|run| fresh.start + run.start..fresh.start + run.end)
        .collect();
    let before = device.work_counts();
    let (call, gpu, completion) = timed(&queue, |encoder| {
        encoder.copy_from_host_ram(&ram, &host, &cache, 0).unwrap();
    });
    let after = device.work_counts();
    eprintln!(
        "FIRST-FILL first: call {:.3} ms, gpu {:.3} ms, commit to completion {:.3} ms; {}",
        call.as_secs_f64() * 1e3,
        gpu.as_secs_f64() * 1e3,
        completion.as_secs_f64() * 1e3,
        moved(&before, &after),
    );
    // Wait for the device's thread to settle, then the steady fills.
    std::thread::sleep(Duration::from_secs(2));
    let (mut calls, mut gpus, mut completions) = (Vec::new(), Vec::new(), Vec::new());
    let before = device.work_counts();
    for _ in 0..ITERATIONS {
        let (call, gpu, completion) = timed(&queue, |encoder| {
            encoder.copy_from_host_ram(&ram, &host, &cache, 0).unwrap();
        });
        calls.push(call);
        gpus.push(gpu);
        completions.push(completion);
    }
    let after = device.work_counts();
    eprintln!(
        "FIRST-FILL steady: call {:.3} ms, gpu {:.3} ms, commit to completion {:.3} ms; {}",
        median(&mut calls),
        median(&mut gpus),
        median(&mut completions),
        moved(&before, &after),
    );
    drop(used);
    drop(ram);
    drop(fresh);
    // Imports of the runs alone, over a second fresh mapping of the same layout.
    for gap in [0, 64 << 10, 1 << 20] {
        let fresh = Fresh::new();
        write_runs(fresh.start, &runs);
        let spans = joined(&runs, gap);
        let started = Instant::now();
        let mut buffers: Vec<Buffer> = Vec::with_capacity(spans.len());
        let mut refused = None;
        for span in &spans {
            let pointer = NonNull::new((fresh.start + span.start) as *mut u8).unwrap();
            // SAFETY: pages of the bench's mapping, which outlives the buffers.
            let memory = unsafe { HostMemory::new(pointer) };
            // SAFETY: as above.
            let made = unsafe {
                device.new_buffer_with_bytes_no_copy(
                    memory,
                    span.end - span.start,
                    ResourceOptions::default(),
                    None,
                )
            };
            match made {
                Ok(buffer) => buffers.push(buffer),
                Err(error) => {
                    refused = Some(error);
                    break;
                }
            }
        }
        let made = started.elapsed();
        let pinned: u64 = (spans[..buffers.len()].iter())
            .map(|span| span.end - span.start)
            .sum();
        eprintln!(
            "FIRST-FILL runs alone, joined below {gap} bytes apart: {} of {} imports made, {} MiB, \
             in {:.1} ms{}",
            buffers.len(),
            spans.len(),
            pinned >> 20,
            made.as_secs_f64() * 1e3,
            refused.map_or(String::new(), |error| format!(", then refused: {error}")),
        );
        drop(buffers);
        drop(fresh);
    }
    // Whole chunks of a fresh mapping, one after another and on a thread each.
    for threads in [false, true] {
        let fresh = Fresh::new();
        write_runs(fresh.start, &runs);
        let import = |chunk: u64| {
            let pointer = NonNull::new((fresh.start + chunk * CHUNK) as *mut u8).unwrap();
            // SAFETY: pages of the bench's mapping, which outlives the buffers.
            let memory = unsafe { HostMemory::new(pointer) };
            // SAFETY: as above.
            unsafe {
                device.new_buffer_with_bytes_no_copy(
                    memory,
                    CHUNK,
                    ResourceOptions::default(),
                    None,
                )
            }
            .unwrap()
        };
        let started = Instant::now();
        let buffers: Vec<Buffer> = match threads {
            false => CHUNKS.iter().map(|&chunk| import(chunk)).collect(),
            true => std::thread::scope(|scope| {
                let import = &import;
                let made: Vec<_> = (CHUNKS.iter())
                    .map(|&chunk| scope.spawn(move || import(chunk)))
                    .collect();
                made.into_iter().map(|made| made.join().unwrap()).collect()
            }),
        };
        let made = started.elapsed();
        eprintln!(
            "FIRST-FILL the {} chunks imported whole, {}: {:.1} ms",
            CHUNKS.len(),
            match threads {
                false => "one after another",
                true => "a thread each",
            },
            made.as_secs_f64() * 1e3,
        );
        drop(buffers);
        drop(fresh);
    }
    // Through the CPU: the runs into a `Shared` buffer, which the GPU copies into the cache.
    let fresh = Fresh::new();
    write_runs(fresh.start, &runs);
    let staging = device.new_buffer(CACHE, ResourceOptions::shared()).unwrap();
    let contents = staging.contents().unwrap().as_ptr();
    let (mut copies, mut gpus) = (Vec::new(), Vec::new());
    for _ in 0..ITERATIONS {
        let started = Instant::now();
        let mut at = 0;
        for run in &runs {
            let length = (run.end - run.start) as usize;
            // SAFETY: bytes of the bench's mapping into the staging buffer's, which no command
            // uses now.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (fresh.start + run.start) as *const u8,
                    contents.add(at),
                    length,
                )
            };
            at += length;
        }
        copies.push(started.elapsed());
        let (_, gpu, _) = timed(&queue, |encoder| {
            encoder
                .copy_from_buffer_to_buffer(&staging, 0, &cache, 0, CACHE)
                .unwrap();
        });
        gpus.push(gpu);
    }
    eprintln!(
        "FIRST-FILL through the CPU: the runs copied in {:.3} ms, the GPU's copy {:.3} ms",
        median(&mut copies),
        median(&mut gpus),
    );
}

/// The contended bench's RAM: 64 chunks.
const CONTENDED_BYTES: u64 = 64 * CHUNK;
/// The threads that fill new surfaces at once (three video clips), and the surfaces each fills.
const CLIPS: u64 = 3;
const SURFACES: u64 = 7;
/// A new surface's pages, one run each: a 4K NV12 frame's (12 MiB) and a P010 frame's (24 MiB),
/// as a consumer's guest decoder lays them.
const NV12_RUNS: u64 = 3040;
const P010_RUNS: u64 = 6080;

/// `count` single pages of the three chunks from chunk `first`, distinct, in a random order, as
/// offsets in the RAM.
fn surface_pages(seed: u64, first: u64, count: u64) -> Vec<Range<u64>> {
    let pages = 3 * CHUNK / PAGE;
    let mut taken = vec![false; pages as usize];
    let mut rng = Rng(seed | 1);
    let mut runs = Vec::with_capacity(count as usize);
    while (runs.len() as u64) < count {
        let page = rng.next() % pages;
        if std::mem::replace(&mut taken[page as usize], true) {
            continue;
        }
        let at = first * CHUNK + page * PAGE;
        runs.push(at..at + PAGE);
    }
    runs
}

/// A milliseconds summary of `values`: median, 90th percentile and slowest.
fn spread(values: &mut [Duration]) -> String {
    values.sort();
    let ms = |at: usize| values[at.min(values.len() - 1)].as_secs_f64() * 1e3;
    format!(
        "median {:.3} ms, p90 {:.3} ms, slowest {:.3} ms",
        ms(values.len() / 2),
        ms(values.len() * 9 / 10),
        ms(values.len() - 1)
    )
}

/// New surfaces filled from a guest's RAM while others are: [`CLIPS`] threads each fill
/// [`SURFACES`] new surfaces, alternately a 4K NV12 frame's [`NV12_RUNS`] pages and a P010
/// frame's [`P010_RUNS`], each over three chunks of the RAM no surface took before (so their
/// imports are under way), by one `copy_from_host_ram` into a new `Private` buffer, committed and
/// waited for, as a consumer's decoder fills each new output surface; meanwhile another thread
/// imports and frees a no-copy buffer over 32 MiB of fresh anonymous pages every 20 ms. Printed
/// (`CONTENDED-FILL`): per surface kind, the calls' and the completions' spread, and the copies'
/// counts over the run (staged, import waits and their time, chunk imports and their time).
#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn host_ram_contended_fill_bench() {
    let options = DeviceOptions {
        kinds: vec![
            DeviceKind::Discrete,
            DeviceKind::Integrated,
            DeviceKind::Virtual,
        ],
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    if device.shared().host_import.is_err() {
        eprintln!("CONTENDED-FILL: the device imports no host memory here");
        return;
    }
    let fresh = Fresh::of(CONTENDED_BYTES);
    let at = NonNull::new(fresh.start as *mut u8).unwrap();
    let keep: KeepAlive = Box::new(());
    // SAFETY: the mapping is readable and writable until `fresh` goes, after the RAM's use.
    let ram = Arc::new(unsafe { HostRam::new(&[(at, CONTENDED_BYTES)], Some(keep)) }.unwrap());
    let used = device.use_host_ram(&ram).unwrap();
    // Let the device's thread import the RAM's start, as at a boot's desktop.
    std::thread::sleep(Duration::from_secs(1));
    let before = device.work_counts();
    let done = std::sync::atomic::AtomicBool::new(false);
    let (times, churns) = std::thread::scope(|scope| {
        let clips: Vec<_> = (0..CLIPS)
            .map(|clip| {
                let (device, ram, fresh) = (&device, &ram, &fresh);
                scope.spawn(move || {
                    let queue = device.new_command_queue().unwrap();
                    let mut times = Vec::new();
                    for surface in 0..SURFACES {
                        let p010 = surface % 2 == 1;
                        let count = if p010 { P010_RUNS } else { NV12_RUNS };
                        let first = 1 + (clip * SURFACES + surface) * 3;
                        let seed = 0x2545_f491_4f6c_dd1d ^ (clip << 32 | surface);
                        let runs = surface_pages(seed, first, count);
                        write_runs(fresh.start, &runs);
                        let host: Vec<Range<u64>> = (runs.iter())
                            .map(|run| fresh.start + run.start..fresh.start + run.end)
                            .collect();
                        let cache = device
                            .new_buffer(count * PAGE, ResourceOptions::private())
                            .unwrap();
                        let (call, _, completion) = timed(&queue, |encoder| {
                            encoder.copy_from_host_ram(ram, &host, &cache, 0).unwrap();
                        });
                        times.push((p010, call, completion));
                    }
                    times
                })
            })
            .collect();
        let churn = scope.spawn(|| {
            const BYTES: usize = 32 << 20;
            let mut churns = 0;
            while !done.load(std::sync::atomic::Ordering::Acquire) {
                // SAFETY: a fresh private anonymous mapping, unmapped below after its buffer.
                let pointer = unsafe {
                    libc::mmap(
                        std::ptr::null_mut(),
                        BYTES,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANON,
                        -1,
                        0,
                    )
                };
                assert_ne!(pointer, libc::MAP_FAILED);
                // SAFETY: the mapping's bytes, which nothing else uses.
                unsafe { std::ptr::write_bytes(pointer.cast::<u8>(), 1, BYTES) };
                // SAFETY: the mapping's pages, which outlive the buffer over them.
                let memory = unsafe { HostMemory::new(NonNull::new(pointer.cast()).unwrap()) };
                // SAFETY: the mapping outlives the buffer, dropped before it is unmapped.
                let buffer = unsafe {
                    device.new_buffer_with_bytes_no_copy(
                        memory,
                        BYTES as u64,
                        ResourceOptions::default(),
                        None,
                    )
                };
                drop(buffer);
                // SAFETY: the mapping made above; its import was freed in the release.
                unsafe { libc::munmap(pointer, BYTES) };
                churns += 1;
                std::thread::sleep(Duration::from_millis(20));
            }
            churns
        });
        let times: Vec<_> = clips
            .into_iter()
            .flat_map(|clip| clip.join().unwrap())
            .collect();
        done.store(true, std::sync::atomic::Ordering::Release);
        (times, churn.join().unwrap())
    });
    let after = device.work_counts();
    eprintln!(
        "CONTENDED-FILL device: {}; {CLIPS} clips of {SURFACES} new surfaces, {churns} imports \
         and frees beside them",
        device.name()
    );
    for (p010, name) in [(false, "NV12 3040 runs"), (true, "P010 6080 runs")] {
        let (mut calls, mut completions): (Vec<Duration>, Vec<Duration>) = (times.iter())
            .filter(|time| time.0 == p010)
            .map(|time| (time.1, time.2))
            .unzip();
        eprintln!(
            "CONTENDED-FILL {name}: {} fills; calls {}; commit to completion {}",
            calls.len(),
            spread(&mut calls),
            spread(&mut completions),
        );
    }
    let ms = |a: u64, b: u64| (b - a) as f64 / 1e6;
    eprintln!(
        "CONTENDED-FILL counts: staged {}, import waits {} for {:.1} ms, chunks imported {} \
         (ahead {}) in {:.1} ms, dispatches {}",
        after.host_ram_copies_staged - before.host_ram_copies_staged,
        after.host_ram_copy_import_waits - before.host_ram_copy_import_waits,
        ms(
            before.host_ram_copy_import_wait_nanos,
            after.host_ram_copy_import_wait_nanos
        ),
        after.host_ram_chunks_imported - before.host_ram_chunks_imported,
        after.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead,
        ms(
            before.host_ram_chunk_import_nanos,
            after.host_ram_chunk_import_nanos
        ),
        after.host_ram_copy_dispatches - before.host_ram_copy_dispatches,
    );
    drop(used);
    drop(ram);
    drop(fresh);
}

/// The block-release bench's RAM: 48 chunks.
const RELEASE_BYTES: u64 = 48 * CHUNK;
/// The scatters of each shape and size.
const RELEASES: u64 = 5;
/// The compositor's frame period, and the GPU time its frame's work takes.
const FRAME: Duration = Duration::from_micros(16_667);
const FRAME_GPU: Duration = Duration::from_millis(2);

/// A release's guest pages, as offsets in the RAM: `bytes` of single pages over the three chunks
/// from chunk `first` ([`surface_pages`]), or, `few`, four runs of a quarter of them, 3 MiB more
/// than a run apart in chunk `first` (as many ranges as a copy imports alone, none joined).
fn release_runs(seed: u64, first: u64, bytes: u64, few: bool) -> Vec<Range<u64>> {
    if !few {
        return surface_pages(seed, first, bytes / PAGE);
    }
    let run = bytes / 4;
    (0..4)
        .map(|at| {
            let start = first * CHUNK + at * (run + (3 << 20));
            start..start + run
        })
        .collect()
}

/// Device caches released to fresh guest pages beside a compositor, as a consumer's clip switch
/// releases its blocks: [`RELEASES`] scatters of each shape (single pages over three chunks no
/// copy took, or four runs in one) and size (4 and 24 MiB), each by one `copy_to_host_ram` from
/// a `Private` buffer the GPU filled, committed by `commit_and_submit` and waited for, while
/// another thread commits a frame of [`FRAME_GPU`] of GPU work on a queue of its own every
/// [`FRAME`] and waits for it. Printed (`BLOCK-RELEASE`): per shape and size, the calls', the
/// commits' and the completions' spread; the frames' commit to completion meanwhile; and the
/// copies' counts over the run.
#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn host_ram_block_release_bench() {
    let options = DeviceOptions {
        kinds: vec![
            DeviceKind::Discrete,
            DeviceKind::Integrated,
            DeviceKind::Virtual,
        ],
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    if device.shared().host_import.is_err() {
        eprintln!("BLOCK-RELEASE: the device imports no host memory here");
        return;
    }
    let fresh = Fresh::of(RELEASE_BYTES);
    let at = NonNull::new(fresh.start as *mut u8).unwrap();
    let keep: KeepAlive = Box::new(());
    // SAFETY: the mapping is readable and writable until `fresh` goes, after the RAM's use.
    let ram = Arc::new(unsafe { HostRam::new(&[(at, RELEASE_BYTES)], Some(keep)) }.unwrap());
    let used = device.use_host_ram(&ram).unwrap();
    // The compositor's frame: a fill of a `Private` buffer as long as takes `FRAME_GPU`.
    let queue = device.new_command_queue().unwrap();
    let probe_bytes: u64 = 64 << 20;
    let probe = device
        .new_buffer(probe_bytes, ResourceOptions::private())
        .unwrap();
    let (_, gpu, _) = timed(&queue, |encoder| {
        encoder.fill_buffer(&probe, 0..probe_bytes, 1).unwrap();
    });
    let scale = FRAME_GPU.as_secs_f64() / gpu.as_secs_f64().max(1e-6);
    let frame_bytes = ((probe_bytes as f64 * scale) as u64).clamp(16 << 20, 1 << 30) / PAGE * PAGE;
    drop(probe);
    let target = device
        .new_buffer(frame_bytes, ResourceOptions::private())
        .unwrap();
    // Let the device's thread import the RAM's start, as at a boot's desktop.
    std::thread::sleep(Duration::from_secs(1));
    let before = device.work_counts();
    let done = std::sync::atomic::AtomicBool::new(false);
    let (times, frames) = std::thread::scope(|scope| {
        let compositor = scope.spawn(|| {
            let queue = device.new_command_queue().unwrap();
            let mut frames = Vec::new();
            let mut gpus = Vec::new();
            while !done.load(std::sync::atomic::Ordering::Acquire) {
                let started = Instant::now();
                let (_, gpu, completion) = timed(&queue, |encoder| {
                    encoder.fill_buffer(&target, 0..frame_bytes, 2).unwrap();
                });
                frames.push(completion);
                gpus.push(gpu);
                std::thread::sleep(FRAME.saturating_sub(started.elapsed()));
            }
            (frames, gpus)
        });
        let queue = device.new_command_queue().unwrap();
        let mut times = Vec::new();
        let mut first = 1;
        for few in [false, true] {
            for bytes in [4u64 << 20, 24 << 20] {
                for release in 0..RELEASES {
                    let seed = 0x9e37_79b9_7f4a_7c15 ^ (bytes << 8 | release << 1 | u64::from(few));
                    let runs = release_runs(seed, first, bytes, few);
                    first += if few { 1 } else { 3 };
                    let host: Vec<Range<u64>> = (runs.iter())
                        .map(|run| fresh.start + run.start..fresh.start + run.end)
                        .collect();
                    let cache = device
                        .new_buffer(bytes, ResourceOptions::private())
                        .unwrap();
                    timed(&queue, |encoder| {
                        encoder.fill_buffer(&cache, 0..bytes, 3).unwrap();
                    });
                    let command_buffer = queue.command_buffer().unwrap();
                    let mut encoder = command_buffer.blit_command_encoder().unwrap();
                    let started = Instant::now();
                    encoder.copy_to_host_ram(&cache, 0, &ram, &host).unwrap();
                    let call = started.elapsed();
                    encoder.end_encoding().unwrap();
                    let committed = Instant::now();
                    command_buffer.commit_and_submit().unwrap();
                    let commit = committed.elapsed();
                    command_buffer.wait_until_completed_for(WAIT).unwrap();
                    let completion = committed.elapsed();
                    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
                    // SAFETY: the run's bytes in the mapping, which no command writes any more.
                    let landed = host.iter().all(|run| {
                        unsafe {
                            std::slice::from_raw_parts(
                                run.start as *const u8,
                                (run.end - run.start) as usize,
                            )
                        }
                        .iter()
                        .all(|&byte| byte == 3)
                    });
                    assert!(landed, "the release's bytes in the pages");
                    times.push((few, bytes, call, commit, completion));
                    // Between releases, as a clip switch's come.
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }
        done.store(true, std::sync::atomic::Ordering::Release);
        (times, compositor.join().unwrap())
    });
    let after = device.work_counts();
    let (mut frames, mut gpus) = frames;
    eprintln!(
        "BLOCK-RELEASE device: {}; a frame of {} MiB filled every {:.1} ms beside, its GPU time {}",
        device.name(),
        frame_bytes >> 20,
        FRAME.as_secs_f64() * 1e3,
        spread(&mut gpus),
    );
    for few in [false, true] {
        for bytes in [4u64 << 20, 24 << 20] {
            let (mut calls, mut commits, mut completions) = (Vec::new(), Vec::new(), Vec::new());
            for time in times.iter().filter(|time| time.0 == few && time.1 == bytes) {
                calls.push(time.2);
                commits.push(time.3);
                completions.push(time.4);
            }
            let shape = match few {
                true => "4 runs",
                false => "single pages over 3 chunks",
            };
            eprintln!(
                "BLOCK-RELEASE {} MiB, {shape}: {} scatters; calls {}; commits {}; commit to \
                 completion {}",
                bytes >> 20,
                calls.len(),
                spread(&mut calls),
                spread(&mut commits),
                spread(&mut completions),
            );
        }
    }
    eprintln!(
        "BLOCK-RELEASE frames: {} beside; commit to completion {}",
        frames.len(),
        spread(&mut frames)
    );
    let ms = |a: u64, b: u64| (b - a) as f64 / 1e6;
    eprintln!(
        "BLOCK-RELEASE counts: run imports made {}, import waits {} for {:.1} ms, chunks imported \
         {} (ahead {}) in {:.1} ms, gathers staged {}",
        after.run_imports_made - before.run_imports_made,
        after.host_ram_copy_import_waits - before.host_ram_copy_import_waits,
        ms(
            before.host_ram_copy_import_wait_nanos,
            after.host_ram_copy_import_wait_nanos
        ),
        after.host_ram_chunks_imported - before.host_ram_chunks_imported,
        after.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead,
        ms(
            before.host_ram_chunk_import_nanos,
            after.host_ram_chunk_import_nanos
        ),
        after.host_ram_copies_staged - before.host_ram_copies_staged,
    );
    drop(used);
    drop(ram);
    drop(fresh);
}
