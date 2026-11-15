//! What a download into a guest RAM costs for each of the RAM's chunks its pages lie in: a
//! `BlitCommandEncoder::copy_to_host_ram` of a surface's whole pages (`host_ram/copy.rs`) from a
//! `Private` buffer, its pages drawn at random from the first 1, 12 or all 128 chunks of an 8 GiB
//! RAM (fresh `memfd` RAM in 4 KiB pages, mapped shared, as a VMM's guest RAM is, never touched
//! by the bench), on a device as the product makes it, whose thread imports the chunks the copies
//! ask for.
//!
//! Two surfaces: a page, and a tile of [`TILE_PAGES`] pages (an 800x512 `BGRA8` one, 1.6 MB). Each
//! case first copies until a copy took the chunks' imports (no staging through the CPU, no wait
//! for an import), then times [`ITERATIONS`] command buffers, each the copy alone, waited for
//! before the next: the encoding thread's CPU time for the call, the GPU's time (the command
//! buffer's timestamps), and the commit's return to the completion seen; with the chunks the
//! pages lie in, the chunks the device holds imported, and the dispatches a copy recorded. The
//! page and the tile in one chunk run twice, before and after the tile over every chunk had all
//! of them imported, so a cost of the chunks held stands apart from one of the chunks a copy
//! touches. For scale: a `Private` to `Shared` blit of each surface's bytes, and an empty blit
//! encoder's command buffer. Prints `HOST-RAM-SPREAD` lines, then each chunk's cost: the tile's
//! figures over every chunk less its figures in one, over the chunks between.
//!
//! Ignored: run it in release, alone on its device,
//! `cargo test --release -p metal2vulkan-runtime --lib host_ram_spread_bench -- --ignored
//! --nocapture` under `--cfg m2v_gpu` (or `--cfg m2v_cpu_icd`, lavapipe: our recording's share).

use super::host_ram_device::{ahead_device, fresh_ram_in, RamPages};
use crate::{Buffer, CommandBufferStatus, CommandQueue, HostRam, ResourceOptions, WorkCounts};
use std::collections::BTreeSet;
use std::ops::Range;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
const PAGE: u64 = 4096;
const CHUNK: u64 = 64 << 20;
const CHUNKS: u64 = 128;
const RAM_BYTES: u64 = CHUNKS * CHUNK;
/// An 800x512 surface of 4-byte texels, in whole pages.
const TILE_PAGES: u64 = (800 * 512 * 4) / PAGE;
const ITERATIONS: usize = 30;
/// The longest a case's warm-up copies for before its chunks are imported.
const WARM: Duration = Duration::from_secs(120);

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

/// `pages` distinct pages drawn from the first `chunks` chunks of the RAM from `start`, a run
/// each, in the order drawn.
fn drawn(start: u64, pages: u64, chunks: u64, rng: &mut Rng) -> Vec<Range<u64>> {
    let within = chunks * CHUNK / PAGE;
    let mut taken = BTreeSet::new();
    let mut runs = Vec::with_capacity(pages as usize);
    while (runs.len() as u64) < pages {
        let page = rng.next() % within;
        if taken.insert(page) {
            let at = start + page * PAGE;
            runs.push(at..at + PAGE);
        }
    }
    runs
}

/// The chunks `runs` lie in.
fn chunks_of(start: u64, runs: &[Range<u64>]) -> usize {
    let chunks: BTreeSet<u64> = runs.iter().map(|run| (run.start - start) / CHUNK).collect();
    chunks.len()
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

/// The median of `values`, in microseconds.
fn median_us(values: &mut [Duration]) -> f64 {
    values.sort();
    values[values.len() / 2].as_secs_f64() * 1e6
}

/// One case's medians, in microseconds: the call's CPU time, the GPU's, commit to completion.
#[derive(Clone, Copy, Debug)]
struct Medians {
    call: f64,
    gpu: f64,
    completion: f64,
}

/// Commit a command buffer whose one blit encoder `encode` fills (its call's CPU time returned by
/// it), wait for it, and give its call's CPU time, the GPU's and commit to completion.
fn timed(
    queue: &CommandQueue,
    encode: impl FnOnce(&mut crate::BlitCommandEncoder) -> Duration,
) -> [Duration; 3] {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    let call = encode(&mut encoder);
    encoder.end_encoding().unwrap();
    let committed = Instant::now();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let completion = committed.elapsed();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
    [call, Duration::from_secs_f64(gpu.max(0.0)), completion]
}

/// [`ITERATIONS`] command buffers of `encode`, timed: their medians.
fn medians(
    queue: &CommandQueue,
    mut encode: impl FnMut(&mut crate::BlitCommandEncoder) -> Duration,
) -> Medians {
    let (mut call, mut gpu, mut completion) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..ITERATIONS {
        let [c, g, t] = timed(queue, &mut encode);
        call.push(c);
        gpu.push(g);
        completion.push(t);
    }
    Medians {
        call: median_us(&mut call),
        gpu: median_us(&mut gpu),
        completion: median_us(&mut completion),
    }
}

/// Whether a copy between `before` and `after` took the chunks' imports: none staged through the
/// CPU, none waited for an import.
fn through_imports(before: &WorkCounts, after: &WorkCounts) -> bool {
    after.host_ram_scatters_staged == before.host_ram_scatters_staged
        && after.host_ram_copy_import_waits == before.host_ram_copy_import_waits
}

/// The download of `runs` from `cache` into `ram`, copied until a copy took the chunks' imports,
/// then timed: its line, and its medians.
fn download(
    device: &crate::Device,
    queue: &CommandQueue,
    (ram, start): (&HostRam, u64),
    (cache, runs): (&Buffer, &[Range<u64>]),
    what: &str,
) -> Medians {
    let copy = |encoder: &mut crate::BlitCommandEncoder| {
        let started = thread_cpu();
        encoder.copy_to_host_ram(cache, 0, ram, runs).unwrap();
        thread_cpu() - started
    };
    let warmed = Instant::now();
    let mut warm = 0usize;
    loop {
        let before = device.work_counts();
        timed(queue, copy);
        warm += 1;
        if through_imports(&before, &device.work_counts()) {
            break;
        }
        assert!(
            warmed.elapsed() < WARM,
            "{what}: its chunks were never imported"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    let before = device.work_counts();
    let figures = medians(queue, copy);
    let after = device.work_counts();
    assert!(
        through_imports(&before, &after),
        "{what}: a timed copy was staged"
    );
    let copies = after.host_ram_copies - before.host_ram_copies;
    let dispatches = after.host_ram_copy_dispatches - before.host_ram_copy_dispatches;
    let regions = after.host_ram_copy_regions - before.host_ram_copy_regions;
    eprintln!(
        "HOST-RAM-SPREAD {what}: {} runs over {} chunks ({} imported), {} dispatches and {} \
         regions a copy ({copies} copies, {warm} to warm in {:.2} s): call cpu {:.1} us, gpu \
         {:.1} us, commit to completion {:.1} us",
        runs.len(),
        chunks_of(start, runs),
        after.host_ram_chunks_live,
        dispatches as f64 / copies.max(1) as f64,
        regions as f64 / copies.max(1) as f64,
        warmed.elapsed().as_secs_f64(),
        figures.call,
        figures.gpu,
        figures.completion,
    );
    figures
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn host_ram_spread_bench() {
    let Some(device) = ahead_device() else {
        return;
    };
    eprintln!("HOST-RAM-SPREAD device: {}", device.name());
    eprintln!(
        "HOST-RAM-SPREAD the pages copy binds {} imports a dispatch",
        metal2vulkan::bind::PagesCopyModule::imports(device.compile_device())
    );
    let (ram, start, _dropped) = fresh_ram_in(&device, RAM_BYTES, RamPages::Small);
    let used = device.use_host_ram(&ram).unwrap();
    let queue = device.new_command_queue().unwrap();
    let tile = TILE_PAGES * PAGE;
    let cache = device.new_buffer(tile, ResourceOptions::private()).unwrap();
    let shared = device.new_buffer(tile, ResourceOptions::shared()).unwrap();
    for (what, length) in [("page", PAGE), ("tile", tile)] {
        let raw = medians(&queue, |encoder| {
            let started = thread_cpu();
            encoder
                .copy_from_buffer_to_buffer(&cache, 0, &shared, 0, length)
                .unwrap();
            thread_cpu() - started
        });
        eprintln!(
            "HOST-RAM-SPREAD raw {what} Private to Shared ({length} bytes): call cpu {:.1} us, \
             gpu {:.1} us, commit to completion {:.1} us",
            raw.call, raw.gpu, raw.completion
        );
    }
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let page = drawn(start, 1, 1, &mut rng);
    let near = drawn(start, TILE_PAGES, 1, &mut rng);
    let some = drawn(start, TILE_PAGES, 12, &mut rng);
    let anywhere = drawn(start, TILE_PAGES, CHUNKS, &mut rng);
    let ram = (&*ram, start);
    let page_few = download(&device, &queue, ram, (&cache, &page), "page in 1 chunk");
    let near_few = download(&device, &queue, ram, (&cache, &near), "tile in 1 chunk");
    download(&device, &queue, ram, (&cache, &some), "tile in 12 chunks");
    let far = download(&device, &queue, ram, (&cache, &anywhere), "tile anywhere");
    let page_many = download(
        &device,
        &queue,
        ram,
        (&cache, &page),
        "page in 1 chunk again",
    );
    let near_many = download(
        &device,
        &queue,
        ram,
        (&cache, &near),
        "tile in 1 chunk again",
    );
    let empty = medians(&queue, |_| Duration::ZERO);
    eprintln!(
        "HOST-RAM-SPREAD an empty blit encoder's: gpu {:.1} us, commit to completion {:.1} us",
        empty.gpu, empty.completion
    );
    let touched = chunks_of(start, &anywhere).saturating_sub(1).max(1) as f64;
    let per = |far: f64, near: f64| (far - near) / touched;
    eprintln!(
        "HOST-RAM-SPREAD each chunk a tile touches past one: call cpu {:.2} us, gpu {:.2} us, \
         commit to completion {:.2} us",
        per(far.call, near_many.call),
        per(far.gpu, near_many.gpu),
        per(far.completion, near_many.completion),
    );
    eprintln!(
        "HOST-RAM-SPREAD every chunk held, a page / a tile in one chunk: commit to completion \
         {:.1} -> {:.1} us / {:.1} -> {:.1} us, gpu {:.1} -> {:.1} us / {:.1} -> {:.1} us",
        page_few.completion,
        page_many.completion,
        near_few.completion,
        near_many.completion,
        page_few.gpu,
        page_many.gpu,
        near_few.gpu,
        near_many.gpu,
    );
    drop(used);
}
