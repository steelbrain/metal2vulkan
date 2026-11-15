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
//! Beside it, [`host_ram_first_fill_spread_bench`]: the first fill of a tile's cache from a fresh
//! RAM, a `copy_from_host_ram` of [`TILE_PAGES`] pages drawn from every chunk, none imported, as a
//! booted guest's first surfaces are: alone, and after a download of one page into the chunk of
//! the tile's first page (imported by then, as an earlier surface's chunk is). Prints
//! `HOST-RAM-FIRST-FILL` lines: the call's time on the encoding thread (wall and CPU), whether the
//! copy was staged through the CPU, and the encoding thread's waits for chunks' imports.
//!
//! And [`host_ram_layer_encode_bench`]: the encoding of a download of a 4K `BGRA8` layer's
//! [`LAYER_PAGES`] pages, a run each at its byte of the cache
//! (`BlitCommandEncoder::copy_runs_to_host_ram`, as a VMM's device cache copies a surface by its
//! pages), drawn from one chunk and from every chunk, once their chunks are imported: the call's
//! CPU time on the encoding thread over [`LAYER_ROUNDS`] command buffers, its share a run, and
//! the copies recorded by the plan kept of the same copy before them
//! (`WorkCounts::host_ram_copy_plans_reused`); then the copy beside a fill of 4 bytes in one
//! encoder, in either order, the call CPU of each (the command after the other takes it into the
//! encoder's order); then the layer over every chunk of a second RAM whose writes a log keeps
//! (`HostRam::with_dirty_pages`, as a VMM keeps its guest RAM's on a discrete GPU), with the runs
//! the log was told a copy. Prints `HOST-RAM-LAYER` lines.
//!
//! Ignored: run them in release, alone on their device,
//! `cargo test --release -p metal2vulkan-runtime --lib host_ram_spread_bench -- --ignored
//! --nocapture` (or `host_ram_first_fill_spread_bench`) under `--cfg m2v_gpu` (or
//! `--cfg m2v_cpu_icd`, lavapipe: our recording's share).

use super::host_ram_device::{ahead_device, fresh_logged_ram_in, fresh_ram_in, settles, RamPages};
use crate::{
    Buffer, CommandBufferStatus, CommandQueue, DirtyPages, DirtyStamp, HostRam, ResourceOptions,
    WorkCounts,
};
use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
const PAGE: u64 = 4096;
const CHUNK: u64 = 64 << 20;
const CHUNKS: u64 = 128;
const RAM_BYTES: u64 = CHUNKS * CHUNK;
/// An 800x512 surface of 4-byte texels, in whole pages.
const TILE_PAGES: u64 = (800 * 512 * 4) / PAGE;
const ITERATIONS: usize = 30;
/// A 3840x2160 surface of 4-byte texels, in whole pages.
const LAYER_PAGES: u64 = (3840 * 2160 * 4) / PAGE;
/// The command buffers a layer's case times.
const LAYER_ROUNDS: usize = 200;
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

/// The counts once the device's thread imported nothing more for 300 ms, or after [`WAIT`].
fn settled(device: &crate::Device) -> WorkCounts {
    let started = Instant::now();
    let mut last = device.work_counts();
    loop {
        std::thread::sleep(Duration::from_millis(300));
        let now = device.work_counts();
        let still = now.host_ram_chunks_imported == last.host_ram_chunks_imported;
        if still || started.elapsed() > WAIT {
            return now;
        }
        last = now;
    }
}

/// The first fill of a tile's cache from a fresh RAM (the module docs), after a download of one
/// page into the chunk of the tile's first page where `after_download`: its line.
fn first_fill(device: &crate::Device, after_download: bool) {
    let (ram, start, _dropped) = fresh_ram_in(device, RAM_BYTES, RamPages::Small);
    let used = device.use_host_ram(&ram).unwrap();
    let queue = device.new_command_queue().unwrap();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let tile = drawn(start, TILE_PAGES, CHUNKS, &mut rng);
    let cache = (device.new_buffer(TILE_PAGES * PAGE, ResourceOptions::private())).unwrap();
    settled(device);
    if after_download {
        let page = [tile[0].clone()];
        timed(&queue, |encoder| {
            encoder.copy_to_host_ram(&cache, 0, &ram, &page).unwrap();
            Duration::ZERO
        });
        settled(device);
    }
    let before = device.work_counts();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    let (cpu, wall) = (thread_cpu(), Instant::now());
    encoder.copy_from_host_ram(&ram, &tile, &cache, 0).unwrap();
    let (cpu, wall) = (thread_cpu() - cpu, wall.elapsed());
    let encoded = device.work_counts();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let delta = |count: fn(&WorkCounts) -> u64| count(&encoded) - count(&before);
    let ahead = delta(|c| c.host_ram_chunks_imported_ahead);
    eprintln!(
        "HOST-RAM-FIRST-FILL {}: {} runs over {} chunks ({} imported before): call {:.1} us \
         wall, {:.1} us cpu; staged {}, import waits {} ({:.1} us), chunks imported by the \
         copy {}, by the device's thread meanwhile {}",
        match after_download {
            true => "after a download into one of its chunks",
            false => "alone",
        },
        tile.len(),
        chunks_of(start, &tile),
        before.host_ram_chunks_live,
        wall.as_secs_f64() * 1e6,
        cpu.as_secs_f64() * 1e6,
        delta(|c| c.host_ram_copies_staged),
        delta(|c| c.host_ram_copy_import_waits),
        delta(|c| c.host_ram_copy_import_wait_nanos) as f64 / 1e3,
        delta(|c| c.host_ram_chunks_imported) - ahead,
        ahead,
    );
    drop((cache, used));
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn host_ram_first_fill_spread_bench() {
    let Some(device) = ahead_device() else {
        return;
    };
    eprintln!("HOST-RAM-FIRST-FILL device: {}", device.name());
    first_fill(&device, false);
    first_fill(&device, true);
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn host_ram_layer_encode_bench() {
    let Some(device) = ahead_device() else {
        return;
    };
    eprintln!("HOST-RAM-LAYER device: {}", device.name());
    let queue = device.new_command_queue().unwrap();
    let cache = (device.new_buffer(LAYER_PAGES * PAGE, ResourceOptions::private())).unwrap();
    let mut rng = Rng(0x6a09_e667_f3bc_c909);
    let (ram, start, dropped) = fresh_ram_in(&device, RAM_BYTES, RamPages::Small);
    let used = device.use_host_ram(&ram).unwrap();
    for (what, chunks) in [("layer in 1 chunk", 1), ("layer anywhere", CHUNKS)] {
        layer(&queue, &cache, (&ram, start), (what, chunks), &mut rng);
    }
    // Its pages let go of before the next RAM's are imported: two at once would hold twice the
    // memory.
    drop((used, ram));
    assert!(settles(&dropped), "the first RAM is let go of");
    // The layer over every chunk of a RAM whose writes a log keeps, as a VMM keeps its guest
    // RAM's on a discrete GPU: each command's uses are noted for the commit's report to it.
    let log = Arc::new(LayerLog::default());
    let (ram, start, _dropped) =
        fresh_logged_ram_in(&device, RAM_BYTES, RamPages::Small, Arc::clone(&log));
    let used = device.use_host_ram(&ram).unwrap();
    let what = "layer anywhere, logged";
    let told = log.runs.load(Ordering::Relaxed);
    let copies = layer(&queue, &cache, (&ram, start), (what, CHUNKS), &mut rng);
    let told = log.runs.load(Ordering::Relaxed) - told;
    eprintln!(
        "HOST-RAM-LAYER {what}: the log was told {:.1} runs a copy",
        told as f64 / copies.max(1) as f64
    );
    drop((cache, used));
}

/// [`host_ram_layer_encode_bench`]'s case `what`: a layer's [`LAYER_PAGES`] pages drawn from the
/// first `chunks` chunks of `ram` (whose first byte is `start`), downloaded from `cache` once
/// its chunks are imported, timed alone and beside a fill. The copies the case made, warming
/// included.
fn layer(
    queue: &CommandQueue,
    cache: &Buffer,
    (ram, start): (&Arc<HostRam>, u64),
    (what, chunks): (&str, u64),
    rng: &mut Rng,
) -> u64 {
    let device = queue.device();
    let first = device.work_counts();
    let runs: Vec<(u64, Range<u64>)> = (drawn(start, LAYER_PAGES, chunks, rng))
        .into_iter()
        .enumerate()
        .map(|(index, run)| (index as u64 * PAGE, run))
        .collect();
    let copy = |encoder: &mut crate::BlitCommandEncoder| {
        let started = thread_cpu();
        encoder.copy_runs_to_host_ram(cache, ram, &runs).unwrap();
        thread_cpu() - started
    };
    let warmed = Instant::now();
    loop {
        let before = device.work_counts();
        timed(queue, copy);
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
    let mut calls: Vec<Duration> = (0..LAYER_ROUNDS).map(|_| timed(queue, copy)[0]).collect();
    let after = device.work_counts();
    assert!(
        through_imports(&before, &after),
        "{what}: a timed copy was staged"
    );
    let copies = (after.host_ram_copies - before.host_ram_copies).max(1);
    let dispatches = after.host_ram_copy_dispatches - before.host_ram_copy_dispatches;
    let reused = after.host_ram_copy_plans_reused - before.host_ram_copy_plans_reused;
    let call = median_us(&mut calls);
    eprintln!(
        "HOST-RAM-LAYER {what}: {} runs over {} chunks, {} dispatches a copy, {reused} of {} \
         copies by a kept plan: call cpu {:.1} us ({:.1} ns a run)",
        runs.len(),
        chunks_of(
            start,
            &runs.iter().map(|(_, run)| run.clone()).collect::<Vec<_>>()
        ),
        dispatches as f64 / copies as f64,
        after.host_ram_copies - before.host_ram_copies,
        call,
        call * 1e3 / runs.len() as f64,
    );
    let [copy_first, fill_after, fill_first, copy_after] = beside_a_fill(queue, |encoder| {
        encoder.copy_runs_to_host_ram(cache, ram, &runs).unwrap();
    });
    eprintln!(
        "HOST-RAM-LAYER {what} beside a fill: copy first {copy_first:.1} us, the fill after it \
         {fill_after:.1} us; a fill first {fill_first:.1} us, the copy after it {copy_after:.1} us"
    );
    device.work_counts().host_ram_copies - first.host_ram_copies
}

/// A log of a RAM's writes ([`DirtyPages`]) that keeps nothing but how many runs the runtime told
/// it of: [`host_ram_layer_encode_bench`]'s logged case.
#[derive(Default)]
struct LayerLog {
    stamp: AtomicU64,
    runs: AtomicU64,
}

impl DirtyPages for LayerLog {
    type Watch = ();

    fn watch(&self, _range: Range<u64>) {}

    fn request(&self) -> DirtyStamp {
        DirtyStamp(self.stamp.fetch_add(1, Ordering::Relaxed) + 1)
    }

    fn wait(&self, _stamp: DirtyStamp) -> bool {
        true
    }

    fn changed(&self, _since: DirtyStamp, _out: &mut Vec<Range<u64>>) {}

    fn wrote(&self, _range: Range<u64>) {
        self.runs.fetch_add(1, Ordering::Relaxed);
    }
}

/// The call CPU medians, in microseconds, of [`LAYER_ROUNDS`] command buffers of one encoder each
/// that records `copy` and a fill of 4 bytes, in either order: `copy` first, the fill after it
/// (which takes the copy into the encoder's order), the fill first, `copy` after it (which
/// compares with that order and joins it).
fn beside_a_fill(queue: &CommandQueue, copy: impl Fn(&mut crate::BlitCommandEncoder)) -> [f64; 4] {
    let small = (queue.device().new_buffer(64, ResourceOptions::private())).unwrap();
    let mut times: [Vec<Duration>; 4] = Default::default();
    let timed_pair = |copy_first: bool, times: &mut [Vec<Duration>; 4]| {
        let mut second = Duration::ZERO;
        let first = timed(queue, |encoder| {
            let fill = |encoder: &mut crate::BlitCommandEncoder| {
                let started = thread_cpu();
                encoder.fill_buffer(&small, 0..4, 1).unwrap();
                thread_cpu() - started
            };
            let copied = |encoder: &mut crate::BlitCommandEncoder| {
                let started = thread_cpu();
                copy(encoder);
                thread_cpu() - started
            };
            match copy_first {
                true => {
                    let first = copied(encoder);
                    second = fill(encoder);
                    first
                }
                false => {
                    let first = fill(encoder);
                    second = copied(encoder);
                    first
                }
            }
        })[0];
        let at = if copy_first { 0 } else { 2 };
        times[at].push(first);
        times[at + 1].push(second);
    };
    for _ in 0..LAYER_ROUNDS {
        timed_pair(true, &mut times);
        timed_pair(false, &mut times);
    }
    times.map(|mut values| median_us(&mut values))
}
