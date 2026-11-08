//! The cost of copying a guest's surface between its RAM and a buffer of the device's own, two
//! ways: by `BlitCommandEncoder::copy_from_host_ram` and `copy_to_host_ram` from the device's
//! imports of the RAM's chunks (`host_ram/copy.rs`), and as a consumer did before them, by a
//! no-copy buffer over a view of the surface's pages (one mapping per run of the RAM's file, the
//! buffer made with `HostMemory::with_runs`, so imported by its pointer) and a blit between it and
//! the buffer.
//!
//! Then the same surface's pages as rows of [`ROW_BYTES`] bytes of texels at a [`PITCH`]-byte
//! pitch, downloaded as a consumer downloads a laid-out device cache: each row's texels alone,
//! by `copy_to_host_ram` of the surface's runs cut to the row, one call a row (and once as one
//! call of every row's runs), each run ending inside a page, so copied by regions; and by the
//! view, a blit a row from the buffer into it (`LAID-OUT` lines).
//!
//! The surface: a 3840x2054 `BGRA8Unorm` one, 31,549,440 bytes, as whole runs of `UNIT` bytes
//! (7,703 of 4 KiB) scattered over a 1 GiB RAM: about 75% of them the page below the one before
//! (a descending neighbour, a run of its own), 20% the page above it (merged into it), and the
//! rest anywhere. The view's runs are the host page's (16 KiB on an Apple GPU's Mac, where a
//! mapping takes no 4 KiB run), so there the gather is measured on both shapes. Each figure is the
//! median of `ITERATIONS` command buffers, each copying the surface alone, waited for before the
//! next: the encoding thread's CPU time for the copy's call, the GPU's time
//! (`gpu_end_time - gpu_start_time`, the device's timestamps, or its host times where it writes
//! none), and the commit's return to the completion seen; and for the view, the view's mapping,
//! the buffer's making and its release. Prints `HOST-RAM-COPY-BENCH` lines. Ignored: run it in
//! release under `--cfg m2v_gpu`,
//! `cargo test --release -p metal2vulkan-runtime --lib host_ram_copy_bench -- --ignored --nocapture`.

use crate::{
    Buffer, CommandBufferStatus, CommandQueue, Device, DeviceKind, DeviceOptions, HostMemory,
    HostRam, KeepAlive, ResourceOptions, RunResolver,
};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
const SURFACE: u64 = 3840 * 2054 * 4;
const RAM_BYTES: u64 = 1 << 30;
const ITERATIONS: usize = 40;
/// The laid-out surface's pitch and its rows' bytes of texels: 2054 rows of a 7677-texel-wide
/// surface of 2-byte texels, each row ending 2 bytes past a word, 6 bytes short of the pitch.
const PITCH: u64 = 3840 * 4;
const ROW_BYTES: u64 = PITCH - 6;

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

/// The RAM's file and its mapping, unmapped by the keep-alive it is given to.
struct Mapping {
    start: u64,
    fd: i32,
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the bench's mapping and descriptor, which no device imports any more.
        unsafe {
            libc::munmap(self.start as *mut libc::c_void, RAM_BYTES as usize);
            libc::close(self.fd);
        }
    }
}

/// A RAM of `RAM_BYTES` in a file the views map again (a `memfd` on Linux, POSIX shared memory
/// elsewhere), every byte touched; its first byte and the file.
fn shared_ram() -> (Arc<HostRam>, u64, i32) {
    #[cfg(target_os = "linux")]
    // SAFETY: a name and flags the call takes.
    let fd = unsafe { libc::memfd_create(c"m2v-copy-bench".as_ptr(), libc::MFD_CLOEXEC) };
    #[cfg(not(target_os = "linux"))]
    let fd = {
        let name = format!("/m2v-copy-bench-{}", std::process::id());
        let name = std::ffi::CString::new(name).unwrap();
        // SAFETY: a name and flags the call takes; the name goes at once, the descriptor keeps it.
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
    assert_eq!(unsafe { libc::ftruncate(fd, RAM_BYTES as libc::off_t) }, 0);
    // SAFETY: a fresh shared mapping of the file.
    let start = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            RAM_BYTES as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd,
            0,
        )
    };
    assert_ne!(start, libc::MAP_FAILED);
    // SAFETY: the mapping's bytes, which nothing else uses yet.
    unsafe { std::ptr::write_bytes(start.cast::<u8>(), 0x5a, RAM_BYTES as usize) };
    let start = start as u64;
    let keep: KeepAlive = Box::new(Mapping { start, fd });
    let at = NonNull::new(start as *mut u8).unwrap();
    // SAFETY: the mapping is readable and writable until the keep-alive unmaps it.
    let ram = unsafe { HostRam::new(&[(at, RAM_BYTES)], Some(keep)) }.unwrap();
    (Arc::new(ram), start, fd)
}

/// The surface's runs of `unit` bytes in a RAM from `start` (the module docs), in its order.
fn surface_runs(start: u64, unit: u64, seed: u64) -> Vec<Range<u64>> {
    let pages = RAM_BYTES / unit;
    let count = SURFACE.div_ceil(unit);
    let mut rng = Rng(seed);
    let mut taken = vec![false; pages as usize];
    let mut runs = Vec::with_capacity(count as usize);
    let mut at = rng.next() % pages;
    for _ in 0..count {
        while taken[at as usize] {
            at = rng.next() % pages;
        }
        taken[at as usize] = true;
        runs.push(start + at * unit..start + (at + 1) * unit);
        let roll = rng.next() % 100;
        let next = match roll {
            0..75 => at.checked_sub(1),
            75..95 => Some(at + 1).filter(|&next| next < pages),
            _ => None,
        };
        at = next.unwrap_or_else(|| rng.next() % pages);
    }
    runs
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

/// The median of `values`, in milliseconds.
fn median(values: &mut [Duration]) -> f64 {
    values.sort();
    values[values.len() / 2].as_secs_f64() * 1e3
}

/// One command buffer's figures: the call's CPU time, the GPU's and commit to completion.
#[derive(Default)]
struct Figures {
    call: Vec<Duration>,
    gpu: Vec<Duration>,
    completion: Vec<Duration>,
}

impl Figures {
    fn print(&mut self, what: &str) {
        eprintln!(
            "HOST-RAM-COPY-BENCH {what}: call cpu {:.3} ms, gpu {:.3} ms, commit to completion {:.3} ms",
            median(&mut self.call),
            median(&mut self.gpu),
            median(&mut self.completion),
        );
    }
}

/// Commit a command buffer whose one blit encoder `encode` fills (its call's CPU time returned by
/// it), wait for it, and note its figures.
fn timed(
    queue: &CommandQueue,
    figures: &mut Figures,
    encode: impl FnOnce(&mut crate::BlitCommandEncoder) -> Duration,
) {
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
    figures.call.push(call);
    figures.gpu.push(Duration::from_secs_f64(gpu.max(0.0)));
    figures.completion.push(completion);
}

/// The gather and the scatter of `runs` between `ram` and `cache`, timed.
fn chunk_copies(
    queue: &CommandQueue,
    ram: &HostRam,
    runs: &[Range<u64>],
    cache: &Buffer,
    what: &str,
) {
    let mut gather = Figures::default();
    let mut scatter = Figures::default();
    for _ in 0..ITERATIONS {
        timed(queue, &mut gather, |encoder| {
            let started = thread_cpu();
            encoder.copy_from_host_ram(ram, runs, cache, 0).unwrap();
            thread_cpu() - started
        });
        timed(queue, &mut scatter, |encoder| {
            let started = thread_cpu();
            encoder.copy_to_host_ram(cache, 0, ram, runs).unwrap();
            thread_cpu() - started
        });
    }
    gather.print(&format!("{what} copy_from_host_ram"));
    scatter.print(&format!("{what} copy_to_host_ram"));
}

/// A view of `runs` of the RAM's file `fd` (the RAM mapped from `start`): one shared mapping of
/// each run, one after another, in a reservation of their bytes; its first byte.
fn map_view(fd: i32, start: u64, runs: &[Range<u64>]) -> u64 {
    let length: u64 = runs.iter().map(|run| run.end - run.start).sum();
    // SAFETY: a fresh reservation, which the runs' mappings replace.
    let view = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length as usize,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert_ne!(view, libc::MAP_FAILED);
    let mut at = view as u64;
    for run in runs {
        let size = run.end - run.start;
        // SAFETY: a mapping over the bench's reservation, of the RAM's file.
        let mapped = unsafe {
            libc::mmap(
                at as *mut libc::c_void,
                size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd,
                (run.start - start) as libc::off_t,
            )
        };
        assert_eq!(mapped as u64, at);
        at += size;
    }
    view as u64
}

/// A resolver of a view from `view` of `runs`: each of its pages to the RAM's.
fn view_resolver(view: u64, runs: &[Range<u64>]) -> RunResolver {
    let runs: Arc<[Range<u64>]> = runs.into();
    Arc::new(move |start, length| {
        let mut out: Vec<Range<u64>> = Vec::new();
        let mut at = 0u64;
        for run in runs.iter() {
            let size = run.end - run.start;
            let (from, to) = (
                (start - view).max(at),
                (start - view + length).min(at + size),
            );
            if from < to {
                out.push(run.start + (from - at)..run.start + (to - at));
            }
            at += size;
        }
        Some(out)
    })
}

/// The consumer's path before: per command buffer, a view of the runs, a no-copy buffer over it
/// (`HostMemory::with_runs`), a blit between it and `cache`, and both let go of.
fn view_copies(
    device: &Device,
    queue: &CommandQueue,
    (fd, start): (i32, u64),
    runs: &[Range<u64>],
    cache: &Buffer,
    what: &str,
) {
    let length: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let mut figures = Figures::default();
    let (mut maps, mut makes, mut releases) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..ITERATIONS {
        let started = Instant::now();
        let view = map_view(fd, start, runs);
        maps.push(started.elapsed());
        let pointer = NonNull::new(view as *mut u8).unwrap();
        // SAFETY: the view's bytes, mapped until the bench unmaps them below, after the buffer
        // went and its command buffer completed.
        let memory = unsafe { HostMemory::new(pointer) }.with_runs(view_resolver(view, runs));
        let started = Instant::now();
        // SAFETY: as above.
        let buffer = unsafe {
            device.new_buffer_with_bytes_no_copy(memory, length, ResourceOptions::default(), None)
        }
        .unwrap();
        makes.push(started.elapsed());
        timed(queue, &mut figures, |encoder| {
            let started = thread_cpu();
            encoder
                .copy_from_buffer_to_buffer(&buffer, 0, cache, 0, length)
                .unwrap();
            thread_cpu() - started
        });
        let started = Instant::now();
        drop(buffer);
        // SAFETY: the bench's view, which nothing uses any more.
        unsafe { libc::munmap(view as *mut libc::c_void, length as usize) };
        device.host_range_remapped(pointer, length);
        releases.push(started.elapsed());
    }
    figures.print(&format!("{what} view copy_from_buffer_to_buffer"));
    eprintln!(
        "HOST-RAM-COPY-BENCH {what} view: map {:.3} ms, buffer made {:.3} ms, released {:.3} ms",
        median(&mut maps),
        median(&mut makes),
        median(&mut releases),
    );
}

/// The runs of `runs` (the surface's, in its order) that hold its bytes `span`, cut to them: a
/// row's texels as the consumer hands them to `copy_to_host_ram`.
fn runs_of(runs: &[Range<u64>], span: &Range<u64>) -> Vec<Range<u64>> {
    let mut out = Vec::new();
    let mut at = 0u64;
    for run in runs {
        let length = run.end - run.start;
        let (from, to) = (span.start.max(at), span.end.min(at + length));
        if from < to {
            out.push(run.start + (from - at)..run.start + (to - at));
        }
        at += length;
        if at >= span.end {
            break;
        }
    }
    out
}

/// The laid-out download (the module docs) of the surface whose pages are `runs`, from `cache`
/// into the RAM: by runs a call a row, by runs in one call, and by the view a blit a row.
fn laid_out_copies(
    device: &Device,
    queue: &CommandQueue,
    (ram, fd, start): (&HostRam, i32, u64),
    runs: &[Range<u64>],
    cache: &Buffer,
) {
    let rows = SURFACE / PITCH;
    let spans: Vec<Range<u64>> = (0..rows)
        .map(|row| row * PITCH..row * PITCH + ROW_BYTES)
        .collect();
    let row_runs: Vec<Vec<Range<u64>>> = spans.iter().map(|span| runs_of(runs, span)).collect();
    let every: Vec<Range<u64>> = row_runs.iter().flatten().cloned().collect();
    let what = format!(
        "LAID-OUT {rows} rows of {ROW_BYTES} at {PITCH} ({} runs)",
        every.len()
    );
    // Warm: every chunk imported, the paths' first use made.
    let mut warm = Figures::default();
    timed(queue, &mut warm, |encoder| {
        for (span, runs) in spans.iter().zip(&row_runs) {
            encoder
                .copy_to_host_ram(cache, span.start, ram, runs)
                .unwrap();
        }
        Duration::ZERO
    });
    let (mut by_row, mut at_once) = (Figures::default(), Figures::default());
    let before = device.work_counts();
    for _ in 0..ITERATIONS {
        timed(queue, &mut by_row, |encoder| {
            let started = thread_cpu();
            for (span, runs) in spans.iter().zip(&row_runs) {
                encoder
                    .copy_to_host_ram(cache, span.start, ram, runs)
                    .unwrap();
            }
            thread_cpu() - started
        });
    }
    let after = device.work_counts();
    let calls = after.host_ram_copies - before.host_ram_copies;
    by_row.print(&format!("{what} copy_to_host_ram a call a row"));
    eprintln!(
        "HOST-RAM-COPY-BENCH {what}: {calls} calls, {} regions and {} dispatches a download",
        (after.host_ram_copy_regions - before.host_ram_copy_regions) / ITERATIONS as u64,
        (after.host_ram_copy_dispatches - before.host_ram_copy_dispatches) / ITERATIONS as u64,
    );
    // The rows' runs in one call, from a buffer of the rows' texels alone.
    let texels = device
        .new_buffer(rows * ROW_BYTES, ResourceOptions::private())
        .unwrap();
    for _ in 0..ITERATIONS {
        timed(queue, &mut at_once, |encoder| {
            let started = thread_cpu();
            encoder.copy_to_host_ram(&texels, 0, ram, &every).unwrap();
            thread_cpu() - started
        });
    }
    at_once.print(&format!("{what} copy_to_host_ram in one call"));
    // The view: per download, a view of the surface's pages, a no-copy buffer over it, a blit a
    // row from the cache into it, both let go of.
    let length: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let mut figures = Figures::default();
    let (mut maps, mut makes, mut releases) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..ITERATIONS {
        let started = Instant::now();
        let view = map_view(fd, start, runs);
        maps.push(started.elapsed());
        let pointer = NonNull::new(view as *mut u8).unwrap();
        // SAFETY: the view's bytes, mapped until the bench unmaps them below, after the buffer
        // went and its command buffer completed.
        let memory = unsafe { HostMemory::new(pointer) }.with_runs(view_resolver(view, runs));
        let started = Instant::now();
        // SAFETY: as above.
        let buffer = unsafe {
            device.new_buffer_with_bytes_no_copy(memory, length, ResourceOptions::default(), None)
        }
        .unwrap();
        makes.push(started.elapsed());
        timed(queue, &mut figures, |encoder| {
            let started = thread_cpu();
            for span in &spans {
                let size = span.end - span.start;
                encoder
                    .copy_from_buffer_to_buffer(cache, span.start, &buffer, span.start, size)
                    .unwrap();
            }
            thread_cpu() - started
        });
        let started = Instant::now();
        drop(buffer);
        // SAFETY: the bench's view, which nothing uses any more.
        unsafe { libc::munmap(view as *mut libc::c_void, length as usize) };
        device.host_range_remapped(pointer, length);
        releases.push(started.elapsed());
    }
    figures.print(&format!("{what} view copy_from_buffer_to_buffer a row"));
    eprintln!(
        "HOST-RAM-COPY-BENCH {what} view: {} mappings, map {:.3} ms, buffer made {:.3} ms, \
         released {:.3} ms",
        runs.len(),
        median(&mut maps),
        median(&mut makes),
        median(&mut releases),
    );
}

#[test]
#[ignore = "a release benchmark: run with --ignored --nocapture"]
fn host_ram_copy_bench() {
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
        eprintln!("HOST-RAM-COPY-BENCH: the device imports no host memory here");
        return;
    }
    eprintln!("HOST-RAM-COPY-BENCH device: {}", device.name());
    let (ram, start, fd) = shared_ram();
    let used = device.use_host_ram(&ram).unwrap();
    let queue = device.new_command_queue().unwrap();
    // SAFETY: a host page size the system answers.
    let host_page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
    let mut units = vec![4096];
    if host_page != 4096 {
        units.push(host_page);
    }
    for &unit in &units {
        let runs = surface_runs(start, unit, 0x2545_f491_4f6c_dd1d);
        let length: u64 = runs.iter().map(|run| run.end - run.start).sum();
        let merged = crate::host_ram::copy::merged(&runs).unwrap().len();
        let descending = (runs.windows(2))
            .filter(|pair| pair[1].end == pair[0].start)
            .count();
        let what = format!("{} runs of {unit}", runs.len());
        eprintln!(
            "HOST-RAM-COPY-BENCH {what}: {length} bytes, {merged} merged, {descending} descending neighbours"
        );
        let cache = device
            .new_buffer(length, ResourceOptions::private())
            .unwrap();
        // A warm-up of each path: its kernels and pipelines made, and every chunk the runs touch
        // imported, as a consumer's RAM is long before its surfaces (the device's thread imports
        // only a window of chunks past the last one taken, `host_ram.rs`).
        let mut warm = Figures::default();
        timed(&queue, &mut warm, |encoder| {
            encoder.copy_from_host_ram(&ram, &runs, &cache, 0).unwrap();
            encoder.copy_to_host_ram(&cache, 0, &ram, &runs).unwrap();
            Duration::ZERO
        });
        let before = device.work_counts();
        chunk_copies(&queue, &ram, &runs, &cache, &what);
        let after = device.work_counts();
        let copies = after.host_ram_copies - before.host_ram_copies;
        eprintln!(
            "HOST-RAM-COPY-BENCH {what}: {copies} copies, {} dispatches each",
            (after.host_ram_copy_dispatches - before.host_ram_copy_dispatches) / copies.max(1)
        );
        if unit == host_page {
            view_copies(&device, &queue, (fd, start), &runs, &cache, &what);
        }
        if unit == 4096 {
            laid_out_copies(&device, &queue, (&ram, fd, start), &runs, &cache);
        }
    }
    // An empty command buffer's commit to completion, for scale.
    let mut empty = Figures::default();
    for _ in 0..ITERATIONS {
        timed(&queue, &mut empty, |_| Duration::ZERO);
    }
    empty.print("an empty blit encoder's");
    drop(used);
}
