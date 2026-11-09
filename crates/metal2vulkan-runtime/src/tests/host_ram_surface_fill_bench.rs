//! A cached surface's first fill from a guest's RAM where each listed import costs every
//! submission (RADV, the RAM with a source of dma-bufs and one of run lists: `host_ram.rs`'s
//! `Plan::listed`): a gather of a full-screen surface of [`SURFACE_PAGES`] 4 KiB pages (28.4 MiB,
//! as a consumer's first fill of a 3648 x 2038 `BGRA8Unorm` surface took them), each page any page
//! of the guest's (as a guest's surface's pages lie: a run each), by `BlitCommandEncoder::copy_from_host_ram` into a `Private`
//! buffer, in a command buffer of its own, submitted on the calling thread
//! (`CommandBuffer::commit_and_submit`) and waited for. The RAM is a `memfd`, sealed and mapped
//! shared as QEMU's `memory-backend-memfd,share=on` makes it, every page written.
//!
//! - **First fills**: each of [`SURFACES`] new surfaces once, its pages taken by none before: the
//!   copy imports its lists of runs (`host_ram/copy.rs`, "Lists of runs").
//! - **Fills again**: each surface again right after its first fill, twice: the copy takes the
//!   lists kept past the first (`host_ram/kept.rs`) and imports nothing.
//! - **Over a chunk imported by pointer**: the same gather over a RAM of private anonymous memory
//!   with no source, once its chunks are imported (the first fill is staged through the CPU and
//!   asks for them): the pages-copy dispatches over imports of host pointers.
//! - **Blits**: a `vkCmdCopyBuffer` of the surface's bytes from a no-copy buffer over private
//!   anonymous memory, and from a `Private` buffer, into the cache: the device's rate reading the
//!   host's pages in a row, and its own memory.
//!
//! Printed (`SURFACE-FILL`), per kind, the median and slowest of: the call's CPU time and wall
//! time; of the call, the source's making of the lists' dma-bufs, the runtime's imports of them
//! (`WorkCounts::host_ram_run_list_import_nanos`, the source's included) and the rest; the lists
//! imported and the dispatches; `commit_and_submit`'s wall time; the GPU's time
//! (`gpu_end_time - gpu_start_time`) and its rate; and commit to completion. Ignored: run it in
//! release under `--cfg m2v_gpu` on Linux, on a host with `/dev/udmabuf`, `cargo nextest run
//! --release -p metal2vulkan-runtime --lib --run-ignored all host_ram_surface_fill_bench
//! --no-capture`.

use super::host_ram_listed_bench::{ram_timed, thread_cpu};
use super::host_residency::{device, Scratch};
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, HostMemory, HostRam,
    KeepAlive, ResourceOptions, WorkCounts,
};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
const PAGE: u64 = 4096;
/// A full-screen surface's pages, as a consumer's guest's lay (the module docs).
const SURFACE_PAGES: u64 = 7261;
const SURFACE_BYTES: u64 = SURFACE_PAGES * PAGE;
/// The guest's RAM: every page written.
const RAM_BYTES: u64 = 1 << 30;
/// The private anonymous RAM the pointer imports are made over.
const ANONYMOUS_BYTES: u64 = 256 << 20;
/// The new surfaces filled, each first and then again.
const SURFACES: usize = 8;
const AGAIN: usize = 2;
/// The fills over the imported chunks, and each blit's.
const ITERATIONS: usize = 6;
/// The longest the bench waits for the anonymous RAM's chunks' imports.
const IMPORTS_WAIT: Duration = Duration::from_secs(10);

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

/// A private anonymous mapping, unmapped by the drop: the keep-alive of a RAM over it, or of the
/// bench's no-copy buffer.
struct Anonymous {
    start: u64,
    length: u64,
}

impl Anonymous {
    /// A new mapping of `length` bytes, every page written.
    fn new(length: u64) -> Anonymous {
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
        let start = start as u64;
        write_pages(start, length);
        Anonymous { start, length }
    }
}

impl Drop for Anonymous {
    fn drop(&mut self) {
        // SAFETY: the bench's mapping, which no import lies over any more: its keep-alive goes
        // after the last.
        unsafe { libc::munmap(self.start as *mut libc::c_void, self.length as usize) };
    }
}

/// Write one byte of each page of `length` bytes from `start`, as a guest's CPU wrote them.
fn write_pages(start: u64, length: u64) {
    for page in 0..length / PAGE {
        // SAFETY: the caller's mapping, readable and writable, which no device uses yet.
        unsafe { *((start + page * PAGE) as *mut u8) = (page % 251) as u8 };
    }
}

/// A surface of [`SURFACE_PAGES`] pages of the RAM from `start`, each any page none `taken`
/// before, in the order its bytes take them; each taken.
fn surface(start: u64, taken: &mut [bool], rng: &mut Rng) -> Vec<Range<u64>> {
    let pages = taken.len() as u64;
    (0..SURFACE_PAGES)
        .map(|_| loop {
            let page = rng.next() % pages;
            if !taken[page as usize] {
                taken[page as usize] = true;
                break start + page * PAGE..start + (page + 1) * PAGE;
            }
        })
        .collect()
}

/// One kind's fills.
#[derive(Default)]
struct Kind {
    cpu: Vec<Duration>,
    wall: Vec<Duration>,
    source: Vec<Duration>,
    imports: Vec<Duration>,
    rest: Vec<Duration>,
    lists: Vec<u64>,
    dispatches: Vec<u64>,
    staged: Vec<u64>,
    submit: Vec<Duration>,
    gpu: Vec<Duration>,
    completion: Vec<Duration>,
}

impl Kind {
    fn print(&mut self, device: &Device, what: &str) {
        let ms = |values: &mut Vec<Duration>| {
            if values.is_empty() {
                return "-".to_owned();
            }
            values.sort();
            format!(
                "{:.3} / {:.3}",
                values[values.len() / 2].as_secs_f64() * 1e3,
                values[values.len() - 1].as_secs_f64() * 1e3
            )
        };
        let count = |values: &mut Vec<u64>| {
            values.sort_unstable();
            values.get(values.len() / 2).copied().unwrap_or(0)
        };
        let mut gpu = self.gpu.clone();
        gpu.sort();
        let rate = gpu.get(gpu.len() / 2).map_or(0.0, |median| {
            SURFACE_BYTES as f64 / median.as_secs_f64().max(1e-9) / 1e9
        });
        eprintln!(
            "SURFACE-FILL {}: {what} ({} of them, ms median / slowest): call cpu {}, wall {}; of \
             it the source {}, the list imports {}, the rest {}; lists {}, dispatches {}, staged \
             {}; commit_and_submit {}; gpu {} ({rate:.1} GB/s at the median); commit to \
             completion {}",
            device.name(),
            self.gpu.len(),
            ms(&mut self.cpu),
            ms(&mut self.wall),
            ms(&mut self.source),
            ms(&mut self.imports),
            ms(&mut self.rest),
            count(&mut self.lists),
            count(&mut self.dispatches),
            count(&mut self.staged),
            ms(&mut self.submit),
            ms(&mut self.gpu),
            ms(&mut self.completion),
        );
    }
}

/// Submit `command_buffer` on this thread, wait for it, and add its submission's, its GPU's and
/// its completion's times to `kind`.
fn submit(command_buffer: &CommandBuffer, kind: &mut Kind) {
    let committed = Instant::now();
    command_buffer.commit_and_submit().unwrap();
    kind.submit.push(committed.elapsed());
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    kind.completion.push(committed.elapsed());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
    kind.gpu.push(Duration::from_secs_f64(gpu.max(0.0)));
}

/// Gather `runs` of `ram` into `cache` on `queue` (the module docs), its figures into `kind`, the
/// source's time read from `listing`.
fn fill(
    (device, queue, cache): (&Device, &CommandQueue, &Buffer),
    (ram, runs): (&HostRam, &[Range<u64>]),
    listing: &AtomicU64,
    kind: &mut Kind,
) {
    let before = device.work_counts();
    let listed = listing.load(Ordering::Relaxed);
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    let (cpu, wall) = (thread_cpu(), Instant::now());
    encoder.copy_from_host_ram(ram, runs, cache, 0).unwrap();
    let (cpu, wall) = (thread_cpu() - cpu, wall.elapsed());
    encoder.end_encoding().unwrap();
    let after = device.work_counts();
    let moved = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
    let source = Duration::from_nanos(listing.load(Ordering::Relaxed) - listed);
    let imports = Duration::from_nanos(moved(|counts| counts.host_ram_run_list_import_nanos));
    kind.cpu.push(cpu);
    kind.wall.push(wall);
    kind.source.push(source);
    kind.imports.push(imports);
    kind.rest.push(wall.saturating_sub(imports));
    kind.lists
        .push(moved(|counts| counts.host_ram_run_list_imports));
    kind.dispatches
        .push(moved(|counts| counts.host_ram_copy_dispatches));
    kind.staged
        .push(moved(|counts| counts.host_ram_copies_staged));
    submit(&command_buffer, kind);
}

/// Copy the surface's bytes from `source` into `cache` by a blit on `queue`, its figures into
/// `kind`.
fn blit(queue: &CommandQueue, source: &Buffer, cache: &Buffer, kind: &mut Kind) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    (encoder.copy_from_buffer_to_buffer(source, 0, cache, 0, SURFACE_BYTES)).unwrap();
    encoder.end_encoding().unwrap();
    submit(&command_buffer, kind);
}

/// The listed RAM's fills: a warm-up surface, then the first fills and fills again.
fn listed(on: (&Device, &CommandQueue, &Buffer), rng: &mut Rng) {
    let device = on.0;
    let listing = Arc::new(AtomicU64::new(0));
    let (ram, start) = ram_timed(RAM_BYTES, Arc::clone(&listing));
    write_pages(start, RAM_BYTES);
    let used = device.use_host_ram(&ram).unwrap();
    let chunks = device.host_ram_chunks();
    let [(listed, spans)] = &chunks[..] else {
        panic!("one RAM: {chunks:?}");
    };
    eprintln!(
        "SURFACE-FILL {}: a {} MiB RAM, listed {listed}, {} chunks; a surface of {SURFACE_PAGES} \
         pages, {} bytes; udmabuf's list_limit {}",
        device.name(),
        RAM_BYTES >> 20,
        spans.len(),
        SURFACE_BYTES,
        std::fs::read_to_string("/sys/module/udmabuf/parameters/list_limit")
            .map_or("unknown".to_owned(), |limit| limit.trim().to_owned()),
    );
    let mut taken = vec![false; (RAM_BYTES / PAGE) as usize];
    let mut warm = Kind::default();
    let runs = surface(start, &mut taken, rng);
    fill(on, (&ram, &runs[..]), &listing, &mut warm);
    warm.print(device, "the warm-up surface's first fill");
    let (mut first, mut again) = (Kind::default(), Kind::default());
    for _ in 0..SURFACES {
        let runs = surface(start, &mut taken, rng);
        fill(on, (&ram, &runs[..]), &listing, &mut first);
        for _ in 0..AGAIN {
            fill(on, (&ram, &runs[..]), &listing, &mut again);
        }
    }
    first.print(device, "first fills, the lists imported");
    again.print(device, "fills again, the lists kept");
    drop((used, ram));
}

/// The fills over a RAM of private anonymous memory with no source, once its chunks are
/// imported by pointer, and the blits.
fn by_pointer(on: (&Device, &CommandQueue, &Buffer), rng: &mut Rng) {
    let (device, queue, cache) = on;
    let anonymous = Anonymous::new(ANONYMOUS_BYTES);
    let at = NonNull::new(anonymous.start as *mut u8).unwrap();
    let start = anonymous.start;
    let keep: KeepAlive = Box::new(anonymous);
    // SAFETY: the mapping is readable and writable until the keep-alive unmaps it.
    let ram = Arc::new(unsafe { HostRam::new(&[(at, ANONYMOUS_BYTES)], Some(keep)) }.unwrap());
    let used = device.use_host_ram(&ram).unwrap();
    let chunks = device.host_ram_chunks();
    let spans = chunks.first().map_or(0, |(_, spans)| spans.len()) as u64;
    let mut taken = vec![false; (ANONYMOUS_BYTES / PAGE) as usize];
    let runs = surface(start, &mut taken, rng);
    let listing = AtomicU64::new(0);
    let mut staged = Kind::default();
    let before = device.work_counts().host_ram_chunks_imported;
    fill(on, (&ram, &runs[..]), &listing, &mut staged);
    staged.print(
        device,
        "a first fill over a RAM of private anonymous memory",
    );
    let began = Instant::now();
    let imported = || device.work_counts().host_ram_chunks_imported - before;
    while imported() < spans && began.elapsed() < IMPORTS_WAIT {
        std::thread::sleep(Duration::from_millis(10));
    }
    eprintln!(
        "SURFACE-FILL {}: {} of the anonymous RAM's {spans} chunks imported {:.0} ms after its \
         first fill",
        device.name(),
        imported(),
        began.elapsed().as_secs_f64() * 1e3
    );
    let mut imported = Kind::default();
    for _ in 0..ITERATIONS {
        fill(on, (&ram, &runs[..]), &listing, &mut imported);
    }
    imported.print(device, "fills over chunks imported by pointer");
    drop((used, ram));
    let anonymous = Anonymous::new(SURFACE_BYTES);
    let at = NonNull::new(anonymous.start as *mut u8).unwrap();
    // SAFETY: the mapping stays mapped, readable and writable, until `anonymous` goes, after the
    // buffer.
    let memory = unsafe { HostMemory::new(at) };
    // SAFETY: as above.
    let host = unsafe {
        device.new_buffer_with_bytes_no_copy(
            memory,
            SURFACE_BYTES,
            ResourceOptions::default(),
            None,
        )
    }
    .unwrap();
    let private = device
        .new_buffer(SURFACE_BYTES, ResourceOptions::private())
        .unwrap();
    let (mut from_host, mut from_private) = (Kind::default(), Kind::default());
    for _ in 0..ITERATIONS {
        blit(queue, &host, cache, &mut from_host);
        blit(queue, &private, cache, &mut from_private);
    }
    from_host.print(
        device,
        "blits from a no-copy buffer over private anonymous memory",
    );
    from_private.print(device, "blits from a Private buffer");
    drop((host, private, anonymous));
}

#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn host_ram_surface_fill_bench() {
    if std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/udmabuf")
        .is_err()
    {
        eprintln!("SURFACE-FILL: /dev/udmabuf cannot be opened: not run");
        return;
    }
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    if device.shared().host_import.is_err() {
        eprintln!("SURFACE-FILL: the device imports no host memory here: not run");
        return;
    }
    let queue = device.new_command_queue().unwrap();
    let cache = device
        .new_buffer(SURFACE_BYTES, ResourceOptions::private())
        .unwrap();
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let on = (&device, &queue, &cache);
    listed(on, &mut rng);
    by_pointer(on, &mut rng);
}
