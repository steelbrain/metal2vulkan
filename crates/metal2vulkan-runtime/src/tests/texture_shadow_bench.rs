//! What a texture over a no-copy buffer of host pages costs a cycle that makes it, uses it and
//! drops it, where its image lies in place over the pages and where it is a shadow of them: the
//! pages a source makes dma-bufs of (`dma_buf.rs`; RADV refuses a shared mapping's pointer), as a
//! consumer's guest makes textures over a buffer of its RAM, uses each in one or two dispatches and
//! drops it.
//!
//! The RAM is a `memfd` sealed against growing and shrinking and mapped shared, as QEMU's
//! `memory-backend-memfd,share=on` makes it, in two kinds of pages, each page written through a
//! shared mapping of the whole file before the bench: **2 MiB pages**, the mapping advised
//! `MADV_HUGEPAGE` as QEMU advises its guest RAM, so that shmem makes huge pages of it where
//! `/sys/kernel/mm/transparent_hugepage/shmem_enabled` allows (`advise`, `within_size`, `always`);
//! and **4 KiB pages**, the mapping advised `MADV_NOHUGEPAGE` (all but `force` then make 4 KiB
//! pages). Printed: the setting, and each RAM's `ShmemPmdMapped` in its mapping (`/proc/self/smaps`):
//! its bytes the kernel mapped by 2 MiB pages. For each shape and kind of pages, three no-copy
//! buffers over views of that RAM:
//! - **in place**: a view of consecutive file pages, one run, which one udmabuf holds: the
//!   texture's image is a `DRM_FORMAT_MOD_LINEAR` image over a dma-buf of its own pages
//!   (`Device::dma_buf_image`, `host_textures_in_place`), which parks as the texture goes: each
//!   cycle first makes such a texture, reads it in one dispatch and drops it, outside its times
//!   and counts, and its own texture takes the image that parked (`dma_buf/parked.rs`,
//!   `dma_buf_images_taken`), as the device's residency releases a parked image at its next pass
//!   where held imports cost every submission, sooner than the arms between two of the arm's
//!   cycles run;
//! - **unparked**: the same over other consecutive pages, its cycles run while the device parks
//!   none (`Device::park_dma_buf_images_at_most(0)`, restored once the cycle's image went, outside
//!   its times): each cycle's texture makes its image and imports a new dma-buf, as before images
//!   parked;
//! - **shadow**: a view whose every page is another page of the file (each page a run, as a
//!   guest's scattered pages lie), more runs than one udmabuf lists (its `list_limit`): the buffer
//!   lies in place over several dma-bufs (a sparse buffer, `dma_buf_sparse_buffers`), and the
//!   source refuses its texture's request, which one dma-buf must answer, so the texture is a
//!   shadow (`host_texture_shadows_refused_source`).
//!
//! The source answers as a consumer's does: one udmabuf list of each chunk of at most
//! `list_limit` runs and `size_limit_mb` MiB (udmabuf's parameters, read from sysfs), split at the
//! request's granule, and a standing refusal (`DmaBufStandingRefusal`) where one chunk must hold
//! more: it keeps no dma-buf of a wider range.
//!
//! Each cycle makes an `RGBA8Unorm`, `ShaderRead` texture over the buffer's first byte at rows of
//! its width, encodes 1 or 2 dispatches (each in an encoder of its own, in one command buffer) over
//! every texel of `rows_read` (a slot that reads it: a shadow's rows are bound in place as a texel
//! buffer, `texture/texel_rows.rs`) or `rows_sample` (a slot that samples it: a shadow is made at
//! that use and copied in, `texture/lazy_shadow.rs`), both writing into one row of a small buffer
//! (`columns` 0), commits it, waits for it and drops the texture. The shapes: 1920x1080 (7.9 MiB),
//! 4096x2048 (32 MiB), 4096x3840 (60 MiB, under udmabuf's default 64 MiB a dma-buf) and 4096x5376
//! (84 MiB, past it). References: for each kind of pages a blit of the scattered buffer's texels
//! into a `Private` texture (what a shadow's copy in moves, alone), and `rows_sample` over that
//! texture (the dispatch over the device's own image). Every arm runs once a round, the rounds
//! [`ROUNDS`], the arms' order turned by one each round; the first [`WARM`] rounds are not counted.
//!
//! Printed (`TEXSHADOW`), per arm, the median and 90th percentile of: `new_texture`'s wall and
//! thread CPU time, the encoding's (the command buffer and its encoders, where a shadow left
//! unmade is made), `commit`, the wait, the command buffer's GPU time (`gpu_end_time -
//! gpu_start_time`, placed by calibrated timestamps), the texture's drop, and the whole cycle's
//! wall time, thread CPU time and process CPU time (every thread's, the runtime's included); and
//! per cycle, the source's calls, refusals and time, and the moves of the counts that show the
//! path (in place, shadowed, made at use, rows bound), the objects made and freed, the
//! shadow's copies in and back and merges, and the submissions' calls into the driver, their time
//! and, under `METAL2VULKAN_QUEUE_TIMES`, the submitting thread's CPU time there (`WorkCounts`). `TEXSHADOW-CYCLE` lines give each
//! counted cycle's times.
//!
//! Ignored: run it in release under `--cfg m2v_gpu` on Linux where `/dev/udmabuf` opens,
//! `cargo nextest run --release -p metal2vulkan-runtime --lib --run-ignored all
//! texture_shadow_bench --no-capture`.

use super::encoder_holds::library;
use super::host_os::{process_cpu, thread_cpu};
use super::host_residency::{device, udmabuf_list, udmabuf_openable, Scratch};
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState,
    Device, DmaBufChunk, DmaBufRequest, DmaBufSource, DmaBufStandingRefusal, HostMemory, Origin,
    PipelineOptions, ResourceOptions, Size, Texture, TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::ops::Range;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const PAGE: u64 = 4096;
const WAIT: Duration = Duration::from_secs(60);
/// The longest the bench waits for the device's in-place checks to answer.
const CHECKS: Duration = Duration::from_secs(120);
/// The textures' shapes, `RGBA8Unorm` texels.
const SHAPES: [(u64, u64); 4] = [(1920, 1080), (4096, 2048), (4096, 3840), (4096, 5376)];
/// The rounds, each arm once in each, and the first ones not counted.
const ROUNDS: usize = 12;
const WARM: usize = 2;

/// udmabuf's parameter `name` (`/sys/module/udmabuf/parameters/`), `default` where unread.
fn udmabuf_parameter(name: &str, default: u64) -> u64 {
    std::fs::read_to_string(format!("/sys/module/udmabuf/parameters/{name}"))
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(default)
}

/// The kind of the RAM's pages (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pages {
    Huge,
    Small,
}

impl Pages {
    fn words(self) -> &'static str {
        match self {
            Pages::Huge => "2 MiB pages",
            Pages::Small => "4 KiB pages",
        }
    }
}

/// The size of a huge page, which the RAM's mapping is aligned to.
const HUGE: usize = 2 << 20;

/// The guest's RAM: a sealed `memfd` of `length` bytes, each page written.
struct Ram {
    fd: Arc<OwnedFd>,
}

impl Ram {
    /// The RAM in `pages` (the module docs): its pages written through a shared mapping of the
    /// whole file, aligned to [`HUGE`] and advised as `pages` says, which goes once they are.
    fn new(length: u64, pages: Pages) -> Ram {
        // SAFETY: a new anonymous file, owned at once, then sized and sealed.
        let fd = unsafe {
            let fd = libc::memfd_create(
                c"m2v-texture-shadow-bench".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            );
            assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
            OwnedFd::from_raw_fd(fd)
        };
        // SAFETY: the descriptor is this object's.
        unsafe {
            assert_eq!(libc::ftruncate(fd.as_raw_fd(), length as libc::off_t), 0);
            let seals = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
            assert_eq!(libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals), 0);
        }
        let reserved = length as usize + HUGE;
        // SAFETY: a new reservation, then the file mapped over its first aligned bytes.
        let reservation = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                reserved,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        assert_ne!(reservation, libc::MAP_FAILED, "mmap of the reservation");
        let offset = (reservation as usize).next_multiple_of(HUGE) - reservation as usize;
        // SAFETY: inside the reservation, mapped over by the whole file.
        let mapped = unsafe {
            libc::mmap(
                reservation.cast::<u8>().add(offset).cast(),
                length as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd.as_raw_fd(),
                0,
            )
        };
        assert_ne!(mapped, libc::MAP_FAILED, "mmap of the file");
        let advice = match pages {
            Pages::Huge => libc::MADV_HUGEPAGE,
            Pages::Small => libc::MADV_NOHUGEPAGE,
        };
        // SAFETY: the file's mapping, just made.
        let advised = unsafe { libc::madvise(mapped, length as usize, advice) };
        let block: Vec<u8> = (0..1u32 << 20).map(|at| (at % 251) as u8).collect();
        for at in (0..length as usize).step_by(block.len()) {
            let bytes = &block[..(length as usize - at).min(block.len())];
            // SAFETY: inside the file's mapping, which nothing else reaches yet.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    mapped.cast::<u8>().add(at),
                    bytes.len(),
                )
            };
        }
        let pmd = pmd_mapped_kib(mapped as u64, mapped as u64 + length);
        // SAFETY: the reservation and the mapping over it are this call's, and no buffer lies
        // over them: the file keeps its pages.
        unsafe { libc::munmap(reservation, reserved) };
        println!(
            "TEXSHADOW RAM of {length} bytes in {}: madvise {advised}, ShmemPmdMapped {} KiB",
            pages.words(),
            pmd.map_or_else(|| "unread".to_owned(), |kib| kib.to_string()),
        );
        Ram { fd: Arc::new(fd) }
    }
}

/// The `ShmemPmdMapped` kibibytes of the process's mappings within the host bytes `start..end`
/// (`/proc/self/smaps`): their bytes the kernel mapped by huge pages. `None` where unread.
fn pmd_mapped_kib(start: u64, end: u64) -> Option<u64> {
    let smaps = std::fs::read_to_string("/proc/self/smaps").ok()?;
    let (mut within, mut kib) = (false, 0);
    for line in smaps.lines() {
        let range = line
            .split_once(' ')
            .and_then(|(range, _)| range.split_once('-'));
        let bounds = range.and_then(|(from, to)| {
            Some((
                u64::from_str_radix(from, 16).ok()?,
                u64::from_str_radix(to, 16).ok()?,
            ))
        });
        if let Some((from, to)) = bounds {
            within = start <= from && to <= end;
        } else if let Some(value) = line.strip_prefix("ShmemPmdMapped:") {
            let value = value.trim().trim_end_matches("kB").trim();
            kib += value.parse::<u64>().ok().filter(|_| within).unwrap_or(0);
        }
    }
    Some(kib)
}

/// The settings of shmem's transparent huge pages, in words.
fn shmem_huge_settings() -> String {
    let read = |path: &str| {
        let value = std::fs::read_to_string(path).unwrap_or_else(|_| "unread".to_owned());
        value.trim().to_owned()
    };
    format!(
        "shmem_enabled {}, hugepages-2048kB/shmem_enabled {}",
        read("/sys/kernel/mm/transparent_hugepage/shmem_enabled"),
        read("/sys/kernel/mm/transparent_hugepage/hugepages-2048kB/shmem_enabled"),
    )
}

/// A view of the RAM: page `i` of it mapped shared from file offset `file[i]`, each run of
/// consecutive file pages one mapping.
struct View {
    start: NonNull<u8>,
    file: Arc<Vec<u64>>,
}

impl View {
    fn new(ram: &Ram, file: Vec<u64>) -> View {
        let length = file.len() * PAGE as usize;
        // SAFETY: a new reservation of the view's addresses, which the runs then replace.
        let start = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        assert_ne!(start, libc::MAP_FAILED, "mmap of the reservation");
        let mut at = 0;
        for (offset, bytes) in runs(&file, 0..file.len()) {
            // SAFETY: inside the view's own reservation, mapped over by a run of the file.
            let mapped = unsafe {
                libc::mmap(
                    start.cast::<u8>().add(at * PAGE as usize).cast(),
                    bytes as usize,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    ram.fd.as_raw_fd(),
                    offset as libc::off_t,
                )
            };
            assert_ne!(mapped, libc::MAP_FAILED, "mmap of a run");
            at += (bytes / PAGE) as usize;
        }
        View {
            start: NonNull::new(start.cast()).unwrap(),
            file: Arc::new(file),
        }
    }

    fn length(&self) -> u64 {
        self.file.len() as u64 * PAGE
    }
}

impl Drop for View {
    fn drop(&mut self) {
        // SAFETY: the view's own mappings, which no buffer lies over any more (the bench drops
        // its buffers and device first).
        unsafe { libc::munmap(self.start.as_ptr().cast(), self.length() as usize) };
    }
}

/// The runs of file pages `pages` of a view whose pages lie at `file`: each a first byte and a
/// length, in order.
fn runs(file: &[u64], pages: Range<usize>) -> Vec<(u64, u64)> {
    let mut runs: Vec<(u64, u64)> = Vec::new();
    for page in pages {
        match runs.last_mut() {
            Some((offset, bytes)) if *offset + *bytes == file[page] => *bytes += PAGE,
            _ => runs.push((file[page], PAGE)),
        }
    }
    runs
}

/// What a source did: its calls, the ones it refused, the chunks it made, and its nanoseconds.
#[derive(Default)]
struct Tally {
    calls: AtomicU64,
    refused: AtomicU64,
    chunks: AtomicU64,
    nanos: AtomicU64,
}

impl Tally {
    fn read(&self) -> [u64; 4] {
        [&self.calls, &self.refused, &self.chunks, &self.nanos].map(|n| n.load(Ordering::Relaxed))
    }
}

/// udmabuf's limits: the runs a list holds, and the pages a dma-buf holds.
#[derive(Clone, Copy)]
struct Limits {
    runs: usize,
    pages: usize,
}

/// A source of the view's pages, as a consumer's makes them (the module docs), tallied in
/// `tally`.
fn source(ram: &Ram, view: &View, limits: Limits, tally: &Arc<Tally>) -> DmaBufSource {
    let fd = Arc::clone(&ram.fd);
    let file = Arc::clone(&view.file);
    let base = view.start.as_ptr() as usize;
    let tally = Arc::clone(tally);
    Arc::new(move |request: DmaBufRequest| {
        let started = Instant::now();
        let answer = answer(&fd, (base, file.as_slice()), limits, request);
        tally.calls.fetch_add(1, Ordering::Relaxed);
        match &answer {
            Ok(chunks) => tally
                .chunks
                .fetch_add(chunks.len() as u64, Ordering::Relaxed),
            Err(_) => tally.refused.fetch_add(1, Ordering::Relaxed),
        };
        let nanos = started.elapsed().as_nanos() as u64;
        tally.nanos.fetch_add(nanos, Ordering::Relaxed);
        answer
    })
}

/// The chunks of `request`'s pages of the view at `base` whose pages lie at `file`: from the
/// first page, as many of the request's granules as one udmabuf list holds, again until the last.
fn answer(
    fd: &OwnedFd,
    (base, file): (usize, &[u64]),
    limits: Limits,
    request: DmaBufRequest,
) -> std::io::Result<Vec<DmaBufChunk>> {
    let at = (request.start.as_ptr() as usize)
        .checked_sub(base)
        .ok_or_else(|| std::io::Error::other("pages before the view"))?;
    let first = at / PAGE as usize;
    let end = first + (request.length / PAGE) as usize;
    if end > file.len() {
        return Err(std::io::Error::other("pages past the view"));
    }
    let granule = ((request.granule / PAGE) as usize).max(1);
    let starts = |pages: Range<usize>, from: usize| {
        pages
            .filter(|&page| page == from || file[page] != file[page - 1] + PAGE)
            .count()
    };
    let mut chunks = Vec::new();
    let mut from = first;
    while from < end {
        let (mut to, mut listed) = (from, 0);
        while to < end {
            let next = (to + granule).min(end);
            let more = starts(to..next, from);
            if listed + more > limits.runs || next - from > limits.pages {
                break;
            }
            (to, listed) = (next, listed + more);
        }
        // Past one udmabuf's runs or pages, for a range that holds no kept dma-buf: no range that
        // holds it fits either, so the refusal stands for them all (`DmaBufStandingRefusal`).
        if to == from {
            let needed = starts(from..(from + granule).min(end), from);
            return Err(std::io::Error::other(DmaBufStandingRefusal::new(format!(
                "{} pages in {needed} runs, past one udmabuf's {} runs or {} pages",
                (from + granule).min(end) - from,
                limits.runs,
                limits.pages
            ))));
        }
        chunks.push(DmaBufChunk {
            file: Arc::new(udmabuf_list(fd, &runs(file, from..to))?),
            offset: 0,
            length: (to - from) as u64 * PAGE,
        });
        from = to;
    }
    if chunks.len() > 1 && !request.length.is_multiple_of(request.granule) {
        return Err(std::io::Error::other("the bench pads no chunks"));
    }
    Ok(chunks)
}

/// How a buffer's pages lie, and whether its texture's image parks (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lie {
    InPlace,
    Unparked,
    Shadow,
}

/// Every lie, in the order of [`Shaped::buffers`].
const LIES: [Lie; 3] = [Lie::InPlace, Lie::Unparked, Lie::Shadow];

/// Every kind of pages, in the order of [`Shaped::buffers`].
const PAGES: [Pages; 2] = [Pages::Huge, Pages::Small];

/// The kernel of an arm's dispatches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    Read,
    Sample,
}

/// What an arm runs a round.
#[derive(Clone, Copy, Debug)]
enum Work {
    /// A cycle over a buffer of `Lie` in `Pages`, of `dispatches` of `Kernel`.
    Cycle(Pages, Lie, Kernel, usize),
    /// The blit of the scattered buffer's texels in `Pages` into the shape's `Private` texture.
    CopyAlone(Pages),
    /// `rows_sample` over the shape's `Private` texture.
    SampleOwn,
}

/// One shape's buffers and sources, by kind of pages and lie, and its `Private` texture.
struct Shaped {
    width: u64,
    height: u64,
    buffers: [[Buffer; 3]; 2],
    tallies: [[Arc<Tally>; 3]; 2],
    own: Texture,
}

/// What every arm runs on.
struct On {
    device: Device,
    queue: CommandQueue,
    read: ComputePipelineState,
    sample: ComputePipelineState,
    output: Buffer,
}

/// An arm's times (microseconds) and the counts it moved over its counted cycles.
#[derive(Default)]
struct Figures {
    times: [Vec<f64>; TIMES.len()],
    moved: [u64; MOVED.len()],
    tally: [u64; 4],
    cycles: u64,
}

/// The times an arm keeps, in [`Figures::times`]' order.
const TIMES: [&str; 12] = [
    "make",
    "make cpu",
    "encode",
    "encode cpu",
    "commit",
    "wait",
    "gpu",
    "drop",
    "drop cpu",
    "cycle",
    "cycle cpu",
    "process cpu",
];

/// The counts an arm sums, per cycle.
const MOVED: [(&str, fn(&WorkCounts) -> u64); 24] = [
    ("in place", |c| c.host_textures_in_place),
    ("taken", |c| c.dma_buf_images_taken),
    ("parked", |c| c.dma_buf_images_parked),
    ("refused source", |c| c.host_texture_shadows_refused_source),
    ("check pending", |c| c.host_texture_shadows_check_pending),
    ("shadowed", |c| c.shadowed_textures),
    ("left unmade", |c| c.shadows_left_unmade),
    ("made at use", |c| c.shadows_made_at_use),
    ("rows bound", |c| c.texel_rows_bound),
    ("dma-buf images", |c| c.dma_buf_images),
    ("dma-buf imports", |c| c.dma_buf_imports),
    ("images made", |c| c.object_churn.vk_images_made),
    ("allocations", |c| c.object_churn.vk_memory_allocations),
    ("frees", |c| c.object_churn.vk_memory_freed),
    ("MiB allocated", |c| c.device_memory_allocated >> 20),
    ("shadow copies", |c| c.shadow_copies),
    ("texels in", |c| c.shadow_texels_copied_in),
    ("texels back", |c| c.shadow_texels_copied_back),
    ("ins dispatched", |c| c.shadow_copies_in_dispatched),
    ("merge words", |c| c.shadow_merge_words_examined),
    ("submits", |c| c.queue_submissions),
    ("submit us", |c| c.batch_submit_nanos / 1000),
    ("submit cpu us", |c| c.batch_submit_cpu_nanos / 1000),
    ("long submits", |c| c.submit_long_calls),
];

/// The `at` quantile of `values`.
fn quantile(values: &[f64], at: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * at) as usize).min(sorted.len().saturating_sub(1));
    sorted.get(index).copied().unwrap_or(0.0)
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e6
}

/// An `RGBA8Unorm` 2D texture of `width` x `height`, `ShaderRead`, of `options`.
fn descriptor(width: u64, height: u64, options: ResourceOptions) -> TextureDescriptor {
    let mut descriptor =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, width, height, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = options;
    descriptor
}

/// Encode a dispatch of `state` over every texel of `texture` (`width` x `height`) into
/// `command_buffer`, in an encoder of its own, its output one row of `output`.
fn dispatch(
    on: &On,
    command_buffer: &CommandBuffer,
    (state, texture): (&ComputePipelineState, &Texture),
    (width, height): (u64, u64),
) {
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_buffer(Some(&on.output), 0, 0).unwrap();
    encoder.set_bytes(&0u32.to_le_bytes(), 1).unwrap();
    (encoder.dispatch_threads(Size::new(width, height, 1), Size::new(8, 8, 1))).unwrap();
    encoder.end_encoding().unwrap();
}

/// Commit `command_buffer`, wait for it, and answer the commit's and the wait's times and its GPU
/// time, in microseconds.
fn run(command_buffer: &CommandBuffer) -> [f64; 3] {
    let committed = Instant::now();
    command_buffer.commit().unwrap();
    let commit = micros(committed.elapsed());
    let waited = Instant::now();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    let wait = micros(waited.elapsed());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let gpu = command_buffer.gpu_end_time().unwrap() - command_buffer.gpu_start_time().unwrap();
    [commit, wait, gpu * 1e6]
}

/// One round's run of `work` over `shaped`: its times, in [`TIMES`]' order (those it has no part
/// of 0), and the counts and the source's tally before and after.
fn once(
    on: &On,
    shaped: &Shaped,
    work: Work,
) -> ([f64; TIMES.len()], [WorkCounts; 2], [[u64; 4]; 2]) {
    let size = (shaped.width, shaped.height);
    // An in-place cycle's texture takes the image its warm-up parked (the module docs).
    if let Work::Cycle(pages, Lie::InPlace, ..) = work {
        let buffer = &shaped.buffers[pages as usize][Lie::InPlace as usize];
        let parked = on.device.work_counts().dma_buf_images_parked;
        let options = ResourceOptions::shared();
        let texture =
            (buffer.new_texture(&descriptor(size.0, size.1, options), 0, size.0 * 4)).unwrap();
        let command_buffer = on.queue.command_buffer().unwrap();
        dispatch(on, &command_buffer, (&on.read, &texture), size);
        run(&command_buffer);
        drop(texture);
        drop(command_buffer);
        let waited = Instant::now();
        while on.device.work_counts().dma_buf_images_parked == parked && waited.elapsed() < WAIT {
            std::thread::sleep(Duration::from_micros(100));
        }
    }
    let tally = match work {
        Work::Cycle(pages, lie, ..) => &shaped.tallies[pages as usize][lie as usize],
        Work::CopyAlone(pages) => &shaped.tallies[pages as usize][Lie::Shadow as usize],
        Work::SampleOwn => &shaped.tallies[Pages::Small as usize][Lie::Shadow as usize],
    };
    let (tallied, before) = (tally.read(), on.device.work_counts());
    let (wall, cpu, process) = (Instant::now(), thread_cpu(), process_cpu());
    let mut times = [0.0; TIMES.len()];
    let mut images_live = None;
    match work {
        Work::Cycle(pages, lie, kernel, dispatches) => {
            let buffer = &shaped.buffers[pages as usize][lie as usize];
            if lie == Lie::Unparked {
                on.device.park_dma_buf_images_at_most(0);
            }
            let made = (Instant::now(), thread_cpu());
            let texture = buffer
                .new_texture(
                    &descriptor(size.0, size.1, ResourceOptions::shared()),
                    0,
                    size.0 * 4,
                )
                .unwrap();
            times[0] = micros(made.0.elapsed());
            times[1] = micros(thread_cpu() - made.1);
            let encoded = (Instant::now(), thread_cpu());
            let command_buffer = on.queue.command_buffer().unwrap();
            let state = match kernel {
                Kernel::Read => &on.read,
                Kernel::Sample => &on.sample,
            };
            for _ in 0..dispatches {
                dispatch(on, &command_buffer, (state, &texture), size);
            }
            times[2] = micros(encoded.0.elapsed());
            times[3] = micros(thread_cpu() - encoded.1);
            [times[4], times[5], times[6]] = run(&command_buffer);
            if lie == Lie::Unparked {
                images_live = Some(on.device.work_counts().dma_buf_image_memories_live);
            }
            let dropped = (Instant::now(), thread_cpu());
            drop(texture);
            drop(command_buffer);
            times[7] = micros(dropped.0.elapsed());
            times[8] = micros(thread_cpu() - dropped.1);
        }
        Work::CopyAlone(pages) => {
            let encoded = (Instant::now(), thread_cpu());
            let command_buffer = on.queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_texture(
                &shaped.buffers[pages as usize][Lie::Shadow as usize],
                0,
                size.0 * 4,
                size.0 * 4 * size.1,
                Size::new(size.0, size.1, 1),
                &shaped.own,
                0,
                0,
                Origin::new(0, 0, 0),
                BlitOption::None,
            )
            .unwrap();
            blit.end_encoding().unwrap();
            times[2] = micros(encoded.0.elapsed());
            times[3] = micros(thread_cpu() - encoded.1);
            [times[4], times[5], times[6]] = run(&command_buffer);
        }
        Work::SampleOwn => {
            let encoded = (Instant::now(), thread_cpu());
            let command_buffer = on.queue.command_buffer().unwrap();
            dispatch(on, &command_buffer, (&on.sample, &shaped.own), size);
            times[2] = micros(encoded.0.elapsed());
            times[3] = micros(thread_cpu() - encoded.1);
            [times[4], times[5], times[6]] = run(&command_buffer);
        }
    }
    times[9] = micros(wall.elapsed());
    times[10] = micros(thread_cpu() - cpu);
    times[11] = micros(process_cpu() - process);
    // An unparked cycle's image goes before the device parks again, outside its times: its
    // command buffer's completion may let go of it a moment after the drop.
    if let Some(live) = images_live {
        let waited = Instant::now();
        while on.device.work_counts().dma_buf_image_memories_live >= live && waited.elapsed() < WAIT
        {
            std::thread::sleep(Duration::from_micros(100));
        }
        on.device
            .park_dma_buf_images_at_most(crate::host_residency::PARKED_MOST);
    }
    let after = on.device.work_counts();
    (times, [before, after], [tallied, tally.read()])
}

/// An arm's name.
fn name(shaped: &Shaped, work: Work) -> String {
    let bytes = shaped.width * shaped.height * 4;
    let shape = format!(
        "{}x{} ({:.1} MiB)",
        shaped.width,
        shaped.height,
        bytes as f64 / 1048576.0
    );
    match work {
        Work::Cycle(pages, lie, kernel, dispatches) => {
            format!("{shape} {} {lie:?} {kernel:?} x{dispatches}", pages.words())
        }
        Work::CopyAlone(pages) => format!(
            "{shape} {} reference: blit of the scattered texels",
            pages.words()
        ),
        Work::SampleOwn => format!("{shape} reference: Sample x1 of a Private texture"),
    }
}

/// The bytes of a shape's buffer: its texels, and 64 KiB more.
fn buffer_length((width, height): (u64, u64)) -> u64 {
    (width * height * 4).next_multiple_of(64 << 10) + (64 << 10)
}

/// The shape's buffers over three views of each of `rams` (the module docs, in [`PAGES`]' order),
/// each view's file pages taken from `next` on.
fn shaped(
    on: &On,
    rams: &[Ram; 2],
    (width, height): (u64, u64),
    next: &mut u64,
    limits: Limits,
) -> (Shaped, Vec<View>) {
    let (mut buffers, mut tallies, mut views) = (Vec::new(), Vec::new(), Vec::new());
    for (ram, kind) in rams.iter().zip(PAGES) {
        let (made, sourced, viewed) = shaped_in(on, (ram, kind), (width, height), *next, limits);
        buffers.push(made);
        tallies.push(sourced);
        views.extend(viewed);
    }
    *next += 3 * buffer_length((width, height)) / PAGE;
    let own = (on.device)
        .new_texture(&descriptor(width, height, ResourceOptions::private()))
        .unwrap();
    let shaped = Shaped {
        width,
        height,
        buffers: buffers.try_into().ok().unwrap(),
        tallies: tallies.try_into().ok().unwrap(),
        own,
    };
    (shaped, views)
}

/// The shape's three buffers over views of `ram`, its pages of `kind`, each view's file pages
/// taken from page `next` on: in [`LIES`]' order, with their sources' tallies and the views.
fn shaped_in(
    on: &On,
    (ram, kind): (&Ram, Pages),
    (width, height): (u64, u64),
    mut next: u64,
    limits: Limits,
) -> ([Buffer; 3], [Arc<Tally>; 3], [View; 3]) {
    let length = buffer_length((width, height));
    let pages = length / PAGE;
    let mut consecutive = || {
        let file: Vec<u64> = (0..pages).map(|page| (next + page) * PAGE).collect();
        next += pages;
        file
    };
    let (in_place, unparked) = (consecutive(), consecutive());
    // Each page another of the file: a stride coprime with the pages, so no two neighbours are
    // neighbours in the file.
    let stride = (pages / 2 + 1..)
        .find(|stride| gcd(*stride, pages) == 1)
        .unwrap();
    let scattered: Vec<u64> = (0..pages)
        .map(|page| (next + page * stride % pages) * PAGE)
        .collect();
    let views = [
        View::new(ram, in_place),
        View::new(ram, unparked),
        View::new(ram, scattered),
    ];
    let tallies = [(); 3].map(|()| Arc::new(Tally::default()));
    let device = &on.device;
    let buffers = [0, 1, 2].map(|at| {
        let before = device.work_counts();
        let source = source(ram, &views[at], limits, &tallies[at]);
        // SAFETY: the views outlive the buffers (the bench drops them first), and the source
        // makes dma-bufs of their pages.
        let buffer = unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::with_dma_buf(views[at].start, source),
                length,
                ResourceOptions::shared(),
                None,
            )
        }
        .unwrap();
        let after = device.work_counts();
        let [calls, refused, chunks, nanos] = tallies[at].read();
        println!(
            "TEXSHADOW {width}x{height}: the {} {:?} buffer of {length} bytes over {} runs: \
             source calls {calls}, refused {refused}, chunks {chunks}, {:.1} us; dma-buf imports \
             {}, sparse buffers {}, copies of refused pages {}",
            kind.words(),
            LIES[at],
            runs(&views[at].file, 0..pages as usize).len(),
            nanos as f64 / 1e3,
            after.dma_buf_imports - before.dma_buf_imports,
            after.dma_buf_sparse_buffers - before.dma_buf_sparse_buffers,
            after.copied_host_buffers_device_refused - before.copied_host_buffers_device_refused,
        );
        buffer
    });
    (buffers, tallies, views)
}

fn gcd(a: u64, b: u64) -> u64 {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}

/// `kernel` of `examples/texel_rows.metal`.
fn state(device: &Device, kernel: &str) -> ComputePipelineState {
    let function = library(device, "examples/texel_rows.ll")
        .new_function_with_name(kernel)
        .unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

#[test]
#[ignore = "a release benchmark: run with --run-ignored all --no-capture"]
fn texture_shadow_bench() {
    if !udmabuf_openable("texture_shadow_bench") {
        return;
    }
    let limits = Limits {
        runs: udmabuf_parameter("list_limit", 1024) as usize,
        pages: (udmabuf_parameter("size_limit_mb", 64) << 20) as usize / PAGE as usize,
    };
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let checked = Instant::now();
    assert!(
        device.settle_in_place_checks(CHECKS),
        "the in-place checks never answered"
    );
    println!(
        "TEXSHADOW on {}: in-place checks answered in {:.2} s; udmabuf list_limit {} runs, \
         size_limit {} pages; {}",
        device.name(),
        checked.elapsed().as_secs_f64(),
        limits.runs,
        limits.pages,
        shmem_huge_settings()
    );
    let widest = SHAPES.iter().map(|&(width, _)| width).max().unwrap();
    let on = On {
        queue: device.new_command_queue().unwrap(),
        read: state(&device, "rows_read"),
        sample: state(&device, "rows_sample"),
        output: device
            .new_buffer(widest * 16, ResourceOptions::private())
            .unwrap(),
        device,
    };
    let ram_bytes: u64 = (SHAPES.iter()).map(|&shape| 3 * buffer_length(shape)).sum();
    let rams = PAGES.map(|kind| Ram::new(ram_bytes, kind));
    let mut next = 0;
    let (shapes, views): (Vec<Shaped>, Vec<Vec<View>>) = (SHAPES.iter())
        .map(|&shape| shaped(&on, &rams, shape, &mut next, limits))
        .unzip();
    let mut arms: Vec<(usize, Work, Figures)> = Vec::new();
    for shape in 0..shapes.len() {
        for pages in PAGES {
            for lie in LIES {
                for kernel in [Kernel::Read, Kernel::Sample] {
                    for dispatches in [1, 2] {
                        let work = Work::Cycle(pages, lie, kernel, dispatches);
                        arms.push((shape, work, Figures::default()));
                    }
                }
            }
            arms.push((shape, Work::CopyAlone(pages), Figures::default()));
        }
        arms.push((shape, Work::SampleOwn, Figures::default()));
    }
    let started = Instant::now();
    for round in 0..ROUNDS {
        let count = arms.len();
        for turn in 0..count {
            let (shape, work, figures) = &mut arms[(turn + round) % count];
            let shaped = &shapes[*shape];
            let (times, [before, after], [tallied, tally]) = once(&on, shaped, *work);
            if round < WARM {
                continue;
            }
            for (kept, time) in figures.times.iter_mut().zip(times) {
                kept.push(time);
            }
            for (sum, (_, of)) in figures.moved.iter_mut().zip(MOVED) {
                *sum += of(&after) - of(&before);
            }
            for (sum, (now, then)) in figures.tally.iter_mut().zip(tally.into_iter().zip(tallied)) {
                *sum += now - then;
            }
            figures.cycles += 1;
            let line: Vec<String> = (TIMES.iter().zip(times))
                .filter(|(_, time)| *time > 0.0)
                .map(|(what, time)| format!("{what} {time:.1}"))
                .collect();
            println!(
                "TEXSHADOW-CYCLE {} round {round}: {} us",
                name(shaped, *work),
                line.join(", ")
            );
        }
    }
    println!(
        "TEXSHADOW: {} rounds of {} arms in {:.1} s; median / p90, us; counts per cycle",
        ROUNDS,
        arms.len(),
        started.elapsed().as_secs_f64()
    );
    for (shape, work, figures) in &arms {
        let cycles = figures.cycles.max(1) as f64;
        let times: Vec<String> = (TIMES.iter().zip(&figures.times))
            .filter(|(_, values)| values.iter().any(|&time| time > 0.0))
            .map(|(what, values)| {
                format!(
                    "{what} {:.1} / {:.1}",
                    quantile(values, 0.5),
                    quantile(values, 0.9)
                )
            })
            .collect();
        let moved: Vec<String> = (MOVED.iter().zip(figures.moved))
            .filter(|(_, sum)| *sum > 0)
            .map(|((what, _), sum)| format!("{what} {:.2}", sum as f64 / cycles))
            .collect();
        let [calls, refused, chunks, nanos] = figures.tally;
        println!(
            "TEXSHADOW {}: {}; source calls {:.2} (refused {:.2}, chunks {:.2}, {:.1} us); {}",
            name(&shapes[*shape], *work),
            times.join(", "),
            calls as f64 / cycles,
            refused as f64 / cycles,
            chunks as f64 / cycles,
            nanos as f64 / 1e3 / cycles,
            moved.join(", ")
        );
    }
    drop(shapes);
    drop(on);
    drop(views);
}
