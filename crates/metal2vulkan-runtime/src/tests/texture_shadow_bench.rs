//! What a texture over a no-copy buffer of host pages costs a cycle that makes it, uses it and
//! drops it, where its image lies in place over the pages and where it is a shadow of them: the
//! pages a source makes dma-bufs of (`dma_buf.rs`; RADV refuses a shared mapping's pointer), as a
//! consumer's guest makes textures over a buffer of its RAM, uses each in one or two dispatches and
//! drops it.
//!
//! The RAM is a `memfd` sealed against growing and shrinking and mapped shared, as QEMU's
//! `memory-backend-memfd,share=on` makes it. For each shape, two no-copy buffers over views of it:
//! - **in place**: a view of consecutive file pages, one run, which one udmabuf holds: the
//!   texture's image is a `DRM_FORMAT_MOD_LINEAR` image over a dma-buf of its own pages
//!   (`Device::dma_buf_image`, `host_textures_in_place`);
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
//! (84 MiB, past it). Two references a shape: a blit of the scattered buffer's texels into a
//! `Private` texture (what a shadow's copy in moves, alone) and `rows_sample` over that texture
//! (the dispatch over the device's own image). Every arm runs once a round, the rounds
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

/// The guest's RAM: a sealed `memfd` of `length` bytes, each page written.
struct Ram {
    fd: Arc<OwnedFd>,
}

impl Ram {
    fn new(length: u64) -> Ram {
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
        let block: Vec<u8> = (0..1u32 << 20).map(|at| (at % 251) as u8).collect();
        for at in (0..length).step_by(block.len()) {
            let bytes = &block[..(length - at).min(block.len() as u64) as usize];
            // SAFETY: `bytes` is readable for its length.
            let written = unsafe {
                libc::pwrite(
                    fd.as_raw_fd(),
                    bytes.as_ptr().cast(),
                    bytes.len(),
                    at as libc::off_t,
                )
            };
            assert_eq!(written, bytes.len() as isize, "pwrite");
        }
        Ram { fd: Arc::new(fd) }
    }
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

/// How a buffer's pages lie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lie {
    InPlace,
    Shadow,
}

/// The kernel of an arm's dispatches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    Read,
    Sample,
}

/// What an arm runs a round.
#[derive(Clone, Copy, Debug)]
enum Work {
    /// A cycle over a buffer of `Lie`, of `dispatches` of `Kernel`.
    Cycle(Lie, Kernel, usize),
    /// The blit of the scattered buffer's texels into the shape's `Private` texture.
    CopyAlone,
    /// `rows_sample` over the shape's `Private` texture.
    SampleOwn,
}

/// One shape's buffers, sources and `Private` texture.
struct Shaped {
    width: u64,
    height: u64,
    buffers: [Buffer; 2],
    tallies: [Arc<Tally>; 2],
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
const MOVED: [(&str, fn(&WorkCounts) -> u64); 22] = [
    ("in place", |c| c.host_textures_in_place),
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
    let tally = match work {
        Work::Cycle(lie, ..) => &shaped.tallies[lie as usize],
        _ => &shaped.tallies[Lie::Shadow as usize],
    };
    let (tallied, before) = (tally.read(), on.device.work_counts());
    let (wall, cpu, process) = (Instant::now(), thread_cpu(), process_cpu());
    let mut times = [0.0; TIMES.len()];
    match work {
        Work::Cycle(lie, kernel, dispatches) => {
            let buffer = &shaped.buffers[lie as usize];
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
            let dropped = (Instant::now(), thread_cpu());
            drop(texture);
            drop(command_buffer);
            times[7] = micros(dropped.0.elapsed());
            times[8] = micros(thread_cpu() - dropped.1);
        }
        Work::CopyAlone => {
            let encoded = (Instant::now(), thread_cpu());
            let command_buffer = on.queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_from_buffer_to_texture(
                &shaped.buffers[Lie::Shadow as usize],
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
        Work::Cycle(lie, kernel, dispatches) => {
            format!("{shape} {lie:?} {kernel:?} x{dispatches}")
        }
        Work::CopyAlone => format!("{shape} reference: blit of the scattered texels"),
        Work::SampleOwn => format!("{shape} reference: Sample x1 of a Private texture"),
    }
}

/// The shape's buffers over two views of `ram` (the module docs), each view's file pages taken
/// from `next` on.
fn shaped(
    on: &On,
    ram: &Ram,
    (width, height): (u64, u64),
    next: &mut u64,
    limits: Limits,
) -> (Shaped, [View; 2]) {
    let length = (width * height * 4).next_multiple_of(64 << 10) + (64 << 10);
    let pages = length / PAGE;
    let consecutive: Vec<u64> = (0..pages).map(|page| (*next + page) * PAGE).collect();
    *next += pages;
    // Each page another of the file: a stride coprime with the pages, so no two neighbours are
    // neighbours in the file.
    let stride = (pages / 2 + 1..)
        .find(|stride| gcd(*stride, pages) == 1)
        .unwrap();
    let scattered: Vec<u64> = (0..pages)
        .map(|page| (*next + page * stride % pages) * PAGE)
        .collect();
    *next += pages;
    let views = [View::new(ram, consecutive), View::new(ram, scattered)];
    let tallies = [Arc::new(Tally::default()), Arc::new(Tally::default())];
    let device = &on.device;
    let buffers = [0, 1].map(|at| {
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
            "TEXSHADOW {width}x{height}: the {:?} buffer of {length} bytes over {} runs: source \
             calls {calls}, refused {refused}, chunks {chunks}, {:.1} us; dma-buf imports {}, \
             sparse buffers {}, copies of refused pages {}",
            [Lie::InPlace, Lie::Shadow][at],
            runs(&views[at].file, 0..pages as usize).len(),
            nanos as f64 / 1e3,
            after.dma_buf_imports - before.dma_buf_imports,
            after.dma_buf_sparse_buffers - before.dma_buf_sparse_buffers,
            after.copied_host_buffers_device_refused - before.copied_host_buffers_device_refused,
        );
        buffer
    });
    let own = device
        .new_texture(&descriptor(width, height, ResourceOptions::private()))
        .unwrap();
    let shaped = Shaped {
        width,
        height,
        buffers,
        tallies,
        own,
    };
    (shaped, views)
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
         size_limit {} pages",
        device.name(),
        checked.elapsed().as_secs_f64(),
        limits.runs,
        limits.pages
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
    let ram_bytes: u64 = (SHAPES.iter())
        .map(|&(width, height)| 2 * ((width * height * 4).next_multiple_of(64 << 10) + (64 << 10)))
        .sum();
    let ram = Ram::new(ram_bytes);
    let mut next = 0;
    let (shapes, views): (Vec<Shaped>, Vec<[View; 2]>) = (SHAPES.iter())
        .map(|&shape| shaped(&on, &ram, shape, &mut next, limits))
        .unzip();
    let mut arms: Vec<(usize, Work, Figures)> = Vec::new();
    for shape in 0..shapes.len() {
        for lie in [Lie::InPlace, Lie::Shadow] {
            for kernel in [Kernel::Read, Kernel::Sample] {
                for dispatches in [1, 2] {
                    let work = Work::Cycle(lie, kernel, dispatches);
                    arms.push((shape, work, Figures::default()));
                }
            }
        }
        arms.push((shape, Work::CopyAlone, Figures::default()));
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
