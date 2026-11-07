//! A command buffer dropped uncommitted, whose draws' pipelines are still being made, under the
//! validation layer: once the drop returns, nothing of it touches host memory, so the program may
//! unmap the pages its draws wrote at once.
//!
//! The consumer (reims-vgpu) unmaps a guest task's pages as soon as the task's objects are let go
//! of, and relies on this: a dropped, uncommitted command buffer's deferred work, a draw whose
//! pipeline is still being made among it, never reaches the scheduler, which takes only committed
//! batches and hands a host step over only for a committed line's head.
//!
//! The draws are of the owned fixture `ob3_index_buffer`, a vertex function with rasterization
//! disabled that marks `seen[min(vertex_id, 7)]`, here through a no-copy buffer over a second
//! mapping of a page of guest RAM (a shared memory object, as the consumer maps it). On Linux the
//! guest RAM is a `memfd` sealed against growing and shrinking, as QEMU's
//! `memory-backend-memfd,share=on` makes it, and each buffer's memory comes with a source of
//! dma-bufs of its pages (`HostMemory::with_dma_buf`): RADV, which refuses the pointer of shared
//! memory, imports those (`WorkCounts::dma_buf_imports`); a device that imports the pointer never
//! calls the source. The state's
//! creation predicts a point list's pipeline, so a line's and a triangle's draw each hand their
//! own pipeline's making to the device's pool, and the pass records them later
//! (`render/pending.rs`). The pool's test hold (`pool::held`) keeps those makings waiting from
//! before the draws until the command buffer is dropped and the pages unmapped, so the command
//! buffer is dropped with its draws' pipelines still being made in every run, not only where the
//! driver was slower to make them than the encoding was to end.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{Scratch, WAIT};
use super::encoder_holds::library;
#[cfg(target_os = "linux")]
use super::host_residency::udmabuf;
use super::layer_claim::Watched;
use super::pending_rebind::device;
use crate::pipeline::pool::held;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, Error, HostMemory,
    PipelineOptions, RenderPassDescriptor, RenderPipelineState, ResourceOptions, WorkCounts,
};
#[cfg(target_os = "linux")]
use crate::{DmaBufChunk, DmaBufRequest, DmaBufSource};
use metal2vulkan::error::Feature;
use metal2vulkan::metal::PrimitiveType;
use metal2vulkan::pipeline::{RenderPipelineDescriptor, StageSource};
use std::ptr::NonNull;
#[cfg(not(target_os = "linux"))]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(target_os = "linux")]
use std::sync::Arc;
use std::time::Instant;

/// The consumer's guest page: 16 KiB, a whole number of any host's pages.
const GUEST_PAGE: usize = 16 << 10;

/// The byte every page of the guest RAM holds until a draw marks it.
const UNTOUCHED: u8 = 0xa5;

/// The words a draw's vertex function marks, from the buffer's start.
const MARKS: usize = 8;

/// One shared memory object: the consumer's guest RAM, with a mapping of its own through which
/// the CPU writes and reads.
struct GuestRam {
    fd: libc::c_int,
    len: usize,
    view: Mapping,
}

/// A new shared memory object of `len` bytes: a `memfd` sealed against growing and shrinking, so
/// that `/dev/udmabuf` makes dma-bufs of its pages.
#[cfg(target_os = "linux")]
fn shared_memory(len: usize) -> libc::c_int {
    // SAFETY: a new anonymous file, then sized and sealed.
    unsafe {
        let fd = libc::memfd_create(
            c"m2v-dropped-uncommitted".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        );
        assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
        assert_eq!(libc::ftruncate(fd, len as libc::off_t), 0, "ftruncate");
        let seals = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK;
        assert_eq!(libc::fcntl(fd, libc::F_ADD_SEALS, seals), 0, "seals");
        fd
    }
}

/// A new shared memory object of `len` bytes, unlinked at once: only the descriptor names it.
#[cfg(not(target_os = "linux"))]
fn shared_memory(len: usize) -> libc::c_int {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let name = std::ffi::CString::new(format!(
        "/m2v-du-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
    .unwrap();
    // SAFETY: a new shared memory object, unlinked at once.
    unsafe {
        let fd = libc::shm_open(
            name.as_ptr(),
            libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
            0o600 as libc::c_uint,
        );
        assert!(fd >= 0, "shm_open: {}", std::io::Error::last_os_error());
        libc::shm_unlink(name.as_ptr());
        assert_eq!(libc::ftruncate(fd, len as libc::off_t), 0, "ftruncate");
        fd
    }
}

impl GuestRam {
    fn new(len: usize) -> GuestRam {
        let fd = shared_memory(len);
        let view = Mapping::of(fd, 0, len);
        GuestRam { fd, len, view }
    }

    /// The memory of `mapping`, one of its mappings, with a source of one dma-buf of the
    /// object's pages behind each request's.
    #[cfg(target_os = "linux")]
    fn memory(&self, mapping: &Mapping) -> HostMemory {
        // SAFETY: the object's descriptor, open while it lives; the source keeps a duplicate.
        let fd = unsafe { std::os::fd::BorrowedFd::borrow_raw(self.fd) };
        let fd = fd.try_clone_to_owned().unwrap();
        let base = mapping.at.as_ptr() as usize;
        let file_at = mapping.file_at;
        let source: DmaBufSource = Arc::new(move |request: DmaBufRequest| {
            let at = (request.start.as_ptr() as usize)
                .checked_sub(base)
                .ok_or_else(|| std::io::Error::other("pages before the mapping"))?;
            Ok(vec![DmaBufChunk {
                fd: udmabuf(&fd, (file_at + at) as u64, request.length)?,
                length: request.length,
            }])
        });
        // SAFETY: the caller keeps the mapping while a buffer over it lives (each buffer goes
        // first), and each dma-buf holds exactly the object's pages the mapping maps there.
        unsafe { HostMemory::with_dma_buf(mapping.at, source) }
    }

    /// The memory of `mapping`, one of its mappings.
    #[cfg(not(target_os = "linux"))]
    fn memory(&self, mapping: &Mapping) -> HostMemory {
        // SAFETY: the caller keeps the mapping while a buffer over it lives (each buffer goes
        // first).
        unsafe { HostMemory::new(mapping.at) }
    }

    /// A second mapping of its `len` bytes from `offset`, as the consumer maps a task's pages.
    fn map(&self, offset: usize, len: usize) -> Mapping {
        assert!(offset + len <= self.len);
        Mapping::of(self.fd, offset, len)
    }

    /// A copy of its bytes `range`.
    fn read(&self, range: std::ops::Range<usize>) -> Vec<u8> {
        // SAFETY: the view maps every byte of the object, readable while it lives.
        unsafe { std::slice::from_raw_parts(self.view.at.as_ptr().add(range.start), range.len()) }
            .to_vec()
    }

    /// Fill all of it with `byte`.
    fn fill(&self, byte: u8) {
        // SAFETY: as in `read`, and writable.
        unsafe { std::ptr::write_bytes(self.view.at.as_ptr(), byte, self.len) };
    }
}

impl Drop for GuestRam {
    fn drop(&mut self) {
        // SAFETY: the descriptor is this object's; its mappings keep the memory alive.
        unsafe { libc::close(self.fd) };
    }
}

/// A shared, readable and writable mapping of the object's bytes from `file_at`, unmapped when
/// dropped.
struct Mapping {
    at: NonNull<u8>,
    len: usize,
    #[cfg(target_os = "linux")]
    file_at: usize,
}

impl Mapping {
    fn of(fd: libc::c_int, offset: usize, len: usize) -> Mapping {
        // SAFETY: a new mapping of the object's own range, at an address the kernel picks.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                offset as libc::off_t,
            )
        };
        assert_ne!(at, libc::MAP_FAILED, "mmap");
        Mapping {
            at: NonNull::new(at.cast()).unwrap(),
            len,
            #[cfg(target_os = "linux")]
            file_at: offset,
        }
    }

    /// Replace it in place by an inaccessible mapping, so any access of its pages faults.
    fn unmap_in_place(&self) {
        // SAFETY: the mapping's own range, which nothing else uses.
        let unmapped = unsafe {
            libc::mmap(
                self.at.as_ptr().cast(),
                self.len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
                -1,
                0,
            )
        };
        assert_eq!(unmapped, self.at.as_ptr().cast(), "unmap in place");
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the mapping's own range.
        unsafe { libc::munmap(self.at.as_ptr().cast(), self.len) };
    }
}

/// The fixture's state: `ob3_index_buffer`, rasterization disabled.
fn unrasterized_state(device: &Device) -> RenderPipelineState {
    let library = library(device, "engine/ob3_index_buffer.ll");
    let function = library.new_function_with_name("ob3_index_buffer").unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    descriptor.rasterization_enabled = false;
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// `device`'s work counts once its pipeline pool holds no piece (`WorkCounts::pipeline_pool_busy`),
/// waiting at most [`WAIT`].
fn pipelines_settled(device: &Device) -> WorkCounts {
    let until = Instant::now() + WAIT;
    loop {
        let counts = device.work_counts();
        if counts.pipeline_pool_busy == 0 {
            return counts;
        }
        assert!(Instant::now() < until, "the pipeline pool still busy");
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

/// A no-copy buffer over `mapping`, one of `ram`'s, with the consumer's resource options (0).
fn over(device: &Device, ram: &GuestRam, mapping: &Mapping) -> Buffer {
    // SAFETY: the caller keeps the mapping while the buffer lives (each buffer goes first).
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            ram.memory(mapping),
            mapping.len as u64,
            ResourceOptions::default(),
            None,
        )
    };
    match made {
        Ok(buffer) => buffer,
        Err(Error::Unsupported(gap)) if gap.feature == Feature::HostMemory => {
            panic!("this device imports no host memory exactly: {gap:?}")
        }
        Err(error) => panic!("no-copy buffer: {error:?}"),
    }
}

/// Encode, into a new command buffer of `queue`, a pass without attachments whose draws of
/// `state` (a line's eight vertices, then a triangle's six) mark `seen` at vertex buffer 0. Not
/// committed.
fn encode_marks(queue: &CommandQueue, state: &RenderPipelineState, seen: &Buffer) -> CommandBuffer {
    let mut pass = RenderPassDescriptor::new();
    pass.render_target_width = 1;
    pass.render_target_height = 1;
    pass.default_raster_sample_count = 1;
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_buffer(Some(seen), 0, 0).unwrap();
    (encoder.draw_primitives(PrimitiveType::Line, 0, MARKS as u32, 1, 0)).unwrap();
    (encoder.draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer
}

/// The guest RAM's bytes of a page the draws marked: each of the [`MARKS`] words 1, every other
/// byte untouched.
fn marked_page() -> Vec<u8> {
    let mut page = vec![UNTOUCHED; GUEST_PAGE];
    for word in page[..4 * MARKS].chunks_mut(4) {
        word.copy_from_slice(&1u32.to_le_bytes());
    }
    page
}

/// A command buffer whose draws' pipelines are still being made as its encoding ends (the pool
/// held), and whose draws write a no-copy buffer over guest pages, is dropped uncommitted; the
/// program lets go of the buffer and unmaps the pages at once (replaced in place by an
/// inaccessible mapping, so any CPU access of them faults), then lets the makings finish and
/// waits until they are done. The guest RAM's page keeps every byte, the validation layer reports
/// nothing, and a command buffer committed after it on the same queue draws with the pipelines
/// now made, through a no-copy buffer over the next page, which it marks: the same draws reach
/// the guest RAM when they run.
#[test]
fn a_dropped_uncommitted_command_buffer_touches_no_host_memory_once_dropped() {
    let watched = Watched::default();
    let scratch = Scratch::new("dropped-uncommitted");
    // Declared before the device, so dropped after it.
    let ram = GuestRam::new(3 * GUEST_PAGE);
    ram.fill(UNTOUCHED);
    // Everything of the device's goes, the device last, before the layer's messages are read.
    {
        let device = device(&watched, &scratch.0);
        let state = unrasterized_state(&device);
        let queue = device.new_command_queue().unwrap();
        // Nothing of the state's creation is left in the pool: what it holds after the encoding is
        // the draws'.
        let before = pipelines_settled(&device);
        let dropped_view = ram.map(GUEST_PAGE, GUEST_PAGE);
        let dropped_buffer = over(&device, &ram, &dropped_view);
        // Every making the pool starts waits from here until the pages are unmapped.
        let hold = held::hold();
        let command_buffer = encode_marks(&queue, &state, &dropped_buffer);
        let encoded = device.work_counts();
        assert_eq!(
            encoded.pipelines_pooled - before.pipelines_pooled,
            2,
            "the line's and the triangle's pipelines went to the pool"
        );
        assert!(
            encoded.pipeline_pool_busy > 0,
            "the draws' pipelines are still being made as the encoding ends"
        );
        drop(command_buffer);
        drop(dropped_buffer);
        dropped_view.unmap_in_place();
        drop(hold);
        pipelines_settled(&device);
        // The same queue, the pipelines made, over the next page.
        let next_view = ram.map(2 * GUEST_PAGE, GUEST_PAGE);
        let next_buffer = over(&device, &ram, &next_view);
        let next = encode_marks(&queue, &state, &next_buffer);
        next.commit().unwrap();
        next.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(next.status(), CommandBufferStatus::Completed);
        assert_eq!(next.error(), None);
        drop(next);
        drop(next_buffer);
        drop(next_view);
        assert!(
            ram.read(GUEST_PAGE..2 * GUEST_PAGE) == vec![UNTOUCHED; GUEST_PAGE],
            "the dropped command buffer's draws reached the guest page"
        );
        assert!(
            ram.read(2 * GUEST_PAGE..3 * GUEST_PAGE) == marked_page(),
            "the next command buffer's draws marked their guest page"
        );
        assert!(
            ram.read(0..GUEST_PAGE) == vec![UNTOUCHED; GUEST_PAGE],
            "a page no buffer was over"
        );
        eprintln!(
            "DROPPED-UNCOMMITTED: {} dma-buf import(s) of the guest pages",
            device.work_counts().dma_buf_imports - before.dma_buf_imports
        );
        drop(dropped_view);
    }
    watched.layer_reported_nothing(
        "a command buffer dropped uncommitted with its draws' pipelines being made",
    );
}
