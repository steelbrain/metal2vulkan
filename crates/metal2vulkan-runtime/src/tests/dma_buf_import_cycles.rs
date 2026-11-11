//! Cycles of each way the runtime imports dma-bufs of host pages and frees them, one test per way,
//! so that a consumer can read the kernel's account of its dma-bufs (Linux's
//! `/sys/kernel/debug/dma_buf/bufinfo`) before and after each, run alone by its name. Each runs
//! [`DEFAULT_CYCLES`] cycles, or as many as `DMA_BUF_CYCLES` names, on the device the cfg selects
//! made without `VK_EXT_external_memory_host` (`tests/no_pointer_import.rs`), so that its no-copy
//! buffers are imported as dma-bufs of their pages on every device that lists the dma-buf route.
//! The pages are the test's own: a sealed anonymous `memfd` mapped shared, whose source makes
//! `udmabuf`s of the pages each request asks for.
//!
//! - **one import**: a no-copy buffer bound in one dma-buf import of its pages, read by one GPU
//!   copy, waited for, released.
//! - **several imports**: a no-copy buffer bound sparsely over several dma-buf imports, one per
//!   granule of the device's sparse buffers (`dma_buf.rs`'s `chunks_buffer`), the same use. Where
//!   the device asks for no split (no `sparseBinding`, no dma-buf import for a sparse buffer, or
//!   a sparse buffer that takes fewer memory types than a plain one: `Device::sparse_granule`)
//!   each request's granule is its length and takes one chunk, one plain buffer's import: the
//!   line says so. Either way every chunk is imported, none left to a copy.
//! - **an image's import**: a 64x64 `RGBA8Unorm` texture that only reads, over a no-copy buffer
//!   made once for all the cycles, in place: its `DRM_FORMAT_MOD_LINEAR` image bound to a
//!   dedicated import of a dma-buf of its own pages; read by one GPU copy, waited for, released.
//!   Where the texture is a shadow (the device's check did not pass for such pages) the line says
//!   so.
//! - **freed once the device is idle**: as one import, its copy waited for, then the device
//!   waited idle (each queue the runtime submits to, `vkQueueWaitIdle` under the queue's lock, and
//!   its staging queue's last submission), then released: the test holds the buffer through the
//!   wait, so the import's `vkFreeMemory` follows it.
//! - **unused**: as one import, released with no GPU use.
//! - **freed while other work runs**: as one import, its copy waited for, then released while a
//!   command buffer of other work (copies between two private buffers) is scheduled on the same
//!   queue and not yet complete.
//! - **freed by its own use's completion**: as one import, released while the command buffer
//!   that copies it (after the other work) is scheduled and not yet complete: the command buffer
//!   holds it, and its completion frees it.
//! - **released while idle and imported again**: a no-copy buffer made once over one import, each
//!   cycle read by one GPU copy, waited for, then its import released while idle
//!   (`Device::release_idle_host_imports`, as the device's residency thread releases it once its
//!   idle window passed, `host_residency.rs`): its `VkBuffer` destroyed and its memory freed, the
//!   dma-buf's descriptor kept open, from which the next cycle's copy imports the same dma-buf
//!   again. The buffer goes after the last cycle. Where the device keeps the import (it kept no
//!   descriptor), the line says so.
//!
//! Each prints one line (`DMA-BUF-CYCLES`): the way, the cycles, and the imports the device made
//! and freed of buffers' pages (`WorkCounts::dma_buf_imports`, and `dma_buf_memories_live`'s
//! fall) and of images (`dma_buf_images`, `dma_buf_image_memories_live`), and the sparse buffers
//! made (`dma_buf_sparse_buffers`). It checks that the runtime freed every import it made, once
//! the last command buffer's completion let go of it, and that its way ran (the imports and
//! buffers it counts, and for the last two the command buffer still in flight at the release).
//!
//! The release order, in the runtime (the last way apart: there the runtime frees the memory and
//! keeps the dma-buf open, and imports it again later): a buffer's `VkBuffer` is destroyed first (a sparse one's
//! after a wait for its binding, where no submission waited for it; its bindings are not unbound
//! first, Vulkan ends them with the buffer), then its imports are freed (`vkFreeMemory`), each in
//! turn, as the last holder of its pages lets go of them (`dma_buf.rs`'s `DmaBufPages`); an
//! image's `VkImage` is destroyed, then its own import freed. All on the releasing thread, or on
//! the completion's thread where a command buffer held it last.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux.
//! Where `/dev/udmabuf` cannot be opened, or the device lists no dma-buf route, it says so and
//! passes without running.

use super::host_residency::{counting, host_page, udmabuf, udmabuf_openable, Scratch};
use super::merge_scratch::{blit, read};
use super::no_pointer_import::{device, lists_dma_buf_route};
use crate::{
    BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, DmaBufChunk,
    DmaBufRequest, DmaBufSource, HostMemory, Origin, ResourceOptions, Size, TextureDescriptor,
    WorkCounts,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const TEST: &str = "DMA-BUF-CYCLES";
/// The cycles each test runs where `DMA_BUF_CYCLES` names none.
const DEFAULT_CYCLES: usize = 200;
/// The variable that names the cycles.
const CYCLES: &str = "DMA_BUF_CYCLES";
/// The bytes of a cycle's no-copy buffer: 256 KiB, four 64 KiB granules of RADV's sparse buffers.
const LENGTH: u64 = 256 << 10;
/// The file's bytes: the buffer's, and room for a sparse request's padding to its granule.
const FILE: u64 = 4 << 20;
/// The bytes of each buffer of the other work, and how many copies it runs.
const WORK_BYTES: u64 = 32 << 20;
const WORK_COPIES: usize = 8;
const WAIT: Duration = Duration::from_secs(60);
const CHECK_WAIT: Duration = Duration::from_secs(120);

/// The cycles to run.
fn cycles() -> usize {
    std::env::var(CYCLES)
        .ok()
        .and_then(|cycles| cycles.parse().ok())
        .unwrap_or(DEFAULT_CYCLES)
}

/// The test's pages: a sealed anonymous `memfd` of [`FILE`] bytes mapped shared, its bytes
/// counting; unmapped as it drops, after every buffer over it went.
struct Pages {
    fd: Arc<OwnedFd>,
    at: NonNull<u8>,
}

impl Pages {
    fn new() -> Pages {
        // SAFETY: a new anonymous file, owned at once, then sized, sealed and mapped.
        unsafe {
            let fd = libc::memfd_create(
                c"m2v-dma-buf-cycles".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            );
            assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
            let fd = OwnedFd::from_raw_fd(fd);
            assert_eq!(libc::ftruncate(fd.as_raw_fd(), FILE as libc::off_t), 0);
            let seals = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
            assert_eq!(libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals), 0);
            let at = libc::mmap(
                std::ptr::null_mut(),
                FILE as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            );
            assert_ne!(at, libc::MAP_FAILED, "mmap");
            let bytes = counting(FILE as usize, 7);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), at.cast(), bytes.len());
            Pages {
                fd: Arc::new(fd),
                at: NonNull::new(at.cast()).unwrap(),
            }
        }
    }

    /// A source of `udmabuf`s of the file's pages behind each request: one chunk, or, where
    /// `per_granule` and the request's granule is less than its length, one chunk per granule,
    /// the last padded past the request to its granule with the file's next pages. Each
    /// request's granule and chunks are noted in `asked`.
    fn source(&self, per_granule: bool, asked: &Arc<Mutex<Vec<(u64, usize)>>>) -> DmaBufSource {
        let fd = Arc::clone(&self.fd);
        let base = self.at.as_ptr() as u64;
        let asked = Arc::clone(asked);
        Arc::new(move |request: DmaBufRequest| {
            let at = (request.start.as_ptr() as u64)
                .checked_sub(base)
                .ok_or_else(|| std::io::Error::other("pages before the file's mapping"))?;
            let (count, each) = match per_granule && request.granule < request.length {
                true => (request.length.div_ceil(request.granule), request.granule),
                false => (1, request.length),
            };
            if at + count * each > FILE {
                return Err(std::io::Error::other("pages past the file"));
            }
            let chunks = (0..count)
                .map(|index| {
                    Ok(DmaBufChunk {
                        fd: udmabuf(&fd, at + index * each, each)?,
                        length: each,
                    })
                })
                .collect::<std::io::Result<Vec<DmaBufChunk>>>()?;
            asked.lock().unwrap().push((request.granule, chunks.len()));
            Ok(chunks)
        })
    }

    /// A no-copy buffer of [`LENGTH`] bytes over the file's first pages, its source
    /// [`source`](Self::source)'s.
    fn buffer(
        &self,
        device: &Device,
        per_granule: bool,
        asked: &Arc<Mutex<Vec<(u64, usize)>>>,
    ) -> Buffer {
        // SAFETY: the mapping outlives the buffer (each test drops its buffers first), and the
        // source makes dma-bufs of its pages.
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                HostMemory::with_dma_buf(self.at, self.source(per_granule, asked)),
                LENGTH,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap()
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: the test's mapping, which no buffer uses any more.
        unsafe { libc::munmap(self.at.as_ptr().cast(), FILE as usize) };
    }
}

/// Commit `command_buffer` and wait for it.
fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None, "{TEST}: the command buffer");
}

/// Encode one copy of `buffer`'s bytes into `out` in `command_buffer`.
fn copy_out(command_buffer: &CommandBuffer, buffer: &Buffer, out: &Buffer) {
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    (encoder.copy_from_buffer_to_buffer(buffer, 0, out, 0, LENGTH)).unwrap();
    encoder.end_encoding().unwrap();
}

/// Other work: two private buffers, copied into each other [`WORK_COPIES`] times by each command
/// buffer [`OtherWork::encode`] encodes into.
struct OtherWork {
    a: Buffer,
    b: Buffer,
}

impl OtherWork {
    fn new(device: &Device) -> OtherWork {
        let make = || {
            device
                .new_buffer(WORK_BYTES, ResourceOptions::private())
                .unwrap()
        };
        OtherWork {
            a: make(),
            b: make(),
        }
    }

    fn encode(&self, command_buffer: &CommandBuffer) {
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        for copy in 0..WORK_COPIES {
            let (from, to) = match copy % 2 {
                0 => (&self.a, &self.b),
                _ => (&self.b, &self.a),
            };
            (encoder.copy_from_buffer_to_buffer(from, 0, to, 0, WORK_BYTES)).unwrap();
        }
        encoder.end_encoding().unwrap();
    }
}

/// The device's counts of dma-buf imports: buffers' chunks made and live, images' made and live,
/// and sparse buffers made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Imports {
    made: u64,
    live: u64,
    images: u64,
    images_live: u64,
    sparse: u64,
}

impl Imports {
    fn of(counts: &WorkCounts) -> Imports {
        Imports {
            made: counts.dma_buf_imports,
            live: counts.dma_buf_memories_live,
            images: counts.dma_buf_images,
            images_live: counts.dma_buf_image_memories_live,
            sparse: counts.dma_buf_sparse_buffers,
        }
    }
}

/// The counts once the imports live at `before` are all that live (the frees a completion makes
/// land on its own thread), or after [`WAIT`].
fn settled(device: &Device, before: Imports) -> Imports {
    let start = Instant::now();
    loop {
        let now = Imports::of(&device.work_counts());
        let settled = now.live <= before.live && now.images_live <= before.images_live;
        if settled || start.elapsed() > WAIT {
            return now;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Print the way's line from the counts `before` and `after` its cycles, with `what` it adds,
/// and check that every import it made was freed.
fn report(
    device: &Device,
    way: &str,
    cycles: usize,
    (before, after): (Imports, Imports),
    what: &str,
) {
    // Signed: an import made before the cycles (the device's check's) may go during them.
    let delta = |after: u64, before: u64| after as i64 - before as i64;
    let made = delta(after.made, before.made);
    let images = delta(after.images, before.images);
    let freed = made - delta(after.live, before.live);
    let images_freed = images - delta(after.images_live, before.images_live);
    println!(
        "{TEST} on {}: {way}: {cycles} cycles; buffers' dma-buf imports made {made}, freed \
         {freed}; images' dma-buf imports made {images}, freed {images_freed}; sparse buffers \
         made {}; {what}",
        device.name(),
        after.sparse - before.sparse,
    );
    assert_eq!(
        (freed, images_freed),
        (made, images),
        "{TEST}: {way}: every import freed"
    );
}

/// The device the cfg selects, without `VK_EXT_external_memory_host`, where `/dev/udmabuf` opens
/// and the device lists the dma-buf route; else `None`, said.
fn cycling_device(scratch: &Scratch, way: &str) -> Option<Device> {
    if !udmabuf_openable(TEST) {
        return None;
    }
    let device = device(&scratch.0, None);
    if !lists_dma_buf_route(&device) {
        println!(
            "{TEST} on {}: {way}: no dma-buf route: not run",
            device.name()
        );
        return None;
    }
    Some(device)
}

/// Run `cycles` cycles of a buffer over one import of its pages ([`Pages::buffer`]), each `used`
/// as it says (given the queue, the buffer and the copy's buffer), then released; the way's line.
fn buffer_cycles(way: &str, used: impl Fn(&CommandQueue, Buffer, &Buffer) -> bool) {
    let scratch = Scratch::new();
    let Some(device) = cycling_device(&scratch, way) else {
        return;
    };
    let queue = device.new_command_queue().unwrap();
    let out = device
        .new_buffer(LENGTH, ResourceOptions::shared())
        .unwrap();
    let pages = Pages::new();
    let asked = Arc::default();
    let cycles = cycles();
    let before = Imports::of(&device.work_counts());
    let mut in_flight = 0;
    for _ in 0..cycles {
        let buffer = pages.buffer(&device, false, &asked);
        in_flight += usize::from(used(&queue, buffer, &out));
    }
    let after = settled(&device, before);
    let asked = asked.lock().unwrap();
    report(
        &device,
        way,
        cycles,
        (before, after),
        &format!(
            "the source asked {} times; {in_flight} released with a command buffer in flight",
            asked.len()
        ),
    );
    assert_eq!(
        after.made - before.made,
        cycles as u64,
        "{TEST}: {way}: one import a cycle"
    );
    drop(out);
    drop(queue);
    drop(device);
    drop(pages);
}

/// A buffer over one dma-buf import, read by one GPU copy, waited for, released.
#[test]
fn dma_buf_cycles_one_import() {
    buffer_cycles("one import", |queue, buffer, out| {
        let command_buffer = queue.command_buffer().unwrap();
        copy_out(&command_buffer, &buffer, out);
        run(&command_buffer);
        assert!(
            read(out, LENGTH) == counting(FILE as usize, 7)[..LENGTH as usize],
            "{TEST}: the copy of the pages"
        );
        drop(buffer);
        false
    });
}

/// Wait until the device is idle: each queue the runtime submits to (`vkQueueWaitIdle` under the
/// queue's lock, as the device's teardown waits, `DeviceShared::queues_idle`), and its staging
/// queue's last submission.
pub(super) fn device_idle(device: &Device) {
    let shared = device.shared();
    let raw = shared.vk();
    for slot in shared.queues.iter() {
        // SAFETY: the device's queue, used under its lock.
        (slot.with(|queue| unsafe { raw.queue_wait_idle(queue) })).unwrap();
    }
    if let Some((_, value)) = device.staging_wait() {
        device.wait_staging(value).unwrap();
    }
}

/// A buffer over one dma-buf import, read by one GPU copy, waited for, then the device waited
/// idle, then released: its import's `vkFreeMemory` after the wait, as the test holds the buffer
/// through it.
#[test]
fn dma_buf_cycles_freed_once_the_device_is_idle() {
    buffer_cycles("freed once the device is idle", |queue, buffer, out| {
        let command_buffer = queue.command_buffer().unwrap();
        copy_out(&command_buffer, &buffer, out);
        run(&command_buffer);
        drop(command_buffer);
        device_idle(queue.device());
        assert!(
            read(out, LENGTH) == counting(FILE as usize, 7)[..LENGTH as usize],
            "{TEST}: the copy of the pages"
        );
        drop(buffer);
        false
    });
}

/// A buffer over one dma-buf import, released with no GPU use.
#[test]
fn dma_buf_cycles_unused() {
    buffer_cycles("unused", |_, buffer, _| {
        drop(buffer);
        false
    });
}

/// A buffer over one dma-buf import, its copy waited for, then released while a command buffer
/// of other work is scheduled on the same queue and not yet complete.
#[test]
fn dma_buf_cycles_freed_while_other_work_runs() {
    let work: Mutex<Option<OtherWork>> = Mutex::default();
    buffer_cycles("freed while other work runs", |queue, buffer, out| {
        let mut work = work.lock().unwrap();
        let work = work.get_or_insert_with(|| OtherWork::new(queue.device()));
        let command_buffer = queue.command_buffer().unwrap();
        copy_out(&command_buffer, &buffer, out);
        run(&command_buffer);
        let other = queue.command_buffer().unwrap();
        work.encode(&other);
        other.commit().unwrap();
        other.wait_until_scheduled().unwrap();
        let in_flight = other.status() != CommandBufferStatus::Completed;
        drop(buffer);
        other.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(other.error(), None, "{TEST}: the other work");
        in_flight
    });
}

/// A buffer over one dma-buf import, released while the command buffer that copies it, after
/// other work, is scheduled and not yet complete: its completion frees it.
#[test]
fn dma_buf_cycles_freed_by_its_own_use_s_completion() {
    let work: Mutex<Option<OtherWork>> = Mutex::default();
    buffer_cycles("freed by its own use's completion", |queue, buffer, out| {
        let mut work = work.lock().unwrap();
        let work = work.get_or_insert_with(|| OtherWork::new(queue.device()));
        let command_buffer = queue.command_buffer().unwrap();
        work.encode(&command_buffer);
        copy_out(&command_buffer, &buffer, out);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_scheduled().unwrap();
        let in_flight = command_buffer.status() != CommandBufferStatus::Completed;
        drop(buffer);
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.error(), None, "{TEST}: the copy");
        in_flight
    });
}

/// A buffer bound sparsely over several dma-buf imports, one per granule, read by one GPU copy,
/// waited for, released.
#[test]
fn dma_buf_cycles_several_imports() {
    const WAY: &str = "several imports";
    let scratch = Scratch::new();
    let Some(device) = cycling_device(&scratch, WAY) else {
        return;
    };
    let queue = device.new_command_queue().unwrap();
    let out = device
        .new_buffer(LENGTH, ResourceOptions::shared())
        .unwrap();
    let pages = Pages::new();
    let asked = Arc::default();
    let cycles = cycles();
    let before = Imports::of(&device.work_counts());
    for _ in 0..cycles {
        let buffer = pages.buffer(&device, true, &asked);
        let command_buffer = queue.command_buffer().unwrap();
        copy_out(&command_buffer, &buffer, &out);
        run(&command_buffer);
        assert!(
            read(&out, LENGTH) == counting(FILE as usize, 7)[..LENGTH as usize],
            "{TEST}: {WAY}: the copy of the pages"
        );
        drop(buffer);
    }
    let after = settled(&device, before);
    let asked = asked.lock().unwrap();
    let (granule, chunks) = asked.first().copied().unwrap_or_default();
    let sparse = after.sparse - before.sparse;
    let what = match granule {
        LENGTH => format!(
            "the device asks for no split: each request's granule is its length ({granule} \
             bytes), one chunk"
        ),
        _ => format!("each request's granule {granule} bytes: {chunks} chunks a buffer"),
    };
    report(&device, WAY, cycles, (before, after), &what);
    assert!(
        asked.iter().all(|&ask| ask == (granule, chunks)),
        "{TEST}: {WAY}: every request alike"
    );
    assert_eq!(
        sparse,
        match chunks {
            1 => 0,
            _ => cycles as u64,
        },
        "{TEST}: {WAY}: a sparse buffer a cycle where it split, else none"
    );
    assert_eq!(
        after.made - before.made,
        (cycles * chunks) as u64,
        "{TEST}: {WAY}: the chunks imported"
    );
    drop(out);
    drop(queue);
    drop(device);
    drop(pages);
}

/// A texture in place over a no-copy buffer made once, its image bound to a dedicated import of
/// a dma-buf of its own pages, read by one GPU copy, waited for, released.
#[test]
fn dma_buf_cycles_image_import() {
    const WAY: &str = "an image's import";
    const SIDE: u64 = 64;
    const ROW: u64 = 256;
    let scratch = Scratch::new();
    let Some(device) = cycling_device(&scratch, WAY) else {
        return;
    };
    assert!(
        device.settle_in_place_checks(CHECK_WAIT),
        "{TEST}: the check"
    );
    let queue = device.new_command_queue().unwrap();
    let out = (device.new_buffer(SIDE * SIDE * 4, ResourceOptions::shared())).unwrap();
    let pages = Pages::new();
    let asked = Arc::default();
    let buffer = pages.buffer(&device, false, &asked);
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    let placement = buffer.texture_placement(&descriptor, 0, ROW).unwrap();
    let cycles = cycles();
    let before = Imports::of(&device.work_counts());
    let mut in_place = 0;
    let page = host_page() as u64;
    let bytes = counting(FILE as usize, 7);
    let texels: Vec<u8> = (0..SIDE)
        .flat_map(|row| &bytes[(row * ROW) as usize..][..(SIDE * 4) as usize])
        .copied()
        .collect();
    for _ in 0..cycles {
        let texture = buffer.new_texture(&descriptor, 0, ROW).unwrap();
        // SAFETY: the handle is only compared, while the texture lives.
        in_place += usize::from(unsafe { texture.vk_image() }.is_some());
        blit(&queue, |encoder| {
            (encoder.copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(SIDE, SIDE, 1),
                &out,
                0,
                SIDE * 4,
                SIDE * SIDE * 4,
                BlitOption::None,
            ))
            .unwrap()
        });
        assert!(
            read(&out, SIDE * SIDE * 4) == texels,
            "{TEST}: {WAY}: the copy of the texels"
        );
        drop(texture);
    }
    let after = settled(&device, before);
    let what = format!(
        "{in_place} of the textures in place ({placement:?}), over one buffer of {LENGTH} bytes \
         whose import is made before the cycles; {page}-byte pages"
    );
    report(&device, WAY, cycles, (before, after), &what);
    assert_eq!(
        (after.images - before.images, after.made - before.made),
        (in_place as u64, 0),
        "{TEST}: {WAY}: an image's import a texture in place, no buffer's"
    );
    drop(buffer);
    drop(out);
    drop(queue);
    drop(device);
    drop(pages);
}

/// A buffer over one dma-buf import made once, each cycle read by one GPU copy, waited for, then
/// its import released while idle, the dma-buf's descriptor kept, which the next cycle's copy
/// imports again.
#[test]
fn dma_buf_cycles_released_while_idle_and_imported_again() {
    const WAY: &str = "released while idle and imported again";
    const SETTLE: Duration = Duration::from_secs(20);
    let scratch = Scratch::new();
    let Some(device) = cycling_device(&scratch, WAY) else {
        return;
    };
    let queue = device.new_command_queue().unwrap();
    let out = device
        .new_buffer(LENGTH, ResourceOptions::shared())
        .unwrap();
    let pages = Pages::new();
    let asked = Arc::default();
    let cycles = cycles();
    let before = Imports::of(&device.work_counts());
    let counts = device.work_counts();
    let (released_before, again_before) =
        (counts.host_imports_released, counts.host_imports_reimported);
    let buffer = pages.buffer(&device, false, &asked);
    let mut kept = false;
    for _ in 0..cycles {
        let command_buffer = queue.command_buffer().unwrap();
        copy_out(&command_buffer, &buffer, &out);
        run(&command_buffer);
        assert!(
            read(&out, LENGTH) == counting(FILE as usize, 7)[..LENGTH as usize],
            "{TEST}: {WAY}: the copy of the pages"
        );
        // The command buffer lets go of the import at its completion, on the device's handler
        // thread.
        let start = Instant::now();
        while device.release_idle_host_imports() == 0 {
            if start.elapsed() > SETTLE {
                kept = true;
                break;
            }
            std::thread::yield_now();
        }
        if kept {
            break;
        }
    }
    drop(buffer);
    let after = settled(&device, before);
    let counts = device.work_counts();
    let what = match kept {
        true => "the device keeps the import (no descriptor kept, or not releasable): not run"
            .to_owned(),
        false => format!(
            "imports released while idle {}, imported again {}",
            counts.host_imports_released - released_before,
            counts.host_imports_reimported - again_before,
        ),
    };
    report(&device, WAY, cycles, (before, after), &what);
    if !kept {
        assert_eq!(
            counts.host_imports_released - released_before,
            cycles as u64,
            "{TEST}: {WAY}: a release a cycle"
        );
        assert_eq!(
            counts.host_imports_reimported - again_before,
            cycles as u64 - 1,
            "{TEST}: {WAY}: an import again a cycle but the first"
        );
    }
    drop(out);
    drop(queue);
    drop(device);
    drop(pages);
}
