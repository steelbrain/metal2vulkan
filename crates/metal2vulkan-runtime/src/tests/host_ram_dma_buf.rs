//! A guest RAM with a source of dma-bufs of its pages (`HostRam::with_dma_buf`, `host_ram.rs`) on
//! the device the cfg selects: its chunks are imported as its device's checks choose.
//!
//! The RAM is a `memfd` sealed against growing and shrinking, mapped shared, as QEMU's
//! `memory-backend-memfd,share=on` makes it, and its source makes dma-bufs of its pages through
//! `/dev/udmabuf`, as a consumer's does. The objects' pointers lie in a second mapping of the same
//! file (the consumer's view of the guest's pages), which their resolver maps to the RAM's
//! addresses, so a request the source is asked at a view's address would be refused.
//!
//! - **The route.** Where the device's held pointer imports tax every submission (the import tax,
//!   `submit_check.rs`: RADV), the chunk an object lies in is imported as one dma-buf of it, the
//!   source asked once for the chunk's whole span at the RAM's addresses, with a granule of its
//!   length (`WorkCounts::host_ram_dma_buf_imports`); elsewhere the pointer is imported where the
//!   driver takes it, the source never asked, and the dma-buf only where it refuses it. Either
//!   way the buffer is bound in the chunk (`chunk_bound_buffers_made`), reports `InHostRam`, and
//!   a second buffer over the chunk imports nothing more.
//! - **The bytes.** The GPU reads the CPU's bytes through the buffer, and its fill reaches the
//!   file's pages, leaving the page past the buffer as it was.
//! - **A texture over it** reads the buffer's texels; over a dma-buf chunk its image is made over
//!   a dma-buf of its own pages, which the source is asked for at the RAM's addresses (never the
//!   view's: `Placement::dma_buf_address`), or it is a shadow, which reads them too.
//! - **The use.** Once the use and the objects go, the RAM's keep-alive goes.
//! - **Few listed imports.** Where each listed import costs every submission (the import tax held,
//!   dma-bufs imported: RADV), a RAM whose source's dma-bufs may be as long as it is one chunk
//!   ([`HostRam::with_largest_dma_buf`]), the source asked once for all of it; no chunk is imported
//!   ahead: the first object over a chunk lies in its own run's import and asks the device's thread
//!   for the chunk, which imports it; those after it lie in the chunk. The runs' imports, which
//!   only blits read, are released once idle, the next blit binding the buffer in the chunk; the
//!   chunk is kept while an object over it lives. On a driver whose barriers do not order blits
//!   through two imports of the same pages (`RuntimeDriverDefect::AliasedImportBlitsUnordered`,
//!   ANV's among the listed), no run is imported alone: each object lies in its chunk's import,
//!   which the first one over it makes. Where the RAM does not fit in the memory the device's
//!   imports count against (made so by the test), the device's thread releases it once none did for
//!   its idle window (`WorkCounts::host_ram_chunks_released`), and a copy over it then imports its
//!   run alone and asks for the chunk again (or, on such a driver, imports the chunk); where the
//!   RAM fits, the chunks stay imported past their objects and the window, and the next object lies
//!   in its chunk with no import. Elsewhere the RAM keeps its 64 MiB chunks.
//! - **A guest's tile copied by runs.** 1,800 scattered pages gathered and scattered back, over
//!   chunks not imported and imported, byte for byte. Where each listed import costs every
//!   submission and the device's barriers order blits through two imports of the same pages
//!   (RADV), a copy over a chunk not imported, of a RAM with a source of run lists
//!   (`HostRam::with_dma_buf_of_runs`, udmabuf's `UDMABUF_CREATE_LIST` here), takes lists of its
//!   runs, at most 1,024 a list, takes no chunk and asks for none, so over a RAM of which the
//!   guest wrote the tile's pages alone it makes no other page of the file resident (`mincore`);
//!   of a RAM without one, it imports the chunk. Elsewhere the source of run lists is never asked.
//!
//! Each run prints the device's verdict and the route its chunk took: the evidence of which path
//! ran. Where `/dev/udmabuf` cannot be opened, or the device imports no host memory, the test says
//! so and passes without running.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux.

use super::host_ram_device::runs_alone;
use super::host_residency::{device, udmabuf, udmabuf_list, Scratch};
use crate::submit_check::{Check, Verdict};
use crate::{
    Buffer, CommandBufferStatus, Device, DmaBufChunk, DmaBufRequest, DmaBufRunsSource,
    DmaBufSource, Error, HostBacking, HostMemory, HostRam, KeepAlive, Origin, ResourceOptions,
    RunResolver, Size, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
/// Past a guest RAM chunk's first idle window (2 s) and a tick of the device's thread.
const IDLE_WAIT: Duration = Duration::from_secs(3);
const MIB: u64 = 1 << 20;
/// The RAM's bytes: two whole chunks.
const RAM_BYTES: u64 = 128 * MIB;
/// Where the buffer starts in the RAM, and its bytes: inside the first chunk.
const AT: u64 = 4 * MIB;
const LENGTH: u64 = MIB;
/// The texture over the buffer: 64 by 64 texels of four bytes, rows 256 bytes apart.
const SIDE: u64 = 64;
const ROW: u64 = SIDE * 4;

/// The file's two shared mappings, unmapped by the keep-alive: the RAM's and the view's.
struct Mappings {
    ram: u64,
    view: u64,
    dropped: Arc<AtomicBool>,
}

impl Drop for Mappings {
    fn drop(&mut self) {
        for start in [self.ram, self.view] {
            // SAFETY: the test's mappings, which no device imports any more: the keep-alive is
            // dropped after the last chunk over them is freed.
            unsafe { libc::munmap(start as *mut libc::c_void, RAM_BYTES as usize) };
        }
        self.dropped.store(true, Ordering::SeqCst);
    }
}

/// What the source was asked: each request's first byte, length and granule, and those it
/// refused as lying outside the RAM; and what the source of run lists was asked, each list's
/// requests.
#[derive(Default)]
struct Asked {
    requests: Vec<(u64, u64, u64)>,
    outside: Vec<(u64, u64)>,
    lists: Vec<Vec<(u64, u64, u64)>>,
}

/// The RAM over a new sealed `memfd`, with a source of dma-bufs of it where `sourced`; the RAM's
/// first byte, the view's, what the source was asked, and whether the keep-alive was dropped.
fn ram(sourced: bool) -> (Arc<HostRam>, u64, u64, Arc<Mutex<Asked>>, Arc<AtomicBool>) {
    listed_ram(sourced, None)
}

/// [`ram`], with a source of run lists too where `lists` gives their most runs
/// (`HostRam::with_dma_buf_of_runs`): udmabuf's list of the runs asked for.
fn listed_ram(
    sourced: bool,
    lists: Option<usize>,
) -> (Arc<HostRam>, u64, u64, Arc<Mutex<Asked>>, Arc<AtomicBool>) {
    // SAFETY: a new anonymous file, owned at once, then sized and sealed.
    let fd = unsafe {
        let fd = libc::memfd_create(
            c"m2v-host-ram-dma-buf".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        );
        assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
        OwnedFd::from_raw_fd(fd)
    };
    // SAFETY: the descriptor is the test's.
    unsafe {
        assert_eq!(libc::ftruncate(fd.as_raw_fd(), RAM_BYTES as libc::off_t), 0);
        let seals = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
        assert_eq!(libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals), 0);
    }
    let map = || {
        // SAFETY: a new shared mapping of the whole file.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                RAM_BYTES as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        assert_ne!(at, libc::MAP_FAILED, "mmap of the file");
        at as u64
    };
    let (start, view) = (map(), map());
    let dropped = Arc::new(AtomicBool::new(false));
    let keep: KeepAlive = Box::new(Mappings {
        ram: start,
        view,
        dropped: Arc::clone(&dropped),
    });
    let asked: Arc<Mutex<Asked>> = Arc::default();
    let memfd = Arc::new(fd);
    let source: DmaBufSource = {
        let asked = Arc::clone(&asked);
        let memfd = Arc::clone(&memfd);
        Arc::new(move |request: DmaBufRequest| {
            let first = request.start.as_ptr() as u64;
            let mut asked = asked.lock().unwrap();
            asked
                .requests
                .push((first, request.length, request.granule));
            let end = first.checked_add(request.length);
            if first < start || end.is_none_or(|end| end > start + RAM_BYTES) {
                asked.outside.push((first, request.length));
                return Err(std::io::Error::other("pages outside the RAM"));
            }
            let fd = udmabuf(&memfd, first - start, request.length)?;
            Ok(vec![DmaBufChunk {
                fd,
                length: request.length,
            }])
        })
    };
    let listing: DmaBufRunsSource = {
        let asked = Arc::clone(&asked);
        Arc::new(move |requests: &[DmaBufRequest]| {
            let list: Vec<(u64, u64, u64)> = (requests.iter())
                .map(|request| {
                    let first = request.start.as_ptr() as u64;
                    (first, request.length, request.granule)
                })
                .collect();
            asked.lock().unwrap().lists.push(list.clone());
            let inside = (list.iter()).all(|&(first, length, _)| {
                first >= start && first.saturating_add(length) <= start + RAM_BYTES
            });
            if !inside {
                return Err(std::io::Error::other("pages outside the RAM"));
            }
            let runs: Vec<(u64, u64)> = (list.iter())
                .map(|&(first, length, _)| (first - start, length))
                .collect();
            let fd = udmabuf_list(&memfd, &runs)?;
            let length = runs.iter().map(|&(_, length)| length).sum();
            Ok(DmaBufChunk { fd, length })
        })
    };
    let at = NonNull::new(start as *mut u8).unwrap();
    // SAFETY: the mapping is readable and writable until the keep-alive unmaps it.
    let ram = unsafe { HostRam::new(&[(at, RAM_BYTES)], Some(keep)) }.unwrap();
    let ram = match sourced {
        true => ram.with_dma_buf(source),
        false => ram,
    };
    let ram = match lists {
        Some(most) => ram.with_dma_buf_of_runs(listing, most),
        None => ram,
    };
    (Arc::new(ram), start, view, asked, dropped)
}

/// The byte the pattern holds `offset` bytes into the file.
fn pattern(offset: u64) -> u8 {
    (offset % 251) as u8
}

/// The RAM's bytes of `range` of its offsets, read through `start`'s mapping while no command
/// buffer runs.
fn bytes(start: u64, range: std::ops::Range<u64>) -> Vec<u8> {
    // SAFETY: the test's mapping, which no command buffer writes now.
    unsafe {
        std::slice::from_raw_parts(
            (start + range.start) as *const u8,
            (range.end - range.start) as usize,
        )
    }
    .to_vec()
}

/// A no-copy buffer of `length` bytes at `offset` in the view, resolved to the same offset in
/// the RAM.
fn buffer(device: &Device, view: u64, start: u64, offset: u64, length: u64) -> Buffer {
    try_buffer(device, view, start, offset, length).unwrap()
}

/// [`buffer`], or why it is refused.
fn try_buffer(
    device: &Device,
    view: u64,
    start: u64,
    offset: u64,
    length: u64,
) -> Result<Buffer, Error> {
    let runs: RunResolver = Arc::new(move |at, length| {
        let ram = at - view + start;
        Some(std::iter::once(ram..ram + length).collect())
    });
    // SAFETY: the view's bytes, mapped until the RAM's keep-alive goes, after every device let go.
    let memory = unsafe { HostMemory::new(NonNull::new((view + offset) as *mut u8).unwrap()) }
        .with_runs(runs);
    // SAFETY: as above.
    unsafe {
        device.new_buffer_with_bytes_no_copy(memory, length, ResourceOptions::default(), None)
    }
}

/// Write the pattern over the view's bytes before `end`, while no command buffer uses them.
fn fill_pattern(view: u64, end: u64) {
    // SAFETY: the view's bytes, which no command buffer uses now.
    unsafe {
        for offset in 0..end {
            *((view + offset) as *mut u8) = pattern(offset);
        }
    }
}

/// A device of the cfg's kind, its checks answered, whose uses of a RAM import each chunk at
/// the first object over it (`Device::hold_host_ram_imports`), as one the device's thread has
/// not reached is; `None` where `/dev/udmabuf` cannot be opened or it imports no host memory.
fn held_device(scratch: &Scratch) -> Option<Device> {
    if !udmabuf_openable() {
        return None;
    }
    let device = device(&scratch.0);
    if device.shared().host_import.is_err() {
        eprintln!("HOST-RAM-DMA-BUF: the device imports no host memory here; nothing to test");
        return None;
    }
    device.hold_host_ram_imports();
    Some(device)
}

/// Whether the device imports dma-bufs.
fn imports_dma_bufs(device: &Device) -> bool {
    (device.shared().host_import.as_ref()).is_ok_and(|importer| importer.imports_dma_bufs())
}

/// Commit one command buffer whose blit encoder `encode` fills, and wait for it.
fn blit(device: &Device, encode: impl FnOnce(&mut crate::BlitCommandEncoder)) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encode(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The contents of a new shared buffer of `length` bytes that `encode` writes.
fn read_back(
    device: &Device,
    length: u64,
    encode: impl FnOnce(&mut crate::BlitCommandEncoder, &Buffer),
) -> Vec<u8> {
    let own = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    blit(device, |encoder| encode(encoder, &own));
    let contents = own.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) }.to_vec()
}

fn udmabuf_openable() -> bool {
    let opened = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/udmabuf");
    if let Err(error) = &opened {
        eprintln!("HOST-RAM-DMA-BUF: /dev/udmabuf cannot be opened ({error}): not run");
    }
    opened.is_ok()
}

#[test]
fn a_ram_with_a_source_takes_the_route_its_device_s_checks_choose() {
    let scratch = Scratch::new();
    let Some(device) = held_device(&scratch) else {
        return;
    };
    let taxed = device.pointer_imports_tax_submissions();
    let imports_dma_bufs = imports_dma_bufs(&device);
    let (ram, start, view, asked, dropped) = ram(true);
    fill_pattern(view, AT + 4 * LENGTH);
    let used = device.use_host_ram(&ram).unwrap();
    let before = device.work_counts();
    let first = buffer(&device, view, start, AT, LENGTH);
    let made = device.work_counts();
    let dma_bufs = made.host_ram_dma_buf_imports - before.host_ram_dma_buf_imports;
    let requests = asked.lock().unwrap().requests.clone();
    println!(
        "HOST-RAM-DMA-BUF {}: import tax {}; the chunk's route: {} dma-buf import(s), the source \
         asked {requests:?}",
        device.name(),
        match taxed {
            true => "held",
            false => "not held",
        },
        dma_bufs,
    );
    assert_eq!(
        made.host_ram_chunks_imported - before.host_ram_chunks_imported,
        1
    );
    assert_eq!(
        made.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
        1
    );
    assert_eq!(first.host_backing(), Some(HostBacking::InHostRam));
    let chunk = (start, 64 * MIB, 64 * MIB);
    match (taxed && imports_dma_bufs, dma_bufs) {
        // Taxed: the dma-buf first, of the whole chunk at the RAM's addresses.
        (true, _) => {
            assert_eq!(dma_bufs, 1, "a taxing device imported the chunk's pointer");
            assert_eq!(requests, [chunk]);
        }
        // Untaxed, or no dma-buf import: the pointer, the source unasked; or the dma-buf where the
        // pointer was refused.
        (false, 0) => assert!(requests.is_empty(), "{requests:?}"),
        (false, _) => assert_eq!(requests, [chunk]),
    }
    // The GPU reads the CPU's bytes, and its fill reaches the file's pages, the page past the
    // buffer as it was.
    let read = read_back(&device, LENGTH, |encoder, own| {
        encoder
            .copy_from_buffer_to_buffer(&first, 0, own, 0, LENGTH)
            .unwrap();
    });
    assert!(
        read.iter()
            .enumerate()
            .all(|(i, &b)| b == pattern(AT + i as u64)),
        "the GPU's read"
    );
    blit(&device, |encoder| {
        encoder.fill_buffer(&first, 0..LENGTH, 0x5a).unwrap()
    });
    assert!(
        bytes(start, AT..AT + LENGTH).iter().all(|&b| b == 0x5a),
        "the GPU's fill"
    );
    let past = bytes(start, AT + LENGTH..AT + LENGTH + MIB);
    assert!(
        past.iter()
            .enumerate()
            .all(|(i, &b)| b == pattern(AT + LENGTH + i as u64)),
        "past it"
    );
    // A second buffer over the same chunk imports nothing more.
    let second = buffer(&device, view, start, AT + 2 * LENGTH, LENGTH);
    let again = device.work_counts();
    assert_eq!(
        again.host_ram_chunks_imported,
        made.host_ram_chunks_imported
    );
    assert_eq!(
        again.host_ram_dma_buf_imports,
        made.host_ram_dma_buf_imports
    );
    assert_eq!(
        again.chunk_bound_buffers_made - made.chunk_bound_buffers_made,
        1
    );
    // A texture over the second buffer reads its texels: over a dma-buf chunk an image over a
    // dma-buf of its own pages, asked at the RAM's addresses, or a shadow.
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ;
    descriptor.resource_options = ResourceOptions::default();
    let texture = second.new_texture(&descriptor, 0, ROW).unwrap();
    let texels = read_back(&device, ROW * SIDE, |encoder, own| {
        let origin = Origin { x: 0, y: 0, z: 0 };
        let size = Size::new(SIDE, SIDE, 1);
        encoder
            .copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                origin,
                size,
                own,
                0,
                ROW,
                ROW * SIDE,
                crate::BlitOption::None,
            )
            .unwrap();
    });
    let texture_at = AT + 2 * LENGTH;
    assert_eq!(
        texels,
        bytes(start, texture_at..texture_at + ROW * SIDE),
        "the texture's texels"
    );
    let textured = device.work_counts();
    let outside = asked.lock().unwrap().outside.clone();
    println!(
        "HOST-RAM-DMA-BUF {}: the texture over it: {} dma-buf image(s); requests outside the RAM \
         {outside:?}",
        device.name(),
        textured.dma_buf_images - again.dma_buf_images,
    );
    assert!(
        outside.is_empty(),
        "the source was asked at the view's addresses: {outside:?}"
    );
    // Once the use and the objects go, the keep-alive goes.
    drop((texture, first, second, used, ram));
    let began = Instant::now();
    while !dropped.load(Ordering::SeqCst) {
        assert!(
            began.elapsed() < WAIT,
            "the RAM's keep-alive was never dropped"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Where the pointer goes first (the import tax not held, forced so on the device), a chunk whose
/// pointer the device refuses is imported as a dma-buf of the RAM's source: the refusal counted
/// (`WorkCounts::host_ram_pointer_imports_failed`), then the dma-buf (`host_ram_dma_buf_imports`).
/// A device that takes the pointer never asks the source.
#[test]
fn where_the_pointer_goes_first_a_refused_one_takes_the_dma_buf() {
    let scratch = Scratch::new();
    let Some(device) = held_device(&scratch) else {
        return;
    };
    (device.shared().submit_checks).force(Check::ImportTax, Verdict::Fails);
    let (ram, start, view, asked, _dropped) = ram(true);
    fill_pattern(view, AT + LENGTH);
    let used = device.use_host_ram(&ram).unwrap();
    let before = device.work_counts();
    let buffer = buffer(&device, view, start, AT, LENGTH);
    let after = device.work_counts();
    let failed = after.host_ram_pointer_imports_failed - before.host_ram_pointer_imports_failed;
    let dma_bufs = after.host_ram_dma_buf_imports - before.host_ram_dma_buf_imports;
    let requests = asked.lock().unwrap().requests.clone();
    println!(
        "HOST-RAM-DMA-BUF {}: the pointer first: {failed} pointer import(s) failed, then {dma_bufs} \
         dma-buf import(s), the source asked {requests:?}",
        device.name(),
    );
    assert_eq!(buffer.host_backing(), Some(HostBacking::InHostRam));
    match failed {
        0 => {
            assert_eq!(dma_bufs, 0);
            assert!(requests.is_empty(), "{requests:?}");
        }
        _ => {
            assert_eq!(failed, 1);
            assert!(
                imports_dma_bufs(&device),
                "a refused pointer with no dma-buf import"
            );
            assert_eq!(dma_bufs, 1);
            assert_eq!(requests, [(start, 64 * MIB, 64 * MIB)]);
        }
    }
    let read = read_back(&device, LENGTH, |encoder, own| {
        (encoder.copy_from_buffer_to_buffer(&buffer, 0, own, 0, LENGTH)).unwrap();
    });
    let expected: Vec<u8> = (AT..AT + LENGTH).map(pattern).collect();
    assert!(read == expected, "the GPU's read");
    drop((buffer, used));
}

/// With no source, a chunk whose pointer the device refuses is not imported, and the object over
/// it takes its own import, refused as its pointer is (`Unsupported`, `HostMemory`), nothing
/// taken as a dma-buf; a device that takes the pointer binds the object in the chunk.
#[test]
fn without_a_source_a_refused_pointer_refuses_the_object() {
    let scratch = Scratch::new();
    let Some(device) = held_device(&scratch) else {
        return;
    };
    let (ram, start, view, asked, _dropped) = ram(false);
    let used = device.use_host_ram(&ram).unwrap();
    let before = device.work_counts();
    let made = try_buffer(&device, view, start, AT, LENGTH);
    let after = device.work_counts();
    let failed = after.host_ram_pointer_imports_failed - before.host_ram_pointer_imports_failed;
    println!(
        "HOST-RAM-DMA-BUF {}: no source: {failed} pointer import(s) failed; the object {}",
        device.name(),
        match &made {
            Ok(_) => "made".to_owned(),
            Err(error) => format!("refused: {error}"),
        },
    );
    assert_eq!(
        after.host_ram_dma_buf_imports,
        before.host_ram_dma_buf_imports
    );
    assert!(asked.lock().unwrap().requests.is_empty());
    match (failed, made) {
        (0, Ok(buffer)) => {
            assert_eq!(buffer.host_backing(), Some(HostBacking::InHostRam));
            assert_eq!(
                after.host_ram_chunks_imported - before.host_ram_chunks_imported,
                1
            );
        }
        (_, Err(Error::Unsupported(refusal))) => {
            assert_eq!(refusal.feature, metal2vulkan::error::Feature::HostMemory);
            assert_eq!(
                after.host_ram_chunks_imported,
                before.host_ram_chunks_imported
            );
        }
        (failed, made) => panic!("{failed} pointer imports failed, and the object: {made:?}"),
    }
    drop(used);
}

/// The most bytes one udmabuf of this host holds: its `size_limit_mb` (64 MiB, the default, where
/// unread).
fn udmabuf_size_limit() -> u64 {
    std::fs::read_to_string("/sys/module/udmabuf/parameters/size_limit_mb")
        .ok()
        .and_then(|limit| limit.trim().parse::<u64>().ok())
        .unwrap_or(64)
        .saturating_mul(MIB)
}

/// Where each listed import costs every submission, a RAM whose dma-bufs may be as long as the
/// host's udmabuf limit allows is imported in chunks that long (the whole RAM where the limit
/// takes it, else two of 64 MiB), none ahead, each by the device's thread once an object over it
/// asked (or, where the device's barriers do not order blits through two imports of the same
/// pages, by that object: [`runs_alone`]), and, the RAM made too long for the memory the
/// device's imports count against, released once idle (the module docs); elsewhere it keeps its
/// 64 MiB chunks. The device's thread runs (no hold), so the release is its.
#[test]
fn where_each_listed_import_costs_a_submission_the_ram_is_long_chunks_released_once_idle() {
    let scratch = Scratch::new();
    if !udmabuf_openable() {
        return;
    }
    let device = device(&scratch.0);
    if device.shared().host_import.is_err() {
        eprintln!("HOST-RAM-LISTED: the device imports no host memory here; nothing to test");
        return;
    }
    // Half the RAM, which it fits in with unified memory or without.
    device.count_host_ram_imports_against(RAM_BYTES / 2);
    let listed = device.pointer_imports_tax_submissions() && imports_dma_bufs(&device);
    let largest = udmabuf_size_limit();
    let (ram, start, view, asked, dropped) = ram(true);
    let ram = Arc::into_inner(ram).expect("the test's RAM");
    let ram = Arc::new(ram.with_largest_dma_buf(largest));
    fill_pattern(view, RAM_BYTES);
    let used = device.use_host_ram(&ram).unwrap();
    let chunks = device.host_ram_chunks();
    println!(
        "HOST-RAM-LISTED {}: each listed import costs a submission: {listed}; the udmabuf limit \
         {} MiB; chunks {chunks:?}",
        device.name(),
        largest >> 20,
    );
    let halves = vec![start..start + 64 * MIB, start + 64 * MIB..start + RAM_BYTES];
    if !listed {
        assert_eq!(chunks, [(false, halves)]);
        drop((used, ram));
        return;
    }
    let spans = match largest >= RAM_BYTES {
        true => std::iter::once(start..start + RAM_BYTES).collect(),
        false => halves,
    };
    assert_eq!(chunks, [(true, spans.clone())]);
    let count = spans.len() as u64;
    let before = device.work_counts();
    let delta =
        |now: &crate::WorkCounts, field: fn(&crate::WorkCounts) -> u64| field(now) - field(&before);
    let run_bound = |counts: &crate::WorkCounts| counts.run_bound_buffers_made;
    let chunk_bound = |counts: &crate::WorkCounts| counts.chunk_bound_buffers_made;
    let imported = |counts: &crate::WorkCounts| counts.host_ram_chunks_imported;
    let runs_live = |counts: &crate::WorkCounts| counts.run_imports_live;
    let until = |done: &dyn Fn(&crate::WorkCounts) -> bool, what: &str| {
        let began = Instant::now();
        loop {
            let counts = device.work_counts();
            if done(&counts) {
                return counts;
            }
            assert!(began.elapsed() < WAIT, "{what}");
            std::thread::sleep(Duration::from_millis(20));
        }
    };
    // The first object over a chunk not imported: bound in an import of its own run, made on
    // this thread, and the chunk asked for, which the device's thread imports; where the device's
    // barriers do not order blits through two imports of the same pages, in the chunk's import,
    // which it makes.
    let alone = runs_alone(&device);
    let first = buffer(&device, view, start, AT, LENGTH);
    let made = device.work_counts();
    assert_eq!(
        (delta(&made, run_bound), delta(&made, chunk_bound)),
        (alone, 1 - alone),
        "the first object over a chunk not imported lies in its run's import where runs are \
         imported alone, else in the chunk's"
    );
    until(
        &|counts| delta(counts, imported) == 1,
        "the asked chunk was never imported",
    );
    // Past the first 64 MiB: in the chunk where the RAM is one, else in its run's import while
    // the device's thread imports the second chunk (where no run is imported alone, in the second
    // chunk's import, which it makes).
    let second = buffer(&device, view, start, 64 * MIB + AT, LENGTH);
    let all = until(
        &|counts| delta(counts, imported) == count,
        "the second asked chunk was never imported",
    );
    // One more over each chunk, now imported: in it, holding it.
    let holders: Vec<Buffer> = (spans.iter())
        .map(|span| {
            buffer(
                &device,
                view,
                start,
                span.start - start + AT + 2 * LENGTH,
                LENGTH,
            )
        })
        .collect();
    let held = device.work_counts();
    assert_eq!(
        delta(&held, run_bound),
        alone * count,
        "one run's import per chunk where runs are imported alone"
    );
    assert_eq!(
        delta(&held, chunk_bound),
        2 + (1 - alone) * count,
        "the objects after the chunks' imports, and the first over each where no run is \
         imported alone"
    );
    assert_eq!(
        delta(&held, |counts| counts.host_ram_chunks_imported_ahead),
        0
    );
    assert_eq!(
        delta(&held, |counts| counts.host_ram_chunk_bytes),
        RAM_BYTES
    );
    assert_eq!(
        delta(&all, |counts| counts.host_ram_dma_buf_imports),
        (1 + alone) * count
    );
    let runs_asked = [start + AT, start + 64 * MIB + AT];
    let mut asked_for: Vec<(u64, u64, u64)> = (spans.iter())
        .map(|span| (span.start, span.end - span.start, span.end - span.start))
        .chain((runs_asked.iter().take((alone * count) as usize)).map(|&at| (at, LENGTH, LENGTH)))
        .collect();
    asked_for.sort_unstable();
    let mut requests = asked.lock().unwrap().requests.clone();
    requests.sort_unstable();
    assert_eq!(
        requests, asked_for,
        "each chunk and each run's import asked once"
    );
    // The GPU's fill through the second buffer, past the first 64 MiB, reaches the file's pages.
    blit(&device, |encoder| {
        encoder.fill_buffer(&second, 0..LENGTH, 0x3c).unwrap()
    });
    let filled = 64 * MIB + AT;
    assert!(
        bytes(start, filled..filled + LENGTH)
            .iter()
            .all(|&b| b == 0x3c),
        "the GPU's fill"
    );
    // Held past the idle window by the buffers in them: kept. The runs' imports, which only a
    // blit read, are released once idle.
    std::thread::sleep(IDLE_WAIT);
    let kept = until(
        &|counts| runs_live(counts) == runs_live(&before),
        "the runs' idle imports were never released",
    );
    assert_eq!(
        kept.host_ram_chunks_released,
        before.host_ram_chunks_released
    );
    assert_eq!(kept.host_ram_chunks_live, held.host_ram_chunks_live);
    assert!(delta(&kept, |counts| counts.host_imports_released) >= alone * count);
    // The next blit through the first is bound in its chunk's import, and reads the file's bytes.
    let read = read_back(&device, LENGTH, |encoder, own| {
        (encoder.copy_from_buffer_to_buffer(&first, 0, own, 0, LENGTH)).unwrap();
    });
    assert!(
        read.iter()
            .zip(AT..)
            .all(|(&b, offset)| b == pattern(offset)),
        "the read through the chunk's import"
    );
    let again_in = device.work_counts();
    assert_eq!(
        runs_live(&again_in),
        runs_live(&before),
        "no run imported again"
    );
    assert!(delta(&again_in, |counts| counts.host_imports_reimported) >= alone);
    // Once they went, released by the device's thread after its window, and freed. The thread
    // counts a release before the chunk's drop frees it, so a count read between the two shows
    // the release with the chunk still live: the wait is for both.
    drop((first, second, holders));
    let gone = Instant::now();
    let before_freed = std::cell::Cell::new(0);
    let released = until(
        &|counts| {
            let released = delta(counts, |counts| counts.host_ram_chunks_released) == count;
            let freed = counts.host_ram_chunks_live + count == held.host_ram_chunks_live;
            before_freed.set(before_freed.get() + u32::from(released && !freed));
            released && freed
        },
        "the idle chunks were never released and freed",
    );
    println!(
        "HOST-RAM-LISTED {}: {count} released {:.1} s after the last buffer went; {} chunk(s) \
         live; {} read(s) of the counts saw the release before the free",
        device.name(),
        gone.elapsed().as_secs_f64(),
        released.host_ram_chunks_live,
        before_freed.get(),
    );
    // A copy over a released chunk: its run imported alone for it, the chunk asked for again;
    // where no run is imported alone, the chunk imported for it.
    let run = start + filled..start + filled + LENGTH;
    let gathered = read_back(&device, LENGTH, |encoder, own| {
        (encoder.copy_from_host_ram(&ram, std::slice::from_ref(&run), own, 0)).unwrap();
    });
    assert!(gathered.iter().all(|&b| b == 0x3c), "the copy's gather");
    let copied = until(
        &|counts| {
            imported(counts) == imported(&released) + 1 && runs_live(counts) == runs_live(&before)
        },
        "the copy's chunk was never imported again, or its run's import outlived it",
    );
    assert_eq!(
        copied.host_ram_dma_buf_imports - released.host_ram_dma_buf_imports,
        1 + alone,
        "the copy's run alone where runs are imported alone, and its chunk"
    );
    // The next object over that chunk lies in it, and reads the GPU's bytes.
    let again = buffer(&device, view, start, filled, LENGTH);
    let after = device.work_counts();
    assert_eq!(chunk_bound(&after) - chunk_bound(&copied), 1);
    let read = read_back(&device, LENGTH, |encoder, own| {
        (encoder.copy_from_buffer_to_buffer(&again, 0, own, 0, LENGTH)).unwrap();
    });
    assert!(
        read.iter().all(|&b| b == 0x3c),
        "the GPU's read after the release"
    );
    drop((again, used, ram));
    let began = Instant::now();
    while !dropped.load(Ordering::SeqCst) {
        assert!(
            began.elapsed() < WAIT,
            "the RAM's keep-alive was never dropped"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// Where each listed import costs every submission and the RAM fits in the memory the device's
/// imports count against (made four times the RAM by the test), its chunks, once imported, stay
/// imported past their objects and the idle window: none is released, and the next object over
/// each lies in its chunk with no import made for it. Elsewhere nothing to test.
#[test]
fn where_the_ram_fits_its_listed_chunks_stay_imported_once_idle() {
    let scratch = Scratch::new();
    if !udmabuf_openable() {
        return;
    }
    let device = device(&scratch.0);
    if device.shared().host_import.is_err() {
        eprintln!("HOST-RAM-KEPT: the device imports no host memory here; nothing to test");
        return;
    }
    device.count_host_ram_imports_against(4 * RAM_BYTES);
    let listed = device.pointer_imports_tax_submissions() && imports_dma_bufs(&device);
    if !listed {
        eprintln!(
            "HOST-RAM-KEPT {}: no listed imports here; nothing to test",
            device.name()
        );
        return;
    }
    let (ram, start, view, _asked, dropped) = ram(true);
    let ram = Arc::into_inner(ram).expect("the test's RAM");
    let ram = Arc::new(ram.with_largest_dma_buf(udmabuf_size_limit()));
    fill_pattern(view, RAM_BYTES);
    let used = device.use_host_ram(&ram).unwrap();
    let chunks = device.host_ram_chunks();
    let [(true, spans)] = &chunks[..] else {
        panic!("one listed RAM: {chunks:?}");
    };
    let count = spans.len() as u64;
    let before = device.work_counts();
    let until = |done: &dyn Fn(&crate::WorkCounts) -> bool, what: &str| {
        let began = Instant::now();
        loop {
            let counts = device.work_counts();
            if done(&counts) {
                return counts;
            }
            assert!(began.elapsed() < WAIT, "{what}");
            std::thread::sleep(Duration::from_millis(20));
        }
    };
    // One object over each chunk asks for it; the device's thread imports each.
    let firsts: Vec<Buffer> = (spans.iter())
        .map(|span| buffer(&device, view, start, span.start - start + AT, LENGTH))
        .collect();
    let imported = until(
        &|counts| counts.host_ram_chunks_imported - before.host_ram_chunks_imported == count,
        "the asked chunks were never imported",
    );
    // The objects gone, past the chunks' idle window and a tick of the thread: kept.
    drop(firsts);
    std::thread::sleep(IDLE_WAIT);
    let idle = device.work_counts();
    assert_eq!(
        idle.host_ram_chunks_released, before.host_ram_chunks_released,
        "a chunk of a RAM that fits was released"
    );
    assert_eq!(idle.host_ram_chunks_live, imported.host_ram_chunks_live);
    // The next object over each lies in its chunk, as imported: no import made for it.
    let nexts: Vec<Buffer> = (spans.iter())
        .map(|span| buffer(&device, view, start, span.start - start + AT, LENGTH))
        .collect();
    let after = device.work_counts();
    assert_eq!(
        (
            after.chunk_bound_buffers_made - idle.chunk_bound_buffers_made,
            after.run_bound_buffers_made - idle.run_bound_buffers_made,
            after.host_ram_chunks_imported - idle.host_ram_chunks_imported,
            after.host_ram_dma_buf_imports - idle.host_ram_dma_buf_imports,
        ),
        (count, 0, 0, 0),
        "the objects after the idle window lie in the kept chunks"
    );
    // The GPU reads the file's bytes through them.
    let read = read_back(&device, LENGTH, |encoder, own| {
        (encoder.copy_from_buffer_to_buffer(&nexts[0], 0, own, 0, LENGTH)).unwrap();
    });
    assert!(
        read.iter()
            .zip(AT..)
            .all(|(&b, offset)| b == pattern(offset)),
        "the read through the kept chunk's import"
    );
    println!(
        "HOST-RAM-KEPT {}: {count} chunk(s) kept {} s past their objects, none released",
        device.name(),
        IDLE_WAIT.as_secs(),
    );
    drop((nexts, used, ram));
    let began = Instant::now();
    while !dropped.load(Ordering::SeqCst) {
        assert!(
            began.elapsed() < WAIT,
            "the RAM's keep-alive was never dropped"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The pages of a guest's tile as a consumer copies them: 4 KiB pages, about 7 MiB.
const TILE_PAGES: u64 = 1800;

/// How a tile's pages lie in the RAM ([`tile_runs`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tile {
    /// Single pages nine apart across the RAM's first 64 MiB, each pair descending, so that no
    /// two are one run: one range of the RAM where the pages between are in memory.
    Near,
    /// 36 groups of 50 pages two apart, the groups 3.5 MiB apart across both of the RAM's 64 MiB:
    /// 36 ranges, past the most a copy imports alone elsewhere (`copy::MOST_ALONE`).
    Spread,
    /// 90 runs of 20 pages, each run's pages descending (a surface's pages as a guest's
    /// allocator hands them out), the runs anywhere in the RAM: 90 ranges.
    Clusters,
}

/// The tile's runs, at the RAM's addresses from `start`, in the order its bytes take them.
fn tile_runs(start: u64, tile: Tile) -> Vec<std::ops::Range<u64>> {
    let pages: Vec<u64> = match tile {
        Tile::Near => (0..TILE_PAGES).map(|i| (i ^ 1) * 9 + 3).collect(),
        Tile::Spread => (0..TILE_PAGES)
            .map(|i| (i / 50) * (896 + 16) + ((i % 50) ^ 1) * 2 + 1)
            .collect(),
        Tile::Clusters => {
            // Each run's first page at a multiple of 32 pages, so that no two runs meet: an
            // xorshift's pick of the RAM's 1024 slots of 32 pages, none twice.
            let mut taken = [false; (RAM_BYTES / (32 * 4096)) as usize];
            let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
            let mut pages = Vec::with_capacity(TILE_PAGES as usize);
            for _ in 0..TILE_PAGES / 20 {
                let slot = loop {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let slot = (state % taken.len() as u64) as usize;
                    if !taken[slot] {
                        taken[slot] = true;
                        break slot as u64;
                    }
                };
                pages.extend((0..20).rev().map(|page| slot * 32 + page));
            }
            pages
        }
    };
    (pages.iter())
        .map(|page| {
            let at = start + page * 4096;
            at..at + 4096
        })
        .collect()
}

/// The runs a copy of `runs` (whole 4 KiB pages) lists on a device whose plan is listed: by
/// address, those that meet one (`copy::page_runs`).
fn sorted_runs(runs: &[std::ops::Range<u64>]) -> u64 {
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

/// The lists a copy of `runs` takes over the chunks `spans` not imported, at most `most` runs a
/// list: each chunk's runs by address, those that meet one ([`sorted_runs`]), every chunk's
/// together, `most` a list.
fn lists_of(runs: &[std::ops::Range<u64>], spans: &[std::ops::Range<u64>], most: usize) -> u64 {
    let sorted: u64 = (spans.iter())
        .map(|span| {
            let inside: Vec<std::ops::Range<u64>> = (runs.iter())
                .filter(|run| span.contains(&run.start))
                .cloned()
                .collect();
            sorted_runs(&inside)
        })
        .sum();
    sorted.div_ceil(most as u64)
}

/// The chunks of `spans` any of `runs` lies in.
fn chunks_of(runs: &[std::ops::Range<u64>], spans: &[std::ops::Range<u64>]) -> u64 {
    (spans.iter())
        .filter(|span| runs.iter().any(|run| span.contains(&run.start)))
        .count() as u64
}

/// The bytes of the RAM's file in memory (`mincore` over its mapping from `start`): the pages
/// the guest wrote, and those a pin made resident.
fn resident_bytes(start: u64) -> u64 {
    let mut held = vec![0u8; (RAM_BYTES / 4096) as usize];
    // SAFETY: the test's mapping of the RAM, mapped while the RAM lives; one byte a page.
    let answered = unsafe {
        libc::mincore(
            start as *mut libc::c_void,
            RAM_BYTES as usize,
            held.as_mut_ptr(),
        )
    };
    assert_eq!(answered, 0, "mincore: {}", std::io::Error::last_os_error());
    held.iter().filter(|&&byte| byte & 1 != 0).count() as u64 * 4096
}

/// A copy of the tile (`copy_from_host_ram` into `buffer`, or `copy_to_host_ram` from it), its
/// command buffer committed and waited for: the call's answer, the encoding's time and the whole
/// copy's.
fn tile_copy(
    device: &Device,
    ram: &HostRam,
    runs: &[std::ops::Range<u64>],
    buffer: &Buffer,
    gather: bool,
) -> (Result<(), Error>, Duration, Duration) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    let began = Instant::now();
    let copied = match gather {
        true => encoder.copy_from_host_ram(ram, runs, buffer, 0),
        false => encoder.copy_to_host_ram(buffer, 0, ram, runs),
    };
    let encoded = began.elapsed();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    (copied, encoded, began.elapsed())
}

/// What one copy of a tile took: the chunks it imported, the dma-bufs of the RAM's source
/// (`host_ram_dma_buf_imports`: chunks and ranges alone) and the lists of its runs, each a
/// dma-buf of the RAM's source of run lists (`host_ram_run_list_imports`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Took {
    chunks: u64,
    dma_bufs: u64,
    lists: u64,
}

/// A tile's copies, both ways: a gather into `own`, which reads the RAM's bytes page by page
/// (each the byte `expect` answers for the tile's page and the file's offset), and a scatter of
/// `own` (each page's index and the round in every byte), which writes each page; where `beside`
/// says, the page above each of the tile's pages (none of them) still holds what it answers.
/// Each copy printed (`HOST-RAM-TILE`) with its times and the imports it took.
struct TileRound<'a> {
    device: &'a Device,
    ram: &'a HostRam,
    runs: &'a [std::ops::Range<u64>],
    own: &'a Buffer,
    start: u64,
    what: String,
}

impl TileRound<'_> {
    /// The round's copies: what each took, gather first, and what failed.
    fn run(
        &self,
        round: u8,
        expect: impl Fn(usize, u64) -> u8,
        beside: Option<&dyn Fn(u64) -> u8>,
    ) -> ([Took; 2], Vec<String>) {
        let contents = self.own.contents().unwrap().as_ptr();
        let tile: std::collections::HashSet<u64> = (self.runs.iter())
            .map(|run| run.start - self.start)
            .collect();
        let mut failures = Vec::new();
        let mut took = [Took {
            chunks: 0,
            dma_bufs: 0,
            lists: 0,
        }; 2];
        for (way, gather) in [true, false].into_iter().enumerate() {
            let before = self.device.work_counts();
            if !gather {
                for page in 0..TILE_PAGES {
                    let byte = (page as u8) ^ (0x5a + round);
                    // SAFETY: the buffer's bytes, which no command uses now.
                    unsafe {
                        std::ptr::write_bytes(contents.add((page * 4096) as usize), byte, 4096)
                    };
                }
            }
            let (copied, encoded, whole) =
                tile_copy(self.device, self.ram, self.runs, self.own, gather);
            let after = self.device.work_counts();
            let delta = |field: fn(&crate::WorkCounts) -> u64| field(&after) - field(&before);
            let name = match gather {
                true => "gather",
                false => "scatter",
            };
            println!(
                "HOST-RAM-TILE {}: {}, {name}: {copied:?} in {:.2} ms encoding, {:.2} ms in all; \
                 copies {}, dispatches {}, regions {}, chunks imported {}, dma-buf imports {}, \
                 run lists imported {}, pointer imports failed {}, run imports live {}",
                self.device.name(),
                self.what,
                encoded.as_secs_f64() * 1e3,
                whole.as_secs_f64() * 1e3,
                delta(|c| c.host_ram_copies),
                delta(|c| c.host_ram_copy_dispatches),
                delta(|c| c.host_ram_copy_regions),
                delta(|c| c.host_ram_chunks_imported),
                delta(|c| c.host_ram_dma_buf_imports),
                delta(|c| c.host_ram_run_list_imports),
                delta(|c| c.host_ram_pointer_imports_failed),
                after.run_imports_live,
            );
            took[way] = Took {
                chunks: delta(|c| c.host_ram_chunks_imported),
                dma_bufs: delta(|c| c.host_ram_dma_buf_imports),
                lists: delta(|c| c.host_ram_run_list_imports),
            };
            let what = format!("{}, {name}", self.what);
            if let Err(error) = &copied {
                failures.push(format!("{what}: {error:?}"));
                continue;
            }
            for (page, run) in self.runs.iter().enumerate() {
                let offset = run.start - self.start;
                let read = bytes(self.start, offset..offset + 4096);
                let good = match gather {
                    true => {
                        // SAFETY: the buffer's bytes, which no command uses now.
                        let got =
                            unsafe { std::slice::from_raw_parts(contents.add(page * 4096), 4096) };
                        got == read.as_slice()
                            && (offset..).zip(got).all(|(at, &b)| b == expect(page, at))
                    }
                    false => {
                        let byte = (page as u8) ^ (0x5a + round);
                        let above = offset + 4096;
                        let kept = match beside {
                            Some(beside) if !tile.contains(&above) => {
                                let held = bytes(self.start, above..above + 4096);
                                (above..).zip(&held).all(|(at, &b)| b == beside(at))
                            }
                            _ => true,
                        };
                        read.iter().all(|&b| b == byte) && kept
                    }
                };
                if !good {
                    failures.push(format!("{what}: page {page} at {offset:#x}"));
                    break;
                }
            }
        }
        (took, failures)
    }
}

/// Whether a copy over a chunk not imported takes lists of its runs, given the RAM's source of
/// them: the device imports a RAM with a source as each listed import costing every submission
/// (`Plan::listed`, RADV's), and its barriers order blits through two imports of the same pages
/// (no `AliasedImportBlitsUnordered`, [`runs_alone`]).
fn copies_by_lists(device: &Device) -> bool {
    device.pointer_imports_tax_submissions() && imports_dma_bufs(device) && runs_alone(device) == 1
}

/// Past a tick of the device's thread, which imports a chunk asked for at once.
const ASK_WAIT: Duration = Duration::from_secs(1);

/// The most runs a list of the tests' sources holds: udmabuf's default `list_limit`.
const LIST_RUNS: usize = 1024;

/// Whether every list `asked` asked for held at most [`LIST_RUNS`] runs, each whole pages with a
/// granule of its length, by start, sharing no page: the source's contract.
fn lists_kept_the_contract(asked: &Asked) -> bool {
    (asked.lists.iter()).all(|list| {
        let whole = (list.iter()).all(|&(first, length, granule)| {
            first % 4096 == 0 && length % 4096 == 0 && length > 0 && granule == length
        });
        let apart = list
            .windows(2)
            .all(|pair| pair[0].0 + pair[0].1 < pair[1].0);
        !list.is_empty() && list.len() <= LIST_RUNS && whole && apart
    })
}

/// A guest's tile of [`TILE_PAGES`] scattered pages copied both ways over a RAM of shared pages
/// (QEMU's `memory-backend-memfd,share=on`) with a source of dma-bufs, as a consumer's cache asks
/// copy it: first with its chunks not imported, then with them imported by objects over them.
/// Each gather reads the RAM's bytes page by page, each scatter writes each page and leaves the
/// page above it alone. Where each listed import costs every submission and the device's barriers
/// order blits through two imports of the same pages (RADV), a copy over a chunk not imported,
/// of a RAM with a source of run lists, takes lists of its runs (1,800 runs, two lists of at most
/// 1,024), no chunk and no other dma-buf, and asks the device's thread for none; of a RAM without
/// one, it imports the chunk, as before. Elsewhere the source of run lists is never asked. Over
/// chunks imported, no import. Printed (`HOST-RAM-TILE`): each copy's answer, its encoding's time
/// and the whole copy's, and the imports it took.
#[test]
fn a_tile_of_scattered_pages_copies_both_ways_over_a_ram_of_shared_pages() {
    let scratch = Scratch::new();
    if !udmabuf_openable() {
        return;
    }
    let device = device(&scratch.0);
    if device.shared().host_import.is_err() {
        eprintln!("HOST-RAM-TILE: the device imports no host memory here; nothing to test");
        return;
    }
    let by_lists = copies_by_lists(&device);
    let own = device
        .new_buffer(TILE_PAGES * 4096, ResourceOptions::shared())
        .unwrap();
    let mut failures = Vec::new();
    for (tile, lists) in [
        (Tile::Near, Some(LIST_RUNS)),
        (Tile::Spread, Some(LIST_RUNS)),
        (Tile::Spread, None),
    ] {
        // A RAM of its own, none of its chunks imported.
        let (ram, start, view, asked, _dropped) = listed_ram(true, lists);
        let ram = Arc::into_inner(ram).expect("the test's RAM");
        let ram = Arc::new(ram.with_largest_dma_buf(udmabuf_size_limit()));
        fill_pattern(view, RAM_BYTES);
        // Before the use: a device that imports ahead may import a chunk at once.
        let first = device.work_counts();
        let used = device.use_host_ram(&ram).unwrap();
        let chunks = device.host_ram_chunks();
        let [(_, spans)] = &chunks[..] else {
            panic!("one RAM: {chunks:?}");
        };
        let runs = tile_runs(start, tile);
        let imported = |counts: &crate::WorkCounts| {
            counts.host_ram_chunks_imported - first.host_ram_chunks_imported
        };
        let named = match lists {
            Some(_) => "lists",
            None => "no lists",
        };
        let round = TileRound {
            device: &device,
            ram: &ram,
            runs: &runs,
            own: &own,
            start,
            what: format!("{tile:?}, {named}, over chunks {chunks:?}, not imported"),
        };
        // The scatter's bytes stay for the next round's gather.
        let (took, failed) = round.run(0, |_, at| pattern(at), Some(&pattern));
        failures.extend(failed);
        match (by_lists, lists) {
            (true, Some(most)) => {
                let each = Took {
                    chunks: 0,
                    dma_bufs: 0,
                    lists: lists_of(&runs, spans, most),
                };
                assert_eq!(
                    took, [each; 2],
                    "{tile:?}: a copy over chunks not imported took lists of its runs alone"
                );
                std::thread::sleep(ASK_WAIT);
                assert_eq!(
                    imported(&device.work_counts()),
                    0,
                    "{tile:?}: a copy asked the device's thread for a chunk"
                );
                assert_eq!(
                    device.work_counts().run_imports_live,
                    first.run_imports_live,
                    "{tile:?}: a list outlived its copy"
                );
                let asked = asked.lock().unwrap();
                assert!(
                    lists_kept_the_contract(&asked),
                    "{tile:?}: a list's requests"
                );
                assert_eq!(asked.lists.len() as u64, 2 * each.lists);
            }
            (true, None) => {
                // The chunks, imported for the gather and kept for the scatter.
                let gather = Took {
                    chunks: chunks_of(&runs, spans),
                    dma_bufs: chunks_of(&runs, spans),
                    lists: 0,
                };
                let scatter = Took {
                    chunks: 0,
                    dma_bufs: 0,
                    lists: 0,
                };
                assert_eq!(
                    took,
                    [gather, scatter],
                    "{tile:?}: without lists, the chunk"
                );
            }
            (false, _) => {
                assert!(
                    asked.lock().unwrap().lists.is_empty(),
                    "{tile:?}: the source of run lists asked where copies take none"
                );
                assert_eq!(took.map(|took| took.lists), [0; 2]);
            }
        }
        // An object over each chunk: the chunks asked for, and imported.
        let holders: Vec<Buffer> = (spans.iter())
            .map(|span| buffer(&device, view, start, span.start - start + AT, LENGTH))
            .collect();
        let began = Instant::now();
        while imported(&device.work_counts()) < spans.len() as u64 {
            assert!(
                began.elapsed() < WAIT,
                "{tile:?}: the objects' chunks were never imported"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        let round = TileRound {
            what: format!("{tile:?}, {named}, over chunks {chunks:?}, imported"),
            ..round
        };
        // The gather reads the first round's scatter.
        let (took, failed) = round.run(1, |page, _| (page as u8) ^ 0x5a, Some(&pattern));
        failures.extend(failed);
        let none = Took {
            chunks: 0,
            dma_bufs: 0,
            lists: 0,
        };
        assert_eq!(
            took, [none; 2],
            "{tile:?}: a copy over imported chunks imported more"
        );
        drop((holders, used, ram));
    }
    drop(own);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A tile copied both ways over a RAM of which the guest wrote the tile's pages alone, the rest
/// of its file never touched, with a source of run lists: where copies take lists (RADV,
/// [`copies_by_lists`]), the copies make no page of the file resident but the tile's own
/// (`mincore` over the RAM's mapping), however its pages lie: lists of its runs, by address
/// across the RAM's two chunks (90 clusters in both, one list), no chunk imported, none asked
/// for. A chunk's import would make all of the chunk resident. Elsewhere the
/// copies' bytes are checked and the resident bytes printed. Printed: `HOST-RAM-TILE` per copy,
/// `HOST-RAM-RESIDENT` per tile.
#[test]
fn copies_over_a_listed_ram_make_no_untouched_page_of_it_resident() {
    let scratch = Scratch::new();
    if !udmabuf_openable() {
        return;
    }
    let device = device(&scratch.0);
    if device.shared().host_import.is_err() {
        eprintln!("HOST-RAM-RESIDENT: the device imports no host memory here; nothing to test");
        return;
    }
    let by_lists = copies_by_lists(&device);
    let own = device
        .new_buffer(TILE_PAGES * 4096, ResourceOptions::shared())
        .unwrap();
    let mut failures = Vec::new();
    for tile in [Tile::Clusters, Tile::Near] {
        let (ram, start, view, _asked, _dropped) = listed_ram(true, Some(LIST_RUNS));
        let ram = Arc::into_inner(ram).expect("the test's RAM");
        // Two chunks of 64 MiB where each listed import costs every submission: the clusters lie
        // in both, and are one list.
        let ram = Arc::new(ram.with_largest_dma_buf(udmabuf_size_limit().min(64 * MIB)));
        let runs = tile_runs(start, tile);
        // The guest's writes: the tile's pages alone.
        for run in &runs {
            let offset = run.start - start;
            // SAFETY: the view's bytes, which no command buffer uses now.
            unsafe {
                for at in offset..offset + 4096 {
                    *((view + at) as *mut u8) = pattern(at);
                }
            }
        }
        let touched = resident_bytes(start);
        let used = device.use_host_ram(&ram).unwrap();
        let first = device.work_counts();
        let chunks = device.host_ram_chunks();
        let [(_, spans)] = &chunks[..] else {
            panic!("one RAM: {chunks:?}");
        };
        let round = TileRound {
            device: &device,
            ram: &ram,
            runs: &runs,
            own: &own,
            start,
            what: format!("{tile:?} over pages the guest wrote alone"),
        };
        // The pages above the tile's are left unread: a read of a page of the file not in memory
        // would make it resident.
        let (took, failed) = round.run(0, |_, at| pattern(at), None);
        failures.extend(failed);
        std::thread::sleep(ASK_WAIT);
        let after = resident_bytes(start);
        let counts = device.work_counts();
        println!(
            "HOST-RAM-RESIDENT {}: {tile:?}: {} KiB of the file in memory before the copies, {} \
             KiB after; {} chunk(s) imported",
            device.name(),
            touched >> 10,
            after >> 10,
            counts.host_ram_chunks_imported - first.host_ram_chunks_imported,
        );
        if by_lists {
            let each = Took {
                chunks: 0,
                dma_bufs: 0,
                lists: lists_of(&runs, spans, LIST_RUNS),
            };
            assert_eq!(took, [each; 2], "{tile:?}: lists of the runs alone");
            assert_eq!(
                counts.host_ram_chunks_imported, first.host_ram_chunks_imported,
                "{tile:?}: a chunk imported or asked for"
            );
            assert_eq!(
                after, touched,
                "{tile:?}: the copies made pages the guest never wrote resident"
            );
        }
        drop((used, ram));
    }
    drop(own);
    assert!(failures.is_empty(), "{failures:#?}");
}
