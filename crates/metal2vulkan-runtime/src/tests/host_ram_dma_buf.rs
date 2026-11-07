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
//! - **Few listed imports.** Where each listed import costs every submission (the import tax
//!   held, dma-bufs imported: RADV), a RAM whose source's dma-bufs may be as long as it is one
//!   chunk ([`HostRam::with_largest_dma_buf`]), the source asked once for all of it; no chunk is
//!   imported ahead: the first object over a chunk lies in its own run's import and asks the
//!   device's thread for the chunk, which imports it; those after it lie in the chunk. The runs'
//!   imports, which only blits read, are released once idle, the next blit binding the buffer in
//!   the chunk; the chunk is kept while an object over it lives, and released by the device's
//!   thread once none did for its idle window (`WorkCounts::host_ram_chunks_released`); a copy
//!   over it then imports its run alone and asks for the chunk again.
//!   Elsewhere the RAM keeps its 64 MiB chunks.
//!
//! Each run prints the device's verdict and the route its chunk took: the evidence of which path
//! ran. Where `/dev/udmabuf` cannot be opened, or the device imports no host memory, the test says
//! so and passes without running.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux.

use super::host_residency::{device, udmabuf, Scratch};
use crate::submit_check::{Check, Verdict};
use crate::{
    Buffer, CommandBufferStatus, Device, DmaBufChunk, DmaBufRequest, DmaBufSource, Error,
    HostBacking, HostMemory, HostRam, KeepAlive, Origin, ResourceOptions, RunResolver, Size,
    TextureDescriptor,
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
/// refused as lying outside the RAM.
#[derive(Default)]
struct Asked {
    requests: Vec<(u64, u64, u64)>,
    outside: Vec<(u64, u64)>,
}

/// The RAM over a new sealed `memfd`, with a source of dma-bufs of it where `sourced`; the RAM's
/// first byte, the view's, what the source was asked, and whether the keep-alive was dropped.
fn ram(sourced: bool) -> (Arc<HostRam>, u64, u64, Arc<Mutex<Asked>>, Arc<AtomicBool>) {
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
    let at = NonNull::new(start as *mut u8).unwrap();
    // SAFETY: the mapping is readable and writable until the keep-alive unmaps it.
    let ram = unsafe { HostRam::new(&[(at, RAM_BYTES)], Some(keep)) }.unwrap();
    let ram = match sourced {
        true => ram.with_dma_buf(source),
        false => ram,
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
/// asked, and released once idle (the module docs); elsewhere it keeps its 64 MiB chunks. The
/// device's thread runs (no hold), so the imports and the release are its.
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
    // this thread, and the chunk asked for, which the device's thread imports.
    let first = buffer(&device, view, start, AT, LENGTH);
    let made = device.work_counts();
    assert_eq!(
        (delta(&made, run_bound), delta(&made, chunk_bound)),
        (1, 0),
        "the first object over a chunk not imported lies in its run's import"
    );
    until(
        &|counts| delta(counts, imported) == 1,
        "the asked chunk was never imported",
    );
    // Past the first 64 MiB: in the chunk where the RAM is one, else in its run's import while
    // the device's thread imports the second chunk.
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
    assert_eq!(delta(&held, run_bound), count, "one run's import per chunk");
    assert_eq!(
        delta(&held, chunk_bound),
        2,
        "the objects after the chunks' imports"
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
        2 * count
    );
    let runs_asked = [start + AT, start + 64 * MIB + AT];
    let mut asked_for: Vec<(u64, u64, u64)> = (spans.iter())
        .map(|span| (span.start, span.end - span.start, span.end - span.start))
        .chain((runs_asked.iter().take(count as usize)).map(|&at| (at, LENGTH, LENGTH)))
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
    assert!(delta(&kept, |counts| counts.host_imports_released) >= count);
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
    assert!(delta(&again_in, |counts| counts.host_imports_reimported) >= 1);
    // Once they went, released by the device's thread after its window.
    drop((first, second, holders));
    let gone = Instant::now();
    let released = until(
        &|counts| delta(counts, |counts| counts.host_ram_chunks_released) == count,
        "the idle chunks were never released",
    );
    println!(
        "HOST-RAM-LISTED {}: {count} released {:.1} s after the last buffer went; {} chunk(s) live",
        device.name(),
        gone.elapsed().as_secs_f64(),
        released.host_ram_chunks_live,
    );
    assert_eq!(
        released.host_ram_chunks_live + count,
        held.host_ram_chunks_live
    );
    // A copy over a released chunk: its run imported alone for it, the chunk asked for again.
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
        2,
        "the copy's run alone, and its chunk"
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
