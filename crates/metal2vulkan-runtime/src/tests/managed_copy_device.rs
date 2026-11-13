//! `Managed` no-copy buffers on a device without unified memory (`managed_copy.rs`): two copies,
//! the caller's pages and the GPU's own in device-local memory, as Metal keeps a managed resource
//! on a Mac with a discrete GPU. Each leg runs over both host copies: an import of the pages
//! ([`Route::Import`]) and the pages alone, where the caller's gate refuses the import
//! ([`Route::Pages`]); the copies leg and the leg over a copy's bytes also over pages the caller
//! vouched writable whose import the gate refuses ([`Route::Vouched`]), where no completion takes
//! the pages again (`WorkCounts::managed_writes_undone`).
//!
//! - **Copies.** The pages' bytes reach the first command buffer; a CPU write reaches the GPU where
//!   `did_modify_range` names it, and not where nothing does; the GPU's writes reach the pages at
//!   a blit's `synchronize_resource`, and not without one. The buffer keeps its Metal address and
//!   its `contents` pointer, no copy of the pages is made (`copied.rs`), and its deallocator runs
//!   once, at its release.
//!   Copies in that wait for one submission land in the order of their calls.
//! - **Bounded.** A 256 MiB buffer over pages no import maps holds its 256 MiB of device-local
//!   memory and, across cycles of a whole `did_modify_range` and a command buffer, at most the
//!   bound of what is staged, none once the cycle completed.
//! - **Rebind.** A rebind copies the new pages in whole before the next command buffer:
//!   `contents` is the new pointer, the Metal address stays, the old pages' deallocator runs.
//! - **Detach.** The device copy stays, which the GPU reads and writes; `contents` is `None`, and
//!   a synchronize copies nothing out; a rebind gives the buffer pages again.
//! - **Read-only pages** (a rebind over pages the process maps read-only): the GPU's writes are
//!   dropped. A synchronize writes nothing to the pages, and the next command buffer reads the
//!   pages' bytes, not the GPU's writes.
//! - **Textures** over the buffer's bytes follow the device copy: their GPU reads take what was
//!   copied in, and a `synchronize_resource` of one copies the GPU's writes out to the pages.
//! - **Recovery** carries the buffer: its host copy made again over the same pages, its device
//!   copy made on the new `VkDevice` and the pages copied in.
//! - **Elsewhere, one copy.** A `Shared` no-copy buffer, and a `Managed` one on a device with
//!   unified memory, lie in their pages. There a `Managed` buffer's `did_modify_range` copies
//!   nothing: neither the runtime's own nor one over the caller's pages copies a byte in or
//!   stages one (`WorkCounts::managed_bytes_copied_in`, `managed_staging_bytes`), and the command
//!   buffer after it reads the CPU's write.
//! - **The copy-in bench** (ignored, `MANAGED-COPY-IN`; run in release): what a
//!   `did_modify_range` of 256 KiB, 4 MiB and 32 MiB, written by the CPU before each commit of a
//!   command buffer that reads the range's last bytes, costs the commit call and the wall time to
//!   the completion, over the runtime's own `Managed` buffer and one over the caller's pages
//!   through their import, on the default device and on one of `DevicePolicy::exact_answers()`:
//!   the medians, the range over the median wall time, and the counts of the bytes copied in and
//!   of the buffers the device parked, took again and destroyed past its bounds.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects. Where the device has unified memory
//! (lavapipe's, Apple silicon's), a two-copy leg checks only that the buffer is one copy.

use super::blit_holds::{device, options, Scratch, WAIT};
use super::host_ram_device::gate;
use super::layer_claim::Watched;
use crate::{
    BlitCommandEncoder, BlitOption, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue,
    CpuCacheMode, Deallocator, Device, DeviceOptions, HazardTrackingMode, HostBacking, HostMemory,
    ImportRequest, Origin, Region, ResourceOptions, ResourceRef, Size, StorageMode,
    TextureDescriptor, WorkCounts,
};
use metal2vulkan::metal::PixelFormat;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The bytes of each buffer of the copy legs: whole pages of any host.
const BYTES: u64 = 1 << 20;

/// The bytes of the bounded leg's buffer.
const LARGE: u64 = 256 << 20;

/// What a copy in may stage before a submission takes it: the bound past which the call makes
/// one (`texture/first_move.rs`'s `STAGED_BYTES`), and one staging buffer more.
const STAGED_MOST: u64 = (256 << 20) + crate::managed_copy::STAGED_CHUNK;

/// The cycles of the bounded leg.
const CYCLES: u8 = 4;

/// The buffer's host copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    /// An import of the pages.
    Import,
    /// The pages alone: the caller's gate refuses the import.
    Pages,
    /// The pages alone, the caller's gate refusing the import of pages whose write permission
    /// the caller vouched for (`HostMemory::assume_writable`).
    Vouched,
}

/// Anonymous private pages of the test's own, zeroed, unmapped when dropped: declared before the
/// device, so dropped after every buffer over them.
struct Pages {
    at: NonNull<u8>,
    length: u64,
}

impl Pages {
    fn new(length: u64) -> Pages {
        // SAFETY: a fresh private anonymous mapping, unmapped in `drop`.
        let at = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(at, libc::MAP_FAILED);
        Pages {
            at: NonNull::new(at.cast()).unwrap(),
            length,
        }
    }

    /// Set `range` of its bytes to `byte` from the CPU.
    fn write(&self, range: std::ops::Range<u64>, byte: u8) {
        assert!(range.end <= self.length);
        // SAFETY: the test's mapping, which the CPU writes only while no command buffer runs.
        unsafe {
            std::ptr::write_bytes(
                self.at.as_ptr().add(range.start as usize),
                byte,
                (range.end - range.start) as usize,
            )
        };
    }

    /// Set its bytes to [`pattern`]'s.
    fn write_pattern(&self) {
        // SAFETY: as in `write`.
        let bytes =
            unsafe { std::slice::from_raw_parts_mut(self.at.as_ptr(), self.length as usize) };
        bytes.copy_from_slice(&pattern(self.length));
    }

    /// Its bytes, as the CPU reads them.
    fn bytes(&self) -> Vec<u8> {
        // SAFETY: the test's mapping, read while no command buffer runs.
        unsafe { std::slice::from_raw_parts(self.at.as_ptr(), self.length as usize) }.to_vec()
    }

    /// Let the process only read them from now on.
    fn read_only(&self) {
        // SAFETY: the test's mapping, whole.
        let done = unsafe {
            libc::mprotect(
                self.at.as_ptr().cast(),
                self.length as usize,
                libc::PROT_READ,
            )
        };
        assert_eq!(done, 0);
    }

    /// Them, for a no-copy buffer or a rebind, by `route`.
    ///
    /// # Safety
    ///
    /// They outlive every buffer over them (declared before the device).
    unsafe fn memory(&self, route: Route) -> HostMemory {
        // SAFETY: the caller's contract.
        unsafe { memory_at(self.at, route) }
    }
}

/// The pages at `at`, for a no-copy buffer or a rebind, by `route`.
///
/// # Safety
///
/// They are writable pages of the test's own (until `read_only`, which no such leg calls), which
/// outlive every buffer over them.
unsafe fn memory_at(at: NonNull<u8>, route: Route) -> HostMemory {
    // SAFETY: the caller's contract.
    let memory = unsafe { HostMemory::new(at) };
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    match route {
        Route::Import => memory,
        Route::Pages => memory.with_import_gate(gate(&asked, false)),
        // SAFETY: the caller's contract: the pages are writable.
        Route::Vouched => unsafe {
            memory
                .with_import_gate(gate(&asked, false))
                .assume_writable()
        },
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: the mapping `new` made, which no buffer lies in any more.
        unsafe { libc::munmap(self.at.as_ptr().cast(), self.length as usize) };
    }
}

/// The bytes `0, 1, ..., 250, 0, 1, ...`: no byte equals its page's neighbour's at the same offset.
fn pattern(length: u64) -> Vec<u8> {
    (0..length).map(|i| (i % 251) as u8).collect()
}

/// `Managed` storage.
fn managed() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Managed,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    )
}

/// A deallocator's runs, each with the range it was handed.
type Ran = Arc<Mutex<Vec<(u64, u64)>>>;

/// A deallocator counting its runs in `ran`.
fn counted(ran: &Ran) -> Deallocator {
    let ran = Arc::clone(ran);
    Box::new(move |pointer, length| ran.lock().unwrap().push((pointer.as_ptr() as u64, length)))
}

/// Whether `ran` came to `runs` runs within [`WAIT`]: a deallocator may run on a handler thread a
/// moment after the release that let it go.
fn ran_to(ran: &Ran, runs: usize) -> bool {
    let began = Instant::now();
    while ran.lock().unwrap().len() < runs {
        if began.elapsed() > WAIT {
            return false;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    true
}

/// A no-copy buffer of `options` over `pages` whole, by `route`, its deallocator counted in `ran`.
fn no_copy(
    device: &Device,
    pages: &Pages,
    route: Route,
    options: ResourceOptions,
    ran: &Ran,
) -> Buffer {
    // SAFETY: the pages outlive every buffer over them.
    unsafe {
        device.new_buffer_with_bytes_no_copy(
            pages.memory(route),
            pages.length,
            options,
            Some(counted(ran)),
        )
    }
    .unwrap()
}

/// One command buffer of `queue` of the blit commands `encode` encodes, completed.
#[track_caller]
fn blit(queue: &CommandQueue, encode: impl FnOnce(&mut BlitCommandEncoder)) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    encode(&mut blit);
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    if command_buffer.wait_until_completed_for(WAIT).is_err() {
        panic!("a command buffer did not complete in {WAIT:?}");
    }
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None)
    );
}

/// `range` of `source` as the GPU reads it, into a new `Shared` buffer.
#[track_caller]
fn read(
    device: &Device,
    queue: &CommandQueue,
    source: &Buffer,
    range: std::ops::Range<u64>,
) -> Vec<u8> {
    let bytes = range.end - range.start;
    let target = device.new_buffer(bytes, ResourceOptions::shared()).unwrap();
    blit(queue, |blit| {
        blit.copy_from_buffer_to_buffer(source, range.start, &target, 0, bytes)
            .unwrap()
    });
    let contents = target.contents().unwrap();
    // SAFETY: the target's bytes, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), bytes as usize) }.to_vec()
}

/// `buffer`'s bytes set to `byte` by the GPU.
#[track_caller]
fn gpu_write(queue: &CommandQueue, buffer: &Buffer, byte: u8) {
    blit(queue, |blit| {
        blit.fill_buffer(buffer, 0..buffer.length(), byte).unwrap()
    });
}

/// A blit's `synchronize_resource` of `buffer`, completed.
#[track_caller]
fn synchronize(queue: &CommandQueue, buffer: &Buffer) {
    blit(queue, |blit| {
        blit.synchronize_resource(ResourceRef::Buffer(buffer))
            .unwrap()
    });
}

/// The device's counts once its staging buffers hold `bytes` bytes, waited for at most [`WAIT`]: a
/// staging buffer goes as its submission's completion is processed.
fn staged_down_to(device: &Device, bytes: u64) -> WorkCounts {
    let began = Instant::now();
    loop {
        let counts = device.work_counts();
        if counts.managed_staging_bytes <= bytes || began.elapsed() > WAIT {
            return counts;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// The device, and whether it keeps a `Managed` no-copy buffer as two copies: where it has unified
/// memory it does not, which the leg named `leg` checks on a buffer by `route`, then stops.
fn two_copy_device(scratch: &Scratch, leg: &str, route: Route) -> Option<Device> {
    let device = device(&scratch.0);
    if !device.properties().has_unified_memory {
        return Some(device);
    }
    eprintln!(
        "MANAGED-COPY {leg} {}: the device has unified memory; a `Managed` no-copy buffer lies in \
         its pages",
        device.name()
    );
    let pages = Pages::new(BYTES);
    let ran = Ran::default();
    let buffer = no_copy(&device, &pages, route, managed(), &ran);
    assert_ne!(buffer.host_backing(), Some(HostBacking::DeviceCopy));
    assert_eq!(device.work_counts().managed_device_copies_made, 0);
    None
}

/// The copies of a two-copy buffer by `route` (the module docs, **Copies**).
fn copies(route: Route) {
    let pages = Pages::new(BYTES);
    pages.write_pattern();
    let scratch = Scratch::new("managed-copies");
    let Some(device) = two_copy_device(&scratch, "copies", route) else {
        return;
    };
    let queue = device.new_command_queue().unwrap();
    let before = device.work_counts();
    let ran = Ran::default();
    let buffer = no_copy(&device, &pages, route, managed(), &ran);
    let made = device.work_counts();
    assert_eq!(buffer.host_backing(), Some(HostBacking::DeviceCopy));
    assert_eq!(buffer.contents(), Some(pages.at));
    assert_eq!(made.managed_device_copies - before.managed_device_copies, 1);
    let held = made.managed_device_copy_bytes - before.managed_device_copy_bytes;
    assert!((BYTES..BYTES + (64 << 10)).contains(&held), "{held}");
    assert!(made.managed_bytes_copied_in - before.managed_bytes_copied_in >= BYTES);
    assert_eq!(
        made.copied_host_buffers_made, before.copied_host_buffers_made,
        "{route:?}: no copy of the pages"
    );
    let address = buffer.gpu_address();
    assert_ne!(address.0, 0);

    // The first command buffer reads the pages' bytes.
    assert_eq!(read(&device, &queue, &buffer, 0..BYTES), pattern(BYTES));

    // A CPU write `did_modify_range` names reaches the GPU; one nothing names does not.
    pages.write(0..BYTES, 0xa5);
    buffer.did_modify_range(0..BYTES / 2).unwrap();
    let seen = read(&device, &queue, &buffer, 0..BYTES);
    let half = (BYTES / 2) as usize;
    assert!(seen[..half].iter().all(|&byte| byte == 0xa5), "{route:?}");
    assert_eq!(seen[half..], pattern(BYTES)[half..], "{route:?}");

    // The GPU's write reaches the pages at a synchronize, and not before.
    gpu_write(&queue, &buffer, 0x3c);
    assert_eq!(pages.bytes(), vec![0xa5; BYTES as usize], "{route:?}");
    let out = device.work_counts().managed_bytes_copied_out;
    synchronize(&queue, &buffer);
    assert_eq!(pages.bytes(), vec![0x3c; BYTES as usize], "{route:?}");
    assert!(device.work_counts().managed_bytes_copied_out - out >= BYTES);
    assert_eq!(buffer.gpu_address(), address);
    assert_eq!(buffer.contents(), Some(pages.at));
    // No completion took the pages again: the GPU's writes stay.
    assert_eq!(
        device.work_counts().managed_writes_undone,
        before.managed_writes_undone,
        "{route:?}"
    );

    // The release: the deallocator runs once, and nothing of the buffer's stays held.
    drop(buffer);
    assert!(ran_to(&ran, 1), "{route:?}: the deallocator did not run");
    let rest = staged_down_to(&device, before.managed_staging_bytes);
    assert_eq!(rest.managed_staging_bytes, before.managed_staging_bytes);
    assert_eq!(rest.managed_device_copies, before.managed_device_copies);
    assert_eq!(
        rest.managed_device_copy_bytes,
        before.managed_device_copy_bytes
    );
    assert_eq!(*ran.lock().unwrap(), [(pages.at.as_ptr() as u64, BYTES)]);
}

#[test]
fn copies_through_an_import() {
    copies(Route::Import);
}

#[test]
fn copies_through_vouched_pages_no_import_maps() {
    copies(Route::Vouched);
}

#[test]
fn copies_through_pages_no_import_maps() {
    copies(Route::Pages);
}

/// Copies in that wait for one submission land in the order of their calls: the making's, then
/// two `did_modify_range`s of the whole buffer, each of other bytes, read by one command buffer
/// (the module docs, **Copies**).
fn ordered(route: Route) {
    let pages = Pages::new(BYTES);
    pages.write_pattern();
    let scratch = Scratch::new("managed-ordered");
    let Some(device) = two_copy_device(&scratch, "ordered", route) else {
        return;
    };
    let queue = device.new_command_queue().unwrap();
    let ran = Ran::default();
    let buffer = no_copy(&device, &pages, route, managed(), &ran);
    for byte in [0x61, 0x62] {
        pages.write(0..BYTES, byte);
        buffer.did_modify_range(0..BYTES).unwrap();
    }
    assert_eq!(
        read(&device, &queue, &buffer, 0..BYTES),
        vec![0x62; BYTES as usize],
        "{route:?}"
    );
}

#[test]
fn copies_in_through_an_import_land_in_their_order() {
    ordered(Route::Import);
}

#[test]
fn copies_in_of_pages_no_import_maps_land_in_their_order() {
    ordered(Route::Pages);
}

/// The two-copy buffers of the making leg.
const MADE: usize = 8;

/// On a device without a fill queue, the copies in of two-copy buffers' makings, through an
/// import of their pages, go with the device's next submission, whose commands reach none of
/// those buffers: no command buffer can have read the bytes they replace, and each holds its
/// buffer's import until it is made (`buffer.rs`'s `Replaces::Unread`). Left for a batch that
/// reaches its buffer, each held its import while no command buffer used the buffer, which every
/// RADV submission lists (`WorkCounts::dma_buf_memories_live`). The buffers read their pages'
/// bytes after.
#[test]
fn makings_copies_in_go_with_the_next_submission() {
    let scratch = Scratch::new("managed-made");
    let Some(device) = two_copy_device(&scratch, "made", Route::Import) else {
        return;
    };
    let name = device.name().to_owned();
    if device.has_fill_queue() {
        // There a batch takes only the fills of the program's memory its commands may reach.
        eprintln!("MADE-COPY-IN on {name}: skipped, the device has a fill queue");
        return;
    }
    let pages: Vec<Pages> = (0..MADE)
        .map(|_| {
            let pages = Pages::new(BYTES);
            pages.write_pattern();
            pages
        })
        .collect();
    let queue = device.new_command_queue().unwrap();
    let other = device.new_buffer(BYTES, ResourceOptions::shared()).unwrap();
    let before = device.work_counts();
    let ran = Ran::default();
    let buffers: Vec<Buffer> = (pages.iter())
        .map(|pages| no_copy(&device, pages, Route::Import, managed(), &ran))
        .collect();
    let made = device.work_counts();
    let firsts = &device.shared().textures.first_moves;
    let waiting = || {
        (buffers.iter())
            .filter(|buffer| firsts.fills_wait(buffer.at(0).0))
            .count()
    };
    let waited = waiting();
    // A command buffer whose commands reach none of them, completed.
    gpu_write(&queue, &other, 0x3c);
    let left = waiting();
    let after = device.work_counts();
    eprintln!(
        "MADE-COPY-IN on {name}: {MADE} two-copy buffers' copies in waiting {waited} at their \
         making, {left} after another buffer's command buffer; dma_buf_memories_live {} -> {} \
         -> {}, host_imports_live {} -> {} -> {}, host_imports_resident {} -> {} -> {}, \
         host_imports_released {} -> {} -> {}",
        before.dma_buf_memories_live,
        made.dma_buf_memories_live,
        after.dma_buf_memories_live,
        before.host_imports_live,
        made.host_imports_live,
        after.host_imports_live,
        before.host_imports_resident,
        made.host_imports_resident,
        after.host_imports_resident,
        before.host_imports_released,
        made.host_imports_released,
        after.host_imports_released,
    );
    assert_eq!(waited, MADE, "each making registers its copy in");
    assert_eq!(left, 0, "the next submission takes every making's copy in");
    for buffer in &buffers {
        assert_eq!(read(&device, &queue, buffer, 0..BYTES), pattern(BYTES));
    }
    drop(buffers);
    assert!(ran_to(&ran, MADE), "the deallocators did not run");
}

/// A 256 MiB two-copy buffer over pages no import maps holds 256 MiB of device-local memory and
/// staging within the bound, none at rest, across cycles (the module docs, **Bounded**).
#[test]
fn a_large_buffer_stages_within_the_bound() {
    let pages = Pages::new(LARGE);
    pages.write_pattern();
    let scratch = Scratch::new("managed-bounded");
    let Some(device) = two_copy_device(&scratch, "bounded", Route::Pages) else {
        return;
    };
    let queue = device.new_command_queue().unwrap();
    let before = device.work_counts();
    let ran = Ran::default();
    let buffer = no_copy(&device, &pages, Route::Pages, managed(), &ran);
    let made = device.work_counts();
    let staged = made.managed_staging_bytes - before.managed_staging_bytes;
    assert!(staged <= STAGED_MOST, "staged {staged} at the making");
    let held = made.managed_device_copy_bytes - before.managed_device_copy_bytes;
    assert!((LARGE..LARGE + (64 << 10)).contains(&held), "{held}");
    let tail = LARGE - BYTES..LARGE;
    assert_eq!(
        read(&device, &queue, &buffer, tail.clone()),
        pattern(LARGE)[tail.start as usize..]
    );
    for cycle in 1..=CYCLES {
        pages.write(0..LARGE, cycle);
        buffer.did_modify_range(0..LARGE).unwrap();
        let now = device.work_counts();
        let staged = now.managed_staging_bytes - before.managed_staging_bytes;
        assert!(staged <= STAGED_MOST, "cycle {cycle}: staged {staged}");
        assert_eq!(
            read(&device, &queue, &buffer, 0..BYTES),
            vec![cycle; BYTES as usize]
        );
        assert_eq!(
            read(&device, &queue, &buffer, tail.clone()),
            vec![cycle; BYTES as usize]
        );
        let rest = staged_down_to(&device, before.managed_staging_bytes);
        assert_eq!(
            rest.managed_staging_bytes, before.managed_staging_bytes,
            "cycle {cycle}: staging held at rest"
        );
        assert_eq!(
            rest.managed_device_copy_bytes - before.managed_device_copy_bytes,
            held
        );
        assert_eq!(
            rest.copied_host_buffers_made,
            before.copied_host_buffers_made
        );
        eprintln!(
            "MANAGED-COPY bounded {}: cycle {cycle}, staged {staged} bytes at most, {held} bytes \
             of device-local memory",
            device.name()
        );
    }
    drop(buffer);
    assert!(ran_to(&ran, 1));
}

/// A rebind copies the new pages in whole (the module docs, **Rebind**).
fn rebind(route: Route) {
    let (first, second) = (Pages::new(BYTES), Pages::new(BYTES));
    first.write(0..BYTES, 1);
    second.write_pattern();
    let scratch = Scratch::new("managed-rebind");
    let Some(device) = two_copy_device(&scratch, "rebind", route) else {
        return;
    };
    let queue = device.new_command_queue().unwrap();
    let (ran, ran_second) = (Ran::default(), Ran::default());
    let buffer = no_copy(&device, &first, route, managed(), &ran);
    let address = buffer.gpu_address();
    assert_eq!(
        read(&device, &queue, &buffer, 0..BYTES),
        vec![1; BYTES as usize]
    );
    // SAFETY: the pages outlive every buffer over them.
    unsafe { buffer.rebind_bytes_no_copy(second.memory(route), Some(counted(&ran_second)), None) }
        .unwrap();
    assert_eq!(buffer.contents(), Some(second.at));
    assert_eq!(buffer.gpu_address(), address);
    assert_eq!(buffer.host_backing(), Some(HostBacking::DeviceCopy));
    assert!(ran_to(&ran, 1), "{route:?}: the old pages' deallocator");
    assert_eq!(read(&device, &queue, &buffer, 0..BYTES), pattern(BYTES));
    // The GPU's writes reach the new pages.
    gpu_write(&queue, &buffer, 0x42);
    synchronize(&queue, &buffer);
    assert_eq!(second.bytes(), vec![0x42; BYTES as usize]);
    assert_eq!(first.bytes(), vec![1; BYTES as usize]);
    drop(buffer);
    assert!(ran_to(&ran_second, 1));
    assert_eq!(ran.lock().unwrap().len(), 1);
}

#[test]
fn a_rebind_through_an_import_copies_the_new_pages_in() {
    rebind(Route::Import);
}

#[test]
fn a_rebind_of_pages_no_import_maps_copies_them_in() {
    rebind(Route::Pages);
}

/// A detach keeps the device copy (the module docs, **Detach**).
fn detach(route: Route) {
    let (first, second) = (Pages::new(BYTES), Pages::new(BYTES));
    first.write_pattern();
    second.write(0..BYTES, 0x77);
    let scratch = Scratch::new("managed-detach");
    let Some(device) = two_copy_device(&scratch, "detach", route) else {
        return;
    };
    let queue = device.new_command_queue().unwrap();
    let ran = Ran::default();
    let buffer = no_copy(&device, &first, route, managed(), &ran);
    let address = buffer.gpu_address();
    assert_eq!(read(&device, &queue, &buffer, 0..BYTES), pattern(BYTES));
    assert_eq!(buffer.detach_backing(), Ok(true));
    assert_eq!(buffer.host_backing(), Some(HostBacking::Detached));
    assert_eq!(buffer.contents(), Some(first.at));
    assert_eq!(buffer.gpu_address(), address);
    // The GPU reads and writes the device copy.
    assert_eq!(read(&device, &queue, &buffer, 0..BYTES), pattern(BYTES));
    gpu_write(&queue, &buffer, 0x5a);
    assert_eq!(
        read(&device, &queue, &buffer, 0..BYTES),
        vec![0x5a; BYTES as usize]
    );
    // A synchronize copies nothing out: the buffer has no pages.
    let out = device.work_counts().managed_bytes_copied_out;
    synchronize(&queue, &buffer);
    assert_eq!(device.work_counts().managed_bytes_copied_out, out);
    assert_eq!(first.bytes(), pattern(BYTES));
    // A rebind gives it pages again, copied in whole.
    // SAFETY: the pages outlive every buffer over them.
    unsafe { buffer.rebind_bytes_no_copy(second.memory(route), None, None) }.unwrap();
    assert_eq!(buffer.host_backing(), Some(HostBacking::DeviceCopy));
    assert_eq!(buffer.contents(), Some(second.at));
    assert_eq!(
        read(&device, &queue, &buffer, 0..BYTES),
        vec![0x77; BYTES as usize]
    );
    assert!(ran_to(&ran, 1), "{route:?}: the first pages' deallocator");
}

#[test]
fn a_detach_through_an_import_keeps_the_device_copy() {
    detach(Route::Import);
}

#[test]
fn a_detach_of_pages_no_import_maps_keeps_the_device_copy() {
    detach(Route::Pages);
}

/// A rebind over pages the process maps read-only drops the GPU's writes (the module docs,
/// **Read-only pages**): a write to them would fault the process.
fn read_only(route: Route) {
    let (first, fixed) = (Pages::new(BYTES), Pages::new(BYTES));
    fixed.write(0..BYTES, 7);
    fixed.read_only();
    let scratch = Scratch::new("managed-read-only");
    let Some(device) = two_copy_device(&scratch, "read-only", route) else {
        return;
    };
    let queue = device.new_command_queue().unwrap();
    let ran = Ran::default();
    let buffer = no_copy(&device, &first, route, managed(), &ran);
    // SAFETY: the pages outlive every buffer over them.
    unsafe { buffer.rebind_bytes_no_copy(fixed.memory(route), None, None) }.unwrap();
    assert_eq!(buffer.contents(), Some(fixed.at));
    assert_eq!(
        read(&device, &queue, &buffer, 0..BYTES),
        vec![7; BYTES as usize]
    );
    let before = device.work_counts();
    // The GPU's write and a synchronize, in one command buffer: nothing reaches the pages.
    blit(&queue, |blit| {
        blit.fill_buffer(&buffer, 0..BYTES, 0x99).unwrap();
        blit.synchronize_resource(ResourceRef::Buffer(&buffer))
            .unwrap();
    });
    assert_eq!(fixed.bytes(), vec![7; BYTES as usize]);
    let after = device.work_counts();
    assert_eq!(
        after.managed_bytes_copied_out,
        before.managed_bytes_copied_out
    );
    assert!(after.managed_writes_undone > before.managed_writes_undone);
    // The next command buffer reads the pages' bytes, not the GPU's write.
    assert_eq!(
        read(&device, &queue, &buffer, 0..BYTES),
        vec![7; BYTES as usize],
        "{route:?}"
    );
    assert_eq!(first.bytes(), vec![0; BYTES as usize]);
}

#[test]
fn read_only_pages_through_an_import_drop_the_gpu_writes() {
    read_only(Route::Import);
}

#[test]
fn read_only_pages_no_import_maps_drop_the_gpu_writes() {
    read_only(Route::Pages);
}

/// A `Shared` no-copy buffer lies in its pages on every device: no device copy.
#[test]
fn a_shared_no_copy_buffer_has_no_device_copy() {
    let pages = Pages::new(BYTES);
    pages.write_pattern();
    let scratch = Scratch::new("managed-shared");
    let device = device(&scratch.0);
    let queue = device.new_command_queue().unwrap();
    let ran = Ran::default();
    let buffer = no_copy(
        &device,
        &pages,
        Route::Import,
        ResourceOptions::shared(),
        &ran,
    );
    assert_ne!(buffer.host_backing(), Some(HostBacking::DeviceCopy));
    assert_eq!(read(&device, &queue, &buffer, 0..BYTES), pattern(BYTES));
    // The GPU's write reaches the pages with no synchronize.
    gpu_write(&queue, &buffer, 0x21);
    assert_eq!(pages.bytes(), vec![0x21; BYTES as usize]);
    let counts = device.work_counts();
    assert_eq!(counts.managed_device_copies_made, 0);
    assert_eq!(counts.managed_bytes_copied_in, 0);
}

/// A `Managed` no-copy buffer by `route` made over exactly the bytes of a live `Shared` no-copy
/// buffer that is a copy of them (the caller's gate refused their import, `copied.rs`) is two
/// copies all the same: its device copy in device-local memory, its host copy the pages or their
/// import, and never the `Shared` buffer's copy, which the GPU reaches over the bus
/// (`host_aliases.rs`).
fn over_a_copy(route: Route) {
    let pages = Pages::new(BYTES);
    pages.write_pattern();
    let scratch = Scratch::new("managed-over-copy");
    let Some(device) = two_copy_device(&scratch, "over-copy", route) else {
        return;
    };
    let queue = device.new_command_queue().unwrap();
    let (copy_ran, ran) = (Ran::default(), Ran::default());
    let shared = ResourceOptions::shared();
    let copy = no_copy(&device, &pages, Route::Pages, shared, &copy_ran);
    assert_eq!(copy.host_backing(), Some(HostBacking::Copied));
    let before = device.work_counts();
    let buffer = no_copy(&device, &pages, route, managed(), &ran);
    let made = device.work_counts();
    eprintln!(
        "MANAGED-COPY over-copy {route:?} {}: {:?}, {} device copies, {} buffers in a copy",
        device.name(),
        buffer.host_backing(),
        made.managed_device_copies - before.managed_device_copies,
        made.copied_host_buffers_shared - before.copied_host_buffers_shared,
    );
    assert_eq!(
        buffer.host_backing(),
        Some(HostBacking::DeviceCopy),
        "{route:?}"
    );
    assert_eq!(made.managed_device_copies - before.managed_device_copies, 1);
    assert_eq!(
        made.copied_host_buffers_shared, before.copied_host_buffers_shared,
        "{route:?}: the buffer lies in no copy"
    );
    assert_eq!(
        made.copied_host_buffers_made,
        before.copied_host_buffers_made
    );
    assert_eq!(made.managed_device_copies_in_place, 0);
    assert_eq!(read(&device, &queue, &buffer, 0..BYTES), pattern(BYTES));
    // The GPU's write stays, and reaches the pages at a synchronize.
    gpu_write(&queue, &buffer, 0x5a);
    synchronize(&queue, &buffer);
    assert_eq!(pages.bytes(), vec![0x5a; BYTES as usize], "{route:?}");
    assert_eq!(
        device.work_counts().managed_writes_undone,
        before.managed_writes_undone,
        "{route:?}"
    );
    drop((buffer, copy));
    assert!(ran_to(&ran, 1) && ran_to(&copy_ran, 1), "{route:?}");
}

#[test]
fn a_managed_no_copy_buffer_over_a_copy_s_bytes_through_an_import_is_two_copies() {
    over_a_copy(Route::Import);
}

#[test]
fn a_managed_no_copy_buffer_over_a_copy_s_bytes_no_import_maps_is_two_copies() {
    over_a_copy(Route::Pages);
}

#[test]
fn a_managed_no_copy_buffer_over_a_copy_s_vouched_bytes_no_import_maps_is_two_copies() {
    over_a_copy(Route::Vouched);
}

/// A recovery carries a two-copy buffer: its host copy made again over the same pages and its
/// device copy on the new generation, the pages copied in; its pointer, Metal address and copies
/// work on as before, and its deallocator runs once, at its release.
fn recovery(route: Route) {
    let pages = Pages::new(BYTES);
    pages.write_pattern();
    let scratch = Scratch::new("managed-recovery");
    let Some(device) = two_copy_device(&scratch, "recovery", route) else {
        return;
    };
    let queue = device.new_command_queue().unwrap();
    let ran = Ran::default();
    let buffer = no_copy(&device, &pages, route, managed(), &ran);
    let address = buffer.gpu_address();
    assert_eq!(read(&device, &queue, &buffer, 0..BYTES), pattern(BYTES));
    device.recover_for_test().unwrap();
    assert!(buffer.device().generation_serves());
    assert_eq!(buffer.contents(), Some(pages.at));
    assert_eq!(buffer.gpu_address(), address);
    assert_eq!(buffer.host_backing(), Some(HostBacking::DeviceCopy));
    assert_eq!(read(&device, &queue, &buffer, 0..BYTES), pattern(BYTES));
    pages.write(0..BYTES, 0x11);
    buffer.did_modify_range(0..BYTES).unwrap();
    assert_eq!(
        read(&device, &queue, &buffer, 0..BYTES),
        vec![0x11; BYTES as usize]
    );
    gpu_write(&queue, &buffer, 0x22);
    synchronize(&queue, &buffer);
    assert_eq!(pages.bytes(), vec![0x22; BYTES as usize]);
    assert!(
        ran.lock().unwrap().is_empty(),
        "{route:?}: ran before the release"
    );
    drop(buffer);
    assert!(ran_to(&ran, 1), "{route:?}: the deallocator did not run");
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(*ran.lock().unwrap(), [(pages.at.as_ptr() as u64, BYTES)]);
}

#[test]
fn a_recovery_carries_a_two_copy_buffer_with_an_import() {
    recovery(Route::Import);
}

#[test]
fn a_recovery_carries_a_two_copy_buffer_over_pages_no_import_maps() {
    recovery(Route::Pages);
}

/// The side of the texture leg's `RGBA8Unorm` texture: its rows tight, 256 bytes each.
const SIDE: u64 = 64;

/// A texture over a two-copy buffer's bytes follows the device copy: its GPU reads take the bytes
/// copied in (at the making, by `did_modify_range`, by its own `replace_region`), and a
/// `synchronize_resource` of it copies the GPU's writes out to the pages.
fn texture(route: Route) {
    let texels = SIDE * SIDE * 4;
    let pages = Pages::new(BYTES);
    pages.write_pattern();
    let scratch = Scratch::new("managed-texture");
    let Some(device) = two_copy_device(&scratch, "texture", route) else {
        return;
    };
    let queue = device.new_command_queue().unwrap();
    let ran = Ran::default();
    let buffer = no_copy(&device, &pages, route, managed(), &ran);
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.resource_options = managed();
    let texture = buffer.new_texture(&descriptor, 0, SIDE * 4).unwrap();
    // The texture's texels as the GPU reads them.
    let gpu_texels = || {
        let target = device
            .new_buffer(texels, ResourceOptions::shared())
            .unwrap();
        blit(&queue, |blit| {
            blit.copy_from_texture_to_buffer(
                &texture,
                0,
                0,
                Origin::new(0, 0, 0),
                Size::new(SIDE, SIDE, 1),
                &target,
                0,
                SIDE * 4,
                texels,
                BlitOption::None,
            )
            .unwrap()
        });
        let contents = target.contents().unwrap();
        // SAFETY: the target's bytes, which no command writes now.
        unsafe { std::slice::from_raw_parts(contents.as_ptr(), texels as usize) }.to_vec()
    };
    assert_eq!(gpu_texels(), pattern(BYTES)[..texels as usize]);
    pages.write(0..texels, 0x33);
    buffer.did_modify_range(0..texels).unwrap();
    assert_eq!(gpu_texels(), vec![0x33; texels as usize], "{route:?}");
    let row = vec![0x44; (SIDE * 4) as usize];
    texture
        .replace_region(Region::new_2d(0, 0, SIDE, 1), 0, 0, &row, SIDE * 4, 0)
        .unwrap();
    let seen = gpu_texels();
    assert_eq!(seen[..row.len()], row, "{route:?}");
    assert!(seen[row.len()..].iter().all(|&byte| byte == 0x33));
    // The GPU's write reaches the pages at the texture's synchronize.
    blit(&queue, |blit| {
        blit.fill_buffer(&buffer, 0..texels, 0x55).unwrap();
        blit.synchronize_resource(ResourceRef::Texture(&texture))
            .unwrap();
    });
    assert_eq!(
        pages.bytes()[..texels as usize],
        vec![0x55; texels as usize]
    );
    let mut read_back = vec![0; texels as usize];
    texture
        .get_bytes(
            &mut read_back,
            SIDE * 4,
            0,
            Region::new_2d(0, 0, SIDE, SIDE),
            0,
            0,
        )
        .unwrap();
    assert_eq!(read_back, vec![0x55; texels as usize]);
}

#[test]
fn a_texture_over_a_two_copy_buffer_through_an_import_follows_its_device_copy() {
    texture(Route::Import);
}

#[test]
fn a_texture_over_a_two_copy_buffer_over_pages_no_import_maps_follows_its_device_copy() {
    texture(Route::Pages);
}

/// The rounds of each thread of the cross-queue leg.
const ROUNDS: u8 = 48;

/// The bytes of a two-copy buffer each round of the cross-queue leg writes and names.
const SPAN: u64 = 64 << 10;

/// A round of the cross-queue leg: the byte its host copy held at the commit, the `Shared` buffer
/// its command buffer copied the range into, and the command buffer.
type Round = (u8, Buffer, CommandBuffer);

/// A buffer of the cross-queue leg and its host copy's bytes, which the CPU writes.
struct HostCopied {
    buffer: Buffer,
    at: usize,
}

impl HostCopied {
    /// Set its first [`SPAN`] bytes to `byte` from the CPU.
    fn write(&self, byte: u8) {
        // SAFETY: the host copy holds at least `SPAN` bytes from `at` while the buffer lives (the
        // runtime's own host copy, or the leg's pages, unmapped only after every buffer went).
        unsafe { std::ptr::write_bytes(self.at as *mut u8, byte, SPAN as usize) };
    }
}

/// `ROUNDS` rounds on `queue`, the byte of each `salt` apart from the other thread's, over buffers
/// `make` makes, a new one every fourth round: write the range, name it, commit a copy of it,
/// then write it again unnamed, or, every fourth round, let the buffer go with its command buffer
/// pending.
fn rounds(
    device: &Device,
    queue: &CommandQueue,
    salt: u8,
    make: impl Fn(usize) -> HostCopied,
) -> Vec<Round> {
    let mut made: Option<HostCopied> = None;
    let mut rounds = Vec::new();
    for round in 0..ROUNDS {
        let copied = made.get_or_insert_with(|| make(usize::from(round / 4)));
        let byte = 1 + 2 * round + salt;
        copied.write(byte);
        copied.buffer.did_modify_range(0..SPAN).unwrap();
        let target = device.new_buffer(SPAN, ResourceOptions::shared()).unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_buffer(&copied.buffer, 0, &target, 0, SPAN)
            .unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        match round % 4 {
            3 => made = None,
            // Made after the commit: no command buffer committed before reads it.
            _ => copied.write(0xee),
        }
        rounds.push((byte, target, command_buffer));
    }
    rounds
}

/// `did_modify_range` across commits of two-copy buffers, the runtime's own host copy and the
/// caller's pages by `route`: on two queues, each on a thread of its own, where `two`, else one
/// after the other on one queue and thread. A commit takes the ranges of the buffers its command
/// buffer uses (`managed.rs`), and its copy in goes with a submission that uses the buffer, after
/// the command buffers before it on its queue: so each command buffer reads the bytes its host
/// copy held at its commit, none written after it, and none of a later round's, whose copy in
/// never lands under it. Every fourth round lets its buffer go with its command buffer pending.
/// No command buffer fails, the validation layer the device asks for reports nothing
/// (`layer_claim.rs`, which passes its whole-pool report of a host query reset), and the staging
/// buffers go once all completed. Printed (`MANAGED-ROUNDS`): the rounds that read other bytes.
fn named_rounds(route: Route, two: bool) {
    let generations = usize::from(ROUNDS / 4);
    let pages: Vec<Pages> = (0..generations).map(|_| Pages::new(BYTES)).collect();
    let addresses: Vec<usize> = pages
        .iter()
        .map(|pages| pages.at.as_ptr() as usize)
        .collect();
    let scratch = Scratch::new("managed-across-queues");
    let watched = Watched::default();
    let options = DeviceOptions {
        layers: watched.layers(),
        messages: Some(watched.sink(|_| {})),
        ..options(&scratch.0)
    };
    let device = Device::system_default_with(&options).unwrap();
    if device.properties().has_unified_memory {
        eprintln!(
            "MANAGED-ACROSS-QUEUES {}: the device has unified memory; one copy",
            device.name()
        );
        return;
    }
    let before = device.work_counts();
    let ran = Ran::default();
    let own = |_: usize| {
        let buffer = device.new_buffer(BYTES, managed()).unwrap();
        let at = buffer.contents().unwrap().as_ptr() as usize;
        HostCopied { buffer, at }
    };
    let callers = |generation: usize| {
        let at = addresses[generation];
        // SAFETY: the leg's writable pages, unmapped only after every buffer over them went.
        let memory = unsafe { memory_at(NonNull::new(at as *mut u8).unwrap(), route) };
        // SAFETY: as above.
        let buffer = unsafe {
            device.new_buffer_with_bytes_no_copy(memory, BYTES, managed(), Some(counted(&ran)))
        }
        .unwrap();
        assert_eq!(buffer.host_backing(), Some(HostBacking::DeviceCopy));
        HostCopied { buffer, at }
    };
    let (first, second) = match two {
        true => std::thread::scope(|scope| {
            let first =
                scope.spawn(|| rounds(&device, &device.new_command_queue().unwrap(), 0, own));
            let second =
                scope.spawn(|| rounds(&device, &device.new_command_queue().unwrap(), 1, callers));
            (first.join().unwrap(), second.join().unwrap())
        }),
        false => {
            let queue = device.new_command_queue().unwrap();
            let first = rounds(&device, &queue, 0, own);
            (first, rounds(&device, &queue, 1, callers))
        }
    };
    let mut wrong = Vec::new();
    for (thread, done) in [("own", first), ("caller's", second)] {
        for (byte, target, command_buffer) in done {
            assert!(
                command_buffer.wait_until_completed_for(WAIT).is_ok(),
                "{route:?} {thread}: a command buffer did not complete in {WAIT:?}"
            );
            assert_eq!(
                (command_buffer.status(), command_buffer.error()),
                (CommandBufferStatus::Completed, None),
                "{route:?} {thread}: round of byte {byte:#x}"
            );
            let contents = target.contents().unwrap();
            // SAFETY: the target's bytes, which no command writes now.
            let seen = unsafe { std::slice::from_raw_parts(contents.as_ptr(), SPAN as usize) };
            if !seen.iter().all(|&seen| seen == byte) {
                wrong.push(format!("{thread} {byte:#x} read {:#x}", seen[0]));
            }
        }
    }
    assert!(
        ran_to(&ran, generations),
        "{route:?}: a deallocator did not run"
    );
    let rest = staged_down_to(&device, before.managed_staging_bytes);
    assert_eq!(rest.managed_staging_bytes, before.managed_staging_bytes);
    eprintln!(
        "MANAGED-ROUNDS {} {route:?}, {} queue(s): {} rounds each of two buffers, ranges named \
         {}; rounds that read other bytes {} {wrong:?}",
        device.name(),
        1 + usize::from(two),
        ROUNDS,
        rest.managed_ranges_modified - before.managed_ranges_modified,
        wrong.len(),
    );
    assert!(
        wrong.is_empty(),
        "{route:?}: rounds read other bytes: {wrong:?}"
    );
    drop(device);
    drop(pages);
    watched.layer_reported_nothing(&format!("did_modify_range rounds {route:?}"));
}

#[test]
fn ranges_named_across_two_queues_reach_each_command_buffer_through_an_import() {
    named_rounds(Route::Import, true);
}

#[test]
fn ranges_named_across_two_queues_reach_each_command_buffer_over_pages_no_import_maps() {
    named_rounds(Route::Pages, true);
}

#[test]
fn ranges_named_on_one_queue_reach_each_command_buffer_through_an_import() {
    named_rounds(Route::Import, false);
}

/// On a device with unified memory, a `Managed` buffer's `did_modify_range` copies nothing (the
/// module docs, **Elsewhere, one copy**): the runtime's own, and one over the caller's pages.
#[test]
fn a_did_modify_range_on_unified_memory_copies_nothing() {
    let pages = Pages::new(BYTES);
    let scratch = Scratch::new("managed-unified");
    let device = device(&scratch.0);
    if !device.properties().has_unified_memory {
        eprintln!(
            "MANAGED-UNIFIED {}: no unified memory; two copies",
            device.name()
        );
        return;
    }
    let queue = device.new_command_queue().unwrap();
    let ran = Ran::default();
    let own = device.new_buffer(BYTES, managed()).unwrap();
    let caller = no_copy(&device, &pages, Route::Import, managed(), &ran);
    assert_ne!(caller.host_backing(), Some(HostBacking::DeviceCopy));
    let before = device.work_counts();
    for (byte, buffer) in [(0x31u8, &own), (0x32, &caller)] {
        let contents = buffer.contents().unwrap();
        // SAFETY: the buffer's bytes, which no command buffer uses now.
        unsafe { std::ptr::write_bytes(contents.as_ptr(), byte, BYTES as usize) };
        buffer.did_modify_range(0..BYTES).unwrap();
        assert_eq!(
            read(&device, &queue, buffer, 0..BYTES),
            vec![byte; BYTES as usize]
        );
    }
    let after = device.work_counts();
    eprintln!(
        "MANAGED-UNIFIED {}: copied in {} bytes, staged {} bytes, two-copy buffers made {}",
        device.name(),
        after.managed_bytes_copied_in - before.managed_bytes_copied_in,
        after.managed_staging_bytes,
        after.managed_device_copies_made,
    );
    assert_eq!(
        after.managed_bytes_copied_in,
        before.managed_bytes_copied_in
    );
    assert_eq!(after.managed_staging_bytes, 0);
    assert_eq!(after.managed_device_copies_made, 0);
}

/// The ranges of the copy-in bench (the module docs), each named whole before every commit.
const BENCH_RANGES: [u64; 3] = [256 << 10, 4 << 20, 32 << 20];

/// The commits of each leg of the copy-in bench.
const BENCH_COMMITS: usize = 40;

/// The middle of `values`, in milliseconds.
fn median_ms(mut values: Vec<Duration>) -> f64 {
    values.sort();
    values[values.len() / 2].as_secs_f64() * 1e3
}

/// The copy-in bench (the module docs).
#[test]
#[ignore = "a release-mode benchmark"]
fn managed_copy_in_bench() {
    let largest = BENCH_RANGES[BENCH_RANGES.len() - 1];
    let pages = Pages::new(largest);
    let scratch = Scratch::new("managed-copy-in");
    for exact in [false, true] {
        let mut options = options(&scratch.0);
        if exact {
            options.policy = metal2vulkan::device::DevicePolicy::exact_answers();
        }
        let device = Device::system_default_with(&options).unwrap();
        let policy = match exact {
            true => "exact answers",
            false => "default",
        };
        let queue = device.new_command_queue().unwrap();
        let ran = Ran::default();
        let own = device.new_buffer(largest, managed()).unwrap();
        let caller = no_copy(&device, &pages, Route::Import, managed(), &ran);
        let out = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
        for (name, buffer) in [("own", &own), ("caller's pages", &caller)] {
            let contents = buffer.contents().unwrap();
            for range in BENCH_RANGES {
                let before = device.work_counts();
                let (mut commits, mut whole) = (Vec::new(), Vec::new());
                for commit in 0..BENCH_COMMITS {
                    let byte = commit as u8 + 1;
                    // SAFETY: the buffer's host copy, which no command buffer reads now.
                    unsafe { std::ptr::write_bytes(contents.as_ptr(), byte, range as usize) };
                    let began = Instant::now();
                    buffer.did_modify_range(0..range).unwrap();
                    let command_buffer = queue.command_buffer().unwrap();
                    let mut blit = command_buffer.blit_command_encoder().unwrap();
                    blit.copy_from_buffer_to_buffer(buffer, range - 4096, &out, 0, 4096)
                        .unwrap();
                    blit.end_encoding().unwrap();
                    let committing = Instant::now();
                    command_buffer.commit().unwrap();
                    commits.push(committing.elapsed());
                    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
                    whole.push(began.elapsed());
                    assert_eq!(command_buffer.error(), None);
                    // SAFETY: the out buffer's bytes, which no command writes now.
                    let seen = unsafe { *out.contents().unwrap().as_ptr() };
                    assert_eq!(seen, byte, "{name} {range}: commit {commit}");
                }
                let after = device.work_counts();
                let (o, b) = (&after.object_churn, &before.object_churn);
                let (commit, wall) = (median_ms(commits.clone()), median_ms(whole));
                let most = commits.iter().max().unwrap().as_secs_f64() * 1e3;
                eprintln!(
                    "MANAGED-COPY-IN {} {policy}, {name}, {} KiB a commit: commit {commit:.3} ms \
                     (most {most:.3}), to completion {wall:.3} ms, {:.2} GB/s; copied in {} \
                     bytes, buffers parked {} taken again {} evicted {}",
                    device.name(),
                    range >> 10,
                    range as f64 / (wall * 1e6),
                    after.managed_bytes_copied_in - before.managed_bytes_copied_in,
                    o.buffers_parked - b.buffers_parked,
                    o.buffers_recycled - b.buffers_recycled,
                    o.buffers_evicted - b.buffers_evicted,
                );
            }
        }
        drop((own, caller, out, queue, device));
        assert!(ran_to(&ran, 1), "the deallocator did not run");
    }
}
