//! Guest RAM imported whole on a device (`host_ram.rs`): a no-copy object whose caller resolves
//! its pages to one run inside one chunk of a RAM the device uses is bound in that chunk's import.
//!
//! - **One run in a chunk.** The buffer is bound in the chunk, imported once, the first time an
//!   object over it is made: no import of its own, its gate not asked, its `contents` the
//!   pointer; the GPU reads the CPU's bytes and its fill reaches them. A second buffer over the
//!   same chunk imports nothing more. A no-copy heap takes the same path.
//! - **Containment.** The buffer's `VkBuffer` reaches no page past its own: a copy out of it
//!   reads no further than the window's end, a fill of all of it leaves the chunk's neighbouring
//!   pages as they were, and so does a kernel (the owned fixture's `scale_and_offset`) whose
//!   threads write three pages of words through a one-page buffer.
//! - **Other answers.** `None`, several runs and one run crossing a chunk's end take the gate's
//!   path (asked once, the pointer imported where it admits); an answer that breaks the
//!   resolver's contract is refused (`Invalid`, `Problem::HostRam`).
//! - **Rebinds.** A buffer's or heap's rebind asks the resolver once with its new window: one run
//!   in a chunk binds there, the gate unasked and nothing imported; several runs take the gate,
//!   and the copy where it refuses.
//! - **After pages the kernel cannot pin.** A no-copy buffer over a dma-buf's mapping
//!   (`VM_PFNMAP`, Linux, where `/dev/udmabuf` opens), made first, takes nothing from the device:
//!   the RAM's first chunk is imported after it, and buffers bound in it, on a driver whose failed
//!   import of such pages breaks its later ones
//!   (`RuntimeDriverDefect::UnpinnablePagesBreakHostMemory`, NVIDIA's) too, where the runtime
//!   refuses the pages before the driver sees them. There a buffer, a heap and a rebind over such
//!   pages with no dma-buf source are each a copy of them, whose bytes the GPU reads.
//! - **The use.** Dropping the use while a command buffer still holds a chunk frees the chunk
//!   after that command buffer completed, and the RAM's keep-alive goes after it; the same RAM
//!   used twice is refused, a RAM apart from it taken; a new RAM over the same ranges is taken once
//!   the first use ended.
//! - **Ahead of the objects.** The device's thread imports chunks of fresh RAM (advised for huge
//!   pages, as QEMU's RAM is: a `memfd` mapped shared, or private anonymous memory on a device
//!   that refuses pointers to shared memory) ahead of the objects: the RAM's first chunk once
//!   the use begins, and the window past it once a heap took it, so the consumer's 2 MiB heaps
//!   over consecutive runs, past a chunk's end, import nothing as they are made; a heap's bytes
//!   go from the CPU to the GPU and back; every heap gone, one made again imports nothing.
//! - **The window.** Before any take the thread imports the RAM's first chunk alone; a take
//!   inside the window opens the full window past it, and a take outside it (a jump) one chunk,
//!   after the chunk taken. It imports nothing past the window or the RAM's end, and never goes
//!   back for a chunk a jump skipped. A run of heaps that stays inside the window imports nothing
//!   on the thread making them; a heap outside it is bound in an import of its own run, and
//!   imports no chunk. An object over a chunk the thread imported, made while the chunk's state
//!   is looked at (as the thread looks at each chunk of its window), is bound in the chunk.
//! - **Down the RAM.** A heap over a chunk not imported, at the RAM's top, is bound in an import
//!   of its own run, whose bytes go from the CPU to the GPU and back, while the thread imports
//!   the chunk, in which the heaps over it after that are bound; a heap next to it below opens
//!   the window below, so a run of heaps down the RAM finds its chunks imported, and no making
//!   imports a chunk. The run's import goes with its heap.
//! - **Two imports of the same pages.** A buffer bound in a run's import and one bound in the
//!   chunk's over the same pages order by their host bytes, a write through either before a read
//!   through the other, in one blit encoder, in two encoders of one command buffer and across
//!   queues; so do a buffer bound in a run's import and a copy of its pages through the chunk's
//!   (`copy_from_host_ram` after a fill through the buffer, `copy_to_host_ram` before a copy out of
//!   it). Within the allocation limit's margin no run is imported alone. On a driver whose barriers
//!   do not order blits through two imports of the same pages
//!   (`RuntimeDriverDefect::AliasedImportBlitsUnordered`, MoltenVK's and ANV's) no run is imported
//!   alone either: an object over a chunk not imported imports the chunk, and the tests hold it to
//!   that.
//!
//! - **Shared RAM a device refuses.** On a device that refuses pointers to shared memory (RADV),
//!   a RAM of a `memfd` mapped shared with no dma-buf source imports no chunk, and an object over
//!   it is refused (`Unsupported`, `HostMemory`).
//!
//! Every other test holds the device's thread back (`Device::hold_host_ram_imports`), so that
//! each chunk is imported by the first object over it, as one the thread has not reached is.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device that imports no host memory says so and passes.

use super::dispatch_pending::state;
use crate::host_ram::AHEAD_CHUNKS;
use crate::{
    Buffer, CommandBufferStatus, Device, DeviceKind, DeviceOptions, Error, EventRef,
    HeapDescriptor, HeapType, HostBacking, HostMemory, HostRam, ImportGate, ImportRequest,
    KeepAlive, Problem, ResourceOptions, RunResolver, Size,
};
use metal2vulkan::error::{Feature, Unsupported};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
const MIB: u64 = 1 << 20;
/// The RAM's bytes: two whole chunks and a short third.
const RAM_BYTES: u64 = 128 * MIB + 256 * 1024;

/// Anonymous memory standing in for a guest's RAM, unmapped by the keep-alive it is given to.
struct Mapping {
    start: u64,
    length: u64,
    dropped: Arc<AtomicBool>,
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the mapping this test made, which no device imports any more: its keep-alive
        // is dropped after the last chunk over it is freed.
        unsafe { libc::munmap(self.start as *mut libc::c_void, self.length as usize) };
        self.dropped.store(true, Ordering::SeqCst);
    }
}

/// A RAM of `RAM_BYTES` in a pattern, its first byte, and whether its keep-alive was dropped.
fn new_ram() -> (Arc<HostRam>, u64, Arc<AtomicBool>) {
    // SAFETY: a fresh private anonymous mapping.
    let start = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            RAM_BYTES as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert_ne!(start, libc::MAP_FAILED);
    let start = start as u64;
    let dropped = Arc::new(AtomicBool::new(false));
    let keep: KeepAlive = Box::new(Mapping {
        start,
        length: RAM_BYTES,
        dropped: Arc::clone(&dropped),
    });
    let at = NonNull::new(start as *mut u8).unwrap();
    // SAFETY: the mapping is readable and writable until the keep-alive unmaps it.
    let ram = unsafe { HostRam::new(&[(at, RAM_BYTES)], Some(keep)) }.unwrap();
    (Arc::new(ram), start, dropped)
}

/// The byte the pattern holds at host address `address`.
fn pattern(address: u64) -> u8 {
    (address % 251) as u8
}

/// Write the pattern over `range` of host addresses, while no command buffer runs.
fn fill_pattern(range: Range<u64>) {
    for address in range {
        // SAFETY: bytes of the test's RAM, which no command buffer uses now.
        unsafe { *(address as *mut u8) = pattern(address) };
    }
}

/// The bytes of `range` of host addresses, while no command buffer runs.
fn bytes(range: Range<u64>) -> Vec<u8> {
    let length = (range.end - range.start) as usize;
    // SAFETY: as above.
    unsafe { std::slice::from_raw_parts(range.start as *const u8, length) }.to_vec()
}

/// A device of the cfg's kind whose uses of a RAM start no thread importing its chunks ahead
/// (`Device::hold_host_ram_imports`); `None` where it imports no host memory.
fn device() -> Option<Device> {
    let device = ahead_device()?;
    device.hold_host_ram_imports();
    Some(device)
}

/// A device of the cfg's kind, as the product makes it; `None` where it imports no host memory.
pub(super) fn ahead_device() -> Option<Device> {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    if device.shared().host_import.is_err() {
        eprintln!("HOST-RAM: the device imports no host memory here; nothing to test");
        return None;
    }
    // Asked now, by an import of one page released at once, so that no test's counts or
    // allocation limit meet that import ([`fresh_ram`] reads the answer).
    device.imports_shared_host_pointers().unwrap();
    Some(device)
}

/// A resolver answering `answer` of each window it is asked, each window into `asked`.
pub(super) fn resolver(
    asked: &Arc<Mutex<Vec<(u64, u64)>>>,
    answer: impl Fn(u64, u64) -> Option<Vec<Range<u64>>> + Send + Sync + 'static,
) -> RunResolver {
    let asked = Arc::clone(asked);
    Arc::new(move |start, length| {
        asked.lock().unwrap().push((start, length));
        answer(start, length)
    })
}

/// The pages are their own run of the RAM (the test's views are the RAM itself).
pub(super) fn itself(start: u64, length: u64) -> Option<Vec<Range<u64>>> {
    Some(std::iter::once(start..start + length).collect())
}

/// A gate answering `answer`, each request into `asked`.
pub(super) fn gate(asked: &Arc<Mutex<Vec<ImportRequest>>>, answer: bool) -> ImportGate {
    let asked = Arc::clone(asked);
    Arc::new(move |request| {
        asked.lock().unwrap().push(request);
        answer
    })
}

/// A no-copy buffer of `length` bytes at host address `at`, resolved by `runs`, under `gate`.
fn buffer(
    device: &Device,
    at: u64,
    length: u64,
    runs: RunResolver,
    gate: ImportGate,
) -> Result<Buffer, Error> {
    // SAFETY: the RAM's bytes, mapped until its keep-alive goes, after every device let go.
    let memory = unsafe { HostMemory::new(NonNull::new(at as *mut u8).unwrap()) }
        .with_runs(runs)
        .with_import_gate(gate);
    // SAFETY: as above.
    unsafe {
        device.new_buffer_with_bytes_no_copy(memory, length, ResourceOptions::default(), None)
    }
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

/// The buffer's first `length` bytes as the GPU reads them.
fn gpu_bytes(device: &Device, buffer: &Buffer, length: u64) -> Vec<u8> {
    let own = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    blit(device, |encoder| {
        encoder
            .copy_from_buffer_to_buffer(buffer, 0, &own, 0, length)
            .unwrap();
    });
    let contents = own.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) }.to_vec()
}

/// Wait until `flag` is set, a release on a handler thread having run.
fn settles(flag: &AtomicBool) -> bool {
    let started = Instant::now();
    while !flag.load(Ordering::SeqCst) {
        if started.elapsed() > Duration::from_secs(20) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    true
}

/// Fresh RAM of `length` bytes, never touched, in 2 MiB pages where the system gives them
/// ([`RamPages::Huge`]), unmapped by the keep-alive it is given to, of the kind `device` imports: a `memfd` mapped shared, as QEMU's
/// `memory-backend-memfd,share=on` RAM is (anonymous shared memory off Linux), on a device that
/// imports pointers to shared memory (`Device::imports_shared_host_pointers`), and private
/// anonymous memory, as QEMU's `memory-backend-ram` RAM is, on one that refuses them (RADV:
/// [`a_ram_of_shared_memory_a_device_refuses_imports_no_chunk_and_refuses_its_objects`] checks
/// shared RAM there); its first byte, and whether its keep-alive was dropped.
pub(super) fn fresh_ram(device: &Device, length: u64) -> (Arc<HostRam>, u64, Arc<AtomicBool>) {
    fresh_ram_in(device, length, RamPages::Huge)
}

/// The pages a fresh RAM lies in on Linux ([`fresh_ram_in`]); elsewhere the system's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RamPages {
    /// 2 MiB pages: the mapping starts at a multiple of 2 MiB and is advised for huge pages, as
    /// QEMU aligns and advises its file-backed RAM, so the system backs it with them where its
    /// shared memory takes the advice (`shmem_enabled` of `advise` or `always`).
    Huge,
    /// 4 KiB pages: the mapping is advised against huge pages.
    Small,
}

/// [`fresh_ram`]'s RAM in `pages`.
pub(super) fn fresh_ram_in(
    device: &Device,
    length: u64,
    pages: RamPages,
) -> (Arc<HostRam>, u64, Arc<AtomicBool>) {
    let shared = device.imports_shared_host_pointers().unwrap();
    if !shared {
        eprintln!(
            "HOST-RAM {}: the device refuses pointers to shared memory; the fresh RAM is private \
             anonymous memory",
            device.name()
        );
    }
    fresh_ram_of(length, shared, pages)
}

/// Fresh RAM of `length` bytes, as [`fresh_ram`]'s: shared memory where `shared`, else private
/// anonymous memory, in `pages`.
fn fresh_ram_of(
    length: u64,
    shared: bool,
    pages: RamPages,
) -> (Arc<HostRam>, u64, Arc<AtomicBool>) {
    #[cfg(target_os = "linux")]
    let (flags, fd) = match shared {
        true => {
            // SAFETY: a name and flags the call takes.
            let fd = unsafe { libc::memfd_create(c"m2v-fresh-ram".as_ptr(), libc::MFD_CLOEXEC) };
            assert!(fd >= 0);
            // SAFETY: the descriptor was just made, and is closed below.
            assert_eq!(unsafe { libc::ftruncate(fd, length as libc::off_t) }, 0);
            (libc::MAP_SHARED, fd)
        }
        false => (libc::MAP_PRIVATE | libc::MAP_ANON, -1),
    };
    #[cfg(not(target_os = "linux"))]
    let (flags, fd) = match shared {
        true => (libc::MAP_SHARED | libc::MAP_ANON, -1),
        false => (libc::MAP_PRIVATE | libc::MAP_ANON, -1),
    };
    // The mapping at a multiple of 2 MiB: a reservation 2 MiB longer, the mapping placed over it
    // at its first such multiple, and the reservation's ends unmapped.
    const ALIGN: usize = 2 << 20;
    let reserved_length = length as usize + ALIGN;
    // SAFETY: a fresh reservation of address space, with no access.
    let reserved = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            reserved_length,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE,
            -1,
            0,
        )
    };
    assert_ne!(reserved, libc::MAP_FAILED);
    let head = (reserved as usize).next_multiple_of(ALIGN) - reserved as usize;
    // SAFETY: a fresh mapping of the file's first `length` bytes (or of anonymous memory), over
    // the test's own reservation, inside it.
    let start = unsafe {
        libc::mmap(
            reserved.cast::<u8>().add(head).cast(),
            length as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            flags | libc::MAP_FIXED,
            fd,
            0,
        )
    };
    assert_ne!(start, libc::MAP_FAILED);
    // SAFETY: the reservation's parts before and past the mapping, which nothing uses.
    unsafe {
        if head > 0 {
            libc::munmap(reserved, head);
        }
        let tail = reserved_length - head - length as usize;
        if tail > 0 {
            libc::munmap(start.cast::<u8>().add(length as usize).cast(), tail);
        }
    }
    if fd >= 0 {
        // SAFETY: the test's descriptor; the mapping keeps the file.
        unsafe { libc::close(fd) };
    }
    #[cfg(target_os = "linux")]
    // SAFETY: advice on the test's own mapping; a kernel without huge pages refuses it, harmlessly.
    unsafe {
        let advice = match pages {
            RamPages::Huge => libc::MADV_HUGEPAGE,
            RamPages::Small => libc::MADV_NOHUGEPAGE,
        };
        libc::madvise(start, length as usize, advice);
    }
    #[cfg(not(target_os = "linux"))]
    let _ = pages;
    let start = start as u64;
    let dropped = Arc::new(AtomicBool::new(false));
    let keep: KeepAlive = Box::new(Mapping {
        start,
        length,
        dropped: Arc::clone(&dropped),
    });
    let at = NonNull::new(start as *mut u8).unwrap();
    // SAFETY: the mapping is readable and writable until the keep-alive unmaps it.
    let ram = unsafe { HostRam::new(&[(at, length)], Some(keep)) }.unwrap();
    (Arc::new(ram), start, dropped)
}

#[test]
fn chunks_are_imported_ahead_so_no_making_waits_for_one() {
    let Some(device) = ahead_device() else {
        return;
    };
    // A RAM of the start's one chunk and a full window past it: the device's thread imports all
    // of it once the first heap took the first chunk.
    const CHUNKS: u64 = 1 + AHEAD_CHUNKS as u64;
    let (ram, start, dropped) = fresh_ram(&device, CHUNKS * 64 * MIB);
    let before = device.work_counts();
    let ahead_of = |counts: &crate::WorkCounts| {
        counts.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead
    };
    let began = Instant::now();
    let used = device.use_host_ram(&ram).unwrap();
    // The consumer's heaps: 2 MiB each over consecutive runs from the RAM's start, past the
    // first chunk's end (heap 32 starts the second).
    const HEAP: u64 = 2 * MIB;
    const HEAPS: u64 = 40;
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let heap_over = |at: u64| {
        // SAFETY: the RAM's bytes, mapped until its keep-alive goes, after every device let go.
        let memory = unsafe { HostMemory::new(NonNull::new(at as *mut u8).unwrap()) }
            .with_runs(resolver(&runs, itself))
            .with_import_gate(gate(&gates, true));
        let descriptor = HeapDescriptor::new(HeapType::Placement, HEAP, ResourceOptions::default());
        // SAFETY: as above.
        unsafe { device.new_heap_with_bytes_no_copy(memory, &descriptor, None, None) }.unwrap()
    };
    // The device's thread imports the RAM's first chunk, which no object asked for; the first
    // heap takes it, which opens the window over the rest.
    let first = counts_until(&device, |counts| ahead_of(counts) == 1);
    assert_eq!(
        ahead_of(&first),
        1,
        "the device's thread imported the RAM's first chunk"
    );
    let mut slowest = Duration::ZERO;
    let making = Instant::now();
    let mut heaps = vec![heap_over(start)];
    slowest = slowest.max(making.elapsed());
    let ahead = counts_until(&device, |counts| ahead_of(counts) == CHUNKS);
    let imported_in = began.elapsed();
    assert_eq!(
        ahead_of(&ahead),
        CHUNKS,
        "the device's thread imported {} of {CHUNKS} chunks",
        ahead_of(&ahead)
    );
    assert_eq!(
        ahead.host_ram_chunks_imported - before.host_ram_chunks_imported,
        CHUNKS,
        "the first heap imported nothing"
    );
    assert_eq!(
        ahead.host_ram_chunks_live - before.host_ram_chunks_live,
        CHUNKS
    );
    heaps.extend((1..HEAPS).map(|index| {
        let making = Instant::now();
        let heap = heap_over(start + index * HEAP);
        slowest = slowest.max(making.elapsed());
        heap
    }));
    let made = device.work_counts();
    assert_eq!(
        made.host_ram_chunks_imported, ahead.host_ram_chunks_imported,
        "no making imported a chunk"
    );
    assert_eq!(
        made.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
        HEAPS
    );
    assert_eq!(
        made.host_pointer_imports_made - ahead.host_pointer_imports_made,
        0
    );
    assert!(gates.lock().unwrap().is_empty(), "the gate was not asked");
    // A heap in the second chunk: the CPU's bytes reach the GPU, and the GPU's fill the CPU.
    let page = device.shared().host_import.as_ref().unwrap().page;
    let at = start + 33 * HEAP;
    fill_pattern(at..at + 4 * page);
    let placed = heaps[33]
        .new_buffer_at_offset(4 * page, ResourceOptions::default(), 0)
        .unwrap();
    assert_eq!(
        gpu_bytes(&device, &placed, 4 * page),
        bytes(at..at + 4 * page)
    );
    blit(&device, |encoder| {
        encoder.fill_buffer(&placed, page..2 * page, 0xA5).unwrap();
    });
    let now = bytes(at..at + 4 * page);
    let pattern_of = |range: Range<u64>| range.map(pattern).collect::<Vec<_>>();
    assert_eq!(now[..page as usize], pattern_of(at..at + page)[..]);
    assert!(now[page as usize..2 * page as usize]
        .iter()
        .all(|&byte| byte == 0xA5));
    assert_eq!(
        now[2 * page as usize..],
        pattern_of(at + 2 * page..at + 4 * page)[..]
    );
    // A chunk is not imported twice: every heap gone, one made again imports nothing.
    drop((placed, heaps));
    let again = heap_over(start);
    assert_eq!(
        device.work_counts().host_ram_chunks_imported,
        made.host_ram_chunks_imported
    );
    eprintln!(
        "HOST-RAM: {CHUNKS} chunks of fresh RAM imported ahead in {imported_in:?}; {HEAPS} heap \
         makings, the slowest {slowest:?}"
    );
    drop((again, used, ram));
    assert!(
        settles(&dropped),
        "the keep-alive goes after the last chunk"
    );
}

/// Wait until `done` holds of the device's counts, at most [`WAIT`]: the counts then.
fn counts_until(device: &Device, done: impl Fn(&crate::WorkCounts) -> bool) -> crate::WorkCounts {
    let started = Instant::now();
    loop {
        let counts = device.work_counts();
        if done(&counts) || started.elapsed() > WAIT {
            return counts;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn the_ahead_thread_leaves_the_allocation_limit_to_the_objects() {
    let Some(device) = ahead_device() else {
        return;
    };
    let allocations = &device.shared().allocations.allocations;
    // A device of a small `maxMemoryAllocationCount`: the smallest limit whose margin (a
    // quarter) leaves the thread room for `AHEAD` chunks of the RAM's twelve, and no more: the
    // start's one, and three of the window an object over the first opens, short of its end.
    const CHUNKS: u64 = 12;
    const AHEAD: u32 = 4;
    const _: () = assert!(AHEAD as usize <= AHEAD_CHUNKS);
    let device_limit = allocations.limit();
    let live = allocations.live();
    let margin = |limit: u32| (limit / 4).max(1);
    let limit = (live + 1..)
        .find(|&limit| limit - margin(limit) - live == AHEAD)
        .unwrap();
    allocations.limit_to(limit);
    assert_eq!(allocations.limit(), limit);
    let (ram, start, dropped) = fresh_ram(&device, CHUNKS * 64 * MIB);
    let before = device.work_counts();
    let used = device.use_host_ram(&ram).unwrap();
    let ahead = |counts: &crate::WorkCounts| {
        counts.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead
    };
    let page = device.shared().host_import.as_ref().unwrap().page;
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    // An object over the first chunk, once the thread imported it, opens the window past it.
    counts_until(&device, |counts| ahead(counts) == 1);
    let opener = buffer(
        &device,
        start,
        4 * page,
        resolver(&runs, itself),
        gate(&gates, true),
    )
    .expect("an object over the first chunk is made");
    counts_until(&device, |counts| ahead(counts) == u64::from(AHEAD));
    // Stopped at the margin: nothing more comes.
    std::thread::sleep(Duration::from_millis(200));
    let stopped = device.work_counts();
    assert_eq!(
        ahead(&stopped),
        u64::from(AHEAD),
        "it stopped at the margin"
    );
    assert_eq!(allocations.live(), limit - margin(limit));
    // An object over a chunk past them imports it as it is made: an allocation the margin left
    // room for, which brings the device within it, so the thread's chunks no object took are
    // given back.
    let past = start + u64::from(AHEAD) * 64 * MIB;
    let object = buffer(
        &device,
        past,
        4 * page,
        resolver(&runs, itself),
        gate(&gates, true),
    )
    .expect("an object past the thread's chunks is made");
    let given = |counts: &crate::WorkCounts| {
        counts.host_ram_chunks_given_back - before.host_ram_chunks_given_back
    };
    // The first chunk, which the opener took, is not given back.
    let untaken = u64::from(AHEAD) - 1;
    let back = counts_until(&device, |counts| {
        given(counts) == untaken && counts.host_ram_chunks_live - before.host_ram_chunks_live == 2
    });
    assert_eq!(given(&back), untaken, "the untaken chunks went back");
    assert_eq!(
        back.host_ram_chunks_imported - before.host_ram_chunks_imported,
        u64::from(AHEAD) + 1
    );
    assert_eq!(
        back.host_ram_chunks_live - before.host_ram_chunks_live,
        2,
        "the objects' chunks alone stay"
    );
    assert_eq!(
        back.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
        2
    );
    // The objects have the rest of the limit: buffers of blocks of their own are made until the
    // device holds its limit, and the next is refused without a call of the driver, none of the
    // thread's chunks among the live allocations then.
    let room = limit - allocations.live();
    let dedicated = crate::memory::BLOCK_BYTES / 2 + page;
    let mut made = Vec::new();
    let refused = loop {
        match device.new_buffer(dedicated, ResourceOptions::private()) {
            Ok(buffer) => made.push(buffer),
            Err(error) => break error,
        }
        assert!(allocations.live() <= limit, "the limit holds");
        assert!(made.len() <= room as usize + 64, "the limit holds");
    };
    assert!(!made.is_empty(), "the room given back is the objects'");
    assert!(allocations.live() <= limit);
    let full = device.work_counts();
    assert!(
        full.memory_allocations_refused_at_limit > before.memory_allocations_refused_at_limit,
        "refused at the limit: {refused:?}"
    );
    assert_eq!(
        full.host_ram_chunks_live - before.host_ram_chunks_live,
        2,
        "no chunk of the thread's no object took was live at the refusal"
    );
    eprintln!(
        "HOST-RAM: maxMemoryAllocationCount {}; a limit of {limit} with {live} live: {AHEAD} \
         chunks imported ahead, {untaken} given back as an object came within the margin of {}, \
         then {} buffers of blocks of their own made of a room of {room}, the next refused \
         ({refused:?})",
        device_limit,
        margin(limit),
        made.len()
    );
    drop((made, object, opener, used, ram));
    assert!(
        settles(&dropped),
        "the keep-alive goes after the last chunk"
    );
}

/// The counts once `done` holds of them and a while has passed with the device's thread free to
/// go on, so that a thread importing past where it should shows.
fn settled(device: &Device, done: impl Fn(&crate::WorkCounts) -> bool) -> crate::WorkCounts {
    let reached = counts_until(device, done);
    std::thread::sleep(Duration::from_millis(300));
    let later = device.work_counts();
    assert_eq!(
        later.host_ram_chunks_imported_ahead, reached.host_ram_chunks_imported_ahead,
        "the device's thread went on past the window"
    );
    later
}

/// 1 where the device binds an object over a chunk not imported yet in an import of its own run, 0
/// where its barriers do not order blits through two imports of the same pages
/// (`RuntimeDriverDefect::AliasedImportBlitsUnordered`, MoltenVK's and ANV's) and the object
/// imports the chunk, as one that finds the device's thread stopped does.
pub(super) fn runs_alone(device: &Device) -> u64 {
    let defect = metal2vulkan::error::RuntimeDriverDefect::AliasedImportBlitsUnordered;
    u64::from(!device.properties().driver_defects.contains(&defect))
}

#[test]
fn the_window_ahead_moves_with_the_objects_and_imports_no_more() {
    let Some(device) = ahead_device() else {
        return;
    };
    const WINDOW: u64 = AHEAD_CHUNKS as u64;
    // The start's chunk, two full windows, a chunk skipped, a jump's one chunk, and a full
    // window past it.
    const CHUNKS: u64 = 3 * WINDOW + 4;
    let (ram, start, dropped) = fresh_ram(&device, CHUNKS * 64 * MIB);
    let before = device.work_counts();
    let used = device.use_host_ram(&ram).unwrap();
    let ahead = |counts: &crate::WorkCounts| {
        counts.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead
    };
    let by_objects = |counts: &crate::WorkCounts| {
        counts.host_ram_chunks_imported - before.host_ram_chunks_imported - ahead(counts)
    };
    let live =
        |counts: &crate::WorkCounts| counts.host_ram_chunks_live - before.host_ram_chunks_live;
    let alone =
        |counts: &crate::WorkCounts| counts.run_bound_buffers_made - before.run_bound_buffers_made;
    // The consumer's heaps: 2 MiB each over consecutive runs.
    const HEAP: u64 = 2 * MIB;
    const PER_CHUNK: u64 = 64 * MIB / HEAP;
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let heap_over = |at: u64| {
        // SAFETY: the RAM's bytes, mapped until its keep-alive goes, after every device let go.
        let memory = unsafe { HostMemory::new(NonNull::new(at as *mut u8).unwrap()) }
            .with_runs(resolver(&runs, itself))
            .with_import_gate(gate(&gates, true));
        let descriptor = HeapDescriptor::new(HeapType::Placement, HEAP, ResourceOptions::default());
        // SAFETY: as above.
        unsafe { device.new_heap_with_bytes_no_copy(memory, &descriptor, None, None) }.unwrap()
    };
    let chunk = |index: u64| start + index * 64 * MIB;
    let run = |chunks: Range<u64>| -> Vec<_> {
        let heaps = chunks.start * PER_CHUNK..chunks.end * PER_CHUNK;
        heaps.map(|index| heap_over(start + index * HEAP)).collect()
    };
    // Before any take: the RAM's first chunk, and none past it.
    let first = settled(&device, |counts| ahead(counts) == 1);
    assert_eq!(ahead(&first), 1, "one chunk from the RAM's start");
    assert_eq!(live(&first), 1, "no chunk past it");
    // A heap over it takes it inside the window: the full window past it.
    let mut heaps = vec![heap_over(chunk(0))];
    let opened = settled(&device, |counts| ahead(counts) == 1 + WINDOW);
    assert_eq!(ahead(&opened), 1 + WINDOW, "a full window past the take");
    assert_eq!(live(&opened), 1 + WINDOW, "no chunk past it");
    // A run over the window's chunks, at once: nothing imported by the makings, and the window
    // past the run's last chunk.
    heaps.extend(run(1..1 + WINDOW));
    assert_eq!(
        by_objects(&device.work_counts()),
        0,
        "a run inside the window imports nothing itself"
    );
    let moved = settled(&device, |counts| ahead(counts) == 1 + 2 * WINDOW);
    assert_eq!(ahead(&moved), 1 + 2 * WINDOW, "the window past the run");
    assert_eq!(live(&moved), 1 + 2 * WINDOW, "no chunk past it");
    // A jump over one chunk, outside the window: the heap is bound in an import of its own run,
    // and the thread imports its chunk and one chunk past it; on a device with
    // `AliasedImportBlitsUnordered` the heap imports its chunk, and the thread the one past it.
    let a = runs_alone(&device);
    let jump = 2 * WINDOW + 2;
    heaps.push(heap_over(chunk(jump)));
    let jumped = settled(&device, |counts| ahead(counts) == 2 + a + 2 * WINDOW);
    assert_eq!(
        (by_objects(&jumped), alone(&jumped)),
        (1 - a, a),
        "the heap outside the window was bound in its own run's import, or imported its chunk"
    );
    assert_eq!(
        ahead(&jumped),
        2 + a + 2 * WINDOW,
        "the chunk a jump took, and one past it"
    );
    assert_eq!(live(&jumped), 3 + 2 * WINDOW);
    // A heap over that chunk goes on from the jump: the full window past it, to the RAM's end.
    heaps.push(heap_over(chunk(jump + 1)));
    let again = settled(&device, |counts| ahead(counts) == 2 + a + 3 * WINDOW);
    assert_eq!(
        ahead(&again),
        2 + a + 3 * WINDOW,
        "a full window past the take"
    );
    assert_eq!((by_objects(&again), alone(&again)), (1 - a, a));
    // The run to the RAM's last chunk: nothing past the end, and nothing back for the chunk the
    // jump skipped.
    heaps.extend(run(jump + 2..CHUNKS));
    let end = settled(&device, |counts| ahead(counts) == 2 + a + 3 * WINDOW);
    assert_eq!(
        (by_objects(&end), alone(&end)),
        (1 - a, a),
        "the run to the end imported nothing itself"
    );
    assert_eq!(live(&end), CHUNKS - 1, "the skipped chunk is not imported");
    assert!(gates.lock().unwrap().is_empty(), "the gate was not asked");
    eprintln!(
        "HOST-RAM: a window of up to {WINDOW} chunks over {CHUNKS}: {} imported ahead, {} by the \
         objects, {} heaps in their own run's import, after {} heap makings",
        ahead(&end),
        by_objects(&end),
        alone(&end),
        heaps.len()
    );
    drop((heaps, used, ram));
    assert!(
        settles(&dropped),
        "the keep-alive goes after the last chunk"
    );
}

/// On a device that refuses pointers to shared memory (`Device::imports_shared_host_pointers`
/// false: RADV, whose amdgpu userptr takes private anonymous pages alone), a RAM of shared memory
/// (a `memfd` mapped shared, as QEMU's `memory-backend-memfd,share=on` RAM is) imports no chunk,
/// ahead or for an object, and an object over it is refused as its own import of the pointer is
/// (`Unsupported`, `HostMemory`), the RAM given no dma-buf source: the chunk's import by its
/// pointer is refused (`WorkCounts::host_ram_pointer_imports_failed`) and none goes as a dma-buf
/// (`host_ram_dma_buf_imports`). The use still ends, the RAM's keep-alive going after it. A device
/// that imports such pointers says so and passes.
#[test]
fn a_ram_of_shared_memory_a_device_refuses_imports_no_chunk_and_refuses_its_objects() {
    let Some(device) = ahead_device() else {
        return;
    };
    if device.imports_shared_host_pointers().unwrap() {
        eprintln!(
            "HOST-RAM {}: the device imports pointers to shared memory; nothing to test",
            device.name()
        );
        return;
    }
    let (ram, start, dropped) = fresh_ram_of(2 * 64 * MIB, true, RamPages::Huge);
    let before = device.work_counts();
    let used = device.use_host_ram(&ram).unwrap();
    let page = device.shared().host_import.as_ref().unwrap().page;
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let made = buffer(
        &device,
        start,
        4 * page,
        resolver(&runs, itself),
        gate(&gates, true),
    );
    assert_eq!(
        made.err(),
        Some(Error::Unsupported(Unsupported::not_representable(
            Feature::HostMemory
        ))),
        "an object over the shared RAM"
    );
    // The device's thread met the same refusal for the RAM's first chunk, or meets it now.
    std::thread::sleep(Duration::from_millis(300));
    let after = device.work_counts();
    assert_eq!(
        after.host_ram_chunks_imported - before.host_ram_chunks_imported,
        0,
        "no chunk of the shared RAM imported"
    );
    assert_eq!(after.host_ram_chunks_live - before.host_ram_chunks_live, 0);
    let failed = after.host_ram_pointer_imports_failed - before.host_ram_pointer_imports_failed;
    let dma_bufs = after.host_ram_dma_buf_imports - before.host_ram_dma_buf_imports;
    eprintln!(
        "HOST-RAM {}: shared RAM, no source: {failed} pointer import(s) refused, {dma_bufs} \
         dma-buf import(s)",
        device.name()
    );
    assert!(failed >= 1, "the chunk's pointer import was refused");
    assert_eq!(dma_bufs, 0, "no dma-buf without a source");
    drop(used);
    drop(ram);
    assert!(settles(&dropped), "the RAM's keep-alive dropped");
}

/// An object over a chunk the device's thread imported, made while the chunk's state is looked at
/// as the thread looks at each chunk of its window as it passes them
/// (`Device::looking_at_host_ram_chunk`), waits for the look and is bound in the chunk: the look
/// is not an import under way, so the object imports nothing of its own. The look lasts past the
/// object's making's start; where the making reached the chunk after it ended, it binds there
/// too.
#[test]
fn an_object_made_while_its_imported_chunk_is_looked_at_is_bound_in_the_chunk() {
    let Some(device) = ahead_device() else {
        return;
    };
    let (ram, start, dropped) = fresh_ram(&device, 2 * 64 * MIB);
    let before = device.work_counts();
    let used = device.use_host_ram(&ram).unwrap();
    let ahead = |counts: &crate::WorkCounts| {
        counts.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead
    };
    let first = counts_until(&device, |counts| ahead(counts) == 1);
    assert_eq!(ahead(&first), 1, "the RAM's first chunk, imported ahead");
    let page = device.shared().host_import.as_ref().unwrap().page;
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let made = std::thread::scope(|scope| {
        let making = device.looking_at_host_ram_chunk(start, || {
            let making = scope.spawn(|| {
                buffer(
                    &device,
                    start,
                    4 * page,
                    resolver(&runs, itself),
                    gate(&gates, true),
                )
            });
            // Long past the making's start, so that it reaches the chunk while it is looked at.
            std::thread::sleep(Duration::from_millis(200));
            making
        });
        making.join().unwrap()
    });
    let object = made.expect("the buffer over the chunk is made");
    let after = device.work_counts();
    assert_eq!(
        (
            after.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
            after.run_bound_buffers_made - before.run_bound_buffers_made,
        ),
        (1, 0),
        "the object made while its chunk was looked at is bound in the chunk, in no run's import"
    );
    assert!(gates.lock().unwrap().is_empty(), "the gate was not asked");
    drop((object, used, ram));
    assert!(
        settles(&dropped),
        "the keep-alive goes after the last chunk"
    );
}

#[test]
fn a_run_of_heaps_down_the_ram_finds_its_chunks_imported_and_none_waits_for_one() {
    let Some(device) = ahead_device() else {
        return;
    };
    const WINDOW: u64 = AHEAD_CHUNKS as u64;
    // The start's chunk, a full window below the top's, and one more.
    const CHUNKS: u64 = 2 * WINDOW + 2;
    let (ram, start, dropped) = fresh_ram(&device, CHUNKS * 64 * MIB);
    let before = device.work_counts();
    let used = device.use_host_ram(&ram).unwrap();
    let ahead = |counts: &crate::WorkCounts| {
        counts.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead
    };
    let by_objects = |counts: &crate::WorkCounts| {
        counts.host_ram_chunks_imported - before.host_ram_chunks_imported - ahead(counts)
    };
    let alone =
        |counts: &crate::WorkCounts| counts.run_bound_buffers_made - before.run_bound_buffers_made;
    let bound = |counts: &crate::WorkCounts| {
        counts.chunk_bound_buffers_made - before.chunk_bound_buffers_made
    };
    // The consumer's heaps: 2 MiB each, from the RAM's top down, each the one below the last.
    const HEAP: u64 = 2 * MIB;
    const PER_CHUNK: u64 = 64 * MIB / HEAP;
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let heap_over = |index: u64| {
        let at = start + index * HEAP;
        // SAFETY: the RAM's bytes, mapped until its keep-alive goes, after every device let go.
        let memory = unsafe { HostMemory::new(NonNull::new(at as *mut u8).unwrap()) }
            .with_runs(resolver(&runs, itself))
            .with_import_gate(gate(&gates, true));
        let descriptor = HeapDescriptor::new(HeapType::Placement, HEAP, ResourceOptions::default());
        // SAFETY: as above.
        unsafe { device.new_heap_with_bytes_no_copy(memory, &descriptor, None, None) }.unwrap()
    };
    let top = CHUNKS * PER_CHUNK - 1;
    let first = settled(&device, |counts| ahead(counts) == 1);
    assert_eq!(ahead(&first), 1, "one chunk from the RAM's start");
    // The top's heap, over a chunk outside the window: bound in an import of its own run, while
    // the thread imports its chunk, and nothing past the RAM's end. On a device with
    // `AliasedImportBlitsUnordered` the heap imports the chunk itself, and waits for it.
    let a = runs_alone(&device);
    let mut heaps = vec![heap_over(top)];
    let topped = settled(&device, |counts| ahead(counts) == 1 + a);
    assert_eq!(
        (by_objects(&topped), alone(&topped), bound(&topped)),
        (1 - a, a, 1 - a),
        "the heap over a chunk not imported was bound in its own run's import, or its chunk's"
    );
    assert_eq!(
        ahead(&topped),
        1 + a,
        "the thread imported the chunk it took"
    );
    // Its bytes: the CPU's reach the GPU, and the GPU's fill the CPU's.
    let page = device.shared().host_import.as_ref().unwrap().page;
    let at = start + top * HEAP;
    fill_pattern(at..at + 4 * page);
    let placed = heaps[0]
        .new_buffer_at_offset(4 * page, ResourceOptions::default(), 0)
        .unwrap();
    assert_eq!(
        gpu_bytes(&device, &placed, 4 * page),
        bytes(at..at + 4 * page)
    );
    blit(&device, |encoder| {
        encoder.fill_buffer(&placed, page..2 * page, 0xA5).unwrap();
    });
    let now = bytes(at..at + 4 * page);
    let pattern_of = |range: Range<u64>| range.map(pattern).collect::<Vec<_>>();
    assert_eq!(now[..page as usize], pattern_of(at..at + page)[..]);
    assert!(now[page as usize..2 * page as usize]
        .iter()
        .all(|&byte| byte == 0xA5));
    assert_eq!(
        now[2 * page as usize..],
        pattern_of(at + 2 * page..at + 4 * page)[..]
    );
    drop(placed);
    // The rest of the top chunk's heaps: bound in its import.
    heaps.extend((1..PER_CHUNK).map(|below| heap_over(top - below)));
    assert_eq!(
        (alone(&device.work_counts()), bound(&device.work_counts())),
        (a, PER_CHUNK - a)
    );
    // The next heap down, next to the chunk last taken: its own run again, the chunk not
    // imported, and the full window below.
    heaps.push(heap_over(top - PER_CHUNK));
    let opened = settled(&device, |counts| ahead(counts) == 1 + 2 * a + WINDOW);
    assert_eq!(
        ahead(&opened),
        1 + 2 * a + WINDOW,
        "the chunk taken and the full window below it"
    );
    assert_eq!(alone(&opened), 2 * a);
    // The run down the window's chunks: every heap bound in a chunk imported ahead of it, and
    // the window below its last, down to the RAM's start.
    heaps.extend((PER_CHUNK + 1..(1 + WINDOW) * PER_CHUNK).map(|below| heap_over(top - below)));
    let end = settled(&device, |counts| ahead(counts) == CHUNKS - 2 + 2 * a);
    assert_eq!(
        ahead(&end),
        CHUNKS - 2 + 2 * a,
        "the window went down with the heaps"
    );
    assert_eq!(
        (by_objects(&end), alone(&end)),
        (2 - 2 * a, 2 * a),
        "the run down imported nothing itself"
    );
    assert_eq!(bound(&end), heaps.len() as u64 - 2 * a);
    assert!(gates.lock().unwrap().is_empty(), "the gate was not asked");
    // The runs' imports live while their heaps do, and go with them.
    let runs_live = |counts: &crate::WorkCounts| counts.run_imports_live - before.run_imports_live;
    assert_eq!(
        runs_live(&end),
        2 * a,
        "each heap alone holds its run's import"
    );
    eprintln!(
        "HOST-RAM: down the RAM: {} chunks imported ahead, {} heaps in their own run's import, {} \
         in a chunk",
        ahead(&end),
        alone(&end),
        bound(&end)
    );
    drop(heaps);
    let gone = counts_until(&device, |counts| runs_live(counts) == 0);
    assert_eq!(runs_live(&gone), 0, "no run's import outlives its heap");
    drop((used, ram));
    assert!(
        settles(&dropped),
        "the keep-alive goes after the last chunk"
    );
}

#[test]
fn an_object_within_the_allocation_margin_imports_its_chunk_and_no_run() {
    let Some(device) = ahead_device() else {
        return;
    };
    let allocations = &device.shared().allocations.allocations;
    let (ram, start, dropped) = fresh_ram(&device, 4 * 64 * MIB);
    let before = device.work_counts();
    let used = device.use_host_ram(&ram).unwrap();
    let ahead = |counts: &crate::WorkCounts| {
        counts.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead
    };
    // The thread imported the RAM's first chunk and waits for a take.
    settled(&device, |counts| ahead(counts) == 1);
    // The device comes within the margin of its limit while the thread waits: a limit that
    // leaves room for an allocation or more, and none ahead (`Allocations::room_ahead`).
    let device_limit = allocations.limit();
    let live = allocations.live();
    let margin = |limit: u32| (limit / 4).max(1);
    let room = (0..)
        .take_while(|&room| margin(live + 1 + room) > room)
        .last();
    let limit = live + 1 + room.unwrap();
    allocations.limit_to(limit);
    assert!(!allocations.room_ahead(), "within the margin");
    // An object over a chunk not imported: its run is not imported alone (one more allocation
    // within the margin); it imports the chunk, and is made. (On a device with
    // `AliasedImportBlitsUnordered` no run is imported alone anywhere: there this holds of
    // that.)
    let page = device.shared().host_import.as_ref().unwrap().page;
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let object = buffer(
        &device,
        start + 2 * 64 * MIB,
        4 * page,
        resolver(&runs, itself),
        gate(&gates, true),
    )
    .expect("an object within the margin is made");
    let made = device.work_counts();
    let by_objects = made.host_ram_chunks_imported - before.host_ram_chunks_imported - ahead(&made);
    let alone = made.run_bound_buffers_made - before.run_bound_buffers_made;
    assert_eq!(
        (by_objects, alone),
        (1, 0),
        "within the margin the object imports its chunk"
    );
    assert_eq!(
        made.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
        1
    );
    allocations.limit_to(device_limit);
    eprintln!(
        "HOST-RAM: a limit of {limit} with {live} live: an object over a chunk not imported \
         imported the chunk"
    );
    drop((object, used, ram));
    assert!(
        settles(&dropped),
        "the keep-alive goes after the last chunk"
    );
}

/// Contents of a shared buffer, while no command buffer runs.
fn contents_of(buffer: &Buffer, length: u64) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) }.to_vec()
}

#[test]
fn a_run_bound_and_a_chunk_bound_buffer_over_the_same_pages_order_by_their_host_bytes() {
    let Some(device) = ahead_device() else {
        return;
    };
    let (ram, start, dropped) = fresh_ram(&device, 4 * 64 * MIB);
    let before = device.work_counts();
    let used = device.use_host_ram(&ram).unwrap();
    let ahead = |counts: &crate::WorkCounts| {
        counts.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead
    };
    settled(&device, |counts| ahead(counts) == 1);
    let page = device.shared().host_import.as_ref().unwrap().page;
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let over = |length: u64| {
        let at = start + 2 * 64 * MIB + 8 * page;
        buffer(
            &device,
            at,
            length,
            resolver(&runs, itself),
            gate(&gates, true),
        )
        .unwrap()
    };
    // Four pages over a chunk not imported: bound in an import of their run; then, the chunk
    // imported (and the one past it), eight from the same page: bound in the chunk. On a device
    // whose barriers do not order blits through two imports of the same pages
    // (`AliasedImportBlitsUnordered`) the first imports the chunk, and both lie in it.
    let a = runs_alone(&device);
    let alone = over(4 * page);
    settled(&device, |counts| ahead(counts) == 2 + a);
    let in_chunk = over(8 * page);
    let made = device.work_counts();
    let by_objects = made.host_ram_chunks_imported - before.host_ram_chunks_imported - ahead(&made);
    assert_eq!(
        (
            made.run_bound_buffers_made - before.run_bound_buffers_made,
            made.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
            by_objects
        ),
        (a, 2 - a, 1 - a),
        "one buffer in its run's import and one in the chunk's, or both in the chunk the first \
         imported"
    );
    let out = device.new_buffer(page, ResourceOptions::shared()).unwrap();
    // The write's work across queues before its fill: 2 GiB of fills of memory of the device's own.
    const LONG_BYTES: u64 = 64 * MIB;
    const LONG_FILLS: u64 = 32;
    let long = (device.new_buffer(LONG_BYTES, ResourceOptions::private())).unwrap();
    let queues = [
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    ];
    let encode =
        |command_buffer: &crate::CommandBuffer, writer: &Buffer, reader: &Buffer, value| {
            let mut encoder = command_buffer.blit_command_encoder().unwrap();
            encoder.fill_buffer(writer, page..2 * page, value).unwrap();
            encoder.end_encoding().unwrap();
            let mut encoder = command_buffer.blit_command_encoder().unwrap();
            (encoder.copy_from_buffer_to_buffer(reader, page, &out, 0, page)).unwrap();
            encoder.end_encoding().unwrap();
        };
    let mut value = 0;
    for (writer, reader, way) in [
        (&alone, &in_chunk, "run to chunk"),
        (&in_chunk, &alone, "chunk to run"),
    ] {
        // One command buffer, one blit encoder: the copy out of the bytes the fill wrote, through
        // the other buffer, waits for the fill.
        value += 1;
        let barriers = device.work_counts().command_barriers;
        let command_buffer = queues[0].command_buffer().unwrap();
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        encoder.fill_buffer(writer, page..2 * page, value).unwrap();
        (encoder.copy_from_buffer_to_buffer(reader, page, &out, 0, page)).unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert!(
            device.work_counts().command_barriers > barriers,
            "{way}: the copy waited for the fill"
        );
        assert!(
            contents_of(&out, page).iter().all(|&byte| byte == value),
            "{way}: the copy in one encoder read the fill"
        );
        // Two encoders of one command buffer.
        value += 1;
        let command_buffer = queues[0].command_buffer().unwrap();
        encode(&command_buffer, writer, reader, value);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert!(
            contents_of(&out, page).iter().all(|&byte| byte == value),
            "{way}: the second encoder read the first's fill"
        );
        // Across queues, in commit order (Metal's for tracked resources): the write's command
        // buffer fills memory of the device's own first, for long enough that a read not ordered
        // after it would copy the bytes before its fill; the read's, on the other queue,
        // committed after it, copies once the write ran.
        value += 1;
        let writing = queues[0].command_buffer().unwrap();
        let mut encoder = writing.blit_command_encoder().unwrap();
        for _ in 0..LONG_FILLS {
            encoder.fill_buffer(&long, 0..LONG_BYTES, value).unwrap();
        }
        encoder.fill_buffer(writer, page..2 * page, value).unwrap();
        encoder.end_encoding().unwrap();
        writing.commit().unwrap();
        let reading = queues[1].command_buffer().unwrap();
        let mut encoder = reading.blit_command_encoder().unwrap();
        (encoder.copy_from_buffer_to_buffer(reader, page, &out, 0, page)).unwrap();
        encoder.end_encoding().unwrap();
        reading.commit().unwrap();
        writing.wait_until_completed_for(WAIT).unwrap();
        reading.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(reading.status(), CommandBufferStatus::Completed);
        assert!(
            contents_of(&out, page).iter().all(|&byte| byte == value),
            "{way}: the read on the other queue read the write"
        );
    }
    eprintln!(
        "HOST-RAM: {} over the same pages: ordered both ways in one encoder, two encoders and \
         across queues",
        match a {
            1 => "a buffer in its run's import and one in the chunk's",
            _ => "two buffers in the chunk's import (AliasedImportBlitsUnordered)",
        }
    );
    drop((alone, in_chunk, out, long, used, ram));
    assert!(
        settles(&dropped),
        "the keep-alive goes after the last chunk"
    );
}

#[test]
fn a_run_bound_buffer_and_a_copy_of_its_pages_order_by_their_host_bytes() {
    let Some(device) = ahead_device() else {
        return;
    };
    let (ram, start, dropped) = fresh_ram(&device, 4 * 64 * MIB);
    let before = device.work_counts();
    let used = device.use_host_ram(&ram).unwrap();
    let ahead = |counts: &crate::WorkCounts| {
        counts.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead
    };
    let by_objects = |counts: &crate::WorkCounts| {
        counts.host_ram_chunks_imported - before.host_ram_chunks_imported - ahead(counts)
    };
    settled(&device, |counts| ahead(counts) == 1);
    let page = device.shared().host_import.as_ref().unwrap().page;
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    // Four pages over a chunk not imported: bound in an import of their run, while the thread
    // imports the chunk (and the one past it), whose import the copies use. On a device whose
    // barriers do not order blits through two imports of the same pages
    // (`AliasedImportBlitsUnordered`) the buffer imports the chunk, and lies in the copies' import.
    let a = runs_alone(&device);
    let at = start + 2 * 64 * MIB + 8 * page;
    let alone = buffer(
        &device,
        at,
        4 * page,
        resolver(&runs, itself),
        gate(&gates, true),
    )
    .unwrap();
    let made = settled(&device, |counts| ahead(counts) == 2 + a);
    assert_eq!(
        (
            made.run_bound_buffers_made - before.run_bound_buffers_made,
            made.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
            by_objects(&made),
            made.run_imports_live - before.run_imports_live,
        ),
        (a, 1 - a, 1 - a, a),
        "the buffer in its run's import, or in the chunk it imported"
    );
    // The copies' run: the buffer's second page.
    let run = at + page..at + 2 * page;
    let pages = std::slice::from_ref(&run);
    let out = device.new_buffer(page, ResourceOptions::shared()).unwrap();
    let source = device.new_buffer(page, ResourceOptions::shared()).unwrap();
    // The write's work across queues before it: 2 GiB of fills of memory of the device's own.
    const LONG_BYTES: u64 = 64 * MIB;
    const LONG_FILLS: u64 = 32;
    let long = (device.new_buffer(LONG_BYTES, ResourceOptions::private())).unwrap();
    let queues = [
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    ];
    // The write: a fill through the buffer, or a scatter of `source` holding `value` (written by
    // the CPU while no command buffer uses it); the read: a gather of the pages into `out`, or a
    // copy out of the buffer into it.
    let write = |encoder: &mut crate::BlitCommandEncoder, scatter: bool, value: u8| match scatter {
        false => encoder.fill_buffer(&alone, page..2 * page, value).unwrap(),
        true => {
            // SAFETY: the buffer's bytes, which no command buffer uses now.
            unsafe { std::ptr::write_bytes(source.contents().unwrap().as_ptr(), value, page as _) };
            encoder.copy_to_host_ram(&source, 0, &ram, pages).unwrap();
        }
    };
    let read = |encoder: &mut crate::BlitCommandEncoder, scatter: bool| match scatter {
        false => encoder.copy_from_host_ram(&ram, pages, &out, 0).unwrap(),
        true => (encoder.copy_from_buffer_to_buffer(&alone, page, &out, 0, page)).unwrap(),
    };
    let mut value = 0;
    for (scatter, way) in [
        (false, "the buffer's fill to a gather"),
        (true, "a scatter to the buffer's copy"),
    ] {
        // One command buffer, one blit encoder: the read of the bytes the write wrote waits for
        // it.
        value += 1;
        let barriers = device.work_counts().command_barriers;
        let command_buffer = queues[0].command_buffer().unwrap();
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        write(&mut encoder, scatter, value);
        read(&mut encoder, scatter);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert!(
            device.work_counts().command_barriers > barriers,
            "{way}: the read waited for the write"
        );
        assert!(
            contents_of(&out, page).iter().all(|&byte| byte == value),
            "{way}: the read in one encoder read the write"
        );
        // Two encoders of one command buffer.
        value += 1;
        let command_buffer = queues[0].command_buffer().unwrap();
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        write(&mut encoder, scatter, value);
        encoder.end_encoding().unwrap();
        let mut encoder = command_buffer.blit_command_encoder().unwrap();
        read(&mut encoder, scatter);
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert!(
            contents_of(&out, page).iter().all(|&byte| byte == value),
            "{way}: the second encoder read the first's write"
        );
        // Across queues, in commit order: the write's command buffer fills memory of the device's
        // own first, for long enough that a read not ordered after it would read the bytes before
        // the write; the read's, on the other queue, committed after it, reads once it ran.
        value += 1;
        let writing = queues[0].command_buffer().unwrap();
        let mut encoder = writing.blit_command_encoder().unwrap();
        for _ in 0..LONG_FILLS {
            encoder.fill_buffer(&long, 0..LONG_BYTES, value).unwrap();
        }
        write(&mut encoder, scatter, value);
        encoder.end_encoding().unwrap();
        writing.commit().unwrap();
        let reading = queues[1].command_buffer().unwrap();
        let mut encoder = reading.blit_command_encoder().unwrap();
        read(&mut encoder, scatter);
        encoder.end_encoding().unwrap();
        reading.commit().unwrap();
        writing.wait_until_completed_for(WAIT).unwrap();
        reading.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(reading.status(), CommandBufferStatus::Completed);
        assert!(
            contents_of(&out, page).iter().all(|&byte| byte == value),
            "{way}: the read on the other queue read the write"
        );
    }
    let copied = device.work_counts();
    assert_eq!(
        (
            copied.host_ram_copies - before.host_ram_copies,
            by_objects(&copied),
            copied.run_imports_live - before.run_imports_live,
        ),
        (6, 1 - a, a),
        "six copies over the chunk's import, which none of them imported"
    );
    eprintln!(
        "HOST-RAM: {} and copies of its pages: ordered both ways in one encoder, two encoders \
         and across queues",
        match a {
            1 => "a buffer in its run's import",
            _ => "a buffer in the chunk's import (AliasedImportBlitsUnordered)",
        }
    );
    drop((alone, out, source, long, used, ram));
    assert!(
        settles(&dropped),
        "the keep-alive goes after the last chunk"
    );
}

#[test]
fn one_run_inside_a_chunk_is_bound_there_with_no_import_of_its_own() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start, dropped) = new_ram();
    let used = device.use_host_ram(&ram).unwrap();
    let page = device.shared().host_import.as_ref().unwrap().page;
    // Three pages a page into the second chunk, the pages around them in the pattern too.
    let at = start + 64 * MIB + page;
    let length = 3 * page;
    fill_pattern(at - page..at + length + page);
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let before = device.work_counts();
    let first = buffer(
        &device,
        at,
        length,
        resolver(&runs, itself),
        gate(&gates, true),
    )
    .unwrap();
    let after = device.work_counts();
    assert_eq!(
        *runs.lock().unwrap(),
        vec![(at, length)],
        "asked once, the window"
    );
    assert!(gates.lock().unwrap().is_empty(), "the gate was not asked");
    assert_eq!(
        after.host_ram_chunks_imported - before.host_ram_chunks_imported,
        1
    );
    assert_eq!(
        after.host_ram_chunk_bytes - before.host_ram_chunk_bytes,
        64 * MIB
    );
    assert_eq!(
        after.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
        1
    );
    assert_eq!(
        after.host_pointer_imports_made - before.host_pointer_imports_made,
        0,
        "no import of its own"
    );
    assert_eq!(
        first.contents(),
        NonNull::new(at as *mut u8),
        "contents are the pointer"
    );
    assert_eq!(first.host_backing(), Some(HostBacking::InHostRam));
    // The GPU reads the CPU's bytes, and a fill reaches them.
    assert_eq!(gpu_bytes(&device, &first, length), bytes(at..at + length));
    blit(&device, |encoder| {
        encoder.fill_buffer(&first, page..2 * page, 0x5A).unwrap();
    });
    let now = bytes(at..at + length);
    assert!(now[page as usize..2 * page as usize]
        .iter()
        .all(|&b| b == 0x5A));
    assert_eq!(now[..page as usize], bytes(at..at + page)[..]);
    // A second buffer over another run of the same chunk imports nothing more.
    let other = at + 8 * page;
    fill_pattern(other..other + page);
    let before = device.work_counts();
    let second = buffer(
        &device,
        other,
        page,
        resolver(&runs, itself),
        gate(&gates, true),
    )
    .unwrap();
    let after = device.work_counts();
    assert_eq!(
        after.host_ram_chunks_imported - before.host_ram_chunks_imported,
        0
    );
    assert_eq!(
        after.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
        1
    );
    assert_eq!(
        gpu_bytes(&device, &second, page),
        bytes(other..other + page)
    );
    eprintln!(
        "HOST-RAM: chunks imported {}, chunk-bound buffers {}",
        after.host_ram_chunks_imported, after.chunk_bound_buffers_made
    );
    drop((first, second, used, ram));
    assert!(
        settles(&dropped),
        "the keep-alive goes after the last chunk"
    );
}

#[test]
fn a_chunk_bound_buffer_reaches_no_page_past_its_own() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start, dropped) = new_ram();
    let used = device.use_host_ram(&ram).unwrap();
    let page = device.shared().host_import.as_ref().unwrap().page;
    // A buffer of a page and a half, a few pages into the chunk: its window is two pages, and
    // the pages around them hold a sentinel.
    let at = start + 4 * page;
    let length = page + page / 2;
    let sentinel = 0xC3u8;
    for address in at - page..at + 3 * page {
        // SAFETY: bytes of the test's RAM, which no command buffer uses now.
        unsafe { *(address as *mut u8) = sentinel };
    }
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let buffer = buffer(
        &device,
        at,
        length,
        resolver(&runs, itself),
        gate(&gates, true),
    )
    .unwrap();
    assert_eq!(
        device.work_counts().chunk_bound_buffers_made,
        1,
        "bound in the chunk"
    );
    // A copy out of it reads no further than its window's end.
    assert!(
        buffer.copy_reach() <= 2 * page,
        "reach {} past the window",
        buffer.copy_reach()
    );
    // A fill of all of it changes its bytes alone.
    blit(&device, |encoder| {
        encoder.fill_buffer(&buffer, 0..length, 0x11).unwrap();
    });
    let around = bytes(at - page..at + 3 * page);
    let (before, rest) = around.split_at(page as usize);
    let (inside, after) = rest.split_at(length as usize);
    assert!(
        before.iter().all(|&b| b == sentinel),
        "the page before stays"
    );
    assert!(
        inside.iter().all(|&b| b == 0x11),
        "the buffer's bytes are filled"
    );
    assert!(after.iter().all(|&b| b == sentinel), "the bytes after stay");
    drop((buffer, used, ram));
    assert!(settles(&dropped));
}

/// Dispatch the owned fixture's `scale_and_offset` over `words` threads, writing `output`, from
/// an input of `0, 1, 2, ...` and a scale of 2, and wait for it.
fn scale_and_offset(device: &Device, output: &Buffer, words: u64) {
    let input: Vec<u8> = (0..words).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let input = device
        .new_buffer_with_bytes(&input, ResourceOptions::shared())
        .unwrap();
    let scale = device
        .new_buffer_with_bytes(&2.0f32.to_le_bytes(), ResourceOptions::shared())
        .unwrap();
    let state = state(device);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_buffer(Some(&input), 0, 0).unwrap();
    encoder.set_buffer(Some(output), 0, 1).unwrap();
    encoder.set_buffer(Some(&scale), 0, 2).unwrap();
    encoder
        .dispatch_threads(Size::new(words, 1, 1), Size::new(64, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// A kernel whose threads write three pages of words through a one-page buffer bound in a chunk
/// changes none of the chunk's pages around it, wherever the device keeps such writes through a
/// buffer of its own out of the next buffer in the same memory: the chunk binding is held to the
/// device's own containment. Where the device does not (MoltenVK on the M3 Max: its out-of-bounds
/// writes reach other bytes of the buffer's allocation, `docs/FAST-PATHS.md`, known gaps), that
/// is said and the chunk's pages are not held to more.
#[test]
fn a_kernel_writing_past_a_chunk_bound_buffer_reaches_no_neighbouring_page() {
    let Some(device) = device() else {
        return;
    };
    let page = device.shared().host_import.as_ref().unwrap().page;
    let words = 3 * page / 4;
    let sentinel = 0xC3u8;
    // The device's own containment: a one-page buffer of its own and the buffer made after it.
    let own = device.new_buffer(page, ResourceOptions::shared()).unwrap();
    let next = device
        .new_buffer(4 * page, ResourceOptions::shared())
        .unwrap();
    let next_bytes = next.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command buffer uses now.
    unsafe { std::ptr::write_bytes(next_bytes.as_ptr(), sentinel, 4 * page as usize) };
    scale_and_offset(&device, &own, words);
    // SAFETY: as above.
    let next_now = unsafe { std::slice::from_raw_parts(next_bytes.as_ptr(), 4 * page as usize) };
    let reached = next_now.iter().filter(|&&b| b != sentinel).count();
    let contains = reached == 0;
    // A one-page output four pages into the chunk, the pages around it a sentinel.
    let (ram, start, dropped) = new_ram();
    let used = device.use_host_ram(&ram).unwrap();
    let at = start + 4 * page;
    for address in at - page..at + 4 * page {
        // SAFETY: bytes of the test's RAM, which no command buffer uses now.
        unsafe { *(address as *mut u8) = sentinel };
    }
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let output = buffer(
        &device,
        at,
        page,
        resolver(&runs, itself),
        gate(&gates, true),
    )
    .unwrap();
    assert_eq!(
        device.work_counts().chunk_bound_buffers_made,
        1,
        "bound in the chunk"
    );
    scale_and_offset(&device, &output, words);
    let around = bytes(at - page..at + 4 * page);
    let (before, rest) = around.split_at(page as usize);
    let (inside, after) = rest.split_at(page as usize);
    assert!(
        before.iter().all(|&b| b == sentinel),
        "the page before stays"
    );
    let first = f32::from_le_bytes(inside[4..8].try_into().unwrap());
    assert_eq!(
        first, 3.0,
        "the kernel wrote the buffer (word 1 is 1 * 2 + 1)"
    );
    let past = after.iter().filter(|&&b| b != sentinel).count();
    eprintln!(
        "HOST-RAM: out-of-bounds writes changed {reached} bytes of a buffer of the device's own \
         after one, and {past} bytes of the chunk's pages after a chunk-bound one"
    );
    if contains {
        assert_eq!(past, 0, "no write reached the pages after the buffer");
    }
    drop((output, used, ram));
    assert!(settles(&dropped));
}

#[test]
fn a_no_copy_heap_over_one_run_is_bound_in_the_chunk() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start, dropped) = new_ram();
    let used = device.use_host_ram(&ram).unwrap();
    let page = device.shared().host_import.as_ref().unwrap().page;
    let at = start + 16 * page;
    let length = 16 * page;
    fill_pattern(at..at + length);
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    // SAFETY: as `buffer`'s.
    let memory = unsafe { HostMemory::new(NonNull::new(at as *mut u8).unwrap()) }
        .with_runs(resolver(&runs, itself))
        .with_import_gate(gate(&gates, true));
    let descriptor = crate::HeapDescriptor::new(
        crate::HeapType::Placement,
        length,
        ResourceOptions::default(),
    );
    let before = device.work_counts();
    // SAFETY: as above.
    let heap =
        unsafe { device.new_heap_with_bytes_no_copy(memory, &descriptor, None, None) }.unwrap();
    let after = device.work_counts();
    assert_eq!(
        after.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
        1
    );
    assert_eq!(
        after.host_pointer_imports_made - before.host_pointer_imports_made,
        0
    );
    assert!(gates.lock().unwrap().is_empty(), "the gate was not asked");
    // A buffer placed at a page reads the RAM's bytes there.
    let placed = heap
        .new_buffer_at_offset(4 * page, ResourceOptions::default(), page)
        .unwrap();
    assert_eq!(
        gpu_bytes(&device, &placed, 4 * page),
        bytes(at + page..at + 5 * page)
    );
    drop((placed, heap, used, ram));
    assert!(settles(&dropped));
}

#[test]
fn other_answers_take_the_gates_path_and_broken_ones_are_refused() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start, dropped) = new_ram();
    let used = device.use_host_ram(&ram).unwrap();
    let page = device.shared().host_import.as_ref().unwrap().page;
    let runs: Arc<Mutex<Vec<(u64, u64)>>> = Arc::default();
    let gates: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    let path = |at: u64, length: u64, answer: RunResolver| {
        fill_pattern(at..at + length);
        gates.lock().unwrap().clear();
        let before = device.work_counts();
        let made = buffer(&device, at, length, answer, gate(&gates, true)).unwrap();
        let after = device.work_counts();
        assert_eq!(gates.lock().unwrap().len(), 1, "the gate was asked once");
        assert_eq!(
            after.chunk_bound_buffers_made,
            before.chunk_bound_buffers_made
        );
        assert_eq!(
            after.host_ram_chunks_imported,
            before.host_ram_chunks_imported
        );
        assert_eq!(
            gpu_bytes(&device, &made, length),
            bytes(at..at + length),
            "the import reads the bytes"
        );
        made
    };
    // `None`: today's path.
    let none = path(start, 2 * page, resolver(&runs, |_, _| None));
    // Two runs.
    let two = path(
        start + 8 * page,
        2 * page,
        resolver(&runs, |start, length| {
            let middle = start + length / 2;
            Some(vec![start..middle, middle..start + length])
        }),
    );
    // One run crossing the first chunk's end.
    let crossing = path(start + 64 * MIB - page, 2 * page, resolver(&runs, itself));
    // An answer short of the window, and one outside the RAM, are refused.
    let short = buffer(
        &device,
        start + 32 * page,
        2 * page,
        resolver(&runs, |start, length| {
            Some(std::iter::once(start..start + length / 2).collect())
        }),
        gate(&gates, true),
    );
    assert!(matches!(
        short,
        Err(Error::Invalid(ref invalid)) if invalid.problem == Problem::HostRam
    ));
    let outside = buffer(
        &device,
        start + 32 * page,
        page,
        resolver(&runs, move |_, length| {
            let past = start + RAM_BYTES + 64 * MIB;
            Some(std::iter::once(past..past + length).collect())
        }),
        gate(&gates, true),
    );
    assert!(matches!(
        outside,
        Err(Error::Invalid(ref invalid)) if invalid.problem == Problem::HostRam
    ));
    drop((none, two, crossing, used, ram));
    assert!(settles(&dropped));
}

#[test]
fn a_chunk_a_command_buffer_holds_outlives_the_use_and_the_keep_alive_goes_after_it() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start, dropped) = new_ram();
    let used = device.use_host_ram(&ram).unwrap();
    let page = device.shared().host_import.as_ref().unwrap().page;
    let live_before = device.work_counts().host_ram_chunks_live;
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let buffer = buffer(
        &device,
        start,
        4 * page,
        resolver(&runs, itself),
        gate(&gates, true),
    )
    .unwrap();
    assert_eq!(device.work_counts().host_ram_chunks_live, live_before + 1);
    // A command buffer that fills it, held back by an event it waits for.
    let queue = device.new_command_queue().unwrap();
    let event = device.new_shared_event().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    (command_buffer.encode_wait_for_event(EventRef::Shared(&event), 1)).unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encoder.fill_buffer(&buffer, 0..4 * page, 0x77).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    // The buffer, the use and the RAM let go of while it waits: the chunk stays.
    drop((buffer, used, ram));
    std::thread::sleep(Duration::from_millis(50));
    assert!(
        !dropped.load(Ordering::SeqCst),
        "the keep-alive waits for the chunk"
    );
    assert_eq!(device.work_counts().host_ram_chunks_live, live_before + 1);
    event.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    drop(command_buffer);
    assert!(
        settles(&dropped),
        "the keep-alive goes once the chunk is freed"
    );
    assert_eq!(device.work_counts().host_ram_chunks_live, live_before);
}

#[test]
fn a_ram_is_used_once_at_a_time_and_a_new_one_after() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start, dropped) = new_ram();
    let used = device.use_host_ram(&ram).unwrap();
    // The same RAM again, or one over the same ranges, is misuse.
    let again = device.use_host_ram(&ram);
    assert!(matches!(
        again,
        Err(Error::Invalid(ref invalid)) if invalid.problem == Problem::HostRam
    ));
    let at = NonNull::new(start as *mut u8).unwrap();
    // SAFETY: the same mapping, kept by the first RAM's keep-alive while this one lives.
    let twin = Arc::new(unsafe { HostRam::new(&[(at, RAM_BYTES)], None) }.unwrap());
    assert!(device.use_host_ram(&twin).is_err(), "ranges that meet");
    // A RAM apart from it is taken.
    let (apart, _, apart_dropped) = new_ram();
    let apart_used = device.use_host_ram(&apart).unwrap();
    drop((apart_used, apart));
    assert!(settles(&apart_dropped));
    // Once the first use ended, a new RAM over the same ranges is taken, and imports again.
    drop(used);
    let new_used = device.use_host_ram(&twin).unwrap();
    let page = device.shared().host_import.as_ref().unwrap().page;
    fill_pattern(start..start + page);
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let before = device.work_counts();
    let made = buffer(
        &device,
        start,
        page,
        resolver(&runs, itself),
        gate(&gates, true),
    )
    .unwrap();
    let after = device.work_counts();
    assert_eq!(
        after.host_ram_chunks_imported - before.host_ram_chunks_imported,
        1
    );
    assert_eq!(gpu_bytes(&device, &made, page), bytes(start..start + page));
    drop((made, new_used, twin, ram));
    assert!(settles(&dropped));
}

/// Give `buffer` the bytes at host address `at`, resolved by `runs`, under `gate`.
fn rebind(buffer: &Buffer, at: u64, runs: RunResolver, gate: ImportGate) -> Result<(), Error> {
    // SAFETY: the RAM's bytes, mapped until its keep-alive goes, after every device let go; no
    // command that uses the buffer is encoded or pending.
    let memory = unsafe { HostMemory::new(NonNull::new(at as *mut u8).unwrap()) }
        .with_runs(runs)
        .with_import_gate(gate);
    // SAFETY: as above.
    unsafe { buffer.rebind_bytes_no_copy(memory, None, None) }
}

/// The window's halves, two runs of the RAM.
fn halves(start: u64, length: u64) -> Option<Vec<Range<u64>>> {
    let half = length / 2;
    Some(vec![start..start + half, start + half..start + length])
}

#[test]
fn a_rebind_asks_the_resolver_once_and_binds_one_run_in_its_chunk() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start, dropped) = new_ram();
    let used = device.use_host_ram(&ram).unwrap();
    let page = device.shared().host_import.as_ref().unwrap().page;
    let length = 4 * page;
    let (first, second) = (start + 2 * page, start + 64 * MIB + 8 * page);
    fill_pattern(first..first + length);
    fill_pattern(second..second + length);
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let made = buffer(
        &device,
        first,
        length,
        resolver(&runs, itself),
        gate(&gates, true),
    )
    .unwrap();
    runs.lock().unwrap().clear();
    let before = device.work_counts();
    rebind(&made, second, resolver(&runs, itself), gate(&gates, false)).unwrap();
    let after = device.work_counts();
    assert_eq!(*runs.lock().unwrap(), vec![(second, length)], "asked once");
    assert!(gates.lock().unwrap().is_empty(), "the gate was not asked");
    assert_eq!(
        after.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
        1
    );
    assert_eq!(
        after.host_pointer_imports_made - before.host_pointer_imports_made,
        0,
        "nothing imported"
    );
    assert_eq!(
        after.copied_rebind_pages_made - before.copied_rebind_pages_made,
        0
    );
    assert_eq!(made.host_backing(), Some(HostBacking::InHostRam));
    assert_eq!(made.contents(), NonNull::new(second as *mut u8));
    assert_eq!(
        gpu_bytes(&device, &made, length),
        bytes(second..second + length)
    );
    blit(&device, |encoder| {
        encoder.fill_buffer(&made, 0..page, 0x6B).unwrap();
    });
    assert!(bytes(second..second + page).iter().all(|&b| b == 0x6B));
    // A heap rebound onto one run of the other chunk binds there too.
    let (heap_at, heap_to) = (start + 32 * page, start + 96 * page);
    fill_pattern(heap_to..heap_to + 16 * page);
    // SAFETY: as `buffer`'s.
    let memory = unsafe { HostMemory::new(NonNull::new(heap_at as *mut u8).unwrap()) }
        .with_runs(resolver(&runs, itself))
        .with_import_gate(gate(&gates, true));
    let descriptor = crate::HeapDescriptor::new(
        crate::HeapType::Placement,
        16 * page,
        ResourceOptions::default(),
    );
    // SAFETY: as above.
    let heap =
        unsafe { device.new_heap_with_bytes_no_copy(memory, &descriptor, None, None) }.unwrap();
    let placed = heap
        .new_buffer_at_offset(4 * page, ResourceOptions::default(), page)
        .unwrap();
    let before = device.work_counts();
    // SAFETY: as above; nothing uses the heap.
    let memory = unsafe { HostMemory::new(NonNull::new(heap_to as *mut u8).unwrap()) }
        .with_runs(resolver(&runs, itself))
        .with_import_gate(gate(&gates, false));
    // SAFETY: as above.
    unsafe { heap.rebind_bytes_no_copy(memory, None, None) }.unwrap();
    let after = device.work_counts();
    assert_eq!(
        after.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
        1
    );
    assert!(gates.lock().unwrap().is_empty(), "the gate was not asked");
    assert_eq!(placed.host_backing(), Some(HostBacking::InHostRam));
    assert_eq!(
        gpu_bytes(&device, &placed, 4 * page),
        bytes(heap_to + page..heap_to + 5 * page)
    );
    eprintln!("HOST-RAM-REBIND: a buffer and a heap rebound onto one run, bound in its chunk");
    drop((placed, heap, made, used, ram));
    assert!(settles(&dropped));
}

#[test]
fn a_rebind_onto_several_runs_takes_the_gate_and_the_copy_where_it_refuses() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start, dropped) = new_ram();
    let used = device.use_host_ram(&ram).unwrap();
    let page = device.shared().host_import.as_ref().unwrap().page;
    let length = 4 * page;
    let (first, second) = (start + 2 * page, start + 20 * page);
    fill_pattern(first..first + length);
    fill_pattern(second..second + length);
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let made = buffer(
        &device,
        first,
        length,
        resolver(&runs, itself),
        gate(&gates, true),
    )
    .unwrap();
    assert_eq!(made.host_backing(), Some(HostBacking::InHostRam));
    runs.lock().unwrap().clear();
    let before = device.work_counts();
    rebind(&made, second, resolver(&runs, halves), gate(&gates, false)).unwrap();
    let after = device.work_counts();
    assert_eq!(*runs.lock().unwrap(), vec![(second, length)], "asked once");
    assert_eq!(gates.lock().unwrap().len(), 1, "the gate asked once");
    assert_eq!(
        after.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
        0
    );
    assert_eq!(
        after.copied_rebind_pages_made - before.copied_rebind_pages_made,
        1
    );
    assert_eq!(made.host_backing(), Some(HostBacking::Copied));
    assert_eq!(
        gpu_bytes(&device, &made, length),
        bytes(second..second + length)
    );
    blit(&device, |encoder| {
        encoder.fill_buffer(&made, page..2 * page, 0x3C).unwrap();
    });
    let now = bytes(second..second + length);
    assert!(now[page as usize..2 * page as usize]
        .iter()
        .all(|&b| b == 0x3C));
    eprintln!("HOST-RAM-REBIND: a rebind onto two runs, refused by the gate, is a copy");
    drop((made, used, ram));
    assert!(settles(&dropped));
}

/// Whether `device`'s driver breaks its later allocations as it fails an import of pages the
/// kernel cannot pin (`RuntimeDriverDefect::UnpinnablePagesBreakHostMemory`, NVIDIA's).
#[cfg(target_os = "linux")]
fn breaks_on_unpinnable_pages(device: &Device) -> bool {
    let defect = metal2vulkan::error::RuntimeDriverDefect::UnpinnablePagesBreakHostMemory;
    device.properties().driver_defects.contains(&defect)
}

/// A new RAM's first chunk is imported on `device`, two buffers bound in it, their bytes the
/// GPU's.
#[cfg(target_os = "linux")]
fn a_chunk_binds(device: &Device) {
    let page = device.shared().host_import.as_ref().unwrap().page;
    let (ram, start, dropped) = new_ram();
    let used = device.use_host_ram(&ram).unwrap();
    let (a, b) = (start + MIB, start + 3 * MIB);
    fill_pattern(a..a + 2 * page);
    fill_pattern(b..b + page);
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let before = device.work_counts();
    let over = |at, length| {
        buffer(
            device,
            at,
            length,
            resolver(&runs, itself),
            gate(&gates, true),
        )
    };
    let (first, second) = (over(a, 2 * page).unwrap(), over(b, page).unwrap());
    let after = device.work_counts();
    let made = (
        after.host_ram_chunks_imported - before.host_ram_chunks_imported,
        after.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
    );
    assert_eq!(
        made,
        (1, 2),
        "the first chunk imported, both buffers bound in it"
    );
    assert_eq!(gpu_bytes(device, &first, 2 * page), bytes(a..a + 2 * page));
    assert_eq!(gpu_bytes(device, &second, page), bytes(b..b + page));
    drop((first, second, used, ram));
    assert!(settles(&dropped));
}

/// A no-copy buffer over a dma-buf's mapping, with a source of its dma-buf, made before the RAM's
/// first chunk is imported, leaves that import and the buffers bound in it as they are without
/// it ([`a_chunk_binds`]). On a driver whose failed import of such pages breaks its later ones,
/// the runtime refuses the pages before the driver sees them, and the buffer is the dma-buf's or
/// a copy of the pages; elsewhere it is whatever the driver makes of them. Linux only, where
/// `/dev/udmabuf` opens.
#[cfg(target_os = "linux")]
#[test]
fn a_chunk_is_imported_after_a_refused_import_of_a_dma_buf_mapping() {
    let Some(device) = device() else {
        return;
    };
    let page = device.shared().host_import.as_ref().unwrap().page;
    let Some(mapping) = super::copied::DmaBufMapping::new(&vec![0x5Au8; 2 * page as usize]) else {
        return;
    };
    let before = device.work_counts();
    // SAFETY: the mapping outlives the buffer, and the source makes dma-bufs of its pages.
    let first = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::with_dma_buf(mapping.pages, mapping.source()),
            2 * page,
            ResourceOptions::default(),
            None,
        )
    };
    let after = device.work_counts();
    let unpinnable = after.host_imports_unpinnable - before.host_imports_unpinnable;
    eprintln!(
        "HOST-RAM: the dma-buf mapping's buffer: {:?}; refused unpinnable {unpinnable}, pointer \
         imports {}, dma-buf imports {}, copies {}",
        first.as_ref().map(|buffer| buffer.copied()),
        after.host_pointer_imports_made - before.host_pointer_imports_made,
        after.dma_buf_imports - before.dma_buf_imports,
        after.copied_host_buffers_made - before.copied_host_buffers_made,
    );
    let breaks = breaks_on_unpinnable_pages(&device);
    assert_eq!(unpinnable, u64::from(breaks), "refused before the driver");
    assert!(!breaks || first.is_ok(), "the dma-buf's or a copy");
    drop(first);
    a_chunk_binds(&device);
    drop(mapping);
}

/// A no-copy buffer, a no-copy heap and a buffer's rebind over pages of a dma-buf's mapping, each
/// with no gate and no source: on a driver whose failed import of such pages breaks its later ones,
/// the runtime refuses the pages before the driver sees them (`WorkCounts::host_imports_unpinnable`)
/// and each is a copy of them (`WorkCounts::copied_host_buffers_made`, for a failed import), whose
/// bytes the GPU reads; the RAM's first chunk is imported after them, and buffers bound in it
/// ([`a_chunk_binds`]). Elsewhere each is whatever the driver makes of the pages, and the chunk is
/// imported all the same. Linux only, where `/dev/udmabuf` opens.
#[cfg(target_os = "linux")]
#[test]
fn pages_refused_before_the_driver_with_no_source_are_copied() {
    let Some(device) = device() else {
        return;
    };
    let importer = device.shared().host_import.as_ref().unwrap();
    let unit = importer.page.max(importer.alignment);
    let length = 4 * unit;
    // Three objects' pages, one after the other: the buffer's, the heap's, the rebind's.
    let pattern: Vec<u8> = (0..3 * length).map(|i| (i % 251) as u8 ^ 0xA5).collect();
    let Some(mapping) = super::copied::DmaBufMapping::new(&pattern) else {
        return;
    };
    let pages = |object: u64| {
        let at = mapping
            .pages
            .as_ptr()
            .wrapping_add((object * length) as usize);
        // SAFETY: the mapping's pages, which outlive every object over them.
        unsafe { HostMemory::new(NonNull::new(at).unwrap()) }
    };
    let expected = |object: u64| &pattern[(object * length) as usize..][..length as usize];
    let (ram, start, dropped) = new_ram();
    // SAFETY: the RAM's bytes, mapped until its keep-alive goes, after every device let go.
    let rebound = unsafe {
        let memory = HostMemory::new(NonNull::new((start + 5 * MIB) as *mut u8).unwrap());
        device.new_buffer_with_bytes_no_copy(memory, length, ResourceOptions::default(), None)
    }
    .unwrap();
    let before = device.work_counts();
    // SAFETY: as above.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(pages(0), length, ResourceOptions::default(), None)
    };
    let descriptor = crate::HeapDescriptor::new(
        crate::HeapType::Placement,
        length,
        ResourceOptions::default(),
    );
    // SAFETY: as above.
    let heap = unsafe { device.new_heap_with_bytes_no_copy(pages(1), &descriptor, None, None) };
    // SAFETY: as above; nothing uses the buffer.
    let rebind = unsafe { rebound.rebind_bytes_no_copy(pages(2), None, None) };
    let after = device.work_counts();
    let counted = |of: fn(&crate::WorkCounts) -> u64| of(&after) - of(&before);
    eprintln!(
        "HOST-RAM: over the dma-buf mapping, no source: buffer {:?}, heap {:?}, rebind {:?}; \
         refused unpinnable {}, copies {} ({} for a failed import), heaps copied {}, rebinds \
         copied {}",
        made.as_ref().map(|buffer| buffer.copied()),
        heap.as_ref().map(|_| ()),
        rebind,
        counted(|counts| counts.host_imports_unpinnable),
        counted(|counts| counts.copied_host_buffers_made),
        counted(|counts| counts.copied_host_buffers_degraded),
        counted(|counts| counts.copied_heaps_made),
        counted(|counts| counts.copied_rebind_pages_made),
    );
    let breaks = breaks_on_unpinnable_pages(&device);
    assert_eq!(
        counted(|counts| counts.host_imports_unpinnable),
        3 * u64::from(breaks),
        "refused before the driver, each"
    );
    if breaks {
        let made = made.unwrap();
        let heap = heap.unwrap();
        rebind.unwrap();
        assert!(made.copied() && rebound.copied(), "copies");
        let copies = (
            counted(|counts| counts.copied_host_buffers_made),
            counted(|counts| counts.copied_host_buffers_degraded),
            counted(|counts| counts.copied_heaps_made),
            counted(|counts| counts.copied_rebind_pages_made),
        );
        assert_eq!(
            copies,
            (3, 3, 1, 1),
            "three copies, each for a failed import"
        );
        let placed = (heap.new_buffer_at_offset(length, ResourceOptions::default(), 0)).unwrap();
        assert!(
            gpu_bytes(&device, &made, length) == expected(0),
            "the buffer's bytes"
        );
        assert!(
            gpu_bytes(&device, &placed, length) == expected(1),
            "the heap's bytes"
        );
        assert!(
            gpu_bytes(&device, &rebound, length) == expected(2),
            "the rebind's bytes"
        );
        drop((made, placed, heap));
    }
    drop((rebound, ram));
    assert!(settles(&dropped));
    a_chunk_binds(&device);
    drop(mapping);
}
