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
//! - **The use.** Dropping the use while a command buffer still holds a chunk frees the chunk
//!   after that command buffer completed, and the RAM's keep-alive goes after it; the same RAM
//!   used twice is refused, a RAM apart from it taken; a new RAM over the same ranges is taken once
//!   the first use ended.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device that imports no host memory says so and passes.

use super::dispatch_pending::state;
use crate::{
    Buffer, CommandBufferStatus, Device, DeviceKind, DeviceOptions, Error, EventRef, HostMemory,
    HostRam, ImportGate, ImportRequest, KeepAlive, Problem, ResourceOptions, RunResolver, Size,
};
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

/// A device of the cfg's kind; `None` where it imports no host memory.
fn device() -> Option<Device> {
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
    Some(device)
}

/// A resolver answering `answer` of each window it is asked, each window into `asked`.
fn resolver(
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
fn itself(start: u64, length: u64) -> Option<Vec<Range<u64>>> {
    Some(std::iter::once(start..start + length).collect())
}

/// A gate answering `answer`, each request into `asked`.
fn gate(asked: &Arc<Mutex<Vec<ImportRequest>>>, answer: bool) -> ImportGate {
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
    assert!(!first.copied());
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
    assert!(!made.copied());
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
    assert!(made.copied());
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
