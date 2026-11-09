//! What a device's recovery carries of the caller's host memory (`device/carry.rs`): its no-copy
//! buffers, made again over the same pages on the new `VkDevice` (an import of the pointer, a copy
//! where the caller's gate refuses one, a buffer over the same bytes lying in the same import, a
//! run of a guest RAM), and its uses of guest RAM. Each serves its pointer and its bytes as they
//! were, its deallocator runs once at its release, after every generation's import of its pages
//! went, and the lost generation goes at its rest, as nothing carried holds it.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault: the loss is marked (`Device::recover_for_test`).

use super::device_recovery::{heard_device, outcome, recovery_notices, Heard, Scratch, WAIT};
use super::host_ram_device::{fresh_ram, gate, itself, resolver};
use crate::{
    Buffer, CommandBufferStatus, CommandQueue, Deallocator, Device, HostBacking, HostMemory,
    ImportRequest, ResourceOptions, WorkCounts,
};
use std::ptr::NonNull;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The bytes of each no-copy buffer the tests make.
const LENGTH: u64 = 16 * 1024;

/// The bytes the tests write and read of each buffer.
const BYTES: u64 = 4096;

/// A fresh private anonymous mapping of `length` bytes, page-aligned: its first byte.
fn mapping(length: u64) -> NonNull<u8> {
    // SAFETY: a fresh private anonymous mapping, unmapped by `unmap`.
    let start = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert_ne!(start, libc::MAP_FAILED);
    NonNull::new(start.cast()).unwrap()
}

/// Unmap [`mapping`]'s `length` bytes at `start`.
fn unmap(start: NonNull<u8>, length: u64) {
    // SAFETY: the test's mapping, which no buffer lies in any more.
    unsafe { libc::munmap(start.as_ptr().cast(), length as usize) };
}

/// Set the first [`BYTES`] bytes at `at` to `byte` from the host, while no command buffer runs.
fn host_fill(at: NonNull<u8>, byte: u8) {
    // SAFETY: the test's mapping, which no command buffer uses now.
    unsafe { std::ptr::write_bytes(at.as_ptr(), byte, BYTES as usize) };
}

/// The first [`BYTES`] bytes at `at`, read by the host, while no command buffer runs.
fn host_bytes(at: NonNull<u8>) -> Vec<u8> {
    // SAFETY: as above.
    unsafe { std::slice::from_raw_parts(at.as_ptr(), BYTES as usize) }.to_vec()
}

/// `buffer`'s first [`BYTES`] bytes set to `byte` by `queue`'s GPU work, completed.
#[track_caller]
fn fill(queue: &CommandQueue, buffer: &Buffer, byte: u8) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..BYTES, byte).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
}

/// `source`'s first [`BYTES`] bytes as `queue`'s GPU work reads them, into a new `Shared` buffer
/// of `device`'s serving generation.
#[track_caller]
fn read(device: &Device, queue: &CommandQueue, source: &Buffer) -> Vec<u8> {
    let target = device.new_buffer(BYTES, ResourceOptions::shared()).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(source, 0, &target, 0, BYTES)
        .unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert_eq!(
        outcome(&command_buffer),
        (CommandBufferStatus::Completed, None)
    );
    let contents = target.contents().unwrap();
    // SAFETY: the target's bytes, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), BYTES as usize) }.to_vec()
}

/// A deallocator counting its runs in `ran`, each with the range it was handed.
fn counted(ran: &Arc<Mutex<Vec<(u64, u64)>>>) -> Deallocator {
    let ran = Arc::clone(ran);
    Box::new(move |pointer, length| ran.lock().unwrap().push((pointer.as_ptr() as u64, length)))
}

/// Whether `ran` came to `runs` runs within [`WAIT`]: a deallocator may run on a handler thread a
/// moment after the release that let it go.
fn ran_to(ran: &Arc<Mutex<Vec<(u64, u64)>>>, runs: usize) -> bool {
    let began = Instant::now();
    while ran.lock().unwrap().len() < runs {
        if began.elapsed() > WAIT {
            return false;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    true
}

/// The page of the serving generation's imports of host memory, `None` where it imports none:
/// read through a handle of that generation the test lets go of at once. A root handle's own
/// reach of a generation's state pins that generation for the root's life (`Device::shared_arc`),
/// which would keep the lost `VkDevice` past the recovery.
fn import_page(device: &Device) -> Option<u64> {
    let serving = device.current();
    let page = serving.shared().host_import.as_ref().ok()?.page;
    Some(page)
}

/// The device's live `VkDevice`s once they are `live`, waited for at most [`WAIT`].
fn live_vk_devices(device: &Device, live: u64) -> u64 {
    let began = Instant::now();
    loop {
        let now = device.work_counts().vk_devices_live;
        if now == live || began.elapsed() > WAIT {
            return now;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// A recovery carries a no-copy buffer imported by its pointer (with a deallocator), a second
/// buffer over the same bytes, which lies in its import there as here, and a buffer whose import
/// the caller's gate refuses, a copy there as here, the gate asked again. Each keeps its
/// `contents` pointer, its Metal address and its bytes; the host's writes reach the GPU and the
/// GPU's the pages after it. The lost `VkDevice` goes at its rest, as none of them holds it; a
/// second recovery carries them on likewise; and the deallocator runs once, at the buffer's
/// release.
#[test]
fn a_recovery_carries_no_copy_buffers_over_the_same_pages() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    if import_page(&device).is_none() {
        eprintln!("NO-COPY-CARRY: the device imports no host memory here; nothing to test");
        return;
    }
    let queue = device.new_command_queue().unwrap();
    let (at, copied_at) = (mapping(LENGTH), mapping(LENGTH));
    host_fill(at, 5);
    host_fill(copied_at, 6);
    let ran = Arc::new(Mutex::new(Vec::new()));
    // SAFETY: the test's mapping, unmapped after every buffer over it went.
    let buffer = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(at),
            LENGTH,
            ResourceOptions::shared(),
            Some(counted(&ran)),
        )
    }
    .unwrap();
    // SAFETY: as above.
    let alias = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(at),
            LENGTH,
            ResourceOptions::shared(),
            None,
        )
    }
    .unwrap();
    let asked: Arc<Mutex<Vec<ImportRequest>>> = Arc::default();
    // SAFETY: as above.
    let copied = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(copied_at).with_import_gate(gate(&asked, false)),
            LENGTH,
            ResourceOptions::shared(),
            None,
        )
    }
    .unwrap();
    let backing = buffer.host_backing();
    assert_eq!(copied.host_backing(), Some(HostBacking::Copied));
    assert_eq!(asked.lock().unwrap().len(), 1);
    assert_eq!(read(&device, &queue, &alias), vec![5; BYTES as usize]);
    assert_eq!(read(&device, &queue, &copied), vec![6; BYTES as usize]);
    let address = buffer.gpu_address();
    let from = heard.lock().unwrap().len();
    device.recover_for_test().unwrap();
    let said = recovery_notices(&heard, from, "the device");
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(
        said[0].contains("it carried 0 events, 3 buffers with their contents"),
        "{said:?}"
    );
    // Each on the new generation, over the same pages as before, the gate asked again.
    for carried in [&buffer, &alias, &copied] {
        assert!(carried.device().generation_serves());
    }
    assert_eq!(buffer.contents(), Some(at));
    assert_eq!(copied.contents(), Some(copied_at));
    assert_eq!(buffer.gpu_address(), address);
    assert_eq!(buffer.host_backing(), backing);
    assert_eq!(copied.host_backing(), Some(HostBacking::Copied));
    assert_eq!(asked.lock().unwrap().len(), 2, "the gate is asked again");
    // The bytes as they were, the host's writes read by the GPU, the GPU's in the pages.
    assert_eq!(read(&device, &queue, &buffer), vec![5; BYTES as usize]);
    assert_eq!(read(&device, &queue, &copied), vec![6; BYTES as usize]);
    host_fill(at, 7);
    host_fill(copied_at, 8);
    assert_eq!(read(&device, &queue, &alias), vec![7; BYTES as usize]);
    assert_eq!(read(&device, &queue, &copied), vec![8; BYTES as usize]);
    fill(&queue, &alias, 9);
    assert_eq!(host_bytes(at), vec![9; BYTES as usize]);
    assert_eq!(read(&device, &queue, &buffer), vec![9; BYTES as usize]);
    // Nothing holds the lost generation: it went at its rest, its imports of the pages with it,
    // and the deallocator waits for the buffer's release.
    assert_eq!(live_vk_devices(&device, 1), 1);
    assert!(ran.lock().unwrap().is_empty(), "{:?}", ran.lock().unwrap());
    // A second recovery carries them on again, from the generation the first made: the
    // deallocator waits for one more import, and still for the release.
    let from = heard.lock().unwrap().len();
    device.recover_for_test().unwrap();
    let said = recovery_notices(&heard, from, "the device");
    assert!(
        said.len() == 1 && said[0].contains("3 buffers with their contents"),
        "{said:?}"
    );
    assert_eq!(
        asked.lock().unwrap().len(),
        3,
        "the gate is asked at each recovery"
    );
    assert_eq!(buffer.contents(), Some(at));
    assert_eq!(read(&device, &queue, &buffer), vec![9; BYTES as usize]);
    assert_eq!(read(&device, &queue, &copied), vec![8; BYTES as usize]);
    assert_eq!(live_vk_devices(&device, 1), 1);
    assert!(ran.lock().unwrap().is_empty(), "{:?}", ran.lock().unwrap());
    drop(buffer);
    assert!(ran_to(&ran, 1), "the deallocator never ran");
    assert_eq!(
        *ran.lock().unwrap(),
        vec![(at.as_ptr() as u64, LENGTH)],
        "the deallocator ran once, with the buffer's range"
    );
    drop((alias, copied));
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(ran.lock().unwrap().len(), 1);
    unmap(at, LENGTH);
    unmap(copied_at, LENGTH);
}

/// A recovery carries a use of guest RAM before the buffers, and a no-copy buffer over a run of
/// the RAM made again over it on the new `VkDevice`, its resolver asked again there: bound in the
/// RAM's chunk there, or in an import of its run alone. The lost generation goes at its rest, its
/// use of the RAM let go of, and the RAM's keep-alive goes once the program let go of its use.
#[test]
fn a_recovery_carries_a_use_of_guest_ram_and_the_buffers_over_it() {
    let scratch = Scratch::new();
    let heard = Heard::default();
    let device = heard_device(&scratch.0, &heard);
    let Some(page) = import_page(&device) else {
        eprintln!("NO-COPY-CARRY: the device imports no host memory here; nothing to test");
        return;
    };
    // Asked now, as a RAM's kind reads the answer (`fresh_ram`).
    device.imports_shared_host_pointers().unwrap();
    assert!(device.settle_submit_checks(crate::tests::SETTLE));
    let queue = device.new_command_queue().unwrap();
    let (ram, start, dropped) = fresh_ram(&device, 64 << 20);
    let used = device.use_host_ram(&ram).unwrap();
    let (runs, gates): (Arc<Mutex<_>>, Arc<Mutex<_>>) = Default::default();
    let at = NonNull::new(start as *mut u8).unwrap();
    // SAFETY: the RAM's bytes, mapped until its keep-alive goes, after every device let go.
    let buffer = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(at)
                .with_runs(resolver(&runs, itself))
                .with_import_gate(gate(&gates, true)),
            (4 * page).max(BYTES),
            ResourceOptions::shared(),
            None,
        )
    }
    .unwrap();
    assert_eq!(runs.lock().unwrap().len(), 1);
    host_fill(at, 3);
    assert_eq!(read(&device, &queue, &buffer), vec![3; BYTES as usize]);
    // The counts are the device's, across its generations.
    let bound =
        |counts: &WorkCounts| counts.chunk_bound_buffers_made + counts.run_bound_buffers_made;
    let bound_before = bound(&device.work_counts());
    let from = heard.lock().unwrap().len();
    device.recover_for_test().unwrap();
    let said = recovery_notices(&heard, from, "the device");
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(
        said[0].contains("it carried 0 events, 1 guest RAM uses, 1 buffers with their contents"),
        "{said:?}"
    );
    assert!(buffer.device().generation_serves());
    assert_eq!(runs.lock().unwrap().len(), 2, "the resolver is asked again");
    let counts = buffer.device().work_counts_now();
    assert_eq!(
        bound(&counts) - bound_before,
        1,
        "the buffer lies in the RAM there: {counts:?}"
    );
    assert_eq!(buffer.contents(), Some(at));
    assert_eq!(read(&device, &queue, &buffer), vec![3; BYTES as usize]);
    fill(&queue, &buffer, 4);
    assert_eq!(host_bytes(at), vec![4; BYTES as usize]);
    // The lost generation went at its rest: its use of the RAM, and the buffer's import there.
    assert_eq!(live_vk_devices(&device, 1), 1);
    drop((buffer, used, ram));
    let began = Instant::now();
    while !dropped.load(Ordering::SeqCst) && began.elapsed() < WAIT {
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        dropped.load(Ordering::SeqCst),
        "the RAM's keep-alive goes after its last chunk"
    );
}
