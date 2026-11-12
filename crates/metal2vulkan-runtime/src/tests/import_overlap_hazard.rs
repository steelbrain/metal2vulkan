//! The shapes of host-pointer imports over pages a live import already maps, run at volume on a
//! GPU, each smaller import freed while the larger one lives: `VK_EXT_external_memory_host`
//! leaves the same host allocation imported twice into one device undefined (its issue 6), and
//! the runtime makes none of them (`host_aliases.rs`, "One page, one import"; `host_ram.rs`,
//! `RamOnDevice::runs_alone`).
//!
//! - **Runs under a chunk.** Objects over chunks of a guest RAM the device's thread has not
//!   imported (each a jump past the window), of 64 KiB to 384 KiB at 64 KiB multiples into the
//!   chunk, written and read by blits, a second buffer over the same pages after the chunk's
//!   import, and the first dropped. A runtime that imports such a run alone by its pointer frees
//!   that import while the chunk's import of the same pages lives.
//! - **Buffers under a heap.** Views of 2 MiB made of 512 scattered pages of a `memfd` (as a VMM
//!   stitches a guest object's pages), a no-copy heap over each, then a buffer over exactly the
//!   heap's bytes and buffers over 64 KiB to 384 KiB at 64 KiB multiples inside them, each
//!   written and read by a blit, then dropped while the heap lives. A runtime that imports each
//!   buffer's pages by their pointer frees each such import while the heap's lives.
//!
//! Each test prints the device's counts of the imports it made and where its objects lay: the
//! evidence of which path ran. Every wait is bounded; a stalled submission is left to the job's
//! limit, and the kernel's log is read by the job (`journalctl -k`), not by the test.
//!
//! A GPU hazard: on NVIDIA's driver, failed DMA mappings in its kernel module came with
//! submissions that never returned and a device lost until the host rebooted, on hosts whose
//! runtime made these shapes among others; whether these shapes alone trip it is what the test
//! asks. It compiles only under `m2v_gpu` with `m2v_gpu_hazards` (docs/VALIDATION.md), on Linux.

use super::host_ram_device::{ahead_device, fresh_ram, gate, itself, resolver};
use crate::host_ram::AHEAD_CHUNKS;
use crate::{
    Buffer, CommandBufferStatus, CommandQueue, Device, HeapDescriptor, HeapType, HostMemory,
    ResourceOptions,
};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const MIB: u64 = 1 << 20;
const CHUNK: u64 = 64 * MIB;
/// A command buffer's longest wait: far past a blit of a few hundred KiB.
const WAIT: Duration = Duration::from_secs(20);

/// The counts that say which path the objects took, all present before and after the runtime
/// stopped making these shapes.
fn counts(device: &Device) -> [u64; 6] {
    let counts = device.work_counts();
    [
        counts.host_pointer_imports_made,
        counts.run_imports_made,
        counts.run_bound_buffers_made,
        counts.chunk_bound_buffers_made,
        counts.copied_host_buffers_made,
        counts.object_churn.imports_inside_live,
    ]
}

/// `now` less `before`, by count.
fn since(now: [u64; 6], before: [u64; 6]) -> [u64; 6] {
    std::array::from_fn(|at| now[at] - before[at])
}

/// Fill `buffer`'s first `length` bytes with `value`, copy them into `out`, and wait.
fn fill_and_read(queue: &CommandQueue, buffer: &Buffer, out: &Buffer, length: u64, value: u8) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encoder.fill_buffer(buffer, 0..length, value).unwrap();
    (encoder.copy_from_buffer_to_buffer(buffer, 0, out, 0, length)).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The first `length` bytes of `buffer`'s contents.
fn contents(buffer: &Buffer, length: u64) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer's bytes, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), length as usize) }.to_vec()
}

/// 64 KiB to 384 KiB, at a 64 KiB multiple: the `index`th of a fixed cycle.
fn piece(index: u64) -> (u64, u64) {
    let length = (1 + index % 6) * 64 * 1024;
    let at = (index * 7 % 24) * 64 * 1024;
    (at, length)
}

#[test]
fn runs_under_a_live_chunk_import_at_volume() {
    let Some(device) = ahead_device() else {
        return;
    };
    let queue = device.new_command_queue().unwrap();
    const CHUNKS: u64 = 24;
    const ROUNDS: u64 = 6;
    // Each jump lands past the window the last one opened.
    let stride = AHEAD_CHUNKS as u64 + 2;
    let order: Vec<u64> = (0..CHUNKS).map(|at| at * stride % CHUNKS).collect();
    let out = device
        .new_buffer(384 * 1024, ResourceOptions::shared())
        .unwrap();
    let start_counts = counts(&device);
    let mut objects = 0u64;
    for round in 0..ROUNDS {
        let (ram, start, _dropped) = fresh_ram(&device, CHUNKS * CHUNK);
        let used = device.use_host_ram(&ram).unwrap();
        let (runs, gates) = (
            Arc::new(Mutex::new(Vec::new())),
            Arc::new(Mutex::new(Vec::new())),
        );
        let over = |at: u64, length: u64| {
            // SAFETY: the RAM's bytes, mapped until its keep-alive goes, after every device let
            // go.
            let memory = unsafe { HostMemory::new(NonNull::new(at as *mut u8).unwrap()) }
                .with_runs(resolver(&runs, itself))
                .with_import_gate(gate(&gates, true));
            // SAFETY: as above.
            unsafe {
                device.new_buffer_with_bytes_no_copy(
                    memory,
                    length,
                    ResourceOptions::default(),
                    None,
                )
            }
            .unwrap()
        };
        for (index, &chunk) in order.iter().enumerate() {
            let (into, length) = piece(round * CHUNKS + index as u64);
            let at = start + chunk * CHUNK + into;
            let first = over(at, length);
            fill_and_read(&queue, &first, &out, length, 0x5a);
            assert!(contents(&out, length).iter().all(|&byte| byte == 0x5a));
            // The same pages again once the device's thread had time to import the chunk.
            std::thread::sleep(Duration::from_millis(20));
            let second = over(at, length);
            fill_and_read(&queue, &second, &out, length, 0xa5);
            drop(first);
            fill_and_read(&queue, &second, &out, length, 0x3c);
            assert!(contents(&out, length).iter().all(|&byte| byte == 0x3c));
            drop(second);
            objects += 2;
        }
        drop((used, ram));
        let made = since(counts(&device), start_counts);
        eprintln!(
            "IMPORT-HAZARD runs: round {round}: {objects} objects; pointer imports {}, run imports \
             {}, run-bound {}, chunk-bound {}, copies {}, imports inside live {}",
            made[0], made[1], made[2], made[3], made[4], made[5]
        );
    }
}

/// A view of `pages` pages of `fd` (a `memfd` of `file_pages` pages), each from a scattered page
/// of the file, mapped shared at consecutive addresses from a multiple of 2 MiB: its first byte.
fn scattered_view(fd: libc::c_int, file_pages: u64, pages: u64, seed: u64, page: u64) -> u64 {
    let length = pages * page;
    // SAFETY: a fresh reservation of addresses, no file behind it.
    let reserved = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            (length + 2 * MIB) as usize,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert_ne!(reserved, libc::MAP_FAILED);
    let start = (reserved as u64).next_multiple_of(2 * MIB);
    for index in 0..pages {
        let from = (seed + index * 2_654_435_761) % file_pages;
        // SAFETY: inside the reservation above, which this test owns.
        let mapped = unsafe {
            libc::mmap(
                (start + index * page) as *mut libc::c_void,
                page as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd,
                (from * page) as libc::off_t,
            )
        };
        assert_ne!(mapped, libc::MAP_FAILED);
    }
    start
}

#[test]
fn buffers_under_a_live_heap_import_at_volume() {
    let Some(device) = ahead_device() else {
        return;
    };
    let queue = device.new_command_queue().unwrap();
    let page = device.shared().host_import.as_ref().unwrap().page;
    const VIEW: u64 = 2 * MIB;
    const VIEWS: u64 = 32;
    const PER_VIEW: u64 = 32;
    let file_pages = 64 * MIB / page;
    // SAFETY: a name and flags the call takes.
    let fd = unsafe { libc::memfd_create(c"m2v-hazard-views".as_ptr(), libc::MFD_CLOEXEC) };
    assert!(fd >= 0);
    // SAFETY: the descriptor was just made.
    assert_eq!(
        unsafe { libc::ftruncate(fd, (file_pages * page) as libc::off_t) },
        0
    );
    let out = device
        .new_buffer(384 * 1024, ResourceOptions::shared())
        .unwrap();
    let start_counts = counts(&device);

    for view in 0..VIEWS {
        let start = scattered_view(fd, file_pages, VIEW / page, view * 977, page);

        let memory = |at: u64| {
            // SAFETY: the view's bytes, mapped until the test's end, after the device went.
            unsafe { HostMemory::new(NonNull::new(at as *mut u8).unwrap()) }
        };
        let descriptor = HeapDescriptor::new(HeapType::Placement, VIEW, ResourceOptions::default());
        // SAFETY: as above.
        let heap =
            unsafe { device.new_heap_with_bytes_no_copy(memory(start), &descriptor, None, None) }
                .unwrap();
        let buffer = |at: u64, length: u64| {
            // SAFETY: as above.
            unsafe {
                device.new_buffer_with_bytes_no_copy(
                    memory(at),
                    length,
                    ResourceOptions::default(),
                    None,
                )
            }
            .unwrap()
        };
        let whole = buffer(start, VIEW);
        fill_and_read(&queue, &whole, &out, 64 * 1024, 0x11);
        for index in 0..PER_VIEW {
            let (at, length) = piece(view * PER_VIEW + index);
            let part = buffer(start + at, length);
            fill_and_read(&queue, &part, &out, length, index as u8);
            assert!(contents(&out, length)
                .iter()
                .all(|&byte| byte == index as u8));
            drop(part);
        }
        drop(whole);
        if view % 8 == 7 {
            let made = since(counts(&device), start_counts);
            eprintln!(
                "IMPORT-HAZARD heaps: {} views, {} buffers in them; pointer imports {}, copies {}, \
                 imports inside live {}",
                view + 1,
                (view + 1) * (PER_VIEW + 1),
                made[0],
                made[4],
                made[5]
            );
        }
        drop(heap);
    }
    // The views stay mapped to the process's end (each test is a process of its own), past the
    // device's last hold on them.
    // SAFETY: the test's descriptor; the mappings keep the file.
    unsafe { libc::close(fd) };
}
