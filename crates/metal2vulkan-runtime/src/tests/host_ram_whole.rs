//! A guest RAM every page of which is resident as a device's use of it begins, on the device the
//! cfg selects (`host_ram.rs`, "A resident RAM imported whole"): its chunks run past
//! `LONGEST_LISTED_CHUNK` (512 MiB) and are all imported, and listed once, before
//! `Device::use_host_ram` returns.
//!
//! The RAM is a sealed `memfd` mapped shared, as QEMU's `memory-backend-memfd,share=on` makes
//! it, its source making dma-bufs of its pages through `/dev/udmabuf` as long as the host's
//! `size_limit_mb` lets one be; the import tax is forced held on the device (as RADV's is), so
//! that the plan is listed wherever the driver takes the RAM's dma-bufs, and the device's imports
//! are counted against four times the RAM, so that its chunks are kept.
//!
//! - **Resident**: a RAM of [`RAM_BYTES`], every page written: one chunk of all of it, imported
//!   by the time the use returns (`WorkCounts::host_ram_chunks_imported_ahead`,
//!   `host_ram_chunks_live`, `host_ram_chunk_bytes`).
//! - **A hole**: the same with one page never written: chunks of 512 MiB and the rest, none
//!   imported as the use begins.
//! - **Said**: the resident RAM's import is said once (`metal2vulkan-host-ram`, `Info`), with its
//!   bytes and chunks; the RAM with a hole says nothing.
//! - **Recovered**: a recovery of the device that uses the resident RAM imports it whole again on
//!   the new device, before the recovery returns, and says so again.
//!
//! A negative control, run by hand: with `listed_chunks_of` answering the cap whatever the
//! residency, the resident RAM takes two chunks and imports none, and its test fails.
//!
//! Where `/dev/udmabuf` cannot be opened, its limit is shorter than the RAM, the device imports no
//! host memory, its limits take chunks no longer than the cap (`maxStorageBufferRange`,
//! `maxMemoryAllocationSize` or `maxBufferSize`: llvmpipe's 128 MiB storage range), or the driver
//! refused the RAM's dma-buf (the RAM not listed), the test says so and passes without running.
//!
//! `whole_ram_bench` (ignored, release, run by name) holds a consumer's RAM of each of
//! [`BENCH_BYTES`], on a device of its own: with one page a hole, every chunk imported by the
//! tests' hook as copies would ask for them (chunks of 512 MiB); then every page resident,
//! imported whole as the use begins, then the device recovered. Printed (`HOST-RAM-WHOLE`): the
//! use's time and its imports' time, the chunks live, the first commit's time from commit to
//! completion after them (the submission that lists them first), and a commit's `vkQueueSubmit`
//! time per submission after it with each set held; the recovery's time; and the device's lines
//! of each whole import.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux.

use super::host_residency::{device_with, host_page, udmabuf, Scratch};
use crate::host_ram::{listed_chunk_bytes, LONGEST_LISTED_CHUNK, WHOLE_ID};
use crate::submit_check::{Check, Verdict};
use crate::{
    Device, DmaBufChunk, DmaBufRequest, DmaBufSource, HostRam, KeepAlive, MessageSink,
    ResourceOptions, WorkCounts,
};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const MIB: u64 = 1 << 20;
/// The RAM of the device test: past the 512 MiB cap.
const RAM_BYTES: u64 = 640 * MIB;
/// The RAMs of the bench: consumers' guests.
const BENCH_BYTES: [u64; 3] = [2 << 30, 4 << 30, 8 << 30];
/// The single commits each measurement of a submission's time takes.
const COMMITS: usize = 300;

/// The RAM's mapping, unmapped by the keep-alive.
struct Mapping {
    at: u64,
    length: u64,
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the test's mapping, which no device imports any more: the keep-alive is dropped
        // after the last chunk over it is freed.
        unsafe { libc::munmap(self.at as *mut libc::c_void, self.length as usize) };
    }
}

/// A sealed `memfd` of `length` bytes, mapped shared: the file and the mapping's first byte.
fn file(length: u64) -> (Arc<OwnedFd>, u64) {
    // SAFETY: a new anonymous file, owned at once, then sized and sealed.
    let fd = unsafe {
        let fd = libc::memfd_create(
            c"m2v-host-ram-whole".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        );
        assert!(fd >= 0, "memfd_create: {}", std::io::Error::last_os_error());
        OwnedFd::from_raw_fd(fd)
    };
    // SAFETY: the descriptor is the test's.
    unsafe {
        assert_eq!(libc::ftruncate(fd.as_raw_fd(), length as libc::off_t), 0);
        let seals = libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;
        assert_eq!(libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, seals), 0);
    }
    // SAFETY: a new shared mapping of the whole file.
    let at = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            length as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            fd.as_raw_fd(),
            0,
        )
    };
    assert_ne!(at, libc::MAP_FAILED, "mmap of the file");
    (Arc::new(fd), at as u64)
}

/// Write a byte into every host page of `length` bytes from `at` but the page at `hole`, where
/// given.
fn touch(at: u64, length: u64, hole: Option<u64>) {
    for offset in (0..length).step_by(host_page()) {
        if hole == Some(offset) {
            continue;
        }
        // SAFETY: a page of the test's mapping, readable and writable.
        unsafe { std::ptr::write_volatile((at + offset) as *mut u8, 1) };
    }
}

/// The RAM over `memfd`'s mapping at `at`, with a source of dma-bufs of its pages as long as
/// `largest`, its keep-alive unmapping it.
fn ram(memfd: &Arc<OwnedFd>, at: u64, length: u64, largest: u64) -> Arc<HostRam> {
    let source: DmaBufSource = {
        let memfd = Arc::clone(memfd);
        Arc::new(move |request: DmaBufRequest| {
            let first = request.start.as_ptr() as u64;
            let fd = udmabuf(&memfd, first - at, request.length)?;
            Ok(vec![DmaBufChunk {
                file: Arc::new(fd),
                offset: 0,
                length: request.length,
            }])
        })
    };
    let keep: KeepAlive = Box::new(Mapping { at, length });
    let start = NonNull::new(at as *mut u8).unwrap();
    // SAFETY: the mapping is readable and writable until the keep-alive unmaps it.
    let ram = unsafe { HostRam::new(&[(start, length)], Some(keep)) }.unwrap();
    Arc::new(ram.with_dma_buf(source).with_largest_dma_buf(largest))
}

/// The host's udmabuf `size_limit_mb`, in bytes: 64 MiB where it cannot be read.
fn udmabuf_size_limit() -> u64 {
    std::fs::read_to_string("/sys/module/udmabuf/parameters/size_limit_mb")
        .ok()
        .and_then(|limit| limit.trim().parse::<u64>().ok())
        .unwrap_or(64)
        .saturating_mul(MIB)
}

/// The texts of the device's lines of whole imports ([`WHOLE_ID`]).
type Said = Arc<Mutex<Vec<String>>>;

/// The device the tests take, its import tax forced held and its imports counted against
/// `against` bytes, its lines of whole imports kept in `said`; `None` (said) where `/dev/udmabuf`
/// cannot be opened, its limit is shorter than `length`, or the device imports no host memory.
fn listed_device(
    scratch: &Scratch,
    said: &Said,
    test: &str,
    length: u64,
    against: u64,
) -> Option<Device> {
    if let Err(error) = (std::fs::OpenOptions::new().read(true).write(true)).open("/dev/udmabuf") {
        eprintln!("{test}: /dev/udmabuf cannot be opened ({error}): not run");
        return None;
    }
    let largest = udmabuf_size_limit();
    if largest < length {
        eprintln!(
            "{test}: udmabuf's size_limit_mb is {} MiB, short of the RAM: not run",
            largest >> 20
        );
        return None;
    }
    let kept = Arc::clone(said);
    let sink = MessageSink::new(move |message| {
        if message.id_name == WHOLE_ID {
            kept.lock().unwrap().push(message.text.clone());
        }
    });
    let device = device_with(&scratch.0, Some(sink));
    if device.shared().host_import.is_err() {
        eprintln!("{test}: the device imports no host memory here: not run");
        return None;
    }
    // A RAM is imported whole only where the cap cuts its chunks shorter than the source's and
    // the device's limits do (`host_ram::listed_chunks_of`): llvmpipe's 128 MiB
    // `maxStorageBufferRange` cuts them first.
    let allocation = device.compile_device().allocation();
    let storage = device
        .properties()
        .description
        .limits
        .max_storage_buffer_range;
    let limits = listed_chunk_bytes(
        largest,
        allocation.max_memory_allocation_size,
        allocation.max_buffer_size,
        u64::from(storage),
    );
    if limits <= LONGEST_LISTED_CHUNK {
        eprintln!(
            "{test}: the device's limits take chunks of {} MiB, within the {} MiB cap: not run",
            limits >> 20,
            LONGEST_LISTED_CHUNK >> 20
        );
        return None;
    }
    (device.shared().submit_checks).force(Check::ImportTax, Verdict::Holds);
    device.count_host_ram_imports_against(against);
    Some(device)
}

/// A use of a RAM ([`use_of`]) and what it found.
struct Used {
    used: crate::HostRamUse,
    ram: Arc<HostRam>,
    device: Device,
    /// The chunks' lengths.
    lengths: Vec<u64>,
    /// The chunks imported ahead and live once the use returned, and their bytes.
    counts: (u64, u64, u64),
    /// The device's lines of whole imports.
    said: Said,
    _scratch: Scratch,
}

/// A use of a [`RAM_BYTES`] RAM, every page written but the one at `hole`; `None` (said) where the
/// RAM is not listed.
fn use_of(test: &str, hole: Option<u64>) -> Option<Used> {
    let scratch = Scratch::new();
    let said = Said::default();
    let device = listed_device(&scratch, &said, test, RAM_BYTES, 4 * RAM_BYTES)?;
    let (memfd, at) = file(RAM_BYTES);
    touch(at, RAM_BYTES, hole);
    let ram = ram(&memfd, at, RAM_BYTES, udmabuf_size_limit());
    let before = device.work_counts();
    let started = Instant::now();
    let used = device.use_host_ram(&ram).unwrap();
    let took = started.elapsed();
    let after = device.work_counts();
    let chunks = device.host_ram_chunks();
    let [(true, spans)] = &chunks[..] else {
        eprintln!(
            "{test} {}: the RAM is not listed (the driver refused its dma-buf): not run",
            device.name()
        );
        drop((used, ram));
        return None;
    };
    let lengths: Vec<u64> = spans.iter().map(|span| span.end - span.start).collect();
    let delta = |of: fn(&WorkCounts) -> u64| of(&after) - of(&before);
    let ahead = delta(|counts| counts.host_ram_chunks_imported_ahead);
    let live = delta(|counts| counts.host_ram_chunks_live);
    let bytes = delta(|counts| counts.host_ram_chunk_bytes);
    println!(
        "{test} {}: chunks {:?} MiB, imported ahead {ahead}, live {live}, {} MiB; the use took \
         {:.1} ms, its imports {:.1} ms",
        device.name(),
        lengths
            .iter()
            .map(|length| length >> 20)
            .collect::<Vec<_>>(),
        bytes >> 20,
        took.as_secs_f64() * 1e3,
        delta(|counts| counts.host_ram_chunk_import_nanos) as f64 / 1e6,
    );
    for line in said.lock().unwrap().iter() {
        println!("{test}: said: {line}");
    }
    Some(Used {
        lengths,
        counts: (ahead, live, bytes),
        said,
        device,
        used,
        ram,
        _scratch: scratch,
    })
}

/// The device's line of a whole import of `bytes` in `chunks` chunks, none left to the copies.
fn whole_line(bytes: u64, chunks: usize) -> String {
    format!(
        "guest RAM imported whole as its use began: {} MiB in {chunks} chunks (0 left to the \
         copies over them)",
        bytes >> 20
    )
}

/// A resident RAM's chunks run past the cap and are imported, all of them, as the use begins,
/// and the device says so once.
#[test]
fn a_resident_ram_is_imported_whole_in_chunks_past_the_cap_as_its_use_begins() {
    const TEST: &str = "HOST-RAM-WHOLE resident";
    let Some(used) = use_of(TEST, None) else {
        return;
    };
    assert_eq!(
        used.lengths,
        [RAM_BYTES],
        "{TEST}: one chunk of the whole RAM"
    );
    assert_eq!(
        used.counts,
        (1, 1, RAM_BYTES),
        "{TEST}: imported as the use began"
    );
    let said = used.said.lock().unwrap().clone();
    assert_eq!(said.len(), 1, "{TEST}: said once: {said:?}");
    assert!(
        said[0].contains(&whole_line(RAM_BYTES, 1)) && said[0].contains("first listed in"),
        "{TEST}: {said:?}"
    );
}

/// A RAM with a page the guest never touched keeps the cap, imports nothing as its use begins,
/// and says nothing of a whole import.
#[test]
fn a_ram_with_a_hole_keeps_the_cap_and_imports_its_chunks_as_copies_ask() {
    const TEST: &str = "HOST-RAM-WHOLE hole";
    let last = RAM_BYTES - host_page() as u64;
    let Some(used) = use_of(TEST, Some(last)) else {
        return;
    };
    assert_eq!(
        used.lengths,
        [512 * MIB, RAM_BYTES - 512 * MIB],
        "{TEST}: chunks at the cap"
    );
    assert_eq!(
        (used.counts.0, used.counts.1),
        (0, 0),
        "{TEST}: none imported as the use began"
    );
    assert_eq!(used.said.lock().unwrap().len(), 0, "{TEST}: nothing said");
}

/// A recovery of a device that imported a resident RAM whole makes its use again on the new
/// device, before the recovery returns: the new device imports it whole in turn, on the
/// recovering thread, and says so. Where the new device's own checks and heaps (the forced
/// verdict and count are the first device's) do not import it whole, the test says so and passes.
#[test]
fn a_recovery_imports_a_resident_ram_whole_again_before_the_new_device_serves() {
    const TEST: &str = "HOST-RAM-WHOLE recovered";
    let Some(used) = use_of(TEST, None) else {
        return;
    };
    assert_eq!(
        used.counts,
        (1, 1, RAM_BYTES),
        "{TEST}: imported whole first"
    );
    let before = used.device.work_counts();
    let started = Instant::now();
    used.device.recover_for_test().unwrap();
    let took = started.elapsed();
    let after = used.device.work_counts();
    let ahead = after.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead;
    let said = used.said.lock().unwrap().clone();
    println!(
        "{TEST} {}: the recovery took {:.1} ms; chunks imported ahead in it {ahead}; said {said:?}",
        used.device.name(),
        took.as_secs_f64() * 1e3,
    );
    // The RAMs of the generation that serves now: the root handle's own went lost.
    let chunks = used.device.current().host_ram_chunks();
    let whole = matches!(&chunks[..], [(true, spans)]
        if spans.len() == 1 && spans[0].end - spans[0].start == RAM_BYTES);
    if !whole {
        eprintln!("{TEST}: the new device does not import the RAM whole ({chunks:?}): not run");
        drop((used.used, used.ram));
        return;
    }
    assert_eq!(ahead, 1, "{TEST}: imported whole again in the recovery");
    assert_eq!(said.len(), 2, "{TEST}: said again: {said:?}");
    assert!(
        said[1].contains(&whole_line(RAM_BYTES, 1)),
        "{TEST}: {said:?}"
    );
    drop((used.used, used.ram));
}

/// Commits of one 4-byte fill, each waited for: the first one's time from its commit to its
/// completion, in milliseconds (where chunks were imported since the device's last submission,
/// the one that lists them first), then a commit's `vkQueueSubmit` time over [`COMMITS`] more,
/// in microseconds a queue submission.
fn submissions(device: &Device) -> (f64, f64) {
    let word = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let commit = || {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&word, 0..4, 1).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer
            .wait_until_completed_for(crate::tests::SETTLE)
            .unwrap();
    };
    let started = Instant::now();
    commit();
    let first = started.elapsed().as_secs_f64() * 1e3;
    let before = device.work_counts();
    for _ in 0..COMMITS {
        commit();
    }
    let after = device.work_counts();
    let calls = (after.queue_submissions - before.queue_submissions).max(1);
    let us = (after.batch_submit_nanos - before.batch_submit_nanos) as f64 / calls as f64 / 1e3;
    (first, us)
}

/// A use of `ram` on `device` for the bench: `name`'s line of the chunks listed and live, the
/// use's time and its imports', the first commit after it and a submission's time; each chunk
/// imported by the tests' hook where `import_each`, and the device recovered after where
/// `recover`.
fn measure(device: &Device, name: &str, ram: &Arc<HostRam>, (import_each, recover): (bool, bool)) {
    const TEST: &str = "HOST-RAM-WHOLE";
    let before = device.work_counts();
    let started = Instant::now();
    let used = device.use_host_ram(ram).unwrap();
    let took = started.elapsed();
    let chunks = device.host_ram_chunks();
    if import_each {
        for span in chunks.iter().flat_map(|(_, spans)| spans) {
            device.import_host_ram_chunk(span.start).unwrap();
        }
    }
    let held = device.work_counts();
    let (first, us) = submissions(device);
    println!(
        "{TEST} {name}: listed {:?}, chunks live {}, the use took {:.1} ms (imports {:.1} ms); \
         the first commit after it {first:.1} ms, then a submission {us:.1} us",
        chunks
            .iter()
            .map(|(listed, spans)| (*listed, spans.len()))
            .collect::<Vec<_>>(),
        held.host_ram_chunks_live - before.host_ram_chunks_live,
        took.as_secs_f64() * 1e3,
        (held.host_ram_chunk_import_nanos - before.host_ram_chunk_import_nanos) as f64 / 1e6,
    );
    if recover {
        let before = device.work_counts();
        let started = Instant::now();
        device.recover_for_test().unwrap();
        let took = started.elapsed();
        let after = device.work_counts();
        println!(
            "{TEST} {name}, recovered: the recovery took {:.1} ms, imported {} chunks ahead",
            took.as_secs_f64() * 1e3,
            after.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead,
        );
    }
    drop(used);
}

/// The module docs' bench: a consumer's RAM of each of [`BENCH_BYTES`] held in 512 MiB chunks,
/// then whole, then recovered.
#[test]
#[ignore = "a release benchmark: run by name with --run-ignored all"]
fn whole_ram_bench() {
    const TEST: &str = "HOST-RAM-WHOLE";
    for length in BENCH_BYTES {
        let scratch = Scratch::new();
        let said = Said::default();
        let Some(device) = listed_device(&scratch, &said, TEST, length, 4 * length) else {
            return;
        };
        let gib = length >> 30;
        println!("{TEST} {gib} GiB, device: {}", device.name());
        let (first, us) = submissions(&device);
        println!(
            "{TEST} {gib} GiB, bare: the first commit {first:.1} ms, then a submission {us:.1} us"
        );
        let (memfd, at) = file(length);
        let hole = length - host_page() as u64;
        touch(at, length, Some(hole));
        let largest = udmabuf_size_limit();
        // With the hole: the cap, each chunk imported as copies would ask for it.
        let capped = ram(&memfd, at, length, largest);
        let name = format!("{gib} GiB, a hole, 512 MiB chunks each imported");
        measure(&device, &name, &capped, (true, false));
        drop(capped);
        // Every page resident: imported whole as the use begins. The capped RAM's chunks went
        // with its use; its keep-alive unmapped the first mapping, so the second maps the file
        // again.
        // SAFETY: a new shared mapping of the whole file.
        let mapped = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                memfd.as_raw_fd(),
                0,
            )
        };
        assert_ne!(mapped, libc::MAP_FAILED, "mmap of the file again");
        let again = mapped as u64;
        touch(again + hole, host_page() as u64, None);
        let whole = ram(&memfd, again, length, largest);
        let name = format!("{gib} GiB, resident, imported whole");
        measure(&device, &name, &whole, (false, true));
        drop(whole);
        for line in said.lock().unwrap().iter() {
            println!("{TEST} {gib} GiB, said: {line}");
        }
        drop(device);
    }
}
