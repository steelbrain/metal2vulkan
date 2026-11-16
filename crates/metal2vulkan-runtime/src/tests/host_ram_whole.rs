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
//!
//! A negative control, run by hand: with `listed_chunks_of` answering the cap whatever the
//! residency, the resident RAM takes two chunks and imports none, and its test fails.
//!
//! Where `/dev/udmabuf` cannot be opened, its limit is shorter than the RAM, the device imports no
//! host memory, or the driver refused the RAM's dma-buf (the RAM not listed), the test says so and
//! passes without running.
//!
//! `whole_ram_bench` (ignored, release, run by name) holds a consumer's 8 GiB RAM: with one page a
//! hole, every chunk imported by the tests' hook as copies would ask for them (16 of 512 MiB); then
//! every page resident, imported whole as the use begins. Printed (`HOST-RAM-WHOLE`): the use's
//! time and its imports' time, the chunks live, and a commit's `vkQueueSubmit` time per
//! submission with each set held.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on Linux.

use super::host_residency::{device, host_page, udmabuf, Scratch};
use crate::submit_check::{Check, Verdict};
use crate::{
    Device, DmaBufChunk, DmaBufRequest, DmaBufSource, HostRam, KeepAlive, ResourceOptions,
    WorkCounts,
};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::Arc;
use std::time::Instant;

const MIB: u64 = 1 << 20;
/// The RAM of the device test: past the 512 MiB cap.
const RAM_BYTES: u64 = 640 * MIB;
/// The RAM of the bench: a consumer's guest.
const BENCH_BYTES: u64 = 8 << 30;
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

/// The device the tests take, its import tax forced held and its imports counted against
/// `against` bytes; `None` (said) where `/dev/udmabuf` cannot be opened, its limit is shorter than
/// `length`, or the device imports no host memory.
fn listed_device(scratch: &Scratch, test: &str, length: u64, against: u64) -> Option<Device> {
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
    let device = device(&scratch.0);
    if device.shared().host_import.is_err() {
        eprintln!("{test}: the device imports no host memory here: not run");
        return None;
    }
    (device.shared().submit_checks).force(Check::ImportTax, Verdict::Holds);
    device.count_host_ram_imports_against(against);
    Some(device)
}

/// A use of a [`RAM_BYTES`] RAM, every page written but the one at `hole`: the chunks' lengths,
/// the chunks imported ahead and live once the use returned, and their bytes; `None` (said) where
/// the RAM is not listed.
fn use_of(test: &str, hole: Option<u64>) -> Option<(Vec<u64>, u64, u64, u64)> {
    let scratch = Scratch::new();
    let device = listed_device(&scratch, test, RAM_BYTES, 4 * RAM_BYTES)?;
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
    drop((used, ram));
    Some((lengths, ahead, live, bytes))
}

/// A resident RAM's chunks run past the cap and are imported, all of them, as the use begins.
#[test]
fn a_resident_ram_is_imported_whole_in_chunks_past_the_cap_as_its_use_begins() {
    const TEST: &str = "HOST-RAM-WHOLE resident";
    let Some((lengths, ahead, live, bytes)) = use_of(TEST, None) else {
        return;
    };
    assert_eq!(lengths, [RAM_BYTES], "{TEST}: one chunk of the whole RAM");
    assert_eq!(
        (ahead, live, bytes),
        (1, 1, RAM_BYTES),
        "{TEST}: imported as the use began"
    );
}

/// A RAM with a page the guest never touched keeps the cap, and imports nothing as its use
/// begins.
#[test]
fn a_ram_with_a_hole_keeps_the_cap_and_imports_its_chunks_as_copies_ask() {
    const TEST: &str = "HOST-RAM-WHOLE hole";
    let last = RAM_BYTES - host_page() as u64;
    let Some((lengths, ahead, live, _)) = use_of(TEST, Some(last)) else {
        return;
    };
    assert_eq!(
        lengths,
        [512 * MIB, RAM_BYTES - 512 * MIB],
        "{TEST}: chunks at the cap"
    );
    assert_eq!(
        (ahead, live),
        (0, 0),
        "{TEST}: none imported as the use began"
    );
}

/// A commit's `vkQueueSubmit` time over [`COMMITS`] commits of one 4-byte fill, each waited for:
/// microseconds a queue submission.
fn submit_us(device: &Device) -> f64 {
    let word = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let before = device.work_counts();
    for _ in 0..COMMITS {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&word, 0..4, 1).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer
            .wait_until_completed_for(crate::tests::SETTLE)
            .unwrap();
    }
    let after = device.work_counts();
    let submissions = (after.queue_submissions - before.queue_submissions).max(1);
    (after.batch_submit_nanos - before.batch_submit_nanos) as f64 / submissions as f64 / 1e3
}

/// The module docs' bench: a consumer's 8 GiB RAM held in 512 MiB chunks, then whole.
#[test]
#[ignore = "a release benchmark: run by name with --run-ignored all"]
fn whole_ram_bench() {
    const TEST: &str = "HOST-RAM-WHOLE";
    let scratch = Scratch::new();
    let Some(device) = listed_device(&scratch, TEST, BENCH_BYTES, 4 * BENCH_BYTES) else {
        return;
    };
    println!("{TEST} device: {}", device.name());
    println!("{TEST} bare: a submission {:.1} us", submit_us(&device));
    let (memfd, at) = file(BENCH_BYTES);
    let hole = BENCH_BYTES - host_page() as u64;
    touch(at, BENCH_BYTES, Some(hole));
    let largest = udmabuf_size_limit();
    let measure = |name: &str, ram: &Arc<HostRam>, import_each: bool| {
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
        let us = submit_us(&device);
        println!(
            "{TEST} {name}: listed {:?}, chunks live {}, the use took {:.1} ms (imports {:.1} \
             ms), a submission {us:.1} us",
            chunks
                .iter()
                .map(|(listed, spans)| (*listed, spans.len()))
                .collect::<Vec<_>>(),
            held.host_ram_chunks_live - before.host_ram_chunks_live,
            took.as_secs_f64() * 1e3,
            (held.host_ram_chunk_import_nanos - before.host_ram_chunk_import_nanos) as f64 / 1e6,
        );
        drop(used);
    };
    // With the hole: the cap, each chunk imported as copies would ask for it.
    let capped = ram(&memfd, at, BENCH_BYTES, largest);
    measure("a hole, 512 MiB chunks each imported", &capped, true);
    drop(capped);
    // Every page resident: imported whole as the use begins. The capped RAM's chunks went with
    // its use; its keep-alive unmapped the first mapping, so the second maps the file again.
    // SAFETY: a new shared mapping of the whole file.
    let mapped = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            BENCH_BYTES as usize,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            memfd.as_raw_fd(),
            0,
        )
    };
    assert_ne!(mapped, libc::MAP_FAILED, "mmap of the file again");
    let again = mapped as u64;
    touch(again + hole, host_page() as u64, None);
    let whole = ram(&memfd, again, BENCH_BYTES, largest);
    measure("resident, imported whole", &whole, false);
    drop(whole);
}
