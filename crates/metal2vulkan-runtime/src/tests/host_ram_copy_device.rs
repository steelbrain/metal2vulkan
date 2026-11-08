//! Copies between a guest RAM and a buffer on a device (`host_ram/copy.rs`):
//! `BlitCommandEncoder::copy_from_host_ram` and `copy_to_host_ram`.
//!
//! - **Both ways, byte for byte.** A gather of ascending runs (adjacent ones merged), descending
//!   neighbours, a run across a chunk's end, a run in the second chunk and a page named twice
//!   lands in a `Shared` and a `Private` buffer as the CPU's view of the pages reads; a scatter
//!   back writes each run's bytes and no other byte of the RAM. One dispatch per chunk the runs
//!   touch, and each chunk imported by the copy where no object imported it (the device's
//!   thread held back), the copy's encoding counted as waiting for those imports once, and a
//!   copy over the imported chunks as waiting for none.
//! - **The CPU's writes before the commit** are in a gather encoded before them, and a scatter's
//!   bytes are in the pages once the command buffer completed.
//! - **Ordered against a no-copy buffer over the same pages**, in one command buffer: a buffer
//!   over the RAM's own addresses (bound in its chunk) read after a scatter, in the same encoder
//!   and in a later one, reads the scatter's bytes, and a gather after a fill through it reads
//!   the fill's; in one encoder a barrier is recorded between them.
//! - **A copied backing over the pages** (the import gate refused, the RAM's log given) takes a
//!   scatter's bytes at the next commit, the log told of them; a gather after its merge back
//!   reads what the GPU wrote through it.
//! - **Memory of Metal's private option bit 24 over the pages**, where the system maps no
//!   read-only view of them: a copy, never bound in the chunk, whose commit after the CPU wrote
//!   one page compares that page alone (the RAM's log, `copied_copy_in_bytes_skipped`); the GPU's
//!   writes to it are dropped.
//! - **Runs inside pages**: rows cut to their texels (2046 bytes at a 2048-byte pitch, from 16
//!   bytes into a page), a run ending 2 bytes before a page's end, one across the first chunk's
//!   end and one of 3 bytes, gathered into a `Shared` and a `Private` buffer and scattered back,
//!   one region for each run's bytes in a chunk and no dispatch: every other byte of their pages
//!   (each a pattern) stays. A scatter whose runs name bytes twice leaves the later's; a buffer
//!   over host pages the runs' bytes overlap is refused.
//! - **Runs of words inside pages**: rows of 4-byte texels cut to them (2044 bytes at a 2048-byte
//!   pitch, from 16 bytes into a page), a run ending a word before a page's end, one of three
//!   pages and more, one across the first chunk's end and a word in the second chunk, gathered
//!   into a `Shared` and a `Private` buffer and scattered back by a dispatch per chunk and no
//!   region: every other byte of their pages (each a pattern) stays. A scatter of such runs that
//!   name bytes twice takes regions, and leaves the later's.
//! - **A laid-out surface in one call**: rows at a pitch over scattered pages (two side by side,
//!   one before the first chunk's end, one in the second), each run at its byte of a buffer that
//!   keeps the layout (`copy_runs_from_host_ram`, `copy_runs_to_host_ram`): rows of 4-byte texels
//!   by a dispatch per chunk both ways, rows of 2-byte texels and rows from a byte of the buffer
//!   off a word by regions; the buffer's padding (a sentinel) and every byte of the RAM the runs
//!   do not name stay. A gather of two runs at one byte of the buffer keeps the later's bytes.
//! - **Refused**: runs off a word, empty or outside the RAM, a RAM the device does not use (or no
//!   longer does), bytes past the buffer and an offset off a word; of runs each at a byte of
//!   their own, one past the buffer or at a byte past `u64`; no runs copy nothing.
//! - **A chunk taken as an object takes it**, on a device whose thread imports ahead: a gather
//!   from the RAM's first chunk finds it imported ahead and opens the full window past it, and a
//!   scatter into a chunk past the window imports its page alone while the thread imports the
//!   chunk (on a device with `AliasedImportBlitsUnordered`, the chunk itself); the thread imports
//!   nothing past the window.
//! - **A first touch through the CPU**, on a device whose thread imports ahead: a gather of
//!   pages scattered over two chunks not imported yet, more ranges in each than are imported
//!   alone, is staged: no wait for an import, the CPU's writes after its encoding and before the
//!   commit gathered (its pages read at its batch's submission), and its chunks imported by the
//!   thread next, and nothing else, so the gather after it reads their imports. So is one of few
//!   ranges, which an import of its ranges alone could take: its encoding imports nothing. One
//!   encoded before a buffer over one of its pages was made and written by the GPU in a command
//!   buffer committed before it waits for that write (its fill gated), and reads it; one behind a
//!   shared event reads the CPU's writes made before the signal.
//!
//! Every other test holds the device's thread back (`Device::hold_host_ram_imports`).
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device that imports no host memory says so and passes.

use crate::host_ram::AHEAD_CHUNKS;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, Device, DeviceKind, DeviceOptions, DirtyPages,
    DirtyStamp, Error, EventRef, HostBacking, HostMemory, HostRam, ImportGate, KeepAlive, Problem,
    ResourceOptions, RunResolver, WorkCounts,
};
use std::num::NonZeroU64;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(60);
const MIB: u64 = 1 << 20;
const PAGE: u64 = 4096;
/// The RAM's bytes: one whole chunk and a short second.
const RAM_BYTES: u64 = 64 * MIB + MIB;
/// The first page of the second chunk.
const SECOND: u64 = 64 * MIB / PAGE;

/// Anonymous memory standing in for a guest's RAM, unmapped by the keep-alive it is given to.
struct Mapping {
    start: u64,
    length: u64,
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the mapping this test made, which no device imports any more: its keep-alive
        // is dropped after the last chunk over it is freed.
        unsafe { libc::munmap(self.start as *mut libc::c_void, self.length as usize) };
    }
}

/// A RAM of `RAM_BYTES`, its writes logged to `log` where given, and its first byte.
fn new_ram(log: Option<Arc<Log>>) -> (Arc<HostRam>, u64) {
    ram_of(RAM_BYTES, log)
}

/// A RAM of `length` bytes, its writes logged to `log` where given, and its first byte.
fn ram_of(length: u64, log: Option<Arc<Log>>) -> (Arc<HostRam>, u64) {
    // SAFETY: a fresh private anonymous mapping.
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
    let start = start as u64;
    let keep: KeepAlive = Box::new(Mapping { start, length });
    let at = NonNull::new(start as *mut u8).unwrap();
    // SAFETY: the mapping is readable and writable until the keep-alive unmaps it.
    let ram = unsafe { HostRam::new(&[(at, length)], Some(keep)) }.unwrap();
    let ram = match log {
        Some(log) => ram.with_dirty_pages(log),
        None => ram,
    };
    (Arc::new(ram), start)
}

/// A device of the cfg's kind whose uses of a RAM start no thread importing its chunks ahead
/// (`Device::hold_host_ram_imports`): each chunk is imported by the first copy or object over
/// it. `None` where it imports no host memory.
fn device() -> Option<Device> {
    let device = ahead_device()?;
    device.hold_host_ram_imports();
    Some(device)
}

/// A device of the cfg's kind, as the product makes it, of this call's own: a pipeline-records
/// hook of its own, which keeps nothing, makes its options unequal to every other call's, so tests
/// run as threads of one process do not share a device, nor its counts
/// (`Device::system_default_with` answers the live device of equal options, a hook compared by
/// identity). `None` where it imports no host memory.
fn ahead_device() -> Option<Device> {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        pipeline_records: Some(Arc::new(|_: crate::PipelineRecord| ())),
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    if device.shared().host_import.is_err() {
        eprintln!("HOST-RAM-COPY: the device imports no host memory here; nothing to test");
        return None;
    }
    Some(device)
}

/// The byte `seed`'s pattern holds at host address `address`: the top byte of a product, which
/// every bit of the address moves, so no two pages hold the same bytes.
fn pattern(address: u64, seed: u64) -> u8 {
    let mixed = (address ^ seed.wrapping_mul(0x9e37_79b9)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    (mixed >> 56) as u8
}

/// Write `seed`'s pattern over `range` of host addresses, while no command buffer runs.
fn write_pattern(range: Range<u64>, seed: u64) {
    for address in range {
        // SAFETY: bytes of the test's RAM, which no command buffer uses now.
        unsafe { *(address as *mut u8) = pattern(address, seed) };
    }
}

/// The bytes of `range` of host addresses, while no command buffer writes them.
fn bytes(range: Range<u64>) -> Vec<u8> {
    let length = (range.end - range.start) as usize;
    // SAFETY: as above.
    unsafe { std::slice::from_raw_parts(range.start as *const u8, length) }.to_vec()
}

/// A `Shared` buffer of `length` bytes holding `fill`.
fn shared(device: &Device, length: u64, fill: impl Fn(u64) -> u8) -> Buffer {
    let buffer = device
        .new_buffer(length, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().unwrap();
    for at in 0..length {
        // SAFETY: the buffer's bytes, which no command uses yet.
        unsafe { *contents.as_ptr().add(at as usize) = fill(at) };
    }
    buffer
}

/// A `Shared` buffer's bytes, while no command writes them.
fn contents(buffer: &Buffer) -> Vec<u8> {
    let contents = buffer.contents().unwrap();
    // SAFETY: as above.
    unsafe { std::slice::from_raw_parts(contents.as_ptr(), buffer.length() as usize) }.to_vec()
}

/// One command buffer `encode` fills, committed (after `before`) and waited for.
fn run(device: &Device, encode: impl FnOnce(&CommandBuffer), before: impl FnOnce()) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    encode(&command_buffer);
    before();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// One command buffer of one blit encoder `encode` fills, committed and waited for.
fn blit(device: &Device, encode: impl FnOnce(&mut crate::BlitCommandEncoder)) {
    run(
        device,
        |command_buffer| {
            let mut encoder = command_buffer.blit_command_encoder().unwrap();
            encode(&mut encoder);
            encoder.end_encoding().unwrap();
        },
        || {},
    );
}

/// The pages are their own run of the RAM (the test's buffers lie over the RAM itself).
fn itself() -> RunResolver {
    Arc::new(|start, length| Some(std::iter::once(start..start + length).collect()))
}

/// The host's page: a resolver's runs are whole ones.
fn host_page() -> u64 {
    // SAFETY: a name the call takes.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as u64 }
}

/// The pages as two runs of the RAM: their first host page, then the rest.
fn two_runs() -> RunResolver {
    let page = host_page();
    Arc::new(move |start, length| Some(vec![start..start + page, start + page..start + length]))
}

/// A no-copy buffer of `length` bytes at host address `at`, resolved by `runs`, its import
/// admitted or refused by its gate as `admit` says.
fn over_ram(device: &Device, at: u64, length: u64, runs: RunResolver, admit: bool) -> Buffer {
    let gate: ImportGate = Arc::new(move |_| admit);
    // SAFETY: the RAM's bytes, mapped until its keep-alive goes, after every device let go.
    let memory = unsafe { HostMemory::new(NonNull::new(at as *mut u8).unwrap()) }
        .with_runs(runs)
        .with_import_gate(gate);
    // SAFETY: as above.
    unsafe {
        device.new_buffer_with_bytes_no_copy(memory, length, ResourceOptions::default(), None)
    }
    .unwrap()
}

/// One run, as a list.
fn one(range: Range<u64>) -> Vec<Range<u64>> {
    std::iter::once(range).collect()
}

/// The bytes `runs` give in order, as the CPU reads them.
fn gathered(runs: &[Range<u64>]) -> Vec<u8> {
    runs.iter().flat_map(|run| bytes(run.clone())).collect()
}

/// Whether `result` is the refusal `Invalid` of `problem`'s kind.
fn refused(result: Result<(), Error>, host_ram: bool) -> bool {
    match result {
        Err(Error::Invalid(invalid)) if host_ram => invalid.problem == Problem::HostRam,
        Err(Error::Invalid(invalid)) => {
            matches!(invalid.problem, Problem::RangeOutOfBounds { .. })
        }
        _ => false,
    }
}

#[test]
fn a_gather_and_a_scatter_move_every_runs_bytes_in_order() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let _used = device.use_host_ram(&ram).unwrap();
    let page = |n: u64| start + n * PAGE;
    let runs = vec![
        // Two ascending neighbours: one run.
        page(10)..page(11),
        page(11)..page(13),
        // Descending neighbours: three runs.
        page(40)..page(41),
        page(39)..page(40),
        page(38)..page(39),
        // Across the first chunk's end.
        page(SECOND - 1)..page(SECOND + 1),
        // In the second chunk, then back in the first.
        page(SECOND + 100)..page(SECOND + 102),
        page(3)..page(4),
    ];
    for run in &runs {
        write_pattern(run.clone(), 1);
    }
    let mut twice = runs.clone();
    // A page named twice, which a gather reads twice.
    twice.push(page(10)..page(11));
    let total: u64 = twice.iter().map(|run| run.end - run.start).sum();
    let before = device.work_counts();
    // Into a `Shared` buffer from byte 260: its bytes before and after stay.
    let out = shared(&device, 260 + total + 64, |_| 0xee);
    blit(&device, |encoder| {
        encoder.copy_from_host_ram(&ram, &twice, &out, 260).unwrap();
    });
    let got = contents(&out);
    assert!(got[..260].iter().all(|&byte| byte == 0xee));
    assert!(got[260 + total as usize..].iter().all(|&byte| byte == 0xee));
    assert!(
        got[260..260 + total as usize] == gathered(&twice)[..],
        "the gathered bytes"
    );
    let after = device.work_counts();
    assert_eq!(after.host_ram_copies - before.host_ram_copies, 1);
    assert_eq!(
        after.host_ram_copy_bytes - before.host_ram_copy_bytes,
        total
    );
    assert_eq!(
        after.host_ram_copy_dispatches - before.host_ram_copy_dispatches,
        2,
        "one dispatch per chunk the runs touch"
    );
    assert_eq!(
        after.host_ram_copy_regions - before.host_ram_copy_regions,
        0,
        "whole pages take no regions"
    );
    assert_eq!(
        after.host_ram_chunks_imported - before.host_ram_chunks_imported,
        2,
        "the copy imported the chunks no object had"
    );
    // The copy's encoding waited for those imports, which it made itself (the thread held back),
    // and the wait is counted once for the copy, with their time.
    assert_eq!(
        after.host_ram_copy_import_waits - before.host_ram_copy_import_waits,
        1
    );
    let waited = after.host_ram_copy_import_wait_nanos - before.host_ram_copy_import_wait_nanos;
    let took = after.host_ram_chunk_import_nanos - before.host_ram_chunk_import_nanos;
    assert!(
        took > 0 && waited >= took,
        "{waited} ns waited, {took} ns importing"
    );
    // Into a `Private` buffer, read back by a blit.
    let private = device
        .new_buffer(total, ResourceOptions::private())
        .unwrap();
    let back = shared(&device, total, |_| 0);
    let before = device.work_counts();
    blit(&device, |encoder| {
        encoder
            .copy_from_host_ram(&ram, &twice, &private, 0)
            .unwrap();
        encoder
            .copy_from_buffer_to_buffer(&private, 0, &back, 0, total)
            .unwrap();
    });
    assert!(
        contents(&back) == gathered(&twice),
        "through a private buffer"
    );
    // The chunks imported, the copy waited for none.
    let after = device.work_counts();
    assert_eq!(
        after.host_ram_copy_import_waits - before.host_ram_copy_import_waits,
        0
    );
    assert_eq!(
        after.host_ram_chunk_import_nanos - before.host_ram_chunk_import_nanos,
        0
    );
    // A scatter back of new bytes, from byte 64 of its source: each run takes its bytes in
    // order, and no other byte of the RAM changes.
    let scattered: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let source = shared(&device, 64 + scattered, |at| pattern(at, 2));
    let mut want = bytes(start..start + RAM_BYTES);
    let mut at = 64;
    for run in &runs {
        for address in run.clone() {
            want[(address - start) as usize] = pattern(at, 2);
            at += 1;
        }
    }
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&source, 64, &ram, &runs).unwrap();
    });
    assert!(
        bytes(start..start + RAM_BYTES) == want,
        "the scattered bytes"
    );
}

/// A surface's rows cut to their texels, 2046 bytes at a 2048-byte pitch from 16 bytes into a
/// page (so that one crosses a page's end), a run ending 2 bytes before a page's end, one across
/// the first chunk's end and one of 3 bytes in the second chunk.
fn runs_inside_pages(start: u64) -> Vec<Range<u64>> {
    let mut runs: Vec<Range<u64>> = (0..6)
        .map(|y| {
            let row = start + 20 * PAGE + 16 + 2048 * y;
            row..row + 2046
        })
        .collect();
    runs.push(start + 30 * PAGE + 100..start + 31 * PAGE - 2);
    runs.push(start + (SECOND - 1) * PAGE + 8..start + SECOND * PAGE + 1001);
    runs.push(start + (SECOND + 5) * PAGE + 4..start + (SECOND + 5) * PAGE + 7);
    runs
}

#[test]
fn runs_inside_pages_move_their_bytes_alone_and_leave_the_rest_of_their_pages() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let _used = device.use_host_ram(&ram).unwrap();
    let runs = runs_inside_pages(start);
    // Every byte of the pages the runs touch holds a pattern: those around the runs are the
    // sentinels, a row's padding and the bytes before and after the surface.
    let touched = [
        start + 20 * PAGE..start + 24 * PAGE,
        start + 30 * PAGE..start + 31 * PAGE,
        start + (SECOND - 1) * PAGE..start + (SECOND + 1) * PAGE,
        start + (SECOND + 5) * PAGE..start + (SECOND + 6) * PAGE,
    ];
    for pages in &touched {
        write_pattern(pages.clone(), 7);
    }
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    // A gather into a `Shared` buffer from byte 260: its bytes before and after stay.
    let before = device.work_counts();
    let out = shared(&device, 260 + total + 64, |_| 0xee);
    blit(&device, |encoder| {
        encoder.copy_from_host_ram(&ram, &runs, &out, 260).unwrap();
    });
    let got = contents(&out);
    assert!(got[..260].iter().all(|&byte| byte == 0xee));
    assert!(got[260 + total as usize..].iter().all(|&byte| byte == 0xee));
    assert!(
        got[260..260 + total as usize] == gathered(&runs)[..],
        "the gathered bytes"
    );
    let after = device.work_counts();
    assert_eq!(after.host_ram_copies - before.host_ram_copies, 1);
    assert_eq!(
        after.host_ram_copy_bytes - before.host_ram_copy_bytes,
        total
    );
    assert_eq!(
        after.host_ram_copy_regions - before.host_ram_copy_regions,
        10,
        "a region for each run's bytes in a chunk: nine runs, one across the chunks' end"
    );
    assert_eq!(
        after.host_ram_copy_dispatches - before.host_ram_copy_dispatches,
        0,
        "no pages copy"
    );
    // Into a `Private` buffer, read back by a blit.
    let private = device
        .new_buffer(total, ResourceOptions::private())
        .unwrap();
    let back = shared(&device, total, |_| 0);
    blit(&device, |encoder| {
        encoder
            .copy_from_host_ram(&ram, &runs, &private, 0)
            .unwrap();
        encoder
            .copy_from_buffer_to_buffer(&private, 0, &back, 0, total)
            .unwrap();
    });
    assert!(
        contents(&back) == gathered(&runs),
        "through a private buffer"
    );
    // A scatter back of new bytes, from byte 64 of its source: each run takes its bytes in order,
    // and no other byte of the RAM changes, the sentinels in the runs' pages among them.
    let source = shared(&device, 64 + total, |at| pattern(at, 2));
    let mut want = bytes(start..start + RAM_BYTES);
    let mut at = 64;
    for run in &runs {
        for address in run.clone() {
            want[(address - start) as usize] = pattern(at, 2);
            at += 1;
        }
    }
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&source, 64, &ram, &runs).unwrap();
    });
    let now = bytes(start..start + RAM_BYTES);
    for pages in &touched {
        let (from, to) = ((pages.start - start) as usize, (pages.end - start) as usize);
        assert!(
            now[from..to] == want[from..to],
            "the runs' pages {pages:x?}"
        );
    }
    assert!(
        now == want,
        "the scattered bytes, and no other byte of the RAM"
    );
    // A scatter whose runs name bytes twice: the later run's bytes are the ones left.
    let twice = [
        start + 40 * PAGE + 16..start + 40 * PAGE + 100,
        start + 40 * PAGE + 96..start + 40 * PAGE + 202,
    ];
    let length: u64 = twice.iter().map(|run| run.end - run.start).sum();
    let source = shared(&device, length, |at| pattern(at, 3));
    let mut want = bytes(start + 40 * PAGE..start + 41 * PAGE);
    let mut at = 0;
    for run in &twice {
        for address in run.clone() {
            want[(address - start - 40 * PAGE) as usize] = pattern(at, 3);
            at += 1;
        }
    }
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&source, 0, &ram, &twice).unwrap();
    });
    assert!(
        bytes(start + 40 * PAGE..start + 41 * PAGE) == want,
        "the later of two runs over the same bytes"
    );
    // A buffer over host pages the runs' bytes overlap is refused: no region reads and writes
    // one byte. One over other pages is not.
    let over = over_ram(&device, start + 50 * PAGE, PAGE, itself(), true);
    let inside = one(start + 50 * PAGE + 16..start + 50 * PAGE + 63);
    let mut encoder_result = None;
    run(
        &device,
        |command_buffer| {
            let mut encoder = command_buffer.blit_command_encoder().unwrap();
            encoder_result = Some(encoder.copy_from_host_ram(&ram, &inside, &over, 0));
            let apart = one(start + 60 * PAGE + 16..start + 60 * PAGE + 64);
            encoder.copy_from_host_ram(&ram, &apart, &over, 0).unwrap();
            encoder.end_encoding().unwrap();
        },
        || {},
    );
    assert!(
        refused(encoder_result.unwrap(), true),
        "a buffer over the runs' own bytes"
    );
}

/// The rows of a surface of 4-byte texels cut to them, 2044 bytes at a 2048-byte pitch from 16
/// bytes into a page (so that one crosses a page's end), a run ending a word before a page's end,
/// one of three pages and more from inside a page, one across the first chunk's end and one of a
/// word in the second chunk: every start and end on a word.
fn runs_of_words_inside_pages(start: u64) -> Vec<Range<u64>> {
    let mut runs: Vec<Range<u64>> = (0..6)
        .map(|y| {
            let row = start + 20 * PAGE + 16 + 2048 * y;
            row..row + 2044
        })
        .collect();
    runs.push(start + 30 * PAGE + 100..start + 31 * PAGE - 4);
    runs.push(start + 33 * PAGE + 12..start + 36 * PAGE + 40);
    runs.push(start + (SECOND - 1) * PAGE + 8..start + SECOND * PAGE + 1000);
    runs.push(start + (SECOND + 5) * PAGE + 4..start + (SECOND + 5) * PAGE + 8);
    runs
}

#[test]
fn runs_of_words_inside_pages_take_a_dispatch_per_chunk_and_leave_the_rest_of_their_pages() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let _used = device.use_host_ram(&ram).unwrap();
    let runs = runs_of_words_inside_pages(start);
    // Every byte of the pages the runs touch holds a pattern: those around the runs are the
    // sentinels, a row's padding and the bytes before and after the surface.
    let touched = [
        start + 20 * PAGE..start + 24 * PAGE,
        start + 30 * PAGE..start + 31 * PAGE,
        start + 33 * PAGE..start + 37 * PAGE,
        start + (SECOND - 1) * PAGE..start + (SECOND + 1) * PAGE,
        start + (SECOND + 5) * PAGE..start + (SECOND + 6) * PAGE,
    ];
    for pages in &touched {
        write_pattern(pages.clone(), 8);
    }
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    // A gather into a `Shared` buffer from byte 260: its bytes before and after stay.
    let before = device.work_counts();
    let out = shared(&device, 260 + total + 64, |_| 0xee);
    blit(&device, |encoder| {
        encoder.copy_from_host_ram(&ram, &runs, &out, 260).unwrap();
    });
    let got = contents(&out);
    assert!(got[..260].iter().all(|&byte| byte == 0xee));
    assert!(got[260 + total as usize..].iter().all(|&byte| byte == 0xee));
    assert!(
        got[260..260 + total as usize] == gathered(&runs)[..],
        "the gathered bytes"
    );
    let after = device.work_counts();
    assert_eq!(after.host_ram_copies - before.host_ram_copies, 1);
    assert_eq!(
        after.host_ram_copy_bytes - before.host_ram_copy_bytes,
        total
    );
    assert_eq!(
        after.host_ram_copy_dispatches - before.host_ram_copy_dispatches,
        2,
        "one dispatch per chunk the runs touch"
    );
    assert_eq!(
        after.host_ram_copy_regions - before.host_ram_copy_regions,
        0,
        "runs of words take no regions"
    );
    // Into a `Private` buffer, read back by a blit.
    let private = device
        .new_buffer(total, ResourceOptions::private())
        .unwrap();
    let back = shared(&device, total, |_| 0);
    blit(&device, |encoder| {
        encoder
            .copy_from_host_ram(&ram, &runs, &private, 0)
            .unwrap();
        encoder
            .copy_from_buffer_to_buffer(&private, 0, &back, 0, total)
            .unwrap();
    });
    assert!(
        contents(&back) == gathered(&runs),
        "through a private buffer"
    );
    // A scatter back of new bytes, from byte 64 of its source: each run takes its bytes in order,
    // and no other byte of the RAM changes, the sentinels in the runs' pages among them.
    let source = shared(&device, 64 + total, |at| pattern(at, 2));
    let mut want = bytes(start..start + RAM_BYTES);
    let mut at = 64;
    for run in &runs {
        for address in run.clone() {
            want[(address - start) as usize] = pattern(at, 2);
            at += 1;
        }
    }
    let before = device.work_counts();
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&source, 64, &ram, &runs).unwrap();
    });
    let after = device.work_counts();
    let scatter = (
        after.host_ram_copy_dispatches - before.host_ram_copy_dispatches,
        after.host_ram_copy_regions - before.host_ram_copy_regions,
    );
    assert_eq!(
        scatter,
        (2, 0),
        "the scatter: a dispatch per chunk, no region"
    );
    let now = bytes(start..start + RAM_BYTES);
    for pages in &touched {
        let (from, to) = ((pages.start - start) as usize, (pages.end - start) as usize);
        assert!(
            now[from..to] == want[from..to],
            "the runs' pages {pages:x?}"
        );
    }
    assert!(
        now == want,
        "the scattered bytes, and no other byte of the RAM"
    );
    // A scatter of runs of words that name bytes twice: copied by regions, a round each, so the
    // later run's bytes are the ones left.
    let twice = [
        start + 40 * PAGE + 16..start + 40 * PAGE + 100,
        start + 40 * PAGE + 96..start + 40 * PAGE + 200,
    ];
    let length: u64 = twice.iter().map(|run| run.end - run.start).sum();
    let source = shared(&device, length, |at| pattern(at, 3));
    let mut want = bytes(start + 40 * PAGE..start + 41 * PAGE);
    let mut at = 0;
    for run in &twice {
        for address in run.clone() {
            want[(address - start - 40 * PAGE) as usize] = pattern(at, 3);
            at += 1;
        }
    }
    let before = device.work_counts();
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&source, 0, &ram, &twice).unwrap();
    });
    let after = device.work_counts();
    assert!(
        bytes(start + 40 * PAGE..start + 41 * PAGE) == want,
        "the later of two runs over the same bytes"
    );
    assert_eq!(
        after.host_ram_copy_regions - before.host_ram_copy_regions,
        2,
        "a scatter that writes a byte twice: a region a run, in rounds"
    );
}

/// The pages of a laid-out surface: its 4 KiB pages lie apart in the RAM, two of them side by
/// side, one before the first chunk's end and one in the second chunk.
const SURFACE_PAGES: [u64; 8] = [30, 12, 13, SECOND - 1, SECOND + 2, 7, 50, 51];

/// A laid-out surface's rows of `row_bytes` texel bytes at a `pitch`, over [`SURFACE_PAGES`], as
/// runs each at its byte of a buffer that keeps the layout from byte `base`: row `r` at
/// `base + r × pitch`, cut where a page of the surface ends (the next lies elsewhere in the RAM).
fn laid_out(start: u64, (pitch, row_bytes): (u64, u64), base: u64) -> Vec<(u64, Range<u64>)> {
    let surface = SURFACE_PAGES.len() as u64 * PAGE;
    let mut runs = Vec::new();
    for row in 0..surface / pitch {
        let mut at = row * pitch;
        let end = at + row_bytes;
        while at < end {
            let cut = end.min((at / PAGE + 1) * PAGE);
            let host = start + SURFACE_PAGES[(at / PAGE) as usize] * PAGE + at % PAGE;
            runs.push((base + at, host..host + (cut - at)));
            at = cut;
        }
    }
    runs
}

/// One laid-out download and upload, each one call over `runs` (at their bytes from `base` of a
/// buffer of `length` bytes): the buffer's bytes between the runs (a row's padding, each a
/// sentinel) and every byte of the RAM the runs do not name stay as they were. The calls'
/// dispatches and regions.
fn laid_out_both_ways(
    device: &Device,
    ram: &HostRam,
    start: u64,
    runs: &[(u64, Range<u64>)],
    length: u64,
) -> [(u64, u64); 2] {
    let counted = |before: &crate::WorkCounts, after: &crate::WorkCounts| {
        (
            after.host_ram_copy_dispatches - before.host_ram_copy_dispatches,
            after.host_ram_copy_regions - before.host_ram_copy_regions,
        )
    };
    for &page in &SURFACE_PAGES {
        write_pattern(start + page * PAGE..start + (page + 1) * PAGE, 11);
    }
    // The gather: each run's bytes at its byte of the buffer, every other byte of it its
    // sentinel.
    let out = shared(device, length, |_| 0xee);
    let mut want = vec![0xee; length as usize];
    for (at, run) in runs {
        let at = *at as usize;
        want[at..at + (run.end - run.start) as usize].copy_from_slice(&bytes(run.clone()));
    }
    let before = device.work_counts();
    blit(device, |encoder| {
        encoder.copy_runs_from_host_ram(ram, runs, &out).unwrap();
    });
    let gather = counted(&before, &device.work_counts());
    assert!(
        contents(&out) == want,
        "the rows at their bytes, the padding's sentinels kept"
    );
    // The scatter: each run takes the bytes at its byte of the source, and no other byte of the
    // RAM changes, the rows' padding in their pages among them.
    let source = shared(device, length, |at| pattern(at, 4));
    let mut want = bytes(start..start + RAM_BYTES);
    for (at, run) in runs {
        for (i, address) in run.clone().enumerate() {
            want[(address - start) as usize] = pattern(at + i as u64, 4);
        }
    }
    let before = device.work_counts();
    blit(device, |encoder| {
        encoder.copy_runs_to_host_ram(&source, ram, runs).unwrap();
    });
    let scatter = counted(&before, &device.work_counts());
    let now = bytes(start..start + RAM_BYTES);
    for &page in &SURFACE_PAGES {
        let at = (page * PAGE) as usize;
        assert!(
            now[at..at + PAGE as usize] == want[at..at + PAGE as usize],
            "the surface's page {page:x}"
        );
    }
    assert!(now == want, "the rows alone, and no other byte of the RAM");
    [gather, scatter]
}

#[test]
fn a_laid_out_surface_copies_in_one_call_each_row_at_its_byte_of_the_buffer() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let _used = device.use_host_ram(&ram).unwrap();
    let length = 64 + SURFACE_PAGES.len() as u64 * PAGE + 64;
    // Rows of 375 4-byte texels at a 1536-byte pitch, from byte 64 of the buffer: words, so a
    // dispatch for each chunk the rows touch, no region, both ways.
    let rows = laid_out(start, (1536, 1500), 64);
    assert!(rows.len() > 21, "rows cut where a page of the surface ends");
    let counts = laid_out_both_ways(&device, &ram, start, &rows, length);
    assert_eq!(
        counts,
        [(2, 0); 2],
        "a dispatch per chunk, no region, both ways"
    );
    // Rows of 751 2-byte texels: each ends 2 bytes past a word, so a region a run's bytes in a
    // chunk, no dispatch.
    let rows = laid_out(start, (1536, 1502), 64);
    let counts = laid_out_both_ways(&device, &ram, start, &rows, length);
    for (dispatches, regions) in counts {
        assert_eq!(dispatches, 0, "no dispatch");
        assert!(regions >= 21, "a region a run: {regions}");
    }
    // Rows of 4-byte texels from a byte of the buffer off a word: regions.
    let rows = laid_out(start, (1536, 1500), 66);
    let counts = laid_out_both_ways(&device, &ram, start, &rows, length);
    for (dispatches, regions) in counts {
        assert_eq!(dispatches, 0, "no dispatch off a word of the buffer");
        assert!(regions >= 21, "a region a run: {regions}");
    }
    // A gather of two runs at one byte of the buffer: regions, a round each, so the later's
    // bytes are the ones left.
    let page = |n: u64| start + n * PAGE;
    write_pattern(page(60)..page(62), 12);
    let twice = [
        (16, page(60)..page(60) + 400),
        (16, page(61) + 8..page(61) + 208),
    ];
    let out = shared(&device, 1024, |_| 0xee);
    let before = device.work_counts();
    blit(&device, |encoder| {
        encoder.copy_runs_from_host_ram(&ram, &twice, &out).unwrap();
    });
    let after = device.work_counts();
    let got = contents(&out);
    assert!(
        got[16..216] == bytes(page(61) + 8..page(61) + 208)[..],
        "the later run's"
    );
    assert!(
        got[216..416] == bytes(page(60) + 200..page(60) + 400)[..],
        "the earlier's rest"
    );
    assert!(got[..16]
        .iter()
        .chain(&got[416..])
        .all(|&byte| byte == 0xee));
    assert_eq!(
        after.host_ram_copy_regions - before.host_ram_copy_regions,
        2,
        "a region a run, in two rounds"
    );
    assert_eq!(
        after.host_ram_copy_dispatches - before.host_ram_copy_dispatches,
        0
    );
}

#[test]
fn the_cpus_writes_before_the_commit_are_gathered_and_a_scatter_is_in_the_pages_after() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let _used = device.use_host_ram(&ram).unwrap();
    let page = |n: u64| start + n * PAGE;
    let runs = vec![
        page(7)..page(9),
        page(5)..page(6),
        page(SECOND + 3)..page(SECOND + 4),
    ];
    write_pattern(page(0)..page(20), 3);
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let out = shared(&device, total, |_| 0);
    run(
        &device,
        |command_buffer| {
            let mut encoder = command_buffer.blit_command_encoder().unwrap();
            encoder.copy_from_host_ram(&ram, &runs, &out, 0).unwrap();
            encoder.end_encoding().unwrap();
        },
        // The CPU writes the pages after the copy was encoded, before the commit.
        || {
            for run in &runs {
                write_pattern(run.clone(), 4);
            }
        },
    );
    assert!(
        contents(&out) == gathered(&runs),
        "the CPU's writes before the commit"
    );
    let source = shared(&device, total, |at| pattern(at, 5));
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&source, 0, &ram, &runs).unwrap();
    });
    assert!(
        gathered(&runs) == contents(&source),
        "the GPU's writes after completion"
    );
}

#[test]
fn copies_are_ordered_against_a_no_copy_buffer_over_the_same_pages() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let _used = device.use_host_ram(&ram).unwrap();
    let page = |n: u64| start + n * PAGE;
    const PAGES: u64 = 4;
    let length = PAGES * PAGE;
    // A no-copy buffer over pages 100 to 103 of the RAM itself, bound in its chunk.
    let over = over_ram(&device, page(100), length, itself(), true);
    assert_eq!(over.host_backing(), Some(HostBacking::InHostRam));
    // The runs over the same pages, descending.
    let runs: Vec<Range<u64>> = (100..100 + PAGES)
        .rev()
        .map(|n| page(n)..page(n + 1))
        .collect();
    // The runs' bytes in the buffer's order: page 103 first.
    let in_buffer = |copied: &[u8]| -> Vec<u8> {
        (0..PAGES as usize)
            .rev()
            .flat_map(|i| copied[i * PAGE as usize..(i + 1) * PAGE as usize].to_vec())
            .collect()
    };
    for (seed, apart) in [(6, false), (7, true)] {
        // A later reader: a scatter, then a copy out of the buffer over the same pages.
        let source = shared(&device, length, |at| pattern(at, seed));
        let out = shared(&device, length, |_| 0);
        let barriers = device.work_counts().command_barriers;
        run(
            &device,
            |command_buffer| {
                let mut encoder = command_buffer.blit_command_encoder().unwrap();
                encoder.copy_to_host_ram(&source, 0, &ram, &runs).unwrap();
                if apart {
                    encoder.end_encoding().unwrap();
                    encoder = command_buffer.blit_command_encoder().unwrap();
                }
                encoder
                    .copy_from_buffer_to_buffer(&over, 0, &out, 0, length)
                    .unwrap();
                encoder.end_encoding().unwrap();
            },
            || {},
        );
        assert!(
            contents(&out) == in_buffer(&contents(&source)),
            "a later reader sees the scatter (apart: {apart})"
        );
        if !apart {
            assert!(
                device.work_counts().command_barriers > barriers,
                "a barrier between the scatter and the read"
            );
        }
        // An earlier writer: a fill through the buffer, then a gather of the same pages.
        let value = 0xa0 + seed as u8;
        let out = shared(&device, length, |_| 0);
        let barriers = device.work_counts().command_barriers;
        run(
            &device,
            |command_buffer| {
                let mut encoder = command_buffer.blit_command_encoder().unwrap();
                encoder.fill_buffer(&over, 0..length, value).unwrap();
                if apart {
                    encoder.end_encoding().unwrap();
                    encoder = command_buffer.blit_command_encoder().unwrap();
                }
                encoder.copy_from_host_ram(&ram, &runs, &out, 0).unwrap();
                encoder.end_encoding().unwrap();
            },
            || {},
        );
        assert!(
            contents(&out).iter().all(|&byte| byte == value),
            "a gather sees an earlier writer (apart: {apart})"
        );
        if !apart {
            assert!(
                device.work_counts().command_barriers > barriers,
                "a barrier between the fill and the gather"
            );
        }
    }
}

/// The test's log of a RAM's writes: every write it is told of, stamped, with its writer (0 the
/// consumer's).
#[derive(Default)]
struct Log {
    next: AtomicU64,
    writes: Mutex<Vec<(u64, Range<u64>, u64)>>,
}

impl Log {
    fn told(&self, range: Range<u64>, writer: u64) {
        let stamp = self.next.fetch_add(1, Ordering::SeqCst) + 1;
        self.writes.lock().unwrap().push((stamp, range, writer));
    }

    /// Whether a write by `writer` covering `range` was told.
    fn heard(&self, range: &Range<u64>, writer: u64) -> bool {
        let writes = self.writes.lock().unwrap();
        (writes.iter()).any(|w| w.2 == writer && w.1.start <= range.start && range.end <= w.1.end)
    }
}

impl DirtyPages for Log {
    type Watch = ();

    fn watch(&self, _range: Range<u64>) {}

    fn request(&self) -> DirtyStamp {
        DirtyStamp(self.next.fetch_add(1, Ordering::SeqCst) + 1)
    }

    fn wait(&self, _stamp: DirtyStamp) -> bool {
        true
    }

    fn changed(&self, since: DirtyStamp, out: &mut Vec<Range<u64>>) {
        let writes = self.writes.lock().unwrap();
        out.extend(
            (writes.iter())
                .filter(|w| w.0 > since.0)
                .map(|w| w.1.clone()),
        );
    }

    fn wrote(&self, range: Range<u64>) {
        self.told(range, 0);
    }

    fn wrote_as(&self, range: Range<u64>, writer: NonZeroU64) {
        self.told(range, writer.get());
    }

    fn changed_except(&self, since: DirtyStamp, writer: NonZeroU64, out: &mut Vec<Range<u64>>) {
        let writes = self.writes.lock().unwrap();
        let changed = (writes.iter()).filter(|w| w.0 > since.0 && w.2 != writer.get());
        out.extend(changed.map(|w| w.1.clone()));
    }
}

#[test]
fn a_copied_backing_over_the_pages_takes_a_scatter_at_the_next_commit() {
    let Some(device) = device() else {
        return;
    };
    let log = Arc::new(Log::default());
    let (ram, start) = new_ram(Some(Arc::clone(&log)));
    let _used = device.use_host_ram(&ram).unwrap();
    let page = |n: u64| start + n * PAGE;
    // Two host pages, from a multiple of one.
    let length = 2 * host_page();
    let at = page(200).next_multiple_of(host_page());
    write_pattern(at..at + length, 8);
    // Two runs of the RAM, its import refused: a copy of the pages, logged by the RAM's log.
    let copied = over_ram(&device, at, length, two_runs(), false);
    assert_eq!(copied.host_backing(), Some(HostBacking::Copied));
    let runs = one(at..at + length);
    let source = shared(&device, length, |at| pattern(at, 9));
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&source, 0, &ram, &runs).unwrap();
    });
    assert!(
        log.heard(&runs[0], 0),
        "the scatter's writes told to the RAM's log"
    );
    let out = shared(&device, length, |_| 0);
    blit(&device, |encoder| {
        encoder
            .copy_from_buffer_to_buffer(&copied, 0, &out, 0, length)
            .unwrap();
    });
    assert!(
        contents(&out) == contents(&source),
        "the copy took the scatter's bytes"
    );
    // The GPU writes through the copy; its merge back puts the bytes in the pages, which a
    // gather in a later command buffer reads.
    blit(&device, |encoder| {
        encoder.fill_buffer(&copied, 0..length, 0x3c).unwrap();
    });
    let out = shared(&device, length, |_| 0);
    blit(&device, |encoder| {
        encoder.copy_from_host_ram(&ram, &runs, &out, 0).unwrap();
    });
    assert!(
        contents(&out).iter().all(|&byte| byte == 0x3c),
        "the merge back gathered"
    );
}

/// Memory of Metal's private option bit 24 over the RAM's pages, where the system maps no
/// read-only view of them: a copy whose GPU writes are dropped, never bound in the RAM's chunk,
/// whose runs give it the RAM's log, so that a commit after the CPU wrote one page compares that
/// page alone; a GPU write to it leaves the pages as they were, and a later command buffer reads
/// their bytes.
#[test]
fn a_private_bit_copy_over_the_pages_copies_in_only_the_pages_the_log_reports() {
    if crate::read_only_view::VIEWS_DROP_WRITES {
        eprintln!("HOST-RAM-COPY: the private bit maps a read-only view here; nothing to test");
        return;
    }
    let Some(device) = device() else {
        return;
    };
    let log = Arc::new(Log::default());
    let (ram, start) = new_ram(Some(Arc::clone(&log)));
    let _used = device.use_host_ram(&ram).unwrap();
    let page = host_page();
    let length = 8 * page;
    let at = (start + 300 * PAGE).next_multiple_of(page);
    write_pattern(at..at + length, 21);
    let options =
        ResourceOptions::from_raw(ResourceOptions::default().raw() | ResourceOptions::PRIVATE_BIT)
            .unwrap();
    let before = device.work_counts();
    // SAFETY: the RAM's bytes, mapped until its keep-alive goes, after every device let go.
    let memory =
        unsafe { HostMemory::new(NonNull::new(at as *mut u8).unwrap()) }.with_runs(itself());
    // SAFETY: as above.
    let copied =
        unsafe { device.new_buffer_with_bytes_no_copy(memory, length, options, None) }.unwrap();
    let after = device.work_counts();
    assert_eq!(copied.host_backing(), Some(HostBacking::Copied));
    assert_eq!(
        after.copied_read_only_memory - before.copied_read_only_memory,
        1,
        "a copy for the private bit"
    );
    assert_eq!(
        after.chunk_bound_buffers_made + after.run_bound_buffers_made,
        before.chunk_bound_buffers_made + before.run_bound_buffers_made,
        "never bound in the RAM's chunk"
    );
    let read = || {
        let out = shared(&device, length, |_| 0);
        blit(&device, |encoder| {
            encoder
                .copy_from_buffer_to_buffer(&copied, 0, &out, 0, length)
                .unwrap();
        });
        contents(&out)
    };
    assert!(
        read() == bytes(at..at + length),
        "the copy holds the pages' bytes"
    );
    // The CPU writes one page, which the RAM's log reports.
    let written = at + 3 * page..at + 4 * page;
    write_pattern(written.clone(), 22);
    log.told(written, 0);
    let before = device.work_counts();
    let out = read();
    let after = device.work_counts();
    let copied_in = after.copied_copy_in_bytes - before.copied_copy_in_bytes;
    let skipped = after.copied_copy_in_bytes_skipped - before.copied_copy_in_bytes_skipped;
    eprintln!(
        "HOST-RAM-COPY: a private-bit copy's commit after one page written: {copied_in} bytes \
         copied in, {skipped} of {length} not compared"
    );
    assert!(
        out == bytes(at..at + length),
        "the copy took the CPU's write"
    );
    assert_eq!(
        skipped,
        length - page,
        "only the page the log reports compared"
    );
    assert!(0 < copied_in && copied_in <= page);
    // The GPU's write to the copy is dropped.
    let kept = bytes(at..at + length);
    let before = device.work_counts();
    blit(&device, |encoder| {
        encoder.fill_buffer(&copied, 0..length, 0x5a).unwrap();
    });
    let after = device.work_counts();
    assert!(bytes(at..at + length) == kept, "the GPU's write dropped");
    assert!(after.copied_dropped_bytes > before.copied_dropped_bytes);
    assert!(
        read() == kept,
        "a later command buffer reads the pages' bytes"
    );
}

#[test]
fn copies_off_a_word_outside_the_ram_or_past_the_buffer_are_refused() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let (other, other_start) = new_ram(None);
    let page = |n: u64| start + n * PAGE;
    let out = shared(&device, 8 * PAGE, |_| 0);
    let before = device.work_counts();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    // A RAM the device does not use yet.
    let first = one(page(1)..page(2));
    assert!(refused(
        encoder.copy_from_host_ram(&ram, &first, &out, 0),
        true
    ));
    let used = device.use_host_ram(&ram).unwrap();
    for runs in [
        one(page(1) + 2..page(2)),
        one(page(1) + 1..page(1) + 100),
        one(page(1)..page(1)),
        one(page(2)..page(1)),
        one(start - PAGE..start),
        one(start + RAM_BYTES..start + RAM_BYTES + PAGE),
        one(start + RAM_BYTES - PAGE..start + RAM_BYTES + PAGE),
        vec![page(1)..page(2), page(3) + 1..page(4)],
    ] {
        assert!(
            refused(encoder.copy_from_host_ram(&ram, &runs, &out, 0), true),
            "{runs:x?}"
        );
        assert!(
            refused(encoder.copy_to_host_ram(&out, 0, &ram, &runs), true),
            "{runs:x?}"
        );
    }
    // Another RAM, which the device does not use.
    let theirs = one(other_start..other_start + PAGE);
    assert!(refused(
        encoder.copy_from_host_ram(&other, &theirs, &out, 0),
        true
    ));
    // Past the buffer, and off a word.
    let nine: Vec<Range<u64>> = (1..10).map(|n| page(n)..page(n + 1)).collect();
    assert!(refused(
        encoder.copy_from_host_ram(&ram, &nine, &out, 0),
        false
    ));
    assert!(refused(
        encoder.copy_to_host_ram(&out, PAGE, &ram, &nine[..8]),
        false
    ));
    assert!(refused(
        encoder.copy_from_host_ram(&ram, &first, &out, 2),
        true
    ));
    // Runs each at a byte of their own: one past the buffer, or at a byte that passes `u64`;
    // a host run off a word.
    for (runs, host_ram) in [
        (
            vec![(0, page(1)..page(2)), (8 * PAGE - 8, page(3)..page(3) + 16)],
            false,
        ),
        (vec![(u64::MAX - 4, page(1)..page(1) + 16)], false),
        (vec![(0, page(1) + 2..page(2))], true),
    ] {
        assert!(
            refused(encoder.copy_runs_from_host_ram(&ram, &runs, &out), host_ram),
            "{runs:x?}"
        );
        assert!(
            refused(encoder.copy_runs_to_host_ram(&out, &ram, &runs), host_ram),
            "{runs:x?}"
        );
    }
    // No runs copy nothing; the encoder goes on.
    encoder.copy_from_host_ram(&ram, &[], &out, 0).unwrap();
    encoder.copy_runs_from_host_ram(&ram, &[], &out).unwrap();
    write_pattern(page(1)..page(2), 10);
    encoder.copy_from_host_ram(&ram, &first, &out, 0).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert!(contents(&out)[..PAGE as usize] == bytes(page(1)..page(2))[..]);
    let after = device.work_counts();
    assert_eq!(
        after.host_ram_copies - before.host_ram_copies,
        1,
        "one copy recorded"
    );
    // The use ended: refused again.
    drop(used);
    blit(&device, |encoder| {
        assert!(refused(
            encoder.copy_from_host_ram(&ram, &first, &out, 0),
            true
        ));
    });
}

/// The counts once `done` holds of them (at most [`WAIT`]) and a while has passed with the
/// device's thread free to go on, so that a thread importing past where it should shows.
fn settled(device: &Device, done: impl Fn(&crate::WorkCounts) -> bool) -> crate::WorkCounts {
    let started = Instant::now();
    let reached = loop {
        let counts = device.work_counts();
        if done(&counts) || started.elapsed() > WAIT {
            break counts;
        }
        std::thread::sleep(Duration::from_millis(2));
    };
    std::thread::sleep(Duration::from_millis(300));
    let later = device.work_counts();
    assert_eq!(
        later.host_ram_chunks_imported_ahead, reached.host_ram_chunks_imported_ahead,
        "the device's thread went on past the window"
    );
    later
}

#[test]
fn a_copy_takes_its_chunks_as_an_object_does_and_the_window_ahead_moves_past_them() {
    let Some(device) = ahead_device() else {
        return;
    };
    const WINDOW: u64 = AHEAD_CHUNKS as u64;
    // The start's chunk, a full window past it, and one chunk past that.
    const CHUNKS: u64 = 2 + WINDOW;
    let (ram, start) = ram_of(CHUNKS * 64 * MIB, None);
    let before = device.work_counts();
    let used = device.use_host_ram(&ram).unwrap();
    let ahead = |counts: &crate::WorkCounts| {
        counts.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead
    };
    let by_copies = |counts: &crate::WorkCounts| {
        counts.host_ram_chunks_imported - before.host_ram_chunks_imported - ahead(counts)
    };
    let page_of = |chunk: u64| {
        let at = start + chunk * 64 * MIB;
        one(at..at + PAGE)
    };
    // Before any take: the RAM's first chunk, and none past it.
    let first = settled(&device, |counts| ahead(counts) == 1);
    assert_eq!(ahead(&first), 1, "one chunk from the RAM's start");
    // A gather from it takes it inside the window: it finds the chunk imported, and the full
    // window past it is imported ahead.
    let first_page = page_of(0);
    write_pattern(first_page[0].clone(), 7);
    let out = shared(&device, PAGE, |_| 0);
    blit(&device, |encoder| {
        encoder
            .copy_from_host_ram(&ram, &first_page, &out, 0)
            .unwrap();
    });
    assert!(contents(&out) == gathered(&first_page), "the gathered page");
    let opened = settled(&device, |counts| ahead(counts) == 1 + WINDOW);
    assert_eq!(
        ahead(&opened),
        1 + WINDOW,
        "a full window past the copy's take"
    );
    assert_eq!(
        by_copies(&opened),
        0,
        "the copy found its chunk imported ahead"
    );
    // A scatter into the chunk past the window: its page imported alone for the copy while the
    // thread imports the chunk, and the window past it is the RAM's end. On a device whose
    // barriers do not order blits through two imports of the same pages
    // (`AliasedImportBlitsUnordered`) the copy imports the chunk itself.
    let defect = metal2vulkan::error::RuntimeDriverDefect::AliasedImportBlitsUnordered;
    let a = u64::from(!device.properties().driver_defects.contains(&defect));
    let last_page = page_of(CHUNKS - 1);
    let scattering = device.work_counts();
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&out, 0, &ram, &last_page).unwrap();
    });
    assert!(
        bytes(last_page[0].clone()) == gathered(&first_page),
        "the scattered page"
    );
    let jumped = settled(&device, |counts| {
        by_copies(counts) + ahead(counts) == 2 + WINDOW
            && counts.run_imports_live == before.run_imports_live
    });
    assert_eq!(
        (by_copies(&jumped), ahead(&jumped)),
        (1 - a, 1 + WINDOW + a),
        "the copy outside the window imported its page alone, or its chunk; nothing past the \
         RAM's end"
    );
    assert_eq!(
        jumped.host_ram_chunks_live - before.host_ram_chunks_live,
        CHUNKS
    );
    assert_eq!(
        (
            jumped.run_imports_made - scattering.run_imports_made,
            jumped.run_imports_live,
        ),
        (a, scattering.run_imports_live),
        "the page's import alone counted made, and gone with the copy"
    );
    drop((used, ram));
}

/// Single pages 3 MiB apart in chunks 2 and 3 of a RAM from `start`: 20 ranges in each, more than
/// a copy imports alone (`copy::MOST_ALONE`), none joined (`copy::JOIN_GAP`).
fn scattered_pages(start: u64) -> Vec<Range<u64>> {
    let mut runs = Vec::new();
    for chunk in [2, 3] {
        for at in 0..20 {
            let page = start + chunk * 64 * MIB + at * 3 * MIB + (at % 3) * PAGE;
            runs.push(page..page + PAGE);
        }
    }
    runs
}

/// A device whose thread imports ahead, using a RAM of four chunks whose first the thread
/// imported, and settled: the device, the RAM's use, the RAM and its first byte, and the counts
/// before the use. `None` where it imports no host memory.
fn ahead_ram() -> Option<(Device, crate::HostRamUse, Arc<HostRam>, u64, WorkCounts)> {
    let device = ahead_device()?;
    let (ram, start) = ram_of(4 * 64 * MIB, None);
    let before = device.work_counts();
    let used = device.use_host_ram(&ram).unwrap();
    let first = settled(&device, |counts| {
        counts.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead == 1
    });
    assert_eq!(
        first.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead,
        1,
        "the RAM's first chunk"
    );
    Some((device, used, ram, start, before))
}

#[test]
fn a_first_gather_over_chunks_the_thread_imports_reads_its_pages_by_the_cpu_at_its_submission() {
    let Some((device, used, ram, start, before)) = ahead_ram() else {
        return;
    };
    let ahead = |counts: &WorkCounts| {
        counts.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead
    };
    let runs = scattered_pages(start);
    for run in &runs {
        write_pattern(run.clone(), 3);
    }
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let cache = device
        .new_buffer(total, ResourceOptions::private())
        .unwrap();
    let back = shared(&device, total, |_| 0);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encoder.copy_from_host_ram(&ram, &runs, &cache, 0).unwrap();
    encoder
        .copy_from_buffer_to_buffer(&cache, 0, &back, 0, total)
        .unwrap();
    encoder.end_encoding().unwrap();
    // The CPU's writes after the encoding, before the commit: in the pages the copy reads.
    write_pattern(runs[5].clone(), 4);
    write_pattern(runs[30].clone(), 4);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert!(
        contents(&back) == gathered(&runs),
        "the pages as they stood at the commit"
    );
    let after = device.work_counts();
    assert_eq!(
        after.host_ram_copies_staged - before.host_ram_copies_staged,
        1
    );
    assert_eq!(
        after.host_ram_copy_import_waits - before.host_ram_copy_import_waits,
        0,
        "no wait for a chunk's import"
    );
    assert_eq!(
        after.host_ram_staged_fills_gated - before.host_ram_staged_fills_gated,
        0
    );
    assert!(after.host_ram_staged_fill_nanos > before.host_ram_staged_fill_nanos);
    // The thread imports the two chunks the copy asked for, and not the one between.
    let imported = settled(&device, |counts| ahead(counts) == 3);
    assert_eq!(
        ahead(&imported),
        3,
        "the start's chunk and the two asked for"
    );
    assert_eq!(
        imported.host_ram_chunks_imported - before.host_ram_chunks_imported,
        3,
        "no other chunk"
    );
    // The gather after it reads their imports: neither staged nor waiting.
    let again = shared(&device, total, |_| 0);
    blit(&device, |encoder| {
        encoder.copy_from_host_ram(&ram, &runs, &again, 0).unwrap();
    });
    assert!(contents(&again) == gathered(&runs), "through the imports");
    let last = device.work_counts();
    assert_eq!(last.host_ram_copies_staged, after.host_ram_copies_staged);
    assert_eq!(
        last.host_ram_copy_import_waits,
        after.host_ram_copy_import_waits
    );
    drop((used, ram));
}

/// A gather of few ranges over chunks not imported yet, which an import of its ranges alone could
/// take (fewer than `copy::MOST_ALONE`, none the whole chunk), is staged as one of many is: its
/// encoding imports nothing (no run import is live once it is encoded), it waits for no import,
/// and it reads the pages.
#[test]
fn a_first_gather_of_few_ranges_imports_nothing_on_the_encoding_thread() {
    let Some((device, used, ram, start, before)) = ahead_ram() else {
        return;
    };
    // Three pages 1 MiB apart in chunk 2, one range alone, and one page in chunk 3.
    let runs: Vec<Range<u64>> = [0, MIB, 2 * MIB, 64 * MIB + 5 * MIB]
        .iter()
        .map(|&at| {
            let page = start + 2 * 64 * MIB + at;
            page..page + PAGE
        })
        .collect();
    for run in &runs {
        write_pattern(run.clone(), 9);
    }
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let cache = device
        .new_buffer(total, ResourceOptions::private())
        .unwrap();
    let back = shared(&device, total, |_| 0);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encoder.copy_from_host_ram(&ram, &runs, &cache, 0).unwrap();
    encoder
        .copy_from_buffer_to_buffer(&cache, 0, &back, 0, total)
        .unwrap();
    encoder.end_encoding().unwrap();
    let encoded = device.work_counts();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert!(contents(&back) == gathered(&runs), "the gathered pages");
    let after = device.work_counts();
    eprintln!(
        "HOST-RAM-FEW-RANGES {}: staged {}, run imports live once encoded {}, import waits {}",
        device.name(),
        after.host_ram_copies_staged - before.host_ram_copies_staged,
        encoded.run_imports_live - before.run_imports_live,
        after.host_ram_copy_import_waits - before.host_ram_copy_import_waits,
    );
    assert_eq!(
        encoded.run_imports_live, before.run_imports_live,
        "the encoding imported the copy's ranges alone"
    );
    assert_eq!(
        after.host_ram_copies_staged - before.host_ram_copies_staged,
        1,
        "the gather was not staged"
    );
    assert_eq!(
        after.host_ram_copy_import_waits - before.host_ram_copy_import_waits,
        0,
        "no wait for a chunk's import"
    );
    drop((used, ram));
}

#[test]
fn a_staged_gather_after_a_gpu_write_over_its_pages_waits_for_that_write() {
    let Some((device, used, ram, start, before)) = ahead_ram() else {
        return;
    };
    let runs = scattered_pages(start);
    for run in &runs {
        write_pattern(run.clone(), 5);
    }
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let out = shared(&device, total, |_| 0);
    let queue = device.new_command_queue().unwrap();
    // Encoded while only the copies' keys lie over its pages: staged.
    let gather = queue.command_buffer().unwrap();
    let mut encoder = gather.blit_command_encoder().unwrap();
    encoder.copy_from_host_ram(&ram, &runs, &out, 0).unwrap();
    encoder.end_encoding().unwrap();
    // Then a buffer over one of its pages, which a command buffer committed before it fills.
    let target = runs[7].clone();
    let over = over_ram(&device, target.start, PAGE, itself(), true);
    let write = queue.command_buffer().unwrap();
    let mut fill = write.blit_command_encoder().unwrap();
    fill.fill_buffer(&over, 0..PAGE, 0xa5).unwrap();
    fill.end_encoding().unwrap();
    write.commit().unwrap();
    gather.commit().unwrap();
    gather.wait_until_completed_for(WAIT).unwrap();
    write.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(gather.status(), CommandBufferStatus::Completed);
    assert!(bytes(target).iter().all(|&byte| byte == 0xa5));
    assert!(
        contents(&out) == gathered(&runs),
        "the GPU's write gathered"
    );
    let after = device.work_counts();
    assert_eq!(
        after.host_ram_copies_staged - before.host_ram_copies_staged,
        1
    );
    assert_eq!(
        after.host_ram_staged_fills_gated - before.host_ram_staged_fills_gated,
        1,
        "its fill waited for the work before it"
    );
    drop((over, used, ram));
}

#[test]
fn a_staged_gather_behind_a_shared_event_reads_the_cpu_s_writes_before_the_signal() {
    let Some((device, used, ram, start, before)) = ahead_ram() else {
        return;
    };
    let runs = scattered_pages(start);
    for run in &runs {
        write_pattern(run.clone(), 6);
    }
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let out = shared(&device, total, |_| 0);
    let event = device.new_shared_event().unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::Shared(&event), 1)
        .unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encoder.copy_from_host_ram(&ram, &runs, &out, 0).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    // Committed, held for the signal: the CPU's writes before it are what the copy reads.
    std::thread::sleep(Duration::from_millis(50));
    write_pattern(runs[3].clone(), 8);
    write_pattern(runs[33].clone(), 8);
    event.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert!(
        contents(&out) == gathered(&runs),
        "the pages as they stood at the signal"
    );
    let after = device.work_counts();
    assert_eq!(
        after.host_ram_copies_staged - before.host_ram_copies_staged,
        1
    );
    drop((used, ram));
}
