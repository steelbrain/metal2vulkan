//! Copies between a guest RAM and a buffer on a device (`host_ram/copy.rs`):
//! `BlitCommandEncoder::copy_from_host_ram` and `copy_to_host_ram`.
//!
//! - **Both ways, byte for byte.** A gather of ascending runs (adjacent ones merged), descending
//!   neighbours, a run across a chunk's end, a run in the second chunk and a page named twice
//!   lands in a `Shared` and a `Private` buffer as the CPU's view of the pages reads; a scatter
//!   back writes each run's bytes and no other byte of the RAM. One dispatch for the two chunks
//!   the runs touch where the device's pages copy binds an array of imports
//!   (`PagesCopyModule::imports`, more than one where it indexes arrays of storage buffers; one
//!   dispatch per chunk where it holds one), and each chunk imported by the copy where no object
//!   imported it (the device's thread held back), the copy's encoding counted as waiting for those
//!   imports once, and a copy over the imported chunks as waiting for none.
//! - **The CPU's writes before the commit** are in a gather encoded before them, and a scatter's
//!   bytes are in the pages once the command buffer completed.
//! - **Ordered against a no-copy buffer over the same pages**, in one command buffer: a buffer
//!   over the RAM's own addresses (bound in its chunk) read after a scatter, in the same encoder
//!   and in a later one, reads the scatter's bytes, and a gather after a fill through it reads
//!   the fill's; in one encoder a barrier is recorded between them.
//! - **A copied backing over the pages** (the import gate refused, the RAM's log given) takes a
//!   scatter's bytes at the next commit, the log told of them; a gather after its merge back
//!   reads what the GPU wrote through it. A `Managed` no-copy buffer over exactly such a copy's
//!   bytes, or a run bound in its chunk, on a device without unified memory, is two copies, its
//!   device copy in device-local memory, never the copy or the chunk.
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
//!   into a `Shared` and a `Private` buffer and scattered back by one dispatch over both chunks
//!   (a dispatch per array of imports) and no region: every other byte of their pages (each a
//!   pattern) stays. A scatter of such runs that
//!   name bytes twice takes regions, and leaves the later's.
//! - **A laid-out surface in one call**: rows at a pitch over scattered pages (two side by side,
//!   one before the first chunk's end, one in the second), each run at its byte of a buffer that
//!   keeps the layout (`copy_runs_from_host_ram`, `copy_runs_to_host_ram`): rows of 4-byte texels
//!   by one dispatch over both chunks both ways, rows of 2-byte texels and rows from a byte of the buffer
//!   off a word by regions; the buffer's padding (a sentinel) and every byte of the RAM the runs
//!   do not name stay. A gather of two runs at one byte of the buffer keeps the later's bytes.
//! - **Refused**: runs off a word, empty or outside the RAM, a RAM the device does not use (or no
//!   longer does), bytes past the buffer and an offset off a word; of runs each at a byte of
//!   their own, one past the buffer or at a byte past `u64`; no runs copy nothing.
//! - **A chunk taken as an object takes it**, on a device whose thread imports ahead: a gather
//!   from the RAM's first chunk finds it imported ahead and opens the full window past it, and a
//!   scatter into a chunk past the window is staged, importing nothing of its own, while the
//!   thread imports the chunk; the thread imports nothing past the window.
//! - **A first touch through the CPU**, on a device whose thread imports ahead: a gather of
//!   pages scattered over two chunks not imported yet, more ranges in each than are imported
//!   alone, is staged: no wait for an import, the CPU's writes after its encoding and before the
//!   commit gathered (its pages read at its batch's submission), and its chunks imported by the
//!   thread next, and nothing else, so the gather after it reads their imports. So is one of few
//!   ranges, which an import of its ranges alone could take: its encoding imports nothing. One
//!   encoded before a buffer over one of its pages was made and written by the GPU in a command
//!   buffer committed before it waits for that write (its fill gated), and reads it; one behind a
//!   shared event reads the CPU's writes made before the signal. A scatter over two such chunks,
//!   from a `Private` buffer the GPU wrote just before it in its encoder, a page named twice among
//!   its runs, is staged: its encoding imports nothing and waits for none, its pages hold its
//!   bytes once it completed (the later's where a page is named twice) and no other page moved,
//!   and a buffer over one of them reads them in a later encoder and in a command buffer of
//!   another queue that waits for an event signalled after it. So is one of few ranges, whose
//!   encoding imports nothing, and the scatter after it, over the chunks the thread imported,
//!   goes through their imports. No staged copy's encoding takes its staging of the command
//!   buffer's arena. A rebind before the first submission of a no-copy buffer that a staged
//!   gather's encoder fills encodes that encoder again, and the command buffer completes with
//!   both copies' bytes in place.
//!
//! - **Shared RAM, whenever the copy comes**: on a device whose thread imports ahead, a gather
//!   over a RAM of shared memory with no dma-buf source once the thread reached its chunk, a
//!   scatter into it and a gather of the scatter's bytes each move the CPU's view of the pages; on
//!   a device that refuses pointers to shared memory (RADV) each is staged through the CPU, the
//!   chunk's import refused, so none is refused for coming after the thread tried the chunk.
//!
//! Every other test holds the device's thread back (`Device::hold_host_ram_imports`).
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device that imports no host memory says so and passes.

use super::rebind_behind_a_wait as rebind;
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
        // The pages this test made, which no device imports any more: its keep-alive is dropped
        // after the last chunk over it is freed.
        let start = NonNull::new(self.start as *mut u8).unwrap();
        super::host_os::release(start, self.length as usize);
    }
}

/// A RAM of `RAM_BYTES`, its writes logged to `log` where given, and its first byte.
fn new_ram(log: Option<Arc<Log>>) -> (Arc<HostRam>, u64) {
    ram_of(RAM_BYTES, log)
}

/// A RAM of `length` bytes, its writes logged to `log` where given, and its first byte.
fn ram_of(length: u64, log: Option<Arc<Log>>) -> (Arc<HostRam>, u64) {
    let start = super::host_os::private(length as usize).as_ptr() as u64;
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
    super::host_os::page()
}

/// The pages as two runs of the RAM: their first host page, then the rest.
fn two_runs() -> RunResolver {
    let page = host_page();
    Arc::new(move |start, length| Some(vec![start..start + page, start + page..start + length]))
}

/// A no-copy buffer of `length` bytes at host address `at`, resolved by `runs`, its import
/// admitted or refused by its gate as `admit` says.
fn over_ram(device: &Device, at: u64, length: u64, runs: RunResolver, admit: bool) -> Buffer {
    over_ram_as(
        device,
        (at, length),
        runs,
        admit,
        ResourceOptions::default(),
    )
}

/// [`over_ram`] of `options`.
fn over_ram_as(
    device: &Device,
    (at, length): (u64, u64),
    runs: RunResolver,
    admit: bool,
    options: ResourceOptions,
) -> Buffer {
    let gate: ImportGate = Arc::new(move |_| admit);
    // SAFETY: the RAM's bytes, mapped until its keep-alive goes, after every device let go.
    let memory = unsafe { HostMemory::new(NonNull::new(at as *mut u8).unwrap()) }
        .with_runs(runs)
        .with_import_gate(gate);
    // SAFETY: as above.
    unsafe { device.new_buffer_with_bytes_no_copy(memory, length, options, None) }.unwrap()
}

/// One run, as a list.
fn one(range: Range<u64>) -> Vec<Range<u64>> {
    std::iter::once(range).collect()
}

/// The dispatches of a copy by words whose runs lie in `imports` imports: one for each array of
/// them the device's pages copy binds (`PagesCopyModule::imports`), so one for up to 30 where it
/// indexes arrays of storage buffers.
fn dispatches_over(device: &Device, imports: u64) -> u64 {
    let array = metal2vulkan::bind::PagesCopyModule::imports(device.compile_device());
    eprintln!("HOST-RAM-COPY: the pages copy binds {array} imports a dispatch");
    imports.div_ceil(u64::from(array))
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
        dispatches_over(&device, 2),
        "one dispatch for each array of the imports the runs touch"
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
fn runs_of_words_inside_pages_take_dispatches_alone_and_leave_the_rest_of_their_pages() {
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
        dispatches_over(&device, 2),
        "one dispatch for each array of the imports the runs touch"
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
        (dispatches_over(&device, 2), 0),
        "the scatter: a dispatch for each array of imports, no region"
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
    // dispatch for each array of the imports the rows touch (two chunks), no region, both ways.
    let rows = laid_out(start, (1536, 1500), 64);
    assert!(rows.len() > 21, "rows cut where a page of the surface ends");
    let counts = laid_out_both_ways(&device, &ram, start, &rows, length);
    assert_eq!(
        counts,
        [(dispatches_over(&device, 2), 0); 2],
        "a dispatch for each array of imports, no region, both ways"
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

/// A `Managed` no-copy buffer over the RAM's pages, made over exactly the bytes of a live `Shared`
/// one, is two copies on a device without unified memory, as over any pages (`managed_copy.rs`):
/// its device copy in device-local memory, and it reads the pages' bytes. The `Shared` one lies in
/// a copy of them where they are two runs whose import the gate refused, and in the chunk's import
/// where they are one run inside it (bound there whatever the gate answers); the `Managed` one lies
/// in neither.
#[test]
fn a_managed_no_copy_buffer_over_a_shared_one_s_run_is_two_copies() {
    let Some(device) = device() else {
        return;
    };
    if device.properties().has_unified_memory {
        eprintln!("HOST-RAM-COPY {}: unified memory, one copy", device.name());
        return;
    }
    let (ram, start) = new_ram(None);
    let _used = device.use_host_ram(&ram).unwrap();
    let managed = ResourceOptions::new(
        crate::StorageMode::Managed,
        crate::CpuCacheMode::DefaultCache,
        crate::HazardTrackingMode::Default,
    );
    let length = 4 * host_page();
    let cases = [
        (two_runs(), 16 * MIB, HostBacking::Copied),
        (itself(), 8 * MIB, HostBacking::InHostRam),
    ];
    for (seed, (runs, at, shared_backing)) in cases.into_iter().enumerate() {
        let at = start + at;
        write_pattern(at..at + length, seed as u64);
        let beside = over_ram(&device, at, length, Arc::clone(&runs), false);
        assert_eq!(beside.host_backing(), Some(shared_backing));
        let before = device.work_counts();
        let buffer = over_ram_as(&device, (at, length), runs, false, managed);
        let made = device.work_counts();
        eprintln!(
            "HOST-RAM-COPY over a {shared_backing:?} run {}: {:?}, {} device copies, {} in a copy",
            device.name(),
            buffer.host_backing(),
            made.managed_device_copies - before.managed_device_copies,
            made.copied_host_buffers_shared - before.copied_host_buffers_shared,
        );
        assert_eq!(buffer.host_backing(), Some(HostBacking::DeviceCopy));
        assert_eq!(made.managed_device_copies - before.managed_device_copies, 1);
        assert_eq!(
            made.copied_host_buffers_shared,
            before.copied_host_buffers_shared
        );
        assert_eq!(made.managed_device_copies_in_place, 0);
        let out = shared(&device, length, |_| 0);
        blit(&device, |encoder| {
            encoder
                .copy_from_buffer_to_buffer(&buffer, 0, &out, 0, length)
                .unwrap();
        });
        assert!(contents(&out) == bytes(at..at + length), "the pages' bytes");
    }
}

/// Memory of Metal's private option bit 24 over the RAM's pages, where the system maps no
/// read-only view of them: a copy whose GPU writes are dropped, never bound in the RAM's chunk
/// (its read twin is, the runtime's buffer over the pages in place, `read_twin.rs`), whose runs
/// give it the RAM's log, so that a commit that uses the copy after the CPU wrote one page
/// compares that page alone (a blit reads the copy after a write of the buffer in its command
/// buffer, and the pages through the twin else, comparing nothing); a GPU write to it leaves the
/// pages as they were, and a later command buffer reads their bytes.
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
    assert!(copied.read_twin().is_some(), "its read twin");
    assert_eq!(
        [
            after.read_twins_made - before.read_twins_made,
            after.chunk_bound_buffers_made - before.chunk_bound_buffers_made,
        ],
        [1, 1],
        "the copy never bound in the RAM's chunk, its read twin bound there"
    );
    // A blit reads the buffer through its twin, the pages in place; after a fill of a word of
    // it in the same command buffer (`through_copy`), through its copy, the fill read back. The
    // word lies in the page the CPU writes below, so the page its dropped write leaves for the
    // copy's next commit to compare is that one.
    let word = 3 * page..3 * page + 4;
    let read = |through_copy: bool| {
        let out = shared(&device, length, |_| 0);
        blit(&device, |encoder| {
            if through_copy {
                encoder.fill_buffer(&copied, word.clone(), 0x5a).unwrap();
            }
            encoder
                .copy_from_buffer_to_buffer(&copied, 0, &out, 0, length)
                .unwrap();
        });
        contents(&out)
    };
    let filled = |mut bytes: Vec<u8>| {
        bytes[word.start as usize..word.end as usize].fill(0x5a);
        bytes
    };
    assert!(
        read(true) == filled(bytes(at..at + length)),
        "the copy holds the pages' bytes"
    );
    // The CPU writes one page, which the RAM's log reports.
    let written = at + 3 * page..at + 4 * page;
    write_pattern(written.clone(), 22);
    log.told(written, 0);
    let before = device.work_counts();
    let out = read(true);
    let after = device.work_counts();
    let copied_in = after.copied_copy_in_bytes - before.copied_copy_in_bytes;
    let skipped = after.copied_copy_in_bytes_skipped - before.copied_copy_in_bytes_skipped;
    eprintln!(
        "HOST-RAM-COPY: a private-bit copy's commit after one page written: {copied_in} bytes \
         copied in, {skipped} of {length} not compared"
    );
    assert_eq!(
        after.read_twin_reads_after_writes - before.read_twin_reads_after_writes,
        1,
        "read through the copy"
    );
    assert!(
        out == filled(bytes(at..at + length)),
        "the copy took the CPU's write"
    );
    assert_eq!(
        skipped,
        length - page,
        "only the page the log reports compared"
    );
    assert!(0 < copied_in && copied_in <= page);
    // Through the twin: the pages' bytes, the fill dropped, the copy compared nowhere.
    let before = device.work_counts();
    let out = read(false);
    let after = device.work_counts();
    assert_eq!(after.read_twin_blit_reads - before.read_twin_blit_reads, 1);
    assert_eq!(
        after.copied_copy_ins_compared, before.copied_copy_ins_compared,
        "the copy compared nowhere"
    );
    assert!(out == bytes(at..at + length), "the twin reads the pages");
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
        read(false) == kept,
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
    // A scatter into the chunk past the window: staged through the CPU, no import of its own,
    // while the thread imports the chunk it asked for; the window past it is the RAM's end.
    let last_page = page_of(CHUNKS - 1);
    let scattering = device.work_counts();
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&out, 0, &ram, &last_page).unwrap();
    });
    assert!(
        bytes(last_page[0].clone()) == gathered(&first_page),
        "the scattered page"
    );
    // Waited for by the thread's count alone: its import of a chunk moves
    // `host_ram_chunks_imported` before `host_ram_chunks_imported_ahead`, so a sum of the two
    // reaches its mark in between, and the thread's import reads as a copy's there.
    let jumped = settled(&device, |counts| {
        ahead(counts) == 2 + WINDOW && counts.run_imports_live == before.run_imports_live
    });
    assert_eq!(
        (by_copies(&jumped), ahead(&jumped)),
        (0, 2 + WINDOW),
        "the copy outside the window imported nothing, the thread its chunk; nothing past the \
         RAM's end"
    );
    assert_eq!(
        jumped.host_ram_chunks_live - before.host_ram_chunks_live,
        CHUNKS
    );
    assert_eq!(
        (
            jumped.run_imports_made - scattering.run_imports_made,
            jumped.host_ram_scatters_staged - scattering.host_ram_scatters_staged,
        ),
        (0, 1),
        "the scatter staged, with no import of its page alone"
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
    assert!(
        command_buffer.state().uploads.bytes() < total,
        "the encoding took the staging"
    );
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
    assert!(
        command_buffer.state().uploads.bytes() < total,
        "the encoding took the staging"
    );
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

/// A staged gather leaves its encoder's log in place (`Deferred::recorded_by_an_encoding_again`):
/// a rebind, before the command buffer's first submission, of a no-copy buffer the same encoder
/// fills encodes the encoder again, which copies the gather through the imports; the command
/// buffer completes with the fill in the pages the buffer lies in now and the gather's bytes in
/// its buffer.
#[test]
fn a_rebind_before_the_first_submission_encodes_a_staged_gather_s_encoder_again() {
    // Declared before the device, so dropped after it.
    let pages = [rebind::Page::new(), rebind::Page::new()];
    let Some((device, used, ram, start, _)) = ahead_ram() else {
        return;
    };
    let runs = scattered_pages(start);
    for run in &runs {
        write_pattern(run.clone(), 14);
    }
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let out = shared(&device, total, |_| 0);
    let length = rebind::BYTES as u64;
    // SAFETY: each page outlives every buffer over it (dropped after the device), and only the
    // buffer's commands use it while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            pages[0].memory(),
            length,
            ResourceOptions::shared(),
            None,
        )
    };
    let over = match made {
        Ok(buffer) => buffer,
        Err(Error::Unsupported(gap)) => {
            eprintln!("HOST-RAM-COPY: the device imports no such pages: {gap:?}");
            return;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    // Its command buffer waits for an event only the host signals, below: the rebind comes
    // before its first submission.
    let gate = device.new_shared_event().unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::Shared(&gate), 1)
        .unwrap();
    let encoding = device.work_counts();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encoder.copy_from_host_ram(&ram, &runs, &out, 0).unwrap();
    encoder.fill_buffer(&over, 0..length, 0x5a).unwrap();
    encoder.end_encoding().unwrap();
    let held = command_buffer.state().uploads.bytes();
    command_buffer.commit().unwrap();
    assert_eq!(over.detach_backing(), Ok(true));
    // SAFETY: as above.
    unsafe { over.rebind_bytes_no_copy(pages[1].memory(), None, None) }.unwrap();
    let rebound = device.work_counts();
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None)
    );
    assert!(held < total, "the first encoding took the staging");
    assert_eq!(
        rebound.host_ram_copies_staged - encoding.host_ram_copies_staged,
        1,
        "the first encoding staged the gather"
    );
    assert_eq!(
        rebound.encoders_reencoded - encoding.encoders_reencoded,
        1,
        "the rebind encoded the encoder again"
    );
    assert!(
        pages[1].written() && !pages[0].written(),
        "the fill in the pages the buffer lies in now"
    );
    assert!(contents(&out) == gathered(&runs), "the gathered pages");
    drop((over, used, ram));
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

/// The bytes `runs` hold once a scatter of `source`'s bytes, one run after another, completed:
/// each distinct run's, the later's where two runs name one.
fn scattered(runs: &[Range<u64>], source: &[u8]) -> Vec<(Range<u64>, Vec<u8>)> {
    let mut landed: Vec<(Range<u64>, Vec<u8>)> = Vec::new();
    let mut at = 0;
    for run in runs {
        let length = (run.end - run.start) as usize;
        let bytes = source[at..at + length].to_vec();
        at += length;
        match landed.iter_mut().find(|(was, _)| was == run) {
            Some(was) => was.1 = bytes,
            None => landed.push((run.clone(), bytes)),
        }
    }
    landed
}

#[test]
fn a_first_scatter_over_chunks_the_thread_imports_is_in_its_pages_once_it_completes() {
    let Some((device, used, ram, start, before)) = ahead_ram() else {
        return;
    };
    let ahead = |counts: &WorkCounts| {
        counts.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead
    };
    let by_copies = |counts: &WorkCounts| {
        counts.host_ram_chunks_imported - before.host_ram_chunks_imported - ahead(counts)
    };
    // The scattered pages, then the first named again: its later bytes stay.
    let mut runs = scattered_pages(start);
    runs.push(runs[0].clone());
    // A page beside one of them, which no run names.
    let beside = runs[1].end..runs[1].end + PAGE;
    for run in runs.iter().chain([&beside]) {
        write_pattern(run.clone(), 2);
    }
    let kept = bytes(beside.clone());
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let source = shared(&device, total, |at| pattern(at, 11));
    let cache = device
        .new_buffer(total, ResourceOptions::private())
        .unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    let encoding = device.work_counts();
    // The cache written by the GPU just before the scatter reads it.
    encoder
        .copy_from_buffer_to_buffer(&source, 0, &cache, 0, total)
        .unwrap();
    encoder.copy_to_host_ram(&cache, 0, &ram, &runs).unwrap();
    encoder.end_encoding().unwrap();
    let encoded = device.work_counts();
    assert!(
        command_buffer.state().uploads.bytes() < total,
        "the encoding took the staging"
    );
    eprintln!(
        "HOST-RAM-FIRST-SCATTER {}: staged {}, run imports made once encoded {}, chunks imported \
         by copies {}, import waits {}",
        device.name(),
        encoded.host_ram_scatters_staged - encoding.host_ram_scatters_staged,
        encoded.run_imports_made - encoding.run_imports_made,
        by_copies(&encoded),
        encoded.host_ram_copy_import_waits - encoding.host_ram_copy_import_waits,
    );
    assert_eq!(
        encoded.host_ram_scatters_staged - encoding.host_ram_scatters_staged,
        1,
        "the scatter was not staged"
    );
    assert_eq!(
        (
            encoded.run_imports_made - encoding.run_imports_made,
            by_copies(&encoded),
            encoded.host_ram_copy_import_waits - encoding.host_ram_copy_import_waits,
        ),
        (0, 0, 0),
        "the encoding imported the copy's ranges or a chunk, or waited for one"
    );
    // Later readers of one of its pages, through a buffer over it: a later encoder of the same
    // command buffer, and a command buffer of another queue committed after it that waits for an
    // event the first signals after the scatter. Commit order alone does not order the two: the
    // first's batches give way while the scatter's host steps have not taken their gate
    // (`scheduler/order.rs`, **Held batches give way**), as an x86 Mac's GPU may take a later
    // commit of another queue first.
    let target = runs[7].clone();
    let over = over_ram(&device, target.start, PAGE, itself(), true);
    let read = shared(&device, PAGE, |_| 0);
    let mut later = command_buffer.blit_command_encoder().unwrap();
    later
        .copy_from_buffer_to_buffer(&over, 0, &read, 0, PAGE)
        .unwrap();
    later.end_encoding().unwrap();
    let ordered = device.new_event().unwrap();
    command_buffer
        .encode_signal_event(EventRef::Event(&ordered), 1)
        .unwrap();
    let other = device.new_command_queue().unwrap();
    let after_it = other.command_buffer().unwrap();
    after_it
        .encode_wait_for_event(EventRef::Event(&ordered), 1)
        .unwrap();
    let read_apart = shared(&device, PAGE, |_| 0);
    let mut apart = after_it.blit_command_encoder().unwrap();
    apart
        .copy_from_buffer_to_buffer(&over, 0, &read_apart, 0, PAGE)
        .unwrap();
    apart.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    after_it.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    // Its completion: the pages hold its bytes.
    let landed = scattered(&runs, &contents(&source));
    for (run, expected) in &landed {
        assert!(
            bytes(run.clone()) == *expected,
            "the scatter's bytes at {run:x?}"
        );
    }
    assert!(bytes(beside) == kept, "a page no run names");
    after_it.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(after_it.status(), CommandBufferStatus::Completed);
    let wanted = &landed.iter().find(|(run, _)| *run == target).unwrap().1;
    assert!(
        contents(&read) == *wanted,
        "a later encoder reads the scatter"
    );
    assert!(
        contents(&read_apart) == *wanted,
        "another queue's command buffer that waits for the scatter reads it"
    );
    let after = device.work_counts();
    assert!(after.host_ram_staged_drain_nanos > encoding.host_ram_staged_drain_nanos);
    drop((over, used, ram));
}

/// A copy over chunks the device's thread imported, three of them, wakes the thread once, not
/// once a chunk (`RamOnDevice::take_for_copies`), and so does the staged copy before it, over
/// those chunks not imported yet, which asked for them.
#[test]
fn a_copy_over_several_chunks_wakes_the_device_s_thread_once() {
    let Some((device, used, ram, start, before)) = ahead_ram() else {
        return;
    };
    let ahead = |counts: &WorkCounts| {
        counts.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead
    };
    let wakes = || device.shared().host_ram.wakes();
    // A page in each of chunks 1, 2 and 3.
    let runs: Vec<Range<u64>> = (1..4)
        .map(|chunk| {
            let page = start + chunk * 64 * MIB + 5 * PAGE;
            page..page + PAGE
        })
        .collect();
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let source = shared(&device, total, |at| pattern(at, 15));
    let staging = wakes();
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&source, 0, &ram, &runs).unwrap();
    });
    let staged = wakes() - staging;
    let imported = settled(&device, |counts| ahead(counts) == 4);
    assert_eq!(
        ahead(&imported),
        4,
        "the start's chunk and the three asked for"
    );
    let copying = wakes();
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&source, 0, &ram, &runs).unwrap();
    });
    let copied = wakes() - copying;
    let last = device.work_counts();
    eprintln!(
        "HOST-RAM-COPY-WAKES {}: the staged copy woke the thread {staged} times, the copy through \
         the imports {copied}",
        device.name()
    );
    assert_eq!(
        last.host_ram_scatters_staged - imported.host_ram_scatters_staged,
        0,
        "the second copy went through the imports"
    );
    assert_eq!(staged, 1, "the staged copy's takes and asks");
    assert_eq!(copied, 1, "the copy's takes of its three chunks");
    assert!(gathered(&runs) == contents(&source), "the scattered pages");
    drop((used, ram));
}

/// A scatter of few ranges over chunks not imported yet, which an import of its ranges alone
/// could take (fewer than `copy::MOST_ALONE`, none the whole chunk), is staged as one of many
/// is: its encoding imports nothing, and its pages hold its bytes once it completed. The chunks
/// it asked for are imported by the device's thread, and the scatter after it goes through
/// their imports, neither staged nor waiting.
#[test]
fn a_first_scatter_of_few_ranges_imports_nothing_on_the_encoding_thread() {
    let Some((device, used, ram, start, before)) = ahead_ram() else {
        return;
    };
    let ahead = |counts: &WorkCounts| {
        counts.host_ram_chunks_imported_ahead - before.host_ram_chunks_imported_ahead
    };
    // Three pages 1 MiB apart in chunk 2, one range alone, and one page in chunk 3.
    let runs: Vec<Range<u64>> = [0, MIB, 2 * MIB, 64 * MIB + 5 * MIB]
        .iter()
        .map(|&at| {
            let page = start + 2 * 64 * MIB + at;
            page..page + PAGE
        })
        .collect();
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let source = shared(&device, total, |at| pattern(at, 12));
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    let encoding = device.work_counts();
    encoder.copy_to_host_ram(&source, 0, &ram, &runs).unwrap();
    encoder.end_encoding().unwrap();
    let encoded = device.work_counts();
    assert!(
        command_buffer.state().uploads.bytes() < total,
        "the encoding took the staging"
    );
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    eprintln!(
        "HOST-RAM-FEW-SCATTER {}: staged {}, run imports made once encoded {}, import waits {}",
        device.name(),
        encoded.host_ram_scatters_staged - encoding.host_ram_scatters_staged,
        encoded.run_imports_made - encoding.run_imports_made,
        encoded.host_ram_copy_import_waits - encoding.host_ram_copy_import_waits,
    );
    assert_eq!(
        encoded.run_imports_made, encoding.run_imports_made,
        "the encoding imported the copy's ranges alone"
    );
    assert_eq!(
        encoded.host_ram_copy_import_waits,
        encoding.host_ram_copy_import_waits
    );
    assert_eq!(
        encoded.host_ram_scatters_staged - encoding.host_ram_scatters_staged,
        1,
        "the scatter was not staged"
    );
    assert!(gathered(&runs) == contents(&source), "the scattered pages");
    // The thread imports the two chunks the copy asked for, and no other.
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
    // The scatter after it goes through their imports.
    let again = shared(&device, total, |at| pattern(at, 13));
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&again, 0, &ram, &runs).unwrap();
    });
    assert!(gathered(&runs) == contents(&again), "through the imports");
    let last = device.work_counts();
    assert_eq!(
        last.host_ram_scatters_staged,
        encoded.host_ram_scatters_staged
    );
    assert_eq!(last.run_imports_made, encoded.run_imports_made);
    assert_eq!(
        last.host_ram_copy_import_waits,
        encoded.host_ram_copy_import_waits
    );
    drop((used, ram));
}

/// A first gather over chunks a scatter before it wrote, in its own encoder, while the device's
/// thread imports the chunks the scatter asked for (each import held in flight a while by the
/// test, `RamSource::making_pause`), is staged as any first gather is: its encoding imports
/// nothing and waits for no chunk's import, though some use but its read wrote its bytes; it goes
/// in a part of its own after the scatter, and reads the scatter's bytes, not those of the GPU's
/// writes of two of its pages recorded after it (a later encoder of its command buffer, and a
/// command buffer of the same queue committed after it).
#[test]
fn a_first_gather_over_chunks_a_scatter_wrote_waits_for_no_import() {
    let Some((device, used, ram, start, _before)) = ahead_ram() else {
        return;
    };
    let runs = scattered_pages(start);
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let source = shared(&device, total, |at| pattern(at, 14));
    let back = shared(&device, total, |_| 0);
    *device.shared().host_ram.making_pause.lock().unwrap() = Some(Arc::new(|_span| {
        std::thread::sleep(Duration::from_millis(300))
    }));
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    let encoding = device.work_counts();
    encoder.copy_to_host_ram(&source, 0, &ram, &runs).unwrap();
    let started = Instant::now();
    encoder.copy_from_host_ram(&ram, &runs, &back, 0).unwrap();
    let took = started.elapsed();
    encoder.end_encoding().unwrap();
    let encoded = device.work_counts();
    *device.shared().host_ram.making_pause.lock().unwrap() = None;
    let delta = |count: fn(&WorkCounts) -> u64| count(&encoded) - count(&encoding);
    eprintln!(
        "HOST-RAM-GATHER-AFTER-SCATTER {}: the gather's call {:.1} us; scatters staged {}, \
         gathers staged {}, import waits {} ({:.1} us), run imports made {}",
        device.name(),
        took.as_secs_f64() * 1e6,
        delta(|c| c.host_ram_scatters_staged),
        delta(|c| c.host_ram_copies_staged),
        delta(|c| c.host_ram_copy_import_waits),
        delta(|c| c.host_ram_copy_import_wait_nanos) as f64 / 1e3,
        delta(|c| c.run_imports_made),
    );
    assert_eq!(
        delta(|c| c.host_ram_scatters_staged),
        1,
        "the scatter was not staged"
    );
    assert_eq!(
        (
            delta(|c| c.host_ram_copy_import_waits),
            delta(|c| c.run_imports_made)
        ),
        (0, 0),
        "the gather's encoding waited for a chunk's import, or imported its ranges"
    );
    assert_eq!(
        delta(|c| c.host_ram_copies_staged),
        1,
        "the gather was not staged"
    );
    // The GPU's writes of two of its pages after it, through buffers over them.
    let (first, second) = (runs[7].clone(), runs[30].clone());
    let over_first = over_ram(&device, first.start, PAGE, itself(), true);
    let over_second = over_ram(&device, second.start, PAGE, itself(), true);
    let mut later = command_buffer.blit_command_encoder().unwrap();
    later.fill_buffer(&over_first, 0..PAGE, 0xa5).unwrap();
    later.end_encoding().unwrap();
    let after_it = queue.command_buffer().unwrap();
    let mut fill = after_it.blit_command_encoder().unwrap();
    fill.fill_buffer(&over_second, 0..PAGE, 0x5a).unwrap();
    fill.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    after_it.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    after_it.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    assert_eq!(after_it.status(), CommandBufferStatus::Completed);
    assert!(
        contents(&back) == contents(&source),
        "the gather read the scatter's bytes, before the writes after it"
    );
    assert!(bytes(first).iter().all(|&byte| byte == 0xa5));
    assert!(bytes(second).iter().all(|&byte| byte == 0x5a));
    drop((over_first, over_second, used, ram));
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

/// A RAM of `length` bytes of shared memory, as a VMM's guest RAM is
/// ([`super::host_os::shared`]), unmapped by its keep-alive, with no dma-buf source, and its first
/// byte.
fn shared_ram_of(length: u64) -> (Arc<HostRam>, u64) {
    let start = super::host_os::shared(length as usize).as_ptr();
    let start = start as u64;
    let keep: KeepAlive = Box::new(Mapping { start, length });
    let at = NonNull::new(start as *mut u8).unwrap();
    // SAFETY: the mapping is readable and writable until the keep-alive unmaps it.
    let ram = unsafe { HostRam::new(&[(at, length)], Some(keep)) }.unwrap();
    (Arc::new(ram), start)
}

/// A RAM of shared memory with no dma-buf source, on a device whose thread imports ahead: once
/// the thread reached the RAM's first chunk (imported it, or met its refusal), a gather of runs of
/// it, a scatter of other bytes into the same runs and a gather of those each move the bytes the
/// CPU's view of the pages holds. On a device that refuses pointers to shared memory (RADV) the
/// chunk's import is refused whatever it holds, and each copy is staged through the CPU
/// (`host_ram_copies_staged`, `host_ram_scatters_staged`): none is refused for coming after the
/// thread tried the chunk, and the gather after the scatter reads the bytes the scatter's CPU
/// copy wrote. Elsewhere they go through the chunk's import.
#[test]
fn copies_over_shared_ram_move_their_bytes_after_the_thread_reached_its_chunk() {
    let Some(device) = ahead_device() else {
        return;
    };
    let refuses = !device.imports_shared_host_pointers().unwrap();
    let (ram, start) = shared_ram_of(2 * 64 * MIB);
    let before = device.work_counts();
    let used = device.use_host_ram(&ram).unwrap();
    let reached = |counts: &WorkCounts| {
        counts.host_ram_chunks_imported_ahead > before.host_ram_chunks_imported_ahead
            || counts.host_ram_pointer_imports_failed > before.host_ram_pointer_imports_failed
    };
    let at = settled(&device, reached);
    assert!(
        reached(&at),
        "the thread never reached the RAM's first chunk"
    );
    let runs = [
        start + PAGE..start + 3 * PAGE,
        start + 5 * MIB..start + 5 * MIB + PAGE,
        start + 9 * MIB + 2 * PAGE..start + 9 * MIB + 6 * PAGE,
    ];
    for run in &runs {
        write_pattern(run.clone(), 11);
    }
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let first = shared(&device, total, |_| 0);
    blit(&device, |encoder| {
        encoder.copy_from_host_ram(&ram, &runs, &first, 0).unwrap();
    });
    assert!(contents(&first) == gathered(&runs), "the first gather");
    let source = shared(&device, total, |at| (at * 7 + 3) as u8);
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&source, 0, &ram, &runs).unwrap();
    });
    assert!(
        gathered(&runs) == contents(&source),
        "the scatter's bytes in the pages"
    );
    let again = shared(&device, total, |_| 0);
    blit(&device, |encoder| {
        encoder.copy_from_host_ram(&ram, &runs, &again, 0).unwrap();
    });
    assert!(
        contents(&again) == contents(&source),
        "the gather after the scatter"
    );
    let after = device.work_counts();
    let staged = after.host_ram_copies_staged - before.host_ram_copies_staged;
    let scatters = after.host_ram_scatters_staged - before.host_ram_scatters_staged;
    eprintln!(
        "HOST-RAM-COPY {}: shared RAM, no source, refused {refuses}: gathers staged {staged}, \
         scatters staged {scatters}, chunks imported {}, pointer imports refused {}",
        device.name(),
        after.host_ram_chunks_imported - before.host_ram_chunks_imported,
        after.host_ram_pointer_imports_failed - before.host_ram_pointer_imports_failed,
    );
    if refuses {
        assert_eq!(staged, 2, "both gathers through the CPU");
        assert_eq!(scatters, 1, "the scatter through the CPU");
    }
    drop((used, ram));
}

/// Four pages of a RAM from `start`, as a copy's runs: three apart in the first chunk and one in
/// the second, a `rebind::Page`'s bytes together.
fn kept_plan_runs(start: u64) -> Vec<Range<u64>> {
    let page = |n: u64| start + n * PAGE;
    vec![
        page(10)..page(11),
        page(3)..page(4),
        page(SECOND + 5)..page(SECOND + 6),
        page(40)..page(41),
    ]
}

/// The copies a device recorded by plans kept of the same copies before them.
fn plans_reused(device: &Device) -> u64 {
    device.work_counts().host_ram_copy_plans_reused
}

/// A copy repeated takes the plan kept of it (`WorkCounts::host_ram_copy_plans_reused`) and moves
/// the pages' bytes as they are now; a copy of another key plans anew: one run changed, the other
/// direction, the same runs at another byte of the buffer, another buffer as long, and the same
/// no-copy buffer rebound to other pages (its `VkBuffer` another).
#[test]
fn a_repeated_copy_takes_its_kept_plan_and_a_copy_of_another_key_plans_anew() {
    // Declared before the device, so dropped after it.
    let pages = [rebind::Page::new(), rebind::Page::new()];
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let used = device.use_host_ram(&ram).unwrap();
    let runs = kept_plan_runs(start);
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let gather = |runs: &[Range<u64>], out: &Buffer, at: u64| {
        blit(&device, |encoder| {
            encoder.copy_from_host_ram(&ram, runs, out, at).unwrap();
        });
    };
    let out = shared(&device, total, |_| 0);
    for run in &runs {
        write_pattern(run.clone(), 31);
    }
    let first = plans_reused(&device);
    gather(&runs, &out, 0);
    assert_eq!(plans_reused(&device), first, "the first copy plans");
    // New bytes in the pages: the copy that takes the plan reads them.
    for run in &runs {
        write_pattern(run.clone(), 32);
    }
    gather(&runs, &out, 0);
    assert_eq!(plans_reused(&device) - first, 1, "the same copy takes it");
    assert!(contents(&out) == gathered(&runs), "the pages' bytes now");
    // Each copy of another key plans anew, and moves its own bytes.
    let anew = |what: &str, copy: &dyn Fn()| {
        let before = plans_reused(&device);
        copy();
        assert_eq!(plans_reused(&device), before, "{what}: planned anew");
    };
    let mut changed = runs.clone();
    changed[2] = start + (SECOND + 7) * PAGE..start + (SECOND + 8) * PAGE;
    write_pattern(changed[2].clone(), 33);
    anew("a run changed", &|| gather(&changed, &out, 0));
    assert!(
        contents(&out) == gathered(&changed),
        "the changed runs' bytes"
    );
    let source = shared(&device, total, |at| (at % 251) as u8);
    anew("the other direction", &|| {
        blit(&device, |encoder| {
            encoder.copy_to_host_ram(&source, 0, &ram, &runs).unwrap();
        });
    });
    assert!(gathered(&runs) == contents(&source), "the scattered bytes");
    let wider = shared(&device, total + 256, |_| 0);
    anew("another byte of the buffer", &|| gather(&runs, &wider, 256));
    assert!(
        contents(&wider)[256..] == gathered(&runs)[..],
        "at byte 256"
    );
    let other = shared(&device, total, |_| 0);
    anew("another buffer as long", &|| gather(&runs, &other, 0));
    assert!(contents(&other) == gathered(&runs), "into the other buffer");
    // A no-copy buffer: its copy repeated takes its plan, until a rebind gives it other pages.
    let length = rebind::BYTES as u64;
    assert_eq!(length, total, "the page holds the runs' bytes");
    // SAFETY: each page outlives every buffer over it (dropped after the device), and only the
    // buffer's commands use it while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            pages[0].memory(),
            length,
            ResourceOptions::shared(),
            None,
        )
    };
    let over = match made {
        Ok(buffer) => buffer,
        Err(Error::Unsupported(gap)) => {
            eprintln!("HOST-RAM-COPY: the device imports no such pages: {gap:?}");
            return;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    let page_bytes = |page: &rebind::Page| -> Vec<u8> {
        (page.words().iter())
            .flat_map(|word| word.to_le_bytes())
            .collect()
    };
    gather(&runs, &over, 0);
    let before = plans_reused(&device);
    gather(&runs, &over, 0);
    assert_eq!(
        plans_reused(&device) - before,
        1,
        "the no-copy buffer's copy again"
    );
    assert!(page_bytes(&pages[0]) == gathered(&runs), "in its pages");
    assert_eq!(over.detach_backing(), Ok(true));
    // SAFETY: as above; no command that uses the buffer is pending.
    unsafe { over.rebind_bytes_no_copy(pages[1].memory(), None, None) }.unwrap();
    anew("the buffer rebound", &|| gather(&runs, &over, 0));
    assert!(page_bytes(&pages[1]) == gathered(&runs), "in its new pages");
    // Runs each at a byte of their own: the same call again takes its plan too.
    let at_own: Vec<(u64, Range<u64>)> = (runs.iter().enumerate())
        .map(|(n, run)| (n as u64 * PAGE, run.clone()))
        .collect();
    let gather_runs = || {
        blit(&device, |encoder| {
            encoder
                .copy_runs_from_host_ram(&ram, &at_own, &other)
                .unwrap();
        });
    };
    anew("runs at their own bytes", &gather_runs);
    let before = plans_reused(&device);
    gather_runs();
    assert_eq!(
        plans_reused(&device) - before,
        1,
        "runs at their own bytes again"
    );
    assert!(contents(&other) == gathered(&runs), "each at its byte");
    drop((over, used, ram));
}

/// A plan the call named as it was made ([`HostRamCall::kept`]) that the RAM no longer keeps
/// when the copy records again is not taken: eight other copies' plans pushed it out before a
/// rebind, before the first submission, encoded the copy's encoder again. That recording plans
/// the call's runs anew, and the copy moves their bytes.
#[test]
fn a_named_plan_the_ram_let_go_of_before_the_recording_is_not_taken() {
    // Declared before the device, so dropped after it.
    let pages = [rebind::Page::new(), rebind::Page::new()];
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let used = device.use_host_ram(&ram).unwrap();
    let runs = kept_plan_runs(start);
    for run in &runs {
        write_pattern(run.clone(), 61);
    }
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let out = shared(&device, total, |_| 0);
    for _ in 0..2 {
        blit(&device, |encoder| {
            encoder.copy_from_host_ram(&ram, &runs, &out, 0).unwrap();
        });
    }
    let length = rebind::BYTES as u64;
    // SAFETY: each page outlives every buffer over it (dropped after the device), and only the
    // buffer's commands use it while it lives.
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(
            pages[0].memory(),
            length,
            ResourceOptions::shared(),
            None,
        )
    };
    let over = match made {
        Ok(buffer) => buffer,
        Err(Error::Unsupported(gap)) => {
            eprintln!("HOST-RAM-COPY: the device imports no such pages: {gap:?}");
            return;
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    };
    // Its command buffer waits for an event only the host signals, below: the rebind comes
    // before its first submission.
    let gate = device.new_shared_event().unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::Shared(&gate), 1)
        .unwrap();
    let before = device.work_counts();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encoder.copy_from_host_ram(&ram, &runs, &out, 0).unwrap();
    encoder.fill_buffer(&over, 0..length, 0x5a).unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    let named = device.work_counts();
    assert_eq!(
        named.host_ram_copy_plans_reused - before.host_ram_copy_plans_reused,
        1,
        "the call named the plan kept"
    );
    // Eight other copies' plans, recorded, committed after: the RAM keeps theirs alone.
    let spare = shared(&device, PAGE, |_| 0);
    let others = queue.command_buffer().unwrap();
    let mut encoder = others.blit_command_encoder().unwrap();
    for n in 0..8 {
        let page = start + (100 + n) * PAGE;
        encoder
            .copy_from_host_ram(&ram, &one(page..page + PAGE), &spare, 0)
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    assert_eq!(over.detach_backing(), Ok(true));
    // SAFETY: as above; the command buffer that uses it waits for its first submission.
    unsafe { over.rebind_bytes_no_copy(pages[1].memory(), None, None) }.unwrap();
    let rebound = device.work_counts();
    assert_eq!(
        rebound.encoders_reencoded - named.encoders_reencoded,
        1,
        "the rebind encoded the encoder again"
    );
    assert_eq!(
        rebound.host_ram_copies - named.host_ram_copies,
        9,
        "the eight copies and the copy recorded again"
    );
    assert_eq!(
        rebound.host_ram_copy_plans_reused, named.host_ram_copy_plans_reused,
        "the plan let go of not taken: the copy planned anew"
    );
    gate.set_signaled_value(1);
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None)
    );
    assert!(contents(&out) == gathered(&runs), "the gathered pages");
    assert!(
        pages[1].written() && !pages[0].written(),
        "the fill in the pages the buffer lies in now"
    );
    others.commit().unwrap();
    others.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(others.status(), CommandBufferStatus::Completed);
    drop((over, used, ram));
}

/// A chunk let go of between two copies, idle or given back, is imported again by the next copy,
/// whose imports are then not the kept plan's (their serial numbers): it plans anew, and the copy
/// after it takes the new plan. Each moves the pages' bytes as they are then.
#[test]
fn a_chunk_let_go_between_two_copies_takes_the_next_one_to_a_new_plan() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let used = device.use_host_ram(&ram).unwrap();
    let runs = kept_plan_runs(start);
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let out = shared(&device, total, |_| 0);
    let mut seed = 40;
    let gather = |seed: u64| {
        for run in &runs {
            write_pattern(run.clone(), seed);
        }
        blit(&device, |encoder| {
            encoder.copy_from_host_ram(&ram, &runs, &out, 0).unwrap();
        });
        assert!(
            contents(&out) == gathered(&runs),
            "the pages' bytes, seed {seed}"
        );
    };
    gather(seed);
    // Let go of as the hook says once only their slots hold them (the command buffers' holds
    // gone): the chunks the copies imported.
    let let_go = |hook: &dyn Fn() -> usize| {
        let began = Instant::now();
        loop {
            match hook() {
                0 => assert!(began.elapsed() < WAIT, "the chunks stay held"),
                gone => return gone,
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    };
    let releases: [(&str, &dyn Fn() -> usize); 2] = [
        ("idle", &|| device.shared().host_ram.release_idle_now()),
        ("given back", &|| device.shared().host_ram.give_back_now()),
    ];
    for (what, hook) in releases {
        seed += 1;
        let before = plans_reused(&device);
        gather(seed);
        assert_eq!(
            plans_reused(&device) - before,
            1,
            "{what}: the plan taken before"
        );
        let gone = let_go(hook);
        assert_eq!(gone, 2, "{what}: both chunks let go of");
        let imported = device.work_counts().host_ram_chunks_imported;
        seed += 1;
        let before = plans_reused(&device);
        gather(seed);
        assert_eq!(
            plans_reused(&device),
            before,
            "{what}: imported again, planned anew"
        );
        assert_eq!(
            device.work_counts().host_ram_chunks_imported - imported,
            2,
            "{what}: the copy imported both again"
        );
        seed += 1;
        gather(seed);
        assert_eq!(
            plans_reused(&device) - before,
            1,
            "{what}: the new plan taken"
        );
    }
    drop((used, ram));
}

/// A new use of a RAM keeps none of the plans of the use before it: its copies plan anew.
#[test]
fn a_new_use_of_a_ram_plans_its_copies_anew() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let runs = kept_plan_runs(start);
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let out = shared(&device, total, |_| 0);
    let gather = || {
        blit(&device, |encoder| {
            encoder.copy_from_host_ram(&ram, &runs, &out, 0).unwrap();
        });
        assert!(contents(&out) == gathered(&runs), "the pages' bytes");
    };
    for run in &runs {
        write_pattern(run.clone(), 51);
    }
    let used = device.use_host_ram(&ram).unwrap();
    let before = plans_reused(&device);
    gather();
    gather();
    assert_eq!(
        plans_reused(&device) - before,
        1,
        "the first use's plan taken"
    );
    drop(used);
    let used = device.use_host_ram(&ram).unwrap();
    let before = plans_reused(&device);
    gather();
    assert_eq!(plans_reused(&device), before, "the new use plans anew");
    gather();
    assert_eq!(plans_reused(&device) - before, 1, "and takes its own plan");
    drop((used, ram));
}

/// A copy by a kept plan is ordered as one planned anew, by its uses: in one command buffer, a
/// scatter, a fill of its buffer, the same scatter again (the pages end with the fill's bytes);
/// and a gather, a scatter of other bytes into the same pages, the same gather again (the buffer
/// ends with the scatter's bytes). Every copy there but the second scatter's takes a plan kept.
#[test]
fn a_copy_by_a_kept_plan_is_ordered_after_the_gpu_s_writes_before_it() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let used = device.use_host_ram(&ram).unwrap();
    let runs = kept_plan_runs(start);
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let source = shared(&device, total, |at| (at % 241) as u8);
    let other = shared(&device, total, |at| (at % 239) as u8 ^ 0x5a);
    let out = shared(&device, total, |_| 0);
    // Plans kept of the scatter from `source` and the gather into `out`.
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&source, 0, &ram, &runs).unwrap();
        encoder.copy_from_host_ram(&ram, &runs, &out, 0).unwrap();
    });
    let before = plans_reused(&device);
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&source, 0, &ram, &runs).unwrap();
        encoder.fill_buffer(&source, 0..total, 0xa7).unwrap();
        encoder.copy_to_host_ram(&source, 0, &ram, &runs).unwrap();
    });
    assert_eq!(
        plans_reused(&device) - before,
        2,
        "both scatters by the plan kept"
    );
    assert!(
        gathered(&runs).iter().all(|&byte| byte == 0xa7),
        "the second scatter read the fill's bytes"
    );
    let before = plans_reused(&device);
    blit(&device, |encoder| {
        encoder.copy_from_host_ram(&ram, &runs, &out, 0).unwrap();
        encoder.copy_to_host_ram(&other, 0, &ram, &runs).unwrap();
        encoder.copy_from_host_ram(&ram, &runs, &out, 0).unwrap();
    });
    assert_eq!(
        plans_reused(&device) - before,
        2,
        "both gathers by the plan kept"
    );
    assert!(
        contents(&out) == contents(&other),
        "the second gather read the scatter's bytes"
    );
    assert!(
        gathered(&runs) == contents(&other),
        "the scatter's bytes in the pages"
    );
    drop((used, ram));
}

/// Twelve pages a page apart in each chunk of the test's RAM, as runs, and the page after each.
fn long_runs(start: u64) -> (Vec<Range<u64>>, Vec<Range<u64>>) {
    let page = |n: u64| start + n * PAGE;
    let runs: Vec<Range<u64>> = ([20, SECOND + 20].into_iter())
        .flat_map(|first| (0..12).map(move |n| page(first + 2 * n)..page(first + 2 * n + 1)))
        .collect();
    let gaps = (runs.iter()).map(|run| run.end..run.end + PAGE).collect();
    (runs, gaps)
}

/// A scatter of more runs of each chunk than the hazard order keeps of a resource, the
/// encoder's first command over the chunks, is ordered as one use of each
/// (`EncoderUses::order_joined`): a gather of its pages and the gaps between them after it in the
/// same encoder reads the scatter's bytes and the gaps' own; a scatter into the gaps after that,
/// its chunks no longer new to the encoder (ordered by its uses), leaves that gather's bytes
/// alone; a last gather reads both scatters'.
#[test]
fn copies_over_a_long_scatter_s_pages_and_gaps_in_its_encoder_keep_their_order() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let used = device.use_host_ram(&ram).unwrap();
    let (runs, gaps) = long_runs(start);
    let pairs: Vec<Range<u64>> = (runs.iter()).map(|run| run.start..run.end + PAGE).collect();
    for pair in &pairs {
        write_pattern(pair.clone(), 40);
    }
    let gaps_before = gathered(&gaps);
    let total = runs.len() as u64 * PAGE;
    let first = shared(&device, total, |at| pattern(at, 41));
    let second = shared(&device, total, |at| pattern(at, 42));
    let read_first = shared(&device, 2 * total, |_| 0);
    let read_last = shared(&device, 2 * total, |_| 0);
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&first, 0, &ram, &runs).unwrap();
        encoder
            .copy_from_host_ram(&ram, &pairs, &read_first, 0)
            .unwrap();
        encoder.copy_to_host_ram(&second, 0, &ram, &gaps).unwrap();
        encoder
            .copy_from_host_ram(&ram, &pairs, &read_last, 0)
            .unwrap();
    });
    let page = PAGE as usize;
    let pages = |bytes: &[u8], at: usize| bytes[at * page..(at + 1) * page].to_vec();
    let (first, second) = (contents(&first), contents(&second));
    let laid = |gap: &dyn Fn(usize) -> Vec<u8>| -> Vec<u8> {
        (0..runs.len())
            .flat_map(|at| [pages(&first, at), gap(at)].concat())
            .collect()
    };
    assert!(
        contents(&read_first) == laid(&|at| pages(&gaps_before, at)),
        "the gather after the scatter: its pages, the gaps as they were"
    );
    assert!(
        contents(&read_last) == laid(&|at| pages(&second, at)),
        "the last gather: both scatters' pages"
    );
    assert!(
        gathered(&pairs) == contents(&read_last),
        "the RAM holds both scatters' pages"
    );
    drop((used, ram));
}

/// A scatter of more runs of each chunk than the hazard order keeps of a resource
/// (`EncoderUses::order_joined` orders them as one use of each) reports every run to the RAM's
/// log and no page between them: the dirty log takes the command's own uses
/// (`EncoderCore::track`).
#[test]
fn a_long_scatter_reports_each_run_to_the_ram_s_log_and_no_page_between_them() {
    let Some(device) = device() else {
        return;
    };
    let log = Arc::new(Log::default());
    let (ram, start) = new_ram(Some(Arc::clone(&log)));
    let used = device.use_host_ram(&ram).unwrap();
    let (runs, gaps) = long_runs(start);
    let source = shared(&device, runs.len() as u64 * PAGE, |at| pattern(at, 44));
    let told = log.writes.lock().unwrap().len();
    blit(&device, |encoder| {
        encoder.copy_to_host_ram(&source, 0, &ram, &runs).unwrap();
    });
    let writes: Vec<Range<u64>> = log.writes.lock().unwrap()[told..]
        .iter()
        .map(|(_, range, _)| range.clone())
        .collect();
    for run in &runs {
        assert!(
            (writes.iter()).any(|write| write.start <= run.start && run.end <= write.end),
            "{run:x?} reported in {writes:x?}"
        );
    }
    for gap in &gaps {
        assert!(
            (writes.iter()).all(|write| write.end <= gap.start || gap.end <= write.start),
            "{gap:x?} not reported, in {writes:x?}"
        );
    }
    assert!(
        gathered(&runs) == contents(&source),
        "the runs hold the scatter's bytes"
    );
    drop((used, ram));
}

/// The plans a device's RAM put their table in a buffer of their own for.
fn tables_kept(device: &Device) -> u64 {
    device.work_counts().host_ram_copy_tables_kept
}

/// A plan puts its table in a buffer of its own at its first reuse
/// (`WorkCounts::host_ram_copy_tables_kept`) and each reuse after it binds that buffer: none
/// made again, and each copy moves the pages' bytes as they are now.
#[test]
fn a_plan_keeps_its_table_in_a_buffer_from_its_first_reuse() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let used = device.use_host_ram(&ram).unwrap();
    let runs = kept_plan_runs(start);
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let out = shared(&device, total, |_| 0);
    let gather = |seed: u64| {
        for run in &runs {
            write_pattern(run.clone(), seed);
        }
        let before = (plans_reused(&device), tables_kept(&device));
        blit(&device, |encoder| {
            encoder.copy_from_host_ram(&ram, &runs, &out, 0).unwrap();
        });
        assert!(
            contents(&out) == gathered(&runs),
            "the pages' bytes, seed {seed}"
        );
        (
            plans_reused(&device) - before.0,
            tables_kept(&device) - before.1,
        )
    };
    assert_eq!(gather(70), (0, 0), "planned: the table uploaded");
    assert_eq!(gather(71), (1, 1), "the first reuse: its table kept");
    assert_eq!(gather(72), (1, 0), "a reuse after it: the kept table bound");
    assert_eq!(gather(73), (1, 0), "and again");
    drop((out, used, ram));
}

/// A command buffer that bound a plan's kept table holds it: the plan let go of before the
/// command buffer runs (eight other copies' plans pushed it out of the RAM's), the copy reads
/// its table as it was and moves the pages' bytes.
#[test]
fn a_kept_table_a_command_buffer_bound_outlives_its_plan() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let used = device.use_host_ram(&ram).unwrap();
    let runs = kept_plan_runs(start);
    for run in &runs {
        write_pattern(run.clone(), 74);
    }
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let out = shared(&device, total, |_| 0);
    for _ in 0..2 {
        blit(&device, |encoder| {
            encoder.copy_from_host_ram(&ram, &runs, &out, 0).unwrap();
        });
    }
    // Its command buffer waits for an event only the host signals, below.
    let gate = device.new_shared_event().unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(EventRef::Shared(&gate), 1)
        .unwrap();
    let cleared = shared(&device, total, |_| 0);
    let before = device.work_counts();
    let mut encoder = command_buffer.blit_command_encoder().unwrap();
    encoder
        .copy_from_host_ram(&ram, &runs, &cleared, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    let after = device.work_counts();
    // The buffer is another: the copy plans anew, its plan kept; the same copy again reuses it.
    let again = queue.command_buffer().unwrap();
    again
        .encode_wait_for_event(EventRef::Shared(&gate), 1)
        .unwrap();
    let mut encoder = again.blit_command_encoder().unwrap();
    encoder
        .copy_from_host_ram(&ram, &runs, &cleared, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    again.commit().unwrap();
    let bound = device.work_counts();
    assert_eq!(
        [
            after.host_ram_copy_plans_reused - before.host_ram_copy_plans_reused,
            bound.host_ram_copy_plans_reused - after.host_ram_copy_plans_reused,
            bound.host_ram_copy_tables_kept - after.host_ram_copy_tables_kept,
        ],
        [0, 1, 1],
        "planned, then reused with its table kept"
    );
    // Eight other copies' plans, each of a page of its own: the RAM keeps theirs alone.
    let spare = shared(&device, PAGE, |_| 0);
    for n in 0..8 {
        let page = start + (100 + n) * PAGE;
        blit(&device, |encoder| {
            encoder
                .copy_from_host_ram(&ram, &one(page..page + PAGE), &spare, 0)
                .unwrap();
        });
    }
    let expected = gathered(&runs);
    gate.set_signaled_value(1);
    for waited in [&command_buffer, &again] {
        waited.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(
            (waited.status(), waited.error()),
            (CommandBufferStatus::Completed, None)
        );
    }
    assert!(
        contents(&cleared) == expected,
        "the pages, by the kept table"
    );
    drop((out, cleared, spare, used, ram));
}

/// A device's recovery carries its use of a RAM to a new one on the next generation, whose plans
/// and their tables start anew: a copy kept and reused before the recovery is planned anew after
/// it, and its first reuse there keeps a table of that generation. The copies after it go to a
/// buffer made after it: a `Shared` buffer of pooled memory stays the lost generation's
/// (`MadeBeforeRecovery`, `device/carry.rs`), as it does where held imports tax submissions.
#[test]
fn a_recovery_starts_a_ram_s_plans_and_their_tables_anew() {
    let Some(device) = device() else {
        return;
    };
    let (ram, start) = new_ram(None);
    let used = device.use_host_ram(&ram).unwrap();
    let runs = kept_plan_runs(start);
    let total: u64 = runs.iter().map(|run| run.end - run.start).sum();
    let gather = |out: &Buffer, seed: u64| {
        for run in &runs {
            write_pattern(run.clone(), seed);
        }
        let before = (plans_reused(&device), tables_kept(&device));
        blit(&device, |encoder| {
            encoder.copy_from_host_ram(&ram, &runs, out, 0).unwrap();
        });
        assert!(
            contents(out) == gathered(&runs),
            "the pages' bytes, seed {seed}"
        );
        (
            plans_reused(&device) - before.0,
            tables_kept(&device) - before.1,
        )
    };
    let lost = shared(&device, total, |_| 0);
    assert_eq!(gather(&lost, 75), (0, 0), "planned");
    assert_eq!(gather(&lost, 76), (1, 1), "reused, its table kept");
    if let Err(error) = device.recover_for_test() {
        eprintln!("HOST-RAM-COPY: the device does not recover here ({error:?}); nothing more");
        return;
    }
    drop(lost);
    let out = shared(&device, total, |_| 0);
    // The new generation's thread imports the RAM's chunks ahead, so a copy before its chunks
    // are imported is staged and keeps no plan: copies until one reuses its plan. That first
    // reuse keeps a table of its own (where the lost generation's plan were taken, it would
    // bind that plan's table and keep none).
    let began = Instant::now();
    let mut seed = 77;
    let reused = loop {
        let counts = gather(&out, seed);
        seed += 1;
        if counts.0 > 0 {
            break counts;
        }
        assert!(
            began.elapsed() < WAIT,
            "no copy reused its plan after the recovery"
        );
    };
    assert_eq!(
        reused,
        (1, 1),
        "its first reuse there keeps a table of its own"
    );
    assert_eq!(gather(&out, seed), (1, 0), "bound after it");
    drop((out, used, ram));
}
