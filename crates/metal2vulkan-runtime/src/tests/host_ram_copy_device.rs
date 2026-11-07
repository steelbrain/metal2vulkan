//! Copies between a guest RAM and a buffer on a device (`host_ram/copy.rs`):
//! `BlitCommandEncoder::copy_from_host_ram` and `copy_to_host_ram`.
//!
//! - **Both ways, byte for byte.** A gather of ascending runs (adjacent ones merged), descending
//!   neighbours, a run across a chunk's end, a run in the second chunk and a page named twice
//!   lands in a `Shared` and a `Private` buffer as the CPU's view of the pages reads; a scatter
//!   back writes each run's bytes and no other byte of the RAM. One dispatch per chunk the runs
//!   touch, and each chunk imported by the copy where no object imported it (the device's
//!   thread held back).
//! - **The CPU's writes before the commit** are in a gather encoded before them, and a scatter's
//!   bytes are in the pages once the command buffer completed.
//! - **Ordered against a no-copy buffer over the same pages**, in one command buffer: a buffer
//!   over the RAM's own addresses (bound in its chunk) read after a scatter, in the same encoder
//!   and in a later one, reads the scatter's bytes, and a gather after a fill through it reads
//!   the fill's; in one encoder a barrier is recorded between them.
//! - **A copied backing over the pages** (the import gate refused, the RAM's log given) takes a
//!   scatter's bytes at the next commit, the log told of them; a gather after its merge back
//!   reads what the GPU wrote through it.
//! - **Refused**: runs off the page, empty or outside the RAM, a RAM the device does not use (or
//!   no longer does), bytes past the buffer and an offset off a word; no runs copy nothing.
//! - **A chunk taken as an object takes it**, on a device whose thread imports ahead: a gather
//!   from the RAM's first chunk finds it imported ahead and opens the full window past it, and a
//!   scatter into a chunk past the window imports its page alone while the thread imports the
//!   chunk (on a device with `AliasedImportBlitsUnordered`, the chunk itself); the thread imports
//!   nothing past the window.
//!
//! Every other test holds the device's thread back (`Device::hold_host_ram_imports`).
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. A device that imports no host memory says so and passes.

use crate::host_ram::AHEAD_CHUNKS;
use crate::{
    Buffer, CommandBuffer, CommandBufferStatus, Device, DeviceKind, DeviceOptions, DirtyPages,
    DirtyStamp, Error, HostBacking, HostMemory, HostRam, ImportGate, KeepAlive, Problem,
    ResourceOptions, RunResolver,
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

/// A device of the cfg's kind, as the product makes it; `None` where it imports no host memory.
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
        after.host_ram_chunks_imported - before.host_ram_chunks_imported,
        2,
        "the copy imported the chunks no object had"
    );
    // Into a `Private` buffer, read back by a blit.
    let private = device
        .new_buffer(total, ResourceOptions::private())
        .unwrap();
    let back = shared(&device, total, |_| 0);
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

#[test]
fn copies_off_the_page_outside_the_ram_or_past_the_buffer_are_refused() {
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
        one(page(1) + 4..page(2) + 4),
        one(page(1)..page(1) + 100),
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
    // No runs copy nothing; the encoder goes on.
    encoder.copy_from_host_ram(&ram, &[], &out, 0).unwrap();
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
    drop((used, ram));
}
