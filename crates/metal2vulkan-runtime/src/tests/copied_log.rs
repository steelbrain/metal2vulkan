//! Copied host buffers over pages whose memory came with a log of the writes to them
//! (`copied.rs`, `dirty_pages.rs`), on plain memory:
//!
//! - a merge back reports the bytes it wrote to the pages' logs as the copy's own writes, a block
//!   at a time, through both the memory's own log and its guest RAM's (whose offsets are host
//!   addresses, through the runs of the RAM the pages are); a merge that writes nothing, and one
//!   whose writes are dropped, report nothing;
//! - a copy in compares only the pages the log reports written since the copy's last, after a
//!   first that compares every byte, and every byte where the log cannot say, and the bytes no
//!   log keeps at every one; one stamp a copy in serves every copy;
//! - a copy takes another copy's merge back over the same pages, and not its own;
//! - a CPU write to a byte the GPU did not change, racing both ways, survives, logged or not.
//!
//! The log is the test's own ([`TestLog`]): it reports the writes it is told of, and lies by
//! leaving out one it is not told, so a block it leaves out shows whether a copy in took it.

use crate::copied::{Copied, CopiedBuffers, Written, SYNC_BLOCK};
use crate::dirty_pages::{hook, DirtySource, DirtyTally, PagesLog};
use crate::work_counts::Counters;
use crate::{DirtyPages, DirtyStamp};
use std::num::NonZeroU64;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const BLOCK: u64 = SYNC_BLOCK as u64;

/// A test's log: every write it is told of, stamped later than every stamp asked before, with
/// its writer (0 the consumer's); the pages it was asked to watch.
#[derive(Default)]
pub(super) struct TestLog {
    next: AtomicU64,
    pub(super) writes: Mutex<Vec<(u64, Range<u64>, u64)>>,
    /// The wait answers `false` while set.
    pub(super) unanswered: AtomicBool,
    pub(super) waits: AtomicU64,
    pub(super) watched: Mutex<Vec<Range<u64>>>,
    /// The questions `changed` and `changed_except` were asked.
    pub(super) asked: AtomicU64,
    pub(super) asked_except: AtomicU64,
}

impl TestLog {
    /// A write of `range` by `writer`, stamped now.
    pub(super) fn told(&self, range: Range<u64>, writer: u64) {
        let stamp = self.next.fetch_add(1, Ordering::SeqCst) + 1;
        self.writes.lock().unwrap().push((stamp, range, writer));
    }

    /// The writes reported by writers other than 0 (the runtime's named ones), in order.
    pub(super) fn named(&self) -> Vec<(Range<u64>, u64)> {
        (self.writes.lock().unwrap().iter())
            .filter(|w| w.2 != 0)
            .map(|w| (w.1.clone(), w.2))
            .collect()
    }
}

impl DirtyPages for TestLog {
    type Watch = ();

    fn watch(&self, range: Range<u64>) {
        self.watched.lock().unwrap().push(range);
    }

    fn request(&self) -> DirtyStamp {
        DirtyStamp(self.next.fetch_add(1, Ordering::SeqCst) + 1)
    }

    fn wait(&self, _stamp: DirtyStamp) -> bool {
        self.waits.fetch_add(1, Ordering::SeqCst);
        !self.unanswered.load(Ordering::SeqCst)
    }

    fn changed(&self, since: DirtyStamp, out: &mut Vec<Range<u64>>) {
        self.asked.fetch_add(1, Ordering::SeqCst);
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
        self.asked_except.fetch_add(1, Ordering::SeqCst);
        let writes = self.writes.lock().unwrap();
        let changed = (writes.iter()).filter(|w| w.0 > since.0 && w.2 != writer.get());
        out.extend(changed.map(|w| w.1.clone()));
    }
}

/// `bytes` as a pointer the comparisons read or write.
pub(super) fn at(bytes: &mut [u8]) -> NonNull<u8> {
    NonNull::new(bytes.as_mut_ptr()).unwrap()
}

/// A plain read of the pages, the copy's first.
pub(super) fn read(from: NonNull<u8>, into: &mut [u8]) -> bool {
    // SAFETY: the tests' pages are `into.len()` bytes of plain memory.
    unsafe { std::ptr::copy_nonoverlapping(from.as_ptr(), into.as_mut_ptr(), into.len()) };
    true
}

/// A merge back reports, to the memory's own log and to the guest RAM's, each block it wrote a
/// byte of, as the copy's own writer (one no other copy has), and no block it wrote nothing of:
/// not one whose only changed byte the CPU changed too, and nothing of a copy whose writes are
/// dropped.
#[test]
fn a_copy_s_merge_back_reports_the_blocks_it_wrote_as_its_own() {
    let length = 5 * BLOCK as usize + 100;
    let mut pages: Vec<u8> = (0..length).map(|i| (i * 3 + 1) as u8).collect();
    let mut copy = vec![0_u8; length];
    let pointer = pages.as_ptr() as u64;
    let own = Arc::new(TestLog::default());
    let ram = Arc::new(TestLog::default());
    let own_source = DirtySource::new(hook(Arc::clone(&own)), pointer);
    let ram_source = DirtySource::new(hook(Arc::clone(&ram)), 0);
    // The import window: two runs of guest RAM (host addresses), the bytes 0x123 into it.
    let (first, second) = (0x10_0000..0x10_2000, 0x40_0000..0x40_6000);
    let window = [first.clone(), second.clone()];
    let log = PagesLog::new(
        (pointer, length as u64),
        Some(&own_source),
        Some((&ram_source, &window[..], 0x123)),
    )
    .unwrap();
    let buffers = Arc::new(CopiedBuffers::default());
    let counts = Counters::default();
    // SAFETY: the copy and the pages are the test's, of `length` bytes, live past the answer.
    let copied = unsafe { buffers.add_reading(at(&mut pages), at(&mut copy), length, read, false) };
    let copied = copied.unwrap();
    copied.set_log(&log);
    // The GPU writes a byte of blocks 0, 1, 2 and 5 (the last, short); in block 3 it writes a
    // byte the CPU wrote too since, which the merge keeps the CPU's of, writing nothing there.
    copy[10] ^= 0xFF;
    copy[BLOCK as usize + 1] ^= 0xFF;
    copy[2 * BLOCK as usize + 5] ^= 0xFF;
    copy[length - 1] ^= 0xFF;
    copy[3 * BLOCK as usize + 7] ^= 0xFF;
    pages[3 * BLOCK as usize + 7] ^= 0x0F;
    buffers.merge_back(&counts, &Written::all());
    assert_eq!(pages[10], copy[10], "merged back");
    let own_named = own.named();
    let writer = own_named.first().map(|w| w.1).expect("reported");
    // Blocks 0 to 2 are one run of blocks.
    let blocks = [0..3 * BLOCK, 5 * BLOCK..length as u64];
    let want: Vec<(Range<u64>, u64)> = blocks.iter().map(|b| (b.clone(), writer)).collect();
    assert_eq!(
        own_named, want,
        "the memory's log: each block written, the copy's writer"
    );
    // The guest RAM's log: the same blocks as host addresses, through the runs; the window's
    // bytes 0..0x2000 are the first run, the rest the second, so block 1 lies in both.
    let host = |byte: u64| match byte + 0x123 {
        w if w < 0x2000 => first.start + w,
        w => second.start + (w - 0x2000),
    };
    let ram_named = ram.named();
    let want_ram: Vec<(Range<u64>, u64)> = vec![
        (host(0)..first.end, writer),
        (second.start..host(3 * BLOCK - 1) + 1, writer),
        (host(5 * BLOCK)..host(length as u64 - 1) + 1, writer),
    ];
    assert_eq!(ram_named, want_ram, "the RAM's log, through its runs");
    // A merge that writes nothing reports nothing.
    buffers.merge_back(&counts, &Written::all());
    assert_eq!(own.named().len(), 2);
    // Another copy of the same pages is another writer.
    let mut other_copy = vec![0_u8; length];
    // SAFETY: as above.
    let other =
        unsafe { buffers.add_reading(at(&mut pages), at(&mut other_copy), length, read, false) };
    let other = other.unwrap();
    other.set_log(&log);
    other_copy[0] ^= 0xFF;
    buffers.merge_back(&counts, &Written::all());
    let last = own.named().last().cloned().unwrap();
    assert_eq!(last.0, 0..BLOCK);
    assert_ne!(last.1, writer, "a writer of its own");
    drop((copied, other));
    // A copy whose GPU writes are dropped writes nothing to the pages, and reports nothing.
    let mut dropping_copy = vec![0_u8; length];
    // SAFETY: as above.
    let dropping =
        unsafe { buffers.add_reading(at(&mut pages), at(&mut dropping_copy), length, read, true) };
    let dropping = dropping.unwrap();
    dropping.set_log(&log);
    let before = (
        own.writes.lock().unwrap().len(),
        ram.writes.lock().unwrap().len(),
    );
    dropping_copy[50] ^= 0xFF;
    buffers.merge_back(&counts, &Written::all());
    let after = (
        own.writes.lock().unwrap().len(),
        ram.writes.lock().unwrap().len(),
    );
    assert_eq!(before, after, "dropped writes reach no page and no log");
    drop(dropping);
}

/// The test's pages and a copy of them, of `blocks` blocks, under `buffers`, logged by `log`: the
/// pages' bytes all different from the copy's start.
struct Pair {
    pages: Vec<u8>,
    copy: Vec<u8>,
}

impl Pair {
    fn new(blocks: usize) -> Pair {
        let length = blocks * BLOCK as usize;
        Pair {
            pages: (0..length).map(|i| (i * 7 + 3) as u8).collect(),
            copy: vec![0; length],
        }
    }

    /// Its copy, registered in `buffers`, logged by `log` where given.
    fn copied(&mut self, buffers: &Arc<CopiedBuffers>, log: Option<&Arc<PagesLog>>) -> Copied {
        let length = self.pages.len();
        // SAFETY: the copy and the pages are the test's, of `length` bytes, live past the answer
        // (each test drops the answer first).
        let copied = unsafe {
            buffers.add_reading(at(&mut self.pages), at(&mut self.copy), length, read, false)
        };
        let copied = copied.unwrap();
        if let Some(log) = log {
            copied.set_log(log);
        }
        copied
    }
}

/// A log of `length` bytes at `pointer`: their memory's own, `log`.
fn own_log(log: &Arc<TestLog>, pointer: u64, length: usize) -> Arc<PagesLog> {
    let source = DirtySource::new(hook(Arc::clone(log)), pointer);
    PagesLog::new((pointer, length as u64), Some(&source), None).unwrap()
}

/// A logged copy's first copy in compares every byte (a write before its watch began included);
/// each later one compares only the blocks the log reports written since: a block the log was
/// told of is taken, one it was not told of is not (the log lies, so the copy in shows it asked).
/// A wait the log answers `false` makes the copy in compare every byte, and the next one too.
#[test]
fn a_logged_copy_in_compares_only_the_pages_the_log_reports() {
    let mut pair = Pair::new(6);
    let pointer = pair.pages.as_ptr() as u64;
    let own = Arc::new(TestLog::default());
    let log = own_log(&own, pointer, pair.pages.len());
    let buffers = Arc::new(CopiedBuffers::default());
    let (counts, tally) = (Counters::default(), DirtyTally::default());
    let copied = pair.copied(&buffers, Some(&log));
    assert_eq!(
        *own.watched.lock().unwrap(),
        vec![0..6 * BLOCK],
        "its pages watched"
    );
    let (b, untold) = (BLOCK as usize, 3 * BLOCK as usize + 9);
    // Written before the first copy in, told to no log.
    pair.pages[5 * b + 3] ^= 0xFF;
    buffers.copy_in(&counts, &tally);
    assert_eq!(
        pair.copy[5 * b + 3],
        pair.pages[5 * b + 3],
        "the first compares every byte"
    );
    // A byte of block 1, told; one of block 3, not told.
    pair.pages[b + 9] ^= 0xFF;
    own.told(BLOCK..2 * BLOCK, 0);
    pair.pages[untold] ^= 0xFF;
    buffers.copy_in(&counts, &tally);
    assert_eq!(pair.copy[b + 9], pair.pages[b + 9], "the told block taken");
    assert_ne!(
        pair.copy[untold], pair.pages[untold],
        "the block the log left out not compared"
    );
    // The log cannot say: every byte compared, and again at the next copy in, which has no
    // stamp to count on.
    own.unanswered.store(true, Ordering::SeqCst);
    buffers.copy_in(&counts, &tally);
    assert_eq!(pair.copy[untold], pair.pages[untold], "every byte compared");
    own.unanswered.store(false, Ordering::SeqCst);
    pair.pages[untold] ^= 0xFF;
    buffers.copy_in(&counts, &tally);
    assert_eq!(
        pair.copy[untold], pair.pages[untold],
        "every byte compared again"
    );
    pair.pages[untold] ^= 0xFF;
    buffers.copy_in(&counts, &tally);
    assert_ne!(pair.copy[untold], pair.pages[untold], "gated again");
    drop(copied);
}

/// One copy in of many copies over one log asks one stamp and makes at most one wait, and asks
/// the log which pages changed since that stamp once for all of them.
#[test]
fn one_stamp_and_one_question_serve_every_copy_of_a_copy_in() {
    let own = Arc::new(TestLog::default());
    let buffers = Arc::new(CopiedBuffers::default());
    let (counts, tally) = (Counters::default(), DirtyTally::default());
    let mut pairs: Vec<Pair> = (0..8).map(|_| Pair::new(2)).collect();
    let copies: Vec<Copied> = (pairs.iter_mut())
        .map(|pair| {
            let log = own_log(&own, pair.pages.as_ptr() as u64, pair.pages.len());
            pair.copied(&buffers, Some(&log))
        })
        .collect();
    // Each memory its own source over one log: the stamps are the log's.
    for round in 0..3 {
        let stamps = tally.stamps.load(Ordering::SeqCst);
        let asked = own.asked.load(Ordering::SeqCst);
        buffers.copy_in(&counts, &tally);
        let stamps = tally.stamps.load(Ordering::SeqCst) - stamps;
        let asked = own.asked.load(Ordering::SeqCst) - asked;
        // Eight memories, each its own source: one stamp of each, the question once each.
        assert_eq!(stamps, 8, "round {round}: one stamp a source");
        let questions = if round == 0 { 0 } else { 8 };
        assert_eq!(
            asked, questions,
            "round {round}: one question a source and stamp"
        );
    }
    drop(copies);
    // One source for every copy (a guest RAM's log, offsets host addresses): one stamp, one
    // question, for all of them.
    let ram = Arc::new(TestLog::default());
    let source = DirtySource::new(hook(Arc::clone(&ram)), 0);
    let copies: Vec<Copied> = (pairs.iter_mut())
        .map(|pair| {
            let (start, length) = (pair.pages.as_ptr() as u64, pair.pages.len() as u64);
            let log = PagesLog::new((start, length), Some(&source), None).unwrap();
            pair.copied(&buffers, Some(&log))
        })
        .collect();
    for round in 0..3 {
        let stamps = tally.stamps.load(Ordering::SeqCst);
        let asked = ram.asked.load(Ordering::SeqCst);
        buffers.copy_in(&counts, &tally);
        assert_eq!(tally.stamps.load(Ordering::SeqCst) - stamps, 1, "one stamp");
        let questions = if round == 0 { 0 } else { 1 };
        assert_eq!(
            ram.asked.load(Ordering::SeqCst) - asked,
            questions,
            "one question"
        );
    }
    drop(copies);
}

/// Two copies over the same pages, under one log: the GPU's write into one, merged back, is
/// reported as its writer's, so the other's next copy in takes it. The merging copy's own next
/// copy in takes the pages every writer wrote, asked once for every copy, after a small merge,
/// and asks apart for every writer's pages but its own (`changed_except`) after a merge of
/// `ASKS_APART` bytes or more.
#[test]
fn a_copy_takes_another_copy_s_merge_back_over_the_same_pages() {
    // 2 MiB: room for a merge of 1 MiB, a byte in each of 256 blocks.
    let mut first = Pair::new(512);
    let pointer = first.pages.as_ptr() as u64;
    let length = first.pages.len();
    let own = Arc::new(TestLog::default());
    let log = own_log(&own, pointer, length);
    let buffers = Arc::new(CopiedBuffers::default());
    let (counts, tally) = (Counters::default(), DirtyTally::default());
    let one = first.copied(&buffers, Some(&log));
    // The second copy is of the same pages.
    let mut second_copy = vec![0_u8; length];
    // SAFETY: as in `Pair::copied`.
    let second = unsafe {
        buffers.add_reading(
            at(&mut first.pages),
            at(&mut second_copy),
            length,
            read,
            false,
        )
    };
    let second = second.unwrap();
    second.set_log(&log);
    buffers.copy_in(&counts, &tally);
    for blocks in [1_usize, 256] {
        // The GPU writes a byte of each of `blocks` blocks of the first copy; the completion
        // merges them back.
        let written: Vec<usize> = (0..blocks)
            .map(|b| (b * 2 + 1) * BLOCK as usize + 17)
            .collect();
        for &byte in &written {
            first.copy[byte] ^= 0xFF;
        }
        buffers.merge_back(&counts, &Written::all());
        assert!(
            written.iter().all(|&b| first.pages[b] == first.copy[b]),
            "merged back"
        );
        let except = own.asked_except.load(Ordering::SeqCst);
        buffers.copy_in(&counts, &tally);
        assert!(
            written.iter().all(|&b| second_copy[b] == first.pages[b]),
            "{blocks} blocks: the other copy took the merge"
        );
        let apart = u64::from(blocks * BLOCK as usize >= 1 << 20);
        assert_eq!(
            own.asked_except.load(Ordering::SeqCst) - except,
            apart,
            "{blocks} blocks: asked apart only after a merge of a MiB or more"
        );
    }
    drop((one, second));
}

/// Bytes no log keeps the writes to (a guest RAM's log whose runs hold only the first two of a
/// copy's four blocks) are compared at every copy in, the logged ones only where written.
#[test]
fn bytes_no_log_keeps_are_compared_at_every_copy_in() {
    let mut pair = Pair::new(4);
    let pointer = pair.pages.as_ptr() as u64;
    let ram = Arc::new(TestLog::default());
    let source = DirtySource::new(hook(Arc::clone(&ram)), 0);
    // The window: one run of host addresses holding the copy's first two blocks.
    let run = pointer..pointer + 2 * BLOCK;
    let window = std::slice::from_ref(&run);
    let log = PagesLog::new((pointer, 4 * BLOCK), None, Some((&source, window, 0))).unwrap();
    let buffers = Arc::new(CopiedBuffers::default());
    let (counts, tally) = (Counters::default(), DirtyTally::default());
    let copied = pair.copied(&buffers, Some(&log));
    buffers.copy_in(&counts, &tally);
    let (logged, unlogged) = (BLOCK as usize + 5, 3 * BLOCK as usize + 5);
    pair.pages[logged] ^= 0xFF;
    pair.pages[unlogged] ^= 0xFF;
    buffers.copy_in(&counts, &tally);
    assert_ne!(
        pair.copy[logged], pair.pages[logged],
        "a logged block the log left out"
    );
    assert_eq!(
        pair.copy[unlogged], pair.pages[unlogged],
        "the unlogged block compared"
    );
    drop(copied);
}

/// A CPU thread writing one byte of the pages over and over, which the GPU never writes, while
/// copies in and merges back run, the GPU writing the bytes around it: the byte ends holding the
/// CPU's last write in the pages, and in the copy after the next copy in, both with the CPU's
/// writes told to a log and with no log.
#[test]
fn a_cpu_write_the_gpu_did_not_make_survives_both_ways_logged_or_not() {
    for logged in [false, true] {
        let mut pair = Pair::new(2);
        let pointer = pair.pages.as_ptr() as u64;
        let length = pair.pages.len();
        let own = Arc::new(TestLog::default());
        let log = own_log(&own, pointer, length);
        let buffers = Arc::new(CopiedBuffers::default());
        let (counts, tally) = (Counters::default(), DirtyTally::default());
        let copied = pair.copied(&buffers, logged.then_some(&log));
        buffers.copy_in(&counts, &tally);
        let cpu_byte = BLOCK as usize + 100;
        let stop = Arc::new(AtomicBool::new(false));
        let pages = pair.pages.as_mut_ptr() as usize;
        let writer = {
            let (stop, own) = (Arc::clone(&stop), Arc::clone(&own));
            std::thread::spawn(move || {
                let mut value = 0u8;
                while !stop.load(Ordering::SeqCst) {
                    value = value.wrapping_add(1) | 1;
                    // SAFETY: a byte of the test's pages, which outlive the thread (joined
                    // below); the runtime's reads of it race as the CPU's and the GPU's do.
                    unsafe { std::ptr::write_volatile((pages as *mut u8).add(cpu_byte), value) };
                    own.told(BLOCK..2 * BLOCK, 0);
                }
                value
            })
        };
        let copy = pair.copy.as_mut_ptr();
        for round in 0..2000_usize {
            // The GPU writes the bytes beside the CPU's (in the copy), and never the CPU's.
            for at in [cpu_byte - 1, cpu_byte + 1, round % (BLOCK as usize)] {
                // SAFETY: bytes of the test's copy.
                unsafe { *copy.add(at) = (*copy.add(at)).wrapping_add(1) };
            }
            buffers.merge_back(&counts, &Written::all());
            buffers.copy_in(&counts, &tally);
        }
        stop.store(true, Ordering::SeqCst);
        let last = writer.join().unwrap();
        buffers.copy_in(&counts, &tally);
        assert_eq!(
            pair.pages[cpu_byte], last,
            "logged {logged}: the pages keep the CPU's"
        );
        assert_eq!(
            pair.copy[cpu_byte], last,
            "logged {logged}: the copy takes the CPU's"
        );
        // The GPU's bytes beside it reached the pages.
        assert_eq!(
            pair.pages[cpu_byte - 1],
            pair.copy[cpu_byte - 1],
            "logged {logged}"
        );
        assert_eq!(
            pair.pages[cpu_byte + 1],
            pair.copy[cpu_byte + 1],
            "logged {logged}"
        );
        drop(copied);
    }
}
