//! The CPU cost of keeping copied host buffers coherent (`copied.rs`), on plain memory: a
//! commit's copy in and a batch completion's merge back over 680 MiB of live copies, the shape a
//! consumer's 4K desktop held (one 64 MiB surface and 2,048 copies of 308 KiB), in three cases:
//! (a) nothing changed, (b) 1% of the pages changed on each side, (c) the 64 MiB copy redrawn
//! whole by the GPU and its pages rewritten by the CPU. Each runs three times: with no log of the
//! pages' writes, each merge back taking every copy as written; with one log over every copy's
//! pages (a guest RAM's, its offsets host addresses, [`BenchLog`]) told of each CPU write, as a
//! consumer's dirty log would be, likewise; and logged, each merge back taking as written only
//! the copies the GPU wrote, as their command buffers' uses would name them (`Written`). Each
//! prints its median over its runs (`COPIED-BENCH`). Ignored: run it in release,
//! `cargo test --release -p metal2vulkan-runtime --lib copied_bench -- --ignored --nocapture`.

use crate::copied::{CopiedBuffers, Written};
use crate::dirty_pages::{hook, DirtySource, DirtyTally, PagesLog};
use crate::resource::ResourceKey;
use crate::work_counts::Counters;
use crate::{DirtyPages, DirtyStamp};
use std::num::NonZeroU64;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const PAGE: usize = 4096;
const BIG: usize = 64 << 20;
const SMALL: usize = 77 * PAGE;
const SMALLS: usize = 2048;
const RUNS: usize = 7;

/// One copy: the caller's pages and the copy, both the bench's own.
struct Pair {
    pages: Vec<u8>,
    copy: Vec<u8>,
}

/// A small deterministic generator (xorshift).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// A log of every write it is told of, by host address, in stamp order: a stamp is published
/// once asked.
#[derive(Default)]
struct BenchLog {
    next: AtomicU64,
    writes: Mutex<Vec<(u64, Range<u64>, u64)>>,
}

impl BenchLog {
    fn told(&self, range: Range<u64>, writer: u64) {
        let stamp = self.next.fetch_add(1, Ordering::SeqCst) + 1;
        self.writes.lock().unwrap().push((stamp, range, writer));
    }

    /// The writes stamped later than `since`, but `except`'s.
    fn since(&self, since: DirtyStamp, except: Option<u64>, out: &mut Vec<Range<u64>>) {
        let writes = self.writes.lock().unwrap();
        let from = writes.partition_point(|w| w.0 <= since.0);
        let later = writes[from..].iter().filter(|w| Some(w.2) != except);
        out.extend(later.map(|w| w.1.clone()));
    }
}

impl DirtyPages for BenchLog {
    type Watch = ();

    fn watch(&self, _range: Range<u64>) {}

    fn request(&self) -> DirtyStamp {
        DirtyStamp(self.next.fetch_add(1, Ordering::SeqCst) + 1)
    }

    fn wait(&self, _stamp: DirtyStamp) -> bool {
        true
    }

    fn changed(&self, since: DirtyStamp, out: &mut Vec<Range<u64>>) {
        self.since(since, None, out);
    }

    fn wrote(&self, range: Range<u64>) {
        self.told(range, 0);
    }

    fn wrote_as(&self, range: Range<u64>, writer: NonZeroU64) {
        self.told(range, writer.get());
    }

    fn changed_except(&self, since: DirtyStamp, writer: NonZeroU64, out: &mut Vec<Range<u64>>) {
        self.since(since, Some(writer.get()), out);
    }
}

fn median(mut times: Vec<f64>) -> f64 {
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}

/// A change of one side of the pairs, telling `tell` the host addresses of the pages it wrote:
/// the pairs it changed.
type Change<'a> = &'a mut dyn FnMut(&mut [Pair], &mut Rng, &dyn Fn(Range<u64>)) -> Vec<usize>;

#[test]
#[ignore = "a release-mode benchmark"]
fn copied_bench() {
    for (logged, named) in [(false, false), (true, false), (true, true)] {
        bench(logged, named);
    }
}

/// The three cases over copies whose pages are `logged` or not, their merges back taking as
/// written only the copies the GPU changed where `named`, else every copy.
fn bench(logged: bool, named: bool) {
    let buffers = Arc::new(CopiedBuffers::default());
    let (counts, tally) = (Counters::default(), DirtyTally::default());
    let log = Arc::new(BenchLog::default());
    let source = DirtySource::new(hook(Arc::clone(&log)), 0);
    let mut pairs: Vec<Pair> = std::iter::once(BIG)
        .chain(std::iter::repeat_n(SMALL, SMALLS))
        .enumerate()
        .map(|(i, length)| {
            let pages: Vec<u8> = (0..length).map(|b| (b ^ i) as u8).collect();
            Pair {
                copy: vec![0; length],
                pages,
            }
        })
        .collect();
    let total: usize = pairs.iter().map(|pair| pair.pages.len()).sum();
    let mut kept = Vec::new();
    let keys: Vec<ResourceKey> = pairs.iter().map(|_| ResourceKey::new()).collect();
    for (pair, key) in pairs.iter_mut().zip(&keys) {
        let pages = NonNull::new(pair.pages.as_mut_ptr()).unwrap();
        let copy = NonNull::new(pair.copy.as_mut_ptr()).unwrap();
        let read = |from: NonNull<u8>, into: &mut [u8]| {
            // SAFETY: the pair's pages, `into.len()` bytes.
            unsafe { std::ptr::copy_nonoverlapping(from.as_ptr(), into.as_mut_ptr(), into.len()) };
            true
        };
        let length = pair.pages.len();
        // SAFETY: both are the pair's own, live while the bench runs.
        let copied = unsafe { buffers.add_reading(pages, copy, length, read, false) }.unwrap();
        if logged {
            let at = pages.as_ptr() as u64;
            let pages_log = PagesLog::new((at, length as u64), Some(&source), None).unwrap();
            copied.set_log(&pages_log);
        }
        copied.name(*key);
        kept.push(copied);
    }
    let written = |changed: &[usize]| match named {
        true => {
            let mut written = Written::default();
            for &pair in changed {
                written.wrote(keys[pair]);
            }
            written
        }
        false => Written::all(),
    };
    let tell = |range: Range<u64>| log.told(range, 0);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let total_pages = total / PAGE;
    let name = match (logged, named) {
        (false, _) => "no log",
        (true, false) => "logged",
        (true, true) => "logged, named",
    };
    let mut report = |case: &str, change_cpu: Change<'_>, change_gpu: Change<'_>| {
        let (mut ins, mut backs) = (Vec::new(), Vec::new());
        for _ in 0..RUNS {
            change_cpu(&mut pairs, &mut rng, &tell);
            let started = Instant::now();
            buffers.copy_in(&counts, &tally);
            ins.push(started.elapsed().as_secs_f64() * 1e3);
            let changed = change_gpu(&mut pairs, &mut rng, &|_| {});
            let written = written(&changed);
            let started = Instant::now();
            buffers.merge_back(&counts, &written);
            backs.push(started.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!(
            "COPIED-BENCH {name} {case}: {} copies, {} MiB: copy in {:.2} ms, merge back {:.2} ms (median of {RUNS})",
            pairs.len(),
            total >> 20,
            median(ins),
            median(backs),
        );
    };
    // Every byte agrees first, and the logged copies hold a stamp.
    buffers.copy_in(&counts, &tally);
    buffers.merge_back(&counts, &Written::all());
    buffers.copy_in(&counts, &tally);
    let mut nothing = |_: &mut [Pair], _: &mut Rng, _: &dyn Fn(Range<u64>)| Vec::new();
    let mut nothing_either = |_: &mut [Pair], _: &mut Rng, _: &dyn Fn(Range<u64>)| Vec::new();
    report("(a) nothing changed", &mut nothing, &mut nothing_either);
    // One byte in each of 1% of the pages, on each side.
    let percent = |side: bool| {
        move |pairs: &mut [Pair], rng: &mut Rng, tell: &dyn Fn(Range<u64>)| {
            let mut changed = Vec::new();
            for _ in 0..total_pages / 100 {
                let mut page = (rng.next() as usize) % total_pages;
                for (index, pair) in pairs.iter_mut().enumerate() {
                    let bytes = match side {
                        true => &mut pair.pages,
                        false => &mut pair.copy,
                    };
                    let pages = bytes.len() / PAGE;
                    if page < pages {
                        let at = page * PAGE + (rng.next() as usize) % PAGE;
                        bytes[at] = bytes[at].wrapping_add(1);
                        let start = bytes.as_ptr() as u64 + (page * PAGE) as u64;
                        tell(start..start + PAGE as u64);
                        changed.push(index);
                        break;
                    }
                    page -= pages;
                }
            }
            changed
        }
    };
    report(
        "(b) 1% of pages changed",
        &mut percent(true),
        &mut percent(false),
    );
    // The 64 MiB copy redrawn whole, its pages rewritten whole.
    let whole = |side: bool| {
        move |pairs: &mut [Pair], rng: &mut Rng, tell: &dyn Fn(Range<u64>)| {
            let value = rng.next() as u8;
            let bytes = match side {
                true => &mut pairs[0].pages,
                false => &mut pairs[0].copy,
            };
            for byte in bytes.iter_mut() {
                *byte = byte.wrapping_add(value | 1);
            }
            let start = bytes.as_ptr() as u64;
            tell(start..start + bytes.len() as u64);
            vec![0]
        }
    };
    report("(c) one copy redrawn", &mut whole(true), &mut whole(false));
    drop(kept);
}
