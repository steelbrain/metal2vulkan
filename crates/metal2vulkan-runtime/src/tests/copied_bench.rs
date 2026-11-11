//! The CPU cost of keeping copied host buffers coherent (`copied.rs`), on plain memory: a
//! commit's copy in and a batch completion's merge back over 680 MiB of live copies, the shape a
//! consumer's 4K desktop held (one 64 MiB surface and 2,048 copies of 308 KiB), in three cases:
//! (a) nothing changed, (b) 1% of the pages changed on each side, (c) the 64 MiB copy redrawn
//! whole by the GPU and its pages rewritten by the CPU. Each runs three times: with no log of the
//! pages' writes, each merge back taking every copy as written; with one log over every copy's
//! pages (a guest RAM's, its offsets host addresses, [`BenchLog`]) told of each CPU write, as a
//! consumer's dirty log would be, likewise; and logged, each merge back taking as written only
//! the copies the GPU wrote, as their command buffers' uses would name them (`CopiesReached`). Each
//! prints its median over its runs (`COPIED-BENCH`). And one copy of 256 MiB
//! ([`copied_256_bench`]): its copy in and merge back with no change, with one byte changed in 1%
//! of its pages, with every byte changed, and with 1% of its blocks changed on both sides (each
//! such block's base read), its base in host memory (here) or in the device's
//! (`tests::copied_base_device`). Ignored: run them in release,
//! `cargo test --release -p metal2vulkan-runtime --lib copied_bench -- --ignored --nocapture`.

use super::copied::add_reading;
use crate::copied::{BaseBytes, CopiedBuffers, CopiesReached};
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

/// A commit whose command buffer uses the copies `used` names and may write each: each takes its
/// base at its first, in host memory.
fn copy_in(buffers: &CopiedBuffers, counts: &Counters, tally: &DirtyTally, used: &CopiesReached) {
    let copied_in = buffers.copy_in(counts, tally, (used, used), Vec::new, &mut |_| None);
    assert_eq!(copied_in, Ok(()));
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
        let copied = unsafe { add_reading(&buffers, pages, copy, length, read, false) }.unwrap();
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
            let mut written = CopiesReached::default();
            for &pair in changed {
                written.reached(keys[pair]);
            }
            written
        }
        false => CopiesReached::all(),
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
            copy_in(&buffers, &counts, &tally, &CopiesReached::all());
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
    copy_in(&buffers, &counts, &tally, &CopiesReached::all());
    buffers.merge_back(&counts, &CopiesReached::all());
    copy_in(&buffers, &counts, &tally, &CopiesReached::all());
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

/// Copies of stitched views, the shape a consumer's guest held once its views' budget ran out
/// (11-07, 5080: 512 copies of 1.25 GiB, each view thousands of 4 KiB runs of its guest RAM):
/// [`STITCHED`] copies of [`STITCHED_PAGES`] pages each, every page its own run of one guest RAM
/// log, no two adjacent. Each commit's copy in, with the RAM written at `writes` pages since the
/// last (none, a few, many: the log's answer grows with them, the copies' runs do not), for a
/// command buffer that uses 8 of the copies and for one whose commands may reach every copy,
/// prints its median over [`STITCHED_RUNS`] commits (`COPIED-STITCHED`). Ignored: run it in release,
/// `cargo test --release -p metal2vulkan-runtime --lib copied_stitched_bench -- --ignored --nocapture`.
#[test]
#[ignore = "a release-mode benchmark"]
fn copied_stitched_bench() {
    const STITCHED: usize = 512;
    const STITCHED_PAGES: usize = 640;
    const STITCHED_RUNS: usize = 15;
    // The guest RAM's host addresses: none is read, so none need be mapped.
    const RAM: u64 = 0x7000_0000_0000;
    let buffers = Arc::new(CopiedBuffers::default());
    let (counts, tally) = (Counters::default(), DirtyTally::default());
    let log = Arc::new(BenchLog::default());
    let source = DirtySource::new(hook(Arc::clone(&log)), RAM);
    let length = STITCHED_PAGES * PAGE;
    let mut pairs: Vec<Pair> = (0..STITCHED)
        .map(|_| Pair {
            pages: vec![0; length],
            copy: vec![0; length],
        })
        .collect();
    // Copy `c`'s page `k` is RAM page `2 * (c * STITCHED_PAGES + k)`: its offset in the log.
    let ram_page = |c: usize, k: usize| (2 * (c * STITCHED_PAGES + k) * PAGE) as u64;
    let keys: Vec<ResourceKey> = (0..STITCHED).map(|_| ResourceKey::new()).collect();
    let mut kept = Vec::new();
    for (c, pair) in pairs.iter_mut().enumerate() {
        let pages = NonNull::new(pair.pages.as_mut_ptr()).unwrap();
        let copy = NonNull::new(pair.copy.as_mut_ptr()).unwrap();
        let read = |from: NonNull<u8>, into: &mut [u8]| {
            // SAFETY: the pair's pages, `into.len()` bytes.
            unsafe { std::ptr::copy_nonoverlapping(from.as_ptr(), into.as_mut_ptr(), into.len()) };
            true
        };
        // SAFETY: both are the pair's own, live while the bench runs.
        let copied = unsafe { add_reading(&buffers, pages, copy, length, read, false) }.unwrap();
        let window: Vec<Range<u64>> = (0..STITCHED_PAGES)
            .map(|k| RAM + ram_page(c, k)..RAM + ram_page(c, k) + PAGE as u64)
            .collect();
        let at = pages.as_ptr() as u64;
        let pages_log =
            PagesLog::new((at, length as u64), None, Some((&source, &window, 0))).unwrap();
        copied.set_log(&pages_log);
        copied.name(keys[c]);
        kept.push(copied);
    }
    // Every byte agrees first, and each copy holds a stamp.
    let every = CopiesReached::all();
    copy_in(&buffers, &counts, &tally, &every);
    copy_in(&buffers, &counts, &tally, &every);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    // A command buffer that uses 8 of the copies, and one whose commands may reach any.
    let mut eight = CopiesReached::default();
    for key in keys.iter().step_by(STITCHED / 8) {
        eight.reached(*key);
    }
    for ((writes, used), uses) in [0usize, 16, 256, 4096]
        .into_iter()
        .flat_map(|writes| [(writes, &eight), (writes, &every)])
        .zip(["8 copies used", "every copy reached"].into_iter().cycle())
    {
        let mut ins = Vec::new();
        for _ in 0..STITCHED_RUNS {
            for _ in 0..writes {
                let (c, k) = (
                    (rng.next() as usize) % STITCHED,
                    (rng.next() as usize) % STITCHED_PAGES,
                );
                log.told(ram_page(c, k)..ram_page(c, k) + PAGE as u64, 0);
            }
            let started = Instant::now();
            copy_in(&buffers, &counts, &tally, used);
            ins.push(started.elapsed().as_secs_f64() * 1e3);
        }
        eprintln!(
            "COPIED-STITCHED {STITCHED} copies of {STITCHED_PAGES} runs, {} MiB, {writes} pages written, {uses}: copy in {:.3} ms (median of {STITCHED_RUNS})",
            (STITCHED * length) >> 20,
            median(ins),
        );
    }
    drop(kept);
}

/// One copy of 256 MiB, its base in host memory ([`bench_256`]).
#[test]
#[ignore = "a release-mode benchmark"]
fn copied_256_bench() {
    bench_256("host base", &mut |_| None);
}

/// The copy in and merge back of one copy of 256 MiB, its base's bytes where `base` gives them
/// (`None`: host memory), printed as the median of [`RUNS`] (`COPIED-256`): (a) nothing changed;
/// (b) one byte in each of 1% of its 4 KiB pages, the CPU's before the copy in and the GPU's
/// before the merge back; (c) every byte, each side; (d) one byte on each side in each of 1% of
/// its pages, the same block, both before the merge back, which reads those blocks' base; (e)
/// every float of the copy rewritten by the GPU as `copied_pass_bench`'s kernel rewrites it
/// (`x + i`: some bytes of each float change), the pages unchanged; (f) as (e), and the CPU's
/// write of a byte of every 1 KiB block of the pages before the merge back, which reads every
/// block's base. On an x86-64 host with AVX-512BW, (e) and (f) again without its masked stores.
pub(super) fn bench_256(label: &str, base: &mut dyn FnMut(usize) -> Option<BaseBytes>) {
    const LENGTH: usize = 256 << 20;
    let buffers = Arc::new(CopiedBuffers::default());
    let (counts, tally) = (Counters::default(), DirtyTally::default());
    let mut pair = Pair {
        pages: (0..LENGTH).map(|b| (b ^ (b >> 11)) as u8).collect(),
        copy: vec![0; LENGTH],
    };
    let pages = NonNull::new(pair.pages.as_mut_ptr()).unwrap();
    let copy = NonNull::new(pair.copy.as_mut_ptr()).unwrap();
    let read = |from: NonNull<u8>, into: &mut [u8]| {
        // SAFETY: the pair's pages, `into.len()` bytes.
        unsafe { std::ptr::copy_nonoverlapping(from.as_ptr(), into.as_mut_ptr(), into.len()) };
        true
    };
    // SAFETY: both are the pair's own, live while the bench runs.
    let copied = unsafe { add_reading(&buffers, pages, copy, LENGTH, read, false) }.unwrap();
    let all = CopiesReached::all();
    let started = Instant::now();
    let copied_in = buffers.copy_in(&counts, &tally, (&all, &all), Vec::new, &mut *base);
    assert_eq!(copied_in, Ok(()));
    let (host, device) = buffers.base_bytes();
    eprintln!(
        "COPIED-256 {label}: base made in {:.1} ms, {} KiB host, {} MiB device",
        started.elapsed().as_secs_f64() * 1e3,
        host >> 10,
        device >> 20,
    );
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let pages_count = LENGTH / PAGE;
    // One byte in each of 1% of the pages of `side`, at `offset` into the page (random where
    // `None`).
    let percent = |side: &mut [u8], rng: &mut Rng, offset: Option<usize>| {
        for _ in 0..pages_count / 100 {
            let page = (rng.next() as usize) % pages_count;
            let at = page * PAGE + offset.unwrap_or((rng.next() as usize) % PAGE);
            side[at] = side[at].wrapping_add(1);
        }
    };
    let mut report = |case: &str, pair: &mut Pair, change: &mut dyn FnMut(&mut Pair, bool)| {
        let (mut ins, mut backs) = (Vec::new(), Vec::new());
        let before = counts.copied_tallies();
        let read = before.base_blocks_read;
        for _ in 0..RUNS {
            change(pair, true);
            let started = Instant::now();
            let copied_in = buffers.copy_in(&counts, &tally, (&all, &all), Vec::new, &mut *base);
            assert_eq!(copied_in, Ok(()));
            ins.push(started.elapsed().as_secs_f64() * 1e3);
            change(pair, false);
            let started = Instant::now();
            buffers.merge_back(&counts, &all);
            backs.push(started.elapsed().as_secs_f64() * 1e3);
        }
        let after = counts.copied_tallies();
        let read = after.base_blocks_read - read;
        let copied_in = (after.copy_in_bytes - before.copy_in_bytes) / RUNS as u64;
        let merged = (after.merge_back_bytes - before.merge_back_bytes) / RUNS as u64;
        let first = backs[0];
        let (ins, backs) = (median(ins), median(backs));
        let rate = |bytes: u64, ms: f64| bytes as f64 / 1e6 / ms;
        eprintln!(
            "COPIED-256 {label} {case}: copy in {ins:.2} ms ({} MB, {:.2} GB/s), merge back {backs:.2} ms ({} MB, {:.2} GB/s) (median of {RUNS}; the first {first:.2} ms), {} base blocks read, {} threads started",
            copied_in / 1_000_000,
            rate(copied_in, ins),
            merged / 1_000_000,
            rate(merged, backs),
            read / RUNS as u64,
            (after.compare_threads - before.compare_threads) / RUNS as u64,
        );
    };
    report("(a) nothing changed", &mut pair, &mut |_, _| {});
    report("(b) 1% of pages changed", &mut pair, &mut |pair, cpu| {
        let side = match cpu {
            true => &mut pair.pages,
            false => &mut pair.copy,
        };
        percent(side, &mut rng, None);
    });
    report("(c) every byte changed", &mut pair, &mut |pair, cpu| {
        let side = match cpu {
            true => &mut pair.pages,
            false => &mut pair.copy,
        };
        for byte in side.iter_mut() {
            *byte = byte.wrapping_add(3);
        }
    });
    report(
        "(d) 1% of pages changed on both sides",
        &mut pair,
        &mut |pair, cpu| {
            // Both sides change the same pages before the merge back: byte 8 of each by the CPU,
            // byte 0 by the GPU, in the same block.
            if !cpu {
                let mut both = Rng(rng.next());
                let mut cpu_side = Rng(both.0);
                percent(&mut pair.pages, &mut cpu_side, Some(8));
                percent(&mut pair.copy, &mut both, Some(0));
            }
        },
    );
    // As `copied_pass_bench`'s kernel writes it: `x + i` for float `i`, which changes some bytes
    // of each float and leaves others.
    let rewrite = |copy: &mut [u8]| {
        let floats = copy.as_chunks_mut::<4>().0.iter_mut();
        for (index, float) in floats.enumerate() {
            *float = (f32::from_le_bytes(*float) + index as f32).to_le_bytes();
        }
    };
    let mut gpu_rewrites = |pair: &mut Pair, cpu: bool| {
        if !cpu {
            rewrite(&mut pair.copy);
        }
    };
    let mut both_write = |pair: &mut Pair, cpu: bool| {
        if !cpu {
            for block in pair.pages.as_chunks_mut::<1024>().0 {
                block[1] = block[1].wrapping_add(1);
            }
            rewrite(&mut pair.copy);
        }
    };
    let (rewrites, both) = (
        "(e) every float rewritten by the GPU",
        "(f) every float rewritten by the GPU, a byte of every block by the CPU",
    );
    report(rewrites, &mut pair, &mut gpu_rewrites);
    report(both, &mut pair, &mut both_write);
    // Where the host stores a masked 16 bytes in one store, (e) and (f) again without.
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx512bw") && std::is_x86_feature_detected!("avx512vl") {
        use crate::copied::x86::MASKED_STORES_OFF;
        MASKED_STORES_OFF.store(true, std::sync::atomic::Ordering::Relaxed);
        let without = |case: &str| format!("{case}, without masked stores");
        report(&without(rewrites), &mut pair, &mut gpu_rewrites);
        report(&without(both), &mut pair, &mut both_write);
        MASKED_STORES_OFF.store(false, std::sync::atomic::Ordering::Relaxed);
    }
    drop(copied);
}

/// The rate of a copy's block tags (`copied::tag`, SSE2 on an x86-64 host) and of their portable
/// form, against comparing two equal streams 4 KiB at a time, over 256 MiB and over 64 KiB that
/// stays in the first-level cache, printed as GB/s, the best of [`RUNS`] (`COPIED-TAG`).
#[test]
#[ignore = "a release-mode benchmark"]
fn copied_tag_bench() {
    use crate::copied::{tag, tag_portable, TAG_BLOCK};
    for length in [64 << 10, 256 << 20] {
        let one: Vec<u8> = (0..length).map(|b: usize| (b ^ (b >> 11)) as u8).collect();
        let two = one.clone();
        let rate = |name: &str, pass: &dyn Fn() -> u64| {
            let rounds = (256 << 20) / length;
            let mut best = f64::MAX;
            let mut sink = 0u64;
            for _ in 0..RUNS {
                let started = Instant::now();
                for _ in 0..rounds {
                    sink = sink.wrapping_add(std::hint::black_box(pass()));
                }
                best = best.min(started.elapsed().as_secs_f64());
            }
            let gb = (rounds * length) as f64 / 1e9;
            eprintln!(
                "COPIED-TAG {name} over {} KiB: {:.2} GB/s ({sink:x})",
                length >> 10,
                gb / best
            );
        };
        let blocks = || one.as_chunks::<TAG_BLOCK>().0.iter();
        rate("portable tag", &|| {
            blocks()
                .map(|b| tag_portable(b)[0])
                .fold(0, u64::wrapping_add)
        });
        rate("tag", &|| {
            blocks().map(|b| tag(b)[0]).fold(0, u64::wrapping_add)
        });
        rate("two-stream compare", &|| {
            let pairs = (one.as_chunks::<4096>().0.iter()).zip(two.as_chunks::<4096>().0);
            pairs
                .filter(|(a, b)| std::hint::black_box(a) != std::hint::black_box(b))
                .count() as u64
        });
    }
}
