//! A logged object's bytes that a log's changed runs meet (`PagesLog::bytes_of`), the runs of a
//! guest RAM window of scattered pieces (descending neighbours, jumps, one page twice, pieces of
//! one to three pages): whether the changed runs are fewer than the object's runs (each finds the
//! runs it meets by their first offsets) or not (every run walked), the bytes are those every
//! piece of the window shares with a changed run, as a walk of every pair finds them.

use crate::dirty_pages::{hook, DirtySource, PagesLog};
use crate::{DirtyPages, DirtyStamp};
use std::ops::Range;
use std::sync::Arc;

const PAGE: u64 = 4096;
/// The window's pieces, in order: each one's first RAM page and its length in pages.
const PIECES: [(u64, u64); 14] = [
    (9, 1),
    (8, 1),
    (30, 3),
    (2, 1),
    (50, 2),
    (12, 1),
    (60, 1),
    (59, 1),
    (12, 1),
    (20, 2),
    (40, 3),
    (5, 1),
    (33, 1),
    (64, 2),
];
/// The RAM's pages the changed runs fall in.
const RAM_PAGES: u64 = 70;

struct Silent;

impl DirtyPages for Silent {
    type Watch = ();

    fn watch(&self, _range: Range<u64>) {}

    fn request(&self) -> DirtyStamp {
        DirtyStamp(0)
    }

    fn wait(&self, _stamp: DirtyStamp) -> bool {
        true
    }

    fn changed(&self, _since: DirtyStamp, _out: &mut Vec<Range<u64>>) {}

    fn wrote(&self, _range: Range<u64>) {}
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

/// `runs` sorted by start, those that meet or touch joined.
fn joined(mut runs: Vec<Range<u64>>) -> Vec<Range<u64>> {
    runs.retain(|run| run.start < run.end);
    runs.sort_by_key(|run| run.start);
    let mut kept: Vec<Range<u64>> = Vec::new();
    for run in runs {
        match kept.last_mut() {
            Some(last) if run.start <= last.end => last.end = last.end.max(run.end),
            _ => kept.push(run),
        }
    }
    kept
}

/// `count` changed runs of the RAM, sorted and apart: parts of pages and runs of pages.
fn changed(rng: &mut Rng, count: usize) -> Vec<Range<u64>> {
    let runs = (0..count).map(|_| {
        let start = rng.next() % (RAM_PAGES * PAGE);
        let length = match rng.next() % 3 {
            0 => 1 + rng.next() % PAGE,
            1 => PAGE,
            _ => (1 + rng.next() % 4) * PAGE,
        };
        start..start + length
    });
    joined(runs.collect())
}

#[test]
fn the_bytes_changed_runs_meet_are_those_every_piece_shares_with_them() {
    let ram = DirtySource::new(hook(Arc::new(Silent)), 0);
    let window: Vec<Range<u64>> = (PIECES.iter())
        .map(|&(page, pages)| page * PAGE..(page + pages) * PAGE)
        .collect();
    let length: u64 = window.iter().map(|run| run.end - run.start).sum();
    let log = PagesLog::new((0x7000_0000, length), None, Some((&ram, &window[..], 0))).unwrap();
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    for count in [
        0,
        1,
        2,
        5,
        PIECES.len() - 1,
        PIECES.len(),
        PIECES.len() + 5,
        60,
    ] {
        for _ in 0..50 {
            let changed = changed(&mut rng, count);
            let mut found = Vec::new();
            log.bytes_of(0, &changed, |bytes| found.push(bytes));
            let mut every = Vec::new();
            let mut at = 0;
            for piece in &window {
                for run in &changed {
                    let (start, stop) = (run.start.max(piece.start), run.end.min(piece.end));
                    if start < stop {
                        every.push(at + (start - piece.start)..at + (stop - piece.start));
                    }
                }
                at += piece.end - piece.start;
            }
            assert_eq!(
                joined(found),
                joined(every),
                "{count} changed runs: {changed:?}"
            );
        }
    }
}
