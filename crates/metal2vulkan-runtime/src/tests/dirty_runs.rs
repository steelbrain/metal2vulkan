//! The runtime's reports of its own writes into logged host pages (`dirty_pages.rs`), on plain
//! memory, over pages whose memory's own log and a guest RAM's log both keep them, the RAM's
//! window 24 scattered pages: each log hears of each write in one call that carries every run of
//! it in that log (`DirtyPages::wrote_runs`, `wrote_runs_as`), and a log that keeps only the
//! per-run hooks (`wrote`, `wrote_as`) hears the same runs, in the same order, with the same
//! writers, one call each (the batched hooks' defaults). The writes: a command buffer's two writes
//! in place as its report names them (`DirtyCommit::report_host_writes`), a write in place with a
//! shadow's merge inside it (`PagesLog::wrote_but`), a copied host buffer's merge back of three
//! blocks (`PagesLog::wrote_as`) and a CPU write (`PagesLog::wrote_at`).

use crate::dirty_pages::{hook, DirtyCommit, DirtyShadows, DirtySource, PagesLog};
use crate::resource::ResourceKey;
use crate::tracker::{Access, Use, UseRange};
use crate::{DirtyPages, DirtyStamp};
use ash::vk;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const PAGE: u64 = 4096;
/// The memory's pointer, the object's byte 0.
const POINTER: u64 = 0x7000_0000;
/// The object's bytes: 22 pages and some, from `OFFSET` into its window.
const LENGTH: u64 = 22 * PAGE + 0x456;
const OFFSET: u64 = 0x123;
/// The RAM's page of each page of the window, in order: descending neighbours and jumps.
const WINDOW: [u64; 24] = [
    9, 8, 7, 30, 2, 1, 50, 49, 12, 60, 59, 58, 20, 21, 40, 5, 33, 32, 61, 15, 14, 44, 3, 27,
];
const RAM_BASE: u64 = 0x10_0000;
/// The writers of a shadow's merge and of a copy's merge back.
const SHADOW: u64 = 77;
const COPY: u64 = 78;

/// A log that keeps the batched hooks: each call's runs and writer (0 the consumer's), and the
/// calls of the per-run hooks it took.
#[derive(Default)]
struct Batched {
    calls: Mutex<Calls>,
    per_run: AtomicU64,
}

impl DirtyPages for Batched {
    type Watch = ();

    fn watch(&self, _range: Range<u64>) {}

    fn request(&self) -> DirtyStamp {
        DirtyStamp(0)
    }

    fn wait(&self, _stamp: DirtyStamp) -> bool {
        true
    }

    fn changed(&self, _since: DirtyStamp, _out: &mut Vec<Range<u64>>) {}

    fn wrote(&self, _range: Range<u64>) {
        self.per_run.fetch_add(1, Ordering::SeqCst);
    }

    fn wrote_as(&self, _range: Range<u64>, _writer: NonZeroU64) {
        self.per_run.fetch_add(1, Ordering::SeqCst);
    }

    fn wrote_runs(&self, runs: &[Range<u64>]) {
        self.calls.lock().unwrap().push((runs.to_vec(), 0));
    }

    fn wrote_runs_as(&self, runs: &[Range<u64>], writer: NonZeroU64) {
        self.calls
            .lock()
            .unwrap()
            .push((runs.to_vec(), writer.get()));
    }
}

/// A log that keeps only the per-run hooks: each run it was told of, with its writer.
#[derive(Default)]
struct PerRun {
    runs: Mutex<Vec<(Range<u64>, u64)>>,
}

impl DirtyPages for PerRun {
    type Watch = ();

    fn watch(&self, _range: Range<u64>) {}

    fn request(&self) -> DirtyStamp {
        DirtyStamp(0)
    }

    fn wait(&self, _stamp: DirtyStamp) -> bool {
        true
    }

    fn changed(&self, _since: DirtyStamp, _out: &mut Vec<Range<u64>>) {}

    fn wrote(&self, range: Range<u64>) {
        self.runs.lock().unwrap().push((range, 0));
    }

    fn wrote_as(&self, range: Range<u64>, writer: NonZeroU64) {
        self.runs.lock().unwrap().push((range, writer.get()));
    }
}

/// What the runtime writes.
#[derive(Clone, Copy, Debug)]
enum Write {
    /// Two writes in place of a command buffer's, as its report names them.
    Report,
    /// A write in place of `0x800..0x9800`, a shadow's merge of `0x1000..0x5000` among it.
    ShadowInside,
    /// A copy's merge back of three blocks.
    MergeBack,
    /// A CPU write through the object's `contents`.
    Cpu,
}

const REPORTED: [Range<u64>; 2] = [0x100..0x9100, 0x1_0000..LENGTH];
const WRITTEN: Range<u64> = 0x800..0x9800;
const MERGED: Range<u64> = 0x1000..0x5000;
const BLOCKS: [Range<u64>; 3] = [0..0x1000, 0x3000..0x5000, 0x9000..LENGTH];
const CPU: Range<u64> = 0x2345..0x7777;

/// The RAM's window: one run a page.
fn window() -> Vec<Range<u64>> {
    (WINDOW.iter())
        .map(|page| RAM_BASE + page * PAGE)
        .map(|start| start..start + PAGE)
        .collect()
}

/// The RAM log's runs of the object's `bytes`, one a page of the window they meet, in order:
/// worked out from the window, apart from the runtime's own.
fn ram_runs(bytes: &Range<u64>) -> Vec<Range<u64>> {
    let (start, end) = (bytes.start + OFFSET, bytes.end + OFFSET);
    (start / PAGE..end.div_ceil(PAGE))
        .map(|page| {
            let (from, to) = (start.max(page * PAGE), end.min((page + 1) * PAGE));
            let host = RAM_BASE + WINDOW[page as usize] * PAGE;
            host + (from - page * PAGE)..host + (to - page * PAGE)
        })
        .collect()
}

/// The RAM log's runs of each of `bytes`, in order.
fn ram_runs_of(bytes: &[Range<u64>]) -> Vec<Range<u64>> {
    bytes.iter().flat_map(ram_runs).collect()
}

/// A log's calls: each one's runs and writer (0 the consumer's).
type Calls = Vec<(Vec<Range<u64>>, u64)>;

/// The calls each log takes for `write`, as the memory's own log and the RAM's.
fn calls(write: Write) -> [Calls; 2] {
    match write {
        Write::Report => [
            REPORTED
                .iter()
                .map(|bytes| (vec![bytes.clone()], 0))
                .collect(),
            REPORTED.iter().map(|bytes| (ram_runs(bytes), 0)).collect(),
        ],
        Write::ShadowInside => {
            let beside = [WRITTEN.start..MERGED.start, MERGED.end..WRITTEN.end];
            [
                vec![(beside.to_vec(), 0)],
                vec![(ram_runs_of(&beside), 0), (ram_runs(&MERGED), SHADOW)],
            ]
        }
        Write::MergeBack => [
            vec![(BLOCKS.to_vec(), COPY)],
            vec![(ram_runs_of(&BLOCKS), COPY)],
        ],
        Write::Cpu => [vec![(vec![CPU], 0)], vec![(ram_runs(&CPU), 0)]],
    }
}

/// The runtime makes `write` into an object over pages logged by `own` (its memory's) and `ram`
/// (its guest RAM's, through [`window`]).
fn make<L: DirtyPages<Watch = ()> + 'static>(write: Write, own: &Arc<L>, ram: &Arc<L>) {
    let own_source = DirtySource::new(hook(Arc::clone(own)), POINTER);
    let ram_source = DirtySource::new(hook(Arc::clone(ram)), 0);
    let window = window();
    let log = PagesLog::new(
        (POINTER, LENGTH),
        Some(&own_source),
        Some((&ram_source, &window[..], OFFSET)),
    )
    .unwrap();
    let writer = |writer: u64| NonZeroU64::new(writer).unwrap();
    match write {
        Write::Report => {
            let shadows = DirtyShadows::default();
            let key = ResourceKey::new();
            shadows.logged_in_place(key, &log);
            let access = Access::new(
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            );
            let uses = REPORTED.map(|bytes| Use::writes(key, true, UseRange::Bytes(bytes), access));
            let mut commit = DirtyCommit::default();
            commit.note(&uses, &shadows);
            commit.report_host_writes();
        }
        Write::ShadowInside => {
            let merged = POINTER + MERGED.start..POINTER + MERGED.end;
            let shadows = [(writer(SHADOW), &own_source, merged)];
            log.wrote_but(std::slice::from_ref(&WRITTEN), &shadows);
        }
        Write::MergeBack => log.wrote_as(&BLOCKS, writer(COPY)),
        Write::Cpu => log.wrote_at(POINTER + CPU.start..POINTER + CPU.end),
    }
}

const WRITES: [Write; 4] = [
    Write::Report,
    Write::ShadowInside,
    Write::MergeBack,
    Write::Cpu,
];

#[test]
fn each_log_hears_of_each_write_in_one_call_carrying_its_runs() {
    for write in WRITES {
        let (own, ram) = (Arc::new(Batched::default()), Arc::new(Batched::default()));
        make(write, &own, &ram);
        let [want_own, want_ram] = calls(write);
        let (own_calls, ram_calls) = (own.calls.lock().unwrap(), ram.calls.lock().unwrap());
        assert_eq!(*own_calls, want_own, "{write:?}: the memory's log");
        assert_eq!(*ram_calls, want_ram, "{write:?}: the RAM's log");
        assert_eq!(
            (
                own.per_run.load(Ordering::SeqCst),
                ram.per_run.load(Ordering::SeqCst)
            ),
            (0, 0),
            "{write:?}: no per-run call"
        );
    }
    // The report over the scattered window: one call per log for each range it names, though
    // the RAM's log holds each range in many runs.
    let [_, ram] = calls(Write::Report);
    assert_eq!(ram.len(), REPORTED.len());
    assert_eq!(
        ram[0].0.len(),
        10,
        "the first range's runs in the RAM's log"
    );
}

#[test]
fn a_log_keeping_only_the_per_run_hooks_hears_every_run() {
    for write in WRITES {
        let (own, ram) = (Arc::new(PerRun::default()), Arc::new(PerRun::default()));
        make(write, &own, &ram);
        let flat = |calls: Calls| -> Vec<(Range<u64>, u64)> {
            (calls.into_iter())
                .flat_map(|(runs, writer)| runs.into_iter().map(move |run| (run, writer)))
                .collect()
        };
        let [want_own, want_ram] = calls(write);
        assert_eq!(
            *own.runs.lock().unwrap(),
            flat(want_own),
            "{write:?}: the memory's log"
        );
        assert_eq!(
            *ram.runs.lock().unwrap(),
            flat(want_ram),
            "{write:?}: the RAM's log"
        );
    }
}
