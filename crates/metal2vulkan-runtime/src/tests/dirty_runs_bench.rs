//! The CPU cost of reporting a command buffer's write in place to a guest RAM's log
//! (`DirtyCommit::report_host_writes`, `dirty_pages.rs`), on plain memory: a 3840x2400
//! `BGRA8Unorm` surface (36,864,000 bytes, 9,000 pages of 4 KiB) imported in place over 9,000
//! runs of the RAM, each one page, none next to the one before (page `i` at RAM page
//! `i * 7919 mod 2^18` of a 1 GiB RAM), written whole by one command (as a render target's use
//! names its whole buffer). The RAM's log is the bench's own and trivial: each call it takes is
//! one fence and one load, and it adds up the bytes it was told of; one keeps only the per-run
//! hooks ([`PerRun`], `DirtyPages::wrote`, which the batched hooks' defaults call once a run),
//! the other the batched ones ([`Batched`], `DirtyPages::wrote_runs`, a fence a call). Prints,
//! for each, the median and 90th percentile of `REPORTS` reports and the log's calls a report
//! (`DIRTY-RUNS-BENCH`). Ignored: run it in release,
//! `cargo test --release -p metal2vulkan-runtime --lib dirty_runs_bench -- --ignored --nocapture`.

use crate::dirty_pages::{hook, DirtyCommit, DirtyShadows, DirtySource, PagesLog};
use crate::resource::ResourceKey;
use crate::tracker::{Access, Use, UseRange};
use crate::{DirtyPages, DirtyStamp};
use ash::vk;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::atomic::{fence, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const PAGE: u64 = 4096;
const SURFACE: u64 = 3840 * 2400 * 4;
const RAM_PAGES: u64 = 1 << 18;
const RAM_BASE: u64 = 0x1_0000_0000;
const STRIDE: u64 = 7919;
const REPORTS: usize = 200;

/// A trivial log that keeps only the per-run hooks: one fence and one load a call.
#[derive(Default)]
struct PerRun {
    epoch: AtomicU64,
    calls: AtomicU64,
    bytes: AtomicU64,
}

impl PerRun {
    fn took(&self, bytes: u64) {
        fence(Ordering::SeqCst);
        let _ = self.epoch.load(Ordering::Acquire);
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
    }
}

impl DirtyPages for PerRun {
    type Watch = ();

    fn watch(&self, _range: Range<u64>) {}

    fn request(&self) -> DirtyStamp {
        DirtyStamp(self.epoch.load(Ordering::Acquire))
    }

    fn wait(&self, _stamp: DirtyStamp) -> bool {
        true
    }

    fn changed(&self, _since: DirtyStamp, _out: &mut Vec<Range<u64>>) {}

    fn wrote(&self, range: Range<u64>) {
        self.took(range.end - range.start);
    }

    fn wrote_as(&self, range: Range<u64>, _writer: NonZeroU64) {
        self.took(range.end - range.start);
    }
}

/// A trivial log that keeps the batched hooks: one fence and one load a call, however many runs
/// it carries.
#[derive(Default)]
struct Batched(PerRun);

impl DirtyPages for Batched {
    type Watch = ();

    fn watch(&self, _range: Range<u64>) {}

    fn request(&self) -> DirtyStamp {
        self.0.request()
    }

    fn wait(&self, _stamp: DirtyStamp) -> bool {
        true
    }

    fn changed(&self, _since: DirtyStamp, _out: &mut Vec<Range<u64>>) {}

    fn wrote(&self, range: Range<u64>) {
        self.0.wrote(range);
    }

    fn wrote_runs(&self, runs: &[Range<u64>]) {
        self.0
            .took(runs.iter().map(|run| run.end - run.start).sum());
    }

    fn wrote_runs_as(&self, runs: &[Range<u64>], _writer: NonZeroU64) {
        self.wrote_runs(runs);
    }
}

/// The surface's window: 9,000 runs of one page each, scattered over the RAM.
fn window() -> Vec<Range<u64>> {
    (0..SURFACE / PAGE)
        .map(|i| RAM_BASE + (i * STRIDE % RAM_PAGES) * PAGE)
        .map(|start| start..start + PAGE)
        .collect()
}

/// The median and 90th percentile of `times`.
fn spread(mut times: Vec<Duration>) -> (Duration, Duration) {
    times.sort_unstable();
    (times[times.len() / 2], times[times.len() * 9 / 10])
}

/// `REPORTS` reports of one command buffer's write of the whole surface to `source`'s log:
/// their median and 90th percentile.
fn reports<D: DirtyPages<Watch = ()> + 'static>(source: &Arc<D>) -> (Duration, Duration) {
    let ram = DirtySource::new(hook(Arc::clone(source)), 0);
    let window = window();
    let log = PagesLog::new((0x7000_0000, SURFACE), None, Some((&ram, &window[..], 0))).unwrap();
    let shadows = DirtyShadows::default();
    let key = ResourceKey::new();
    shadows.logged_in_place(key, &log);
    let mut commit = DirtyCommit::default();
    let write = Access::new(
        vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
    );
    let uses = [Use::writes(key, true, UseRange::Bytes(0..SURFACE), write)];
    commit.note(&uses, &shadows);
    // Warm: the first reports.
    for _ in 0..10 {
        commit.report_host_writes();
    }
    let times = (0..REPORTS)
        .map(|_| {
            let started = Instant::now();
            commit.report_host_writes();
            started.elapsed()
        })
        .collect();
    spread(times)
}

/// Print the reports' `median` and `p90` to the log `name`, whose counts `took` are.
fn print(name: &str, took: &PerRun, (median, p90): (Duration, Duration)) {
    let made = (REPORTS + 10) as u64;
    let calls = took.calls.load(Ordering::Relaxed) / made;
    assert_eq!(took.bytes.load(Ordering::Relaxed), made * SURFACE);
    println!(
        "DIRTY-RUNS-BENCH log={name} runs={} calls/report={calls} median={:.1}us p90={:.1}us",
        SURFACE / PAGE,
        median.as_secs_f64() * 1e6,
        p90.as_secs_f64() * 1e6,
    );
}

#[test]
#[ignore = "a release bench: cargo test --release -p metal2vulkan-runtime --lib dirty_runs_bench -- --ignored --nocapture"]
fn dirty_runs_bench() {
    let per_run = Arc::new(PerRun::default());
    let spread = reports(&per_run);
    print("per-run", &per_run, spread);
    let batched = Arc::new(Batched::default());
    let spread = reports(&batched);
    print("batched", &batched.0, spread);
}
