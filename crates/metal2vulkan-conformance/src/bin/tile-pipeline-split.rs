//! Where a **warm** `tile_pipeline` call's cost goes: the floor under every pipeline creation,
//! measured on the one family where the call asks the driver for nothing at all.
//!
//! `known-slow-calls.tsv`'s `tile_pipeline` row is not a cost target — per case we are 31x faster
//! than Metal on that family — and this probe does not propose a fix for it. It is a **control**.
//! The row's three `tile-state-thread-limits` calls cost a flat ~200 us each with
//! `per_state pipelines` 0, `functions_shared` 3 and `shader_modules_made` 1 over the three, so
//! nothing is translated, no `VkPipeline` is made and the driver is never asked: whatever those
//! ~200 us are, they are ours and they are the floor. The `render_pipeline` row's warm creation
//! costs 240-285 us and **contains** MoltenVK's `vkCreateGraphicsPipelines` on the calling thread
//! (`RenderVariant::make_at_creation` hands its one piece to `join_all(.., first_here = true)`,
//! which `pool.rs` runs there), so that figure cannot be split by a clock. This one can.
//!
//! # What the splits are, and why each is measured the way it is
//!
//! The row leaves four candidates UNAUDITED: key derivation, the artifact decode with
//! `cache/scan.rs`'s linear module scan, `vkCreateShaderModule`, and the cache's disk read. Two
//! more sit on the path and the row does not name them, read out of `pipeline/mod.rs`'s
//! `of_kernel`: the artifact **encode** and store that a creation whose key is new pays (its
//! function's: `Sources::kernel` links the pipeline from the function and stores none), and the
//! bind plan plus the neutral resources a creation makes before it returns (`base.plan(..)`,
//! `make_neutrals`, `keep_room`).
//!
//! Counters settle what *happens*; only a clock settles how much a step *costs*. So:
//!
//! - **`case`** runs the case's own calls through the suite's side with the counters and both
//!   clocks read around each one, as `render-pipeline-split` does. It says what a call does.
//! - **`loop`** makes tile states directly, thousands of them, and reports the **median** of the
//!   per-call costs after discarding a warm-up prefix. A process's first pipeline creation has
//!   cost 1.0-13.7 ms and moved an order of magnitude between runs, so no figure here is ever a
//!   first call: the median of a long run cannot be a warm-up and the probe prints the first
//!   call separately so a reader can see the two apart.
//! - **`parts`** calls the translator's own public steps in their own loops, over the same
//!   descriptor the case makes, and reports each step's median. `tile_pipeline_key`,
//!   `ComputePipeline::to_bytes` and `ComputePipeline::from_bytes_for_key` (which runs
//!   `scan_words` over the module, `cache/artifacts.rs`) are all public, so the three candidates
//!   that are translator steps are measured **as themselves** rather than by subtracting totals.
//!   A marginal cost and never a total divided by a count.
//!
//! `loop` takes `--fresh-key` or `--same-key`. Fresh is the case's shape: every creation carries a
//! `max_total_threads_per_threadgroup` no earlier one used, so its pipeline key is new and the
//! memo cannot answer it. Same repeats one descriptor, which the memo answers from the second
//! call on. The difference between them is the whole of what a new key costs — the lookup, the
//! disk read, the decode and scan or the compile, and the store — and what is left in the
//! `--same-key` figure is the floor that no cache of ours can remove.
//!
//! # Running it
//!
//! ```text
//! tile-pipeline-split case  <case> <cache-directory> [--no-memo]
//! tile-pipeline-split loop  <iterations> <cache-directory> [--no-memo] [--same-key]
//! tile-pipeline-split parts <iterations> <cache-directory>
//! ```
//!
//! A reading, not a gate: it prints and exits 0 unless the run itself failed.

use metal2vulkan::cache::PipelineArtifact;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::{
    ComputePipeline, FunctionDescriptor, LoweringInputs, StageSource, TileRenderPipelineDescriptor,
};
use metal2vulkan::Budget;
use metal2vulkan_conformance::device::Notices;
use metal2vulkan_conformance::fixture::Fixture;
use metal2vulkan_conformance::runtime_side::RuntimeSide;
use metal2vulkan_conformance::transcript::Sites;
use metal2vulkan_conformance::watch::{Watcher, Watching};
use metal2vulkan_runtime::{
    Degradation, Device, DeviceKind, DeviceOptions, PipelineOptions, ShaderCacheOptions,
};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

/// The call `case` mode reads. Every other call is forwarded untimed and uncounted.
const CALL: &str = "tile_pipeline";

/// The fixture `tile-state-thread-limits` runs, and its one color attachment format, as
/// `cases/tile_threadgroups.rs` names them. Kept here rather than reached for, because the case's
/// own constants are private to it; a drift between the two would show as a different module
/// count in `case` mode beside `loop`'s.
const TILE_FIXTURE: &str = "k10ts_tile_order";

/// The case's one color attachment format.
const TILE_ATTACHMENT: PixelFormat = PixelFormat::R32Float;

/// The counters as they stand, of the device and its shader cache.
#[derive(Clone, Copy, Debug, Default)]
struct Readings {
    keys: u64,
    entries: u64,
    entries_shared: u64,
    translated: u64,
    functions_shared: u64,
    modules_made: u64,
    modules_shared: u64,
    layouts_made: u64,
    layouts_shared: u64,
    pipelines_made: u64,
    cache_hits: u64,
    cache_misses: u64,
    cache_writes: u64,
    writes_on_thread: u64,
    queued_hits: u64,
    blocks_made: u64,
    backings_made: u64,
    addressable_buffers: u64,
    samplers_made: u64,
}

/// What one call did, as the counters and the clocks read it.
#[derive(Clone, Copy, Debug, Default)]
struct Row {
    wall: u64,
    on_cpu: u64,
    counts: Readings,
}

fn readings(device: &Device) -> Readings {
    let counts = device.work_counts();
    let cache = device.shader_cache_statistics();
    Readings {
        keys: counts.descriptor_keys_built,
        entries: counts.stage_entries_specialized,
        entries_shared: counts.stage_entries_shared,
        translated: counts.functions_translated,
        functions_shared: counts.functions_shared,
        modules_made: counts.shader_modules_made,
        modules_shared: counts.shader_modules_shared,
        layouts_made: counts.pipeline_layouts_made,
        layouts_shared: counts.pipeline_layouts_shared,
        pipelines_made: counts.variant_pipelines_made,
        cache_hits: cache.hits,
        cache_misses: cache.misses,
        cache_writes: cache.writes,
        writes_on_thread: cache.writes_on_thread,
        queued_hits: cache.queued_hits,
        blocks_made: counts.pooled_memory_blocks_made,
        backings_made: counts.pooled_backings_made,
        addressable_buffers: counts.addressable_buffers_made,
        samplers_made: counts.sampler_sets_made,
    }
}

impl Readings {
    /// Every field of `self` and `other` under `each`.
    fn zip(self, other: Readings, each: impl Fn(u64, u64) -> u64) -> Readings {
        Readings {
            keys: each(self.keys, other.keys),
            entries: each(self.entries, other.entries),
            entries_shared: each(self.entries_shared, other.entries_shared),
            translated: each(self.translated, other.translated),
            functions_shared: each(self.functions_shared, other.functions_shared),
            modules_made: each(self.modules_made, other.modules_made),
            modules_shared: each(self.modules_shared, other.modules_shared),
            layouts_made: each(self.layouts_made, other.layouts_made),
            layouts_shared: each(self.layouts_shared, other.layouts_shared),
            pipelines_made: each(self.pipelines_made, other.pipelines_made),
            cache_hits: each(self.cache_hits, other.cache_hits),
            cache_misses: each(self.cache_misses, other.cache_misses),
            cache_writes: each(self.cache_writes, other.cache_writes),
            writes_on_thread: each(self.writes_on_thread, other.writes_on_thread),
            queued_hits: each(self.queued_hits, other.queued_hits),
            blocks_made: each(self.blocks_made, other.blocks_made),
            backings_made: each(self.backings_made, other.backings_made),
            addressable_buffers: each(self.addressable_buffers, other.addressable_buffers),
            samplers_made: each(self.samplers_made, other.samplers_made),
        }
    }

    fn since(self, before: Readings) -> Readings {
        self.zip(before, |after, before| after.saturating_sub(before))
    }

    fn and(self, other: Readings) -> Readings {
        self.zip(other, |one, two| one + two)
    }

    fn fields(self) -> [u64; 19] {
        [
            self.keys,
            self.entries,
            self.entries_shared,
            self.translated,
            self.functions_shared,
            self.modules_made,
            self.modules_shared,
            self.layouts_made,
            self.layouts_shared,
            self.pipelines_made,
            self.cache_hits,
            self.cache_misses,
            self.cache_writes,
            self.writes_on_thread,
            self.queued_hits,
            self.blocks_made,
            self.backings_made,
            self.addressable_buffers,
            self.samplers_made,
        ]
    }
}

/// The column heads of [`Readings::fields`], in its order.
const HEADS: [&str; 19] = [
    "keys", "entr", "entS", "xlat", "fshr", "modM", "modS", "layM", "layS", "pipM", "chit", "cmis",
    "cwrt", "wthr", "qhit", "blkM", "bakM", "abuf", "smpM",
];

fn header(first: &str) {
    println!(
        "{first:>34}  {:>10} {:>10} {}",
        "wall us",
        "cpu us",
        HEADS
            .iter()
            .map(|word| format!("{word:>5}"))
            .collect::<String>()
    );
}

fn line(at: &str, row: &Row) {
    println!(
        "{at:>34}  {:>10.1} {:>10.1} {}",
        row.wall as f64 / 1000.0,
        row.on_cpu as f64 / 1000.0,
        row.counts
            .fields()
            .iter()
            .map(|count| format!("{count:>5}"))
            .collect::<String>()
    );
}

/// The median of `all`, in microseconds, and `all` is left sorted.
fn median_us(all: &mut [u64]) -> f64 {
    if all.is_empty() {
        return 0.0;
    }
    all.sort_unstable();
    all[all.len() / 2] as f64 / 1000.0
}

/// The `at`th percentile of a sorted `all`, in microseconds.
fn percentile_us(sorted: &[u64], at: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * at).round() as usize;
    sorted[index] as f64 / 1000.0
}

/// The device and the log a span writes into.
struct Shared {
    device: Device,
    log: RefCell<Vec<Row>>,
}

/// The watcher: it reads the counters and the clocks around [`CALL`] and forwards the rest.
#[derive(Clone)]
struct Split(Rc<Shared>);

impl Watcher for Split {
    type Span = Span;

    fn enter(&self, call: &'static str) -> Span {
        if call != CALL {
            return Span {
                shared: None,
                before: Readings::default(),
                on_cpu: None,
                started: Instant::now(),
            };
        }
        // The counters first and the wall clock last, so neither counter read falls inside the
        // call's own window; the reverse at the other end, for the same reason.
        let before = readings(&self.0.device);
        let on_cpu = metal2vulkan_conformance::clocks::thread_cpu_nanos();
        Span {
            shared: Some(Rc::clone(&self.0)),
            before,
            on_cpu,
            started: Instant::now(),
        }
    }
}

/// One call's reading, open until the call returns.
struct Span {
    shared: Option<Rc<Shared>>,
    before: Readings,
    on_cpu: Option<u64>,
    started: Instant,
}

impl Drop for Span {
    fn drop(&mut self) {
        let Some(shared) = self.shared.take() else {
            return;
        };
        let wall = u64::try_from(self.started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let on_cpu = match (
            self.on_cpu,
            metal2vulkan_conformance::clocks::thread_cpu_nanos(),
        ) {
            (Some(before), Some(after)) => after.saturating_sub(before),
            _ => 0,
        };
        let after = readings(&shared.device);
        shared.log.borrow_mut().push(Row {
            wall,
            on_cpu,
            counts: after.since(self.before),
        });
    }
}

/// The suite's device options, with the shader cache in `directory`. `memo` false bounds the
/// device's memo to one byte, which keeps nothing; zero is refused at device creation
/// (`Problem::ShaderCacheCapacity`).
fn options(notices: &Notices, directory: &str, memo: bool) -> DeviceOptions {
    let mut options = DeviceOptions::default();
    options.policy = options.policy.exact_answers();
    options.announce_each = Degradation::ALL.iter().copied().collect();
    options.shader_cache = ShaderCacheOptions::in_directory(std::path::Path::new(directory));
    if !memo {
        options.shader_cache.memory_capacity = 1;
    }
    options.messages = Some(notices.sink());
    options.kinds = vec![
        DeviceKind::Discrete,
        DeviceKind::Integrated,
        DeviceKind::Virtual,
    ];
    options
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(why) => {
            eprintln!("tile-pipeline-split: {why}");
            std::process::ExitCode::FAILURE
        }
    }
}

const USAGE: &str = "usage: tile-pipeline-split case <case> <dir> [--no-memo]\n       \
                     tile-pipeline-split loop <iterations> <dir> [--no-memo] [--same-key]\n       \
                     tile-pipeline-split parts <iterations> <dir>\n       \
                     tile-pipeline-split set   <rounds> <dir> [name-prefix]";

fn run() -> Result<(), String> {
    let words: Vec<String> = std::env::args().skip(1).collect();
    let mode = words.first().ok_or(USAGE)?.as_str();
    let first = words.get(1).ok_or(USAGE)?.clone();
    let directory = words.get(2).ok_or(USAGE)?.clone();
    let flag = |name: &str| words.iter().any(|word| word == name);
    match mode {
        "case" => case_mode(&first, &directory, !flag("--no-memo")),
        "loop" => {
            let times: usize = first.parse().map_err(|_| USAGE.to_owned())?;
            loop_mode(times, &directory, !flag("--no-memo"), !flag("--same-key"))
        }
        "parts" => {
            let times: usize = first.parse().map_err(|_| USAGE.to_owned())?;
            parts_mode(times, &directory)
        }
        "set" => {
            let rounds: usize = first.parse().map_err(|_| USAGE.to_owned())?;
            let select = words.get(3).cloned().unwrap_or_default();
            set_mode(rounds.max(1), &select, &directory)
        }
        _ => Err(USAGE.to_owned()),
    }
}

/// The case's own calls, with the counters and the clocks around each.
fn case_mode(name: &str, directory: &str, memo: bool) -> Result<(), String> {
    let case = metal2vulkan_conformance::cases::named(name).ok_or(format!("no case {name}"))?;
    let mut notices = Notices::default();
    let device = Device::system_default_with(&options(&notices, directory, memo))
        .map_err(|error| format!("no device: {error}"))?;
    notices.venue();
    let shared = Rc::new(Shared {
        device: device.clone(),
        log: RefCell::new(Vec::new()),
    });
    let mut sites = Sites::new();
    let side = RuntimeSide::new(&device, notices.open(case.name()));
    let mut watching = Watching::with(side, Split(Rc::clone(&shared)));
    let outcome = case.run(&mut watching, &mut sites);
    drop(watching);
    let rows = shared.log.borrow().clone();
    println!(
        "case {name} venue {} dir {directory} memo {}",
        device.name(),
        if memo { "on" } else { "off" }
    );
    header("call");
    for (at, row) in rows.iter().enumerate() {
        line(&at.to_string(), row);
    }
    let mut total = Row::default();
    for row in &rows {
        total.wall += row.wall;
        total.on_cpu += row.on_cpu;
        total.counts = total.counts.and(row.counts);
    }
    line("SUM", &total);
    if let Err(abort) = outcome {
        return Err(format!(
            "the case aborted at {}: {:?}",
            abort.at, abort.failure
        ));
    }
    Ok(())
}

/// The bytes the shader cache stores for the tile function compiled under `inputs`: what
/// `Sources::function` reads back and decodes when the directory answers its lookup.
fn compiled_function_bytes(
    compile: &metal2vulkan::device::Device,
    function: &FunctionDescriptor,
    inputs: &LoweringInputs,
    budget: &Budget,
) -> Result<Vec<u8>, String> {
    compile
        .compile_function(function, inputs, budget)
        .map(|compiled| compiled.to_bytes())
        .map_err(|error| format!("the function compile refused: {error}"))
}

/// The suite's own measurement shape, with the counters the suite's records do not carry.
///
/// `times_on_under` runs the whole selection in the case set's order, **a fresh device and a fresh
/// shader-cache directory each round**, the cache carried over between the cases of a round, and
/// takes a figure as the median of the rounds' own medians. The recorded 203.3 us for
/// `tile-state-thread-limits` is that figure. Nothing in the record says who wrote the pipeline
/// artifacts, because `writes_on_thread` is a `ShaderCacheStatistics` field and a cost record
/// carries `WorkCounts`; so this runs the same shape and prints it.
///
/// `select` is a prefix: the empty string selects the whole set, which is the recorded shape, and
/// `tile-` selects this family alone. The contrast between the two says whether the cost is the
/// call's own or is made by what ran before it.
fn set_mode(rounds: usize, select: &str, directory: &str) -> Result<(), String> {
    let chosen: Vec<Box<dyn metal2vulkan_conformance::case::Case>> =
        metal2vulkan_conformance::cases::all()
            .into_iter()
            .filter(|case| case.name().starts_with(select))
            .collect();
    if chosen.is_empty() {
        return Err(format!("no case's name starts with {select:?}"));
    }
    println!(
        "set rounds {rounds} select {select:?} cases {} dir {directory}",
        chosen.len()
    );
    header("round");
    let mut medians: Vec<f64> = Vec::new();
    for round in 0..rounds {
        // A device and a cache directory of this round's own, as the suite's rounds take.
        let per_round = format!("{directory}/round-{round}");
        let mut notices = Notices::default();
        let device = Device::system_default_with(&options(&notices, &per_round, true))
            .map_err(|error| format!("no device: {error}"))?;
        notices.venue();
        let shared = Rc::new(Shared {
            device: device.clone(),
            log: RefCell::new(Vec::new()),
        });
        // Where each case's calls start in the log, so a case's own calls can be read out of a
        // round: the watcher sees calls and not cases.
        let mut starts: Vec<(usize, &str)> = Vec::new();
        for case in &chosen {
            starts.push((shared.log.borrow().len(), case.name()));
            let mut sites = Sites::new();
            let side = RuntimeSide::new(&device, notices.open(case.name()));
            let mut watching = Watching::with(side, Split(Rc::clone(&shared)));
            // A case that aborts is still a round's worth of calls, and this reads calls rather
            // than verdicts, so an abort is printed and the round goes on.
            if let Err(abort) = case.run(&mut watching, &mut sites) {
                eprintln!("{} aborted at {}", case.name(), abort.at);
            }
            drop(watching);
        }
        let rows = shared.log.borrow().clone();
        for (at, (from, name)) in starts.iter().enumerate() {
            let to = starts.get(at + 1).map_or(rows.len(), |(next, _)| *next);
            for (which, row) in rows[*from..to].iter().enumerate() {
                line(&format!("{name}#{which}"), row);
            }
        }
        let mut total = Row::default();
        for row in &rows {
            total.wall += row.wall;
            total.on_cpu += row.on_cpu;
            total.counts = total.counts.and(row.counts);
        }
        let mut walls: Vec<u64> = rows.iter().map(|row| row.wall).collect();
        let round_median = median_us(&mut walls);
        medians.push(round_median);
        println!(
            "round {round}: {} calls, median {round_median:.1} us, total {:.1} us",
            rows.len(),
            total.wall as f64 / 1000.0
        );
        line(&format!("r{round}"), &total);
        drop(device);
    }
    // The suite's figure: the median of the rounds' own medians.
    medians.sort_by(|one, two| one.partial_cmp(two).expect("no NaN"));
    println!(
        "MEDIAN OF THE ROUNDS' MEDIANS {:.1} us, least {:.1}, most {:.1}",
        medians[medians.len() / 2],
        medians.first().copied().unwrap_or_default(),
        medians.last().copied().unwrap_or_default(),
    );
    Ok(())
}

/// The fixture's library on `device`, and the tile function's descriptor.
fn tile_library(device: &Device) -> Result<FunctionDescriptor, String> {
    let fixture = Fixture::new(TILE_FIXTURE);
    let bitcode = fixture.bitcode()?;
    let library = metal2vulkan::library::Library::parse(&bitcode)
        .map_err(|error| format!("the fixture did not parse: {error}"))?;
    let library = device.new_library_with_compile_library(library);
    let function = library
        .new_function_with_name(TILE_FIXTURE)
        .map_err(|error| format!("no function {TILE_FIXTURE}: {error}"))?;
    Ok(function.function_descriptor())
}

/// The case's tile descriptor with `limit` its `max_total_threads_per_threadgroup`.
fn tile_descriptor(stage: StageSource, limit: Option<u32>) -> TileRenderPipelineDescriptor {
    let mut descriptor = TileRenderPipelineDescriptor::new(stage);
    descriptor.color_attachment_formats[0] = Some(TILE_ATTACHMENT);
    descriptor.max_total_threads_per_threadgroup = limit;
    descriptor
}

/// `tile_inputs` of the case's descriptor, as `pipeline/compile/api.rs` builds it: the lowering
/// inputs a tile function of one `R32Float` attachment compiles under.
fn tile_inputs() -> LoweringInputs {
    let mut inputs = LoweringInputs::new();
    inputs.tile_attachments = Some([(0u32, TILE_ATTACHMENT)].into_iter().collect());
    inputs
}

/// `times` tile-state creations in one process, each timed, the median reported.
///
/// `fresh` gives every creation a `max_total_threads_per_threadgroup` no earlier one used, so its
/// pipeline key is new and the memo cannot answer it, which is the case's shape. Otherwise one
/// descriptor repeats and the memo answers from the second call on.
fn loop_mode(times: usize, directory: &str, memo: bool, fresh: bool) -> Result<(), String> {
    let notices = Notices::default();
    let device = Device::system_default_with(&options(&notices, directory, memo))
        .map_err(|error| format!("no device: {error}"))?;
    let function = tile_library(&device)?;
    let made = PipelineOptions::default();
    println!(
        "loop {times} venue {} dir {directory} memo {} keys {}",
        device.name(),
        if memo { "on" } else { "off" },
        if fresh { "fresh" } else { "same" }
    );
    header("at");
    let mut rows: Vec<Row> = Vec::with_capacity(times);
    // The states are kept so none is dropped inside another's window: a drop frees a module and
    // its layouts, which is work the creation did not do.
    let mut kept = Vec::with_capacity(times);
    for at in 0..times {
        // 64 upward: the limit only has to differ, and a tile function's own threadgroup bound is
        // what Metal clamps to, so no value here changes what is compiled.
        let limit = match fresh {
            true => Some(64 + at as u32),
            false => Some(512),
        };
        let descriptor = tile_descriptor(StageSource::Function(function.clone()), limit);
        let before = readings(&device);
        let cpu_before = metal2vulkan_conformance::clocks::thread_cpu_nanos();
        let started = Instant::now();
        let state = device.new_tile_render_pipeline_state(&descriptor, &made);
        let wall = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        let on_cpu = match (
            cpu_before,
            metal2vulkan_conformance::clocks::thread_cpu_nanos(),
        ) {
            (Some(before), Some(after)) => after.saturating_sub(before),
            _ => 0,
        };
        let after = readings(&device);
        kept.push(state.map_err(|error| format!("creation {at} refused: {error}"))?);
        rows.push(Row {
            wall,
            on_cpu,
            counts: after.since(before),
        });
    }
    for (at, row) in rows.iter().enumerate().take(8) {
        line(&at.to_string(), row);
    }
    // The first calls of a process are a warm-up and not calls: a process's first pipeline
    // creation has cost 1.0-13.7 ms. The medians are of everything after the eighth, and the
    // eight are printed above so a reader sees the two apart.
    let settled = rows.get(8..).unwrap_or_default();
    let mut walls: Vec<u64> = settled.iter().map(|row| row.wall).collect();
    let mut cpus: Vec<u64> = settled.iter().map(|row| row.on_cpu).collect();
    let wall_median = median_us(&mut walls);
    let cpu_median = median_us(&mut cpus);
    let mut counts = Readings::default();
    for row in settled {
        counts = counts.and(row.counts);
    }
    println!(
        "SETTLED {} calls  wall median {:.1} us p10 {:.1} p90 {:.1}  cpu median {:.1} us p10 \
         {:.1} p90 {:.1}",
        settled.len(),
        wall_median,
        percentile_us(&walls, 0.10),
        percentile_us(&walls, 0.90),
        cpu_median,
        percentile_us(&cpus, 0.10),
        percentile_us(&cpus, 0.90),
    );
    line(
        "PERCALL",
        &Row {
            wall: 0,
            on_cpu: 0,
            counts,
        },
    );
    // The cost in order, a tenth of the run at a time. A floor that is flat across the run is the
    // creation's own; one that climbs is paid for by something the run is filling, the memo among
    // it, and a single median over the whole run would read as the average of a slope.
    let tenth = (settled.len() / 10).max(1);
    let blocks: Vec<String> = settled
        .chunks(tenth)
        .map(|block| {
            let mut all: Vec<u64> = block.iter().map(|row| row.on_cpu).collect();
            format!("{:.1}", median_us(&mut all))
        })
        .collect();
    println!("CPU MEDIAN PER TENTH, in order: {}", blocks.join("  "));
    // The split the tail asks for. A creation whose artifact the cache thread could not take
    // writes it itself, and the statistics count that as `writes_on_thread`: four journalled
    // filesystem calls inside the creation, where a queued write costs it none
    // (`ShaderCacheStatistics::writes_on_thread`). So the calls are partitioned by that count
    // rather than by their cost, which is what makes this a reading of the cause and not of the
    // tail.
    let partition = |on_thread: bool| {
        let mut all: Vec<u64> = settled
            .iter()
            .filter(|row| (row.counts.writes_on_thread > 0) == on_thread)
            .map(|row| row.on_cpu)
            .collect();
        let count = all.len();
        (count, median_us(&mut all))
    };
    let (queued, queued_median) = partition(false);
    let (on_thread, on_thread_median) = partition(true);
    println!(
        "BY WHO WROTE: {queued} calls whose artifact the cache thread took, median {queued_median:.1} \
         us; {on_thread} that wrote it themselves, median {on_thread_median:.1} us"
    );
    Ok(())
}

/// The translator's own steps, each in its own loop over the case's descriptor: the marginal cost
/// of each, never a total divided by a count.
///
/// The step a **warm** creation pays is the link of a pipeline from a function already compiled,
/// so the loops below compile the function once and then run `compile_tile_pipeline` over a
/// descriptor whose stage is that `Compiled` function -- which is what `Variant::compiled` hands
/// `Sources::kernel` after the function memo answers (`pipeline/sources.rs`). The same call over a
/// `Function` stage is the cold form, and it is timed too, so the two are on one page.
fn parts_mode(times: usize, directory: &str) -> Result<(), String> {
    let notices = Notices::default();
    let device = Device::system_default_with(&options(&notices, directory, true))
        .map_err(|error| format!("no device: {error}"))?;
    let function = tile_library(&device)?;
    let compile = device.compile_device();
    let budget = Budget::standard();
    let inputs = tile_inputs();
    let cold = tile_descriptor(StageSource::Function(function.clone()), Some(512));
    println!("parts {times} venue {} dir {directory}", device.name());

    // The function compiled once, and the warm descriptor that carries it: every loop below reads
    // these rather than repeating them.
    let compiled = compile
        .compile_function(&function, &inputs, &budget)
        .map_err(|error| format!("the function compile refused: {error}"))?;
    let warm = tile_descriptor(StageSource::Compiled(compiled), Some(512));
    let key = compile.tile_pipeline_key(&warm);
    let pipeline = compile
        .compile_tile_pipeline(&warm, &budget)
        .map_err(|error| format!("the tile link refused: {error}"))?;
    let bytes = pipeline.to_bytes();
    println!(
        "the artifact is {} bytes, its module {} words; the key is the same over the two stage \
         forms: {}",
        bytes.len(),
        pipeline.stage.spirv.len(),
        compile.tile_pipeline_key(&cold) == key,
    );

    let timed = |name: &str, mut each: Box<dyn FnMut() -> u64 + '_>| {
        // A warm-up of a sixteenth of the run, discarded: the first calls of any loop pay the
        // instruction cache and the allocator's first growth.
        let warming = (times / 16).max(4);
        for _ in 0..warming {
            each();
        }
        let mut all: Vec<u64> = (0..times).map(|_| each()).collect();
        let median = median_us(&mut all);
        println!(
            "{name:<38} median {median:>9.3} us  p10 {:>9.3}  p90 {:>9.3}  over {times}",
            percentile_us(&all, 0.10),
            percentile_us(&all, 0.90),
        );
    };

    timed(
        "tile_pipeline_key (Function stage)",
        Box::new(|| {
            let started = Instant::now();
            let made = compile.tile_pipeline_key(&cold);
            let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            std::hint::black_box(&made);
            elapsed
        }),
    );
    timed(
        "tile_pipeline_key (Compiled stage)",
        Box::new(|| {
            let started = Instant::now();
            let made = compile.tile_pipeline_key(&warm);
            let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            std::hint::black_box(&made);
            elapsed
        }),
    );
    timed(
        "compile_tile_pipeline (warm: the link)",
        Box::new(|| {
            let started = Instant::now();
            let made = compile.compile_tile_pipeline(&warm, &budget);
            let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            std::hint::black_box(&made);
            elapsed
        }),
    );
    // The function artifact, which is the one a warm tile creation in the suite's own shape reads
    // back: `Sources::function` looks the function up before `Sources::kernel` looks the pipeline
    // up, and the directory answers it. Its decode runs `scan_relocatable` where a pipeline's runs
    // `scan_words` (`cache/artifacts.rs`), so this is the linear module scan on the tile path.
    let function_bytes = compiled_function_bytes(compile, &function, &inputs, &budget)?;
    println!("the function artifact is {} bytes", function_bytes.len());
    let function_key = compile.function_compile_key(&function, &inputs);
    timed(
        "function_compile_key",
        Box::new(|| {
            let started = Instant::now();
            let made = compile.function_compile_key(&function, &inputs);
            let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            std::hint::black_box(&made);
            elapsed
        }),
    );
    timed(
        "CompiledFunction::from_bytes_for_key",
        Box::new(|| {
            let started = Instant::now();
            let made = metal2vulkan::library::CompiledFunction::from_bytes_for_key(
                &function_bytes,
                &function_key,
                compile,
            );
            let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            std::hint::black_box(&made);
            elapsed
        }),
    );
    timed(
        "ComputePipeline::to_bytes (encode)",
        Box::new(|| {
            let started = Instant::now();
            let made = pipeline.to_bytes();
            let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            std::hint::black_box(&made);
            elapsed
        }),
    );
    timed(
        "from_bytes_for_key (decode + scan)",
        Box::new(|| {
            let started = Instant::now();
            let made = ComputePipeline::from_bytes_for_key(&bytes, &key, compile);
            let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            std::hint::black_box(&made);
            elapsed
        }),
    );
    // The cold form last: it is the dearest and it is not what a warm call pays, and running it
    // last keeps its allocations out of the loops above.
    timed(
        "compile_tile_pipeline (cold: translate)",
        Box::new(|| {
            let started = Instant::now();
            let made = compile.compile_tile_pipeline(&cold, &budget);
            let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
            std::hint::black_box(&made);
            elapsed
        }),
    );
    Ok(())
}
