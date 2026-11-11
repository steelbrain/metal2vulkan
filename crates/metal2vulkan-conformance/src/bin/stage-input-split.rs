//! Which side holds the cost of a `compute_pipeline_with_stage_input` call: our translation, the
//! driver's compile, or the rest of the call.
//!
//! The call's per-case total is the one pipeline family on `known-slow-calls.tsv` whose case total
//! goes the wrong way, and the ledger row localises its cost to the **first** creation on a shader
//! (1,900-6,900 us against about 55 us for the ones after it) without saying which side of us that
//! first creation pays. "Our own CPU" does not separate the two: a driver's pipeline compile runs
//! on the calling thread too.
//!
//! So this runs a case's own calls, with the case's own descriptors, on our runtime, and reads the
//! counters that tell the two apart **around each call**:
//!
//! | counter | what a `+1` says |
//! |---|---|
//! | `functions_translated` | we lowered the stage's function to SPIR-V on this call |
//! | `functions_shared` | we did not: it came from the device's memo, an archive or the cache |
//! | `shader_modules_made` | we made a `VkShaderModule` of it |
//! | `variant_pipelines_made` | we made a `VkPipeline`, so the driver was asked for one |
//! | `driver_pipeline_misses` | the driver says it compiled rather than read its cache |
//! | `driver_pipeline_hits` | the driver says it found the pipeline, or every stage of it |
//!
//! # The three cells, and how to get them
//!
//! The shader cache keeps our translations under `<root>/v1/<build>/` and the driver's own pipeline
//! cache data under `<root>/v1/<content>/` (`shader_cache/store.rs`), two directories a run can
//! clear apart. So one process per cell, over one cache directory:
//!
//! - **both** — a cache directory nothing has written. We translate and the driver compiles.
//! - **driver only** — the same directory with the content directory removed. We read our
//!   translation back and the driver compiles again.
//! - **neither** — the same directory untouched. We read ours back and the driver reads its own.
//!
//! The difference of the first two is our translation's share; of the second and third, the
//! driver's; what is left is the rest of the call. Each cell's counters say it was the cell it
//! claims to be, so a cell that did not land is visible rather than quietly averaged in.
//!
//! Metal underneath MoltenVK is warm whatever this does: `MTLCompilerService` keys compiled output
//! on the AIR and keeps it across processes, and no public call empties it. That is the same
//! footing the suite's own leg runs on, so these cells are of the suite's warm driver, never of a
//! cold machine.
//!
//! # What it read, and the one count that answers the question
//!
//! **No `VkPipeline` is made inside the call at all.** Over `attribute-format-fetches`' 29 calls,
//! in all three cells and in two runs of the three, `variant_pipelines_made`,
//! `driver_pipeline_hits` and `driver_pipeline_misses` are **0 on every call**, while
//! `shader_modules_made` reads 1 and exactly one of `functions_translated` and
//! `functions_shared` reads 1. The creation predicts a region's pipeline only where the descriptor states
//! `requiredThreadsPerThreadgroup` (`pipeline/dispatch_making.rs`), which these descriptors do
//! not, so the dispatch hands the making to the device's pool and the driver's compile is never on
//! the calling thread. Removing the driver's records changes neither a count nor the cost, which
//! is the same statement from the other side.
//!
//! So the call's cost is **ours**, and the cells split it (Apple M3 Max, macOS 27.0.1, MoltenVK
//! 1.4.2, release, with the GPU to itself, the median of the 28 calls after the first; two runs at
//! one-minute loads of 6.65 and 6.21, whose absolutes move 1.3x and whose share does not):
//!
//! | part | us a call | share |
//! |---|---|---|
//! | our AIR-to-SPIR-V translation | 243 / 322 | 62% / 63% |
//! | the rest of our call, a module still made | 149 / 190 | 38% / 37% |
//! | the driver's pipeline compile | 0 | 0% |
//!
//! The first call of a process is not one of these: it costs 1.0 to 13.7 ms and moves by an order
//! of magnitude between runs, so a case making one such call reads a process's warm-up and not a
//! call.
//!
//! # Running it
//!
//! ```text
//! stage-input-split <case> <cache-directory>
//! ```
//!
//! A reading, not a gate: it prints a row per call and exits 0 unless the case itself failed.

use metal2vulkan_conformance::device::Notices;
use metal2vulkan_conformance::runtime_side::RuntimeSide;
use metal2vulkan_conformance::transcript::Sites;
use metal2vulkan_conformance::watch::{Watcher, Watching};
use metal2vulkan_runtime::{Degradation, Device, DeviceKind, DeviceOptions, ShaderCacheOptions};
use std::cell::RefCell;
use std::rc::Rc;
use std::time::Instant;

/// The call this reads. Every other call is forwarded untimed and uncounted, so no other call's
/// pair of counter reads lands inside this one's window.
const CALL: &str = "compute_pipeline_with_stage_input";

/// What one call did, as the counters and the clocks read it.
#[derive(Clone, Copy, Debug, Default)]
struct Row {
    wall: u64,
    on_cpu: u64,
    translated: u64,
    functions_shared: u64,
    modules_made: u64,
    modules_shared: u64,
    pipelines_made: u64,
    driver_hits: u64,
    driver_misses: u64,
}

/// The counters as they stand, of both halves.
#[derive(Clone, Copy, Debug, Default)]
struct Readings {
    translated: u64,
    functions_shared: u64,
    modules_made: u64,
    modules_shared: u64,
    pipelines_made: u64,
    driver_hits: u64,
    driver_misses: u64,
}

fn readings(device: &Device) -> Readings {
    let counts = device.work_counts();
    let cache = device.shader_cache_statistics();
    Readings {
        translated: counts.functions_translated,
        functions_shared: counts.functions_shared,
        modules_made: counts.shader_modules_made,
        modules_shared: counts.shader_modules_shared,
        pipelines_made: counts.variant_pipelines_made,
        driver_hits: cache.driver_pipeline_hits,
        driver_misses: cache.driver_pipeline_misses,
    }
}

/// The device and the log a span writes into. It owns a clone of the device (a retain) so a span
/// borrows nothing: [`Watcher::Span`] has no lifetime of its own.
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
        let before = self.before;
        shared.log.borrow_mut().push(Row {
            wall,
            on_cpu,
            translated: after.translated - before.translated,
            functions_shared: after.functions_shared - before.functions_shared,
            modules_made: after.modules_made - before.modules_made,
            modules_shared: after.modules_shared - before.modules_shared,
            pipelines_made: after.pipelines_made - before.pipelines_made,
            driver_hits: after.driver_hits - before.driver_hits,
            driver_misses: after.driver_misses - before.driver_misses,
        });
    }
}

/// The suite's device options, with the shader cache in `directory` rather than a scratch
/// directory the run owns: the cells are three processes over one directory.
fn options(notices: &Notices, directory: &str) -> DeviceOptions {
    let mut options = DeviceOptions::default();
    options.policy = options.policy.exact_answers();
    options.announce_each = Degradation::ALL.iter().copied().collect();
    options.shader_cache = ShaderCacheOptions::in_directory(std::path::Path::new(directory));
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
            eprintln!("stage-input-split: {why}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut words = std::env::args().skip(1);
    let name = words
        .next()
        .ok_or("usage: stage-input-split <case> <cache-directory>")?;
    let directory = words
        .next()
        .ok_or("usage: stage-input-split <case> <cache-directory>")?;
    let case = metal2vulkan_conformance::cases::named(&name).ok_or(format!("no case {name}"))?;
    let mut notices = Notices::default();
    let device = Device::system_default_with(&options(&notices, &directory))
        .map_err(|error| format!("no device: {error}"))?;
    notices.venue();
    let shared = Rc::new(Shared {
        device: device.clone(),
        log: RefCell::new(Vec::new()),
    });
    let watcher = Split(Rc::clone(&shared));
    let mut sites = Sites::new();
    let side = RuntimeSide::new(&device, notices.open(case.name()));
    let mut watching = Watching::with(side, watcher);
    let outcome = case.run(&mut watching, &mut sites);
    drop(watching);
    let rows = shared.log.borrow().clone();
    println!("case {name} venue {} dir {directory}", device.name());
    println!(
        "{:>4}  {:>10} {:>10}  {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4}",
        "call", "wall us", "cpu us", "xlat", "fshr", "modm", "mods", "pipm", "dhit", "dmis"
    );
    for (at, row) in rows.iter().enumerate() {
        println!(
            "{at:>4}  {:>10.1} {:>10.1}  {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4}",
            row.wall as f64 / 1000.0,
            row.on_cpu as f64 / 1000.0,
            row.translated,
            row.functions_shared,
            row.modules_made,
            row.modules_shared,
            row.pipelines_made,
            row.driver_hits,
            row.driver_misses
        );
    }
    let total: u64 = rows.iter().map(|row| row.wall).sum();
    let mut walls: Vec<u64> = rows.iter().map(|row| row.wall).collect();
    walls.sort_unstable();
    let median = walls.get(walls.len() / 2).copied().unwrap_or_default();
    println!(
        "TOTALS calls {} wall {:.1} us median {:.1} us  xlat {} fshr {} modm {} mods {} pipm {} \
         dhit {} dmis {}",
        rows.len(),
        total as f64 / 1000.0,
        median as f64 / 1000.0,
        rows.iter().map(|r| r.translated).sum::<u64>(),
        rows.iter().map(|r| r.functions_shared).sum::<u64>(),
        rows.iter().map(|r| r.modules_made).sum::<u64>(),
        rows.iter().map(|r| r.modules_shared).sum::<u64>(),
        rows.iter().map(|r| r.pipelines_made).sum::<u64>(),
        rows.iter().map(|r| r.driver_hits).sum::<u64>(),
        rows.iter().map(|r| r.driver_misses).sum::<u64>()
    );
    if let Err(abort) = outcome {
        return Err(format!(
            "the case aborted at {}: {:?}",
            abort.at, abort.failure
        ));
    }
    Ok(())
}
