//! Whole-corpus driver acceptance: does the Vulkan driver accept the pipeline we produce? Nothing
//! is bound or submitted, nothing is read back.
//!
//! That question has no other asker in this repository. `spirv-val` checks the module against the
//! SPIR-V specification; the authored cases put a real pipeline on the device but reach only a
//! fraction of the sources. A module can be valid SPIR-V, translate without error, and still be
//! refused by the driver.
//!
//! It runs the engine: `--profile P` compiles every kernel (compute pipeline), vertex function
//! (rasterization-disabled render pipeline, with a vertex descriptor for the attributes it
//! declares), and fragment function (drawn with its generated passthrough vertex companion into
//! the attachments its outputs imply) with the engine's public compile API for profile P
//! (`acceptance::row`). A render row's pipeline is created from the **binder layout the artifact
//! carries** (set layouts, push ranges, specialization, subgroup size, vertex input, graphics
//! state), its stages at their default specialization, not the graphics plan's (a limit). A
//! fragment function's attachments have no more channels than its outputs write. A kernel row is
//! made as a guest's kernel is, through the runtime (`candidate::RuntimeKernels`): its
//! `ComputePipelineState` on the sweep's one runtime device, opened once for all its kernel rows,
//! with the row's function-constant values, and one dispatch of one representative threadgroup
//! (the state's thread execution width of threads, or its `maxTotalThreadsPerThreadgroup` where
//! fewer) encoded into a command buffer that is never committed, so the runtime asks the driver
//! for the pipeline that threadgroup runs on. The
//! runtime's typed refusal (the translator's, a known gap, a device limit) is `RUNTIME_REFUSED`,
//! counted; a kernel that writes no memory, whose dispatch the runtime runs as Metal's without
//! effects and makes no pipeline for, is `NO_EFFECT`, counted. The make runs in this process,
//! held to the per-attempt ceilings, after the row's compile worker compiled it within them. It
//! times each `vkCreate*Pipelines` (`budgets::CreationClock`: its wall time less its thread's wait
//! for a CPU; a kernel's, the runtime's own count of its pipeline creation), with the drivers'
//! shader disk caches off (`budgets::turn_driver_shader_caches_off`); a creation above 5 s is a
//! bug. A row's detail prints its own stage's module words, which are what the translation sweep
//! (`corpus-triage --audit translation`) counts for it, a render row's other stages' apart as
//! `companion_words`, then the pipeline's precision notes as `precision=[...]`. `--driver-caches-on` leaves the drivers' shader disk caches as the environment sets
//!   them, so a creation may hit what an earlier process compiled: run twice, the second run times
//!   the warm creation a program's later launches see, beside the cold one the ceiling times (the
//!   summary's `driver_caches`). `--keep-modules DIR` writes each compiled row's stage modules,
//!   their specialization layouts and lowering strategies' words into DIR, a diagnostic (with
//!   `--compile-only` on any host). `--pulled` compiles every vertex
//!   attribute pulled in the shader instead of the projection the product derives: the repeatable
//!   form of the pulled sweep, with `--compile-only` or on a device. `--fixtures` sweeps the owned
//!   `validation/fixtures/public/engine` sources instead of the corpus, which is what the Linux
//!   containers can run. A row that retains no bitcode (a public source) is assembled from its AIR
//!   text.
//!
//! It compiles every row in a killable compile worker (this executable started with
//! `worker::compile::WORKER_ARG`) under the 20-s and 500-MiB ceilings, which cover assembling a
//! row's text and deriving its descriptor or companion; the pipeline comes back as its cache
//! artifact bytes, and a row the worker boundary stopped is `WORKER_STOPPED`, which fails the sweep
//! A row's last column is its compile worker's time in milliseconds (`-` where the worker reported
//! none), beside the creation's in the sixth.
//!
//! `--specializations FILE` sweeps only the corpus rows FILE lists, each compiled and created
//! with its listed function-constant values, its descriptor and companion derived under them
//! (`acceptance::row`): one JSON object per line, `{"row": "<source hash>", "set": <set>}`, the
//! set as the specialization planner writes it (`specialization::Specialization`; a constant it
//! does not supply stays undefined). Every other row compiles with no values, Metal's undefined,
//! where a row whose code uses a constant's value stops at `needs_input`. A row's detail then
//! starts with `set=<kinds>:<id>` (the set's kinds and the first 16 digits of its identity), and
//! the summary's `specializations` is the number of rows FILE lists (`none` without it).
//!
//! `--draws FILE` compiles and creates the draws an application logged instead of rows: one JSON
//! object per line, `{"id": <draw id>, "vertex": <source hash>, "fragment": <source hash>,
//! "template": <RenderTemplate>}`, each the vertex and fragment functions of two corpus rows
//! drawn together under the descriptor the application created the pipeline with (its vertex
//! descriptor, color attachments with their blend state and write masks, depth and stencil
//! formats, topology class and sample count; `acceptance::compile_draw_in_worker`), with no
//! function-constant values. A draw's line is keyed by its id, its entry is the two functions'
//! entries joined by `+`, and its stage `Draw`; the summary's `draws` is the number of draws
//! (`none` without it). A source a draw names that the corpus lacks fails the sweep.
//!
//! `--partition K/N` sweeps only the rows whose source shard is congruent to K modulo N, so N
//! processes sweep disjoint parts of the corpus side by side and none reads another's shards. A
//! refusal on MoltenVK carries the driver's report, and the row's last column names the Metal
//! compile step that failed and whether Metal's compiler service crashed or the step timed out
//! (`acceptance::metal_report`); the summary counts the rows per class.
//!
//! The device context puts back the loader's dispatch word that MoltenVK 1.4.1 overwrites when it
//! reports a message about the device, before the loader reads it again (in `vkDestroyDevice` at
//! exit, where a run faulted after its summary before; `candidate::platform::loader_word`).

use metal2vulkan::library::FunctionConstantValues;
use metal2vulkan::pipeline::PrecisionNote;
use metal2vulkan_validation::acceptance::metal_report::MetalCompileFailure;
use metal2vulkan_validation::acceptance::row::UnmodelledKind;
use metal2vulkan_validation::acceptance::{
    compile_draw_in_worker, compile_row_in_worker, Acceptance, AcceptanceError, Projection,
    RowCompile, RowPipeline, RowStage, RuntimeRow,
};
use metal2vulkan_validation::budgets::PIPELINE_CREATION_CEILING;
use metal2vulkan_validation::candidate::engine::graphics::RenderTemplate;
use metal2vulkan_validation::candidate::RuntimeKernels;
use metal2vulkan_validation::engine::{self, FailureKind, Outcome, RowInput, RowSource};
use metal2vulkan_validation::gpu::wait::{parse_wait_timeout, set_wait_timeout};
use metal2vulkan_validation::gpu::DeviceType;
use metal2vulkan_validation::source::{
    self, for_each_source_shard, for_each_source_shard_analysis, shard_index_for_hash,
    source_shard_path, SourceRow, SHARD_COUNT,
};
use metal2vulkan_validation::specialization::Specialization;
use metal2vulkan_validation::worker::alloc::CountingAllocator;
use metal2vulkan_validation::worker::compile::{CompileWorker, WorkerDevice};
use metal2vulkan_validation::worker::Measured;
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;

// A compile worker's in-child budget is this allocator's (worker recipe step 2).
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// The flag that leaves the drivers' shader disk caches on.
const DRIVER_CACHES_ON: &str = "--driver-caches-on";

fn main() {
    // Every creation this sweep times compiles cold, not a hit in the driver's disk cache, unless
    // the run asks for warm creations. Decided here, before any thread or the loader.
    if !std::env::args().skip(1).any(|arg| arg == DRIVER_CACHES_ON) {
        metal2vulkan_validation::budgets::turn_driver_shader_caches_off();
    }
    metal2vulkan_validation::worker::compile::serve_if_requested();
    if let Err(error) = run() {
        eprintln!("corpus-pipeline: {error}");
        std::process::exit(1);
    }
}

fn usage() -> String {
    "usage: corpus-pipeline --profile P [--corpus DIR] [--private-corpus DIR] [--limit N]\n\
     \x20      [--after HASH | --hash HASH | --fixtures] [--partition K/N] [--attempting PATH]\n\
     \x20      [--pulled] [--driver-caches-on]\n\
     \x20      [--keep-modules DIR] [--specializations FILE | --draws FILE]\n\
     \x20      (--device-type cpu|gpu [--gpu-wait-timeout SECONDS] [--profiles-layer P]\n\
     \x20       | --compile-only)\n\
     It runs the engine. --profile P names a committed profile describing the device; it may take\n\
     --fixtures, --pulled (every vertex attribute pulled in the shader), or --compile-only\n\
     (compile every row, open no device).\n\
     --driver-caches-on leaves the drivers' shader disk caches as the environment sets them (warm\n\
     creations, when an earlier run compiled the same pipelines); every other run compiles cold.\n\
     --keep-modules DIR writes each compiled row's stage modules into DIR (a diagnostic): per stage\n\
     <hash>.<stage>.spv, and its entry point, specialization layout and strategy census (each\n\
     lowering strategy's uses and words) in <hash>.<stage>.json.\n\
     --draws FILE compiles and creates the draws FILE lists (one JSON object per line, {\"id\",\n\
     \"vertex\": <source hash>, \"fragment\": <source hash>, \"template\": <render template>}),\n\
     each its vertex and fragment functions under the descriptor an application logged.\n\
     --specializations FILE sweeps only the corpus rows FILE lists (one JSON object per line,\n\
     {\"row\": \"<source hash>\", \"set\": <specialization set>}), each with its function-constant\n\
     values.\n\
     --partition K/N sweeps only the corpus rows whose source shard is K modulo N (0 <= K < N <= 64),\n\
     so N processes sweep disjoint parts; --after then names a row of that part.\n\
     --profiles-layer P (on a device, naming --profile's P) restricts the device to\n\
     P through VK_LAYER_KHRONOS_profiles, after checking that it has P and reports exactly P\n\
     (validation spec 4.1.2's profile-restricted pipeline creation). A drm-shim gpu device requires scripts/drm-shim-exec inside the drm-shim\n\
     container; cpu matches only a CPU Vulkan implementation."
        .to_string()
}

struct Options {
    root: PathBuf,
    limit: usize,
    after: Option<String>,
    only: Option<String>,
    /// `--partition K/N`: the part of the corpus this process sweeps.
    partition: Option<Partition>,
    attempting: Option<PathBuf>,
    /// The committed profile `--profiles-layer` restricts the device to.
    profiles_layer: Option<String>,
    /// `--driver-caches-on`: `main` left the drivers' shader disk caches on.
    driver_caches: bool,
    /// `--keep-modules DIR`: where each compiled row's stage modules are written.
    keep_modules: Option<PathBuf>,
    /// `--specializations FILE`: the rows this sweep compiles, each with its values.
    specializations: Option<Specializations>,
    /// `--draws FILE`: the logged draws this sweep compiles instead of rows.
    draws: Option<Draws>,
}

fn run() -> Result<(), String> {
    let mut options = Options {
        root: source::corpus_root(),
        limit: usize::MAX,
        after: None,
        only: None,
        partition: None,
        attempting: None,
        profiles_layer: None,
        driver_caches: false,
        keep_modules: None,
        specializations: None,
        draws: None,
    };
    let mut device_type = None;
    let mut profile = None;
    let mut fixtures = false;
    let mut compile_only = false;
    let mut projection = Projection::Derived;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--corpus" => options.root = PathBuf::from(value()?),
            "--private-corpus" => {
                metal2vulkan_validation::source::use_private_corpus_root(PathBuf::from(value()?))?
            }
            "--limit" => {
                options.limit = value()?
                    .parse()
                    .map_err(|error| format!("--limit: {error}"))?
            }
            "--after" => options.after = Some(value()?),
            "--hash" => options.only = Some(value()?),
            "--partition" => options.partition = Some(Partition::parse(&value()?)?),
            "--attempting" => options.attempting = Some(PathBuf::from(value()?)),
            "--device-type" => device_type = Some(DeviceType::parse(&value()?)?),
            "--gpu-wait-timeout" => set_wait_timeout(parse_wait_timeout(&value()?)?)?,
            "--profile" => profile = Some(value()?),
            "--fixtures" => fixtures = true,
            "--compile-only" => compile_only = true,
            "--profiles-layer" => options.profiles_layer = Some(value()?),
            "--pulled" => projection = Projection::Pulled,
            DRIVER_CACHES_ON => options.driver_caches = true,
            "--keep-modules" => options.keep_modules = Some(PathBuf::from(value()?)),
            "--draws" => {
                let path = PathBuf::from(value()?);
                let text = std::fs::read_to_string(&path)
                    .map_err(|error| format!("read {}: {error}", path.display()))?;
                options.draws = Some(
                    Draws::parse(&text).map_err(|error| format!("{}: {error}", path.display()))?,
                );
            }
            "--specializations" => {
                let path = PathBuf::from(value()?);
                let text = std::fs::read_to_string(&path)
                    .map_err(|error| format!("read {}: {error}", path.display()))?;
                options.specializations = Some(
                    Specializations::parse(&text)
                        .map_err(|error| format!("{}: {error}", path.display()))?,
                );
            }
            "--help" | "-h" => {
                println!("{}", usage());
                return Ok(());
            }
            other => return Err(format!("unrecognized argument {other:?}\n{}", usage())),
        }
    }

    if options.only.is_some() && options.after.is_some() {
        return Err("--hash and --after are mutually exclusive".into());
    }
    check_partition(&options, fixtures)?;
    if compile_only && device_type.is_some() {
        return Err("--compile-only opens no device".into());
    }
    if let Some(restricted) = &options.profiles_layer {
        if compile_only {
            return Err(
                "--profiles-layer restricts a device, which --compile-only opens none of".into(),
            );
        }
        // The sweep compiles for P and creates the pipelines on the device restricted to P.
        if profile.as_deref() != Some(restricted.as_str()) {
            return Err(format!(
                "--profiles-layer {restricted} must name the --profile the sweep compiles for"
            ));
        }
        metal2vulkan_validation::candidate::restrict_devices_to(restricted)?;
    }
    let device_type = match device_type {
        Some(device_type) => Some(device_type),
        None if compile_only => None,
        None => return Err(format!("--device-type is required\n{}", usage())),
    };
    let profile = profile.ok_or("the engine needs --profile P")?;
    if fixtures && (options.after.is_some() || options.only.is_some()) {
        return Err("--fixtures takes no --after or --hash".into());
    }
    if fixtures && options.specializations.is_some() {
        return Err("--specializations lists corpus rows; --fixtures sweeps none".into());
    }
    if options.draws.is_some()
        && (fixtures
            || options.specializations.is_some()
            || options.partition.is_some()
            || options.only.is_some()
            || options.after.is_some())
    {
        return Err(
            "--draws sweeps the draws it lists: it takes no --fixtures, --specializations, \
             --partition, --hash or --after"
                .into(),
        );
    }
    let compiler = compile_device(&profile)?;
    let worker = CompileWorker::current()?;
    sweep_new(
        device_type,
        &worker,
        &compiler,
        &Sweep {
            profile: &profile,
            fixtures,
            projection,
        },
        &options,
    )
}

/// The compile device of an engine sweep of profile `stem` as the runtime's device describes it
/// (`engine::runtime_description`), created under the default policy.
fn compile_device(stem: &str) -> Result<WorkerDevice, String> {
    let description = engine::runtime_description(stem)?;
    WorkerDevice::new(description, metal2vulkan::device::DevicePolicy::default())
        .map_err(|error| format!("{stem}: {error}"))
}

/// What one engine sweep compiles: profile P, the corpus or the owned fixtures, and the
/// projection of its vertex attributes.
struct Sweep<'a> {
    profile: &'a str,
    fixtures: bool,
    projection: Projection,
}

/// The part of the corpus one process of a split sweep visits (`--partition K/N`): every row
/// whose source shard ([`shard_index_for_hash`], the corpus's own bucketing) is `index` modulo
/// `count`. Shards outside the part are never opened, and each public row falls in the part of
/// its hash's shard, so the N parts are disjoint and together are the whole sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Partition {
    index: usize,
    count: usize,
}

impl Partition {
    fn parse(text: &str) -> Result<Partition, String> {
        let invalid =
            || format!("--partition {text:?}: expected K/N with 0 <= K < N <= {SHARD_COUNT}");
        let (index, count) = text.split_once('/').ok_or_else(invalid)?;
        let index = index.parse::<usize>().map_err(|_| invalid())?;
        let count = count.parse::<usize>().map_err(|_| invalid())?;
        if count == 0 || count > SHARD_COUNT || index >= count {
            return Err(invalid());
        }
        Ok(Partition { index, count })
    }

    fn holds_shard(&self, shard: usize) -> bool {
        shard % self.count == self.index
    }
}

impl std::fmt::Display for Partition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.index, self.count)
    }
}

/// `--partition`'s rules: it splits the corpus walk, so it takes no `--hash` or `--fixtures`, and
/// `--after` must name a row of the part (a resumed part restarts past one of its own rows; any
/// other hash would skip the whole part).
fn check_partition(options: &Options, fixtures: bool) -> Result<(), String> {
    let Some(partition) = options.partition else {
        return Ok(());
    };
    if options.only.is_some() || fixtures {
        return Err("--partition splits the corpus: it takes no --hash or --fixtures".into());
    }
    if let Some(after) = &options.after {
        if !partition.holds_shard(shard_index_for_hash(after)?) {
            return Err(format!(
                "--after {after} is not a row of --partition {partition}"
            ));
        }
    }
    Ok(())
}

/// Visit every corpus source in the fixed shard order, then the `public` rows (the public
/// fixtures, [`source::public_sources`]), honoring `--limit`/`--after`/`--hash`/`--partition`.
/// `full` reads each row's retained bitcode.
fn visit_sources(
    options: &Options,
    full: bool,
    public: impl FnOnce() -> Result<Vec<SourceRow>, String>,
    mut visit: impl FnMut(SourceRow) -> Result<(), String>,
) -> Result<(), String> {
    if let Some(hash) = &options.only {
        let row = source::find_source(&options.root, hash)?
            .ok_or_else(|| format!("source {hash} was not found"))?;
        return visit(row);
    }
    // `--after` resumes a sweep the driver killed: shards are walked in a fixed order, so
    // restarting past the last hash reported covers exactly the remainder.
    let mut skipping = options.after.is_some();
    let mut filtered = |row: SourceRow| -> Result<(), String> {
        if skipping {
            if options.after.as_deref() == Some(row.key().as_ref()) {
                skipping = false;
            }
            return Ok(());
        }
        visit(row)
    };
    let in_part = |shard: usize| options.partition.is_none_or(|part| part.holds_shard(shard));
    for shard in (0..SHARD_COUNT).filter(|&shard| in_part(shard)) {
        let path = source_shard_path(&options.root, shard);
        if !path.is_file() {
            continue;
        }
        if full {
            for_each_source_shard(&path, &mut filtered)?;
        } else {
            for_each_source_shard_analysis(&path, &mut filtered)?;
        }
    }
    for row in public()? {
        if in_part(shard_index_for_hash(&row.key())?) {
            filtered(row)?;
        }
    }
    Ok(())
}

/// A driver that wedges takes the sweep with it: record the row BEFORE attempting it, so a
/// supervisor that kills a stalled run can resume past the module that stalled.
fn mark_attempting(options: &Options, key: &str) -> Result<(), String> {
    if let Some(path) = &options.attempting {
        std::fs::write(path, format!("{key}\n"))
            .map_err(|error| format!("write {}: {error}", path.display()))?;
    }
    Ok(())
}

/// One engine row: a corpus source (its retained bitcode, or the AIR text of a public row
/// that retains none) or an owned fixture's entry, keyed by its source hash or file name.
struct NewRow {
    /// The source hash, fixture file name or draw id a result line and the summary name it by.
    key: String,
    /// The function's entry point; a draw's vertex and fragment entries, joined by `+`.
    entry: String,
    /// The function's stage as the index records it; `Draw` for a draw.
    stage: String,
    what: RowWork,
}

/// What a row compiles.
enum RowWork {
    /// The function alone, into the pipeline the sweep derives for it, with its function-constant
    /// values (`--specializations`; none: undefined).
    Function {
        input: RowInput<'static>,
        specialization: Option<Specialization>,
    },
    /// A logged draw (`--draws`): its vertex and fragment functions' sources, in that order, and
    /// the descriptor the application created the pipeline with.
    Draw {
        libraries: Box<[RowSource<'static>; 2]>,
        vertex: String,
        fragment: String,
        template: RenderTemplate,
    },
}

/// The draws `--draws FILE` lists: one JSON object per line, `{"id": <draw id>, "vertex": <source
/// hash>, "fragment": <source hash>, "template": <RenderTemplate>}`, the template the descriptor
/// an application created the pipeline with (its vertex descriptor, color attachments with their
/// blend state and write masks, depth and stencil formats, topology class and sample count).
struct Draws(Vec<Draw>);

#[derive(Deserialize)]
struct Draw {
    id: String,
    vertex: String,
    fragment: String,
    template: RenderTemplate,
}

impl Draws {
    /// The draws of `text`; a line that does not parse, or an id listed twice, is an error, and
    /// blank lines are skipped.
    fn parse(text: &str) -> Result<Draws, String> {
        let mut draws: Vec<Draw> = Vec::new();
        for (number, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let draw: Draw = serde_json::from_str(line)
                .map_err(|error| format!("line {}: {error}", number + 1))?;
            if draws.iter().any(|other| other.id == draw.id) {
                return Err(format!(
                    "line {}: draw {} is listed twice",
                    number + 1,
                    draw.id
                ));
            }
            draws.push(draw);
        }
        Ok(Draws(draws))
    }

    /// Every source the draws compile.
    fn sources(&self) -> std::collections::BTreeSet<&str> {
        self.0
            .iter()
            .flat_map(|draw| [draw.vertex.as_str(), draw.fragment.as_str()])
            .collect()
    }
}

/// The rows `--specializations FILE` lists, each with the function-constant value set it is
/// compiled and created with.
struct Specializations(BTreeMap<String, Specialization>);

impl Specializations {
    /// The rows of `text`: one JSON object per line, `{"row": "<source hash>", "set": <set>}`.
    /// A line that does not parse, or a row listed twice, is an error; blank lines are skipped.
    fn parse(text: &str) -> Result<Specializations, String> {
        #[derive(Deserialize)]
        struct Line {
            row: String,
            set: Specialization,
        }
        let mut rows = BTreeMap::new();
        for (number, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let line: Line = serde_json::from_str(line)
                .map_err(|error| format!("line {}: {error}", number + 1))?;
            if rows.insert(line.row.clone(), line.set).is_some() {
                return Err(format!(
                    "line {}: row {} is listed twice",
                    number + 1,
                    line.row
                ));
            }
        }
        Ok(Specializations(rows))
    }

    /// The set row `key` is listed with.
    fn of(&self, key: &str) -> Option<&Specialization> {
        self.0.get(key)
    }
}

/// How a row's detail names its set: its kinds and the first 16 digits of its identity.
fn set_label(set: &Specialization) -> String {
    let id = set.id();
    format!("{}:{}", set.kinds_key(), id.get(..16).unwrap_or(&id))
}

/// What the OK and COMPILED detail of a row prints: `words=N`, the row's own function's stage, the
/// module `corpus-triage --audit translation` compiles for the row and counts; then
/// `companion_words=M` when the pipeline has other stages (a fragment row's generated passthrough
/// vertex companion, a post-tessellation vertex row's synthesized stages), which the translation
/// sweep does not compile; then `precision=[...]` when any stage of the pipeline carries a
/// precision note.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RowDetail {
    own: u64,
    companion: u64,
    precision: Vec<PrecisionNote>,
}

impl RowDetail {
    fn of(pipeline: &RowPipeline) -> RowDetail {
        RowDetail {
            own: pipeline.words(),
            companion: pipeline.companion_words(),
            precision: pipeline.precision_notes(),
        }
    }
}

impl std::fmt::Display for RowDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "words={}", self.own)?;
        if self.companion > 0 {
            write!(f, " companion_words={}", self.companion)?;
        }
        if !self.precision.is_empty() {
            write!(f, " precision={:?}", self.precision)?;
        }
        Ok(())
    }
}

/// The tallies of an engine sweep; any refusal, over-ceiling creation, worker-boundary stop,
/// or tooling error other than a modelled limit (an unmodelled row input, a vertex-input state
/// the creation path does not build) fails it (see [`Tally::verdict`]).
#[derive(Default)]
struct Tally {
    rows: usize,
    compiled: usize,
    created: usize,
    refused: usize,
    not_enabled: usize,
    not_compiled: usize,
    unmodelled: usize,
    /// The `UNMODELLED` rows by kind (`acceptance::row::UnmodelledKind`).
    unmodelled_kinds: BTreeMap<UnmodelledKind, usize>,
    /// Kernel rows the runtime refused with its typed error (`RUNTIME_REFUSED`).
    runtime_refused: usize,
    /// Kernel rows whose dispatch the runtime ran without effects, making no pipeline
    /// (`NO_EFFECT`).
    no_effect: usize,
    stage_not_covered: usize,
    no_bitcode: usize,
    tooling: usize,
    errors: usize,
    over_ceiling: usize,
    invalid_usage: usize,
    worker_stopped: usize,
    /// The refused and invalid-usage rows by the Metal compile step MoltenVK reported failing
    /// ([`MetalCompileFailure`]).
    metal_failures: BTreeMap<MetalCompileFailure, usize>,
    slowest: Option<(String, Duration)>,
    /// The slowest compile worker (input preparation through the decoded artifact).
    slowest_compile: Option<(String, Duration)>,
    /// The current row's compile worker time, when the worker reported one: its row's last
    /// column.
    row_compile: Option<Duration>,
    /// The peak compile-worker process-group RSS.
    peak_compile_rss: Option<(String, u64)>,
    /// The largest counting-allocator peak a compile worker reported, which the RSS sampler can
    /// miss.
    peak_compile_allocator: Option<(String, u64)>,
}

impl Tally {
    /// Count what the driver did with row `key`'s pipeline (its [`RowDetail`]) and return
    /// its `(STATUS, detail, create_ms)`. An unavailable device is `Err`: it ends the sweep,
    /// because every later row would be untried too and a sweep that created nothing must not
    /// pass (verifier finding). A modelled limit of the creation path is `TOOLING`; any other
    /// tooling error is `ERROR` and fails the sweep through [`Tally::verdict`].
    fn record(
        &mut self,
        key: &str,
        detail: RowDetail,
        result: Result<Acceptance, AcceptanceError>,
    ) -> Result<(&'static str, String, u128), String> {
        match result {
            Ok(Acceptance::Created { create }) => {
                self.created += 1;
                if self.slowest.as_ref().is_none_or(|(_, d)| create > *d) {
                    self.slowest = Some((key.to_string(), create));
                }
                if create > PIPELINE_CREATION_CEILING {
                    self.over_ceiling += 1;
                    return Ok(("OVER_CEILING", String::new(), create.as_millis()));
                }
                Ok(("OK", detail.to_string(), create.as_millis()))
            }
            Ok(Acceptance::NotEnabled(missing)) => {
                self.not_enabled += 1;
                Ok(("NOT_ENABLED", missing.join(", "), 0))
            }
            Ok(Acceptance::Refused(error)) => {
                self.refused += 1;
                Ok(("REFUSED", error, 0))
            }
            Ok(Acceptance::InvalidUsage(messages)) => {
                self.invalid_usage += 1;
                Ok(("INVALID_USAGE", messages.join(" | "), 0))
            }
            Ok(Acceptance::RuntimeRefused(why)) => {
                self.runtime_refused += 1;
                Ok(("RUNTIME_REFUSED", why, 0))
            }
            Ok(Acceptance::NoEffect) => {
                self.no_effect += 1;
                Ok(("NO_EFFECT", detail.to_string(), 0))
            }
            Err(AcceptanceError::Unmodelled(why)) => {
                self.tooling += 1;
                Ok(("TOOLING", why, 0))
            }
            Err(AcceptanceError::Tooling(why)) => {
                self.errors += 1;
                Ok(("ERROR", why, 0))
            }
            Err(error @ AcceptanceError::DeviceUnavailable(_)) => {
                Err(format!("{key}: {error}; the sweep stops"))
            }
        }
    }

    /// The class column of a row whose result is `status` with `detail`: the Metal compile step
    /// that failed, read from MoltenVK's report of a refused or invalid-usage row and counted, or
    /// `-`.
    fn metal_failure(&mut self, status: &str, detail: &str) -> String {
        if !matches!(status, "REFUSED" | "INVALID_USAGE") {
            return "-".into();
        }
        match MetalCompileFailure::classify(detail) {
            Some(class) => {
                *self.metal_failures.entry(class).or_default() += 1;
                class.key()
            }
            None => "-".into(),
        }
    }

    /// Count what the parent measured around row `key`'s compile worker.
    fn measured_compile(&mut self, key: &str, measured: Measured) {
        self.row_compile = Some(measured.elapsed);
        if self
            .slowest_compile
            .as_ref()
            .is_none_or(|(_, elapsed)| measured.elapsed > *elapsed)
        {
            self.slowest_compile = Some((key.to_string(), measured.elapsed));
        }
        for (bytes, peak) in [
            (measured.peak_resident_bytes, &mut self.peak_compile_rss),
            (
                measured.allocator_peak_bytes,
                &mut self.peak_compile_allocator,
            ),
        ] {
            if let Some(bytes) = bytes {
                if peak.as_ref().is_none_or(|(_, most)| bytes > *most) {
                    *peak = Some((key.to_string(), bytes));
                }
            }
        }
    }

    /// Whether the sweep passed: no refusal, no creation above the ceiling, no message from the
    /// validation layer, no compile the worker boundary stopped, and no tooling error other than
    /// a modelled limit (of the row's inputs or of the creation path).
    fn verdict(&self) -> Result<(), String> {
        if self.refused + self.over_ceiling + self.invalid_usage + self.errors + self.worker_stopped
            > 0
        {
            return Err(format!(
                "{} pipelines refused, {} created above the {} s ceiling, {} reported by the \
                 validation layer, {} compiles stopped at the worker boundary, and {} tooling \
                 errors",
                self.refused,
                self.over_ceiling,
                PIPELINE_CREATION_CEILING.as_secs(),
                self.invalid_usage,
                self.worker_stopped,
                self.errors
            ));
        }
        Ok(())
    }
}

/// `device` is `None` for `--compile-only`: every row is compiled and nothing is created.
fn sweep_new(
    device: Option<DeviceType>,
    worker: &CompileWorker,
    compiler: &WorkerDevice,
    sweep: &Sweep<'_>,
    options: &Options,
) -> Result<(), String> {
    let &Sweep {
        profile,
        fixtures,
        projection,
    } = sweep;
    let mut out = rows_output()?;
    let mut tally = Tally::default();
    // Opened at the sweep's first kernel row, and finished once the last row is done.
    let mut kernels = None;
    let mut attempt = |row: NewRow| -> Result<(), String> {
        if tally.rows >= options.limit {
            return Ok(());
        }
        tally.rows += 1;
        mark_attempting(options, &row.key)?;
        tally.row_compile = None;
        let (status, detail, create_ms) = attempt_row(
            device,
            &mut kernels,
            worker,
            compiler,
            projection,
            &row,
            options.keep_modules.as_deref(),
            &mut tally,
        )?;
        let class = tally.metal_failure(status, &detail);
        let detail = match &row.what {
            RowWork::Function {
                specialization: Some(set),
                ..
            } => format!("set={} {detail}", set_label(set)),
            _ => detail,
        };
        let compile_ms = tally.row_compile.map_or_else(
            || "-".to_string(),
            |elapsed| elapsed.as_millis().to_string(),
        );
        writeln!(
            out,
            "{}\t{}\t{}\t{status}\t{}\t{create_ms}\t{class}\t{compile_ms}",
            row.key,
            row.entry,
            row.stage,
            one_line(&detail)
        )
        .map_err(|error| format!("write result: {error}"))?;
        out.flush().map_err(|error| format!("flush: {error}"))
    };
    if fixtures {
        for row in fixture_rows()? {
            attempt(row)?;
        }
    } else if let Some(draws) = &options.draws {
        // Each source once, whatever shard holds it; the draws then in the file's order.
        let wanted = draws.sources();
        let mut sources = BTreeMap::new();
        visit_sources(
            options,
            true,
            || Ok(Vec::new()),
            |row| {
                if wanted.contains(row.key().as_ref()) {
                    sources.insert(row.key().to_string(), RowInput::from(row));
                }
                Ok(())
            },
        )?;
        for draw in &draws.0 {
            let source = |hash: &str| {
                sources
                    .get(hash)
                    .ok_or_else(|| format!("draw {}: source {hash} is not in the corpus", draw.id))
            };
            let (vertex, fragment) = (source(&draw.vertex)?, source(&draw.fragment)?);
            attempt(NewRow {
                key: draw.id.clone(),
                entry: format!("{}+{}", vertex.entry, fragment.entry),
                stage: "Draw".into(),
                what: RowWork::Draw {
                    libraries: Box::new([vertex.source.clone(), fragment.source.clone()]),
                    vertex: vertex.entry.to_string(),
                    fragment: fragment.entry.to_string(),
                    template: draw.template.clone(),
                },
            })?;
        }
    } else {
        visit_sources(options, true, source::public_sources, |row| {
            let specialization = match &options.specializations {
                None => None,
                Some(listed) => match listed.of(row.key().as_ref()) {
                    Some(set) => Some(set.clone()),
                    // A row the file does not list is not this sweep's.
                    None => return Ok(()),
                },
            };
            let stage = row.stage.clone();
            let input = RowInput::from(row);
            attempt(NewRow {
                key: input.air_sha256.to_string(),
                entry: input.entry.to_string(),
                stage,
                what: RowWork::Function {
                    input,
                    specialization,
                },
            })
        })?;
    }
    if let Some(kernels) = kernels {
        if let Err(error) = kernels.finish() {
            eprintln!("corpus-pipeline: the kernel rows' runtime device: {error}");
            tally.errors += 1;
        }
    }
    let (slowest_key, slowest_ms) = tally
        .slowest
        .as_ref()
        .map_or(("none", 0), |(key, d)| (key.as_str(), d.as_millis()));
    let (slowest_compile_key, slowest_compile_ms) = tally
        .slowest_compile
        .as_ref()
        .map_or(("none", 0), |(key, d)| (key.as_str(), d.as_millis()));
    let (peak_rss_key, peak_rss) = tally
        .peak_compile_rss
        .as_ref()
        .map_or(("none", 0), |(key, rss)| (key.as_str(), *rss));
    let (peak_allocator_key, peak_allocator) = tally
        .peak_compile_allocator
        .as_ref()
        .map_or(("none", 0), |(key, bytes)| (key.as_str(), *bytes));
    let unmodelled_kinds = tally
        .unmodelled_kinds
        .iter()
        .map(|(kind, count)| format!("{}:{count}", kind.key()))
        .collect::<Vec<_>>()
        .join(",");
    let metal_failures = tally
        .metal_failures
        .iter()
        .map(|(class, count)| format!("{}:{count}", class.key()))
        .collect::<Vec<_>>()
        .join(",");
    let partition = options
        .partition
        .map_or_else(|| "none".to_string(), |part| part.to_string());
    let specializations = options
        .specializations
        .as_ref()
        .map_or_else(|| "none".to_string(), |listed| listed.0.len().to_string());
    let draws = options
        .draws
        .as_ref()
        .map_or_else(|| "none".to_string(), |listed| listed.0.len().to_string());
    eprintln!(
        "corpus-pipeline-summary\tengine=new\tprofile={profile}\tprofiles_layer={}\tprojection={projection:?}\tdevice_type={}\trows={}\tcompiled_only={}\tcreated={}\trefused={}\tnot_enabled={}\tnot_compiled={}\tunmodelled={}\tunmodelled_kinds={unmodelled_kinds}\truntime_refused={}\tno_effect={}\tstage_not_covered={}\tno_bitcode={}\ttooling={}\terrors={}\tcreation_over_ceiling={}\tinvalid_usage={}\tworker_stopped={}\tmetal_failures={metal_failures}\tpartition={partition}\tslowest_creation={slowest_key}\tslowest_creation_ms={slowest_ms}\tslowest_compile={slowest_compile_key}\tslowest_compile_ms={slowest_compile_ms}\tpeak_compile_rss={peak_rss_key}\tpeak_compile_rss_bytes={peak_rss}\tpeak_compile_allocator={peak_allocator_key}\tpeak_compile_allocator_bytes={peak_allocator}\tdriver_caches={}\tspecializations={specializations}\tdraws={draws}",
        options.profiles_layer.as_deref().unwrap_or("none"),
        device.map_or("none", DeviceType::as_str),
        tally.rows,
        tally.compiled,
        tally.created,
        tally.refused,
        tally.not_enabled,
        tally.not_compiled,
        tally.unmodelled,
        tally.runtime_refused,
        tally.no_effect,
        tally.stage_not_covered,
        tally.no_bitcode,
        tally.tooling,
        tally.errors,
        tally.over_ceiling,
        tally.invalid_usage,
        tally.worker_stopped,
        match options.driver_caches {
            true => "on",
            false => "off",
        },
    );
    tally.verdict()
}

/// Where the sweep writes its result rows: the process's standard output, which nothing else
/// writes to. Whatever else this process writes to its standard output descriptor goes to its
/// standard error instead: the validation layer names its log file there for each instance a kernel
/// row's runtime makes, which would otherwise land between two rows.
fn rows_output() -> Result<Box<dyn Write>, String> {
    #[cfg(unix)]
    {
        use std::os::fd::FromRawFd;
        // SAFETY: duplicates this process's own standard output descriptor, which stays open.
        let rows = unsafe { libc::dup(libc::STDOUT_FILENO) };
        if rows < 0 {
            return Err(format!(
                "duplicate the standard output: {}",
                std::io::Error::last_os_error()
            ));
        }
        // SAFETY: points the standard output descriptor at the standard error; both stay open for
        // the process's life, and nothing was written to the standard output yet.
        if unsafe { libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO) } < 0 {
            return Err(format!(
                "point the standard output at the standard error: {}",
                std::io::Error::last_os_error()
            ));
        }
        // SAFETY: `rows` is a descriptor this process just made and nothing else owns or closes.
        let rows = unsafe { std::fs::File::from_raw_fd(rows) };
        Ok(Box::new(std::io::LineWriter::new(rows)))
    }
    #[cfg(not(unix))]
    Ok(Box::new(std::io::stdout()))
}

/// Compile one row in a compile worker and create its pipeline (a kernel row's, made through the
/// runtime from the row's AIR on the sweep's one runtime device, `kernels`, opened at the first
/// kernel row): `(STATUS, detail, create_ms)`. With `keep`, the compiled row's stage modules are
/// written there first ([`keep_modules`]).
fn attempt_row(
    device: Option<DeviceType>,
    kernels: &mut Option<RuntimeKernels>,
    worker: &CompileWorker,
    compiler: &WorkerDevice,
    projection: Projection,
    row: &NewRow,
    keep: Option<&Path>,
    tally: &mut Tally,
) -> Result<(&'static str, String, u128), String> {
    let key = row.key.as_str();
    let constants = match &row.what {
        RowWork::Function {
            specialization: Some(set),
            ..
        } => set.values(),
        _ => FunctionConstantValues::defaults(),
    };
    let (compiled, measured) = match &row.what {
        RowWork::Function { input, .. } => {
            let Ok(stage) = RowStage::parse(&row.stage) else {
                tally.stage_not_covered += 1;
                return Ok(("STAGE_NOT_COVERED", format!("stage {}", row.stage), 0));
            };
            compile_row_in_worker(
                worker,
                compiler,
                &input.source,
                &input.entry,
                stage,
                projection,
                &constants,
            )
        }
        RowWork::Draw {
            libraries,
            vertex,
            fragment,
            template,
        } => compile_draw_in_worker(worker, compiler, libraries, vertex, fragment, template),
    };
    if let Some(measured) = measured {
        tally.measured_compile(key, measured);
    }
    let pipeline = match compiled {
        RowCompile::Compiled(pipeline) => pipeline,
        RowCompile::Stopped(outcome, detail) => {
            tally.worker_stopped += 1;
            return Ok((
                "WORKER_STOPPED",
                format!("{} {} {detail}", outcome.label(), outcome.key()),
                0,
            ));
        }
        // `llvm-as` rejected the text of a row that retains no bitcode: recorded, not a defect
        // of the driver or the tool.
        RowCompile::NotCompiled(
            Outcome::Failed {
                failure: FailureKind::NoBitcode,
            },
            detail,
        ) => {
            tally.no_bitcode += 1;
            return Ok(("NO_BITCODE", detail, 0));
        }
        // A corrupt index or missing tooling while preparing the row is the tool's failure.
        RowCompile::NotCompiled(
            Outcome::Failed {
                failure: FailureKind::Tooling,
            },
            detail,
        ) => {
            tally.errors += 1;
            return Ok(("ERROR", detail, 0));
        }
        RowCompile::NotCompiled(outcome, detail) => {
            tally.not_compiled += 1;
            return Ok((
                "NOT_COMPILED",
                format!("{} {} {detail}", outcome.label(), outcome.key()),
                0,
            ));
        }
        RowCompile::Unmodelled(why) => {
            tally.unmodelled += 1;
            *tally.unmodelled_kinds.entry(why.kind).or_default() += 1;
            return Ok(("UNMODELLED", why.to_string(), 0));
        }
    };
    if let Some(dir) = keep {
        keep_modules(dir, key, &pipeline)?;
    }
    let detail = RowDetail::of(&pipeline);
    let Some(device) = device else {
        tally.compiled += 1;
        return Ok(("COMPILED", detail.to_string(), 0));
    };
    let result = match (&*pipeline, &row.what) {
        (RowPipeline::Render { pipeline, .. }, _) => {
            metal2vulkan_validation::candidate::create_acceptance_pipeline(device, pipeline)
        }
        (RowPipeline::Compute { .. }, RowWork::Function { input, .. }) => {
            let bitcode = match input.source.bitcode() {
                Ok(bitcode) => bitcode,
                Err(error) => {
                    tally.errors += 1;
                    return Ok(("ERROR", format!("the row's bitcode: {error}"), 0));
                }
            };
            let kernel = RuntimeRow {
                bitcode: &bitcode,
                entry: &input.entry,
                constants: &constants,
            };
            let kernels = match kernels {
                Some(kernels) => kernels,
                None => match RuntimeKernels::open(device) {
                    Ok(opened) => kernels.insert(opened),
                    Err(error) => return tally.record(key, detail, Err(error)),
                },
            };
            kernels.accept(&kernel)
        }
        (RowPipeline::Compute { .. }, RowWork::Draw { .. }) => {
            return Err(format!(
                "{key}: a logged draw compiled to a compute pipeline"
            ));
        }
    };
    tally.record(key, detail, result)
}

/// Write row `key`'s stage modules into `dir` (`--keep-modules`): per stage, the SPIR-V as
/// `<key>.<stage>.spv`, and its entry point, specialization layout and strategy census as
/// `<key>.<stage>.json`.
fn keep_modules(dir: &Path, key: &str, pipeline: &RowPipeline) -> Result<(), String> {
    let write = |name: String, bytes: &[u8]| {
        let path = dir.join(name);
        std::fs::write(&path, bytes).map_err(|error| format!("write {}: {error}", path.display()))
    };
    std::fs::create_dir_all(dir).map_err(|error| format!("create {}: {error}", dir.display()))?;
    for stage in pipeline.stages() {
        let name = format!("{key}.{}", stage_name(stage.stage));
        let words = stage
            .spirv
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        write(format!("{name}.spv"), &words)?;
        let meta = serde_json::json!({
            "entry_point": stage.entry_point,
            "specialization": stage.specialization,
            "strategies": stage.strategies,
        });
        let meta = serde_json::to_vec_pretty(&meta)
            .map_err(|error| format!("encode {name}.json: {error}"))?;
        write(format!("{name}.json"), &meta)?;
    }
    Ok(())
}

/// A stage's file-name part, as the CLI names its modules (`out.vert.spv`).
fn stage_name(stage: metal2vulkan::vk::ShaderStage) -> &'static str {
    use metal2vulkan::vk::ShaderStage as S;
    match stage {
        S::VERTEX => "vert",
        S::FRAGMENT => "frag",
        S::COMPUTE => "comp",
        S::TESSELLATION_CONTROL => "tesc",
        S::TESSELLATION_EVALUATION => "tese",
        _ => "stage",
    }
}

/// Every entry of every owned engine `.ll` fixture (not the `.air.ll` companions, which need
/// `xcrun metal-as`), in file order then the library's entry order: each row is the fixture's
/// text, which the worker assembles.
fn fixture_rows() -> Result<Vec<NewRow>, String> {
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
    let dir = engine_dir();
    let mut paths = std::fs::read_dir(&dir)
        .map_err(|error| format!("read {}: {error}", dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read {}: {error}", dir.display()))?;
    paths.retain(|path| {
        path.extension().is_some_and(|e| e == "ll") && !path.to_string_lossy().ends_with(".air.ll")
    });
    paths.sort();
    let mut rows = Vec::new();
    for path in paths {
        let name = file_name(&path);
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        let bitcode = match llvm_assemble(&path) {
            Ok(bitcode) => bitcode,
            Err(AssembleError::Missing(why)) => return Err(format!("llvm-as: {why}")),
            Err(AssembleError::Rejected(why)) => return Err(format!("{name}: {why}")),
        };
        let library = metal2vulkan::library::Library::from_air(&bitcode)
            .map_err(|error| format!("{name}: {error}"))?;
        for info in library.functions() {
            rows.push(NewRow {
                key: name.clone(),
                entry: info.name.to_string(),
                stage: format!("{:?}", info.stage),
                what: RowWork::Function {
                    input: RowInput {
                        air_sha256: Cow::Owned(name.clone()),
                        entry: Cow::Owned(info.name.to_string()),
                        source: RowSource::Text(Cow::Owned(text.clone())),
                    },
                    specialization: None,
                },
            });
        }
    }
    Ok(rows)
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Translator errors chain with `; ` and driver errors do not embed newlines, but a panic message
/// forwarded as a string can. Keep every result on one TSV line.
fn one_line(error: &str) -> String {
    error.replace(['\n', '\r', '\t'], " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A kernel row's detail.
    const DETAIL: RowDetail = RowDetail {
        own: 7,
        companion: 0,
        precision: Vec::new(),
    };

    /// A row's detail prints its own stage's words, what the translation sweep counts for it, and
    /// the other stages' apart.
    #[test]
    fn a_rows_detail_is_its_own_stage_words_and_its_companions() {
        assert_eq!(DETAIL.to_string(), "words=7");
        let fragment = RowDetail {
            own: 1148,
            companion: 276,
            precision: Vec::new(),
        };
        assert_eq!(fragment.to_string(), "words=1148 companion_words=276");
        let mut tally = Tally::default();
        let created = || {
            Ok(Acceptance::Created {
                create: Duration::from_millis(3),
            })
        };
        let (_, detail, _) = tally
            .record("fragment", fragment.clone(), created())
            .unwrap();
        assert_eq!(detail, "words=1148 companion_words=276");
        // A pipeline whose stages carry precision notes names them.
        let noted = RowDetail {
            precision: vec![PrecisionNote::DeviceArithmetic],
            ..fragment
        };
        assert_eq!(
            noted.to_string(),
            "words=1148 companion_words=276 precision=[DeviceArithmetic]"
        );
    }

    /// `--specializations FILE`: each line lists a row and the set it compiles with, as the
    /// planner writes a set; a row's detail names the set by its kinds and identity; a line that
    /// does not parse, or a row listed twice, refuses the file.
    #[test]
    fn a_specializations_file_lists_each_row_once_with_its_set() {
        let set = r#"{"kinds":["zero"],"constants":{"2":{"ty":"Bool","bytes":[0]}}}"#;
        let text =
            format!("{{\"row\":\"aa\",\"set\":{set}}}\n\n{{\"row\":\"bb\",\"set\":{set}}}\n");
        let listed = Specializations::parse(&text).unwrap();
        assert_eq!(listed.0.len(), 2);
        let aa = listed.of("aa").unwrap();
        assert_eq!(aa.constants.len(), 1);
        assert_eq!(
            aa.values()
                .by_index
                .get(&2)
                .map(|value| value.bytes.as_ref()),
            Some(&[0u8][..])
        );
        assert!(listed.of("cc").is_none());
        let label = set_label(aa);
        assert_eq!(label, format!("zero:{}", &aa.id()[..16]));

        let twice = format!("{{\"row\":\"aa\",\"set\":{set}}}\n{{\"row\":\"aa\",\"set\":{set}}}");
        let error = Specializations::parse(&twice).err().unwrap();
        assert!(
            error.contains("line 2") && error.contains("twice"),
            "{error}"
        );
        let error = Specializations::parse("{\"row\":\"aa\"}").err().unwrap();
        assert!(error.starts_with("line 1"), "{error}");
    }

    /// `--draws FILE`: each line is a draw's id, its two sources and its template; an id listed
    /// twice, or a line that does not parse, refuses the file.
    #[test]
    fn a_draws_file_lists_each_draw_once_with_its_template() {
        let template = serde_json::to_string(&RenderTemplate::new(
            metal2vulkan::metal::PrimitiveTopologyClass::Triangle,
        ))
        .unwrap();
        let line = |id: &str| {
            format!(
                "{{\"id\":\"{id}\",\"vertex\":\"aa\",\"fragment\":\"bb\",\"template\":{template}}}"
            )
        };
        let draws = Draws::parse(&format!("{}\n\n{}\n", line("d1"), line("d2"))).unwrap();
        assert_eq!(draws.0.len(), 2);
        assert_eq!(
            draws.0[1].template,
            RenderTemplate::new(metal2vulkan::metal::PrimitiveTopologyClass::Triangle)
        );
        assert_eq!(
            draws.sources().into_iter().collect::<Vec<_>>(),
            ["aa", "bb"]
        );
        let error = Draws::parse(&format!("{}\n{}", line("d1"), line("d1")))
            .err()
            .unwrap();
        assert!(
            error.contains("line 2") && error.contains("twice"),
            "{error}"
        );
        assert!(Draws::parse("{\"id\":\"d1\"}")
            .err()
            .unwrap()
            .starts_with("line 1"));
    }

    /// The runtime's typed refusal of a kernel row and a kernel row it makes no pipeline for are
    /// counted and pass; the driver's refusal of the pipeline it asks for fails the sweep.
    #[test]
    fn a_runtime_refusal_and_a_kernel_without_effects_are_counted_and_pass() {
        let mut tally = Tally::default();
        let refusal = "unsupported: the state's imageblock length is not measured against Metal's";
        let (status, detail, _) = tally
            .record(
                "gap",
                DETAIL,
                Ok(Acceptance::RuntimeRefused(refusal.into())),
            )
            .unwrap();
        assert_eq!((status, detail.as_str()), ("RUNTIME_REFUSED", refusal));
        let (status, detail, _) = tally
            .record("still", DETAIL, Ok(Acceptance::NoEffect))
            .unwrap();
        assert_eq!((status, detail.as_str()), ("NO_EFFECT", "words=7"));
        assert_eq!((tally.runtime_refused, tally.no_effect), (1, 1));
        assert!(tally.verdict().is_ok());
        tally
            .record("refused", DETAIL, Ok(Acceptance::Refused("no".into())))
            .unwrap();
        assert!(tally.verdict().unwrap_err().contains("1 pipelines refused"));
    }

    #[test]
    fn an_unavailable_device_stops_the_sweep_and_other_tooling_errors_fail_it() {
        let mut tally = Tally::default();
        // Opening the context failed: the row is not counted, and the sweep stops.
        let stopped = tally.record(
            "row",
            DETAIL,
            Err(AcceptanceError::DeviceUnavailable("no CPU device".into())),
        );
        assert!(stopped.unwrap_err().contains("no CPU device"));
        assert_eq!(tally.tooling + tally.errors + tally.created, 0);

        // A modelled limit of the creation path is a row it does not build; the sweep passes.
        let (status, _, _) = tally
            .record(
                "divisor",
                DETAIL,
                Err(AcceptanceError::Unmodelled("divisor".into())),
            )
            .unwrap();
        assert_eq!(status, "TOOLING");
        let (status, _, _) = tally
            .record(
                "ok",
                DETAIL,
                Ok(Acceptance::Created {
                    create: Duration::from_millis(3),
                }),
            )
            .unwrap();
        assert_eq!(status, "OK");
        let (status, _, _) = tally
            .record(
                "ne",
                DETAIL,
                Ok(Acceptance::NotEnabled(vec!["ShaderInt64".into()])),
            )
            .unwrap();
        assert_eq!(status, "NOT_ENABLED");
        assert_eq!((tally.tooling, tally.created, tally.not_enabled), (1, 1, 1));
        assert!(tally.verdict().is_ok());

        // Any other tooling error fails the sweep.
        let (status, _, _) = tally
            .record(
                "inspect",
                DETAIL,
                Err(AcceptanceError::Tooling("inspect".into())),
            )
            .unwrap();
        assert_eq!(status, "ERROR");
        assert!(tally.verdict().unwrap_err().contains("1 tooling errors"));
    }

    #[test]
    fn a_compile_the_worker_boundary_stopped_fails_the_sweep_and_compiles_are_measured() {
        let mut tally = Tally::default();
        let measured = |ms, rss, allocator| Measured {
            elapsed: Duration::from_millis(ms),
            peak_resident_bytes: rss,
            allocator_peak_bytes: allocator,
        };
        tally.measured_compile("fast", measured(3, Some(10), Some(4)));
        tally.measured_compile("slow", measured(9, Some(5), Some(8)));
        tally.measured_compile("unsampled", measured(1, None, None));
        assert_eq!(
            tally.slowest_compile,
            Some(("slow".into(), Duration::from_millis(9)))
        );
        assert_eq!(tally.peak_compile_rss, Some(("fast".into(), 10)));
        // A peak the sampler missed is still counted, from the worker's own allocator.
        assert_eq!(tally.peak_compile_allocator, Some(("slow".into(), 8)));
        assert!(tally.verdict().is_ok());
        tally.worker_stopped += 1;
        assert!(tally
            .verdict()
            .unwrap_err()
            .contains("1 compiles stopped at the worker boundary"));
    }

    /// N parts split the corpus's shards disjointly and cover them all, and a part that could not
    /// hold a shard is refused.
    #[test]
    fn partitions_split_the_shards_disjointly_and_cover_them() {
        for count in [1, 6, 64] {
            let parts: Vec<Partition> = (0..count)
                .map(|index| Partition::parse(&format!("{index}/{count}")).unwrap())
                .collect();
            for shard in 0..SHARD_COUNT {
                let holders = parts.iter().filter(|part| part.holds_shard(shard)).count();
                assert_eq!(holders, 1, "shard {shard} in {holders} of {count} parts");
            }
        }
        assert_eq!(
            Partition::parse("2/6").unwrap(),
            Partition { index: 2, count: 6 }
        );
        assert_eq!(Partition::parse("2/6").unwrap().to_string(), "2/6");
        for invalid in ["6/6", "0/0", "1/65", "1", "a/6", "1/b", "-1/6"] {
            assert!(Partition::parse(invalid).is_err(), "{invalid} parsed");
        }
    }

    /// A hash of shard `shard`: the corpus buckets a hash by its first byte's top bits.
    fn hash_in_shard(shard: usize) -> String {
        format!(
            "{:02x}{}",
            shard << (8 - source::SHARD_BITS),
            "0".repeat(62)
        )
    }

    fn options(root: &Path, partition: Option<Partition>, after: Option<String>) -> Options {
        Options {
            root: root.to_path_buf(),
            limit: usize::MAX,
            after,
            only: None,
            partition,
            attempting: None,
            profiles_layer: None,
            driver_caches: false,
            keep_modules: None,
            specializations: None,
            draws: None,
        }
    }

    /// Over a synthetic corpus (rows in their hash's shard files, and public rows), each part
    /// visits exactly the rows of its shards, the parts together are the whole walk with no row
    /// twice, and `--after` resumes a part past one of its rows.
    #[test]
    fn a_partitioned_walk_visits_its_shards_and_public_rows_once() {
        let scratch = metal2vulkan_validation::ScratchDir::new("corpus-pipeline-partition")
            .expect("scratch directory");
        let rows: Vec<SourceRow> = (0..48)
            .map(|i| {
                let air_ll = format!("; synthetic source {i}\n");
                SourceRow {
                    air_sha256: metal2vulkan_validation::hash::sha256_bytes(air_ll.as_bytes()),
                    stage: "Kernel".into(),
                    entry: format!("k{i}"),
                    air_ll,
                    blob_b64: None,
                    lib_sha256s: Vec::new(),
                    label: format!("synthetic/{i}"),
                }
            })
            .collect();
        let (private, public) = rows.split_at(40);
        std::fs::create_dir_all(source::source_shards_dir(scratch.path())).unwrap();
        for row in private {
            let shard = shard_index_for_hash(&row.air_sha256).unwrap();
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(source_shard_path(scratch.path(), shard))
                .unwrap();
            writeln!(file, "{}", serde_json::to_string(row).unwrap()).unwrap();
        }
        let walk = |partition, after| -> Vec<String> {
            let mut seen = Vec::new();
            visit_sources(
                &options(scratch.path(), partition, after),
                false,
                || Ok(public.to_vec()),
                |row| {
                    seen.push(row.air_sha256);
                    Ok(())
                },
            )
            .unwrap();
            seen
        };
        let mut whole = walk(None, None);
        assert_eq!(whole.len(), rows.len());
        let mut joined = Vec::new();
        for index in 0..6 {
            let part = Partition { index, count: 6 };
            let seen = walk(Some(part), None);
            for hash in &seen {
                assert!(part.holds_shard(shard_index_for_hash(hash).unwrap()));
            }
            // A part with two rows or more resumes past its first.
            if seen.len() >= 2 {
                assert_eq!(walk(Some(part), Some(seen[0].clone())), seen[1..]);
            }
            joined.extend(seen);
        }
        whole.sort();
        joined.sort();
        assert_eq!(joined, whole, "the parts are the whole walk, each row once");
    }

    /// `--after` must name a row of the part, and a part takes no `--hash` or `--fixtures`.
    #[test]
    fn a_part_refuses_an_after_row_outside_it() {
        let root = Path::new("/nonexistent");
        let part = Some(Partition { index: 0, count: 6 });
        let outside = options(root, part, Some(hash_in_shard(1)));
        let error = check_partition(&outside, false).unwrap_err();
        assert!(error.contains("is not a row of --partition 0/6"), "{error}");
        assert!(check_partition(&options(root, part, Some(hash_in_shard(6))), false).is_ok());
        assert!(check_partition(&options(root, part, None), true).is_err());
        let mut hashed = options(root, part, None);
        hashed.only = Some(hash_in_shard(0));
        assert!(check_partition(&hashed, false).is_err());
        assert!(check_partition(&options(root, None, Some(hash_in_shard(1))), true).is_ok());
    }

    /// A refused or invalid-usage row carries MoltenVK's report, whose class the row prints and the
    /// summary counts; other rows and other drivers' refusals have none.
    #[test]
    fn a_moltenvk_refusal_is_classed_and_counted() {
        let mut tally = Tally::default();
        let crash = "error mvk-error: VK_ERROR_INITIALIZATION_FAILED: Compute pipeline compile \
                     failed (Error code 2): Compilation failed due to an interrupted connection: \
                     XPC_ERROR_CONNECTION_INTERRUPTED.";
        assert_eq!(
            tally.metal_failure("INVALID_USAGE", crash),
            "pipeline:service_crash"
        );
        assert_eq!(
            tally.metal_failure("REFUSED", crash),
            "pipeline:service_crash"
        );
        assert_eq!(tally.metal_failure("OK", crash), "-");
        assert_eq!(
            tally.metal_failure("REFUSED", "create pipeline: ERROR_UNKNOWN"),
            "-"
        );
        assert_eq!(
            tally.metal_failures.values().copied().collect::<Vec<_>>(),
            vec![2]
        );
        // The class is a label: the refusal fails the sweep whatever it is.
        tally
            .record("refused", DETAIL, Ok(Acceptance::Refused(crash.into())))
            .unwrap();
        assert!(tally.verdict().is_err());
    }

    #[test]
    fn refusals_and_slow_creations_fail_the_sweep() {
        let mut tally = Tally::default();
        tally
            .record("refused", DETAIL, Ok(Acceptance::Refused("no".into())))
            .unwrap();
        assert!(tally.verdict().is_err());

        let mut tally = Tally::default();
        let slow = PIPELINE_CREATION_CEILING + Duration::from_millis(1);
        let (status, _, _) = tally
            .record("slow", DETAIL, Ok(Acceptance::Created { create: slow }))
            .unwrap();
        assert_eq!(status, "OVER_CEILING");
        assert!(tally.verdict().is_err());

        // A message from the validation layer fails the row whatever the driver did.
        let mut tally = Tally::default();
        let (status, detail, _) = tally
            .record(
                "vuid",
                DETAIL,
                Ok(Acceptance::InvalidUsage(vec!["VUID-x".into()])),
            )
            .unwrap();
        assert_eq!((status, detail.as_str()), ("INVALID_USAGE", "VUID-x"));
        assert!(tally
            .verdict()
            .unwrap_err()
            .contains("1 reported by the validation layer"));
    }
}
