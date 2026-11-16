//! Harvest macOS `.metallib` files directly into JSONL shards.
//!
//! The only durable private artifacts are:
//!
//! - `validation/corpus/local/sources/shard_NNN.jsonl`
//! - `validation/corpus/local/library-modules/shard_NNN.jsonl`
//!
//! `--only-stages kernel,vertex,fragment,object,mesh` keeps only the modules whose entry lists
//! name one of those stages. A walk under it skips, before any `llvm-dis`, every metallib whose
//! function table lists no function of those stages, and disassembles only the listed functions
//! of the rest. `--from-archive DIR` reads the shards of an earlier harvest's corpus root `DIR` (a
//! separate archive, never synced or indexed) instead of walking metallibs, and merges the
//! source rows a selection keeps into `--out`: a later selection needs no second walk. The
//! selection is the stage filter, `--rows FILE` (the source rows whose `air_sha256` the file
//! lists), or both. Each kept row brings the archive's library modules of its parent libraries,
//! the dependencies function-table linking may resolve it against.

use base64::Engine as _;
use metal2vulkan_validation::air::{listed_stages, stage_entry_from_ll};
use metal2vulkan_validation::case::Stage;
use metal2vulkan_validation::hash::sha256_bytes;
use metal2vulkan_validation::library_module::{self, LibraryModuleRow};
use metal2vulkan_validation::source::{self, SourceRow};
use metal2vulkan_validation::ScratchDir;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use wait_timeout::ChildExt as _;

const PROGRAM: &str = "corpus-harvest";
const DEFAULT_MAX_AIR_BYTES: usize = 1024 * 1024;
const DEFAULT_LLVM_DIS_TIMEOUT: Duration = Duration::from_secs(60);
const AIR_WRAP: &[u8; 4] = b"\xde\xc0\x17\x0b";
/// The text and blob bytes of new rows a walk holds before it merges them into the shards, which
/// bounds its memory however many metallibs it reads.
const FLUSH_HELD_BYTES: usize = 512 << 20;

#[derive(Debug)]
struct Options {
    out: PathBuf,
    limit: Option<usize>,
    offset: usize,
    start_set: StartSet,
    include_apps: bool,
    metallibs: Vec<PathBuf>,
    llvm_dis: Option<PathBuf>,
    max_air_bytes: usize,
    llvm_dis_timeout: Duration,
    /// `--only-stages`: keep only modules listing an entry of these stages.
    only_stages: Option<StageFilter>,
    /// `--from-archive`: select from this corpus root's shards instead of walking metallibs.
    from_archive: Option<PathBuf>,
    /// `--rows`: with `--from-archive`, keep only the source rows this file lists.
    rows: Option<PathBuf>,
    /// `--jobs`: metallibs examined (read, carved, disassembled) at once.
    jobs: usize,
}

/// The stages a harvest keeps (`--only-stages`), by their entry-list names (`object`, `mesh`).
#[derive(Clone, Debug, PartialEq, Eq)]
struct StageFilter(BTreeSet<Stage>);

impl StageFilter {
    /// The filter a comma-separated list of entry-list names (`object,mesh`) names.
    fn parse(list: &str) -> Result<Self, String> {
        let mut stages = BTreeSet::new();
        for name in list.split(',').map(str::trim) {
            let stage = Stage::ALL
                .into_iter()
                .find(|stage| stage.entry_list() == name)
                .ok_or_else(|| {
                    format!(
                        "--only-stages: {name:?} is none of kernel, vertex, fragment, object, mesh"
                    )
                })?;
            stages.insert(stage);
        }
        Ok(Self(stages))
    }

    fn admits(&self, stage: Stage) -> bool {
        self.0.contains(&stage)
    }

    /// Whether the module `ll` lists an entry of a kept stage. A library module lists none.
    fn admits_module(&self, ll: &str) -> bool {
        listed_stages(ll)
            .into_iter()
            .any(|stage| self.admits(stage))
    }

    fn names(&self) -> String {
        self.0
            .iter()
            .map(|stage| stage.entry_list())
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// The stage of a metallib function-table `TYPE`, when it is one.
fn container_stage(stage: metal2vulkan_air::container::FunctionStage) -> Option<Stage> {
    use metal2vulkan_air::container::FunctionStage as F;
    match stage {
        F::Vertex => Some(Stage::Vertex),
        F::Fragment => Some(Stage::Fragment),
        F::Kernel => Some(Stage::Kernel),
        F::Object => Some(Stage::Object),
        F::Mesh => Some(Stage::Mesh),
        _ => None,
    }
}

/// What a library's function table says under a stage filter.
#[derive(Debug, PartialEq, Eq)]
enum Container {
    /// No stage filter: every carved module is disassembled.
    Unfiltered,
    /// The function table lists no function of a kept stage: nothing is carved or disassembled.
    Rejected,
    /// The byte offsets of the functions of a kept stage: only their modules are disassembled.
    Selected(BTreeSet<usize>),
    /// The library has no function table the container parser reads (why): every carved module
    /// is disassembled, then kept by its entry lists.
    Unread(String),
}

/// The function table's verdict on `lib_bytes` under `filter`.
fn container_selection(filter: &StageFilter, lib_bytes: &[u8]) -> Container {
    use metal2vulkan_air::container::{parse, Library};
    let base = lib_bytes.as_ptr() as usize;
    match parse(lib_bytes) {
        Ok(Library::Metallib(metallib)) => {
            let selected: BTreeSet<usize> = metallib
                .functions
                .iter()
                .filter(|f| container_stage(f.stage).is_some_and(|stage| filter.admits(stage)))
                .filter_map(|f| (f.bitcode.as_ptr() as usize).checked_sub(base))
                .collect();
            if selected.is_empty() {
                Container::Rejected
            } else {
                Container::Selected(selected)
            }
        }
        Ok(Library::Module(_)) => Container::Unread("a bitcode module, no function table".into()),
        Err(error) => Container::Unread(error.to_string()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StartSet {
    System,
    Apps,
    All,
}

impl StartSet {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "system" => Some(Self::System),
            "apps" => Some(Self::Apps),
            "all" => Some(Self::All),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::Apps => "apps",
            Self::All => "all",
        }
    }
}

#[derive(Debug)]
struct AirBlob {
    offset: usize,
    bytes: Vec<u8>,
}

#[derive(Default)]
struct HarvestStats {
    libs_ok: usize,
    libs_failed: usize,
    /// Libraries whose function table lists no function of a kept stage (never disassembled).
    libs_rejected: usize,
    /// Libraries under a stage filter whose function table the container parser did not read.
    container_unread: usize,
    /// Functions of a kept stage no carved module (within the size limit) starts at.
    container_unmatched: usize,
    /// Disassembled modules listing no entry of a kept stage.
    filtered: usize,
    blobs_carved: usize,
    rows_kept: usize,
    library_modules_kept: usize,
    skipped_large: usize,
    dropped_nonfunctions: usize,
    /// Modules refused as source rows ([`Refusal`]), each named on its own report line.
    refused: usize,
    llvm_failed: usize,
    duplicates: usize,
}

/// Why a carved module is no private source row. Each is counted and named in the report: a gap
/// handed back, never a silent skip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Refusal {
    /// The module declares several entry points. A stored source row is one module with one
    /// entry; a row per entry exists only for public fixtures, which are read from the checkout, so
    /// harvesting one is a hand-back (no private module has one today).
    SeveralEntries { entries: usize },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SeveralEntries { entries } => write!(
                f,
                "several_entries entries={entries}: a private source row is one module with one \
                 entry"
            ),
        }
    }
}

/// Why the module `ll` is refused as a source row, if it is.
fn refusal(ll: &str) -> Option<Refusal> {
    match metal2vulkan_validation::air::entry_count_from_ll(ll) {
        entries @ 2.. => Some(Refusal::SeveralEntries { entries }),
        _ => None,
    }
}

fn main() {
    let Some(opts) = parse_args() else {
        return;
    };
    let code = run(opts);
    std::process::exit(code);
}

fn parse_args() -> Option<Options> {
    let manifest_dir = metal2vulkan_devtools::validation_dir();
    let env_limit = std::env::var("METAL2VULKAN_HARVEST_LIMIT").ok();
    let env_offset = std::env::var("METAL2VULKAN_HARVEST_OFFSET").ok();
    let env_set = std::env::var("METAL2VULKAN_HARVEST_START_SET").ok();
    let env_max = std::env::var("METAL2VULKAN_HARVEST_MAX_AIR_BYTES").ok();

    let mut opts = Options {
        out: default_output_dir(&manifest_dir),
        limit: env_limit
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(parse_usize_env)
            .transpose()
            .unwrap_or_else(|e| fatal(&e)),
        offset: env_offset
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(parse_usize_env)
            .transpose()
            .unwrap_or_else(|e| fatal(&e))
            .unwrap_or(0),
        start_set: env_set
            .as_deref()
            .and_then(StartSet::parse)
            .unwrap_or(StartSet::System),
        include_apps: false,
        metallibs: Vec::new(),
        llvm_dis: None,
        max_air_bytes: env_max
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(parse_usize_env)
            .transpose()
            .unwrap_or_else(|e| fatal(&e))
            .unwrap_or(DEFAULT_MAX_AIR_BYTES),
        llvm_dis_timeout: DEFAULT_LLVM_DIS_TIMEOUT,
        only_stages: None,
        from_archive: None,
        rows: None,
        jobs: std::thread::available_parallelism().map_or(1, usize::from),
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return None;
            }
            "--out" => {
                opts.out =
                    PathBuf::from(args.next().unwrap_or_else(|| fatal("--out requires path")))
            }
            "--limit" => {
                let n = args.next().unwrap_or_else(|| fatal("--limit requires N"));
                opts.limit = Some(parse_usize_arg("--limit", &n));
            }
            "--offset" => {
                let n = args.next().unwrap_or_else(|| fatal("--offset requires N"));
                opts.offset = parse_usize_arg("--offset", &n);
            }
            "--start-set" => {
                let set = args
                    .next()
                    .unwrap_or_else(|| fatal("--start-set requires system|apps|all"));
                opts.start_set = StartSet::parse(&set)
                    .unwrap_or_else(|| fatal(&format!("bad --start-set {set:?}")));
            }
            "--include-apps" => opts.include_apps = true,
            "--metallib" => {
                opts.metallibs.push(PathBuf::from(
                    args.next()
                        .unwrap_or_else(|| fatal("--metallib requires path")),
                ));
            }
            "--llvm-dis" => {
                opts.llvm_dis = Some(PathBuf::from(
                    args.next()
                        .unwrap_or_else(|| fatal("--llvm-dis requires path")),
                ));
            }
            "--max-air-bytes" => {
                let n = args
                    .next()
                    .unwrap_or_else(|| fatal("--max-air-bytes requires N"));
                opts.max_air_bytes = parse_usize_arg("--max-air-bytes", &n);
            }
            "--llvm-dis-timeout-secs" => {
                let seconds = args
                    .next()
                    .unwrap_or_else(|| fatal("--llvm-dis-timeout-secs requires N"));
                opts.llvm_dis_timeout = parse_timeout_arg(&seconds);
            }
            "--only-stages" => {
                let list = args
                    .next()
                    .unwrap_or_else(|| fatal("--only-stages requires a stage list"));
                opts.only_stages = Some(StageFilter::parse(&list).unwrap_or_else(|e| fatal(&e)));
            }
            "--from-archive" => {
                opts.from_archive = Some(PathBuf::from(
                    args.next()
                        .unwrap_or_else(|| fatal("--from-archive requires DIR")),
                ));
            }
            "--rows" => {
                opts.rows = Some(PathBuf::from(
                    args.next().unwrap_or_else(|| fatal("--rows requires FILE")),
                ));
            }
            "--jobs" => {
                let n = args.next().unwrap_or_else(|| fatal("--jobs requires N"));
                opts.jobs = parse_jobs_arg(&n);
            }
            other if other.starts_with("--out=") => {
                opts.out = PathBuf::from(other.trim_start_matches("--out="));
            }
            other if other.starts_with("--limit=") => {
                let n = other.trim_start_matches("--limit=");
                opts.limit = Some(parse_usize_arg("--limit", n));
            }
            other if other.starts_with("--offset=") => {
                let n = other.trim_start_matches("--offset=");
                opts.offset = parse_usize_arg("--offset", n);
            }
            other if other.starts_with("--start-set=") => {
                let set = other.trim_start_matches("--start-set=");
                opts.start_set = StartSet::parse(set)
                    .unwrap_or_else(|| fatal(&format!("bad --start-set {set:?}")));
            }
            other if other.starts_with("--metallib=") => {
                opts.metallibs
                    .push(PathBuf::from(other.trim_start_matches("--metallib=")));
            }
            other if other.starts_with("--llvm-dis=") => {
                opts.llvm_dis = Some(PathBuf::from(other.trim_start_matches("--llvm-dis=")));
            }
            other if other.starts_with("--max-air-bytes=") => {
                let n = other.trim_start_matches("--max-air-bytes=");
                opts.max_air_bytes = parse_usize_arg("--max-air-bytes", n);
            }
            other if other.starts_with("--llvm-dis-timeout-secs=") => {
                opts.llvm_dis_timeout =
                    parse_timeout_arg(other.trim_start_matches("--llvm-dis-timeout-secs="));
            }
            other if other.starts_with("--only-stages=") => {
                let list = other.trim_start_matches("--only-stages=");
                opts.only_stages = Some(StageFilter::parse(list).unwrap_or_else(|e| fatal(&e)));
            }
            other if other.starts_with("--from-archive=") => {
                opts.from_archive =
                    Some(PathBuf::from(other.trim_start_matches("--from-archive=")));
            }
            other if other.starts_with("--rows=") => {
                opts.rows = Some(PathBuf::from(other.trim_start_matches("--rows=")));
            }
            other if other.starts_with("--jobs=") => {
                opts.jobs = parse_jobs_arg(other.trim_start_matches("--jobs="));
            }
            other => fatal(&format!("unknown arg: {other}")),
        }
    }

    Some(opts)
}

fn default_output_dir(manifest_dir: &Path) -> PathBuf {
    manifest_dir.join(source::DEFAULT_CORPUS_REL)
}

fn print_usage() {
    eprintln!(
        "usage: {PROGRAM} [--out DIR] [--limit N] [--offset N]\n\
                \t\t[--start-set system|apps|all] [--include-apps]\n\
                \t\t[--metallib PATH ...] [--llvm-dis PATH]\n\
                \t\t[--max-air-bytes N] [--llvm-dis-timeout-secs N]\n\
                \t\t[--only-stages LIST] [--jobs N]\n\
         \x20      {PROGRAM} --from-archive ARCHIVE [--only-stages LIST] [--rows FILE] [--out DIR]\n\
         \n\
         DIR is the corpus root. Harvests metallib AIR directly into\n\
         DIR/local/sources/shard_NNN.jsonl.\n\
         LIST is a comma-separated list of kernel, vertex, fragment, object,\n\
         mesh: only modules listing an entry of one of them are kept, and a\n\
         metallib whose function table lists none is not disassembled.\n\
         --from-archive selects from ARCHIVE, the corpus root of an earlier\n\
         harvest, instead of walking metallibs: the source rows of the stages\n\
         LIST names, the rows FILE lists (the first field of each line, a\n\
         lowercase SHA-256), or the rows both admit, each with the library\n\
         modules of its parent libraries. --jobs (default: the logical\n\
         CPUs) is how many metallibs are examined at once.\n\
         No local/air, local/metallib, local/ledger, or local/tmp\n\
         intermediates are retained. Environment defaults: METAL2VULKAN_HARVEST_LIMIT,\n\
         METAL2VULKAN_HARVEST_OFFSET, METAL2VULKAN_HARVEST_START_SET,\n\
         METAL2VULKAN_HARVEST_MAX_AIR_BYTES, METAL2VULKAN_LLVM_DIS."
    );
}

fn parse_usize_env(s: &str) -> Result<usize, String> {
    s.parse::<usize>()
        .map_err(|e| format!("bad harvest env integer {s:?}: {e}"))
}

fn parse_usize_arg(flag: &str, s: &str) -> usize {
    s.parse::<usize>()
        .unwrap_or_else(|e| fatal(&format!("bad {flag} {s:?}: {e}")))
}

fn parse_jobs_arg(value: &str) -> usize {
    match parse_usize_arg("--jobs", value) {
        0 => fatal("--jobs must be greater than zero"),
        n => n,
    }
}

fn parse_timeout_arg(value: &str) -> Duration {
    let seconds = parse_usize_arg("--llvm-dis-timeout-secs", value);
    if seconds == 0 {
        fatal("--llvm-dis-timeout-secs must be greater than zero");
    }
    Duration::from_secs(seconds as u64)
}

fn fatal(msg: &str) -> ! {
    eprintln!("{PROGRAM}: {msg}");
    std::process::exit(64);
}

fn run(opts: Options) -> i32 {
    let t0 = Instant::now();
    eprintln!("# {PROGRAM}");
    eprintln!("# out       {}", opts.out.display());
    if let Some(filter) = &opts.only_stages {
        eprintln!("# stages    {}", filter.names());
    }
    if let Some(archive) = &opts.from_archive {
        return match cherry_pick(&opts, archive) {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("{PROGRAM}: {error}");
                1
            }
        };
    }
    eprintln!("# max AIR   {} bytes", opts.max_air_bytes);
    eprintln!("# jobs      {}", opts.jobs);

    if cfg!(not(target_os = "macos")) && opts.metallibs.is_empty() {
        eprintln!("{PROGRAM}: system metallib enumeration requires macOS; pass --metallib PATH");
        return 1;
    }

    let llvm_dis = match resolve_llvm_dis(opts.llvm_dis.as_deref()) {
        Ok(path) => path,
        Err(e) => {
            eprintln!("{PROGRAM}: {e}");
            return 1;
        }
    };
    eprintln!("# llvm-dis  {}", llvm_dis.display());

    let (batch, total_found) = match select_metallibs(&opts) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{PROGRAM}: {e}");
            return 1;
        }
    };
    eprintln!(
        "# start-set={} total_found={} batch={} offset={} limit={}",
        opts.start_set.as_str(),
        total_found,
        batch.len(),
        opts.offset,
        opts.limit
            .map(|n| n.to_string())
            .unwrap_or_else(|| "none".into())
    );

    cleanup_legacy_intermediates(&opts.out);
    let _cleanup_intermediates_on_exit = HarvestIntermediateCleanup::new(opts.out.clone());

    let indexed_source_memberships = match source::indexed_source_memberships(&opts.out) {
        Ok(memberships) => memberships,
        Err(error) => {
            eprintln!("{PROGRAM}: load source index: {error}");
            return 1;
        }
    };
    let mut stats = HarvestStats::default();
    let mut merged = Merged::default();
    let mut indexed_entry_duplicates = 0usize;
    // Workers examine metallibs (read, carve, disassemble); this thread absorbs each outcome and
    // merges the held rows into the shards whenever they pass FLUSH_HELD_BYTES. The channel holds
    // at most one outcome per worker, so a slow merge stalls the workers instead of growing.
    let next = AtomicUsize::new(0);
    let walked = std::thread::scope(|scope| -> Result<Held, String> {
        let mut held = Held::default();
        let (tx, rx) =
            std::sync::mpsc::sync_channel::<(PathBuf, Result<Examined, String>)>(opts.jobs.max(1));
        for _ in 0..opts.jobs.max(1) {
            let tx = tx.clone();
            let (next, batch, llvm_dis, opts) = (&next, &batch, &llvm_dis, &opts);
            scope.spawn(move || {
                while let Some(lib) = batch.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let examined = examine_lib(lib, llvm_dis, opts);
                    if tx.send((lib.clone(), examined)).is_err() {
                        return;
                    }
                }
            });
        }
        drop(tx);
        for (lib, examined) in rx {
            absorb(&lib, examined, &mut held, &mut stats);
            // Dependency modules are deliberately retained independently. A stage-entry row can
            // be discarded only when SQLite already knows every parent-library membership
            // observed in this batch. Apply that filter after each library so a broad reharvest
            // cannot accumulate gigabytes of already-complete entries.
            let before_filter = held.sources.len();
            held.sources.retain(|hash, row| {
                !source_memberships_already_indexed(&indexed_source_memberships, hash, row)
            });
            indexed_entry_duplicates += before_filter - held.sources.len();
            if held.bytes > FLUSH_HELD_BYTES {
                merged.add(flush(&opts.out, held)?);
                held = Held::default();
            }
        }
        Ok(held)
    });
    let flushed = walked.and_then(|held| flush(&opts.out, held));
    match flushed {
        Ok(last) => merged.add(last),
        Err(error) => {
            eprintln!("{PROGRAM}: {error}");
            return 1;
        }
    }

    if merged.is_empty() {
        eprintln!(
            "# RESULT: no new AIR modules (libs_ok={} libs_failed={} libs_rejected={} carved={} filtered={} llvm_failed={})",
            stats.libs_ok,
            stats.libs_failed,
            stats.libs_rejected,
            stats.blobs_carved,
            stats.filtered,
            stats.llvm_failed
        );
        return if stats.libs_failed == 0 { 0 } else { 1 };
    }
    let (merge, library_merge) = (merged.sources, merged.library_modules);
    eprintln!(
        "# RESULT: libs_ok={} libs_failed={} libs_rejected={} container_unread={} container_unmatched={} carved={} entry_batch_unique={} entry_inserted={} entry_memberships_added={} entry_replaced={} entry_dup={} entry_shards={} library_module_batch_unique={} library_module_inserted={} library_memberships_added={} library_module_dup={} library_module_shards={} large={} dropped_nonfunctions={} filtered={} refused={} llvm_failed={} ({:.1}s)",
        stats.libs_ok,
        stats.libs_failed,
        stats.libs_rejected,
        stats.container_unread,
        stats.container_unmatched,
        stats.blobs_carved,
        stats.rows_kept,
        merge.inserted,
        merge.merged_memberships,
        merge.replaced,
        stats.duplicates + merge.duplicates + indexed_entry_duplicates,
        merge.affected_shards,
        stats.library_modules_kept,
        library_merge.inserted,
        library_merge.merged_memberships,
        library_merge.duplicates,
        library_merge.affected_shards,
        stats.skipped_large,
        stats.dropped_nonfunctions,
        stats.filtered,
        stats.refused,
        stats.llvm_failed,
        t0.elapsed().as_secs_f64()
    );
    if stats.libs_failed > 0 && stats.libs_ok == 0 {
        1
    } else {
        0
    }
}

/// The new rows a walk holds until it merges them into the shards ([`flush`]).
#[derive(Default)]
struct Held {
    sources: HashMap<String, SourceRow>,
    library_modules: HashMap<String, LibraryModuleRow>,
    /// The text and blob bytes of the rows held.
    bytes: usize,
}

/// The merge statistics of every flush of a walk, summed.
#[derive(Default)]
struct Merged {
    sources: source::SourceMergeStats,
    library_modules: library_module::LibraryModuleMergeStats,
}

impl Merged {
    fn add(
        &mut self,
        (sources, modules): (
            source::SourceMergeStats,
            library_module::LibraryModuleMergeStats,
        ),
    ) {
        let s = &mut self.sources;
        s.affected_shards += sources.affected_shards;
        s.inserted += sources.inserted;
        s.merged_memberships += sources.merged_memberships;
        s.replaced += sources.replaced;
        s.duplicates += sources.duplicates;
        let m = &mut self.library_modules;
        m.affected_shards += modules.affected_shards;
        m.inserted += modules.inserted;
        m.merged_memberships += modules.merged_memberships;
        m.duplicates += modules.duplicates;
    }

    fn is_empty(&self) -> bool {
        self.sources == source::SourceMergeStats::default()
            && self.library_modules == library_module::LibraryModuleMergeStats::default()
    }
}

/// Merge the held rows into `out`'s shards; only the hash-aligned shards receiving rows are
/// rewritten.
fn flush(
    out: &Path,
    held: Held,
) -> Result<
    (
        source::SourceMergeStats,
        library_module::LibraryModuleMergeStats,
    ),
    String,
> {
    let sources = if held.sources.is_empty() {
        source::SourceMergeStats::default()
    } else {
        source::merge_source_shards(out, held.sources.into_values())
            .map_err(|error| format!("merge source shards: {error}"))?
    };
    let modules = if held.library_modules.is_empty() {
        library_module::LibraryModuleMergeStats::default()
    } else {
        library_module::merge_library_module_shards(out, held.library_modules.into_values())
            .map_err(|error| format!("merge library-module shards: {error}"))?
    };
    Ok((sources, modules))
}

/// What a `--from-archive` selection keeps: the source rows the stage filter admits, the rows the
/// `--rows` file lists, or (both given) the rows both admit.
struct Selection<'a> {
    stages: Option<&'a StageFilter>,
    rows: Option<BTreeSet<String>>,
}

impl Selection<'_> {
    fn admits(&self, row: &SourceRow) -> bool {
        self.stages
            .is_none_or(|filter| filter.admits_module(&row.air_ll))
            && self
                .rows
                .as_ref()
                .is_none_or(|rows| rows.contains(&row.air_sha256))
    }
}

/// The rows a `--rows` file lists: the first whitespace-separated field of each non-empty line, a
/// lowercase SHA-256 (the contract of `corpus-triage --hash-file`). A private source row is one
/// module with one entry, so its `air_sha256` is its row identity.
fn read_rows_file(path: &Path) -> Result<BTreeSet<String>, String> {
    let text =
        fs::read_to_string(path).map_err(|error| format!("--rows {}: {error}", path.display()))?;
    let mut rows = BTreeSet::new();
    for (index, line) in text.lines().enumerate() {
        let Some(hash) = line.split_whitespace().next() else {
            continue;
        };
        if hash.len() != 64 || !hash.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
            return Err(format!(
                "--rows {}:{}: the first field must be a lowercase SHA-256",
                path.display(),
                index + 1
            ));
        }
        rows.insert(hash.to_string());
    }
    if rows.is_empty() {
        return Err(format!("--rows {}: lists no row", path.display()));
    }
    Ok(rows)
}

/// Select from the archive `archive` (the corpus root of an earlier harvest) the source rows the
/// selection (`opts.only_stages`, `opts.rows`) keeps, with the archive's library modules of their
/// parent libraries, and merge them into `opts.out`. A library module lists no entry, so no stage
/// filter admits one; it comes with a kept row whose parent library it shares, because
/// function-table linking resolves a row's dependencies across its parent libraries. The
/// archive's shards are streamed one row at a time; only the kept rows are held. A `--rows` file
/// naming a row the archive lacks is an error, and nothing is merged.
fn cherry_pick(opts: &Options, archive: &Path) -> Result<(), String> {
    let t0 = Instant::now();
    let selection = Selection {
        stages: opts.only_stages.as_ref(),
        rows: opts.rows.as_deref().map(read_rows_file).transpose()?,
    };
    if selection.stages.is_none() && selection.rows.is_none() {
        return Err(
            "--from-archive needs --only-stages or --rows: a selection, never a copy of the archive"
                .into(),
        );
    }
    let same = |a: &Path, b: &Path| match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if same(archive, &opts.out) {
        return Err(format!("--from-archive {} is --out", archive.display()));
    }
    if !source::source_shards_dir(archive).is_dir() {
        return Err(format!(
            "--from-archive {}: no local/sources shards",
            archive.display()
        ));
    }
    eprintln!("# archive   {}", archive.display());
    let (mut read_sources, mut read_modules) = (0usize, 0usize);
    let mut selected = Held::default();
    let mut per_stage = BTreeMap::<String, (usize, usize)>::new();
    let mut parents = HashSet::<String>::new();
    for shard in 0..source::SHARD_COUNT {
        let path = source::source_shard_path(archive, shard);
        if path.is_file() {
            source::for_each_source_shard(&path, |row| {
                read_sources += 1;
                if selection.admits(&row) {
                    let bytes = row.air_ll.len() + row.blob_b64.as_ref().map_or(0, String::len);
                    let counted = per_stage.entry(row.stage.clone()).or_default();
                    counted.0 += 1;
                    counted.1 += bytes;
                    selected.bytes += bytes;
                    parents.extend(row.lib_sha256s.iter().cloned());
                    selected.sources.insert(row.air_sha256.clone(), row);
                }
                Ok(())
            })?;
        }
    }
    if let Some(rows) = &selection.rows {
        let missing: Vec<&String> = rows
            .iter()
            .filter(|row| !selected.sources.contains_key(*row))
            .collect();
        if let Some(first) = missing.first() {
            return Err(format!(
                "--rows: {} listed row(s) are no source row of the archive{}, the first {first}",
                missing.len(),
                if selection.stages.is_some() {
                    " the stage filter admits"
                } else {
                    ""
                }
            ));
        }
    }
    for shard in 0..source::SHARD_COUNT {
        let path = library_module::library_module_shard_path(archive, shard);
        if path.is_file() {
            library_module::for_each_library_module_shard(&path, |row| {
                read_modules += 1;
                if row.lib_sha256s.iter().any(|lib| parents.contains(lib)) {
                    selected.bytes += row.air_ll.len() + row.blob_b64.len();
                    selected
                        .library_modules
                        .insert(row.module_sha256.clone(), row);
                }
                Ok(())
            })?;
        }
    }
    for (stage, (rows, bytes)) in &per_stage {
        eprintln!("# SELECTED stage={stage} rows={rows} bytes={bytes}");
    }
    eprintln!(
        "# SELECTED sources={} of {read_sources} library_modules={} of {read_modules} parent_libraries={} bytes={}",
        selected.sources.len(),
        selected.library_modules.len(),
        parents.len(),
        selected.bytes
    );
    let indexed = source::indexed_source_memberships(&opts.out)
        .map_err(|error| format!("load source index: {error}"))?;
    let before = selected.sources.len();
    selected
        .sources
        .retain(|hash, row| !source_memberships_already_indexed(&indexed, hash, row));
    let indexed_duplicates = before - selected.sources.len();
    let (merge, library_merge) = flush(&opts.out, selected)?;
    eprintln!(
        "# RESULT: entry_inserted={} entry_memberships_added={} entry_replaced={} entry_dup={} entry_shards={} library_module_inserted={} library_memberships_added={} library_module_dup={} library_module_shards={} ({:.1}s)",
        merge.inserted,
        merge.merged_memberships,
        merge.replaced,
        merge.duplicates + indexed_duplicates,
        merge.affected_shards,
        library_merge.inserted,
        library_merge.merged_memberships,
        library_merge.duplicates,
        library_merge.affected_shards,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn source_memberships_already_indexed(
    indexed: &HashMap<String, HashSet<String>>,
    hash: &str,
    row: &SourceRow,
) -> bool {
    indexed.get(hash).is_some_and(|memberships| {
        row.lib_sha256s
            .iter()
            .all(|library| memberships.contains(library))
    })
}

fn select_metallibs(opts: &Options) -> Result<(Vec<PathBuf>, usize), String> {
    if !opts.metallibs.is_empty() {
        for path in &opts.metallibs {
            if !path.is_file() {
                return Err(format!("not a file: {}", path.display()));
            }
        }
        return Ok((opts.metallibs.clone(), opts.metallibs.len()));
    }

    let mut roots = Vec::new();
    match opts.start_set {
        StartSet::System | StartSet::All => {
            roots.push(PathBuf::from("/System/Library"));
            roots.push(PathBuf::from("/Library"));
            if opts.include_apps && opts.start_set == StartSet::System {
                roots.push(PathBuf::from("/Applications"));
            }
        }
        StartSet::Apps => {}
    }
    match opts.start_set {
        StartSet::Apps | StartSet::All => roots.push(PathBuf::from("/Applications")),
        StartSet::System => {}
    }

    let mut found = Vec::new();
    for root in roots {
        scan_metallibs(&root, &mut found);
    }
    let mut unique = HashMap::<String, PathBuf>::new();
    for path in found {
        let key = path
            .canonicalize()
            .unwrap_or_else(|_| path.clone())
            .display()
            .to_string();
        unique.insert(key, path);
    }
    let mut all: Vec<_> = unique.into_values().collect();
    all.sort_by_key(|a| priority_key(a));
    let total = all.len();
    let start = opts.offset.min(total);
    let end = opts
        .limit
        .map(|limit| start.saturating_add(limit).min(total))
        .unwrap_or(total);
    Ok((all[start..end].to_vec(), total))
}

fn scan_metallibs(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for ent in entries.flatten() {
        let path = ent.path();
        let Ok(ft) = ent.file_type() else {
            continue;
        };
        if ft.is_dir() {
            scan_metallibs(&path, out);
        } else if ft.is_file()
            && path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e == "metallib")
        {
            out.push(path);
        }
    }
}

fn priority_key(path: &Path) -> (String, String) {
    let s = path.display().to_string();
    let band = if [
        "WindowServer",
        "SkyLight",
        "CoreDisplay",
        "CoreAnimation",
        "QuartzCore",
    ]
    .iter()
    .any(|needle| s.contains(needle))
    {
        "00"
    } else if ["AppleMetalOpenGLRenderer", "PixelConverter", "OpenGL"]
        .iter()
        .any(|needle| s.contains(needle))
    {
        "01"
    } else if s.contains("AGXMetal") {
        "02"
    } else if s.starts_with("/System/Library") {
        "03"
    } else if s.starts_with("/Library") {
        "04"
    } else if s.starts_with("/Applications") {
        "05"
    } else {
        "99"
    };
    (band.to_string(), s)
}

fn resolve_llvm_dis(explicit: Option<&Path>) -> Result<PathBuf, String> {
    let mut candidates = Vec::<PathBuf>::new();
    if let Some(path) = explicit {
        candidates.push(path.to_path_buf());
    }
    if let Some(path) = std::env::var_os("METAL2VULKAN_LLVM_DIS") {
        candidates.push(PathBuf::from(path));
    }
    candidates.push(PathBuf::from("/opt/homebrew/opt/llvm/bin/llvm-dis"));
    candidates.push(PathBuf::from("/usr/local/opt/llvm/bin/llvm-dis"));
    candidates.push(PathBuf::from("llvm-dis"));

    for candidate in candidates {
        if candidate.components().count() > 1 && candidate.is_file() {
            return Ok(candidate);
        }
        if candidate.components().count() == 1 {
            if let Some(found) = find_on_path(&candidate) {
                return Ok(found);
            }
        }
    }
    Err("llvm-dis not found (pass --llvm-dis or set METAL2VULKAN_LLVM_DIS)".into())
}

fn find_on_path(name: &Path) -> Option<PathBuf> {
    let name = name.to_str()?;
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

struct HarvestIntermediateCleanup {
    out: PathBuf,
}

impl HarvestIntermediateCleanup {
    fn new(out: PathBuf) -> Self {
        Self { out }
    }
}

impl Drop for HarvestIntermediateCleanup {
    fn drop(&mut self) {
        cleanup_intermediate_dirs(&self.out.join("local"));
    }
}

fn cleanup_legacy_intermediates(out: &Path) {
    let local = out.join("local");
    cleanup_intermediate_dirs(&local);
    cleanup_legacy_root_shards(&local);
}

fn cleanup_intermediate_dirs(out: &Path) {
    for sub in ["air", "metallib", "ledger", "tmp", "shards"] {
        let path = out.join(sub);
        if path.exists() {
            if let Err(e) = fs::remove_dir_all(&path) {
                eprintln!("# warn: remove intermediate {}: {e}", path.display());
            } else {
                eprintln!("# removed intermediate {}", path.display());
            }
        }
    }
}

fn cleanup_legacy_root_shards(out: &Path) {
    let Ok(entries) = fs::read_dir(out) else {
        return;
    };
    for ent in entries.flatten() {
        let path = ent.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.starts_with("shard_") && name.ends_with(".jsonl") && path.is_file() {
            if let Err(e) = fs::remove_file(&path) {
                eprintln!("# warn: remove legacy shard {}: {e}", path.display());
            } else {
                eprintln!("# removed legacy shard {}", path.display());
            }
        }
    }
}

/// What examining one metallib found, before any row is merged.
struct Examined {
    lib_sha: String,
    carved: usize,
    skipped_large: usize,
    container: Container,
    /// Functions of a kept stage that no carved module starts at.
    unmatched: usize,
    modules: Vec<ModuleOutcome>,
}

/// What one carved module is.
enum ModuleOutcome {
    Source(SourceRow),
    LibraryModule(LibraryModuleRow),
    Refused {
        offset: usize,
        air_sha: String,
        refusal: Refusal,
    },
    /// A module defining no function.
    NonFunction,
    /// A module listing no entry of a kept stage (`--only-stages`).
    Filtered,
    LlvmFailed {
        offset: usize,
        error: String,
    },
}

/// Read, carve, and disassemble the metallib `lib`: under a stage filter, only the modules of the
/// functions its function table lists for a kept stage (none at all when it lists none).
fn examine_lib(lib: &Path, llvm_dis: &Path, opts: &Options) -> Result<Examined, String> {
    let lib_bytes = fs::read(lib).map_err(|e| format!("read {}: {e}", lib.display()))?;
    let lib_sha = sha256_bytes(&lib_bytes);
    let container = match &opts.only_stages {
        Some(filter) => container_selection(filter, &lib_bytes),
        None => Container::Unfiltered,
    };
    let (blobs, skipped_large) = match &container {
        Container::Rejected => (Vec::new(), 0),
        _ => extract_air_blobs(&lib_bytes, opts.max_air_bytes),
    };
    let carved = blobs.len();
    let (blobs, unmatched) = match &container {
        Container::Selected(offsets) => {
            let found = offsets
                .iter()
                .filter(|offset| blobs.iter().any(|blob| blob.offset == **offset))
                .count();
            let kept = blobs
                .into_iter()
                .filter(|blob| offsets.contains(&blob.offset))
                .collect::<Vec<_>>();
            (kept, offsets.len() - found)
        }
        _ => (blobs, 0),
    };
    drop(lib_bytes);
    let modules = blobs
        .into_iter()
        .map(|blob| examine_module(blob, &lib_sha, llvm_dis, opts))
        .collect();
    Ok(Examined {
        lib_sha,
        carved,
        skipped_large,
        container,
        unmatched,
        modules,
    })
}

/// Disassemble one carved module and classify it.
fn examine_module(blob: AirBlob, lib_sha: &str, llvm_dis: &Path, opts: &Options) -> ModuleOutcome {
    let raw_ll = match ScratchDir::new("harvest-disassemble")
        .and_then(|scratch| disassemble_air(&blob.bytes, llvm_dis, scratch, opts.llvm_dis_timeout))
    {
        Ok(ll) => ll,
        Err(error) => {
            return ModuleOutcome::LlvmFailed {
                offset: blob.offset,
                error,
            }
        }
    };
    let ll_text = metal2vulkan_validation::air::sanitize::sanitize_ll_text(&raw_ll);
    if opts
        .only_stages
        .as_ref()
        .is_some_and(|filter| !filter.admits_module(&ll_text))
    {
        return ModuleOutcome::Filtered;
    }
    let air_sha = sha256_bytes(ll_text.as_bytes());
    if let Some(refusal) = refusal(&ll_text) {
        return ModuleOutcome::Refused {
            offset: blob.offset,
            air_sha,
            refusal,
        };
    }
    let blob_b64 = base64::engine::general_purpose::STANDARD.encode(&blob.bytes);
    let Some((stage, entry)) = classify_module(&ll_text) else {
        if !ll_text
            .lines()
            .any(|line| line.trim_start().starts_with("define "))
        {
            return ModuleOutcome::NonFunction;
        }
        return ModuleOutcome::LibraryModule(LibraryModuleRow {
            module_sha256: air_sha.clone(),
            air_ll: ll_text,
            blob_b64,
            lib_sha256s: vec![lib_sha.to_string()],
            label: format!("local/library-module/{air_sha}.ll"),
        });
    };
    let label = format!("local/{air_sha}.ll");
    ModuleOutcome::Source(SourceRow {
        air_sha256: air_sha,
        stage,
        entry,
        air_ll: ll_text,
        blob_b64: Some(blob_b64),
        lib_sha256s: vec![lib_sha.to_string()],
        label,
    })
}

/// Count what examining `lib` found and hold its new rows.
fn absorb(
    lib: &Path,
    examined: Result<Examined, String>,
    held: &mut Held,
    stats: &mut HarvestStats,
) {
    let examined = match examined {
        Ok(examined) => examined,
        Err(error) => {
            stats.libs_failed += 1;
            eprintln!("  FAIL {error}");
            return;
        }
    };
    stats.blobs_carved += examined.carved;
    stats.skipped_large += examined.skipped_large;
    match &examined.container {
        Container::Rejected => {
            stats.libs_rejected += 1;
            stats.libs_ok += 1;
            return;
        }
        Container::Unread(why) => {
            stats.container_unread += 1;
            eprintln!("  UNREAD function table {}: {why}", lib.display());
        }
        Container::Unfiltered | Container::Selected(_) => {}
    }
    if examined.unmatched > 0 {
        stats.container_unmatched += examined.unmatched;
        eprintln!(
            "  UNMATCHED {} function(s) of a kept stage start no carved module within the size \
             limit: {}",
            examined.unmatched,
            lib.display()
        );
    }
    let mut kept_for_lib = 0usize;
    for module in examined.modules {
        match module {
            ModuleOutcome::Source(row) => {
                let bytes = row.air_ll.len() + row.blob_b64.as_ref().map_or(0, String::len);
                if merge_source_row(&mut held.sources, row, stats) {
                    kept_for_lib += 1;
                    held.bytes += bytes;
                }
            }
            ModuleOutcome::LibraryModule(row) => {
                let bytes = row.air_ll.len() + row.blob_b64.len();
                if merge_library_module_row(&mut held.library_modules, row, stats) {
                    held.bytes += bytes;
                }
            }
            ModuleOutcome::Refused {
                offset,
                air_sha,
                refusal,
            } => {
                stats.refused += 1;
                eprintln!(
                    "  REFUSED {} off={offset} air={air_sha}: {refusal}",
                    lib.display()
                );
            }
            ModuleOutcome::NonFunction => stats.dropped_nonfunctions += 1,
            ModuleOutcome::Filtered => stats.filtered += 1,
            ModuleOutcome::LlvmFailed { offset, error } => {
                stats.llvm_failed += 1;
                eprintln!("  FAIL llvm-dis {} off={offset} ({error})", lib.display());
            }
        }
    }
    stats.libs_ok += 1;
    eprintln!(
        "  ok kept={} carved={} lib={} {}",
        kept_for_lib,
        examined.carved,
        &examined.lib_sha[..12],
        lib.display()
    );
}

/// Hold `row`, or add its library to the held row of the same module; whether it was new.
fn merge_library_module_row(
    by_hash: &mut HashMap<String, LibraryModuleRow>,
    row: LibraryModuleRow,
    stats: &mut HarvestStats,
) -> bool {
    let hash = row.module_sha256.clone();
    match by_hash.get_mut(&hash) {
        None => {
            by_hash.insert(hash, row);
            stats.library_modules_kept += 1;
            true
        }
        Some(previous) => {
            previous.lib_sha256s.extend(row.lib_sha256s);
            previous.lib_sha256s.sort();
            previous.lib_sha256s.dedup();
            if row.blob_b64 < previous.blob_b64 {
                previous.blob_b64 = row.blob_b64;
            }
            false
        }
    }
}

fn merge_source_row(
    by_hash: &mut HashMap<String, SourceRow>,
    row: SourceRow,
    stats: &mut HarvestStats,
) -> bool {
    let hash = row.air_sha256.clone();
    match by_hash.get_mut(&hash) {
        None => {
            by_hash.insert(hash, row);
            stats.rows_kept += 1;
            true
        }
        Some(previous) => {
            previous.lib_sha256s.extend(row.lib_sha256s);
            previous.lib_sha256s.sort();
            previous.lib_sha256s.dedup();
            if source::source_blob_is_preferred(
                previous.blob_b64.as_deref(),
                row.blob_b64.as_deref(),
            ) {
                previous.blob_b64 = row.blob_b64;
            }
            stats.duplicates += 1;
            false
        }
    }
}

/// The bitcode-wrapped modules in `data` no larger than `max_air_bytes`, and how many larger ones
/// were skipped.
fn extract_air_blobs(data: &[u8], max_air_bytes: usize) -> (Vec<AirBlob>, usize) {
    let mut out = Vec::new();
    let mut skipped_large = 0usize;
    let mut i = 0usize;
    while let Some(rel) = data[i..]
        .windows(AIR_WRAP.len())
        .position(|w| w == AIR_WRAP)
    {
        let j = i + rel;
        if j + 0x14 <= data.len() {
            let bc_off = u32::from_le_bytes(data[j + 8..j + 12].try_into().unwrap()) as usize;
            let bc_size = u32::from_le_bytes(data[j + 12..j + 16].try_into().unwrap()) as usize;
            let blen = bc_off.saturating_add(bc_size);
            if 0x14 <= blen && blen <= data.len() - j {
                if blen <= max_air_bytes {
                    out.push(AirBlob {
                        offset: j,
                        bytes: data[j..j + blen].to_vec(),
                    });
                } else {
                    skipped_large += 1;
                }
            }
        }
        i = j + 1;
    }
    (out, skipped_large)
}

fn disassemble_air(
    air: &[u8],
    llvm_dis: &Path,
    scratch: ScratchDir,
    timeout: Duration,
) -> Result<String, String> {
    let in_air = scratch.path().join("case.air");
    let out_ll = scratch.path().join("case.ll");
    let stdout_path = scratch.path().join("stdout.txt");
    let stderr_path = scratch.path().join("stderr.txt");
    fs::write(&in_air, air).map_err(|e| format!("write {}: {e}", in_air.display()))?;
    let mut child = Command::new(llvm_dis)
        .arg(&in_air)
        .arg("-o")
        .arg(&out_ll)
        .stdout(Stdio::from(fs::File::create(&stdout_path).map_err(
            |error| format!("create {}: {error}", stdout_path.display()),
        )?))
        .stderr(Stdio::from(fs::File::create(&stderr_path).map_err(
            |error| format!("create {}: {error}", stderr_path.display()),
        )?))
        .spawn()
        .map_err(|e| format!("spawn {}: {e}", llvm_dis.display()))?;
    let status = child
        .wait_timeout(timeout)
        .map_err(|error| format!("wait for {}: {error}", llvm_dis.display()))?;
    let Some(status) = status else {
        let _ = child.kill();
        let _ = child.wait();
        return Err(format!(
            "llvm-dis timed out after {} seconds",
            timeout.as_secs_f64()
        ));
    };
    if status.success() && out_ll.is_file() {
        fs::read_to_string(&out_ll).map_err(|e| format!("read {}: {e}", out_ll.display()))
    } else {
        let stdout = fs::read(&stdout_path).unwrap_or_default();
        let stderr = fs::read(&stderr_path).unwrap_or_default();
        Err(format!(
            "llvm-dis exited {status}: {}{}",
            String::from_utf8_lossy(&stdout),
            String::from_utf8_lossy(&stderr)
        ))
    }
}

fn classify_module(ll: &str) -> Option<(String, String)> {
    let (stage, entry) = stage_entry_from_ll(ll)?;

    if entry.starts_with("_GLOBAL__sub_I_") || entry.contains("_stitching_traits_impl") {
        return None;
    }
    Some((stage.to_string(), entry))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn default_output_is_the_corpus_root() {
        let manifest_dir = Path::new("/validation");
        let output = default_output_dir(manifest_dir);
        assert_eq!(output, Path::new("/validation/corpus"));
        assert_eq!(
            source::source_shards_dir(&output),
            Path::new("/validation/corpus/local/sources")
        );
    }

    #[test]
    fn exit_cleanup_removes_air_and_metallib_dirs() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "m2v-corpus-harvest-cleanup-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("local/air")).expect("create air dir");
        fs::create_dir_all(root.join("local/metallib")).expect("create metallib dir");
        fs::write(root.join("local/air/case.air"), b"air").expect("write air fixture");
        fs::write(root.join("local/metallib/case.metallib"), b"metallib")
            .expect("write metallib fixture");

        {
            let _cleanup = HarvestIntermediateCleanup::new(root.clone());
        }

        assert!(!root.join("local/air").exists());
        assert!(!root.join("local/metallib").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn extracts_valid_wrappers_and_enforces_size_limit() {
        let payload = b"ABCD";
        let mut wrapper = vec![0u8; 0x18];
        wrapper[..4].copy_from_slice(AIR_WRAP);
        wrapper[8..12].copy_from_slice(&0x14u32.to_le_bytes());
        wrapper[12..16].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        wrapper[0x14..].copy_from_slice(payload);
        let (blobs, skipped) = extract_air_blobs(&wrapper, wrapper.len());
        assert_eq!((blobs.len(), skipped), (1, 0));
        assert_eq!(blobs[0].bytes, wrapper);
        let (blobs, skipped) = extract_air_blobs(&wrapper, wrapper.len() - 1);
        assert_eq!((blobs.len(), skipped), (0, 1));
    }

    /// Examine `lib` and absorb what it found, as the walk does for each metallib.
    fn harvest_one(lib: &Path, llvm_dis: &Path, opts: &Options) -> (Held, HarvestStats) {
        let (mut held, mut stats) = (Held::default(), HarvestStats::default());
        absorb(lib, examine_lib(lib, llvm_dis, opts), &mut held, &mut stats);
        (held, stats)
    }

    fn options(out: PathBuf, metallibs: Vec<PathBuf>, llvm_dis: &Path) -> Options {
        Options {
            out,
            limit: None,
            offset: 0,
            start_set: StartSet::System,
            include_apps: false,
            metallibs,
            llvm_dis: Some(llvm_dis.to_path_buf()),
            max_air_bytes: DEFAULT_MAX_AIR_BYTES,
            llvm_dis_timeout: DEFAULT_LLVM_DIS_TIMEOUT,
            only_stages: None,
            from_archive: None,
            rows: None,
            jobs: 1,
        }
    }

    /// `stream` in a bitcode wrapper.
    fn wrap(stream: &[u8]) -> Vec<u8> {
        let mut out = AIR_WRAP.to_vec();
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&0x14u32.to_le_bytes());
        out.extend_from_slice(&(stream.len() as u32).to_le_bytes());
        out.extend_from_slice(&u32::MAX.to_le_bytes());
        out.extend_from_slice(stream);
        out
    }

    /// A metallib in the layout Apple's toolchain writes (the product's `air::container`
    /// parser reads it): each function's name, `TYPE`, and wrapped bitcode.
    fn metallib(functions: &[(&str, u8, Vec<u8>)]) -> Vec<u8> {
        fn tag(out: &mut Vec<u8>, fourcc: &[u8; 4], bytes: &[u8]) {
            out.extend_from_slice(fourcc);
            out.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            out.extend_from_slice(bytes);
        }
        let empty_metadata = || {
            let mut out = 4u32.to_le_bytes().to_vec();
            out.extend_from_slice(b"ENDT");
            out
        };
        let (mut public, mut private, mut bitcode, mut entries) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (name, type_code, wrapped) in functions {
            let (pub_at, priv_at, bc_at) = (public.len(), private.len(), bitcode.len());
            public.extend(empty_metadata());
            private.extend(empty_metadata());
            bitcode.extend_from_slice(wrapped);
            let mut list = Vec::new();
            tag(&mut list, b"NAME", &[name.as_bytes(), &[0]].concat());
            tag(&mut list, b"TYPE", &[*type_code]);
            let hex = sha256_bytes(wrapped);
            let hash: Vec<u8> = (0..32)
                .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
                .collect();
            tag(&mut list, b"HASH", &hash);
            tag(&mut list, b"MDSZ", &(wrapped.len() as u64).to_le_bytes());
            let offt: Vec<u8> = [pub_at, priv_at, bc_at]
                .iter()
                .flat_map(|v| (*v as u64).to_le_bytes())
                .collect();
            tag(&mut list, b"OFFT", &offt);
            tag(&mut list, b"VERS", &[2, 0, 8, 0, 4, 0, 0, 0]);
            list.extend_from_slice(b"ENDT");
            let mut entry = ((list.len() + 4) as u32).to_le_bytes().to_vec();
            entry.extend(list);
            entries.push(entry);
        }
        let entries_len: usize = entries.iter().map(Vec::len).sum();
        let list_off = 0x58usize;
        let pub_off = list_off + 4 + entries_len + 4;
        let priv_off = pub_off + public.len();
        let bc_off = priv_off + private.len();
        let file_size = bc_off + bitcode.len();
        let mut out = b"MTLB".to_vec();
        for half in [0x8001u16, 2, 9] {
            out.extend_from_slice(&half.to_le_bytes());
        }
        out.extend_from_slice(&[0, 0x81]);
        out.extend_from_slice(&26u16.to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes());
        for v in [
            file_size,
            list_off,
            entries_len,
            pub_off,
            public.len(),
            priv_off,
            private.len(),
            bc_off,
            bitcode.len(),
        ] {
            out.extend_from_slice(&(v as u64).to_le_bytes());
        }
        out.extend_from_slice(&(functions.len() as u32).to_le_bytes());
        for entry in entries {
            out.extend(entry);
        }
        out.extend_from_slice(b"ENDT");
        out.extend(public);
        out.extend(private);
        out.extend(bitcode);
        assert_eq!(out.len(), file_size);
        out
    }

    #[test]
    fn a_stage_filter_names_entry_lists() {
        let filter = StageFilter::parse("object, mesh").unwrap();
        assert!(filter.admits(Stage::Object) && filter.admits(Stage::Mesh));
        assert!(!filter.admits(Stage::Kernel));
        assert_eq!(filter.names(), "object,mesh");
        assert!(StageFilter::parse("tile").unwrap_err().contains("\"tile\""));
        assert!(StageFilter::parse("").is_err());
    }

    /// A stage filter keeps a module by its entry lists: a kept stage's entry, never a library
    /// module or another stage's entry.
    #[test]
    fn a_stage_filter_keeps_only_modules_listing_a_kept_stage() {
        let filter = StageFilter::parse("object,mesh").unwrap();
        let mesh = "define void @m() { ret void }\n!air.mesh = !{!0}\n!0 = !{ptr @m}\n";
        let object = "define void @o() { ret void }\n!air.object = !{!0}\n!0 = !{ptr @o}\n";
        let kernel = "define void @k() { ret void }\n!air.kernel = !{!0}\n!0 = !{ptr @k}\n";
        assert!(filter.admits_module(mesh) && filter.admits_module(object));
        assert!(!filter.admits_module(kernel));
        assert!(!filter.admits_module("define void @visible() { ret void }\n"));
    }

    /// Under a stage filter the function table decides before any disassembly: a metallib that
    /// lists no function of a kept stage is never carved or disassembled, one that does has only
    /// those functions' modules disassembled, and a library the container parser cannot read
    /// falls back to every carved module. The disassembler here does not exist, so every module
    /// handed to it is an `LlvmFailed` at its offset: that is what proves which were.
    #[test]
    fn the_function_table_selects_modules_before_disassembly() {
        let scratch = ScratchDir::new("harvest-stage-filter").unwrap();
        let missing = scratch.path().join("no-llvm-dis");
        let kernel = wrap(b"BC\xc0\xde kernel");
        let mesh = wrap(b"BC\xc0\xde mesh!");
        let filter = StageFilter::parse("object,mesh").unwrap();
        let mixed = metallib(&[("k", 2, kernel.clone()), ("m", 7, mesh.clone())]);
        let only_kernel = metallib(&[("k", 2, kernel.clone())]);
        let mesh_at = mixed.len() - mesh.len();
        assert_eq!(
            container_selection(&filter, &mixed),
            Container::Selected(BTreeSet::from([mesh_at]))
        );
        assert_eq!(
            container_selection(&filter, &only_kernel),
            Container::Rejected
        );
        assert!(matches!(
            container_selection(&filter, &[b"junk".as_slice(), &kernel].concat()),
            Container::Unread(_)
        ));

        let write = |name: &str, bytes: &[u8]| {
            let path = scratch.path().join(name);
            fs::write(&path, bytes).unwrap();
            path
        };
        let (mixed, only_kernel, unread) = (
            write("mixed.metallib", &mixed),
            write("kernel.metallib", &only_kernel),
            write(
                "unread.metallib",
                &[b"junk".as_slice(), &kernel, &mesh].concat(),
            ),
        );
        let mut opts = options(scratch.path().join("out"), Vec::new(), &missing);
        opts.only_stages = Some(filter);
        let failed_at = |held: &Held, stats: &HarvestStats| {
            assert!(held.sources.is_empty() && held.library_modules.is_empty());
            stats.llvm_failed
        };
        let (held, stats) = harvest_one(&only_kernel, &missing, &opts);
        assert_eq!((stats.libs_rejected, stats.blobs_carved), (1, 0));
        assert_eq!(failed_at(&held, &stats), 0, "no module was disassembled");
        let (held, stats) = harvest_one(&mixed, &missing, &opts);
        assert_eq!((stats.libs_rejected, stats.blobs_carved), (0, 2));
        assert_eq!(
            failed_at(&held, &stats),
            1,
            "only the mesh function's module"
        );
        let examined = examine_lib(&mixed, &missing, &opts).unwrap();
        assert!(matches!(
            examined.modules.as_slice(),
            [ModuleOutcome::LlvmFailed { offset, .. }] if *offset == mesh_at
        ));
        let (held, stats) = harvest_one(&unread, &missing, &opts);
        assert_eq!(stats.container_unread, 1);
        assert_eq!(failed_at(&held, &stats), 2, "every carved module");
    }

    /// A source row and a library module for the archive tests: `ll` under the parent library
    /// `lib`.
    fn archive_row(ll: &str, lib: &str) -> SourceRow {
        let (stage, entry) = classify_module(ll).unwrap();
        let hash = sha256_bytes(ll.as_bytes());
        SourceRow {
            air_sha256: hash.clone(),
            stage,
            entry,
            air_ll: ll.into(),
            blob_b64: Some(base64::engine::general_purpose::STANDARD.encode(b"blob")),
            lib_sha256s: vec![lib.repeat(32)],
            label: format!("local/{hash}.ll"),
        }
    }

    fn archive_module(ll: &str, lib: &str) -> LibraryModuleRow {
        LibraryModuleRow {
            module_sha256: sha256_bytes(ll.as_bytes()),
            air_ll: ll.into(),
            blob_b64: base64::engine::general_purpose::STANDARD.encode(b"bitcode"),
            lib_sha256s: vec![lib.repeat(32)],
            label: "local/library-module/helper.ll".into(),
        }
    }

    /// The names of the functions the library modules under `root` define, sorted.
    fn merged_module_functions(root: &Path) -> Vec<String> {
        let mut names = Vec::new();
        for shard in 0..source::SHARD_COUNT {
            let path = library_module::library_module_shard_path(root, shard);
            if path.is_file() {
                for row in library_module::read_library_module_shard(&path).unwrap() {
                    names.extend(library_module::defined_function_names(&row.air_ll));
                }
            }
        }
        names.sort();
        names
    }

    const MESH: &str = "define void @m() { ret void }\n!air.mesh = !{!0}\n!0 = !{ptr @m}\n";
    const KERNEL: &str = "define void @k() { ret void }\n!air.kernel = !{!0}\n!0 = !{ptr @k}\n";

    /// An archive of a mesh row (library `11`), a kernel row (library `22`), and a library module
    /// in each library and in a third (`33`) that no row comes from.
    fn archive_of_two_libraries(archive: &Path) {
        source::merge_source_shards(
            archive,
            [archive_row(MESH, "11"), archive_row(KERNEL, "22")],
        )
        .unwrap();
        library_module::merge_library_module_shards(
            archive,
            [
                archive_module("define void @in_11() { ret void }\n", "11"),
                archive_module("define void @in_22() { ret void }\n", "22"),
                archive_module("define void @in_33() { ret void }\n", "33"),
            ],
        )
        .unwrap();
    }

    /// `--from-archive` selects the kept stages' rows of an earlier harvest's shards and merges
    /// only them into `--out`, with the library modules of their parent libraries; another
    /// stage's row and another library's module stay in the archive.
    #[test]
    fn an_archive_selection_merges_only_kept_rows() {
        let scratch = ScratchDir::new("harvest-archive").unwrap();
        let (archive, out) = (scratch.path().join("archive"), scratch.path().join("out"));
        archive_of_two_libraries(&archive);
        let mut opts = options(out.clone(), Vec::new(), Path::new("unused"));
        opts.from_archive = Some(archive.clone());
        assert!(cherry_pick(&opts, &archive)
            .unwrap_err()
            .contains("--only-stages or --rows"));
        opts.only_stages = Some(StageFilter::parse("object,mesh").unwrap());
        cherry_pick(&opts, &archive).unwrap();
        let kept = source::read_all_private_sources(&out).unwrap();
        assert_eq!(
            kept.iter()
                .map(|row| row.entry.as_str())
                .collect::<Vec<_>>(),
            ["m"]
        );
        assert_eq!(merged_module_functions(&out), ["in_11"]);
        assert!(cherry_pick(&opts, &out).unwrap_err().contains("is --out"));
    }

    /// `--rows` selects the listed source rows, with the library modules of their parent
    /// libraries. A listed row the archive lacks, or one the stage filter given beside it does not
    /// admit, fails the selection before anything is merged, and so does a line whose first field
    /// is no lowercase SHA-256.
    #[test]
    fn a_row_selection_merges_the_listed_rows_and_their_libraries_modules() {
        let scratch = ScratchDir::new("harvest-archive-rows").unwrap();
        let (archive, out) = (scratch.path().join("archive"), scratch.path().join("out"));
        archive_of_two_libraries(&archive);
        let list = scratch.path().join("rows.txt");
        let kernel = sha256_bytes(KERNEL.as_bytes());
        let mut opts = options(out.clone(), Vec::new(), Path::new("unused"));
        opts.from_archive = Some(archive.clone());
        opts.rows = Some(list.clone());

        fs::write(
            &list,
            format!("{}\tabsent\n{kernel} kernel\n", "ab".repeat(32)),
        )
        .unwrap();
        let error = cherry_pick(&opts, &archive).unwrap_err();
        assert!(error.contains("1 listed row(s)"), "{error}");
        assert!(!source::source_shards_dir(&out).exists());

        fs::write(&list, "K\n").unwrap();
        assert!(cherry_pick(&opts, &archive)
            .unwrap_err()
            .contains("lowercase SHA-256"));

        fs::write(&list, format!("{kernel}\n")).unwrap();
        opts.only_stages = Some(StageFilter::parse("mesh").unwrap());
        assert!(cherry_pick(&opts, &archive)
            .unwrap_err()
            .contains("the stage filter admits"));
        assert!(!source::source_shards_dir(&out).exists());

        opts.only_stages = None;
        cherry_pick(&opts, &archive).unwrap();
        let kept = source::read_all_private_sources(&out).unwrap();
        assert_eq!(
            kept.iter()
                .map(|row| row.air_sha256.as_str())
                .collect::<Vec<_>>(),
            [kernel.as_str()]
        );
        assert_eq!(merged_module_functions(&out), ["in_22"]);
    }

    #[test]
    fn classification_requires_stable_stage_metadata() {
        let ll = "define void @k() { ret void }\n!air.kernel = !{!0}\n!0 = !{ptr @k}";
        assert_eq!(classify_module(ll), Some(("Kernel".into(), "k".into())));
        assert_eq!(classify_module("define void @helper() { ret void }"), None);
        assert_eq!(
            classify_module("define void @k() { ret void }\n!air.kernel = !{!0}\n!0 = !{i32 1}"),
            None
        );
    }

    /// An object or mesh function is a source row of its stage, not a library module: harvest reads
    /// the `!air.object` and `!air.mesh` lists through the one stage vocabulary, and a module
    /// listing an object and a mesh function is several entries.
    #[test]
    fn object_and_mesh_functions_are_source_rows() {
        let mesh = "define void @m() { ret void }\n!air.mesh = !{!0}\n!0 = !{ptr @m}\n";
        let object = "define void @o() { ret void }\n!air.object = !{!0}\n!0 = !{ptr @o}\n";
        assert_eq!(classify_module(mesh), Some(("Mesh".into(), "m".into())));
        assert_eq!(classify_module(object), Some(("Object".into(), "o".into())));
        let both = "define void @o() { ret void }\ndefine void @m() { ret void }\n\
                    !air.object = !{!0}\n!air.mesh = !{!1}\n!0 = !{ptr @o}\n!1 = !{ptr @m}\n";
        assert_eq!(refusal(both), Some(Refusal::SeveralEntries { entries: 2 }));
    }

    /// A module with several entry points is refused by name and count, and is no row a shard
    /// could hold: the row its first entry would make fails validation.
    #[test]
    fn a_module_with_several_entries_is_a_typed_refusal() {
        let one = "define void @k() { ret void }\n!air.kernel = !{!0}\n!0 = !{ptr @k}\n";
        let two = "define void @a() { ret void }\ndefine void @b() { ret void }\n\
                   !air.kernel = !{!0}\n!air.fragment = !{!1}\n!0 = !{ptr @a}\n!1 = !{ptr @b}\n";
        assert_eq!(refusal(one), None);
        assert_eq!(refusal(two), Some(Refusal::SeveralEntries { entries: 2 }));
        assert!(refusal(two)
            .unwrap()
            .to_string()
            .starts_with("several_entries entries=2:"));
        let (stage, entry) = classify_module(two).unwrap();
        let row = SourceRow {
            air_sha256: sha256_bytes(two.as_bytes()),
            stage,
            entry,
            air_ll: two.into(),
            blob_b64: None,
            lib_sha256s: vec!["11".repeat(32)],
            label: "local/two.ll".into(),
        };
        assert!(row
            .validate()
            .unwrap_err()
            .contains("declares 2 entry points"));
    }

    /// Harvest refuses a carved module with several entries by its typed refusal: counted, and no
    /// source row or library module is kept for it, while a single-entry module beside it in the
    /// same library is kept.
    #[test]
    fn harvest_refuses_a_carved_module_with_several_entries() {
        use metal2vulkan_devtools::{llvm_assemble, AssembleError};
        let Ok(llvm_dis) = resolve_llvm_dis(None) else {
            eprintln!("SKIPPED: llvm-dis not found");
            return;
        };
        let scratch = ScratchDir::new("harvest-several-entries").unwrap();
        let wrapped = |name: &str, ll: &str| -> Option<Vec<u8>> {
            let path = scratch.path().join(format!("{name}.ll"));
            fs::write(&path, ll).unwrap();
            let bitcode = match llvm_assemble(&path) {
                Ok(bitcode) => bitcode,
                Err(AssembleError::Missing(why)) => {
                    eprintln!("SKIPPED: {why}");
                    return None;
                }
                Err(AssembleError::Rejected(why)) => panic!("llvm-as: {why}"),
            };
            let mut wrapper = vec![0u8; 0x14];
            wrapper[..4].copy_from_slice(AIR_WRAP);
            wrapper[8..12].copy_from_slice(&0x14u32.to_le_bytes());
            wrapper[12..16].copy_from_slice(&(bitcode.len() as u32).to_le_bytes());
            wrapper.extend(bitcode);
            Some(wrapper)
        };
        let Some(one) = wrapped(
            "one",
            "define void @k() {\n  ret void\n}\n!air.kernel = !{!0}\n!0 = !{ptr @k}\n",
        ) else {
            return;
        };
        let two = wrapped(
            "two",
            "define void @a() {\n  ret void\n}\ndefine void @b() {\n  ret void\n}\n\
             !air.kernel = !{!0, !1}\n!0 = !{ptr @a}\n!1 = !{ptr @b}\n",
        )
        .unwrap();
        let lib = scratch.path().join("library.metallib");
        fs::write(&lib, [one, two].concat()).unwrap();
        let opts = options(scratch.path().join("out"), vec![lib.clone()], &llvm_dis);
        let (held, stats) = harvest_one(&lib, &llvm_dis, &opts);
        let (rows, modules) = (held.sources, held.library_modules);
        assert_eq!(
            (
                stats.blobs_carved,
                stats.llvm_failed,
                stats.refused,
                stats.rows_kept
            ),
            (2, 0, 1, 1)
        );
        assert_eq!(
            rows.values()
                .map(|row| row.entry.as_str())
                .collect::<Vec<_>>(),
            ["k"]
        );
        assert!(modules.is_empty());
    }

    #[test]
    fn non_entry_function_modules_retain_every_parent_library() {
        let ll = "define void @visible() { ret void }";
        assert_eq!(classify_module(ll), None);
        let mut modules = HashMap::new();
        let mut stats = HarvestStats::default();
        for library in ["11".repeat(32), "22".repeat(32)] {
            merge_library_module_row(
                &mut modules,
                LibraryModuleRow {
                    module_sha256: sha256_bytes(ll.as_bytes()),
                    air_ll: ll.into(),
                    blob_b64: base64::engine::general_purpose::STANDARD.encode(b"bitcode"),
                    lib_sha256s: vec![library],
                    label: "local/library-module.ll".into(),
                },
                &mut stats,
            );
        }
        assert_eq!(stats.library_modules_kept, 1);
        assert_eq!(modules.values().next().unwrap().lib_sha256s.len(), 2);
    }

    #[test]
    fn deduplication_is_by_sanitized_air_hash_and_order_independent() {
        fn row(library: &str) -> SourceRow {
            SourceRow {
                air_sha256: "11".repeat(32),
                stage: "Kernel".into(),
                entry: "k".into(),
                air_ll: "ll".into(),
                blob_b64: Some("YmxvYg==".into()),
                lib_sha256s: vec![library.into()],
                label: "local/test.ll".into(),
            }
        }
        let mut forward = HashMap::new();
        let mut forward_stats = HarvestStats::default();
        merge_source_row(&mut forward, row("bb"), &mut forward_stats);
        merge_source_row(&mut forward, row("aa"), &mut forward_stats);

        let mut reverse = HashMap::new();
        let mut reverse_stats = HarvestStats::default();
        merge_source_row(&mut reverse, row("aa"), &mut reverse_stats);
        merge_source_row(&mut reverse, row("bb"), &mut reverse_stats);
        assert_eq!(
            forward[&"11".repeat(32)].lib_sha256s,
            ["aa".to_string(), "bb".to_string()]
        );
        assert_eq!(
            reverse[&"11".repeat(32)].lib_sha256s,
            ["aa".to_string(), "bb".to_string()]
        );
        assert_eq!(forward_stats.duplicates, 1);
        assert_eq!(reverse_stats.duplicates, 1);
    }

    #[test]
    fn indexed_entry_is_reopened_only_for_a_new_parent_library() {
        let hash = "11".repeat(32);
        let mut row = SourceRow {
            air_sha256: hash.clone(),
            stage: "Kernel".into(),
            entry: "k".into(),
            air_ll: "ll".into(),
            blob_b64: Some("YmxvYg==".into()),
            lib_sha256s: vec!["aa".into()],
            label: "local/test.ll".into(),
        };
        let mut indexed = HashMap::new();
        assert!(!source_memberships_already_indexed(&indexed, &hash, &row));

        indexed.insert(hash.clone(), HashSet::from(["aa".to_string()]));
        assert!(source_memberships_already_indexed(&indexed, &hash, &row));
        row.lib_sha256s.push("bb".into());
        assert!(!source_memberships_already_indexed(&indexed, &hash, &row));
    }

    /// Execute the fake disassembler `tool` once, untimed, and ignore the outcome. macOS scans a
    /// newly written executable on its first exec (syspolicyd; a fresh script took 58-85 s to
    /// start under a loaded gate), which would otherwise land inside the timeout.
    #[cfg(unix)]
    fn warm(tool: &Path) {
        let _ = std::process::Command::new(tool)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }

    #[cfg(unix)]
    #[test]
    fn disassembler_scratch_is_removed_on_success_failure_timeout_and_signal() {
        use std::os::unix::fs::PermissionsExt as _;

        let outer = ScratchDir::new("harvest-disassembler-test").unwrap();
        let tool = outer.path().join("llvm-dis.sh");
        for (script, timeout, succeeds) in [
            (
                "#!/bin/sh\nprintf 'define void @k() { ret void }' > \"$3\"\n",
                Duration::from_secs(20),
                true,
            ),
            ("#!/bin/sh\nexit 1\n", Duration::from_secs(20), false),
            (
                "#!/bin/sh\nexec sleep 2\n",
                Duration::from_millis(30),
                false,
            ),
            ("#!/bin/sh\nkill -TERM $$\n", Duration::from_secs(20), false),
        ] {
            fs::write(&tool, script).unwrap();
            let mut permissions = fs::metadata(&tool).unwrap().permissions();
            permissions.set_mode(0o700);
            fs::set_permissions(&tool, permissions).unwrap();
            warm(&tool);
            let scratch = ScratchDir::new("harvest-disassemble-test").unwrap();
            let scratch_path = scratch.path().to_path_buf();
            let result = disassemble_air(b"air", &tool, scratch, timeout);
            assert_eq!(result.is_ok(), succeeds, "{result:?}");
            assert!(!scratch_path.exists());
        }
    }
}
