//! The authored compile census: every committed authored case compiled without a device, with the
//! requests the engine executor builds for it, on each profile of a committed list, and
//! compared with the committed expectations.
//!
//! The plain translation census translates each row with no constants and no linked functions,
//! so a product change that breaks an authored case's own inputs (its constants, linked modules,
//! bind shape, dispatch, companion stage, or replay templates) passed it; only a device run
//! translated those inputs, and a case's committed device rows went stale instead of failing. This
//! census closes that gap:
//!
//! - **The request.** A case is admitted and prepared by the executor's own device-free half
//!   (`engine::prepare_compile`: its run's preparation, except that an object or mesh case, whose
//!   mesh render pipeline the executor does not draw, is prepared to compile it), and the one
//!   pipeline a run compiles before it reads device output, the case's own, comes from
//!   `engine::case_compile`. It compiles in a compile worker (the 20-s and 500-MiB per-attempt
//!   ceilings, a process-group kill).
//! - **The bounds.** Each case is admitted, prepared, and compiled in its own case worker
//!   ([`CaseWorkers`]) under the 500-MiB ceiling, so the census's process holds only the
//!   committed cases and their cells; `jobs` cases run at once, each case's compiles one at a time.
//!   A case worker reads private sources by the byte ranges the census's index records
//!   (`source::use_source_index`), and parses a whole shard only for a source the index does not
//!   locate; the report counts both.
//! - **The profiles.** The profiles of the expectation file's header, each a committed profile
//!   as the runtime's device describes it (`engine::runtime_description`: read at the runtime's
//!   Vulkan version, restricted to what the runtime enables of it), derived under the default
//!   policy as every census row is. A device run compiles for the opened device, restricted to
//!   what the executor enables, which is the same set.
//! - **The baseline.** The committed expectation file ([`EXPECTATIONS`] under the corpus root):
//!   one line per case, its cell on each profile ([`Cell`]): `compiled` or the executor's
//!   stable refusal label (`NotSupported::outcome`).
//!   A run fails on any failure (an internal error, a worker stop, a case that does not check or
//!   map) and on any difference from the file: a compiled case refused, a refusal label changed,
//!   a refused case compiled, a new case, a removed case. `--bless` rewrites the file and prints the same differences; it never writes a
//!   failure. `--bless-additions` ([`Mode::BlessAdditions`], what each train's gate runs) fails as
//!   a check does on any failure and on any changed cell of a case and profile the file
//!   already holds, and otherwise writes the file with new cases and every C3 venue's missing
//!   profiles added and removed cases dropped: a new case has no baseline to flip from, and a
//!   removed case cannot regress.
//! - **The cache.** A table of the index holds each case's cell on each profile under the
//!   census key ([`profile_key`]): the plain census's per-profile key (the product sources,
//!   `SEMANTICS_VERSION`, the profile's stem and fingerprint, the tools), the executor's ABI
//!   (which covers how it builds requests), and [`AUTHORED_COMPILE_CENSUS_VERSION`]. A case's id
//!   is content-addressed over its manifest (its AIR, linked modules, and templates by hash), so
//!   selection reads only the committed case shards and the table: a warm rerun admits, reads,
//!   and compiles nothing. Failures are never cached.

use crate::candidate::engine::artifact::CaseArtifact;
use crate::candidate::engine::{
    case_compile, prepare_compile, CompileFailure, PreparedCase, ENGINE_EXECUTOR_ABI,
};
use crate::case::AuthoredCase;
use crate::check::admission::{admit, ExecutionRequest, ExecutorError, NotSupported};
use crate::engine::bounded_detail;
use crate::gpu::DeviceType;
use crate::pool::parallel;
use crate::source::SourceReads;
use crate::store::CorpusStore;
use crate::translation_audit::CensusKey;
use crate::worker::compile::{CompileWorker, NotCompiled, WorkerDevice};
use crate::worker::{run_json_within, Measured};
use metal2vulkan::device::{DeviceDescription, DevicePolicy};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

#[cfg(test)]
mod tests;

/// The census's own version. It stands for everything under `validation/` that can change a
/// recorded cell and that neither the product's census key nor the executor's ABI covers: which
/// compiles a cell is made of and in which order, how their results are classified into an outcome,
/// and which outcomes are recorded. Bump it with any change to one of those.
///
/// 1: the first census. 2: each recorded cell holds its uncovered count (pipelines the census does
///    not compile), which the expectation file records and a run compares.
/// 3: each recorded cell holds the digest of what a run of its case executes there,
///    which the expectation file records on the C3 venues' profiles, at the build its
///    `artifacts_at` line names.
/// 4: a case the runtime runs compiles its own pipeline although the executor's context does not
///    replay its templates, each executed command buffer uncovered (`engine::case_compiles`).
/// 5: an object or mesh case compiles its mesh render pipeline, where it was refused
///    (`engine::mesh::mesh_request`).
/// 6: an object or mesh case whose only stop is that the executor draws no mesh render pipeline
///    (`mesh_render_pipeline`) compiles it, where the run's preparation refuses it
///    (`engine::prepare_compile`).
/// 7: a cell holds no uncovered count: a run compiles only its case's own pipeline
///    (`engine::case_compile`).
pub const AUTHORED_COMPILE_CENSUS_VERSION: u32 = 7;

/// The expectation file's name under the corpus root (`validation/corpus`).
pub const EXPECTATIONS: &str = "authored-compile.tsv";

/// The index table of recorded cells.
const TABLE: &str = "authored_compile_census";

/// Cells written to the index per transaction.
const CACHE_BATCH: usize = 256;

/// The expectation file of the corpus at `root`.
pub fn expectations_path(root: &Path) -> PathBuf {
    root.join(EXPECTATIONS)
}

/// A committed profile the census compiles for, by its stem.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProfileStem {
    /// The committed profile's stem (`validation/profiles/<stem>.json`).
    pub stem: String,
}

impl ProfileStem {
    /// The label the expectation file's header and the report name it by: its stem.
    pub fn label(&self) -> String {
        self.stem.clone()
    }

    /// The profile `label` names.
    pub fn parse(label: &str) -> Result<ProfileStem, String> {
        if label.is_empty() || label.contains(|c: char| c.is_whitespace() || c == '+') {
            return Err(format!(
                "{label:?} names no profile (a committed profile's stem)"
            ));
        }
        Ok(ProfileStem { stem: label.into() })
    }
}

/// A profile the census compiles for: the device a compile worker rebuilds, and the census key
/// its cells are recorded under ([`profile_key`]).
pub struct CensusProfile {
    pub profile: ProfileStem,
    pub device: WorkerDevice,
    pub key: String,
}

impl CensusProfile {
    /// The committed profile `profile` as the runtime's device describes it, under the default
    /// policy.
    pub fn load(profile: &ProfileStem) -> Result<CensusProfile, String> {
        let description = crate::engine::runtime_description(&profile.stem)?;
        let device = WorkerDevice::new(description, DevicePolicy::default())
            .map_err(|error| format!("{}: {error}", profile.label()))?;
        let key = profile_key(&CensusKey::engine(
            &profile.label(),
            device.device().profile(),
        ));
        Ok(CensusProfile {
            profile: profile.clone(),
            device,
            key,
        })
    }
}

/// The key a profile's cells are recorded under: SHA-256 over a domain tag, the plain
/// census's key for the profile (`census`), the executor's ABI, and
/// [`AUTHORED_COMPILE_CENSUS_VERSION`], each length-prefixed.
pub fn profile_key(census: &CensusKey) -> String {
    let mut bytes = Vec::new();
    for field in [
        b"metal2vulkan-authored-compile-census-key".as_slice(),
        census.fingerprint().as_bytes(),
        ENGINE_EXECUTOR_ABI.as_bytes(),
        &AUTHORED_COMPILE_CENSUS_VERSION.to_le_bytes(),
    ] {
        bytes.extend_from_slice(&(field.len() as u64).to_le_bytes());
        bytes.extend_from_slice(field);
    }
    crate::hash::sha256_bytes(&bytes)
}

/// A case's outcome on one profile, as the expectation file writes it.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CellOutcome {
    /// Every pipeline the census compiles for the case compiled.
    Compiled,
    /// The executor's stable refusal label (`NotSupported::outcome`, `ExecutionRefused::outcome`)
    /// of the first refusal. A label holds words and spaces (a product refusal's key names its
    /// feature and the capability it needs, e.g. `not_supported{device_capability_missing:Format
    /// needs Format { format: VK_FORMAT_R16_UINT, usage: StorageImage }}`); the census builds one
    /// only through [`CellOutcome::refused`], so the expectation file and the cache can hold it.
    Refused(String),
}

impl CellOutcome {
    const COMPILED: &'static str = "compiled";

    /// The cell's text.
    pub fn as_str(&self) -> &str {
        match self {
            CellOutcome::Compiled => Self::COMPILED,
            CellOutcome::Refused(label) => label,
        }
    }

    /// The refusal `label`, when a cell of the expectation file (a tab-separated line) and the
    /// cache can hold it and read it back as itself: not empty, not `compiled`, no control
    /// character (a tab, a line break), no space at either end, and no last word that starts
    /// with `artifact=` (a cell's artifact digest, [`Cell`]).
    pub fn refused(label: String) -> Result<CellOutcome, String> {
        let last_word = label.rsplit(' ').next().unwrap_or_default();
        if label.is_empty()
            || label == Self::COMPILED
            || label.contains(char::is_control)
            || label.trim() != label
            || last_word.starts_with(Cell::ARTIFACT)
        {
            return Err(format!("{label:?} is not a refusal label a cell can hold"));
        }
        Ok(CellOutcome::Refused(label))
    }

    /// The outcome a cell's text states: `compiled`, or a refusal label
    /// ([`CellOutcome::refused`]).
    pub fn parse(cell: &str) -> Result<CellOutcome, String> {
        if cell == Self::COMPILED {
            Ok(CellOutcome::Compiled)
        } else {
            CellOutcome::refused(cell.into()).map_err(|_| format!("{cell:?} is not an outcome"))
        }
    }
}

/// A case's cell on one profile, as the expectation file records it: its outcome, and, on a C3
/// venue's profile, the digest of what a run of the case executes there
/// ([`CellResult::artifact`]): for a compiled case. A cell whose outcome changed is a
/// [`Difference::Changed`], and so is one whose digest changed at the build the file's digests
/// were computed at ([`ArtifactBuild`]).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cell {
    pub outcome: CellOutcome,
    pub artifact: Option<String>,
}

impl Cell {
    /// The start of the word after `compiled` that states the artifact digest.
    const ARTIFACT: &'static str = "artifact=";

    /// A cell with no artifact digest.
    pub fn bare(outcome: CellOutcome) -> Cell {
        Cell {
            outcome,
            artifact: None,
        }
    }

    /// The cell's text: its outcome, then ` artifact=SHA256` when it has a digest.
    pub fn text(&self) -> String {
        match &self.artifact {
            None => self.outcome.as_str().to_string(),
            Some(artifact) => format!("{} {}{artifact}", self.outcome.as_str(), Self::ARTIFACT),
        }
    }

    /// The cell a cell's text states: an outcome ([`CellOutcome::parse`]), and, after `compiled`,
    /// a last word `artifact=SHA256` for a lowercase SHA-256, or none.
    pub fn parse(cell: &str) -> Result<Cell, String> {
        let Some((outcome, word)) = cell.rsplit_once(' ') else {
            return Ok(Cell::bare(CellOutcome::parse(cell)?));
        };
        let Some(artifact) = word.strip_prefix(Self::ARTIFACT) else {
            return Ok(Cell::bare(CellOutcome::parse(cell)?));
        };
        crate::observation::validate_hash("the artifact digest", artifact)
            .map_err(|why| format!("{cell:?}: {why}"))?;
        if outcome != CellOutcome::COMPILED {
            return Err(format!("{cell:?} states a digest of no compiled case"));
        }
        Ok(Cell {
            outcome: CellOutcome::Compiled,
            artifact: Some(artifact.into()),
        })
    }

    /// The cell of `outcome`, keeping `artifact` only where a cell holds one: on a profile that
    /// records digests (`digested`), for a compiled case.
    fn of(outcome: CellOutcome, artifact: Option<String>, digested: bool) -> Cell {
        let holds = digested && outcome == CellOutcome::Compiled;
        Cell {
            outcome,
            artifact: artifact.filter(|_| holds),
        }
    }

    /// Whether `self` and `other` state the same outcome, whatever their artifact digests.
    fn same_outcome(&self, other: &Cell) -> bool {
        self.outcome == other.outcome
    }

    /// This cell, when [`Cell::parse`] reads its text back as itself.
    fn holdable(&self) -> Result<&Cell, String> {
        match Cell::parse(&self.text()) {
            Ok(read) if read == *self => Ok(self),
            Ok(_) => Err(format!("{:?} reads back as another cell", self.text())),
            Err(error) => Err(error),
        }
    }
}

/// What kind of failure a cell is: never an outcome, never cached, never blessed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FailureClass {
    /// The product failed (a typed defect outcome: an internal error, a panic, an exhausted
    /// in-process budget).
    Internal,
    /// The compile worker's boundary stopped the compile (the deadline, the memory limit, a
    /// crashed worker, an artifact that does not decode).
    WorkerStop,
    /// The case did not check against its source or map onto the executor (or its manifest's id
    /// is not its content's).
    Case,
    /// The tool failed around a compile (inputs it does not build, tables that do not encode).
    Tool,
}

impl FailureClass {
    pub fn as_str(self) -> &'static str {
        match self {
            FailureClass::Internal => "internal",
            FailureClass::WorkerStop => "worker_stop",
            FailureClass::Case => "case",
            FailureClass::Tool => "tool",
        }
    }
}

/// A cell that is a failure.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellFailure {
    pub class: FailureClass,
    /// The typed outcome key, where there is one (`engine::Outcome::key`).
    pub key: String,
    pub detail: String,
}

/// What the compiles of one case on one profile measured: summed and peak.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cost {
    /// Every compile's time, from preparing its input through decoding its artifact.
    pub elapsed: Duration,
    /// The slowest compile's time.
    pub slowest: Duration,
    /// The largest sampled process-group RSS of a compile worker.
    pub peak_resident_bytes: Option<u64>,
    /// The largest counting-allocator peak a compile worker reported.
    pub allocator_peak_bytes: Option<u64>,
}

impl Cost {
    fn add(&mut self, measured: Option<&Measured>) {
        let Some(measured) = measured else {
            return;
        };
        self.elapsed += measured.elapsed;
        self.slowest = self.slowest.max(measured.elapsed);
        self.peak_resident_bytes = self.peak_resident_bytes.max(measured.peak_resident_bytes);
        self.allocator_peak_bytes = self.allocator_peak_bytes.max(measured.allocator_peak_bytes);
    }
}

/// One case compiled on one profile.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellResult {
    pub outcome: Result<CellOutcome, CellFailure>,
    /// The compiles run (a failure stops the cell at its compile, as it stops the executor).
    pub compiles: usize,
    pub cost: Cost,
    /// The digest of what a run of the case executes on the profile, as the executor records
    /// it (`candidate::engine::artifact`): for a case that compiled, `None` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<String>,
}

impl CellResult {
    /// A cell decided before any compile.
    fn decided(outcome: Result<CellOutcome, CellFailure>) -> CellResult {
        CellResult {
            outcome,
            compiles: 0,
            cost: Cost::default(),
            artifact: None,
        }
    }
}

/// What compiling one case came to: a cell per profile, what the parent measured around the
/// case worker that compiled it, and how that worker found the private sources it read (`None`
/// for a compiler that runs no case worker, or a worker that reported nothing).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaseResult {
    pub cells: Vec<CellResult>,
    pub worker: Option<Measured>,
    pub sources: Option<SourceReads>,
}

/// What compiles a case on profiles: a case worker per case ([`CaseWorkers`], the census's),
/// the compile workers from this process ([`WorkerCompiler`], what a case worker runs), or a
/// test's.
pub trait CaseCompiler: Sync {
    /// `case` compiled on each of `profiles`, one cell per profile, in order.
    fn compile(&self, case: &AuthoredCase, profiles: &[&CensusProfile]) -> CaseResult;
}

/// The executor's admission and compile preparation ([`prepare_compile`]) of `case` over the
/// corpus at `root`, then the pipeline [`case_compile`] plans for each of `devices`, each in a
/// compile worker of `worker`: one cell per device, in order.
pub fn compile_case(
    root: &Path,
    worker: &CompileWorker,
    case: &AuthoredCase,
    devices: &[&WorkerDevice],
) -> Vec<CellResult> {
    // The device type is the admission's only device input, and the engine is refused on
    // neither.
    let device = DeviceType::Cpu;
    let prepared = admit(
        root,
        case.clone(),
        ExecutionRequest::EngineCandidate { device },
    )
    .and_then(|admitted| prepare_compile(device, admitted));
    match prepared {
        Ok(prepared) => devices
            .iter()
            .map(|device| compile_prepared(&prepared, worker, device))
            .collect(),
        Err(error) => {
            let outcome = executor_outcome(error, FailureClass::Case);
            devices
                .iter()
                .map(|_| CellResult::decided(outcome.clone()))
                .collect()
        }
    }
}

/// [`compile_case`] in this process: what a case worker runs, and a test's compiler.
pub struct WorkerCompiler {
    pub root: PathBuf,
    pub worker: CompileWorker,
}

impl CaseCompiler for WorkerCompiler {
    fn compile(&self, case: &AuthoredCase, profiles: &[&CensusProfile]) -> CaseResult {
        let devices = profiles
            .iter()
            .map(|profile| &profile.device)
            .collect::<Vec<_>>();
        CaseResult {
            cells: compile_case(&self.root, &self.worker, case, &devices),
            worker: None,
            sources: None,
        }
    }
}

/// The first argument that starts a validation binary as the census's case worker.
pub const CASE_WORKER_ARG: &str = "--authored-compile-case-worker";

/// The census's compiler: each case in its own case worker, a child of `program` (a validation
/// binary that calls [`serve_if_requested`] first in `main` and serves the compile worker), which
/// runs [`compile_case`] over the corpus at `root`, reading private sources by the byte ranges the
/// index at `index` records (`source::use_source_index`). The case's admission, its sources, its linked
/// modules, its templates, and its compile inputs live in that child, under the 500-MiB ceiling
/// (its process group's RSS and its counting allocator), and go with it; the census's process
/// holds the committed cases and the cells, whatever the cases' sizes. The child's wall ceiling is
/// a safety rail of [`CaseWorkers::deadline`]: each compile in it has its own 20-s ceiling.
pub struct CaseWorkers {
    pub root: PathBuf,
    pub index: PathBuf,
    pub program: PathBuf,
}

impl CaseWorkers {
    /// The case worker's wall ceiling on `profiles` profiles: the 20-s ceiling for the case's
    /// admission, and twice it per profile.
    pub fn deadline(profiles: usize) -> Duration {
        crate::worker::DEADLINE * (1 + 2 * profiles as u32)
    }
}

/// A case worker's input, as the parent writes it: the corpus, the index whose source locations
/// it reads, the case, and each profile's description and policy, from which the child
/// rebuilds the same `Device`.
#[derive(Serialize)]
struct CaseInput<'a> {
    root: &'a Path,
    index: &'a Path,
    case: &'a AuthoredCase,
    profiles: Vec<(&'a DeviceDescription, &'a DevicePolicy)>,
}

/// A case worker's input, as it reads it.
#[derive(Deserialize)]
struct CaseInputOwned {
    root: PathBuf,
    index: PathBuf,
    case: AuthoredCase,
    profiles: Vec<(DeviceDescription, DevicePolicy)>,
}

/// A case worker's report: its cells, and how it found the private sources it read.
#[derive(Serialize, Deserialize)]
struct CaseOutput {
    cells: Vec<CellResult>,
    sources: SourceReads,
}

impl CaseCompiler for CaseWorkers {
    fn compile(&self, case: &AuthoredCase, profiles: &[&CensusProfile]) -> CaseResult {
        let mut command = Command::new(&self.program);
        command.arg(CASE_WORKER_ARG);
        let input = CaseInput {
            root: &self.root,
            index: &self.index,
            case,
            profiles: profiles
                .iter()
                .map(|profile| (profile.device.description(), profile.device.policy()))
                .collect(),
        };
        let run = run_json_within::<CaseOutput>(
            command,
            "authored-compile-case",
            CaseWorkers::deadline(profiles.len()),
            |file| {
                serde_json::to_writer(std::io::BufWriter::new(file), &input)
                    .map_err(|error| format!("encode the case worker's input: {error}"))
            },
        );
        let stopped = |key: String, detail: String, worker: Option<Measured>| CaseResult {
            cells: profiles
                .iter()
                .map(|_| {
                    CellResult::decided(Err(CellFailure {
                        class: FailureClass::WorkerStop,
                        key: key.clone(),
                        detail: detail.clone(),
                    }))
                })
                .collect(),
            worker,
            sources: None,
        };
        match run {
            Ok(run) if run.result.cells.len() == profiles.len() => {
                let worker = Some(run.measured());
                CaseResult {
                    cells: run.result.cells,
                    worker,
                    sources: Some(run.result.sources),
                }
            }
            Ok(run) => stopped(
                "case_worker".into(),
                format!(
                    "the case worker returned {} cells for {} profiles",
                    run.result.cells.len(),
                    profiles.len()
                ),
                Some(run.measured()),
            ),
            Err(stop) => stopped(
                stop.outcome().key(),
                bounded_detail(&format!("the case worker: {}", stop.message())),
                stop.measured(),
            ),
        }
    }
}

/// Serve the census's case worker when this process was started as one ([`CASE_WORKER_ARG`]):
/// read the case, its corpus, the census's index, and the profiles' descriptions from stdin,
/// read private sources through that index's locations (`source::use_source_index`), run
/// [`compile_case`] with this executable as the compile worker, report the cells and how the
/// sources were read, and exit.
/// Every binary that runs the census calls this first in `main`, and installs the counting
/// allocator.
pub fn serve_if_requested() {
    if std::env::args_os().nth(1).as_deref() != Some(std::ffi::OsStr::new(CASE_WORKER_ARG)) {
        return;
    }
    std::process::exit(crate::worker::serve_process(
        "authored compile case worker",
        crate::worker::alloc::installed(),
        |input, _started| {
            let input: CaseInputOwned = serde_json::from_slice(input)
                .map_err(|error| format!("decode the case worker's input: {error}"))?;
            crate::source::use_source_index(input.index)?;
            let worker = CompileWorker::current()?;
            let devices = input
                .profiles
                .into_iter()
                .map(|(description, policy)| {
                    WorkerDevice::new(description, policy).map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let devices = devices.iter().collect::<Vec<_>>();
            let cells = compile_case(&input.root, &worker, &input.case, &devices);
            Ok(CaseOutput {
                cells,
                sources: crate::source::source_reads(),
            })
        },
    ))
}

/// `prepared` compiled on `device`: the pipeline of [`case_compile`], and the digest of what a run
/// there executes when it compiled, as the executor's run records it.
pub fn compile_prepared(
    prepared: &PreparedCase,
    worker: &CompileWorker,
    device: &WorkerDevice,
) -> CellResult {
    let planned = match case_compile(prepared, device) {
        Ok(planned) => planned,
        Err(error) => return CellResult::decided(executor_outcome(error, FailureClass::Case)),
    };
    let mut artifact = CaseArtifact::default();
    let (result, measured) = planned.compile(worker, device, &mut artifact);
    let mut cell = CellResult {
        outcome: Ok(CellOutcome::Compiled),
        compiles: 1,
        cost: Cost::default(),
        artifact: None,
    };
    cell.cost.add(measured.as_ref());
    match result {
        Ok(()) => cell.artifact = artifact.sha256(),
        Err(failure) => cell.outcome = compile_outcome(failure),
    }
    cell
}

/// The outcome an executor error is: a typed refusal's label, or a failure of `class`.
fn executor_outcome(error: ExecutorError, class: FailureClass) -> Result<CellOutcome, CellFailure> {
    match error {
        ExecutorError::NotSupported(not_supported) => refusal(not_supported.outcome()),
        ExecutorError::Refused(refused) => refusal(refused.outcome().into()),
        ExecutorError::Failed(detail) => Err(CellFailure {
            class,
            key: "failed".into(),
            detail: bounded_detail(&detail),
        }),
        other => Err(CellFailure {
            class,
            key: "failed".into(),
            detail: bounded_detail(&other.to_string()),
        }),
    }
}

/// The outcome a compile's failure is, classified as the executor classifies it
/// (`ExecutorError::from`): a product outcome that is a defect fails, any other refuses with its
/// label; a stop at the worker boundary fails.
fn compile_outcome(failure: CompileFailure) -> Result<CellOutcome, CellFailure> {
    match failure {
        CompileFailure::Stopped(stop) => Err(CellFailure {
            class: FailureClass::WorkerStop,
            key: stop.outcome().key(),
            detail: bounded_detail(&stop.message()),
        }),
        CompileFailure::NotCompiled(NotCompiled::Product(outcome, detail)) => {
            if outcome.is_defect() {
                Err(CellFailure {
                    class: FailureClass::Internal,
                    key: outcome.key(),
                    detail,
                })
            } else {
                refusal(
                    NotSupported::Product {
                        outcome,
                        detail,
                        at_bind: false,
                    }
                    .outcome(),
                )
            }
        }
        CompileFailure::NotCompiled(NotCompiled::Unmodelled(why)) => Err(CellFailure {
            class: FailureClass::Tool,
            key: "unmodelled".into(),
            detail: bounded_detail(&format!("the compile worker built no inputs: {why:?}")),
        }),
        CompileFailure::Artifact(error) => executor_outcome(error, FailureClass::Tool),
    }
}

/// The refusal `label` as a cell's outcome, or, when no cell can hold it
/// ([`CellOutcome::refused`]), a tool failure: never a label the expectation file would not read
/// back.
fn refusal(label: String) -> Result<CellOutcome, CellFailure> {
    CellOutcome::refused(label).map_err(|why| CellFailure {
        class: FailureClass::Tool,
        key: "refusal_label".into(),
        detail: bounded_detail(&why),
    })
}

/// The build an expectation file's artifact digests were computed at: the product's source
/// fingerprint (`TRANSLATOR_FINGERPRINT`) and the executor's ABI, which covers how it builds its
/// requests. The digests are a cache of the census at that build: C3's currency trusts them only at
/// it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactBuild {
    pub product: String,
    pub executor: String,
}

impl ArtifactBuild {
    /// The line's first field.
    const LINE: &'static str = "artifacts_at";

    /// This build.
    pub fn this() -> ArtifactBuild {
        ArtifactBuild {
            product: crate::observation::TRANSLATOR_FINGERPRINT.into(),
            executor: ENGINE_EXECUTOR_ABI.into(),
        }
    }

    /// The file's line: `artifacts_at`, `product=FINGERPRINT`, `executor=ABI`, tab-separated.
    fn line(&self) -> String {
        format!(
            "{}\tproduct={}\texecutor={}",
            Self::LINE,
            self.product,
            self.executor
        )
    }

    /// The build a line states, or `None` for a line that is not an `artifacts_at` line.
    fn parse(line: &str) -> Option<Result<ArtifactBuild, String>> {
        let mut fields = line.split('\t');
        if fields.next() != Some(Self::LINE) {
            return None;
        }
        let (Some(product), Some(executor), None) = (
            fields.next().and_then(|f| f.strip_prefix("product=")),
            fields.next().and_then(|f| f.strip_prefix("executor=")),
            fields.next(),
        ) else {
            return Some(Err(format!(
                "{line:?} is not `{}\tproduct=FINGERPRINT\texecutor=ABI`",
                Self::LINE
            )));
        };
        Some(
            crate::observation::validate_hash("the artifacts' product", product)
                .and_then(|()| {
                    if executor.is_empty() {
                        Err("the artifacts' executor is empty".into())
                    } else {
                        Ok(())
                    }
                })
                .map(|()| ArtifactBuild {
                    product: product.into(),
                    executor: executor.into(),
                }),
        )
    }
}

/// The committed expectations: the build their artifact digests were computed at, the
/// profiles of the header, and each case's cell on each.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Expectations {
    /// `None` for a file that records no artifact digest.
    pub build: Option<ArtifactBuild>,
    pub profiles: Vec<ProfileStem>,
    pub cases: BTreeMap<String, Vec<Cell>>,
}

impl Expectations {
    const HEADER: &'static str = "case_id";

    /// The file's text: its comment, the build line when it has one, the header, and one line per
    /// case in id order. Every cell is one [`Expectations::parse`] reads back as itself, and a
    /// file with a digest has a build, or this is an error naming it.
    pub fn render(&self) -> Result<String, String> {
        let mut out = String::from(
            "# The authored compile census's expectations: one line per committed authored\n\
             # case, its id, then its cell on each profile of the header: `compiled`, or the \
             executor's\n\
             # stable refusal label. On a C3 venue's profile, a compiled cell is followed by\n\
             # ` artifact=SHA256`, the digest of what a run of the case executes there, computed \
             at the\n\
             # build the `artifacts_at` line names. Written only by `corpus-triage --audit \
             authored-compile\n\
             # --bless` or `--bless-additions`; a merge conflict here is resolved by blessing \
             again at the\n\
             # merged tree, never by taking a side.\n",
        );
        let digests = self
            .cases
            .values()
            .flatten()
            .any(|cell| cell.artifact.is_some());
        match &self.build {
            Some(build) => {
                out.push_str(&build.line());
                out.push('\n');
            }
            None if digests => return Err("artifact digests need the build they are at".into()),
            None => {}
        }
        out.push_str(Self::HEADER);
        for profile in &self.profiles {
            let _ = write!(out, "\t{}", profile.label());
        }
        out.push('\n');
        for (case_id, cells) in &self.cases {
            if cells.len() != self.profiles.len() {
                return Err(format!(
                    "case {case_id} has {} cells for {} profiles",
                    cells.len(),
                    self.profiles.len()
                ));
            }
            out.push_str(case_id);
            for (cell, profile) in cells.iter().zip(&self.profiles) {
                let cell = cell
                    .holdable()
                    .map_err(|why| format!("case {case_id} on {}: {why}", profile.label()))?;
                let _ = write!(out, "\t{}", cell.text());
            }
            out.push('\n');
        }
        Ok(out)
    }

    /// The expectations `text` states: comment lines, the build of its artifact digests (an
    /// `artifacts_at` line, before the header; needed when a cell has a digest), a header naming
    /// each profile once, and one line per case (a lowercase SHA-256, once) with one cell
    /// ([`Cell::parse`]) per profile.
    pub fn parse(text: &str) -> Result<Expectations, String> {
        let mut lines = text
            .lines()
            .enumerate()
            .filter(|(_, line)| !line.starts_with('#') && !line.is_empty())
            .peekable();
        let build = match lines
            .peek()
            .and_then(|(_, line)| ArtifactBuild::parse(line))
        {
            Some(build) => {
                lines.next();
                Some(build?)
            }
            None => None,
        };
        let (_, header) = lines.next().ok_or("no header line")?;
        let mut columns = header.split('\t');
        if columns.next() != Some(Self::HEADER) {
            return Err(format!("the header does not start with {:?}", Self::HEADER));
        }
        let profiles = columns
            .map(ProfileStem::parse)
            .collect::<Result<Vec<_>, _>>()?;
        if profiles.is_empty() {
            return Err("the header names no profile".into());
        }
        if profiles.iter().collect::<BTreeSet<_>>().len() != profiles.len() {
            return Err("the header names a profile twice".into());
        }
        let mut cases = BTreeMap::new();
        for (number, line) in lines {
            let at = |why: String| format!("line {}: {why}", number + 1);
            let mut fields = line.split('\t');
            let case_id = fields.next().unwrap_or_default();
            crate::observation::validate_hash("case_id", case_id).map_err(at)?;
            let cells = fields
                .map(Cell::parse)
                .collect::<Result<Vec<_>, _>>()
                .map_err(at)?;
            if cells.len() != profiles.len() {
                return Err(at(format!(
                    "{} cells for {} profiles",
                    cells.len(),
                    profiles.len()
                )));
            }
            if build.is_none() && cells.iter().any(|cell| cell.artifact.is_some()) {
                return Err(at("an artifact digest, but no `artifacts_at` line".into()));
            }
            if cases.insert(case_id.to_string(), cells).is_some() {
                return Err(at(format!("case {case_id} is listed twice")));
            }
        }
        Ok(Expectations {
            build,
            profiles,
            cases,
        })
    }

    /// The expectations at `path`, or `None` when there is no file.
    pub fn load(path: &Path) -> Result<Option<Expectations>, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => Expectations::parse(&text)
                .map(Some)
                .map_err(|error| format!("{}: {error}", path.display())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(format!("read {}: {error}", path.display())),
        }
    }

    /// Write the expectations to `path` (a sibling file, then a rename).
    pub fn write(&self, path: &Path) -> Result<(), String> {
        let staged = path.with_extension("tsv.new");
        std::fs::write(&staged, self.render()?)
            .map_err(|error| format!("write {}: {error}", staged.display()))?;
        std::fs::rename(&staged, path)
            .map_err(|error| format!("rename {} into place: {error}", staged.display()))
    }
}

/// This build's artifact digests: each case's digest on each C3 venue's profile, as an
/// expectation file records them when its `artifacts_at` line names this build. They are what C3's
/// currency holds a venue's record of another build to (`c3::VenueKeys`). A file at another build
/// (a branch's copy of master's, until the gate blesses the merged tree), one without digests, and
/// no file all hold none, and the currency then holds every record to the build that recorded it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CurrentArtifacts {
    by_case: BTreeMap<String, BTreeMap<ProfileStem, String>>,
}

impl CurrentArtifacts {
    /// The digests `expectations` records when its build is `build`; none otherwise.
    pub fn of(expectations: &Expectations, build: &ArtifactBuild) -> CurrentArtifacts {
        if expectations.build.as_ref() != Some(build) {
            return CurrentArtifacts::default();
        }
        let by_case = expectations
            .cases
            .iter()
            .map(|(case_id, cells)| {
                let digests = expectations
                    .profiles
                    .iter()
                    .zip(cells)
                    .filter_map(|(stem, cell)| Some((stem.clone(), cell.artifact.clone()?)))
                    .collect::<BTreeMap<_, _>>();
                (case_id.clone(), digests)
            })
            .filter(|(_, digests)| !digests.is_empty())
            .collect();
        CurrentArtifacts { by_case }
    }

    /// The digests the expectation file of the corpus at `root` records at this build
    /// ([`ArtifactBuild::this`]): none when there is no file; an error when it does not parse.
    pub fn load(root: &Path) -> Result<CurrentArtifacts, String> {
        Ok(Expectations::load(&expectations_path(root))?
            .map(|expectations| CurrentArtifacts::of(&expectations, &ArtifactBuild::this()))
            .unwrap_or_default())
    }

    /// This build's digest of `case_id` on `stem`, when the file records one.
    pub fn get(&self, case_id: &str, stem: &ProfileStem) -> Option<&str> {
        self.by_case.get(case_id)?.get(stem).map(String::as_str)
    }
}

/// One difference between a run and the expectations.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Difference {
    /// A profile of the run the expectations do not name (a bless with other `--profile`s).
    AddedProfile(String),
    /// A profile the expectations name that the run did not compile.
    RemovedProfile(String),
    /// A committed case the expectations do not list.
    NewCase { case_id: String },
    /// A case the expectations list that is no longer committed.
    RemovedCase { case_id: String },
    /// A case whose cell on a profile changed: its outcome, or, at the build the file's digests
    /// were computed at, its artifact digest.
    Changed {
        case_id: String,
        profile: String,
        was: Cell,
        now: Cell,
    },
}

impl Difference {
    /// Whether the difference only adds to the expectations or drops what is gone (a new case, a
    /// removed case, an added profile), so no cell the file holds changes: what
    /// [`Mode::BlessAdditions`] writes.
    pub fn is_addition(&self) -> bool {
        matches!(
            self,
            Difference::NewCase { .. }
                | Difference::RemovedCase { .. }
                | Difference::AddedProfile(_)
        )
    }

    /// Whether the difference is a regression: a compiled case that is now refused.
    pub fn is_regression(&self) -> bool {
        matches!(
            self,
            Difference::Changed { was, now, .. }
                if was.outcome == CellOutcome::Compiled
                    && matches!(now.outcome, CellOutcome::Refused(_))
        )
    }

    /// The report's line for it, `prefix` naming the line's kind.
    pub fn line(&self, prefix: &str) -> String {
        match self {
            Difference::AddedProfile(label) => format!("{prefix}\tadded-profile\t{label}"),
            Difference::RemovedProfile(label) => format!("{prefix}\tremoved-profile\t{label}"),
            Difference::NewCase { case_id } => format!("{prefix}\tnew-case\t{case_id}"),
            Difference::RemovedCase { case_id } => format!("{prefix}\tremoved-case\t{case_id}"),
            Difference::Changed {
                case_id,
                profile,
                was,
                now,
            } => {
                let kind = match (&was.outcome, &now.outcome) {
                    (CellOutcome::Compiled, CellOutcome::Refused(_)) => "regressed",
                    (CellOutcome::Refused(_), CellOutcome::Compiled) => "compiles",
                    (was_outcome, now_outcome) if was_outcome == now_outcome => "artifact-changed",
                    _ => "refusal-changed",
                };
                format!(
                    "{prefix}\t{kind}\t{case_id}\tprofile={profile}\twas={}\tnow={}",
                    was.text(),
                    now.text()
                )
            }
        }
    }
}

/// The differences between `expected` and the run's `cells` over `profiles` (every cell an
/// outcome), in order. A cell's artifact digest differs only when the file's digests are at this
/// build (`same_build`): at another, a moved digest is no difference ([`moved_artifacts`]).
pub fn differences(
    expected: &Expectations,
    profiles: &[ProfileStem],
    cells: &BTreeMap<String, Vec<Cell>>,
    same_build: bool,
) -> Vec<Difference> {
    let mut out = Vec::new();
    for profile in profiles {
        if !expected.profiles.contains(profile) {
            out.push(Difference::AddedProfile(profile.label()));
        }
    }
    for profile in &expected.profiles {
        if !profiles.contains(profile) {
            out.push(Difference::RemovedProfile(profile.label()));
        }
    }
    let shared = profiles
        .iter()
        .enumerate()
        .filter_map(|(now, profile)| {
            let was = expected.profiles.iter().position(|p| p == profile)?;
            Some((profile.label(), was, now))
        })
        .collect::<Vec<_>>();
    for (case_id, now_cells) in cells {
        let Some(was_cells) = expected.cases.get(case_id) else {
            out.push(Difference::NewCase {
                case_id: case_id.clone(),
            });
            continue;
        };
        for (label, was, now) in &shared {
            let (was_cell, now_cell) = (&was_cells[*was], &now_cells[*now]);
            let differs = if same_build {
                was_cell != now_cell
            } else {
                !was_cell.same_outcome(now_cell)
            };
            if differs {
                out.push(Difference::Changed {
                    case_id: case_id.clone(),
                    profile: label.clone(),
                    was: was_cells[*was].clone(),
                    now: now_cells[*now].clone(),
                });
            }
        }
    }
    for case_id in expected.cases.keys() {
        if !cells.contains_key(case_id) {
            out.push(Difference::RemovedCase {
                case_id: case_id.clone(),
            });
        }
    }
    out
}

/// The cells of `profiles` whose artifact digest moved from `expected`'s while their outcome did
/// not: what a bless records.
pub fn moved_artifacts(
    expected: &Expectations,
    profiles: &[ProfileStem],
    cells: &BTreeMap<String, Vec<Cell>>,
) -> usize {
    let shared = profiles
        .iter()
        .enumerate()
        .filter_map(|(now, profile)| {
            Some((expected.profiles.iter().position(|p| p == profile)?, now))
        })
        .collect::<Vec<_>>();
    cells
        .iter()
        .filter_map(|(case_id, now_cells)| Some((expected.cases.get(case_id)?, now_cells)))
        .map(|(was_cells, now_cells)| {
            shared
                .iter()
                .filter(|&&(was, now)| {
                    let (was, now) = (&was_cells[was], &now_cells[now]);
                    was.same_outcome(now) && was.artifact != now.artifact
                })
                .count()
        })
        .sum()
}

/// Whether a run writes the expectation file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    /// Compare with the file and write nothing (`corpus-triage`'s default).
    #[default]
    Check,
    /// Rewrite the file from the run, on the file's profiles or the named ones (`--bless`):
    /// never with a failed cell.
    Bless,
    /// Write the file only with additions (`--bless-additions`): the file's profiles plus
    /// [`Options::required`]'s missing ones are compiled, and the run fails, writing nothing, on any
    /// failed cell and on any difference that is not [`Difference::is_addition`]. Otherwise new
    /// cases and profiles are added, removed cases dropped, and every artifact digest and the
    /// `artifacts_at` line recorded at this build (digests that moved with the product are no
    /// difference). The file is written only when that changes it.
    BlessAdditions,
}

/// The profiles of every C3 venue (`validation/cutover/c3-venues.json`), in list order, once each.
/// What [`Mode::BlessAdditions`] adds to the file when it lacks one.
pub fn venue_profiles(venues: &crate::c3::Venues) -> Vec<ProfileStem> {
    let mut profiles = Vec::new();
    for venue in &venues.venues {
        let wanted = ProfileStem {
            stem: venue.profile.clone(),
        };
        if !profiles.contains(&wanted) {
            profiles.push(wanted);
        }
    }
    profiles
}

/// What one census run is asked to do.
pub struct Options<'a> {
    /// The corpus root: its committed case shards and its expectation file.
    pub root: &'a Path,
    /// The index whose table caches cells.
    pub index: &'a Path,
    /// The expectation file ([`expectations_path`] of `root` by default).
    pub expectations: &'a Path,
    /// The profiles to compile: `None` for the expectation file's. Only a bless or a probe
    /// may name others.
    pub profiles: Option<Vec<ProfileStem>>,
    /// A probe: only these committed cases, each cell reported, compared with the expectation
    /// file's rows for them when the file exists; never a bless. `None` for every case.
    pub cases: Option<BTreeSet<String>>,
    /// Cases compiled at once (each in its own case worker, its compiles one at a time).
    pub jobs: usize,
    /// Whether and how the run writes the expectation file.
    pub mode: Mode,
    /// The profiles a [`Mode::BlessAdditions`] run adds when the file does not name them
    /// ([`venue_profiles`] for the census; a test's own): unused by the other modes.
    pub required: Vec<ProfileStem>,
    /// The profiles whose cells keep their artifact digest: the C3 venues' ([`venue_profiles`]) for
    /// the census, a test's own. Every other cell is written without one.
    pub artifacts: Vec<ProfileStem>,
}

/// One cell of a case that is a failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Failed {
    pub case_id: String,
    pub profile: String,
    pub failure: CellFailure,
}

/// A cell's measure, for the report's slowest and peaks: of one profile's compiles, or of a
/// case worker over all of a case's selected profiles (`profile` "*").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Measure {
    pub case_id: String,
    pub profile: String,
    pub value: u64,
}

/// A probe's cell ([`Options::cases`]): its outcome or failure, and what its compiles measured
/// (`None` when the cache held it).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProbedCell {
    pub case_id: String,
    pub profile: String,
    pub outcome: Result<CellOutcome, CellFailure>,
    pub compiled: Option<(usize, Cost)>,
}

impl ProbedCell {
    fn line(&self) -> String {
        let outcome = match &self.outcome {
            Ok(outcome) => outcome.as_str().to_string(),
            Err(failure) => format!("failed:{}:{}", failure.class.as_str(), failure.key),
        };
        let mut line = format!(
            "authored-compile-cell\t{}\tprofile={}\toutcome={outcome}",
            self.case_id, self.profile
        );
        match &self.compiled {
            None => line.push_str("\tcached=true"),
            Some((compiles, cost)) => {
                let bytes = |value: Option<u64>| value.map_or("none".into(), |b| b.to_string());
                let _ = write!(
                    line,
                    "\tcached=false\tcompiles={compiles}\telapsed_ms={}\tslowest_compile_ms={}\tpeak_worker_rss_bytes={}\tpeak_worker_allocator_bytes={}",
                    cost.elapsed.as_millis(),
                    cost.slowest.as_millis(),
                    bytes(cost.peak_resident_bytes),
                    bytes(cost.allocator_peak_bytes),
                );
            }
        }
        line
    }
}

/// What one census run did and found.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub profiles: Vec<String>,
    /// Committed cases.
    pub cases: usize,
    /// Cells (cases x profiles) the cache held under their current key.
    pub cached_cells: usize,
    /// Cases admitted and compiled by this run, and their cells.
    pub selected_cases: usize,
    pub selected_cells: usize,
    /// Compiles run (worker spawns).
    pub compiles: usize,
    /// Cells by outcome, over every case and profile.
    pub compiled: usize,
    pub refused: usize,
    pub failures: Vec<Failed>,
    pub differences: Vec<Difference>,
    /// Whether the expectation file's artifact digests are at this build ([`ArtifactBuild`]), and
    /// how many of its cells' digests this run moved while their outcome held.
    pub artifacts_at_this_build: bool,
    pub moved_artifacts: usize,
    /// The run's mode.
    pub mode: Mode,
    /// Whether the expectation file now holds this run's outcomes (a bless that met its mode's
    /// conditions), and whether the run wrote it (it differed).
    pub blessed: bool,
    pub written: bool,
    /// Of the cells compiled in this run: the slowest single compile, the slowest cell (its
    /// compiles summed), and the largest worker RSS and allocator peak.
    pub slowest_compile: Option<Measure>,
    pub slowest_cell: Option<Measure>,
    pub peak_resident: Option<Measure>,
    pub peak_allocator: Option<Measure>,
    /// Of the case workers this run spawned: the slowest, and the largest RSS and allocator peak.
    pub case_workers: usize,
    /// How the case workers found the private sources they read: by the index's byte ranges, or
    /// by parsing a whole shard (a source the index does not locate).
    pub sources: SourceReads,
    pub slowest_case: Option<Measure>,
    pub peak_case_resident: Option<Measure>,
    pub peak_case_allocator: Option<Measure>,
    /// A probe's cells, in case and profile order (empty unless [`Options::cases`]).
    pub probed: Vec<ProbedCell>,
    pub wall: Duration,
}

impl Report {
    /// Whether the run passes: no failure, and (unless it blessed) no difference.
    pub fn passed(&self) -> bool {
        self.failures.is_empty() && (self.blessed || self.differences.is_empty())
    }

    fn measure(slot: &mut Option<Measure>, case_id: &str, profile: &str, value: Option<u64>) {
        let Some(value) = value else {
            return;
        };
        if slot.as_ref().is_none_or(|known| value > known.value) {
            *slot = Some(Measure {
                case_id: case_id.into(),
                profile: profile.into(),
                value,
            });
        }
    }

    /// The report's lines: a summary, then every failure and difference.
    pub fn lines(&self) -> Vec<String> {
        let measure = |name: &str, slot: &Option<Measure>| match slot {
            Some(m) => format!(
                "\t{name}={}\t{name}_case={}\t{name}_profile={}",
                m.value, m.case_id, m.profile
            ),
            None => format!("\t{name}=none"),
        };
        let mut summary = format!(
            "authored-compile-summary\tcases={}\tprofiles={}\tcells={}\tcached_cells={}\tselected_cases={}\tselected_cells={}\tcompiles={}\tcompiled={}\trefused={}\tfailed={}\tdifferences={}\tregressions={}\tartifacts_at_this_build={}\tmoved_artifacts={}\tblessed={}\twritten={}\twall_ms={}",
            self.cases,
            self.profiles.len(),
            self.cases * self.profiles.len(),
            self.cached_cells,
            self.selected_cases,
            self.selected_cells,
            self.compiles,
            self.compiled,
            self.refused,
            self.failures.len(),
            self.differences.len(),
            self.differences.iter().filter(|d| d.is_regression()).count(),
            self.artifacts_at_this_build,
            self.moved_artifacts,
            self.blessed,
            self.written,
            self.wall.as_millis(),
        );
        summary.push_str(&measure("slowest_compile_ms", &self.slowest_compile));
        summary.push_str(&measure("slowest_cell_ms", &self.slowest_cell));
        summary.push_str(&measure("peak_worker_rss_bytes", &self.peak_resident));
        summary.push_str(&measure(
            "peak_worker_allocator_bytes",
            &self.peak_allocator,
        ));
        let _ = write!(
            summary,
            "\tcase_workers={}\tsources_by_range={}\tsources_by_shard={}",
            self.case_workers, self.sources.by_range, self.sources.by_shard
        );
        summary.push_str(&measure("slowest_case_ms", &self.slowest_case));
        summary.push_str(&measure(
            "peak_case_worker_rss_bytes",
            &self.peak_case_resident,
        ));
        summary.push_str(&measure(
            "peak_case_worker_allocator_bytes",
            &self.peak_case_allocator,
        ));
        let mut out = vec![
            format!("authored-compile-profiles\t{}", self.profiles.join("\t")),
            summary,
        ];
        out.extend(self.probed.iter().map(ProbedCell::line));
        for failed in &self.failures {
            out.push(format!(
                "authored-compile-failure\t{}\tprofile={}\tclass={}\tkey={}\tdetail={}",
                failed.case_id,
                failed.profile,
                failed.failure.class.as_str(),
                failed.failure.key,
                failed.failure.detail.replace(['\t', '\n'], " ")
            ));
        }
        let prefix = if self.blessed {
            "authored-compile-blessed"
        } else {
            "authored-compile-difference"
        };
        out.extend(self.differences.iter().map(|d| d.line(prefix)));
        if self.mode == Mode::BlessAdditions {
            let count =
                |kind: fn(&Difference) -> bool| self.differences.iter().filter(|d| kind(d)).count();
            let added_profiles = self
                .differences
                .iter()
                .filter_map(|d| match d {
                    Difference::AddedProfile(label) => Some(label.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            out.push(format!(
                "authored-compile-additions\tadded_cases={}\tremoved_cases={}\tadded_profiles={}\tchanged_cells={}\tblessed={}\twritten={}",
                count(|d| matches!(d, Difference::NewCase { .. })),
                count(|d| matches!(d, Difference::RemovedCase { .. })),
                if added_profiles.is_empty() {
                    "none".to_string()
                } else {
                    added_profiles.join(",")
                },
                count(|d| !d.is_addition()),
                self.blessed,
                self.written,
            ));
        }
        out
    }
}

/// One cached cell: its outcome and its artifact digest.
struct Cached {
    outcome: CellOutcome,
    artifact: Option<String>,
}

/// Run the census: compile every committed case on every profile whose cell the index does
/// not hold under its current key, in `options.jobs` cases at once through `compiler`, record
/// every cell that is an outcome, and compare the whole matrix with the expectations (or, with
/// `bless`, rewrite them when no cell failed).
pub fn run(options: &Options<'_>, compiler: &dyn CaseCompiler) -> Result<Report, String> {
    let started = Instant::now();
    let expected = Expectations::load(options.expectations)?;
    let this_build = ArtifactBuild::this();
    let same_build = expected
        .as_ref()
        .is_some_and(|expected| expected.build.as_ref() == Some(&this_build));
    if options.mode != Mode::Check && options.cases.is_some() {
        return Err("a bless rewrites every case's row; --case selects a probe's cases".into());
    }
    let stems = match (&options.profiles, &expected) {
        (Some(_), _) if options.mode == Mode::BlessAdditions => {
            return Err(
                "--bless-additions compiles the expectation file's profiles and adds every C3 \
                 venue's; --profile names a full bless's"
                    .into(),
            )
        }
        (Some(_), _) if options.mode == Mode::Check && options.cases.is_none() => {
            return Err(
                "--profile names the profiles of a bless or a probe (--case); a check \
                        compiles the expectation file's"
                    .into(),
            )
        }
        (Some(profiles), _) => profiles.clone(),
        (None, Some(expected)) if options.mode == Mode::BlessAdditions => {
            let mut stems = expected.profiles.clone();
            for stem in &options.required {
                if !stems.contains(stem) {
                    stems.push(stem.clone());
                }
            }
            stems
        }
        (None, Some(expected)) => expected.profiles.clone(),
        (None, None) => {
            return Err(format!(
                "{} does not exist: bless it first with --bless and a --profile per profile",
                options.expectations.display()
            ))
        }
    };
    if stems.is_empty() || stems.iter().collect::<BTreeSet<_>>().len() != stems.len() {
        return Err("name each profile once, and at least one".into());
    }
    let profiles = stems
        .iter()
        .map(CensusProfile::load)
        .collect::<Result<Vec<_>, _>>()?;
    let labels = stems.iter().map(ProfileStem::label).collect::<Vec<_>>();
    // A cell keeps its artifact digest on the profiles that record them.
    let digested = stems
        .iter()
        .map(|stem| options.artifacts.contains(stem))
        .collect::<Vec<_>>();
    let mut cases = CorpusStore::new(options.root).read_all_cases()?;
    // A probe compiles its committed cases and compares only their rows.
    let mut expected = expected;
    if let Some(selected) = &options.cases {
        let committed = cases
            .iter()
            .map(|case| case.case_id.as_str())
            .collect::<BTreeSet<_>>();
        let unknown = selected
            .iter()
            .filter(|id| !committed.contains(id.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        if !unknown.is_empty() {
            return Err(format!("no committed case {}", unknown.join(", ")));
        }
        cases.retain(|case| selected.contains(&case.case_id));
        if let Some(expected) = &mut expected {
            expected
                .cases
                .retain(|case_id, _| selected.contains(case_id));
        }
    }

    let connection = open(options.index)?;
    let mut cached: Vec<BTreeMap<String, Cached>> = Vec::with_capacity(profiles.len());
    for (profile, label) in profiles.iter().zip(&labels) {
        cached.push(read_cached(&connection, label, &profile.key)?);
    }

    // Each case's identity is its content's id; a manifest whose id is not it is a failure.
    let mut report = Report {
        profiles: labels.clone(),
        cases: cases.len(),
        mode: options.mode,
        ..Report::default()
    };
    let mut cells: BTreeMap<String, Vec<Option<Result<Cell, CellFailure>>>> = BTreeMap::new();
    let mut work = Vec::new();
    for (position, case) in cases.iter().enumerate() {
        let row = cells
            .entry(case.case_id.clone())
            .or_insert_with(|| vec![None; profiles.len()]);
        match case.computed_case_id() {
            Ok(computed) if computed == case.case_id => {}
            other => {
                let detail = match other {
                    Ok(computed) => format!("the manifest's id is not its content's ({computed})"),
                    Err(error) => error,
                };
                for cell in row.iter_mut() {
                    *cell = Some(Err(CellFailure {
                        class: FailureClass::Case,
                        key: "case_id".into(),
                        detail: detail.clone(),
                    }));
                }
                continue;
            }
        }
        let mut uncached = Vec::new();
        for (index, profile_cache) in cached.iter().enumerate() {
            match profile_cache.get(&case.case_id) {
                Some(hit) => {
                    row[index] = Some(Ok(Cell::of(
                        hit.outcome.clone(),
                        hit.artifact.clone(),
                        digested[index],
                    )));
                    report.cached_cells += 1;
                    if options.cases.is_some() {
                        report.probed.push(ProbedCell {
                            case_id: case.case_id.clone(),
                            profile: labels[index].clone(),
                            outcome: Ok(hit.outcome.clone()),
                            compiled: None,
                        });
                    }
                }
                None => uncached.push(index),
            }
        }
        if !uncached.is_empty() {
            report.selected_cells += uncached.len();
            work.push((position, uncached));
        }
    }
    report.selected_cases = work.len();

    let total = work.len();
    let mut done = 0usize;
    let mut batch = Vec::with_capacity(CACHE_BATCH);
    parallel(
        options.jobs.min(total.max(1)),
        1,
        |send| {
            for item in work {
                send(item)?;
            }
            Ok(())
        },
        |(position, selected): (usize, Vec<usize>)| {
            let chosen = selected.iter().map(|&i| &profiles[i]).collect::<Vec<_>>();
            let results = compiler.compile(&cases[position], &chosen);
            (position, selected, results)
        },
        |(position, selected, results)| {
            let CaseResult {
                cells: results,
                worker,
                sources,
            } = results;
            if let Some(sources) = sources {
                report.sources.by_range += sources.by_range;
                report.sources.by_shard += sources.by_shard;
            }
            if results.len() != selected.len() {
                return Err(format!(
                    "the compiler returned {} cells for {} profiles",
                    results.len(),
                    selected.len()
                ));
            }
            let case_id = &cases[position].case_id;
            if let Some(worker) = worker {
                report.case_workers += 1;
                Report::measure(
                    &mut report.slowest_case,
                    case_id,
                    "*",
                    Some(worker.elapsed.as_millis() as u64),
                );
                Report::measure(
                    &mut report.peak_case_resident,
                    case_id,
                    "*",
                    worker.peak_resident_bytes,
                );
                Report::measure(
                    &mut report.peak_case_allocator,
                    case_id,
                    "*",
                    worker.allocator_peak_bytes,
                );
            }
            let row = cells
                .get_mut(case_id)
                .ok_or_else(|| format!("case {case_id} has no row"))?;
            for (index, result) in selected.into_iter().zip(results) {
                let label = &labels[index];
                report.compiles += result.compiles;
                if result.compiles > 0 {
                    Report::measure(
                        &mut report.slowest_compile,
                        case_id,
                        label,
                        Some(result.cost.slowest.as_millis() as u64),
                    );
                    Report::measure(
                        &mut report.slowest_cell,
                        case_id,
                        label,
                        Some(result.cost.elapsed.as_millis() as u64),
                    );
                    Report::measure(
                        &mut report.peak_resident,
                        case_id,
                        label,
                        result.cost.peak_resident_bytes,
                    );
                    Report::measure(
                        &mut report.peak_allocator,
                        case_id,
                        label,
                        result.cost.allocator_peak_bytes,
                    );
                }
                if options.cases.is_some() {
                    report.probed.push(ProbedCell {
                        case_id: case_id.clone(),
                        profile: label.clone(),
                        outcome: result.outcome.clone(),
                        compiled: Some((result.compiles, result.cost)),
                    });
                }
                if let Ok(outcome) = &result.outcome {
                    batch.push(Recorded {
                        case_id: case_id.clone(),
                        profile: label.clone(),
                        key: profiles[index].key.clone(),
                        outcome: outcome.clone(),
                        result: result.clone(),
                    });
                }
                let artifact = result.artifact;
                row[index] = Some(
                    result
                        .outcome
                        .map(|outcome| Cell::of(outcome, artifact, digested[index])),
                );
            }
            if batch.len() >= CACHE_BATCH {
                write_cached(&connection, &batch)?;
                batch.clear();
            }
            done += 1;
            if done.is_multiple_of(100) || done == total {
                eprintln!(
                    "# authored-compile progress={done}/{total} jobs={}",
                    options.jobs
                );
            }
            Ok(())
        },
    )?;
    write_cached(&connection, &batch)?;

    let mut outcomes = BTreeMap::new();
    for (case_id, row) in cells {
        let mut outcome_row = Vec::with_capacity(row.len());
        for (index, cell) in row.into_iter().enumerate() {
            match cell.ok_or_else(|| format!("case {case_id} has no cell on {}", labels[index]))? {
                Ok(cell) => {
                    match cell.outcome {
                        CellOutcome::Compiled => report.compiled += 1,
                        CellOutcome::Refused(_) => report.refused += 1,
                    }
                    outcome_row.push(cell);
                }
                Err(failure) => report.failures.push(Failed {
                    case_id: case_id.clone(),
                    profile: labels[index].clone(),
                    failure,
                }),
            }
        }
        if outcome_row.len() == labels.len() {
            outcomes.insert(case_id, outcome_row);
        }
    }
    report.probed.sort_by(|a, b| {
        (&a.case_id, labels.iter().position(|l| *l == a.profile))
            .cmp(&(&b.case_id, labels.iter().position(|l| *l == b.profile)))
    });
    // A probe with no expectation file reports its cells and compares nothing.
    report.artifacts_at_this_build = same_build;
    report.differences = match (&expected, &options.cases) {
        (None, Some(_)) => Vec::new(),
        (expected, _) => differences(
            &expected.clone().unwrap_or_default(),
            &stems,
            &outcomes,
            same_build,
        ),
    };
    if let Some(expected) = &expected {
        report.moved_artifacts = moved_artifacts(expected, &stems, &outcomes);
    }
    // A case with a failed cell has no row to compare: it is a failure, not a removed case.
    let failed = report
        .failures
        .iter()
        .map(|failed| failed.case_id.clone())
        .collect::<BTreeSet<_>>();
    report.differences.retain(|difference| {
        !matches!(difference, Difference::RemovedCase { case_id } if failed.contains(case_id))
    });
    let bless = match options.mode {
        Mode::Check => false,
        Mode::Bless => report.failures.is_empty(),
        Mode::BlessAdditions => {
            report.failures.is_empty() && report.differences.iter().all(Difference::is_addition)
        }
    };
    if bless {
        let digests = outcomes
            .values()
            .flatten()
            .any(|cell| cell.artifact.is_some());
        let blessed = Expectations {
            build: digests.then_some(this_build),
            profiles: stems,
            cases: outcomes,
        };
        let text = blessed.render()?;
        report.written = std::fs::read_to_string(options.expectations).ok() != Some(text);
        if report.written {
            blessed.write(options.expectations)?;
        }
        report.blessed = true;
    }
    report.wall = started.elapsed();
    Ok(report)
}

/// A cell to record.
struct Recorded {
    case_id: String,
    profile: String,
    key: String,
    outcome: CellOutcome,
    result: CellResult,
}

fn open(index: &Path) -> Result<Connection, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    // A table of census version 6 or older holds an uncovered count per cell (`host_replayed`);
    // its cells are under older keys, so none is read, and it is made again.
    let old = connection
        .prepare(&format!(
            "SELECT 1 FROM pragma_table_info('{TABLE}') WHERE name='host_replayed'"
        ))
        .and_then(|mut statement| statement.exists([]))
        .map_err(|error| format!("read the authored compile census table's columns: {error}"))?;
    if old {
        connection
            .execute_batch(&format!("DROP TABLE {TABLE};"))
            .map_err(|error| format!("drop the older authored compile census table: {error}"))?;
    }
    connection
        .execute_batch(&format!(
            "CREATE TABLE IF NOT EXISTS {TABLE} (
               case_id TEXT NOT NULL,
               profile TEXT NOT NULL,
               key TEXT NOT NULL,
               outcome TEXT NOT NULL,
               compiles INTEGER NOT NULL,
               elapsed_ms INTEGER NOT NULL,
               peak_resident_bytes INTEGER,
               allocator_peak_bytes INTEGER,
               artifact TEXT,
               PRIMARY KEY(case_id, profile)
             );"
        ))
        .map_err(|error| format!("create the authored compile census table: {error}"))?;
    Ok(connection)
}

/// The cells the index holds for profile `label` under `key`, by case id.
fn read_cached(
    connection: &Connection,
    label: &str,
    key: &str,
) -> Result<BTreeMap<String, Cached>, String> {
    let mut statement = connection
        .prepare(&format!(
            "SELECT case_id, outcome, artifact FROM {TABLE} \
             WHERE profile=?1 AND key=?2"
        ))
        .map_err(|error| format!("prepare the authored compile census selection: {error}"))?;
    let rows = statement
        .query_map(params![label, key], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, Option<String>>(2)?,
            ))
        })
        .map_err(|error| format!("select authored compile census cells: {error}"))?;
    let mut out = BTreeMap::new();
    for row in rows {
        let (case_id, outcome, artifact) =
            row.map_err(|error| format!("read an authored compile census cell: {error}"))?;
        out.insert(
            case_id,
            Cached {
                outcome: CellOutcome::parse(&outcome)?,
                artifact,
            },
        );
    }
    Ok(out)
}

/// Record `cells`, each replacing the case's cell on its profile (one per case and
/// profile: a key's cells replace an older key's, so the table stays bounded).
fn write_cached(connection: &Connection, cells: &[Recorded]) -> Result<(), String> {
    if cells.is_empty() {
        return Ok(());
    }
    let transaction = connection
        .unchecked_transaction()
        .map_err(|error| format!("begin an authored compile census write: {error}"))?;
    {
        let mut statement = transaction
            .prepare(&format!(
                "INSERT OR REPLACE INTO {TABLE} (case_id, profile, key, outcome, compiles, \
                 elapsed_ms, peak_resident_bytes, allocator_peak_bytes, artifact) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)"
            ))
            .map_err(|error| format!("prepare an authored compile census write: {error}"))?;
        for cell in cells {
            let to_i64 = |value: u64| i64::try_from(value).unwrap_or(i64::MAX);
            statement
                .execute(params![
                    cell.case_id,
                    cell.profile,
                    cell.key,
                    cell.outcome.as_str(),
                    to_i64(cell.result.compiles as u64),
                    to_i64(cell.result.cost.elapsed.as_millis() as u64),
                    cell.result.cost.peak_resident_bytes.map(to_i64),
                    cell.result.cost.allocator_peak_bytes.map(to_i64),
                    cell.result.artifact,
                ])
                .map_err(|error| format!("record an authored compile census cell: {error}"))?;
        }
    }
    transaction
        .commit()
        .map_err(|error| format!("commit authored compile census cells: {error}"))
}
