//! Batched engine runs of the Metal-qualified authored cases on a Vulkan device
//! (`corpus-vulkan --batch`), or on MoltenVK (`corpus-moltenvk --batch`); and, on the same driver,
//! the Metal oracle's qualifications (`corpus-metal --batch`, [`metal`]): one [`Pass::Metal`] run
//! per case, which records the case's Metal row, on the GPU only, with the same job stores,
//! done-list, build record, tripwire, and stops.
//!
//! The batch selects every committed case with a current Metal observation in one Metal
//! environment ([`qualified_cases`]: the row the executor compares against, by
//! [`MetalObservation::dependency_matches`] under the current oracle) and runs it, in each
//! requested robustness leg, through its backend's executor (`corpus-vulkan` or
//! `corpus-moltenvk`, the engine) with `--device-type TYPE`: one process per (case, leg), exactly
//! the run a single-case invocation makes, so a row or a refusal it records is the executor's own.
//! The batch never runs Metal: it reads the committed Metal rows. With a profile to restrict to,
//! every run is the executor's `--profiles-layer P` run, stored in the `ID@engine@P` slots.
//!
//! - **Engine only.** The executors run only the engine (the old engine's candidate is
//!   deleted, cutover X2a).
//! - **A CPU device, or one GPU run at a time** ([`BatchDevice`]). CPU runs are concurrent. On a
//!   GPU the batch runs one executor at a time, which the GPU safety protocol requires: it checks
//!   the host's hang tripwire before every run and stops at a tripped one, and it stops after a
//!   run whose device wait timed out ([`HANG_EXIT_CODE`]), whose device worker crashed, or that
//!   outlived the per-run timeout, writing the tripwire for that last itself (its submission may
//!   still be running; the executor's own protocol writes it for the other two, and the batch
//!   writes it too when it is absent). A human follows the suspected-hang protocol in
//!   `docs/VALIDATION.md` before the next run. A GPU batch takes the GPU as its own: whatever
//!   schedules it holds the GPU for the batch, so a hang names that run's case alone.
//! - **Bounded.** At most `jobs` executors run at once (the bin defaults to the available
//!   parallelism on a CPU device, which in a container is its CPU share, and to one on a GPU), each
//!   under a per-run timeout; at most `limit` cases run per invocation. A run's executor starts its
//!   own session, and when the run ends (or times out) every process left in that session, its
//!   compile and device workers included, is killed, and the scratch directory it was given
//!   (`TMPDIR`) is removed.
//! - **Resumable, by one build.** Every finished run appends one line to the out directory's
//!   done-list ([`DONE_LIST`]); a later invocation skips each (case, leg) listed there, so a rerun
//!   continues where an interrupted one stopped. Delete a line to run it again. A last line without
//!   its newline (an append cut short) is not a finished run: the readers drop it and
//!   [`open_lines`] truncates it before the next append. The out directory records the build and
//!   venue its runs belong to ([`BatchBuild`], [`BUILD`]): a batch refuses to resume one another
//!   build or venue started, or one whose runs predate that record, so no done-list of an older
//!   build passes for this build's finished runs.
//! - **Scratch only.** Each job runs in its own store under the temporary directory: links to the
//!   corpus's cases, reviews, Metal rows, the other backend's rows, and private sources, and its
//!   own empty candidate and refusal shards of its backend, so concurrent runs never share a shard
//!   and the corpus is never written. A row a run recorded is appended verbatim to [`ROWS`], and a refusal it recorded
//!   (`crate::refusal`) to [`REFUSALS`]. [`install`] writes the evidence of each slot's last run
//!   into a store through [`CorpusStore::upsert_candidate`] and [`CorpusStore::upsert_refusal`],
//!   the executor's own writes.
//! - **Report.** [`report`] reads the done-list, rows, and refusals back beside the committed cases
//!   and Metal rows: per leg, [`family`], and [`scope`], the runs by [`Class`], and every run that
//!   did not match with its detail (for a mismatch, the 32-bit words that differ from Metal's
//!   output; for a refusal, the class of its recorded refusal).

use crate::candidate::engine::{engine_slot, is_robustness_off_slot, ENGINE_EXECUTOR_ABI};
use crate::candidate::exit_status;
use crate::case::{AuthoredCase, Stage};
use crate::coverage::Leg;
use crate::gpu::tripwire::{write_record, HangKind, HangRecord, TripwirePaths};
use crate::gpu::wait::{SubmissionBackend, HANG_EXIT_CODE};
use crate::gpu::DeviceType;
use crate::metal::{ORACLE_ABI, REFERENCE_GAP_EXIT};
use crate::observation::{
    Backend, CandidateObservation, CandidateStatus, MetalObservation, MetalStatus,
    TRANSLATOR_FINGERPRINT,
};
use crate::refusal::RefusalObservation;
use crate::source::shard_name_for_hash;
use crate::store::{refusal_directory, CorpusStore};
use crate::ScratchDir;
use base64::Engine as _;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// The out directory's done-list: one tab-separated line per finished run
/// (`case_id leg outcome ended seconds detail`), appended as each run ends.
pub const DONE_LIST: &str = "done.tsv";
/// The out directory's rows: each row a run recorded, its store line verbatim, appended before
/// the run's done-list line.
pub const ROWS: &str = "rows.jsonl";
/// The out directory's refusals: each refusal a run recorded (`crate::refusal`), its store line
/// verbatim, appended before the run's done-list line.
pub const REFUSALS: &str = "refusals.jsonl";
/// The out directory's logs: the executor's stdout and stderr of each run that did not match.
pub const LOGS: &str = "logs";
/// The out directory's build record ([`BatchBuild`]).
pub const BUILD: &str = "build.json";

/// Bytes of a run's stdout and stderr kept in its log (each stream's tail).
const LOG_TAIL_BYTES: usize = 16 * 1024;
/// Characters of a run's detail kept in its done-list line.
const DETAIL_CHARS: usize = 400;

/// What one run of the executor came to, from its exit and the row it recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Outcome {
    /// Exit 0 with a matching row.
    Match,
    /// A Metal run's exit 0 with a qualified Metal row.
    Qualified,
    /// A Metal run's [`REFERENCE_GAP_EXIT`] with a reference-gap Metal row: the reference device
    /// cannot run the case, by Metal's own answer.
    ReferenceGap,
    /// Exit 1 with a mismatching row.
    Mismatch,
    /// `not_supported{...}`: the product refused the case, the device does not enable what it
    /// needs, or the executor does not implement it.
    NotSupported,
    /// `execution_refused{...}`: refused by policy.
    Refused,
    /// `device_crash{...}`: the device worker died by a signal.
    DeviceCrash,
    /// `needs_metal_qualification{...}`: no current Metal row, although the batch selected the
    /// case for one (a tooling disagreement).
    NeedsMetal,
    /// `venue_mismatch{...}`: the device differs from the venue's committed profile. Nothing was
    /// recorded, and the batch stops ([`stops_the_batch`]).
    VenueMismatch,
    /// A device wait timed out ([`HANG_EXIT_CODE`]).
    WaitTimeout,
    /// The run outlived the batch's per-run timeout and was killed.
    Timeout,
    /// Any other exit, or an exit its row does not agree with.
    Failed,
}

const OUTCOMES: [Outcome; 12] = [
    Outcome::Match,
    Outcome::Qualified,
    Outcome::ReferenceGap,
    Outcome::Mismatch,
    Outcome::NotSupported,
    Outcome::Refused,
    Outcome::DeviceCrash,
    Outcome::NeedsMetal,
    Outcome::VenueMismatch,
    Outcome::WaitTimeout,
    Outcome::Timeout,
    Outcome::Failed,
];

/// The report's four classes of [`Outcome`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    /// A match, or a Metal run that qualified.
    Match,
    Mismatch,
    /// The case did not run: a typed refusal.
    Refusal,
    /// The run did not finish as evidence: a crash, a timeout, or a tooling failure.
    Failure,
}

const CLASSES: [Class; 4] = [
    Class::Match,
    Class::Mismatch,
    Class::Refusal,
    Class::Failure,
];

impl Outcome {
    pub fn label(self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::Qualified => "qualified",
            Self::ReferenceGap => "reference_gap",
            Self::Mismatch => "mismatch",
            Self::NotSupported => "not_supported",
            Self::Refused => "refused",
            Self::DeviceCrash => "device_crash",
            Self::NeedsMetal => "needs_metal",
            Self::VenueMismatch => "venue_mismatch",
            Self::WaitTimeout => "wait_timeout",
            Self::Timeout => "timeout",
            Self::Failed => "failed",
        }
    }

    pub fn parse(label: &str) -> Option<Self> {
        OUTCOMES
            .into_iter()
            .find(|outcome| outcome.label() == label)
    }

    pub fn class(self) -> Class {
        match self {
            Self::Match | Self::Qualified => Class::Match,
            Self::Mismatch => Class::Mismatch,
            Self::NotSupported | Self::Refused | Self::ReferenceGap => Class::Refusal,
            Self::DeviceCrash
            | Self::NeedsMetal
            | Self::VenueMismatch
            | Self::WaitTimeout
            | Self::Timeout
            | Self::Failed => Class::Failure,
        }
    }
}

impl Class {
    pub fn label(self) -> &'static str {
        match self {
            Self::Match => "match",
            Self::Mismatch => "mismatch",
            Self::Refusal => "refusal",
            Self::Failure => "failure",
        }
    }
}

/// What one run of a case is: a robustness leg of a candidate backend's executor, or the Metal
/// oracle's qualification, which has no legs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Pass {
    Leg(Leg),
    Metal,
}

impl Pass {
    /// The done-list's spelling: the leg's ([`leg_label`]), or `metal`.
    pub fn label(self) -> &'static str {
        match self {
            Self::Leg(leg) => leg_label(leg),
            Self::Metal => "metal",
        }
    }

    pub fn parse(label: &str) -> Result<Self, String> {
        match label {
            "metal" => Ok(Self::Metal),
            _ => parse_leg(label).map(Self::Leg),
        }
    }
}

/// The command-line spelling of a robustness leg.
pub fn leg_label(leg: Leg) -> &'static str {
    match leg {
        Leg::Stated => "stated",
        Leg::RobustnessOff => "robustness-off",
    }
}

pub fn parse_leg(label: &str) -> Result<Leg, String> {
    match label {
        "stated" => Ok(Leg::Stated),
        "robustness-off" => Ok(Leg::RobustnessOff),
        _ => Err(format!("unknown leg {label:?} (stated or robustness-off)")),
    }
}

/// How a run's executor ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ended {
    Code(i32),
    Signal(i32),
    /// The batch killed it at its timeout.
    TimedOut,
}

impl Ended {
    fn of(status: ExitStatus) -> Self {
        if let Some(code) = status.code() {
            return Self::Code(code);
        }
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt as _;
            if let Some(signal) = status.signal() {
                return Self::Signal(signal);
            }
        }
        Self::Code(-1)
    }

    fn label(self) -> String {
        match self {
            Self::Code(code) => format!("exit={code}"),
            Self::Signal(signal) => format!("signal={signal}"),
            Self::TimedOut => "timeout".into(),
        }
    }
}

/// The outcome of a run that ended as `ended` with `row` (the status of the row it recorded in
/// its slot, if any) and `refused` (whether it recorded a refusal there). Only a match or a
/// mismatch records a row, and only one whose status agrees with the exit; only a typed refusal
/// (`not_supported`) records a refusal, and it may not (one before a device opened records
/// none): anything else is [`Outcome::Failed`].
pub fn classify(ended: Ended, row: Option<CandidateStatus>, refused: bool) -> Outcome {
    match (ended, row, refused) {
        (Ended::TimedOut, _, _) => Outcome::Timeout,
        (Ended::Code(0), Some(CandidateStatus::Match), false) => Outcome::Match,
        (Ended::Code(1), Some(CandidateStatus::Mismatch), false) => Outcome::Mismatch,
        (Ended::Code(exit_status::NOT_SUPPORTED), None, _) => Outcome::NotSupported,
        (Ended::Code(exit_status::REFUSED), None, false) => Outcome::Refused,
        (Ended::Code(exit_status::DEVICE_CRASH), None, false) => Outcome::DeviceCrash,
        (Ended::Code(exit_status::NEEDS_METAL), None, false) => Outcome::NeedsMetal,
        (Ended::Code(exit_status::VENUE_MISMATCH), None, false) => Outcome::VenueMismatch,
        (Ended::Code(HANG_EXIT_CODE), None, false) => Outcome::WaitTimeout,
        _ => Outcome::Failed,
    }
}

/// The outcome of a Metal run (`corpus-metal`) that ended as `ended` with `row` (the status of the
/// Metal row it recorded in its slot, if any). A qualification records a qualified row with exit 0,
/// and a reference gap a reference-gap row with [`REFERENCE_GAP_EXIT`]; nothing else records one. A
/// refusal by policy exits [`exit_status::REFUSED`], and a timed-out device wait
/// [`HANG_EXIT_CODE`]. The Metal process holds its device itself, so its death by a signal is the
/// device crash a candidate's executor reports for its device worker, and stops a GPU batch
/// ([`stops_a_gpu`]). Anything else is [`Outcome::Failed`].
pub fn classify_metal(ended: Ended, row: Option<MetalStatus>) -> Outcome {
    match (ended, row) {
        (Ended::TimedOut, _) => Outcome::Timeout,
        (Ended::Code(0), Some(MetalStatus::Qualified)) => Outcome::Qualified,
        (Ended::Code(REFERENCE_GAP_EXIT), Some(MetalStatus::ReferenceGap(_))) => {
            Outcome::ReferenceGap
        }
        (Ended::Code(exit_status::REFUSED), None) => Outcome::Refused,
        (Ended::Code(HANG_EXIT_CODE), None) => Outcome::WaitTimeout,
        (Ended::Signal(_), _) => Outcome::DeviceCrash,
        _ => Outcome::Failed,
    }
}

/// One finished run, as its done-list line records it.
#[derive(Clone, Debug, PartialEq)]
pub struct DoneRun {
    pub case_id: String,
    pub pass: Pass,
    pub outcome: Outcome,
    /// [`Ended`]'s label (`exit=N`, `signal=N`, `timeout`).
    pub ended: String,
    pub seconds: f64,
    /// The executor's verdict line or last error line; empty for a match.
    pub detail: String,
}

impl DoneRun {
    pub fn line(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{:.3}\t{}\n",
            self.case_id,
            self.pass.label(),
            self.outcome.label(),
            self.ended,
            self.seconds,
            self.detail
        )
    }

    pub fn parse(line: &str) -> Result<Self, String> {
        let fields = line.splitn(6, '\t').collect::<Vec<_>>();
        let [case_id, pass, outcome, ended, seconds, detail] = fields[..] else {
            return Err(format!(
                "done-list line has {} fields: {line:?}",
                fields.len()
            ));
        };
        crate::observation::validate_hash("case_id", case_id)?;
        Ok(Self {
            case_id: case_id.into(),
            pass: Pass::parse(pass)?,
            outcome: Outcome::parse(outcome)
                .ok_or_else(|| format!("unknown outcome {outcome:?} in {line:?}"))?,
            ended: ended.into(),
            seconds: seconds
                .parse()
                .map_err(|error| format!("bad seconds {seconds:?} in {line:?}: {error}"))?,
            detail: detail.into(),
        })
    }
}

/// The runs a done-list holds, in file order. A missing file holds none. A last line without its
/// newline is a run whose append was cut short: it is not done, and is dropped.
pub fn read_done_list(path: &Path) -> Result<Vec<DoneRun>, String> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("read {}: {error}", path.display())),
    };
    let complete = match text.rfind('\n') {
        Some(end) => &text[..end],
        None => "",
    };
    complete
        .split('\n')
        .filter(|line| !line.is_empty())
        .map(DoneRun::parse)
        .collect()
}

/// The rows a rows file holds, in file order (a missing file holds none). A last line without
/// its newline is a row whose append was cut short, and is dropped.
pub fn read_rows(path: &Path) -> Result<Vec<CandidateObservation>, String> {
    read_records(path, "row")
}

/// The refusals a refusals file holds, read as [`read_rows`] reads rows.
pub fn read_refusals(path: &Path) -> Result<Vec<RefusalObservation>, String> {
    read_records(path, "refusal")
}

/// What an out directory's runs left: its done-list, rows, and refusals, each in file order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BatchOutput {
    pub done: Vec<DoneRun>,
    pub rows: Vec<CandidateObservation>,
    pub refusals: Vec<RefusalObservation>,
}

impl BatchOutput {
    /// The out directory `out`'s [`DONE_LIST`], [`ROWS`], and [`REFUSALS`] (missing files hold
    /// none).
    pub fn read(out: &Path) -> Result<Self, String> {
        Ok(Self {
            done: read_done_list(&out.join(DONE_LIST))?,
            rows: read_rows(&out.join(ROWS))?,
            refusals: read_refusals(&out.join(REFUSALS))?,
        })
    }
}

fn read_records<T: DeserializeOwned>(path: &Path, what: &str) -> Result<Vec<T>, String> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("read {}: {error}", path.display())),
    };
    let complete = match text.rfind('\n') {
        Some(end) => &text[..end],
        None => "",
    };
    complete
        .split('\n')
        .filter(|line| !line.is_empty())
        .map(|line| {
            serde_json::from_str(line)
                .map_err(|error| format!("{}: bad {what}: {error}", path.display()))
        })
        .collect()
}

/// Open the out directory's line file `path` to append to, creating it when missing. A last line
/// without its newline (an append cut short, which the readers drop) is truncated first, so the
/// next line starts on a line of its own instead of extending the cut one.
pub fn open_lines(path: &Path) -> Result<File, String> {
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("open {}: {error}", path.display()))?;
    let complete = complete_length(&mut file)
        .and_then(|complete| {
            if complete < file.metadata()?.len() {
                file.set_len(complete)?;
            }
            Ok(())
        })
        .map_err(|error| format!("drop the cut line of {}: {error}", path.display()));
    complete.map(|()| file)
}

/// The length of `file` through its last newline (0 without one), read back from its end.
fn complete_length(file: &mut File) -> std::io::Result<u64> {
    const BLOCK: u64 = 64 * 1024;
    let mut end = file.metadata()?.len();
    let mut buffer = vec![0; BLOCK as usize];
    while end > 0 {
        let start = end.saturating_sub(BLOCK);
        let block = &mut buffer[..(end - start) as usize];
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(block)?;
        if let Some(at) = block.iter().rposition(|&byte| byte == b'\n') {
            return Ok(start + at as u64 + 1);
        }
        end = start;
    }
    Ok(0)
}

/// The cases with a current Metal row in `metal_environment_id` (the row
/// [`MetalObservation::current`] returns), in case-ID order.
pub fn qualified_cases(
    cases: Vec<AuthoredCase>,
    metals: &[MetalObservation],
    metal_environment_id: &str,
) -> Vec<AuthoredCase> {
    let mut first = HashMap::<&str, &MetalObservation>::new();
    for metal in metals
        .iter()
        .filter(|metal| metal.environment_id == metal_environment_id)
    {
        first.entry(metal.case_id.as_str()).or_insert(metal);
    }
    let mut selected = cases
        .into_iter()
        .filter(|case| {
            first
                .get(case.case_id.as_str())
                .is_some_and(|metal| metal.dependency_matches(case, ORACLE_ABI))
        })
        .collect::<Vec<_>>();
    selected.sort_by(|left, right| left.case_id.cmp(&right.case_id));
    selected
}

/// The cases of `qualified` (what [`qualified_cases`] selected in `metal_environment_id`) that
/// `case_ids` requests, in their order; all of them when none is requested. A requested case that
/// `qualified` lacks (it has no current Metal row there, or no case) is an error naming it, so
/// neither a batch nor its report silently narrows to fewer cases than were asked for.
pub fn requested_cases(
    mut qualified: Vec<AuthoredCase>,
    case_ids: &[String],
    metal_environment_id: &str,
) -> Result<Vec<AuthoredCase>, String> {
    if case_ids.is_empty() {
        return Ok(qualified);
    }
    let requested = case_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
    qualified.retain(|case| requested.contains(case.case_id.as_str()));
    let found = qualified
        .iter()
        .map(|case| case.case_id.as_str())
        .collect::<BTreeSet<_>>();
    let missing = requested.difference(&found).copied().collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "no current Metal row in {metal_environment_id} (or no case) for: {}",
            missing.join(", ")
        ));
    }
    Ok(qualified)
}

/// One (case, pass) to run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Run {
    pub case_id: String,
    pub air_sha256: String,
    pub pass: Pass,
}

/// The runs of `cases` in `passes` that `done` does not list, case by case in the given order,
/// for at most `limit` cases.
pub fn plan(
    cases: &[AuthoredCase],
    passes: &[Pass],
    done: &[DoneRun],
    limit: Option<usize>,
) -> Vec<Run> {
    let done = done
        .iter()
        .map(|run| (run.case_id.as_str(), run.pass))
        .collect::<HashSet<_>>();
    let mut runs = Vec::new();
    let mut taken = 0usize;
    for case in cases {
        let pending = passes
            .iter()
            .copied()
            .filter(|pass| !done.contains(&(case.case_id.as_str(), *pass)))
            .collect::<Vec<_>>();
        if pending.is_empty() {
            continue;
        }
        if limit.is_some_and(|limit| taken == limit) {
            break;
        }
        taken += 1;
        runs.extend(pending.into_iter().map(|pass| Run {
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            pass,
        }));
    }
    runs
}

/// The device a batch runs on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatchDevice {
    /// A CPU Vulkan implementation (`--device-type cpu`): runs may be concurrent.
    Cpu,
    /// A GPU (`--device-type gpu`): one run at a time, under the host's hang tripwire at these
    /// paths (`TripwirePaths::for_host` for the host's own).
    Gpu(TripwirePaths),
}

impl BatchDevice {
    pub fn device_type(&self) -> DeviceType {
        match self {
            Self::Cpu => DeviceType::Cpu,
            Self::Gpu(_) => DeviceType::Gpu,
        }
    }
}

/// Whose rows a batch records: a candidate backend's (its executor's engine runs), or Metal's
/// (the oracle's qualifications).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchBackend {
    Moltenvk,
    Vulkan,
    Metal,
}

impl BatchBackend {
    /// The candidate backend, for a candidate batch.
    pub fn candidate(self) -> Option<Backend> {
        match self {
            Self::Moltenvk => Some(Backend::Moltenvk),
            Self::Vulkan => Some(Backend::Vulkan),
            Self::Metal => None,
        }
    }

    /// Its rows' directory under `observations/` (`moltenvk`, `vulkan`, `metal`).
    pub fn directory(self) -> &'static str {
        match self.candidate() {
            Some(backend) => backend.directory(),
            None => "metal",
        }
    }
}

impl From<Backend> for BatchBackend {
    fn from(backend: Backend) -> Self {
        match backend {
            Backend::Moltenvk => Self::Moltenvk,
            Backend::Vulkan => Self::Vulkan,
        }
    }
}

/// What a batch runs and where.
#[derive(Clone, Debug)]
pub struct BatchConfig {
    /// The executor of `backend` (`corpus-vulkan`, `corpus-moltenvk`, or `corpus-metal`).
    pub executor: PathBuf,
    /// Whose rows the runs record: the slots, shards, and record are this backend's.
    pub backend: BatchBackend,
    /// The corpus whose cases, Metal rows, and private sources the runs read (never written).
    pub corpus: PathBuf,
    /// Where the build record, done-list, rows, refusals, and logs go.
    pub out: PathBuf,
    /// The Metal environment whose rows a candidate run compares against; a Metal batch's is the
    /// one it records.
    pub metal_environment_id: String,
    /// The operator's environment ID: a candidate batch's rows go to its engine slots, a
    /// Metal batch's (the same as `metal_environment_id`) to its Metal slot.
    pub environment_id: String,
    /// The committed profile every run restricts the device to (`--profiles-layer P`), if any:
    /// rows go to the `ID@engine@P` slots.
    pub restricted_to: Option<String>,
    pub device: BatchDevice,
    pub jobs: usize,
    pub timeout: Duration,
    /// Passed to the executor as `--gpu-wait-timeout`.
    pub gpu_wait_timeout: Option<u64>,
    /// Passed to the executor as `--source-index`: the corpus index whose byte locations it reads
    /// private sources through, read only and never trusted.
    pub source_index: Option<PathBuf>,
}

/// The build and venue an out directory's runs belong to ([`BUILD`]): what decides whether a row
/// or refusal they recorded is current (the executor ABI, the product's source fingerprint and
/// translation semantics, from which every engine translator fingerprint is derived, and the
/// Metal oracle), and the venue they ran on (the backend, the Metal and operator environments, the
/// profile the device was restricted to, and the device type). A batch resumes only an out
/// directory of its own build and venue.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchBuild {
    pub executor_abi: String,
    pub product: String,
    pub semantics_version: u32,
    pub oracle_abi: String,
    /// Absent in a record of a batch before MoltenVK had one: those were Vulkan.
    #[serde(default = "vulkan")]
    pub backend: BatchBackend,
    pub metal_environment_id: String,
    pub environment_id: String,
    /// Absent for an unrestricted batch (and in a record from before restricted batches).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub restricted_to: Option<String>,
    pub device_type: String,
}

fn vulkan() -> BatchBackend {
    BatchBackend::Vulkan
}

impl BatchBuild {
    /// This build's record for `config`.
    pub fn of(config: &BatchConfig) -> Self {
        Self::current(
            config.backend,
            &config.metal_environment_id,
            &config.environment_id,
            config.restricted_to.as_deref(),
            config.device.device_type(),
        )
    }

    /// This build's record for a venue.
    pub fn current(
        backend: BatchBackend,
        metal_environment_id: &str,
        environment_id: &str,
        restricted_to: Option<&str>,
        device_type: DeviceType,
    ) -> Self {
        Self {
            executor_abi: ENGINE_EXECUTOR_ABI.into(),
            product: TRANSLATOR_FINGERPRINT.into(),
            semantics_version: metal2vulkan::cache::SEMANTICS_VERSION,
            oracle_abi: ORACLE_ABI.into(),
            backend,
            metal_environment_id: metal_environment_id.into(),
            environment_id: environment_id.into(),
            restricted_to: restricted_to.map(str::to_string),
            device_type: device_type.as_str().into(),
        }
    }

    /// Whether the record's runs are of this build: its executor ABI, product, semantics, and
    /// oracle are this build's, whatever its venue. Only then are the rows and refusals they
    /// recorded current here, so only then does installing them add evidence.
    pub fn is_this_build(&self) -> bool {
        let current = Self::current(
            self.backend,
            &self.metal_environment_id,
            &self.environment_id,
            self.restricted_to.as_deref(),
            DeviceType::Gpu,
        );
        self.executor_abi == current.executor_abi
            && self.product == current.product
            && self.semantics_version == current.semantics_version
            && self.oracle_abi == current.oracle_abi
    }

    /// The record of the out directory `out`, if it holds one.
    pub fn read(out: &Path) -> Result<Option<Self>, String> {
        let path = out.join(BUILD);
        match fs::read(&path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map(Some)
                .map_err(|error| format!("{}: {error}", path.display())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(format!("read {}: {error}", path.display())),
        }
    }

    /// Claim `config.out` for this build: write its record into an out directory with no finished
    /// run, or accept one that holds this exact record. An out directory whose record names
    /// another build or venue, or whose done-list has runs and no record (runs of a batch before
    /// the record), is refused: its finished runs are not this build's.
    pub fn claim(config: &BatchConfig) -> Result<(), String> {
        let build = Self::of(config);
        match Self::read(&config.out)? {
            Some(recorded) if recorded == build => Ok(()),
            Some(recorded) => Err(format!(
                "{} holds runs of another build or venue ({}), not this one ({}): use a fresh \
                 --out",
                config.out.display(),
                serde_json::to_string(&recorded).unwrap_or_default(),
                serde_json::to_string(&build).unwrap_or_default()
            )),
            None if !read_done_list(&config.out.join(DONE_LIST))?.is_empty() => Err(format!(
                "{} holds finished runs but no {BUILD}: their build is unknown, so they cannot \
                 count as this build's; use a fresh --out",
                config.out.display()
            )),
            None => {
                let path = config.out.join(BUILD);
                let json = serde_json::to_vec_pretty(&build)
                    .map_err(|error| format!("encode {BUILD}: {error}"))?;
                fs::write(&path, [json, b"\n".to_vec()].concat())
                    .map_err(|error| format!("write {}: {error}", path.display()))
            }
        }
    }
}

/// Run `runs` under `config`, `jobs` at a time, appending each finished run to the out
/// directory and handing it to `finished` (in completion order). The out directory must be this
/// build's ([`BatchBuild::claim`]). Stops at the first I/O error (after the runs in flight end)
/// and returns it; after a run whose device is unlike its venue's profile ([`stops_the_batch`]);
/// on a GPU, also at a tripped tripwire before a run, and after a run that hung the device
/// ([`stops_a_gpu`]). A run that stops the batch is recorded first.
pub fn run_batch(
    config: &BatchConfig,
    runs: &[Run],
    finished: impl Fn(&DoneRun) + Sync,
) -> Result<Vec<DoneRun>, String> {
    if config.jobs == 0 {
        return Err("jobs must be at least 1".into());
    }
    if matches!(config.device, BatchDevice::Gpu(_)) && config.jobs != 1 {
        return Err(format!(
            "a GPU batch runs one executor at a time (jobs 1), not {}",
            config.jobs
        ));
    }
    if config.backend == BatchBackend::Metal {
        if !matches!(config.device, BatchDevice::Gpu(_)) {
            return Err("Metal has no CPU device: a Metal batch runs on the GPU".into());
        }
        if config.restricted_to.is_some() {
            return Err("a Metal batch restricts its device to no profile".into());
        }
        if runs.iter().any(|run| run.pass != Pass::Metal) {
            return Err("a Metal batch runs Metal qualifications only".into());
        }
    } else if runs.iter().any(|run| run.pass == Pass::Metal) {
        return Err(format!(
            "a {} batch runs robustness legs, never a Metal qualification",
            config.backend.directory()
        ));
    }
    let corpus = fs::canonicalize(&config.corpus)
        .map_err(|error| format!("corpus {}: {error}", config.corpus.display()))?;
    fs::create_dir_all(config.out.join(LOGS))
        .map_err(|error| format!("create {}: {error}", config.out.display()))?;
    BatchBuild::claim(config)?;
    let outputs = Mutex::new(Outputs {
        done: open_lines(&config.out.join(DONE_LIST))?,
        rows: open_lines(&config.out.join(ROWS))?,
        refusals: open_lines(&config.out.join(REFUSALS))?,
        finished: Vec::new(),
    });
    let scratch = ScratchDir::new("vulkan-batch")?;
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let errors = Mutex::new(Vec::<String>::new());
    let jobs = config.jobs.min(runs.len().max(1));
    std::thread::scope(|scope| {
        for job in 0..jobs {
            let (outputs, next, stop, errors, finished) =
                (&outputs, &next, &stop, &errors, &finished);
            let store = scratch.path().join(format!("job-{job}"));
            let corpus = &corpus;
            scope.spawn(move || {
                let result = (|| -> Result<(), String> {
                    worker_store(corpus, &store, config.backend)?;
                    while !stop.load(Ordering::Relaxed) {
                        if let BatchDevice::Gpu(tripwire) = &config.device {
                            tripwire.refuse_if_tripped()?;
                        }
                        let Some(run) = runs.get(next.fetch_add(1, Ordering::Relaxed)) else {
                            break;
                        };
                        let executed = execute_run(config, &store, run)?;
                        let mut outputs = outputs
                            .lock()
                            .map_err(|_| "output lock poisoned".to_string())?;
                        outputs.record(config, run, &executed)?;
                        finished(&executed.done);
                        let hung = match &config.device {
                            BatchDevice::Gpu(tripwire) if stops_a_gpu(executed.done.outcome) => {
                                Some(hang(
                                    tripwire,
                                    config.backend,
                                    run,
                                    &executed.done,
                                    config.timeout,
                                ))
                            }
                            _ => None,
                        };
                        let mismatch = stops_the_batch(executed.done.outcome)
                            .then(|| venue_mismatch(run, &executed.done));
                        outputs.finished.push(executed.done);
                        if let Some(hung) = hung {
                            return Err(hung);
                        }
                        if let Some(mismatch) = mismatch {
                            return Err(mismatch);
                        }
                    }
                    Ok(())
                })();
                if let Err(error) = result {
                    stop.store(true, Ordering::Relaxed);
                    if let Ok(mut errors) = errors.lock() {
                        errors.push(error);
                    }
                }
            });
        }
    });
    let errors = errors
        .into_inner()
        .map_err(|_| "error lock poisoned".to_string())?;
    if let Some(error) = errors.into_iter().next() {
        return Err(error);
    }
    Ok(outputs
        .into_inner()
        .map_err(|_| "output lock poisoned".to_string())?
        .finished)
}

/// The out directory's open line files and the runs finished so far.
struct Outputs {
    done: File,
    rows: File,
    refusals: File,
    finished: Vec<DoneRun>,
}

impl Outputs {
    /// Append `run`'s log, its row or refusal, then its done-list line (last, so a run listed
    /// done has its evidence on disk).
    fn record(
        &mut self,
        config: &BatchConfig,
        run: &Run,
        executed: &Executed,
    ) -> Result<(), String> {
        if let Some(log) = &executed.log {
            let path =
                config
                    .out
                    .join(LOGS)
                    .join(format!("{}.{}.txt", run.case_id, run.pass.label()));
            fs::write(&path, log).map_err(|error| format!("write {}: {error}", path.display()))?;
        }
        for (file, line, name) in [
            (&mut self.rows, &executed.row, ROWS),
            (&mut self.refusals, &executed.refusal, REFUSALS),
        ] {
            if let Some(line) = line {
                file.write_all(format!("{line}\n").as_bytes())
                    .and_then(|()| file.flush())
                    .map_err(|error| format!("append {name}: {error}"))?;
            }
        }
        self.done
            .write_all(executed.done.line().as_bytes())
            .and_then(|()| self.done.flush())
            .map_err(|error| format!("append {DONE_LIST}: {error}"))
    }
}

/// Whether any batch stops after a run that came to `outcome` (once it is recorded): the venue's
/// device is unlike its profile, which every later run would meet too.
pub fn stops_the_batch(outcome: Outcome) -> bool {
    outcome == Outcome::VenueMismatch
}

/// The batch's error after `done` found the venue's device unlike its profile: the batch stops
/// until the profile is recaptured.
fn venue_mismatch(run: &Run, done: &DoneRun) -> String {
    format!(
        "case {} ({}) found the device unlike the venue's profile; the batch stopped: {}",
        run.case_id,
        run.pass.label(),
        done.detail
    )
}

/// Whether a GPU batch stops after a run that came to `outcome`: its device wait timed out, its
/// device worker crashed, or it outlived the batch's timeout. Each may leave a submission running.
pub fn stops_a_gpu(outcome: Outcome) -> bool {
    matches!(
        outcome,
        Outcome::WaitTimeout | Outcome::DeviceCrash | Outcome::Timeout
    )
}

/// The batch's error after `done` hung the GPU, having made sure the tripwire at `tripwire` is
/// written: the executor's hang protocol writes it after a device-wait timeout or a device crash,
/// but nothing did for a run the batch killed at its timeout, and a missing one is written here
/// either way (an existing one is kept: the first incident is the one to explain).
fn hang(
    tripwire: &TripwirePaths,
    backend: BatchBackend,
    run: &Run,
    done: &DoneRun,
    timeout: Duration,
) -> String {
    let kind = match done.outcome {
        Outcome::DeviceCrash => HangKind::Crashed,
        _ => HangKind::Timeout,
    };
    let submissions = match backend {
        BatchBackend::Metal => SubmissionBackend::Metal,
        BatchBackend::Moltenvk | BatchBackend::Vulkan => SubmissionBackend::VulkanGpu,
    };
    let record = HangRecord::now(
        kind,
        submissions,
        &format!("corpus-{} --batch run", backend.directory()),
        format!(
            "the run of case {} ({}) came to {} ({}); its submission may still be running",
            run.case_id,
            run.pass.label(),
            done.outcome.label(),
            done.ended
        ),
        timeout,
        Some(run.case_id.clone()),
        None,
    );
    let written = match tripwire.current.try_exists() {
        Ok(true) => String::new(),
        _ => match write_record(&tripwire.current, &record) {
            Ok(()) => format!("; wrote the tripwire {}", tripwire.current.display()),
            Err(error) => format!("; and could not write the tripwire: {error}"),
        },
    };
    format!(
        "the GPU run of case {} ({}) came to {}: the batch stops{written}; follow the \
         suspected-hang protocol in docs/VALIDATION.md before another GPU run",
        run.case_id,
        run.pass.label(),
        done.outcome.label()
    )
}

/// A job's store at `store`: links to `corpus`'s cases, reviews, private sources, and library
/// modules (each that exists), and the rows `backend`'s runs read; and empty shard directories of
/// its own for the rows they write. A candidate run reads the Metal rows and the other backend's,
/// and writes `backend`'s rows and refusals; a Metal run reads no rows and writes Metal's.
fn worker_store(corpus: &Path, store: &Path, backend: BatchBackend) -> Result<(), String> {
    let (own, read) = match backend.candidate() {
        Some(backend) => {
            let other = match backend {
                Backend::Vulkan => Backend::Moltenvk,
                Backend::Moltenvk => Backend::Vulkan,
            };
            (
                vec![
                    Path::new("observations").join(backend.directory()),
                    refusal_directory(backend),
                ],
                vec![
                    PathBuf::from("observations/metal"),
                    Path::new("observations").join(other.directory()),
                ],
            )
        }
        None => (vec![PathBuf::from("observations/metal")], Vec::new()),
    };
    for own in own {
        let own = store.join(own);
        fs::create_dir_all(&own).map_err(|error| format!("create {}: {error}", own.display()))?;
    }
    fs::create_dir_all(store.join("local"))
        .map_err(|error| format!("create {}: {error}", store.display()))?;
    for relative in [PathBuf::from("cases"), PathBuf::from("reviews")]
        .into_iter()
        .chain(read)
        .chain([
            PathBuf::from("local/sources"),
            PathBuf::from("local/library-modules"),
        ])
    {
        let target = corpus.join(&relative);
        if target.exists() {
            link(&target, &store.join(&relative))?;
        }
    }
    Ok(())
}

#[cfg(unix)]
fn link(target: &Path, link: &Path) -> Result<(), String> {
    std::os::unix::fs::symlink(target, link)
        .map_err(|error| format!("link {} to {}: {error}", link.display(), target.display()))
}

#[cfg(not(unix))]
fn link(_target: &Path, _link: &Path) -> Result<(), String> {
    Err("the batch needs a Unix host".into())
}

/// What one run left: its done-list entry, its row's store line (a match or a mismatch), its
/// refusal's store line (a typed refusal after a device opened), and its log (any run but a
/// match).
struct Executed {
    done: DoneRun,
    row: Option<String>,
    refusal: Option<String>,
    log: Option<String>,
}

/// Run one (case, pass) in the job store `store`.
fn execute_run(config: &BatchConfig, store: &Path, run: &Run) -> Result<Executed, String> {
    let stdout_path = store.join("run.out");
    let stderr_path = store.join("run.err");
    let tmp = store.join("tmp");
    fs::create_dir_all(&tmp).map_err(|error| format!("create {}: {error}", tmp.display()))?;
    let stdout = File::create(&stdout_path)
        .map_err(|error| format!("{}: {error}", stdout_path.display()))?;
    let stderr = File::create(&stderr_path)
        .map_err(|error| format!("{}: {error}", stderr_path.display()))?;
    let mut command = Command::new(&config.executor);
    // The case's AIR: the executor reads the case from that one case shard.
    command
        .arg("--corpus")
        .arg(store)
        .args(["--case-id", &run.case_id])
        .args(["--air-sha256", &run.air_sha256])
        .args(["--environment-id", &config.environment_id])
        .args(["--device-type", config.device.device_type().as_str()])
        .env("TMPDIR", &tmp)
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr);
    if let Pass::Leg(leg) = run.pass {
        command.args(["--metal-environment-id", &config.metal_environment_id]);
        if leg == Leg::RobustnessOff {
            command.arg("--robustness-off");
        }
        if let Some(profile) = &config.restricted_to {
            command.args(["--profiles-layer", profile]);
        }
    }
    if let Some(seconds) = config.gpu_wait_timeout {
        command.args(["--gpu-wait-timeout", &seconds.to_string()]);
    }
    if let Some(index) = &config.source_index {
        command.arg("--source-index").arg(index);
    }
    let started = Instant::now();
    let mut child = session::spawn(&mut command)
        .map_err(|error| format!("spawn {}: {error}", config.executor.display()))?;
    let ended = wait_until(&mut child, started + config.timeout)?;
    let leftover = session::end(&mut child);
    let seconds = started.elapsed().as_secs_f64();
    let scratch_left = fs::remove_dir_all(&tmp)
        .err()
        .map(|error| format!("remove {}: {error}", tmp.display()));

    let recorded = match run.pass {
        Pass::Leg(leg) => {
            let backend = config
                .backend
                .candidate()
                .ok_or("a Metal batch has no robustness legs")?;
            let slot = engine_slot(&config.environment_id, config.restricted_to.as_deref(), leg);
            let row = find_row(store, backend, run, &slot)?;
            let refusal = find_refusal(store, backend, run, &slot)?;
            Recorded {
                outcome: classify(
                    ended,
                    row.as_ref().map(|(_, status)| *status),
                    refusal.is_some(),
                ),
                row: row.map(|(line, status)| (line, format!("{status:?}"))),
                refusal,
                slot,
            }
        }
        Pass::Metal => {
            let row = find_metal_row(store, run, &config.environment_id)?;
            Recorded {
                outcome: classify_metal(ended, row.as_ref().map(|(_, status)| status.clone())),
                row: row.map(|(line, status)| (line, format!("{status:?}"))),
                refusal: None,
                slot: config.environment_id.clone(),
            }
        }
    };
    let Recorded {
        outcome,
        row,
        refusal,
        slot,
    } = recorded;
    let stdout = read_lossy(&stdout_path)?;
    let stderr = read_lossy(&stderr_path)?;
    let last_error = || {
        last_line(&stderr)
            .or_else(|| last_line(&stdout))
            .unwrap_or_default()
    };
    let mut detail = match outcome {
        Outcome::Match | Outcome::Qualified => String::new(),
        // A Metal process holds its device itself: a signal ended it, with no verdict of its own.
        Outcome::DeviceCrash if run.pass == Pass::Metal => {
            format!("{}: {}", ended.label(), last_error())
        }
        Outcome::Mismatch
        | Outcome::ReferenceGap
        | Outcome::NotSupported
        | Outcome::Refused
        | Outcome::DeviceCrash
        | Outcome::NeedsMetal
        | Outcome::VenueMismatch => first_line(&stdout).unwrap_or_default().into(),
        Outcome::WaitTimeout => last_error().into(),
        Outcome::Timeout => format!("killed after {} s", config.timeout.as_secs()),
        Outcome::Failed => match (&row, &refusal) {
            (Some((_, status)), _) => {
                format!("{} with a {status} row: {}", ended.label(), last_error())
            }
            (None, Some(_)) => format!("{} with a refusal: {}", ended.label(), last_error()),
            (None, None) => format!(
                "{} without a row in {slot}: {}",
                ended.label(),
                last_error()
            ),
        },
    };
    for error in leftover.err().into_iter().chain(scratch_left) {
        detail = format!("{detail} [{error}]");
    }
    let log = (outcome.class() != Class::Match).then(|| {
        format!(
            "== {} {} {} {}\n== stdout\n{}\n== stderr\n{}\n",
            run.case_id,
            run.pass.label(),
            outcome.label(),
            ended.label(),
            tail(&stdout),
            tail(&stderr)
        )
    });
    let row = row
        .filter(|_| {
            matches!(
                outcome,
                Outcome::Match | Outcome::Mismatch | Outcome::Qualified | Outcome::ReferenceGap
            )
        })
        .map(|(line, _)| line);
    let refusal = refusal.filter(|_| outcome == Outcome::NotSupported);
    let done = DoneRun {
        case_id: run.case_id.clone(),
        pass: run.pass,
        outcome,
        ended: ended.label(),
        seconds,
        detail: sanitize(&detail),
    };
    Ok(Executed {
        done,
        row,
        refusal,
        log,
    })
}

/// What a run left in its slot: its outcome, its row's store line and status, its refusal's store
/// line, and the slot.
struct Recorded {
    outcome: Outcome,
    row: Option<(String, String)>,
    refusal: Option<String>,
    slot: String,
}

fn wait_until(child: &mut Child, deadline: Instant) -> Result<Ended, String> {
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(Ended::of(status)),
            Ok(None) if Instant::now() >= deadline => return Ok(Ended::TimedOut),
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(error) => return Err(format!("wait for the executor: {error}")),
        }
    }
}

/// The store line and status of `run`'s row of `backend` in `slot`, if the job store holds one.
fn find_row(
    store: &Path,
    backend: Backend,
    run: &Run,
    slot: &str,
) -> Result<Option<(String, CandidateStatus)>, String> {
    #[derive(serde::Deserialize)]
    struct Key {
        case_id: String,
        environment_id: String,
        status: CandidateStatus,
    }
    let directory = store.join("observations").join(backend.directory());
    Ok(find_line::<Key>(&directory, run, |key| {
        key.case_id == run.case_id && key.environment_id == slot
    })?
    .map(|(line, key)| (line, key.status)))
}

/// The store line of `run`'s refusal of `backend` in `slot`, if the job store holds one.
fn find_refusal(
    store: &Path,
    backend: Backend,
    run: &Run,
    slot: &str,
) -> Result<Option<String>, String> {
    #[derive(serde::Deserialize)]
    struct Key {
        case_id: String,
        environment_id: String,
    }
    let directory = store.join(refusal_directory(backend));
    Ok(find_line::<Key>(&directory, run, |key| {
        key.case_id == run.case_id && key.environment_id == slot
    })?
    .map(|(line, _)| line))
}

/// The store line and status of `run`'s Metal row in `environment_id`, if the job store holds one.
fn find_metal_row(
    store: &Path,
    run: &Run,
    environment_id: &str,
) -> Result<Option<(String, MetalStatus)>, String> {
    #[derive(serde::Deserialize)]
    struct Key {
        case_id: String,
        environment_id: String,
        status: MetalStatus,
    }
    let directory = store.join("observations/metal");
    Ok(find_line::<Key>(&directory, run, |key| {
        key.case_id == run.case_id && key.environment_id == environment_id
    })?
    .map(|(line, key)| (line, key.status)))
}

/// The first line of `run`'s shard under `directory` whose key `matches`, with its key.
fn find_line<K: DeserializeOwned>(
    directory: &Path,
    run: &Run,
    matches: impl Fn(&K) -> bool,
) -> Result<Option<(String, K)>, String> {
    let path = directory.join(shard_name_for_hash(&run.air_sha256)?);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("read {}: {error}", path.display())),
    };
    for line in text.lines() {
        let key: K = serde_json::from_str(line)
            .map_err(|error| format!("{}: bad line: {error}", path.display()))?;
        if matches(&key) {
            return Ok(Some((line.to_string(), key)));
        }
    }
    Ok(None)
}

fn read_lossy(path: &Path) -> Result<String, String> {
    fs::read(path)
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
        .map_err(|error| format!("read {}: {error}", path.display()))
}

fn first_line(text: &str) -> Option<&str> {
    text.lines().map(str::trim).find(|line| !line.is_empty())
}

fn last_line(text: &str) -> Option<&str> {
    text.lines().map(str::trim).rfind(|line| !line.is_empty())
}

fn tail(text: &str) -> &str {
    if text.len() <= LOG_TAIL_BYTES {
        return text;
    }
    let mut start = text.len() - LOG_TAIL_BYTES;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    &text[start..]
}

/// `detail` on one done-list field: no tabs or line breaks, at most [`DETAIL_CHARS`].
fn sanitize(detail: &str) -> String {
    detail
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .take(DETAIL_CHARS)
        .collect()
}

/// What [`install`] wrote, and the record of the runs it came from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Installed {
    pub rows: usize,
    pub refusals: usize,
    /// The out directory's [`BatchBuild`], if it holds one (a batch before the record has none).
    pub build: Option<BatchBuild>,
}

/// Write the evidence of each slot's last run in the out directory `out` into `store` (rows, then
/// refusals, each in (case, slot) order) through the store's own validated writes: for each
/// (case, leg), the last
/// done-list run that recorded evidence decides. A match or a mismatch installs the slot's last
/// row ([`ROWS`]), a typed refusal its last refusal ([`REFUSALS`]; a refusal before a device
/// opened recorded none, and installs nothing). Evidence no finished run lists (an append cut
/// before its done-list line) is not installed. The store keeps a slot's row or its refusal,
/// never both, so the installed one replaces the other. Every row and refusal must be of
/// `backend`, and so must the out directory's record (one without a record is Vulkan's). The
/// record is returned: only rows of this build ([`BatchBuild::is_this_build`]) are current once
/// installed, which the caller checks before it commits them.
pub fn install(out: &Path, store: &CorpusStore, backend: Backend) -> Result<Installed, String> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Evidence {
        Row,
        Refusal,
    }
    let build = BatchBuild::read(out)?;
    let recorded = build
        .as_ref()
        .map_or(BatchBackend::Vulkan, |build| build.backend);
    if recorded != BatchBackend::from(backend) {
        return Err(format!(
            "{} holds runs of the {} backend, not {}",
            out.display(),
            recorded.directory(),
            backend.directory()
        ));
    }
    let output = BatchOutput::read(out)?;
    let mut last = HashMap::<(String, Pass), Evidence>::new();
    for run in output.done {
        let evidence = match run.outcome {
            Outcome::Match | Outcome::Mismatch => Evidence::Row,
            Outcome::NotSupported => Evidence::Refusal,
            _ => continue,
        };
        last.insert((run.case_id, run.pass), evidence);
    }
    let leg_of = |slot: &str| {
        Pass::Leg(if is_robustness_off_slot(slot) {
            Leg::RobustnessOff
        } else {
            Leg::Stated
        })
    };
    let mut rows = BTreeMap::new();
    for row in output.rows {
        if row.backend != backend {
            return Err(format!(
                "row {} {} is not a {} row",
                row.case_id,
                row.environment_id,
                backend.directory()
            ));
        }
        rows.insert((row.case_id.clone(), row.environment_id.clone()), row);
    }
    let mut refusals = BTreeMap::new();
    for refusal in output.refusals {
        if refusal.backend != backend {
            return Err(format!(
                "refusal {} {} is not a {} refusal",
                refusal.case_id,
                refusal.environment_id,
                backend.directory()
            ));
        }
        refusals.insert(
            (refusal.case_id.clone(), refusal.environment_id.clone()),
            refusal,
        );
    }
    let decides = |(case_id, slot): &(String, String), evidence: Evidence| {
        last.get(&(case_id.clone(), leg_of(slot))) == Some(&evidence)
    };
    let mut installed = Installed {
        build,
        ..Installed::default()
    };
    for (key, row) in rows {
        if decides(&key, Evidence::Row) {
            store.upsert_candidate(row)?;
            installed.rows += 1;
        }
    }
    for (key, refusal) in refusals {
        if decides(&key, Evidence::Refusal) {
            store.upsert_refusal(refusal)?;
            installed.refusals += 1;
        }
    }
    Ok(installed)
}

/// Whether a case's AIR is a public fixture (`public`) or private AIR (`private`).
pub fn scope(case: &AuthoredCase) -> &'static str {
    if case.public_source.is_some() {
        "public"
    } else {
        "private"
    }
}

/// The report's family of a case: its stage and the first structural resource family it uses,
/// in this order: ray tracing, tessellation, indirect commands, imageblocks, function tables,
/// out-of-bounds targets, textures or samplers, argument buffers or device-buffer arrays, else
/// plain buffers.
pub fn family(case: &AuthoredCase) -> String {
    let stage = match case.stage {
        Stage::Kernel => "kernel",
        Stage::Vertex => "vertex",
        Stage::Fragment => "fragment",
        Stage::Object => "object",
        Stage::Mesh => "mesh",
    };
    let shape = if !case.acceleration_structures.is_empty()
        || !case.intersection_function_tables.is_empty()
        || !case.argument_buffer_intersection_function_tables.is_empty()
    {
        "ray_tracing"
    } else if case.tessellation.is_some() {
        "tessellation"
    } else if !case.command_buffers.is_empty() || !case.pipeline_states.is_empty() {
        "indirect_commands"
    } else if case.imageblock.is_some() || case.fragment_imageblock.is_some() {
        "imageblock"
    } else if !case.visible_function_tables.is_empty()
        || !case.visible_function_references.is_empty()
    {
        "function_tables"
    } else if case.out_of_bounds.is_some() {
        "out_of_bounds"
    } else if !case.textures.is_empty()
        || !case.texture_arrays.is_empty()
        || !case.argument_buffer_textures.is_empty()
        || !case.argument_buffer_samplers.is_empty()
        || !case.samplers.is_empty()
    {
        "textures"
    } else if !case.argument_buffer_buffers.is_empty() || !case.device_buffer_arrays.is_empty() {
        "argument_buffers"
    } else {
        "buffers"
    };
    format!("{stage}.{shape}")
}

/// The batch's report over `output` (the last run of each (case, leg) wins), against the
/// `selected` cases (what [`qualified_cases`] selects in `metal_environment_id`, narrowed by
/// [`requested_cases`]: a run of any other case is not counted), for `legs` and the operator's
/// `environment_id` (restricted to the profile `restricted_to`, if any: its rows and refusals are
/// in that profile's slots). Lines:
///
/// - `batch-report leg=L selected=N ran=N pending=N match=N mismatch=N refusal=N failure=N`, then
///   the count of each refusal and failure [`Outcome`] (`not_supported=N ... failed=N`);
/// - `batch-family leg=L family=F cases=N match=N mismatch=N refusal=N failure=N` per [`family`],
///   and `batch-scope leg=L scope=S ...` the same per [`scope`];
/// - `batch-run case=ID leg=L scope=S family=F outcome=O seconds=S name=NAME detail=D` for every
///   run that did not match; a mismatch adds `comparison=C words=DIFFERING/TOTAL
///   first=I:METAL/CANDIDATE` from its row and the case's current Metal row, and a typed refusal
///   `refusal=CLASS` from its recorded refusal (`device`, `product`, `executor`), or
///   `refusal=unrecorded` for one before a device opened;
/// - `batch-slowest case=ID leg=L seconds=S outcome=O` for the slowest run.
pub fn report(
    selected: &[AuthoredCase],
    metals: &[MetalObservation],
    metal_environment_id: &str,
    environment_id: &str,
    restricted_to: Option<&str>,
    legs: &[Leg],
    output: &BatchOutput,
) -> Vec<String> {
    let BatchOutput {
        done,
        rows,
        refusals,
    } = output;
    let by_id = selected
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect::<HashMap<_, _>>();
    let mut latest = BTreeMap::<(&str, Pass), &DoneRun>::new();
    for run in done {
        latest.insert((run.case_id.as_str(), run.pass), run);
    }
    let row_of = rows
        .iter()
        .map(|row| ((row.case_id.as_str(), row.environment_id.as_str()), row))
        .collect::<HashMap<_, _>>();
    let refusal_of = refusals
        .iter()
        .map(|refusal| {
            (
                (refusal.case_id.as_str(), refusal.environment_id.as_str()),
                refusal,
            )
        })
        .collect::<HashMap<_, _>>();
    let metal_of = metals
        .iter()
        .filter(|metal| metal.environment_id == metal_environment_id)
        .map(|metal| (metal.case_id.as_str(), metal))
        .collect::<HashMap<_, _>>();
    let mut lines = Vec::new();
    for &leg in legs {
        let runs = latest
            .iter()
            .filter(|((case_id, pass), _)| *pass == Pass::Leg(leg) && by_id.contains_key(case_id))
            .map(|(_, run)| *run)
            .collect::<Vec<_>>();
        let mut outcomes = BTreeMap::<Outcome, usize>::new();
        let mut groups = BTreeMap::<(&str, String), BTreeMap<Class, usize>>::new();
        for run in &runs {
            *outcomes.entry(run.outcome).or_default() += 1;
            let case = by_id[run.case_id.as_str()];
            for group in [("scope", scope(case).to_string()), ("family", family(case))] {
                *groups
                    .entry(group)
                    .or_default()
                    .entry(run.outcome.class())
                    .or_default() += 1;
            }
        }
        let class_count = |class: Class| {
            outcomes
                .iter()
                .filter(|(outcome, _)| outcome.class() == class)
                .map(|(_, count)| count)
                .sum::<usize>()
        };
        let mut line = format!(
            "batch-report leg={} selected={} ran={} pending={}",
            leg_label(leg),
            selected.len(),
            runs.len(),
            selected.len() - runs.len()
        );
        for class in CLASSES {
            line.push_str(&format!(" {}={}", class.label(), class_count(class)));
        }
        // A match and a mismatch are their own classes; the other classes' outcomes follow (a
        // Metal run's outcomes are no candidate run's).
        for outcome in OUTCOMES.into_iter().filter(|outcome| {
            !matches!(
                outcome,
                Outcome::Match | Outcome::Mismatch | Outcome::Qualified | Outcome::ReferenceGap
            )
        }) {
            line.push_str(&format!(
                " {}={}",
                outcome.label(),
                outcomes.get(&outcome).copied().unwrap_or(0)
            ));
        }
        lines.push(line);
        for ((kind, name), classes) in &groups {
            let mut line = format!(
                "batch-{kind} leg={} {kind}={name} cases={}",
                leg_label(leg),
                classes.values().sum::<usize>()
            );
            for class in CLASSES {
                line.push_str(&format!(
                    " {}={}",
                    class.label(),
                    classes.get(&class).copied().unwrap_or(0)
                ));
            }
            lines.push(line);
        }
        let slot = engine_slot(environment_id, restricted_to, leg);
        for run in runs.iter().filter(|run| run.outcome != Outcome::Match) {
            let case = by_id[run.case_id.as_str()];
            let mut line = format!(
                "batch-run case={} leg={} scope={} family={} outcome={} seconds={:.3} name={} \
                 detail={}",
                run.case_id,
                leg_label(leg),
                scope(case),
                family(case),
                run.outcome.label(),
                run.seconds,
                case.name,
                run.detail
            );
            if run.outcome == Outcome::Mismatch {
                match row_of.get(&(run.case_id.as_str(), slot.as_str())) {
                    Some(row) => {
                        let comparison = serde_json::to_string(&row.comparison)
                            .unwrap_or_else(|error| format!("?{error}"));
                        line.push_str(&format!(" comparison={comparison}"));
                        if let Some(metal) = metal_of.get(run.case_id.as_str()) {
                            line.push_str(&format!(
                                " {}",
                                word_difference(&metal.output_b64, &row.output_b64)
                            ));
                        }
                    }
                    None => line.push_str(" row=missing"),
                }
            }
            if run.outcome == Outcome::NotSupported {
                let class = refusal_of
                    .get(&(run.case_id.as_str(), slot.as_str()))
                    .map_or("unrecorded", |refusal| refusal.refusal.class.as_str());
                line.push_str(&format!(" refusal={class}"));
            }
            lines.push(line);
        }
    }
    if let Some(slowest) = latest
        .values()
        .filter(|run| by_id.contains_key(run.case_id.as_str()))
        .max_by(|left, right| left.seconds.total_cmp(&right.seconds))
    {
        lines.push(format!(
            "batch-slowest case={} leg={} seconds={:.3} outcome={}",
            slowest.case_id,
            slowest.pass.label(),
            slowest.seconds,
            slowest.outcome.label()
        ));
    }
    lines
}

/// `words=DIFFERING/TOTAL first=I:METAL/CANDIDATE` over the 32-bit little-endian words of two
/// base64 outputs (a trailing partial word counts as one), or `bytes=METAL/CANDIDATE` when
/// their lengths differ.
fn word_difference(metal_b64: &str, candidate_b64: &str) -> String {
    let decode = |b64: &str| base64::engine::general_purpose::STANDARD.decode(b64);
    let (Ok(metal), Ok(candidate)) = (decode(metal_b64), decode(candidate_b64)) else {
        return "words=undecodable".into();
    };
    if metal.len() != candidate.len() {
        return format!("bytes={}/{}", metal.len(), candidate.len());
    }
    let word = |chunk: &[u8]| {
        let mut bytes = [0u8; 4];
        bytes[..chunk.len()].copy_from_slice(chunk);
        u32::from_le_bytes(bytes)
    };
    let differing = metal
        .chunks(4)
        .zip(candidate.chunks(4))
        .enumerate()
        .filter(|(_, (left, right))| left != right)
        .map(|(index, (left, right))| (index, word(left), word(right)))
        .collect::<Vec<_>>();
    let total = metal.len().div_ceil(4);
    match differing.first() {
        Some((index, left, right)) => format!(
            "words={}/{total} first={index}:{left:08x}/{right:08x}",
            differing.len()
        ),
        None => format!("words=0/{total}"),
    }
}

/// Process sessions: a run's executor starts its own, and every process left in it when the run
/// ends is killed, the executor's compile and device workers included (each runs in a process
/// group of its own inside the session, which a group kill would miss).
mod session {
    use std::process::{Child, Command};
    use std::time::{Duration, Instant};

    /// How long the processes left in a session may take to go after SIGKILL.
    const DRAIN: Duration = Duration::from_secs(5);

    #[cfg(unix)]
    pub fn spawn(command: &mut Command) -> std::io::Result<Child> {
        use std::os::unix::process::CommandExt as _;
        // SAFETY: setsid is async-signal-safe and touches no memory of the parent.
        unsafe {
            command.pre_exec(|| {
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        command.spawn()
    }

    #[cfg(not(unix))]
    pub fn spawn(_command: &mut Command) -> std::io::Result<Child> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "sessions need a Unix host",
        ))
    }

    /// SIGKILL every process left in `child`'s session (`child` itself if it still runs), reap
    /// `child`, and wait up to [`DRAIN`] for the rest to go. An error names what survived.
    #[cfg(unix)]
    pub fn end(child: &mut Child) -> Result<(), String> {
        let session = child.id() as libc::pid_t;
        let kill = |members: &[libc::pid_t]| {
            for &pid in members {
                // SAFETY: plain syscall on a pid read from this session.
                unsafe {
                    libc::kill(pid, libc::SIGKILL);
                }
            }
        };
        kill(&members(session)?);
        let _ = child.kill();
        let _ = child.wait();
        let started = Instant::now();
        loop {
            let left = members(session)?;
            if left.is_empty() {
                return Ok(());
            }
            if started.elapsed() >= DRAIN {
                return Err(format!("session {session} left processes {left:?}"));
            }
            kill(&left);
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    #[cfg(not(unix))]
    pub fn end(child: &mut Child) -> Result<(), String> {
        let _ = child.kill();
        let _ = child.wait();
        Ok(())
    }

    /// The live processes of `session`, never this process.
    #[cfg(unix)]
    fn members(session: libc::pid_t) -> Result<Vec<libc::pid_t>, String> {
        let own = std::process::id() as libc::pid_t;
        Ok(all_pids()?
            .into_iter()
            .filter(|&pid| pid > 0 && pid != own)
            // SAFETY: plain syscall; an exited pid or one not visible returns -1.
            .filter(|&pid| unsafe { libc::getsid(pid) } == session)
            .filter(|&pid| !is_zombie(pid))
            .collect())
    }

    #[cfg(target_os = "linux")]
    fn all_pids() -> Result<Vec<libc::pid_t>, String> {
        let entries =
            std::fs::read_dir("/proc").map_err(|error| format!("enumerate processes: {error}"))?;
        Ok(entries
            .filter_map(Result::ok)
            .filter_map(|entry| entry.file_name().to_str()?.parse::<libc::pid_t>().ok())
            .collect())
    }

    /// A zombie waits for its parent (the container's init for an orphan) and holds nothing.
    #[cfg(target_os = "linux")]
    fn is_zombie(pid: libc::pid_t) -> bool {
        std::fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
            stat.rsplit_once(')')
                .is_some_and(|(_, rest)| rest.trim_start().starts_with('Z'))
        })
    }

    #[cfg(target_os = "macos")]
    fn all_pids() -> Result<Vec<libc::pid_t>, String> {
        let mut pids = vec![0 as libc::pid_t; 16384];
        let capacity = std::mem::size_of_val(pids.as_slice()) as libc::c_int;
        // SAFETY: the buffer holds `capacity` bytes; PROC_ALL_PIDS (1) ignores the argument.
        let bytes = unsafe { libc::proc_listpids(1, 0, pids.as_mut_ptr().cast(), capacity) };
        if bytes <= 0 {
            return Err(format!(
                "enumerate processes: {}",
                std::io::Error::last_os_error()
            ));
        }
        pids.truncate(bytes as usize / std::mem::size_of::<libc::pid_t>());
        Ok(pids)
    }

    /// macOS: an orphan is reaped by launchd as soon as it dies.
    #[cfg(target_os = "macos")]
    fn is_zombie(_pid: libc::pid_t) -> bool {
        false
    }

    #[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
    fn all_pids() -> Result<Vec<libc::pid_t>, String> {
        Err("enumerating a session needs Linux or macOS".into())
    }

    #[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
    fn is_zombie(_pid: libc::pid_t) -> bool {
        false
    }
}

pub mod metal;

#[cfg(test)]
mod tests;
