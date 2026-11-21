//! The engine candidate's device work in a child process.
//!
//! A Vulkan driver that faults kills the process holding its device: lavapipe at a 128-bit vector
//! width segfaults in every acceleration-structure build, and before this boundary that killed
//! `corpus-vulkan` (exit 139) and would have ended a sweep at the case. `execute`
//! runs the device half of one case (`candidate::engine::run_on_device`: open the device,
//! compile in a compile worker of its own, bind, submit, read back) in a child of the tool's own
//! executable started with [`WORKER_ARG`], and the tool keeps the device-free half (the checks, the
//! Metal observation, the comparison, the row). A child that dies by a signal is a typed
//! [`DeviceCrash`], never a row and never a match; for a GPU the parent first writes the host
//! tripwire, since whatever the child submitted may still be running.
//!
//! The exchange, step by step:
//!
//! 1. The parent writes a `DeviceInput` to the file that becomes the child's stdin: the case,
//!    the corpus root, the device type, the bound of every device wait
//!    ([`wait_timeout()`]), the committed profile its contexts are restricted to
//!    (`--profiles-layer P`), the
//!    read-only source index its private sources are read through (`--source-index`), and the
//!    digest of the input its own check produced. That is every process-wide setting the device
//!    half reads; the environment (the loader's and the layers' variables, `METAL2VULKAN_*`,
//!    `HOME`) is inherited.
//! 2. The child ([`serve_if_requested`]) sets the wait bound, reads private sources through the
//!    parent's source index if it was handed one, restricts its contexts to P and
//!    sets its generated-commands setting before it opens any device, turns on the
//!    validation layer's echo (`candidate::echo_validation_messages`: each failing message goes
//!    to stderr as the layer reports it, at most as many as a row keeps, so a crash still shows
//!    them), admits and prepares the case again, requires the parent's input digest, runs it,
//!    and reports `Result<DeviceRun, ExecutorError>` with its build identity: its product
//!    fingerprint ([`TRANSLATOR_FINGERPRINT`]) and its validation-crate fingerprint
//!    ([`VALIDATION_FINGERPRINT`]), since it runs this crate's executor (binding, readback, the
//!    comparison class, the census, the environment) as well as the product; and with whether its
//!    contexts enabled `VK_EXT_device_generated_commands`.
//! 3. The parent waits without a deadline or a memory ceiling: a GPU child is never killed while
//!    it may have work in flight. The run is therefore bounded only as far as the child bounds
//!    itself: every device wait (`gpu::wait`) and its compile (its compile worker) are; a driver
//!    call that never returns, such as creating a device or a pipeline, holds the parent, as it
//!    held the tool before the device moved out. The parent then classifies the exit (`Ended`)
//!    before it writes anything, so a GPU child's crash writes the tripwire first; only then does
//!    it relay the child's stderr, ignoring a closed stream. A report is decoded and refused
//!    unless the child's build identity and generated-commands setting are the parent's and its
//!    census names the parent's restriction; the hang protocol's exit comes back to
//!    `execute`, which leaves with the same status once the scratch files are gone, as the
//!    in-process wait did; a signal is a [`DeviceCrash`]; any other exit is `Failed`, with the
//!    stderr.

use super::compile::CompileWorker;
use super::{bounded, report, MonitoredChild, ProcessControl, Reported};
use crate::candidate::engine::artifact::ExecutedArtifact;
use crate::case::AuthoredCase;
use crate::check::admission::{DeviceCrash, ExecutorError};
use crate::coverage::ExecutedCensus;
use crate::gpu::tripwire::{self, HangKind, HangRecord};
use crate::gpu::wait::{
    report_incident, set_wait_timeout, wait_timeout, SubmissionBackend, HANG_EXIT_CODE,
};
use crate::gpu::DeviceType;
use crate::observation::{CaseTiming, TRANSLATOR_FINGERPRINT};
use crate::observation_contract::ComparisonClass;
use crate::ScratchDir;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read as _, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::Duration;

/// The first argument that starts a validation binary as a device worker.
pub const WORKER_ARG: &str = "--engine-device-worker";

/// This crate's source fingerprint (its manifest, its build script, and every file under its
/// `src/`): with the product's [`TRANSLATOR_FINGERPRINT`], the build identity a device worker
/// reports. An executable rebuilt since its parent started may run another executor, and its run
/// would be stored under the parent's executor ABI.
pub const VALIDATION_FINGERPRINT: &str = env!("METAL2VULKAN_VALIDATION_FINGERPRINT");

/// What the parent hands a device worker.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct DeviceInput {
    /// The corpus the case and its sources come from.
    pub(crate) root: PathBuf,
    /// The case as the parent admitted it.
    pub(crate) case: AuthoredCase,
    pub(crate) device: DeviceType,
    /// The digest of the parent's checked input (`CheckedCase::primary.input_sha256`): the
    /// child's own check must produce it, so both halves ran the same case on the same sources.
    pub(crate) input_sha256: String,
    /// The bound of every device wait in the parent ([`wait_timeout()`]), in milliseconds.
    pub(crate) wait_timeout_ms: u64,
    /// The committed profile the parent's contexts are restricted to (`--profiles-layer P`): the
    /// worker restricts its own to it before it opens a device, and the census of its run must name
    /// it.
    pub(crate) restricted_to: Option<String>,
    /// Whether the parent's contexts enable `VK_EXT_device_generated_commands` where their device
    /// supports it (`--without-generated-commands` clears it): the worker opens its own the same
    /// way, and its report must name it.
    pub(crate) generated_commands: bool,
    /// The C3 venue the parent records the run under, if any: the worker holds the device it opens
    /// to the venue's profile before it runs anything, and its run or refusal must record that
    /// profile.
    pub(crate) venue: Option<VenueRun>,
    /// The read-only source index the parent reads private sources through
    /// (`source::use_source_index`, `--source-index`), if any: the worker's admission reads them
    /// through it too, falling back to the whole shard wherever it is wrong.
    pub(crate) source_index: Option<PathBuf>,
}

/// The C3 venue a run records under: its operator environment, the committed profile the venue
/// names, and the device identities it allows ([`crate::c3::Venue::devices`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct VenueRun {
    pub(crate) environment_id: String,
    pub(crate) profile: String,
    pub(crate) allowed: Vec<crate::c3::DeviceIdentity>,
}

/// What one device run read back: the device-side fields of the observation the parent stores.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceRun {
    /// The selected output bytes.
    #[serde(with = "base64_bytes")]
    pub output: Vec<u8>,
    /// SHA-256 of the pipeline's SPIR-V bytes (before specialization), where the executor compiled
    /// it itself: `None` for a run through the runtime, which keeps its modules to itself.
    pub spv_sha256: Option<String>,
    pub translator_fingerprint: String,
    pub environment: serde_json::Value,
    pub comparison: ComparisonClass,
    pub census: ExecutedCensus,
    /// What the execution cost in time, measured in the worker around the run itself
    /// ([`CaseTiming`]), which is the figure the row records. The worker measures every run it
    /// reports, so this is not optional: the parent has a figure for every row it stores.
    pub timing: CaseTiming,
}

/// The device a device worker opened before its run ended in a typed refusal: the facts the row
/// would have recorded, which the parent stores with the refusal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenedDevice {
    /// The environment a row on this device records: the device and driver, the layers, the
    /// robustness features, the profiles layer's restriction, and the profile and its fingerprint.
    pub environment: serde_json::Value,
    /// The engine's fingerprint of the opened device's profile under this product.
    pub translator_fingerprint: String,
    /// What the run had compiled when it refused, which the refusal records.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<ExecutedArtifact>,
    /// What the refused attempt cost in time ([`CaseTiming`]), which the refusal records. The
    /// worker's clock is read around everything after the case was prepared, which is every
    /// attempt that gets this far — a refusal of the restricted device itself names one before it
    /// opened, and that attempt is timed too — so this is not optional either.
    pub timing: CaseTiming,
}

/// What the device half of one case came to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DeviceOutcome {
    /// The run, or why it produced none.
    pub(crate) result: Result<DeviceRun, ExecutorError>,
    /// For a typed refusal after the device opened, that device; `None` otherwise.
    pub(crate) refused_on: Option<OpenedDevice>,
}

impl DeviceOutcome {
    /// An outcome with no opened device behind it.
    fn of(result: Result<DeviceRun, ExecutorError>) -> Self {
        Self {
            result,
            refused_on: None,
        }
    }
}

/// What a device worker reports.
#[derive(Debug, Serialize, Deserialize)]
struct DeviceOutput {
    /// The worker's [`TRANSLATOR_FINGERPRINT`]: an executable rebuilt since the parent started
    /// translates differently, so its run is not the parent's evidence.
    product_fingerprint: String,
    /// The worker's [`VALIDATION_FINGERPRINT`]: it may run another executor.
    validation_fingerprint: String,
    /// Whether the worker's contexts enable `VK_EXT_device_generated_commands` where their device
    /// supports it: a run on the other path than the parent's is not the parent's evidence.
    generated_commands: bool,
    result: Result<DeviceRun, ExecutorError>,
    /// For a typed refusal after the device opened, that device ([`OpenedDevice`]).
    refused_on: Option<OpenedDevice>,
}

/// Run the device half of `input`'s case in a device worker of `worker`'s executable (the tool's
/// own, which serves both workers) and wait for it.
pub(crate) fn execute(worker: &CompileWorker, input: &DeviceInput) -> DeviceOutcome {
    let mut command = worker.command();
    command.arg(WORKER_ARG);
    match supervise(
        command,
        input,
        tripwire::write_host_record,
        std::io::stderr(),
    ) {
        Ok(outcome) => outcome,
        // The worker printed its hang record (and, for a GPU, wrote the tripwire); the tool
        // leaves as the in-process wait did, so the protocol's exit status stands. `supervise`
        // has removed its scratch files.
        Err(HangExit) => std::process::exit(HANG_EXIT_CODE),
    }
}

/// A device worker left through the hang protocol ([`HANG_EXIT_CODE`]).
#[derive(Debug, PartialEq, Eq)]
struct HangExit;

/// What [`supervise`] saw a device worker do: the run's result (a report, a failure, or a crash),
/// or the hang protocol's exit.
type Supervised = Result<DeviceOutcome, HangExit>;

/// How a device worker ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ended {
    /// It exited 0, having written its report.
    Reported,
    /// It left through the hang protocol ([`HANG_EXIT_CODE`]): a device wait timed out.
    Hang,
    /// It died by this signal.
    Signal(i32),
    /// It exited with any other status.
    Exited,
}

impl Ended {
    fn of(status: ExitStatus) -> Ended {
        if status.success() {
            return Ended::Reported;
        }
        if status.code() == Some(HANG_EXIT_CODE) {
            return Ended::Hang;
        }
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt as _;
            if let Some(signal) = status.signal() {
                return Ended::Signal(signal);
            }
        }
        Ended::Exited
    }
}

/// [`execute`] in the worker `command` starts: a GPU worker's crash record goes through
/// `write_tripwire`, and the worker's stderr to `relay` (the tests start a shell that ends as a
/// worker can, keep the record, and keep or refuse what is relayed).
fn supervise(
    command: Command,
    input: &DeviceInput,
    write_tripwire: impl FnOnce(&HangRecord) -> Result<(), String>,
    mut relay: impl Write,
) -> Supervised {
    let scratch = match ScratchDir::new("device-worker") {
        Ok(scratch) => scratch,
        Err(error) => return Ok(DeviceOutcome::of(Err(error.into()))),
    };
    let (supervised, stderr) = run_worker(&scratch, command, input, write_tripwire);
    // What the worker wrote to stderr is the tool's, as it was before the device moved out; a
    // stream nobody reads any more (a closed pipe) loses it, and changes nothing else.
    let _ = relay.write_all(&stderr).and_then(|()| relay.flush());
    supervised
}

/// Start the worker `command` on `input` with its streams in `scratch`, wait for it, and classify
/// its exit before anything is written: a GPU worker's crash writes the tripwire first. Returns
/// what it saw and the worker's stderr.
fn run_worker(
    scratch: &ScratchDir,
    mut command: Command,
    input: &DeviceInput,
    write_tripwire: impl FnOnce(&HangRecord) -> Result<(), String>,
) -> (Supervised, Vec<u8>) {
    let stdout_path = scratch.path().join("stdout.json");
    let stderr_path = scratch.path().join("stderr.txt");
    let child = match start(&mut command, input, scratch, &stdout_path, &stderr_path) {
        Ok(child) => child,
        Err(error) => return (Ok(DeviceOutcome::of(Err(error.into()))), Vec::new()),
    };
    let pid = child.id();
    // No deadline and no memory ceiling (module docs, step 3).
    let exit = match child.wait(&ProcessControl::default()) {
        Ok(exit) => exit,
        Err(stop) => {
            let error = format!("supervise the device worker: {stop}");
            return (Ok(DeviceOutcome::of(Err(error.into()))), Vec::new());
        }
    };
    let stderr = fs::read(&stderr_path).unwrap_or_else(|error| {
        format!("(the device worker's stderr cannot be read: {error})\n").into_bytes()
    });
    let text = String::from_utf8_lossy(&stderr);
    let supervised = match Ended::of(exit.status) {
        Ended::Reported => Ok(reported(&stdout_path, input)),
        Ended::Hang => Err(HangExit),
        Ended::Signal(signal) => Ok(DeviceOutcome::of(Err(ExecutorError::DeviceCrash(crash(
            signal,
            pid,
            input,
            &text,
            write_tripwire,
        ))))),
        Ended::Exited => Ok(DeviceOutcome::of(Err(format!(
            "the device worker {}: {}",
            exit.status,
            bounded(&text)
        )
        .into()))),
    };
    (supervised, stderr)
}

/// Write `input` into `scratch` as the stdin of `command`, send its stdout and stderr to the two
/// paths, and start it.
fn start(
    command: &mut Command,
    input: &DeviceInput,
    scratch: &ScratchDir,
    stdout_path: &Path,
    stderr_path: &Path,
) -> Result<MonitoredChild, String> {
    let stdin_path = scratch.path().join("stdin.json");
    let file = |path: &Path, create: bool| {
        if create {
            fs::File::create(path)
        } else {
            fs::File::open(path)
        }
        .map_err(|error| format!("device worker scratch {}: {error}", path.display()))
    };
    serde_json::to_writer(file(&stdin_path, true)?, input)
        .map_err(|error| format!("encode the device worker's input: {error}"))?;
    command
        .stdin(Stdio::from(file(&stdin_path, false)?))
        .stdout(Stdio::from(file(stdout_path, true)?))
        .stderr(Stdio::from(file(stderr_path, true)?));
    MonitoredChild::spawn(command).map_err(|error| format!("spawn the device worker: {error}"))
}

/// The outcome a worker that exited 0 reported in the last line of `stdout_path`, as every
/// worker's report is ([`super::report`]: a library in the worker may write a line before it),
/// refused unless the worker is this build and ran under `input`'s generated-commands setting and
/// restriction. A device it names beside a refusal must have run under the same restriction, and
/// only a typed refusal names one.
fn reported(stdout_path: &Path, input: &DeviceInput) -> DeviceOutcome {
    let mut refused_on = None;
    let result = reported_run(stdout_path, input, &mut refused_on);
    DeviceOutcome {
        refused_on: refused_on.filter(|_| result.is_err()),
        result,
    }
}

fn reported_run(
    stdout_path: &Path,
    input: &DeviceInput,
    refused_on: &mut Option<OpenedDevice>,
) -> Result<DeviceRun, ExecutorError> {
    let bytes = fs::read(stdout_path)
        .map_err(|error| format!("read the device worker's report: {error}"))?;
    let Reported { result, .. }: Reported<DeviceOutput> =
        serde_json::from_slice(super::last_line(&bytes))
            .map_err(|error| format!("decode the device worker's report: {error}"))?;
    if result.product_fingerprint != TRANSLATOR_FINGERPRINT
        || result.validation_fingerprint != VALIDATION_FINGERPRINT
    {
        return Err(format!(
            "the device worker is build {} (product) {} (validation), not this tool's \
             {TRANSLATOR_FINGERPRINT} {VALIDATION_FINGERPRINT}; its run is not this tool's \
             evidence (rebuild and rerun)",
            result.product_fingerprint, result.validation_fingerprint
        )
        .into());
    }
    if result.generated_commands != input.generated_commands {
        let path = |enabled: bool| {
            if enabled {
                "with VK_EXT_device_generated_commands where supported"
            } else {
                "without VK_EXT_device_generated_commands"
            }
        };
        return Err(format!(
            "the device worker opened its devices {}, and this tool asked for them {}; its run \
             is not this tool's evidence",
            path(result.generated_commands),
            path(input.generated_commands)
        )
        .into());
    }
    if let Some(opened) = result.refused_on {
        if !matches!(result.result, Err(ExecutorError::NotSupported(_))) {
            return Err(
                "the device worker names an opened device beside an outcome that is no typed \
                 refusal"
                    .to_string()
                    .into(),
            );
        }
        opened_under(&opened, input)?;
        recorded_venue(&opened.environment, input)?;
        *refused_on = Some(opened);
    }
    let run = result.result?;
    recorded_venue(&run.environment, input)?;
    if run.census.restricted_to != input.restricted_to {
        let profile = |restriction: &Option<String>| match restriction {
            Some(profile) => format!("restricted to profile {profile}"),
            None => "unrestricted".to_string(),
        };
        return Err(format!(
            "the device worker ran {}, and this tool asked for it {}; its run is not this \
             tool's evidence",
            profile(&run.census.restricted_to),
            profile(&input.restricted_to)
        )
        .into());
    }
    Ok(run)
}

/// Whether `opened`, the device a worker refused a case on, ran under `input`'s restriction, as its
/// environment records it.
fn opened_under(opened: &OpenedDevice, input: &DeviceInput) -> Result<(), String> {
    let restricted_to = opened
        .environment
        .get("profiles_layer")
        .and_then(|restriction| restriction.get("profile"))
        .and_then(serde_json::Value::as_str);
    if restricted_to != input.restricted_to.as_deref() {
        return Err(format!(
            "the device worker refused the case on a device restricted to {restricted_to:?}, and \
             this tool asked for {:?}; its refusal is not this tool's evidence",
            input.restricted_to
        ));
    }
    Ok(())
}

/// Whether a worker's run or refusal recorded the venue profile `input` names, or none where it
/// names no venue: the worker checked its device against that profile before it ran.
fn recorded_venue(environment: &serde_json::Value, input: &DeviceInput) -> Result<(), String> {
    let recorded = crate::c3::VenueProfile::recorded_stem(environment);
    let asked = input.venue.as_ref().map(|venue| venue.profile.as_str());
    if recorded != asked {
        return Err(format!(
            "the device worker recorded venue profile {recorded:?}, and this tool asked for \
             {asked:?}; its run is not this tool's evidence"
        ));
    }
    Ok(())
}

/// The [`DeviceCrash`] of a worker `pid` that `signal` ended while it ran `input`'s case. For a
/// GPU, the tripwire is written first through `write_tripwire` (a [`HangKind::Crashed`] record):
/// what the worker submitted may still be running, and the protocol in `docs/VALIDATION.md`
/// decides what follows.
fn crash(
    signal: i32,
    pid: u32,
    input: &DeviceInput,
    stderr: &str,
    write_tripwire: impl FnOnce(&HangRecord) -> Result<(), String>,
) -> DeviceCrash {
    let backend = SubmissionBackend::vulkan(input.device);
    let mut detail = String::new();
    if backend.is_gpu() {
        let record = HangRecord::now(
            HangKind::Crashed,
            backend,
            "engine device worker",
            format!("worker {pid} died by signal {signal}"),
            wait_timeout(),
            Some(input.case.case_id.clone()),
            None,
        );
        report_incident(&record, write_tripwire);
        detail.push_str("the host GPU tripwire was written (docs/VALIDATION.md); ");
    }
    let stderr = bounded(stderr);
    if stderr.is_empty() {
        detail.push_str("the worker wrote nothing to stderr");
    } else {
        detail.push_str("the worker's stderr: ");
        detail.push_str(&stderr);
    }
    DeviceCrash {
        signal,
        device: input.device,
        detail,
    }
}

/// Serve a device worker when this process was started as one ([`WORKER_ARG`]): run the case its
/// stdin names and exit. Every binary that executes engine cases calls this first in `main`,
/// beside the compile worker's `serve_if_requested`.
pub fn serve_if_requested() {
    if std::env::args_os().nth(1).as_deref() != Some(std::ffi::OsStr::new(WORKER_ARG)) {
        return;
    }
    let status = match serve() {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("engine device worker: {error}");
            1
        }
    };
    std::process::exit(status)
}

/// The worker's run of the [`DeviceInput`] on its stdin, reported on its stdout. A typed refusal
/// or failure of the run is a result, never an `Err`.
fn serve() -> Result<(), String> {
    crate::source::adopt_parent_private_corpus()?;
    let mut bytes = Vec::new();
    std::io::stdin()
        .lock()
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read the device worker's input: {error}"))?;
    let input: DeviceInput = serde_json::from_slice(&bytes)
        .map_err(|error| format!("decode the device worker's input: {error}"))?;
    set_wait_timeout(Duration::from_millis(input.wait_timeout_ms))?;
    if let Some(index) = &input.source_index {
        crate::source::use_source_index(index.clone())?;
    }
    // Before any device opens: a context opened first would decide that there is none.
    if let Some(profile) = &input.restricted_to {
        crate::candidate::restrict_devices_to(profile)?;
    }
    crate::candidate::decide_generated_commands(input.generated_commands)?;
    crate::candidate::echo_validation_messages();
    let DeviceOutcome { result, refused_on } = crate::candidate::engine::run_on_device(&input);
    report(&DeviceOutput {
        product_fingerprint: TRANSLATOR_FINGERPRINT.into(),
        validation_fingerprint: VALIDATION_FINGERPRINT.into(),
        generated_commands: crate::candidate::requested_generated_commands(),
        result,
        refused_on,
    })
}

/// `Vec<u8>` as one base64 string.
mod base64_bytes {
    use base64::Engine as _;
    use serde::{de::Error as _, Deserialize as _, Deserializer, Serializer};

    pub(super) fn serialize<S: Serializer>(bytes: &[u8], serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&base64::engine::general_purpose::STANDARD.encode(bytes))
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Vec<u8>, D::Error> {
        let text = String::deserialize(deserializer)?;
        base64::engine::general_purpose::STANDARD
            .decode(text)
            .map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests;
