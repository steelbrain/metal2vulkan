//! The killable worker boundary of validation sweeps.
//!
//! The engine is in-process only: it enforces its budgets deterministically (work units, the
//! byte ledger, the pre-emission size estimate, a transient wall deadline). The validation worker
//! is the backstop around it: one attempt per child process, supervised as a process group under
//! the 20-second and 500-MiB ceilings ([`process`]), with a counting global allocator ([`alloc`])
//! that both enforces the byte budget inside the child and measures the peak attributable bytes
//! that the ledger must account for.
//!
//! [`run_json`] is the one parent-side exchange: the input goes to the child's stdin from a
//! scratch file (no feeder thread per row), the child writes one JSON report to stdout
//! ([`report`]: its result, its counting allocator's process peak and its resident high water),
//! and every scratch file is removed before the call returns. The parent samples the group's RSS
//! every 10 ms, which can miss a short peak or a whole worker that runs between two samples: the
//! result's peak also takes the high water the worker states, and the allocator's peak, counted
//! exactly in the child, is recorded beside it in every result ([`Measured`]). Each member is
//! charged only memory of its own, never its parent's while it still runs in it ([`process`]),
//! and a supervising process gives its large freed buffers back to the system
//! ([`alloc::release_large_frees`]), so its own RSS does not grow from sweep to sweep.
//! [`compile`] runs every engine pipeline compile of a device-holding tool (L4 acceptance, the
//! engine candidate, the device tests) through it, and [`audit`] every attempt of the canon and
//! bitcode-reader audit lanes. [`device`] is the other boundary: the
//! engine candidate's device work runs in a child that the parent waits for without a deadline
//! or a ceiling (the child bounds its own device waits), so a driver fault ends the child, not the
//! tool. A translation worker whose attempt linked a module runs `spirv-val` in a fresh image of
//! itself ([`l3`]), so the attempt's peak is the larger of its translation's and its validation's,
//! never their sum.

pub mod alloc;
pub mod audit;
pub mod compile;
pub mod device;
pub mod l3;
pub mod process;

pub use process::{MonitoredChild, ProcessControl, ProcessExit, ProcessFailure, ProcessStop};

use crate::engine::{FailureKind, Outcome};
use crate::ScratchDir;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::{Read as _, Write as _};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// The per-attempt wall ceiling, from handing the input to the worker through its report.
pub const DEADLINE: Duration = Duration::from_secs(20);

/// The per-attempt memory ceiling: the worker's process-group RSS (sampled by the parent) and
/// the live heap bytes its counting allocator admits above its startup state.
pub const MEMORY_LIMIT_BYTES: u64 = 500 * 1024 * 1024;

/// What the parent measured around a worker that ran, whether or not it reported a result.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Measured {
    /// From preparing the input through the result or the stop.
    pub elapsed: Duration,
    /// The peak resident size up to the result or the stop: for a worker that exited, the larger
    /// of the sampled process-group RSS and the kernel's peak for the worker at its reap
    /// ([`ProcessExit::peak_resident_bytes`]), and for one that reported, of its own resident
    /// high water as its report states it ([`report`]), so known for every worker that reported
    /// on Linux and macOS; for a stopped one, the sampled peak alone.
    pub peak_resident_bytes: Option<u64>,
    /// The worker's counting-allocator peak ([`alloc::peak_bytes`]): the most live heap bytes it
    /// held at once, which the RSS sampler can miss. `None` when the worker reported no result
    /// (a stop, an exit, an unreadable report) or counts no allocations.
    pub allocator_peak_bytes: Option<u64>,
}

impl Measured {
    /// The larger of the sampled RSS and the allocator peak, whichever are known: the measure
    /// the 500-MiB ceiling ([`MEMORY_LIMIT_BYTES`]) is checked against.
    pub fn peak_bytes(&self) -> Option<u64> {
        self.peak_resident_bytes.max(self.allocator_peak_bytes)
    }
}

/// What a worker writes to stdout ([`report`]): its result, its counting allocator's process
/// peak, and its own resident high water.
#[derive(Serialize)]
struct Report<'a, R> {
    allocator_peak_bytes: Option<u64>,
    resident_peak_bytes: Option<u64>,
    result: &'a R,
}

/// [`Report`] as the parent decodes it.
#[derive(Deserialize)]
struct Reported<R> {
    allocator_peak_bytes: Option<u64>,
    resident_peak_bytes: Option<u64>,
    result: R,
}

/// What a worker states of its own memory in its report: its counting allocator's process peak
/// ([`alloc::peak_bytes`]) and its resident high water ([`process::own_peak_resident_bytes`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct Peaks {
    allocator_peak_bytes: Option<u64>,
    resident_peak_bytes: Option<u64>,
}

impl Peaks {
    /// This process's, now.
    fn now() -> Peaks {
        Peaks {
            allocator_peak_bytes: alloc::peak_bytes(),
            resident_peak_bytes: process::own_peak_resident_bytes(),
        }
    }

    /// The larger of each of `self`'s and `other`'s.
    fn max(self, other: Peaks) -> Peaks {
        Peaks {
            allocator_peak_bytes: self.allocator_peak_bytes.max(other.allocator_peak_bytes),
            resident_peak_bytes: self.resident_peak_bytes.max(other.resident_peak_bytes),
        }
    }
}

/// Write `result` to stdout as this worker's one report, with its counting allocator's process
/// peak ([`alloc::peak_bytes`]) and its own resident high water
/// ([`process::own_peak_resident_bytes`]), for [`run_json`] to decode: one line of JSON, the last
/// of its stdout. Every worker child reports through this, as its last act.
///
/// A library the worker loaded may write to stdout too: the validation layer names its log file
/// there, through C's buffered stdout, which flushes at exit. So the report is a line of its own,
/// and once it is written the process's stdout is its stderr: what anything writes after the
/// report, the C buffer flushed at exit included, reaches the worker's stderr, not its report.
pub fn report<R: Serialize>(result: &R) -> Result<(), String> {
    report_with_peaks(result, Peaks::now())
}

/// [`report`] with the peaks `peaks`: the L3 stage's, which states the larger of the
/// translation's and its own ([`l3::serve`]).
fn report_with_peaks<R: Serialize>(result: &R, peaks: Peaks) -> Result<(), String> {
    let mut stdout = std::io::stdout().lock();
    stdout
        .write_all(b"\n")
        .map_err(|error| format!("write the worker's report: {error}"))?;
    serde_json::to_writer(
        &mut stdout,
        &Report {
            allocator_peak_bytes: peaks.allocator_peak_bytes,
            resident_peak_bytes: peaks.resident_peak_bytes,
            result,
        },
    )
    .map_err(|error| format!("encode the worker's report: {error}"))?;
    stdout
        .write_all(b"\n")
        .and_then(|()| stdout.flush())
        .map_err(|error| format!("flush the worker's report: {error}"))?;
    #[cfg(unix)]
    {
        // SAFETY: duplicates the process's own stderr descriptor onto its stdout descriptor; both
        // stay open for the process's life, and the report written above is already flushed.
        if unsafe { libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO) } < 0 {
            return Err(format!(
                "point stdout at stderr after the worker's report: {}",
                std::io::Error::last_os_error()
            ));
        }
    }
    Ok(())
}

/// A worker child's process once its arguments selected it (each worker's `serve_if_requested`):
/// refuse to serve unless `counting` (the process counts its allocations, [`alloc::installed`]),
/// set the 500-MiB allocator budget, read the input from stdin, `serve` it under the deadline from
/// then, and [`report`] its result; an error is written to stderr after `worker`'s name. Returns
/// the exit status: 0 when a result was written, 1 otherwise.
pub(crate) fn serve_process<R: Serialize>(
    worker: &str,
    counting: bool,
    serve: impl FnOnce(&[u8], Instant) -> Result<R, String>,
) -> i32 {
    let served = (|| {
        // The worker recipe's step 2: the in-child budget is the allocator's, set once at startup.
        if !counting {
            return Err("this binary does not install the counting allocator".to_string());
        }
        crate::source::adopt_parent_private_corpus()?;
        alloc::enable_budget(MEMORY_LIMIT_BYTES);
        let started = Instant::now();
        let mut input = Vec::new();
        std::io::stdin()
            .lock()
            .read_to_end(&mut input)
            .map_err(|error| format!("read the worker's input: {error}"))?;
        let output = serve(&input, started)?;
        drop(input);
        report(&output)
    })();
    match served {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("{worker}: {error}");
            1
        }
    }
}

/// Why a killable worker produced no result. Every stop after the worker was spawned carries what
/// the parent measured up to it, so a row the boundary stopped at the deadline or the memory
/// ceiling is as measured as a finished one.
#[derive(Debug)]
pub enum WorkerFailure {
    /// The parent could not prepare or start the worker (scratch, I/O, spawn).
    Setup(String),
    /// The watchdog stopped the worker (deadline, memory, cancellation, monitoring), with its
    /// bounded stderr when the stop happened after spawning.
    Process(ProcessFailure, Option<String>, Measured),
    /// The worker exited unsuccessfully (including an allocator-budget abort), with its stderr.
    Exit(String, Measured),
    /// The worker exited successfully but its result could not be read.
    Output(String, Measured),
}

impl WorkerFailure {
    /// The failure text. The old engine's census keys failure shapes on it, so it is stable.
    pub fn message(&self) -> String {
        match self {
            WorkerFailure::Setup(error) | WorkerFailure::Output(error, _) => error.clone(),
            WorkerFailure::Process(failure, None, _) => failure.to_string(),
            WorkerFailure::Process(failure, Some(stderr), _) => format!("{failure}: {stderr}"),
            WorkerFailure::Exit(stderr, _) => format!(
                "translation worker failed under {} MiB memory limit: {stderr}",
                MEMORY_LIMIT_BYTES / (1024 * 1024),
            ),
        }
    }

    /// The typed outcome class of an engine attempt the worker boundary stopped.
    pub fn outcome(&self) -> Outcome {
        let failure = match self {
            WorkerFailure::Setup(_) => FailureKind::Tooling,
            WorkerFailure::Process(ProcessFailure::DeadlineExceeded, ..) => FailureKind::Timeout,
            WorkerFailure::Process(ProcessFailure::Cancelled, ..) => FailureKind::Cancelled,
            WorkerFailure::Process(ProcessFailure::MemoryLimitExceeded { .. }, ..) => {
                FailureKind::Oom
            }
            WorkerFailure::Process(ProcessFailure::Monitoring(_), ..) => FailureKind::Tooling,
            WorkerFailure::Exit(..) | WorkerFailure::Output(..) => FailureKind::WorkerExit,
        };
        Outcome::Failed { failure }
    }

    /// What the parent measured before the stop; `None` when no worker was spawned.
    pub fn measured(&self) -> Option<Measured> {
        match self {
            WorkerFailure::Setup(_) => None,
            WorkerFailure::Process(_, _, measured)
            | WorkerFailure::Exit(_, measured)
            | WorkerFailure::Output(_, measured) => Some(*measured),
        }
    }
}

/// A worker's decoded result and what the parent measured around it.
#[derive(Debug)]
pub struct WorkerRun<R> {
    pub result: R,
    /// The larger of the peak sampled process-group RSS (the worker and every descendant, e.g.
    /// `spirv-val`, summed), the kernel's peak for the worker alone at its reap
    /// ([`ProcessExit::peak_resident_bytes`]), and the worker's own resident high water as its
    /// report states it ([`report`]).
    pub peak_resident_bytes: Option<u64>,
    /// The worker's counting-allocator peak, as its report states it ([`report`]).
    pub allocator_peak_bytes: Option<u64>,
    /// Wall time from preparing the input through decoding the result.
    pub elapsed: Duration,
}

impl<R> WorkerRun<R> {
    /// What the parent measured around this run.
    pub fn measured(&self) -> Measured {
        Measured {
            elapsed: self.elapsed,
            peak_resident_bytes: self.peak_resident_bytes,
            allocator_peak_bytes: self.allocator_peak_bytes,
        }
    }
}

/// The supervising process's allocator policy, set before its first worker: a large buffer it
/// frees goes back to the system ([`alloc::release_large_frees`]).
static SUPERVISOR_HEAP: std::sync::Once = std::sync::Once::new();

/// Run `command` as a killable worker under [`DEADLINE`] and [`MEMORY_LIMIT_BYTES`]:
/// `write_input` fills the file that becomes the child's stdin, and the last line of the child's
/// stdout is decoded as its one [`report`] of an `R` (a test binary serving a worker, the lib
/// tests' compile worker, prints its harness's line before it). The deadline covers input
/// preparation, the child, and result decoding; the process group is killed on every exit path
/// and the scratch directory is removed.
pub fn run_json<R: DeserializeOwned>(
    command: Command,
    purpose: &str,
    write_input: impl FnOnce(fs::File) -> Result<(), String>,
) -> Result<WorkerRun<R>, WorkerFailure> {
    run_json_within(command, purpose, DEADLINE, write_input)
}

/// [`run_json`] under the wall ceiling `deadline` in place of [`DEADLINE`], the memory ceiling
/// unchanged: for a worker that runs a bounded number of attempts, each in its own worker under
/// the per-attempt ceilings (the authored compile census's case worker).
pub fn run_json_within<R: DeserializeOwned>(
    mut command: Command,
    purpose: &str,
    deadline: Duration,
    write_input: impl FnOnce(fs::File) -> Result<(), String>,
) -> Result<WorkerRun<R>, WorkerFailure> {
    SUPERVISOR_HEAP.call_once(alloc::release_large_frees);
    let started = Instant::now();
    let setup = WorkerFailure::Setup;
    let scratch = ScratchDir::new(purpose).map_err(setup)?;
    let stdin_path = scratch.path().join("stdin.json");
    let stdout_path = scratch.path().join("stdout.json");
    let stderr_path = scratch.path().join("stderr.txt");
    let stdin_file = fs::File::create(&stdin_path)
        .map_err(|error| setup(format!("create {}: {error}", stdin_path.display())))?;
    write_input(stdin_file).map_err(setup)?;
    command
        // A regular file lets the child decode arbitrarily large inputs under the watchdog
        // without one short-lived parent feeder thread per row, so thread count stays bounded by
        // the sweep's jobs over a full census.
        .stdin(Stdio::from(fs::File::open(&stdin_path).map_err(
            |error| setup(format!("open {}: {error}", stdin_path.display())),
        )?))
        .stdout(Stdio::from(fs::File::create(&stdout_path).map_err(
            |error| setup(format!("create {}: {error}", stdout_path.display())),
        )?))
        .stderr(Stdio::from(fs::File::create(&stderr_path).map_err(
            |error| setup(format!("create {}: {error}", stderr_path.display())),
        )?));
    let control = ProcessControl {
        deadline: Some(started + deadline),
        memory_limit_bytes: Some(MEMORY_LIMIT_BYTES),
        cancelled: None,
    };
    let child = MonitoredChild::spawn(&mut command)
        .map_err(|error| setup(format!("spawn translation worker: {error}")))?;
    let exit = child.wait(&control);
    let stderr = || fs::read_to_string(&stderr_path).unwrap_or_default();
    let measured = |peak_resident_bytes| Measured {
        elapsed: started.elapsed(),
        peak_resident_bytes,
        allocator_peak_bytes: None,
    };
    let exit = exit.map_err(|stop| {
        WorkerFailure::Process(
            stop.failure,
            Some(bounded(&stderr())),
            measured(stop.peak_resident_bytes),
        )
    })?;
    let exited = exit.peak_resident_bytes;
    if !exit.status.success() {
        return Err(WorkerFailure::Exit(
            single_line(&stderr()),
            measured(exited),
        ));
    }
    let output = fs::read(&stdout_path).map_err(|error| {
        WorkerFailure::Output(
            format!("read {}: {error}", stdout_path.display()),
            measured(exited),
        )
    })?;
    let Reported {
        allocator_peak_bytes,
        resident_peak_bytes,
        result,
    } = serde_json::from_slice(last_line(&output)).map_err(|error| {
        WorkerFailure::Output(
            format!("decode translation worker result: {error}"),
            measured(exited),
        )
    })?;
    // `None` orders below every `Some`.
    let peak = exited.max(resident_peak_bytes);
    drop(scratch);
    control.check().map_err(|error| {
        WorkerFailure::Process(
            error,
            None,
            Measured {
                allocator_peak_bytes,
                ..measured(peak)
            },
        )
    })?;
    Ok(WorkerRun {
        result,
        peak_resident_bytes: peak,
        allocator_peak_bytes,
        elapsed: started.elapsed(),
    })
}

/// The last line of `output` that holds anything but whitespace (a worker's [`report`]), or none.
fn last_line(output: &[u8]) -> &[u8] {
    output
        .split(|byte| *byte == b'\n')
        .rfind(|line| !line.trim_ascii().is_empty())
        .unwrap_or_default()
}

fn single_line(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn bounded(value: &str) -> String {
    single_line(value).chars().take(4096).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shell(script: &str) -> Command {
        let mut command = Command::new("sh");
        command.args(["-c", script]);
        command
    }

    #[cfg(unix)]
    #[test]
    fn a_worker_reads_its_input_file_and_reports_json() {
        let run: WorkerRun<Vec<u32>> = run_json(
            shell(
                "read line; echo \"{\\\"allocator_peak_bytes\\\": 4096, \\\"result\\\": [$line, 2]}\"",
            ),
            "worker-test",
            |mut file| {
                use std::io::Write as _;
                writeln!(file, "1").map_err(|error| error.to_string())
            },
        )
        .unwrap();
        assert_eq!(run.result, vec![1, 2]);
        assert!(run.elapsed < DEADLINE);
        // The allocator's peak is recorded beside the sampled RSS.
        assert_eq!(run.allocator_peak_bytes, Some(4096));
        let measured = run.measured();
        assert_eq!(measured.allocator_peak_bytes, Some(4096));
        assert_eq!(measured.peak_resident_bytes, run.peak_resident_bytes);
    }

    /// A worker's report is the last line of its stdout: what a test harness serving the worker
    /// printed before it is not the report.
    #[cfg(unix)]
    #[test]
    fn a_worker_s_report_is_the_last_line_of_its_stdout() {
        let run: WorkerRun<u32> = run_json(
            shell("printf '\\nrunning 1 test\\n{\"allocator_peak_bytes\": null, \"result\": 7}'"),
            "worker-test",
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(run.result, 7);
        let failure = run_json::<u32>(
            shell("printf '{\"allocator_peak_bytes\": null, \"result\": 7}\\nrunning 1 test\\n'"),
            "worker-test",
            |_| Ok(()),
        )
        .unwrap_err();
        assert!(
            matches!(&failure, WorkerFailure::Output(message, _) if message.contains("decode")),
            "{failure:?}"
        );
    }

    /// The bug class: what a library in a worker writes through C's buffered stdout, which
    /// flushes at exit, after the report on the report's own stream, is not the report's: the
    /// worker's report decodes. The fixture, in a fresh process, writes a line before its
    /// [`report`] and one after it, both through C's stdout, and exits as a worker does.
    #[cfg(unix)]
    mod a_library_s_stdout {
        use super::*;
        use std::process::Command;

        const FIXTURE: &str = "worker::tests::a_library_s_stdout::reporting_fixture";
        const MODE: &str = "M2V_TEST_LIBRARY_STDOUT";

        /// The fixture: no-op unless [`MODE`] is set.
        #[test]
        fn reporting_fixture() {
            if std::env::var_os(MODE).is_none() {
                return;
            }
            // SAFETY: `puts` writes a NUL-terminated string to C's stdout.
            unsafe { libc::puts(c"a library's line before the report".as_ptr()) };
            report(&7u32).unwrap();
            // SAFETY: as above.
            unsafe { libc::puts(c"a library's line after the report".as_ptr()) };
            std::process::exit(0);
        }

        #[test]
        fn is_not_the_worker_s_report() {
            let mut command = Command::new(std::env::current_exe().unwrap());
            command
                .args(["--exact", FIXTURE, "--nocapture"])
                .env(MODE, "1");
            let run: WorkerRun<u32> = run_json(command, "worker-test", |_| Ok(())).unwrap();
            assert_eq!(run.result, 7);
        }
    }

    /// A worker's report states its counting allocator's process peak and its resident high
    /// water, which survive the round trip; a peak the sampler missed still sets the ceiling
    /// measure.
    #[test]
    fn a_report_carries_the_allocator_peak_and_it_bounds_the_ceiling_measure() {
        let peaks = Peaks::now();
        let bytes = serde_json::to_vec(&Report {
            allocator_peak_bytes: peaks.allocator_peak_bytes,
            resident_peak_bytes: peaks.resident_peak_bytes,
            result: &7u32,
        })
        .unwrap();
        let reported: Reported<u32> = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(reported.result, 7);
        // The lib's test binary installs the counting allocator (`alloc::tests`).
        assert!(reported.allocator_peak_bytes.is_some_and(|peak| peak > 0));
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(reported.resident_peak_bytes.is_some_and(|peak| peak > 0));
        }
        assert_eq!(reported.resident_peak_bytes, peaks.resident_peak_bytes);

        let sampled_low = Measured {
            elapsed: Duration::ZERO,
            peak_resident_bytes: Some(1 << 20),
            allocator_peak_bytes: Some(MEMORY_LIMIT_BYTES + 1),
        };
        assert_eq!(sampled_low.peak_bytes(), Some(MEMORY_LIMIT_BYTES + 1));
        let unreported = Measured {
            allocator_peak_bytes: None,
            ..sampled_low
        };
        assert_eq!(unreported.peak_bytes(), Some(1 << 20));
        assert_eq!(Measured::default().peak_bytes(), None);
    }

    /// A worker that execs and exits between two of the sampler's ticks leaves no sample, and on
    /// Linux the kernel's figure at its reap may be this supervisor's own high water, which is not
    /// used (`process::own_kernel_peak`): the run's peak takes the high water the worker's report
    /// states.
    #[cfg(unix)]
    #[test]
    fn a_run_s_peak_takes_the_resident_high_water_its_report_states() {
        // Far above anything `sh` holds and this process's own high water, so no sample and no
        // kernel figure stands in for it.
        let stated = 1u64 << 40;
        let run: WorkerRun<u32> = run_json(
            shell(&format!(
                "echo '{{\"allocator_peak_bytes\": null, \"resident_peak_bytes\": {stated}, \
                 \"result\": 7}}'"
            )),
            "worker-test",
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(run.result, 7);
        assert_eq!(run.peak_resident_bytes, Some(stated));
        assert_eq!(run.measured().peak_bytes(), Some(stated));
    }

    #[cfg(unix)]
    #[test]
    fn worker_failures_are_typed() {
        let exit =
            run_json::<u32>(shell("echo nope >&2; exit 3"), "worker-test", |_| Ok(())).unwrap_err();
        assert!(matches!(&exit, WorkerFailure::Exit(stderr, _) if stderr == "nope"));
        assert_eq!(exit.outcome().key(), "worker_exit");
        // A stop after spawning is measured like a result.
        let measured = exit.measured().unwrap();
        assert!(measured.elapsed > Duration::ZERO && measured.elapsed < DEADLINE);

        let output =
            run_json::<u32>(shell("echo not-json"), "worker-test", |_| Ok(())).unwrap_err();
        assert!(matches!(output, WorkerFailure::Output(..)));
        assert!(output.measured().is_some());

        let setup =
            run_json::<u32>(shell("true"), "worker-test", |_| Err("no input".into())).unwrap_err();
        assert!(matches!(&setup, WorkerFailure::Setup(error) if error == "no input"));
        assert!(setup.outcome().is_transient());
        assert_eq!(setup.measured(), None);
    }
}
