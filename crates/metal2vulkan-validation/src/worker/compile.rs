//! Engine pipeline compiles under the worker boundary.
//!
//! Every validation path that compiles an engine pipeline for a device (L4 driver acceptance,
//! `corpus-pipeline`, and the engine candidate executor behind
//! `corpus-vulkan`) compiles in a killable child
//! process, never in the process that holds the Vulkan device. [`compile_compute`],
//! [`compile_compute_request`], and [`compile_render_request`] run one child per pipeline
//! through [`super::run_json`], so the 20-second
//! deadline (input preparation through decoding the artifact), the 500-MiB process-group RSS
//! limit, and the process-group kill apply to the compile.
//!
//! The recipe (`docs/HOWTO.md` section 5), step by step:
//!
//! 1. The input is the source of each library the request's functions come from (one, or a
//!    draw's two: the case's and its generated companion's), then each library of the functions
//!    they link, each a [`RowSource`]: its bitcode, or AIR text (a public row that retains none, a
//!    generated companion) the child assembles with `llvm-as` under its own deadline; the
//!    [`PipelineRequest`] (kind, entries, function constants, linked functions); and the device's
//!    `DeviceDescription` and `DevicePolicy` ([`WorkerDevice`]), from which the child rebuilds the
//!    parent's `Device`. The parent writes it to the file that becomes the child's stdin.
//! 2. The child ([`serve_if_requested`]) refuses to serve without the counting allocator, sets its
//!    500-MiB budget, and compiles through the public API ([`compile_pipeline`]:
//!    `Device::compile_compute_pipeline` or `Device::compile_render_pipeline`) under a `Budget`
//!    with the 20-s deadline.
//! 3. The parent polls the group (deadline, RSS) and kills it on any breach ([`super::run_json`]).
//! 4. The child writes the pipeline's cache artifact bytes (`PipelineArtifact::to_bytes`) to the
//!    file the parent names in [`ARTIFACT_FILE`], a chunk at a time (`PipelineArtifact::write_to`),
//!    so reporting holds little beside the pipeline: a constant blob the ledger admits is never
//!    encoded whole in memory, let alone as base64. It reports the artifact's length with the table
//!    handles of each request function's linked functions ([`crate::acceptance::LinkedHandles`]),
//!    together with the SHA-256 of the input bytes it decoded and its product fingerprint. The
//!    parent requires that digest to be its own input's and that fingerprint to be its own (an
//!    executable rebuilt since the parent started is another compiler, whose pipeline the parent's
//!    observations would misattribute), requires the file to hold the reported length, decodes the
//!    bytes with `PipelineArtifact::from_bytes` for the kind it asked for against its own `Device`
//!    (so bytes of another profile are `WrongDevice`, and stale, corrupt, or other-kind bytes are
//!    refused), requires the handles to answer its request (`PipelineRequest::check_handles`), and
//!    re-checks the deadline after decoding.
//! 5. A typed product refusal comes back as its [`Outcome`] (deterministic), a library whose
//!    source gives no bitcode as the row's outcome (`BitcodeError::attempt`), and pipeline
//!    inputs the tool does not build for the function as [`NotCompiled::Unmodelled`]; a stop at
//!    the boundary is a [`WorkerFailure`] (the caller reports it, never retries with another
//!    input).
//!
//! The recipe's parent-side key check (`Device::compute_pipeline_key` of the request) needs a
//! `Function`, which for a bare AIR module (every validation input) exists only after
//! `Library::from_air` decodes the whole module: computing it in the parent would put exactly the
//! input-scaled decode the boundary exists to contain back into the parent. The input digest
//! stands in for it here: the parent verifies that the child compiled the bytes it wrote.

use super::alloc;
use super::{run_json, Measured, ProcessFailure, WorkerFailure, DEADLINE};
use crate::acceptance::{
    compile_pipeline, AcceptancePipeline, CompiledRequest, LinkedHandles, PipelineCompileError,
    PipelineRequest,
};
use crate::engine::{bounded_detail, Outcome, RowSource};
use crate::hash::sha256_bytes;
use crate::observation::TRANSLATOR_FINGERPRINT;
use crate::ScratchDir;
use metal2vulkan::cache::{ArtifactError, PipelineArtifact};
use metal2vulkan::device::{Device, DeviceDescription, DevicePolicy};
use metal2vulkan::error::Error;
use metal2vulkan::library::FunctionConstantValues;
use metal2vulkan::pipeline::{ComputePipeline, RenderPipeline};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

/// The first argument that starts a validation binary as a compile worker.
pub const WORKER_ARG: &str = "--engine-compile-worker";

/// The environment variable naming the file a compile worker writes its pipeline's artifact to,
/// in a scratch directory the parent owns and removes.
pub const ARTIFACT_FILE: &str = "M2V_COMPILE_WORKER_ARTIFACT";

/// The executable that serves compile workers: a validation binary that calls
/// [`serve_if_requested`] first in `main` and installs
/// [`CountingAllocator`](super::alloc::CountingAllocator) as its global allocator, or this crate's
/// lib test binary, which installs it too (`alloc::tests`) and serves through a fixture test
/// (`CompileWorker::lib_tests`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompileWorker {
    program: PathBuf,
    /// The arguments before the worker's own: none for a binary, which reads its worker argument
    /// first; the lib test binary's selection of its fixture test.
    selection: Vec<String>,
    /// The variables the worker starts with: none for a binary; the fixture's role for the lib
    /// test binary.
    environment: Vec<(String, String)>,
}

impl CompileWorker {
    /// The running executable (every binary that compiles through this module serves workers).
    pub fn current() -> Result<CompileWorker, String> {
        std::env::current_exe()
            .map(CompileWorker::program)
            .map_err(|error| format!("resolve the compile worker executable: {error}"))
    }

    /// `program`, a validation binary that serves [`WORKER_ARG`] (integration tests name one with
    /// `env!("CARGO_BIN_EXE_…")`).
    pub fn program(program: impl Into<PathBuf>) -> CompileWorker {
        CompileWorker {
            program: program.into(),
            selection: Vec::new(),
            environment: Vec::new(),
        }
    }

    /// This crate's lib test binary, the running one, serving each compile through its fixture
    /// test ([`LIB_TESTS_FIXTURE`]) as a binary serves it: in a child of its own process group,
    /// under the deadline, the process-group RSS limit and the allocator budget, its scratch
    /// removed. Any selection that runs the lib's tests holds it, where a `--lib` build builds no
    /// package binary.
    #[cfg(test)]
    pub(crate) fn lib_tests() -> Result<CompileWorker, String> {
        CompileWorker::lib_tests_as(LIB_TESTS_SERVE)
    }

    /// [`CompileWorker::lib_tests`], its fixture serving `role` ([`LIB_TESTS_ROLE`]). The test
    /// harness parses the binary's arguments, so the selection runs the fixture alone and ends
    /// with `--`: the worker argument after it is a filter that names no test. The harness's
    /// terse output prints one line before the test runs and none of the test's name, so the
    /// worker's report stays the last line of its stdout (`worker::run_json`).
    #[cfg(test)]
    fn lib_tests_as(role: &str) -> Result<CompileWorker, String> {
        let program = std::env::current_exe()
            .map_err(|error| format!("resolve the lib test binary: {error}"))?;
        Ok(CompileWorker {
            program,
            selection: ["--exact", LIB_TESTS_FIXTURE, "--nocapture", "--quiet", "--"]
                .map(String::from)
                .to_vec(),
            environment: vec![(LIB_TESTS_ROLE.to_string(), role.to_string())],
        })
    }

    /// A command that starts the worker's executable, before its worker argument: also the
    /// engine device worker's (`super::device`), which every tool that runs cases serves.
    pub(crate) fn command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.selection).envs(
            self.environment
                .iter()
                .map(|(name, value)| (name.as_str(), value.as_str())),
        );
        command
    }
}

/// The lib test binary's compile worker fixture ([`CompileWorker::lib_tests`]).
#[cfg(test)]
const LIB_TESTS_FIXTURE: &str = "worker::compile::tests::compile_worker_fixture";

/// The variable that makes [`LIB_TESTS_FIXTURE`] a compile worker child, naming what it serves:
/// [`LIB_TESTS_SERVE`], or one of its limit probes.
#[cfg(test)]
const LIB_TESTS_ROLE: &str = "M2V_TEST_COMPILE_WORKER";

/// [`LIB_TESTS_ROLE`]'s value for a compile.
#[cfg(test)]
const LIB_TESTS_SERVE: &str = "serve";

/// The stack [`LIB_TESTS_FIXTURE`] serves on: a binary serves on its main thread, whose stack is
/// 8 MiB (Linux's default stack limit; macOS's main thread), where the harness runs a test on a
/// 2-MiB thread.
#[cfg(test)]
const LIB_TESTS_STACK: usize = 8 << 20;

/// The device a worker compiles for: the description and policy the child rebuilds it from, and
/// the parent's `Device` built from exactly that pair, against which the artifact is decoded.
#[derive(Clone, Debug)]
pub struct WorkerDevice {
    description: DeviceDescription,
    policy: DevicePolicy,
    device: Device,
}

impl WorkerDevice {
    /// Derive the `Device` of `description` under `policy` (`Device::from_queries`'s refusals are its own).
    pub fn new(
        description: DeviceDescription,
        policy: DevicePolicy,
    ) -> Result<WorkerDevice, Error> {
        let device = Device::from_queries(description.clone(), policy)?;
        Ok(WorkerDevice {
            description,
            policy,
            device,
        })
    }

    /// The parent's `Device`.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The description the child rebuilds the `Device` from.
    pub fn description(&self) -> &DeviceDescription {
        &self.description
    }

    /// The policy the child rebuilds the `Device` under.
    pub fn policy(&self) -> &DevicePolicy {
        &self.policy
    }
}

/// Why a worker compile built no pipeline.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NotCompiled {
    /// The product refused or failed, or a library gave no bitcode: the typed outcome and a
    /// bounded detail.
    Product(Outcome, String),
    /// The request asks for pipeline inputs the tool does not build for this function (an L4
    /// row's vertex descriptor or companion, `acceptance::row`): a modelled limit of the tool.
    Unmodelled(crate::acceptance::row::Unmodelled),
}

/// A pipeline a worker compiled, with the table handles of the functions each of its request's
/// functions links ([`CompiledRequest::handles`]).
#[derive(Debug)]
pub struct Compiled<P> {
    pub pipeline: P,
    pub handles: Vec<LinkedHandles>,
}

/// What one worker compile returned.
#[derive(Debug)]
pub struct WorkerCompile<P> {
    /// The pipeline, or why none was built.
    pub result: Result<Compiled<P>, NotCompiled>,
    /// Peak live heap bytes the compile allocated in the worker (its counting allocator).
    pub allocator_peak: Option<u64>,
    /// From preparing the input through decoding the artifact, and the group's peak RSS.
    pub measured: Measured,
}

/// Compile kernel `entry` of `library` with `constants` into a compute pipeline for `device`, in
/// a killable worker.
pub fn compile_compute(
    worker: &CompileWorker,
    device: &WorkerDevice,
    library: &RowSource<'_>,
    entry: &str,
    constants: &FunctionConstantValues,
) -> Result<WorkerCompile<ComputePipeline>, WorkerFailure> {
    let request = PipelineRequest::Compute {
        entry: entry.into(),
        constants: constants.clone(),
    };
    compile(worker, device, std::slice::from_ref(library), &request)
}

/// Compile `request`, a compute pipeline's (`PipelineRequest::BoundCompute`, or an L4 kernel row's
/// `PipelineRequest::Row`), from `libraries` as [`PipelineRequest::libraries`] orders them (the
/// kernel's, then those of the functions it links) for `device`, in a killable worker.
pub fn compile_compute_request(
    worker: &CompileWorker,
    device: &WorkerDevice,
    libraries: &[RowSource<'_>],
    request: &PipelineRequest,
) -> Result<WorkerCompile<ComputePipeline>, WorkerFailure> {
    compile(worker, device, libraries, request)
}

/// Compile `request`, a render pipeline's (`PipelineRequest::VertexDraw` from the Metal oracle's
/// descriptor template, `PipelineRequest::Draw` of a case and its generated companion, either with
/// the bind shape's inputs, or an L4 vertex or fragment row's `PipelineRequest::Row`), from
/// `libraries` as [`PipelineRequest::libraries`] orders them (the stage libraries, then those of
/// the functions each links) for `device`, in a killable worker: a case and its companion compile
/// in one child, under one deadline and one limit.
pub fn compile_render_request(
    worker: &CompileWorker,
    device: &WorkerDevice,
    libraries: &[RowSource<'_>],
    request: &PipelineRequest,
) -> Result<WorkerCompile<RenderPipeline>, WorkerFailure> {
    compile(worker, device, libraries, request)
}

/// The worker's stdin.
#[derive(Debug, Serialize, Deserialize)]
struct CompileInput<'a> {
    description: DeviceDescription,
    policy: DevicePolicy,
    request: PipelineRequest,
    /// The source of each library the compile takes, in the request's order.
    libraries: Vec<RowSource<'a>>,
}

/// The worker's stdout.
#[derive(Debug, Serialize, Deserialize)]
struct CompileOutput {
    /// The worker's product fingerprint ([`TRANSLATOR_FINGERPRINT`]).
    product_fingerprint: String,
    /// SHA-256 of the input bytes the worker decoded.
    input_sha256: String,
    result: CompileResult,
    /// Peak live heap bytes the compile allocated.
    allocator_peak: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
enum CompileResult {
    /// The length of the pipeline's cache artifact (`PipelineArtifact::to_bytes`), which the
    /// worker wrote to its [`ARTIFACT_FILE`], and the table handles of each request function's
    /// linked functions ([`CompiledRequest::handles`]).
    Compiled {
        artifact_bytes: u64,
        handles: Vec<LinkedHandles>,
    },
    /// The product refused or failed, or a library gave no bitcode: its typed outcome and a
    /// bounded detail.
    NotCompiled { outcome: Outcome, detail: String },
    /// The tool does not build the request's pipeline inputs for this function: its kind and
    /// detail.
    Unmodelled {
        reason: crate::acceptance::row::Unmodelled,
    },
}

/// Compile `request` over `libraries` (each library's source, in the request's order) for
/// `device` in a killable worker, as the kind `P` the request builds.
fn compile<P: PipelineArtifact>(
    worker: &CompileWorker,
    device: &WorkerDevice,
    libraries: &[RowSource<'_>],
    request: &PipelineRequest,
) -> Result<WorkerCompile<P>, WorkerFailure> {
    let mut command = worker.command();
    command.arg(WORKER_ARG);
    compile_in(command, device, libraries, request)
}

/// [`compile`] in the worker `command` starts.
fn compile_in<P: PipelineArtifact>(
    mut command: Command,
    device: &WorkerDevice,
    libraries: &[RowSource<'_>],
    request: &PipelineRequest,
) -> Result<WorkerCompile<P>, WorkerFailure> {
    let scratch = ScratchDir::new("engine-compile-artifact").map_err(WorkerFailure::Setup)?;
    let artifact_path = scratch.path().join("artifact.bin");
    command.env(ARTIFACT_FILE, &artifact_path);
    let mut input_sha256 = None;
    let run = run_json::<CompileOutput>(command, "engine-compile", |mut file| {
        let input = serde_json::to_vec(&compile_input(device, libraries, request))
            .map_err(|error| format!("encode the compile worker's input: {error}"))?;
        input_sha256 = Some(sha256_bytes(&input));
        file.write_all(&input)
            .map_err(|error| format!("write the compile worker's input: {error}"))
    })?;
    let decode_started = Instant::now();
    let reported = run.measured();
    let measured = |run_elapsed| Measured {
        elapsed: run_elapsed + decode_started.elapsed(),
        ..reported
    };
    let output = run.result;
    // The artifact, when the worker reports one; the scratch directory goes once it is read.
    let artifact = match &output.result {
        CompileResult::Compiled { artifact_bytes, .. } => {
            read_artifact(&artifact_path, *artifact_bytes)
                .map_err(|error| WorkerFailure::Output(error, measured(run.elapsed)))?
        }
        _ => Vec::new(),
    };
    drop(scratch);
    let accepted = accept(
        &output,
        &artifact,
        input_sha256.as_deref().unwrap_or_default(),
        device.device(),
    );
    // The pipeline holds what it decoded (its blob included), so the bytes go now.
    drop(artifact);
    let result = accepted
        .and_then(|result| match result {
            Ok(compiled) => request
                .check_handles(&compiled.handles)
                .map(|()| Ok(compiled))
                .map_err(|error| format!("the compile worker's result: {error}")),
            Err(refused) => Ok(Err(refused)),
        })
        .map_err(|error| WorkerFailure::Output(error, measured(run.elapsed)))?;
    let measured = measured(run.elapsed);
    // The deadline covers decoding the result (worker recipe step 3): late is a timeout.
    if measured.elapsed > DEADLINE {
        return Err(WorkerFailure::Process(
            ProcessFailure::DeadlineExceeded,
            None,
            measured,
        ));
    }
    Ok(WorkerCompile {
        result,
        allocator_peak: output.allocator_peak,
        measured,
    })
}

/// The `reported` bytes of the artifact file at `path`, or why they cannot be used: a file of
/// another length is refused before any of it is read.
fn read_artifact(path: &std::path::Path, reported: u64) -> Result<Vec<u8>, String> {
    use std::io::Read as _;
    let file = std::fs::File::open(path)
        .map_err(|error| format!("open the compile worker's artifact: {error}"))?;
    let len = file
        .metadata()
        .map_err(|error| format!("read the compile worker's artifact: {error}"))?
        .len();
    if len != reported {
        return Err(format!(
            "the compile worker reported a {reported}-byte artifact and wrote {len} bytes"
        ));
    }
    let mut bytes = Vec::with_capacity(usize::try_from(len).unwrap_or(0));
    file.take(len)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("read the compile worker's artifact: {error}"))?;
    if u64::try_from(bytes.len()).ok() != Some(len) {
        return Err(format!(
            "the compile worker's artifact ended at {} of its {len} bytes",
            bytes.len()
        ));
    }
    Ok(bytes)
}

/// The worker's input for `request` over `libraries` on `device`.
fn compile_input<'a>(
    device: &WorkerDevice,
    libraries: &[RowSource<'a>],
    request: &PipelineRequest,
) -> CompileInput<'a> {
    CompileInput {
        description: device.description.clone(),
        policy: device.policy,
        request: request.clone(),
        libraries: libraries.to_vec(),
    }
}

/// The pipeline of `output` (with its handles) if this product compiled it, it answers the input
/// of digest `input_sha256`, and its `artifact` (the bytes the worker wrote) is as long as the
/// worker reported and loads for `device` as the kind `P`; an `Err` names why the result cannot
/// be used.
fn accept<P: PipelineArtifact>(
    output: &CompileOutput,
    artifact: &[u8],
    input_sha256: &str,
    device: &Device,
) -> Result<Result<Compiled<P>, NotCompiled>, String> {
    if output.product_fingerprint != TRANSLATOR_FINGERPRINT {
        return Err(format!(
            "the compile worker runs product {} and this process product \
             {TRANSLATOR_FINGERPRINT}: the executable was rebuilt since this process started",
            output.product_fingerprint
        ));
    }
    if output.input_sha256 != input_sha256 {
        return Err(format!(
            "the compile worker answered input {} for input {input_sha256}",
            output.input_sha256
        ));
    }
    match &output.result {
        CompileResult::NotCompiled { outcome, detail } => {
            Ok(Err(NotCompiled::Product(outcome.clone(), detail.clone())))
        }
        CompileResult::Unmodelled { reason } => Ok(Err(NotCompiled::Unmodelled(reason.clone()))),
        CompileResult::Compiled {
            artifact_bytes,
            handles,
        } => {
            if u64::try_from(artifact.len()).ok() != Some(*artifact_bytes) {
                return Err(format!(
                    "the compile worker reported a {artifact_bytes}-byte artifact and wrote {} bytes",
                    artifact.len()
                ));
            }
            P::from_bytes(artifact, device)
                .map(|pipeline| {
                    Ok(Compiled {
                        pipeline,
                        handles: handles.clone(),
                    })
                })
                .map_err(artifact_refused)
        }
    }
}

fn artifact_refused(error: ArtifactError) -> String {
    let why = match error {
        ArtifactError::WrongDevice => "was compiled for another device",
        ArtifactError::Stale => "has another format or semantics version",
        ArtifactError::Corrupt => "fails its checksum or sanity scan, or is another kind",
        _ => "was refused",
    };
    format!("the compile worker's pipeline artifact {why} ({error})")
}

/// Serve one compile when this process was started with [`WORKER_ARG`], then exit (0 when a
/// result was written, 1 otherwise); return at once otherwise. Call it first in `main`.
pub fn serve_if_requested() {
    if std::env::args_os().nth(1).as_deref() != Some(std::ffi::OsStr::new(WORKER_ARG)) {
        return;
    }
    std::process::exit(super::serve_process(
        "engine compile worker",
        alloc::installed(),
        serve_to_artifact_file,
    ))
}

/// [`serve`], writing the artifact to the file [`ARTIFACT_FILE`] names.
fn serve_to_artifact_file(input: &[u8], started: Instant) -> Result<CompileOutput, String> {
    let path = std::env::var_os(ARTIFACT_FILE)
        .ok_or_else(|| format!("{ARTIFACT_FILE} names no artifact file"))?;
    let mut file = std::fs::File::create(&path)
        .map_err(|error| format!("create the compile worker's artifact: {error}"))?;
    serve(input, started, &mut file)
}

/// A writer that counts the bytes it passes on.
struct Counted<'a> {
    out: &'a mut dyn Write,
    bytes: u64,
}

impl Write for Counted<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let n = self.out.write(bytes)?;
        self.bytes = self.bytes.saturating_add(n as u64);
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.out.flush()
    }
}

/// The worker's compile of `input` (the bytes of a [`CompileInput`]), under the deadline from
/// `started`, writing a pipeline's artifact to `artifact`. A typed product error is a result,
/// never an `Err`.
fn serve(
    input: &[u8],
    started: Instant,
    artifact: &mut dyn Write,
) -> Result<CompileOutput, String> {
    let input_sha256 = sha256_bytes(input);
    let decoded: CompileInput<'_> = serde_json::from_slice(input)
        .map_err(|error| format!("decode the compile worker's input: {error}"))?;
    let not_compiled = |error: &Error| CompileResult::NotCompiled {
        outcome: Outcome::from(error),
        detail: bounded_detail(&error.to_string()),
    };
    let output = |result, allocator_peak| CompileOutput {
        product_fingerprint: TRANSLATOR_FINGERPRINT.into(),
        input_sha256: input_sha256.clone(),
        result,
        allocator_peak,
    };
    // Each library's bitcode: a retained blob decoded, AIR text assembled here, under the
    // worker's deadline and limit. A source that gives none is the row's outcome.
    let mut libraries = Vec::with_capacity(decoded.libraries.len());
    for source in &decoded.libraries {
        match source.bitcode() {
            Ok(bitcode) => libraries.push(bitcode),
            Err(error) => {
                let attempt = error.attempt();
                return Ok(output(
                    CompileResult::NotCompiled {
                        outcome: attempt.outcome,
                        detail: attempt.detail.unwrap_or_default(),
                    },
                    None,
                ));
            }
        }
    }
    let device = match Device::from_queries(decoded.description, decoded.policy) {
        Ok(device) => device,
        Err(error) => return Ok(output(not_compiled(&error), None)),
    };
    let (compiled, allocator_peak) = alloc::measure(|| {
        compile_pipeline(
            &libraries.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &decoded.request,
            &device,
            started.checked_add(DEADLINE),
        )
    });
    let result = match compiled {
        Ok(CompiledRequest { pipeline, handles }) => {
            let mut out = Counted {
                out: artifact,
                bytes: 0,
            };
            match &pipeline {
                AcceptancePipeline::Compute(pipeline) => pipeline.write_to(&mut out),
                AcceptancePipeline::Render(pipeline) => pipeline.write_to(&mut out),
            }
            .and_then(|()| out.flush())
            .map_err(|error| format!("write the compile worker's artifact: {error}"))?;
            CompileResult::Compiled {
                artifact_bytes: out.bytes,
                handles,
            }
        }
        Err(PipelineCompileError::Product(error)) => not_compiled(&error),
        Err(PipelineCompileError::Unmodelled(reason)) => CompileResult::Unmodelled {
            reason: crate::acceptance::row::Unmodelled {
                kind: reason.kind,
                detail: bounded_detail(&reason.detail),
            },
        },
        // The host's tooling, for this row: the row's outcome, which a sweep counts as its error.
        Err(PipelineCompileError::Tooling(detail)) => CompileResult::NotCompiled {
            outcome: Outcome::Failed {
                failure: crate::engine::FailureKind::Tooling,
            },
            detail: bounded_detail(&detail),
        },
        // Another number of libraries than the request takes is the exchange's error, refused
        // by `compile_pipeline` before any library is decoded as AIR, never a compile outcome.
        Err(error @ PipelineCompileError::Libraries { .. }) => {
            return Err(format!("the compile worker's input: {error}"))
        }
    };
    Ok(output(result, allocator_peak))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acceptance::RequestFunction;
    use crate::worker::MEMORY_LIMIT_BYTES;
    use metal2vulkan::bind::{BindShape, BufferRanges};
    use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};

    fn fixture(name: &str) -> Option<Vec<u8>> {
        match llvm_assemble(&engine_dir().join(name)) {
            Ok(bytes) => Some(bytes),
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                None
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
        }
    }

    fn worker_device(stem: &str) -> WorkerDevice {
        WorkerDevice::new(
            crate::engine::load_description(stem).unwrap(),
            DevicePolicy::default(),
        )
        .unwrap()
    }

    /// The worker's compile of `input` from `started`, with the artifact it wrote.
    fn served(input: &[u8], started: Instant) -> Result<(CompileOutput, Vec<u8>), String> {
        let mut artifact = Vec::new();
        serve(input, started, &mut artifact).map(|output| (output, artifact))
    }

    /// The input the parent writes for `request` over one library, and its digest.
    fn input(device: &WorkerDevice, bitcode: &[u8], request: PipelineRequest) -> (Vec<u8>, String) {
        libraries_input(device, &[bitcode], request)
    }

    /// The input the parent writes for `request` over `libraries`, and its digest.
    fn libraries_input(
        device: &WorkerDevice,
        libraries: &[&[u8]],
        request: PipelineRequest,
    ) -> (Vec<u8>, String) {
        let sources: Vec<RowSource<'_>> = libraries
            .iter()
            .map(|bitcode| crate::acceptance::retained(bitcode))
            .collect();
        sources_input(device, &sources, request)
    }

    /// The input the parent writes for `request` over library `sources`, and its digest.
    fn sources_input(
        device: &WorkerDevice,
        sources: &[RowSource<'_>],
        request: PipelineRequest,
    ) -> (Vec<u8>, String) {
        let input = serde_json::to_vec(&compile_input(device, sources, &request)).unwrap();
        let digest = sha256_bytes(&input);
        (input, digest)
    }

    /// The child's compile, its JSON exchange, and the parent's acceptance together give the
    /// pipeline the in-process public API compiles, for a kernel and a vertex function.
    #[test]
    fn the_worker_returns_the_pipeline_the_public_api_compiles() {
        let (Some(bitcode), Some(stage_io)) = (fixture("store_const.ll"), fixture("stage_io.ll"))
        else {
            return;
        };
        let device = worker_device("moltenvk-m3");
        let request = PipelineRequest::Compute {
            entry: "store_const".into(),
            constants: FunctionConstantValues::defaults(),
        };
        let (bytes, digest) = input(&device, &bitcode, request.clone());
        let (output, artifact) = served(&bytes, Instant::now()).unwrap();
        // Through JSON, as the parent reads the worker's stdout.
        let output: CompileOutput =
            serde_json::from_slice(&serde_json::to_vec(&output).unwrap()).unwrap();
        let pipeline = accept::<ComputePipeline>(&output, &artifact, &digest, device.device())
            .unwrap()
            .unwrap()
            .pipeline;
        let expected = compile_pipeline(&[&bitcode], &request, device.device(), None)
            .unwrap()
            .pipeline;
        let AcceptancePipeline::Compute(expected) = expected else {
            panic!("a kernel compiles to a compute pipeline");
        };
        assert_eq!(pipeline, *expected);
        assert_eq!(pipeline.profile, device.device().profile().fingerprint());
        // The lib test binary counts allocations, so the compile's peak is measured.
        assert!(output.allocator_peak.is_some_and(|peak| peak > 0));

        // A kernel decoded as a render pipeline is another kind: refused, never used.
        let refused =
            accept::<RenderPipeline>(&output, &artifact, &digest, device.device()).unwrap_err();
        assert!(refused.contains("artifact"), "{refused}");

        // A vertex function's L4 row (one without vertex attributes, so no descriptor).
        let request = PipelineRequest::Row {
            entry: "stage_io_vertex_position".into(),
            stage: crate::acceptance::RowStage::Vertex,
            projection: crate::acceptance::Projection::Derived,
            constants: Default::default(),
        };
        let (bytes, digest) = input(&device, &stage_io, request.clone());
        let (output, artifact) = served(&bytes, Instant::now()).unwrap();
        let output: CompileOutput =
            serde_json::from_slice(&serde_json::to_vec(&output).unwrap()).unwrap();
        let pipeline = accept::<RenderPipeline>(&output, &artifact, &digest, device.device())
            .unwrap()
            .unwrap()
            .pipeline;
        let AcceptancePipeline::Render(expected) =
            compile_pipeline(&[&stage_io], &request, device.device(), None)
                .unwrap()
                .pipeline
        else {
            panic!("a vertex function compiles to a vertex-only render pipeline");
        };
        assert_eq!(pipeline, *expected);
        assert_eq!(pipeline.stages.len(), 1);
        let refused =
            accept::<ComputePipeline>(&output, &artifact, &digest, device.device()).unwrap_err();
        assert!(refused.contains("artifact"), "{refused}");
    }

    /// A kernel with a storage texture crosses the exchange with the bind shape it is compiled
    /// for: the worker answers the texture's format input from the shape, and its pipeline is the
    /// one compiled in process, whose plan of that shape asks for exactly the inputs it was
    /// compiled with.
    #[test]
    fn a_bound_compute_request_crosses_the_exchange_with_its_shape() {
        use crate::candidate::engine::{compute_shape, variants, ComputeCase, ComputeOutput};
        let Some(bitcode) = fixture("texture_ops.ll") else {
            return;
        };
        // SwiftShader reads storage images only with a format, so the texture needs its input.
        let device = worker_device("runtime-floor");
        let case = ComputeCase {
            out_of_bounds: Vec::new(),
            creation_bound: crate::budgets::PIPELINE_CREATION_CEILING,
            entry: "read_write_fence".into(),
            constants: FunctionConstantValues::defaults(),
            buffers: Default::default(),
            rate_maps: Default::default(),
            argument_buffer_buffers: Default::default(),
            argument_buffer_rate_maps: Default::default(),
            argument_buffer_textures: Default::default(),
            argument_buffer_samplers: Default::default(),
            objects: Default::default(),
            command_buffers: Default::default(),
            textures: [(
                0,
                crate::literal::LiteralTexture {
                    binding: 0,
                    role: crate::case::ResourceRole::InOut,
                    texture_type: crate::case::TextureType::D2,
                    format: crate::case::TextureFormat::Rgba32Uint,
                    dimensions: [2, 2, 1],
                    sample_count: 1,
                    bytes: vec![0; 64],
                    sparse: None,
                    mip_levels: 1,
                },
            )]
            .into(),
            samplers: Default::default(),
            function_tables: Default::default(),
            acceleration_structures: Default::default(),
            threadgroup_lengths: Default::default(),
            imageblock: None,
            tile: None,
            stage_input: None,
            stage_in_region: None,
            attribute_strides: Default::default(),
            dispatch: metal2vulkan::bind::Dispatch::Threads {
                threads: [2, 2, 1],
                threads_per_group: [1, 1, 1],
            },
            output: ComputeOutput::None,
        };
        let shape = compute_shape(&case, false, BufferRanges::default()).unwrap();
        let request = PipelineRequest::BoundCompute {
            function: RequestFunction::new(case.entry.as_str(), case.constants.clone()),
            shape: shape.clone(),
            dispatch: None,
            stage_input: None,
            tile: None,
            replay: false,
        };
        let (bytes, digest) = input(&device, &bitcode, request.clone());
        let (output, artifact) = served(&bytes, Instant::now()).unwrap();
        let output: CompileOutput =
            serde_json::from_slice(&serde_json::to_vec(&output).unwrap()).unwrap();
        let pipeline = accept::<ComputePipeline>(&output, &artifact, &digest, device.device())
            .unwrap()
            .unwrap()
            .pipeline;
        let AcceptancePipeline::Compute(expected) =
            compile_pipeline(&[&bitcode], &request, device.device(), None)
                .unwrap()
                .pipeline
        else {
            panic!("a kernel compiles to a compute pipeline");
        };
        assert_eq!(pipeline, *expected);
        let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
        let function =
            metal2vulkan::pipeline::FunctionDescriptor::new(library.function(&case.entry).unwrap());
        let (_, inputs) = variants::compile_compute_for_shape(
            device.device(),
            &function,
            &shape,
            None,
            None,
            &metal2vulkan::Budget::standard(),
        )
        .unwrap();
        assert!(!inputs.storage_images.is_empty(), "{inputs:?}");
        let plan = pipeline.plan(device.device(), &shape).unwrap();
        assert!(variants::compiled_for(&inputs, &plan.variant));
    }

    /// The narrow-attribute vertex case's draw crosses the exchange with the Metal oracle's
    /// descriptor template (its vertex descriptor and attachment): the worker's pipeline is the one
    /// compiled in process from that template.
    #[test]
    fn a_vertex_draw_request_crosses_the_exchange_with_its_template() {
        use crate::candidate::engine::graphics::{render_case, render_template};
        let path = metal2vulkan_devtools::fixtures_dir().join("vertex_narrow_attributes.ll");
        let bitcode = match llvm_assemble(&path) {
            Ok(bytes) => bytes,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {path:?}: {why}"),
        };
        let authored =
            crate::case::narrow_vertex_test_case(String::new(), "vertex_narrow_attributes".into());
        let case = render_case(
            &authored,
            &crate::check::interface::tests::vertex_buffers_at(&[0]),
            crate::candidate::engine::graphics::tests::WITHOUT_SLICE,
            None,
            None,
        )
        .unwrap();
        let template = render_template(&case);
        assert!(template.vertex_descriptor.is_some());
        let device = worker_device("lavapipe");
        let request = PipelineRequest::VertexDraw {
            function: RequestFunction::new(
                case.vertex.entry.as_str(),
                case.vertex.constants.clone(),
            ),
            template,
            shape: Some(
                crate::candidate::engine::graphics::render_shape(
                    &case,
                    false,
                    BufferRanges::default(),
                )
                .unwrap(),
            ),
            form: crate::candidate::engine::variants::DrawForm::Bound,
        };
        let (bytes, digest) = input(&device, &bitcode, request.clone());
        let (output, artifact) = served(&bytes, Instant::now()).unwrap();
        let output: CompileOutput =
            serde_json::from_slice(&serde_json::to_vec(&output).unwrap()).unwrap();
        let pipeline = accept::<RenderPipeline>(&output, &artifact, &digest, device.device())
            .unwrap()
            .unwrap()
            .pipeline;
        let AcceptancePipeline::Render(expected) =
            compile_pipeline(&[&bitcode], &request, device.device(), None)
                .unwrap()
                .pipeline
        else {
            panic!("a vertex draw compiles to a render pipeline");
        };
        assert_eq!(pipeline, *expected);
        assert!(pipeline.graphics.rasterizer_discard);
        assert!(!pipeline.vertex_input.bindings.is_empty());
    }

    /// A fragment case's draw crosses the exchange with both of its libraries: the generated
    /// passthrough vertex companion's and the case's, in stage order, compiled in one child into
    /// the rasterizing pipeline the in-process public API compiles. A library count the request
    /// does not take is the exchange's error, not a compile.
    #[test]
    fn a_draw_request_crosses_the_exchange_with_its_companion_library() {
        use crate::candidate::engine::graphics::RenderTemplate;
        use crate::case::{Stage, TextureFormat};
        use crate::companion::tests::{case, derived};
        let path = metal2vulkan_devtools::fixtures_dir().join("fragment_varying_color.ll");
        let fragment = match llvm_assemble(&path) {
            Ok(bytes) => bytes,
            Err(AssembleError::Missing(why)) => {
                eprintln!("SKIPPED: {why}");
                return;
            }
            Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {path:?}: {why}"),
        };
        let authored = case(
            Stage::Fragment,
            "fragment_varying_color",
            TextureFormat::Rgba32Float,
        );
        let companion = derived(&authored, &fragment).unwrap().unwrap();
        let vertex = companion.opaque_bitcode().unwrap();
        let mut template =
            RenderTemplate::new(metal2vulkan::metal::PrimitiveTopologyClass::Triangle);
        template.color_attachments = [(
            0,
            metal2vulkan::pipeline::ColorAttachmentDescriptor::new(
                metal2vulkan::metal::PixelFormat::RGBA32Float,
            ),
        )]
        .into();
        let request = PipelineRequest::Draw {
            vertex: RequestFunction::new(companion.entry(), FunctionConstantValues::defaults()),
            fragment: RequestFunction::new(
                authored.entry.as_str(),
                FunctionConstantValues::defaults(),
            ),
            template,
            shape: None,
        };
        let device = worker_device("lavapipe");
        let libraries = [vertex.as_slice(), fragment.as_slice()];
        let (bytes, digest) = libraries_input(&device, &libraries, request.clone());
        let (output, artifact) = served(&bytes, Instant::now()).unwrap();
        let output: CompileOutput =
            serde_json::from_slice(&serde_json::to_vec(&output).unwrap()).unwrap();
        let pipeline = accept::<RenderPipeline>(&output, &artifact, &digest, device.device())
            .unwrap()
            .unwrap()
            .pipeline;
        let AcceptancePipeline::Render(expected) =
            compile_pipeline(&libraries, &request, device.device(), None)
                .unwrap()
                .pipeline
        else {
            panic!("a draw compiles to a render pipeline");
        };
        assert_eq!(pipeline, *expected);
        assert_eq!(pipeline.stages.len(), 2);
        assert!(!pipeline.graphics.rasterizer_discard);

        // One library for a request that takes two is refused before any compile.
        let (bytes, _) = libraries_input(&device, &libraries[..1], request);
        let refused = served(&bytes, Instant::now()).unwrap_err();
        assert!(refused.contains("takes 2 libraries"), "{refused}");
    }

    /// A kernel's linked libraries cross the exchange after its own: the worker links the functions
    /// the request names from them into the pipeline the in-process public API compiles, and
    /// returns the product's table handles of those functions with it; an input without one of them
    /// is the exchange's error, not a compile.
    #[test]
    fn a_linking_request_crosses_the_exchange_with_its_linked_libraries() {
        use crate::acceptance::LinkedLibrary;
        let (Some(caller), Some(functions)) = (
            fixture("linked_visible_caller.ll"),
            fixture("linked_visible_functions.ll"),
        ) else {
            return;
        };
        let device = worker_device("lavapipe");
        let request = PipelineRequest::BoundCompute {
            function: RequestFunction {
                entry: "linked_visible_caller".into(),
                constants: FunctionConstantValues::defaults(),
                linked: vec![
                    LinkedLibrary {
                        functions: vec!["linked_scale".into(), "linked_blend".into()],
                    },
                    LinkedLibrary {
                        functions: vec!["linked_offset".into()],
                    },
                ],
            },
            shape: BindShape::default(),
            dispatch: None,
            stage_input: None,
            tile: None,
            replay: false,
        };
        let libraries = [caller.as_slice(), functions.as_slice(), caller.as_slice()];
        let (bytes, digest) = libraries_input(&device, &libraries, request.clone());
        let (output, artifact) = served(&bytes, Instant::now()).unwrap();
        let output: CompileOutput =
            serde_json::from_slice(&serde_json::to_vec(&output).unwrap()).unwrap();
        let compiled = accept::<ComputePipeline>(&output, &artifact, &digest, device.device())
            .unwrap()
            .unwrap();
        let expected = compile_pipeline(&libraries, &request, device.device(), None).unwrap();
        let AcceptancePipeline::Compute(pipeline) = &expected.pipeline else {
            panic!("a kernel compiles to a compute pipeline");
        };
        assert_eq!(&compiled.pipeline, &**pipeline);
        assert_eq!(compiled.handles, expected.handles);
        request.check_handles(&compiled.handles).unwrap();

        let (bytes, _) = libraries_input(&device, &libraries[..2], request);
        let refused = served(&bytes, Instant::now()).unwrap_err();
        assert!(refused.contains("takes 3 libraries"), "{refused}");
    }

    #[test]
    fn a_product_refusal_is_a_typed_outcome_not_a_worker_failure() {
        let Some(bitcode) = fixture("store_const.ll") else {
            return;
        };
        let device = worker_device("runtime-floor");
        for (source, entry, label) in [
            (bitcode.as_slice(), "no_such_entry", "invalid_descriptor"),
            (b"not bitcode".as_slice(), "main", "invalid_input"),
        ] {
            let (bytes, digest) = input(
                &device,
                source,
                PipelineRequest::Compute {
                    entry: entry.into(),
                    constants: FunctionConstantValues::defaults(),
                },
            );
            let (output, artifact) = served(&bytes, Instant::now()).unwrap();
            match accept::<ComputePipeline>(&output, &artifact, &digest, device.device()).unwrap() {
                Err(NotCompiled::Product(outcome, detail)) => {
                    assert_eq!(outcome.label(), label, "{detail}")
                }
                Err(other) => panic!("{entry}: {other:?}"),
                Ok(_) => panic!("{entry} compiled"),
            }
        }
    }

    /// The parent uses an artifact only when it answers its own input and was compiled for its
    /// own device.
    #[test]
    fn the_parent_refuses_another_inputs_or_another_devices_artifact() {
        let Some(bitcode) = fixture("store_const.ll") else {
            return;
        };
        let device = worker_device("moltenvk-m3");
        let (bytes, digest) = input(
            &device,
            &bitcode,
            PipelineRequest::Compute {
                entry: "store_const".into(),
                constants: FunctionConstantValues::defaults(),
            },
        );
        let (output, artifact) = served(&bytes, Instant::now()).unwrap();
        assert_eq!(output.product_fingerprint, TRANSLATOR_FINGERPRINT);
        let other_input =
            accept::<ComputePipeline>(&output, &artifact, &"0".repeat(64), device.device())
                .unwrap_err();
        assert!(other_input.contains("answered input"), "{other_input}");
        let other_device = worker_device("lavapipe");
        let wrong = accept::<ComputePipeline>(&output, &artifact, &digest, other_device.device())
            .unwrap_err();
        assert!(wrong.contains("WrongDevice"), "{wrong}");
        // A file shorter than the worker reported, and one with a byte changed.
        let short = accept::<ComputePipeline>(
            &output,
            &artifact[..artifact.len() - 1],
            &digest,
            device.device(),
        )
        .unwrap_err();
        assert!(short.contains("wrote"), "{short}");
        let mut corrupt = artifact.clone();
        corrupt[artifact.len() / 2] ^= 1;
        let refused =
            accept::<ComputePipeline>(&output, &corrupt, &digest, device.device()).unwrap_err();
        assert!(refused.contains("Corrupt"), "{refused}");
    }

    /// The parent reads the artifact file only when it holds the length the worker reported.
    #[test]
    fn the_parent_reads_an_artifact_of_the_reported_length_only() {
        let scratch = ScratchDir::new("artifact-length-test").unwrap();
        let path = scratch.path().join("artifact.bin");
        std::fs::write(&path, b"0123456789").unwrap();
        assert_eq!(read_artifact(&path, 10).unwrap(), b"0123456789");
        for reported in [9, 11, 1 << 40] {
            let refused = read_artifact(&path, reported).unwrap_err();
            assert!(refused.contains("wrote 10 bytes"), "{reported}: {refused}");
        }
        let missing = read_artifact(&scratch.path().join("none.bin"), 0).unwrap_err();
        assert!(missing.contains("open"), "{missing}");
    }

    /// A worker whose executable was rebuilt after the parent started is another compiler: its
    /// pipeline and its refusal are both refused, so neither is attributed to this product.
    #[test]
    fn the_parent_refuses_another_products_result() {
        let Some(bitcode) = fixture("store_const.ll") else {
            return;
        };
        let device = worker_device("moltenvk-m3");
        for entry in ["store_const", "no_such_entry"] {
            let (bytes, digest) = input(
                &device,
                &bitcode,
                PipelineRequest::Compute {
                    entry: entry.into(),
                    constants: FunctionConstantValues::defaults(),
                },
            );
            let (mut output, artifact) = served(&bytes, Instant::now()).unwrap();
            assert!(
                accept::<ComputePipeline>(&output, &artifact, &digest, device.device()).is_ok()
            );
            output.product_fingerprint = "0".repeat(64);
            let refused = accept::<ComputePipeline>(&output, &artifact, &digest, device.device())
                .unwrap_err();
            assert!(refused.contains("was rebuilt"), "{entry}: {refused}");
        }
    }

    /// A library the parent sends as AIR text is assembled in the worker, under its deadline: the
    /// pipeline equals the one compiled from the assembled bitcode, a text `llvm-as` rejects is the
    /// row's `failed no_bitcode` outcome, and a row whose inputs the tool does not build crosses as
    /// `Unmodelled`, never as a product outcome.
    #[test]
    fn a_text_library_is_assembled_in_the_worker() {
        let Some(bitcode) = fixture("store_const.ll") else {
            return;
        };
        let text = std::fs::read_to_string(engine_dir().join("store_const.ll")).unwrap();
        let device = worker_device("moltenvk-m3");
        let request = PipelineRequest::Compute {
            entry: "store_const".into(),
            constants: FunctionConstantValues::defaults(),
        };
        let (bytes, digest) = sources_input(
            &device,
            &[RowSource::Text(text.as_str().into())],
            request.clone(),
        );
        let (output, artifact) = served(&bytes, Instant::now()).unwrap();
        let pipeline = accept::<ComputePipeline>(&output, &artifact, &digest, device.device())
            .unwrap()
            .unwrap()
            .pipeline;
        let AcceptancePipeline::Compute(expected) =
            compile_pipeline(&[&bitcode], &request, device.device(), None)
                .unwrap()
                .pipeline
        else {
            panic!("a kernel compiles to a compute pipeline");
        };
        assert_eq!(pipeline, *expected);

        let (bytes, digest) = sources_input(
            &device,
            &[RowSource::Text("this is not LLVM assembly".into())],
            request,
        );
        let (output, artifact) = served(&bytes, Instant::now()).unwrap();
        match accept::<ComputePipeline>(&output, &artifact, &digest, device.device()).unwrap() {
            Err(NotCompiled::Product(outcome, detail)) => {
                assert_eq!(outcome.key(), "no_bitcode", "{detail}")
            }
            other => panic!("{other:?}"),
        }

        // A vertex attribute of a type no vertex format reads exactly (a 64-bit integer).
        let wide = text_module(
            "define <4 x float> @v(i64 %a) {\n  ret <4 x float> zeroinitializer\n}\n\n\
             !air.vertex = !{!0}\n!0 = !{ptr @v, !1, !2}\n!1 = !{!3}\n\
             !3 = !{!\"air.position\", !\"air.arg_type_name\", !\"float4\"}\n!2 = !{!4}\n\
             !4 = !{i32 0, !\"air.vertex_input\", !\"air.location_index\", i32 0, i32 1, \
             !\"air.arg_type_name\", !\"ulong\"}\n",
        );
        let (bytes, digest) = sources_input(
            &device,
            &[RowSource::Text(wide.as_str().into())],
            PipelineRequest::Row {
                entry: "v".into(),
                stage: crate::acceptance::RowStage::Vertex,
                projection: crate::acceptance::Projection::Derived,
                constants: Default::default(),
            },
        );
        let (output, artifact) = served(&bytes, Instant::now()).unwrap();
        match accept::<RenderPipeline>(&output, &artifact, &digest, device.device()).unwrap() {
            Err(NotCompiled::Unmodelled(why)) => {
                assert_eq!(
                    why.kind,
                    crate::acceptance::row::UnmodelledKind::VertexAttribute
                );
                assert!(why.detail.contains("vertex attribute 0"), "{why}")
            }
            other => panic!("{other:?}"),
        }
    }

    /// An AIR text module with this body.
    fn text_module(body: &str) -> String {
        format!(
            "target datalayout = \"e-p:64:64:64\"\ntarget triple = \"air64-apple-macosx14.0.0\"\n\n{body}"
        )
    }

    #[test]
    fn an_undecodable_input_is_a_worker_error() {
        assert!(served(b"{}", Instant::now())
            .unwrap_err()
            .contains("decode the compile worker's input"));
    }

    /// The child's compile runs under the deadline from when it started serving: a compile whose
    /// deadline has passed is the product's `timeout`, never a pipeline.
    #[test]
    fn the_worker_compiles_under_the_deadline_from_its_start() {
        let Some(bitcode) = fixture("store_const.ll") else {
            return;
        };
        let device = worker_device("moltenvk-m3");
        let request = PipelineRequest::Compute {
            entry: "store_const".into(),
            constants: FunctionConstantValues::defaults(),
        };
        let (bytes, digest) = input(&device, &bitcode, request);
        let expired = Instant::now()
            .checked_sub(DEADLINE + std::time::Duration::from_secs(1))
            .expect("the host's monotonic clock reaches back past the deadline");
        let (output, artifact) = served(&bytes, expired).unwrap();
        match accept::<ComputePipeline>(&output, &artifact, &digest, device.device()).unwrap() {
            Err(NotCompiled::Product(outcome, detail)) => {
                assert_eq!(outcome.key(), "timeout", "{detail}")
            }
            other => panic!("a compile past its deadline: {other:?}"),
        }
    }

    /// The compile path's limit probes: test-only requests a compile worker child serves in place
    /// of a compile, each breaching one limit of the boundary. The child is the worker the lib's
    /// device tests compile through (`CompileWorker::lib_tests`), this test binary running
    /// [`compile_worker_fixture`], which counts its allocations (`alloc::tests` installs the
    /// counting allocator), through the worker's own process (`worker::serve_process`); the
    /// parent is [`compile`], as every tool's compile.
    #[derive(Clone, Copy, Debug)]
    enum Probe {
        /// Never answer: the deadline stops the worker.
        Stall,
        /// Hold resident memory the counting allocator never sees (as a driver's or a tool's
        /// would be) past the limit: the process-group RSS watchdog stops the worker.
        Resident,
        /// Allocate close to the 500-MiB budget, then past it: the allocator admits the first and
        /// refuses the second, which aborts the worker.
        Budget,
        /// Serve as a binary without the counting allocator would: the worker refuses.
        Uncounted,
    }

    impl Probe {
        const ALL: [Probe; 4] = [
            Probe::Stall,
            Probe::Resident,
            Probe::Budget,
            Probe::Uncounted,
        ];

        fn name(self) -> String {
            format!("{self:?}")
        }

        /// The probe's request, in place of a compile of `input` from `started`.
        fn serve(self, input: &[u8], started: Instant) -> Result<CompileOutput, String> {
            const MIB: usize = 1024 * 1024;
            let limit = MEMORY_LIMIT_BYTES as usize;
            match self {
                Probe::Stall => std::thread::sleep(DEADLINE * 2),
                Probe::Resident => {
                    let length = limit + 100 * MIB;
                    let layout = std::alloc::Layout::from_size_align(length, 4096).unwrap();
                    // SAFETY: a non-zero size with a valid alignment; `System` is not the global
                    // allocator, so the counting allocator neither sees nor refuses it.
                    let held =
                        unsafe { std::alloc::GlobalAlloc::alloc(&std::alloc::System, layout) };
                    assert!(!held.is_null(), "the system refused {length} bytes");
                    for page in (0..length).step_by(4096) {
                        // SAFETY: `page` is inside the allocation just made.
                        unsafe { held.add(page).write_volatile(1) };
                    }
                    std::thread::sleep(DEADLINE * 2);
                }
                Probe::Budget => {
                    // Zeroed and never touched, so the pages stay out of the resident set: only
                    // the allocator's budget can refuse these.
                    let admitted = std::hint::black_box(vec![0u8; limit - 64 * MIB]);
                    eprintln!("admitted {} bytes", admitted.len());
                    let refused = std::hint::black_box(vec![0u8; 128 * MIB]);
                    eprintln!("the budget admitted {} more bytes", refused.len());
                }
                Probe::Uncounted => {}
            }
            serve_to_artifact_file(input, started)
        }
    }

    /// The lib test binary's compile worker child ([`CompileWorker::lib_tests`]) when
    /// [`LIB_TESTS_ROLE`] is set: a compile ([`LIB_TESTS_SERVE`]), or the [`Probe`] it names,
    /// served as a binary serves one ([`serve_if_requested`]) on a stack of a binary's main
    /// thread ([`LIB_TESTS_STACK`]); otherwise nothing. It serves the compile worker alone: the
    /// worker argument its parent passed is its last.
    #[test]
    fn compile_worker_fixture() {
        let Ok(role) = std::env::var(LIB_TESTS_ROLE) else {
            return;
        };
        assert_eq!(
            std::env::args().next_back().as_deref(),
            Some(WORKER_ARG),
            "the lib test binary serves only the compile worker"
        );
        let probe = Probe::ALL.into_iter().find(|probe| probe.name() == role);
        assert!(
            probe.is_some() || role == LIB_TESTS_SERVE,
            "{LIB_TESTS_ROLE} names no role: {role}"
        );
        let counting = alloc::installed() && !matches!(probe, Some(Probe::Uncounted));
        let served = std::thread::Builder::new()
            .stack_size(LIB_TESTS_STACK)
            .spawn(move || {
                crate::worker::serve_process("engine compile worker", counting, |input, started| {
                    match probe {
                        Some(probe) => probe.serve(input, started),
                        None => serve_to_artifact_file(input, started),
                    }
                })
            })
            .unwrap()
            .join()
            .unwrap();
        std::process::exit(served);
    }

    /// Compile through a worker child that serves `probe`, as a tool compiles a kernel: the lib
    /// tests' worker ([`CompileWorker::lib_tests`]) serving it in place of a compile.
    fn compile_probe(probe: Probe) -> Result<WorkerCompile<ComputePipeline>, WorkerFailure> {
        let request = PipelineRequest::Compute {
            entry: "main".into(),
            constants: FunctionConstantValues::defaults(),
        };
        compile(
            &CompileWorker::lib_tests_as(&probe.name()).unwrap(),
            &worker_device("runtime-floor"),
            &[crate::acceptance::retained(b"bitcode")],
            &request,
        )
    }

    /// The lib tests' worker ([`CompileWorker::lib_tests`]), the one their device tests compile
    /// through, returns across the process boundary the pipeline the public API compiles in
    /// process, with its counting allocator's peaks, as every binary that serves the worker does
    /// (`tests/integration/compile_worker.rs`, through `corpus-pipeline`).
    #[test]
    fn the_lib_tests_worker_returns_the_pipeline_the_public_api_compiles() {
        let Some(bitcode) = fixture("store_const.ll") else {
            return;
        };
        let device = worker_device("moltenvk-m3");
        let request = PipelineRequest::Compute {
            entry: "store_const".into(),
            constants: FunctionConstantValues::defaults(),
        };
        let AcceptancePipeline::Compute(expected) =
            compile_pipeline(&[&bitcode], &request, device.device(), None)
                .unwrap()
                .pipeline
        else {
            panic!("a kernel compiles to a compute pipeline");
        };
        let compiled = compile_compute(
            &CompileWorker::lib_tests().unwrap(),
            &device,
            &crate::acceptance::retained(&bitcode),
            "store_const",
            &FunctionConstantValues::defaults(),
        )
        .unwrap_or_else(|failure| panic!("{}", failure.message()));
        let pipeline = compiled.result.unwrap();
        assert_eq!(pipeline.pipeline, *expected);
        assert!(pipeline.handles.is_empty());
        assert!(compiled.allocator_peak.is_some_and(|peak| peak > 0));
        assert!(compiled.measured.elapsed < DEADLINE);
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(compiled.measured.peak_resident_bytes.is_some());
        }
        let process_peak = compiled.measured.allocator_peak_bytes;
        assert!(process_peak.is_some_and(|peak| peak >= compiled.allocator_peak.unwrap()));
    }

    /// A compile worker that never answers is stopped at the 20-second deadline, measured, and
    /// reported as a `timeout`.
    #[cfg(unix)]
    #[test]
    fn the_compile_path_stops_a_worker_at_the_deadline() {
        match compile_probe(Probe::Stall) {
            Err(failure @ WorkerFailure::Process(ProcessFailure::DeadlineExceeded, ..)) => {
                assert_eq!(failure.outcome().key(), "timeout");
                let elapsed = failure.measured().unwrap().elapsed;
                assert!(elapsed >= DEADLINE && elapsed < DEADLINE * 2, "{elapsed:?}");
            }
            other => panic!("{other:?}"),
        }
    }

    /// A compile worker whose process group holds more than 500 MiB resident is killed and
    /// reported as `oom`, even when its counting allocator never saw the memory.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    #[test]
    fn the_compile_path_kills_a_worker_over_the_memory_limit() {
        match compile_probe(Probe::Resident) {
            Err(
                failure @ WorkerFailure::Process(
                    ProcessFailure::MemoryLimitExceeded { limit, measured },
                    ..,
                ),
            ) => {
                assert_eq!(limit, MEMORY_LIMIT_BYTES);
                assert!(measured > MEMORY_LIMIT_BYTES, "{measured}");
                assert_eq!(failure.outcome().key(), "oom");
                assert!(failure.measured().unwrap().elapsed < DEADLINE);
            }
            other => panic!("{other:?}"),
        }
    }

    /// The compile worker enables its 500-MiB allocator budget before serving: an allocation
    /// within it is admitted, one past it aborts the worker, a `worker_exit` stop.
    #[cfg(unix)]
    #[test]
    fn the_compile_worker_budgets_its_allocations() {
        match compile_probe(Probe::Budget) {
            Err(failure @ WorkerFailure::Exit(..)) => {
                let message = failure.message();
                assert!(message.contains("admitted"), "{message}");
                assert!(
                    message.contains(&format!("memory allocation of {} bytes failed", 128 << 20)),
                    "{message}"
                );
                assert!(!message.contains("more bytes"), "{message}");
                assert_eq!(failure.outcome().key(), "worker_exit");
            }
            other => panic!("{other:?}"),
        }
    }

    /// A compile worker whose binary does not count its allocations refuses to serve, so no
    /// compile runs without the budget.
    #[cfg(unix)]
    #[test]
    fn the_compile_worker_refuses_to_serve_without_the_counting_allocator() {
        match compile_probe(Probe::Uncounted) {
            Err(failure @ WorkerFailure::Exit(..)) => {
                let message = failure.message();
                assert!(
                    message.contains("does not install the counting allocator"),
                    "{message}"
                );
                assert_eq!(failure.outcome().key(), "worker_exit");
            }
            other => panic!("{other:?}"),
        }
    }
}
