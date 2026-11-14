//! The authored compile census through real compile workers: `corpus-triage --audit
//! authored-compile` serves the compile worker, compiles a committed public case's executor request
//! in it, blesses, adds the committed C3 venues' profile-legs with an additions bless, reruns warm
//! with no compile, and fails on a difference from the expectations, which an additions bless
//! refuses to write; a worker that stops is a failed cell. The case is copied into an owned scratch
//! corpus; nothing is read from the developer's private corpus or index.

use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
use metal2vulkan_validation::authored_compile::{
    expectations_path, run, venue_legs, ArtifactBuild, Cell, CellOutcome, Expectations,
    FailureClass, Mode, Options, ProfileLeg, WorkerCompiler,
};
use metal2vulkan_validation::c3::Venues;
use metal2vulkan_validation::store::CorpusStore;
use metal2vulkan_validation::worker::compile::CompileWorker;
use metal2vulkan_validation::ScratchDir;
use std::path::Path;
use std::process::Command;

/// The committed public kernel case the census compiles.
const CASE: &str = "single-thread-invert-word";

/// Whether `llvm-as` assembles the public fixtures here (a public row is assembled from its
/// text); without it the test is skipped, as the other fixture tests are.
fn llvm_as_available() -> bool {
    match llvm_assemble(&engine_dir().join("men2_eight_bit_points.ll")) {
        Ok(_) => true,
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            false
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected a public fixture: {why}"),
    }
}

/// A scratch corpus holding the committed case [`CASE`].
fn corpus() -> ScratchDir {
    let scratch = ScratchDir::new("authored-compile-census-cli").unwrap();
    let committed = CorpusStore::new(metal2vulkan_validation::source::corpus_root())
        .read_all_cases()
        .unwrap()
        .into_iter()
        .find(|case| case.name == CASE)
        .unwrap_or_else(|| panic!("no committed case {CASE}"));
    assert!(committed.public_source.is_some(), "{CASE} is a public case");
    CorpusStore::new(scratch.path())
        .put_case(committed)
        .unwrap();
    scratch
}

fn triage(root: &Path, extra: &[&str]) -> (bool, String) {
    let output = Command::new(bin_exe!("corpus-triage"))
        .arg("--corpus")
        .arg(root)
        .arg("--index")
        .arg(root.join("census.sqlite"))
        .args(["--audit", "authored-compile", "--jobs", "2"])
        .args(extra)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr);
    (output.status.success(), format!("{stdout}\n{stderr}"))
}

/// The summary line's `field`.
fn summary<'a>(output: &'a str, field: &str) -> &'a str {
    let line = output
        .lines()
        .find(|line| line.starts_with("authored-compile-summary\t"))
        .unwrap_or_else(|| panic!("no summary in {output}"));
    line.split('\t')
        .find_map(|cell| cell.strip_prefix(&format!("{field}=")))
        .unwrap_or_else(|| panic!("no {field} in {line}"))
}

#[test]
fn the_census_compiles_blesses_reruns_warm_and_fails_on_a_difference() {
    if !llvm_as_available() {
        return;
    }
    let scratch = corpus();
    let root = scratch.path();
    let (ok, output) = triage(root, &["--bless", "--profile", "lavapipe"]);
    assert!(ok, "{output}");
    assert_eq!(summary(&output, "compiles"), "1", "{output}");
    assert_eq!(summary(&output, "failed"), "0", "{output}");
    assert_eq!(summary(&output, "compiled"), "1", "{output}");
    let expected = Expectations::load(&expectations_path(root))
        .unwrap()
        .unwrap();
    assert_eq!(
        expected.cases.values().collect::<Vec<_>>(),
        vec![&vec![Cell::covered(CellOutcome::Compiled)]]
    );

    // Nothing changed: no case is admitted, no compile runs.
    let (ok, output) = triage(root, &[]);
    assert!(ok, "{output}");
    assert_eq!(summary(&output, "compiles"), "0", "{output}");
    assert_eq!(summary(&output, "selected_cells"), "0", "{output}");
    assert_eq!(summary(&output, "cached_cells"), "1", "{output}");

    // An additions bless adds every committed C3 venue's profile-leg the file lacks, compiling
    // only their cells; run again, it changes nothing.
    let mut legs = vec![ProfileLeg::parse("lavapipe").unwrap()];
    for leg in venue_legs(&Venues::load(&Venues::committed_path()).unwrap()) {
        if !legs.contains(&leg) {
            legs.push(leg);
        }
    }
    let (ok, output) = triage(root, &["--bless-additions"]);
    assert!(ok, "{output}");
    assert_eq!(
        summary(&output, "selected_cells"),
        (legs.len() - 1).to_string(),
        "{output}"
    );
    assert_eq!(summary(&output, "written"), "true", "{output}");
    let expected = Expectations::load(&expectations_path(root))
        .unwrap()
        .unwrap();
    assert_eq!(expected.profiles, legs);
    // Each venue's compiled cell holds the digest of what a run executes there, at this build; the
    // file's own non-venue profile-leg holds none.
    assert_eq!(expected.build, Some(ArtifactBuild::this()));
    for (leg, cell) in legs.iter().zip(expected.cases.values().next().unwrap()) {
        let venue = leg.label() != "lavapipe";
        if cell.outcome == CellOutcome::Compiled && cell.uncovered == 0 {
            assert_eq!(cell.artifact.is_some(), venue, "{}", leg.label());
        } else {
            assert_eq!(cell.artifact, None, "{}", leg.label());
        }
    }
    assert!(expected
        .cases
        .values()
        .flatten()
        .any(|cell| cell.artifact.is_some()));
    let (ok, output) = triage(root, &["--bless-additions"]);
    assert!(ok, "{output}");
    assert_eq!(summary(&output, "compiles"), "0", "{output}");
    assert_eq!(summary(&output, "written"), "false", "{output}");
    // Compiled again from nothing, every digest is the same: a check at this build passes.
    let output = Command::new(bin_exe!("corpus-triage"))
        .arg("--corpus")
        .arg(root)
        .arg("--index")
        .arg(root.join("cold.sqlite"))
        .args(["--audit", "authored-compile", "--jobs", "2"])
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&output.stdout).into_owned()
        + &String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{text}");
    assert_eq!(summary(&text, "artifacts_at_this_build"), "true", "{text}");
    assert_eq!(summary(&text, "moved_artifacts"), "0", "{text}");
    assert_ne!(summary(&text, "compiles"), "0", "{text}");

    // An expectation the run does not meet fails, naming the cell.
    let stale = Expectations {
        cases: expected
            .cases
            .keys()
            .map(|id| {
                (
                    id.clone(),
                    vec![
                        Cell::covered(CellOutcome::Refused("not_supported{bind}".into()));
                        legs.len()
                    ],
                )
            })
            .collect(),
        ..expected
    };
    stale.write(&expectations_path(root)).unwrap();
    let (ok, output) = triage(root, &[]);
    assert!(!ok, "{output}");
    assert!(
        output.contains("authored-compile-difference\tcompiles\t"),
        "{output}"
    );
    // An additions bless refuses it and leaves the file as it was.
    let before = std::fs::read_to_string(expectations_path(root)).unwrap();
    let (ok, output) = triage(root, &["--bless-additions"]);
    assert!(!ok, "{output}");
    assert_eq!(summary(&output, "written"), "false", "{output}");
    assert_eq!(
        std::fs::read_to_string(expectations_path(root)).unwrap(),
        before
    );
}

#[test]
fn a_stopped_compile_worker_fails_its_cell() {
    if !llvm_as_available() {
        return;
    }
    let scratch = corpus();
    let root = scratch.path();
    let index = root.join("census.sqlite");
    let expectations = expectations_path(root);
    // `true` exits at once without a result: the worker boundary's stop.
    let compiler = WorkerCompiler {
        root: root.to_path_buf(),
        worker: CompileWorker::program("true"),
    };
    let report = run(
        &Options {
            root,
            index: &index,
            expectations: &expectations,
            profiles: Some(vec![ProfileLeg::parse("lavapipe").unwrap()]),
            cases: None,
            jobs: 1,
            mode: Mode::Bless,
            required: Vec::new(),
            artifacts: Vec::new(),
        },
        &compiler,
    )
    .unwrap();
    assert!(!report.passed() && !report.blessed);
    assert_eq!(report.failures.len(), 1, "{:#?}", report.lines());
    assert_eq!(report.failures[0].failure.class, FailureClass::WorkerStop);
    assert!(!expectations.exists());
}

/// A scratch corpus holding the committed case named `name` and every pipeline template it names.
fn corpus_with_templates(name: &str) -> (ScratchDir, metal2vulkan_validation::case::AuthoredCase) {
    let scratch = ScratchDir::new("authored-compile-census-replay").unwrap();
    let committed = CorpusStore::new(metal2vulkan_validation::source::corpus_root())
        .read_all_cases()
        .unwrap();
    let case = committed
        .iter()
        .find(|case| case.name == name)
        .unwrap_or_else(|| panic!("no committed case {name}"))
        .clone();
    let templates = case
        .command_buffers
        .iter()
        .filter_map(|command| Some(command.execute.as_ref()?.pipeline.as_ref()?.case_id.clone()))
        .chain(
            case.pipeline_states
                .iter()
                .map(|state| state.case_id.clone()),
        )
        .collect::<std::collections::BTreeSet<_>>();
    let store = CorpusStore::new(scratch.path());
    for template in &templates {
        let template = committed
            .iter()
            .find(|row| &row.case_id == template)
            .unwrap_or_else(|| panic!("no committed template {template}"));
        store.put_case(template.clone()).unwrap();
    }
    store.put_case(case.clone()).unwrap();
    (scratch, case)
}

/// The executor's requests for a case whose executed compute command buffer the device replay
/// runs: on a profile with device addresses, the case's pipeline and each registered variant
/// compile in the worker; on one without, the command buffer is host-replayed and only the case's
/// own pipeline is compiled. A render command buffer is always host-replayed.
#[test]
fn case_compiles_lists_the_replay_variants_where_the_device_replay_runs() {
    use metal2vulkan_validation::authored_compile::{compile_prepared, CensusProfile};
    use metal2vulkan_validation::candidate::engine::{case_compiles, prepare_case, CompileSubject};
    use metal2vulkan_validation::check::admission::{admit, ExecutionRequest};
    use metal2vulkan_validation::gpu::DeviceType;
    if !llvm_as_available() {
        return;
    }
    let worker = {
        let program = bin_exe!("corpus-triage");
        let _ = Command::new(&program).arg("--help").output();
        CompileWorker::program(program)
    };
    let prepared = |root: &Path, case: &metal2vulkan_validation::case::AuthoredCase| {
        let device = DeviceType::Cpu;
        let admitted = admit(
            root,
            case.clone(),
            ExecutionRequest::EngineCandidate { device },
        )
        .unwrap_or_else(|error| panic!("admission: {error}"));
        prepare_case(device, admitted).unwrap_or_else(|error| panic!("preparation: {error}"))
    };
    let profile = |label: &str| CensusProfile::load(&ProfileLeg::parse(label).unwrap()).unwrap();
    let (lavapipe, floor) = (profile("lavapipe"), profile("vk12-floor"));

    let (scratch, case) =
        corpus_with_templates("gpu-encoded-buffers-keep-the-host-encoded-dispatches");
    let compute = prepared(scratch.path(), &case);
    let on_device = case_compiles(&compute, &lavapipe.device).unwrap();
    assert_eq!(on_device.host_replayed, 0);
    assert_eq!(on_device.compiles[0].subject, CompileSubject::Case);
    assert!(on_device.compiles.len() > 1);
    assert!(on_device.compiles[1..]
        .iter()
        .all(|compile| matches!(compile.subject, CompileSubject::ReplayVariant { .. })));
    let cell = compile_prepared(&compute, &worker, &lavapipe.device);
    assert_eq!(cell.outcome, Ok(CellOutcome::Compiled), "{cell:?}");
    assert_eq!(cell.compiles, on_device.compiles.len());
    // What a run executes there is digested: the case, each variant, and the preprocess module; the
    // same compiles digest the same.
    let digest = cell
        .artifact
        .clone()
        .expect("a device-replayed case's digest");
    let again = compile_prepared(&compute, &worker, &lavapipe.device);
    assert_eq!(again.artifact, Some(digest));
    let on_host = case_compiles(&compute, &floor.device).unwrap();
    assert_eq!(on_host.host_replayed, 1);
    assert_eq!(on_host.compiles.len(), 1);
    // Where the host replay runs a command buffer, the census compiles less than the run does:
    // no digest.
    let host = compile_prepared(&compute, &worker, &floor.device);
    assert_eq!(host.artifact, None, "{host:?}");

    let (scratch, case) =
        corpus_with_templates("gpu-encoded-vertex-buffers-run-the-host-encoded-draws");
    let render = prepared(scratch.path(), &case);
    let drawn = case_compiles(&render, &lavapipe.device).unwrap();
    assert_eq!(drawn.host_replayed, 1);
    assert_eq!(drawn.compiles.len(), 1);
}
