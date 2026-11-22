//! The authored compile census through real compile workers: `corpus-triage --audit
//! authored-compile` serves the compile worker, compiles a committed public case's executor request
//! in it, blesses, adds the committed C3 venues' profiles with an additions bless, reruns warm
//! with no compile, and fails on a difference from the expectations, which an additions bless
//! refuses to write; a worker that stops is a failed cell. The case is copied into an owned scratch
//! corpus; nothing is read from the developer's private corpus or index.

use metal2vulkan_devtools::{engine_dir, llvm_assemble, AssembleError};
use metal2vulkan_validation::authored_compile::{
    expectations_path, run, venue_profiles, ArtifactBuild, Cell, CellOutcome, Expectations,
    FailureClass, Mode, Options, ProfileStem, WorkerCompiler,
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
        vec![&vec![Cell::bare(CellOutcome::Compiled)]]
    );

    // Nothing changed: no case is admitted, no compile runs.
    let (ok, output) = triage(root, &[]);
    assert!(ok, "{output}");
    assert_eq!(summary(&output, "compiles"), "0", "{output}");
    assert_eq!(summary(&output, "selected_cells"), "0", "{output}");
    assert_eq!(summary(&output, "cached_cells"), "1", "{output}");

    // An additions bless adds every committed C3 venue's profile the file lacks, compiling
    // only their cells; run again, it changes nothing.
    let mut profiles = vec![ProfileStem::parse("lavapipe").unwrap()];
    for profile in venue_profiles(&Venues::load(&Venues::committed_path()).unwrap()) {
        if !profiles.contains(&profile) {
            profiles.push(profile);
        }
    }
    let (ok, output) = triage(root, &["--bless-additions"]);
    assert!(ok, "{output}");
    assert_eq!(
        summary(&output, "selected_cells"),
        (profiles.len() - 1).to_string(),
        "{output}"
    );
    assert_eq!(summary(&output, "written"), "true", "{output}");
    let expected = Expectations::load(&expectations_path(root))
        .unwrap()
        .unwrap();
    assert_eq!(expected.profiles, profiles);
    // Each venue's compiled cell holds the digest of what a run executes there, at this build; the
    // file's own non-venue profile holds none.
    assert_eq!(expected.build, Some(ArtifactBuild::this()));
    for (profile, cell) in profiles.iter().zip(expected.cases.values().next().unwrap()) {
        let venue = profile.label() != "lavapipe";
        if cell.outcome == CellOutcome::Compiled {
            assert_eq!(cell.artifact.is_some(), venue, "{}", profile.label());
        } else {
            assert_eq!(cell.artifact, None, "{}", profile.label());
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
                        Cell::bare(CellOutcome::Refused("not_supported{bind}".into()));
                        profiles.len()
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
            profiles: Some(vec![ProfileStem::parse("lavapipe").unwrap()]),
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

/// A case over an indirect command buffer, which no candidate runs (here the runtime's limits on
/// compute commands and on commands the host encodes), is refused in the census by those
/// requirements, as a run's preparation refuses it: one cell per device, no compile, no digest.
#[test]
fn the_census_refuses_command_buffer_cases_by_their_requirements() {
    use metal2vulkan_validation::authored_compile::compile_case;
    if !llvm_as_available() {
        return;
    }
    let worker = {
        let program = bin_exe!("corpus-triage");
        let _ = Command::new(&program).arg("--help").output();
        CompileWorker::program(program)
    };
    let lavapipe = metal2vulkan_validation::worker::compile::WorkerDevice::new(
        metal2vulkan_validation::engine::load_description("lavapipe").unwrap(),
        metal2vulkan::device::DevicePolicy::RESIDENT_TABLES,
    )
    .unwrap();
    let (scratch, case) =
        corpus_with_templates("gpu-encoded-buffers-keep-the-host-encoded-dispatches");
    let cells = compile_case(scratch.path(), &worker, &case, &[&lavapipe]);
    assert_eq!(cells.len(), 1, "{cells:?}");
    assert_eq!(
        cells[0].outcome,
        Ok(CellOutcome::Refused(
            "not_supported{tooling_requirement:indirect_compute_commands,\
             host_encoded_indirect_commands}"
                .into()
        )),
        "{cells:?}"
    );
    assert_eq!(cells[0].compiles, 0, "{cells:?}");
    assert_eq!(cells[0].artifact, None, "{cells:?}");
}

/// An object or mesh case, which no candidate draws (`mesh_render_pipeline`), still compiles its
/// mesh render pipeline in the census: a run's preparation refuses the committed public mesh and
/// object cases by that requirement alone, the census's prepares them, and each one's pipeline,
/// its partners' stages with it, compiles in the worker as its one compile. Under the resident-table
/// domains: the identity domains refuse every stage but kernel, vertex and fragment.
#[test]
fn the_census_compiles_the_mesh_pipelines_no_candidate_draws() {
    use metal2vulkan_validation::authored_compile::compile_case;
    use metal2vulkan_validation::candidate::engine::{case_compile, prepare_case, prepare_compile};
    use metal2vulkan_validation::check::admission::{
        admit, ExecutionRequest, ExecutorError, NotSupported,
    };
    use metal2vulkan_validation::gpu::DeviceType;
    use metal2vulkan_validation::requirement::ToolingRequirement;
    if !llvm_as_available() {
        return;
    }
    let worker = {
        let program = bin_exe!("corpus-triage");
        let _ = Command::new(&program).arg("--help").output();
        CompileWorker::program(program)
    };
    let lavapipe = metal2vulkan_validation::worker::compile::WorkerDevice::new(
        metal2vulkan_validation::engine::load_description("lavapipe").unwrap(),
        metal2vulkan::device::DevicePolicy::RESIDENT_TABLES,
    )
    .unwrap();
    let device = DeviceType::Cpu;
    for name in [
        "a-mesh-grid-of-threadgroups-draws-through-its-fragment",
        "an-object-function-launches-its-mesh-partner-with-a-payload",
    ] {
        let (scratch, case) = corpus_with_templates(name);
        let admitted = || {
            admit(
                scratch.path(),
                case.clone(),
                ExecutionRequest::EngineCandidate { device },
            )
            .unwrap_or_else(|error| panic!("{name}: admission: {error}"))
        };
        let refused = prepare_case(device, admitted()).err();
        assert!(
            matches!(&refused, Some(ExecutorError::NotSupported(NotSupported::Requirements(stops)))
                if stops.iter().eq([&ToolingRequirement::MeshRenderPipeline])),
            "{name}: {refused:?}"
        );
        let prepared = prepare_compile(device, admitted())
            .unwrap_or_else(|error| panic!("{name}: preparation: {error}"));
        case_compile(&prepared).unwrap();
        let cells = compile_case(scratch.path(), &worker, &case, &[&lavapipe]);
        assert_eq!(cells.len(), 1, "{name}");
        assert_eq!(
            cells[0].outcome,
            Ok(CellOutcome::Compiled),
            "{name}: {cells:?}"
        );
        assert_eq!(cells[0].compiles, 1, "{name}");
    }
}

/// A case whose argument buffer holds a texture buffer, which a run draws through the runtime,
/// compiles in the census as the runtime compiles it: each committed case that holds one, on
/// lavapipe.
#[test]
fn the_census_compiles_a_texture_buffer_an_argument_buffer_holds() {
    use metal2vulkan_validation::authored_compile::compile_case;
    use metal2vulkan_validation::case::TextureType;
    if !llvm_as_available() {
        return;
    }
    let worker = {
        let program = bin_exe!("corpus-triage");
        let _ = Command::new(&program).arg("--help").output();
        CompileWorker::program(program)
    };
    let lavapipe = metal2vulkan_validation::worker::compile::WorkerDevice::new(
        metal2vulkan_validation::engine::load_description("lavapipe").unwrap(),
        metal2vulkan::device::DevicePolicy::default(),
    )
    .unwrap();
    let holding = CorpusStore::new(metal2vulkan_validation::source::corpus_root())
        .read_all_cases()
        .unwrap()
        .into_iter()
        .filter(|case| {
            case.argument_buffer_textures
                .iter()
                .any(|member| member.texture_type == TextureType::Buffer)
        })
        .collect::<Vec<_>>();
    assert!(
        !holding.is_empty(),
        "no committed case holds a texture buffer member"
    );
    for held in &holding {
        let (scratch, case) = corpus_with_templates(&held.name);
        let cells = compile_case(scratch.path(), &worker, &case, &[&lavapipe]);
        assert_eq!(cells.len(), 1, "{}", case.name);
        assert_eq!(
            cells[0].outcome,
            Ok(CellOutcome::Compiled),
            "{}: {cells:?}",
            case.name
        );
        assert_eq!(cells[0].compiles, 1, "{}", case.name);
    }
}
