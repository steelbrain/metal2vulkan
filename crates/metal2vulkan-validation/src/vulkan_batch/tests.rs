//! The batch over synthetic cases, Metal rows, and fake executors (the tests own their inputs).

use super::*;
use crate::case::{
    BufferResource, Comparison, Dispatch, ExecutionSafety, OutputSelection, ResourceRole,
};
use crate::check::admission::NotSupported;
use crate::hash::sha256_bytes;
use crate::observation::{MetalStatus, ReferenceGap};
use crate::observation_contract::ComparisonClass;
use crate::refusal::Refusal;

const METAL_ENV: &str = "metal-test";
const ENV: &str = "lavapipe-test";

/// A one-word kernel case over the synthetic AIR `tag`.
fn case(tag: u8, name: &str) -> AuthoredCase {
    let mut case = AuthoredCase {
        air_sha256: format!("{tag:02x}").repeat(32),
        case_id: String::new(),
        name: name.into(),
        entry: "main".into(),
        stage: Stage::Kernel,
        buffers: vec![BufferResource {
            binding: 0,
            role: ResourceRole::Output,
            bytes_b64: None,
            initial_bytes_b64: Some("AAAAAA==".into()),
            rasterization_rate_map: None,
        }],
        argument_buffer_buffers: vec![],
        device_buffer_arrays: vec![],
        command_buffers: vec![],
        pipeline_states: vec![],
        threadgroup_memory: vec![],
        imageblock: None,
        fragment_imageblock: None,
        acceleration_structures: vec![],
        visible_function_references: vec![],
        visible_function_tables: vec![],
        intersection_function_tables: vec![],
        argument_buffer_intersection_function_tables: vec![],
        textures: vec![],
        texture_arrays: vec![],
        argument_buffer_textures: vec![],
        argument_buffer_samplers: vec![],
        samplers: vec![],
        render_targets: vec![],
        depth_stencil: None,
        vertex_inputs: vec![],
        vertex_observation: None,
        varying_values: Vec::new(),
        kernel_stage_inputs: vec![],
        function_constants: vec![],
        dispatch: Some(Dispatch {
            grid: [1, 1, 1],
            threads_per_threadgroup: [1, 1, 1],
            stage_in_region: None,
        }),
        draw: None,
        tessellation: None,
        mesh_draw: None,
        output: OutputSelection::Buffer {
            binding: 0,
            offset: 0,
            length: 4,
        },
        compare: Comparison::Exact,
        execution_safety: ExecutionSafety::LoopFree,
        fuel: None,
        out_of_bounds: None,
        render_pass_rate_map: None,
        raster_sample_count: None,
        rationale: None,
        authored_by: None,
        public_source: None,
        no_effect: None,
        creation_bound: None,
    };
    case.case_id = case.computed_case_id().unwrap();
    case
}

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// `case`'s Metal row in `environment_id` under `oracle_abi`, reading `output`.
fn metal(
    case: &AuthoredCase,
    environment_id: &str,
    oracle_abi: &str,
    output: &[u8],
) -> MetalObservation {
    MetalObservation {
        timing: Some(crate::observation::test_timing()),
        case_id: case.case_id.clone(),
        air_sha256: case.air_sha256.clone(),
        input_sha256: case.computed_input_sha256().unwrap(),
        metal_output_sha256: sha256_bytes(output),
        output_b64: b64(output),
        environment_id: environment_id.into(),
        environment: serde_json::json!({}),
        oracle_abi: oracle_abi.into(),
        status: MetalStatus::Qualified,
    }
}

/// A row of `case` in the slot `environment_id`, reading `output` against Metal's `golden`.
fn row(
    case: &AuthoredCase,
    environment_id: &str,
    output: &[u8],
    golden: &[u8],
) -> CandidateObservation {
    CandidateObservation {
        timing: Some(crate::observation::test_timing()),
        case_id: case.case_id.clone(),
        air_sha256: case.air_sha256.clone(),
        input_sha256: case.computed_input_sha256().unwrap(),
        golden_output_sha256: sha256_bytes(golden),
        spv_sha256: "33".repeat(32),
        translator_fingerprint: "44".repeat(32),
        candidate_output_sha256: sha256_bytes(output),
        output_b64: b64(output),
        backend: Backend::Vulkan,
        environment_id: environment_id.into(),
        environment: serde_json::json!({}),
        executor_abi: "v1".into(),
        comparison: ComparisonClass::Exact,
        census: None,
        status: if output == golden {
            CandidateStatus::Match
        } else {
            CandidateStatus::Mismatch
        },
    }
}

/// A refusal of `case` in the slot `environment_id` that ran in `leg`: a feature the device did
/// not enable.
fn refusal(case: &AuthoredCase, environment_id: &str, leg: &str) -> RefusalObservation {
    RefusalObservation {
        timing: Some(crate::observation::test_timing()),
        case_id: case.case_id.clone(),
        air_sha256: case.air_sha256.clone(),
        input_sha256: case.computed_input_sha256().unwrap(),
        translator_fingerprint: "44".repeat(32),
        backend: Backend::Vulkan,
        environment_id: environment_id.into(),
        environment: serde_json::json!({"robustness": {"leg": leg}}),
        executor_abi: "v1".into(),
        refusal: Refusal::of(&NotSupported::NotEnabled(vec!["rayQuery".into()])),
        artifact: None,
    }
}

fn done(case: &AuthoredCase, leg: Leg, outcome: Outcome, seconds: f64, detail: &str) -> DoneRun {
    DoneRun {
        case_id: case.case_id.clone(),
        pass: Pass::Leg(leg),
        outcome,
        ended: "exit=0".into(),
        seconds,
        detail: detail.into(),
    }
}

/// Only a case whose Metal row in the named environment is current under this oracle (the row
/// the executor compares against) is selected; the selection is in case-ID order.
#[test]
fn only_cases_with_a_current_metal_row_are_selected() {
    let current = case(1, "current");
    let other_environment = case(2, "other-environment");
    let old_oracle = case(3, "old-oracle");
    let other_input = case(4, "other-input");
    let none = case(5, "none");
    let also_current = case(6, "also-current");
    let mut changed_input = metal(&other_input, METAL_ENV, ORACLE_ABI, &[0; 4]);
    changed_input.input_sha256 = "55".repeat(32);
    let metals = [
        metal(&current, METAL_ENV, ORACLE_ABI, &[0; 4]),
        metal(&other_environment, "metal-elsewhere", ORACLE_ABI, &[0; 4]),
        metal(
            &old_oracle,
            METAL_ENV,
            "metal-literal-resources-v1",
            &[0; 4],
        ),
        changed_input,
        metal(&also_current, METAL_ENV, ORACLE_ABI, &[0; 4]),
    ];
    let cases = vec![
        also_current.clone(),
        none,
        other_input,
        old_oracle,
        other_environment,
        current.clone(),
    ];
    let selected = qualified_cases(cases, &metals, METAL_ENV)
        .into_iter()
        .map(|case| case.case_id)
        .collect::<Vec<_>>();
    let mut expected = vec![current.case_id, also_current.case_id];
    expected.sort();
    assert_eq!(selected, expected);
}

/// A rerun skips every (case, leg) the done-list holds, and `limit` bounds the cases (not the
/// runs) of one invocation.
#[test]
fn a_plan_skips_done_runs_and_bounds_its_cases() {
    let cases = [case(1, "a"), case(2, "b"), case(3, "c")];
    let legs = [Leg::Stated, Leg::RobustnessOff].map(Pass::Leg);
    let finished = [
        done(&cases[0], Leg::Stated, Outcome::Match, 1.0, ""),
        done(&cases[2], Leg::Stated, Outcome::Timeout, 1.0, ""),
        done(&cases[2], Leg::RobustnessOff, Outcome::Failed, 1.0, ""),
    ];
    let runs = |limit| {
        plan(&cases, &legs, &finished, limit)
            .into_iter()
            .map(|run| (run.case_id, run.pass))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        runs(None),
        [
            (cases[0].case_id.clone(), Pass::Leg(Leg::RobustnessOff)),
            (cases[1].case_id.clone(), Pass::Leg(Leg::Stated)),
            (cases[1].case_id.clone(), Pass::Leg(Leg::RobustnessOff)),
        ]
    );
    assert_eq!(
        runs(Some(1)),
        [(cases[0].case_id.clone(), Pass::Leg(Leg::RobustnessOff))]
    );
    assert!(plan(&cases[2..], &legs, &finished, None).is_empty());
}

/// A done-list line round-trips; a last line cut short (no newline) is not a finished run.
#[test]
fn the_done_list_round_trips_and_drops_a_cut_line() {
    let scratch = ScratchDir::new("batch-done-list").unwrap();
    let path = scratch.path().join(DONE_LIST);
    assert!(read_done_list(&path).unwrap().is_empty());
    let first = done(
        &case(1, "a"),
        Leg::RobustnessOff,
        Outcome::NotSupported,
        2.5,
        "not_supported{x}",
    );
    let second = done(&case(2, "b"), Leg::Stated, Outcome::Match, 0.25, "");
    let cut = second.line();
    fs::write(
        &path,
        format!("{}{}{}", first.line(), second.line(), &cut[..cut.len() / 2]),
    )
    .unwrap();
    assert_eq!(
        read_done_list(&path).unwrap(),
        [first.clone(), second.clone()]
    );
    // An append after the cut starts its own line: opening drops the cut line first.
    let third = done(
        &case(3, "c"),
        Leg::Stated,
        Outcome::Mismatch,
        1.0,
        "Mismatch c",
    );
    open_lines(&path)
        .unwrap()
        .write_all(third.line().as_bytes())
        .unwrap();
    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        format!("{}{}{}", first.line(), second.line(), third.line())
    );
    assert_eq!(
        read_done_list(&path).unwrap(),
        [first.clone(), second, third.clone()]
    );
    // A file that is all cut line is emptied; a complete file is kept as it is.
    fs::write(&path, &cut[..3]).unwrap();
    open_lines(&path).unwrap();
    assert_eq!(fs::read_to_string(&path).unwrap(), "");
    fs::write(&path, third.line()).unwrap();
    open_lines(&path).unwrap();
    assert_eq!(fs::read_to_string(&path).unwrap(), third.line());
    // A cut row is dropped by the reader and by the next append alike.
    let rows = scratch.path().join(ROWS);
    fs::write(&rows, "{\"cut").unwrap();
    assert!(read_rows(&rows).unwrap().is_empty());
    open_lines(&rows).unwrap();
    assert_eq!(fs::read_to_string(&rows).unwrap(), "");
    assert_eq!(sanitize("a\tb\nc"), "a b c");
    assert!(DoneRun::parse("not\ta\tline").is_err());
    fs::write(&path, "zz\tstated\tmatch\texit=0\t1.0\t\n").unwrap();
    assert!(read_done_list(&path).is_err());
}

/// An exit and the row or refusal a run recorded decide its outcome together: a row whose status
/// disagrees with the exit, a match exit without a row, a typed refusal that recorded a row, or a
/// refusal beside any exit but a typed refusal's is a failure. A typed refusal may record a
/// refusal or not (one before a device opened records none).
#[test]
fn an_outcome_reads_the_exit_and_the_row_together() {
    use CandidateStatus::{Match, Mismatch};
    for (ended, row, expected) in [
        (
            Ended::Code(exit_status::NOT_SUPPORTED),
            None,
            Outcome::NotSupported,
        ),
        (Ended::Code(0), Some(Match), Outcome::Failed),
        (Ended::Code(1), Some(Mismatch), Outcome::Failed),
        (Ended::Code(exit_status::REFUSED), None, Outcome::Failed),
        (Ended::Code(HANG_EXIT_CODE), None, Outcome::Failed),
        (
            Ended::Code(exit_status::VENUE_MISMATCH),
            None,
            Outcome::Failed,
        ),
        (
            Ended::Code(exit_status::NOT_SUPPORTED),
            Some(Match),
            Outcome::Failed,
        ),
        (Ended::TimedOut, None, Outcome::Timeout),
    ] {
        assert_eq!(classify(ended, row, true), expected, "{ended:?} {row:?}");
    }
    let cases = [
        (Ended::Code(0), Some(Match), Outcome::Match),
        (Ended::Code(1), Some(Mismatch), Outcome::Mismatch),
        (
            Ended::Code(exit_status::NOT_SUPPORTED),
            None,
            Outcome::NotSupported,
        ),
        (Ended::Code(exit_status::REFUSED), None, Outcome::Refused),
        (
            Ended::Code(exit_status::DEVICE_CRASH),
            None,
            Outcome::DeviceCrash,
        ),
        (
            Ended::Code(exit_status::NEEDS_METAL),
            None,
            Outcome::NeedsMetal,
        ),
        (Ended::Code(HANG_EXIT_CODE), None, Outcome::WaitTimeout),
        (
            Ended::Code(exit_status::VENUE_MISMATCH),
            None,
            Outcome::VenueMismatch,
        ),
        (
            Ended::Code(exit_status::VENUE_MISMATCH),
            Some(Match),
            Outcome::Failed,
        ),
        (Ended::TimedOut, Some(Match), Outcome::Timeout),
        (Ended::Code(0), None, Outcome::Failed),
        (Ended::Code(0), Some(Mismatch), Outcome::Failed),
        (Ended::Code(1), Some(Match), Outcome::Failed),
        (Ended::Code(1), None, Outcome::Failed),
        (
            Ended::Code(exit_status::NOT_SUPPORTED),
            Some(Match),
            Outcome::Failed,
        ),
        (Ended::Signal(11), None, Outcome::Failed),
    ];
    for (ended, row, expected) in cases {
        assert_eq!(classify(ended, row, false), expected, "{ended:?} {row:?}");
    }
    for outcome in OUTCOMES {
        assert_eq!(Outcome::parse(outcome.label()), Some(outcome));
    }
    assert_eq!(Outcome::NotSupported.class(), Class::Refusal);
    assert_eq!(Outcome::DeviceCrash.class(), Class::Failure);
    assert_eq!(Outcome::VenueMismatch.class(), Class::Failure);
    assert_eq!(Outcome::VenueMismatch.label(), "venue_mismatch");
    for outcome in OUTCOMES {
        assert_eq!(
            stops_the_batch(outcome),
            outcome == Outcome::VenueMismatch,
            "{outcome:?}"
        );
    }
}

/// A run whose device is unlike its venue's profile (exit 7) is recorded with the executor's
/// verdict line as its detail and stops any batch, on a CPU as on a GPU: the batch returns an error
/// naming the case, and no later run starts (every later one would meet the same device); a resumed
/// batch starts at the next run.
#[cfg(unix)]
#[test]
fn a_venue_mismatch_is_recorded_and_stops_the_batch() {
    let scratch = ScratchDir::new("batch-venue").unwrap();
    let corpus = scratch.path().join("corpus");
    let store = CorpusStore::new(&corpus);
    let (first, second) = (case(1, "a"), case(2, "b"));
    for case in [&first, &second] {
        store.put_case(case.clone()).unwrap();
    }
    let started = scratch.path().join("started");
    let executor = scratch.path().join("fake-corpus-vulkan");
    executable(
        &executor,
        &format!(
            r#"#!/bin/sh
id=
while [ $# -gt 0 ]; do
  case "$1" in
    --case-id) id=$2; shift 2 ;;
    --robustness-off) shift ;;
    *) shift 2 ;;
  esac
done
echo "$id" >> '{started}'
echo "venue_mismatch{{toy}}: the device differs from venue v's profile toy in 1 member(s): x"
echo "recapture the profile" >&2
exit {code}
"#,
            started = started.display(),
            code = exit_status::VENUE_MISMATCH,
        ),
    );
    let cases = [first.clone(), second.clone()];
    let runs = plan(&cases, &[Pass::Leg(Leg::Stated)], &[], None);
    for config in [
        BatchConfig {
            jobs: 1,
            ..config(scratch.path(), executor.clone(), Duration::from_secs(60))
        },
        gpu_config(scratch.path(), executor.clone(), Duration::from_secs(60)),
    ] {
        let _ = fs::remove_dir_all(&config.out);
        let _ = fs::remove_file(&started);
        let error = run_batch(&config, &runs, |_| {}).unwrap_err();
        assert!(error.contains("the batch stopped"), "{error}");
        assert!(error.contains(&first.case_id), "{error}");
        assert_eq!(
            fs::read_to_string(&started).unwrap(),
            format!("{}\n", first.case_id),
            "a run started after the venue mismatch"
        );
        let listed = read_done_list(&config.out.join(DONE_LIST)).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].outcome, Outcome::VenueMismatch);
        assert!(
            listed[0].detail.starts_with("venue_mismatch{toy}"),
            "{}",
            listed[0].detail
        );
        let resumed = plan(&cases, &[Pass::Leg(Leg::Stated)], &listed, None);
        assert_eq!(resumed.len(), 1);
        assert_eq!(resumed[0].case_id, second.case_id);
        if let BatchDevice::Gpu(tripwire) = &config.device {
            assert!(!tripwire.current.exists(), "a venue mismatch is no hang");
        }
    }
}

#[cfg(unix)]
fn executable(path: &Path, script: &str) {
    use std::os::unix::fs::PermissionsExt as _;
    fs::write(path, script).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

#[cfg(unix)]
fn config(scratch: &Path, executor: PathBuf, timeout: Duration) -> BatchConfig {
    BatchConfig {
        executor,
        backend: BatchBackend::Vulkan,
        corpus: scratch.join("corpus"),
        out: scratch.join("out"),
        metal_environment_id: METAL_ENV.into(),
        environment_id: ENV.into(),
        restricted_to: None,
        device: BatchDevice::Cpu,
        jobs: 2,
        timeout,
        gpu_wait_timeout: Some(30),
        source_index: None,
    }
}

/// [`config`] on a GPU, one run at a time, under a tripwire of the test's own.
#[cfg(unix)]
fn gpu_config(scratch: &Path, executor: PathBuf, timeout: Duration) -> BatchConfig {
    BatchConfig {
        device: BatchDevice::Gpu(TripwirePaths {
            current: scratch.join("gpu-state/hang"),
            legacy: scratch.join("legacy-hang"),
        }),
        jobs: 1,
        ..config(scratch, executor, timeout)
    }
}

/// Every file under `root` with its bytes, by path.
#[cfg(unix)]
fn snapshot(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else {
                files.insert(path.clone(), fs::read(&path).unwrap());
            }
        }
    }
    files
}

/// A batch runs each (case, leg) through the executor on its device type in a job store of its
/// own (links to the corpus, its private sources included, its own candidate and refusal shards,
/// its own TMPDIR), naming each case's AIR so the executor reads only its case shard and naming no
/// source index when it has none, records its build in the out directory, keeps the rows of matches
/// and mismatches and the refusals of typed refusals verbatim, logs every other run, never writes
/// the corpus (removing the job stores leaves the linked private sources whole), and a second
/// invocation finds nothing left to run.
#[cfg(unix)]
#[test]
fn a_batch_runs_every_case_in_a_job_store_and_resumes() {
    let scratch = ScratchDir::new("batch-run").unwrap();
    let corpus = scratch.path().join("corpus");
    let store = CorpusStore::new(&corpus);
    let (matching, mismatching, refused) = (case(1, "a"), case(2, "b"), case(3, "c"));
    for case in [&matching, &mismatching, &refused] {
        store.put_case(case.clone()).unwrap();
    }
    fs::create_dir_all(corpus.join("observations/vulkan")).unwrap();
    // Stand-ins for the private sources and library modules: each job store links them, and
    // removing the stores must leave them as they were.
    for (directory, file) in [
        ("local/sources", "shard_000.jsonl"),
        ("local/library-modules", "shard_000.jsonl"),
    ] {
        fs::create_dir_all(corpus.join(directory)).unwrap();
        fs::write(corpus.join(directory).join(file), b"{}\n").unwrap();
    }
    let before = snapshot(&corpus);
    let shard = |case: &AuthoredCase| shard_name_for_hash(&case.air_sha256).unwrap();
    let executor = scratch.path().join("fake-corpus-vulkan");
    executable(
        &executor,
        &format!(
            r#"#!/bin/sh
store= id= air= slot={ENV}@engine wait= device=
while [ $# -gt 0 ]; do
  case "$1" in
    --corpus) store=$2; shift 2 ;;
    --case-id) id=$2; shift 2 ;;
    --air-sha256) air=$2; shift 2 ;;
    --robustness-off) slot={ENV}@engine+robustness-off; shift ;;
    --gpu-wait-timeout) wait=$2; shift 2 ;;
    --device-type) device=$2; shift 2 ;;
    --metal-environment-id|--environment-id) shift 2 ;;
    *) echo "unexpected $1" >&2; exit 9 ;;
  esac
done
case "$id:$air" in
  {m}:{ma}|{x}:{xa}|{r}:{ra}) ;;
  *) echo "the batch named AIR '$air' for $id" >&2; exit 9 ;;
esac
[ -L "$store/cases" ] && [ -L "$store/local/sources" ] || exit 9
[ -L "$store/local/library-modules" ] || exit 9
[ -d "$store/observations/vulkan" ] && [ "$wait" = 30 ] && [ "$device" = cpu ] || exit 9
[ -d "$store/observations/refusals/vulkan" ] && [ ! -L "$store/observations/refusals" ] || exit 9
[ -d "$TMPDIR" ] && touch "$TMPDIR/left-behind" || exit 9
row() {{
  printf '{{"case_id":"%s","environment_id":"%s","status":"%s"}}\n' "$id" "$slot" "$1" \
    >> "$store/observations/vulkan/$2"
}}
case "$id" in
  {m}) row match {ms}; echo "Match $id"; exit 0 ;;
  {x}) row mismatch {xs}; echo "Mismatch $id"; echo "candidate output does not match" >&2; exit 1 ;;
  {r})
    # The stated leg refuses after the device opened (a refusal record); the other before.
    [ "$slot" = {ENV}@engine ] && printf '{{"case_id":"%s","environment_id":"%s"}}\n' \
      "$id" "$slot" >> "$store/observations/refusals/vulkan/{rs}"
    echo "not_supported{{product: test}}"; exit 4 ;;
esac
exit 9
"#,
            m = matching.case_id,
            ma = matching.air_sha256,
            ms = shard(&matching),
            x = mismatching.case_id,
            xa = mismatching.air_sha256,
            xs = shard(&mismatching),
            r = refused.case_id,
            ra = refused.air_sha256,
            rs = shard(&refused),
        ),
    );
    let config = config(scratch.path(), executor, Duration::from_secs(60));
    let cases = [matching.clone(), mismatching.clone(), refused.clone()];
    let legs = [Leg::Stated, Leg::RobustnessOff].map(Pass::Leg);
    let runs = plan(&cases, &legs, &[], None);
    assert_eq!(runs.len(), 6);
    let seen = Mutex::new(0usize);
    let finished = run_batch(&config, &runs, |_| *seen.lock().unwrap() += 1).unwrap();
    assert_eq!(*seen.lock().unwrap(), 6);
    assert_eq!(finished.len(), 6);

    let listed = read_done_list(&config.out.join(DONE_LIST)).unwrap();
    let outcome_of = |case: &AuthoredCase, leg: Pass| {
        let runs = listed
            .iter()
            .filter(|run| run.case_id == case.case_id && run.pass == leg)
            .collect::<Vec<_>>();
        assert_eq!(runs.len(), 1);
        (runs[0].outcome, runs[0].detail.clone())
    };
    for leg in legs {
        assert_eq!(outcome_of(&matching, leg), (Outcome::Match, String::new()));
        assert_eq!(
            outcome_of(&mismatching, leg),
            (
                Outcome::Mismatch,
                format!("Mismatch {}", mismatching.case_id)
            )
        );
        assert_eq!(
            outcome_of(&refused, leg),
            (Outcome::NotSupported, "not_supported{product: test}".into())
        );
    }
    let mut rows = fs::read_to_string(config.out.join(ROWS))
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect::<Vec<_>>();
    rows.sort();
    let mut expected = Vec::new();
    for (case, status) in [(&matching, "match"), (&mismatching, "mismatch")] {
        for slot in ["@engine", "@engine+robustness-off"] {
            expected.push(format!(
                r#"{{"case_id":"{}","environment_id":"{ENV}{slot}","status":"{status}"}}"#,
                case.case_id
            ));
        }
    }
    expected.sort();
    assert_eq!(rows, expected);
    assert_eq!(
        fs::read_to_string(config.out.join(REFUSALS)).unwrap(),
        format!(
            "{{\"case_id\":\"{}\",\"environment_id\":\"{ENV}@engine\"}}\n",
            refused.case_id
        )
    );
    assert_eq!(
        BatchBuild::read(&config.out).unwrap(),
        Some(BatchBuild::of(&config))
    );
    let mut logs = fs::read_dir(config.out.join(LOGS))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect::<Vec<_>>();
    logs.sort();
    let mut expected_logs = [&mismatching, &refused]
        .iter()
        .flat_map(|case| {
            ["stated", "robustness-off"].map(|leg| format!("{}.{leg}.txt", case.case_id))
        })
        .collect::<Vec<_>>();
    expected_logs.sort();
    assert_eq!(logs, expected_logs);
    // Every run wrote its job store: the corpus is byte for byte what it was.
    assert_eq!(snapshot(&corpus), before);

    let rerun = plan(&cases, &legs, &listed, None);
    assert!(rerun.is_empty(), "{rerun:?}");
}

/// A MoltenVK batch runs its executor in job stores of MoltenVK's own: their candidate and refusal
/// shards are MoltenVK's, the Vulkan rows are linked like the Metal rows, and the rows and
/// refusals it keeps are the MoltenVK slots'. A batch given a source index names it to every run,
/// with each case's AIR.
#[cfg(unix)]
#[test]
fn a_moltenvk_batch_records_moltenvk_rows() {
    let scratch = ScratchDir::new("batch-moltenvk").unwrap();
    let corpus = scratch.path().join("corpus");
    let store = CorpusStore::new(&corpus);
    let (matching, refused) = (case(1, "a"), case(3, "c"));
    for case in [&matching, &refused] {
        store.put_case(case.clone()).unwrap();
    }
    fs::create_dir_all(corpus.join("observations/vulkan")).unwrap();
    let shard = |case: &AuthoredCase| shard_name_for_hash(&case.air_sha256).unwrap();
    let index = scratch.path().join("corpus-index.sqlite");
    let executor = scratch.path().join("fake-corpus-moltenvk");
    executable(
        &executor,
        &format!(
            r#"#!/bin/sh
store= id= index= slot={ENV}@engine
while [ $# -gt 0 ]; do
  case "$1" in
    --corpus) store=$2; shift 2 ;;
    --case-id) id=$2; shift 2 ;;
    --source-index) index=$2; shift 2 ;;
    --robustness-off) slot={ENV}@engine+robustness-off; shift ;;
    --device-type) [ "$2" = gpu ] || exit 9; shift 2 ;;
    --metal-environment-id|--environment-id|--gpu-wait-timeout|--air-sha256) shift 2 ;;
    *) echo "unexpected $1" >&2; exit 9 ;;
  esac
done
[ "$index" = {index} ] || {{ echo "source index '$index'" >&2; exit 9; }}
[ -L "$store/observations/vulkan" ] && [ ! -e "$store/observations/refusals/vulkan" ] || exit 9
[ -d "$store/observations/moltenvk" ] && [ ! -L "$store/observations/moltenvk" ] || exit 9
[ -d "$store/observations/refusals/moltenvk" ] || exit 9
case "$id" in
  {m}) printf '{{"case_id":"%s","environment_id":"%s","status":"match"}}\n' "$id" "$slot" \
    >> "$store/observations/moltenvk/{ms}"; echo "Match $id"; exit 0 ;;
  {r}) printf '{{"case_id":"%s","environment_id":"%s"}}\n' "$id" "$slot" \
    >> "$store/observations/refusals/moltenvk/{rs}"
    echo "not_supported{{product: test}}"; exit 4 ;;
esac
exit 9
"#,
            m = matching.case_id,
            ms = shard(&matching),
            r = refused.case_id,
            rs = shard(&refused),
            index = index.display(),
        ),
    );
    let config = BatchConfig {
        backend: BatchBackend::Moltenvk,
        source_index: Some(index.clone()),
        ..gpu_config(scratch.path(), executor, Duration::from_secs(60))
    };
    let runs = plan(
        &[matching.clone(), refused.clone()],
        &[Pass::Leg(Leg::Stated)],
        &[],
        None,
    );
    let finished = run_batch(&config, &runs, |_| {}).unwrap();
    assert_eq!(
        finished
            .iter()
            .map(|run| run.outcome)
            .collect::<std::collections::BTreeSet<_>>(),
        [Outcome::Match, Outcome::NotSupported].into()
    );
    assert_eq!(
        fs::read_to_string(config.out.join(ROWS)).unwrap(),
        format!(
            "{{\"case_id\":\"{}\",\"environment_id\":\"{ENV}@engine\",\"status\":\"match\"}}\n",
            matching.case_id
        )
    );
    assert_eq!(
        fs::read_to_string(config.out.join(REFUSALS)).unwrap(),
        format!(
            "{{\"case_id\":\"{}\",\"environment_id\":\"{ENV}@engine\"}}\n",
            refused.case_id
        )
    );
    let recorded = BatchBuild::read(&config.out).unwrap().unwrap();
    assert_eq!(recorded.backend, BatchBackend::Moltenvk);
    assert_eq!(recorded, BatchBuild::of(&config));
}

/// A restricted batch runs every case under `--profiles-layer P` and keeps the rows and refusals
/// of the `ID@engine@P` slots, which its report reads.
#[cfg(unix)]
#[test]
fn a_restricted_batch_runs_every_case_under_its_profile() {
    let scratch = ScratchDir::new("batch-restricted").unwrap();
    let corpus = scratch.path().join("corpus");
    let store = CorpusStore::new(&corpus);
    let matching = case(1, "a");
    store.put_case(matching.clone()).unwrap();
    let shard = shard_name_for_hash(&matching.air_sha256).unwrap();
    let executor = scratch.path().join("fake-corpus-vulkan");
    executable(
        &executor,
        &format!(
            r#"#!/bin/sh
store= id= slot= profile=
leg=
while [ $# -gt 0 ]; do
  case "$1" in
    --corpus) store=$2; shift 2 ;;
    --case-id) id=$2; shift 2 ;;
    --profiles-layer) profile=$2; shift 2 ;;
    --robustness-off) leg=+robustness-off; shift ;;
    *) shift 2 ;;
  esac
done
[ "$profile" = vk12-floor ] || exit 9
printf '{{"case_id":"%s","environment_id":"%s","status":"match"}}\n' "$id" "{ENV}@engine@$profile$leg" \
  >> "$store/observations/vulkan/{shard}"
echo "Match $id"
"#
        ),
    );
    let config = BatchConfig {
        restricted_to: Some("vk12-floor".into()),
        ..config(scratch.path(), executor, Duration::from_secs(60))
    };
    let legs = [Leg::Stated, Leg::RobustnessOff].map(Pass::Leg);
    let runs = plan(std::slice::from_ref(&matching), &legs, &[], None);
    let finished = run_batch(&config, &runs, |_| {}).unwrap();
    assert!(
        finished.iter().all(|run| run.outcome == Outcome::Match),
        "{finished:?}"
    );
    let mut rows = fs::read_to_string(config.out.join(ROWS))
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect::<Vec<_>>();
    rows.sort();
    assert_eq!(
        rows,
        ["@engine@vk12-floor", "@engine@vk12-floor+robustness-off"].map(|slot| format!(
            r#"{{"case_id":"{}","environment_id":"{ENV}{slot}","status":"match"}}"#,
            matching.case_id
        ))
    );
    assert_eq!(
        BatchBuild::read(&config.out)
            .unwrap()
            .unwrap()
            .restricted_to
            .as_deref(),
        Some("vk12-floor")
    );
}

/// A run past its timeout is a timeout, and its whole session goes with it: a worker the
/// executor started in a process group of its own (as its compile and device workers are) is
/// killed too, not left behind.
#[cfg(unix)]
#[test]
fn a_timed_out_run_takes_its_whole_session_with_it() {
    let scratch = ScratchDir::new("batch-timeout").unwrap();
    let corpus = scratch.path().join("corpus");
    let stuck = case(7, "stuck");
    CorpusStore::new(&corpus).put_case(stuck.clone()).unwrap();
    let pid_file = scratch.path().join("worker.pid");
    let executor = scratch.path().join("fake-corpus-vulkan");
    executable(
        &executor,
        &format!(
            "#!/bin/sh\nperl -e 'setpgrp(0, 0); open(my $f, \">\", $ARGV[0]) or die; print $f $$; \
             close($f); sleep 600' '{}' &\nsleep 600\n",
            pid_file.display()
        ),
    );
    let config = config(scratch.path(), executor, Duration::from_secs(2));
    let runs = plan(
        std::slice::from_ref(&stuck),
        &[Pass::Leg(Leg::Stated)],
        &[],
        None,
    );
    let finished = run_batch(&config, &runs, |_| {}).unwrap();
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0].outcome, Outcome::Timeout);
    assert_eq!(finished[0].ended, "timeout");
    assert_eq!(finished[0].detail, "killed after 2 s");
    let worker = fs::read_to_string(&pid_file)
        .unwrap()
        .trim()
        .parse::<libc::pid_t>()
        .unwrap();
    // SAFETY: a signal-0 probe of a pid this test's worker wrote.
    let alive = unsafe { libc::kill(worker, 0) } == 0;
    assert!(
        !alive || session_member_is_zombie(worker),
        "worker {worker} survived its run's timeout"
    );
}

#[cfg(target_os = "linux")]
fn session_member_is_zombie(pid: libc::pid_t) -> bool {
    fs::read_to_string(format!("/proc/{pid}/stat")).is_ok_and(|stat| {
        stat.rsplit_once(')')
            .is_some_and(|(_, rest)| rest.trim_start().starts_with('Z'))
    })
}

#[cfg(all(unix, not(target_os = "linux")))]
fn session_member_is_zombie(_pid: libc::pid_t) -> bool {
    false
}

/// The report counts each leg's runs by class and outcome, and each family's by class, names
/// every run that did not match, gives a mismatch its comparison and differing words against
/// Metal and a typed refusal its recorded class, and counts the selected cases a leg has not run
/// as pending.
#[test]
fn the_report_counts_by_leg_family_and_class_and_names_each_non_match() {
    let (matching, mismatching) = (case(1, "a"), case(2, "b"));
    let mut refused = case(3, "c");
    refused.public_source = Some("crates/metal2vulkan-validation/fixtures/public/c.ll".into());
    let mut vertex = case(4, "d");
    vertex.stage = Stage::Vertex;
    vertex.case_id = vertex.computed_case_id().unwrap();
    let golden = [1, 0, 0, 0, 2, 0, 0, 0];
    let output = [1, 0, 0, 0, 3, 0, 0, 0];
    let metals = [
        metal(&matching, METAL_ENV, ORACLE_ABI, &[0; 4]),
        metal(&mismatching, METAL_ENV, ORACLE_ABI, &golden),
        metal(&refused, METAL_ENV, ORACLE_ABI, &[0; 4]),
        metal(&vertex, METAL_ENV, ORACLE_ABI, &[0; 4]),
    ];
    let slot = engine_slot(ENV, None, Leg::Stated);
    let rows = [row(&mismatching, &slot, &output, &golden)];
    let runs = [
        done(&matching, Leg::Stated, Outcome::Failed, 9.0, "first try"),
        done(&matching, Leg::Stated, Outcome::Match, 1.0, ""),
        done(
            &mismatching,
            Leg::Stated,
            Outcome::Mismatch,
            2.0,
            "Mismatch",
        ),
        done(
            &refused,
            Leg::Stated,
            Outcome::NotSupported,
            0.5,
            "not_supported{x}",
        ),
        done(
            &vertex,
            Leg::Stated,
            Outcome::DeviceCrash,
            3.0,
            "device_crash{SIGSEGV}",
        ),
        done(&matching, Leg::RobustnessOff, Outcome::Match, 1.5, ""),
        done(
            &refused,
            Leg::RobustnessOff,
            Outcome::NotSupported,
            0.5,
            "not_supported{y}",
        ),
    ];
    let lines = report(
        &qualified_cases(
            vec![
                matching.clone(),
                mismatching.clone(),
                refused.clone(),
                vertex.clone(),
            ],
            &metals,
            METAL_ENV,
        ),
        &metals,
        METAL_ENV,
        ENV,
        None,
        &[Leg::Stated, Leg::RobustnessOff],
        &BatchOutput {
            done: runs.to_vec(),
            rows: rows.to_vec(),
            refusals: vec![refusal(&refused, &slot, "stated")],
        },
    );
    let expect = |line: &str| assert!(lines.iter().any(|l| l == line), "{line}\n{lines:#?}");
    expect(
        "batch-report leg=stated selected=4 ran=4 pending=0 match=1 mismatch=1 refusal=1 \
         failure=1 not_supported=1 refused=0 device_crash=1 needs_metal=0 venue_mismatch=0 \
         wait_timeout=0 timeout=0 failed=0",
    );
    expect(
        "batch-family leg=stated family=kernel.buffers cases=3 match=1 mismatch=1 refusal=1 \
         failure=0",
    );
    expect(
        "batch-family leg=stated family=vertex.buffers cases=1 match=0 mismatch=0 refusal=0 \
         failure=1",
    );
    expect("batch-scope leg=stated scope=private cases=3 match=1 mismatch=1 refusal=0 failure=1");
    expect("batch-scope leg=stated scope=public cases=1 match=0 mismatch=0 refusal=1 failure=0");
    expect(&format!(
        "batch-run case={} leg=stated scope=private family=kernel.buffers outcome=mismatch \
         seconds=2.000 name=b detail=Mismatch comparison=\"exact\" words=1/2 \
         first=1:00000002/00000003",
        mismatching.case_id
    ));
    expect(&format!(
        "batch-run case={} leg=stated scope=public family=kernel.buffers \
         outcome=not_supported seconds=0.500 name=c detail=not_supported{{x}} refusal=device",
        refused.case_id
    ));
    expect(&format!(
        "batch-run case={} leg=robustness-off scope=public family=kernel.buffers \
         outcome=not_supported seconds=0.500 name=c detail=not_supported{{y}} refusal=unrecorded",
        refused.case_id
    ));
    expect(
        "batch-report leg=robustness-off selected=4 ran=2 pending=2 match=1 mismatch=0 refusal=1 \
         failure=0 not_supported=1 refused=0 device_crash=0 needs_metal=0 venue_mismatch=0 \
         wait_timeout=0 timeout=0 failed=0",
    );
    expect(&format!(
        "batch-slowest case={} leg=stated seconds=3.000 outcome=device_crash",
        vertex.case_id
    ));
    assert!(
        !lines.iter().any(|line| line.contains("first try")),
        "{lines:#?}"
    );
}

/// A report narrowed by `--case-id` counts only the requested cases: its `selected=` is the
/// requested count, and a run of another case in the same output is neither counted nor named. A
/// requested case without a current Metal row is an error naming it, as it is for a batch.
#[test]
fn a_report_narrowed_to_requested_cases_counts_only_them() {
    let (wanted, other, unqualified) = (case(1, "a"), case(2, "b"), case(3, "c"));
    let metals = [
        metal(&wanted, METAL_ENV, ORACLE_ABI, &[0; 4]),
        metal(&other, METAL_ENV, ORACLE_ABI, &[0; 4]),
    ];
    let qualified = || {
        qualified_cases(
            vec![wanted.clone(), other.clone(), unqualified.clone()],
            &metals,
            METAL_ENV,
        )
    };
    assert_eq!(
        requested_cases(qualified(), &[], METAL_ENV).unwrap().len(),
        2
    );
    let selected = requested_cases(
        qualified(),
        std::slice::from_ref(&wanted.case_id),
        METAL_ENV,
    )
    .unwrap();
    assert_eq!(selected, std::slice::from_ref(&wanted));
    let error = requested_cases(
        qualified(),
        &[wanted.case_id.clone(), unqualified.case_id.clone()],
        METAL_ENV,
    )
    .unwrap_err();
    assert!(
        error.contains(&unqualified.case_id) && !error.contains(&wanted.case_id),
        "{error}"
    );
    let runs = [
        done(&wanted, Leg::Stated, Outcome::Match, 1.0, ""),
        done(&other, Leg::Stated, Outcome::Failed, 2.0, "other failed"),
    ];
    let lines = report(
        &selected,
        &metals,
        METAL_ENV,
        ENV,
        None,
        &[Leg::Stated],
        &BatchOutput {
            done: runs.to_vec(),
            rows: Vec::new(),
            refusals: Vec::new(),
        },
    );
    assert!(
        lines.iter().any(|line| line
            == "batch-report leg=stated selected=1 ran=1 pending=0 match=1 mismatch=0 \
                refusal=0 failure=0 not_supported=0 refused=0 device_crash=0 needs_metal=0 \
                venue_mismatch=0 wait_timeout=0 timeout=0 failed=0"),
        "{lines:#?}"
    );
    assert!(
        !lines.iter().any(|line| line.contains(&other.case_id)),
        "{lines:#?}"
    );
}

/// Installing writes, for each (case, leg), the evidence of its last run that recorded any,
/// through the store's validated writes: the slot's last row after a match or a mismatch, its last
/// refusal after a typed refusal (which replaces a row the store held), nothing after a refusal
/// before a device opened, and nothing no finished run lists. A row that is not of the backend
/// installed is refused, and so is an out directory whose record names the other backend; the
/// record comes back with what was installed (an out directory without one is Vulkan's).
#[test]
fn installed_evidence_is_each_slots_last_run() {
    let scratch = ScratchDir::new("batch-install").unwrap();
    let store = CorpusStore::new(scratch.path().join("corpus"));
    let (first, second, third) = (case(1, "a"), case(2, "b"), case(3, "c"));
    for case in [&first, &second, &third] {
        store.put_case(case.clone()).unwrap();
    }
    let stated = engine_slot(ENV, None, Leg::Stated);
    let off = engine_slot(ENV, None, Leg::RobustnessOff);
    // The store holds a row the second case's last run replaces with a refusal.
    store
        .upsert_candidate(row(&second, &stated, &[0; 4], &[0; 4]))
        .unwrap();
    let out = scratch.path().join("out");
    fs::create_dir_all(&out).unwrap();
    let stale = row(&first, &stated, &[9, 0, 0, 0], &[0; 4]);
    let fresh = row(&first, &stated, &[0; 4], &[0; 4]);
    let cut = row(&first, &off, &[0; 4], &[0; 4]);
    let replaced = row(&second, &stated, &[0; 4], &[0; 4]);
    let after_refusal = row(&third, &stated, &[0; 4], &[0; 4]);
    let refused = refusal(&second, &stated, "stated");
    let superseded = refusal(&third, &stated, "stated");
    fs::write(
        out.join(ROWS),
        [&stale, &replaced, &fresh, &cut, &after_refusal]
            .map(line)
            .concat(),
    )
    .unwrap();
    fs::write(
        out.join(REFUSALS),
        [&superseded, &refused].map(line).concat(),
    )
    .unwrap();
    let runs = [
        done(&first, Leg::Stated, Outcome::Mismatch, 1.0, ""),
        done(&second, Leg::Stated, Outcome::Match, 1.0, ""),
        done(&first, Leg::Stated, Outcome::Match, 1.0, ""),
        done(&third, Leg::Stated, Outcome::NotSupported, 1.0, ""),
        done(&second, Leg::Stated, Outcome::NotSupported, 1.0, ""),
        done(&second, Leg::RobustnessOff, Outcome::NotSupported, 1.0, ""),
        done(&third, Leg::Stated, Outcome::Match, 1.0, ""),
        done(&third, Leg::Stated, Outcome::Timeout, 1.0, ""),
    ];
    fs::write(
        out.join(DONE_LIST),
        runs.iter().map(DoneRun::line).collect::<String>(),
    )
    .unwrap();
    assert_eq!(
        install(&out, &store, Backend::Vulkan).unwrap(),
        Installed {
            rows: 2,
            refusals: 1,
            build: None,
        }
    );
    let mut installed = store.read_candidates(Backend::Vulkan).unwrap();
    installed.sort_by(|left, right| left.case_id.cmp(&right.case_id));
    let mut expected = vec![fresh, after_refusal];
    expected.sort_by(|left, right| left.case_id.cmp(&right.case_id));
    assert_eq!(installed, expected);
    assert_eq!(store.read_refusals(Backend::Vulkan).unwrap(), [refused]);

    let mut moltenvk = row(&second, &stated, &[0; 4], &[0; 4]);
    moltenvk.backend = Backend::Moltenvk;
    fs::write(out.join(ROWS), line(&moltenvk)).unwrap();
    assert!(install(&out, &store, Backend::Vulkan)
        .unwrap_err()
        .contains("not a vulkan row"));
    // Without a record the runs are Vulkan's: MoltenVK installs none of them.
    assert!(install(&out, &store, Backend::Moltenvk)
        .unwrap_err()
        .contains("runs of the vulkan backend"));

    // A MoltenVK batch's record: its rows install as MoltenVK's, and Vulkan refuses them.
    let build = BatchBuild::current(
        BatchBackend::Moltenvk,
        METAL_ENV,
        ENV,
        None,
        DeviceType::Gpu,
    );
    fs::write(out.join(BUILD), serde_json::to_vec(&build).unwrap()).unwrap();
    fs::write(
        out.join(DONE_LIST),
        done(&second, Leg::Stated, Outcome::Match, 1.0, "").line(),
    )
    .unwrap();
    fs::write(out.join(REFUSALS), "").unwrap();
    assert!(install(&out, &store, Backend::Vulkan)
        .unwrap_err()
        .contains("runs of the moltenvk backend"));
    assert_eq!(
        install(&out, &store, Backend::Moltenvk).unwrap(),
        Installed {
            rows: 1,
            refusals: 0,
            build: Some(build),
        }
    );
    assert_eq!(
        store.read_candidates(Backend::Moltenvk).unwrap(),
        [moltenvk]
    );
}

/// The store line of a row or a refusal.
fn line<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap() + "\n"
}

/// An out directory belongs to one build and venue: a batch claims a fresh one, resumes its own,
/// and refuses one another build or venue started, or one whose finished runs predate the record
/// (their build is unknown), so no done-list of an older build counts as this build's.
#[test]
fn an_out_directory_resumes_only_its_own_build() {
    let scratch = ScratchDir::new("batch-build").unwrap();
    let config = BatchConfig {
        executor: PathBuf::new(),
        backend: BatchBackend::Vulkan,
        corpus: scratch.path().join("corpus"),
        out: scratch.path().join("out"),
        metal_environment_id: METAL_ENV.into(),
        environment_id: ENV.into(),
        restricted_to: None,
        device: BatchDevice::Cpu,
        jobs: 1,
        timeout: Duration::from_secs(1),
        gpu_wait_timeout: None,
        source_index: None,
    };
    fs::create_dir_all(&config.out).unwrap();
    BatchBuild::claim(&config).unwrap();
    let recorded = BatchBuild::read(&config.out).unwrap().unwrap();
    assert_eq!(recorded, BatchBuild::of(&config));
    assert_eq!(recorded.executor_abi, ENGINE_EXECUTOR_ABI);
    assert_eq!(recorded.product, TRANSLATOR_FINGERPRINT);
    assert_eq!(recorded.device_type, "cpu");
    fs::write(
        config.out.join(DONE_LIST),
        done(&case(1, "a"), Leg::Stated, Outcome::Match, 1.0, "").line(),
    )
    .unwrap();
    BatchBuild::claim(&config).unwrap();

    let other_venue = BatchConfig {
        environment_id: "elsewhere".into(),
        ..config.clone()
    };
    let error = BatchBuild::claim(&other_venue).unwrap_err();
    assert!(error.contains("another build or venue"), "{error}");
    let other_device = BatchConfig {
        device: BatchDevice::Gpu(TripwirePaths {
            current: scratch.path().join("hang"),
            legacy: scratch.path().join("legacy"),
        }),
        ..config.clone()
    };
    assert!(BatchBuild::claim(&other_device).is_err());
    let other_backend = BatchConfig {
        backend: BatchBackend::Moltenvk,
        ..config.clone()
    };
    assert!(BatchBuild::claim(&other_backend).is_err());
    let restricted = BatchConfig {
        restricted_to: Some("vk12-floor".into()),
        ..config.clone()
    };
    assert!(BatchBuild::claim(&restricted).is_err());
    // A restricted batch's record names its profile; an unrestricted one's has no field for it.
    assert_eq!(
        BatchBuild::of(&restricted).restricted_to.as_deref(),
        Some("vk12-floor")
    );
    assert!(!serde_json::to_string(&recorded)
        .unwrap()
        .contains("restricted_to"));
    // A record from before the backend was recorded is a Vulkan batch's.
    let mut before_backend = serde_json::to_value(&recorded).unwrap();
    before_backend.as_object_mut().unwrap().remove("backend");
    fs::write(
        config.out.join(BUILD),
        serde_json::to_vec(&before_backend).unwrap(),
    )
    .unwrap();
    assert_eq!(
        BatchBuild::read(&config.out).unwrap(),
        Some(recorded.clone())
    );
    BatchBuild::claim(&config).unwrap();
    assert!(recorded.is_this_build());
    let mut older = recorded.clone();
    older.executor_abi = "vulkan-engine-v1".into();
    fs::write(config.out.join(BUILD), serde_json::to_vec(&older).unwrap()).unwrap();
    let error = BatchBuild::claim(&config).unwrap_err();
    assert!(error.contains("another build or venue"), "{error}");
    assert!(!older.is_this_build());
    for other in [
        BatchBuild {
            product: "55".repeat(32),
            ..recorded.clone()
        },
        BatchBuild {
            semantics_version: recorded.semantics_version + 1,
            ..recorded.clone()
        },
        BatchBuild {
            oracle_abi: "metal-v0".into(),
            ..recorded.clone()
        },
    ] {
        assert!(!other.is_this_build(), "{other:?}");
    }
    // The venue is not the build: a record of another venue is still this build's.
    assert!(BatchBuild {
        environment_id: "elsewhere".into(),
        device_type: "gpu".into(),
        ..recorded.clone()
    }
    .is_this_build());

    fs::remove_file(config.out.join(BUILD)).unwrap();
    let error = BatchBuild::claim(&config).unwrap_err();
    assert!(error.contains("their build is unknown"), "{error}");
    assert_eq!(BatchBuild::read(&config.out).unwrap(), None);
}

/// A GPU batch runs one executor at a time on `--device-type gpu`, starts no run while the
/// tripwire exists, and stops after a run whose device wait timed out, whose device worker
/// crashed, or that outlived its timeout: that run is recorded, the batch returns an error, the
/// tripwire exists (written by the batch when the executor left none, an existing one kept), and
/// no later run starts.
#[cfg(unix)]
#[test]
fn a_gpu_batch_runs_one_at_a_time_and_stops_at_a_hang() {
    let scratch = ScratchDir::new("batch-gpu").unwrap();
    let corpus = scratch.path().join("corpus");
    let store = CorpusStore::new(&corpus);
    let (first, second) = (case(1, "a"), case(2, "b"));
    for case in [&first, &second] {
        store.put_case(case.clone()).unwrap();
    }
    let started = scratch.path().join("started");
    let executor = scratch.path().join("fake-corpus-vulkan");
    // The first case ends as `$BEHAVIOR` says; the second would match.
    let script = |behavior: &str| {
        format!(
            r#"#!/bin/sh
id= device=
while [ $# -gt 0 ]; do
  case "$1" in
    --case-id) id=$2; shift 2 ;;
    --device-type) device=$2; shift 2 ;;
    --robustness-off) shift ;;
    *) shift 2 ;;
  esac
done
[ "$device" = gpu ] || exit 9
echo "$id" >> '{started}'
[ "$id" = {first} ] || exit 9
{behavior}
"#,
            started = started.display(),
            first = first.case_id,
        )
    };
    let cases = [first.clone(), second.clone()];
    let runs = plan(&cases, &[Pass::Leg(Leg::Stated)], &[], None);
    let jobs = BatchConfig {
        jobs: 2,
        ..gpu_config(scratch.path(), executor.clone(), Duration::from_secs(60))
    };
    let error = run_batch(&jobs, &runs, |_| {}).unwrap_err();
    assert!(error.contains("one executor at a time"), "{error}");

    for (behavior, outcome, kind) in [
        (
            format!("echo 'device wait timed out' >&2; exit {HANG_EXIT_CODE}"),
            Outcome::WaitTimeout,
            HangKind::Timeout,
        ),
        (
            format!(
                "echo 'device_crash{{SIGSEGV}}'; exit {}",
                exit_status::DEVICE_CRASH
            ),
            Outcome::DeviceCrash,
            HangKind::Crashed,
        ),
        ("sleep 600".to_string(), Outcome::Timeout, HangKind::Timeout),
    ] {
        executable(&executor, &script(&behavior));
        let config = gpu_config(scratch.path(), executor.clone(), Duration::from_secs(2));
        let _ = fs::remove_dir_all(&config.out);
        let _ = fs::remove_file(&started);
        let BatchDevice::Gpu(tripwire) = &config.device else {
            unreachable!()
        };
        let _ = fs::remove_file(&tripwire.current);
        let error = run_batch(&config, &runs, |_| {}).unwrap_err();
        assert!(error.contains("the batch stops"), "{error}");
        assert!(error.contains("wrote the tripwire"), "{error}");
        assert_eq!(
            fs::read_to_string(&started).unwrap(),
            format!("{}\n", first.case_id),
            "{outcome:?}: a run started after the hang"
        );
        let listed = read_done_list(&config.out.join(DONE_LIST)).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].outcome, outcome);
        let record = crate::gpu::tripwire::read_record(&tripwire.current).unwrap();
        assert_eq!(record.kind, kind);
        assert_eq!(record.case_id.as_deref(), Some(first.case_id.as_str()));

        // While the tripwire exists, a resumed batch starts no run; an existing one is kept.
        let resumed = plan(&cases, &[Pass::Leg(Leg::Stated)], &listed, None);
        let error = run_batch(&config, &resumed, |_| {}).unwrap_err();
        assert!(error.contains("GPU hang tripwire"), "{error}");
        assert_eq!(
            fs::read_to_string(&started).unwrap(),
            format!("{}\n", first.case_id)
        );
        assert_eq!(read_done_list(&config.out.join(DONE_LIST)).unwrap(), listed);
        assert_eq!(
            crate::gpu::tripwire::read_record(&tripwire.current).unwrap(),
            record
        );
    }
}

/// A Metal run's exit and row decide its outcome together: only exit 0 with a qualified row
/// qualifies, only the reference-gap exit with a reference-gap row is a reference gap, and a signal
/// is a device crash (the Metal process holds its device itself).
#[test]
fn a_metal_outcome_reads_the_exit_and_the_row_together() {
    let qualified = || Some(MetalStatus::Qualified);
    let gap = || {
        Some(MetalStatus::ReferenceGap(ReferenceGap {
            apple_family: 10,
            message: "needs GPUFamily10".into(),
        }))
    };
    for (ended, row, outcome) in [
        (Ended::Code(0), qualified(), Outcome::Qualified),
        (Ended::Code(0), None, Outcome::Failed),
        (Ended::Code(0), gap(), Outcome::Failed),
        (Ended::Code(1), qualified(), Outcome::Failed),
        (Ended::Code(1), None, Outcome::Failed),
        (
            Ended::Code(REFERENCE_GAP_EXIT),
            gap(),
            Outcome::ReferenceGap,
        ),
        (Ended::Code(REFERENCE_GAP_EXIT), None, Outcome::Failed),
        (
            Ended::Code(REFERENCE_GAP_EXIT),
            qualified(),
            Outcome::Failed,
        ),
        (Ended::Code(exit_status::REFUSED), None, Outcome::Refused),
        (
            Ended::Code(exit_status::REFUSED),
            qualified(),
            Outcome::Failed,
        ),
        (Ended::Code(HANG_EXIT_CODE), None, Outcome::WaitTimeout),
        (
            Ended::Code(crate::gpu::compile::COMPILE_TIMEOUT_EXIT_CODE),
            None,
            Outcome::Failed,
        ),
        (Ended::Signal(libc::SIGSEGV), None, Outcome::DeviceCrash),
        (
            Ended::Signal(libc::SIGKILL),
            qualified(),
            Outcome::DeviceCrash,
        ),
        (Ended::TimedOut, qualified(), Outcome::Timeout),
    ] {
        assert_eq!(
            classify_metal(ended, row.clone()),
            outcome,
            "{ended:?} {row:?}"
        );
    }
    for outcome in [Outcome::Qualified, Outcome::ReferenceGap, Outcome::Refused] {
        assert!(!stops_a_gpu(outcome));
    }
    assert_eq!(Outcome::ReferenceGap.class(), Class::Refusal);
    assert_eq!(Outcome::parse("reference_gap"), Some(Outcome::ReferenceGap));
    assert_eq!(Pass::parse("metal"), Ok(Pass::Metal));
    let line = done(&case(1, "a"), Leg::Stated, Outcome::Qualified, 1.5, "");
    let metal_line = DoneRun {
        pass: Pass::Metal,
        ..line
    };
    let text = metal_line.line();
    assert_eq!(
        DoneRun::parse(text.strip_suffix('\n').unwrap()),
        Ok(metal_line)
    );
}

/// A Metal batch runs `corpus-metal` once per case on the GPU in a job store of its own (links to
/// the cases, reviews, private sources, and library modules; its own Metal shards; no candidate
/// shards), naming each case's AIR, its Metal environment, the wait bound, and the source index,
/// and nothing of a candidate run. It keeps the Metal row of each run that recorded one (a
/// qualification, or a reference gap) verbatim, logs every other run, never writes the corpus,
/// resumes with nothing left, reports by class, and installs those rows through the store's own
/// write, into a store and never from or into another backend's out directory.
#[cfg(unix)]
#[test]
fn a_metal_batch_qualifies_each_case_and_installs_its_rows() {
    let scratch = ScratchDir::new("batch-metal").unwrap();
    let corpus = scratch.path().join("corpus");
    let store = CorpusStore::new(&corpus);
    let (qualifying, refused, failing, gapped) =
        (case(1, "a"), case(2, "b"), case(3, "c"), case(4, "d"));
    for case in [&qualifying, &refused, &failing, &gapped] {
        store.put_case(case.clone()).unwrap();
    }
    fs::create_dir_all(corpus.join("reviews")).unwrap();
    for (directory, file) in [
        ("local/sources", "shard_000.jsonl"),
        ("local/library-modules", "shard_000.jsonl"),
    ] {
        fs::create_dir_all(corpus.join(directory)).unwrap();
        fs::write(corpus.join(directory).join(file), b"{}\n").unwrap();
    }
    let before = snapshot(&corpus);
    let row = metal(&qualifying, METAL_ENV, ORACLE_ABI, &[1, 0, 0, 0]);
    let gap_row = MetalObservation {
        status: MetalStatus::ReferenceGap(ReferenceGap {
            apple_family: 10,
            message: "needs GPUFamily10".into(),
        }),
        ..metal(&gapped, METAL_ENV, ORACLE_ABI, &[])
    };
    let index = scratch.path().join("corpus-index.sqlite");
    let executor = scratch.path().join("fake-corpus-metal");
    executable(
        &executor,
        &format!(
            r#"#!/bin/sh
store= id= air= env= device= wait= index=
while [ $# -gt 0 ]; do
  case "$1" in
    --corpus) store=$2; shift 2 ;;
    --case-id) id=$2; shift 2 ;;
    --air-sha256) air=$2; shift 2 ;;
    --environment-id) env=$2; shift 2 ;;
    --device-type) device=$2; shift 2 ;;
    --gpu-wait-timeout) wait=$2; shift 2 ;;
    --source-index) index=$2; shift 2 ;;
    *) echo "unexpected $1" >&2; exit 9 ;;
  esac
done
case "$id:$air" in
  {q}:{qa}|{r}:{ra}|{f}:{fa}|{g}:{ga}) ;;
  *) echo "the batch named AIR '$air' for $id" >&2; exit 9 ;;
esac
[ "$env" = {METAL_ENV} ] && [ "$device" = gpu ] && [ "$wait" = 30 ] || exit 9
[ "$index" = {index} ] || exit 9
[ -L "$store/cases" ] && [ -L "$store/reviews" ] && [ -L "$store/local/sources" ] || exit 9
[ -L "$store/local/library-modules" ] || exit 9
[ -d "$store/observations/metal" ] && [ ! -L "$store/observations/metal" ] || exit 9
[ ! -e "$store/observations/vulkan" ] && [ ! -e "$store/observations/refusals" ] || exit 9
case "$id" in
  {q}) printf '%s' '{row}' >> "$store/observations/metal/{qs}"
       echo "qualified $id output_sha256=x"; exit 0 ;;
  {r}) echo "execution_refused{{test}}"; exit {refused_exit} ;;
  {f}) echo "corpus-metal: Metal qualification is nondeterministic" >&2; exit 1 ;;
  {g}) printf '%s' '{gap_row}' >> "$store/observations/metal/{gs}"
       echo "reference_gap $id apple_family=10"; exit {gap_exit} ;;
esac
exit 9
"#,
            q = qualifying.case_id,
            qa = qualifying.air_sha256,
            qs = shard_name_for_hash(&qualifying.air_sha256).unwrap(),
            r = refused.case_id,
            ra = refused.air_sha256,
            f = failing.case_id,
            fa = failing.air_sha256,
            g = gapped.case_id,
            ga = gapped.air_sha256,
            gs = shard_name_for_hash(&gapped.air_sha256).unwrap(),
            row = line(&row),
            gap_row = line(&gap_row),
            gap_exit = REFERENCE_GAP_EXIT,
            index = index.display(),
            refused_exit = exit_status::REFUSED,
        ),
    );
    let config = BatchConfig {
        backend: BatchBackend::Metal,
        environment_id: METAL_ENV.into(),
        source_index: Some(index),
        ..gpu_config(scratch.path(), executor, Duration::from_secs(60))
    };
    let cases = metal::selected_cases(store.read_all_cases().unwrap(), &[]).unwrap();
    assert_eq!(cases.len(), 4);
    let error = metal::selected_cases(cases.clone(), &["0".repeat(64)]).unwrap_err();
    assert!(error.contains("no case for"), "{error}");
    let runs = plan(&cases, &[Pass::Metal], &[], None);
    assert_eq!(runs.len(), 4);
    let finished = run_batch(&config, &runs, |_| {}).unwrap();
    assert_eq!(finished.len(), 4);

    let listed = read_done_list(&config.out.join(DONE_LIST)).unwrap();
    let outcome_of = |case: &AuthoredCase| {
        let runs = listed
            .iter()
            .filter(|run| run.case_id == case.case_id && run.pass == Pass::Metal)
            .collect::<Vec<_>>();
        assert_eq!(runs.len(), 1);
        (runs[0].outcome, runs[0].detail.clone())
    };
    assert_eq!(outcome_of(&qualifying), (Outcome::Qualified, String::new()));
    assert_eq!(
        outcome_of(&refused),
        (Outcome::Refused, "execution_refused{test}".into())
    );
    assert_eq!(
        outcome_of(&failing),
        (
            Outcome::Failed,
            format!(
                "exit=1 without a row in {METAL_ENV}: corpus-metal: Metal qualification is \
                 nondeterministic"
            )
        )
    );
    assert_eq!(
        outcome_of(&gapped),
        (
            Outcome::ReferenceGap,
            format!("reference_gap {} apple_family=10", gapped.case_id)
        )
    );
    let mut rows = fs::read_to_string(config.out.join(ROWS))
        .unwrap()
        .lines()
        .map(|row| format!("{row}\n"))
        .collect::<Vec<_>>();
    rows.sort();
    let mut expected_rows = vec![line(&row), line(&gap_row)];
    expected_rows.sort();
    assert_eq!(rows, expected_rows);
    assert!(
        !config.out.join(REFUSALS).exists() || {
            fs::read_to_string(config.out.join(REFUSALS))
                .unwrap()
                .is_empty()
        }
    );
    assert_eq!(
        BatchBuild::read(&config.out).unwrap().unwrap().backend,
        BatchBackend::Metal
    );
    let mut logs = fs::read_dir(config.out.join(LOGS))
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect::<Vec<_>>();
    logs.sort();
    let mut expected_logs = [&refused, &failing, &gapped]
        .map(|case| format!("{}.metal.txt", case.case_id))
        .to_vec();
    expected_logs.sort();
    assert_eq!(logs, expected_logs);
    assert_eq!(snapshot(&corpus), before);
    assert!(plan(&cases, &[Pass::Metal], &listed, None).is_empty());

    let report = metal::report(&cases, &listed);
    assert_eq!(
        report[0],
        "batch-report pass=metal selected=4 ran=4 pending=0 qualified=1 refusal=2 failure=1 \
         reference_gap=1 refused=1 device_crash=0 wait_timeout=0 timeout=0 failed=1"
    );
    let refused_line = format!(
        "batch-run case={} pass=metal scope=private family=kernel.buffers outcome=refused \
         seconds={:.3} name=b detail=execution_refused{{test}}",
        refused.case_id,
        listed
            .iter()
            .find(|run| run.case_id == refused.case_id)
            .unwrap()
            .seconds
    );
    assert!(report.contains(&refused_line), "{report:#?}");
    assert!(!report
        .iter()
        .any(|line| line.contains(&format!("batch-run case={}", qualifying.case_id))));

    // Installed: the qualified run's row and the reference gap's, through the store's own write;
    // the store's Metal row of each case is the batch's.
    let target = CorpusStore::new(scratch.path().join("target"));
    for case in &cases {
        target.put_case(case.clone()).unwrap();
    }
    let installed = metal::install(&config.out, &target).unwrap();
    assert_eq!(installed.rows, 2);
    assert!(installed.build.as_ref().unwrap().is_this_build());
    let mut target_rows = target.read_metal().unwrap();
    target_rows.sort_by(|left, right| left.case_id.cmp(&right.case_id));
    let mut expected_rows = vec![row, gap_row];
    expected_rows.sort_by(|left, right| left.case_id.cmp(&right.case_id));
    assert_eq!(target_rows, expected_rows);
    let error = install(&config.out, &target, Backend::Vulkan).unwrap_err();
    assert!(error.contains("not vulkan"), "{error}");
    let vulkan_out = scratch.path().join("vulkan-out");
    fs::create_dir_all(&vulkan_out).unwrap();
    let error = metal::install(&vulkan_out, &target).unwrap_err();
    assert!(error.contains("not metal"), "{error}");
}

/// A Metal batch runs on the GPU only, under no profile, and only Metal passes; a candidate batch
/// never runs a Metal pass.
#[cfg(unix)]
#[test]
fn a_metal_batch_is_a_gpu_batch_of_metal_passes() {
    let scratch = ScratchDir::new("batch-metal-config").unwrap();
    let executor = scratch.path().join("never-run");
    let metal_config = BatchConfig {
        backend: BatchBackend::Metal,
        environment_id: METAL_ENV.into(),
        ..gpu_config(scratch.path(), executor.clone(), Duration::from_secs(60))
    };
    let cases = [case(1, "a")];
    let metal_runs = plan(&cases, &[Pass::Metal], &[], None);
    let leg_runs = plan(&cases, &[Pass::Leg(Leg::Stated)], &[], None);
    for (config, runs, expected) in [
        (
            BatchConfig {
                device: BatchDevice::Cpu,
                ..metal_config.clone()
            },
            &metal_runs,
            "Metal has no CPU device",
        ),
        (
            BatchConfig {
                restricted_to: Some("profile".into()),
                ..metal_config.clone()
            },
            &metal_runs,
            "restricts its device to no profile",
        ),
        (metal_config.clone(), &leg_runs, "Metal qualifications only"),
        (
            gpu_config(scratch.path(), executor.clone(), Duration::from_secs(60)),
            &metal_runs,
            "never a Metal qualification",
        ),
    ] {
        let error = run_batch(&config, runs, |_| {}).unwrap_err();
        assert!(error.contains(expected), "{error}");
    }
    assert!(!scratch.path().join("out").exists());
}

/// A Metal run that a signal ends stops the GPU batch like a candidate's device crash: it is
/// recorded, the tripwire names it as a Metal submission's crash, and no later run starts.
#[cfg(unix)]
#[test]
fn a_metal_run_ended_by_a_signal_stops_the_batch() {
    let scratch = ScratchDir::new("batch-metal-crash").unwrap();
    let corpus = scratch.path().join("corpus");
    let store = CorpusStore::new(&corpus);
    let (first, second) = (case(1, "a"), case(2, "b"));
    for case in [&first, &second] {
        store.put_case(case.clone()).unwrap();
    }
    let started = scratch.path().join("started");
    let executor = scratch.path().join("fake-corpus-metal");
    executable(
        &executor,
        &format!(
            r#"#!/bin/sh
id=
while [ $# -gt 0 ]; do
  case "$1" in
    --case-id) id=$2; shift 2 ;;
    *) shift 2 ;;
  esac
done
echo "$id" >> '{started}'
[ "$id" = {first} ] || exit 9
kill -SEGV $$
"#,
            started = started.display(),
            first = first.case_id,
        ),
    );
    let config = BatchConfig {
        backend: BatchBackend::Metal,
        environment_id: METAL_ENV.into(),
        ..gpu_config(scratch.path(), executor, Duration::from_secs(60))
    };
    let cases = [first.clone(), second];
    let runs = plan(&cases, &[Pass::Metal], &[], None);
    let error = run_batch(&config, &runs, |_| {}).unwrap_err();
    assert!(error.contains("the batch stops"), "{error}");
    assert!(error.contains("wrote the tripwire"), "{error}");
    assert_eq!(
        fs::read_to_string(&started).unwrap(),
        format!("{}\n", first.case_id)
    );
    let listed = read_done_list(&config.out.join(DONE_LIST)).unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].outcome, Outcome::DeviceCrash);
    assert!(
        listed[0]
            .detail
            .starts_with(&format!("signal={}", libc::SIGSEGV)),
        "{}",
        listed[0].detail
    );
    let BatchDevice::Gpu(tripwire) = &config.device else {
        unreachable!()
    };
    let record = crate::gpu::tripwire::read_record(&tripwire.current).unwrap();
    assert_eq!(record.kind, HangKind::Crashed);
    assert_eq!(record.backend, SubmissionBackend::Metal);
    assert_eq!(record.case_id.as_deref(), Some(first.case_id.as_str()));
}
