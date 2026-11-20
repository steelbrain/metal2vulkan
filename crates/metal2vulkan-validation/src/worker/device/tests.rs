//! The device worker's boundary, with shells standing in for the worker: each reads its input as a
//! worker does, then reports, exits, or dies by a signal.

use super::*;
use crate::check::admission::{ExecutionRefused, NotSupported};
use crate::check::loop_free::CycleEvidence;
use crate::engine::{Outcome, UnsupportedReason};
use crate::requirement::ToolingRequirement;
use std::cell::RefCell;
use std::io;

fn input(device: DeviceType) -> DeviceInput {
    DeviceInput {
        root: PathBuf::from("/nonexistent/device-worker-corpus"),
        case: crate::case::attachmentless_fragment_test_case("00".repeat(32), "frag".into()),
        device,
        backend: Backend::Vulkan,
        input_sha256: "11".repeat(32),
        wait_timeout_ms: 7_000,
        restricted_to: None,
        leg: Leg::Stated,
        generated_commands: true,
        venue: None,
        source_index: None,
    }
}

fn restricted(device: DeviceType, profile: &str) -> DeviceInput {
    DeviceInput {
        restricted_to: Some(profile.into()),
        ..input(device)
    }
}

/// [`supervise`] of a worker that does not hang, relaying into nothing.
fn ran(
    command: Command,
    input: &DeviceInput,
    write_tripwire: impl FnOnce(&HangRecord) -> Result<(), String>,
) -> Result<DeviceRun, ExecutorError> {
    match supervise(command, input, write_tripwire, io::sink()) {
        Ok(outcome) => outcome.result,
        Err(HangExit) => panic!("the worker left through the hang protocol"),
    }
}

/// A stderr that logs each write into a shared record of events, then fails as a closed pipe
/// does.
struct ClosedPipe<'a>(&'a RefCell<Vec<String>>);

impl Write for ClosedPipe<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0
            .borrow_mut()
            .push(format!("relay {}", String::from_utf8_lossy(bytes)));
        Err(io::ErrorKind::BrokenPipe.into())
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::ErrorKind::BrokenPipe.into())
    }
}

fn shell(script: &str) -> Command {
    let mut command = Command::new("sh");
    command.args(["-c", script]);
    command
}

fn run() -> DeviceRun {
    run_restricted_to(None)
}

fn run_restricted_to(restricted_to: Option<String>) -> DeviceRun {
    DeviceRun {
        timing: crate::observation::test_timing(),
        output: vec![0, 1, 2, 0xff],
        spv_sha256: Some("ab".repeat(32)),
        translator_fingerprint: "cd".repeat(32),
        environment: serde_json::json!({}),
        comparison: ComparisonClass::Exact,
        census: ExecutedCensus {
            out_of_bounds: None,
            rate_map_pass: None,
            device: None,
            product: TRANSLATOR_FINGERPRINT.into(),
            restricted_to,
            leg: Some(Leg::Stated),
            modules: Vec::new(),
            artifact: None,
        },
    }
}

/// A run the worker reports carries what it cost in time, and the parent cannot read one that does
/// not. [`DeviceRun::timing`] is not optional, so a figure is required of every reported run by
/// type: the Vulkan side records a cost for every case it runs rather than for a selection, and a
/// run whose report lost its figure is a decode failure rather than a row with a hole in it.
///
/// The figure itself is measured noise and nothing compares it: two reports that differ only here
/// are the same run.
#[test]
fn a_reported_run_carries_what_it_cost() {
    let reported = serde_json::to_value(run()).unwrap();
    assert!(reported.get("timing").is_some());
    let mut without = reported.clone();
    without.as_object_mut().unwrap().remove("timing");
    let refused = serde_json::from_value::<DeviceRun>(without).unwrap_err();
    assert!(refused.to_string().contains("timing"), "{refused}");
    assert_eq!(
        serde_json::from_value::<DeviceRun>(reported).unwrap(),
        run()
    );
    let mut slower = run();
    slower.timing = crate::observation::CaseTiming {
        runs: 1,
        total_micros: 9_000_000,
        least_micros: 9_000_000,
    };
    assert_eq!(slower, run());

    // A refused attempt is reported the same way: an opened device implies a figure, because the
    // worker reads its clock around everything after the case was prepared.
    let opened = OpenedDevice {
        environment: serde_json::json!({}),
        translator_fingerprint: "cd".repeat(32),
        artifact: None,
        timing: crate::observation::test_timing(),
    };
    let reported = serde_json::to_value(&opened).unwrap();
    assert!(reported.get("timing").is_some());
    let mut without = reported.clone();
    without.as_object_mut().unwrap().remove("timing");
    let refused = serde_json::from_value::<OpenedDevice>(without).unwrap_err();
    assert!(refused.to_string().contains("timing"), "{refused}");
    assert_eq!(
        serde_json::from_value::<OpenedDevice>(reported).unwrap(),
        opened
    );
}

/// What this build's worker reports for `result`.
fn output(result: Result<DeviceRun, ExecutorError>) -> DeviceOutput {
    DeviceOutput {
        product_fingerprint: TRANSLATOR_FINGERPRINT.into(),
        validation_fingerprint: VALIDATION_FINGERPRINT.into(),
        generated_commands: true,
        result,
        refused_on: None,
    }
}

/// A worker that reads its input into `seen` (in `scratch`) and reports `output`.
fn reporting(scratch: &ScratchDir, output: &DeviceOutput) -> Command {
    let report = scratch.path().join("report.json");
    let envelope = serde_json::json!({"allocator_peak_bytes": null, "result": output});
    fs::write(&report, serde_json::to_vec(&envelope).unwrap()).unwrap();
    shell(&format!(
        "cat > '{}'; cat '{}'",
        scratch.path().join("seen.json").display(),
        report.display()
    ))
}

/// The bug class: a device worker's report is the last line of its stdout, as every worker's is,
/// and a line a library in the worker wrote to stdout before it (the validation layer names its
/// log file there) is not the report.
#[cfg(unix)]
#[test]
fn a_worker_s_report_is_the_last_line_of_its_stdout() {
    let scratch = ScratchDir::new("device-worker-test").unwrap();
    let report = scratch.path().join("report.json");
    let envelope = serde_json::json!({"allocator_peak_bytes": null, "result": output(Ok(run()))});
    fs::write(&report, serde_json::to_vec(&envelope).unwrap()).unwrap();
    let worker = shell(&format!(
        "cat >/dev/null; echo 'Validation Layer Info - Logging validation error to x.log'; \
         echo; cat '{}'; echo",
        report.display()
    ));
    assert_eq!(ran(worker, &input(DeviceType::Cpu), no_tripwire), Ok(run()));
}

fn no_tripwire(record: &HangRecord) -> Result<(), String> {
    panic!("a CPU worker wrote the host tripwire: {record:?}")
}

/// The bug class: a device worker a signal ends is the parent's typed outcome, never the parent's
/// death, and the validation messages the worker echoed before it died are in it.
#[cfg(unix)]
#[test]
fn a_worker_a_signal_ends_is_a_typed_crash_with_what_it_wrote() {
    let worker = shell(
        "cat >/dev/null; \
         echo 'VK_LAYER_KHRONOS_validation: error VUID-test: reported before the fault' >&2; \
         kill -KILL $$",
    );
    let error = ran(worker, &input(DeviceType::Cpu), no_tripwire).unwrap_err();
    let ExecutorError::DeviceCrash(crash) = error else {
        panic!("{error:?}")
    };
    assert_eq!(crash.signal, 9);
    assert_eq!(crash.device, DeviceType::Cpu);
    assert_eq!(crash.outcome(), "device_crash{SIGKILL}");
    assert!(
        crash
            .detail
            .contains("VUID-test: reported before the fault"),
        "{}",
        crash.detail
    );
    assert!(!crash.detail.contains("tripwire"), "{}", crash.detail);
    assert!(
        crash
            .to_string()
            .starts_with("device_crash{SIGKILL}: the cpu device worker died by signal 9"),
        "{crash}"
    );
}

/// A GPU worker's crash writes the host tripwire before the parent writes anything, since
/// whatever it submitted may still be running; and a stderr closed under the parent (`2>&1 |
/// head`) loses the worker's lines without taking the parent or the crash with it.
#[cfg(unix)]
#[test]
fn a_gpu_worker_crash_writes_the_tripwire_before_any_output() {
    let events = RefCell::new(Vec::new());
    let input = input(DeviceType::Gpu);
    let worker = shell("cat >/dev/null; echo 'submitted' >&2; kill -KILL $$");
    let written = RefCell::new(None);
    let supervised = supervise(
        worker,
        &input,
        |record| {
            events.borrow_mut().push("tripwire".to_string());
            *written.borrow_mut() = Some(record.clone());
            Ok(())
        },
        ClosedPipe(&events),
    );
    assert_eq!(
        events.into_inner(),
        ["tripwire".to_string(), "relay submitted\n".to_string()]
    );
    let record = written
        .into_inner()
        .expect("the GPU crash wrote no tripwire");
    assert_eq!(record.kind, HangKind::Crashed);
    assert_eq!(record.backend, SubmissionBackend::VulkanGpu);
    assert_eq!(record.case_id.as_deref(), Some(input.case.case_id.as_str()));
    assert!(
        record.detail.ends_with("died by signal 9"),
        "{}",
        record.detail
    );
    match supervised {
        Ok(DeviceOutcome {
            result: Err(ExecutorError::DeviceCrash(crash)),
            refused_on: None,
        }) => {
            assert_eq!(crash.device, DeviceType::Gpu);
            assert!(crash
                .detail
                .starts_with("the host GPU tripwire was written"));
            assert!(crash.detail.ends_with("submitted"), "{}", crash.detail);
        }
        other => panic!("{other:?}"),
    }
}

/// A worker that leaves through the hang protocol comes back for [`execute`] to leave with the
/// protocol's status, its stderr relayed; a report or an exit that is not valid UTF-8 keeps
/// every byte it can read.
#[cfg(unix)]
#[test]
fn a_hang_comes_back_for_the_tool_to_leave_with_its_stderr() {
    let events = RefCell::new(Vec::new());
    let worker = shell(&format!(
        "cat >/dev/null; printf 'm2v-gpu: hang \\377\\n' >&2; exit {HANG_EXIT_CODE}"
    ));
    let supervised = supervise(
        worker,
        &input(DeviceType::Cpu),
        no_tripwire,
        ClosedPipe(&events),
    );
    assert_eq!(supervised, Err(HangExit));
    assert_eq!(events.into_inner(), ["relay m2v-gpu: hang \u{fffd}\n"]);
    let exited = shell("cat >/dev/null; printf 'no device \\377 here' >&2; exit 1");
    match ran(exited, &input(DeviceType::Cpu), no_tripwire) {
        Err(ExecutorError::Failed(error)) => {
            assert!(error.ends_with("no device \u{fffd} here"), "{error}")
        }
        other => panic!("{other:?}"),
    }
}

/// The worker reads exactly the parent's input, and the run and every typed refusal or failure
/// it reports comes back unchanged.
#[cfg(unix)]
#[test]
fn the_input_and_every_typed_result_cross_the_boundary() {
    let results: Vec<Result<DeviceRun, ExecutorError>> = vec![
        Ok(run()),
        Err(ExecutorError::Failed("no device".into())),
        Err(ExecutionRefused::NotFuelBounded(CycleEvidence::CfgCycle {
            function: "main".into(),
        })
        .into()),
        Err(NotSupported::Product {
            outcome: Outcome::Unsupported {
                reason: UnsupportedReason::NotRepresentable,
                feature: "RayTracing".into(),
            },
            detail: "an instance mask of 32 bits".into(),
            at_bind: false,
        }
        .into()),
        Err(NotSupported::Bind("a range past the limit".into()).into()),
        Err(NotSupported::NotEnabled(vec!["rayQuery".into()]).into()),
        Err(NotSupported::NotImplemented("linked intersection functions".into()).into()),
        Err(
            NotSupported::Requirements([ToolingRequirement::RenderAccelerationStructure].into())
                .into(),
        ),
        Err(ExecutorError::DeviceCrash(DeviceCrash {
            signal: 11,
            device: DeviceType::Gpu,
            detail: "a nested worker's".into(),
        })),
        Err(ExecutorError::VenueMismatch(
            crate::check::admission::VenueMismatch {
                environment_id: "venue".into(),
                profile: "toy".into(),
                differences: vec!["identity driver: 26.2.4 (26.2.5)".into()],
            },
        )),
    ];
    for input in [
        input(DeviceType::Cpu),
        restricted(DeviceType::Cpu, "gate-lavapipe"),
        DeviceInput {
            leg: Leg::RobustnessOff,
            ..restricted(DeviceType::Cpu, "vk12-floor")
        },
        DeviceInput {
            generated_commands: false,
            ..restricted(DeviceType::Cpu, "gate-lavapipe")
        },
    ] {
        let mut own = run_restricted_to(input.restricted_to.clone());
        own.census.leg = Some(input.leg);
        let results = results.iter().cloned().chain([Ok(own)]);
        for result in results {
            let scratch = ScratchDir::new("device-worker-test").unwrap();
            let report = DeviceOutput {
                generated_commands: input.generated_commands,
                ..output(result.clone())
            };
            let worker = reporting(&scratch, &report);
            let expected = match result {
                // An unrestricted run is the restricted parent's refusal, a stated one the
                // robustness-off parent's ([`reported`]).
                Ok(run)
                    if run.census.restricted_to != input.restricted_to
                        || run.census.leg != Some(input.leg) =>
                {
                    continue
                }
                result => result,
            };
            assert_eq!(ran(worker, &input, no_tripwire), expected);
            let seen: DeviceInput =
                serde_json::from_slice(&fs::read(scratch.path().join("seen.json")).unwrap())
                    .unwrap();
            assert_eq!(seen, input);
        }
    }
}

/// A worker of another build ran another translator or another executor: its run is not this
/// tool's, and would be stored under this tool's executor ABI.
#[cfg(unix)]
#[test]
fn a_worker_of_another_build_is_refused() {
    let builds = [
        DeviceOutput {
            product_fingerprint: "another-product".into(),
            ..output(Ok(run()))
        },
        DeviceOutput {
            validation_fingerprint: "another-executor".into(),
            ..output(Ok(run()))
        },
    ];
    for build in builds {
        let scratch = ScratchDir::new("device-worker-test").unwrap();
        let worker = reporting(&scratch, &build);
        match ran(worker, &input(DeviceType::Cpu), no_tripwire) {
            Err(ExecutorError::Failed(error)) => {
                assert!(error.starts_with("the device worker is build"), "{error}");
                assert!(error.contains("rebuild and rerun"), "{error}");
            }
            other => panic!("{other:?}"),
        }
    }
}

/// A worker whose census names another restriction than its parent asked for ran on another
/// device than the row would say: a native row stored where P was asked, or the reverse.
#[cfg(unix)]
#[test]
fn a_worker_run_under_another_restriction_is_refused() {
    let runs = [
        (restricted(DeviceType::Cpu, "gate-lavapipe"), None),
        (input(DeviceType::Cpu), Some("gate-lavapipe")),
        (
            restricted(DeviceType::Cpu, "gate-lavapipe"),
            Some("vk12-floor"),
        ),
    ];
    for (input, census) in runs {
        let scratch = ScratchDir::new("device-worker-test").unwrap();
        let worker = reporting(
            &scratch,
            &output(Ok(run_restricted_to(census.map(String::from)))),
        );
        match ran(worker, &input, no_tripwire) {
            Err(ExecutorError::Failed(error)) => {
                assert!(error.starts_with("the device worker ran "), "{error}");
                assert!(error.contains("not this tool's evidence"), "{error}");
            }
            other => panic!("{other:?}"),
        }
    }
}

/// A worker whose census names another robustness leg than its parent asked for compiled under
/// other containment encodings than the row's slot would say: a stated run stored as the
/// robustness-off leg, or the reverse; a census of no leg is no leg asked for.
#[cfg(unix)]
#[test]
fn a_worker_run_in_another_leg_is_refused() {
    let runs = [
        (Leg::Stated, Some(Leg::RobustnessOff)),
        (Leg::RobustnessOff, Some(Leg::Stated)),
        (Leg::Stated, None),
    ];
    for (asked, ran_in) in runs {
        let scratch = ScratchDir::new("device-worker-test").unwrap();
        let mut reported = run();
        reported.census.leg = ran_in;
        let worker = reporting(&scratch, &output(Ok(reported)));
        let parent = DeviceInput {
            leg: asked,
            ..input(DeviceType::Cpu)
        };
        match ran(worker, &parent, no_tripwire) {
            Err(ExecutorError::Failed(error)) => {
                assert!(
                    error.starts_with("the device worker ran in robustness leg"),
                    "{error}"
                );
                assert!(error.contains("not this tool's evidence"), "{error}");
            }
            other => panic!("{other:?}"),
        }
    }
}

/// A venue's worker holds its device to the venue's profile and records the profile in its run
/// and in the device it refused a case on: one that recorded another profile, none where the parent
/// named a venue, or one where the parent named none, is not this tool's evidence.
#[cfg(unix)]
#[test]
fn a_worker_run_under_another_venue_profile_is_refused() {
    let venue = |profile: &str| DeviceInput {
        venue: Some(VenueRun {
            environment_id: "venue".into(),
            profile: profile.into(),
            allowed: Vec::new(),
        }),
        ..input(DeviceType::Gpu)
    };
    let recorded = |profile: Option<&str>| {
        let mut environment = serde_json::json!({"robustness": {"leg": "stated"}});
        if let Some(profile) = profile {
            environment[crate::c3::VENUE_PROFILE] =
                serde_json::json!({"profile": profile, "sha256": "00", "fingerprint": "11"});
        }
        environment
    };
    let refusal: Result<DeviceRun, ExecutorError> =
        Err(NotSupported::NotImplemented("a family".into()).into());
    let outcome = |report: &DeviceOutput, input: &DeviceInput| {
        let scratch = ScratchDir::new("device-worker-test").unwrap();
        let worker = reporting(&scratch, report);
        match supervise(worker, input, no_tripwire, io::sink()) {
            Ok(outcome) => outcome,
            Err(HangExit) => panic!("the worker left through the hang protocol"),
        }
    };
    for (asked, profile, holds) in [
        (venue("toy"), Some("toy"), true),
        (input(DeviceType::Gpu), None, true),
        (venue("toy"), None, false),
        (venue("toy"), Some("other"), false),
        (input(DeviceType::Gpu), Some("toy"), false),
    ] {
        let mut reported = run();
        reported.environment = recorded(profile);
        let ran = outcome(&output(Ok(reported.clone())), &asked).result;
        let refused = outcome(
            &DeviceOutput {
                refused_on: Some(OpenedDevice {
                    timing: crate::observation::test_timing(),
                    environment: recorded(profile),
                    translator_fingerprint: "cd".repeat(32),
                    artifact: None,
                }),
                ..output(refusal.clone())
            },
            &asked,
        );
        if holds {
            assert_eq!(ran, Ok(reported), "{profile:?}");
            assert_eq!(refused.result, refusal);
            assert!(refused.refused_on.is_some());
            continue;
        }
        for result in [ran, refused.result] {
            let Err(ExecutorError::Failed(error)) = result else {
                panic!("{profile:?}: {result:?}")
            };
            assert!(
                error.starts_with("the device worker recorded venue profile"),
                "{error}"
            );
            assert!(error.contains("not this tool's evidence"), "{error}");
        }
        assert_eq!(refused.refused_on, None);
    }
}

/// A worker that exits without a report fails the run with what it wrote; one that reports
/// nothing readable fails it too.
#[cfg(unix)]
#[test]
fn a_worker_without_a_report_is_a_failure_with_its_stderr() {
    let exited = shell("cat >/dev/null; echo 'cannot open the device' >&2; exit 1");
    match ran(exited, &input(DeviceType::Cpu), no_tripwire) {
        Err(ExecutorError::Failed(error)) => {
            assert!(error.contains("cannot open the device"), "{error}")
        }
        other => panic!("{other:?}"),
    }
    let garbled = shell("cat >/dev/null; echo 'not a report'");
    match ran(garbled, &input(DeviceType::Cpu), no_tripwire) {
        Err(ExecutorError::Failed(error)) => {
            assert!(
                error.starts_with("decode the device worker's report"),
                "{error}"
            )
        }
        other => panic!("{other:?}"),
    }
}

/// Exit statuses: 0 reported, the hang protocol's status a hang, a signal a crash, anything
/// else an exit.
#[cfg(unix)]
#[test]
fn exit_statuses_classify() {
    use std::os::unix::process::ExitStatusExt as _;
    assert_eq!(Ended::of(ExitStatus::from_raw(0)), Ended::Reported);
    assert_eq!(
        Ended::of(ExitStatus::from_raw(HANG_EXIT_CODE << 8)),
        Ended::Hang
    );
    assert_eq!(Ended::of(ExitStatus::from_raw(11)), Ended::Signal(11));
    assert_eq!(Ended::of(ExitStatus::from_raw(1 << 8)), Ended::Exited);
}

#[test]
fn a_crash_names_its_signal() {
    let crash = |signal| DeviceCrash {
        signal,
        device: DeviceType::Cpu,
        detail: String::new(),
    };
    assert_eq!(crash(11).outcome(), "device_crash{SIGSEGV}");
    assert_eq!(crash(6).outcome(), "device_crash{SIGABRT}");
    assert_eq!(crash(4).outcome(), "device_crash{SIGILL}");
    assert_eq!(crash(99).outcome(), "device_crash{signal99}");
}

/// A worker that opened its devices on the other generated-commands path than its parent asked
/// for ran the other device replay: its run, or its refusal, is not the parent's evidence, and a
/// run without the extension would be stored only under the default.
#[cfg(unix)]
#[test]
fn a_worker_run_on_another_generated_commands_path_is_refused() {
    let refused: Result<DeviceRun, ExecutorError> =
        Err(NotSupported::NotEnabled(vec!["deviceGeneratedCommands".into()]).into());
    for asked in [true, false] {
        let input = DeviceInput {
            generated_commands: asked,
            ..input(DeviceType::Cpu)
        };
        for result in [Ok(run()), refused.clone()] {
            let scratch = ScratchDir::new("device-worker-test").unwrap();
            let report = DeviceOutput {
                generated_commands: !asked,
                ..output(result)
            };
            let worker = reporting(&scratch, &report);
            match ran(worker, &input, no_tripwire) {
                Err(ExecutorError::Failed(error)) => {
                    assert!(
                        error.starts_with("the device worker opened its devices"),
                        "{error}"
                    );
                    assert!(error.contains("not this tool's evidence"), "{error}");
                }
                other => panic!("{other:?}"),
            }
        }
    }
}

/// A typed refusal after the device opened crosses the boundary with that device's facts; a device
/// named beside anything but a typed refusal, or one opened in another leg or under another
/// restriction than the parent asked for, is not this tool's evidence.
#[cfg(unix)]
#[test]
fn a_refusal_names_the_device_it_was_refused_on() {
    let refusal: Result<DeviceRun, ExecutorError> = Err(NotSupported::Product {
        outcome: Outcome::DeviceCapabilityMissing {
            feature: "FramebufferFetch".into(),
            capability: "Feature(RasterizationOrderColorAttachmentAccess)".into(),
        },
        detail: "the device has no ordered attachment access".into(),
        at_bind: false,
    }
    .into());
    let opened = |leg: &str, profile: Option<&str>| OpenedDevice {
        timing: crate::observation::test_timing(),
        environment: serde_json::json!({
            "device": "AMD Radeon 780M Graphics (RADV PHOENIX)",
            "robustness": {"leg": leg},
            "profiles_layer": profile.map(|profile| serde_json::json!({"profile": profile})),
        }),
        translator_fingerprint: "cd".repeat(32),
        artifact: None,
    };
    let outcome = |report: &DeviceOutput, input: &DeviceInput| {
        let scratch = ScratchDir::new("device-worker-test").unwrap();
        let worker = reporting(&scratch, report);
        match supervise(worker, input, no_tripwire, io::sink()) {
            Ok(outcome) => outcome,
            Err(HangExit) => panic!("the worker left through the hang protocol"),
        }
    };
    let stated = input(DeviceType::Gpu);
    let named = DeviceOutput {
        refused_on: Some(opened("stated", None)),
        ..output(refusal.clone())
    };
    assert_eq!(
        outcome(&named, &stated),
        DeviceOutcome {
            result: refusal.clone(),
            refused_on: Some(opened("stated", None)),
        }
    );
    // Restricted and in the robustness-off leg, as the parent asked.
    let restricted_off = DeviceInput {
        leg: Leg::RobustnessOff,
        ..restricted(DeviceType::Gpu, "amd-rdna3-radv")
    };
    let named_off = DeviceOutput {
        refused_on: Some(opened("robustness_off", Some("amd-rdna3-radv"))),
        ..output(refusal.clone())
    };
    assert_eq!(
        outcome(&named_off, &restricted_off).refused_on,
        Some(opened("robustness_off", Some("amd-rdna3-radv")))
    );
    // Another leg, or another restriction, than the parent's.
    for (report, input) in [(&named, &restricted_off), (&named_off, &stated)] {
        let refused = outcome(report, input);
        assert_eq!(refused.refused_on, None);
        let Err(ExecutorError::Failed(error)) = refused.result else {
            panic!("{refused:?}")
        };
        assert!(
            error.contains("its refusal is not this tool's evidence"),
            "{error}"
        );
    }
    // A device beside a run, or beside an outcome that is no typed refusal.
    for result in [Ok(run()), Err(ExecutorError::Failed("no device".into()))] {
        let report = DeviceOutput {
            refused_on: Some(opened("stated", None)),
            ..output(result)
        };
        let refused = outcome(&report, &stated);
        assert_eq!(refused.refused_on, None);
        let Err(ExecutorError::Failed(error)) = refused.result else {
            panic!("{refused:?}")
        };
        assert!(error.contains("no typed refusal"), "{error}");
    }
    // Without a device, a refusal is only a refusal.
    assert_eq!(
        outcome(&output(refusal.clone()), &stated),
        DeviceOutcome {
            result: refusal,
            refused_on: None,
        }
    );
}
