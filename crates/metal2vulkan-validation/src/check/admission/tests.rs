use super::*;
use crate::observation::Backend;
use crate::source::SourceRow;
use crate::ScratchDir;

const METADATA: &str = r#"
!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3}
!3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.write", !"air.address_space", i32 1, !"air.arg_type_name", !"uint"}
"#;

const LOOP_FREE: &str = r#"
define void @main(ptr addrspace(1) %output) {
entry:
  store i32 42, ptr addrspace(1) %output, align 4
  ret void
}
"#;

const CYCLIC: &str = r#"
define void @main(ptr addrspace(1) %output) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %next = add i32 %i, 1
  %again = icmp ult i32 %next, 4
  br i1 %again, label %loop, label %done
done:
  store i32 %next, ptr addrspace(1) %output, align 4
  ret void
}
"#;

const HOST_BACKEND: Backend = Backend::Vulkan;

/// A scratch corpus holding one installed case over `body`.
fn corpus(body: &str, fuel: Option<u32>) -> (ScratchDir, AuthoredCase) {
    let scratch = ScratchDir::new("admission").unwrap();
    let air_ll = format!("{body}{METADATA}");
    let source = SourceRow {
        air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
        stage: "Kernel".into(),
        entry: "main".into(),
        air_ll,
        blob_b64: None,
        lib_sha256s: vec!["11".repeat(32)],
        label: "local/admission.ll".into(),
    };
    crate::source::write_source_shards(scratch.path(), [source.clone()]).unwrap();
    let cyclic =
        !super::super::loop_free::classify_text_module("main", &source.air_ll).is_loop_free();
    let mut case: AuthoredCase = serde_json::from_value(serde_json::json!({
        "air_sha256": source.air_sha256,
        "case_id": "pending",
        "name": "admission",
        "entry": "main",
        "stage": "kernel",
        "buffers": [{"binding": 0, "role": "output", "initial_bytes_b64": "q6urqw=="}],
        "dispatch": {"grid": [1, 1, 1], "threads_per_threadgroup": [1, 1, 1]},
        "output": {"kind": "buffer", "binding": 0, "offset": 0, "length": 4},
        "compare": {"kind": "exact"},
        "execution_safety": if cyclic { "authored_bounded" } else { "loop_free" },
        "rationale": "The only loop runs exactly four iterations.",
    }))
    .unwrap();
    case.fuel = fuel;
    case.case_id = case.computed_case_id().unwrap();
    crate::store::CorpusStore::new(scratch.path())
        .put_case(case.clone())
        .unwrap();
    (scratch, case)
}

/// A case whose only loop is behind recursion: cyclic, and never fuel-bounded.
const RECURSIVE: &str = r#"
define internal void @down(ptr addrspace(1) %output, i32 %n) {
entry:
  %stop = icmp eq i32 %n, 0
  br i1 %stop, label %done, label %again
again:
  %m = sub i32 %n, 1
  call void @down(ptr addrspace(1) %output, i32 %m)
  br label %done
done:
  ret void
}

define void @main(ptr addrspace(1) %output) {
entry:
  call void @down(ptr addrspace(1) %output, i32 4)
  store i32 4, ptr addrspace(1) %output, align 4
  ret void
}
"#;

#[test]
fn the_gpu_rules() {
    let evidence = CycleEvidence::CfgCycle {
        function: "main".into(),
    };
    let cyclic = LoopClass::Cyclic(evidence.clone());
    let covered = FuelScope::Covered;
    let uncovered = FuelScope::Uncovered(evidence);
    assert_eq!(
        refusal(
            &LoopClass::LoopFree,
            &covered,
            ExecutionRequest::MetalOracle
        ),
        None
    );
    // The Metal oracle runs a cyclic case fuel-instrumented when fuel bounds every cycle.
    assert_eq!(
        refusal(&cyclic, &covered, ExecutionRequest::MetalOracle),
        None
    );
    assert!(matches!(
        refusal(&cyclic, &uncovered, ExecutionRequest::MetalOracle),
        Some(ExecutionRefused::NotFuelBounded(_))
    ));
    assert_eq!(
        ExecutionRefused::NotFuelBounded(CycleEvidence::CfgCycle {
            function: "main".into()
        })
        .outcome(),
        "execution_refused{not_fuel_bounded}"
    );
}

#[test]
fn the_metal_oracle_refuses_a_cycle_fuel_cannot_bound_before_any_device_opens() {
    let (root, case) = corpus(RECURSIVE, None);
    let error = crate::metal::qualify(root.path(), case, "admission-oracle").unwrap_err();
    assert!(
        matches!(
            &error,
            ExecutorError::Refused(ExecutionRefused::NotFuelBounded(CycleEvidence::CallCycle { function }))
                if function == "down"
        ),
        "{error}"
    );
    assert!(error
        .to_string()
        .starts_with("execution_refused{not_fuel_bounded}"));

    // A loop fuel bounds, with or without a declared fuel, and a loop-free case pass admission.
    // They are admitted, not qualified: qualification would open the Metal device.
    for (body, fuel) in [(CYCLIC, None), (CYCLIC, Some(64)), (LOOP_FREE, None)] {
        let (root, case) = corpus(body, fuel);
        if let Err(refused) = admit(root.path(), case, ExecutionRequest::MetalOracle) {
            panic!("the case must not be refused: {refused}");
        }
    }
    let (root, case) = corpus(CYCLIC, None);
    let admitted = admit(root.path(), case, ExecutionRequest::MetalOracle).unwrap();
    assert!(!admitted.checked().loop_class().is_loop_free());
    assert!(admitted.checked().fuel_scope().is_covered());
}

/// What a candidate reports for `case` without current Metal evidence in `admission-oracle`.
fn needs_metal(case: &AuthoredCase, gap: MetalGap) -> ExecutorError {
    ExecutorError::NeedsMetal(NeedsMetal {
        case_id: case.case_id.clone(),
        environment_id: "admission-oracle".into(),
        gap,
    })
}

/// A Metal row an earlier oracle recorded is never compared against: the candidate reports the case
/// as needing Metal qualification, typed, naming the recorded and the current oracle, and before
/// any device opens.
#[test]
fn a_metal_row_from_an_earlier_oracle_needs_metal_qualification() {
    use base64::Engine as _;
    let (root, case) = corpus(LOOP_FREE, None);
    let output = [42u8, 0, 0, 0];
    let recorded = "metal-literal-resources-v32";
    assert_ne!(recorded, crate::metal::ORACLE_ABI);
    crate::store::CorpusStore::new(root.path())
        .upsert_metal(crate::observation::MetalObservation {
            timing: Some(crate::observation::test_timing()),
            case_id: case.case_id.clone(),
            air_sha256: case.air_sha256.clone(),
            input_sha256: case.computed_input_sha256().unwrap(),
            metal_output_sha256: crate::hash::sha256_bytes(&output),
            output_b64: base64::engine::general_purpose::STANDARD.encode(output),
            environment_id: "admission-oracle".into(),
            environment: serde_json::json!({"device": "synthetic"}),
            oracle_abi: recorded.into(),
            status: crate::observation::MetalStatus::Qualified,
        })
        .unwrap();
    let expected = needs_metal(
        &case,
        MetalGap::OracleAbi {
            recorded: recorded.into(),
            current: crate::metal::ORACLE_ABI.into(),
        },
    );
    let run = |environment_id, device| {
        crate::candidate::engine::execute(
            root.path(),
            case.clone(),
            device,
            HOST_BACKEND,
            environment_id,
            "admission-candidate",
            // The run stops before any compile, so no worker is started.
            &crate::worker::compile::CompileWorker::program("no-compile-worker"),
        )
        .unwrap_err()
    };
    for device in [DeviceType::Cpu, DeviceType::Gpu] {
        assert_eq!(
            run("admission-oracle", device),
            expected,
            "engine on {device:?}"
        );
    }
    assert!(expected
        .to_string()
        .starts_with("needs_metal_qualification{oracle_abi}: "));
    // Another environment's rows are not this one's: there the case has none.
    assert_eq!(
        run("another-oracle", DeviceType::Cpu),
        ExecutorError::NeedsMetal(NeedsMetal {
            case_id: case.case_id.clone(),
            environment_id: "another-oracle".into(),
            gap: MetalGap::Missing,
        })
    );
}

#[test]
fn an_admission_authorizes_only_its_own_executor_and_device() {
    let (root, case) = corpus(LOOP_FREE, None);
    let admitted = admit(
        root.path(),
        case.clone(),
        ExecutionRequest::EngineCandidate {
            device: DeviceType::Cpu,
        },
    )
    .unwrap();
    let error = admitted
        .into_checked_for(ExecutionRequest::EngineCandidate {
            device: DeviceType::Gpu,
        })
        .err()
        .expect("a CPU admission cannot authorize a GPU run");
    assert!(error.contains("cannot run as"), "{error}");
    let admitted = admit(root.path(), case, ExecutionRequest::MetalOracle).unwrap();
    assert!(admitted.checked().loop_class().is_loop_free());
    assert!(admitted
        .into_checked_for(ExecutionRequest::MetalOracle)
        .is_ok());
}

/// The engine candidate is not held to loop-freedom: admission lets a cyclic case through on
/// either device, its modules running as compiled, and the run then stops at the missing Metal
/// observation, before any device opens.
#[test]
fn the_engine_candidate_is_admitted_whatever_its_loops() {
    for device in [DeviceType::Cpu, DeviceType::Gpu] {
        assert_eq!(
            refusal(
                &LoopClass::Cyclic(CycleEvidence::CfgCycle {
                    function: "main".into(),
                }),
                // Not even oracle fuel's scope matters: the oracle's fuel is the oracle's alone.
                &FuelScope::Uncovered(CycleEvidence::CfgCycle {
                    function: "main".into(),
                }),
                ExecutionRequest::EngineCandidate { device },
            ),
            None
        );
    }
    let (root, case) = corpus(CYCLIC, Some(1 << 20));
    let run = |device| {
        crate::candidate::engine::execute(
            root.path(),
            case.clone(),
            device,
            HOST_BACKEND,
            "admission-oracle",
            "admission-candidate",
            // The run stops before any compile, so no worker is started.
            &crate::worker::compile::CompileWorker::program("no-compile-worker"),
        )
        .unwrap_err()
    };
    for device in [DeviceType::Cpu, DeviceType::Gpu] {
        assert_eq!(
            run(device),
            needs_metal(&case, MetalGap::Missing),
            "a {device:?} run must reach the Metal observation"
        );
    }
    // An admission for the Metal oracle does not authorize the candidate.
    let (root, case) = corpus(LOOP_FREE, None);
    let admitted = admit(root.path(), case, ExecutionRequest::MetalOracle).unwrap();
    let error = admitted
        .into_checked_for(ExecutionRequest::EngineCandidate {
            device: DeviceType::Cpu,
        })
        .err()
        .expect("a Metal admission cannot authorize the candidate");
    assert!(error.contains("cannot run as"), "{error}");
}

/// The loop-budget environment bypass and override are gone: no source in this crate names them,
/// so neither can come back as a side door around admission.
#[test]
fn no_environment_variable_selects_the_loop_budget() {
    fn sources(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                sources(&path, out);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                out.push(path);
            }
        }
    }
    let root = metal2vulkan_devtools::validation_dir();
    let mut files = Vec::new();
    sources(&root.join("src"), &mut files);
    sources(&root.join("tests"), &mut files);
    // Spelled in pieces so this file does not match itself.
    let names = [
        concat!("METAL2VULKAN_UNSAFE_", "NO_LOOP_BUDGET"),
        concat!("METAL2VULKAN_", "LOOP_BUDGET"),
    ];
    for file in files {
        let text = std::fs::read_to_string(&file).unwrap();
        for name in names {
            assert!(!text.contains(name), "{} names {name}", file.display());
        }
    }
}
