//! Every authored workload's first run matches its CPU reference on each runtime, or the runtime
//! refuses it with a typed refusal, which the test prints; every creation workload's buffers
//! start zero, and its pipeline states are made, the first with its VkPipeline made within its
//! timed first, on the device the cfg names (docs/VALIDATION.md). On the new runtime, every gauge
//! reads zero once the device is dropped.
#![cfg(any(m2v_gpu, m2v_cpu_icd))]

use metal2vulkan_harness::run::{Harness, Runs, Session};
use metal2vulkan_harness::{authored, old, rt, Refused};

/// Each authored and creation workload on `session`, with one timed run each.
fn every_workload(session: &mut dyn Session) {
    let runs = Runs {
        warmup: 0,
        settle: false,
        timed: 1,
    };
    for workload in authored::WORKLOADS {
        let name = workload.name;
        let plan = (workload.plan)();
        match session.run(&plan, runs) {
            Ok(report) => {
                assert_eq!(report.gpu_us.len(), 1, "{name}");
                let handlers = if plan.commit_to_complete {
                    plan.command_buffers()
                } else {
                    0
                };
                assert_eq!(report.commit_us.len(), handlers, "{name}");
                assert!(
                    report.commit_us.iter().all(|times| times.len() == 1),
                    "{name}"
                );
            }
            Err(why) => match why.downcast::<Refused>() {
                Ok(refused) => println!("{name}: REFUSED: {refused}"),
                Err(why) => panic!("{name}: {why}"),
            },
        }
    }
    for create in authored::CREATES {
        let name = create.name;
        let made = session.pipelines_created();
        let report =
            (session.create(create.makes, runs)).unwrap_or_else(|why| panic!("{name}: {why}"));
        match create.makes {
            authored::Makes::Buffers { .. } => {
                assert_eq!(report.create_us.len(), 1, "{name}");
            }
            authored::Makes::States { .. } | authored::Makes::RenderStates { .. } => {
                let timed = authored::STATE_CREATIONS as usize;
                assert_eq!(report.create_us.len(), timed, "{name}");
                assert!(report.first_us.is_some(), "{name}");
                // The first state's timed work holds its VkPipeline's compile on either runtime:
                // in the creation, or at its one dispatch.
                assert!(
                    session.pipelines_created() > made,
                    "{name}: the first state made no VkPipeline"
                );
            }
        }
    }
}

#[test]
fn every_authored_workload_matches_its_reference_on_the_old_runtime() {
    let air = authored::library_air().unwrap();
    every_workload(&mut Harness::new(old::system_default(None).unwrap(), &air).unwrap());
}

#[test]
fn every_authored_workload_matches_its_reference_on_the_new_runtime() {
    let air = authored::library_air().unwrap();
    let mut session = Harness::new(rt::system_default(None).unwrap(), &air).unwrap();
    every_workload(&mut session);
    let waits = session.first_use_waits().unwrap();
    assert!(waits.count > 0 && waits.nanos > 0, "{waits:?}");
    let gauges = Box::new(session).finish().unwrap();
    assert!(gauges.iter().all(|(_, value)| *value == 0), "{gauges:?}");
}
