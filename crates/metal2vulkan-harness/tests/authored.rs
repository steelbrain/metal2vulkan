//! Every authored workload's first run matches its CPU reference on each runtime, and every
//! creation workload's buffers start zero, on the device the cfg names (docs/VALIDATION.md). On
//! the new runtime, every gauge reads zero once the device is dropped.
#![cfg(any(m2v_gpu, m2v_cpu_icd))]

use metal2vulkan_harness::run::{Harness, Runs, Session};
use metal2vulkan_harness::{authored, old, rt};

/// Each authored and creation workload on `session`, with one timed run each.
fn every_workload(session: &mut dyn Session) {
    let runs = Runs {
        warmup: 0,
        settle: false,
        timed: 1,
    };
    for workload in authored::WORKLOADS {
        let name = workload.name;
        let report =
            (session.run(&(workload.plan)(), runs)).unwrap_or_else(|why| panic!("{name}: {why}"));
        assert_eq!(report.gpu_us.len(), 1, "{name}");
    }
    for create in authored::CREATES {
        let name = create.name;
        let report =
            (session.create(create.length, runs)).unwrap_or_else(|why| panic!("{name}: {why}"));
        assert_eq!(report.create_us.len(), 1, "{name}");
    }
}

#[test]
fn every_authored_workload_matches_its_reference_on_the_old_runtime() {
    let air = authored::library_air().unwrap();
    every_workload(&mut Harness::new(old::system_default().unwrap(), &air).unwrap());
}

#[test]
fn every_authored_workload_matches_its_reference_on_the_new_runtime() {
    let air = authored::library_air().unwrap();
    let mut session = Harness::new(rt::system_default().unwrap(), &air).unwrap();
    every_workload(&mut session);
    let waits = session.first_use_waits().unwrap();
    assert!(waits.count > 0 && waits.nanos > 0, "{waits:?}");
    let gauges = Box::new(session).finish().unwrap();
    assert!(gauges.iter().all(|(_, value)| *value == 0), "{gauges:?}");
}
