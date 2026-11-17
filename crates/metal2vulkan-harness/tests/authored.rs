//! Every authored workload's first run matches its CPU reference on the old runtime, on the
//! device the cfg names (docs/VALIDATION.md).
#![cfg(any(m2v_gpu, m2v_cpu_icd))]

use metal2vulkan_harness::run::{Harness, Runs, Session};
use metal2vulkan_harness::{authored, old};

#[test]
fn every_authored_workload_matches_its_reference_on_the_old_runtime() {
    let air = authored::library_air().unwrap();
    let mut harness = Harness::new(old::system_default().unwrap(), &air).unwrap();
    let runs = Runs {
        warmup: 0,
        timed: 1,
    };
    for workload in authored::WORKLOADS {
        let name = workload.name;
        let report =
            (harness.run(&(workload.plan)(), runs)).unwrap_or_else(|why| panic!("{name}: {why}"));
        assert_eq!(report.gpu_us.len(), 1, "{name}");
    }
}
