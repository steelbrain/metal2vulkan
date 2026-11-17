//! Every authored workload's first run matches its CPU reference on the old runtime, on the
//! device the cfg names (docs/VALIDATION.md).
#![cfg(any(m2v_gpu, m2v_cpu_icd))]

use metal2vulkan_harness::run::{Harness, Runs};
use metal2vulkan_harness::{authored, old, Code, Device};

#[test]
fn every_authored_workload_matches_its_reference_on_the_old_runtime() {
    let device = old::system_default().unwrap();
    let air = authored::library_air().unwrap();
    let library = device.new_library(Code::Air(&air)).unwrap();
    let harness = Harness::new(&device, &library).unwrap();
    let runs = Runs {
        warmup: 0,
        timed: 1,
    };
    for workload in authored::WORKLOADS {
        let name = workload.name;
        let report = (harness.run(&library, &(workload.plan)(), runs))
            .unwrap_or_else(|why| panic!("{name}: {why}"));
        assert_eq!(report.gpu_us.len(), 1, "{name}");
    }
}
