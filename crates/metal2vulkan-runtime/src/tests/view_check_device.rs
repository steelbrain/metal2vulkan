//! The view check (`view_check.rs`) on a device: it starts as the device is made, settles with its
//! medians, and a later device of the same driver over the same shader cache reads its record as it
//! is made, before its first pipeline.
//!
//! - **Settles.** On lavapipe (`m2v_cpu_icd`), which writes timestamps, it is measured: a verdict,
//!   the per-lane ratio, and the report naming every arm's median. On a GPU it is measured, or
//!   unchecked with why; the report is printed either way.
//! - **Taken.** The compile device's profile carries exactly the answer's facts.
//! - **Kept.** Where it was measured, the next device over the same cache, made after the first
//!   went, answers the same at once, from the record, running no check.

use super::blit_holds::{options, Scratch};
use crate::{Device, ViewVerdict};
use std::time::Duration;

/// How long a test waits for the check: lavapipe runs its kernels on the CPU.
const WAIT: Duration = Duration::from_secs(120);

#[test]
fn the_check_settles_with_its_medians_and_a_later_device_reads_its_record() {
    let scratch = Scratch::new("view-check");
    let mut options = options(&scratch.0);
    options.view_facts = None;
    let device = Device::system_default_with(&options).unwrap();
    assert!(device.settle_view_check(WAIT), "the check answered");
    let costs = device.view_costs();
    let report = device.view_check_report();
    println!("view check on {}: {report}", device.name());
    assert_ne!(costs.verdict, ViewVerdict::Pending);
    if cfg!(m2v_cpu_icd) {
        assert_eq!(costs.verdict, ViewVerdict::Measured, "{report}");
    }
    let workarounds = device.compile_device().profile().workarounds().to_vec();
    for fact in [
        metal2vulkan::device::DriverWorkaround::UniformBufferPerLaneReadsSerialize,
        metal2vulkan::device::DriverWorkaround::ExternalUniformReadsSkipACache,
    ] {
        assert_eq!(
            workarounds.contains(&fact),
            costs.facts.contains(&fact),
            "{fact:?}: {report}"
        );
    }
    if costs.verdict == ViewVerdict::Unchecked {
        assert!(costs.facts.is_empty(), "{report}");
        return;
    }
    assert!(costs.per_lane.is_some(), "{report}");
    for named in [
        "Uniform/PerLane/device/1KiB",
        "Storage/PerLane/device/4KiB",
        "ran",
    ] {
        assert!(report.contains(named), "{named}: {report}");
    }
    drop(device);
    let again = Device::system_default_with(&options).unwrap();
    assert_eq!(again.view_costs(), costs, "the record's answer, at once");
    let report = again.view_check_report();
    assert!(
        report.contains("read from the shader cache's record"),
        "{report}"
    );
}
