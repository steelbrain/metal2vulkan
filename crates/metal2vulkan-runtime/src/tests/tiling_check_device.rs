//! The tiling-cost check (`tiling_check.rs`) on a device: it answers once asked, with its
//! medians, and a later device of the same driver over the same shader cache reads its record.
//!
//! - **Asked.** A device's answer is pending until `settle_tiling_cost_check` is called, which
//!   starts the check and waits for it. On lavapipe (`m2v_cpu_icd`), which writes timestamps,
//!   imports host pointers and makes linear images in place, it is measured: a verdict, the four
//!   medians, the linear and optimal frames above zero, and the report naming them. On a GPU
//!   it is measured, or unchecked with why; the report is printed either way.
//! - **Kept.** Where it was measured, the next device over the same cache, made after the first
//!   went, answers the same medians at once, from the record, running no check.

use super::blit_holds::{device, Scratch};
use crate::{Device, TilingVerdict};
use std::time::Duration;

/// How long a test waits for the check: lavapipe draws its frames on the CPU.
const WAIT: Duration = Duration::from_secs(120);

/// The check's answer on a device over `cache`, settled, with its report printed.
fn settled(cache: &std::path::Path) -> (Device, crate::TilingCost, String) {
    let device = device(cache);
    assert_eq!(
        device.tiling_cost().verdict,
        TilingVerdict::Pending,
        "not asked yet"
    );
    assert!(device.settle_tiling_cost_check(WAIT), "the check answered");
    let cost = device.tiling_cost();
    let report = device.tiling_cost_check_report();
    println!("tiling-cost check on {}: {report}", device.name());
    (device, cost, report)
}

#[test]
fn the_check_answers_once_asked_and_a_later_device_reads_its_record() {
    let scratch = Scratch::new("tiling-check");
    let (device, cost, report) = settled(&scratch.0);
    assert_ne!(cost.verdict, TilingVerdict::Pending);
    if cfg!(m2v_cpu_icd) {
        assert_ne!(
            cost.verdict,
            TilingVerdict::Unchecked,
            "measured on lavapipe: {report}"
        );
    }
    if cost.verdict == TilingVerdict::Unchecked {
        assert!(cost.linear.is_none() && cost.optimal.is_none(), "{report}");
        assert!(report.starts_with("verdict Unchecked: "), "{report}");
        return;
    }
    let medians = [cost.linear, cost.optimal, cost.download, cost.upload];
    assert!(medians.iter().all(Option::is_some), "{report}");
    assert!(cost.linear.unwrap() > Duration::ZERO && cost.optimal.unwrap() > Duration::ZERO);
    for named in ["linear", "optimal", "download", "upload", "ran"] {
        assert!(report.contains(named), "{named}: {report}");
    }
    drop(device);
    let (_again, recalled, report) = settled(&scratch.0);
    assert_eq!(recalled, cost, "the record's answer");
    assert!(
        report.contains("read from the shader cache's record"),
        "{report}"
    );
}
