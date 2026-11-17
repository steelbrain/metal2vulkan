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

/// How long the program's work goes on in [`the_check_stands_by_while_the_device_s_work_goes_on`],
/// and how long into it the check's first kernel, made as the device was made before any work,
/// has surely ended.
const STREAM: Duration = Duration::from_millis(900);
const SETTLED: Duration = Duration::from_millis(300);

/// A command buffer of `queue`'s filling the first 64 KiB of `target`, committed.
fn commit_fill(queue: &crate::CommandQueue, target: &crate::Buffer) -> crate::CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..(64 << 10), 0x5a).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
}

/// The program's work never waits on the check: while a queue keeps a command buffer committed or
/// in flight, the check stands by, making no kernel and submitting no dispatch, its answer
/// pending; once the work stops it answers, its stand-bys counted. Each command buffer of the
/// work completes in its own time (the longest is printed); none waits behind the check's work.
#[test]
fn the_check_stands_by_while_the_device_s_work_goes_on() {
    let scratch = Scratch::new("view-check-stand-by");
    let mut options = options(&scratch.0);
    options.view_facts = None;
    let device = Device::system_default_with(&options).unwrap();
    let queue = device.new_command_queue().unwrap();
    let target = device
        .new_buffer(64 << 10, crate::ResourceOptions::private())
        .unwrap();
    let began = std::time::Instant::now();
    let mut previous: Option<(crate::CommandBuffer, std::time::Instant)> = None;
    let mut settled = None;
    let mut longest = Duration::ZERO;
    while began.elapsed() < STREAM {
        let next = (commit_fill(&queue, &target), std::time::Instant::now());
        if let Some((command_buffer, committed)) = previous.replace(next) {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            longest = longest.max(committed.elapsed());
        }
        if settled.is_none() && began.elapsed() >= SETTLED {
            settled = Some(device.work_counts());
        }
    }
    let during = device.work_counts();
    let pending = device.view_costs().verdict;
    if let Some((command_buffer, _)) = previous {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
    }
    let settled = settled.expect("the work went on past SETTLED");
    println!(
        "the work's longest command buffer, commit to completion: {:.2} ms",
        longest.as_secs_f64() * 1e3
    );
    assert_eq!(
        pending,
        ViewVerdict::Pending,
        "no answer while the work went on"
    );
    assert_eq!(
        during.compute_pipeline_create_nanos, settled.compute_pipeline_create_nanos,
        "no kernel made while the work went on"
    );
    assert_eq!(
        during.internal_submissions, settled.internal_submissions,
        "no dispatch submitted while the work went on"
    );
    assert!(device.settle_view_check(WAIT), "the check answered");
    let costs = device.view_costs();
    let checks = device.work_counts().device_checks;
    let report = device.view_check_report();
    if cfg!(m2v_cpu_icd) {
        assert_eq!(costs.verdict, ViewVerdict::Measured, "{report}");
    }
    assert!(checks.view_check_stand_bys >= 1, "{checks:?}");
    let stood = Duration::from_nanos(checks.view_check_stand_by_nanos);
    assert!(stood >= STREAM - SETTLED, "stood by {stood:?}: {checks:?}");
}
