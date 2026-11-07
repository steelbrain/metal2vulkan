//! The device's submission checks (`submit_check.rs`) on the device the cfg selects: they answer,
//! a second device of the same driver reads their record and runs none, and the paths they choose
//! follow their verdicts alone.
//!
//! - **They answer.** A device made over a fresh shader cache runs its checks once
//!   (`WorkCounts::submit_checks_run`), each verdict an answer (never pending once settled; the
//!   follows check answered on Linux where the device imports host pointers), their calls counted
//!   apart (`WorkCounts::submit_check_calls`); a second device over the same cache recalls them
//!   (`submit_checks_recalled`), runs none, and holds the same verdicts.
//! - **The paths follow them.** Grouped batches where the merge holds
//!   (`WorkCounts::grouped_submissions`), a long `Shared` buffer in fresh pages where a held
//!   import adds nothing to a submission (`fresh_page_buffers_made`), a no-copy buffer's pointer
//!   import released while idle where pointer imports follow their address
//!   (`host_imports_released`, `tests/pointer_residency.rs` for the release itself). Each
//!   verdict is printed, with what the checks measured: the evidence of which path the device
//!   took.
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::blit_holds::{Scratch, WAIT};
use crate::fresh_pages::FRESH_PAGES_PAST;
use crate::submit_check::{record_name, Check, Verdict};
use crate::tests::SETTLE;
use crate::{
    CommandBufferStatus, Device, DeviceKind, DeviceOptions, EventRef, ResourceOptions,
    ShaderCacheOptions,
};
use std::path::Path;

/// The device the cfg selects, its shader cache in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// The files under `directory` named `name`.
fn files_named(directory: &Path, name: &str) -> Vec<std::path::PathBuf> {
    let mut found = Vec::new();
    let mut pending = vec![directory.to_owned()];
    while let Some(at) = pending.pop() {
        for entry in std::fs::read_dir(&at).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if entry.file_name().to_str() == Some(name) {
                found.push(path);
            }
        }
    }
    found
}

/// Each check's verdict on `device`.
fn verdicts(device: &Device) -> [Verdict; 3] {
    Check::ALL.map(|check| device.shared().submit_checks.verdict(check))
}

#[test]
fn the_checks_answer_once_and_a_second_device_recalls_them() {
    let scratch = Scratch::new("submit-checks");
    let first = device(&scratch.0);
    assert!(
        first.settle_submit_checks(SETTLE),
        "the checks never answered"
    );
    let answered = verdicts(&first);
    println!("{}: {}", first.name(), first.submit_check_report());
    assert!(!answered.contains(&Verdict::Pending), "{answered:?}");
    let imports = first.shared().host_import.is_ok();
    if imports && cfg!(target_os = "linux") {
        assert_ne!(
            answered[2],
            Verdict::Unchecked,
            "the follows check did not run"
        );
    }
    let counts = first.work_counts();
    assert_eq!(counts.submit_checks_run, 1);
    assert_eq!(counts.submit_checks_recalled, 0);
    assert!(
        counts.submit_check_calls > 0,
        "the checks' calls were not counted"
    );
    // A live device with equal options is what `system_default_with` returns again: the first
    // goes before the second is made, its cache thread writing what it queued (the record among
    // it) as it goes, so the second is a new device over the same directory, as a later
    // process's would be.
    let identity = first.driver_identity();
    drop(first);
    let start = std::time::Instant::now();
    loop {
        let second = device(&scratch.0);
        assert!(
            second.settle_submit_checks(SETTLE),
            "the second device never answered"
        );
        let counts = second.work_counts();
        if counts.submit_checks_recalled == 1 {
            assert_eq!(counts.submit_checks_run, 0);
            assert_eq!(counts.submit_check_calls, 0);
            assert_eq!(verdicts(&second), answered);
            break;
        }
        if start.elapsed() >= WAIT {
            let name = record_name(&second.driver_identity());
            let found = files_named(&scratch.0, &name);
            let statistics = second.shader_cache_statistics();
            panic!(
                "no device recalled the record {name}: found at {found:?}; the second device's \
                 cache rejected {} and failed {} reads; identities {:?} and {:?}",
                statistics.rejected,
                statistics.read_failures,
                identity,
                second.driver_identity(),
            );
        }
        drop(second);
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// The paths the verdicts choose: grouped batches, fresh pages, releasable pointer imports.
#[test]
fn the_paths_follow_the_verdicts() {
    let scratch = Scratch::new("submit-check-paths");
    let device = device(&scratch.0);
    assert!(
        device.settle_submit_checks(SETTLE),
        "the checks never answered"
    );
    println!("{}: {}", device.name(), device.submit_check_report());
    // Batches ready together behind an event the host raises go grouped where the merge holds.
    let queue = device.new_command_queue().unwrap();
    let event = device.new_shared_event().unwrap();
    let target = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let before = device.work_counts();
    let mut last = None;
    for round in 0..8u8 {
        let command_buffer = queue.command_buffer().unwrap();
        if round == 0 {
            command_buffer
                .encode_wait_for_event(EventRef::Shared(&event), 1)
                .unwrap();
        }
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&target, 0..4096, round).unwrap();
        blit.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        last = Some(command_buffer);
    }
    std::thread::sleep(std::time::Duration::from_millis(50));
    event.set_signaled_value(1);
    let last = last.unwrap();
    assert!(last.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(last.status(), CommandBufferStatus::Completed);
    let after = device.work_counts();
    let grouped = after.grouped_submissions - before.grouped_submissions;
    println!("grouped submissions: {grouped}");
    assert_eq!(
        grouped > 0,
        device.submissions_merge(),
        "grouping against the merge verdict"
    );
    // A long `Shared` buffer lies in fresh pages where a held import costs nothing.
    let before = device.work_counts().fresh_page_buffers_made;
    let long = device
        .new_buffer(FRESH_PAGES_PAST + 4096, ResourceOptions::shared())
        .unwrap();
    let made = device.work_counts().fresh_page_buffers_made - before;
    println!("fresh-page buffers made: {made}");
    let imports = device.shared().host_import.is_ok();
    assert_eq!(
        made == 1,
        imports && device.fresh_pages_pay(),
        "fresh pages against the verdict"
    );
    assert_eq!(long.in_fresh_pages(), made == 1);
}
