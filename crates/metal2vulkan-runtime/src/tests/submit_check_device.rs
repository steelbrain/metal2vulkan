//! The device's submission checks (`submit_check.rs`) on the device the cfg selects: they answer,
//! a second device of the same driver reads their record and runs none, and the paths they choose
//! follow their verdicts alone.
//!
//! - **They answer.** A device made over a fresh shader cache runs its checks once
//!   (`WorkCounts::submit_checks_run`), each verdict an answer (never pending once settled; the
//!   follows check answered on Linux where the device imports host pointers), their calls counted
//!   apart (`WorkCounts::submit_check_calls`), their allocations too (`device_checks`), and their
//!   commands among none of the device's
//!   (`commands_recorded`: a device that ran them recorded the copies one that recalled them did,
//!   the follows check's three among neither); a later device over the same cache recalls them
//!   (`submit_checks_recalled`), runs none, and holds the verdicts of the last device that ran
//!   them: the first's, or, where load split its tries or its import tax answered `Holds` short
//!   of a record and nothing was recorded, those of the next device that ran them.
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
    CommandBufferStatus, Device, DeviceKind, DeviceOptions, EventRef, LayerSetting,
    LayerSettingValue, ResourceOptions, ShaderCacheOptions,
};
use metal2vulkan::vk::DriverId;
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
    let mut copies = counts.commands_recorded.cmd_copy_buffer;
    assert_eq!(counts.submit_checks_run, 1);
    assert_eq!(counts.submit_checks_recalled, 0);
    assert!(
        counts.submit_check_calls > 0,
        "the checks' calls were not counted"
    );
    assert!(
        counts.device_checks.vk_memory_allocations > 0,
        "the checks' allocations were not counted apart"
    );
    // A live device with equal options is what `system_default_with` returns again: the first
    // goes before the second is made, its cache thread writing what it queued (the record among
    // it) as it goes, so the second is a new device over the same directory, as a later
    // process's would be.
    let identity = first.driver_identity();
    drop(first);
    // The verdicts of the last device that ran the checks, which its record holds where it wrote
    // one (a device whose tries split, or whose import tax `Holds` fell short of a record, writes
    // none, and the next runs them again).
    let mut ran = answered;
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
            assert_eq!(verdicts(&second), ran);
            assert_eq!(
                counts.commands_recorded.cmd_copy_buffer, copies,
                "the device that ran the checks recorded other copies than one that recalled them"
            );
            break;
        }
        if counts.submit_checks_run == 1 {
            println!("{} again: {}", second.name(), second.submit_check_report());
            ran = verdicts(&second);
            copies = counts.commands_recorded.cmd_copy_buffer;
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

/// MoltenVK's settings under which every activity it tracks is logged at once, each message
/// naming the device and so writing over its loader word (`runtime_lane::device_commands`, which
/// says why a `VkBool32` of 1 is the immediate style). Other drivers ignore them.
fn immediate_tracking() -> Vec<LayerSetting> {
    [
        c"MVK_CONFIG_PERFORMANCE_TRACKING",
        c"MVK_CONFIG_ACTIVITY_PERFORMANCE_LOGGING_STYLE",
    ]
    .map(|name| {
        LayerSetting::new(
            c"MoltenVK".to_owned(),
            name.to_owned(),
            LayerSettingValue::Bool(true),
        )
    })
    .to_vec()
}

/// The checks hold the device's loader word across their whole steps, the wait for their
/// submissions included, only on a device whose word was found overwritten (`loader_word.rs`):
/// a fact of the device, not of its driver. Two devices each run the checks (each over a cache of
/// its own, so neither recalls them): one made as ever, whose word nothing overwrites on any
/// driver, which holds no step; and one made with MoltenVK's activities logged at once, which on
/// MoltenVK names the device in each message, and holds every step after its word was found
/// overwritten (its first steps among them, as the device's own calls found it so first), and,
/// where its queue's failures are relayed, waits for the relay's marks after its submissions
/// before it lets go (MoltenVK names the device as each mark completes too); the queue-switch
/// check's held rounds wait for fences of their own on their batches, which cover them where no
/// mark follows (an adopted device's queues, never relayed). A third device is made with its
/// word taken as found overwritten from the start (`loader_word::take_words_as_overwritten`, a
/// test's mark), on every driver: it holds every step, and where the queue-switch check runs its
/// rounds wait for fences of their own: the path an adopted MoltenVK device's rounds take, seen
/// on a driver whose check measures (MoltenVK's answers `Unchecked`). The checks' threads
/// make no call the loader serves (their command buffers are made as the device is made).
/// Printed (`LOADER-WORD`): each device's steps held, the marks waited for, the restores on the
/// checks' threads and its checks' run.
#[test]
fn the_checks_hold_the_loader_word_across_steps_only_where_it_was_overwritten() {
    for (immediate, taken) in [(false, false), (true, false), (false, true)] {
        let scratch = Scratch::new(&format!("submit-checks-word-{immediate}-{taken}"));
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
                directory: Some(scratch.0.clone()),
                ..ShaderCacheOptions::default()
            },
            layer_settings: match immediate {
                true => immediate_tracking(),
                false => Vec::new(),
            },
            ..DeviceOptions::default()
        };
        crate::loader_word::take_words_as_overwritten(taken);
        let device = Device::system_default_with(&options);
        crate::loader_word::take_words_as_overwritten(false);
        let device = device.unwrap();
        assert!(
            device.settle_submit_checks(SETTLE),
            "the checks never answered"
        );
        assert!(
            device.settle_queue_switch_check(SETTLE),
            "the queue-switch check never answered"
        );
        let moltenvk = device.properties().description.driver_id == Some(DriverId::MOLTENVK);
        let word = &device.shared().loader_word;
        // SAFETY: the device is live.
        let overwritten = unsafe { word.overwritten() };
        let held = word.steps_held();
        let marks = word.marks_waited();
        let restores = word.check_thread_restores();
        let relayed = device.shared().flights(0).is_some();
        let fences = device.shared().queue_switch.fences_waited();
        let counts = device.work_counts();
        println!(
            "LOADER-WORD on {}: MoltenVK {moltenvk}, activities logged at once {immediate}, \
             taken as overwritten {taken}: overwritten {overwritten}, steps held {held}, relay's \
             marks waited {marks} (relayed {relayed}), restores on the checks' threads {restores}; checks run {}: {}; \
             queue-switch checks run {}, their own fences waited {fences}: {}",
            device.name(),
            counts.submit_checks_run,
            device.submit_check_report(),
            counts.queue_switch_checks_run,
            device.queue_switch_check_report()
        );
        assert_eq!(counts.submit_checks_run, 1, "the checks did not run");
        assert_eq!(
            overwritten,
            (moltenvk && immediate) || taken,
            "the word was found overwritten on another device than MoltenVK's with its \
             activities logged at once, or one taken so"
        );
        match overwritten {
            true => assert!(
                held > 0,
                "no step was held on a device whose word was overwritten"
            ),
            false => assert_eq!(
                held, 0,
                "a step was held on a device whose word nothing wrote"
            ),
        }
        assert_eq!(
            restores, 0,
            "a check's thread made a call the loader serves, which reads the word"
        );
        assert_eq!(
            marks > 0,
            overwritten && relayed,
            "the held steps waited for the relay's marks elsewhere than where the word was \
             overwritten and the queue relayed"
        );
        assert_eq!(
            fences > 0,
            overwritten && counts.queue_switch_checks_run > 0,
            "the queue-switch check's held rounds waited for fences of their own elsewhere than \
             where the word was overwritten and the check ran"
        );
    }
}
