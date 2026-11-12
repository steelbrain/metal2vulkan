//! The device's submission checks (`submit_check.rs`) on the device the cfg selects: they answer,
//! a second device of the same driver reads their record and runs none, and the paths they choose
//! follow their verdicts alone.
//!
//! - **They answer.** A device made over a fresh shader cache runs its checks once
//!   (`WorkCounts::submit_checks_run`), each verdict an answer (never pending once settled), their
//!   calls counted apart (`WorkCounts::submit_check_calls`), their allocations too
//!   (`device_checks`), and their commands among none of the device's (`commands_recorded`: a
//!   device that ran them recorded the copies one that recalled them did, besides its priming's,
//!   which a device made again on what stays loaded leaves out); a later device over the
//!   same cache recalls them
//!   (`submit_checks_recalled`), runs none, and holds the verdicts of the last device that ran
//!   them: the first's, or, where load split its tries or its import tax answered `Holds` short
//!   of a record and nothing was recorded, those of the next device that ran them.
//! - **No remap under an import.** The checks map no page over a live import of it: they make and
//!   bind no buffer, and every import of theirs is freed by the time they answer
//!   ([`a_device_s_checks_map_no_pages_over_a_live_import`]).
//! - **The paths follow them.** Grouped batches where the merge holds
//!   (`WorkCounts::grouped_submissions`), a long `Shared` buffer in fresh pages where a held
//!   import adds nothing to a submission (`fresh_page_buffers_made`), and a no-copy buffer's
//!   pointer import released while idle where one does (`host_imports_released`: on Linux,
//!   `tests/pointer_residency.rs`). Each
//!   verdict is printed, with what the checks measured: the evidence of which path the device
//!   took.
//! - **A stall ends them, and the teardown.** A submission of theirs that never completes stalls
//!   them at their bound, and the device's teardown returns within it again, the device left to
//!   the process's end
//!   ([`a_submission_that_never_completes_stalls_the_checks_and_the_teardown_returns`]).
//!
//! Device tests: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`).

use super::blit_holds::{Scratch, WAIT};
use crate::fresh_pages::FRESH_PAGES_PAST;
use crate::submit_check::{record_name, Check, Verdict, MERGED, TAX_IMPORTS};
use crate::tests::SETTLE;
use crate::{
    CommandBufferStatus, Device, DeviceKind, DeviceOptions, EventRef, LayerSetting,
    LayerSettingValue, MessageSink, ResourceOptions, ShaderCacheOptions,
};
use metal2vulkan::vk::DriverId;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The device the cfg selects, its shader cache in `cache`.
fn device(cache: &Path) -> Device {
    Device::system_default_with(&options(cache)).unwrap()
}

/// The options of the device the cfg selects, its shader cache in `cache`.
fn options(cache: &Path) -> DeviceOptions {
    DeviceOptions {
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
    }
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
fn verdicts(device: &Device) -> [Verdict; Check::COUNT] {
    Check::ALL.map(|check| device.shared().submit_checks.verdict(check))
}

/// The copies a device's priming records among the device's commands (`device/prime.rs`: one,
/// within its buffer).
const PRIMING_COPIES: u64 = 1;

/// The device the cfg selects, its shader cache in `cache`, and the copies its priming recorded:
/// [`PRIMING_COPIES`] where it was primed, none where it was made again on what stays loaded and
/// not primed.
fn device_primed(cache: &Path) -> (Device, u64) {
    let before = crate::exit_guard::registrations().primings;
    let device = device(cache);
    let primed = crate::exit_guard::registrations().primings - before;
    (device, primed * PRIMING_COPIES)
}

#[test]
fn the_checks_answer_once_and_a_second_device_recalls_them() {
    let scratch = Scratch::new("submit-checks");
    let (first, primed) = device_primed(&scratch.0);
    assert!(
        first.settle_submit_checks(SETTLE),
        "the checks never answered"
    );
    let answered = verdicts(&first);
    println!("{}: {}", first.name(), first.submit_check_report());
    assert!(!answered.contains(&Verdict::Pending), "{answered:?}");
    let imports = first.shared().host_import.is_ok();
    let counts = first.work_counts();
    // The copies besides its priming's, which a device made again on what stays loaded leaves
    // out.
    let mut copies = counts.commands_recorded.cmd_copy_buffer - primed;
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
    // Each import tax try makes its imports once and releases them once (`tax_holds`).
    let tries = (first.shared().submit_checks.tax_tries).load(std::sync::atomic::Ordering::Relaxed);
    let checks = &counts.device_checks;
    println!(
        "the checks' allocations {}, frees {}, over {tries} import tax tries",
        checks.vk_memory_allocations, checks.vk_memory_freed,
    );
    if imports {
        assert!(
            checks.vk_memory_allocations <= tries * TAX_IMPORTS,
            "{} allocations over {tries} tries of {TAX_IMPORTS} imports",
            checks.vk_memory_allocations,
        );
    }
    // A live device with equal options is what `system_default_with` returns again: the first
    // goes before the second is made, its cache thread writing what it queued (the record among
    // it) as it goes, so the second is a new device over the same directory, as a later
    // process's would be.
    let identity = first.driver_identity();
    drop(first);
    // The verdicts of the last device that ran the checks, which its record holds where it wrote
    // one (a device whose tries split, or whose import tax `Holds` fell short of its margin or of
    // its record, writes none, and the next runs them again).
    let mut ran = answered;
    let start = std::time::Instant::now();
    loop {
        let (second, primed) = device_primed(&scratch.0);
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
                counts.commands_recorded.cmd_copy_buffer - primed,
                copies,
                "the device that ran the checks recorded other copies than one that recalled them"
            );
            break;
        }
        if counts.submit_checks_run == 1 {
            println!("{} again: {}", second.name(), second.submit_check_report());
            ran = verdicts(&second);
            copies = counts.commands_recorded.cmd_copy_buffer - primed;
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

/// A device's making maps no page over a live import of it, on its own thread or the checks':
/// the submission checks' imports are the import tax's (`submit_check.rs`), over pages the checks
/// map where the kernel places them, never map over, and unmap only once every import of them was
/// freed; they make and bind no buffer, so none of their commands reads imported pages. A device
/// made over a fresh shader cache runs them (`WorkCounts::submit_checks_run`), counted apart
/// (`WorkCounts::device_checks`, the queue-switch check's work among it, which makes no buffer or
/// memory): no `vkCreateBuffer` or `vkBindBufferMemory`, at most [`TAX_IMPORTS`] memories a try,
/// and, where the import tax answered, every one freed by then. Printed (`NO-REMAP`).
#[test]
fn a_device_s_checks_map_no_pages_over_a_live_import() {
    let scratch = Scratch::new("submit-checks-no-remap");
    let device = device(&scratch.0);
    assert!(
        device.settle_submit_checks(SETTLE),
        "the checks never answered"
    );
    let counts = device.work_counts();
    assert_eq!(counts.submit_checks_run, 1, "the checks did not run");
    let checks = &counts.device_checks;
    let tries = (device.shared().submit_checks.tax_tries).load(Ordering::Relaxed);
    let taxed = device.shared().submit_checks.verdict(Check::ImportTax);
    println!(
        "NO-REMAP {}: the checks' memories {} allocated, {} freed over {tries} import tax tries \
         (import tax {taxed:?}); buffers {} made, {} bound: {}",
        device.name(),
        checks.vk_memory_allocations,
        checks.vk_memory_freed,
        checks.vk_buffers_made,
        checks.vk_buffer_bindings,
        device.submit_check_report(),
    );
    assert_eq!(
        (checks.vk_buffers_made, checks.vk_buffer_bindings),
        (0, 0),
        "the checks made or bound a buffer"
    );
    assert!(
        checks.vk_memory_allocations <= tries * TAX_IMPORTS,
        "{} memories over {tries} tries of {TAX_IMPORTS} imports",
        checks.vk_memory_allocations,
    );
    if taxed != Verdict::Unchecked {
        assert_eq!(
            checks.vk_memory_freed, checks.vk_memory_allocations,
            "an import of the checks outlived their answer"
        );
    }
}

/// A loss of the device the checks meet stops them at the call that answers it, here a wait for
/// their fence answering `VK_ERROR_DEVICE_LOST` (injected, `SubmitChecks::lost_at_wait`): nothing
/// is submitted after it, so no fence or command buffer of theirs goes to the lost device again,
/// and they record nothing, each check that had not answered `Unchecked` for the device's life.
/// On the M3 Max (MoltenVK 1.4.2), a real loss under the program's own
/// `MVK_CONFIG_RESUME_LOST_DEVICE` off had them submit their fence and command buffer again, which
/// the validation layer read as still pending (`VUID-vkQueueSubmit-fence-00064`,
/// `VUID-vkQueueSubmit-pCommandBuffers-00071`).
#[test]
fn a_loss_stops_the_checks_with_nothing_submitted_again_or_recorded() {
    let scratch = Scratch::new("submit-check-loss");
    let device = device(&scratch.0);
    let checks = Arc::clone(&device.shared().submit_checks);
    checks.held.hold();
    // The checks started as the device was made: they wait at their next step, or ended.
    let deadline = Instant::now() + WAIT;
    while checks.held.waiting() == 0 && !device.settle_submit_checks(Duration::ZERO) {
        assert!(
            Instant::now() < deadline,
            "the checks neither held nor ended"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    if checks.held.waiting() == 0 {
        checks.held.release();
        println!(
            "SKIPPED {}: the device's submission checks ended before the hold",
            device.name()
        );
        return;
    }
    let before = device.work_counts().submit_check_calls;
    checks.lost_at_wait.store(true, Ordering::Release);
    checks.held.release();
    assert!(
        device.settle_submit_checks(SETTLE),
        "the checks never ended"
    );
    let counts = device.work_counts();
    let calls = counts.submit_check_calls - before;
    let answered = verdicts(&device);
    println!(
        "{}: {calls} calls from the hold, verdicts {answered:?}",
        device.name()
    );
    assert!(
        !checks.lost_at_wait.load(Ordering::Acquire),
        "the checks waited for no fence after the hold"
    );
    // One use of the queue, the one whose wait met the loss: its calls (a merge round's apart
    // form makes [`MERGED`]) and, where the device relays its queues' failures, the relay's mark
    // after them, counted with them (`device/relay.rs`, `By::Checks`).
    assert!(
        (1..=MERGED as u64 + 1).contains(&calls),
        "{calls} calls from the hold: the checks submitted again after the loss"
    );
    assert_eq!(counts.submit_checks_run, 0, "the checks ran to their end");
    assert!(
        !answered.contains(&Verdict::Pending),
        "the stopped checks left {answered:?}"
    );
    // The device goes, its cache thread writing what it queued as it goes: no record among it.
    let name = record_name(&device.driver_identity());
    drop(device);
    let found = files_named(&scratch.0, &name);
    assert!(
        found.is_empty(),
        "a lost device's checks recorded {found:?}"
    );
}

/// The bound the stall test gives the checks' waits and the teardown's, in place of `STALL`.
const STALL_BOUND: Duration = Duration::from_secs(2);

/// What the stall test allows past [`STALL_BOUND`] for the checks to settle, or the teardown to
/// return, beside the device tests' load.
const STALL_SLACK: Duration = Duration::from_secs(10);

/// Signals the stall test's timeline semaphore to 1 as it goes, where the device's `VkDevice` is
/// still live (held, or left to the process's end), so its queue drains: a failing assertion
/// before the device's drop unblocks the submission the drop's teardown would wait for.
struct Unblock {
    raw: crate::device::functions::Vk,
    semaphore: ash::vk::Semaphore,
    lineage: Arc<crate::device::line::Lineage>,
}

impl Drop for Unblock {
    fn drop(&mut self) {
        if self.lineage.counts().0 == 0 {
            return;
        }
        let signal = ash::vk::SemaphoreSignalInfo::default()
            .semaphore(self.semaphore)
            .value(1);
        // SAFETY: the semaphore is the test's, a timeline at 0, of a live device.
        let _ = unsafe { self.raw.signal_semaphore(&signal) };
    }
}

/// A submission of the checks that never completes (injected: its first `VkSubmitInfo` waits for
/// a timeline semaphore of the test's that nothing signals, `SubmitChecks::wait_at_submit`)
/// stalls them at their bound (the test's [`STALL_BOUND`] in place of `STALL`,
/// `SubmitChecks::stall`): they settle, no verdict pending, nothing submitted after it, nothing
/// recorded, the sink told once (`metal2vulkan-submit-check-stalled`). The device's teardown then
/// waits for that submission up to the bound again and returns, leaving its `VkDevice` live to
/// the process's end (`WorkCounts::vk_devices_live`'s count, read on the line after the drop).
/// Without the bound, the checks' wait and the teardown behind it never return. The test then
/// signals the semaphore, so the queue drains.
#[test]
fn a_submission_that_never_completes_stalls_the_checks_and_the_teardown_returns() {
    stall("submit-check-stall", false);
}

/// As [`a_submission_that_never_completes_stalls_the_checks_and_the_teardown_returns`], the
/// submission one of several calls whose second is refused (injected,
/// `Injected::refuse_second`): the checks wait for the first call's work, which never completes,
/// by a submission of their fence alone (`SubmitChecks::idles`), within the same bound, where they
/// waited for the queue's idle with no end.
#[test]
fn work_before_a_refused_call_that_never_completes_stalls_the_checks_and_the_teardown_returns() {
    stall("submit-check-stall-refused", true);
}

/// The stall tests: a submission of the checks that never completes, the second of its several
/// calls refused where `refuse_second`.
fn stall(name: &str, refuse_second: bool) {
    use crate::submit_check::{Injected, STALLED_ID};
    use ash::vk;
    let scratch = Scratch::new(name);
    let heard: Arc<std::sync::Mutex<Vec<String>>> = Arc::default();
    let mut options = options(&scratch.0);
    let hears = Arc::clone(&heard);
    options.messages = Some(MessageSink::new(move |message| {
        if message.id_name == STALLED_ID {
            hears.lock().unwrap().push(message.text.clone());
        }
    }));
    let device = Device::system_default_with(&options).unwrap();
    let checks = Arc::clone(&device.shared().submit_checks);
    checks.held.hold();
    let deadline = Instant::now() + WAIT;
    while checks.held.waiting() == 0 && !device.settle_submit_checks(Duration::ZERO) {
        assert!(
            Instant::now() < deadline,
            "the checks neither held nor ended"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    if checks.held.waiting() == 0 {
        checks.held.release();
        println!(
            "SKIPPED {}: the device's submission checks ended before the hold",
            device.name()
        );
        return;
    }
    let raw = device.shared().vk().clone();
    let mut kind = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
    // SAFETY: the device is live.
    let semaphore = unsafe { raw.create_semaphore(&info, None) }.unwrap();
    let lineage = Arc::clone(&device.shared().lineage);
    // Made after the device, so a failing assertion unblocks the submission before the device
    // goes.
    let _unblock = Unblock {
        raw: raw.clone(),
        semaphore,
        lineage: Arc::clone(&lineage),
    };
    *crate::device::lock(&checks.stall) = Some(STALL_BOUND);
    *crate::device::lock(&checks.wait_at_submit) = Some(Injected {
        semaphore,
        value: 1,
        refuse_second,
    });
    let before = device.work_counts().submit_check_calls;
    let released = Instant::now();
    checks.held.release();
    assert!(
        device.settle_submit_checks(STALL_BOUND + STALL_SLACK),
        "the checks never ended"
    );
    let settled = released.elapsed();
    if crate::device::lock(&checks.wait_at_submit).is_some() {
        println!(
            "SKIPPED {}: the checks held past their last submission of several calls",
            device.name()
        );
        return;
    }
    let counts = device.work_counts();
    let calls = counts.submit_check_calls - before;
    let idles = checks.idles.load(Ordering::Relaxed);
    let answered = verdicts(&device);
    let report = device.submit_check_report();
    println!(
        "{}: settled {settled:?} after the hold, {calls} calls, {idles} of the fence alone, \
         verdicts {answered:?}; {report}",
        device.name()
    );
    assert!(checks.stalled(), "the checks took no submission as stalled");
    assert!(
        settled >= STALL_BOUND,
        "the checks settled {settled:?} after the hold, before their bound"
    );
    assert!(
        checks.left_to_teardown(),
        "the stalled submission's objects were not left to the teardown"
    );
    // One use of the queue, the stalled one, and where a call was refused one more, of the fence
    // alone: their calls and, where the device relays its queues' failures, the relay's mark
    // after each.
    let (most, fence_alone) = match refuse_second {
        true => (MERGED as u64 + 3, 1),
        false => (MERGED as u64 + 1, 0),
    };
    assert!(
        (1..=most).contains(&calls),
        "{calls} calls from the hold: the checks submitted again after the stall"
    );
    assert_eq!(idles, fence_alone, "submissions of the fence alone");
    assert_eq!(counts.submit_checks_run, 0, "the checks ran to their end");
    assert!(
        !answered.contains(&Verdict::Pending),
        "the stalled checks left {answered:?}"
    );
    let told = heard.lock().unwrap().clone();
    assert_eq!(told.len(), 1, "the sink heard {told:?}");
    // SAFETY: the semaphore is the test's, of a live device.
    let value = unsafe { raw.get_semaphore_counter_value(semaphore) }.unwrap();
    assert_eq!(value, 0, "the submission's wait was met");
    let record = record_name(&device.driver_identity());
    let (live, _, _) = lineage.counts();
    let dropped = Instant::now();
    drop(device);
    let teardown = dropped.elapsed();
    let (left, _, _) = lineage.counts();
    println!("teardown {teardown:?}, VkDevices live {live} before it and {left} after");
    assert!(
        teardown >= STALL_BOUND && teardown < STALL_BOUND + STALL_SLACK,
        "the teardown took {teardown:?}: it waited for the stalled submission up to {STALL_BOUND:?}"
    );
    assert_eq!(
        left, live,
        "the teardown destroyed the VkDevice of a submission still pending"
    );
    let found = files_named(&scratch.0, &record);
    assert!(found.is_empty(), "stalled checks recorded {found:?}");
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
    // Batches group only on a queue whose failures are not relayed, each relayed batch taking a
    // fence (`command_buffer/group.rs`): MoltenVK's, which answered `Holds` under load.
    let shared = device.shared();
    let relays = (0..shared.queues.len()).any(|queue| shared.flights(queue).is_some());
    assert_eq!(
        grouped > 0,
        device.submissions_merge() && !relays,
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
