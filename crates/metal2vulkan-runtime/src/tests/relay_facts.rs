//! `device/relay.rs`, on the device the cfg selects: its queues' failures are relayed exactly
//! where its driver has the fact that earns the relay
//! (`DriverFact::QueueFailuresReportedByMessenger`) and its instance has the relay's messenger,
//! and a use of a `VkQueue` outside a batch makes the relay's mark of no work exactly then:
//! counted among the device's submissions for the runtime's use, among the submission checks'
//! calls for theirs (`By::Checks`).
//!
//! `tests::batches::only_drivers_that_report_their_failures_by_messenger_relay_them` reads the
//! committed profiles, whose identities are the releases the lanes measured. A host running a
//! release no profile holds is the one identity those cannot speak for, and the relay is the
//! defect whose loss is silent: without it a GPU fault loses the whole device, where Metal fails
//! the command buffer it struck and the queue runs on. So this reads the live device.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::device::relay::By;
use crate::device::submit_stalls::Uses;
use crate::tests::SETTLE;
use crate::{Device, DeviceKind, DeviceOptions, ShaderCacheOptions};
use ash::vk;
use metal2vulkan::device::DriverFact;
use std::path::{Path, PathBuf};

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!("m2v-relay-facts-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`.
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

/// The live device's queues keep their submissions in flight exactly where its driver has the
/// fact, and its identity is printed either way.
#[test]
fn the_live_device_relays_its_queue_failures_exactly_where_its_driver_has_the_fact() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let properties = device.properties();
    let description = &properties.description;
    let fact = DriverFact::QueueFailuresReportedByMessenger;
    let has = properties.driver_facts.contains(&fact);
    println!(
        "RELAY-FACT {:?} {} ({:#x}/{:#x}, driver version {:#x}): the fact {has}",
        description.driver_id,
        properties.name,
        properties.vendor_id,
        properties.device_id,
        description.driver_version,
    );
    let queues = device.shared().queues.len();
    let relays = (0..queues).any(|queue| device.shared().flights(queue).is_some());
    // The relay also needs the instance's messenger, which an adopted instance and one without
    // `VK_EXT_debug_utils` lack: so the fact is necessary, not sufficient.
    assert!(
        !relays || has,
        "the device relays its queue failures without the fact that earns it"
    );
    assert!(
        relays || !has,
        "the device has the fact that earns the relay and relays nothing: its instance has no \
         messenger, or the relay was not installed"
    );
    drop(device);
}

/// A use of the first `VkQueue` outside a batch makes one mark where the device relays its
/// failures, none elsewhere: the runtime's counted among the device's submissions
/// (`WorkCounts::queue_submissions`), the submission checks' among their calls
/// (`submit_check_calls`), which the device's leave out. Each use's mark is in the log, made,
/// until a later use takes the marks done before it.
#[test]
fn a_use_s_mark_is_counted_as_its_user_s_submission() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    // Nothing of the device's own checks submits beside the uses counted here (its check of
    // textures over host pages starts only at such a texture's first use).
    assert!(
        device.settle_submit_checks(SETTLE),
        "the submission checks answered"
    );
    assert!(
        device.settle_queue_switch_check(SETTLE),
        "the queue-switch check answered"
    );
    let shared = device.shared();
    let marks = u64::from(shared.flights(0).is_some());
    println!("RELAY-MARK on {}: a mark a use {marks}", device.name());
    for (by, name, own, checks) in [
        (By::Runtime, "the runtime's", marks, 0),
        (By::Checks, "the checks'", 0, marks),
    ] {
        let before = device.work_counts();
        shared.use_queue(0, by, Uses::Unknown, |_| ());
        let after = device.work_counts();
        assert_eq!(
            after.queue_submissions - before.queue_submissions,
            own,
            "{name} use: the device's submissions"
        );
        assert_eq!(
            after.submit_check_calls - before.submit_check_calls,
            checks,
            "{name} use: the checks' calls"
        );
        if let Some(flights) = shared.flights(0) {
            assert!(
                flights.in_flight() >= 1,
                "{name} use: its mark is in the log"
            );
        }
    }
    drop(device);
}

/// A submission in the log whose call has not returned is pending and its fence unread
/// (`Flights::made`): a driver may signal a mark's fence before its `vkQueueSubmit` returns, and
/// another thread's use, taking the marks done before its own, then reset it while the call ran
/// (the validation layer's threading report, on MoltenVK). So a mark logged with a signaled fence
/// is taken by no use until it is made, and then by the next.
#[test]
fn a_mark_whose_call_has_not_returned_is_never_taken() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    assert!(
        device.settle_submit_checks(SETTLE),
        "the submission checks answered"
    );
    let shared = device.shared();
    let Some(flights) = shared.flights(0) else {
        println!(
            "RELAY-MADE on {}: the device relays no failures",
            device.name()
        );
        return;
    };
    let raw = shared.vk();
    // The log's reads are the device's, inside a section of the exit guard.
    let _busy = crate::exit_guard::enter().unwrap();
    // The marks of the device's making complete and leave the log.
    let start = std::time::Instant::now();
    while flights.in_flight() > 0 {
        for done in flights.done_marks() {
            // SAFETY: the fence is signaled and out of the log, its submission completed.
            unsafe { crate::release::object(raw, done) };
        }
        assert!(
            start.elapsed() < SETTLE,
            "the device's marks never completed"
        );
        std::thread::yield_now();
    }
    let info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
    // SAFETY: the device is live.
    let fence = unsafe { raw.create_fence(&info, None) }.unwrap();
    flights.push(fence, By::Runtime);
    assert!(
        flights.done_marks().is_empty(),
        "a mark whose call has not returned was taken"
    );
    assert_eq!(flights.in_flight(), 1);
    flights.made(fence);
    assert_eq!(flights.done_marks(), [fence], "the made mark is taken");
    assert_eq!(flights.in_flight(), 0);
    // SAFETY: the fence is out of the log, in no submission.
    unsafe { crate::release::object(raw, fence) };
    drop(_busy);
    drop(device);
}
