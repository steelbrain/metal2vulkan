//! The completion's stall clock, probe and watchdog (`command_buffer/stall.rs`), over a command
//! buffer that waits on its `VkQueue` for a signal the caller reserved of a plain event
//! (`Event::reserve_vk_signal`) and has not submitted: it is pending, and no work of the device
//! advances. The clock's and the watchdog's bounds are shortened through the line's hooks
//! (`device/line.rs`).
//!
//! - Where the probe is answered (a device with a staging queue) and the front advances before
//!   the watchdog's bound, nothing fails: the probe is counted and said, and the command buffer
//!   completes once the host signals the semaphore. A device without a staging queue sends no
//!   probe, and nothing fails either.
//! - Where no probe is seen answered, as on a GPU whose every engine stopped, the generation
//!   stops: the command buffer fails as the hang (`GpuHang`), the device recovers on a new
//!   `VkDevice` whose notice says the old one stopped, the old `VkDevice` is parked
//!   (`Device::is_parked`), and a command buffer runs after it.
//! - Where the probe is answered but the front does not advance within the watchdog's bound, as
//!   on NVIDIA's driver after an Xid 109 (its copy engine running on), the command buffer fails
//!   as Metal's timed-out one (`GpuHang`) no sooner than the bound, the generation stops and is
//!   parked, and a command buffer runs after it.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs, on the kind of device
//! the cfg selects (docs/VALIDATION.md). No GPU fault and no spin: the stuck work is a semaphore
//! wait. The stopped test signals it on the parked `VkDevice` once it has checked the recovery, so
//! the parked generation's queue drains before the process exits: NVIDIA's driver (620.70.06, an
//! RTX 3090) tears down every `VkDevice` left at exit in an exit handler of `libEGL_nvidia`, which
//! spun without end on the parked one while its batch waited.

use super::device_recovery::Scratch;
use crate::{
    CommandBufferError, CommandBufferStatus, Degradation, DegradeNotice, Device, DeviceKind,
    DeviceOptions, MessageSink, ResourceOptions, ShaderCacheOptions, VkEventSignal,
};
use ash::vk;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The clock's bound in these tests, in place of `stall::STALL`.
const STALL: Duration = Duration::from_millis(200);

/// The longest a test waits for the stall's answer: the bound, the probe's 2 s, and room.
const WAIT: Duration = Duration::from_secs(20);

/// The watchdog's bound in the test of it, in place of `stall::WATCHDOG`.
const WATCHDOG: Duration = Duration::from_secs(1);

/// The watchdog's bound in the test whose front advances: past the second or so it holds the
/// front on either kind of device.
const ADVANCES_WITHIN: Duration = Duration::from_secs(3);

/// What a device's sink heard: its degrade notices, and its informational texts.
#[derive(Default)]
struct Heard {
    notices: Vec<DegradeNotice>,
    texts: Vec<String>,
}

/// The device the cfg selects, its messages kept in `heard`, its stall clock's bound [`STALL`],
/// its watchdog's `watchdog` where set, and no probe seen answered where `unanswered`.
fn device(
    cache: &Path,
    heard: &Arc<Mutex<Heard>>,
    unanswered: bool,
    watchdog: Option<Duration>,
) -> Device {
    let heard = Arc::clone(heard);
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Integrated,
                DeviceKind::Discrete,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        messages: Some(MessageSink::new(move |message| {
            let mut heard = heard.lock().unwrap();
            match &message.degradation {
                Some(notice) => heard.notices.push(notice.clone()),
                None => heard.texts.push(message.text.clone()),
            }
        })),
        ..DeviceOptions::default()
    };
    let device = Device::system_default_with(&options).unwrap();
    let hooks = &device.line().hooks;
    *crate::device::lock(&hooks.stall_after) = Some(STALL);
    *crate::device::lock(&hooks.watchdog_after) = watchdog;
    hooks.probe_unanswered.store(unanswered, Ordering::Release);
    device
}

/// The device the cfg selects with its encoder trail on (`encoder_trail.rs`), its messages kept in
/// `heard`, its stall clock's bound left as it is (no probe within a test's wait).
fn trail_device(cache: &Path, heard: &Arc<Mutex<Heard>>) -> Device {
    let heard = Arc::clone(heard);
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Integrated,
                DeviceKind::Discrete,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        messages: Some(MessageSink::new(move |message| {
            if message.degradation.is_none() {
                heard.lock().unwrap().texts.push(message.text.clone());
            }
        })),
        encoder_trail: true,
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// A committed command buffer that waits for `event` to reach 10, a signal of it reserved for the
/// caller and not submitted, so it is pending on its `VkQueue`; then fills `buffer`. The
/// reserved counter value.
fn stuck(
    device: &Device,
    event: &crate::Event,
    buffer: &crate::Buffer,
) -> (crate::CommandBuffer, u64) {
    // SAFETY: the signal is made from the host below, once, or never where the generation stops
    // (its waits are parked with it).
    let reserved = unsafe { event.reserve_vk_signal(10) }.unwrap();
    let VkEventSignal::Counter { counter, after } = reserved else {
        panic!("a first reservation names a counter: {reserved:?}");
    };
    assert_eq!(after, None);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer
        .encode_wait_for_event(event.into(), 10)
        .unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..4096, 1).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    (command_buffer, counter)
}

/// Wait up to [`WAIT`] for `done` to hold of the device's counts.
fn until(device: &Device, done: impl Fn(&crate::WorkCounts) -> bool) -> bool {
    let deadline = Instant::now() + WAIT;
    while Instant::now() < deadline {
        if done(&device.work_counts()) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

/// A stall whose probe is answered, its front advancing within the watchdog's bound, fails
/// nothing: the probe is counted and said, the stall's length kept, and the stuck command buffer
/// completes once the host signals the semaphore.
#[test]
fn a_stall_whose_probe_is_answered_fails_nothing() {
    let scratch = Scratch::new();
    let heard = Arc::new(Mutex::new(Heard::default()));
    let device = device(&scratch.0, &heard, false, Some(ADVANCES_WITHIN));
    let event = device.new_event().unwrap();
    let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let (command_buffer, counter) = stuck(&device, &event, &buffer);
    let probes = device.has_staging_queue();
    match probes {
        true => assert!(
            until(&device, |counts| counts.stall_probes_sent > 0),
            "no probe went"
        ),
        false => std::thread::sleep(STALL * 4),
    }
    assert!(command_buffer.wait_until_completed_for(STALL).is_err());
    // SAFETY: the event's semaphore, signalled once from the host to the reserved counter, which
    // nothing before it waits for.
    let semaphore = unsafe { event.vk_semaphore() };
    let info = vk::SemaphoreSignalInfo::default()
        .semaphore(semaphore)
        .value(counter);
    // SAFETY: as above; the device is live.
    unsafe { device.shared().vk().signal_semaphore(&info) }.unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (CommandBufferStatus::Completed, None)
    );
    let counts = device.work_counts();
    let heard = heard.lock().unwrap();
    let said: Vec<&String> = (heard.texts.iter())
        .filter(|text| text.contains("a probe on the staging queue"))
        .collect();
    eprintln!(
        "STALL-ANSWERED {}: staging queue {probes}, probes {}, wedges {}, longest stall {:?}; {said:?}",
        device.name(),
        counts.stall_probes_sent,
        counts.wedges_declared,
        Duration::from_nanos(counts.longest_stall_nanos),
    );
    assert_eq!(counts.wedges_declared, 0);
    assert_eq!(counts.watchdog_timeouts, 0);
    assert_eq!(counts.device_recoveries, 0);
    match probes {
        true => {
            assert!(counts.longest_stall_nanos >= STALL.as_nanos() as u64);
            assert!(!said.is_empty(), "the probe was not said");
            assert!(said.iter().all(|text| text.contains("the GPU runs")));
        }
        false => assert_eq!(counts.stall_probes_sent, 0),
    }
}

/// A stall no probe answers stops the generation: the stuck command buffer fails as the hang, the
/// device recovers on a new `VkDevice` whose notice says the old one stopped, the old one is
/// parked and the new one is not, and a command buffer committed after runs.
#[test]
fn a_stall_no_probe_answers_stops_and_parks_the_generation() {
    let scratch = Scratch::new();
    let heard = Arc::new(Mutex::new(Heard::default()));
    let device = device(&scratch.0, &heard, true, None);
    // SAFETY: the functions and handle of the generation that stops, used only to signal the
    // event's semaphore below, on the parked `VkDevice`, which is never destroyed.
    let parked = unsafe { device.raw() }.device.clone();
    let stopped = parked.handle();
    let event = device.new_event().unwrap();
    // SAFETY: the stopping generation's semaphore of the event, which stays that generation's
    // (`Event::vk_semaphore`), signalled once from the host to the reserved counter below.
    let semaphore = unsafe { event.vk_semaphore() };
    let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let began = Instant::now();
    let (command_buffer, counter) = stuck(&device, &event, &buffer);
    assert!(
        command_buffer.wait_until_completed_for(WAIT).is_ok(),
        "the stuck command buffer did not fail in {WAIT:?}"
    );
    let failed = began.elapsed();
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (
            CommandBufferStatus::Error,
            Some(CommandBufferError::GpuHang)
        )
    );
    assert!(device.is_parked(stopped));
    // The next call recovers: a command buffer runs on the new `VkDevice`.
    let after = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let runs = queue.command_buffer().unwrap();
    let mut blit = runs.blit_command_encoder().unwrap();
    blit.fill_buffer(&after, 0..4096, 2).unwrap();
    blit.end_encoding().unwrap();
    runs.commit().unwrap();
    assert!(runs.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(runs.error(), None);
    let ran = began.elapsed();
    // SAFETY: as above.
    let serving = unsafe { device.raw() }.device.handle();
    assert_ne!(serving, stopped);
    assert!(!device.is_parked(serving));
    let counts = device.work_counts();
    let heard = heard.lock().unwrap();
    let recovered: Vec<&DegradeNotice> = (heard.notices.iter())
        .filter(|notice| notice.degradation == Degradation::DeviceRecovered)
        .collect();
    let said: Vec<&String> = (heard.texts.iter())
        .filter(|text| text.contains("a probe on the staging queue"))
        .collect();
    eprintln!(
        "STALL-STOPPED {}: failed after {failed:?}, the next ran after {ran:?}; probes {}, wedges \
         {}, recoveries {}, longest stall {:?}; {said:?}; {:?}",
        device.name(),
        counts.stall_probes_sent,
        counts.wedges_declared,
        counts.device_recoveries,
        Duration::from_nanos(counts.longest_stall_nanos),
        recovered
            .iter()
            .map(|notice| &notice.reason)
            .collect::<Vec<_>>(),
    );
    assert_eq!(counts.wedges_declared, 1);
    assert_eq!(counts.watchdog_timeouts, 0);
    assert_eq!(counts.device_recoveries, 1);
    assert!(counts.stall_probes_sent >= 1);
    assert!(said
        .iter()
        .any(|text| text.contains("the generation stopped")));
    assert!(
        recovered
            .iter()
            .any(|notice| notice.reason.contains("stopped")),
        "the recovery's notice does not say the generation stopped"
    );
    let info = vk::SemaphoreSignalInfo::default()
        .semaphore(semaphore)
        .value(counter);
    // SAFETY: the parked `VkDevice` and its semaphore, both live (never destroyed); the counter
    // the reservation named, which nothing before it waits for.
    unsafe { parked.signal_semaphore(&info) }.unwrap();
}

/// A front that does not advance within the watchdog's bound fails as Metal's timed-out command
/// buffer, whatever the probe answers: no sooner than the bound, as the hang (`GpuHang`), with
/// every probe sent meanwhile answered where the device has a staging queue; the generation
/// stops and is parked, the notice says the watchdog stopped it, and a command buffer committed
/// after runs on the new `VkDevice`.
#[test]
fn a_front_past_the_watchdog_fails_as_metals_timeout() {
    let scratch = Scratch::new();
    let heard = Arc::new(Mutex::new(Heard::default()));
    let device = device(&scratch.0, &heard, false, Some(WATCHDOG));
    // SAFETY: the functions and handle of the generation that stops, used only to signal the
    // event's semaphore below, on the parked `VkDevice`, which is never destroyed.
    let parked = unsafe { device.raw() }.device.clone();
    let stopped = parked.handle();
    let event = device.new_event().unwrap();
    // SAFETY: the stopping generation's semaphore of the event, which stays that generation's
    // (`Event::vk_semaphore`), signalled once from the host to the reserved counter below.
    let semaphore = unsafe { event.vk_semaphore() };
    let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let began = Instant::now();
    let (command_buffer, counter) = stuck(&device, &event, &buffer);
    assert!(
        command_buffer.wait_until_completed_for(WAIT).is_ok(),
        "the stuck command buffer did not fail in {WAIT:?}"
    );
    let failed = began.elapsed();
    assert_eq!(
        (command_buffer.status(), command_buffer.error()),
        (
            CommandBufferStatus::Error,
            Some(CommandBufferError::GpuHang)
        )
    );
    assert!(
        failed >= WATCHDOG,
        "failed after {failed:?}, before the bound"
    );
    assert!(device.is_parked(stopped));
    // The next call recovers: a command buffer runs on the new `VkDevice`.
    let after = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let queue = device.new_command_queue().unwrap();
    let runs = queue.command_buffer().unwrap();
    let mut blit = runs.blit_command_encoder().unwrap();
    blit.fill_buffer(&after, 0..4096, 2).unwrap();
    blit.end_encoding().unwrap();
    runs.commit().unwrap();
    assert!(runs.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(runs.error(), None);
    // SAFETY: as above.
    let serving = unsafe { device.raw() }.device.handle();
    assert_ne!(serving, stopped);
    assert!(!device.is_parked(serving));
    let counts = device.work_counts();
    let heard = heard.lock().unwrap();
    let recovered: Vec<&DegradeNotice> = (heard.notices.iter())
        .filter(|notice| notice.degradation == Degradation::DeviceRecovered)
        .collect();
    let probes: Vec<&String> = (heard.texts.iter())
        .filter(|text| text.contains("a probe on the staging queue"))
        .collect();
    let timed: Vec<&String> = (heard.texts.iter())
        .filter(|text| text.contains("past Metal's watchdog bound"))
        .collect();
    eprintln!(
        "STALL-WATCHDOG {}: staging queue {}, failed after {failed:?}; probes {}, wedges {}, \
         watchdog timeouts {}, recoveries {}; {probes:?}; {timed:?}; {:?}",
        device.name(),
        device.has_staging_queue(),
        counts.stall_probes_sent,
        counts.wedges_declared,
        counts.watchdog_timeouts,
        counts.device_recoveries,
        recovered
            .iter()
            .map(|notice| &notice.reason)
            .collect::<Vec<_>>(),
    );
    assert_eq!(counts.watchdog_timeouts, 1);
    assert_eq!(counts.wedges_declared, 0);
    assert_eq!(counts.device_recoveries, 1);
    assert_eq!(timed.len(), 1, "the watchdog was not said once");
    if device.has_staging_queue() {
        assert!(
            counts.stall_probes_sent >= 1,
            "no probe went before the bound"
        );
        assert!(probes.iter().all(|text| text.contains("the GPU runs")));
    }
    assert!(
        recovered
            .iter()
            .any(|notice| notice.reason.contains("watchdog")),
        "the recovery's notice does not say the watchdog stopped the generation"
    );
    let info = vk::SemaphoreSignalInfo::default()
        .semaphore(semaphore)
        .value(counter);
    // SAFETY: the parked `VkDevice` and its semaphore, both live (never destroyed); the counter
    // the reservation named, which nothing before it waits for.
    unsafe { parked.signal_semaphore(&info) }.unwrap();
}

/// With the encoder trail on (`encoder_trail.rs`), each encoder goes in a submission of its own,
/// and once no work advanced for 2 s, a line names the encoder its queue's oldest pending batch
/// holds and the last batch that queue completed: here the first of two blit encoders after a wait
/// for a signal reserved and not yet made, and the one blit encoder of a command buffer of the
/// queue that completed before. The stuck command buffer completes once the host signals.
#[test]
fn the_encoder_trail_names_the_stuck_encoder_and_the_last_one_completed() {
    let scratch = Scratch::new();
    let heard = Arc::new(Mutex::new(Heard::default()));
    let device = trail_device(&scratch.0, &heard);
    let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    // An unshared queue never moves between `VkQueue`s (`queue_share.rs`).
    let queue = device.new_command_queue_unshared(4).unwrap();
    let done = queue.command_buffer().unwrap();
    let mut blit = done.blit_command_encoder().unwrap();
    blit.fill_buffer(&buffer, 0..4096, 2).unwrap();
    blit.end_encoding().unwrap();
    done.commit().unwrap();
    assert!(done.wait_until_completed_for(WAIT).is_ok());
    let event = device.new_event().unwrap();
    // SAFETY: the signal is made from the host below, once.
    let reserved = unsafe { event.reserve_vk_signal(10) }.unwrap();
    let VkEventSignal::Counter { counter, .. } = reserved else {
        panic!("a first reservation names a counter: {reserved:?}");
    };
    let before = device.work_counts().queue_submissions;
    let stuck = queue.command_buffer().unwrap();
    stuck.encode_wait_for_event((&event).into(), 10).unwrap();
    for value in [3, 4] {
        let mut blit = stuck.blit_command_encoder().unwrap();
        blit.fill_buffer(&buffer, 0..4096, value).unwrap();
        blit.end_encoding().unwrap();
    }
    stuck.commit().unwrap();
    let deadline = Instant::now() + WAIT;
    let line = loop {
        let found = (heard.lock().unwrap().texts.iter())
            .find(|text| text.contains("encoder trail"))
            .cloned();
        if let Some(line) = found {
            break line;
        }
        assert!(Instant::now() < deadline, "no trail line in {WAIT:?}");
        std::thread::sleep(Duration::from_millis(50));
    };
    // SAFETY: the event's semaphore, signalled once from the host to the reserved counter, which
    // nothing before it waits for.
    let semaphore = unsafe { event.vk_semaphore() };
    let info = vk::SemaphoreSignalInfo::default()
        .semaphore(semaphore)
        .value(counter);
    // SAFETY: as above; the device is live.
    unsafe { device.shared().vk().signal_semaphore(&info) }.unwrap();
    assert!(stuck.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(
        (stuck.status(), stuck.error()),
        (CommandBufferStatus::Completed, None)
    );
    let submitted = device.work_counts().queue_submissions - before;
    eprintln!(
        "ENCODER-TRAIL {}: {submitted} submissions for the stuck command buffer's 2 encoders; {line}",
        device.name()
    );
    let stuck_id = stuck.object_id().get();
    let oldest = format!("oldest pending batch: Blit encoder 0 of command buffer {stuck_id}");
    assert!(line.contains(&oldest), "{line}");
    assert!(
        !line.contains(&format!("encoder 1 of command buffer {stuck_id}")),
        "the second encoder went in the first's batch: {line}"
    );
    let done_id = done.object_id().get();
    let last = format!("the last batch it completed: Blit encoder 0 of command buffer {done_id}");
    assert!(line.contains(&last), "{line}");
    assert!(submitted >= 2, "{submitted} submissions for 2 encoders");
}
