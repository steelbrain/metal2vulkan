//! Device tests of a command buffer's GPU and kernel times on the device the cfg selects, under the
//! validation layer: Metal's `GPUStartTime`, `GPUEndTime`, `kernelStartTime` and `kernelEndTime`,
//! in the host's time base (`runtime::host_time`).
//!
//! Each test commits three command buffers (one with no work, one fill, and a fill on each side
//! of an event's signal and wait, two batches), each reading all four times as 0 before its
//! commit; once it completed, the kernel times lie in order between the host times around its
//! commit and wait, and the GPU times in order no earlier than the kernel start and no later than
//! the host time after the wait, within [`SLACK`]. The venue's timestamp facts (the runtime
//! family's `timestampValidBits`, `timestampPeriod`, the calibrated timestamps extension it lists
//! and the time domains it offers) and each command buffer's GPU duration, kernel span and wall
//! time are printed: report only.
//!
//! The created device enables calibrated timestamps where listed: where it can calibrate (the
//! device's own domain and a monotonic host one), it announces nothing, and the calibration pairs
//! that placed the times stated deviations within [`SLACK`], each the least of two to four
//! readings; elsewhere it announces `Degradation::UncalibratedGpuTimes` once. So does a calibrating
//! device whose pairs placed a command buffer's times outside its life
//! (`WorkCounts::calibration_disagreements`, printed): its timestamps count on another clock than
//! the one it calibrates, as a Radeon 780M's after two MODE2 GPU resets, 65 s ahead, and those
//! command buffers are timed by their completion. The adopted device names no calibrated
//! timestamps, so it times every command buffer by its completion on every venue.
//!
//! A further test holds [`IN_FLIGHT`] command buffers in flight at once, more than Metal keeps
//! counter sample buffers live, and times each: the device's first timestamp pool serves them
//! all. Another holds more than one pool has slots ([`POOL_SLOTS`] each): where the driver's
//! pools are not Metal counter sample buffers, more than [`HELD_POOLS`] pools hold, every one
//! timed from pools made as the command buffers in flight need them, and nothing announced; where
//! they are (one pool), one more than it holds, every one but the last timed, the last untimed, its
//! GPU times refused, and the device announces `Degradation::UntimedCommandBuffer` once. The first
//! one's wait returns a few submissions into the rest, not once all are submitted. It prints how
//! long the commits and the completions took (`GPU-TIMES-HELD`). Each of their command buffers
//! has an encoder (an empty blit encoder): one with none writes no timestamps.
//!
//! Command buffers committed one after another, each once the one before it completed, take
//! the slot the one before gave back: each is still timed inside its own commit and wait, after
//! the one before it, its start written by its fill's own command buffer as its encoder opened
//! where the device's timestamp pools are not Metal counter sample buffers and its batches need no
//! command buffer of barriers for completed work, else by kept command buffers that reset the
//! slot and write its timestamps (`command_buffer/kept_barriers.rs`). A
//! fill committed behind another still running, the same bytes, takes the start a command buffer
//! of barriers writes after the barrier its batch plans, and starts no earlier than the first
//! ended, as Metal's does.
//!
//! A command buffer with no encoder and no wait writes no timestamps and answers where its queue
//! reached it as both its GPU times, as Metal answers for one: its kernel end time, or the GPU
//! end of the queue's command buffer before it where later. It takes no slot of the device's
//! timestamp pool (`Device::work_counts`), where one with an encoder, or a wait, takes one. On a
//! device made with `gpu_times_from_host`, none takes a slot, the GPU times of one with work are
//! its kernel end (or its queue's latest end) and its completion's host time, in Metal's order, one
//! after another on its queue, and the device announces `Degradation::GpuTimesFromHost` once.
//!
//! A driver whose timestamps wait for every queue's (`DriverFact::TimestampsWaitForEveryQueue`:
//! MoltenVK 1.4.1 on Apple's GPUs) answers from the host's clock with the default options too:
//! there the tests of the timestamp pool's slots check that no command buffer takes one, and
//! `host_clock_gpu_times_keep_metal_s_rules` checks the four rules of host-clock times
//! (`gpu_time.rs`) on the runtime's answers and, on a Mac's GPU, on Metal's own in the same test:
//! the same clock, each command buffer's life in order up to the moment its completed handler runs,
//! and its queue's command buffers in order (Metal's overlaps counted and printed; the runtime's
//! none).

use super::commit_work::orders_by_ends;
use super::degrade_notices::{assert_once, counts, FROM_HOST, GPU_TIMES, UNTIMED};
use super::presenting::Adopted;
use super::{validated_device, validated_device_with, Messages, ScratchCache};
use crate::gpu::TEST_DEVICE;
use ash::vk;
use metal2vulkan::device::{BatchOrderPolicy, DriverFact};
use metal2vulkan::error::{Feature, RuntimeDriverDefect};
use metal2vulkan_runtime::{
    host_time, CommandBuffer, CommandQueue, Degradation, Device, Error, EventRef, ResourceOptions,
    SharedEvent,
};
use std::collections::BTreeSet;
use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

/// The longest a command buffer here may take.
const WAIT: Duration = Duration::from_secs(60);

/// How far, in seconds, a GPU time may lie outside the host times around it: a calibrated
/// time's error (the calibration's deviation, and the clocks' drift over the life of a pair,
/// `gpu_time::REFRESH`) is microseconds on the devices measured. 20 us (1 ms before): every time on
/// lavapipe, RADV, ANV and the RTX 3090 lay 36 us or more inside its window, so the
/// bound only covers the calibration's own error.
const SLACK: f64 = 20e-6;

/// The command buffers held in flight at once by
/// [`more_command_buffers_in_flight_than_metal_keeps_sample_buffers_are_timed`]: more than the 32
/// counter sample buffers Metal keeps live per GPU (measured through MoltenVK 1.4.1 on an M3 Max
/// and an M4), each of which a timestamp query pool is on MoltenVK.
const IN_FLIGHT: usize = 40;

/// The command buffers [`command_buffers_one_after_another_are_each_timed_on_their_own`] commits,
/// one after another.
const ONE_AFTER_ANOTHER: usize = 8;

/// The slots of each of a device's timestamp pools (the runtime's `gpu_time::SLOTS`): the command
/// buffers in flight one pool times at once.
const POOL_SLOTS: usize = 2048;

/// The pools whose slots, and one more command buffer,
/// [`command_buffers_past_a_pool_s_slots_are_timed_by_pools_made_as_needed`] holds in flight
/// where the driver's pools are not Metal counter sample buffers: the eight a device was first
/// bounded at, 16385 command buffers, which took about 55 s to commit over 261 queues on RADV and
/// ANV before the scheduler asked only the heads that go.
const HELD_POOLS: usize = 8;

/// The command buffers one queue of
/// [`command_buffers_past_a_pool_s_slots_are_timed_by_pools_made_as_needed`] holds in flight at most:
/// within the 64 uncompleted command buffers Metal's default queue takes before `commandBuffer`
/// waits.
const PER_QUEUE: usize = 63;

/// The bytes each fill writes.
const LENGTH: u64 = 1 << 20;

/// The pairs [`a_start_behind_a_planned_barrier_follows_the_work_before_it`] commits.
const BEHIND_ROUNDS: usize = 4;

/// The bytes each of its fills writes: enough that the second fill's submission, made in the same
/// pass of the scheduler as the first's, finds the first still running.
const BEHIND_LENGTH: u64 = 64 << 20;

/// What the venue states of its timestamps.
struct Venue {
    bits: u32,
    period: f32,
    /// The calibrated timestamps extension it lists, `KHR` before `EXT`.
    extension: Option<CString>,
    /// The time domains it offers for calibration, raw.
    domains: Vec<i32>,
}

impl Venue {
    /// The venue of `device`'s Vulkan objects.
    fn of(device: &Device) -> Venue {
        // SAFETY: read while the device lives; only instance-level queries of its physical
        // device are made (none the loader serves through the device's word).
        let raw = unsafe { device.raw() };
        let (instance, physical) = (raw.instance, raw.physical_device);
        // SAFETY: a physical device of the instance.
        let families = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let bits = families[raw.queue_family as usize].timestamp_valid_bits;
        // SAFETY: as above.
        let period = unsafe { instance.get_physical_device_properties(physical) }
            .limits
            .timestamp_period;
        // SAFETY: as above.
        let listed = unsafe { instance.enumerate_device_extension_properties(physical) }
            .expect("the device's extensions");
        let lists = |name: &CStr| {
            listed
                .iter()
                .any(|extension| extension.extension_name_as_c_str() == Ok(name))
        };
        let extension = [
            vk::KHR_CALIBRATED_TIMESTAMPS_NAME,
            vk::EXT_CALIBRATED_TIMESTAMPS_NAME,
        ]
        .into_iter()
        .find(|name| lists(name))
        .map(CStr::to_owned);
        let domains = match &extension {
            Some(name) if name.as_c_str() == vk::KHR_CALIBRATED_TIMESTAMPS_NAME => {
                let functions = ash::khr::calibrated_timestamps::Instance::new(raw.entry, instance);
                // SAFETY: the device lists the extension, whose command this is.
                unsafe { functions.get_physical_device_calibrateable_time_domains(physical) }
            }
            Some(_) => {
                let functions = ash::ext::calibrated_timestamps::Instance::new(raw.entry, instance);
                // SAFETY: as above.
                unsafe { functions.get_physical_device_calibrateable_time_domains(physical) }
            }
            None => Ok(Vec::new()),
        }
        .expect("the calibrateable time domains")
        .into_iter()
        .map(vk::TimeDomainKHR::as_raw)
        .collect();
        Venue {
            bits,
            period,
            extension,
            domains,
        }
    }

    /// Whether a device that enables the extension calibrates: the device's own domain and
    /// `CLOCK_MONOTONIC` or `CLOCK_MONOTONIC_RAW` (`gpu_time::host_domain`).
    fn calibrates(&self) -> bool {
        let offers = |domain: vk::TimeDomainKHR| self.domains.contains(&domain.as_raw());
        self.extension.is_some()
            && offers(vk::TimeDomainKHR::DEVICE)
            && (offers(vk::TimeDomainKHR::CLOCK_MONOTONIC)
                || offers(vk::TimeDomainKHR::CLOCK_MONOTONIC_RAW))
    }

    fn print(&self, device: &str) {
        eprintln!(
            "runtime gpu times: {device}: {} timestamp bits, {} ns a tick, extension {}, \
             domains {:?}, calibrates {}",
            self.bits,
            self.period,
            self.extension
                .as_deref()
                .map_or("none".into(), CStr::to_string_lossy),
            self.domains,
            self.calibrates()
        );
    }
}

/// Whether `device`'s driver answers GPU times from the host's clock with the default options
/// (`DriverFact::TimestampsWaitForEveryQueue`), writing no timestamps.
fn from_host_by_driver(device: &Device) -> bool {
    device
        .properties()
        .driver_facts
        .contains(&DriverFact::TimestampsWaitForEveryQueue)
}

/// Whether a timed command buffer's first encoder writes its start on `device` (`gpu_time.rs`):
/// where its timestamp pools are not Metal counter sample buffers
/// (`DriverFact::TimestampPoolsAreCounterSampleBuffers`) and its batches need no command buffer of
/// barriers for completed work (`DriverFact::SemaphoreOperationsStallSubmits`).
fn starts_in_the_first_encoder(device: &Device) -> bool {
    let facts = &device.properties().driver_facts;
    !facts.contains(&DriverFact::TimestampPoolsAreCounterSampleBuffers)
        && !facts.contains(&DriverFact::SemaphoreOperationsStallSubmits)
}

/// The kind a read of a timed command buffer's GPU times announces on `device`, made with the
/// default options, as `command_buffers_are_timed_in_the_hosts_time_base` checks: [`FROM_HOST`]'s
/// where its driver answers them from the host, [`GPU_TIMES`]' where it writes timestamps it does
/// not calibrate or its calibration disagreed with them so far
/// (`WorkCounts::calibration_disagreements`), none elsewhere (a calibrating device, or one that
/// writes no timestamps and refuses the read).
pub(super) fn read_kinds(device: &Device) -> BTreeSet<Degradation> {
    if from_host_by_driver(device) {
        return BTreeSet::from(FROM_HOST);
    }
    let venue = Venue::of(device);
    let disagreed = device.work_counts().calibration_disagreements > 0;
    match venue.bits > 0 && (!venue.calibrates() || disagreed) {
        true => BTreeSet::from(GPU_TIMES),
        false => BTreeSet::new(),
    }
}

/// The kind a read of a timed command buffer's GPU times may announce on `device` once, or not,
/// where [`read_kinds`] does not already ask for it: a device that writes timestamps and
/// calibrates them can still place one command buffer's times outside its life, which the
/// runtime finds at the read (`WorkCounts::calibration_disagreements`) and announces then. A test
/// whose own work reads GPU times cannot know in advance whether one of its command buffers
/// disagrees, so it allows the kind rather than requiring it (the RX 7900 XTX under RADV
/// disagreed in `first_use`'s busy work and not in the test beside it).
pub(super) fn disagreeable_kinds(device: &Device) -> BTreeSet<Degradation> {
    let venue = Venue::of(device);
    let may = !from_host_by_driver(device) && venue.bits > 0 && venue.calibrates();
    match may {
        true => BTreeSet::from(GPU_TIMES),
        false => BTreeSet::new(),
    }
}

/// `cb`'s four times: GPU start and end, kernel start and end.
fn times(cb: &CommandBuffer) -> [f64; 4] {
    [
        cb.gpu_start_time().unwrap(),
        cb.gpu_end_time().unwrap(),
        cb.kernel_start_time(),
        cb.kernel_end_time(),
    ]
}

/// Commit `cb`, labelled `label`, wait for it, and check its times against the host's around it
/// (the module docs); its GPU duration, kernel span and wall time, printed.
fn timed(label: &str, cb: &CommandBuffer) {
    assert_eq!(times(cb), [0.0; 4], "{label}: before its commit");
    let before = host_time();
    cb.commit().unwrap();
    cb.wait_until_completed_for(WAIT).unwrap();
    within(label, cb, before, host_time());
}

/// Check `cb`'s times, labelled `label`, against the host times `before` its commit and `after`
/// its wait returned (the module docs); its GPU duration, kernel span and wall time, printed.
fn within(label: &str, cb: &CommandBuffer, before: f64, after: f64) {
    assert_eq!(cb.error(), None, "{label}");
    let read = times(cb);
    let [gpu_start, gpu_end, kernel_start, kernel_end] = read;
    let around = format!("{label}: {read:?} between {before} and {after}");
    assert!(
        before <= kernel_start && kernel_start <= kernel_end && kernel_end <= after,
        "{around}"
    );
    assert!(gpu_start > 0.0 && gpu_start <= gpu_end, "{around}");
    assert!(
        gpu_start >= kernel_start - SLACK && gpu_end <= after + SLACK,
        "{around}"
    );
    // Read again: the same answers.
    assert_eq!(times(cb), read, "{label}");
    eprintln!(
        "runtime gpu times: {label}: gpu {:.1} us, kernel {:.1} us, wall {:.1} us, gpu start \
         {:+.1} us after the kernel start, gpu end {:+.1} us before the wait returned",
        (gpu_end - gpu_start) * 1e6,
        (kernel_end - kernel_start) * 1e6,
        (after - before) * 1e6,
        (gpu_start - kernel_start) * 1e6,
        (after - gpu_end) * 1e6,
    );
}

/// An empty blit encoder on `cb`: an encoder, so `cb` writes timestamps.
fn encoder(cb: &CommandBuffer) {
    cb.blit_command_encoder().unwrap().end_encoding().unwrap();
}

/// The three command buffers of the module docs on `device`'s queue `queue`, labelled by `on`.
fn three_command_buffers(device: &Device, queue: &CommandQueue, on: &str) {
    let buffer = device
        .new_buffer(LENGTH, ResourceOptions::private())
        .unwrap();
    timed(&format!("{on}, no work"), &queue.command_buffer().unwrap());

    let fill = queue.command_buffer().unwrap();
    let mut blit = fill.blit_command_encoder().unwrap();
    blit.fill_buffer(&buffer, 0..LENGTH, 0x5a).unwrap();
    blit.end_encoding().unwrap();
    timed(&format!("{on}, a fill"), &fill);

    let event = device.new_event().unwrap();
    let batches = queue.command_buffer().unwrap();
    let mut blit = batches.blit_command_encoder().unwrap();
    blit.fill_buffer(&buffer, 0..LENGTH, 0x11).unwrap();
    blit.end_encoding().unwrap();
    batches
        .encode_signal_event(EventRef::from(&event), 1)
        .unwrap();
    batches
        .encode_wait_for_event(EventRef::from(&event), 1)
        .unwrap();
    let mut blit = batches.blit_command_encoder().unwrap();
    blit.fill_buffer(&buffer, 0..LENGTH, 0x22).unwrap();
    blit.end_encoding().unwrap();
    timed(&format!("{on}, two batches"), &batches);
}

/// On the device the runtime creates, with calibrated timestamps where listed: the three command
/// buffers' times, and the notice where the device does not calibrate. A device whose family
/// writes no timestamps refuses the GPU times, typed.
#[test]
fn command_buffers_are_timed_in_the_hosts_time_base() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let venue = Venue::of(&device);
        venue.print(device.name());
        let queue = device.new_command_queue().unwrap();
        let from_host = from_host_by_driver(&device);
        if venue.bits == 0 && !from_host {
            let cb = queue.command_buffer().unwrap();
            match cb.gpu_start_time() {
                Err(Error::Unsupported(unsupported)) => {
                    assert_eq!(unsupported.feature, Feature::GpuTimes);
                }
                other => panic!("GPU times without timestamps: {other:?}"),
            }
            return;
        }
        three_command_buffers(&device, &queue, "created");
        let notices = messages.take_notices();
        if from_host {
            assert_once(&notices, &FROM_HOST);
            assert_eq!(counts(&notices).len(), 1, "{:?}", counts(&notices));
        } else if venue.calibrates() {
            match device.work_counts().calibration_disagreements {
                0 => assert_eq!(counts(&notices), Default::default(), "a calibrated device"),
                _ => {
                    assert_once(&notices, &GPU_TIMES);
                    assert_eq!(counts(&notices).len(), 1, "{:?}", counts(&notices));
                }
            }
            calibrated_within_slack(&device);
        } else {
            assert_once(&notices, &GPU_TIMES);
        }
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// On a `device` that calibrates: the calibration pairs that placed its command buffers' times,
/// each the reading of the smallest deviation of two to four (`WorkCounts::calibration_readings`),
/// stated deviations within [`SLACK`] (`WorkCounts::calibration_deviation_most`): on an Iris Xe
/// (ANV, i915), the first reading after the GPU idled stated 165 us at the median and placed every
/// time 156 us early, the next 7.6 us; a reading of a thread preempted inside it states the time
/// it lost, and the pair reads again (`gpu_time.rs`).
fn calibrated_within_slack(device: &Device) {
    let counts = device.work_counts();
    let (pairs, readings, most, disagreed) = (
        counts.calibrations,
        counts.calibration_readings,
        counts.calibration_deviation_most,
        counts.calibration_disagreements,
    );
    eprintln!(
        "runtime gpu times: {}: {pairs} calibration pairs of {readings} readings, the largest \
         deviation {:.1} us; {disagreed} command buffers whose pair placed their times outside \
         their life, timed by their completion",
        device.name(),
        most as f64 / 1e3
    );
    assert!(pairs > 0, "no calibration pair taken");
    assert!(
        (2 * pairs..=4 * pairs).contains(&readings),
        "{readings} readings of {pairs} pairs: two to four each"
    );
    assert!(
        most as f64 <= SLACK * 1e9,
        "a calibration pair's deviation {most} ns, beyond the slack"
    );
}

/// [`ONE_AFTER_ANOTHER`] command buffers on one queue, each a fill committed once the one before it
/// completed, each taking the slot the one before gave back: each is timed inside its own commit
/// and wait, and starts no earlier than the one before it ended, so none reads a timestamp an
/// earlier command buffer wrote into the slot. Where the first encoder writes the start
/// ([`starts_in_the_first_encoder`]), each fill's own command buffer wrote its start as its
/// encoder opened, and its batch began with no command buffer of barriers
/// (`WorkCounts::prefixed_starts`, one each; no stamped command buffer recorded or taken again).
/// Elsewhere, where the device's command buffers end at their timeline's value, the starts come
/// from kept command buffers,
/// recorded once and taken again by the command buffers that hold the same slot later
/// (`command_buffer/kept_barriers.rs`, `WorkCounts::stamped_buffers_kept`,
/// `WorkCounts::stamped_buffers_reused`, printed; at least one taken again). Each one's batch
/// ends in its fill's own command buffer, the end timestamp written there
/// (`WorkCounts::finales_folded`, one each, and no end apart). The validation layer reported
/// nothing.
#[test]
fn command_buffers_one_after_another_are_each_timed_on_their_own() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let venue = Venue::of(&device);
        if venue.bits == 0 || from_host_by_driver(&device) {
            eprintln!(
                "runtime gpu times: {}: writes no timestamps, nothing to check",
                device.name()
            );
            return;
        }
        let queue = device.new_command_queue().unwrap();
        let buffer = device
            .new_buffer(LENGTH, ResourceOptions::private())
            .unwrap();
        let before = device.work_counts();
        let mut last_end = 0.0;
        for n in 0..ONE_AFTER_ANOTHER {
            let cb = queue.command_buffer().unwrap();
            let mut blit = cb.blit_command_encoder().unwrap();
            blit.fill_buffer(&buffer, 0..LENGTH, n as u8).unwrap();
            blit.end_encoding().unwrap();
            let label = format!("command buffer {n} of {ONE_AFTER_ANOTHER}");
            timed(&label, &cb);
            let [gpu_start, gpu_end, ..] = times(&cb);
            assert!(
                gpu_start >= last_end - SLACK,
                "{label}: starts at {gpu_start}, before the one before it ended at {last_end}"
            );
            last_end = gpu_end;
        }
        let after = device.work_counts();
        let kept = after.stamped_buffers_kept - before.stamped_buffers_kept;
        let reused = after.stamped_buffers_reused - before.stamped_buffers_reused;
        let prefixed = after.prefixed_starts - before.prefixed_starts;
        let folded = after.finales_folded - before.finales_folded;
        let apart = after.finales_apart - before.finales_apart;
        assert_eq!(
            (folded, apart),
            (ONE_AFTER_ANOTHER as u64, 0),
            "the batches' ends folded into their fills, and apart"
        );
        eprintln!(
            "runtime gpu times: {}: {ONE_AFTER_ANOTHER} command buffers one after another: \
             {prefixed} started by their fill's own command buffer, {kept} stamped barrier \
             command buffers recorded to keep, {reused} taken again",
            device.name()
        );
        let at_fence = (device.properties().driver_defects)
            .contains(&RuntimeDriverDefect::ReadsCommandBuffersAfterTheirSignal);
        if starts_in_the_first_encoder(&device) {
            assert_eq!(
                (prefixed, kept, reused),
                (ONE_AFTER_ANOTHER as u64, 0, 0),
                "each start written by its fill's own command buffer"
            );
        } else if at_fence {
            assert_eq!((prefixed, kept, reused), (0, 0, 0));
        } else {
            // A start each, kept or taken again (the end is in the fill's command buffer); one
            // the scheduler found still pending is recorded by its batch.
            assert_eq!(prefixed, 0);
            assert!(kept + reused <= ONE_AFTER_ANOTHER as u64);
            assert!(reused > 0, "no kept stamped command buffer taken again");
        }
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// [`BEHIND_ROUNDS`] pairs of fills of the same [`BEHIND_LENGTH`] bytes on one queue, the first
/// waiting for a shared event the host signals once both are committed: the scheduler submits
/// both in one pass, and the second's batch plans a barrier before its encoder, for the first's
/// writes, still running. So its start is not its fill's own: a command buffer of barriers writes
/// it after that barrier, once every command before it completed (`command_buffer/submit.rs`),
/// and the start its fill wrote as it opened is never read. The first, with nothing planned
/// before its fill, starts in its fill's own command buffer, after the event's wait. Each is timed
/// inside its commit and wait, the second ends no earlier than the first, and where the device
/// calibrates its timestamps (each time its own) the second starts no earlier than the first
/// ended, as Metal starts a command buffer behind the work it depends on (probe `gpustart`, an
/// M3 Max under macOS 27.0.1: 0.5 to 0.9 us after it, 10 of 10). Where the first encoder writes
/// the start ([`starts_in_the_first_encoder`]), every first started in its fill's command buffer
/// and at least one second did not (`WorkCounts::prefixed_starts`, printed); elsewhere none did.
/// The validation layer reported nothing. Under `BatchOrderPolicy::Exact`; the default's is
/// [`a_start_behind_work_of_batches_ordered_by_their_ends_follows_it`].
#[test]
fn a_start_behind_a_planned_barrier_follows_the_work_before_it() {
    start_behind_work(BatchOrderPolicy::Exact);
}

/// The same under the default. Where the device orders batches by their ends (the runtime's
/// `batch_order.rs`), the second plans no barrier for the first's writes, as the first's end
/// orders it, and its fill's own command buffer writes its start at the bottom of the pipe, once
/// every command before it completed: every first and every second started in its fill's command
/// buffer where the first encoder writes the start, and each second no earlier than the first
/// ended. Elsewhere as under `BatchOrderPolicy::Exact`.
#[test]
fn a_start_behind_work_of_batches_ordered_by_their_ends_follows_it() {
    start_behind_work(BatchOrderPolicy::Ends);
}

/// The two tests above under `batch_order`.
fn start_behind_work(batch_order: BatchOrderPolicy) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.batch_order = batch_order;
        })
        .unwrap();
        let by_ends = orders_by_ends(&device, batch_order);
        let venue = Venue::of(&device);
        if venue.bits == 0 || from_host_by_driver(&device) {
            eprintln!(
                "runtime gpu times: {}: writes no timestamps, nothing to check",
                device.name()
            );
            return;
        }
        let calibrates = venue.calibrates();
        let queue = device.new_command_queue().unwrap();
        let buffer = device
            .new_buffer(BEHIND_LENGTH, ResourceOptions::private())
            .unwrap();
        let fill = |byte: u8| {
            let cb = queue.command_buffer().unwrap();
            let mut blit = cb.blit_command_encoder().unwrap();
            blit.fill_buffer(&buffer, 0..BEHIND_LENGTH, byte).unwrap();
            blit.end_encoding().unwrap();
            cb
        };
        // The first takes any first work the device's creation left, and the buffer's.
        let warm = fill(0);
        warm.commit().unwrap();
        warm.wait_until_completed_for(WAIT).unwrap();
        let before_rounds = device.work_counts();
        for round in 0..BEHIND_ROUNDS {
            let event: SharedEvent = device.new_shared_event().unwrap();
            let first = queue.command_buffer().unwrap();
            first
                .encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            let mut blit = first.blit_command_encoder().unwrap();
            blit.fill_buffer(&buffer, 0..BEHIND_LENGTH, 2 * round as u8 + 1)
                .unwrap();
            blit.end_encoding().unwrap();
            let second = fill(2 * round as u8 + 2);
            let before = host_time();
            first.commit().unwrap();
            second.commit().unwrap();
            event.set_signaled_value(1);
            first.wait_until_completed_for(WAIT).unwrap();
            second.wait_until_completed_for(WAIT).unwrap();
            let after = host_time();
            within(&format!("round {round}, the first"), &first, before, after);
            within(
                &format!("round {round}, the second"),
                &second,
                before,
                after,
            );
            let ([_, first_end, ..], [second_start, second_end, ..]) =
                (times(&first), times(&second));
            // Each its own timestamp where the device calibrates; else placed by its completion,
            // late by the completion turn's latency, which differs between the two.
            if calibrates && device.work_counts().calibration_disagreements == 0 {
                assert!(
                    second_start >= first_end - SLACK,
                    "round {round}: the second started at {second_start}, before the first \
                     ended at {first_end}"
                );
            }
            assert!(
                second_end >= first_end - SLACK,
                "round {round}: the second ended at {second_end}, before the first at {first_end}"
            );
        }
        let after_rounds = device.work_counts();
        let prefixed = after_rounds.prefixed_starts - before_rounds.prefixed_starts;
        eprintln!(
            "runtime gpu times: {} {batch_order:?}: {BEHIND_ROUNDS} pairs, the second behind the \
             first: {prefixed} started by their fill's own command buffer",
            device.name()
        );
        match (starts_in_the_first_encoder(&device), by_ends) {
            (false, _) => assert_eq!(prefixed, 0),
            (true, true) => assert_eq!(
                prefixed,
                2 * BEHIND_ROUNDS as u64,
                "started by their fill's command buffer: every first and every second"
            ),
            (true, false) => assert!(
                (BEHIND_ROUNDS as u64..2 * BEHIND_ROUNDS as u64).contains(&prefixed),
                "{prefixed} started by their fill's command buffer: every first, and not every \
                 second"
            ),
        }
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// On a device the test made and the runtime adopts without calibrated timestamps: the three
/// command buffers' times, each ending at its completion.
#[test]
fn an_adopted_device_without_calibration_times_by_completion() {
    let adopted = Adopted::open(false).unwrap();
    let cache = ScratchCache::new().unwrap();
    {
        let device = adopted.runtime(cache.path());
        let venue = Venue::of(&device);
        if venue.bits > 0 {
            let queue = device.new_command_queue().unwrap();
            three_command_buffers(&device, &queue, "adopted");
        }
    }
    let reported = adopted.close();
    assert!(
        reported.is_empty(),
        "the validation layer reported: {reported:#?}"
    );
}

/// [`IN_FLIGHT`] command buffers, each of its own queue and waiting for one shared event, are
/// committed before the event is signaled, so all are in flight at once. Each is timed, its GPU
/// times in order and its end no earlier than the signal, and the validation layer reported
/// nothing. With a timestamp query pool per command buffer, MoltenVK could not make the 33rd's
/// counter sample buffer and reported it.
#[test]
fn more_command_buffers_in_flight_than_metal_keeps_sample_buffers_are_timed() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        if Venue::of(&device).bits > 0 {
            let event: SharedEvent = device.new_shared_event().unwrap();
            let committed: Vec<CommandBuffer> = (0..IN_FLIGHT)
                .map(|_| {
                    let cb = device
                        .new_command_queue()
                        .unwrap()
                        .command_buffer()
                        .unwrap();
                    cb.encode_wait_for_event(EventRef::from(&event), 1).unwrap();
                    encoder(&cb);
                    cb.commit().unwrap();
                    cb
                })
                .collect();
            let signaled = host_time();
            event.set_signaled_value(1);
            for (at, cb) in committed.iter().enumerate() {
                cb.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(cb.error(), None, "{at}");
            }
            for (at, cb) in committed.iter().enumerate() {
                let [gpu_start, gpu_end, kernel_start, kernel_end] = times(cb);
                let read = format!("{at}: {:?}, signaled at {signaled}", times(cb));
                assert!(gpu_start > 0.0 && gpu_start <= gpu_end, "{read}");
                assert!(gpu_end >= signaled - SLACK, "{read}");
                assert!(kernel_start <= kernel_end, "{read}");
            }
            eprintln!(
                "runtime gpu times: {IN_FLIGHT} command buffers in flight at once, each timed"
            );
        }
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// Command buffers each waiting for one shared event are committed before it is signaled, so all
/// are in flight at once, round queues of at most [`PER_QUEUE`] each. Where the driver's pools
/// are not Metal counter sample buffers, [`HELD_POOLS`] pools' slots ([`POOL_SLOTS`] each) and
/// one more: every one is timed, past the first pool's slots of pools made as they needed them,
/// and nothing is announced but what reading GPU times announces there ([`read_kinds`]: none
/// where the device's calibration agreed with its timestamps). Where they are (`DriverFact::TimestampPoolsAreCounterSampleBuffers`,
/// one pool), one more than [`POOL_SLOTS`]: each but the last is timed, and the last found no
/// slot free and no pool more: its GPU times are refused (`Unsupported`, `Feature::GpuTimes`),
/// its kernel times kept, and the device announces `untimed-command-buffer` once, naming the
/// slots. The slots taken are the timed command buffers'. The first one's wait returns before
/// the device made as many submissions after the signal as there are command buffers: the
/// scheduler lets go of each few it submits (`scheduler/mod.rs`), where a pass that submitted all
/// of them would hold the first one's waiter until the last was submitted. It prints how long the
/// commits took, how long after the signal the first and every one had completed, and the
/// submissions made by then. The validation layer reported nothing.
#[test]
fn command_buffers_past_a_pool_s_slots_are_timed_by_pools_made_as_needed() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        // A device answering from the host's clock by its driver has no slots to run out of.
        if from_host_by_driver(&device) {
            eprintln!("runtime gpu times: the driver answers from the host's clock: no slots");
        } else if Venue::of(&device).bits > 0 {
            let scarce = device
                .properties()
                .driver_facts
                .contains(&DriverFact::TimestampPoolsAreCounterSampleBuffers);
            let count = POOL_SLOTS * if scarce { 1 } else { HELD_POOLS } + 1;
            let taken = device.work_counts().timestamp_slots_taken;
            let event: SharedEvent = device.new_shared_event().unwrap();
            let queues: Vec<CommandQueue> = (0..count.div_ceil(PER_QUEUE))
                .map(|_| device.new_command_queue().unwrap())
                .collect();
            // The longest one `commit` took, and the commits' time when one pool's slots and
            // one more were in flight.
            let (mut longest, mut at_one_pool) = (Duration::ZERO, Duration::ZERO);
            let started = Instant::now();
            let committed: Vec<CommandBuffer> = (0..count)
                .map(|at| {
                    let cb = queues[at % queues.len()].command_buffer().unwrap();
                    cb.encode_wait_for_event(EventRef::from(&event), 1).unwrap();
                    encoder(&cb);
                    let commit = Instant::now();
                    cb.commit().unwrap();
                    longest = longest.max(commit.elapsed());
                    if at == POOL_SLOTS {
                        at_one_pool = started.elapsed();
                    }
                    cb
                })
                .collect();
            let committing = started.elapsed();
            let submitted = device.work_counts().queue_submissions;
            let signaled = Instant::now();
            event.set_signaled_value(1);
            // The first one's wait returns once its work completed and the scheduler let go of
            // it, a few submissions into the rest (`scheduler/mod.rs`), not once all are.
            committed[0].wait_until_completed_for(WAIT).unwrap();
            let first = signaled.elapsed();
            let at_first = device.work_counts().queue_submissions - submitted;
            assert!(
                at_first < count as u64,
                "the first command buffer's wait returned {:.3} s after the signal, {at_first} \
                 submissions in, once all {count} command buffers were submitted",
                first.as_secs_f64()
            );
            for (at, cb) in committed.iter().enumerate() {
                cb.wait_until_completed_for(WAIT).unwrap();
                assert_eq!(cb.error(), None, "{at}");
            }
            let completing = signaled.elapsed();
            let at_all = device.work_counts().queue_submissions - submitted;
            let (timed, untimed) = committed.split_at(count - usize::from(scarce));
            for (at, cb) in timed.iter().enumerate() {
                let [gpu_start, gpu_end, ..] = times(cb);
                assert!(
                    gpu_start > 0.0 && gpu_start <= gpu_end,
                    "{at}: {gpu_start} {gpu_end}"
                );
            }
            for cb in untimed {
                for refused in [cb.gpu_start_time(), cb.gpu_end_time()] {
                    match refused {
                        Err(Error::Unsupported(unsupported)) => {
                            assert_eq!(unsupported.feature, Feature::GpuTimes);
                        }
                        other => panic!("the untimed command buffer's GPU time: {other:?}"),
                    }
                }
                let (kernel_start, kernel_end) = (cb.kernel_start_time(), cb.kernel_end_time());
                assert!(kernel_start > 0.0 && kernel_start <= kernel_end);
            }
            assert_eq!(
                device.work_counts().timestamp_slots_taken - taken,
                timed.len() as u64
            );
            let notices = messages.take_notices();
            if scarce {
                assert_once(&notices, &UNTIMED);
                let named = format!("{POOL_SLOTS} timestamp slots");
                assert!(
                    notices.iter().any(|notice| notice.text.contains(&named)),
                    "the notice names {named}: {notices:#?}"
                );
            } else {
                // What reading GPU times announces there, after the reads: nothing on a device
                // whose calibration agreed with its timestamps.
                let read: Vec<Degradation> = read_kinds(&device).into_iter().collect();
                assert_once(&notices, &read);
            }
            eprintln!(
                "GPU-TIMES-HELD {count} command buffers over {} queues on {}: committed in \
                 {:.3} s ({:.3} s for the first {}), the longest commit {:.3} ms, the first \
                 completed {:.3} s after the signal ({at_first} submissions in), all {:.3} s \
                 after it ({at_all} submissions in), {} untimed",
                queues.len(),
                device.name(),
                committing.as_secs_f64(),
                at_one_pool.as_secs_f64(),
                POOL_SLOTS + 1,
                longest.as_secs_f64() * 1e3,
                first.as_secs_f64(),
                completing.as_secs_f64(),
                untimed.len()
            );
        }
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// Command buffers with no encoder answer where their queue reached them as both GPU times and
/// take no slot of the device's timestamp pool: on their own, their kernel end time; each behind
/// a fill on its queue, the later of that and the fill's GPU end. Each fill, an encoder's work,
/// takes a slot, and so does a command buffer with no encoder that waits for an event (the
/// module docs).
#[test]
fn a_command_buffer_with_no_encoder_answers_where_its_queue_reached_it() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        if Venue::of(&device).bits > 0 {
            let queue = device.new_command_queue().unwrap();
            let buffer = device
                .new_buffer(LENGTH, ResourceOptions::private())
                .unwrap();
            let slots = || device.work_counts().timestamp_slots_taken;
            let taken = slots();
            // A slot per command buffer with work or a wait, none where the driver answers from
            // the host's clock.
            let per = u64::from(!from_host_by_driver(&device));
            for run in 0..8 {
                let cb = queue.command_buffer().unwrap();
                timed(&format!("no encoder {run}"), &cb);
                let [gpu_start, gpu_end, _, kernel_end] = times(&cb);
                assert_eq!([gpu_start, gpu_end], [kernel_end; 2], "no encoder {run}");
            }
            assert_eq!(
                slots(),
                taken,
                "command buffers with no encoder took timestamp slots"
            );
            for run in 0..8 {
                let fill = queue.command_buffer().unwrap();
                let mut blit = fill.blit_command_encoder().unwrap();
                blit.fill_buffer(&buffer, 0..LENGTH, run).unwrap();
                blit.end_encoding().unwrap();
                fill.commit().unwrap();
                let empty = queue.command_buffer().unwrap();
                empty.commit().unwrap();
                for cb in [&fill, &empty] {
                    cb.wait_until_completed_for(WAIT).unwrap();
                    assert_eq!(cb.error(), None, "behind a fill {run}");
                }
                let [fill_start, fill_end, ..] = times(&fill);
                assert!(0.0 < fill_start && fill_start <= fill_end, "fill {run}");
                let read = times(&empty);
                let [gpu_start, gpu_end, kernel_start, kernel_end] = read;
                let reached = kernel_end.max(fill_end);
                assert_eq!(
                    [gpu_start, gpu_end],
                    [reached; 2],
                    "behind a fill {run}: {read:?}, the fill's end {fill_end}"
                );
                assert!(0.0 < kernel_start && kernel_start <= kernel_end);
                eprintln!(
                    "runtime gpu times: no encoder behind a fill {run}: gpu start {:+.1} us after \
                     the fill's gpu end, {:+.1} us after its kernel end",
                    (gpu_start - fill_end) * 1e6,
                    (gpu_start - kernel_end) * 1e6,
                );
            }
            assert_eq!(
                slots(),
                taken + 8 * per,
                "each fill takes one timestamp slot"
            );
            let event: SharedEvent = device.new_shared_event().unwrap();
            event.set_signaled_value(1);
            let waiting = queue.command_buffer().unwrap();
            waiting
                .encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            timed("no encoder, a wait", &waiting);
            assert_eq!(
                slots(),
                taken + 9 * per,
                "a command buffer that waits takes a slot"
            );
        }
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// On a device made with `gpu_times_from_host`: the three command buffers' times hold the module
/// docs' facts; [`IN_FLIGHT`] command buffers committed in a row on a queue of their own, fills
/// with one in four encoding nothing, start exactly where the rule puts them: at the later of
/// their kernel end and the GPU end of the one before them (the first at its kernel end), one
/// encoding nothing ending where it starts (exact); no command buffer takes a timestamp slot; and
/// the device announces `gpu-times-from-host` once, and nothing else.
#[test]
fn a_device_answering_gpu_times_from_the_host_writes_no_timestamps() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.gpu_times_from_host = true;
        })
        .unwrap();
        let queue = device.new_command_queue().unwrap();
        three_command_buffers(&device, &queue, "from the host");
        // A queue of their own: no command buffer before the first.
        let queue = device.new_command_queue().unwrap();
        let buffer = device
            .new_buffer(LENGTH, ResourceOptions::private())
            .unwrap();
        let fills: Vec<CommandBuffer> = (0..IN_FLIGHT)
            .map(|at| {
                let cb = queue.command_buffer().unwrap();
                if at % 4 != 3 {
                    let mut blit = cb.blit_command_encoder().unwrap();
                    blit.fill_buffer(&buffer, 0..LENGTH, at as u8).unwrap();
                    blit.end_encoding().unwrap();
                }
                cb.commit().unwrap();
                cb
            })
            .collect();
        for (at, cb) in fills.iter().enumerate() {
            cb.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(cb.error(), None, "{at}");
        }
        let read: Vec<[f64; 4]> = fills.iter().map(times).collect();
        let mut before_end = 0.0f64;
        for (at, &[gpu_start, gpu_end, kernel_start, kernel_end]) in read.iter().enumerate() {
            assert!(
                0.0 < kernel_start && kernel_start <= kernel_end,
                "{at}: {:?}",
                read[at]
            );
            // The later of its kernel end and the queue's latest GPU end, exactly.
            assert_eq!(
                gpu_start,
                kernel_end.max(before_end),
                "{at}: {:?} after a GPU end of {before_end}",
                read[at]
            );
            match at % 4 == 3 {
                true => assert_eq!(gpu_end, gpu_start, "{at}: encoded nothing"),
                false => assert!(gpu_start <= gpu_end, "{at}: {:?}", read[at]),
            }
            before_end = gpu_end;
        }
        for (at, pair) in read.windows(2).enumerate() {
            let ([start, end, ..], [next_start, next_end, ..]) = (pair[0], pair[1]);
            assert!(
                start <= next_start && end <= next_end && end <= next_start,
                "{at} and the next: {pair:?}"
            );
        }
        assert_eq!(device.work_counts().timestamp_slots_taken, 0);
        let notices = messages.take_notices();
        assert_once(&notices, &FROM_HOST);
        assert_eq!(counts(&notices).len(), 1, "{:?}", counts(&notices));
        eprintln!(
            "runtime gpu times: {IN_FLIGHT} command buffers on one queue, answered from the host"
        );
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// The command buffers each queue of [`host_clock_gpu_times_keep_metal_s_rules`] commits, in a
/// row, interleaved with the other queue's: within the 64 uncompleted a queue takes.
const IN_A_ROW: usize = 24;

/// What one command buffer of the rules test answered: the host time just before its commit,
/// the host time its completed handler began, and its four times read there (GPU start and end,
/// kernel start and end).
#[derive(Clone, Copy, Debug, Default)]
struct Record {
    commit: f64,
    handler: f64,
    times: [f64; 4],
}

/// The bytes the `at`-th command buffer of a row fills: a small fill, a large one, none (no
/// encoder), and two fills in two encoders, in turn.
fn row_fills(at: usize) -> &'static [u64] {
    match at % 4 {
        0 => &[256],
        1 => &[4 << 20],
        2 => &[],
        _ => &[256, 1 << 20],
    }
}

/// Where `queues`' records (each queue's in commit order) break the rules of host-clock GPU
/// times (`gpu_time.rs`) that hold on any serial queue: every time positive; the commit, kernel
/// start, kernel end, GPU start, GPU end and the completed handler's start in that order, each
/// time seconds of `host_time`'s clock; and a queue's GPU starts and ends never going back,
/// among each queue's command buffers with an encoder, or every one where `every`. Metal answers
/// an encoder-less command buffer a zero window at a time after the next command buffer of its
/// queue ran (6 of 12 such went back), so its order is the order of the work
/// that ran; the runtime places every one in its queue's order. Where `late_kernel_ends`, a
/// kernel end after the GPU start (and end) breaks no rule, the kernel end still between the
/// kernel start and the handler: Metal answers so on macOS 27 ([`LATE_KERNEL_END`]). The broken
/// rules; the command buffers so ordered that started before the one before them on their queue
/// ended; and those whose kernel end came after their GPU start.
fn broken_rules(
    queues: &[Vec<Record>],
    every: bool,
    late_kernel_ends: bool,
) -> (Vec<String>, usize, usize) {
    let mut broken = Vec::new();
    let mut overlaps = 0;
    let mut late = 0;
    let in_order = |times: &[f64]| times.windows(2).all(|w| w[0] <= w[1]);
    for (queue, records) in queues.iter().enumerate() {
        for (at, record) in records.iter().enumerate() {
            let [gpu_start, gpu_end, kernel_start, kernel_end] = record.times;
            let (commit, handler) = (record.commit, record.handler);
            let strict = [
                commit,
                kernel_start,
                kernel_end,
                gpu_start,
                gpu_end,
                handler,
            ];
            let ordered = match late_kernel_ends {
                false => in_order(&strict),
                true => {
                    in_order(&[commit, kernel_start, gpu_start, gpu_end, handler])
                        && in_order(&[kernel_start, kernel_end, handler])
                }
            };
            if record.times.iter().any(|&time| time <= 0.0) || !ordered {
                broken.push(format!("queue {queue}, {at}: out of order: {record:?}"));
            } else if kernel_end > gpu_start {
                late += 1;
            }
        }
        let ordered: Vec<(usize, &Record)> = records
            .iter()
            .enumerate()
            .filter(|&(at, _)| every || !row_fills(at).is_empty())
            .collect();
        for pair in ordered.windows(2) {
            let [(at, record), (next_at, next)] = [pair[0], pair[1]];
            let ([start, end, ..], [next_start, next_end, ..]) = (record.times, next.times);
            if next_start < start || next_end < end {
                broken.push(format!(
                    "queue {queue}, {at} and {next_at} went back: {record:?}, {next:?}"
                ));
            }
            if next_start < end {
                overlaps += 1;
            }
        }
    }
    (broken, overlaps, late)
}

/// A command buffer's kernel end after its GPU start, as plain Metal answered on the M3 Max under
/// macOS 27.0.1: a fill's `kernelEndTime` 121311.732792 s against its `GPUStartTime` .732711 and
/// `GPUEndTime` .732714, another's .736230 against .733545 and .733550. Not Metal's rule there,
/// so Metal's answers are not held to it: counted and printed. The runtime's answers keep it by construction
/// (`gpu_time.rs`: the GPU start is its kernel end time or later).
const LATE_KERNEL_END: &str = "a kernel end after the GPU start";

/// A record whose kernel end came after its GPU start and end, as one of Metal's did on the M3
/// under macOS 27.0.1 ([`LATE_KERNEL_END`], its times), is counted, not broken, where late kernel
/// ends are admitted (Metal's answers), and breaks the order where not (the runtime's); one whose
/// GPU start came before its kernel start breaks it either way.
#[test]
fn a_late_kernel_end_is_counted_for_metal_and_broken_for_the_runtime() {
    let late = Record {
        commit: 121311.710109333,
        handler: 121311.735203166,
        times: [
            121311.732711125,
            121311.73271366667,
            121311.73249775001,
            121311.73279187501,
        ],
    };
    let early = Record {
        times: [late.times[0], late.times[1], 121311.7328, 121311.7329],
        ..late
    };
    let rules = |record: Record, admitted: bool| {
        let (broken, _, late) = broken_rules(&[vec![record]], true, admitted);
        (broken.len(), late)
    };
    assert_eq!(rules(late, true), (0, 1), "{LATE_KERNEL_END}, admitted");
    assert_eq!(
        rules(late, false),
        (1, 0),
        "{LATE_KERNEL_END}, not admitted"
    );
    assert_eq!(
        rules(early, true),
        (1, 0),
        "a GPU start before the kernel start"
    );
}

/// The rules test's rows on the runtime's `device`: two queues, [`IN_A_ROW`] command buffers each,
/// committed in turn with no wait between them, then waited for.
fn runtime_rows(device: &Device) -> Vec<Vec<Record>> {
    let buffer = device
        .new_buffer(4 << 20, ResourceOptions::private())
        .unwrap();
    let queues = [
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    ];
    let records: Vec<Arc<Mutex<Record>>> = (0..2 * IN_A_ROW).map(|_| Arc::default()).collect();
    let mut committed = Vec::new();
    for at in 0..IN_A_ROW {
        for (queue_at, queue) in queues.iter().enumerate() {
            let cb = queue.command_buffer().unwrap();
            for &bytes in row_fills(at) {
                let mut blit = cb.blit_command_encoder().unwrap();
                blit.fill_buffer(&buffer, 0..bytes, at as u8).unwrap();
                blit.end_encoding().unwrap();
            }
            let record = Arc::clone(&records[queue_at * IN_A_ROW + at]);
            cb.add_completed_handler(Box::new(move |cb| {
                let handler = host_time();
                let times = [
                    cb.gpu_start_time().unwrap_or(-1.0),
                    cb.gpu_end_time().unwrap_or(-1.0),
                    cb.kernel_start_time(),
                    cb.kernel_end_time(),
                ];
                let mut record = record.lock().unwrap_or_else(PoisonError::into_inner);
                record.handler = handler;
                record.times = times;
            }))
            .unwrap();
            records[queue_at * IN_A_ROW + at]
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .commit = host_time();
            cb.commit().unwrap();
            committed.push(cb);
        }
    }
    for (at, cb) in committed.iter().enumerate() {
        cb.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(cb.error(), None, "{at}");
    }
    records
        .chunks(IN_A_ROW)
        .map(|row| {
            row.iter()
                .map(|record| *record.lock().unwrap_or_else(PoisonError::into_inner))
                .collect()
        })
        .collect()
}

/// The rules test's rows on plain Metal on this Mac's GPU, as [`runtime_rows`] commits them.
#[cfg(all(m2v_gpu, target_os = "macos"))]
fn metal_rows() -> Vec<Vec<Record>> {
    use crate::gpu::wait::metal::{settle_watched, watch};
    use block2::RcBlock;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSRange;
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice,
        MTLResourceOptions,
    };
    use std::ptr::NonNull;
    let device = crate::metal::system_device().unwrap();
    let buffer = device
        .newBufferWithLength_options(4 << 20, MTLResourceOptions::StorageModePrivate)
        .unwrap();
    let queues = [
        device.newCommandQueue().unwrap(),
        device.newCommandQueue().unwrap(),
    ];
    let records: Vec<Arc<Mutex<Record>>> = (0..2 * IN_A_ROW).map(|_| Arc::default()).collect();
    let mut committed = Vec::new();
    for at in 0..IN_A_ROW {
        for (queue_at, queue) in queues.iter().enumerate() {
            let cb = queue.commandBuffer().unwrap();
            for &bytes in row_fills(at) {
                let blit = cb.blitCommandEncoder().unwrap();
                blit.fillBuffer_range_value(&buffer, NSRange::new(0, bytes as usize), at as u8);
                blit.endEncoding();
            }
            let record = Arc::clone(&records[queue_at * IN_A_ROW + at]);
            let handler = RcBlock::new(move |cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let handler = host_time();
                // SAFETY: Metal passes the live command buffer the handler was added to.
                let cb = unsafe { cb.as_ref() };
                let times = [
                    cb.GPUStartTime(),
                    cb.GPUEndTime(),
                    cb.kernelStartTime(),
                    cb.kernelEndTime(),
                ];
                let mut record = record.lock().unwrap_or_else(PoisonError::into_inner);
                record.handler = handler;
                record.times = times;
            });
            // SAFETY: the block is valid for the call, and Metal copies it; added before the
            // commit.
            unsafe { cb.addCompletedHandler(RcBlock::as_ptr(&handler)) };
            let watched = watch(&cb);
            records[queue_at * IN_A_ROW + at]
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .commit = host_time();
            cb.commit();
            committed.push((cb, watched));
        }
    }
    for (cb, watched) in committed {
        settle_watched(&cb, watched, "the GPU time rules").unwrap();
    }
    records
        .chunks(IN_A_ROW)
        .map(|row| {
            row.iter()
                .map(|record| *record.lock().unwrap_or_else(PoisonError::into_inner))
                .collect()
        })
        .collect()
}

/// The four rules of host-clock GPU times (`gpu_time.rs`), checked on answers, not values. On a
/// device answering from the host's clock (by its driver, or told to): two queues of [`IN_A_ROW`]
/// command buffers each (fills, none, two encoders), committed in turn with no wait between them,
/// each answer seconds of `host_time`'s clock, in the order commit, kernel start, kernel end, GPU
/// start, GPU end, completed handler, each queue's in order and none starting before the one before
/// it on its queue ended; the device announces `gpu-times-from-host` once. The fourth rule, no
/// counter samples, holds as the runtime has none. On a Mac's GPU, Metal's own answers to the same
/// rows, in the same test, keep the same rules among the command buffers with an encoder but one
/// (its overlaps, its encoder-less command buffers out of its queue's order, and its kernel ends
/// after their GPU starts, [`LATE_KERNEL_END`], counted and printed; [`broken_rules`]): the rules
/// are Metal's, its clock `host_time`'s. The runtime's answers keep that one too.
#[test]
fn host_clock_gpu_times_keep_metal_s_rules() {
    #[cfg(all(m2v_gpu, target_os = "macos"))]
    {
        let rows = metal_rows();
        let (broken, overlaps, late) = broken_rules(&rows, false, true);
        let (every, _, _) = broken_rules(&rows, true, true);
        eprintln!(
            "runtime gpu times: Metal's answers: {} broken rules, {overlaps} overlaps and {late} \
             with {LATE_KERNEL_END} among the command buffers with an encoder; {} broken with \
             the encoder-less ones too",
            broken.len(),
            every.len()
        );
        assert!(
            broken.is_empty(),
            "Metal's answers broke a rule: {broken:#?}"
        );
    }
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let by_driver = from_host_by_driver(&device);
        let told = if by_driver {
            None
        } else {
            Some(
                validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
                    options.gpu_times_from_host = true;
                })
                .unwrap(),
            )
        };
        let rows = runtime_rows(told.as_ref().unwrap_or(&device));
        let (broken, overlaps, _) = broken_rules(&rows, true, false);
        assert!(
            broken.is_empty(),
            "the runtime's answers broke a rule: {broken:#?}"
        );
        assert_eq!(overlaps, 0, "{rows:#?}");
        let notices = messages.take_notices();
        assert_once(&notices, &FROM_HOST);
        eprintln!(
            "runtime gpu times: the runtime's answers from the host's clock ({}) keep the rules",
            if by_driver {
                "by the driver"
            } else {
                "told to"
            }
        );
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}
