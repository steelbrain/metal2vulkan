//! A GPU that stopped: work pending that never completes, and no error from the driver.
//!
//! Metal fails GPU work that runs too long: the command buffer fails with
//! `MTLCommandBufferErrorTimeout` (code 2, the runtime's `GpuHang`) and the device goes on. On an
//! Apple M3 Max (macOS 27.0.1) a kernel of 2^29 threadgroups finished in 4.98 s and one of 2^30
//! did not finish (`error.rs`, `GpuRecoveryVictim`); on x86 Macs an AMD GPU failed such work after
//! 20.2 s and an Intel one after 16.1 s (`kIOAccelCommandBufferCallbackErrorTimeout`, code 2).
//! Vulkan has no such bound: a driver ends a hang as a lost device (`VK_ERROR_DEVICE_LOST`), which
//! the device's line recovers from (`device/line.rs`). On NVIDIA's driver after an Xid 109 (a
//! context switch timeout) the device's queues stopped instead: no batch completed, and every
//! wait of the completion's waiters answered `VK_TIMEOUT` (`WAIT_SLICE`, `completion.rs`), never a
//! loss, so the program's command buffers waited without end.
//!
//! - **The stall clock** ([`Clock`]): one per device, the completion's waiters' own, under their
//!   lock. It runs while a waiter's wait ran out with its queue's front batch pending, and stops
//!   whenever any queue's timeline advances, so it measures how long no work of the device
//!   advanced while some was pending. Counted: `WorkCounts::longest_stall_nanos`.
//! - **The probe** ([`probe`]). Past [`STALL`], one waiter sends a probe to another engine: the
//!   staging queue (`device/staging.rs`; NVIDIA's copy engine), a submission of nothing that waits
//!   for nothing but the staging submissions before it, which wait for no work in flight. For up
//!   to [`PROBE`] it looks, each [`LOOK`], at the probe's answer and at the stuck front's: its
//!   fence (a wait of no time, which answers its status) and its queue's timeline
//!   (`vkGetSemaphoreCounterValue`). A loss either read answers takes the failed wait's path at
//!   once, as the device's own signal. One informational message names the stall, the probe's
//!   answer and the last result of each read (`metal2vulkan-stall-probe`). Counted:
//!   `WorkCounts::stall_probes_sent`.
//! - **The verdict.** Where the probe completed or the front advanced, the GPU runs: nothing
//!   fails, and the clock starts again, so a kernel that runs long on a device that still makes
//!   progress is never failed, whatever its length (Metal's bounds above are not imitated). Where
//!   neither did within [`PROBE`], the generation stopped: it is marked so and lost
//!   (`Device::mark_wedged`), and the waiter's wait fails as a lost device's: the oldest command
//!   buffer in flight fails as the hang (`GpuHang`) and the rest as its victims
//!   (`GpuRecoveryVictim`), and the line recovers on a new `VkDevice`, announced
//!   (`Degradation::DeviceRecovered`, which says the generation stopped). Counted:
//!   `WorkCounts::wedges_declared`.
//! - **Parked, never destroyed.** The stopped generation's work never completes, so nothing of
//!   it may be destroyed (each `vkDestroy*` asks that nothing pending use the object) and no wait
//!   for its queues returns: its `VkDevice` is parked (`release.rs`), none of its objects is
//!   released from then on, the device itself included, and nothing waits for its queues
//!   (`DeviceShared::queues_idle`). A consumer that holds its handles (`Device::raw`) asks
//!   [`Device::is_parked`](crate::Device::is_parked) to leave them likewise. A leak of what the
//!   generation held (docs/FAST-PATHS.md, known gaps).
//! - **Bounds.** [`STALL`] is 10 s, twice the M3 Max's bound and under the x86 Macs'; a stopped
//!   generation fails after 12 s at least. A device without a staging queue sends no probe, and
//!   a stop there is not detected (docs/FAST-PATHS.md, known gaps).

use crate::exit_guard::{self, Call};
use crate::Device;
use ash::vk;
use std::time::{Duration, Instant};

/// How long no queue's work may advance, with some pending, before a probe asks whether the GPU
/// still runs anything (the module docs).
pub(super) const STALL: Duration = Duration::from_secs(10);

/// How long a probe waits for an answer before the generation counts as stopped.
pub(super) const PROBE: Duration = Duration::from_secs(2);

/// How long one look of a probe waits for its answer.
const LOOK: Duration = Duration::from_millis(100);

/// The device's stall clock (the module docs), under the completion's lock.
#[derive(Default)]
pub(super) struct Clock {
    /// Since when no queue's timeline advanced while a front was pending.
    since: Option<Instant>,
    /// A waiter probes: no other does meanwhile.
    probing: bool,
}

impl Clock {
    /// A queue's timeline advanced: the clock stops. How long it ran, where it ran.
    pub(super) fn advanced(&mut self) -> Option<Duration> {
        self.since.take().map(|since| since.elapsed())
    }

    /// A waiter's wait ran out with its queue's front pending: the clock runs. How long it has run
    /// where it passed `after` and no waiter probes, this one then probing.
    pub(super) fn ran_out(&mut self, after: Duration) -> Option<Duration> {
        let stalled = self.since.get_or_insert_with(Instant::now).elapsed();
        let due = !self.probing && stalled >= after;
        self.probing |= due;
        due.then_some(stalled)
    }

    /// The probe ended, the GPU running: the clock starts again.
    pub(super) fn probed(&mut self) {
        self.probing = false;
        self.since = Some(Instant::now());
    }
}

/// What a probe found ([`probe`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Verdict {
    /// The front advanced or the probe completed (or could not be sent, which proves nothing).
    Runs,
    /// A read answered this failure: the failed wait's path takes it.
    Failed(vk::Result),
    /// Neither within [`PROBE`]: the generation stopped.
    Stopped,
}

/// How long the clock runs before a probe on `device`: [`STALL`], or what a device test set
/// (`device/line.rs`'s hooks).
pub(super) fn stall_after(device: &Device) -> Duration {
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    if let Some(after) = *crate::device::lock(&device.line().hooks.stall_after) {
        return after;
    }
    let _ = device;
    STALL
}

/// Whether `device` sends probes: where it has a staging queue (the module docs, **Bounds**), or a
/// device test has no probe seen answered (`device/line.rs`'s hooks).
pub(super) fn probes(device: &Device) -> bool {
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    if (device.line().hooks.probe_unanswered).load(std::sync::atomic::Ordering::Acquire) {
        return true;
    }
    device.has_staging_queue()
}

/// Probe `device` after `stalled` with no queue's work advancing (the module docs), the stuck
/// front the batch of `VkQueue` `queue` that signals `value`, and `fence` where it has one.
pub(super) fn probe(
    device: &Device,
    queue: usize,
    (value, fence): (u64, Option<vk::Fence>),
    stalled: Duration,
) -> Verdict {
    let shared = device.shared();
    shared.counts.stall_probe_sent(stalled);
    let raw = shared.vk();
    let timeline = shared.queues[queue].timeline;
    let sent = device
        .staging_lock()
        .and_then(|mut staging| staging.probe());
    let began = Instant::now();
    let mut fence_read: Option<vk::Result> = None;
    let mut timeline_read: Result<u64, vk::Result> = Ok(0);
    let mut answered: Result<bool, String> = Ok(false);
    let verdict = loop {
        if let Some(fence) = fence {
            // SAFETY: the fence is the device's, in a submission made and not yet reset (its batch
            // is pending, and is not taken while its waiter probes), live while the device is
            // (held here). A wait of no time answers its status.
            let looked = unsafe { raw.wait_for_fences(&[fence], true, 0) };
            exit_guard::called(Call::WaitForFences);
            fence_read = Some(looked.err().unwrap_or(vk::Result::SUCCESS));
        }
        // SAFETY: the semaphore is the device's timeline, live while the device is (held here).
        timeline_read = unsafe { raw.get_semaphore_counter_value(timeline) };
        exit_guard::called(Call::GetSemaphoreCounterValue);
        let failed = match (fence_read, timeline_read) {
            (Some(error @ vk::Result::ERROR_DEVICE_LOST), _) | (_, Err(error)) => Some(error),
            _ => None,
        };
        if let Some(error) = failed {
            break Verdict::Failed(error);
        }
        if fence_read == Some(vk::Result::SUCCESS) || timeline_read.is_ok_and(|read| read >= value)
        {
            break Verdict::Runs;
        }
        answered = match &sent {
            Ok(sent) => device
                .staging_reached(*sent, LOOK)
                .map_err(|e| e.to_string()),
            Err(error) => Err(error.to_string()),
        };
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        if device
            .line()
            .hooks
            .probe_unanswered
            .load(std::sync::atomic::Ordering::Acquire)
        {
            answered = Ok(false);
            std::thread::sleep(LOOK);
        }
        match answered {
            Ok(true) => break Verdict::Runs,
            // A probe that could not be sent, or whose wait failed, answers nothing: no verdict.
            Err(_) => break Verdict::Runs,
            Ok(false) if began.elapsed() >= PROBE => break Verdict::Stopped,
            Ok(false) => {}
        }
    };
    let probe = match &answered {
        Ok(true) => format!("completed in {:?}", began.elapsed()),
        Ok(false) if verdict == Verdict::Stopped => format!("did not complete in {PROBE:?}"),
        Ok(false) => "was still pending".to_owned(),
        Err(error) => format!("answered nothing ({error})"),
    };
    let fence_read = match fence_read {
        Some(result) => format!("{result:?}"),
        None => "none".to_owned(),
    };
    let timeline_read = match timeline_read {
        Ok(read) => format!("{read} of {value}"),
        Err(error) => format!("{error:?}"),
    };
    let said = match verdict {
        Verdict::Runs => "the GPU runs: nothing fails".to_owned(),
        Verdict::Failed(error) => format!("a read answered {error:?}: the device is lost"),
        Verdict::Stopped => "the generation stopped: its work fails, and it is parked".to_owned(),
    };
    device.inform(
        "metal2vulkan-stall-probe",
        format!(
            "metal2vulkan: no work of the device advanced for {stalled:?} with some pending on \
             VkQueue {queue}; a probe on the staging queue {probe}; the stuck batch's fence \
             read {fence_read}, its timeline {timeline_read}: {said}"
        ),
    );
    verdict
}
