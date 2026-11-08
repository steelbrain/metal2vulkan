//! The device's uses of its `VkQueue`s seen as one: how many are under way, and when the last one
//! ended, for work that holds every submission of the device while it runs, so that it begins
//! right after a burst of submissions returned rather than beside one.
//!
//! **What it is for.** A pooled block's `vkAllocateMemory` holds the device's submissions on some
//! drivers: on RADV (a Radeon 780M, kernel 7.3.1) the kernel allocates a block's pages under the
//! reservation of the device's GPU address space, which every submission of the device takes, so
//! a submission made meanwhile waited up to 29 ms for a 64 MiB block, and on NVIDIA's driver each
//! `vkAllocateMemory` holds every submission until it returns (`memory.rs`). Blocks made ahead
//! (`memory/ready.rs`) wait here for a moment ([`QueueQuiet::wait_after_burst`]).
//!
//! **What counts as a use.** A hold of a queue's lock (`SlotHold`), which every use of one of the
//! device's `VkQueue`s the runtime makes takes, and a submission to its staging queue. The
//! program's own uses of its reserved or extra queues, and of an adopted device's, are not seen.

use crate::gpu_time;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// How long no use may be under way, after one ended, before a burst of uses is taken to have
/// ended: the gap that begins a burst (`submit_stalls::IDLE_NANOS`).
pub(crate) const SETTLE: Duration = Duration::from_nanos(super::submit_stalls::IDLE_NANOS);

/// How often a wait looks again while a use is under way.
const LOOK_EVERY: Duration = Duration::from_micros(250);

/// The device's queue uses (the module docs).
#[derive(Default)]
pub(crate) struct QueueQuiet {
    /// The uses under way.
    in_use: AtomicU32,
    /// When the last use ended, in [`gpu_time::host_nanos`]; 0 before the first.
    ended_at: AtomicU64,
}

/// How a [`QueueQuiet::wait_after_burst`] ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Waited {
    /// No use was under way as the wait began, and none had ended for its bound: the device had
    /// no submissions to fit between.
    Idle,
    /// A burst of uses ended after the wait began, and [`SETTLE`] passed with none under way.
    AfterBurst,
    /// The wait's bound passed first.
    Bound,
    /// The caller's `now` said to stop waiting.
    Asked,
}

impl QueueQuiet {
    /// A use begins.
    pub(crate) fn began(&self) {
        self.in_use.fetch_add(1, Ordering::AcqRel);
    }

    /// A use that [began](Self::began) ends.
    pub(crate) fn ended(&self) {
        self.ended_at
            .store(gpu_time::host_nanos().max(1), Ordering::Release);
        self.in_use.fetch_sub(1, Ordering::AcqRel);
    }

    /// Wait for a burst of uses to end after the call: a use ended since, and none has been under
    /// way for [`SETTLE`] since the last one ended. At most `most`, and no longer once `now` says
    /// so (asked between looks). None at all where no use is under way and none ended within
    /// `most`: a device that has not used its queues for that long has no submissions to fit
    /// between, and whenever the work runs it is as likely to meet the next one.
    pub(crate) fn wait_after_burst(&self, most: Duration, now: impl Fn() -> bool) -> Waited {
        let began = Instant::now();
        let seen = self.ended_at.load(Ordering::Acquire);
        let since = gpu_time::host_nanos().saturating_sub(seen);
        let idle = seen == 0 || Duration::from_nanos(since) >= most;
        if idle && self.in_use.load(Ordering::Acquire) == 0 {
            return Waited::Idle;
        }
        loop {
            if now() {
                return Waited::Asked;
            }
            let left = most.saturating_sub(began.elapsed());
            if left.is_zero() {
                return Waited::Bound;
            }
            let ended_at = self.ended_at.load(Ordering::Acquire);
            let pause = match self.in_use.load(Ordering::Acquire) {
                0 if ended_at != seen => {
                    let since = gpu_time::host_nanos().saturating_sub(ended_at);
                    match SETTLE.checked_sub(Duration::from_nanos(since)) {
                        None => return Waited::AfterBurst,
                        Some(pause) => pause,
                    }
                }
                _ => LOOK_EVERY,
            };
            std::thread::sleep(pause.min(LOOK_EVERY).min(left));
        }
    }
}
