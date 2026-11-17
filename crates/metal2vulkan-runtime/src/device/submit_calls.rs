//! How long a `VkQueue`'s `vkQueueSubmit` calls of batches take, measured on the calls the runtime
//! makes, and whether a committing thread leaves its batches there to the scheduler's thread.
//!
//! **The cost.** A batch's `vkQueueSubmit` is the driver's work on the calling thread, and its
//! length is the driver's and the kernel's, not the runtime's: RADV lists every memory the device
//! imported in each kernel submission, and amdgpu gets every page of each pointer import again
//! (`submit_check.rs`, "Import tax"). A guest on a host of unified memory keeps all its memory in
//! place as such imports, so on the Radeon 780M its batches' calls took 245 us at the median (394
//! at the 90th percentile; reims's traced macOS guest walk, Mesa 26.2.6), where the same driver
//! submits an empty command buffer in 13.2 us (`submit_check.rs`, "Merge"). Where a committing
//! thread made those calls (`CommandBuffer::commit_and_submit`'s pass), each commit of the guest's
//! held its thread that long, about 210k user instructions a submission and the kernel's call,
//! which the scheduler's thread made before.
//!
//! **Measured, not named.** Each successful call that submits a batch (one that signals the
//! queue's timeline, `DeviceShared::submit_logged`) of one command buffer or more is timed under
//! the queue's lock, the time inside the call alone, whichever thread made it. The floor of the
//! last [`CALLS`], the least of them, is the queue's cost, published for reads without the lock;
//! none before [`CALLS`] calls were timed. A floor, as the submission checks read floors
//! (`submit_check.rs`, "Load beside
//! it"): other work on the host or the device lengthens a call and never shortens it, and a
//! driver's own long calls come among short ones (NVIDIA's stall in one signalling call of 1024,
//! `submit_stalls.rs`; lavapipe's first calls of a fresh device), so only a cost every call pays
//! moves the floor. A median moved by them: on lavapipe beside the runtime's device tests, the
//! median of a fresh device's last 15 calls stood past [`SLOW_NANOS`] for 14 of its first 200
//! commits, where its calls then took 5.3 us at the median.
//!
//! **Calls of commands alone.** A batch with no work, no wait and no timestamp holds no command
//! buffer (`command_buffer/submit.rs`), as that of an empty command buffer that signals an event
//! (one that signals nothing makes no call), and its call is another call: the driver submits no commands, where it lists every import for a submission of
//! them (ANV on i915 lists two objects in a submission of none, `submit_check.rs`). Kept in, one
//! such call among [`CALLS`] holds the floor at its own length. In reims's traced macOS guest on
//! the Radeon 780M (RADV, a window drag, about 140 dma-buf imports live), 8.8% of the batches had
//! no GPU work; most of those a committing thread submitted took 15 to 40 us for the whole commit,
//! where the batches with work took 173 us at their first quartile, so the floor stayed under
//! [`SLOW_NANOS`] and the committing threads made 85 to 88% of the calls. Of the calls of commands
//! alone, the least of each 15 the window server's thread made was 143 us at its tenth
//! percentile. On the Iris Xe (ANV), a third of a Safari video step's batches had no GPU work, and
//! its calls, 306 to 374 us on the average, stayed on the committing threads. A batch with no
//! command buffer is left to the scheduler's thread with the rest of its queue's where those are
//! slow, and is counted as such.
//!
//! **Left to the scheduler's thread.** Where a `VkQueue`'s cost is [`SLOW_NANOS`] or more, a
//! committing thread's pass leaves every batch there to the scheduler's thread
//! (`scheduler/mod.rs`, `plain`; `WorkCounts::scheduler_at_commit_left_slow_calls`), which it
//! wakes at once: the committing thread pays its pass, and the scheduler's thread the calls,
//! as it paid them for every batch over host pages before a committing thread submitted those.
//! Below it, the committing thread submits the batch itself. A thread about to wait for a command
//! buffer submits as before, whatever the cost: it would only wait meanwhile.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// The calls whose floor is a queue's cost.
pub(crate) const CALLS: usize = 15;

/// A `VkQueue` whose batches' last [`CALLS`] calls each took this long or more, in nanoseconds, is
/// one whose batches a committing thread leaves to the scheduler's thread. The hand-off costs a
/// batch the scheduler thread's wake before its call, and saves the committing thread the call:
/// at 100 us a call costs the committing thread several times what a wake delays the batch.
pub(crate) const SLOW_NANOS: u64 = 100_000;

/// A queue's last [`CALLS`] batch calls, a ring, and their floor (the module docs).
#[derive(Default)]
pub(crate) struct SubmitCalls {
    /// The calls' times in nanoseconds, the next to replace at `timed % CALLS`.
    ring: Mutex<([u64; CALLS], usize)>,
    /// The least of the ring once it is full, in nanoseconds; 0 before.
    floor: AtomicU64,
}

impl SubmitCalls {
    /// A call that submitted a batch of one command buffer or more took `nanos`: kept, and the
    /// floor published once [`CALLS`] were timed.
    pub(crate) fn called(&self, nanos: u64) {
        let mut ring = super::lock(&self.ring);
        let (calls, timed) = &mut *ring;
        calls[*timed % CALLS] = nanos;
        *timed += 1;
        if *timed >= CALLS {
            let least = calls.iter().copied().min().unwrap_or(0);
            self.floor.store(least, Ordering::Relaxed);
        }
    }

    /// The least of the last [`CALLS`] calls, in nanoseconds; 0 before [`CALLS`] were timed.
    pub(crate) fn floor(&self) -> u64 {
        self.floor.load(Ordering::Relaxed)
    }

    /// Whether a committing thread leaves the queue's batches to the scheduler's thread: their
    /// calls were timed and each of the last [`CALLS`] took `bound` or more ([`SLOW_NANOS`] but in
    /// the device tests).
    pub(crate) fn slow(&self, bound: u64) -> bool {
        let floor = self.floor();
        floor > 0 && floor >= bound
    }
}
