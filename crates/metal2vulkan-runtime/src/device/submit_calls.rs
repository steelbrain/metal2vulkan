//! How long a `VkQueue`'s batches take to submit, measured on the submissions the runtime makes,
//! and whether a committing thread leaves its batches there to the scheduler's thread.
//!
//! **The cost.** A batch's `vkQueueSubmit` is the driver's work on the calling thread, and its
//! length is the driver's and the kernel's, not the runtime's: RADV lists every memory the device
//! imported in each kernel submission, and amdgpu gets every page of each pointer import again
//! (`submit_check.rs`, "Import tax"). A guest on a host of unified memory keeps all its memory in
//! place as such imports, so on the Radeon 780M its batches' calls took 245 us at the median (394
//! at the 90th percentile; reims's traced macOS guest walk, Mesa 26.2.6), where the same driver
//! submits an empty command buffer in 13.2 us (`submit_check.rs`, "Merge"). Before the call, the
//! thread that submits plans the batch against the tracker and records its barriers, transitions
//! and timestamps (`CommandBufferShared::submit_batch`). Where a committing thread submitted those
//! batches (`CommandBuffer::commit_and_submit`'s pass), each commit of the guest's held its thread
//! for both, which the scheduler's thread paid before.
//!
//! **Measured, not named.** Each call of one command buffer or more is timed as the whole
//! submission of the last batch it carries, from the start of that batch's submission to the
//! call's return, whichever thread made it: what a committing thread pays to submit a batch. The
//! median of the queue's last [`CALLS`] such submissions is its cost, published for reads without
//! the lock; none before [`CALLS`] were timed. The scheduler's thread's submissions are timed as
//! the committing threads' are, so a queue whose submissions turn cheap again goes back to its
//! committing threads once half its last [`CALLS`] are under the bound. A median, so the cost is the one a commit pays
//! half the time or more: a driver's rare long call (NVIDIA's stall in one signalling call of
//! 1024, `submit_stalls.rs`) or a few submissions other work on the host lengthened move it no
//! more than a guest queue's smaller batches among another's slow ones do (in reims's traced
//! window drag on the Radeon 780M, 12 to 20% of one guest queue's calls came in under
//! [`SLOW_NANOS`] among two queues' parts whose commit calls took 154 us or more at their fifth
//! percentile); half of them, or more, move it. A floor (the least of the last [`CALLS`] calls but
//! the shortest, before) judged the queue by its cheapest calls: on the Iris Xe (ANV; reims's
//! desk walk, the runtime at 5fc0508f0b), the calls of the compositor's parts submitted inside
//! their commit took, with the commit's return after them, 57 to 243 us from the first to the
//! ninth decile, 36% of them under [`SLOW_NANOS`], so the floor stayed under it most of the time,
//! and 40 to 69% of those parts were still submitted inside their commit, each such commit call
//! about 220 us against 37 to 70 us for the others. Their whole submissions took 86 to 313 us,
//! 11 to 17% of them under [`SLOW_NANOS`]. On lavapipe beside the runtime's device tests, the
//! median of a fresh device's last 15 calls stood past [`SLOW_NANOS`] for 14 of its first 200
//! commits, where its calls then took 5.3 us at the median: those commits go to the scheduler's
//! thread (the tests that count every commit at commit raise the bound,
//! `Device::leave_calls_from`).
//!
//! **Calls of commands alone.** A batch with no work, no wait and no timestamp holds no command
//! buffer (`command_buffer/submit.rs`), as that of an empty command buffer that signals an event
//! (one that signals nothing makes no call), and its call is another call: the driver submits no
//! commands, where it lists every import for a submission of them (ANV on i915 lists two objects
//! in a submission of none, `submit_check.rs`). It is not timed: in reims's traced macOS guest on
//! the Radeon 780M (RADV, a window drag, about 140 dma-buf imports live), 8.8% of the batches had
//! no GPU work, most of those a committing thread submitted took 15 to 40 us for the whole commit,
//! where the batches with work took 173 us at their first quartile. On the Iris Xe (ANV), a third
//! of a video step's batches had no GPU work, and its calls took 306 to 374 us on the average. A
//! batch with no command buffer is left to the scheduler's thread with the rest of its queue's
//! where those are slow, and is counted as such.
//!
//! **A group's call.** The scheduler's thread sends the batches it held in a group in one call
//! (`command_buffer/group.rs`), with a batch after them or not. That call is timed for its last
//! batch: one after the group from the start of its own submission, and the last one held from as
//! far before the call as its submission took to its hold. A committing thread would have paid
//! that batch's submission and a call. Where the group's call went untimed, a queue left to the
//! scheduler's thread, which groups most of its batches where the driver merges submissions, kept
//! its cost for good: on the Radeon RX 7900 XTX (RADV) in `tests/drag_submit_bench.rs`, the
//! queue's cost stayed at 196.5 us through every leg, though the driver's own calls took 72.6 us
//! on the average, and none of the parts went at its commit where 94.3% had.
//!
//! **Left to the scheduler's thread.** Where a `VkQueue`'s cost is [`SLOW_NANOS`] or more, a
//! committing thread's pass leaves every batch there to the scheduler's thread
//! (`scheduler/mod.rs`, `plain`; `WorkCounts::scheduler_at_commit_left_slow_calls`), which it
//! wakes at once: the committing thread pays its pass, and the scheduler's thread the
//! submissions, as it paid them for every batch over host pages before a committing thread
//! submitted those. Below it, the committing thread submits the batch itself. A thread about to
//! wait for a command buffer submits as before, whatever the cost: it would only wait meanwhile.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// The submissions whose median is a queue's cost.
pub(crate) const CALLS: usize = 15;

/// A `VkQueue` whose last [`CALLS`] batch submissions took this long or more at their median, in
/// nanoseconds, is one whose batches a committing thread leaves to the scheduler's thread. The
/// hand-off costs a batch the scheduler thread's wake before its call, and saves the committing
/// thread the submission: at 100 us a submission costs the committing thread several times what a
/// wake delays the batch.
pub(crate) const SLOW_NANOS: u64 = 100_000;

/// A queue's last [`CALLS`] batch submissions, a ring, and their median (the module docs).
#[derive(Default)]
pub(crate) struct SubmitCalls {
    /// The submissions' times in nanoseconds, the next to replace at `timed % CALLS`.
    ring: Mutex<([u64; CALLS], usize)>,
    /// The median of the ring once it is full, in nanoseconds; 0 before.
    cost: AtomicU64,
}

impl SubmitCalls {
    /// A batch of one command buffer or more went to the driver in a call of its own, `nanos` from
    /// the start of its submission to the call's return: kept, and the cost published once
    /// [`CALLS`] were timed.
    pub(crate) fn called(&self, nanos: u64) {
        let mut ring = super::lock(&self.ring);
        let (calls, timed) = &mut *ring;
        calls[*timed % CALLS] = nanos;
        *timed += 1;
        if *timed >= CALLS {
            let mut ranked = *calls;
            let (_, median, _) = ranked.select_nth_unstable(CALLS / 2);
            self.cost.store(*median, Ordering::Relaxed);
        }
    }

    /// The median of the last [`CALLS`] batch submissions, in nanoseconds; 0 before [`CALLS`]
    /// were timed.
    pub(crate) fn cost(&self) -> u64 {
        self.cost.load(Ordering::Relaxed)
    }

    /// How many submissions were timed, in the device tests' builds only.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn timed(&self) -> usize {
        super::lock(&self.ring).1
    }

    /// Whether a committing thread leaves the queue's batches to the scheduler's thread: their
    /// submissions were timed and their median took `bound` or more ([`SLOW_NANOS`] but in the
    /// device tests).
    pub(crate) fn slow(&self, bound: u64) -> bool {
        let cost = self.cost();
        cost > 0 && cost >= bound
    }
}
