//! Command queues that share a `VkQueue` take turns there (`QueueTurnPolicy::Fair`, the default):
//! a command queue not made unshared keeps only about 2 ms of its own expected GPU work submitted
//! there while another command queue submits there, or 16 ms while none does, and another queue's
//! later commit goes ahead of its held batches.
//!
//! - **Why.** A `VkQueue` runs its submissions in order. Where one command queue commits a burst
//!   of command buffers (a long command buffer cut into parts, a backlog of frames) on the one
//!   graphics `VkQueue` of RADV, ANV, lavapipe or another driver that lends one, every ready batch
//!   went to the `VkQueue` at its commit, and a command buffer another queue committed after the
//!   burst waited for the whole of it. An M3 Max (macOS 27.0.1) ran a second queue's small
//!   command buffer among a first queue's burst of 64, ahead of 56 to 62 of them; a Radeon Pro
//!   5300M (macOS 26.7.1) a blit or render one so, a compute one after the burst; a UHD 630
//!   (macOS 26.7.1) after the burst in every case (`metal2vulkan-conformance`'s
//!   `probes/cross_queue_burst_turns.m`).
//! - **Where.** A batch of a command queue not made unshared, on a `VkQueue` two or more command
//!   queues hold (`SlotShare::held_by_several`: the round robin's queues and the unshared ones
//!   placed there), on a device whose policy is `Fair`. An unshared queue's batch is never held for
//!   its turn: `unshared_hold.rs` bounds the other queues' work ahead of it, and a pass sends it
//!   first.
//! - **The bound.** Such a batch, otherwise ready, goes only while less than the budget of its own
//!   queue's expected work is in flight there (`unshared_hold::Flight::own_ahead`): its queue's
//!   batches submitted there and not seen complete, each at its queue's expected run now
//!   (`unshared_hold::Expected`), less what the oldest batch there ran since it became so, where
//!   that one is its queue's. Each batch counts its queue's run now, not the one its queue had at
//!   its submission, so a run longer than the queue's recent ones prices the rest of a burst at
//!   once.
//! - **Two budgets.** `unshared_hold::BUDGET` (2 ms) while another command queue took a batch on
//!   that `VkQueue` within `unshared_hold::ACTIVE` (50 ms; the flight keeps the latest batch's
//!   queue and time, and the same of the latest of another queue before it), else
//!   `unshared_hold::IDLE_BUDGET` (16 ms): a queue alone keeps about 16 ms queued, so a command
//!   buffer another queue commits after a quiet stretch goes in behind about that much and one
//!   batch, and from its submission on the 2 ms bound applies (`WorkCounts::heads_let_go_turn_idle`
//!   counts the heads 2 ms would have held).
//! - **Unknown length.** A queue with no run seen yet keeps one batch in flight there until it is
//!   seen complete, which gives the queue its run: it holds back its own next batches alone.
//! - **Turns.** A held batch keeps its place in the scheduler's order (`scheduler/order.rs`), as
//!   one held for an unshared queue does: a later commit of another queue whose footprint runs in
//!   commit order with it waits behind it, and one that shares nothing with it goes ahead of it.
//!   It waits for its own queue's work already submitted alone, never for a later commit.
//! - **Whole command buffers.** Only a command buffer's first batch takes a turn
//!   (`CommandBufferShared::held_for_turn`): once it went, each later batch of it (a command
//!   buffer splits at an event wait after its work, or at a signal before more:
//!   `command_buffer/batches.rs`) goes as it is ready, as Metal schedules a command buffer whole.
//!   Held for the turn behind its own first batch, the rest would wait for that batch's
//!   completion: a command buffer whose first batch waits on the GPU for a signal not yet given
//!   (another `VkQueue`'s, or the caller's, `Event::reserve_vk_signal`) would read `Scheduled` only
//!   once that signal came, and another queue's command buffer that waits for one of its later
//!   signals would be held meanwhile (its wait goes to the `VkQueue` once the signal is
//!   submitted), where Metal schedules both at once (`runtime_lane::submission_checks`: a program
//!   that gives the signal once the other command buffer is scheduled waited for the watchdog,
//!   `command_buffer/stall.rs`).
//! - **When it goes.** A pass that holds such a batch leaves the scheduler's thread waiting for
//!   the completion that lets it go, or that makes the batch of its queue whose run does so the
//!   oldest on its `VkQueue` (`unshared_hold::turn_through`), not for each completion before it;
//!   where the oldest batch is its own queue's, for the time its expected run brings the work left
//!   under the budget, so the next batch goes before the one ahead ends; and, where 2 ms holds it
//!   and 16 ms would not, for the other queue's last batch there to be 50 ms old. A sleep that
//!   ends at such a time is no sleep slice's (`WorkCounts::scheduler_slice_rescues` counts none
//!   for it). A committing or waiting thread's pass leaves such a batch to the scheduler's thread
//!   (`WorkCounts::scheduler_at_commit_left_turns`).
//! - **What it costs.** A batch already submitted runs to its end: one longer than its queue's
//!   recent ones holds every later command buffer there for its whole length, as one command
//!   buffer is one submission. The GPU waits for the scheduler's thread only where that thread
//!   answers a completion later than the budget less one batch: a queue alone keeps 16 ms queued,
//!   two queues submitting keep 2 ms each. Every batch on such a device takes a place in its
//!   `VkQueue`'s flight, under the lock its waits take already. Counted:
//!   `WorkCounts::heads_held_for_queue_turns` (each look that held a head so).

impl crate::Device {
    /// Whether the device's command queues take turns on the `VkQueue`s they share (the module
    /// docs): its policy is `QueueTurnPolicy::Fair`.
    pub(crate) fn takes_turns(&self) -> bool {
        self.compile_device().policy().queue_turns == metal2vulkan::device::QueueTurnPolicy::Fair
    }

    /// Whether a batch of a command queue not made unshared waits for its turn on `VkQueue`
    /// `index` (the module docs): the device's queues take turns, and two or more command queues
    /// hold it.
    pub(crate) fn takes_turns_on(&self, index: usize) -> bool {
        self.takes_turns() && self.shared().queues[index].share.held_by_several()
    }
}
