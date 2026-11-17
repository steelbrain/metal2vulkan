//! Batches that go to the driver in one kernel submission (`CompletionPolicy::Grouped`).
//!
//! On a device whose check found one `vkQueueSubmit` of `VkSubmitInfo`s that signal nothing to
//! cost about one of its submissions (`submit_check.rs`, `Check::Merge`: Mesa's common queue
//! merges each into the next one's kernel submission, `vulkan/runtime/vk_queue.c`), batches that
//! signal nothing but their queue's timeline may wait and go in one call for the cost of one
//! kernel submission. The question is the device's, and its check answers it, never the driver's
//! name nor its GPU's: a device whose check found otherwise, or has not answered, takes the
//! ungrouped path, each batch its own `vkQueueSubmit`, which is what every batch took before
//! grouping and is exact alike -- a group changes when a `vkQueueSubmit` is made and nothing it
//! submits. Each command buffer of a group is exact on every driver by Vulkan's own rules (the
//! bullets below), so the check decides only where grouping pays.
//!
//! - **Marked under `CompletionPolicy::Exact`.** Under `Exact`, a batch waits here only where
//!   its command buffer is marked to join the next ([`CommandBuffer::set_completion_joins_next`]);
//!   the first batch of its `VkQueue` found ready behind such batches in the same pass that is not
//!   marked may go last in their group (`CommandBufferShared::closes_groups`), and the group then
//!   goes at once (`GroupEnd::Closed`), that batch's own signals ending it, so it is seen complete
//!   at its own end and held back for nothing. No group is kept past its pass under `Exact`
//!   ([`Group::keeps`], `Alone::Unkept`): a kept group may reach the GPU after the work ahead of
//!   it ended, which would make the unmarked batch behind it complete later.
//! - **What waits.** A batch the scheduler submits waits here, submitted by nothing yet, where it
//!   signals nothing but its queue's timeline and its events' semaphores: no binary semaphore,
//!   no fence; binds no sparse update; does no first work; moves none of the caller's wrapped
//!   images; and waits for no event index a batch held signals (only the group's end would meet
//!   it: on one `VkQueue` that wait would never be met). Every other batch goes as before, after
//!   the group held on its `VkQueue` went (`Submitting::AfterGroup`), and so does one of another
//!   `VkQueue`, so a wait there for a held signal follows the group's submission too. A batch
//!   that waits for anything not seen complete (an event, another `VkQueue`'s work, first work)
//!   leads a group and never joins one: it goes after a group held, in a submission of its own.
//!   An unshared queue's batch on a `VkQueue` other queues' work is held back from
//!   (`unshared_hold.rs`) never waits either: held, it would go with the next batch of theirs and
//!   be seen complete at that batch's end, however long, and kept, behind the batch the hold let
//!   go meanwhile.
//!   A group goes at once when it holds [`GROUP_BATCHES`], so no submission holds more.
//! - **Bounded by its expected run.** The driver runs a group as one kernel job, which a kernel's
//!   job timeout ends as a whole: Linux's amdgpu ends one past 2 s by default, with no error the
//!   device reports (`gpu_time.rs`), where each batch alone may finish within it. So a batch joins
//!   only where its queue's expected GPU run is known and keeps the group's expected runs, summed,
//!   within [`GROUP_RUN`] (16 ms): merging adds at most that much of the others' expected work to
//!   any batch's job. The expected run follows the queue's command buffers' own GPU times as each
//!   completes (`CommandBufferShared::place_gpu_times`), by their timestamps, which bracket their
//!   own work alone, whatever other `VkDevice`s the GPU ran meanwhile; where the device answers GPU
//!   times from the host (`DeviceOptions::gpu_times_from_host`), by those host times, which hold
//!   other clients' work on the GPU too, so under contention there a queue's batches may stop
//!   grouping. Otherwise the group goes first (`GroupEnd::Long`), and the batch leads the next
//!   group where its own run is known and within the bound, else goes alone: a queue whose batches
//!   none was seen complete yet, or whose recent ones ran longer, groups none until it has a run
//!   within the bound. An expected run is a few recent batches' (raised at once by a longer one,
//!   halved toward a shorter one), so a batch longer than its queue's recent ones may still join
//!   a group; the others' expected runs it then carries stay within the bound.
//! - **Checked ahead.** A batch with submission checks is ready only once they ran
//!   (`checks.rs`), which the scheduler does for each line's head; here it runs those of the
//!   batches right behind a head that goes too, so a line of checked batches (the consumer checks
//!   every command buffer) is found ready together (`scheduler::check_heads`).
//! - **One submission.** [`Group::submit`] submits every batch held, in order, in one
//!   `vkQueueSubmit` where only the last `VkSubmitInfo` signals: each event's semaphore to the
//!   highest index a batch held signals it to, then the queue's timeline to the last batch's
//!   value. Each is above every other batch's signal of it, so it reaches theirs too ("the
//!   semaphore's value is set to the signal value" once the batch's commands complete, and a value
//!   reached is reached by every wait for a lower one; the first synchronization scope of a signal
//!   holds "every command in the same batch ... and all commands that occur earlier in submission
//!   order", `vkQueueSubmit`). An event's indices rise in its signals' submission order
//!   (`event_timeline.rs`), so its earlier values land with the group's end, late, never wrong:
//!   a wait on another `VkQueue` for one is met by the higher index, and the host's waits, value
//!   reads and listeners see the event's value rise once that index is seen. Each batch keeps its
//!   own commands, barriers and its end's barrier to the host. Only the first batch may wait for a
//!   value not seen complete, for a signal submitted before it, the scheduler's order: the merge
//!   gathers every batch's waits before the first one's commands, so a later batch's wait would
//!   hold back the earlier ones' start until it is met, and none is held.
//! - **Taken as made.** What a batch's submission changes in the scheduler's state (the tracker's
//!   plan, its queue's value, the completed values its queue waited for) is changed as it is held,
//!   so the next batch is planned after it. Where it ends its command buffer, on which no switch of
//!   storage forms rides, the command buffer's kernel end and its transition to `Scheduled` come as
//!   it is held, its submission handlers run before that on the device's handler threads, which
//!   the scheduler's thread hands them to as its pass goes on, as a submitted batch's
//!   ([`Group::schedule_held`]), as Metal schedules a command
//!   buffer committed behind running work at once: on an M3 Max under macOS 27.0.1, each of 8
//!   committed behind a 55 ms kernel on its queue was `Scheduled` 0.04 to 0.18 ms after its commit,
//!   its scheduled handler run, about 55 ms before that kernel ended (probe `scheduled`). Its queue
//!   waiter's watch comes once the group is submitted. A submission that fails without losing the
//!   device changed nothing (`vkQueueSubmit`), so the events' semaphores and the timeline are
//!   brought to the group's end by a submission of those signals alone, the waits taken are put
//!   back, and each batch's command buffer fails (one already `Scheduled` from there); where that
//!   fails too, the device is lost to the runtime, its timeline short of the values its tracker
//!   holds.
//! - **Kept while its queue runs earlier work** ([`Group::keeps`]). A pass that finds no batch
//!   ready sends the group, unless two or more submissions of its `VkQueue` are not seen complete
//!   (`QueueSlot::pending_beyond_one`): the GPU runs those first in any case, in submission order,
//!   so the group stays held for the batches committed meanwhile, gated on the value whose
//!   completion leaves one submission in flight, at which a completion turn wakes the scheduler.
//!   It goes with one in flight rather than two: on the UHD 770 two cut the kernel submissions
//!   less (30 a frame against 24, at 67 command buffers a frame committed 1 ms apart behind 1.3 ms
//!   of GPU work each) and left the GPU as idle (0.62 ms a frame with either).
//!   It goes then, or once it fills, a batch that cannot join sends it, a head waits for the GPU
//!   (at a gate, or its checks for a signal to land, which may be the group's), or a thread waits
//!   for one of its command buffers' completion (`CommandBuffer::wait_until_completed` marks it
//!   and wakes the scheduler: [`CommandBufferShared::waits_for_completion`]). Nor is it kept
//!   where the thread runs, once the pass ends, the submission handlers of a command buffer of
//!   another logical queue (a batch held, or one submitted, taken by a pass that settled a switch
//!   of storage forms: `scheduler::holds_back_another_queue`; every other pass hands them to the
//!   device's handler threads, where they hold back none of its passes): it goes first, so no
//!   handler that waits for another queue's work holds that work back, as Metal schedules one
//!   queue's command buffers whatever another queue's handler does (probe
//!   `cross_queue_handler_waits`). The handlers of the batches' own queue keep it held: Metal
//!   runs one queue's handlers one at a time, so a handler there that waits for a completed
//!   handler of its own queue never sees it either (the probe's `scheduled-one-queue`). A group
//!   whose batch ends a command buffer with a switch of storage forms riding on it is never kept:
//!   the switch settles by the submission's outcome, before the command buffer is `Scheduled`.
//!   Nor is one on a `VkQueue` other queues' work is held back from for an unshared queue
//!   (`unshared_hold.rs`): the hold bounds the work ahead there by its expected GPU time, and a
//!   group kept past the point the hold let its batches go would leave the GPU idle once the
//!   work ahead ended.
//!   A head of another queue whose wait is for a value only a batch the group holds signals
//!   (`HeldFor::EventValue`: no submitted signal reaches it yet) does not send it: it is held
//!   until the group goes, at the completion turn that leaves one submission in flight at the
//!   latest, and is then submitted with its wait on the `VkQueue`
//!   (`runtime_lane::grouped_submissions`). A kept batch is seen
//!   complete late by its group's later batches' GPU time, as any member, and where the one
//!   submission left in flight ends before the group reaches the GPU, by that gap besides: at most
//!   a completion turn's reaction and the group's `vkQueueSubmit`. Counted:
//!   `WorkCounts::groups_kept_pending`, `WorkCounts::groups_sent_for_waits`.
//! - **Counted by its end.** Each group held counts once by what sent it ([`GroupEnd`],
//!   `WorkCounts::group_ends_full` and kin): so the consumer's counts say why its groups hold
//!   fewer batches than they may. Each batch on a `VkQueue` whose batches group that goes in a
//!   kernel submission of its own counts once by why ([`Alone`],
//!   `WorkCounts::batches_alone_unshared` and kin): what refused it a place with no group held
//!   (counted nowhere else), a committing thread's submission, or, where its group went holding
//!   it alone, what sent that group; so the consumer's counts say why its batches did not group.
//! - **No allocation.** A group's submission names at most [`MOST_INFOS`] `VkSubmitInfo`s, built
//!   in arrays on the thread's stack; its end's signals and its submissions go to lists the group
//!   keeps with their room: the scheduler thread allocates nothing for it
//!   (`runtime_lane::frame_costs`, `a_frame_s_commits_allocate_within_their_bound`).

use super::submit::Submission;
use super::{submission_error, CommandBuffer, CommandBufferShared};
use crate::degradation::{Degradation, NoticesFor};
use crate::device::relay::Relayed;
use crate::error::{invalid, CommandBufferError, Error, Problem};
use crate::exit_guard::{self, Call};
use crate::gpu_time;
use crate::submit_failure::{submit_failed, try_lock, Carried, Failed, TypeFlags};
use crate::Device;
use ash::vk;
use metal2vulkan::device::CompletionPolicy;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// The most batches one kernel submission holds: the first is seen complete late by the GPU time
/// of at most this many less one.
pub(crate) const GROUP_BATCHES: usize = 4;

/// The most expected GPU run, in nanoseconds, of the batches one group holds, summed (the module
/// docs, "Bounded by its expected run"): 16 ms, the owner's line for an intolerable wait, which no
/// member's completion is then seen late by for the others' expected runs. It is 1/125 of the job
/// timeout Linux's amdgpu ends a kernel job at by default (2 s, its `lockup_timeout`, Linux 7.3.3),
/// past which a Radeon 780M's kernel ended a single command buffer of 3.2 s and dropped the one
/// behind it (`gpu_time.rs`).
pub(crate) const GROUP_RUN: u64 = 16_000_000;

/// The most `VkSubmitInfo`s a batch names: its work's and its finale's (`submit.rs`).
const BATCH_INFOS: usize = 2;

/// The most `VkSubmitInfo`s one group's `vkQueueSubmit` names: [`BATCH_INFOS`] for each batch
/// held, and for the batch after them its first work's beside its own.
const MOST_INFOS: usize = GROUP_BATCHES * BATCH_INFOS + 1 + BATCH_INFOS;

/// Whether `device`'s batches on `VkQueue` `queue` may go to the driver in one kernel submission
/// at all: its check found a submission that signals nothing merged into the next (the module
/// docs), the queue's failures are not relayed (each relayed batch takes a fence,
/// `device/relay.rs`), and its encoder trail is off (`encoder_trail.rs`: each batch goes alone).
pub(crate) fn merges(device: &Device, queue: usize) -> bool {
    device.submissions_merge_now()
        && device.shared().flights(queue).is_none()
        // The encoder trail submits each batch alone (`encoder_trail.rs`).
        && !device.shared().encoder_trail
}

/// Whether every batch of `device`'s on `VkQueue` `queue` may wait in a group: its policy groups
/// them (`CompletionPolicy::Grouped`) and they merge ([`merges`]). Under `CompletionPolicy::Exact`
/// only a command buffer marked to join the next may (`CommandBufferShared::groups`).
pub(crate) fn groups(device: &Device, queue: usize) -> bool {
    device.compile_device().policy().completion == CompletionPolicy::Grouped
        && merges(device, queue)
}

impl CommandBuffer {
    /// Mark the command buffer's completion as joining the next one's: a runtime extension, no
    /// Metal call (the crate's README, "Runtime extensions"). Metal has no counterpart; it reports
    /// each command buffer complete at its own end. Under `CompletionPolicy::Exact`, on a device
    /// whose batches merge (the device's check found one `vkQueueSubmit` of `VkSubmitInfo`s that
    /// signal nothing to cost about one of its submissions, `Device::submissions_merge`; its
    /// queue's failures not relayed), a batch of a marked command buffer may wait in the
    /// scheduler's group as `CompletionPolicy::Grouped` lets every batch wait
    /// (`command_buffer/group.rs`): up to four batches of one `VkQueue` found ready together go
    /// in one kernel submission, only the last signalling. A batch of a command buffer not
    /// marked never waits in a group, and is never held back for one: it may go last in the
    /// group held before it, in the same pass and the same call, which then goes at once, and it
    /// is seen complete at its own end, as alone. No group is kept past the pass that found it
    /// under `Exact` (`CompletionPolicy::Grouped` keeps one while earlier work of its queue
    /// runs), so no marked batch waits for later work to be committed.
    ///
    /// What it delays: the marked command buffer is seen complete (its status, completed
    /// handlers and waits), and the event values it signals are reached, once the last batch of
    /// its kernel submission completed: late by the GPU time of the batches after it there, at
    /// most three, never early and in commit order. Its submission waits for the next batch of
    /// its queue found ready in the same pass of the scheduler, at most that pass. The
    /// submission checks of the batches right behind it, the first one not marked among them, run
    /// with its own, before its submission, so that they are found ready together; each still
    /// runs once, in commit order, before its own batch's submission. Its commands, barriers and
    /// GPU times are unchanged. The device announces `Degradation::GroupedCompletion` once, at its
    /// first such submission.
    ///
    /// It changes nothing where it cannot apply: under `CompletionPolicy::Grouped`, which groups
    /// every batch already; on a device whose check did not find submissions merged, has not
    /// answered, or relays its queues' failures (MoltenVK); with the encoder trail on; and for a
    /// batch that groups in no case (it signals or waits for a binary semaphore, takes a fence,
    /// binds a sparse update, does first work, or is an unshared queue's on a `VkQueue` other
    /// queues' work is held back for it). Counted: `WorkCounts::grouped_submissions`,
    /// `WorkCounts::batches_grouped`, `WorkCounts::group_ends_closed` and kin.
    ///
    /// Refused, changing nothing, once the command buffer is committed (`Invalid`,
    /// `Problem::AlreadyCommitted`).
    pub fn set_completion_joins_next(&self, joins: bool) -> Result<(), Error> {
        let state = self.state();
        if state.enqueued() || state.committing {
            return Err(invalid(
                "set_completion_joins_next",
                Problem::AlreadyCommitted,
            ));
        }
        match joins {
            true => self.0.group_hold.fetch_or(JOINS_NEXT, Ordering::SeqCst),
            false => self.0.group_hold.fetch_and(!JOINS_NEXT, Ordering::SeqCst),
        };
        Ok(())
    }
}

impl CommandBufferShared {
    /// Whether its batches may wait in a group: every batch where its queue's batches group
    /// ([`groups`]), and under `CompletionPolicy::Exact` those of a command buffer marked to join
    /// the next where they merge ([`merges`], [`CommandBuffer::set_completion_joins_next`]); never
    /// an unshared queue's on a `VkQueue` other queues' work is held back from (the module docs).
    pub(crate) fn groups(&self) -> bool {
        let queue = self.queue.vk_queue();
        let may = match self.joins_next() {
            true => merges(&self.device, queue),
            false => groups(&self.device, queue),
        };
        may && self.not_held_apart(queue)
    }

    /// Whether a batch of it that waits in no group ([`groups`](Self::groups)) may still go last
    /// in the group held before it on its `VkQueue`, which then goes at once: its batches merge
    /// ([`merges`]), and it is no unshared queue's on a `VkQueue` other queues' work is held back
    /// from. So under `CompletionPolicy::Exact` the command buffer right behind marked ones goes
    /// in their kernel submission, seen complete at its own end
    /// ([`CommandBuffer::set_completion_joins_next`]).
    pub(crate) fn closes_groups(&self) -> bool {
        let queue = self.queue.vk_queue();
        merges(&self.device, queue) && self.not_held_apart(queue)
    }

    /// Whether it is not an unshared queue's on `VkQueue` `queue` while other queues' work is held
    /// back there for it, whose batches never wait in a group (the module docs).
    fn not_held_apart(&self, queue: usize) -> bool {
        !(self.queue.unshared() && self.device.holds_back_for_unshared(queue))
    }

    /// Whether it is marked to join the next ([`CommandBuffer::set_completion_joins_next`]): set
    /// before its commit, read after it.
    fn joins_next(&self) -> bool {
        self.group_hold.load(Ordering::SeqCst) & JOINS_NEXT != 0
    }

    /// Whether a committing thread submits a batch of it that may wait in a group
    /// ([`groups`]) alone, rather than leave it to the scheduler's thread
    /// (`scheduler::plain`, `CommandBuffer::commit_and_submit`): where its `VkQueue` runs at most
    /// one earlier submission not seen complete, the pass that held it would send the group at its
    /// end ([`Group::keeps`]), so it reaches the GPU no later, and the commit saves the scheduler's
    /// wake and pass. Where two or more run, the scheduler's thread holds it in the group, with
    /// the commits that join it. Read without the queue's lock, as `keeps` reads it: a submission
    /// seen complete meanwhile only lowers the count.
    pub(crate) fn goes_alone_at_commit(&self) -> bool {
        (self.device.shared().queues[self.queue.vk_queue()].pending_beyond_one()).is_none()
    }

    /// Whether a committing thread leaves its batches to the scheduler's thread, as its
    /// `VkQueue`'s batches' `vkQueueSubmit` calls are slow (`device/submit_calls.rs`): read
    /// without the queue's lock.
    pub(crate) fn submits_slowly(&self) -> bool {
        self.device.shared().submits_slowly(self.queue.vk_queue())
    }

    /// Its queue's identity: the logical queue its batches line up on in the scheduler's order
    /// (`scheduler/order.rs`).
    pub(crate) fn logical_queue(&self) -> u64 {
        self.queue.id()
    }

    /// A batch of it was held in the scheduler's group.
    pub(crate) fn held_in_group(&self) {
        self.group_hold.fetch_or(IN_GROUP, Ordering::SeqCst);
    }

    /// Its batches held in the group went, or failed.
    pub(crate) fn left_group(&self) {
        self.group_hold.fetch_and(!IN_GROUP, Ordering::SeqCst);
    }

    /// A thread begins to wait for its completion: where a batch of it is held in the group, the
    /// scheduler is woken, and sends the group at once ([`Group::keeps`]). The scheduler marks a
    /// batch held before it looks for a waiter, and the waiter marks its wait before it looks for
    /// a batch held, each on the same word, so one of the two sees the other.
    ///
    /// Else the thread submits for itself what still waits in the scheduler's inbox, before it
    /// sleeps (`Scheduler::submit_before_waiting`): a commit submits nothing, as Metal's returns
    /// before its command buffer is scheduled, so a thread that commits and then blocks would
    /// otherwise wait for a wake and a whole pass first.
    pub(crate) fn waits_for_completion(&self) {
        let before = self.group_hold.fetch_or(WAITED_FOR, Ordering::SeqCst);
        let scheduler = &self.device.shared().scheduler;
        if before & IN_GROUP != 0 {
            scheduler.wake();
            return;
        }
        scheduler.submit_before_waiting(&self.device);
    }

    /// A thread begins to wait for its scheduling (`CommandBuffer::wait_until_scheduled`): it
    /// submits for itself what still waits in the scheduler's inbox before it sleeps
    /// (`Scheduler::submit_before_waiting`), as a thread waiting for the completion does.
    ///
    /// It marks no wait on the group, where [`waits_for_completion`](Self::waits_for_completion)
    /// does: a batch held in a group makes its command buffer `Scheduled` as it is held
    /// ([`Group::schedule_held`]), so this wait is answered by the hold, and sending the group for
    /// it would spend its kernel submission for nothing. Nor does the pass it makes submit a
    /// batch of a queue whose batches group at all (`scheduler::plain_ready`).
    pub(crate) fn waits_for_scheduling(&self) {
        (self.device.shared().scheduler).submit_before_waiting(&self.device);
    }

    /// Whether a thread waits for its completion.
    fn is_waited_for(&self) -> bool {
        self.group_hold.load(Ordering::SeqCst) & WAITED_FOR != 0
    }
}

/// A command buffer's `group_hold` bit: a batch of it is held in the scheduler's group.
const IN_GROUP: u8 = 1;

/// A command buffer's `group_hold` bit: a thread waits for its completion.
const WAITED_FOR: u8 = 2;

/// A command buffer's `group_hold` bit: it is marked to join the next
/// ([`CommandBuffer::set_completion_joins_next`]).
const JOINS_NEXT: u8 = 4;

/// Whether a batch of expected run `run` (nanoseconds; `None` where its queue has none yet) may
/// join batches whose expected runs sum to `held`: its run is known and the sum stays within
/// [`GROUP_RUN`] (the module docs, "Bounded by its expected run").
pub(crate) fn within_run(held: u64, run: Option<u64>) -> bool {
    run.is_some_and(|run| held.saturating_add(run) <= GROUP_RUN)
}

/// What sent a group held (`WorkCounts::group_ends_full` and kin).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GroupEnd {
    /// It held [`GROUP_BATCHES`].
    Full,
    /// A batch waits: for an event index a batch held signals, which only the group's end meets,
    /// or for anything else not seen complete (it goes after the group, never in it).
    Wait,
    /// A batch that may not wait in a group (a binary semaphore, a fence, a sparse update, a move
    /// of bytes that waits for every submission), or one of another `VkQueue`.
    Apart,
    /// A batch that may join did first work or names a wrapped image: it went last in the group's
    /// call.
    FirstWork,
    /// A batch that may otherwise join is of unknown length, or would take the group's expected
    /// run past [`GROUP_RUN`].
    Long,
    /// The pass found no batch left to submit.
    Drained,
    /// The pass found none ready, with submission checks or host steps left for the thread to run.
    Checks,
    /// The pass found none ready, the rest held for events, gates, binary semaphores or decodes.
    Held,
    /// A batch of a command buffer not marked to join the next went last in it under
    /// `CompletionPolicy::Exact` ([`CommandBuffer::set_completion_joins_next`]): it goes at once.
    Closed,
}

/// Why a batch on a `VkQueue` whose batches group ([`groups`]) went in a kernel submission of its
/// own (`WorkCounts::batches_alone_unshared` and kin): each such batch counted once, by the first
/// reason that holds in this order. The first ten are batches that went with no group held and
/// were never held; the rest were held, and their group went holding them alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Alone {
    /// An unshared queue's batch on a `VkQueue` other queues' work is held back from for it
    /// (`unshared_hold.rs`), which never waits in a group (the module docs).
    Unshared,
    /// It waits for a binary semaphore (`encode_wait_for_vk_semaphore`, a sync file's wait).
    BinaryWait,
    /// It signals a binary semaphore (`encode_signal_vk_semaphore`, a completion sync file).
    BinarySignal,
    /// It binds a sparse update.
    Sparse,
    /// A segment cut its work into pieces, each a `VkSubmitInfo` of its own.
    Pieced,
    /// It ends its command buffer at a fence (`fences.rs`).
    Fence,
    /// Its queue's expected run is unknown: none of its command buffers was seen complete.
    RunUnknown,
    /// Its queue's expected run alone is past the group's bound.
    RunLong,
    /// It took the device's first work, staged copies or a fill, or names the caller's wrapped
    /// image, with no group held: such a batch goes last in a group's call, never in a group.
    FirstWork,
    /// A committing thread submitted it alone, its `VkQueue` running at most one earlier
    /// submission not seen complete (`CommandBufferShared::goes_alone_at_commit`).
    AtCommit,
    /// Held; the pass found no batch ready, and its `VkQueue` ran at most one earlier submission
    /// not seen complete, so the group was not kept ([`Group::keeps`]).
    OnePending,
    /// Held; its `VkQueue` holds an unshared queue whose isolation holds other queues' work back
    /// there, where no group is kept.
    UnsharedHold,
    /// Held; a head waited for the GPU (at its gate, or for a signal its checks wait for).
    GpuWait,
    /// Held; the thread ran another logical queue's submission handlers next.
    Handlers,
    /// Held; a thread waited for its command buffer's completion.
    WaitedFor,
    /// Held and kept while its queue ran earlier work, and no batch joined it before one
    /// submission was left in flight.
    Kept,
    /// Held; a batch that may not join it came next: of another `VkQueue`, one refused, one
    /// past the group's bound, or one that waits for a signal it holds.
    SentByNext,
    /// Held; a switch of storage forms rides on it, or the device was lost.
    Other,
    /// Held, of a command buffer marked to join the next under `CompletionPolicy::Exact`
    /// ([`CommandBuffer::set_completion_joins_next`]); the pass found no batch ready, and no
    /// group is kept under `Exact`.
    Unkept,
}

impl Alone {
    /// How many there are, for the counters' array.
    pub(crate) const COUNT: usize = 19;
}

/// One `VkSubmitInfo`'s lists, kept with their room from batch to batch.
#[derive(Default)]
pub(crate) struct Info {
    wait_semaphores: Vec<vk::Semaphore>,
    wait_values: Vec<u64>,
    wait_stages: Vec<vk::PipelineStageFlags>,
    buffers: Vec<vk::CommandBuffer>,
    signal_semaphores: Vec<vk::Semaphore>,
    signal_values: Vec<u64>,
}

/// What one `VkSubmitInfo` waits for, runs and signals, borrowed.
pub(crate) struct InfoParts<'a> {
    pub(crate) waits: (&'a [vk::Semaphore], &'a [u64], &'a [vk::PipelineStageFlags]),
    pub(crate) buffers: &'a [vk::CommandBuffer],
    pub(crate) signals: (&'a [vk::Semaphore], &'a [u64]),
}

impl Info {
    fn fill(&mut self, parts: &InfoParts<'_>) {
        let (semaphores, values, stages) = parts.waits;
        self.wait_semaphores.clear();
        self.wait_semaphores.extend_from_slice(semaphores);
        self.wait_values.clear();
        self.wait_values.extend_from_slice(values);
        self.wait_stages.clear();
        self.wait_stages.extend_from_slice(stages);
        self.buffers.clear();
        self.buffers.extend_from_slice(parts.buffers);
        let (semaphores, values) = parts.signals;
        self.signal_semaphores.clear();
        self.signal_semaphores.extend_from_slice(semaphores);
        self.signal_values.clear();
        self.signal_values.extend_from_slice(values);
    }
}

/// A batch held: its `VkSubmitInfo`s, how many command buffers they hold, its submission,
/// published once the group is submitted, and its queue's expected run as it was held, in
/// nanoseconds.
struct Member {
    infos: Vec<Info>,
    buffers: usize,
    submission: Submission,
    run: u64,
}

/// The batches held for one kernel submission on one `VkQueue` (the module docs), in the
/// scheduler's state.
#[derive(Default)]
pub(crate) struct Group {
    queue: usize,
    members: Vec<Member>,
    /// It stayed held past a pass for the work its queue ran ([`keeps`](Group::keeps)).
    kept: bool,
    /// Its last batch held is of a command buffer not marked to join the next, under
    /// `CompletionPolicy::Exact` ([`closed`](Group::closed)).
    closed: bool,
    /// The completed values `queue` waited for before the first batch held: put back where the
    /// group's submission fails.
    waited: Vec<u64>,
    /// The events' semaphores the batches held signal, each with the lowest and the highest index
    /// they signal it to: the group's end signals the highest (the module docs).
    events: Vec<(vk::Semaphore, u64, u64)>,
    /// The lists of batches submitted, kept with their room.
    spare: Vec<Vec<Info>>,
    /// The end's signals: each event's semaphore, then the queue's timeline, and their values;
    /// kept with their room.
    end_semaphores: Vec<vk::Semaphore>,
    end_values: Vec<u64>,
    /// The batches the group's last submission made, in order, until the scheduler takes them
    /// ([`made`](Group::made)); kept with their room.
    made: Vec<Submission>,
    /// What the group's last submission did, until the scheduler takes it
    /// ([`take_outcome`](Group::take_outcome)): made ([`made`](Group::made)), or failed.
    outcome: Option<Result<(), CommandBufferError>>,
}

/// A batch submitted after the group, in the same `vkQueueSubmit` ([`Group::submit`]).
pub(crate) struct After<'a> {
    /// Its `VkQueue`: the group's, where one is held.
    pub(crate) queue: usize,
    pub(crate) infos: &'a [vk::SubmitInfo<'a>],
    pub(crate) buffers: usize,
    /// The value it signals on the queue's timeline.
    pub(crate) value: u64,
    pub(crate) fence: Option<vk::Fence>,
    pub(crate) relayed: &'a Arc<Relayed>,
    /// What it carried, for the line its failed submission logs (`submit_failure.rs`): called
    /// only once the call failed.
    pub(crate) carry: &'a dyn Fn(&mut Carried, &TypeFlags),
}

impl Group {
    /// The `VkQueue` of the batches held.
    pub(crate) fn queue(&self) -> usize {
        self.queue
    }

    /// Whether it holds a batch.
    pub(crate) fn holds(&self) -> bool {
        !self.members.is_empty()
    }

    /// Whether a batch on `VkQueue` `queue` may be submitted now, the group held so far going
    /// first: the group is empty, or the batch may wait in it (`joins`) on its queue with room.
    pub(crate) fn admits(&self, queue: usize, joins: bool) -> bool {
        self.members.is_empty()
            || (joins && self.queue == queue && self.members.len() < GROUP_BATCHES)
    }

    /// Whether a batch of expected run `run` (nanoseconds; `None` where unknown) may wait in the
    /// group as far as its run goes ([`within_run`]).
    pub(crate) fn fits(&self, run: Option<u64>) -> bool {
        within_run(self.members.iter().map(|member| member.run).sum(), run)
    }

    /// Whether a batch held signals the event of `semaphore` to `index` or above: only the group's
    /// end meets a wait for it, so a batch with that wait goes after the group, in a submission
    /// of its own (the module docs). A lower index was signalled before the group.
    pub(crate) fn meets(&self, (semaphore, index): (vk::Semaphore, u64)) -> bool {
        (self.events.iter()).any(|&(signaled, lowest, _)| signaled == semaphore && index >= lowest)
    }

    /// Whether the group holds [`GROUP_BATCHES`]: the scheduler submits it at once. A batch that
    /// may join, found with room, goes with at most a batch less held, so no submission holds
    /// more.
    pub(crate) fn full(&self) -> bool {
        self.members.len() >= GROUP_BATCHES
    }

    /// Whether its last batch held is of a command buffer that waits in no group, which went last
    /// in it ([`CommandBufferShared::closes_groups`]): the scheduler submits it at once, so that
    /// batch is held back for nothing.
    pub(crate) fn closed(&self) -> bool {
        self.closed
    }

    /// A batch on `VkQueue` `queue` is held, before its submission changes the scheduler's
    /// state: `waited` is what `queue` waited for so far, put back where the group's submission
    /// fails. Its lists come once its changes are made ([`hold`](Group::hold)).
    pub(crate) fn open(&mut self, queue: usize, waited: &[u64]) {
        if self.members.is_empty() {
            self.queue = queue;
            self.waited.clear();
            self.waited.extend_from_slice(waited);
        }
    }

    /// The batch [`open`](Group::open) took: its `VkSubmitInfo`s are `infos` (at most
    /// [`BATCH_INFOS`]), with `buffers` command buffers, and its submission is published once the
    /// group's is made; `run` its queue's expected run ([`fits`](Group::fits)). It signals each
    /// event's semaphore it names to its index, which the group's end signals instead.
    pub(crate) fn hold(
        &mut self,
        infos: &[InfoParts<'_>],
        buffers: usize,
        submission: Submission,
        (run, closes): (u64, bool),
    ) {
        self.closed = closes;
        for (semaphore, index) in submission.event_signals() {
            match self.events.iter_mut().find(|(held, ..)| *held == semaphore) {
                Some((_, lowest, highest)) => {
                    *lowest = (*lowest).min(index);
                    *highest = (*highest).max(index);
                }
                None => self.events.push((semaphore, index, index)),
            }
        }
        let mut held = self.spare.pop().unwrap_or_default();
        held.resize_with(infos.len(), Info::default);
        for (info, parts) in held.iter_mut().zip(infos) {
            info.fill(parts);
        }
        self.members.push(Member {
            infos: held,
            buffers,
            submission,
            run,
        });
    }

    /// The batch [`hold`](Group::hold) took last, after the section that held it ended: where it
    /// ends its command buffer and no switch of storage forms rides on it, its kernel end and its
    /// transition to `Scheduled` now, not once the group is submitted, and its submission
    /// handlers for the scheduler's thread to run now, once its lock is free (the module docs):
    /// whether it has them.
    pub(crate) fn schedule_held(&mut self) -> bool {
        (self.members.last_mut()).is_some_and(|member| member.submission.schedule_held())
    }

    /// Whether the batches held stay held past the pass that found no batch ready (the module
    /// docs): the group has room, two or more submissions of its `VkQueue` are not seen complete,
    /// no batch held ends a command buffer with a switch of storage forms riding on it (it settles
    /// by the submission's outcome), no thread waits for one of their command buffers, and the
    /// device is not lost. The scheduler asks only where no handler it runs next is of another
    /// logical queue's command buffer (the module docs).
    /// The value of that queue's timeline whose completion leaves one submission in flight, for
    /// the completion turns to wake the scheduler at; else why it goes, for a group of one batch
    /// ([`Alone`]). Counted once per group, and once more where a waiter sends a group kept
    /// before.
    pub(crate) fn keeps(&mut self) -> Result<u64, Alone> {
        let Some(first) = self.members.first() else {
            return Err(Alone::Other);
        };
        let device = first.submission.buffer().device();
        if self.full()
            || device.is_lost()
            || (self.members.iter()).any(|member| member.submission.rides_forms())
        {
            return Err(Alone::Other);
        }
        // Under `Exact` its batches are of command buffers marked to join the next, which wait for
        // no later commit (`CommandBuffer::set_completion_joins_next`).
        if device.compile_device().policy().completion != CompletionPolicy::Grouped {
            return Err(Alone::Unkept);
        }
        if device.holds_back_for_unshared(self.queue) {
            return Err(Alone::UnsharedHold);
        }
        let counts = &device.shared().counts;
        if (self.members.iter()).any(|member| member.submission.buffer().is_waited_for()) {
            if self.kept {
                counts.group_sent_for_wait();
            }
            return Err(Alone::WaitedFor);
        }
        let Some(value) = device.shared().queues[self.queue].pending_beyond_one() else {
            return Err(match self.kept {
                true => Alone::Kept,
                false => Alone::OnePending,
            });
        };
        if !self.kept {
            self.kept = true;
            counts.group_kept_pending();
        }
        Ok(value)
    }

    /// Whether it holds exactly one batch: a group that goes so saves no kernel submission, and
    /// its batch is counted by why it went alone ([`Alone`]).
    pub(crate) fn holds_one(&self) -> bool {
        self.members.len() == 1
    }

    /// Whether its `VkQueue` has not completed `value` yet, as the host saw ([`keeps`]'s value),
    /// read again after the scheduler left its gate for the completion turns.
    ///
    /// [`keeps`]: Group::keeps
    pub(crate) fn pending_through(&self, value: u64) -> bool {
        (self.members.first()).is_some_and(|first| {
            let device = first.submission.buffer().device();
            device.shared().queues[self.queue].completed_value() < value
        })
    }

    /// What the group's last submission did, for the scheduler to publish its batches
    /// ([`made`](Group::made)) or fail them.
    pub(crate) fn take_outcome(&mut self) -> Option<Result<(), CommandBufferError>> {
        self.outcome.take()
    }

    /// The batches the group's last submission made, in order, for the scheduler to publish.
    pub(crate) fn made(&mut self) -> std::vec::Drain<'_, Submission> {
        self.made.drain(..)
    }

    /// Submit every batch held, then `after` where given, in one `vkQueueSubmit` on the group's
    /// `VkQueue` of `device`, only the last signalling (the module docs); `waited` is the
    /// scheduler's list of what that queue waited for. Inside a section of the exit guard. The
    /// held batches' outcome is kept for the scheduler ([`take_outcome`](Group::take_outcome));
    /// the call's own result is `after`'s, as the `vkQueueSubmit` returned it: on success, the
    /// host time at which the scheduler called it, the kernel end of a command buffer whose last
    /// batch it carried (now where it submitted nothing). Taken as the call is made, never once
    /// it returned: the GPU may run the work, and its timestamps place its end, before the call
    /// returns (a driver that runs it as it is submitted, lavapipe's or NVIDIA's) or before a
    /// scheduler thread preempted after the call takes the time, and Metal's `GPUEndTime` is never
    /// earlier than its `kernelEndTime`.
    pub(crate) fn submit(
        &mut self,
        device: &Device,
        waited: &mut [u64],
        after: Option<After<'_>>,
    ) -> Result<u64, vk::Result> {
        if self.members.is_empty() {
            return match after {
                None => Ok(gpu_time::host_nanos()),
                Some(after) => submit_alone(device, after),
            };
        }
        let shared = device.shared();
        let raw = shared.vk();
        let queue = self.queue;
        let count = self.members.len();
        let alone = after.is_none();
        self.kept = false;
        self.closed = false;
        let held_value = self.members[count - 1].submission.value();
        // The held batches' end: each event's semaphore to the highest index a batch held
        // signals it to, then the queue's timeline to the last one's value. It ends the
        // submission, or, where a batch goes after the group and the held ones signal an event,
        // their part of it (that batch's infos then follow a signal, so the driver submits them
        // apart).
        let ends = alone || !self.events.is_empty();
        self.end_semaphores.clear();
        (self.end_semaphores).extend((self.events.iter()).map(|&(semaphore, ..)| semaphore));
        (self.end_semaphores).push(shared.queues[queue].timeline);
        self.end_values.clear();
        (self.end_values).extend((self.events.iter()).map(|&(_, _, highest)| highest));
        self.end_values.push(held_value);
        let (end_semaphores, end_values) = (&self.end_semaphores, &self.end_values);
        // Only the held batches' last `VkSubmitInfo` signals: every other one's signals (its
        // queue's timeline and its events') are reached by the end's higher values.
        let held = || (self.members.iter()).flat_map(|member| &member.infos);
        let infos = held().count();
        let submitting = infos + after.as_ref().map_or(0, |after| after.infos.len());
        // Each batch's `VkSubmitInfo`s are its work's and its finale's (`submit.rs`), and the
        // group holds at most `GROUP_BATCHES`: none is left out of the arrays.
        assert!(
            submitting <= MOST_INFOS,
            "{submitting} VkSubmitInfos in one submission"
        );
        let signals = |at: usize| ends && at + 1 == infos;
        let mut timelines = [vk::TimelineSemaphoreSubmitInfo::default(); MOST_INFOS];
        for (at, (timeline, info)) in timelines.iter_mut().zip(held()).enumerate() {
            let values: &[u64] = if signals(at) { end_values } else { &[] };
            *timeline = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&info.wait_values)
                .signal_semaphore_values(values);
        }
        let mut submits = [vk::SubmitInfo::default(); MOST_INFOS];
        let built = submits.iter_mut().zip(timelines.iter_mut()).zip(held());
        for (at, ((submit, timeline), info)) in built.enumerate() {
            let semaphores: &[vk::Semaphore] = if signals(at) { end_semaphores } else { &[] };
            *submit = vk::SubmitInfo::default()
                .wait_semaphores(&info.wait_semaphores)
                .wait_dst_stage_mask(&info.wait_stages)
                .command_buffers(&info.buffers)
                .signal_semaphores(semaphores)
                .push_next(timeline);
        }
        let mut buffers: usize = (self.members.iter()).map(|member| member.buffers).sum();
        let last = &self.members[count - 1].submission;
        let relayed = Arc::clone(match &after {
            Some(after) => after.relayed,
            None => &last.buffer().relayed,
        });
        let (value, fence) = match &after {
            Some(after) => {
                submits[infos..submitting].copy_from_slice(after.infos);
                buffers += after.buffers;
                (after.value, after.fence)
            }
            None => (held_value, None),
        };
        let submits = &submits[..submitting];
        // SAFETY: as each batch's own submission (`submit.rs`): every command buffer is
        // executable and submitted once; every wait is for a signal submitted before; the one
        // timeline signal is above the queue's value; the `VkQueue` is used under its lock.
        let mut called = 0;
        let submitted =
            shared.submit_logged((queue, Some(value)), fence, &relayed, |vk_queue| unsafe {
                shared.counts.queue_submitted(submits.len(), buffers);
                called = gpu_time::host_nanos();
                raw.queue_submit(vk_queue, submits, fence.unwrap_or(vk::Fence::null()))
            });
        // Where the device notes queue times, the call's return, for each command buffer a batch
        // held ends (`CommandBuffer::submit_returned_time`).
        let returned = shared.timing.queue_times().then(gpu_time::host_nanos);
        exit_guard::called(Call::QueueSubmit);
        self.events.clear();
        let several = count + usize::from(!alone) > 1;
        let outcome = match submitted {
            Ok(()) => {
                for member in self.members.drain(..) {
                    self.spare.push(member.infos);
                    member.submission.buffer().kernel.submitted_at(called);
                    if member.submission.schedules() && !member.submission.scheduled() {
                        member.submission.buffer().kernel.end_at(called);
                    }
                    if let (true, Some(at)) = (member.submission.schedules(), returned) {
                        member.submission.buffer().kernel.returned_at(at);
                    }
                    self.made.push(member.submission);
                }
                if several {
                    shared.counts.batches_grouped(count + usize::from(!alone));
                }
                Ok(())
            }
            Err(error) => {
                // Its line, before the end's signals alone move the timelines on.
                let what = match alone {
                    true => "the batches a group held",
                    false => "the batches a group held, and a batch after them",
                };
                let failed = Failed {
                    what,
                    queue: Some(queue),
                    result: error,
                    infos: submits,
                    fence,
                };
                let (members, events) = (&self.members, &self.end_semaphores);
                let after = after.as_ref().map(|after| after.carry);
                submit_failed(device, &failed, |carried, types| {
                    carried.events.extend_from_slice(events);
                    for member in members {
                        let buffer = member.submission.buffer();
                        // The batch after them may be of one of their command buffers, whose
                        // state its submitter holds: that one states it.
                        match try_lock(&buffer.state) {
                            Some(state) => buffer.carry(&state, (None, None), carried, types),
                            None => carried.buffers.push(format!("#{}", buffer.id.get())),
                        }
                    }
                    if let Some(carry) = after {
                        carry(carried, types);
                    }
                });
                for member in self.members.drain(..) {
                    self.spare.push(member.infos);
                }
                waited.copy_from_slice(&self.waited);
                let failure = submission_error(device, error);
                if !device.is_lost()
                    && signal_alone(
                        device,
                        queue,
                        (&self.end_semaphores, &self.end_values),
                        &relayed,
                    )
                    .is_err()
                {
                    device.mark_lost();
                }
                if matches!(failure, CommandBufferError::DeviceLost) && alone {
                    device.drain_queues();
                }
                Err(failure)
            }
        };
        if several && outcome.is_ok() {
            let batches = count + usize::from(!alone);
            // Several command buffers' batches: none of them named.
            let _several = NoticesFor::none();
            device.degrade_once(Degradation::GroupedCompletion, || {
                (
                    format!("{batches} batches of command buffers on one queue"),
                    "the scheduler found them ready together, on a driver that merges a \
                     submission that signals nothing into the next one"
                        .to_owned(),
                )
            });
        }
        self.outcome = Some(outcome);
        submitted.map(|()| called)
    }
}

/// Submit `after` alone on its `VkQueue` of `device`: the group's submission with no batch held.
/// On success, the host time at which its `vkQueueSubmit` was called ([`Group::submit`]).
fn submit_alone(device: &Device, after: After<'_>) -> Result<u64, vk::Result> {
    let shared = device.shared();
    let raw = shared.vk();
    let After {
        queue,
        infos,
        buffers,
        value,
        fence,
        relayed,
        carry,
    } = after;
    let mut called = 0;
    // SAFETY: the batch's submission's own (`submit.rs`).
    let submitted = shared.submit_logged((queue, Some(value)), fence, relayed, |vk_queue| unsafe {
        shared.counts.queue_submitted(infos.len(), buffers);
        called = gpu_time::host_nanos();
        raw.queue_submit(vk_queue, infos, fence.unwrap_or(vk::Fence::null()))
    });
    exit_guard::called(Call::QueueSubmit);
    if let Err(result) = submitted {
        let failed = Failed {
            what: "a batch of a command buffer",
            queue: Some(queue),
            result,
            infos,
            fence,
        };
        submit_failed(device, &failed, carry);
    }
    submitted.map(|()| called)
}

/// Signal the end a failed group held (its events' semaphores to their indices, then `VkQueue`
/// `queue`'s timeline to its last value) with a submission of nothing else, after the group's
/// submission failed without losing the device (the module docs).
fn signal_alone(
    device: &Device,
    queue: usize,
    (semaphores, values): (&[vk::Semaphore], &[u64]),
    relayed: &Arc<Relayed>,
) -> Result<(), vk::Result> {
    let shared = device.shared();
    let raw = shared.vk();
    let value = values.last().copied();
    let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(values);
    let info = vk::SubmitInfo::default()
        .signal_semaphores(semaphores)
        .push_next(&mut timeline);
    // SAFETY: a submission of the signals a failed group's end held, each above its semaphore's
    // value (no batch signalled them); the `VkQueue` is used under its lock.
    let infos = [info];
    let submitted = shared.submit_logged((queue, value), None, relayed, |vk_queue| unsafe {
        shared.counts.queue_submitted(1, 0);
        raw.queue_submit(vk_queue, &infos, vk::Fence::null())
    });
    exit_guard::called(Call::QueueSubmit);
    if let Err(result) = submitted {
        let failed = Failed {
            what: "the end of a failed group's batches, its signals alone",
            queue: Some(queue),
            result,
            infos: &infos,
            fence: None,
        };
        submit_failed(device, &failed, |carried, _| {
            carried.events.extend_from_slice(semaphores);
        });
    }
    submitted
}
