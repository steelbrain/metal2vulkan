//! Batches that go to the driver in one kernel submission (`CompletionPolicy::Grouped`).
//!
//! On a driver measured to merge each `vkQueueSubmit`'s `VkSubmitInfo`s that signal nothing into
//! the next one's kernel submission ([`DriverFact::SubmissionsWithoutSignalsMerge`], whose
//! documentation holds what was measured where), batches that signal nothing but their queue's
//! timeline may wait and go in one call for the cost of one kernel submission. The question is
//! the driver's, and the fact answers it, never the driver's name: a driver without the fact
//! takes the ungrouped path, each batch its own `vkQueueSubmit`, which is what every batch took
//! before grouping and is exact alike -- a group changes when a `vkQueueSubmit` is made and
//! nothing it submits.
//!
//! - **What waits.** A batch the scheduler submits waits here, submitted by nothing yet, where it
//!   signals nothing but its queue's timeline and its events' semaphores: no binary semaphore,
//!   no fence; binds no sparse update; does no first work; moves none of the caller's wrapped
//!   images; and waits for no event index a batch held signals (only the group's end would meet
//!   it: on one `VkQueue` that wait would never be met). Every other batch goes as before, after
//!   the group held on its `VkQueue` went (`Submitting::AfterGroup`), and so does one of another
//!   `VkQueue`, so a wait there for a held signal follows the group's submission too. A batch
//!   that waits for anything not seen complete (an event, another `VkQueue`'s work, first work)
//!   leads a group and never joins one: it goes after a group held, in a submission of its own.
//!   A group goes at once when it holds [`GROUP_BATCHES`], so no submission holds more.
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
//!   it is held, its submission handlers run before that on the scheduler's thread once its lock is
//!   free, as a submitted batch's do ([`Group::schedule_held`]), as Metal schedules a command
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
//!   (at a gate, or for a signal to land, which may be the group's), or a thread waits for one of
//!   its command buffers' completion (`CommandBuffer::wait_until_completed` marks it and wakes the
//!   scheduler: [`CommandBufferShared::waits_for_completion`]). A group whose batch ends a command
//!   buffer with a switch of storage forms riding on it is never kept: the switch settles by the
//!   submission's outcome, before the command buffer is `Scheduled`. A kept batch is seen
//!   complete late by its group's later batches' GPU time, as any member, and where the one
//!   submission left in flight ends before the group reaches the GPU, by that gap besides: at most
//!   a completion turn's reaction and the group's `vkQueueSubmit`. Counted:
//!   `WorkCounts::groups_kept_pending`, `WorkCounts::groups_sent_for_waits`.
//! - **Counted by its end.** Each group held counts once by what sent it ([`GroupEnd`],
//!   `WorkCounts::group_ends_full` and kin): so the consumer's counts say why its groups hold
//!   fewer batches than they may.
//! - **No allocation.** A group's submission names at most [`MOST_INFOS`] `VkSubmitInfo`s, built
//!   in arrays on the thread's stack; its end's signals and its submissions go to lists the group
//!   keeps with their room: the scheduler thread allocates nothing for it
//!   (`runtime_lane::frame_costs`, `a_frame_s_commits_allocate_within_their_bound`).

use super::submit::Submission;
use super::{submission_error, CommandBufferShared};
use crate::degradation::Degradation;
use crate::device::relay::Relayed;
use crate::error::CommandBufferError;
use crate::exit_guard::{self, Call};
use crate::gpu_time;
use crate::Device;
use ash::vk;
use metal2vulkan::device::{CompletionPolicy, DriverFact};
use std::collections::BTreeSet;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// The most batches one kernel submission holds: the first is seen complete late by the GPU time
/// of at most this many less one.
pub(crate) const GROUP_BATCHES: usize = 4;

/// The most `VkSubmitInfo`s a batch names: its work's and its finale's (`submit.rs`).
const BATCH_INFOS: usize = 2;

/// The most `VkSubmitInfo`s one group's `vkQueueSubmit` names: [`BATCH_INFOS`] for each batch
/// held, and for the batch after them its first work's beside its own.
const MOST_INFOS: usize = GROUP_BATCHES * BATCH_INFOS + 1 + BATCH_INFOS;

/// Whether `device`'s batches on `VkQueue` `queue` may wait in a group: its policy groups them,
/// its driver is measured to merge a submission that signals nothing into the next (the module
/// docs), and the queue's failures are not relayed (each relayed batch takes a fence,
/// `device/relay.rs`).
pub(crate) fn groups(device: &Device, queue: usize) -> bool {
    device.compile_device().policy().completion == CompletionPolicy::Grouped
        && merges_submissions(&device.properties().driver_facts)
        && device.shared().flights(queue).is_none()
}

impl CommandBufferShared {
    /// Whether its batches may wait in a group ([`groups`]).
    pub(crate) fn groups(&self) -> bool {
        groups(&self.device, self.queue.vk_queue())
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
    pub(crate) fn waits_for_completion(&self) {
        let before = self.group_hold.fetch_or(WAITED_FOR, Ordering::SeqCst);
        if before & IN_GROUP != 0 {
            self.device.shared().scheduler.wake();
        }
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

/// Whether the driver merges a `VkSubmitInfo` that signals nothing into the next one's kernel
/// submission ([`DriverFact::SubmissionsWithoutSignalsMerge`]): the driver's facts alone answer
/// it, and nothing of the driver's identity, so a test can hand it the facts of each committed
/// profile (`tests::batches::only_drivers_measured_to_merge_their_submissions_group_their_batches`).
pub(crate) fn merges_submissions(facts: &BTreeSet<DriverFact>) -> bool {
    facts.contains(&DriverFact::SubmissionsWithoutSignalsMerge)
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
    /// The pass found no batch left to submit.
    Drained,
    /// The pass found none ready, with submission checks or host steps left for the thread to run.
    Checks,
    /// The pass found none ready, the rest held for events, gates, binary semaphores or decodes.
    Held,
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

/// A batch held: its `VkSubmitInfo`s, how many command buffers they hold, and its submission,
/// published once the group is submitted.
struct Member {
    infos: Vec<Info>,
    buffers: usize,
    submission: Submission,
}

/// The batches held for one kernel submission on one `VkQueue` (the module docs), in the
/// scheduler's state.
#[derive(Default)]
pub(crate) struct Group {
    queue: usize,
    members: Vec<Member>,
    /// It stayed held past a pass for the work its queue ran ([`keeps`](Group::keeps)).
    kept: bool,
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
    /// group's is made. It signals each event's semaphore it names to its index, which the
    /// group's end signals instead.
    pub(crate) fn hold(&mut self, infos: &[InfoParts<'_>], buffers: usize, submission: Submission) {
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
    /// device is not lost.
    /// The value of that queue's timeline whose completion leaves one submission in flight, for
    /// the completion turns to wake the scheduler at. Counted once per group, and once more where
    /// a waiter sends a group kept before.
    pub(crate) fn keeps(&mut self) -> Option<u64> {
        let first = self.members.first()?;
        let device = first.submission.buffer().device();
        if self.full()
            || device.is_lost()
            || (self.members.iter()).any(|member| member.submission.rides_forms())
        {
            return None;
        }
        let counts = &device.shared().counts;
        if (self.members.iter()).any(|member| member.submission.buffer().is_waited_for()) {
            if self.kept {
                counts.group_sent_for_wait();
            }
            return None;
        }
        let value = device.shared().queues[self.queue].pending_beyond_one()?;
        if !self.kept {
            self.kept = true;
            counts.group_kept_pending();
        }
        Some(value)
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
    /// the call's own result is `after`'s, as the `vkQueueSubmit` returned it.
    pub(crate) fn submit(
        &mut self,
        device: &Device,
        waited: &mut [u64],
        after: Option<After<'_>>,
    ) -> Result<(), vk::Result> {
        if self.members.is_empty() {
            return match after {
                None => Ok(()),
                Some(after) => submit_alone(device, after),
            };
        }
        let shared = device.shared();
        let raw = shared.vk();
        let queue = self.queue;
        let count = self.members.len();
        let alone = after.is_none();
        self.kept = false;
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
        let submitted =
            shared.submit_logged((queue, Some(value)), fence, &relayed, |vk_queue| unsafe {
                shared.counts.queue_submitted(submits.len(), buffers);
                raw.queue_submit(vk_queue, submits, fence.unwrap_or(vk::Fence::null()))
            });
        exit_guard::called(Call::QueueSubmit);
        self.events.clear();
        let several = count + usize::from(!alone) > 1;
        let outcome = match submitted {
            Ok(()) => {
                let now = gpu_time::host_nanos();
                for member in self.members.drain(..) {
                    self.spare.push(member.infos);
                    if member.submission.schedules() && !member.submission.scheduled() {
                        member.submission.buffer().kernel.end_at(now);
                    }
                    self.made.push(member.submission);
                }
                if several {
                    shared.counts.batches_grouped(count + usize::from(!alone));
                }
                Ok(())
            }
            Err(error) => {
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
        submitted
    }
}

/// Submit `after` alone on its `VkQueue` of `device`: the group's submission with no batch held.
fn submit_alone(device: &Device, after: After<'_>) -> Result<(), vk::Result> {
    let shared = device.shared();
    let raw = shared.vk();
    let After {
        queue,
        infos,
        buffers,
        value,
        fence,
        relayed,
    } = after;
    // SAFETY: the batch's submission's own (`submit.rs`).
    let submitted = shared.submit_logged((queue, Some(value)), fence, relayed, |vk_queue| unsafe {
        shared.counts.queue_submitted(infos.len(), buffers);
        raw.queue_submit(vk_queue, infos, fence.unwrap_or(vk::Fence::null()))
    });
    exit_guard::called(Call::QueueSubmit);
    submitted
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
    let submitted = shared.submit_logged((queue, value), None, relayed, |vk_queue| unsafe {
        shared.counts.queue_submitted(1, 0);
        raw.queue_submit(vk_queue, &[info], vk::Fence::null())
    });
    exit_guard::called(Call::QueueSubmit);
    submitted
}
