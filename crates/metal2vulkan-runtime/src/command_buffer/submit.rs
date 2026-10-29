//! Submitting one batch of a command buffer, on the scheduler's thread, in its order, under its
//! lock.
//!
//! One `vkQueueSubmit` of two `VkSubmitInfo`s on the queue's `VkQueue`:
//! 1. The work: the batch's waits, Vulkan waits for its events' signal indices, for the caller's
//!    binary semaphores and the tracker's waits for other `VkQueue`s' timelines, then its encoders'
//!    `VkCommandBuffer`s (an encoder's segments among its own), each encoder after the barrier the
//!    tracker planned for it. What completed work wrote is made visible by a wait or a barrier,
//!    see below.
//! 2. The end: the barrier that makes every write visible to the host ([`Barrier::to_host`]), or,
//!    where batches are ordered by their ends (`batch_order.rs`), to the host and to every later
//!    command of the `VkQueue` ([`Barrier::to_all`]; so a later batch there plans no barrier from
//!    this one's accesses, `tracker.rs`), then the signals: the events' kept signal indices, the
//!    caller's binary semaphores and the queue's timeline value. When an event's previous kept
//!    signal is pending on another `VkQueue`, this part waits for it, so the event's semaphore is
//!    signaled in index order without holding back the work.
//!
//! **Grouped** ([`Submitting::Grouped`], `group.rs`). Where the scheduler groups batches (under
//! `CompletionPolicy::Grouped`, on a driver that merges submissions), a batch that signals nothing
//! but its queue's timeline and its events, past its first work, without a fence or a wrapped
//! image to move, and waits for no event index a batch held signals, is not submitted: its two
//! `VkSubmitInfo`s are held, its signals left to the group's end, and it is tracked as submitted
//! (its events keep their signals' indices). One that may join but took first work or names a
//! wrapped image goes with the held group, in its call, last; one that may not, or finds the
//! group of another `VkQueue`, changes nothing ([`Submitting::AfterGroup`]): the scheduler
//! submits the group and offers the batch again.
//!
//! **What completed work wrote.** The tracker forgets a batch once the host saw its queue's
//! timeline pass it, and the device's own submissions outside the batches (`device/submit.rs`,
//! `device/staging.rs`) it never tracks: their writes are ordered before a later batch by their
//! completion alone. The work waits for each such timeline, each `VkQueue`'s and the device's two
//! own, at the value the host saw it reach, where a batch of its `VkQueue` did not wait for that
//! value already: a wait that is satisfied as it is submitted. The signal made the writes
//! available ("The first access scope includes all memory access performed by the device",
//! semaphore signaling), and the wait makes them visible: the submission's "second
//! synchronization scope of each semaphore wait operation defined by this command includes every
//! command in the same batch that the wait operation is defined in, and all commands that occur
//! later in submission order", and the wait's "second access scope includes all memory access
//! performed by the device" (`vkQueueSubmit`, semaphore waiting). So one wait per value serves
//! every later batch of its `VkQueue`. A batch with nothing planned before its first encoder then
//! needs no command buffer before it, a timed command buffer's first batch included where its
//! first encoder wrote the start (**GPU times**, below): on an RTX 5080 and a Radeon 780M, nearly
//! every batch of a frame of draws, and of command buffers committed from several threads. A wait costs CPU in `vkQueueSubmit`: about
//! 1.1 us on ms01's Iris Xe (ANV), 1.6 us on a Radeon 780M (RADV) and 0.5 us on an RTX 5080,
//! where a command buffer of barriers cost about 12 us of the Iris Xe's GPU. Counted:
//! `WorkCounts::completed_waits`, `WorkCounts::bare_starts` (the batches with work that began
//! with no command buffer of barriers).
//!
//! A driver measured to stall its submissions on semaphore operations
//! (`DriverFact::SemaphoreOperationsStallSubmits`: NVIDIA's 615.71.09 on the RTX 5080 Laptop GPU
//! and the RTX 3090, about 2 ms of CPU in one `vkQueueSubmit` in about every thousand waits and
//! signals) takes no such wait: each batch with work starts instead with a command buffer of
//! barriers whose first is [`Barrier::after_completed`], which makes what completed batches
//! wrote visible to every later command, and the staging queue's copies signal no timeline.
//! Where every batch waited, about twice as many of its submissions took that long as where
//! batches only signal.
//!
//! Each barrier goes in a command buffer of barriers alone, which the scheduler records once per
//! list of barriers and submits again once the batch that last carried it completed
//! ([`kept_barriers`](super::kept_barriers)); one that also writes a command buffer's timestamps
//! once per list and slot, submitted again only by a later holding of the slot; one that moves
//! the caller's images is recorded by its batch.
//!
//! **The end in the work's last command buffer.** An encoder's last `VkCommandBuffer` stays
//! recording after the encoder ended, as its command buffer's tail, until an encoder after it
//! joins its batch, its batch ends at encode time, or the command buffer commits
//! (`CommandBufferShared::close_tail`). Where its batch ended there, the end goes into it before
//! it ends: the barrier to the host, then, in the last batch of a timed command buffer, the end
//! timestamp, as the end's own command buffer holds them. Only where the batch leaves its
//! submission no segment or sparse update (recorded at the submission, among or before the
//! work's command buffers), and names no texture over the caller's image that moves between
//! layouts (its move back is planned at the submission, before the end). The batch's submission
//! then has no end of its own, unless an event's kept signal waits for another `VkQueue`'s
//! (above): its end then holds the barrier to the host alone, between that wait and the signals,
//! the work's end timestamp written. Else the signals join the work's `VkSubmitInfo`: "The first
//! synchronization scope of each semaphore signal operation defined by this command includes
//! every command in the same batch that the signal operation is defined in, and all commands that
//! occur earlier in submission order" (`vkQueueSubmit`). Where that command buffer does not go
//! to the queue as recorded (its encoder runs nothing, or a rebind encoded it again), the end is
//! a command buffer of its own, as every other batch's. Each command buffer a submission holds
//! costs GPU time of its own: a probe of one 1x1 copy took 42.6 us of GPU time as three primary
//! command buffers (a start, the copy, an end) and 25.6 us as one on ms01's Iris Xe (ANV), 19.6
//! and 13.6 cold on a Radeon 780M (RADV), 15.4 to 17.4 and 7.9 to 9.0 on an RTX 5080. Counted:
//! `WorkCounts::finales_folded`, `WorkCounts::finales_apart`.
//!
//! A batch with no work is one `VkSubmitInfo` of every wait, the end and the signals: a wait
//! orders only its own `VkSubmitInfo`'s commands, and a signal follows commands, not waits
//! (`vkQueueSubmit`'s synchronization scopes), so a signal after a wait needs both in one, with
//! the end's barrier, a command, between. Where it has no wait, no image to move back and no
//! timestamp to write either (an empty command buffer's batch, one of signals alone), its end
//! records nothing and the `VkSubmitInfo` holds no `VkCommandBuffer`: none of its commands wrote
//! anything, every batch before it made its own writes visible to the host before its signals,
//! and its signals follow every command submitted before them. ANV on i915 lists every
//! `VkDeviceMemory` of the device, and its own buffer objects, in each kernel submission of
//! commands, and two objects in one of none (Mesa 26.2.4, `i915/anv_batch_chain.c`).
//!
//! **The caller's images** ([`Device::new_texture_with_vk_image`]). A batch whose encoders use a
//! wrapped image that is not at rest in `GENERAL` moves it there before its first encoder and back
//! to its home layout in the end, before the signals. The moves are planned with the encoders, as
//! writes of the whole image outside them
//! ([`WrappedImage::transition_use`](crate::texture::wrapped::WrappedImage::transition_use)),
//! so the tracker orders them against every other use of the image on any queue, and each is
//! recorded as an image memory barrier from all commands before to all commands after.
//!
//! **Held depths** (`texture/held_depth.rs`). Before each encoder that names a held multisampled
//! depth, the encoder's command buffer of barriers moves it into the layout the encoder needs
//! where the batches submitted before, or the batch's earlier encoders, left it in another. The
//! moves are planned with the encoders as the caller's images' are, each a write of the whole
//! image outside the encoder joined to its summary, and recorded the same way, after the barrier
//! the tracker planned for the encoder. The layouts the batch leaves count from its submission on.
//!
//! **New resources' first work** (`texture/first_move.rs`). Every batch first does the first GPU
//! work of the resources made since the device's last submission: each new image's move to
//! `GENERAL`, and the zeros a new texture, stand-in or buffer reads. It runs before every command
//! of the batch, whether a use names the resource or not (a command may reach a texture through its
//! resource ID, a buffer through its address), and the batch holds what it worked on until it
//! completes. On a device of several `VkQueue`s every batch on another one waits for it until it
//! completed. It is recorded first in the command buffer that leads the batch's work's
//! `VkSubmitInfo` (in a batch with no work, its end's) -- its own, or, where the batch also has
//! host calls' staged copies, the one that holds the copies after it -- so it waits for what the
//! work waits for: Vulkan's
//! submission order puts it before every command of the batch ("the order in which command
//! buffers are specified in the pCommandBuffers member of VkSubmitInfo ... determines their
//! submission order", the Vulkan specification, Submission Order), and its last barrier,
//! recorded outside a render pass, orders it before them all: "the first synchronization scope
//! includes all commands that occur earlier in submission order ... the second synchronization
//! scope includes all commands that occur later in submission order" (`vkCmdPipelineBarrier`).
//! MoltenVK 1.4.1 runs each `VkSubmitInfo` as a Metal command buffer of its own
//! (`MVKQueue::submit`), each about 65 us of a thread of Metal's on an M3 Max, so the first work
//! adds none (`WorkCounts::submit_infos`). It takes no barrier that names a host read of an image
//! (`image_barrier.rs`), the one MoltenVK took for a read back over a heap texture's own storage
//! at its Metal command buffer's completion. A failed `vkQueueSubmit` made none of it, or lost
//! the device.
//!
//! [`Device::new_texture_with_vk_image`]: crate::Device::new_texture_with_vk_image
//!
//! The submission of the batch that ends a command buffer also signals a fence of the device's
//! where its driver reads the command buffer's objects after its value
//! ([`fences`](super::fences)), at which the command buffer then finishes.
//!
//! **GPU times** (`gpu_time.rs`). In a timed command buffer, the start timestamp is written at the
//! top of the pipe, before every barrier and encoder of its first batch: by its first encoder's
//! first command buffer, which wrote it into its slot's third query as it opened, where the batch
//! plans nothing before that encoder (no barrier, no image moved, and no
//! [`Barrier::after_completed`] where the driver stalls on semaphore operations) and the command
//! buffer goes to the queue as recorded first (`WorkCounts::prefixed_starts`); else by the batch's
//! first command buffer, a command buffer of barriers, which resets the slot's first two queries
//! first. Where the batch plans a barrier before its first encoder for earlier work (the encoder's
//! own, or the one before the images it moves), the start is written after that barrier instead, at
//! the bottom of the pipe: once every command submitted before it on the queue completed. Vulkan
//! (`vkCmdWriteTimestamp`): its "first synchronization scope includes all commands that occur
//! earlier in submission order", and `VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT` "is equivalent to
//! `VK_PIPELINE_STAGE_ALL_COMMANDS_BIT` with `VkAccessFlags` set to `0` when specified in the first
//! synchronization scope". So a command buffer whose first encoder depends on work still running
//! starts after that work ended, as Metal's does: on an M3 Max under macOS 27.0.1, a fill committed
//! behind a running fill of the same bytes started 0.5 to 0.9 us after the first's `GPUEndTime` (10
//! of 10), and one of other bytes while the first ran (probe `gpustart`). Where batches are ordered
//! by their ends (`batch_order.rs`), every start is written at the bottom of the pipe, as such a
//! driver starts no batch before the ones before it ended. The end of its last batch writes the end
//! timestamp at the bottom of the pipe, after its barriers. The scheduler's host times as it begins
//! the first batch and ends the last submission are the kernel times.
//!
//! The tracker keeps the plan, and the events their kept signals, only once the submission is
//! made; a batch that is not submitted changes neither.
//!
//! The submission is a section of the exit guard ([`exit_guard`]). What it did is published
//! ([`CommandBufferShared::publish`]: the transition to `Scheduled`, which runs the scheduled
//! handlers and ends waits for scheduling, and the queue waiter's watch) only after that
//! section ended. A section ends by registering the exit hook again when its calls may have
//! registered destructors, so an exit the program makes once it learned of the submission runs
//! the hook before them.
//!
//! The submission of a command buffer's last batch, or its hold in a group (`group.rs`), leaves its
//! submission handlers to the submitting thread, once the scheduler's lock is free; its transition
//! to `Scheduled` waits for them.

use super::completion::Submitted;
use super::gate::Hold;
use super::group::{self, After, GroupEnd, InfoParts};
use super::hang;
use super::kept_barriers::{KeptBarriers, SlotStamps};
use super::{lock, recording_error, submission_error, CommandBufferShared, State};
use crate::deferred::Deferred;
use crate::encoder::RecordedSegment;
use crate::error::CommandBufferError;
use crate::error::{internal, Error};
use crate::event_timeline::Signal;
use crate::exit_guard::{self, Call};
use crate::gpu_time;
use crate::release;
use crate::resource::ResourceKey;
use crate::retained::Retained;
use crate::scheduler;
use crate::sparse::bind::{bind_batch, BindFailure, Bound};
use crate::sparse::encoder::SparseUpdate;
use crate::texture::decoded_twin::{SecondDecode, TwinWrite};
use crate::texture::wrapped::Transition;
use crate::texture::Texture;
use crate::tracker::{Barrier, EncoderUses, Point, Summary};
use crate::work_counts::Begun;
use ash::vk;
use metal2vulkan::bind::ReplayFaults;
use std::collections::BTreeSet;
use std::sync::Arc;

/// The lists a batch's submission fills and empties, kept in the scheduler's state from batch to
/// batch with their room, so that a submission allocates none of them once they have room for
/// what it puts in them ([`CommandBufferShared::submit_batch`]).
#[derive(Default)]
pub(crate) struct BatchLists {
    /// What each timeline whose writes the tracker forgets completed through as the host saw
    /// it, each `VkQueue`'s, then the device's own: none where it is at 0.
    reached: Vec<Option<(vk::Semaphore, u64)>>,
    /// The work's waits.
    waits: BatchWaits,
    /// Of `reached`, the timelines (by place) and values the work waits for.
    waited: Vec<(usize, u64)>,
}

/// A `VkSubmitInfo`'s waits: the semaphores, their values and their stages.
#[derive(Default)]
struct BatchWaits {
    semaphores: Vec<vk::Semaphore>,
    values: Vec<u64>,
    stages: Vec<vk::PipelineStageFlags>,
}

/// A batch submitted, which nothing has seen yet: for the scheduler to
/// [`publish`](CommandBufferShared::publish).
#[must_use = "a batch not published never finishes its command buffer"]
pub(crate) struct Submission {
    /// Its `VkQueue`.
    queue: usize,
    batch: Submitted,
    /// It ends a command buffer with submission handlers, or with a switch of storage forms riding
    /// on it, which the submitting thread still runs or ends once the scheduler's lock is free
    /// ([`Submission::runs_handlers`]); false once a hold in a group gave them to the thread
    /// ([`schedule_held`](Submission::schedule_held)).
    handlers: bool,
    /// It ends a command buffer with a switch of storage forms riding on it, which settles by its
    /// submission's outcome ([`Submission::rides_forms`]).
    forms: bool,
    /// It makes its command buffer `Scheduled`: it ends it, and no hang's thread makes or made it
    /// (`hang::schedules`).
    schedules: bool,
    /// Its command buffer became `Scheduled` as it was held in a group
    /// ([`schedule_held`](Submission::schedule_held)).
    scheduled: bool,
}

/// What [`CommandBufferShared::submit_batch`] did with a batch.
#[must_use = "a batch submitted or held must be published, one held back must be put back"]
pub(crate) enum Submitting {
    /// Submitted, alone or after the group held before it: for the scheduler to publish.
    Submitted(Submission),
    /// Held in the scheduler's group (`group.rs`), its submission's changes made: its command
    /// buffer's transition to `Scheduled` as it is held, where it ends it
    /// ([`Submission::schedule_held`]), the rest published once the group is submitted.
    Grouped,
    /// Nothing submitted or changed: the submission of its event group waits for encoders of it a
    /// rebind encodes again (`reencode.rs`).
    Later,
    /// Nothing submitted or changed: the group held goes first, sent by what this says.
    AfterGroup(GroupEnd),
}

impl Submission {
    /// The value its queue's timeline reaches once it completed.
    pub(super) fn value(&self) -> u64 {
        self.batch.value
    }

    /// Whether it makes its command buffer `Scheduled` ([`Submission::schedules`]).
    pub(super) fn schedules(&self) -> bool {
        self.schedules
    }

    /// Its command buffer.
    pub(super) fn buffer(&self) -> &Hold<CommandBufferShared> {
        &self.batch.buffer
    }

    /// Each event's semaphore it signals, with the signal's index.
    pub(super) fn event_signals(&self) -> impl Iterator<Item = (vk::Semaphore, u64)> + '_ {
        (self.batch.signals.iter()).map(|(event, index)| (event.semaphore(), *index))
    }

    /// Whether the submitting thread runs its command buffer's submission handlers
    /// ([`CommandBufferShared::run_submission_handlers`]) once it holds no lock, after it ended
    /// the switch of storage forms riding on it: its transition to `Scheduled` waits for them.
    pub(crate) fn runs_handlers(&self) -> bool {
        self.handlers
    }

    /// Whether it ends a command buffer on which a switch of storage forms rides
    /// (`texture/storage_form.rs`): the switch settles by its submission's outcome, so it is
    /// `Scheduled` and its submission handlers run only once the group holding it was submitted.
    pub(super) fn rides_forms(&self) -> bool {
        self.forms
    }

    /// It was held in a group (`group.rs`), on the scheduler's thread under its lock, after the
    /// section that held it ended: where it ends its command buffer and no switch of storage
    /// forms rides on it, the command buffer's kernel end and its transition to `Scheduled` now,
    /// which [`publish`](CommandBufferShared::publish) then leaves out, and its submission
    /// handlers go to the scheduler's thread to run once its lock is free, before that
    /// transition, as a submitted batch's do: whether they do (none is left for the group's
    /// submission). Its tracker plan, its queue's value and its events' signal indices were
    /// taken as it was held, so nothing committed later on its queue goes before it; only its
    /// kernel submission and its queue waiter's watch wait for the group.
    pub(super) fn schedule_held(&mut self) -> bool {
        if !self.schedules || self.forms || self.scheduled {
            return false;
        }
        self.scheduled = true;
        self.batch.buffer.kernel.end_at(gpu_time::host_nanos());
        self.batch.buffer.schedule();
        let handlers = self.handlers;
        self.handlers = false;
        handlers
    }

    /// Whether its command buffer became `Scheduled` as it was held.
    pub(super) fn scheduled(&self) -> bool {
        self.scheduled
    }
}

impl CommandBufferShared {
    /// Submit batch `index` of the command buffer (`last` when it ends it) after the Vulkan waits
    /// `event_waits` for its events, against the scheduler's tracker and timeline values. The
    /// batch is for the caller to [`publish`](Self::publish) once its section ended
    /// ([`Submitting::Submitted`]), or waits in the scheduler's group for one kernel submission
    /// with the batches after it, its submission's changes made ([`Submitting::Grouped`],
    /// `group.rs`). Nothing submitted or changed while the submission of its event group waits for
    /// encoders of it a rebind encodes again (`reencode.rs`, [`Submitting::Later`]), or while the
    /// group held must go first ([`Submitting::AfterGroup`]): the caller holds the batch back at its
    /// place. Where the group went in this batch's submission, its outcome is the group's
    /// (`Group::take_outcome`).
    pub(crate) fn submit_batch(
        self: &Arc<Self>,
        index: usize,
        last: bool,
        event_waits: &[(vk::Semaphore, u64)],
        scheduler: &mut scheduler::State,
    ) -> Result<Submitting, CommandBufferError> {
        let device = &self.device;
        if device.is_lost() {
            return Err(CommandBufferError::DeviceLost);
        }
        let shared = device.shared();
        let raw = shared.vk();
        let queue = self.queue.vk_queue();
        let mut state = lock(&self.state);
        // Whether the batch may wait in the scheduler's group for one kernel submission with the
        // batches after it (`group.rs`), as far as is known before anything changes; first work
        // it takes still sends it now. One that may not, or of another `VkQueue`, waits,
        // nothing changed, for the group to go first. Its event signals ride to the group's end;
        // a wait for an event a batch held signals, which only that end meets, goes after it.
        let met = (event_waits.iter()).any(|&wait| scheduler.group.meets(wait));
        let joins = {
            let sync = &state.batches.all()[index];
            group::groups(device, queue)
                && !(last && super::fences::ends_at_fence(device, queue))
                && !met
                && sync.binary_waits.is_empty()
                && sync.binary_signals.is_empty()
                && (state.recorded[state.batch_range(index)].iter())
                    .all(|recorded| recorded.sparse.is_empty())
        };
        if !scheduler.group.admits(queue, joins) {
            let end = match met {
                true => GroupEnd::Wait,
                false => GroupEnd::Apart,
            };
            return Ok(Submitting::AfterGroup(end));
        }
        // A host step that failed fails the command buffer here; else the batch's segments go in
        // their places.
        if let Some(error) = state.step_failure.take() {
            return Err(error);
        }
        // Its first submission resolves what its commands reach as Metal's does, before any of
        // its batches goes to the `VkQueue` (`first_submission.rs`).
        if !state.submitted {
            let first = state
                .first_submission(device, self)
                .map_err(|error| recording_error(device, error))?;
            if !first {
                return Ok(Submitting::Later);
            }
        }
        // A later event group's, as its first batch is submitted, unless a rebind encodes
        // encoders of it again.
        if !state.later_group(device, self, index) {
            return Ok(Submitting::Later);
        }
        // Each of Metal's submissions it found rebound, or holding a detached buffer, fails as
        // its own batch is submitted, after the batch's checks: one a check failed keeps the
        // check's code.
        let rebound = state.failing_at(device, index);
        let range = state.batch_range(index);
        // A batch that begins with resource state encoders binds their updates first, after its
        // waits, which its work then waits for through the bind.
        let updates: Vec<SparseUpdate> = state.recorded[range.clone()]
            .iter()
            .flat_map(|recorded| recorded.sparse.iter().cloned())
            .collect();
        // Where the updates leave work on emulated textures or tile maps and bind nothing, the
        // values the work waits for in the bind's place: every queue's last.
        let mut after_all = None;
        let bound = if updates.is_empty() {
            None
        } else {
            let binary = &state.batches.all()[index].binary_waits;
            let waits: Vec<(vk::Semaphore, u64)> = event_waits
                .iter()
                .copied()
                .chain(binary.iter().map(|&semaphore| (semaphore, 0)))
                .collect();
            let Bound {
                unbound,
                signalled,
                mappings,
            } = bind_batch(device, queue, &updates, &waits, &scheduler.values)
                .map_err(|failure| bind_error(device, failure))?;
            state.retained.push(Box::new(unbound));
            if !mappings.is_empty() {
                // The work's first command buffer, in the first resource state encoder's place,
                // which the tracker sees write what it writes.
                let uses = mappings
                    .uses()
                    .map_err(|error| recording_error(device, error))?;
                let raw = self.record_raw(&mut state, |state, buffer| {
                    // SAFETY: the command buffer is recording; the textures are live (the
                    // command buffer holds them), in `GENERAL`; the work waits for the bind, or
                    // for every queue's last value.
                    let held = unsafe { mappings.record(device, &mut state.uploads, buffer) }?;
                    if !held.is_empty() {
                        state.retained.push(Box::new(held));
                    }
                    Ok(())
                })?;
                let first = &mut state.recorded[range.start];
                first.raws.push(raw);
                first.summary.join(Summary::outside_encoders(uses));
                if !signalled {
                    after_all = Some(scheduler.values.clone());
                }
            }
            signalled.then(|| {
                scheduler.values[queue] += 1;
                scheduler.values[queue]
            })
        };
        // The scrubs of emulated textures' writes, as mapped now, then every segment in its place.
        self.record_scrubs(&mut state, range.clone())?;
        state
            .place(range.clone())
            .map_err(|error| recording_error(device, error))?;
        // No work of an encoder that recorded nothing for a command (a buffer it named was
        // detached), nor of one whose submission failed rebound or detached, goes to the queue.
        state
            .runs_nothing_unrunnable(range.clone())
            .map_err(|error| recording_error(device, error))?;
        // The decodes into ASTC carriers' second decoded twins: those queued, before its work;
        // its blits' writes', after each part.
        self.record_second_decodes(&mut state, range.clone())?;
        let at = Point {
            queue,
            value: scheduler.values[queue] + 1,
        };
        // What each timeline whose writes the tracker forgets completed through as the host saw
        // it, each `VkQueue`'s, then the device's own (the module docs): read once, for the
        // tracker and for the work's waits alike.
        let BatchLists {
            reached,
            waits,
            waited,
        } = &mut scheduler.lists;
        reached.clear();
        for (slot, vk_queue) in shared.queues.iter().enumerate() {
            let value = vk_queue.completed_value();
            scheduler.tracker.completed(slot, value);
            reached.push((value > 0).then_some((vk_queue.timeline, value)));
        }
        reached.extend(device.own_completed());
        // The batches before it on its queue that the host has not seen complete: each took the
        // next value (`WorkCounts::batches_with_none_pending` and kin).
        let completed = shared.queues[queue].completed_value();
        (shared.counts).batch_submitted_behind(scheduler.values[queue].saturating_sub(completed));
        // One a hang's thread made `Scheduled` early began then (`hang.rs`).
        if index == 0 && self.kernel.start() == 0 {
            self.kernel.begin();
        }
        let slot = state
            .times
            .slot
            .as_ref()
            .map(|slot| (slot.queries(), slot.first(), slot.lease()));
        // The start timestamp goes into the first command buffer of the first batch, a barrier
        // one (with the barrier planned before its first encoder, if any), or into its end when
        // it has no work, unless its first encoder wrote it (below).
        let mut start_stamp = slot.filter(|_| index == 0);
        let batches = state.batches.all();
        let (start, end) = (range.start, range.end);
        let binary_waits = batches[index].binary_waits.clone();
        let binary_signals = batches[index].binary_signals.clone();
        let mut kept = Vec::new();
        let mut order_waits = Vec::new();
        for (event, value) in &batches[index].signals {
            if let Signal::Kept {
                index: signal,
                after,
            } = event.signal(*value, queue)
            {
                if let Some(after) = after {
                    order_waits.push((event.semaphore(), after));
                }
                kept.push((Arc::clone(event), signal, *value));
            }
        }
        let images = shared.interop.wrapped.among(
            state.recorded[start..end]
                .iter()
                .flat_map(|recorded| recorded.summary.keys()),
        );
        // The textures its encoders made for their own commands (`Device::new_planned_texture`),
        // which it holds until it completes.
        let own: Vec<Texture> = lock(&self.own_textures).drain(..).collect();
        if !own.is_empty() {
            state.retained.push(Box::new(own));
        }
        let encoders = &state.recorded[start..end];
        let mut entries = Vec::new();
        let mut exits = Vec::new();
        let (mut entry_uses, mut exit_uses) = (Vec::new(), Vec::new());
        for image in &images {
            let (entry, exit) = image.transitions();
            if let Some(entry) = entry {
                entries.push(entry);
                entry_uses.push(image.transition_use());
            }
            if let Some(exit) = exit {
                exits.push(exit);
                exit_uses.push(image.transition_use());
            }
        }
        // The new resources' first work, before every command (`texture/first_move.rs`), given
        // back if the submission is not made. A batch's commands may reach every texture
        // unnamed, so it takes every move, those of images bound to host pages among them, whose
        // pages it then holds. Its commands bind the heap sets, so it holds the texture slots'
        // writes until its submission was made (`heap_sets::hold`).
        let firsts = &shared.textures.first_moves;
        let first_work = firsts.take_holding(|_| true, || crate::heap_sets::hold(device));
        let entry = (!entries.is_empty()).then(|| Summary::outside_encoders(entry_uses));
        let exit = (!exits.is_empty()).then(|| Summary::outside_encoders(exit_uses));
        // The held depths' moves before each encoder that names one, into the layout it needs
        // (`texture/held_depth.rs`): writes the tracker plans with the encoder's uses, before
        // them, so each waits for what came before it and orders what follows.
        let unnamed = BTreeSet::new();
        let held_moves = shared.textures.held.moves(encoders.iter().map(|recorded| {
            match state.ended.get(recorded.encoder) {
                Some(record) => (&record.keys, record.held.as_slice()),
                None => (&unnamed, &[][..]),
            }
        }));
        // Each encoder's summary with its moves' uses joined, where it has moves.
        let moved: Vec<Option<Summary<ResourceKey>>> = match held_moves.is_empty() {
            true => Vec::new(),
            false => (encoders.iter().enumerate())
                .map(|(at, recorded)| {
                    let uses = held_moves.uses(at);
                    (!uses.is_empty()).then(|| {
                        let mut summary = recorded.summary.clone();
                        summary.join(Summary::outside_encoders(uses.to_vec()));
                        summary
                    })
                })
                .collect(),
        };
        let summaries = (encoders.iter().enumerate()).map(|(at, recorded)| {
            let moved = moved.get(at).and_then(Option::as_ref);
            moved.unwrap_or(&recorded.summary)
        });
        let plan = scheduler
            .tracker
            .plan(entry.iter().chain(summaries).chain(&exit), at);
        // One barrier per summary planned: the entry's, the encoders', the exit's.
        let mut barriers = plan.barriers();
        let mut entry_barrier = None;
        if let (Some(_), Some((&first, rest))) = (&entry, barriers.split_first()) {
            entry_barrier = first;
            barriers = rest;
        }
        let mut exit_barrier = None;
        if let (Some(_), Some((&last, rest))) = (&exit, barriers.split_last()) {
            exit_barrier = last;
            barriers = rest;
        }
        // Where the driver's submissions stall on semaphore operations, the first command buffer
        // of barriers makes what completed work wrote visible, in place of the work's waits (the
        // module docs).
        let by_barrier = shared.completed_by_barrier;
        let mut completed_barrier = by_barrier.then(Barrier::after_completed);
        // The first encoder's prefix wrote the start where the batch plans nothing before that
        // encoder (no barrier, no image moved, no barrier for completed work) and its first
        // command buffer goes to the queue as recorded (`gpu_time.rs`): then no command buffer of
        // barriers starts the batch for the timestamp's sake.
        let prefix = state.prefix;
        let prefixed = start_stamp.is_some()
            && completed_barrier.is_none()
            && entry.is_none()
            && barriers.first().is_some_and(Option::is_none)
            && held_moves.transitions(0).is_empty()
            && prefix.is_some_and(|raw| {
                state
                    .recorded
                    .get(start)
                    .and_then(|first| first.raws.first())
                    == Some(&raw)
            });
        if prefixed {
            start_stamp = None;
        }
        if index == 0 {
            state.times.prefixed = prefixed;
        }
        let mut work = Vec::with_capacity(2 * (end - start) + 1);
        // Whether a command buffer of barriers goes before the first encoder.
        let mut started = false;
        // A barrier planned before the first encoder, for earlier work that may still run, or a
        // held depth's move there, which waits for all of it: the start goes after the last such
        // barrier, once every command before it completed (the module docs).
        let encoder_behind =
            barriers.first().is_some_and(Option::is_some) || !held_moves.transitions(0).is_empty();
        if entry.is_some() {
            started = true;
            let globals: Vec<Barrier> = completed_barrier
                .take()
                .into_iter()
                .chain(entry_barrier)
                .collect();
            let stamps = Stamps {
                start: start_stamp.filter(|_| !encoder_behind),
                after: entry_barrier.is_some(),
                end: None,
            };
            if stamps.start.is_some() {
                start_stamp = None;
            }
            work.push(self.record_barriers(&mut state, &globals, &entries, stamps)?);
        }
        for (offset, barrier) in barriers.iter().enumerate() {
            let globals: Vec<Barrier> = completed_barrier
                .take()
                .into_iter()
                .chain(*barrier)
                .collect();
            let moves = held_moves.transitions(offset);
            if !globals.is_empty() || start_stamp.is_some() || !moves.is_empty() {
                started |= offset == 0;
                let stamps = Stamps {
                    start: start_stamp.take(),
                    after: barrier.is_some() || !moves.is_empty(),
                    end: None,
                };
                let kept = &mut scheduler.kept;
                work.push(self.barrier_buffer(&mut state, kept, at, &globals, moves, stamps)?);
            }
            work.extend(&state.recorded[start + offset].raws);
        }
        // The first work goes first in the batch's work, or in a batch with no work in its end
        // (the module docs). Host calls' staged copies go at the batch's start, after it, in the
        // same command buffer: the two went in two command buffers of one `VkSubmitInfo`, which
        // puts them in the same submission order one command buffer's recording order does ("the
        // order in which command buffers are specified in the pCommandBuffers member of
        // VkSubmitInfo ... determines their submission order", the Vulkan specification,
        // Submission Order; inside one command buffer, the order they were recorded), and a
        // command buffer boundary is no dependency of its own, so the copies' own leading barrier
        // (`FirstMoves::record_staged`) covers the moves recorded before it either way. One
        // command buffer where the batch has both (counted:
        // `WorkCounts::command_buffers_begun_batch_work`).
        let has_first = first_work.has_first();
        let has_staged = first_work.has_staged();
        let recorded = (has_first || has_staged)
            .then(|| {
                self.record_raw(&mut state, |_, buffer| {
                    if has_first {
                        // SAFETY: the command buffer is recording, outside a render pass; it is
                        // submitted first in the batch, before any later one, and every batch on
                        // another `VkQueue` waits for the batch (`FirstMoves::waits`).
                        unsafe {
                            let family = shared.queue_flags_of(queue);
                            first_work.record((raw, &shared.counts), buffer, family);
                        };
                    }
                    if has_staged {
                        // SAFETY: the command buffer is recording, outside a render pass; every
                        // image's move is in this command buffer, before it, or a submission
                        // before.
                        unsafe { first_work.record_staged(raw, buffer) };
                    }
                    Ok(())
                })
            })
            .transpose()?;
        // Where it holds staged copies it goes at the batch's start, as they did, so the batch
        // has work; the first work alone goes first in the work, or in the end where the batch
        // has none (below).
        let lead = match has_staged {
            true => {
                work.insert(0, recorded.expect("recorded where there are staged copies"));
                None
            }
            false => recorded,
        };
        let BatchWaits {
            semaphores: wait_semaphores,
            values: wait_values,
            stages: wait_stages,
        } = waits;
        wait_semaphores.clear();
        wait_values.clear();
        wait_stages.clear();
        if let Some(value) = bound {
            // The bind waited for the batch's waits; the work waits for the bind.
            wait_semaphores.push(shared.queues[queue].timeline);
            wait_values.push(value);
            wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
        } else {
            for &(semaphore, value) in event_waits {
                wait_semaphores.push(semaphore);
                wait_values.push(value);
                wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
            }
            for &semaphore in &binary_waits {
                // A binary semaphore's value is ignored.
                wait_semaphores.push(semaphore);
                wait_values.push(0);
                wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
            }
            for (slot, &value) in shared.queues.iter().zip(after_all.iter().flatten()) {
                if value > 0 {
                    wait_semaphores.push(slot.timeline);
                    wait_values.push(value);
                    wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
                }
            }
        }
        for wait in plan.waits() {
            wait_semaphores.push(shared.queues[wait.at.queue].timeline);
            wait_values.push(wait.at.value);
            wait_stages.push(wait.legacy_stages());
        }
        // Every batch on another `VkQueue` that did new resources' first work, until it
        // completed: any command may reach those resources unnamed.
        let completed = |point: Point| shared.queues[point.queue].completed_value() >= point.value;
        for point in firsts.waits(queue, completed) {
            wait_semaphores.push(shared.queues[point.queue].timeline);
            wait_values.push(point.value);
            wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
        }
        // And the internal submission on the first `VkQueue` that did some, until it completed.
        if let Some((timeline, value)) = firsts.internal_wait().filter(|_| queue != 0) {
            wait_semaphores.push(timeline);
            wait_values.push(value);
            wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
        }
        // And the last binding of a no-copy buffer's dma-bufs, on every `VkQueue`, the first
        // too (a binding is ordered with no later submission), until it completed.
        if let Some((timeline, value)) = device.binding_wait() {
            wait_semaphores.push(timeline);
            wait_values.push(value);
            wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
        }
        let mut order_semaphores: Vec<vk::Semaphore> =
            order_waits.iter().map(|&(s, _)| s).collect();
        let mut order_values: Vec<u64> = order_waits.iter().map(|&(_, value)| value).collect();
        let mut order_stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; order_waits.len()];
        // No work: every wait goes with the end, which it then orders.
        let single = work.is_empty();
        if single {
            order_semaphores.append(wait_semaphores);
            order_values.append(wait_values);
            order_stages.append(wait_stages);
        }
        // The work waits for what completed, where its `VkQueue` did not already and no barrier
        // made it visible (the module docs); the waits are taken as made once the submission is.
        waited.clear();
        if !single && !by_barrier {
            for (source, &timeline) in reached.iter().enumerate() {
                let Some((semaphore, value)) = timeline else {
                    continue;
                };
                if value > scheduler.waited[queue][source] {
                    wait_semaphores.push(semaphore);
                    wait_values.push(value);
                    wait_stages.push(vk::PipelineStageFlags::ALL_COMMANDS);
                    waited.push((source, value));
                }
            }
        }
        // The end went into the work's last command buffer as the batch ended (the module docs),
        // where that command buffer goes to the queue as recorded: then the batch has no end of
        // its own. One that does not go (its encoder runs nothing, or was encoded again) leaves
        // the end to its own command buffer, as every other batch's.
        let in_work = match state.fold(index).filter(|(raw, _)| work.contains(raw)) {
            None => false,
            Some((raw, stamped)) => {
                let ends = work.last() == Some(&raw)
                    && exit_barrier.is_none()
                    && exits.is_empty()
                    && start_stamp.is_none()
                    && stamped == (last && slot.is_some());
                debug_assert!(ends, "a batch's end recorded where the batch does not end");
                if !ends {
                    let error = internal("a batch's end recorded where the batch does not end");
                    return Err(recording_error(device, error));
                }
                true
            }
        };
        // An event's kept signal that waits for another `VkQueue`'s earlier one goes after that
        // wait, in an end of its own (the module docs): the end the work holds made its writes
        // visible and wrote its end timestamp, so that one records the barrier alone.
        let folded = in_work && order_semaphores.is_empty();
        // The end, where the batch has something to end. A batch with no work, no wait, no image
        // to move back and no timestamp records none: none of its commands wrote anything, every
        // batch before it on any `VkQueue` made its own writes visible to the host before its
        // signals, and its signals follow every command submitted before them. A wait needs it:
        // a wait orders the commands of its batch, and a signal follows those, not the wait.
        let stamps = Stamps {
            start: start_stamp.take(),
            after: false,
            end: slot.filter(|_| last && !in_work),
        };
        let ends = !single
            || !order_semaphores.is_empty()
            || exit_barrier.is_some()
            || !exits.is_empty()
            || stamps.start.is_some()
            || stamps.end.is_some();
        let mut finale: Vec<vk::CommandBuffer> = match ends && !folded {
            true => {
                let finale_barriers: Vec<Barrier> = exit_barrier
                    .into_iter()
                    .chain([crate::batch_order::end(shared.by_ends)])
                    .collect();
                let kept = &mut scheduler.kept;
                vec![self.barrier_buffer(&mut state, kept, at, &finale_barriers, &exits, stamps)?]
            }
            false => Vec::new(),
        };
        let signal_semaphores: Vec<vk::Semaphore> = kept
            .iter()
            .map(|(event, _, _)| event.semaphore())
            .chain(binary_signals.iter().copied())
            .chain([shared.queues[queue].timeline])
            .collect();
        let signal_values: Vec<u64> = kept
            .iter()
            .map(|&(_, signal, _)| signal)
            .chain(binary_signals.iter().map(|_| 0))
            .chain([at.value])
            .collect();
        // A batch whose end its work holds signals with its work.
        let (work_signals, work_signal_values) = match folded {
            true => (&signal_semaphores[..], &signal_values[..]),
            false => (&[][..], &[][..]),
        };
        // The first work, first in the batch's work or, where it has none, in its end: it waits
        // for what the batch's commands wait for (a heap's fill another queue did comes before a
        // move of an image placed in it; an event the program waits for before reusing a heap's
        // memory).
        if let Some(lead) = lead {
            match single {
                true => finale.insert(0, lead),
                false => work.insert(0, lead),
            }
        }
        let mut work_timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(wait_values)
            .signal_semaphore_values(work_signal_values);
        let first = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(&work)
            .signal_semaphores(work_signals)
            .push_next(&mut work_timeline);
        // The fence the last batch signals where the driver reads a command buffer's objects
        // after its value, and every batch where the queue's failures are relayed
        // (`device/relay.rs`): taken before the submission, given back if it fails. Elsewhere
        // the batch takes none (`fences::ends_at_fence`).
        let fence = if (last || shared.flights(queue).is_some())
            && super::fences::ends_at_fence(device, queue)
        {
            // SAFETY: the device's fences, inside the scheduler's section.
            Some(
                unsafe { shared.fences.take(raw) }
                    .map_err(|error| submission_error(device, error))?,
            )
        } else {
            None
        };
        let mut end_timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&order_values)
            .signal_semaphore_values(&signal_values);
        let second = vk::SubmitInfo::default()
            .wait_semaphores(&order_semaphores)
            .wait_dst_stage_mask(&order_stages)
            .command_buffers(&finale)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut end_timeline);
        let infos = [first, second];
        let infos = match (single, folded) {
            (true, _) => &infos[1..],
            (false, true) => &infos[..1],
            (false, false) => &infos[..],
        };
        let submits = infos;
        let buffers = (submits.iter())
            .map(|info| info.command_buffer_count as usize)
            .sum();
        // From here its heap set versions may be pending: nothing writes them any more, and its
        // first batch takes the ticket a texture that goes from now on waits for
        // (`heap_sets.rs`). One committed and not yet submitted holds no gone texture.
        if let Some(ticket) = state.heaps.submit(device) {
            state.retained.push(Box::new(ticket));
        }
        // The batch waits in the group, its submission's changes made below, where it may and
        // took no first work, the device wrapped no image of the caller's it names since it was
        // found ready, and it waits for nothing but values seen complete where a group is held: a
        // batch with a wait leads a group, never joins one, so the merge, which gathers every
        // batch's waits before the first one's commands, never holds back an earlier batch for
        // another's wait. Else it goes now, after the group where one is held.
        let waits = wait_semaphores.len() - waited.len() + order_semaphores.len();
        let leads = !scheduler.group.holds() || waits == 0;
        let grouped = joins
            && leads
            && !first_work.has_first()
            && !first_work.has_staged()
            && images.is_empty();
        if grouped {
            scheduler.group.open(queue, &scheduler.waited[queue]);
        } else {
            // SAFETY: every command buffer is executable and submitted once; every wait is for a
            // signal already submitted (the scheduler's order, and for a binary semaphore the
            // caller's contract of `encode_wait_for_vk_semaphore`); every signal of a timeline is
            // above its value and any earlier one (the queue's values rise, and an event's kept
            // indices rise and wait for the previous one on another queue), and every binary one
            // is of an unsignaled semaphore with no signal pending (`encode_signal_vk_semaphore`);
            // every transition is of a live image of the caller's in the layout it is in (its
            // contract, `new_texture_with_vk_image`); the fence is unsignaled and in no pending
            // submission; the `VkQueue` is used under its lock (`Group::submit`).
            if scheduler.group.holds() {
                let end = match leads {
                    true => GroupEnd::FirstWork,
                    false => GroupEnd::Wait,
                };
                shared.counts.group_ended(end);
            }
            let after = After {
                queue,
                infos: submits,
                buffers,
                value: at.value,
                fence,
                relayed: &self.relayed,
            };
            let submitted =
                scheduler
                    .group
                    .submit(device, &mut scheduler.waited[queue], Some(after));
            if let Err(error) = submitted {
                if let Some(fence) = fence {
                    // SAFETY: the failed submission is not pending; inside the section.
                    // Released, not kept: Vulkan leaves a fence's state after a failed
                    // submission unsaid.
                    unsafe { release::object(raw, fence) };
                }
                // A `vkQueueSubmit` that fails leaves what its submissions reference as it was,
                // or loses the device: the first work goes back, still to be done, as
                // `first_work` drops. Where the device is lost, its command buffers, of this
                // command buffer's pool that goes as the command buffer fails, go once the
                // driver's own completion of what the call ran did (`Device::drain_queues`).
                let error = submission_error(device, error);
                if matches!(error, CommandBufferError::DeviceLost) {
                    device.drain_queues();
                }
                return Err(error);
            }
        }
        // It makes its command buffer `Scheduled` where it ends it, unless a hang's thread makes
        // it (`hang.rs`): the command buffer's own hang's, once this batch, the one before the
        // hang's wait, is taken (the thread waits for it on `changed`), or, for one committed
        // during a hang, early, where it did not wait past the hang. A held batch's kernel end is
        // its group's (`Group::submit`).
        let (schedules, took) = hang::schedules(&mut state.hang_scheduling, index, last);
        if took {
            let parked = state.parked > 0;
            self.wake_parked(parked);
        }
        if schedules && !grouped {
            self.kernel.end_at(gpu_time::host_nanos());
        }
        if !single {
            match folded {
                true => shared.counts.finale_folded(),
                false => shared.counts.finale_apart(),
            }
        }
        for &(source, value) in waited.iter() {
            scheduler.waited[queue][source] = value;
        }
        shared.counts.completed_waited(waited.len());
        if !single && !started {
            shared.counts.bare_start();
        }
        if prefixed {
            shared.counts.prefixed_start();
        }
        // Its submission handlers run on this thread once the scheduler's lock is free, and a
        // switch of storage forms riding on it ends there first (`scheduler/mod.rs`); its
        // transition to `Scheduled`, posted as it is published or held, waits for them.
        let forms = last && state.storage_form.is_some();
        let handlers = (schedules && !state.submission.is_empty()) || forms;
        state.submitting = handlers;
        scheduler.tracker.apply(plan);
        held_moves.submitted();
        shared.counts.held_depths_released(held_moves.released);
        let several = shared.queues.len() > 1;
        let held = first_work.submitted(several.then_some(at));
        if !held.is_empty() {
            state.retained.push(Box::new(held));
        }
        scheduler.values[queue] = at.value;
        for image in &images {
            image.submitted();
            // One that rests in `GENERAL` from now on (an encode-time image after its first
            // batch) is moved by no later batch.
            if image.settled() {
                shared.interop.wrapped.forget(image);
            }
        }
        // Its signals are seen at its completion turn alone where it may write a copied host
        // buffer, whose bytes reach the pages there (`event_timeline.rs`).
        let behind_merge = !kept.is_empty() && shared.copied.any();
        if behind_merge {
            shared.counts.event_signals_behind_merge(kept.len());
        }
        for (event, signal, value) in &kept {
            event.keep(*signal, *value, queue, behind_merge);
        }
        // Its commands may reach every texture through its resource ID where one bound a heap
        // set: the stale-reach degrade's trigger decides (`detach.rs`).
        let reaches = state.heaps.bound();
        let rebound_use = state.rebound_use.clone();
        // The holds its completion lets go (`State::holds_until`).
        let holds = state.holds_until(index);
        drop(state);
        if rebound {
            device.degrade_encoded_use_rebound(rebound_use);
        }
        device.reach_stale_detached(reaches);
        let submission = Submission {
            queue,
            batch: Submitted {
                value: at.value,
                buffer: self.hold(),
                last,
                fence,
                signals: kept
                    .into_iter()
                    .map(|(event, signal, _)| (event, signal))
                    .collect(),
                holds,
            },
            handlers,
            forms,
            schedules,
            scheduled: false,
        };
        if !grouped {
            return Ok(Submitting::Submitted(submission));
        }
        let first = InfoParts {
            waits: (wait_semaphores, wait_values, wait_stages),
            buffers: &work,
            signals: (work_signals, work_signal_values),
        };
        let second = InfoParts {
            waits: (&order_semaphores, &order_values, &order_stages),
            buffers: &finale,
            signals: (&signal_semaphores, &signal_values),
        };
        let parts = [first, second];
        let parts = match (single, folded) {
            (true, _) => &parts[1..],
            (false, true) => &parts[..1],
            (false, false) => &parts[..],
        };
        scheduler.group.hold(parts, buffers, submission);
        Ok(Submitting::Grouped)
    }

    /// Publish a batch [`submit_batch`](Self::submit_batch) submitted, after the section that
    /// submitted it ended: when it ends the command buffer, its transition to `Scheduled`
    /// (`CommandBufferShared::schedule`) unless it made it as it was held in a group
    /// ([`Submission::schedule_held`]), then its queue waiter's watch for it. From the watch
    /// on, the command buffer may complete and run its release on its queue's handler thread
    /// before the caller lets go of anything it took, so the caller holds nothing of what its
    /// commands or encoders took (`scheduler::Spent`).
    pub(crate) fn publish(self: &Arc<Self>, submission: Submission) {
        let Submission {
            queue,
            batch,
            schedules,
            scheduled,
            ..
        } = submission;
        if schedules && !scheduled {
            // Before a completion turn can see it finish, so the queue's context runs its
            // transition to `Scheduled` first.
            self.schedule();
        }
        self.device.shared().completion.watch(queue, batch);
    }

    /// A command buffer of this command buffer's pool holding what `record` records into it,
    /// with the state at hand.
    fn record_raw(
        &self,
        state: &mut State,
        record: impl FnOnce(&mut State, vk::CommandBuffer) -> Result<(), Error>,
    ) -> Result<vk::CommandBuffer, CommandBufferError> {
        let device = &self.device;
        let raw = device.shared().vk();
        let buffer = self
            .allocate_raw(state)
            .map_err(|error| recording_error(device, error))?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer was just allocated from the pool, under the state lock, and
        // only this call records it.
        let begun = unsafe { raw.begin_command_buffer(buffer, &begin) };
        exit_guard::called(Call::BeginCommandBuffer);
        begun.map_err(|error| submission_error(device, error))?;
        device
            .shared()
            .counts
            .command_buffer_begun(Begun::BatchWork);
        record(state, buffer).map_err(|error| recording_error(device, error))?;
        // SAFETY: as above.
        let ended = unsafe { self.end_raw(buffer) };
        exit_guard::called(Call::EndCommandBuffer);
        ended.map_err(|error| submission_error(device, error))?;
        Ok(buffer)
    }

    /// Record the segments of encoders `range` that their batch's submission records: the
    /// scrubs (`Deferred::SparseScrub`), with the mappings it applied, and the gated copies in
    /// (`Deferred::GatedCopyIn`), with the regions their logs say changed since the commit's
    /// stamp, published by now.
    fn record_scrubs(
        &self,
        state: &mut State,
        range: std::ops::Range<usize>,
    ) -> Result<(), CommandBufferError> {
        let mut scrubs = Vec::new();
        for (encoder, recorded) in state.recorded[range.clone()].iter().enumerate() {
            for (at, segment) in recorded.segments.iter().enumerate() {
                let at_submission = matches!(
                    segment.work,
                    Deferred::SparseScrub(_) | Deferred::GatedCopyIn(_)
                );
                if at_submission && segment.recorded.is_none() {
                    scrubs.push((range.start + encoder, at, segment.work.clone()));
                }
            }
        }
        let device = &self.device;
        for (encoder, at, work) in scrubs {
            let raw = self.record_raw(state, |state, buffer| {
                match &work {
                    // SAFETY: the command buffer is recording; the texture is live (its encoder's
                    // command buffer holds it), in `GENERAL`, and its batch's updates applied.
                    Deferred::SparseScrub(scrub) => {
                        let held = unsafe { scrub.record(device, buffer) }?;
                        if !held.is_empty() {
                            state.retained.push(Box::new(held));
                        }
                    }
                    // SAFETY: the command buffer is recording, outside a render pass; the
                    // shadowed texture is live (its encoder's command buffer holds it), its
                    // image in `GENERAL`.
                    Deferred::GatedCopyIn(copy) => unsafe {
                        copy.record(device.shared().vk(), buffer)
                    },
                    _ => {}
                }
                Ok(())
            })?;
            state.recorded[encoder].segments[at].recorded = Some(RecordedSegment {
                raws: vec![raw],
                summary: EncoderUses::new(true).finish(),
                retained: Retained::default(),
                executions: Vec::new(),
                keys: Default::default(),
                taken: Default::default(),
                resident_misses: 0,
                faults: ReplayFaults::default(),
            });
        }
        Ok(())
    }

    /// Record the decodes into ASTC carriers' second decoded twins (`texture/decoded_twin.rs`) of
    /// the batch of encoders `range`, a batch with work: those queued (as each twin was made,
    /// and after host calls wrote its carrier), in one command buffer before the first part's
    /// own, their uses joined to its summary, so the barrier and waits planned before that part
    /// order them after the earlier writes of their images; but those of an image a part of the
    /// batch wrote without a decode (the image had no second twin as the write was encoded), which
    /// stay queued for a later batch: a twin made since reads the carrier as this batch leaves it.
    /// Then after each part's own command buffers, the decodes its writes made as they were
    /// encoded, their uses joined to its summary. Nothing, and no command buffer, where there is
    /// none (counted: `WorkCounts::second_twin_checks`, `WorkCounts::second_twin_decodes`).
    fn record_second_decodes(
        &self,
        state: &mut State,
        range: std::ops::Range<usize>,
    ) -> Result<(), CommandBufferError> {
        let device = &self.device;
        let counts = &device.shared().counts;
        if range.is_empty() {
            return Ok(());
        }
        let held: Vec<ResourceKey> = state.recorded[range.clone()]
            .iter()
            .flat_map(|part| &part.twin_writes)
            .filter_map(TwinWrite::without_decode)
            .collect();
        let queued = device.take_second_decodes(|key| held.contains(&key));
        if !queued.is_empty() {
            let raw = self.record_raw(state, |_, buffer| {
                for decode in &queued {
                    // SAFETY: the command buffer is recording, outside a render pass, on the
                    // device's queue family; the decode is retained until the command buffer
                    // completes.
                    unsafe { decode.record(device.shared().vk(), buffer) };
                }
                Ok(())
            })?;
            let uses = queued.iter().flat_map(|decode| decode.uses()).cloned();
            let first = &mut state.recorded[range.start];
            first.raws.insert(0, raw);
            first
                .summary
                .join(Summary::outside_encoders(uses.collect()));
            counts.second_twin_decodes(queued.len());
            state.retained.push(Box::new(queued));
        }
        for part in range {
            let mut checked = 0;
            let mut decodes: Vec<SecondDecode> = Vec::new();
            for write in state.recorded[part].twin_writes.drain(..) {
                checked += 1;
                decodes.extend(write.into_decode());
            }
            counts.second_twin_checks(checked);
            if decodes.is_empty() {
                continue;
            }
            let raw = self.record_raw(state, |_, buffer| {
                for decode in &decodes {
                    // SAFETY: as above.
                    unsafe { decode.record(device.shared().vk(), buffer) };
                }
                Ok(())
            })?;
            let uses = decodes.iter().flat_map(|decode| decode.uses()).cloned();
            let recorded = &mut state.recorded[part];
            recorded.raws.push(raw);
            recorded
                .summary
                .join(Summary::outside_encoders(uses.collect()));
            counts.second_twin_decodes(decodes.len());
            state.retained.push(Box::new(decodes));
        }
        Ok(())
    }

    /// A command buffer holding only `barriers`, in order, then `transitions` of the caller's
    /// images or of held depths, between the `stamps` it writes, for the batch that signals `at`: where it moves no
    /// image, one the scheduler keeps (`kept_barriers.rs`; one that writes timestamps, kept for
    /// their slot) on a queue whose command buffers end at their timeline's value
    /// (`fences::ends_at_fence`), and else, or past the kept ones' bounds, one recorded into this
    /// command buffer's pool ([`record_barriers`](Self::record_barriers)).
    fn barrier_buffer(
        &self,
        state: &mut State,
        kept: &mut KeptBarriers,
        at: Point,
        barriers: &[Barrier],
        transitions: &[Transition],
        stamps: Stamps,
    ) -> Result<vk::CommandBuffer, CommandBufferError> {
        let device = &self.device;
        if transitions.is_empty() && !super::fences::ends_at_fence(device, at.queue) {
            let taken = kept
                .take(device, at.queue, barriers, stamps.kept(), at.value)
                .map_err(|error| recording_error(device, error))?;
            if let Some(buffer) = taken {
                return Ok(buffer);
            }
        }
        self.record_barriers(state, barriers, transitions, stamps)
    }

    /// A command buffer of this command buffer's pool holding only `barriers`, in order, then
    /// `transitions` of the caller's images or of held depths, from all commands before to all
    /// commands after, between the `stamps` it writes.
    fn record_barriers(
        &self,
        state: &mut State,
        barriers: &[Barrier],
        transitions: &[Transition],
        stamps: Stamps,
    ) -> Result<vk::CommandBuffer, CommandBufferError> {
        let device = &self.device;
        let raw = device.shared().vk();
        let buffer = self
            .allocate_raw(state)
            .map_err(|error| recording_error(device, error))?;
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer was just allocated from the pool, under the state lock, and
        // only this call records it.
        unsafe {
            let begun = raw.begin_command_buffer(buffer, &begin);
            exit_guard::called(Call::BeginCommandBuffer);
            begun.map_err(|error| submission_error(device, error))?;
            device
                .shared()
                .counts
                .command_buffer_begun(Begun::BatchBarriers);
            if let Some((queries, first, _)) = stamps.start {
                // The slot's start and end, where it was not reset on the host (`gpu_time.rs`).
                if !device.shared().timing.resets_on_host() {
                    raw.cmd_reset_query_pool(buffer, queries, first, 2);
                    exit_guard::called(Call::CmdResetQueryPool);
                }
                if !stamps.after {
                    let stage = crate::batch_order::start_stage(device.shared().by_ends);
                    raw.cmd_write_timestamp(buffer, stage, queries, first);
                    exit_guard::called(Call::CmdWriteTimestamp);
                }
            }
            for barrier in barriers {
                barrier.record(raw, buffer);
                exit_guard::called(Call::CmdPipelineBarrier);
            }
            if !transitions.is_empty() {
                let family = device.shared().queue_flags_of(self.queue.vk_queue());
                let images: Vec<vk::ImageMemoryBarrier<'_>> = (transitions.iter())
                    .map(|transition| transition.barrier(family))
                    .collect();
                let all = vk::PipelineStageFlags::ALL_COMMANDS;
                raw.cmd_pipeline_barrier(
                    buffer,
                    all,
                    all,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &images,
                );
                exit_guard::called(Call::CmdPipelineBarrier);
            }
            if let Some((queries, first, _)) = stamps.start.filter(|_| stamps.after) {
                let bottom = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
                raw.cmd_write_timestamp(buffer, bottom, queries, first);
                exit_guard::called(Call::CmdWriteTimestamp);
            }
            if let Some((queries, first, _)) = stamps.end {
                let bottom = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
                raw.cmd_write_timestamp(buffer, bottom, queries, first + 1);
                exit_guard::called(Call::CmdWriteTimestamp);
            }
            let ended = self.end_raw(buffer);
            exit_guard::called(Call::EndCommandBuffer);
            ended.map_err(|error| submission_error(device, error))?;
        }
        Ok(buffer)
    }
}

/// The command buffer's failure for a batch's sparse binding that failed: the device's memory, or
/// the bind itself.
fn bind_error(device: &crate::Device, failure: BindFailure) -> CommandBufferError {
    match failure {
        BindFailure::Memory(error) => recording_error(device, error),
        BindFailure::Vulkan(error) => submission_error(device, error),
    }
}

/// The timestamps a barrier command buffer writes into its command buffer's slot of the device's
/// timestamp pools, each the slot's pool, its first query and the holding's lease
/// (`gpu_time::Slot::lease`): the start (after resetting the slot) before its barriers at the top
/// of the pipe, or with `after` after them at the bottom of the pipe; the end after them.
#[derive(Clone, Copy)]
struct Stamps {
    start: Option<(vk::QueryPool, u32, u64)>,
    after: bool,
    end: Option<(vk::QueryPool, u32, u64)>,
}

impl Stamps {
    /// What a kept command buffer writes for these (`kept_barriers.rs`): `None` where they are
    /// none.
    fn kept(self) -> Option<SlotStamps> {
        let (queries, first, lease) = self.start.or(self.end)?;
        Some(SlotStamps {
            queries,
            first,
            lease,
            start: self.start.is_some(),
            after: self.after,
            end: self.end.is_some(),
        })
    }
}
