//! A batch's barrier command buffers, recorded once and submitted again.
//!
//! A batch with encoders submits a command buffer of barriers before each encoder the tracker
//! planned a barrier for (before its first, whatever was planned, where the driver's submissions
//! stall on semaphore operations: `Barrier::after_completed`, `submit.rs`), and, where its end did
//! not go into its last encoder's command buffer, one after its last (`Barrier::to_host`, or
//! `Barrier::to_all`: `batch_order.rs`; `submit.rs`): batch after batch, the same few lists of
//! barriers. A timed command buffer's first batch also resets its slot of the device's timestamp
//! pools and writes the start timestamp first, in a command buffer before its first encoder (of the
//! barrier planned for it, or of none), and its last batch writes the end timestamp in its end
//! (`gpu_time.rs`). Each list's command buffer is recorded once, kept by the scheduler, and
//! submitted again by a later batch once the batch that last carried it completed, its queue's
//! timeline at that batch's value as the host saw it: the same commands, in the same place of the
//! same submission, and no command buffer begun for them as a batch is submitted. ANV clears a
//! command buffer's state as its pool resets it and again as it begins it (Mesa 26.2.4,
//! `anv_cmd_state_init`, 62,776 bytes), and packs the standard sample pattern at each primary one's
//! begin where `VK_EXT_sample_locations` is enabled (`genX(BeginCommandBuffer)`): on ms01's Iris Xe
//! those begins were 53% of the scheduler thread's user time, two a command buffer.
//!
//! - **Never pending twice.** A kept command buffer is recorded without
//!   `VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT` and goes to one batch at a time: ANV chains a
//!   submission's command buffers into one batch buffer only where none has that flag
//!   (`anv_cmd_buffer_is_chainable`), and submits the rest one execbuffer each. The batch that
//!   takes one marks it with its own timeline value before it is submitted: a batch whose
//!   submission fails leaves that value to the queue's next batch, so the command buffer waits
//!   for that one instead. Chained so, a kept command buffer names in each submission every
//!   command buffer that ever came next after it, and the submission fails once one of those was
//!   trimmed away (`RuntimeDriverDefect::TrimmedCommandBufferFailsResubmission`): there the
//!   spares free command buffers without trimming their pools (`spares.rs`).
//! - **Timestamps, per slot.** A command buffer that writes timestamps writes them into one slot's
//!   two queries, so it is kept for its slot ([`SlotStamps`]): one per slot and list, taken again
//!   only by a batch of a later holding of that slot (`gpu_time::Slot::lease`), once the batch
//!   that last took it completed. A slot is taken again only once given back, which its command
//!   buffer does after the completion turn read its timestamps (`read_stamps`): so a kept
//!   command buffer writes the slot's queries again only after the results the last holding
//!   wrote were read, and after the slot was reset on the host as it was taken again
//!   (`gpu_time.rs`). On a device that does not reset slots on the host (an adopted one), the
//!   start's reset of the slot's two queries stays in the kept command buffer, before its start
//!   timestamp. Vulkan (`vkCmdResetQueryPool`): "When executed on a queue, this command sets the
//!   status of query indices [firstQuery, firstQuery + queryCount - 1] to unavailable": at each
//!   submission of the kept command buffer again.
//! - **Where.** On every `VkQueue` but one whose command buffers end at a fence
//!   (`fences::ends_at_fence`): MoltenVK reads a command buffer after its timeline signal and
//!   refuses one submitted again before its completed handler ran. A barrier command buffer that
//!   moves the caller's images is recorded by its batch, as before.
//! - **Bounded.** At most [`LISTS`] lists of barriers alone per `VkQueue`, each with at most
//!   [`KEPT`] command buffers, as many as its batches in flight at once needed, and as many lists
//!   that write timestamps, each with at most [`KEPT`] slots' command buffers: past them a batch
//!   records its own into its command buffer's pool. They live in a command pool of the queue's
//!   family, made at the first, and go with the device once its queues are idle.
//! - **Lists that went out of use give way.** A list met while [`LISTS`] of its kind are kept
//!   replaces the one taken least lately among those no batch took in the last [`STALE`] values
//!   of the queue's timeline and whose command buffers all completed, so the lists a program's
//!   first batches met do not hold the places for its life. Its command buffers are kept for the
//!   lists' later recordings, each recorded again from the start (the pool resets one at its
//!   begin, `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT`). Lists all in use in turn, more
//!   than [`LISTS`] of them, replace none: the first kept stay kept and the rest are recorded by
//!   their batches, where replacing one at each use would record every list at each of its uses.
//! - **Counted.** `WorkCounts::barrier_buffers_kept` and `WorkCounts::stamped_buffers_kept` count
//!   the recordings to keep, `WorkCounts::stamped_buffers_reused` the stamped ones taken again.
//! - **Inside the scheduler's section**: every call is in the exit guard's table.

use crate::error::{vulkan, Error};
use crate::exit_guard::{self, Call};
use crate::release;
use crate::tracker::Barrier;
use crate::work_counts::Begun;
use crate::Device;
use ash::vk;
use std::collections::VecDeque;

/// The most lists of barriers a `VkQueue` keeps command buffers of.
pub(crate) const LISTS: usize = 8;

/// The most command buffers a `VkQueue` keeps of one list of barriers.
pub(crate) const KEPT: usize = 128;

/// How many values of its queue's timeline a list no batch took stays kept for before a new list
/// may replace it: a few frames of a window server's (67 or more batches a frame).
pub(crate) const STALE: u64 = 256;

/// The scheduler's kept barrier command buffers, by `VkQueue` (`scheduler::State::kept`).
#[derive(Default)]
pub(crate) struct KeptBarriers {
    queues: Vec<QueueKept>,
}

/// The timestamps a barrier command buffer writes into a timed command buffer's slot of the
/// device's timestamp pools (`gpu_time::Slot`): with `start`, the slot reset before its barriers
/// and the start timestamp before them at the top of the pipe (the bottom where batches are
/// ordered by their ends: `batch_order.rs`), or with `after` after them at the bottom of the pipe;
/// with `end`, the end timestamp after them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SlotStamps {
    /// The slot's timestamp pool.
    pub(crate) queries: vk::QueryPool,
    /// Its first query, the start; the end is the next.
    pub(crate) first: u32,
    /// The holding of the slot the batch is of (`gpu_time::Slot::lease`).
    pub(crate) lease: u64,
    pub(crate) start: bool,
    /// The start written after the barriers, once every command before them completed
    /// (`command_buffer/submit.rs`).
    pub(crate) after: bool,
    pub(crate) end: bool,
}

/// One `VkQueue`'s.
#[derive(Default)]
pub(crate) struct QueueKept {
    /// The pool they are made from; null until the first.
    pool: vk::CommandPool,
    /// Lists of barriers alone.
    lists: Vec<KeptList>,
    /// Lists that also write a slot's timestamps, kept per slot.
    stamped: Vec<KeptList>,
    /// The command buffers of the lists replaced, each completed, for later recordings.
    spare: Vec<vk::CommandBuffer>,
}

/// What a batch takes of a `VkQueue`'s kept command buffers ([`QueueKept::take`]).
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Take {
    /// A kept command buffer of the list, whose last batch completed: the batch's now.
    Kept(vk::CommandBuffer),
    /// A command buffer to record for the list at this index, then kept
    /// ([`QueueKept::recorded`]): a spare one where there is one, else one allocated.
    Record(usize),
    /// None: past the bounds, the batch records its own.
    Past,
}

/// The command buffers of one list of barriers, in the order of the batches that last took them.
struct KeptList {
    barriers: Vec<Barrier>,
    /// Which timestamps its command buffers write, `(start, after, end)` as [`SlotStamps`]
    /// says: none for barriers alone.
    stamps: Option<(bool, bool, bool)>,
    buffers: VecDeque<Kept>,
}

/// A kept command buffer, and the timeline value of the batch that last took it.
struct Kept {
    raw: vk::CommandBuffer,
    after: u64,
    /// Where it writes timestamps: the slot's pool and first query, and the lease of the holding
    /// whose batch last took it.
    slot: Option<(vk::QueryPool, u32, u64)>,
}

impl KeptList {
    /// The last batch that took one of its command buffers; every one completed with it.
    fn last(&self) -> u64 {
        self.buffers.back().map_or(0, |kept| kept.after)
    }
}

impl QueueKept {
    /// What the batch that signals `at` takes for `barriers`, writing `stamps` where given, its
    /// queue's timeline completed through `completed`.
    pub(crate) fn take(
        &mut self,
        barriers: &[Barrier],
        stamps: Option<SlotStamps>,
        at: u64,
        completed: u64,
    ) -> Take {
        let kind = stamps.map(|stamps| (stamps.start, stamps.after, stamps.end));
        let lists = match stamps {
            Some(_) => &mut self.stamped,
            None => &mut self.lists,
        };
        let found = lists
            .iter()
            .position(|list| list.barriers == barriers && list.stamps == kind);
        let index = match found {
            Some(index) => index,
            None if lists.len() < LISTS => {
                lists.push(KeptList {
                    barriers: barriers.to_vec(),
                    stamps: kind,
                    buffers: VecDeque::new(),
                });
                lists.len() - 1
            }
            None => match out_of_use(lists, at, completed) {
                Some(index) => {
                    let list = &mut lists[index];
                    self.spare
                        .extend(list.buffers.drain(..).map(|kept| kept.raw));
                    list.barriers.clear();
                    list.barriers.extend_from_slice(barriers);
                    list.stamps = kind;
                    index
                }
                None => return Take::Past,
            },
        };
        let list = &mut lists[index];
        let Some(stamps) = stamps else {
            // The batches that took them did so in their timeline values' order: the first is
            // the one that may have completed.
            if list.buffers.front().is_some_and(|k| k.after <= completed) {
                if let Some(mut taken) = list.buffers.pop_front() {
                    taken.after = at;
                    let raw = taken.raw;
                    list.buffers.push_back(taken);
                    return Take::Kept(raw);
                }
            }
            if list.buffers.len() >= KEPT {
                return Take::Past;
            }
            return Take::Record(index);
        };
        // The slot's own: taken again by a later holding of the slot, once it completed. One
        // that is not (pending, or this holding's already) leaves the batch to record its own.
        let of_slot = list.buffers.iter().position(|kept| {
            kept.slot.is_some_and(|(queries, first, _)| {
                queries == stamps.queries && first == stamps.first
            })
        });
        match of_slot {
            Some(at_index) => {
                let ready = list.buffers[at_index].after <= completed
                    && list.buffers[at_index]
                        .slot
                        .is_some_and(|(_, _, lease)| lease < stamps.lease);
                if !ready {
                    return Take::Past;
                }
                let Some(mut taken) = list.buffers.remove(at_index) else {
                    return Take::Past;
                };
                taken.after = at;
                taken.slot = Some((stamps.queries, stamps.first, stamps.lease));
                let raw = taken.raw;
                list.buffers.push_back(taken);
                Take::Kept(raw)
            }
            None if list.buffers.len() >= KEPT => Take::Past,
            None => Take::Record(index),
        }
    }

    /// A spare command buffer to record, where a replaced list left one.
    pub(crate) fn spare(&mut self) -> Option<vk::CommandBuffer> {
        self.spare.pop()
    }

    /// `raw`, recorded for the list at `list` ([`Take::Record`]) of the kind `stamps` gives,
    /// taken by the batch that signals `at`.
    pub(crate) fn recorded(
        &mut self,
        list: usize,
        raw: vk::CommandBuffer,
        at: u64,
        stamps: Option<SlotStamps>,
    ) {
        let lists = match stamps {
            Some(_) => &mut self.stamped,
            None => &mut self.lists,
        };
        lists[list].buffers.push_back(Kept {
            raw,
            after: at,
            slot: stamps.map(|stamps| (stamps.queries, stamps.first, stamps.lease)),
        });
    }

    /// Each list's barriers and its command buffers in order, those of barriers alone then those
    /// that write timestamps, with the spare ones.
    #[cfg(test)]
    pub(crate) fn lists(
        &self,
    ) -> (
        Vec<(Vec<Barrier>, Vec<vk::CommandBuffer>)>,
        Vec<vk::CommandBuffer>,
    ) {
        let lists = self
            .lists
            .iter()
            .chain(&self.stamped)
            .map(|list| {
                let buffers = list.buffers.iter().map(|kept| kept.raw).collect();
                (list.barriers.clone(), buffers)
            })
            .collect();
        (lists, self.spare.clone())
    }
}

/// The list of `lists` a new one replaces, its queue's timeline at `at` and completed through
/// `completed`: of those no batch took since `at - STALE` and whose command buffers all
/// completed, the one taken least lately.
fn out_of_use(lists: &[KeptList], at: u64, completed: u64) -> Option<usize> {
    lists
        .iter()
        .enumerate()
        .filter_map(|(index, list)| {
            let last = list.last();
            (last <= completed && last.saturating_add(STALE) <= at).then_some((last, index))
        })
        .min()
        .map(|(_, index)| index)
}

impl KeptBarriers {
    /// A command buffer of `device`'s `VkQueue` `queue` holding only `barriers` (each recorded
    /// as [`Barrier::record`] records it), between the timestamps `stamps` gives, for the batch
    /// that signals `at` on that queue's timeline: a kept one whose last batch completed (and,
    /// with timestamps, of an earlier holding of the slot), else one recorded now and kept.
    /// `None` past the bounds (the module docs), where the batch records its own.
    pub(crate) fn take(
        &mut self,
        device: &Device,
        queue: usize,
        barriers: &[Barrier],
        stamps: Option<SlotStamps>,
        at: u64,
    ) -> Result<Option<vk::CommandBuffer>, Error> {
        let shared = device.shared();
        let completed = shared.queues[queue].completed_value();
        if self.queues.len() <= queue {
            self.queues.resize_with(queue + 1, QueueKept::default);
        }
        let kept = &mut self.queues[queue];
        let list = match kept.take(barriers, stamps, at, completed) {
            Take::Kept(raw) => {
                if stamps.is_some() {
                    shared.counts.stamped_buffer_reused();
                }
                return Ok(Some(raw));
            }
            Take::Past => return Ok(None),
            Take::Record(list) => list,
        };
        let raw = shared.vk();
        if kept.pool == vk::CommandPool::null() {
            // Each command buffer is reset at its begin: a replaced list's are recorded again.
            let info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(shared.queue_family_of(queue));
            // SAFETY: the device is live and `info` valid.
            let pool = unsafe { raw.create_command_pool(&info, None) };
            exit_guard::called(Call::CreateCommandPool);
            kept.pool = pool.map_err(vulkan)?;
        }
        let buffer = match kept.spare() {
            Some(buffer) => buffer,
            None => {
                let info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(kept.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                device.allocate_command_buffers(&info)?[0]
            }
        };
        // Submitted again, one batch at a time: neither one-time nor simultaneous.
        let begin = vk::CommandBufferBeginInfo::default();
        // SAFETY: the command buffer is the scheduler pool's, used under its lock, and only this
        // call records it: just allocated, or a spare one, whose every batch completed (none is
        // pending), which its begin resets (the pool's `RESET_COMMAND_BUFFER`). The slot's
        // queries are the timestamp pool's, live while the device is.
        unsafe {
            let begun = raw.begin_command_buffer(buffer, &begin);
            exit_guard::called(Call::BeginCommandBuffer);
            begun.map_err(vulkan)?;
            shared.counts.command_buffer_begun(Begun::KeptBarriers);
            if let Some(stamps) = stamps.filter(|stamps| stamps.start) {
                if !shared.timing.resets_on_host() {
                    raw.cmd_reset_query_pool(buffer, stamps.queries, stamps.first, 2);
                    exit_guard::called(Call::CmdResetQueryPool);
                }
                if !stamps.after {
                    let stage = crate::batch_order::start_stage(shared.by_ends);
                    raw.cmd_write_timestamp(buffer, stage, stamps.queries, stamps.first);
                    exit_guard::called(Call::CmdWriteTimestamp);
                }
            }
            for barrier in barriers {
                barrier.record(raw, buffer);
                exit_guard::called(Call::CmdPipelineBarrier);
            }
            if let Some(stamps) = stamps.filter(|stamps| stamps.start && stamps.after) {
                let bottom = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
                raw.cmd_write_timestamp(buffer, bottom, stamps.queries, stamps.first);
                exit_guard::called(Call::CmdWriteTimestamp);
            }
            if let Some(stamps) = stamps.filter(|stamps| stamps.end) {
                let bottom = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
                raw.cmd_write_timestamp(buffer, bottom, stamps.queries, stamps.first + 1);
                exit_guard::called(Call::CmdWriteTimestamp);
            }
            let ended = raw.end_command_buffer(buffer);
            exit_guard::called(Call::EndCommandBuffer);
            ended.map_err(vulkan)?;
        }
        match stamps {
            Some(_) => shared.counts.stamped_buffer_kept(),
            None => shared.counts.barrier_buffer_kept(),
        }
        self.queues[queue].recorded(list, buffer, at, stamps);
        Ok(Some(buffer))
    }

    /// Release the pools, with every kept command buffer, as the device is torn down after its
    /// queues are idle.
    pub(crate) fn release(&mut self, raw: &ash::Device) {
        for kept in self.queues.drain(..) {
            if kept.pool != vk::CommandPool::null() {
                // SAFETY: the device's queues are idle: no kept command buffer is pending.
                unsafe { release::object(raw, kept.pool) };
            }
        }
    }
}
