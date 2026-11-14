//! Consecutive compute encoders of one command buffer recorded into one `VkCommandBuffer`.
//!
//! Every encoder records into a primary `VkCommandBuffer` of its own (`encoder.rs`), and each
//! batch's submission puts the barrier the tracker plans before each encoder in a command buffer
//! of barriers of its own (`command_buffer/submit.rs`, `command_buffer/kept_barriers.rs`): a
//! command buffer of `N` one-dispatch compute encoders went to the queue as `2N - 1` command
//! buffers. Each boundary between command buffers costs GPU time beyond a barrier inside one
//! (`examples/dispatch_chain.rs`). A compute encoder that opens where the encoder before it
//! leaves nothing that needs a boundary records on in that encoder's `VkCommandBuffer`, the
//! command buffer's tail, and its batch's submission submits no command buffer for it.
//!
//! - **Where.** A compute encoder takes the tail where the encoder before it was a compute
//!   encoder of the same dispatch type ([`Fold`], kept by the command buffer as that one ended)
//!   whose recording is one part, with no work left for later (no segment), no fence update and
//!   nothing that fails or faults its command buffer, in the same batch (no event or semaphore
//!   wait or signal, host step or resource state encoder since: `Batches`), where the command
//!   buffer used no command predication, the encoder trail is off, its protection options are 0,
//!   and both encoders log their calls or neither does (`replay.rs`). So a group lies inside one
//!   batch and one unit of command predication (`predication.rs`), and a fence orders nothing
//!   inside it: what a later encoder waits for is updated after the group's last. Counted:
//!   `WorkCounts::encoders_folded`.
//! - **The barrier between them** is recorded as the encoder records, by the rule the tracker
//!   applies between two encoders of one batch: from each earlier write of a tracked resource's
//!   overlapping range to a later read or write, its accesses made visible, and from each earlier
//!   read to a later write, an execution dependency ([`FoldOrder`]). Each command's barrier joins
//!   the one its own encoder orders it by (a serial encoder's, `tracker::EncoderUses`), in one
//!   `vkCmdPipelineBarrier` with one global `VkMemoryBarrier`; encoders that share no tracked
//!   resource get none, as the tracker plans none between them. A recording that bypasses that
//!   per-command order (an unordered encoder's shadow copy, a memory barrier the program asks for)
//!   first records one barrier from every access of the group's earlier encoders to every later
//!   access ([`FoldOrder::everything`]).
//! - **Its submission.** The folded encoder's part holds no `VkCommandBuffer`, and its summary
//!   holds its fence updates alone; its uses and its fence waits join the summary of the group's
//!   first part, whose `VkCommandBuffer` holds them all, so what the tracker plans before that
//!   part (a barrier, a wait for another queue) orders every encoder of the group after earlier
//!   work. A folded encoder that leaves work for later (a dispatch whose pipelines are being
//!   made, an indirect command buffer's execution) ends the shared `VkCommandBuffer` there and
//!   goes on in its own, which its part holds, planned as any part is.
//! - **Encoded again** (`command_buffer/reencode.rs`). A rebind that moves a buffer one encoder of
//!   a group named encodes every encoder of the group again, each into `VkCommandBuffer`s of its
//!   own, or none: the shared one is never submitted. So each group member's log is held until the
//!   group closes and kept for every member where one member keeps its own. A group with a member
//!   that cannot be encoded again is not encoded again, and its moved members are refused.
//! - **Failed.** A failure of the command buffer that fails one encoder of a group fails every
//!   encoder of it (`State::widen_failure`): a recording failure at a folded encoder's end, and
//!   `CommandBuffer::commit_failed` with a set or a first failed encoder that falls inside a
//!   group. A first failed encoder at a group's boundary (an encoder made under protection
//!   options, which never folds, or one that is not a compute encoder) fails what it failed
//!   before (docs/API.md, "A command buffer committed as failed").

use crate::key_hash::KeyMap;
use crate::replay::Replay;
use crate::resource::ResourceKey;
use crate::tracker::{Access, Barrier, Summary, Use, UseRange, ENTRIES_PER_KEY};
use ash::vk;
use metal2vulkan::metal::DispatchType;
use std::hash::Hash;
use std::ops::Range;

/// What a group's earlier encoders did to one range of one tracked resource, and what the
/// barriers recorded since ordered after it.
#[derive(Clone)]
struct Prior {
    range: UseRange,
    /// Their writes: empty where none wrote.
    write: Access,
    /// What the barriers since made those writes visible to.
    visible: Access,
    /// The stages they read in since the last write.
    reads: vk::PipelineStageFlags2,
    /// The stages the barriers since ordered after those reads.
    after_reads: vk::PipelineStageFlags2,
}

impl Prior {
    /// What it holds once `other`, over a range that overlaps it, joins it over `range`: the
    /// writes and reads of both, made visible and ordered only where both were.
    fn absorb(&mut self, other: Prior, range: UseRange) {
        self.range = range;
        self.write = self.write.union(other.write);
        self.visible = Access::new(
            self.visible.stages & other.visible.stages,
            self.visible.access & other.visible.access,
        );
        self.reads |= other.reads;
        self.after_reads &= other.after_reads;
    }
}

/// The order of a group's next encoder's commands after the group's earlier encoders (the module
/// docs): what those did to the tracked resources, by resource and range.
pub(crate) struct FoldOrder<K> {
    priors: KeyMap<K, Vec<Prior>>,
}

impl<K: Copy + Eq + Hash> FoldOrder<K> {
    /// The order after the encoder whose summary is `summary`.
    pub(crate) fn after(summary: &Summary<K>) -> FoldOrder<K> {
        let mut order = FoldOrder {
            priors: KeyMap::default(),
        };
        order.join(summary);
        order
    }

    /// Note the next encoder of the group ended, with `summary`: its uses come after everything
    /// before it in the group. A write of a range the earlier encoders used replaces what they did
    /// there: a barrier ordered them before it as its commands recorded.
    pub(crate) fn join(&mut self, summary: &Summary<K>) {
        for used in summary.uses() {
            let priors = self.priors.entry(used.key).or_default();
            if !used.write.is_empty() {
                priors.retain(|prior| !used.range.contains(&prior.range));
            }
            match priors.iter_mut().find(|prior| prior.range == used.range) {
                Some(prior) => {
                    // A read alone: no barrier orders it yet after anything.
                    prior.reads |= used.read.stages;
                    prior.after_reads = vk::PipelineStageFlags2::NONE;
                }
                None => priors.push(Prior {
                    range: used.range.clone(),
                    write: used.write,
                    visible: Access::NONE,
                    reads: used.read.stages,
                    after_reads: vk::PipelineStageFlags2::NONE,
                }),
            }
            if priors.len() > ENTRIES_PER_KEY {
                coalesce(priors);
            }
        }
    }

    /// The barrier a command with `uses` needs after the group's earlier encoders: from a write
    /// of an overlapping range of a tracked resource not yet visible to the command's read or
    /// write of it, and from a read of one not yet ordered before its write, as the tracker
    /// plans between two parts of a batch (`tracker::Entry::depend`). `None` where it needs none.
    pub(crate) fn barrier(&self, uses: &[Use<K>]) -> Option<Barrier> {
        let mut barrier = Barrier::default();
        let mut add = |src: Access, dst: Access| {
            barrier.src = barrier.src.union(src);
            barrier.dst = barrier.dst.union(dst);
        };
        for used in uses
            .iter()
            .filter(|used| used.tracked && !used.is_nothing())
        {
            let Some(priors) = self.priors.get(&used.key) else {
                continue;
            };
            for prior in priors
                .iter()
                .filter(|prior| prior.range.overlaps(&used.range))
            {
                let written = !prior.write.is_empty();
                if !used.read.is_empty() && written && !prior.visible.covers(used.read) {
                    add(prior.write, used.read);
                }
                if used.write.is_empty() {
                    continue;
                }
                if written && !prior.visible.covers(used.write) {
                    add(prior.write, used.write);
                }
                if !prior.reads.is_empty() && !prior.after_reads.contains(used.write.stages) {
                    add(
                        Access::new(prior.reads, vk::AccessFlags2::NONE),
                        Access::new(used.write.stages, vk::AccessFlags2::NONE),
                    );
                }
            }
        }
        (!barrier.src.is_empty() && !barrier.dst.is_empty()).then_some(barrier)
    }

    /// Note `barrier` recorded: it made each earlier write its source covers visible to its
    /// destination, and ordered each earlier read its source stages hold before its destination
    /// stages (a global memory barrier orders every command before it in its source scope).
    pub(crate) fn ordered(&mut self, barrier: Barrier) {
        for prior in self.priors.values_mut().flatten() {
            if barrier.src.covers(prior.write) {
                prior.visible = prior.visible.union(barrier.dst);
            }
            if barrier.src.stages.contains(prior.reads) {
                prior.after_reads |= barrier.dst.stages;
            }
        }
    }

    /// The barrier from every access of the group's earlier encoders not yet ordered to every
    /// later access: what a recording that bypasses the per-command order records first. `None`
    /// where none is left to order.
    pub(crate) fn everything(&self) -> Option<Barrier> {
        let all = vk::PipelineStageFlags2::ALL_COMMANDS;
        let mut src = Access::NONE;
        for prior in self.priors.values().flatten() {
            if !prior.write.is_empty() && !prior.visible.covers(Barrier::everything().dst) {
                src = src.union(prior.write);
            }
            if !prior.reads.is_empty() && !prior.after_reads.contains(all) {
                src = src.union(Access::new(prior.reads, vk::AccessFlags2::NONE));
            }
        }
        (!src.is_empty()).then_some(Barrier {
            src,
            dst: Barrier::everything().dst,
        })
    }
}

/// Merge a resource's priors into one per kind of range.
fn coalesce(priors: &mut Vec<Prior>) {
    let mut first = 0;
    while first < priors.len() {
        let mut next = first + 1;
        while next < priors.len() {
            match priors[first].range.hull(&priors[next].range) {
                Some(range) => {
                    let other = priors.swap_remove(next);
                    priors[first].absorb(other, range);
                }
                None => next += 1,
            }
        }
        first += 1;
    }
}

/// `own`, the barrier a command's encoder orders it by, joined with `fold`'s: one barrier of
/// both sources and both destinations, which orders at least what each does.
pub(crate) fn joined(own: Option<Barrier>, fold: Option<Barrier>) -> Option<Barrier> {
    match (own, fold) {
        (Some(own), Some(fold)) => Some(Barrier {
            src: own.src.union(fold.src),
            dst: own.dst.union(fold.dst),
        }),
        (own, fold) => own.or(fold),
    }
}

/// A log of a group's member not kept by itself, held until the group closes (the module docs).
pub(crate) struct Stashed {
    /// The member's place among the command buffer's ended encoders.
    pub(crate) encoder: usize,
    pub(crate) log: Replay,
    /// Its executions' place among the command buffer's.
    pub(crate) executions: Range<usize>,
}

/// What a compute encoder asks of the group its command buffer's tail may hold, as it opens: its
/// dispatch type, and whether it logs its calls (`replay.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Joining {
    pub(crate) dispatch_type: DispatchType,
    pub(crate) logs: bool,
}

/// The group of compute encoders whose last ended one's `VkCommandBuffer` is its command buffer's
/// tail and may take the next compute encoder (the module docs): kept by the command buffer
/// between encoders, held by the encoder that joined it while it is open.
pub(crate) struct Fold {
    /// The tail: the group's shared `VkCommandBuffer`, the last of its first part's.
    pub(crate) tail: vk::CommandBuffer,
    /// The part whose last `VkCommandBuffer` is the tail: the group's first part, whose summary
    /// its later encoders' uses join.
    pub(crate) leader: usize,
    /// The group's encoders, by their places among the command buffer's ended encoders.
    pub(crate) members: Range<usize>,
    /// How many batches the command buffer had as the group's last encoder ended: a later
    /// encoder joins it only in that batch.
    pub(crate) batches: usize,
    /// What its encoders asked as they opened: one dispatch type, and logs or none.
    pub(crate) joining: Joining,
    /// What its encoders did, which the next one's commands order after.
    pub(crate) order: FoldOrder<ResourceKey>,
    /// Its members' logs not kept by themselves.
    pub(crate) stashed: Vec<Stashed>,
}

/// A compute encoder's end as the group of compute encoders it begins or goes on in takes it
/// (`State::fold_ended`).
pub(crate) struct FoldEnd {
    /// What it asked as it opened; `None` where no group may take it (the encoder trail is on).
    pub(crate) joining: Option<Joining>,
    /// The group it joined, where it joined one.
    pub(crate) fold: Option<Box<Fold>>,
    /// Its place among its command buffer's ended encoders.
    pub(crate) encoder: usize,
    /// What its one part used, where that part leaves nothing that needs a `VkCommandBuffer`
    /// boundary after it: no work for later, no fence update, nothing that fails or faults its
    /// command buffer. `None` where it is not so.
    pub(crate) summary: Option<Summary<ResourceKey>>,
    /// Its log, which it did not keep, with its executions' place.
    pub(crate) log: Option<(Replay, Range<usize>)>,
}
