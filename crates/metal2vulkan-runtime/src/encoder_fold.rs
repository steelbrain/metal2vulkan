//! Consecutive compute, blit and render encoders of one command buffer recorded into one
//! `VkCommandBuffer`.
//!
//! Every encoder records into a primary `VkCommandBuffer` of its own (`encoder.rs`), and each
//! batch's submission puts the barrier the tracker plans before each encoder in a command buffer
//! of barriers of its own (`command_buffer/submit.rs`, `command_buffer/kept_barriers.rs`): a
//! command buffer of `N` one-dispatch compute encoders went to the queue as `2N - 1` command
//! buffers. Each boundary between command buffers costs GPU time beyond a barrier inside one
//! (`examples/dispatch_chain.rs`), and each `vkBeginCommandBuffer` host time (ANV clears its
//! command buffer state there). A compute or blit encoder that opens where the encoder before it
//! leaves nothing that needs a boundary records on in that encoder's `VkCommandBuffer`, the
//! command buffer's tail, and its batch's submission submits no command buffer for it; so does a
//! render encoder whose pass records at its end (**Render encoders**, below).
//!
//! - **Where.** A compute or blit encoder takes the tail where the encoder before it was a
//!   compute, blit or render encoder ([`Fold`], kept by the command buffer as that one ended,
//!   **Render encoders** below) whose recording is
//!   one part, with no work left for later (no segment), no fence update and nothing that fails
//!   or faults its command buffer, nor work its batch's submission records after it or judges
//!   by its part (a write into an ASTC carrier, whose second twin's decode the submission records
//!   after the part; a texture it writes whole before any of its commands reads it, whose zeros
//!   the submission leaves out by what its part read: `Overwrite`), in the same batch (no event or
//!   semaphore wait or signal, host step or resource state encoder since: `Batches`), where the
//!   command buffer used no command predication, the encoder trail is off and its protection
//!   options are 0; a compute encoder only where the group's compute encoders are of its dispatch
//!   type ([`Joining`]). So a group lies inside one batch, and a fence orders nothing inside it:
//!   what a later encoder waits for is updated after the group's last. A blit encoder ends a unit
//!   of command predication (`predication.rs`), so a group that holds one may hold compute
//!   encoders of two units: a record that leaves its unit's work out (an unbalanced one) from an
//!   encoder past the group's first splits the group at the commit (**Failed**, below). Every
//!   compute and blit encoder logs its calls (`replay.rs`), and a render encoder takes a group
//!   only with a log, so each of a group's encoders can be encoded again apart. Counted:
//!   `WorkCounts::encoders_folded`.
//! - **Render encoders.** A render encoder takes a group only where it logs its calls (`replay.rs`:
//!   the device has a buffer over host pages; a device with none keeps one `VkCommandBuffer` for
//!   each render encoder), its pass is of one color group and over no rasterization-rate map
//!   (its later groups, and a reset of its map, encode it again alone), and its depth attachment
//!   is no image the device holds out of `GENERAL` between passes (`texture/held_depth.rs`: a
//!   holding encoder's moves go before its own `VkCommandBuffer`; and a group's other encoders'
//!   moves go before the group, `GroupPlaces`, each into `GENERAL`, which no member then leaves).
//!   One whose pass records at its end, its draws in a secondary command buffer the pass executes
//!   there, joins a group as a blit encoder does (`Joining::Render`); one whose first instance
//!   records inline (`render/inline_first.rs`) begins its pass at its first draw and records the
//!   commands its end finds before the pass in a `VkCommandBuffer` that goes before the pass's, so
//!   it joins none, and begins its own. Either may begin a group, as its end leaves the tail
//!   outside any render pass instance; one that begins a group no encoder joins pays for it as
//!   the next encoder opens (the group closed, nothing built for it: [`Fold::order`] waits for a
//!   second encoder). No barrier may record inside a pass instance, and the
//!   encoder records its pass, and the steps before and after it, through commands that name none
//!   of the pass's uses: one that joined a group records one barrier from everything of the
//!   group's earlier encoders not yet ordered to everything after ([`FoldOrder::everything`]) at
//!   its open, before any command of its own, as an encoder past the scheduler's planned
//!   barriers takes one from everything before at its open (`encoder.rs`). What its pass leaves
//!   for later (a draw or tile dispatch whose pipeline is being made, an execution of render
//!   commands: `deferred.rs` segments), a fence update, or a write it judges by its part
//!   (`Overwrite`) ends the group after it, as for any encoder; a late resolve target
//!   (`render/late_resolve.rs`) is the pass's own, recorded at its end in its `VkCommandBuffer`.
//! - **The barrier between them** is recorded as the encoder records, by the rule the tracker
//!   applies between two encoders of one batch: from each earlier write of a tracked resource's
//!   overlapping range to a later read or write, its accesses made visible, and from each earlier
//!   read to a later write, an execution dependency ([`FoldOrder`]). Each command's barrier joins
//!   the one its own encoder orders it by (a serial encoder's, `tracker::EncoderUses`), in one
//!   `vkCmdPipelineBarrier` with one global `VkMemoryBarrier`; encoders that share no tracked
//!   resource get none, as the tracker plans none between them. A recording that bypasses that
//!   per-command order (an unordered encoder's shadow copy, a barrier of the runtime's own between
//!   its commands) first records one barrier from every access of the group's earlier encoders to
//!   every later access ([`FoldOrder::everything`]). A memory barrier the program asks for
//!   touches no memory and records none of it: each command after it takes its own order.
//! - **Image uses after buffer writes** (`alias_split.rs`). On a device whose driver's image uses
//!   miss a write through a buffer over their memory recorded before them in the same
//!   `VkCommandBuffer` (`DriverFact::ImageUsesMissBufferWritesInOneCommandBuffer`), a command
//!   that uses an image over bytes the group's earlier encoders wrote ([`FoldOrder::byte_writes`])
//!   or its own encoder's earlier commands wrote ends the tail and goes on in a `VkCommandBuffer`
//!   of its own, as a folded encoder that leaves work for later does (below), and a render
//!   encoder joins no group whose earlier encoders wrote bytes.
//! - **Its submission.** The folded encoder's part holds no `VkCommandBuffer`, and its summary
//!   holds its fence updates alone; its uses and its fence waits join the summary of the group's
//!   first part, whose `VkCommandBuffer` holds them all, so what the tracker plans before that
//!   part (a barrier, a wait for another queue) orders every encoder of the group after earlier
//!   work. A folded encoder that leaves work for later (a dispatch whose pipelines are being
//!   made, an indirect command buffer's execution) ends the shared `VkCommandBuffer` there and
//!   goes on in its own, which its part holds, planned as any part is.
//! - **Encoded again** (`command_buffer/reencode.rs`). A rebind that moves a buffer one encoder of
//!   a group named encodes every encoder of the group again, each into `VkCommandBuffer`s of its
//!   own, or none: the shared one is never submitted. A group with a member that cannot be encoded
//!   again is split where each such member lies in one of Metal's submissions
//!   (`checks::metal_submissions`: a blit encoder's and a compute encoder's are two) that holds
//!   one of them the rebind moved, which the first submission finds rebound and runs nothing of:
//!   the group's encoders outside those submissions are encoded again apart and run, as Metal
//!   runs them. Else it is not encoded again, and its moved members are refused.
//! - **Failed.** A failure of the command buffer that fails some of a group's encoders and not
//!   the others (`CommandBuffer::commit_failed`'s first failed encoder or set inside a group, a
//!   recording failure or a fault at a folded encoder), and a command-predication record that
//!   leaves out the work of its unit's encoders from one past a group's first
//!   (`State::leave_out_ended`), splits the group at the commit (`State::split_failure`): the
//!   encoders that run are encoded again from their logs, each into `VkCommandBuffer`s of its
//!   own, before the first submission, the shared one never submitted, and the failed or left out
//!   ones run nothing: exactly the encoders the failure names, or the record leaves out, run
//!   nothing, as on Metal. Where the encoders that run cannot all be encoded again (a compute
//!   encoder whose unit of command predication laid a loop or left work out, so a command of its
//!   encoded again would run outside its block: `Predication::alone`; a blit or render encoder
//!   belongs to no unit; or a call answers otherwise when encoded again), the failure fails the
//!   whole group (docs/API.md, "A command buffer committed as failed"). A command buffer with a
//!   submission check (`command_buffer/checks.rs`), which fails one of Metal's submissions as it
//!   is submitted, splits so at the commit each group whose shared `VkCommandBuffer` holds
//!   commands of more than one of them (a render pass is a submission of its own), those
//!   commands' encoders encoded again apart, so a check fails its own submission's encoders alone
//!   (an encoder that went on in `VkCommandBuffer`s of its own before any command of its recorded
//!   there is failed by its own parts); one that cannot be split so stays whole, and a check that
//!   fails one of those submissions fails each of them.
//! - **Logs.** A group's members' logs are held from the group's close for the commit, which
//!   keeps them for every member where one member keeps its own for a rebind, and keeps those of
//!   a group its failure splits; the rest stay held until the command buffer's release, which
//!   gives them back to the device's spares under one lock on its queue's handler thread
//!   (`SpareLogs::give_held`), as it gives back the logs its encoders kept. The spares keep as
//!   many as it gave back within 1 MiB of room, so a next command buffer's encoders reuse them
//!   rather than allocate. A log holds one
//!   `ComputeCall` per call (at most 64 bytes, `tests::encoder_fold`) besides the bytes and handles
//!   its slices take in its arenas: a one-dispatch encoder's five calls (state, three buffers, the
//!   dispatch) hold about 320 bytes, so 2,300 such encoders hold about 0.7 MiB until their command buffer completes.

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
    /// The order after nothing.
    pub(crate) fn none() -> FoldOrder<K> {
        FoldOrder {
            priors: KeyMap::default(),
        }
    }

    /// The order after the encoder whose summary is `summary`.
    pub(crate) fn after(summary: &Summary<K>) -> FoldOrder<K> {
        let mut order = FoldOrder::none();
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

    /// Whether the group's earlier encoders used the resource of `key`.
    fn names(&self, key: K) -> bool {
        self.priors.contains_key(&key)
    }

    /// The bytes of each tracked resource the group's earlier encoders wrote, in its shared
    /// `VkCommandBuffer`: what an encoder that joins the group on a device that splits at aliases
    /// starts from (`alias_split.rs`).
    pub(crate) fn byte_writes(&self) -> impl Iterator<Item = (K, Range<u64>)> + '_ {
        self.priors.iter().flat_map(|(&key, priors)| {
            (priors.iter())
                .filter(|prior| !prior.write.is_empty())
                .filter_map(move |prior| match &prior.range {
                    UseRange::Bytes(bytes) => Some((key, bytes.clone())),
                    UseRange::Image { .. } => None,
                })
        })
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

/// A log of a group's member not kept by itself, held for the commit (the module docs).
pub(crate) struct Stashed {
    /// The member's place among the command buffer's ended encoders.
    pub(crate) encoder: usize,
    pub(crate) log: Replay,
    /// Its executions' place among the command buffer's.
    pub(crate) executions: Range<usize>,
}

/// What an encoder asks of the group its command buffer's tail may hold, as it opens: a compute
/// encoder joins one whose compute encoders are of its dispatch type (or that holds none), as a
/// unit of command predication holds compute encoders of one dispatch type (`predication.rs`); a
/// blit encoder joins any; a render encoder joins any where `joins`, else none, and may begin one
/// either way (the module docs, **Render encoders**).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Joining {
    Compute(DispatchType),
    Blit,
    Render { joins: bool },
}

impl Joining {
    /// The dispatch type of the compute encoders of a group it joins: its own, a compute
    /// encoder's; `group`'s, a blit or render encoder's.
    fn dispatch_type(self, group: Option<DispatchType>) -> Option<DispatchType> {
        match self {
            Joining::Compute(dispatch_type) => Some(dispatch_type),
            Joining::Blit | Joining::Render { .. } => group,
        }
    }
}

/// The group of encoders whose last ended one's `VkCommandBuffer` is its
/// command buffer's tail and may take the next such encoder (the module docs): kept by the
/// command buffer between encoders, held by the encoder that joined it while it is open.
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
    /// The dispatch type of its compute encoders; `None` while it holds blit encoders alone.
    pub(crate) dispatch_type: Option<DispatchType>,
    /// Whether every encoder it holds is a blit encoder: a later one's write of a texture whole
    /// joins its first part's (`Fold::overwrites_join`).
    pub(crate) blits_alone: bool,
    /// What its encoders did, which the next one's commands order after: empty until a second
    /// encoder joins, which builds it from its first part's summary (`State::join_fold`), so a
    /// group no encoder joins costs no copy of what its first encoder used.
    pub(crate) order: FoldOrder<ResourceKey>,
    /// Its members' logs not kept by themselves.
    pub(crate) stashed: Vec<Stashed>,
}

impl Fold {
    /// Whether an encoder that asks `joining` may join it, by what it holds.
    pub(crate) fn admits(&self, joining: Joining) -> bool {
        match joining {
            Joining::Compute(dispatch_type) => self
                .dispatch_type
                .is_none_or(|group| group == dispatch_type),
            Joining::Blit => true,
            Joining::Render { joins } => joins,
        }
    }

    /// The group `joining`'s encoder begins, its first part `leader`, its first `VkCommandBuffer`
    /// the tail; its order is built as a second encoder joins.
    pub(crate) fn begun(
        tail: vk::CommandBuffer,
        (leader, encoder): (usize, usize),
        (batches, joining): (usize, Joining),
        stashed: Vec<Stashed>,
    ) -> Fold {
        Fold {
            tail,
            leader,
            members: encoder..encoder + 1,
            batches,
            dispatch_type: joining.dispatch_type(None),
            blits_alone: joining == Joining::Blit,
            order: FoldOrder::none(),
            stashed,
        }
    }

    /// `joining`'s encoder, which ended with `summary`, goes on in the group.
    pub(crate) fn went_on(&mut self, joining: Joining, summary: &Summary<ResourceKey>) {
        self.dispatch_type = joining.dispatch_type(self.dispatch_type);
        self.blits_alone &= joining == Joining::Blit;
        self.order.join(summary);
    }

    /// Whether a folded encoder's write of the image of `key` whole, before any of its commands
    /// reads it (`Overwrite`), stands as the write of the group's first part would: every
    /// earlier encoder of the group is a blit encoder (any other's commands may reach a texture
    /// unnamed) and none named the image. Its first part's then holds it, which the batch's
    /// submission judges as it judged the encoder's own part before any encoder was folded
    /// (`submit.rs`'s `first_overwrites`).
    pub(crate) fn overwrites_join(&self, key: ResourceKey) -> bool {
        self.blits_alone && !self.order.names(key)
    }
}

/// The place, among a batch's parts, of the part whose `VkCommandBuffer` begins each encoder's
/// commands, for what goes before them (the held depths' moves, `texture/held_depth.rs`): a
/// part's own, or, for an encoder of a group past its first, the group's first part's, whose
/// `VkCommandBuffer` holds the group's commands. A group lies inside one batch, its first
/// encoder's one part before its other encoders' parts.
#[derive(Default)]
pub(crate) struct GroupPlaces {
    /// The first encoder of the last group met, and the place of its part.
    leader: Option<(usize, usize)>,
}

impl GroupPlaces {
    /// The place for the part at `at` of `encoder`, of the group of encoders `group` where it is
    /// one of a group's (`EncoderRecord::group`).
    pub(crate) fn place(
        &mut self,
        at: usize,
        encoder: usize,
        group: Option<&Range<usize>>,
    ) -> usize {
        let Some(group) = group else {
            return at;
        };
        match self.leader {
            Some((first, place)) if first == group.start && encoder != first => place,
            _ => {
                if encoder == group.start {
                    self.leader = Some((encoder, at));
                }
                at
            }
        }
    }
}

/// A compute, blit or render encoder's end as the group it begins or goes on in takes it
/// (`State::fold_ended`).
pub(crate) struct FoldEnd {
    /// What it asked as it opened; `None` where no group may take it (the encoder trail is on).
    pub(crate) joining: Option<Joining>,
    /// The group it joined, where it joined one.
    pub(crate) fold: Option<Box<Fold>>,
    /// Whether a command of its recorded into that group's tail.
    pub(crate) in_tail: bool,
    /// Its place among its command buffer's ended encoders.
    pub(crate) encoder: usize,
    /// Whether its one part leaves nothing that needs a `VkCommandBuffer` boundary after it: no
    /// work for later, no fence update, nothing that fails or faults its command buffer, no write
    /// into an ASTC carrier and no `Overwrite`.
    pub(crate) plain: bool,
    /// What that part used, where it is plain and joined a group, whose next encoders order
    /// after it; `None` else (the first part of a group it begins holds its summary, which the
    /// group's second encoder orders after: `State::join_fold`).
    pub(crate) summary: Option<Summary<ResourceKey>>,
    /// Its log, which it did not keep, with its executions' place.
    pub(crate) log: Option<(Replay, Range<usize>)>,
}
