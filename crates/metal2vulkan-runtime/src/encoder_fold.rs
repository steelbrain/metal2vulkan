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
//!   one part, with no work left for later (no segment) and nothing that fails
//!   or faults its command buffer, nor work its batch's submission records after it or judges
//!   by its part (a write into an ASTC carrier, whose second twin's decode the submission records
//!   after the part; a texture it writes whole before any of its commands reads it, whose zeros
//!   the submission leaves out by what its part read: `Overwrite`), in the same batch (no event or
//!   semaphore wait or signal, host step or resource state encoder since: `Batches`), where the
//!   command buffer used no command predication, the encoder trail is off and its protection
//!   options are 0; a compute encoder only where the group's compute encoders are of its dispatch
//!   type ([`Joining`]). So a group lies inside one batch. A blit encoder and a fence update end
//!   a unit of command predication (`predication.rs`), so a group that holds one may hold compute
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
//!   commands: `deferred.rs` segments) or a write it judges by its part
//!   (`Overwrite`) ends the group after it, as for any encoder; a late resolve target
//!   (`render/late_resolve.rs`) is the pass's own, recorded at its end in its `VkCommandBuffer`.
//! - **Continued passes.** A render encoder whose pass recorded every instance inline and whose
//!   end records nothing after the last leaves that last render pass instance open in its last
//!   `VkCommandBuffer`, which goes on as the group's tail ([`OpenPass`], `Fold::open`), where that
//!   instance touched nothing but its attachments: no draw wrote anything else (a storage write)
//!   or used the attachments' images or bytes another way, none of its colors is in the feedback
//!   loop layout, and no command of the pass reached memory it does not name (a root or handle
//!   table, a heap, an execution of render commands); its loads and stores are `Load` or `Clear`
//!   and `Store`, as an instance recorded inline has (`render/inline_first.rs`), with no resolve,
//!   visibility result or rasterization-rate map, and the device times no encoder. Its earlier
//!   instances end, and store, in that `VkCommandBuffer` as before; the last one's store is its
//!   render pass's `Store`, at the end of the instance, whoever ends it. The next render encoder, whose
//!   first instance may record inline (`Joining::Render`'s `continues`), keeps that group as it
//!   opens; where its first instance renders what the open one does
//!   (`render_target::PassKey`: compatible render passes of one shape, the same views over the
//!   same extent, layers and render area) and loads every attachment (`Load`), its draws are kept
//!   as they come
//!   (`render/inline_splits.rs`'s `Kept`), and at its first split or its end it records them into
//!   the open instance, with no end, barrier or begin between, where its encoder recorded no
//!   command before the pass (its pass took its own `VkCommandBuffer`, unbegun by anything, past
//!   no barrier at its open) and the instance has no copy or step before it, its draws touched
//!   nothing but its attachments either, and the group admits it as it admits a joining encoder
//!   (its tail and batch unchanged, no command predication, no protection options, no encoder
//!   trail, no device that splits at aliases): it joins the group there, its uses joining the
//!   group's first part's summary, its own `VkCommandBuffer` ended unsubmitted. Else it closes
//!   the group, ending the open instance in the tail, and records its own as before. Anything
//!   else that records into the tail ends the instance first: an encoder that joins the group,
//!   the group's close, the tail's end. Counted: `WorkCounts::render_passes_continued`.
//!
//!   Its pass's end notes its uses with no barrier after the group's earlier encoders
//!   (`EncoderCore::command_recording_nothing`), which would end the instance it leaves open for
//!   nothing: a group whose tail holds an open instance holds its leader and the encoders that
//!   continued it alone (any other that joins ends the instance), each of whose last instance
//!   touched nothing but its attachments, so every other access of theirs came before a barrier
//!   from everything before (a split's, `render/inline_splits.rs`) that came before that
//!   instance; the continuing pass's first instance is ordered after their attachment accesses
//!   by rasterization order and after the rest by that barrier, and its later instances follow
//!   its own split's barrier. An encoder that joins later orders itself after all of theirs.
//!
//!   Why it is Metal's answer. Metal runs the two passes one after the other, the second loading
//!   what the first stored. In one subpass, "for a given (x,y,layer,sample) sample location, the
//!   following operations are guaranteed to execute in rasterization order, for each separate
//!   primitive that includes that sample location: Fragment operations, in the order defined;
//!   Blending, logic operations, and color writes" (Rasterization Order), and rasterization order
//!   follows primitive order, of which "Submission order determines the initial ordering"
//!   (Primitive Order): every fragment operation of the later encoder's draws comes after the
//!   earlier's at each sample, as its load (`VK_ATTACHMENT_LOAD_OP_LOAD`, "the previous contents
//!   of the image within the render area will be preserved as the initial values") after the
//!   earlier's store (`VK_ATTACHMENT_STORE_OP_STORE`, "the contents generated during the render
//!   pass and within the render area are written to memory") would have it. Neither pass wrote
//!   anything else, so nothing else of the earlier one needs ordering before the later one's
//!   draws, nor anything a fence between them covers; what reads the attachments otherwise (a
//!   sampled read of a target, a copy) records after a split or the instance's end, behind a
//!   barrier as before. The later encoder's draws record every state they take into the
//!   `VkCommandBuffer` from nothing bound (`render/bound.rs`), as into a new one, so each starts
//!   from Metal's defaults (its viewport, scissor, depth bias, stencil reference, blend color,
//!   cull mode, winding and fill mode, its own or Metal's). Its label and place among the
//!   command buffer's ended encoders are its own, as every encoder's of a group. A rebind or a
//!   failure encodes or splits the group as any (**Encoded again**, **Failed**): its encoders
//!   encode again apart, each its own pass, from their logs. "There is no implicit ordering
//!   between separate render passes, even in the same command buffer, and even when the
//!   attachments match" (Render Pass), so two passes ordered only by one instance's
//!   rasterization order must be one instance. A memory barrier between the later encoder's
//!   draws records inside the continued instance as inside its own: "If vkCmdPipelineBarrier is
//!   called within a render pass instance using a VkRenderPass object, the render pass must have
//!   been created with at least one subpass dependency that expresses a dependency from the
//!   current subpass to itself" whose scopes hold the barrier's
//!   (`VUID-vkCmdPipelineBarrier-None-07889`). Compatible render passes are "otherwise
//!   identical" but for layouts and load and store operations, so they hold the same subpass
//!   dependencies, and every render pass of a shape depends on itself as `render_target`'s
//!   **Between draws** says, by region and view by view as the barrier is
//!   (`VUID-vkCmdPipelineBarrier-dependencyFlags-07891`, `VUID-vkCmdPipelineBarrier-None-07893`);
//!   its first scope takes the earlier encoder's draws too, which only adds an order Metal has
//!   between the two encoders.
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
//! - **Fences.** An encoder that updates a fence goes on in its group, and the next encoder may
//!   join it. Metal orders a fence's wait after its last update committed or encoded before it
//!   (the `MTLFence` page: "commit the command buffers with the producing passes before those
//!   with the consuming passes", and in one command buffer "encode the producing passes before
//!   the consuming passes"; across queues too: "Fences can synchronize passes you submit to
//!   different queues"), and the tracker evaluates each wait at submission, after the fence's
//!   last update in submission order (`tracker.rs`): a wait never refers to a later update, so no
//!   two waits can wait for each other. A wait of another command buffer or queue orders after
//!   the whole group, its `VkCommandBuffer` one in submission order and its batch's signal at its
//!   end, as it ordered after the batch's end before; a group never crosses a batch. Inside the
//!   group, every command of a later member orders after the writes the update covers
//!   ([`FoldOrder::fence_barrier`]), whether or not its encoder waits for the fence: a wait
//!   covers all of the encoder's commands, "a fence that the pass waits for before it runs any of
//!   its commands" (`waitForFence:`), and the runtime waits for fences at an encoder's start, but
//!   the call may come after some of its commands went into the group's `VkCommandBuffer`. The
//!   barrier is the narrowest that covers a fence, joined to the command's own: from the
//!   updating encoders' writes in the stages the update covers (storage writes in shader stages,
//!   transfer writes in transfer stages, attachment writes in a pass's), to what the command may
//!   reach of memory no use names in its own stages (storage and sampled reads and storage writes
//!   in shader stages, transfer reads and writes in transfer stages) and to its untracked uses'
//!   own reads and writes, so to an indirect argument or uniform read only where such a use reads
//!   one (the tracked resources take the tracker's rule, above). One
//!   recorded, a later command that reaches no more records none for it. It orders more than
//!   Metal, which orders only an encoder that waits. A member's wait for a fence updated before
//!   the group joins the group's first part's summary (**Its submission**, below), which the
//!   tracker plans before the group. Counted: `WorkCounts::fold_fence_barriers`.
//! - **Refusals.** An encoder that asks for a group and joins none is counted as it opens by the
//!   first reason that holds ([`Refusal`], `WorkCounts::folds_refused_kind` and kin): the
//!   group's encoders are of another kind, its tail moved, a batch boundary came since, the
//!   command buffer's state (command predication, protection options, the encoder trail), or,
//!   where the tail holds no group, what the encoder before left (work for later or more than
//!   one part, or something that ends a group). The command buffer's first encoder is refused
//!   nothing. A refusal costs one counter's increment; a join, none.
//! - **Image uses after buffer writes** (`alias_split.rs`). On a device whose driver's image uses
//!   miss a write through a buffer over their memory recorded before them in the same
//!   `VkCommandBuffer` (`DriverFact::ImageUsesMissBufferWritesInOneCommandBuffer`), a command
//!   that uses an image over bytes the group's earlier encoders wrote ([`FoldOrder::byte_writes`])
//!   or its own encoder's earlier commands wrote ends the tail and goes on in a `VkCommandBuffer`
//!   of its own, as a folded encoder that leaves work for later does (below); a render encoder
//!   that joined a group does so before its pass, at the first command it records into the tail
//!   once a draw uses such an image or a write meets an image a draw uses, or at its end where its
//!   pass's own uses meet them.
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
use crate::render_target::PassKey;
use crate::replay::Replay;
use crate::resource::ResourceKey;
use crate::tracker::{Access, Barrier, Summary, Use, UseRange, ENTRIES_PER_KEY};
use crate::Device;
use ash::vk;
use metal2vulkan::metal::DispatchType;
use std::hash::Hash;
use std::ops::Range;
use std::sync::Arc;

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
/// docs): what those did to the tracked resources, by resource and range, and the writes of those
/// that updated a fence (**Fences** in the module docs).
pub(crate) struct FoldOrder<K> {
    priors: KeyMap<K, Vec<Prior>>,
    /// The writes the group's fence updates cover, in the stages the updating encoders' commands
    /// ran in that each update covers: every later command of the group orders after them, to
    /// what it may reach of memory no use names. Empty where no member updated a fence.
    fenced: Access,
    /// What the barriers since the last update made those writes visible to.
    fenced_visible: Access,
}

impl<K: Copy + Eq + Hash> FoldOrder<K> {
    /// The order after nothing.
    pub(crate) fn none() -> FoldOrder<K> {
        FoldOrder {
            priors: KeyMap::default(),
            fenced: Access::NONE,
            fenced_visible: Access::NONE,
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
    /// there: a barrier ordered them before it as its commands recorded. Its fence updates add the
    /// writes of the stages they cover to the fenced writes, visible to nothing yet.
    pub(crate) fn join(&mut self, summary: &Summary<K>) {
        let updated = summary.update_stages();
        if !updated.is_empty() {
            self.fenced = self.fenced.union(Access::new(updated, writes_in(updated)));
            self.fenced_visible = Access::NONE;
        }
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

    /// The barrier a command in `stages` with `uses` needs after the group's fenced writes: from
    /// them to what the command may reach of memory no use names in its stages
    /// ([`unnamed_access`]), and to each of its untracked uses' reads and writes (an indirect
    /// argument or uniform read only where such a use reads one; a tracked resource is ordered by
    /// [`barrier`](Self::barrier)), where the barriers since made them visible to none of it.
    /// `None` where no member updated a fence, or they did.
    pub(crate) fn fence_barrier(
        &self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<K>],
    ) -> Option<Barrier> {
        if self.fenced.is_empty() {
            return None;
        }
        let untracked = uses.iter().filter(|used| !used.tracked);
        let dst = untracked.fold(unnamed_access(stages), |dst, used| {
            dst.union(used.read).union(used.write)
        });
        (!dst.is_empty() && !self.fenced_visible.covers(dst)).then_some(Barrier {
            src: self.fenced,
            dst,
        })
    }

    /// Whether fenced writes are left that a barrier from everything would order: what
    /// [`everything`](Self::everything) records from besides the tracked resources' writes.
    pub(crate) fn fenced_open(&self) -> bool {
        !self.fenced.is_empty() && !self.fenced_visible.covers(Barrier::everything().dst)
    }

    /// Note `barrier` recorded: it made each earlier write its source covers visible to its
    /// destination, and ordered each earlier read its source stages hold before its destination
    /// stages (a global memory barrier orders every command before it in its source scope).
    pub(crate) fn ordered(&mut self, barrier: Barrier) {
        if !self.fenced.is_empty() && barrier.src.covers(self.fenced) {
            self.fenced_visible = self.fenced_visible.union(barrier.dst);
        }
        for prior in self.priors.values_mut().flatten() {
            if barrier.src.covers(prior.write) {
                prior.visible = prior.visible.union(barrier.dst);
            }
            if barrier.src.stages.contains(prior.reads) {
                prior.after_reads |= barrier.dst.stages;
            }
        }
    }

    /// Note a render pass instance continued (**Continued passes** in the module docs): the
    /// attachment accesses of the group's earlier encoders, which rasterization order orders
    /// before the attachment accesses of every later draw of the instance, their writes visible
    /// to them; so too their fenced writes. Their other accesses came before a barrier from
    /// everything before the instance began (no encoder that touched anything else in its last
    /// instance left it open).
    pub(crate) fn continued(&mut self) {
        let attachments = unnamed_access(ATTACHMENT_STAGES);
        for prior in self.priors.values_mut().flatten() {
            if attachments.covers(prior.write) {
                prior.visible = prior.visible.union(attachments);
            }
            // Their attachment reads (a load) come before every later draw's attachment writes
            // at each sample too.
            if ATTACHMENT_STAGES.contains(prior.reads) {
                prior.after_reads |= ATTACHMENT_STAGES;
            }
        }
        if !self.fenced.is_empty() {
            self.fenced_visible = self.fenced_visible.union(attachments);
        }
    }

    /// The barrier from every access of the group's earlier encoders not yet ordered, and their
    /// fenced writes, to every later access: what a recording that bypasses the per-command order
    /// records first. `None` where none is left to order.
    pub(crate) fn everything(&self) -> Option<Barrier> {
        let all = vk::PipelineStageFlags2::ALL_COMMANDS;
        let mut src = Access::NONE;
        if self.fenced_open() {
            src = src.union(self.fenced);
        }
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

/// The stages of shaders: their storage and sampled accesses reach memory no use names (an
/// argument buffer's, a device address's).
const SHADER_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::VERTEX_SHADER.as_raw()
        | vk::PipelineStageFlags2::TESSELLATION_CONTROL_SHADER.as_raw()
        | vk::PipelineStageFlags2::TESSELLATION_EVALUATION_SHADER.as_raw()
        | vk::PipelineStageFlags2::GEOMETRY_SHADER.as_raw()
        | vk::PipelineStageFlags2::FRAGMENT_SHADER.as_raw()
        | vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw()
        | vk::PipelineStageFlags2::TASK_SHADER_EXT.as_raw()
        | vk::PipelineStageFlags2::MESH_SHADER_EXT.as_raw()
        | vk::PipelineStageFlags2::PRE_RASTERIZATION_SHADERS.as_raw(),
);

/// The stages of transfers.
const TRANSFER_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::ALL_TRANSFER.as_raw()
        | vk::PipelineStageFlags2::COPY.as_raw()
        | vk::PipelineStageFlags2::BLIT.as_raw()
        | vk::PipelineStageFlags2::RESOLVE.as_raw()
        | vk::PipelineStageFlags2::CLEAR.as_raw(),
);

/// The stages of a pass's attachments.
const ATTACHMENT_STAGES: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT.as_raw()
        | vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
);

/// The stages that name every command's.
const EVERY_STAGE: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::from_raw(
    vk::PipelineStageFlags2::ALL_COMMANDS.as_raw() | vk::PipelineStageFlags2::ALL_GRAPHICS.as_raw(),
);

/// The write access types of commands in `stages`: storage writes in shader stages, transfer
/// writes in transfer stages, attachment writes in a pass's attachment stages, and every write
/// where `stages` names every command's.
fn writes_in(stages: vk::PipelineStageFlags2) -> vk::AccessFlags2 {
    let mut access = vk::AccessFlags2::NONE;
    if stages.intersects(EVERY_STAGE) {
        access |= vk::AccessFlags2::MEMORY_WRITE;
    }
    if stages.intersects(SHADER_STAGES) {
        access |= vk::AccessFlags2::SHADER_STORAGE_WRITE;
    }
    if stages.intersects(TRANSFER_STAGES) {
        access |= vk::AccessFlags2::TRANSFER_WRITE;
    }
    if stages.intersects(ATTACHMENT_STAGES) {
        access |= vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
    }
    access
}

/// What a command in `stages` may reach of memory none of its uses names, which a fence orders
/// (**Fences** in the module docs): storage and sampled reads and storage writes in shader
/// stages, transfer reads and writes in transfer stages, attachment reads and writes in a pass's
/// attachment stages, and every access where `stages` names every command's. An indirect
/// argument, uniform, vertex or index read comes only from a use that names it.
fn unnamed_access(stages: vk::PipelineStageFlags2) -> Access {
    let mut access = vk::AccessFlags2::NONE;
    if stages.intersects(EVERY_STAGE) {
        access |= vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE;
    }
    if stages.intersects(SHADER_STAGES) {
        access |= vk::AccessFlags2::SHADER_STORAGE_READ
            | vk::AccessFlags2::SHADER_SAMPLED_READ
            | vk::AccessFlags2::SHADER_STORAGE_WRITE;
    }
    if stages.intersects(TRANSFER_STAGES) {
        access |= vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::TRANSFER_WRITE;
    }
    if stages.intersects(ATTACHMENT_STAGES) {
        access |= vk::AccessFlags2::COLOR_ATTACHMENT_READ
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
    }
    let named = SHADER_STAGES | TRANSFER_STAGES | ATTACHMENT_STAGES | EVERY_STAGE;
    match access.is_empty() {
        true => Access::NONE,
        false => Access::new(stages & named, access),
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
/// either way (the module docs, **Render encoders**). One whose first instance may record inline
/// (`continues`) may continue the render pass instance a group's tail holds open, as it decides
/// later (**Continued passes** in the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Joining {
    Compute(DispatchType),
    Blit,
    Render { joins: bool, continues: bool },
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

/// Why an encoder that asked for a group joined none (**Refusals** in the module docs): counted
/// as the encoder opens (`WorkCounts::folds_refused_kind` and kin).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    /// The group's encoders are of another kind: a compute encoder of another dispatch type than
    /// the group's compute encoders, or a render encoder that may not join.
    Kind,
    /// The group's tail is no longer the command buffer's last `VkCommandBuffer`.
    Tail,
    /// A batch boundary since the group's last encoder: an event's wait or signal, a host step,
    /// a resource state encoder, or a signal pending.
    Batch,
    /// The command buffer used command predication, its protection options are not 0, or the
    /// encoder trail is on.
    State,
    /// The encoder before left work for later (a segment) or went in more than one part.
    Parts,
    /// The encoder before fails or faults its command buffer, a command-predication record left
    /// its work out, or its batch's submission records after it or judges by it (a write into an
    /// ASTC carrier, an `Overwrite`).
    Ends,
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
    /// The render pass instance its last encoder left open in the tail, which the next render
    /// encoder may continue (**Continued passes** in the module docs); ended before anything else
    /// records into the tail.
    pub(crate) open: Option<OpenPass>,
}

impl Fold {
    /// Whether an encoder that asks `joining` may join it, by what it holds.
    pub(crate) fn admits(&self, joining: Joining) -> bool {
        match joining {
            Joining::Compute(dispatch_type) => self
                .dispatch_type
                .is_none_or(|group| group == dispatch_type),
            Joining::Blit => true,
            Joining::Render { joins, .. } => joins,
        }
    }

    /// The group `joining`'s encoder begins, its first part `leader`, its first `VkCommandBuffer`
    /// the tail; its order is built as a second encoder joins.
    pub(crate) fn begun(
        tail: vk::CommandBuffer,
        (leader, encoder): (usize, usize),
        (batches, joining): (usize, Joining),
        (stashed, open): (Vec<Stashed>, Option<OpenPass>),
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
            open,
        }
    }

    /// End the render pass instance the tail holds open, where it holds one: before anything
    /// else records into the tail, and before the tail ends.
    pub(crate) fn end_open(&mut self) {
        if let Some(open) = self.open.take() {
            open.end();
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
    /// What its one part leaves that needs a `VkCommandBuffer` boundary after it, so no group
    /// goes on past it: work for later or more than one part ([`Refusal::Parts`]), or something
    /// that fails or faults its command buffer, work a command-predication record left out, a
    /// write into an ASTC carrier or an `Overwrite` ([`Refusal::Ends`]); `None` where it leaves
    /// nothing.
    pub(crate) ends: Option<Refusal>,
    /// What that part used, where it leaves nothing (`ends` is `None`) and joined a group, whose
    /// next encoders order
    /// after it; `None` else (the first part of a group it begins holds its summary, which the
    /// group's second encoder orders after: `State::join_fold`).
    pub(crate) summary: Option<Summary<ResourceKey>>,
    /// Its log, which it did not keep, with its executions' place.
    pub(crate) log: Option<(Replay, Range<usize>)>,
    /// The render pass instance it left open in its last `VkCommandBuffer`, where it left one
    /// (**Continued passes** in the module docs): the group it begins or goes on in keeps it;
    /// else it ends here.
    pub(crate) open: Option<OpenPass>,
}

/// A render pass instance a render encoder left open at its end in its group's tail, for the
/// next render encoder to continue (**Continued passes** in the module docs).
pub(crate) struct OpenPass {
    device: Device,
    /// The tail, recording inside the instance.
    raw: vk::CommandBuffer,
    /// What the instance renders.
    key: Arc<PassKey>,
}

impl OpenPass {
    /// The instance `raw` records inside, which renders what `key` says.
    pub(crate) fn new(device: &Device, raw: vk::CommandBuffer, key: Arc<PassKey>) -> OpenPass {
        OpenPass {
            device: device.clone(),
            raw,
            key,
        }
    }

    /// What it renders.
    pub(crate) fn key(&self) -> &Arc<PassKey> {
        &self.key
    }

    /// The tail it records inside.
    pub(crate) fn raw(&self) -> vk::CommandBuffer {
        self.raw
    }

    /// End it: `vkCmdEndRenderPass` into the tail.
    pub(crate) fn end(self) {
        // SAFETY: the tail is recording inside the instance, and only the caller records it now
        // (the command buffer's state lock, or the encoder that holds its group).
        unsafe { self.device.shared().vk().cmd_end_render_pass(self.raw) };
    }
}
