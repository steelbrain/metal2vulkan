//! The core every encoder shares: its command buffer, its own `VkCommandBuffer`, and the rule of
//! one open encoder per command buffer.
//!
//! - **Recording.** `open` allocates a primary `VkCommandBuffer` from the command buffer's pool
//!   and begins it; `end` ends it and appends it to the command buffer's recordings, which
//!   `commit` submits in that order.
//! - **Segments**. An indirect command buffer's commands may change until the command buffer is
//!   committed, so an encoder that executes them [`defer`](EncoderCore::defer)s them: it ends its
//!   `VkCommandBuffer` there and goes on in a new one, and the segment's own is recorded later
//!   ([`EncoderCore::segment`]), behind a barrier from everything before it and with one to
//!   everything after it, and goes between the two as its batch is submitted, its uses joined to
//!   its part's summary. `commit` records it from the records as they stand; a buffer the GPU may
//!   write (`IndirectCommandBuffer::gpu_written`) has its commands known only once the work before
//!   them completed, so its segment starts a part of the encoder of its own, in a batch of its own,
//!   which the scheduler's host step records (`indirect_execute.rs`).
//! - **Ordering** ([`tracker`](super::tracker)). Each command goes through [`EncoderCore::command`]
//!   with its uses: in an encoder whose commands Metal runs in order (blits, serial dispatches) it
//!   gets the barrier those uses need after the earlier commands. `end` hands the encoder's summary
//!   to the command buffer; the scheduler records the barriers between encoders, batches and queues
//!   as it submits each batch.
//! - **Shadowed textures** (`shadow.rs`). The encoder copies a 2D texture or array over a buffer,
//!   or a texture buffer viewed through a staging buffer, from the buffer before the commands
//!   that use it, and back after the ones that write it, as commands of its own; `end` copies
//!   back what is left.
//! - **Abandoned encoders.** An encoder dropped without `end_encoding` marks its command buffer
//!   abandoned: its commands are lost, so `commit` refuses rather than run without them.
//! - Encoders are `Send` but not `Sync`: every method takes `&mut self`, and the core holds a
//!   `!Sync` marker so no encoder is shared between threads by reference.

use super::binding::{self, LastSetZero, Write, WrittenSet};
use super::buffer::Buffer;
use super::churn_counts::KeepBy;
use super::command_buffer::rebound_use::Unlogged;
use super::command_buffer::{recording_error, CommandBuffer};
use super::deferred::{Deferred, StepInputs};
use super::dirty_pages::CopyInGate;
use super::error::{internal, invalid, vulkan, CommandBufferError, Error, Problem};
use super::execution::Execution;
use super::fresh_unwritten::FreshRead;
use super::heap::Heap;
use super::index_bounds::{BoundsRunRequest, BoundsSlot};
use super::index_rewrite::{
    self, CommandedList, GatherHeld, GatherRequest, GatherRuns, GatheredList,
    IndirectGatherRequest, RewriteRunRequest,
};
use super::noted_apart::{NotedApart, NotedInto};
use super::pending_dispatch::PendingDispatch;
use super::pipeline::VariantObjects;
use super::rebind::{Holds, Taken};
use super::replay::Replay;
use super::residency::{Residency, Used};
use super::resource::ResourceKey;
use super::retained::Retained;
use super::row_copies::RowCopies;
use super::shadow::{CopyCause, GatedCopyIn, ShadowCopy, ShadowSync, ShadowUse, Step, Within};
use super::shadow_copy_in::CopyIn;
use super::shadow_merge::ShadowMerge;
use super::sparse::encoder::SparseUpdate;
use super::texture::decoded_twin::TwinWrite;
use super::texture::kept_image::KeptLease;
use super::tracker::{Barrier, EncoderUses, Reach, Summary, Use, UseRange};
use super::unique_identifier::ScheduledWork;
use super::used_resources::EncoderKind;
use super::Device;
use crate::device::functions::Vk;
use crate::work_counts::Begun;
use ash::vk;
use metal2vulkan::bind::ReplayFaults;
use std::any::Any;
use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::panic::Location;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The next core's [`EncoderCore::id`]: from 1, so none is `retained::HeldBy`'s 0.
static CORES: AtomicU64 = AtomicU64::new(1);

/// One ended encoder's recording, or one part of it (`EncoderUses::split`), in its command
/// buffer's encoding order.
pub(super) struct Recorded {
    /// The encoder's place among its command buffer's ended encoders (`State::encoder_ended`),
    /// which each of its parts carries.
    pub(super) encoder: usize,
    /// Its `VkCommandBuffer`s, in order; once its batch is submitted, each segment's among them.
    pub(super) raws: Vec<vk::CommandBuffer>,
    /// Its segments, in order, until its batch is submitted.
    pub(super) segments: Vec<Segment>,
    /// What the encoder used, for the barriers its submission records; each segment's uses join
    /// it as its batch is submitted.
    pub(super) summary: Summary<ResourceKey>,
    /// A resource state encoder's sparse mapping updates, which its batch binds before its work
    /// (`sparse/`); none for any other encoder.
    pub(super) sparse: Vec<SparseUpdate>,
    /// A blit encoder's writes into ASTC carriers, which its batch's submission decodes into
    /// their second decoded twins after its work (`texture/decoded_twin.rs`); none for any other
    /// encoder.
    pub(super) twin_writes: Vec<TwinWrite>,
}

/// Commands an encoder left to record later (`deferred.rs`).
pub(super) struct Segment {
    /// Where its `VkCommandBuffer` goes: after this many of its part's own.
    pub(super) at: usize,
    /// Where its dispatches go among the command buffer's executions: after this many, the
    /// segments' placed before it not counted.
    pub(super) executions_at: usize,
    pub(super) work: Deferred,
    /// The GPU may write what the work reads (`Deferred::gpu_written` at the call): the segment
    /// starts a part of its own, and the host step records it.
    pub(super) host: bool,
    /// What `commit` left for its host step: one it foresaw, or one it did not (what the work
    /// reads written by the GPU only since).
    pub(super) inputs: Option<StepInputs>,
    /// Its recording until its batch is submitted: `commit`'s, or the host step's.
    pub(super) recorded: Option<RecordedSegment>,
    /// The host step recorded it.
    pub(super) stepped: bool,
    /// `commit` left its recording to the host step for the pipelines it records, which were
    /// being made then (`render/pending.rs`).
    pub(super) pipelines: bool,
    /// `commit` recorded it on the GPU alone, for good (`Prepared::for_good`), and no host step
    /// recorded it since: its resident set and what its commands named were resolved then, so a
    /// rebind before the command buffer's first submission records it again
    /// (`command_buffer/reencode.rs`).
    pub(super) gpu_alone: bool,
}

impl Segment {
    /// Whether the host step records it before its batch is submitted: the GPU may write what
    /// its work reads, whether it could at the call or only since (`Deferred::gpu_written` is
    /// for good), or `commit` left it for the pipelines it records, or recorded it from records
    /// the CPU wrote since (`Deferred::rewritten`); or `commit`'s recording on the
    /// GPU alone must be made again ([`rerecords`](Self::rerecords)); or `commit` recorded it on
    /// the GPU for good, and a route encoded since may write more than it decided on, or gives an
    /// execution's check something to see (`Deferred::outgrown`); or its step left work its batch
    /// waits for (`Deferred::holds`).
    pub(super) fn needs_step(&self) -> bool {
        (!self.stepped
            && (self.host || self.pipelines || self.work.gpu_written() || self.work.rewritten()))
            || self.rerecords()
            || self.work.outgrown()
            || self.work.holds()
    }

    /// Whether `commit` recorded it on the GPU alone with no barrier between its commands and its
    /// batch's submission records it again with them, waiting for no work on the GPU
    /// (`gpu_execute::Rerecord`): a route encoded since may set a barrier in a record.
    pub(super) fn rerecords(&self) -> bool {
        self.inputs.as_ref().is_some_and(StepInputs::rerecords)
    }

    /// Whether its host step waits for the work submitted before it to complete: the GPU may
    /// write what its work reads. A step for its pipelines alone waits for them alone.
    pub(super) fn steps_after_the_gpu(&self) -> bool {
        (!self.stepped && (self.host || self.work.gpu_written())) || self.work.outgrown()
    }

    /// Whether its host step waits for a pipeline still being made, or its batch for what its
    /// step left (`Deferred::holds`), which wakes the scheduler as it ends.
    pub(super) fn steps_after_pipelines(&self) -> bool {
        (!self.stepped && self.work.making()) || self.work.holds()
    }

    /// [`steps_after_pipelines`](Self::steps_after_pipelines) once its command buffer's bound
    /// passed (`CommandBuffer::making_deadline`): its step records a draw or dispatch whose
    /// pipeline is still being made as nothing, so its batch waits only for what its step left
    /// (`Deferred::holds`).
    pub(super) fn holds(&self) -> bool {
        self.work.holds()
    }

    /// Whether it is a dispatch whose pipelines were being made (`pending_dispatch.rs`): an
    /// encoder encoded again (`replay.rs`) leaves one of its own in its place, prepared as the
    /// commit prepared this one, and waits for no making.
    pub(super) fn pends(&self) -> bool {
        matches!(self.work, Deferred::Pending(_))
    }

    /// Whether an encoder encoded again (`replay.rs`) leaves it as a segment of its own, prepared
    /// as the commit prepared the first encoding's ([`Deferred::prepare`]), rather than recording
    /// it on the GPU alone: a dispatch whose pipelines were being made ([`pends`](Self::pends)),
    /// and a render pass of a draw's or a tile dispatch's pipeline still being made, or whose
    /// making failed (`render/pending.rs`, `render/pending_tile.rs`), which recording on the GPU
    /// alone would wait for, and make itself, on the program's thread
    /// (`DeferredPass::record_on_gpu`).
    pub(super) fn keeps_pipelines(&self) -> bool {
        self.pends() || self.work.making() || self.work.failed()
    }

    /// Whether an encoder encoded again (`replay.rs`) records its place: a dispatch whose
    /// pipelines were being made ([`pends`](Self::pends)), left as a segment again, or work a
    /// commit may record on the GPU alone, recorded there again (`Deferred::record_on_gpu`). Its
    /// log is kept only where every segment does.
    pub(super) fn records_again(&self) -> bool {
        self.pends() || self.work.may_run_on_gpu_alone()
    }

    /// What its recording will use, on `device`, while `commit` left it to the host step for the
    /// pipelines it records alone (and not for what the GPU may write): its work's uses, which
    /// the encoder knew, only the `VkPipeline` missing (`Deferred::known_summary`). `None` where
    /// the step or the submission finds them.
    pub(super) fn known_summary(&self, device: &Device) -> Option<Summary<ResourceKey>> {
        match self.pipelines && !self.steps_after_the_gpu() {
            true => self.work.known_summary(device),
            false => None,
        }
    }
}

/// The `VkCommandBuffer` an encoder goes on in after a segment, begun
/// ([`EncoderCore::continuation`]), and whether it goes on inside a predicated loop's laying.
pub(super) struct Continuation(vk::CommandBuffer, Laid);

/// Whether a [`Continuation`] goes on inside a predicated loop's laying (`predication.rs`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Laid {
    Outside,
    Inside,
}

/// A segment recorded: its `VkCommandBuffer`s, ended, in order, and what it used and ran. One,
/// but for a render pass that executes render commands whose first instance recorded inline
/// (`render/executions.rs`): what the pass records before itself, the first instance's own, then
/// the rest ([`cut_before`](EncoderCore::cut_before)).
pub(super) struct RecordedSegment {
    pub(super) raws: Vec<vk::CommandBuffer>,
    pub(super) summary: Summary<ResourceKey>,
    pub(super) retained: Retained,
    pub(super) executions: Vec<Execution>,
    /// The keys its commands' uses named (`used_resources.rs`).
    pub(super) keys: BTreeSet<ResourceKey>,
    /// What its commands took of buffers over host pages, and which they named (`rebind.rs`).
    pub(super) taken: Taken,
    /// The resident-set misses its recording found (`indirect_execute.rs`), which the command
    /// buffer's execution report counts.
    pub(super) resident_misses: u32,
    /// Why some of its commands ran nothing: records the host step could not run
    /// (`indirect_execute.rs`), which the execution report states.
    pub(super) faults: ReplayFaults,
}

/// The state every open encoder carries.
pub(super) struct EncoderCore {
    open: OpenGuard,
    /// The `VkCommandBuffer` its commands record into; null while a pass holds the one before
    /// (`pass`) and no command came since, which begins another ([`recording`](Self::recording)).
    raw: vk::CommandBuffer,
    /// Something recorded into `raw` that a command recorded before a pass must follow: any
    /// command of the encoder's, not the first encoder's start (`gpu_time.rs`), which a pass may
    /// hold the command buffer of ([`hold_pass`](Self::hold_pass)).
    recorded: bool,
    /// The `VkCommandBuffer` a render pass instance records its draws inline in, from the pass's
    /// first draw to its end (`render.rs`, [`hold_pass`](Self::hold_pass)): the encoder's
    /// commands meanwhile record into `raw`, which goes before it in submission order, as they go
    /// before a pass recorded at its end.
    pass: Option<vk::CommandBuffer>,
    /// The encoder's `VkCommandBuffer`s ended before a segment, in order.
    ended: Vec<vk::CommandBuffer>,
    /// The segments it left for commit, `executions_at` counting its own executions.
    segments: Vec<Segment>,
    /// Its first parts, which a command-predication record left out of the command buffer's
    /// work ([`leave_out_recorded`](Self::leave_out_recorded)): each ends running nothing.
    left_out: usize,
    /// The order of the encoder's commands, and its summary.
    uses: EncoderUses<ResourceKey>,
    /// The uses of the command [`note_reaching`](Self::note_reaching) notes, with those of the
    /// host memory beside them ([`aliased`](Self::aliased)): kept, so a draw makes no vector.
    aliased_uses: Vec<Use<ResourceKey>>,
    /// The open runs of the index gather pass ([`gather`](Self::gather)), which record before
    /// the encoder's next command, and before its `VkCommandBuffer` ends.
    gathers: GatherRuns,
    /// What the encoder's commands use; `end` hands it to the command buffer.
    retained: Retained,
    /// What its dispatches and draws executed; `end` hands it to the command buffer.
    executions: Vec<Execution>,
    /// Where the shadowed textures the encoder used stand.
    shadows: ShadowSync,
    /// The keys its commands' uses named, and its bindings' (`used_resources.rs`): `end` finds the
    /// heaps among them, and hands them to the command buffer.
    keys: BTreeSet<ResourceKey>,
    /// The key of each use the last command [`touch_reaching`](Self::touch_reaching) touched took,
    /// in order, and whether it named it (`Taken::name`): both only grow while the encoder is
    /// open, so a use at that place of the same key, named only where that one was, is kept and
    /// named already.
    touched: Vec<(ResourceKey, bool)>,
    /// The buffers the encoder made resident since its last command (`made_resident`), which its
    /// next command holds.
    resident_unheld: Vec<ResourceKey>,
    /// The shadowed textures its pass took as attachments (`render/shadows.rs`): `end` notes them
    /// in the command buffer (`WorkCounts::shadow_attachment_passes`).
    attachment_shadows: Vec<ResourceKey>,
    /// Its first copies in of shadowed textures, each copy's texture and texels: `end` notes them
    /// in the command buffer (`WorkCounts::shadow_texels_first_copied_in_again`).
    first_copies: Vec<(ResourceKey, u64)>,
    /// Its holds on the kept images its merges took from its queue (`texture/kept_image.rs`):
    /// no keep of another encoder takes one before this one ends, its merges done.
    kept_leases: Vec<KeptLease>,
    /// The heaps its commands used through what the encoder made resident; `end` hands them,
    /// and those of `keys`, to the command buffer, whose commit marks them used.
    heaps: Vec<Heap>,
    /// Its holds on the buffers over host pages its commands used (`rebind.rs`): `end` hands
    /// them to the command buffer, whose commit lets go of them.
    holds: Holds,
    /// The pages its commands took of those buffers, and the keys a command named: `end` hands
    /// them to the command buffer, which holds the pages until it completes (weakly from its
    /// commit to its first submission).
    taken: Taken,
    /// Metal fails the command buffer from this encoder on with this error (a render pass
    /// without attachments past 16384 a side, or at a default sample count Metal fails it at):
    /// `end` says so to the command buffer, whose commit runs nothing from it on
    /// (`State::fails_from`).
    fails_command_buffer: Option<CommandBufferError>,
    /// Metal's GPU faults this encoder, with this error (an execution of commands that inherit a
    /// pipeline state made without `supportIndirectCommandBuffers`, an indexed or indirect draw
    /// of primitive type 9): `end` says so to the command buffer, whose commit runs nothing of it
    /// (`State::fault`).
    faults: Option<CommandBufferError>,
    /// Its writes into ASTC carriers ([`wrote_twin`](Self::wrote_twin)): `end` hands them to the
    /// command buffer with its last part.
    twin_writes: Vec<TwinWrite>,
    /// The commands it records now were called while no buffer of the device was detached: a
    /// render encoder lowers calls it logged (`render/deferral.rs`), whose detachment they judge
    /// as of their calls ([`judges_unattached`](Self::judges_unattached)).
    unattached_as_called: bool,
    /// The core's own number in the process ([`id`](Self::id)).
    id: u64,
    /// It encodes an ended encoder's calls again, its command buffer committed (`replay.rs`), or
    /// records again work a commit recorded (`command_buffer/reencode.rs`): its commands take
    /// what they use without holding it (`Rebinds::pin`).
    replaying: bool,
    /// The set 0 its last command allocated and wrote, which a later command that writes the same
    /// into a set of the same layout binds again ([`set_zero`](Self::set_zero)).
    last_set_zero: LastSetZero,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

/// An encoder encoded again (`EncoderCore::replaying`), ended: what its place among its command
/// buffer's recordings takes in place of what the encoder recorded (`State::splice_replay`).
pub(super) struct Replayed {
    /// Its parts, as its first encoding's: a segment the GPU may write what it reads for starts
    /// one of its own.
    pub(super) parts: Vec<ReplayedPart>,
    /// The images it holds in their attachment layout (`texture/held_depth.rs`).
    pub(super) held: Vec<ResourceKey>,
    pub(super) retained: Retained,
    pub(super) executions: Vec<Execution>,
    pub(super) keys: BTreeSet<ResourceKey>,
    pub(super) taken: Taken,
    pub(super) heaps: Vec<Heap>,
}

/// One part of an encoder encoded again ([`Replayed::parts`]): its `VkCommandBuffer`s, what they
/// use, and its segments, each recorded at once on the GPU alone, its place among the
/// executions counted from the encoder's first.
pub(super) struct ReplayedPart {
    pub(super) raws: Vec<vk::CommandBuffer>,
    pub(super) summary: Summary<ResourceKey>,
    pub(super) segments: Vec<Segment>,
}

/// An ended encoder's `VkCommandBuffer`s `raws`, in order, and its `segments`, in parts: a segment
/// the GPU may write what it reads for starts a part of its own (the module's segments), each
/// segment's place counted among its part's `VkCommandBuffer`s, and its place among the
/// executions from `base`, the encoder's first.
fn into_parts(
    raws: Vec<vk::CommandBuffer>,
    segments: Vec<Segment>,
    base: usize,
) -> Vec<(Vec<vk::CommandBuffer>, Vec<Segment>)> {
    let mut raws = raws.into_iter();
    // The raws the parts before the current one took.
    let mut taken = 0;
    let mut parts = Vec::new();
    let mut current = Vec::new();
    for mut segment in segments {
        segment.executions_at += base;
        if segment.host {
            let before = segment.at - taken;
            parts.push((raws.by_ref().take(before).collect::<Vec<_>>(), current));
            current = Vec::new();
            taken = segment.at;
        }
        segment.at -= taken;
        current.push(segment);
    }
    parts.push((raws.collect(), current));
    parts
}

/// The open encoder's hold on its command buffer: dropped before `end` ran, it marks the command
/// buffer abandoned. It is apart from [`EncoderCore`] so `end` can take the core apart, and the
/// resource state encoder holds one too.
pub(super) struct OpenGuard {
    pub(super) command_buffer: CommandBuffer,
    /// `end` ran: the drop does not abandon the command buffer.
    pub(super) ended: bool,
}

impl Drop for OpenGuard {
    fn drop(&mut self) {
        if !self.ended {
            let mut state = self.command_buffer.state();
            state.encoder_open = false;
            state.abandoned = true;
        }
    }
}

/// Refuse an object of `other` in Metal's `call` of an encoder of `device` unless `other` is
/// `device`: an encoder's check ([`EncoderCore::check_device`]), and a render encoder's at the call
/// (`render/call_checks.rs`).
pub(super) fn check_device(
    device: &Device,
    call: &'static str,
    other: &Device,
) -> Result<(), Error> {
    match device.same(other) {
        true => Ok(()),
        false => Err(invalid(call, Problem::ForeignObject)),
    }
}

impl EncoderCore {
    /// Open an encoder on `command_buffer` for Metal's `call`, `ordered` when Metal runs its
    /// commands in encoded order (a blit or serial compute encoder). Refused when the command
    /// buffer is committed, has an open encoder, or has an abandoned one.
    pub(super) fn open(
        command_buffer: &CommandBuffer,
        call: &'static str,
        kind: EncoderKind,
        ordered: bool,
    ) -> Result<EncoderCore, Error> {
        // The last render encoder's end, where its call left it to the device's lowering thread.
        command_buffer.settle_end();
        let mut state = command_buffer.state();
        if let Some(problem) = state.refusal() {
            return Err(invalid(call, problem));
        }
        // The pool's primary begun ahead where it holds one (`spares.rs`).
        let (raw, begun) = command_buffer.encoder_primary(&mut state)?;
        let shared = command_buffer.device().shared();
        if !begun {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            // SAFETY: the command buffer was just handed out of the pool, under the state lock.
            unsafe { shared.vk().begin_command_buffer(raw, &begin) }.map_err(vulkan)?;
        }
        shared.counts.encoder_command_buffer_begun(Begun::Encoder);
        if begun {
            shared.counts.encoder_primary_begun_ahead();
        }
        // The first encoder of a timed command buffer writes its start first (`gpu_time.rs`).
        command_buffer.open_start(&mut state, raw);
        state.encoder_open = true;
        // Made under the protection options set now (`command_buffer/protection.rs`).
        state.protected_made();
        // What the command buffer's earlier encoders left shadowed textures' images holding,
        // which this one's first uses of them take instead of copying the truth in again
        // (`shadow::ShadowCarry`).
        let carry = state.shadows_to_carry();
        drop(state);
        command_buffer
            .device()
            .shared()
            .identifiers
            .encoder_opened(kind);
        Ok(EncoderCore {
            open: OpenGuard {
                command_buffer: command_buffer.clone(),
                ended: false,
            },
            raw,
            recorded: false,
            pass: None,
            ended: Vec::new(),
            segments: Vec::new(),
            left_out: 0,
            uses: EncoderUses::new(ordered),
            retained: Retained::default(),
            executions: Vec::new(),
            shadows: ShadowSync::carrying(carry),
            keys: BTreeSet::new(),
            touched: Vec::new(),
            resident_unheld: Vec::new(),
            attachment_shadows: Vec::new(),
            first_copies: Vec::new(),
            kept_leases: Vec::new(),
            heaps: Vec::new(),
            holds: Holds::new(command_buffer.device()),
            taken: Taken::default(),
            fails_command_buffer: None,
            faults: None,
            twin_writes: Vec::new(),
            unattached_as_called: false,
            aliased_uses: Vec::new(),
            gathers: GatherRuns::default(),
            id: CORES.fetch_add(1, Ordering::Relaxed),
            replaying: false,
            last_set_zero: LastSetZero::default(),
            _not_sync: PhantomData,
        })
    }

    /// The core a segment of `command_buffer` records into at its commit, `ordered` as the
    /// encoder that left it: a new `VkCommandBuffer` of its pool, begun, its first command a
    /// barrier from everything recorded before. Recorded, it ends with
    /// [`end_segment`](Self::end_segment); dropped, it abandons nothing.
    pub(super) fn segment(
        command_buffer: &CommandBuffer,
        ordered: bool,
    ) -> Result<EncoderCore, Error> {
        let mut core = EncoderCore::segment_raw(command_buffer, ordered)?;
        // SAFETY: the command buffer is recording, and only this core records it.
        unsafe { Barrier::everything().record(core.device().shared().vk(), core.raw) };
        core.recorded = true;
        Ok(core)
    }

    /// The core an ended encoder of `command_buffer`, committed and not yet first submitted, is
    /// encoded again into (`replay.rs`), `ordered` as that encoder: a new `VkCommandBuffer` of
    /// its pool, begun, as [`open`](Self::open) begins one, without its refusals. Ended with
    /// [`finish_replay`](Self::finish_replay); dropped, it abandons nothing.
    pub(super) fn replaying(
        command_buffer: &CommandBuffer,
        ordered: bool,
    ) -> Result<EncoderCore, Error> {
        let mut core = EncoderCore::segment_raw(command_buffer, ordered)?;
        core.replaying = true;
        Ok(core)
    }

    /// A core of `command_buffer` over a new `VkCommandBuffer` of its pool, begun, that abandons
    /// nothing when dropped.
    fn segment_raw(command_buffer: &CommandBuffer, ordered: bool) -> Result<EncoderCore, Error> {
        let raw = {
            let mut state = command_buffer.state();
            command_buffer.allocate_raw(&mut state)?
        };
        let device = command_buffer.device().shared().vk();
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer was just allocated from the pool, and only this core
        // records it.
        unsafe { device.begin_command_buffer(raw, &begin) }.map_err(vulkan)?;
        command_buffer
            .device()
            .shared()
            .counts
            .encoder_command_buffer_begun(Begun::Segment);
        Ok(EncoderCore {
            open: OpenGuard {
                command_buffer: command_buffer.clone(),
                ended: true,
            },
            raw,
            recorded: false,
            pass: None,
            ended: Vec::new(),
            segments: Vec::new(),
            left_out: 0,
            uses: EncoderUses::new(ordered),
            retained: Retained::default(),
            executions: Vec::new(),
            shadows: ShadowSync::default(),
            keys: BTreeSet::new(),
            touched: Vec::new(),
            resident_unheld: Vec::new(),
            attachment_shadows: Vec::new(),
            first_copies: Vec::new(),
            kept_leases: Vec::new(),
            heaps: Vec::new(),
            holds: Holds::new(command_buffer.device()),
            taken: Taken::default(),
            fails_command_buffer: None,
            faults: None,
            twin_writes: Vec::new(),
            unattached_as_called: false,
            aliased_uses: Vec::new(),
            gathers: GatherRuns::default(),
            id: CORES.fetch_add(1, Ordering::Relaxed),
            // A segment a rebind records again, its command buffer committed.
            replaying: crate::replay::is_replaying(),
            last_set_zero: LastSetZero::default(),
            _not_sync: PhantomData,
        })
    }

    /// The device.
    pub(super) fn device(&self) -> &Device {
        self.open.command_buffer.device()
    }

    /// Set 0 of a command of `objects`'s variant that writes `writes` (set 0's), and `offsets`, the
    /// dynamic offsets the command binds it with: the writes of the bindings the variant makes
    /// dynamic made so first (`binding::dynamic_writes`), each at offset 0 and its offset among
    /// `offsets`, so commands that differ in those offsets alone write the same set. The set the
    /// encoder's last command allocated, where it is of the same layout and holds the same writes
    /// (`binding::LastSetZero`), else such a set an earlier command of the command buffer
    /// allocated (`binding::SetCache`), else one allocated from the command buffer's pools and
    /// written, which the caller keeps for later commands once the command records
    /// ([`keep_set_zero`](Self::keep_set_zero)). `None` where the variant allocates none
    /// (`VariantObjects::set_zero`), its writes then refused as `binding::write_sets` refuses
    /// them.
    pub(super) fn set_zero(
        &mut self,
        objects: &VariantObjects,
        (writes, offsets): (&mut [Write], &mut Vec<u32>),
    ) -> Result<(Option<vk::DescriptorSet>, Option<WrittenSet>), Error> {
        binding::dynamic_writes(writes, objects.dynamic_bindings(), offsets);
        let layout = objects.set_zero_layout();
        if let Some(layout) = layout {
            if let Some(set) = self.last_set_zero.holds(layout, writes) {
                self.device().shared().counts.set_zero_set_reused();
                return Ok((Some(set), None));
            }
            let found = (self.command_buffer().state().descriptors).find(layout, writes);
            if let Some(set) = found {
                self.device().shared().counts.set_zero_set_found();
                self.last_set_zero.keep(layout, set, writes);
                return Ok((Some(set), None));
            }
        }
        let set = objects.set_zero(&mut self.command_buffer().state().descriptors)?;
        // SAFETY: the set was just allocated and no command uses it yet; every buffer, view and
        // sampler written is live (the command buffer retains them, and a refusal leaves the set
        // unused in its pool).
        unsafe { binding::write_sets(self.device(), set.as_slice(), writes) }?;
        let written = match (layout, set) {
            (Some(layout), Some(set)) => Some(WrittenSet { layout, set }),
            _ => None,
        };
        Ok((set, written))
    }

    /// Keep `written`, the set 0 a command that records allocated and wrote with `writes`
    /// ([`set_zero`](Self::set_zero)), for the encoder's next command (`binding::LastSetZero`)
    /// and the command buffer's later ones (`binding::SetCache`): kept only once the command
    /// records, so every object its writes name is held until the command buffer completes.
    pub(super) fn keep_set_zero(&mut self, written: WrittenSet, writes: &[Write]) {
        self.last_set_zero.keep(written.layout, written.set, writes);
        (self.command_buffer().state().descriptors).keep(written.layout, written.set, writes);
    }

    /// The queue flags of the family its command buffer records for: its queue's `VkQueue`'s.
    pub(super) fn queue_flags(&self) -> vk::QueueFlags {
        let queue = self.open.command_buffer.command_queue().vk_queue();
        self.device().shared().queue_flags_of(queue)
    }

    /// The core's own number, which no other core of the process has: what an argument table
    /// retained into it holds until it changes (`ArgumentTable::retain`).
    pub(super) fn id(&self) -> u64 {
        self.id
    }

    /// The command buffer the encoder records into.
    pub(super) fn command_buffer(&self) -> &CommandBuffer {
        &self.open.command_buffer
    }

    /// Refuse an object of `device` in Metal's `call` unless `device` is the encoder's.
    pub(super) fn check_device(&self, call: &'static str, device: &Device) -> Result<(), Error> {
        check_device(self.device(), call, device)
    }

    /// Hold `object` until the command buffer completes: the encoder's commands use it.
    pub(super) fn retain(&mut self, object: impl Any + Send + Sync) {
        self.retained.boxed(Box::new(object));
    }

    /// A command of the encoder wrote `write` into an ASTC carrier's blocks: its batch's
    /// submission decodes it into the image's second decoded twin (`texture/decoded_twin.rs`).
    pub(super) fn wrote_twin(&mut self, write: TwinWrite) {
        self.twin_writes.push(write);
    }

    /// Hold `object`, a shared one (`Buffer::held` and its kin), until the command buffer
    /// completes: by another reference to it, without a box of its own (`retained.rs`).
    pub(super) fn retain_shared(&mut self, object: Arc<dyn Any + Send + Sync>) {
        self.retained.shared(object);
    }

    /// `buffer`'s `VkBuffer`, and where its byte `offset` lies in it, for a blit's command
    /// ([`Buffer::blit_at`]): a no-copy buffer's import the device released while idle is
    /// imported again first, and what the command reads of it held until the command buffer
    /// completes (`host_residency.rs`). Refused where the device cannot import it again.
    ///
    /// The buffer itself is held until the command buffer completes too, whatever path records
    /// the command (a per-sample copy's plan holds only the raw `VkBuffer`): its `VkBuffer`, and a
    /// no-copy buffer's import of the caller's pages, go only once nothing holds the buffer
    /// (`HostImport`'s drop), so a program that lets go of it once it committed (as Metal lets it,
    /// a command buffer retaining what its commands use) frees nothing a command in flight writes.
    pub(super) fn blit_at(
        &mut self,
        buffer: &Buffer,
        offset: u64,
    ) -> Result<(vk::Buffer, u64), Error> {
        let (at, hold) = buffer.blit_at(offset)?;
        if let Some(hold) = hold {
            self.taken.hold_import(hold);
        }
        self.retain_shared(buffer.held());
        Ok(at)
    }

    /// Add a dispatch or draw the encoder took to the command buffer's execution record, as
    /// `execution` builds it. Only where the device keeps records
    /// ([`records_executions`](Self::records_executions)): elsewhere nothing is built.
    pub(super) fn executed(&mut self, execution: impl FnOnce() -> Execution) {
        if self.records_executions() {
            self.executions.push(execution());
        }
    }

    /// Whether the device keeps execution records (`DeviceOptions::record_executions`).
    pub(super) fn records_executions(&self) -> bool {
        self.device().shared().record_executions
    }

    /// Record a barrier the program asked for (a compute encoder's memory barriers).
    pub(super) fn barrier(&mut self, barrier: Barrier) {
        let Some(raw) = self.recording() else {
            return;
        };
        // SAFETY: the encoder's command buffer is recording, and only this encoder records it.
        unsafe { barrier.record(self.device().shared().vk(), raw) };
    }

    /// `request`'s direct indexed draw joins an open run of the index gather pass, which records
    /// before the encoder's next command (`index_rewrite.rs`): where its list lies, or `None`
    /// where no run takes it. Its reads `uses` are tracked here, the run's barriers before and
    /// after it ordering them. Refused as the run's objects are refused.
    pub(super) fn gather(
        &mut self,
        request: GatherRequest<'_>,
        uses: &[Use<ResourceKey>],
    ) -> Result<Option<GatheredList>, Error> {
        if self.gathers.full() {
            self.flush_gathers();
        }
        let device = self.device().clone();
        let joined = {
            let mut state = self.open.command_buffer.state();
            self.gathers.join(&device, &mut state.uploads, request)?
        };
        let Some((list, held)) = joined else {
            return Ok(None);
        };
        self.joined(held, uses);
        self.device().shared().counts.gathered_draw();
        Ok(Some(list))
    }

    /// `request`'s indirect draw joins an open indirect run of the index gather pass, which
    /// records before the encoder's next command (`index_rewrite.rs`): where its list and command
    /// lie, or `None` where no run takes it. Its reads `uses` are tracked here, the run's barriers
    /// before and after it ordering them. Refused as the run's objects, the command buffer's
    /// index pool and its status words are refused.
    pub(super) fn gather_indirect(
        &mut self,
        request: IndirectGatherRequest<'_>,
        uses: &[Use<ResourceKey>],
    ) -> Result<Option<CommandedList>, Error> {
        if self.gathers.full() {
            self.flush_gathers();
        }
        let device = self.device().clone();
        let queue = self.open.command_buffer.command_queue().id();
        let joined = {
            let mut state = self.open.command_buffer.state();
            let pool = state.index_pool(&device, queue)?;
            let status = state.status_words(&device)?;
            self.gathers
                .join_indirect(&device, (&mut state.uploads, &status, &pool), request)?
        };
        let Some((list, held)) = joined else {
            return Ok(None);
        };
        self.joined(held, uses);
        self.device().shared().counts.gathered_draw();
        Ok(Some(list))
    }

    /// `request`'s direct indexed draw joins an open run of the index rewrite pass, which records
    /// before the encoder's next command (`index_rewrite.rs`): where its list and command lie, or
    /// `None` where no run takes it. Its reads `uses` are tracked here, the run's barriers before
    /// and after it ordering them. Refused as the run's objects and the command buffer's status
    /// words are refused.
    pub(super) fn rewrite_run(
        &mut self,
        request: RewriteRunRequest<'_>,
        uses: &[Use<ResourceKey>],
    ) -> Result<Option<CommandedList>, Error> {
        if self.gathers.full() {
            self.flush_gathers();
        }
        let device = self.device().clone();
        let joined = {
            let mut state = self.open.command_buffer.state();
            let status = state.status_words(&device)?;
            self.gathers
                .join_rewrite(&device, (&mut state.uploads, &status), request)?
        };
        let Some((list, held)) = joined else {
            return Ok(None);
        };
        self.joined(held, uses);
        self.device().shared().counts.gathered_draw();
        Ok(Some(list))
    }

    /// `request`'s indexed indirect draw joins an open run of the index bounds pass, which
    /// records before the encoder's next command (`index_bounds.rs`): its slot, or `None` where
    /// no run takes it. Its reads `uses` are tracked here, the run's barriers before and after it
    /// ordering them. Refused as the run's objects, the command buffer's index pool and its
    /// status words are refused.
    pub(super) fn bounds_run(
        &mut self,
        request: BoundsRunRequest<'_>,
        uses: &[Use<ResourceKey>],
    ) -> Result<Option<BoundsSlot>, Error> {
        if self.gathers.full() {
            self.flush_gathers();
        }
        let device = self.device().clone();
        let queue = self.open.command_buffer.command_queue().id();
        let joined = {
            let mut state = self.open.command_buffer.state();
            let pool = state.index_pool(&device, queue)?;
            let status = state.status_words(&device)?;
            self.gathers
                .join_bounds(&device, (&mut state.uploads, &status, &pool), request)?
        };
        let Some((slot, held)) = joined else {
            return Ok(None);
        };
        self.joined(held, uses);
        self.device().shared().counts.bounded_run_draw();
        Ok(Some(slot))
    }

    /// A draw joined a run: hold what a new run's command buffer holds, and track the draw's
    /// reads `uses` (the run's barrier from everything before orders them: the one tracking
    /// returns goes unrecorded).
    fn joined(&mut self, held: Option<GatherHeld>, uses: &[Use<ResourceKey>]) {
        if let Some(held) = held {
            if let Some(output) = held.output {
                self.retain(output);
            }
            for sets in held.sets {
                self.retain(sets);
            }
        }
        let _ = self.command_later(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            uses,
            &BTreeSet::new(),
        );
    }

    /// Record the open gather runs, where any is open, into the encoder's `VkCommandBuffer`.
    fn flush_gathers(&mut self) {
        if !self.gathers.is_empty() {
            self.recording();
        }
    }

    /// Record the open gather runs into `raw`, the encoder's `VkCommandBuffer`.
    fn record_gathers(&mut self, raw: vk::CommandBuffer) {
        if self.gathers.is_empty() {
            return;
        }
        let runs = self.gathers.take();
        let device = self.open.command_buffer.device().shared().vk();
        // SAFETY: `raw` is the encoder's, recording outside a render pass (a pass records into
        // the one it holds), and only this encoder records it; each run's kernel is the device's,
        // and its set and output are held by the command buffer (`gather`), as its table's region
        // is the command buffer's.
        unsafe { index_rewrite::record_runs(device, raw, &runs) };
    }

    /// The `VkCommandBuffer` the encoder's next command records into: `raw`, or where a pass
    /// holds the one before and no command came since, a new one of the command buffer's pool,
    /// begun, which goes before the pass's; the open gather runs record into it first
    /// ([`gather`](Self::gather)). `None` where none can be made, which abandons the command
    /// buffer (as a failure to end one does, [`defer`](Self::defer)).
    fn recording(&mut self) -> Option<vk::CommandBuffer> {
        let raw = self.raw_recording()?;
        self.record_gathers(raw);
        Some(raw)
    }

    /// [`recording`](Self::recording) before the open gather runs record into it.
    fn raw_recording(&mut self) -> Option<vk::CommandBuffer> {
        if self.raw == vk::CommandBuffer::null() {
            match self.continuation() {
                Ok(next) => self.raw = next.0,
                Err(_) => {
                    self.open.command_buffer.state().abandoned = true;
                    return None;
                }
            }
        }
        self.recorded = true;
        Some(self.raw)
    }

    /// Hold a `VkCommandBuffer` for a render pass instance whose draws record inline in it from
    /// now to the pass's end (`render.rs`): the encoder's own where nothing recorded into it
    /// yet, else a new one of the pool, begun. Until [`release_pass`](Self::release_pass), the
    /// encoder's commands record into another, which goes before it in submission order: where
    /// they go before a pass recorded at its end. Refused where one is held, or none can be made.
    ///
    /// The first encoder's own holds its command buffer's start, written first
    /// (`gpu_time.rs`): where a command then records before the pass, in a command buffer that
    /// goes before it, the batch's first command buffer is no longer the one that wrote it, and
    /// the batch reads the start its command buffer of barriers writes before everything instead
    /// (`command_buffer/submit.rs`, which compares the batch's first command buffer with the one
    /// that wrote the start).
    pub(super) fn hold_pass(&mut self) -> Result<vk::CommandBuffer, Error> {
        if self.pass.is_some() {
            return Err(internal("a render pass held twice"));
        }
        self.flush_gathers();
        let held = match self.recorded {
            true => self.continuation()?.0,
            false => {
                let own = self.raw;
                self.raw = vk::CommandBuffer::null();
                own
            }
        };
        self.pass = Some(held);
        Ok(held)
    }

    /// End the hold [`hold_pass`](Self::hold_pass) took: the render pass instance recorded in
    /// the held command buffer ends there, whatever the encoder's commands record from here on
    /// (a command that names a detached buffer records nothing, [`skips`](Self::skips)); the
    /// command buffer of the commands before the pass, where any recorded, ends and goes before
    /// it; and the encoder's commands record on into the pass's, after it. Refused only where
    /// that one does not end, which abandons the command buffer.
    pub(super) fn release_pass(&mut self) -> Result<(), Error> {
        self.flush_gathers();
        let Some(pass) = self.pass.take() else {
            return Ok(());
        };
        // SAFETY: the held command buffer is recording inside the render pass instance its pass
        // began, and only this encoder records it.
        unsafe { self.device().shared().vk().cmd_end_render_pass(pass) };
        if self.raw != vk::CommandBuffer::null() {
            // SAFETY: as in `barrier`.
            if let Err(error) = unsafe { self.open.command_buffer.end_raw(self.raw) } {
                self.open.command_buffer.state().abandoned = true;
                return Err(vulkan(error));
            }
            self.ended.push(self.raw);
        }
        self.raw = pass;
        self.recorded = true;
        Ok(())
    }

    /// After [`release_pass`](Self::release_pass), hand the pass's command buffer, where its first
    /// instance and what followed it recorded, over to the segment the pass records later in
    /// (`render/executions.rs`), ended: it goes between what the segment records before the pass
    /// and the pass's later instances ([`cut_before`](Self::cut_before)), so nothing the segment
    /// records before the pass at the commit runs after the first instance. The encoder's next
    /// command begins another command buffer. Refused only where it does not end, which abandons
    /// the command buffer.
    pub(super) fn hand_over_pass(&mut self) -> Result<vk::CommandBuffer, Error> {
        self.flush_gathers();
        let pass = self.raw;
        if pass == vk::CommandBuffer::null() {
            return Err(internal(
                "a render pass handed over without its command buffer",
            ));
        }
        // SAFETY: as in `barrier`.
        if let Err(error) = unsafe { self.open.command_buffer.end_raw(pass) } {
            self.open.command_buffer.state().abandoned = true;
            return Err(vulkan(error));
        }
        self.raw = vk::CommandBuffer::null();
        self.recorded = false;
        Ok(pass)
    }

    /// In a [`segment`](Self::segment): place `before`, an ended command buffer handed over to it
    /// ([`hand_over_pass`](Self::hand_over_pass)), after what the segment recorded so far, between
    /// barriers from everything before and to everything after, and go on in a new command buffer.
    /// Refused where the current one does not end or the new one cannot be begun.
    pub(super) fn cut_before(&mut self, before: vk::CommandBuffer) -> Result<(), Error> {
        self.barrier(Barrier::everything());
        // SAFETY: as in `barrier`.
        unsafe { self.open.command_buffer.end_raw(self.raw) }.map_err(vulkan)?;
        self.ended.push(self.raw);
        self.ended.push(before);
        self.raw = self.continuation()?.0;
        self.barrier(Barrier::everything());
        Ok(())
    }

    /// Record one command that ran in `stages` with `uses`: the barrier the uses need after the
    /// encoder's earlier commands (in an ordered encoder), then the command through `record`.
    /// Call it once every check of the command passed: recording cannot fail.
    pub(super) fn command(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<ResourceKey>],
        record: impl FnOnce(&Vk, vk::CommandBuffer),
    ) {
        self.command_reaching(stages, uses, &BTreeSet::new(), record);
    }

    /// [`command`](Self::command) of a command whose uses name every byte of untracked memory it
    /// touches (`tracker::Reach::Named`): a blit encoder's command, a shadow's copy. In an
    /// ordered encoder it orders after the earlier such commands per key and range alone, and
    /// after every untracked access of the others. It reads the boxes `reads` of textures'
    /// images (a blit's sources): a read of a new texture's unwritten pattern there is announced
    /// only where a box reaches an unwritten texel (`fresh_unwritten.rs`); its other reads as
    /// `command`'s.
    pub(super) fn command_named(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<ResourceKey>],
        reads: &[FreshRead],
        record: impl FnOnce(&Vk, vk::CommandBuffer),
    ) {
        let none = BTreeSet::new();
        self.command_observing(stages, uses, (&none, Reach::Named), reads, record);
    }

    /// `command` of a dispatch or draw whose uses of the keys `resident` reach
    /// their buffers through the resident set alone (`binding::CommandUses`): the command does
    /// not name them.
    pub(super) fn command_reaching(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<ResourceKey>],
        resident: &BTreeSet<ResourceKey>,
        record: impl FnOnce(&Vk, vk::CommandBuffer),
    ) {
        self.command_observing(stages, uses, (resident, Reach::Unnamed), &[], record);
    }

    /// [`command_reaching`](Self::command_reaching) reading the boxes `reads`
    /// ([`command_named`](Self::command_named)), whose uses say what untracked memory it touches
    /// as `reach` does.
    fn command_observing(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<ResourceKey>],
        (resident, reach): (&BTreeSet<ResourceKey>, Reach),
        reads: &[FreshRead],
        record: impl FnOnce(&Vk, vk::CommandBuffer),
    ) {
        let Some(barrier) = self.track(stages, uses, (resident, reach), reads) else {
            return;
        };
        let Some(raw) = self.recording() else {
            return;
        };
        let device = self.open.command_buffer.device().shared().vk();
        if let Some(barrier) = barrier {
            // SAFETY: as in `barrier`.
            unsafe { barrier.record(device, raw) };
        }
        record(device, raw);
    }

    /// [`command_reaching`](Self::command_reaching) of a command recorded later, into a segment
    /// of its own (a dispatch whose pipelines are being made, `pending_dispatch.rs`): everything
    /// of it but its recording, here, in the encoder's order. The barrier its uses need after the
    /// encoder's earlier commands is returned for its recording to record first: the segment
    /// lies between the encoder's `VkCommandBuffer`s in submission order, so the barriers of the
    /// encoder's later commands order them after it as they would in one. `None` where it
    /// records nothing, as `command_reaching` (a detached buffer).
    pub(super) fn command_later(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<ResourceKey>],
        resident: &BTreeSet<ResourceKey>,
    ) -> Option<Option<Barrier>> {
        self.track(stages, uses, (resident, Reach::Unnamed), &[])
    }

    /// Everything of a command [`command_observing`](Self::command_observing) records but its
    /// recording: its reads of new textures announced, its uses tracked, and their keys kept and
    /// held. The barrier the command records first, or `None` where it records nothing.
    fn track(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<ResourceKey>],
        (resident, reach): (&BTreeSet<ResourceKey>, Reach),
        reads: &[FreshRead],
    ) -> Option<Option<Barrier>> {
        // A command that names a detached buffer records nothing, nor does any after it.
        let named = uses
            .iter()
            .map(|used| used.key)
            .filter(|key| !resident.contains(key));
        if self.skips(named) {
            self.touch_reaching(uses, resident);
            return None;
        }
        // A command reading a new texture's levels that read zero where Metal reads its pattern,
        // before a clear covered them, announces it (`fresh_unwritten.rs`).
        self.observe_fresh(uses, reads);
        // Hazard tracking takes the uses with their buffers' host memory's; the keys kept and
        // held are the uses' own.
        let aliased = self.aliased(uses);
        let tracked = aliased.as_deref().unwrap_or(uses);
        // A write of a carried shadowed texture's truth another way leaves its image stale: the
        // texture is no longer carried, whatever this encoder uses (`shadow::ShadowCarry`).
        self.shadows.wrote(tracked);
        self.open.command_buffer.note_dirty(tracked);
        let barrier = self.uses.barrier(tracked, reach);
        self.uses.record_reaching(stages, tracked, barrier, reach);
        if barrier.is_some() {
            self.device().shared().counts.command_barrier();
        }
        self.touch_reaching(uses, resident);
        Some(barrier)
    }

    /// Announce `fresh-pattern-zeroed` for each texture `uses` read in a level and layer still
    /// unwritten since it was made (`fresh_unwritten.rs`): a blit's, a shader's or an input
    /// attachment's read. An attachment's own reads are its pass's load, which the pass decides
    /// as it begins (`FreshUnwritten::begin_pass`). A use of a texture `reads` names boxes of is
    /// read in those boxes alone.
    fn observe_fresh(&self, uses: &[Use<ResourceKey>], reads: &[FreshRead]) {
        let device = self.device();
        let fresh = &device.shared().fresh;
        if !fresh.any() {
            return;
        }
        let attachment = vk::AccessFlags2::COLOR_ATTACHMENT_READ
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
        let reading = |used: &&Use<ResourceKey>| !(used.read.access & !attachment).is_empty();
        for used in uses.iter().filter(reading) {
            if let UseRange::Image {
                aspects,
                levels,
                layers,
            } = &used.range
            {
                let mut boxes = reads.iter().filter(|read| read.key == used.key).peekable();
                let notice = match boxes.peek().is_some() {
                    true => boxes.find_map(|read| {
                        fresh.observed_within(read.key, read.aspects, &read.texels)
                    }),
                    false => {
                        let (levels, layers) = (levels.clone(), layers.clone());
                        fresh.observed(used.key, *aspects, levels, layers)
                    }
                };
                if let Some(notice) = notice {
                    device.degrade_fresh_pattern(notice);
                }
            }
        }
    }

    /// `uses` with a use of the host memory's key beside each use of a no-copy buffer that
    /// joined another's memory (`host_aliases.rs`), which hazard tracking and the shadows take;
    /// `None` where no use names one.
    fn aliased(&self, uses: &[Use<ResourceKey>]) -> Option<Vec<Use<ResourceKey>>> {
        self.device().shared().aliases.augment(uses)
    }

    /// Keep the keys `uses` name (`keys`); and hold each buffer over host pages among them the
    /// first time (`rebind.rs`), retaining the pages the commands took until the command buffer
    /// completes.
    fn touch(&mut self, uses: &[Use<ResourceKey>]) {
        self.touch_reaching(uses, &BTreeSet::new());
    }

    /// `touch` `uses`, of which those of the keys `resident` reach their buffers
    /// through the resident set alone: every other key is named (`Taken`), where the device has
    /// a buffer over host pages.
    fn touch_reaching(&mut self, uses: &[Use<ResourceKey>], resident: &BTreeSet<ResourceKey>) {
        self.hold_resident();
        let rebinds = self.device().shared().rebinds.any();
        for (at, used) in uses.iter().enumerate() {
            let key = used.key;
            let named = rebinds && !resident.contains(&key);
            let before = self.touched.get(at).copied();
            if before.map(|(touched, _)| touched) != Some(key) && self.keys.insert(key) && rebinds {
                self.hold(key);
            }
            if named && before != Some((key, true)) {
                self.taken.name(key);
            }
            match self.touched.get_mut(at) {
                Some(touched) => *touched = (key, named),
                None => self.touched.push((key, named)),
            }
        }
        self.touched.truncate(uses.len());
    }

    /// Keep `keys`, those of resources a command uses (its own uses', or its bindings', which
    /// it may not read), holding each buffer over host pages among them as `touch` does.
    pub(super) fn touch_keys(&mut self, keys: impl IntoIterator<Item = ResourceKey>) {
        let rebinds = self.device().shared().rebinds.any();
        for key in keys {
            if self.keys.insert(key) && rebinds {
                self.hold(key);
            }
        }
    }

    /// Retain `used`, a resource the encoder first made resident (`use_resource`, `use_heap`),
    /// until the command buffer completes; and, where it is a buffer over host pages or a heap
    /// whose bytes are, hold it from the encoder's next command on, whether or not a command
    /// reaches it, as Metal holds every buffer an encoder made resident from the submission of a
    /// command it encoded after, and fails that submission where the buffer is detached by then
    /// (code 8). An encoder that encodes no command after it holds nothing of it, as Metal's.
    pub(super) fn made_resident(&mut self, used: Used) {
        if let Some(key) = used.bytes_key() {
            if self.device().shared().rebinds.any() {
                self.resident_unheld.push(key);
            }
        }
        self.retain(used);
    }

    /// Hold the buffers the encoder made resident since its last command (`made_resident`), as
    /// its commands' uses are held: once each, the keys kept.
    fn hold_resident(&mut self) {
        while let Some(key) = self.resident_unheld.pop() {
            if self.keys.insert(key) {
                self.hold(key);
            }
        }
    }

    /// Hold each buffer over host pages among `keys` from now on, as a command that names it
    /// does (`touch_keys`), without keeping the keys: the bytes a render pass's attachments lie
    /// in, which it holds from its open (`render.rs`). Free while the device has no buffer over
    /// host pages.
    pub(super) fn hold_bytes(&mut self, keys: impl IntoIterator<Item = ResourceKey>) {
        if !self.device().shared().rebinds.any() {
            return;
        }
        for key in keys {
            self.hold(key);
        }
    }

    /// Hold `key` where it names a buffer over host pages (`rebind.rs`).
    fn hold(&mut self, key: ResourceKey) {
        // Encoded again after the commit: nothing refuses a rebind for it.
        if self.replaying {
            if let Some(pinned) = self.holds.pin(key) {
                self.taken.pin(key, pinned);
            }
            return;
        }
        if let Some(pinned) = self.holds.hold(key) {
            self.taken.pin(key, pinned);
        }
    }

    /// Whether the commands it records from now on were called while no buffer of the device was
    /// detached (`unattached_as_called`): `true` while a render encoder lowers calls it logged,
    /// whose [`skips`](Self::skips) then skips nothing a detach after their calls would.
    pub(super) fn judges_unattached(&mut self, unattached: bool) {
        self.unattached_as_called = unattached;
    }

    /// Whether the encoder records nothing for a command that names `keys`: one of them is a
    /// detached buffer's (`detach.rs`), or an earlier command of the encoder named one. From
    /// that command on the encoder records nothing (`Taken::die`): its submission fails at
    /// the command buffer's first submission, `CommandBufferError::DetachedBacking` while the
    /// buffer is detached and `EncodedUseRebound` where it lies in pages again by then, so
    /// nothing it would record runs. The keys are kept, held and named as a recorded command's
    /// are. Free while no buffer is detached. Call it once every check of the command passed,
    /// before any Vulkan call that takes one of the buffers.
    pub(super) fn skips(&mut self, keys: impl IntoIterator<Item = ResourceKey> + Clone) -> bool {
        if !self.taken.dead() {
            if self.unattached_as_called {
                return false;
            }
            let rebinds = &self.device().shared().rebinds;
            if !rebinds.any_detached(keys.clone()) {
                return false;
            }
            self.taken.die();
        }
        self.touch_keys(keys.clone());
        for key in keys {
            self.taken.name(key);
        }
        true
    }

    /// The keys its commands named so far, their uses' and their bindings' (`touch`).
    pub(super) fn named_keys(&self) -> &BTreeSet<ResourceKey> {
        &self.keys
    }

    /// Whether its commands reached a heap so far: one they made resident, or the heap of a
    /// resource they named.
    pub(super) fn reached_heaps(&self) -> bool {
        let heap_keys = &self.device().shared().heap_keys;
        !self.heaps.is_empty() || !heap_keys.among(self.keys.iter().copied()).is_empty()
    }

    /// Note `heaps`, which a command the encoder recorded reached through what it made resident.
    pub(super) fn touch_heaps(&mut self, heaps: impl IntoIterator<Item = Heap>) {
        let rebinds = self.device().shared().rebinds.any();
        for heap in heaps {
            if !self.heaps.contains(&heap) {
                // A heap over host pages is held as its resources' uses are.
                let key = heap.bytes().map(|bytes| bytes.identity().key);
                if let Some(key) = key.filter(|&key| rebinds && self.keys.insert(key)) {
                    self.hold(key);
                }
                self.heaps.push(heap);
            }
        }
    }

    /// Note one command recorded outside the encoder's own `VkCommandBuffer` (a render encoder's
    /// draw, in its secondary command buffer) that runs in `stages` with `uses`. Only an
    /// unordered encoder's commands are recorded so: they need no barrier between them. Call it
    /// before recording the command; it refuses (internal) an ordered encoder. A draw noted with
    /// the same uses, reaching the same keys through the resident set alone, as the last one
    /// `noted` keeps (a render encoder's draws'), into the same state, is noted already
    /// (`noted_apart.rs`); a command with no `noted` (a tile dispatch) is noted whole.
    pub(super) fn note_reaching(
        &mut self,
        stages: vk::PipelineStageFlags2,
        (uses, resident): (&[Use<ResourceKey>], &BTreeSet<ResourceKey>),
        noted: Option<&mut NotedApart>,
    ) -> Result<(), Error> {
        if self.uses.ordered() {
            return Err(super::error::internal(
                "a command recorded apart in an ordered encoder",
            ));
        }
        self.observe_fresh(uses, &[]);
        self.hold_resident();
        let Some(noted) = noted else {
            self.record_aliased(stages, uses);
            self.touch_reaching(uses, resident);
            return Ok(());
        };
        let shared = self.device().shared();
        // Read before the host aliases' uses are worked out, so a change made since counts.
        let into = NotedInto {
            core: self.id,
            parts: self.uses.parts(),
            held: self.uses.held().len(),
            rebinds: shared.rebinds.any(),
            aliases: shared.aliases.version(),
        };
        if noted.holds(stages, (uses, resident), into) {
            shared.counts.command_noted_as_before();
            return Ok(());
        }
        let fresh = noted.fresh(uses, into);
        self.record_aliased(stages, fresh);
        self.touch_reaching(uses, resident);
        noted.note(stages, (uses, resident), into);
        Ok(())
    }

    /// Record one command apart that runs in `stages` with `uses`, and the uses of the host
    /// memory beside them ([`aliased`](Self::aliased)), worked out into the vector the encoder
    /// keeps for them.
    fn record_aliased(&mut self, stages: vk::PipelineStageFlags2, uses: &[Use<ResourceKey>]) {
        let aliases = &self.open.command_buffer.device().shared().aliases;
        let all = match aliases.augment_into(uses, &mut self.aliased_uses) {
            true => &self.aliased_uses[..],
            false => uses,
        };
        self.open.command_buffer.note_dirty(all);
        self.uses.record(stages, all, None);
    }

    /// The copies a command that uses the shadowed textures `shadows` and has `uses` needs
    /// before it, which the encoder then counts as recorded: the scratch of a merge (`shadow.rs`,
    /// `shadow_merge.rs`) retained until the command buffer completes. Refused, changing
    /// nothing, as the scratch is. Call it once every other check of the command passed, and
    /// record the copies with [`shadow_copies`](Self::shadow_copies), then the command.
    #[track_caller]
    pub(super) fn plan_shadows(
        &mut self,
        shadows: &[ShadowUse],
        uses: &[Use<ResourceKey>],
    ) -> Result<Vec<ShadowCopy>, Error> {
        let device = self.device().clone();
        let aliased = self.aliased(uses);
        let uses = aliased.as_deref().unwrap_or(uses);
        // The kept images its merges take are held to the encoder's end, those of a refused
        // command let go of here.
        let images = self.open.command_buffer.command_queue().kept_images();
        let held = self.kept_leases.len();
        let leases = &mut self.kept_leases;
        let planned = self
            .shadows
            .before(shadows, uses, |copies, made_for, area| {
                let merging = (made_for.two_ways, made_for.blocks, area);
                let (merge, lease) = ShadowMerge::new(&device, copies, merging, images)?;
                leases.extend(lease);
                Ok(Arc::new(merge))
            });
        let copies = match planned {
            Ok(copies) => copies,
            Err(refused) => {
                self.kept_leases.truncate(held);
                return Err(refused);
            }
        };
        let copies = self.sample_moves(copies);
        self.count_shadow_copies(&copies);
        // How this command's first copies in stood against their images' last standing, which
        // sizes the carry on a run without a timing (`shadow::CarryTally`).
        let tally = self.shadows.tally();
        (device.shared().counts).shadow_first_copies_in_tallied(tally);
        let by = match Location::caller().file() {
            file if file.ends_with("compute.rs") => KeepBy::Compute,
            file if file.ends_with("blit.rs") => KeepBy::Blit,
            _ => KeepBy::Render,
        };
        for copy in &copies {
            if let Step::Keep(merge) = &copy.step {
                let churn = &device.shared().counts.churn;
                churn.kept(by, merge.cuts(), merge.texels_kept());
                self.retain(Arc::clone(merge));
            }
            if copy.cause == Some(CopyCause::First) {
                (self.first_copies).push((copy.copies.shadow_key, copy.texels_copied_in()));
            }
        }
        Ok(copies)
    }

    /// `copies` with the moves between each heap's multisample texture's image and Metal's tiles
    /// their records take (`ShadowCopy::plan_moves`, `texture/heap_tiled.rs`), planned now from
    /// the images the encoder's queue keeps (`texture/copy_images.rs`), each held until the
    /// command buffer completes. Where they cannot be made (the device's memory, a broken
    /// invariant), the command buffer fails from the encoder on, as a recording that fails does
    /// (`command_buffer::recording_error`), and the copies record no moves.
    fn sample_moves(&mut self, mut copies: Vec<ShadowCopy>) -> Vec<ShadowCopy> {
        if !copies.iter().any(ShadowCopy::moves_samples) {
            return copies;
        }
        let device = self.device().clone();
        let mut planned = Vec::new();
        let mut failed = None;
        {
            let images = self.open.command_buffer.command_queue().copy_images();
            let maker = images.maker(&device);
            for copy in &mut copies {
                match copy.plan_moves(&maker) {
                    Ok(Some(moves)) => {
                        let moves = Arc::new(moves);
                        planned.push(Arc::clone(&moves));
                        copy.moves = Some(moves);
                    }
                    Ok(None) => {}
                    Err(error) => {
                        failed.get_or_insert(error);
                    }
                }
            }
        }
        for moves in planned {
            self.retain_shared(moves);
        }
        if let Some(error) = failed {
            self.fail_command_buffer(recording_error(&device, error));
        }
        copies
    }

    /// Note that the encoder's pass takes the shadowed texture `key` as an attachment.
    pub(super) fn shadow_attached(&mut self, key: ResourceKey) {
        if !self.attachment_shadows.contains(&key) {
            self.attachment_shadows.push(key);
        }
    }

    /// Count shadowed textures' `copies` as planned (`WorkCounts::shadow_copies`), the texels
    /// their copies in and back of buffers' textures' images move, each copy in's texels by why
    /// it is made and what holds its truth (`WorkCounts::shadow_texels_first_copied_in` and kin),
    /// those copies in of what changed within a rectangle alone
    /// (`WorkCounts::bounded_shadow_copies`), and the keeps and merges whose copies of the shadow
    /// into the scratch dispatch (`WorkCounts::shadow_merge_copies_dispatched`), and the texels the
    /// keeps copy (`WorkCounts::shadow_texels_kept`).
    fn count_shadow_copies(&self, copies: &[ShadowCopy]) {
        let texels_in = copies.iter().map(ShadowCopy::texels_in).sum();
        let texels_back = copies.iter().map(ShadowCopy::texels_back).sum();
        let counts = &self.device().shared().counts;
        counts.shadow_copies_planned(copies.len(), texels_in, texels_back);
        for copy in copies {
            counts.shadow_copy_in_planned(copy);
        }
        let bounded = (copies.iter())
            .filter(|copy| copy.cause == Some(CopyCause::Bounded))
            .count();
        if bounded > 0 {
            counts.bounded_shadow_copies_planned(bounded);
        }
        let dispatched = copies
            .iter()
            .filter(|copy| match &copy.step {
                Step::Keep(merge) => merge.dispatches(),
                Step::Merge(merge) => merge.changed_dispatches(),
                Step::Into | Step::Back => false,
            })
            .count();
        if dispatched > 0 {
            counts.shadow_merge_copies_dispatched(dispatched);
        }
        let kept: u64 = copies
            .iter()
            .filter_map(|copy| match &copy.step {
                Step::Keep(merge) => Some(merge.texels_kept()),
                _ => None,
            })
            .sum();
        if kept > 0 {
            counts.shadow_texels_kept(kept);
        }
    }

    /// Note that the encoder's work from here on waits for another writer the program ordered
    /// before it (a fence's update, `fence.rs`): Metal shows it that writer's bytes, so no
    /// shadowed texture's image is carried on and each copies its truth in again at its next use
    /// (`shadow::ShadowCarry`).
    pub(super) fn ordered_other_work(&mut self) {
        self.shadows.not_carried();
    }

    /// Whether the encoder used a shadowed texture so far: where it did not, a command that uses
    /// none takes no copy and notes no use ([`plan_shadows`](Self::plan_shadows),
    /// [`shadows_used`](Self::shadows_used)).
    pub(super) fn used_shadows(&self) -> bool {
        !self.shadows.is_empty()
    }

    /// Note a recorded command that used the shadowed textures `shadows` and had `uses`, and
    /// whose writes `within` change the bytes of a texture inside a rectangle of its texels alone
    /// (`shadow::Within`).
    pub(super) fn shadows_used(
        &mut self,
        shadows: &[ShadowUse],
        uses: &[Use<ResourceKey>],
        within: &[Within],
    ) {
        let aliased = self.aliased(uses);
        let within: Vec<Within> = within
            .iter()
            .map(|part| Within {
                uses: self
                    .aliased(&part.uses)
                    .unwrap_or_else(|| part.uses.clone()),
                texels: part.texels.clone(),
                area: part.area,
            })
            .collect();
        self.shadows
            .after(shadows, aliased.as_deref().unwrap_or(uses), &within);
    }

    /// Record shadowed textures' `copies` as commands of the encoder: ordered as any command
    /// is, in an encoder Metal orders; in one whose commands Metal leaves unordered, as one run
    /// ([`ShadowRun`]) between a barrier from everything before its first copy and one to
    /// everything after its last. A copy back the device runs faster as a dispatch is that
    /// dispatch (`shadow_copy_back.rs`), a merged texture's copy in and its keep are one dispatch
    /// at the copy in where the device takes it (`shadow_copy_in.rs`), and the rows a copy back
    /// moves out of a staging buffer are a copy of rows (`row_copies.rs`); the command buffer
    /// retains each until it completes.
    pub(super) fn shadow_copies(&mut self, copies: &[ShadowCopy]) {
        let device = self.device().clone();
        let mut run = ShadowRun::default();
        // The keeps a dispatch at their copy in took, which record nothing, and those that copy
        // the staging buffer their copy in filled.
        let mut taken = Vec::new();
        let mut staged = Vec::new();
        for (i, copy) in copies.iter().enumerate() {
            if taken.contains(&i) {
                continue;
            }
            if let Some((keep, both, uses)) = keep_with_copy_in(&device, copies, i) {
                taken.push(keep);
                let record = |device: &Vk, raw: vk::CommandBuffer| {
                    // SAFETY: the encoder's command buffer is recording outside a render pass;
                    // its commands retain the shadowed texture (and its view), the merge's scratch
                    // and the dispatch until it completes, and every image rests in the `GENERAL`
                    // layout.
                    unsafe { both.record(device, raw) }
                };
                let stages = vk::PipelineStageFlags2::COMPUTE_SHADER;
                self.shadow_command(&mut run, (stages, &uses, false), record);
                device.shared().counts.shadow_copy_in_dispatched();
                self.retain(both);
                continue;
            }
            // A copy back or merge into host pages a consumer logs is reported to the log as the
            // shadow's own at the commit (`dirty_pages.rs`).
            if let Some((log, bytes)) = copy.written_log(&device) {
                self.open.command_buffer.dirty().wrote_back(log, bytes);
            }
            // A shadow takes the view a dispatch reads on a device that dispatches alone.
            let dispatched = copy.dispatched(&device);
            let rows = match dispatched {
                Some(_) => None,
                None => copy.staged_back(&device),
            };
            let (stages, uses) = match &dispatched {
                Some(_) => (
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    copy.dispatched_uses(),
                ),
                None => (
                    copy.stages_with(rows.as_ref()),
                    copy.uses_with(rows.as_ref()),
                ),
            };
            // A copy in over host pages a consumer logs takes the regions its gate decides as its
            // batch is submitted (`dirty_pages.rs`, `shadow::GatedCopyIn`).
            let gate = match &dispatched {
                Some(_) => None,
                None => copy.gate_log(&device).map(|(log, texel)| {
                    let shadows = &device.shared().dirty;
                    (self.open.command_buffer.dirty()).gate((log, &copy.copies, texel), shadows)
                }),
            };
            // A keep may copy the staging buffer its copy in filled only where that copy in
            // copies the texture's bytes whole: recorded here, ungated.
            if dispatched.is_none() && gate.is_none() {
                staged.extend(staged_keep(copies, i));
            }
            let from_staging = staged.contains(&i);
            if let Some(gate) = gate {
                // Tracked here as the copy, recorded at its batch's submission.
                let named = std::cell::Cell::new(false);
                let tracked = |_: &Vk, _: vk::CommandBuffer| named.set(true);
                self.shadow_command(&mut run, (stages, &uses, copy.writes_truth()), tracked);
                if named.get() {
                    self.defer_copy_in(copy, gate);
                }
                if let Some(sets) = rows.and_then(RowCopies::retained) {
                    self.retain(sets);
                }
                continue;
            }
            let kept = std::cell::Cell::new(false);
            let record = |device: &Vk, raw: vk::CommandBuffer| match &dispatched {
                // SAFETY: the encoder's command buffer is recording outside a render pass; its
                // commands retain the shadowed textures (and their views) and the dispatch until
                // it completes, and every image rests in the `GENERAL` layout.
                Some(back) => unsafe { back.record(device, raw) },
                None if from_staging => {
                    // SAFETY: as above; the keep follows its texture's copy in, after the barrier
                    // its read of the shadow takes from that copy's write, with no copy of the
                    // shadow between them (`staged_keep`).
                    unsafe { copy.record_keep_from_staging(device, raw) };
                    kept.set(true);
                }
                // SAFETY: as above; the rows are the copy's own, made just now.
                None => unsafe { copy.record_gated(device, raw, rows.as_ref(), None) },
            };
            self.shadow_command(&mut run, (stages, &uses, copy.writes_truth()), record);
            if kept.get() {
                device.shared().counts.shadow_keep_from_staging();
            }
            if let Some(back) = dispatched {
                device.shared().counts.shadow_copy_back_dispatched();
                self.retain(back);
            }
            if let Some(sets) = rows.and_then(RowCopies::retained) {
                self.retain(sets);
            }
        }
        if run.recorded() {
            self.barrier(Barrier::everything());
        }
    }

    /// Record one shadowed texture's copy, of `uses` in `stages`, writing its truth where
    /// `writes_truth`, by `record`: as any command in an encoder Metal orders, else in `run`,
    /// after a barrier from everything before where the run takes one ([`ShadowRun::before`]).
    /// A copy of a detached buffer's bytes records nothing, nor does what follows
    /// ([`skips`](Self::skips)).
    fn shadow_command(
        &mut self,
        run: &mut ShadowRun,
        (stages, uses, writes_truth): (vk::PipelineStageFlags2, &[Use<ResourceKey>], bool),
        record: impl FnOnce(&Vk, vk::CommandBuffer),
    ) {
        if self.uses.ordered() {
            self.command_named(stages, uses, &[], record);
            return;
        }
        if self.skips(uses.iter().map(|used| used.key)) {
            return;
        }
        let aliased = self.aliased(uses);
        let tracked = aliased.as_deref().unwrap_or(uses);
        if run.before(tracked, writes_truth) {
            self.barrier(Barrier::everything());
        }
        let Some(raw) = self.recording() else {
            return;
        };
        record(self.device().shared().vk(), raw);
        self.open.command_buffer.note_dirty(tracked);
        self.uses.record(stages, tracked, None);
        self.touch(uses);
    }

    /// Leave `copy`, a shadow's copy in over host pages a consumer logs that the encoder tracked
    /// already ([`shadow_command`](Self::shadow_command)), to its batch's submission, its regions
    /// decided by `gate` then (`shadow::GatedCopyIn`), the encoder going on in a new
    /// `VkCommandBuffer`. A render pass held for an instance recorded inline
    /// ([`hold_pass`](Self::hold_pass)) stays held: the copies of its first instance go before
    /// the pass, so the segment goes between the command buffers of the commands before it,
    /// ahead of the pass's draws that read the copy. Recorded whole here instead inside a
    /// predicated loop's laying, which its check may submit again, or where no new
    /// `VkCommandBuffer` can be made.
    fn defer_copy_in(&mut self, copy: &ShadowCopy, gate: Arc<CopyInGate>) {
        let laying = (self.open.command_buffer.state().predication_mut()).laying();
        let next = match laying {
            true => None,
            false => self.continuation().ok(),
        };
        let Some(next) = next else {
            if let Some(raw) = self.recording() {
                // SAFETY: the encoder's command buffer is recording outside a render pass; its
                // commands retain the shadowed texture until it completes, and its image rests in
                // the `GENERAL` layout.
                unsafe { copy.record(self.device().shared().vk(), raw, None) };
            }
            return;
        };
        let work = Deferred::GatedCopyIn(Arc::new(GatedCopyIn::new(copy.clone(), gate)));
        // Refused only where the current `VkCommandBuffer` does not end, which abandons the
        // command buffer.
        let _abandoned = self.defer_into(next, work);
    }

    /// The `VkCommandBuffer` a render pass holds ([`hold_pass`](Self::hold_pass)), until its
    /// release.
    pub(super) fn held_pass(&self) -> Option<vk::CommandBuffer> {
        self.pass
    }

    /// Record shadowed textures' `copies` into the command buffer a render pass holds
    /// ([`hold_pass`](Self::hold_pass)), between two of its render pass instances, as the pass's
    /// end records them there (`render/inline_splits.rs`): as one run ([`ShadowRun`]) after the
    /// barrier from everything before that the caller records at the instance's end, with one
    /// to everything after its last copy; their uses the encoder's, an unordered one's. A copy of
    /// a detached buffer's bytes records nothing, nor does what follows ([`skips`](Self::skips)).
    /// Nothing where no pass holds one.
    pub(super) fn held_shadow_copies(&mut self, copies: &[ShadowCopy]) {
        let Some(held) = self.pass else {
            return;
        };
        let device = self.device().clone();
        let raw = device.shared().vk();
        let mut run = ShadowRun::after_barrier();
        for copy in copies {
            let rows = copy.staged_back(&device);
            let uses = copy.uses_with(rows.as_ref());
            if self.skips(uses.iter().map(|used| used.key)) {
                continue;
            }
            let aliased = self.aliased(&uses);
            let tracked = aliased.as_deref().unwrap_or(&uses);
            let before = run.before(tracked, copy.writes_truth());
            // SAFETY: the held command buffer is recording outside a render pass instance (the
            // caller ended the one before), and only this encoder records it; its commands retain
            // the shadowed textures (and their views) and the rows' set until it completes, and
            // every image rests in the `GENERAL` layout.
            unsafe {
                if before {
                    Barrier::everything().record(raw, held);
                }
                copy.record(raw, held, rows.as_ref());
            }
            let stages = copy.stages_with(rows.as_ref());
            self.open.command_buffer.note_dirty(tracked);
            self.uses.record(stages, tracked, None);
            self.touch(&uses);
            if let Some(sets) = rows.and_then(RowCopies::retained) {
                self.retain(sets);
            }
        }
        if run.recorded() {
            // SAFETY: as above.
            unsafe { Barrier::everything().record(raw, held) };
        }
    }

    /// The encoder's uses, for its fence waits and updates.
    pub(super) fn uses(&mut self) -> &mut EncoderUses<ResourceKey> {
        &mut self.uses
    }

    /// Leave `work` to record here later: the shadowed textures written since their last copy
    /// back are copied back where the work leaves them (`Deferred::leaves_shadows`), the
    /// encoder's `VkCommandBuffer` ends, and its later commands go into a new one. Where the GPU
    /// may write what the work reads, the segment starts a part of the encoder of its own (the
    /// module's segments). Refused, changing nothing, when the new one cannot be made; a failure
    /// to end the current one abandons the command buffer.
    pub(super) fn defer(&mut self, work: Deferred) -> Result<(), Error> {
        self.release_pass()?;
        let next = self.continuation()?;
        self.defer_into(next, work)
    }

    /// The new `VkCommandBuffer` the encoder's commands go on in after a segment
    /// ([`defer`](Self::defer)), allocated from the command buffer's pool and begun. Left unused
    /// on a refusal, it goes with the pool.
    pub(super) fn continuation(&self) -> Result<Continuation, Error> {
        let command_buffer = &self.open.command_buffer;
        let (next, laying) = {
            let mut state = command_buffer.state();
            let next = command_buffer.allocate_raw(&mut state)?;
            (next, state.predication_mut().laying())
        };
        if !laying {
            self.begin(next, vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
            return Ok(Continuation(next, Laid::Outside));
        }
        // Inside a predicated loop's laying, which its check may submit again
        // (`command_buffer/rerun.rs`): begun to be submitted again, behind a barrier from
        // everything before it, the work left apart before it among them.
        self.begin(next, vk::CommandBufferUsageFlags::empty())?;
        // SAFETY: just begun, and only this encoder records it.
        unsafe { Barrier::everything().record(self.device().shared().vk(), next) };
        command_buffer.state().predication_mut().laid_into(next);
        Ok(Continuation(next, Laid::Inside))
    }

    /// End the `VkCommandBuffer` the encoder records into and go on in a new one, begun to be
    /// submitted again, for a predicated loop's laying (`predication.rs`), which its check may
    /// run again: the textures it shadows copied back first, so the laying copies in again what
    /// it reads, and a barrier from everything before at its start. Its handle. Refused where the
    /// current one does not end, or none can be made, which abandons the command buffer.
    pub(super) fn cut_reusable(&mut self) -> Result<vk::CommandBuffer, Error> {
        let copies = self.shadows.leave();
        let copies = self.sample_moves(copies);
        self.shadow_copies(&copies);
        self.barrier(Barrier::everything());
        let next = {
            let command_buffer = &self.open.command_buffer;
            let mut state = command_buffer.state();
            command_buffer.allocate_raw(&mut state)
        };
        let ended = next.and_then(|next| {
            // SAFETY: as in `barrier`.
            unsafe { self.open.command_buffer.end_raw(self.raw) }.map_err(vulkan)?;
            Ok(next)
        });
        let next = match ended {
            Ok(next) => next,
            Err(error) => {
                self.open.command_buffer.state().abandoned = true;
                return Err(error);
            }
        };
        self.ended.push(self.raw);
        self.raw = next;
        if let Err(error) = self.begin(next, vk::CommandBufferUsageFlags::empty()) {
            self.open.command_buffer.state().abandoned = true;
            return Err(error);
        }
        self.barrier(Barrier::everything());
        Ok(next)
    }

    /// End a predicated loop's laying (`predication.rs`) with its check, `work`: the textures it
    /// shadows copied back, a barrier from everything before to every later command and to the
    /// host, which reads the unit's halt word; the encoder goes on after the check.
    pub(super) fn end_laying(&mut self, work: Deferred) -> Result<(), Error> {
        let copies = self.shadows.leave();
        let copies = self.sample_moves(copies);
        self.shadow_copies(&copies);
        self.barrier(Barrier::to_all());
        self.defer(work)
    }

    /// Leave what the encoder recorded so far out of the command buffer's work, as a
    /// command-predication record that drops it does (`predication.rs`): its `VkCommandBuffer`s so
    /// far, ended, are never submitted; the work it left apart is never recorded, and is held as
    /// what its commands use is; the textures it shadows are forgotten unread, so nothing is
    /// copied back and a later command copies in again; and its later commands go on in a new
    /// `VkCommandBuffer`, in a part of their own, which needs no barrier after the commands left
    /// out. Its parts so far end as a failed encoder's do, with their fences' waits alone
    /// ([`end_with_replay`](Self::end_with_replay)). Refused where no new one can be made, or the
    /// current one does not end, which abandons the command buffer.
    pub(super) fn leave_out_recorded(&mut self) -> Result<(), Error> {
        self.release_pass()?;
        self.flush_gathers();
        let next = self.continuation()?;
        if self.raw != vk::CommandBuffer::null() {
            // SAFETY: as in `barrier`.
            if let Err(error) = unsafe { self.open.command_buffer.end_raw(self.raw) } {
                self.open.command_buffer.state().abandoned = true;
                return Err(vulkan(error));
            }
        }
        self.raw = next.0;
        self.recorded = false;
        self.ended.clear();
        // A segment the GPU may write what it reads for started a part of its own.
        self.left_out += 1 + self.segments.iter().filter(|segment| segment.host).count();
        for segment in self.segments.drain(..) {
            self.retained
                .boxed(Box::new((segment.work, segment.inputs)));
        }
        self.uses.split();
        self.shadows.forget();
        Ok(())
    }

    /// Begin `raw`, a `VkCommandBuffer` the command buffer's pool just handed out, with `flags`.
    fn begin(
        &self,
        raw: vk::CommandBuffer,
        flags: vk::CommandBufferUsageFlags,
    ) -> Result<(), Error> {
        let begin = vk::CommandBufferBeginInfo::default().flags(flags);
        // SAFETY: the command buffer was just allocated from the pool; only this encoder
        // records it.
        unsafe {
            self.device()
                .shared()
                .vk()
                .begin_command_buffer(raw, &begin)
        }
        .map_err(vulkan)?;
        let counts = &self.device().shared().counts;
        counts.encoder_command_buffer_begun(Begun::Continuation);
        Ok(())
    }

    /// [`defer`](Self::defer) `work`, the encoder going on in `next`: refused only where the
    /// current `VkCommandBuffer` does not end, which abandons the command buffer.
    fn defer_into(&mut self, next: Continuation, work: Deferred) -> Result<(), Error> {
        self.flush_gathers();
        if work.leaves_shadows() {
            let copies = self.shadows.leave();
            let copies = self.sample_moves(copies);
            self.shadow_copies(&copies);
        }
        // None where a pass handed its command buffer over to the work
        // ([`hand_over_pass`](Self::hand_over_pass)) and no command came since.
        if self.raw != vk::CommandBuffer::null() {
            // SAFETY: as in `barrier`.
            if let Err(error) = unsafe { self.open.command_buffer.end_raw(self.raw) } {
                self.open.command_buffer.state().abandoned = true;
                return Err(vulkan(error));
            }
            self.ended.push(self.raw);
        }
        self.raw = next.0;
        self.recorded = false;
        // Inside a predicated loop's laying, the work goes among its pieces, before the command
        // buffer the encoder goes on in (`command_buffer/rerun.rs`).
        if next.1 == Laid::Inside {
            (self.open.command_buffer.state().predication_mut()).laid_apart(&work);
        }
        let host = work.gpu_written();
        if host {
            self.uses.split();
        }
        self.segments.push(Segment {
            at: self.ended.len(),
            executions_at: self.executions.len(),
            work,
            host,
            inputs: None,
            recorded: None,
            stepped: false,
            pipelines: false,
            gpu_alone: false,
        });
        Ok(())
    }

    /// Leave `dispatch`, whose pipelines are being made and which the encoder tracked already
    /// ([`command_later`](Self::command_later)), to record later (`pending_dispatch.rs`), the
    /// encoder going on in `next`. Refused only where the current `VkCommandBuffer` does not
    /// end, which abandons the command buffer.
    pub(super) fn pend(
        &mut self,
        next: Continuation,
        dispatch: PendingDispatch,
    ) -> Result<(), Error> {
        self.defer_into(next, Deferred::Pending(Arc::new(dispatch)))
    }

    /// Metal fails the command buffer from this encoder on with `error`, as it commits: the
    /// encoder and every one after it run nothing (`State::fail_from`).
    pub(super) fn fail_command_buffer(&mut self, error: CommandBufferError) {
        self.fails_command_buffer = Some(error);
    }

    /// A command of this encoder uses a memoryless texture's memory, which it has none of: plain
    /// Metal fails the command buffer from the submission that holds the encoder on
    /// (`CommandBufferError::InvalidResource`, `State::fail_submission_from`), its GPU running
    /// none of that submission, so neither a hang nor a discard there happens. Another failure the
    /// encoder found first keeps its error.
    pub(super) fn fail_memoryless_use(&mut self) {
        match self.fails_command_buffer {
            None | Some(CommandBufferError::GpuHang | CommandBufferError::GpuRecoveryVictim) => {
                self.fails_command_buffer = Some(CommandBufferError::InvalidResource);
            }
            Some(_) => {}
        }
    }

    /// Metal's GPU hangs at this encoder (a direct draw of primitive type 9, or a pass of no draw
    /// call over a mutable rasterization-rate map, `rate_mapped_pass.rs`), as the command
    /// buffer commits: it fails from the encoder on with `CommandBufferError::GpuHang`, unless
    /// the encoder fails it otherwise first, and holds what follows (`State::hang_from`).
    pub(super) fn hang(&mut self) {
        self.fails_command_buffer
            .get_or_insert(CommandBufferError::GpuHang);
    }

    /// Metal's GPU discards the command buffer at this encoder (an object grid past what it runs
    /// within its watchdog), as it commits: it fails from the encoder on with
    /// `CommandBufferError::GpuRecoveryVictim`, unless the encoder fails it otherwise first.
    pub(super) fn discard(&mut self) {
        self.fails_command_buffer
            .get_or_insert(CommandBufferError::GpuRecoveryVictim);
    }

    /// Metal's GPU faults this encoder, as the command buffer commits: the encoder runs nothing,
    /// and the command buffer completes with `error` (`CommandBufferError::PageFault`, or
    /// `GpuRecoveryVictim`), the first faulted encoder's (`State::fault`).
    pub(super) fn fault(&mut self, error: CommandBufferError) {
        self.faults.get_or_insert(error);
    }

    /// Note that its render pass is over `map`, which it took as `state`: a reset of the map
    /// before its command buffer's submission has it encoded again (`rate_mapped_pass.rs`).
    pub(super) fn take_rate_map(
        &mut self,
        map: crate::rate_map::RasterizationRateMap,
        state: std::sync::Arc<crate::rate_map::MapState>,
    ) {
        self.taken.take_rate_map(map, state);
    }

    /// End the encoder, one of `kind` that made `residency` resident (a compute or render
    /// encoder's): its shadowed textures written since their last copy back are copied back,
    /// it joins the command buffer's ended encoders (`State::encoder_ended`), and its recording
    /// and what it used join the command buffer's, in order, one part at a time, each part
    /// after the first in a batch of its own.
    /// The command buffer keeps `replay`, the encoder's log (`replay.rs`), where a rebind or a
    /// reset before its first submission may need the encoder encoded again (a command of it
    /// named a buffer over host pages, detached ones included, or its pass is over a
    /// rasterization-rate map) and the encoder can be (it left no segment for
    /// later but one an encoding again records at once, `Segment::records_again`, and fails
    /// nothing as it commits); else the log goes here.
    pub(super) fn end_with_replay(
        mut self,
        kind: EncoderKind,
        residency: Option<Residency>,
        replay: Option<Replay>,
    ) -> Result<(), Error> {
        // A pass ended holds no command buffer (`render.rs`); this one ends here if one did.
        self.release_pass()?;
        // What Metal's identifier allocations follow as the command buffer is scheduled
        // (`unique_identifier.rs`): a compute or blit encoder that encoded a command, and any
        // render encoder.
        let worked = self.recorded || !self.executions.is_empty() || !self.segments.is_empty();
        let work = ScheduledWork {
            dispatch: worked && matches!(kind, EncoderKind::Compute | EncoderKind::Blit),
            render: kind == EncoderKind::Render,
        };
        self.open.command_buffer.note_scheduled_work(work);
        let sharing = self.shadows.sharing_truth();
        if sharing > 0 {
            self.device().shared().counts.shadows_sharing_truth(sharing);
        }
        let copies = self.shadows.finish();
        let copies = self.sample_moves(copies);
        self.count_shadow_copies(&copies);
        self.shadow_copies(&copies);
        // Whether it keeps its log, once its last commands, the copies back above among them,
        // named what they name: a copy back may be the only command that names a texture's
        // bytes over host pages (a render pass that clears a shadowed target copies nothing in).
        // Why it keeps none, where a command named a buffer over host pages: the rebound notice
        // says (`command_buffer/rebound_use.rs`).
        let unlogged = match &replay {
            _ if !self.taken.encodes_again() => None,
            None => Some(Unlogged::NoCalls),
            Some(_) if !self.segments.iter().all(Segment::records_again) => {
                Some(Unlogged::Segments)
            }
            Some(_) if self.fails_command_buffer.is_some() || self.faults.is_some() => {
                Some(Unlogged::Fails)
            }
            Some(_) => None,
        };
        // A log not kept goes back to the device's spares (`replay.rs`).
        let replay = replay.and_then(|log| {
            let keeps = self.segments.iter().all(Segment::records_again)
                && self.fails_command_buffer.is_none()
                && self.faults.is_none()
                && self.taken.encodes_again();
            if !keeps {
                self.device().shared().scheduler.spare_logs().give(log);
                return None;
            }
            Some(log)
        });
        // Its commands were recorded for one layout of each image it holds
        // (`texture/held_depth.rs`).
        if self.uses.broke_hold() {
            return Err(internal("a held depth used but as its attachment"));
        }
        // Its last `VkCommandBuffer` stays recording, the command buffer's tail: its batch's end
        // may go into it (`CommandBufferShared::close_tail`).
        let EncoderCore {
            mut open,
            raw,
            mut ended,
            segments,
            left_out,
            uses,
            retained,
            executions,
            mut shadows,
            keys,
            attachment_shadows,
            first_copies,
            mut heaps,
            holds,
            taken,
            fails_command_buffer,
            faults,
            twin_writes,
            ..
        } = self;
        ended.push(raw);
        let keyed = open
            .command_buffer
            .device()
            .shared()
            .heap_keys
            .among(keys.iter().copied());
        heaps.extend(keyed);
        let mut state = open.command_buffer.state();
        if !attachment_shadows.is_empty() {
            let first = state.shadows_attached(&attachment_shadows);
            (open.command_buffer.device().shared().counts)
                .shadow_attachments(attachment_shadows.len(), first);
        }
        // What the encoder leaves the command buffer's later encoders (`shadow::ShadowCarry`).
        state.shadows_carried(shadows.ended());
        if !first_copies.is_empty() {
            let again = state.shadows_copied_in(&first_copies);
            if again > 0 {
                (open.command_buffer.device().shared().counts)
                    .shadow_texels_first_copied_in_again(again);
            }
        }
        state.use_heaps(heaps);
        let encoder = state.encoder_ended(kind, keys, residency, taken);
        if let Some(why) = unlogged {
            state.encoder_unlogged(encoder, why);
        }
        state.encoder_holds(encoder, uses.held().to_vec());
        match fails_command_buffer {
            Some(CommandBufferError::GpuHang) => {
                state.hang_from(encoder, open.command_buffer.device())?;
            }
            // A memoryless texture's use fails Metal's whole submission (`InvalidResource`).
            Some(error @ CommandBufferError::InvalidResource) => {
                state.fail_submission_from(encoder, error);
            }
            Some(error) => state.fail_from(encoder, error),
            None => {}
        }
        if let Some(error) = faults {
            state.fault(encoder, error, open.command_buffer.device())?;
        }
        let base = state.executions.len();
        // One part, its executions these: a log not kept comes back, to go without the lock.
        let logged = replay.as_ref().map(Replay::bytes);
        let unkept = state.keep_replay(encoder, replay, base..base + executions.len());
        if let (Some(bytes), None) = (logged, &unkept) {
            open.command_buffer
                .device()
                .shared()
                .counts
                .replay_log_kept(bytes);
        }
        let mut summaries = uses.finish_parts().into_iter();
        // The parts a command-predication record left out run nothing, each with its fences'
        // waits alone (`leave_out_recorded`), in the batch the encoder's first part joins.
        for _ in 0..left_out {
            let summary = summaries
                .next()
                .ok_or_else(|| internal("an encoder's part without its summary"))?;
            state.end_encoder(
                &open.command_buffer,
                Recorded {
                    encoder,
                    raws: Vec::new(),
                    segments: Vec::new(),
                    summary: summary.fences_only(),
                    sparse: Vec::new(),
                    twin_writes: Vec::new(),
                },
            );
        }
        let parts = into_parts(ended, segments, base);
        // The writes into ASTC carriers go with the last part (a blit encoder's only one).
        let count = parts.len();
        let mut twin_writes = Some(twin_writes);
        for (at, (raws, segments)) in parts.into_iter().enumerate() {
            let summary = summaries
                .next()
                .ok_or_else(|| internal("an encoder's part without its summary"))?;
            if at > 0 {
                state.host_step();
            }
            let twin_writes = match at + 1 == count {
                true => twin_writes.take().unwrap_or_default(),
                false => Vec::new(),
            };
            state.end_encoder(
                &open.command_buffer,
                Recorded {
                    encoder,
                    raws,
                    segments,
                    summary,
                    sparse: Vec::new(),
                    twin_writes,
                },
            );
        }
        state.tail = Some(raw);
        state.retained.extend(retained.into_entry());
        state.executions.extend(executions);
        if !holds.is_empty() {
            state.holds.push(holds);
        }
        drop(state);
        if let Some(log) = unkept {
            let device = open.command_buffer.device();
            device.shared().scheduler.spare_logs().give(log);
        }
        open.ended = true;
        Ok(())
    }

    /// End a core made by [`replaying`](Self::replaying): its shadowed textures written since
    /// their last copy back are copied back, and what it recorded comes back for its encoder's
    /// place (`State::splice_replay`), in its parts, each segment it left recorded at once on the
    /// GPU alone but one whose pipelines are being made or failed to make, left as the first
    /// encoding left it ([`Segment::keeps_pipelines`]). Refused (internal) where the encoder, encoded again, failed or faulted its
    /// command buffer, or met a detached buffer, which its first encoding kept none of, or left
    /// work that no longer runs on the GPU alone: its place stays as it was.
    pub(super) fn finish_replay(mut self) -> Result<Replayed, Error> {
        let copies = self.shadows.finish();
        let copies = self.sample_moves(copies);
        self.count_shadow_copies(&copies);
        self.shadow_copies(&copies);
        if self.uses.broke_hold() {
            return Err(internal("a held depth used but as its attachment"));
        }
        self.flush_gathers();
        // SAFETY: as in `barrier`.
        unsafe { self.open.command_buffer.end_raw(self.raw) }.map_err(vulkan)?;
        let EncoderCore {
            open,
            raw,
            mut ended,
            segments,
            uses,
            retained,
            executions,
            keys,
            heaps,
            taken,
            fails_command_buffer,
            faults,
            ..
        } = self;
        if fails_command_buffer.is_some() || faults.is_some() || taken.dead() {
            return Err(internal(
                "an encoder encoded again differs from its first encoding",
            ));
        }
        let held = uses.held().to_vec();
        let summaries = uses.finish_parts();
        ended.push(raw);
        let split = into_parts(ended, segments, 0);
        if split.len() != summaries.len() {
            return Err(internal("an encoder's part without its summary"));
        }
        // Its segments recorded at once on the GPU alone, as its commit recorded the first
        // encoding's (`command_buffer/reencode.rs`): work that no longer runs there refuses the
        // encoding again, but work whose pipelines are being made, which keeps its segment.
        let mut parts = Vec::with_capacity(split.len());
        for ((raws, mut segments), summary) in split.into_iter().zip(summaries) {
            for segment in &mut segments {
                // Work whose pipelines are being made, or whose making failed, is left as the
                // first encoding left it, prepared as its commit prepared that one
                // (`Deferred::prepare`): recorded here where they are made, else by the host step
                // once they are (`pending_dispatch.rs`, `render/pending_tile.rs`). The rebind's
                // call waits for no making, and makes none itself.
                if segment.keeps_pipelines() {
                    let prepared = segment.work.prepare(&open.command_buffer, false)?;
                    segment.recorded = prepared.recorded;
                    segment.inputs = Some(prepared.inputs);
                    segment.pipelines = prepared.pipelines;
                    continue;
                }
                let recorded = segment.work.record_on_gpu(&open.command_buffer)?;
                let Some((recorded, inputs)) = recorded else {
                    return Err(internal(
                        "an encoder encoded again whose work no longer runs on the GPU alone",
                    ));
                };
                segment.recorded = Some(recorded);
                segment.inputs = Some(inputs);
                segment.stepped = true;
                segment.gpu_alone = true;
            }
            parts.push(ReplayedPart {
                raws,
                summary,
                segments,
            });
        }
        Ok(Replayed {
            parts,
            held,
            retained,
            executions,
            keys,
            taken,
            heaps,
        })
    }
}

impl EncoderCore {
    /// A segment that records nothing and takes no `VkCommandBuffer`, for work of an encoder
    /// whose submission runs nothing: dead as one whose command named a detached buffer is
    /// ([`skips`](Self::skips), `Taken::die`), so the submission never places it
    /// (`State::place`) and no part of it reaches a `VkQueue`
    /// (`State::runs_nothing_unrunnable`). Its caller has read that the encoder's submission
    /// fails (`CommandBuffer::encoder_rebound`), which no later judging unreads: pages a
    /// command took are compared by identity, and a buffer never lies in the pages it left.
    pub(super) fn unrecorded_segment() -> RecordedSegment {
        let mut taken = Taken::default();
        taken.die();
        RecordedSegment {
            raws: Vec::new(),
            summary: EncoderUses::new(false).finish(),
            retained: Retained::default(),
            executions: Vec::new(),
            keys: BTreeSet::new(),
            taken,
            resident_misses: 0,
            faults: ReplayFaults::default(),
        }
    }

    /// A segment of `command_buffer` that `record` records alone into a new `VkCommandBuffer` of
    /// its pool, retaining `retained`: no barrier around it and nothing tracked, its encoder
    /// having tracked its commands where it left them ([`command_later`](Self::command_later)),
    /// so it uses nothing past its encoder's summary.
    pub(super) fn untracked_segment(
        command_buffer: &CommandBuffer,
        record: impl FnOnce(&Vk, vk::CommandBuffer),
        retained: Box<dyn Any + Send + Sync>,
    ) -> Result<RecordedSegment, Error> {
        let mut core = EncoderCore::segment_raw(command_buffer, false)?;
        let device = core.device().shared().vk();
        record(device, core.raw);
        // SAFETY: as in `barrier`.
        unsafe { core.open.command_buffer.end_raw(core.raw) }.map_err(vulkan)?;
        core.retained.boxed(retained);
        let EncoderCore {
            raw,
            uses,
            retained,
            executions,
            keys,
            taken,
            ..
        } = core;
        Ok(RecordedSegment {
            raws: vec![raw],
            summary: uses.finish(),
            retained,
            executions,
            keys,
            taken,
            resident_misses: 0,
            faults: ReplayFaults::default(),
        })
    }

    /// End a [`segment`](Self::segment): its shadowed textures written since their last copy
    /// back are copied back, then a barrier to everything recorded after it.
    pub(super) fn end_segment(mut self) -> Result<RecordedSegment, Error> {
        let copies = self.shadows.finish();
        // Its commands wrote the images of the shadowed textures it used, so no command buffer
        // carries one of them on (`shadow::ShadowCarry`): a segment's work lies where the encoder
        // left it, which carries nothing past it either (`shadow::ShadowSync::leave`).
        drop(self.shadows.ended());
        let copies = self.sample_moves(copies);
        self.count_shadow_copies(&copies);
        self.shadow_copies(&copies);
        self.barrier(Barrier::everything());
        // SAFETY: as in `barrier`.
        unsafe { self.open.command_buffer.end_raw(self.raw) }.map_err(vulkan)?;
        let EncoderCore {
            raw,
            mut ended,
            uses,
            retained,
            executions,
            keys,
            taken,
            ..
        } = self;
        ended.push(raw);
        Ok(RecordedSegment {
            raws: ended,
            summary: uses.finish(),
            retained,
            executions,
            keys,
            taken,
            resident_misses: 0,
            faults: ReplayFaults::default(),
        })
    }
}

/// A run of shadowed textures' copies in an encoder whose commands Metal leaves unordered
/// (`EncoderCore::shadow_copies`, `held_shadow_copies`): one barrier from everything before its
/// first copy and one to everything after its last order the run against the encoder's other
/// commands, as a barrier around each copy did, a global barrier's first scope being every
/// command before it and its second every command after. Inside the run a copy goes after a
/// barrier only where it needs one:
///
/// - **Writes of the truth.** A copy back or a merge writes the texture's truth: a buffer's bytes
///   (another key's too, where buffers over the same memory are not one key,
///   `host_aliases.rs`) or a texture's image. It takes a barrier before it and one after it, as
///   every copy did.
/// - **The others** (copies in and keeps) write only memory the runtime made for the texture
///   alone: its shadow, a merge's scratch or kept image, each of a key of its own. One goes after
///   a barrier where a use of it and a use of a copy since the last barrier name the same key
///   (aliases added) over ranges that share memory, and one of the two writes it: the tracker's
///   rule for two commands that must run in order. Copies in of different textures (reads of
///   their truths, writes of their own shadows) run together; a keep after the copy in of its
///   texture (the shadow written, then read) takes its barrier.
///
/// Every image rests in `GENERAL`, so no barrier the run leaves out carried a layout transition.
#[derive(Default)]
struct ShadowRun {
    /// The uses of the copies since the run's last barrier (aliases added); `None` before its
    /// first.
    since: Option<Vec<Use<ResourceKey>>>,
    /// Whether the last copy wrote its truth: the next copy takes a barrier.
    fenced: bool,
    /// Whether a copy was recorded in the run.
    recorded: bool,
}

impl ShadowRun {
    /// A run whose first copy follows a barrier from everything before, which the caller records.
    fn after_barrier() -> ShadowRun {
        ShadowRun {
            since: Some(Vec::new()),
            fenced: false,
            recorded: false,
        }
    }

    /// Whether the next copy, of `uses` (aliases added), writing its truth where `writes_truth`,
    /// goes after a barrier (the type's rules); noted as recorded.
    fn before(&mut self, uses: &[Use<ResourceKey>], writes_truth: bool) -> bool {
        let barrier = match &self.since {
            None => true,
            Some(since) => self.fenced || writes_truth || meets(since, uses),
        };
        let since = self.since.get_or_insert_with(Vec::new);
        if barrier {
            since.clear();
        }
        since.extend(uses.iter().cloned());
        self.fenced = writes_truth;
        self.recorded = true;
        barrier
    }

    /// Whether a copy was recorded in the run, which then takes a barrier to everything after.
    fn recorded(&self) -> bool {
        self.recorded
    }
}

/// Whether a use of `later` and one of `earlier` name the same key over ranges that share memory,
/// and one of the two writes it.
fn meets(earlier: &[Use<ResourceKey>], later: &[Use<ResourceKey>]) -> bool {
    later.iter().any(|used| {
        earlier.iter().any(|before| {
            before.key == used.key
                && before.range.overlaps(&used.range)
                && !(before.write.is_empty() && used.write.is_empty())
        })
    })
}

/// The place in `copies` of the keep that copies the staging buffer the copy in at `copies[i]`
/// fills (`ShadowCopy::keeps_from_staging`): its texture's next copy, the copies between them
/// other textures' shadows', which leave its shadow and staging buffer as the copy in left them.
/// `None` where the keep moves its texels out of the image, or no keep follows.
fn staged_keep(copies: &[ShadowCopy], i: usize) -> Option<usize> {
    let copy = &copies[i];
    if !matches!(copy.step, Step::Into) {
        return None;
    }
    let shadow = copy.copies.shadow_key;
    let (j, keep) = (copies.iter().enumerate().skip(i + 1))
        .find(|(_, later)| later.copies.shadow_key == shadow)?;
    (Arc::ptr_eq(&keep.copies, &copy.copies) && keep.keeps_from_staging()).then_some(j)
}

/// The dispatch that takes the copy in at `copies[i]` and its texture's keep after it together
/// (`ShadowMerge::keep_in`, `shadow_copy_in.rs`), where `device` takes one: the keep's place in
/// `copies`, the dispatch and its uses. The copies between them are of other textures' shadows,
/// so the keep's texels are the copy in's wherever it is recorded. `None` where the two stay
/// apart.
fn keep_with_copy_in(
    device: &Device,
    copies: &[ShadowCopy],
    i: usize,
) -> Option<(usize, CopyIn, Vec<Use<ResourceKey>>)> {
    let copy = &copies[i];
    let region = copy.copied_region()?;
    let shadow = copy.copies.shadow_key;
    let (j, merge) = copies
        .iter()
        .enumerate()
        .skip(i + 1)
        .find(|(_, later)| later.copies.shadow_key == shadow)
        .and_then(|(j, later)| match &later.step {
            Step::Keep(merge) => Some((j, merge)),
            _ => None,
        })?;
    let both = merge.keep_in(device, &copy.copies, &region)?;
    Some((j, both, merge.keep_in_uses(&copy.copies)))
}
