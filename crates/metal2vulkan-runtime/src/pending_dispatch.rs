//! A compute encoder's dispatch encoded while a `VkPipeline` it records is still being made
//! (`pipeline/dispatch_making.rs`), and the recording of a dispatch's commands.
//!
//! - **What records later.** The dispatch's commands go into a `VkCommandBuffer` of their own,
//!   a segment of its encoder (`encoder.rs`, `deferred.rs`): the encoder's ends before it and a
//!   new one goes on after it. Everything else of the dispatch is done at the dispatch, as for
//!   one recorded there: its checks and refusals, its bindings and descriptor sets, its shadowed
//!   textures' copies in, its gates inside a block of command predication, its notices, its
//!   execution record (its pipelines' creation times set as it records), and its encoder's
//!   tracking of what it uses (`EncoderCore::command_later`), whose barrier the segment records
//!   first, as the encoder would have before the command. The segments lie between the
//!   encoder's `VkCommandBuffer`s in submission order, so the barriers the encoder records
//!   before its later commands order those after the dispatch as in one `VkCommandBuffer`, and
//!   nothing orders more than Metal asks. Only the recording of its commands, which name the
//!   `VkPipeline`, moves.
//! - **When it records.** `commit` records it where its pipelines are made by then; otherwise
//!   the scheduler's host step records it once they are made (`scheduler`), its batch held in
//!   its queue's order meanwhile, with no wait for the GPU: the batches committed after it on its
//!   queue complete after it, as Metal's do, and a batch of another queue waits for it only
//!   where the two share a resource one of them writes (`scheduler/order.rs`).
//! - **A failed making.** A making that failed, or that its state's release stopped
//!   (`piece_holds.rs`), is made again by the recording, as a caller after a failed making does
//!   (`making.rs`), on the scheduler's thread; where that fails too, the command buffer completes
//!   with `CommandBufferError::PipelineCreation`, its batch and those after it unrun, and
//!   `CommandBuffer::failed_pipeline` names the dispatch: its encoder's place and its order among
//!   the encoder's dispatches, the state, its variant and specialization, and the driver's error.
//! - **A making past its bound.** Under `PipelineWaitPolicy::Bounded` (the default) the recording
//!   waits for a making until its command buffer's bound at most, its commit plus the policy's
//!   `pipeline_commit_wait` (4.5 s by default), as a draw's does (`render/pending.rs`): a dispatch
//!   whose making is still in flight then records its barrier and no dispatch, so writes nothing,
//!   the command buffer running on without error. `CommandBuffer::dropped_draw` names the first
//!   such command with the error `VK_TIMEOUT`; counted,
//!   `WorkCounts::dispatches_dropped_still_making`; announced `dispatch-dropped-still-making`.
//!   Under `PipelineWaitPolicy::Exact` it waits until each is made.

use super::binding::{self, PushWrite, SetWrite};
use super::command_buffer::CommandBuffer;
use super::encoder::{EncoderCore, RecordedSegment};
use super::error::{vulkan, Error, FailedPipeline, FailedState};
use super::indirect_dispatch::FlatPreprocess;
use super::pipeline::{ComputePipelineState, RawPipeline, RegionKey, RegionPipeline, Variant};
use super::resource::ResourceKey;
use super::tracker::{Barrier, EncoderUses, Summary};
use super::upload::Upload;
use crate::device::functions::Vk;
use crate::pipeline::budget::Urgency;
use ash::vk;
use metal2vulkan::bind::PredicationStep;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

/// The commands of one dispatch, but its `VkPipeline`s: its sets, its regions' group counts and
/// push constants, and on the GPU path of an indirect dispatch its preprocess, inside a block of
/// command predication its gates.
pub(super) struct Recording {
    pub(super) layout: vk::PipelineLayout,
    /// Set 0, where the variant binds one.
    pub(super) set: Option<vk::DescriptorSet>,
    /// The dynamic offsets set 0 is bound with (`binding::dynamic_writes`).
    pub(super) offsets: Vec<u32>,
    /// Set 0's writes, where the variant pushes it (`pipeline::objects::pushes_set_zero`):
    /// pushed with the device's push command (`Vk::push_descriptor`).
    pub(super) pushed: Vec<SetWrite>,
    /// Set 1, the heap, where the variant has one.
    pub(super) heap: Option<vk::DescriptorSet>,
    pub(super) preprocess: Option<FlatPreprocess>,
    pub(super) gates: Vec<Upload>,
    /// Each region's group counts and push constants, in order.
    pub(super) regions: Vec<([u32; 3], Vec<PushWrite>)>,
}

impl Recording {
    /// Record the dispatch's commands into `command_buffer` with `pipelines`, its regions'.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, by this caller alone; every object
    /// recorded is live until it completes, and `pipelines` are made with the variant's layout,
    /// one per region.
    pub(super) unsafe fn record(
        &self,
        raw: &Vk,
        command_buffer: vk::CommandBuffer,
        pipelines: &[Arc<RawPipeline>],
    ) {
        // SAFETY: the caller's contract; the gates lie in the command buffer's arena.
        unsafe {
            if let Some(preprocess) = &self.preprocess {
                preprocess.record(raw, command_buffer, self.gates.first());
            }
            match raw.push_descriptor().filter(|_| !self.pushed.is_empty()) {
                Some(push) => binding::push_set(
                    push,
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    &self.pushed,
                ),
                None => {
                    if let Some(set) = self.set {
                        raw.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            self.layout,
                            0,
                            &[set],
                            &self.offsets,
                        );
                    }
                }
            }
            if let Some(heap) = self.heap {
                raw.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    self.layout,
                    1,
                    &[heap],
                    &[],
                );
            }
            for (index, ((groups, push), pipeline)) in
                self.regions.iter().zip(pipelines).enumerate()
            {
                if index > 0 {
                    Barrier::between_dispatches().record(raw, command_buffer);
                }
                raw.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline.raw);
                for (stages, offset, bytes) in push {
                    raw.cmd_push_constants(command_buffer, self.layout, *stages, *offset, bytes);
                }
                match (&self.preprocess, self.gates.get(index)) {
                    (Some(preprocess), _) => preprocess.dispatch(raw, command_buffer),
                    (None, Some(gate)) => raw.cmd_dispatch_indirect(
                        command_buffer,
                        gate.raw(),
                        gate.offset() + PredicationStep::GATE_OUT_OFFSET,
                    ),
                    (None, None) => {
                        let [x, y, z] = *groups;
                        raw.cmd_dispatch(command_buffer, x, y, z);
                    }
                }
            }
        }
    }
}

/// A dispatch whose pipelines are being made (the module docs).
pub(super) struct PendingDispatch {
    pub(super) variant: Arc<Variant>,
    /// Its regions' `VkPipeline`s, made or being made, each with its key, in its regions' order.
    pub(super) pipelines: Vec<(RegionKey, RegionPipeline)>,
    pub(super) recording: Recording,
    /// The barrier its uses need after its encoder's earlier commands
    /// (`EncoderCore::command_later`), which it records first.
    pub(super) barrier: Option<Barrier>,
    /// Its execution record's creation times, which its recording sets, where the device keeps
    /// records (`EncoderCore::records_executions`).
    pub(super) creation: Option<Arc<OnceLock<Vec<Duration>>>>,
    /// What a failed making names (`FailedPipeline`): the state it dispatched, its encoder's
    /// place among the command buffer's ended encoders, and its order among the encoder's
    /// dispatches.
    pub(super) state: ComputePipelineState,
    pub(super) encoder: usize,
    pub(super) dispatch: usize,
}

impl PendingDispatch {
    /// The makings in flight or finished it records, by region.
    fn makings(&self) -> impl Iterator<Item = &crate::pipeline::Pending<RawPipeline>> {
        self.pipelines
            .iter()
            .filter_map(|(_, pipeline)| match pipeline {
                RegionPipeline::Making(pending) => Some(pending),
                RegionPipeline::Made(_) => None,
            })
    }

    /// Whether a making it records is still in flight.
    pub(super) fn making(&self) -> bool {
        self.makings().any(|pending| pending.outcome().is_none())
    }

    /// Whether a making it records failed: its recording makes it again.
    pub(super) fn failed(&self) -> bool {
        self.makings()
            .any(|pending| matches!(pending.outcome(), Some(Err(_))))
    }

    /// Run `waker` each time a making it records finishes: at once for one finished already.
    pub(super) fn when_made(&self, waker: &(impl Fn() + Clone + Send + 'static)) {
        for pending in self.makings() {
            let waker = waker.clone();
            pending.when_finished(Box::new(move |_| waker()));
        }
    }

    /// Run its makings before the device's pool's pieces of a lower rank than `urgency`: its
    /// batch's submission waits for them.
    pub(super) fn hasten(&self, urgency: Urgency) {
        for pending in self.makings() {
            pending.hasten(urgency);
        }
    }

    /// What its recording uses past its encoder's summary: nothing, its encoder having tracked
    /// its uses at the dispatch (`EncoderCore::command_later`).
    pub(super) fn known_summary(&self) -> Summary<ResourceKey> {
        EncoderUses::new(false).finish()
    }

    /// Its regions' `VkPipeline`s, waiting for each making until `command_buffer`'s bound at
    /// most, where it has one: one whose making failed made again here. Where that fails,
    /// `command_buffer` fails so (`CommandBuffer::fail_pipeline`). `None` where a making is still
    /// in flight at the bound: the dispatch is dropped (`CommandBuffer::drop_late_dispatch`, the
    /// module docs).
    fn made(&self, command_buffer: &CommandBuffer) -> Result<Option<Vec<Arc<RawPipeline>>>, Error> {
        let deadline = command_buffer.making_deadline();
        let failed = |key: &RegionKey, error| FailedPipeline {
            encoder: self.encoder,
            draw: self.dispatch,
            state: FailedState::Compute(self.state.clone()),
            variant: self.variant.key().clone(),
            specialization: self.variant.constants(key),
            error,
        };
        let mut made = Vec::with_capacity(self.pipelines.len());
        for (key, pipeline) in &self.pipelines {
            let outcome = match pipeline {
                RegionPipeline::Made(pipeline) => Ok(Arc::clone(pipeline)),
                RegionPipeline::Making(pending) => {
                    let waited = match deadline {
                        Some(deadline) => pending.wait_until(deadline),
                        None => Some(pending.wait()),
                    };
                    let Some(waited) = waited else {
                        // Still being made at the bound: the dispatch runs nothing, and a later
                        // command buffer dispatches with the pipeline once it is made.
                        let late = failed(key, vulkan(vk::Result::TIMEOUT));
                        command_buffer.drop_late_dispatch(late, false);
                        return Ok(None);
                    };
                    waited.or_else(|_| self.variant.make_region(key))
                }
            };
            match outcome {
                Ok(pipeline) => made.push(pipeline),
                Err(error) => {
                    command_buffer.fail_pipeline(failed(key, error.clone()));
                    return Err(error);
                }
            }
        }
        Ok(Some(made))
    }

    /// Its segment of `command_buffer`, recorded once its pipelines are made: its barrier, then
    /// its commands.
    ///
    /// Nothing is recorded where a buffer over host pages one of its encoder's commands named
    /// lies elsewhere now than the command took it (`CommandBuffer::encoder_rebound`): a rebind
    /// since the commit that no encoding again reached, so the submission holding the dispatch
    /// runs nothing and the command buffer fails `CommandBufferError::EncodedUseRebound`
    /// (`command_buffer/first_submission.rs`). What it would record resolved as it was encoded,
    /// its set 0 among it, and the rebind freed the `VkBuffer` those descriptors name: a
    /// descriptor a command names must be valid as it is recorded, where its binding is not
    /// `UPDATE_AFTER_BIND` (the Vulkan specification, Descriptor Set Binding: such contents
    /// "may be consumed ... during host execution of the command, or during shader execution of
    /// the resulting draws and dispatches, or any time in between"), and a driver that reads one
    /// as it encodes reads freed memory. Its pipelines are made all the same, so a making that
    /// failed fails the command buffer as it does where the dispatch records.
    pub(super) fn record(&self, command_buffer: &CommandBuffer) -> Result<RecordedSegment, Error> {
        let pipelines = self.made(command_buffer)?;
        if let (Some(creation), Some(pipelines)) = (&self.creation, &pipelines) {
            creation.get_or_init(|| pipelines.iter().map(|made| made.created_in).collect());
        }
        if command_buffer.encoder_rebound(self.encoder) {
            return Ok(EncoderCore::unrecorded_segment());
        }
        let (barrier, recording, made) = (self.barrier, &self.recording, &pipelines);
        let record = |raw: &Vk, cb: vk::CommandBuffer| {
            // SAFETY: the segment's command buffer is recording outside a render pass, by this
            // call alone; what the dispatch records was retained at the dispatch, its pipelines
            // with the segment, and they are its variant's, one per region. A dropped dispatch
            // (none) records its barrier alone, which the encoder's later commands count on.
            unsafe {
                if let Some(barrier) = barrier {
                    barrier.record(raw, cb);
                }
                if let Some(made) = made {
                    recording.record(raw, cb, made);
                }
            }
        };
        let retained = pipelines.clone().unwrap_or_default();
        EncoderCore::untracked_segment(command_buffer, record, Box::new(retained))
    }
}
