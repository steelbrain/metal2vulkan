//! A draw encoded while its `VkPipeline` is still being made.
//!
//! - **Where creation runs.** A draw whose `VkPipeline` its variant has not made yet hands the
//!   making to the device's pool (`pipeline/pool.rs`) and goes on: the draw returns once encoded,
//!   as any draw, and the makings of several draws, of one command buffer or of many, run side
//!   by side. The encode-time checks and refusals all stay at the draw; only the driver's
//!   creation call moves.
//! - **What records later.** The draw's own commands go into a secondary command buffer of their
//!   own, begun at the draw against the instance's pass and recorded once the pipeline is made:
//!   the instance's secondary ends before it and a new one goes on after it, as around an
//!   execution of render commands ([`Piece::Pending`](super::Piece)), and the pass records as a
//!   segment of its encoder (`deferred.rs`). A draw's recording binds everything it uses and
//!   sets every dynamic state, so it records alike in a secondary of its own. Around it: the
//!   barrier from the draws before that wrote storage the pixel interlock orders (the fragment
//!   imageblock's, or an interlocked color's), recorded first in it; and an active visibility query, which ends in the secondary before and continues in
//!   the one after, the draw counting in a query lent to it of the same offset
//!   (`visibility::Lent`), or in none where it counts no sample. A draw whose snapshots lead its
//!   instance (`render/snapshot.rs`, leading snapshots) records its snapshot draws into a second
//!   secondary of its own, begun at the draw too, which the leading render pass instance
//!   executes once recorded.
//! - **When it records.** `commit` records the pass where every pipeline it waits for is made
//!   by then; otherwise it records nothing of it, and the scheduler's host step records it once
//!   they are made (`scheduler`), its batch held in its queue's order meanwhile, with no wait for
//!   the GPU. A pass that executes render commands the CPU writes records at commit, which waits
//!   for its pipelines: its commands are read as they stand then.
//! - **A failed making.** The draw runs nothing and the command buffer runs on: its secondary
//!   records the barriers and the query it would record around the draw, and no draw.
//!   `CommandBuffer::dropped_draw` names the first such draw (its encoder's place and its order
//!   there), the state, its variant and specialization, and the driver's error. Counted:
//!   `WorkCounts::draws_dropped_unmade`. Where creation ran at the draw, the draw returned that
//!   error.
//! - **However long the making takes.** Under either `PipelineWaitPolicy` the batch waits for the
//!   draw's making until it is made, past its command buffer's bound
//!   (`CommandBuffer::making_deadline`) too (`Segment::steps_after_pipelines`): the draw renders
//!   what Metal's renders, its command buffer completing later, as on a slower GPU. Only its
//!   queue's line waits for it; `commit` records the pass only where its pipelines are made, a
//!   pass that executes render commands the CPU writes after waiting for them until that bound
//!   at most, and leaves it to the host step otherwise.
//! - **An execution's replay.** A draw of an execution's commands records as the replay runs
//!   (the host step, or `commit` for records the CPU writes), so it takes no pending recording:
//!   its pipeline, most begun on the device's pool by the execution's settle at commit
//!   (`RenderExecution::settle`), which the batch's host step waits for as for a pending draw's;
//!   a fan-out's probe's and fanned-out draw's (`render/fan_out.rs`), which the settle begins
//!   too, the step takes where it fans the pass out, waiting for one still being made.
//!   A making the replay begins itself, one the settle did not foresee, is waited for in a host
//!   step until it is made, on the step's thread; in `commit`, until its command buffer's bound
//!   at most, past which the replay stops, before anything of the draw is tracked, and leaves
//!   the pass to the host step, which records it once that making is made
//!   (`CommandBuffer::leave_unmade`, `DeferredPass::left_for_makings`). A device replay's draw,
//!   one recording for every command, waits until the pipeline is made.
//! - **Recorded once.** A pass recorded at commit and again by a host step (an execution's
//!   buffer the GPU may write since) records each such draw once, its secondary executed by the
//!   recording that is submitted.

use super::bound::{Bound, Recorded};
use super::{DrawState, Recording};
use crate::binding::PushWrite;
use crate::command_buffer::CommandBuffer;
use crate::device::lock;
use crate::draw_source::IndexBind;
use crate::dynamic_state::ExtendedDynamicState;
use crate::error::{vulkan, Error, FailedPipeline, FailedState};
use crate::pipeline::{Pending, RawPipeline, RenderPipelineState};
use crate::render_target;
use crate::tracker::LegacyBarrier;
use crate::visibility::Lent;
use ash::vk;
use metal2vulkan::cache::VariantKey;
use metal2vulkan::pipeline::SpecKey;
use std::sync::Mutex;

/// One recording of a draw, owned until its pipeline is made: [`Recording`]'s but for the
/// device, the command buffer and the pipeline.
pub(super) struct OwnedRecording {
    pub(super) layout: vk::PipelineLayout,
    pub(super) shape: u64,
    pub(super) sets: Vec<vk::DescriptorSet>,
    pub(super) offsets: Vec<u32>,
    pub(super) pushed: Vec<crate::binding::SetWrite>,
    pub(super) heap: Option<vk::DescriptorSet>,
    pub(super) push: Vec<PushWrite>,
    pub(super) vertex_buffers: Vec<(u32, vk::Buffer, u64, u64)>,
    pub(super) index: Option<IndexBind>,
    pub(super) indirect: Option<vk::Buffer>,
    pub(super) dynamic: Vec<vk::DynamicState>,
    pub(super) topology: metal2vulkan::vk::PrimitiveTopology,
    pub(super) primitive_restart: bool,
    pub(super) state: DrawState,
    pub(super) extended: Option<ExtendedDynamicState>,
    pub(super) draw: metal2vulkan::vk::DrawCommand,
    pub(super) pair: Option<super::snapshot::Pair>,
}

/// A draw whose pipeline is being made (the module docs).
pub(super) struct PendingDraw {
    /// Its secondary command buffer, begun at the draw against its instance's pass.
    pub(super) draws: vk::CommandBuffer,
    /// The secondary command buffer of its snapshot draws, where they lead its instance in a
    /// render pass instance of their own (`render/snapshot.rs`, leading snapshots), begun at the
    /// draw against its instance's pass too: its recordings of the snapshot phase record there.
    pub(super) leading: Option<vk::CommandBuffer>,
    pub(super) pipeline: Pending<RawPipeline>,
    /// The state it draws, which a failed making names.
    pub(super) state: RenderPipelineState,
    /// What a failed making names besides (`FailedPipeline`): its encoder's place among the
    /// command buffer's ended encoders, its order among the encoder's draw calls, the variant
    /// of the state it draws, and its pipeline's specialization constants' values.
    pub(super) encoder: usize,
    pub(super) draw: usize,
    pub(super) variant: VariantKey,
    pub(super) specialization: Vec<(SpecKey, u64)>,
    /// The view mask and kind of the barrier after its instance's earlier draws it records first
    /// (`render_target::DrawsBarrier`), where it has one.
    pub(super) storage_barrier: Option<(u32, render_target::DrawsBarrier)>,
    /// The view mask and destination stages of the barrier from its instance's earlier
    /// indirect argument reads it records first (`render_target::after_indirect_reads`), where it
    /// has one.
    pub(super) indirect_barrier: Option<(u32, vk::PipelineStageFlags)>,
    /// The view mask and masks of the barrier from its instance's earlier writes before their
    /// fragments it records first (`render_target::after_geometry_writes`), where it has one.
    pub(super) geometry_barrier: Option<(u32, LegacyBarrier)>,
    /// The visibility query it counts in.
    pub(super) query: Option<Lent>,
    pub(super) recordings: Vec<OwnedRecording>,
    /// Its recording, once made.
    pub(super) recorded: Mutex<Option<Result<(), Error>>>,
}

impl PendingDraw {
    /// Whether its pipeline's making finished, made or failed.
    pub(super) fn made(&self) -> bool {
        self.pipeline.outcome().is_some()
    }

    /// Whether its pipeline's making failed.
    pub(super) fn failed(&self) -> bool {
        matches!(self.pipeline.outcome(), Some(Err(_)))
    }

    /// Run `waker` once its pipeline's making finished.
    pub(super) fn when_made(&self, waker: impl FnOnce() + Send + 'static) {
        self.pipeline.when_finished(Box::new(move |_| waker()));
    }

    /// Its secondary command buffer, recorded and ended once its pipeline is made, waiting for
    /// that: recorded the first time, as `command_buffer`'s, without the draw where its making
    /// failed (`CommandBuffer::drop_draw`).
    pub(super) fn record(
        &self,
        command_buffer: &CommandBuffer,
    ) -> Result<vk::CommandBuffer, Error> {
        let mut recorded = lock(&self.recorded);
        if recorded.is_none() {
            *recorded = Some(self.record_once(command_buffer));
        }
        match recorded.as_ref() {
            Some(Ok(())) => Ok(self.draws),
            Some(Err(error)) => Err(error.clone()),
            None => Ok(self.draws),
        }
    }

    /// Its leading snapshots' secondary command buffer, where it has one, recorded and ended
    /// with [`record`](Self::record)'s.
    pub(super) fn leading(
        &self,
        command_buffer: &CommandBuffer,
    ) -> Result<Option<vk::CommandBuffer>, Error> {
        self.record(command_buffer)?;
        Ok(self.leading)
    }

    fn record_once(&self, command_buffer: &CommandBuffer) -> Result<(), Error> {
        // Until it is made, which its batch's host step waits for before it records
        // (`Segment::steps_after_pipelines`), past its command buffer's bound too.
        let waited = self.pipeline.wait();
        // A device test's failure of the making (`CommandBuffer::fail_draw_making`).
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        let waited = match command_buffer.state().injected_making_failure() {
            Some(error) => Err(error),
            None => waited,
        };
        let failed = |error| FailedPipeline {
            encoder: self.encoder,
            draw: self.draw,
            state: FailedState::Render(self.state.clone()),
            variant: self.variant.clone(),
            specialization: self.specialization.clone(),
            error,
        };
        let pipeline = match waited {
            Ok(pipeline) => Some(pipeline),
            Err(error) => {
                command_buffer.drop_draw(failed(error));
                None
            }
        };
        let device = command_buffer.device();
        let shared = device.shared();
        let raw = shared.vk();
        let cb = self.draws;
        // SAFETY: the secondary command buffer was begun at the draw inside a pass compatible
        // with the pipeline's, and only this recording records it, once, after its encoder
        // ended (the command buffer's pool is no encoder's then); every object recorded is live
        // until the command buffer completes (the encoder retained them at the draw, and the
        // pipeline is held with the pass); the barrier is the one the draw would record into its
        // instance, valid in the pass's subpass; the lent query is a continuation of the offset
        // the encoder's draws count in, with none active in this secondary before it. Without its
        // pipeline it records them around no draw. The leading snapshots' secondary, begun so
        // too, takes the recordings of the snapshot phase, with no barrier of theirs.
        unsafe {
            if let Some((view_mask, barrier)) = self.storage_barrier {
                barrier.record(raw, cb, view_mask);
            }
            if let Some((view_mask, stages)) = self.indirect_barrier {
                render_target::after_indirect_reads(raw, cb, view_mask, stages);
            }
            if let Some((view_mask, barrier)) = self.geometry_barrier {
                render_target::after_geometry_writes(raw, cb, view_mask, barrier);
            }
            if let Some(query) = &self.query {
                query.begin(raw, cb);
            }
            let mut bound = Bound::default();
            let mut leading_bound = Bound::default();
            let mut recorded = Recorded::default();
            if let Some(pipeline) = &pipeline {
                for owned in &self.recordings {
                    let snapshot = (owned.pair.as_ref())
                        .is_some_and(|pair| pair.phase == super::snapshot::Phase::Snapshot);
                    let (command_buffer, bound) = match (self.leading, snapshot) {
                        (Some(leading), true) => (leading, &mut leading_bound),
                        _ => (cb, &mut bound),
                    };
                    recorded += Recording {
                        raw,
                        command_buffer,
                        layout: owned.layout,
                        shape: owned.shape,
                        pipeline: pipeline.raw,
                        sets: &owned.sets,
                        offsets: &owned.offsets,
                        pushed: &owned.pushed,
                        push_descriptor: shared.push_descriptor,
                        pushes_after_pipeline_binds: shared.pushes_after_pipeline_binds,
                        heap: owned.heap,
                        push: &owned.push,
                        vertex_buffers: &owned.vertex_buffers,
                        index: owned.index,
                        indirect: owned.indirect,
                        dynamic: &owned.dynamic,
                        settled: None,
                        topology: owned.topology,
                        primitive_restart: owned.primitive_restart,
                        state: &owned.state,
                        extended: owned.extended,
                        mesh: shared.mesh_draws.as_ref(),
                        multi: shared.multi_draws.as_ref(),
                        draw: owned.draw,
                        pair: owned.pair.clone(),
                        routed: None,
                    }
                    .record(bound);
                }
            }
            recorded.count(shared.counts.thread());
            if let Some(query) = &self.query {
                query.end(raw, cb);
            }
            if let Some(leading) = self.leading {
                command_buffer.end_raw(leading).map_err(vulkan)?;
            }
            command_buffer.end_raw(cb).map_err(vulkan)?;
        }
        Ok(())
    }
}
