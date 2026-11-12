//! A tile dispatch encoded while a `VkPipeline` it records is still being made, as a compute
//! encoder's dispatch is (`pipeline/dispatch_making.rs`, `pending_dispatch.rs`).
//!
//! - **A tile dispatch never waits.** One whose region's `VkPipeline` the tile function's variant
//!   has not made hands the making to the device's pool and goes on; a making in flight already
//!   is taken as it stands. Everything else of the dispatch is done at the call, as for one whose
//!   pipelines are made: its checks and refusals, its bindings and descriptor sets, its scratch
//!   images, its notices, and its execution record (its pipelines' creation times set as its pass
//!   records). Its makings rank from the dispatch on as a committed command buffer's
//!   (`Urgency::Committed`): its command buffer waits for them.
//! - **Its pass records later** where a making is still in flight, or failed, as the encoder
//!   ends, as the pass of a draw whose pipeline is being made does (`render/pending.rs`): `commit`
//!   records it where its pipelines are made by then; otherwise the scheduler's host step records
//!   it once they are, its batch held in its queue's order meanwhile, with no wait for the GPU. A
//!   pass whose makings all finished made as the encoder ends records then. A dispatch after the
//!   pass's instances recorded inline (`render/inline_first.rs`) records in the pass's segment,
//!   before its later instances, not in the command buffer the encoder hands over with them.
//! - **A failed making.** A making that failed, or that its state's release stopped
//!   (`pipeline/piece_holds.rs`), is made again by the pass's recording, as a caller after a
//!   failed making does (`pipeline/making.rs`); where that fails too, the command buffer completes
//!   with `CommandBufferError::PipelineCreation`, its batch and those after it unrun, and
//!   `CommandBuffer::failed_pipeline` names the dispatch: its encoder's place, its order among the
//!   encoder's tile dispatch calls, the tile state, its variant and specialization, and the
//!   driver's error.
//! - **However long the making takes.** Under either `PipelineWaitPolicy` the pass's batch waits
//!   for its tile dispatches' makings until they are made, past its command buffer's bound
//!   (`CommandBuffer::making_deadline`) too (`Segment::steps_after_pipelines`), as a
//!   compute encoder's dispatch's does (`pending_dispatch.rs`): a tile dispatch writes what
//!   Metal's writes, its command buffer completing later.
//! - **Encoded again** (`replay.rs`): an encoder encoded again waits for its tile dispatches'
//!   makings no more than its first encoding did, and leaves its pass to record later in the
//!   first encoding's place (`command_buffer/reencode.rs`), prepared as the commit prepared that
//!   one: the submission or the host step records it once the makings land, so no Metal compile
//!   runs or is waited for on the program's thread inside the rebind's call. Its makings get the
//!   scheduler's waker there, as the commit gives the first encoding's.

use super::pass_steps::PassSteps;
use crate::command_buffer::CommandBuffer;
use crate::degradation::BoundFormats;
use crate::error::{Error, FailedPipeline, FailedState};
use crate::pipeline::budget::Urgency;
use crate::pipeline::{
    ComputePipelineState, Pending, RawPipeline, RegionKey, RegionPipeline, RenderPipelineState,
    Variant,
};
use metal2vulkan::bind::DispatchRegion;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// A tile dispatch's regions' `VkPipeline`s, in its regions' order.
pub(super) enum TilePipelines {
    /// Made at the dispatch.
    Made(Vec<Arc<RawPipeline>>),
    /// One or more being made at the dispatch (the module docs).
    Making(Box<PendingTile>),
}

/// A tile dispatch whose pipelines were being made at its call.
pub(super) struct PendingTile {
    variant: Arc<Variant>,
    /// Its regions' `VkPipeline`s, made or being made, each with its key.
    pipelines: Vec<(RegionKey, RegionPipeline)>,
    /// Its execution record's creation times, set once its pipelines are made, where the device
    /// keeps records (`EncoderCore::records_executions`).
    creation: Option<Arc<OnceLock<Vec<Duration>>>>,
    /// What a failed making names.
    call: TileCall,
}

/// What a tile dispatch's failed making names (`FailedPipeline`): its tile state, its encoder's
/// place among the command buffer's ended encoders, and its order among the encoder's tile
/// dispatch calls.
pub(super) struct TileCall {
    pub(super) state: RenderPipelineState,
    pub(super) encoder: usize,
    pub(super) dispatch: usize,
}

impl TilePipelines {
    /// The `VkPipeline`s of `regions` of `variant`, `kernel`'s, for a tile dispatch `call` that
    /// binds `formats`: each made, or handed to the device's pool and kept by `kernel`
    /// (`ComputePipelineState::begin_region_pipeline`), every making hastened to a committed
    /// command buffer's rank. `creation` takes the creation times of pipelines being made.
    pub(super) fn begin(
        (kernel, variant): (&ComputePipelineState, &Arc<Variant>),
        regions: &[DispatchRegion],
        formats: BoundFormats<'_>,
        call: TileCall,
        creation: Option<Arc<OnceLock<Vec<Duration>>>>,
    ) -> TilePipelines {
        let pipelines: Vec<_> = regions
            .iter()
            .map(|region| {
                let pipeline = kernel.begin_region_pipeline(variant, region, formats);
                (RegionKey::of(region), pipeline)
            })
            .collect();
        let made: Option<Vec<_>> = pipelines
            .iter()
            .map(|(_, pipeline)| match pipeline {
                RegionPipeline::Made(made) => Some(Arc::clone(made)),
                RegionPipeline::Making(_) => None,
            })
            .collect();
        if let Some(made) = made {
            return TilePipelines::Made(made);
        }
        let pending = PendingTile {
            variant: Arc::clone(variant),
            pipelines,
            creation,
            call,
        };
        let pipelines = TilePipelines::Making(Box::new(pending));
        pipelines.hasten(Urgency::Committed);
        pipelines
    }

    /// Its makings in flight or finished, by region.
    fn makings(&self) -> impl Iterator<Item = &Pending<RawPipeline>> {
        let pipelines = match self {
            TilePipelines::Made(_) => &[][..],
            TilePipelines::Making(pending) => &pending.pipelines[..],
        };
        pipelines.iter().filter_map(|(_, pipeline)| match pipeline {
            RegionPipeline::Making(pending) => Some(pending),
            RegionPipeline::Made(_) => None,
        })
    }

    /// Whether a making is still in flight.
    pub(super) fn making(&self) -> bool {
        self.makings().any(|pending| pending.outcome().is_none())
    }

    /// Whether a making failed: the pass's recording makes it again.
    pub(super) fn failed(&self) -> bool {
        self.makings()
            .any(|pending| matches!(pending.outcome(), Some(Err(_))))
    }

    /// Run `waker` each time a making finishes: at once for one finished already.
    pub(super) fn when_made(&self, waker: &(impl Fn() + Clone + Send + 'static)) {
        for pending in self.makings() {
            let waker = waker.clone();
            pending.when_finished(Box::new(move |_| waker()));
        }
    }

    /// Run its makings before the device's pool's pieces of a lower rank than `urgency`.
    pub(super) fn hasten(&self, urgency: Urgency) {
        for pending in self.makings() {
            pending.hasten(urgency);
        }
    }

    /// Wait until every making finished, made or failed, until `deadline` at most, its command
    /// buffer's bound where it has one (the module docs).
    pub(super) fn wait(&self, deadline: Option<Instant>) {
        for pending in self.makings() {
            // Its outcome is the recording's to read.
            match deadline {
                Some(deadline) => drop(pending.wait_until(deadline)),
                None => drop(pending.wait()),
            }
        }
    }

    /// Its `VkPipeline`s where every making finished made, their creation times set; `None`
    /// while one is in flight, or where one failed.
    pub(super) fn made_now(&self) -> Option<Vec<Arc<RawPipeline>>> {
        let pending = match self {
            TilePipelines::Made(made) => return Some(made.clone()),
            TilePipelines::Making(pending) => pending,
        };
        let made: Vec<_> = pending
            .pipelines
            .iter()
            .map(|(_, pipeline)| match pipeline {
                RegionPipeline::Made(made) => Some(Arc::clone(made)),
                RegionPipeline::Making(making) => making.outcome()?.ok(),
            })
            .collect::<Option<_>>()?;
        Some(pending.created(made))
    }

    /// Its `VkPipeline`s, waiting for each making until it is made, which its pass's host step
    /// runs only after (`Segment::steps_after_pipelines`): one whose making failed made again here, on this
    /// thread. Where that fails, `command_buffer` fails so (`CommandBuffer::fail_pipeline`).
    pub(super) fn made(
        &self,
        command_buffer: &CommandBuffer,
    ) -> Result<Vec<Arc<RawPipeline>>, Error> {
        let pending = match self {
            TilePipelines::Made(made) => return Ok(made.clone()),
            TilePipelines::Making(pending) => pending,
        };
        let failed = |key: &RegionKey, error| FailedPipeline {
            encoder: pending.call.encoder,
            draw: pending.call.dispatch,
            state: FailedState::Render(pending.call.state.clone()),
            variant: pending.variant.key().clone(),
            specialization: pending.variant.constants(key),
            error,
        };
        let mut made = Vec::with_capacity(pending.pipelines.len());
        for (key, pipeline) in &pending.pipelines {
            let outcome = match pipeline {
                RegionPipeline::Made(made) => Ok(Arc::clone(made)),
                RegionPipeline::Making(making) => {
                    (making.wait()).or_else(|_| pending.variant.make_region(key))
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
        Ok(pending.created(made))
    }
}

impl PendingTile {
    /// `made`, its pipelines, once their creation times are set in its execution record.
    fn created(&self, made: Vec<Arc<RawPipeline>>) -> Vec<Arc<RawPipeline>> {
        if let Some(creation) = &self.creation {
            creation.get_or_init(|| made.iter().map(|made| made.created_in).collect());
        }
        made
    }
}

impl PassSteps {
    /// Its tile dispatches' pipelines, in the dispatches' order.
    pub(super) fn tile_pipelines(&self) -> impl Iterator<Item = &TilePipelines> {
        self.dispatches
            .iter()
            .map(|staged| &staged.dispatch.pipelines)
    }

    /// Each tile dispatch's `VkPipeline`s, in the dispatches' order ([`TilePipelines::made`]),
    /// which `after` records.
    pub(super) fn tiles_made(
        &self,
        command_buffer: &CommandBuffer,
    ) -> Result<Vec<Vec<Arc<RawPipeline>>>, Error> {
        self.tile_pipelines()
            .map(|pipelines| pipelines.made(command_buffer))
            .collect()
    }
}
