//! The work an encoder leaves to record later: what one segment of it records (`encoder.rs`). Each
//! kind states what `commit` makes of it, whether the GPU may write what it reads (so a host step
//! records it once the work before it completed, `scheduler`), how that step records it, and what
//! the device announces for the step. The command buffer handles every kind alike
//! (`command_buffer`).

use super::command_buffer::rerun::LoopCheck;
use super::command_buffer::CommandBuffer;
use super::device::lock;
use super::encoder::RecordedSegment;
use super::error::{internal, Error};
use super::gpu_execute::{self, Rerecord};
use super::indirect_dispatch::ArgumentDispatch;
use super::indirect_execute::{self, Execute, Settled, Unforeseen};
use super::pending_dispatch::PendingDispatch;
use super::pipeline::ComputePipelineState;
use super::render::DeferredPass;
use super::resource::ResourceKey;
use super::shadow::GatedCopyIn;
use super::sparse::emulated::Scrub;
use super::tracker::Summary;
use super::Device;
use crate::pipeline::budget::Urgency;
use std::any::Any;
use std::sync::Arc;

/// Work an encoder left to record later: one segment's.
#[derive(Clone)]
pub(super) enum Deferred {
    /// An indirect command buffer's execution (`indirect_execute.rs`).
    Execute(Arc<Execute>),
    /// An indirect dispatch, whose arguments the GPU may write (`indirect_dispatch.rs`).
    Dispatch(Arc<ArgumentDispatch>),
    /// The scrub of a blit's or a render pass's writes of a sparse texture the runtime zeroes
    /// (`sparse/emulated.rs`): neither `commit` nor a host step records it, but its batch's
    /// submission, with the mappings the batch applied.
    SparseScrub(Arc<Scrub>),
    /// A shadow's copy in over host pages a consumer logs, whose regions the log decides: as with
    /// a scrub, neither `commit` nor a host step records it, but its batch's submission, once the
    /// commit's stamp is published (`shadow::GatedCopyIn`, `dirty_pages.rs`).
    GatedCopyIn(Arc<GatedCopyIn>),
    /// A render pass that executes render commands (`render/executions.rs`).
    RenderPass(Arc<DeferredPass>),
    /// A compute encoder's dispatch whose pipelines were being made (`pending_dispatch.rs`).
    Pending(Arc<PendingDispatch>),
    /// The check after a predicated loop's laying, which runs it again while the loop would run
    /// on (`command_buffer/rerun.rs`): it records nothing, and its batch waits for it.
    Loop(Arc<LoopCheck>),
}

/// What `commit` left a segment for its host step.
#[derive(Clone)]
pub(super) enum StepInputs {
    /// An execution's: the pipeline states its records may name, as `commit` settled them.
    Settled(Vec<ComputePipelineState>),
    /// An execution `commit` recorded from the CPU's records, which its buffer watches for a host
    /// step `commit` did not foresee (`indirect_execute::Unforeseen`).
    Unforeseen(Arc<Unforeseen>),
    /// An indirect dispatch's: nothing past the work itself, which holds the encoder's tables.
    /// And a dispatch's whose pipelines were being made, which holds its recording.
    Dispatch,
    /// A scrub's: none, as it has no host step.
    None,
    /// A render pass's: nothing past the pass itself, which holds the encoder's state at each
    /// execution; its step names states among those alive then, and compiles only what `commit`
    /// could not settle (`DeferredPass::settle`).
    RenderPass,
    /// A loop's check: nothing past the check itself.
    Loop,
    /// An execution `commit` recorded on the GPU alone with no barrier between its commands, which
    /// its batch's submission records again with them where a route encoded since may set a
    /// barrier in a record (`gpu_execute::Rerecord`); no work on the GPU is waited for.
    Rerecord(Arc<Rerecord>),
}

impl StepInputs {
    /// Whether the segment's recording must be made again before its batch is submitted, though
    /// a recording is in place: [`Rerecord::due`].
    pub(super) fn rerecords(&self) -> bool {
        matches!(self, StepInputs::Rerecord(rerecord) if rerecord.due())
    }
}

/// What `commit` made of a segment's work ([`Deferred::prepare`]).
pub(super) struct Prepared {
    /// Its recording, where `commit` recorded it.
    pub(super) recorded: Option<RecordedSegment>,
    /// `commit` recorded it for good, though the GPU may write what it reads: it runs on the GPU
    /// alone (`gpu_execute.rs`, `render/executions/gpu.rs`), and no host step records it.
    pub(super) for_good: bool,
    /// What its host step needs.
    pub(super) inputs: StepInputs,
    /// What the command buffer holds until it completes.
    pub(super) retained: Option<Box<dyn Any + Send + Sync>>,
    /// `commit` left its recording to a host step for the pipelines it records, being made (or
    /// failed) then (`render/pending.rs`).
    pub(super) pipelines: bool,
}

impl Deferred {
    /// Whether `self` and `other` are the same work: one segment's.
    pub(super) fn same(&self, other: &Deferred) -> bool {
        match (self, other) {
            (Deferred::Execute(a), Deferred::Execute(b)) => Arc::ptr_eq(a, b),
            (Deferred::Dispatch(a), Deferred::Dispatch(b)) => Arc::ptr_eq(a, b),
            (Deferred::SparseScrub(a), Deferred::SparseScrub(b)) => Arc::ptr_eq(a, b),
            (Deferred::GatedCopyIn(a), Deferred::GatedCopyIn(b)) => Arc::ptr_eq(a, b),
            (Deferred::RenderPass(a), Deferred::RenderPass(b)) => Arc::ptr_eq(a, b),
            (Deferred::Pending(a), Deferred::Pending(b)) => Arc::ptr_eq(a, b),
            (Deferred::Loop(a), Deferred::Loop(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// Whether the encoder copies back its shadowed textures before the work and forgets them
    /// (`EncoderCore::defer`): work recorded apart, with copies of its own. A render pass that
    /// executes render commands is not: its shadowed textures are its encoder's, copied in
    /// before it and back as the encoder ends (`render/shadows.rs`).
    /// Nor is a dispatch whose pipelines were being made: its shadowed textures were copied in
    /// at the dispatch, and its writes are its encoder's to copy back. Nor is a gated copy in,
    /// which is one of the encoder's shadow copies.
    pub(super) fn leaves_shadows(&self) -> bool {
        match self {
            Deferred::Execute(_)
            | Deferred::Dispatch(_)
            | Deferred::SparseScrub(_)
            | Deferred::Loop(_) => true,
            Deferred::RenderPass(_) | Deferred::Pending(_) | Deferred::GatedCopyIn(_) => false,
        }
    }

    /// Whether the GPU may write what the work reads now: a host step records it, before its
    /// batch is submitted. For an execution, whether its buffer was made resident for writing
    /// (`IndirectCommandBuffer::gpu_written`, for good), or it reads its range from a buffer
    /// (`Execute::indirect`), or it lies inside a block of command predication, whose counter its
    /// host step reads (`Execute::predicated`); an indirect dispatch's arguments always are GPU
    /// memory the work
    /// before it may write. A render pass's, whether an encoder made the buffer of one of its
    /// executions resident for writing, or one of them reads its range from a buffer.
    pub(super) fn gpu_written(&self) -> bool {
        match self {
            Deferred::Execute(execute) => {
                execute.icb.gpu_written() || execute.indirect() || execute.predicated()
            }
            Deferred::Dispatch(_) | Deferred::Loop(_) => true,
            Deferred::SparseScrub(_) | Deferred::GatedCopyIn(_) | Deferred::Pending(_) => false,
            Deferred::RenderPass(pass) => pass.gpu_written(),
        }
    }

    /// `commit`'s work on it, with no lock of `command_buffer` held: for a segment its host step
    /// records (`step`), what the step needs; else its recording. An execution its step would
    /// record runs on the GPU alone instead where every record it may hold is covered
    /// (`gpu_execute.rs`), recorded now for good; otherwise its step needs the states its
    /// records may name and their variants ([`Execute::settle`]). An execution recorded now from
    /// the CPU's records is watched by its buffer for a host step `commit` did not foresee. An
    /// indirect dispatch's step needs its variants ([`ArgumentDispatch::settle`]); it is never
    /// recorded before its step. A render pass is recorded now from the CPU's records, unless
    /// its step records it; a pass its step would record is recorded now for good where its
    /// executions run on the GPU alone (`DeferredPass::record_on_gpu`); otherwise the step's draws
    /// need the variants and pipelines of the states its records may name
    /// ([`DeferredPass::settle`]).
    pub(super) fn prepare(
        &self,
        command_buffer: &CommandBuffer,
        step: bool,
    ) -> Result<Prepared, Error> {
        let device = command_buffer.device();
        match self {
            Deferred::Execute(execute) if step => {
                // On the GPU alone where it is total (`gpu_execute.rs`): not a checked execution
                // whose check may see something, whose step reads its records for the check, nor
                // one inside a block of command predication without its preprocess's gate, whose
                // step reads the counter. A checked one's host step, where a route encoded since
                // gives its check something to see (`Execute::outgrown`), is settled now.
                if let Some((recorded, inputs)) = self.record_on_gpu(command_buffer)? {
                    let retained: Option<Box<dyn Any + Send + Sync>> = match &execute.check {
                        Some(_) => {
                            let residency = command_buffer.residency()?.kernel();
                            let Settled { states, variants } =
                                execute.settle(device, execute.states(device), &residency);
                            *lock(&execute.unchecked) = Some(states);
                            Some(Box::new(variants))
                        }
                        None => None,
                    };
                    return Ok(Prepared {
                        recorded: Some(recorded),
                        for_good: true,
                        inputs,
                        retained,
                        pipelines: false,
                    });
                }
                let residency = command_buffer.residency()?.kernel();
                let Settled { states, variants } =
                    execute.settle(device, execute.states(device), &residency);
                Ok(Prepared {
                    recorded: None,
                    for_good: false,
                    inputs: StepInputs::Settled(states),
                    retained: Some(Box::new(variants)),
                    pipelines: false,
                })
            }
            Deferred::Execute(execute) => {
                let recorded = execute.record(command_buffer, None)?;
                let residency = command_buffer.residency()?.kernel();
                let unforeseen = Unforeseen::new(device, execute, residency);
                execute.icb.watch(&unforeseen);
                Ok(Prepared {
                    recorded: Some(recorded),
                    for_good: false,
                    inputs: StepInputs::Unforeseen(unforeseen),
                    retained: None,
                    pipelines: false,
                })
            }
            Deferred::Dispatch(dispatch) if step => Ok(Prepared {
                recorded: None,
                for_good: false,
                inputs: StepInputs::Dispatch,
                retained: Some(Box::new(
                    dispatch.settle(device, &command_buffer.residency()?.kernel()),
                )),
                pipelines: false,
            }),
            Deferred::Dispatch(_) => Err(internal(
                "an indirect dispatch recorded before the work it waits for completed",
            )),
            Deferred::SparseScrub(_) | Deferred::GatedCopyIn(_) => Ok(Prepared {
                recorded: None,
                for_good: false,
                inputs: StepInputs::None,
                retained: None,
                pipelines: false,
            }),
            // Recorded now for good where every execution whose records the GPU may write runs
            // on the GPU alone (`render/executions/gpu.rs`). Otherwise its step's draws compile
            // what `commit` cannot foresee alone: it settles their variants and pipelines now
            // (`DeferredPass::settle`), held until the command buffer completes.
            // The step waits for the pipelines its draws record too (`render/pending.rs`).
            // And makes now what the step would make (`render/executions/ahead.rs`).
            Deferred::RenderPass(pass) if step => {
                if let Some((recorded, inputs)) = self.record_on_gpu(command_buffer)? {
                    return Ok(Prepared {
                        recorded: Some(recorded),
                        for_good: true,
                        inputs,
                        retained: None,
                        pipelines: false,
                    });
                }
                pass.make_ahead(command_buffer);
                Ok(Prepared {
                    recorded: None,
                    for_good: false,
                    inputs: StepInputs::RenderPass,
                    retained: Some(Box::new(pass.settle(command_buffer))),
                    pipelines: pass.making() || pass.failed(),
                })
            }
            // A pass of draws whose pipelines are being made, or one of which failed, is
            // recorded by a host step once they are made, which drops the draw of a failed one
            // (`render/pending.rs`). One whose executions read the CPU's records as
            // they stand now waits for them first.
            Deferred::RenderPass(pass) => {
                if pass.reads_at_commit() {
                    pass.wait_made(command_buffer.making_deadline());
                }
                if pass.making() || pass.failed() {
                    return Ok(Prepared {
                        recorded: None,
                        for_good: false,
                        inputs: StepInputs::RenderPass,
                        retained: None,
                        pipelines: true,
                    });
                }
                Ok(Prepared {
                    recorded: Some(pass.record(command_buffer, false)?),
                    for_good: false,
                    inputs: StepInputs::RenderPass,
                    retained: None,
                    pipelines: false,
                })
            }
            // Its host step reads the unit's halt word once the laying completed.
            Deferred::Loop(_) => Ok(Prepared {
                recorded: None,
                for_good: false,
                inputs: StepInputs::Loop,
                retained: None,
                pipelines: false,
            }),
            // Recorded now where its pipelines are made; else by a host step once they are,
            // which makes again one whose making failed.
            Deferred::Pending(dispatch) => {
                let later = step || dispatch.making() || dispatch.failed();
                Ok(Prepared {
                    recorded: match later {
                        true => None,
                        false => Some(dispatch.record(command_buffer)?),
                    },
                    for_good: false,
                    inputs: StepInputs::Dispatch,
                    retained: None,
                    pipelines: later,
                })
            }
        }
    }

    /// Whether a commit may record it on the GPU alone, for good
    /// ([`record_on_gpu`](Self::record_on_gpu)): an execution's, or a render pass's that executes
    /// render commands.
    pub(super) fn may_run_on_gpu_alone(&self) -> bool {
        matches!(self, Deferred::Execute(_) | Deferred::RenderPass(_))
    }

    /// Its recording on the GPU alone, for good, with what its host step needs, where it runs
    /// there over the buffers as they lie now (`gpu_execute.rs`, `DeferredPass::record_on_gpu`);
    /// `None` where it does not, or is work that never runs there. With no lock of
    /// `command_buffer` held: at the commit, and at a rebind before the command buffer's first
    /// submission, which records it again (`command_buffer/reencode.rs`).
    pub(super) fn record_on_gpu(
        &self,
        command_buffer: &CommandBuffer,
    ) -> Result<Option<(RecordedSegment, StepInputs)>, Error> {
        match self {
            Deferred::Execute(execute) => {
                let recorded = gpu_execute::record(command_buffer, execute)?;
                Ok(recorded.map(|(recorded, rerecord)| {
                    let inputs = match rerecord {
                        Some(rerecord) => StepInputs::Rerecord(rerecord),
                        None => StepInputs::Settled(Vec::new()),
                    };
                    (recorded, inputs)
                }))
            }
            Deferred::RenderPass(pass) => Ok(pass
                .record_on_gpu(command_buffer)?
                .map(|recorded| (recorded, StepInputs::RenderPass))),
            Deferred::Dispatch(_)
            | Deferred::SparseScrub(_)
            | Deferred::GatedCopyIn(_)
            | Deferred::Pending(_)
            | Deferred::Loop(_) => Ok(None),
        }
    }

    /// Its host step, on the scheduler's thread with no lock of `command_buffer` held: its
    /// recording from what it reads now and `inputs`. An execution's records name their states
    /// among those `commit` settled, or for one `commit` recorded from the CPU's records, those
    /// `use_resource` settled since. An indirect dispatch's arguments state its size. A render
    /// pass's executions read their records as the GPU and the CPU left them.
    pub(super) fn step(
        &self,
        command_buffer: &CommandBuffer,
        inputs: &StepInputs,
    ) -> Result<RecordedSegment, Error> {
        match self {
            Deferred::Execute(execute) => {
                // Recorded on the GPU alone with a check a route encoded since gave something to
                // see: recorded from the records as the GPU and the CPU left them, naming states
                // among those `commit` settled, for good.
                if let Some(states) = execute.take_outgrown() {
                    return execute.record(command_buffer, Some(&states));
                }
                let states = match inputs {
                    StepInputs::Settled(states) => states.clone(),
                    StepInputs::Unforeseen(unforeseen) => {
                        unforeseen.states(command_buffer.device())
                    }
                    // Recorded again on the GPU alone, with the barriers: no record is read.
                    StepInputs::Rerecord(rerecord) => {
                        return rerecord.record(command_buffer, execute);
                    }
                    StepInputs::Dispatch
                    | StepInputs::None
                    | StepInputs::RenderPass
                    | StepInputs::Loop => {
                        return Err(internal("an execution's host step without its states"));
                    }
                };
                // The states go after the recording, with no lock held: one may be a state's
                // last handle.
                execute.record(command_buffer, Some(&states))
            }
            Deferred::Dispatch(dispatch) => dispatch.record(command_buffer),
            Deferred::SparseScrub(_) => Err(internal("a sparse scrub's host step")),
            Deferred::GatedCopyIn(_) => Err(internal("a gated copy in's host step")),
            Deferred::RenderPass(pass) => pass.record(command_buffer, true),
            Deferred::Pending(dispatch) => dispatch.record(command_buffer),
            Deferred::Loop(check) => check.step(command_buffer),
        }
    }

    /// Whether a route encoded since `commit` recorded the work for good on the GPU may write
    /// more than `commit` decided on (`DeferredPass::outgrown`), or gives the check of an
    /// execution recorded there something to see (`Execute::outgrown`): its host step records it
    /// instead, before its batch is submitted. A render pass's executions' GPU form, and an
    /// execution's on the GPU alone with a check.
    pub(super) fn outgrown(&self) -> bool {
        match self {
            Deferred::RenderPass(pass) => pass.outgrown(),
            Deferred::Execute(execute) => execute.outgrown(),
            Deferred::Dispatch(_)
            | Deferred::SparseScrub(_)
            | Deferred::GatedCopyIn(_)
            | Deferred::Pending(_)
            | Deferred::Loop(_) => false,
        }
    }

    /// Whether `commit` recorded it from the CPU's records of an indirect command buffer the CPU
    /// wrote since (`Execute::rewritten`, `DeferredPass::rewritten`): its host step records it
    /// again from the records as they stand then, before its batch is submitted, waiting for no
    /// work on the GPU.
    pub(super) fn rewritten(&self) -> bool {
        match self {
            Deferred::Execute(execute) => execute.rewritten(),
            Deferred::RenderPass(pass) => pass.rewritten(),
            Deferred::Dispatch(_)
            | Deferred::SparseScrub(_)
            | Deferred::GatedCopyIn(_)
            | Deferred::Pending(_)
            | Deferred::Loop(_) => false,
        }
    }

    /// Whether its batch waits for it after its host step: a loop's check while it runs the
    /// laying again (`command_buffer/rerun.rs`), which wakes the scheduler as it ends.
    pub(super) fn holds(&self) -> bool {
        match self {
            Deferred::Loop(check) => check.running(),
            Deferred::Execute(_)
            | Deferred::Dispatch(_)
            | Deferred::SparseScrub(_)
            | Deferred::GatedCopyIn(_)
            | Deferred::RenderPass(_)
            | Deferred::Pending(_) => false,
        }
    }

    /// Whether a pipeline the work records is still being made (`render/pending.rs`).
    pub(super) fn making(&self) -> bool {
        match self {
            Deferred::RenderPass(pass) => pass.making(),
            Deferred::Pending(dispatch) => dispatch.making(),
            Deferred::Execute(_)
            | Deferred::Dispatch(_)
            | Deferred::SparseScrub(_)
            | Deferred::GatedCopyIn(_)
            | Deferred::Loop(_) => false,
        }
    }

    /// Whether a host step records it from what its calls resolved alone, nothing of the buffers
    /// it names read again then: the buffers over host pages its commands used let their pages go
    /// at the commit instead of keeping them until it completes (`CommandBuffer::commit`'s
    /// `steps_hold`, `rebind.rs`), so a rebind of one is not refused but encodes its encoder
    /// again (`command_buffer/reencode.rs`). A dispatch whose pipelines were being made records
    /// what its encoding resolved (`pending_dispatch.rs`), and so does a render pass `commit`
    /// left for the pipelines its draws and tile dispatches record (`pipelines`) that executes no
    /// render commands, whose instances' commands its calls recorded
    /// (`render/pending_tile.rs`). Everything else its step resolves as it records: a rebind
    /// meanwhile is refused.
    pub(super) fn resolved_at_its_calls(&self, pipelines: bool) -> bool {
        match self {
            Deferred::Pending(_) => true,
            Deferred::RenderPass(pass) => pipelines && !pass.executes(),
            Deferred::Execute(_)
            | Deferred::Dispatch(_)
            | Deferred::SparseScrub(_)
            | Deferred::GatedCopyIn(_)
            | Deferred::Loop(_) => false,
        }
    }

    /// Whether the making of a pipeline the work records failed (`DeferredPass::failed`,
    /// `PendingDispatch::failed`): a dispatch's, which its recording makes again, on the thread
    /// that records it, the command buffer completing with
    /// `CommandBufferError::PipelineCreation` where that fails too; a draw's, which its
    /// recording drops (`render/pending.rs`).
    pub(super) fn failed(&self) -> bool {
        match self {
            Deferred::RenderPass(pass) => pass.failed(),
            Deferred::Pending(dispatch) => dispatch.failed(),
            Deferred::Execute(_)
            | Deferred::Dispatch(_)
            | Deferred::SparseScrub(_)
            | Deferred::GatedCopyIn(_)
            | Deferred::Loop(_) => false,
        }
    }

    /// What its recording uses, on `device`, where that is known before it records
    /// (`DeferredPass::known_summary`): a render pass's that executes no render commands. `None`
    /// for the rest, whose uses their recording finds: an execution's or an indirect dispatch's
    /// from the records and arguments it reads, a scrub's at its submission; a loop's check's are
    /// its laying's (`CommandBufferShared::footprint`).
    pub(super) fn known_summary(&self, device: &Device) -> Option<Summary<ResourceKey>> {
        match self {
            Deferred::RenderPass(pass) => pass.known_summary(device),
            Deferred::Pending(dispatch) => Some(dispatch.known_summary()),
            Deferred::Execute(_)
            | Deferred::Dispatch(_)
            | Deferred::SparseScrub(_)
            | Deferred::GatedCopyIn(_)
            | Deferred::Loop(_) => None,
        }
    }

    /// Run `waker` each time the making of a pipeline the work records finishes.
    pub(super) fn when_made(&self, waker: &(impl Fn() + Clone + Send + 'static)) {
        match self {
            Deferred::RenderPass(pass) => pass.when_made(waker),
            Deferred::Pending(dispatch) => dispatch.when_made(waker),
            Deferred::Execute(_)
            | Deferred::Dispatch(_)
            | Deferred::SparseScrub(_)
            | Deferred::GatedCopyIn(_)
            | Deferred::Loop(_) => {}
        }
    }

    /// Run the makings of the pipelines the work records before the device's pool's pieces of a
    /// lower rank than `urgency` (`pipeline/pool.rs`): its submission waits for them.
    pub(super) fn hasten(&self, urgency: Urgency) {
        match self {
            Deferred::RenderPass(pass) => pass.hasten(urgency),
            Deferred::Pending(dispatch) => dispatch.hasten(urgency),
            Deferred::Execute(_)
            | Deferred::Dispatch(_)
            | Deferred::SparseScrub(_)
            | Deferred::GatedCopyIn(_)
            | Deferred::Loop(_) => {}
        }
    }

    /// Announce on `device`, once, that a host step records such work.
    pub(super) fn degrade(&self, device: &Device) {
        match self {
            Deferred::Execute(execute) => indirect_execute::degrade_host_replay(device, execute),
            Deferred::Dispatch(dispatch) => dispatch.degrade(device),
            Deferred::SparseScrub(_)
            | Deferred::GatedCopyIn(_)
            | Deferred::Pending(_)
            | Deferred::Loop(_) => {}
            Deferred::RenderPass(pass) => pass.degrade(device),
        }
    }
}
