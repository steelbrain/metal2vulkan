//! A dispatch's `VkPipeline`s made on the device's pool, as Metal makes a compute state's at its
//! creation.
//!
//! - **One `VkPipeline` per threadgroup size.** The local size is a specialization, so each
//!   threadgroup size, and each edge region of a `dispatchThreads`, is a `VkPipeline` of its own,
//!   which only a dispatch names. The first dispatch of each made it on the program's thread:
//!   0.6 to 1.5 s for a bicubic kernel on NVIDIA's 615 driver, and 14 to 107 ms for a matrix
//!   kernel at a second threadgroup size, where Metal's dispatch waits for no compile.
//! - **A dispatch never waits.** One whose region's `VkPipeline` its variant has not made hands
//!   the making to the device's pool (`pool.rs`) and goes on, as a draw does
//!   (`render/pending.rs`); a making in flight already (another dispatch's) is taken as it stands.
//!   The dispatch is recorded later (`pending_dispatch.rs`); a render encoder's tile dispatch
//!   alike, its pass recorded later (`render/pending_tile.rs`). Its specialized notices are announced
//!   at the dispatch that hands the making over, on the program's thread, naming what it binds: the
//!   pool decides nothing the program sees.
//! - **The predicted region, made at the creation, where the request states its threadgroup.**
//!   Metal compiles a compute state inside `newComputePipelineState`, and its first dispatch
//!   compiles nothing. Where the descriptor states `requiredThreadsPerThreadgroup`, that size is
//!   the threadgroup every dispatch of the state must have, so the local size of a first
//!   dispatch's region is part of the request itself. The creation then makes that
//!   region's `VkPipeline` ([`Variant::make_at_creation`]): a `dispatchThreadgroups:` on the base
//!   variant, planned for nothing bound but Metal's default sampler at each runtime sampler slot
//!   that does not compare (as a render state's prediction, `predicted.rs`), at that size
//!   ([`predicted_threads`]). A dispatch whose region has that key takes the pipeline made; one
//!   whose key differs from it only where it reads a uniform view at its bind offset alone takes
//!   it too, at once, while its own key is made on the pool, nothing waiting for it, as a draw
//!   takes its key's residual form (`render/residual.rs`); any other hands its own making to the
//!   pool, as before (on MoltenVK the kernel's MSL library is then compiled already, so that
//!   making is its specialization and Metal's pipeline compile).
//!   Under `PipelineWaitPolicy::Bounded` (the default) the making is handed to the device's pool,
//!   kept by the state's piece holds as a dispatch's is, and the creation waits for it the
//!   policy's `pipeline_creation_wait` (4.5 s by default) at most, as a render state's does
//!   (`render.rs`): a driver's cold
//!   compile holds the thread creating the state, which a host may create inside a command
//!   buffer's encoding, no longer, and a dispatch that finds the making in flight takes it, its
//!   command buffer waiting a bound for it in turn (`WorkCounts::creations_left_making`). A
//!   making still running then may run past the state's release, holding the variant and the
//!   device until it ends. Under `PipelineWaitPolicy::Exact` it runs on the creating thread and
//!   ends before the creation returns: none runs past the state's release, which would hold the
//!   variant and the device past the program's last drop, which Metal's never does. Its
//!   specialized notices wait for the first dispatch that takes it, which announces them naming
//!   what it binds. A creation inside a host step, a tile function's state and a stand-in compile
//!   make nothing here.
//! - **Where the request states no threadgroup, the creation predicts nothing**, because nothing
//!   in the request answers what the dispatch will ask for. The local size is the dispatch's own
//!   call, and a size guessed from the state instead -- Apple's guide to threadgroup sizes
//!   ("Calculating Threadgroup and Grid Sizes") computes `threadExecutionWidth` across by
//!   `maxTotalThreadsPerThreadgroup / threadExecutionWidth` down, which is `[32, 32, 1]` on an M3
//!   -- is a `VkPipeline` only a dispatch that happens to choose it can take. **Measured over the
//!   whole conformance suite on an M3 Max, one in-set run of all 1,121 cases: 141 such pipelines
//!   made in 106 cases and 0 taken by any dispatch**, while the same run's render prediction
//!   (`predicted.rs`) was taken 764 times in 794. The committed recordings' own 8-run census
//!   reads the same thing from the other side: 0 of 461 taken over the 277 rows that dispatch and
//!   do not draw (`cost.rs`'s `the_creation_prediction`). So the guess is not made, and the first
//!   dispatch hands its region's making to the pool as it did before, waiting for nothing.
//! - **What the prediction still cannot know**, where the request does state the threadgroup. It
//!   cannot know whether the dispatch will be a `dispatchThreads:` with a ragged edge, whose
//!   interior and edge regions are two local sizes and so two keys of which the prediction can
//!   hold at most one; nor what the dispatch will bind, which decides its variant
//!   (`VariantKey::extended_range` and `VariantKey::swizzled_storage` read the bound texture's
//!   pixel format and swizzle, `bind::plan::resources`), so for a kernel whose dispatches all
//!   bind a texture the prediction's base variant is one no dispatch takes and its stage's
//!   `VkShaderModule` is a module of its own. `creation_dispatch_pipelines_made` against
//!   `creation_dispatch_pipelines_drawn` is the measure of it, apart from the render
//!   prediction's own pair.
//! - **Kept by the state** (`piece_holds.rs`). What a piece needs (its variant and its making's
//!   ticket) is kept by the state, never by the pool's thread; the state's release stops a piece
//!   not started, its making failing, and a dispatch that took it (of another state of the variant)
//!   makes its own (`pending_dispatch.rs`), as a caller after a failed making does (`making.rs`).

use super::graphics::Notices;
use super::kept;
use super::making::{Begun, Pending, Ticket};
use super::objects::RawPipeline;
use super::piece_holds::PieceHolds;
use super::pool::Place;
use super::{ComputePipelineState, RegionKey, Variant};
use crate::degradation::{BoundFormats, Occurrence};
use crate::error::Error;
use crate::Device;
use metal2vulkan::bind::{BindState, BoundSampler, Dispatch, DispatchRegion, SamplerDescriptor};
use metal2vulkan::device::PipelineWaitPolicy;
use metal2vulkan::pipeline::{ComputePipeline, ComputePipelineDescriptor, SlotInterface};
use std::sync::Arc;
use std::time::Instant;

/// A region's `VkPipeline` for a dispatch that records it later
/// ([`Variant::begin_region_pipeline`]).
pub(crate) enum RegionPipeline {
    Made(Arc<RawPipeline>),
    Making(Pending<RawPipeline>),
}

/// What a compute state keeps for one of its pool pieces (`piece_holds.rs`): each holds the
/// device.
pub(crate) enum RegionHeld {
    /// A making: its variant, its region's key, and its making's ticket.
    Making(Arc<Variant>, RegionKey, Box<Ticket<RegionKey, RawPipeline>>),
    /// What a making gives back once settled, held until let go of: its variant.
    Made { _variant: Arc<Variant> },
}

impl RegionKey {
    /// The key of `region`'s `VkPipeline`.
    pub(crate) fn of(region: &DispatchRegion) -> RegionKey {
        RegionKey {
            specialization: region.specialization.clone(),
            stage_flags: region.stage_flags,
            required_subgroup_size: region.required_subgroup_size,
        }
    }

    /// Whether it is `region`'s key, [`of`](Self::of) it.
    pub(crate) fn is_of(&self, region: &DispatchRegion) -> bool {
        self.specialization == region.specialization
            && self.stage_flags == region.stage_flags
            && self.required_subgroup_size == region.required_subgroup_size
    }
}

impl Variant {
    /// Announce the specialized notices of `key`'s `VkPipeline` for a dispatch that binds
    /// `formats`.
    fn announce(&self, key: &RegionKey, formats: BoundFormats<'_>) {
        self.objects.device.degrade_specialized(
            &self.objects.specialized,
            &key.specialization,
            Some(formats),
            Occurrence::Made,
        );
    }

    /// The `VkPipeline` of `region` for a dispatch that binds `formats` and records it later
    /// (the module docs), a render encoder's tile dispatch where `tile`: from the memo, the
    /// making in flight for it, or handed now to the device's pool, its making kept by `holds`,
    /// its state's.
    fn begin_region_pipeline(
        self: &Arc<Self>,
        holds: &Arc<PieceHolds<RegionHeld>>,
        (region, tile): (&DispatchRegion, bool),
        formats: BoundFormats<'_>,
    ) -> RegionPipeline {
        let key = RegionKey::of(region);
        match self.pipelines.begin(&key) {
            Begun::Made(made) => {
                self.first_use(&made, &key, formats);
                RegionPipeline::Made(made)
            }
            // The command that began it announces its kinds. Meanwhile the dispatch takes the
            // key's residual form where it is made (`render/residual.rs`).
            Begun::Pending(pending) => match self.residual_made(&key, formats) {
                Some(made) => RegionPipeline::Made(made),
                None => RegionPipeline::Making(pending),
            },
            Begun::Yours(ticket) => {
                let counts = &self.objects.device.shared().counts;
                match tile {
                    true => counts.tile_pipeline_pooled(),
                    false => counts.dispatch_pipeline_pooled(),
                }
                self.announce(&key, formats);
                let residual = self.residual_made(&key, formats);
                let pending = ticket.pending();
                pending.queued_at(self.pooled(holds, key, ticket, false));
                match residual {
                    // Handed to the pool: nothing waits for the making.
                    Some(made) => {
                        counts.thread().residual_form_making();
                        RegionPipeline::Made(made)
                    }
                    None => RegionPipeline::Making(pending),
                }
            }
        }
    }

    /// Whether `made` is the `VkPipeline` made for `key` itself, not its residual form's, which a
    /// dispatch takes while `key`'s is being made ([`residual_made`](Self::residual_made)).
    pub(crate) fn made_as(&self, key: &RegionKey, made: &Arc<RawPipeline>) -> bool {
        (self.pipelines.made(key)).is_some_and(|own| Arc::ptr_eq(&own, made))
    }

    /// The made `VkPipeline` of `key`'s residual form, each of its uniform views read at its
    /// residual, where `key` is not its own residual form (`render/residual.rs`), for a dispatch
    /// that binds `formats` and takes it at once, which counts it as a residual-form command.
    fn residual_made(
        &self,
        key: &RegionKey,
        formats: BoundFormats<'_>,
    ) -> Option<Arc<RawPipeline>> {
        let constants = &self.pipeline.stage.specialization.constants;
        let residual = RegionKey {
            specialization: super::residual_form(constants.iter(), &key.specialization)?,
            ..key.clone()
        };
        let made = self.pipelines.made(&residual)?;
        self.first_use(&made, &residual, formats);
        self.objects
            .device
            .shared()
            .counts
            .thread()
            .residual_form_drawn();
        Some(made)
    }

    /// The `VkPipeline` of `key`, from the memo, the making in flight for it, or made now on
    /// this thread: what a dispatch whose making failed makes itself (`pending_dispatch.rs`),
    /// its notices announced at the dispatch.
    pub(crate) fn make_region(&self, key: &RegionKey) -> Result<Arc<RawPipeline>, Error> {
        self.pipelines.get_or_make(key, || {
            let made = self.objects.pipeline(key, Notices::Announced)?;
            Ok(made.kept_at(&self.objects, kept::Place::Region(key.clone())))
        })
    }

    /// A dispatch that binds `formats` takes `made`, its `VkPipeline` of `key`: where `made` was
    /// made at its state's creation and this is its first dispatch, the notices its making would
    /// have announced are announced now, naming what the dispatch binds (the module docs).
    pub(super) fn first_use(&self, made: &RawPipeline, key: &RegionKey, formats: BoundFormats<'_>) {
        if made.first_drawn() {
            self.objects
                .device
                .shared()
                .counts
                .creation_pipeline_drawn(crate::work_counts::Predicted::Dispatch);
            self.announce(key, formats);
        }
    }

    /// Make the `VkPipeline` of the region a first dispatch of a state of `descriptor` on this
    /// variant, its base, is predicted to take (the module docs), where no caller made or began
    /// it: its notices wait for the first dispatch that takes it. Under
    /// `PipelineWaitPolicy::Bounded` handed to the device's pool, kept by `holds`, the state's,
    /// and waited for the policy's `pipeline_creation_wait` at most; under `Exact` made on this
    /// thread. A
    /// descriptor that requires no threadgroup size predicts no region at all, and a prediction
    /// the plan refuses or that runs on another variant makes nothing; a failed making is not
    /// kept and refuses nothing, its dispatch meeting it as before.
    pub(super) fn make_at_creation(
        self: &Arc<Self>,
        device: &Device,
        descriptor: &ComputePipelineDescriptor,
        holds: &Arc<PieceHolds<RegionHeld>>,
    ) {
        if crate::exit_guard::in_host_step() {
            return;
        }
        let Some(key) = self.predicted_region(device, descriptor) else {
            return;
        };
        let Begun::Yours(ticket) = self.pipelines.begin(&key) else {
            return;
        };
        let policy = device.compile_device().policy();
        if policy.pipeline_waits == PipelineWaitPolicy::Bounded {
            let pending = ticket.pending();
            pending.queued_at(self.pooled(holds, key, ticket, true));
            // A making still running at the bound goes on for the dispatches that take it.
            if pending
                .wait_until(Instant::now() + policy.pipeline_creation_wait)
                .is_none()
            {
                device.shared().counts.creation_left_making();
            }
            return;
        }
        let made = self
            .objects
            .pipeline(&key, Notices::FirstDraw)
            .map(|made| made.kept_at(&self.objects, kept::Place::Region(key.clone())));
        if made.is_ok() {
            device
                .shared()
                .counts
                .creation_pipeline_made(crate::work_counts::Predicted::Dispatch);
        }
        drop(ticket.settle(made));
    }

    /// The key of the region of the predicted first dispatch of `descriptor` on this variant
    /// (the module docs): one `dispatchThreadgroups:` threadgroup of [`predicted_threads`],
    /// planned for [`predicted_bindings`]. `None` where `descriptor` requires no threadgroup
    /// size, so the request names none to predict; and where it plans or instantiates no region,
    /// or runs on another variant (a threadgroup the kernel's simd tier cannot run whole).
    fn predicted_region(
        &self,
        device: &Device,
        descriptor: &ComputePipelineDescriptor,
    ) -> Option<RegionKey> {
        // The threadgroup first, before anything is planned: where the request states none there
        // is no region to predict, and the plan below would be built for a key nobody makes.
        let threads_per_group = predicted_threads(descriptor.required_threads_per_threadgroup)?;
        let state = predicted_bindings(&self.pipeline);
        let plan = self.plan(device, &state.shape()).ok()?;
        let dispatch = Dispatch::Threadgroups {
            groups: [1, 1, 1],
            threads_per_group,
        };
        if plan.dispatch_variant(&dispatch).ok()? != self.key {
            return None;
        }
        let mut commands = plan.instantiate_dispatch(&state, &dispatch).ok()?;
        self.keyed_regions(device, &mut commands);
        commands.regions.first().map(RegionKey::of)
    }

    /// Hand the making of `key`, `ticket`'s, to the device's pool, kept by `holds`: a
    /// dispatch's, which announced its notices, or the creation's where `at_creation`, whose
    /// notices wait for the first dispatch that takes it, counted as the creation's making. Where
    /// it waits there.
    fn pooled(
        self: &Arc<Self>,
        holds: &Arc<PieceHolds<RegionHeld>>,
        key: RegionKey,
        ticket: Ticket<RegionKey, RawPipeline>,
        at_creation: bool,
    ) -> Place {
        let keeping = holds.keep(RegionHeld::Making(Arc::clone(self), key, Box::new(ticket)));
        let device = self.objects.device.clone();
        device.shared().pipeline_pool.run(Box::new(move || {
            // Where the state let go meanwhile, the piece stops: its ticket went with the
            // state, failing the making.
            let Some((lease, held)) = keeping.start() else {
                return;
            };
            let (variant, key, ticket) = match held {
                RegionHeld::Making(variant, key, ticket) => (variant, key, *ticket),
                other => return lease.give_back(other, None),
            };
            let notices = match at_creation {
                true => Notices::FirstDraw,
                false => Notices::Announced,
            };
            let made = variant
                .objects
                .pipeline(&key, notices)
                .map(|made| made.kept_at(&variant.objects, kept::Place::Region(key.clone())));
            if at_creation && made.is_ok() {
                let counts = &variant.objects.device.shared().counts;
                counts.creation_pipeline_made(crate::work_counts::Predicted::Dispatch);
            }
            // Its callers take it from the making: from the settle, the batches that wait for it
            // may run and complete while this piece still holds the variant, and the device with
            // it, until it gave it back. A trail spans that (`trails.rs`).
            let trail = variant.objects.device.shared().trails.open();
            drop(ticket.settle(made));
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            super::pool::held::settled();
            lease.give_back(RegionHeld::Made { _variant: variant }, Some(trail));
        }))
    }
}

impl ComputePipelineState {
    /// The `VkPipeline` of `region` of `variant`, the state's, for a dispatch that binds
    /// `formats` and records it later ([`Variant::begin_region_pipeline`]), its making kept by
    /// the state: a compute encoder's dispatch, or a render encoder's tile dispatch of a tile
    /// function's state (`render/pending_tile.rs`).
    pub(crate) fn begin_region_pipeline(
        &self,
        variant: &Arc<Variant>,
        region: &DispatchRegion,
        formats: BoundFormats<'_>,
    ) -> RegionPipeline {
        let holds = &self.0.program.pieces;
        variant.begin_region_pipeline(holds, (region, self.is_tile()), formats)
    }
}

/// The threads per threadgroup a first dispatch of a state is predicted to take (the module
/// docs): its descriptor's `requiredThreadsPerThreadgroup` (`required`) where it states one,
/// which is then the threadgroup of every dispatch of the state; `None` where it states none
/// (Metal's all-zero size requires nothing), because the size is then the dispatch's own call and
/// no part of the request answers it. A size with one zero axis never reaches here: the creation
/// refuses it before the compile (`additions::check_kernel`, `RequiredThreadsProblem::EmptyAxis`).
fn predicted_threads(required: Option<[u32; 3]>) -> Option<[u32; 3]> {
    super::additions::required_size(required)
}

/// What the predicted first dispatch of `pipeline`'s kernel binds (the module docs): Metal's
/// default sampler (`MTLSamplerDescriptor`'s defaults) at each runtime sampler slot that does not
/// compare, and nothing else.
pub(super) fn predicted_bindings(pipeline: &ComputePipeline) -> BindState {
    let mut state = BindState::default();
    for (slot, used) in &pipeline.interface.slots {
        if let SlotInterface::Sampler(sampler) = used {
            if !sampler.compare {
                let table = state.stages.entry(slot.stage).or_default();
                table
                    .samplers
                    .insert(slot.index, BoundSampler::new(SamplerDescriptor::new()));
            }
        }
    }
    state
}

#[cfg(test)]
mod tests {
    use super::predicted_threads;

    /// The predicted threadgroup is the required size and nothing else: a state whose descriptor
    /// requires none predicts no region at all, so the creation makes no `VkPipeline` for a
    /// threadgroup the request never named. Measured over the whole conformance suite, such a
    /// guess was taken by 0 dispatches of 141 made (the module docs).
    #[test]
    fn the_predicted_threadgroup_is_the_required_one_and_nothing_is_predicted_without_one() {
        assert_eq!(predicted_threads(Some([8, 8, 1])), Some([8, 8, 1]));
        assert_eq!(predicted_threads(Some([1, 1, 1])), Some([1, 1, 1]));
        assert_eq!(predicted_threads(None), None);
        // Metal's all-zero size requires nothing, so it predicts nothing either.
        assert_eq!(predicted_threads(Some([0, 0, 0])), None);
    }
}
