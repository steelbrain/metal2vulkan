//! A kernel's unit-axes variant (`VariantKey::unit_axes`), made on the device's pool ahead of the
//! dispatches that take it.
//!
//! - **What it is.** A kernel whose emulated simdgroups are walked predicated, where its threads'
//!   positions along some axes alone steer a synchronization (`RuntimeFact::UnitAxes`), runs a
//!   dispatch of threadgroups of one thread along such axes on a variant that takes those
//!   positions as uniform, as Metal's contract makes them there (`threads_per_threadgroup` 1:
//!   every thread's `thread_position_in_threadgroup` is 0 and its `thread_position_in_grid` its
//!   threadgroup's position, partial threadgroups included). Its walk drops the live flags,
//!   demotions and loop votes those blocks took; it gives the plain variant's answers. The plan
//!   offers it (`BindingPlan::unit_axes_variant`).
//! - **Never on the exec path.** A dispatch takes the variant only where its translation is
//!   done and each of its regions' `VkPipeline`s is made ([`ComputePipelineState::unit_axes`]).
//!   Otherwise it settles on its plain variant (`WorkCounts::unit_axes_meanwhile`), and the
//!   variant is begun on the pool: its translation, a piece of its own kept by the state
//!   (`piece_holds.rs`), at the first such dispatch (`WorkCounts::unit_axes_variants_begun`, its
//!   dispatch recorded for a later process's warm pass, `dispatch_record.rs`); at the next, on
//!   the program's thread, its objects (shader module and layouts, as any variant's) with its
//!   pipeline's notices, since no piece announces any (`degradation.rs`), and its regions'
//!   makings, handed to the pool as any dispatch's are (`dispatch_making.rs`), their notices
//!   announced there, naming what it binds. Nothing waits for the translation or a making. What a
//!   dispatch planned meanwhile is kept neither by the state's latest plans nor by its encoder's
//!   memo, so the next dispatch of its inputs asks again, and takes the variant once it is made
//!   (`WorkCounts::unit_axes_dispatches`).
//! - **Bounded.** A variant per plain variant and mask of unit axes that relieves it: at most
//!   seven, one per threadgroup shape's axes in practice. They live among the state's other
//!   variants, at most `VARIANTS`, the least recently used going first; the state tracks at most
//!   `VARIANTS` of them being made or refused at once, and a dispatch whose variant finds that
//!   full keeps its plain variant. A translation the compile layer refuses and a making that
//!   fails are remembered: the plain variant runs those dispatches, as it did before.

use super::dispatch_making::{RegionHeld, RegionPipeline};
use super::making::Pending;
use super::memo::Remembered;
use super::objects::RawPipeline;
use super::sources::Sources;
use super::{ComputePipelineState, Kernel, Variant, VARIANTS};
use crate::degradation::BoundFormats;
use crate::device::lock;
use crate::error::{internal, Error};
use crate::Device;
use metal2vulkan::bind::{BindShape, BindState, BindingPlan, Dispatch};
use metal2vulkan::cache::VariantKey;
use metal2vulkan::pipeline::{ComputePipeline, FunctionDescriptor};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// What a state knows of a unit-axes variant it began and does not hold among its variants yet.
pub(super) enum Ahead {
    /// Its translation is on the pool.
    Compiling,
    /// Its translation is done, by the piece that put it here: its pipeline, or the live variant
    /// of it the device's memo knew.
    Compiled(Remembered<ComputePipeline, Variant>),
    /// Its regions' `VkPipeline`s are being made, for the dispatch that began them.
    Making(Vec<Pending<RawPipeline>>),
    /// The compile layer refused it, or a making failed.
    Refused,
}

/// A state's unit-axes variants begun, by key: shared with the translation's piece, which hands
/// its variant over here.
pub(super) type Aheads = Arc<Mutex<HashMap<VariantKey, Ahead>>>;

/// What a unit-axes variant's translation on the pool needs: kept by the state
/// (`piece_holds.rs`), so the pool's thread never holds the device.
pub(crate) struct CompilePiece {
    device: Device,
    function: FunctionDescriptor,
    descriptor: Kernel,
    key: VariantKey,
    aheads: Aheads,
}

/// What a direct dispatch settled (`compute::settle_direct`): the variant it runs on and its plan
/// for its bind state's shape; and whether that is its plain variant for now alone, its unit-axes
/// variant being made meanwhile, so no later dispatch takes what it planned.
pub(crate) struct Direct {
    pub(crate) variant: Arc<Variant>,
    pub(crate) plan: Arc<BindingPlan>,
    pub(crate) meanwhile: bool,
}

impl Direct {
    /// `variant` and its `plan`, settled for good.
    pub(crate) fn of((variant, plan): (Arc<Variant>, Arc<BindingPlan>)) -> Direct {
        Direct {
            variant,
            plan,
            meanwhile: false,
        }
    }
}

/// What a dispatch does about a unit-axes variant (`ComputePipelineState::ahead_of`).
enum Step {
    /// Keeps its plain variant: the variant is refused, or the state tracks as many as it may.
    Plain,
    /// Keeps its plain variant while the variant is being made.
    Meanwhile,
    /// Takes the variant where its regions' `VkPipeline`s are made.
    Compiled(Arc<Variant>),
    /// Keeps its plain variant and begins the variant's translation.
    Begin,
}

/// Whether every one of `pending`'s makings succeeded (`Some(true)`), one failed (`Some(false)`),
/// or one is still being made (`None`).
fn made(pending: &[Pending<RawPipeline>]) -> Option<bool> {
    let mut all = Some(true);
    for making in pending {
        match making.outcome() {
            Some(Ok(_)) => {}
            Some(Err(_)) => return Some(false),
            None => all = None,
        }
    }
    all
}

impl ComputePipelineState {
    /// The variant a direct dispatch of `dispatch` over `state`, of shape `shape`, that binds
    /// `formats`, runs on, `settled` (its variant and plan) otherwise: its unit-axes variant
    /// where `settled`'s plan offers one (`BindingPlan::unit_axes_variant`) and it is made, its
    /// translation and each of its regions' `VkPipeline`s (the module docs); else `settled`,
    /// for now alone where the variant is being made, which this dispatch may begin.
    pub(crate) fn unit_axes(
        &self,
        settled: (Arc<Variant>, Arc<BindingPlan>),
        (shape, dispatch): (&BindShape, &Dispatch),
        (state, formats): (&BindState, BoundFormats<'_>),
    ) -> Result<Direct, Error> {
        let Some(key) = settled.1.unit_axes_variant(dispatch).map_err(Error::Bind)? else {
            return Ok(Direct::of(settled));
        };
        let device = &self.0.device;
        let counts = &device.shared().counts;
        let meanwhile = |(variant, plan): (Arc<Variant>, Arc<BindingPlan>)| {
            counts.unit_axes_meanwhile();
            Ok(Direct {
                variant,
                plan,
                meanwhile: true,
            })
        };
        let variant = match self.ahead_of(&key) {
            Step::Plain => return Ok(Direct::of(settled)),
            Step::Meanwhile => return meanwhile(settled),
            Step::Begin => {
                self.begin_unit_axes(key);
                counts.unit_axes_variant_begun();
                self.record_dispatch(dispatch, state);
                return meanwhile(settled);
            }
            Step::Compiled(variant) => variant,
        };
        let plan = variant.plan(device, shape)?;
        if plan.dispatch_variant(dispatch).map_err(Error::Bind)? != key {
            return Err(internal(
                "the plan of a unit-axes variant asks for another variant",
            ));
        }
        // A dispatch the variant's plan refuses meets the plain variant's own refusal.
        let Ok(mut commands) = plan.instantiate_dispatch(state, dispatch) else {
            return Ok(Direct::of(settled));
        };
        variant.keyed_regions(device, &mut commands);
        let pending: Vec<_> = (commands.regions.iter())
            .filter_map(
                |region| match self.begin_region_pipeline(&variant, region, formats) {
                    RegionPipeline::Made(_) => None,
                    RegionPipeline::Making(pending) => Some(pending),
                },
            )
            .collect();
        if pending.is_empty() {
            counts.unit_axes_dispatch();
            return Ok(Direct {
                variant,
                plan,
                meanwhile: false,
            });
        }
        let mut aheads = lock(&self.0.unit_axes);
        if aheads.len() >= VARIANTS {
            return Ok(Direct::of(settled));
        }
        aheads.insert(key, Ahead::Making(pending));
        drop(aheads);
        meanwhile(settled)
    }

    /// What a dispatch whose plan offers the unit-axes variant of `key` does about it (the
    /// module docs): a variant the piece translated is made here, on the program's thread, and
    /// joins the state's variants.
    fn ahead_of(&self, key: &VariantKey) -> Step {
        let mut aheads = lock(&self.0.unit_axes);
        let compiled = match aheads.get(key) {
            Some(Ahead::Compiling) => return Step::Meanwhile,
            Some(Ahead::Refused) => return Step::Plain,
            Some(Ahead::Making(pending)) => match made(pending) {
                None => return Step::Meanwhile,
                Some(false) => {
                    aheads.insert(key.clone(), Ahead::Refused);
                    return Step::Plain;
                }
                Some(true) => {
                    aheads.remove(key);
                    None
                }
            },
            Some(Ahead::Compiled(_)) => match aheads.remove(key) {
                Some(Ahead::Compiled(found)) => Some(found),
                _ => None,
            },
            None => None,
        };
        // Made under the lock, so no other dispatch begins it again meanwhile.
        if let Some(found) = compiled {
            return match Variant::of(&self.0.device, key.clone(), found, &self.0.function) {
                Ok(variant) => {
                    lock(&self.0.others).insert(key.clone(), Arc::clone(&variant), 1);
                    Step::Compiled(variant)
                }
                Err(_) => {
                    aheads.insert(key.clone(), Ahead::Refused);
                    Step::Plain
                }
            };
        }
        if let Some(found) = lock(&self.0.others).get(key) {
            return Step::Compiled(Arc::clone(found));
        }
        if aheads.len() >= VARIANTS {
            return Step::Plain;
        }
        aheads.insert(key.clone(), Ahead::Compiling);
        Step::Begin
    }

    /// Hand the translation of the unit-axes variant of `key` to the device's pool, kept by the
    /// state: what it compiles, or its refusal, goes to the state's [`Aheads`]. The translation
    /// announces nothing; the variant's objects and notices wait for the next dispatch
    /// ([`ahead_of`](Self::ahead_of)).
    fn begin_unit_axes(&self, key: VariantKey) {
        let piece = CompilePiece {
            device: self.0.device.clone(),
            function: self.0.function.clone(),
            descriptor: self.0.descriptor.clone(),
            key,
            aheads: Arc::clone(&self.0.unit_axes),
        };
        let keeping = (self.0.program().pieces).keep(RegionHeld::Compile(Box::new(piece)));
        let pool = &self.0.device.shared().pipeline_pool;
        drop(pool.run(Box::new(move || {
            // Where the state let go meanwhile, the piece stops.
            let Some((lease, held)) = keeping.start() else {
                return;
            };
            let piece = match held {
                RegionHeld::Compile(piece) => piece,
                other => return lease.give_back(other, None),
            };
            let ahead = {
                let _noted = super::NoteEntries(&piece.device);
                let sources = Sources::later(&piece.device);
                match Variant::compiled(&sources, &piece.function, &piece.descriptor, &piece.key) {
                    Ok((found, _)) => Ahead::Compiled(found),
                    Err(_) => Ahead::Refused,
                }
            };
            lock(&piece.aheads).insert(piece.key.clone(), ahead);
            lease.give_back(RegionHeld::Compile(piece), None);
        })));
    }

    /// On a warm pass's thread, the unit-axes variant `plan` offers a dispatch of `dispatch` of a
    /// bind state of shape `shape`, compiled where it is not among the state's variants, and its
    /// plan for `shape`: `None` where `plan` offers none, or the compile layer refuses it
    /// (`dispatch_record.rs`).
    pub(super) fn warm_unit_axes(
        &self,
        plan: &BindingPlan,
        shape: &BindShape,
        dispatch: &Dispatch,
    ) -> Option<(Arc<Variant>, Arc<BindingPlan>)> {
        let key = plan.unit_axes_variant(dispatch).ok()??;
        let variant = self.variant(&key).ok()?;
        let plan = variant.plan(&self.0.device, shape).ok()?;
        Some((variant, plan))
    }
}
