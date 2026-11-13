//! What a device's recovery carries of a drawing state, a mesh one or a twin among them
//! (`pipeline/carry.rs`, whose chain of programs a drawing state keeps as a compute state does):
//! the same state object on the new `VkDevice`, its program made again there.
//!
//! - **What is made again.** Its base variant's objects, and its large-buffer form's where it has
//!   one, from the pipelines its creation compiled, or the live variants of them the new
//!   generation holds already ([`RenderVariant::made_again`]); then, as at its creation, its
//!   neutrals and its depth stand-in's copies, and the `VkPipeline` of its predicted first draw
//!   handed to the making pool, waited for by nothing (`GraphicsState::make_at_creation`). Its
//!   other variants, plans, settles, routings and `VkPipeline`s are made there as its draws need
//!   them.
//! - **What it keeps.** Its descriptors and their keys, its label and its added binary functions'
//!   counts; and its twins (its color groups', its single-sample and its factor-scaling states),
//!   each a state the recovery carries on its own. Its visible function tables are its lost
//!   generation's, as is every object it made there.

use super::super::carry::{
    compiled_otherwise, compiles_alike, generation_of, last, link, OfGeneration, StateCarried,
};
use super::{
    lock, pipeline_names, AtBindViews, CompiledVariant, GraphicsState, KeptShapes, Lru, PieceHolds,
    Program, ReadSlots, Remembered, RenderVariant, StateDescriptor, StateLog, TextureAccessMemo,
    LAST_PIPELINES, SETTLED, VARIANTS,
};
use crate::degradation::NoticesFor;
use crate::error::{Error, RuntimeGap};
use crate::Device;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

impl OfGeneration for Program {
    fn generation(&self) -> usize {
        generation_of(&self.device)
    }

    fn lineage(&self) -> &crate::device::line::Lineage {
        &self.device.shared().lineage
    }
}

impl GraphicsState {
    /// Kept by its generation for a recovery to carry, with its color groups' twins: carried by
    /// this maker where that generation's recovery took its objects before. A warm pass's state
    /// is no program's.
    pub(super) fn kept(&self, device: &Device) {
        if crate::warm::warming() {
            return;
        }
        let twins = self.0.color_groups.get().into_iter().flatten();
        for state in std::iter::once(self).chain(twins) {
            if device.shared().carry.render_state(state) {
                device.carry_late(|serving| state.carry(serving).unwrap_or(Ok(())));
            }
        }
    }

    /// Carry the state, made on a generation of its device that a recovery leaves, to `to`, the
    /// generation that recovery made (the module docs).
    pub(crate) fn carry(&self, to: &Device) -> StateCarried {
        let shared = &self.0;
        let (last, slot) = last(&shared.program, &shared.next);
        if last.generation() == generation_of(to) {
            return None;
        }
        let program = match last.made_again(to) {
            Ok(program) => program,
            Err(error) => {
                shared.remake_failed.store(true, Ordering::Release);
                return Some(Err(error));
            }
        };
        if !link(shared, &last.device, slot, program) {
            return None;
        }
        shared.carried.store(true, Ordering::Release);
        {
            // What its creation made, made on `to`: this thread reaches the program there, inside
            // whatever command buffer's work it does.
            let _reach = NoticesFor::of_generation(generation_of(to));
            self.make_neutrals();
            // A state drawn in color groups draws as its twins, which predict their own.
            if self.0.color_groups.get().is_none() {
                self.make_at_creation(false);
            }
            self.make_stand_in_copies();
        }
        // Kept there for a later recovery to carry on.
        if to.shared().carry.render_state(self) {
            to.carry_late(|serving| self.carry(serving).unwrap_or(Ok(())));
        }
        Some(Ok(()))
    }

    /// The device a bind of the state checks it against: its program's (`Device::foreign`
    /// answers a lost generation's `MadeBeforeRecovery`); refused where its making on the
    /// generation a recovery made failed.
    pub(crate) fn bound_device(&self) -> Result<&Device, Error> {
        match self.0.remake_failed.load(Ordering::Acquire) {
            false => Ok(self.device()),
            true => Err(Error::NotImplemented(RuntimeGap::RemadeAfterRecoveryFailed)),
        }
    }
}

impl Program {
    /// This program made again on `to`, the generation a recovery made (the module docs).
    /// Refused as its variants' objects' making refuses, and where `to` compiles otherwise.
    fn made_again(&self, to: &Device) -> Result<Program, Error> {
        if !compiles_alike(&self.device, to) {
            return Err(compiled_otherwise());
        }
        let base = self.base.made_again(to, &self.descriptor)?;
        let large = (self.large.as_ref())
            .map(|large| large.made_again(to, &self.descriptor))
            .transpose()?;
        let indirect = self.descriptor.support_indirect_command_buffers();
        let read = ReadSlots::of(base.shared_pipeline(), indirect);
        let at_bind_views = AtBindViews::new(to, &base.shared_pipeline().interface, read);
        let log = StateLog::new(
            self.label.as_deref(),
            &self.descriptor.function_names(),
            || self.descriptor.logged(),
        );
        Ok(Program {
            device: to.clone(),
            descriptor: self.descriptor.clone(),
            drawn: self.drawn.clone(),
            base,
            stand_in: self.stand_in,
            others: Mutex::new(Lru::counted(VARIANTS)),
            large,
            scaling: Mutex::new(lock(&self.scaling).clone()),
            added_unsupported: self.added_unsupported,
            single_sample: Mutex::new(lock(&self.single_sample).clone()),
            color_groups: self.color_groups.clone(),
            log,
            label: self.label.clone(),
            texture_accesses: TextureAccessMemo::new(),
            descriptor_key: self.descriptor_key.clone(),
            compiled_key: self.compiled_key.clone(),
            settled: Mutex::new(VecDeque::with_capacity(SETTLED)),
            made_tables: AtomicBool::new(false),
            last_pipelines: Mutex::new(VecDeque::with_capacity(LAST_PIPELINES)),
            routed: Mutex::new(None),
            pieces: PieceHolds::new(),
            at_bind_views,
            band_refusals: Mutex::new([None; 2]),
            kept_shapes: KeptShapes::default(),
        })
    }
}

impl RenderVariant {
    /// This variant made again on `to`, a generation a recovery made, of a state of
    /// `descriptor`: the live variant of its pipeline there where one of its key lives (the
    /// variant a state carried before made of it: one variant of a pipeline on a generation, as
    /// creations share it), else one made of its pipeline, its objects made there.
    fn made_again(
        &self,
        to: &Device,
        descriptor: &StateDescriptor,
    ) -> Result<Arc<RenderVariant>, Error> {
        let memo = &to.shared().shader_cache.memo;
        let found = (memo.render(&self.pipeline.key))
            .unwrap_or_else(|| Remembered::Compiled(Arc::clone(&self.pipeline)));
        RenderVariant::of(to, self.key.clone(), CompiledVariant::found(found), || {
            pipeline_names(descriptor.stages())
        })
    }
}
