//! What a device's recovery carries of the program's pipeline states (`device/carry.rs`): each
//! state the same object on the new `VkDevice`, its program made again there.
//!
//! - **What is made again.** A state's program (`Program`) holds its generation's objects: its
//!   variants' shader modules, layouts and `VkPipeline`s, its plans, its pool makings. The
//!   recovery makes the state's base variant again on the new generation from the pipeline its
//!   creation compiled, which the state holds, so nothing is translated again; its other
//!   variants, plans and `VkPipeline`s are made there as its dispatches and draws need them, the
//!   device's memo of compiled pipelines carried before (`PipelineMemo::carry_from`) sparing
//!   their translations. The `VkPipeline` its creation predicted is queued on the device's
//!   making pool with no wait: a dispatch or draw that binds the state before it is made holds
//!   its command buffer for it, as at creation.
//! - **What stays.** The state object: its handles, ID, `uniqueIdentifier`, label and sampler
//!   numbers. Its program on the next generation is linked after the lost one's
//!   ([`ProgramLink`]); its handles reach it from then on, and the work of a command buffer of
//!   the lost generation reaches that generation's ([`reached`]).
//! - **A failed making** leaves the state's program the lost generation's: the recovery's notice
//!   counts it, and a bind of the state is refused (`RuntimeGap::RemadeAfterRecoveryFailed`).

use super::memo::Remembered;
use super::{
    piece_holds, AtBindViews, ComputePipelineState, Kernel, Lru, Program, ReadSlots, Variant,
    PLANNED, VARIANTS,
};
use crate::degradation::NoticesFor;
use crate::error::{Error, RuntimeGap};
use crate::Device;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// A state's program on a generation a recovery carried the state to, and its program on the
/// generation after, once a later recovery carried it on.
pub(crate) struct ProgramLink<P> {
    program: Arc<P>,
    next: OnceLock<Box<ProgramLink<P>>>,
}

/// A program of one generation of its device: the address of the generation's `DeviceShared`,
/// which the work of its command buffers names (`NoticesFor::generation`).
pub(crate) trait OfGeneration {
    fn generation(&self) -> usize;
}

/// The address of `device`'s generation.
pub(crate) fn generation_of(device: &Device) -> usize {
    std::ptr::from_ref(device.shared()).addr()
}

impl OfGeneration for Program {
    fn generation(&self) -> usize {
        generation_of(&self.device)
    }
}

/// The program of a state whose first program is `first` and whose later ones follow `next`
/// that this thread reaches: inside a command buffer's work, that command buffer's
/// generation's, where the state has one there; else the last.
pub(crate) fn reached<'a, P: OfGeneration>(
    first: &'a Arc<P>,
    next: &'a OnceLock<Box<ProgramLink<P>>>,
) -> &'a Arc<P> {
    let records = NoticesFor::generation();
    let (mut program, mut next) = (first, next);
    loop {
        if records == Some(program.generation()) {
            return program;
        }
        match next.get() {
            Some(link) => (program, next) = (&link.program, &link.next),
            None => return program,
        }
    }
}

/// The last program of a state whose first program is `first` and whose later ones follow
/// `next`, with the slot the program of the generation after it goes in.
pub(crate) fn last<'a, P>(
    first: &'a Arc<P>,
    next: &'a OnceLock<Box<ProgramLink<P>>>,
) -> (&'a Arc<P>, &'a OnceLock<Box<ProgramLink<P>>>) {
    let (mut program, mut next) = (first, next);
    while let Some(link) = next.get() {
        (program, next) = (&link.program, &link.next);
    }
    (program, next)
}

/// `program`, linked into `slot` (the one [`last`] answered): false where another carry linked
/// one first.
pub(crate) fn link<P>(slot: &OnceLock<Box<ProgramLink<P>>>, program: P) -> bool {
    let link = ProgramLink {
        program: Arc::new(program),
        next: OnceLock::new(),
    };
    slot.set(Box::new(link)).is_ok()
}

/// What one state's carry did, for the recovery's notice: made again there, or its making
/// there failed. `None` where it was carried there already.
pub(crate) type StateCarried = Option<Result<(), Error>>;

impl ComputePipelineState {
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
        if !link(slot, program) {
            return None;
        }
        shared.carried.store(true, Ordering::Release);
        // Kept there for a later recovery to carry on.
        if to.shared().carry.compute_state(self) {
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
    /// This program made again on `to`, the generation a recovery made (the module docs): its
    /// base variant's objects made there from its compiled pipeline, its predicted `VkPipeline`
    /// queued on the making pool with no wait. Refused as the objects' making refuses.
    fn made_again(&self, to: &Device) -> Result<Program, Error> {
        let found = Remembered::Compiled(Arc::clone(&self.base.pipeline));
        let base = Variant::of(to, self.base.key.clone(), found, &self.function)?;
        let pieces = piece_holds::PieceHolds::new();
        if let (Kernel::Compute(descriptor), false) = (&self.descriptor, self.stand_in) {
            base.make_predicted(to, descriptor, &pieces, false);
        }
        let interface = &base.shared_pipeline().interface;
        let at_bind_views = AtBindViews::new(to, interface, ReadSlots::every());
        Ok(Program {
            device: to.clone(),
            function: self.function.clone(),
            descriptor: self.descriptor.clone(),
            compiled: self.compiled.clone(),
            base,
            stand_in: self.stand_in,
            others: Mutex::new(Lru::counted(VARIANTS)),
            flat_refused: AtomicBool::new(false),
            replay_refused: AtomicBool::new(false),
            folded_refused: AtomicBool::new(false),
            added_unsupported: self.added_unsupported,
            label: self.label.clone(),
            pieces,
            at_bind_views,
            planned: Mutex::new(VecDeque::with_capacity(PLANNED)),
            dispatch_records: OnceLock::new(),
        })
    }
}
