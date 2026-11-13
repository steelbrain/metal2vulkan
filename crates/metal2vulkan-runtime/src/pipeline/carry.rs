//! What a device's recovery carries of the program's pipeline states (`device/carry.rs`): each
//! state the same object on the new `VkDevice`, its program made again there.
//!
//! - **What is made again.** A state's program (`Program`) holds its generation's objects: its
//!   variants' shader modules, layouts and `VkPipeline`s, its plans, its pool makings. The
//!   recovery makes the state's base variant again on the new generation from the pipeline its
//!   creation compiled, which the state holds, so nothing is translated again (a drawing state's
//!   large-buffer form's too); its other variants, plans and `VkPipeline`s are made there as its
//!   dispatches and draws need them, the device's memo of compiled pipelines carried before
//!   (`PipelineMemo::carry_from`) sparing their translations. The `VkPipeline` its creation
//!   predicted is queued on the device's making pool with no wait: a dispatch or draw that binds
//!   the state before it is made holds its command buffer for it, as at creation. A compiled
//!   pipeline is its compile profile's: where the new generation compiles otherwise
//!   ([`compiles_alike`]), no state is made again there.
//! - **What stays.** The state object: its handles, ID, `uniqueIdentifier`, label and sampler
//!   numbers. Its program on the next generation is linked after the lost one's
//!   ([`ProgramLink`]); its handles reach it from then on, and the work of a command buffer of
//!   the lost generation reaches that generation's ([`reached`]). Its program on the lost
//!   generation, and with it the state's hold on the lost `VkDevice`, goes at that generation's
//!   rest ([`LostProgram`], `device/carry.rs`): a call on the state outside command buffers' work
//!   holds its programs while it is under way ([`Programs::calls`]), so one that reached the lost
//!   program before the carry lets it go as it ends. Its label is the state's own copy and its
//!   `device` an owned handle, so no borrow the program holds reaches a program let go of.
//! - **A failed making** leaves the state's program the lost generation's: the recovery's notice
//!   counts it, and a bind of the state is refused (`RuntimeGap::RemadeAfterRecoveryFailed`).
//!
//! A compute state (and so a tile state's kernel) is carried here; a drawing state, mesh ones
//! among them, by `render/carry.rs`, its program's fields being its module's.

use super::memo::Remembered;
use super::{
    piece_holds, AtBindViews, ComputePipelineState, Kernel, Lru, Program, ReadSlots, Variant,
    PLANNED, VARIANTS,
};
use crate::degradation::NoticesFor;
use crate::device::carry::{LetGo, LostParts, PartsCalls};
use crate::error::{Error, RuntimeGap};
use crate::Device;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// A state's program on a generation a recovery carried the state to, and its program on the
/// generation after, once a later recovery carried it on. A program on a generation a later
/// recovery left is let go of at that generation's rest ([`LostProgram`]).
pub(crate) struct ProgramLink<P> {
    program: LetGo<Arc<P>>,
    next: OnceLock<Box<ProgramLink<P>>>,
}

/// A state whose programs a recovery carries: a compute state's (`StateShared`) or a drawing
/// state's (`render.rs`'s).
pub(crate) trait Programs: Send + Sync + 'static {
    type Program: OfGeneration + Send + Sync + 'static;

    /// Its program on the generation it was made on, and the links of its programs on the later
    /// ones.
    fn programs(&self) -> (&LetGo<Arc<Self::Program>>, &ProgramLinks<Self::Program>);

    /// The calls outside command buffers' work under way on the state, which hold every program
    /// they may reach (`PartsCalls::hold`): its public calls, a handle of what it runs taken
    /// (`ComputePipelineState::program`), and a bind of it in an argument buffer.
    fn calls(&self) -> &PartsCalls;
}

/// The links of a state's programs on the generations its device's recoveries carried it to.
pub(crate) type ProgramLinks<P> = OnceLock<Box<ProgramLink<P>>>;

/// A carried state's program on a lost generation (`device/carry.rs`): its variants' objects
/// there, its plans and makings, and its hold on the generation's `VkDevice`, let go of at the
/// generation's rest, the state living on its later programs. Once no call under way on the
/// state holds it ([`Programs::calls`]): a call that reached it before the carry lets it go as it
/// ends. Counted: `WorkCounts::lost_programs_released`.
struct LostProgram<S> {
    state: Weak<S>,
    /// The lost generation's address ([`generation_of`]).
    generation: usize,
}

impl<S: Programs> LostParts for LostProgram<S> {
    fn release(self: Box<Self>) {
        // The state went, and its programs with it.
        let Some(state) = self.state.upgrade() else {
            return;
        };
        // The last call under way on the state lets it go.
        let Some(lost) = state.calls().unless_held(self) else {
            return;
        };
        let (first, next) = state.programs();
        let (mut program, mut next) = (first, next);
        loop {
            if !program.gone() && program.generation() == lost.generation {
                // Counted on the line, which outlives the generation the program may hold last.
                program.lineage().program_released();
                // SAFETY: the state was carried past the program's generation, which is at rest:
                // none of its command buffers records ([`reached`] skips it from now on), no call
                // under way on the state reached it (`PartsCalls::unless_held`), and every other
                // reach goes to a later program (`reached`, `last`), its label and device the
                // state's own copies, so no borrow of it is left.
                unsafe { program.let_go() };
                return;
            }
            match next.get() {
                Some(link) => (program, next) = (&link.program, &link.next),
                None => return,
            }
        }
    }
}

/// Hand the registry of `from`, the lost generation `state` was carried from, its program there,
/// to let go of at the generation's rest ([`LostProgram`]).
fn lost<S: Programs>(state: &Arc<S>, from: &Device) {
    let parts = Box::new(LostProgram {
        state: Arc::downgrade(state),
        generation: generation_of(from),
    });
    from.shared().carry.lost(from.shared(), parts);
}

/// A program of one generation of its device: the address of the generation's `DeviceShared`,
/// which the work of its command buffers names (`NoticesFor::generation`).
pub(crate) trait OfGeneration {
    fn generation(&self) -> usize;

    /// Its device's line's counts (`WorkCounts`).
    fn lineage(&self) -> &crate::device::line::Lineage;
}

/// The address of `device`'s generation.
pub(crate) fn generation_of(device: &Device) -> usize {
    std::ptr::from_ref(device.shared()).addr()
}

/// Whether `to`, a generation a recovery made, compiles as `from`, the lost one, does: one compile
/// profile, so what one compiled the other takes (the memo's pipelines, a state's own).
pub(crate) fn compiles_alike(from: &Device, to: &Device) -> bool {
    let profile = |device: &Device| device.compile_device().profile_fingerprint();
    profile(from) == profile(to)
}

/// What a state's making again on a generation that compiles otherwise answers
/// ([`compiles_alike`]): refused, as its compiled pipeline is not that generation's.
pub(crate) fn compiled_otherwise() -> Error {
    Error::NotImplemented(RuntimeGap::RemadeAfterRecoveryFailed)
}

impl OfGeneration for Program {
    fn generation(&self) -> usize {
        generation_of(&self.device)
    }

    fn lineage(&self) -> &crate::device::line::Lineage {
        &self.device.shared().lineage
    }
}

/// The program of a state whose first program is `first` and whose later ones follow `next`
/// that this thread reaches: inside a command buffer's work, that command buffer's
/// generation's, where the state has one there; else the last. Never one let go of: no command
/// buffer of its generation records once it is ([`LostProgram`]), and the last is never one.
pub(crate) fn reached<'a, P: OfGeneration>(
    first: &'a LetGo<Arc<P>>,
    next: &'a ProgramLinks<P>,
) -> &'a Arc<P> {
    let records = NoticesFor::generation();
    let (mut program, mut next) = (first, next);
    loop {
        if !program.gone() && records == Some(program.generation()) {
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
    first: &'a LetGo<Arc<P>>,
    next: &'a ProgramLinks<P>,
) -> (&'a Arc<P>, &'a ProgramLinks<P>) {
    let (mut program, mut next) = (first, next);
    while let Some(link) = next.get() {
        (program, next) = (&link.program, &link.next);
    }
    (program, next)
}

/// `program`, linked into `slot` (the one [`last`] answered) of `state`, the program before it
/// handed `from`'s registry, the generation it is of, to let go of at that generation's rest
/// ([`LostProgram`]): false where another carry linked one first.
pub(crate) fn link<S: Programs>(
    state: &Arc<S>,
    from: &Device,
    slot: &ProgramLinks<S::Program>,
    program: S::Program,
) -> bool {
    let link = ProgramLink {
        program: LetGo::new(Arc::new(program)),
        next: OnceLock::new(),
    };
    if slot.set(Box::new(link)).is_err() {
        return false;
    }
    lost(state, from);
    true
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
        if !link(shared, &last.device, slot, program) {
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
    /// generation a recovery made failed. A borrow of the program a command buffer's work or a
    /// call that holds the state ([`Programs::calls`]) reads.
    pub(crate) fn bound_device(&self) -> Result<&Device, Error> {
        match self.0.remake_failed.load(Ordering::Acquire) {
            false => Ok(&self.0.device),
            true => Err(Error::NotImplemented(RuntimeGap::RemadeAfterRecoveryFailed)),
        }
    }
}

impl Program {
    /// This program made again on `to`, the generation a recovery made (the module docs): its
    /// base variant's objects made there from its compiled pipeline, its predicted `VkPipeline`
    /// queued on the making pool with no wait. Refused as the objects' making refuses, and where
    /// `to` compiles otherwise.
    fn made_again(&self, to: &Device) -> Result<Program, Error> {
        if !compiles_alike(&self.device, to) {
            return Err(compiled_otherwise());
        }
        // The variant a state carried before made of the same pipeline there, where it lives: one
        // variant of a pipeline on a generation, as its creations share it.
        let pipeline = &self.base.pipeline;
        let memo = &to.shared().shader_cache.memo;
        let found = (memo.compute(&pipeline.key))
            .unwrap_or_else(|| Remembered::Compiled(Arc::clone(pipeline)));
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
            call_flat_refused: AtomicBool::new(false),
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
