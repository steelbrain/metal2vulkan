//! The programs of released pipeline states that records the GPU wrote still run.
//!
//! - **What Metal runs** (an M3 Max on macOS 27, plain Metal, a kernel encoding a compute command
//!   through an argument buffer that names a state made with `supportIndirectCommandBuffers`). A
//!   GPU write of a pipeline state ID resolves the ID as it runs, to the state that took it last,
//!   released or not: a state released before the write, its ID taken by no state since, runs.
//!   The record runs that state's program, and its `getPipelineStateUniqueIdentifier` answers
//!   that state's, from then on: after the program releases the state, after another state takes
//!   the ID, and after four rounds of 48 states took the ID and gave it back. A write of an ID
//!   another state took before the write runs that state.
//!   The M3 Max and an Intel UHD 630 run a state released before the write, its ID taken by no
//!   state, with no error; a Radeon Pro 5300M fails the command buffer (`Timeout`) and runs
//!   nothing (the consumer's oracle, two runs each): the runtime runs it, as the first two do.
//! - **The runtime.** A record holds the ID alone, and names the state that held the ID as the
//!   GPU last could write the buffer's records (`IndirectCommandBuffer::window_holder`, which the
//!   getter answers too), where a kernel wrote its pipeline (`CommandRecord::kernel_pipeline`;
//!   one the CPU set names the state the CPU set). A buffer the GPU may write keeps, from its first
//!   such write (`IndirectCommandBuffer::gpu_write`), the program of each state made with
//!   `supportIndirectCommandBuffers` the program releases since ([`Keepers::released`]), and
//!   those the device's freed IDs keep then ([`FreedPrograms`]); a blit's copy of commands keeps in
//!   its destination the source's kept programs and those of the released states the CPU set on
//!   the commands it copies: a record naming the ID runs that program where the state held the ID
//!   then. The device keeps, for each pipeline state ID whose last holder the program released and
//!   no state took since, that holder's program ([`FreedPrograms`]), while an indirect command
//!   buffer of it lives: every such buffer holds the store, the device holds it weakly, as a
//!   program holds its device.
//! - **Bounded.** A buffer keeps an ID's last [`PIPELINE_HOLDERS`] released programs, as the
//!   device keeps its last holders: a fifth drops the oldest, whose records then run nothing,
//!   announced (below). So what the buffers keep is at most four programs per pipeline state ID
//!   the program used, each one `Arc` the buffers share: with plain allocation, which hands out
//!   the lowest free ID as Metal does, four times the peak count of live states made with
//!   `supportIndirectCommandBuffers`, however long the program makes and releases them; each ID a
//!   placement (`PipelineOptions::placement`) names besides counts four more. The freed IDs keep
//!   one program each, which a state that takes the ID lets go: at most the peak count of live
//!   such states, and one more for each placed ID, whatever the count of buffers, which share it.
//! - **Not kept** (`Degradation::ReleasedCommandState`, the command running nothing): the program
//!   of a state released while no indirect command buffer of the device lived (or on a generation
//!   a recovery replaced, for a buffer the GPU first writes after it), which Metal runs where no
//!   state took the ID since; and one past an ID's last [`PIPELINE_HOLDERS`] holders.

use super::device::lock;
use super::pipeline::{ComputePipelineState, ComputeProgram, RenderPipelineState, RenderProgram};
use super::resource_ids::PIPELINE_HOLDERS;
use super::Device;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, Weak};

/// A released pipeline state's program.
#[derive(Clone)]
pub(super) enum KeptProgram {
    Compute(ComputeProgram),
    Render(RenderProgram),
}

impl KeptProgram {
    /// A compute program, as a state of no ID (`ComputeProgram::state`).
    pub(super) fn compute(&self) -> Option<ComputePipelineState> {
        match self {
            KeptProgram::Compute(program) => Some(program.state()),
            KeptProgram::Render(_) => None,
        }
    }

    /// A render program, as a state of no ID (`RenderProgram::state`).
    pub(super) fn render(&self) -> Option<RenderPipelineState> {
        match self {
            KeptProgram::Render(program) => Some(program.state()),
            KeptProgram::Compute(_) => None,
        }
    }

    /// The program made again on `to`, the generation a recovery made, for a buffer it carries
    /// (`indirect_command_buffer/carry.rs`); this one where its making there fails, refused where
    /// bound (`RuntimeGap::MadeBeforeRecovery`).
    fn made_again(&self, to: &Device) -> KeptProgram {
        match self {
            KeptProgram::Compute(program) => {
                KeptProgram::Compute(program.made_again(to).unwrap_or_else(|_| program.clone()))
            }
            KeptProgram::Render(program) => {
                KeptProgram::Render(program.made_again(to).unwrap_or_else(|_| program.clone()))
            }
        }
    }
}

/// The programs a buffer keeps: by pipeline state ID, each with its state's hold serial
/// (`IdHold::serial`), the last kept last, at most [`PIPELINE_HOLDERS`] of an ID.
#[derive(Clone, Default)]
pub(super) struct KeptPrograms(BTreeMap<u64, VecDeque<(u64, KeptProgram)>>);

impl KeptPrograms {
    /// Keep `program`, of the state whose hold of ID `id` had `serial`.
    pub(super) fn keep(&mut self, id: u64, serial: u64, program: KeptProgram) {
        let kept = self.0.entry(id).or_default();
        if kept.iter().any(|&(held, _)| held == serial) {
            return;
        }
        if kept.len() == PIPELINE_HOLDERS {
            kept.pop_front();
        }
        kept.push_back((serial, program));
    }

    /// Keep what `other` keeps.
    pub(super) fn keep_all(&mut self, other: KeptPrograms) {
        for (id, kept) in other.0 {
            for (serial, program) in kept {
                self.keep(id, serial, program);
            }
        }
    }

    /// The program of the state whose hold of ID `id` had `serial`, where kept.
    pub(super) fn program(&self, id: u64, serial: u64) -> Option<&KeptProgram> {
        self.0
            .get(&id)?
            .iter()
            .find(|&&(held, _)| held == serial)
            .map(|(_, program)| program)
    }

    /// The IDs it keeps programs of.
    pub(super) fn ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.0.keys().copied()
    }

    /// Each program it keeps made again on `to`, the generation a recovery made, for the buffer
    /// it carries ([`KeptProgram::made_again`]).
    pub(super) fn carry(&mut self, to: &Device) {
        for kept in self.0.values_mut() {
            for (_, program) in kept.iter_mut() {
                *program = program.made_again(to);
            }
        }
    }

    /// How many programs it keeps.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(super) fn count(&self) -> usize {
        self.0.values().map(VecDeque::len).sum()
    }
}

/// The program of the last holder of each pipeline state ID the program released while no state
/// took the ID since, with its hold serial (`IdHold::serial`): one an ID. Every live indirect
/// command buffer of the device holds it, and a buffer the GPU may write keeps what it holds at
/// its first such write ([`Keepers::add`]).
#[derive(Default)]
pub(super) struct FreedPrograms(Mutex<BTreeMap<u64, (u64, KeptProgram)>>);

impl FreedPrograms {
    /// What it holds, as a buffer keeps it.
    fn programs(&self) -> KeptPrograms {
        let mut programs = KeptPrograms::default();
        for (&id, (serial, program)) in lock(&self.0).iter() {
            programs.keep(id, *serial, program.clone());
        }
        programs
    }
}

/// The buffers of a device whose records the GPU may write, each by what it keeps, and its freed
/// IDs' programs, each weakly: the buffers hold the programs, so the device holds no program (a
/// program holds its device).
#[derive(Default)]
pub(super) struct Keepers {
    buffers: Mutex<Vec<Weak<Mutex<KeptPrograms>>>>,
    freed: Mutex<Weak<FreedPrograms>>,
}

impl Keepers {
    /// The device's freed IDs' programs, for an indirect command buffer made now to hold for its
    /// life: the live store, or a new one, empty, where no buffer holds one.
    pub(super) fn freed(&self) -> Arc<FreedPrograms> {
        let mut freed = lock(&self.freed);
        match freed.upgrade() {
            Some(held) => held,
            None => {
                let held = Arc::new(FreedPrograms::default());
                *freed = Arc::downgrade(&held);
                held
            }
        }
    }

    /// From now on keep, in `kept`, the programs of the states the program releases, and now
    /// those `freed`, the device's freed IDs', holds.
    pub(super) fn add(&self, kept: &Arc<Mutex<KeptPrograms>>, freed: &FreedPrograms) {
        {
            let mut keepers = lock(&self.buffers);
            keepers.retain(|keeper| keeper.strong_count() > 0);
            keepers.push(Arc::downgrade(kept));
        }
        // After the buffer is listed: a release meanwhile, which a freed ID holds before it lists
        // the buffers (`released`), is in one or both, and kept once.
        let programs = freed.programs();
        lock(kept).keep_all(programs);
    }

    /// The program released a state made with `supportIndirectCommandBuffers` whose hold of ID
    /// `id` had `serial`, and which runs `program`: the ID's freed program is it, and each buffer
    /// whose records the GPU may write keeps it. Nothing, and `program` is not called, where no
    /// buffer holds the freed IDs' programs and none keeps.
    pub(super) fn released(&self, id: u64, serial: u64, program: impl FnOnce() -> KeptProgram) {
        // The freed ID's before the buffers are listed (`add`); the program made once, where
        // either keeps it.
        let freed = lock(&self.freed).upgrade();
        let program = match freed {
            Some(freed) => {
                let program = program();
                // Let go of with no lock held.
                let _gone = { lock(&freed.0).insert(id, (serial, program.clone())) };
                Ok(program)
            }
            None => Err(program),
        };
        let keepers: Vec<Arc<Mutex<KeptPrograms>>> = {
            let mut keepers = lock(&self.buffers);
            keepers.retain(|keeper| keeper.strong_count() > 0);
            keepers.iter().filter_map(Weak::upgrade).collect()
        };
        if keepers.is_empty() {
            return;
        }
        let program = program.unwrap_or_else(|make| make());
        for keeper in keepers {
            lock(&keeper).keep(id, serial, program.clone());
        }
    }

    /// A state took pipeline state ID `id`: a record naming it runs that one from now on, so the
    /// program of the ID's last released holder goes from the freed IDs'.
    pub(super) fn taken(&self, id: u64) {
        let Some(freed) = lock(&self.freed).upgrade() else {
            return;
        };
        // Let go of with no lock held.
        let _gone = { lock(&freed.0).remove(&id) };
    }
}
