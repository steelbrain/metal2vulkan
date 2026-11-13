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
//! - **The runtime.** A record holds the ID alone, and names the state that held the ID as the
//!   GPU last could write the buffer's records (`IndirectCommandBuffer::window_holder`, which the
//!   getter answers too). A buffer the GPU may write keeps, from its first such write
//!   (`IndirectCommandBuffer::gpu_write`), the program of each state made with
//!   `supportIndirectCommandBuffers` the program releases since ([`Keepers::released`]), and a
//!   blit's copy of commands keeps in its destination the source's kept programs and those of
//!   the released states the CPU set on the commands it copies: a record naming the ID runs that
//!   program where the state held the ID then. Nothing is kept for a buffer the GPU never writes,
//!   so a program that writes no records on the GPU keeps nothing.
//! - **Bounded.** A buffer keeps an ID's last [`PIPELINE_HOLDERS`] released programs, as the
//!   device keeps its last holders: a fifth drops the oldest, whose records then run nothing,
//!   announced (below). So what the buffers keep is at most four programs per pipeline state ID
//!   the program used, each one `Arc` the buffers share: with plain allocation, which hands out
//!   the lowest free ID as Metal does, four times the peak count of live states made with
//!   `supportIndirectCommandBuffers`, however long the program makes and releases them; each ID a
//!   placement (`PipelineOptions::placement`) names besides counts four more.
//! - **Not kept** (`Degradation::ReleasedCommandState`, the command running nothing): the program
//!   of a state released before any buffer whose records the GPU writes naming its ID could be
//!   written, which Metal runs where no state took the ID since; and one past an ID's last
//!   [`PIPELINE_HOLDERS`] holders.

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

/// The buffers of a device whose records the GPU may write, each by what it keeps, weakly: the
/// buffer holds its programs, so the device holds no program (a program holds its device).
#[derive(Default)]
pub(super) struct Keepers(Mutex<Vec<Weak<Mutex<KeptPrograms>>>>);

impl Keepers {
    /// From now on keep, in `kept`, the programs of the states the program releases.
    pub(super) fn add(&self, kept: &Arc<Mutex<KeptPrograms>>) {
        let mut keepers = lock(&self.0);
        keepers.retain(|keeper| keeper.strong_count() > 0);
        keepers.push(Arc::downgrade(kept));
    }

    /// The program released a state made with `supportIndirectCommandBuffers` whose hold of ID
    /// `id` had `serial`, and which runs `program`: each buffer whose records the GPU may write
    /// keeps it. Nothing, and `program` is not called, where there is none.
    pub(super) fn released(&self, id: u64, serial: u64, program: impl FnOnce() -> KeptProgram) {
        let keepers: Vec<Arc<Mutex<KeptPrograms>>> = {
            let mut keepers = lock(&self.0);
            keepers.retain(|keeper| keeper.strong_count() > 0);
            keepers.iter().filter_map(Weak::upgrade).collect()
        };
        if keepers.is_empty() {
            return;
        }
        let program = program();
        for keeper in keepers {
            lock(&keeper).keep(id, serial, program.clone());
        }
    }
}
