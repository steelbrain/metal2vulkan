//! Whether the GPU may write an indirect command buffer's records
//! (`IndirectCommandBuffer::gpu_written`): only where a route that may write them meets the
//! buffer ([`Routes`], [`GpuWriter`]).
//!
//! - **The routes.** A function reaches a buffer's records only where its command buffer made the
//!   buffer resident for writing (`useResource:usage:` or `useResources:count:usage:` with
//!   `MTLResourceUsageWrite`, in any of its compute or render encoders: residency is the command
//!   buffer's, `residency.rs`), and writes them only through an indirect-command operation, which
//!   the translator reads from the function's AIR (`indirect_command_writes`: the operations its
//!   body may call, every one where it makes a call the translator cannot see into). So the
//!   buffer is written by the GPU where its command buffer also runs a function that may call one:
//!   a state set in any of its encoders, before the residency or after it, whose functions'
//!   `indirect_command_writes` is not empty; or an execution in any of them of indirect commands
//!   whose records may name such a state (`WrittenCommands::named`). Metal runs the commands such
//!   a function writes with the buffer resident for writing in any encoder of the command buffer,
//!   one before the function's or one after it (the probe
//!   `crates/metal2vulkan-conformance/probes/icb_routed_writes.m`).
//! - **What they write.** What every route of the command buffer may write joins what the records
//!   of every buffer it made resident for writing may hold (`written_commands.rs`), at the call
//!   that brings either (for two encoders open together, a parallel render encoder's, at the
//!   commit): an execution decides on that summary (`render/executions/gpu.rs` runs draws of the
//!   one primitive type it allows on the GPU alone), so it holds what a writer in another encoder
//!   may write as it holds a writer's in the encoder that made the buffer resident.
//! - **When.** As soon as both meet, at the call that completes the pair: an encoder opens with
//!   what its command buffer's ended encoders met ([`Routes::carried`]), so a later encoder's
//!   execution of the buffer is encoded as one of records the GPU may write, a part of its own
//!   that waits for the writer (`encoder.rs`, `Segment::host`), as one after the residency in the
//!   same encoder is. The commit decides again, from what the records name then, and a buffer it
//!   left unmarked is marked later where an executed
//!   buffer's records come to name such a state ([`Unmarked`]: a CPU edit after the commit, which
//!   a host step recording the execution again reads, or a GPU writer of those records). A blit's
//!   copy into the buffer and its reset of a range in it write the records themselves, and mark
//!   it at the call (`blit_indirect.rs`).
//! - **What reaches no record.** A heap holds no indirect command buffer
//!   (`IndirectCommandBuffer::heap` is `None`), so `useHeap:` makes none resident; residency sets
//!   are refused (`RuntimeGap::ResidencySets`); a buffer made resident for reading alone, or not
//!   at all, is outside Metal's contract for a write, as before this.
//! - **The counter.** A buffer a commit leaves unmarked though its command buffer made it resident
//!   for writing is counted once per commit (`WorkCounts::indirect_write_residencies_unmarked`).

use std::fmt;
use std::sync::{Arc, Weak};

use metal2vulkan::pipeline::CommandWrites;

use super::indirect_command_buffer::IndirectCommandBuffer;

/// What met an indirect command buffer that may write its records, named in the host-replay
/// notice (`Degradation::HostReplayedIndirectCommands`): for reporting alone, nothing decides by
/// it.
#[derive(Clone, Debug)]
pub(crate) enum GpuWriter {
    /// A pipeline state set in a command buffer that made the buffer resident for writing, whose
    /// functions (`functions`, named as the program made them) may call `writes`' operations.
    State {
        functions: Vec<String>,
        writes: CommandWrites,
    },
    /// An execution, in a command buffer that made the buffer resident for writing, of indirect
    /// commands whose records may name such a state.
    Execution,
    /// A blit's copy of indirect commands into the buffer.
    Copy,
    /// A blit's reset of commands in the buffer.
    Reset,
}

impl fmt::Display for GpuWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuWriter::State { functions, writes } => {
                let operations = operation_names(*writes);
                write!(
                    f,
                    "a pipeline state of {} set in a command buffer that made it resident for \
                     writing, whose functions may call {} kinds of indirect-command operation \
                     ({})",
                    functions.join(", "),
                    operations.len(),
                    operations.join(", ")
                )
            }
            GpuWriter::Execution => f.write_str(
                "an execution, in a command buffer that made it resident for writing, of \
                 indirect commands whose records may name a state whose functions may encode",
            ),
            GpuWriter::Copy => f.write_str("a blit's copy of indirect commands into it"),
            GpuWriter::Reset => f.write_str("a blit's reset of commands in it"),
        }
    }
}

/// The host-replay notice's words for `icb`, which the GPU may write: `what` its execution runs,
/// and the route that met it first.
pub(crate) fn replay_why(icb: &IndirectCommandBuffer, what: &str) -> String {
    match icb.gpu_writer() {
        Some(writer) => format!("the GPU may write {what} ({writer})"),
        None => format!("the GPU may write {what}"),
    }
}

/// The names of the operations `writes` may call, in its bits' order.
fn operation_names(writes: CommandWrites) -> Vec<&'static str> {
    const NAMES: [(u32, &str); 15] = [
        (CommandWrites::DRAW, "draw_primitives"),
        (CommandWrites::DRAW_INDEXED, "draw_indexed_primitives"),
        (CommandWrites::DRAW_PATCHES, "draw_patches"),
        (CommandWrites::DRAW_MESH, "draw_mesh"),
        (CommandWrites::DISPATCH, "concurrent_dispatch"),
        (CommandWrites::PIPELINE, "set_pipeline_state"),
        (CommandWrites::BUFFER, "a buffer setter"),
        (CommandWrites::BARRIER, "set_barrier"),
        (CommandWrites::CLEAR_BARRIER, "clear_barrier"),
        (CommandWrites::RENDER_STATE, "a render state setter"),
        (
            CommandWrites::THREADGROUP_MEMORY,
            "a threadgroup memory length setter",
        ),
        (CommandWrites::STAGE_IN_REGION, "set_stage_in_region"),
        (CommandWrites::IMAGEBLOCK_SIZE, "set_imageblock_size"),
        (CommandWrites::RESET, "reset"),
        (CommandWrites::COPY, "copy_command"),
    ];
    NAMES
        .iter()
        .filter(|(bit, _)| writes.may(*bit))
        .map(|(_, name)| *name)
        .collect()
}

/// The routes an encoder, and then its command buffer, met (the module docs): what it made
/// resident for writing, the first state set that may write records, what they all may write,
/// and what it executed under the states its records name, each buffer once.
#[derive(Debug, Default)]
pub(crate) struct Routes {
    writable: Vec<IndirectCommandBuffer>,
    writer: Option<GpuWriter>,
    /// What the states set may write, and the states the executed buffers' records may name, as
    /// each was met: what a buffer made resident for writing gains (`WrittenCommands::add_writer`).
    writes: CommandWrites,
    executed: Vec<IndirectCommandBuffer>,
}

impl Routes {
    /// What an encoder opens with: what its command buffer's ended encoders met (the module
    /// docs), which its end joins again.
    pub(crate) fn carried(&self) -> Routes {
        Routes {
            writable: self.writable.clone(),
            writer: self.writer.clone(),
            writes: self.writes,
            executed: self.executed.clone(),
        }
    }

    /// The encoder made `icb` resident for writing: its records may hold what the routes the
    /// command buffer met may write, and it is marked at once where one met it.
    pub(crate) fn writable(&mut self, icb: &IndirectCommandBuffer) {
        if !self.writable.contains(icb) {
            self.writable.push(icb.clone());
        }
        icb.written().add_writer(self.writes).mark();
        if icb.gpu_written() {
            return;
        }
        if let Some(writer) = self.writer() {
            icb.written_by_gpu(writer);
        }
    }

    /// The encoder set a state whose functions may call `writes`' operations, of the functions
    /// `functions` names, asked for the first such state alone: where they may call one, the
    /// records of the buffers made resident for writing may hold what they write, and the buffers
    /// are marked at once.
    pub(crate) fn state_set(
        &mut self,
        writes: CommandWrites,
        functions: impl FnOnce() -> Vec<String>,
    ) {
        if writes.is_empty() {
            return;
        }
        self.wrote(writes);
        let writer = self.writer.get_or_insert_with(|| GpuWriter::State {
            functions: functions(),
            writes,
        });
        mark(&self.writable, writer);
    }

    /// The encoder executed commands of `icb` under the states their records name: where they may
    /// name one whose functions may write records, the records of the buffers made resident for
    /// writing may hold what those write, and the buffers are marked at once.
    pub(crate) fn executed(&mut self, icb: &IndirectCommandBuffer) {
        if !self.executed.contains(icb) {
            self.executed.push(icb.clone());
        }
        let named = icb.written().named().writes;
        if !named.is_empty() {
            self.wrote(named);
            mark(&self.writable, &GpuWriter::Execution);
        }
    }

    /// A route that may write `writes`: the records of every buffer made resident for writing may
    /// hold them, and of every one made resident later.
    fn wrote(&mut self, writes: CommandWrites) {
        self.writes = self.writes.union(writes);
        for icb in &self.writable {
            icb.written().add_writer(writes).mark();
        }
    }

    /// The route that may write records met so far: the first state set, else an execution whose
    /// records may name one.
    fn writer(&self) -> Option<GpuWriter> {
        let named = |icb: &IndirectCommandBuffer| !icb.written().named().writes.is_empty();
        match &self.writer {
            Some(writer) => Some(writer.clone()),
            None => self
                .executed
                .iter()
                .any(named)
                .then_some(GpuWriter::Execution),
        }
    }

    /// An ended encoder's routes join its command buffer's. An encoder opened with what those
    /// ended before it met, so its buffers took their writes as it ran; what two encoders open
    /// together (a parallel render encoder's) did not see of each other's, the commit adds.
    pub(crate) fn join(&mut self, ended: Routes) {
        let Routes {
            writable,
            writer,
            writes,
            executed,
        } = ended;
        for icb in writable {
            if !self.writable.contains(&icb) {
                self.writable.push(icb);
            }
        }
        if self.writer.is_none() {
            self.writer = writer;
        }
        for icb in executed {
            if !self.executed.contains(&icb) {
                self.executed.push(icb);
            }
        }
        self.writes = self.writes.union(writes);
    }

    /// What the commit decides (the module docs): the buffers to mark, each with what met it, or
    /// those it leaves unmarked, which an executed buffer's records may yet mark ([`Unmarked`],
    /// held by the command buffer until it completes; `None` where it executed none); and the
    /// buffers the summaries' gains set off ([`Followed`]), which the caller marks without its
    /// lock: each buffer made resident for writing takes what every route of the command buffer
    /// may write, first. The routes hold no buffer after it: a buffer the command buffer made
    /// resident or executed lives no longer for them than for its work, which releases it as it
    /// completes.
    pub(crate) fn at_commit(&mut self) -> (AtCommit, Followed) {
        let mut followed = Followed::default();
        for icb in &self.writable {
            followed.0.extend(icb.written().add_writer(self.writes).0);
        }
        let decided = self.decide();
        self.writable.clear();
        self.writer = None;
        self.writes = CommandWrites::NONE;
        self.executed.clear();
        (decided, followed)
    }

    /// [`at_commit`](Self::at_commit)'s decision.
    fn decide(&self) -> AtCommit {
        let writable: Vec<IndirectCommandBuffer> = (self.writable.iter())
            .filter(|icb| !icb.gpu_written())
            .cloned()
            .collect();
        if writable.is_empty() {
            return AtCommit::Unmarked(0, None);
        }
        if let Some(writer) = &self.writer {
            return AtCommit::Mark(writable, writer.clone());
        }
        let count = writable.len() as u64;
        let unmarked = Unmarked::of(writable);
        // Each executed buffer's check and its hold of `unmarked` are one, under its summary's
        // lock (`WrittenCommands::follow`): a record that comes to name a state meanwhile marks
        // them either here or as it is written.
        let named = (self.executed.iter()).any(|icb| !icb.written().follow(&unmarked));
        match named {
            true => AtCommit::Mark(unmarked.buffers.clone(), GpuWriter::Execution),
            false => AtCommit::Unmarked(count, (!self.executed.is_empty()).then_some(unmarked)),
        }
    }
}

/// Mark each of `buffers` not yet marked, with `writer`.
fn mark(buffers: &[IndirectCommandBuffer], writer: &GpuWriter) {
    for icb in buffers.iter().filter(|icb| !icb.gpu_written()) {
        icb.written_by_gpu(writer.clone());
    }
}

/// What a commit decides ([`Routes::at_commit`]).
pub(crate) enum AtCommit {
    /// Mark these, with what met them.
    Mark(Vec<IndirectCommandBuffer>, GpuWriter),
    /// So many left unmarked, and their hold where the command buffer executed indirect commands.
    Unmarked(u64, Option<Arc<Unmarked>>),
}

/// The buffers a commit left unmarked though its command buffer made them resident for writing
/// and executed indirect commands (the module docs), which those commands' buffers hold weakly
/// (`WrittenCommands::follow`): where their records come to name a state whose functions may
/// encode, they mark these ([`Followed`]).
#[derive(Debug)]
pub(crate) struct Unmarked {
    buffers: Vec<IndirectCommandBuffer>,
}

impl Unmarked {
    /// A hold of `buffers`.
    pub(crate) fn of(buffers: Vec<IndirectCommandBuffer>) -> Arc<Unmarked> {
        Arc::new(Unmarked { buffers })
    }
}

/// The holds of [`Unmarked`] buffers an add to a summary set off (`WrittenCommands::add`), which
/// the caller marks once it holds no lock of the buffers' ([`mark`](Self::mark)).
#[must_use]
#[derive(Debug, Default)]
pub(crate) struct Followed(pub(crate) Vec<Arc<Unmarked>>);

impl Followed {
    /// Mark the buffers.
    pub(crate) fn mark(self) {
        for unmarked in self.0 {
            for icb in &unmarked.buffers {
                icb.written_by_gpu(GpuWriter::Execution);
            }
        }
    }
}

/// The holds `followers` keeps, those that live, taken: the records came to name a state whose
/// functions may encode.
pub(crate) fn followed(followers: &mut Vec<Weak<Unmarked>>) -> Followed {
    Followed(
        followers
            .drain(..)
            .filter_map(|held| held.upgrade())
            .collect(),
    )
}
