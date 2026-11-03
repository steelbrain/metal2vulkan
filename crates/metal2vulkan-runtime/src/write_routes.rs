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
//!   whose records may name such a state (`WrittenCommands::named`). The commit decides from what
//!   the records name then, and a buffer it left unmarked is marked later where an executed
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
/// resident for writing, the first state set that may write records, and what it executed under
/// the states its records name, each buffer once.
#[derive(Debug, Default)]
pub(crate) struct Routes {
    writable: Vec<IndirectCommandBuffer>,
    writer: Option<GpuWriter>,
    executed: Vec<IndirectCommandBuffer>,
}

impl Routes {
    /// The encoder made `icb` resident for writing.
    pub(crate) fn writable(&mut self, icb: &IndirectCommandBuffer) {
        if !icb.gpu_written() && !self.writable.contains(icb) {
            self.writable.push(icb.clone());
        }
    }

    /// The encoder set a state whose functions may call `writes`' operations, of the functions
    /// `functions` names, asked for the first such state alone.
    pub(crate) fn state_set(
        &mut self,
        writes: CommandWrites,
        functions: impl FnOnce() -> Vec<String>,
    ) {
        if self.writer.is_none() && !writes.is_empty() {
            self.writer = Some(GpuWriter::State {
                functions: functions(),
                writes,
            });
        }
    }

    /// The encoder executed commands of `icb` under the states their records name.
    pub(crate) fn executed(&mut self, icb: &IndirectCommandBuffer) {
        if !self.executed.contains(icb) {
            self.executed.push(icb.clone());
        }
    }

    /// An ended encoder's routes join its command buffer's.
    pub(crate) fn join(&mut self, ended: Routes) {
        let Routes {
            writable,
            writer,
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
    }

    /// What the commit decides (the module docs): the buffers to mark, each with what met it, or
    /// those it leaves unmarked, which an executed buffer's records may yet mark ([`Unmarked`],
    /// held by the command buffer until it completes; `None` where it executed none).
    pub(crate) fn at_commit(&self) -> AtCommit {
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
