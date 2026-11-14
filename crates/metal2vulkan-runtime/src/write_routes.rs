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
//! - **Of their kind alone.** An indirect command buffer holds commands of one kind, render or
//!   compute (Metal refuses a descriptor that mixes draw and dispatch command types), and a
//!   function's operations write commands of their intrinsic family's kind
//!   (`CommandWrites::kinds`: a `render_command`'s or a `compute_command`'s). So a route reaches
//!   the buffers of its kinds alone: a kernel that encodes dispatches neither marks nor adds to a
//!   buffer of draws its command buffer made resident for writing, and a buffer of no command
//!   types is reached by every route.
//! - **What they write.** What every route of the command buffer may write joins what the records
//!   of every buffer of its kinds it made resident for writing may hold (`written_commands.rs`),
//!   at the call
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

use metal2vulkan::metal::IndirectCommandType;
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

/// The command kinds of [`Routes`]' per-kind fields, by slot (`CommandWrites::*_COMMANDS`).
const KINDS: [u8; 2] = [
    CommandWrites::RENDER_COMMANDS,
    CommandWrites::COMPUTE_COMMANDS,
];

/// The slots of [`KINDS`] that `kinds` names.
fn slots(kinds: u8) -> impl Iterator<Item = usize> {
    (0..KINDS.len()).filter(move |&slot| kinds & KINDS[slot] != 0)
}

/// The command kinds `writes` may write (`CommandWrites::kinds`): both where it names none
/// though it writes something (no write the AIR walk reads names none).
fn kinds_of_writes(writes: CommandWrites) -> u8 {
    match writes.kinds {
        0 if !writes.is_empty() => CommandWrites::ALL_KINDS,
        kinds => kinds,
    }
}

/// The command kind `icb` holds: compute where its command types are dispatches, render where
/// they are draws (Metal refuses a descriptor that mixes them); both where it has none.
fn kinds_of(icb: &IndirectCommandBuffer) -> u8 {
    let types = icb.descriptor().command_types;
    let dispatches = IndirectCommandType::CONCURRENT_DISPATCH
        .union(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);
    let mut kinds = 0;
    if types.intersects(dispatches) {
        kinds |= CommandWrites::COMPUTE_COMMANDS;
    }
    if !types.difference(dispatches).is_empty() {
        kinds |= CommandWrites::RENDER_COMMANDS;
    }
    match kinds {
        0 => CommandWrites::ALL_KINDS,
        kinds => kinds,
    }
}

/// The routes an encoder, and then its command buffer, met (the module docs): what it made
/// resident for writing, and by command kind the first state set that may write records of that
/// kind and what they all may write, and what it executed under the states its records name, each
/// buffer once. A route reaches the buffers of the kinds its operations write alone
/// (`CommandWrites::kinds`): a kernel that encodes dispatches writes no draw into a buffer of
/// render commands its command buffer made resident for writing.
#[derive(Debug, Default)]
pub(crate) struct Routes {
    writable: Vec<IndirectCommandBuffer>,
    writer: [Option<GpuWriter>; 2],
    /// What the states set may write, and the states the executed buffers' records may name, as
    /// each was met, by command kind ([`KINDS`]): what a buffer made resident for writing gains
    /// (`WrittenCommands::add_writer`), its kind's.
    writes: [CommandWrites; 2],
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

    /// What the routes met may write into a buffer of `kinds`.
    fn writes_for(&self, kinds: u8) -> CommandWrites {
        slots(kinds).fold(CommandWrites::NONE, |writes, slot| {
            writes.union(self.writes[slot])
        })
    }

    /// The encoder made `icb` resident for writing: its records may hold what the routes the
    /// command buffer met may write into a buffer of its kind, and it is marked at once where one
    /// met it.
    pub(crate) fn writable(&mut self, icb: &IndirectCommandBuffer) {
        if !self.writable.contains(icb) {
            self.writable.push(icb.clone());
        }
        let kinds = kinds_of(icb);
        icb.written().add_writer(self.writes_for(kinds)).mark();
        if icb.gpu_written() {
            return;
        }
        if let Some(writer) = self.writer(kinds) {
            icb.written_by_gpu(writer);
        }
    }

    /// The encoder set a state whose functions may call `writes`' operations, of the functions
    /// `functions` names, asked for the first such state of each kind alone: where they may call
    /// one, the records of the buffers of its kinds made resident for writing may hold what they
    /// write, and those buffers are marked at once.
    pub(crate) fn state_set(
        &mut self,
        writes: CommandWrites,
        functions: impl FnOnce() -> Vec<String>,
    ) {
        if writes.is_empty() {
            return;
        }
        self.wrote(writes);
        let kinds = kinds_of_writes(writes);
        let first: Vec<usize> = slots(kinds)
            .filter(|&slot| self.writer[slot].is_none())
            .collect();
        if !first.is_empty() {
            let writer = GpuWriter::State {
                functions: functions(),
                writes,
            };
            for slot in first {
                self.writer[slot] = Some(writer.clone());
            }
        }
        for icb in &self.writable {
            if icb.gpu_written() || kinds_of(icb) & kinds == 0 {
                continue;
            }
            if let Some(writer) = self.writer(kinds_of(icb)) {
                icb.written_by_gpu(writer);
            }
        }
    }

    /// The encoder executed commands of `icb` under the states their records name: where they may
    /// name one whose functions may write records, the records of the buffers of those writes'
    /// kinds made resident for writing may hold what those write, and those buffers are marked at
    /// once.
    pub(crate) fn executed(&mut self, icb: &IndirectCommandBuffer) {
        if !self.executed.contains(icb) {
            self.executed.push(icb.clone());
        }
        let named = icb.written().named().writes;
        if !named.is_empty() {
            self.wrote(named);
            let kinds = kinds_of_writes(named);
            let reached: Vec<IndirectCommandBuffer> = (self.writable.iter())
                .filter(|writable| kinds_of(writable) & kinds != 0)
                .cloned()
                .collect();
            mark(&reached, &GpuWriter::Execution);
        }
    }

    /// A route that may write `writes`: the records of every buffer of its kinds made resident for
    /// writing may hold them, and of every one made resident later.
    fn wrote(&mut self, writes: CommandWrites) {
        let kinds = kinds_of_writes(writes);
        for slot in slots(kinds) {
            self.writes[slot] = self.writes[slot].union(writes);
        }
        for icb in &self.writable {
            if kinds_of(icb) & kinds != 0 {
                icb.written().add_writer(writes).mark();
            }
        }
    }

    /// The route that may write records of `kinds` met so far: the first state set that may
    /// write that kind, else an execution whose records may name a state that may.
    fn writer(&self, kinds: u8) -> Option<GpuWriter> {
        if let Some(writer) = self.state_writer(kinds) {
            return Some(writer);
        }
        let named = |icb: &IndirectCommandBuffer| {
            let writes = icb.written().named().writes;
            !writes.is_empty() && kinds_of_writes(writes) & kinds != 0
        };
        self.executed
            .iter()
            .any(named)
            .then_some(GpuWriter::Execution)
    }

    /// The first state set that may write records of `kinds`.
    fn state_writer(&self, kinds: u8) -> Option<GpuWriter> {
        slots(kinds).find_map(|slot| self.writer[slot].clone())
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
        for (mine, theirs) in self.writer.iter_mut().zip(writer) {
            if mine.is_none() {
                *mine = theirs;
            }
        }
        for icb in executed {
            if !self.executed.contains(&icb) {
                self.executed.push(icb);
            }
        }
        for (mine, theirs) in self.writes.iter_mut().zip(writes) {
            *mine = mine.union(theirs);
        }
    }

    /// What the commit decides (the module docs): the buffers to mark, each with what met it, and
    /// those it leaves unmarked, which an executed buffer's records may yet mark ([`Unmarked`],
    /// held by the command buffer until it completes; `None` where it executed none); and the
    /// buffers the summaries' gains set off ([`Followed`]), which the caller marks without its
    /// lock: each buffer made resident for writing takes what every route of the command buffer
    /// may write into a buffer of its kind, first. The routes hold no buffer after it: a buffer
    /// the command buffer made resident or executed lives no longer for them than for its work,
    /// which releases it as it completes.
    pub(crate) fn at_commit(&mut self) -> (AtCommit, Followed) {
        let mut followed = Followed::default();
        for icb in &self.writable {
            let writes = self.writes_for(kinds_of(icb));
            followed.0.extend(icb.written().add_writer(writes).0);
        }
        let decided = self.decide();
        self.writable.clear();
        self.writer = [None, None];
        self.writes = [CommandWrites::NONE; 2];
        self.executed.clear();
        (decided, followed)
    }

    /// [`at_commit`](Self::at_commit)'s decision: each buffer not yet marked that a state set
    /// may write, marked with the first such state of its kind; the rest marked where an executed
    /// buffer's records may name a state whose functions may encode, else left unmarked.
    fn decide(&self) -> AtCommit {
        let mut marked = Vec::new();
        let mut writable = Vec::new();
        for icb in self.writable.iter().filter(|icb| !icb.gpu_written()) {
            match self.state_writer(kinds_of(icb)) {
                Some(writer) => marked.push((icb.clone(), writer)),
                None => writable.push(icb.clone()),
            }
        }
        if writable.is_empty() {
            return AtCommit {
                marked,
                unmarked: (0, None),
            };
        }
        let count = writable.len() as u64;
        let unmarked = Unmarked::of(writable);
        // Each executed buffer's check and its hold of `unmarked` are one, under its summary's
        // lock (`WrittenCommands::follow`): a record that comes to name a state meanwhile marks
        // them either here or as it is written.
        let named = (self.executed.iter()).any(|icb| !icb.written().follow(&unmarked));
        if named {
            let execution = GpuWriter::Execution;
            marked.extend((unmarked.buffers.iter()).map(|icb| (icb.clone(), execution.clone())));
            return AtCommit {
                marked,
                unmarked: (0, None),
            };
        }
        let held = (!self.executed.is_empty()).then_some(unmarked);
        AtCommit {
            marked,
            unmarked: (count, held),
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
pub(crate) struct AtCommit {
    /// Mark these, each with what met it.
    pub(crate) marked: Vec<(IndirectCommandBuffer, GpuWriter)>,
    /// So many left unmarked, and their hold where the command buffer executed indirect commands.
    pub(crate) unmarked: (u64, Option<Arc<Unmarked>>),
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
