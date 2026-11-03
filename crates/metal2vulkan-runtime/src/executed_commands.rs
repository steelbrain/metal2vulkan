//! What an execution of an indirect command buffer runs, as the runtime reads its records: a
//! runtime extension, with no Metal counterpart.
//!
//! - **Why.** Which pipeline state an indirect command runs is decided by its record, which the
//!   CPU or the GPU may write until the runtime reads it: in `commit` for the CPU's records, in
//!   the host step for records the GPU may write (`IndirectCommandBuffer::gpu_written`), or on
//!   the GPU alone (`gpu_execute.rs`). A program that decides something of its own by the
//!   states the commands run (whether one can encode indirect commands, and which indirect
//!   command buffers the command buffer made resident for writing) learns them there, from the
//!   runtime, without reading the records again.
//! - **The hook.** [`CommandBuffer::add_executed_commands_handler`] adds, before `commit`, a
//!   handler the runtime calls with an [`ExecutedCommands`] each time it reads an execution's
//!   records, on the thread that reads them ([`RecordsRead`]), with no lock of the command buffer
//!   held. An execution `commit` recorded from the CPU's records whose buffer an encoder makes
//!   resident for writing before its batch is submitted is read again in its host step, and the
//!   handler sees it again. An execution in an encoder the command buffer's failure runs nothing
//!   of (`commit_failed`) is never read.
//! - **Its cost.** None without a handler but one lock of the command buffer's state per
//!   execution read; with one, the facts are gathered from what the reading resolved already.

use super::command_buffer::CommandBuffer;
use super::error::{invalid, Error, Problem};
use super::indirect_command_buffer::IndirectCommandBuffer;
use super::residency::Residency;
use std::sync::Arc;

/// A handler of a command buffer's executions ([`CommandBuffer::add_executed_commands_handler`]).
pub type ExecutedCommandsHandler = Arc<dyn Fn(&ExecutedCommands) + Send + Sync + 'static>;

/// Where the runtime read an execution's records ([`ExecutedCommands::read`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordsRead {
    /// In `commit`, on the committing thread: the CPU's records as they stood then.
    Commit,
    /// In the host step, on the scheduler's thread: the records as the GPU and the CPU left
    /// them, once the work before the execution completed.
    HostStep,
    /// Not on the host: the GPU reads them as it runs (`gpu_execute.rs`), so
    /// [`ExecutedCommands::commands`] is empty and
    /// [`ExecutedCommands::unread_states_can_encode`] tells of every state the records may name.
    /// Decided in `commit`, on the committing thread, and again on the thread of each rebind of
    /// a buffer over host pages before the command buffer's first submission, which records the
    /// execution again.
    Gpu,
}

/// A command of an execution that runs, as the runtime read it: one that dispatches or draws
/// with a pipeline state, its own or the encoder's that the buffer inherits.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutedCommand {
    /// Its index in the buffer.
    pub index: u64,
    /// Whether the state it runs may encode, reset or copy indirect commands
    /// (`ComputePipelineState::can_encode_indirect_commands`,
    /// `RenderPipelineState::can_encode_indirect_commands`).
    pub can_encode_indirect_commands: bool,
}

/// An execution of an indirect command buffer's commands as the runtime read its records
/// (the module docs).
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ExecutedCommands {
    /// The buffer executed.
    pub buffer: IndirectCommandBuffer,
    /// Where its records were read.
    pub read: RecordsRead,
    /// The commands of its range that run, in order. Empty where [`RecordsRead::Gpu`].
    pub commands: Vec<ExecutedCommand>,
    /// Where [`RecordsRead::Gpu`]: whether a state its records may name (the encoder's where the
    /// buffer inherits it, else every state made with `supportIndirectCommandBuffers` alive at
    /// the commit) may encode, reset or copy indirect commands. `false` otherwise.
    pub unread_states_can_encode: bool,
    /// The indirect command buffers the command buffer's encoders made resident for writing
    /// (`use_resource` or `use_resources` with `ResourceUsage::WRITE`), as its residency stood
    /// where the records were read: Metal's residency is the command buffer's, every encoder's.
    pub writable: Vec<IndirectCommandBuffer>,
}

impl CommandBuffer {
    /// Call `handler` each time the runtime reads the records of an execution of an indirect
    /// command buffer of this command buffer, on the thread that reads them, with no lock of the
    /// command buffer held (`executed_commands.rs`). A runtime extension, with no Metal
    /// counterpart. Refused once the command buffer is committed (`Invalid`,
    /// `Problem::HandlerAfterCommit`).
    pub fn add_executed_commands_handler(
        &self,
        handler: ExecutedCommandsHandler,
    ) -> Result<(), Error> {
        let mut state = self.state();
        if state.enqueued() {
            return Err(invalid(
                "add_executed_commands_handler",
                Problem::HandlerAfterCommit,
            ));
        }
        state.executed_handlers.push(handler);
        Ok(())
    }

    /// Hand the handlers the facts `facts` gathers of an execution of `buffer` read at `read`,
    /// under `residency`, the command buffer's, where it has handlers; with no lock held.
    /// Whether the command buffer has an executed-commands handler, which an execution reports
    /// the commands it ran to ([`report_executed`](Self::report_executed)).
    pub(crate) fn reports_executed(&self) -> bool {
        !self.state().executed_handlers.is_empty()
    }

    pub(crate) fn report_executed(
        &self,
        buffer: &IndirectCommandBuffer,
        read: RecordsRead,
        residency: &Residency,
        facts: impl FnOnce() -> (Vec<ExecutedCommand>, bool),
    ) {
        let handlers = self.state().executed_handlers.clone();
        if handlers.is_empty() {
            return;
        }
        let (commands, unread_states_can_encode) = facts();
        let executed = ExecutedCommands {
            buffer: buffer.clone(),
            read,
            commands,
            unread_states_can_encode,
            writable: residency.writable_indirect_command_buffers(),
        };
        for handler in &handlers {
            handler(&executed);
        }
    }
}
