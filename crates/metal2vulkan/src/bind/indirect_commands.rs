//! The indirect-command record format: what a CPU encoder writes into an indirect command
//! buffer's storage, a header and one fixed-size record per command. No translated function
//! writes it: a function's command encoders (`compute_command`, `render_command`) are refused,
//! `Unsupported{GpuEncodedIndirectCommands, NotImplemented}`.
//!
//! A consumer that writes an indirect command buffer's records on the CPU and reads them back
//! uses it (validation's case model and Metal executor do): [`CommandBuffer::initialize`] writes
//! the header a [`CommandDescriptor`] and an [`Inheritance`] fix and resets every record,
//! [`CommandRecord::encode_into`] writes one command as the CPU's encoder sets it, and
//! [`CommandBuffer::decode`] checks the storage (its sticky encoding fault, the header,
//! every record against the descriptor) before [`CommandBuffer::records`] reads a range back as
//! [`CommandRecord`]s.

pub use crate::indirect_commands::{
    Action, Bindings, BufferBinding, BufferStage, CommandBuffer, CommandBufferError,
    CommandDescriptor, CommandRecord, ComputeDispatch, Inheritance, RecordError, RecordView,
    RenderDraws, StateSource, BINDING_COUNT, COMMAND_BUFFER_HEADER_BYTES, COMMAND_BYTES,
    THREADGROUP_BUFFER_ARGUMENT_COUNT,
};
