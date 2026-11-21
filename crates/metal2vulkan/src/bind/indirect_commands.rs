//! The indirect-command record format: what a CPU encoder writes into an indirect command
//! buffer's storage, a header and one fixed-size record per command. No translated function
//! writes it: a function's command encoders (`compute_command`, `render_command`) are refused,
//! `Unsupported{GpuEncodedIndirectCommands, NotImplemented}`.
//!
//! A consumer that runs its own runtime on the compile layer needs it to execute and read back
//! indirect command buffers: [`CommandBuffer::initialize`] writes the header a
//! [`CommandDescriptor`] and an [`Inheritance`] fix and resets every record,
//! [`CommandRecord::encode_into`] writes one command as the CPU's encoder sets it, and
//! [`CommandBuffer::decode`] checks the storage (its sticky encoding fault, the header,
//! every record against the descriptor) before [`CommandBuffer::records`] reads a range back as
//! [`CommandRecord`]s. [`CommandRecord::resolve_bindings`] picks each command's pipeline and
//! buffers from itself or from the executing encoder, and [`BufferBinding::resolve_buffer`] finds
//! the [`RegisteredBuffer`] whose interval holds a buffer's address. The device replay (`replay_preprocess`)
//! reads the same records on the GPU. Storage may hold a written flag per command past the
//! records ([`written_flag_offset`]), which the consumer's writes of a record set, so a reset need
//! touch only the flagged records.

pub use crate::indirect_commands::records_address;
pub use crate::indirect_commands::written_flag_offset;
pub use crate::indirect_commands::{
    Action, AddressError, BindingError, Bindings, BufferBinding, BufferStage, CommandBuffer,
    CommandBufferError, CommandDescriptor, CommandRecord, ComputeDispatch, Inheritance,
    RecordError, RecordField, RecordView, RegisteredBuffer, RenderDraws, ResolvedBindings,
    ResolvedBuffer, StateSource, BINDING_COUNT, COMMAND_BUFFER_HEADER_BYTES, COMMAND_BYTES,
    RECORDS_ADDRESS_BASE, RECORDS_ID_LIMIT, THREADGROUP_BUFFER_ARGUMENT_COUNT, WRITTEN_FLAG_BYTES,
};
