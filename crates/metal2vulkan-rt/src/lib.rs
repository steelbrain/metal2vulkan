//! The Metal API on Vulkan, with one path for every device.
//!
//! Every difference between devices comes from a Vulkan query: a feature, a property, a format, a
//! limit or a memory type. The crate holds no vendor, device or driver identity, no environment
//! gate and no runtime policy switch; `tests::one_path` enforces this over its sources.
//!
//! It builds only on `metal2vulkan`'s public API, as a separate crate must.

#![warn(missing_docs)]

mod buffer;
mod command;
mod complete;
mod counters;
mod device;
mod encoder;
mod error;
mod event;
mod extension;
mod fill;
pub mod floor;
mod hazard;
mod instance;
mod memory;
mod pipeline;
mod queue;
mod readback;
mod reaper;
mod record;
mod ring;
mod submit;
#[cfg(test)]
mod tests;
mod times;

pub use buffer::Buffer;
pub use command::{CommandBuffer, CommandBufferError, CommandBufferStatus};
pub use counters::{Counters, Gauges, Metrics};
pub use device::{Device, DeviceOptions, RawDevice};
pub use encoder::{ComputeCommandEncoder, DispatchType, Resource, ResourceUsage};
pub use error::{Error, Gap, Limit, Refusal};
pub use event::{Event, SharedEvent};
pub use extension::{ConsumerFeature, DeviceExtension};
pub use instance::{Instance, PhysicalDevice};
pub use memory::{BlockId, BlockInfo, BlockMappings, MemoryClass, BLOCK_LEN};
/// `MTLLibrary`: the translator's, read from AIR or a metallib.
pub use metal2vulkan::library::Library;
/// `MTLComputePipelineDescriptor`, its kernel, and a pipeline's precision notes: the
/// translator's.
pub use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, FunctionDescriptor, PrecisionNote, StageSource,
};
pub use pipeline::ComputePipelineState;
pub use queue::CommandQueue;
pub use times::host_time;
