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
mod error;
mod event;
mod extension;
mod fill;
pub mod floor;
mod instance;
mod memory;
mod queue;
mod record;
mod submit;
#[cfg(test)]
mod tests;

pub use buffer::Buffer;
pub use command::{CommandBuffer, CommandBufferError, CommandBufferStatus};
pub use counters::{Counters, Gauges, Metrics};
pub use device::{Device, DeviceOptions, RawDevice};
pub use error::{Error, Limit, Refusal};
pub use event::{Event, SharedEvent};
pub use extension::DeviceExtension;
pub use instance::{Instance, PhysicalDevice};
pub use memory::{BlockId, BlockInfo, BlockMappings, MemoryClass, BLOCK_LEN};
pub use queue::CommandQueue;
