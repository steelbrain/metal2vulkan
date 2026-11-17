//! The Metal API on Vulkan, with one path for every device.
//!
//! Every difference between devices comes from a Vulkan query: a feature, a property, a format, a
//! limit or a memory type. The crate holds no vendor, device or driver identity, no environment
//! gate and no runtime policy switch; `tests::one_path` enforces this over its sources.
//!
//! It builds only on `metal2vulkan`'s public API, as a separate crate must.

#![warn(missing_docs)]

mod counters;
mod device;
mod error;
pub mod floor;
mod instance;
#[cfg(test)]
mod tests;

pub use counters::{Counters, Gauges, Metrics};
pub use device::{Device, RawDevice};
pub use error::Error;
pub use instance::{Instance, PhysicalDevice};
