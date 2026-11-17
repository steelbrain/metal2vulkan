//! The error every fallible call returns: one typed variant per case, never text to parse.

use crate::floor::BelowFloor;
use ash::vk;
use std::fmt;

/// What a failing call returns.
///
/// Metal's counterpart is the `NSError` a creation call fills in, or the nil it returns. Here each
/// case is a variant of its own, so a caller decides by matching, never by reading a message.
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// [`Instance::new`](crate::Instance::new) found no Vulkan loader to load. Metal has no
    /// counterpart: Metal is part of the system.
    Loader(ash::LoadingError),
    /// [`Device::new`](crate::Device::new): the device is below the runtime's floor, with every
    /// item it lacks. Metal has no counterpart: every Metal device runs every program its family
    /// supports.
    BelowFloor(BelowFloor),
    /// [`Device::new`](crate::Device::new): the device has no queue family that does both graphics
    /// and compute, which every command queue submits to. Metal has no counterpart.
    NoGraphicsComputeQueue,
    /// A Vulkan call failed: `call` names the Vulkan command and `result` is its error (out of
    /// host or device memory, a lost device, an extension the loader does not offer, ...).
    Vulkan {
        /// The Vulkan command, such as `vkCreateDevice`.
        call: &'static str,
        /// What it returned.
        result: vk::Result,
    },
}

impl Error {
    /// The error of a failed Vulkan `call`, for `map_err`.
    pub(crate) fn vulkan(call: &'static str) -> impl FnOnce(vk::Result) -> Error {
        move |result| Error::Vulkan { call, result }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Loader(error) => write!(f, "no Vulkan loader: {error}"),
            Error::BelowFloor(below) => write!(f, "{below}"),
            Error::NoGraphicsComputeQueue => {
                f.write_str("the device has no queue family that does graphics and compute")
            }
            Error::Vulkan { call, result } => write!(f, "{call} failed: {result}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Loader(error) => Some(error),
            Error::BelowFloor(below) => Some(below),
            Error::NoGraphicsComputeQueue => None,
            Error::Vulkan { result, .. } => Some(result),
        }
    }
}
