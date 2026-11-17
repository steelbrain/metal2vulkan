//! The error every fallible call returns: one typed variant per case, never text to parse.

use crate::floor::BelowFloor;
use crate::memory::BlockId;
use ash::vk;
use std::{fmt, io};

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
    /// A Vulkan call failed: `call` names the Vulkan command and `result` is its error (out of
    /// host or device memory, a lost device, an extension the loader does not offer, ...).
    Vulkan {
        /// The Vulkan command, such as `vkCreateDevice`.
        call: &'static str,
        /// What it returned.
        result: vk::Result,
    },
    /// A call Metal refuses: its validation stops it, or it answers nil.
    Refused(Refusal),
    /// One of the device's or the runtime's fixed limits.
    Limit(Limit),
    /// A block the consumer does not hold: never taken, or already released. Metal has no
    /// counterpart: blocks are the consumer's memory, not Metal objects.
    NoSuchBlock(BlockId),
    /// The runtime could not start one of its threads. Metal has no counterpart.
    Thread(io::Error),
}

/// A call Metal refuses, with what it was asked.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// [`Device::new_buffer`](crate::Device::new_buffer) with a length of 0: Metal answers nil
    /// for a buffer of no bytes.
    EmptyBuffer,
    /// [`Device::new_buffer`](crate::Device::new_buffer) with a range that leaves its block:
    /// Metal answers nil for a buffer placed past the end of its heap.
    BufferPastBlock {
        /// The buffer's offset in the block.
        offset: u64,
        /// The buffer's length.
        length: u64,
        /// The block's length.
        block_len: u64,
    },
}

/// A limit a call ran into, with what it was asked.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limit {
    /// [`Device::take_block`](crate::Device::take_block) for more bytes than one allocation and
    /// one buffer can hold on the device (`maxMemoryAllocationSize` and `maxBufferSize`). Metal's
    /// counterpart is `maxBufferLength`, past which it answers nil.
    BlockLen {
        /// The length asked for, rounded up to the allocation granule.
        len: u64,
        /// The most the device holds in one block.
        max: u64,
    },
    /// [`Device::take_block`](crate::Device::take_block) after every block number has been used:
    /// numbers are never reused during a device's life. Metal has no counterpart.
    BlockIds,
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
            Error::Vulkan { call, result } => write!(f, "{call} failed: {result}"),
            Error::Refused(refusal) => write!(f, "{refusal}"),
            Error::Limit(limit) => write!(f, "{limit}"),
            Error::NoSuchBlock(block) => {
                write!(f, "block {} is not one the consumer holds", block.get())
            }
            Error::Thread(error) => write!(f, "a runtime thread did not start: {error}"),
        }
    }
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Refusal::EmptyBuffer => f.write_str("Metal makes no buffer of length 0"),
            Refusal::BufferPastBlock {
                offset,
                length,
                block_len,
            } => write!(
                f,
                "a buffer of {length} bytes at offset {offset} leaves its block of {block_len} bytes"
            ),
        }
    }
}

impl fmt::Display for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Limit::BlockLen { len, max } => write!(
                f,
                "a block of {len} bytes is past the device's largest, {max} bytes"
            ),
            Limit::BlockIds => f.write_str("every block number has been used"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Loader(error) => Some(error),
            Error::BelowFloor(below) => Some(below),
            Error::Vulkan { result, .. } => Some(result),
            Error::Thread(error) => Some(error),
            Error::Refused(_) | Error::Limit(_) | Error::NoSuchBlock(_) => None,
        }
    }
}
