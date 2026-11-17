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
    /// A call that needs one of the runtime's threads, after
    /// [`Device::quiesce`](crate::Device::quiesce) stopped them. Metal has no counterpart.
    Quiesced,
}

/// A call Metal refuses, with what it was asked.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// A buffer of no bytes ([`Device::new_buffer`](crate::Device::new_buffer) with a length of
    /// 0). Metal answers nil.
    ///
    /// Basis: metal-measured (an Apple M3 Max: newBufferWithLength:0 answers nil).
    EmptyBuffer,
    /// A buffer whose range leaves its block ([`Device::new_buffer`](crate::Device::new_buffer)).
    /// Metal answers nil for a buffer placed past the end of its heap.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: a placement heap of 1 MiB answers nil
    /// for a 4096-byte buffer at offset 1,047,552).
    BufferPastBlock {
        /// The buffer's offset in the block.
        offset: u64,
        /// The buffer's length.
        length: u64,
        /// The block's length.
        block_len: u64,
    },
    /// A call on a command buffer already committed: encoding, adding a handler, enqueueing or
    /// committing again.
    ///
    /// Basis: metal-documented (Metal's validation asserts "commit an already committed command
    /// buffer", "Completed handler provided after commit call" and "Scheduled handler provided
    /// after commit call").
    Committed,
    /// An event, a buffer or a pipeline state of another device in an encode call
    /// ([`CommandBuffer::encode_wait_for_event`](crate::CommandBuffer::encode_wait_for_event),
    /// [`CommandBuffer::encode_signal_event`](crate::CommandBuffer::encode_signal_event), or a
    /// [`ComputeCommandEncoder`](crate::ComputeCommandEncoder) call).
    ///
    /// Basis: ours (an event's semaphore, a buffer's block and a pipeline belong to the
    /// `VkDevice` that made them).
    OtherDevice,
    /// A queue that may hold no command buffer
    /// ([`Device::new_command_queue`](crate::Device::new_command_queue) with a count of 0).
    ///
    /// Basis: ours (its first `commandBuffer` call would wait forever).
    NoCommandBuffers,
    /// A second encoder, an encode call, or a commit or enqueue, while an encoder of the command
    /// buffer is open.
    ///
    /// Basis: metal-documented (a command buffer has one active encoder at a time, ended by
    /// `endEncoding` before another is made or the command buffer is committed).
    EncoderOpen,
    /// A dispatch with no pipeline state set.
    ///
    /// Basis: metal-documented (a compute dispatch needs a compute pipeline state set first).
    NoPipeline,
    /// A buffer argument index past Metal's 31 entries (`setBuffer`, `setBytes`,
    /// `setBufferOffset`, `setThreadgroupMemoryLength`).
    ///
    /// Basis: metal-documented (the buffer argument table has 31 entries per stage).
    BufferIndex {
        /// The index asked for.
        index: u32,
    },
    /// `setBufferOffset` at an index where no buffer is bound.
    ///
    /// Basis: ours (there is no buffer whose offset to change).
    NoBufferAtIndex {
        /// The index asked for.
        index: u32,
    },
    /// A threadgroup of no threads, or of more than the pipeline state's
    /// `maxTotalThreadsPerThreadgroup`.
    ///
    /// Basis: metal-documented (a threadgroup's threads total at most the state's
    /// `maxTotalThreadsPerThreadgroup`).
    ThreadgroupSize {
        /// The threadgroup size asked for.
        size: [u32; 3],
        /// The state's `maxTotalThreadsPerThreadgroup`.
        max: u32,
    },
    /// An indirect dispatch's argument offset that is not a multiple of 4, or that leaves fewer
    /// than 12 bytes (three 32-bit counts) in the buffer.
    ///
    /// Basis: ours (`vkCmdDispatchIndirect` reads three 32-bit counts at a 4-byte aligned offset
    /// inside the buffer).
    IndirectOffset {
        /// The offset asked for.
        offset: u64,
        /// The buffer's length.
        length: u64,
    },
}

/// A limit a call ran into, with what it was asked.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Limit {
    /// A block longer than one allocation and one buffer hold on the device
    /// ([`Device::take_block`](crate::Device::take_block) past `maxMemoryAllocationSize` or
    /// `maxBufferSize`). Metal's counterpart is `maxBufferLength`, past which it answers nil.
    ///
    /// Basis: device (the device's `maxMemoryAllocationSize` and `maxBufferSize`).
    BlockLen {
        /// The length asked for, rounded up to the allocation granule.
        len: u64,
        /// The most the device holds in one block.
        max: u64,
    },
    /// A block after every block number has been used
    /// ([`Device::take_block`](crate::Device::take_block)): numbers are never reused during a
    /// device's life. Metal has no counterpart.
    ///
    /// Basis: ours (block numbers are 32 bits).
    BlockIds,
    /// A dispatch of more threadgroups on an axis than the device's `maxComputeWorkGroupCount`.
    /// Metal's grids go further.
    ///
    /// Basis: device (the device's `maxComputeWorkGroupCount`).
    GroupCount {
        /// The threadgroups asked for.
        count: [u32; 3],
        /// The device's most.
        max: [u32; 3],
    },
    /// A threadgroup longer on an axis than the device's `maxComputeWorkGroupSize` (the floor holds
    /// 1,024 by 1,024 by 64). Metal's go to 1,024 on every axis.
    ///
    /// Basis: device (the device's `maxComputeWorkGroupSize`).
    ThreadgroupShape {
        /// The threadgroup size asked for.
        size: [u32; 3],
        /// The device's most.
        max: [u32; 3],
    },
    /// `setBytes` data longer than one parameter ring chunk. Metal documents the call for data
    /// under 4 KB.
    ///
    /// Basis: ours (a ring chunk is 64 KiB).
    BytesLength {
        /// The length asked for.
        len: u64,
        /// The most one call takes.
        max: u64,
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
            Error::Vulkan { call, result } => write!(f, "{call} failed: {result}"),
            Error::Refused(refusal) => write!(f, "{refusal}"),
            Error::Limit(limit) => write!(f, "{limit}"),
            Error::NoSuchBlock(block) => {
                write!(f, "block {} is not one the consumer holds", block.get())
            }
            Error::Thread(error) => write!(f, "a runtime thread did not start: {error}"),
            Error::Quiesced => f.write_str("the device is quiesced"),
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
            Refusal::Committed => f.write_str("the command buffer is already committed"),
            Refusal::OtherDevice => f.write_str("the object is another device's"),
            Refusal::NoCommandBuffers => {
                f.write_str("a command queue must hold at least one command buffer")
            }
            Refusal::EncoderOpen => f.write_str("an encoder of the command buffer is open"),
            Refusal::NoPipeline => f.write_str("no compute pipeline state is set"),
            Refusal::BufferIndex { index } => {
                write!(f, "buffer index {index} is past the 31 entries")
            }
            Refusal::NoBufferAtIndex { index } => write!(f, "no buffer is bound at index {index}"),
            Refusal::ThreadgroupSize { size, max } => write!(
                f,
                "a threadgroup of {size:?} threads is empty or past the state's {max}"
            ),
            Refusal::IndirectOffset { offset, length } => write!(
                f,
                "indirect arguments at offset {offset} of a {length}-byte buffer are misaligned or past its end"
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
            Limit::GroupCount { count, max } => {
                write!(f, "{count:?} threadgroups are past the device's {max:?}")
            }
            Limit::ThreadgroupShape { size, max } => {
                write!(f, "a threadgroup of {size:?} is past the device's {max:?}")
            }
            Limit::BytesLength { len, max } => {
                write!(f, "{len} bytes are past the most one call takes, {max}")
            }
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
            Error::Refused(_) | Error::Limit(_) | Error::NoSuchBlock(_) | Error::Quiesced => None,
        }
    }
}
