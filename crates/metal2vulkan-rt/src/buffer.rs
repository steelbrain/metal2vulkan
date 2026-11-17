//! Buffers: views of a range of a block, with no Vulkan object of their own.

use crate::counters::Gauge;
use crate::device::Device;
use crate::error::{Error, Refusal};
use crate::memory::{self, Block, BlockId, Mapping};
use std::ptr::NonNull;
use std::sync::Arc;

/// `MTLBuffer`: `length` bytes at `offset` in a block the consumer took, with no `VkBuffer` or
/// memory of its own. Shaders reach it by its device address.
///
/// A cheap handle: clones share one buffer. The buffer holds its block and the device, so a
/// released block stays alive until its last buffer is gone.
#[derive(Clone)]
pub struct Buffer {
    inner: Arc<Inner>,
}

struct Inner {
    device: Device,
    block: Arc<Block>,
    offset: u64,
    length: u64,
    contents: Option<Mapping>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.device.core().counters.fall(Gauge::Buffers);
    }
}

impl Device {
    /// `newBufferWithLength:options:offset:` of a placement heap, over a block the consumer
    /// holds. The consumer chooses the range; it reuses a range only once the work that named the
    /// buffer there before has completed, as Metal frees a buffer's memory only then.
    ///
    /// `zeroed` zeroes the range before the call returns, as `newBufferWithLength:options:` does
    /// in Metal: through the mapping in a Shared block, by a GPU fill submitted and waited for in a
    /// Private one. A buffer placed in a heap passes false: Metal does not zero heap memory.
    ///
    /// Err: [`Refusal::EmptyBuffer`] for a length of 0 and [`Refusal::BufferPastBlock`] for a
    /// range that leaves the block, where Metal answers nil; [`Error::NoSuchBlock`] for a block the
    /// consumer does not hold; [`Error::Vulkan`] where a Private fill fails.
    pub fn new_buffer(
        &self,
        block: BlockId,
        offset: u64,
        length: u64,
        zeroed: bool,
    ) -> Result<Buffer, Error> {
        if length == 0 {
            return Err(Error::Refused(Refusal::EmptyBuffer));
        }
        let block = self.block(block)?;
        let block_len = block.memory.len;
        let end = offset
            .checked_add(length)
            .filter(|&end| end <= block_len)
            .ok_or(Error::Refused(Refusal::BufferPastBlock {
                offset,
                length,
                block_len,
            }))?;
        if zeroed {
            memory::zero(self.core(), &block.memory, offset, end)?;
        }
        let contents = block.memory.mapping.map(|mapping| {
            // SAFETY: `offset` is inside the block.
            Mapping::new(unsafe { mapping.at(offset) })
        });
        self.core().counters.rise(Gauge::Buffers);
        Ok(Buffer {
            inner: Arc::new(Inner {
                device: self.clone(),
                block,
                offset,
                length,
                contents,
            }),
        })
    }
}

impl Buffer {
    /// `gpuAddress`: the block's device address plus the buffer's offset.
    pub fn gpu_address(&self) -> u64 {
        self.inner.block.memory.address + self.inner.offset
    }

    /// `length`.
    pub fn length(&self) -> u64 {
        self.inner.length
    }

    /// `contents`: the block's mapping plus the buffer's offset in a Shared block; `None` in a
    /// Private one, where Metal answers `NULL`.
    pub fn contents(&self) -> Option<NonNull<u8>> {
        self.inner.contents.map(Mapping::get)
    }

    /// No Metal counterpart: the block the buffer is in.
    pub fn block(&self) -> BlockId {
        self.inner.block.memory.id
    }

    /// No Metal counterpart: the buffer's offset in its block.
    pub fn offset(&self) -> u64 {
        self.inner.offset
    }
}
