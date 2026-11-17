//! Buffers: views of a range of a block, with no Vulkan object of their own.

use crate::counters::Gauge;
use crate::device::Device;
use crate::error::{Error, Refusal};
use crate::memory::{Block, BlockId, Mapping, MemoryClass};
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
    /// Its resource id, for the hazard sets.
    id: u32,
    block: Arc<Block>,
    offset: u64,
    length: u64,
    contents: Option<Mapping>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        let core = self.device.core();
        core.ids.give_back(self.id);
        core.counters.fall(Gauge::Buffers);
    }
}

impl Device {
    /// `newBufferWithLength:options:offset:` of a placement heap, over a block the consumer
    /// holds. The consumer chooses the range; it reuses a range only once the work that named the
    /// buffer there before has completed, as Metal frees a buffer's memory only then.
    ///
    /// `zeroed` zeroes the range, as `newBufferWithLength:options:` does in Metal: in a Shared
    /// block through the mapping, before the call returns; in a Private block by a GPU fill in the
    /// upload, which is submitted ahead of every command buffer committed after this call and
    /// holds the block until it completes. A buffer placed in a heap passes false: Metal does not
    /// zero heap memory.
    ///
    /// Err: [`Refusal::EmptyBuffer`] for a length of 0 and [`Refusal::BufferPastBlock`] for a
    /// range that leaves the block, where Metal answers nil; [`Error::NoSuchBlock`] for a block the
    /// consumer does not hold; for a zeroed Private buffer, [`Error::Quiesced`], and
    /// [`Error::Vulkan`] where the upload's command buffer cannot be made.
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
        let contents = block.memory.mapping.map(|mapping| {
            // SAFETY: `offset` is inside the block.
            Mapping::new(unsafe { mapping.at(offset) })
        });
        if zeroed {
            match contents {
                // SAFETY: `offset..end` lies in the block, and the range is the new buffer's.
                Some(contents) => unsafe {
                    contents.get().as_ptr().write_bytes(0, length as usize)
                },
                None => self.core().work.zero(self.core(), &block, offset, end)?,
            }
        }
        self.core().counters.rise(Gauge::Buffers);
        Ok(Buffer {
            inner: Arc::new(Inner {
                device: self.clone(),
                id: self.core().ids.take(),
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

    /// Whether the buffer is in Shared memory, which the host reads.
    pub(crate) fn is_shared(&self) -> bool {
        self.held_block().memory.class == MemoryClass::Shared
    }

    /// The block, for a command buffer to keep.
    pub(crate) fn held_block(&self) -> &Arc<Block> {
        &self.inner.block
    }

    /// Its resource id, for the hazard sets.
    pub(crate) fn id(&self) -> u32 {
        self.inner.id
    }

    /// Whether `other` is the device this buffer was made on.
    pub(crate) fn is_of(&self, core: &Arc<crate::device::Core>) -> bool {
        Arc::ptr_eq(self.inner.device.core(), core)
    }
}
