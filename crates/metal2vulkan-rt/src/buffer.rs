//! Buffers: views of a range of a block, with no Vulkan object of their own; a Managed buffer's
//! device copy, where the device has one, is a place in a copy block ([`crate::places`]).

use crate::counters::Gauge;
use crate::device::{Core, Device};
use crate::error::{Error, Refusal};
use crate::fill;
use crate::memory::{Block, BlockId, Mapping, Memory, MemoryClass};
use crate::places::Place;
use crate::submit::Held;
use ash::vk;
use std::ptr::NonNull;
use std::sync::Arc;

/// The longest buffer [`Device::new_buffer`] makes: 2 GiB. A shader reaches a buffer argument's
/// bytes by 32-bit offsets from its bind address, which the translator may read as signed, 2 GiB
/// either side of it; a byte of a longer buffer could lie where no access reaches it. Metal
/// answers nil past a device's `maxBufferLength`; the consumer states one no larger than this.
pub const MAX_BUFFER_LENGTH: u64 = 1 << 31;

/// The alignment of a device copy in its block: a multiple of every storage buffer, uniform buffer
/// and copy offset alignment Vulkan lets a device require.
const COPY_ALIGN: u64 = 256;

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
    /// The block the consumer placed it in: of a Managed buffer, the block of its bytes.
    block: Arc<Block>,
    offset: u64,
    length: u64,
    contents: Option<Mapping>,
    /// A Managed buffer's device copy, on a device whose Shared memory is not device-local.
    copy: Option<Arc<DeviceCopy>>,
}

/// A Managed buffer's device copy: a place in a copy block of Private memory, which shaders reach.
/// The buffer, each command buffer that uses it and each upload that copies into it hold it; the
/// last gone frees its place.
pub(crate) struct DeviceCopy {
    core: Arc<Core>,
    place: Place,
    block: Memory,
}

impl DeviceCopy {
    /// The copy block's buffer.
    pub(crate) fn buffer(&self) -> vk::Buffer {
        self.block.buffer
    }

    /// The copy's offset in [`buffer`](Self::buffer).
    pub(crate) fn offset(&self) -> u64 {
        self.place.offset
    }
}

impl Drop for DeviceCopy {
    fn drop(&mut self) {
        self.core.copies.release(&self.core, self.place);
    }
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
    /// Err: [`Refusal::EmptyBuffer`] for a length of 0, [`Refusal::BufferTooLong`] for a length
    /// past [`MAX_BUFFER_LENGTH`] and [`Refusal::BufferPastBlock`] for a range that leaves the
    /// block, where Metal answers nil; [`Error::NoSuchBlock`] for a block the
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
        if length > MAX_BUFFER_LENGTH {
            return Err(Error::Refused(Refusal::BufferTooLong {
                length,
                max: MAX_BUFFER_LENGTH,
            }));
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
        Ok(self.buffer(block, offset, length, contents, None))
    }

    /// `newBufferWithLength:options:` and `newBufferWithBytes:length:options:` with
    /// `MTLResourceStorageModeManaged`: `length` bytes at `offset` in a Shared block the consumer
    /// holds are the consumer's bytes, which [`Buffer::contents`] answers. Which form the buffer
    /// takes is the device's memory-type query's answer:
    ///
    /// - where the Shared memory type is device-local, the buffer is its bytes, as a Shared buffer
    ///   is, and [`Buffer::did_modify_range`] does nothing;
    /// - elsewhere shaders reach a device copy in Private memory the device holds, at
    ///   [`Buffer::gpu_address`]. Its first `copied` bytes (all of them where `copied` is the
    ///   length or more) are copied from the consumer's bytes, and the rest are zeroed, ahead of
    ///   every command buffer committed after this call; [`Buffer::did_modify_range`] copies a
    ///   range again. The consumer's own bytes are as it wrote them.
    ///
    /// Metal keeps such a copy on a Mac with a discrete GPU. The consumer chooses the range of its
    /// bytes, as for [`Device::new_buffer`]. A buffer a command buffer uses holds its device copy
    /// until the command buffer retires. Where device-local memory has no room for the copy, the
    /// buffer is its bytes, as on a device whose Shared type is device-local: Metal never fails a
    /// Managed buffer for want of video memory, and no second copy is kept in system memory.
    ///
    /// Err: as [`Device::new_buffer`]; [`Refusal::ManagedBlockNotShared`] for a Private block;
    /// [`Error::Quiesced`], and what taking a Private block or the upload's command buffer can
    /// return, for a device copy.
    pub fn new_managed_buffer(
        &self,
        block: BlockId,
        offset: u64,
        length: u64,
        copied: u64,
    ) -> Result<Buffer, Error> {
        if length == 0 {
            return Err(Error::Refused(Refusal::EmptyBuffer));
        }
        if length > MAX_BUFFER_LENGTH {
            return Err(Error::Refused(Refusal::BufferTooLong {
                length,
                max: MAX_BUFFER_LENGTH,
            }));
        }
        let block = self.block(block)?;
        let block_len = block.memory.len;
        offset
            .checked_add(length)
            .filter(|&end| end <= block_len)
            .ok_or(Error::Refused(Refusal::BufferPastBlock {
                offset,
                length,
                block_len,
            }))?;
        let Some(mapping) = block.memory.mapping else {
            return Err(Error::Refused(Refusal::ManagedBlockNotShared {
                block: block.memory.id,
            }));
        };
        // SAFETY: `offset` is inside the block.
        let contents = Some(Mapping::new(unsafe { mapping.at(offset) }));
        let core = self.core();
        if core.shared_is_device_local() {
            return Ok(self.buffer(block, offset, length, contents, None));
        }
        let (place, copy_block) = match core.copies.place(core, length, COPY_ALIGN) {
            Ok(found) => found,
            // No device-local room for the copy: the consumer's Shared range, as on a unified
            // device, chosen here for the buffer's life.
            Err(error) if error.is_out_of_device_memory() => {
                return Ok(self.buffer(block, offset, length, contents, None));
            }
            Err(error) => return Err(error),
        };
        let copy = Arc::new(DeviceCopy {
            core: Arc::clone(core),
            place,
            block: copy_block,
        });
        let copied = copied.min(length);
        if copied < length {
            let start = copy.offset() + copied;
            let end = copy.offset() + length;
            core.work.upload(
                core,
                None,
                [Held::Copy(Arc::clone(&copy))],
                // SAFETY: the upload's command buffer is recording, under the work lock; the copy
                // block's buffer has every usage, `start..end` lies in it, and the upload holds
                // the copy.
                |device, command| unsafe {
                    fill::record(
                        device,
                        command,
                        core.filler.zeros(),
                        copy.buffer(),
                        start,
                        end,
                    )
                },
            )?;
        }
        if copied > 0 {
            core.work.modify(core, &copy, &block, offset, 0, copied)?;
        }
        Ok(self.buffer(block, offset, length, contents, Some(copy)))
    }

    /// A new buffer's handle, counted in [`Gauges::buffers`](crate::Gauges::buffers).
    fn buffer(
        &self,
        block: Arc<Block>,
        offset: u64,
        length: u64,
        contents: Option<Mapping>,
        copy: Option<Arc<DeviceCopy>>,
    ) -> Buffer {
        self.core().counters.rise(Gauge::Buffers);
        Buffer {
            inner: Arc::new(Inner {
                device: self.clone(),
                id: self.core().ids.take(),
                block,
                offset,
                length,
                contents,
                copy,
            }),
        }
    }
}

impl Buffer {
    /// `gpuAddress`: the block's device address plus the buffer's offset; of a Managed buffer with
    /// a device copy, the copy's address.
    pub fn gpu_address(&self) -> u64 {
        match &self.inner.copy {
            Some(copy) => copy.block.address + copy.offset(),
            None => self.inner.block.memory.address + self.inner.offset,
        }
    }

    /// `didModifyRange:`: the consumer changed `length` bytes from `start` of the buffer's bytes.
    /// Of a Managed buffer with a device copy, the next submission's upload copies them into the
    /// copy, ahead of every command buffer committed after this call on every queue; the copy
    /// reads the bytes when the GPU reaches it, so a later write the consumer makes before then
    /// may be seen. Of any other buffer it does nothing, as the GPU reads the consumer's bytes.
    ///
    /// Err: [`Refusal::ModifiedRange`] for a range that leaves the buffer; for a copy,
    /// [`Error::Quiesced`], [`Error::Vulkan`] where the upload's command buffer cannot be made,
    /// and with `VK_ERROR_DEVICE_LOST` where the device is lost while the call waits for the
    /// upload to have room.
    pub fn did_modify_range(&self, start: u64, length: u64) -> Result<(), Error> {
        let inner = &self.inner;
        let end = (start.checked_add(length))
            .filter(|&end| end <= inner.length)
            .ok_or(Error::Refused(Refusal::ModifiedRange {
                start,
                length,
                buffer_length: inner.length,
            }))?;
        let Some(copy) = &inner.copy else {
            return Ok(());
        };
        if length == 0 {
            return Ok(());
        }
        let core = inner.device.core();
        core.work
            .modify(core, copy, &inner.block, inner.offset, start, end)
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

    /// Whether shaders reach the buffer in Shared memory, which the host reads.
    pub(crate) fn is_shared(&self) -> bool {
        self.inner.copy.is_none() && self.inner.block.memory.class == MemoryClass::Shared
    }

    /// The device it was made on.
    pub(crate) fn device(&self) -> &Device {
        &self.inner.device
    }

    /// The block, and a Managed buffer's device copy, for a command buffer to keep.
    pub(crate) fn held(&self) -> (&Arc<Block>, Option<&Arc<DeviceCopy>>) {
        (&self.inner.block, self.inner.copy.as_ref())
    }

    /// The `VkBuffer` shaders reach the buffer through and the buffer's offset in it.
    pub(crate) fn device_range(&self) -> (vk::Buffer, u64) {
        match &self.inner.copy {
            Some(copy) => (copy.buffer(), copy.offset()),
            None => (self.inner.block.memory.buffer, self.inner.offset),
        }
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
