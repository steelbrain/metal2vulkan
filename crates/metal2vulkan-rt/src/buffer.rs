//! Buffers: views of a range of a block, with no Vulkan object of their own; a Managed buffer's
//! device copy, where the device has one, is a range of a Private block the consumer placed it in.

use crate::capture::Named;
use crate::counters::Gauge;
use crate::device::{Core, Device};
use crate::error::{Error, Refusal};
use crate::fill;
use crate::memory::{Block, BlockId, Mapping, Memory, MemoryClass};
use crate::over::ManagedOvers;
use crate::submit::Held;
use ash::vk;
use std::ptr::NonNull;
use std::sync::Arc;

/// The longest buffer [`Device::new_buffer`] makes: 2 GiB. A shader reaches a buffer argument's
/// bytes by 32-bit offsets from its bind address, which the translator may read as signed, 2 GiB
/// either side of it; a byte of a longer buffer could lie where no access reaches it. Metal
/// answers nil past a device's `maxBufferLength`; the consumer states one no larger than this.
pub const MAX_BUFFER_LENGTH: u64 = 1 << 31;

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
    /// Its textures over it in images of their own, as a Managed buffer with a device copy keeps
    /// them (`crate::over::ManagedOvers`); a capture of its ranges holds them too.
    overs: Arc<ManagedOvers>,
}

/// A Managed buffer's device copy: `len` bytes at `offset` in a Private block the consumer placed
/// it in, which shaders reach. The buffer, each command buffer that uses it and each upload that
/// copies into it hold it, and so the block. Where the block spilled to another memory type than
/// the Private one, its bytes count in
/// [`Gauges::managed_spilled_bytes`](crate::Gauges::managed_spilled_bytes) while it lives.
pub(crate) struct DeviceCopy {
    core: Arc<Core>,
    block: Arc<Block>,
    offset: u64,
    len: u64,
}

impl DeviceCopy {
    /// The copy block's buffer.
    pub(crate) fn buffer(&self) -> vk::Buffer {
        self.block.memory.buffer
    }

    /// The copy block's memory.
    pub(crate) fn memory(&self) -> &Memory {
        &self.block.memory
    }

    /// The copy's offset in [`buffer`](Self::buffer).
    pub(crate) fn offset(&self) -> u64 {
        self.offset
    }
}

impl Drop for DeviceCopy {
    fn drop(&mut self) {
        if self.block.memory.spilled() {
            (self.core.counters).fall_by(Gauge::ManagedSpilledBytes, self.len);
        }
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
    /// A block's bytes are undefined ([`Device::take_block`]). `zero_from` zeroes the buffer's
    /// bytes from that offset in it to its end: `Some(0)` all of them, as
    /// `newBufferWithLength:options:` zeroes a new buffer in Metal, and `None` none, for a buffer
    /// placed in a heap, whose memory Metal does not zero, or one whose bytes the consumer writes
    /// itself. An offset at or past the length zeroes nothing. In a Shared block the bytes are
    /// zeroed through the mapping, before the call returns; in a Private block by a GPU fill in the
    /// upload, which is submitted ahead of every command buffer committed after this call and ends
    /// with a barrier that orders the fill before every later command (a copy into the buffer
    /// among them), and which holds the block until it completes. The bytes zeroed are counted in
    /// [`Metrics::bytes_zeroed`](crate::Metrics::bytes_zeroed).
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
        zero_from: Option<u64>,
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
        let block = self.placed(block, offset, length)?;
        let end = offset + length;
        let contents = block.memory.mapping.map(|mapping| {
            // SAFETY: `offset` is inside the block.
            Mapping::new(unsafe { mapping.at(offset) })
        });
        if let Some(from) = zero_from.filter(|&from| from < length) {
            match contents {
                // SAFETY: `from..length` lies in the new buffer, whose range is in the block.
                Some(contents) => unsafe {
                    (contents.get().as_ptr())
                        .add(from as usize)
                        .write_bytes(0, (length - from) as usize)
                },
                None => (self.core().work).zero(self.core(), &block, offset + from, end)?,
            }
            self.core().counters.zeroed(length - from);
        }
        Ok(self.buffer(block, offset, length, contents, None))
    }

    /// `newBufferWithLength:options:`, `newBufferWithBytes:length:options:` and
    /// `newBufferWithBytesNoCopy:length:options:deallocator:` with
    /// `MTLResourceStorageModeManaged`: `length` bytes at `bytes` (a block the consumer holds and
    /// the offset in it) are the consumer's bytes, which [`Buffer::contents`] answers. The block is
    /// one the host maps: a Shared block. Which form the buffer takes is the device's memory-type
    /// query's answer:
    ///
    /// - where Vulkan's device type is integrated or CPU ([`Device::has_unified_memory`]), the
    ///   buffer is its bytes, as a Shared buffer is, its [`Buffer::gpu_address`] the bytes' block's
    ///   address plus their offset, and [`Buffer::did_modify_range`] does nothing. `copy` is
    ///   `None`;
    /// - elsewhere shaders reach a device copy at `copy`, `length` bytes at that offset in a
    ///   Private block the consumer holds, its [`Buffer::gpu_address`] that block's address plus
    ///   the offset. The rest past its first `copied` bytes is zeroed, ahead of every command
    ///   buffer committed after this call, and those first bytes (all of them where `copied` is
    ///   the length or more) are named as [`Buffer::did_modify_range`] names a range: the next
    ///   commit takes them from the consumer's bytes. The consumer's own bytes are as it wrote
    ///   them.
    ///
    /// Metal keeps such a copy on a Mac with a discrete GPU, and places it itself. Here the
    /// consumer places both, as it places every buffer ([`Device::new_buffer`]): any byte offset,
    /// and it reuses either range only once the work that named the buffer there before has
    /// completed. The runtime writes the copy's `length` bytes and no byte past them, so the
    /// slack the consumer leaves after either range is the consumer's. A buffer a command buffer
    /// uses holds its device copy's block until the command buffer retires. A copy placed in a
    /// Private block that spilled to another memory type is still the buffer's second copy, so
    /// the two-copy rule holds on every such device; its kernels then read host memory, its bytes
    /// counted in [`Gauges::managed_spilled_bytes`](crate::Gauges::managed_spilled_bytes) while the
    /// buffer lives.
    ///
    /// Err: as [`Device::new_buffer`], for either range; [`Refusal::ManagedBlockNotShared`] for
    /// bytes in a block the host does not map; [`Refusal::ManagedCopyMissing`] for no `copy` where
    /// the device keeps one, [`Refusal::ManagedCopyOnUnified`] for a `copy` where it keeps none
    /// and [`Refusal::ManagedCopyNotPrivate`] for a `copy` in a Shared block; for a device copy,
    /// [`Error::Quiesced`], and [`Error::Vulkan`] where the upload's command buffer cannot be made.
    pub fn new_managed_buffer(
        &self,
        bytes: (BlockId, u64),
        copy: Option<(BlockId, u64)>,
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
        let (block, offset) = bytes;
        let block = self.placed(block, offset, length)?;
        let Some(mapping) = block.memory.mapping else {
            return Err(Error::Refused(Refusal::ManagedBlockNotShared {
                block: block.memory.id,
            }));
        };
        let core = self.core();
        let copy = match (core.unified_memory(), copy) {
            (true, None) => None,
            (true, Some(_)) => return Err(Error::Refused(Refusal::ManagedCopyOnUnified)),
            (false, None) => return Err(Error::Refused(Refusal::ManagedCopyMissing)),
            (false, Some((copy_block, copy_offset))) => {
                let copy_block = self.placed(copy_block, copy_offset, length)?;
                if copy_block.memory.class != MemoryClass::Private {
                    return Err(Error::Refused(Refusal::ManagedCopyNotPrivate {
                        block: copy_block.memory.id,
                    }));
                }
                Some((copy_block, copy_offset))
            }
        };
        // SAFETY: `offset` is inside the block.
        let contents = Some(Mapping::new(unsafe { mapping.at(offset) }));
        let Some((copy_block, copy_offset)) = copy else {
            return Ok(self.buffer(block, offset, length, contents, None));
        };
        if copy_block.memory.spilled() {
            core.counters.rise_by(Gauge::ManagedSpilledBytes, length);
        }
        let copy = Arc::new(DeviceCopy {
            core: Arc::clone(core),
            block: copy_block,
            offset: copy_offset,
            len: length,
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
            core.counters.zeroed(length - copied);
        }
        let buffer = self.buffer(block, offset, length, contents, Some(copy));
        if copied > 0 {
            buffer.did_modify_range(0, copied)?;
        }
        Ok(buffer)
    }

    /// The block `block`, which the consumer holds, where `length` bytes at `offset` lie in it.
    ///
    /// Err: [`Error::NoSuchBlock`]; [`Refusal::BufferPastBlock`].
    fn placed(&self, block: BlockId, offset: u64, length: u64) -> Result<Arc<Block>, Error> {
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
        Ok(block)
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
                overs: Arc::default(),
            }),
        }
    }
}

impl Buffer {
    /// `gpuAddress`: the block's device address plus the buffer's offset; of a Managed buffer with
    /// a device copy, the copy block's device address plus the copy's offset.
    pub fn gpu_address(&self) -> u64 {
        match &self.inner.copy {
            Some(copy) => copy.block.memory.address + copy.offset(),
            None => self.inner.block.memory.address + self.inner.offset,
        }
    }

    /// `didModifyRange:`: the consumer changed `length` bytes from `start` of the buffer's bytes.
    /// Of a Managed buffer with a device copy, the range is named, and the next commit of a command
    /// buffer with commands, on any queue, copies the consumer's bytes of every range named since
    /// into staging memory before it returns, once each, merged where they overlap or touch. The
    /// copies into the device copy, and into the images of its textures over the range in images
    /// of their own, go ahead of that command buffer's commands. So the command buffers committed
    /// after this call read the bytes as they were at that commit; Metal takes them shortly after
    /// the commit, before the command buffer is scheduled. A write or a didModifyRange made while
    /// a command buffer runs does not reach it. A commit naming more than the staging ring holds
    /// sends its ring-fulls ahead and waits for them, or, behind a command buffer held on the
    /// host, takes staging of its own for the rest (the crate's README, Memory blocks and
    /// buffers). Of any other buffer it does nothing, as the GPU reads the consumer's bytes.
    ///
    /// Err: [`Refusal::ModifiedRange`] for a range that leaves the buffer; for a copy,
    /// [`Error::Quiesced`].
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
        // A Managed buffer's block is mapped (Refusal::ManagedBlockNotShared otherwise).
        let Some(contents) = inner.contents.filter(|_| length > 0) else {
            return Ok(());
        };
        let named = Named {
            copy,
            cpu: &inner.block,
            contents,
            offset: inner.offset,
            overs: &inner.overs,
        };
        inner.device.core().work.modify(named, start, end)
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

    /// Its textures over its device copy in images of their own, where it is a Managed buffer
    /// with a device copy and has one (`crate::over::ManagedOvers`).
    pub(crate) fn managed_overs(&self) -> Option<&ManagedOvers> {
        (self.inner.copy.is_some() && self.inner.overs.any()).then_some(&*self.inner.overs)
    }

    /// Its textures over it in images of their own, as it keeps them.
    pub(crate) fn over_entries(&self) -> &ManagedOvers {
        &self.inner.overs
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
