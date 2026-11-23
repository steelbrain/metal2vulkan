//! Buffers: views of a range of a block, with no Vulkan object of their own; a Managed buffer's
//! device copy, where the device has one, is a range of a Private block the consumer placed it in.

use crate::capture::Modified;
use crate::counters::{Gauge, Part};
use crate::device::{Core, Device};
use crate::error::{Error, Refusal};
use crate::hazard::HeapId;
use crate::memory::{Block, BlockId, Mapping, Memory, MemoryClass};
use crate::over::{ManagedOvers, Span};
use crate::submit::Held;
use crate::texture::{HeapRange, SizeAndAlign};
use ash::vk;
use std::ptr::NonNull;
use std::sync::Arc;

/// The longest buffer [`Device::new_buffer`] makes: 2 GiB. A shader reaches a buffer argument's
/// bytes by 32-bit offsets from its bind address, which the translator may read as signed, 2 GiB
/// either side of it; a byte of a longer buffer could lie where no access reaches it. Metal
/// answers nil past a device's `maxBufferLength`; the consumer states one no larger than this.
pub const MAX_BUFFER_LENGTH: u64 = 1 << 31;

/// What a buffer's offset in a heap is a multiple of ([`Device::heap_buffer_size_and_align`]):
/// 256 bytes on every device, Metal's answer (an Apple M3 Max, macOS 27.0.1, the conformance
/// recording `device-heap-buffer-sizes`).
///
/// The runtime takes a buffer at any byte of a block ([`Device::new_buffer`]); what it asks of
/// a buffer's place is per use, and every such rule holds for a buffer that starts at a multiple
/// of 256 of its block, whose own first byte lies at a multiple of 64 KiB of its allocation: a
/// bound argument's alignment, at most 16
/// ([`Refusal::BufferOffsetAlignment`]), indirect arguments at a multiple of 4 and indices at a
/// multiple of their size ([`Refusal::ArgumentPlace`]), a fill's 4-byte words and a copy's texel
/// block. It is also no less than any device may ask of a buffer's offset: Vulkan bounds
/// `minUniformBufferOffsetAlignment`, `minStorageBufferOffsetAlignment` and
/// `minTexelBufferOffsetAlignment` at 256. So it is a constant, not a device query: an answer
/// that moved with the host would buy nothing. A placement rule added later changes it here.
const HEAP_BUFFER_ALIGNMENT: u64 = 256;

/// [`Device::heap_buffer_size_and_align`]'s answer for `length` bytes.
pub(crate) fn heap_buffer_size_and_align(length: u64) -> SizeAndAlign {
    SizeAndAlign {
        size: length,
        align: HEAP_BUFFER_ALIGNMENT,
    }
}

/// `MTLBuffer`: `length` bytes at `offset` in a block the consumer took, with no `VkBuffer` or
/// memory of its own. Shaders reach it by its device address.
///
/// A cheap handle: clones share one buffer. A handle holds the buffer and the device; the buffer
/// holds its block, so a released block stays alive until its last buffer is gone. What a command
/// buffer keeps of it is the buffer alone, which holds no handle of the device.
#[derive(Clone)]
pub struct Buffer {
    inner: Arc<BufferInner>,
    device: Device,
}

/// The buffer itself: what a handle, a command buffer that uses it and a texture over it hold.
pub(crate) struct BufferInner {
    core: Arc<Core>,
    /// Its resource id, for the hazard sets.
    id: u32,
    /// The heap the consumer placed it in ([`Device::new_heap_buffer`]), whose id its uses take in
    /// the hazard sets.
    heap: Option<Arc<HeapId>>,
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
    /// The `VkBuffer` commands reach the copy's byte `offset` through, and that byte's offset in
    /// it ([`Memory::range`]).
    pub(crate) fn range(&self, offset: u64) -> (vk::Buffer, u64) {
        self.block.memory.range(self.offset + offset)
    }

    /// The copy block's memory.
    pub(crate) fn memory(&self) -> &Memory {
        &self.block.memory
    }

    /// The copy's offset in its block.
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

impl BufferInner {
    /// A capture's take of `ranges` of the buffer ([`crate::capture::Changes::take`]), holding its
    /// blocks from here; none where it has no device copy.
    pub(crate) fn modified(self: &Arc<Self>, ranges: Vec<(u64, u64)>) -> Option<Modified> {
        let copy = self.copy.as_ref()?;
        Some(Modified {
            buffer: Arc::downgrade(self),
            copy: Arc::clone(copy),
            cpu: Arc::clone(&self.block),
            offset: self.offset,
            overs: Arc::clone(&self.overs),
            ranges,
        })
    }
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        let core = &self.core;
        core.ids.give_back(self.id);
        core.counters.fall(Gauge::Buffers);
        let gauge = self.block.memory.class.buffer_gauge();
        core.counters.fall_by(gauge, self.length);
    }
}

impl Device {
    /// `heapBufferSizeAndAlignWithLength:options:`: the bytes and alignment a buffer of `length`
    /// bytes takes in a heap, `length` itself at a multiple of 256, whatever the options (Metal's
    /// raw `MTLResourceOptions`, which the answer does not read). It is Metal's answer: an Apple
    /// M3 Max under macOS 27.0.1 answers the length unrounded and 256 for lengths from 1 to
    /// 1,048,576 and the default, Private, Managed, write-combined and untracked options (the
    /// conformance recording `device-heap-buffer-sizes`). What differs: nothing, but that it is
    /// the same on every device, where Metal's answer is its GPU's. The alignment is the runtime's
    /// placement rule: every rule a use of a buffer has (an argument's alignment, indirect
    /// arguments' and indices' places, a fill's words, a copy's texel blocks) holds for a buffer
    /// that starts at a multiple of 256 of its block, and Vulkan bounds every buffer offset
    /// alignment a device may ask at 256. So a buffer the consumer places by these answers
    /// ([`Device::new_buffer`]) meets every use the runtime makes of it. No Vulkan call is made.
    pub fn heap_buffer_size_and_align(&self, length: u64, _options: u64) -> SizeAndAlign {
        heap_buffer_size_and_align(length)
    }

    /// `newBufferWithLength:options:offset:` of a placement heap, over a block the consumer
    /// holds. The consumer chooses the range; it reuses a range only once the work that named the
    /// buffer there before has completed, as Metal frees a buffer's memory only then.
    ///
    /// A block's bytes are undefined ([`Device::take_block`]). `zero_from` zeroes the buffer's
    /// bytes from that offset in it to its end: `Some(0)` all of them, as
    /// `newBufferWithLength:options:` zeroes a new buffer in Metal, and `None` none, for a buffer
    /// placed in a heap, whose memory Metal does not zero, or one whose bytes the consumer writes
    /// itself. An offset at or past the length zeroes nothing. In a block the host maps (Shared,
    /// or host memory imported by its address) the bytes are zeroed through the mapping, before
    /// the call returns; in one it does not map (Private, or an imported dma-buf) by a GPU fill in
    /// the upload of the next commit, which is submitted ahead of every command buffer committed
    /// after this call, the last upload of its submission ending with a barrier that orders the
    /// fill before every later command (a copy into the buffer among them), and which holds the
    /// block until it completes. The bytes zeroed are counted in
    /// [`Metrics::bytes_zeroed`](crate::Metrics::bytes_zeroed).
    ///
    /// Err: [`Refusal::EmptyBuffer`] for a length of 0, [`Refusal::BufferTooLong`] for a length
    /// past [`MAX_BUFFER_LENGTH`] and [`Refusal::BufferPastBlock`] for a range that leaves the
    /// block, where Metal answers nil; [`Error::NoSuchBlock`] for a block the
    /// consumer does not hold; for a buffer zeroed by a GPU fill, [`Refusal::InScheduledHandler`]
    /// in a scheduled handler, [`Error::Quiesced`], and [`Error::Vulkan`] where the upload's
    /// command buffer cannot be made, or what making the zero source returns where a fill copies
    /// from it and the block-maker thread made none.
    pub fn new_buffer(
        &self,
        block: BlockId,
        offset: u64,
        length: u64,
        zero_from: Option<u64>,
    ) -> Result<Buffer, Error> {
        self.placed_buffer(block, offset, length, zero_from, None)
    }

    /// `MTLHeap`'s `newBufferWithLength:options:offset:`: a buffer of `length` bytes at `offset`
    /// in `heap`, where the consumer's own placement put it
    /// ([`Device::heap_buffer_size_and_align`]), made as [`Device::new_buffer`] makes one at the
    /// heap's offset plus `offset` in its block, with none of its bytes zeroed: Metal does not zero
    /// a heap's memory. Its uses take the heap's id in the compute encoders' hazard sets, as the
    /// uses of a texture placed in the heap do ([`Device::new_heap_texture`]): Metal tracks a
    /// heap's resources as one, so within a command buffer a write through any of them orders
    /// every later use of any other, whatever bytes each reaches. The heap is the one at its
    /// offset in its block; its id lives while a resource placed in it does.
    ///
    /// Err: [`Refusal::HeapBufferPlace`] for a range that leaves the heap; as
    /// [`Device::new_buffer`] otherwise.
    pub fn new_heap_buffer(
        &self,
        heap: HeapRange,
        offset: u64,
        length: u64,
    ) -> Result<Buffer, Error> {
        if offset.checked_add(length).is_none_or(|end| end > heap.len) {
            return Err(Error::Refused(Refusal::HeapBufferPlace {
                offset,
                length,
                heap_len: heap.len,
            }));
        }
        let id = (self.core().heap_ids).of(self.core(), heap.block, heap.offset);
        let at = heap.offset.saturating_add(offset);
        self.placed_buffer(heap.block, at, length, None, Some(id))
    }

    /// [`Device::new_buffer`], the buffer taking `heap`'s id in the hazard sets where it has one.
    fn placed_buffer(
        &self,
        block: BlockId,
        offset: u64,
        length: u64,
        zero_from: Option<u64>,
        heap: Option<Arc<HeapId>>,
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
        // A block the host does not map (Private, or an imported dma-buf) is zeroed by a GPU fill.
        if block.memory.mapping.is_none() && zero_from.is_some_and(|from| from < length) {
            crate::command::not_in_scheduled_handler()?;
        }
        let end = offset + length;
        let contents = block.memory.mapping.map(|mapping| {
            // SAFETY: `offset` is inside the block.
            Mapping::new(unsafe { mapping.at(offset) })
        });
        if let Some(from) = zero_from.filter(|&from| from < length) {
            match contents {
                Some(contents) => {
                    let started = crate::host_time();
                    // SAFETY: `from..length` lies in the new buffer, whose range is in the block.
                    unsafe {
                        (contents.get().as_ptr())
                            .add(from as usize)
                            .write_bytes(0, (length - from) as usize)
                    };
                    let nanos = crate::host_time() - started;
                    (self.core().counters).timed(Part::ZeroWrite, length - from, nanos, false);
                    if block.memory.len > MemoryClass::Shared.block_len() {
                        let part = Part::LongTakeZeroWrite;
                        (self.core().counters).timed(part, length - from, nanos, false);
                    }
                }
                None => (self.core().work).zero(self.core(), &block, offset + from, end)?,
            }
            self.core().counters.zeroed(length - from);
        }
        Ok(self.buffer(block, offset, length, contents, None, heap))
    }

    /// `newBufferWithLength:options:`, `newBufferWithBytes:length:options:` and
    /// `newBufferWithBytesNoCopy:length:options:deallocator:` with
    /// `MTLResourceStorageModeManaged`: `length` bytes at `bytes` (a Shared block the consumer
    /// holds and the offset in it) are the consumer's bytes, which [`Buffer::contents`] answers
    /// where the runtime maps the block (a block it took from the pool), and `None` where it does
    /// not (an imported block, whose pages the consumer reaches through its own mapping, as for
    /// [`Device::new_buffer`]). Which form the buffer takes is the device's memory-type query's
    /// answer:
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
    ///   commit copies them from the consumer's bytes on the GPU. The consumer's own bytes are as
    ///   it wrote them. The consumer's block lives until every copy from it has run.
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
    /// Err: [`Refusal::InScheduledHandler`] in a scheduled handler, on every device; as
    /// [`Device::new_buffer`], for either range; [`Refusal::ManagedBlockPrivate`] for bytes in a
    /// Private block; [`Refusal::ManagedCopyMissing`] for no `copy` where
    /// the device keeps one, [`Refusal::ManagedCopyOnUnified`] for a `copy` where it keeps none
    /// [`Refusal::ManagedCopyNotPrivate`] for a `copy` in a Shared block and
    /// [`Refusal::ManagedCopyExported`] for one in an exported block; for a device copy,
    /// [`Error::Quiesced`], and [`Error::Vulkan`] where the upload's command buffer cannot be made,
    /// or what making the zero source returns where the fill copies from it and the block-maker
    /// thread made none; for one whose first bytes are copied, what [`Buffer::did_modify_range`]
    /// returns.
    pub fn new_managed_buffer(
        &self,
        bytes: (BlockId, u64),
        copy: Option<(BlockId, u64)>,
        length: u64,
        copied: u64,
    ) -> Result<Buffer, Error> {
        crate::command::not_in_scheduled_handler()?;
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
        if block.memory.class == MemoryClass::Private {
            return Err(Error::Refused(Refusal::ManagedBlockPrivate {
                block: block.memory.id,
            }));
        }
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
                if copy_block.memory.exported() {
                    return Err(Error::Refused(Refusal::ManagedCopyExported {
                        block: copy_block.memory.id,
                    }));
                }
                Some((copy_block, copy_offset))
            }
        };
        let contents = block.memory.mapping.map(|mapping| {
            // SAFETY: `offset` is inside the block.
            Mapping::new(unsafe { mapping.at(offset) })
        });
        let Some((copy_block, copy_offset)) = copy else {
            return Ok(self.buffer(block, offset, length, contents, None, None));
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
            let (copy_buffer, start) = copy.range(copied);
            let end = start + (length - copied);
            let held = Held::Copy(Arc::clone(&copy));
            core.work.fill(core, held, copy_buffer, start, end)?;
            core.counters.zeroed(length - copied);
        }
        let buffer = self.buffer(block, offset, length, contents, Some(copy), None);
        if copied > 0 {
            buffer.did_modify_range(0, copied)?;
        }
        Ok(buffer)
    }

    /// Nearest Metal counterpart: the zeroed contents of a new buffer of
    /// `newBufferWithLength:options:`, for `length` bytes at `offset` of a Private block the
    /// consumer holds, where the consumer places such a buffer.
    ///
    /// What differs: the consumer asks, and the runtime infers nothing. A take zeroes no block,
    /// and this zeroes only the range named. The range is zeroed by a GPU fill in the upload, as
    /// [`Device::new_buffer`]'s zeroing of a Private buffer is: submitted ahead of every command
    /// buffer committed after this call, ending with a barrier that orders the fill before every
    /// later command, and holding the block until it completes. The call never waits for the
    /// GPU. Metal owes zeros for that call's buffers only, not for a heap's memory or a
    /// texture's. [`Metrics::block_zero_fills`](crate::Metrics::block_zero_fills) and
    /// [`Metrics::block_zero_fill_bytes`](crate::Metrics::block_zero_fill_bytes) count the fills
    /// and their bytes. A length of 0 zeroes nothing.
    ///
    /// Err: [`Refusal::InScheduledHandler`] in a scheduled handler; [`Error::NoSuchBlock`] for a
    /// block the consumer does not hold; [`Refusal::ZeroRangePastBlock`] for a range that leaves
    /// the block;
    /// [`Refusal::ZeroRangeShared`] for a Shared block, whose bytes the consumer writes through
    /// the mapping; [`Error::Quiesced`], and [`Error::Vulkan`] where the upload's command buffer
    /// cannot be made, or what making the zero source returns where the fill copies from it and
    /// the block-maker thread made none.
    pub fn zero_block_range(&self, block: BlockId, offset: u64, length: u64) -> Result<(), Error> {
        crate::command::not_in_scheduled_handler()?;
        let block = self.block(block)?;
        let block_len = block.memory.len;
        let end = offset
            .checked_add(length)
            .filter(|&end| end <= block_len)
            .ok_or(Error::Refused(Refusal::ZeroRangePastBlock {
                offset,
                length,
                block_len,
            }))?;
        if block.memory.class != MemoryClass::Private {
            return Err(Error::Refused(Refusal::ZeroRangeShared {
                block: block.memory.id,
            }));
        }
        if length == 0 {
            return Ok(());
        }
        (self.core().work).zero(self.core(), &block, offset, end)?;
        self.core().counters.zero_filled(length);
        Ok(())
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

    /// A new buffer's handle, counted in [`Gauges::buffers`](crate::Gauges::buffers), placed in
    /// `heap` where it has one: a heap's, or an exported surface's ([`crate::Surface::buffer`]).
    pub(crate) fn buffer(
        &self,
        block: Arc<Block>,
        offset: u64,
        length: u64,
        contents: Option<Mapping>,
        copy: Option<Arc<DeviceCopy>>,
        heap: Option<Arc<HeapId>>,
    ) -> Buffer {
        Buffer {
            inner: BufferInner::new(self.core(), block, offset, length, contents, copy, heap),
            device: self.clone(),
        }
    }
}

impl BufferInner {
    /// A new buffer of `core`'s device, counted in [`Gauges::buffers`](crate::Gauges::buffers):
    /// what [`Device::buffer`] makes a handle of, and a visible-function table's words
    /// (`crate::function_table`).
    pub(crate) fn new(
        core: &Arc<Core>,
        block: Arc<Block>,
        offset: u64,
        length: u64,
        contents: Option<Mapping>,
        copy: Option<Arc<DeviceCopy>>,
        heap: Option<Arc<HeapId>>,
    ) -> Arc<BufferInner> {
        core.counters.rise(Gauge::Buffers);
        let gauge = block.memory.class.buffer_gauge();
        core.counters.rise_by(gauge, length);
        Arc::new(BufferInner {
            core: Arc::clone(core),
            id: core.ids.take(),
            heap,
            block,
            offset,
            length,
            contents,
            copy,
            overs: Arc::default(),
        })
    }
}

impl crate::kept::Handle for Buffer {
    type Kept = Arc<BufferInner>;

    fn object(&self) -> &Arc<BufferInner> {
        &self.inner
    }

    fn address_of(object: &Arc<BufferInner>) -> usize {
        Arc::as_ptr(object) as *const () as usize
    }
}

impl Buffer {
    /// `gpuAddress`: the block's device address plus the buffer's offset; of a Managed buffer with
    /// a device copy, the copy block's device address plus the copy's offset. The first call that
    /// answers an address in an exported block counts in
    /// [`Metrics::exported_blocks_addressed`](crate::Metrics::exported_blocks_addressed): from
    /// then on a write through an address makes the textures over that block owe their copies in
    /// ([`Buffer::new_texture`]).
    pub fn gpu_address(&self) -> u64 {
        BufferObject::address_out(self);
        BufferObject::gpu_address(self)
    }

    /// `didModifyRange:`: the consumer changed `length` bytes from `start` of the buffer's bytes.
    /// Of a Managed buffer with a device copy, the range is named, and the next commit of a command
    /// buffer, with commands or none, on any queue, takes every range named since, once each,
    /// merged where they overlap or touch: GPU copies from the consumer's block into the device
    /// copy, and into the images of its textures over the range in images of their own, go ahead
    /// of that command buffer's commands in its submission. One with nothing for the GPU (no
    /// commands, or only dispatches that record none) submits the copies alone and completes after
    /// them. So a command buffer committed after this call reads the bytes the consumer wrote
    /// before its commit, as Metal owes. Metal on a discrete Mac GPU (an AMD Radeon Pro 5300M)
    /// takes a named range at the first commit whose encoder binds its buffer; this takes it at
    /// the next commit of any command buffer, the same commit where that one binds the buffer, and
    /// never a later one. The copy reads the consumer's bytes as it runs, so a write made after the
    /// commit may reach it, where Metal leaves the range undefined for the GPU. Of any other
    /// buffer it does nothing, as the GPU reads the consumer's bytes. It allocates nothing.
    ///
    /// Err: [`Refusal::ModifiedRange`] for a range that leaves the buffer; for a copy,
    /// [`Error::Quiesced`].
    pub fn did_modify_range(&self, start: u64, length: u64) -> Result<(), Error> {
        BufferObject::did_modify_range(self, start, length)
    }

    /// `length`.
    pub fn length(&self) -> u64 {
        BufferObject::length(self)
    }

    /// `contents`: the block's mapping plus the buffer's offset in a Shared block; `None` in a
    /// Private one, where Metal answers `NULL`.
    pub fn contents(&self) -> Option<NonNull<u8>> {
        BufferObject::contents(self)
    }

    /// No Metal counterpart: the block the buffer is in.
    pub fn block(&self) -> BlockId {
        self.inner.block.memory.id
    }

    /// No Metal counterpart: the buffer's offset in its block.
    pub fn offset(&self) -> u64 {
        self.inner.offset
    }

    /// The device it was made on.
    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    /// A handle of `inner`, a buffer of `device`.
    pub(crate) fn of(inner: Arc<BufferInner>, device: Device) -> Buffer {
        Buffer { inner, device }
    }
}

/// What the runtime reads of a buffer, through the consumer's handle ([`Buffer`]) or through the
/// buffer as a command buffer, a texture over it or the submit thread keeps it
/// ([`BufferInner`]), which holds no handle of the device.
pub(crate) trait BufferObject {
    /// The buffer itself.
    fn object(&self) -> &Arc<BufferInner>;

    /// [`Buffer::gpu_address`].
    fn gpu_address(&self) -> u64 {
        let inner = self.object();
        match &inner.copy {
            Some(copy) => copy.block.memory.address + copy.offset(),
            None => inner.block.memory.address + inner.offset,
        }
    }

    /// A public call put the bytes shaders reach (a Managed buffer's device copy, where it has
    /// one) where a shader may write them outside every bound slot: their block is marked where it
    /// is exported ([`Block::address_out`]).
    fn address_out(&self) {
        let inner = self.object();
        (inner.copy.as_ref())
            .map_or(&inner.block, |copy| &copy.block)
            .address_out();
    }

    /// [`Buffer::did_modify_range`].
    fn did_modify_range(&self, start: u64, length: u64) -> Result<(), Error> {
        let inner = self.object();
        let end = (start.checked_add(length))
            .filter(|&end| end <= inner.length)
            .ok_or(Error::Refused(Refusal::ModifiedRange {
                start,
                length,
                buffer_length: inner.length,
            }))?;
        if inner.copy.is_none() || length == 0 {
            return Ok(());
        }
        inner.core.work.modify(inner, start, end)
    }

    /// [`Buffer::length`].
    fn length(&self) -> u64 {
        self.object().length
    }

    /// [`Buffer::contents`].
    fn contents(&self) -> Option<NonNull<u8>> {
        self.object().contents.map(Mapping::get)
    }

    /// Whether shaders reach the buffer in Shared memory, which the host reads.
    fn is_shared(&self) -> bool {
        let inner = self.object();
        inner.copy.is_none() && inner.block.memory.class == MemoryClass::Shared
    }

    /// Its textures over its device copy in images of their own, where it is a Managed buffer
    /// with a device copy and has one (`crate::over::ManagedOvers`).
    fn managed_overs(&self) -> Option<&ManagedOvers> {
        let inner = self.object();
        (inner.copy.is_some() && inner.overs.any()).then_some(&*inner.overs)
    }

    /// Its textures over it in images of their own, as it keeps them.
    fn over_entries(&self) -> &ManagedOvers {
        &self.object().overs
    }

    /// The block, and a Managed buffer's device copy, for a command buffer to keep.
    fn held(&self) -> (&Arc<Block>, Option<&Arc<DeviceCopy>>) {
        let inner = self.object();
        (&inner.block, inner.copy.as_ref())
    }

    /// The `VkBuffer` shaders reach the buffer through and the buffer's offset in it.
    fn device_range(&self) -> (vk::Buffer, u64) {
        let inner = self.object();
        match &inner.copy {
            Some(copy) => copy.range(0),
            None => inner.block.memory.range(inner.offset),
        }
    }

    /// Its own resource id, whatever heap it is in.
    fn id(&self) -> u32 {
        self.object().id
    }

    /// The memory it lies in, as the copies over buffers track it (`crate::over`): its bytes of
    /// the block the consumer placed it in (of a Managed buffer, the block of its bytes).
    fn span(&self) -> Span {
        let inner = self.object();
        Span {
            block: inner.block.memory.id,
            start: inner.offset,
            end: inner.offset + inner.length,
        }
    }

    /// Its id in the compute encoders' hazard sets: its heap's where the consumer placed it in a
    /// heap ([`Device::new_heap_buffer`]), its surface's for an exported surface's buffer in place
    /// ([`crate::Surface::buffer`]), its own elsewhere.
    fn tracked_id(&self) -> u32 {
        let inner = self.object();
        (inner.heap.as_ref()).map_or(inner.id, |heap| heap.get())
    }

    /// Whether `core` is the device this buffer was made on.
    fn is_of(&self, core: &Arc<Core>) -> bool {
        Arc::ptr_eq(&self.object().core, core)
    }
}

impl BufferObject for Buffer {
    fn object(&self) -> &Arc<BufferInner> {
        &self.inner
    }
}

impl BufferObject for Arc<BufferInner> {
    fn object(&self) -> &Arc<BufferInner> {
        self
    }
}
