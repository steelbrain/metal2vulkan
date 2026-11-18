//! The parameter ring: memory of the floor's ring type (device-local, host-visible, coherent) that
//! the host writes and the GPU reads by address. It holds parameter blocks longer than the push
//! range and the bytes of `set_bytes`.
//!
//! The ring is made of 2 MiB blocks, each split into 64 KiB chunks. A command buffer takes chunks
//! as it records, writes each byte once, and gives them back when it retires, so the GPU never
//! reads a chunk the host is writing. A block is made when no chunk is free and kept for the
//! device's life: the ring holds what the command buffers in flight at once have needed. The
//! runtime never reads the ring back.

use crate::counters::{Gauge, Metric};
use crate::device::{lock, Core};
use crate::error::Error;
use crate::floor;
use crate::memory::{self, Mapping};
use ash::vk;
use std::ptr::NonNull;
use std::sync::Mutex;

/// A chunk's length: the most one write takes.
pub(crate) const CHUNK: u64 = 64 * 1024;
/// A ring block's length.
const BLOCK: u64 = 32 * CHUNK;
/// The alignment of every write: a parameter block's, and Metal's for `setBytes` data.
const ALIGN: u64 = 16;

/// The device's ring.
pub(crate) struct Ring {
    state: Mutex<State>,
}

struct State {
    blocks: Vec<RingBlock>,
    free: Vec<Chunk>,
}

struct RingBlock {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

/// A chunk: its device address and its host address.
#[derive(Clone, Copy)]
pub(crate) struct Chunk {
    address: u64,
    host: Mapping,
}

impl Ring {
    pub(crate) fn new() -> Ring {
        Ring {
            state: Mutex::new(State {
                blocks: Vec::new(),
                free: Vec::new(),
            }),
        }
    }

    /// A free chunk, making a block where none is free.
    fn take(&self, core: &Core) -> Result<Chunk, Error> {
        let mut state = lock(&self.state);
        if let Some(chunk) = state.free.pop() {
            return Ok(chunk);
        }
        let (block, address, host) = make(core)?;
        state.blocks.push(block);
        for index in (1..BLOCK / CHUNK).rev() {
            state.free.push(Chunk {
                address: address + index * CHUNK,
                // SAFETY: the chunk lies in the block's mapping.
                host: Mapping::new(unsafe { host.at(index * CHUNK) }),
            });
        }
        Ok(Chunk { address, host })
    }

    /// Takes back a retired command buffer's chunks.
    fn give_back(&self, chunks: Vec<Chunk>) {
        lock(&self.state).free.extend(chunks);
    }

    /// Frees every block, returning how many it freed.
    ///
    /// # Safety
    ///
    /// No pending work reads the ring.
    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) -> usize {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        let freed = state.blocks.len();
        for block in state.blocks.drain(..) {
            // SAFETY: by the caller; freeing the memory unmaps it.
            unsafe {
                device.destroy_buffer(block.buffer, None);
                device.free_memory(block.memory, None);
            }
        }
        state.free.clear();
        freed
    }
}

/// Makes a ring block: a buffer with every buffer usage on an allocation of the ring type with a
/// device address, mapped. Returns it with its device and host addresses.
fn make(core: &Core) -> Result<(RingBlock, u64, Mapping), Error> {
    let device = &core.device;
    let info = vk::BufferCreateInfo::default()
        .size(BLOCK)
        .usage(floor::BUFFER_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: the device is live.
    let buffer =
        unsafe { device.create_buffer(&info, None) }.map_err(Error::vulkan("vkCreateBuffer"))?;
    // SAFETY: `buffer` is the device's.
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(core.memory_types.ring)
        .push_next(&mut flags);
    // SAFETY: the floor chose the ring type for buffers of this usage, and `bufferDeviceAddress`
    // is enabled.
    let allocated =
        unsafe { memory::allocate(device, &core.counters, core.max_allocations, &allocate) };
    let memory = match allocated {
        Ok(memory) => memory,
        Err(error) => {
            // SAFETY: `buffer` is unused.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(error);
        }
    };
    core.counters.count(Metric::BlocksMade);
    let block = RingBlock { buffer, memory };
    match finish(device, &block) {
        Ok((address, host)) => Ok((block, address, host)),
        Err(error) => {
            // SAFETY: nothing uses the block.
            unsafe {
                device.destroy_buffer(block.buffer, None);
                device.free_memory(block.memory, None);
            }
            core.counters.fall(Gauge::Blocks);
            Err(error)
        }
    }
}

/// Binds a ring block's memory, reads its address and maps it.
fn finish(device: &ash::Device, block: &RingBlock) -> Result<(u64, Mapping), Error> {
    // SAFETY: the allocation is the buffer's size and of a type its requirements allow.
    unsafe { device.bind_buffer_memory(block.buffer, block.memory, 0) }
        .map_err(Error::vulkan("vkBindBufferMemory"))?;
    let info = vk::BufferDeviceAddressInfo::default().buffer(block.buffer);
    // SAFETY: the buffer has device-address usage and is bound.
    let address = unsafe { device.get_buffer_device_address(&info) };
    // SAFETY: the ring type is host-visible, and the memory is not mapped.
    let pointer =
        unsafe { device.map_memory(block.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
            .map_err(Error::vulkan("vkMapMemory"))?;
    let host = NonNull::new(pointer.cast::<u8>()).ok_or(Error::Vulkan {
        call: "vkMapMemory",
        result: vk::Result::ERROR_MEMORY_MAP_FAILED,
    })?;
    Ok((address, Mapping::new(host)))
}

/// A command buffer's part of the ring: the chunks it took, the last one being written.
#[derive(Default)]
pub(crate) struct Writes {
    chunks: Vec<Chunk>,
    /// Bytes used of the last chunk.
    used: u64,
}

impl Writes {
    /// Copies `bytes`, at most a chunk, into the ring at an address aligned to 16 bytes, and
    /// returns that address.
    pub(crate) fn write(&mut self, core: &Core, bytes: &[u8]) -> Result<u64, Error> {
        let len = bytes.len() as u64;
        debug_assert!(len <= CHUNK);
        let start = self.used.next_multiple_of(ALIGN);
        let chunk = match self.chunks.last() {
            Some(chunk) if start + len <= CHUNK => {
                self.used = start + len;
                (*chunk, start)
            }
            _ => {
                let chunk = core.ring.take(core)?;
                self.chunks.push(chunk);
                self.used = len;
                (chunk, 0)
            }
        };
        let (Chunk { address, host }, offset) = chunk;
        // SAFETY: `offset..offset + len` lies in the chunk, which this command buffer alone writes
        // until it retires, and which no pending work reads: its last command buffer retired.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), host.at(offset).as_ptr(), bytes.len())
        };
        Ok(address + offset)
    }

    /// Gives the chunks back once the command buffer retired or never ran.
    pub(crate) fn give_back(self, core: &Core) {
        if !self.chunks.is_empty() {
            core.ring.give_back(self.chunks);
        }
    }
}
