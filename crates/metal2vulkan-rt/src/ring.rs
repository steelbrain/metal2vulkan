//! The parameter ring: memory of the floor's ring type (device-local, host-visible, coherent) that
//! the host writes and the GPU reads by address. It holds parameter blocks longer than the push
//! range and the bytes of `set_bytes`, which a draw's vertex input also reads through the block's
//! buffer.
//!
//! The ring is made of 2 MiB blocks, each split into 32 chunks of 64 KiB. A command buffer takes
//! chunks as it records, writes each byte once, and gives them back when it retires, so the GPU
//! never reads a chunk the host is writing. Each block is kept for the device's life. The
//! block-maker thread makes them, off every caller's thread (`crate::memory::Internal`): the
//! first as its first work once the device's making starts it, and the next each time a take
//! leaves fewer chunks free than a block holds, so a block's chunks are free ahead of the takes
//! and the ring holds at most one block more than the most chunks in use at once need. It makes
//! one block at a time, without the ring's lock: a take or a give-back never waits for the
//! driver's allocation. A take that finds no chunk free while the block-maker thread makes the
//! next waits for it, or for a chunk given back, counted in
//! [`Metrics::maker_waits`](crate::Metrics::maker_waits). Where it made none (its make failed, or
//! it had stopped), a take that finds no chunk free makes the block on its own thread, without
//! the ring's lock, counted in
//! [`Metrics::ring_block_allocations`](crate::Metrics::ring_block_allocations), and takes that
//! meet it then wait for it. The runtime never reads the ring back.
//!
//! One chunk, taken at the first ask and kept for the device's life, is the nil buffer
//! ([`Ring::nil`]): 64 KiB of zeros, at a multiple of 16 like every chunk, so it meets any buffer
//! argument's alignment. A buffer argument nothing is bound to reads it, as Metal's nil buffer
//! reads zeros. A read past its 64 KiB, and a write through it, are not covered.
//!
//! Another, taken the same way, is the zero range ([`Ring::zero`]): zeros that no command and no
//! buffer argument reaches, which a checked load of a buffer argument outside its buffer reads
//! (the translator's `PushField::ZeroAddress`), so it reads zero.

use crate::counters::{Gauge, Metric, Site};
use crate::device::{lock, Core};
use crate::error::Error;
use crate::floor;
use crate::memory::{self, Internal, Mapping};
use ash::vk;
use std::ptr::NonNull;
use std::sync::{Condvar, Mutex, PoisonError};

/// A chunk's length: the most one write takes.
pub(crate) const CHUNK: u64 = 64 * 1024;
/// A ring block's length.
pub(crate) const BLOCK: u64 = 32 * CHUNK;
/// The chunks of a block: a take that leaves fewer free asks the block-maker thread for the
/// next block.
pub(crate) const SPARE: usize = (BLOCK / CHUNK) as usize;
/// The alignment of every write: a parameter block's, and Metal's for `setBytes` data.
const ALIGN: u64 = 16;

/// The device's ring.
pub(crate) struct Ring {
    state: Mutex<State>,
    /// A block made, or chunks given back, while a block was being made.
    changed: Condvar,
    /// For the device tests: where set, the making of a block, the block-maker thread's or a
    /// take's, meets the test at the barrier twice before it allocates, once parked and once
    /// released.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) park: Mutex<Option<std::sync::Arc<std::sync::Barrier>>>,
}

struct State {
    blocks: Vec<RingBlock>,
    free: Vec<Chunk>,
    /// The nil buffer's address, once asked ([`Ring::nil`]).
    nil: Option<u64>,
    /// The zero range's address, once asked ([`Ring::zero`]).
    zero: Option<u64>,
    /// Who makes the next block.
    next: Next,
}

/// Who makes the ring's next block.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Next {
    /// No one: a block's chunks are free, or the block-maker thread made none.
    Unasked,
    /// The block-maker thread, asked for it ([`Internal::RingBlock`]).
    Maker,
    /// A take, on its own thread, where the block-maker thread made none.
    Take,
}

struct RingBlock {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

/// A chunk: its device address, its host address, and its block's buffer with its offset there.
#[derive(Clone, Copy)]
pub(crate) struct Chunk {
    address: u64,
    host: Mapping,
    buffer: vk::Buffer,
    offset: u64,
}

/// Where a write into the ring lies: its device address, and its block's buffer with its offset
/// there.
#[derive(Clone, Copy)]
pub(crate) struct Placed {
    pub(crate) address: u64,
    pub(crate) buffer: vk::Buffer,
    pub(crate) offset: u64,
}

impl Ring {
    /// The ring of a new device, whose first block the block-maker thread makes as its first
    /// work (`crate::memory::run`).
    pub(crate) fn new() -> Ring {
        Ring {
            state: Mutex::new(State {
                blocks: Vec::new(),
                free: Vec::new(),
                nil: None,
                zero: None,
                next: Next::Maker,
            }),
            changed: Condvar::new(),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            park: Mutex::new(None),
        }
    }

    /// A free chunk. A take that leaves fewer than [`SPARE`] free, with no block being made, asks
    /// the block-maker thread for the next. Where none is free, the take waits while the
    /// block-maker thread or another take makes a block (a wait for the block-maker thread is
    /// counted), and makes one itself where neither does.
    fn take(&self, core: &Core) -> Result<Chunk, Error> {
        let mut state = lock(&self.state);
        // When this take began to wait for the block-maker thread's block.
        let mut since = None;
        loop {
            if let Some(chunk) = state.free.pop() {
                let ask = state.next == Next::Unasked && state.free.len() < SPARE;
                if ask {
                    state.next = Next::Maker;
                }
                drop(state);
                waited(core, since);
                if ask && !core.pool.ask(Internal::RingBlock) {
                    self.unasked();
                }
                return Ok(chunk);
            }
            match state.next {
                Next::Unasked => break,
                Next::Maker => {
                    since.get_or_insert_with(crate::host_time);
                }
                Next::Take => {}
            }
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        waited(core, since);
        state.next = Next::Take;
        drop(state);
        let made = make(core);
        let mut state = lock(&self.state);
        state.next = Next::Unasked;
        self.changed.notify_all();
        let (block, address, host) = made?;
        Ok(state.add(block, address, host))
    }

    /// On the block-maker thread: makes the block asked of it ([`Internal::RingBlock`]) and hands
    /// its chunks to the takes. Err where the make failed: a take that finds no chunk free then
    /// makes the next block.
    pub(crate) fn make_ahead(&self, core: &Core) -> Result<(), Error> {
        let made = make(core);
        let mut state = lock(&self.state);
        state.next = Next::Unasked;
        self.changed.notify_all();
        let (block, address, host) = made?;
        let first = state.add(block, address, host);
        state.free.push(first);
        Ok(())
    }

    /// The block-maker thread makes no block asked of it: it has stopped, or (in the device
    /// tests) its make fails at once. A take that finds no chunk free makes the block.
    pub(crate) fn unasked(&self) {
        let mut state = lock(&self.state);
        if state.next == Next::Maker {
            state.next = Next::Unasked;
            self.changed.notify_all();
        }
    }

    /// The nil buffer's address (the module docs): a chunk of zeros, taken and zeroed at the
    /// first ask and never given back, so no command buffer writes it.
    pub(crate) fn nil(&self, core: &Core) -> Result<u64, Error> {
        self.kept(core, |state| &mut state.nil)
    }

    /// The zero range's address (the module docs): a chunk of zeros taken as the nil buffer's
    /// is, which nothing is bound to, so nothing writes it. A multiple of 16.
    pub(crate) fn zero(&self, core: &Core) -> Result<u64, Error> {
        self.kept(core, |state| &mut state.zero)
    }

    /// The address of the chunk of zeros `which` holds, taken and zeroed at its first ask and
    /// never given back.
    fn kept(&self, core: &Core, which: fn(&mut State) -> &mut Option<u64>) -> Result<u64, Error> {
        if let Some(address) = *which(&mut lock(&self.state)) {
            return Ok(address);
        }
        let chunk = self.take(core)?;
        // SAFETY: the chunk is out of every command buffer's hands and no pending work reads it:
        // the host alone writes it, once.
        unsafe { std::ptr::write_bytes(chunk.host.at(0).as_ptr(), 0, CHUNK as usize) };
        let mut state = lock(&self.state);
        match *which(&mut state) {
            // Another thread's ask made it first: the chunk goes back.
            Some(address) => {
                state.free.push(chunk);
                if state.next != Next::Unasked {
                    self.changed.notify_all();
                }
                Ok(address)
            }
            None => {
                *which(&mut state) = Some(chunk.address);
                Ok(chunk.address)
            }
        }
    }

    /// Takes back a retired command buffer's chunks.
    fn give_back(&self, chunks: Vec<Chunk>) {
        let mut state = lock(&self.state);
        state.free.extend(chunks);
        if state.next != Next::Unasked {
            self.changed.notify_all();
        }
    }

    /// For the device tests: the ring's blocks and free chunks where its lock is free now, else
    /// None.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn chunks(&self) -> Option<(usize, usize)> {
        (self.state.try_lock().ok()).map(|state| (state.blocks.len(), state.free.len()))
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
        state.nil = None;
        state.zero = None;
        freed
    }
}

impl State {
    /// Keeps a block made at `address`, mapped at `host`: its chunks but the first are free, and
    /// the first is the caller's.
    fn add(&mut self, block: RingBlock, address: u64, host: Mapping) -> Chunk {
        let buffer = block.buffer;
        self.blocks.push(block);
        let chunk = |index: u64| Chunk {
            address: address + index * CHUNK,
            // SAFETY: the chunk lies in the block's mapping.
            host: Mapping::new(unsafe { host.at(index * CHUNK) }),
            buffer,
            offset: index * CHUNK,
        };
        self.free.extend((1..BLOCK / CHUNK).rev().map(chunk));
        chunk(0)
    }
}

/// Counts a take's wait for the block-maker thread's block, begun at `since` where it waited.
fn waited(core: &Core, since: Option<u64>) {
    if let Some(since) = since {
        (core.counters).maker_wait(crate::host_time().saturating_sub(since));
    }
}

/// Makes a ring block: a buffer with every buffer usage on an allocation of the ring type with a
/// device address, mapped. Returns it with its device and host addresses. Its allocation is
/// counted at [`Site::RingBlock`] on a consumer's thread.
fn make(core: &Core) -> Result<(RingBlock, u64, Mapping), Error> {
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    {
        let park = lock(&core.ring.park).clone();
        if let Some(park) = park {
            park.wait();
            park.wait();
        }
    }
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
    let allocated = unsafe {
        memory::allocate(
            device,
            &core.counters,
            core.max_allocations,
            &allocate,
            Some(Site::RingBlock),
        )
    };
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
    /// returns where it lies.
    pub(crate) fn write(&mut self, core: &Core, bytes: &[u8]) -> Result<Placed, Error> {
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
        let (chunk, offset) = chunk;
        // SAFETY: `offset..offset + len` lies in the chunk, which this command buffer alone writes
        // until it retires, and which no pending work reads: its last command buffer retired.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                chunk.host.at(offset).as_ptr(),
                bytes.len(),
            )
        };
        Ok(Placed {
            address: chunk.address + offset,
            buffer: chunk.buffer,
            offset: chunk.offset + offset,
        })
    }

    /// Gives the chunks back once the command buffer retired or never ran.
    pub(crate) fn give_back(self, core: &Core) {
        if !self.chunks.is_empty() {
            core.ring.give_back(self.chunks);
        }
    }
}
