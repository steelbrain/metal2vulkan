//! Memory blocks: what the consumer takes and releases, the pool that makes them ahead of need on
//! its own thread, and how a block is made, zeroed and freed.

use crate::counters::{Gauge, Metric};
use crate::device::{lock, Core};
use crate::error::{Error, Limit};
use crate::floor;
use ash::vk;
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

/// No Metal counterpart: a pooled block's length, 256 MiB.
///
/// [`Device::take_block`](crate::Device::take_block) gives a pooled block for any length up to
/// this, and a dedicated one past it.
pub const BLOCK_LEN: u64 = 256 << 20;

/// Blocks of each class the block-maker thread keeps ready.
const READY: usize = 1;

/// The most idle blocks of a class the pool keeps. A block released past them is freed.
const IDLE_MAX: usize = 2;

/// A dedicated block's length is the length asked for, rounded up to this.
const DEDICATED_GRANULE: u64 = 64 << 10;

/// `MTLStorageMode` of a block, as the buffers in it see it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryClass {
    /// `MTLStorageModePrivate`: device-local memory with no host mapping.
    Private,
    /// `MTLStorageModeShared`: host-visible, host-coherent and host-cached memory, mapped for the
    /// block's life. On a discrete GPU it is system memory the GPU reads across its bus, as on a
    /// Mac with a discrete GPU.
    Shared,
}

impl MemoryClass {
    const ALL: [MemoryClass; 2] = [MemoryClass::Private, MemoryClass::Shared];

    fn index(self) -> usize {
        match self {
            MemoryClass::Private => 0,
            MemoryClass::Shared => 1,
        }
    }
}

/// No Metal counterpart: a block's number. Numbers start at 1 and are never reused during the
/// device's life; a block released and taken again keeps its number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(u32);

impl BlockId {
    /// The number.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// Nearest Metal counterpart: the memory of a placement `MTLHeap`, which the consumer places
/// buffers in ([`Device::new_buffer`](crate::Device::new_buffer)).
///
/// Plain data, not a handle: the block lives until it is released and its last buffer is gone, or
/// until the device is torn down. One `VkBuffer` spans it, with every buffer usage and a device
/// address. Nothing in it is exportable.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct BlockInfo {
    /// The block's number.
    pub id: BlockId,
    /// Its storage mode.
    pub class: MemoryClass,
    /// [`BLOCK_LEN`] for a pooled block; for a dedicated one, the length asked for rounded up to
    /// 64 KiB.
    pub len: u64,
    /// `gpuAddress` of the block's first byte: its `VkBuffer`'s device address.
    pub gpu_address: u64,
    /// `contents` of the block's first byte: the host mapping of a Shared block, `None` for a
    /// Private one.
    pub contents: Option<NonNull<u8>>,
}

// SAFETY: plain data. `contents` is an address; reading or writing through it is the consumer's
// own unsafe code, under the contract of `BlockMappings`.
unsafe impl Send for BlockInfo {}
// SAFETY: as above.
unsafe impl Sync for BlockInfo {}

/// No Metal counterpart: the consumer's hook before the runtime frees memory it handed out.
///
/// The runtime calls [`unmap`](BlockMappings::unmap) for every block it frees that
/// [`Device::take_block`](crate::Device::take_block) ever returned: a surplus released block, a
/// released dedicated block, and at teardown every such block still alive. Blocks the consumer
/// never took are freed without it.
pub trait BlockMappings: Send + Sync {
    /// Returns only once every mapping of `block` the consumer made is gone and none of its
    /// threads touches the block's memory. The memory stays mapped and valid until it returns.
    ///
    /// It runs on the block-maker thread, or at teardown on the thread that drops the device's
    /// last handle, after the runtime's threads are joined. It makes no call into the runtime and
    /// drops no runtime handle.
    fn unmap(&self, block: BlockId);
}

/// A block's memory: its allocation, the `VkBuffer` that spans it, and its mapping.
#[derive(Clone, Copy)]
pub(crate) struct Memory {
    pub(crate) id: BlockId,
    pub(crate) class: MemoryClass,
    pub(crate) len: u64,
    /// Made by the pool, to return to it when released; a dedicated block is freed instead.
    pooled: bool,
    pub(crate) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    pub(crate) address: u64,
    pub(crate) mapping: Option<Mapping>,
}

impl Memory {
    pub(crate) fn info(&self) -> BlockInfo {
        BlockInfo {
            id: self.id,
            class: self.class,
            len: self.len,
            gpu_address: self.address,
            contents: self.mapping.map(Mapping::get),
        }
    }
}

/// A host address in a Shared block's mapping, which lives as long as the block: its first byte,
/// or a buffer's.
#[derive(Clone, Copy)]
pub(crate) struct Mapping(NonNull<u8>);

// SAFETY: an address. The runtime writes through it only while it owns the bytes: the block-maker
// thread zeroing a block no one holds, and `new_buffer` zeroing the range of the buffer it makes.
unsafe impl Send for Mapping {}
// SAFETY: as above.
unsafe impl Sync for Mapping {}

impl Mapping {
    pub(crate) fn new(address: NonNull<u8>) -> Mapping {
        Mapping(address)
    }

    pub(crate) fn get(self) -> NonNull<u8> {
        self.0
    }

    /// The address `offset` bytes on.
    ///
    /// # Safety
    ///
    /// `self` is a block's first byte and `offset` is at most the block's length.
    pub(crate) unsafe fn at(self, offset: u64) -> NonNull<u8> {
        // SAFETY: the mapping spans the block, and `offset` is inside it or one past its end; a
        // block that was mapped has a length that fits the address space.
        unsafe { self.0.add(offset as usize) }
    }
}

/// A block that has been handed out. The consumer's hold on it and each buffer of it is one
/// reference; the last one dropped gives it back to the pool.
pub(crate) struct Block {
    pub(crate) memory: Memory,
    core: Arc<Core>,
}

impl Block {
    pub(crate) fn new(memory: Memory, core: Arc<Core>) -> Block {
        Block { memory, core }
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        self.core.pool.give_back(self.memory);
    }
}

/// The pool: blocks made ahead of need and blocks given back, which the block-maker thread
/// ([`run`]) zeroes, keeps or frees.
pub(crate) struct Pool {
    state: Mutex<State>,
    /// Signalled whenever the state changes, for the block-maker thread and for takers waiting
    /// on it.
    changed: Condvar,
    mappings: Option<Arc<dyn BlockMappings>>,
}

struct State {
    /// Per class, zeroed blocks no one holds, the latest last.
    idle: [Vec<Idle>; 2],
    /// Blocks given back, for the block-maker thread.
    returned: Vec<Memory>,
    /// Per class, why the block-maker thread could not make the last block it tried, for the
    /// next taker. It tries again once a taker has the error.
    failed: [Option<Error>; 2],
    stop: bool,
    next_id: u32,
}

struct Idle {
    memory: Memory,
    /// Whether the consumer has held it, so that freeing it calls the hook.
    handed: bool,
}

impl Pool {
    pub(crate) fn new(mappings: Option<Arc<dyn BlockMappings>>) -> Pool {
        Pool {
            state: Mutex::new(State {
                idle: [Vec::new(), Vec::new()],
                returned: Vec::new(),
                failed: [None, None],
                stop: false,
                next_id: 1,
            }),
            changed: Condvar::new(),
            mappings,
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    fn wait<'a>(&self, state: MutexGuard<'a, State>) -> MutexGuard<'a, State> {
        self.changed
            .wait(state)
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// A new block number.
    pub(crate) fn number(&self) -> Result<BlockId, Error> {
        number(&mut self.lock())
    }

    /// Takes a ready block of `class`, waiting for the block-maker thread where none is ready.
    /// Err: why the block-maker thread could not make one.
    pub(crate) fn take(&self, class: MemoryClass) -> Result<Memory, Error> {
        let index = class.index();
        let mut state = self.lock();
        loop {
            if let Some(idle) = state.idle[index].pop() {
                self.changed.notify_all();
                return Ok(idle.memory);
            }
            if let Some(error) = state.failed[index].take() {
                self.changed.notify_all();
                return Err(error);
            }
            state = self.wait(state);
        }
    }

    fn give_back(&self, memory: Memory) {
        self.lock().returned.push(memory);
        self.changed.notify_all();
    }

    /// Stops the block-maker thread once it finishes what it is doing.
    pub(crate) fn stop(&self) {
        self.lock().stop = true;
        self.changed.notify_all();
    }

    /// At teardown, after the block-maker thread is joined: every block the pool still has, and
    /// whether the consumer held it.
    pub(crate) fn remaining(&mut self) -> Vec<(Memory, bool)> {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        let mut remaining: Vec<(Memory, bool)> = state
            .idle
            .iter_mut()
            .flat_map(|idle| idle.drain(..))
            .map(|idle| (idle.memory, idle.handed))
            .collect();
        remaining.extend(state.returned.drain(..).map(|memory| (memory, true)));
        remaining
    }
}

fn number(state: &mut State) -> Result<BlockId, Error> {
    let id = state.next_id;
    state.next_id = id.checked_add(1).ok_or(Error::Limit(Limit::BlockIds))?;
    Ok(BlockId(id))
}

/// The block-maker thread, until the device's teardown stops it. Off every caller's path, it
/// zeroes each pooled block given back and keeps it while its class has fewer than [`IDLE_MAX`]
/// idle, and frees the rest through the hook; then it makes blocks until each class has
/// [`READY`] idle.
pub(crate) fn run(core: Arc<Core>) {
    let pool = &core.pool;
    let mut state = pool.lock();
    loop {
        if state.stop {
            return;
        }
        if let Some(memory) = state.returned.pop() {
            let room = memory.pooled && state.idle[memory.class.index()].len() < IDLE_MAX;
            drop(state);
            // Only this thread adds idle blocks, so the room is still there once it is zeroed. A
            // block that cannot be zeroed is freed.
            let kept = room && zero(&core, &memory, 0, memory.len).is_ok();
            if !kept {
                free(&core, memory, true);
            }
            state = pool.lock();
            if kept {
                state.idle[memory.class.index()].push(Idle {
                    memory,
                    handed: true,
                });
                pool.changed.notify_all();
            }
            continue;
        }
        let wanted = MemoryClass::ALL.into_iter().find(|class| {
            state.idle[class.index()].len() < READY && state.failed[class.index()].is_none()
        });
        if let Some(class) = wanted {
            let id = number(&mut state);
            drop(state);
            let made = id.and_then(|id| make(&core, class, BLOCK_LEN, id, true));
            state = pool.lock();
            match made {
                Ok(memory) => state.idle[class.index()].push(Idle {
                    memory,
                    handed: false,
                }),
                Err(error) => state.failed[class.index()] = Some(error),
            }
            pool.changed.notify_all();
            continue;
        }
        state = pool.wait(state);
    }
}

/// The length of a dedicated block for `len` bytes: `len` rounded up to 64 KiB, where the device
/// holds that in one allocation and one buffer (`max`).
pub(crate) fn dedicated_len(len: u64, max: u64) -> Result<u64, Error> {
    match len.checked_next_multiple_of(DEDICATED_GRANULE) {
        Some(rounded) if rounded <= max => Ok(rounded),
        rounded => Err(Error::Limit(Limit::BlockLen {
            len: rounded.unwrap_or(len),
            max,
        })),
    }
}

/// Makes a zeroed block of `len` bytes of `class`: a `VkBuffer` with every buffer usage, bound to
/// an allocation of its own with a device address, and for Shared mapped and zeroed through the
/// mapping, which also makes every page resident before anyone holds the block.
pub(crate) fn make(
    core: &Core,
    class: MemoryClass,
    len: u64,
    id: BlockId,
    pooled: bool,
) -> Result<Memory, Error> {
    let device = &core.device;
    let info = vk::BufferCreateInfo::default()
        .size(len)
        .usage(floor::BUFFER_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: the device is live; `len` is at most the device's `maxBufferSize`.
    let buffer =
        unsafe { device.create_buffer(&info, None) }.map_err(Error::vulkan("vkCreateBuffer"))?;
    // SAFETY: `buffer` is the device's.
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let allocate = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(core.memory_type(class))
        .push_next(&mut flags);
    // SAFETY: the floor chose the memory type for buffers of this usage on this device, and
    // `bufferDeviceAddress` is enabled.
    let memory = match unsafe { device.allocate_memory(&allocate, None) } {
        Ok(memory) => memory,
        Err(result) => {
            // SAFETY: `buffer` is unused.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(Error::Vulkan {
                call: "vkAllocateMemory",
                result,
            });
        }
    };
    core.counters.rise(Gauge::Blocks);
    core.counters.count(Metric::BlocksMade);
    let mut block = Memory {
        id,
        class,
        len,
        pooled,
        buffer,
        memory,
        address: 0,
        mapping: None,
    };
    match finish(core, &mut block) {
        Ok(()) => Ok(block),
        Err(error) => {
            free(core, block, false);
            Err(error)
        }
    }
}

/// Binds a new block's memory, reads its address, maps it if Shared, and zeroes it.
fn finish(core: &Core, block: &mut Memory) -> Result<(), Error> {
    let device = &core.device;
    // SAFETY: the allocation is the buffer's size and of a type its requirements allow.
    unsafe { device.bind_buffer_memory(block.buffer, block.memory, 0) }
        .map_err(Error::vulkan("vkBindBufferMemory"))?;
    let address = vk::BufferDeviceAddressInfo::default().buffer(block.buffer);
    // SAFETY: the buffer has device-address usage and is bound.
    block.address = unsafe { device.get_buffer_device_address(&address) };
    if block.class == MemoryClass::Shared {
        // SAFETY: the memory type is host-visible, and the memory is not mapped.
        let pointer = unsafe {
            device.map_memory(block.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }
        .map_err(Error::vulkan("vkMapMemory"))?;
        block.mapping = NonNull::new(pointer.cast::<u8>()).map(Mapping);
    }
    zero(core, block, 0, block.len)
}

/// Zeroes `offset..end` of a block: through the mapping for Shared, by a GPU fill for Private.
/// The bytes belong to no one else: a block no one holds, or the range of a buffer being made.
pub(crate) fn zero(core: &Core, memory: &Memory, offset: u64, end: u64) -> Result<(), Error> {
    match memory.mapping {
        Some(mapping) => {
            // SAFETY: `offset..end` lies in the block, which the mapping spans.
            unsafe {
                mapping
                    .at(offset)
                    .as_ptr()
                    .write_bytes(0, (end - offset) as usize)
            };
            Ok(())
        }
        None => core.filler.zero(core, memory.buffer, offset, end),
    }
}

/// Frees a block, calling the consumer's hook first where it held the block. Freeing the memory
/// also unmaps it.
pub(crate) fn free(core: &Core, memory: Memory, handed: bool) {
    if handed {
        if let Some(mappings) = &core.pool.mappings {
            mappings.unmap(memory.id);
        }
    }
    let device = &core.device;
    // SAFETY: no GPU work that uses the block is pending: the only GPU work on a block is its
    // zero fills, each waited for. The consumer's mappings are gone.
    unsafe {
        device.destroy_buffer(memory.buffer, None);
        device.free_memory(memory.memory, None);
    }
    core.counters.fall(Gauge::Blocks);
}
