//! Memory blocks: what the consumer takes and releases, the pool that makes them ahead of need on
//! its own thread, and how a block is made, zeroed and freed.

use crate::counters::{Counters, Gauge, Metric};
use crate::device::{lock, Core};
use crate::error::{Error, Limit};
use crate::floor;
use ash::vk;
use std::collections::HashSet;
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

/// Blocks of each class the device's making makes ([`stock`]) and the block-maker thread keeps
/// ready after it.
const READY: usize = 1;

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

    /// No Metal counterpart: the length of the class's pooled block, on every device: 16 MiB of
    /// Shared memory, 64 MiB of Private.
    ///
    /// [`Device::take_block`](crate::Device::take_block) gives a pooled block for any length up
    /// to it, and a dedicated one past it.
    ///
    /// A driver may hold the device's other calls (pipeline creation, submissions, other
    /// allocations) for the whole of a block's `vkAllocateMemory`, and the hold grows with the
    /// bytes, so the length sets how long one hold lasts. On an RTX 3090 (NVIDIA's Linux
    /// driver) a pipeline state's creation beside a Shared block's making took 40 to 45 ms at
    /// 256 MiB and 6 to 8 ms at 16 MiB, against 3 to 4 ms settled. Shared blocks are 16 MiB, the
    /// largest piece a guest's allocator takes for its buffers. Private memory holds less: 3.4 to
    /// 4.6 ms beside a 64 MiB block, at most 1 ms more than beside a 32 MiB one, and at 64 MiB a
    /// texture or a Managed buffer's device copy of 32 MiB takes a pooled block.
    pub const fn block_len(self) -> u64 {
        match self {
            MemoryClass::Private => 64 << 20,
            MemoryClass::Shared => 16 << 20,
        }
    }

    /// The most bytes of the class's kept idle blocks, pooled and dedicated together: 256 MiB of
    /// Shared memory, 128 MiB of Private. The class's ready block, its latest idle pooled one, is
    /// a reserve outside the bound. A block given back is zeroed and kept as the class's latest
    /// idle block; while the kept blocks pass the bound, the oldest is freed. A dedicated block so
    /// kept serves a later request it fits and that fills at least half of it
    /// ([`Pool::take_held`]), with no allocation: a program that makes and releases large buffers
    /// frame after frame pays the driver's allocation, and the hold on the device's other calls
    /// that comes with it, once.
    pub(crate) fn idle_bound(self) -> u64 {
        match self {
            MemoryClass::Private => 128 << 20,
            MemoryClass::Shared => 256 << 20,
        }
    }

    /// The gauge of the class's idle bytes.
    fn idle_gauge(self) -> Gauge {
        match self {
            MemoryClass::Private => Gauge::PrivateIdleBytes,
            MemoryClass::Shared => Gauge::SharedIdleBytes,
        }
    }

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
    /// [`MemoryClass::block_len`] for a pooled block; for a dedicated one, the length asked for
    /// rounded up to 64 KiB.
    pub len: u64,
    /// `gpuAddress` of the block's first byte: its `VkBuffer`'s device address.
    pub gpu_address: u64,
    /// `contents` of the block's first byte: the host mapping of a Shared block, `None` for a
    /// Private one.
    pub contents: Option<NonNull<u8>>,
    /// Whether its memory type is device-local. A Private block is, unless the device-local type
    /// was out of memory when it was made and it went to the next type its buffer takes (counted
    /// in [`Gauges::spilled_bytes`](crate::Gauges::spilled_bytes)); a Shared block is where the
    /// device's Shared type is.
    pub device_local: bool,
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
    /// It runs once per block: on the block-maker thread; in
    /// [`Device::quiesce`](crate::Device::quiesce), on the thread that calls it, for every block
    /// still handed out; or at teardown, on the thread that ends it, once the runtime's other
    /// threads are joined. It makes no call into the runtime and drops no runtime handle.
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
    /// Its memory type is device-local.
    device_local: bool,
    /// A Private block in another memory type than the Private one, which was out of memory.
    spilled: bool,
}

impl Memory {
    /// The allocation.
    pub(crate) fn device_memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    /// Whether the pool made it, to keep once it is given back; a dedicated block is freed then.
    pub(crate) fn pooled(&self) -> bool {
        self.pooled
    }

    /// Whether it is a Private block in another memory type than the Private one.
    pub(crate) fn spilled(&self) -> bool {
        self.spilled
    }

    pub(crate) fn info(&self) -> BlockInfo {
        BlockInfo {
            id: self.id,
            class: self.class,
            len: self.len,
            gpu_address: self.address,
            contents: self.mapping.map(Mapping::get),
            device_local: self.device_local,
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

/// A block that has been handed out. The consumer's hold on it, each buffer of it, and each
/// command buffer and upload that uses it until it retires are one reference each; the last one
/// dropped gives it back to the pool.
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
    /// The device's counters: the idle bytes and the dedicated blocks taken again.
    counters: Counters,
}

/// Whether a dedicated block of `held` bytes serves a dedicated request of `want`: it fits, and
/// the request fills at least half of it.
pub(crate) fn fits(held: u64, want: u64) -> bool {
    held >= want && want.saturating_mul(2) >= held
}

struct State {
    /// Per class, zeroed blocks no one holds, pooled and dedicated, the oldest first: the ready
    /// block (the latest pooled one) and at most the class's [`MemoryClass::idle_bound`] bytes
    /// beside it.
    idle: [Vec<Memory>; 2],
    /// Blocks given back, for the block-maker thread.
    returned: Vec<Memory>,
    /// Per class, why the block-maker thread could not make the last block it tried, for the
    /// next taker. It tries again once a taker has the error.
    failed: [Option<Error>; 2],
    stop: bool,
    next_id: u32,
    /// The blocks `take_block` returned that are not yet freed: freeing one calls the hook.
    handed: HashSet<BlockId>,
    /// The hook has run for every block handed out ([`Pool::unmap_all`]); it never runs again.
    unmapped: bool,
}

impl Pool {
    pub(crate) fn new(mappings: Option<Arc<dyn BlockMappings>>, counters: Counters) -> Pool {
        Pool {
            state: Mutex::new(State {
                idle: [Vec::new(), Vec::new()],
                returned: Vec::new(),
                failed: [None, None],
                stop: false,
                next_id: 1,
                handed: HashSet::new(),
                unmapped: false,
            }),
            changed: Condvar::new(),
            mappings,
            counters,
        }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        lock(&self.state)
    }

    /// The zeroed blocks of `class` no one holds: the tests wait for the block-maker thread to
    /// have made its ready ones, and with them its fills' submissions.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn idle(&self, class: MemoryClass) -> usize {
        self.lock().idle[class.index()].len()
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
    /// Err: why the block-maker thread could not make one; [`Error::Quiesced`] once it is stopped.
    pub(crate) fn take(&self, class: MemoryClass) -> Result<Memory, Error> {
        let index = class.index();
        let mut state = self.lock();
        loop {
            if state.stop {
                return Err(Error::Quiesced);
            }
            if let Some(at) = state.idle[index].iter().rposition(Memory::pooled) {
                let memory = state.idle[index].remove(at);
                self.counters.fall_by(class.idle_gauge(), memory.len);
                self.changed.notify_all();
                return Ok(memory);
            }
            if let Some(error) = state.failed[index].take() {
                self.changed.notify_all();
                return Err(error);
            }
            state = self.wait(state);
        }
    }

    /// An idle dedicated block of `class` for a dedicated request of `len` bytes: the shortest
    /// that [`fits`] it, which reads zero and keeps its number, address and mapping. None where
    /// no idle block does, for the caller to make one.
    pub(crate) fn take_held(&self, class: MemoryClass, len: u64) -> Option<Memory> {
        let mut state = self.lock();
        let idle = &mut state.idle[class.index()];
        let at = (0..idle.len())
            .filter(|&at| !idle[at].pooled && fits(idle[at].len, len))
            .min_by_key(|&at| idle[at].len)?;
        let memory = idle.remove(at);
        self.counters.fall_by(class.idle_gauge(), memory.len);
        self.counters.count(Metric::DedicatedBlocksReused);
        Some(memory)
    }

    /// Keeps `memory`, zeroed, as its class's latest idle block, and returns the oldest kept
    /// blocks of the class that then pass its bound, the ready block aside, for the caller to free
    /// without the lock. `memory` is at most the bound, so it stays.
    fn keep(&self, state: &mut State, memory: Memory) -> Vec<Memory> {
        let class = memory.class;
        let idle = &mut state.idle[class.index()];
        idle.push(memory);
        self.counters.rise_by(class.idle_gauge(), memory.len);
        let mut gone = Vec::new();
        loop {
            let ready = idle.iter().rposition(Memory::pooled);
            let mut kept = (0..idle.len()).filter(|&at| Some(at) != ready);
            let total: u64 = kept.clone().map(|at| idle[at].len).sum();
            let Some(oldest) = kept.next().filter(|_| total > class.idle_bound()) else {
                return gone;
            };
            let oldest = idle.remove(oldest);
            self.counters.fall_by(class.idle_gauge(), oldest.len);
            gone.push(oldest);
        }
    }

    /// Records a block as handed to the consumer, so that freeing it calls the hook.
    pub(crate) fn hand(&self, block: BlockId) {
        let mut state = self.lock();
        if !state.unmapped {
            state.handed.insert(block);
        }
    }

    /// In [`Device::quiesce`](crate::Device::quiesce) and at teardown, once the runtime's other
    /// threads are stopped: calls the hook for every block handed out and not yet freed, on this
    /// thread. No block's hook runs again: a block freed later is freed without it.
    pub(crate) fn unmap_all(&self) {
        let mut handed: Vec<BlockId> = {
            let mut state = self.lock();
            state.unmapped = true;
            state.handed.drain().collect()
        };
        handed.sort_unstable();
        if let Some(mappings) = &self.mappings {
            for block in handed {
                mappings.unmap(block);
            }
        }
    }

    /// Takes back a block taken with [`take`](Self::take), for the block-maker thread to zero and
    /// keep, or free.
    pub(crate) fn give_back(&self, memory: Memory) {
        self.lock().returned.push(memory);
        self.changed.notify_all();
    }

    /// Stops the block-maker thread once it finishes what it is doing.
    pub(crate) fn stop(&self) {
        self.lock().stop = true;
        self.changed.notify_all();
    }

    /// At teardown, after the block-maker thread is joined: every block the pool still has.
    pub(crate) fn remaining(&mut self) -> Vec<Memory> {
        let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
        let mut remaining = Vec::new();
        for class in MemoryClass::ALL {
            for memory in state.idle[class.index()].drain(..) {
                self.counters.fall_by(class.idle_gauge(), memory.len);
                remaining.push(memory);
            }
        }
        remaining.append(&mut state.returned);
        remaining
    }

    /// For the device tests: whether the pool has the block back, idle or still to be zeroed.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn holds(&self, block: BlockId) -> bool {
        let state = self.lock();
        state
            .idle
            .iter()
            .flatten()
            .chain(&state.returned)
            .any(|memory| memory.id == block)
    }
}

fn number(state: &mut State) -> Result<BlockId, Error> {
    let id = state.next_id;
    state.next_id = id.checked_add(1).ok_or(Error::Limit(Limit::BlockIds))?;
    Ok(BlockId(id))
}

/// In the device's making, before the block-maker thread starts: makes [`READY`] blocks of each
/// class, so the consumer's first block of each class is ready as the device is, and no block is
/// made beside the consumer's first calls. A driver may hold the device's other calls while it
/// allocates a block's memory: on an RTX 3090 (NVIDIA's Linux driver), a pipeline state's creation
/// right after the device's making waited 47 to 61 ms behind the block-maker thread's first 256 MiB
/// Shared block, against 2.4 to 3.1 ms once the block was made here. A class whose block cannot be made
/// keeps why for its next taker, as the block-maker thread does.
pub(crate) fn stock(core: &Core) {
    for class in MemoryClass::ALL {
        for _ in 0..READY {
            let made =
                (core.pool.number()).and_then(|id| make(core, class, class.block_len(), id, true));
            let mut state = core.pool.lock();
            let gone = match made {
                Ok(memory) => core.pool.keep(&mut state, memory),
                Err(error) => {
                    state.failed[class.index()] = Some(error);
                    break;
                }
            };
            drop(state);
            for memory in gone {
                free(core, memory);
            }
        }
    }
}

/// The block-maker thread, until the device's teardown stops it. Off every caller's path, it
/// zeroes each block given back, pooled or dedicated, and keeps it as its class's latest idle
/// block, freeing through the hook the oldest kept blocks past the class's
/// [`MemoryClass::idle_bound`], the ready block aside (and at once a block longer than the bound,
/// or spilled); then it makes pooled blocks until each class has [`READY`] idle.
pub(crate) fn run(core: Arc<Core>) {
    let pool = &core.pool;
    let mut state = pool.lock();
    loop {
        if state.stop {
            return;
        }
        if let Some(memory) = state.returned.pop() {
            // A spilled block is freed, so the pool's next block tries device-local memory again.
            let room = !memory.spilled && memory.len <= memory.class.idle_bound();
            drop(state);
            // A block that cannot be zeroed is freed.
            let zeroed = room && zero(&core, &memory, 0, memory.len).is_ok();
            let gone = match zeroed {
                true => {
                    let gone = pool.keep(&mut pool.lock(), memory);
                    pool.changed.notify_all();
                    gone
                }
                false => vec![memory],
            };
            for memory in gone {
                free(&core, memory);
            }
            state = pool.lock();
            continue;
        }
        let wanted = MemoryClass::ALL.into_iter().find(|class| {
            state.idle[class.index()].len() < READY && state.failed[class.index()].is_none()
        });
        if let Some(class) = wanted {
            let id = number(&mut state);
            drop(state);
            let made = id.and_then(|id| make(&core, class, class.block_len(), id, true));
            state = pool.lock();
            let gone = match made {
                Ok(memory) => pool.keep(&mut state, memory),
                Err(error) => {
                    state.failed[class.index()] = Some(error);
                    Vec::new()
                }
            };
            pool.changed.notify_all();
            if !gone.is_empty() {
                drop(state);
                for memory in gone {
                    free(&core, memory);
                }
                state = pool.lock();
            }
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

/// [`allocate`] of `info`, and for Private memory, where the device-local type is out of device
/// memory, of each other type `bits` allows in the device's order until one is not: the spill
/// the plan decided, on the allocation's own error and never a budget poll. Returns the memory
/// and whether it spilled.
///
/// # Safety
///
/// As [`allocate`], for every type `bits` allows.
pub(crate) unsafe fn allocate_spilling(
    core: &Core,
    class: MemoryClass,
    bits: u32,
    info: &vk::MemoryAllocateInfo<'_>,
) -> Result<(vk::DeviceMemory, bool), Error> {
    let out = |result: &Result<vk::DeviceMemory, Error>| {
        result.as_ref().is_err_and(Error::is_out_of_device_memory)
    };
    let first = info.memory_type_index;
    let mut result = match core.fails_private(class) {
        true => Err(Error::out_of_device_memory()),
        // SAFETY: the caller's.
        false => unsafe { allocate(&core.device, &core.counters, core.max_allocations, info) },
    };
    if class == MemoryClass::Shared {
        return result.map(|memory| (memory, false));
    }
    for index in (0..u32::BITS).filter(|&index| index != first && bits & (1 << index) != 0) {
        if !out(&result) {
            break;
        }
        let mut next = *info;
        next.memory_type_index = index;
        // SAFETY: the caller's; `bits` allows the type.
        result = unsafe { allocate(&core.device, &core.counters, core.max_allocations, &next) };
        if let Ok(memory) = result {
            return Ok((memory, true));
        }
    }
    result.map(|memory| (memory, false))
}

/// `vkAllocateMemory`, every allocation the runtime makes: one more live allocation in
/// `counters`' blocks gauge, refused with [`Limit::MemoryAllocations`] before the driver is called
/// where `max`, the device's `maxMemoryAllocationCount`, are live. Each free falls the gauge.
///
/// # Safety
///
/// `info` is valid for `device`.
pub(crate) unsafe fn allocate(
    device: &ash::Device,
    counters: &Counters,
    max: u32,
    info: &vk::MemoryAllocateInfo<'_>,
) -> Result<vk::DeviceMemory, Error> {
    if !counters.rise_below(Gauge::Blocks, u64::from(max)) {
        return Err(Error::Limit(Limit::MemoryAllocations { max }));
    }
    // SAFETY: the caller's.
    unsafe { device.allocate_memory(info, None) }.map_err(|result| {
        counters.fall(Gauge::Blocks);
        Error::Vulkan {
            call: "vkAllocateMemory",
            result,
        }
    })
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
    let allocation = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(core.memory_type(class))
        .push_next(&mut flags);
    let bits = requirements.memory_type_bits;
    // SAFETY: the floor chose the memory type for buffers of this usage on this device, and
    // `bufferDeviceAddress` is enabled; a spill takes a type the buffer's requirements allow.
    let allocated = unsafe { allocate_spilling(core, class, bits, &allocation) };
    let (memory, spilled) = match allocated {
        Ok(memory) => memory,
        Err(error) => {
            // SAFETY: `buffer` is unused.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(error);
        }
    };
    core.counters.count(Metric::BlocksMade);
    let device_local = match spilled {
        true => false,
        false => core.device_local & (1 << core.memory_type(class)) != 0,
    };
    if spilled {
        core.counters.rise_by(Gauge::SpilledBytes, len);
    }
    let mut block = Memory {
        id,
        class,
        len,
        pooled,
        buffer,
        memory,
        address: 0,
        mapping: None,
        device_local,
        spilled,
    };
    match finish(core, &mut block) {
        Ok(()) => Ok(block),
        Err(error) => {
            free(core, block);
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
fn zero(core: &Core, memory: &Memory, offset: u64, end: u64) -> Result<(), Error> {
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

/// Frees a block, calling the consumer's hook first where it was handed out and the hook has not
/// run for it. Freeing the memory also unmaps it.
pub(crate) fn free(core: &Core, memory: Memory) {
    let handed = core.pool.lock().handed.remove(&memory.id);
    if handed {
        if let Some(mappings) = &core.pool.mappings {
            mappings.unmap(memory.id);
        }
    }
    let device = &core.device;
    // SAFETY: no GPU work that uses the block is pending: a command buffer or upload that uses it
    // holds it until it retires, and the fills of the block-maker thread are waited for. The
    // consumer's mappings are gone.
    unsafe {
        device.destroy_buffer(memory.buffer, None);
        device.free_memory(memory.memory, None);
    }
    core.counters.fall(Gauge::Blocks);
    if memory.spilled {
        core.counters.fall_by(Gauge::SpilledBytes, memory.len);
    }
}
