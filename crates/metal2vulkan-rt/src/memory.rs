//! Memory: the blocks the consumer takes and releases, each a range of one of the device's
//! allocations; the pool that hands them out, keeps them, and frees idle allocations; the
//! allocations its own thread makes ahead of need, while a class's free bytes run low; and how
//! memory is made, zeroed and freed.

use crate::counters::{Counters, Gauge, Metric};
use crate::device::{lock, Core};
use crate::error::{Error, Limit};
use crate::floor;
use crate::places::Ranges;
use ash::vk;
use std::collections::{HashSet, VecDeque};
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

/// The most bytes of a class's allocations in use (holding a live block) in each of the last
/// [`WINDOW_SECS`] seconds that had a take or a give-back, as `(second, bytes)`, the oldest first.
#[derive(Default)]
pub(crate) struct Peaks(VecDeque<(u64, u64)>);

impl Peaks {
    /// Records `live` bytes in use in second `now`, and forgets the seconds past the window.
    pub(crate) fn note(&mut self, now: u64, live: u64) {
        match self.0.back_mut() {
            Some((second, most)) if *second == now => *most = (*most).max(live),
            _ => self.0.push_back((now, live)),
        }
        while (self.0.front()).is_some_and(|&(second, _)| second + WINDOW_SECS <= now) {
            self.0.pop_front();
        }
    }

    /// The most bytes in use recorded in the window.
    pub(crate) fn most(&self) -> u64 {
        self.0.iter().map(|&(_, bytes)| bytes).max().unwrap_or(0)
    }
}

/// How many of a class's idle allocations of `idle` bytes, the oldest first, the idle rule frees:
/// the oldest, while the rest pass `peak`, the most bytes in use in the window, less `live`, the
/// bytes in use now, or `cap`.
pub(crate) fn surplus(idle: &[u64], peak: u64, live: u64, cap: u64) -> usize {
    let room = peak.saturating_sub(live).min(cap);
    let mut left: u64 = idle.iter().sum();
    idle.iter()
        .take_while(|&&len| {
            let over = left > room;
            left -= len;
            over
        })
        .count()
}

/// The length of each allocation the block-maker thread makes, and of an allocation a take makes
/// where no free range fits it, unless the take is longer: 16 MiB of either class.
///
/// A driver may hold the device's other calls (pipeline creation, submissions, other allocations)
/// for the whole of an allocation's `vkAllocateMemory`, and the hold grows with the bytes: on an
/// RTX 3090 (NVIDIA's Linux driver) about 0.25 ms per MiB of Shared memory and a third of that of
/// Private. On a UHD Graphics 630 (Intel's Windows driver) the call takes 0.1 to 0.2 ms for 16 to
/// 64 MiB of Private memory and holds nothing measurable, while the making of a new Private
/// allocation, whose zero fill is its first use, held other submissions 0.4 to 0.75 ms per MiB.
/// So allocations are short, and a take longer than one pays its own length once: the allocation
/// it makes serves later takes as free ranges.
const ALLOCATION_LEN: u64 = 16 << 20;

/// The block-maker thread makes an allocation of a class while the class's free bytes, the free
/// ranges of all its allocations, are fewer than this: half an allocation. A take that leaves more
/// free makes nothing happen beside the consumer's next calls, so a small take beside a pipeline's
/// creation does not set off an allocation that the driver holds the creation for.
const MAKE_BELOW: u64 = ALLOCATION_LEN / 2;

/// A block's length, and its first byte's offset in its allocation, are multiples of this, or of
/// the device's `bufferImageGranularity` where that is larger: no block shares a page of that
/// granularity with another, so a block of buffers lies beside a block of optimal images.
const GRANULE: u64 = 64 << 10;

/// A block's first byte lies at a multiple of its length rounded up to a power of two, up to this
/// much, in its allocation: an image placed in a block at an offset its alignment allows from the
/// block's first byte is placed at one from the allocation's.
const BASE_ALIGN: u64 = 2 << 20;

/// The idle rule's window: the pool keeps a class's idle allocations while their bytes and those
/// of the class's allocations in use stay at or under the most bytes the class had in use over the
/// last this many seconds. An allocation is in use while it holds a live block; weighing whole
/// allocations, not the blocks' bytes, keeps an allocation that serves the same blocks each frame,
/// whatever room they leave in it.
const WINDOW_SECS: u64 = 10;

/// The share of the heap of a class's memory type its idle allocations stay within, whatever the
/// window holds: an eighth.
const IDLE_SHARE: u64 = 8;

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

    /// No Metal counterpart: the length of the allocations the device makes for the class ahead
    /// of need, 16 MiB of either class.
    ///
    /// A [`Device::take_block`](crate::Device::take_block) is a range of one of the class's
    /// allocations, the first with room; the block-maker thread makes another while the class's
    /// free bytes are fewer than half this. A take that no free range fits makes an allocation of
    /// this length, or of its own where it is longer, which later takes share.
    pub const fn block_len(self) -> u64 {
        ALLOCATION_LEN
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
/// device's life; a block released and taken again at its length keeps its number.
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
/// Plain data, which the block's handle ([`Block`](crate::Block)) derefs to: the block lives
/// until its last handle and its last buffer, command buffer or upload are gone, or until the
/// device is torn down. One `VkBuffer` spans it, with every buffer usage and a device
/// address. Nothing in it is exportable.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct BlockInfo {
    /// The block's number.
    pub id: BlockId,
    /// Its storage mode.
    pub class: MemoryClass,
    /// The length asked for, rounded up to 64 KiB (to the device's `bufferImageGranularity`
    /// where that is larger).
    pub len: u64,
    /// `gpuAddress` of the block's first byte: its `VkBuffer`'s device address.
    pub gpu_address: u64,
    /// `contents` of the block's first byte: the host mapping of a Shared block, `None` for a
    /// Private one.
    pub contents: Option<NonNull<u8>>,
    /// Whether its memory type is device-local. A Private block is, unless the device-local type
    /// was out of memory when its allocation was made and it went to the next type its buffer
    /// takes (counted in [`Gauges::spilled_bytes`](crate::Gauges::spilled_bytes)); a Shared
    /// block is where the device's Shared type is.
    pub device_local: bool,
}

// SAFETY: plain data. `contents` is an address; reading or writing through it is the consumer's
// own unsafe code, under the contract of `BlockMappings`.
unsafe impl Send for BlockInfo {}
// SAFETY: as above.
unsafe impl Sync for BlockInfo {}

/// No Metal counterpart: the consumer's hook before the runtime frees memory it handed out, or
/// hands it out again in another block.
///
/// The runtime calls [`unmap`](BlockMappings::unmap) once for every block
/// [`Device::take_block`](crate::Device::take_block) returned, before the block's bytes are freed
/// or become part of another block: when the pool gives a kept block's bytes back to its
/// allocation for a take of another length, when it frees an idle allocation, when a block given
/// back could not be zeroed or spilled, and at teardown for every block still alive. A block
/// released and taken again at its length keeps its number, address and mapping, and its hook has
/// not run. Blocks the consumer never took are freed without it.
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

/// A block: a range of an allocation, the `VkBuffer` bound to it alone, and its mapping.
#[derive(Clone, Copy)]
pub(crate) struct Memory {
    pub(crate) id: BlockId,
    pub(crate) class: MemoryClass,
    pub(crate) len: u64,
    /// Its allocation, and its first byte's offset there.
    memory: vk::DeviceMemory,
    base: u64,
    pub(crate) buffer: vk::Buffer,
    pub(crate) address: u64,
    pub(crate) mapping: Option<Mapping>,
    /// Its memory type is device-local.
    device_local: bool,
    /// A Private block in another memory type than the Private one, which was out of memory.
    spilled: bool,
}

impl Memory {
    /// The allocation the block is a range of; an image bound in the block binds at
    /// [`base`](Self::base) plus its offset in the block.
    pub(crate) fn device_memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    /// The offset of the block's first byte in its allocation.
    pub(crate) fn base(&self) -> u64 {
        self.base
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

/// A host address in a Shared allocation's mapping, which lives as long as the allocation: its
/// first byte, a block's or a buffer's.
#[derive(Clone, Copy)]
pub(crate) struct Mapping(NonNull<u8>);

// SAFETY: an address. The runtime writes through it only while it owns the bytes: the block-maker
// thread zeroing memory no one holds, and `new_buffer` zeroing the range of the buffer it makes.
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
    /// `self` is an allocation's or a block's first byte and `offset` is at most its length.
    pub(crate) unsafe fn at(self, offset: u64) -> NonNull<u8> {
        // SAFETY: the mapping spans the memory, and `offset` is inside it or one past its end;
        // memory that was mapped has a length that fits the address space.
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

/// One `VkDeviceMemory` of a class and the ranges of it no block holds.
struct Allocation {
    memory: vk::DeviceMemory,
    len: u64,
    mapping: Option<Mapping>,
    device_local: bool,
    spilled: bool,
    /// Bytes no block holds, handed out or kept: zeroed.
    free: Ranges,
    /// Its blocks handed out and not yet given back to the pool.
    live: usize,
    /// When its last live block came back, in the pool's give-backs: the oldest idle allocation
    /// is freed first.
    emptied: u64,
}

/// The pool: every allocation of each class, and the blocks given back and kept for a take of
/// their length.
///
/// The block-maker thread ([`run`]) makes an allocation of a class while the class's free bytes
/// are fewer than [`MAKE_BELOW`]; a take never waits for it. A take that no free range fits, and
/// that no kept block of its length serves, makes an allocation itself, and where the class keeps
/// blocks of other lengths, asks the block-maker thread to give their bytes back to their
/// allocations, after the consumer's hook on that thread. Only where its allocation fails does it
/// wait, for those bytes, and look for a free range once more. It waits only while the block-maker
/// thread works for it, by one predicate both read: [`State::needs_maker`].
///
/// The idle rule never frees an allocation that the class's free bytes need to stay at or above
/// [`MAKE_BELOW`], so it does not free what the block-maker thread would make again.
///
/// The idle rule runs as a block is taken or given back, never on a timer: a program that goes
/// quiet keeps its idle allocations until its next take or release, or until the device is torn
/// down.
pub(crate) struct Pool {
    state: Mutex<State>,
    /// Signalled whenever the state changes, for the block-maker thread and for takers waiting
    /// on it.
    changed: Condvar,
    mappings: Option<Arc<dyn BlockMappings>>,
    /// The device's counters: the idle bytes, the allocations made and the blocks taken again.
    counters: Counters,
    /// Per class, the most bytes of idle allocations: [`IDLE_SHARE`] of its memory type's heap.
    caps: [u64; 2],
    /// A block's length and its first byte's offset are multiples of this: [`GRANULE`], or the
    /// device's `bufferImageGranularity` where that is larger.
    granule: u64,
    /// The idle rule's clock: seconds since the pool was made.
    clock: Clock,
}

/// The idle rule's clock, which the device tests move on.
struct Clock {
    origin: Instant,
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    skew: Mutex<Duration>,
}

impl Clock {
    fn new() -> Clock {
        Clock {
            origin: Instant::now(),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            skew: Mutex::new(Duration::ZERO),
        }
    }

    fn seconds(&self) -> u64 {
        let elapsed: Duration = self.origin.elapsed();
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        let elapsed = elapsed + *lock(&self.skew);
        elapsed.as_secs()
    }
}

struct State {
    /// Per class, its allocations in the order they were made.
    allocations: [Vec<Allocation>; 2],
    /// Per class, blocks given back, zeroed, the oldest first, with their number, buffer, address
    /// and mapping, for a take of their length. Their bytes stay out of their allocation's free
    /// ranges until the block-maker thread gives them back there, after the consumer's hook.
    kept: [Vec<Memory>; 2],
    /// Per class, a take asked the block-maker thread for the class's kept blocks' bytes.
    unkeep: [bool; 2],
    /// Blocks given back, for the block-maker thread.
    returned: Vec<Memory>,
    /// Per class, the block-maker thread could not make the last allocation it tried. It tries
    /// again after the class's next take or give-back.
    failed: [bool; 2],
    stop: bool,
    next_id: u32,
    /// The blocks `take_block` returned whose hook has not run: running it removes one.
    handed: HashSet<BlockId>,
    /// The hook has run for every block handed out ([`Pool::unmap_all`]); it never runs again.
    unmapped: bool,
    /// Per class, the most bytes in use in each of the last seconds with a take or a give-back.
    peaks: [Peaks; 2],
    /// Per class, the idle allocations' bytes the class's gauge shows.
    idle_bytes: [u64; 2],
    /// Blocks given back so far: when each idle allocation emptied.
    give_backs: u64,
}

thread_local! {
    /// Whether this thread makes allocations ahead of need: the block-maker thread ([`run`]), or
    /// the device's making while it makes the first allocations ([`stock`]). For the
    /// allocation-time metrics.
    static ON_MAKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// What a block's `VkBuffer` is bound to: a range of an allocation.
#[derive(Clone, Copy)]
struct Carve {
    memory: vk::DeviceMemory,
    base: u64,
    mapping: Option<Mapping>,
    device_local: bool,
    spilled: bool,
}

impl Pool {
    /// The pool of a device whose classes' memory types are in heaps of `heaps` bytes (Private,
    /// Shared) and whose `bufferImageGranularity` is `granularity`.
    pub(crate) fn new(
        mappings: Option<Arc<dyn BlockMappings>>,
        counters: Counters,
        heaps: [u64; 2],
        granularity: u64,
    ) -> Pool {
        Pool {
            state: Mutex::new(State {
                allocations: [Vec::new(), Vec::new()],
                kept: [Vec::new(), Vec::new()],
                unkeep: [false, false],
                returned: Vec::new(),
                failed: [false, false],
                stop: false,
                next_id: 1,
                handed: HashSet::new(),
                unmapped: false,
                peaks: [Peaks::default(), Peaks::default()],
                idle_bytes: [0, 0],
                give_backs: 0,
            }),
            changed: Condvar::new(),
            mappings,
            counters,
            caps: heaps.map(|heap| heap / IDLE_SHARE),
            granule: GRANULE.max(granularity.next_power_of_two()),
            clock: Clock::new(),
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

    /// For the device tests: whether the class's free bytes are at least [`MAKE_BELOW`], so the
    /// block-maker thread makes nothing for it (an allocation it made is counted once its fill's
    /// submission completed).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn stocked(&self, class: MemoryClass) -> bool {
        self.lock().free(class) >= MAKE_BELOW
    }

    /// For the device tests: the class's blocks given back, zeroed and kept.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn kept(&self, class: MemoryClass) -> usize {
        self.lock().kept[class.index()].len()
    }

    /// For the device tests: moves the idle rule's clock on by `by`.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn advance_clock(&self, by: Duration) {
        *lock(&self.clock.skew) += by;
    }

    /// The length and alignment of a block for a take of `len` bytes: its buffer's size rounded
    /// up to the granule, at a multiple of that rounded up to a power of two (at most
    /// [`BASE_ALIGN`]), of its buffer's alignment and of the granule.
    ///
    /// Err: [`Limit::BlockLen`] past `max`, the device's longest allocation and buffer.
    fn shape(&self, core: &Core, len: u64, max: u64) -> Result<(u64, u64), Error> {
        let asked = round(len, self.granule, max)?;
        let info = vk::BufferCreateInfo::default()
            .size(asked)
            .usage(floor::BUFFER_USAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let of = vk::DeviceBufferMemoryRequirements::default().create_info(&info);
        let mut needs = vk::MemoryRequirements2::default();
        // SAFETY: the device is live, and the create info valid: `asked` is at most its
        // `maxBufferSize`.
        unsafe {
            core.device
                .get_device_buffer_memory_requirements(&of, &mut needs)
        };
        let requirements = needs.memory_requirements;
        let len = round(requirements.size.max(asked), self.granule, max)?;
        Ok((len, base_align(len, requirements.alignment, self.granule)))
    }

    /// Takes a block of `class` of at least `len` bytes for `core`: a kept block of its length,
    /// with its number, buffer, address and mapping; else a free range, the first that fits in the
    /// order the allocations were made, with a new number and buffer; else a range of an
    /// allocation made here, of [`ALLOCATION_LEN`] or the block's length where that is longer.
    /// Where the class keeps blocks of other lengths, that take also asks the block-maker thread to
    /// give their bytes back; and where its allocation fails, it waits for them, and takes a free
    /// range of them where one fits. Every block reads zero.
    ///
    /// Err: [`Limit::BlockLen`] past the device's longest allocation or buffer;
    /// [`Limit::BlockIds`]; what [`make_allocation`] and binding the block's buffer return;
    /// [`Error::Quiesced`] once the pool is stopped.
    pub(crate) fn take(
        &self,
        core: &Core,
        class: MemoryClass,
        len: u64,
        max: u64,
    ) -> Result<Memory, Error> {
        let (len, align) = self.shape(core, len, max)?;
        let index = class.index();
        let mut state = self.lock();
        // A take is a pool event: the block-maker thread tries a failed allocation again.
        state.failed[index] = false;
        // Why this take's allocation failed, once it has waited for the kept blocks' bytes.
        let mut failure = None;
        let made = loop {
            if state.stop {
                return Err(Error::Quiesced);
            }
            if let Some(at) = state.kept[index].iter().rposition(|kept| kept.len == len) {
                let memory = state.kept[index].remove(at);
                self.counters.count(Metric::BlocksReused);
                self.hold(&mut state, &memory);
                self.settle(&mut state, class);
                self.changed.notify_all();
                return Ok(memory);
            }
            if let Some(carve) = state.fit(class, len, align) {
                let id = number(&mut state)?;
                drop(state);
                break self.bind_range(core, class, len, carve, id);
            }
            if let Some(error) = failure {
                return Err(error);
            }
            let asked = !state.kept[index].is_empty();
            if asked {
                // The block-maker thread gives the kept blocks' bytes back, after the hook; this
                // take does not wait for it.
                state.unkeep[index] = true;
                self.changed.notify_all();
            }
            let id = number(&mut state)?;
            drop(state);
            match self.own(core, class, len, align, id) {
                Ok(memory) => break Ok(memory),
                Err(error) => {
                    state = self.lock();
                    if !asked && !state.needs_maker(class) {
                        return Err(error);
                    }
                    while state.needs_maker(class) && !state.stop {
                        state = self.wait(state);
                    }
                    failure = Some(error);
                }
            }
        };
        let memory = made?;
        let mut state = self.lock();
        state.note(class, self.clock.seconds());
        self.settle(&mut state, class);
        // The block-maker thread runs the idle rule on this take, and makes an allocation where
        // the class's free bytes ran low.
        self.changed.notify_all();
        Ok(memory)
    }

    /// Makes block `id` of `len` bytes at `carve`, a range reserved for it; where that fails, the
    /// range goes back, its bytes still zero.
    fn bind_range(
        &self,
        core: &Core,
        class: MemoryClass,
        len: u64,
        carve: Carve,
        id: BlockId,
    ) -> Result<Memory, Error> {
        let bound = bind(core, class, len, carve, id);
        if bound.is_err() {
            let mut state = self.lock();
            let emptied = state.give_backs;
            if let Some(allocation) = state.find(class, carve.memory) {
                allocation.free.give(carve.base, len);
                allocation.live -= 1;
                if allocation.live == 0 {
                    allocation.emptied = emptied;
                }
            }
            self.settle(&mut state, class);
        }
        bound
    }

    /// Makes block `id` of `len` bytes at a multiple of `align` at the start of an allocation of
    /// its own, of [`ALLOCATION_LEN`] or `len` where that is longer, whose rest later takes share.
    fn own(
        &self,
        core: &Core,
        class: MemoryClass,
        len: u64,
        align: u64,
        id: BlockId,
    ) -> Result<Memory, Error> {
        let mut allocation = make_allocation(core, class, len.max(ALLOCATION_LEN))?;
        let base = allocation.free.take(len, align).unwrap_or(0);
        let carve = allocation.carve(base);
        match bind(core, class, len, carve, id) {
            Ok(memory) => {
                allocation.live = 1;
                self.lock().allocations[class.index()].push(allocation);
                Ok(memory)
            }
            Err(error) => {
                free_allocation(core, allocation);
                Err(error)
            }
        }
    }

    /// Counts a kept block taken again as live in its allocation.
    fn hold(&self, state: &mut State, memory: &Memory) {
        if let Some(allocation) = state.find(memory.class, memory.memory) {
            allocation.live += 1;
        }
        state.note(memory.class, self.clock.seconds());
    }

    /// Sets the class's idle gauge to its idle allocations' bytes.
    fn settle(&self, state: &mut State, class: MemoryClass) {
        let index = class.index();
        let idle: u64 = (state.allocations[index].iter())
            .filter(|allocation| allocation.live == 0)
            .map(|allocation| allocation.len)
            .sum();
        let was = state.idle_bytes[index];
        if idle > was {
            self.counters.rise_by(class.idle_gauge(), idle - was);
        } else {
            self.counters.fall_by(class.idle_gauge(), was - idle);
        }
        state.idle_bytes[index] = idle;
    }

    /// The idle rule for `class`: takes out its oldest idle allocations while their bytes pass the
    /// most bytes the class had in use over the last [`WINDOW_SECS`] seconds less its bytes in use
    /// now, or the class's cap; each with its kept blocks, for the caller to free without the
    /// lock. The newest idle allocations with free bytes that the class's free bytes need to reach
    /// [`MAKE_BELOW`] are its reserve: the rule neither counts nor frees them.
    fn trim(&self, state: &mut State, class: MemoryClass) -> Vec<(Allocation, Vec<Memory>)> {
        let index = class.index();
        state.note(class, self.clock.seconds());
        let allocations = &state.allocations[index];
        let mut free: u64 = (allocations.iter())
            .filter(|allocation| allocation.live > 0)
            .map(|allocation| allocation.free.bytes())
            .sum();
        let mut idle: Vec<(u64, vk::DeviceMemory, u64, u64)> = (allocations.iter())
            .filter(|allocation| allocation.live == 0)
            .map(|allocation| {
                let bytes = allocation.free.bytes();
                (allocation.emptied, allocation.memory, allocation.len, bytes)
            })
            .collect();
        idle.sort_unstable_by_key(|&(emptied, ..)| std::cmp::Reverse(emptied));
        idle.retain(|&(.., bytes)| {
            let reserve = free < MAKE_BELOW && bytes > 0;
            if reserve {
                free += bytes;
            }
            !reserve
        });
        idle.reverse();
        let lens: Vec<u64> = idle.iter().map(|&(_, _, len, _)| len).collect();
        let peak = state.peaks[index].most();
        let surplus = surplus(&lens, peak, state.in_use(class), self.caps[index]);
        let gone = (idle.iter().take(surplus))
            .filter_map(|&(_, memory, ..)| state.take_out(class, memory))
            .collect();
        self.settle(state, class);
        gone
    }

    /// Records a block as handed to the consumer, so that its hook runs before its bytes are
    /// freed or handed out in another block.
    pub(crate) fn hand(&self, block: BlockId) {
        let mut state = self.lock();
        if !state.unmapped {
            state.handed.insert(block);
        }
    }

    /// Runs the consumer's hook for `block` where it was handed out and its hook has not run.
    fn unmap(&self, block: BlockId) {
        let handed = self.lock().handed.remove(&block);
        if let Some(mappings) = self.mappings.as_ref().filter(|_| handed) {
            mappings.unmap(block);
        }
    }

    /// In [`Device::quiesce`](crate::Device::quiesce) and at teardown, once the runtime's other
    /// threads are stopped: calls the hook for every block handed out whose hook has not run, on
    /// this thread. No block's hook runs again: a block freed later is freed without it.
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
    /// keep, or give back to its allocation.
    pub(crate) fn give_back(&self, memory: Memory) {
        self.lock().returned.push(memory);
        self.changed.notify_all();
    }

    /// Stops the block-maker thread once it finishes what it is doing.
    pub(crate) fn stop(&self) {
        self.lock().stop = true;
        self.changed.notify_all();
    }

    /// For the device tests: whether the pool has the block back, kept or still to be zeroed.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn holds(&self, block: BlockId) -> bool {
        let state = self.lock();
        (state.kept.iter().flatten())
            .chain(&state.returned)
            .any(|memory| memory.id == block)
    }
}

impl State {
    /// Whether the block-maker thread works for a take of `class`: a take asked for the class's
    /// kept blocks' bytes, which the thread has not yet given back to their allocations.
    /// [`Pool::take`] waits only while this holds, and [`run`] works while it holds for any class.
    fn needs_maker(&self, class: MemoryClass) -> bool {
        self.unkeep[class.index()]
    }

    /// The bytes of the free ranges of the class's allocations.
    fn free(&self, class: MemoryClass) -> u64 {
        (self.allocations[class.index()].iter())
            .map(|allocation| allocation.free.bytes())
            .sum()
    }

    /// The allocation of `class` whose memory is `memory`.
    fn find(&mut self, class: MemoryClass, memory: vk::DeviceMemory) -> Option<&mut Allocation> {
        (self.allocations[class.index()].iter_mut()).find(|allocation| allocation.memory == memory)
    }

    /// Reserves `len` bytes at a multiple of `align` in the first allocation of `class` with such
    /// a free range, counting a block live there.
    fn fit(&mut self, class: MemoryClass, len: u64, align: u64) -> Option<Carve> {
        (self.allocations[class.index()].iter_mut()).find_map(|allocation| {
            let base = allocation.free.take(len, align)?;
            allocation.live += 1;
            Some(allocation.carve(base))
        })
    }

    /// The bytes of the class's allocations that hold a live block.
    fn in_use(&self, class: MemoryClass) -> u64 {
        (self.allocations[class.index()].iter())
            .filter(|allocation| allocation.live > 0)
            .map(|allocation| allocation.len)
            .sum()
    }

    /// Records the class's bytes in use now in the second `now`.
    fn note(&mut self, class: MemoryClass, now: u64) {
        let in_use = self.in_use(class);
        self.peaks[class.index()].note(now, in_use);
    }

    /// Takes the allocation of `class` whose memory is `memory` out of the pool, with its kept
    /// blocks, for the caller to free once they are retired.
    fn take_out(
        &mut self,
        class: MemoryClass,
        memory: vk::DeviceMemory,
    ) -> Option<(Allocation, Vec<Memory>)> {
        let index = class.index();
        let at =
            (self.allocations[index].iter()).position(|allocation| allocation.memory == memory)?;
        let allocation = self.allocations[index].remove(at);
        let (kept, stay): (Vec<Memory>, Vec<Memory>) =
            (self.kept[index].drain(..)).partition(|kept| kept.memory == memory);
        self.kept[index] = stay;
        Some((allocation, kept))
    }
}

impl Allocation {
    /// What a block's buffer at `base` binds to.
    fn carve(&self, base: u64) -> Carve {
        Carve {
            memory: self.memory,
            base,
            // SAFETY: `base` is inside the allocation, which the mapping spans.
            mapping: self
                .mapping
                .map(|mapping| Mapping(unsafe { mapping.at(base) })),
            device_local: self.device_local,
            spilled: self.spilled,
        }
    }
}

/// A block's length for `len` bytes: `len` (at least 1) rounded up to `granule`, a power of two.
///
/// Err: [`Limit::BlockLen`] past `max`, the device's longest allocation and buffer.
pub(crate) fn round(len: u64, granule: u64, max: u64) -> Result<u64, Error> {
    match len.max(1).checked_next_multiple_of(granule) {
        Some(rounded) if rounded <= max => Ok(rounded),
        rounded => Err(Error::Limit(Limit::BlockLen {
            len: rounded.unwrap_or(u64::MAX),
            max,
        })),
    }
}

/// The alignment of the first byte of a block of `len` bytes in its allocation: `len` rounded up
/// to a power of two, at most [`BASE_ALIGN`], and at least its buffer's `alignment` and the
/// `granule`.
pub(crate) fn base_align(len: u64, alignment: u64, granule: u64) -> u64 {
    (len.checked_next_power_of_two().unwrap_or(BASE_ALIGN))
        .min(BASE_ALIGN)
        .max(alignment)
        .max(granule)
}

fn number(state: &mut State) -> Result<BlockId, Error> {
    let id = state.next_id;
    state.next_id = id.checked_add(1).ok_or(Error::Limit(Limit::BlockIds))?;
    Ok(BlockId(id))
}

/// In the device's making, before the block-maker thread starts: makes each class's first
/// allocation, so the consumer's first blocks of each class are ranges of it as the device is
/// ready, and nothing is allocated beside the consumer's first calls. Where one cannot be made,
/// the block-maker thread tries again after the class's next take or give-back.
pub(crate) fn stock(core: &Core) {
    // Made ahead of need, as the block-maker thread's are: counted with them.
    ON_MAKER.set(true);
    for class in MemoryClass::ALL {
        let made = make_allocation(core, class, ALLOCATION_LEN);
        let mut state = core.pool.lock();
        match made {
            Ok(allocation) => state.allocations[class.index()].push(allocation),
            Err(_) => state.failed[class.index()] = true,
        }
        core.pool.settle(&mut state, class);
    }
    ON_MAKER.set(false);
}

/// The block-maker thread, until the device's teardown stops it. Off every caller's path, in
/// order: it gives a class's kept blocks' bytes back to their allocations where a take asked,
/// after the consumer's hook; it zeroes each block given back and keeps it for a take of its
/// length (or gives its bytes back, after the hook, where it cannot be zeroed or spilled), then
/// runs the class's idle rule, freeing what it takes out after the hook of each kept block in it;
/// and it makes an allocation of a class whose free bytes are fewer than [`MAKE_BELOW`].
pub(crate) fn run(core: Arc<Core>) {
    ON_MAKER.set(true);
    let pool = &core.pool;
    let mut state = pool.lock();
    loop {
        if state.stop {
            return;
        }
        if let Some(class) = MemoryClass::ALL
            .into_iter()
            .find(|c| state.unkeep[c.index()])
        {
            let kept: Vec<Memory> = state.kept[class.index()].split_off(0);
            drop(state);
            for memory in &kept {
                retire(&core, memory);
            }
            state = pool.lock();
            for memory in &kept {
                if let Some(allocation) = state.find(class, memory.memory) {
                    allocation.free.give(memory.base, memory.len);
                }
            }
            state.unkeep[class.index()] = false;
            pool.changed.notify_all();
            continue;
        }
        if let Some(memory) = state.returned.pop() {
            drop(state);
            // A block that cannot be zeroed, or spilled, is not kept: its bytes go back to its
            // allocation, and a spilled allocation that empties is freed, so the next tries
            // device-local memory again.
            let kept = !memory.spilled && zero(&core, &memory, 0, memory.len).is_ok();
            if !kept {
                retire(&core, &memory);
            }
            state = pool.lock();
            let class = memory.class;
            // The block's allocation was in use until now, however long ago it was taken.
            state.note(class, pool.clock.seconds());
            state.give_backs += 1;
            state.failed[class.index()] = false;
            let emptied = state.give_backs;
            let mut spilled = None;
            if let Some(allocation) = state.find(class, memory.memory) {
                allocation.live -= 1;
                if !kept {
                    allocation.free.give(memory.base, memory.len);
                }
                if allocation.live == 0 {
                    allocation.emptied = emptied;
                    spilled = allocation.spilled.then_some(allocation.memory);
                }
            }
            if kept {
                state.kept[class.index()].push(memory);
            }
            let mut gone = pool.trim(&mut state, class);
            if let Some(spilled) = spilled.and_then(|memory| state.take_out(class, memory)) {
                gone.push(spilled);
                pool.settle(&mut state, class);
            }
            pool.changed.notify_all();
            drop(state);
            for (allocation, kept) in gone {
                for memory in &kept {
                    retire(&core, memory);
                }
                free_allocation(&core, allocation);
            }
            state = pool.lock();
            continue;
        }
        let gone: Vec<(Allocation, Vec<Memory>)> = (MemoryClass::ALL.into_iter())
            .flat_map(|class| pool.trim(&mut state, class))
            .collect();
        if !gone.is_empty() {
            drop(state);
            for (allocation, kept) in gone {
                for memory in &kept {
                    retire(&core, memory);
                }
                free_allocation(&core, allocation);
            }
            state = pool.lock();
            continue;
        }
        let wanted = (MemoryClass::ALL.into_iter())
            .find(|&class| !state.failed[class.index()] && state.free(class) < MAKE_BELOW);
        if let Some(class) = wanted {
            drop(state);
            let made = make_allocation(&core, class, ALLOCATION_LEN);
            state = pool.lock();
            match made {
                Ok(mut allocation) => {
                    // Newer than every idle allocation: the idle rule frees it last.
                    allocation.emptied = state.give_backs;
                    state.allocations[class.index()].push(allocation);
                }
                Err(_) => state.failed[class.index()] = true,
            }
            pool.settle(&mut state, class);
            pool.changed.notify_all();
            continue;
        }
        state = pool.wait(state);
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
/// where `max`, the device's `maxMemoryAllocationCount`, are live. Each free falls the gauge. The
/// call's time is counted on the block-maker thread or on the consumer's.
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
    let start = crate::host_time();
    // SAFETY: the caller's.
    let allocated = unsafe { device.allocate_memory(info, None) };
    counters.allocation_time(crate::host_time().saturating_sub(start), ON_MAKER.get());
    allocated.map_err(|result| {
        counters.fall(Gauge::Blocks);
        Error::Vulkan {
            call: "vkAllocateMemory",
            result,
        }
    })
}

/// A `VkBuffer` of `len` bytes with every buffer usage.
fn new_buffer(device: &ash::Device, len: u64) -> Result<vk::Buffer, Error> {
    let info = vk::BufferCreateInfo::default()
        .size(len)
        .usage(floor::BUFFER_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: the device is live; `len` is at most the device's `maxBufferSize`.
    unsafe { device.create_buffer(&info, None) }.map_err(Error::vulkan("vkCreateBuffer"))
}

/// Makes a zeroed allocation of `len` bytes of `class` with device addresses, mapped whole if
/// Shared, its free ranges all of it. A Shared allocation is zeroed through the mapping, which
/// also makes every page resident before any block holds it; a Private one by a GPU fill through
/// a buffer over it, destroyed after.
fn make_allocation(core: &Core, class: MemoryClass, len: u64) -> Result<Allocation, Error> {
    let device = &core.device;
    let buffer = new_buffer(device, len)?;
    // SAFETY: `buffer` is the device's.
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size.max(len))
        .memory_type_index(core.memory_type(class))
        .push_next(&mut flags);
    let bits = requirements.memory_type_bits;
    // SAFETY: the floor chose the memory type for buffers of this usage on this device, and
    // `bufferDeviceAddress` is enabled; a spill takes a type the buffer's requirements allow.
    let allocated = unsafe { allocate_spilling(core, class, bits, &info) };
    let (memory, spilled) = match allocated {
        Ok(memory) => memory,
        Err(error) => {
            // SAFETY: `buffer` is unused.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(error);
        }
    };
    core.counters.count(Metric::BlocksMade);
    core.counters.allocated(len);
    if spilled {
        core.counters.rise_by(Gauge::SpilledBytes, len);
    }
    let device_local = !spilled && core.device_local & (1 << core.memory_type(class)) != 0;
    let mut allocation = Allocation {
        memory,
        len,
        mapping: None,
        device_local,
        spilled,
        free: Ranges::new(len),
        live: 0,
        emptied: 0,
    };
    let zeroed = finish(core, class, &mut allocation, buffer);
    // SAFETY: the fill that used it completed, or none was recorded.
    unsafe { device.destroy_buffer(buffer, None) };
    match zeroed {
        Ok(()) => Ok(allocation),
        Err(error) => {
            free_allocation(core, allocation);
            Err(error)
        }
    }
}

/// Maps a new Shared allocation, and zeroes a new allocation through `buffer`, bound to it whole
/// for a Private one.
fn finish(
    core: &Core,
    class: MemoryClass,
    allocation: &mut Allocation,
    buffer: vk::Buffer,
) -> Result<(), Error> {
    let device = &core.device;
    if class == MemoryClass::Shared {
        // SAFETY: the memory type is host-visible, and the memory is not mapped.
        let pointer = unsafe {
            device.map_memory(
                allocation.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
        }
        .map_err(Error::vulkan("vkMapMemory"))?;
        allocation.mapping = NonNull::new(pointer.cast::<u8>()).map(Mapping);
        if let Some(mapping) = allocation.mapping {
            // SAFETY: the mapping spans the allocation, which no one else holds.
            unsafe { mapping.0.as_ptr().write_bytes(0, allocation.len as usize) };
        }
        return Ok(());
    }
    // SAFETY: the allocation is the buffer's size and of a type its requirements allow.
    unsafe { device.bind_buffer_memory(buffer, allocation.memory, 0) }
        .map_err(Error::vulkan("vkBindBufferMemory"))?;
    core.filler.zero(core, buffer, 0, allocation.len)
}

/// Makes block `id` of `class`: a buffer of `len` bytes bound at `carve`'s range, its address and
/// its mapping. Its bytes are already zero.
fn bind(
    core: &Core,
    class: MemoryClass,
    len: u64,
    carve: Carve,
    id: BlockId,
) -> Result<Memory, Error> {
    let device = &core.device;
    let buffer = new_buffer(device, len)?;
    // SAFETY: the range is `len` bytes of an allocation of a type the buffer's requirements allow,
    // at a multiple of their alignment ([`Pool::shape`]), and no other buffer is bound there.
    if let Err(result) = unsafe { device.bind_buffer_memory(buffer, carve.memory, carve.base) } {
        // SAFETY: `buffer` is unused.
        unsafe { device.destroy_buffer(buffer, None) };
        return Err(Error::Vulkan {
            call: "vkBindBufferMemory",
            result,
        });
    }
    let info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    // SAFETY: the buffer has device-address usage and is bound.
    let address = unsafe { device.get_buffer_device_address(&info) };
    Ok(Memory {
        id,
        class,
        len,
        memory: carve.memory,
        base: carve.base,
        buffer,
        address,
        mapping: carve.mapping,
        device_local: carve.device_local,
        spilled: carve.spilled,
    })
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

/// Ends a block: the consumer's hook where it was handed out and its hook has not run, then its
/// buffer. Its bytes stay in its allocation.
fn retire(core: &Core, memory: &Memory) {
    core.pool.unmap(memory.id);
    // SAFETY: no GPU work that uses the block is pending: a command buffer or upload that uses it
    // holds it until it retires, and the fills of the block-maker thread are waited for. The
    // consumer's mappings are gone.
    unsafe { core.device.destroy_buffer(memory.buffer, None) };
}

/// Frees an allocation, whose blocks are all retired. Freeing the memory also unmaps it.
fn free_allocation(core: &Core, allocation: Allocation) {
    // SAFETY: no buffer is bound to it any more, and no GPU work uses it.
    unsafe { core.device.free_memory(allocation.memory, None) };
    core.counters.fall(Gauge::Blocks);
    if allocation.spilled {
        core.counters.fall_by(Gauge::SpilledBytes, allocation.len);
    }
}

/// At teardown, after the block-maker thread is joined and every resource is freed: retires
/// `blocks`, the blocks the runtime's own places still hold, and the blocks the pool has back,
/// then frees every allocation.
pub(crate) fn free_all(core: &mut Core, blocks: Vec<Memory>) {
    let mut retired: Vec<Memory> = blocks;
    let mut allocations = Vec::new();
    let mut idle = [0; 2];
    {
        let state = (core.pool.state.get_mut()).unwrap_or_else(PoisonError::into_inner);
        for class in MemoryClass::ALL {
            let index = class.index();
            retired.append(&mut state.kept[index]);
            allocations.append(&mut state.allocations[index]);
            idle[index] = state.idle_bytes[index];
            state.idle_bytes[index] = 0;
        }
        retired.append(&mut state.returned);
    }
    for class in MemoryClass::ALL {
        core.counters
            .fall_by(class.idle_gauge(), idle[class.index()]);
    }
    for memory in &retired {
        retire(core, memory);
    }
    for allocation in allocations {
        free_allocation(core, allocation);
    }
}
