//! Memory: the blocks the consumer takes and releases, each a range of one of the device's
//! allocations; the pool that hands them out from the allocations' free ranges and frees idle
//! allocations; the allocations its own thread makes ahead of need, while a class's free bytes run
//! low; and how memory is made and freed. A block's bytes are undefined but for its zero mark
//! ([`BlockInfo::zeroed`]): the block-maker thread zeroes each Shared allocation it makes ahead of
//! need, and the free bytes blocks gave back, in steps, and a take answers how many of the
//! block's first bytes it zeroed. A buffer reads zero where
//! [`Device::new_buffer`](crate::Device::new_buffer) zeroes its range.

use crate::counters::{Counters, Gauge, Metric};
use crate::device::{lock, Core};
use crate::error::{Error, Limit};
use crate::floor;
use crate::places::Ranges;
use ash::vk;
use std::collections::HashSet;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed, Ordering::SeqCst};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

/// The length of each allocation the block-maker thread makes, and of an allocation a take makes
/// where no free range fits it, unless the take is longer: 16 MiB of either class, the unit.
///
/// A driver may hold the device's other calls (pipeline creation, submissions, other allocations)
/// for the whole of an allocation's `vkAllocateMemory`, and the hold grows with the bytes: on an
/// RTX 3090 (NVIDIA's Linux driver) about 0.25 ms per MiB of Shared memory and a third of that of
/// Private. The call alone, on an otherwise idle device, takes 2.9 to 4 ms for 16 MiB of Shared
/// memory on an RTX 2080 or 3090 and 0.55 ms of Private, against 0.3 ms or less for either on
/// RADV, Mesa's Intel driver and Intel's Windows driver for the UHD Graphics 630. So allocations
/// are short, and a take longer than one pays its own length once: the allocation it makes, a
/// long allocation, serves later takes of that length.
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

/// The free Shared bytes the block-maker thread zeroes in one step. A take that no other free
/// range fits waits for the step that runs, at most this many bytes, and finds the bytes zeroed
/// so far ([`BlockInfo::zeroed`]).
pub(crate) const ZERO_STEP: u64 = 256 << 10;

/// The idle age, in nanoseconds of [`host_time`](crate::host_time): 250 ms, one value for every
/// device. An allocation is idle once it holds no live block: its idle time starts when its last
/// live block comes back, and only a take that lands in it again ends it. Commits and submissions
/// play no part, so a program that keeps committing gives back what it stopped using. The
/// block-maker thread frees an allocation idle this long, after the consumer's unmap of it, one at
/// a time and between the allocations it makes, so a free never runs beside its own
/// `vkAllocateMemory`; a class's reserve ([`MAKE_BELOW`]) is not freed. A driver may hold a
/// submission made during a free for the rest of it (on the NVIDIA cards measured in
/// Thunderbolt enclosures, freeing host-visible memory takes about 1.1 ms per MiB):
/// [`Metrics::submissions_during_frees`](crate::Metrics::submissions_during_frees) counts them.
pub(crate) const IDLE_AGE_NANOS: u64 = 250_000_000;

/// The idle clock, the one clock for everything the runtime gives back: what became idle at
/// `since` and stayed idle is due at `now` ([`host_time`](crate::host_time)) once
/// [`IDLE_AGE_NANOS`] has passed (`None`); before then, the host time it becomes due.
pub(crate) fn idle_due(since: u64, now: u64) -> Option<u64> {
    let due = since.saturating_add(IDLE_AGE_NANOS);
    (now < due).then_some(due)
}

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
    /// this length, which later takes share, or of its own where it is longer, which only later
    /// takes of that length share.
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
/// device's life: each take is a new number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(u32);

impl BlockId {
    /// The number.
    pub fn get(self) -> u32 {
        self.0
    }
}

/// No Metal counterpart: an allocation's number, the `VkDeviceMemory` blocks are ranges of.
/// Numbers start at 1 and are never reused during the device's life.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AllocationId(u32);

impl AllocationId {
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
/// device is torn down. It is a range of one allocation, whose one `VkBuffer`, with every buffer
/// usage and a device address, spans it. Nothing in it is exportable.
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
    /// The allocation it is a range of, which [`BlockMappings::map`] was told of before this
    /// block went out.
    pub allocation: AllocationId,
    /// The offset of its first byte in that allocation.
    pub offset: u64,
    /// `gpuAddress` of the block's first byte: its allocation's device address plus
    /// [`offset`](Self::offset).
    pub gpu_address: u64,
    /// `contents` of the block's first byte: the host mapping of a Shared block, `None` for a
    /// Private one.
    pub contents: Option<NonNull<u8>>,
    /// Whether its memory type is device-local. A Private block is, unless the device-local type
    /// was out of memory when its allocation was made and it went to the next type its buffer
    /// takes (counted in [`Gauges::spilled_bytes`](crate::Gauges::spilled_bytes)); a Shared
    /// block is where the device's Shared type is.
    pub device_local: bool,
    /// The zero mark: how many of the block's first bytes the runtime itself zeroed and nothing
    /// held since, as [`Device::take_block`](crate::Device::take_block) found them. They read
    /// zero. The runtime zeroes Shared memory on the block-maker thread alone: an allocation it
    /// makes ahead of need, whole, before any take lands in it, and the free bytes blocks gave
    /// back, in steps of 256 KiB from the front of each allocation's first free bytes not zeroed.
    /// A take lands where its whole length is zeroed if any free range of that length is, so the
    /// mark is any length up to [`len`](Self::len). Nothing else counts: not the driver, the
    /// memory type or a new allocation. 0 for a Private block and for a block of an allocation a
    /// take made itself.
    ///
    /// Bytes past the mark are the consumer's to zero where Metal owes zeros. The mark is the
    /// take's: the consumer writes the bytes inside it only through a buffer it places over them
    /// ([`Device::new_buffer`](crate::Device::new_buffer)), whose zeroing skips the bytes the
    /// mark still covers.
    pub zeroed: u64,
}

// SAFETY: plain data. `contents` is an address; reading or writing through it is the consumer's
// own unsafe code, under the contract of `BlockMappings`.
unsafe impl Send for BlockInfo {}
// SAFETY: as above.
unsafe impl Sync for BlockInfo {}

/// No Metal counterpart: an allocation of the pool, as [`BlockMappings::map`] is told of it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct AllocationInfo {
    /// The allocation's number, which the blocks of it carry
    /// ([`BlockInfo::allocation`](crate::BlockInfo::allocation)).
    pub id: AllocationId,
    /// The storage mode of its blocks.
    pub class: MemoryClass,
    /// Its length: 16 MiB ([`MemoryClass::block_len`]), or a take's own length where that is
    /// longer.
    pub len: u64,
    /// The device address of its first byte.
    pub gpu_address: u64,
    /// The host mapping of its first byte, which lives as long as the allocation, for a Shared
    /// allocation; `None` for a Private one, which has no host mapping.
    pub contents: Option<NonNull<u8>>,
}

// SAFETY: plain data, as `BlockInfo`.
unsafe impl Send for AllocationInfo {}
// SAFETY: as above.
unsafe impl Sync for AllocationInfo {}

/// No Metal counterpart: the consumer's hooks around each allocation of the pool, for a consumer
/// that maps the allocations into a guest's address range itself.
///
/// The runtime calls [`map`](BlockMappings::map) once for every allocation of the pool it makes,
/// before any block of it goes out, and [`unmap`](BlockMappings::unmap) once for every allocation
/// `map` accepted, before the allocation is freed. The guest's range holds whole allocations,
/// their free ranges and the runtime's own staging blocks included; the consumer hands the guest
/// blocks alone. A block given back stays mapped with its allocation: a guest kernel that writes
/// a free range damages only its own later blocks' bytes. Imported blocks are not the pool's, and
/// neither hook runs for them.
pub trait BlockMappings: Send + Sync {
    /// The allocation `allocation` describes was made. True once the consumer has mapped it; false
    /// where it has no place for it (no guest address): the runtime then frees the allocation and
    /// hands out none of its blocks, and a take that made it returns
    /// [`Limit::AllocationNotMapped`](crate::Limit::AllocationNotMapped), counted in
    /// [`Metrics::allocation_map_refusals`](crate::Metrics::allocation_map_refusals).
    ///
    /// It runs where the allocation is made: on the block-maker thread for the allocations made
    /// ahead of need, on the thread in [`Device::new`](crate::Device::new) for each class's first,
    /// and inside [`Device::take_block`](crate::Device::take_block), on the consumer's thread, for
    /// a take that no free range fits. Never under the pool's lock. It makes no call into the
    /// runtime and drops no runtime handle.
    fn map(&self, allocation: &AllocationInfo) -> bool;

    /// Returns only once every mapping of `allocation` the consumer made is gone and none of its
    /// threads touches the allocation's memory. The memory stays mapped and valid until it
    /// returns, and is freed after.
    ///
    /// It runs once per allocation `map` accepted: on the block-maker thread before an idle
    /// allocation's free; in [`Device::quiesce`](crate::Device::quiesce), on the thread that calls
    /// it, for every allocation still mapped; or at teardown, on the thread that ends it, once the
    /// runtime's other threads are joined. Never under the pool's lock. It makes no call into the
    /// runtime and drops no runtime handle.
    fn unmap(&self, allocation: AllocationId);
}

/// A block: a range of an allocation, its allocation's `VkBuffer`, and its mapping.
#[derive(Clone, Copy)]
pub(crate) struct Memory {
    pub(crate) id: BlockId,
    pub(crate) class: MemoryClass,
    pub(crate) len: u64,
    /// Its allocation, and its first byte's offset there.
    allocation: AllocationId,
    memory: vk::DeviceMemory,
    base: u64,
    /// The allocation's `VkBuffer`, which spans the allocation ([`range`](Self::range)).
    buffer: vk::Buffer,
    /// The device address of the block's first byte.
    pub(crate) address: u64,
    pub(crate) mapping: Option<Mapping>,
    /// Its memory type is device-local.
    device_local: bool,
    /// A Private block in another memory type than the Private one, which was out of memory.
    spilled: bool,
    /// How many of its first bytes the runtime zeroed and nothing held since, at the take
    /// ([`BlockInfo::zeroed`]).
    zeroed: u64,
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

    /// The `VkBuffer` that commands reach the block's byte `offset` through, and that byte's offset
    /// in it: the allocation's buffer, which spans the allocation, at the block's base plus
    /// `offset`.
    pub(crate) fn range(&self, offset: u64) -> (vk::Buffer, u64) {
        (self.buffer, self.base + offset)
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
            allocation: self.allocation,
            offset: self.base,
            gpu_address: self.address,
            contents: self.mapping.map(Mapping::get),
            device_local: self.device_local,
            zeroed: self.zeroed,
        }
    }
}

/// A host address in a Shared allocation's mapping, which lives as long as the allocation: its
/// first byte, a block's or a buffer's.
#[derive(Clone, Copy)]
pub(crate) struct Mapping(NonNull<u8>);

// SAFETY: an address. The runtime writes through it only while it owns the bytes: `new_buffer`
// zeroing the range of the buffer it makes, and the block-maker thread zeroing an allocation
// before it joins the pool or free bytes that no take hands out meanwhile.
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
    /// The zero mark as it stands: the take's ([`BlockInfo::zeroed`]), lowered to the offset of
    /// each buffer placed in the block.
    mark: AtomicU64,
}

impl Block {
    pub(crate) fn new(memory: Memory, core: Arc<Core>) -> Block {
        let mark = AtomicU64::new(memory.zeroed);
        Block { memory, core, mark }
    }

    /// A buffer placed at `offset`: the bytes from there on are the consumer's to write, so the
    /// mark falls to `offset` where it was past it. Answers the mark before: the bytes below it
    /// read zero.
    pub(crate) fn place_at(&self, offset: u64) -> u64 {
        self.mark.fetch_min(offset, Relaxed)
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        self.core.pool.give_back(self.memory);
    }
}

/// One `VkDeviceMemory` of a class, the `VkBuffer` that spans it, and the ranges of it no block
/// holds.
struct Allocation {
    id: AllocationId,
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    /// The device address of its first byte.
    address: u64,
    len: u64,
    mapping: Option<Mapping>,
    device_local: bool,
    spilled: bool,
    /// Bytes no block holds.
    free: Ranges,
    /// Free bytes the runtime zeroed that no block held since: all of a Shared allocation made
    /// ahead of need, less each block carved from it, and the free bytes the block-maker thread
    /// zeroed since.
    clean: Ranges,
    /// Its blocks handed out and not yet given back to the pool.
    live: usize,
    /// When it became idle, in nanoseconds of [`host_time`](crate::host_time): when it was made,
    /// or when its last live block came back. The oldest idle allocation is freed first.
    idle_since: u64,
    /// A spilled allocation whose last block came back: no take lands in it, its bytes are not
    /// the class's free bytes, and the idle rule frees it next, so the next allocation tries
    /// device-local memory again.
    retiring: bool,
    /// A long allocation, longer than [`ALLOCATION_LEN`], that holds a block of a take after the
    /// one it was made for, until it is idle again
    /// ([`Gauges::shared_long_reused_bytes`](crate::Gauges::shared_long_reused_bytes)).
    reused: bool,
}

impl Allocation {
    /// Whether it is a unit allocation, of [`ALLOCATION_LEN`], which any take that fits shares,
    /// where a long one, made for a take longer than the unit, serves only a take of its own
    /// length.
    fn unit(&self) -> bool {
        self.len <= ALLOCATION_LEN
    }

    /// Whether a take of `len` bytes may land in it: a unit allocation for any take, a long one
    /// for a take of exactly its length.
    fn serves(&self, len: u64) -> bool {
        !self.retiring && (self.unit() || self.len == len)
    }
}

/// The pool: every allocation of each class, and the ranges of each that no block holds.
///
/// A take is the first free range of the class's allocations that fits it whose bytes are all
/// zeroed, else the first that fits, in the order the allocations were made; a long allocation
/// serves only takes of its own length. Where none fits, the take makes an allocation itself, of
/// [`ALLOCATION_LEN`] or of its own length where that is longer. A block given back is a free
/// range of its allocation at once. The block-maker thread ([`run`]) makes an allocation of a
/// class while the class's free bytes are fewer than [`MAKE_BELOW`], and zeroes the free Shared
/// bytes blocks gave back; a take waits for it only for a zero step whose bytes it needs.
///
/// The idle rule never frees a unit allocation that the class's free bytes need to stay at or
/// above [`MAKE_BELOW`], so it does not free what the block-maker thread would make again. A long
/// allocation is never kept so.
///
/// The idle rule runs on the block-maker thread as a block is taken or given back, and again when
/// the oldest idle allocation reaches [`IDLE_AGE_NANOS`]: it frees each idle allocation of that
/// age, one at a time, whatever the device's queues are doing.
pub(crate) struct Pool {
    state: Mutex<State>,
    /// Signalled whenever the state changes, for the block-maker thread and for takers waiting
    /// on it.
    changed: Condvar,
    mappings: Option<Arc<dyn BlockMappings>>,
    /// The device's counters: the idle bytes, the allocations made and the long reuses.
    counters: Counters,
    /// A block's length and its first byte's offset are multiples of this: [`GRANULE`], or the
    /// device's `bufferImageGranularity` where that is larger.
    granule: u64,
    /// The alignment of a buffer of the pool's usage, which Vulkan holds the same for every such
    /// buffer.
    alignment: u64,
    /// Whether the block-maker thread is in an idle allocation's `vkFreeMemory`.
    freeing: AtomicBool,
    /// The consumer's memsets of Shared bytes that run ([`Pool::fill`]): the block-maker thread
    /// starts no zero step while one does, so the two never share the memory bus.
    fills: AtomicUsize,
    /// The block-maker thread holds a zero step back for a memset that runs: the last memset to
    /// end wakes it.
    fill_held: AtomicBool,
    /// For the device tests: while set, the block-maker thread waits inside each idle
    /// allocation's free, with [`freeing`](Self::freeing) set, before it calls `vkFreeMemory`.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    hold_frees: Mutex<bool>,
    /// For the device tests: each idle allocation the block-maker thread freed, as when it became
    /// idle, when its free began and when it ended, in nanoseconds of
    /// [`host_time`](crate::host_time).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    frees: Mutex<Vec<(u64, u64, u64)>>,
}

/// What the idle rule asks of a class.
enum Rule {
    /// No idle allocation that the rule weighs.
    Keep,
    /// Idle allocations younger than [`IDLE_AGE_NANOS`]: the oldest reaches it at this
    /// [`host_time`](crate::host_time).
    KeepUntil(u64),
    /// The allocation to free next: a retiring one, else the oldest idle one of
    /// [`IDLE_AGE_NANOS`] or more.
    Free(vk::DeviceMemory),
}

/// The step of free Shared bytes the block-maker thread zeroes off the lock: `len` bytes from
/// `start` of the allocation `memory`, out of its free ranges until the step ends.
#[derive(Clone, Copy)]
struct Zeroing {
    memory: vk::DeviceMemory,
    start: u64,
    len: u64,
}

struct State {
    /// Per class, its allocations in the order they were made.
    allocations: [Vec<Allocation>; 2],
    /// Per class, the block-maker thread could not make the last allocation it tried. It tries
    /// again after the class's next take or give-back.
    failed: [bool; 2],
    stop: bool,
    next_id: u32,
    next_allocation: u32,
    /// The allocations [`BlockMappings::map`] accepted whose unmap has not run.
    mapped: HashSet<AllocationId>,
    /// The unmap hook has run for every allocation mapped ([`Pool::unmap_all`]); no hook runs
    /// again.
    unmapped: bool,
    /// Per class, the idle allocations' bytes the class's gauge shows.
    idle_bytes: [u64; 2],
    /// The step the block-maker thread is zeroing, off the lock.
    zeroing: Option<Zeroing>,
    /// Takes waiting for that step to end: the block-maker thread starts no step while one waits,
    /// so a take waits for one step at most.
    zero_waits: usize,
    /// For the device tests: how many more steps the block-maker thread may zero, any number for
    /// `None`.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    zero_budget: Option<u64>,
    /// For the device tests: the blocks given back.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    returned: HashSet<BlockId>,
}

/// A consumer memset of Shared bytes that runs ([`Pool::fill`]).
pub(crate) struct Fill<'a>(&'a Pool);

impl Drop for Fill<'_> {
    fn drop(&mut self) {
        let pool = self.0;
        // The block-maker thread sets `fill_held` before it reads `fills`, and this reads it after
        // the memset left `fills`: one of the two sees the other. The notify is under the lock,
        // so the thread is either before its read or waiting.
        if pool.fills.fetch_sub(1, SeqCst) == 1 && pool.fill_held.load(SeqCst) {
            let _state = pool.lock();
            pool.changed.notify_all();
        }
    }
}

thread_local! {
    /// Whether this thread makes allocations ahead of need: the block-maker thread ([`run`]), or
    /// the device's making while it makes the first allocations ([`stock`]). For the
    /// allocation-time metrics.
    static ON_MAKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// What a block is: a range of an allocation.
#[derive(Clone, Copy)]
struct Carve {
    allocation: AllocationId,
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    /// The device address of the range's first byte.
    address: u64,
    base: u64,
    mapping: Option<Mapping>,
    device_local: bool,
    spilled: bool,
    /// How many of the range's first bytes are zeroed ([`Allocation::clean`]).
    zeroed: u64,
}

impl Carve {
    /// Block `id` of `class` and `len` bytes, the range.
    fn block(self, id: BlockId, class: MemoryClass, len: u64) -> Memory {
        Memory {
            id,
            class,
            len,
            allocation: self.allocation,
            memory: self.memory,
            base: self.base,
            buffer: self.buffer,
            address: self.address,
            mapping: self.mapping,
            device_local: self.device_local,
            spilled: self.spilled,
            zeroed: self.zeroed,
        }
    }
}

impl Pool {
    /// The pool of a device whose `bufferImageGranularity` is `granularity`, and whose buffers of
    /// the pool's usage have the alignment `alignment`.
    pub(crate) fn new(
        mappings: Option<Arc<dyn BlockMappings>>,
        counters: Counters,
        granularity: u64,
        alignment: u64,
    ) -> Pool {
        Pool {
            state: Mutex::new(State {
                allocations: [Vec::new(), Vec::new()],
                failed: [false, false],
                stop: false,
                next_id: 1,
                next_allocation: 1,
                mapped: HashSet::new(),
                unmapped: false,
                idle_bytes: [0, 0],
                zeroing: None,
                zero_waits: 0,
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                zero_budget: None,
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                returned: HashSet::new(),
            }),
            changed: Condvar::new(),
            mappings,
            counters,
            granule: GRANULE.max(granularity.next_power_of_two()),
            alignment,
            freeing: AtomicBool::new(false),
            fills: AtomicUsize::new(0),
            fill_held: AtomicBool::new(false),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            hold_frees: Mutex::new(false),
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            frees: Mutex::new(Vec::new()),
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

    fn wait_for<'a>(
        &self,
        state: MutexGuard<'a, State>,
        timeout: Duration,
    ) -> MutexGuard<'a, State> {
        match self.changed.wait_timeout(state, timeout) {
            Ok((state, _)) => state,
            Err(poisoned) => poisoned.into_inner().0,
        }
    }

    /// Whether the block-maker thread is in an idle allocation's `vkFreeMemory`, which a driver
    /// may hold a submission made now for ([`Metrics::submissions_during_frees`]).
    ///
    /// [`Metrics::submissions_during_frees`]: crate::Metrics::submissions_during_frees
    pub(crate) fn freeing(&self) -> bool {
        self.freeing.load(Relaxed)
    }

    /// For the device tests: whether the class's free bytes are at least [`MAKE_BELOW`], so the
    /// block-maker thread makes nothing for it (an allocation it made is counted once its fill's
    /// submission completed).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn stocked(&self, class: MemoryClass) -> bool {
        self.lock().free(class) >= MAKE_BELOW
    }

    /// For the device tests: the allocations of both classes with a block in use, counted where
    /// the pool keeps them. An allocation the block-maker thread is making or freeing is not
    /// among them, where the `Blocks` gauge counts it: the gauge rises before the allocation joins
    /// the pool and falls after it left.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn in_use(&self) -> usize {
        let state = self.lock();
        (state.allocations.iter().flatten())
            .filter(|allocation| allocation.live > 0)
            .count()
    }

    /// For the device tests: the class's allocations, as each one's number, length, live blocks
    /// and free bytes, in the order they were made.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn allocations(&self, class: MemoryClass) -> Vec<(AllocationId, u64, usize, u64)> {
        let state = self.lock();
        (state.allocations[class.index()].iter())
            .map(|a| (a.id, a.len, a.live, a.free.bytes()))
            .collect()
    }

    /// For the device tests: the free bytes of `allocation` that are zeroed, where it is in the
    /// pool.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn clean(&self, allocation: AllocationId) -> Option<u64> {
        let state = self.lock();
        (state.allocations.iter().flatten())
            .find(|a| a.id == allocation)
            .map(|a| a.clean.bytes())
    }

    /// For the device tests: the block-maker thread zeroes at most `steps` more steps of free
    /// bytes, or any number for `None`, once the step that runs has ended.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn zero_steps(&self, steps: Option<u64>) {
        let mut state = self.lock();
        while state.zeroing.is_some() {
            state = self.wait(state);
        }
        state.zero_budget = steps;
        self.changed.notify_all();
    }

    /// For the device tests: holds each idle allocation's free, with [`freeing`](Self::freeing)
    /// set, while `hold` is true.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn hold_frees(&self, hold: bool) {
        *lock(&self.hold_frees) = hold;
    }

    /// For the device tests: each idle allocation the block-maker thread freed so far, as when
    /// it became idle, when its free began and when it ended ([`host_time`](crate::host_time)).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn frees(&self) -> Vec<(u64, u64, u64)> {
        lock(&self.frees).clone()
    }

    /// For the device tests: whether the pool has the block back.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn holds(&self, block: BlockId) -> bool {
        self.lock().returned.contains(&block)
    }

    /// The length and alignment of a block for a take of `len` bytes: `len` rounded up to the
    /// granule, at a multiple of that rounded up to a power of two (at most [`BASE_ALIGN`]), of
    /// the alignment of the pool's buffers and of the granule. No Vulkan call.
    ///
    /// Err: [`Limit::BlockLen`] past `max`, the device's longest allocation and buffer.
    fn shape(&self, len: u64, max: u64) -> Result<(u64, u64), Error> {
        let len = round(len, self.granule, max)?;
        Ok((len, base_align(len, self.alignment, self.granule)))
    }

    /// Takes a block of `class` of at least `len` bytes for `core`: the first free range that
    /// fits whose bytes are all zeroed, else the first that fits, in the order the allocations
    /// were made, a long allocation only for a take of its own length; with no Vulkan call. Where
    /// none fits and the block-maker thread is zeroing a step of the class's free bytes, it waits
    /// for that step and looks again. Else a range of an allocation made here, of
    /// [`ALLOCATION_LEN`] or the block's length where that is longer, which the consumer's
    /// [`BlockMappings::map`] maps first. A block's bytes are undefined past its zero mark: the
    /// first bytes of its range that are clean.
    ///
    /// Err: [`Limit::BlockLen`] past the device's longest allocation or buffer;
    /// [`Limit::BlockIds`]; what [`make_allocation`] returns; [`Limit::AllocationNotMapped`]
    /// where the consumer did not map the allocation the take made; [`Error::Quiesced`] once the
    /// pool is stopped.
    pub(crate) fn take(
        &self,
        core: &Core,
        class: MemoryClass,
        len: u64,
        max: u64,
    ) -> Result<Memory, Error> {
        let (len, align) = self.shape(len, max)?;
        let index = class.index();
        let mut state = self.lock();
        // A take is a pool event: the block-maker thread tries a failed allocation again.
        state.failed[index] = false;
        let id = number(&mut state)?;
        loop {
            if state.stop {
                return Err(Error::Quiesced);
            }
            if let Some((carve, idle_since, long)) = state.fit(class, len, align) {
                if let Some(since) = idle_since {
                    (self.counters).reused_idle(crate::host_time().saturating_sub(since));
                }
                if long {
                    (self.counters).long_reuse(class == MemoryClass::Shared);
                }
                let memory = carve.block(id, class, len);
                self.counters.carved(memory.zeroed);
                self.settle(&mut state, class);
                // The block-maker thread runs the idle rule on this take, and makes an allocation
                // where the class's free bytes ran low.
                self.changed.notify_all();
                return Ok(memory);
            }
            if class != MemoryClass::Shared || state.zeroing.is_none() {
                break;
            }
            // The block-maker thread zeroes a step of free bytes off the lock: the take waits for
            // that step, and the thread starts no other while it waits.
            state.zero_waits += 1;
            while state.zeroing.is_some() {
                state = self.wait(state);
            }
            state.zero_waits -= 1;
        }
        drop(state);
        let memory = self.own(core, class, len, align, id)?;
        let mut state = self.lock();
        self.settle(&mut state, class);
        self.changed.notify_all();
        Ok(memory)
    }

    /// Makes block `id` of `len` bytes at a multiple of `align` at the start of an allocation of
    /// its own, of [`ALLOCATION_LEN`] or `len` where that is longer, whose rest later takes share;
    /// the consumer maps the allocation first.
    fn own(
        &self,
        core: &Core,
        class: MemoryClass,
        len: u64,
        align: u64,
        id: BlockId,
    ) -> Result<Memory, Error> {
        let mut allocation = make_allocation(core, class, len.max(ALLOCATION_LEN))?;
        if !self.map(&allocation, class, true) {
            free_allocation(core, allocation);
            return Err(Error::Limit(Limit::AllocationNotMapped { class, len }));
        }
        let base = allocation.free.take(len, align).unwrap_or(0);
        let memory = allocation.carve(base, len).block(id, class, len);
        allocation.live = 1;
        self.lock().allocations[class.index()].push(allocation);
        Ok(memory)
    }

    /// The consumer's [`BlockMappings::map`] of `allocation`, made on a take's thread where
    /// `on_take`, off the pool's lock: true where it mapped the allocation or has no hook. An
    /// allocation mapped while a quiesce ran the unmaps has its own unmap at once.
    fn map(&self, allocation: &Allocation, class: MemoryClass, on_take: bool) -> bool {
        let Some(mappings) = &self.mappings else {
            return true;
        };
        if self.lock().unmapped {
            return true;
        }
        let info = allocation.info(class);
        if !mappings.map(&info) {
            self.counters.map_refused();
            return false;
        }
        self.counters.mapped(on_take);
        let mut state = self.lock();
        if state.unmapped {
            drop(state);
            mappings.unmap(info.id);
        } else {
            state.mapped.insert(info.id);
        }
        true
    }

    /// Runs the consumer's unmap of `allocation` where it was mapped and its unmap has not run,
    /// off the pool's lock.
    fn unmap(&self, allocation: AllocationId) {
        let mapped = self.lock().mapped.remove(&allocation);
        if let Some(mappings) = self.mappings.as_ref().filter(|_| mapped) {
            mappings.unmap(allocation);
        }
    }

    /// In [`Device::quiesce`](crate::Device::quiesce) and at teardown, once the runtime's other
    /// threads are stopped: calls the unmap hook for every allocation mapped whose unmap has not
    /// run, on this thread. No hook runs again: an allocation freed later is freed without it.
    pub(crate) fn unmap_all(&self) {
        let mut mapped: Vec<AllocationId> = {
            let mut state = self.lock();
            state.unmapped = true;
            state.mapped.drain().collect()
        };
        mapped.sort_unstable();
        if let Some(mappings) = &self.mappings {
            for allocation in mapped {
                mappings.unmap(allocation);
            }
        }
    }

    /// Sets the class's idle gauges: its idle allocations' bytes, when the oldest idle one the
    /// idle rule weighs became idle, its reserve's bytes, and the bytes of its long allocations
    /// that hold a block of a later take.
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
        let (weighed, reserve) = state.aging(class);
        let oldest = weighed.first().map_or(0, |&(since, _)| since);
        let shared = class == MemoryClass::Shared;
        (self.counters).idle_since(shared, oldest);
        (self.counters).reserve_bytes(shared, reserve);
        let reused: u64 = (state.allocations[index].iter())
            .filter(|allocation| allocation.reused)
            .map(|allocation| allocation.len)
            .sum();
        (self.counters).long_reused_bytes(shared, reused);
    }

    /// The idle rule for `class` at `now` ([`host_time`](crate::host_time)): its next allocation
    /// to free is a retiring one, a spilled one that emptied, so the next allocation tries
    /// device-local memory again; else the oldest idle one it weighs ([`State::aging`]), once it
    /// has been idle for [`IDLE_AGE_NANOS`].
    fn rule(&self, state: &State, class: MemoryClass, now: u64) -> Rule {
        let allocations = &state.allocations[class.index()];
        if let Some(retiring) = (allocations.iter()).find(|allocation| allocation.retiring) {
            return Rule::Free(retiring.memory);
        }
        match state.aging(class).0.first() {
            Some(&(since, memory)) => {
                idle_due(since, now).map_or(Rule::Free(memory), Rule::KeepUntil)
            }
            None => Rule::Keep,
        }
    }

    /// Takes back a block taken with [`take`](Self::take): its bytes are a free range of its
    /// allocation at once, not zeroed, as the consumer may have written any of them.
    pub(crate) fn give_back(&self, memory: Memory) {
        let class = memory.class;
        let mut state = self.lock();
        state.failed[class.index()] = false;
        if let Some(allocation) = state.find(class, memory.memory) {
            allocation.free.give(memory.base, memory.len);
            allocation.live -= 1;
            if allocation.live == 0 {
                // Its idle time starts now, however long ago the block was released.
                allocation.idle_since = crate::host_time();
                allocation.retiring = allocation.spilled;
                allocation.reused = false;
            }
        }
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        state.returned.insert(memory.id);
        self.settle(&mut state, class);
        self.changed.notify_all();
    }

    /// A memset of Shared bytes on the consumer's thread begins
    /// ([`Device::new_buffer`](crate::Device::new_buffer)'s zeroing): the block-maker thread
    /// starts no zero step until the returned guard drops, so the memset does not share the memory
    /// bus with one. A step that runs already ends first: at most [`ZERO_STEP`] bytes.
    pub(crate) fn fill(&self) -> Fill<'_> {
        self.fills.fetch_add(1, SeqCst);
        Fill(self)
    }

    /// Stops the block-maker thread once it finishes what it is doing.
    pub(crate) fn stop(&self) {
        self.lock().stop = true;
        self.changed.notify_all();
    }
}

impl State {
    /// The bytes of the free ranges of the class's allocations, but those retiring.
    fn free(&self, class: MemoryClass) -> u64 {
        (self.allocations[class.index()].iter())
            .filter(|allocation| !allocation.retiring)
            .map(|allocation| allocation.free.bytes())
            .sum()
    }

    /// The allocation of `class` whose memory is `memory`.
    fn find(&mut self, class: MemoryClass, memory: vk::DeviceMemory) -> Option<&mut Allocation> {
        (self.allocations[class.index()].iter_mut()).find(|allocation| allocation.memory == memory)
    }

    /// Reserves `len` bytes at a multiple of `align` in the class's allocations that serve a take
    /// of `len` ([`Allocation::serves`]), counting a block live there: the first such range whose
    /// bytes are all clean, else the first that is free, in the order the allocations were made.
    /// With the range, when the allocation became idle where it was, and whether it is a long
    /// allocation.
    fn fit(
        &mut self,
        class: MemoryClass,
        len: u64,
        align: u64,
    ) -> Option<(Carve, Option<u64>, bool)> {
        let allocations = &mut self.allocations[class.index()];
        // Clean bytes are free bytes: a range of them is free whole.
        let clean = (allocations.iter().enumerate())
            .filter(|(_, allocation)| allocation.serves(len))
            .find_map(|(at, allocation)| Some((at, allocation.clean.find(len, align)?)));
        let (at, base) = clean.or_else(|| {
            (allocations.iter().enumerate())
                .filter(|(_, allocation)| allocation.serves(len))
                .find_map(|(at, allocation)| Some((at, allocation.free.find(len, align)?)))
        })?;
        let allocation = &mut allocations[at];
        allocation.free.cut(base, len);
        let idle_since = (allocation.live == 0).then_some(allocation.idle_since);
        allocation.live += 1;
        let long = !allocation.unit();
        if long {
            allocation.reused = true;
        }
        Some((allocation.carve(base, len), idle_since, long))
    }

    /// The class's idle allocations the idle rule weighs, as when each became idle and its memory,
    /// the oldest first, and the bytes of its reserve. The newest idle unit allocations, of
    /// [`ALLOCATION_LEN`], with free bytes that the class's free bytes need to reach [`MAKE_BELOW`]
    /// are its reserve, made ahead of need: the rule neither weighs nor frees them. A long
    /// allocation, which a take longer than the unit made for itself, is never the reserve: the
    /// rule frees it at its idle age, so it is not held for the device's life.
    fn aging(&self, class: MemoryClass) -> (Vec<(u64, vk::DeviceMemory)>, u64) {
        let allocations = &self.allocations[class.index()];
        let mut free: u64 = (allocations.iter())
            .filter(|allocation| allocation.live > 0 && !allocation.retiring)
            .map(|allocation| allocation.free.bytes())
            .sum();
        let mut idle: Vec<(u64, vk::DeviceMemory, u64, u64)> = (allocations.iter())
            .filter(|allocation| allocation.live == 0 && !allocation.retiring)
            .map(|allocation| {
                let bytes = allocation.free.bytes();
                (
                    allocation.idle_since,
                    allocation.memory,
                    bytes,
                    allocation.len,
                )
            })
            .collect();
        idle.sort_unstable_by_key(|&(since, ..)| std::cmp::Reverse(since));
        let mut reserved = 0;
        idle.retain(|&(_, _, bytes, len)| {
            let reserve = free < MAKE_BELOW && bytes > 0 && len <= ALLOCATION_LEN;
            if reserve {
                free += bytes;
                reserved += len;
            }
            !reserve
        });
        idle.reverse();
        let weighed: Vec<(u64, vk::DeviceMemory)> = (idle.into_iter())
            .map(|(since, memory, ..)| (since, memory))
            .collect();
        (weighed, reserved)
    }

    /// The next step of zeroing the free Shared bytes: the first free bytes not zeroed, in the
    /// order the allocations were made, none in a retiring one; at most [`ZERO_STEP`] of them.
    /// None while a take waits for a step.
    fn zero_step(&self) -> Option<(Zeroing, Mapping)> {
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        if self.zero_budget == Some(0) {
            return None;
        }
        if self.zero_waits > 0 {
            return None;
        }
        let allocations = &self.allocations[MemoryClass::Shared.index()];
        allocations.iter().find_map(|allocation| {
            let mapping = allocation.mapping.filter(|_| !allocation.retiring)?;
            let (start, end) = allocation.free.first_outside(&allocation.clean)?;
            let len = (end - start).min(ZERO_STEP);
            // SAFETY: `start` is inside the allocation, which the mapping spans.
            let at = Mapping(unsafe { mapping.at(start) });
            let step = Zeroing {
                memory: allocation.memory,
                start,
                len,
            };
            Some((step, at))
        })
    }

    /// Takes the allocation of `class` whose memory is `memory` out of the pool, for the caller to
    /// free.
    fn take_out(&mut self, class: MemoryClass, memory: vk::DeviceMemory) -> Option<Allocation> {
        let index = class.index();
        let at =
            (self.allocations[index].iter()).position(|allocation| allocation.memory == memory)?;
        Some(self.allocations[index].remove(at))
    }
}

impl Allocation {
    /// What a block of `len` bytes at `base`, a range taken from the free bytes, is: the range's
    /// zeroed first bytes leave the clean ones, now the block's.
    fn carve(&mut self, base: u64, len: u64) -> Carve {
        let zeroed = self.clean.run_at(base, len);
        self.clean.cut(base, len);
        Carve {
            allocation: self.id,
            memory: self.memory,
            buffer: self.buffer,
            address: self.address + base,
            base,
            // SAFETY: `base` is inside the allocation, which the mapping spans.
            mapping: self
                .mapping
                .map(|mapping| Mapping(unsafe { mapping.at(base) })),
            device_local: self.device_local,
            spilled: self.spilled,
            zeroed,
        }
    }

    /// Zeroes a Shared allocation made ahead of need, through its mapping, before it joins the
    /// pool: its bytes are all clean. A Private one is left as it is.
    fn zero_ahead(&mut self, counters: &Counters) {
        let Some(mapping) = self.mapping else {
            return;
        };
        let start = crate::host_time();
        // SAFETY: the mapping spans the allocation's `len` bytes, and no block holds any of them:
        // the allocation is not in the pool yet.
        unsafe { mapping.get().as_ptr().write_bytes(0, self.len as usize) };
        counters.zeroed_ahead(crate::host_time().saturating_sub(start));
        self.clean = Ranges::new(self.len);
    }

    /// The allocation as the consumer's hooks are told of it.
    fn info(&self, class: MemoryClass) -> AllocationInfo {
        AllocationInfo {
            id: self.id,
            class,
            len: self.len,
            gpu_address: self.address,
            contents: self.mapping.map(Mapping::get),
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

/// The alignment of a buffer of the pool's usage on `device`, the same for every such buffer.
pub(crate) fn buffer_alignment(device: &ash::Device) -> u64 {
    let buffer = vk::BufferCreateInfo::default()
        .size(GRANULE)
        .usage(floor::BUFFER_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let of = vk::DeviceBufferMemoryRequirements::default().create_info(&buffer);
    let mut needs = vk::MemoryRequirements2::default();
    // SAFETY: the device is live, and the create info valid.
    unsafe { device.get_device_buffer_memory_requirements(&of, &mut needs) };
    needs.memory_requirements.alignment
}

/// Makes an allocation ahead of need of `class`, zeroing a Shared one whole, and has the consumer
/// map it: the allocation, or `None` where it was not made or not mapped.
fn ahead(core: &Core, class: MemoryClass) -> Option<Allocation> {
    let mut allocation = make_allocation(core, class, ALLOCATION_LEN).ok()?;
    allocation.zero_ahead(&core.counters);
    if !core.pool.map(&allocation, class, false) {
        free_allocation(core, allocation);
        return None;
    }
    Some(allocation)
}

/// In the device's making, before the block-maker thread starts: makes each class's first
/// allocation, zeroing the Shared one, and has the consumer map each, so the consumer's first
/// blocks of each class are ranges of it as the device is ready, and nothing is allocated beside
/// the consumer's first calls. Where one is not made, the block-maker thread tries again after
/// the class's next take or give-back.
pub(crate) fn stock(core: &Core) {
    // Made ahead of need, as the block-maker thread's are: counted with them.
    ON_MAKER.set(true);
    for class in MemoryClass::ALL {
        let made = ahead(core, class);
        let mut state = core.pool.lock();
        match made {
            Some(allocation) => state.allocations[class.index()].push(allocation),
            None => state.failed[class.index()] = true,
        }
        core.pool.settle(&mut state, class);
    }
    ON_MAKER.set(false);
}

/// The block-maker thread, until the device's teardown stops it. Off every caller's path, in
/// order: it makes an allocation of a class whose free bytes are fewer than [`MAKE_BELOW`],
/// zeroing a Shared one whole and having the consumer map it before it joins the pool; it frees
/// each allocation idle for [`IDLE_AGE_NANOS`], after the consumer's unmap of it, one at a time;
/// and last it zeroes the free Shared bytes blocks gave back, the first of the allocations in the
/// order they were made, [`ZERO_STEP`] bytes at a time, taking up its other work again between
/// two steps. It sleeps until the next change, or until the oldest idle allocation reaches the
/// idle age.
pub(crate) fn run(core: Arc<Core>) {
    ON_MAKER.set(true);
    let pool = &core.pool;
    let mut state = pool.lock();
    loop {
        if state.stop {
            return;
        }
        let wanted = (MemoryClass::ALL.into_iter())
            .find(|&class| !state.failed[class.index()] && state.free(class) < MAKE_BELOW);
        if let Some(class) = wanted {
            drop(state);
            let made = ahead(&core, class);
            state = pool.lock();
            match made {
                // Idle from now: the reserve, which the idle rule does not weigh.
                Some(allocation) => state.allocations[class.index()].push(allocation),
                None => state.failed[class.index()] = true,
            }
            pool.settle(&mut state, class);
            pool.changed.notify_all();
            continue;
        }
        // The idle rule, both classes: the next allocation to free.
        let now = crate::host_time();
        let mut wake: Option<Duration> = None;
        let mut gone = None;
        for class in MemoryClass::ALL {
            let due = match pool.rule(&state, class, now) {
                Rule::Free(memory) => {
                    gone = state.take_out(class, memory);
                    pool.settle(&mut state, class);
                    break;
                }
                Rule::KeepUntil(at) => Duration::from_nanos(at.saturating_sub(now)),
                Rule::Keep => continue,
            };
            wake = Some(wake.map_or(due, |wake| wake.min(due)));
        }
        if let Some(allocation) = gone {
            pool.changed.notify_all();
            drop(state);
            free_idle(&core, allocation);
            state = pool.lock();
            continue;
        }
        // Last, the free Shared bytes not zeroed, a step at a time, off the lock and out of the
        // free ranges meanwhile; none while a take waits for a step, and none while a consumer
        // memset runs ([`Pool::fill`]): the step waits for the memset's end.
        let next = state.zero_step();
        pool.fill_held.store(next.is_some(), SeqCst);
        let held = next.is_some() && pool.fills.load(SeqCst) > 0;
        if held {
            core.counters.zero_step_held();
        }
        if let Some((step, at)) = next.filter(|_| !held) {
            pool.fill_held.store(false, Relaxed);
            if let Some(allocation) = state.find(MemoryClass::Shared, step.memory) {
                allocation.free.cut(step.start, step.len);
            }
            state.zeroing = Some(step);
            drop(state);
            let start = crate::host_time();
            // SAFETY: `at` is `step.len` bytes inside a Shared allocation's mapping, which lives
            // while the allocation is in the pool: only this thread frees it, and no take hands
            // the bytes out while they are out of the free ranges.
            unsafe { at.get().as_ptr().write_bytes(0, step.len as usize) };
            core.counters
                .zeroed_free(step.len, crate::host_time().saturating_sub(start));
            state = pool.lock();
            if let Some(allocation) = state.find(MemoryClass::Shared, step.memory) {
                allocation.free.give(step.start, step.len);
                allocation.clean.give(step.start, step.len);
            }
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            if let Some(steps) = state.zero_budget.as_mut() {
                *steps = steps.saturating_sub(1);
            }
            state.zeroing = None;
            pool.changed.notify_all();
            continue;
        }
        state = match wake {
            Some(timeout) => pool.wait_for(state, timeout),
            None => pool.wait(state),
        };
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

/// Makes an allocation of `len` bytes of `class` with device addresses, its number, the
/// `VkBuffer` with every buffer usage that spans it and that buffer's address, mapped whole if
/// Shared, its free ranges all of it. Its bytes are undefined: nothing writes them, and a Shared
/// allocation's pages become resident as they are first written.
fn make_allocation(core: &Core, class: MemoryClass, len: u64) -> Result<Allocation, Error> {
    let device = &core.device;
    let id = {
        let mut state = core.pool.lock();
        let id = state.next_allocation;
        state.next_allocation = id.checked_add(1).ok_or(Error::Limit(Limit::BlockIds))?;
        AllocationId(id)
    };
    let info = vk::BufferCreateInfo::default()
        .size(len)
        .usage(floor::BUFFER_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: the device is live, and the create info valid: `len` is at most its
    // `maxBufferSize`.
    let buffer =
        unsafe { device.create_buffer(&info, None) }.map_err(Error::vulkan("vkCreateBuffer"))?;
    // SAFETY: the buffer is the device's.
    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size.max(len))
        .memory_type_index(core.memory_type(class))
        .push_next(&mut flags);
    let bits = requirements.memory_type_bits;
    // SAFETY: the floor chose the memory type for buffers of this usage on this device, and
    // `bufferDeviceAddress` is enabled; a spill takes a type the buffer's requirements allow.
    let (memory, spilled) = match unsafe { allocate_spilling(core, class, bits, &allocate_info) } {
        Ok(made) => made,
        Err(error) => {
            // SAFETY: the buffer is unused.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(error);
        }
    };
    let mut allocation = Allocation {
        id,
        memory,
        buffer,
        address: 0,
        len,
        mapping: None,
        device_local: !spilled && core.device_local & (1 << core.memory_type(class)) != 0,
        spilled,
        free: Ranges::new(len),
        clean: Ranges::empty(),
        live: 0,
        idle_since: crate::host_time(),
        retiring: false,
        reused: false,
    };
    core.counters.count(Metric::BlocksMade);
    core.counters.allocated(len);
    if spilled {
        core.counters.rise_by(Gauge::SpilledBytes, len);
    }
    // SAFETY: the memory is of a type the buffer's requirements allow, at least their size, and
    // nothing else is bound to the buffer.
    if let Err(result) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        free_allocation(core, allocation);
        return Err(Error::Vulkan {
            call: "vkBindBufferMemory",
            result,
        });
    }
    let of = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    // SAFETY: the buffer has device-address usage and is bound.
    allocation.address = unsafe { device.get_buffer_device_address(&of) };
    if class == MemoryClass::Shared {
        // SAFETY: the memory type is host-visible, and the memory is not mapped.
        let mapped = unsafe {
            device.map_memory(
                allocation.memory,
                0,
                vk::WHOLE_SIZE,
                vk::MemoryMapFlags::empty(),
            )
        };
        match mapped {
            Ok(pointer) => allocation.mapping = NonNull::new(pointer.cast::<u8>()).map(Mapping),
            Err(result) => {
                free_allocation(core, allocation);
                return Err(Error::Vulkan {
                    call: "vkMapMemory",
                    result,
                });
            }
        }
    }
    Ok(allocation)
}

/// On the block-maker thread: frees an allocation the idle rule took out, after the consumer's
/// unmap of it, and counts its idle time. The submissions made during its `vkFreeMemory` are
/// counted.
fn free_idle(core: &Core, allocation: Allocation) {
    core.pool.unmap(allocation.id);
    // The bytes zeroed for a take that never came.
    (core.counters).zeroed_freed(allocation.clean.bytes());
    let pool = &core.pool;
    let since = allocation.idle_since;
    let began = crate::host_time();
    core.counters.freed_idle(began.saturating_sub(since));
    pool.freeing.store(true, Relaxed);
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    while *lock(&pool.hold_frees) {
        std::thread::sleep(Duration::from_millis(1));
    }
    free_allocation(core, allocation);
    pool.freeing.store(false, Relaxed);
    core.counters.count(Metric::AllocationsFreed);
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    lock(&pool.frees).push((since, began, crate::host_time()));
}

/// Frees an allocation and its buffer, whose blocks are all given back or gone with the device,
/// and whose unmap has run. Freeing the memory also unmaps the runtime's own mapping of it.
fn free_allocation(core: &Core, allocation: Allocation) {
    // SAFETY: no GPU work uses the buffer or the memory: a command buffer or upload that uses a
    // block holds it until it retires.
    unsafe {
        core.device.destroy_buffer(allocation.buffer, None);
        core.device.free_memory(allocation.memory, None);
    }
    core.counters.fall(Gauge::Blocks);
    if allocation.spilled {
        core.counters.fall_by(Gauge::SpilledBytes, allocation.len);
    }
}

/// At teardown, after the block-maker thread is joined and every resource is freed: frees every
/// allocation, each after the consumer's unmap of it where it is mapped.
pub(crate) fn free_all(core: &mut Core) {
    let mut allocations = Vec::new();
    let mut idle = [0; 2];
    {
        let state = (core.pool.state.get_mut()).unwrap_or_else(PoisonError::into_inner);
        for class in MemoryClass::ALL {
            let index = class.index();
            allocations.append(&mut state.allocations[index]);
            idle[index] = state.idle_bytes[index];
            state.idle_bytes[index] = 0;
        }
    }
    for class in MemoryClass::ALL {
        let shared = class == MemoryClass::Shared;
        core.counters
            .fall_by(class.idle_gauge(), idle[class.index()]);
        (core.counters).idle_since(shared, 0);
        (core.counters).reserve_bytes(shared, 0);
        (core.counters).long_reused_bytes(shared, 0);
    }
    for allocation in allocations {
        core.pool.unmap(allocation.id);
        free_allocation(core, allocation);
    }
}
