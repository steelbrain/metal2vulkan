//! Memory: the blocks the consumer takes and releases, each a range of one of the device's
//! allocations; the pool that hands them out, keeps them, and frees idle allocations; the
//! allocations its own thread makes ahead of need, while a class's free bytes run low; and how
//! memory is made and freed. A block's bytes are undefined but for its zero mark
//! ([`BlockInfo::zeroed`]): the block-maker thread zeroes each Shared allocation it makes ahead of
//! need, and each Shared block given back and kept, from its first byte in steps, and a take
//! answers how many of the block's first bytes it zeroed. A buffer reads zero where
//! [`Device::new_buffer`](crate::Device::new_buffer) zeroes its range.

use crate::counters::{Counters, Gauge, Metric};
use crate::device::{lock, Core};
use crate::error::{Error, Limit};
use crate::floor;
use crate::places::Ranges;
use ash::vk;
use std::collections::{HashSet, VecDeque};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed, Ordering::SeqCst};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

/// The length of each allocation the block-maker thread makes, and of an allocation a take makes
/// where no free range fits it, unless the take is longer: 16 MiB of either class.
///
/// A driver may hold the device's other calls (pipeline creation, submissions, other allocations)
/// for the whole of an allocation's `vkAllocateMemory`, and the hold grows with the bytes: on an
/// RTX 3090 (NVIDIA's Linux driver) about 0.25 ms per MiB of Shared memory and a third of that of
/// Private. The call alone, on an otherwise idle device, takes 2.9 to 4 ms for 16 MiB of Shared
/// memory on an RTX 2080 or 3090 and 0.55 ms of Private, against 0.3 ms or less for either on
/// RADV, Mesa's Intel driver and Intel's Windows driver for the UHD Graphics 630. So allocations
/// are short, and a take longer than one pays its own length once: the allocation it makes serves
/// later takes as free ranges.
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

/// The bytes of a kept Shared block the block-maker thread zeroes in one step. A take of the
/// block waits for the step that runs, at most this many bytes, and finds the bytes zeroed so far
/// ([`BlockInfo::zeroed`]).
pub(crate) const ZERO_STEP: u64 = 256 << 10;

/// The idle age, in nanoseconds of [`host_time`](crate::host_time): 250 ms, one value for every
/// device. An allocation is idle once it holds no live block: its idle time starts when its last
/// live block comes back, and only a take that lands in it again ends it. Commits and submissions
/// play no part, so a program that keeps committing gives back what it stopped using. The
/// block-maker thread frees an allocation idle this long, after the hook of each block kept in
/// it, one at a time and between the allocations it makes, so a free never runs beside its own
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
    /// The zero mark: how many of the block's first bytes the runtime itself zeroed and nothing
    /// held since, as [`Device::take_block`](crate::Device::take_block) found them. They read
    /// zero. The runtime zeroes from a block's first byte only, on the block-maker thread: a
    /// Shared allocation it makes ahead of need, whole, before any take lands in it, and a
    /// Shared block given back and kept, in steps of 256 KiB while it waits for a take of its
    /// length. A take between two steps finds the bytes zeroed so far, so the mark is any length
    /// up to [`len`](Self::len). Nothing else counts: not the driver, the memory type or a new
    /// allocation. 0 for a Private block and for a block of an allocation a take made itself.
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

/// No Metal counterpart: the consumer's hook before the runtime frees memory it handed out, or
/// hands it out again in another block.
///
/// The runtime calls [`unmap`](BlockMappings::unmap) once for every block
/// [`Device::take_block`](crate::Device::take_block) returned, before the block's bytes are freed
/// or become part of another block: when the pool gives a kept block's bytes back to its
/// allocation for a take of another length, when it frees an idle allocation, when a block given
/// back was in a spilled allocation, and at teardown for every block still alive. A block
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
    /// How many of its first bytes the runtime zeroed and nothing held since: at a take, the
    /// take's mark ([`BlockInfo::zeroed`]); kept, the bytes the block-maker thread zeroed so far.
    /// 0 once it is given back, as the consumer may have written any byte of it.
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
// before it joins the pool or a kept block that no take hands out meanwhile.
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

/// One `VkDeviceMemory` of a class and the ranges of it no block holds.
struct Allocation {
    memory: vk::DeviceMemory,
    len: u64,
    mapping: Option<Mapping>,
    device_local: bool,
    spilled: bool,
    /// Bytes no block holds, handed out or kept.
    free: Ranges,
    /// Free bytes the runtime zeroed that no block held since: all of a Shared allocation made
    /// ahead of need, less each block carved from it, and the zeroed bytes of a kept block given
    /// back to it.
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
/// The idle rule never frees a unit allocation, one of [`ALLOCATION_LEN`], that the class's free
/// bytes need to stay at or above [`MAKE_BELOW`], so it does not free what the block-maker thread
/// would make again. An allocation a longer take made for itself is never kept so.
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
    /// The device's counters: the idle bytes, the allocations made and the blocks taken again.
    counters: Counters,
    /// A block's length and its first byte's offset are multiples of this: [`GRANULE`], or the
    /// device's `bufferImageGranularity` where that is larger.
    granule: u64,
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

struct State {
    /// Per class, its allocations in the order they were made.
    allocations: [Vec<Allocation>; 2],
    /// Per class, blocks given back, the oldest first, with their number, buffer, address and
    /// mapping, for a take of their length. Their bytes stay out of their allocation's free
    /// ranges until the block-maker thread gives them back there, after the consumer's hook.
    kept: [Vec<Memory>; 2],
    /// Per class, a take asked the block-maker thread for the class's kept blocks' bytes.
    unkeep: [bool; 2],
    /// Blocks given back, for the block-maker thread, in the order they came back.
    returned: VecDeque<Memory>,
    /// Per class, the block-maker thread could not make the last allocation it tried. It tries
    /// again after the class's next take or give-back.
    failed: [bool; 2],
    stop: bool,
    next_id: u32,
    /// The blocks `take_block` returned whose hook has not run: running it removes one.
    handed: HashSet<BlockId>,
    /// The hook has run for every block handed out ([`Pool::unmap_all`]); it never runs again.
    unmapped: bool,
    /// Per class, the idle allocations' bytes the class's gauge shows.
    idle_bytes: [u64; 2],
    /// The kept Shared block whose bytes the block-maker thread is zeroing a step of, off the
    /// lock: no take hands it out until the step ends.
    zeroing: Option<BlockId>,
    /// Takes waiting for a step to end, for the block it zeroes: the block-maker thread starts no
    /// step while one waits, so a take waits for one step at most.
    zero_waits: usize,
    /// For the device tests: how many more steps the block-maker thread may zero, any number for
    /// `None`.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    zero_budget: Option<u64>,
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

/// What a block's `VkBuffer` is bound to: a range of an allocation.
#[derive(Clone, Copy)]
struct Carve {
    memory: vk::DeviceMemory,
    base: u64,
    mapping: Option<Mapping>,
    device_local: bool,
    spilled: bool,
    /// How many of the range's first bytes are zeroed ([`Allocation::clean`]).
    zeroed: u64,
}

impl Pool {
    /// The pool of a device whose `bufferImageGranularity` is `granularity`.
    pub(crate) fn new(
        mappings: Option<Arc<dyn BlockMappings>>,
        counters: Counters,
        granularity: u64,
    ) -> Pool {
        Pool {
            state: Mutex::new(State {
                allocations: [Vec::new(), Vec::new()],
                kept: [Vec::new(), Vec::new()],
                unkeep: [false, false],
                returned: VecDeque::new(),
                failed: [false, false],
                stop: false,
                next_id: 1,
                handed: HashSet::new(),
                unmapped: false,
                idle_bytes: [0, 0],
                zeroing: None,
                zero_waits: 0,
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                zero_budget: None,
            }),
            changed: Condvar::new(),
            mappings,
            counters,
            granule: GRANULE.max(granularity.next_power_of_two()),
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

    /// For the device tests: the class's blocks given back and kept.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn kept(&self, class: MemoryClass) -> usize {
        self.lock().kept[class.index()].len()
    }

    /// For the device tests: the class's kept blocks, as each one's number and how many of its
    /// first bytes the block-maker thread zeroed so far.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn kept_zeroed(&self, class: MemoryClass) -> Vec<(BlockId, u64)> {
        let state = self.lock();
        (state.kept[class.index()].iter())
            .map(|memory| (memory.id, memory.zeroed))
            .collect()
    }

    /// For the device tests: the block-maker thread zeroes at most `steps` more steps of kept
    /// blocks, or any number for `None`, once the step that runs has ended.
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

    /// Takes a block of `class` of at least `len` bytes for `core`: a kept block of its length, or
    /// one given back that the block-maker thread would keep and has not kept yet, with its
    /// number, buffer, address and mapping; else a free range, the first that fits in the
    /// order the allocations were made, with a new number and buffer; else a range of an
    /// allocation made here, of [`ALLOCATION_LEN`] or the block's length where that is longer.
    /// Where the class keeps blocks of other lengths, that take also asks the block-maker thread to
    /// give their bytes back; and where its allocation fails, it waits for them, and takes a free
    /// range of them where one fits. A block's bytes are undefined past its zero mark: a kept
    /// block's bytes the block-maker thread zeroed so far (a take of one it is zeroing a step of
    /// waits for that step), a free range's first bytes that are clean, none of one given back
    /// and not yet kept.
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
                let id = state.kept[index][at].id;
                if state.zeroing == Some(id) {
                    // The block-maker thread zeroes a step of it: the take waits for that step,
                    // and the thread starts no other while it waits.
                    state.zero_waits += 1;
                    while state.zeroing == Some(id) {
                        state = self.wait(state);
                    }
                    state.zero_waits -= 1;
                    continue;
                }
                let memory = state.kept[index].remove(at);
                self.counters.zeroed_taken(memory.zeroed);
                self.counters.count(Metric::BlocksReused);
                self.hold(&mut state, &memory);
                self.settle(&mut state, class);
                self.changed.notify_all();
                return Ok(memory);
            }
            // A block given back that the block-maker thread has not kept yet, which it would
            // keep: taken as it is, its allocation in use throughout. The thread can be inside a
            // driver's allocation or free for milliseconds, and a take that allocated beside the
            // block would leave its allocation idle, to be freed.
            let returned = (state.returned.iter())
                .rposition(|memory| memory.class == class && memory.len == len && !memory.spilled);
            if let Some(memory) = returned.and_then(|at| state.returned.remove(at)) {
                self.counters.count(Metric::BlocksReused);
                return Ok(memory);
            }
            if let Some((carve, idle_since)) = state.fit(class, len, align) {
                if let Some(since) = idle_since {
                    (self.counters).reused_idle(crate::host_time().saturating_sub(since));
                }
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
        self.counters.zeroed_taken(memory.zeroed);
        let mut state = self.lock();
        self.settle(&mut state, class);
        // The block-maker thread runs the idle rule on this take, and makes an allocation where
        // the class's free bytes ran low.
        self.changed.notify_all();
        Ok(memory)
    }

    /// Makes block `id` of `len` bytes at `carve`, a range reserved for it; where that fails, the
    /// range goes back.
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
            if let Some(allocation) = state.find(class, carve.memory) {
                allocation.free.give(carve.base, len);
                if carve.zeroed > 0 {
                    allocation.clean.give(carve.base, carve.zeroed);
                }
                allocation.live -= 1;
                if allocation.live == 0 {
                    allocation.idle_since = crate::host_time();
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
        let carve = allocation.carve(base, len);
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

    /// Counts a kept block taken again as live in its allocation, and a take of an idle
    /// allocation as a reuse after its idle time.
    fn hold(&self, state: &mut State, memory: &Memory) {
        if let Some(allocation) = state.find(memory.class, memory.memory) {
            if allocation.live == 0 {
                let idle = crate::host_time().saturating_sub(allocation.idle_since);
                self.counters.reused_idle(idle);
            }
            allocation.live += 1;
        }
    }

    /// Sets the class's idle gauges: its idle allocations' bytes, when the oldest idle one the
    /// idle rule weighs became idle, and its reserve's bytes.
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
        (self.counters).idle_since(class == MemoryClass::Shared, oldest);
        (self.counters).reserve_bytes(class == MemoryClass::Shared, reserve);
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

    /// Takes back a block taken with [`take`](Self::take), for the block-maker thread to keep, or
    /// give back to its allocation.
    pub(crate) fn give_back(&self, mut memory: Memory) {
        // The consumer may have written any byte of it.
        memory.zeroed = 0;
        self.lock().returned.push_back(memory);
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

    /// For the device tests: whether the pool has the block back, kept or still to be kept.
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

    /// Reserves `len` bytes at a multiple of `align` in the first allocation of `class` with such
    /// a free range, counting a block live there; none in a retiring one. With the range, when
    /// the allocation became idle where it was.
    fn fit(&mut self, class: MemoryClass, len: u64, align: u64) -> Option<(Carve, Option<u64>)> {
        (self.allocations[class.index()].iter_mut()).find_map(|allocation| {
            if allocation.retiring {
                return None;
            }
            let base = allocation.free.take(len, align)?;
            let idle_since = (allocation.live == 0).then_some(allocation.idle_since);
            allocation.live += 1;
            Some((allocation.carve(base, len), idle_since))
        })
    }

    /// The class's idle allocations the idle rule weighs, as when each became idle and its memory,
    /// the oldest first, and the bytes of its reserve. The newest idle unit allocations, of
    /// [`ALLOCATION_LEN`], with free bytes that the class's free bytes need to reach [`MAKE_BELOW`]
    /// are its reserve, made ahead of need: the rule neither weighs nor frees them. An allocation
    /// a take longer than the unit made for itself is never the reserve, even once its block's
    /// bytes are free ranges: the rule frees it at its idle age, so it is not held for the
    /// device's life.
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

    /// The next step of zeroing the kept Shared blocks: the newest kept block not zeroed to its
    /// end, where its next step starts in its mapping and how many bytes it zeroes. None while a
    /// take waits for a step.
    fn zero_step(&self) -> Option<(BlockId, Mapping, u64)> {
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        if self.zero_budget == Some(0) {
            return None;
        }
        if self.zero_waits > 0 {
            return None;
        }
        let kept = &self.kept[MemoryClass::Shared.index()];
        let memory = (kept.iter().rev())
            .find(|memory| memory.zeroed < memory.len && memory.mapping.is_some())?;
        let mapping = memory.mapping?;
        let len = (memory.len - memory.zeroed).min(ZERO_STEP);
        // SAFETY: `zeroed` is less than the block's length, which its mapping spans.
        let at = Mapping(unsafe { mapping.at(memory.zeroed) });
        Some((memory.id, at, len))
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
    /// What a block's buffer of `len` bytes at `base`, a range taken from the free bytes, binds
    /// to: the range's zeroed first bytes leave the clean ones, now the block's.
    fn carve(&mut self, base: u64, len: u64) -> Carve {
        let zeroed = self.clean.run_at(base, len);
        self.clean.cut(base, len);
        Carve {
            memory: self.memory,
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
/// allocation, zeroing the Shared one, so the consumer's first blocks of each class are ranges of
/// it as the device is ready, and nothing is allocated beside the consumer's first calls. Where
/// one cannot be made, the block-maker thread tries again after the class's next take or
/// give-back.
pub(crate) fn stock(core: &Core) {
    // Made ahead of need, as the block-maker thread's are: counted with them.
    ON_MAKER.set(true);
    for class in MemoryClass::ALL {
        let made = make_allocation(core, class, ALLOCATION_LEN).map(|mut allocation| {
            allocation.zero_ahead(&core.counters);
            allocation
        });
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
/// after the consumer's hook; it keeps each block given back for a take of its length (or gives
/// its bytes back, after the hook, where its allocation spilled); it makes an allocation of a
/// class whose free bytes are fewer than [`MAKE_BELOW`], zeroing a Shared one whole before it
/// joins the pool; it frees each allocation idle for [`IDLE_AGE_NANOS`], after the hook of each
/// kept block in it, one at a time; and last it zeroes the kept Shared blocks, the newest first,
/// [`ZERO_STEP`] bytes at a time from each one's first byte, taking up its other work again
/// between two steps. It sleeps until the next change, or until the oldest idle allocation
/// reaches the idle age.
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
                    if memory.zeroed > 0 {
                        allocation.clean.give(memory.base, memory.zeroed);
                    }
                }
            }
            state.unkeep[class.index()] = false;
            pool.changed.notify_all();
            continue;
        }
        if let Some(memory) = state.returned.pop_front() {
            // A block of a spilled allocation is not kept: its bytes go back to its allocation,
            // and a spilled allocation that empties is the idle rule's next to free, so the next
            // tries device-local memory again. A kept block stays under the lock from the queue to
            // the kept blocks, so a take finds it in one or the other.
            let kept = !memory.spilled;
            if !kept {
                drop(state);
                retire(&core, &memory);
                state = pool.lock();
            }
            let class = memory.class;
            state.failed[class.index()] = false;
            if let Some(allocation) = state.find(class, memory.memory) {
                allocation.live -= 1;
                if !kept {
                    allocation.free.give(memory.base, memory.len);
                }
                if allocation.live == 0 {
                    // Its idle time starts now, however long ago the block was released.
                    allocation.idle_since = crate::host_time();
                    allocation.retiring = allocation.spilled;
                }
            }
            if kept {
                state.kept[class.index()].push(memory);
            }
            pool.settle(&mut state, class);
            pool.changed.notify_all();
            continue;
        }
        let wanted = (MemoryClass::ALL.into_iter())
            .find(|&class| !state.failed[class.index()] && state.free(class) < MAKE_BELOW);
        if let Some(class) = wanted {
            drop(state);
            let made = make_allocation(&core, class, ALLOCATION_LEN).map(|mut allocation| {
                allocation.zero_ahead(&core.counters);
                allocation
            });
            state = pool.lock();
            match made {
                // Idle from now: the reserve, which the idle rule does not weigh.
                Ok(allocation) => state.allocations[class.index()].push(allocation),
                Err(_) => state.failed[class.index()] = true,
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
        if let Some((allocation, kept)) = gone {
            pool.changed.notify_all();
            drop(state);
            free_idle(&core, allocation, &kept);
            state = pool.lock();
            continue;
        }
        // Last, the kept Shared blocks' bytes, the newest block first, a step at a time from its
        // first byte, off the lock; none while a take waits for a step, and none while a consumer
        // memset runs ([`Pool::fill`]): the step waits for the memset's end.
        let step = state.zero_step();
        pool.fill_held.store(step.is_some(), SeqCst);
        let held = step.is_some() && pool.fills.load(SeqCst) > 0;
        if held {
            core.counters.zero_step_held();
        }
        if let Some((id, at, len)) = step.filter(|_| !held) {
            pool.fill_held.store(false, Relaxed);
            state.zeroing = Some(id);
            drop(state);
            let start = crate::host_time();
            // SAFETY: `at` is `len` bytes inside a kept block's mapping, which lives while the
            // block is kept: no take hands it out, and nothing frees it, while `zeroing` names it.
            unsafe { at.get().as_ptr().write_bytes(0, len as usize) };
            core.counters
                .zeroed_kept(len, crate::host_time().saturating_sub(start));
            state = pool.lock();
            let mut kept = state.kept[MemoryClass::Shared.index()].iter_mut();
            if let Some(memory) = kept.find(|memory| memory.id == id) {
                memory.zeroed += len;
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

/// A `VkBuffer` of `len` bytes with every buffer usage.
fn new_buffer(device: &ash::Device, len: u64) -> Result<vk::Buffer, Error> {
    let info = vk::BufferCreateInfo::default()
        .size(len)
        .usage(floor::BUFFER_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    // SAFETY: the device is live; `len` is at most the device's `maxBufferSize`.
    unsafe { device.create_buffer(&info, None) }.map_err(Error::vulkan("vkCreateBuffer"))
}

/// Makes an allocation of `len` bytes of `class` with device addresses, mapped whole if
/// Shared, its free ranges all of it. Its bytes are undefined: nothing writes them, and a Shared
/// allocation's pages become resident as they are first written.
fn make_allocation(core: &Core, class: MemoryClass, len: u64) -> Result<Allocation, Error> {
    let device = &core.device;
    let buffer = vk::BufferCreateInfo::default()
        .size(len)
        .usage(floor::BUFFER_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let of = vk::DeviceBufferMemoryRequirements::default().create_info(&buffer);
    let mut needs = vk::MemoryRequirements2::default();
    // SAFETY: the device is live, and the create info valid: `len` is at most its
    // `maxBufferSize`.
    unsafe { device.get_device_buffer_memory_requirements(&of, &mut needs) };
    let requirements = needs.memory_requirements;
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size.max(len))
        .memory_type_index(core.memory_type(class))
        .push_next(&mut flags);
    let bits = requirements.memory_type_bits;
    // SAFETY: the floor chose the memory type for buffers of this usage on this device, and
    // `bufferDeviceAddress` is enabled; a spill takes a type the buffer's requirements allow.
    let (memory, spilled) = unsafe { allocate_spilling(core, class, bits, &info) }?;
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
        clean: Ranges::empty(),
        live: 0,
        idle_since: crate::host_time(),
        retiring: false,
    };
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

/// Makes block `id` of `class`: a buffer of `len` bytes bound at `carve`'s range, its address and
/// its mapping.
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
        zeroed: carve.zeroed,
    })
}

/// Ends a block: the consumer's hook where it was handed out and its hook has not run, then its
/// buffer. Its bytes stay in its allocation.
fn retire(core: &Core, memory: &Memory) {
    core.pool.unmap(memory.id);
    // SAFETY: no GPU work that uses the block is pending: a command buffer or upload that uses it
    // holds it until it retires. The consumer's mappings are gone.
    unsafe { core.device.destroy_buffer(memory.buffer, None) };
}

/// On the block-maker thread: frees an allocation the idle rule took out, after the hook of each
/// block kept in it, and counts its idle time. The submissions made during its `vkFreeMemory` are
/// counted.
fn free_idle(core: &Core, allocation: Allocation, kept: &[Memory]) {
    for memory in kept {
        retire(core, memory);
    }
    // The bytes zeroed for a take that never came.
    let kept_zeroed: u64 = kept.iter().map(|memory| memory.zeroed).sum();
    (core.counters).zeroed_freed(allocation.clean.bytes() + kept_zeroed);
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
        retired.extend(state.returned.drain(..));
    }
    for class in MemoryClass::ALL {
        core.counters
            .fall_by(class.idle_gauge(), idle[class.index()]);
        (core.counters).idle_since(class == MemoryClass::Shared, 0);
        (core.counters).reserve_bytes(class == MemoryClass::Shared, 0);
    }
    for memory in &retired {
        retire(core, memory);
    }
    for allocation in allocations {
        free_allocation(core, allocation);
    }
}
