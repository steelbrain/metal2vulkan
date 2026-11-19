//! Memory: the blocks the consumer takes and releases, each a range of one of the device's
//! allocations; the pool that hands them out from the allocations' free ranges and frees idle
//! allocations; the allocations its own thread makes ahead of need, while a class holds no wholly
//! free unit, and the runtime's fixed-size internals it makes for the rest of the device
//! ([`Internal`]); and how memory is made and freed. A block's bytes are undefined: the runtime
//! zeroes nothing ahead of need. A buffer reads zero where
//! [`Device::new_buffer`](crate::Device::new_buffer) zeroes its range.

use crate::counters::{Counters, Gauge, Metric, Site};
use crate::device::{lock, Core};
use crate::error::{Error, Limit};
use crate::floor;
use crate::places::Ranges;
use ash::vk;
use std::collections::{HashSet, VecDeque};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

/// The length of each allocation the block-maker thread makes, and of an allocation a take makes
/// where no free range fits it, unless the take is longer: 16 MiB of either class, the unit.
///
/// A driver may hold the device's other calls (pipeline creation, submissions, other allocations)
/// for the whole of an allocation's `vkAllocateMemory` and `vkFreeMemory`, and the hold grows with
/// the bytes. reims measured it on an RTX 3090 and an RTX 2080 (NVIDIA's Linux driver 620.70.06):
/// one thread allocating memory with a device address while another times empty
/// `vkQueueSubmit`s in a loop, the longest of which takes 1.76 ms on the 3090 and 2.08 ms on the
/// 2080 with no allocation beside it. A 16 MiB Private allocation holds no submission past that,
/// and one of 256 MiB holds one for 6.6 to 8 ms. A 16 MiB Shared allocation (host-visible,
/// host-coherent and host-cached) holds one for 3.3 ms at the median and 4.5 ms at most, and one
/// of 256 MiB for 51 ms on the 3090 and 58 ms on the 2080. Freeing 16 MiB of Private memory holds
/// none past the baseline, and 256 MiB about 5 ms; freeing 16 MiB of Shared memory holds one for
/// 4.1 ms at the median and up to 19 ms, and 256 MiB for 93 ms at the median and up to 242 ms. So
/// allocations are short, and a take longer than one pays its own length
/// once: the allocation it makes, a long allocation, serves later takes of that length.
const ALLOCATION_LEN: u64 = 16 << 20;

/// A block's length, and its first byte's offset in its allocation, are multiples of this, or of
/// the device's `bufferImageGranularity` where that is larger: no block shares a page of that
/// granularity with another, so a block of buffers lies beside a block of optimal images.
pub(crate) const GRANULE: u64 = 64 << 10;

/// A block's first byte lies at a multiple of its length rounded up to a power of two, up to this
/// much, in its allocation: an image placed in a block at an offset its alignment allows from the
/// block's first byte is placed at one from the allocation's.
const BASE_ALIGN: u64 = 2 << 20;

/// The idle age, in nanoseconds of [`host_time`](crate::host_time): 250 ms, one value for every
/// device. An allocation is idle once it holds no live block: its idle time starts when its last
/// live block comes back, and only a take that lands in it again ends it. Commits and submissions
/// play no part, so a program that keeps committing gives back what it stopped using. The
/// block-maker thread frees an allocation idle this long, after the consumer's unmap of it, one at
/// a time and between the allocations it makes, so a free never runs beside its own
/// `vkAllocateMemory`; a class's reserve ([`State::whole`]) is not freed. A driver may hold a
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
    /// allocations, the first with room; the block-maker thread makes another while none of the
    /// class's allocations of this length is wholly free. A take that no free range fits makes an
    /// allocation of this length, which later takes share, or of its own where it is longer, which
    /// only later takes of that length share.
    pub const fn block_len(self) -> u64 {
        ALLOCATION_LEN
    }

    /// The gauge of the bytes of the class's live allocations.
    fn allocation_gauge(self) -> Gauge {
        match self {
            MemoryClass::Private => Gauge::PrivateAllocationBytes,
            MemoryClass::Shared => Gauge::SharedAllocationBytes,
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
/// buffers in ([`Device::new_buffer`](crate::Device::new_buffer)); for an imported block
/// (`Device::import_dma_buf` on Unix,
/// [`Device::import_host_memory`](crate::Device::import_host_memory)), the memory of a no-copy
/// buffer (`newBufferWithBytesNoCopy:length:options:deallocator:`); for an exported block
/// ([`Device::new_exported_block`](crate::Device::new_exported_block)), none.
///
/// Plain data, which the block's handle ([`Block`](crate::Block)) derefs to: the block lives
/// until its last handle and its last buffer, command buffer or upload are gone, or until the
/// device is torn down. It is a range of one allocation, whose one `VkBuffer`, with every buffer
/// usage and a device address, spans it; an imported or exported block is the whole of an
/// allocation of its own. Only an exported block is exportable.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub struct BlockInfo {
    /// The block's number.
    pub id: BlockId,
    /// Its storage mode.
    pub class: MemoryClass,
    /// The length asked for, rounded up to 64 KiB (to the device's `bufferImageGranularity`
    /// where that is larger); an imported block's, the length imported.
    pub len: u64,
    /// The allocation it is a range of, which [`BlockMappings::map`] was told of before this
    /// block went out; an imported or exported block's own, of which no hook is told.
    pub allocation: AllocationId,
    /// The offset of its first byte in that allocation.
    pub offset: u64,
    /// `gpuAddress` of the block's first byte: its allocation's device address plus
    /// [`offset`](Self::offset).
    pub gpu_address: u64,
    /// `contents` of the block's first byte: the host mapping of a Shared block, `None` for a
    /// Private one; for an imported block, the host memory imported, or `None` for a dma-buf,
    /// whose pages the consumer reaches through its own mappings.
    pub contents: Option<NonNull<u8>>,
    /// Whether its memory type is device-local. A Private block is, unless the device-local type
    /// was out of memory when its allocation was made and it went to the next type its buffer
    /// takes (counted in [`Gauges::spilled_bytes`](crate::Gauges::spilled_bytes)); a Shared
    /// block is where the device's Shared type is, and an imported one where the type imported
    /// into is.
    pub device_local: bool,
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
/// a free range damages only its own later blocks' bytes. Imported and exported blocks are not
/// the pool's, and neither hook runs for them.
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
    /// An import's block ([`import`]) or an export's ([`export`]): an allocation of its own,
    /// which no pool holds and which is freed once the block is given back.
    pub(crate) own: Option<Own>,
}

/// What a block that is an allocation of its own is.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Own {
    /// An import of memory the consumer brought ([`import`]).
    Import,
    /// A dedicated allocation the consumer shares by a handle of this kind ([`export`]).
    Export(crate::ExportHandle),
    /// An exported surface's buffer ([`crate::Surface::buffer`]): on one more import of its
    /// dma-buf, an allocation of its own (`import`), or on the opaque handle's one allocation,
    /// which the block under it holds and frees ([`surface_view`]).
    Surface {
        /// Its memory is an import of its own, freed with it.
        import: bool,
    },
    /// An import over several dma-bufs ([`crate::Device::import_dma_bufs`]): its sparse buffer,
    /// bound to its chunks' imports by the bind that signals `bind` on the binds' timeline
    /// ([`crate::import::Sparse`]), whose memory requirements name `space` bytes of the sparse
    /// address space. Its memory is its chunks', each a block of its own given back after it
    /// ([`sparse_block`]).
    #[cfg(unix)]
    Sparse {
        /// The bind's value.
        bind: u64,
        /// The bytes of the sparse address space it takes.
        space: u64,
    },
    /// One dma-buf's import under such a buffer: memory and no buffer.
    #[cfg(unix)]
    Chunk,
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

    /// Whether it is memory the consumer brought: one import, or several behind one buffer.
    pub(crate) fn imported(&self) -> bool {
        match self.own {
            Some(Own::Import) => true,
            #[cfg(unix)]
            Some(Own::Sparse { .. }) => true,
            _ => false,
        }
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
        }
    }
}

/// A host address in a Shared allocation's mapping, which lives as long as the allocation: its
/// first byte, a block's or a buffer's.
#[derive(Clone, Copy)]
pub(crate) struct Mapping(NonNull<u8>);

// SAFETY: an address. The runtime writes through it only while it owns the bytes: `new_buffer`
// zeroing the range of the buffer it makes.
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

/// The consumer's release of an imported block's memory: Metal's deallocator, run once the
/// memory is freed.
pub(crate) type Release = Box<dyn FnOnce() + Send>;

/// A block that has been handed out. The consumer's hold on it, each buffer of it, and each
/// command buffer and upload that uses it until it retires are one reference each; the last one
/// dropped gives it back to the pool, or, for an import, to the block-maker thread to free.
pub(crate) struct Block {
    pub(crate) memory: Memory,
    core: Arc<Core>,
    /// An import's release, which runs once its memory is freed: behind a lock, as the block is
    /// shared between threads and the release is only `Send`.
    release: Mutex<Option<Release>>,
    /// The block whose allocation this block's buffer is bound in ([`surface_view`]): given back
    /// after this one, so the block-maker thread destroys the buffer before it frees the memory.
    _under: Option<Arc<Block>>,
    /// An import over several dma-bufs' chunks, in order ([`sparse_block`]): given back after
    /// this one, so the block-maker thread destroys the sparse buffer before it frees any of
    /// their memory, the last of them holding the release.
    _chunks: Vec<Block>,
}

impl Block {
    pub(crate) fn new(memory: Memory, core: Arc<Core>) -> Block {
        Block {
            memory,
            core,
            release: Mutex::new(None),
            _under: None,
            _chunks: Vec::new(),
        }
    }

    /// An import's block ([`import`]), whose `release` runs once its memory is freed.
    pub(crate) fn imported(memory: Memory, core: Arc<Core>, release: Option<Release>) -> Block {
        Block {
            memory,
            core,
            release: Mutex::new(release),
            _under: None,
            _chunks: Vec::new(),
        }
    }

    /// A surface's buffer on the allocation `under` holds ([`surface_view`]).
    fn over(memory: Memory, core: Arc<Core>, under: Arc<Block>) -> Block {
        Block {
            memory,
            core,
            release: Mutex::new(None),
            _under: Some(under),
            _chunks: Vec::new(),
        }
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        if self.memory.own.is_some() {
            let release = (self.release.get_mut())
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            self.core.pool.give_back_own(self.memory, release);
        } else {
            self.core.pool.give_back(self.memory);
        }
    }
}

/// One `VkDeviceMemory` of a class, the `VkBuffer` that spans it, and the ranges of it no block
/// holds.
struct Allocation {
    id: AllocationId,
    class: MemoryClass,
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
/// A take is the first free range of the class's allocations that fits it, in the order the
/// allocations were made; a long allocation serves only takes of its own length. Where none fits,
/// the take makes an allocation itself, of [`ALLOCATION_LEN`] or of its own length where that is
/// longer. A block given back is a free range of its allocation at once. The block-maker thread
/// ([`run`]) makes an allocation of a class while the class holds no wholly free unit
/// ([`State::whole`]); a take never waits for it.
///
/// The idle rule never frees the class's reserve, its last idle unit allocation in the order they
/// were made, so it does not free what the block-maker thread would make again. A long allocation
/// is never kept so.
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

/// What the block-maker thread makes for the device's other parts besides the pool's
/// allocations: the runtime's fixed-size internals, off every caller's thread. The device's making
/// asks it for the zero source, the parameter ring's first block and the staging region, its first
/// work once it starts, and the ring asks it for each later block ([`crate::ring`]). A call that
/// needs one while it is asked waits for its make ([`Pool::await_made`]), counted in
/// [`Metrics::maker_waits`](crate::Metrics::maker_waits); where the make failed, or the thread had
/// stopped, the call makes it on its own thread with the same function, counted at its [`Site`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Internal {
    /// The zero source of the fills' and new textures' copies ([`crate::fill::Filler`]).
    ZeroSource,
    /// The parameter ring's next block ([`crate::ring`]).
    RingBlock,
    /// The 16 MiB staging region of the copies between textures and the CPU
    /// ([`crate::staging`]).
    StagingRegion,
}

/// The number of [`Internal`]s.
const INTERNALS: usize = 3;

impl Internal {
    /// Every internal, in the order the block-maker thread makes those asked.
    const ALL: [Internal; INTERNALS] = [
        Internal::ZeroSource,
        Internal::RingBlock,
        Internal::StagingRegion,
    ];

    fn index(self) -> usize {
        self as usize
    }
}

/// For the device tests: what the block-maker thread does with the internals asked of it
/// ([`DeviceOptions::maker`](crate::DeviceOptions)).
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
#[derive(Clone, Default)]
pub(crate) enum MakerTest {
    /// Makes them.
    #[default]
    Make,
    /// Meets the test at the barrier twice before its first work, once parked and once released;
    /// then makes them.
    Park(Arc<std::sync::Barrier>),
    /// Every make of an internal fails, before the driver is called, so the calls that need them
    /// make them.
    Fail,
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
    /// Imported and exported blocks given back, with the imports' releases, for the block-maker
    /// thread to free.
    owned: VecDeque<(Memory, Option<Release>)>,
    /// Per class, the block-maker thread could not make the last allocation it tried. It tries
    /// again after the class's next take or give-back.
    failed: [bool; 2],
    /// The internals asked of the block-maker thread and not yet made, by [`Internal::index`]:
    /// the zero source, the ring's first block and the staging region from the device's making.
    asked: [bool; INTERNALS],
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
    /// For the device tests: the blocks given back.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    returned: HashSet<BlockId>,
}

thread_local! {
    /// Whether this thread makes allocations ahead of need: the block-maker thread ([`run`]), its
    /// internals among them, or the device's making while it makes the first allocations
    /// ([`stock`]). For the allocation-time metrics.
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
            own: None,
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
                owned: VecDeque::new(),
                failed: [false, false],
                // The device's fixed-size internals: the block-maker thread's first work.
                asked: [true; INTERNALS],
                stop: false,
                next_id: 1,
                next_allocation: 1,
                mapped: HashSet::new(),
                unmapped: false,
                idle_bytes: [0, 0],
                #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
                returned: HashSet::new(),
            }),
            changed: Condvar::new(),
            mappings,
            counters,
            granule: GRANULE.max(granularity.next_power_of_two()),
            alignment,
            freeing: AtomicBool::new(false),
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

    /// For the device tests: whether the class holds a wholly free unit ([`State::whole`]), so
    /// the block-maker thread makes nothing for it.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn stocked(&self, class: MemoryClass) -> bool {
        self.lock().whole(class)
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
            .map(|a| {
                let free = a.free.free().iter().map(|&(start, end)| end - start).sum();
                (a.id, a.len, a.live, free)
            })
            .collect()
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
    /// fits, in the order the allocations were made, a long allocation only for a take of its own
    /// length; with no Vulkan call. Where none fits, a range of an allocation made here, of
    /// [`ALLOCATION_LEN`] or the block's length where that is longer, which the consumer's
    /// [`BlockMappings::map`] maps first, counted at `site`. A block's bytes are undefined.
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
        site: Site,
    ) -> Result<Memory, Error> {
        let (len, align) = self.shape(len, max)?;
        let index = class.index();
        let mut state = self.lock();
        // A take is a pool event: the block-maker thread tries a failed allocation again.
        state.failed[index] = false;
        let id = number(&mut state)?;
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
            self.counters.carved();
            self.settle(&mut state, class);
            // The block-maker thread runs the idle rule on this take, and makes an allocation
            // where it took the class's last wholly free unit.
            self.changed.notify_all();
            return Ok(carve.block(id, class, len));
        }
        drop(state);
        let memory = self.own(core, class, len, align, id, site)?;
        let mut state = self.lock();
        self.settle(&mut state, class);
        self.changed.notify_all();
        Ok(memory)
    }

    /// Makes block `id` of `len` bytes at a multiple of `align` at the start of an allocation of
    /// its own, of [`ALLOCATION_LEN`] or `len` where that is longer, whose rest later takes share,
    /// counted at `site`; the consumer maps the allocation first.
    fn own(
        &self,
        core: &Core,
        class: MemoryClass,
        len: u64,
        align: u64,
        id: BlockId,
        site: Site,
    ) -> Result<Memory, Error> {
        let mut allocation = make_allocation(core, class, len.max(ALLOCATION_LEN), Some(site))?;
        if !self.map(&allocation, class, true) {
            free_allocation(core, allocation);
            return Err(Error::Limit(Limit::AllocationNotMapped { class, len }));
        }
        let base = allocation.free.take(len, align).unwrap_or(0);
        let memory = allocation.carve(base).block(id, class, len);
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
    /// allocation at once.
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

    /// Takes back an imported or exported block, for the block-maker thread to free, and then to
    /// run an import's release.
    pub(crate) fn give_back_own(&self, memory: Memory, release: Option<Release>) {
        self.lock().owned.push_back((memory, release));
        self.changed.notify_all();
    }

    /// An import's or an export's numbers: its block's, and its allocation's. Err:
    /// [`Limit::BlockIds`].
    pub(crate) fn own_numbers(&self) -> Result<(BlockId, AllocationId), Error> {
        let mut state = self.lock();
        Ok((number(&mut state)?, allocation_number(&mut state)?))
    }

    /// Stops the block-maker thread once it finishes what it is doing.
    pub(crate) fn stop(&self) {
        self.lock().stop = true;
        self.changed.notify_all();
    }

    /// Asks the block-maker thread to make `internal`, which it makes before its other work. False
    /// where the thread has stopped: the call that needs it makes it.
    pub(crate) fn ask(&self, internal: Internal) -> bool {
        let mut state = self.lock();
        if state.stop {
            return false;
        }
        state.asked[internal.index()] = true;
        self.changed.notify_all();
        true
    }

    /// Waits while `internal` is asked of the block-maker thread: until it has made it, its make
    /// failed, or it stopped. A wait is counted in [`Metrics::maker_waits`] and its time in
    /// [`Metrics::maker_wait_max_nanos`]; where it is not asked, the call returns at once.
    ///
    /// [`Metrics::maker_waits`]: crate::Metrics::maker_waits
    /// [`Metrics::maker_wait_max_nanos`]: crate::Metrics::maker_wait_max_nanos
    pub(crate) fn await_made(&self, internal: Internal) {
        let mut state = self.lock();
        if !state.asked[internal.index()] {
            return;
        }
        let since = crate::host_time();
        while state.asked[internal.index()] {
            state = self.wait(state);
        }
        drop(state);
        (self.counters).maker_wait(crate::host_time().saturating_sub(since));
    }

    /// For the device tests: whether the block-maker thread is told to stop.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn stopped(&self) -> bool {
        self.lock().stop
    }

    /// For the device tests: waits until the block-maker thread has made, or failed to make,
    /// every internal asked of it.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn await_internals(&self) {
        let mut state = self.lock();
        while state.asked.contains(&true) {
            state = self.wait(state);
        }
    }
}

impl State {
    /// Whether the class holds a wholly free unit: a unit allocation of [`ALLOCATION_LEN`], not
    /// retiring, that no block holds, so a take of up to a whole unit has a free range with no
    /// Vulkan call. The block-maker thread makes the class's next allocation while it holds none
    /// ([`run`]), and the idle rule keeps the last such unit in the order made, the reserve
    /// ([`aging`](Self::aging)): once the class is idle past [`IDLE_AGE_NANOS`], it holds that one
    /// wholly free unit and no other. The trade: a small take that breaks the class's last whole
    /// unit sets off an allocation beside the consumer's next calls, where a rule on the class's
    /// free bytes would wait until they ran low; in return a take of up to a unit, the 16 MiB
    /// ones among them, lands in memory made ahead once the block-maker thread has made it.
    fn whole(&self, class: MemoryClass) -> bool {
        (self.allocations[class.index()].iter())
            .any(|allocation| allocation.unit() && !allocation.retiring && allocation.live == 0)
    }

    /// The allocation of `class` whose memory is `memory`.
    fn find(&mut self, class: MemoryClass, memory: vk::DeviceMemory) -> Option<&mut Allocation> {
        (self.allocations[class.index()].iter_mut()).find(|allocation| allocation.memory == memory)
    }

    /// Reserves `len` bytes at a multiple of `align` in the class's allocations that serve a take
    /// of `len` ([`Allocation::serves`]), counting a block live there: the first such free range,
    /// in the order the allocations were made. With the range, when the allocation became idle
    /// where it was, and whether it is a long allocation.
    fn fit(
        &mut self,
        class: MemoryClass,
        len: u64,
        align: u64,
    ) -> Option<(Carve, Option<u64>, bool)> {
        let (allocation, base) = (self.allocations[class.index()].iter_mut())
            .filter(|allocation| allocation.serves(len))
            .find_map(|allocation| {
                let base = allocation.free.take(len, align)?;
                Some((allocation, base))
            })?;
        let idle_since = (allocation.live == 0).then_some(allocation.idle_since);
        allocation.live += 1;
        let long = !allocation.unit();
        if long {
            allocation.reused = true;
        }
        Some((allocation.carve(base), idle_since, long))
    }

    /// The class's idle allocations the idle rule weighs, as when each became idle and its memory,
    /// the oldest first, and the bytes of its reserve. The last idle unit allocation, of
    /// [`ALLOCATION_LEN`], in the order the allocations were made is the reserve, the class's
    /// wholly free unit ([`whole`](Self::whole)): the rule neither weighs nor frees it. Takes land
    /// in the first allocation with room, so the last is the one they reach last, and a class whose
    /// takes come and go within a unit's bytes keeps its reserve with no allocation made or freed.
    /// A long allocation, which a take longer than the unit made for itself, is never the reserve:
    /// the rule frees it at its idle age, so it is not held for the device's life.
    fn aging(&self, class: MemoryClass) -> (Vec<(u64, vk::DeviceMemory)>, u64) {
        let idle = (self.allocations[class.index()].iter())
            .filter(|allocation| allocation.live == 0 && !allocation.retiring);
        let reserve = idle.clone().rfind(|allocation| allocation.unit());
        let mut weighed: Vec<(u64, vk::DeviceMemory)> = idle
            .filter(|allocation| reserve.is_none_or(|reserve| reserve.memory != allocation.memory))
            .map(|allocation| (allocation.idle_since, allocation.memory))
            .collect();
        weighed.sort_unstable_by_key(|&(since, _)| since);
        (weighed, reserve.map_or(0, |reserve| reserve.len))
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
    /// What a block at `base`, a range taken from the free bytes, is.
    fn carve(&self, base: u64) -> Carve {
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
        }
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

fn allocation_number(state: &mut State) -> Result<AllocationId, Error> {
    let id = state.next_allocation;
    state.next_allocation = id.checked_add(1).ok_or(Error::Limit(Limit::BlockIds))?;
    Ok(AllocationId(id))
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

/// Makes an allocation ahead of need of `class`, and has the consumer map it: the allocation, or
/// `None` where it was not made or not mapped.
fn ahead(core: &Core, class: MemoryClass) -> Option<Allocation> {
    let allocation = make_allocation(core, class, ALLOCATION_LEN, None).ok()?;
    if !core.pool.map(&allocation, class, false) {
        free_allocation(core, allocation);
        return None;
    }
    Some(allocation)
}

/// In the device's making, before the block-maker thread starts: makes each class's first
/// allocation, and has the consumer map each, so the consumer's first
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

/// The block-maker thread, until the device's teardown or a quiesce stops it. Off every caller's
/// path, in order: it makes each internal asked of it ([`Internal`]), the device's fixed-size
/// ones first of all; it frees each imported or exported block given back, then runs an import's
/// release; it makes an allocation of a class that holds no wholly free unit ([`State::whole`]),
/// having the consumer map it before it joins the pool; and it frees each allocation idle for
/// [`IDLE_AGE_NANOS`], after the consumer's unmap of it, one at a time. It sleeps until the next
/// change, or until the oldest idle allocation reaches the idle age. Stopped, it makes nothing
/// more: what is asked of it and not made, the calls that need it make.
pub(crate) fn run(core: Arc<Core>) {
    ON_MAKER.set(true);
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    if let MakerTest::Park(park) = &core.maker_test {
        park.wait();
        park.wait();
    }
    let pool = &core.pool;
    let mut state = pool.lock();
    loop {
        if state.stop {
            let ring = state.asked[Internal::RingBlock.index()];
            state.asked = [false; INTERNALS];
            pool.changed.notify_all();
            drop(state);
            if ring {
                core.ring.unasked();
            }
            return;
        }
        if let Some(internal) = (Internal::ALL.into_iter()).find(|i| state.asked[i.index()]) {
            drop(state);
            make_internal(&core, internal);
            state = pool.lock();
            state.asked[internal.index()] = false;
            pool.changed.notify_all();
            continue;
        }
        if let Some((memory, release)) = state.owned.pop_front() {
            drop(state);
            free_own(&core, &memory, release);
            state = pool.lock();
            continue;
        }
        let wanted = (MemoryClass::ALL.into_iter())
            .find(|&class| !state.failed[class.index()] && !state.whole(class));
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
        state = match wake {
            Some(timeout) => pool.wait_for(state, timeout),
            None => pool.wait(state),
        };
    }
}

/// On the block-maker thread: makes `internal` with the function a call that needs it makes it
/// with, into the part of the device that holds it. A make that fails leaves it unmade: the call
/// that needs it makes it on its own thread, where the allocation may succeed then.
fn make_internal(core: &Arc<Core>, internal: Internal) {
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    if let MakerTest::Fail = core.maker_test {
        if internal == Internal::RingBlock {
            core.ring.unasked();
        }
        return;
    }
    let _ = match internal {
        Internal::ZeroSource => crate::fill::make(core),
        Internal::RingBlock => core.ring.make_ahead(core),
        Internal::StagingRegion => core.work.make_staging_region(core),
    };
}

/// [`allocate`] of `info`, and for Private memory, where the device-local type is out of device
/// memory, of each other type `bits` allows in the device's order until one is not: the spill
/// the plan decided, on the allocation's own error and never a budget poll. Returns the memory
/// and whether it spilled. Each call is counted at `site`, where it names one.
///
/// # Safety
///
/// As [`allocate`], for every type `bits` allows.
pub(crate) unsafe fn allocate_spilling(
    core: &Core,
    class: MemoryClass,
    bits: u32,
    info: &vk::MemoryAllocateInfo<'_>,
    site: Option<Site>,
) -> Result<(vk::DeviceMemory, bool), Error> {
    let out = |result: &Result<vk::DeviceMemory, Error>| {
        result.as_ref().is_err_and(Error::is_out_of_device_memory)
    };
    let first = info.memory_type_index;
    let mut result = match core.fails_private(class) {
        true => Err(Error::out_of_device_memory()),
        // SAFETY: the caller's.
        false => unsafe {
            allocate(
                &core.device,
                &core.counters,
                core.max_allocations,
                info,
                site,
            )
        },
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
        result = unsafe {
            allocate(
                &core.device,
                &core.counters,
                core.max_allocations,
                &next,
                site,
            )
        };
        if let Ok(memory) = result {
            return Ok((memory, true));
        }
    }
    result.map(|memory| (memory, false))
}

/// `vkAllocateMemory`, every allocation the runtime makes: one more live allocation in
/// `counters`' blocks gauge, refused with [`Limit::MemoryAllocations`] before the driver is called
/// where `max`, the device's `maxMemoryAllocationCount`, are live. Each free falls the gauge. The
/// call's time is counted on the block-maker thread or on the consumer's, and on the consumer's
/// at `site` too, where the call names one.
///
/// # Safety
///
/// `info` is valid for `device`.
pub(crate) unsafe fn allocate(
    device: &ash::Device,
    counters: &Counters,
    max: u32,
    info: &vk::MemoryAllocateInfo<'_>,
    site: Option<Site>,
) -> Result<vk::DeviceMemory, Error> {
    if !counters.rise_below(Gauge::Blocks, u64::from(max)) {
        return Err(Error::Limit(Limit::MemoryAllocations { max }));
    }
    let start = crate::host_time();
    // SAFETY: the caller's.
    let allocated = unsafe { device.allocate_memory(info, None) };
    let nanos = crate::host_time().saturating_sub(start);
    let maker = ON_MAKER.get();
    counters.allocation_time(nanos, maker);
    if let (false, Some(site)) = (maker, site) {
        counters.site_allocation(site, info.allocation_size, nanos);
    }
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
/// Shared, its free ranges all of it. Nothing of the runtime writes its bytes, and a Shared
/// allocation's pages become resident as they are first written. Its bytes are undefined. Counted
/// at `site`, where it names one.
fn make_allocation(
    core: &Core,
    class: MemoryClass,
    len: u64,
    site: Option<Site>,
) -> Result<Allocation, Error> {
    let device = &core.device;
    let id = allocation_number(&mut core.pool.lock())?;
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
    let (memory, spilled) =
        match unsafe { allocate_spilling(core, class, bits, &allocate_info, site) } {
            Ok(made) => made,
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { device.destroy_buffer(buffer, None) };
                return Err(error);
            }
        };
    let mut allocation = Allocation {
        id,
        class,
        memory,
        buffer,
        address: 0,
        len,
        mapping: None,
        device_local: !spilled && core.device_local & (1 << core.memory_type(class)) != 0,
        spilled,
        free: Ranges::new(len),
        live: 0,
        idle_since: crate::host_time(),
        retiring: false,
        reused: false,
    };
    core.counters.count(Metric::BlocksMade);
    core.counters.allocated(len);
    core.counters.rise_by(class.allocation_gauge(), len);
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

/// The memory requirements of an import's buffer of `len` bytes for `handle`.
pub(crate) fn import_requirements(
    core: &Core,
    len: u64,
    handle: vk::ExternalMemoryHandleTypeFlags,
) -> vk::MemoryRequirements {
    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
    let buffer = vk::BufferCreateInfo::default()
        .size(len)
        .usage(floor::BUFFER_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external);
    let of = vk::DeviceBufferMemoryRequirements::default().create_info(&buffer);
    let mut needs = vk::MemoryRequirements2::default();
    // SAFETY: the device is live, and the create info valid: `len` is at most its
    // `maxBufferSize`, and its buffers import `handle` ([`crate::import`]).
    unsafe {
        core.device
            .get_device_buffer_memory_requirements(&of, &mut needs)
    };
    needs.memory_requirements
}

/// Why an import's block was not made.
pub(crate) enum ImportFailed {
    /// Nothing was imported: the buffer, or `vkAllocateMemory` (or the count of allocations ahead
    /// of it), failed. The handle is still the caller's.
    Before(Error),
    /// The import was made, then freed, with the handle it took: binding the buffer, or the
    /// block's numbers, failed.
    After(Error),
}

impl ImportFailed {
    pub(crate) fn error(self) -> Error {
        match self {
            ImportFailed::Before(error) | ImportFailed::After(error) => error,
        }
    }
}

/// Makes an import's block: a buffer of `info`'s length for `handle`, the import `info` asks
/// for (counted against the device's allocations, [`allocate`]), bound to the buffer at its first
/// byte, the buffer's address, and its numbers: the block's, and its allocation's, which no hook
/// is told of. `mapping` is the host address of its bytes, where the consumer imported them by
/// address. `own` is [`Own::Import`] for the consumer's import, Shared as Metal's no-copy buffers
/// are, counted in the imports gauge; or [`Own::Surface`] for a dma-buf surface's buffer
/// ([`crate::Surface::buffer`]), Private as an export is, counted in the surface gauges.
///
/// # Safety
///
/// `info` is a valid import of `handle` into a type its buffers can use.
pub(crate) unsafe fn import(
    core: &Core,
    info: &vk::MemoryAllocateInfo<'_>,
    handle: vk::ExternalMemoryHandleTypeFlags,
    mapping: Option<Mapping>,
    own: Own,
) -> Result<Memory, ImportFailed> {
    let device = &core.device;
    let len = info.allocation_size;
    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
    let buffer = vk::BufferCreateInfo::default()
        .size(len)
        .usage(floor::BUFFER_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external);
    // SAFETY: as for `import_requirements`.
    let buffer = unsafe { device.create_buffer(&buffer, None) }.map_err(|result| {
        ImportFailed::Before(Error::Vulkan {
            call: "vkCreateBuffer",
            result,
        })
    })?;
    // SAFETY: the caller's.
    let memory = match unsafe { allocate(device, &core.counters, core.max_allocations, info, None) }
    {
        Ok(memory) => memory,
        Err(error) => {
            // SAFETY: `buffer` is unused.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(ImportFailed::Before(error));
        }
    };
    let undo = |error: Error| {
        // SAFETY: nothing uses the buffer or the memory.
        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }
        core.counters.fall(Gauge::Blocks);
        ImportFailed::After(error)
    };
    // SAFETY: the memory is of a type the buffer can use, and at least its size
    // ([`crate::import`] checks both), and no other buffer is bound to it.
    if let Err(result) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        return Err(undo(Error::Vulkan {
            call: "vkBindBufferMemory",
            result,
        }));
    }
    let (id, allocation) = core.pool.own_numbers().map_err(undo)?;
    let address = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    // SAFETY: the buffer has device-address usage and is bound.
    let address = unsafe { device.get_buffer_device_address(&address) };
    let class = match own {
        Own::Import => {
            core.counters.rise(Gauge::Imports);
            MemoryClass::Shared
        }
        _ => {
            core.counters.rise(Gauge::SurfaceBuffers);
            #[cfg(unix)]
            core.counters.rise(Gauge::SurfaceImports);
            MemoryClass::Private
        }
    };
    Ok(Memory {
        id,
        class,
        len,
        allocation,
        memory,
        base: 0,
        buffer,
        address,
        mapping,
        device_local: core.device_local & (1 << info.memory_type_index) != 0,
        spilled: false,
        own: Some(own),
    })
}

/// Makes the sparse buffer of an import over several dma-bufs ([`crate::Device::import_dma_bufs`]):
/// `len` bytes of the floor's usage, for dma-bufs, made with `VK_BUFFER_CREATE_SPARSE_BINDING_BIT`,
/// bound to nothing yet; with its memory requirements.
///
/// Err: [`Error::Vulkan`] where the buffer cannot be made.
///
/// # Safety
///
/// The device was made with `sparseBinding` for such buffers ([`crate::import::binds_dma_bufs`]),
/// and `len` is at most its `maxBufferSize`.
#[cfg(unix)]
pub(crate) unsafe fn sparse_buffer(
    core: &Core,
    len: u64,
) -> Result<(vk::Buffer, vk::MemoryRequirements), Error> {
    let device = &core.device;
    let mut external = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let info = vk::BufferCreateInfo::default()
        .flags(vk::BufferCreateFlags::SPARSE_BINDING)
        .size(len)
        .usage(floor::BUFFER_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external);
    // SAFETY: the caller's.
    let buffer =
        unsafe { device.create_buffer(&info, None) }.map_err(Error::vulkan("vkCreateBuffer"))?;
    // SAFETY: the buffer is live.
    let needs = unsafe { device.get_buffer_memory_requirements(buffer) };
    Ok((buffer, needs))
}

/// Imports one dma-buf of an import over several, `info`'s, with no buffer: counted against the
/// device's allocations ([`allocate`]) and in the imports gauge.
///
/// # Safety
///
/// `info` is a valid import of a dma-buf.
#[cfg(unix)]
pub(crate) unsafe fn import_chunk(
    core: &Core,
    info: &vk::MemoryAllocateInfo<'_>,
) -> Result<vk::DeviceMemory, Error> {
    // SAFETY: the caller's.
    let memory = unsafe {
        allocate(
            &core.device,
            &core.counters,
            core.max_allocations,
            info,
            None,
        )
    }?;
    core.counters.rise(Gauge::Imports);
    Ok(memory)
}

/// Frees a chunk's import that no block holds: an import over several dma-bufs that failed past
/// it, before its bind.
///
/// # Safety
///
/// Nothing is bound to `memory`, and nothing uses it.
#[cfg(unix)]
pub(crate) unsafe fn free_chunk(core: &Core, memory: vk::DeviceMemory) {
    // SAFETY: the caller's.
    unsafe { core.device.free_memory(memory, None) };
    core.counters.fall(Gauge::Blocks);
    core.counters.fall(Gauge::Imports);
}

/// What [`sparse_block`] makes a block of: the sparse buffer of `len` bytes, its chunks' imports
/// in order with their lengths, the memory type they are of, the bind's value, the sparse address
/// space its memory requirements name, and its numbers.
#[cfg(unix)]
pub(crate) struct SparseParts {
    pub(crate) buffer: vk::Buffer,
    pub(crate) len: u64,
    pub(crate) chunks: Vec<(vk::DeviceMemory, u64)>,
    pub(crate) index: u32,
    pub(crate) bind: u64,
    pub(crate) space: u64,
    pub(crate) numbers: (BlockId, AllocationId),
}

/// The block of an import over several dma-bufs: Shared, as an import is, over the sparse buffer
/// from its first byte, at the buffer's device address, never mapped. Each chunk is a block of its
/// own under it ([`Block::_chunks`]), of the same numbers, its memory and no buffer, the last one
/// holding `release`.
#[cfg(unix)]
pub(crate) fn sparse_block(
    core: &Arc<Core>,
    parts: SparseParts,
    release: Option<Release>,
) -> Block {
    let SparseParts {
        buffer,
        len,
        chunks,
        index,
        bind,
        space,
        numbers: (id, allocation),
    } = parts;
    let address = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    // SAFETY: the buffer has device-address usage, and it is sparse
    // (`VUID-vkGetBufferDeviceAddress-bufferDeviceAddress-03324`). Its addresses run over its
    // whole size (`VUID-VkDeviceAddress-size-11364`), across its chunks.
    let address = unsafe { core.device.get_buffer_device_address(&address) };
    let device_local = core.device_local & (1 << index) != 0;
    let memory =
        |memory: vk::DeviceMemory, len: u64, buffer: vk::Buffer, address: u64, own: Own| Memory {
            id,
            class: MemoryClass::Shared,
            len,
            allocation,
            memory,
            base: 0,
            buffer,
            address,
            mapping: None,
            device_local,
            spilled: false,
            own: Some(own),
        };
    let chunk = |(memory_of, chunk_len): (vk::DeviceMemory, u64), release: Option<Release>| {
        let held = memory(memory_of, chunk_len, vk::Buffer::null(), 0, Own::Chunk);
        Block::imported(held, Arc::clone(core), release)
    };
    let chunks = match chunks.split_last() {
        Some((&last, before)) => (before.iter())
            .map(|&before| chunk(before, None))
            .chain(std::iter::once(chunk(last, release)))
            .collect(),
        None => Vec::new(),
    };
    let own = Own::Sparse { bind, space };
    Block {
        memory: memory(vk::DeviceMemory::null(), len, buffer, address, own),
        core: Arc::clone(core),
        release: Mutex::new(None),
        _under: None,
        _chunks: chunks,
    }
}

/// Makes an export's block: a buffer of `len` bytes rounded up to 64 KiB, as a take's are, that
/// exports `handle`, its dedicated allocation in the Private memory type, exportable as `handle`
/// (counted against the device's allocations, [`allocate`]), bound at its first byte, the
/// buffer's address, and its numbers: the block's, and its allocation's, which no hook is told
/// of. Private; never mapped. With it, the allocation's size, which an import of
/// the handle names.
///
/// Err: [`Limit::BlockLen`] past the device's longest allocation or buffer; [`Limit::ExportType`]
/// where the buffer's memory requirements do not list the Private type;
/// [`Limit::MemoryAllocations`]; [`Limit::BlockIds`]; [`Error::Vulkan`] where the buffer, the
/// allocation or its binding fails.
///
/// # Safety
///
/// The device's buffers export `handle` with the floor's usage, its extensions enabled
/// ([`crate::export`]).
pub(crate) unsafe fn export(
    core: &Core,
    len: u64,
    handle: crate::ExportHandle,
) -> Result<(Memory, u64), Error> {
    let device = &core.device;
    let handle_type = handle.vk();
    let len = round(len, GRANULE, core.max_block_len)?;
    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle_type);
    let info = vk::BufferCreateInfo::default()
        .size(len)
        .usage(floor::BUFFER_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external);
    // SAFETY: `len` is at most `maxBufferSize` (`round`), and the buffers export `handle` (the
    // caller's).
    let buffer =
        unsafe { device.create_buffer(&info, None) }.map_err(Error::vulkan("vkCreateBuffer"))?;
    let destroy = || {
        // SAFETY: nothing uses the buffer.
        unsafe { device.destroy_buffer(buffer, None) }
    };
    // SAFETY: the buffer is live.
    let needs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let index = core.memory_type(MemoryClass::Private);
    if needs.memory_type_bits & (1 << index) == 0 {
        destroy();
        return Err(Error::Limit(Limit::ExportType {
            handle,
            listed: Some(needs.memory_type_bits),
        }));
    }
    let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(handle_type);
    let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(needs.size)
        .memory_type_index(index)
        .push_next(&mut export)
        .push_next(&mut dedicated)
        .push_next(&mut flags);
    // SAFETY: a dedicated allocation of the buffer's size in a type it lists, exportable as a
    // handle its buffers export.
    let memory =
        match unsafe { allocate(device, &core.counters, core.max_allocations, &info, None) } {
            Ok(memory) => memory,
            Err(error) => {
                destroy();
                return Err(error);
            }
        };
    let undo = |error: Error| {
        // SAFETY: nothing uses the buffer or the memory.
        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }
        core.counters.fall(Gauge::Blocks);
        error
    };
    // SAFETY: the memory is the buffer's dedicated allocation, of its size and a type it lists.
    if let Err(result) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        return Err(undo(Error::Vulkan {
            call: "vkBindBufferMemory",
            result,
        }));
    }
    let (id, allocation) = core.pool.own_numbers().map_err(undo)?;
    let address = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    // SAFETY: the buffer has device-address usage and is bound.
    let address = unsafe { device.get_buffer_device_address(&address) };
    core.counters.rise(Gauge::Exports);
    let memory = Memory {
        id,
        class: MemoryClass::Private,
        len,
        allocation,
        memory,
        base: 0,
        buffer,
        address,
        mapping: None,
        device_local: core.device_local & (1 << index) != 0,
        spilled: false,
        own: Some(Own::Export(handle)),
    };
    Ok((memory, needs.size))
}

/// Makes an opaque handle's surface allocation in place ([`crate::surface`]): `len` bytes rounded
/// up to 64 KiB of memory type `index`, exportable as `handle`, dedicated to nothing and with
/// device addresses, counted against the device's allocations ([`allocate`]), and its numbers. It
/// holds no buffer: the surface's plane images are bound in it, and its block's buffer is null
/// and its address 0. The surface's buffer, where the consumer asks for one, is bound in it later
/// ([`surface_view`]). Counted as an export.
///
/// Err: [`Limit::BlockLen`] past the device's longest allocation; [`Limit::MemoryAllocations`];
/// [`Limit::BlockIds`]; [`Error::Vulkan`] where the allocation fails.
///
/// # Safety
///
/// Every image bound in it is made for `handle`, and none requires an allocation of its own.
pub(crate) unsafe fn export_surface(
    core: &Core,
    len: u64,
    index: u32,
    handle: crate::ExportHandle,
) -> Result<Memory, Error> {
    let device = &core.device;
    let len = round(len, GRANULE, core.max_block_len)?;
    let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(handle.vk());
    // A buffer bound in it later takes a device address
    // (`VUID-vkBindBufferMemory-bufferDeviceAddress-03339`).
    let mut flags =
        vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(len)
        .memory_type_index(index)
        .push_next(&mut export)
        .push_next(&mut flags);
    // SAFETY: the caller's; the type is the images'.
    let memory = unsafe { allocate(device, &core.counters, core.max_allocations, &info, None) }?;
    let (id, allocation) = match core.pool.own_numbers() {
        Ok(numbers) => numbers,
        Err(error) => {
            // SAFETY: nothing is bound to the memory.
            unsafe { device.free_memory(memory, None) };
            core.counters.fall(Gauge::Blocks);
            return Err(error);
        }
    };
    core.counters.rise(Gauge::Exports);
    Ok(Memory {
        id,
        class: MemoryClass::Private,
        len,
        allocation,
        memory,
        base: 0,
        buffer: vk::Buffer::null(),
        address: 0,
        mapping: None,
        device_local: core.device_local & (1 << index) != 0,
        spilled: false,
        own: Some(Own::Export(handle)),
    })
}

/// Makes an opaque surface's buffer ([`crate::Surface::buffer`]) on its allocation, which `under`
/// holds: a buffer of the allocation's length that names `handle`, bound at its first byte, its
/// address, and its numbers, which no hook is told of. Private and never mapped, as the
/// allocation is; it frees no memory, and holds `under` until it is given back, which then goes
/// back after it. Counted in the surface buffers gauge.
///
/// Err: [`Limit::ExportType`] where the buffer's memory requirements do not list the
/// allocation's type `index`, ask more than its length or require an allocation of its own;
/// [`Limit::BlockIds`]; [`Error::Vulkan`] where the buffer or its binding fails.
///
/// # Safety
///
/// `under` is an opaque surface's allocation ([`export_surface`]) of type `index`, exportable as
/// `handle`.
pub(crate) unsafe fn surface_view(
    core: &Arc<Core>,
    under: &Arc<Block>,
    handle: crate::ExportHandle,
    index: u32,
) -> Result<Block, Error> {
    let device = &core.device;
    let len = under.memory.len;
    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle.vk());
    let info = vk::BufferCreateInfo::default()
        .size(len)
        .usage(floor::BUFFER_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external);
    // SAFETY: `len` is the allocation's, at most `maxBufferSize` (`export_surface` rounds it), and
    // the device's buffers export `handle` (`crate::Device::new_exported_surface` checked it).
    let buffer =
        unsafe { device.create_buffer(&info, None) }.map_err(Error::vulkan("vkCreateBuffer"))?;
    let destroy = || {
        // SAFETY: nothing uses the buffer.
        unsafe { device.destroy_buffer(buffer, None) }
    };
    let mut dedicated = vk::MemoryDedicatedRequirements::default();
    let mut needs = vk::MemoryRequirements2::default().push_next(&mut dedicated);
    let of = vk::BufferMemoryRequirementsInfo2::default().buffer(buffer);
    // SAFETY: the buffer is live.
    unsafe { device.get_buffer_memory_requirements2(&of, &mut needs) };
    let needs = needs.memory_requirements;
    // A buffer that requires an allocation of its own is bound in no other
    // (`VUID-vkBindBufferMemory-buffer-01444`).
    if needs.memory_type_bits & (1 << index) == 0
        || needs.size > len
        || dedicated.requires_dedicated_allocation != vk::FALSE
    {
        destroy();
        return Err(Error::Limit(Limit::ExportType {
            handle,
            listed: Some(needs.memory_type_bits),
        }));
    }
    // SAFETY: the allocation is of a type the buffer lists, at least its size, dedicated to
    // nothing, made with device addresses and exportable as a handle the buffer names
    // (`VUID-vkBindBufferMemory-memory-02726`); an offset of 0 meets every alignment.
    if let Err(result) = unsafe { device.bind_buffer_memory(buffer, under.memory.memory, 0) } {
        destroy();
        return Err(Error::Vulkan {
            call: "vkBindBufferMemory",
            result,
        });
    }
    let (id, allocation) = match core.pool.own_numbers() {
        Ok(numbers) => numbers,
        Err(error) => {
            destroy();
            return Err(error);
        }
    };
    let address = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    // SAFETY: the buffer has device-address usage and is bound.
    let address = unsafe { device.get_buffer_device_address(&address) };
    core.counters.rise(Gauge::SurfaceBuffers);
    let memory = Memory {
        id,
        class: MemoryClass::Private,
        len,
        allocation,
        memory: under.memory.memory,
        base: 0,
        buffer,
        address,
        mapping: None,
        device_local: under.memory.device_local,
        spilled: false,
        own: Some(Own::Surface { import: false }),
    };
    Ok(Block::over(memory, Arc::clone(core), Arc::clone(under)))
}

/// Frees an imported or exported block given back: its buffer, then its memory, then runs an
/// import's release, as the bytes are the consumer's alone again; of a surface's buffer on the
/// opaque handle's allocation, the buffer alone; of an import over several dma-bufs, the sparse
/// buffer alone, once its bind completed, and of each of its chunks, the memory alone, the last
/// one's release after it. The submissions made during the `vkFreeMemory` are counted.
fn free_own(core: &Core, memory: &Memory, release: Option<Release>) {
    match memory.own {
        // An import over several dma-bufs' buffer: its bind, the one use of it no command buffer
        // held, completes before it goes (`VUID-vkDestroyBuffer-buffer-00922`), and its chunks,
        // given back after it, free the memory.
        #[cfg(unix)]
        Some(Own::Sparse { bind, space }) => {
            let sparse = core.imports.sparse.as_ref();
            if let Some(sparse) = sparse {
                sparse.wait(&core.device, bind);
            }
            // SAFETY: as below, and the bind completed.
            unsafe { core.device.destroy_buffer(memory.buffer, None) };
            if let Some(sparse) = sparse {
                sparse.unreserve(space);
            }
            return;
        }
        // A chunk has no buffer.
        #[cfg(unix)]
        Some(Own::Chunk) => {}
        // SAFETY: no GPU work that uses the block is pending (a command buffer or upload that
        // uses it holds it until it retires), and no buffer of it is left.
        _ => unsafe { core.device.destroy_buffer(memory.buffer, None) },
    }
    // A surface's buffer on the opaque handle's allocation frees no memory: the block under it,
    // given back after it, does.
    if memory.own == Some(Own::Surface { import: false }) {
        core.counters.fall(Gauge::SurfaceBuffers);
        return;
    }
    core.pool.freeing.store(true, Relaxed);
    // SAFETY: as above; nothing else is bound to the memory.
    unsafe { core.device.free_memory(memory.memory, None) };
    core.pool.freeing.store(false, Relaxed);
    core.counters.fall(Gauge::Blocks);
    match memory.own {
        Some(Own::Export(_)) => core.counters.fall(Gauge::Exports),
        Some(Own::Surface { .. }) => {
            core.counters.fall(Gauge::SurfaceBuffers);
            #[cfg(unix)]
            core.counters.fall(Gauge::SurfaceImports);
        }
        _ => core.counters.fall(Gauge::Imports),
    }
    if let Some(release) = release {
        release();
    }
}

/// On the block-maker thread: frees an allocation the idle rule took out, after the consumer's
/// unmap of it, and counts its idle time. The submissions made during its `vkFreeMemory` are
/// counted.
fn free_idle(core: &Core, allocation: Allocation) {
    core.pool.unmap(allocation.id);
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
    (core.counters).fall_by(allocation.class.allocation_gauge(), allocation.len);
    if allocation.spilled {
        core.counters.fall_by(Gauge::SpilledBytes, allocation.len);
    }
}

/// At teardown, after the block-maker thread is joined and every resource is freed: frees the
/// imported and exported blocks given back and runs the imports' releases, then every
/// allocation, each after the
/// consumer's unmap of it where it is mapped.
pub(crate) fn free_all(core: &mut Core) {
    let mut allocations = Vec::new();
    let mut idle = [0; 2];
    let owned: Vec<(Memory, Option<Release>)> = {
        let state = (core.pool.state.get_mut()).unwrap_or_else(PoisonError::into_inner);
        state.owned.drain(..).collect()
    };
    for (memory, release) in owned {
        free_own(core, &memory, release);
    }
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
