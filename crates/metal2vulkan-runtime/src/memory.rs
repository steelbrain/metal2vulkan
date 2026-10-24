//! The runtime's own memory sub-allocator: blocks of `VkDeviceMemory` per memory type, each with a
//! free list of placed ranges.
//!
//! - **One mechanism.** A block with explicit placement serves the runtime's pools for resources
//!   made outside a heap, and later a Metal heap (step 7: a heap is one block, its resources placed
//!   at offsets).
//! - **Pooling.** Resources share blocks of [`BLOCK_BYTES`], so `maxMemoryAllocationCount` (4096
//!   on many devices) stays out of reach. A request larger than half a block takes a dedicated
//!   block of its own size, freed with it. A pooled block that empties is kept for the next
//!   placements while fewer than [`KEPT_EMPTY_BLOCKS`] empty ones are kept in its pool, else freed:
//!   a working set that dips by a block and comes back a frame later (a frame's buffers, returned
//!   as its command buffer completes) makes and frees no block for it. Every kept empty block but
//!   one goes once it stayed empty for [`KEPT_EMPTY_FOR`] ([`Pool::stale`]), and every one of a
//!   memory type goes before a block of that type fails to be made ([`place`]).
//! - **Entry point.** [`allocate`] with the resource's memory requirements and its
//!   [`MemoryClass`]; [`Memory::free`] with the placement it gave. Buffers use it in step 1;
//!   textures and heaps add their classes and callers in their own steps. Imported host memory
//!   has its own entry, [`Memory::import_host`]: one dedicated block per import, never
//!   pooled, freed through the same [`Memory::free`]. So has a buffer's fresh pages
//!   ([`import_own_pages`], `fresh_pages.rs`): a dedicated block that unmaps them once its
//!   memory is freed, as an import through a read-only view of the caller's pages unmaps the view
//!   (`read_only_view.rs`). A dma-buf's import
//!   (`Memory::import_dma_buf`) is in no block: the placement in it keeps it, and frees it as it
//!   goes (`dma_buf.rs`'s `DmaBufPages`).
//!
//! - **Addresses** ([`allocate_addressed`]). A buffer placed for a device address gets it from its
//!   block: a block that holds such a buffer has one buffer over its whole memory with the
//!   device-address usage (`BlockAddresses`), and a placed buffer's address is that buffer's plus
//!   its offset. The placed buffers themselves are made without the usage (`buffer.rs`), so the
//!   device's count of addressable buffers is the count of such blocks, not of the program's
//!   buffers: MoltenVK makes each addressable buffer resident in every encoder whose pipeline reads
//!   pointers, a cost each command buffer paid per live buffer (2.5 ms a commit at 32768 live
//!   buffers on an Apple M4; MoltenVK's `MVKDevice::encodeGPUAddressableBuffers`).
//!
//! - **Freed after the batches in flight** where the driver fails pending work when memory a
//!   device-address buffer lay in is freed (`freed_memory.rs`): a block that goes has its buffer
//!   of addresses destroyed at once, and its memory waits, parked, until the batches and the
//!   device's own submissions made before it completed.
//!
//! - **Images apart**. An image the runtime places in memory of its own takes a block that holds
//!   images alone ([`Holds`], [`allocate_image`]): no buffer, a block's buffer of addresses among
//!   them, is bound into such a block, nor such an image into a buffers' block. An image over a
//!   buffer's or a heap's bytes lies in that buffer's memory, at the offset Metal's layout gives
//!   it. Images are placed at a multiple of `bufferImageGranularity` and rounded up to it
//!   (`texture.rs`'s `ImagePlacement`).

use super::buffer::usage;
use super::device::lock;
#[cfg(unix)]
use super::dma_buf::DmaBufPages;
use super::error::{internal, vulkan, Error, MemoryKind};
use super::freed_memory::{Parked, Parking};
use super::fresh_pages::OwnPages;
use super::memory_log::{self, MemoryLog};
use super::range::RangeAllocator;
use super::read_only_view::ReadOnlyView;
use super::sharing::Sharing;
use super::work_counts::Counters;
use crate::release;
use ash::vk;
use metal2vulkan::vk::DriverId;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The size of a pooled block, at most the device's `maxMemoryAllocationSize`.
pub(super) const BLOCK_BYTES: u64 = 64 << 20;

/// The most empty pooled blocks a pool keeps for its next placements ([`Pool::release`]), at
/// most 128 MiB of empty memory a pool. In a pool the encodings place in, one of them is mostly
/// the block [`keep_room`] made for room: with one kept, every other block that emptied was freed,
/// and a block made again as the next placements filled that one, so a frame's buffers, returned
/// each frame, made and freed a block a frame. With two they make and free none
/// (`runtime_lane::pool_churn`).
pub(super) const KEPT_EMPTY_BLOCKS: usize = 2;

/// How long a kept empty pooled block beyond its pool's one stays empty before it is freed
/// ([`Pool::stale`]): a frame's dip comes back well within it.
pub(super) const KEPT_EMPTY_FOR: Duration = Duration::from_secs(1);

/// The least bytes of a pooled placement the memory log names as it is placed and returned
/// (`memory_log.rs`): a few such placements fill a block, so their comings and goings are what
/// makes and frees blocks.
const LOGGED_PLACEMENT_BYTES: u64 = 4 << 20;

/// What memory a resource needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum MemoryClass {
    /// Host-visible and coherent memory the host reads as well as writes (`Shared` storage of
    /// the default CPU cache mode, staging for the host's reads and writes): host-cached where the
    /// device has such, else device-local where it has such ([`candidates`]). Without unified
    /// memory that is system memory, as a shared resource's on a Mac with a discrete GPU. The
    /// upload arena's chunks take it (`upload.rs`).
    Shared,
    /// Host-visible and coherent memory the GPU uses and the host writes, reading little or
    /// nothing of it (`Shared` storage of the write-combined CPU cache mode, a function table's
    /// words): without unified memory device-local where the device has such
    /// (resizable BAR), else as `Shared`; with unified memory as `Shared` ([`candidates`]).
    WriteCombined,
    /// Device-local (`Private` storage), host-invisible where the device has a choice.
    Private,
    /// Host-visible and coherent (`Managed` storage), host-cached where the device has such
    /// memory: a managed resource is there for the CPU to read back. On a device with unified
    /// memory the GPU uses it too, and coherent memory needs no flush or invalidate, so
    /// `did_modify_range` and `synchronize_resource` record nothing beyond their order. On
    /// one without, it holds a managed buffer's host copy, and the buffer's own bytes are `Private`
    /// (`buffer.rs`'s `Mirror`, `managed.rs`).
    Managed,
    /// Device-local and lazily allocated where the device has such memory (`Memoryless`
    /// storage: a transient attachment's image), else as `Private`.
    Memoryless,
}

/// What a block of the runtime's holds: buffers (a heap's bytes and every buffer of addresses
/// among them), or images, never both. On an RTX 5080 a new texture's image bound into the block
/// that held the running work's buffers, 1024 bytes past a buffer's, waited in
/// `vkBindImageMemory` for that work to end (493 ms and 502 ms behind a 0.5 s kernel), where images
/// bound into a block of images, or a new one, took microseconds. So images take blocks
/// of their own, one pool per memory type beside the buffers'.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Holds {
    Buffers,
    Images,
}

/// Whether a placement may share a pooled block (`Pool::place`), or takes a block of its own
/// whatever its size: memory no submission has used, so its bind waits for none. On an RTX 5080
/// a neutral image bound into a block of images, 1024 bytes past another neutral image that a
/// queued command buffer sampled, waited in `vkBindImageMemory` (NVIDIA's ioctl) until that work
/// ended (490 ms behind a 0.5 s kernel in our probe). On lavapipe the
/// same case waits for its queue lock instead, in making the image's view.
///
/// Other drivers bind without waiting. Measured with a probe outside the runtime: an image bound
/// at the start of a block, a submission of long fills of a large buffer and then a clear of that
/// image, and at once a second image bound 1 MiB into the same block and its view made, each
/// timed, the submission's fence read after both. On ANV (Mesa 26.2.4, the UHD 770 of RPL-S) the
/// bind took 0.2 to 1.8 us and the view 1.8 to 5.1 us; on RADV (Mesa 26.0.8) 0.0 to 0.6 us and
/// 0.6 to 2.0 us on the Radeon 780M and the RX 7900 XTX; the submission still pending after both
/// in every round, as with a fresh block. There ([`binds_wait`]) such an image shares a pooled
/// block ([`allocate_image_while_queued`]), and every submission names one buffer object fewer
/// to the kernel for each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fit {
    Pooled,
    Apart,
}

/// Why [`Memory::import_host`] imported nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ImportFailure {
    /// No memory type of the class is among the allowed ones.
    NoMemoryType,
    /// The allocation of the last type tried failed.
    Vulkan(vk::Result),
}

/// Whether the runtime may allocate memory of a type with `flags`: any type but the
/// device-coherent and device-uncached ones of `VK_AMD_device_coherent_memory`, which only a
/// device with its `deviceCoherentMemory` feature enabled may allocate
/// (`VUID-vkAllocateMemory-deviceCoherentMemory-02790`), and the runtime enables no such feature.
/// RADV lists them whether or not it is enabled (types 13 to 16 on a Radeon 780M),
/// with the properties of its other types beside those bits.
pub(super) fn allocatable(flags: vk::MemoryPropertyFlags) -> bool {
    !flags.intersects(
        vk::MemoryPropertyFlags::DEVICE_COHERENT_AMD | vk::MemoryPropertyFlags::DEVICE_UNCACHED_AMD,
    )
}

/// The memory types among `types` the host caches, as bits: those a host image copy
/// (`texture/host_copy.rs`) reads at the host's speed. The host reads uncached memory slowly, and
/// a host copy reads through the host's mapping of the image's memory: on an RTX 3090
/// `vkCopyImageToMemoryEXT` read a 1920 x 1080 `BGRA8Unorm` image's 8 MiB in 758 ms from
/// device-local host-visible memory (resizable BAR), in 127 ms from uncached system memory and in
/// 2.9 ms from host-cached system memory; on a Radeon 780M in 144 ms from RADV's device-local
/// host-visible type and in 4.6 ms from its host-cached one.
///
/// Only coherent ones: the runtime flushes and invalidates no mapped range, as Metal's `Shared`
/// and `Managed` textures need no flush, so a cached type that is not coherent leaves the image's
/// texels to the driver's own cache upkeep around each host copy. ANV (Mesa 26.2.4) on Arrow
/// Lake-S's Xe-LPG (`deviceID` 0x7d67), which has no last-level cache shared with the GPU, lists
/// such a type (`HOST_CACHED` without `HOST_COHERENT`): with images there, the GPU read stale
/// 64-byte lines of the consumer's macOS guest's textures (4x4 texels of a Tile4 32-bit image; a
/// third of the boot logo's interior), and none once they took its coherent type.
pub(super) fn host_cached(types: &[vk::MemoryType]) -> u32 {
    use vk::MemoryPropertyFlags as P;
    let cached = P::HOST_VISIBLE | P::HOST_COHERENT | P::HOST_CACHED;
    (types.iter().enumerate())
        .filter(|(_, ty)| ty.property_flags.contains(cached))
        .fold(0, |bits, (index, _)| bits | 1 << index)
}

/// The memory types among `types` the host maps without coherence (`HOST_VISIBLE` without
/// `HOST_COHERENT`), as bits: those no image with host-transfer usage takes, cached or not, as
/// the runtime flushes no mapped range ([`host_cached`]).
pub(super) fn host_incoherent(types: &[vk::MemoryType]) -> u32 {
    use vk::MemoryPropertyFlags as P;
    (types.iter().enumerate())
        .filter(|(_, ty)| {
            ty.property_flags.contains(P::HOST_VISIBLE)
                && !ty.property_flags.contains(P::HOST_COHERENT)
        })
        .fold(0, |bits, (index, _)| bits | 1 << index)
}

/// The memory class an image of `class` with host-transfer usage takes in memory the host caches,
/// and its memory types among `allowed`, as bits: the host-cached types ([`host_cached`]) of
/// `class` where it has one among them; else, on a device with unified memory (`unified`), those
/// of `Shared` memory where it has one among them, as Metal's `Shared` and `Managed` textures on
/// unified memory lie in memory the CPU reads at its speed (on a Radeon 780M, which has no
/// device-local type the host caches, in system memory the host caches: the GPU samples, renders
/// into, clears and copies a 1080p image there as fast as in its device-local memory or faster).
/// `None` where it has none: the image takes `class` and every allowed type,
/// as any image of the class does. A discrete GPU with no host-cached memory of the class (NVIDIA
/// on an RTX 3090) so still copies on the host, slowly: a copy through a submission would queue
/// behind the program's work, and its GPU reaches system memory across the bus.
pub(super) fn host_cached_placement(
    types: &[vk::MemoryType],
    allowed: u32,
    class: MemoryClass,
    unified: bool,
) -> Option<(MemoryClass, u32)> {
    let cached = allowed & host_cached(types);
    let has = |class: MemoryClass| !candidates(types, cached, class, unified).is_empty();
    let class = if has(class) {
        class
    } else if unified && has(MemoryClass::Shared) {
        MemoryClass::Shared
    } else {
        return None;
    };
    Some((class, cached))
}

/// The memory types `class` may use among `types` (the bits of `allowed`), best first, on a
/// device with unified memory (`unified`, `DeviceProperties::has_unified_memory`) or without.
/// Only `Memoryless` uses a lazily allocated type, which serves transient attachments alone, and
/// no class uses a type the runtime may not allocate ([`allocatable`]).
///
/// `Shared` and `WriteCombined` memory is host-visible and coherent, as Metal's shared storage
/// needs no flush. `Shared` memory prefers a host-cached type, then a device-local one: the host
/// reads uncached memory slowly (0.14 GB/s from RADV's device-local host-visible type on a Radeon
/// 780M, against 21 GB/s from its cached one; 0.01 GB/s from resizable BAR on an RTX 3090, against
/// 45 GB/s), while an integrated GPU uses either at one speed. A discrete GPU uses host-cached
/// system memory across the bus (on the RTX 3090, 16-byte stores at 1.7 GB/s and 4-byte stores 12
/// bytes apart at 0.08 GB/s, against 425 and 850 GB/s in device-local memory), as a discrete Mac's
/// GPU uses a shared resource; `Device::new_buffer` announces it
/// (`Degradation::SharedInSystemMemory`). So without unified memory `WriteCombined` memory, which
/// the host does not read, prefers a device-local type; with unified memory it is `Shared` memory.
pub(super) fn candidates(
    types: &[vk::MemoryType],
    allowed: u32,
    class: MemoryClass,
    unified: bool,
) -> Vec<u32> {
    use vk::MemoryPropertyFlags as P;
    let usable = |index: usize| allowed & (1 << index) != 0;
    // The flags required; those preferred, the first outweighing the second; those avoided,
    // weighed as the first preference.
    let (required, preferred, avoided) = match class {
        MemoryClass::WriteCombined if !unified => (
            P::HOST_VISIBLE | P::HOST_COHERENT,
            [P::DEVICE_LOCAL, P::empty()],
            P::empty(),
        ),
        MemoryClass::Shared | MemoryClass::WriteCombined => (
            P::HOST_VISIBLE | P::HOST_COHERENT,
            [P::HOST_CACHED, P::DEVICE_LOCAL],
            P::empty(),
        ),
        MemoryClass::Private => (P::DEVICE_LOCAL, [P::empty(); 2], P::HOST_VISIBLE),
        MemoryClass::Memoryless => (
            P::DEVICE_LOCAL,
            [P::LAZILY_ALLOCATED, P::empty()],
            P::HOST_VISIBLE,
        ),
        MemoryClass::Managed => (
            P::HOST_VISIBLE | P::HOST_COHERENT,
            [P::HOST_CACHED, P::empty()],
            P::empty(),
        ),
    };
    let lacks = |flags: P, wanted: P| u8::from(!wanted.is_empty() && !flags.contains(wanted));
    let mut ranked: Vec<((u8, u8), u32)> = types
        .iter()
        .enumerate()
        .filter(|&(index, ty)| {
            usable(index)
                && allocatable(ty.property_flags)
                && ty.property_flags.contains(required)
                && (class == MemoryClass::Memoryless
                    || !ty.property_flags.contains(P::LAZILY_ALLOCATED))
        })
        .map(|(index, ty)| {
            let flags = ty.property_flags;
            let first = lacks(flags, preferred[0])
                + u8::from(!avoided.is_empty() && flags.intersects(avoided));
            ((first, lacks(flags, preferred[1])), index as u32)
        })
        .collect();
    // Vulkan lists memory types best first within equal properties; the sort is stable.
    ranked.sort_by_key(|&(rank, _)| rank);
    ranked.into_iter().map(|(_, index)| index).collect()
}

/// The block bookkeeping of one memory type: pure, so the CPU tests cover it. `T` is what the
/// caller keeps per block (its `VkDeviceMemory` and mapping).
#[derive(Debug)]
pub(super) struct Pool<T> {
    blocks: Vec<Option<PoolBlock<T>>>,
    /// The largest alignment a pooled placement asked of it so far ([`Pool::has_room`]).
    align: u64,
    /// The pooled blocks added and freed so far ([`Pool::turnover`]).
    made: u64,
    freed: u64,
}

#[derive(Debug)]
struct PoolBlock<T> {
    ranges: RangeAllocator,
    dedicated: bool,
    payload: T,
    /// When it emptied, while it is empty and kept ([`Pool::release`]); `None` for a block
    /// added with nothing placed in it, kept as long as it stays empty.
    emptied: Option<Instant>,
}

impl<T> PoolBlock<T> {
    /// Whether it is a pooled block with nothing placed in it, kept for the next placements.
    fn kept(&self) -> bool {
        !self.dedicated && self.ranges.is_empty()
    }
}

impl<T> Default for Pool<T> {
    fn default() -> Pool<T> {
        Pool {
            blocks: Vec::new(),
            align: 1,
            made: 0,
            freed: 0,
        }
    }
}

impl<T> Pool<T> {
    /// A place for `size` bytes aligned to `align` in a pooled block that has room: (block,
    /// offset).
    pub(super) fn place(&mut self, size: u64, align: u64) -> Option<(usize, u64)> {
        self.align = self.align.max(align);
        self.blocks
            .iter_mut()
            .enumerate()
            .find_map(|(index, block)| {
                let block = block.as_mut().filter(|block| !block.dedicated)?;
                let offset = block.ranges.allocate(size, align)?;
                block.emptied = None;
                Some((index, offset))
            })
    }

    /// Add a block of `size` bytes holding `payload`, and place `used` bytes at its start when
    /// given: the block's index.
    pub(super) fn add(&mut self, size: u64, dedicated: bool, payload: T, used: u64) -> usize {
        let mut ranges = RangeAllocator::new(0, size);
        if used > 0 {
            let at = ranges.allocate(used, 1);
            debug_assert_eq!(at, Some(0));
        }
        let block = Some(PoolBlock {
            ranges,
            dedicated,
            payload,
            emptied: None,
        });
        self.made += u64::from(!dedicated);
        match self.blocks.iter().position(Option::is_none) {
            Some(index) => {
                self.blocks[index] = block;
                index
            }
            None => {
                self.blocks.push(block);
                self.blocks.len() - 1
            }
        }
    }

    /// Whether a pooled block has room for `size` bytes at the largest alignment asked of the
    /// pool so far.
    pub(super) fn has_room(&self, size: u64) -> bool {
        (self.blocks.iter().flatten())
            .any(|block| !block.dedicated && block.ranges.longest(self.align) >= size)
    }

    /// Whether block `index` is a live pooled block, not a dedicated one.
    fn pooled(&self, index: usize) -> bool {
        (self.blocks.get(index))
            .and_then(Option::as_ref)
            .is_some_and(|block| !block.dedicated)
    }

    /// The payload of block `index`.
    pub(super) fn payload(&self, index: usize) -> Option<&T> {
        self.blocks.get(index)?.as_ref().map(|block| &block.payload)
    }

    /// The payload of block `index`, to change.
    fn payload_mut(&mut self, index: usize) -> Option<&mut T> {
        self.blocks
            .get_mut(index)?
            .as_mut()
            .map(|block| &mut block.payload)
    }

    /// Return `[offset, offset + size)` of block `index`, at `now`. The block's payload when the
    /// block must now be freed: a dedicated block, or a pooled block that emptied while
    /// [`KEPT_EMPTY_BLOCKS`] other empty pooled blocks are kept.
    pub(super) fn release(
        &mut self,
        index: usize,
        offset: u64,
        size: u64,
        now: Instant,
    ) -> Option<T> {
        let block = self.blocks.get_mut(index)?.as_mut()?;
        block.ranges.free(offset, size);
        if !block.ranges.is_empty() {
            return None;
        }
        let dedicated = block.dedicated;
        let kept = (self.blocks.iter().enumerate())
            .filter(|&(other, block)| other != index && block.as_ref().is_some_and(PoolBlock::kept))
            .count();
        if !dedicated && kept < KEPT_EMPTY_BLOCKS {
            if let Some(block) = self.blocks[index].as_mut() {
                block.emptied = Some(now);
            }
            return None;
        }
        self.take(index)
    }

    /// At `now`, the payload of the kept empty pooled block that emptied earliest, where it
    /// stayed empty for [`KEPT_EMPTY_FOR`], taken out to be freed, while another empty pooled
    /// block stays kept.
    pub(super) fn stale(&mut self, now: Instant) -> Option<T> {
        let kept = (self.blocks.iter().flatten())
            .filter(|block| block.kept())
            .count();
        if kept < 2 {
            return None;
        }
        let (index, _) = (self.blocks.iter().enumerate())
            .filter_map(|(index, block)| {
                let block = block.as_ref().filter(|block| block.kept())?;
                Some((index, block.emptied?))
            })
            .filter(|&(_, at)| now.saturating_duration_since(at) >= KEPT_EMPTY_FOR)
            .min_by_key(|&(_, at)| at)?;
        self.take(index)
    }

    /// Every kept empty pooled block's payload, taken out to be freed.
    pub(super) fn take_kept(&mut self) -> Vec<T> {
        let kept: Vec<usize> = (self.blocks.iter().enumerate())
            .filter(|(_, block)| block.as_ref().is_some_and(PoolBlock::kept))
            .map(|(index, _)| index)
            .collect();
        kept.into_iter()
            .filter_map(|index| self.take(index))
            .collect()
    }

    /// Block `index`'s payload, its slot emptied.
    fn take(&mut self, index: usize) -> Option<T> {
        let block = self.blocks.get_mut(index)?.take()?;
        self.freed += u64::from(!block.dedicated);
        Some(block.payload)
    }

    /// The pooled blocks added and freed so far.
    pub(super) fn turnover(&self) -> (u64, u64) {
        (self.made, self.freed)
    }

    /// Every block's payload, emptying the pool.
    pub(super) fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.blocks.drain(..).flatten().map(|block| block.payload)
    }

    /// The number of live blocks.
    #[cfg(test)]
    pub(super) fn blocks(&self) -> usize {
        self.blocks.iter().flatten().count()
    }

    /// Its live blocks, by kind, into `census`: those whose payload `imported` names count as
    /// imports, whatever else they are.
    fn count_into(&self, census: &mut Census, imported: impl Fn(&T) -> bool) {
        for block in self.blocks.iter().flatten() {
            let count = match (imported(&block.payload), block.dedicated) {
                (true, _) => &mut census.imported,
                (false, true) => &mut census.dedicated,
                (false, false) if block.ranges.is_empty() => &mut census.empty,
                (false, false) => &mut census.pooled,
            };
            *count += 1;
        }
    }
}

/// A device's live `VkDeviceMemory` blocks, by kind ([`Memory::census`]): each one buffer
/// object the driver lists in every submission where it lists every allocation (ANV on i915).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Census {
    /// Pooled blocks with a placement in them.
    pub(super) pooled: u64,
    /// Pooled blocks with none, kept for the next placement ([`Pool::release`]).
    pub(super) empty: u64,
    /// Dedicated blocks: a placement larger than half a block, or one placed apart.
    pub(super) dedicated: u64,
    /// The caller's host memory imported ([`Memory::import_host`]), a block each.
    pub(super) imported: u64,
}

/// A host pointer the runtime may send between threads: the mapping of a block it owns.
#[derive(Clone, Copy, Debug)]
pub(super) struct Mapped(NonNull<u8>);

// SAFETY: the pointer names persistently mapped device memory the runtime owns; who writes
// through it is the resource API's contract (Metal's `contents`).
unsafe impl Send for Mapped {}
// SAFETY: as above.
unsafe impl Sync for Mapped {}

impl Mapped {
    pub(super) fn as_ptr(self) -> *mut u8 {
        self.0.as_ptr()
    }
}

/// One `VkDeviceMemory` block.
#[derive(Debug)]
pub(super) struct Block {
    memory: vk::DeviceMemory,
    /// The bytes its placements take.
    size: u64,
    mapped: Option<Mapped>,
    /// Its buffer of addresses, from the first placement that needs an address (the module
    /// docs).
    addresses: Option<BlockAddresses>,
    /// It is the caller's host memory, imported ([`Memory::import_host`]).
    imported: bool,
    /// The pages its memory was imported from, where the runtime mapped them for it
    /// ([`import_own_pages`]): unmapped once the memory is freed.
    pages: Option<OwnPages>,
    /// The read-only view of the caller's pages its memory was imported from, which drops the
    /// GPU's writes ([`Memory::import_host`], `read_only_view.rs`): unmapped once the memory is
    /// freed.
    view: Option<ReadOnlyView>,
    /// Its memory type.
    type_index: u32,
}

/// What the runtime mapped that a block's memory was imported from (`Block::pages`,
/// `Block::view`), unmapped once the memory is freed.
type MappedPages = (Option<OwnPages>, Option<ReadOnlyView>);

/// A buffer over the first `Block::size` bytes of a block, with every usage of an addressed
/// buffer (`buffer::usage`), whose device address gives each buffer placed in the block its own
/// (the module docs).
#[derive(Debug)]
struct BlockAddresses {
    raw: vk::Buffer,
    address: u64,
}

impl Block {
    /// Free the block's memory, its buffer of addresses first.
    ///
    /// # Safety
    ///
    /// Nothing is placed in the block any more, and no command that reads it is pending.
    unsafe fn release(self, device: &ash::Device) {
        // SAFETY: the caller's contract.
        let (memory, pages) = unsafe { self.release_addresses(device) };
        // SAFETY: the caller's contract; freeing unmaps it.
        unsafe { release::object(device, memory) };
        // The pages it was imported from go after it (`VkImportMemoryHostPointerInfoEXT`).
        drop(pages);
    }

    /// Destroy the block's buffer of addresses: its memory, to free, and the pages or the view
    /// it was imported from where the runtime mapped them, to unmap once it is freed.
    ///
    /// # Safety
    ///
    /// Nothing is placed in the block any more, and no command that reads the buffer is pending.
    unsafe fn release_addresses(
        self,
        device: &ash::Device,
    ) -> (vk::DeviceMemory, Option<MappedPages>) {
        let Block {
            memory,
            addresses,
            pages,
            view,
            ..
        } = self;
        if let Some(addresses) = addresses {
            // SAFETY: the caller's contract.
            unsafe { release::object(device, addresses.raw) };
        }
        let mapped_pages = match (pages, view) {
            (None, None) => None,
            mapped_pages => Some(mapped_pages),
        };
        (memory, mapped_pages)
    }
}

/// A buffer of `size` bytes with every usage of an addressed buffer, shared as `sharing` says,
/// unbound, and the bytes of memory it requires.
fn address_buffer(
    device: &ash::Device,
    size: u64,
    sharing: Sharing<'_>,
) -> Result<(vk::Buffer, u64), Error> {
    let info = sharing.buffer(
        vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage(true)),
    );
    // SAFETY: the device is live and `info` valid.
    let raw = unsafe { crate::creation::buffer(device, &info) }.map_err(vulkan)?;
    // SAFETY: `raw` is a live buffer of this device.
    let required = unsafe { device.get_buffer_memory_requirements(raw) }.size;
    Ok((raw, required))
}

/// `raw` ([`address_buffer`]'s) bound at the start of `memory`, with its device address; released
/// when the binding fails.
///
/// # Safety
///
/// `memory` is the device's, allocated with the device-address flag, of a type `raw` takes and
/// at least as long as `raw` requires.
unsafe fn bind_addresses(
    device: &ash::Device,
    raw: vk::Buffer,
    memory: vk::DeviceMemory,
) -> Result<BlockAddresses, Error> {
    // SAFETY: the caller's contract.
    if let Err(error) = unsafe { crate::creation::bind_buffer(device, raw, memory, 0) } {
        // SAFETY: the buffer is unused.
        unsafe { release::object(device, raw) };
        return Err(vulkan(error));
    }
    let info = vk::BufferDeviceAddressInfo::default().buffer(raw);
    // SAFETY: the buffer has the device-address usage and is bound whole to memory allocated
    // with the device-address flag.
    let address = unsafe { device.get_buffer_device_address(&info) };
    Ok(BlockAddresses { raw, address })
}

/// Where a resource's memory is.
#[derive(Debug)]
pub(super) struct Placement {
    type_index: u32,
    /// What its block holds.
    holds: Holds,
    /// The runtime's block the placement is in; `None` for memory the caller made and keeps
    /// (`Device::new_buffer_with_vk_buffer`), which the runtime never frees.
    block: Option<usize>,
    size: u64,
    pub(super) memory: vk::DeviceMemory,
    pub(super) offset: u64,
    /// The host address of `offset` when the memory is host-visible.
    pub(super) mapped: Option<Mapped>,
    /// The device address of `offset`, from the block's buffer of addresses, where the placement
    /// asked for one ([`allocate_addressed`]) on a device that enables
    /// `bufferDeviceAddress`.
    pub(super) address: Option<u64>,
    /// It lies in pages the runtime mapped and imported ([`import_own_pages`]).
    own_pages: bool,
    /// It lies in an import of a read-only view of the caller's pages (`read_only_view.rs`).
    read_only_view: bool,
    /// The imports of the caller's dma-bufs the placement lies in, with the source that made them
    /// (`dma_buf.rs`), which no block of the runtime's holds: kept until the placement goes, and
    /// freed, the source dropped, when their last holder goes.
    #[cfg(unix)]
    dma_buf: Option<Arc<DmaBufPages>>,
}

impl Placement {
    /// The placement of `size` bytes at `offset` in `memory`, of memory type `type_index`, that
    /// the caller made and keeps (`wrapped_buffer.rs`): no block of the runtime's, so freeing it
    /// returns nothing. `mapped` is the host address of `offset`, where the caller gives one.
    pub(super) fn caller(
        memory: vk::DeviceMemory,
        offset: u64,
        size: u64,
        type_index: u32,
        mapped: Option<NonNull<u8>>,
    ) -> Placement {
        Placement {
            type_index,
            holds: Holds::Buffers,
            block: None,
            size,
            memory,
            offset,
            mapped: mapped.map(Mapped),
            address: None,
            own_pages: false,
            read_only_view: false,
            #[cfg(unix)]
            dma_buf: None,
        }
    }

    /// The placement of a no-copy buffer's `size` bytes of host pages in the imports of the
    /// caller's dma-bufs of them, `pages`: at offset 0 of `memory`, of type `type_index`, where
    /// they are one import, or no memory (`VK_NULL_HANDLE`) where they are several bound to a
    /// sparse buffer; their host address is `mapped`, the pages' pointer (the runtime never maps
    /// a dma-buf). No block of the runtime's, so freeing it returns nothing; it keeps the imports
    /// until it goes.
    #[cfg(unix)]
    pub(super) fn dma_buf(
        pages: Arc<DmaBufPages>,
        memory: vk::DeviceMemory,
        type_index: u32,
        size: u64,
        mapped: NonNull<u8>,
    ) -> Placement {
        Placement {
            type_index,
            holds: Holds::Buffers,
            block: None,
            size,
            memory,
            offset: 0,
            mapped: Some(Mapped(mapped)),
            address: None,
            own_pages: false,
            read_only_view: false,
            dma_buf: Some(pages),
        }
    }

    /// The index of the memory type the placement's memory is of.
    pub(super) fn type_index(&self) -> u32 {
        self.type_index
    }

    /// The bytes placed.
    pub(super) fn size(&self) -> u64 {
        self.size
    }

    /// Whether the placement lies in imports of the caller's dma-bufs (`dma_buf.rs`), to which no
    /// image is bound (an image over them is bound to a dma-buf of its own pages).
    pub(super) fn over_dma_buf(&self) -> bool {
        #[cfg(unix)]
        {
            self.dma_buf.is_some()
        }
        #[cfg(not(unix))]
        {
            false
        }
    }

    /// Whether the placement lies in pages the runtime mapped and imported
    /// ([`import_own_pages`]), to which no image is bound (`fresh_pages.rs`).
    pub(super) fn over_own_pages(&self) -> bool {
        self.own_pages
    }

    /// Whether the placement lies in an import of a read-only view of the caller's pages, which
    /// drops the GPU's writes to them (`read_only_view.rs`).
    pub(super) fn over_read_only_view(&self) -> bool {
        self.read_only_view
    }

    /// The imports of the caller's dma-bufs the placement lies in, with their source
    /// (`dma_buf.rs`); `None` for any other placement.
    #[cfg(unix)]
    pub(super) fn dma_buf_pages(&self) -> Option<&Arc<DmaBufPages>> {
        self.dma_buf.as_ref()
    }

    /// Wait until the sparse binding of the buffer over the placement's dma-bufs completed,
    /// where it may not have (`dma_buf.rs`'s `DmaBufPages::settle`): before that buffer is
    /// destroyed. Any other placement has nothing to wait for.
    pub(super) fn settle(&self) {
        #[cfg(unix)]
        if let Some(pages) = &self.dma_buf {
            pages.settle();
        }
    }

    /// The same placement, for another owner to free from now on: an image's first memory, which
    /// goes with its first `VkImage` once the image takes its storage form
    /// (`texture/storage_form.rs`). The image frees its storage form's memory instead, never
    /// this one. Or a parked image's or buffer's memory, which the recycler frees as it lets the
    /// object go, in place of its owner (`recycle.rs`).
    pub(super) fn handed_over(&self) -> Placement {
        Placement {
            type_index: self.type_index,
            holds: self.holds,
            block: self.block,
            size: self.size,
            memory: self.memory,
            offset: self.offset,
            mapped: self.mapped,
            address: self.address,
            own_pages: self.own_pages,
            read_only_view: self.read_only_view,
            #[cfg(unix)]
            dma_buf: self.dma_buf.clone(),
        }
    }
}

/// The device's memory: one [`Pool`] per memory type for buffers, and one for images.
pub(super) struct Memory {
    types: Vec<vk::MemoryType>,
    /// The buffers' pools by memory type, then the images' ([`Memory::pool`]).
    pools: Vec<Pool<Block>>,
    block_bytes: u64,
    /// Blocks are allocated with `VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT`: the device enabled
    /// `bufferDeviceAddress`, so buffers carry device addresses.
    device_address: bool,
    /// The memory types a buffer of addresses takes, as bits, once asked
    /// ([`allocate_addressed`]).
    address_types: Option<u32>,
    /// The pools the runtime's encodings and commits place their own resources in, once asked
    /// ([`Memory::kept`], [`keep_room`]).
    kept: Option<Vec<(Holds, u32)>>,
    /// The device has unified memory (`DeviceProperties::has_unified_memory`), which
    /// [`candidates`] ranks `WriteCombined` memory by.
    unified: bool,
    /// The device's queue families, which its buffers of addresses are shared among
    /// (`sharing.rs`).
    families: Vec<u32>,
    /// The bytes of the blocks allocated so far ([`Memory::allocated`]).
    allocated: u64,
    /// Where the device's driver fails pending work when memory a device-address buffer lay in
    /// is freed, the record its blocks that go wait by (`freed_memory.rs`).
    parking: Option<Arc<Parking>>,
    /// The blocks that went and wait, their buffers of addresses destroyed.
    parked: Vec<Parked>,
    /// The blocks that waited so far ([`Memory::parked`]).
    parked_count: u64,
    /// The memory log, where `METAL2VULKAN_LOG_MEMORY` is set (`memory_log.rs`).
    log: Option<MemoryLog>,
    /// A bind into memory that queued work uses waits for that work on the device's driver
    /// ([`binds_wait`]), so an image bound while such work may be queued takes a block of its own
    /// ([`allocate_image_while_queued`]).
    binds_wait: bool,
}

impl Memory {
    /// The memory of a device with `properties`, whose largest allocation is
    /// `max_allocation` bytes, with unified memory or without (`unified`), whose queue families
    /// are `families`, whose driver is `driver`.
    pub(super) fn new(
        properties: &vk::PhysicalDeviceMemoryProperties,
        max_allocation: u64,
        device_address: bool,
        unified: bool,
        families: Vec<u32>,
        parking: Option<Arc<Parking>>,
        driver: Option<DriverId>,
    ) -> Memory {
        let count = properties.memory_type_count as usize;
        let types = properties.memory_types[..count.min(vk::MAX_MEMORY_TYPES)].to_vec();
        Memory {
            pools: (types.iter().chain(&types))
                .map(|_| Pool::default())
                .collect(),
            types,
            block_bytes: BLOCK_BYTES.min(max_allocation.max(1)),
            device_address,
            address_types: None,
            kept: None,
            unified,
            families,
            allocated: 0,
            parking,
            parked: Vec::new(),
            parked_count: 0,
            log: None,
            binds_wait: binds_wait(driver),
        }
    }

    /// Log each `VkDeviceMemory` allocated, imported or freed from now on to `log`
    /// (`memory_log.rs`).
    pub(super) fn log_to(&mut self, log: Option<MemoryLog>) {
        self.log = log;
    }

    /// Say `what` happened to `block`, where the log is on: its device address from its buffer of
    /// addresses, else none (`memory_log.rs`).
    fn say_block(&self, what: &str, block: &Block) {
        let Some(log) = &self.log else {
            return;
        };
        let address = block.addresses.as_ref().map(|addresses| addresses.address);
        log.say(what, block.memory, block.type_index, block.size, address);
    }

    /// Say `what` happened to `placement`, a placement of `class` where given, where the log is on
    /// and the placement is in a pooled block and at least [`LOGGED_PLACEMENT_BYTES`] long.
    fn say_placement(&self, what: &str, placement: &Placement, class: Option<MemoryClass>) {
        let Some(log) = &self.log else {
            return;
        };
        let pooled = placement.block.is_some_and(|index| {
            self.pool_ref(placement.holds, placement.type_index)
                .pooled(index)
        });
        if !pooled || placement.size < LOGGED_PLACEMENT_BYTES {
            return;
        }
        log.line(memory_log::placement_line(
            what,
            placement.size,
            class,
            placement.holds == Holds::Images,
            (placement.memory, placement.offset, placement.type_index),
        ));
    }

    /// The blocks that waited for the batches in flight as they went, so far, and those that
    /// wait now (`freed_memory.rs`).
    pub(super) fn parked(&self) -> (u64, u64) {
        (self.parked_count, self.parked.len() as u64)
    }

    /// Let the blocks that go from now on wait by `parking` (`freed_memory.rs`), as where the
    /// device's driver has the defect: the crate's tests' way to take that path on any device.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(super) fn park_by(&mut self, parking: Arc<Parking>) {
        self.parking = Some(parking);
    }

    /// The bytes of `VkDeviceMemory` allocated so far for blocks of the runtime's own memory,
    /// pooled or dedicated, whether freed since or not: host memory imported counts in none.
    pub(super) fn allocated(&self) -> u64 {
        self.allocated
    }

    /// Its live blocks, by kind.
    pub(super) fn census(&self) -> Census {
        let mut census = Census::default();
        for pool in &self.pools {
            pool.count_into(&mut census, |block| block.imported);
        }
        census
    }

    /// Whether one of the memory types `allowed` names is of `class`.
    pub(super) fn has_type(&self, allowed: u32, class: MemoryClass) -> bool {
        self.class_types(allowed, class) != 0
    }

    /// Whether an image of `class` with host-transfer usage may lie in memory the host caches
    /// ([`host_cached_placement`]), whatever types its requirements allow.
    pub(super) fn has_host_cached(&self, class: MemoryClass) -> bool {
        host_cached_placement(&self.types, u32::MAX, class, self.unified).is_some()
    }

    /// The memory class an image of `class` with host-transfer usage takes in memory the host
    /// caches, and its memory types among `allowed`, as bits ([`host_cached_placement`]).
    pub(super) fn host_cached_placement(
        &self,
        allowed: u32,
        class: MemoryClass,
    ) -> Option<(MemoryClass, u32)> {
        host_cached_placement(&self.types, allowed, class, self.unified)
    }

    /// The memory types the host maps without coherence, as bits ([`host_incoherent`]).
    pub(super) fn host_incoherent(&self) -> u32 {
        host_incoherent(&self.types)
    }

    /// The memory types of `class` among those `allowed` names, as bits.
    pub(super) fn class_types(&self, allowed: u32, class: MemoryClass) -> u32 {
        candidates(&self.types, allowed, class, self.unified)
            .into_iter()
            .fold(0, |bits, index| bits | 1 << index)
    }

    /// The index in `pools` of the pool of blocks of memory type `type_index` that hold `holds`.
    fn pool_at(&self, holds: Holds, type_index: u32) -> usize {
        type_index as usize + usize::from(holds == Holds::Images) * self.types.len()
    }

    /// The pool of blocks of memory type `type_index` that hold `holds`.
    fn pool(&mut self, holds: Holds, type_index: u32) -> &mut Pool<Block> {
        let at = self.pool_at(holds, type_index);
        &mut self.pools[at]
    }

    /// [`pool`](Self::pool), to read.
    fn pool_ref(&self, holds: Holds, type_index: u32) -> &Pool<Block> {
        &self.pools[self.pool_at(holds, type_index)]
    }

    /// The memory types a buffer of addresses takes, as bits, asked once: `None` without
    /// `bufferDeviceAddress`, where blocks carry no buffer of addresses.
    fn address_types(&mut self, device: &ash::Device) -> Result<Option<u32>, Error> {
        if !self.device_address {
            return Ok(None);
        }
        if self.address_types.is_none() {
            let sharing = Sharing::new(&self.families);
            self.address_types = Some(buffer_types(device, usage(true), sharing)?);
        }
        Ok(self.address_types)
    }

    /// `size` bytes aligned to `align` in a pooled block of `pool` that has room, in a block
    /// with a buffer of addresses where `addressed` gives the counts of such buffers made:
    /// `None` where no block has room.
    fn place_pooled(
        &mut self,
        device: &ash::Device,
        (holds, type_index): (Holds, u32),
        (size, align): (u64, u64),
        addressed: Option<&Counters>,
        class: MemoryClass,
    ) -> Result<Option<Placement>, Error> {
        let Some((block, offset)) = self.pool(holds, type_index).place(size, align) else {
            return Ok(None);
        };
        if let Some(counts) = addressed {
            if let Err(error) = self.address_block(device, type_index, block, counts) {
                drop(self.release_range(device, (holds, type_index), block, offset, size, None));
                return Err(error);
            }
        }
        let addressed = addressed.is_some();
        let placement = self.placement((holds, type_index), block, offset, size, addressed);
        self.say_placement("placed", &placement, Some(class));
        Ok(Some(placement))
    }

    /// Block `block` of `size` bytes, just made, added to the pool of memory type `type_index`
    /// that holds `holds`, dedicated or pooled, with `used` bytes at its start placed: its index.
    fn add_block(
        &mut self,
        (holds, type_index): (Holds, u32),
        block: Block,
        size: u64,
        dedicated: bool,
        used: u64,
    ) -> usize {
        if self.log.is_some() {
            let what = match (dedicated, holds) {
                (false, Holds::Buffers) => "made a pooled block of buffers",
                (false, Holds::Images) => "made a pooled block of images",
                (true, Holds::Buffers) => "made a dedicated block of a buffer",
                (true, Holds::Images) => "made a dedicated block of an image",
            };
            self.say_block(what, &block);
        }
        self.allocated += size;
        self.pool(holds, type_index)
            .add(size, dedicated, block, used)
    }

    /// What making a block takes, copied out so that the block is made with the memory's lock
    /// released ([`place`], [`keep_room`]).
    fn maker(&self) -> BlockMaker {
        let host_visible = (self.types.iter().enumerate())
            .filter(|(_, ty)| (ty.property_flags).contains(vk::MemoryPropertyFlags::HOST_VISIBLE))
            .fold(0, |bits, (index, _)| bits | 1 << index);
        BlockMaker {
            block_bytes: self.block_bytes,
            host_visible,
            device_address: self.device_address,
            families: self.families.clone(),
        }
    }

    /// The pools the runtime's encodings and commits place their own resources in, found once:
    /// of buffers, the best memory type of `Private`, `WriteCombined` and `Shared` memory among
    /// those a buffer the runtime places takes (`buffer::usage(false)`, its address from a buffer
    /// of addresses, [`allocate_addressed`]); of images, the best `Private` type among those a
    /// color image of optimal tiling takes, which every such image takes (the Vulkan
    /// specification's `VkMemoryRequirements`).
    fn kept(&mut self, device: &ash::Device) -> Result<Vec<(Holds, u32)>, Error> {
        if let Some(kept) = &self.kept {
            return Ok(kept.clone());
        }
        let addresses = self.address_types(device)?.unwrap_or(u32::MAX);
        let sharing = Sharing::new(&self.families);
        let buffers = buffer_types(device, usage(false), sharing)? & addresses;
        let images = color_image_types(device, sharing)?;
        let best = |bits: u32, class: MemoryClass| {
            candidates(&self.types, bits, class, self.unified)
                .first()
                .copied()
        };
        let mut kept = Vec::new();
        let classes = [
            MemoryClass::Private,
            MemoryClass::WriteCombined,
            MemoryClass::Shared,
        ];
        let buffer_pools = (classes.into_iter())
            .filter_map(|class| best(buffers, class))
            .map(|type_index| (Holds::Buffers, type_index));
        let image_pool =
            best(images, MemoryClass::Private).map(|type_index| (Holds::Images, type_index));
        for pool in buffer_pools.chain(image_pool) {
            if !kept.contains(&pool) {
                kept.push(pool);
            }
        }
        self.kept = Some(kept.clone());
        Ok(kept)
    }

    /// The placement of `size` bytes at `offset` in block `block` of memory type `type_index`,
    /// with the block's device address of `offset` where `addressed`. A placement that asked for
    /// none has none, though its block has a buffer of addresses: its buffer has an address of
    /// its own where it needs one (a preprocess buffer's must be its own, of its usage).
    fn placement(
        &self,
        (holds, type_index): (Holds, u32),
        block: usize,
        offset: u64,
        size: u64,
        addressed: bool,
    ) -> Placement {
        let payload = (self.pool_ref(holds, type_index))
            .payload(block)
            .expect("a placed block is live");
        Placement {
            type_index,
            holds,
            block: Some(block),
            size,
            memory: payload.memory,
            offset,
            mapped: payload.mapped.map(|mapped| {
                // SAFETY: `offset + size` is inside the block, which is mapped whole.
                Mapped(unsafe { NonNull::new_unchecked(mapped.as_ptr().add(offset as usize)) })
            }),
            address: payload
                .addresses
                .as_ref()
                .filter(|_| addressed)
                .map(|addresses| addresses.address + offset),
            own_pages: payload.pages.is_some(),
            read_only_view: payload.view.is_some(),
            #[cfg(unix)]
            dma_buf: None,
        }
    }

    /// Give pooled block `block` of memory type `type_index` a buffer of addresses where it has
    /// none: over its whole size, which it takes since the block is of a type such a buffer
    /// takes ([`allocate_addressed`]); refused (an invariant) where the buffer would
    /// need more bytes than the block has.
    fn address_block(
        &mut self,
        device: &ash::Device,
        type_index: u32,
        block: usize,
        counts: &Counters,
    ) -> Result<(), Error> {
        let at = self.pool_at(Holds::Buffers, type_index);
        let sharing = Sharing::new(&self.families);
        let payload = self.pools[at]
            .payload_mut(block)
            .expect("a placed block is live");
        if payload.addresses.is_some() {
            return Ok(());
        }
        let (raw, required) = address_buffer(device, payload.size, sharing)?;
        if required > payload.size {
            // SAFETY: the buffer is unused.
            unsafe { release::object(device, raw) };
            return Err(internal(
                "a buffer over a whole memory block requires more bytes than the block has",
            ));
        }
        // SAFETY: the block's memory is the device's, allocated with the device-address flag
        // (`device_address`), of a type the buffer takes, and as long as the buffer requires.
        payload.addresses = Some(unsafe { bind_addresses(device, raw, payload.memory) }?);
        counts.buffer_made(usage(true));
        Ok(())
    }

    /// Return `[offset, offset + size)` of block `block` of memory type `type_index`, freeing the
    /// block when it empties ([`Pool::release`]); `after` goes once it is freed. `after`, given
    /// back for the caller to let go of with no lock held, unless the block waits with it. A kept
    /// empty block of the pool that stayed empty too long goes too ([`Pool::stale`]).
    fn release_range(
        &mut self,
        device: &ash::Device,
        (holds, type_index): (Holds, u32),
        block: usize,
        offset: u64,
        size: u64,
        after: Option<Box<dyn Send>>,
    ) -> Option<Box<dyn Send>> {
        let now = Instant::now();
        let pool = self.pool(holds, type_index);
        let (freed, stale) = (pool.release(block, offset, size, now), pool.stale(now));
        if let Some(stale) = stale {
            // SAFETY: nothing is placed in a kept empty block, and no command reads it.
            drop(unsafe { self.let_go(device, stale, None) });
        }
        match freed {
            // SAFETY: no resource is placed in the block any more.
            Some(block) => unsafe { self.let_go(device, block, after) },
            None => after,
        }
    }

    /// Release every kept empty pooled block of memory type `type_index`, of buffers and of
    /// images, as a block of that type fails to be made ([`place`]): how many went.
    fn release_kept(&mut self, device: &ash::Device, type_index: u32) -> usize {
        let blocks: Vec<Block> = [Holds::Buffers, Holds::Images]
            .into_iter()
            .flat_map(|holds| self.pool(holds, type_index).take_kept())
            .collect();
        let count = blocks.len();
        for block in blocks {
            // SAFETY: nothing is placed in a kept empty block, and no command reads it.
            drop(unsafe { self.let_go(device, block, None) });
        }
        count
    }

    /// The pooled blocks made and freed so far, of every pool.
    pub(super) fn turnover(&self) -> (u64, u64) {
        (self.pools.iter().map(Pool::turnover))
            .fold((0, 0), |(made, freed), (m, f)| (made + m, freed + f))
    }

    /// Free `block`, or where the device parks freed memory (`freed_memory.rs`), destroy its
    /// buffer of addresses and let its memory wait, with `after`, for the batches submitted
    /// before now, unless they completed. `after`, given back where the memory was freed here.
    ///
    /// # Safety
    ///
    /// Nothing is placed in the block any more, and no command that reads it is pending.
    unsafe fn let_go(
        &mut self,
        device: &ash::Device,
        block: Block,
        after: Option<Box<dyn Send>>,
    ) -> Option<Box<dyn Send>> {
        let Some(parking) = &self.parking else {
            self.say_block("freed a block", &block);
            // SAFETY: the caller's.
            unsafe { block.release(device) };
            return after;
        };
        self.say_block(
            "a block goes, freed once the batches in flight complete",
            &block,
        );
        // Out of the driver's list first, with no submission inside its `vkQueueSubmit`, whose
        // encoding may list it (`freed_memory.rs`), then the batches that may list it noted.
        // SAFETY: the caller's.
        let ((memory, pages), noted) =
            parking.note_destroying(|| unsafe { block.release_addresses(device) });
        // The pages the memory was imported from go after it, with what goes once it is freed.
        let after = match (pages, after) {
            (None, after) => after,
            (Some(pages), None) => Some(Box::new(pages) as Box<dyn Send>),
            (Some(pages), Some(after)) => Some(Box::new((pages, after)) as Box<dyn Send>),
        };
        if !parking.reached(&noted) {
            parking.waits();
            // Checked again once counted: a waiter that published a value before the count did
            // not see it, so the value is seen here (both sequentially consistent).
            if !parking.reached(&noted) {
                self.parked_count += 1;
                self.parked.push(Parked {
                    noted,
                    memory,
                    after,
                });
                return None;
            }
            parking.freed(1);
        }
        // SAFETY: the caller's; every batch that may list it completed.
        unsafe { release::object(device, memory) };
        after
    }

    /// The blocks that wait whose values were reached, out of the list, to free with no lock held
    /// (`freed_memory.rs`).
    pub(super) fn reached(&mut self) -> Vec<Parked> {
        let Some(parking) = &self.parking else {
            return Vec::new();
        };
        let reached: Vec<Parked> = self
            .parked
            .extract_if(.., |parked| parking.reached(&parked.noted))
            .collect();
        parking.freed(reached.len());
        reached
    }

    /// Import the caller's host memory, `size` bytes at `pointer` (both multiples of the device's
    /// import alignment), as a dedicated block of the best memory type of `class` among
    /// `allowed` (the types both the pointer and the resource take): its placement at offset 0,
    /// whose `mapped` is `pointer`. Where `view` is given, a read-only view of those pages
    /// (`read_only_view.rs`), the view is imported in their place, and unmapped once the memory
    /// is freed. The block is never pooled and is freed with the placement, as a dedicated block
    /// is (`host_memory.rs`). Refused when no type of `class` is among `allowed`, else with the
    /// failure of the last type tried, the view unmapped.
    ///
    /// # Safety
    ///
    /// The `size` bytes at `pointer` are host memory of this process that stays mapped until the
    /// placement is freed; `view`, where given, views them whole.
    pub(super) unsafe fn import_host(
        &mut self,
        device: &ash::Device,
        (pointer, view): (NonNull<u8>, Option<ReadOnlyView>),
        size: u64,
        allowed: u32,
        class: MemoryClass,
    ) -> Result<Placement, ImportFailure> {
        let types = candidates(&self.types, allowed, class, self.unified);
        let imported = view.as_ref().map_or(pointer, ReadOnlyView::pointer);
        // SAFETY: the caller's contract; the view stays mapped while the block holds it.
        let (memory, type_index) =
            unsafe { import_pointer(device, imported, size, &types, self.device_address) }?;
        let block = Block {
            memory,
            size,
            mapped: Some(Mapped(pointer)),
            addresses: None,
            imported: true,
            pages: None,
            view,
            type_index,
        };
        let host = pointer.as_ptr() as usize;
        self.say_block(&format!("imported host memory 0x{host:x}"), &block);
        Ok(self.add_import(block, type_index, size))
    }

    /// The best memory type of `class` among `allowed`, where it has one.
    pub(super) fn best_type(&self, allowed: u32, class: MemoryClass) -> Option<u32> {
        candidates(&self.types, allowed, class, self.unified)
            .first()
            .copied()
    }

    /// `block`, `size` bytes of memory type `type_index` just imported, added to its pool as a
    /// dedicated block: its placement at offset 0.
    fn add_import(&mut self, block: Block, type_index: u32, size: u64) -> Placement {
        let index = self
            .pool(Holds::Buffers, type_index)
            .add(size, true, block, size);
        self.placement((Holds::Buffers, type_index), index, 0, size, false)
    }

    /// Import the dma-buf `fd`, `size` bytes, as memory of the best type of `class` among
    /// `allowed` (the types both the dma-buf and what is bound in it take): the memory and its
    /// type, the caller's to free (`dma_buf.rs`'s `DmaBufPages`), in no block and never mapped.
    /// For a buffer, allocated with the device-address flag where the device enables
    /// `bufferDeviceAddress`; for `image`, its dedicated allocation (`size` its requirements').
    /// Refused when no type of `class` is among `allowed`, else with the failure of the last type
    /// tried.
    ///
    /// Each attempt hands the driver a **clone** of `fd`, never `fd` itself, and never closes a
    /// clone it has handed over. Vulkan takes a descriptor's ownership only on a successful
    /// import ("Importing memory from a file descriptor transfers ownership of the file descriptor
    /// from the application to the Vulkan implementation. The application must not perform any
    /// operations on the file descriptor after a successful import", `VkImportMemoryFdInfoKHR`),
    /// so on failure it is the application's to close -- but NVIDIA's 615.71.09 closes it anyway
    /// (measured: `F_GETFD` on it reads `EBADF` the instant `vkAllocateMemory` returns
    /// `ERROR_OUT_OF_DEVICE_MEMORY`), and closing it again aborts the process on Rust's I/O safety
    /// check. Whether the number is still ours cannot be asked without a race, since the kernel
    /// may have given it to another thread's `open` by then, so a handed-over clone is let go of
    /// instead: one descriptor per failed import, in place of a process that dies. `fd` itself is
    /// closed here as it always was, once, on the way out.
    #[cfg(unix)]
    pub(super) fn import_dma_buf(
        &self,
        device: &ash::Device,
        fd: std::os::fd::OwnedFd,
        size: u64,
        (allowed, class): (u32, MemoryClass),
        image: Option<vk::Image>,
    ) -> Result<(vk::DeviceMemory, u32), ImportFailure> {
        use std::os::fd::{AsRawFd, IntoRawFd};
        let mut failure = ImportFailure::NoMemoryType;
        for type_index in candidates(&self.types, allowed, class, self.unified) {
            // The descriptor the driver takes, which it owns once the import succeeds and may
            // close even when it fails (the doc above): never `fd`, and never closed here.
            let handed = match fd.try_clone() {
                Ok(handed) => handed,
                Err(_) => {
                    failure = ImportFailure::Vulkan(vk::Result::ERROR_TOO_MANY_OBJECTS);
                    break;
                }
            };
            let mut import = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
                .fd(handed.as_raw_fd());
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default();
            let mut info = vk::MemoryAllocateInfo::default()
                .allocation_size(size)
                .memory_type_index(type_index)
                .push_next(&mut import);
            match image {
                Some(image) => {
                    dedicated = dedicated.image(image);
                    info = info.push_next(&mut dedicated);
                }
                None if self.device_address => info = info.push_next(&mut flags),
                None => {}
            }
            // SAFETY: `info` and its chain live for the call; the type index is the device's and
            // allows the dma-buf (and the image, unbound and made for dma-bufs, whose size this
            // is). The descriptor it takes is the clone, whose fate is the driver's either way.
            match unsafe { crate::creation::allocate_memory(device, &info) } {
                Ok(memory) => {
                    // A successful import owns the descriptor: Vulkan closes it with the memory.
                    let _ = handed.into_raw_fd();
                    if let Some(log) = &self.log {
                        let what = match image {
                            Some(_) => "imported a dma-buf for an image",
                            None => "imported a dma-buf",
                        };
                        log.say(what, memory, type_index, size, None);
                    }
                    return Ok((memory, type_index));
                }
                Err(error) => {
                    // Let go of it rather than close it: a driver that took it on a failure has
                    // closed it, and the number may be another thread's by now (the doc above).
                    let _ = handed.into_raw_fd();
                    failure = ImportFailure::Vulkan(error);
                }
            }
        }
        Err(failure)
    }

    /// The property flags of memory type `index`, where the device has it.
    pub(super) fn type_flags(&self, index: u32) -> Option<vk::MemoryPropertyFlags> {
        self.types
            .get(index as usize)
            .map(|memory_type| memory_type.property_flags)
    }

    /// Whether `placement` lies in device-local memory.
    pub(super) fn is_device_local(&self, placement: &Placement) -> bool {
        self.type_flags(placement.type_index)
            .is_some_and(|flags| flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
    }

    /// Return `placement`'s memory, freeing its block when it empties.
    pub(super) fn free(&mut self, device: &ash::Device, placement: &Placement) {
        drop(self.release_then(device, placement, None));
    }

    /// [`free`](Self::free), and `after` goes once the block is freed: given back for the caller
    /// to let go of with no lock held, unless the block waits with it (`freed_memory.rs`).
    pub(super) fn release_then(
        &mut self,
        device: &ash::Device,
        placement: &Placement,
        after: Option<Box<dyn Send>>,
    ) -> Option<Box<dyn Send>> {
        // The caller's memory is the caller's to free.
        let Some(index) = placement.block else {
            return after;
        };
        self.say_placement("returned", placement, None);
        self.release_range(
            device,
            (placement.holds, placement.type_index),
            index,
            placement.offset,
            placement.size,
            after,
        )
    }

    /// Free every block, those that wait too: the device is being destroyed, its queues idle,
    /// and no resource is left. What waited with them, for the caller to let go of with no lock
    /// held.
    pub(super) fn destroy(&mut self, device: &ash::Device) -> Vec<Box<dyn Send>> {
        let blocks: Vec<Block> = (self.pools.iter_mut())
            .flat_map(|pool| pool.drain())
            .collect();
        for block in blocks {
            self.say_block("freed a block as the device is destroyed", &block);
            // SAFETY: no resource is left, and the device is idle.
            unsafe { block.release(device) };
        }
        let mut after = Vec::new();
        for parked in self.parked.drain(..) {
            // SAFETY: nothing is placed in it, and the device is idle.
            after.extend(unsafe { parked.release(device) });
        }
        after
    }
}

/// What making a block of the runtime's memory takes ([`Memory::maker`]): a copy of the memory's
/// constants, so that the block is made with its lock released.
struct BlockMaker {
    /// A pooled block's size ([`BLOCK_BYTES`], or the device's largest allocation).
    block_bytes: u64,
    /// The host-visible memory types, as bits: a block of one is mapped whole.
    host_visible: u32,
    /// As [`Memory`]'s.
    device_address: bool,
    /// As [`Memory`]'s.
    families: Vec<u32>,
}

impl BlockMaker {
    /// A new block of `size` bytes of memory type `type_index`, with a buffer of addresses where
    /// `addressed` gives the counts of such buffers made (its memory then as long as that buffer
    /// requires, if longer).
    fn make(
        &self,
        device: &ash::Device,
        type_index: u32,
        size: u64,
        addressed: Option<&Counters>,
    ) -> Result<Block, Error> {
        let addresses = if addressed.is_some() && self.device_address {
            Some(address_buffer(device, size, Sharing::new(&self.families))?)
        } else {
            None
        };
        let release_addresses = || {
            if let Some((raw, _)) = addresses {
                // SAFETY: the buffer is unused.
                unsafe { release::object(device, raw) };
            }
        };
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(addresses.map_or(size, |(_, required)| size.max(required)))
            .memory_type_index(type_index);
        if self.device_address {
            info = info.push_next(&mut flags);
        }
        // SAFETY: `info` and its chain live for the call; the type index is the device's.
        let memory = match unsafe { crate::creation::allocate_memory(device, &info) } {
            Ok(memory) => memory,
            Err(error) => {
                release_addresses();
                return Err(vulkan(error));
            }
        };
        let host_visible = self.host_visible & 1 << type_index != 0;
        let mapped = if host_visible {
            // SAFETY: the memory is host-visible, unmapped, and mapped whole once.
            match unsafe {
                device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            } {
                Ok(pointer) => NonNull::new(pointer.cast::<u8>()).map(Mapped),
                Err(error) => {
                    release_addresses();
                    // SAFETY: nothing uses the memory yet.
                    unsafe { release::object(device, memory) };
                    return Err(vulkan(error));
                }
            }
        } else {
            None
        };
        let addresses = match addresses {
            // SAFETY: the memory was just allocated with the device-address flag, of a type the
            // buffer takes (`allocate_addressed`), and as long as it requires.
            Some((raw, _)) => match unsafe { bind_addresses(device, raw, memory) } {
                Ok(addresses) => {
                    if let Some(counts) = addressed {
                        counts.buffer_made(usage(true));
                    }
                    Some(addresses)
                }
                Err(error) => {
                    // SAFETY: nothing uses the memory yet; freeing unmaps it.
                    unsafe { release::object(device, memory) };
                    return Err(error);
                }
            },
            None => None,
        };
        Ok(Block {
            memory,
            size,
            mapped,
            addresses,
            imported: false,
            pages: None,
            view: None,
            type_index,
        })
    }
}

/// The memory types a buffer of `usage` takes, as bits: those of one made for the query and
/// destroyed at once, with no memory bound. Every buffer of one usage takes the same types.
fn buffer_types(
    device: &ash::Device,
    usage: vk::BufferUsageFlags,
    sharing: Sharing<'_>,
) -> Result<u32, Error> {
    let info = sharing.buffer(vk::BufferCreateInfo::default().size(16).usage(usage));
    // SAFETY: the device is live and `info` valid.
    let raw = unsafe { crate::creation::buffer(device, &info) }.map_err(vulkan)?;
    // SAFETY: `raw` is a live buffer of this device, destroyed right after the query with no
    // memory bound.
    Ok(unsafe {
        let types = device.get_buffer_memory_requirements(raw).memory_type_bits;
        release::object(device, raw);
        types
    })
}

/// The memory types a color image of optimal tiling takes, as bits: those of a 1 x 1
/// `R8G8B8A8_UNORM` image made for the query and destroyed at once, with no memory bound.
fn color_image_types(device: &ash::Device, sharing: Sharing<'_>) -> Result<u32, Error> {
    let info = sharing.image(
        vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .initial_layout(vk::ImageLayout::UNDEFINED),
    );
    // SAFETY: the device is live and `info` valid: the format is sampled and a transfer
    // destination with optimal tiling on every device.
    let raw = unsafe { crate::creation::image(device, &info) }?;
    // SAFETY: `raw` is a live image of this device, destroyed right after the query with no
    // memory bound.
    Ok(unsafe {
        let types = device.get_image_memory_requirements(raw).memory_type_bits;
        release::object(device, raw);
        types
    })
}

/// Import `pages`, which the runtime mapped for a buffer of its own (`fresh_pages.rs`), as
/// memory of type `type_index` in `memory`, as [`Memory::import_host`] imports the caller's: a
/// dedicated block, which unmaps the pages once its memory is freed, and its placement at
/// offset 0. The import is made with `memory`'s lock released, as a new block is ([`place`]).
/// Refused with the driver's failure, the pages unmapped.
pub(super) fn import_own_pages(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    pages: OwnPages,
    type_index: u32,
) -> Result<Placement, ImportFailure> {
    let device_address = lock(memory).device_address;
    let (pointer, size) = (pages.pointer(), pages.length());
    // SAFETY: the pages stay mapped until the block that takes them is freed, and their pointer
    // and length are multiples of the import alignment (`Device::fresh_page_buffer`).
    let (raw, type_index) =
        unsafe { import_pointer(device, pointer, size, &[type_index], device_address) }?;
    let block = Block {
        memory: raw,
        size,
        mapped: Some(Mapped(pointer)),
        addresses: None,
        imported: false,
        pages: Some(pages),
        view: None,
        type_index,
    };
    let mut memory = lock(memory);
    memory.say_block("imported fresh pages for a buffer", &block);
    Ok(memory.add_import(block, type_index, size))
}

/// The memory of `size` bytes of host memory at `pointer`, imported as the first of `types` the
/// driver takes, with the device-address flag where `device_address`, and its type. Refused when
/// `types` is empty, else with the failure of the last type tried.
///
/// # Safety
///
/// The `size` bytes at `pointer` are host memory of this process that stays mapped until the
/// memory is freed, both multiples of the device's import alignment, and each of `types` allows
/// the pointer.
unsafe fn import_pointer(
    device: &ash::Device,
    pointer: NonNull<u8>,
    size: u64,
    types: &[u32],
    device_address: bool,
) -> Result<(vk::DeviceMemory, u32), ImportFailure> {
    let mut failure = ImportFailure::NoMemoryType;
    for &type_index in types {
        let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
            .host_pointer(pointer.as_ptr().cast());
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(type_index)
            .push_next(&mut import);
        if device_address {
            info = info.push_next(&mut flags);
        }
        // SAFETY: `info` and its chain live for the call; the type index is the device's and
        // allows the pointer; the caller's contract keeps the memory mapped.
        match unsafe { crate::creation::allocate_memory(device, &info) } {
            Ok(memory) => return Ok((memory, type_index)),
            Err(error) => failure = ImportFailure::Vulkan(error),
        }
    }
    Err(failure)
}

/// Place memory meeting a buffer's `requirements` of `class` in `memory`: in a pooled block with
/// room, else in a new block (dedicated when larger than half a block), trying the class's
/// memory types best first ([`place`]).
pub(super) fn allocate(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    requirements: &vk::MemoryRequirements,
    class: MemoryClass,
) -> Result<Placement, Error> {
    place(
        memory,
        device,
        requirements,
        class,
        None,
        Holds::Buffers,
        Fit::Pooled,
    )
}

/// [`allocate`] for an image: in a block that holds images alone ([`Holds`]).
pub(super) fn allocate_image(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    requirements: &vk::MemoryRequirements,
    class: MemoryClass,
) -> Result<Placement, Error> {
    place(
        memory,
        device,
        requirements,
        class,
        None,
        Holds::Images,
        Fit::Pooled,
    )
}

/// [`allocate_image`] for an image the runtime binds while work that uses others of its kind may
/// be queued (a neutral image, `neutral.rs`; a storage placeholder, `storage_placeholder.rs`): in
/// a block of its own (`Fit::Apart`) where such a bind waits for that work ([`binds_wait`]),
/// freed with it, else in a pooled block like any image.
pub(super) fn allocate_image_while_queued(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    requirements: &vk::MemoryRequirements,
    class: MemoryClass,
) -> Result<Placement, Error> {
    let fit = match lock(memory).binds_wait {
        true => Fit::Apart,
        false => Fit::Pooled,
    };
    place(
        memory,
        device,
        requirements,
        class,
        None,
        Holds::Images,
        fit,
    )
}

/// Whether binding memory that queued work uses waits for that work on a device of `driver`:
/// on every driver but those measured to bind at once (the [`Fit`] docs).
fn binds_wait(driver: Option<DriverId>) -> bool {
    !matches!(
        driver,
        Some(DriverId::INTEL_OPEN_SOURCE_MESA | DriverId::MESA_RADV)
    )
}

/// [`allocate`] for a buffer made without the device-address usage that needs a device address
/// (the module docs): in a block of a memory type a buffer of addresses takes, which gets one if
/// it has none (counted in `counts`), and with the placement's `address`. Without
/// `bufferDeviceAddress`, as `allocate`.
pub(super) fn allocate_addressed(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    requirements: &vk::MemoryRequirements,
    class: MemoryClass,
    counts: &Counters,
) -> Result<Placement, Error> {
    let Some(types) = lock(memory).address_types(device)? else {
        return allocate(memory, device, requirements, class);
    };
    let requirements = vk::MemoryRequirements {
        memory_type_bits: requirements.memory_type_bits & types,
        ..*requirements
    };
    place(
        memory,
        device,
        &requirements,
        class,
        Some(counts),
        Holds::Buffers,
        Fit::Pooled,
    )
}

/// [`allocate`], in a block with a buffer of addresses where `addressed` gives the counts of the
/// buffers of addresses made, of the blocks that hold `holds`, pooled or in a block of its own
/// as `fit` says. A new block is made with `memory`'s lock released: the device's other threads'
/// placements, which the lock orders, wait for no `vkAllocateMemory`, which on NVIDIA's driver
/// waited 2.37 s for other processes' GPU work.
fn place(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    requirements: &vk::MemoryRequirements,
    class: MemoryClass,
    addressed: Option<&Counters>,
    holds: Holds,
    fit: Fit,
) -> Result<Placement, Error> {
    let size = requirements.size.max(1);
    let align = requirements.alignment.max(1);
    let (candidates, maker) = {
        let memory = lock(memory);
        let bits = requirements.memory_type_bits;
        (
            candidates(&memory.types, bits, class, memory.unified),
            memory.maker(),
        )
    };
    if candidates.is_empty() {
        return Err(Error::OutOfMemory(MemoryKind::Device));
    }
    let dedicated = fit == Fit::Apart || size > maker.block_bytes / 2;
    let block_size = if dedicated { size } else { maker.block_bytes };
    let mut failure = Error::OutOfMemory(MemoryKind::Device);
    for type_index in candidates {
        let pool = (holds, type_index);
        if fit == Fit::Pooled {
            if let Some(placement) =
                lock(memory).place_pooled(device, pool, (size, align), addressed, class)?
            {
                return Ok(placement);
            }
        }
        // The type's kept empty blocks go before a block of it fails to be made.
        let made = match maker.make(device, type_index, block_size, addressed) {
            Err(Error::OutOfMemory(_)) if lock(memory).release_kept(device, type_index) > 0 => {
                maker.make(device, type_index, block_size, addressed)
            }
            made => made,
        };
        match made {
            Ok(block) => {
                let mut memory = lock(memory);
                let index = memory.add_block(pool, block, block_size, dedicated, size);
                let placement = memory.placement(pool, index, 0, size, addressed.is_some());
                memory.say_placement("placed", &placement, Some(class));
                return Ok(placement);
            }
            Err(error @ Error::OutOfMemory(_)) => failure = error,
            Err(error) => return Err(error),
        }
    }
    Err(failure)
}

/// Keep room for any pooled placement (half a block) in each pool the runtime's encodings and
/// commits place their own resources in ([`Memory::kept`]), making a block, with `memory`'s lock
/// released, for a pool that lacks it. The device's creation calls, which may wait, keep it, so
/// that an encoding or a commit, which must wait for no other process's GPU work, finds a block
/// with room rather than making one. A block of buffers gets its buffer of addresses (counted in
/// `counts`) where blocks carry one. A block that fails to be made, or a pool the query fails for,
/// is left to the placement that needs it.
pub(super) fn keep_room(memory: &Mutex<Memory>, device: &ash::Device, counts: &Counters) {
    let (lacking, maker) = {
        let mut memory = lock(memory);
        let Ok(kept) = memory.kept(device) else {
            return;
        };
        let room = memory.block_bytes / 2;
        let lacking: Vec<(Holds, u32)> = (kept.into_iter())
            .filter(|&(holds, type_index)| !memory.pool_ref(holds, type_index).has_room(room))
            .collect();
        (lacking, memory.maker())
    };
    for pool in lacking {
        let addressed = (pool.0 == Holds::Buffers).then_some(counts);
        let Ok(block) = maker.make(device, pool.1, maker.block_bytes, addressed) else {
            continue;
        };
        let mut memory = lock(memory);
        if memory
            .pool_ref(pool.0, pool.1)
            .has_room(maker.block_bytes / 2)
        {
            let what = "made a pooled block and freed it at once, its pool having room";
            memory.say_block(what, &block);
            drop(memory);
            // SAFETY: nothing is placed in the block, which no command reads.
            unsafe { block.release(device) };
            continue;
        }
        memory.add_block(pool, block, maker.block_bytes, false, 0);
    }
}
