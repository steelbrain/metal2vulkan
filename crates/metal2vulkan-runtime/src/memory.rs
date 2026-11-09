//! The runtime's own memory sub-allocator: blocks of `VkDeviceMemory` per memory type, each with a
//! free list of placed ranges.
//!
//! - **One mechanism.** A block with explicit placement serves the runtime's pools for resources
//!   made outside a heap, and later a Metal heap (step 7: a heap is one block, its resources placed
//!   at offsets).
//! - **Pooling.** Resources share blocks of [`BLOCK_BYTES`], so `maxMemoryAllocationCount` (4096
//!   on many devices) stays out of reach. A request larger than half a block takes a dedicated
//!   block of its own size.
//! - **Kept for reuse.** A block that empties is kept for the next placements rather than freed:
//!   a pooled one while fewer than [`KEPT_EMPTY_BLOCKS`] empty ones are kept in its pool, and
//!   past that, as a dedicated one of the runtime's own memory, while the empty blocks kept in its
//!   heap stay within the heap's budget ([`Memory::kept_budget`]: an eighth of the heap, at most
//!   [`KEPT_EMPTY_BYTES`]; with unified memory, where every heap is the same memory, an eighth of
//!   them all, which the empty blocks of every heap count against); else it is freed. A kept dedicated block takes the next dedicated
//!   placement of its pool that its size fits ([`Pool::place_dedicated`]). So a working set that
//!   dips and comes back makes and frees no block for it: a frame's buffers, returned as its
//!   command buffer completes, and a video clip's buffers, textures and decoded pictures, released
//!   at its end and made again at the next clip's start. On NVIDIA's driver each
//!   `vkAllocateMemory` and `vkFreeMemory` holds every submission until it returns. Every kept
//!   block goes once it stayed empty for [`KEPT_EMPTY_FOR`], but one pooled block per pool
//!   ([`Pool::stale`]), and every one of a memory type goes before a block of that type fails to
//!   be made ([`place`]).
//! - **Entry point.** [`allocate`] with the resource's memory requirements and its
//!   [`MemoryClass`]; [`Memory::free`] with the placement it gave. Buffers use it in step 1;
//!   textures and heaps add their classes and callers in their own steps. Imported host memory
//!   has its own entry, [`import_host`]: one dedicated block per import, never
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
//! - **Freed off the releasing thread** (`free_thread.rs`): a block that goes, other than an
//!   import, is freed on the device's free thread, counted against its heap and the device's
//!   allocations until it is; a block that would take its heap past the heap's size waits for
//!   the frees of that heap first ([`place`]).
//!
//! - **Images apart**. An image the runtime places in memory of its own takes a block that holds
//!   images alone ([`Holds`], [`allocate_image`]): no buffer, a block's buffer of addresses among
//!   them, is bound into such a block, nor such an image into a buffers' block. An image over a
//!   buffer's or a heap's bytes lies in that buffer's memory, at the offset Metal's layout gives
//!   it. Images are placed at a multiple of `bufferImageGranularity` and rounded up to it
//!   (`texture.rs`'s `ImagePlacement`).
//!
//! - **Exportable buffers apart** ([`allocate_exported`], `exported_memory.rs`). A buffer whose
//!   memory another API imports lies in a block allocated exportable as its handle type, an
//!   opaque file descriptor ([`Holds::Exports`]) or a dma-buf ([`Holds::DmaBufs`]), which such
//!   buffers alone share, pooled and kept as any other; or in its dedicated allocation, freed with
//!   it, where the device needs or prefers one for the export.
//!
//! - **Buffers in exportable blocks** ([`Holds::FdBuffers`], [`fd_buffer_types`]). On a device
//!   with unified memory whose policy takes textures over exported memory in place
//!   (`HostPageTexturePolicy::InPlace`) and whose memory types of a heap that is not device-local
//!   (the system heap, where `Private` memory lies first, [`candidates`]) export a dma-buf of a
//!   pooled buffer's memory (`vkGetPhysicalDeviceExternalBufferProperties`, for the program's
//!   buffers' usage and a buffer of addresses' usage: exportable, and not only from a dedicated
//!   allocation), and that enables `VK_EXT_image_drm_format_modifier`, the buffers the runtime's
//!   buffer path makes (`buffer.rs`'s `placed_new_buffer`: the program's, those of buffer
//!   descriptors, and the runtime's own, not a heap's own buffer, a copied host buffer's copy nor
//!   the in-place check's buffer of plain pooled memory) are made with
//!   `VkExternalMemoryBufferCreateInfo` naming that handle type ([`FD_BLOCKS_HANDLE`]) and lie in
//!   blocks of their own, whose blocks of those memory types are allocated exportable
//!   (`VkExportMemoryAllocateInfo`) as it; never exported. Their buffers of addresses are made
//!   for it, as a buffer bound in such memory must name it
//!   (`VUID-vkBindBufferMemory-memory-02726`), and so is an image bound in it
//!   (`VUID-vkBindImageMemory-memory-02728`, [`BoundMemory::Blocks`]: `texture.rs`'s
//!   `alias_unbound`), which is so a `DRM_FORMAT_MOD_LINEAR` one: the handle types of such an
//!   allocation must be ones the image format query takes for each resource in it
//!   (`VUID-VkExportMemoryAllocateInfo-handleTypes-09860`), and RADV's takes a dma-buf for no
//!   image of linear tiling, and an opaque file descriptor for a `DRM_FORMAT_MOD_LINEAR` one only
//!   from a dedicated allocation (`VUID-VkMemoryAllocateInfo-pNext-00639`). Such an image starts
//!   `UNDEFINED` (`VUID-VkImageCreateInfo-pNext-01443`) where over a plain block's buffer the
//!   device's linear image starts `PREINITIALIZED`. So the buffers lie there only once the device's in-place
//!   check of such blocks (`in_place.rs`'s `PagesClass::FdBlocks`) passed what a texture over one
//!   takes that it takes over a plain block's (`in_place.rs`'s `fd_blocks_take`): from the check's
//!   record as the device is made, else once its check answered, in plain blocks until then
//!   (`WorkCounts::fd_blocks_awaited`) and where it refused them (`fd_blocks_refused`). A
//!   two-plane texture's planes over such a buffer are read through the texel buffer, the
//!   device's conversion image not made (`ycbcr_planes_images_kept_exportable_block`): the check
//!   covers single-plane images alone. On a Radeon 780M (RADV) a block's `vkAllocateMemory` of a
//!   buffer object private to the process takes the reservation of the process's GPU address
//!   space, which every submission also takes, for as long as the kernel fills its pages: 64 MiB
//!   blocks of the system heap held the scheduler's submissions about 9 ms each on a quiet host
//!   and up to 56 ms on one whose free memory lay in single pages, where memory allocated
//!   exportable as an opaque file descriptor is a buffer object of its own reservation, whose
//!   making held none, as memory exportable as a dma-buf is (`tests::reclaim_submit_bench`).
//!   Each such block held costs each
//!   submission about 1 us (`docs/FAST-PATHS.md`). Every other block of buffers, of images and of
//!   a heap is allocated as before.
//!
//! - **Blocks made ahead** (`memory/ready.rs`). Each pool the encodings and commits place their
//!   own resources in ([`keep_room`]) gets a block made off the program's thread while it grows,
//!   and the pool a color texture's host-transfer image places in ([`Memory::host_copy_pool`])
//!   gets its blocks so: each made right after a burst of the device's submissions ended, since
//!   a block's `vkAllocateMemory` holds the device's submissions while it runs on RADV and on
//!   NVIDIA's driver, then every page of its mapping read once before anything is placed in it,
//!   so that no host copy's first touch faults on memory whose kernel work waits behind the
//!   program's queued GPU work. A call that needs room takes the pool's block made ahead, or
//!   waits for its making, rather than make a block of its own.

use super::buffer::usage;
use super::device::lock;
#[cfg(unix)]
use super::dma_buf::DmaBufPages;
use super::error::{internal, vulkan, Error, MemoryKind};
use super::free_thread::{FreeThread, Posted};
use super::freed_memory::{Parked, Parking};
use super::fresh_pages::OwnPages;
use super::host_ram::RamChunk;
use super::memory_log::{self, MemoryLog};
use super::range::RangeAllocator;
use super::read_only_view::ReadOnlyView;
use super::sharing::Sharing;
use super::work_counts::Counters;
use crate::device::functions::Functions;
use crate::device::queue_quiet::{QueueQuiet, Waited};
use crate::release;
use ash::vk;
use metal2vulkan::vk::DriverId;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

mod ready;

/// The size of a pooled block, at most the device's `maxMemoryAllocationSize`.
pub(super) const BLOCK_BYTES: u64 = 64 << 20;

/// The empty pooled blocks a pool keeps for its next placements whatever its heap's budget
/// ([`Pool::release`], [`Memory::kept_budget`]). In a pool the encodings place in, one of them is
/// mostly the block [`keep_room`] made for room: with one kept, every other block that emptied was
/// freed, and a block made again as the next placements filled that one, so a frame's buffers,
/// returned each frame, made and freed a block a frame. With two they make and free none
/// (`runtime_lane::pool_churn`).
pub(super) const KEPT_EMPTY_BLOCKS: usize = 2;

/// The most bytes of empty blocks a heap keeps past its pools' [`KEPT_EMPTY_BLOCKS`]
/// ([`Memory::kept_budget`]): 2 GiB, the eighth of an RTX 5080 Laptop GPU's 16 GiB. There a guest's
/// page of two 4K HEVC clips made 12 pooled blocks and 66 decoded pictures of about 15 MiB each in
/// exportable buffers (`exported_memory.rs`), 1.8 GB in all, and the clips' ends freed them.
pub(super) const KEPT_EMPTY_BYTES: u64 = 2 << 30;

/// The share of its heap's size a heap keeps of empty blocks, at most [`KEPT_EMPTY_BYTES`]: an
/// eighth ([`Memory::kept_budget`]).
const KEPT_EMPTY_SHARE: u64 = 8;

/// The most bytes of empty blocks heap `heap` of heaps of `sizes` keeps for reuse: an eighth of the
/// heap, at most [`KEPT_EMPTY_BYTES`]. With unified memory (`unified`) every heap is the same
/// memory, and the budget is one: an eighth of all of them, at most [`KEPT_EMPTY_BYTES`], which the
/// empty blocks of every heap count against ([`Memory::kept_budget`]). A Radeon 780M (RADV) lists
/// its system memory as two heaps, 11.05 GiB device-local and 5.53 GiB not: an eighth of the
/// second, 707 MiB, which its `Private` placements take first (`candidates`), left 3 of a video
/// clip's 16 blocks of 64 MiB to be freed at each clip's end and each later clip made 4 again
/// (`tests::clip_bursts`), where one budget of 2 GiB keeps them all.
pub(super) fn kept_budget(sizes: &[u64], heap: usize, unified: bool) -> u64 {
    let size = match unified {
        true => sizes.iter().sum(),
        false => sizes.get(heap).copied().unwrap_or(0),
    };
    (size / KEPT_EMPTY_SHARE).min(KEPT_EMPTY_BYTES)
}

/// How long a kept empty block stays empty before it is freed, but its pool's last pooled one
/// ([`Pool::stale`]): a frame's dip comes back well within it, and a guest's next video clip
/// started 1 s after the last one ended.
pub(super) const KEPT_EMPTY_FOR: Duration = Duration::from_secs(10);

/// How often the device's creation calls look for kept empty blocks past [`KEPT_EMPTY_FOR`] in
/// every pool ([`Memory::sweep_stale`]), beside each release's look in its own pool: a kept block
/// goes within this long of its quiet period's end wherever the program still makes resources.
const KEPT_SWEEP_EVERY: Duration = Duration::from_secs(1);

/// How much longer than a dedicated placement a kept dedicated block may be and still take it
/// ([`Pool::place_dedicated`]): a quarter of the placement.
const DEDICATED_SLACK: u64 = 4;

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
    /// Device-local (`Private` storage), host-invisible where the device has a choice; where no
    /// device-local type has room, system memory ([`overflow_candidates`]). With unified memory
    /// and a heap that is not device-local, that heap's memory first ([`candidates`]).
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
///
/// Two more kinds hold buffers whose memory another API imports (`exported_memory.rs`): their
/// blocks are allocated exportable, as an opaque file descriptor ([`EXPORT_HANDLE`]) or as a
/// dma-buf ([`DMA_BUF_HANDLE`]), which a buffer made without that handle type may not be bound to
/// (`VUID-vkBindBufferMemory-memory-02726`), and the buffers of each handle type share them.
///
/// A fifth, [`Holds::FdBuffers`], holds the buffers the runtime's buffer path makes on a device
/// whose blocks of them in memory of a heap that is not device-local are allocated exportable as
/// a dma-buf ([`FD_BLOCKS_HANDLE`]), never exported (the module docs, "Buffers in exportable
/// blocks").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Holds {
    Buffers,
    Images,
    Exports,
    DmaBufs,
    /// The buffers `buffer.rs`'s `placed_new_buffer` makes for the program and for the runtime
    /// (not a heap's own buffer, nor the in-place check's buffer of pooled memory), on a device
    /// that takes such blocks ([`Memory::fd_buffer_blocks`]): each made with
    /// `VkExternalMemoryBufferCreateInfo` naming [`FD_BLOCKS_HANDLE`], so that it may be bound in a
    /// block allocated exportable as that handle type (`VUID-vkBindBufferMemory-memory-02726`).
    /// Its blocks of the memory types [`fd_buffer_types`] names are allocated so, its others as
    /// any block of buffers.
    FdBuffers,
}

impl Holds {
    /// The kinds, in the order of the device's pools ([`Memory::pool`]).
    const ALL: [Holds; 5] = [
        Holds::Buffers,
        Holds::Images,
        Holds::Exports,
        Holds::DmaBufs,
        Holds::FdBuffers,
    ];

    /// The handle type its blocks are allocated exportable as for another API's import; `None`
    /// for blocks never exported, [`Holds::FdBuffers`]' among them.
    fn handle(self) -> Option<vk::ExternalMemoryHandleTypeFlags> {
        match self {
            Holds::Buffers | Holds::Images | Holds::FdBuffers => None,
            Holds::Exports => Some(EXPORT_HANDLE),
            Holds::DmaBufs => Some(DMA_BUF_HANDLE),
        }
    }

    /// Whether its blocks hold buffers, and so may carry a buffer of addresses.
    fn buffers(self) -> bool {
        matches!(self, Holds::Buffers | Holds::FdBuffers)
    }
}

/// The memory types whose [`Holds::FdBuffers`] blocks a device allocates exportable as
/// [`FD_BLOCKS_HANDLE`], as bits (the module docs, "Buffers in exportable blocks"), of a device
/// with `properties`, with unified memory or without (`unified`), whose answers for such an
/// export of a pooled buffer's memory and the images over it are `exports`: with unified
/// memory, the types of a heap that is not device-local, where the device exports such memory
/// from a block that buffers share. Else why it takes none.
pub(super) fn fd_buffer_types(
    properties: &vk::PhysicalDeviceMemoryProperties,
    unified: bool,
    exports: Result<(), &'static str>,
) -> Result<u32, &'static str> {
    if !unified {
        return Err("the device has no unified memory");
    }
    exports?;
    let count = (properties.memory_type_count as usize).min(vk::MAX_MEMORY_TYPES);
    let heaps = &properties.memory_heaps;
    let types = (properties.memory_types[..count].iter().enumerate())
        .filter(|(_, memory_type)| {
            (heaps.get(memory_type.heap_index as usize))
                .is_some_and(|heap| !heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        })
        .fold(0, |bits, (index, _)| bits | 1 << index);
    match types {
        0 => Err("the device has no memory type of a heap that is not device-local"),
        types => Ok(types),
    }
}

/// What a new block's memory is allocated exportable as ([`BlockMaker::make_as`]).
#[derive(Clone, Copy)]
enum Exportable {
    /// Nothing.
    No,
    /// The handle type of the exportable buffers it holds (`exported_memory.rs`), for its buffer:
    /// that buffer's dedicated allocation where `Exported::dedicated` says so.
    For(vk::Buffer, Exported),
    /// [`FD_BLOCKS_HANDLE`], never exported: a block of [`Holds::FdBuffers`] of a memory type
    /// [`fd_buffer_types`] names.
    Fd,
}

impl Exportable {
    /// The handle type it names; `None` for [`Exportable::No`].
    fn handle(self) -> Option<vk::ExternalMemoryHandleTypeFlags> {
        match self {
            Exportable::No => None,
            Exportable::For(_, exported) => Some(exported.handle),
            Exportable::Fd => Some(FD_BLOCKS_HANDLE),
        }
    }
}

/// The handle type the runtime exports a buffer's memory as (`exported_memory.rs`): an opaque file
/// descriptor. The blocks of exportable buffers are allocated exportable as it
/// ([`Holds::Exports`]).
pub(super) const EXPORT_HANDLE: vk::ExternalMemoryHandleTypeFlags =
    vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;

/// The handle type the runtime exports a dma-buf-exportable buffer's memory as
/// (`exported_memory.rs`, `VK_EXT_external_memory_dma_buf`). The blocks of such buffers are
/// allocated exportable as it ([`Holds::DmaBufs`]).
pub(super) const DMA_BUF_HANDLE: vk::ExternalMemoryHandleTypeFlags =
    vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;

/// The handle type the blocks of [`Holds::FdBuffers`] are allocated exportable as, never
/// exported (the module docs, "Buffers in exportable blocks"): a dma-buf. Every image bound there
/// is a `DRM_FORMAT_MOD_LINEAR` one, which RADV's image format query takes as a dma-buf from a
/// block that buffers share, where it takes an opaque file descriptor for such an image only from
/// a dedicated allocation (`VUID-VkMemoryAllocateInfo-pNext-00639`) and a dma-buf for no image of
/// linear tiling (`VUID-VkExportMemoryAllocateInfo-handleTypes-09860`).
pub(super) const FD_BLOCKS_HANDLE: vk::ExternalMemoryHandleTypeFlags = DMA_BUF_HANDLE;

/// `VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT` (`VK_EXT_zero_initialize_device_memory`), which
/// `ash 0.38` predates.
pub(super) const ZERO_INITIALIZE: vk::MemoryAllocateFlags = vk::MemoryAllocateFlags::from_raw(0x8);

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

/// Why [`import_host`] imported nothing.
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
///
/// With unified memory, a heap that is not device-local is the same memory as the device-local
/// one, and the device-local heap may be a carve-out the kernel reserves from it for every
/// process's GPU memory. So `Private` and `Memoryless` memory, where the resource takes a type
/// that is neither device-local nor lazily allocated, takes those types first (host-invisible,
/// then uncached), and device-local ones after (`Memoryless` its lazily allocated ones before
/// all). RADV lists a Radeon 780M's 2 GiB carve-out and its system memory as one device-local heap
/// of 11.05 GiB with the system memory's types beside it in a heap of 5.53 GiB that is not
/// device-local, and the kernel backs its device-local types from the carve-out first, evicting
/// to make room once the carve-out is full, during the submissions of every process that holds
/// memory there; `VK_EXT_memory_budget` reports that heap's budget, not the carve-out's, so no
/// budget keeps the runtime's placements within it.
pub(super) fn candidates(
    types: &[vk::MemoryType],
    allowed: u32,
    class: MemoryClass,
    unified: bool,
) -> Vec<u32> {
    use vk::MemoryPropertyFlags as P;
    let usable = |index: usize| allowed & (1 << index) != 0;
    if unified && matches!(class, MemoryClass::Private | MemoryClass::Memoryless) {
        let fits = |index: usize, flags: P| {
            usable(index)
                && allocatable(flags)
                && !flags.contains(P::PROTECTED)
                && (class == MemoryClass::Memoryless || !flags.contains(P::LAZILY_ALLOCATED))
        };
        let system = (types.iter().enumerate()).any(|(index, ty)| {
            fits(index, ty.property_flags)
                && !ty
                    .property_flags
                    .intersects(P::DEVICE_LOCAL | P::LAZILY_ALLOCATED)
        });
        if system {
            let has = |flags: P, bit: P| u8::from(flags.contains(bit));
            let mut ranked: Vec<((u8, u8, u8, u8), u32)> = (types.iter().enumerate())
                .filter(|&(index, ty)| fits(index, ty.property_flags))
                .map(|(index, ty)| {
                    let flags = ty.property_flags;
                    let rank = (
                        1 - has(flags, P::LAZILY_ALLOCATED),
                        has(flags, P::DEVICE_LOCAL),
                        has(flags, P::HOST_CACHED),
                        has(flags, P::HOST_VISIBLE),
                    );
                    (rank, index as u32)
                })
                .collect();
            // Vulkan lists memory types best first within equal properties; the sort is stable.
            ranked.sort_by_key(|&(rank, _)| rank);
            return ranked.into_iter().map(|(_, index)| index).collect();
        }
    }
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

/// The memory types a resource of `class` overflows into among `types` (the bits of `allowed`)
/// where none of its own [`candidates`] has room: for a class that requires device-local memory
/// (`Private`, `Memoryless`), every other type the resource may use that the runtime may allocate
/// ([`allocatable`]) and that is not lazily allocated, those the host does not map first. Metal
/// places such a resource on a discrete GPU past the GPU's memory rather than refusing it (on a
/// 4 GiB Radeon Pro 5300M it made, used and read back 12 GiB of `Managed` buffers, six 1 GiB
/// `Private` buffers and a 17 GB `Private` heap), so the runtime places it in system memory, which
/// the GPU reads and writes across its bus (`Degradation::DeviceMemoryOverflow`). On an RTX 5080
/// Laptop GPU (Linux), a buffer the runtime places overflows into its types 0, 2 and 3 (system
/// memory the host does not map, then the uncached and cached kinds it maps), and an optimal color
/// image into type 0. None for the other classes, whose candidates already take every host-visible
/// coherent type, nor on a device whose every type is device-local (MoltenVK on an M3 Max), nor on
/// one with unified memory (`unified`), whose candidates already take every type that is not
/// device-local ([`candidates`]).
pub(super) fn overflow_candidates(
    types: &[vk::MemoryType],
    allowed: u32,
    class: MemoryClass,
    unified: bool,
) -> Vec<u32> {
    use vk::MemoryPropertyFlags as P;
    if unified || !matches!(class, MemoryClass::Private | MemoryClass::Memoryless) {
        return Vec::new();
    }
    let mut ranked: Vec<(bool, u32)> = types
        .iter()
        .enumerate()
        .filter(|&(index, ty)| {
            allowed & (1 << index) != 0
                && allocatable(ty.property_flags)
                && !ty
                    .property_flags
                    .intersects(P::DEVICE_LOCAL | P::LAZILY_ALLOCATED)
        })
        .map(|(index, ty)| (ty.property_flags.contains(P::HOST_VISIBLE), index as u32))
        .collect();
    // Vulkan lists memory types best first within equal properties; the sort is stable.
    ranked.sort_by_key(|&(mapped, _)| mapped);
    ranked.into_iter().map(|(_, index)| index).collect()
}

/// Where a device's memory may lie ([`Memory::type_view`]), for the crate's device tests.
#[cfg(all(test, any(m2v_cpu_icd, all(m2v_gpu, m2v_gpu_hazards))))]
pub(super) struct TypeView {
    /// The device's memory types.
    pub(super) types: Vec<vk::MemoryType>,
    /// Its memory heaps' sizes.
    pub(super) heap_sizes: Vec<u64>,
    /// The types a buffer the runtime places takes, as bits.
    pub(super) buffers: u32,
    /// The types a color image of optimal tiling takes, as bits.
    pub(super) images: u32,
}

/// The block bookkeeping of one memory type: pure, so the CPU tests cover it. `T` is what the
/// caller keeps per block (its `VkDeviceMemory` and mapping).
#[derive(Debug)]
pub(super) struct Pool<T> {
    blocks: Vec<Option<PoolBlock<T>>>,
    /// The largest alignment a pooled placement asked of it so far ([`Pool::room`]).
    align: u64,
    /// The longest free range a pooled placement could take at [`Pool::align`], derived over the
    /// blocks once and held until something that can move it changes ([`Pool::room`]); `None` to
    /// derive again.
    room: Option<u64>,
    /// How many live block slots [`Pool::room`]'s derivations have walked, and how many free
    /// ranges they have visited (`tests::pool_room`).
    walked: u64,
    visited: u64,
    /// The pooled blocks added and freed so far ([`Pool::turnover`]).
    made: u64,
    freed: u64,
    /// The bytes of its live blocks with nothing placed in them: the empty blocks it keeps
    /// ([`Pool::kept_bytes`]).
    kept_bytes: u64,
    /// When it last grew: a pooled block added to it while it held another ([`Pool::grew`]).
    grew: Option<Instant>,
}

#[derive(Debug)]
struct PoolBlock<T> {
    ranges: RangeAllocator,
    dedicated: bool,
    /// Its bytes.
    size: u64,
    payload: T,
    /// When it emptied, while it is empty and kept ([`Pool::release`]); `None` for a block
    /// added with nothing placed in it, kept as long as it stays empty.
    emptied: Option<Instant>,
    /// Its longest free range at its pool's [`Pool::align`], derived from its free ranges once
    /// and held until its ranges or that alignment change ([`Pool::room`]); `None` to derive
    /// again.
    longest: Option<u64>,
    /// Its memory was allocated with `VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT`, which reads
    /// zero until written ([`Memory::zero_image_blocks`]).
    zeroed: bool,
    /// The end of the furthest range ever placed in it: past it nothing was placed since the
    /// block was made, so that where the block is `zeroed` those bytes read zero
    /// ([`Pool::place_fresh`]).
    placed_to: u64,
}

impl<T> PoolBlock<T> {
    /// Whether it is a pooled block with nothing placed in it, kept for the next placements.
    fn kept(&self) -> bool {
        !self.dedicated && self.ranges.is_empty()
    }

    /// Whether nothing is placed in it: a kept block, pooled or dedicated.
    fn empty(&self) -> bool {
        self.ranges.is_empty()
    }
}

impl<T> Default for Pool<T> {
    fn default() -> Pool<T> {
        Pool {
            blocks: Vec::new(),
            align: 1,
            room: None,
            walked: 0,
            visited: 0,
            made: 0,
            freed: 0,
            kept_bytes: 0,
            grew: None,
        }
    }
}

impl<T> Pool<T> {
    /// A place for `size` bytes aligned to `align` in a pooled block that has room: (block,
    /// offset).
    #[cfg(test)]
    pub(super) fn place(&mut self, size: u64, align: u64) -> Option<(usize, u64)> {
        (self.place_fresh(size, align)).map(|(index, offset, _)| (index, offset))
    }

    /// A place for `size` bytes aligned to `align` in a pooled block that has room, (block,
    /// offset), with whether the place reads zero: it lies in a block allocated zero-initialized,
    /// past every range placed in that block before, so that no resource used its bytes since the
    /// driver zeroed them.
    pub(super) fn place_fresh(&mut self, size: u64, align: u64) -> Option<(usize, u64, bool)> {
        if align > self.align {
            // Every block's longest free range was derived at the smaller alignment.
            self.align = align;
            self.forget_longest();
        }
        let placed = self
            .blocks
            .iter_mut()
            .enumerate()
            .find_map(|(index, block)| {
                let block = block.as_mut().filter(|block| !block.dedicated)?;
                let was_empty = block.empty();
                let offset = block.ranges.allocate(size, align)?;
                block.emptied = None;
                block.longest = None;
                let fresh = block.zeroed && offset >= block.placed_to;
                block.placed_to = block.placed_to.max(offset + size);
                Some((index, offset, fresh, was_empty.then_some(block.size)))
            });
        let (index, offset, fresh, was_kept) = placed?;
        self.room = None;
        self.kept_bytes -= was_kept.unwrap_or(0);
        Some((index, offset, fresh))
    }

    /// A kept dedicated block for a dedicated placement of `size` bytes: the shortest kept one at
    /// least `size` long and at most a [`DEDICATED_SLACK`]th longer, its first `size` bytes placed
    /// (at offset 0, which meets every alignment): its index.
    pub(super) fn place_dedicated(&mut self, size: u64) -> Option<usize> {
        let most = size.saturating_add(size / DEDICATED_SLACK);
        let (index, _) = (self.blocks.iter().enumerate())
            .filter_map(|(index, block)| {
                let block = block
                    .as_ref()
                    .filter(|block| block.dedicated && block.empty())?;
                (block.size >= size && block.size <= most).then_some((index, block.size))
            })
            .min_by_key(|&(_, bytes)| bytes)?;
        let block = self.blocks[index].as_mut()?;
        let offset = block.ranges.allocate(size, 1)?;
        debug_assert_eq!(offset, 0);
        block.emptied = None;
        block.longest = None;
        block.placed_to = block.placed_to.max(size);
        self.kept_bytes -= block.size;
        Some(index)
    }

    /// Forget every longest free range held, which the pool's alignment growing can move.
    fn forget_longest(&mut self) {
        for block in self.blocks.iter_mut().flatten() {
            block.longest = None;
        }
        self.room = None;
    }

    /// Add a block of `size` bytes holding `payload`, and place `used` bytes at its start when
    /// given: the block's index.
    pub(super) fn add(&mut self, size: u64, dedicated: bool, payload: T, used: u64) -> usize {
        self.add_as(size, dedicated, payload, used, false)
    }

    /// [`Pool::add`] of a block whose memory reads zero until written where `zeroed`: allocated
    /// with `VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT` ([`Pool::place_fresh`]).
    pub(super) fn add_as(
        &mut self,
        size: u64,
        dedicated: bool,
        payload: T,
        used: u64,
        zeroed: bool,
    ) -> usize {
        let mut ranges = RangeAllocator::new(0, size);
        if used > 0 {
            let at = ranges.allocate(used, 1);
            debug_assert_eq!(at, Some(0));
        }
        let block = Some(PoolBlock {
            ranges,
            dedicated,
            size,
            payload,
            emptied: None,
            longest: None,
            zeroed,
            placed_to: used,
        });
        self.made += u64::from(!dedicated);
        if used == 0 {
            self.kept_bytes += size;
        }
        self.room = None;
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
    /// pool so far: whether [`Pool::room`] is that long.
    ///
    /// For a `size` above 0, which is every `size` asked -- [`keep_room`] asks half a block, and
    /// [`BLOCK_BYTES`] is capped by `maxMemoryAllocationSize`, which Vulkan requires to be at
    /// least a gibibyte -- that is the question exactly. For a `size` of 0 it is whether the pool
    /// has any room at all rather than whether it has a pooled block, which is the same answer
    /// wherever a pooled block holds a free byte.
    pub(super) fn has_room(&mut self, size: u64) -> bool {
        self.room() >= size
    }

    /// The longest free range a pooled placement could take at [`Pool::align`]: the longest over
    /// the pooled blocks, a dedicated block's free bytes counting for none.
    ///
    /// Every creation call the runtime makes asks whether its pools have room ([`keep_room`]),
    /// where each answer is the same until a placement, a block or the alignment changes it, so
    /// the pool holds the answer and derives it once: a creation that places nothing in a pool
    /// walks none of its blocks, and one that places walks the blocks but derives the longest
    /// free range of the one block whose ranges moved. Deriving per call instead walks every free
    /// range of every block per creation, which grows with the pool's history; the count that
    /// holds it to once is `tests::pool_room`.
    fn room(&mut self) -> u64 {
        if let Some(room) = self.room {
            return room;
        }
        let align = self.align;
        let (mut room, mut walked, mut visited) = (0, 0, 0);
        for block in self.blocks.iter_mut().flatten() {
            walked += 1;
            if block.dedicated {
                continue;
            }
            let longest = match block.longest {
                Some(longest) => longest,
                None => {
                    visited += block.ranges.range_count();
                    *block.longest.insert(block.ranges.longest(align))
                }
            };
            room = room.max(longest);
        }
        self.walked += walked;
        self.visited += visited;
        self.room = Some(room);
        room
    }

    /// How many live block slots the pool's derivations of [`Pool::room`] have walked, and how
    /// many free ranges they have visited.
    #[cfg(test)]
    pub(super) fn room_work(&self) -> (u64, u64) {
        (self.walked, self.visited)
    }

    /// Its live block slots, and the free ranges its pooled blocks hold: what one query of
    /// [`Pool::room`] walks and visits where it derives per call.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(super) fn room_per_call(&self) -> (u64, u64) {
        let live = (self.blocks.iter().flatten()).count() as u64;
        let ranges = (self.blocks.iter().flatten())
            .filter(|block| !block.dedicated)
            .map(|block| block.ranges.range_count())
            .sum();
        (live, ranges)
    }

    /// [`Pool::room`] derived now, holding nothing: what the held answer must equal.
    #[cfg(test)]
    pub(super) fn room_derived_now(&self) -> u64 {
        (self.blocks.iter().flatten())
            .filter(|block| !block.dedicated)
            .map(|block| block.ranges.longest(self.align))
            .max()
            .unwrap_or(0)
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

    /// Return `[offset, offset + size)` of block `index`, at `now`, where the empty blocks kept in
    /// the pool's heap may grow by `spare` bytes more ([`Memory::kept_budget`]). A block that
    /// empties is kept: a pooled one while fewer than [`KEPT_EMPTY_BLOCKS`] other empty pooled
    /// blocks are, or its bytes fit in `spare`; a dedicated one where its bytes fit in `spare` and
    /// `reusable` says another placement may take its memory ([`Pool::place_dedicated`]). Else the
    /// block's payload, to be freed now, with whether it was a dedicated block rather than a pooled
    /// one: the two reasons a release frees a block, which its caller counts apart ([`FreeCause`]).
    pub(super) fn release(
        &mut self,
        index: usize,
        (offset, size): (u64, u64),
        now: Instant,
        spare: u64,
        reusable: impl FnOnce(&T) -> bool,
    ) -> Option<(T, bool)> {
        let block = self.blocks.get_mut(index)?.as_mut()?;
        block.ranges.free(offset, size);
        block.longest = None;
        self.room = None;
        let block = self.blocks[index].as_ref().expect("the block is live");
        if !block.empty() {
            return None;
        }
        let (dedicated, bytes) = (block.dedicated, block.size);
        self.kept_bytes += bytes;
        let keep = match dedicated {
            false => {
                let kept = (self.blocks.iter().enumerate())
                    .filter(|&(other, block)| {
                        other != index && block.as_ref().is_some_and(PoolBlock::kept)
                    })
                    .count();
                kept < KEPT_EMPTY_BLOCKS || bytes <= spare
            }
            true => bytes <= spare && reusable(&block.payload),
        };
        if keep {
            if let Some(block) = self.blocks[index].as_mut() {
                block.emptied = Some(now);
            }
            return None;
        }
        Some((self.take(index)?, dedicated))
    }

    /// At `now`, the payload of the kept empty block that emptied earliest, where it stayed empty
    /// for [`KEPT_EMPTY_FOR`], taken out to be freed: a dedicated one, or a pooled one while
    /// another empty pooled block stays kept.
    pub(super) fn stale(&mut self, now: Instant) -> Option<T> {
        let pooled = (self.blocks.iter().flatten())
            .filter(|block| block.kept())
            .count();
        let (index, _) = (self.blocks.iter().enumerate())
            .filter_map(|(index, block)| {
                let block = block.as_ref().filter(|block| block.empty())?;
                (block.dedicated || pooled >= 2).then_some((index, block.emptied?))
            })
            .filter(|&(_, at)| now.saturating_duration_since(at) >= KEPT_EMPTY_FOR)
            .min_by_key(|&(_, at)| at)?;
        self.take(index)
    }

    /// Every kept empty block's payload, pooled and dedicated, taken out to be freed.
    pub(super) fn take_kept(&mut self) -> Vec<T> {
        let kept: Vec<usize> = (self.blocks.iter().enumerate())
            .filter(|(_, block)| block.as_ref().is_some_and(PoolBlock::empty))
            .map(|(index, _)| index)
            .collect();
        kept.into_iter()
            .filter_map(|index| self.take(index))
            .collect()
    }

    /// The bytes of the empty blocks it keeps, pooled and dedicated.
    pub(super) fn kept_bytes(&self) -> u64 {
        self.kept_bytes
    }

    /// Block `index`'s payload, its slot emptied.
    fn take(&mut self, index: usize) -> Option<T> {
        let block = self.blocks.get_mut(index)?.take()?;
        self.freed += u64::from(!block.dedicated);
        if block.empty() {
            self.kept_bytes -= block.size;
        }
        self.room = None;
        Some(block.payload)
    }

    /// The pooled blocks added and freed so far.
    pub(super) fn turnover(&self) -> (u64, u64) {
        (self.made, self.freed)
    }

    /// When it last grew: a pooled block was added to it while it held another, at `now` where
    /// `adding` names one being added. Blocks made ahead are made for a pool that grew
    /// (`memory/ready.rs`).
    pub(super) fn grew(&mut self, adding: Option<Instant>) -> Option<Instant> {
        if let Some(now) = adding {
            if self.blocks.iter().flatten().any(|block| !block.dedicated) {
                self.grew = Some(now);
            }
        }
        self.grew
    }

    /// Every block's payload, emptying the pool.
    pub(super) fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.room = None;
        self.kept_bytes = 0;
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
                (false, _) if block.empty() => &mut census.empty,
                (false, true) => &mut census.dedicated,
                (false, false) => &mut census.pooled,
            };
            *count += 1;
        }
    }
}

/// Why a block of this allocator's memory was freed ([`Memory::let_go`],
/// [`Memory::destroy`]): one `vkFreeMemory` each, counted where the block went, before any wait
/// for the batches in flight delays the call (`freed_memory.rs`). A driver that unmaps a buffer
/// object's pages as it frees it charges the page-table update to the free, so what a workload
/// frees, and why, is what sets that cost; `WorkCounts::memory_frees` reports these.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FreeCause {
    /// A block of its own, freed as the one resource placed in it went: a placement larger than
    /// half a block, or an image bound while work was queued ([`allocate_image_while_queued`]).
    Dedicated = 0,
    /// A pooled block freed as its last placement left it, with [`KEPT_EMPTY_BLOCKS`] other
    /// empty pooled blocks already kept.
    PooledEmptied = 1,
    /// A kept empty pooled block freed after [`KEPT_EMPTY_FOR`] without a placement
    /// ([`Pool::stale`]).
    KeptStale = 2,
    /// A kept empty pooled block freed so that a block of its memory type can be made
    /// ([`Memory::release_kept`]).
    KeptForRoom = 3,
    /// The caller's host memory, imported as a dedicated block ([`import_host`]) and
    /// freed as its placement went.
    HostImport = 4,
    /// A buffer's fresh pages, imported as a dedicated block ([`import_own_pages`],
    /// `fresh_pages.rs`) and freed as its placement went.
    OwnPages = 5,
    /// Freed as the device was destroyed, its queues idle ([`Memory::destroy`]).
    Teardown = 6,
}

impl FreeCause {
    /// How many causes there are: every block this allocator frees is counted under exactly one.
    pub(super) const COUNT: usize = 7;
}

/// A device's live `VkDeviceMemory` blocks, by kind ([`Memory::census`]): each one buffer
/// object the driver lists in every submission where it lists every allocation (ANV on i915).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Census {
    /// Pooled blocks with a placement in them.
    pub(super) pooled: u64,
    /// Blocks with none, pooled or dedicated, kept for the next placements ([`Pool::release`]).
    pub(super) empty: u64,
    /// Dedicated blocks with their placement: a placement larger than half a block, or one placed
    /// apart.
    pub(super) dedicated: u64,
    /// The caller's host memory imported ([`import_host`]), a block each.
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

    /// The caller's bytes at `pointer`, as the host address of a copied host buffer's memory
    /// (`copied.rs`): its `contents`, which the runtime's comparisons keep coherent with the copy.
    pub(super) fn of_caller(pointer: NonNull<u8>) -> Mapped {
        Mapped(pointer)
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
    /// It is the caller's host memory, imported ([`import_host`]).
    imported: bool,
    /// The pages its memory was imported from, where the runtime mapped them for it
    /// ([`import_own_pages`]): unmapped once the memory is freed.
    pages: Option<OwnPages>,
    /// The read-only view of the caller's pages its memory was imported from, which drops the
    /// GPU's writes ([`import_host`], `read_only_view.rs`): unmapped once the memory is
    /// freed.
    view: Option<ReadOnlyView>,
    /// Its memory type.
    type_index: u32,
    /// What it was allocated exportable as, where it was ([`allocate_exported`]).
    exported: Option<Exported>,
    /// It was allocated exportable as [`FD_BLOCKS_HANDLE`] and is never exported
    /// ([`Holds::FdBuffers`]): an image bound in it names that handle type
    /// ([`Memory::bound_memory`]).
    fd: bool,
    /// It was allocated with `VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT`, which reads zero until
    /// written ([`Memory::zero_image_blocks`]).
    zeroed: bool,
}

/// The external memory a placement's memory is, as an image bound in it names it
/// ([`Memory::bound_memory`]).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum BoundMemory {
    /// Memory the device allocated with no export: an image bound in it names no handle type.
    Own,
    /// Memory exported or imported as this handle type, which an image bound in it names.
    External(vk::ExternalMemoryHandleTypeFlags),
    /// A block of the runtime's own buffers allocated exportable as [`FD_BLOCKS_HANDLE`] and
    /// never exported ([`Holds::FdBuffers`]): an image bound in it names that handle type
    /// (`VUID-vkBindImageMemory-memory-02728`) and is a `DRM_FORMAT_MOD_LINEAR` one
    /// (`VUID-VkExportMemoryAllocateInfo-handleTypes-09860`).
    Blocks,
    /// Memory no image of the runtime's is bound in.
    NoImage(NoImage),
}

/// Why no image of the runtime's is bound in a placement's memory ([`BoundMemory::NoImage`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NoImage {
    /// It is a buffer's dedicated allocation (`VkMemoryDedicatedAllocateInfo::buffer`), to which
    /// no image may be bound (`VUID-vkBindImageMemory-memory-10926`): an exportable buffer's where
    /// the driver exports such memory only so, or prefers or requires it (`exported_memory.rs`).
    Dedicated,
    /// It is memory the caller made and keeps (`Device::new_buffer_with_vk_buffer`), whose handle
    /// types the runtime does not know.
    Caller,
}

impl std::fmt::Debug for BoundMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BoundMemory::Own => f.write_str("Own"),
            BoundMemory::External(handle) => write!(f, "External({:#x})", handle.as_raw()),
            BoundMemory::Blocks => f.write_str("Blocks"),
            BoundMemory::NoImage(no) => write!(f, "NoImage({no:?})"),
        }
    }
}

impl NoImage {
    /// Why, in words, for a notice.
    pub(super) fn words(self) -> &'static str {
        match self {
            NoImage::Dedicated => {
                "the buffer's memory is its dedicated allocation, the driver's for memory exported \
                 so, to which no image may be bound (VUID-vkBindImageMemory-memory-10926)"
            }
            NoImage::Caller => {
                "the buffer's memory is the caller's, whose external handle types the runtime does \
                 not know"
            }
        }
    }
}

/// What a block's memory was allocated exportable as (`VkExportMemoryAllocateInfo`), for the
/// buffers placed in it (`exported_memory.rs`): one, where it is that buffer's dedicated
/// allocation.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Exported {
    /// The handle type it exports.
    pub(super) handle: vk::ExternalMemoryHandleTypeFlags,
    /// It is the buffer's dedicated allocation (`VkMemoryDedicatedAllocateInfo`).
    pub(super) dedicated: bool,
}

impl std::fmt::Debug for Exported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Exported")
            .field("handle", &self.handle.as_raw())
            .field("dedicated", &self.dedicated)
            .finish()
    }
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
    /// Whether another placement may take its memory once its own went: memory the runtime
    /// allocated for itself, neither an import nor a buffer's dedicated allocation
    /// (`VkMemoryDedicatedAllocateInfo`, which binds that buffer alone).
    fn reusable(&self) -> bool {
        !self.imported
            && self.pages.is_none()
            && self.view.is_none()
            && !self.exported.is_some_and(|exported| exported.dedicated)
    }

    /// The cause its `vkFreeMemory` is counted under. A block freed as the one placement in it
    /// went ([`FreeCause::Dedicated`]) that is an import of its own is counted as that import
    /// instead: the caller's host memory ([`import_host`]) or a buffer's fresh pages
    /// ([`import_own_pages`]), each a dedicated block of one placement. Any other cause stands:
    /// a block freed at the device's teardown is counted there whatever it held.
    fn cause(&self, given: FreeCause) -> FreeCause {
        match (given, self.imported, self.pages.is_some()) {
            (FreeCause::Dedicated, true, _) => FreeCause::HostImport,
            (FreeCause::Dedicated, false, true) => FreeCause::OwnPages,
            _ => given,
        }
    }

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
/// made for `handle` where given (for a block exportable as it,
/// `VUID-vkBindBufferMemory-memory-02726`), unbound, and the bytes of memory it requires.
fn address_buffer(
    device: &ash::Device,
    size: u64,
    sharing: Sharing<'_>,
    handle: Option<vk::ExternalMemoryHandleTypeFlags>,
) -> Result<(vk::Buffer, u64), Error> {
    let mut external =
        vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle.unwrap_or_default());
    let mut info = sharing.buffer(
        vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage(true)),
    );
    if handle.is_some() {
        info = info.push_next(&mut external);
    }
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
    /// It lies past its class's own memory types, in one it overflowed into where none of them
    /// had room ([`overflow_candidates`]).
    overflowed: bool,
    /// It lies in a block allocated zero-initialized, past every range placed in that block
    /// before: its bytes read zero, as no resource used them since the driver zeroed them
    /// ([`Memory::zero_image_blocks`], [`Placement::fresh`]).
    fresh: bool,
    /// The imports of the caller's dma-bufs the placement lies in, with the source that made them
    /// (`dma_buf.rs`), which no block of the runtime's holds: kept until the placement goes, and
    /// freed, the source dropped, when their last holder goes.
    #[cfg(unix)]
    dma_buf: Option<Arc<DmaBufPages>>,
    /// The device's import of a chunk of a guest RAM the placement lies in (`host_ram.rs`), which
    /// no block of the placement's holds: kept until the placement goes.
    ram_chunk: Option<Arc<RamChunk>>,
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
            overflowed: false,
            fresh: false,
            #[cfg(unix)]
            dma_buf: None,
            ram_chunk: None,
        }
    }

    /// The placement of a no-copy object's `size` bytes of host pages, from `offset` of the
    /// memory of `chunk`, the device's import of the chunk of a guest RAM they lie in
    /// (`host_ram.rs`), whose host address the caller reaches at `mapped` (its own view of the
    /// pages). No block of the runtime's, so freeing it returns nothing; it keeps the chunk until
    /// it goes. A chunk imported as a dma-buf lends the placement its import, with the RAM's
    /// source (`dma_buf_pages`), for the images of textures over the bytes.
    pub(super) fn ram_chunk(
        chunk: Arc<RamChunk>,
        offset: u64,
        size: u64,
        mapped: Mapped,
    ) -> Placement {
        #[cfg(unix)]
        let dma_buf = chunk.dma_buf_pages().cloned();
        Placement {
            type_index: chunk.type_index(),
            holds: Holds::Buffers,
            block: None,
            size,
            memory: chunk.memory(),
            offset,
            mapped: Some(mapped),
            address: None,
            own_pages: false,
            read_only_view: false,
            overflowed: false,
            fresh: false,
            #[cfg(unix)]
            dma_buf,
            ram_chunk: Some(chunk),
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
            overflowed: false,
            fresh: false,
            dma_buf: Some(pages),
            ram_chunk: None,
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

    /// Whether the placement lies in the device's import of a chunk of a guest RAM
    /// (`host_ram.rs`), with no import of its own.
    pub(super) fn over_ram_chunk(&self) -> bool {
        self.ram_chunk.is_some()
    }

    /// Whether the placement lies in the device's import of one object's run of a guest RAM
    /// alone, made while the run's chunk was not imported (`host_ram.rs`).
    pub(super) fn over_ram_run(&self) -> bool {
        (self.ram_chunk.as_ref()).is_some_and(|chunk| chunk.is_run())
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

    /// Whether the placement's bytes read zero: it lies in a block allocated zero-initialized,
    /// past every range placed in that block before ([`Memory::zero_image_blocks`]).
    pub(super) fn fresh(&self) -> bool {
        self.fresh
    }

    /// Whether the placement lies past its class's own memory types, in one it overflowed into
    /// where none of them had room ([`overflow_candidates`]): system memory, where its class
    /// requires device-local memory (`Degradation::DeviceMemoryOverflow`).
    pub(super) fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// The imports of the caller's dma-bufs the placement lies in, with their source
    /// (`dma_buf.rs`): its own, or those of the guest RAM's chunk it lies in (`host_ram.rs`);
    /// `None` for any other placement.
    #[cfg(unix)]
    pub(super) fn dma_buf_pages(&self) -> Option<&Arc<DmaBufPages>> {
        self.dma_buf.as_ref()
    }

    /// The address, among the pages its dma-buf imports were made of ([`dma_buf_pages`]), of
    /// the byte the caller reaches at host address `view`: `view` itself, but in a guest RAM's
    /// chunk (`host_ram.rs`), whose import is of the RAM's own addresses while the caller reaches
    /// the placement's bytes through its own view of them, from `mapped`, the RAM's address of
    /// the same byte. What a texture over the bytes asks the source for its own pages at.
    ///
    /// [`dma_buf_pages`]: Placement::dma_buf_pages
    #[cfg(unix)]
    pub(super) fn dma_buf_address(&self, view: u64) -> u64 {
        match (&self.ram_chunk, self.mapped) {
            (Some(chunk), Some(mapped)) => (chunk.ram_address(self.offset))
                .wrapping_add(view.wrapping_sub(mapped.as_ptr() as u64)),
            _ => view,
        }
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
            overflowed: self.overflowed,
            fresh: self.fresh,
            #[cfg(unix)]
            dma_buf: self.dma_buf.clone(),
            ram_chunk: self.ram_chunk.clone(),
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
    /// As `address_types`, of a buffer of addresses made for [`FD_BLOCKS_HANDLE`], which the
    /// blocks of [`Holds::FdBuffers`] that are allocated exportable carry.
    fd_address_types: Option<u32>,
    /// The memory types whose [`Holds::FdBuffers`] blocks are allocated exportable, as bits, or
    /// why none are ([`fd_buffer_types`]).
    pub(super) fd_buffer_types: Result<u32, &'static str>,
    fd_blocks: FdBlocks,
    /// The pools the runtime's encodings and commits place their own resources in, once asked
    /// ([`Memory::kept`], [`keep_room`]).
    kept: Option<Vec<(Holds, u32)>>,
    /// The device has unified memory (`DeviceProperties::has_unified_memory`), which
    /// [`candidates`] ranks `WriteCombined` memory by.
    unified: bool,
    /// The device's queue families, which its images are shared among (`sharing.rs`).
    families: Vec<u32>,
    /// Those families, then the staging queue's where the device reserved one: what its buffers,
    /// those of addresses among them, are shared among (`sharing.rs`).
    buffer_families: Vec<u32>,
    /// The bytes of the blocks allocated so far ([`Memory::allocated`]).
    allocated: u64,
    /// Where the device's driver fails pending work when memory a device-address buffer lay in
    /// is freed, the record its blocks that go wait by (`freed_memory.rs`).
    parking: Option<Arc<Parking>>,
    /// The blocks that went and wait, their buffers of addresses destroyed.
    parked: Vec<Parked>,
    /// The blocks that waited so far ([`Memory::parked`]).
    parked_count: u64,
    /// The blocks freed so far by [`FreeCause`]: one `vkFreeMemory` each ([`Memory::frees`]).
    freed_by: [u64; FreeCause::COUNT],
    /// The memory log, where `METAL2VULKAN_LOG_MEMORY` is set (`memory_log.rs`).
    log: Option<MemoryLog>,
    /// The device's free thread, which the blocks that go are posted to, once set
    /// ([`Memory::post_frees_to`], `free_thread.rs`).
    frees: Option<Arc<FreeThread>>,
    /// Per memory heap: its size, and the bytes of the blocks of the runtime's own memory live in
    /// it ([`Memory::add_block`]), imports aside.
    heaps: Vec<(u64, u64)>,
    /// A bind into memory that queued work uses waits for that work on the device's driver
    /// ([`binds_wait`]), so an image bound while such work may be queued takes a block of its own
    /// ([`allocate_image_while_queued`]).
    binds_wait: bool,
    /// The placements so far that lie past their class's own memory types, in one they
    /// overflowed into ([`overflow_candidates`], `WorkCounts::device_memory_overflows`).
    overflows: u64,
    /// A test's stand-in for a driver slow to import a dma-buf ([`import_dma_buf`]), called with
    /// the lock released where the driver's import runs.
    #[cfg(test)]
    pub(crate) import_hold: Option<Arc<dyn Fn() + Send + Sync>>,
    /// A test's stand-in for every heap's budget of kept empty blocks ([`Memory::kept_budget`]).
    #[cfg(test)]
    pub(crate) kept_budget_stand_in: Option<u64>,
    /// Blocks of images are allocated with `VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT`
    /// ([`Memory::zero_image_blocks`]).
    zero_images: bool,
    /// The pool a color texture's host-transfer image places in, once asked: `Some(None)` where
    /// the device copies no color texture on the host ([`Memory::host_copy_pool`]).
    host_copy_pool: Option<Option<(Holds, u32)>>,
    /// The blocks being made ahead, or made and not yet in their pool, one a pool at most: each
    /// enters its pool at a call that needs room there (`memory/ready.rs`).
    readying: Vec<ready::Readying>,
    /// Of that pool, its blocks made ready ahead and added to it, and the blocks a placement in it
    /// made itself, finding none ready (`WorkCounts`).
    host_copy_blocks: HostCopyBlocks,
    /// The blocks made ahead and made in place in the pools blocks are made ahead for
    /// (`memory/ready.rs`, `WorkCounts`).
    ahead: AheadBlocks,
    /// When the creation calls next look for stale kept blocks in every pool
    /// ([`Memory::sweep_stale`]).
    next_sweep: Instant,
}

/// The counts of the blocks made ahead (`memory/ready.rs`), and of those made in place in the pools
/// blocks are made ahead for ([`Memory::makes_ahead`]), by the kind of pool.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct AheadBlocks {
    /// Blocks of buffers made ahead and added to their pool.
    pub(super) buffers_ahead: u64,
    /// Blocks of buffers a call made itself, finding none made ahead.
    pub(super) buffers_in_place: u64,
    /// Blocks of images made ahead and added to their pool.
    pub(super) images_ahead: u64,
    /// Blocks of images a call made itself, finding none made ahead.
    pub(super) images_in_place: u64,
    /// Of the blocks made ahead, those a call waited for, their making under way.
    pub(super) waited_for: u64,
    /// Of those, the blocks whose making a call's wait cut short in its wait for a burst of
    /// submissions to end, so that the call waited for the allocation alone.
    pub(super) hurried: u64,
    /// Of the blocks made ahead, those whose wait for a burst of submissions to end reached its
    /// bound (`ready::QUIET_MOST`).
    pub(super) past_quiet_wait: u64,
    /// The blocks made ahead that went unused, their pool no longer growing ([`Memory::sweep_stale`]).
    pub(super) unused: u64,
}

impl AheadBlocks {
    /// A block of a pool of `holds` was made ahead and added, a call having waited for it where
    /// `waited_for`, after its wait for a burst's end ended as `waited` says.
    fn made_ahead(&mut self, holds: Holds, waited_for: bool, waited: Waited) {
        match holds {
            Holds::Images => self.images_ahead += 1,
            _ => self.buffers_ahead += 1,
        }
        self.waited_for += u64::from(waited_for);
        self.hurried += u64::from(waited == Waited::Asked);
        self.past_quiet_wait += u64::from(waited == Waited::Bound);
    }

    /// A call made a block of a pool of `holds` itself.
    fn made_in_place(&mut self, holds: Holds) {
        match holds {
            Holds::Images => self.images_in_place += 1,
            _ => self.buffers_in_place += 1,
        }
    }
}

/// The counts of the blocks of [`Holds::FdBuffers`] made so far.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct FdBlocks {
    /// Allocated exportable as a dma-buf: of a memory type [`fd_buffer_types`]
    /// names.
    pub(super) exportable: u64,
    /// Allocated as any block of buffers: of a memory type it does not name (device-local memory
    /// the program's buffers overflowed into, or that their class takes first).
    pub(super) plain: u64,
}

/// The counts of the blocks of the pool host copies place in ([`Memory::host_copy_pool`]).
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct HostCopyBlocks {
    /// Made ready ahead off the program's thread and added to the pool by a placement that found
    /// no room (`memory/ready.rs`).
    pub(super) readied: u64,
    /// Blocks a placement made itself, finding no room and none made ready.
    pub(super) made_in_place: u64,
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
        let heaps = properties.memory_heaps
            [..(properties.memory_heap_count as usize).min(vk::MAX_MEMORY_HEAPS)]
            .iter()
            .map(|heap| (heap.size, 0))
            .collect();
        Memory {
            pools: (Holds::ALL.iter().flat_map(|_| &types))
                .map(|_| Pool::default())
                .collect(),
            types,
            block_bytes: BLOCK_BYTES.min(max_allocation.max(1)),
            device_address,
            address_types: None,
            fd_address_types: None,
            fd_buffer_types: Err("the device was not asked for exportable buffer memory"),
            fd_blocks: FdBlocks::default(),
            kept: None,
            unified,
            buffer_families: families.clone(),
            families,
            allocated: 0,
            parking,
            parked: Vec::new(),
            parked_count: 0,
            freed_by: [0; FreeCause::COUNT],
            log: None,
            frees: None,
            heaps,
            binds_wait: binds_wait(driver),
            overflows: 0,
            zero_images: false,
            host_copy_pool: None,
            readying: Vec::new(),
            host_copy_blocks: HostCopyBlocks::default(),
            ahead: AheadBlocks::default(),
            next_sweep: Instant::now() + KEPT_SWEEP_EVERY,
            #[cfg(test)]
            import_hold: None,
            #[cfg(test)]
            kept_budget_stand_in: None,
        }
    }

    /// Allocate every block of images from now on with
    /// `VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT`, for a device that enabled
    /// `zeroInitializeDeviceMemory`: a placement in such a block past
    /// every range placed in it before reads zero ([`Placement::fresh`]), which an image made in
    /// `VK_IMAGE_LAYOUT_ZERO_INITIALIZED_EXT` keeps as it moves out of it
    /// (`texture/host_copy.rs`). The blocks are as many as without it: only ranges no placement
    /// used read zero, and an image placed in any other starts as before.
    pub(super) fn zero_image_blocks(&mut self) {
        self.zero_images = true;
    }

    /// Share its buffers among `families`: the device's, then its staging queue's (`sharing.rs`).
    pub(super) fn share_buffers_among(&mut self, families: Vec<u32>) {
        self.buffer_families = families;
    }

    /// Allocate the blocks of [`Holds::FdBuffers`] of the memory types `types` names (as bits)
    /// exportable as [`FD_BLOCKS_HANDLE`] from now on, where it names any, else note why not (the
    /// memory log says which), as the device is made: `buffer.rs`'s `placed_new_buffer` then
    /// makes its buffers for that handle type and places them there once the device's check of
    /// such blocks passed ([`Memory::own_buffers`]).
    pub(super) fn fd_buffer_blocks(&mut self, types: Result<u32, &'static str>) {
        if let Some(log) = &self.log {
            log.line(match types {
                Ok(types) => format!(
                    "the runtime's own buffers lie in blocks of memory types {types:#x} exportable \
                     as dma-bufs once the device's check of them passed"
                ),
                Err(why) => format!("the runtime's own buffers lie in plain blocks: {why}"),
            });
        }
        self.fd_buffer_types = types;
    }

    /// The kind of block the buffers `buffer.rs`'s `placed_new_buffer` makes for the program and
    /// for the runtime take: [`Holds::FdBuffers`] where the device allocates any of them
    /// exportable ([`Memory::fd_buffer_blocks`]) and its check of such blocks passed (`checked`,
    /// `in_place.rs`'s `Device::fd_blocks_checked`), else [`Holds::Buffers`].
    pub(super) fn own_buffers(&self, checked: bool) -> Holds {
        match self.fd_buffer_types {
            Ok(_) if checked => Holds::FdBuffers,
            Ok(_) | Err(_) => Holds::Buffers,
        }
    }

    /// The counts of the blocks of [`Holds::FdBuffers`] made so far.
    pub(super) fn fd_blocks(&self) -> FdBlocks {
        self.fd_blocks
    }

    /// Log each `VkDeviceMemory` allocated, imported or freed from now on to `log`
    /// (`memory_log.rs`).
    pub(super) fn log_to(&mut self, log: Option<MemoryLog>) {
        self.log = log;
    }

    /// Post the frees of the blocks that go from now on to `frees` (`free_thread.rs`).
    pub(super) fn post_frees_to(&mut self, frees: Arc<FreeThread>) {
        self.frees = Some(frees);
    }

    /// The memory heap of memory type `type_index`.
    fn heap_of(&self, type_index: u32) -> usize {
        self.types[type_index as usize].heap_index as usize
    }

    /// The free thread to wait for before a block of `size` bytes of memory type `type_index` is
    /// made: where frees of its heap are posted and the heap's live blocks, those frees and the new
    /// block would pass the heap's size (`free_thread.rs`).
    fn frees_before(&self, type_index: u32, size: u64) -> Option<Arc<FreeThread>> {
        let frees = self.frees.as_ref()?;
        let heap = self.heap_of(type_index);
        let pending = frees.heap_pending(heap);
        let (heap_size, live) = *self.heaps.get(heap)?;
        let over = live.saturating_add(pending).saturating_add(size) > heap_size;
        (pending > 0 && over).then(|| Arc::clone(frees))
    }

    /// The device's free thread, once set.
    fn posted_frees(&self) -> Option<Arc<FreeThread>> {
        self.frees.clone()
    }

    /// The placements so far that lie past their class's own memory types, in one they
    /// overflowed into ([`overflow_candidates`]).
    pub(super) fn overflows(&self) -> u64 {
        self.overflows
    }

    /// `placement`, marked and counted as overflowed where it lies in a type its class
    /// overflowed into (`overflowing`, [`place`]).
    fn mark_overflowed(&mut self, mut placement: Placement, overflowing: bool) -> Placement {
        if overflowing {
            placement.overflowed = true;
            self.overflows += 1;
        }
        placement
    }

    /// The most bytes of empty blocks heap `heap` keeps for reuse ([`Pool::release`],
    /// [`kept_budget`]). Its pools' first [`KEPT_EMPTY_BLOCKS`] empty pooled blocks are kept
    /// whatever it says, and count against it.
    fn kept_budget(&self, heap: usize) -> u64 {
        #[cfg(test)]
        if let Some(budget) = self.kept_budget_stand_in {
            return budget;
        }
        let sizes: Vec<u64> = self.heaps.iter().map(|&(size, _)| size).collect();
        kept_budget(&sizes, heap, self.unified)
    }

    /// The bytes of the empty blocks kept against heap `heap`'s budget: those of its pools, and
    /// with unified memory those of every heap's ([`kept_budget`]), and of the blocks made ahead
    /// for them and not yet in them (`memory/ready.rs`).
    fn kept_in_heap(&self, heap: usize) -> u64 {
        let count = self.types.len();
        let in_heap = |type_index: usize| {
            self.unified
                || (self.types.get(type_index)).is_some_and(|ty| ty.heap_index as usize == heap)
        };
        let kept: u64 = (self.pools.iter().enumerate())
            .filter(|&(at, _)| in_heap(at % count))
            .map(|(_, pool)| pool.kept_bytes())
            .sum();
        let ahead = (self.readying.iter())
            .filter(|ready| in_heap(ready.pool().1 as usize))
            .count() as u64;
        kept + ahead * self.block_bytes
    }

    /// A block of `size` bytes of memory type `type_index` left the heap's live blocks.
    fn left_heap(&mut self, type_index: u32, size: u64) {
        let heap = self.heap_of(type_index);
        if let Some((_, live)) = self.heaps.get_mut(heap) {
            *live = live.saturating_sub(size);
        }
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

    /// The blocks this allocator freed so far, by [`FreeCause`]: one `vkFreeMemory` each,
    /// counted where the block went. Every block it frees falls under exactly one cause, so
    /// their sum is its whole share of the process's `vkFreeMemory` calls
    /// (`WorkCounts::object_churn`'s `vk_memory_freed`); what is left of that total was freed
    /// outside it (a dma-buf import, a GPU pool's block, a neutral or a spare).
    pub(super) fn frees(&self) -> [u64; FreeCause::COUNT] {
        self.freed_by
    }

    /// Hold each memory heap to its live blocks and the frees of it posted, from now on: the next
    /// block made in it passes its size (`free_thread.rs`). The heaps' sizes, to put back.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(super) fn hold_heaps_to_live(&mut self) -> Vec<u64> {
        let frees = self.frees.clone();
        (self.heaps.iter_mut().enumerate())
            .map(|(heap, (size, live))| {
                let pending = frees.as_ref().map_or(0, |frees| frees.heap_pending(heap));
                let was = *size;
                *size = *live + pending;
                was
            })
            .collect()
    }

    /// Put the memory heaps' sizes back to `sizes`
    /// ([`hold_heaps_to_live`](Self::hold_heaps_to_live)).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(super) fn put_heaps_back(&mut self, sizes: &[u64]) {
        for ((size, _), &was) in self.heaps.iter_mut().zip(sizes) {
            *size = was;
        }
    }

    /// Where the device's memory may lie, for the crate's device tests: its memory types, its
    /// heaps' sizes, and the types a buffer the runtime places and a color image of optimal tiling
    /// take ([`Memory::kept`]'s queries).
    #[cfg(all(test, any(m2v_cpu_icd, all(m2v_gpu, m2v_gpu_hazards))))]
    pub(super) fn type_view(&self, device: &ash::Device) -> Result<TypeView, Error> {
        let sharing = Sharing::with_buffers(&self.families, &self.buffer_families);
        Ok(TypeView {
            types: self.types.clone(),
            heap_sizes: self.heaps.iter().map(|&(size, _)| size).collect(),
            buffers: buffer_types(device, usage(false), sharing, None)?,
            images: color_image_types(device, sharing)?,
        })
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

    /// The memory class and types, as bits, an image of `class` with host-transfer usage whose
    /// requirements allow `allowed` takes (`texture/host_copy.rs`), and whether they are types the
    /// host caches: never one the host maps without coherence ([`host_incoherent`]); those the
    /// host caches where the image may take one ([`host_cached_placement`]); else `class` and
    /// every other type allowed.
    pub(super) fn host_image_types(
        &self,
        allowed: u32,
        class: MemoryClass,
    ) -> (MemoryClass, u32, bool) {
        let allowed = allowed & !self.host_incoherent();
        match self.host_cached_placement(allowed, class) {
            Some((class, types)) => (class, types, true),
            None => (class, allowed, false),
        }
    }

    /// The pool a color texture's host-transfer image places in, found once, where `copies` (read
    /// only then) says the device copies color textures on the host (`texture/host_copy.rs`): of
    /// images, the best type of the class an `R8G8B8A8_UNORM` image of optimal tiling with
    /// host-transfer usage takes ([`Memory::host_image_types`], of `Private`, the class of every texture's image but a
    /// memoryless one's). Its blocks are made ready ahead (`memory/ready.rs`). `None` where the
    /// device copies none, or the query fails.
    fn host_copy_pool(&mut self, device: &ash::Device, copies: bool) -> Option<(Holds, u32)> {
        if let Some(pool) = self.host_copy_pool {
            return pool;
        }
        let pool = match copies {
            true => (host_copy_image_types(device, Sharing::new(&self.families)).ok())
                .and_then(|allowed| {
                    let (class, types, _) = self.host_image_types(allowed, MemoryClass::Private);
                    candidates(&self.types, types, class, self.unified)
                        .first()
                        .copied()
                })
                .map(|type_index| (Holds::Images, type_index)),
            false => None,
        };
        self.host_copy_pool = Some(pool);
        pool
    }

    /// The counts of the blocks of the pool host copies place in ([`HostCopyBlocks`]).
    pub(super) fn host_copy_blocks(&self) -> HostCopyBlocks {
        self.host_copy_blocks
    }

    /// The counts of the blocks made ahead and in place ([`AheadBlocks`]).
    pub(super) fn ahead_blocks(&self) -> AheadBlocks {
        self.ahead
    }

    /// Whether no block is being made ahead: none is held, or every making ended
    /// (`memory/ready.rs`).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(super) fn readying_ended(&self) -> bool {
        self.readying.iter().all(ready::Readying::finished)
    }

    /// The place of the block made ahead for `pool` among those held, where one is.
    fn readying_at(&self, pool: (Holds, u32)) -> Option<usize> {
        self.readying.iter().position(|ready| ready.pool() == pool)
    }

    /// Whether blocks are made ahead for `pool` (`memory/ready.rs`): one the encodings and commits
    /// place their own resources in ([`Memory::kept`], once found), or the pool host copies place
    /// in ([`Memory::host_copy_pool`], once found).
    fn makes_ahead(&self, pool: (Holds, u32)) -> bool {
        self.host_copy_pool == Some(Some(pool))
            || (self.kept.as_ref()).is_some_and(|kept| kept.contains(&pool))
    }

    /// Add the block made ahead for `pool` to it where its making ended (`memory/ready.rs`):
    /// whether one was added. A making that failed is let go of.
    fn add_ready(&mut self, pool: (Holds, u32)) -> bool {
        let Some(at) = self.readying_at(pool) else {
            return false;
        };
        if !self.readying[at].finished() {
            return false;
        }
        let readying = self.readying.swap_remove(at);
        let waited_for = readying.waited_for();
        let Some(made) = readying.take() else {
            return false;
        };
        let bytes = self.block_bytes;
        self.add_block(pool, made.block, bytes, false, 0);
        self.ahead.made_ahead(pool.0, waited_for, made.waited);
        if self.host_copy_pool == Some(Some(pool)) {
            self.host_copy_blocks.readied += 1;
        }
        true
    }

    /// The memory types of `class` among those `allowed` names, as bits.
    pub(super) fn class_types(&self, allowed: u32, class: MemoryClass) -> u32 {
        candidates(&self.types, allowed, class, self.unified)
            .into_iter()
            .fold(0, |bits, index| bits | 1 << index)
    }

    /// The index in `pools` of the pool of blocks of memory type `type_index` that hold `holds`.
    fn pool_at(&self, holds: Holds, type_index: u32) -> usize {
        type_index as usize + holds as usize * self.types.len()
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

    /// The memory types a buffer of addresses of a block that holds `holds` takes, as bits,
    /// asked once: `None` without `bufferDeviceAddress`, where blocks carry no buffer of
    /// addresses. Of [`Holds::FdBuffers`] on a device that allocates its blocks exportable, a
    /// buffer made for their handle type ([`FD_BLOCKS_HANDLE`]).
    fn address_types(&mut self, device: &ash::Device, holds: Holds) -> Result<Option<u32>, Error> {
        if !self.device_address {
            return Ok(None);
        }
        let fd = holds == Holds::FdBuffers && self.fd_buffer_types.is_ok();
        let known = match fd {
            true => self.fd_address_types,
            false => self.address_types,
        };
        if let Some(types) = known {
            return Ok(Some(types));
        }
        let sharing = Sharing::with_buffers(&self.families, &self.buffer_families);
        let handle = fd.then_some(FD_BLOCKS_HANDLE);
        let types = buffer_types(device, usage(true), sharing, handle)?;
        match fd {
            true => self.fd_address_types = Some(types),
            false => self.address_types = Some(types),
        }
        Ok(Some(types))
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
        let Some((block, offset, fresh)) = self.pool(holds, type_index).place_fresh(size, align)
        else {
            return Ok(None);
        };
        if let Some(counts) = addressed {
            if let Err(error) = self.address_block(device, (holds, type_index), block, counts) {
                drop(self.release_range(device, (holds, type_index), block, offset, size, None));
                return Err(error);
            }
        }
        let addressed = addressed.is_some();
        let mut placement = self.placement((holds, type_index), block, offset, size, addressed);
        placement.fresh = fresh;
        self.say_placement("placed", &placement, Some(class));
        Ok(Some(placement))
    }

    /// [`Memory::place_pooled`] for a dedicated placement of `size` bytes: in a kept dedicated
    /// block of the pool that its size fits ([`Pool::place_dedicated`]), given a buffer of
    /// addresses where `addressed` and it has none. `None` where the pool keeps no such block.
    fn place_kept_dedicated(
        &mut self,
        device: &ash::Device,
        (holds, type_index): (Holds, u32),
        size: u64,
        addressed: Option<&Counters>,
        class: MemoryClass,
    ) -> Result<Option<Placement>, Error> {
        let Some(block) = self.pool(holds, type_index).place_dedicated(size) else {
            return Ok(None);
        };
        if let Some(counts) = addressed {
            if let Err(error) = self.address_block(device, (holds, type_index), block, counts) {
                drop(self.release_range(device, (holds, type_index), block, 0, size, None));
                return Err(error);
            }
        }
        let placement = self.placement((holds, type_index), block, 0, size, addressed.is_some());
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
                (false, Holds::Exports) => "made a pooled block of exportable buffers",
                (false, Holds::DmaBufs) => "made a pooled block of dma-buf-exportable buffers",
                (false, Holds::FdBuffers) if block.fd => {
                    "made a pooled block of fd-exportable buffers"
                }
                (false, Holds::FdBuffers) => "made a pooled block of buffers",
                (true, Holds::Buffers) => "made a dedicated block of a buffer",
                (true, Holds::Images) => "made a dedicated block of an image",
                (true, Holds::Exports) => "made a dedicated block of an exportable buffer",
                (true, Holds::DmaBufs) => "made a dedicated block of a dma-buf-exportable buffer",
                (true, Holds::FdBuffers) if block.fd => {
                    "made a dedicated block of an fd-exportable buffer"
                }
                (true, Holds::FdBuffers) => "made a dedicated block of a buffer",
            };
            self.say_block(what, &block);
        }
        if holds == Holds::FdBuffers {
            match block.fd {
                true => self.fd_blocks.exportable += 1,
                false => self.fd_blocks.plain += 1,
            }
        }
        self.allocated += size;
        let heap = self.heap_of(type_index);
        if let Some((_, live)) = self.heaps.get_mut(heap) {
            *live += size;
        }
        let zeroed = block.zeroed;
        let pool = self.pool(holds, type_index);
        if !dedicated {
            pool.grew(Some(Instant::now()));
        }
        pool.add_as(size, dedicated, block, used, zeroed)
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
            fd_types: self.fd_buffer_types.unwrap_or(0),
            device_address: self.device_address,
            families: self.buffer_families.clone(),
            zero_images: self.zero_images,
        }
    }

    /// The pools the runtime's encodings and commits place their own resources in, found once:
    /// of buffers, the best memory type of `Private`, `WriteCombined` and `Shared` memory among
    /// those a buffer the runtime places takes (`buffer::usage(false)`, its address from a buffer
    /// of addresses, [`allocate_addressed`]), in the blocks of [`Holds::Buffers`] and, where the
    /// device allocates blocks of [`Holds::FdBuffers`] exportable, in those too, of a buffer made
    /// for [`FD_BLOCKS_HANDLE`]; of images, the best `Private` type among those a color image of
    /// optimal tiling takes, which every such image takes (the Vulkan specification's
    /// `VkMemoryRequirements`).
    fn kept(&mut self, device: &ash::Device) -> Result<Vec<(Holds, u32)>, Error> {
        if let Some(kept) = &self.kept {
            return Ok(kept.clone());
        }
        let addresses = self
            .address_types(device, Holds::Buffers)?
            .unwrap_or(u32::MAX);
        let fd = match self.fd_buffer_types {
            Ok(_) => Some(
                self.address_types(device, Holds::FdBuffers)?
                    .unwrap_or(u32::MAX),
            ),
            Err(_) => None,
        };
        let sharing = Sharing::with_buffers(&self.families, &self.buffer_families);
        let buffers = buffer_types(device, usage(false), sharing, None)? & addresses;
        let fd_buffers = match fd {
            Some(addresses) => {
                buffer_types(device, usage(false), sharing, Some(FD_BLOCKS_HANDLE))? & addresses
            }
            None => 0,
        };
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
        let fd_pools = (classes.into_iter())
            .filter_map(|class| best(fd_buffers, class))
            .map(|type_index| (Holds::FdBuffers, type_index));
        let image_pool =
            best(images, MemoryClass::Private).map(|type_index| (Holds::Images, type_index));
        for pool in buffer_pools.chain(fd_pools).chain(image_pool) {
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
            overflowed: false,
            fresh: false,
            #[cfg(unix)]
            dma_buf: None,
            ram_chunk: None,
        }
    }

    /// Give pooled block `block` of the pool of memory type `type_index` that holds `holds` a
    /// buffer of addresses where it has none: over its whole size, which it takes since the
    /// block is of a type such a buffer takes ([`allocate_addressed`]), made for the block's
    /// handle type where it is exportable (`VUID-vkBindBufferMemory-memory-02726`); refused (an
    /// invariant) where the buffer would need more bytes than the block has.
    fn address_block(
        &mut self,
        device: &ash::Device,
        (holds, type_index): (Holds, u32),
        block: usize,
        counts: &Counters,
    ) -> Result<(), Error> {
        let at = self.pool_at(holds, type_index);
        let sharing = Sharing::with_buffers(&self.families, &self.buffer_families);
        let payload = self.pools[at]
            .payload_mut(block)
            .expect("a placed block is live");
        if payload.addresses.is_some() {
            return Ok(());
        }
        let handle = payload.fd.then_some(FD_BLOCKS_HANDLE);
        let (raw, required) = address_buffer(device, payload.size, sharing, handle)?;
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
    /// block when it empties and is not kept ([`Pool::release`], within its heap's
    /// [`Memory::kept_budget`]); `after` goes once it is freed. `after`, given back for the caller
    /// to let go of with no lock held, unless the block waits with it. A kept empty block of the
    /// pool that stayed empty too long goes too ([`Pool::stale`]).
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
        let heap = self.heap_of(type_index);
        let spare = self
            .kept_budget(heap)
            .saturating_sub(self.kept_in_heap(heap));
        let half = self.block_bytes / 2;
        let reusable = |block: &Block| block.reusable() && block.size > half;
        let pool = self.pool(holds, type_index);
        let freed = pool.release(block, (offset, size), now, spare, reusable);
        let stale = pool.stale(now);
        if let Some(stale) = stale {
            // SAFETY: nothing is placed in a kept empty block, and no command reads it.
            drop(unsafe { self.let_go(device, stale, FreeCause::KeptStale, None) });
        }
        match freed {
            Some((block, dedicated)) => {
                let cause = match dedicated {
                    true => FreeCause::Dedicated,
                    false => FreeCause::PooledEmptied,
                };
                // SAFETY: no resource is placed in the block any more.
                unsafe { self.let_go(device, block, cause, after) }
            }
            None => after,
        }
    }

    /// Free the kept empty blocks of every pool that stayed empty for [`KEPT_EMPTY_FOR`]
    /// ([`Pool::stale`]), at most once every [`KEPT_SWEEP_EVERY`] (at `now`): a pool whose
    /// resources no longer come and go keeps none past its quiet period but its last pooled one.
    fn sweep_stale(&mut self, device: &ash::Device, now: Instant) {
        if now < self.next_sweep {
            return;
        }
        self.next_sweep = now + KEPT_SWEEP_EVERY;
        for at in 0..self.pools.len() {
            while let Some(stale) = self.pools[at].stale(now) {
                // SAFETY: nothing is placed in a kept empty block, and no command reads it.
                drop(unsafe { self.let_go(device, stale, FreeCause::KeptStale, None) });
            }
        }
        // A block made ahead for a pool that stopped growing that long ago, where the pool has
        // room for half a block (`memory/ready.rs`).
        let room = self.block_bytes / 2;
        let mut at = 0;
        while at < self.readying.len() {
            let (holds, type_index) = self.readying[at].pool();
            let started = self.readying[at].started();
            let pool = self.pool(holds, type_index);
            let since = pool.grew(None).map_or(started, |grew| grew.max(started));
            let unused = now.saturating_duration_since(since) >= KEPT_EMPTY_FOR
                && pool.has_room(room)
                && self.readying[at].finished();
            if !unused {
                at += 1;
                continue;
            }
            if let Some(made) = self.readying.swap_remove(at).take() {
                self.ahead.unused += 1;
                // SAFETY: nothing was placed in the block, and no command reads it.
                drop(unsafe { self.let_go(device, made.block, FreeCause::KeptStale, None) });
            }
        }
    }

    /// Release every kept empty block of memory type `type_index`, pooled and dedicated, of every
    /// kind ([`Holds`]), as a block of that type fails to be made ([`place`]): how many went.
    fn release_kept(&mut self, device: &ash::Device, type_index: u32) -> usize {
        let blocks: Vec<Block> = (Holds::ALL.into_iter())
            .flat_map(|holds| self.pool(holds, type_index).take_kept())
            .collect();
        let count = blocks.len();
        for block in blocks {
            // SAFETY: nothing is placed in a kept empty block, and no command reads it.
            drop(unsafe { self.let_go(device, block, FreeCause::KeptForRoom, None) });
        }
        count
    }

    /// Over every pool: the block slots the room queries have walked and the free ranges they
    /// have visited ([`Pool::room_work`]), then what one query of each pool would walk and visit
    /// where it derived per call ([`Pool::room_per_call`]).
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(super) fn room_work(&self) -> (u64, u64, u64, u64) {
        (self.pools.iter()).fold((0, 0, 0, 0), |(w, v, lw, lv), pool| {
            let (walked, visited) = pool.room_work();
            let (live, ranges) = pool.room_per_call();
            (w + walked, v + visited, lw + live, lv + ranges)
        })
    }

    /// The pooled blocks made and freed so far, of every pool, with those made ahead that went
    /// unused (`memory/ready.rs`).
    pub(super) fn turnover(&self) -> (u64, u64) {
        let unused = self.ahead.unused;
        (self.pools.iter().map(Pool::turnover)).fold((unused, unused), |(made, freed), (m, f)| {
            (made + m, freed + f)
        })
    }

    /// Free `block`, or where the device parks freed memory (`freed_memory.rs`), destroy its
    /// buffer of addresses and let its memory wait, with `after`, for the batches submitted
    /// before now, unless they completed. `after`, given back where the memory was freed here.
    /// A block freed now that is no import, with nothing to go after it, is posted to the
    /// device's free thread, unless its queue is full (`free_thread.rs`).
    ///
    /// The block's `vkFreeMemory` is counted under `cause`, or, where the block is an import of
    /// its own, under the import's cause instead: counted here, where the block goes, whether
    /// the call runs now, on the free thread or once the batches in flight complete
    /// ([`Memory::frees`]).
    ///
    /// # Safety
    ///
    /// Nothing is placed in the block any more, and no command that reads it is pending.
    unsafe fn let_go(
        &mut self,
        device: &ash::Device,
        block: Block,
        cause: FreeCause,
        after: Option<Box<dyn Send>>,
    ) -> Option<Box<dyn Send>> {
        self.freed_by[block.cause(cause) as usize] += 1;
        // An import was never among the heap's blocks ([`Memory::add_import`]).
        if !block.imported && block.pages.is_none() {
            self.left_heap(block.type_index, block.size);
        }
        // Freed on the free thread where it may be: no import of the caller's memory (a view of
        // the caller's pages is one), with nothing of the caller's to go after it
        // (`free_thread.rs`).
        let frees = match (block.imported, &after) {
            (false, None) => self.frees.clone(),
            _ => None,
        };
        let (heap, size) = (self.heap_of(block.type_index), block.size);
        let Some(parking) = &self.parking else {
            self.say_block("freed a block", &block);
            let Some(frees) = frees else {
                // SAFETY: the caller's.
                unsafe { block.release(device) };
                return after;
            };
            let Block {
                memory,
                addresses,
                pages,
                ..
            } = block;
            let posted = Posted {
                memory,
                addresses: addresses.map(|addresses| addresses.raw),
                after: pages.map(|pages| Box::new(pages) as Box<dyn Send>),
                heap,
                size,
            };
            // SAFETY: the caller's.
            return unsafe { frees.post_or_free(posted, device) };
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
        // Every batch that may list it completed.
        let Some(frees) = frees else {
            // SAFETY: the caller's; every batch that may list it completed.
            unsafe { release::object(device, memory) };
            return after;
        };
        let posted = Posted {
            memory,
            addresses: None,
            after,
            heap,
            size,
        };
        // SAFETY: the caller's; its buffer of addresses went, and every batch that may list it
        // completed.
        unsafe { frees.post_or_free(posted, device) }
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

    /// The property flags of memory type `index`, where the device has it.
    pub(super) fn type_flags(&self, index: u32) -> Option<vk::MemoryPropertyFlags> {
        self.types
            .get(index as usize)
            .map(|memory_type| memory_type.property_flags)
    }

    /// What `placement`'s block was allocated exportable as ([`allocate_exported`]), with the
    /// block's size, the bytes of its allocation; `None` for any other placement.
    pub(super) fn exported(&self, placement: &Placement) -> Option<(Exported, u64)> {
        let block =
            (self.pool_ref(placement.holds, placement.type_index)).payload(placement.block?)?;
        block.exported.map(|exported| (exported, block.size))
    }

    /// The external memory `placement`'s memory is, from its own record, which an image bound in
    /// it names (`VkExternalMemoryImageCreateInfo::handleTypes`): an exported block's handle type
    /// (`VUID-vkBindImageMemory-memory-02728`), an import's (`VUID-vkBindImageMemory-memory-02989`:
    /// host allocations for the caller's pages imported by their pointer, a read-only view of them,
    /// pages the runtime mapped and imported, and a guest RAM's chunk imported so; dma-bufs for the
    /// caller's dma-bufs and a chunk imported as one); none for a block the device allocated
    /// with no export. No image at all in a buffer's dedicated allocation
    /// (`VUID-vkBindImageMemory-memory-10926`) or in memory the caller made and keeps, whose
    /// handle types the runtime does not know.
    pub(super) fn bound_memory(&self, placement: &Placement) -> BoundMemory {
        #[cfg(unix)]
        if placement.dma_buf.is_some() {
            return BoundMemory::External(DMA_BUF_HANDLE);
        }
        if let Some(chunk) = &placement.ram_chunk {
            return BoundMemory::External(chunk.handle_type());
        }
        let Some(index) = placement.block else {
            return BoundMemory::NoImage(NoImage::Caller);
        };
        let pool = self.pool_ref(placement.holds, placement.type_index);
        let Some(block) = pool.payload(index) else {
            return BoundMemory::NoImage(NoImage::Caller);
        };
        match block.exported {
            Some(Exported {
                dedicated: true, ..
            }) => BoundMemory::NoImage(NoImage::Dedicated),
            Some(exported) => BoundMemory::External(exported.handle),
            None if block.fd => BoundMemory::Blocks,
            None if block.imported || block.pages.is_some() || block.view.is_some() => {
                BoundMemory::External(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT)
            }
            None => BoundMemory::Own,
        }
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
        // The blocks being made ahead, those whose making began waited for: nothing is placed in
        // them.
        let readied: Vec<Block> = (self.readying.drain(..))
            .filter_map(ready::Readying::stop)
            .collect();
        let blocks: Vec<Block> = (self.pools.iter_mut())
            .flat_map(|pool| pool.drain())
            .chain(readied)
            .collect();
        for block in blocks {
            self.say_block("freed a block as the device is destroyed", &block);
            self.freed_by[block.cause(FreeCause::Teardown) as usize] += 1;
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
    /// The memory types whose [`Holds::FdBuffers`] blocks are allocated exportable, as bits.
    fd_types: u32,
    /// As [`Memory`]'s.
    device_address: bool,
    /// The families its buffers of addresses are shared among: [`Memory`]'s buffers'.
    families: Vec<u32>,
    /// As [`Memory`]'s.
    zero_images: bool,
}

impl BlockMaker {
    /// A new block of `size` bytes of memory type `type_index` that holds `holds`, with a buffer
    /// of addresses where `addressed` gives the counts of such buffers made (its memory then as
    /// long as that buffer requires, if longer), allocated zero-initialized where it holds images
    /// on a device whose blocks of images are ([`Memory::zero_image_blocks`]), and exportable as
    /// the handle type of the exportable buffers it holds ([`Holds::handle`]), or as the runtime's
    /// own buffers' handle type ([`FD_BLOCKS_HANDLE`]) where it is a block of
    /// [`Holds::FdBuffers`] of a memory type [`fd_buffer_types`] names.
    fn make(
        &self,
        device: &ash::Device,
        (holds, type_index): (Holds, u32),
        size: u64,
        addressed: Option<&Counters>,
    ) -> Result<Block, Error> {
        let zeroed = holds == Holds::Images && self.zero_images;
        let export = match holds.handle() {
            Some(handle) => Exportable::For(
                vk::Buffer::null(),
                Exported {
                    handle,
                    dedicated: false,
                },
            ),
            None if holds == Holds::FdBuffers && self.fd_types & 1 << type_index != 0 => {
                Exportable::Fd
            }
            None => Exportable::No,
        };
        self.make_as(device, type_index, size, addressed, export, zeroed)
    }

    /// [`make`](Self::make), its memory allocated exportable as `export` says, the dedicated
    /// allocation of its buffer where it says so ([`allocate_exported`]), and with
    /// `VK_MEMORY_ALLOCATE_ZERO_INITIALIZE_BIT_EXT` where `zeroed`. Its buffer of addresses is
    /// made for the handle type it is exportable as (`VUID-vkBindBufferMemory-memory-02726`).
    fn make_as(
        &self,
        device: &ash::Device,
        type_index: u32,
        size: u64,
        addressed: Option<&Counters>,
        export: Exportable,
        zeroed: bool,
    ) -> Result<Block, Error> {
        let addresses = if addressed.is_some() && self.device_address {
            let sharing = Sharing::new(&self.families);
            Some(address_buffer(device, size, sharing, export.handle())?)
        } else {
            None
        };
        let release_addresses = || {
            if let Some((raw, _)) = addresses {
                // SAFETY: the buffer is unused.
                unsafe { release::object(device, raw) };
            }
        };
        let mut allocate = vk::MemoryAllocateFlags::empty();
        if self.device_address {
            allocate |= vk::MemoryAllocateFlags::DEVICE_ADDRESS;
        }
        if zeroed {
            allocate |= ZERO_INITIALIZE;
        }
        let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(allocate);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(addresses.map_or(size, |(_, required)| size.max(required)))
            .memory_type_index(type_index);
        if !allocate.is_empty() {
            info = info.push_next(&mut flags);
        }
        let handle = export.handle();
        let mut exportable =
            vk::ExportMemoryAllocateInfo::default().handle_types(handle.unwrap_or_default());
        let dedicated_to = match export {
            Exportable::For(buffer, exported) if exported.dedicated => buffer,
            _ => vk::Buffer::null(),
        };
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(dedicated_to);
        if handle.is_some() {
            info = info.push_next(&mut exportable);
        }
        if dedicated_to != vk::Buffer::null() {
            info = info.push_next(&mut dedicated);
        }
        // SAFETY: `info` and its chain live for the call; the type index is the device's; an
        // export's handle type and dedicated buffer are the caller's ([`allocate_exported`]).
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
            exported: match export {
                Exportable::For(_, exported) => Some(exported),
                Exportable::No | Exportable::Fd => None,
            },
            fd: matches!(export, Exportable::Fd),
            zeroed,
        })
    }
}

/// The memory types a buffer of `usage` takes, made for `handle` where given, as bits: those of
/// one made for the query and destroyed at once, with no memory bound. Every buffer of one usage
/// and handle type takes the same types.
fn buffer_types(
    device: &ash::Device,
    usage: vk::BufferUsageFlags,
    sharing: Sharing<'_>,
    handle: Option<vk::ExternalMemoryHandleTypeFlags>,
) -> Result<u32, Error> {
    let mut external =
        vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle.unwrap_or_default());
    let mut info = sharing.buffer(vk::BufferCreateInfo::default().size(16).usage(usage));
    if handle.is_some() {
        info = info.push_next(&mut external);
    }
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
    let usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
    image_types(device, sharing, usage)
}

/// The memory types an `R8G8B8A8_UNORM` image of optimal tiling with host-transfer usage takes,
/// sampled and copied to and from (`texture/host_copy.rs`), as bits ([`Memory::host_copy_pool`]),
/// on a device that enabled `hostImageCopy` and whose format has `HOST_IMAGE_TRANSFER` at optimal
/// tiling.
fn host_copy_image_types(device: &ash::Device, sharing: Sharing<'_>) -> Result<u32, Error> {
    let usage = vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST
        | vk::ImageUsageFlags::HOST_TRANSFER_EXT;
    image_types(device, sharing, usage)
}

/// The memory types a 1x1 `R8G8B8A8_UNORM` image of optimal tiling and `usage` takes, as bits.
fn image_types(
    device: &ash::Device,
    sharing: Sharing<'_>,
    usage: vk::ImageUsageFlags,
) -> Result<u32, Error> {
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
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED),
    );
    // SAFETY: the device is live and `info` valid: the format is sampled and a transfer source
    // and destination with optimal tiling on every device, and a host-transfer usage is the
    // caller's to have checked.
    let raw = unsafe { crate::creation::image(device, &info) }?;
    // SAFETY: `raw` is a live image of this device, destroyed right after the query with no
    // memory bound.
    Ok(unsafe {
        let types = device.get_image_memory_requirements(raw).memory_type_bits;
        release::object(device, raw);
        types
    })
}

/// Import the dma-buf `fd`, `size` bytes, as memory of `memory`'s best type of `class` among
/// `allowed` (the types both the dma-buf and what is bound in it take): the memory and its type,
/// the caller's to free (`dma_buf.rs`'s `DmaBufPages`), in no block and never mapped. Allocated
/// with the device-address flag where the device enables `bufferDeviceAddress`, as every block of
/// the runtime's is (`BlockMaker::make_as`), and for `image` as its dedicated allocation (`size`
/// its requirements'): every import of one dma-buf, a buffer's and its textures' images', and the
/// device's own allocation of it agree, as ANV (Mesa 26.2.6) refuses an import of memory it holds
/// whose flag differs (`VK_ERROR_INVALID_EXTERNAL_HANDLE`, of every memory type:
/// `dma_buf::ExportedPages`). The import is made with `memory`'s lock released, as
/// [`import_host`]'s is: the types are chosen under it, and the import logged under it once made.
/// A driver's import of a dma-buf maps its pages for the device, which may take long (a consumer
/// measured no-copy buffers over dma-bufs on the Radeon 780M taking 120 to 273 ms to make), and
/// the device's other placements, imports and frees, all under that lock, would wait for it.
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
    memory: &Mutex<Memory>,
    device: &ash::Device,
    fd: std::os::fd::OwnedFd,
    size: u64,
    (allowed, class): (u32, MemoryClass),
    image: Option<vk::Image>,
) -> Result<(vk::DeviceMemory, u32), ImportFailure> {
    use std::os::fd::{AsRawFd, IntoRawFd};
    let (types, device_address) = {
        let memory = lock(memory);
        (
            candidates(&memory.types, allowed, class, memory.unified),
            memory.device_address,
        )
    };
    #[cfg(test)]
    let hold = lock(memory).import_hold.clone();
    let mut failure = ImportFailure::NoMemoryType;
    for type_index in types {
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
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default();
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(type_index)
            .push_next(&mut import);
        if let Some(image) = image {
            dedicated = dedicated.image(image);
            info = info.push_next(&mut dedicated);
        }
        if device_address {
            info = info.push_next(&mut flags);
        }
        // A test's stand-in for a driver slow to import, where the driver's import runs
        // (`tests::host_residency`).
        #[cfg(test)]
        if let Some(hold) = &hold {
            hold();
        }
        // SAFETY: `info` and its chain live for the call; the type index is the device's and
        // allows the dma-buf (and the image, unbound and made for dma-bufs, whose size this
        // is). The descriptor it takes is the clone, whose fate is the driver's either way.
        match unsafe { crate::creation::allocate_memory(device, &info) } {
            Ok(raw) => {
                // A successful import owns the descriptor: Vulkan closes it with the memory.
                let _ = handed.into_raw_fd();
                if let Some(log) = &lock(memory).log {
                    let what = match image {
                        Some(_) => "imported a dma-buf for an image",
                        None => "imported a dma-buf",
                    };
                    log.say(what, raw, type_index, size, None);
                }
                return Ok((raw, type_index));
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

/// Import the caller's host memory, `size` bytes at `pointer` (both multiples of the device's
/// import alignment), as a dedicated block of `memory`'s best memory type of `class` among
/// `allowed` (the types both the pointer and the resource take): its placement at offset 0,
/// whose `mapped` is `pointer`. Where `view` is given, a read-only view of those pages
/// (`read_only_view.rs`), the view is imported in their place, and unmapped once the memory
/// is freed. The block is never pooled and is freed with the placement, as a dedicated block
/// is (`host_memory.rs`). The import, which faults in and pins the pages, is made with
/// `memory`'s lock released, as a new block is ([`place`]): the device's other placements wait
/// for none of it. Refused when no type of `class` is among `allowed`, else with the failure of
/// the last type tried, the view unmapped.
///
/// # Safety
///
/// The `size` bytes at `pointer` are host memory of this process that stays mapped until the
/// placement is freed; `view`, where given, views them whole.
pub(super) unsafe fn import_host(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    (pointer, view): (NonNull<u8>, Option<ReadOnlyView>),
    size: u64,
    allowed: u32,
    class: MemoryClass,
) -> Result<Placement, ImportFailure> {
    let (types, device_address) = {
        let memory = lock(memory);
        (
            candidates(&memory.types, allowed, class, memory.unified),
            memory.device_address,
        )
    };
    let imported = view.as_ref().map_or(pointer, ReadOnlyView::pointer);
    // SAFETY: the caller's contract; the view stays mapped while the block holds it.
    let (raw, type_index) =
        unsafe { import_pointer(device, imported, size, &types, device_address) }?;
    let block = Block {
        memory: raw,
        size,
        mapped: Some(Mapped(pointer)),
        addresses: None,
        imported: true,
        pages: None,
        view,
        type_index,
        exported: None,
        fd: false,
        zeroed: false,
    };
    let host = pointer.as_ptr() as usize;
    let mut memory = lock(memory);
    memory.say_block(&format!("imported host memory 0x{host:x}"), &block);
    Ok(memory.add_import(block, type_index, size))
}

/// Import `pages`, which the runtime mapped for a buffer of its own (`fresh_pages.rs`), as
/// memory of type `type_index` in `memory`, as [`import_host`] imports the caller's: a
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
        exported: None,
        fd: false,
        zeroed: false,
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
/// memory types best first, then those it overflows into ([`overflow_candidates`], [`place`]).
pub(super) fn allocate(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    requirements: &vk::MemoryRequirements,
    class: MemoryClass,
) -> Result<Placement, Error> {
    allocate_in(memory, device, requirements, class, Holds::Buffers)
}

/// [`allocate`] in a block that holds `holds`: [`Holds::Buffers`], or [`Holds::FdBuffers`] for a
/// buffer made for [`FD_BLOCKS_HANDLE`] (`buffer.rs`'s `placed_new_buffer`).
pub(super) fn allocate_in(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    requirements: &vk::MemoryRequirements,
    class: MemoryClass,
    holds: Holds,
) -> Result<Placement, Error> {
    place(
        memory,
        device,
        requirements,
        class,
        None,
        holds,
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
    allocate_addressed_in(
        memory,
        device,
        requirements,
        (class, Holds::Buffers),
        counts,
    )
}

/// [`allocate_addressed`] in a block that holds `holds` ([`allocate_in`]).
pub(super) fn allocate_addressed_in(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    requirements: &vk::MemoryRequirements,
    (class, holds): (MemoryClass, Holds),
    counts: &Counters,
) -> Result<Placement, Error> {
    let Some(types) = lock(memory).address_types(device, holds)? else {
        return allocate_in(memory, device, requirements, class, holds);
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
        holds,
        Fit::Pooled,
    )
}

/// Memory meeting `buffer`'s `requirements` of `class`, allocated exportable as `exported` says
/// (`exported_memory.rs`). Where `exported.dedicated`, the buffer's dedicated allocation: a block
/// of its own that holds `buffer` alone, and its placement at offset 0, mapped whole where it is
/// host-visible, with no buffer of addresses (a dedicated allocation binds its buffer alone; the
/// buffer has the device-address usage of its own). The class's memory types are tried best
/// first, then those it overflows into, as [`place`] tries them, and the block is made with
/// `memory`'s lock released; refused with the failure of the last type tried. Else, as
/// [`allocate`] places a buffer, in the blocks of exportable buffers of its handle type
/// ([`Holds::Exports`], [`Holds::DmaBufs`]): pooled, or of its own where it is larger than half a
/// block, and kept for reuse as any block.
///
/// `buffer` was made with `VkExternalMemoryBufferCreateInfo` naming `exported.handle`, which the
/// device exports a buffer's memory as ([`EXPORT_HANDLE`] or [`DMA_BUF_HANDLE`]), and
/// `requirements` are its own.
pub(super) fn allocate_exported(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    requirements: &vk::MemoryRequirements,
    class: MemoryClass,
    (buffer, exported): (vk::Buffer, Exported),
) -> Result<Placement, Error> {
    if !exported.dedicated {
        let holds = match exported.handle {
            DMA_BUF_HANDLE => Holds::DmaBufs,
            _ => Holds::Exports,
        };
        let fit = Fit::Pooled;
        return place(memory, device, requirements, class, None, holds, fit);
    }
    let size = requirements.size.max(1);
    let (candidates, overflow, maker) = {
        let memory = lock(memory);
        let bits = requirements.memory_type_bits;
        (
            candidates(&memory.types, bits, class, memory.unified),
            overflow_candidates(&memory.types, bits, class, memory.unified),
            memory.maker(),
        )
    };
    let mut failure = Error::OutOfMemory(MemoryKind::Device);
    let order = (candidates.iter().map(|&type_index| (type_index, false)))
        .chain(overflow.iter().map(|&type_index| (type_index, true)));
    for (type_index, overflowing) in order {
        // As a new block of [`place`]'s, the frees posted of a heap it would fill complete first.
        let before = lock(memory).frees_before(type_index, size);
        if let Some(frees) = before {
            frees.wait();
        }
        let export = Exportable::For(buffer, exported);
        match maker.make_as(device, type_index, size, None, export, false) {
            Ok(block) => {
                let pool = (Holds::Buffers, type_index);
                let mut memory = lock(memory);
                let index = memory.add_block(pool, block, size, true, size);
                let placement = memory.placement(pool, index, 0, size, false);
                memory.say_placement("placed", &placement, Some(class));
                return Ok(memory.mark_overflowed(placement, overflowing));
            }
            Err(error @ Error::OutOfMemory(_)) => failure = error,
            Err(error) => return Err(error),
        }
    }
    Err(failure)
}

/// [`allocate`], in a block with a buffer of addresses where `addressed` gives the counts of the
/// buffers of addresses made, of the blocks that hold `holds`, pooled or in a block of its own
/// as `fit` says. A new block is made with `memory`'s lock released: the device's other threads'
/// placements, which the lock orders, wait for no `vkAllocateMemory`, which on NVIDIA's driver
/// waited 2.37 s for other processes' GPU work.
///
/// Where the class's own types have no room, the resource goes to those it overflows into
/// ([`overflow_candidates`]), each placement trying its own types first again, with no record of
/// a full heap: a device-local allocation that fails for want of room is cheap. On an RTX 5080
/// Laptop GPU (Linux) past its full 17.09 GB heap, eight of 48 MiB each answered
/// `VK_ERROR_OUT_OF_DEVICE_MEMORY` within 4 µs (median under 1 µs), where a 48 MiB `Private`
/// buffer's whole make took 6.9 ms there (median of 8, longest 7.2 ms; 0.47 ms in device-local
/// memory) and a 4096 x 4096 `RGBA8Unorm` texture's 9.2 ms (longest 9.5 ms; 0.65 ms).
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
    let (candidates, overflow, maker) = {
        let memory = lock(memory);
        let bits = requirements.memory_type_bits;
        (
            candidates(&memory.types, bits, class, memory.unified),
            overflow_candidates(&memory.types, bits, class, memory.unified),
            memory.maker(),
        )
    };
    if candidates.is_empty() && overflow.is_empty() {
        return Err(Error::OutOfMemory(MemoryKind::Device));
    }
    let dedicated = fit == Fit::Apart || size > maker.block_bytes / 2;
    let block_size = if dedicated { size } else { maker.block_bytes };
    let mut failure = Error::OutOfMemory(MemoryKind::Device);
    // The class's own types first, then those it overflows into where none of them has room.
    let order = (candidates.iter().map(|&type_index| (type_index, false)))
        .chain(overflow.iter().map(|&type_index| (type_index, true)));
    for (type_index, overflowing) in order {
        let pool = (holds, type_index);
        if fit == Fit::Pooled {
            let mut guard = lock(memory);
            if let Some(placement) =
                guard.place_pooled(device, pool, (size, align), addressed, class)?
            {
                return Ok(guard.mark_overflowed(placement, overflowing));
            }
            // A dedicated placement takes a kept dedicated block its size fits.
            if dedicated {
                if let Some(placement) =
                    guard.place_kept_dedicated(device, pool, size, addressed, class)?
                {
                    return Ok(guard.mark_overflowed(placement, overflowing));
                }
            }
            // A pooled placement without room: the block made ahead for the pool, rather than
            // one of the placement's own (`memory/ready.rs`), waited for where its making is
            // under way, but in the pool host copies place in, whose making may itself wait
            // behind the program's GPU work, which no making waits for.
            if !dedicated && take_ready(memory, guard, pool) {
                let mut guard = lock(memory);
                if let Some(placement) =
                    guard.place_pooled(device, pool, (size, align), addressed, class)?
                {
                    return Ok(guard.mark_overflowed(placement, overflowing));
                }
            }
        }
        // Frees posted still count against their heap: where the block would take its heap past
        // the heap's size, those of the heap complete first (`free_thread.rs`).
        let before = lock(memory).frees_before(type_index, block_size);
        if let Some(frees) = before {
            frees.wait();
        }
        // The type's kept empty blocks go, and the frees posted complete, before a block of it
        // fails to be made.
        let make = || maker.make(device, pool, block_size, addressed);
        let made = match make() {
            Err(Error::OutOfMemory(_)) if make_room(memory, device, type_index) => make(),
            made => made,
        };
        match made {
            Ok(block) => {
                let mut memory = lock(memory);
                if !dedicated && memory.makes_ahead(pool) {
                    memory.ahead.made_in_place(holds);
                }
                if !dedicated && memory.host_copy_pool == Some(Some(pool)) {
                    memory.host_copy_blocks.made_in_place += 1;
                }
                let fresh = block.zeroed;
                let index = memory.add_block(pool, block, block_size, dedicated, size);
                let mut placement = memory.placement(pool, index, 0, size, addressed.is_some());
                placement.fresh = fresh;
                memory.say_placement("placed", &placement, Some(class));
                return Ok(memory.mark_overflowed(placement, overflowing));
            }
            Err(error @ Error::OutOfMemory(_)) => failure = error,
            Err(error) => return Err(error),
        }
    }
    Err(failure)
}

/// Room for a block of memory type `type_index` that failed to be made for room ([`place`]): the
/// type's kept empty blocks go, and every free posted completes (`free_thread.rs`), waited for
/// with `memory`'s lock released. Whether any went.
fn make_room(memory: &Mutex<Memory>, device: &ash::Device, type_index: u32) -> bool {
    let (released, frees) = {
        let mut memory = lock(memory);
        (
            memory.release_kept(device, type_index),
            memory.posted_frees(),
        )
    };
    let waited = frees.is_some_and(|frees| frees.wait());
    released > 0 || waited
}

/// Keep room for any pooled placement (half a block) in each pool the runtime's encodings and
/// commits place their own resources in ([`Memory::kept`]), and start making a block ahead for
/// each such pool that grows (`memory/ready.rs`). The device's creation calls, which may wait,
/// keep it, so that an encoding or a commit, which must wait for no other process's GPU work,
/// finds a block with room rather than making one. A pool without that room takes its block made
/// ahead, waited for where its making is under way, or else a block made here, with `memory`'s
/// lock released; a block of buffers gets its buffer of addresses (counted in `counts`) where
/// blocks carry one. A block that fails to be made, or a pool the query fails for, is left to the
/// placement that needs it. A pool that grew within [`KEPT_EMPTY_FOR`] and lacks room for a whole
/// block, holding no block made ahead, gets one started, through `functions`, made once a burst
/// of `quiet`'s queue uses ended. The pool host copies place in, where `copies` says the device
/// copies color textures on the host, gets its block made ahead once it lacks room for half a
/// block; its making is never waited for here, as its pages' reads may wait behind the program's
/// GPU work, and its block is never added here: the placement there that finds no room adds it
/// ([`place`]), so that the pool's blocks change only at its placements, whenever the making
/// ended. The kept empty blocks past their quiet period go here too, once a second at most
/// ([`Memory::sweep_stale`]).
pub(super) fn keep_room(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    counts: &Arc<Counters>,
    (functions, quiet): (&Arc<Functions>, &Arc<QueueQuiet>),
    copies: impl FnOnce() -> bool,
) {
    ready_ahead(memory, device, (functions, quiet), copies);
    let (kept, maker) = {
        let mut memory = lock(memory);
        memory.sweep_stale(device, Instant::now());
        let Ok(kept) = memory.kept(device) else {
            return;
        };
        (kept, memory.maker())
    };
    let room = maker.block_bytes / 2;
    for pool in kept {
        let addressed = pool.0.buffers().then_some(&**counts);
        loop {
            let mut guard = lock(memory);
            if guard.pool(pool.0, pool.1).has_room(room) {
                break;
            }
            // The host copies' block made ahead joins its pool at the placement that needs it
            // (`place`), never here, whenever its making ended: its pool's blocks change only at
            // its placements.
            let host_copies = guard.host_copy_pool == Some(Some(pool));
            if host_copies && guard.readying_at(pool).is_some() {
                break;
            }
            if guard.add_ready(pool) {
                break;
            }
            if guard.readying_at(pool).is_some() {
                // Another call's making to wait for.
                if take_ready(memory, guard, pool) {
                    break;
                }
                continue;
            }
            drop(guard);
            let Ok(block) = maker.make(device, pool, maker.block_bytes, addressed) else {
                break;
            };
            let mut guard = lock(memory);
            guard.add_block(pool, block, maker.block_bytes, false, 0);
            guard.ahead.made_in_place(pool.0);
            break;
        }
        let mut guard = lock(memory);
        let pool_of = guard.pool(pool.0, pool.1);
        let grew = (pool_of.grew(None)).is_some_and(|at| at.elapsed() < KEPT_EMPTY_FOR);
        if !grew || pool_of.has_room(maker.block_bytes) || guard.readying_at(pool).is_some() {
            continue;
        }
        let addressed = pool.0.buffers().then(|| Arc::clone(counts));
        let started = ready::Readying::start(
            guard.maker(),
            Arc::clone(functions),
            Arc::clone(quiet),
            addressed,
            (pool, maker.block_bytes),
        );
        guard.readying.extend(started);
    }
}

/// Start making a block ahead for the pool host copies place in (`memory/ready.rs`,
/// [`Memory::host_copy_pool`], where `copies`, asked with `memory`'s lock released until the pool
/// was found, says the device copies color textures on the host), through `functions`, made once
/// a burst of `quiet`'s queue uses ended, where it lacks room for half a block and holds no block
/// made ahead.
fn ready_ahead(
    memory: &Mutex<Memory>,
    device: &ash::Device,
    (functions, quiet): (&Arc<Functions>, &Arc<QueueQuiet>),
    copies: impl FnOnce() -> bool,
) {
    let copies = lock(memory).host_copy_pool.is_none() && copies();
    let mut memory = lock(memory);
    let Some(pool) = memory.host_copy_pool(device, copies) else {
        return;
    };
    let room = memory.block_bytes / 2;
    if memory.readying_at(pool).is_some() || memory.pool(pool.0, pool.1).has_room(room) {
        return;
    }
    let maker = memory.maker();
    let bytes = maker.block_bytes;
    let functions = Arc::clone(functions);
    let started = ready::Readying::start(maker, functions, Arc::clone(quiet), None, (pool, bytes));
    memory.readying.extend(started);
}

/// Add the block made ahead for `pool` to it, `guard` held on `memory` (`memory/ready.rs`): at
/// once where its making ended; where it is under way, once it ends, waited for with the lock
/// released, but in the pool host copies place in. Whether one may have been added, so that the
/// caller looks for room again; the lock is released as it returns.
fn take_ready(
    memory: &Mutex<Memory>,
    mut guard: MutexGuard<'_, Memory>,
    pool: (Holds, u32),
) -> bool {
    if guard.add_ready(pool) {
        return true;
    }
    let Some(at) = guard.readying_at(pool) else {
        return false;
    };
    if guard.host_copy_pool == Some(Some(pool)) {
        return false;
    }
    let waiter = guard.readying[at].waiter();
    drop(guard);
    waiter.wait();
    lock(memory).add_ready(pool);
    true
}
