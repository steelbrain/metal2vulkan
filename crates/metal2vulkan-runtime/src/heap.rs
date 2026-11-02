//! `Heap` (`MTLHeap`) and `Device::new_heap`, with the buffers placed in heaps (step 7's
//! heaps-with-buffers part). A heap's textures are `texture/heap.rs`'s, images bound into the
//! heap's memory; a sparse heap's are `sparse/`'s: a sparse heap holds no buffer and no memory of
//! its own, only Metal's count of its tiles.
//!
//! - **One buffer.** A heap is one buffer of the runtime's own ([`Device::zeroed_buffer`]): one
//!   `VkBuffer` over one placement in the sub-allocator, and one range of the device's Metal
//!   address space, the heap's size long. A buffer placed in the heap is that buffer's bytes from
//!   its offset (`Bytes::Heap` in buffer.rs): the same `VkBuffer`, memory and addresses. So
//!   overlapping buffers alias byte for byte, a buffer's Metal address is the heap's plus its
//!   offset, and its bytes past its end are the heap's next ones, as on Metal. The bind layer
//!   still states each buffer at its own length (`Buffer::space`), so a contained access stays
//!   in the buffer's own bytes.
//! - **Metal's rules.** Measured on an M3 Max (macOS 26.7) and an M4 (macOS 27.0.1), the same on
//!   both, plain and under API validation; what Metal ends the process for even without validation
//!   is refused `Invalid` (a storage or CPU cache mode other than the heap's, a `Managed` or
//!   `Memoryless` heap, an alignment that is not a power of two), and what plain Metal answers nil
//!   for, its validation stopping it, `Nil`:
//!   - a heap's size rounds up to 16 KiB, and a size of 0 answers nil;
//!   - a placement heap places a buffer at an offset that is a multiple of 256 bytes, its bytes
//!     inside the heap (ending at the heap's end is fine; another offset, or bytes past the end,
//!     answer nil), and never clears them: a new heap's
//!     bytes read zero, and a buffer over bytes another wrote reads them;
//!   - an automatic heap places a buffer at the lowest offset with room, at a multiple of 256
//!     bytes (first fit), its used size the sum of its buffers' lengths, and `make_aliasable`
//!     gives a buffer's range to the next buffer; a full heap answers nil without validation
//!     stopping it ([`MemoryKind::Heap`]);
//!   - a buffer takes its heap's storage, CPU cache and hazard tracking modes: a tracking mode
//!     named other than the heap's becomes the heap's, as plain Metal makes it where its
//!     validation stops it, the default takes the heap's, and a heap's own default is untracked;
//!   - a placement heap's buffer answers aliasable from the start; an automatic heap's once
//!     `make_aliasable` gave its range back;
//!   - Metal tracks a tracked heap's resources as one (Apple's "tracking is done at the
//!     granularity of the heap"): a heap buffer's hazard identity is its heap's, over the heap's
//!     bytes (`Buffer::tracked`), so a write through one buffer orders a read through another
//!     over the same bytes;
//!   - a heap's purgeable state answers as a buffer's does; emptying a `Shared` heap zeroes its
//!     bytes until a committed command used the heap, and keeps them from then on, and emptying
//!     a `Private` one keeps them, as Metal did; a heap's buffer answers `NonVolatile` to every
//!     purgeable state and changes nothing, as plain Metal answers where its validation stops each.
//! - **The device's values.** Placement offsets and `heap_buffer_size_and_align` take Metal's
//!   256-byte alignment on every device: the device's own offset alignments divide it (Vulkan
//!   caps each at 256, as powers of two), and a smaller one would let a program pack its buffers
//!   tighter than on Metal and change what a read past a buffer's end reaches. The 16-KiB size
//!   granule is the address space's page, Metal's too.
//! - **The largest heap.** One allocation of the device, at most `maxMemoryAllocationSize` (a
//!   larger heap answers out of device memory, as Metal answers nil for a heap too large), and one
//!   `VkBuffer`, at most `maxBufferSize`: a heap past that is refused
//!   ([`RuntimeGap::HeapPastMaxBufferSize`]). Only ANV among the owned venues has a
//!   `maxBufferSize` (4 GiB) below its `maxMemoryAllocationSize`.10 records the exact emulation,
//!   windows of `VkBuffer`s over the one allocation.

use super::address::{AddressPlacement, GpuAddress, RangeToken, PAGE};
use super::attributes::PurgeableState;
use super::attributes::ResourceAttributes;
use super::buffer::{buffer_class, Buffer, Modes, WeakBuffer};
use super::device::lock;
use super::error::{invalid, nil, Error, MemoryKind, Problem, RuntimeGap};
use super::host_memory::{Deallocator, HostMemory, HostRelease, KeepAlive};
use super::key_hash::KeyMap;
use super::range::RangeAllocator;
use super::resource::{ObjectId, ResourceKey};
use super::sparse::heap::SparseHeap;
use super::sparse::texture::sparse_refusal;
use super::sparse::SparsePageSize;
use super::texture::{Texture, TextureShared};
use super::values::{CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode};
use super::Device;
use metal2vulkan::device::AllocationLimits;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// The alignment of every buffer a heap places, and the one `heap_buffer_size_and_align`
/// answers: Metal's, on every device (see the module).
pub(super) const PLACEMENT_ALIGNMENT: u64 = 256;

/// `MTLHeapType`, with Metal's raw values.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HeapType {
    /// `MTLHeapTypeAutomatic`: the heap places each resource.
    #[default]
    Automatic,
    /// `MTLHeapTypePlacement`: the program places each resource at an offset.
    Placement,
    /// `MTLHeapTypeSparse`: sparse textures' tiles.
    Sparse,
}

impl HeapType {
    /// Metal's raw value.
    pub fn raw(self) -> u64 {
        match self {
            HeapType::Automatic => 0,
            HeapType::Placement => 1,
            HeapType::Sparse => 2,
        }
    }

    /// The type of Metal's raw value `raw`.
    pub fn from_raw(raw: u64) -> Result<HeapType, Error> {
        Ok(match raw {
            0 => HeapType::Automatic,
            1 => HeapType::Placement,
            2 => HeapType::Sparse,
            raw => {
                return Err(invalid(
                    "from_raw",
                    Problem::UndefinedValue {
                        ty: "MTLHeapType",
                        raw,
                    },
                ))
            }
        })
    }
}

/// `MTLHeapDescriptor`. Its default is Metal's (measured): an automatic heap of size 0, in
/// `Private` storage with the default CPU cache and hazard tracking modes, at an address the
/// runtime picks.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HeapDescriptor {
    /// `type`.
    pub heap_type: HeapType,
    /// `size`, in bytes.
    pub size: u64,
    /// `resourceOptions`: the storage mode, CPU cache mode and hazard tracking mode.
    pub resource_options: ResourceOptions,
    /// `pinnedGPUAddress` (Apple's private property, a runtime extension): where the heap goes in
    /// the Metal address space (`address.rs`).
    pub placement: AddressPlacement,
    /// `sparsePageSize`: a sparse heap's tile (16 KiB by default); other heaps ignore it.
    pub sparse_page_size: SparsePageSize,
}

impl HeapDescriptor {
    /// A heap of `heap_type`, `size` bytes, and `resource_options`.
    pub fn new(
        heap_type: HeapType,
        size: u64,
        resource_options: ResourceOptions,
    ) -> HeapDescriptor {
        HeapDescriptor {
            heap_type,
            size,
            resource_options,
            placement: AddressPlacement::Any,
            sparse_page_size: SparsePageSize::default(),
        }
    }

    /// This descriptor at `placement` in the address space.
    pub fn with_placement(self, placement: AddressPlacement) -> HeapDescriptor {
        HeapDescriptor { placement, ..self }
    }

    /// This descriptor with sparse tiles of `page`.
    pub fn with_sparse_page_size(self, sparse_page_size: SparsePageSize) -> HeapDescriptor {
        HeapDescriptor {
            sparse_page_size,
            ..self
        }
    }
}

impl Default for HeapDescriptor {
    fn default() -> HeapDescriptor {
        HeapDescriptor::new(HeapType::Automatic, 0, ResourceOptions::private())
    }
}

/// `MTLSizeAndAlign`: the bytes a resource takes of a heap, and the alignment of its offset.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SizeAndAlign {
    /// `size`, in bytes.
    pub size: u64,
    /// `align`, in bytes.
    pub align: u64,
}

/// What a heap call did wrong (`Problem::Heap`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeapProblem {
    /// A heap of size 0: plain Metal answers nil (`Nil`) for every type and storage mode.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal answers nil for a heap of
    /// size 0).
    Size,
    /// The call places resources in heaps of the other type: `new_buffer` in an automatic heap,
    /// `new_buffer_at_offset` in a placement heap.
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal answers nil for
    /// `newBufferWithLength:options:` on a placement heap, where an automatic heap's is made).
    HeapType {
        /// The heap's type.
        heap_type: HeapType,
    },
    /// A placement offset that is not a multiple of the resource's alignment, 256 bytes for a
    /// buffer and `Device::heap_texture_size_and_align`'s for a texture: plain Metal answers nil
    /// (`Nil`).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: a placement heap of 1 MiB answers nil
    /// for a buffer at offset 255 and for a 64 by 64 texture at offset 129, where offset 256 and
    /// the texture's own alignment of 128 make them).
    Offset {
        /// The offset given.
        offset: u64,
    },
    /// A resource whose `length` bytes from `offset` run past its heap's end, of `heap_size`
    /// bytes: a placement heap's buffer or texture, or an automatic heap's buffer longer than
    /// the heap. Plain Metal answers nil (`Nil`).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: a placement heap of 1 MiB answers nil
    /// for a 4096-byte buffer at offset 1,047,552 and for a texture at offset 1,048,448, and an
    /// automatic heap of 1 MiB for a buffer of 4 MiB, where each resource inside the heap is made).
    PastEnd {
        /// The resource's offset in the heap (0 for an automatic heap's).
        offset: u64,
        /// The resource's bytes.
        length: u64,
        /// The heap's size.
        heap_size: u64,
    },
    /// A resource's storage mode other than its heap's (Metal ends the process).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "The requested
    /// storage mode (MTLStorageModeShared) is not compatible with the heap's mode
    /// (MTLStorageModePrivate)").
    StorageMode {
        /// The heap's.
        heap: StorageMode,
        /// The resource's.
        resource: StorageMode,
    },
    /// A resource's CPU cache mode other than its heap's (Metal ends the process).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts "The requested CPU
    /// cache mode (MTLCPUCacheModeWriteCombined) does not match the heap's mode
    /// (MTLCPUCacheModeDefaultCache)").
    CpuCacheMode {
        /// The heap's.
        heap: CpuCacheMode,
        /// The resource's.
        resource: CpuCacheMode,
    },
    /// An alignment other than 0 that is not a power of two (`max_available_size`; Metal ends
    /// the process).
    ///
    /// Basis: metal-measured (an Apple M3 Max, macOS 27.0.1: plain Metal asserts `!(alignment &
    /// (alignment - 1))` in `-[IOGPUMetalHeap maxAvailableSizeWithAlignment:]`, where 256 answers).
    Alignment {
        /// The alignment given.
        alignment: u64,
    },
    /// A no-copy heap's memory that does not start at a multiple of `alignment`, the larger of
    /// the host page and the device's `minImportedHostPointerAlignment`
    /// (`Device::new_heap_with_bytes_no_copy`): its offsets would not be its memory's.
    ///
    /// Basis: ours (a runtime extension: Metal makes no heap over the caller's memory, so where its
    /// offsets begin is this crate's own contract).
    HostMemoryAlignment {
        /// The alignment the memory's first byte needs.
        alignment: u64,
    },
}

/// A Metal heap (`MTLHeap`). Cheap to clone (a retain); `Send + Sync`. A buffer placed in it holds it. Two
/// handles are equal when they are the same heap.
#[derive(Clone)]
pub struct Heap(Arc<HeapShared>);

/// A heap held without keeping it (`used_resources.rs`).
#[derive(Clone)]
pub(super) struct WeakHeap(Weak<HeapShared>);

impl WeakHeap {
    /// The heap, while it lives.
    pub(super) fn upgrade(&self) -> Option<Heap> {
        self.0.upgrade().map(Heap)
    }

    /// Whether the heap lives.
    pub(super) fn lives(&self) -> bool {
        self.0.strong_count() > 0
    }
}

impl Heap {
    /// A handle that does not keep the heap (`WeakHeap`).
    pub(super) fn downgrade(&self) -> WeakHeap {
        WeakHeap(Arc::downgrade(&self.0))
    }
}

struct HeapShared {
    storage: HeapStorage,
    heap_type: HeapType,
    /// An automatic heap's free ranges of `[0, size)`; a placement heap's and a sparse heap's
    /// stay whole.
    ranges: Mutex<RangeAllocator>,
    /// The textures made in the heap, while they live, which `use_heap` makes resident, and an
    /// encoder that used the heap makes resident as it encodes each later command
    /// (`residency.rs`).
    textures: Mutex<Numbered<TextureShared>>,
    /// A committed command buffer holds a command that used the heap: one of its resources, or
    /// the heap or a resource of it made resident by the command's encoder. Metal's `Empty`
    /// keeps a `Shared` heap's bytes from then on.
    used: AtomicBool,
    /// Its identity (`object_id`).
    object_id: ObjectId,
    /// The buffers placed in it, and whether a command buffer's commands used it, from which on
    /// a command reaches them unnamed (`live_buffers.rs`).
    placed: Mutex<Placed>,
}

/// The buffers placed in a heap, held weakly, and whether a command buffer's commands used the
/// heap (`Heap::reached`).
#[derive(Default)]
struct Placed {
    buffers: Vec<WeakBuffer>,
    reached: bool,
}

impl Drop for HeapShared {
    fn drop(&mut self) {
        if let HeapStorage::Bytes(bytes) = &self.storage {
            bytes
                .device()
                .shared()
                .heap_keys
                .forget(bytes.identity().key);
        }
    }
}

/// A device's heaps with bytes of their own, by the key their resources' uses name: a heap
/// buffer's and a heap texture's hazard identity is the heap's buffer's (`Buffer::tracked`,
/// `texture/heap.rs`'s `HeapBinding`), so an encoder finds the heaps its commands used from
/// their uses' keys (`EncoderCore::end_with_replay`).
#[derive(Default)]
pub(super) struct HeapKeys {
    /// Each heap held weakly, with its identity.
    heaps: Mutex<KeyMap<ResourceKey, (Weak<HeapShared>, ObjectId)>>,
}

impl HeapKeys {
    fn register(&self, key: ResourceKey, heap: Weak<HeapShared>, id: ObjectId) {
        lock(&self.heaps).insert(key, (heap, id));
    }

    fn forget(&self, key: ResourceKey) {
        lock(&self.heaps).remove(&key);
    }

    /// Each heap `keys` name, held weakly, to `found`: no heap is let go under this lock.
    pub(super) fn each_among<'a>(
        &self,
        keys: impl IntoIterator<Item = &'a ResourceKey>,
        mut found: impl FnMut(WeakHeap),
    ) {
        let heaps = lock(&self.heaps);
        for (heap, _) in keys.into_iter().filter_map(|key| heaps.get(key)) {
            found(WeakHeap(heap.clone()));
        }
    }

    /// The live heaps `keys` name.
    pub(super) fn among(&self, keys: impl IntoIterator<Item = ResourceKey>) -> Vec<Heap> {
        let heaps = lock(&self.heaps);
        // A device without a live heap's key: none of `keys` names one.
        if heaps.is_empty() {
            return Vec::new();
        }
        keys.into_iter()
            .filter_map(|key| heaps.get(&key)?.0.upgrade().map(Heap))
            .collect()
    }

    /// Whether one of `keys` names a live heap among `objects`, by its identity: no heap is held,
    /// so none is let go under this lock.
    pub(super) fn names_any<'a>(
        &self,
        keys: impl IntoIterator<Item = &'a ResourceKey>,
        objects: &[ObjectId],
    ) -> bool {
        let heaps = lock(&self.heaps);
        keys.into_iter()
            .filter_map(|key| heaps.get(key))
            .any(|(heap, id)| objects.contains(id) && heap.strong_count() > 0)
    }
}

/// What a heap holds of the textures it made (`HeapShared::textures`): each that may still
/// live, weakly, numbered in the order the heap made them, and how many it numbered. Generic, so
/// its numbering is tested without a device.
#[derive(Debug)]
pub(super) struct Numbered<T> {
    made: u64,
    held: Vec<(u64, Weak<T>)>,
}

impl<T> Default for Numbered<T> {
    fn default() -> Numbered<T> {
        Numbered {
            made: 0,
            held: Vec::new(),
        }
    }
}

impl<T> Numbered<T> {
    /// Hold `item`, numbered after every one held before; those gone are forgotten.
    pub(super) fn hold(&mut self, item: Weak<T>) {
        self.held.retain(|(_, held)| held.strong_count() > 0);
        self.held.push((self.made, item));
        self.made += 1;
    }

    /// How many it numbered: the number the next one takes.
    pub(super) fn made(&self) -> u64 {
        self.made
    }

    /// Those numbered `from` or later that may still live, in the order they were held.
    pub(super) fn since(&self, from: u64) -> impl Iterator<Item = &Weak<T>> {
        self.held
            .iter()
            .filter(move |(number, _)| *number >= from)
            .map(|(_, held)| held)
    }
}

/// What a heap holds.
enum HeapStorage {
    /// An automatic or placement heap's own buffer: its whole size, memory and addresses, its
    /// label and purgeable state the heap's.
    Bytes(Buffer),
    /// A sparse heap's count of its tiles (`sparse/heap.rs`).
    Sparse(SparseHeap),
}

impl std::fmt::Debug for Heap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Heap")
            .field("heap_type", &self.0.heap_type)
            .field("size", &self.size())
            .field("gpu_address", &self.gpu_address())
            .field("options", &self.resource_options())
            .finish()
    }
}

impl PartialEq for Heap {
    fn eq(&self, other: &Heap) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Heap {}

impl std::hash::Hash for Heap {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Arc::as_ptr(&self.0), state);
    }
}

impl Device {
    /// `newHeapWithDescriptor:`: a heap of the descriptor's size rounded up to 16 KiB, all zero.
    ///
    /// A sparse heap holds no memory of its own: its size rounds up to its `sparse_page_size`, its
    /// storage is `Private` alone, its default tracking mode is untracked, and it has no range of
    /// the address space.
    ///
    /// Refused: a size of 0 (`Nil`, as plain Metal answers nil); `Managed` or `Memoryless`
    /// storage (and for a sparse heap any but `Private`, which Metal stops even without
    /// validation), an undefined option (`Invalid`); a
    /// sparse heap on a device that lacks sparse binding or residency of any image type where its
    /// functions read residency from the image (`Unsupported(Feature::SparseTextures)`, naming
    /// it; a device without `shaderResourceResidency` makes its sparse textures ordinary images
    /// instead); a size past the device's `maxMemoryAllocationSize` (`OutOfMemory`) or
    /// `maxBufferSize` (`NotImplemented(HeapPastMaxBufferSize)`); a pinned address whose range
    /// meets another resource's bytes or leaves the space (`PlacementConflict`).
    pub fn new_heap(&self, descriptor: &HeapDescriptor) -> Result<Heap, Error> {
        self.heap_with_token(descriptor, None)
    }

    /// [`Device::new_heap`] with `token` (a runtime extension), dropped once the heap's range of
    /// the address space is free ([`RangeToken`]): after the heap and every resource placed in
    /// it are gone, as its own bytes go. A refused call's drops before it returns (after any
    /// range it took is free again).
    ///
    /// A sparse heap's drops once the runtime frees the heap: after the heap, every texture made
    /// from it, and the command buffers that hold any of them are gone.
    pub fn new_heap_with_range_token(
        &self,
        descriptor: &HeapDescriptor,
        token: RangeToken,
    ) -> Result<Heap, Error> {
        self.heap_with_token(descriptor, Some(token))
    }

    /// [`Device::new_heap`], the heap's own bytes keeping `token` where given.
    fn heap_with_token(
        &self,
        descriptor: &HeapDescriptor,
        token: Option<RangeToken>,
    ) -> Result<Heap, Error> {
        const CALL: &str = "new_heap";
        self.keep_room();
        let descriptor = &HeapDescriptor {
            resource_options: self.take_options(CALL, descriptor.resource_options),
            ..*descriptor
        };
        let modes = Modes::of(descriptor.resource_options)?;
        let class = match modes.storage {
            StorageMode::Shared | StorageMode::Private => buffer_class(CALL, modes)?,
            StorageMode::Managed | StorageMode::Memoryless => {
                return Err(invalid(CALL, Problem::StorageMode))
            }
        };
        if descriptor.heap_type == HeapType::Sparse {
            return self.new_sparse_heap(descriptor, modes, token);
        }
        if descriptor.size == 0 {
            return Err(nil(CALL, Problem::Heap(HeapProblem::Size)));
        }
        let size = heap_size(descriptor.size, self.compile_device().allocation())?;
        let bytes = self.zeroed_buffer(CALL, size, modes, class, true, descriptor.placement)?;
        if let Some(token) = token {
            bytes.keep_range_token(token)?;
        }
        self.degrade_system_memory(&bytes, class, || format!("a `Shared` heap of {size} bytes"));
        Ok(self.heap_over(bytes, descriptor.heap_type, size))
    }

    /// A heap over the caller's host memory (a runtime extension, the heap twin of
    /// [`Device::new_buffer_with_bytes_no_copy`]), its pointer wrapped in `memory`
    /// ([`HostMemory`]): an automatic or placement heap of the descriptor's size rounded up to 16
    /// KiB whose bytes are the caller's, from `memory.pointer()`. Its buffers and textures alias at
    /// their offsets as in any heap of its type, and the CPU's writes through the pointer are what
    /// its resources' commands read. `deallocator` runs once, with the pointer and the heap's size,
    /// when the heap's last reference goes (its resources hold it): on a runtime thread when that
    /// is a command buffer completing. A refused call never runs it. `keep_alive` ([`KeepAlive`])
    /// is dropped once the device's import of the memory is freed: with one, `deallocator` runs at
    /// the heap's release, though a command buffer submitted before a texture placed in the heap
    /// was released may hold the import until it completes; with none, `deallocator` waits for the
    /// import (`host-deallocator-delayed`).
    ///
    /// The memory is imported here, as a no-copy buffer's is (`host_memory.rs`): by its pointer, or
    /// where the device refuses that and `memory` carries a dma-buf source
    /// ([`HostMemory::with_dma_buf`]), by the dma-bufs it makes of the pages. A buffer placed in
    /// the heap is its bytes; a texture is an image bound to them where the device binds such an
    /// image to imported host memory (its optimal image, else a linear one; over dma-bufs, an
    /// image over a dma-buf of its own pages), else an image of its own whose texels the encoders copy from
    /// the heap's bytes and back, laid out by level and layer from the texture's offset, which the
    /// device announces once (`shadowed-texture`; `texture/heap.rs`). Its purgeable state answers
    /// as a no-copy buffer's: non-volatile, and emptying it keeps the caller's bytes. With Metal's
    /// private option bit 24 the GPU's writes to the memory are dropped as a no-copy buffer's are
    /// ([`Device::new_buffer_with_bytes_no_copy`]).
    ///
    /// Where the caller's gate refuses the import
    /// ([`HostMemory::with_import_gate`](crate::HostMemory::with_import_gate)), or the import the
    /// device and the gate admitted fails (out of memory, another Vulkan error), the heap's own
    /// buffer is a copy of the pages in memory of the device's own instead, its buffers lying in
    /// the copy and its textures shadows of its bytes, announced (`copied-host-pages`,
    /// `copied.rs`).
    ///
    /// Refused: a size of 0 (`Nil`, as [`Device::new_heap`] answers it); storage other than
    /// `Shared`, a sparse heap, an undefined option (`Invalid`); memory whose first byte is not a multiple of the import's unit
    /// (`Invalid`, [`HeapProblem::HostMemoryAlignment`]); and as
    /// [`Device::new_buffer_with_bytes_no_copy`] refuses the memory: a device without
    /// `VK_EXT_external_memory_host` (`Unsupported`, naming it), memory it cannot import exactly
    /// (`Unsupported`, `HostMemory`), a size past the device's allocation (`OutOfMemory`) or
    /// `maxBufferSize` (`NotImplemented(HeapPastMaxBufferSize)`); a pinned address the space
    /// cannot honour (`PlacementConflict`).
    ///
    /// # Safety
    ///
    /// `memory` meets the contract of the [`HostMemory`] constructor that made it
    /// ([`HostMemory::new`], [`HostMemory::with_dma_buf`]) for the heap's [`size`](Heap::size).
    pub unsafe fn new_heap_with_bytes_no_copy(
        &self,
        memory: HostMemory,
        descriptor: &HeapDescriptor,
        deallocator: Option<Deallocator>,
        keep_alive: Option<KeepAlive>,
    ) -> Result<Heap, Error> {
        let release = HostRelease::new(deallocator, keep_alive);
        // SAFETY: the caller's contract.
        unsafe { self.no_copy_heap_with_token(memory, descriptor, release, None) }
    }

    /// [`Device::new_heap_with_bytes_no_copy`] with `token` (a runtime extension), dropped once
    /// the heap's range of the address space is free ([`RangeToken`]): after the heap and every
    /// resource placed in it are gone, as its own bytes go, before their deallocator runs.
    ///
    /// # Safety
    ///
    /// As [`Device::new_heap_with_bytes_no_copy`]'s.
    pub unsafe fn new_heap_with_bytes_no_copy_and_range_token(
        &self,
        memory: HostMemory,
        descriptor: &HeapDescriptor,
        deallocator: Option<Deallocator>,
        keep_alive: Option<KeepAlive>,
        token: RangeToken,
    ) -> Result<Heap, Error> {
        let release = HostRelease::new(deallocator, keep_alive);
        // SAFETY: the caller's contract.
        unsafe { self.no_copy_heap_with_token(memory, descriptor, release, Some(token)) }
    }

    /// [`Device::new_heap_with_bytes_no_copy`], given back to `release`, the heap's own bytes
    /// keeping `token` where given.
    ///
    /// # Safety
    ///
    /// As [`Device::new_heap_with_bytes_no_copy`]'s.
    unsafe fn no_copy_heap_with_token(
        &self,
        memory: HostMemory,
        descriptor: &HeapDescriptor,
        release: HostRelease,
        token: Option<RangeToken>,
    ) -> Result<Heap, Error> {
        const CALL: &str = "new_heap_with_bytes_no_copy";
        let (options, drops_writes) = self.no_copy_options(CALL, descriptor.resource_options);
        let modes = Modes::of(options)?;
        if modes.storage != StorageMode::Shared || descriptor.heap_type == HeapType::Sparse {
            return Err(invalid(CALL, Problem::StorageMode));
        }
        if descriptor.size == 0 {
            return Err(nil(CALL, Problem::Heap(HeapProblem::Size)));
        }
        let size = heap_size(descriptor.size, self.compile_device().allocation())?;
        // SAFETY: the caller's contract, for the heap's size.
        let bytes = unsafe {
            self.buffer_over_host_memory(
                CALL,
                memory,
                size,
                options,
                release,
                descriptor.placement.into(),
                true,
                drops_writes,
            )
        }?;
        if let Some(token) = token {
            bytes.keep_range_token(token)?;
        }
        Ok(self.heap_over(bytes, descriptor.heap_type, size))
    }

    /// The automatic or placement heap of `heap_type` and `size` bytes whose own buffer is
    /// `bytes`, registered by its buffer's key.
    fn heap_over(&self, bytes: Buffer, heap_type: HeapType, size: u64) -> Heap {
        let key = bytes.identity().key;
        let heap = Arc::new(HeapShared {
            storage: HeapStorage::Bytes(bytes),
            heap_type,
            ranges: Mutex::new(RangeAllocator::new(0, size)),
            textures: Mutex::new(Numbered::default()),
            used: AtomicBool::new(false),
            object_id: ObjectId::new(),
            placed: Mutex::new(Placed::default()),
        });
        let id = heap.object_id;
        self.shared()
            .heap_keys
            .register(key, Arc::downgrade(&heap), id);
        Heap(heap)
    }

    /// A sparse heap of `descriptor`, whose options are `modes`, keeping `token` where given.
    fn new_sparse_heap(
        &self,
        descriptor: &HeapDescriptor,
        modes: Modes,
        token: Option<RangeToken>,
    ) -> Result<Heap, Error> {
        const CALL: &str = "new_heap";
        if modes.storage != StorageMode::Private {
            return Err(invalid(CALL, Problem::StorageMode));
        }
        if descriptor.size == 0 {
            return Err(nil(CALL, Problem::Heap(HeapProblem::Size)));
        }
        // Where no function reads an image's own residency, a texture the device makes no
        // sparse image of is emulated over an ordinary image, whatever the device's sparse
        // binding.
        if self.reads_image_residency() {
            self.shared().sparse.heaps().map_err(sparse_refusal)?;
        }
        let page = descriptor.sparse_page_size;
        let size = descriptor
            .size
            .checked_next_multiple_of(page.bytes())
            .ok_or(Error::OutOfMemory(MemoryKind::Device))?;
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let hazard = match modes.hazard {
            HazardTrackingMode::Default => HazardTrackingMode::Untracked,
            hazard => hazard,
        };
        let options = ResourceOptions::new(modes.storage, modes.cache, hazard);
        let sparse = SparseHeap::new(self.clone(), size, options, page);
        if let Some(token) = token {
            sparse.range_token.keep(token)?;
        }
        Ok(Heap(Arc::new(HeapShared {
            storage: HeapStorage::Sparse(sparse),
            heap_type: HeapType::Sparse,
            ranges: Mutex::new(RangeAllocator::new(0, size)),
            textures: Mutex::new(Numbered::default()),
            used: AtomicBool::new(false),
            object_id: ObjectId::new(),
            placed: Mutex::new(Placed::default()),
        })))
    }

    /// `heapBufferSizeAndAlignWithLength:options:`: `length` bytes at a multiple of 256, for every
    /// length and options, as Metal answers (measured).
    pub fn heap_buffer_size_and_align(
        &self,
        length: u64,
        _options: ResourceOptions,
    ) -> SizeAndAlign {
        SizeAndAlign {
            size: length,
            align: PLACEMENT_ALIGNMENT,
        }
    }
}

/// The size of a heap the descriptor asks `size` bytes (above 0) of, on a device of allocation
/// sizes `limits`: the size rounded up to 16 KiB. Refused past one allocation of the device
/// (`OutOfMemory`, as Metal answers nil) or one `VkBuffer` (`NotImplemented(HeapPastMaxBufferSize)`;
/// a `maxBufferSize` of 0 is one the device does not state).
pub(super) fn heap_size(size: u64, limits: &AllocationLimits) -> Result<u64, Error> {
    let size = size
        .checked_next_multiple_of(PAGE)
        .ok_or(Error::OutOfMemory(MemoryKind::Device))?;
    if size > limits.max_memory_allocation_size {
        return Err(Error::OutOfMemory(MemoryKind::Device));
    }
    if limits.max_buffer_size != 0 && size > limits.max_buffer_size {
        return Err(Error::NotImplemented(RuntimeGap::HeapPastMaxBufferSize {
            size,
            max_buffer_size: limits.max_buffer_size,
        }));
    }
    Ok(size)
}

impl Heap {
    /// The heap's own buffer (see [`HeapStorage::Bytes`]); `None` for a sparse heap, which has
    /// none.
    pub(super) fn bytes(&self) -> Option<&Buffer> {
        match &self.0.storage {
            HeapStorage::Bytes(bytes) => Some(bytes),
            HeapStorage::Sparse(_) => None,
        }
    }

    /// A sparse heap's state; `None` for any other heap.
    pub(super) fn sparse(&self) -> Option<&SparseHeap> {
        match &self.0.storage {
            HeapStorage::Bytes(_) => None,
            HeapStorage::Sparse(sparse) => Some(sparse),
        }
    }

    fn attributes(&self) -> &ResourceAttributes {
        match &self.0.storage {
            HeapStorage::Bytes(bytes) => bytes.attributes(),
            HeapStorage::Sparse(sparse) => &sparse.attributes,
        }
    }

    /// `device`.
    pub fn device(&self) -> &Device {
        match &self.0.storage {
            HeapStorage::Bytes(bytes) => bytes.device(),
            HeapStorage::Sparse(sparse) => &sparse.device,
        }
    }

    /// `setLabel:`.
    pub fn set_label(&self, label: &str) {
        self.attributes().set_label(label);
    }

    /// `label`: `None` until one is set.
    pub fn label(&self) -> Option<String> {
        self.attributes().label()
    }

    /// `type`.
    pub fn heap_type(&self) -> HeapType {
        self.0.heap_type
    }

    /// `size`: the descriptor's size rounded up to 16 KiB (a sparse heap's, to its page).
    pub fn size(&self) -> u64 {
        match &self.0.storage {
            HeapStorage::Bytes(bytes) => bytes.length(),
            HeapStorage::Sparse(sparse) => sparse.size,
        }
    }

    /// `usedSize`: an automatic heap's buffers' lengths, summed (those made aliasable left
    /// out); 0 for a placement heap; a sparse heap's mapped tiles times its page, a released
    /// texture's tiles still counted; as Metal answers.
    pub fn used_size(&self) -> u64 {
        match (&self.0.storage, self.0.heap_type) {
            (HeapStorage::Sparse(sparse), _) => sparse.used_size(),
            (HeapStorage::Bytes(_), HeapType::Automatic) => lock(&self.0.ranges).used(),
            (HeapStorage::Bytes(_), HeapType::Placement | HeapType::Sparse) => 0,
        }
    }

    /// `currentAllocatedSize`: the heap's size.
    pub fn current_allocated_size(&self) -> u64 {
        self.size()
    }

    /// `maxAvailableSizeWithAlignment:`: the longest buffer an automatic heap can place at a
    /// multiple of `alignment` (0 counts as 1); 0 for a placement heap, and a sparse heap's size
    /// whatever it maps; as Metal answers.
    ///
    /// Refused: an `alignment` other than 0 that is not a power of two (`Invalid`).
    pub fn max_available_size(&self, alignment: u64) -> Result<u64, Error> {
        if alignment != 0 && !alignment.is_power_of_two() {
            let problem = HeapProblem::Alignment { alignment };
            return Err(invalid("max_available_size", Problem::Heap(problem)));
        }
        Ok(match self.0.heap_type {
            HeapType::Automatic => lock(&self.0.ranges).longest(alignment.max(1)),
            HeapType::Placement => 0,
            HeapType::Sparse => self.size(),
        })
    }

    /// `storageMode`.
    pub fn storage_mode(&self) -> StorageMode {
        match &self.0.storage {
            HeapStorage::Bytes(bytes) => bytes.storage_mode(),
            HeapStorage::Sparse(_) => StorageMode::Private,
        }
    }

    /// `cpuCacheMode`.
    pub fn cpu_cache_mode(&self) -> CpuCacheMode {
        match &self.0.storage {
            HeapStorage::Bytes(bytes) => bytes.cpu_cache_mode(),
            HeapStorage::Sparse(sparse) => sparse
                .options
                .cpu_cache_mode()
                .expect("checked when the heap was made"),
        }
    }

    /// `hazardTrackingMode`: `Tracked` or `Untracked`; a heap's `Default` is untracked.
    pub fn hazard_tracking_mode(&self) -> HazardTrackingMode {
        match &self.0.storage {
            HeapStorage::Bytes(bytes) => bytes.hazard_tracking_mode(),
            HeapStorage::Sparse(sparse) => sparse
                .options
                .hazard_tracking_mode()
                .expect("resolved when the heap was made"),
        }
    }

    /// `resourceOptions`: the heap's modes, its hazard tracking mode resolved.
    pub fn resource_options(&self) -> ResourceOptions {
        match &self.0.storage {
            HeapStorage::Bytes(bytes) => bytes.resource_options(),
            HeapStorage::Sparse(sparse) => sparse.options,
        }
    }

    /// The heap's identity, a runtime extension ([`ObjectId`]): Metal names a heap by its
    /// pointer alone, and a sparse heap's `gpu_address` is 0, so this tells heaps apart where an
    /// address cannot, unique among the process's buffers, textures and heaps for the heap's
    /// life and never reused.
    pub fn object_id(&self) -> ObjectId {
        self.0.object_id
    }

    /// `gpuAddress` (Apple's private property, a runtime extension): the start of the heap's range
    /// of the Metal address space. A buffer placed at `offset` has this address plus `offset`. A
    /// sparse heap has no range: 0.
    pub fn gpu_address(&self) -> GpuAddress {
        match &self.0.storage {
            HeapStorage::Bytes(bytes) => bytes.gpu_address(),
            HeapStorage::Sparse(_) => GpuAddress(0),
        }
    }

    /// `setPurgeableState:`: the state before the call, or the state now for `KeepCurrent`, as a
    /// buffer's. Emptying a `Shared` heap zeroes its bytes until a committed command buffer held
    /// a command that used the heap, and keeps them from then on; emptying a `Private` one keeps
    /// them; as Metal did. A command uses the heap when it uses one of its resources (a blit, a
    /// render pass's attachment, a dispatch or draw whose functions read a bound heap buffer) or
    /// runs in an encoder that made the heap or one of its resources resident; a binding no
    /// function reads, and an encoder that encoded nothing after `use_heap`, do not. A heap over
    /// the caller's memory ([`Device::new_heap_with_bytes_no_copy`]) answers `NonVolatile` and
    /// keeps its bytes, as a no-copy buffer does.
    ///
    /// Fails only when zeroing the bytes fails.
    pub fn set_purgeable_state(&self, state: PurgeableState) -> Result<PurgeableState, Error> {
        // A heap over the caller's memory answers as a no-copy buffer: the bytes stay.
        if self
            .bytes()
            .is_some_and(|bytes| bytes.caller_memory().is_some())
        {
            return Ok(PurgeableState::NonVolatile);
        }
        let answer = self.attributes().set_purgeable_state(state);
        if let (PurgeableState::Empty, Some(bytes)) = (state, self.bytes()) {
            if self.storage_mode() == StorageMode::Shared && !self.0.used.load(Ordering::Relaxed) {
                bytes.clear()?;
            }
        }
        Ok(answer)
    }

    /// A committed command buffer held a command that used the heap (`set_purgeable_state`).
    pub(super) fn mark_used(&self) {
        self.0.used.store(true, Ordering::Relaxed);
    }

    /// Note `buffer`, just placed in the heap: whether a command buffer's commands used the heap
    /// already, so a command reaches it unnamed from now on (`live_buffers.rs`).
    pub(super) fn place(&self, buffer: &Buffer) -> bool {
        let mut placed = lock(&self.0.placed);
        placed.buffers.retain(WeakBuffer::lives);
        placed.buffers.push(buffer.downgrade());
        placed.reached
    }

    /// A command buffer's commands used the heap: the buffers placed in it that live, which a
    /// command reaches unnamed from now on, as it does a buffer placed in it later
    /// ([`place`](Self::place)).
    pub(super) fn reached(&self) -> Vec<Buffer> {
        let mut placed = lock(&self.0.placed);
        placed.reached = true;
        placed.buffers.retain(WeakBuffer::lives);
        placed
            .buffers
            .iter()
            .filter_map(WeakBuffer::upgrade)
            .collect()
    }

    /// The heap's pages were replaced (`rebind.rs`): no command buffer's commands used it since,
    /// so a command reaches its buffers unnamed again only once one does
    /// ([`reached`](Self::reached)). The buffers placed in it that live.
    pub(super) fn unreached(&self) -> Vec<Buffer> {
        let mut placed = lock(&self.0.placed);
        placed.reached = false;
        placed.buffers.retain(WeakBuffer::lives);
        placed
            .buffers
            .iter()
            .filter_map(WeakBuffer::upgrade)
            .collect()
    }

    /// `newBufferWithLength:options:` of an automatic heap: a buffer of `length` bytes at the
    /// lowest offset with room, a multiple of 256. Its bytes are the heap's as they are.
    ///
    /// Refused: a length of 0 or past the device's `max_buffer_length` (`Nil`,
    /// `Problem::BufferLength`) or the heap's size (`Nil`, `HeapProblem::PastEnd`), as plain
    /// Metal answers nil; a placement heap, options other than the heap's (`Invalid`); no room
    /// (`OutOfMemory(Heap)`).
    pub fn new_buffer(&self, length: u64, options: ResourceOptions) -> Result<Buffer, Error> {
        const CALL: &str = "new_buffer";
        // Metal makes a placed buffer with the private bit as without it (the heap's own options
        // decide where its writes go).
        let options = options.without_private_bit().0;
        let options = self.placed_options(CALL, options)?;
        let bytes = self.check_type(CALL, HeapType::Automatic)?;
        if length == 0 {
            return Err(nil(CALL, Problem::BufferLength { length }));
        }
        self.check_length(CALL, length)?;
        if length > self.size() {
            return Err(nil(CALL, self.past_end(0, length)));
        }
        if self.device().is_lost() {
            return Err(Error::DeviceLost);
        }
        let offset = self
            .allocate(length, PLACEMENT_ALIGNMENT)
            .ok_or(Error::OutOfMemory(MemoryKind::Heap))?;
        Ok(self.program_buffer(bytes, offset, length, options))
    }

    /// `newBufferWithLength:options:offset:` of a placement heap: a buffer of `length` bytes
    /// `offset` bytes into the heap. Its bytes are the heap's as they are: a buffer over bytes
    /// another wrote reads them. A length of 0 makes a buffer of no bytes at `offset`, as plain
    /// Metal makes it where its validation stops it (Metal heap probe `length_zero` on an M4:
    /// length and `allocatedSize` 0).
    ///
    /// Refused: a length past the device's `max_buffer_length` (`Problem::BufferLength`), an
    /// offset that is not a multiple of 256 (`HeapProblem::Offset`), bytes past the heap's end
    /// (`HeapProblem::PastEnd`), each `Nil` as plain Metal answers nil; an
    /// automatic heap, options other than the heap's (`Invalid`).
    pub fn new_buffer_at_offset(
        &self,
        length: u64,
        options: ResourceOptions,
        offset: u64,
    ) -> Result<Buffer, Error> {
        const CALL: &str = "new_buffer_at_offset";
        // As `new_buffer`: the private bit is the heap's to take.
        let options = options.without_private_bit().0;
        let options = self.placed_options(CALL, options)?;
        let bytes = self.check_type(CALL, HeapType::Placement)?;
        self.check_length(CALL, length)?;
        if !offset.is_multiple_of(PLACEMENT_ALIGNMENT) {
            return Err(nil(CALL, Problem::Heap(HeapProblem::Offset { offset })));
        }
        if offset
            .checked_add(length)
            .is_none_or(|end| end > self.size())
        {
            return Err(nil(CALL, self.past_end(offset, length)));
        }
        if self.device().is_lost() {
            return Err(Error::DeviceLost);
        }
        Ok(self.program_buffer(bytes, offset, length, options))
    }

    /// The program's buffer of `length` bytes with `options`, placed `offset` bytes into the
    /// heap (`placed_buffer`), listed among the device's live buffers at its own range where a
    /// command buffer's commands used the heap already (`live_buffers.rs`).
    fn program_buffer(
        &self,
        bytes: &Buffer,
        offset: u64,
        length: u64,
        options: ResourceOptions,
    ) -> Buffer {
        let buffer = self.placed_buffer(bytes, offset, length, options);
        self.device().shared().live_buffers.placed(self, &buffer);
        buffer
    }

    /// The options of a buffer or texture `call` places with `options`: the heap's, its hazard
    /// tracking mode resolved, where `options` names the heap's storage and CPU cache modes. A
    /// hazard tracking mode named other than the heap's becomes the heap's, as plain Metal makes
    /// it where its validation stops it (measured for a buffer, and for a texture by our probe).
    /// Refused: another storage or CPU cache mode, which ends Metal's process.
    pub(super) fn placed_options(
        &self,
        call: &'static str,
        options: ResourceOptions,
    ) -> Result<ResourceOptions, Error> {
        let asked = Modes::of(options)?;
        let heap = self.resource_options();
        let (storage, cache) = (self.storage_mode(), self.cpu_cache_mode());
        let problem = if asked.storage != storage {
            HeapProblem::StorageMode {
                heap: storage,
                resource: asked.storage,
            }
        } else if asked.cache != cache {
            HeapProblem::CpuCacheMode {
                heap: cache,
                resource: asked.cache,
            }
        } else {
            return Ok(heap);
        };
        Err(invalid(call, Problem::Heap(problem)))
    }

    /// The heap's own buffer, refused unless the heap is of `heap_type` (automatic or
    /// placement).
    pub(super) fn check_type(
        &self,
        call: &'static str,
        heap_type: HeapType,
    ) -> Result<&Buffer, Error> {
        if let (true, Some(bytes)) = (self.0.heap_type == heap_type, self.bytes()) {
            return Ok(bytes);
        }
        let problem = HeapProblem::HeapType {
            heap_type: self.0.heap_type,
        };
        Err(invalid(call, Problem::Heap(problem)))
    }

    /// Refused for a length past the device's `max_buffer_length`, as a buffer's (`Nil`, as
    /// plain Metal answers nil).
    fn check_length(&self, call: &'static str, length: u64) -> Result<(), Error> {
        if length > self.device().properties().max_buffer_length {
            return Err(nil(call, Problem::BufferLength { length }));
        }
        Ok(())
    }

    /// `length` bytes at `offset` past the heap's end.
    pub(super) fn past_end(&self, offset: u64, length: u64) -> Problem {
        Problem::Heap(HeapProblem::PastEnd {
            offset,
            length,
            heap_size: self.size(),
        })
    }

    /// An automatic heap's lowest free range of `length` bytes at a multiple of `alignment`, now
    /// taken; `None` where none has room.
    pub(super) fn allocate(&self, length: u64, alignment: u64) -> Option<u64> {
        lock(&self.0.ranges).allocate(length, alignment)
    }

    /// `texture`, made in the heap, is among the heap's textures while it lives, numbered after
    /// every texture the heap made before it.
    pub(super) fn hold_texture(&self, texture: &Texture) {
        lock(&self.0.textures).hold(texture.downgrade());
    }

    /// The number the next texture the heap makes takes: how many it made.
    pub(super) fn textures_made(&self) -> u64 {
        lock(&self.0.textures).made()
    }

    /// The heap's live textures numbered `from` or later, in the order they were made, and the
    /// number the next texture it makes takes.
    pub(super) fn textures_from(&self, from: u64) -> (Vec<Texture>, u64) {
        let textures = lock(&self.0.textures);
        let live = textures.since(from).filter_map(Texture::upgrade).collect();
        (live, textures.made())
    }

    /// The resource of `length` bytes at `offset` is gone: an automatic heap's range goes back
    /// to the heap unless `make_aliasable` gave it back already.
    pub(super) fn release(&self, offset: u64, length: u64, aliasable: &AtomicBool) {
        if self.0.heap_type != HeapType::Automatic {
            return;
        }
        let mut ranges = lock(&self.0.ranges);
        if !aliasable.load(Ordering::Acquire) {
            ranges.free(offset, length);
        }
    }

    /// `makeAliasable` of the resource of `length` bytes at `offset`: marked, and in an
    /// automatic heap its range given back once.
    pub(super) fn make_aliasable(&self, offset: u64, length: u64, aliasable: &AtomicBool) {
        let mut ranges = lock(&self.0.ranges);
        if !aliasable.swap(true, Ordering::AcqRel) && self.0.heap_type == HeapType::Automatic {
            ranges.free(offset, length);
        }
    }
}
