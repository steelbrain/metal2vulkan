//! `Buffer` (`MTLBuffer`) and `Device::new_buffer`.
//!
//! A buffer is a `VkBuffer` placed in the device's sub-allocator (one [`MemoryClass`] per storage
//! mode, `Shared`'s by its CPU cache mode, [`buffer_class`]; at a multiple of
//! [`LINEAR_IMAGE_ALIGNMENT`]), or in the caller's host memory for a no-copy buffer (step 4,
//! `host_memory.rs`), and a range of the device's Metal address space, which `gpu_address`
//! answers. The `VkBuffer` is the length rounded up to 16 bytes (the binder's
//! padding contract), and has a Vulkan device address where the device enables
//! `bufferDeviceAddress`: a placed buffer its block's (`memory.rs`, `memory::allocate_addressed`;
//! made without the device-address usage itself, so the driver counts no addressable buffer per
//! program buffer), any other its own. Every new buffer the runtime allocates reads zero, as Metal's do: memory
//! the host reaches is cleared through its mapping, the rest by a fill on the device before
//! `new_buffer` returns.
//!
//! Step 1 makes `Shared` and `Private` buffers, step 2 `Managed` ones (host-coherent memory,
//! cached where the device has it, and on a device without unified memory a device-local copy
//! with such a host copy, [`Mirror`]: `managed.rs`), step 4 buffers over host memory and from bytes
//! (`buffer_descriptor.rs`), labels and purgeable state (`attributes.rs`); `Memoryless` is not a
//! buffer's storage mode in Metal (refused `Invalid`). Step 7 places buffers in heaps
//! (`heap.rs`): such a buffer is bytes of its heap's own buffer, from its offset there
//! ([`Bytes::Heap`]), so it has no `VkBuffer`, memory or address range of its own.

use super::address::{AddressPlacement, GpuAddress, Held, KeptToken, Place, RangeToken};
use super::attributes::ResourceAttributes;
use super::buffer_texture::TexelViews;
use super::churn_counts::Recycled;
use super::degradation::Degradation;
use super::device::{lock, DeviceShared};
use super::dirty_pages::{DirtySource, PagesLog};
use super::error::{internal, invalid, nil, vulkan, Error, Problem};
use super::exit_guard;
use super::fresh_pages;
use super::generated_commands::PREPROCESS_BUFFER;
use super::heap::{Heap, HeapType};
use super::host_aliases::Alias;
use super::host_memory::{HeldBack, KeepAlive, Release};
use super::host_ram::RamOnDevice;
#[cfg(unix)]
use super::host_residency::{Again, Residency};
use super::memory::{Mapped, MemoryClass, Placement, BLOCK_BYTES};
use super::program_objects::ProgramObject;
use super::rebind::{ImportHold, Unexact};
use super::recycle::{self, BufferMaking};
use super::resource::{Identity, ObjectId, ResourceKey};
use super::retained::HeldBy;
use super::texture::first_move::Filled;
use super::values::{CpuCacheMode, HazardTrackingMode, ResourceOptions, StorageMode};
use super::Device;
use crate::release;
use ash::vk;
use metal2vulkan::bind::ResidentRange;
use std::any::Any;
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::time::Instant;

/// A Metal buffer (`MTLBuffer`). Cheap to clone (a retain); `Send + Sync`. Two handles are equal
/// when they are the same buffer.
#[derive(Clone)]
pub struct Buffer(Arc<BufferShared>);

/// A buffer held without keeping it (`used_resources.rs`).
#[derive(Clone)]
pub(super) struct WeakBuffer(Weak<BufferShared>);

impl WeakBuffer {
    /// The buffer, while it lives.
    pub(super) fn upgrade(&self) -> Option<Buffer> {
        self.0.upgrade().map(Buffer)
    }

    /// Whether the buffer lives.
    pub(super) fn lives(&self) -> bool {
        self.0.strong_count() > 0
    }
}

struct BufferShared {
    device: Device,
    /// Its identity (`object_id`).
    object_id: ObjectId,
    length: u64,
    /// The options with the hazard tracking mode resolved, as `resourceOptions` reports them.
    options: ResourceOptions,
    address: GpuAddress,
    attributes: ResourceAttributes,
    bytes: Bytes,
    /// The encoder core that holds it from an argument table's retain (`retained::HeldBy`).
    held_by: HeldBy,
}

/// Where a buffer's bytes are.
enum Bytes {
    /// Its own: a `VkBuffer`, memory and a range of the address space, released with it.
    Own(OwnBytes),
    /// Its heap's (`heap.rs`).
    Heap(HeapBytes),
}

/// A buffer's own bytes.
struct OwnBytes {
    /// Where they lie: for good, or a no-copy buffer's host pages, which a rebind replaces
    /// (`rebind.rs`).
    lie: Lie,
    /// The bytes of the Metal address space the buffer holds.
    reserved: Held,
    identity: Identity,
    owner: Owner,
    /// A `Managed` buffer's host copy on a device without unified memory: its `VkBuffer` is then
    /// in device-local memory, and `contents` maps the copy. Shared with the submissions that
    /// upload from it.
    mirror: Option<Arc<Mirror>>,
    /// The caller's token for the buffer's range (a buffer's `range_token`, or its heap's when
    /// the buffer is a heap's own), set once as it is made: dropped right after the range is
    /// free and the buffer's teardown leaves its exit section, no runtime lock held, on the
    /// thread that drops the buffer's last holder, as the program's code (`exit_guard::program`);
    /// left to the process's exit with the buffer where the runtime leaves the buffer to it
    /// (`BufferShared`'s drop).
    range_token: KeptToken,
}

/// Where a buffer of its own lies: its `VkBuffer` in its memory.
pub(super) struct Pages {
    raw: vk::Buffer,
    memory: Bound,
    /// Where the buffer's bytes start in its placement: a no-copy buffer's offset in its pages.
    offset: u64,
    /// Where the buffer's byte 0 lies in `raw`: 0, but for a no-copy buffer whose pointer misses
    /// the buffer's memory alignment, bound that many bytes below it.
    base: u64,
    /// The bytes from byte 0 a copy reads ([`Buffer::copy_reach`]).
    reach: u64,
}

/// Where a buffer of its own lies, for good or until a rebind.
enum Lie {
    /// The runtime's memory, or the caller's `VkBuffer`.
    Fixed(Arc<FixedPages>),
    /// A no-copy buffer's host pages: the pages it lies in now, which a rebind replaces.
    Host(Arc<HostCell>),
}

/// A no-copy buffer's (or a no-copy heap's own buffer's) hold on its host pages, which
/// `rebind_bytes_no_copy` replaces (`rebind.rs`) and `detach_backing` lets go of (`detach.rs`).
pub(super) struct HostCell {
    pages: Mutex<Lying>,
    /// The texel buffer views of the textures over the buffer's bytes (a no-copy heap's: over
    /// its buffers'), which view its pages: a rebind empties them.
    texels: Mutex<Vec<Weak<TexelViews>>>,
    /// The images of the textures over the buffer's bytes (a no-copy heap's: over its buffers'
    /// and in it), which a detach makes present stand-ins (`detach.rs`), and the textures being
    /// made over them, which a detach refuses while one is.
    textures: Mutex<TexturesOver>,
    /// The images in place over the buffer's bytes (a no-copy heap's: over its buffers'), which a
    /// rebind makes again over its new pages (`texture/in_place_image.rs`).
    in_place: Mutex<Vec<InPlaceWatch>>,
    /// The reasons the buffer's Metal address was handed out for (`Buffer::gpu_address`,
    /// `Buffer::exposed_address`), a bit each ([`ReadApart`]): once any is, its import is kept
    /// imported for good (`host_residency.rs`), and a call for a reason noted need not look
    /// again.
    exposed: AtomicU8,
    /// The device whose work counts each look at its pages adds to
    /// (`WorkCounts::no_copy_lies_read`).
    device: Device,
    /// The GPU's writes to the buffer are dropped, as Metal's private option bit 24 drops them:
    /// its pages are imported through a read-only view of them (`read_only_view.rs`), which a
    /// rebind's are too.
    drops_writes: bool,
    /// The deallocator of the pages its last detach let go of (`detach.rs`), held back until its
    /// next rebind or its release, as Metal runs it then ([`Release::hold_back`]).
    held_back: Mutex<Option<Arc<HeldBack>>>,
}

/// An image in place over a buffer's host pages ([`HostCell`], `texture/in_place_image.rs`),
/// which a rebind makes again over the new pages: the image (`texture.rs`'s `Image`), held weakly
/// and by type erased as [`NotedTexture`] holds it, where its texels lie in bytes from the pages'
/// first byte, and how Metal lays them out from there.
#[derive(Clone)]
pub(super) struct InPlaceWatch {
    pub(super) image: Weak<dyn Any + Send + Sync>,
    pub(super) from_base: u64,
    pub(super) layout: crate::linear_plan::BufferLayout,
}

impl InPlaceWatch {
    /// Whether the image lives.
    fn lives(&self) -> bool {
        self.image.strong_count() > 0
    }
}

/// The textures over a buffer's host pages ([`HostCell`]): their images, which a detach makes
/// present stand-ins, and how many are being made, which a detach refuses while any is
/// (`detach.rs`).
#[derive(Default)]
pub(super) struct TexturesOver {
    /// The textures made over them, each its own entry (a view's too).
    noted: Vec<NotedTexture>,
    /// How many are being made now ([`TextureMaking`]).
    making: u32,
}

/// A texture over a buffer's host pages (`TexturesOver`): the texture and its image
/// (`texture.rs`'s `TextureShared` and `Image`), held weakly and by type erased: each holds its
/// device, so naming their types here would make the buffer's types recursive through them,
/// which rustdoc's search for their auto traits overflows on.
pub(super) struct NotedTexture {
    texture: Weak<dyn Any + Send + Sync>,
    image: Weak<dyn Any + Send + Sync>,
    /// The texture's `ObjectId`.
    pub(super) id: ObjectId,
    /// Whether a stand-in of zero bytes presents its image exactly: not for an image of texels
    /// its copies convert from the bytes (`texture/stand_in.rs`).
    exact: bool,
    /// Whether its image's stand-in has a sink (its plan has `STORAGE` usage): a read through a
    /// storage-class element there reads what an unnamed write left (`Device::note_stale`).
    sink: bool,
}

impl NotedTexture {
    /// `texture`, whose image is `image`, of object `id`; `exact` and `sink` as the fields say.
    pub(super) fn new(
        texture: Weak<dyn Any + Send + Sync>,
        image: Weak<dyn Any + Send + Sync>,
        id: ObjectId,
        (exact, sink): (bool, bool),
    ) -> NotedTexture {
        NotedTexture {
            texture,
            image,
            id,
            exact,
            sink,
        }
    }

    /// Whether its image's stand-in has a sink (`NotedTexture::sink`).
    pub(super) fn sink(&self) -> bool {
        self.sink
    }

    /// Whether the texture lives.
    fn lives(&self) -> bool {
        self.texture.strong_count() > 0
    }

    /// The texture, by type erased, while it lives.
    pub(super) fn texture(&self) -> Weak<dyn Any + Send + Sync> {
        Weak::clone(&self.texture)
    }
}

impl TexturesOver {
    /// The images of the live textures, each once, by type erased; and whether a stand-in of
    /// zero bytes presents each exactly (`NotedTexture::exact`).
    pub(super) fn images(&self) -> (Vec<Arc<dyn Any + Send + Sync>>, bool) {
        let mut images: Vec<Arc<dyn Any + Send + Sync>> = Vec::new();
        let mut exact = true;
        for noted in self.noted.iter().filter(|noted| noted.lives()) {
            let Some(image) = noted.image.upgrade() else {
                continue;
            };
            exact &= noted.exact;
            if !images.iter().any(|held| Arc::ptr_eq(held, &image)) {
                images.push(image);
            }
        }
        (images, exact)
    }

    /// The live textures.
    pub(super) fn textures(&self) -> impl Iterator<Item = &NotedTexture> {
        self.noted.iter().filter(|noted| noted.lives())
    }
}

/// How many times a no-copy buffer of the process (or a no-copy heap's own buffer) came to lie
/// elsewhere ([`HostCell::replace`]: rebound or detached).
static HOST_CELLS_MOVED: AtomicU64 = AtomicU64::new(0);

/// The count of the times a no-copy buffer of the process came to lie elsewhere (rebound or
/// detached) so far: while it holds, every buffer's binding is what it was (its base, its
/// addresses), which is what an encoder's kept bind state reads of them (`live_bind.rs`).
pub(super) fn host_cells_moved() -> u64 {
    HOST_CELLS_MOVED.load(Ordering::Acquire)
}

/// A texture being made over a buffer's host pages ([`Buffer::making_texture`]): a detach
/// refuses until it goes, once the texture is noted (`Buffer::note_texture`) or its making
/// failed.
pub(super) struct TextureMaking(Arc<HostCell>);

impl Drop for TextureMaking {
    fn drop(&mut self) {
        let mut textures = lock(&self.0.textures);
        textures.making = textures.making.saturating_sub(1);
    }
}

/// Where a no-copy buffer (or a no-copy heap's own buffer) lies: in host pages, or detached
/// from them (`detach.rs`).
#[derive(Clone)]
pub(super) enum Lying {
    /// The host pages it lies in.
    In(Arc<HostPages>),
    /// Detached: in no pages.
    Detached(Arc<Detached>),
}

/// What a detached no-copy buffer's commands take of it while it is detached (`detach.rs`): the
/// device's stand-in's `VkBuffer`, at the place the buffer's bytes had in the pages it was detached
/// from (their offset, base and reach), and the stand-in's memory. No command that takes it runs:
/// an encoder that names a detached buffer records nothing from that command on, and its submission
/// fails (`EncoderCore::skips`).
pub(super) struct Detached {
    /// The stand-in, a small buffer of the runtime's own, kept while a buffer is detached.
    _stand_in: Buffer,
    pages: Pages,
    /// The index of the memory type of the pages it was detached from: a texture made over the
    /// bytes meanwhile takes the route it would over them (`texture/heap.rs`).
    type_index: u32,
    /// Whether the pages it was detached from were imported as dma-bufs, for the same.
    over_dma_buf: bool,
}

impl Detached {
    /// What a buffer detached from `from` takes: `stand_in`'s `VkBuffer` and memory, at the
    /// place its bytes had in `from`. No memory is placed for it: the placement names the
    /// stand-in's, unmapped and freed by none, so `contents` answers `None` while it is detached.
    /// Refused (an internal error) for a stand-in placed in a heap.
    pub(super) fn new(stand_in: Buffer, from: &HostPages) -> Result<Detached, Error> {
        let Bytes::Own(own) = &stand_in.0.bytes else {
            return Err(internal("a detached buffer's stand-in placed in a heap"));
        };
        let old = from.import.shape;
        let pages = own.with(ReadApart::Binding, |pages| {
            let placement = &pages.memory.placement;
            Pages {
                raw: pages.raw,
                memory: Bound {
                    placement: Placement::caller(
                        placement.memory,
                        placement.offset,
                        placement.size(),
                        placement.type_index(),
                        None,
                    ),
                    device_address: pages.memory.device_address,
                },
                offset: old.offset,
                base: old.base,
                reach: old.reach,
            }
        });
        Ok(Detached {
            _stand_in: stand_in,
            pages,
            type_index: from.type_index(),
            over_dma_buf: from.over_dma_buf(),
        })
    }
}

impl HostCell {
    /// Whether the GPU's writes to the buffer are dropped (`read_only_view.rs`).
    pub(super) fn drops_writes(&self) -> bool {
        self.drops_writes
    }

    /// The pages the buffer lies in now; `None` while it is detached.
    pub(super) fn current(&self) -> Option<Arc<HostPages>> {
        self.device.shared().counts.no_copy_lie_read();
        match &*lock(&self.pages) {
            Lying::In(pages) => Some(Arc::clone(pages)),
            Lying::Detached(_) => None,
        }
    }

    /// Where the buffer lies now.
    pub(super) fn lying(&self) -> Lying {
        self.device.shared().counts.no_copy_lie_read();
        lock(&self.pages).clone()
    }

    /// The host addresses of the buffer's bytes, from its byte 0 over the bytes a copy reads,
    /// where it lies in host pages now; `None` while it is detached.
    pub(super) fn host_range(&self) -> Option<std::ops::Range<usize>> {
        match self.lying() {
            Lying::In(pages) => {
                let shape = pages.import.shape;
                let mapped = shape.mapped?;
                let start = (mapped.as_ptr() as usize).checked_add(shape.offset as usize)?;
                Some(start..start.checked_add(shape.reach as usize)?)
            }
            Lying::Detached(_) => None,
        }
    }

    /// Whether the buffer is detached.
    pub(super) fn detached(&self) -> bool {
        matches!(*lock(&self.pages), Lying::Detached(_))
    }

    /// Where the bytes start in their first page: the pages', or those it was detached from.
    pub(super) fn offset(&self) -> u64 {
        match &*lock(&self.pages) {
            Lying::In(pages) => pages.offset(),
            Lying::Detached(detached) => detached.pages.offset,
        }
    }

    /// Lie `lying` from now on: where the buffer lay, which the caller lets go of. Counted in
    /// [`host_cells_moved`]: a buffer binding that reads where it lies (its base, its device
    /// address) may change.
    pub(super) fn replace(&self, lying: Lying) -> Lying {
        let lay = {
            let mut pages = lock(&self.pages);
            let lay = std::mem::replace(&mut *pages, lying);
            HOST_CELLS_MOVED.fetch_add(1, Ordering::AcqRel);
            lay
        };
        lay
    }

    /// A detach let go of `pages`: their deallocator is held back until the buffer's next rebind
    /// or its release ([`HostCell::release_held_back`]), as Metal never runs it at a detach.
    pub(super) fn hold_back(&self, pages: &HostPages) {
        let Some(held) = pages.release.get().and_then(Release::hold_back) else {
            return;
        };
        let earlier = lock(&self.held_back).replace(held);
        // Never so: a detached buffer is detached again by no detach, and a rebind runs what
        // was held back first. Were it to be, the earlier one is due now.
        if let Some(earlier) = earlier {
            self.held_back_due(&earlier);
        }
    }

    /// The buffer's next rebind (inside the call, once its new pages are in place) or its release
    /// came: the deallocator its last detach held back runs now, on this thread, as Metal runs it
    /// there, or, where the pages it let go of are still imported (a texture's image over them
    /// waits for a command buffer that may read it) and no keep-alive keeps them mapped, once
    /// their import is freed, announced (`host-deallocator-delayed`).
    pub(super) fn release_held_back(&self) {
        let held = lock(&self.held_back).take();
        if let Some(held) = held {
            self.held_back_due(&held);
        }
    }

    /// `held` is due ([`HostCell::release_held_back`]).
    fn held_back_due(&self, held: &HeldBack) {
        if !held.due() {
            return;
        }
        let device = &self.device;
        device.shared().counts.host_deallocator_delayed();
        device.degrade_once(Degradation::HostDeallocatorDelayed, || {
            (
                "a detached no-copy buffer".to_owned(),
                "the pages its detach let go of are still imported as it is rebound or released \
                 (a texture's image over them may still be read through its resource ID by a \
                 command buffer), and the caller gave no keep-alive: its deallocator runs once \
                 their import is freed"
                    .to_owned(),
            )
        });
    }

    /// Note `noted`, a texture over the buffer's bytes (`TexturesOver`), forgetting those
    /// gone.
    fn watch_texture(&self, noted: NotedTexture) {
        let mut textures = lock(&self.textures);
        textures.noted.retain(NotedTexture::lives);
        textures.noted.push(noted);
    }

    /// Note `watch`, an image in place over the buffer's bytes, forgetting those gone.
    fn watch_in_place(&self, watch: InPlaceWatch) {
        let mut in_place = lock(&self.in_place);
        in_place.retain(InPlaceWatch::lives);
        in_place.push(watch);
    }

    /// The live images in place over the buffer's bytes, which a rebind makes again.
    pub(super) fn in_place_images(&self) -> Vec<InPlaceWatch> {
        let mut in_place = lock(&self.in_place);
        in_place.retain(InPlaceWatch::lives);
        in_place.clone()
    }

    /// The textures over the buffer's bytes, locked, where none is being made: none is made
    /// while it is held (`Buffer::making_texture`). `None` where one is.
    pub(super) fn settled(&self) -> Option<MutexGuard<'_, TexturesOver>> {
        let textures = lock(&self.textures);
        (textures.making == 0).then_some(textures)
    }

    /// The texel views of the live textures over the buffer's bytes.
    pub(super) fn texels(&self) -> Vec<Arc<TexelViews>> {
        lock(&self.texels)
            .iter()
            .filter_map(Weak::upgrade)
            .collect()
    }

    /// Note `views`, a texture's over the buffer's bytes, forgetting those gone.
    fn watch(&self, views: &Arc<TexelViews>) {
        let mut texels = lock(&self.texels);
        texels.retain(|views| views.strong_count() > 0);
        texels.push(Arc::downgrade(views));
    }
}

/// A hold on the pages a buffer of its own lies in (`Buffer::resident`), kept and never read:
/// they stay while it lives.
#[derive(Clone)]
pub(super) enum PagesHold {
    Fixed { _pages: Arc<FixedPages> },
    Host { _pages: ListedPages },
}

/// A live buffers' table's hold on host pages (`PagesHold::Host`), counted
/// (`HostPages::listed`): one left as the buffer is released is a table a command buffer that may
/// reach the pages unnamed holds, past Metal's release (`HostPages::released_listed`).
pub(super) struct ListedPages(Arc<HostPages>);

impl ListedPages {
    fn new(pages: Arc<HostPages>) -> ListedPages {
        pages.listed.fetch_add(1, Ordering::AcqRel);
        pages.import.note_listed();
        ListedPages(pages)
    }
}

impl Clone for ListedPages {
    fn clone(&self) -> ListedPages {
        ListedPages::new(Arc::clone(&self.0))
    }
}

impl Drop for ListedPages {
    fn drop(&mut self) {
        self.0.listed.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A hold on the memory a buffer's bytes lie in now, and on nothing else of the buffer
/// (`Buffer::memory_hold`), kept and never read: an image bound to them (a texture's over the
/// buffer, a heap's texture's) keeps it while a command buffer may still reach the image, past the
/// buffer and texture objects. Metal frees those at their release: the buffer's range of the
/// address space and its range token go with the buffer, and a no-copy buffer's deallocator runs as
/// its pages go; only the `VkBuffer` and its memory, or host pages' import, stay.
#[derive(Clone)]
pub(super) enum MemoryHold {
    Fixed { _pages: Arc<FixedPages> },
    Host { _import: Arc<HostImport> },
}

/// The pages a buffer of its own lies in for good: its `VkBuffer` and its placement. Held by the
/// buffer, and by each command buffer whose resident set lists them until it completed
/// (`live_buffers.rs`): the last lets go, which parks the `VkBuffer` and its placement for the
/// next buffer of the same making where the recycler noted them (`recycle.rs`), else destroys the
/// `VkBuffer` (unless it is the caller's, `Owner::Vulkan`) and frees the placement.
pub(super) struct FixedPages {
    device: Device,
    pages: Pages,
    /// The `VkBuffer` is the runtime's to destroy.
    owned: bool,
}

impl Drop for FixedPages {
    fn drop(&mut self) {
        // The Vulkan teardown is a section of the exit guard.
        let Some(busy) = exit_guard::enter() else {
            // The process is exiting: the buffer and its memory are left to its end.
            return;
        };
        let shared = self.device.shared();
        // No pending command buffer uses the pages: one whose commands used the buffer holds it,
        // and one whose resident set lists them holds them, until it completed.
        let placement = &self.pages.memory.placement;
        let parked = match self.owned {
            true => (lock(&shared.recycled_buffers))
                .park(self.pages.raw, placement.handed_over(), Instant::now())
                .ok(),
            false => None,
        };
        match parked {
            Some(gone) => {
                shared.counts.churn.parked(Recycled::Buffer);
                self.device.let_go_parked_buffers(gone);
            }
            None => {
                if self.owned {
                    // SAFETY: as above.
                    unsafe { release::object(shared.vk(), self.pages.raw) };
                }
                // The caller's `VkBuffer`'s placement returns nothing.
                lock(&shared.memory).free(shared.vk(), placement);
            }
        }
        drop(busy);
    }
}

/// Host pages a no-copy buffer lies in: its `VkBuffer` over them and their import
/// (`HostImport`), the images a rebind parked, and their release. Held by the buffer while it
/// lies in them and by each command buffer whose commands used them until it completed
/// (`rebind.rs`); the last lets go: the parked images are destroyed, the import let go (destroyed
/// where these pages held it last), and the deallocator run.
///
/// The import may outlive them: a texture over the pages whose image a command buffer submitted
/// before the texture's release may still read through its resource ID holds it (`heap_sets.rs`).
/// Vulkan requires the pages to stay mapped while imported, so the deallocator runs as they go only
/// where the caller's keep-alive keeps the pages mapped (the import keeps it until freed), or
/// another buffer lies in the import, whose contract keeps them mapped; else it runs once the
/// import is freed, at once where nothing else holds it, and announced where something does
/// (`host-deallocator-delayed`). A keep-alive goes as they go where the import's pages stay mapped
/// without it: another buffer lies in the import, or the import keeps another's keep-alive already
/// (every buffer in it was made over the same bytes, `host_aliases.rs`). So an import keeps one
/// keep-alive at most, however many buffers come and go over its bytes while it lives.
pub(super) struct HostPages {
    /// The images and views over them a rebind or a detach replaced (a no-copy heap's textures',
    /// texel buffer views), which commands recorded before it may use: destroyed with them,
    /// first.
    parked: Mutex<Vec<Box<dyn Any + Send + Sync>>>,
    /// Their `VkBuffer` and its memory, which no-copy buffers made over the same bytes share
    /// (`host_aliases.rs`).
    import: Arc<HostImport>,
    /// Their release to the caller's deallocator and keep-alive, set once they are the buffer's
    /// (a refused rebind drops its pages without it, and the deallocator unrun).
    release: OnceLock<Release>,
    /// The live buffers' tables' holds on them ([`ListedPages`]).
    listed: AtomicUsize,
    /// The caller's log of the writes to them, where it gave one with their memory
    /// (`HostMemory::with_dirty_pages`, `dirty_pages.rs`).
    dirty: OnceLock<Arc<DirtySource>>,
    /// The guest RAM their caller resolved them to and the runs of it they are, in their byte
    /// order (`HostMemory::with_runs`, `host_ram.rs`), kept until the rebind or the release.
    ram_runs: OnceLock<(Arc<RamOnDevice>, Box<[Range<u64>]>)>,
}

impl HostPages {
    /// New pages lying in `import`, with no release yet.
    fn new(import: Arc<HostImport>) -> HostPages {
        if import.lying.fetch_add(1, Ordering::AcqRel) > 0 {
            // Another buffer lies in it: imported for good (`host_residency.rs`).
            import.expose(ReadApart::SecondBuffer);
        }
        HostPages {
            parked: Mutex::new(Vec::new()),
            import,
            release: OnceLock::new(),
            listed: AtomicUsize::new(0),
            dirty: OnceLock::new(),
            ram_runs: OnceLock::new(),
        }
    }

    /// They lie in a copy of the caller's pages, registered as `copied` (`copied.rs`).
    pub(super) fn set_copied(&self, copied: crate::copied::Copied) {
        self.import.set_copied(copied);
    }

    /// Keep the caller's log of the writes to them (`dirty_pages.rs`), given with their memory.
    pub(super) fn keep_dirty(&self, dirty: Arc<DirtySource>) {
        let _kept = self.dirty.set(dirty);
    }

    /// Keep the logs of the writes to them (`dirty_pages::PagesLog`), with their import.
    pub(super) fn keep_log(&self, log: Arc<PagesLog>) {
        self.import.keep_log(log);
    }

    /// The logs of the writes to them, where their memory came with one.
    pub(super) fn log(&self) -> Option<&Arc<PagesLog>> {
        self.import.log.get()
    }

    /// Name `key`, a buffer's (or a heap's bytes') lying in them, to their copy, where they are
    /// one: its merges back compare the copy where a batch wrote that key (`copied.rs`).
    pub(super) fn copied_as(&self, key: ResourceKey) {
        if let Some(copied) = self.import.copied.get() {
            copied.name(key);
        }
    }

    /// Report the GPU's writes through the uses of `key`, a buffer's (or a heap's bytes') lying
    /// in them, to their logs (`DirtyShadows::logged_in_place`), where they have one and lie in
    /// an import the GPU writes in place: a copy reports its merges back instead (`copied.rs`),
    /// and a read-only view takes no write.
    pub(super) fn logged_as(&self, key: ResourceKey) {
        let import = &self.import;
        if import.copied.get().is_some() || import.drops_writes() {
            return;
        }
        if let Some(log) = import.log.get() {
            import.device.shared().dirty.logged_in_place(key, log);
        }
    }

    /// Keep the guest RAM and the runs of it their caller resolved them to (`host_ram.rs`).
    pub(super) fn keep_runs(&self, runs: (Arc<RamOnDevice>, Box<[Range<u64>]>)) {
        let _kept = self.ram_runs.set(runs);
    }

    /// The caller's log of the writes to them, where it gave one.
    pub(super) fn dirty(&self) -> Option<&Arc<DirtySource>> {
        self.dirty.get()
    }

    /// Keep `objects`, over these pages, until they go.
    pub(super) fn park(&self, objects: impl Any + Send + Sync) {
        lock(&self.parked).push(Box::new(objects));
    }

    /// Their memory and where the bytes start in it (a heap's textures bind images there): a
    /// read that keeps the import imported for good (`host_residency.rs`).
    pub(super) fn memory(&self) -> (vk::DeviceMemory, u64) {
        let pages = self.import.pages(ReadApart::Texture);
        let placement = &pages.memory.placement;
        (placement.memory, placement.offset + pages.offset)
    }

    /// Where the bytes start in their first imported page.
    pub(super) fn offset(&self) -> u64 {
        self.import.shape.offset
    }

    /// Whether they lie in fresh pages the runtime mapped and imported (`fresh_pages.rs`): a copy
    /// of the caller's pages that images in place ride on (`Device::copied_pages`).
    pub(super) fn over_own_pages(&self) -> bool {
        let pages = self.import.pages(ReadApart::Texture);
        pages.memory.placement.over_own_pages()
    }

    /// The bytes of their memory from where the bytes start to the placement's end: the room an
    /// image bound at their bytes has.
    pub(super) fn room(&self) -> u64 {
        let pages = self.import.pages(ReadApart::Texture);
        pages.memory.placement.size().saturating_sub(pages.offset)
    }

    /// The index of their memory's type.
    pub(super) fn type_index(&self) -> u32 {
        self.import.shape.type_index
    }

    /// Whether they were imported as dma-bufs (`dma_buf.rs`), not by their pointer.
    pub(super) fn over_dma_buf(&self) -> bool {
        self.import.shape.over_dma_buf
    }

    /// Whether they lie in a copy of the caller's pages (`copied.rs`), not an import of them.
    pub(super) fn copied(&self) -> bool {
        self.import.copied()
    }

    /// Whether they are bound in the device's import of a chunk of a guest RAM (`host_ram.rs`),
    /// with no import of their own.
    pub(super) fn in_ram_chunk(&self) -> bool {
        self.import.shape.in_ram_chunk
    }

    /// Where they lie as dma-bufs (`dma_buf.rs`): the imports, with the caller's source; the host
    /// address where their bytes start (from which an image over them reckons its texels'); and
    /// the imports' memory type. `None` for pages imported by their pointer. A read that keeps
    /// the import imported for good (`host_residency.rs`).
    #[cfg(unix)]
    pub(super) fn dma_buf(&self) -> Option<(Arc<crate::dma_buf::DmaBufPages>, u64, u32)> {
        let pages = self.import.pages(ReadApart::Texture);
        let placement = &pages.memory.placement;
        let imports = Arc::clone(placement.dma_buf_pages()?);
        let start = (placement.mapped?.as_ptr() as u64).checked_add(pages.offset)?;
        Some((imports, start, placement.type_index()))
    }

    /// As the buffer that lies in them is released (`BufferShared`'s drop, a no-copy heap's own
    /// buffer's included), its listing let go of: a live buffers' table that still lists them is a
    /// command buffer's, one that may reach them unnamed (its first submission's, a host step's,
    /// or one a root table recorded before its first submission holds, `live_buffers.rs`), and
    /// the pages stay until it lets go of it. Where the caller gave a keep-alive, that keeps the
    /// pages mapped until the import is freed, so the deallocator runs here, at Metal's moment
    /// ([`HostPages::run_kept`]), and the import keeps the keep-alive unless it keeps another's.
    /// Else the deallocator waits for the pages to go, where Metal's runs at the release:
    /// announced once per device (`Degradation::HostDeallocatorDelayed`).
    fn released_listed(&self, device: &Device) {
        let Some(release) = self.release.get() else {
            return;
        };
        if self.listed.load(Ordering::Acquire) == 0 {
            return;
        }
        // The exit guard runs no deallocator and drops no keep-alive: the pages' drop keeps them.
        let Some(busy) = exit_guard::enter() else {
            return;
        };
        drop(busy);
        if let Some(keep_alive) = release.keep_alive() {
            // These pages still lie in the import, and another buffer lying there may have let go
            // of its keep-alive here too: only a keep-alive the import keeps covers this one.
            self.run_kept(release, keep_alive, false);
            return;
        }
        if !release.deallocates() {
            return;
        }
        device.shared().counts.host_deallocator_delayed();
        let length = release.length();
        device.degrade_once(Degradation::HostDeallocatorDelayed, || {
            (
                format!("a no-copy buffer of {length} bytes"),
                "a command buffer that may reach its pages unnamed (a live buffers' table it \
                 holds lists them) holds them as it is released, and the caller gave no \
                 keep-alive"
                    .to_owned(),
            )
        });
    }

    /// Run `release`'s deallocator now, at Metal's moment, the caller's `keep_alive` keeping the
    /// pages mapped until the import is freed: the import keeps it, unless it keeps another's
    /// already, or `others` (another buffer lies in the import, whose own contract keeps the same
    /// bytes mapped until it is freed); then it is dropped after the deallocator, which may read
    /// the bytes.
    fn run_kept(&self, release: &Release, keep_alive: ProgramObject<KeepAlive>, others: bool) {
        let counts = &self.import.device.shared().counts;
        let covered = {
            let mut kept = lock(&self.import.kept);
            match others || kept.is_some() {
                true => Some(keep_alive),
                false => {
                    // Counted before the free can take it.
                    counts.host_keep_alive_kept();
                    *kept = Some(keep_alive);
                    None
                }
            }
        };
        release.run();
        if let Some(keep_alive) = covered {
            counts.host_keep_alive_covered();
            exit_guard::program_release(move || drop(keep_alive));
        }
    }

    /// Release the pages to `release` when they go.
    pub(super) fn release_to(&self, release: Release) {
        // A refused set leaves the deallocator unrun, as a refused call does: never reached,
        // since each pages' release is set once, as they become a buffer's.
        let _ = self.release.set(release);
    }

    /// Their import, which a no-copy buffer made over the same bytes shares.
    pub(super) fn import(&self) -> &Arc<HostImport> {
        &self.import
    }
}

/// The import of host pages one or more no-copy buffers lie in: a `VkBuffer` over them and the
/// memory imported from them. Every no-copy buffer made over the same bytes as a live one, of
/// its memory class, lies in its import (`host_aliases.rs`): one resource to the driver, as one
/// IOSurface's memory is to Metal. The last hold on it destroys the `VkBuffer` and frees the
/// import (in the release, or on a handler thread where a keep-alive alone waits for the free),
/// but for a releasable one's, which the command buffers that read them hold too
/// (`host_residency.rs`).
pub(super) struct HostImport {
    device: Device,
    /// Where the bytes lie in its `VkBuffer` and memory: the same in each import of them again.
    shape: Shape,
    /// Its `VkBuffer` and memory: taken as it goes, to be freed in the release or, where a
    /// keep-alive alone waits for the free, on a handler thread.
    objects: ManuallyDrop<Objects>,
    /// The [`HostPages`] lying in it.
    lying: AtomicUsize,
    /// The reads other than a blit's that took it, a bit each ([`HostImport::note_read_apart`]):
    /// counted in `WorkCounts::host_imports_read_apart` and the gauge of each reason while it
    /// lives.
    read_apart: AtomicU8,
    /// Why the device never releases it while idle, for an import it keeps: counted in that
    /// reason's `WorkCounts` gauge while it lives. `None` for a releasable one.
    kept_because: Option<KeptBecause>,
    /// A rebind's import that no live buffers' table listed yet ([`ListedPages`]): counted in
    /// `WorkCounts::host_imports_rebound_unlisted` until one does or it goes.
    unlisted: AtomicBool,
    /// The keep-alive of the pages that went while it lived with none other lying in it, dropped
    /// once it is freed: it keeps the bytes mapped for every buffer made over them, since all were
    /// made over the same bytes, so a later one's keep-alive is dropped as its pages go
    /// ([`HostPages`]).
    kept: Mutex<Option<ProgramObject<KeepAlive>>>,
    /// The releases of pages that went while it lived with no keep-alive: their deallocators
    /// run once it is freed.
    after_free: Mutex<Vec<Release>>,
    /// Its place among its device's copied host buffers, where its memory is a copy of the
    /// caller's pages, not their import (`copied.rs`): unregistered as it goes, before its memory.
    copied: OnceLock<crate::copied::Copied>,
    /// The logs of the writes to its pages, where their memory came with one
    /// (`dirty_pages::PagesLog`): every buffer lying in it shares them.
    log: OnceLock<Arc<PagesLog>>,
}

/// Where a host import's bytes lie in its `VkBuffer` and memory, which reading needs no
/// `VkBuffer`: the same in each import of them again (`host_residency.rs`).
#[derive(Clone, Copy)]
pub(super) struct Shape {
    /// Where the bytes start in the first imported page.
    pub(super) offset: u64,
    /// Where the buffer's byte 0 lies in its `VkBuffer`.
    pub(super) base: u64,
    /// The bytes from byte 0 a copy reads ([`Buffer::copy_reach`]).
    pub(super) reach: u64,
    /// The host address of the first imported byte.
    pub(super) mapped: Option<Mapped>,
    /// The index of the memory's type.
    pub(super) type_index: u32,
    /// The pages were imported as dma-bufs (`dma_buf.rs`), not by their pointer.
    pub(super) over_dma_buf: bool,
    /// The bytes are bound in the device's import of a chunk of a guest RAM (`host_ram.rs`), with
    /// no import of their own.
    pub(super) in_ram_chunk: bool,
    /// The pages were imported through a read-only view of them, which drops the GPU's writes
    /// (`read_only_view.rs`).
    pub(super) drops_writes: bool,
}

impl Shape {
    /// The shape of `pages`.
    pub(super) fn of(pages: &Pages) -> Shape {
        let placement = &pages.memory.placement;
        Shape {
            offset: pages.offset,
            base: pages.base,
            reach: pages.reach,
            mapped: placement.mapped,
            type_index: placement.type_index(),
            over_dma_buf: placement.over_dma_buf(),
            in_ram_chunk: placement.over_ram_chunk(),
            drops_writes: placement.over_read_only_view(),
        }
    }

    /// Whether `other` lays the bytes as this does: an import again of them as their first.
    pub(super) fn same(&self, other: &Shape) -> bool {
        let at = |shape: &Shape| shape.mapped.map(Mapped::as_ptr);
        (
            self.offset,
            self.base,
            self.reach,
            self.type_index,
            self.over_dma_buf,
            self.in_ram_chunk,
            self.drops_writes,
        ) == (
            other.offset,
            other.base,
            other.reach,
            other.type_index,
            other.over_dma_buf,
            other.in_ram_chunk,
            other.drops_writes,
        ) && at(self) == at(other)
    }
}

/// Why a read other than a blit's took a host import ([`HostImport::note_read_apart`]), each
/// counted once per import in its `WorkCounts::host_imports_read_apart_*` gauge while the import
/// lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadApart {
    /// The buffer's Metal address handed out: `gpu_address`, or written for the caller.
    Address,
    /// The buffer's `VkBuffer` handed out (`Buffer::vk_buffer`).
    VkBuffer,
    /// A command's read other than a blit's copy: a binding, `use_resource`, or a copy of a
    /// texture over the bytes.
    Binding,
    /// A texture over the bytes: an image bound to the memory, or a texel view over the
    /// `VkBuffer`.
    Texture,
    /// A second buffer lying in the import, or sharing its buffer's range.
    SecondBuffer,
    /// No thread to release it while idle (`host_residency.rs`).
    Unreleased,
}

impl ReadApart {
    /// How many reasons there are.
    pub(crate) const COUNT: usize = 6;
    const ALL: [ReadApart; ReadApart::COUNT] = [
        ReadApart::Address,
        ReadApart::VkBuffer,
        ReadApart::Binding,
        ReadApart::Texture,
        ReadApart::SecondBuffer,
        ReadApart::Unreleased,
    ];

    /// Its bit in [`HostImport::read_apart`].
    fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// Why the device keeps a host import imported whatever reads it, never releasing it while idle
/// (`host_residency.rs`), each counted in its `WorkCounts::host_imports_kept_*` gauge while the
/// import lives. Imports of dma-bufs kept as their descriptors did not fit count apart
/// (`WorkCounts::host_imports_kept_for_fds`), as pointer imports here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KeptBecause {
    /// A heap's own import.
    Heap,
    /// A rebind's import (`rebind.rs`).
    Rebind,
    /// An import by pointer on a device whose driver is not measured to follow the pointer
    /// (`DriverFact::PointerImportsFollowTheirAddress`), or off Unix.
    Pointer,
    /// No stand-in buffer for a read of a released one.
    NoStandIn,
}

impl KeptBecause {
    /// How many reasons there are.
    pub(crate) const COUNT: usize = 4;
}

/// A host import's `VkBuffer` and memory.
enum Objects {
    /// Imported until the import goes.
    Fixed(Imported),
    /// Released while idle, and imported again at the next read (`host_residency.rs`).
    #[cfg(unix)]
    Releasable(Residency),
}

/// A host import's `VkBuffer` and memory as one read takes them: borrowed from an import that
/// keeps them, or held, from one that may release them, for as long as the read lasts.
pub(super) enum ImportPages<'a> {
    Fixed(&'a Pages),
    #[cfg(unix)]
    Held(Arc<Imported>),
    /// The device's stand-in, which a read of a released import takes where the device cannot
    /// import it again ([`HostImport::pages`]).
    #[cfg(unix)]
    StandIn(Arc<FixedPages>),
}

impl std::ops::Deref for ImportPages<'_> {
    type Target = Pages;

    fn deref(&self) -> &Pages {
        match self {
            ImportPages::Fixed(pages) => pages,
            #[cfg(unix)]
            ImportPages::Held(imported) => &imported.pages,
            #[cfg(unix)]
            ImportPages::StandIn(stand_in) => &stand_in.pages,
        }
    }
}

impl HostImport {
    /// The import of `pages`, no pages lying in it yet, which the device keeps imported for
    /// `because` (`None`: an import of dma-bufs whose descriptors did not fit, counted as such).
    fn new(device: &Device, pages: Pages, because: Option<KeptBecause>) -> HostImport {
        if let Some(because) = because {
            device.shared().counts.host_import_kept(because, true);
        }
        let unlisted = because == Some(KeptBecause::Rebind);
        if unlisted {
            device.shared().counts.host_import_rebound_unlisted(true);
        }
        HostImport {
            device: device.clone(),
            shape: Shape::of(&pages),
            objects: ManuallyDrop::new(Objects::Fixed(Imported::new(device, pages))),
            lying: AtomicUsize::new(0),
            read_apart: AtomicU8::new(0),
            kept_because: because,
            unlisted: AtomicBool::new(unlisted),
            kept: Mutex::new(None),
            after_free: Mutex::new(Vec::new()),
            copied: OnceLock::new(),
            log: OnceLock::new(),
        }
    }

    /// The import of `pages`, no pages lying in it yet, which the device releases while idle and
    /// `again` imports again (`host_residency.rs`).
    #[cfg(unix)]
    fn releasable(
        device: &Device,
        pages: Pages,
        again: Again,
        stand_in: Arc<FixedPages>,
    ) -> HostImport {
        HostImport {
            device: device.clone(),
            shape: Shape::of(&pages),
            objects: ManuallyDrop::new(Objects::Releasable(Residency::new(
                device, pages, again, stand_in,
            ))),
            lying: AtomicUsize::new(0),
            read_apart: AtomicU8::new(0),
            kept_because: None,
            unlisted: AtomicBool::new(false),
            kept: Mutex::new(None),
            after_free: Mutex::new(Vec::new()),
            copied: OnceLock::new(),
            log: OnceLock::new(),
        }
    }

    /// A live buffers' table listed pages lying in it ([`ListedPages`]): a rebind's import is
    /// listed from now on (`WorkCounts::host_imports_rebound_unlisted`). A load once listed.
    fn note_listed(&self) {
        if self.unlisted.load(Ordering::Relaxed) && self.unlisted.swap(false, Ordering::Relaxed) {
            (self.device.shared().counts).host_import_rebound_unlisted(false);
        }
    }

    /// Its memory is a copy of the caller's pages, registered as `copied` (`copied.rs`), not
    /// their import.
    pub(super) fn set_copied(&self, copied: crate::copied::Copied) {
        // Set once, as the copied buffer is made.
        let _ = self.copied.set(copied);
    }

    /// Whether its memory is a copy of the caller's pages (`copied.rs`), not their import.
    pub(super) fn copied(&self) -> bool {
        self.copied.get().is_some()
    }

    /// Keep the logs of the writes to its pages (set once, as its first buffer is made), which
    /// its copy, where it is one, reports its merges back to (`copied.rs`).
    pub(super) fn keep_log(&self, log: Arc<PagesLog>) {
        if let Some(copied) = self.copied.get() {
            copied.set_log(&log);
        }
        self.device.shared().dirty.note_logged();
        let _ = self.log.set(log);
    }

    /// Whether its pages were imported through a read-only view of them, which drops the GPU's
    /// writes (`read_only_view.rs`).
    pub(super) fn drops_writes(&self) -> bool {
        self.shape.drops_writes
    }

    /// Its residency, where the device may release it while idle.
    #[cfg(unix)]
    pub(super) fn residency(&self) -> Option<&Residency> {
        match &*self.objects {
            Objects::Fixed(_) => None,
            Objects::Releasable(residency) => Some(residency),
        }
    }

    /// Its `VkBuffer` and memory for a read other than a blit's, for `reason`, which keeps a
    /// releasable import imported for good from then on (`host_residency.rs`). Released, it is imported again
    /// first; where the device cannot, the device is lost (its work no longer runs) and the read
    /// takes the device's stand-in.
    pub(super) fn pages(&self, reason: ReadApart) -> ImportPages<'_> {
        self.note_read_apart(reason);
        match &*self.objects {
            Objects::Fixed(imported) => ImportPages::Fixed(&imported.pages),
            #[cfg(unix)]
            Objects::Releasable(residency) => match residency.exposed() {
                Some(held) => ImportPages::Held(held),
                None => {
                    // No work that read it runs on a lost device.
                    self.device.mark_lost();
                    self.device.shared().counts.host_import_lost();
                    ImportPages::StandIn(residency.stand_in())
                }
            },
        }
    }

    /// Its `VkBuffer` and memory for a blit's command (`Buffer::blit_at`), imported again where
    /// released, with what the command buffer holds of them (`rebind::ImportHold`): nothing of an
    /// import that keeps them. Refused where the device cannot import them again.
    pub(super) fn blit_pages(&self) -> Result<(ImportPages<'_>, Option<ImportHold>), Error> {
        match &*self.objects {
            Objects::Fixed(imported) => Ok((ImportPages::Fixed(&imported.pages), None)),
            #[cfg(unix)]
            Objects::Releasable(residency) => {
                let held = residency.for_blit()?;
                let hold = ImportHold::Held(Arc::clone(&held));
                Ok((ImportPages::Held(held), Some(hold)))
            }
        }
    }

    /// Its `VkBuffer` and memory for a live-table listing (`live_buffers.rs`), which reads them
    /// without exposing them: `None` while a releasable import is released.
    fn listing(&self) -> Option<ImportPages<'_>> {
        match &*self.objects {
            Objects::Fixed(imported) => Some(ImportPages::Fixed(&imported.pages)),
            #[cfg(unix)]
            Objects::Releasable(residency) => residency.listing().map(ImportPages::Held),
        }
    }

    /// Keep it imported for good from now on, for `reason` (`host_residency.rs`): whether it was
    /// ever released, so a range of it noted before may be stale.
    pub(super) fn expose(&self, reason: ReadApart) -> bool {
        self.note_read_apart(reason);
        match &*self.objects {
            Objects::Fixed(_) => false,
            #[cfg(unix)]
            Objects::Releasable(residency) => residency.expose(),
        }
    }

    /// Note a read other than a blit's, for `reason` ([`HostImport::pages`],
    /// [`HostImport::expose`]): the reads after which the device's residency keeps a releasable
    /// import for good (`host_residency.rs`), noted for an import of any kind, counted once while
    /// it lives (`WorkCounts::host_imports_read_apart`), and once for each reason. A load once
    /// noted.
    fn note_read_apart(&self, reason: ReadApart) {
        let bit = reason.bit();
        if self.read_apart.load(Ordering::Relaxed) & bit != 0 {
            return;
        }
        let before = self.read_apart.fetch_or(bit, Ordering::Relaxed);
        if before & bit != 0 {
            return;
        }
        let counts = &self.device.shared().counts;
        if before == 0 {
            counts.host_import_read_apart(true);
        }
        counts.host_import_read_apart_for(reason, true);
    }
}

impl Device {
    /// On a device that keeps MoltenVK's residency set (`argument_buffers.rs`), a line to the
    /// device's message sink (`Verbose`, id name `metal2vulkan-host-import-freed`) for a host
    /// import of `bytes` freed: MoltenVK removes its buffer from the set and commits the set at
    /// once, with the submissions in flight on the device's queues it names. Elsewhere nothing,
    /// and nothing made where the sink hears no `Verbose` message
    /// (`MessageSink::with_least_severity`).
    pub(super) fn note_import_freed(&self, bytes: u64) {
        let severity = super::MessageSeverity::Verbose;
        let shared = self.shared();
        let Some(sink) = (shared.instance.sink()).filter(|sink| sink.hears(severity)) else {
            return;
        };
        let facts = super::device::argument_buffers::Facts::of(&self.properties().description);
        if !facts.keeps_residency_set() {
            return;
        }
        let in_flight: usize = shared
            .relayed_queues()
            .map(|(_, flights)| flights.in_flight())
            .sum();
        sink.deliver(&super::DebugMessage {
            severity,
            kind: super::MessageKind {
                general: true,
                ..super::MessageKind::default()
            },
            id_name: "metal2vulkan-host-import-freed".into(),
            id: 0,
            text: format!(
                "metal2vulkan: a host import of {bytes} bytes freed with {in_flight} submissions \
                 in flight on the device's queues: MoltenVK removes it from its device-wide \
                 residency set and commits the set"
            ),
            degradation: None,
        });
    }
}

impl Drop for HostImport {
    fn drop(&mut self) {
        // A copy leaves its device's comparisons before its memory can go (`copied.rs`).
        drop(self.copied.take());
        // The Vulkan teardown is a section of the exit guard.
        let Some(busy) = exit_guard::enter() else {
            // The process is exiting: the pages stay imported to its end (`exit_guard`), so no
            // deallocator runs and no keep-alive goes.
            for release in lock(&self.after_free).iter() {
                release.keep();
            }
            if let Some(kept) = lock(&self.kept).take() {
                std::mem::forget(kept);
            }
            // SAFETY: dropped once, here, and never read again; its free finds the exit too.
            unsafe { ManuallyDrop::drop(&mut self.objects) };
            return;
        };
        let shared = self.device.shared();
        let read_apart = self.read_apart.load(Ordering::Relaxed);
        if read_apart != 0 {
            shared.counts.host_import_read_apart(false);
        }
        for reason in ReadApart::ALL {
            if read_apart & reason.bit() != 0 {
                shared.counts.host_import_read_apart_for(reason, false);
            }
        }
        if let Some(because) = self.kept_because {
            shared.counts.host_import_kept(because, false);
        }
        if self.unlisted.load(Ordering::Relaxed) {
            shared.counts.host_import_rebound_unlisted(false);
        }
        drop(busy);
        // Once freed, the keep-alive goes, and the deallocators that waited for the free run.
        let kept = lock(&self.kept).take();
        if kept.is_some() {
            shared.counts.host_keep_alive_freed();
        }
        let releases: Vec<Release> = lock(&self.after_free).drain(..).collect();
        let delayed = releases
            .iter()
            .find(|release| release.deallocates())
            .map(Release::length);
        // The free goes off the releasing thread only where a keep-alive (a runtime extension
        // whose drop tells the caller the pages may go) alone waits for it. A deallocator waiting
        // for it runs in the release, as Metal runs it there (an M3 Max under macOS 27.0.1 ran an
        // unused buffer's deallocator inside its final release, on the releasing thread, every
        // time). With nothing waiting the caller may unmap the pages once the release returns, and
        // `VK_EXT_external_memory_host` requires the imported range to stay valid for the imported
        // memory's whole life (`VkImportMemoryHostPointerInfoEXT`), so the free is made here too.
        let posts =
            delayed.is_none() && (kept.is_some() || releases.iter().any(Release::keeps_alive));
        let left = AfterImport {
            after: Box::new(AfterFree { kept, releases }),
            delayed,
        };
        // SAFETY: taken once, here, as the import goes, and never read again.
        let objects = unsafe { ManuallyDrop::take(&mut self.objects) };
        let going: Box<dyn Send> = match objects {
            // No pending command buffer uses the pages: one whose commands used them holds them
            // until it completed (`rebind.rs`), one that may read a texture's image over them
            // holds the image, which holds the import (`heap_sets.rs`), and every buffer over them
            // let go of them.
            Objects::Fixed(imported) => {
                imported.leave(left);
                Box::new(imported)
            }
            // Its objects go with their last holder, here or the last command buffer that read
            // them (a blit's, which holds them and not the buffer once a rebind or detach
            // replaced its pages), and what waits for the free with them: a driver may list a
            // pointer import's memory in every submission, which fails where its pages were
            // unmapped.
            #[cfg(unix)]
            Objects::Releasable(residency) => match residency.let_go() {
                Some(held) => {
                    held.leave(left);
                    Box::new(held)
                }
                // Released: nothing to free.
                None => left.after,
            },
        };
        if posts {
            shared.counts.host_import_release_posted();
            // A tuple drops in order: the release, then its count.
            let posted = (going, PostedDone(self.device.clone()));
            self.device.let_go_elsewhere(Box::new(posted));
        } else {
            drop(going);
        }
    }
}

/// Counts a host import's release posted to a handler thread (a keep-alive alone waiting for its
/// free) run as it drops, after the release.
struct PostedDone(Device);

impl Drop for PostedDone {
    fn drop(&mut self) {
        self.0.shared().counts.host_import_release_done();
    }
}

/// A host import's `VkBuffer` and memory, which the last holder lets go of, destroying the
/// `VkBuffer` and freeing the memory: the import's own, and a releasable import's, held by the
/// import while it is imported and by each command buffer whose blit read them until it completed
/// (`EncoderCore::blit_at`, `host_residency.rs`).
pub(super) struct Imported {
    device: Device,
    pub(super) pages: Pages,
    /// What goes once its memory is freed, which the import left as it went ([`Imported::leave`]).
    after: Mutex<Option<AfterImport>>,
}

/// What a going import leaves with its objects ([`Imported::leave`]): the caller's keep-alive and
/// deallocators, which go once the memory is freed, and the length of a deallocator's pages,
/// announced where the free waits for the batches in flight.
pub(super) struct AfterImport {
    after: Box<dyn Send>,
    delayed: Option<u64>,
}

impl Imported {
    /// The objects of `pages`, imported now.
    pub(super) fn new(device: &Device, pages: Pages) -> Imported {
        Imported {
            device: device.clone(),
            pages,
            after: Mutex::new(None),
        }
    }

    /// Leave `left` to go once the memory is freed, the import going.
    fn leave(&self, left: AfterImport) {
        *lock(&self.after) = Some(left);
    }
}

impl Drop for Imported {
    fn drop(&mut self) {
        let left = (self.after.get_mut())
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        let (after, delayed) = match left {
            Some(AfterImport { after, delayed }) => (Some(after), delayed),
            None => (None, None),
        };
        // SAFETY: no pending command buffer uses them: an import's own go as the import goes
        // (`HostImport::drop`), a releasable import's as the last command buffer whose blit read
        // them completed, and no other read reached them (`host_residency.rs`).
        match unsafe { self.pages.release_import(&self.device, after) } {
            // Freed now: what waited goes here, or, for imports of the caller's dma-bufs, once
            // their last holder (these pages' placement, as it drops after this) frees them.
            Some(after) => self.pages.after_free(after),
            // Its memory waits for the batches in flight (`freed_memory.rs`), and what the import
            // left with it, past Metal's release.
            None => {
                if let Some(length) = delayed {
                    deallocator_delayed(&self.device, length);
                }
            }
        }
    }
}

/// Count and announce once that a deallocator of a no-copy buffer of `length` bytes waits for
/// the batches submitted before its release (`freed_memory.rs`): its import's memory, freed,
/// waits for them, and the deallocator with it.
fn deallocator_delayed(device: &Device, length: u64) {
    device.shared().counts.host_deallocator_delayed();
    device.degrade_once(Degradation::HostDeallocatorDelayed, || {
        (
            format!("a no-copy buffer of {length} bytes"),
            "its pages' import, freed, waits for the batches submitted before its release: the \
             device's driver fails pending work whose pipelines read device addresses where \
             such memory is freed, and the caller gave no keep-alive"
                .to_owned(),
        )
    });
}

/// What goes once a host import's memory is freed: the keep-alive of the pages that went while it
/// lived, and the deallocators that waited for the free, each panic caught. It waits with the
/// memory where that waits for the batches in flight (`freed_memory.rs`).
struct AfterFree {
    kept: Option<ProgramObject<KeepAlive>>,
    releases: Vec<Release>,
}

impl Drop for AfterFree {
    fn drop(&mut self) {
        let kept = self.kept.take();
        let Some(busy) = exit_guard::enter() else {
            // The process is exiting: the pages stay imported to its end (`exit_guard`), so no
            // deallocator runs and no keep-alive goes.
            for release in &self.releases {
                release.keep();
            }
            if let Some(kept) = kept {
                std::mem::forget(kept);
            }
            return;
        };
        drop(busy);
        if let Some(keep_alive) = kept {
            exit_guard::program_release(move || drop(keep_alive));
        }
        for release in &self.releases {
            release.run();
        }
    }
}

impl Drop for HostPages {
    fn drop(&mut self) {
        // The images and views a rebind or a detach parked go first: they may hold the import.
        lock(&self.parked).clear();
        let others = self.import.lying.fetch_sub(1, Ordering::AcqRel) > 1;
        let Some(release) = self.release.take() else {
            return;
        };
        // The exit guard runs no deallocator and drops no keep-alive.
        let Some(busy) = exit_guard::enter() else {
            // The process is exiting: the pages stay imported to its end (`exit_guard`).
            release.keep();
            return;
        };
        drop(busy);
        let import = &self.import;
        match release.keep_alive() {
            // The caller's keep-alive keeps the pages mapped until the import is freed: the
            // deallocator runs now, at Metal's moment. Where another buffer lies in the import,
            // or the import keeps another's keep-alive, that keeps the same bytes mapped until it
            // is freed: this one goes now too, after the deallocator, which may read the bytes. (A
            // buffer released while a table listed its pages ran this at its release,
            // `released_listed`, and left the import a keep-alive: none is left here.)
            Some(keep_alive) => self.run_kept(&release, keep_alive, others),
            // Another buffer lies in the import, whose contract keeps the pages mapped: the
            // deallocator runs now, as with an import of its own (`host_aliases.rs`).
            None if others => release.run(),
            // The deallocator runs once the import is freed: as these pages let go of it below,
            // where nothing else holds it; else later, announced.
            None => {
                if release.deallocates() && Arc::strong_count(import) > 1 {
                    import.device.shared().counts.host_deallocator_delayed();
                    let length = release.length();
                    import
                        .device
                        .degrade_once(Degradation::HostDeallocatorDelayed, || {
                            (
                                format!("a no-copy buffer of {length} bytes"),
                                "its pages' import is still held as it is released (a texture \
                                 over them may still be read through its resource ID by a \
                                 command buffer submitted before the texture's release), and \
                                 the caller gave no keep-alive"
                                    .to_owned(),
                            )
                        });
                }
                lock(&import.after_free).push(release);
            }
        }
    }
}

impl Pages {
    /// Destroy the `VkBuffer` of a host import's pages and free their memory, on `device`, once
    /// their sparse binding completed, and note it (`Device::note_import_freed`): an import's
    /// last hold lets go. Left to the process's end while it exits. `after` goes once the memory
    /// is freed: given back for the caller to let go of ([`after_free`](Self::after_free)),
    /// unless it waits with the memory for the batches in flight (`freed_memory.rs`).
    ///
    /// # Safety
    ///
    /// No pending command buffer uses the pages, and nothing else frees them.
    pub(super) unsafe fn release_import(
        &self,
        device: &Device,
        after: Option<Box<dyn Send>>,
    ) -> Option<Box<dyn Send>> {
        // The Vulkan teardown is a section of the exit guard.
        let Some(busy) = exit_guard::enter() else {
            return after;
        };
        let shared = device.shared();
        // A submission that used the pages waited for their sparse binding: waited for here where
        // none did.
        self.memory.placement.settle();
        // SAFETY: the caller's.
        unsafe { release::object(shared.vk(), self.raw) };
        let after = lock(&shared.memory).release_then(shared.vk(), &self.memory.placement, after);
        drop(busy);
        device.note_import_freed(self.memory.placement.size());
        after
    }

    /// Let `after` go once the pages' memory is freed, their import released
    /// ([`release_import`](Self::release_import)): at once, but where the memory is imports of
    /// the caller's dma-bufs (`dma_buf.rs`), which the placement holds and are freed as their
    /// last holder lets go of them, then.
    fn after_free(&self, after: Box<dyn Send>) {
        #[cfg(unix)]
        if let Some(imports) = self.memory.placement.dma_buf_pages() {
            imports.after_free(after);
            return;
        }
        drop(after);
    }
}

/// A buffer's bound memory: its placement, and its Vulkan device address where the device
/// enables `bufferDeviceAddress`.
pub(super) struct Bound {
    pub(super) placement: Placement,
    pub(super) device_address: Option<u64>,
}

impl OwnBytes {
    /// `f` of the pages the buffer lies in now: a detached buffer's stand-in (`Detached`). A
    /// read of a no-copy buffer's import other than a blit's, for `reason`, which keeps it
    /// imported for good (`HostImport::pages`).
    fn with<R>(&self, reason: ReadApart, f: impl FnOnce(&Pages) -> R) -> R {
        match &self.lie {
            Lie::Fixed(fixed) => f(&fixed.pages),
            Lie::Host(cell) => match cell.lying() {
                Lying::In(pages) => f(&pages.import.pages(reason)),
                Lying::Detached(detached) => f(&detached.pages),
            },
        }
    }

    /// [`with`](Self::with) while the buffer lies in pages: `None` while it is detached.
    fn with_attached<R>(&self, reason: ReadApart, f: impl FnOnce(&Pages) -> R) -> Option<R> {
        match &self.lie {
            Lie::Fixed(fixed) => Some(f(&fixed.pages)),
            Lie::Host(cell) => match cell.lying() {
                Lying::In(pages) => Some(f(&pages.import.pages(reason))),
                Lying::Detached(_) => None,
            },
        }
    }

    /// Where the bytes lie in the pages the buffer lies in now ([`Shape`]), read without their
    /// `VkBuffer`.
    fn shape(&self) -> Shape {
        match &self.lie {
            Lie::Fixed(fixed) => Shape::of(&fixed.pages),
            Lie::Host(cell) => match cell.lying() {
                Lying::In(pages) => pages.import.shape,
                Lying::Detached(detached) => Shape::of(&detached.pages),
            },
        }
    }
}

/// A `Managed` buffer's host copy on a device without unified memory (`managed.rs`): a
/// `VkBuffer` of the buffer's padded length in host-visible, coherent memory, cached where the
/// device has such ([`MemoryClass::Managed`]), which `contents` maps. The GPU uses the buffer's
/// own `VkBuffer`, in device-local memory: Metal's two copies of a managed resource on a device
/// without unified memory, which `did_modify_range` and `synchronize_resource` copy between.
/// Released as the last holder lets it go: its buffer, or a submission that uploads from it.
pub(super) struct Mirror {
    raw: vk::Buffer,
    placement: Placement,
    /// The device it releases its `VkBuffer` and memory into.
    device: Device,
}

impl Drop for Mirror {
    fn drop(&mut self) {
        // The Vulkan teardown is a section of the exit guard.
        let Some(busy) = exit_guard::enter() else {
            // The process is exiting: the host copy and its memory are left to its end.
            return;
        };
        // SAFETY: no pending command uses the host copy: its buffer is gone, and so is every
        // submission that uploads from it (each holds it), and a command buffer that copies into
        // it holds the buffer.
        unsafe { release_mirror(self.device.shared(), self) };
        drop(busy);
    }
}

/// The bytes of a heap a buffer placed in it is: its heap's buffer's from `offset`, a multiple of
/// [`PLACEMENT_ALIGNMENT`](super::heap::PLACEMENT_ALIGNMENT).
struct HeapBytes {
    heap: Heap,
    /// The heap's own buffer.
    bytes: Buffer,
    offset: u64,
    /// `makeAliasable` was called: an automatic heap's range is free for other resources.
    aliasable: AtomicBool,
}

/// Who owns a buffer's bytes.
pub(super) enum Owner {
    /// The runtime: the sub-allocator's memory.
    Runtime,
    /// The caller: host memory a no-copy buffer wraps ([`HostPages`]), released to its
    /// deallocator if it gave one.
    Caller,
    /// The caller: a `VkBuffer` and the memory it is bound to, which the caller made and keeps
    /// (`wrapped_buffer.rs`). The runtime destroys and frees neither.
    Vulkan,
}

/// Whose memory a buffer's bytes are, where they are the caller's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CallerMemory {
    /// Host memory a no-copy buffer imported (`host_memory.rs`).
    HostPages,
    /// The memory the caller's own `VkBuffer` is bound to (`wrapped_buffer.rs`).
    VkBuffer,
}

impl BufferShared {
    /// [`Buffer::memory_hold`].
    fn memory_hold(&self) -> Option<MemoryHold> {
        match &self.bytes {
            Bytes::Own(own) => match &own.lie {
                Lie::Fixed(fixed) => Some(MemoryHold::Fixed {
                    _pages: Arc::clone(fixed),
                }),
                Lie::Host(cell) => cell.current().map(|host| MemoryHold::Host {
                    _import: Arc::clone(host.import()),
                }),
            },
            Bytes::Heap(heap) => heap.bytes.memory_hold(),
        }
    }
}

impl Filled for BufferShared {
    fn memory(&self) -> Option<Box<dyn Any + Send + Sync>> {
        self.memory_hold()
            .map(|memory| Box::new(memory) as Box<dyn Any + Send + Sync>)
    }
}

impl Drop for BufferShared {
    fn drop(&mut self) {
        let own = match &mut self.bytes {
            Bytes::Own(own) => own,
            Bytes::Heap(bytes) => {
                // An automatic heap's range goes back to it unless `makeAliasable` returned it.
                bytes
                    .heap
                    .release(bytes.offset, self.length, &bytes.aliasable);
                let live = &self.device.shared().live_buffers;
                live.forget(self.address.0, self.object_id);
                return;
            }
        };
        let shared = self.device.shared();
        if let Lie::Host(_) = own.lie {
            shared.rebinds.forget(own.identity.key);
            shared.aliases.forget(own.identity.key);
        }
        // The Vulkan teardown is a section of the exit guard. A no-copy buffer's pages go with
        // the last hold on them (`HostPages`), which runs their deallocator.
        let Some(busy) = exit_guard::enter() else {
            // The process is exiting: the buffer and its memory are left to its end.
            return;
        };
        // The buffer leaves the device's live buffers; its pages go with the last hold on them
        // (`FixedPages`, `HostPages`).
        shared.live_buffers.forget(self.address.0, self.object_id);
        // The host copy goes with its last holder (`Mirror`'s drop).
        lock(&shared.addresses).free(own.reserved);
        drop(busy);
        // The range is free and no lock is held: the caller's token goes, so a creation at the
        // range from its drop on succeeds.
        own.range_token.release();
        if let Lie::Host(cell) = &own.lie {
            if let Lying::In(pages) = cell.lying() {
                pages.released_listed(&self.device);
            }
            // A detached buffer's deallocator runs at its release, as Metal's does.
            cell.release_held_back();
        }
    }
}

/// Release `mirror`, a buffer's host copy on `shared`'s device: its `VkBuffer` and memory.
///
/// # Safety
///
/// No pending command uses the host copy.
unsafe fn release_mirror(shared: &DeviceShared, mirror: &Mirror) {
    // SAFETY: the caller's.
    unsafe { release::object(shared.vk(), mirror.raw) };
    lock(&shared.memory).free(shared.vk(), &mirror.placement);
}

/// A buffer's parts once its `VkBuffer` is bound and its addresses held
/// ([`Device::buffer_of`]).
pub(super) struct BufferParts {
    pub(super) raw: vk::Buffer,
    /// The placement it is bound to.
    pub(super) memory: Placement,
    pub(super) offset: u64,
    pub(super) base: u64,
    pub(super) length: u64,
    /// The bytes from byte 0 a copy reads in the `VkBuffer` ([`Buffer::copy_reach`]): the
    /// length, or a no-copy buffer's bytes to the end of its `VkBuffer` in its pages.
    pub(super) reach: u64,
    pub(super) modes: Modes,
    pub(super) address: GpuAddress,
    pub(super) reserved: Held,
    pub(super) owner: Owner,
    /// A no-copy buffer's release to its caller's deallocator, where it gave one.
    pub(super) release: Option<Release>,
    /// The buffer is a heap's own (`heap.rs`): `Default` hazard tracking resolves to untracked.
    pub(super) of_heap: bool,
    /// A `Managed` buffer's host copy, on a device without unified memory.
    pub(super) mirror: Option<Arc<Mirror>>,
    /// What imports a no-copy buffer's dma-bufs again, where the device may release its import
    /// while idle (`host_residency.rs`).
    #[cfg(unix)]
    pub(super) again: Option<Again>,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("length", &self.0.length)
            .field("gpu_address", &self.0.address)
            .field("options", &self.0.options)
            .finish()
    }
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Buffer) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Buffer {}

impl std::hash::Hash for Buffer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Arc::as_ptr(&self.0), state);
    }
}

/// The three modes of a buffer's resource options, each checked to be one Metal defines.
#[derive(Clone, Copy, Debug)]
pub(super) struct Modes {
    pub(super) storage: StorageMode,
    pub(super) cache: CpuCacheMode,
    pub(super) hazard: HazardTrackingMode,
}

impl Modes {
    /// The modes of `options`; refused when one is not defined.
    pub(super) fn of(options: ResourceOptions) -> Result<Modes, Error> {
        Ok(Modes {
            storage: options.storage_mode()?,
            cache: options.cpu_cache_mode()?,
            hazard: options.hazard_tracking_mode()?,
        })
    }
}

/// The alignment every buffer the runtime places in its memory starts at: the largest
/// linear-image alignment measured (256 bytes on RADV, 128 on NVIDIA, 64 on lavapipe, 16 on
/// MoltenVK), so a texture over the buffer aliases its memory whenever the texture's own offset
/// allows. Imported host memory starts at a host page and needs nothing.
pub(super) const LINEAR_IMAGE_ALIGNMENT: u64 = 256;

/// A buffer's memory `requirements`, placed at a multiple of [`LINEAR_IMAGE_ALIGNMENT`] as well
/// (Vulkan's alignments are powers of two, so the larger one is a multiple of both), and, up to
/// half a pooled block, taking whole units of that alignment. So the next buffer placed in the
/// pool starts where the last ends: a buffer shorter than the alignment would leave the rest of
/// its unit free, a gap no buffer takes, and the pool's first-fit search (`range.rs`) walks every
/// such gap for each later placement, which made each new buffer cost time in proportion to the
/// buffers before it. A buffer past half a pooled block keeps its own size: it never lies in a
/// pooled block but takes a block of its own (`memory.rs`'s `place`; a pooled block is
/// [`BLOCK_BYTES`] on every device, whose `maxMemoryAllocationSize` Vulkan puts at 1 GiB or
/// more), where nothing follows it.
/// Rounded up to a heap's 256 KiB (`texture/heap.rs`'s `HEAP_BASE_ALIGNMENT`), the largest heap
/// would ask past the device's `maxMemoryAllocationSize`: 4 GiB on RADV, whose largest allocation
/// is 4 GiB less 4 bytes, and 41,747,218,432 bytes on MoltenVK on an M3 Max, whose largest is
/// 41,747,087,360.
pub(super) fn placed(requirements: vk::MemoryRequirements) -> vk::MemoryRequirements {
    let alignment = requirements.alignment.max(LINEAR_IMAGE_ALIGNMENT);
    let size = if requirements.size > BLOCK_BYTES / 2 {
        requirements.size
    } else {
        requirements.size.next_multiple_of(alignment)
    };
    vk::MemoryRequirements {
        alignment,
        size,
        ..requirements
    }
}

/// The usages every buffer is made with: whatever a Metal buffer can be bound or used as, and the
/// device-address usage where `device_address`. A buffer the runtime places in its memory takes
/// its address from its block's buffer of addresses, which has these usages
/// (`memory::allocate_addressed`), and is made with `usage(false)` itself.
pub(super) fn usage(device_address: bool) -> vk::BufferUsageFlags {
    use vk::BufferUsageFlags as U;
    let usage = U::TRANSFER_SRC
        | U::TRANSFER_DST
        | U::UNIFORM_BUFFER
        | U::STORAGE_BUFFER
        | U::UNIFORM_TEXEL_BUFFER
        | U::STORAGE_TEXEL_BUFFER
        | U::INDEX_BUFFER
        | U::VERTEX_BUFFER
        | U::INDIRECT_BUFFER;
    if device_address {
        usage | U::SHADER_DEVICE_ADDRESS
    } else {
        usage
    }
}

impl Device {
    /// `newBufferWithLength:options:`: a buffer of `length` bytes, all zero.
    ///
    /// Where it lies follows docs/API.md's "The runtime's memory model": on a device without
    /// unified memory, `Shared` storage of the default CPU cache mode lies in host-cached system
    /// memory, as on a Mac with a discrete GPU, which the device announces once
    /// (`metal2vulkan-degrade-shared-in-system-memory`), and of the write-combined CPU cache
    /// mode in device-local memory where the device maps such.
    ///
    /// Refused: a length of 0 or past the device's `max_buffer_length` (`Nil`, as plain Metal
    /// answers nil, its validation stopping it), `Memoryless` storage
    /// (`Invalid`).
    pub fn new_buffer(&self, length: u64, options: ResourceOptions) -> Result<Buffer, Error> {
        self.keep_room();
        let options = self.own_memory_options("new_buffer", options)?;
        let (buffer, class) =
            self.buffer_of_length("new_buffer", length, options, AddressPlacement::Any)?;
        self.degrade_system_memory(&buffer, class, || {
            format!("a `Shared` buffer of {length} bytes")
        });
        Ok(self.visible(buffer))
    }

    /// A buffer of `length` bytes, all zero, with `options`, that the runtime makes for its own
    /// use: refused as [`Device::new_buffer`] refuses it, and announcing nothing. It lies outside
    /// the Metal address space (`AddressPlacement::Outside`), so no program buffer's address
    /// depends on it.
    pub(super) fn own_buffer(
        &self,
        length: u64,
        options: ResourceOptions,
    ) -> Result<Buffer, Error> {
        Ok(self
            .buffer_of_length("new_buffer", length, options, AddressPlacement::Outside)?
            .0)
    }

    /// A `Shared` buffer of `length` bytes, all zero, of the write-combined CPU cache mode, that
    /// the runtime makes for its own use ([`Device::own_buffer`]): one the GPU alone uses past
    /// the host's zeroes, or that the host only writes (`MemoryClass::WriteCombined`).
    pub(super) fn write_combined_buffer(&self, length: u64) -> Result<Buffer, Error> {
        self.own_buffer(
            length,
            ResourceOptions::new(
                StorageMode::Shared,
                CpuCacheMode::WriteCombined,
                HazardTrackingMode::Default,
            ),
        )
    }

    /// A buffer of `length` bytes, all zero, with `options`, at `placement` in the address space,
    /// for `call`, and the memory class it took: refused as [`Device::new_buffer`] refuses it,
    /// and for a pin the address space cannot honour.
    pub(super) fn buffer_of_length(
        &self,
        call: &'static str,
        length: u64,
        options: ResourceOptions,
        placement: impl Into<Place>,
    ) -> Result<(Buffer, MemoryClass), Error> {
        let modes = Modes::of(options)?;
        let class = buffer_class(call, modes)?;
        if length == 0 || length > self.properties().max_buffer_length {
            return Err(nil(call, Problem::BufferLength { length }));
        }
        let buffer = self.zeroed_buffer(call, length, modes, class, false, placement)?;
        Ok((buffer, class))
    }

    /// Announce [`Degradation::SharedInSystemMemory`] once where `buffer`, which the program made
    /// (`what` names it) in memory of `class`, is `Shared` memory a device without unified memory
    /// placed in system memory (`MemoryClass::Shared`). A heap's buffer is its heap's.
    pub(super) fn degrade_system_memory(
        &self,
        buffer: &Buffer,
        class: MemoryClass,
        what: impl FnOnce() -> String,
    ) {
        if class != MemoryClass::Shared || self.properties().has_unified_memory {
            return;
        }
        if buffer.own_pages_device_local() != Some(false) {
            return;
        }
        self.degrade(Degradation::SharedInSystemMemory, || {
            (
                what(),
                format!(
                    "{} has no unified memory, and the default CPU cache mode lets the host \
                     read it, so it lies in host-cached system memory; `Private` or `Managed` \
                     storage, or the write-combined CPU cache mode, puts the GPU's bytes in \
                     device-local memory",
                    self.name()
                ),
            )
        });
    }

    /// A buffer of `length` bytes (above 0), all zero, in memory of `class` with `modes`, at
    /// `placement` in the address space: a heap's own buffer when `of_heap` (`heap.rs`). Its
    /// length is the caller's to check.
    pub(super) fn zeroed_buffer(
        &self,
        call: &'static str,
        length: u64,
        modes: Modes,
        class: MemoryClass,
        of_heap: bool,
        placement: impl Into<Place>,
    ) -> Result<Buffer, Error> {
        let (buffer, bytes) =
            self.placed_new_buffer(call, length, modes, class, of_heap, placement.into())?;
        // Fresh pages read zero as mapped (`fresh_pages.rs`): nothing is written.
        if bytes == NewBytes::Zero {
            return Ok(buffer);
        }
        // Dropping `buffer` on failure releases it.
        buffer.clear_first()?;
        Ok(buffer)
    }

    /// A `Private` buffer of `length` bytes that the runtime makes for its own use and whose
    /// bytes hold no value before the GPU writes them: made without a submission, so that one of
    /// the runtime's sections, a host step's replay of an emulated mesh draw among them, makes
    /// it with the exit guard's entry points alone (`exit_guard`). Refused as
    /// [`Device::new_buffer`] refuses its length.
    pub(super) fn scratch_buffer(&self, length: u64) -> Result<Buffer, Error> {
        self.unzeroed_buffer(length, HazardTrackingMode::Untracked)
    }

    /// [`scratch_buffer`](Device::scratch_buffer) of hazard tracking `tracking`: the tracker
    /// orders its uses as any buffer's of that mode.
    pub(super) fn unzeroed_buffer(
        &self,
        length: u64,
        tracking: HazardTrackingMode,
    ) -> Result<Buffer, Error> {
        let options =
            ResourceOptions::new(StorageMode::Private, CpuCacheMode::DefaultCache, tracking);
        Ok(self.unzeroed_buffer_with(length, options)?.0)
    }

    /// [`unzeroed_buffer`](Device::unzeroed_buffer) with `options`: a buffer of `length` bytes
    /// that the runtime makes for its own use, whose bytes hold no value before its owner writes
    /// them, in memory `options` asks for, and what its bytes hold as it is made.
    ///
    /// The upload arena's chunks are such a buffer (`upload.rs`): nothing reads a byte of a chunk
    /// outside a region it handed out, and it sets every byte of a region that the region's data
    /// does not cover to zero as it hands it out, so what a chunk would pay a creation-time clear
    /// for is paid by the region, for the region's bytes alone. A chunk made in fresh pages reads
    /// zero unwritten ([`NewBytes::Zero`], `fresh_pages.rs`), so a region of one writes nothing
    /// past its data: that is what the second half of the answer is for.
    pub(super) fn unzeroed_buffer_with(
        &self,
        length: u64,
        options: ResourceOptions,
    ) -> Result<(Buffer, NewBytes), Error> {
        const CALL: &str = "new_buffer";
        let modes = Modes::of(options)?;
        let class = buffer_class(CALL, modes)?;
        if length == 0 || length > self.properties().max_buffer_length {
            return Err(nil(CALL, Problem::BufferLength { length }));
        }
        self.placed_new_buffer(
            CALL,
            length,
            modes,
            class,
            false,
            AddressPlacement::Outside.into(),
        )
    }

    /// [`zeroed_buffer`](Device::zeroed_buffer)'s buffer before its bytes are set to zero, and
    /// what they hold: its memory's bytes as placed, or zero where it lies in fresh pages it
    /// imported (`fresh_pages.rs`). A buffer not of a heap is a parked one of the same making
    /// where the recycler holds one, and is noted as one it may park as it goes (`recycle.rs`).
    fn placed_new_buffer(
        &self,
        call: &'static str,
        length: u64,
        modes: Modes,
        class: MemoryClass,
        of_heap: bool,
        placement: Place,
    ) -> Result<(Buffer, NewBytes), Error> {
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let shared = self.shared();
        let raw_device = shared.vk();
        // Padded to 16 bytes, so a device fill clears every byte too.
        let size = length
            .checked_next_multiple_of(16)
            .ok_or_else(|| nil(call, Problem::BufferLength { length }))?;
        // A heap is never `Managed` (`heap.rs`).
        let mirror = if class == MemoryClass::Managed && !self.properties().has_unified_memory {
            Some(Arc::new(self.mirror(size)?))
        } else {
            None
        };
        let class = match mirror {
            Some(_) => MemoryClass::Private,
            None => class,
        };
        // Its device address is its block's (`place_buffer_with`).
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage(false)),
        );
        // A heap's own buffer lies where its textures can too, and is never parked. A long
        // host-visible one of the runtime's own lies in fresh pages where the device takes them.
        let wants_fresh =
            !of_heap && fresh_pages::takes_fresh_pages(size, class) && shared.host_import.is_ok();
        let making = (!of_heap).then_some(BufferMaking {
            size,
            class,
            fresh_pages: wants_fresh,
        });
        let mut taken = making.and_then(|making| self.recycled_buffer(&making));
        let fresh = match (&taken, wants_fresh) {
            (None, true) => self.fresh_page_buffer(size, class)?,
            _ => None,
        };
        // Where the device took no fresh pages, a parked buffer of its own memory, made so as the
        // buffer is made below.
        if let (None, None, Some(making)) = (&taken, &fresh, making.filter(|_| wants_fresh)) {
            let making = BufferMaking {
                fresh_pages: false,
                ..making
            };
            taken = self.recycled_buffer(&making);
        }
        let bytes = match fresh {
            Some(_) => NewBytes::Zero,
            None => NewBytes::AsPlaced,
        };
        // Its range of the address space is a new one, as a new buffer's; the buffer and its
        // memory go again where none is left.
        let place_address = |raw, memory: Placement| {
            let address = lock(&shared.addresses).place(placement, length);
            match address {
                Ok(address) => Ok((raw, (memory, address))),
                Err(error) => {
                    lock(&shared.recycled_buffers).forget(raw);
                    recycle::destroy(raw_device, &mut lock(&shared.memory), [(raw, memory)]);
                    Err(error)
                }
            }
        };
        let made = match (taken, fresh) {
            (Some((raw, memory)), _) => place_address(raw, memory),
            (None, Some((raw, memory))) => {
                if let Some(making) = making {
                    lock(&shared.recycled_buffers).made(raw, making);
                }
                place_address(raw, memory)
            }
            (None, None) => (|| {
                // SAFETY: the device is live and `info` valid.
                let raw = unsafe { crate::creation::buffer(raw_device, &info) }.map_err(vulkan)?;
                shared.counts.buffer_made(info.usage);
                let placed = if of_heap {
                    self.place_heap_buffer(raw, length, class, placement)
                } else {
                    self.place_buffer(raw, length, class, placement)
                };
                match placed {
                    Ok(placed) => {
                        // Not in fresh pages, whatever was asked: the device took none.
                        if let Some(making) = making {
                            let making = BufferMaking {
                                fresh_pages: false,
                                ..making
                            };
                            lock(&shared.recycled_buffers).made(raw, making);
                        }
                        Ok((raw, placed))
                    }
                    Err(error) => {
                        // SAFETY: the buffer is unused.
                        unsafe { release::object(raw_device, raw) };
                        Err(error)
                    }
                }
            })(),
        };
        // On failure the host copy, unused, goes with `mirror` (`Mirror`'s drop).
        let (raw, (placement, (address, reserved))) = made?;
        let buffer = self.buffer_of(BufferParts {
            raw,
            memory: placement,
            offset: 0,
            base: 0,
            length,
            reach: length,
            modes,
            address,
            reserved,
            owner: Owner::Runtime,
            release: None,
            of_heap,
            mirror,
            #[cfg(unix)]
            again: None,
        });
        Ok((buffer, bytes))
    }

    /// A parked buffer of `making` (`recycle.rs`), now live, where the recycler holds one.
    fn recycled_buffer(&self, making: &BufferMaking) -> Option<(vk::Buffer, Placement)> {
        let shared = self.shared();
        let (taken, gone) = lock(&shared.recycled_buffers).take(making, Instant::now());
        self.let_go_parked_buffers(gone);
        if taken.is_some() {
            shared.counts.churn.taken(Recycled::Buffer);
        }
        taken
    }

    /// Destroy the parked buffers the recycler let go of, `gone`, and free their memory.
    fn let_go_parked_buffers(&self, gone: recycle::Gone<vk::Buffer>) {
        if gone.is_empty() {
            return;
        }
        let shared = self.shared();
        shared.counts.churn.evicted(Recycled::Buffer, gone.len());
        recycle::destroy(shared.vk(), &mut lock(&shared.memory), gone);
    }

    /// A `Managed` buffer's host copy of `size` bytes (its padded length) on a device without
    /// unified memory ([`Mirror`]): copied to and from only.
    fn mirror(&self, size: u64) -> Result<Mirror, Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST),
        );
        // SAFETY: the device is live and `info` valid.
        let raw = unsafe { crate::creation::buffer(raw_device, &info) }.map_err(vulkan)?;
        // SAFETY: `raw` is a live buffer of this device.
        let requirements = unsafe { raw_device.get_buffer_memory_requirements(raw) };
        let allocated = crate::memory::allocate(
            &shared.memory,
            raw_device,
            &requirements,
            MemoryClass::Managed,
        );
        let placed = allocated.and_then(|placement| {
            // SAFETY: the placement meets the buffer's requirements and is not bound elsewhere.
            match unsafe {
                crate::creation::bind_buffer(raw_device, raw, placement.memory, placement.offset)
            } {
                Ok(()) => Ok(placement),
                Err(error) => {
                    lock(&shared.memory).free(raw_device, &placement);
                    Err(vulkan(error))
                }
            }
        });
        match placed {
            Ok(placement) if placement.mapped.is_some() => Ok(Mirror {
                raw,
                placement,
                device: self.clone(),
            }),
            Ok(placement) => {
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw_device, raw) };
                lock(&shared.memory).free(raw_device, &placement);
                Err(internal("a managed buffer's host copy is not mapped"))
            }
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw_device, raw) };
                Err(error)
            }
        }
    }

    /// A buffer of preprocess memory for executions of generated commands
    /// (`generated_execute.rs`), as `vkGetGeneratedCommandsMemoryRequirementsEXT` states it in
    /// `required` (above 0 bytes): `VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT` and device-address
    /// usage (`maintenance5`'s usage flags, which the device enabled with
    /// `VK_EXT_device_generated_commands`), in device-local memory of a type and at an alignment
    /// that meet both the buffer's requirements and `required`. Its contents are the driver's.
    pub(super) fn preprocess_buffer(
        &self,
        required: vk::MemoryRequirements,
    ) -> Result<Buffer, Error> {
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let modes = Modes::of(ResourceOptions::private())?;
        let shared = self.shared();
        let raw_device = shared.vk();
        let mut usage = vk::BufferUsageFlags2CreateInfoKHR::default()
            .usage(PREPROCESS_BUFFER | vk::BufferUsageFlags2KHR::SHADER_DEVICE_ADDRESS);
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .size(required.size)
                .push_next(&mut usage),
        );
        // SAFETY: the device is live and `info` and its chain valid.
        let raw = unsafe { crate::creation::buffer(raw_device, &info) }.map_err(vulkan)?;
        shared
            .counts
            .buffer_made(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        // SAFETY: `raw` is a live buffer of this device.
        let own = unsafe { raw_device.get_buffer_memory_requirements(raw) };
        let requirements = vk::MemoryRequirements {
            size: own.size,
            alignment: own.alignment.max(required.alignment),
            memory_type_bits: own.memory_type_bits & required.memory_type_bits,
        };
        // Its own address: a preprocess address is one of a buffer of preprocess usage.
        let made = self.place_buffer_with(
            raw,
            requirements,
            required.size,
            MemoryClass::Private,
            AddressPlacement::Outside.into(),
            false,
        );
        let (placement, (address, reserved)) = match made {
            Ok(made) => made,
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw_device, raw) };
                return Err(error);
            }
        };
        Ok(self.buffer_of(BufferParts {
            raw,
            memory: placement,
            offset: 0,
            base: 0,
            length: required.size,
            reach: required.size,
            modes,
            address,
            reserved,
            owner: Owner::Runtime,
            release: None,
            of_heap: false,
            mirror: None,
            #[cfg(unix)]
            again: None,
        }))
    }

    /// The buffer of `parts`. Dropping the result releases every part.
    pub(super) fn buffer_of(&self, parts: BufferParts) -> Buffer {
        let BufferParts {
            raw,
            memory,
            offset,
            base,
            length,
            reach,
            modes,
            address,
            reserved,
            owner,
            release,
            of_heap,
            mirror,
            #[cfg(unix)]
            again,
        } = parts;
        let pages = self.pages_of(raw, memory, offset, base, reach);
        let identity = Identity::new(modes.hazard, of_heap);
        let lie = match owner {
            // A no-copy buffer's pages are its until a rebind (`rebind.rs`).
            Owner::Caller => {
                // Kept imported where nothing imports it again: a heap's, an import of dma-bufs
                // whose descriptors did not fit (counted as such), or one by pointer.
                let kept = match (of_heap, pages.memory.placement.over_dma_buf()) {
                    (true, _) => Some(KeptBecause::Heap),
                    (false, true) => None,
                    (false, false) => Some(KeptBecause::Pointer),
                };
                #[cfg(unix)]
                let import = self.host_import(pages, again, kept);
                #[cfg(not(unix))]
                let import = Arc::new(HostImport::new(self, pages, kept));
                self.host_lie(identity.key, import, release)
            }
            Owner::Runtime | Owner::Vulkan => Lie::Fixed(Arc::new(FixedPages {
                device: self.clone(),
                pages,
                owned: matches!(owner, Owner::Runtime),
            })),
        };
        let own = OwnBytes {
            lie,
            reserved,
            identity,
            owner,
            mirror,
            range_token: KeptToken::default(),
        };
        self.buffer_lying(own, length, modes, address)
    }

    /// The import of a no-copy buffer's `pages`: one the device releases while idle where
    /// `again` imports them again and the device keeps a stand-in for it and a thread to release
    /// it (`host_residency.rs`), else one it keeps, for `kept` where nothing imports it again.
    #[cfg(unix)]
    fn host_import(
        &self,
        pages: Pages,
        again: Option<Again>,
        kept: Option<KeptBecause>,
    ) -> Arc<HostImport> {
        let Some(again) = again else {
            return Arc::new(HostImport::new(self, pages, kept));
        };
        let stand_in = self.detach_stand_in().ok();
        let Some(stand_in) = stand_in.as_ref().and_then(Buffer::fixed_pages) else {
            let kept = Some(KeptBecause::NoStandIn);
            return Arc::new(HostImport::new(self, pages, kept));
        };
        let import = Arc::new(HostImport::releasable(self, pages, again, stand_in));
        if !self.shared().residency.note(&import) {
            // No thread releases it: kept imported.
            import.expose(ReadApart::Unreleased);
        }
        import
    }

    /// A no-copy buffer of `length` bytes lying in `import`, the import of the same bytes a live
    /// no-copy buffer lies in (`host_aliases.rs`), at `address`, released to `release` once
    /// nothing holds it. Dropping the result releases every part but the import, which the last
    /// buffer over it releases.
    pub(super) fn buffer_in_import(
        &self,
        import: Arc<HostImport>,
        length: u64,
        modes: Modes,
        (address, reserved): (GpuAddress, Held),
        release: Option<Release>,
    ) -> Buffer {
        let identity = Identity::new(modes.hazard, false);
        let own = OwnBytes {
            lie: self.host_lie(identity.key, import, release),
            reserved,
            identity,
            owner: Owner::Caller,
            mirror: None,
            range_token: KeptToken::default(),
        };
        self.buffer_lying(own, length, modes, address)
    }

    /// A no-copy buffer's lie, of `key`: pages of its own in `import`, released to `release`
    /// once nothing holds them, which a rebind replaces (`rebind.rs`).
    fn host_lie(&self, key: ResourceKey, import: Arc<HostImport>, release: Option<Release>) -> Lie {
        let drops_writes = import.drops_writes();
        let host = HostPages::new(import);
        if let Some(release) = release {
            host.release_to(release);
        }
        let cell = Arc::new(HostCell {
            pages: Mutex::new(Lying::In(Arc::new(host))),
            texels: Mutex::new(Vec::new()),
            textures: Mutex::new(TexturesOver::default()),
            in_place: Mutex::new(Vec::new()),
            exposed: AtomicU8::new(0),
            device: self.clone(),
            drops_writes,
            held_back: Mutex::new(None),
        });
        self.shared().rebinds.register(key, &cell);
        Lie::Host(cell)
    }

    /// The buffer of `own`'s bytes, `length` long, of `modes`, at `address`.
    fn buffer_lying(
        &self,
        own: OwnBytes,
        length: u64,
        modes: Modes,
        address: GpuAddress,
    ) -> Buffer {
        let options = ResourceOptions::new(modes.storage, modes.cache, own.identity.mode());
        Buffer(Arc::new(BufferShared {
            device: self.clone(),
            object_id: ObjectId::new(),
            length,
            options,
            address,
            attributes: ResourceAttributes::new(),
            bytes: Bytes::Own(own),
            held_by: HeldBy::default(),
        }))
    }

    /// The pages of a buffer `raw` in `memory`, its bytes `offset` bytes into its placement and
    /// `base` bytes into `raw`, `reach` bytes from byte 0 copied out.
    pub(super) fn pages_of(
        &self,
        raw: vk::Buffer,
        placement: Placement,
        offset: u64,
        base: u64,
        reach: u64,
    ) -> Pages {
        let memory = Bound {
            // A placed buffer's address is its block's (`place_buffer_with`); any other's its own.
            device_address: match placement.address {
                Some(address) => Some(address),
                // SAFETY: the buffer is bound to the placement, and was made with `usage`
                // (`device_address_of`).
                None => unsafe { self.device_address_of(raw) },
            },
            placement,
        };
        Pages {
            raw,
            memory,
            offset,
            base,
            reach,
        }
    }

    /// Host pages a rebind gives a no-copy buffer (`rebind.rs`): `raw` in `memory`, as
    /// [`Device::pages_of`] says, without their release, which the rebind sets once they are
    /// the buffer's.
    pub(super) fn host_pages(
        &self,
        raw: vk::Buffer,
        memory: Placement,
        offset: u64,
        base: u64,
        reach: u64,
    ) -> Arc<HostPages> {
        let pages = self.pages_of(raw, memory, offset, base, reach);
        let import = HostImport::new(self, pages, Some(KeptBecause::Rebind));
        Arc::new(HostPages::new(Arc::new(import)))
    }

    /// Host pages a rebind gives a no-copy buffer lying in `import`, the import of the same bytes
    /// a live no-copy buffer was made over (`rebind.rs`, `host_aliases.rs`), without their
    /// release, as [`Device::host_pages`] gives them.
    pub(super) fn pages_in_import(&self, import: Arc<HostImport>) -> Arc<HostPages> {
        Arc::new(HostPages::new(import))
    }

    /// The Vulkan device address of `raw`, where the device enables `bufferDeviceAddress`.
    ///
    /// # Safety
    ///
    /// `raw` is a buffer of this device made with [`usage`], bound to memory.
    pub(super) unsafe fn device_address_of(&self, raw: vk::Buffer) -> Option<u64> {
        let shared = self.shared();
        shared.device_address.then(|| {
            let info = vk::BufferDeviceAddressInfo::default().buffer(raw);
            // SAFETY: the buffer was made with `SHADER_DEVICE_ADDRESS` usage (the caller's
            // contract) and is bound to memory allocated with the device-address flag.
            unsafe { shared.vk().get_buffer_device_address(&info) }
        })
    }

    /// Place `raw` (for a buffer of `length` bytes) in memory of `class` and at `at` in the
    /// address space; released again when either fails.
    pub(super) fn place_buffer(
        &self,
        raw: vk::Buffer,
        length: u64,
        class: MemoryClass,
        at: Place,
    ) -> Result<(Placement, (GpuAddress, Held)), Error> {
        // SAFETY: `raw` is a live buffer of this device.
        let requirements = unsafe { self.shared().vk().get_buffer_memory_requirements(raw) };
        self.place_buffer_with(raw, requirements, length, class, at, true)
    }

    /// [`place_buffer`](Device::place_buffer) for a heap's own buffer: where the heap's textures
    /// can lie too (`Device::heap_requirements`).
    pub(super) fn place_heap_buffer(
        &self,
        raw: vk::Buffer,
        length: u64,
        class: MemoryClass,
        at: Place,
    ) -> Result<(Placement, (GpuAddress, Held)), Error> {
        // SAFETY: `raw` is a live buffer of this device.
        let own = unsafe { self.shared().vk().get_buffer_memory_requirements(raw) };
        let requirements = self.heap_requirements(own, class)?;
        self.place_buffer_with(raw, requirements, length, class, at, true)
    }

    /// [`place_buffer`](Device::place_buffer) in memory that meets `requirements`, which hold
    /// the buffer's own. Where `block_address`, `raw` was made without the device-address usage
    /// and its device address is its block's (`memory::allocate_addressed`, the placement's
    /// `address`); else its own, where it has the usage.
    fn place_buffer_with(
        &self,
        raw: vk::Buffer,
        requirements: vk::MemoryRequirements,
        length: u64,
        class: MemoryClass,
        at: Place,
        block_address: bool,
    ) -> Result<(Placement, (GpuAddress, Held)), Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let requirements = placed(requirements);
        let placement = if block_address {
            crate::memory::allocate_addressed(
                &shared.memory,
                raw_device,
                &requirements,
                class,
                &shared.counts,
            )?
        } else {
            crate::memory::allocate(&shared.memory, raw_device, &requirements, class)?
        };
        // SAFETY: the placement meets the buffer's requirements and is not bound elsewhere.
        let bound = unsafe {
            crate::creation::bind_buffer(raw_device, raw, placement.memory, placement.offset)
        };
        let address = match bound {
            Ok(()) => lock(&shared.addresses).place(at, length),
            Err(error) => {
                lock(&shared.memory).free(raw_device, &placement);
                return Err(vulkan(error));
            }
        };
        match address {
            Ok(address) => Ok((placement, address)),
            Err(error) => {
                lock(&shared.memory).free(raw_device, &placement);
                Err(error)
            }
        }
    }
}

/// What a new buffer's bytes hold as [`Device::placed_new_buffer`] makes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NewBytes {
    /// Its memory's bytes as placed: undefined, or a parked buffer's last owner's.
    AsPlaced,
    /// Zero: fresh pages it imported (`fresh_pages.rs`).
    Zero,
}

/// The memory class of a buffer of `modes`' storage and CPU cache modes, for `call`: refused for
/// `Memoryless`, which is no buffer's. `Shared` storage is `MemoryClass::WriteCombined` in the
/// write-combined CPU cache mode, which the host does not read.
pub(super) fn buffer_class(call: &'static str, modes: Modes) -> Result<MemoryClass, Error> {
    match modes.storage {
        StorageMode::Shared => Ok(match modes.cache {
            CpuCacheMode::DefaultCache => MemoryClass::Shared,
            CpuCacheMode::WriteCombined => MemoryClass::WriteCombined,
        }),
        StorageMode::Private => Ok(MemoryClass::Private),
        StorageMode::Managed => Ok(MemoryClass::Managed),
        StorageMode::Memoryless => Err(invalid(call, Problem::StorageMode)),
    }
}

impl Heap {
    /// A buffer of `length` bytes with `options` (checked against the heap's), placed `offset`
    /// bytes into the heap: its bytes are the heap's from there, as they are.
    pub(super) fn placed_buffer(
        &self,
        heap_bytes: &Buffer,
        offset: u64,
        length: u64,
        options: ResourceOptions,
    ) -> Buffer {
        Buffer(Arc::new(BufferShared {
            device: heap_bytes.device().clone(),
            object_id: ObjectId::new(),
            length,
            options,
            address: GpuAddress(heap_bytes.gpu_address().0 + offset),
            attributes: ResourceAttributes::new(),
            bytes: Bytes::Heap(HeapBytes {
                heap: self.clone(),
                bytes: heap_bytes.clone(),
                offset,
                // A placement heap's buffer answers aliasable from the start, as Metal's does.
                aliasable: AtomicBool::new(self.heap_type() == HeapType::Placement),
            }),
            held_by: HeldBy::default(),
        }))
    }
}

impl Buffer {
    /// Whether the buffer lies in fresh pages it imported (`fresh_pages.rs`), to which no image
    /// is bound: false for a no-copy buffer, a heap's and one over the caller's `VkBuffer`.
    pub(super) fn in_fresh_pages(&self) -> bool {
        match &self.0.bytes {
            Bytes::Own(OwnBytes {
                lie: Lie::Fixed(fixed),
                ..
            }) => fixed.pages.memory.placement.over_own_pages(),
            Bytes::Own(_) | Bytes::Heap(_) => false,
        }
    }

    /// Whether the buffer's own pages, of the runtime's memory, lie in device-local memory; `None`
    /// for a heap's buffer, whose bytes are its heap's. A buffer's pages are its for good.
    pub(super) fn own_pages_device_local(&self) -> Option<bool> {
        match &self.0.bytes {
            Bytes::Own(own) => own.with(ReadApart::Binding, |pages| {
                let memory = lock(&self.0.device.shared().memory);
                Some(memory.is_device_local(&pages.memory.placement))
            }),
            Bytes::Heap(_) => None,
        }
    }

    /// The length padded to 16 bytes: the bytes the `VkBuffer` holds from [`Buffer::base`].
    pub(super) fn padded_length(&self) -> u64 {
        self.0.length.next_multiple_of(16)
    }

    /// The bytes of the buffer's memory from its byte 0 ([`Buffer::memory`]'s offset) to the end
    /// of the memory's placement: an import's, for a buffer over host pages (its last page's bytes
    /// past the buffer's among them), or a heap's.
    pub(super) fn memory_room(&self) -> u64 {
        match &self.0.bytes {
            Bytes::Own(own) => own.with(ReadApart::Texture, |pages| {
                pages.memory.placement.size().saturating_sub(pages.offset)
            }),
            Bytes::Heap(bytes) => bytes.bytes.memory_room().saturating_sub(bytes.offset),
        }
    }

    /// Set the buffer's padded length of bytes to zero: through the mapping for `Shared` and
    /// `Managed` storage (host-coherent memory), else by a device fill the call waits for; both
    /// for a `Managed` buffer with a host copy ([`Mirror`]). Racing the GPU's use of the buffer
    /// is the caller's to avoid.
    pub(super) fn clear(&self) -> Result<(), Error> {
        let size = self.padded_length();
        if let Some(contents) = self.contents() {
            // SAFETY: the buffer's bytes, its padded length from `contents`, are host-coherent
            // memory (`MemoryClass::Shared` and `Managed`, or imported host memory), so the
            // device sees the writes at the next submission.
            unsafe { std::ptr::write_bytes(contents.as_ptr(), 0, size as usize) };
            if self.mirror().is_none() {
                return Ok(());
            }
        }
        let (raw, start) = self.at(0);
        self.0.device.submit_internal(|device, command_buffer| {
            // SAFETY: the internal command buffer is recording; the buffer is live, and its
            // padded length (a multiple of 16) lies in the `VkBuffer` from `start`, a multiple
            // of 4: 0, or a heap buffer's offset in its heap's (a buffer without contents is
            // none over host memory).
            unsafe {
                crate::word_fill::record(
                    &self.0.device,
                    command_buffer,
                    raw,
                    start..start + size,
                    0,
                );
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
                device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
        })
    }

    /// Set a new buffer's padded length of bytes to zero as [`clear`](Self::clear) does, without
    /// waiting for the GPU: the host's bytes (`Shared` and `Managed` storage, a `Managed`
    /// buffer's host copy) at once, so `contents` reads zero before any submission, and bytes the
    /// host cannot write (`Private` storage, a `Managed` buffer's device copy) by the device's
    /// next submission, before any of its commands (`texture/first_move.rs`).
    fn clear_first(&self) -> Result<(), Error> {
        let size = self.padded_length();
        if let Some(contents) = self.contents() {
            // SAFETY: as in `clear`: the buffer's host-coherent bytes, which nothing else uses
            // yet.
            unsafe { std::ptr::write_bytes(contents.as_ptr(), 0, size as usize) };
            self.0.device.shared().counts.new_buffer_cleared(size);
            if self.mirror().is_none() {
                return Ok(());
            }
        }
        let (raw, start) = self.at(0);
        let device = &self.0.device;
        let weak: Weak<BufferShared> = Arc::downgrade(&self.0);
        let keeper: Weak<dyn Filled> = weak;
        (device.shared().textures.first_moves).register_fill(
            device,
            keeper,
            raw,
            start..start + size,
        )
    }

    /// Give a new buffer whose bytes the host cannot write (`Private` storage) the first `length`
    /// bytes `staging` holds, without waiting for the GPU: copied by the device's next
    /// submission before any of its commands, after the fill with zeros it took as it was made
    /// (`texture/first_move.rs`); that submission holds `staging` until it completed. Past the
    /// bound of what is staged, a submission waited for makes it now.
    pub(super) fn fill_first_from(&self, staging: Buffer, length: u64) -> Result<(), Error> {
        let device = &self.0.device;
        let weak: Weak<BufferShared> = Arc::downgrade(&self.0);
        let keeper: Weak<dyn Filled> = weak;
        let source = staging.at(0);
        let past = (device.shared().textures.first_moves).register_bytes(
            keeper,
            self.at(0),
            source,
            length,
            Box::new(staging),
        );
        match past {
            true => device.submit_internal(|_, _| {}),
            false => Ok(()),
        }
    }

    /// The bytes from byte 0 a copy out of the buffer reads: its length, but for a no-copy
    /// buffer the rest of the host pages its range touches, as far as its `VkBuffer` holds them
    /// (all of them where the device's buffer of them fits the pages, `host_memory.rs`). Plain
    /// Metal reads the caller's bytes there, current ones, from the pointer to its pages' end
    /// (an M4 under macOS 27.0.1). Past them, and past another buffer's length, its API validation
    /// stops the copy, and plain Metal reads zeros, bytes written past the length, or another
    /// resource's: refused (`Invalid`).
    pub(super) fn copy_reach(&self) -> u64 {
        match &self.0.bytes {
            Bytes::Own(own) => own.shape().reach,
            Bytes::Heap(_) => self.0.length,
        }
    }

    /// The `VkBuffer`, and where the buffer's byte `offset` lies in it. Every Vulkan use of a
    /// buffer takes its offsets from here, since a buffer's byte 0 may lie past the `VkBuffer`'s:
    /// a no-copy buffer's ([`Buffer::base`]), and a heap buffer's, which is its heap's.
    pub(super) fn at(&self, offset: u64) -> (vk::Buffer, u64) {
        self.at_for(offset, ReadApart::Binding)
    }

    /// [`Buffer::at`] for `reason`, which a no-copy buffer's import counts its read apart for
    /// (`HostImport::note_read_apart`).
    pub(super) fn at_for(&self, offset: u64, reason: ReadApart) -> (vk::Buffer, u64) {
        match &self.0.bytes {
            Bytes::Own(own) => own.with(reason, |pages| (pages.raw, pages.base + offset)),
            Bytes::Heap(bytes) => bytes.bytes.at_for(bytes.offset + offset, reason),
        }
    }

    /// [`Buffer::at`] while the buffer's bytes lie in pages, read with them at once: `None` while
    /// they are detached (`detach.rs`), when its `VkBuffer` is the device's stand-in, which holds
    /// none of its bytes.
    pub(super) fn attached_at(&self, offset: u64) -> Option<(vk::Buffer, u64)> {
        match &self.0.bytes {
            Bytes::Own(own) => {
                own.with_attached(ReadApart::Binding, |pages| (pages.raw, pages.base + offset))
            }
            Bytes::Heap(bytes) => bytes.bytes.attached_at(bytes.offset + offset),
        }
    }

    /// Where the buffer's byte 0 lies in its `VkBuffer` ([`Buffer::at`]'s offset of byte 0), read
    /// without the `VkBuffer`: a no-copy buffer's [`Buffer::base`], a heap buffer's offset in its
    /// heap's.
    pub(super) fn start(&self) -> u64 {
        match &self.0.bytes {
            Bytes::Own(own) => own.shape().base,
            Bytes::Heap(bytes) => bytes.bytes.start() + bytes.offset,
        }
    }

    /// The `VkBuffer`, and where the space the bind layer states the buffer in starts in it:
    /// 0 for a buffer of its own, a heap buffer's offset in its heap's. The bind layer's offsets
    /// (`ArgumentTable::stage_table`) count from there, [`Buffer::base`] bytes before byte 0.
    /// A multiple of 256 bytes, so every alignment Vulkan asks of an offset holds of it.
    pub(super) fn space(&self) -> (vk::Buffer, u64) {
        // `at(0)` less `base()`, from one look at where the buffer lies.
        let space = |pages: &Pages, base: u64| (pages.raw, pages.base - base);
        match &self.0.bytes {
            Bytes::Own(own) => match &own.lie {
                Lie::Fixed(fixed) => space(&fixed.pages, Shape::of(&fixed.pages).base),
                Lie::Host(cell) => match cell.lying() {
                    Lying::In(pages) => space(
                        &pages.import.pages(ReadApart::Binding),
                        pages.import.shape.base,
                    ),
                    Lying::Detached(detached) => {
                        space(&detached.pages, Shape::of(&detached.pages).base)
                    }
                },
            },
            Bytes::Heap(bytes) => bytes.bytes.at_for(bytes.offset, ReadApart::Binding),
        }
    }

    /// Where the buffer's byte 0 lies in its bind layer space ([`Buffer::space`]): 0, but for a
    /// no-copy buffer whose pointer misses the buffer's memory alignment (bound that many bytes
    /// below it). An alignment Vulkan asks of an offset holds of `base` plus Metal's offset.
    pub(super) fn base(&self) -> u64 {
        match &self.0.bytes {
            Bytes::Own(own) => own.shape().base,
            Bytes::Heap(_) => 0,
        }
    }

    /// [`base`](Self::base) and [`device_address`](Self::device_address) together, from one look
    /// at where the buffer lies, a no-copy buffer's pages read for `reason`
    /// (`HostImport::note_read_apart`): what a binding of it reads (`argument_table::bound_buffer`),
    /// each as its own call reads it.
    pub(super) fn base_and_device_address(&self, reason: ReadApart) -> (u64, Option<u64>) {
        let address = |pages: &Pages| Some(pages.memory.device_address? + pages.base);
        match &self.0.bytes {
            Bytes::Own(own) => match &own.lie {
                Lie::Fixed(fixed) => (Shape::of(&fixed.pages).base, address(&fixed.pages)),
                Lie::Host(cell) => match cell.lying() {
                    Lying::In(pages) => {
                        let read = pages.import.pages(reason);
                        (pages.import.shape.base, address(&read))
                    }
                    Lying::Detached(detached) => {
                        (Shape::of(&detached.pages).base, address(&detached.pages))
                    }
                },
            },
            Bytes::Heap(bytes) => (
                0,
                (bytes.bytes.device_address()).map(|address| address + bytes.offset),
            ),
        }
    }

    /// The memory the buffer's bytes lie in: the `VkDeviceMemory`, the offset of the buffer's
    /// first byte in it, and its memory type's index. A texture over the buffer aliases it there
    /// (`texture.rs`).
    pub(super) fn memory(&self) -> (vk::DeviceMemory, u64, u32) {
        match &self.0.bytes {
            Bytes::Own(own) => own.with(ReadApart::Texture, |pages| {
                let placement = &pages.memory.placement;
                (
                    placement.memory,
                    placement.offset + pages.offset,
                    placement.type_index(),
                )
            }),
            Bytes::Heap(bytes) => {
                let (memory, offset, type_index) = bytes.bytes.memory();
                (memory, offset + bytes.offset, type_index)
            }
        }
    }

    /// A `Managed` buffer's host copy, on a device without unified memory ([`Mirror`]).
    fn mirror(&self) -> Option<&Arc<Mirror>> {
        match &self.0.bytes {
            Bytes::Own(own) => own.mirror.as_ref(),
            Bytes::Heap(_) => None,
        }
    }

    /// The `VkBuffer` of the buffer's host copy, whose byte 0 is the buffer's, where it has one
    /// ([`Mirror`]): a blit encoder's `synchronize_resource` copies into it.
    pub(super) fn host_copy(&self) -> Option<vk::Buffer> {
        self.mirror().map(|mirror| mirror.raw)
    }

    /// The CPU address of the bytes the GPU uses: [`Buffer::contents`], but `None` for a buffer
    /// with a host copy ([`Mirror`]), whose `contents` are that copy. A reader of the GPU's
    /// writes takes the bytes from here, else through a device copy.
    pub(super) fn device_contents(&self) -> Option<NonNull<u8>> {
        match self.mirror() {
            Some(_) => None,
            None => self.contents(),
        }
    }

    /// Copy the bytes `range` of a buffer with a host copy ([`Mirror`]) from that copy to the
    /// buffer's own, by the device's next submission, before any of its commands, after the
    /// commands before it on its `VkQueue` (`texture/first_move.rs`), which the call does not wait
    /// for: every command buffer committed after the call reads the bytes. The copy reads the host
    /// copy as that submission runs, and the submission holds the host copy until it completed.
    /// Past the bound of what is staged, a submission waited for makes it now. Nothing for a
    /// buffer without a host copy, nor for an empty range.
    pub(super) fn upload(&self, range: Range<u64>) -> Result<(), Error> {
        let Some(mirror) = self.mirror() else {
            return Ok(());
        };
        if range.is_empty() {
            return Ok(());
        }
        let device = &self.0.device;
        let weak: Weak<BufferShared> = Arc::downgrade(&self.0);
        let keeper: Weak<dyn Filled> = weak;
        let past = (device.shared().textures.first_moves).register_bytes(
            keeper,
            self.at(range.start),
            (mirror.raw, range.start),
            range.end - range.start,
            Box::new(Arc::clone(mirror)),
        );
        match past {
            true => device.submit_internal(|_, _| {}),
            false => Ok(()),
        }
    }

    /// Keep `token` for the buffer's range, dropped once the range is free (`RangeToken`): a
    /// buffer of its own, as a creator makes it. Internal for a buffer of a heap, which holds no
    /// range of its own, or one that keeps a token already: no creator gives one so.
    pub(super) fn keep_range_token(&self, token: RangeToken) -> Result<(), Error> {
        match &self.0.bytes {
            Bytes::Own(own) => own.range_token.keep(token),
            Bytes::Heap(_) => Err(internal("a range token for a heap's buffer")),
        }
    }

    /// Another reference to the buffer, which a command buffer holds it by
    /// (`EncoderCore::retain_shared`).
    pub(super) fn held(&self) -> Arc<dyn Any + Send + Sync> {
        self.0.clone()
    }

    /// The encoder core that holds it from an argument table's retain (`retained::HeldBy`).
    pub(super) fn held_by(&self) -> &HeldBy {
        &self.0.held_by
    }

    /// The buffer's identity for hazard tracking: a heap buffer's is its heap's, as Metal tracks
    /// a heap's resources as one ([`Buffer::tracked`]).
    pub(super) fn identity(&self) -> Identity {
        match &self.0.bytes {
            Bytes::Own(own) => own.identity,
            Bytes::Heap(bytes) => bytes.bytes.identity(),
        }
    }

    /// The key of the host memory a no-copy buffer joined, made over the bytes another covers
    /// (`host_aliases.rs`): its uses carry one of that key too. `None` for any other buffer.
    pub(super) fn alias(&self) -> Option<Alias> {
        match &self.0.bytes {
            Bytes::Own(own) if matches!(own.lie, Lie::Host(_)) => {
                self.0.device.shared().aliases.of(own.identity.key)
            }
            Bytes::Own(_) | Bytes::Heap(_) => None,
        }
    }

    /// The bytes `range` of the buffer as hazard tracking keys them: its identity
    /// ([`Buffer::identity`]), and the range in its bytes, or for a heap buffer in its heap's.
    pub(super) fn tracked(&self, range: Range<u64>) -> (Identity, Range<u64>) {
        let shift = match &self.0.bytes {
            Bytes::Own(_) => 0,
            Bytes::Heap(bytes) => bytes.offset,
        };
        (self.identity(), range.start + shift..range.end + shift)
    }

    /// The Vulkan device address of the buffer's byte 0, where the device enables
    /// `bufferDeviceAddress`.
    pub(super) fn device_address(&self) -> Option<u64> {
        match &self.0.bytes {
            Bytes::Own(own) => own.with(ReadApart::Binding, |pages| {
                let address = pages.memory.device_address?;
                Some(address + pages.base)
            }),
            Bytes::Heap(bytes) => bytes
                .bytes
                .device_address()
                .map(|address| address + bytes.offset),
        }
    }

    /// The buffer's range for a resident set, where the device gives buffers device addresses:
    /// in its `VkBuffer`'s space, as the residency states a used buffer (`residency.rs`), with
    /// a hold on its pages that keeps them while a command buffer may reach them. A buffer placed
    /// in a heap: its own range, as the residency states it, with a hold on the heap's pages.
    pub(super) fn resident(&self) -> Option<(ResidentRange, PagesHold)> {
        let own = match &self.0.bytes {
            Bytes::Own(own) => own,
            Bytes::Heap(heap) => {
                let (whole, held) = heap.bytes.resident()?;
                let guest = self.0.address.0;
                let host = whole.host + guest.checked_sub(whole.guest)?;
                return Some((ResidentRange::new(host, guest, self.0.length), held));
            }
        };
        let range = |pages: &Pages| {
            let host = pages.memory.device_address?;
            let guest = self.0.address.0.checked_sub(pages.base)?;
            Some(ResidentRange::new(host, guest, pages.base + self.0.length))
        };
        match &own.lie {
            Lie::Fixed(fixed) => Some((
                range(&fixed.pages)?,
                PagesHold::Fixed {
                    _pages: Arc::clone(fixed),
                },
            )),
            // A releasable import released is listed without a range: no shader reaches it
            // unnamed (`host_residency.rs`).
            Lie::Host(cell) => {
                let host = cell.current()?;
                let listing = host.import.listing()?;
                let listed = range(&listing)?;
                Some((
                    listed,
                    PagesHold::Host {
                        _pages: ListedPages::new(host),
                    },
                ))
            }
        }
    }

    /// A hold on the memory the buffer's bytes lie in now, which keeps nothing else of it
    /// ([`MemoryHold`]): a heap's buffer's, its heap's. `None` while its bytes are detached.
    pub(super) fn memory_hold(&self) -> Option<MemoryHold> {
        self.0.memory_hold()
    }

    /// A handle that does not keep the buffer (`WeakBuffer`).
    pub(super) fn downgrade(&self) -> WeakBuffer {
        WeakBuffer(Arc::downgrade(&self.0))
    }

    /// The buffer's label and purgeable state.
    pub(super) fn attributes(&self) -> &ResourceAttributes {
        &self.0.attributes
    }

    /// What a buffer made over the same host memory shares of this one (`BufferDescriptor::
    /// range_of`): its address, the range it holds, and the first byte of host memory its bytes
    /// are now; `None` but for a no-copy buffer of its own range.
    pub(super) fn sharable_range(&self) -> Option<(GpuAddress, Held, NonNull<u8>)> {
        match &self.0.bytes {
            Bytes::Own(
                own @ OwnBytes {
                    owner: Owner::Caller,
                    ..
                },
            ) => Some((
                self.exposed_address(ReadApart::SecondBuffer),
                own.reserved,
                self.contents()?,
            )),
            Bytes::Own(_) | Bytes::Heap(_) => None,
        }
    }

    /// Whose memory the buffer's bytes are, where they are the caller's: a no-copy buffer's
    /// host pages, or the memory of the caller's `VkBuffer`; a heap buffer's, its heap's (a heap
    /// over host pages, `Device::new_heap_with_bytes_no_copy`).
    pub(super) fn caller_memory(&self) -> Option<CallerMemory> {
        match &self.0.bytes {
            Bytes::Own(OwnBytes {
                owner: Owner::Caller,
                ..
            }) => Some(CallerMemory::HostPages),
            Bytes::Own(OwnBytes {
                owner: Owner::Vulkan,
                ..
            }) => Some(CallerMemory::VkBuffer),
            Bytes::Own(_) => None,
            // A heap's buffer is its heap's bytes: the caller's where the heap is over them.
            Bytes::Heap(bytes) => bytes.bytes.caller_memory(),
        }
    }

    /// The pages a buffer of its own lies in for good (the runtime's memory, or the caller's
    /// `VkBuffer`); `None` for a no-copy buffer and a heap's.
    #[cfg(unix)]
    fn fixed_pages(&self) -> Option<Arc<FixedPages>> {
        match &self.0.bytes {
            Bytes::Own(OwnBytes {
                lie: Lie::Fixed(fixed),
                ..
            }) => Some(Arc::clone(fixed)),
            Bytes::Own(_) | Bytes::Heap(_) => None,
        }
    }

    /// The buffer's hold on its host pages: a no-copy buffer's, or a no-copy heap's own
    /// buffer's; `None` for any other buffer, a heap's buffer among them.
    pub(super) fn host_cell(&self) -> Option<&Arc<HostCell>> {
        match &self.0.bytes {
            Bytes::Own(OwnBytes {
                lie: Lie::Host(cell),
                ..
            }) => Some(cell),
            Bytes::Own(_) | Bytes::Heap(_) => None,
        }
    }

    /// Note `views`, a texture's over the buffer's bytes, where a rebind gives those other host
    /// pages (a no-copy buffer's, or for a heap's buffer its heap's): the rebind empties them.
    pub(super) fn watch_views(&self, views: &Arc<TexelViews>) {
        match &self.0.bytes {
            Bytes::Own(OwnBytes {
                lie: Lie::Host(cell),
                ..
            }) => cell.watch(views),
            Bytes::Own(_) => {}
            Bytes::Heap(bytes) => bytes.bytes.watch_views(views),
        }
    }

    /// The hold on the host pages the buffer's bytes lie in: a no-copy buffer's own, a buffer
    /// of a no-copy heap its heap's buffer's; `None` for bytes of the runtime's memory or of the
    /// caller's `VkBuffer`.
    pub(super) fn pages_cell(&self) -> Option<&Arc<HostCell>> {
        match &self.0.bytes {
            Bytes::Own(OwnBytes {
                lie: Lie::Host(cell),
                ..
            }) => Some(cell),
            Bytes::Own(_) => None,
            Bytes::Heap(bytes) => bytes.bytes.pages_cell(),
        }
    }

    /// The buffer's Metal address, read for the runtime's own bookkeeping (a live-table key, a
    /// lookup), which hands it out to no one: [`Buffer::gpu_address`] without its exposure.
    pub(super) fn metal_address(&self) -> GpuAddress {
        self.0.address
    }

    /// Keep a no-copy buffer's import imported for good from now on, its Metal address handed
    /// out for `reason` (`host_residency.rs`): imported again now where the device released it,
    /// and its live-table range read again where one was noted before a release.
    fn expose(&self, reason: ReadApart) {
        let Some(cell) = self.host_cell() else {
            return;
        };
        let bit = reason.bit();
        let before = cell.exposed.load(Ordering::Acquire);
        if before & bit != 0 {
            return;
        }
        if let Some(pages) = cell.current() {
            match before {
                0 => {
                    if pages.import.expose(reason) {
                        self.device().shared().live_buffers.refresh(self);
                    }
                }
                // Kept imported already: the new reason is only counted.
                _ => pages.import.note_read_apart(reason),
            }
        }
        cell.exposed.fetch_or(bit, Ordering::Release);
    }

    /// [`Buffer::at`] for a blit's command, which reads the `VkBuffer` without exposing the
    /// buffer: a no-copy buffer's import the device released while idle is imported again first
    /// (`host_residency.rs`). With what the command buffer holds of the import's objects, where
    /// the device may release them (`rebind::ImportHold`). Refused where the device cannot import
    /// them again.
    pub(super) fn blit_at(
        &self,
        offset: u64,
    ) -> Result<((vk::Buffer, u64), Option<ImportHold>), Error> {
        if let Bytes::Own(OwnBytes {
            lie: Lie::Host(cell),
            ..
        }) = &self.0.bytes
        {
            if let Lying::In(pages) = cell.lying() {
                let (held, hold) = pages.import.blit_pages()?;
                return Ok(((held.raw, held.base + offset), hold));
            }
        }
        Ok((self.at(offset), None))
    }

    /// The host pages the buffer's bytes lie in now ([`Buffer::pages_cell`]): a no-copy
    /// buffer's, or a no-copy heap's for a buffer of the heap; `None` for any other buffer, and
    /// while they are detached.
    pub(super) fn host_pages(&self) -> Option<Arc<HostPages>> {
        self.pages_cell()?.current()
    }

    /// The runtime wrote the host addresses `addresses` of the buffer's bytes through its
    /// `contents` itself (a `replaceRegion`, an argument encoder's or a rasterization-rate map's
    /// write): reported to the logs of the writes to its pages, where their memory came with one
    /// (`dirty_pages::PagesLog`, writer 0), so every copy of the pages takes them. The pages are
    /// looked at only on a device some of whose pages came with a log.
    pub(super) fn wrote_host(&self, addresses: Range<u64>) {
        if !self.0.device.shared().dirty.logged() {
            return;
        }
        if let Some(log) = self.host_pages().as_ref().and_then(|pages| pages.log()) {
            log.wrote_at(addresses);
        }
    }

    /// The host imports `buffers` lie in now, each counted once however many lie in it
    /// (`host_aliases.rs`): a committed command buffer's used buffers'
    /// (`WorkCounts::host_imports_used`). Buffers over no host pages count none.
    pub(super) fn distinct_host_imports(buffers: &[Buffer]) -> u64 {
        let mut imports: Vec<*const HostImport> = buffers
            .iter()
            .filter_map(|buffer| buffer.host_pages())
            .map(|pages| Arc::as_ptr(&pages.import))
            .collect();
        imports.sort_unstable();
        imports.dedup();
        imports.len() as u64
    }

    /// Note `noted`, a texture over the buffer's bytes (`TexturesOver`), where they are host
    /// pages (a no-copy buffer's, or for a heap's buffer its heap's): a detach makes its image
    /// present a stand-in (`detach.rs`).
    pub(super) fn note_texture(&self, noted: NotedTexture) {
        if let Some(cell) = self.pages_cell() {
            // Made over detached bytes, its stand-in's sink may be read through.
            if noted.sink && cell.detached() {
                let reached = std::iter::once((noted.texture(), noted.id, Unexact::StorageSink));
                self.device().note_stale(cell, reached);
            }
            cell.watch_texture(noted);
        }
    }

    /// Note `watch`, an image in place over the buffer's bytes, where they are host pages (a
    /// no-copy buffer's, or for a heap's buffer its heap's): a rebind makes it again over its new
    /// pages (`texture/in_place_image.rs`).
    pub(super) fn watch_in_place(&self, watch: InPlaceWatch) {
        if let Some(cell) = self.pages_cell() {
            cell.watch_in_place(watch);
        }
    }

    /// Whether the buffer's bytes are detached host pages (a no-copy buffer's, or for a heap's
    /// buffer its heap's, `detach.rs`).
    pub(super) fn detached(&self) -> bool {
        self.pages_cell().is_some_and(|cell| cell.detached())
    }

    /// A texture is being made over the buffer's bytes: where they are host pages (a no-copy
    /// buffer's, or for a heap's buffer its heap's), a detach, and a rebind of them while they
    /// are detached, refuse until the hold goes (`detach.rs`, `buffer_descriptor.rs`), so
    /// whether they are detached holds while the texture is made. Take it before the texture
    /// binds or views them, and keep it until the texture is noted.
    pub(super) fn making_texture(&self) -> Option<TextureMaking> {
        let cell = self.pages_cell()?;
        lock(&cell.textures).making += 1;
        Some(TextureMaking(Arc::clone(cell)))
    }

    /// The index of the memory type of the host pages the buffer's bytes lie in (a no-copy
    /// buffer's, or for a heap's buffer its heap's), or lay in while they are detached; else of
    /// its memory's.
    pub(super) fn pages_type_index(&self) -> u32 {
        match self.pages_cell().map(|cell| cell.lying()) {
            Some(Lying::In(pages)) => pages.type_index(),
            Some(Lying::Detached(detached)) => detached.type_index,
            None => self.memory().2,
        }
    }

    /// Whether the buffer's bytes lie in a copy of the caller's pages (a copied host buffer,
    /// `copied.rs`), not an import of them ([`Buffer::host_backing`]); false for a buffer over no
    /// host pages, or detached.
    pub(crate) fn copied(&self) -> bool {
        self.host_backing() == Some(crate::HostBacking::Copied)
    }

    /// Whether the host pages the buffer's bytes lie in (a no-copy buffer's, or for a heap's
    /// buffer its heap's), or lay in while they are detached, were imported as dma-bufs
    /// (`dma_buf.rs`); false for a buffer over no host pages.
    pub(super) fn over_dma_buf(&self) -> bool {
        match self.pages_cell().map(|cell| cell.lying()) {
            Some(Lying::In(pages)) => pages.over_dma_buf(),
            Some(Lying::Detached(detached)) => detached.over_dma_buf,
            None => false,
        }
    }

    /// `device`.
    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// `length`: the length the buffer was made with.
    pub fn length(&self) -> u64 {
        self.0.length
    }

    /// `contents`: the CPU address of the buffer's bytes, or `None` for `Private` storage. A
    /// no-copy buffer's is the caller's pointer. Reading or writing through it while the GPU
    /// writes those bytes is a data race, as in Metal.
    pub fn contents(&self) -> Option<NonNull<u8>> {
        if self.storage_mode() == StorageMode::Private {
            return None;
        }
        match &self.0.bytes {
            // A host copy holds the buffer's bytes from its start.
            Bytes::Own(OwnBytes {
                mirror: Some(mirror),
                ..
            }) => mirror
                .placement
                .mapped
                .and_then(|mapped| NonNull::new(mapped.as_ptr())),
            // The offset is inside the placement, which the mapping covers.
            Bytes::Own(own) => {
                let shape = own.shape();
                let mapped = shape.mapped?;
                NonNull::new(mapped.as_ptr().wrapping_add(shape.offset as usize))
            }
            // The heap's storage mode is the buffer's, and the offset lies inside the heap.
            Bytes::Heap(bytes) => bytes.bytes.contents().and_then(|contents| {
                NonNull::new(contents.as_ptr().wrapping_add(bytes.offset as usize))
            }),
        }
    }

    /// The buffer's identity, a runtime extension ([`ObjectId`]): Metal names a buffer by its
    /// pointer; this is unique among the process's buffers, textures and heaps for the buffer's
    /// life and never reused, where a `gpu_address` is taken again once its buffer is freed.
    pub fn object_id(&self) -> ObjectId {
        self.0.object_id
    }

    /// `gpuAddress`: the buffer's address in the device's Metal address space. A heap buffer's
    /// is its heap's plus its offset there (`Heap::gpu_address`).
    ///
    /// A no-copy buffer whose import the device may release while idle and import again at its
    /// next blit (an import of dma-bufs: a runtime-internal residency, which nothing of Metal's
    /// observes) is kept imported for good from this call on: a command may reach it unnamed by
    /// this address.
    pub fn gpu_address(&self) -> GpuAddress {
        self.exposed_address(ReadApart::Address)
    }

    /// [`Buffer::gpu_address`] for the runtime's use that hands the address on for `reason` (a
    /// binding's, a buffer sharing the range): the import kept as `gpu_address` keeps it,
    /// counted for that reason.
    pub(super) fn exposed_address(&self, reason: ReadApart) -> GpuAddress {
        self.expose(reason);
        self.0.address
    }

    /// `resourceOptions`: the options the buffer was made with, its hazard tracking mode
    /// resolved (`Default` reports `Tracked` for a buffer made outside a heap, as Metal's do, and
    /// a heap buffer reports its heap's).
    pub fn resource_options(&self) -> ResourceOptions {
        self.0.options
    }

    /// `cpuCacheMode`.
    pub fn cpu_cache_mode(&self) -> CpuCacheMode {
        self.0
            .options
            .cpu_cache_mode()
            .expect("checked when the buffer was made")
    }

    /// `storageMode`.
    pub fn storage_mode(&self) -> StorageMode {
        self.0
            .options
            .storage_mode()
            .expect("checked when the buffer was made")
    }

    /// `hazardTrackingMode`: `Tracked` or `Untracked`, `Default` resolved as Metal resolves it.
    pub fn hazard_tracking_mode(&self) -> HazardTrackingMode {
        self.identity().mode()
    }

    /// `heap`: the heap the buffer is placed in, if it is.
    pub fn heap(&self) -> Option<&Heap> {
        match &self.0.bytes {
            Bytes::Own(_) => None,
            Bytes::Heap(bytes) => Some(&bytes.heap),
        }
    }

    /// The bytes of the heap the buffer is placed in, and where its byte 0 lies among them:
    /// `None` for a buffer outside a heap.
    pub(super) fn heap_bytes(&self) -> Option<(&Buffer, u64)> {
        match &self.0.bytes {
            Bytes::Own(_) => None,
            Bytes::Heap(bytes) => Some((&bytes.bytes, bytes.offset)),
        }
    }

    /// `heapOffset`: where the buffer lies in a placement heap. Metal answers 0 for a buffer of an
    /// automatic heap, whatever its place (measured), and for one outside a heap.
    pub fn heap_offset(&self) -> u64 {
        match &self.0.bytes {
            Bytes::Heap(bytes) if bytes.heap.heap_type() == HeapType::Placement => bytes.offset,
            Bytes::Heap(_) | Bytes::Own(_) => 0,
        }
    }

    /// `makeAliasable`: a heap buffer's bytes may be another resource's from now on. An automatic
    /// heap gives its range to the next resource it places (which then reads the bytes as they
    /// are); a placement heap's buffer is aliasable from the start, so nothing changes. Using the
    /// buffer after is the program's error, as in Metal.
    ///
    /// A buffer outside a heap stays as it is, not aliasable, as plain Metal leaves it where its
    /// validation stops the call.
    pub fn make_aliasable(&self) -> Result<(), Error> {
        let Bytes::Heap(bytes) = &self.0.bytes else {
            return Ok(());
        };
        bytes
            .heap
            .make_aliasable(bytes.offset, self.0.length, &bytes.aliasable);
        Ok(())
    }

    /// `isAliasable`: whether `make_aliasable` was called on an automatic heap's buffer; true for
    /// a placement heap's from the start, and false outside a heap, as Metal answers.
    pub fn is_aliasable(&self) -> bool {
        match &self.0.bytes {
            Bytes::Heap(bytes) => bytes.aliasable.load(Ordering::Acquire),
            Bytes::Own(_) => false,
        }
    }
}
