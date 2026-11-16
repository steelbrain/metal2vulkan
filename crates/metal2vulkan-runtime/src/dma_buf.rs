//! Dma-bufs of host pages (a runtime extension: Metal has none), the last resort of a no-copy
//! import (`host_memory.rs`) on a device that refuses the pages' pointer, as RADV refuses any
//! shared mapping (a `memfd`'s among them), and the only import on one that imports no pointer (it
//! enables no `VK_EXT_external_memory_host`). A device that lists `VK_EXT_external_memory_dma_buf`
//! and `VK_KHR_external_memory_fd` enables both, outside the vocabulary
//! (`device::create::dma_buf_names`), and has a dma-buf importer; one that imports the pointer
//! never calls a source. The caller's contract is [`DmaBufSource`]'s, [`DmaBufRequest`]'s and
//! `HostMemory::with_dma_buf`'s; this is how the runtime keeps it.
//!
//! - **The imports.** A chunk that is a dma-buf of its own (at offset 0, the runtime its `Arc`'s
//!   only holder once the source returns: [`Piece`]) is its own memory, of the best type of the
//!   buffer's memory class (`memory.rs`'s `candidates`: host-visible and coherent, never a
//!   device-coherent type of `VK_AMD_device_coherent_memory`, which the runtime does not enable)
//!   among those `vkGetMemoryFdPropertiesKHR` and the buffer take. A source that was called is
//!   kept with the imports ([`DmaBufPages`]) and dropped with them.
//! - **Shared files.** Any other chunk is a piece of a file the source shares among several
//!   buffers and heaps (a no-copy buffer's, heap's or rebind's request takes them,
//!   [`DmaBufRequest::shared_files`]; an image's, a guest RAM's and a list of runs' take a dma-buf
//!   of its own alone). Each device generation imports each such file once ([`DmaBufFiles`],
//!   keyed by the address of the file's `Arc`, which the import holds a clone of, never by the
//!   descriptor's file), as one `VkDeviceMemory` of the whole dma-buf, of the type the first
//!   buffer's class ranks best, dedicated to nothing; a later piece of the file joins that import
//!   where its buffer takes the import's type (counted: `WorkCounts::dma_buf_pieces_joined`), and
//!   its buffer is a copy of the pages, as for a chunk no type takes, where it does not. The import
//!   is made with no lock held, and entered in the device's map under its leaf lock; one made
//!   meanwhile by another thread for the same file is freed and that thread's joined. Each
//!   buffer's or heap's pages hold the import ([`DmaBufFile`]); the last holder takes it out of
//!   the map and frees it (`WorkCounts::dma_buf_shared_imports_live`), after its own `VkBuffer`
//!   went, so the import lives while any buffer bound in it does, and so while any command buffer
//!   that holds one runs. The driver imports a duplicate of the file's descriptor. A shared file's
//!   import is never dedicated, as several buffers bind in it: where the driver refuses it so
//!   (NVIDIA's 615.71.09 refuses small dma-bufs dedicated to nothing), the buffer is refused as
//!   for a refused chunk (a copy of the pages), and the source is not asked again. A recovery's
//!   new generation asks the source again and imports its answer in a map of its own. A caller
//!   may import such a file ahead and keep it ([`Device::import_shared_dma_buf`], `kept.rs`): its
//!   entry in the map holds the import while a handle keeps the file, so the last buffer to let
//!   go leaves it, and the next piece joins it (`WorkCounts::dma_buf_shared_imports_kept`).
//! - **The buffer** is a `VkBuffer` made for dma-bufs over the pages from their first byte, so its
//!   bytes lie at their offset in their first page whatever the chunks, and a rebind between one
//!   chunk and several keeps their place in it. **One chunk** gives a plain buffer bound at the
//!   chunk's offset in its import (the start of a dma-buf of its own, the piece's offset in a
//!   shared file's), a multiple of the buffer's memory alignment (the request's
//!   `one_chunk_alignment` holds it, `VUID-vkBindBufferMemory-None-10739`), whatever the
//!   request's granule; a dma-buf of its own is imported dedicated to nothing, and where the driver refuses that
//!   in every type and the buffer is the only resource bound in the import for its life (a
//!   no-copy buffer's own import, never a heap's; a copy's list of a guest RAM's runs; the page a
//!   use of a RAM asks the driver with), dedicated to the buffer (`memory.rs`'s
//!   `DmaBufFor::Buffer`), counted (`WorkCounts::dma_buf_buffer_imports_dedicated`). A guest RAM's
//!   chunk or an object's run inside one never is: its objects' buffers are bound in it too
//!   (`host_ram.rs`). **Several chunks** give a sparse one (`SPARSE_BINDING`), each chunk bound at
//!   its offset from the request's start, from its offset in its import, by one
//!   `vkQueueBindSparse` on the device's first queue
//!   (`device/submit.rs`): the call does not wait for it, which would wait behind that queue's
//!   work in flight; every later submission on every queue waits for it until it completed, and
//!   the buffer's release does where none did. The granule is the alignment of such a buffer
//!   where the device has `sparseBinding`, its queue family binds sparse memory, it imports
//!   dma-bufs for such a buffer, and such a buffer takes every memory type a plain one made for
//!   dma-bufs takes (64 KiB on RADV of Mesa 26.2.4, 4 KiB on 26.0.8); elsewhere the request's
//!   length, so that exactly one chunk is taken: a split never asks for chunks that import as no
//!   type the buffer takes where one chunk would import (NVIDIA 615.71.09's sparse buffers take
//!   none of the types its dma-bufs of host pages import as, `Device::sparse_granule`). The
//!   padding past the pages lies inside the `VkBuffer` but past the bytes a copy or a binding
//!   reaches.
//! - **The bytes.** The runtime never maps a dma-buf: `contents` stays the caller's pointer, and
//!   the CPU's writes through it and the GPU's through the imports meet in the same pages with no
//!   flush or invalidate (the types are host-coherent).
//! - **Textures** placed in a heap over such pages (and their views), whatever Metal's layout of
//!   them there (one Metal tiles is a shadow of its elements in Metal's tiles only where no such
//!   image serves: `texture/heap.rs`), where the device also lists
//!   `VK_EXT_image_drm_format_modifier` and `VK_EXT_queue_family_foreign` (enabled beside the
//!   route, `create.rs`'s `dma_buf_image_names`), are each an image of `DRM_FORMAT_MOD_LINEAR` made
//!   for dma-bufs and bound to a dedicated import (`VkMemoryDedicatedAllocateInfo`) of the one
//!   dma-buf the source makes of the pages its memory requirements cover
//!   ([`Device::dma_buf_image`]). Its one plane starts at the texels' offset in their first page,
//!   its rows the driver's own (`VkImageDrmFormatModifierListCreateInfoEXT`), as a texture placed
//!   in a heap over pages imported by their pointer is the driver's linear image. Where the
//!   driver lays a plane out at another offset than the one asked (Honeykrisp, Mesa 25.3.6 on
//!   Apple's M2 Pro, answers offset 0 for a plane 8192 bytes into a 16 KiB page), the image's
//!   plane starts at offset 0 and the image is bound at the texels' offset of an import of the
//!   pages that is no dedicated allocation, where the driver takes such a binding (it requires no
//!   dedicated allocation of the image, and its memory's alignment divides the offset: lavapipe of
//!   Mesa 26.2.6 takes one, aligned to 64 bytes; RADV of Mesa 26.2.6 requires a dedicated
//!   allocation of every such image); counted (`WorkCounts::dma_buf_images_at_offset`). Its
//!   subresource holds the same texels and bytes
//!   between rows as the explicit plane's would, and the pages' bytes before its binding lie
//!   outside it, so the device's check of the dma-buf class (`in_place_check.rs`, whose images'
//!   planes start at their pages' first byte, bound to dedicated imports) judges its accesses as
//!   it judges those. Elsewhere such a texture is a shadow. Its features
//!   are the modifier's (64-bit where the device has `format_feature_flags2`: `modifier_features`),
//!   checked as a linear image's are against the optimal tiling's for the texture's usage, and its
//!   limits the format query's for that modifier and dma-bufs. RADV (Mesa 26.2.4 and 26.0.8) lays
//!   a plane only at an offset that is a multiple of 256 bytes and rows a multiple of 256 apart,
//!   and refuses any other layout (`VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT`): the
//!   runtime asks the driver and takes its answer, never a rule of its own. The source is called
//!   only once the image is made and checked; the import keeps it, and goes after the image.
//!   Counted: `WorkCounts::dma_buf_images`. A texture over a no-copy buffer's dma-buf pages, where
//!   it is taken in place (`in_place.rs`, `texture/in_place_image.rs`), takes the image whose
//!   plane starts at offset 0 bound at its texels' offset in the import the buffer lies in, where
//!   that is one import dedicated to nothing, the device's check of such images
//!   (`PagesClass::DmaBufInImport`) passed, and the driver takes such a binding as above
//!   ([`Device::in_import_image_unbound`]; `WorkCounts::dma_buf_images_in_import`): no source
//!   asked and no import made. Else it takes the same image over a dma-buf of its own pages, and
//!   else is a shadow (`texture.rs`'s `storage_over_buffer`), whose copies go through the
//!   buffer's import. Its making asks the source once for each plan
//!   of the image it tries, but none for a range holding one the source refused with a
//!   [`DmaBufStandingRefusal`], and none for a plan whose image an earlier texture over the same
//!   pages parked as it went, which it takes (`dma_buf/parked.rs`).
//! - **Their layouts.** The image is made `UNDEFINED`, as Vulkan makes an image of external
//!   memory (`VUID-VkImageCreateInfo-pNext-01443`), and the first submission that uses it acquires
//!   it from `VK_QUEUE_FAMILY_FOREIGN_EXT` into `GENERAL` (`texture/first_move.rs`), from
//!   `GENERAL`, which keeps its contents: Vulkan leaves undefined the contents of an exclusive
//!   resource first used without its ownership transferred from the foreign family. It rests in
//!   `GENERAL` from then on, with no acquire or release around later submissions: the pages are
//!   the caller's host memory, which only the CPU (through `contents`) and this device touch, and
//!   the CPU's writes there are host writes to host-coherent memory, which a submission orders
//!   after them, as for the buffer's import; a `DRM_FORMAT_MOD_LINEAR` image keeps no metadata a
//!   foreign write could leave stale. Measured on RADV of both versions: a CPU write between two
//!   submissions, with no acquire between, is what the second reads; the GPU's writes reach the
//!   pages at the texels' places and leave the rows' padding.
//! - **Their hazards.** The image takes the identity of the heap's bytes it lies over, so hazard
//!   tracking orders its uses against the heap's other resources' over the same pages, as for any
//!   aliasing image, though its import is its own.
//! - **Where none is made** (for the reasons [`Device::dma_buf_image`] lists), the texture is a
//!   shadow of the heap's bytes, counted (`WorkCounts::dma_buf_image_fallbacks`) and announced
//!   (`shadowed-texture`, naming why).
//! - **Rebinds** of a heap onto pages imported as dma-bufs make each such texture's image again
//!   over its own new pages before the rebind settles (`rebind.rs`), and are refused
//!   (`NotImplemented(RebindTextures)`) where one is not made.
//! - **Not imported**: a source that fails, chunks the device imports as no type the buffer takes,
//!   a driver that refuses a chunk, and a sparse buffer whose alignment misses the granule asked.
//!   The pointer's refusal stands: a no-copy buffer or heap whose pointer the device refused is a
//!   copy of the pages, as where no source is given (`copied.rs`,
//!   `WorkCounts::copied_host_buffers_device_refused`); a rebind, memory imported only as dma-bufs
//!   and pages whose window the device's import alignment does not cover are refused
//!   (`Unsupported`, `HostMemory`). Counted: `WorkCounts::dma_buf_imports` (each chunk of its own,
//!   and each shared file's first import) and `WorkCounts::dma_buf_sparse_buffers`; every import
//!   tried, made or failed, is timed (`WorkCounts::dma_buf_import_calls`, `dma_buf_import_nanos`),
//!   and so is each join (`WorkCounts::dma_buf_join_nanos`).

#[cfg(unix)]
mod kept;
#[cfg(unix)]
pub(crate) mod parked;
#[cfg(unix)]
pub(crate) use kept::KeptFile;
#[cfg(unix)]
pub use kept::SharedDmaBufImport;

#[cfg(unix)]
use super::buffer::{usage, Pages};
#[cfg(unix)]
use super::buffer_descriptor::HostBuffer;
use super::device::lock;
use super::error::Error;
#[cfg(unix)]
use super::error::{internal, invalid, vulkan, MemoryKind, Problem};
#[cfg(unix)]
use super::exit_guard::{self, Call};
#[cfg(unix)]
use super::host_memory::{based_size, import_refused, not_importable, ImportWindow, Importer};
#[cfg(unix)]
use super::host_residency::Again;
use super::image_plan::{ImageFacts, ImagePlan};
#[cfg(unix)]
use super::in_place_check::Unmade;
use super::linear_plan::{check_linear_limits, tiling_lack, viewed_formats, LinearLack};
#[cfg(unix)]
use super::memory::{self, DmaBufFor, ImportFailure, MemoryClass, Placement};
#[cfg(unix)]
use super::program_objects::ProgramObject;
use super::Device;
#[cfg(unix)]
use crate::release;
use ash::vk;
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::vk::{Format, FormatFeatureFlags2, ImageCreateFlags, SampleCountFlags};
#[cfg(unix)]
use std::ops::Range;
#[cfg(unix)]
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
#[cfg(unix)]
use std::ptr::NonNull;
#[cfg(unix)]
use std::sync::{Arc, Mutex, PoisonError, Weak};
#[cfg(unix)]
use std::time::{Duration, Instant};

/// A runtime extension (Metal has none): makes dma-bufs of the host pages a no-copy import
/// covers, for a device that refuses the pages' pointer
/// ([`HostMemory::with_dma_buf`](crate::HostMemory::with_dma_buf)).
///
/// Called on the thread making the object, with no lock of the runtime's held: for a buffer or
/// heap (or its rebind, or a buffer's making again over the same pages at its device's recovery,
/// on the thread that recovers, before the new `VkDevice` serves, so it makes no call on the
/// device), only where the device refuses the pointer and imports dma-bufs for it;
/// then once for each image the runtime makes over a texture's own pages in them (over the
/// buffer, placed in the heap, a view of one, or remade by a rebind). A texture in place over a
/// buffer's pages may take the image an earlier texture over the same pages kept as it went, of
/// the same plan at the same texels, and asks nothing (`WorkCounts::dma_buf_images_taken`); or,
/// where the buffer lies in one import dedicated to nothing and the device's check passed images
/// bound in such an import, its image is bound there, at its texels' offset, and asks nothing
/// either (`WorkCounts::dma_buf_images_in_import`). A source the runtime never calls is dropped with the call; one it called is kept with the
/// imports it made, until they go.
///
/// It returns dma-bufs of the pages asked for ([`DmaBufRequest`]) in chunks ([`DmaBufChunk`]), in
/// order: each chunk's length a positive multiple of the host page, its offset in its file a
/// multiple of the request's `one_chunk_alignment` where it answers alone, else of its
/// `alignment`, the chunk within its file's size, and every boundary between two chunks, as an
/// offset from the request's `start`, a multiple of its `granule`. A
/// chunk is a dma-buf of its own (a file the source keeps no clone of, the chunk at offset 0),
/// or, where the request's `shared_files` allows, a piece of a file the source shares among the
/// buffers and heaps it answers (several pieces of one file, in one answer or in many, each the
/// same `Arc`): the runtime imports each such file once on the device, whatever buffers lie in
/// it, and binds each buffer at its piece's offset there, so a device that lists every import in
/// every submission (RADV) lists one per file, not one per buffer. A file's import lives while
/// any buffer or heap bound in it does. The descriptors are the runtime's from then on: Vulkan's
/// once imported, closed otherwise; a shared file's stays open while the runtime holds its `Arc`.
/// An `Err` keeps the pointer's refusal (`Unsupported`, `HostMemory`), or makes a texture a
/// shadow of the bytes instead (announced `shadowed-texture`); chunks that break these rules or
/// the request's, or a descriptor that is no dma-buf, are refused as misuse (`Invalid`,
/// `Problem::DmaBuf`). A texture's making asks once for each plan of its image it tries in place
/// (its own usage, and wider ones before it), each for the pages that plan's image covers: an
/// `Err` refuses that request alone, and the next plan's is asked, but for an `Err` holding a
/// [`DmaBufStandingRefusal`], after which the making asks for no range that holds the refused one.
#[cfg(unix)]
pub type DmaBufSource =
    Arc<dyn Fn(DmaBufRequest) -> std::io::Result<Vec<DmaBufChunk>> + Send + Sync + 'static>;

/// A runtime extension (Metal has none): a [`DmaBufSource`]'s refusal of a texture's request that
/// stands for every range of pages that holds the range refused, for the rest of that texture's
/// making.
///
/// A source returns it as its error, `Err(std::io::Error::other(DmaBufStandingRefusal::new(why)))`.
/// The runtime tries a texture over a no-copy buffer's pages imported as dma-bufs in place with
/// each plan of its image in turn, an image of usages wider than the texture's first and its own
/// last, and asks the source for the pages each plan's image covers ([`DmaBufRequest`]: a range
/// from the page of the texels' first byte, which each plan's image may cover farther). After this
/// refusal the making asks the source for no range that holds the refused one: those plans are
/// refused with the source's words, unasked, and where none is made the texture is a shadow of
/// its pages (`shadowed-texture`, naming the words), as after any refusal. Any other error refuses
/// the request it answers alone, and the next plan's request is asked.
///
/// Return it only where no range that holds the refused one can ever be accepted while the
/// texture is made: a source of `udmabuf`s whose pages there are more runs of its file than one
/// `udmabuf` lists (its `list_limit`), with no dma-buf it keeps of a wider range over them. Where
/// a wider range may be accepted (a dma-buf kept of exactly that range, a split at another
/// granule), refuse with any other error.
#[cfg(unix)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DmaBufStandingRefusal {
    why: String,
}

#[cfg(unix)]
impl DmaBufStandingRefusal {
    /// A standing refusal for `why`, the source's words, which the texture's notice names.
    pub fn new(why: impl Into<String>) -> DmaBufStandingRefusal {
        DmaBufStandingRefusal { why: why.into() }
    }
}

#[cfg(unix)]
impl std::fmt::Display for DmaBufStandingRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.why)
    }
}

#[cfg(unix)]
impl std::error::Error for DmaBufStandingRefusal {}

/// What a [`DmaBufSource`] is asked for: `length` bytes of host pages from `start`, both multiples
/// of the host page, to be returned as chunks split only at multiples of `granule` from `start`,
/// one chunk at an offset in its file that is a multiple of `one_chunk_alignment`, or several each
/// at a multiple of `alignment`, pieces of shared files only where `shared_files` says so.
///
/// **Picking an alignment.** An answer of one chunk is bound to a plain buffer, which Vulkan binds
/// at any multiple of the buffer's memory alignment (`VUID-vkBindBufferMemory-None-10739`): its
/// offset need be a multiple of `one_chunk_alignment` alone, the host page on RADV, whatever the
/// `granule`. An answer of several chunks is bound to a sparse buffer, which Vulkan binds only at
/// multiples of its alignment in both the buffer and the memory
/// (`VUID-VkSparseMemoryBind-resourceOffset-09491`): each chunk's offset a multiple of `alignment`,
/// the granule where the request may split (64 KiB on RADV of Mesa 26.2.4). So a source that holds
/// the pages in a file it shares answers with one piece wherever they lie in one such file at a
/// multiple of `one_chunk_alignment`, and splits them only where they lie in several, at multiples
/// of `alignment`. `one_chunk_alignment` divides `alignment`, and is it where the `granule` is the
/// `length`.
///
/// A buffer's or heap's request is the caller's range rounded out to the host page, whatever the
/// device's import alignment for host pointers (a dma-buf has none). A texture's starts at the page
/// its texels start in and runs through the pages its image's memory covers, inside an import's
/// pages; its `granule` is its `length`, and the image is bound to its one chunk's import alone.
///
/// One chunk holds the `length` bytes, or them rounded up to `granule`. Several hold exactly that
/// rounding (Vulkan binds a sparse buffer's memory in whole granules,
/// `VUID-VkSparseMemoryBind-resourceOffset-09491`): the last chunk's pages past `length` are
/// padding, which the buffer's bytes never reach and the GPU reaches only through an access past
/// the buffer's pages. Pad with a scratch page of the caller's own, repeated, not pages of other
/// memory. Chunks that total otherwise are refused (`Invalid`, `Problem::DmaBuf`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(unix)]
pub struct DmaBufRequest {
    /// The first byte of the first page.
    pub start: NonNull<u8>,
    /// The bytes of the pages.
    pub length: u64,
    /// The multiple of which every boundary between two chunks lies from `start`: the alignment
    /// of a sparse buffer made for dma-bufs, where the device binds one in several chunks
    /// (a multiple of the host page); `length` where it does not, so that exactly one chunk is
    /// taken.
    pub granule: u64,
    /// The multiple of which each chunk's offset in its file is in an answer of several chunks:
    /// the `granule` where the request may be answered so (each binds a sparse buffer's memory at
    /// its offset, `VUID-VkSparseMemoryBind-resourceOffset-09491`), else `one_chunk_alignment`. A
    /// power of two. A chunk at another offset is refused (`Invalid`, `Problem::DmaBuf`).
    pub alignment: u64,
    /// The multiple of which the offset in its file of a chunk that answers the request alone is:
    /// the host page, or the memory alignment of the plain buffer bound there where that is larger
    /// (Vulkan binds memory only at a multiple of it, `VUID-vkBindBufferMemory-None-10739`), the
    /// same for every buffer the device makes for dma-bufs. A power of two that divides
    /// `alignment`. A lone chunk at another offset is refused (`Invalid`, `Problem::DmaBuf`).
    pub one_chunk_alignment: u64,
    /// Whether the chunks may be pieces of files the source shares, or lie at an offset in their
    /// file: a no-copy buffer's, heap's or rebind's pages. Where `false` each chunk must be a
    /// dma-buf of its own at offset 0, the runtime the only holder of its `Arc` once the source
    /// returns (a texture's image's dedicated import, a guest RAM's chunk or a use's probe of
    /// one, `HostRam::with_dma_buf`); another answer is refused as the source's failure is there.
    pub shared_files: bool,
}

// SAFETY: a request is an address and two lengths, plain data: nothing is reached through the
// address by having the request.
#[cfg(unix)]
unsafe impl Send for DmaBufRequest {}
// SAFETY: as above.
#[cfg(unix)]
unsafe impl Sync for DmaBufRequest {}

/// A runtime extension (Metal has none): makes one dma-buf of a list of runs of a guest RAM's
/// host pages, for a copy by runs over them
/// ([`HostRam::with_dma_buf_of_runs`](crate::HostRam::with_dma_buf_of_runs)); on Linux, udmabuf's
/// `UDMABUF_CREATE_LIST`.
///
/// Each request of the list is whole host pages ([`DmaBufRequest`]: `start` and `length`
/// multiples of the host page, its `granule` its `length`); the requests are by start and share
/// no page, each inside one chunk of the RAM (they may lie in several chunks, and in several of
/// its ranges), and at most as many as the RAM was given. It returns
/// one [`DmaBufChunk`]: a dma-buf of its own (each request's `shared_files` is `false`) whose
/// bytes are the requests' pages one after another, in order, its `length` their total. The
/// descriptor is the runtime's from then on: Vulkan's once imported, closed otherwise. An `Err`,
/// or a chunk of another length, a piece of a shared file or at an offset, or no dma-buf, leaves
/// the copy to what it takes without the source.
#[cfg(unix)]
pub type DmaBufRunsSource =
    Arc<dyn Fn(&[DmaBufRequest]) -> std::io::Result<DmaBufChunk> + Send + Sync + 'static>;

/// One chunk of a [`DmaBufSource`]'s answer (a runtime extension: Metal has none): the bytes
/// `offset..offset + length` of the dma-buf `file`, which are the requested pages from where the
/// chunks before it end.
///
/// A dma-buf of its own is `file: Arc::new(fd), offset: 0`, imported alone, as its own memory. A
/// piece of a file the source shares is the same `Arc` (a clone of it) in each chunk over that
/// file: the runtime knows a file by its `Arc`'s identity alone (never by the descriptor's number
/// or the file's inode), imports a duplicate of its descriptor once on each device, as one
/// `VkDeviceMemory` of the whole dma-buf, dedicated to nothing, and binds every buffer over its
/// pieces at their offsets in that import ([`DmaBufSource`]).
#[derive(Clone, Debug)]
#[cfg(unix)]
pub struct DmaBufChunk {
    /// The dma-buf the chunk lies in.
    pub file: Arc<OwnedFd>,
    /// Where the chunk starts in the dma-buf: a multiple of the request's `alignment`, and 0 for
    /// a dma-buf of its own.
    pub offset: u64,
    /// The bytes of the chunk, a multiple of the host page, within the dma-buf from `offset`.
    pub length: u64,
}

/// The imports of the chunks a no-copy buffer's host pages lie in, and the source that made them:
/// held by the pages' placement (`memory.rs`), so by the buffers and images over them, and freed,
/// the source dropped, once the last of those goes. Every `VkBuffer` bound to them is destroyed
/// first (`buffer.rs`'s `HostImport`). A chunk of a file the source shares lies in the device's
/// one import of that file ([`DmaBufFile`]), which the pages hold beside the other buffers' and
/// heaps' pages bound in it: they let go of their hold as they go, and the last to let go frees
/// it ([`DmaBufFiles::let_go`]).
///
/// The last holder may go on a runtime thread (a command buffer's completion lets go of the
/// buffer): the imports are freed inside a section of the exit guard, as every Vulkan call there
/// is, and the source, the program's code, is dropped after it as the program's
/// (`exit_guard::program`), its panic caught. While the process exits both are left to its end.
#[cfg(unix)]
pub(super) struct DmaBufPages {
    device: Device,
    /// The imports of the pages' chunks that are dma-bufs of their own, freed as the pages go.
    memories: Vec<vk::DeviceMemory>,
    /// The device's imports of the shared files the pages' other chunks lie in, one hold per
    /// such chunk: let go of as the pages go, freed with the last hold.
    files: Vec<Arc<DmaBufFile>>,
    /// The imports the pages made, of their own chunks and of shared files no buffer had
    /// imported yet: what `WorkCounts::dma_buf_imports` counts of them.
    made: usize,
    /// The imports are an image's dedicated one (`Device::dma_buf_image`), not a buffer's chunks:
    /// what the live gauges count them as (`WorkCounts::dma_buf_image_memories_live`).
    of_image: bool,
    /// The caller's source, held as the program's object (`program_objects.rs`); taken only as
    /// the pages drop.
    source: Option<ProgramObject<DmaBufSource>>,
    /// The host address of the first page, and the bytes of the pages: what the source made the
    /// imports of.
    start: u64,
    length: u64,
    /// The sparse binding of the buffer over several imports, which nothing waited for: the
    /// internal timeline and the value it signals (`Device::bind_sparse_buffer`).
    bound: Option<(vk::Semaphore, u64)>,
    /// A no-copy buffer's own import that the device would release while idle but keeps
    /// imported, as the descriptors to import it again did not fit (`host_residency.rs`):
    /// counted in `WorkCounts::host_imports_kept_for_fds` while the pages live.
    kept_for_fds: bool,
    /// An import of a chunk of their own was made dedicated to the buffer bound in it, where the
    /// driver refused it dedicated to nothing ([`Device::import_chunks`]): no image binds there
    /// (`VUID-vkBindImageMemory-memory-10926`), so a texture over the pages takes no image in
    /// their import ([`Device::in_import_image_unbound`]).
    dedicated_to_buffer: bool,
    /// What waits for the imports' free ([`DmaBufPages::after_free`]): the caller's keep-alive
    /// and the deallocators of the pages that went, let go of once the imports are freed.
    after_free: Mutex<Vec<Box<dyn Send>>>,
}

#[cfg(unix)]
impl Drop for DmaBufPages {
    fn drop(&mut self) {
        // The Vulkan teardown is a section of the exit guard.
        let Some(busy) = exit_guard::enter() else {
            // The process is exiting: the imports stay to its end, and the source and what waits
            // for their free with them.
            if let Some(source) = self.source.take() {
                std::mem::forget(source);
            }
            for file in self.files.drain(..) {
                std::mem::forget(file);
            }
            let after_free = self.after_free.get_mut();
            for after in after_free.unwrap_or_else(PoisonError::into_inner).drain(..) {
                std::mem::forget(after);
            }
            return;
        };
        let shared = self.device.shared();
        let raw_device = shared.vk();
        // The shared files' imports whose last hold these pages were, out of the device's map.
        let last = shared.dma_buf_files.let_go(self.files.drain(..));
        let freed: Vec<vk::DeviceMemory> = (self.memories.iter().copied())
            .chain(last.iter().map(|file| file.memory))
            .collect();
        if let Some(log) = crate::memory_log::MemoryLog::new(shared.instance.sink()) {
            use ash::vk::Handle;
            let memories: Vec<String> = (freed.iter())
                .map(|memory| format!("0x{:x}", memory.as_raw()))
                .collect();
            log.line(format!(
                "metal2vulkan: memory: freed the dma-buf imports of host memory 0x{:x}, {} \
                 bytes: memory {}",
                self.start,
                self.length,
                memories.join(", ")
            ));
        }
        for &memory in &freed {
            // SAFETY: nothing is bound to the imports any more and no command reads them (the
            // struct's docs): a shared file's, once its last holder let go of it, has no buffer
            // bound in it either.
            unsafe { release::object(raw_device, memory) };
        }
        (shared.counts).dma_buf_memories_freed(self.of_image, freed.len());
        shared.counts.dma_buf_shared_imports_freed(last.len());
        if self.kept_for_fds {
            shared.counts.host_import_kept_for_fds(false);
        }
        drop(busy);
        if let Some(source) = self.source.take() {
            exit_guard::program_release(move || drop(source));
        }
        // Freed: what waited for it (`after_free`, the keep-alive that kept the pages mapped
        // among it) goes as the fields drop, after this.
    }
}

#[cfg(unix)]
impl DmaBufPages {
    /// The imports `source` made of the pages `request` asked for, none imported yet: an image's
    /// dedicated one where `of_image`, else a buffer's chunks.
    fn new(
        device: &Device,
        source: DmaBufSource,
        request: DmaBufRequest,
        of_image: bool,
    ) -> DmaBufPages {
        DmaBufPages {
            device: device.clone(),
            memories: Vec::new(),
            files: Vec::new(),
            made: 0,
            of_image,
            source: Some(ProgramObject::new(source)),
            start: request.start.as_ptr() as u64,
            length: request.length,
            bound: None,
            kept_for_fds: false,
            dedicated_to_buffer: false,
            after_free: Mutex::new(Vec::new()),
        }
    }

    /// What the pages hold of the device's imports, which a release of the import they lie in
    /// lets go of (`host_residency.rs`): whether they hold imports of their own, and their holds
    /// of shared files' imports, one per piece.
    pub(super) fn listed(&self) -> (bool, &[Arc<DmaBufFile>]) {
        (!self.memories.is_empty(), &self.files)
    }

    /// The imports of one dma-buf of a list of runs of a guest RAM's pages, the first from host
    /// address `start` ([`Device::ram_runs_as_dma_buf`]), none imported yet: a buffer's, with no
    /// source kept (the RAM keeps its source of run lists, and the import holds the RAM until it
    /// is freed: `host_ram.rs`'s `RamChunk`) and no bytes held in host order (the runs lie apart),
    /// so no texture's request is ever over them.
    fn of_runs(device: &Device, start: NonNull<u8>) -> DmaBufPages {
        DmaBufPages {
            device: device.clone(),
            memories: Vec::new(),
            files: Vec::new(),
            made: 0,
            of_image: false,
            source: None,
            start: start.as_ptr() as u64,
            length: 0,
            bound: None,
            kept_for_fds: false,
            dedicated_to_buffer: false,
            after_free: Mutex::new(Vec::new()),
        }
    }

    /// The import of a dma-buf of the runtime's own memory exported as one, `length` bytes, none
    /// imported yet: an image's dedicated one ([`Device::bind_exported_dma_buf_image`]), with no
    /// source and no host pages (the buffer holds the memory, and the texture the buffer's).
    fn of_exported(device: &Device, length: u64) -> DmaBufPages {
        DmaBufPages {
            device: device.clone(),
            memories: Vec::new(),
            files: Vec::new(),
            made: 0,
            of_image: true,
            source: None,
            start: 0,
            length,
            bound: None,
            kept_for_fds: false,
            dedicated_to_buffer: false,
            after_free: Mutex::new(Vec::new()),
        }
    }

    /// Let `after` go once the imports are freed: as their last holder lets go of them, which
    /// may be a moment after the caller (an import going, `buffer.rs`'s `Imported`) lets go of
    /// its own hold.
    pub(super) fn after_free(&self, after: Box<dyn Send>) {
        lock(&self.after_free).push(after);
    }

    /// Hold the device through a handle the program's last drop waits for, in place of its own
    /// (`Device::held_for_release`): the imports go in a release posted to a handler thread
    /// (`buffer.rs`'s `HostImport`).
    pub(super) fn hold_for_release(&mut self) {
        self.device = self.device.held_for_release();
    }

    /// Hold `memory`, imported from one of the source's dma-bufs, until the pages go; counted
    /// live.
    fn hold(&mut self, memory: vk::DeviceMemory) {
        self.memories.push(memory);
        self.made += 1;
        (self.device.shared().counts).dma_buf_memory_imported(self.of_image);
    }

    /// Wait until the sparse binding of the buffer over the imports completed, where it may not
    /// have: before the buffer is destroyed (`buffer.rs`'s `HostImport`, or a buffer not made
    /// after all). A submission that used the buffer waited for it, so this waits only for a
    /// buffer let go of before any did, behind the first queue's work submitted before it. A lost
    /// device ends the wait: its work counts as complete.
    pub(super) fn settle(&self) {
        let Some((timeline, value)) = self.bound else {
            return;
        };
        let raw_device = self.device.shared().vk();
        let semaphores = [timeline];
        let values = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        loop {
            // SAFETY: the timeline is the device's, live while the device is.
            let waited = unsafe { raw_device.wait_semaphores(&info, WAIT_SLICE.as_nanos() as u64) };
            exit_guard::called(Call::WaitSemaphores);
            // A stopped device's binding never completes (`command_buffer/stall.rs`).
            if waited != Err(vk::Result::TIMEOUT) || crate::release::parked(raw_device.handle()) {
                return;
            }
        }
    }

    /// Whether an import of theirs is dedicated to the buffer bound in it, where the driver
    /// refused it dedicated to nothing: an image binds in none of their memory then
    /// (`VUID-vkBindImageMemory-memory-10926`).
    pub(super) fn dedicated_to_buffer(&self) -> bool {
        self.dedicated_to_buffer
    }

    /// Whether the pages hold the `length` bytes from host address `start`.
    fn holds(&self, start: u64, length: u64) -> bool {
        let end = start.checked_add(length);
        start >= self.start && end.is_some_and(|end| end <= self.start + self.length)
    }
}

#[cfg(unix)]
impl std::fmt::Debug for DmaBufPages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DmaBufPages")
            .field("memories", &self.memories)
            .field("files", &self.files.len())
            .finish_non_exhaustive()
    }
}

/// A chunk of a source's answer as the runtime takes it ([`DmaBufChunk`]).
#[derive(Debug)]
#[cfg(unix)]
pub(super) enum Piece {
    /// A dma-buf of its own, at offset 0, the runtime its `Arc`'s only holder: `length` bytes of
    /// it imported alone, as today's chunk always was, the import its own memory.
    Own { fd: OwnedFd, length: u64 },
    /// A piece of a file the source shares, or at an offset in its file: bound at `offset` of the
    /// device's one import of the file ([`DmaBufFiles`]).
    Shared {
        file: Arc<OwnedFd>,
        offset: u64,
        length: u64,
    },
}

#[cfg(unix)]
impl Piece {
    /// `chunk` as a dma-buf of its own where it is one (at offset 0, and the source kept no
    /// clone of its `Arc`: taking the descriptor out is atomic, so a source's clone made at once
    /// on another thread leaves it shared), else as a piece of a shared file.
    pub(super) fn of(chunk: DmaBufChunk) -> Piece {
        let DmaBufChunk {
            file,
            offset,
            length,
        } = chunk;
        if offset != 0 {
            return Piece::Shared {
                file,
                offset,
                length,
            };
        }
        match Arc::try_unwrap(file) {
            Ok(fd) => Piece::Own { fd, length },
            Err(file) => Piece::Shared {
                file,
                offset,
                length,
            },
        }
    }

    /// The bytes of the piece.
    fn length(&self) -> u64 {
        match self {
            Piece::Own { length, .. } | Piece::Shared { length, .. } => *length,
        }
    }

    /// Where the piece lies in its import: 0 in its own, its offset in its file's.
    fn offset(&self) -> u64 {
        match self {
            Piece::Own { .. } => 0,
            Piece::Shared { offset, .. } => *offset,
        }
    }

    /// Whether it is a dma-buf of its own, which holds a descriptor of the runtime's.
    pub(super) fn own(&self) -> bool {
        matches!(self, Piece::Own { .. })
    }

    /// The same piece again, for an import made again of it (`host_residency.rs`): a duplicate of
    /// its own descriptor, or a clone of its file's `Arc`, which opens no descriptor.
    pub(super) fn again(&self) -> std::io::Result<Piece> {
        Ok(match self {
            Piece::Own { fd, length } => Piece::Own {
                fd: fd.try_clone()?,
                length: *length,
            },
            Piece::Shared {
                file,
                offset,
                length,
            } => Piece::Shared {
                file: Arc::clone(file),
                offset: *offset,
                length: *length,
            },
        })
    }
}

/// A device's one import of a dma-buf a caller's source shares among the chunks of several
/// buffers and heaps ([`DmaBufChunk`]): the whole dma-buf, dedicated to nothing, of one memory
/// type. Held by the pages of each buffer and heap bound in it ([`DmaBufPages`]), and by its
/// entry in the device's map while a handle keeps its file ([`SharedDmaBufImport`]), and freed by
/// the last of them to let go ([`DmaBufFiles::let_go`], [`DmaBufFiles::unkeep`]): every hold is
/// made and let go of under the device's map's lock, so it has no handle of the device and no
/// `Drop` of its own.
#[derive(Debug)]
#[cfg(unix)]
pub(super) struct DmaBufFile {
    /// The file, held so that no other file's `Arc` takes its address while it is imported.
    file: Arc<OwnedFd>,
    memory: vk::DeviceMemory,
    type_index: u32,
}

#[cfg(test)]
#[cfg(unix)]
impl DmaBufFile {
    /// A stand-in import of `file`, of no memory: for the map's own tests
    /// (`tests/dma_buf_pieces.rs`).
    pub(super) fn stand_in(file: Arc<OwnedFd>) -> DmaBufFile {
        DmaBufFile {
            file,
            memory: vk::DeviceMemory::null(),
            type_index: 0,
        }
    }
}

/// A device generation's one import of each dma-buf a caller's source shares
/// ([`DmaBufFile`]), by the address of the file's `Arc`: the file's identity, never the
/// descriptor's number or the file's inode (a dma-buf's inode is no documented identity). An
/// entry lives while its import does or a handle keeps the file ([`FileEntry`]): the last holder
/// lets go of the import and removes the entry under the map's lock, so the map holds at most one
/// entry per live import or kept file, as many as the buffers and heaps bound in them and the
/// live [`SharedDmaBufImport`]s at most. A recovery's next generation has a map of its own, which
/// the recovery gives the kept files' keeps (`dma_buf/kept.rs`). The lock is a leaf: nothing else
/// is taken while it is held, and no driver call is made under it.
#[derive(Default)]
#[cfg(unix)]
pub(crate) struct DmaBufFiles(Mutex<std::collections::HashMap<usize, FileEntry>>);

/// A file's entry in a generation's map: its live import, held weakly, and the handles that keep
/// the file ([`SharedDmaBufImport`]). While a handle keeps it and it is imported, the entry holds
/// the import as well, so the last buffer or heap bound in it to let go leaves it imported, and
/// the import made while one keeps it is held as it is entered.
#[derive(Default)]
#[cfg(unix)]
struct FileEntry {
    import: Weak<DmaBufFile>,
    keeps: usize,
    kept: Option<Arc<DmaBufFile>>,
}

/// What a keep of a file found in a generation's map ([`DmaBufFiles::keep`]).
#[cfg(unix)]
pub(super) enum Keep {
    /// The file is imported, and kept: by this keep first where `first`.
    Imported { first: bool },
    /// The file is not imported on the generation: the import made next is kept as it is
    /// entered.
    Unimported,
}

/// What letting go of a keep of a file left ([`DmaBufFiles::unkeep`]).
#[cfg(unix)]
#[derive(Default)]
pub(super) struct Unkept {
    /// The entry held the import no more: the last keep went while the file was imported.
    pub(super) let_go: bool,
    /// The import, where that hold was its last, out of the map, for the caller to free.
    pub(super) last: Option<DmaBufFile>,
}

/// A file's key in the map: its `Arc`'s address.
#[cfg(unix)]
fn file_key(file: &Arc<OwnedFd>) -> usize {
    Arc::as_ptr(file) as usize
}

#[cfg(unix)]
impl DmaBufFiles {
    /// The live import of `file`, a new hold of it; `None` where the device holds none.
    pub(super) fn live(&self, file: &Arc<OwnedFd>) -> Option<Arc<DmaBufFile>> {
        lock(&self.0)
            .get(&file_key(file))
            .and_then(|entry| entry.import.upgrade())
    }

    /// `made`, a new import of its file made with no lock held, entered as the file's import, and
    /// whether the entry keeps it (a handle keeps the file); or, where another thread entered one
    /// of the same file meanwhile, a hold of that one, with `made` back for the caller to free.
    pub(super) fn enter(
        &self,
        made: DmaBufFile,
    ) -> Result<(Arc<DmaBufFile>, bool), (Arc<DmaBufFile>, DmaBufFile)> {
        let key = file_key(&made.file);
        let mut map = lock(&self.0);
        let entry = map.entry(key).or_default();
        if let Some(live) = entry.import.upgrade() {
            return Err((live, made));
        }
        let entered = Arc::new(made);
        entry.import = Arc::downgrade(&entered);
        let kept = entry.keeps > 0;
        if kept {
            entry.kept = Some(Arc::clone(&entered));
        }
        Ok((entered, kept))
    }

    /// Let go of `holds`: the imports they were the last holds of, out of the map, for the
    /// caller to free. A hold is taken only under the lock ([`live`](Self::live),
    /// [`enter`](Self::enter)), so the last is known here exactly; none is while a handle keeps
    /// its file, as the entry holds the import then.
    pub(super) fn let_go(&self, holds: impl Iterator<Item = Arc<DmaBufFile>>) -> Vec<DmaBufFile> {
        let mut map = lock(&self.0);
        let mut last = Vec::new();
        for hold in holds {
            let key = file_key(&hold.file);
            let at = Arc::as_ptr(&hold);
            if let Some(file) = Arc::into_inner(hold) {
                let gone = |entry: &FileEntry| entry.import.as_ptr() == at && entry.keeps == 0;
                if map.get(&key).is_some_and(gone) {
                    map.remove(&key);
                }
                last.push(file);
            }
        }
        last
    }

    /// Keep `file` for a handle ([`SharedDmaBufImport`]): the entry holds its import while any
    /// keep lasts, the one live now or the one entered next.
    pub(super) fn keep(&self, file: &Arc<OwnedFd>) -> Keep {
        let mut map = lock(&self.0);
        let entry = map.entry(file_key(file)).or_default();
        entry.keeps += 1;
        match entry.import.upgrade() {
            Some(live) => {
                let first = entry.kept.is_none();
                entry.kept = Some(live);
                Keep::Imported { first }
            }
            None => Keep::Unimported,
        }
    }

    /// Whether the entry of `file` holds its import for the handles that keep it.
    pub(super) fn kept(&self, file: &Arc<OwnedFd>) -> bool {
        (lock(&self.0).get(&file_key(file))).is_some_and(|entry| entry.kept.is_some())
    }

    /// Let go of a keep of `file` ([`keep`](Self::keep)): with the last, the entry's hold of the
    /// import, freed where it was the last hold (no buffer or heap is bound in it).
    pub(super) fn unkeep(&self, file: &Arc<OwnedFd>) -> Unkept {
        let key = file_key(file);
        let mut map = lock(&self.0);
        let Some(entry) = map.get_mut(&key) else {
            return Unkept::default();
        };
        entry.keeps = entry.keeps.saturating_sub(1);
        if entry.keeps > 0 {
            return Unkept::default();
        }
        let kept = entry.kept.take();
        let let_go = kept.is_some();
        let last = kept.and_then(Arc::into_inner);
        if entry.import.strong_count() == 0 {
            map.remove(&key);
        }
        Unkept { let_go, last }
    }
}

/// Whether chunks of `lengths` bytes, in order, answer `request` as the [`DmaBufSource`] contract
/// allows on a host of `page`-byte pages: at least one, each a positive multiple of the page;
/// every boundary between two a multiple of the granule from the request's start; together the
/// request's length rounded up to the granule, or, one chunk, the length itself.
#[cfg(unix)]
pub(super) fn chunks_cover(lengths: &[u64], request: &DmaBufRequest, page: u64) -> bool {
    let page = page.max(1);
    let granule = request.granule.max(1);
    let Some(rounded) = request.length.checked_next_multiple_of(granule) else {
        return false;
    };
    let mut end = 0u64;
    for &length in lengths {
        if !end.is_multiple_of(granule) || length == 0 || !length.is_multiple_of(page) {
            return false;
        }
        end = match end.checked_add(length) {
            Some(end) => end,
            None => return false,
        };
    }
    match lengths.len() {
        0 => false,
        1 => end == request.length || end == rounded,
        _ => end == rounded,
    }
}

/// Whether a failed import of a guest RAM's dma-buf (`Device::ram_span_as_dma_buf`,
/// `ram_runs_as_dma_buf`) is a refusal that lasts for the device's life: the driver refuses the
/// dma-buf's descriptor, or its import in every memory type and form tried
/// (`ERROR_INVALID_EXTERNAL_HANDLE`), imports it as no type its buffer takes, or the buffer's
/// requirements miss its pages: each [`not_importable`]. Memory that ran out (`OutOfMemory`,
/// the driver's answer in every type and form tried) is that import's alone: a later dma-buf,
/// of another length, or with memory freed, may be taken.
#[cfg(unix)]
fn lasting_refusal(error: &Error) -> bool {
    *error == not_importable()
}

/// Whether `chunks` answer `request` as the [`DmaBufSource`] contract allows on a host of
/// `page`-byte pages ([`chunks_cover`]), each within its dma-buf from its offset, a multiple of
/// the request's alignment for its answer ([`answer_alignment`]).
#[cfg(unix)]
fn answers(chunks: &[DmaBufChunk], request: &DmaBufRequest, page: u64) -> bool {
    let lengths: Vec<u64> = chunks.iter().map(|chunk| chunk.length).collect();
    let alignment = answer_alignment(request, chunks.len());
    let held = |chunk: &DmaBufChunk| {
        let end = chunk.offset.checked_add(chunk.length);
        chunk.offset.is_multiple_of(alignment)
            && (end.zip(dma_buf_length(chunk.file.as_fd())))
                .is_some_and(|(end, bytes)| end <= bytes)
    };
    chunks_cover(&lengths, request, page) && chunks.iter().all(held)
}

/// The multiple of which each chunk's offset in its file lies in an answer of `chunks` chunks to
/// `request`: one is bound to a plain buffer at its offset, which Vulkan takes at any multiple of
/// the buffer's memory alignment (`VUID-vkBindBufferMemory-None-10739`,
/// [`DmaBufRequest::one_chunk_alignment`]); several to a sparse buffer, which Vulkan binds only at
/// multiples of its alignment (`VUID-VkSparseMemoryBind-resourceOffset-09491`,
/// [`DmaBufRequest::alignment`]).
#[cfg(unix)]
fn answer_alignment(request: &DmaBufRequest, chunks: usize) -> u64 {
    match chunks {
        1 => request.one_chunk_alignment.max(1),
        _ => request.alignment.max(1),
    }
}

/// `chunks` as the runtime takes them ([`Piece::of`]), where they answer `request` as the
/// [`DmaBufSource`] contract allows on a host of `page`-byte pages ([`answers`]), each a dma-buf
/// of its own where the request takes no piece of a shared file; `None` otherwise, every
/// descriptor closed or let go of.
#[cfg(unix)]
fn taken(chunks: Vec<DmaBufChunk>, request: &DmaBufRequest, page: u64) -> Option<Vec<Piece>> {
    if !answers(&chunks, request, page) {
        return None;
    }
    let pieces: Vec<Piece> = chunks.into_iter().map(Piece::of).collect();
    (request.shared_files || pieces.iter().all(Piece::own)).then_some(pieces)
}

/// The bytes of the dma-buf `fd`; `None` where it is no dma-buf (its file is not on the kernel's
/// dma-buf file system).
#[cfg(target_os = "linux")]
fn dma_buf_length(fd: BorrowedFd<'_>) -> Option<u64> {
    use std::os::fd::AsRawFd;
    /// `DMA_BUF_MAGIC` of Linux's `include/uapi/linux/magic.h`.
    const DMA_BUF_MAGIC: u64 = 0x444d_4142;
    let mut stat = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: the descriptor is open for the call, and `fstatfs` writes the structure.
    if unsafe { libc::fstatfs(fd.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: the call succeeded, so it wrote the structure.
    let stat = unsafe { stat.assume_init() };
    // The field's type differs between architectures and C libraries.
    if i128::from(stat.f_type) != i128::from(DMA_BUF_MAGIC) {
        return None;
    }
    // SAFETY: a dma-buf answers `SEEK_END` with its size (`dma_buf_llseek`), and nothing reads
    // through its file position.
    let end = unsafe { libc::lseek(fd.as_raw_fd(), 0, libc::SEEK_END) };
    u64::try_from(end).ok()
}

/// No file is a dma-buf on this system.
#[cfg(all(unix, not(target_os = "linux")))]
fn dma_buf_length(_fd: BorrowedFd<'_>) -> Option<u64> {
    None
}

/// The nanoseconds since `began`.
#[cfg(unix)]
fn nanos_since(began: Instant) -> u64 {
    u64::try_from(began.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// How long one wait for the sparse binding lasts before it is asked again.
#[cfg(unix)]
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// The handle type of every import of a dma-buf, and of every buffer and image made for one.
const DMA_BUF: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;

#[cfg(unix)]
impl Device {
    /// The `VkBuffer` of a no-copy buffer of `length` bytes at `pointer`, made for dma-bufs and
    /// bound in the imports of the chunks `source` makes of `window`'s pages, as memory of
    /// `class`, for `call`: the buffer and its placement, which keeps the imports and the source.
    /// The source is asked once the device is known to import dma-bufs for such a buffer, with
    /// no lock of the runtime's held, and never again here. Where `keep`, a descriptor of each
    /// dma-buf is kept too, which imports them again once the device released the import while
    /// idle (`host_residency.rs`): `None` where the host gives no descriptor more, or the device
    /// keeps as many as it may, the import then kept imported and counted
    /// (`WorkCounts::host_imports_kept_for_fds`). Refused, everything made released, as the
    /// module says.
    ///
    /// # Safety
    ///
    /// The window's pages are host memory of this process that stays mapped until the placement
    /// is freed (`HostMemory::new`), and the chunks the source returns hold exactly those pages
    /// (`HostMemory::with_dma_buf`). Where `sole`, the buffer is the only buffer ever bound in
    /// the imports, and an image is bound there only where they are dedicated to nothing
    /// ([`Device::chunks_buffer`]).
    pub(super) unsafe fn import_from_source(
        &self,
        call: &'static str,
        importer: &Importer,
        pointer: NonNull<u8>,
        length: u64,
        window: &ImportWindow,
        source: DmaBufSource,
        class: MemoryClass,
        (keep, sole): (bool, bool),
    ) -> Result<(HostBuffer, Placement, Option<Again>), Error> {
        if !self.imports_dma_buf_buffers(vk::BufferCreateFlags::empty()) {
            return Err(not_importable());
        }
        // The window starts `window.offset` bytes before the pointer, in the pages it touches.
        let start = NonNull::new(pointer.as_ptr().wrapping_sub(window.offset as usize))
            .ok_or_else(not_importable)?;
        let granule = match self.sparse_granule(importer) {
            Some(granule) if granule < window.size => granule,
            _ => window.size,
        };
        let one_chunk_alignment = self.plain_alignment(importer);
        let request = DmaBufRequest {
            start,
            length: window.size,
            granule,
            // Several chunks bind a sparse buffer, whose alignment the granule is a multiple of
            // (`Device::sparse_granule`); one binds a plain buffer.
            alignment: match granule < window.size {
                true => one_chunk_alignment.max(granule),
                false => one_chunk_alignment,
            },
            one_chunk_alignment,
            shared_files: true,
        };
        let chunks = source(request).map_err(|_| not_importable())?;
        let pieces = (taken(chunks, &request, importer.page))
            .ok_or_else(|| invalid(call, Problem::DmaBuf))?;
        // The pieces again, kept where the device may release the import: a duplicate of each
        // descriptor of its own (the import takes the originals), a clone of each shared file's
        // `Arc`.
        let kept: Option<Vec<Piece>> = keep
            .then(|| pieces.iter().map(|piece| piece.again().ok()).collect())
            .flatten();
        let kept_source = kept.as_ref().map(|_| Arc::clone(&source));
        let mut pages = DmaBufPages::new(self, source, request, false);
        // SAFETY: the caller's.
        let (buffer, bound) = unsafe {
            self.chunks_buffer(importer, pieces, &mut pages, length, window, (class, sole))
        }?;
        let counts = &self.shared().counts;
        counts.dma_buf_imported(pages.made);
        if pages.bound.is_some() {
            counts.dma_buf_sparse_buffer_made();
        }
        let again = kept.zip(kept_source).and_then(|(pieces, source)| {
            Again::dma_bufs(self, pieces, source, request, (length, *window, class))
        });
        if keep && again.is_none() {
            pages.kept_for_fds = true;
            counts.host_import_kept_for_fds(true);
        }
        let placement = Placement::dma_buf(Arc::new(pages), bound, window.size, start);
        Ok((buffer, placement, again))
    }

    /// `span`, host pages of a guest RAM (`host_ram.rs`), imported as the one dma-buf of them
    /// that the RAM's `source` makes: a chunk of the RAM or an object's run inside one, asked for
    /// with a granule of its whole length, so that one import holds it; or, where `sole`, a span
    /// whose buffer is the only resource ever bound in its import ([`Device::chunks_buffer`]: the
    /// page a use of the RAM asks the driver with, [`Device::ram_takes_dma_bufs`]). The `VkBuffer`
    /// over the whole span, made for dma-bufs and bound to the import at its start, and its
    /// placement, which keeps the import and the source. Refused, everything made released, as
    /// [`Device::import_from_source`] is, where the device makes no such buffer, the source fails
    /// or answers more than one chunk, or the driver refuses the dma-buf: each of the last three
    /// counted (`WorkCounts::host_ram_dma_buf_imports_failed`). A refusal that lasts
    /// ([`lasting_refusal`]: the driver's of the dma-buf itself, in every memory type and form the
    /// import tries, `memory.rs`'s `import_dma_buf`) is remembered for the device's life
    /// (`RamSource::refuse_dma_bufs`): from then on every span is refused here unasked and
    /// uncounted, and takes its pointer, so that no later chunk pays a dma-buf, a failed import
    /// and a descriptor for nothing. Memory that ran out is that span's alone, as the source's
    /// failures are: a later span asks again.
    ///
    /// # Safety
    ///
    /// `span` is pages of the RAM's ranges, which stay mapped while the placement holds the
    /// import, and the source's dma-bufs hold exactly those pages (`HostRam::with_dma_buf`).
    pub(super) unsafe fn ram_span_as_dma_buf(
        &self,
        importer: &Importer,
        span: Range<u64>,
        source: &DmaBufSource,
        sole: bool,
    ) -> Result<(vk::Buffer, Placement), Error> {
        let shared = self.shared();
        if !importer.imports_dma_bufs()
            || !self.imports_dma_buf_buffers(vk::BufferCreateFlags::empty())
            || shared.host_ram.refuses_dma_bufs()
        {
            return Err(not_importable());
        }
        let start = NonNull::new(span.start as *mut u8).ok_or_else(not_importable)?;
        let length = span.end - span.start;
        let request = DmaBufRequest {
            start,
            length,
            granule: length,
            alignment: importer.page,
            one_chunk_alignment: importer.page,
            shared_files: false,
        };
        let answered = source(request).map(|chunks| match chunks.len() {
            1 => taken(chunks, &request, importer.page),
            _ => None,
        });
        let pieces = match answered {
            Ok(Some(pieces)) => pieces,
            answered => {
                shared.counts.host_ram_dma_buf_import_failed();
                return Err(match answered {
                    Ok(_) => invalid("use_host_ram", Problem::DmaBuf),
                    Err(_) => not_importable(),
                });
            }
        };
        // SAFETY: the caller's.
        let imported = unsafe { self.ram_span_import(importer, pieces, request, source, sole) };
        if let Err(error) = &imported {
            shared.counts.host_ram_dma_buf_import_failed();
            if lasting_refusal(error) {
                shared.host_ram.refuse_dma_bufs();
            }
        }
        imported
    }

    /// The driver's import of `pieces`, the one dma-buf of its own `source` answered `request`
    /// with ([`Device::ram_span_as_dma_buf`], whose `sole` this is): the `VkBuffer` over it and
    /// its placement.
    ///
    /// # Safety
    ///
    /// As [`Device::ram_span_as_dma_buf`]'s, for the request's span.
    unsafe fn ram_span_import(
        &self,
        importer: &Importer,
        pieces: Vec<Piece>,
        request: DmaBufRequest,
        source: &DmaBufSource,
        sole: bool,
    ) -> Result<(vk::Buffer, Placement), Error> {
        let length = request.length;
        let window = ImportWindow {
            start: request.start.as_ptr() as u64,
            size: length,
            offset: 0,
        };
        let mut pages = DmaBufPages::new(self, Arc::clone(source), request, false);
        // SAFETY: the caller's.
        let (buffer, bound) = unsafe {
            self.chunks_buffer(
                importer,
                pieces,
                &mut pages,
                length,
                &window,
                (MemoryClass::Shared, sole),
            )
        }?;
        self.shared().counts.dma_buf_imported(pages.made);
        let placement = Placement::dma_buf(Arc::new(pages), bound, length, request.start);
        Ok((buffer.raw, placement))
    }

    /// Whether the driver takes the dma-bufs a guest RAM's `source` makes of its pages
    /// (`host_ram.rs`'s `Plan::listed`, whose chunks are those dma-bufs): asked of it by an import
    /// of `page`, the RAM's first host page, as one ([`Device::ram_span_as_dma_buf`], which counts
    /// a failure and remembers a refusal that lasts), released at once. The import takes the form
    /// a chunk's does, dedicated to nothing, and where the driver refuses that, dedicated to the
    /// page's buffer, the only resource bound in it (`memory.rs`'s `DmaBufFor::Buffer`): a driver
    /// that refuses one page's dma-buf in the first form alone (NVIDIA's 615.71.09, which takes a
    /// 64 KiB one so, and a 64 MiB chunk is longer) is no driver that refuses the RAM's dma-bufs.
    /// Not where the device imports no dma-buf or makes no buffer for one, or the driver refused a
    /// dma-buf of a RAM's source before (`RamSource::refuses_dma_bufs`), the source unasked; taken
    /// where the source alone failed the page, or memory ran out for it, those the page's alone.
    /// The page's dma-buf pins it, and udmabuf allocates it in the guest's file where the guest
    /// never wrote it.
    ///
    /// # Safety
    ///
    /// As [`Device::ram_span_as_dma_buf`]'s, for `page`.
    pub(super) unsafe fn ram_takes_dma_bufs(
        &self,
        importer: &Importer,
        page: Range<u64>,
        source: &DmaBufSource,
    ) -> bool {
        let refused = || self.shared().host_ram.refuses_dma_bufs();
        if !importer.imports_dma_bufs()
            || !self.imports_dma_buf_buffers(vk::BufferCreateFlags::empty())
            || refused()
        {
            return false;
        }
        // SAFETY: the caller's.
        match unsafe { self.ram_span_as_dma_buf(importer, page, source, true) } {
            Ok((raw, placement)) => {
                self.discard_import(raw, placement);
                true
            }
            Err(_) => !refused(),
        }
    }

    /// `runs`, host pages of a guest RAM (`host_ram.rs`: by start, sharing no page, each inside
    /// one chunk), imported as the one dma-buf of them, their pages one after another, that the
    /// RAM's `source` of run lists makes (`HostRam::with_dma_buf_of_runs`): the `VkBuffer` over
    /// all their bytes, made for dma-bufs and bound to the import at its start, and its
    /// placement, which keeps the import. Only a copy's commands name the buffer: no object is
    /// placed in it, so it is the only resource bound in the import ([`Device::chunks_buffer`]).
    /// Refused, everything made released, as [`Device::ram_span_as_dma_buf`] is, where the device
    /// makes no such buffer, the source fails or answers a chunk of another length, or the driver
    /// refuses the dma-buf; and, the source unasked, where the driver refused a dma-buf of a
    /// RAM's source before (`RamSource::refuses_dma_bufs`). A refusal of the list's that lasts
    /// ([`lasting_refusal`]) is remembered so too, for the device's life; memory that ran out and
    /// the source's failures are that list's alone.
    ///
    /// # Safety
    ///
    /// The runs are pages of the RAM's ranges, which stay mapped while the placement holds the
    /// import, and the source's dma-buf holds exactly those pages, in order.
    pub(super) unsafe fn ram_runs_as_dma_buf(
        &self,
        importer: &Importer,
        runs: &[Range<u64>],
        source: &DmaBufRunsSource,
    ) -> Result<(vk::Buffer, Placement), Error> {
        let shared = self.shared();
        if !importer.imports_dma_bufs()
            || !self.imports_dma_buf_buffers(vk::BufferCreateFlags::empty())
            || shared.host_ram.refuses_dma_bufs()
        {
            return Err(not_importable());
        }
        let requests: Option<Vec<DmaBufRequest>> = (runs.iter())
            .map(|run| {
                let start = NonNull::new(run.start as *mut u8)?;
                let length = run.end.checked_sub(run.start)?;
                Some(DmaBufRequest {
                    start,
                    length,
                    granule: length,
                    alignment: importer.page,
                    one_chunk_alignment: importer.page,
                    shared_files: false,
                })
            })
            .collect();
        let requests = requests.ok_or_else(not_importable)?;
        let start = requests.first().ok_or_else(not_importable)?.start;
        let length: u64 = requests.iter().map(|request| request.length).sum();
        let chunk = source(&requests).map_err(|_| not_importable())?;
        let whole = DmaBufRequest {
            start,
            length,
            granule: length,
            alignment: importer.page,
            one_chunk_alignment: importer.page,
            shared_files: false,
        };
        let pieces = taken(vec![chunk], &whole, importer.page)
            .ok_or_else(|| invalid("use_host_ram", Problem::DmaBuf))?;
        let window = ImportWindow {
            start: start.as_ptr() as u64,
            size: length,
            offset: 0,
        };
        let mut pages = DmaBufPages::of_runs(self, start);
        // SAFETY: the caller's.
        let (buffer, bound) = unsafe {
            self.chunks_buffer(
                importer,
                pieces,
                &mut pages,
                length,
                &window,
                (MemoryClass::Shared, true),
            )
        }
        .inspect_err(|error| {
            if lasting_refusal(error) {
                shared.host_ram.refuse_dma_bufs();
            }
        })?;
        shared.counts.dma_buf_imported(pages.made);
        let placement = Placement::dma_buf(Arc::new(pages), bound, length, start);
        Ok((buffer.raw, placement))
    }

    /// The pages of a released import made again from the pieces `kept` of it, which `source`
    /// made for `request` (`host_residency.rs`), as `again` made it: a new `VkBuffer` made and
    /// bound as the first was, over imports of the same dma-bufs (a shared file's import live
    /// still, or made again), so over the pages they hold whatever the address maps now, at the
    /// same offsets in the same memory type. Never calls the source. Refused (out of device
    /// memory, or the device lost) where the device cannot import them again, or lays them
    /// otherwise.
    pub(super) fn import_dma_bufs_again(
        &self,
        kept: &[Piece],
        source: &DmaBufSource,
        request: DmaBufRequest,
        again: &Again,
    ) -> Result<Pages, Error> {
        if self.is_lost() {
            return Err(self.lost_refusal());
        }
        let refused = |_| Error::OutOfMemory(MemoryKind::Device);
        let shared = self.shared();
        let importer = shared
            .host_import
            .as_ref()
            .map_err(|_| internal("a dma-buf import made again on a device that imports none"))?;
        let pieces = (kept.iter())
            .map(|piece| {
                piece
                    .again()
                    .map_err(|_| Error::OutOfMemory(MemoryKind::Host))
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let mut pages = DmaBufPages::new(self, Arc::clone(source), request, false);
        let window = &again.window;
        // SAFETY: the dma-bufs are those the first import took, which hold the pages it was
        // made over for as long as they are open (the import holds them). Kept only by a
        // no-copy buffer's own import (`Device::import_from_source`'s `keep`), its buffer the
        // only resource bound in its own pieces' imports, as the new one is.
        let (buffer, bound) = unsafe {
            self.chunks_buffer(
                importer,
                pieces,
                &mut pages,
                again.length,
                window,
                (again.class, true),
            )
        }
        .map_err(|error| match error {
            Error::DeviceLost => Error::DeviceLost,
            other => refused(other),
        })?;
        let start = request.start;
        let placement = Placement::dma_buf(Arc::new(pages), bound, window.size, start);
        let made = self.pages_of(
            buffer.raw,
            placement,
            window.offset,
            buffer.base,
            buffer.size - buffer.base,
        );
        Ok(made)
    }

    /// The `VkBuffer` of a no-copy buffer of `length` bytes over `window`'s pages from their
    /// first byte, its bytes `window.offset` bytes into it whatever the pieces, made for
    /// dma-bufs, each of `pieces` imported or joined (held in `pages`) as memory of `class` and
    /// bound at its offset from the window's start, from its offset in its import: one piece by
    /// `vkBindBufferMemory` to a plain buffer, several by a sparse binding of a `SPARSE_BINDING`
    /// one (`Device::bind_sparse_buffer`, noted in `pages`). The buffer and where it is bound
    /// ([`BoundIn`]). A dma-buf of its own is imported dedicated to nothing; where `sole` (the
    /// caller binds no other buffer in its import, ever, and an image only where it is dedicated
    /// to nothing: [`DmaBufPages::dedicated_to_buffer`]) and the buffer is a plain one, an import
    /// the driver refuses so is made dedicated to the buffer (`memory.rs`'s
    /// `DmaBufFor::Buffer`), and noted so in `pages`. A piece of a shared file lies in the device's one import of the file
    /// ([`Device::file_import`]), never dedicated: where the driver refuses that, the buffer is
    /// refused as for a refused chunk. Refused, the buffer released, where the bytes padded to 16
    /// run past the pieces, the buffer's memory does not fit them or its alignment misses their
    /// boundaries or offsets, a piece's types take none of the class's the buffer takes, the
    /// driver refuses a piece, or the binding fails.
    ///
    /// # Safety
    ///
    /// As [`Device::import_from_source`]'s; `pieces` meet [`chunks_cover`].
    unsafe fn chunks_buffer(
        &self,
        importer: &Importer,
        pieces: Vec<Piece>,
        pages: &mut DmaBufPages,
        length: u64,
        window: &ImportWindow,
        (class, sole): (MemoryClass, bool),
    ) -> Result<(HostBuffer, BoundIn), Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let lengths: Vec<u64> = pieces.iter().map(Piece::length).collect();
        let size: u64 = lengths.iter().sum();
        // A range the binding layer pads to 16 bytes from the bytes stays inside the buffer, as
        // a pointer's import holds it (`based_size`).
        if based_size(window.offset, length).is_none_or(|padded| padded > size) {
            return Err(not_importable());
        }
        let sparse = lengths.len() > 1;
        let flags = match sparse {
            true => vk::BufferCreateFlags::SPARSE_BINDING,
            false => vk::BufferCreateFlags::empty(),
        };
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(DMA_BUF);
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .flags(flags)
                .size(size)
                .usage(usage(shared.device_address))
                .push_next(&mut external),
        );
        // SAFETY: the device is live and `info` and its chain valid; a sparse one has
        // `sparseBinding` (`Device::sparse_granule`).
        let raw = unsafe { crate::creation::buffer(raw_device, &info) }.map_err(vulkan)?;
        shared.counts.buffer_made(info.usage);
        // SAFETY: `raw` is a live buffer of this device.
        let requirements = unsafe { raw_device.get_buffer_memory_requirements(raw) };
        let bound = (|| {
            let alignment = requirements.alignment.max(1);
            // Each piece binds at its offset in its import, which the alignment must divide
            // (the request's alignment holds it: Vulkan keeps a buffer's alignment the same for
            // every buffer of its usage and flags).
            let aligned = (pieces.iter()).all(|piece| piece.offset().is_multiple_of(alignment));
            let fits = match sparse {
                // Each piece binds whole at its offset in the buffer, which the alignment must
                // divide, as it must divide each piece's length.
                true => lengths
                    .iter()
                    .all(|length| length.is_multiple_of(alignment)),
                false => requirements.size <= size,
            };
            if !aligned || !fits {
                return Err(not_importable());
            }
            // A plain buffer bound at its piece's offset, its requirements' size within it.
            let dedicated_to = (sole && !sparse).then_some(raw);
            let memories = self.import_chunks(
                importer,
                pieces,
                pages,
                &requirements,
                (class, dedicated_to),
            )?;
            if !sparse {
                let one = memories[0];
                // SAFETY: the buffer is unbound, made for dma-bufs; the memory is of a type it
                // takes and holds its requirements from the piece's offset, a multiple of their
                // alignment.
                unsafe { crate::creation::bind_buffer(raw_device, raw, one.memory, one.offset) }
                    .map_err(vulkan)?;
                return Ok(one);
            }
            let mut offset = 0;
            let binds: Vec<vk::SparseMemoryBind> = (memories.iter().zip(&lengths))
                .map(|(at, &length)| {
                    let bind = vk::SparseMemoryBind::default()
                        .resource_offset(offset)
                        .size(length)
                        .memory(at.memory)
                        .memory_offset(at.offset);
                    offset += length;
                    bind
                })
                .collect();
            pages.bound = Some(self.bind_sparse_buffer(raw, &binds)?);
            Ok(BoundIn {
                memory: vk::DeviceMemory::null(),
                offset: 0,
                type_index: memories[0].type_index,
            })
        })();
        match bound {
            // The bytes lie `window.offset` bytes into the buffer, which is bound from the
            // window's first byte; it holds the pages, and its padding past them is no one's.
            Ok(bound) => {
                let buffer = HostBuffer {
                    raw,
                    base: window.offset,
                    size: window.size,
                    requirements,
                };
                Ok((buffer, bound))
            }
            Err(error) => {
                // SAFETY: no command uses the buffer; a failed binding left nothing pending.
                unsafe { release::object(raw_device, raw) };
                Err(error)
            }
        }
    }

    /// Each of `pieces`, in order, as memory of `class` of a type both it and a buffer of
    /// `requirements` take, held in `pages` (which frees or lets go of them), with its type and
    /// its offset there: a dma-buf of its own imported, where `dedicated_to` names the buffer
    /// (one piece, its import the buffer's alone), dedicated to it where the driver refuses the
    /// import dedicated to nothing, counted (`WorkCounts::dma_buf_buffer_imports_dedicated`); a
    /// piece of a shared file in the device's import of the file ([`Device::file_import`]).
    /// Refused (`Unsupported`, `HostMemory`) where a piece's types take none of the class's the
    /// buffer takes, or the driver refuses it; the descriptors not yet imported are closed then.
    fn import_chunks(
        &self,
        importer: &Importer,
        pieces: Vec<Piece>,
        pages: &mut DmaBufPages,
        requirements: &vk::MemoryRequirements,
        (class, dedicated_to): (MemoryClass, Option<vk::Buffer>),
    ) -> Result<Vec<BoundIn>, Error> {
        let shared = self.shared();
        let mut imported = Vec::with_capacity(pieces.len());
        for piece in pieces {
            let (fd, length) = match piece {
                Piece::Own { fd, length } => (fd, length),
                Piece::Shared { file, offset, .. } => {
                    let (memory, type_index) =
                        self.file_import(importer, &file, pages, requirements, class)?;
                    imported.push(BoundIn {
                        memory,
                        offset,
                        type_index,
                    });
                    continue;
                }
            };
            let allowed = (requirements.memory_type_bits, class);
            let purpose = dedicated_to.map_or(DmaBufFor::Shared, DmaBufFor::Buffer);
            // SAFETY: the importer is this device's, the descriptor a dma-buf, and the buffer it
            // may be dedicated to this device's, unbound and made for dma-bufs.
            let made = unsafe { self.import_timed(importer, fd.as_fd(), length, allowed, purpose) }
                .map_err(import_refused)?;
            let (memory, type_index, dedicated) = made.map_err(|failure| match failure {
                ImportFailure::NoMemoryType => not_importable(),
                ImportFailure::Vulkan(error) => import_refused(error),
            })?;
            if dedicated {
                pages.dedicated_to_buffer = true;
                shared.counts.dma_buf_buffer_import_dedicated();
            }
            pages.hold(memory);
            imported.push(BoundIn {
                memory,
                offset: 0,
                type_index,
            });
        }
        Ok(imported)
    }

    /// The device's import of the shared dma-buf `file`, held in `pages` (which lets go of it),
    /// and its memory type, which a buffer of `requirements` takes: the live one, joined
    /// (`WorkCounts::dma_buf_pieces_joined`), or else a new import of the whole dma-buf
    /// ([`import_file`](Self::import_file)) entered in the device's map ([`DmaBufFiles`]),
    /// counted live (`WorkCounts::dma_buf_shared_imports_live`), and kept by the map where a
    /// handle keeps the file ([`SharedDmaBufImport`]); where another thread entered one of the
    /// same file meanwhile, that one is joined and this one freed. Refused (`Unsupported`,
    /// `HostMemory`) where the live import's type is none the buffer takes, or as
    /// [`import_file`](Self::import_file) refuses.
    fn file_import(
        &self,
        importer: &Importer,
        file: &Arc<OwnedFd>,
        pages: &mut DmaBufPages,
        requirements: &vk::MemoryRequirements,
        class: MemoryClass,
    ) -> Result<(vk::DeviceMemory, u32), Error> {
        let shared = self.shared();
        // A join from `began`: the look in the map that found the live import, and its hold.
        let join = |pages: &mut DmaBufPages, live: Arc<DmaBufFile>, began: Instant| {
            let at = (live.memory, live.type_index);
            pages.files.push(live);
            if requirements.memory_type_bits & (1 << at.1) == 0 {
                return Err(not_importable());
            }
            shared.counts.dma_buf_piece_joined(nanos_since(began));
            Ok(at)
        };
        let began = Instant::now();
        if let Some(live) = shared.dma_buf_files.live(file) {
            return join(pages, live, began);
        }
        let made = self.import_file(importer, file, requirements.memory_type_bits, class)?;
        let at = (made.memory, made.type_index);
        match self.enter_file(made) {
            Ok(entered) => {
                pages.files.push(entered);
                pages.made += 1;
                Ok(at)
            }
            // Another thread's import, entered meanwhile, held as `enter_file` freed this one.
            Err(live) => join(pages, live, Instant::now()),
        }
    }

    /// A new import of the shared dma-buf `file`, whole, dedicated to nothing (several buffers
    /// bind in it), as memory of `class` of a type both it and a buffer of `allowed` types take,
    /// made with no lock held. The driver takes a duplicate of the file's descriptor, the `Arc`'s
    /// staying open while the import holds it. Refused (`Unsupported`, `HostMemory`) where the
    /// file is no dma-buf, its types take none of the class's among `allowed`, or the driver
    /// refuses it.
    fn import_file(
        &self,
        importer: &Importer,
        file: &Arc<OwnedFd>,
        allowed: u32,
        class: MemoryClass,
    ) -> Result<DmaBufFile, Error> {
        let size = dma_buf_length(file.as_fd()).ok_or_else(not_importable)?;
        let purpose = DmaBufFor::Shared;
        // SAFETY: the importer is this device's, and the descriptor a dma-buf.
        let made =
            unsafe { self.import_timed(importer, file.as_fd(), size, (allowed, class), purpose) }
                .map_err(import_refused)?;
        let (memory, type_index, _) = made.map_err(|failure| match failure {
            ImportFailure::NoMemoryType => not_importable(),
            ImportFailure::Vulkan(error) => import_refused(error),
        })?;
        Ok(DmaBufFile {
            file: Arc::clone(file),
            memory,
            type_index,
        })
    }

    /// `made`, a new import of a shared file ([`import_file`](Self::import_file)), entered in the
    /// device's map, a hold of it, counted live (and kept, where the map keeps it); or, where
    /// another thread entered one of the same file meanwhile, a hold of that one, `made` freed.
    fn enter_file(&self, made: DmaBufFile) -> Result<Arc<DmaBufFile>, Arc<DmaBufFile>> {
        let shared = self.shared();
        match shared.dma_buf_files.enter(made) {
            Ok((entered, kept)) => {
                shared.counts.dma_buf_memory_imported(false);
                shared.counts.dma_buf_shared_import_made();
                if kept {
                    shared.counts.dma_buf_shared_import_kept(true);
                }
                Ok(entered)
            }
            Err((live, made)) => {
                // SAFETY: nothing is bound to the import, made just now, and no command uses it.
                unsafe { release::object(shared.vk(), made.memory) };
                Err(live)
            }
        }
    }

    /// The import of the dma-buf `fd`, `size` bytes, as memory of `class` of a type both the
    /// driver takes it as (`vkGetMemoryFdPropertiesKHR`) and `allowed` holds, in `purpose`'s
    /// forms (`memory.rs`'s `import_dma_buf`): timed, made or failed
    /// (`WorkCounts::dma_buf_import_nanos`). `Err` where the driver refuses the dma-buf's types.
    ///
    /// # Safety
    ///
    /// `importer` is this device's, `fd` a dma-buf, and what `purpose` names this device's,
    /// unbound and made for dma-bufs.
    unsafe fn import_timed(
        &self,
        importer: &Importer,
        fd: BorrowedFd<'_>,
        size: u64,
        (allowed, class): (u32, MemoryClass),
        purpose: DmaBufFor,
    ) -> Result<Result<(vk::DeviceMemory, u32, bool), ImportFailure>, vk::Result> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let began = Instant::now();
        // SAFETY: the caller's.
        let made = unsafe { importer.dma_buf_types(raw_device, fd) }.map(|types| {
            let allowed = (types & allowed, class);
            memory::import_dma_buf(&shared.memory, raw_device, fd, size, allowed, purpose)
        });
        shared.counts.dma_buf_import_took(nanos_since(began));
        made
    }

    /// The multiple of which a lone chunk's offset in its file lies for a buffer's request
    /// ([`DmaBufRequest::one_chunk_alignment`]): the host page, or the memory alignment of a
    /// plain buffer made for dma-bufs where that is larger (asked once,
    /// [`Importer::dma_buf_alignment`]: Vulkan keeps it the same for every buffer of the same
    /// usage, flags and handle types). Each a power of two, so the larger is a multiple of the
    /// other, as the granule, a multiple of the page, is of both where a request may split.
    fn plain_alignment(&self, importer: &Importer) -> u64 {
        let plain = *importer.dma_buf_alignment.get_or_init(|| {
            self.dma_buf_buffer_requirements(importer.page, vk::BufferCreateFlags::empty())
                .map(|requirements| requirements.alignment)
        });
        importer.page.max(plain.unwrap_or(1))
    }

    /// The granule a dma-buf source's chunks split at, where the device assembles a buffer of
    /// several: it has `sparseBinding`, its queue family binds sparse memory, it imports
    /// dma-bufs for a sparse buffer without a dedicated allocation, and such a buffer takes every
    /// memory type a plain buffer made for dma-bufs takes that an import may be of
    /// (`Memory::dma_buf_importable`); then the alignment of such a buffer (of the host page),
    /// rounded up to the host page. `None` elsewhere, so that a request is one chunk. A dma-buf's
    /// own types are known only from its descriptor, once the source made it, so the split asks
    /// for none it may not import whole: wherever one chunk of the pages imports as a type the
    /// plain buffer takes, each of several chunks imports as that type, which the sparse buffer
    /// takes too. NVIDIA 615.71.09's sparse buffers (an RTX 5080 Laptop GPU) take none of the
    /// types its dma-bufs of host pages import as (dma-buf imports otherwise work there): split,
    /// the chunks imported as no type the buffer took, and the buffer was a copy of the pages
    /// where one chunk imports. On 620.70.06 (an RTX 2080, an RTX 3090) an earlier condition
    /// fails, so its requests are one chunk before any type is compared, as on ANV (Mesa 26.2.6
    /// on the Iris Xe of RPL-P), which has `sparseBinding` and a queue family that binds sparse
    /// memory but imports no dma-buf for a sparse buffer (`imports_dma_buf_buffers`). The
    /// granule is the sparse buffer's alignment because Vulkan binds a sparse buffer's memory in
    /// multiples of it (`VUID-VkSparseMemoryBind-resourceOffset-09491`; the alignment is the
    /// sparse block size, "Sparse Buffer and Fully-Resident Image Block Size"). Asked of the
    /// device once ([`Importer::sparse_granule`]).
    fn sparse_granule(&self, importer: &Importer) -> Option<u64> {
        *importer.sparse_granule.get_or_init(|| {
            let shared = self.shared();
            let flags = vk::BufferCreateFlags::SPARSE_BINDING;
            if !shared.sparse.binds() || !self.imports_dma_buf_buffers(flags) {
                return None;
            }
            let sparse = self.dma_buf_buffer_requirements(importer.page, flags)?;
            let plain =
                self.dma_buf_buffer_requirements(importer.page, vk::BufferCreateFlags::empty())?;
            let importable = lock(&shared.memory).dma_buf_importable();
            if plain.memory_type_bits & importable & !sparse.memory_type_bits != 0 {
                return None;
            }
            let alignment = sparse.alignment;
            alignment
                .max(1)
                .checked_next_multiple_of(importer.page)
                .filter(|granule| granule.is_multiple_of(alignment.max(1)))
        })
    }

    /// The memory requirements of a buffer of `bytes` made for dma-bufs with `flags` (a sparse one
    /// only where the device has `sparseBinding`); `None` where the driver refuses to make it.
    fn dma_buf_buffer_requirements(
        &self,
        bytes: u64,
        flags: vk::BufferCreateFlags,
    ) -> Option<vk::MemoryRequirements> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(DMA_BUF);
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .flags(flags)
                .size(bytes)
                .usage(usage(shared.device_address))
                .push_next(&mut external),
        );
        // SAFETY: the device is live, has `sparseBinding` where `flags` asks for a sparse
        // buffer (the caller's), and `info` and its chain are valid.
        let raw = unsafe { crate::creation::buffer(raw_device, &info) }.ok()?;
        shared.counts.buffer_made(info.usage);
        // SAFETY: `raw` is a live buffer of this device.
        let requirements = unsafe { raw_device.get_buffer_memory_requirements(raw) };
        // SAFETY: the buffer is unused.
        unsafe { release::object(raw_device, raw) };
        Some(requirements)
    }

    /// Whether the device imports a dma-buf for a no-copy buffer's `VkBuffer` made with `flags`
    /// without a dedicated allocation (`vkGetPhysicalDeviceExternalBufferProperties`), as it must
    /// import host allocations for one (`PointerImport::new`). Vulkan imports a handle only of a
    /// type this query reports supported for import (`IMPORTABLE`,
    /// `VUID-VkImportMemoryFdInfoKHR-handleType-09862`), and binds imported memory to a sparse
    /// buffer only of a handle type the buffer was made for
    /// (`VUID-VkSparseMemoryBind-memory-02731`), so a buffer over several dma-bufs is made only
    /// where the query with `SPARSE_BINDING` reports them importable. RADV does; ANV (Mesa 26.2.6
    /// on the Iris Xe of RPL-P) answers features 0 with `SPARSE_BINDING` for dma-bufs, host
    /// allocations and opaque descriptors alike, and 0x6 (importable) for a plain buffer's
    /// dma-bufs.
    fn imports_dma_buf_buffers(&self, flags: vk::BufferCreateFlags) -> bool {
        let shared = self.shared();
        let info = vk::PhysicalDeviceExternalBufferInfo::default()
            .flags(flags)
            .usage(usage(shared.device_address))
            .handle_type(DMA_BUF);
        let mut external = vk::ExternalBufferProperties::default();
        // SAFETY: the physical device is the instance's; the query is core in Vulkan 1.1.
        unsafe {
            shared
                .instance
                .raw
                .get_physical_device_external_buffer_properties(
                    shared.physical,
                    &info,
                    &mut external,
                )
        };
        let features = external.external_memory_properties.external_memory_features;
        features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
            && !features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
    }
}

/// Where a buffer over dma-buf pages is bound ([`Device::chunks_buffer`]): its one import and
/// its offset there (`VK_NULL_HANDLE` and 0 for a sparse buffer over several), and the memory
/// type of its first import.
#[derive(Clone, Copy, Debug)]
#[cfg(unix)]
pub(super) struct BoundIn {
    pub(super) memory: vk::DeviceMemory,
    pub(super) offset: u64,
    pub(super) type_index: u32,
}

/// `DRM_FORMAT_MOD_LINEAR` of Linux's `drm_fourcc.h`: one plane of rows one after another, each
/// the plane's pitch from the last.
pub(super) const DRM_FORMAT_MOD_LINEAR: u64 = 0;

/// Where a texture's texels lie in dma-buf pages, for its image over them
/// ([`Device::dma_buf_image`]).
#[cfg(unix)]
pub(super) struct DmaBufTexels<'a> {
    /// The imports of the pages (a no-copy heap's), with the caller's source.
    pub(super) pages: &'a DmaBufPages,
    /// Their memory type, which the image's import takes too.
    pub(super) type_index: u32,
    /// The host address of the texels' first byte.
    pub(super) at: u64,
    /// The bytes from a row's first texel to the next row's.
    pub(super) row_pitch: u64,
    /// The bytes from a slice's first texel to the next slice's, for a plan of several layers.
    pub(super) slice_pitch: u64,
}

/// A `DRM_FORMAT_MOD_LINEAR` image with an explicit plane, unbound: the image, its memory
/// requirements and its plane's layout as the driver reports it (`Device::explicit_plane_image`).
#[cfg(unix)]
pub(super) type ExplicitPlane = (vk::Image, vk::MemoryRequirements, vk::SubresourceLayout);

/// A texture's texels in the runtime's own memory exported as a dma-buf (`exported_memory.rs`),
/// which an image reaches through a dedicated import of that memory's own dma-buf
/// ([`Device::exported_dma_buf_image_unbound`]).
#[cfg(unix)]
pub(super) struct ExportedTexels {
    /// The memory, allocated exportable as a dma-buf: a shared block or the buffer's dedicated
    /// allocation.
    pub(super) memory: vk::DeviceMemory,
    /// Its bytes, which its dma-buf holds.
    pub(super) size: u64,
    /// Its memory type, which the image's import takes too.
    pub(super) type_index: u32,
    /// The buffer's memory class, whose types the import ranks as the buffer's allocation did.
    pub(super) class: MemoryClass,
    /// The texels' first byte in the memory.
    pub(super) at: u64,
    /// The bytes from a row's first texel to the next row's.
    pub(super) row_pitch: u64,
    /// The bytes from a slice's first texel to the next slice's, for a plan of several layers.
    pub(super) slice_pitch: u64,
    /// The descriptor of the memory's payload, where a recovery carried the buffer into it
    /// imported again (`exported_memory.rs`'s `Buffer::carried_payload`): the dma-buf the image's
    /// import takes a clone of, as the memory exports none of its own.
    #[cfg(unix)]
    pub(super) payload: Option<crate::memory::Payload>,
}

/// A texture's image over dma-buf pages, made and bound ([`Device::dma_buf_image`]): its
/// `VkImage`, which the caller destroys, and the import of its pages, which it drops after.
#[cfg(unix)]
pub(super) struct DmaBufImage {
    pub(super) raw: vk::Image,
    pub(super) import: Arc<DmaBufPages>,
}

/// The layout the driver gives a `DRM_FORMAT_MOD_LINEAR` image of its own choosing
/// ([`Device::dma_buf_natural_layout`]): its plane's row and array pitches, and its memory's
/// bytes from the plane's first byte.
#[cfg(unix)]
pub(super) struct NaturalLayout {
    pub(super) row_pitch: u64,
    pub(super) slice_pitch: u64,
    pub(super) size: u64,
}

/// The one memory plane of an image of `DRM_FORMAT_MOD_LINEAR`, whose layout Vulkan reports
/// (`VUID-vkGetImageSubresourceLayout-tiling-09433`).
#[cfg(unix)]
pub(super) fn memory_plane_0() -> vk::ImageSubresource {
    vk::ImageSubresource {
        aspect_mask: vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
        mip_level: 0,
        array_layer: 0,
    }
}

/// Why the device made no image over a dma-buf of a texture's own pages
/// ([`Device::dma_buf_image_or_why`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DmaBufRefusal {
    /// The device makes no such image of the plan: the extensions, its levels, depth, samples or
    /// format, the modifier's features, or the format query's limits.
    Image,
    /// The driver refuses the plane at the texels' offset and pitches, lays it out otherwise, or
    /// renders an explicit plane's rows past its height.
    Pitch,
    /// The image's memory runs past the pages, or takes none of their memory type.
    #[cfg(unix)]
    Pages,
    /// The caller's source made no one dma-buf of the image's pages: it failed (a source of
    /// `udmabuf`s does where the pages are more runs of its file than one `udmabuf` takes), or it
    /// answered with other than one chunk. It `stands` where its error holds a
    /// [`DmaBufStandingRefusal`]: for every range that holds the one it refused, while the texture
    /// that asked is made.
    #[cfg(unix)]
    Source { stands: bool },
    /// The driver refuses the dma-buf, its import, or its binding.
    #[cfg(unix)]
    Import,
}

/// Why the driver made no `DRM_FORMAT_MOD_LINEAR` image of an explicit plane as asked
/// (`Device::explicit_plane_image`).
#[cfg(unix)]
pub(super) enum PlaneMiss {
    /// It refuses the layout: why, in words.
    Refused(String),
    /// It lays the plane out otherwise (`vkGetImageSubresourceLayout`): its answer, and the words.
    Answered(vk::SubresourceLayout, String),
}

#[cfg(unix)]
impl PlaneMiss {
    /// Why, in words.
    pub(super) fn words(self) -> String {
        match self {
            PlaneMiss::Refused(why) | PlaneMiss::Answered(_, why) => why,
        }
    }
}

/// The route a test takes for an image over dma-buf pages whose texels lie past their first page's
/// first byte (`DeviceShared::plane_route_forced`), whatever the driver answers of the explicit
/// plane at their offset.
#[cfg(all(test, unix))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlaneRoute {
    /// The driver's answers decide, as in the product.
    Answered = 0,
    /// The explicit plane at the texels' offset is taken as made, its layout unchecked: what such
    /// an image reads, on a driver that lays it out at another offset.
    Unchecked = 1,
    /// The explicit plane at the texels' offset is taken as laid out at offset 0, so the image of
    /// its plane at 0 bound at the offset is tried: that route, on a driver that answers the
    /// offset asked.
    AtOffset = 2,
}

/// An image over dma-buf pages made and not yet bound: its requirements, the request for its
/// pages, and where it binds in their import: at offset 0 of a dedicated one (`None`), or at its
/// texels' offset of one that is no dedicated allocation, its plane at offset 0 (`Some`:
/// [`Device::offset_bound_image`]).
#[cfg(unix)]
pub(super) struct Unbound {
    raw: vk::Image,
    requirements: vk::MemoryRequirements,
    request: DmaBufRequest,
    bound_at: Option<u64>,
}

#[cfg(unix)]
impl Unbound {
    /// The host addresses of the pages its bind asks the caller's source for: its request's.
    pub(super) fn asked(&self) -> Range<u64> {
        let start = self.request.start.as_ptr() as u64;
        start..start + self.request.length
    }

    /// Release the image, which no command uses and no memory is bound to, unbound.
    pub(super) fn release(self, device: &Device) {
        // SAFETY: the image is live and unbound, and no command uses it.
        unsafe { release::object(device.shared().vk(), self.raw) };
    }
}

impl Device {
    /// The image of `plan` (a heap texture's, viewed as `facts` says) over `texels`, where the
    /// texels lie in the caller's dma-buf pages: a `DRM_FORMAT_MOD_LINEAR` image
    /// (`VK_EXT_image_drm_format_modifier`) whose one plane starts at the texels' offset in their
    /// first page, rows `row_pitch` apart (slices `slice_pitch`), made for dma-bufs, and bound to
    /// a dedicated import (`VkMemoryDedicatedAllocateInfo`) of the one dma-buf the caller's source
    /// makes of exactly the pages its memory requirements cover (`DmaBufRequest { granule:
    /// length }`), of the pages' imports' memory type; counted (`WorkCounts::dma_buf_images`).
    /// Where the driver lays that plane out at another offset than the texels' (it answers
    /// `vkGetImageSubresourceLayout` so), the image's one plane starts at offset 0 instead, and it
    /// is bound at the texels' offset of an import of the pages that is no dedicated allocation,
    /// where the driver takes such a binding (`offset_bound_image`;
    /// `WorkCounts::dma_buf_images_at_offset`). It starts `UNDEFINED`, as Vulkan makes an image
    /// of external memory, for its first move to acquire from `VK_QUEUE_FAMILY_FOREIGN_EXT`
    /// (`FirstMove::acquired`).
    ///
    /// Why not, for the notice of the path taken instead (`shadowed-texture`), where the device
    /// makes none: it does not enable the extensions; the plan has several levels, depth or
    /// samples, or a mutable format with no list of its views; the modifier's tiling lacks a
    /// feature the plan's optimal tiling has for its usage; the format query refuses such an
    /// image, or the plan passes its limits; the driver refuses the plane layout, or lays it out
    /// otherwise and makes no image of its plane at offset 0 bound at the texels' offset (the
    /// image requires a dedicated allocation, its memory's alignment does not divide the offset,
    /// or the format query takes dma-bufs for it in a dedicated allocation alone); the image's
    /// memory runs past the pages or takes no memory of their type; the source fails or answers
    /// against its contract; the driver refuses the import. Everything made is released then.
    /// The source is called at most once, once the image is made and checked, with no lock of
    /// the runtime's held (the caller holds none).
    #[cfg(unix)]
    pub(super) fn dma_buf_image(
        &self,
        plan: &ImagePlan,
        facts: &ImageFacts<'_>,
        texels: &DmaBufTexels<'_>,
    ) -> Result<Result<DmaBufImage, String>, Error> {
        let made = self.dma_buf_image_or_why(plan, facts, texels)?;
        Ok(made.map_err(|(_, why)| why))
    }

    /// [`Device::dma_buf_image`], with the kind of each refusal ([`DmaBufRefusal`]), which a
    /// texture in place over a buffer's dma-buf pages is counted by (`in_place.rs`).
    #[cfg(unix)]
    fn dma_buf_image_or_why(
        &self,
        plan: &ImagePlan,
        facts: &ImageFacts<'_>,
        texels: &DmaBufTexels<'_>,
    ) -> Result<Result<DmaBufImage, (DmaBufRefusal, String)>, Error> {
        match self.dma_buf_image_unbound(plan, facts, texels)? {
            Ok(unbound) => self.bind_dma_buf_image(unbound, texels),
            Err(why) => Ok(Err(why)),
        }
    }

    /// The queue family the first move of an image over dma-buf pages acquires it for: the
    /// device's one family, or `VK_QUEUE_FAMILY_IGNORED` where its images are concurrent among
    /// several (`sharing.rs`), as Vulkan has an acquire of such an image name it.
    pub(super) fn foreign_acquirer(&self) -> u32 {
        match &self.shared().queue_families[..] {
            [family] => *family,
            _ => vk::QUEUE_FAMILY_IGNORED,
        }
    }

    /// What the device answers of [`Device::dma_buf_image`]'s image of `plan` before it makes
    /// one, by the checks [`dma_buf_image_unbound`](Self::dma_buf_image_unbound) runs first: the
    /// host's page and the formats it lists for its views
    /// ([`dma_buf_image_checks`](Self::dma_buf_image_checks)), and a driver that renders an
    /// explicit plane's rows past its height ([`explicit_plane_rows`](Self::explicit_plane_rows));
    /// why not, of the kind it is counted by.
    pub(super) fn dma_buf_unmade(
        &self,
        plan: &ImagePlan,
        facts: &ImageFacts<'_>,
    ) -> Result<Result<(u64, Vec<vk::Format>), (DmaBufRefusal, String)>, Error> {
        let checked = match self.dma_buf_image_checks(plan, facts)? {
            Ok(checked) => checked,
            Err(why) => return Ok(Err((DmaBufRefusal::Image, why))),
        };
        if let Some(why) = self.explicit_plane_rows(plan) {
            return Ok(Err((DmaBufRefusal::Pitch, why)));
        }
        Ok(Ok(checked))
    }

    /// The image of `plan` (a texture's in place, viewed as `facts` says) over texels at `at` of
    /// the memory of an import of their dma-buf pages dedicated to nothing, rows and slices
    /// `pitches` apart, to be bound there beside the buffer bound in that import
    /// (`texture/in_place_image.rs`): the `DRM_FORMAT_MOD_LINEAR` image made for dma-bufs whose one
    /// plane lies at offset 0 at those pitches ([`offset_bound_image`](Self::offset_bound_image)),
    /// unbound, with its memory requirements. It takes no dma-buf of the source's and no import of
    /// its own: Vulkan binds an image at an offset of memory it shares (`vkBindImageMemory`'s
    /// `memoryOffset`), so its texels are the import's bytes from `at`, which hold the pages'
    /// from their first. Why not, of the kind a refusal is counted by: the checks of
    /// [`dma_buf_unmade`](Self::dma_buf_unmade); then, of its plane's kind
    /// ([`DmaBufRefusal::Pitch`]), the driver refuses the plane or lays it out otherwise, the
    /// image requires a dedicated allocation, its memory's alignment does not divide `at`, or the
    /// image format query takes dma-bufs for it in a dedicated allocation alone.
    #[cfg(unix)]
    pub(super) fn in_import_image_unbound(
        &self,
        plan: &ImagePlan,
        facts: &ImageFacts<'_>,
        at: u64,
        pitches: (u64, u64),
    ) -> Result<Result<(vk::Image, vk::MemoryRequirements), (DmaBufRefusal, String)>, Error> {
        let view_formats = match self.dma_buf_unmade(plan, facts)? {
            Ok((_, view_formats)) => view_formats,
            Err(refused) => return Ok(Err(refused)),
        };
        let bound = self.offset_bound_image(plan, &view_formats, at, pitches)?;
        Ok(bound.map_err(|why| (DmaBufRefusal::Pitch, why)))
    }

    /// [`Device::dma_buf_image`]'s image made and checked, unbound, with the request for its
    /// pages; why not where the device makes none before the source is asked.
    #[cfg(unix)]
    pub(super) fn dma_buf_image_unbound(
        &self,
        plan: &ImagePlan,
        facts: &ImageFacts<'_>,
        texels: &DmaBufTexels<'_>,
    ) -> Result<Result<Unbound, (DmaBufRefusal, String)>, Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let (page, view_formats) = match self.dma_buf_unmade(plan, facts)? {
            Ok(checked) => checked,
            Err(refused) => return Ok(Err(refused)),
        };
        let offset = texels.at % page;
        let pitches = (texels.row_pitch, texels.slice_pitch);
        let (raw, requirements, bound_at) =
            match self.plane_image(plan, &view_formats, offset, pitches)? {
                Ok(made) => made,
                Err(why) => return Ok(Err((DmaBufRefusal::Pitch, why))),
            };
        let refuse = |kind: DmaBufRefusal, why: String| {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, raw) };
            Ok(Err((kind, why)))
        };
        // The pages from the texels' first, through the image's memory from where it binds.
        let first = texels.at - offset;
        let end = (bound_at.unwrap_or(0)).checked_add(requirements.size);
        let length = match end.and_then(|end| end.checked_next_multiple_of(page)) {
            Some(length) if texels.pages.holds(first, length) => length,
            _ => {
                return refuse(
                    DmaBufRefusal::Pages,
                    format!(
                        "its image's memory, {} bytes from {} bytes into the first page of its \
                         texels, runs past the pages the buffer's range touches",
                        requirements.size,
                        bound_at.unwrap_or(0)
                    ),
                )
            }
        };
        if requirements.memory_type_bits & (1 << texels.type_index) == 0 {
            return refuse(
                DmaBufRefusal::Pages,
                format!(
                    "its image takes none of the pages' memory type {}",
                    texels.type_index
                ),
            );
        }
        let Some(start) = NonNull::new(first as *mut u8) else {
            return refuse(
                DmaBufRefusal::Pages,
                "its texels lie in the null page".to_owned(),
            );
        };
        Ok(Ok(Unbound {
            raw,
            requirements,
            request: DmaBufRequest {
                start,
                length,
                granule: length,
                alignment: page,
                one_chunk_alignment: page,
                shared_files: false,
            },
            bound_at,
        }))
    }

    /// The `DRM_FORMAT_MOD_LINEAR` image of `plan` for dma-bufs over texels `offset` bytes into
    /// their first page, rows and slices `pitches` apart, unbound, with its memory requirements
    /// and where it binds in the import of their pages ([`Unbound`]): its one plane at `offset`,
    /// for a dedicated import (`None`), where the driver lays that plane out there; else, where it
    /// lays it out at another offset and `offset` is not 0, the image of its plane at offset 0 for
    /// a binding at `offset` ([`offset_bound_image`](Self::offset_bound_image), `Some(offset)`).
    /// Why not, in words, where the driver refuses the plane, lays it out otherwise and makes no
    /// such image bound at the offset, or lays out its rows or slices otherwise.
    #[cfg(unix)]
    fn plane_image(
        &self,
        plan: &ImagePlan,
        view_formats: &[vk::Format],
        offset: u64,
        pitches: (u64, u64),
    ) -> Result<Result<(vk::Image, vk::MemoryRequirements, Option<u64>), String>, Error> {
        let explicit =
            self.explicit_plane_image(plan, view_formats, (offset, "its first page"), pitches)?;
        // A test's stand-in for a driver that lays the plane out at offset 0.
        #[cfg(test)]
        let explicit = match explicit {
            Ok((raw, _, report))
                if offset != 0 && self.plane_route_forced() == PlaneRoute::AtOffset =>
            {
                // SAFETY: the image is live and unbound, and no command uses it.
                unsafe { release::object(self.shared().vk(), raw) };
                let answered = vk::SubresourceLayout {
                    offset: 0,
                    ..report
                };
                let why = format!("a test takes the plane at offset {offset} as laid out at 0");
                Err(PlaneMiss::Answered(answered, why))
            }
            explicit => explicit,
        };
        let why = match explicit {
            Ok((raw, requirements, _)) => return Ok(Ok((raw, requirements, None))),
            Err(PlaneMiss::Answered(report, why)) if offset != 0 && report.offset != offset => why,
            Err(miss) => return Ok(Err(miss.words())),
        };
        let bound = self.offset_bound_image(plan, view_formats, offset, pitches)?;
        Ok(match bound {
            Ok((raw, requirements)) => Ok((raw, requirements, Some(offset))),
            Err(why_not) => Err(format!(
                "{why}; and an image of its plane at offset 0 is not bound at offset {offset} of \
                 an import of its pages: {why_not}"
            )),
        })
    }

    /// The `DRM_FORMAT_MOD_LINEAR` image of `plan` for dma-bufs, viewed as `view_formats`, its one
    /// plane at offset 0 with rows and slices `pitches` apart, unbound, with its memory
    /// requirements: to be bound at `offset` of an import of its texels' pages that is no
    /// dedicated allocation, where the driver lays out an explicit plane at `offset` at another
    /// offset, or at `offset` of the import a buffer over the pages is bound in
    /// ([`in_import_image_unbound`](Self::in_import_image_unbound)). Vulkan places an image's memory in memory it shares by its binding's offset
    /// (`vkBindImageMemory`'s `memoryOffset`: the image's bytes are those from there), and lays a
    /// `DRM_FORMAT_MOD_LINEAR` plane's texels from the plane's offset in them, so this image's
    /// texels are those from `offset` of the import, at the plane's pitches. The bytes of the
    /// pages before `offset` lie outside the image, which a dedicated import's binding at offset 0
    /// would hold. Why not, the image released, where the driver refuses the plane or lays it out
    /// otherwise; the image requires a dedicated allocation
    /// (`VkMemoryDedicatedRequirements::requiresDedicatedAllocation`,
    /// `VUID-vkBindImageMemory-image-01445`), which binds an image at offset 0 alone
    /// (`VUID-vkBindImageMemory-memory-02628`); its memory's alignment does not divide `offset`
    /// (`VUID-vkBindImageMemory-None-10735`); or the image format query takes dma-bufs for it in
    /// a dedicated allocation alone (`VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT`).
    #[cfg(unix)]
    fn offset_bound_image(
        &self,
        plan: &ImagePlan,
        view_formats: &[vk::Format],
        offset: u64,
        pitches: (u64, u64),
    ) -> Result<Result<(vk::Image, vk::MemoryRequirements), String>, Error> {
        let raw_device = self.shared().vk();
        let within = (0, "the binding at its texels");
        let raw = match self.explicit_plane_image(plan, view_formats, within, pitches)? {
            Ok((raw, _, _)) => raw,
            Err(miss) => return Ok(Err(miss.words())),
        };
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut asked = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        let query = vk::ImageMemoryRequirementsInfo2::default().image(raw);
        // SAFETY: `raw` is a live image of this device; the query is core in Vulkan 1.1.
        unsafe { raw_device.get_image_memory_requirements2(&query, &mut asked) };
        let requirements = asked.memory_requirements;
        let requires_dedicated = dedicated.requires_dedicated_allocation != vk::FALSE;
        let features = match self.modifier_format_query(plan, view_formats, ModifierMemory::DmaBuf)
        {
            Ok((_, features)) => features,
            Err(error) => {
                // SAFETY: the image is live and unbound, and no command uses it.
                unsafe { release::object(raw_device, raw) };
                return Ok(Err(format!(
                    "the device's image format query reports no such image ({error:?})"
                )));
            }
        };
        let why = if requires_dedicated {
            "it requires a dedicated allocation, which binds it at offset 0 alone".to_owned()
        } else if !offset.is_multiple_of(requirements.alignment.max(1)) {
            format!(
                "its memory's alignment of {} bytes does not divide the offset",
                requirements.alignment
            )
        } else if features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY) {
            format!(
                "the image format query takes dma-bufs for it in a dedicated allocation alone \
                 (features {:#x})",
                features.as_raw()
            )
        } else {
            return Ok(Ok((raw, requirements)));
        };
        // SAFETY: the image is live and unbound, and no command uses it.
        unsafe { release::object(raw_device, raw) };
        Ok(Err(why))
    }

    /// A `DRM_FORMAT_MOD_LINEAR` image of `plan` for dma-bufs, viewed as `view_formats`, its one
    /// plane at `offset` of its memory (`within`, in words) with rows and slices `pitches` apart,
    /// unbound and made `UNDEFINED`, with its memory requirements and its plane's layout as the
    /// driver reports it. Why not where the driver refuses the layout, or lays the plane out
    /// otherwise (with its answer: [`PlaneMiss`]).
    #[cfg(unix)]
    fn explicit_plane_image(
        &self,
        plan: &ImagePlan,
        view_formats: &[vk::Format],
        (offset, within): (u64, &str),
        (row_pitch, slice_pitch): (u64, u64),
    ) -> Result<Result<ExplicitPlane, PlaneMiss>, Error> {
        let raw_device = self.shared().vk();
        let array_pitch = if plan.layers > 1 { slice_pitch } else { 0 };
        let planes = [vk::SubresourceLayout {
            offset,
            size: 0,
            row_pitch,
            array_pitch,
            depth_pitch: 0,
        }];
        if let Some(why) = self.explicit_plane_pitch(row_pitch) {
            return Ok(Err(PlaneMiss::Refused(why)));
        }
        let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(DRM_FORMAT_MOD_LINEAR)
            .plane_layouts(&planes);
        let dma_buf = ModifierMemory::DmaBuf;
        let raw = match self.drm_linear_image(plan, view_formats, &mut explicit, dma_buf)? {
            Ok(raw) => raw,
            Err(error) => {
                return Ok(Err(PlaneMiss::Refused(format!(
                    "the driver refuses a DRM_FORMAT_MOD_LINEAR plane at offset {offset} of \
                     {within} with rows {row_pitch} bytes apart ({error:?})"
                ))))
            }
        };
        // SAFETY: `raw` is a live image of this device.
        let requirements = unsafe { raw_device.get_image_memory_requirements(raw) };
        // SAFETY: `raw` is a live image of a DRM format modifier's tiling, whose one memory plane
        // is plane 0 (`VUID-vkGetImageSubresourceLayout-tiling-09433`).
        let report = unsafe { raw_device.get_image_subresource_layout(raw, memory_plane_0()) };
        exit_guard::called(Call::GetImageSubresourceLayout);
        // A test's stand-in for a driver whose answer is not checked: the image as made.
        #[cfg(test)]
        if self.plane_route_forced() == PlaneRoute::Unchecked {
            return Ok(Ok((raw, requirements, report)));
        }
        if report.offset != offset
            || report.row_pitch != row_pitch
            || (plan.layers > 1 && report.array_pitch != array_pitch)
        {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, raw) };
            let why = format!(
                "the driver lays the plane out at offset {}, rows {} and slices {} bytes apart, \
                 where the texels lie at offset {offset}, rows {row_pitch} and slices \
                 {array_pitch} apart",
                report.offset, report.row_pitch, report.array_pitch
            );
            return Ok(Err(PlaneMiss::Answered(report, why)));
        }
        Ok(Ok((raw, requirements, report)))
    }

    /// The image of a texture over `texels` in the runtime's own memory exported as a dma-buf
    /// (`exported_memory.rs`: a shared block of such buffers, or a buffer's dedicated allocation)
    /// where no image binds in that memory itself (`texture.rs`'s `alias_unbound`): a
    /// `DRM_FORMAT_MOD_LINEAR` image made for dma-bufs, its one plane at the texels' offset in the
    /// memory with Metal's pitches, unbound, with its memory requirements and its plane's layout.
    /// It is bound to a
    /// dedicated import of that memory's own dma-buf ([`Device::bind_exported_dma_buf_image`]),
    /// as an image over dma-buf pages is to one of theirs ([`Device::dma_buf_image`]): its memory
    /// is the buffer's, through a second `VkDeviceMemory` of the same payload, which Vulkan
    /// imports into the instance that exported it (`VkImportMemoryFdInfoKHR`; the dma-buf is the
    /// device's own, `VUID-VkImportMemoryFdInfoKHR-fd-00668`). Why not, the image released, where
    /// the device makes no such image for dma-bufs (the reasons of [`Device::dma_buf_image`]
    /// before the plane's), refuses the plane, or the image's memory runs past the buffer's
    /// memory or takes none of its type.
    #[cfg(unix)]
    pub(super) fn exported_dma_buf_image_unbound(
        &self,
        plan: &ImagePlan,
        facts: &ImageFacts<'_>,
        texels: &ExportedTexels,
    ) -> Result<Result<ExplicitPlane, String>, Error> {
        let raw_device = self.shared().vk();
        let view_formats = match self.dma_buf_image_checks(plan, facts)? {
            Ok((_, view_formats)) => view_formats,
            Err(why) => return Ok(Err(why)),
        };
        if let Some(why) = self.explicit_plane_rows(plan) {
            return Ok(Err(why));
        }
        let pitches = (texels.row_pitch, texels.slice_pitch);
        let within = (texels.at, "the buffer's memory");
        let made = self.explicit_plane_image(plan, &view_formats, within, pitches)?;
        let (raw, requirements, report) = match made {
            Ok(made) => made,
            Err(miss) => return Ok(Err(miss.words())),
        };
        let why = if requirements.size > texels.size {
            Some(format!(
                "its image's memory, {} bytes, runs past the {} bytes of the buffer's memory",
                requirements.size, texels.size
            ))
        } else if requirements.memory_type_bits & (1 << texels.type_index) == 0 {
            Some(format!(
                "its image takes none of the buffer's memory type {}",
                texels.type_index
            ))
        } else {
            None
        };
        if let Some(why) = why {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, raw) };
            return Ok(Err(why));
        }
        Ok(Ok((raw, requirements, report)))
    }

    /// `raw`, an image [`exported_dma_buf_image_unbound`](Self::exported_dma_buf_image_unbound)
    /// made over `texels` with its `requirements`, bound to a dedicated import of a new dma-buf of
    /// the buffer's memory (`vkGetMemoryFdKHR`, as the buffer's own export), of the memory's own
    /// type, and counted (`WorkCounts::exported_dma_buf_images`); the import is held as an image's
    /// dedicated one (`WorkCounts::dma_buf_image_memories_live`) and freed after the image. The
    /// driver takes the descriptor it imports and the runtime closes its own either way
    /// (`memory::import_dma_buf`). Why not, the image released, where the device exports or
    /// imports no such dma-buf, or the driver refuses the dma-buf, its import or its binding.
    #[cfg(unix)]
    pub(super) fn bind_exported_dma_buf_image(
        &self,
        (raw, requirements, _): ExplicitPlane,
        texels: &ExportedTexels,
    ) -> Result<Result<DmaBufImage, String>, Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let refuse = |why: String| {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, raw) };
            Ok(Err(why))
        };
        let Ok(importer) = shared.host_import.as_ref() else {
            return refuse("the device imports no dma-buf".to_owned());
        };
        let fd = match &texels.payload {
            // A carried buffer's memory is its payload imported again: a clone of its dma-buf.
            Some(payload) => match payload.try_clone() {
                Ok(fd) => Ok(Ok(fd)),
                Err(error) => Ok(Err(format!(
                    "the buffer memory's dma-buf takes no clone ({error})"
                ))),
            },
            None => self.dma_buf_of(texels.memory),
        };
        let fd = match fd {
            Ok(Ok(fd)) => fd,
            Ok(Err(why)) => return refuse(why),
            Err(error) => {
                // SAFETY: the image is live and unbound, and no command uses it.
                unsafe { release::object(raw_device, raw) };
                return Err(error);
            }
        };
        // The buffer memory's own type, dedicated to the image at its requirements' size exactly
        // (`VUID-VkMemoryDedicatedAllocateInfo-image-01433`), within the dma-buf.
        let allowed = (1 << texels.type_index, texels.class);
        let (size, purpose) = (requirements.size, DmaBufFor::Image(raw));
        // SAFETY: the importer is this device's, the descriptor a dma-buf, and the image this
        // device's, unbound, made for dma-bufs.
        let imported =
            match unsafe { self.import_timed(importer, fd.as_fd(), size, allowed, purpose) } {
                Ok(imported) => imported,
                Err(error) => {
                    return refuse(format!(
                        "the driver refuses the buffer memory's dma-buf ({error:?})"
                    ))
                }
            };
        let memory = match imported {
            Ok((memory, _, _)) => memory,
            Err(ImportFailure::NoMemoryType) => {
                return refuse(format!(
                    "the driver imports the buffer memory's dma-buf as none of its memory type {}",
                    texels.type_index
                ))
            }
            Err(ImportFailure::Vulkan(vk::Result::ERROR_DEVICE_LOST)) => {
                // SAFETY: the image is live and unbound, and no command uses it.
                unsafe { release::object(raw_device, raw) };
                return Err(Error::DeviceLost);
            }
            Err(ImportFailure::Vulkan(error)) => {
                return refuse(format!(
                    "the driver refuses the import of the buffer memory's dma-buf ({error:?})"
                ))
            }
        };
        // SAFETY: the image is unbound; the memory is its dedicated import, of its size.
        if let Err(error) = unsafe { crate::creation::bind_image(raw_device, raw, memory, 0) } {
            // SAFETY: nothing is bound to the memory, and no command uses it.
            unsafe { release::object(raw_device, memory) };
            return refuse(format!(
                "the driver refuses to bind the import of the buffer memory's dma-buf ({error:?})"
            ));
        }
        shared.counts.exported_dma_buf_image_made();
        let mut import = DmaBufPages::of_exported(self, texels.size);
        import.hold(memory);
        Ok(Ok(DmaBufImage {
            raw,
            import: Arc::new(import),
        }))
    }

    /// `unbound`'s image bound to an import of the dma-buf the caller's source makes of its
    /// pages, of `texels`' memory type, and counted: a dedicated import, the image at its offset
    /// 0; or, for an image bound at its texels' offset ([`Unbound`]), an import of the dma-buf's
    /// pages whole that is none, the image at that offset
    /// (`WorkCounts::dma_buf_images_at_offset`). Why not, the image released, where the source
    /// fails or answers against its contract or the driver refuses the import or binding.
    #[cfg(unix)]
    pub(super) fn bind_dma_buf_image(
        &self,
        unbound: Unbound,
        texels: &DmaBufTexels<'_>,
    ) -> Result<Result<DmaBufImage, (DmaBufRefusal, String)>, Error> {
        let Unbound {
            raw,
            requirements,
            request,
            bound_at,
        } = unbound;
        let shared = self.shared();
        let raw_device = shared.vk();
        let refuse = |kind: DmaBufRefusal, why: String| {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, raw) };
            Ok(Err((kind, why)))
        };
        let (Ok(importer), Some(source)) = (
            shared.host_import.as_ref(),
            texels.pages.source.as_ref().map(ProgramObject::get),
        ) else {
            return refuse(
                DmaBufRefusal::Source { stands: false },
                "the pages keep no source of dma-bufs".to_owned(),
            );
        };
        // The source is the program's code, called with no lock of the runtime's held.
        let chunks = match source(request) {
            Ok(chunks) => chunks,
            Err(error) => {
                let inner = error.get_ref();
                let stands = inner.is_some_and(|inner| inner.is::<DmaBufStandingRefusal>());
                return refuse(
                    DmaBufRefusal::Source { stands },
                    format!(
                        "the caller's source made no dma-buf of its {} bytes of pages: {error}",
                        request.length
                    ),
                );
            }
        };
        // A request whose granule is its length takes one chunk (`chunks_cover`), a dma-buf of
        // its own (the request takes no shared file: the import is the image's dedicated one).
        let answered = taken(chunks, &request, importer.page).map(<[Piece; 1]>::try_from);
        let Some(Ok([Piece::Own { fd, .. }])) = answered else {
            return refuse(
                DmaBufRefusal::Source { stands: false },
                "the caller's source answered other than one dma-buf of its own of the pages \
                 asked"
                    .to_owned(),
            );
        };
        // The pages' type (the class's, or another the dma-buf takes: `memory.rs`'s
        // `dma_buf_candidates`), dedicated to the image at its requirements' size exactly
        // (`VUID-VkMemoryDedicatedAllocateInfo-image-01433`), within the dma-buf's pages; or, for
        // an image bound at its texels' offset, the dma-buf's pages whole, dedicated to nothing.
        let allowed = (1 << texels.type_index, MemoryClass::Shared);
        let (size, purpose) = match bound_at {
            None => (requirements.size, DmaBufFor::Image(raw)),
            Some(_) => (request.length, DmaBufFor::Shared),
        };
        // SAFETY: the importer is this device's, the descriptor a dma-buf, and the image this
        // device's, unbound, made for dma-bufs.
        let imported =
            match unsafe { self.import_timed(importer, fd.as_fd(), size, allowed, purpose) } {
                Ok(imported) => imported,
                Err(error) => {
                    let why = format!("the driver refuses the dma-buf ({error:?})");
                    return refuse(DmaBufRefusal::Import, why);
                }
            };
        let memory = match imported {
            Ok((memory, _, _)) => memory,
            Err(ImportFailure::NoMemoryType) => {
                return refuse(
                    DmaBufRefusal::Import,
                    format!(
                        "the driver imports the dma-buf as none of the pages' memory type {}",
                        texels.type_index
                    ),
                )
            }
            Err(ImportFailure::Vulkan(vk::Result::ERROR_DEVICE_LOST)) => {
                // SAFETY: the image is live and unbound, and no command uses it.
                unsafe { release::object(raw_device, raw) };
                return Err(Error::DeviceLost);
            }
            Err(ImportFailure::Vulkan(error)) => {
                return refuse(
                    DmaBufRefusal::Import,
                    format!("the driver refuses the import ({error:?})"),
                )
            }
        };
        // SAFETY: the image is unbound; the memory is its dedicated allocation, of its size, or
        // an import of its pages that is none, of a type it takes, where its requirements' size
        // from the offset, a multiple of their alignment, lies within it (`offset_bound_image`,
        // `dma_buf_image_unbound`).
        let at = bound_at.unwrap_or(0);
        if let Err(error) = unsafe { crate::creation::bind_image(raw_device, raw, memory, at) } {
            // SAFETY: nothing is bound to the memory, and no command uses it.
            unsafe { release::object(raw_device, memory) };
            let why = format!("the driver refuses to bind the import at offset {at} ({error:?})");
            return refuse(DmaBufRefusal::Import, why);
        }
        shared.counts.dma_buf_image_made();
        if bound_at.is_some() {
            shared.counts.dma_buf_image_bound_at_offset();
        }
        let mut import = DmaBufPages::new(self, Arc::clone(source), request, true);
        import.hold(memory);
        Ok(Ok(DmaBufImage {
            raw,
            import: Arc::new(import),
        }))
    }

    /// What a `DRM_FORMAT_MOD_LINEAR` image of `plan` for dma-bufs takes before its plane is laid
    /// out: the host's page, which its import starts at, and the formats it lists for its views;
    /// why not where the device makes none (the reasons of [`Device::dma_buf_image`] before the
    /// plane's).
    fn dma_buf_image_checks(
        &self,
        plan: &ImagePlan,
        facts: &ImageFacts<'_>,
    ) -> Result<Result<(u64, Vec<vk::Format>), String>, Error> {
        let page = match self.shared().host_import.as_ref() {
            Ok(importer) if importer.makes_dma_buf_images() => importer.page,
            _ => {
                return Ok(Err(
                    "the device enables no VK_EXT_image_drm_format_modifier and \
                     VK_EXT_queue_family_foreign beside its dma-buf imports"
                        .to_owned(),
                ))
            }
        };
        let checked = self.modifier_image_checks(plan, facts, ModifierMemory::DmaBuf)?;
        Ok(checked.map(|view_formats| (page, view_formats)))
    }

    /// What a `DRM_FORMAT_MOD_LINEAR` image of `plan` for `memory` (dma-bufs, or host allocations
    /// at an explicit pitch) takes before its plane is laid out, on a device that enables
    /// `VK_EXT_image_drm_format_modifier`: the formats it lists for its views; why not where the
    /// device makes none: the plan has several levels, depth or samples, or a mutable format with
    /// no list of its views; its format has a depth or stencil component; the modifier's tiling
    /// lacks a feature the plan's optimal tiling has for its usage; the format query refuses such
    /// an image for `memory`, or the plan passes its limits.
    ///
    /// The runtime takes a modifier image's plane layout from the driver
    /// (`vkGetImageSubresourceLayout`), and Vulkan has no valid such query of a format with a
    /// depth or stencil component: of a DRM format modifier's tiling its aspect must be a memory
    /// plane's (`VUID-vkGetImageSubresourceLayout-tiling-09433`), of a format with depth or
    /// stencil it must contain that aspect (`-format-04462`, `-format-04463`), and it is one bit
    /// (`-aspectMask-00997`). So it makes none of such a format, wherever a driver lists the
    /// modifier for it.
    pub(super) fn modifier_image_checks(
        &self,
        plan: &ImagePlan,
        facts: &ImageFacts<'_>,
        memory: ModifierMemory,
    ) -> Result<Result<Vec<vk::Format>, String>, Error> {
        if plan.levels != 1 || plan.extent[2] != 1 || plan.samples != SampleCountFlags::COUNT_1 {
            return Ok(Err(format!(
                "its image has {} levels, a depth of {} and {:?} samples, where a \
                 DRM_FORMAT_MOD_LINEAR image over the pages has one of each",
                plan.levels, plan.extent[2], plan.samples
            )));
        }
        if (plan.format.block())
            .is_some_and(|block| block.depth.is_some() || block.stencil.is_some())
        {
            return Ok(Err(format!(
                "{:?} has a depth or stencil component, and Vulkan has no query of where a \
                 DRM_FORMAT_MOD_LINEAR plane of such a format lies",
                plan.format
            )));
        }
        // A DRM format modifier's mutable image lists its views' formats
        // (`VUID-VkImageCreateInfo-tiling-02353`): a plan whose views take any compatible format
        // lists each the runtime views a texture as (`linear_plan::viewed_formats`), its own
        // format first.
        let mutable = plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT);
        let formats: Vec<Format> = match mutable && plan.view_formats.is_empty() {
            true => std::iter::once(plan.format)
                .chain(viewed_formats(plan, facts))
                .collect(),
            false => plan.view_formats.clone(),
        };
        if let Some(LinearLack { format, features }) =
            tiling_lack(plan, facts, |format| self.modifier_features(format))
        {
            return Ok(Err(format!(
                "the DRM_FORMAT_MOD_LINEAR tiling of {format:?} lacks {features:?}, which its \
                 optimal tiling has for the texture's usage"
            )));
        }
        let view_formats: Vec<vk::Format> = formats.iter().map(|&format| format.into()).collect();
        let limits = match self.modifier_image_limits(plan, &view_formats, memory) {
            Ok(limits) => limits,
            Err(why) => return Ok(Err(why)),
        };
        if let Err(need) = check_linear_limits(plan, &limits) {
            return Ok(Err(format!(
                "its image passes the device's limits for a DRM_FORMAT_MOD_LINEAR image of \
                 {:?} for {}: {need:?}",
                plan.format,
                memory.words(),
            )));
        }
        Ok(Ok(view_formats))
    }

    /// The layout the driver gives a `DRM_FORMAT_MOD_LINEAR` image of `plan` for dma-bufs where it
    /// chooses it (`VkImageDrmFormatModifierListCreateInfoEXT`), its plane at offset 0: what a
    /// heap's texture over dma-buf pages lays its texels out as (`texture/heap.rs`), as the
    /// driver's linear image lays out one over pages imported by their pointer. Why not where the
    /// device makes none (the reasons of [`Device::dma_buf_image`] before the plane's, or the
    /// driver refuses such an image).
    #[cfg(unix)]
    pub(super) fn dma_buf_natural_layout(
        &self,
        plan: &ImagePlan,
        facts: &ImageFacts<'_>,
    ) -> Result<Result<NaturalLayout, String>, Error> {
        let raw_device = self.shared().vk();
        let view_formats = match self.dma_buf_image_checks(plan, facts)? {
            Ok((_, view_formats)) => view_formats,
            Err(why) => return Ok(Err(why)),
        };
        let modifiers = [DRM_FORMAT_MOD_LINEAR];
        let mut chosen =
            vk::ImageDrmFormatModifierListCreateInfoEXT::default().drm_format_modifiers(&modifiers);
        let dma_buf = ModifierMemory::DmaBuf;
        let raw = match self.drm_linear_image(plan, &view_formats, &mut chosen, dma_buf)? {
            Ok(raw) => raw,
            Err(error) => {
                return Ok(Err(format!(
                    "the driver refuses a DRM_FORMAT_MOD_LINEAR image of its own layout \
                     ({error:?})"
                )))
            }
        };
        // SAFETY: `raw` is a live image of a DRM format modifier's tiling, whose one memory plane
        // is plane 0; it is destroyed right after the queries, with no memory bound.
        let (report, requirements) = unsafe {
            let report = raw_device.get_image_subresource_layout(raw, memory_plane_0());
            let requirements = raw_device.get_image_memory_requirements(raw);
            release::object(raw_device, raw);
            (report, requirements)
        };
        Ok(Ok(NaturalLayout {
            row_pitch: report.row_pitch,
            slice_pitch: report.array_pitch,
            size: requirements.size,
        }))
    }

    /// A `DRM_FORMAT_MOD_LINEAR` image of `plan` for `memory` (dma-bufs, host allocations, or the
    /// runtime's own memory), viewed as `view_formats`, its plane laid out as `modifier` says (an
    /// explicit layout, or the modifier listed for the driver to lay out), made `UNDEFINED`; or the
    /// driver's refusal of it.
    #[cfg(unix)]
    pub(super) fn drm_linear_image(
        &self,
        plan: &ImagePlan,
        view_formats: &[vk::Format],
        modifier: &mut impl vk::ExtendsImageCreateInfo,
        memory: ModifierMemory,
    ) -> Result<Result<vk::Image, vk::Result>, Error> {
        let shared = self.shared();
        let mut external = vk::ExternalMemoryImageCreateInfo::default();
        let mut list = vk::ImageFormatListCreateInfo::default();
        let mut info = plan
            .create_info(shared.sharing(), view_formats, &mut list)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(modifier);
        if let Some(handle) = memory.handle_type() {
            external = external.handle_types(handle);
            info = info.push_next(&mut external);
        }
        // SAFETY: the device is live and enables the extension; `info` and its chain are valid
        // and within the device's limits for such an image (`modifier_image_limits`), its one
        // plane the modifier's one (`modifier_features`), made `UNDEFINED` as Vulkan makes an
        // image of external memory and every image of a DRM format modifier's tiling
        // (`VUID-VkImageCreateInfo-initialLayout-12478`: `PREINITIALIZED` is for linear tiling
        // alone).
        unsafe { crate::creation::image_or_refusal(shared.vk(), &info) }
    }

    /// The features of `format`'s `DRM_FORMAT_MOD_LINEAR` tiling where the device lists that
    /// modifier for it with one plane; none elsewhere. The 64-bit list
    /// (`VkDrmFormatModifierPropertiesList2EXT`) where the device has `format_feature_flags2`, as
    /// the optimal tiling's features come from it: the 32-bit list cannot carry the storage bits
    /// above bit 31 (`STORAGE_READ_WITHOUT_FORMAT` and `STORAGE_WRITE_WITHOUT_FORMAT`), so a
    /// writable texture would always lack them. Asked of the driver once per format and kept
    /// (`TextureState::modifier_features`): a physical device's answer does not change, and ANV's
    /// took most of the making of a texture over host pages at Metal's row pitch.
    pub(super) fn modifier_features(&self, format: Format) -> FormatFeatureFlags2 {
        let kept = &self.shared().textures.modifier_features;
        if let Some(&features) = lock(kept).get(&format) {
            return features;
        }
        let features = self.modifier_features_asked(format);
        self.shared().counts.modifier_features_asked();
        lock(kept).insert(format, features);
        features
    }

    /// [`Device::modifier_features`]' answer, asked of the driver.
    fn modifier_features_asked(&self, format: Format) -> FormatFeatureFlags2 {
        let shared = self.shared();
        let format: vk::Format = format.into();
        let query = |list: &mut dyn vk::ExtendsFormatProperties2| {
            let mut properties = vk::FormatProperties2::default().push_next(list);
            // SAFETY: the physical device is the instance's, whose device enables the extension;
            // a list without entries asks the count alone, and one with entries holds as many as
            // its count for the driver to write.
            unsafe {
                shared.instance.raw.get_physical_device_format_properties2(
                    shared.physical,
                    format,
                    &mut properties,
                )
            }
        };
        let linear = |modifier: u64, planes: u32| modifier == DRM_FORMAT_MOD_LINEAR && planes == 1;
        let features = if self.properties().description.has_format_feature_flags2() {
            let mut list = vk::DrmFormatModifierPropertiesList2EXT::default();
            query(&mut list);
            let count = list.drm_format_modifier_count as usize;
            let mut modifiers = vec![vk::DrmFormatModifierProperties2EXT::default(); count];
            let mut list = vk::DrmFormatModifierPropertiesList2EXT::default()
                .drm_format_modifier_properties(&mut modifiers);
            query(&mut list);
            let written = list.drm_format_modifier_count as usize;
            modifiers.iter().take(written).find_map(|modifier| {
                linear(
                    modifier.drm_format_modifier,
                    modifier.drm_format_modifier_plane_count,
                )
                .then_some(modifier.drm_format_modifier_tiling_features.as_raw())
            })
        } else {
            let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
            query(&mut list);
            let count = list.drm_format_modifier_count as usize;
            let mut modifiers = vec![vk::DrmFormatModifierPropertiesEXT::default(); count];
            let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
                .drm_format_modifier_properties(&mut modifiers);
            query(&mut list);
            let written = list.drm_format_modifier_count as usize;
            modifiers.iter().take(written).find_map(|modifier| {
                linear(
                    modifier.drm_format_modifier,
                    modifier.drm_format_modifier_plane_count,
                )
                .then_some(u64::from(
                    modifier.drm_format_modifier_tiling_features.as_raw(),
                ))
            })
        };
        features.map_or(FormatFeatureFlags2::empty(), FormatFeatureFlags2::from_raw)
    }

    /// The device's limits for a `DRM_FORMAT_MOD_LINEAR` image of `plan` for `memory`, viewed as
    /// `view_formats`; why not where its format query refuses such an image or it imports no such
    /// memory for one (for host allocations, also where only a dedicated allocation takes it).
    fn modifier_image_limits(
        &self,
        plan: &ImagePlan,
        view_formats: &[vk::Format],
        memory: ModifierMemory,
    ) -> Result<vk::ImageFormatProperties, String> {
        let (limits, features) = match self.modifier_format_query(plan, view_formats, memory) {
            Ok(answer) => answer,
            Err(error) => {
                return Err(format!(
                    "the device's image format query reports no DRM_FORMAT_MOD_LINEAR image of \
                     {:?}, usage {:?}, flags {:?} for {} ({error:?})",
                    plan.format,
                    plan.usage,
                    plan.flags,
                    memory.words(),
                ))
            }
        };
        // An image of the runtime's own memory imports none.
        if memory.handle_type().is_none() {
            return Ok(limits);
        }
        // An image for host allocations is bound at an offset of a buffer's import, and one for
        // exported memory at an offset of an exported block: neither is a dedicated allocation.
        let dedicated_only = memory.shared_memory()
            && features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY);
        if !features.contains(memory.needs()) || dedicated_only {
            return Err(format!(
                "its image format query refuses {} to a DRM_FORMAT_MOD_LINEAR image of {:?}, \
                 or takes them in a dedicated allocation alone (features {:#x})",
                memory.words(),
                plan.format,
                features.as_raw()
            ));
        }
        Ok(limits)
    }

    /// The image format query's answer for a `DRM_FORMAT_MOD_LINEAR` image of `plan` for
    /// `memory`, viewed as `view_formats`: its limits, and the external memory features of
    /// `memory`'s handle type for it (none for the runtime's own memory, of no handle type); or
    /// the query's refusal.
    fn modifier_format_query(
        &self,
        plan: &ImagePlan,
        view_formats: &[vk::Format],
        memory: ModifierMemory,
    ) -> Result<(vk::ImageFormatProperties, vk::ExternalMemoryFeatureFlags), vk::Result> {
        let shared = self.shared();
        let families = &shared.queue_families;
        let mut modifier = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
            .drm_format_modifier(DRM_FORMAT_MOD_LINEAR);
        modifier = match families.len() > 1 {
            true => modifier
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(families),
            false => modifier.sharing_mode(vk::SharingMode::EXCLUSIVE),
        };
        let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default();
        let mut list = vk::ImageFormatListCreateInfo::default().view_formats(view_formats);
        let mut query = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(plan.format.into())
            .ty(plan.image_type.into())
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(plan.usage.into())
            .flags(plan.flags.into())
            .push_next(&mut modifier);
        if let Some(handle) = memory.handle_type() {
            external = external.handle_type(handle);
            query = query.push_next(&mut external);
        }
        if !view_formats.is_empty() {
            query = query.push_next(&mut list);
        }
        // The import's properties are asked only beside the handle type they answer for: asked
        // without one, the driver leaves them untouched, and the validation layer warns
        // (`WARNING-vkGetPhysicalDeviceImageFormatProperties2-missing-external-info`).
        let mut imported = vk::ExternalImageFormatProperties::default();
        let mut properties = vk::ImageFormatProperties2::default();
        if memory.handle_type().is_some() {
            properties = properties.push_next(&mut imported);
        }
        // SAFETY: the physical device is the instance's, whose device enables the extensions;
        // `query` and `properties` and their chains live for the call.
        let queried = unsafe {
            shared
                .instance
                .raw
                .get_physical_device_image_format_properties2(
                    shared.physical,
                    &query,
                    &mut properties,
                )
        };
        queried?;
        let limits = properties.image_format_properties;
        let features = imported.external_memory_properties.external_memory_features;
        Ok((limits, features))
    }
}

#[cfg(test)]
impl Device {
    /// The route a test forced (`DeviceShared::plane_route_forced`, as `route as u8`).
    #[cfg(unix)]
    fn plane_route_forced(&self) -> PlaneRoute {
        match (self.shared().plane_route_forced).load(std::sync::atomic::Ordering::Relaxed) {
            1 => PlaneRoute::Unchecked,
            2 => PlaneRoute::AtOffset,
            _ => PlaneRoute::Answered,
        }
    }
}

impl Device {
    /// Whether the device makes `DRM_FORMAT_MOD_LINEAR` images: it enables
    /// `VK_EXT_image_drm_format_modifier`, which it does beside its dma-buf imports
    /// (`HostImport::makes_dma_buf_images`).
    pub(super) fn makes_modifier_images(&self) -> bool {
        (self.shared().host_import.as_ref()).is_ok_and(|importer| importer.makes_dma_buf_images())
    }

    /// Why the device makes no `DRM_FORMAT_MOD_LINEAR` image of `plan` with an explicit plane,
    /// where its driver renders such an image's rows past its height
    /// ([`RuntimeDriverDefect::ExplicitPlaneRowsPastHeight`]) and the plan's height is no multiple
    /// of 4: `None` elsewhere.
    pub(super) fn explicit_plane_rows(&self, plan: &ImagePlan) -> Option<String> {
        let defect = RuntimeDriverDefect::ExplicitPlaneRowsPastHeight;
        let height = plan.extent[1];
        (!height.is_multiple_of(4) && self.properties().driver_defects.contains(&defect)).then(
            || {
                format!(
                    "the driver renders an explicit DRM_FORMAT_MOD_LINEAR plane's rows past its \
                 height of {height} to a multiple of 4 ({defect:?})"
                )
            },
        )
    }

    /// Why the device makes no `DRM_FORMAT_MOD_LINEAR` image with an explicit plane whose rows lie
    /// `row_pitch` bytes apart, where its driver's shaders read such a plane's rows at another
    /// pitch ([`RuntimeDriverDefect::ExplicitPlanePitchMisread`], or a test's stand-in for it:
    /// `DeviceShared::explicit_plane_pitch_misread_forced`) and `row_pitch` is no multiple of
    /// [`EXPLICIT_PLANE_PITCH_READ`]: `None` elsewhere.
    #[cfg(unix)]
    pub(super) fn explicit_plane_pitch(&self, row_pitch: u64) -> Option<String> {
        let defect = RuntimeDriverDefect::ExplicitPlanePitchMisread;
        #[cfg(test)]
        let forced = (self.shared().explicit_plane_pitch_misread_forced)
            .load(std::sync::atomic::Ordering::Relaxed);
        #[cfg(not(test))]
        let forced = false;
        let misread = forced || self.properties().driver_defects.contains(&defect);
        (misread && !row_pitch.is_multiple_of(EXPLICIT_PLANE_PITCH_READ)).then(|| {
            format!(
                "the driver's shaders read an explicit DRM_FORMAT_MOD_LINEAR plane's rows {row_pitch} \
                 bytes apart at another pitch, as at every pitch that is no multiple of \
                 {EXPLICIT_PLANE_PITCH_READ} ({defect:?})"
            )
        })
    }
}

/// Where a driver's shaders read an explicit `DRM_FORMAT_MOD_LINEAR` plane's rows at another
/// pitch than its own ([`RuntimeDriverDefect::ExplicitPlanePitchMisread`]), the row pitches they
/// read right: multiples of this many bytes.
#[cfg(unix)]
pub(super) const EXPLICIT_PLANE_PITCH_READ: u64 = 32;

/// The memory a `DRM_FORMAT_MOD_LINEAR` image is made for: a dma-buf of its own pages
/// ([`Device::dma_buf_image`]); host pages imported by their pointer, where a texture's image in
/// place lays its rows out at Metal's pitch (`texture/in_place_image.rs`); or a buffer's memory of
/// the runtime's own, where a texture over the buffer lays its rows out at Metal's pitch
/// (`texture.rs`'s `alias_image`): with no external memory, or exported as the handle type its
/// record names (`memory.rs`'s `Memory::bound_memory`, `exported_memory.rs`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ModifierMemory {
    DmaBuf,
    Host,
    Own,
    Exported(vk::ExternalMemoryHandleTypeFlags),
}

impl std::fmt::Debug for ModifierMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModifierMemory::DmaBuf => f.write_str("DmaBuf"),
            ModifierMemory::Host => f.write_str("Host"),
            ModifierMemory::Own => f.write_str("Own"),
            ModifierMemory::Exported(handle) => write!(f, "Exported({:#x})", handle.as_raw()),
        }
    }
}

impl ModifierMemory {
    /// The external memory handle type the image is made for; none for the runtime's own memory
    /// of no export.
    pub(super) fn handle_type(self) -> Option<vk::ExternalMemoryHandleTypeFlags> {
        match self {
            ModifierMemory::DmaBuf => Some(DMA_BUF),
            ModifierMemory::Host => Some(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT),
            ModifierMemory::Own => None,
            ModifierMemory::Exported(handle) => Some(handle),
        }
    }

    /// The memory, in words.
    pub(super) fn words(self) -> &'static str {
        match self {
            ModifierMemory::DmaBuf => "dma-bufs",
            ModifierMemory::Host => "host allocations",
            ModifierMemory::Own => "the runtime's own memory",
            ModifierMemory::Exported(DMA_BUF) => "the runtime's memory exported as dma-bufs",
            ModifierMemory::Exported(_) => {
                "the runtime's memory exported as opaque file descriptors"
            }
        }
    }

    /// The external memory features the image's handle type must have
    /// (`VkExternalImageFormatProperties`): exportable for memory the runtime exports, importable
    /// for an import.
    pub(super) fn needs(self) -> vk::ExternalMemoryFeatureFlags {
        match self {
            ModifierMemory::Exported(_) => vk::ExternalMemoryFeatureFlags::EXPORTABLE,
            ModifierMemory::DmaBuf | ModifierMemory::Host | ModifierMemory::Own => {
                vk::ExternalMemoryFeatureFlags::IMPORTABLE
            }
        }
    }

    /// Whether the image is bound at an offset of memory it shares, which a handle type that
    /// takes only a dedicated allocation (`VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT`) does
    /// not: host allocations (a buffer's import) and the runtime's exported blocks; a dma-buf of
    /// the image's own pages is its dedicated import.
    pub(super) fn shared_memory(self) -> bool {
        match self {
            ModifierMemory::Host | ModifierMemory::Exported(_) => true,
            ModifierMemory::DmaBuf | ModifierMemory::Own => false,
        }
    }
}

/// An allocation of the device's own, host-visible and exported as a dma-buf, mapped: the pages of
/// the device's check of textures over dma-buf pages (`in_place_check.rs`), which the runtime
/// runs no `udmabuf` code to make. Freed (and so unmapped) as it drops.
#[cfg(unix)]
pub(super) struct ExportedPages {
    device: Device,
    memory: vk::DeviceMemory,
    start: NonNull<u8>,
    length: u64,
    fd: OwnedFd,
}

// SAFETY: the allocation's handle and mapping are the device's, used by one holder at a time; the
// mapping's address is plain data.
#[cfg(unix)]
unsafe impl Send for ExportedPages {}

#[cfg(unix)]
impl ExportedPages {
    /// `length` bytes of host-visible, host-coherent memory, exported as a dma-buf and mapped, of
    /// the first such type that is host-cached, one that is not device-local before one that is,
    /// else the first that is not device-local, else the first: memory as the CPU caches it, as a
    /// guest's pages are, which the check reads every byte of back. That took 3.1 s for a
    /// 3840x2160 surface's 33 MB through RADV's uncached type on the Radeon 780M, against 7.2 ms
    /// cached; and on the Iris Xe (ANV), whose host-visible types are all device-local, the whole
    /// check took 19.0 s over its first type, which is uncached. Allocated with the device-address
    /// flag where the device enables `bufferDeviceAddress`, as each of the runtime's allocations
    /// and imports is: ANV (Mesa 26.2.6) refuses an import of a dma-buf of its own memory whose
    /// flag differs from the allocation's (`VK_ERROR_INVALID_EXTERNAL_HANDLE`, of every memory
    /// type), which left the check's class unchecked there. Why not, where the device imports no
    /// dma-bufs, makes no images over them, exports no such memory but as a dedicated allocation,
    /// each a cause that lasts, or refuses it, one that may pass (the check's note).
    pub(super) fn make(device: &Device, length: u64) -> Result<ExportedPages, Unmade> {
        let shared = device.shared();
        let importer = (shared.host_import.as_ref())
            .map_err(|_| Unmade::lasting("the device imports no host memory"))?;
        if !importer.makes_dma_buf_images() {
            return Err(Unmade::lasting("the device makes no images over dma-bufs"));
        }
        // The device exports a buffer's memory as a dma-buf, not only a dedicated allocation.
        let query = vk::PhysicalDeviceExternalBufferInfo::default()
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .handle_type(DMA_BUF);
        let mut properties = vk::ExternalBufferProperties::default();
        // SAFETY: the physical device is the instance's, at Vulkan 1.2 or later.
        unsafe {
            shared
                .instance
                .raw
                .get_physical_device_external_buffer_properties(
                    shared.physical,
                    &query,
                    &mut properties,
                )
        };
        let features = properties
            .external_memory_properties
            .external_memory_features;
        if !features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
            || features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
        {
            return Err(Unmade::lasting(format!(
                "the device exports a buffer's memory as a dma-buf only as a dedicated \
                 allocation, or not at all (external memory features {:#x})",
                features.as_raw()
            )));
        }
        let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let type_index = {
            let memory = lock(&shared.memory);
            let types: Vec<(u32, vk::MemoryPropertyFlags)> = (0..vk::MAX_MEMORY_TYPES as u32)
                .map_while(|index| memory.type_flags(index).map(|flags| (index, flags)))
                .filter(|(_, flags)| flags.contains(wanted))
                .collect();
            // Host-cached first, then system memory first; the first of the best.
            (types.iter())
                .min_by_key(|(_, flags)| {
                    (
                        !flags.contains(vk::MemoryPropertyFlags::HOST_CACHED),
                        flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL),
                    )
                })
                .map(|&(index, _)| index)
                .ok_or_else(|| {
                    Unmade::lasting("the device has no host-visible, host-coherent memory")
                })?
        };
        let raw_device = shared.vk();
        let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(DMA_BUF);
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut info = vk::MemoryAllocateInfo::default()
            .allocation_size(length)
            .memory_type_index(type_index)
            .push_next(&mut export);
        if shared.device_address {
            info = info.push_next(&mut flags);
        }
        // SAFETY: the device is live; the type exports dma-bufs of a buffer's memory (above).
        let memory =
            unsafe { crate::creation::allocate_memory(raw_device, &info) }.map_err(|error| {
                Unmade::passing(format!(
                    "its exportable allocation of {length} bytes of type {type_index} failed: \
                     {error:?}"
                ))
            })?;
        let free = || {
            // SAFETY: the allocation is unused.
            unsafe { release::object(raw_device, memory) };
        };
        // SAFETY: the allocation is host-visible, unmapped, and `length` bytes long.
        let mapped = unsafe {
            raw_device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        };
        exit_guard::called(Call::MapMemory);
        let Some(start) = mapped
            .ok()
            .and_then(|mapped| NonNull::new(mapped.cast::<u8>()))
        else {
            free();
            return Err(Unmade::passing(format!(
                "its allocation of type {type_index} was not mapped"
            )));
        };
        // SAFETY: the importer is this device's; the allocation was made exportable as a dma-buf.
        let fd = match unsafe { importer.exported_dma_buf(raw_device, memory) } {
            Ok(fd) => fd,
            Err(error) => {
                free();
                return Err(Unmade::passing(format!(
                    "its allocation's dma-buf was not exported: {error:?}"
                )));
            }
        };
        Ok(ExportedPages {
            // A holder's handle (`device/holds.rs`): the pages are a no-copy buffer's keep-alive,
            // which a handler thread drops once the buffer's import is freed, past the program's
            // drop of the buffer; the program's last drop of the device waits for it then.
            device: device.held_for_release(),
            memory,
            start,
            length,
            fd,
        })
    }

    /// The host address of the pages' first byte.
    pub(super) fn start(&self) -> NonNull<u8> {
        self.start
    }

    /// A source of dma-bufs of the pages: each request over them from their first byte, of at
    /// most their length, answered with one chunk of their dma-buf, a duplicate of its descriptor
    /// of its own.
    pub(super) fn source(&self) -> DmaBufSource {
        let start = self.start.as_ptr() as u64;
        let length = self.length;
        let fd = self.fd.try_clone();
        Arc::new(move |request: DmaBufRequest| {
            let fd = fd
                .as_ref()
                .map_err(|error| std::io::Error::new(error.kind(), "no dma-buf"))?;
            if request.start.as_ptr() as u64 != start || request.length > length {
                return Err(std::io::Error::other(
                    "the check's dma-buf holds its pages from their first byte alone",
                ));
            }
            Ok(vec![DmaBufChunk {
                file: Arc::new(fd.try_clone()?),
                offset: 0,
                length: request.length,
            }])
        })
    }
}

#[cfg(unix)]
impl Drop for ExportedPages {
    fn drop(&mut self) {
        let raw_device = self.device.shared().vk();
        // SAFETY: nothing is bound to the allocation, and no command reads it: the buffer and the
        // images over its dma-buf have their own imports, and these, the buffer's keep-alive, go
        // once its import was freed. Freeing it unmaps it.
        unsafe { release::object(raw_device, self.memory) };
    }
}
