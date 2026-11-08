//! Dma-bufs of host pages (a runtime extension: Metal has none), the last resort of a no-copy
//! import (`host_memory.rs`) on a device that refuses the pages' pointer, as RADV refuses any
//! shared mapping (a `memfd`'s among them). A device that lists `VK_EXT_external_memory_dma_buf`
//! and `VK_KHR_external_memory_fd` enables both, outside the vocabulary
//! (`device::create::dma_buf_names`), and has a dma-buf importer; one that imports the pointer
//! never calls a source. The caller's contract is [`DmaBufSource`]'s, [`DmaBufRequest`]'s and
//! `HostMemory::with_dma_buf`'s; this is how the runtime keeps it.
//!
//! - **The imports.** Each chunk is its own memory, of the best type of the buffer's memory class
//!   (`memory.rs`'s `candidates`: host-visible and coherent, never a device-coherent type of
//!   `VK_AMD_device_coherent_memory`, which the runtime does not enable) among those
//!   `vkGetMemoryFdPropertiesKHR` and the buffer take. A source that was called is kept with the
//!   imports ([`DmaBufPages`]) and dropped with them.
//! - **The buffer** is a `VkBuffer` made for dma-bufs over the pages from their first byte, so its
//!   bytes lie at their offset in their first page whatever the chunks, and a rebind between one
//!   chunk and several keeps their place in it. **One chunk** gives a plain buffer bound at the
//!   import's start. **Several chunks** give a sparse one (`SPARSE_BINDING`), each chunk bound at
//!   its offset from the request's start by one `vkQueueBindSparse` on the device's first queue
//!   (`device/submit.rs`): the call does not wait for it, which would wait behind that queue's
//!   work in flight; every later submission on every queue waits for it until it completed, and
//!   the buffer's release does where none did. The granule is the alignment of such a buffer
//!   where the device has `sparseBinding`, its queue family binds sparse memory and it imports
//!   dma-bufs for such a buffer (64 KiB on RADV of Mesa 26.2.4, 4 KiB on 26.0.8); elsewhere the
//!   request's length, so that exactly one chunk is taken (NVIDIA's sparse buffers take no type
//!   its dma-bufs import as; it imports the pointer anyway). The padding past the pages lies
//!   inside the `VkBuffer` but past the bytes a copy or a binding reaches.
//! - **The bytes.** The runtime never maps a dma-buf: `contents` stays the caller's pointer, and
//!   the CPU's writes through it and the GPU's through the imports meet in the same pages with no
//!   flush or invalidate (the types are host-coherent).
//! - **Textures** placed in a heap over such pages (and their views) that Metal does not tile
//!   there (one it keeps losslessly compressed, one of a format or shape whose tiles are not
//!   measured: `texture/heap.rs`; one Metal tiles is a shadow of its elements in Metal's tiles),
//!   where the device also lists
//!   `VK_EXT_image_drm_format_modifier` and `VK_EXT_queue_family_foreign` (enabled beside the
//!   route, `create.rs`'s `dma_buf_image_names`), are each an image of `DRM_FORMAT_MOD_LINEAR` made
//!   for dma-bufs and bound to a dedicated import (`VkMemoryDedicatedAllocateInfo`) of the one
//!   dma-buf the source makes of the pages its memory requirements cover
//!   ([`Device::dma_buf_image`]). Its one plane starts at the texels' offset in their first page,
//!   its rows the driver's own (`VkImageDrmFormatModifierListCreateInfoEXT`), as a texture placed
//!   in a heap over pages imported by their pointer is the driver's linear image. Its features are
//!   the modifier's (64-bit where the device has `format_feature_flags2`: `modifier_features`),
//!   checked as a linear image's are against the optimal tiling's for the texture's usage, and its
//!   limits the format query's for that modifier and dma-bufs. RADV (Mesa 26.2.4 and 26.0.8) lays
//!   a plane only at an offset that is a multiple of 256 bytes and rows a multiple of 256 apart,
//!   and refuses any other layout (`VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT`): the
//!   runtime asks the driver and takes its answer, never a rule of its own. The source is called
//!   only once the image is made and checked; the import keeps it, and goes after the image.
//!   Counted: `WorkCounts::dma_buf_images`. A texture over a no-copy buffer's dma-buf pages takes
//!   the same image over a dma-buf of its own pages where it is taken in place (`in_place.rs`,
//!   `texture/in_place_image.rs`), and else is a shadow (`texture.rs`'s `storage_over_buffer`),
//!   whose copies go through the buffer's import.
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
//! - **Refused** (the pointer's refusal: `Unsupported`, `HostMemory`): a source that fails, chunks
//!   the device imports as no type the buffer takes, a driver that refuses a chunk, and a sparse
//!   buffer whose alignment misses the granule asked. Counted: `WorkCounts::dma_buf_imports` (each
//!   chunk) and `WorkCounts::dma_buf_sparse_buffers`.

use super::buffer::{usage, Pages};
use super::buffer_descriptor::HostBuffer;
use super::device::lock;
use super::error::{internal, invalid, vulkan, Error, MemoryKind, Problem};
use super::exit_guard::{self, Call};
use super::host_memory::{based_size, import_refused, not_importable, ImportWindow, Importer};
use super::host_residency::Again;
use super::image_plan::{ImageFacts, ImagePlan};
use super::linear_plan::{check_linear_limits, tiling_lack, viewed_formats, LinearLack};
use super::memory::{ImportFailure, MemoryClass, Placement};
use super::program_objects::ProgramObject;
use super::Device;
use crate::release;
use ash::vk;
use metal2vulkan::error::RuntimeDriverDefect;
use metal2vulkan::vk::{Format, FormatFeatureFlags2, ImageCreateFlags, SampleCountFlags};
use std::ops::Range;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

/// A runtime extension (Metal has none): makes dma-bufs of the host pages a no-copy import
/// covers, for a device that refuses the pages' pointer
/// ([`HostMemory::with_dma_buf`](crate::HostMemory::with_dma_buf)).
///
/// Called on the thread making the object, with no lock of the runtime's held: for a buffer or
/// heap (or its rebind), only where the device refuses the pointer and imports dma-bufs for it;
/// then once for each image the runtime makes over a texture's own pages in them (over the
/// buffer, placed in the heap, a view of one, or remade by a rebind). A source the runtime never
/// calls is dropped with the call; one it called is kept with the imports it made, until they go.
///
/// It returns dma-bufs of the pages asked for ([`DmaBufRequest`]) in chunks ([`DmaBufChunk`]), in
/// order: each chunk's length a positive multiple of the host page, at most its dma-buf's size,
/// and every boundary between two chunks, as an offset from the request's `start`, a multiple of
/// its `granule`. The descriptors are the runtime's from then on: Vulkan's once imported, closed
/// otherwise. An `Err` keeps the pointer's refusal (`Unsupported`, `HostMemory`), or makes a
/// texture a shadow of the bytes instead (announced `shadowed-texture`); chunks that break these
/// rules or the request's, or a descriptor that is no dma-buf, are refused as misuse (`Invalid`,
/// `Problem::DmaBuf`).
pub type DmaBufSource =
    Arc<dyn Fn(DmaBufRequest) -> std::io::Result<Vec<DmaBufChunk>> + Send + Sync + 'static>;

/// What a [`DmaBufSource`] is asked for: `length` bytes of host pages from `start`, both multiples
/// of the host page, to be returned as chunks split only at multiples of `granule` from `start`.
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
}

// SAFETY: a request is an address and two lengths, plain data: nothing is reached through the
// address by having the request.
unsafe impl Send for DmaBufRequest {}
// SAFETY: as above.
unsafe impl Sync for DmaBufRequest {}

/// One chunk of a [`DmaBufSource`]'s answer: a dma-buf whose bytes `0..length` are the requested
/// pages from where the chunks before it end.
#[derive(Debug)]
pub struct DmaBufChunk {
    /// The dma-buf.
    pub fd: OwnedFd,
    /// The bytes of the chunk, a multiple of the host page.
    pub length: u64,
}

/// The imports of the chunks a no-copy buffer's host pages lie in, and the source that made them:
/// held by the pages' placement (`memory.rs`), so by the buffers and images over them, and freed,
/// the source dropped, once the last of those goes. Every `VkBuffer` bound to them is destroyed
/// first (`buffer.rs`'s `HostImport`).
///
/// The last holder may go on a runtime thread (a command buffer's completion lets go of the
/// buffer): the imports are freed inside a section of the exit guard, as every Vulkan call there
/// is, and the source, the program's code, is dropped after it as the program's
/// (`exit_guard::program`), its panic caught. While the process exits both are left to its end.
pub(super) struct DmaBufPages {
    device: Device,
    memories: Vec<vk::DeviceMemory>,
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
    /// What waits for the imports' free ([`DmaBufPages::after_free`]): the caller's keep-alive
    /// and the deallocators of the pages that went, let go of once the imports are freed.
    after_free: Mutex<Vec<Box<dyn Send>>>,
}

impl Drop for DmaBufPages {
    fn drop(&mut self) {
        // The Vulkan teardown is a section of the exit guard.
        let Some(busy) = exit_guard::enter() else {
            // The process is exiting: the imports stay to its end, and the source and what waits
            // for their free with them.
            if let Some(source) = self.source.take() {
                std::mem::forget(source);
            }
            let after_free = self.after_free.get_mut();
            for after in after_free.unwrap_or_else(PoisonError::into_inner).drain(..) {
                std::mem::forget(after);
            }
            return;
        };
        let shared = self.device.shared();
        let raw_device = shared.vk();
        if let Some(log) = crate::memory_log::MemoryLog::new(shared.instance.sink()) {
            use ash::vk::Handle;
            let memories: Vec<String> = (self.memories.iter())
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
        for &memory in &self.memories {
            // SAFETY: nothing is bound to the imports any more and no command reads them (the
            // struct's docs).
            unsafe { release::object(raw_device, memory) };
        }
        (shared.counts).dma_buf_memories_freed(self.of_image, self.memories.len());
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
            of_image,
            source: Some(ProgramObject::new(source)),
            start: request.start.as_ptr() as u64,
            length: request.length,
            bound: None,
            kept_for_fds: false,
            after_free: Mutex::new(Vec::new()),
        }
    }

    /// Let `after` go once the imports are freed: as their last holder lets go of them, which
    /// may be a moment after the caller (an import going, `buffer.rs`'s `Imported`) lets go of
    /// its own hold.
    pub(super) fn after_free(&self, after: Box<dyn Send>) {
        lock(&self.after_free).push(after);
    }

    /// Hold `memory`, imported from one of the source's dma-bufs, until the pages go; counted
    /// live.
    fn hold(&mut self, memory: vk::DeviceMemory) {
        self.memories.push(memory);
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
            if waited != Err(vk::Result::TIMEOUT) {
                return;
            }
        }
    }

    /// Whether the pages hold the `length` bytes from host address `start`.
    fn holds(&self, start: u64, length: u64) -> bool {
        let end = start.checked_add(length);
        start >= self.start && end.is_some_and(|end| end <= self.start + self.length)
    }
}

impl std::fmt::Debug for DmaBufPages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DmaBufPages")
            .field("memories", &self.memories)
            .finish_non_exhaustive()
    }
}

/// Whether chunks of `lengths` bytes, in order, answer `request` as the [`DmaBufSource`] contract
/// allows on a host of `page`-byte pages: at least one, each a positive multiple of the page;
/// every boundary between two a multiple of the granule from the request's start; together the
/// request's length rounded up to the granule, or, one chunk, the length itself.
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

/// Whether `chunks` answer `request` as the [`DmaBufSource`] contract allows on a host of
/// `page`-byte pages ([`chunks_cover`]), each a dma-buf of at least its length.
fn answers(chunks: &[DmaBufChunk], request: &DmaBufRequest, page: u64) -> bool {
    let lengths: Vec<u64> = chunks.iter().map(|chunk| chunk.length).collect();
    let held = |chunk: &DmaBufChunk| {
        dma_buf_length(chunk.fd.as_fd()).is_some_and(|bytes| chunk.length <= bytes)
    };
    chunks_cover(&lengths, request, page) && chunks.iter().all(held)
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
#[cfg(not(target_os = "linux"))]
fn dma_buf_length(_fd: BorrowedFd<'_>) -> Option<u64> {
    None
}

/// How long one wait for the sparse binding lasts before it is asked again.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// The handle type of every import of a dma-buf, and of every buffer and image made for one.
const DMA_BUF: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;

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
    /// (`HostMemory::with_dma_buf`).
    pub(super) unsafe fn import_from_source(
        &self,
        call: &'static str,
        importer: &Importer,
        pointer: NonNull<u8>,
        length: u64,
        window: &ImportWindow,
        source: DmaBufSource,
        class: MemoryClass,
        keep: bool,
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
        let request = DmaBufRequest {
            start,
            length: window.size,
            granule,
        };
        let chunks = source(request).map_err(|_| not_importable())?;
        if !answers(&chunks, &request, importer.page) {
            return Err(invalid(call, Problem::DmaBuf));
        }
        // Duplicates of the descriptors, kept where the device may release the import: the
        // import takes the originals.
        let kept: Option<Vec<(OwnedFd, u64)>> = keep
            .then(|| {
                (chunks.iter())
                    .map(|chunk| Some((chunk.fd.try_clone().ok()?, chunk.length)))
                    .collect()
            })
            .flatten();
        let kept_source = kept.as_ref().map(|_| Arc::clone(&source));
        let mut pages = DmaBufPages::new(self, source, request, false);
        // SAFETY: the caller's.
        let (buffer, memory, type_index) =
            unsafe { self.chunks_buffer(importer, chunks, &mut pages, length, window, class) }?;
        let counts = &self.shared().counts;
        counts.dma_buf_imported(pages.memories.len());
        if pages.bound.is_some() {
            counts.dma_buf_sparse_buffer_made();
        }
        let again = kept.zip(kept_source).and_then(|(chunks, source)| {
            Again::dma_bufs(self, chunks, source, request, (length, *window, class))
        });
        if keep && again.is_none() {
            pages.kept_for_fds = true;
            counts.host_import_kept_for_fds(true);
        }
        let placement = Placement::dma_buf(Arc::new(pages), memory, type_index, window.size, start);
        Ok((buffer, placement, again))
    }

    /// `span`, host pages of a guest RAM (`host_ram.rs`), imported as the one dma-buf of them
    /// that the RAM's `source` makes: a chunk of the RAM or an object's run inside one, asked for
    /// with a granule of its whole length, so that one import holds it. The `VkBuffer` over the
    /// whole span, made for dma-bufs and bound to the import at its start, and its placement,
    /// which keeps the import and the source. Refused, everything made released, as
    /// [`Device::import_from_source`] is, where the device makes no such buffer, the source fails
    /// or answers more than one chunk, or the driver refuses the dma-buf.
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
    ) -> Result<(vk::Buffer, Placement), Error> {
        if !importer.imports_dma_bufs()
            || !self.imports_dma_buf_buffers(vk::BufferCreateFlags::empty())
        {
            return Err(not_importable());
        }
        let start = NonNull::new(span.start as *mut u8).ok_or_else(not_importable)?;
        let length = span.end - span.start;
        let request = DmaBufRequest {
            start,
            length,
            granule: length,
        };
        let chunks = source(request).map_err(|_| not_importable())?;
        if chunks.len() != 1 || !answers(&chunks, &request, importer.page) {
            return Err(invalid("use_host_ram", Problem::DmaBuf));
        }
        let window = ImportWindow {
            start: span.start,
            size: length,
            offset: 0,
        };
        let mut pages = DmaBufPages::new(self, Arc::clone(source), request, false);
        // SAFETY: the caller's.
        let (buffer, memory, type_index) = unsafe {
            self.chunks_buffer(
                importer,
                chunks,
                &mut pages,
                length,
                &window,
                MemoryClass::Shared,
            )
        }?;
        self.shared().counts.dma_buf_imported(pages.memories.len());
        let placement = Placement::dma_buf(Arc::new(pages), memory, type_index, length, start);
        Ok((buffer.raw, placement))
    }

    /// The pages of a released import made again from the dma-bufs `chunks` it kept, which
    /// `source` made for `request` (`host_residency.rs`), as `again` made it: a new `VkBuffer`
    /// made and bound as the first was, over imports of the same dma-bufs, so over the pages they
    /// hold whatever the address maps now, at the same offsets in the same memory type. Never
    /// calls the source. Refused (out of device memory, or the device lost) where the device
    /// cannot import them again, or lays them otherwise.
    pub(super) fn import_dma_bufs_again(
        &self,
        chunks: &[(OwnedFd, u64)],
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
        let chunks = chunks
            .iter()
            .map(|(fd, length)| {
                let fd = fd
                    .try_clone()
                    .map_err(|_| Error::OutOfMemory(MemoryKind::Host))?;
                Ok(DmaBufChunk {
                    fd,
                    length: *length,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let mut pages = DmaBufPages::new(self, Arc::clone(source), request, false);
        let window = &again.window;
        // SAFETY: the dma-bufs are those the first import took, which hold the pages it was
        // made over for as long as they are open (the import holds them).
        let (buffer, memory, type_index) = unsafe {
            self.chunks_buffer(
                importer,
                chunks,
                &mut pages,
                again.length,
                window,
                again.class,
            )
        }
        .map_err(|error| match error {
            Error::DeviceLost => Error::DeviceLost,
            other => refused(other),
        })?;
        let start = request.start;
        let placement = Placement::dma_buf(Arc::new(pages), memory, type_index, window.size, start);
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
    /// first byte, its bytes `window.offset` bytes into it whatever the chunks, made for
    /// dma-bufs, each of `chunks` imported (noted in `pages`) as memory of `class` and bound at
    /// its offset from the window's start: one chunk by `vkBindBufferMemory` to a plain buffer,
    /// several by a sparse binding of a `SPARSE_BINDING` one (`Device::bind_sparse_buffer`, noted
    /// in `pages`). The buffer, its one import (`VK_NULL_HANDLE` for several) and the first
    /// import's type. Refused, the buffer released, where the bytes padded to 16 run past the
    /// chunks, the buffer's memory does not fit them or its alignment misses their boundaries, a
    /// chunk's types take none of the class's the buffer takes, the driver refuses a chunk, or
    /// the binding fails.
    ///
    /// # Safety
    ///
    /// As [`Device::import_from_source`]'s; `chunks` meet [`chunks_cover`].
    unsafe fn chunks_buffer(
        &self,
        importer: &Importer,
        chunks: Vec<DmaBufChunk>,
        pages: &mut DmaBufPages,
        length: u64,
        window: &ImportWindow,
        class: MemoryClass,
    ) -> Result<(HostBuffer, vk::DeviceMemory, u32), Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let lengths: Vec<u64> = chunks.iter().map(|chunk| chunk.length).collect();
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
            let fits = match sparse {
                // Each chunk binds whole at its offset, which the alignment must divide, as it
                // must divide each chunk's length.
                true => lengths
                    .iter()
                    .all(|length| length.is_multiple_of(alignment)),
                false => requirements.size <= size,
            };
            if !fits {
                return Err(not_importable());
            }
            let memories = self.import_chunks(importer, chunks, pages, &requirements, class)?;
            if !sparse {
                // SAFETY: the buffer is unbound, made for dma-bufs; the memory is of a type it
                // takes and holds its requirements from 0.
                unsafe { crate::creation::bind_buffer(raw_device, raw, memories[0].0, 0) }
                    .map_err(vulkan)?;
                return Ok(memories[0]);
            }
            let mut offset = 0;
            let binds: Vec<vk::SparseMemoryBind> = (memories.iter().zip(&lengths))
                .map(|(&(memory, _), &length)| {
                    let bind = vk::SparseMemoryBind::default()
                        .resource_offset(offset)
                        .size(length)
                        .memory(memory)
                        .memory_offset(0);
                    offset += length;
                    bind
                })
                .collect();
            pages.bound = Some(self.bind_sparse_buffer(raw, &binds)?);
            Ok((vk::DeviceMemory::null(), memories[0].1))
        })();
        match bound {
            // The bytes lie `window.offset` bytes into the buffer, which is bound from the
            // window's first byte; it holds the pages, and its padding past them is no one's.
            Ok((memory, type_index)) => {
                let buffer = HostBuffer {
                    raw,
                    base: window.offset,
                    size: window.size,
                    requirements,
                };
                Ok((buffer, memory, type_index))
            }
            Err(error) => {
                // SAFETY: no command uses the buffer; a failed binding left nothing pending.
                unsafe { release::object(raw_device, raw) };
                Err(error)
            }
        }
    }

    /// Each of `chunks` imported as memory of `class` of a type both it and a buffer of
    /// `requirements` take, in order, noted in `pages` (which frees them), with its type.
    /// Refused (`Unsupported`, `HostMemory`) where a chunk's types take none of the class's the
    /// buffer takes, or the driver refuses it; the descriptors not yet imported are closed then.
    fn import_chunks(
        &self,
        importer: &Importer,
        chunks: Vec<DmaBufChunk>,
        pages: &mut DmaBufPages,
        requirements: &vk::MemoryRequirements,
        class: MemoryClass,
    ) -> Result<Vec<(vk::DeviceMemory, u32)>, Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let mut imported = Vec::with_capacity(chunks.len());
        for DmaBufChunk { fd, length } in chunks {
            // SAFETY: the importer is this device's, and the descriptor a dma-buf.
            let types = unsafe { importer.dma_buf_types(raw_device, fd.as_fd()) }
                .map_err(import_refused)?;
            let allowed = types & requirements.memory_type_bits;
            let (memory, type_index) = lock(&shared.memory)
                .import_dma_buf(raw_device, fd, length, (allowed, class), None)
                .map_err(|failure| match failure {
                    ImportFailure::NoMemoryType => not_importable(),
                    ImportFailure::Vulkan(error) => import_refused(error),
                })?;
            pages.hold(memory);
            imported.push((memory, type_index));
        }
        Ok(imported)
    }

    /// The granule a dma-buf source's chunks split at, where the device assembles a buffer of
    /// several: it has `sparseBinding`, its queue family binds sparse memory, and it imports
    /// dma-bufs for a sparse buffer without a dedicated allocation; then the alignment of such a
    /// buffer (of the host page), rounded up to the host page. `None` elsewhere. Asked of the
    /// device once ([`Importer::sparse_granule`]).
    fn sparse_granule(&self, importer: &Importer) -> Option<u64> {
        *importer.sparse_granule.get_or_init(|| {
            let shared = self.shared();
            let flags = vk::BufferCreateFlags::SPARSE_BINDING;
            if !shared.sparse.binds() || !self.imports_dma_buf_buffers(flags) {
                return None;
            }
            let raw_device = shared.vk();
            let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(DMA_BUF);
            let info = shared.sharing().buffer(
                vk::BufferCreateInfo::default()
                    .flags(flags)
                    .size(importer.page)
                    .usage(usage(shared.device_address))
                    .push_next(&mut external),
            );
            // SAFETY: the device is live, has `sparseBinding`, and `info` and its chain are valid.
            let raw = unsafe { crate::creation::buffer(raw_device, &info) }.ok()?;
            shared.counts.buffer_made(info.usage);
            // SAFETY: `raw` is a live buffer of this device.
            let alignment = unsafe { raw_device.get_buffer_memory_requirements(raw) }.alignment;
            // SAFETY: the buffer is unused.
            unsafe { release::object(raw_device, raw) };
            alignment
                .max(1)
                .checked_next_multiple_of(importer.page)
                .filter(|granule| granule.is_multiple_of(alignment.max(1)))
        })
    }

    /// Whether the device imports a dma-buf for a no-copy buffer's `VkBuffer` made with `flags`
    /// without a dedicated allocation (`vkGetPhysicalDeviceExternalBufferProperties`), as it must
    /// import host allocations for one (`Importer::new`).
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

/// `DRM_FORMAT_MOD_LINEAR` of Linux's `drm_fourcc.h`: one plane of rows one after another, each
/// the plane's pitch from the last.
pub(super) const DRM_FORMAT_MOD_LINEAR: u64 = 0;

/// Where a texture's texels lie in dma-buf pages, for its image over them
/// ([`Device::dma_buf_image`]).
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

/// A texture's image over dma-buf pages, made and bound ([`Device::dma_buf_image`]): its
/// `VkImage`, which the caller destroys, and the import of its pages, which it drops after.
pub(super) struct DmaBufImage {
    pub(super) raw: vk::Image,
    pub(super) import: Arc<DmaBufPages>,
}

/// The layout the driver gives a `DRM_FORMAT_MOD_LINEAR` image of its own choosing
/// ([`Device::dma_buf_natural_layout`]): its plane's row and array pitches, and its memory's
/// bytes from the plane's first byte.
pub(super) struct NaturalLayout {
    pub(super) row_pitch: u64,
    pub(super) slice_pitch: u64,
    pub(super) size: u64,
}

/// The one memory plane of an image of `DRM_FORMAT_MOD_LINEAR`, whose layout Vulkan reports
/// (`VUID-vkGetImageSubresourceLayout-tiling-09433`).
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
    Pages,
    /// The caller's source made no one dma-buf of the image's pages: it failed (a source of
    /// `udmabuf`s does where the pages are more runs of its file than one `udmabuf` takes), or it
    /// answered with other than one chunk.
    Source,
    /// The driver refuses the dma-buf, its import, or its binding.
    Import,
}

/// An image over dma-buf pages made and not yet bound: its requirements, and the request for
/// its pages.
pub(super) struct Unbound {
    raw: vk::Image,
    requirements: vk::MemoryRequirements,
    request: DmaBufRequest,
}

impl Unbound {
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
    /// length }`), of the pages' imports' memory type; counted (`WorkCounts::dma_buf_images`). It
    /// starts `UNDEFINED`, as Vulkan makes an image of external memory, for its first move to
    /// acquire from `VK_QUEUE_FAMILY_FOREIGN_EXT` (`FirstMove::acquired`).
    ///
    /// Why not, for the notice of the path taken instead (`shadowed-texture`), where the device
    /// makes none: it does not enable the extensions; the plan has several levels, depth or
    /// samples, or a mutable format with no list of its views; the modifier's tiling lacks a
    /// feature the plan's optimal tiling has for its usage; the format query refuses such an
    /// image, or the plan passes its limits; the driver refuses the plane layout or lays it out
    /// otherwise; the image's memory runs past the pages or takes no memory of their type; the
    /// source fails or answers against its contract; the driver refuses the import. Everything
    /// made is released then. The source is called at most once, once the image is made and
    /// checked, with no lock of the runtime's held (the caller holds none).
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

    /// [`Device::dma_buf_image`]'s image made and checked, unbound, with the request for its
    /// pages; why not where the device makes none before the source is asked.
    pub(super) fn dma_buf_image_unbound(
        &self,
        plan: &ImagePlan,
        facts: &ImageFacts<'_>,
        texels: &DmaBufTexels<'_>,
    ) -> Result<Result<Unbound, (DmaBufRefusal, String)>, Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let (page, view_formats) = match self.dma_buf_image_checks(plan, facts)? {
            Ok(checked) => checked,
            Err(why) => return Ok(Err((DmaBufRefusal::Image, why))),
        };
        if let Some(why) = self.explicit_plane_rows(plan) {
            return Ok(Err((DmaBufRefusal::Pitch, why)));
        }
        let offset = texels.at % page;
        let array_pitch = if plan.layers > 1 {
            texels.slice_pitch
        } else {
            0
        };
        let planes = [vk::SubresourceLayout {
            offset,
            size: 0,
            row_pitch: texels.row_pitch,
            array_pitch,
            depth_pitch: 0,
        }];
        let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(DRM_FORMAT_MOD_LINEAR)
            .plane_layouts(&planes);
        let dma_buf = ModifierMemory::DmaBuf;
        let raw = match self.drm_linear_image(plan, &view_formats, &mut explicit, dma_buf)? {
            Ok(raw) => raw,
            Err(error) => {
                return Ok(Err((
                    DmaBufRefusal::Pitch,
                    format!(
                        "the driver refuses a DRM_FORMAT_MOD_LINEAR plane at offset {offset} of \
                         its first page with rows {} bytes apart ({error:?})",
                        texels.row_pitch
                    ),
                )))
            }
        };
        let refuse = |kind: DmaBufRefusal, why: String| {
            // SAFETY: the image is live and unbound, and no command uses it.
            unsafe { release::object(raw_device, raw) };
            Ok(Err((kind, why)))
        };
        // Bound to a dedicated import, whether or not the driver asks one.
        // SAFETY: `raw` is a live image of this device.
        let requirements = unsafe { raw_device.get_image_memory_requirements(raw) };
        // SAFETY: `raw` is a live image of a DRM format modifier's tiling, whose one memory plane
        // is plane 0 (`VUID-vkGetImageSubresourceLayout-tiling-09433`).
        let report = unsafe { raw_device.get_image_subresource_layout(raw, memory_plane_0()) };
        exit_guard::called(Call::GetImageSubresourceLayout);
        if report.offset != offset
            || report.row_pitch != texels.row_pitch
            || (plan.layers > 1 && report.array_pitch != array_pitch)
        {
            return refuse(
                DmaBufRefusal::Pitch,
                format!(
                "the driver lays the plane out at offset {}, rows {} and slices {} bytes apart, \
                 where the texels lie at offset {offset}, rows {} and slices {array_pitch} apart",
                report.offset, report.row_pitch, report.array_pitch, texels.row_pitch
            ),
            );
        }
        // The pages from the texels' first, through the image's memory.
        let first = texels.at - offset;
        let length = match requirements.size.checked_next_multiple_of(page) {
            Some(length) if texels.pages.holds(first, length) => length,
            _ => {
                return refuse(
                    DmaBufRefusal::Pages,
                    format!(
                    "its image's memory, {} bytes from the first page of its texels, runs past \
                     the pages the buffer's range touches",
                    requirements.size
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
            },
        }))
    }

    /// `unbound`'s image bound to a dedicated import of the dma-buf the caller's source makes of
    /// its pages, of `texels`' memory type, and counted; why not, the image released, where the
    /// source fails or answers against its contract or the driver refuses the import or binding.
    pub(super) fn bind_dma_buf_image(
        &self,
        unbound: Unbound,
        texels: &DmaBufTexels<'_>,
    ) -> Result<Result<DmaBufImage, (DmaBufRefusal, String)>, Error> {
        let Unbound {
            raw,
            requirements,
            request,
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
                DmaBufRefusal::Source,
                "the pages keep no source of dma-bufs".to_owned(),
            );
        };
        // The source is the program's code, called with no lock of the runtime's held.
        let chunks = match source(request) {
            Ok(chunks) => chunks,
            Err(error) => {
                return refuse(
                    DmaBufRefusal::Source,
                    format!(
                        "the caller's source made no dma-buf of its {} bytes of pages: {error}",
                        request.length
                    ),
                )
            }
        };
        // A request whose granule is its length takes one chunk (`chunks_cover`).
        let answered = answers(&chunks, &request, importer.page);
        let (true, Ok([chunk])) = (answered, <[DmaBufChunk; 1]>::try_from(chunks)) else {
            return refuse(
                DmaBufRefusal::Source,
                "the caller's source answered other than one dma-buf of the pages asked".to_owned(),
            );
        };
        // SAFETY: the importer is this device's, and the descriptor a dma-buf.
        let types = match unsafe { importer.dma_buf_types(raw_device, chunk.fd.as_fd()) } {
            Ok(types) => types,
            Err(error) => {
                let why = format!("the driver refuses the dma-buf ({error:?})");
                return refuse(DmaBufRefusal::Import, why);
            }
        };
        // The pages' type (host-visible and coherent, as every class of theirs is), dedicated to
        // the image at its requirements' size exactly
        // (`VUID-VkMemoryDedicatedAllocateInfo-image-01433`), within the dma-buf's pages.
        let allowed = (types & (1 << texels.type_index), MemoryClass::Shared);
        let imported = lock(&shared.memory).import_dma_buf(
            raw_device,
            chunk.fd,
            requirements.size,
            allowed,
            Some(raw),
        );
        let memory = match imported {
            Ok((memory, _)) => memory,
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
        // SAFETY: the image is unbound; the memory is its dedicated allocation, of its size.
        if let Err(error) = unsafe { crate::creation::bind_image(raw_device, raw, memory, 0) } {
            // SAFETY: nothing is bound to the memory, and no command uses it.
            unsafe { release::object(raw_device, memory) };
            let why = format!("the driver refuses to bind the import ({error:?})");
            return refuse(DmaBufRefusal::Import, why);
        }
        shared.counts.dma_buf_image_made();
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

    /// A `DRM_FORMAT_MOD_LINEAR` image of `plan` for `memory` (dma-bufs, or host allocations),
    /// viewed as `view_formats`, its plane laid out as `modifier` says (an explicit layout, or the
    /// modifier listed for the driver to lay out), made `UNDEFINED` as Vulkan makes an image of
    /// external memory; or the driver's refusal of it.
    pub(super) fn drm_linear_image(
        &self,
        plan: &ImagePlan,
        view_formats: &[vk::Format],
        modifier: &mut impl vk::ExtendsImageCreateInfo,
        memory: ModifierMemory,
    ) -> Result<Result<vk::Image, vk::Result>, Error> {
        let shared = self.shared();
        let mut external =
            vk::ExternalMemoryImageCreateInfo::default().handle_types(memory.handle_type());
        let mut list = vk::ImageFormatListCreateInfo::default();
        let info = plan
            .create_info(shared.sharing(), view_formats, &mut list)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external)
            .push_next(modifier);
        // SAFETY: the device is live and enables the extension; `info` and its chain are valid
        // and within the device's limits for such an image (`modifier_image_limits`), its one
        // plane the modifier's one (`modifier_features`), made `UNDEFINED` for external memory.
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
    fn modifier_features(&self, format: Format) -> FormatFeatureFlags2 {
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
        let mut external =
            vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(memory.handle_type());
        let mut list = vk::ImageFormatListCreateInfo::default().view_formats(view_formats);
        let mut query = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(plan.format.into())
            .ty(plan.image_type.into())
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(plan.usage.into())
            .flags(plan.flags.into())
            .push_next(&mut external)
            .push_next(&mut modifier);
        if !view_formats.is_empty() {
            query = query.push_next(&mut list);
        }
        let mut imported = vk::ExternalImageFormatProperties::default();
        let mut properties = vk::ImageFormatProperties2::default().push_next(&mut imported);
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
        if let Err(error) = queried {
            return Err(format!(
                "the device's image format query reports no DRM_FORMAT_MOD_LINEAR image of {:?}, \
                 usage {:?}, flags {:?} for {} ({error:?})",
                plan.format,
                plan.usage,
                plan.flags,
                memory.words(),
            ));
        }
        let limits = properties.image_format_properties;
        let features = imported.external_memory_properties.external_memory_features;
        // An image for host allocations is bound at an offset of a buffer's import, which is no
        // dedicated allocation.
        let dedicated_only = memory == ModifierMemory::Host
            && features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY);
        if !features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE) || dedicated_only {
            return Err(format!(
                "it imports no {} for a DRM_FORMAT_MOD_LINEAR image of {:?} (importable {}, \
                 dedicated only {})",
                memory.words(),
                plan.format,
                features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE),
                features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
            ));
        }
        Ok(limits)
    }
}

impl Device {
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
}

/// The memory a `DRM_FORMAT_MOD_LINEAR` image is made for: a dma-buf of its own pages
/// ([`Device::dma_buf_image`]), or host pages imported by their pointer, where a texture's image in
/// place lays its rows out at Metal's pitch (`texture/in_place_image.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ModifierMemory {
    DmaBuf,
    Host,
}

impl ModifierMemory {
    /// The external memory handle type the image is made for.
    fn handle_type(self) -> vk::ExternalMemoryHandleTypeFlags {
        match self {
            ModifierMemory::DmaBuf => DMA_BUF,
            ModifierMemory::Host => vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
        }
    }

    /// The memory, in words.
    fn words(self) -> &'static str {
        match self {
            ModifierMemory::DmaBuf => "dma-bufs",
            ModifierMemory::Host => "host allocations",
        }
    }
}

/// An allocation of the device's own, host-visible and exported as a dma-buf, mapped: the pages of
/// the device's check of textures over dma-buf pages (`in_place_check.rs`), which the runtime
/// runs no `udmabuf` code to make. Freed (and so unmapped) as it drops.
pub(super) struct ExportedPages {
    device: Device,
    memory: vk::DeviceMemory,
    start: NonNull<u8>,
    length: u64,
    fd: OwnedFd,
}

// SAFETY: the allocation's handle and mapping are the device's, used by one holder at a time; the
// mapping's address is plain data.
unsafe impl Send for ExportedPages {}

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
    /// type), which left the check's class unchecked there. `None` where the device imports no
    /// dma-bufs, makes no images over them, exports no such memory but as a dedicated allocation,
    /// or refuses it.
    pub(super) fn make(device: &Device, length: u64) -> Option<ExportedPages> {
        let shared = device.shared();
        let importer = shared.host_import.as_ref().ok()?;
        if !importer.makes_dma_buf_images() {
            return None;
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
            return None;
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
                .map(|&(index, _)| index)?
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
        let memory = unsafe { crate::creation::allocate_memory(raw_device, &info) }.ok()?;
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
            return None;
        };
        // SAFETY: the importer is this device's; the allocation was made exportable as a dma-buf.
        let Ok(fd) = (unsafe { importer.exported_dma_buf(raw_device, memory) }) else {
            free();
            return None;
        };
        Some(ExportedPages {
            device: device.clone(),
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
    /// most their length, answered with one chunk of their dma-buf.
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
                fd: fd.try_clone()?,
                length: request.length,
            }])
        })
    }
}

impl Drop for ExportedPages {
    fn drop(&mut self) {
        let raw_device = self.device.shared().vk();
        // SAFETY: nothing is bound to the allocation, and no command reads it: the buffer and the
        // images over its dma-buf have their own imports, and these, the buffer's keep-alive, go
        // once its import was freed. Freeing it unmaps it.
        unsafe { release::object(raw_device, self.memory) };
    }
}
