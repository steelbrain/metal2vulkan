//! Buffers over the caller's host memory without a copy (`newBufferWithBytesNoCopy:`), through
//! `VK_EXT_external_memory_host`, or, where the device refuses the pointer, dma-buf memory of the
//! pages the caller imports (`VK_EXT_external_memory_dma_buf`, [`DmaBufSource`], `dma_buf.rs`).
//!
//! - **What Metal takes** (measured on an Apple M3 Max and an M4, plain and under API validation):
//!   any pointer and any length but 0; the buffer covers the whole host pages its range touches
//!   (`allocatedSize`), its `contents` is the pointer, and its `gpuAddress` keeps the pointer's
//!   offset in its page. `Private` and `Memoryless` storage are stopped.
//! - **The import.** The runtime imports the same pages, rounded out to the larger of the host page
//!   and the device's `minImportedHostPointerAlignment` ([`import_window`]), as one dedicated
//!   `VkDeviceMemory` of a host-visible coherent type both the pointer and the buffer allow, and
//!   binds the buffer at the pointer's offset in them. Pages are mapped whole, so the caller's
//!   guarantee for its bytes covers every page they touch; the GPU can reach the rest of those
//!   pages through an out-of-range access, as on Metal. The `VkBuffer` holds the rest of the
//!   pages from the pointer where the device's requirements of it fit them, and a copy out of
//!   the buffer reads them past its length, the caller's bytes, as plain Metal reads them
//!   (`Buffer::copy_reach`). The import is never a dedicated allocation in Vulkan's sense
//!   (`VkMemoryDedicatedAllocateInfo`), which host memory cannot be.
//! - **What it refuses**, typed: a device without the extension (`Unsupported` naming it), and
//!   what this device cannot import exactly (`Unsupported`, `not_representable`): an import
//!   alignment above the host page whose rounding would take pages the range does not touch, an
//!   offset in the pages that misses the buffer's memory alignment, no memory type both allow, or
//!   a driver that refuses the pointer.
//! - **Pages the process cannot write** are refused typed on Linux, before the buffer's own import,
//!   unless the caller gave a dma-buf source the device can use: then they are imported from it
//!   alone, never by their pointer, and the device announces `read-only-pages-written`, as the
//!   GPU's writes reach them where Metal's leave the CPU's view of them as it was
//!   ([`refuse_unwritable`] asks the kernel for the process's own mappings over the window, with
//!   `PROCMAP_QUERY` on `/proc/self/maps`, a file kept open for them (`kept_maps`), or reads
//!   that file's lines where the kernel predates the query, Linux 6.11;
//!   `WorkCounts::host_mappings_queried`). A buffer that lies in a live
//!   import of the same bytes (`host_aliases.rs`) asks nothing: it pins no page, that import's
//!   making asked for the same window, and [`HostMemory::new`]'s contract keeps imported pages
//!   writable. Nor does one over pages whose write permission the caller vouched for
//!   ([`HostMemory::assume_writable`]). Linux drivers pin user pages for writing: NVIDIA failed
//!   a read-only page as out of device memory, RADV refused it, and ANV imported it and lost the
//!   device at the GPU's first read. lavapipe read one, so there the rule refuses a little more
//!   than the driver would. On macOS, MoltenVK hands the range to Metal, which reads read-only
//!   pages, so there is no check.
//! - **What drivers answered**. Private anonymous memory was imported on every venue measured. RADV
//!   refused shared memory typed: anonymous shared memory, and any mapping of a file, a shared
//!   memory object or a `memfd` among them (QEMU's `memory-backend-memfd,share=on` guest RAM). It
//!   imports a host pointer through amdgpu's userptr, which the kernel takes only over anonymous
//!   pages. Mesa 26.2.4 answers `VK_ERROR_INVALID_EXTERNAL_HANDLE`, 26.0.8 `VK_ERROR_UNKNOWN`, and
//!   both are refusals ([`import_refused`]). NVIDIA and lavapipe imported the same shared memory.
//! - **Dma-bufs of the pages** ([`HostMemory::with_dma_buf`]), the last resort before that
//!   refusal: where the device refuses the pointer ([`refused`]: the import window, the buffer's
//!   binding in it, the memory types, or the driver) and imports dma-bufs, the buffer is bound in
//!   imports of the source's dma-bufs of the pages instead (`dma_buf.rs`). Every import the device
//!   takes by its pointer is made exactly as before, with no call more.
//! - **When**. The pages are imported when the buffer is made, on every platform, as Metal wires
//!   them then: the GPU uses the pages mapped at the pointer when the buffer was made, whatever is
//!   mapped there later, but on a driver whose pointer imports follow their address
//!   (`DriverFact::PointerImportsFollowTheirAddress`: ANV on i915, RADV, lavapipe), the import of
//!   the pointer reads the pages mapped there at each submission (a remap under a live buffer,
//!   which [`HostMemory::new`] forbids, reaches it there; `docs/FAST-PATHS.md`, known gaps). Memory
//!   the device cannot import is refused by the call that makes the buffer, as Metal answers nil
//!   there. The import costs work per page there: on macOS MoltenVK's
//!   `newBufferWithBytesNoCopy:` and its residency set's commit, about 11 to 20 ms per GiB (an M4,
//!   an M3 Max); Metal makes the buffer at once and pays at the first command using it (24 to 88 ms
//!   per GiB). The runtime once deferred the macOS import to the buffer's first GPU use, which
//!   saw pages remapped in between and lost the device where the import then failed.
//! - **The deallocator** runs once, with the pointer and the length, when the buffer's last
//!   reference goes: after the imported memory is freed. Where that release frees the import, the
//!   free and the deallocator after it run on the thread that drops it, before the release
//!   returns, as Metal runs the deallocator in the release (an M3 Max under macOS 27.0.1 ran an
//!   unused buffer's deallocator inside its final release, on the releasing thread, every time);
//!   the free waits there on the driver's locks a submission holds, ANV's device lock among them.
//!   Only where a keep-alive alone waits for the free ([`KeepAlive`], a runtime extension) do the
//!   free and the keep-alive's drop go to one of the device's handler threads, in a context of its
//!   own, the release returning first. With neither a deallocator nor a keep-alive the free is
//!   made in the release: the caller may unmap the pages once it returns, and
//!   `VK_EXT_external_memory_host` requires the imported range to stay valid for the imported
//!   memory's whole life. A committed
//!   command buffer holds the buffer until it completed and its completed handlers ran, so then it
//!   runs on one of the device's handler threads, in the command buffer's queue's context, after
//!   the GPU's last use, as Metal's runs on one of its threads: what its commands retained and its
//!   encoders' logs (`replay.rs`) go in its release, and no thread that submits command buffers
//!   or sees them complete holds a handle past it (`exit_guard::program_release`). One released
//!   while only command buffers not yet submitted reach it unnamed goes at once, as Metal's does
//!   (it runs in the release, on the releasing thread, while such a command buffer waits
//!   committed: probe `dealloc-probe3`, the M3 Max under macOS 27.0.1): on the releasing thread,
//!   or, where such a command buffer's root table held it until its first submission (a rebind's
//!   encoding again), on a handler thread then (`command_buffer/first_submission.rs`). A refused
//!   call never runs it.
//! - **A keep-alive** ([`KeepAlive`]). The import can outlive the buffer: a texture over the buffer
//!   that a command buffer submitted before its release may still read through its resource ID
//!   keeps its image, and the image the import, until that command buffer completes
//!   (`heap_sets.rs`). Vulkan requires imported pages to stay mapped while the import lives, so
//!   with no keep-alive the deallocator waits for the import's free, announced
//!   (`host-deallocator-delayed`). With a keep-alive, which the caller's mapping lives in, the
//!   deallocator runs at the release, as Metal's does, and the keep-alive is dropped once the
//!   import is freed: the pages must stay mapped until then. A buffer over the bytes another
//!   live buffer lies in shares its import (`host_aliases.rs`), whose pages that buffer's
//!   contract keeps mapped: its deallocator runs at its release either way, and so is its
//!   keep-alive dropped, after the deallocator. So is the keep-alive of a buffer whose import
//!   keeps another's already: every buffer in an import was made over the same bytes, so one
//!   keep-alive keeps them mapped for all, and an import keeps one at most, however many buffers
//!   come and go over its bytes while it lives (a consumer's buffer over a surface made at each
//!   present and released after the next one's). Counted: `WorkCounts::host_keep_alives_kept`
//!   (held now) and `WorkCounts::host_keep_alives_covered` (dropped at the release).

use super::device::lock;
use super::dirty_pages::{hook, DirtyHook, DirtyPages, DirtySource};
#[cfg(unix)]
use super::dma_buf::DmaBufSource;
use super::error::{vulkan, Error};
use super::exit_guard;
use super::loader_word::LoaderWord;
use super::program_objects::ProgramObject;
use super::work_counts::Counters;
use ash::vk;
use metal2vulkan::device::{EnableSet, VkExtension};
use metal2vulkan::error::{Feature, Unsupported};
use std::ffi::c_void;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// Host memory a buffer or heap wraps without a copy: a pointer whose bytes the caller vouches for
/// (the `pointer` of `newBufferWithBytesNoCopy:length:options:deallocator:`), and, as runtime
/// extensions, a source of dma-bufs of its pages for a device that refuses the pointer itself
/// ([`HostMemory::with_dma_buf`]) and a log of the writes to its pages
/// ([`HostMemory::with_dirty_pages`]).
pub struct HostMemory {
    pointer: NonNull<u8>,
    source: Option<Source>,
    /// The caller vouched that the process maps every page writable
    /// ([`HostMemory::assume_writable`]), so the runtime asks the kernel nothing of them.
    writable: bool,
    /// The caller's log of the pages written ([`HostMemory::with_dirty_pages`]).
    dirty: Option<Arc<dyn DirtyHook>>,
    /// The caller's gate over the import of the pages ([`HostMemory::with_import_gate`]).
    gate: Option<ImportGate>,
    /// The caller's resolver of the pages' runs in its guest RAM ([`HostMemory::with_runs`]).
    runs: Option<RunResolver>,
}

/// A consumer's resolver of a no-copy object's pages to the runs of guest RAM they are views of
/// (a runtime extension, [`HostMemory::with_runs`]): asked with the first byte and the length of
/// the whole host pages the runtime would import for the object (its import window, as an
/// [`ImportRequest`] names it), it answers the host addresses of the same bytes in the RAM's
/// ranges ([`HostRam`](crate::HostRam)), in the object's byte order, as runs of whole host pages
/// that together cover exactly the length asked; or `None` where the pages are no view of the
/// RAM (or not all of them, or some are read-only), which leaves the object on the runtime's
/// other paths. Asked at most once per object, as a no-copy buffer or heap is made over the
/// pages (a rebind to them does not ask it: its pages take the rebind's other paths), on the
/// calling thread, holding no lock of the runtime's, before the import gate ([`ImportGate`]);
/// only on a device that uses a RAM
/// ([`Device::use_host_ram`](crate::Device::use_host_ram)), and only where the object would import
/// the pages itself (not one that lies in a live import of the same bytes, nor one whose GPU
/// writes Metal's private option bit 24 drops). The answer is kept with the object until its
/// rebind or release; a later remap of the view does not change it. An answer that breaks this
/// contract is refused (`Invalid`, `Problem::HostRam`). No Metal counterpart: Metal wires the
/// pages at the pointer.
pub type RunResolver = Arc<dyn Fn(u64, u64) -> Option<Vec<Range<u64>>> + Send + Sync + 'static>;

/// A consumer's gate over the import of a no-copy object's pages (a runtime extension,
/// [`HostMemory::with_import_gate`]): asked once, just before the runtime imports the pages for a
/// no-copy buffer or heap made over them, or for a buffer's rebind to them
/// (`Buffer::rebind_bytes_no_copy`), with the pages the import would map. `true` lets the import
/// go on; `false` refuses it, and the buffer is a copied host buffer instead (the heap's own
/// buffer too, the rebind's new pages a copy of them), whose bytes the GPU uses in memory of the
/// device's own, kept coherent with the pages at each commit and each completion, so the pages
/// take no kernel mapping (`copied.rs`, docs/HOWTO.md, "Copied host buffers"). Called on the
/// thread that makes the buffer, heap or rebind, holding no lock of the runtime's. Not asked for a
/// buffer or rebind that lies in a live import (one made over the same bytes), nor where the device
/// refused the import before it (no host-pointer import, an import alignment the range cannot
/// meet), nor for memory whose GPU writes Metal's private option bit 24 drops, nor for a heap's
/// rebind, which imports without asking. An import it lets go on that then fails (out of memory,
/// another Vulkan error) makes the copy all the same. Every copy is announced
/// (`copied-host-pages`). No Metal counterpart.
pub type ImportGate = Arc<dyn Fn(ImportRequest) -> bool + Send + Sync + 'static>;

/// The pages an [`ImportGate`] is asked about: those an import would map for the buffer, from the
/// host page its first byte lies in through the one its last lies in.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImportRequest {
    /// The first byte of the first page.
    pub start: NonNull<u8>,
    /// The bytes of the pages.
    pub length: u64,
}

// SAFETY: a request is an address and a length, plain data: nothing is reached through the
// address.
unsafe impl Send for ImportRequest {}
// SAFETY: as above.
unsafe impl Sync for ImportRequest {}

/// What a [`HostMemory`] may carry besides its pointer: a [`DmaBufSource`] where the system has
/// file descriptors, nothing elsewhere.
#[cfg(unix)]
pub(super) type Source = DmaBufSource;

/// What a [`HostMemory`] may carry besides its pointer: nothing on this system.
#[cfg(not(unix))]
pub(super) type Source = std::convert::Infallible;

// SAFETY: a `HostMemory` is an address and the caller's guarantee for it (`HostMemory::new`),
// and a source that is `Send`; the runtime touches the bytes only as a buffer of them, under that
// guarantee.
unsafe impl Send for HostMemory {}
// SAFETY: as above; a shared `HostMemory` answers its pointer alone, and its source is reached
// only by value ([`HostMemory::into_parts`]).
unsafe impl Sync for HostMemory {}

impl std::fmt::Debug for HostMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostMemory")
            .field("pointer", &self.pointer)
            .field("dma_buf", &self.source.is_some())
            .field("writable", &self.writable)
            .field("dirty_pages", &self.dirty.is_some())
            .field("import_gate", &self.gate.is_some())
            .field("runs", &self.runs.is_some())
            .finish()
    }
}

impl HostMemory {
    /// The host memory at `pointer`, for a buffer of some length `length` made from it.
    ///
    /// # Safety
    ///
    /// The `length` bytes at `pointer`:
    /// - are mapped memory of this process, readable, and writable wherever the CPU or the GPU
    ///   writes the buffer;
    /// - stay mapped, at this address and with the same backing: with a keep-alive
    ///   ([`KeepAlive`]), until it is dropped; else until the buffer's deallocator runs; with
    ///   neither, until the buffer's last reference is dropped (a command buffer that uses the
    ///   buffer holds a reference until it completed) and every command buffer submitted before
    ///   a texture over the buffer was released completed;
    /// - are not freed, unmapped or remapped by anyone meanwhile.
    ///
    /// Reading or writing them through the pointer while the GPU writes them is a data race, as
    /// Metal's `contents` is.
    ///
    /// On Linux a buffer over a page this process maps without write permission is refused
    /// (`Unsupported`, `HostMemory`), though Metal reads such pages: Linux drivers pin imported
    /// pages for writing, and one (ANV) imported a read-only page and then lost the device at the
    /// GPU's first read. So on Linux the pages also stay writable while they are imported: RADV
    /// failed every command buffer submitted after a page of an import was made read-only
    /// (`mprotect`). With a dma-buf source ([`HostMemory::with_dma_buf`]) pages the process maps
    /// read-only are instead imported from the source alone, announced
    /// (`read-only-pages-written`). Some drivers also refuse shared memory (RADV refused anonymous
    /// shared memory and mappings of shared memory objects, `memfd` mappings among them); private
    /// memory imports everywhere measured. [`HostMemory::with_dma_buf`] imports such memory where
    /// the device imports dma-bufs.
    pub unsafe fn new(pointer: NonNull<u8>) -> HostMemory {
        HostMemory {
            pointer,
            source: None,
            writable: false,
            dirty: None,
            gate: None,
            runs: None,
        }
    }

    /// As [`HostMemory::new`], plus a source of dma-bufs of the pages for a device that refuses
    /// the pointer (a runtime extension, [`DmaBufSource`]). The runtime first imports the pointer
    /// exactly as [`HostMemory::new`]'s; only where the device refuses it (a driver refusal, or a
    /// range its import alignment cannot cover) and enables `VK_EXT_external_memory_dma_buf`, it
    /// imports the source's chunks instead: one chunk binds a plain buffer, several a sparse one.
    /// A texture over such pages (over the buffer, placed in the heap, or a view of one) is an
    /// image over a dma-buf of its own pages where the device lists
    /// `VK_EXT_image_drm_format_modifier` and `VK_EXT_queue_family_foreign`, else a shadow of the
    /// bytes (announced `shadowed-texture`). Refused as [`HostMemory::new`]'s memory is
    /// (`Unsupported`, `HostMemory`) where the device imports no chunk or the source fails; chunks
    /// that break the source's contract are misuse (`Invalid`, `Problem::DmaBuf`). Measured: RADV
    /// refuses the pointer of `memfd` and other shared mappings and imports `udmabuf`s of their
    /// pages; NVIDIA and lavapipe import the pointer and never call the source.
    ///
    /// A no-copy buffer's own import of the chunks (not a heap's or a rebind's) may be released
    /// while idle, because RADV lists every import in every submission. The release frees its
    /// `VkBuffer` and memory and keeps the chunks open: the runtime keeps a duplicate descriptor of
    /// each, within a quarter of the process's `RLIMIT_NOFILE` soft limit. The next blit imports
    /// them again, on the thread that encodes it, which costs time then. A buffer whose
    /// descriptors pass that quarter (read once, at the process's first such import) stays
    /// imported, in every kernel submission's list (`WorkCounts::host_imports_kept_for_fds` counts
    /// them). The runtime never changes the limit: a program that makes many no-copy buffers at
    /// once raises its soft limit toward the hard one at startup (`setrlimit`), as QEMU does. Only
    /// a buffer that blits alone have read is released. One whose `gpu_address` or `VkBuffer` was
    /// handed out, or with a texture over its bytes, stays imported. Nothing of Metal's observes
    /// the release: the bytes, the address and the pages stay, the CPU's writes made meanwhile are
    /// read, and the source is not called again. A blit that finds the device unable to import the
    /// chunks again (out of memory) is refused.
    ///
    /// # Safety
    ///
    /// [`HostMemory::new`]'s contract for the bytes at `pointer`, holding while the chunks are
    /// imported as while the pointer is; and the chunks `source` returns hold exactly the pages it
    /// is asked for, in order (then padding, [`DmaBufRequest`](crate::DmaBufRequest)), the same
    /// pieces of the same files for every request over the same pages, so that the GPU's bytes,
    /// the buffer's and its textures', are the CPU's.
    #[cfg(unix)]
    pub unsafe fn with_dma_buf(pointer: NonNull<u8>, source: DmaBufSource) -> HostMemory {
        HostMemory {
            pointer,
            source: Some(source),
            writable: false,
            dirty: None,
            gate: None,
            runs: None,
        }
    }

    /// This memory, its write permission vouched for by the caller: a runtime extension, for a
    /// caller that knows how its pages are mapped. On Linux the runtime then asks the kernel
    /// nothing of the pages before it imports their pointer, where it otherwise asks about every
    /// mapping over them (`PROCMAP_QUERY`, one call per mapping per import, each taking the
    /// process's memory-map lock; `WorkCounts::host_mappings_queried` counts none of them). Off
    /// Linux nothing is asked either way, and this changes nothing.
    ///
    /// # Safety
    ///
    /// As well as [`HostMemory::new`]'s contract (or [`HostMemory::with_dma_buf`]'s): every host
    /// page the buffer's bytes touch is mapped by this process with write permission when the
    /// buffer (or heap, or rebind) is made over them, and stays so while they are imported. The
    /// Vulkan import of a page the process cannot write is undefined on drivers that pin pages
    /// for writing (one lost its device at the GPU's first read of such a page), which the
    /// runtime's own check would have refused.
    pub unsafe fn assume_writable(self) -> HostMemory {
        HostMemory {
            writable: true,
            ..self
        }
    }

    /// This memory, with the caller's log of the writes to its pages (a runtime extension,
    /// [`DirtyPages`], whose offsets are bytes from [`pointer`](Self::pointer)). Every import made
    /// from this memory keeps the log, on every device, discrete and integrated alike: the
    /// buffer's or the heap's, and the pages a rebind gives either. A tracked 2D texture's shadow
    /// over the pages (`shadowed-texture`), whose texels a command buffer's first use copies
    /// straight from them into its image, then copies in only the full-width bands of rows of
    /// the pages the log reports changed since the command buffer whose end last left the image
    /// holding every texel, where no command of the device wrote the bytes or the image since,
    /// and nothing where no page changed (bands 8 rows apart or less are one; half the rows or
    /// more, or more than 1024 bands, are every row). The decision is made as the command
    /// buffer's commands are recorded for the GPU, after its commit and before its submission: a
    /// stamp asked at the commit ([`DirtyPages::request`], once per log and commit, only where the
    /// command buffer has such a copy in) and waited for then ([`DirtyPages::wait`], for the
    /// latest stamp asked of the log so far, so one wait serves every command buffer committed
    /// before it; none where the copy has no earlier copy to count on). A command buffer that
    /// waits for an event or a semaphore, or wrote the bytes before the copy, copies every row.
    /// The runtime reports its own writes into the pages back to the log: a shadow's copy back
    /// or merge ([`DirtyPages::wrote_as`], the shadow its writer, at the commit and again at the
    /// completion), which that shadow's own copies in leave out
    /// ([`DirtyPages::changed_except`]), and a `replaceRegion` of a texture over them
    /// ([`DirtyPages::wrote`]). A wait the log answers `false` makes the copy take every row, and
    /// is not made again for a stamp at or below it. Such a shadow's copy in is never cut to the rectangle a pass's scissors bound, so its
    /// image holds every texel for the next command buffer. Every other copy of the pages (a
    /// shadow copied through a staging buffer, of another kind, or untracked) takes every byte,
    /// as without a log. Counted:
    /// `WorkCounts::dirty_copies_in_gated` and kin.
    ///
    /// `source` is any [`DirtyPages`]: a concrete type, or a trait object that names its watch
    /// type (`Arc<dyn DirtyPages<Watch = W>>`). No Metal counterpart: Metal's GPU reads a no-copy
    /// object's pages in place.
    ///
    /// Exact where the log is true: every byte a skipped copy would have moved is one the copy
    /// holds already.
    pub fn with_dirty_pages<D>(self, source: Arc<D>) -> HostMemory
    where
        D: DirtyPages + ?Sized + 'static,
        D::Watch: 'static,
    {
        HostMemory {
            dirty: Some(hook(source)),
            ..self
        }
    }

    /// This memory, with the caller's gate over the import of its pages (a runtime extension,
    /// [`ImportGate`]): asked once, just before a no-copy buffer or heap made over it, or a
    /// buffer's rebind to it, imports the pages; where it answers `false`, the buffer is a copied
    /// host buffer (the heap's own buffer too, the rebind's new pages a copy), which no import maps,
    /// announced (`copied-host-pages`) and counted (`WorkCounts::host_imports_refused_by_host`,
    /// `WorkCounts::copied_host_buffers_refused`). No Metal counterpart.
    pub fn with_import_gate(self, gate: ImportGate) -> HostMemory {
        HostMemory {
            gate: Some(gate),
            ..self
        }
    }

    /// This memory, with the caller's resolver of its pages to the runs of guest RAM they are
    /// views of (a runtime extension, [`RunResolver`]). On a device that uses that RAM
    /// ([`Device::use_host_ram`](crate::Device::use_host_ram)), a no-copy buffer or heap whose
    /// pages the resolver answers as one run inside one of the device's chunks of the RAM is a
    /// `VkBuffer` bound in that chunk's import, at the run, sized and bound against the object's
    /// own pages as an import of them would be, so no access through the buffer reaches the rest
    /// of the chunk: no import of its own, no kernel mapping of the pages, and the import gate
    /// ([`HostMemory::with_import_gate`]) is not asked. Pages of more than one run, or of a run
    /// crossing a chunk's end, take the runtime's other paths (the gate, then an import or a
    /// copy), the runs kept with the object. Counted: `WorkCounts::chunk_bound_buffers_made`. No
    /// Metal counterpart.
    pub fn with_runs(self, resolver: RunResolver) -> HostMemory {
        HostMemory {
            runs: Some(resolver),
            ..self
        }
    }

    /// The pointer.
    pub fn pointer(&self) -> NonNull<u8> {
        self.pointer
    }

    /// The pointer, the dma-buf source where the caller gave one, whether the caller vouched
    /// for the pages' write permission ([`HostMemory::assume_writable`]), and its log of the pages
    /// written ([`HostMemory::with_dirty_pages`]), kept with the pointer its offsets count from,
    /// its gate over their import ([`HostMemory::with_import_gate`]), and its resolver of their
    /// runs in guest RAM ([`HostMemory::with_runs`]).
    pub(super) fn into_parts(self) -> HostParts {
        let dirty = (self.dirty).map(|hook| DirtySource::new(hook, self.pointer.as_ptr() as u64));
        HostParts {
            pointer: self.pointer,
            source: self.source,
            writable: self.writable,
            dirty,
            gate: self.gate,
            runs: self.runs,
        }
    }
}

/// A [`HostMemory`]'s parts ([`HostMemory::into_parts`]).
pub(super) struct HostParts {
    pub(super) pointer: NonNull<u8>,
    pub(super) source: Option<Source>,
    pub(super) writable: bool,
    pub(super) dirty: Option<Arc<DirtySource>>,
    pub(super) gate: Option<ImportGate>,
    pub(super) runs: Option<RunResolver>,
}

/// A no-copy buffer's deallocator (Metal's block): called once, with the pointer and the length,
/// when the buffer is released. It may run on a runtime thread, as Metal's runs on any thread: a
/// command buffer's queue's handler thread, where that command buffer's release lets go of the
/// buffer, or, where the release frees the import, the releasing thread before the release
/// returns, as Metal's; never a thread that submits command buffers or sees them complete. A panic
/// inside it is caught.
pub type Deallocator = Box<dyn FnOnce(NonNull<u8>, u64) + Send + 'static>;

/// What keeps the caller's host memory mapped (a runtime extension: Metal has none): given with a
/// no-copy buffer's or heap's memory, it is dropped once the device's import of the pages is
/// freed, which may be after the deallocator ran, or at the buffer's release, after its
/// deallocator, where the import's pages stay mapped without it: another buffer made over the
/// same bytes lies in the import, or the import keeps such a buffer's keep-alive already (the
/// module docs). With one, the deallocator runs when the buffer
/// is released, as Metal's does, though a command buffer submitted before a texture over the
/// pages was released may still hold the import; with none, the deallocator waits for the import
/// (the module docs). Dropped on a runtime thread, possibly; a panic inside its drop is caught.
/// Never dropped while the process exits: the pages stay imported to its end (`exit_guard`).
pub type KeepAlive = Box<dyn Send + 'static>;

/// What the caller gives back with its host memory: the deallocator, and the keep-alive, each
/// held as the program's object (`program_objects.rs`).
pub(super) struct HostRelease {
    deallocator: Option<ProgramObject<Deallocator>>,
    keep_alive: Option<ProgramObject<KeepAlive>>,
}

impl HostRelease {
    /// Hold the caller's `deallocator` and `keep_alive`, at the call that gives them.
    pub(super) fn new(deallocator: Option<Deallocator>, keep_alive: Option<KeepAlive>) -> Self {
        HostRelease {
            deallocator: deallocator.map(ProgramObject::new),
            keep_alive: keep_alive.map(ProgramObject::new),
        }
    }

    /// The release of `length` bytes at `pointer` to it, where it gives anything.
    pub(super) fn of(self, pointer: NonNull<u8>, length: u64) -> Option<Release> {
        let HostRelease {
            deallocator,
            keep_alive,
        } = self;
        (deallocator.is_some() || keep_alive.is_some()).then(|| Release {
            pointer,
            length,
            deallocator: Mutex::new(deallocator),
            keep_alive: Mutex::new(keep_alive),
            held_back: Mutex::new(None),
        })
    }
}

/// The host pages an import covers: `size` bytes from `start`, the buffer's bytes at `offset`
/// in them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ImportWindow {
    pub(super) start: u64,
    pub(super) size: u64,
    pub(super) offset: u64,
}

/// Why an import window or binding does not exist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WindowRefusal {
    /// The range runs past the end of the address space.
    Overflow,
    /// The device's import alignment exceeds the host page, and rounding the range out to it
    /// would take pages the range does not touch.
    BeyondPages,
    /// The buffer, bound at its memory alignment below its bytes ([`buffer_base`]), runs past
    /// the window, or its memory requirements do.
    Binding,
}

/// The import of `length` bytes at `pointer` on a host of `page`-byte pages and a device that
/// imports at multiples of `alignment`: the range rounded out to the larger of the two. Refused
/// when the rounding would take pages the range does not touch, so a window made is always the
/// pages the range touches, the one an `alignment` of 1 makes (a dma-buf's, which has none).
pub(super) fn import_window(
    pointer: u64,
    length: u64,
    page: u64,
    alignment: u64,
) -> Result<ImportWindow, WindowRefusal> {
    let unit = page.max(alignment).max(1);
    let end = pointer.checked_add(length).ok_or(WindowRefusal::Overflow)?;
    let round_up = |value: u64, to: u64| value.checked_next_multiple_of(to);
    let start = pointer - pointer % unit;
    let window_end = round_up(end, unit).ok_or(WindowRefusal::Overflow)?;
    let page_start = pointer - pointer % page.max(1);
    let page_end = round_up(end, page.max(1)).ok_or(WindowRefusal::Overflow)?;
    if start < page_start || window_end > page_end {
        return Err(WindowRefusal::BeyondPages);
    }
    Ok(ImportWindow {
        start,
        size: window_end - start,
        offset: pointer - start,
    })
}

/// Where the bytes of a no-copy buffer at `offset` in its window lie in its `VkBuffer`, whose
/// memory alignment is `alignment` (a power of two): `offset` modulo the alignment, or modulo 16
/// where that is larger. The `VkBuffer` is bound that many bytes below the bytes, at a multiple
/// of both in the window (and in the Metal address space, whose ranges start at pages), so its
/// bytes are congruent to their Metal addresses modulo 16, as the runtime's other buffers' are.
pub(super) fn buffer_base(offset: u64, alignment: u64) -> u64 {
    offset % alignment.max(16)
}

/// The size of the `VkBuffer` of a no-copy buffer of `length` bytes whose bytes start at `base`
/// in it: through the padded length, rounded up to 16 bytes, so that a range the binding layer pads
/// to 16 bytes from `base` lies inside it. `None` past `u64`.
pub(super) fn based_size(base: u64, length: u64) -> Option<u64> {
    length
        .checked_next_multiple_of(16)?
        .checked_add(base)?
        .checked_next_multiple_of(16)
}

/// Where a `VkBuffer` of `size` bytes with `requirements`, whose bytes start `base` bytes into
/// it, binds inside `window`: `base` bytes below the window's offset, which must be a multiple
/// of the memory alignment, with room in the window for the buffer and its requirements.
pub(super) fn binds_in(
    window: &ImportWindow,
    base: u64,
    size: u64,
    requirements: &vk::MemoryRequirements,
) -> Result<u64, WindowRefusal> {
    let at = window
        .offset
        .checked_sub(base)
        .ok_or(WindowRefusal::Binding)?;
    let aligned = at.is_multiple_of(requirements.alignment.max(1));
    let room = window.size - at;
    if !aligned || size > room || requirements.size > room {
        return Err(WindowRefusal::Binding);
    }
    Ok(at)
}

/// The host's page size, where the runtime knows how to ask for it.
#[cfg(unix)]
pub(super) fn host_page() -> Option<u64> {
    // SAFETY: `sysconf` reads a system constant.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    u64::try_from(page).ok().filter(|&page| page > 0)
}

/// The host's page size, where the runtime knows how to ask for it: not on this system.
#[cfg(not(unix))]
pub(super) fn host_page() -> Option<u64> {
    None
}

/// The refusal of host memory this device cannot import exactly.
pub(super) fn not_importable() -> Error {
    Error::Unsupported(Unsupported::not_representable(Feature::HostMemory))
}

/// Whether `error` is the refusal of host memory this device cannot import exactly
/// ([`not_importable`]): the window, the buffer's binding in it, the memory types or the driver
/// refused the pointer, which a dma-buf of its pages may serve ([`DmaBufSource`]).
pub(super) fn refused(error: &Error) -> bool {
    *error == not_importable()
}

/// The error of a failed import call (`vkGetMemoryHostPointerPropertiesEXT` or the allocation): a
/// driver that refuses the memory is a device gap, anything else a Vulkan failure. Refusing
/// drivers answer `VK_ERROR_INVALID_EXTERNAL_HANDLE`, but RADV of Mesa 26.0.8 answers
/// `VK_ERROR_UNKNOWN` for the shared memory 26.2.4 refuses with it, so that is a refusal too.
pub(super) fn import_refused(error: vk::Result) -> Error {
    match error {
        vk::Result::ERROR_INVALID_EXTERNAL_HANDLE | vk::Result::ERROR_UNKNOWN => not_importable(),
        other => vulkan(other),
    }
}

/// Refuse the window's pages, typed, unless the process maps every one of them writable: Linux
/// drivers pin imported pages for writing. Asked by each call that imports the pages itself, not
/// by a buffer that lies in a live import of them. The kernel answers for the mappings over the
/// window ([`queried_permitted`]) at each call, so a page remapped or reprotected since an earlier
/// import is judged as it is now: one query per mapping, so its cost grows with the mappings over
/// the window (a view of scattered pages maps each run apart). The queries go through the file
/// the process keeps open ([`kept_maps`]). A kernel without the query (before Linux 6.11) has the
/// lines of `/proc/self/maps` read up to the window instead ([`permitted`]), from a file opened for
/// the reading, which cost about 4.5 ms per import in a process of 34,000 mappings. Each mapping
/// asked is counted in `counts` (`WorkCounts::host_mappings_queried`). When the file cannot be
/// opened, every import is refused as not implemented here.
#[cfg(target_os = "linux")]
pub(super) fn refuse_unwritable(window: &ImportWindow, counts: &Counters) -> Result<(), Error> {
    let mut asked = 0;
    let answer = mappings_permitted(
        window.start,
        window.start + window.size,
        Access::Write,
        &mut asked,
    );
    counts.host_mappings_queried(asked);
    match answer {
        Some(true) => Ok(()),
        Some(false) => Err(not_importable()),
        None => Err(Error::Unsupported(Unsupported::not_implemented(
            Feature::HostMemory,
        ))),
    }
}

/// Whether the process maps every page of `[start, end)` with read permission, as the kernel
/// answers now ([`mappings_permitted`]): where it does, the CPU reads the bytes with plain loads,
/// as a copied host buffer's comparisons read them at each commit (`copied.rs`). `false` where
/// the mappings cannot be read (`/proc/self/maps` does not open).
#[cfg(target_os = "linux")]
pub(super) fn mapped_readable(start: u64, end: u64) -> bool {
    mappings_permitted(start, end, Access::Read, &mut 0) == Some(true)
}

/// [`mapped_readable`] where the runtime does not ask the system for its mappings: `false`.
#[cfg(not(target_os = "linux"))]
pub(super) fn mapped_readable(_start: u64, _end: u64) -> bool {
    false
}

/// A permission the process's mappings are asked for.
#[cfg(target_os = "linux")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Access {
    /// `VM_READ`, the `r` of a mapping's line in the maps: a copy reads the pages (`copied.rs`).
    Read,
    /// `VM_WRITE`, the `w`: Linux drivers pin imported pages for writing ([`refuse_unwritable`]).
    Write,
}

#[cfg(target_os = "linux")]
impl Access {
    /// The mapping's flag of the permission in `PROCMAP_QUERY`'s answer (`vma_flags`).
    fn queried(self) -> u64 {
        match self {
            Access::Read => PROCMAP_QUERY_VMA_READABLE,
            Access::Write => PROCMAP_QUERY_VMA_WRITABLE,
        }
    }

    /// The permission's place and letter in a maps line's permissions (`rw-p`).
    fn letter(self) -> (usize, u8) {
        match self {
            Access::Read => (0, b'r'),
            Access::Write => (1, b'w'),
        }
    }
}

/// Whether the process maps `[start, end)` without a gap, each page with `access`, as the kernel
/// answers now: for the mappings over it ([`queried_permitted`]), or from the lines of
/// `/proc/self/maps` where the kernel predates the query ([`permitted`]); `None` when that file
/// cannot be opened. Each mapping asked is added to `asked`.
#[cfg(target_os = "linux")]
fn mappings_permitted(start: u64, end: u64, access: Access, asked: &mut u64) -> Option<bool> {
    use std::io::BufRead;
    let maps = kept_maps()?;
    if let Some(permitted) = queried_permitted(&maps, start, end, access, asked) {
        return Some(permitted);
    }
    // The lines are read from a file of their own: reading moves its position, which the kept
    // one's queries share among threads.
    let maps = std::fs::File::open("/proc/self/maps").ok()?;
    let lines = std::io::BufReader::new(maps).lines().map_while(Result::ok);
    Some(permitted(lines, start, end, access, asked))
}

/// This process's `/proc/self/maps`, opened once for [`queried_permitted`] and kept with the id of
/// the process that opened it, or opened again (`None` where it cannot be): each `PROCMAP_QUERY`
/// answers from the process's mappings as they are at the call, so a file kept open reads the same
/// answers as one opened for the call, without the opening, which cost each import about as much
/// as its queries (an import over 57 mappings, the consumer's guest pages). A process forked
/// since holds the file of its parent's mappings, so one of another id opens its own.
#[cfg(target_os = "linux")]
pub(super) fn kept_maps() -> Option<std::sync::Arc<std::fs::File>> {
    static KEPT: Mutex<Option<(u32, std::sync::Arc<std::fs::File>)>> = Mutex::new(None);
    let process = std::process::id();
    let mut kept = lock(&KEPT);
    if let Some((_, maps)) = kept.as_ref().filter(|(opener, _)| *opener == process) {
        return Some(std::sync::Arc::clone(maps));
    }
    let maps = std::sync::Arc::new(std::fs::File::open("/proc/self/maps").ok()?);
    *kept = Some((process, std::sync::Arc::clone(&maps)));
    Some(maps)
}

/// `struct procmap_query` of Linux's `include/uapi/linux/fs.h` (Linux 6.11): the argument of
/// [`PROCMAP_QUERY`].
#[cfg(target_os = "linux")]
#[repr(C)]
#[derive(Default)]
struct ProcmapQuery {
    size: u64,
    query_flags: u64,
    query_addr: u64,
    vma_start: u64,
    vma_end: u64,
    vma_flags: u64,
    vma_page_size: u64,
    vma_offset: u64,
    inode: u64,
    dev_major: u32,
    dev_minor: u32,
    vma_name_size: u32,
    build_id_size: u32,
    vma_name_addr: u64,
    build_id_addr: u64,
}

#[cfg(target_os = "linux")]
const _: () = assert!(std::mem::size_of::<ProcmapQuery>() == 104);

/// `PROCMAP_QUERY`, `_IOWR('f', 17, struct procmap_query)` in the generic ioctl encoding (an
/// architecture of another encoding answers `ENOTTY`, and its imports take the lines).
#[cfg(target_os = "linux")]
const PROCMAP_QUERY: u64 =
    (3 << 30) | ((std::mem::size_of::<ProcmapQuery>() as u64) << 16) | ((b'f' as u64) << 8) | 17;

/// `PROCMAP_QUERY_VMA_READABLE`: the mapping's `VM_READ`, the `r` of its line in the maps.
#[cfg(target_os = "linux")]
const PROCMAP_QUERY_VMA_READABLE: u64 = 0x01;

/// `PROCMAP_QUERY_VMA_WRITABLE`: the mapping's `VM_WRITE`, the `w` of its line in the maps.
#[cfg(target_os = "linux")]
const PROCMAP_QUERY_VMA_WRITABLE: u64 = 0x02;

/// Whether the process maps `[start, end)` without a gap, each mapping with `access`, as the
/// kernel answers `PROCMAP_QUERY` on `maps` (this process's `/proc/self/maps`): one query per
/// mapping the window touches, each for the mapping covering the first address not yet covered,
/// and each added to `asked`. `None` where the kernel does not answer the query (before Linux
/// 6.11, `ENOTTY`) or fails it otherwise; a missing mapping (`ENOENT`) is an answer, not
/// permitted.
#[cfg(target_os = "linux")]
pub(super) fn queried_permitted(
    maps: &std::fs::File,
    start: u64,
    end: u64,
    access: Access,
    asked: &mut u64,
) -> Option<bool> {
    use std::os::fd::AsRawFd;
    let mut covered = start;
    while covered < end {
        *asked += 1;
        let mut query = ProcmapQuery {
            size: std::mem::size_of::<ProcmapQuery>() as u64,
            query_addr: covered,
            ..ProcmapQuery::default()
        };
        // SAFETY: `query` is a `struct procmap_query` of its stated size, which the kernel reads
        // and writes during the call; it asks for no name or build ID (sizes 0).
        let result = unsafe {
            libc::ioctl(
                maps.as_raw_fd(),
                PROCMAP_QUERY as _,
                std::ptr::from_mut(&mut query),
            )
        };
        if result != 0 {
            let error = std::io::Error::last_os_error().raw_os_error();
            return (error == Some(libc::ENOENT)).then_some(false);
        }
        // The mapping covers `covered`; an answer that does not is not taken.
        if query.vma_start > covered || query.vma_end <= covered {
            return None;
        }
        if query.vma_flags & access.queried() == 0 {
            return Some(false);
        }
        covered = query.vma_end;
    }
    Some(true)
}

/// Refuse the window's pages unless the process can write them: no check off Linux (MoltenVK
/// hands the range to Metal, which reads read-only pages).
#[cfg(not(target_os = "linux"))]
pub(super) fn refuse_unwritable(_window: &ImportWindow, _counts: &Counters) -> Result<(), Error> {
    Ok(())
}

/// Whether the mappings of `maps` (lines of `/proc/<pid>/maps`, in address order) cover
/// `[start, end)` without a gap, each with `access`, adding to `asked` each mapping it judged (one
/// ending past the address it reached). Reading stops at the first line past `end`; a line that
/// does not parse is skipped.
#[cfg(target_os = "linux")]
pub(super) fn permitted(
    maps: impl IntoIterator<Item = String>,
    start: u64,
    end: u64,
    access: Access,
    asked: &mut u64,
) -> bool {
    let mut covered = start;
    for line in maps {
        if covered >= end {
            break;
        }
        let Some((from, to, allowed)) = mapping(&line, access) else {
            continue;
        };
        if to <= covered {
            continue;
        }
        *asked += 1;
        if from > covered || !allowed {
            return false;
        }
        covered = to;
    }
    covered >= end
}

/// One line of `/proc/<pid>/maps`: the mapping's start, its end, and whether it has `access`.
#[cfg(target_os = "linux")]
fn mapping(line: &str, access: Access) -> Option<(u64, u64, bool)> {
    let mut fields = line.split_ascii_whitespace();
    let (from, to) = fields.next()?.split_once('-')?;
    let permissions = fields.next()?;
    let (place, letter) = access.letter();
    Some((
        u64::from_str_radix(from, 16).ok()?,
        u64::from_str_radix(to, 16).ok()?,
        permissions.as_bytes().get(place) == Some(&letter),
    ))
}

/// What a device knows to import host memory.
pub(super) struct Importer {
    functions: ash::ext::external_memory_host::DeviceFn,
    /// The host's page size.
    pub(super) page: u64,
    /// `minImportedHostPointerAlignment`.
    pub(super) alignment: u64,
    /// `vkGetMemoryFdPropertiesKHR`, where the device enables the dma-buf route (`dma_buf.rs`).
    dma_buf: Option<ash::khr::external_memory_fd::DeviceFn>,
    /// Whether the device also enables the route's image extensions (`create.rs`'s
    /// `dma_buf_image_names`), for images of textures over dma-buf pages (`dma_buf.rs`).
    dma_buf_images: bool,
    /// The granule a dma-buf source's chunks split at, where the device assembles a buffer of
    /// several (`dma_buf.rs`): the alignment of a sparse buffer made for dma-bufs. Asked once,
    /// the first time a pointer is refused and a source called; `None` where the device makes no
    /// such buffer.
    pub(super) sparse_granule: std::sync::OnceLock<Option<u64>>,
}

impl Importer {
    /// The importer of a device, or the refusal every no-copy buffer on it gets: the device did
    /// not enable `VK_EXT_external_memory_host`, the runtime does not know the host's page size,
    /// or the device cannot bind a buffer of `usage` to imported host memory.
    ///
    /// # Safety
    ///
    /// `device` is a live device of `physical`, made from `instance` with `enabled` enabled, and
    /// `loader_word` is its loader word; made with the dma-buf route's extensions where `dma_buf`,
    /// and its image extensions where `dma_buf_images` (`create.rs`).
    pub(super) unsafe fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        device: &ash::Device,
        enabled: &EnableSet,
        usage: vk::BufferUsageFlags,
        loader_word: &LoaderWord,
        dma_buf: bool,
        dma_buf_images: bool,
    ) -> Result<Importer, Unsupported> {
        if !enabled
            .extensions
            .contains(&VkExtension::ExtExternalMemoryHost)
        {
            return Err(Unsupported::needs(
                Feature::HostMemory,
                metal2vulkan::error::Capability::Extension(VkExtension::ExtExternalMemoryHost),
            ));
        }
        let page = host_page().ok_or_else(|| Unsupported::not_implemented(Feature::HostMemory))?;
        let mut host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut host);
        // SAFETY: the caller's contract; the structure belongs to an enabled extension.
        unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
        let alignment = host.min_imported_host_pointer_alignment;
        let info = vk::PhysicalDeviceExternalBufferInfo::default()
            .usage(usage)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let mut external = vk::ExternalBufferProperties::default();
        // SAFETY: the caller's contract; the query is core in Vulkan 1.1.
        unsafe {
            instance.get_physical_device_external_buffer_properties(physical, &info, &mut external)
        };
        let features = external.external_memory_properties.external_memory_features;
        if !features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
            || features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
        {
            return Err(Unsupported::not_representable(Feature::HostMemory));
        }
        let handle = device.handle();
        let load = |name: &std::ffi::CStr| {
            // SAFETY: the device is live, and the loader serves `vkGetDeviceProcAddr` after its
            // word is put back; the extension is enabled.
            let function = unsafe {
                loader_word.restore();
                instance.get_device_proc_addr(handle, name.as_ptr())
            };
            function.map_or(std::ptr::null(), |function| function as *const c_void)
        };
        let functions = ash::ext::external_memory_host::DeviceFn::load(load);
        let dma_buf = dma_buf.then(|| ash::khr::external_memory_fd::DeviceFn::load(load));
        let dma_buf_images = dma_buf.is_some() && dma_buf_images;
        Ok(Importer {
            functions,
            page,
            alignment,
            dma_buf,
            dma_buf_images,
            sparse_granule: std::sync::OnceLock::new(),
        })
    }

    /// Whether the device imports dma-bufs (`VK_EXT_external_memory_dma_buf`), which a no-copy
    /// buffer's pages may come as where it refuses their pointer ([`DmaBufSource`]).
    pub(super) fn imports_dma_bufs(&self) -> bool {
        self.dma_buf.is_some()
    }

    /// Whether the device makes images over dma-bufs of host pages (`dma_buf.rs`): it imports
    /// dma-bufs and enables `VK_EXT_image_drm_format_modifier` and `VK_EXT_queue_family_foreign`
    /// (`create.rs`'s `dma_buf_image_names`).
    pub(super) fn makes_dma_buf_images(&self) -> bool {
        self.dma_buf_images
    }

    /// The memory types the dma-buf `fd` can be imported into (`vkGetMemoryFdPropertiesKHR`);
    /// `ERROR_FEATURE_NOT_PRESENT` where the device imports no dma-bufs.
    ///
    /// # Safety
    ///
    /// `device` is the importer's device.
    #[cfg(unix)]
    pub(super) unsafe fn dma_buf_types(
        &self,
        device: &ash::Device,
        fd: std::os::fd::BorrowedFd<'_>,
    ) -> Result<u32, vk::Result> {
        use std::os::fd::AsRawFd;
        let functions = self
            .dma_buf
            .as_ref()
            .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
        let mut properties = vk::MemoryFdPropertiesKHR::default();
        // SAFETY: the caller's contract; the function was loaded for this device, and the
        // descriptor is open for the call (the query does not take it).
        let result = unsafe {
            (functions.get_memory_fd_properties_khr)(
                device.handle(),
                vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                fd.as_raw_fd(),
                &mut properties,
            )
        };
        result.result_with_success(properties.memory_type_bits)
    }

    /// A dma-buf of `memory`, an allocation of the importer's device made exportable as one
    /// (`vkGetMemoryFdKHR`): the device's check of textures over dma-buf pages makes its pages so
    /// (`in_place_check.rs`). `ERROR_FEATURE_NOT_PRESENT` where the device imports no dma-bufs.
    ///
    /// # Safety
    ///
    /// `device` is the importer's device, and `memory` a live allocation of it made with
    /// `VkExportMemoryAllocateInfo` naming dma-bufs.
    #[cfg(unix)]
    pub(super) unsafe fn exported_dma_buf(
        &self,
        device: &ash::Device,
        memory: vk::DeviceMemory,
    ) -> Result<std::os::fd::OwnedFd, vk::Result> {
        use std::os::fd::FromRawFd;
        let functions = self
            .dma_buf
            .as_ref()
            .ok_or(vk::Result::ERROR_FEATURE_NOT_PRESENT)?;
        let info = vk::MemoryGetFdInfoKHR::default()
            .memory(memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut fd = -1;
        // SAFETY: the caller's contract; the function was loaded for this device.
        let result = unsafe { (functions.get_memory_fd_khr)(device.handle(), &info, &mut fd) };
        result.result()?;
        // SAFETY: a new descriptor Vulkan handed over, which the caller owns from here on.
        Ok(unsafe { std::os::fd::OwnedFd::from_raw_fd(fd) })
    }

    /// The memory types `pointer` can be imported into (`vkGetMemoryHostPointerPropertiesEXT`).
    ///
    /// # Safety
    ///
    /// `device` is the importer's device; `pointer` is host memory aligned to its import
    /// alignment.
    pub(super) unsafe fn memory_types(
        &self,
        device: &ash::Device,
        pointer: NonNull<u8>,
    ) -> Result<u32, vk::Result> {
        let mut properties = vk::MemoryHostPointerPropertiesEXT::default();
        // SAFETY: the caller's contract; the function was loaded for this device.
        let result = unsafe {
            (self.functions.get_memory_host_pointer_properties_ext)(
                device.handle(),
                vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                pointer.as_ptr().cast_const().cast(),
                &mut properties,
            )
        };
        result.result_with_success(properties.memory_type_bits)
    }
}

/// What a no-copy buffer gives back when its last reference goes: the caller's range, to its
/// deallocator, and its keep-alive.
pub(super) struct Release {
    pointer: NonNull<u8>,
    length: u64,
    deallocator: Mutex<Option<ProgramObject<Deallocator>>>,
    keep_alive: Mutex<Option<ProgramObject<KeepAlive>>>,
    /// Where a detach held both back ([`Release::hold_back`]): told as the release would have run
    /// the deallocator.
    held_back: Mutex<Option<Arc<HeldBack>>>,
}

// SAFETY: the pointer is only handed to the deallocator, which is `Send`; the deallocator and
// the keep-alive are reached only through their locks.
unsafe impl Send for Release {}
// SAFETY: as above.
unsafe impl Sync for Release {}

impl Release {
    /// Never run the deallocator, nor drop it or the keep-alive: the memory stays imported, the
    /// process is exiting (`exit_guard`).
    pub(super) fn keep(&self) {
        if let Some(deallocator) = lock(&self.deallocator).take() {
            std::mem::forget(deallocator);
        }
        if let Some(keep_alive) = lock(&self.keep_alive).take() {
            std::mem::forget(keep_alive);
        }
    }

    /// Its keep-alive, for the import to keep until it is freed, or to drop at the release where
    /// the import's pages stay mapped without it (`buffer.rs`); `None` where the caller gave none.
    pub(super) fn keep_alive(&self) -> Option<ProgramObject<KeepAlive>> {
        lock(&self.keep_alive).take()
    }

    /// Whether its deallocator is yet to run.
    pub(super) fn deallocates(&self) -> bool {
        lock(&self.deallocator).is_some()
    }

    /// Whether it still holds the caller's keep-alive.
    pub(super) fn keeps_alive(&self) -> bool {
        lock(&self.keep_alive).is_some()
    }

    /// The length of the caller's range.
    pub(super) fn length(&self) -> u64 {
        self.length
    }

    /// Run the deallocator, once; a panic inside it is caught. Where a detach held it back, the
    /// [`HeldBack`] is told instead: the pages may be unmapped from now on.
    pub(super) fn run(&self) {
        if let Some(held) = lock(&self.held_back).take() {
            held.unmappable();
            return;
        }
        let deallocator = lock(&self.deallocator).take();
        if let Some(deallocator) = deallocator {
            let (pointer, length) = (self.pointer, self.length);
            exit_guard::program_release(move || {
                deallocator.run(|deallocator| deallocator(pointer, length));
            });
        }
    }

    /// Hold back its deallocator and its keep-alive for a detach (`detach.rs`): Metal runs a
    /// no-copy buffer's deallocator at its next replace, inside the call, or at its release, never
    /// at a detach (measured on the M3 Max, macOS 27.0.1). Both go to a [`HeldBack`], which this
    /// release tells, in the deallocator's place, once the pages may be unmapped (their import
    /// freed, or another buffer lying in it: [`HeldBack::unmappable`]); the keep-alive goes once
    /// the deallocator ran and the pages may be unmapped, so the deallocator may read the bytes, as
    /// with a release. `None` where it has no deallocator left to run.
    pub(super) fn hold_back(&self) -> Option<Arc<HeldBack>> {
        let deallocator = lock(&self.deallocator).take()?;
        let keep_alive = lock(&self.keep_alive).take();
        let held = Arc::new(HeldBack {
            state: Mutex::new(Back {
                deallocator: Some(deallocator),
                keep_alive,
                unmappable: false,
                due: false,
            }),
            pointer: self.pointer,
            length: self.length,
        });
        *lock(&self.held_back) = Some(Arc::clone(&held));
        Some(held)
    }
}

/// A no-copy buffer's deallocator and keep-alive a detach held back ([`Release::hold_back`]). The
/// deallocator runs once the buffer's next rebind or its release came ([`HeldBack::due`]) and the
/// pages are kept mapped or may be unmapped: at once where the caller gave a keep-alive, which
/// keeps them mapped, as Metal runs it then; else once the pages' release reached it
/// ([`HeldBack::unmappable`]), at once where nothing still holds the import. The keep-alive goes
/// once both the deallocator ran and the pages may be unmapped, on the thread whose call came last.
pub(super) struct HeldBack {
    state: Mutex<Back>,
    pointer: NonNull<u8>,
    length: u64,
}

/// What a [`HeldBack`] holds, and what came.
struct Back {
    deallocator: Option<ProgramObject<Deallocator>>,
    keep_alive: Option<ProgramObject<KeepAlive>>,
    unmappable: bool,
    due: bool,
}

// SAFETY: the pointer is only handed to the deallocator, which is `Send`; the deallocator and
// the keep-alive are reached only through their lock.
unsafe impl Send for HeldBack {}
// SAFETY: as above.
unsafe impl Sync for HeldBack {}

/// What a [`HeldBack`] runs and drops now, taken under its lock.
type Ready = (
    Option<ProgramObject<Deallocator>>,
    Option<ProgramObject<KeepAlive>>,
);

impl HeldBack {
    /// What runs and goes now: the deallocator once due and the pages are mapped (a keep-alive)
    /// or may be unmapped; the keep-alive once the deallocator ran and they may be unmapped.
    fn ready(back: &mut Back) -> Ready {
        let deallocator = match back.due && (back.unmappable || back.keep_alive.is_some()) {
            true => back.deallocator.take(),
            false => None,
        };
        let keep_alive = match back.unmappable && back.deallocator.is_none() {
            true => back.keep_alive.take(),
            false => None,
        };
        (deallocator, keep_alive)
    }

    /// Run the deallocator and drop the keep-alive of `ready`, in that order; the process
    /// exiting runs and drops neither: the pages stay imported to its end.
    fn run(&self, (deallocator, keep_alive): Ready) {
        let Some(busy) = exit_guard::enter() else {
            std::mem::forget(deallocator);
            std::mem::forget(keep_alive);
            return;
        };
        drop(busy);
        let (pointer, length) = (self.pointer, self.length);
        exit_guard::program_release(move || {
            if let Some(deallocator) = deallocator {
                deallocator.run(|deallocator| deallocator(pointer, length));
            }
            drop(keep_alive);
        });
    }

    /// The pages' release reached the deallocator ([`Release::run`]): they may be unmapped from
    /// now on.
    fn unmappable(&self) {
        let ready = {
            let mut back = lock(&self.state);
            back.unmappable = true;
            HeldBack::ready(&mut back)
        };
        self.run(ready);
    }

    /// The buffer's next rebind, or its release, came: the deallocator runs now, but where no
    /// keep-alive keeps the pages mapped and they are still imported; `true` where it waits so.
    pub(super) fn due(&self) -> bool {
        let (ready, waits) = {
            let mut back = lock(&self.state);
            back.due = true;
            let ready = HeldBack::ready(&mut back);
            (ready, back.deallocator.is_some())
        };
        self.run(ready);
        waits
    }
}

impl Drop for HeldBack {
    /// The process exiting leaves what it still holds unrun and undropped (`exit_guard`).
    fn drop(&mut self) {
        let back = self
            .state
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if exit_guard::enter().is_none() {
            std::mem::forget(back.deallocator.take());
            std::mem::forget(back.keep_alive.take());
        }
    }
}
