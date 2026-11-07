//! Guest RAM imported whole (a runtime extension: Metal has none): a VMM whose device refuses
//! views of guest pages made through dma-bufs gives the device the guest's RAM once
//! ([`HostRam`], [`Device::use_host_ram`]), and each no-copy object over a view of that RAM says
//! which runs of it its pages are (`HostMemory::with_runs`).
//!
//! - **The RAM** is the consumer's ranges of host memory, each mapped, readable and writable, for
//!   as long as any device uses it ([`HostRam::new`]). A device that uses it imports it by host
//!   pointer (`VK_EXT_external_memory_host`), or as dma-bufs (below), in chunks of [`CHUNK_BYTES`]
//!   (longer where each listed import costs every submission, below), never across a range's
//!   end, each once ([`Import`], a chunk's state behind a lock of its own, never held across an
//!   import). The chunks stay imported while the device uses the RAM (but there), and are freed
//!   once the use is let go of ([`HostRamUse`]'s drop) and nothing holds them: no buffer bound in
//!   them, no command buffer that used one. The RAM's keep-alive is dropped after its last chunk
//!   is freed.
//! - **As dma-bufs.** A RAM with a source of dma-bufs of its pages ([`HostRam::with_dma_buf`])
//!   has each import of it, a chunk or an object's run alone, made as one dma-buf of those pages
//!   where the device's held pointer imports add to the cost of every submission (its submission
//!   checks' import tax, `submit_check.rs`: RADV lists every import in each kernel submission and
//!   amdgpu gets every page of a pointer import again there), the pointer where that fails;
//!   elsewhere the pointer first, the dma-buf where the pointer is refused (RADV refuses any
//!   shared mapping). A dma-buf costs each submission what one import does, whatever its length.
//!   A `VkBuffer` bound in such a chunk is made for dma-bufs; a texture over it is an image over a
//!   dma-buf of its own pages from the same source (`dma_buf.rs`), asked at the RAM's addresses
//!   of them (`Placement::dma_buf_address`). Counted: `WorkCounts::host_ram_dma_buf_imports`.
//! - **Few listed imports** ([`Plan::listed`]). Where the device imports dma-bufs, the RAM has a
//!   source and the device's held imports add to the cost of every submission (RADV lists every
//!   import in each submission: measured with a probe on the Radeon 780M, Mesa 26.2.6, Linux
//!   7.3.1, an 8 GiB RAM held as 128 dma-bufs of 64 MiB raised a submission's median from 14 us
//!   to 60 to 88 us, as 3 of 2730 MiB to 16 us), a chunk is as long as one dma-buf of the
//!   source holds ([`HostRam::with_largest_dma_buf`]) and the device's
//!   `maxMemoryAllocationSize`, `maxBufferSize` and `maxStorageBufferRange` take, in whole
//!   [`CHUNK_BYTES`] ([`listed_chunk_bytes`]): an 8 GiB RAM is three imports where RADV's
//!   allocation limit is just under 4 GiB. There no thread imports ahead: a chunk is imported
//!   once an object over it asks, by a thread of the device's ([`listed_chunks`]), off the
//!   object's thread, as one long chunk's import pins all its pages (measured with a probe on the
//!   Radeon 780M and the RX 7900 XTX, Mesa 26.2.6: 660 to 692 ms for 2.7 GiB, which a making
//!   that waited for it took). Meanwhile each object over it is bound in an import of its own run
//!   alone, as below; those after it in the chunk's import. A buffer's run import is one more
//!   listed import, released while idle as a no-copy buffer's own import is (`host_residency.rs`),
//!   and the buffer is bound in the chunk at its next use; a heap's, or a buffer's whose address
//!   was handed out, stays while it lives. A chunk only its slot holds (no buffer bound in it, no
//!   pending command buffer that used one) is released once it stayed so for its window, as a
//!   no-copy buffer's idle import is (2 s, doubled at each release of the chunk up to 64 s), by
//!   the same thread; the next object over it asks for it again. Counted:
//!   `WorkCounts::host_ram_chunks_released`.
//! - **A window ahead of the objects.** From [`Device::use_host_ram`] on, a thread of the
//!   device's ([`import_ahead`]) keeps imported the chunk an object last took and a window of
//!   chunks past it, in the direction the takes go, up the RAM or down it, and waits while they
//!   are ([`Window`]): one chunk past a take that starts no run (and the RAM's first chunk before
//!   any take), [`AHEAD_CHUNKS`] once a take lands inside the window or next to the chunk last
//!   taken, as a run of objects over consecutive RAM goes on either way. Each import faults in
//!   and pins its pages, which is what an object over fresh RAM waited for at every chunk's first
//!   object (a 2 MiB heap in 32 over consecutive fresh RAM), so such a run finds its next chunks
//!   imported while it stays inside the window. The thread faults in no more than the window
//!   past the objects, and never goes back: the RAM the objects never reach stays as the guest
//!   left it, not resident nor pinned for the device. Counted:
//!   `WorkCounts::host_ram_chunks_imported_ahead`.
//! - **No object waits for a chunk.** An object over a chunk not imported yet (outside the
//!   window, or the one the thread is importing, [`Import::Making`]) is bound in an import of its
//!   own run alone, made on the thread making the object, which faults in and pins the run's
//!   pages and no more ([`RamOnDevice::for_object`]), while the device's thread imports the chunk,
//!   first of its window since that take (or as the object asked, [`Plan::listed`]); the objects
//!   over the chunk after that are bound in the chunk's import. A buffer's (not a heap's) run
//!   import is released while idle, its address never handed out, and the buffer bound in the
//!   chunk's import at its next use where that is made by then, else in its run's alone again
//!   (`host_residency.rs`). The run's import and the chunk's are two imports of the same pages, as an object's
//!   own import of a view of pages a chunk holds is; the object's uses order against those of
//!   every other object over its bytes, and of every copy of them through the chunk's import
//!   (`copy.rs`), by their host addresses, whichever import each is bound in
//!   (`host_aliases.rs`). The run's import is one more allocation, made only where the
//!   device's allocations leave the margin the thread leaves (below), and goes with its last
//!   holder. That is the trade for no making waiting: up to one more allocation and one more pin
//!   of its pages for each object made while its chunk is not imported, for as long as the
//!   object lives. A burst of heaps that outruns the thread takes one for each heap over a chunk
//!   the thread has not imported yet: 128 of the 256 2 MiB heaps of the upward order of
//!   `tests/host_ram_window_bench.rs` on an RTX 5080 laptop, every heap of its first burst of
//!   four chunks. Counted: `WorkCounts::run_bound_buffers_made` (each such object),
//!   `run_imports_live` (those live). Where the device's thread does not run (never started, or
//!   ended at the use's end or the allocation limit's margin) or failed the chunk, where the
//!   device's allocations are within that margin, where the device's barriers do not order blits
//!   through two imports of the same pages (`RuntimeDriverDefect::AliasedImportBlitsUnordered`,
//!   MoltenVK's: there every object over a chunk's pages lies in the chunk's one import, which
//!   Metal orders), where the run is the whole chunk, or where the run's import fails, the object
//!   imports the chunk on its own thread, or waits for the import of it under way.
//! - **Paced.** The thread runs as the system's batch work (never at a weight the system may
//!   starve: it holds a chunk's lock and, pinning, the process's memory map lock), yields between
//!   chunks, and on Linux faults each chunk's pages in 2 MiB at a time before its import, so that
//!   the import pins pages in place rather than holding the memory map lock while it faults in
//!   64 MiB ([`pace`], [`fault_in`]). Each import is made with the device's memory lock released
//!   (`memory::import_host`), so the device's other placements wait for none of it.
//! - **Short of the allocation limit.** The thread stops where one more import would leave the
//!   device less than a quarter of its `maxMemoryAllocationCount` (`allocations.rs`), the rest
//!   left to the objects over them. Once the device's allocations come within that quarter, the
//!   chunks it imported that no object took are given back
//!   ([`Device::give_back_ahead_near_the_limit`], `WorkCounts::host_ram_chunks_given_back`), so
//!   that they are never what an allocation is refused for.
//! - **Only here.** The thread is started by [`Device::use_host_ram`] alone, for the chunks of a
//!   RAM. A no-copy object's own imports, of its pointer or of the
//!   caller's dma-bufs (`dma_buf.rs`), never start one and import nothing ahead.
//! - **One run in place.** A no-copy object whose pages are one run inside one chunk is a
//!   `VkBuffer` bound in that chunk's import (`buffer_descriptor.rs`), made and bound as an import
//!   of the object's own pages would be: its size reaches no further than the object's last page,
//!   so neither its Metal address range, a descriptor's robust range nor a root entry reaches the
//!   rest of the chunk.
//! - **The dirty log** ([`HostRam::with_dirty_pages`]): one log for the whole RAM, its offsets
//!   host addresses, so one stamp per commit serves every copy over the RAM.
//! - **Copies by the GPU** between runs of the RAM and a buffer, from the chunks' imports
//!   (`BlitCommandEncoder::copy_from_host_ram`, `copy_to_host_ram`): `host_ram/copy.rs`.
//!
//! [`Device::use_host_ram`]: crate::Device::use_host_ram

use super::buffer::{usage, Pages};
use super::buffer_descriptor::HostBuffer;
use super::device::lock;
use super::dirty_pages::{hook, DirtyPages, DirtySource};
use super::error::{internal, invalid, vulkan, Error, MemoryKind, Problem};
use super::exit_guard;
use super::host_memory::{
    binds_in, host_page, import_refused, not_importable, ImportWindow, Importer, KeepAlive,
    RunResolver,
};
use super::host_residency;
#[cfg(unix)]
use super::host_residency::Again;
use super::memory::{ImportFailure, Mapped, MemoryClass, Placement};
use super::program_objects::ProgramObject;
use super::Device;
use crate::device::WeakDevice;
use crate::dma_buf::DmaBufSource;
use crate::release;
use ash::vk;
use metal2vulkan::error::{Feature, RuntimeDriverDefect, Unsupported};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, Weak};
use std::time::{Duration, Instant};

pub(crate) mod copy;

/// The bytes of a chunk of the RAM a device imports at once; the last chunk of a range may be
/// fewer.
pub(crate) const CHUNK_BYTES: u64 = 64 << 20;

/// The chunks past the one an object last took that the device's thread keeps imported once
/// the objects' takes run on into its window ([`Window`]): what a run of objects over
/// consecutive RAM goes on into without waiting for an import.
pub(crate) const AHEAD_CHUNKS: usize = 4;

/// A consumer's guest RAM (a runtime extension: Metal has none): ranges of this process's host
/// memory that no-copy objects' pages are views of, which a device imports whole, in chunks
/// ([`Device::use_host_ram`]), so that an object over one run of it is bound in a chunk with no
/// import of its own (`HostMemory::with_runs`). Made once per guest RAM and shared, as an
/// `Arc<HostRam>`, by every device that uses it.
pub struct HostRam {
    /// The ranges, as host addresses, by start, apart.
    ranges: Vec<Range<u64>>,
    /// The caller's log of the writes to the RAM, its offsets host addresses.
    dirty: Option<Arc<DirtySource>>,
    /// What keeps the ranges mapped, dropped once no device holds a chunk of them.
    keep_alive: Mutex<Option<ProgramObject<KeepAlive>>>,
    /// The caller's source of dma-bufs of the RAM's pages ([`HostRam::with_dma_buf`]).
    dma_buf: Option<DmaBufSource>,
    /// The most bytes one dma-buf of that source holds ([`HostRam::with_largest_dma_buf`]).
    largest_dma_buf: u64,
}

impl std::fmt::Debug for HostRam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostRam")
            .field("ranges", &self.ranges.len())
            .field("dirty_pages", &self.dirty.is_some())
            .field("dma_buf", &self.dma_buf.is_some())
            .field("largest_dma_buf", &self.largest_dma_buf)
            .finish()
    }
}

impl HostRam {
    /// The RAM of `ranges` (each a first byte and a length), kept mapped by `keep_alive`, which is
    /// dropped once no device holds a chunk of it any more (`None` for none).
    ///
    /// Refused: no range, or a range that is empty, starts or ends off the host page, runs past
    /// the address space's end, or overlaps another (`Invalid`, `Problem::HostRam`); a system
    /// whose page size the runtime cannot ask (`Unsupported`, `HostMemory`).
    ///
    /// # Safety
    ///
    /// Every range is mapped memory of this process, readable and writable, and stays so, at its
    /// address and with the same backing, until `keep_alive` is dropped (or, with none, until the
    /// `HostRam` is dropped and no device holds a chunk of it: every [`HostRamUse`] of it dropped
    /// and every buffer bound in a chunk released, and every command buffer that used one
    /// completed); nothing frees, unmaps, remaps or write-protects it meanwhile. Reading or
    /// writing its bytes while the GPU writes them is a data race, as Metal's `contents` is.
    pub unsafe fn new(
        ranges: &[(NonNull<u8>, u64)],
        keep_alive: Option<KeepAlive>,
    ) -> Result<HostRam, Error> {
        const CALL: &str = "HostRam::new";
        let page = host_page()
            .ok_or_else(|| Error::Unsupported(Unsupported::not_implemented(Feature::HostMemory)))?;
        let pairs = ranges
            .iter()
            .map(|(start, length)| (start.as_ptr() as u64, *length));
        let ranges = checked_ranges(pairs, page).ok_or_else(|| invalid(CALL, Problem::HostRam))?;
        Ok(HostRam {
            ranges,
            dirty: None,
            keep_alive: Mutex::new(keep_alive.map(ProgramObject::new)),
            dma_buf: None,
            largest_dma_buf: CHUNK_BYTES,
        })
    }

    /// This RAM, with the caller's log of the writes to its pages (a runtime extension,
    /// [`DirtyPages`]), whose offsets are **host addresses** (the log's pointer is address 0): one
    /// log for every range, so that one stamp per commit serves every copy the runtime keeps of
    /// pages of the RAM. Objects off the RAM keep their own logs (`HostMemory::with_dirty_pages`).
    /// No Metal counterpart.
    pub fn with_dirty_pages<D>(mut self, source: Arc<D>) -> HostRam
    where
        D: DirtyPages + ?Sized + 'static,
        D::Watch: 'static,
    {
        self.dirty = Some(DirtySource::new(hook(source), 0));
        self
    }

    /// This RAM, with the caller's source of dma-bufs of its pages (a runtime extension,
    /// [`DmaBufSource`]): a device that imports dma-bufs imports each chunk of the RAM as one
    /// dma-buf of its pages where its held pointer imports add to the cost of every submission
    /// (the device's submission checks' import tax, `submit_check.rs`), and, where they do not,
    /// where it refuses the chunk's pointer (RADV refuses any shared mapping, a `memfd`'s among
    /// them); elsewhere it imports the pointer as without one. A chunk imported as a dma-buf costs
    /// each submission what one import does, whatever its length, where a pointer import's cost
    /// grows with its pages (measured with a probe on the Radeon 780M, Mesa 26.2.6, Linux 7.3.1:
    /// one held import of 64 MiB by pointer added 24.1 us to every submission). Each request is
    /// for whole host pages inside one chunk of the RAM (a chunk: 64 MiB, or as long as
    /// [`HostRam::with_largest_dma_buf`] allows where the device's held imports add to the cost of
    /// every submission, or the rest of its range; an object's run inside one; a texture's pages
    /// inside one), its `granule` its `length`: the source answers one dma-buf of exactly those
    /// pages. The source is called on
    /// the thread importing the chunk (the device's, or one making an object over it), with no
    /// lock of the runtime's held, and is kept with each import it made until that import is
    /// freed. A source that fails, or answers otherwise, leaves the chunk to its pointer. No Metal
    /// counterpart.
    pub fn with_dma_buf(mut self, source: DmaBufSource) -> HostRam {
        self.dma_buf = Some(source);
        self
    }

    /// This RAM, whose source of dma-bufs ([`HostRam::with_dma_buf`]) makes one dma-buf of up to
    /// `bytes` bytes (a runtime extension): on Linux, udmabuf's `size_limit_mb` (its module
    /// parameter, 64 MiB by default). A device whose held imports add to the cost of every
    /// submission imports the RAM in chunks as long as that, as its own limits allow
    /// ([`Device::use_host_ram`]), so that each submission lists a few imports of the RAM, not
    /// one per 64 MiB. Without this call, 64 MiB, the udmabuf default. No Metal counterpart.
    pub fn with_largest_dma_buf(mut self, bytes: u64) -> HostRam {
        self.largest_dma_buf = bytes;
        self
    }

    /// Whether any range of this RAM meets any of `other`'s.
    fn meets(&self, other: &HostRam) -> bool {
        (self.ranges.iter())
            .any(|a| (other.ranges.iter()).any(|b| a.start < b.end && b.start < a.end))
    }
}

impl Drop for HostRam {
    fn drop(&mut self) {
        let keep_alive = (self.keep_alive.get_mut())
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        // The process exiting leaves the ranges imported to its end (`exit_guard`).
        if exit_guard::enter().is_none() {
            std::mem::forget(keep_alive);
            return;
        }
        if let Some(keep_alive) = keep_alive {
            exit_guard::program_release(move || drop(keep_alive));
        }
    }
}

/// `pairs` (first bytes and lengths) as ranges by start, where each is non-empty, whole `page`s,
/// within the address space, and apart from the others; `None` otherwise, or for no range.
fn checked_ranges(
    pairs: impl IntoIterator<Item = (u64, u64)>,
    page: u64,
) -> Option<Vec<Range<u64>>> {
    let mut ranges = Vec::new();
    for (start, length) in pairs {
        let end = start.checked_add(length)?;
        let whole = start.is_multiple_of(page) && length.is_multiple_of(page);
        if length == 0 || !whole {
            return None;
        }
        ranges.push(start..end);
    }
    ranges.sort_by_key(|range| range.start);
    let apart = ranges.windows(2).all(|pair| pair[0].end <= pair[1].start);
    (apart && !ranges.is_empty()).then_some(ranges)
}

/// A device's use of a [`HostRam`] ([`Device::use_host_ram`]): while it lives, the device imports
/// the RAM's chunks as objects over them are made, and keeps them. Dropping it lets go of the
/// use: each chunk is freed once nothing holds it (no buffer bound in it, no command buffer that
/// used one), and the RAM's keep-alive goes after the last. It holds the device. No Metal
/// counterpart.
#[must_use = "dropping the use lets go of the RAM at once"]
pub struct HostRamUse {
    device: Device,
    ram: Arc<RamOnDevice>,
}

impl std::fmt::Debug for HostRamUse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostRamUse")
            .field("ram", &self.ram.ram)
            .finish()
    }
}

impl Drop for HostRamUse {
    fn drop(&mut self) {
        self.ram.copies.unplace(&self.device);
        let gone = self.device.shared().host_ram.remove(&self.ram);
        // The device's thread sees the use ended, and ends.
        self.ram.wake.notify();
        // The chunks no object holds are freed here, with no lock of the source held.
        drop(gone);
    }
}

/// A RAM as one device uses it: the RAM, and its chunks, each imported once, by the device's
/// thread ahead of the objects over it ([`import_ahead`]) or by the first object over it.
pub(crate) struct RamOnDevice {
    ram: Arc<HostRam>,
    /// How the device imports its chunks.
    plan: Plan,
    /// Its chunks, by start ([`chunk_spans`]).
    slots: Vec<ChunkSlot>,
    /// The chunks the device's thread keeps imported ([`RamOnDevice::next_ahead`]).
    window: Mutex<Window>,
    /// What wakes the device's thread once the window moves or the use ends.
    wake: Arc<Wake>,
    /// Its chunks' keys for the copies between it and buffers (`host_ram/copy.rs`).
    copies: copy::ChunkKeys,
}

/// The chunks of a RAM the device's thread keeps imported for the objects ([`import_ahead`]): the
/// chunk an object last took, `taken`, first (an object over it while it is not imported binds
/// in an import of its own run, [`RamOnDevice::for_object`]), then `chunks` past it in the
/// direction the takes go, up the RAM or down it (`down`). A take inside the window, or next to
/// the chunk last taken (a run of objects going on, either way), opens the next to
/// [`AHEAD_CHUNKS`] in the take's direction; a take elsewhere (a jump) gives one chunk past it,
/// in the jump's direction, so that one take never has the thread fault in more than
/// [`AHEAD_CHUNKS`] chunks no object took, and a take that starts no run more than one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Window {
    taken: Option<usize>,
    down: bool,
    chunks: usize,
}

impl Window {
    /// Before any take: one chunk from the RAM's start.
    const START: Window = Window {
        taken: None,
        down: false,
        chunks: 1,
    };

    /// The window's slots, in the order the device's thread imports them: the chunk last taken,
    /// then those past it, the nearest first; before any take, those from the RAM's start.
    fn slots(self) -> impl Iterator<Item = usize> {
        let past = (1..=self.chunks).filter_map(move |step| match self.taken {
            None => Some(step - 1),
            Some(taken) if self.down => taken.checked_sub(step),
            Some(taken) => taken.checked_add(step),
        });
        self.taken.into_iter().chain(past)
    }

    /// The window once an object took chunk `slot`, another than the last taken: the
    /// [`AHEAD_CHUNKS`] past it, away from the chunk last taken, where it lay inside this window
    /// or next to that chunk, else the one past it.
    fn after_take(self, slot: usize) -> Window {
        let next_to = self.taken.is_some_and(|taken| taken.abs_diff(slot) == 1);
        let goes_on = next_to || self.slots().any(|inside| inside == slot);
        Window {
            taken: Some(slot),
            down: self.taken.is_some_and(|taken| slot < taken),
            chunks: if goes_on { AHEAD_CHUNKS } else { 1 },
        }
    }
}

/// What wakes the device's thread ([`import_ahead`]) while its window is imported: an object
/// that took another chunk, or the use's end. Shared with the thread, which holds nothing else
/// of the device's or the RAM's while it waits.
#[derive(Default)]
struct Wake {
    pending: Mutex<bool>,
    condvar: Condvar,
    /// Whether the thread runs: set as it is started, cleared as it ends ([`Ends`]).
    running: AtomicBool,
}

/// Clears [`Wake::running`] as the device's thread ends, however it ends.
struct Ends(Arc<Wake>);

impl Drop for Ends {
    fn drop(&mut self) {
        self.0.running.store(false, Ordering::Relaxed);
    }
}

impl Wake {
    /// Wake the thread, or have its next wait return at once.
    fn notify(&self) {
        *lock(&self.pending) = true;
        self.condvar.notify_one();
    }

    /// Wait until the next [`Wake::notify`], or return at once for one since the last wait.
    fn wait(&self) {
        let mut pending = lock(&self.pending);
        while !*pending {
            pending = (self.condvar.wait(pending)).unwrap_or_else(PoisonError::into_inner);
        }
        *pending = false;
    }

    /// As [`Wake::wait`], for at most `timeout`.
    fn wait_for(&self, timeout: Duration) {
        let pending = lock(&self.pending);
        let (mut pending, _) = (self.condvar)
            .wait_timeout_while(pending, timeout, |pending| !*pending)
            .unwrap_or_else(PoisonError::into_inner);
        *pending = false;
    }
}

/// One chunk of a RAM a device uses: its host bytes, and its import ([`Import`]) behind a lock of
/// its own, so the chunk is imported once and objects over other chunks wait for none of it.
struct ChunkSlot {
    span: Range<u64>,
    /// Held only to read or change the state, never across an import: a look at the chunk (the
    /// device's thread's [`RamOnDevice::next_ahead`], another object's) is never taken for an
    /// import under way.
    import: Mutex<Import>,
    /// Told as an import under way ends, made or failed.
    made: Condvar,
    /// The device's thread failed to import the chunk, or gave its import back: it does not try
    /// again, and the first object over it imports it, as it would have.
    tried: AtomicBool,
    /// An object over the chunk found it not imported and was bound in its own run's import, on
    /// a device whose plan is [`Plan::listed`]: the device's thread imports it next
    /// ([`RamOnDevice::next_asked`]). Changed inside [`ChunkSlot::import`]'s lock.
    asked: AtomicBool,
    /// The device's thread imported the chunk and no object took it since: given back once the
    /// device's allocations come near its limit ([`Device::give_back_ahead_near_the_limit`]).
    untaken: AtomicBool,
    /// Since when only the slot holds the chunk, and how long it may stay so before it is
    /// released, on a device that releases them ([`Plan::listed`], [`RamOnDevice::idle`]).
    /// Taken inside [`ChunkSlot::import`]'s lock.
    idle: Mutex<Idle>,
}

/// A chunk's idleness ([`ChunkSlot::idle`]).
struct Idle {
    /// Since when only its slot holds it; `None` while anything else does, or it is not imported.
    since: Option<Instant>,
    /// How long it may stay so: `host_residency::IDLE` at first, doubled at each release, up to
    /// `host_residency::IDLE_MOST`, as a no-copy buffer's idle import's window is.
    window: Duration,
}

/// Where a chunk's import stands ([`ChunkSlot::import`]).
enum Import {
    /// Not imported: the next to import it makes it.
    None,
    /// Being imported, by the device's thread or an object, with no lock held: an object that
    /// may be bound in its own run's import is ([`RamOnDevice::for_object`]); one that may not
    /// waits for it ([`ChunkSlot::made`]).
    Making,
    /// Imported, kept while the device uses the RAM or until given back.
    Made(Arc<RamChunk>),
}

/// An import of a chunk under way ([`Import::Making`]): where it ends without its answer (a
/// panic), the chunk is left not imported, and those waiting for it told.
struct Making<'a> {
    slot: &'a ChunkSlot,
    ended: bool,
}

impl<'a> Making<'a> {
    /// The import's answer into the slot, and those waiting for it told, under its lock, which
    /// the caller holds after: `made`, or not imported where it failed.
    fn end(mut self, made: Option<Arc<RamChunk>>) -> MutexGuard<'a, Import> {
        self.ended = true;
        let slot = self.slot;
        let mut import = lock(&slot.import);
        *import = made.map_or(Import::None, Import::Made);
        slot.made.notify_all();
        import
    }
}

impl Drop for Making<'_> {
    fn drop(&mut self) {
        if !self.ended {
            *lock(&self.slot.import) = Import::None;
            self.slot.made.notify_all();
        }
    }
}

/// Who imports a chunk: the first object over it, on the thread making it; the device's
/// thread, ahead of every object over it ([`import_ahead`]); or the device's thread, for an
/// object over it that was bound in its own run's import meanwhile, on a device whose plan is
/// [`Plan::listed`] ([`listed_chunks`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum By {
    Object,
    Ahead,
    Asked,
}

/// What an import of a RAM's pages ([`RamChunk`]) is: one of its chunks, kept in its slot while
/// the device uses the RAM, or the run of one object over a chunk not imported yet
/// ([`RamOnDevice::for_object`]), held by that object's placement alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Imported {
    Chunk,
    Run,
}

/// How a device imports a RAM's chunks ([`Device::use_host_ram`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    /// Each chunk's bytes; the last of a range may be fewer.
    pub(crate) bytes: u64,
    /// The device's held imports add to the cost of every submission and the RAM's chunks are
    /// its dma-bufs (the module docs): its chunks are as long as [`listed_chunk_bytes`] allows,
    /// none is imported ahead, each is imported by the device's thread once an object over it
    /// asked, and one no object holds is released while idle.
    pub(crate) listed: bool,
}

impl Plan {
    /// Chunks of [`CHUNK_BYTES`], imported ahead of the objects and kept.
    pub(crate) const AHEAD: Plan = Plan {
        bytes: CHUNK_BYTES,
        listed: false,
    };
}

/// The bytes of each chunk of a RAM on a device whose held imports add to the cost of every
/// submission ([`Plan::listed`]): as many whole [`CHUNK_BYTES`] as one dma-buf of the RAM's source
/// holds (`largest_dma_buf`) and the device's `maxMemoryAllocationSize`, `maxBufferSize` (a chunk
/// is one `VkBuffer`; 0 where the device states none) and `maxStorageBufferRange` (a copy binds a
/// chunk's whole `VkBuffer` as a storage buffer, `host_ram/copy.rs`) all take; [`CHUNK_BYTES`]
/// where any is shorter.
pub(crate) fn listed_chunk_bytes(
    largest_dma_buf: u64,
    allocation: u64,
    buffer: u64,
    storage: u64,
) -> u64 {
    let buffer = if buffer == 0 { u64::MAX } else { buffer };
    let most = largest_dma_buf.min(allocation).min(buffer).min(storage);
    (most / CHUNK_BYTES).max(1) * CHUNK_BYTES
}

/// The chunks of `ranges` (host addresses, by start): `bytes` from each range's start, the last
/// of each range ending at its end, so that no chunk crosses a range's end.
fn chunk_spans(ranges: &[Range<u64>], bytes: u64) -> Vec<Range<u64>> {
    let mut spans = Vec::new();
    for range in ranges {
        let mut start = range.start;
        while start < range.end {
            let end = range.end.min(start.saturating_add(bytes));
            spans.push(start..end);
            start = end;
        }
    }
    spans
}

/// The index of the span of `spans` (by start, apart) that holds the host address `address`.
fn span_of(spans: &[Range<u64>], address: u64) -> Option<usize> {
    let after = spans.partition_point(|span| span.start <= address);
    let at = after.checked_sub(1)?;
    spans[at].contains(&address).then_some(at)
}

impl RamOnDevice {
    /// The RAM's log of the writes to its pages (`HostRam::with_dirty_pages`), where it came with
    /// one: its offsets are host addresses.
    pub(crate) fn dirty(&self) -> Option<&Arc<DirtySource>> {
        self.ram.dirty.as_ref()
    }

    /// `ram`, imported as `plan` says, none of its chunks imported.
    fn new(ram: &Arc<HostRam>, plan: Plan) -> RamOnDevice {
        let slots: Vec<ChunkSlot> = (chunk_spans(&ram.ranges, plan.bytes).into_iter())
            .map(|span| ChunkSlot {
                span,
                import: Mutex::new(Import::None),
                made: Condvar::new(),
                tried: AtomicBool::new(false),
                asked: AtomicBool::new(false),
                untaken: AtomicBool::new(false),
                idle: Mutex::new(Idle {
                    since: None,
                    window: host_residency::IDLE,
                }),
            })
            .collect();
        RamOnDevice {
            ram: Arc::clone(ram),
            plan,
            copies: copy::ChunkKeys::new(slots.len()),
            slots,
            window: Mutex::new(Window::START),
            wake: Arc::default(),
        }
    }

    /// The chunks' spans.
    fn spans(&self) -> Vec<Range<u64>> {
        self.slots.iter().map(|slot| slot.span.clone()).collect()
    }

    /// The import of chunk `slot` on `device`, for `by`: kept, or made now, once
    /// (`Device::import_chunk`), with the chunk's lock released and its state
    /// [`Import::Making`], so an object over the chunk the device's thread (or another object)
    /// is importing waits for that import. A failed import is not kept: the next object over the
    /// chunk tries again. An object's call moves the device's thread's window to this chunk
    /// ([`RamOnDevice::take`]), and wakes the thread where that moved it, on a device that
    /// imports ahead (not [`Plan::listed`]). An import ahead is counted once the chunk is in its
    /// slot, so that the count of imports ahead never runs before the chunks an object finds
    /// there.
    fn chunk(
        &self,
        device: &Device,
        importer: &Importer,
        slot: usize,
        by: By,
    ) -> Result<Arc<RamChunk>, Error> {
        if by == By::Object && !self.plan.listed && self.take(slot) {
            self.wake.notify();
        }
        let span = &self.slots[slot];
        let mut import = lock(&span.import);
        if by == By::Object {
            span.untaken.store(false, Ordering::Relaxed);
        }
        loop {
            match &*import {
                Import::Made(chunk) => return Ok(Arc::clone(chunk)),
                Import::Making => {
                    import = (span.made.wait(import)).unwrap_or_else(PoisonError::into_inner);
                }
                Import::None => break,
            }
        }
        *import = Import::Making;
        drop(import);
        let making = Making {
            slot: span,
            ended: false,
        };
        // SAFETY: the span is pages of the RAM's ranges, which stay mapped, readable and
        // writable, while a chunk holds the RAM (`HostRam::new`).
        let made =
            unsafe { device.import_chunk(importer, &self.ram, span.span.clone(), Imported::Chunk) };
        let chunk = match made {
            Ok(chunk) => Arc::new(chunk),
            Err(error) => {
                drop(making.end(None));
                return Err(error);
            }
        };
        let import = making.end(Some(Arc::clone(&chunk)));
        if by == By::Ahead {
            let shared = device.shared();
            span.untaken.store(true, Ordering::Relaxed);
            shared.allocations.allocations.made_ahead();
            shared.counts.host_ram_chunk_imported_ahead();
        }
        drop(import);
        Ok(chunk)
    }

    /// The import an object over `run`, host pages inside chunk `slot`, is bound in
    /// ([`Device::bind_in_chunk`]), never waiting for the chunk's: the chunk's import where it is
    /// made; else, while the device's thread runs and has not failed the chunk, an import of the
    /// run alone ([`RamOnDevice::run_alone`]), made now on the object's thread, which faults in
    /// and pins the run's pages and no more, while the thread imports the chunk: first of its
    /// window since this take ([`Window`]), or, on a device whose plan is [`Plan::listed`], as
    /// the object asked it to ([`listed_chunks`]). The run's import is one more allocation, made
    /// only where one more leaves the device the margin of its `maxMemoryAllocationCount` the
    /// thread's imports leave (`Allocations::room_ahead`), and freed as its last holder goes, the
    /// object or a command buffer that used it (`WorkCounts::run_imports_live`); a buffer's is
    /// released while idle, and the buffer bound in the chunk at its next use where the chunk is
    /// imported by then (`host_residency.rs`). Where the run is the whole chunk, the thread does
    /// not run (never started, or ended: the use's end, the allocation limit's margin) or failed
    /// the chunk, the device's allocations are within that margin, the device's barriers do not
    /// order blits through two imports of the same pages
    /// (`RuntimeDriverDefect::AliasedImportBlitsUnordered`: there every object over a chunk's
    /// pages lies in its one import), or the run's import fails, the chunk's import, made here or
    /// waited for ([`RamOnDevice::chunk`]).
    fn for_object(
        &self,
        device: &Device,
        importer: &Importer,
        slot: usize,
        run: Range<u64>,
    ) -> Result<Arc<RamChunk>, Error> {
        let span = &self.slots[slot];
        let unordered = RuntimeDriverDefect::AliasedImportBlitsUnordered;
        let alone = run != span.span
            && self.wake.running.load(Ordering::Relaxed)
            && !span.tried.load(Ordering::Relaxed)
            && device.shared().allocations.allocations.room_ahead()
            && !device.properties().driver_defects.contains(&unordered);
        if !alone {
            return self.chunk(device, importer, slot, By::Object);
        }
        (self.run_alone(device, importer, slot, run))
            .or_else(|_| self.chunk(device, importer, slot, By::Object))
    }

    /// The imports a copy over chunk `slot` reads or writes its bytes `ranges` (offsets in the
    /// chunk, by start, apart) through (`host_ram/copy.rs`), never waiting for the chunk's, as an
    /// object over it is bound ([`RamOnDevice::for_object`]), the device's thread's window moved
    /// to the chunk as an object's take moves it: the chunk's import where it is
    /// made; else, where an object's run would be imported alone and the ranges are at most
    /// `copy::MOST_ALONE`, an import of each range alone, widened to whole host pages within the
    /// chunk ([`RamOnDevice::run_alone`]: the device's thread asked for the chunk), held by the
    /// copy's command buffer until it completed; else the chunk's, made here or waited for
    /// ([`RamOnDevice::chunk`]). Each import's span lies in the
    /// chunk's and holds every range it is given for.
    pub(crate) fn for_copy(
        &self,
        device: &Device,
        importer: &Importer,
        slot: usize,
        ranges: &[Range<u64>],
    ) -> Result<Vec<Arc<RamChunk>>, Error> {
        let span = &self.slots[slot];
        if !self.plan.listed && self.take(slot) {
            self.wake.notify();
        }
        if let Some(chunk) = self.made(slot) {
            return Ok(vec![chunk]);
        }
        let page = importer.page.max(1);
        let unordered = RuntimeDriverDefect::AliasedImportBlitsUnordered;
        let alone = self.wake.running.load(Ordering::Relaxed)
            && !span.tried.load(Ordering::Relaxed)
            && device.shared().allocations.allocations.room_ahead()
            && !device.properties().driver_defects.contains(&unordered);
        let runs: Vec<Range<u64>> = (ranges.iter())
            .map(|range| {
                let start = span.span.start + range.start / page * page;
                let end = span.span.start + range.end.div_ceil(page) * page;
                start..end.min(span.span.end)
            })
            .collect();
        let few = runs.len() <= copy::MOST_ALONE;
        if !alone || !few || runs.contains(&span.span) {
            return Ok(vec![self.chunk(device, importer, slot, By::Object)?]);
        }
        let mut imports: Vec<Arc<RamChunk>> = Vec::with_capacity(runs.len());
        for run in runs {
            if imports
                .iter()
                .any(|import| import.span.start <= run.start && run.end <= import.span.end)
            {
                continue;
            }
            match self.run_alone(device, importer, slot, run) {
                Ok(import) if !import.is_run() => return Ok(vec![import]),
                Ok(import) => imports.push(import),
                Err(_) => return Ok(vec![self.chunk(device, importer, slot, By::Object)?]),
            }
        }
        Ok(imports)
    }

    /// Chunk `slot`'s import where it is made; else an import of `run` (host pages inside it)
    /// alone ([`Imported::Run`]), made now on this thread, with the device's thread asked for the
    /// chunk: its window moved to the chunk ([`RamOnDevice::take`]), or, on a device whose plan
    /// is [`Plan::listed`], the chunk asked for ([`ChunkSlot::asked`]); woken either way. Never
    /// the chunk's own import, nor a wait for one under way. Refused as the run's import is.
    fn run_alone(
        &self,
        device: &Device,
        importer: &Importer,
        slot: usize,
        run: Range<u64>,
    ) -> Result<Arc<RamChunk>, Error> {
        let span = &self.slots[slot];
        let listed = self.plan.listed;
        if !listed && self.take(slot) {
            self.wake.notify();
        }
        // The chunk's import where it is made; none where it is not, or is being made (by the
        // thread, or an object): the run's own import then.
        let (found, ask) = match &*lock(&span.import) {
            Import::Made(chunk) => {
                span.untaken.store(false, Ordering::Relaxed);
                (Some(Arc::clone(chunk)), false)
            }
            Import::None => (None, listed && !span.asked.swap(true, Ordering::Relaxed)),
            Import::Making => (None, false),
        };
        if let Some(chunk) = found {
            return Ok(chunk);
        }
        if ask {
            self.wake.notify();
        }
        fault_in(&run);
        // SAFETY: the run is pages of the RAM's ranges inside the chunk, which stay mapped,
        // readable and writable, while an import holds the RAM (`HostRam::new`).
        let alone = unsafe { device.import_chunk(importer, &self.ram, run, Imported::Run) };
        alone.map(Arc::new)
    }

    /// Chunk `slot`'s import, where it is made: what a buffer bound in an import of its own run
    /// alone is bound in again once that import was released (`host_residency.rs`).
    fn made(&self, slot: usize) -> Option<Arc<RamChunk>> {
        let span = &self.slots[slot];
        match &*lock(&span.import) {
            Import::Made(chunk) => {
                span.untaken.store(false, Ordering::Relaxed);
                Some(Arc::clone(chunk))
            }
            Import::None | Import::Making => None,
        }
    }

    /// Take out of their slots the chunks the device's thread imported that no object took nor
    /// holds, none of them imported ahead again, for the caller to let go of with no lock held.
    fn untaken(&self) -> Vec<Arc<RamChunk>> {
        let mut gone = Vec::new();
        for slot in &self.slots {
            if !slot.untaken.load(Ordering::Relaxed) {
                continue;
            }
            let mut import = lock(&slot.import);
            // Only the slot holds it: no buffer is bound in it, no command buffer used it.
            if let Import::Made(chunk) = &*import {
                if Arc::strong_count(chunk) == 1 {
                    gone.push(Arc::clone(chunk));
                    *import = Import::None;
                    slot.tried.store(true, Ordering::Relaxed);
                }
            }
            slot.untaken.store(false, Ordering::Relaxed);
        }
        gone
    }

    /// Take out of their slots the chunks only their slot has held for their window by `now`
    /// (no buffer bound in one, no command buffer that used one pending), each window doubled for
    /// the next time, for the caller to let go of with no lock held; each is imported again by the
    /// next object over it. A chunk anything else holds starts its idleness again.
    fn idle(&self, now: Instant) -> Vec<Arc<RamChunk>> {
        let mut gone = Vec::new();
        for slot in &self.slots {
            let mut import = lock(&slot.import);
            let mut idle = lock(&slot.idle);
            let Import::Made(chunk) = &*import else {
                idle.since = None;
                continue;
            };
            // Only the slot holds it, and none can take it while its lock is held.
            if Arc::strong_count(chunk) > 1 {
                idle.since = None;
                continue;
            }
            let since = *idle.since.get_or_insert(now);
            if now.duration_since(since) < idle.window {
                continue;
            }
            gone.push(Arc::clone(chunk));
            *import = Import::None;
            idle.since = None;
            idle.window = (2 * idle.window).min(host_residency::IDLE_MOST);
        }
        gone
    }

    /// An object took chunk `slot`: the window moves to it ([`Window::after_take`]) unless it is
    /// the chunk last taken. Whether it moved.
    fn take(&self, slot: usize) -> bool {
        let mut window = lock(&self.window);
        if window.taken == Some(slot) {
            return false;
        }
        *window = window.after_take(slot);
        true
    }

    /// The chunk the device's thread imports next: the first of the window ([`Window::slots`]),
    /// within the RAM, that is neither imported nor failed by that thread, nor being imported
    /// now ([`Import::Making`]: by an object, which imports it itself); `None` while there is
    /// none.
    fn next_ahead(&self) -> Option<usize> {
        let count = self.slots.len();
        let window = *lock(&self.window);
        (window.slots().filter(|&index| index < count)).find(|&index| {
            let slot = &self.slots[index];
            !slot.tried.load(Ordering::Relaxed) && matches!(*lock(&slot.import), Import::None)
        })
    }

    /// The chunk the device's thread imports next on a device whose plan is [`Plan::listed`]
    /// ([`listed_chunks`]): the first an object asked for ([`ChunkSlot::asked`]) that is neither
    /// imported, nor being imported now, nor failed by that thread; `None` while there is none.
    /// Each ask it passes is answered: cleared.
    fn next_asked(&self) -> Option<usize> {
        (0..self.slots.len()).find(|&index| {
            let slot = &self.slots[index];
            if !slot.asked.load(Ordering::Relaxed) {
                return false;
            }
            let import = lock(&slot.import);
            slot.asked.store(false, Ordering::Relaxed);
            matches!(*import, Import::None) && !slot.tried.load(Ordering::Relaxed)
        })
    }
}

/// The device's thread that imports a RAM's chunks ahead of the objects over them
/// ([`Device::use_host_ram`]): its window's chunks one at a time ([`RamOnDevice::next_ahead`]),
/// each once ([`RamOnDevice::chunk`]) and inside a section of the exit guard; once the window
/// holds none to import, it waits for an object to move the window ([`Wake`]). It ends once the
/// use ended, the device went or was lost, the process exits, or one more import would leave the
/// device less than a quarter of its `maxMemoryAllocationCount` (`allocations.rs`), the rest then
/// left to the objects over them. It holds the device and the RAM weakly between chunks and while it waits,
/// so neither outlives its last holder for it by more than one import. A chunk it fails to
/// import (out of memory, pages the driver or the runtime refuses) is left to the first object
/// over it, which meets the same answer on its own thread, as without this thread.
///
/// Paced ([`pace`], [`fault_in`]): it runs as the system's batch work, yields between chunks,
/// and on Linux faults a chunk's pages in 2 MiB at a time before its import, so that the
/// import, which pins them, finds them in place.
fn import_ahead(device: WeakDevice, ram: Weak<RamOnDevice>, wake: Arc<Wake>) {
    let _ends = Ends(Arc::clone(&wake));
    pace();
    loop {
        let (Some(device), Some(ram)) = (device.upgrade(), ram.upgrade()) else {
            return;
        };
        let shared = device.shared();
        let Ok(importer) = shared.host_import.as_ref() else {
            return;
        };
        if device.is_lost()
            || !shared.host_ram.uses(&ram)
            || !shared.allocations.allocations.room_ahead()
        {
            return;
        }
        let Some(slot) = ram.next_ahead() else {
            drop((ram, device));
            wake.wait();
            continue;
        };
        // With no lock held: an object over the chunk meanwhile imports it itself, as before.
        fault_in(&ram.slots[slot].span);
        let Some(busy) = exit_guard::enter() else {
            return;
        };
        if ram.chunk(&device, importer, slot, By::Ahead).is_err() {
            ram.slots[slot].tried.store(true, Ordering::Relaxed);
        }
        drop(busy);
        drop((ram, device));
        std::thread::yield_now();
    }
}

/// The device's thread for a RAM it imports as [`Plan::listed`] says, in place of
/// [`import_ahead`]: it imports each chunk an object over it asked for
/// ([`RamOnDevice::next_asked`]), one at a time, each once ([`RamOnDevice::chunk`]) and inside a
/// section of the exit guard, off the threads making the objects, which are bound in imports of
/// their own runs meanwhile ([`RamOnDevice::run_alone`]); between imports, and every
/// `host_residency::TICK` while it waits, it releases the chunks only their slot has held for
/// their window ([`RamOnDevice::idle`]), each freed inside a section of the exit guard as its last
/// holder goes, and imported again once an object over it asks. Counted:
/// `WorkCounts::host_ram_chunks_released`. A chunk it fails to import is left to the objects
/// over it, which import it on their own threads, as without this thread; so is a chunk asked
/// for once one more import would leave the device less than a quarter of its
/// `maxMemoryAllocationCount` (`allocations.rs`). It ends once the use ended, the device went or
/// was lost, or the process exits, and holds the device and the RAM weakly between chunks and
/// while it waits. Paced as [`import_ahead`] is ([`pace`], [`fault_in`]).
fn listed_chunks(device: WeakDevice, ram: Weak<RamOnDevice>, wake: Arc<Wake>) {
    let _ends = Ends(Arc::clone(&wake));
    pace();
    loop {
        let (Some(device), Some(ram)) = (device.upgrade(), ram.upgrade()) else {
            return;
        };
        let shared = device.shared();
        let Ok(importer) = shared.host_import.as_ref() else {
            return;
        };
        if device.is_lost() || !shared.host_ram.uses(&ram) {
            return;
        }
        let asked = (ram.next_asked()).filter(|_| shared.allocations.allocations.room_ahead());
        if let Some(slot) = asked {
            // With no lock held: an object over the chunk meanwhile is bound in its run's import.
            fault_in(&ram.slots[slot].span);
            let Some(busy) = exit_guard::enter() else {
                return;
            };
            if ram.chunk(&device, importer, slot, By::Asked).is_err() {
                ram.slots[slot].tried.store(true, Ordering::Relaxed);
            }
            drop(busy);
            drop((ram, device));
            std::thread::yield_now();
            continue;
        }
        let gone = ram.idle(Instant::now());
        for _ in &gone {
            shared.counts.host_ram_chunk_released();
        }
        drop(gone);
        drop((ram, device));
        wake.wait_for(host_residency::TICK);
    }
}

/// The scheduling the device's thread ([`import_ahead`]) runs at: the system's batch work, never
/// a lower weight. It holds a chunk's lock, which an object over the chunk waits on, and the
/// pin it makes holds the process's memory map lock, which every `mmap`, `munmap` and, behind
/// one queued, every page fault of the process waits on: a thread the system starves would hold
/// those for as long as it is starved. On Linux, `SCHED_BATCH` (its nice value kept): measured on
/// an RTX 5080 laptop's 24 threads all busy, a thread faulting in 64 MiB pieces of fresh memory
/// at nice 19 kept another thread's `mmap` waiting up to 511 ms and at `SCHED_IDLE` 1039 ms,
/// where at `SCHED_BATCH` it waited at most 12 ms, as at the default. On Apple's systems, the
/// utility class, the lowest the system does not hold to its efficiency cores alone.
fn pace() {
    #[cfg(target_os = "linux")]
    {
        let keep = libc::sched_param { sched_priority: 0 };
        // SAFETY: the calling thread's own policy, from a live parameter; a refusal leaves it.
        let _ = unsafe { libc::sched_setscheduler(0, libc::SCHED_BATCH, &keep) };
    }
    #[cfg(target_vendor = "apple")]
    {
        // SAFETY: the calling thread's own class; a refusal leaves it.
        let _ =
            unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_UTILITY, 0) };
    }
}

/// Fault in the pages of `span`, a chunk of the RAM, 2 MiB at a time, yielding between
/// (`MADV_POPULATE_WRITE`, Linux 5.14: each page as a write would, no byte changed), so that the
/// chunk's import pins pages in place: each step holds the process's memory map lock for one
/// huge page's fault, where an import of fresh pages holds it while it faults in all 64 MiB.
/// A kernel that refuses (an older one) leaves the faults to the import, as before.
#[cfg(target_os = "linux")]
fn fault_in(span: &Range<u64>) {
    const STEP: u64 = 2 * 1024 * 1024;
    let mut at = span.start;
    while at < span.end {
        let length = STEP.min(span.end - at);
        // SAFETY: the span is pages of the RAM's ranges, mapped readable and writable while the
        // RAM lives (`HostRam::new`), which the caller holds; the advice writes no byte.
        let faulted = unsafe {
            libc::madvise(
                at as *mut libc::c_void,
                length as usize,
                libc::MADV_POPULATE_WRITE,
            )
        };
        if faulted != 0 {
            return;
        }
        std::thread::yield_now();
        at += length;
    }
}

/// Elsewhere the import faults the chunk's pages in itself.
#[cfg(not(target_os = "linux"))]
fn fault_in(_: &Range<u64>) {}

impl Device {
    /// Give the chunks the device's thread imported ahead and no object took back once the
    /// device's allocations come within a quarter of its limit (`allocations.rs`): on a thread of
    /// its own, started then, which frees them with no lock of the allocating caller held, so
    /// that they are gone before they could take the last allocations. An object over one later
    /// imports it as it is made. Counted: `WorkCounts::host_ram_chunks_given_back`.
    pub(crate) fn give_back_ahead_near_the_limit(&self) {
        let weak = self.downgrade();
        let give_back = move || {
            let device = weak.clone();
            let builder = std::thread::Builder::new().name("m2v-host-ram".to_owned());
            // A thread the system cannot start leaves the chunks imported.
            let _ = exit_guard::spawn(builder, move || give_back_untaken(device));
        };
        (self.shared().allocations.allocations).give_back_with(Box::new(give_back));
    }
}

/// [`Device::give_back_ahead_near_the_limit`]'s thread: every RAM's untaken chunks out of their
/// slots, then let go of, each freed as its last holder goes.
fn give_back_untaken(device: WeakDevice) {
    let Some(device) = device.upgrade() else {
        return;
    };
    let shared = device.shared();
    let gone: Vec<Arc<RamChunk>> = (shared.host_ram.used().iter())
        .flat_map(|ram| ram.untaken())
        .collect();
    for _ in &gone {
        shared.counts.host_ram_chunk_given_back();
    }
    drop(gone);
}

/// A device's import of one chunk of a RAM (`VK_EXT_external_memory_host`), or of one object's
/// run inside a chunk not imported yet ([`Imported`]): the memory, and a `VkBuffer` over the whole
/// of it, for the runtime's own copies between the chunk and memory of the device's own alone; no
/// Metal buffer, root entry or binding ever names it. Buffers bound
/// in the chunk ([`Device::bind_in_chunk`]) hold it through their placement (`memory.rs`), and so
/// do the command buffers that use them, until they completed; the device holds it while it uses
/// the RAM ([`HostRamUse`]). The last holder destroys the `VkBuffer` and frees the memory, and the
/// RAM, which the chunk holds, goes after the free: its keep-alive with it once no chunk holds it.
/// Released while idle only where each listed import costs every submission ([`Plan::listed`],
/// [`listed_chunks`]). Left to the process's end while it exits.
pub(crate) struct RamChunk {
    device: Device,
    /// The host bytes imported.
    span: Range<u64>,
    /// A chunk, or one object's run alone.
    of: Imported,
    /// The `VkBuffer` over the whole chunk.
    raw: vk::Buffer,
    /// The import's placement: a dedicated block of the device's memory (`memory::import_host`),
    /// or the import of the RAM's dma-buf of the span, which the placement keeps
    /// (`Placement::dma_buf`).
    placement: Placement,
    /// The RAM, let go of once the memory is freed; `None` once let go of.
    ram: Option<Arc<HostRam>>,
}

impl std::fmt::Debug for RamChunk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RamChunk")
            .field("start", &self.span.start)
            .field("bytes", &(self.span.end - self.span.start))
            .finish()
    }
}

impl RamChunk {
    /// The memory imported.
    pub(crate) fn memory(&self) -> vk::DeviceMemory {
        self.placement.memory
    }

    /// Whether it is one object's run alone ([`Imported::Run`]), not a chunk.
    pub(crate) fn is_run(&self) -> bool {
        self.of == Imported::Run
    }

    /// The index of the memory's type.
    pub(crate) fn type_index(&self) -> u32 {
        self.placement.type_index()
    }

    /// The external memory the import is, which a `VkBuffer` bound in it is made for
    /// (`VUID-vkBindBufferMemory-memory-02985`).
    fn handle_type(&self) -> vk::ExternalMemoryHandleTypeFlags {
        match self.placement.over_dma_buf() {
            true => vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            false => vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
        }
    }

    /// The RAM's address of the byte at `offset` of the chunk's memory.
    pub(crate) fn ram_address(&self, offset: u64) -> u64 {
        self.span.start + (offset - self.placement.offset)
    }

    /// The import of the RAM's dma-buf of the span, with the RAM's source, where the chunk is
    /// one: what a texture over an object bound in it makes the image of its own pages from
    /// (`dma_buf.rs`).
    #[cfg(unix)]
    pub(crate) fn dma_buf_pages(&self) -> Option<&Arc<crate::dma_buf::DmaBufPages>> {
        self.placement.dma_buf_pages()
    }
}

impl Drop for RamChunk {
    fn drop(&mut self) {
        let ram = self.ram.take();
        // The Vulkan teardown is a section of the exit guard.
        let Some(busy) = exit_guard::enter() else {
            // The process is exiting: the chunk stays imported to its end, and the RAM's
            // keep-alive with it.
            std::mem::forget(ram);
            return;
        };
        let shared = self.device.shared();
        // SAFETY: nothing uses the buffer: no command names it but the runtime's copies, whose
        // command buffers hold the chunk until they completed.
        unsafe { release::object(shared.vk(), self.raw) };
        let after = ram.map(|ram| Box::new(RamAfterFree(Some(ram))) as Box<dyn Send>);
        // Every buffer bound in the chunk went before it (each holds it), and every command
        // buffer that used one completed; the free may wait for the batches in flight
        // (`freed_memory.rs`), and the RAM with it. An import of a dma-buf is freed as the
        // placement drops, after this: the RAM waits for that free.
        let after = lock(&shared.memory).release_then(shared.vk(), &self.placement, after);
        #[cfg(unix)]
        let after = match (after, self.placement.dma_buf_pages()) {
            (Some(after), Some(imports)) => {
                imports.after_free(after);
                None
            }
            (after, _) => after,
        };
        match self.of {
            Imported::Chunk => shared.counts.host_ram_chunk_freed(),
            Imported::Run => shared.counts.run_import_freed(),
        }
        drop(busy);
        drop(after);
    }
}

/// The RAM a chunk held, let go of once the chunk's memory is freed: as the program's, since its
/// drop may drop the caller's keep-alive.
struct RamAfterFree(Option<Arc<HostRam>>);

impl Drop for RamAfterFree {
    fn drop(&mut self) {
        let Some(ram) = self.0.take() else {
            return;
        };
        if exit_guard::enter().is_none() {
            std::mem::forget(ram);
            return;
        }
        exit_guard::program_release(move || drop(ram));
    }
}

/// Where a resolver's answer puts a no-copy object's pages (`HostMemory::with_runs`).
pub(crate) enum Routed {
    /// One run inside chunk `slot` of `ram`, from host address `start`: the object is bound in
    /// the chunk ([`Device::bind_in_chunk`]).
    InChunk {
        ram: Arc<RamOnDevice>,
        slot: usize,
        start: u64,
    },
    /// More than one run, or one crossing a chunk's end: the object takes the gate's path, the
    /// runs kept with it.
    Runs {
        ram: Arc<RamOnDevice>,
        runs: Vec<Range<u64>>,
    },
}

impl Routed {
    /// The RAM and the runs, kept with the object until its rebind or release.
    pub(crate) fn kept(&self) -> (Arc<RamOnDevice>, Box<[Range<u64>]>) {
        match self {
            Routed::InChunk { ram, slot, start } => {
                let end = ram.slots[*slot].span.end;
                (Arc::clone(ram), std::iter::once(*start..end).collect())
            }
            Routed::Runs { ram, runs } => (Arc::clone(ram), runs.clone().into_boxed_slice()),
        }
    }

    /// What binds a buffer routed so and bound at `placement` again once its import is released
    /// while idle (`host_residency.rs`), where that is an import of its own run alone
    /// ([`Imported::Run`]): its chunk was not imported at its making.
    pub(crate) fn run_again(&self, placement: &Placement) -> Option<RunAgain> {
        let Routed::InChunk { ram, slot, start } = self else {
            return None;
        };
        (placement.over_ram_run()).then(|| RunAgain {
            ram: Arc::clone(ram),
            slot: *slot,
            start: *start,
        })
    }
}

/// What binds a buffer bound in an import of its own run alone ([`Imported::Run`]) again, once
/// that import is released while idle (`host_residency.rs`): the RAM, the run's chunk, and the
/// run's first host address.
pub(crate) struct RunAgain {
    ram: Arc<RamOnDevice>,
    slot: usize,
    start: u64,
}

/// How `runs`, a resolver's answer for `length` bytes of whole `page`s, lie in the chunks
/// `spans` of a RAM: `Ok(Some(slot))` for one run inside one chunk, `Ok(None)` for runs of the
/// RAM that are not (several, or one crossing a chunk's end); `Err` where the answer breaks the
/// resolver's contract: no run, a run that is empty or off the page, runs that do not total
/// `length`, or a run outside the RAM's `ranges`.
fn place_runs(
    runs: &[Range<u64>],
    length: u64,
    page: u64,
    ranges: &[Range<u64>],
    spans: &[Range<u64>],
) -> Result<Option<usize>, ()> {
    let mut total: u64 = 0;
    for run in runs {
        let whole = run.start.is_multiple_of(page) && run.end.is_multiple_of(page);
        if run.start >= run.end || !whole {
            return Err(());
        }
        let inside = ranges
            .iter()
            .any(|range| range.start <= run.start && run.end <= range.end);
        if !inside {
            return Err(());
        }
        total = total.checked_add(run.end - run.start).ok_or(())?;
    }
    if runs.is_empty() || total != length {
        return Err(());
    }
    let [run] = runs else {
        return Ok(None);
    };
    let slot = span_of(spans, run.start).ok_or(())?;
    Ok((run.end <= spans[slot].end).then_some(slot))
}

/// The RAMs a device uses ([`Device::use_host_ram`]).
#[derive(Default)]
pub(crate) struct RamSource {
    rams: Mutex<Vec<Arc<RamOnDevice>>>,
    /// The tests' hold on the device's thread ([`Device::hold_host_ram_imports`]): a use begun
    /// while it is set starts none, so its chunks are imported by the objects over them alone.
    #[cfg(test)]
    held: AtomicBool,
}

impl RamSource {
    /// Add `ram`, unless one of its ranges meets a range of a RAM the device uses.
    fn add(&self, ram: &Arc<HostRam>, plan: Plan) -> Option<Arc<RamOnDevice>> {
        let mut rams = lock(&self.rams);
        if rams.iter().any(|used| used.ram.meets(ram)) {
            return None;
        }
        let used = Arc::new(RamOnDevice::new(ram, plan));
        rams.push(Arc::clone(&used));
        Some(used)
    }

    /// Remove `ram`: what the device held of it, for the caller to drop with no lock held.
    fn remove(&self, ram: &Arc<RamOnDevice>) -> Option<Arc<RamOnDevice>> {
        let mut rams = lock(&self.rams);
        let at = rams.iter().position(|used| Arc::ptr_eq(used, ram))?;
        Some(rams.swap_remove(at))
    }

    /// The RAMs the device uses now.
    fn used(&self) -> Vec<Arc<RamOnDevice>> {
        lock(&self.rams).clone()
    }

    /// Whether the device still uses `ram`: its use has not ended.
    fn uses(&self, ram: &Arc<RamOnDevice>) -> bool {
        lock(&self.rams).iter().any(|used| Arc::ptr_eq(used, ram))
    }
}

#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
impl Device {
    /// Start no thread importing a RAM's chunks ahead ([`import_ahead`]) for the uses begun from
    /// now on: the tests' way to see each chunk imported by the first object over it, as one the
    /// thread has not reached is.
    pub(crate) fn hold_host_ram_imports(&self) {
        self.shared().host_ram.held.store(true, Ordering::Relaxed);
    }

    /// Each RAM the device uses: whether it imports it as each listed import costing every
    /// submission ([`Plan::listed`]), and its chunks (read by the dma-buf tests, Linux's).
    #[cfg(target_os = "linux")]
    pub(crate) fn host_ram_chunks(&self) -> Vec<(bool, Vec<Range<u64>>)> {
        let rams = self.shared().host_ram.used();
        (rams.iter())
            .map(|ram| (ram.plan.listed, ram.spans()))
            .collect()
    }

    /// Run `during` with the lock of the state of the chunk that holds host address `at` held, a
    /// chunk of a RAM the device uses, as the device's thread holds it as it passes each chunk
    /// of its window ([`RamOnDevice::next_ahead`]): the tests' way to make an object over the
    /// chunk meanwhile.
    pub(crate) fn looking_at_host_ram_chunk<R>(&self, at: u64, during: impl FnOnce() -> R) -> R {
        let rams = self.shared().host_ram.used();
        let (ram, slot) = (rams.iter())
            .find_map(|ram| Some((ram, span_of(&ram.spans(), at)?)))
            .expect("a chunk of a RAM the device uses");
        let _look = lock(&ram.slots[slot].import);
        during()
    }
}

impl Device {
    /// Use the guest RAM `ram` (a runtime extension: Metal has none): from now on, a no-copy
    /// buffer or heap made on this device over pages its caller resolves to one run of the RAM
    /// (`HostMemory::with_runs`) is bound in the device's import of the chunk of the RAM that run
    /// lies in, in place of an import of its own (the module docs of `host_ram.rs`). For a device
    /// that refuses views of the pages made through dma-bufs, so that each object's own import
    /// would take a kernel mapping per page. A thread of the device's, started here, keeps the
    /// chunk an object last took imported, and a window of chunks past it in the direction the
    /// takes go, up the RAM or down it: one chunk (from the RAM's start before any take), four
    /// once a take lands inside the window or next to the chunk last taken, so that a run of
    /// objects over consecutive RAM finds its next chunks imported
    /// (`WorkCounts::host_ram_chunks_imported_ahead`). An object over a chunk not imported yet is
    /// bound in an import of its own run alone while the thread imports the chunk, so that no
    /// making waits for a chunk's import (`WorkCounts::run_bound_buffers_made`); on a driver whose
    /// barriers do not order blits through two imports of the same pages
    /// (`RuntimeDriverDefect::AliasedImportBlitsUnordered`, MoltenVK's) it imports the chunk
    /// instead, or waits for the thread's import of it. The thread
    /// faults in and pins no more than the window past the objects, and never goes back. It is
    /// paced (the system's batch work, a yield between chunks) and stops a quarter of the
    /// device's `maxMemoryAllocationCount` short of it, the chunks it imported that no object
    /// took given back once the device's allocations come that near
    /// (`WorkCounts::host_ram_chunks_given_back`); no run is imported alone that near either. A
    /// system that cannot start the thread, or a thread that stopped, leaves each chunk to the
    /// first object over it, which imports it. The use lasts until the returned [`HostRamUse`]
    /// is dropped; a device may use several RAMs at once whose ranges are apart, and a new RAM
    /// once an earlier use of the same ranges ended (a VMM's next guest).
    ///
    /// Where the device's held imports add to the cost of every submission and it imports the
    /// RAM's dma-bufs (RADV), each listed import costs every submission, so the RAM is imported
    /// in a few long chunks instead: as long as one dma-buf of its source holds
    /// ([`HostRam::with_largest_dma_buf`]) and the device's `maxMemoryAllocationSize`,
    /// `maxBufferSize` and `maxStorageBufferRange` take, in whole 64 MiB. There the thread
    /// imports nothing ahead: the first object over a chunk not imported asks it for the chunk,
    /// which it imports off the object's thread (a long chunk's import pins all its pages,
    /// hundreds of milliseconds), and that object and those over the chunk meanwhile are bound in
    /// imports of their own runs alone, as above; a buffer's is released while idle, the buffer
    /// bound in the chunk at its next use, and a heap's, or a buffer's whose address was handed
    /// out, stays while it lives. The thread releases a chunk no object nor pending command
    /// buffer held for 2 s, the window doubled at each release of the chunk up to 64 s
    /// (`WorkCounts::host_ram_chunks_released`); the next object over it asks for it again.
    ///
    /// A RAM with a source of dma-bufs ([`HostRam::with_dma_buf`]) has its chunks imported as
    /// dma-bufs where the device's held pointer imports add to every submission's cost, or where
    /// it refuses their pointer (`WorkCounts::host_ram_dma_buf_imports`); for it the call first
    /// waits for the device's submission checks, which take milliseconds, for at most 10 s. On a
    /// device that refuses pointers to shared memory ([`Device::imports_shared_host_pointers`]
    /// false: RADV), a RAM of shared memory (QEMU's `memory-backend-memfd,share=on` RAM) is
    /// imported so with a source; with none, it is taken but imports no chunk, and each object
    /// over it is refused as its own import of the pointer is (`Unsupported`, `HostMemory`).
    ///
    /// Refused: a device without `VK_EXT_external_memory_host` (`Unsupported`, naming it); a RAM
    /// whose ranges meet those of a RAM the device uses already, itself among them (`Invalid`,
    /// `Problem::HostRam`); a lost device.
    pub fn use_host_ram(&self, ram: &Arc<HostRam>) -> Result<HostRamUse, Error> {
        const CALL: &str = "use_host_ram";
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let shared = self.shared();
        if let Err(unsupported) = &shared.host_import {
            return Err(Error::Unsupported(unsupported.clone()));
        }
        // A RAM with a source of dma-bufs imports its chunks by the import tax's verdict, so the
        // device's submission checks answer first (`submit_check.rs`), waited for at most
        // `ANSWER_WAIT`: they take milliseconds.
        if ram.dma_buf.is_some() {
            self.settle_submit_checks(crate::submit_check::ANSWER_WAIT);
        }
        let plan = self.ram_plan(ram);
        let used =
            (shared.host_ram.add(ram, plan)).ok_or_else(|| invalid(CALL, Problem::HostRam))?;
        #[cfg(test)]
        let held = shared.host_ram.held.load(Ordering::Relaxed);
        #[cfg(not(test))]
        let held = false;
        if !held {
            let (device, ram) = (self.downgrade(), Arc::downgrade(&used));
            let wake = Arc::clone(&used.wake);
            let builder = std::thread::Builder::new().name("m2v-host-ram".to_owned());
            used.wake.running.store(true, Ordering::Relaxed);
            // A thread the system cannot start leaves every chunk to the first object over it,
            // and, on a device whose plan is listed, every chunk imported while the use lasts.
            let started = match plan.listed {
                true => exit_guard::spawn(builder, move || listed_chunks(device, ram, wake)),
                false => exit_guard::spawn(builder, move || import_ahead(device, ram, wake)),
            };
            if started.is_err() {
                used.wake.running.store(false, Ordering::Relaxed);
            }
        }
        Ok(HostRamUse {
            device: self.clone(),
            ram: used,
        })
    }
}

impl Device {
    /// The pages of a buffer bound in an import of its own run alone, that import released
    /// while idle (`host_residency.rs`), bound again in the run's chunk's import as `again`
    /// says (its window and length), as at its making ([`Device::bind_in`]): `None` where the
    /// chunk is not imported, or the buffer cannot be bound in it.
    #[cfg(unix)]
    pub(crate) fn ram_run_in_chunk(&self, run: &RunAgain, again: &Again) -> Option<Pages> {
        let chunk = run.ram.made(run.slot)?;
        // SAFETY: the window's pages are the bytes of the run in the RAM, as at the buffer's
        // making (`HostMemory::with_runs`).
        let (buffer, placement) =
            unsafe { self.bind_in(&chunk, run.start, &again.window, again.length) }?;
        let base = buffer.base;
        Some(self.pages_of(
            buffer.raw,
            placement,
            again.window.offset,
            base,
            buffer.size - base,
        ))
    }

    /// The pages of a buffer bound in an import of its own run alone, that import released
    /// while idle (`host_residency.rs`), bound again in a new import of the run alone
    /// ([`RamOnDevice::run_alone`]: the chunk's import where it is made, the device's thread
    /// asked for it where not), never the chunk's own import nor a wait for one. Refused as the
    /// run's import is, or with the device lost.
    #[cfg(unix)]
    pub(crate) fn ram_run_alone_again(
        &self,
        run: &RunAgain,
        again: &Again,
    ) -> Result<Pages, Error> {
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let importer = (self.shared().host_import.as_ref()).map_err(|_| {
            internal("a guest RAM's run imported again on a device that imports none")
        })?;
        let window = &again.window;
        let span = run.start..run.start + window.size;
        let chunk = run.ram.run_alone(self, importer, run.slot, span)?;
        // SAFETY: as in `ram_run_in_chunk`.
        let (buffer, placement) = unsafe { self.bind_in(&chunk, run.start, window, again.length) }
            .ok_or_else(|| internal("a buffer bound again in its guest RAM run's import"))?;
        let base = buffer.base;
        Ok(self.pages_of(
            buffer.raw,
            placement,
            window.offset,
            base,
            buffer.size - base,
        ))
    }
}

impl Device {
    /// How the device imports `ram`'s chunks (the module docs): [`Plan::listed`] where the RAM has
    /// a source of dma-bufs, the device imports dma-bufs and its held imports add to the cost of
    /// every submission (its submission checks' import tax, `submit_check.rs`), so that its
    /// chunks are its dma-bufs; [`Plan::AHEAD`] elsewhere.
    fn ram_plan(&self, ram: &HostRam) -> Plan {
        let imports_dma_bufs =
            (self.shared().host_import.as_ref()).is_ok_and(|importer| importer.imports_dma_bufs());
        if ram.dma_buf.is_none() || !imports_dma_bufs || !self.pointer_imports_tax_submissions() {
            return Plan::AHEAD;
        }
        let storage = self
            .properties()
            .description
            .limits
            .max_storage_buffer_range;
        let allocation = self.compile_device().allocation();
        Plan {
            bytes: listed_chunk_bytes(
                ram.largest_dma_buf,
                allocation.max_memory_allocation_size,
                allocation.max_buffer_size,
                u64::from(storage),
            ),
            listed: true,
        }
    }
}

impl Device {
    /// Where `resolver` puts the pages of `window` (`HostMemory::with_runs`): asked once, here,
    /// with no lock of the runtime's held, where the device uses a RAM; `None` where it uses none
    /// or the resolver answers `None`. An answer that breaks the resolver's contract (runs that
    /// are not whole host pages of one RAM the device uses, or do not total the window) is
    /// refused (`Invalid`, `Problem::HostRam`).
    pub(crate) fn route_runs(
        &self,
        call: &'static str,
        importer: &Importer,
        resolver: &RunResolver,
        window: &ImportWindow,
    ) -> Result<Option<Routed>, Error> {
        let rams = self.shared().host_ram.used();
        if rams.is_empty() {
            return Ok(None);
        }
        let Some(runs) = resolver(window.start, window.size) else {
            return Ok(None);
        };
        let refused = || invalid(call, Problem::HostRam);
        let first = runs.first().ok_or_else(refused)?.start;
        let ram = (rams.into_iter())
            .find(|used| used.ram.ranges.iter().any(|range| range.contains(&first)))
            .ok_or_else(refused)?;
        let spans = ram.spans();
        let placed = place_runs(&runs, window.size, importer.page, &ram.ram.ranges, &spans);
        Ok(Some(match placed.map_err(|()| refused())? {
            Some(slot) => Routed::InChunk {
                ram,
                slot,
                start: runs[0].start,
            },
            None => Routed::Runs { ram, runs },
        }))
    }

    /// A no-copy object's `VkBuffer`, of `length` bytes `window.offset` into `window`, bound in
    /// the chunk its pages' one run lies in (`routed`, [`Routed::InChunk`]), or, where that chunk
    /// is not imported yet, in an import of the run alone ([`RamOnDevice::for_object`]): made as
    /// an import of the window's own pages would make it (`Device::host_buffer`), for the
    /// import's memory (host allocations, or dma-bufs, [`HostRam::with_dma_buf`]), and bound where
    /// its bytes lie in the import, with `binds_in`'s checks against the window, so its size
    /// reaches no page past the window's. The placement holds the import; its host address is the
    /// window's, so the buffer's `contents` stays the caller's pointer. `None` where neither can
    /// be imported or the buffer cannot be bound there exactly (its alignment, its memory types):
    /// the object then takes the runtime's other paths.
    ///
    /// # Safety
    ///
    /// The window's pages are the bytes of the run in the RAM's chunk: the same backing.
    pub(crate) unsafe fn bind_in_chunk(
        &self,
        importer: &Importer,
        routed: &Routed,
        window: &ImportWindow,
        length: u64,
    ) -> Option<(HostBuffer, Placement)> {
        let Routed::InChunk { ram, slot, start } = routed else {
            return None;
        };
        let run = *start..*start + window.size;
        let chunk = ram.for_object(self, importer, *slot, run).ok()?;
        // SAFETY: the caller's.
        let bound = unsafe { self.bind_in(&chunk, *start, window, length) }?;
        match chunk.of {
            Imported::Chunk => self.shared().counts.chunk_bound_buffer_made(),
            Imported::Run => self.shared().counts.run_bound_buffer_made(),
        }
        Some(bound)
    }

    /// A `VkBuffer` of `length` bytes over `window`, whose pages are the RAM's from host address
    /// `start`, inside `chunk`'s span, bound where they lie in `chunk`'s import, as
    /// [`Device::bind_in_chunk`] binds it, and its placement, which holds the import; `None`
    /// where it cannot be bound there exactly (its alignment, its memory types).
    ///
    /// # Safety
    ///
    /// The window's pages are the bytes of the run in the RAM's chunk: the same backing.
    unsafe fn bind_in(
        &self,
        chunk: &Arc<RamChunk>,
        start: u64,
        window: &ImportWindow,
        length: u64,
    ) -> Option<(HostBuffer, Placement)> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let buffer = self.host_buffer(window, length, chunk.handle_type()).ok()?;
        let requirements = buffer.requirements;
        // Where the window starts in the import, and the buffer in the window.
        let in_chunk = start - chunk.span.start;
        let bound_at = binds_in(window, buffer.base, buffer.size, &requirements)
            .ok()
            .map(|at| in_chunk + at)
            .filter(|at| at.is_multiple_of(requirements.alignment.max(1)))
            .filter(|_| requirements.memory_type_bits & (1 << chunk.type_index()) != 0);
        let mapped = NonNull::new(window.start as *mut u8);
        let (Some(bound_at), Some(mapped)) = (bound_at, mapped) else {
            // SAFETY: the buffer is unused.
            unsafe { release::object(raw_device, buffer.raw) };
            return None;
        };
        let offset = chunk.placement.offset + bound_at;
        // SAFETY: the buffer is unbound and made for host allocations, and the chunk's memory
        // holds its size and requirements from `offset` (`binds_in` against the window, which
        // lies in the chunk).
        let bound =
            unsafe { crate::creation::bind_buffer(raw_device, buffer.raw, chunk.memory(), offset) };
        if bound.is_err() {
            // SAFETY: the buffer is unused.
            unsafe { release::object(raw_device, buffer.raw) };
            return None;
        }
        let placement = Placement::ram_chunk(
            Arc::clone(chunk),
            chunk.placement.offset + in_chunk,
            window.size,
            Mapped::of_caller(mapped),
        );
        Some((buffer, placement))
    }

    /// Import `span`, host pages of `ram`, as `of` (a chunk, or an object's run inside one): by
    /// its pointer ([`Device::import_chunk_by_pointer`]), or, for a RAM with a source of
    /// dma-bufs ([`HostRam::with_dma_buf`]), as one dma-buf of it (`Device::ram_span_as_dma_buf`):
    /// that first where the device's held pointer imports add to the cost of every submission
    /// (its submission checks' import tax, `submit_check.rs`), the pointer where that fails; the
    /// pointer first elsewhere, the dma-buf where the pointer is refused. Counted:
    /// `WorkCounts::host_ram_dma_buf_imports` (each import of a dma-buf) and
    /// `host_ram_pointer_imports_failed` (each pointer import that failed).
    ///
    /// # Safety
    ///
    /// `span` is pages of `ram`'s ranges, which stay mapped while the chunk holds `ram`, and the
    /// RAM's source's dma-bufs hold exactly the pages asked for.
    unsafe fn import_chunk(
        &self,
        importer: &Importer,
        ram: &Arc<HostRam>,
        span: Range<u64>,
        of: Imported,
    ) -> Result<RamChunk, Error> {
        let by_pointer = |span: Range<u64>| {
            // SAFETY: the caller's.
            let imported = unsafe { self.import_chunk_by_pointer(importer, ram, span, of) };
            if imported.is_err() {
                self.shared().counts.host_ram_pointer_import_failed();
            }
            imported
        };
        let Some(source) = ram.dma_buf.as_ref() else {
            return by_pointer(span);
        };
        let as_dma_buf = |span: Range<u64>| {
            // SAFETY: the caller's.
            let (raw, placement) =
                unsafe { self.ram_span_as_dma_buf(importer, span.clone(), source) }?;
            self.shared().counts.host_ram_dma_buf_imported();
            Ok(self.ram_chunk(ram, span, of, raw, placement))
        };
        match self.pointer_imports_tax_submissions() {
            true => as_dma_buf(span.clone()).or_else(|_: Error| by_pointer(span)),
            false => match by_pointer(span.clone()) {
                Ok(chunk) => Ok(chunk),
                Err(refused) => as_dma_buf(span).map_err(|_: Error| refused),
            },
        }
    }

    /// The chunk of `span`, as `of`, whose `VkBuffer` `raw` over the whole of it is bound in
    /// `placement`, holding `ram`; counted made.
    fn ram_chunk(
        &self,
        ram: &Arc<HostRam>,
        span: Range<u64>,
        of: Imported,
        raw: vk::Buffer,
        placement: Placement,
    ) -> RamChunk {
        let counts = &self.shared().counts;
        match of {
            Imported::Chunk => counts.host_ram_chunk_imported(span.end - span.start),
            Imported::Run => counts.run_import_made(),
        }
        RamChunk {
            device: self.clone(),
            span,
            of,
            raw,
            placement,
            ram: Some(Arc::clone(ram)),
        }
    }

    /// Import `span`, host pages of `ram`, as `of` (a chunk, or an object's run inside one), by
    /// its pointer: a dedicated block of memory of a coherent type both the pages and a
    /// host-allocation buffer allow, with the device-address flag where the device enables
    /// `bufferDeviceAddress` (`memory::import_host`), and a `VkBuffer` over the whole of it.
    /// Refused where the device's import alignment misses the span's start or end, it passes the
    /// device's allocation size, or the driver refuses the pages (or the runtime does for it,
    /// `Device::refuse_unpinnable`).
    ///
    /// # Safety
    ///
    /// `span` is pages of `ram`'s ranges, which stay mapped while the chunk holds `ram`.
    unsafe fn import_chunk_by_pointer(
        &self,
        importer: &Importer,
        ram: &Arc<HostRam>,
        span: Range<u64>,
        of: Imported,
    ) -> Result<RamChunk, Error> {
        let shared = self.shared();
        let raw_device = shared.vk();
        let size = span.end - span.start;
        let alignment = importer.alignment.max(1);
        if !span.start.is_multiple_of(alignment) || !size.is_multiple_of(alignment) {
            return Err(not_importable());
        }
        let largest = self
            .compile_device()
            .allocation()
            .max_memory_allocation_size;
        if size > largest {
            return Err(Error::OutOfMemory(MemoryKind::Device));
        }
        let pointer = NonNull::new(span.start as *mut u8).ok_or_else(not_importable)?;
        // Pages the kernel cannot pin, never handed to a driver whose failed import of them
        // breaks its later allocations.
        self.refuse_unpinnable(pointer, size)?;
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let info = shared.sharing().buffer(
            vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage(shared.device_address))
                .push_next(&mut external),
        );
        // SAFETY: the device is live and `info` and its chain valid.
        let raw = unsafe { crate::creation::buffer(raw_device, &info) }.map_err(vulkan)?;
        shared.counts.buffer_made(info.usage);
        let discard = || {
            // SAFETY: the buffer is unused.
            unsafe { release::object(raw_device, raw) };
        };
        // SAFETY: `raw` is a live buffer of this device.
        let requirements = unsafe { raw_device.get_buffer_memory_requirements(raw) };
        // SAFETY: the pages are host memory aligned to the import alignment (checked above).
        let types = match unsafe { importer.memory_types(raw_device, pointer) } {
            Ok(types) => types & requirements.memory_type_bits,
            Err(error) => {
                discard();
                return Err(import_refused(error));
            }
        };
        if requirements.size > size {
            discard();
            return Err(not_importable());
        }
        // SAFETY: the caller's contract keeps the pages mapped while the chunk, which frees the
        // placement as it goes, holds the RAM.
        let imported = unsafe {
            super::memory::import_host(
                &shared.memory,
                raw_device,
                (pointer, None),
                size,
                types,
                MemoryClass::Shared,
            )
        };
        let placement = match imported {
            Ok(placement) => placement,
            Err(failure) => {
                discard();
                return Err(match failure {
                    ImportFailure::NoMemoryType => not_importable(),
                    ImportFailure::Vulkan(error) => import_refused(error),
                });
            }
        };
        // SAFETY: the placement is the pages' memory, of the buffer's size and requirements, and
        // the buffer is unbound.
        let bound = unsafe {
            crate::creation::bind_buffer(raw_device, raw, placement.memory, placement.offset)
        };
        if let Err(error) = bound {
            lock(&shared.memory).free(raw_device, &placement);
            discard();
            return Err(vulkan(error));
        }
        Ok(self.ram_chunk(ram, span, of, raw, placement))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: u64 = 4096;

    #[test]
    fn ranges_of_whole_pages_apart_are_taken_by_start() {
        let ranges = checked_ranges([(3 * PAGE, PAGE), (0, 2 * PAGE)], PAGE);
        assert_eq!(ranges, Some(vec![0..2 * PAGE, 3 * PAGE..4 * PAGE]));
        // Ranges that touch are apart.
        let touching = checked_ranges([(0, PAGE), (PAGE, PAGE)], PAGE);
        assert_eq!(touching, Some(vec![0..PAGE, PAGE..2 * PAGE]));
    }

    #[test]
    fn ranges_off_the_page_empty_overlapping_or_none_are_refused() {
        assert_eq!(checked_ranges([], PAGE), None);
        assert_eq!(checked_ranges([(PAGE, 0)], PAGE), None);
        assert_eq!(checked_ranges([(PAGE + 1, PAGE)], PAGE), None);
        assert_eq!(checked_ranges([(PAGE, PAGE + 1)], PAGE), None);
        assert_eq!(checked_ranges([(0, 2 * PAGE), (PAGE, PAGE)], PAGE), None);
        assert_eq!(checked_ranges([(u64::MAX - PAGE + 1, PAGE)], PAGE), None);
    }

    #[test]
    fn host_ram_new_checks_its_ranges() {
        let page = host_page().expect("a Unix host's page size");
        let mut pages = vec![0u8; 4 * page as usize];
        let base = pages.as_mut_ptr() as u64;
        let start = base.next_multiple_of(page);
        let at = |address: u64| NonNull::new(address as *mut u8).unwrap();
        // SAFETY: the bytes are a live allocation of this test, never imported: no device uses
        // the RAM.
        let ram = unsafe { HostRam::new(&[(at(start), 2 * page)], None) }.unwrap();
        assert_eq!(ram.ranges, vec![start..start + 2 * page]);
        assert!(ram.dirty.is_none());
        // SAFETY: as above; refused before anything is kept.
        let off = unsafe { HostRam::new(&[(at(start + 1), page)], None) };
        assert!(matches!(
            off,
            Err(Error::Invalid(ref invalid)) if invalid.problem == Problem::HostRam
        ));
        // SAFETY: as above.
        let none = unsafe { HostRam::new(&[], None) };
        assert!(none.is_err());
    }

    #[test]
    fn host_ram_drops_its_keep_alive_with_it() {
        let page = host_page().expect("a Unix host's page size");
        let mut pages = vec![0u8; 2 * page as usize];
        let start = (pages.as_mut_ptr() as u64).next_multiple_of(page);
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        struct Flag(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Flag {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let keep: KeepAlive = Box::new(Flag(Arc::clone(&dropped)));
        let at = NonNull::new(start as *mut u8).unwrap();
        // SAFETY: the bytes are a live allocation of this test, never imported.
        let ram = unsafe { HostRam::new(&[(at, page)], Some(keep)) }.unwrap();
        assert!(!dropped.load(std::sync::atomic::Ordering::SeqCst));
        drop(ram);
        assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
    }

    const MIB: u64 = 1 << 20;

    /// One range, as a list.
    fn one(range: Range<u64>) -> Vec<Range<u64>> {
        std::iter::once(range).collect()
    }

    #[test]
    fn chunks_are_64_mib_from_each_range_start_never_across_its_end() {
        let ranges = [0..150 * MIB, 1024 * MIB..1088 * MIB];
        let spans = chunk_spans(&ranges, CHUNK_BYTES);
        assert_eq!(
            spans,
            vec![
                0..64 * MIB,
                64 * MIB..128 * MIB,
                128 * MIB..150 * MIB,
                1024 * MIB..1088 * MIB,
            ]
        );
        assert_eq!(span_of(&spans, 0), Some(0));
        assert_eq!(span_of(&spans, 64 * MIB - 1), Some(0));
        assert_eq!(span_of(&spans, 64 * MIB), Some(1));
        assert_eq!(span_of(&spans, 149 * MIB), Some(2));
        assert_eq!(span_of(&spans, 150 * MIB), None, "between the ranges");
        assert_eq!(span_of(&spans, 1087 * MIB), Some(3));
        assert_eq!(span_of(&spans, 1088 * MIB), None, "past the last");
    }

    #[test]
    fn listed_chunks_are_as_long_as_every_limit_takes_in_whole_64_mib() {
        const GIB: u64 = 1 << 30;
        // RADV: an allocation just under 4 GiB, a buffer as long, storage to 4 GiB less one.
        let radv = listed_chunk_bytes(u64::MAX, 4 * GIB - 4, 4 * GIB - 4, u64::from(u32::MAX));
        assert_eq!(radv, 63 * CHUNK_BYTES);
        // The source's dma-buf is the shortest; a maxBufferSize of 0 states none.
        assert_eq!(listed_chunk_bytes(GIB, 4 * GIB, 0, 4 * GIB), GIB);
        assert_eq!(listed_chunk_bytes(2 * GIB, 4 * GIB, GIB + 1, 4 * GIB), GIB);
        // A limit shorter than a chunk leaves the 64 MiB chunks.
        assert_eq!(
            listed_chunk_bytes(CHUNK_BYTES, 4 * GIB, 0, 4 * GIB),
            CHUNK_BYTES
        );
        assert_eq!(listed_chunk_bytes(MIB, 4 * GIB, 0, 4 * GIB), CHUNK_BYTES);
        assert_eq!(
            listed_chunk_bytes(4 * GIB, 4 * GIB, 0, 128 * MIB - 1),
            CHUNK_BYTES
        );
    }

    #[test]
    fn long_chunks_cut_each_range_from_its_start() {
        let chunk = 63 * CHUNK_BYTES;
        let ranges = [0..8192 * MIB, 9000 * MIB..9100 * MIB];
        assert_eq!(
            chunk_spans(&ranges, chunk),
            vec![
                0..chunk,
                chunk..2 * chunk,
                2 * chunk..8192 * MIB,
                9000 * MIB..9100 * MIB,
            ]
        );
    }

    #[test]
    fn one_run_inside_one_chunk_is_bound_there() {
        let ranges = one(0..150 * MIB);
        let spans = chunk_spans(&ranges, CHUNK_BYTES);
        let run = one(70 * MIB..70 * MIB + 3 * PAGE);
        assert_eq!(
            place_runs(&run, 3 * PAGE, PAGE, &ranges, &spans),
            Ok(Some(1))
        );
        // The last pages of a chunk.
        let last = one(64 * MIB - PAGE..64 * MIB);
        assert_eq!(place_runs(&last, PAGE, PAGE, &ranges, &spans), Ok(Some(0)));
        // A range's last, shorter chunk.
        let tail = one(150 * MIB - PAGE..150 * MIB);
        assert_eq!(place_runs(&tail, PAGE, PAGE, &ranges, &spans), Ok(Some(2)));
    }

    #[test]
    fn several_runs_or_one_crossing_a_chunk_take_the_gates_path() {
        let ranges = one(0..150 * MIB);
        let spans = chunk_spans(&ranges, CHUNK_BYTES);
        let two = [0..PAGE, 8 * PAGE..9 * PAGE];
        assert_eq!(place_runs(&two, 2 * PAGE, PAGE, &ranges, &spans), Ok(None));
        let crossing = one(64 * MIB - PAGE..64 * MIB + PAGE);
        assert_eq!(
            place_runs(&crossing, 2 * PAGE, PAGE, &ranges, &spans),
            Ok(None)
        );
    }

    #[test]
    fn a_take_inside_the_window_or_next_to_the_last_opens_it_in_the_takes_direction() {
        let window = |taken, down, chunks| Window {
            taken,
            down,
            chunks,
        };
        let slots = |window: Window| window.slots().collect::<Vec<_>>();
        assert_eq!(
            Window::START,
            window(None, false, 1),
            "one chunk from the RAM's start"
        );
        assert_eq!(slots(Window::START), [0]);
        // A run up from the start: each take inside the window opens the full window past it,
        // the chunk taken first.
        let run = Window::START.after_take(0);
        assert_eq!(run, window(Some(0), false, AHEAD_CHUNKS));
        assert_eq!(slots(run), (0..=AHEAD_CHUNKS).collect::<Vec<_>>());
        assert_eq!(
            run.after_take(AHEAD_CHUNKS),
            window(Some(AHEAD_CHUNKS), false, AHEAD_CHUNKS)
        );
        // A jump past the window: one chunk past the take, in the jump's direction.
        let jump = run.after_take(AHEAD_CHUNKS + 5);
        assert_eq!(jump, window(Some(AHEAD_CHUNKS + 5), false, 1));
        assert_eq!(slots(jump), [AHEAD_CHUNKS + 5, AHEAD_CHUNKS + 6]);
        // A run goes on from a jump once a take lands in its one chunk.
        assert_eq!(
            jump.after_take(AHEAD_CHUNKS + 6),
            window(Some(AHEAD_CHUNKS + 6), false, AHEAD_CHUNKS)
        );
        // A run down: a take next to the chunk last taken, below it, opens the full window
        // below.
        let high = 3 * AHEAD_CHUNKS;
        let top = jump.after_take(high);
        assert_eq!(top, window(Some(high), false, 1));
        let down = top.after_take(high - 1);
        assert_eq!(down, window(Some(high - 1), true, AHEAD_CHUNKS));
        assert_eq!(
            slots(down),
            (high - 1 - AHEAD_CHUNKS..=high - 1)
                .rev()
                .collect::<Vec<_>>()
        );
        assert_eq!(
            down.after_take(high - 3),
            window(Some(high - 3), true, AHEAD_CHUNKS),
            "a take inside the window below"
        );
        // A jump down: one chunk below the take; next to the last take the other way: the full
        // window that way.
        assert_eq!(down.after_take(1), window(Some(1), true, 1));
        assert_eq!(
            down.after_take(high),
            window(Some(high), false, AHEAD_CHUNKS)
        );
        // The window below ends at the RAM's start.
        assert_eq!(slots(window(Some(1), true, AHEAD_CHUNKS)), [1, 0]);
    }

    #[test]
    fn the_devices_thread_imports_the_window_past_the_last_take_and_no_more() {
        // Three chunks more than the full window, of address space mapped and never touched (no
        // device imports them).
        let full = AHEAD_CHUNKS;
        let length = (full as u64 + 3) * CHUNK_BYTES;
        // SAFETY: a fresh anonymous mapping that reserves no memory until touched.
        let start = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        assert_ne!(start, libc::MAP_FAILED);
        let at = NonNull::new(start.cast::<u8>()).unwrap();
        // SAFETY: the mapping is readable and writable until the end of the test, and nothing
        // imports it.
        let ram = Arc::new(unsafe { HostRam::new(&[(at, length)], None) }.unwrap());
        let on = RamOnDevice::new(&ram, Plan::AHEAD);
        assert_eq!(on.slots.len(), full + 3);
        let done =
            |on: &RamOnDevice, slot: usize| on.slots[slot].tried.store(true, Ordering::Relaxed);
        assert_eq!(on.next_ahead(), Some(0), "one chunk from the start");
        done(&on, 0);
        assert_eq!(on.next_ahead(), None, "none past the start's one chunk");
        // A take inside the window: the full window past it, and none past that.
        assert!(on.take(0));
        assert!(!on.take(0), "the chunk last taken moves nothing");
        assert_eq!(on.next_ahead(), Some(1));
        (1..=full).for_each(|slot| done(&on, slot));
        assert_eq!(on.next_ahead(), None, "none past the window");
        // The run goes on: the window moves past chunk 2.
        assert!(on.take(2));
        assert_eq!(on.next_ahead(), Some(full + 1));
        // A chunk being imported by an object: the thread passes it.
        *lock(&on.slots[full + 1].import) = Import::Making;
        assert_eq!(on.next_ahead(), Some(full + 2));
        *lock(&on.slots[full + 1].import) = Import::None;
        done(&on, full + 1);
        done(&on, full + 2);
        assert_eq!(on.next_ahead(), None, "the RAM's end");
        // A jump: the chunk taken, one chunk past it, and nothing back at the RAM's start, whose
        // chunk 0 a fresh use has not imported.
        let fresh = RamOnDevice::new(&ram, Plan::AHEAD);
        assert!(fresh.take(3));
        assert_eq!(fresh.next_ahead(), Some(3), "the chunk a jump took first");
        done(&fresh, 3);
        assert_eq!(fresh.next_ahead(), Some(4), "one chunk past a jump");
        done(&fresh, 4);
        assert_eq!(
            fresh.next_ahead(),
            None,
            "no more past a jump, nothing back"
        );
        // A take next to it, below: the chunk taken, then the window below it, to the RAM's
        // start.
        assert!(fresh.take(2));
        for slot in [2, 1, 0] {
            assert_eq!(fresh.next_ahead(), Some(slot), "down the RAM");
            done(&fresh, slot);
        }
        assert_eq!(fresh.next_ahead(), None, "nothing below the RAM's start");
        let last = RamOnDevice::new(&ram, Plan::AHEAD);
        assert!(last.take(full + 2));
        assert_eq!(last.next_ahead(), Some(full + 2), "the chunk taken");
        done(&last, full + 2);
        assert_eq!(last.next_ahead(), None, "nothing past the RAM's end");
        drop((on, fresh, last, ram));
        // SAFETY: the test's mapping, which nothing uses any more.
        unsafe { libc::munmap(start, length as usize) };
    }

    #[test]
    fn a_listed_ram_s_thread_imports_the_chunks_asked_for_and_only_those() {
        // Four long chunks of address space mapped and never touched (no device imports them).
        let length = 4 * CHUNK_BYTES;
        // SAFETY: a fresh anonymous mapping that reserves no memory until touched.
        let start = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        assert_ne!(start, libc::MAP_FAILED);
        let at = NonNull::new(start.cast::<u8>()).unwrap();
        // SAFETY: the mapping is readable and writable until the end of the test, and nothing
        // imports it.
        let ram = Arc::new(unsafe { HostRam::new(&[(at, length)], None) }.unwrap());
        let plan = Plan {
            bytes: CHUNK_BYTES,
            listed: true,
        };
        let on = RamOnDevice::new(&ram, plan);
        let ask = |slot: usize| on.slots[slot].asked.store(true, Ordering::Relaxed);
        assert_eq!(on.next_asked(), None, "nothing ahead of an ask");
        // A take moves no window the thread reads: it imports none but those asked for.
        assert!(on.take(0));
        assert_eq!(on.next_asked(), None, "a take asks for nothing");
        ask(3);
        ask(1);
        assert_eq!(on.next_asked(), Some(1), "the first asked for");
        assert_eq!(on.next_asked(), Some(3), "each ask answered once");
        assert_eq!(on.next_asked(), None);
        // A chunk being imported (by an object) or failed by the thread: its ask passed, and
        // answered.
        *lock(&on.slots[2].import) = Import::Making;
        ask(2);
        on.slots[0].tried.store(true, Ordering::Relaxed);
        ask(0);
        assert_eq!(
            on.next_asked(),
            None,
            "neither a chunk under way nor a failed one"
        );
        *lock(&on.slots[2].import) = Import::None;
        assert_eq!(on.next_asked(), None, "those asks were answered");
        drop((on, ram));
        // SAFETY: the test's mapping, which nothing uses any more.
        unsafe { libc::munmap(start, length as usize) };
    }

    #[test]
    fn answers_that_break_the_contract_are_refused() {
        let ranges = one(0..150 * MIB);
        let spans = chunk_spans(&ranges, CHUNK_BYTES);
        let place = |runs: &[Range<u64>], length| place_runs(runs, length, PAGE, &ranges, &spans);
        assert_eq!(place(&[], 0), Err(()), "no run");
        assert_eq!(place(&one(PAGE..PAGE), 0), Err(()), "an empty run");
        assert_eq!(place(&one(1..PAGE + 1), PAGE), Err(()), "off the page");
        assert_eq!(
            place(&one(0..PAGE), 2 * PAGE),
            Err(()),
            "short of the window"
        );
        assert_eq!(place(&one(0..2 * PAGE), PAGE), Err(()), "past the window");
        let outside = one(150 * MIB..150 * MIB + PAGE);
        assert_eq!(place(&outside, PAGE), Err(()), "outside the RAM");
    }
}
