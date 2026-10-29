//! Guest RAM imported whole (a runtime extension: Metal has none): a VMM whose device refuses
//! views of guest pages made through dma-bufs gives the device the guest's RAM once
//! ([`HostRam`], [`Device::use_host_ram`]), and each no-copy object over a view of that RAM says
//! which runs of it its pages are (`HostMemory::with_runs`).
//!
//! - **The RAM** is the consumer's ranges of host memory, each mapped, readable and writable, for
//!   as long as any device uses it ([`HostRam::new`]). A device that uses it imports it by host
//!   pointer (`VK_EXT_external_memory_host`) in chunks of [`CHUNK_BYTES`], never across a range's
//!   end, lazily: each chunk the first time an object over it is made, on the thread making that
//!   object, once (a lock per chunk). The chunks stay imported while the device uses the RAM, are
//!   never released while idle (`host_residency.rs`), and are freed once the use is let go of
//!   ([`HostRamUse`]'s drop) and nothing holds them: no buffer bound in them, no command buffer
//!   that used one. The RAM's keep-alive is dropped after its last chunk is freed.
//! - **One run in place.** A no-copy object whose pages are one run inside one chunk is a
//!   `VkBuffer` bound in that chunk's import (`buffer_descriptor.rs`), made and bound as an import
//!   of the object's own pages would be: its size reaches no further than the object's last page,
//!   so neither its Metal address range, a descriptor's robust range nor a root entry reaches the
//!   rest of the chunk.
//! - **The dirty log** ([`HostRam::with_dirty_pages`]): one log for the whole RAM, its offsets
//!   host addresses, so one stamp per commit serves every copy over the RAM.
//!
//! [`Device::use_host_ram`]: crate::Device::use_host_ram

use super::buffer::usage;
use super::buffer_descriptor::HostBuffer;
use super::device::lock;
use super::dirty_pages::{hook, DirtyPages, DirtySource};
use super::error::{invalid, vulkan, Error, MemoryKind, Problem};
use super::exit_guard;
use super::host_memory::{
    binds_in, host_page, import_refused, not_importable, ImportWindow, Importer, KeepAlive,
    RunResolver,
};
use super::memory::{ImportFailure, Mapped, MemoryClass, Placement};
use super::program_objects::ProgramObject;
use super::Device;
use crate::release;
use ash::vk;
use metal2vulkan::error::{Feature, Unsupported};
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// The bytes of a chunk of the RAM a device imports at once; the last chunk of a range may be
/// fewer.
pub(crate) const CHUNK_BYTES: u64 = 64 << 20;

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
}

impl std::fmt::Debug for HostRam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostRam")
            .field("ranges", &self.ranges.len())
            .field("dirty_pages", &self.dirty.is_some())
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
        let gone = self.device.shared().host_ram.remove(&self.ram);
        // The chunks no object holds are freed here, with no lock of the source held.
        drop(gone);
    }
}

/// A RAM as one device uses it: the RAM, and its chunks, each imported once, the first time an
/// object over it is made.
pub(crate) struct RamOnDevice {
    ram: Arc<HostRam>,
    /// Its chunks, by start ([`chunk_spans`]).
    slots: Vec<ChunkSlot>,
}

/// One chunk of a RAM a device uses: its host bytes, and its import once made, behind a lock of
/// its own, so the first objects over it import it once and objects over other chunks wait for
/// none of it.
struct ChunkSlot {
    span: Range<u64>,
    chunk: Mutex<Option<Arc<RamChunk>>>,
}

/// The chunks of `ranges` (host addresses, by start): [`CHUNK_BYTES`] from each range's start,
/// the last of each range ending at its end, so that no chunk crosses a range's end.
fn chunk_spans(ranges: &[Range<u64>]) -> Vec<Range<u64>> {
    let mut spans = Vec::new();
    for range in ranges {
        let mut start = range.start;
        while start < range.end {
            let end = range.end.min(start.saturating_add(CHUNK_BYTES));
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

    /// `ram`, none of its chunks imported.
    fn new(ram: &Arc<HostRam>) -> RamOnDevice {
        let slots = (chunk_spans(&ram.ranges).into_iter())
            .map(|span| ChunkSlot {
                span,
                chunk: Mutex::new(None),
            })
            .collect();
        RamOnDevice {
            ram: Arc::clone(ram),
            slots,
        }
    }

    /// The chunks' spans.
    fn spans(&self) -> Vec<Range<u64>> {
        self.slots.iter().map(|slot| slot.span.clone()).collect()
    }

    /// The import of chunk `slot` on `device`: kept, or made now, once, under the chunk's own
    /// lock (`Device::import_chunk`). A failed import is not kept: the next object over the chunk
    /// tries again.
    fn chunk(
        &self,
        device: &Device,
        importer: &Importer,
        slot: usize,
    ) -> Result<Arc<RamChunk>, Error> {
        let slot = &self.slots[slot];
        let mut held = lock(&slot.chunk);
        if let Some(chunk) = held.as_ref() {
            return Ok(Arc::clone(chunk));
        }
        // SAFETY: the span is pages of the RAM's ranges, which stay mapped, readable and
        // writable, while a chunk holds the RAM (`HostRam::new`).
        let chunk =
            Arc::new(unsafe { device.import_chunk(importer, &self.ram, slot.span.clone()) }?);
        *held = Some(Arc::clone(&chunk));
        Ok(chunk)
    }
}

/// A device's import of one chunk of a RAM (`VK_EXT_external_memory_host`): the memory, and a
/// `VkBuffer` over the whole chunk, for the runtime's own copies between the chunk and memory of
/// the device's own alone; no Metal buffer, root entry or binding ever names it. Buffers bound
/// in the chunk ([`Device::bind_in_chunk`]) hold it through their placement (`memory.rs`), and so
/// do the command buffers that use them, until they completed; the device holds it while it uses
/// the RAM ([`HostRamUse`]). The last holder destroys the `VkBuffer` and frees the memory, and the
/// RAM, which the chunk holds, goes after the free: its keep-alive with it once no chunk holds it.
/// Never released while idle (`host_residency.rs`). Left to the process's end while it exits.
pub(crate) struct RamChunk {
    device: Device,
    /// The host bytes imported.
    span: Range<u64>,
    /// The `VkBuffer` over the whole chunk.
    raw: vk::Buffer,
    /// The import's placement: a dedicated block of the device's memory (`Memory::import_host`).
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

    /// The index of the memory's type.
    pub(crate) fn type_index(&self) -> u32 {
        self.placement.type_index()
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
        // (`freed_memory.rs`), and the RAM with it.
        let after = lock(&shared.memory).release_then(shared.vk(), &self.placement, after);
        shared.counts.host_ram_chunk_freed();
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
}

impl RamSource {
    /// Add `ram`, unless one of its ranges meets a range of a RAM the device uses.
    fn add(&self, ram: &Arc<HostRam>) -> Option<Arc<RamOnDevice>> {
        let mut rams = lock(&self.rams);
        if rams.iter().any(|used| used.ram.meets(ram)) {
            return None;
        }
        let used = Arc::new(RamOnDevice::new(ram));
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
}

impl Device {
    /// Use the guest RAM `ram` (a runtime extension: Metal has none): from now on, a no-copy
    /// buffer or heap made on this device over pages its caller resolves to one run of the RAM
    /// (`HostMemory::with_runs`) is bound in the device's import of the chunk of the RAM that run
    /// lies in, imported as the first object over it is made, in place of an import of its own
    /// (the module docs of `host_ram.rs`). For a device that refuses views of the pages made
    /// through dma-bufs, so that each object's own import would take a kernel mapping per page.
    /// The use lasts until the returned [`HostRamUse`] is dropped; a device may use several RAMs
    /// at once whose ranges are apart, and a new RAM once an earlier use of the same ranges ended
    /// (a VMM's next guest).
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
        let used = (shared.host_ram.add(ram)).ok_or_else(|| invalid(CALL, Problem::HostRam))?;
        Ok(HostRamUse {
            device: self.clone(),
            ram: used,
        })
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
    /// the chunk its pages' one run lies in (`routed`, [`Routed::InChunk`]), the chunk imported
    /// first where no object over it was made yet: made as an import of the window's own pages
    /// would make it (`Device::host_buffer`) and bound where its bytes lie in the chunk, with
    /// `binds_in`'s checks against the window, so its size reaches no page past the window's.
    /// The placement holds the chunk; its host address is the window's, so the buffer's
    /// `contents` stays the caller's pointer. `None` where the chunk cannot be imported or the
    /// buffer cannot be bound there exactly (its alignment, its memory types): the object then
    /// takes the runtime's other paths.
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
        let chunk = ram.chunk(self, importer, *slot).ok()?;
        let shared = self.shared();
        let raw_device = shared.vk();
        let buffer = self.host_buffer(window, length).ok()?;
        let requirements = buffer.requirements;
        // Where the window starts in the chunk, and the buffer in the window.
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
        shared.counts.chunk_bound_buffer_made();
        let placement = Placement::ram_chunk(
            Arc::clone(&chunk),
            chunk.placement.offset + in_chunk,
            window.size,
            Mapped::of_caller(mapped),
        );
        Some((buffer, placement))
    }

    /// Import `span`, host pages of `ram`, as a chunk: a dedicated block of memory of a coherent
    /// type both the pages and a host-allocation buffer allow, with the device-address flag where
    /// the device enables `bufferDeviceAddress` (`Memory::import_host`), and a `VkBuffer` over the
    /// whole of it. Refused where the device's import alignment misses the span's start or end,
    /// it passes the device's allocation size, or the driver refuses the pages (or the runtime
    /// does for it, `Device::refuse_unpinnable`).
    ///
    /// # Safety
    ///
    /// `span` is pages of `ram`'s ranges, which stay mapped while the chunk holds `ram`.
    unsafe fn import_chunk(
        &self,
        importer: &Importer,
        ram: &Arc<HostRam>,
        span: Range<u64>,
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
            lock(&shared.memory).import_host(
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
        shared.counts.host_ram_chunk_imported(size);
        Ok(RamChunk {
            device: self.clone(),
            span,
            raw,
            placement,
            ram: Some(Arc::clone(ram)),
        })
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
        let spans = chunk_spans(&ranges);
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
    fn one_run_inside_one_chunk_is_bound_there() {
        let ranges = one(0..150 * MIB);
        let spans = chunk_spans(&ranges);
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
        let spans = chunk_spans(&ranges);
        let two = [0..PAGE, 8 * PAGE..9 * PAGE];
        assert_eq!(place_runs(&two, 2 * PAGE, PAGE, &ranges, &spans), Ok(None));
        let crossing = one(64 * MIB - PAGE..64 * MIB + PAGE);
        assert_eq!(
            place_runs(&crossing, 2 * PAGE, PAGE, &ranges, &spans),
            Ok(None)
        );
    }

    #[test]
    fn answers_that_break_the_contract_are_refused() {
        let ranges = one(0..150 * MIB);
        let spans = chunk_spans(&ranges);
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
