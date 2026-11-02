//! The device's Metal address space.
//!
//! Vulkan cannot place a buffer at an address the caller picks, and the translated shaders do
//! not need it to: a pointer value is the Metal address the binder was told (the resident table).
//! So each device hands every buffer a range of its own 64-bit Metal address space, and the program
//! sees those values everywhere: `gpu_address`, pointers a kernel derives, compares, stores, and
//! the CPU reads back. Vulkan device addresses are never shown.
//!
//! - **The space** is `[2^32, 2^47)`: never 0 (the null pointer), above every 32-bit value, and
//!   past every address Metal pinned (on an M3 Max and an M4 every pin at 2^42 or above failed).
//!   Ranges the runtime picks are aligned and sized to Metal's 16-KiB page, as Metal places
//!   buffers on Apple silicon.
//! - **Pinned placement** (`setPinnedGPUAddress:`, a runtime extension; measured on an M3 Max and
//!   an M4, and by the consumer's M3 Max probes): a buffer or heap made at a caller's address
//!   has exactly that address, at any byte of a 32-KiB granule's first page, and holds Metal's span
//!   of it: the pages its bytes touch and one guard page after them, rounded up to 32 KiB (2 pages
//!   for 1, 4 for 2 or 3, 6 for 4 or 5, 66 for 64 or 65), so the page before a pinned range cannot
//!   start another pin either. A pin whose first page is not a 32-KiB granule's (`OffGranule`),
//!   whose span meets bytes another resource holds, or that leaves the space, is refused
//!   ([`PlacementConflict`]); Metal answers nil for each and aliases nothing. A released pin frees
//!   its whole span, as Metal's does. Metal's own process layout refused some ranges inside the
//!   space (below 2^36, 2^39, just below 2^40, 2^42 and above); that is no contract, and the
//!   runtime pins any free range of the space.
//! - **A shared range** (the consumer's ask C3): a no-copy buffer made over the same first byte of
//!   host memory as a live no-copy buffer, no longer than it (`BufferDescriptor::range_of`),
//!   answers that buffer's address and holds its range with it, as Metal answers one IOSurface's
//!   address for every buffer over the surface (the consumer's probe on an M3 Max under macOS
//!   26.7). The range is free once every buffer holding it is gone; each one's range token drops
//!   with it.
//! - **Pools and their slots** (the consumer's ask "sub-page placement", its `vaprobe smallpool`
//!   and `texpool2` probes on an M3 Max under macOS 26.7). Metal makes each buffer of its own
//!   memory shorter than a page in a pool of 0x20000 bytes, a plain range of the space, at a
//!   256-byte-aligned slot of `round_up(length, 256)` bytes, so buffers share pages. A caller that
//!   models those pools reserves each one (`Device::reserve_address_pool`, a runtime extension):
//!   the pool holds the span a pin of its length holds (its pages, one guard page, rounded up to 32
//!   KiB), so no other range, pin or pick lands in it while it lives. A buffer placed in a slot of
//!   it (`BufferDescriptor::with_pool_slot`) has the pool's address plus the slot's offset, and
//!   holds only `round_up(length, 256)` bytes of the pool, with no guard and no page rounding; a
//!   slot whose offset is not a multiple of 256 bytes (`OffSlot`), that passes the pool's end
//!   (`OutsidePool`), or that meets a live slot (`Held`), is refused. The slot is free once its
//!   buffer is gone (its range token drops then). A dropped pool's span stays held until the last
//!   buffer in its slots is gone. Which slot a buffer takes, and when a pool is needed or goes, is
//!   the caller's to model, as are Metal's own allocations in its pools.
//! - **Outside the space**: the runtime's own buffers (the index buffers it generates, staging for
//!   fills, uploads and copies, scratch, tile memory, shadows, and the buffers of the device's
//!   in-place check, `in_place_check.rs`, which runs on a thread of its own while the program makes
//!   its resources) take ranges of a second space above it, `[2^47, 2^48)`, so a program's buffer
//!   addresses never depend on whether or when the runtime's own work runs or completes, as
//!   Metal's own allocations never move a program's, and no pin of the program's meets them. So
//!   does the storage of a program's object whose address Metal shows no program: a texture
//!   buffer's texels the device makes (`newTextureWithDescriptor:`,
//!   `newSharedTextureWithDescriptor:`) and a visible function table's words, each reached
//!   through its binding or its `gpuResourceID`, never at an address of the space, as a texture
//!   of its own memory takes none of it. The space holds only what Metal gives an address (a
//!   buffer's `gpuAddress`; a heap's range, which its resources lie in) and the pools a caller
//!   reserves.
//!   [`AddressPlacement::Outside`], a runtime extension, puts a caller's helper buffer there too. A
//!   pin never reaches it.
//! - **Indirect command buffers' records**: an indirect command buffer's ID is the one Metal's pool
//!   of their own space hands out, from 1, as Metal's `gpuResourceID` is (an M3 Max and an M4; the
//!   pool's rule, first 32 IDs, is `resource_ids.rs`'), or the one it is placed at, and its records
//!   start at `records_address` of it, `2^63 + id * 2^31`, the address a translated kernel
//!   encoding through the ID reaches: records of more than 2^31 bytes run past later IDs' starts,
//!   which the kernel's search tells apart, as it finds records only at their start. So the
//!   records take no Metal address, as Metal's indirect command buffers take none of its buffers'
//!   (four of 1024 commands moved no buffer made after them).
//! - **Records IDs** (the M4 probe, plain Metal). The buffer holds its ID
//!   ([`RecordsId`]), not its records' memory: the ID is free again once the buffer and every
//!   command that holds it are gone, a compute or render execution and a `useResource` until their
//!   command buffer completes or is dropped uncommitted, while a blit's copy, reset or optimize
//!   holds only the records' memory (Metal handed a blit's buffer's ID out again while the blit
//!   waited). A placed buffer's ID is not the pool's: plain allocation neither skips it nor gets it
//!   back, but it counts as an allocation for a later reservation. Each ID has one slot, the buffer
//!   last made with it, placed or plain, which a kernel encoding through the ID reaches; once that
//!   buffer is gone the ID reaches none, even while an earlier buffer with it lives
//!   ([`AddressSpace::holds_records_slot`]).

use super::device::lock;
use super::error::{internal, Error, MemoryKind};
use super::exit_guard;
use super::program_objects::ProgramObject;
use super::range::RangeAllocator;
use super::resource_ids::Pool;
use metal2vulkan::bind::indirect_commands::{records_address, RECORDS_ID_LIMIT};
use std::collections::BTreeMap;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// A Metal GPU address: what `gpuAddress` answers and what a kernel's pointers hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuAddress(pub u64);

/// The lowest address the space hands out.
pub(super) const SPACE_BASE: u64 = 1 << 32;
/// The end of the space.
pub(super) const SPACE_END: u64 = 1 << 47;
/// The end of the space outside the Metal one, `[SPACE_END, OUTSIDE_END)` (module docs).
pub(super) const OUTSIDE_END: u64 = 1 << 48;
/// The alignment and size granule of every range: Metal's page on Apple silicon.
pub(super) const PAGE: u64 = 16 << 10;
/// The granule of a pinned range: its first page starts one, and its span is a multiple of it.
pub(super) const GRANULE: u64 = 2 * PAGE;
/// The alignment and size granule of a pool's slot: Metal's for a buffer shorter than a page.
pub(super) const SLOT: u64 = 0x100;

/// Where a buffer or heap goes in the Metal address space: where the runtime picks, or at the
/// caller's address (`setPinnedGPUAddress:`, a runtime extension), or outside the space (a
/// runtime extension, no Metal counterpart).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AddressPlacement {
    /// A range the runtime picks.
    #[default]
    Any,
    /// At this address, which the resource's `gpu_address` then answers.
    At(GpuAddress),
    /// A range the runtime picks outside the Metal address space, `[2^47, 2^48)`, where the
    /// runtime's own buffers lie: for a caller's helper buffer that must take no Metal address,
    /// so no Metal buffer's address depends on it. Its `gpu_address` answers that range's
    /// address, which no Metal resource shares and no pin reaches. A runtime extension: Metal has
    /// no such placement.
    Outside,
}

/// Where the runtime places a buffer it makes: as a placement says, or as the records of the
/// indirect command buffer that holds records ID `id`, at `records_address` of it (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Place {
    At(AddressPlacement),
    Records(u64),
    /// At `address`, sharing `held`, a live buffer's range (module docs).
    Shared {
        address: GpuAddress,
        held: Held,
    },
    /// At `offset` in the pool reserved at `pool`, the descriptor's placement being `placement`
    /// (module docs).
    InPool {
        pool: u64,
        offset: u64,
        placement: AddressPlacement,
    },
}

impl From<AddressPlacement> for Place {
    fn from(placement: AddressPlacement) -> Place {
        Place::At(placement)
    }
}

/// A caller's token for a resource's range of the Metal address space (a runtime extension, no
/// Metal counterpart): `BufferDescriptor::range_token`, `Device::new_heap_with_range_token`,
/// `Device::new_heap_with_bytes_no_copy_and_range_token`, and
/// `Device::new_texture_with_range_token` and `Device::new_shared_texture_with_range_token`. The
/// runtime drops it right after it frees the resource, at or before the point Metal frees a
/// released resource's range. Metal never frees that range inside a completed handler of the last
/// command buffer that held it, which is a guarantee: 4,800 of 4,800 ICBs and 4,800 of 4,800
/// buffers held their ID and `gpuAddress` there (the consumer's probe on an Apple M3 Max, macOS
/// 27.0.1). After those handlers it USUALLY frees before that command buffer's waiters wake, but
/// that is its usual order and not a rule: 10 of 8,400 ICBs were still held under load, coming back
/// up to 884 us later, against 0 of 6,000 unloaded, and a dispatch-bound buffer's `gpuAddress` was
/// still held in 16 of 3,000 unloaded and 19 of 8,400 loaded runs, up to 3.9 ms after the wait. So
/// the runtime's own order -- status, the completed handlers, the releases, then the gate that
/// wakes the waiters (`CommandBufferShared::complete`) -- is the deterministic side of a race Metal
/// leaves open, and is within Metal's behaviour rather than a copy of a rule of Metal's. A caller
/// that must see the range free waits for the handlers, which is what Metal guarantees, and not for
/// a waiter to wake. It drops with no runtime lock held, so a
/// creation at the range from its drop on succeeds; on the thread that drops the resource's last
/// holder (the program's, or a runtime thread that runs a command buffer's completed handlers, as
/// the program's code there, a panic in it caught, as a no-copy buffer's deallocator), an
/// uncommitted command buffer's holds included; before a refused creation returns; and never where
/// the runtime leaves the resource to the process's exit (leaked with it). A heap's token goes with
/// the heap's own bytes, once the heap and every resource placed in it are gone; a sparse heap's
/// once the heap, its textures and the command buffers that hold them are gone. A texture's goes as
/// Metal frees the texture: once the texture, every view and texture opened from it, and the
/// command buffers that hold any of them are gone, whatever the runtime keeps of its image after
/// that (for a command buffer committed before it went that may reach it by its resource ID, or the
/// batch that took its first move), as Metal frees a released texture's range while a command
/// buffer that only names it in an argument buffer still runs (an Apple M4).
pub type RangeToken = Box<dyn std::any::Any + Send + Sync>;

/// Where a resource keeps its caller's [`RangeToken`], held as the program's object
/// (`program_objects.rs`): set once as the resource is made ([`keep`](Self::keep)), and dropped by [`release`](Self::release), which the resource's
/// teardown calls once its range is free. Never dropped otherwise: a resource the runtime leaves
/// to the process's exit leaks its token with it (`ManuallyDrop`).
#[derive(Default)]
pub(super) struct KeptToken(ManuallyDrop<Mutex<Option<ProgramObject<RangeToken>>>>);

impl KeptToken {
    /// Keep `token`. Internal where one is kept already: no creator gives two.
    pub(super) fn keep(&self, token: RangeToken) -> Result<(), Error> {
        let mut kept = lock(&self.0);
        if kept.is_some() {
            return Err(internal("a resource's range token kept twice"));
        }
        *kept = Some(ProgramObject::new(token));
        Ok(())
    }

    /// Drop the token kept, if any: the resource's range is free, outside its teardown's exit
    /// section and with no runtime lock held, so a creation at the range from its drop on
    /// succeeds. Its drop is the program's code, which a runtime thread (a command buffer's
    /// completion) runs as such, a panic in it caught, as a deallocator (`exit_guard::program`).
    /// Left to the process's exit (leaked) once the process is exiting, as the resource is.
    pub(super) fn release(&self) {
        if exit_guard::closed() {
            return;
        }
        let token = lock(&self.0).take();
        if let Some(token) = token {
            exit_guard::program_release(move || drop(token));
        }
    }
}

/// A texture's [`RangeToken`], kept for the texture objects over its image or texels (the
/// texture, its views, the textures opened from its handles): dropped as the last of them goes,
/// as Metal frees the texture, whatever the runtime keeps of the image or texels after that (a
/// gone texture's image retired for the command buffers that may reach it by its resource ID,
/// the batch that took its first move). A command buffer holds the texture objects it uses, so
/// a texture it uses goes with it.
#[derive(Default)]
pub(super) struct ObjectsToken {
    objects: AtomicUsize,
    token: KeptToken,
}

impl ObjectsToken {
    /// A texture object over the image or texels was made.
    pub(super) fn object_made(&self) {
        self.objects.fetch_add(1, Ordering::Relaxed);
    }

    /// Keep `token`, as [`KeptToken::keep`].
    pub(super) fn keep(&self, token: RangeToken) -> Result<(), Error> {
        self.token.keep(token)
    }

    /// A texture object over the image or texels went, its teardown done and no lock held: the
    /// token drops with the last ([`KeptToken::release`]). Whether it was the last (a heap
    /// texture's range then goes back to its heap).
    pub(super) fn object_gone(&self) -> bool {
        let last = self.objects.fetch_sub(1, Ordering::AcqRel) == 1;
        if last {
            self.token.release();
        }
        last
    }
}

/// Bytes of the Metal address space. No Metal counterpart: a placement refusal's range.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AddressRange {
    /// The first byte.
    pub start: GpuAddress,
    /// The bytes.
    pub length: u64,
}

impl AddressRange {
    pub(super) fn of(start: u64, length: u64) -> AddressRange {
        AddressRange {
            start: GpuAddress(start),
            length,
        }
    }
}

/// A pinned address the space cannot honour (`Error::PlacementConflict`). `requested` is the
/// range the pin needs: the pages its bytes touch and the guard page after them, rounded up to
/// 32 KiB (module docs); for a buffer in a pool's slot, the slot's bytes.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlacementConflict {
    /// The range meets bytes another resource holds; `held` is the first run of held bytes it
    /// meets (of one resource or of several beside each other).
    Held {
        /// The range the pin needs.
        requested: AddressRange,
        /// The first run of held bytes it meets.
        held: AddressRange,
    },
    /// The range leaves the space, `[2^32, 2^47)`.
    OutsideSpace {
        /// The range the pin needs.
        requested: AddressRange,
    },
    /// The pin's first page is not the first of a 32-KiB granule: Metal pins none there (an
    /// Apple M4 and an M3 Max answered nil for every such pin).
    OffGranule {
        /// The range the pin needs.
        requested: AddressRange,
    },
    /// A buffer made to share another's range (`BufferDescriptor::range_of`) that cannot share
    /// it: it is not a no-copy buffer over the same first byte of host memory as the other, the
    /// other is not a no-copy buffer of its own range, it is longer than the other, or its
    /// placement names another address. A runtime extension: Metal has no such call.
    Unshared {
        /// The other buffer's bytes.
        held: AddressRange,
    },
    /// A buffer made in a pool's slot (`BufferDescriptor::with_pool_slot`) whose offset in the
    /// pool is not a multiple of 256 bytes, where Metal places no buffer in its pools, or whose
    /// descriptor's placement names another address. A runtime extension: Metal has no such call.
    OffSlot {
        /// The bytes the slot needs: from the pool's address plus the offset, the buffer's length
        /// rounded up to 256 bytes.
        requested: AddressRange,
        /// The pool's bytes.
        pool: AddressRange,
    },
    /// A buffer made in a pool's slot whose bytes pass the pool's end. A runtime extension.
    OutsidePool {
        /// The bytes the slot needs.
        requested: AddressRange,
        /// The pool's bytes.
        pool: AddressRange,
    },
}

/// Bytes of the space a resource holds until it is gone ([`AddressSpace::free`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Held {
    start: u64,
    size: u64,
    /// The address of the pool whose slot the bytes are; `None` for a range of the space.
    pool: Option<u64>,
}

impl Held {
    /// A range of the space (or of the space outside it), no pool's slot.
    fn range(start: u64, size: u64) -> Held {
        Held {
            start,
            size,
            pool: None,
        }
    }
}

/// One device's Metal address space, the space outside it, and the indirect command buffers'
/// records IDs (module docs).
#[derive(Debug)]
pub(super) struct AddressSpace {
    ranges: RangeAllocator,
    outside: RangeAllocator,
    records: RecordIds,
    /// Each shared range's holders past its first, by its start: [`free`](Self::free) frees the
    /// range only once none is left.
    shares: BTreeMap<u64, usize>,
    /// Each reserved pool, by its address (module docs).
    pools: BTreeMap<u64, SlotPool>,
}

/// A reserved pool (module docs): the span it holds of the space, its length, its slots, and
/// whether its caller let it go, its span then free once no slot is held.
#[derive(Debug)]
struct SlotPool {
    span: Held,
    length: u64,
    slots: RangeAllocator,
    released: bool,
}

/// The records IDs (module docs): Metal's pool of indirect command buffer IDs, its first 32 IDs
/// (`resource_ids.rs`), and each held ID's slot.
#[derive(Debug)]
struct RecordIds {
    /// The plain IDs.
    pool: Pool,
    /// Each held ID's slot: the serial of the buffer last made with it.
    slots: BTreeMap<u64, u64>,
    /// The serial of the buffer made last.
    serial: u64,
}

/// An indirect command buffer's hold on its records ID (module docs): the ID, the buffer's
/// serial, which orders the buffers made with one ID, and whether plain allocation handed it out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct RecordsId {
    /// The ID: what `gpuResourceID` answers.
    pub(super) id: u64,
    serial: u64,
    plain: bool,
}

impl RecordIds {
    /// A new buffer's ID, now held and its slot's: `placed`, or the one Metal hands out next;
    /// `None` when that is [`RECORDS_ID_LIMIT`] or more.
    fn take(&mut self, placed: Option<u64>) -> Option<RecordsId> {
        let (id, plain) = match placed {
            Some(id) => {
                self.pool.start();
                (id, false)
            }
            None if self.pool.next() >= RECORDS_ID_LIMIT => return None,
            None => (self.pool.take(), true),
        };
        self.serial += 1;
        self.slots.insert(id, self.serial);
        Some(RecordsId {
            id,
            serial: self.serial,
            plain,
        })
    }

    /// `held` gone: its slot empty where it was the slot's, and a plain ID free again.
    fn give(&mut self, held: RecordsId) {
        if self.slots.get(&held.id) == Some(&held.serial) {
            self.slots.remove(&held.id);
        }
        if held.plain {
            self.pool.give(held.id);
        }
    }
}

impl AddressSpace {
    /// The whole space free, the space outside it, and every records ID.
    pub(super) fn new() -> AddressSpace {
        AddressSpace {
            ranges: RangeAllocator::new(SPACE_BASE, SPACE_END),
            outside: RangeAllocator::new(SPACE_END, OUTSIDE_END),
            records: RecordIds {
                pool: Pool::new(32),
                slots: BTreeMap::new(),
                serial: 0,
            },
            shares: BTreeMap::new(),
            pools: BTreeMap::new(),
        }
    }

    /// The bytes a resource of `length` bytes holds of the space: whole pages, at least one.
    pub(super) fn span(length: u64) -> Option<u64> {
        length.max(1).checked_next_multiple_of(PAGE)
    }

    /// A free range of the runtime's choice for a resource of `length` bytes, now held: its
    /// start, and what to free; `None` when the space is full.
    pub(super) fn allocate(&mut self, length: u64) -> Option<(GpuAddress, Held)> {
        let size = Self::span(length)?;
        let start = self.ranges.allocate(size, PAGE)?;
        Some((GpuAddress(start), Held::range(start, size)))
    }

    /// [`AddressSpace::allocate`] outside the Metal space (module docs).
    pub(super) fn allocate_outside(&mut self, length: u64) -> Option<(GpuAddress, Held)> {
        let size = Self::span(length)?;
        let start = self.outside.allocate(size, PAGE)?;
        Some((GpuAddress(start), Held::range(start, size)))
    }

    /// The range a resource of `length` bytes pinned at `address` needs, now held: the pages
    /// its bytes touch and the guard page after them, rounded up to 32 KiB (module docs).
    pub(super) fn pin(
        &mut self,
        address: GpuAddress,
        length: u64,
    ) -> Result<Held, PlacementConflict> {
        let start = address.0 - address.0 % PAGE;
        let requested = address
            .0
            .checked_add(length.max(1))
            .and_then(|end| end.checked_next_multiple_of(PAGE))
            .and_then(|end| span_end(start, end))
            .map(|end| AddressRange::of(start, end - start));
        let Some(requested) = requested else {
            return Err(PlacementConflict::OutsideSpace {
                requested: AddressRange::of(start, u64::MAX - start),
            });
        };
        if start < SPACE_BASE || requested.length > SPACE_END.saturating_sub(start) {
            return Err(PlacementConflict::OutsideSpace { requested });
        }
        if !start.is_multiple_of(GRANULE) {
            return Err(PlacementConflict::OffGranule { requested });
        }
        self.ranges
            .allocate_at(start, requested.length)
            .map_err(|(from, length)| PlacementConflict::Held {
                requested,
                held: AddressRange::of(from, length),
            })?;
        Ok(Held::range(start, requested.length))
    }

    /// Keep indirect command buffer IDs `ids` from the records' pool for good (`Pool::reserve`,
    /// the IDs below [`RECORDS_ID_LIMIT`]): false, reserving nothing, once the pool handed out
    /// an ID or took a reservation.
    pub(super) fn reserve_records(&mut self, ids: &[u64]) -> bool {
        self.records.pool.reserve(ids.iter().copied())
    }

    /// Whether the records' pool handed out an ID or took a reservation: a reservation is
    /// ignored then (`Pool::started`).
    pub(super) fn records_started(&self) -> bool {
        self.records.pool.started()
    }

    /// A new indirect command buffer's records ID, now held and its slot's (module docs):
    /// `placed`, or the one Metal hands out next; `None` when that is [`RECORDS_ID_LIMIT`] or
    /// more. The buffer gives it back with [`give_records_id`](Self::give_records_id).
    pub(super) fn take_records_id(&mut self, placed: Option<u64>) -> Option<RecordsId> {
        self.records.take(placed)
    }

    /// `held`'s buffer gone: its slot empty where it was the slot's, and a plain ID free again.
    pub(super) fn give_records_id(&mut self, held: RecordsId) {
        self.records.give(held)
    }

    /// Whether `held`'s buffer holds its ID's slot: what a kernel encoding through the ID
    /// reaches (module docs).
    pub(super) fn holds_records_slot(&self, held: RecordsId) -> bool {
        self.records.slots.get(&held.id) == Some(&held.serial)
    }

    /// The records of an indirect command buffer of `length` bytes holding records ID `id`: from
    /// `records_address` of it, of any length that ends below 2^64 (2^62 bytes past the last
    /// ID's start). `None` for an ID of [`RECORDS_ID_LIMIT`] or more, or a longer length.
    fn records(id: u64, length: u64) -> Option<(GpuAddress, Held)> {
        let size = Self::span(length)?;
        if id >= RECORDS_ID_LIMIT || records_address(id).checked_add(size).is_none() {
            return None;
        }
        let start = records_address(id);
        Some((GpuAddress(start), Held::range(start, size)))
    }

    /// The address and held bytes of a resource of `length` bytes at `place`: refused when the
    /// space is full (`OutOfMemory`) or the pin cannot be honoured (`PlacementConflict`).
    pub(super) fn place(
        &mut self,
        place: impl Into<Place>,
        length: u64,
    ) -> Result<(GpuAddress, Held), Error> {
        let placement = match place.into() {
            Place::At(placement) => placement,
            Place::Records(id) => {
                return Self::records(id, length).ok_or(Error::OutOfMemory(MemoryKind::Device))
            }
            // The range's holder lives (its caller holds it), so the range is held still.
            Place::Shared { address, held } => {
                *self.shares.entry(held.start).or_default() += 1;
                return Ok((address, held));
            }
            Place::InPool {
                pool,
                offset,
                placement,
            } => return self.slot(pool, offset, placement, length),
        };
        match placement {
            AddressPlacement::Any => self
                .allocate(length)
                .ok_or(Error::OutOfMemory(MemoryKind::Device)),
            AddressPlacement::At(address) => self
                .pin(address, length)
                .map(|held| (address, held))
                .map_err(Error::PlacementConflict),
            AddressPlacement::Outside => self
                .allocate_outside(length)
                .ok_or(Error::OutOfMemory(MemoryKind::Device)),
        }
    }

    /// Give back what [`AddressSpace::allocate`], [`AddressSpace::allocate_outside`] or
    /// [`AddressSpace::pin`] held. Records hold nothing of the spaces: their buffer gives its
    /// ID back ([`give_records_id`](Self::give_records_id)).
    pub(super) fn free(&mut self, held: Held) {
        // A shared range stays while another buffer holds it.
        if let Some(others) = self.shares.get_mut(&held.start) {
            *others -= 1;
            if *others == 0 {
                self.shares.remove(&held.start);
            }
            return;
        }
        // A slot goes back to its pool, and a released pool's span with its last slot.
        if let Some(address) = held.pool {
            if let Some(pool) = self.pools.get_mut(&address) {
                pool.slots.free(held.start, held.size);
                if pool.released && pool.slots.used() == 0 {
                    self.drop_pool(address);
                }
            }
            return;
        }
        self.give_back_range(held);
    }

    /// Give back a range of the space or of the space outside it.
    fn give_back_range(&mut self, held: Held) {
        if held.start < SPACE_END {
            self.ranges.free(held.start, held.size);
        } else if held.start < OUTSIDE_END {
            self.outside.free(held.start, held.size);
        }
    }

    /// Reserve a pool of `length` bytes at `placement` (module docs): its address. It holds the
    /// span a pin of `length` bytes holds, at the address asked (refused as [`pin`](Self::pin)
    /// refuses it), or at a 32-KiB granule the runtime picks in the space or outside it
    /// (`OutOfMemory` when none is free). Its caller lets it go with
    /// [`release_pool`](Self::release_pool).
    pub(super) fn reserve_pool(
        &mut self,
        placement: AddressPlacement,
        length: u64,
    ) -> Result<GpuAddress, Error> {
        let picked = |space: &mut RangeAllocator| {
            let pages = length.max(1).checked_next_multiple_of(PAGE)?;
            let size = span_end(0, pages)?;
            space
                .allocate(size, GRANULE)
                .map(|start| (GpuAddress(start), Held::range(start, size)))
        };
        let (address, span) = match placement {
            AddressPlacement::Any => picked(&mut self.ranges),
            AddressPlacement::Outside => picked(&mut self.outside),
            AddressPlacement::At(address) => {
                let span = self
                    .pin(address, length)
                    .map_err(Error::PlacementConflict)?;
                Some((address, span))
            }
        }
        .ok_or(Error::OutOfMemory(MemoryKind::Device))?;
        // The span holds the pool's bytes: `pin` and `picked` hold at least `length` bytes from
        // the address.
        let slots = RangeAllocator::new(address.0, address.0 + length);
        let pool = SlotPool {
            span,
            length,
            slots,
            released: false,
        };
        self.pools.insert(address.0, pool);
        Ok(address)
    }

    /// The caller let the pool reserved at `address` go: its span is free now where no slot of it
    /// is held, else once the last is.
    pub(super) fn release_pool(&mut self, address: GpuAddress) {
        let Some(pool) = self.pools.get_mut(&address.0) else {
            return;
        };
        pool.released = true;
        if pool.slots.used() == 0 {
            self.drop_pool(address.0);
        }
    }

    /// The pool at `address` gone, its span free.
    fn drop_pool(&mut self, address: u64) {
        if let Some(pool) = self.pools.remove(&address) {
            self.give_back_range(pool.span);
        }
    }

    /// The slot of a buffer of `length` bytes at `offset` in the pool reserved at `pool`, now
    /// held, the descriptor's placement being `placement` (module docs): refused `OffSlot`,
    /// `OutsidePool` or `Held` (the first run of held slot bytes it meets).
    fn slot(
        &mut self,
        pool: u64,
        offset: u64,
        placement: AddressPlacement,
        length: u64,
    ) -> Result<(GpuAddress, Held), Error> {
        let Some(slots) = self.pools.get_mut(&pool) else {
            return Err(internal("a slot of a pool not reserved"));
        };
        let pool_range = AddressRange::of(pool, slots.length);
        let start = pool.saturating_add(offset);
        let size = length.max(1).div_ceil(SLOT).saturating_mul(SLOT);
        let requested = AddressRange::of(start, size);
        let named = match placement {
            AddressPlacement::Any => true,
            AddressPlacement::At(address) => address.0 == start,
            AddressPlacement::Outside => false,
        };
        let conflict = if !offset.is_multiple_of(SLOT) || !named {
            Some(PlacementConflict::OffSlot {
                requested,
                pool: pool_range,
            })
        } else if offset
            .checked_add(size)
            .is_none_or(|end| end > pool_range.length)
        {
            Some(PlacementConflict::OutsidePool {
                requested,
                pool: pool_range,
            })
        } else {
            None
        };
        if let Some(conflict) = conflict {
            return Err(Error::PlacementConflict(conflict));
        }
        slots
            .slots
            .allocate_at(start, size)
            .map_err(|(from, length)| {
                Error::PlacementConflict(PlacementConflict::Held {
                    requested,
                    held: AddressRange::of(from, length),
                })
            })?;
        let held = Held {
            start,
            size,
            pool: Some(pool),
        };
        Ok((GpuAddress(start), held))
    }

    /// Whether no range is held, in the space or outside it, and no records ID.
    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.ranges.is_empty()
            && self.outside.is_empty()
            && self.records.pool.is_empty()
            && self.records.slots.is_empty()
            && self.shares.is_empty()
            && self.pools.is_empty()
    }
}

/// The end of the span a pinned range of the pages `[start, end)` holds, as Metal holds it: one
/// guard page after them, rounded up to 32 KiB from `start` (module docs); `None` past `u64`.
fn span_end(start: u64, end: u64) -> Option<u64> {
    let pages = end.checked_add(PAGE)? - start;
    start.checked_add(pages.checked_next_multiple_of(GRANULE)?)
}
