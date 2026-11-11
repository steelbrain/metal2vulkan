//! The program's live buffers that a command reaches without naming them on a device, and the
//! resident set of them that every command buffer's root tables search.
//!
//! - **What Metal reaches** (the M3 Max under macOS 26.7, and an M4 under 27.0.1; every run alike,
//!   plain and with `MTL_DEBUG_LAYER`, the chips alike). A pointer loaded from memory, an argument
//!   buffer's pointer, and an indirect command's buffer address, read or written, reach a buffer of
//!   its own that no encoder of the command buffer named, from its making on. A no-copy buffer and
//!   a heap's buffer are reached so only once a command buffer's commands used it (a binding, a
//!   blit's operand, `useResource:`; for a heap's, any resource of the heap, or `useHeap:`): from
//!   that command buffer on, its own commands included, before that use or after it, and whether or
//!   not it completed. Before, the access is dropped: a read is 0, a write lands nowhere, and the
//!   command buffer completes with no error.
//! - **What is listed.** So every buffer the program made of its own
//!   ([`Device::new_buffer`], `new_buffer_with_bytes`, `new_buffer_with_descriptor`,
//!   `new_buffer_with_vk_buffer`) is listed here while it lives (`Device::visible`); a no-copy
//!   buffer (`new_buffer_with_bytes_no_copy`, the caller's host pages) from the commit of the
//!   first command buffer whose commands use it, and every buffer the program placed in a heap
//!   (`Heap::new_buffer`, `new_buffer_at_offset`) from the commit of the first whose commands use
//!   the heap, as `CommandBuffer::used_resources` lists them (`LiveBuffers::reach`), or from its
//!   placing in a heap so used. A rebind of a no-copy buffer's or a heap's pages
//!   (`rebind.rs`) ends the use: the buffer, or the heap's buffers, are listed again only once a
//!   command buffer uses it after the rebind, as Metal's M3 Max reaches neither the old pages nor
//!   the new until then (`LiveBuffers::rebound`). Each is listed at its own range, so an access
//!   past a placed buffer is contained as a bound one's is, not in the heap's other bytes
//!   (the runtime once listed the heap's whole buffer). The runtime's own buffers are
//!   not listed. What a command buffer's encoders made resident reaches its commands besides
//!   (`residency.rs`).
//! - **The table** (`LiveTable`). One resident set of every listed buffer, coalesced as a root
//!   table's (`bind::ResidentSet`), uploaded to a buffer of the runtime's own at a 32-byte-aligned
//!   device address. It is made at the first commit that needs it after a change of the list (a
//!   buffer made or gone, a rebind), kept by the device until the next change, and shared by
//!   every command buffer committed meanwhile. A command buffer holds it weakly from its commit
//!   to its first submission, which takes the table of the list then, the same one where the
//!   list is unchanged, and holds it, and with it each listed buffer's pages, until it completes
//!   (Metal resolves what a command buffer reaches as it submits it). A host step after then
//!   resolves against the table current as it runs, which the command buffer holds from then on
//!   too (`CommandBuffer::residency`): the commands the step records may reach its buffers. So a
//!   buffer released while a submitted command buffer may reach it keeps its memory until that
//!   command buffer completed: Metal's deallocator runs at the release, and the command buffer's
//!   later accesses reach nothing (the M3 Max under macOS 27.0.1, probe `dealloc-probe4`), where
//!   Vulkan frees no memory that submitted commands may still use
//!   (`VUID-vkFreeMemory-memory-00677`). With the caller's keep-alive the deallocator runs at the
//!   release all the same, and the import keeps the keep-alive; without one it waits, announced
//!   (`host-deallocator-delayed`). One the command buffer's commands name goes as
//!   it completes, where Metal's goes too, announced only where another submitted command buffer's
//!   table lists it: a command buffer lets go of its tables before what it retained
//!   (`CommandBufferShared::complete`). One released while only command buffers not
//!   yet submitted reach it goes at once, as Metal's does (the consumer's G344: a release after the
//!   commit lets the pages go, and the reach reads 0). A commit while the list is unchanged makes
//!   nothing (the cost, measured on an Apple M4 and a Radeon 780M).
//! - **Kept as the list changes**. The device keeps the list sorted, each buffer with its range and
//!   a hold on its pages from its listing on, in chunks the tables share (`sorted_chunks.rs`): a
//!   buffer made or gone changes one chunk, and the next table copies none of the others, upgrades
//!   no buffer and asks no buffer for its pages again. Its set is written into the buffer of the
//!   table it replaces where no command buffer held that table and the set fits, else into a new
//!   one. Before, each table made after a change upgraded, sorted and held every listed buffer
//!   again, and uploaded its set to a new buffer.
//!   A command buffer's root tables point at it, or at a set of it and what the command buffer's
//!   encoders bound and made resident beyond it (`command_buffer`).
//! - **On the host.** An indirect command's buffer address that no resource the command
//!   buffer's encoders made resident holds resolves against the table's buffers
//!   (`LiveTable::resolve`), as a pointer does on the GPU.
//! - **Of any count**. The search covers a set of any count (`metal2vulkan::bind::ResidentSet`);
//!   before, past 2^16 entries after coalescing, a command buffer's loaded pointers reached what
//!   its encoders made resident alone, announced (`unnamed-buffers-past-capacity`).

use super::buffer::{Buffer, CallerMemory, PagesHold, WeakBuffer};
use super::device::lock;
use super::error::{internal, Error};
use super::heap::Heap;
use super::resource::ObjectId;
use super::sorted_chunks::SortedChunks;
use super::used_resources::UsedResources;
use super::Device;
use metal2vulkan::bind::{ResidentRange, ResidentSet, RootTable};
use std::sync::{Arc, Mutex};

/// The program's live buffers on a device that a command reaches without naming them (the
/// module docs).
#[derive(Default)]
pub(super) struct LiveBuffers {
    inner: Mutex<Inner>,
}

/// A listed buffer's key: its Metal address, and its identity (Buffers placed in a heap may share
/// an address).
type Key = (u64, ObjectId);

/// A listed buffer (kept from its listing on, not asked again for each table).
#[derive(Clone)]
struct Listed {
    buffer: WeakBuffer,
    length: u64,
    /// Its range in a resident set, and a hold on its pages, once its memory is bound.
    resident: Option<(ResidentRange, PagesHold)>,
}

#[derive(Default)]
struct Inner {
    /// Each listed buffer by its key, in the chunks the tables share (`sorted_chunks.rs`).
    listed: SortedChunks<Key, Listed>,
    /// Bumped by each change of what the table lists.
    generation: u64,
    /// Bumped by each rebind or detach ([`LiveBuffers::rebound`]): a listing whose pages were
    /// read before one is read again ([`LiveBuffers::register`]).
    rebinds: u64,
    /// The table of the current generation, once made: dropped at the next change, so a gone
    /// buffer's pages go with the last command buffer that holds an older table. It holds the
    /// device only while a listed buffer lives, which holds it too.
    current: Option<Arc<LiveTable>>,
    /// The buffer of a table that no command buffer held when a change replaced it, which the
    /// next table's set is written into where it fits; kept only while a buffer is listed, so it
    /// holds the device only while a listed buffer does.
    spare: Option<Buffer>,
}

/// The resident set of the live buffers at one generation (the module docs).
pub(super) struct LiveTable {
    /// The set.
    set: ResidentSet,
    /// The buffer the set is uploaded to and the device address of its first entry; `None`
    /// for an empty set.
    upload: Option<(Buffer, u64)>,
    /// The listed buffers as the table was made, each with its range and the hold on its pages
    /// that keeps them while the table lives (shared with the tables before and after it).
    listed: SortedChunks<Key, Listed>,
    /// The generation of the list it was made of: a command buffer waiting for its first
    /// submission holds the table weakly and asks whether the list changed since.
    generation: u64,
}

/// What a change of the list let go of, for the caller to drop with no lock held (a last hold's
/// drop forgets a buffer or frees pages): the table made before it, and buffers no longer
/// listed.
#[must_use = "dropped with no lock held"]
struct Stale {
    table: Option<Arc<LiveTable>>,
    listed: Vec<Listed>,
    spare: Option<Buffer>,
}

impl LiveBuffers {
    /// List `buffer`, one of the program's own, where a command reaches it unnamed from its
    /// making on: any but a no-copy buffer, which [`reach`](Self::reach) lists once used.
    pub(super) fn made(&self, buffer: &Buffer) {
        if buffer.caller_memory() != Some(CallerMemory::HostPages) {
            self.register(buffer);
        }
    }

    /// List what the commands of a command buffer about to commit use (`used`, of what
    /// `CommandBuffer::used_resources` lists them, at least the no-copy buffers and the heaps:
    /// `used_resources::Reached`) that a command reaches unnamed only once used: each no-copy
    /// buffer, and each buffer the program placed in a heap they used, each
    /// at its own range (The heap's whole buffer, which the runtime once listed, let an
    /// access past a placed buffer reach the heap's other bytes, where a bound one's is contained).
    /// Listed from this commit on, for its own commands too; a buffer placed in such a heap later,
    /// from its placing ([`placed`](Self::placed)).
    pub(super) fn reach(&self, used: &UsedResources) {
        let no_copy = used
            .buffers
            .iter()
            .filter(|buffer| buffer.caller_memory() == Some(CallerMemory::HostPages));
        for buffer in no_copy {
            self.register(buffer);
        }
        for buffer in used.heaps.iter().flat_map(Heap::reached) {
            self.register(&buffer);
        }
    }

    /// `buffer`, one the program placed in `heap` just now: listed where a command buffer's
    /// commands used the heap already ([`reach`](Self::reach)).
    pub(super) fn placed(&self, heap: &Heap, buffer: &Buffer) {
        if heap.place(buffer) {
            self.register(buffer);
        }
    }

    /// List `buffer`, unless it is: a change of the list only where it was not. Its pages are
    /// read with no lock held, so a rebind or detach of them meanwhile ([`rebound`](Self::rebound),
    /// which follows the swap of its pages) has them read again: else the list would hold the
    /// pages it let go of until the buffer goes, and their deallocator with them.
    fn register(&self, buffer: &Buffer) {
        let key = (buffer.metal_address().0, buffer.object_id());
        loop {
            let rebinds = {
                let inner = lock(&self.inner);
                if inner.listed.contains(&key) {
                    return;
                }
                inner.rebinds
            };
            // Its range and pages, asked once, with no lock held.
            let listed = Listed {
                buffer: buffer.downgrade(),
                length: buffer.length(),
                resident: buffer.resident(),
            };
            let mut inner = lock(&self.inner);
            if inner.rebinds != rebinds {
                // Dropped with no lock held, as `let_go` drops: it may hold the old pages last.
                drop(inner);
                drop(listed);
                continue;
            }
            let stale = inner.listed.insert(key, listed).then(|| inner.change());
            drop(inner);
            self.let_go(stale);
            return;
        }
    }

    /// `buffer`'s import was released while idle and imported again before its Metal address was
    /// handed out (`host_residency.rs`): where it is listed, its range is read again, its new
    /// `VkBuffer`'s, or none while it is released.
    pub(super) fn refresh(&self, buffer: &Buffer) {
        let (address, id) = (buffer.metal_address().0, buffer.object_id());
        if lock(&self.inner).listed.contains(&(address, id)) {
            self.forget(address, id);
            self.register(buffer);
        }
    }

    /// The buffer `id` at Metal address `address` is gone.
    pub(super) fn forget(&self, address: u64, id: ObjectId) {
        let stale = {
            let mut inner = lock(&self.inner);
            match inner.listed.remove(&(address, id)) {
                Some(listed) => {
                    let mut stale = inner.change();
                    stale.listed.push(listed);
                    Some(stale)
                }
                None => None,
            }
        };
        self.let_go(stale);
    }

    /// `buffer`'s pages were replaced (`rebind.rs`), or where `heap` is given those of the heap
    /// whose own buffer it is: Metal reaches neither the old pages nor the new unnamed until a
    /// command buffer's commands use the buffer, or the heap, again (the consumer's measurement
    /// on the M3 Max: a use does not survive a detach and replace). So the buffer, or each
    /// buffer placed in the heap, is no longer listed, and the heap no longer used, until
    /// [`reach`](Self::reach) lists them again. Another no-copy buffer that shared the old
    /// pages' import (`aliases.rs`) keeps its listing.
    pub(super) fn rebound(&self, buffer: &Buffer, heap: Option<&Heap>) {
        let gone = match heap {
            Some(heap) => heap.unreached(),
            None => vec![buffer.clone()],
        };
        let stale = {
            let mut inner = lock(&self.inner);
            inner.rebinds = inner.rebinds.wrapping_add(1);
            let mut listed = Vec::new();
            for buffer in &gone {
                let key = (buffer.metal_address().0, buffer.object_id());
                listed.extend(inner.listed.remove(&key));
            }
            let mut stale = inner.change();
            stale.listed = listed;
            stale
        };
        self.let_go(Some(stale));
        drop(gone);
    }

    /// Let go of what a change made stale, with no lock held: a table no command buffer held
    /// gives its buffer back as the spare, while a buffer is listed (`Inner::spare`).
    fn let_go(&self, stale: Option<Stale>) {
        let Some(Stale {
            table,
            listed,
            spare,
        }) = stale
        else {
            return;
        };
        let freed = table
            .and_then(|table| Arc::try_unwrap(table).ok())
            .and_then(|table| table.upload)
            .map(|(buffer, _)| buffer);
        let refused = freed.and_then(|buffer| {
            let mut inner = lock(&self.inner);
            if inner.listed.is_empty() || inner.spare.is_some() {
                return Some(buffer);
            }
            inner.spare = Some(buffer);
            None
        });
        // Dropped with no lock held: a last hold's drop frees pages or forgets a buffer.
        drop(refused);
        drop(listed);
        drop(spare);
    }

    /// The generation of the list now: bumped by each change of what the table lists.
    pub(super) fn generation(&self) -> u64 {
        lock(&self.inner).generation
    }

    /// The table of the live buffers now: the one made since the last change of the list, else a
    /// new one, which the device keeps until the next change. Refused as `device` refuses its upload.
    pub(super) fn table(&self, device: &Device) -> Result<Arc<LiveTable>, Error> {
        let (generation, listed, spare) = {
            let mut inner = lock(&self.inner);
            if let Some(table) = &inner.current {
                return Ok(Arc::clone(table));
            }
            (inner.generation, inner.listed.clone(), inner.spare.take())
        };
        self.make(device, generation, listed, spare)
    }

    /// The table of `listed`, the buffers listed at `generation`, its set written into `spare`
    /// where it fits: kept as the current one where the list
    /// is still at `generation`.
    fn make(
        &self,
        device: &Device,
        generation: u64,
        listed: SortedChunks<Key, Listed>,
        spare: Option<Buffer>,
    ) -> Result<Arc<LiveTable>, Error> {
        let ranges = listed
            .iter()
            .filter_map(|(_, listed)| listed.resident.as_ref().map(|(range, _)| *range));
        let set = ResidentSet::new(ranges);
        let (upload, spare) = match set.is_empty() {
            false => (Some(upload(device, &set, listed.len(), spare)?), None),
            true => (None, spare),
        };
        let table = Arc::new(LiveTable {
            set,
            upload,
            listed,
            generation,
        });
        let (replaced, spare) = {
            let mut inner = lock(&self.inner);
            match inner.generation == generation {
                true => {
                    let replaced = inner.current.replace(Arc::clone(&table));
                    match (inner.spare.is_none() && !inner.listed.is_empty(), spare) {
                        (true, Some(spare)) => {
                            inner.spare = Some(spare);
                            (replaced, None)
                        }
                        (_, spare) => (replaced, spare),
                    }
                }
                false => (None, spare),
            }
        };
        // Dropped with no lock held: a last hold's drop forgets a buffer.
        drop(replaced);
        drop(spare);
        Ok(table)
    }
}

impl Inner {
    /// A change of what the table lists: the table made before it, and the spare once no buffer
    /// is listed, which the caller drops with no lock held.
    fn change(&mut self) -> Stale {
        self.generation += 1;
        Stale {
            table: self.current.take(),
            listed: Vec::new(),
            spare: self.listed.is_empty().then(|| self.spare.take()).flatten(),
        }
    }
}

impl LiveTable {
    /// The generation of the list it was made of.
    pub(super) fn generation(&self) -> u64 {
        self.generation
    }

    /// The set.
    pub(super) fn set(&self) -> &ResidentSet {
        &self.set
    }

    /// Each listed buffer's range at its own length, before the set coalesced them, in the
    /// list's order: what a replay's resident set lists apart (`gpu_execute.rs`,
    /// `BindState::resident_ranges`).
    pub(super) fn ranges(&self) -> impl Iterator<Item = ResidentRange> + '_ {
        self.listed
            .iter()
            .filter_map(|(_, listed)| listed.resident.as_ref().map(|(range, _)| *range))
    }

    /// The device address of the set's first entry; `None` for an empty set.
    pub(super) fn address(&self) -> Option<u64> {
        self.upload.as_ref().map(|&(_, address)| address)
    }

    /// The listed buffer that Metal address `address` lies in, and its offset there, as
    /// `Residency::resolve` finds a resident one: an address at a buffer's end is in it too.
    /// `None`: in none, or in one gone since the table was made.
    pub(super) fn resolve(&self, address: u64) -> Option<(Buffer, u64)> {
        let ((start, _), listed) = self.listed.last_where(|&(start, _)| start <= address)?;
        let offset = address - start;
        if offset > listed.length {
            return None;
        }
        Some((listed.buffer.upgrade()?, offset))
    }
}

/// `set`, of a table of `listed` buffers, in a buffer of `device`'s own, from a 32-byte-aligned
/// device address: `spare` where it holds it (a table's that no command buffer holds), else a new
/// one with room for the entries of as many listed buffers as the power of two at or above
/// `listed`, so the next tables fit it while the list stays within that. The buffer and that
/// address.
fn upload(
    device: &Device,
    set: &ResidentSet,
    listed: usize,
    spare: Option<Buffer>,
) -> Result<(Buffer, u64), Error> {
    let bytes = set.bytes();
    let alignment = RootTable::ENTRY_ALIGNMENT;
    let length = bytes.len() as u64 + alignment;
    let buffer = match spare {
        Some(spare) if spare.length() >= length => spare,
        _ => {
            device.shared().counts.live_table_uploaded();
            // A set has at most an entry a listed buffer: coalescing only drops and trims.
            let entry = bytes.len() as u64 / set.len().max(1);
            let room = (listed.next_power_of_two() as u64).max(set.len()) * entry;
            device.write_combined_buffer(room + alignment)?
        }
    };
    let start = buffer
        .device_address()
        .ok_or_else(|| internal("a resident set on a device without device addresses"))?;
    let skip = start.next_multiple_of(alignment) - start;
    let Some(mapped) = buffer.mapped_contents() else {
        return Err(internal("a resident set's buffer without host bytes"));
    };
    // SAFETY: the buffer is `Shared`, mapped, and `skip + bytes.len()` bytes long at least; no
    // command reads it: a new one, or a spare no command buffer holds.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            mapped.as_ptr().add(skip as usize),
            bytes.len(),
        );
    }
    Ok((buffer, start + skip))
}
