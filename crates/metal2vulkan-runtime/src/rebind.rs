//! Rebinding a no-copy buffer, or a heap over the caller's host memory, to other host pages
//! (a runtime extension, the companion of `new_buffer_with_bytes_no_copy` and
//! `new_heap_with_bytes_no_copy`).
//!
//! A consumer that gathers its own pages into one contiguous range (Mach VM remapping on macOS,
//! `memfd` mappings on Linux) gives an existing no-copy buffer or heap a new range of the same
//! length. Metal, and MoltenVK, bind a no-copy buffer's physical pages when it is made, so a
//! later remap over its range never reaches the GPU; the runtime instead keeps the object and
//! its Metal address, and binds it to the new range from then on.
//!
//! - **What stays.** The buffer or heap and every resource placed in it, their `gpu_address`
//!   (the device's own Metal address space, `address.rs`: the resident table maps each address
//!   to the `VkBuffer` behind it at encode time), their resource IDs, labels, offsets and
//!   options. `contents` answers the new pointer at once.
//! - **The ordering rule, by submission** (by commit before). Metal resolves a command buffer's
//!   uses as it submits each of its submissions (the consumer's rebind probe, tables 2 and 3, the
//!   M3 Max; and a rebind between two submissions of one command buffer, the M3 Max under macOS
//!   27.0.1): one submitted before the rebind keeps the old pages until its work completed; one
//!   committed before it and submitted after reaches the new pages through what it names or made
//!   resident (a later submission of a command buffer whose earlier one ran included), and none
//!   through what it only reaches unnamed (the rebind ends the use). The runtime resolves both as
//!   each event group is submitted (`command_buffer/first_submission.rs`); a command it encoded
//!   that names the buffer took the pages as it was encoded (`Taken`), so the rebind encodes its
//!   compute, blit and render encoders that named the buffer again, in the rebind's call, before
//!   their group's submission (`replay.rs`, `command_buffer/reencode.rs`); an encoder it cannot
//!   encode again (a compute encoder its unit of command predication reaches past, among others)
//!   fails
//!   `CommandBufferError::EncodedUseRebound` there, as all did before. Commands encoded after
//!   the rebind use the new pages. A rebind while a command buffer that is not yet committed has
//!   encoded a use of the buffer or heap (one of its resources included), or has an encoder open
//!   that used one (a render encoder uses its attachments' bytes from its open), is refused
//!   (`Invalid`, [`RebindProblem::EncodedUses`]): its commands took the old pages, and committing
//!   them after the rebind would be neither order. A command buffer committed with commands the
//!   runtime records only once earlier work completed (executions of an indirect command buffer
//!   the GPU writes, and a dispatch left to its host step, `indirect_dispatch.rs`: an indirect
//!   one off the GPU's own path, any while an indirect stage-in region is set), whose host step
//!   resolves the buffers its encoder named as it records (`Deferred::resolved_at_its_calls`),
//!   counts as uncommitted until it completed: a rebind meanwhile is refused likewise
//!   (`EncodedUses`), where Metal's replace before the command buffer's submission reaches the new
//!   pages, and one after it keeps the old pages until it completed. So does one whose predicated
//!   loop's laying left such work, or work on the GPU alone, apart, which each run again of the
//!   loop records again (`command_buffer/rerun.rs`). One committed with commands
//!   left for a `VkPipeline` still being made does not: a dispatch whose pipelines were being
//!   made (`pending_dispatch.rs`) and a render pass left for its draws' and tile dispatches'
//!   (`render/pending_tile.rs`) record what their calls resolved, so the rebind encodes their
//!   encoder again as it does any other, and Metal's replace fails for neither. Nor does one
//!   whose predicated loop's laying left no work apart, whose runs again submit the laying as
//!   encoded: a rebind after its submission keeps the old pages for every run again, as Metal's
//!   loop runs on over them, and one before it fails that submission (`EncodedUseRebound`), as
//!   the laying's encoders cannot be encoded again.
//! - **Other threads.** A rebind may run while other threads encode and commit command buffers that
//!   use the buffer or heap. Its publish runs from the moment the buffer lies in the new pages
//!   ([`Rebinds::replace`]) to the moment everything over its bytes (aliasing images and their
//!   views, texel views, the heap sets' elements) is made over them ([`Rebinds::published`]); the
//!   old pages, their twins and their deallocator go only after it closes, and the close counts as
//!   a move (`buffer::host_cells_moved`), so no binding an encoder read while it was open is kept
//!   past it. Each encoder call reads, as it begins, how many publishes closed (one atomic load,
//!   [`Rebinds::call_begins`]), and its holds compare it: a call that holds or takes the buffer
//!   while its publish is open, or after it closed having begun before, raced the rebind, and what
//!   it read of the buffer before its hold may be the old pages' (`Taken::raced`, counted by
//!   `WorkCounts::encoders_raced_rebind`). Such an encoder runs only encoded again over the new
//!   pages, after the publish closed: by its commit, in the commit's call, or by the rebind's call
//!   where it committed while the publish was open. That holds for blit encoders and for render
//!   encoders that keep a log. The calls of the encoders the runtime may not encode again are
//!   gated instead, so none races a rebind ([`CallGate`]): every compute encoder's (its unit's
//!   records of command predication, or a dispatch it leaves to its host step, a GPU-written
//!   indirect one among them, can leave it one the runtime cannot encode again, from the call that
//!   makes it so or one of a later encoder of its unit), and a render encoder's that opened while
//!   the device had no buffer over host pages, which keeps no log. A rebind opens its publish just
//!   before its replace, once no such call is under way, and each such call that begins while one
//!   is open waits for it to close before it reads anything (one sequentially consistent
//!   handshake of two counts, [`Rebinds::quiet_calls`]): so each reads the buffer wholly before
//!   the publish, holding it, which refuses a rebind until its command buffer commits, or wholly
//!   after it. A rebind that waited 2 ms for such a call still under way (one waiting for a
//!   `VkPipeline` being made, say) is refused (`EncodedUses`, counted by
//!   `WorkCounts::rebinds_refused_calls_in_flight`), never the encoder failed later; the caller may
//!   try again. Nothing waits for that rebind: a call under way never waits for one, and a call
//!   that waits does so as it begins. A blit or logged render encoder whose call raced still fails
//!   `CommandBufferError::EncodedUseRebound` at its submission where its command buffer's commit
//!   fails it, or a call that answers otherwise than at its encoding refuses its encoding again
//!   (`command_buffer/rebound_use.rs`; a resource state encoder takes no pages, so races none). A
//!   commit and a submission order against the rebind under `buffers`' lock as above. Another rebind of the same buffer or heap whose publish is open is refused
//!   (`EncodedUses`); a rebind or detach of it on another thread, a texture made over its bytes and
//!   a resource placed in the heap during the call stay the caller's to exclude
//!   (`Buffer::rebind_bytes_no_copy`'s **Safety**).
//! - **The new pages' import.** New memory that is the bytes a live no-copy buffer was made over,
//!   of its memory class, lies in that buffer's import, as a buffer made over them does
//!   (`host_aliases.rs`): the rebind imports nothing, pins no page and asks the kernel about no
//!   mapping (`WorkCounts::host_mappings_queried`). A heap's, and a buffer's with a texture's
//!   aliasing image over its bytes, import their new memory themselves.
//! - **The deallocators.** The old range's runs once every command buffer first submitted
//!   before the rebind that used it completed (at the rebind where none holds it: a command
//!   buffer not yet submitted holds its pages weakly); the new range's at the next rebind's turn,
//!   or when the buffer or heap goes. A detached buffer's or heap's rebind runs, inside the call,
//!   the deallocator of the range its detach let go of (`detach.rs`), as Metal's replace does.
//! - **Textures.** Nothing of the old contents is copied. A heap texture bound to the heap's
//!   bytes (a no-copy heap's linear image), and a texture over a no-copy buffer (or over a
//!   buffer of a no-copy heap) whose linear image aliases its bytes, has its image made again over
//!   the new pages, at its offset, before the rebind is made; the image swaps in under the texture
//!   and every view of it (`Image::swap_in`), and the old image and its views go with the old
//!   pages, which the command buffers that used them hold. A texture held as a shadow of the bytes
//!   (a heap's, or one over a no-copy buffer) reads its buffer's `VkBuffer` as each copy is
//!   recorded (`ShadowBuffer::Rebound`); a texture buffer's views are emptied, going with the old
//!   pages, and made again at the next bind. The texture slots' generations change, so each command
//!   buffer committed after the rebind writes its set-1 elements again (`heap_sets.rs`). The
//!   copies, views and aliasing images are planned at the bytes' offset in their `VkBuffer`, which
//!   a heap's rebind keeps (its first byte at the import's unit); a no-copy buffer with a texture
//!   over it given memory at another offset in its page lies in a copy of it laid out at the old
//!   offset (`copied.rs`, `copied-host-pages`). A rebind is refused
//!   (`NotImplemented(RebindTextures)`) where the device cannot make an aliasing image again over
//!   the new pages (their memory type is not one it takes). Where the caller's gate refuses the
//!   new pages' import, or the admitted import fails, a buffer's or a heap's rebind takes a copy
//!   of them (`copied.rs`), its images made again over the copy. Over new pages imported as
//!   dma-bufs, each such image is made again over a dma-buf of its own new pages before the rebind
//!   settles (`Device::dma_buf_images_again`, `dma_buf.rs`), and refused likewise where it is not.

use super::buffer::Imported;
use super::buffer::{Buffer, Detached, HostCell, HostPages, LeftTwins, Lying, WeakBuffer};
use super::command_buffer::CommandBufferShared;
use super::device::lock;
use super::error::{invalid, Error, Problem};
use super::heap::Heap;
use super::host_memory::{Deallocator, HostMemory, HostRelease, KeepAlive};
use super::key_hash::KeyMap;
use super::program_objects;
use super::rate_map::{MapState, RasterizationRateMap};
use super::resource::{ObjectId, ResourceKey};
use super::Device;
use std::any::Any;
use std::cell::Cell;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::{Duration, Instant};

/// What `rebind_bytes_no_copy` refuses as Metal's validation would a misuse (`Invalid`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RebindProblem {
    /// The buffer is not a no-copy buffer of its own (a buffer of the runtime's memory, one over
    /// the caller's `VkBuffer`, or a heap's buffer, whose heap is rebound instead), or the heap is
    /// not over the caller's host memory.
    NotHostMemory,
    /// A command buffer that is not yet committed has encoded a use of the buffer or heap, or of
    /// a resource placed in it, or has an encoder open that used one (a render encoder uses its
    /// attachments' bytes from its open), or one committed whose commands the runtime records
    /// only once earlier work completed (an indirect command buffer's executions the GPU writes,
    /// a dispatch left to its host step, a predicated loop's run again of such work or of work on
    /// the GPU alone): commit or drop it first, or wait for its completion. Also a call on
    /// another thread of an encoder the runtime may not encode again (a compute encoder's, or a
    /// render encoder's that keeps no log) still under way 2 ms after the rebind asked to go on
    /// (the module docs of `rebind.rs`, **Other threads**): try again.
    EncodedUses,
}

/// The device's buffers over host pages, by their hazard-tracking key, with the command buffers
/// not yet committed whose commands used each.
#[derive(Default)]
pub(super) struct Rebinds {
    live: AtomicUsize,
    buffers: Mutex<KeyMap<ResourceKey, Entry>>,
    /// The keys `buffers` may hold ([`HostedKeys`]): a use of any other names no buffer over host
    /// pages, which an encoder's hold tells without taking `buffers`' lock.
    hosted: HostedKeys,
    /// How many of them are detached (`detach.rs`): else no command names a detached one.
    detached: AtomicUsize,
    /// Signaled as the hold of a submitted batch that took one of them goes, its work having
    /// completed (`SubmittedHolds`): a buffer's detach waits for no such hold to be left.
    completed: Condvar,
    /// The device's stand-in a detached buffer's commands take (`buffer::Detached`), while one
    /// is detached: made again once every one let go of it.
    pub(super) stand_in: Mutex<Option<WeakBuffer>>,
    /// The stand-ins the images of textures over detached bytes present, by plan
    /// (`texture/stand_in.rs`).
    pub(super) stand_ins: super::texture::stand_in::StandIns,
    /// The textures over detached bytes whose unnamed reads may not be Metal's, not yet
    /// announced (`Device::reach_stale_detached`).
    pub(super) stale: Mutex<Vec<StaleTexture>>,
    /// How many there are: the check at each submission costs one load while none is.
    pub(super) stale_count: AtomicUsize,
    /// The command buffers committed whose last event group is not yet submitted that an encoder
    /// of theirs can encode again (`replay.rs`), or work of theirs on the GPU alone can be
    /// recorded again, with the keys of the buffers over host pages those encoders and that work
    /// named: a rebind of one of them has the encoders of the groups not yet submitted encoded
    /// again, and that work recorded again before the first submission. Taken under `buffers`'
    /// lock, after it.
    waiting: Mutex<Waiting>,
    /// How many rebinds closed their publish ([`Rebinds::published`]): an encoder's call reads it
    /// as it begins ([`Rebinds::call_begins`]), and a hold or pin in the call of a buffer whose
    /// publish closed past that, or is open, finds the call raced it (the module docs, **Other
    /// threads**).
    published: AtomicU64,
    /// How many rebinds' publishes are open ([`Rebinds::replace`] to [`Rebinds::published`]),
    /// each from just before its replace, as it lets the gated calls under way end
    /// ([`Rebinds::quiet_calls`]).
    open: AtomicUsize,
    /// Signaled as a rebind closes its publish, or gives up opening one: an encoding again that
    /// raced one waits for none to be open before it encodes again
    /// (`CommandBufferShared::reencode`), and so does a gated call before it begins
    /// ([`CallGate`]).
    closed: Condvar,
    /// How many gated calls are under way ([`CallGate`]): the calls of encoders the runtime may
    /// not encode again, which no publish overlaps (the module docs, **Other threads**). With
    /// `open`, the two sides of one sequentially consistent handshake.
    calls: AtomicUsize,
    /// Held as a rebind waits for the gated calls under way to end ([`Rebinds::quiet_calls`]),
    /// and as the last of them signals `drained`.
    drain: Mutex<()>,
    drained: Condvar,
}

/// The command buffers waiting for the submission of an event group whose encoders a rebind
/// encodes again,
/// or whose work on the GPU alone it records again ([`Rebinds::wait_for_replay`]), each with the
/// keys they named, sorted.
#[derive(Default)]
struct Waiting {
    buffers: Vec<(Weak<CommandBufferShared>, Box<[ResourceKey]>)>,
    /// The command buffers waiting so whose encoders' passes are over rasterization-rate maps,
    /// with the maps' identities (`RasterizationRateMap::id`): a reset of one has those encoders
    /// encoded again ([`Rebinds::reset_map`]).
    maps: Vec<(Weak<CommandBufferShared>, Box<[usize]>)>,
    /// The length past which the next registration first lets go of the entries whose command
    /// buffer went or was submitted.
    prune_at: usize,
}

impl Waiting {
    /// Before a registration: where the entries reached `prune_at`, let go of those whose
    /// command buffer went before its first submission (its submission failed first), never
    /// upgraded, so none goes under the lock.
    fn prune(&mut self) {
        if self.buffers.len() < self.prune_at {
            return;
        }
        self.buffers.retain(|(buffer, _)| buffer.strong_count() > 0);
        self.maps.retain(|(buffer, _)| buffer.strong_count() > 0);
        self.prune_at = (2 * self.buffers.len().max(self.maps.len())).max(16);
    }
}

/// A texture over detached bytes whose unnamed reads a submission's reach may not answer as
/// Metal's (`Rebinds::stale`, `Device::note_stale`): the texture (by type erased,
/// `buffer::NotedTexture`), its object's ID, the hold on the pages of the bytes, whose rebind
/// ends it, and why.
pub(super) struct StaleTexture {
    pub(super) texture: Weak<dyn Any + Send + Sync>,
    pub(super) id: ObjectId,
    pub(super) cell: Weak<HostCell>,
    pub(super) why: Unexact,
}

/// Why a texture over detached bytes is noted ([`StaleTexture`]), each announced by a kind of
/// its own.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Unexact {
    /// Its bytes were detached after a submitted command buffer made their pages resident:
    /// Metal's unnamed reads read those pages (`Degradation::StaleDetachedReach`).
    StalePages,
    /// Its stand-in has a sink: a read through a storage-class element reads what an unnamed
    /// write left there, where Metal's reads zero (`Degradation::DetachedStorageKeepsWrites`).
    StorageSink,
}

/// One buffer over host pages.
struct Entry {
    cell: Weak<HostCell>,
    /// The holds of encoders and of command buffers not yet committed on it ([`Holds`]).
    pending: u32,
    /// The holds of submitted work that took it (`SubmittedHolds`), each from the submission of
    /// its batches until their work completed, and, for a command buffer's last batches, until
    /// its completion is processed, before its completed handlers run: a detach refuses while one
    /// is, and a buffer's waits for the work.
    submitted: u32,
    /// It is detached (`detach.rs`), as its cell says.
    detached: bool,
    /// A submitted command buffer took the pages it lies in now and found them its own: Metal
    /// made them resident, so a detach leaves their textures' unnamed reads reading those pages
    /// (`Device::reach_stale_detached`).
    resident: bool,
    /// A rebind gave it new pages and has not closed its publish yet ([`Rebinds::published`]):
    /// the images, views and heap elements over its bytes may still be the old pages'.
    publishing: bool,
    /// [`Rebinds::published`]'s count as its last rebind's publish closed (0 before any).
    published: u64,
}

impl Entry {
    /// Whether an encoder's call begun at `call` ([`Rebinds::call_begins`]) that holds or pins the
    /// buffer now raced a rebind of it: the rebind's publish is open, or closed after the call
    /// began, so what the call read of the buffer before this hold may be the old pages', or
    /// half made (the module docs, **Other threads**).
    fn raced(&self, call: u64) -> bool {
        self.publishing || self.published > call
    }
}

/// What a command takes of a buffer over host pages that lies as `cell` says: the pages it lies
/// in, or none while it is detached.
fn pinned(cell: &HostCell) -> Pinned {
    match cell.lying() {
        Lying::In(pages) => Pinned::Held(pages),
        Lying::Detached(_) | Lying::Gone(_) => Pinned::Detached,
    }
}

/// The keys a device registered buffers over host pages under ([`Rebinds::register`]), as a filter
/// of bits by key that never forgets one: a key whose bit is clear names no such buffer, for a
/// buffer registers its key as it is made, before any command can name it, and a key is never
/// another resource's. A set bit may be another key's, or a buffer's gone since: such a key is
/// looked up as every key was before. So an encoder's commands, which name each texture and
/// buffer they use, take the lock that a submission holds as it judges its holds
/// (`Submitting`) only for a buffer over host pages, and the rare key that shares its bit
/// (`WorkCounts::host_holds_looked_up`).
pub(super) struct HostedKeys([AtomicU64; HostedKeys::WORDS]);

impl HostedKeys {
    /// The filter's 64-bit words: 4096 bits.
    const WORDS: usize = 64;

    /// The word and bit of `key`.
    fn bit(key: ResourceKey) -> (usize, u64) {
        let at = key.ordinal() as usize % (Self::WORDS * 64);
        (at / 64, 1 << (at % 64))
    }

    /// Note `key`, a buffer's over host pages, before any command can name it.
    fn insert(&self, key: ResourceKey) {
        let (word, bit) = Self::bit(key);
        self.0[word].fetch_or(bit, Ordering::Release);
    }

    /// Whether `key` may be one noted: `false` only for a key never noted.
    pub(super) fn may_hold(&self, key: ResourceKey) -> bool {
        let (word, bit) = Self::bit(key);
        self.0[word].load(Ordering::Acquire) & bit != 0
    }
}

impl Default for HostedKeys {
    fn default() -> HostedKeys {
        HostedKeys(std::array::from_fn(|_| AtomicU64::new(0)))
    }
}

impl Rebinds {
    /// The host addresses of the bytes of the buffer of `key` over host pages
    /// (`HostCell::host_range`); `None` for any other key, and for a buffer gone or detached.
    pub(super) fn host_range(&self, key: ResourceKey) -> Option<std::ops::Range<usize>> {
        let cell = lock(&self.buffers).get(&key)?.cell.upgrade()?;
        cell.host_range()
    }

    /// Whether the device has a buffer over host pages: else no use names one.
    pub(super) fn any(&self) -> bool {
        self.live.load(Ordering::Relaxed) > 0
    }

    /// Note the buffer of `key` over host pages, `cell` its hold on them.
    pub(super) fn register(&self, key: ResourceKey, cell: &Arc<HostCell>) {
        let entry = Entry {
            cell: Arc::downgrade(cell),
            pending: 0,
            submitted: 0,
            detached: false,
            resident: false,
            publishing: false,
            published: 0,
        };
        self.hosted.insert(key);
        if lock(&self.buffers).insert(key, entry).is_none() {
            self.live.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The buffer of `key` is gone.
    pub(super) fn forget(&self, key: ResourceKey) {
        if let Some(entry) = lock(&self.buffers).remove(&key) {
            self.live.fetch_sub(1, Ordering::Relaxed);
            if entry.detached {
                self.detached.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    /// Hold `key` for an encoder's commands, which used it: the pages they took, which their
    /// command buffer retains until it completed, or `Pinned::Detached` while it is detached;
    /// `None` where `key` names no buffer over host pages. The hold refuses a rebind until
    /// [`Rebinds::release`]. With whether the encoder's call that takes it, begun at `call`
    /// ([`Rebinds::call_begins`]), raced a rebind of it ([`Entry::raced`]).
    fn hold(&self, key: ResourceKey, call: u64) -> Option<(Pinned, bool)> {
        let mut buffers = lock(&self.buffers);
        let entry = buffers.get_mut(&key)?;
        let cell = entry.cell.upgrade()?;
        entry.pending += 1;
        Some((pinned(&cell), entry.raced(call)))
    }

    /// What a command an encoder encodes again took of the buffer of `key` (`replay.rs`), as
    /// [`Rebinds::hold`] answers it, without the hold: its command buffer is committed, so
    /// nothing it encodes refuses a rebind.
    fn pin(&self, key: ResourceKey, call: u64) -> Option<(Pinned, bool)> {
        let buffers = lock(&self.buffers);
        let entry = buffers.get(&key)?;
        let cell = entry.cell.upgrade()?;
        Some((pinned(&cell), entry.raced(call)))
    }

    /// The count an encoder's call reads as it begins, before it reads anything of a buffer over
    /// host pages (`Holds::begin_call`): a hold or pin in the call of a buffer whose rebind's
    /// publish closed past that count, or is open, finds the call raced the rebind
    /// ([`Entry::raced`]).
    pub(super) fn call_begins(&self) -> u64 {
        self.published.load(Ordering::Acquire)
    }

    /// The rebind that gave the buffer of `key` new pages ([`Rebinds::replace`]) closes its
    /// publish: everything over its bytes is made over the new pages (images swapped in, texel
    /// views let go of, heap elements written again). An encoder's call begun before now that
    /// holds or pins the buffer finds it raced the rebind ([`Entry::raced`]). The command buffers
    /// waiting for the submission of an event group whose replayable encoders named it, marked
    /// under this lock to encode them again as `replace` marks them, for the caller to run: those
    /// committed while the publish was open among them, whose raced encoders their commit encoded
    /// again then (`State::raced_again`) over what may not yet have been the new pages'.
    pub(super) fn published(&self, key: ResourceKey) -> Vec<Arc<CommandBufferShared>> {
        let mut buffers = lock(&self.buffers);
        let Some(entry) = buffers.get_mut(&key).filter(|entry| entry.publishing) else {
            return Vec::new();
        };
        entry.publishing = false;
        // What a binding read of the buffers over host pages while the publish was open may be of
        // the old pages' images and views: a move, for every binding kept since
        // (`buffer::host_cells_moved`), before a call that begins after the close can read the
        // count.
        super::buffer::host_cells_settled();
        entry.published = self.published.fetch_add(1, Ordering::AcqRel) + 1;
        self.open.fetch_sub(1, Ordering::SeqCst);
        #[cfg(debug_assertions)]
        PUBLISHES_OPEN.fetch_sub(1, Ordering::AcqRel);
        let again = self.mark_waiting(key);
        drop(buffers);
        self.closed.notify_all();
        again
    }

    /// Wait until no rebind's publish is open, with no other lock of the runtime's held: an
    /// encoding again that raced one encodes again after (`CommandBufferShared::reencode`), and a
    /// publish waits for nothing an encoder holds.
    pub(super) fn wait_published(&self) {
        let mut buffers = lock(&self.buffers);
        while self.open.load(Ordering::Acquire) > 0 {
            buffers = match self.closed.wait(buffers) {
                Ok(buffers) => buffers,
                Err(poisoned) => poisoned.into_inner(),
            };
        }
    }

    /// Open a publish ahead of its replace, once the gated calls under way ended ([`CallGate`]):
    /// from the count's rise every gated call that begins waits for the publish to close, so
    /// those under way end, which this waits for, up to [`CALLS_DRAIN`]. A call held past it
    /// (one waiting for a `VkPipeline` being made, say) refuses the rebind instead: `false`,
    /// nothing opened, the waiting calls let go. Its wait holds no lock of the runtime's, and a
    /// gated call's end takes `drain` alone, so the wait ends within its bound whatever the call
    /// under way waits for.
    ///
    /// The handshake with [`Rebinds::enter_call`]: each side writes its count, then reads the
    /// other's, all four sequentially consistent, so of a rebind and a call that meet at least one
    /// reads the other's rise. A call that reads `open` raised backs off before it reads anything
    /// of a buffer over host pages; a rebind that reads `calls` raised waits for it to fall.
    fn quiet_calls(&self) -> bool {
        self.open.fetch_add(1, Ordering::SeqCst);
        if self.calls.load(Ordering::SeqCst) == 0 {
            return true;
        }
        let deadline = Instant::now() + CALLS_DRAIN;
        let mut drain = lock(&self.drain);
        while self.calls.load(Ordering::SeqCst) > 0 {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                drop(drain);
                self.unopen(lock(&self.buffers));
                return false;
            }
            drain = match self.drained.wait_timeout(drain, left) {
                Ok((drain, _)) => drain,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
        true
    }

    /// A publish [`quiet_calls`](Self::quiet_calls) opened closes unpublished (its rebind
    /// refused, or its buffer not one of `buffers`, whose lock the caller hands in): the gated
    /// calls and encodings again waiting for it go on.
    fn unopen(&self, buffers: MutexGuard<'_, KeyMap<ResourceKey, Entry>>) {
        self.open.fetch_sub(1, Ordering::SeqCst);
        drop(buffers);
        self.closed.notify_all();
    }

    /// A gated call begins ([`CallGate`]): counted under way, once no publish is open, waiting
    /// for each open one to close first ([`Rebinds::wait_published`]). Whether it waited. Its
    /// caller reads nothing of a buffer over host pages before.
    fn enter_call(&self) -> bool {
        let mut waited = false;
        loop {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.open.load(Ordering::SeqCst) == 0 {
                return waited;
            }
            self.leave_call();
            waited = true;
            self.wait_published();
        }
    }

    /// A gated call ends ([`CallGate`]): a rebind waiting for the calls under way hears it.
    fn leave_call(&self) {
        self.calls.fetch_sub(1, Ordering::SeqCst);
        if self.open.load(Ordering::SeqCst) > 0 {
            let _drain = lock(&self.drain);
            self.drained.notify_all();
        }
    }

    /// Mark each command buffer waiting for the submission of an event group whose replayable
    /// encoders or work on the GPU alone named the buffer of `key` to encode or record them again
    /// before that submission (`CommandBufferShared::reencode`), each once, for the caller to run
    /// and let go of without this lock. The caller holds `buffers`' lock.
    fn mark_waiting(&self, key: ResourceKey) -> Vec<Arc<CommandBufferShared>> {
        let mut again: Vec<Arc<CommandBufferShared>> = Vec::new();
        let waiting = lock(&self.waiting);
        let named = (waiting.buffers.iter())
            .filter(|(_, keys)| keys.binary_search(&key).is_ok())
            .map(|(buffer, _)| buffer);
        for buffer in named {
            let Some(buffer) = buffer.upgrade() else {
                continue;
            };
            if again.iter().any(|had| Arc::ptr_eq(had, &buffer)) {
                continue;
            }
            buffer.mark_reencode();
            again.push(buffer);
        }
        again
    }

    /// Note `buffer`, committed now and not yet first submitted, whose encoders that can be
    /// encoded again named the buffers of `keys` (sorted): a rebind of one of them before the
    /// submission of their event group has them encoded again ([`Rebinds::replace`]). The
    /// submission of its last group lets go of it ([`Submitting::submitted_now`]), and so does
    /// its completion.
    pub(super) fn wait_for_replay(
        &self,
        buffer: &Arc<CommandBufferShared>,
        keys: Box<[ResourceKey]>,
        maps: Box<[usize]>,
    ) {
        let mut waiting = lock(&self.waiting);
        waiting.prune();
        if !keys.is_empty() {
            waiting.buffers.push((Arc::downgrade(buffer), keys));
        }
        if !maps.is_empty() {
            waiting.maps.push((Arc::downgrade(buffer), maps));
        }
    }

    /// `buffer` waits for no rebind's replay from here: its last event group was submitted, or
    /// it completed.
    pub(super) fn no_replays(&self, buffer: &CommandBufferShared) {
        let mut waiting = lock(&self.waiting);
        waiting
            .buffers
            .retain(|(waiting, _)| !std::ptr::eq(Weak::as_ptr(waiting), buffer));
        waiting
            .maps
            .retain(|(waiting, _)| !std::ptr::eq(Weak::as_ptr(waiting), buffer));
    }

    /// Reset the rasterization-rate map of identity `map` by `swap`, which puts its new state in
    /// place, under the lock a group's submission reads its mark under: the command buffers
    /// waiting for the submission of an event group whose encoders' passes are over it
    /// ([`Rebinds::wait_for_replay`]) are each marked to encode those encoders again before that
    /// submission (`CommandBufferShared::reencode`), which the caller runs. So a group's
    /// submission came either before the reset (and draws by the map as its passes took it, as
    /// Metal's GPU, under way, read it) or waits for its passes encoded again.
    pub(super) fn reset_map(
        &self,
        map: usize,
        swap: impl FnOnce(),
    ) -> Vec<Arc<CommandBufferShared>> {
        let _submitting = lock(&self.buffers);
        swap();
        let waiting = lock(&self.waiting);
        let mut again: Vec<Arc<CommandBufferShared>> = Vec::new();
        let over = (waiting.maps.iter())
            .filter(|(_, maps)| maps.contains(&map))
            .map(|(buffer, _)| buffer);
        for buffer in over {
            let Some(buffer) = buffer.upgrade() else {
                continue;
            };
            if again.iter().any(|had| Arc::ptr_eq(had, &buffer)) {
                continue;
            }
            buffer.mark_reencode();
            again.push(buffer);
        }
        again
    }

    /// Whether a buffer of the device is detached now (`detach.rs`): a render encoder logs no call
    /// while one is (`render/deferral.rs`).
    pub(super) fn any_detached_now(&self) -> bool {
        self.detached.load(Ordering::Acquire) > 0
    }

    /// Whether one of `keys` names a detached buffer (`detach.rs`): at no cost while none is.
    pub(super) fn any_detached(&self, keys: impl IntoIterator<Item = ResourceKey>) -> bool {
        if self.detached.load(Ordering::Acquire) == 0 {
            return false;
        }
        let buffers = lock(&self.buffers);
        keys.into_iter()
            .any(|key| buffers.get(&key).is_some_and(|entry| entry.detached))
    }

    /// Detach the buffer of `key`, whose hold on its pages is `cell`, to `detached`, unless
    /// submitted work holds it (`SubmittedHolds`): where it lay, the read twins it let go of,
    /// which the caller drops without this lock ([`LeftTwins`]), and whether a submitted command
    /// buffer made those pages resident, or `None` where such a command buffer refuses it.
    /// Already detached, it stays so.
    pub(super) fn detach(
        &self,
        key: ResourceKey,
        cell: &HostCell,
        detached: Arc<Detached>,
    ) -> Option<(Lying, LeftTwins, bool)> {
        let mut buffers = lock(&self.buffers);
        let entry = buffers.get_mut(&key)?;
        if entry.submitted > 0 {
            return None;
        }
        if !entry.detached {
            entry.detached = true;
            self.detached.fetch_add(1, Ordering::AcqRel);
        }
        let resident = entry.resident;
        entry.resident = false;
        let (lying, twins) = cell.replace(Lying::Detached(detached));
        Some((lying, twins, resident))
    }

    /// Wait until no submitted work holds the buffer of `key` any longer (`SubmittedHolds`), with
    /// no other lock of the runtime's held: the completion path lets each hold go without this
    /// one's caller. Settles on a lost device, whose command buffers may never complete, with its
    /// refusal (`Device::lost_refusal`).
    pub(super) fn wait_ran(&self, device: &Device, key: ResourceKey) -> Result<(), Error> {
        let mut buffers = lock(&self.buffers);
        while buffers.get(&key).is_some_and(|entry| entry.submitted > 0) {
            if device.is_lost() {
                return Err(device.lost_refusal());
            }
            buffers = match self.completed.wait_timeout(buffers, LOST_POLL) {
                Ok((buffers, _)) => buffers,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
        Ok(())
    }

    /// Whether submitted work holds the buffer of `key` (`SubmittedHolds`).
    pub(super) fn submitted(&self, key: ResourceKey) -> bool {
        lock(&self.buffers)
            .get(&key)
            .is_some_and(|entry| entry.submitted > 0)
    }

    /// Whether a buffer a command of `taken` named lies elsewhere now than the command took it,
    /// not detached ([`Submitting::rebound`]), taking nothing: what a command buffer's submission
    /// of such a command finds, read before the submission by work of its that records later
    /// (`pending_dispatch.rs`). One load while the device has no buffer over host pages.
    pub(super) fn rebound_now(&self, taken: &Taken) -> bool {
        self.any() && self.submitting().rebound(taken)
    }

    /// The buffers over host pages as a command buffer's batch is submitted, locked: what its
    /// commands took is judged and held at once (`Submitting`), so no detach falls between.
    pub(super) fn submitting(&self) -> Submitting<'_> {
        Submitting {
            buffers: lock(&self.buffers),
        }
    }

    /// The holds of submitted work on `keys` go, its work having completed: a detach may go on,
    /// and one waiting for that work answers.
    fn let_go(&self, keys: &[ResourceKey]) {
        let mut buffers = lock(&self.buffers);
        for key in keys {
            if let Some(entry) = buffers.get_mut(key) {
                entry.submitted = entry.submitted.saturating_sub(1);
            }
        }
        drop(buffers);
        self.completed.notify_all();
    }

    /// Let go of one hold of `key`.
    fn release(&self, key: ResourceKey) {
        if let Some(entry) = lock(&self.buffers).get_mut(&key) {
            entry.pending = entry.pending.saturating_sub(1);
        }
    }

    /// Whether a hold refuses a rebind of `key`.
    fn pending(&self, key: ResourceKey) -> bool {
        lock(&self.buffers)
            .get(&key)
            .is_some_and(|entry| entry.pending > 0)
    }

    /// Give the buffer of `key`, whose hold on its pages is `cell`, `pages` from now on, unless
    /// a hold refuses it, or a gated call under way ([`Rebinds::quiet_calls`]): where it lay (a detached one is attached again), the read twins it let
    /// go of, which the caller drops without this lock ([`LeftTwins`]), and the command
    /// buffers waiting for the submission of an event group whose replayable encoders or work on
    /// the GPU alone named it ([`Rebinds::wait_for_replay`]), each marked under this lock to
    /// encode or record them again before that submission (`CommandBufferShared::reencode`), which
    /// the caller runs; `Err` says why, `pages` dropped once this lock went. Its publish is open
    /// from here until the caller closes it ([`Rebinds::published`]).
    pub(super) fn replace(
        &self,
        key: ResourceKey,
        cell: &HostCell,
        pages: Arc<HostPages>,
    ) -> Result<(Lying, LeftTwins, Vec<Arc<CommandBufferShared>>), Unreplaced> {
        // The publish opens before the lock, once no gated call is under way: none overlaps it.
        if !self.quiet_calls() {
            return Err(Unreplaced::Calls);
        }
        let mut buffers = lock(&self.buffers);
        let Some(entry) = buffers.get_mut(&key) else {
            let (lying, twins) = cell.replace(Lying::In(pages));
            self.unopen(buffers);
            return Ok((lying, twins, Vec::new()));
        };
        // A rebind of it on another thread whose publish is open refuses this one as a hold does,
        // where the caller's contract has none (`Buffer::rebind_bytes_no_copy`).
        if entry.pending > 0 || entry.publishing {
            self.unopen(buffers);
            return Err(Unreplaced::Held);
        }
        if entry.detached {
            entry.detached = false;
            self.detached.fetch_sub(1, Ordering::AcqRel);
        }
        // No command buffer made the new pages resident yet.
        entry.resident = false;
        // What lies over the bytes is the old pages' until the publish closes, which `open`
        // counts from `quiet_calls`.
        entry.publishing = true;
        #[cfg(debug_assertions)]
        PUBLISHES_OPEN.fetch_add(1, Ordering::AcqRel);
        let (lying, twins) = cell.replace(Lying::In(pages));
        // Marked before this lock goes, so a group's submission either came before the rebind
        // (and keeps the old pages) or waits for the encoders encoded again.
        // Each upgraded one goes to the caller, which lets go of it without this lock.
        let again = self.mark_waiting(key);
        Ok((lying, twins, again))
    }
}

/// How often a detach waiting for submitted command buffers looks whether the device was lost.
const LOST_POLL: Duration = Duration::from_millis(20);

/// How long a rebind waits for the gated calls under way to end before it refuses
/// ([`Rebinds::quiet_calls`]): an encoder call takes microseconds, so a call still under way
/// past this waits for something else (a `VkPipeline` being made, a lock another thread holds).
const CALLS_DRAIN: Duration = Duration::from_millis(2);

/// Why [`Rebinds::replace`] refused: as `RebindProblem::EncodedUses` both.
pub(super) enum Unreplaced {
    /// A hold of an encoder or a command buffer not yet committed, or another rebind's open
    /// publish.
    Held,
    /// A gated call under way that did not end within [`CALLS_DRAIN`] ([`CallGate`]).
    Calls,
}

thread_local! {
    /// How deep this thread is in gated calls ([`CallGate`]): a call an encoder's call makes of
    /// its own public calls (a blit's `unlogged`, a loop's copies) is under way already.
    static GATED: Cell<u32> = const { Cell::new(0) };
}

/// A call of an encoder whose calls are gated ([`Holds::gate_calls`]): counted under way while it
/// lives, so no rebind's publish overlaps it ([`Rebinds::quiet_calls`]). Such an encoder is one the
/// runtime may not encode again over the new pages (the module docs, **Other threads**): a compute
/// encoder, any of which a record of command predication of its unit, or a dispatch it leaves to
/// a host step, may make one, and a render encoder that keeps no log. Every call of such an encoder
/// reads where the buffer lay wholly before a publish, or wholly after its close: none raced a
/// rebind, so none fails `EncodedUseRebound` for one. Free where the device has no buffer over host
/// pages, and for every other encoder.
#[must_use = "the call is under way while it lives"]
pub(super) enum CallGate {
    /// Not gated.
    Free,
    /// Inside a gated call of this thread's.
    Nested,
    /// Counted under way on the device.
    Entered(Device),
}

impl CallGate {
    /// A gated call of an encoder of `device` begins, before it reads anything of a buffer over
    /// host pages.
    fn enter(device: &Device) -> CallGate {
        let depth = GATED.get();
        if depth > 0 {
            GATED.set(depth + 1);
            return CallGate::Nested;
        }
        let shared = device.shared();
        if !shared.rebinds.any() {
            return CallGate::Free;
        }
        GATED.set(1);
        if shared.rebinds.enter_call() {
            shared.counts.encoder_call_held_by_rebind();
        }
        CallGate::Entered(device.clone())
    }
}

impl Drop for CallGate {
    fn drop(&mut self) {
        match self {
            CallGate::Free => {}
            CallGate::Nested => GATED.set(GATED.get() - 1),
            CallGate::Entered(device) => {
                GATED.set(GATED.get() - 1);
                device.shared().rebinds.leave_call();
            }
        }
    }
}

/// The buffers over host pages, locked as a command buffer's batch is submitted
/// (`Rebinds::submitting`): each `Taken` is judged against where its buffers lie now and its
/// pages held from now on under the one lock, so a detach either came before (and fails the
/// submission) or waits for the command buffer.
pub(super) struct Submitting<'a> {
    buffers: MutexGuard<'a, KeyMap<ResourceKey, Entry>>,
}

/// What a submission found of what an encoder's commands, or a segment's, took.
#[derive(Clone, Copy, Default)]
pub(super) struct Found {
    /// A buffer lies elsewhere than a command took it: rebound, detached, attached again, or
    /// gone.
    pub(super) moved: bool,
    /// Such a buffer a command named: the submission fails `EncodedUseRebound`, unless one is
    /// detached.
    pub(super) rebound: bool,
    /// A buffer a command took is detached now: the submission fails `DetachedBacking`.
    pub(super) detached: bool,
}

impl Found {
    /// The failure it gives the submission holding it, if any.
    pub(super) fn failure(&self) -> Option<crate::error::CommandBufferError> {
        use crate::error::CommandBufferError;
        match (self.detached, self.rebound) {
            (true, _) => Some(CommandBufferError::DetachedBacking),
            (false, true) => Some(CommandBufferError::EncodedUseRebound),
            (false, false) => None,
        }
    }
}

impl Submitting<'_> {
    /// Whether a buffer a command of `taken` named lies elsewhere than the command took it, not
    /// detached, or a command of it raced a rebind ([`Taken::raced`]): what
    /// [`Submitting::submit`] finds `rebound`, judged without taking anything.
    pub(super) fn rebound(&self, taken: &Taken) -> bool {
        taken.raced
            || taken.pins.iter().any(|(key, pinned)| {
                if !taken.named.contains(key) {
                    return false;
                }
                let entry = self.buffers.get(key);
                let cell = entry.and_then(|entry| entry.cell.upgrade());
                match (pinned, cell.as_ref().map(|cell| cell.lying())) {
                    (_, None) | (_, Some(Lying::Detached(_) | Lying::Gone(_))) => false,
                    (Pinned::Detached, Some(Lying::In(_))) => true,
                    (Pinned::Held(pages), Some(Lying::In(current))) => {
                        !Arc::ptr_eq(pages, &current)
                    }
                    (Pinned::Waiting(pages), Some(Lying::In(current))) => {
                        !std::ptr::eq(Weak::as_ptr(pages), Arc::as_ptr(&current))
                    }
                }
            })
    }

    /// `buffer`'s last event group is submitted now: it waits for no rebind's replay from here
    /// ([`Rebinds::wait_for_replay`]).
    pub(super) fn submitted_now(&self, rebinds: &Rebinds, buffer: &CommandBufferShared) {
        rebinds.no_replays(buffer);
    }

    /// Judge `taken`, and hold every pin whose pages still live in `retained` until the command
    /// buffer completes, counted in `holds` for a detach to wait for. Its named keys stay for the
    /// caller. (Every key was named where the command buffer's root
    /// tables searched their own sets, past the search's capacity, which a set of any count no
    /// longer meets.)
    pub(super) fn submit(
        &mut self,
        taken: &mut Taken,
        retained: &mut Vec<Box<dyn Any + Send + Sync>>,
        holds: &mut SubmittedHolds,
    ) -> Found {
        let mut found = Found::default();
        for (key, pinned) in taken.pins.drain(..) {
            let entry = self.buffers.get_mut(&key);
            let cell = entry.as_ref().and_then(|entry| entry.cell.upgrade());
            let now = cell.as_ref().map(|cell| cell.lying());
            let detached = matches!(now, Some(Lying::Detached(_)));
            // Compared by identity, so a pin whose pages went still compares; neither is the last
            // hold: the cell holds what it lies in, and the buffer its cell.
            let moved = match (&pinned, &now) {
                (_, None | Some(Lying::Gone(_))) => true,
                (Pinned::Detached, Some(Lying::Detached(_))) => false,
                (Pinned::Detached, Some(Lying::In(_))) => true,
                (Pinned::Held(_) | Pinned::Waiting(_), Some(Lying::Detached(_))) => true,
                (Pinned::Held(pages), Some(Lying::In(current))) => !Arc::ptr_eq(pages, current),
                (Pinned::Waiting(pages), Some(Lying::In(current))) => {
                    !std::ptr::eq(Weak::as_ptr(pages), Arc::as_ptr(current))
                }
            };
            found.detached |= detached;
            found.moved |= moved || detached;
            found.rebound |= moved && !detached && taken.named.contains(&key);
            let pages = match pinned {
                Pinned::Held(pages) => Some(pages),
                Pinned::Waiting(pages) => pages.upgrade(),
                Pinned::Detached => None,
            };
            if let (Some(pages), Some(entry)) = (pages, entry) {
                // Pages the command buffer finds its own, Metal makes resident.
                entry.resident |= !moved && !detached;
                entry.submitted += 1;
                holds.keys.push(key);
                retained.push(Box::new(pages));
            }
        }
        // A blit's objects of a releasable import, held until it completes. Gone only with their
        // import, as a rebind or a detach replaced its buffer's pages, which a pin above found
        // moved: the recording that named them runs nothing.
        for hold in taken.imports.drain(..) {
            match hold.retained() {
                Some(held) => retained.push(held),
                None => found.moved = true,
            }
        }
        // A command that raced a rebind of a buffer it took may have read the old pages before
        // its hold (`Taken::raced`): its recording runs nothing, as one whose buffer moved.
        if taken.raced {
            found.moved = true;
            found.rebound |= !found.detached;
        }
        // `cell` and `now` drop here, under the lock: never the last hold of anything (above).
        found
    }
}

/// The holds the commands of a command buffer's submitted batches took on buffers over host
/// pages (`Submitting::submit`), counted per buffer, as Metal holds what a submission uses
/// (`command_buffer/first_submission.rs`): from the submission of the batches' event group until
/// their work completed, where they are dropped. For the command buffer's last group that is as
/// its completion is processed, before its completed handlers run
/// (`CommandBufferShared::complete`), so a detach made from one of them detaches. Until they are
/// dropped a detach answers `false`, and a buffer's waits for the work first.
pub(super) struct SubmittedHolds {
    device: Device,
    keys: Vec<ResourceKey>,
}

impl SubmittedHolds {
    /// None yet, on `device`.
    pub(super) fn new(device: &Device) -> SubmittedHolds {
        SubmittedHolds {
            device: device.clone(),
            keys: Vec::new(),
        }
    }

    /// Whether it counts none.
    pub(super) fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

impl Drop for SubmittedHolds {
    fn drop(&mut self) {
        if !self.keys.is_empty() {
            self.device.shared().rebinds.let_go(&self.keys);
        }
    }
}

/// The holds an encoder's commands, or a command buffer's ended encoders', took on buffers over
/// host pages ([`Rebinds::hold`]): let go when the command buffer commits (or completes, where
/// commands are recorded later), or goes uncommitted.
pub(super) struct Holds {
    device: Device,
    keys: Vec<ResourceKey>,
    /// [`Rebinds::call_begins`] as the encoder's call under way began ([`begin_call`](
    /// Self::begin_call)), or as the holds were made: a hold or pin in the call finds whether
    /// the call raced a rebind of what it takes against it.
    call: u64,
    /// Its encoder's calls are gated ([`CallGate`]).
    gates: bool,
}

impl Holds {
    /// No holds yet, on `device`, its encoder's first call begun now.
    pub(super) fn new(device: &Device) -> Holds {
        #[cfg(debug_assertions)]
        CALL_MOVED.set(super::buffer::host_cells_moved());
        Holds {
            device: device.clone(),
            keys: Vec::new(),
            call: device.shared().rebinds.call_begins(),
            gates: false,
        }
    }

    /// Gate its encoder's calls from its next on ([`CallGate`]): an encoder the runtime may not
    /// encode again.
    pub(super) fn gate_calls(&mut self) {
        self.gates = true;
    }

    /// The encoder's next call begins, before it reads anything of a buffer over host pages (an
    /// encoder's logged call, a render encoder's lowered one, an end): one load, and for a gated
    /// encoder's the call counted under way, which the caller keeps until the call returns.
    pub(super) fn begin_call(&mut self) -> CallGate {
        let gate = match self.gates {
            true => CallGate::enter(&self.device),
            false => CallGate::Free,
        };
        #[cfg(debug_assertions)]
        CALL_MOVED.set(super::buffer::host_cells_moved());
        self.call = self.device.shared().rebinds.call_begins();
        gate
    }

    /// The encoder's call begins as one began at `call` ([`Rebinds::call_begins`]): a render
    /// encoder's open, read before the pass's attachments were.
    pub(super) fn begun_at(&mut self, call: u64) {
        self.call = call;
    }

    /// Hold `key` where it names a buffer over host pages: the pages the commands took, to
    /// retain until the command buffer completes (none while it is detached), with whether the
    /// call under way raced a rebind of it ([`Rebinds::hold`]).
    /// A key whose bit the device's filter of such keys never set names none, told without the
    /// lock (`HostedKeys`).
    pub(super) fn hold(&mut self, key: ResourceKey) -> Option<(Pinned, bool)> {
        let shared = self.device.shared();
        if !shared.rebinds.hosted.may_hold(key) {
            return None;
        }
        shared.counts.thread().host_hold_looked_up();
        #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
        pause::at(pause::Point::Holding, key);
        let taken = shared.rebinds.hold(key, self.call)?;
        self.keys.push(key);
        Some(taken)
    }

    /// Whether it holds nothing.
    pub(super) fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Take `other`'s holds, which it then lets go of nothing of.
    pub(super) fn join(&mut self, mut other: Holds) {
        self.keys.append(&mut other.keys);
    }

    /// What a command an encoder encodes again took of `key` ([`Rebinds::pin`]), held by
    /// nothing here, with whether the call under way raced a rebind of it.
    pub(super) fn pin(&self, key: ResourceKey) -> Option<(Pinned, bool)> {
        self.device.shared().rebinds.pin(key, self.call)
    }
}

/// How many rebinds of the process have their publish open ([`Rebinds::replace`] to
/// [`Rebinds::published`]), for [`call_may_have_raced`]. Debug builds.
#[cfg(debug_assertions)]
static PUBLISHES_OPEN: AtomicUsize = AtomicUsize::new(0);

#[cfg(debug_assertions)]
thread_local! {
    /// `buffer::host_cells_moved` as the encoder call under way on this thread began
    /// ([`Holds::begin_call`]), for [`call_may_have_raced`]. Debug builds.
    static CALL_MOVED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Whether what the encoder call under way on this thread read of buffers over host pages may
/// differ from a reading of them now, as a call that raced a rebind on another thread reads (the
/// module docs, **Other threads**): a no-copy buffer moved since the call began, or a rebind's
/// publish is open. A debug check that compares what the call kept with a reading now
/// (`ArgumentTable::check_reading`) holds only where neither, the call's hold telling a race apart
/// (`Taken::raced`). Debug builds.
#[cfg(debug_assertions)]
pub(super) fn call_may_have_raced() -> bool {
    CALL_MOVED.get() != super::buffer::host_cells_moved()
        || PUBLISHES_OPEN.load(Ordering::Acquire) > 0
}

/// Points a device test holds a thread at, to order an encoder's call against a rebind's
/// publish (`tests::rebind_races`): armed for a buffer's key, each holds the first thread that
/// meets it there, once, until the test lets it go.
#[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
pub(crate) mod pause {
    use super::ResourceKey;
    use std::sync::mpsc::{channel, Receiver, Sender};
    use std::sync::Mutex;

    /// Where a thread is held.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Point {
        /// An encoder about to hold the buffer (`Holds::hold`), after what its call read before.
        Holding,
        /// A rebind whose publish is open: the buffer lies in its new pages, and what lies over
        /// its bytes is not made over them yet (`Device::rebind_host_pages`).
        Publishing,
    }

    /// An armed point: its key, the sender that says a thread reached it, and the receiver it
    /// waits on.
    type Armed = (Point, ResourceKey, Sender<()>, Receiver<()>);

    static ARMED: Mutex<Vec<Armed>> = Mutex::new(Vec::new());

    /// Arm `point` for the buffer of `key`: the receiver hears a thread reach it, and the sender
    /// lets it go.
    pub(crate) fn arm(point: Point, key: ResourceKey) -> (Receiver<()>, Sender<()>) {
        let (reached, hears) = channel();
        let (go, waits) = channel();
        ARMED.lock().unwrap().push((point, key, reached, waits));
        (hears, go)
    }

    /// Hold this thread at `point` for `key`, where a test armed it, until the test lets it go.
    pub(crate) fn at(point: Point, key: ResourceKey) {
        let armed = {
            let mut armed = ARMED.lock().unwrap();
            let at = armed.iter().position(|(p, k, ..)| *p == point && *k == key);
            at.map(|at| armed.remove(at))
        };
        if let Some((_, _, reached, waits)) = armed {
            let _ = reached.send(());
            let _ = waits.recv();
        }
    }
}

/// The pages a command took of a buffer over host pages (`Holds::hold`): held while it is
/// encoded and from its command buffer's first submission until it completes, and only weakly
/// from the commit to that submission, so a rebind or a release meanwhile lets the pages go
/// where nothing submitted holds them. The submission finds them moved (`Taken::moved`).
pub(super) enum Pinned {
    Held(Arc<HostPages>),
    Waiting(Weak<HostPages>),
    /// Taken while the buffer was detached (`detach.rs`): no pages, and the encoder recorded
    /// nothing from that command on (`EncoderCore::skips`).
    Detached,
}

/// What an encoder's commands, or a segment's, took of buffers over host pages: the pages of each,
/// by its key, and the keys a command named. A named key's commands were resolved as they were
/// encoded: a binding's descriptor or its address in a root table's statics, a blit's operand, an
/// attachment, a vertex, index or indirect buffer, a texture over the pages (its views), an
/// indirect command's resolved buffer. A key no command named is reached through the resident set
/// alone (an encoder's `use_resource`, `use_heap`, a bound buffer no command reads), whose ranges
/// the command buffer resolves again as it is first submitted.
#[derive(Default)]
pub(super) struct Taken {
    pins: Vec<(ResourceKey, Pinned)>,
    /// The objects of releasable imports its blits took ([`ImportHold`]).
    imports: Vec<ImportHold>,
    named: BTreeSet<ResourceKey>,
    /// A command named a detached buffer, and the encoder recorded nothing from it on
    /// (`EncoderCore::skips`): its submission never runs.
    dead: bool,
    /// The rasterization-rate map its render pass is over, and the map as the pass took it,
    /// which its draws were planned by: a reset of the map before the submission of its event
    /// group has the encoder encoded again (`rate_mapped_pass.rs`).
    rate_map: Option<(RasterizationRateMap, Arc<MapState>)>,
    /// A command of it raced a rebind of a buffer it took ([`Taken::race`]): what its call read of
    /// the buffer before its hold may be the old pages', so its submission runs it only encoded
    /// again, as one whose buffer moved.
    raced: bool,
}

/// The objects of a releasable import a blit's command took (`Buffer::blit_at`,
/// `host_residency.rs`): held while its encoder records, weakly from the commit until the command
/// buffer's first submission, as its pages are, then until it completes. Held weakly, they keep
/// the device's residency from releasing them (the committed recording names their `VkBuffer`),
/// but not the import from going with its buffer's pages (a rebind, a detach), as Metal holds
/// nothing of a command buffer not yet submitted: their free, and the deallocator after it, run
/// in that call, and the recording that named them never runs (its pin of the pages moved).
pub(super) enum ImportHold {
    Held(Arc<Imported>),
    Waiting(Weak<Imported>),
}

impl ImportHold {
    /// Held weakly from now on.
    fn wait(&mut self) {
        match self {
            ImportHold::Held(held) => *self = ImportHold::Waiting(Arc::downgrade(held)),
            ImportHold::Waiting(_) => {}
        }
    }

    /// The objects, held until the command buffer completes: `None` where they went.
    fn retained(self) -> Option<Box<dyn Any + Send + Sync>> {
        match self {
            ImportHold::Held(held) => Some(Box::new(Retained::new(held))),
            ImportHold::Waiting(held) => held
                .upgrade()
                .map(|held| Box::new(Retained::new(held)) as Box<dyn Any + Send + Sync>),
        }
    }
}

/// A command buffer's hold of a releasable import's objects until it completes
/// ([`ImportHold::retained`]): where its completion's release lets go of them last, the import
/// gone and a deallocator or a keep-alive waiting for their free, the free goes to the device's
/// free thread (`free_thread.rs`, `Imported::let_go_at_completion`).
struct Retained(std::mem::ManuallyDrop<Arc<Imported>>);

impl Retained {
    fn new(held: Arc<Imported>) -> Retained {
        Retained(std::mem::ManuallyDrop::new(held))
    }
}

impl Drop for Retained {
    fn drop(&mut self) {
        // SAFETY: taken once, here, as the hold goes, and never read again.
        let held = unsafe { std::mem::ManuallyDrop::take(&mut self.0) };
        if let Some(imported) = Arc::into_inner(held) {
            Imported::let_go_at_completion(imported);
        }
    }
}

impl Taken {
    /// Note `hold`, the objects of a releasable import a blit took.
    pub(super) fn hold_import(&mut self, hold: ImportHold) {
        self.imports.push(hold);
    }

    /// Note `pinned`, what a command took of the buffer of `key`.
    pub(super) fn pin(&mut self, key: ResourceKey, pinned: Pinned) {
        self.pins.push((key, pinned));
    }

    /// Note that a command named `key`.
    pub(super) fn name(&mut self, key: ResourceKey) {
        self.named.insert(key);
    }

    /// The keys a command named.
    pub(super) fn named(&self) -> &BTreeSet<ResourceKey> {
        &self.named
    }

    /// Whether a command named a buffer over host pages ([`named_pins`](Self::named_pins) holds
    /// one).
    pub(super) fn names_a_pin(&self) -> bool {
        self.pins.iter().any(|(key, _)| self.named.contains(key))
    }

    /// The keys of the buffers over host pages a command named, sorted, each once.
    pub(super) fn named_pins(&self) -> Vec<ResourceKey> {
        let mut keys: Vec<ResourceKey> = self
            .pins
            .iter()
            .map(|(key, _)| *key)
            .filter(|key| self.named.contains(key))
            .collect();
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    /// Join `other`'s pins and names.
    pub(super) fn join(&mut self, other: Taken) {
        self.pins.extend(other.pins);
        self.named.extend(other.named);
        self.dead |= other.dead;
        self.raced |= other.raced;
        if self.rate_map.is_none() {
            self.rate_map = other.rate_map;
        }
    }

    /// A command of it raced a rebind of a buffer it took: the call that encoded it began before
    /// the rebind's publish closed ([`Rebinds::hold`]).
    pub(super) fn race(&mut self) {
        self.raced = true;
    }

    /// Whether a command of it raced a rebind ([`race`](Self::race)).
    pub(super) fn raced(&self) -> bool {
        self.raced
    }

    /// Note that the encoder's render pass is over `map`, which it took as `state`.
    pub(super) fn take_rate_map(&mut self, map: RasterizationRateMap, state: Arc<MapState>) {
        self.rate_map = Some((map, state));
    }

    /// Whether a rebind or a reset may have the encoder encoded again: a command named a buffer
    /// over host pages, its pass is over a rasterization-rate map, or a command raced a rebind.
    pub(super) fn encodes_again(&self) -> bool {
        self.names_a_pin() || self.rate_map.is_some() || self.raced
    }

    /// The identity of the rasterization-rate map its pass is over (`Rebinds::reset_map`).
    pub(super) fn rate_map_id(&self) -> Option<usize> {
        self.rate_map.as_ref().map(|(map, _)| map.id())
    }

    /// Whether its pass took a rasterization-rate map since reset: its draws were planned by a
    /// map Metal's GPU no longer reads (`rate_mapped_pass.rs`).
    pub(super) fn rate_map_reset(&self) -> bool {
        self.rate_map
            .as_ref()
            .is_some_and(|(map, state)| !map.holds(state))
    }

    /// The encoder records nothing more: a command named a detached buffer.
    pub(super) fn die(&mut self) {
        self.dead = true;
    }

    /// Whether the encoder recorded nothing from some command on (`Taken::die`).
    pub(super) fn dead(&self) -> bool {
        self.dead
    }

    /// Hold every pin weakly: its command buffer committed and waits for its first submission,
    /// its commands recorded already.
    pub(super) fn wait(&mut self) {
        for (_, pinned) in &mut self.pins {
            if let Pinned::Held(pages) = pinned {
                *pinned = Pinned::Waiting(Arc::downgrade(pages));
            }
        }
        for hold in &mut self.imports {
            hold.wait();
        }
    }

    /// Its named keys let go, once judged (`Submitting::submit`).
    pub(super) fn judged(&mut self) {
        self.named.clear();
    }

    /// Hand every pin's pages that still live to `retained`, upgraded, which the command buffer
    /// holds until it completes and releases without its lock: as it is first submitted, and as
    /// a segment recorded later is placed or let go. Nothing is let go here but weak pins, so no
    /// pages go (the caller holds locks).
    pub(super) fn retain_into(&mut self, retained: &mut Vec<Box<dyn Any + Send + Sync>>) {
        let live = self.pins.drain(..).filter_map(|(_, pinned)| match pinned {
            Pinned::Held(pages) => Some(pages),
            Pinned::Waiting(pages) => pages.upgrade(),
            Pinned::Detached => None,
        });
        retained.extend(live.map(|pages| Box::new(pages) as Box<dyn Any + Send + Sync>));
        retained.extend(self.imports.drain(..).filter_map(ImportHold::retained));
        self.named.clear();
    }
}

impl Drop for Holds {
    fn drop(&mut self) {
        let rebinds = &self.device.shared().rebinds;
        for &key in &self.keys {
            rebinds.release(key);
        }
    }
}

impl Buffer {
    /// Give the no-copy buffer other host memory of its length from now on (a runtime extension;
    /// the module docs of `rebind.rs` state the rule): `memory` ([`HostMemory`]), under its
    /// constructor's contract for the buffer's [`length`](Buffer::length), imported as
    /// [`Device::new_buffer_with_bytes_no_copy`] imports memory. The buffer, its `gpu_address` and
    /// its options stay; `contents` answers `memory.pointer()` from now on. Command buffers first
    /// submitted before the call keep the old memory until they complete; one committed before it
    /// and first submitted after reaches the new memory through what it made resident and what its
    /// commands reach unnamed, and through what its compute, blit and render encoders named, which
    /// this call encodes again; it fails `CommandBufferError::EncodedUseRebound` where an encoder
    /// this call cannot encode again named the buffer (one of a command buffer that used command
    /// predication, among others); commands encoded after it use the new. The old memory's
    /// deallocator runs once every command buffer submitted before the call that used it completed
    /// (in the call, where none holds it), and that of the memory a detach let go of in the call,
    /// as Metal's replace runs it; `deallocator` runs for the new memory at the next rebind's
    /// turn, or when the buffer goes. A refused call runs neither and changes nothing.
    /// `keep_alive` ([`KeepAlive`]) is dropped once the device's import of the new memory is freed:
    /// with one, `deallocator` runs at that turn though something still holds the import (a command
    /// buffer submitted before a texture over the bytes was released); with none, `deallocator`
    /// waits for the import (`host-deallocator-delayed`).
    ///
    /// A texture over the buffer (a shadow of its bytes, or a texture buffer) reads the new
    /// memory from its next use, and one whose linear image aliases its bytes has its image made
    /// again over the new memory before the call returns; nothing of the old contents is copied.
    ///
    /// Where the caller's gate refuses the new pages' import
    /// ([`HostMemory::with_import_gate`](crate::HostMemory::with_import_gate)), or the import the
    /// device and the gate admitted fails (out of memory, another Vulkan error), the buffer is
    /// given a copy of the new pages in memory of the device's own instead, its `contents` the new
    /// pointer, announced (`copied-host-pages`, `copied.rs`), an image in place over its bytes made
    /// again over the copy. New pages the process maps read-only (Metal's
    /// `replaceBackingWithRanges:readOnly:`) are copied so too, and the copy drops the GPU's
    /// writes, as Metal drops them. New memory at another offset in its page than the old, where a texture
    /// over the buffer lives, is given such a copy laid out at the old offset, the gate unasked.
    /// Metal rebinds over any memory; the copy is the difference.
    ///
    /// A `Managed` no-copy buffer on a device without unified memory is two copies
    /// ([`HostBacking::DeviceCopy`](crate::HostBacking::DeviceCopy)): the call gives its host copy
    /// the new memory, imported as at the making but never a copy of it (where the import is not
    /// made, the pages are reached through staging buffers), and copies the new pages into the
    /// GPU's copy by the device's next submission, before any of its commands, so every command
    /// buffer committed after the call reads them; nothing is refused for uses encoded before it.
    /// The old memory's deallocator runs once nothing holds its host copy (in the call, where
    /// nothing does). Over pages the process maps read-only, the GPU's writes are dropped:
    /// `synchronize_resource` copies nothing out to them, and they are copied in again at the
    /// completion of each command buffer that used the buffer.
    ///
    /// Refused: a buffer that is not a no-copy buffer of its own (`Invalid`,
    /// [`RebindProblem::NotHostMemory`]); a command buffer not yet committed whose commands used
    /// the buffer, or an open encoder that did (a render encoder with an attachment over it, from
    /// its open), or one committed whose commands used it that the runtime records once earlier
    /// work completed (an indirect command buffer's executions the GPU writes, a dispatch left to
    /// its host step, a predicated loop's run again of such work or of work on the GPU alone),
    /// until it completed (`Invalid`, [`RebindProblem::EncodedUses`]), where Metal's replace
    /// before its submission reaches the new pages; new memory at another offset
    /// in its page than the old where a texture over the buffer lives and the buffer's GPU writes
    /// are dropped (`NotImplemented(RebindTextures)`), or where the device cannot make a texture's
    /// aliasing image again over the new memory or its copy (`NotImplemented(RebindTextures)`,
    /// which says when); memory the device cannot import exactly, as
    /// `new_buffer_with_bytes_no_copy` refuses it; a lost device.
    ///
    /// # Safety
    ///
    /// `memory` meets the contract of the [`HostMemory`] constructor that made it
    /// ([`HostMemory::new`], [`HostMemory::with_dma_buf`]) for the buffer's length. The old memory
    /// stays mapped until its keep-alive is dropped, else until its deallocator runs, or with
    /// neither until every command buffer first submitted before the call completed. No other
    /// thread rebinds or detaches the buffer, or makes a texture over its bytes, during the call.
    ///
    /// Other threads may encode and commit command buffers that use the buffer during the call.
    /// A blit encoder's, or a render encoder's that keeps a log, whose call raced it (held the
    /// buffer while the call was making what lies over its bytes again, or after, having begun
    /// before) is encoded again over the new memory before its first submission (the module docs
    /// of `rebind.rs`, **Other threads**, counted by `WorkCounts::encoders_raced_rebind`); it
    /// fails `CommandBufferError::EncodedUseRebound` at that submission only where its command
    /// buffer's commit fails it, or a call that answers otherwise than at its encoding refuses its
    /// encoding again. The calls of the encoders the runtime may not encode again (every compute
    /// encoder's, its predicated ones and those with a GPU-written indirect dispatch among them,
    /// and a render encoder's that opened while the device had no buffer over host pages) never
    /// race it: the call waits up to 2 ms for such a call under way to end, and is refused
    /// otherwise (`Invalid`, [`RebindProblem::EncodedUses`], counted by
    /// `WorkCounts::rebinds_refused_calls_in_flight`), never failing that encoder later; such a
    /// call that begins during it waits for it, so it reads the old memory or the new one, whole.
    pub unsafe fn rebind_bytes_no_copy(
        &self,
        memory: HostMemory,
        deallocator: Option<Deallocator>,
        keep_alive: Option<KeepAlive>,
    ) -> Result<(), Error> {
        const CALL: &str = "rebind_bytes_no_copy";
        let Some(cell) = self.host_cell() else {
            // A two-copy buffer's host copy takes the new pages, copied in whole
            // (`managed_copy.rs`).
            if let Some(caller) = self.caller_copy() {
                // SAFETY: the caller's contract.
                return unsafe { caller.rebind(self, memory, deallocator, keep_alive) };
            }
            return Err(invalid(CALL, Problem::Rebind(RebindProblem::NotHostMemory)));
        };
        // SAFETY: the caller's contract.
        let rebound = unsafe {
            self.generation().rebind_host_pages(
                CALL,
                self,
                cell,
                memory,
                HostRelease::new(deallocator, keep_alive),
                None,
            )
        };
        // The buffer a pending command buffer may hold has the program's objects now.
        program_objects::settle();
        rebound
    }
}

impl Heap {
    /// Give the heap over the caller's host memory (`Device::new_heap_with_bytes_no_copy`) other
    /// host memory of its size from now on (a runtime extension), as
    /// [`Buffer::rebind_bytes_no_copy`] gives a no-copy buffer: `memory` ([`HostMemory`]), its
    /// first byte at a multiple of the import's unit as the heap's was. Every buffer and texture
    /// placed in the heap stays, at its offset, with its `gpu_address` and resource ID; a buffer's
    /// `contents` answers its offset from `memory.pointer()` from now on. A texture bound to the
    /// heap's bytes has its image made again over the new memory, at its offset, before the call
    /// returns, as has a texture over one of the heap's buffers whose linear image aliases its
    /// bytes; a texture held as a shadow of the heap's bytes, a texture buffer and a shadowed
    /// texture over one of the heap's buffers read them from their next use. Nothing of the old
    /// contents is copied. Where the caller's gate refuses the new memory's import, or the import
    /// it admitted fails, the heap is given a copy of the new memory instead, as
    /// [`Buffer::rebind_bytes_no_copy`]'s buffer is, its bound textures' images made again over the
    /// copy, announced (`copied-host-pages`).
    ///
    /// Refused: a heap not over the caller's host memory (`Invalid`,
    /// [`RebindProblem::NotHostMemory`]); a command buffer not yet committed whose commands used
    /// the heap or a resource placed in it, or an open encoder that did (a render encoder with an
    /// attachment in it, from its open), or one committed as [`Buffer::rebind_bytes_no_copy`]
    /// names, until it completed (`Invalid`, [`RebindProblem::EncodedUses`]); memory whose
    /// first byte is off the import's unit (`Invalid`, `HeapProblem::HostMemoryAlignment`); memory the device cannot import exactly;
    /// an image the device cannot make again there (as `new_texture_at_offset` refuses it); a
    /// lost device.
    ///
    /// # Safety
    ///
    /// As [`Buffer::rebind_bytes_no_copy`]'s, for the heap's [`size`](Heap::size) and the uses
    /// of the heap and its resources: no other thread rebinds the heap, detaches one of its
    /// resources' bytes, or places a resource in it, during the call. Other threads may encode and
    /// commit command buffers that use the heap or its resources during the call, with the outcomes
    /// [`Buffer::rebind_bytes_no_copy`]'s **Safety** names: an encoder the runtime may not encode
    /// again never races it, its call under way refusing the rebind (`EncodedUses`) where it does
    /// not end within 2 ms, and a blit or logged render encoder that raced it is encoded again.
    pub unsafe fn rebind_bytes_no_copy(
        &self,
        memory: HostMemory,
        deallocator: Option<Deallocator>,
        keep_alive: Option<KeepAlive>,
    ) -> Result<(), Error> {
        const CALL: &str = "rebind_bytes_no_copy";
        let Some((own, cell)) = self
            .bytes()
            .and_then(|own| own.host_cell().map(|cell| (own, cell)))
        else {
            return Err(invalid(CALL, Problem::Rebind(RebindProblem::NotHostMemory)));
        };
        // SAFETY: the caller's contract.
        let rebound = unsafe {
            self.generation().rebind_host_pages(
                CALL,
                own,
                cell,
                memory,
                HostRelease::new(deallocator, keep_alive),
                Some(self),
            )
        };
        // The heap a pending command buffer may hold has the program's objects now.
        program_objects::settle();
        rebound
    }
}

impl Device {
    /// Whether a hold refuses a rebind of the buffer of `key` (`Rebinds::pending`).
    pub(super) fn rebind_pending(&self, key: ResourceKey) -> bool {
        self.shared().rebinds.pending(key)
    }
}
