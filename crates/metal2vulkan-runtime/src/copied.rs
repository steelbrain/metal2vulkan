//! Copied host buffers: a no-copy buffer whose import of the caller's pages the caller's gate
//! refused ([`HostMemory::with_import_gate`](crate::HostMemory)), or whose import the gate and
//! the device admitted and that then failed (out of memory, another Vulkan error, or the runtime's
//! refusal of pages the kernel cannot pin on a driver whose failed import of them breaks it,
//! `RuntimeDriverDefect::UnpinnablePagesBreakHostMemory`, that no dma-buf of them served), is made
//! all the same over a copy of its bytes (`Device::buffer_over_host_memory`), announced
//! (`copied-host-pages`). No import maps the pages, so they take no kernel mapping.
//!
//! - **The copy.** A `VkBuffer` of the buffer's length in host-visible, coherent memory of the
//!   runtime's own ([`MemoryClass::Shared`](crate::memory::MemoryClass)), which the GPU uses; the
//!   buffer's `contents` stays the caller's pointer, as Metal's is. Made over pages the process
//!   reads ([`read_caller`]): the first copy reads them through the kernel ([`read_pages`]), and
//!   where the kernel fails the read, with plain loads where the process maps every page with
//!   read permission ([`mapped_readable`]): on Linux the kernel reads no page it cannot pin, such
//!   as a mapping of a dma-buf (`VM_PFNMAP`). Pages not mapped, or mapped without read permission
//!   (memory Metal makes no buffer over), keep the import's own refusal.
//! - **Coherence.** The runtime keeps, for each copy, the bytes the pages and the copy last agreed
//!   on (its snapshot). At each commit, before any of the command buffer's work is submitted,
//!   every byte the CPU changed in the pages since then is written into the copy
//!   ([`CopiedBuffers::copy_in`]). At each batch's completion, before its events are seen
//!   signaled, the scheduler is told, its handlers run or a waiter returns, every byte the GPU
//!   changed in the copy is written back into the pages, where the CPU did not change that byte
//!   since ([`CopiedBuffers::merge_back`]): a byte both changed keeps the CPU's, which the next
//!   commit copies in. The completion compares only the copies its batches may have written: one
//!   whose buffer's (or heap's) key a use of their command buffers' commands wrote, and every
//!   copy where one of those may have written memory no use names (a function's loaded
//!   addresses, an indirect command buffer's executions; `Written`), or kept its writes not all
//!   apart. Counted: `WorkCounts::copied_merges_compared`, `copied_merges_skipped`. Bytes are compared a page (4 KiB) at a time, and inside a page that
//!   differs 8 at a time and then one at a time, so a merge writes no byte the GPU did not change
//!   ([`sync`]). Each batch ends with a barrier that makes its writes visible to the host
//!   (`tracker::Barrier::to_host`). Where the pages' memory came with a log of the writes to them
//!   (`HostMemory::with_dirty_pages`, or its guest RAM's, `HostRam::with_dirty_pages`), each merge
//!   back reports the blocks it wrote to the log as the copy's own writes
//!   (`DirtyPages::wrote_as`, a writer of the copy's own) before it returns: the log sees no
//!   write the runtime makes, and every other copy and shadow of the pages takes them.
//! - **Copies in the log gates.** Where the pages' memory came with such a log, a commit's copy
//!   in compares only the pages the log reports written since the copy's last copy in, and the
//!   bytes no log keeps (a guest RAM's runs that miss some of them): a stamp of each log is asked
//!   at the commit, once for every copy, its publication waited for (bounded) with no lock of
//!   the copies' or the command buffer's held, and the copy synced to it
//!   ([`CopiedBuffers::copy_in`]); every write made before the commit is in that publication,
//!   the GPU's in place through other imports of the pages and the runtime's own CPU writes among
//!   them, which the runtime reports (`dirty_pages.rs`). A copy compares every byte at its first
//!   copy in after its watch began, and where a log cannot say. Counted:
//!   `WorkCounts::copied_copy_in_bytes_skipped`.
//! - **Pages the process cannot write** (read-only pages, as a consumer's alias of a read-only
//!   replace, `replaceBackingWithRanges:readOnly:`, maps them), asked of the system as the copy is
//!   made ([`mapped_writable`]): the GPU's writes to the copy are dropped, as Metal drops the GPU's
//!   writes to read-only memory (`read_only_view.rs`). At each batch's completion every byte the
//!   GPU changed in the copy takes back the byte the pages and the copy last agreed on
//!   ([`CopiedBuffers::merge_back`]), and nothing is written to the pages: later commands of the
//!   same batch read the write back, and a later command buffer the pages' bytes. The CPU's writes
//!   to the pages through another mapping are copied in at each commit, as above. Counted:
//!   `WorkCounts::copied_dropped_bytes`.
//! - **Event signals.** A batch's event signals, submitted while a copy is live, are seen at its
//!   completion turn alone, after the merge: no read of the value observes them earlier, and a
//!   wait for them is held on the host until that turn (`event_timeline.rs`), so the GPU's writes
//!   before a signal are in the pages once its value is seen, as Metal's are. No submission is
//!   added; a reader sees the value a completion turn later than the semaphore's counter shows it.
//! - **Cost.** A copy is compared whole at each batch's completion that may have written it, and
//!   at each commit where its pages carry no log: O(bytes) per such batch, and per commit without
//!   a log, a fallback's cost (`docs/FAST-PATHS.md`, known gaps). A command buffer whose
//!   functions read addresses may write every copy. A page that agrees costs one read of each
//!   side and a slice comparison: 680 MiB of copies that agree took 28-30 ms each way on an M3
//!   Max, 52-53 ms on a Core Ultra 9 275HX; their logged copy in took 0.15 ms (0.3) where no page
//!   changed, and 4.6 ms (3.5) where 1% of the pages changed on each side; a merge back of no
//!   copy written, none (`tests/copied_bench.rs`). Counted:
//!   `WorkCounts::copied_copy_in_bytes`, `WorkCounts::copied_merge_back_bytes`,
//!   `WorkCounts::copied_dropped_bytes`.
//! - **Heaps and rebinds.** A no-copy heap's own buffer is copied as a buffer is, where the heap's
//!   textures could lie, with a heap's hazard identity; its buffers lie in the copy at their
//!   offsets, and its textures are shadows of its bytes (`AliasRoute::MemoryShadow` or Metal's
//!   tiles). A buffer's or heap's rebind gives it a copy of its new pages laid out as their import
//!   would be (`Device::copied_pages`), so texel views planned at the bytes' offset lie where they
//!   lay, and at the old offset where the bytes moved in their page under a texture; the heap's
//!   textures bound to its bytes and the images in place over the buffer are made again over the
//!   copy: in fresh pages the runtime imported where an image in place rides on it, made as over
//!   the caller's pages; else linear images of its memory that start `PREINITIALIZED`.
//! - **What is not copied.** Memory whose GPU writes Metal's private option bit 24 drops imports as
//!   before. A texture made over a copied buffer is a shadow of its bytes (`shadowed-texture`).

use crate::device::lock;
use crate::dirty_pages::{new_writer, DirtySource, DirtyTally, PagesLog};
use crate::host_memory::{mapped_readable, mapped_writable};
use crate::key_hash::KeyMap;
use crate::resource::ResourceKey;
use crate::work_counts::Counters;
use crate::DirtyStamp;
use std::num::NonZeroU64;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// A device's live copied host buffers, compared at each commit and each batch's completion.
#[derive(Default)]
pub(crate) struct CopiedBuffers {
    entries: Mutex<Vec<Entry>>,
    /// How many are live: a commit or completion looks no further where none is.
    live: AtomicUsize,
    /// The next copy's id.
    next: AtomicU64,
    /// A batch finished with no merge back (a failed wait): the next merge back compares every
    /// copy, whatever its batches wrote.
    owed: AtomicBool,
}

/// What the batches a completion turn finishes may have written ([`CopiedBuffers::merge_back`]):
/// the keys their command buffers' commands wrote (`DirtyCommit::copies_written`), or every copy.
#[derive(Default)]
pub(crate) struct Written {
    keys: KeyMap<ResourceKey, ()>,
    all: bool,
}

impl Written {
    /// Every copy may have been written.
    pub(crate) fn all() -> Written {
        Written {
            keys: KeyMap::default(),
            all: true,
        }
    }

    /// A command wrote the bytes of `key`.
    pub(crate) fn wrote(&mut self, key: ResourceKey) {
        if !self.all {
            self.keys.insert(key, ());
        }
    }

    /// Some command may have written a copy no use of it names.
    pub(crate) fn wrote_all(&mut self) {
        self.all = true;
        self.keys.clear();
    }

    /// Whether a command may have written the copy whose uses take `keys`.
    fn meets(&self, keys: &[ResourceKey]) -> bool {
        self.all || keys.iter().any(|key| self.keys.contains_key(key))
    }
}

/// Why a copy was not made ([`CopiedBuffers::add`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NotCopied {
    /// A page is not mapped, or mapped without read permission: memory Metal makes no buffer over.
    Unreadable,
    /// The host has no memory for the copy's snapshot.
    OutOfHostMemory,
}

/// A copy and the caller's pages it mirrors.
struct Entry {
    id: u64,
    /// The caller's bytes (the buffer's `contents`).
    pages: NonNull<u8>,
    /// The copy's bytes, mapped.
    copy: NonNull<u8>,
    /// The bytes the pages and the copy last agreed on.
    snapshot: Box<[u8]>,
    /// The process cannot write the pages: the GPU's writes to the copy are dropped, never merged
    /// back.
    drops_writes: bool,
    /// The logs of the writes to the pages, where their memory came with one, and the writer its
    /// merges back are reported as ([`Copied::set_log`]).
    log: Option<CopyLog>,
    /// The keys whose uses' writes reach the copy: the buffer's, or the heap's bytes' (whose
    /// buffers' and textures' uses carry it), the buffer's again after a rebind
    /// ([`Copied::name`]).
    keys: Vec<ResourceKey>,
}

/// A copy's logs of its pages, and its writer in them.
struct CopyLog {
    pages: Arc<PagesLog>,
    writer: NonZeroU64,
    /// Its watches of the pages in each log, held while the copy lives.
    _watches: Vec<Box<dyn Send + Sync>>,
    /// The copy's bytes no log keeps the writes to: every copy in compares them.
    unlogged: Arc<[Range<u64>]>,
    /// The stamps, one of each log in [`PagesLog::sources`]' order, as of which the copy holds
    /// every byte the CPU wrote in the pages: a copy in compares only the pages the logs report
    /// written since. `None` until a copy in compared every byte after the watches began, and
    /// after one some log could not answer for.
    since: Option<Arc<[DirtyStamp]>>,
    /// The bytes its merges back wrote into the pages since `since`: at [`ASKS_APART`] or more,
    /// its copy in asks the logs for the pages any writer but it wrote
    /// ([`DirtyPages::changed_except`](crate::DirtyPages)), where a copy otherwise takes the
    /// pages every writer wrote, asked once per log and stamp for every copy, and compares again
    /// the pages it merged.
    merged: u64,
}

/// The bytes a copy's merges back wrote since its last copy in from which its copy in asks the
/// logs apart, for every writer's pages but its own (one more question of each log, which a
/// consumer answers by a pass over its watches), rather than compare its own writes again
/// (about 40 µs a MiB on an M3 Max). Both are exact.
const ASKS_APART: u64 = 1 << 20;

// SAFETY: the pointers name the caller's pages, which stay mapped while the buffer's import lives
// (`HostMemory::new`), and the copy's mapped memory, which the import holds; both are reached only
// under the registry's lock, which the import's drop takes before either goes ([`Copied`]).
unsafe impl Send for Entry {}

/// A live copy's place in its device's [`CopiedBuffers`]: dropped (unregistered) before the copy's
/// memory is freed, so no comparison reads it after (`HostImport`'s drop).
pub(crate) struct Copied {
    buffers: Arc<CopiedBuffers>,
    id: u64,
    drops_writes: bool,
}

impl Copied {
    /// Whether the GPU's writes to the copy are dropped: the process cannot write the pages.
    pub(crate) fn drops_writes(&self) -> bool {
        self.drops_writes
    }

    /// Keep `log`, the logs of the writes to the copy's pages (`dirty_pages::PagesLog`), with a
    /// writer of its own: each merge back reports the bytes it wrote to them as that writer's
    /// ([`CopiedBuffers::merge_back`]). Set once.
    ///
    /// Its pages are watched in each log from here on ([`DirtyPages::watch`](crate::DirtyPages)),
    /// asked with no lock of the copies' held: its copies in compare every byte until one did
    /// after the watches began ([`CopiedBuffers::copy_in`]).
    pub(crate) fn set_log(&self, log: &Arc<PagesLog>) {
        let Some(writer) = new_writer() else {
            return;
        };
        let watches = log.watch();
        let unlogged: Arc<[Range<u64>]> = log.unlogged().into();
        let mut entries = lock(&self.buffers.entries);
        let entry = entries.iter_mut().find(|entry| entry.id == self.id);
        let unkept = match entry {
            Some(entry) if entry.log.is_none() => {
                entry.log = Some(CopyLog {
                    pages: Arc::clone(log),
                    writer,
                    _watches: watches,
                    unlogged,
                    since: None,
                    merged: 0,
                });
                None
            }
            _ => Some(watches),
        };
        // A watch's end is the consumer's call: made with no lock of the copies' held.
        drop(entries);
        drop(unkept);
    }

    /// The uses of `key` (a buffer's or a heap's bytes' lying in the copy) write the copy: a
    /// completion turn merges it back only where a batch it finishes wrote one of its keys, or
    /// may have written memory its uses do not name ([`CopiedBuffers::merge_back`]).
    pub(crate) fn name(&self, key: ResourceKey) {
        let mut entries = lock(&self.buffers.entries);
        if let Some(entry) = entries.iter_mut().find(|entry| entry.id == self.id) {
            if !entry.keys.contains(&key) {
                entry.keys.push(key);
            }
        }
    }
}

impl Drop for Copied {
    fn drop(&mut self) {
        let mut entries = lock(&self.buffers.entries);
        let removed = match entries.iter().position(|entry| entry.id == self.id) {
            Some(index) => {
                self.buffers.live.fetch_sub(1, Ordering::Relaxed);
                Some(entries.swap_remove(index))
            }
            None => None,
        };
        // Its watches end with no lock of the copies' held: a watch's end is the consumer's call.
        drop(entries);
        drop(removed);
    }
}

impl CopiedBuffers {
    /// Whether a copy is live: a batch submitted now may write one, so its event signals are seen
    /// at its completion turn alone (`event_timeline.rs`).
    pub(crate) fn any(&self) -> bool {
        self.live.load(Ordering::Relaxed) > 0
    }

    /// Register the copy at `copy` of the `length` bytes at `pages`, its bytes read from the pages
    /// now ([`read_caller`]: through the kernel, or where it fails, with plain loads of pages
    /// mapped readable). Where the process cannot write every page ([`mapped_writable`]), the
    /// GPU's writes to the copy are dropped (the module docs). Refused, registering nothing, where
    /// the process cannot read every page, or the host has no memory for the bytes the copy and
    /// the pages last agreed on.
    ///
    /// # Safety
    ///
    /// `copy` is `length` bytes of mapped memory nothing else uses yet, which stays mapped until
    /// the answer is dropped; `pages` the caller's bytes, which stay mapped as long
    /// (`HostMemory::new`), and which the CPU reads without a fault wherever the process maps
    /// them with read permission.
    pub(crate) unsafe fn add(
        self: &Arc<Self>,
        pages: NonNull<u8>,
        copy: NonNull<u8>,
        length: usize,
    ) -> Result<Copied, NotCopied> {
        let read = |pages: NonNull<u8>, into: &mut [u8]| {
            // SAFETY: the caller's bytes, read with plain loads only where the process maps them
            // readable (the caller's contract).
            unsafe { read_caller(pages, into, read_pages, mapped_readable) }
        };
        let start = pages.as_ptr() as u64;
        let drops_writes = !mapped_writable(start, start.saturating_add(length as u64));
        // SAFETY: the caller's.
        unsafe { self.add_reading(pages, copy, length, read, drops_writes) }
    }

    /// [`CopiedBuffers::add`], the first bytes of the pages read by `read`, which answers whether
    /// it read every one; the GPU's writes to the copy dropped where `drops_writes`.
    ///
    /// # Safety
    ///
    /// As [`CopiedBuffers::add`]'s.
    pub(crate) unsafe fn add_reading(
        self: &Arc<Self>,
        pages: NonNull<u8>,
        copy: NonNull<u8>,
        length: usize,
        read: impl FnOnce(NonNull<u8>, &mut [u8]) -> bool,
        drops_writes: bool,
    ) -> Result<Copied, NotCopied> {
        let mut snapshot = Vec::new();
        if snapshot.try_reserve_exact(length).is_err() {
            return Err(NotCopied::OutOfHostMemory);
        }
        snapshot.resize(length, 0u8);
        let mut snapshot = snapshot.into_boxed_slice();
        if !read(pages, &mut snapshot) {
            return Err(NotCopied::Unreadable);
        }
        // SAFETY: the copy is `length` bytes nothing else uses yet (the caller's).
        unsafe { std::ptr::copy_nonoverlapping(snapshot.as_ptr(), copy.as_ptr(), length) };
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        lock(&self.entries).push(Entry {
            id,
            pages,
            copy,
            snapshot,
            drops_writes,
            log: None,
            keys: Vec::new(),
        });
        self.live.fetch_add(1, Ordering::Relaxed);
        Ok(Copied {
            buffers: Arc::clone(self),
            id,
            drops_writes,
        })
    }

    /// At a commit: write into each copy the bytes the CPU changed in its pages since they last
    /// agreed, before any of the command buffer's work is submitted.
    ///
    /// A copy whose pages' memory came with a log of the writes to them ([`Copied::set_log`])
    /// compares only the pages the log reports written since the stamp its last copy in synced
    /// to, and the bytes no log keeps: a stamp of each log is asked now, once for every copy
    /// (`DirtyPages::request`), and its publication waited for (`DirtyPages::wait`, bounded),
    /// with no lock of the copies' held; the pages written since are asked once per log and
    /// stamp for every copy (`DirtyPages::changed`), but by a copy whose merges back wrote
    /// [`ASKS_APART`] bytes or more of the pages since, which asks for every writer's but its
    /// own (`DirtyPages::changed_except`).
    /// Every write made before this call is in a publication the wait found, so the copy takes
    /// each. Every byte is compared where the copy has no earlier copy in to count on, or a log
    /// answered `false`. `tally` counts the stamps and waits (`WorkCounts::dirty_stamps_asked`
    /// and kin).
    pub(crate) fn copy_in(&self, counts: &Counters, tally: &DirtyTally) {
        if self.live.load(Ordering::Relaxed) == 0 {
            return;
        }
        let asked: Vec<Asked> = (lock(&self.entries).iter())
            .filter_map(|entry| {
                let log = entry.log.as_ref()?;
                Some(Asked {
                    id: entry.id,
                    length: entry.snapshot.len() as u64,
                    pages: Arc::clone(&log.pages),
                    writer: log.writer,
                    unlogged: Arc::clone(&log.unlogged),
                    since: log.since.clone(),
                    merged: log.merged,
                })
            })
            .collect();
        let plans = planned(asked, tally);
        let (mut moved, mut skipped) = (0u64, 0u64);
        for entry in lock(&self.entries).iter_mut() {
            let plan = match &entry.log {
                Some(_) => (plans.binary_search_by_key(&entry.id, |(id, _)| *id).ok())
                    .map(|at| &plans[at].1),
                None => None,
            };
            let (pages, copy, length) = (entry.pages, entry.copy, entry.snapshot.len());
            match plan.and_then(|plan| plan.bytes.as_ref()) {
                Some(bytes) => {
                    let mut compared = 0u64;
                    for range in bytes {
                        let (start, end) = (range.start as usize, range.end as usize);
                        // SAFETY: the entry's pointers are live while it is registered
                        // (`Entry`), and `start..end` lies in its bytes (`blocks`).
                        moved += unsafe {
                            sync(
                                NonNull::new_unchecked(pages.as_ptr().add(start)),
                                NonNull::new_unchecked(copy.as_ptr().add(start)),
                                &mut entry.snapshot[start..end],
                                false,
                            )
                        };
                        compared += range.end - range.start;
                    }
                    skipped += length as u64 - compared;
                }
                // SAFETY: the entry's pointers are live while it is registered (`Entry`).
                None => moved += unsafe { sync(pages, copy, &mut entry.snapshot, false) },
            }
            if let (Some(log), Some(plan)) = (&mut entry.log, plan) {
                log.since = plan.stamps.clone();
                log.merged = 0;
            }
        }
        counts.copied_copied_in(moved);
        counts.copied_copy_in_skipped(skipped);
    }

    /// At a batch's completion: write back into each copy's pages the bytes the GPU changed in the
    /// copy since they last agreed, where the CPU did not change them since, before anything sees
    /// the batch complete. A copy of pages the process cannot write takes back instead the bytes
    /// they last agreed on wherever the GPU changed it: the GPU's writes are dropped, and nothing
    /// is written to the pages.
    ///
    /// Only the copies the finished batches may have written are compared (`written`): one a
    /// use of their command buffers' commands wrote by one of its keys ([`Copied::name`]), and
    /// every copy where a command may have written memory no use names (its function's loaded
    /// addresses, an indirect command buffer's executions), or where its command buffer's writes
    /// were not all kept apart. Each batch's writes are done by its own completion turn, which
    /// compares what they reach; a copy no batch wrote since its last comparison holds what its
    /// pages do, but for the CPU's writes, which the next commit copies in. After a batch finished
    /// with no merge (a failed wait, [`CopiedBuffers::owe`]), the next compares every copy.
    /// Counted: `WorkCounts::copied_merges_compared`, `copied_merges_skipped`.
    ///
    /// The bytes written into pages whose memory came with a log are reported to it, as the
    /// copy's own writes ([`DirtyPages::wrote_as`](crate::DirtyPages::wrote_as)), a block of
    /// [`SYNC_BLOCK`] bytes at a time, before this returns: the log sees no write the runtime
    /// makes, and every other copy and shadow of those pages must take them, while the copy's
    /// own copies in need not.
    pub(crate) fn merge_back(&self, counts: &Counters, written: &Written) {
        if self.live.load(Ordering::Relaxed) == 0 {
            return;
        }
        let owed = self.owed.swap(false, Ordering::AcqRel);
        let (mut moved, mut dropped) = (0u64, 0u64);
        let (mut compared, mut skipped) = (0u64, 0u64);
        let mut reports = Vec::new();
        for entry in lock(&self.entries).iter_mut() {
            if !owed && !written.meets(&entry.keys) {
                skipped += 1;
                continue;
            }
            compared += 1;
            let (copy, pages) = (entry.copy, entry.pages);
            match (entry.drops_writes, &mut entry.log) {
                // SAFETY: the entry's copy is live while it is registered (`Entry`).
                (true, _) => dropped += unsafe { restore(copy, &entry.snapshot) },
                (false, Some(log)) => {
                    let mut blocks = Vec::new();
                    // SAFETY: the entry's pointers are live while it is registered (`Entry`).
                    moved +=
                        unsafe { sync_noting(copy, pages, &mut entry.snapshot, true, &mut blocks) };
                    if !blocks.is_empty() {
                        let wrote: u64 = blocks.iter().map(|bytes| bytes.end - bytes.start).sum();
                        log.merged = log.merged.saturating_add(wrote);
                        reports.push((Arc::clone(&log.pages), log.writer, blocks));
                    }
                }
                // SAFETY: as above.
                (false, None) => moved += unsafe { sync(copy, pages, &mut entry.snapshot, true) },
            }
        }
        // Reported with no lock of the copies' held.
        for (log, writer, blocks) in reports {
            for bytes in blocks {
                log.wrote_as(bytes, writer);
            }
        }
        counts.copied_merged_back(moved);
        counts.copied_dropped(dropped);
        counts.copied_merges(compared, skipped);
    }

    /// A batch finished with no merge back (its queue's wait failed): the next merge back
    /// compares every copy.
    pub(crate) fn owe(&self) {
        self.owed.store(true, Ordering::Release);
    }
}

/// What a logged copy asks its logs at a copy in ([`CopiedBuffers::copy_in`]).
struct Asked {
    id: u64,
    length: u64,
    pages: Arc<PagesLog>,
    writer: NonZeroU64,
    unlogged: Arc<[Range<u64>]>,
    since: Option<Arc<[DirtyStamp]>>,
    merged: u64,
}

/// A logged copy's copy in, as its logs answered: the bytes it compares (`None`: every byte),
/// and the stamps it syncs to (`None`: a log could not say).
struct Plan {
    bytes: Option<Vec<Range<u64>>>,
    stamps: Option<Arc<[DirtyStamp]>>,
}

/// One stamp of a log at a copy in, and whether its publication was found.
struct Stamped {
    source: Arc<DirtySource>,
    stamp: DirtyStamp,
    published: bool,
}

/// The copies in of the copies `asked`, by their ids, in order: a stamp of each log asked, then
/// waited for (counted in `tally`); the pages each log reports written since a copy's stamp
/// asked once per log and stamp (or for one copy, of a copy whose merge back wrote its pages),
/// with no lock of the copies' held.
fn planned(asked: Vec<Asked>, tally: &DirtyTally) -> Vec<(u64, Plan)> {
    let mut stamped: Vec<Stamped> = Vec::new();
    for source in asked.iter().flat_map(|asked| asked.pages.sources()) {
        if !stamped.iter().any(|s| Arc::ptr_eq(&s.source, source)) {
            stamped.push(Stamped {
                source: Arc::clone(source),
                stamp: source.request(tally),
                published: false,
            });
        }
    }
    for stamp in &mut stamped {
        stamp.published = stamp.source.wait(stamp.stamp, tally);
    }
    // The pages each log reports written since a stamp, by every writer: asked once per log and
    // stamp.
    let mut written: Vec<(*const DirtySource, DirtyStamp, Vec<Range<u64>>)> = Vec::new();
    let mut plans: Vec<(u64, Plan)> = Vec::with_capacity(asked.len());
    for asked in asked {
        let of: Vec<&Stamped> = (asked.pages.sources())
            .filter_map(|source| stamped.iter().find(|s| Arc::ptr_eq(&s.source, source)))
            .collect();
        let now: Option<Vec<DirtyStamp>> = (of.iter())
            .map(|stamped| stamped.published.then_some(stamped.stamp))
            .collect();
        // Copies over the same logs sync to one set of stamps.
        let stamps: Option<Arc<[DirtyStamp]>> =
            now.map(
                |now| match plans.last().and_then(|(_, plan)| plan.stamps.as_ref()) {
                    Some(last) if last[..] == now[..] => Arc::clone(last),
                    _ => now.into(),
                },
            );
        let bytes = match (&asked.since, &stamps) {
            (Some(since), Some(_)) if since.len() == of.len() => {
                let mut bytes: Vec<Range<u64>> = asked.unlogged.to_vec();
                let mut own = Vec::new();
                for (index, (stamped, since)) in of.iter().zip(since.iter()).enumerate() {
                    let changed: &[Range<u64>] = match asked.merged >= ASKS_APART {
                        true => {
                            own.clear();
                            let writer = Some(asked.writer);
                            stamped.source.changed(*since, writer, &mut own);
                            joined(&mut own);
                            &own
                        }
                        false => {
                            let key = (Arc::as_ptr(&stamped.source), *since);
                            let at = match (written.iter()).position(|w| (w.0, w.1) == key) {
                                Some(at) => at,
                                None => {
                                    let mut runs = Vec::new();
                                    stamped.source.changed(*since, None, &mut runs);
                                    joined(&mut runs);
                                    written.push((key.0, key.1, runs));
                                    written.len() - 1
                                }
                            };
                            &written[at].2
                        }
                    };
                    asked.pages.bytes_of(index, changed, |run| bytes.push(run));
                }
                Some(blocks(bytes, asked.length))
            }
            _ => None,
        };
        plans.push((asked.id, Plan { bytes, stamps }));
    }
    plans.sort_by_key(|(id, _)| *id);
    plans
}

/// `runs` sorted by start, those that meet or touch joined.
fn joined(runs: &mut Vec<Range<u64>>) {
    runs.retain(|run| run.start < run.end);
    runs.sort_by_key(|run| run.start);
    let mut kept: Vec<Range<u64>> = Vec::with_capacity(runs.len());
    for run in runs.drain(..) {
        match kept.last_mut() {
            Some(last) if run.start <= last.end => last.end = last.end.max(run.end),
            _ => kept.push(run),
        }
    }
    *runs = kept;
}

/// `bytes` of a copy of `length` bytes, each widened to the [`SYNC_BLOCK`]s it meets (counted
/// from the copy's first byte) and cut to the copy, sorted and joined.
fn blocks(mut bytes: Vec<Range<u64>>, length: u64) -> Vec<Range<u64>> {
    let block = SYNC_BLOCK as u64;
    for run in &mut bytes {
        let start = run.start / block * block;
        let end = run.end.div_ceil(block).saturating_mul(block).min(length);
        *run = start.min(end)..end;
    }
    joined(&mut bytes);
    bytes
}

/// The bytes [`sync`] reads of `from` at once and compares whole with the snapshot's: a host page.
pub(crate) const SYNC_BLOCK: usize = 4096;

/// Write into `to` each byte of `from` that differs from `snapshot`, and take it into `snapshot`;
/// where `keep_to`, a byte of `to` that differs from `snapshot` already (written since they last
/// agreed) is kept, though `snapshot` takes `from`'s, so the next copy the other way carries it.
/// The bytes written.
///
/// Read a block of [`SYNC_BLOCK`] bytes at a time, each read of `from` once and compared whole
/// with the snapshot's (slice equality, which the compiler vectorizes); only a block that differs
/// is looked into, 8 bytes at a time and then byte by byte, so no byte the comparison found equal
/// is written ([`merge_block`]).
///
/// # Safety
///
/// `from` and `to` are `snapshot.len()` bytes of mapped memory each.
pub(crate) unsafe fn sync(
    from: NonNull<u8>,
    to: NonNull<u8>,
    snapshot: &mut [u8],
    keep_to: bool,
) -> u64 {
    // SAFETY: the caller's.
    unsafe { sync_blocks(from, to, snapshot, keep_to, |_| {}) }
}

/// [`sync`], noting in `blocks` the bytes of each block it wrote a byte of, adjacent blocks
/// joined, in order.
///
/// # Safety
///
/// As [`sync`]'s.
pub(crate) unsafe fn sync_noting(
    from: NonNull<u8>,
    to: NonNull<u8>,
    snapshot: &mut [u8],
    keep_to: bool,
    blocks: &mut Vec<Range<u64>>,
) -> u64 {
    let note = |block: Range<u64>| match blocks.last_mut() {
        Some(last) if last.end == block.start => last.end = block.end,
        _ => blocks.push(block),
    };
    // SAFETY: the caller's.
    unsafe { sync_blocks(from, to, snapshot, keep_to, note) }
}

/// [`sync`], telling `wrote` the bytes of each block it wrote a byte of.
///
/// # Safety
///
/// As [`sync`]'s.
unsafe fn sync_blocks(
    from: NonNull<u8>,
    to: NonNull<u8>,
    snapshot: &mut [u8],
    keep_to: bool,
    mut wrote: impl FnMut(Range<u64>),
) -> u64 {
    let (from, to) = (from.as_ptr(), to.as_ptr());
    let mut written = 0u64;
    let mut block = [0u8; SYNC_BLOCK];
    let mut at = 0;
    while at < snapshot.len() {
        let span = (snapshot.len() - at).min(SYNC_BLOCK);
        let now = &mut block[..span];
        // SAFETY: `span` bytes at `at` lie in `from` (the caller's); `block` is this call's own.
        unsafe { std::ptr::copy_nonoverlapping(from.add(at), now.as_mut_ptr(), span) };
        let was = &mut snapshot[at..at + span];
        if now != was {
            // SAFETY: `span` bytes at `at` lie in `to` (the caller's).
            let merged = unsafe { merge_block(now, to.add(at), was, keep_to) };
            if merged > 0 {
                wrote(at as u64..(at + span) as u64);
            }
            written += merged;
        }
        at += span;
    }
    written
}

/// [`sync`] inside one block whose bytes `now` (read of `from`) differ from the snapshot's `was`:
/// 8 bytes at a time, and in each 8 that differ, byte by byte, each byte of `now` that differs
/// from `was` written into `to` (where `keep_to`, unless `to`'s byte differs from `was` already)
/// and taken into `was`: eight in one store where all eight differ (and, where `keep_to`, none
/// of `to`'s does), so no byte that did not change is written. The bytes written.
///
/// # Safety
///
/// `to` is `now.len()` bytes of mapped memory, `was` as long.
unsafe fn merge_block(now: &[u8], to: *mut u8, was: &mut [u8], keep_to: bool) -> u64 {
    let mut written = 0u64;
    let mut at = 0;
    while at < now.len() {
        let span = (now.len() - at).min(8);
        let (now, was) = (&now[at..at + span], &mut was[at..at + span]);
        if now != was {
            let mut there = [0u8; 8];
            if keep_to {
                // SAFETY: `span` bytes at `at` lie in `to` (the caller's).
                unsafe { std::ptr::copy_nonoverlapping(to.add(at), there.as_mut_ptr(), span) };
            }
            // Eight bytes that all changed, none of them written in `to` since: the same bytes
            // the loop below writes, in one store.
            let every = now.iter().zip(was.iter()).all(|(now, was)| now != was);
            if let (Ok(word), true) = (<[u8; 8]>::try_from(now), every) {
                if !keep_to || there[..] == was[..] {
                    // SAFETY: as above; `[u8; 8]` is aligned to 1.
                    unsafe { to.add(at).cast::<[u8; 8]>().write_volatile(word) };
                    written += 8;
                    was.copy_from_slice(now);
                    at += span;
                    continue;
                }
            }
            for byte in 0..span {
                if now[byte] == was[byte] {
                    continue;
                }
                if !keep_to || there[byte] == was[byte] {
                    // SAFETY: as above.
                    unsafe { to.add(at + byte).write_volatile(now[byte]) };
                    written += 1;
                }
                was[byte] = now[byte];
            }
        }
        at += span;
    }
    written
}

/// Write into `copy` each byte of `snapshot` that `copy` differs from: the GPU's writes to a copy
/// of pages the process cannot write, dropped. The bytes written.
///
/// # Safety
///
/// `copy` is `snapshot.len()` bytes of mapped memory.
pub(crate) unsafe fn restore(copy: NonNull<u8>, snapshot: &[u8]) -> u64 {
    let copy = copy.as_ptr();
    let mut written = 0u64;
    let mut at = 0;
    while at < snapshot.len() {
        let span = (snapshot.len() - at).min(8);
        let mut now = [0u8; 8];
        // SAFETY: `span` bytes at `at` lie in `copy` (the caller's).
        unsafe { std::ptr::copy_nonoverlapping(copy.add(at), now.as_mut_ptr(), span) };
        if now[..span] != snapshot[at..at + span] {
            for byte in 0..span {
                if now[byte] != snapshot[at + byte] {
                    // SAFETY: as above.
                    unsafe { copy.add(at + byte).write_volatile(snapshot[at + byte]) };
                    written += 1;
                }
            }
        }
        at += span;
    }
    written
}

/// Read the caller's `into.len()` bytes at `pages` into `into`: through the kernel (`kernel`,
/// [`read_pages`]), and where it fails, with plain loads where every page of them is mapped with
/// read permission (`readable`, asked for the bytes' addresses, [`mapped_readable`]). Whether
/// they were read: `false` where the kernel failed and a page is not mapped readable.
///
/// The kernel's read is a test that faults on nothing, but on Linux it pins each page, and no
/// page of a `VM_PFNMAP` or `VM_IO` mapping can be pinned: a mapping of a dma-buf (udmabuf's on
/// Linux 7.2, measured) is mapped readable and fails it all the same. The plain loads are the
/// access a copy's comparisons make at each commit ([`sync`]).
///
/// # Safety
///
/// `pages` is `into.len()` bytes the CPU reads without a fault wherever `readable` answers `true`
/// for them.
pub(crate) unsafe fn read_caller(
    pages: NonNull<u8>,
    into: &mut [u8],
    kernel: impl FnOnce(NonNull<u8>, &mut [u8]) -> bool,
    readable: impl FnOnce(u64, u64) -> bool,
) -> bool {
    if kernel(pages, into) {
        return true;
    }
    let start = pages.as_ptr() as u64;
    if !readable(start, start + into.len() as u64) {
        return false;
    }
    // SAFETY: the bytes are mapped readable (the caller's); `into` is the runtime's own, apart.
    unsafe { std::ptr::copy_nonoverlapping(pages.as_ptr(), into.as_mut_ptr(), into.len()) };
    true
}

/// Read the caller's `into.len()` bytes at `pages` into `into` through the kernel, which answers a
/// page the process cannot read (unmapped, or mapped without read permission) with a failure, not a
/// fault: `false` then. On Linux `process_vm_readv` of the process's own pages; on macOS
/// `mach_vm_read_overwrite` of its own task's; elsewhere nothing is read and the answer is `false`.
#[cfg(target_os = "linux")]
pub(crate) fn read_pages(pages: NonNull<u8>, into: &mut [u8]) -> bool {
    let mut done = 0;
    while done < into.len() {
        let local = libc::iovec {
            iov_base: into[done..].as_mut_ptr().cast(),
            iov_len: into.len() - done,
        };
        let remote = libc::iovec {
            iov_base: pages.as_ptr().wrapping_add(done).cast(),
            iov_len: into.len() - done,
        };
        // SAFETY: the local vector names `into`'s bytes, which the call writes; the remote one is
        // read through the kernel, which answers an unreadable page with an error.
        let read = unsafe { libc::process_vm_readv(libc::getpid(), &local, 1, &remote, 1, 0) };
        if read <= 0 {
            return false;
        }
        done += read as usize;
    }
    true
}

/// [`read_pages`] on macOS.
#[cfg(target_os = "macos")]
pub(crate) fn read_pages(pages: NonNull<u8>, into: &mut [u8]) -> bool {
    unsafe extern "C" {
        /// `<mach/mach_init.h>`: the port of this process's task.
        static mach_task_self_: libc::mach_port_t;
        fn mach_vm_read_overwrite(
            target_task: libc::vm_map_t,
            address: libc::mach_vm_address_t,
            size: libc::mach_vm_size_t,
            data: libc::mach_vm_address_t,
            outsize: *mut libc::mach_vm_size_t,
        ) -> libc::kern_return_t;
    }
    let mut read = 0;
    // SAFETY: the call writes `into`'s bytes alone, and answers an unreadable page of the source
    // with an error.
    let answer = unsafe {
        mach_vm_read_overwrite(
            mach_task_self_,
            pages.as_ptr() as u64,
            into.len() as u64,
            into.as_mut_ptr() as u64,
            &mut read,
        )
    };
    answer == libc::KERN_SUCCESS && read == into.len() as u64
}

/// [`read_pages`] where the system is not asked: nothing is copied.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub(crate) fn read_pages(_pages: NonNull<u8>, _into: &mut [u8]) -> bool {
    false
}
