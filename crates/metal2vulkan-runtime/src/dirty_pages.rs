//! A consumer's log of the host pages written in a no-copy object's memory (a runtime extension:
//! Metal has none), by which every copy of those pages into memory of the device's own copies in
//! only what changed since its last copy ([`HostMemory::with_dirty_pages`]).
//!
//! A consumer that logs every write to its pages (a VMM, from its hypervisor's dirty log) says
//! which pages changed through [`DirtyPages`], given with the memory. The runtime keeps it with
//! every import made from that memory, on every device, discrete and integrated alike.
//!
//! - **The stamps.** A [`DirtyStamp`] names one of the source's publications of its log. The
//!   runtime asks for a stamp at a command buffer's commit ([`DirtyPages::request`]) where one of
//!   its copies takes the pages again, and waits for that publication as the batch that holds
//!   the copy is submitted, which records the copy then (`shadow::GatedCopyIn`,
//!   [`DirtyPages::wait`]); a copied host buffer (`copied.rs`), whose copy in is the commit's
//!   own, asks and waits at the commit, one stamp of each log for every copy. A copy then syncs
//!   to that stamp: every write made after it is stamped later.
//! - **The pages changed** ([`DirtyPages::changed_except`]) since the stamp a copy synced to, by
//!   any writer but that copy's own, are the only bytes it may lack: they are copied in, the rest
//!   skipped. Copied host buffers ask once per log and stamp for all of them, every writer's
//!   ([`DirtyPages::changed`]), but one whose merges back wrote a MiB or more since, which asks
//!   apart for every writer's but its own.
//! - **The runtime's own writes** into the pages (a shadow's copy back or merge, a copied host
//!   buffer's merge back, `copied.rs`) are reported ([`DirtyPages::wrote_as`]), naming the copy
//!   as their writer, so other copies of the same
//!   pages see them as changed and the copy's own copies in, under a log that keeps writers
//!   apart, do not. So are the GPU's writes into pages imported in place, which no log of the
//!   host's sees, and the runtime's CPU writes into the pages ([`DirtyPages::wrote`]).
//! - **The GPU's writes in place** are those a command buffer's commands name (their uses, of a
//!   buffer over the pages, a heap's buffers or textures, a texture over a buffer's bytes: a
//!   texture's use names its whole buffer or heap, and so does a command buffer that wrote more
//!   than 64 ranges of one), reported at its commit, again before anything sees each of its
//!   batches complete, and at its completion ([`DirtyCommit::report_host_writes`]), but for the
//!   bytes its shadows' copies back report as their own. Metal's contract has a command reach a
//!   resource only through a binding or a residency it named.
//! - **Correctness first.** A wait that answers `false`, a source that cannot say, and a copy
//!   that holds nothing it may count on all copy every byte.
//!
//! [`HostMemory::with_dirty_pages`]: crate::HostMemory::with_dirty_pages

use super::copied::Written;
use super::device::lock;
use super::key_hash::KeyMap;
use super::resource::ResourceKey;
use super::shadow::ShadowCopies;
use super::tracker::{Use, UseRange};
use ash::vk;
use std::fmt;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// One publication of a [`DirtyPages`] source's log, in the order the source publishes them,
/// which every source over the same host bytes shares: a later publication's stamp is larger.
/// Opaque to the runtime, which only compares and hands back stamps a source gave it. No Metal
/// counterpart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DirtyStamp(pub u64);

/// A consumer's log of the writes to the host pages of a no-copy object's memory, which lets
/// every copy the runtime makes of those pages into memory of the device's own copy in only the
/// pages written since its last copy ([`HostMemory::with_dirty_pages`]). No Metal counterpart:
/// Metal's GPU reads a no-copy object's pages in place.
///
/// **Offsets** are bytes from the memory's pointer ([`HostMemory::pointer`]), the first byte
/// the consumer gave. A page that starts before the pointer is reported from 0; a range reported
/// past the bytes the runtime reads is cut to them.
///
/// **Its promises.** It logs at a granularity of whole pages, of up to 256 KiB. A page counts as
/// changed for any stamp earlier than the start of the log's continuous watch over it
/// ([`watch`](Self::watch)), which the first watch over the page begins, whatever source asks;
/// a page no one watches any more counts as changed for every stamp until one watches it again.
/// A write the host makes (the consumer's own, and those [`wrote`](Self::wrote) reports) is
/// stamped at once. The sources over the same host bytes share one order of stamps: each answers
/// [`changed`](Self::changed) for a stamp another of them gave as for its own. A closed source
/// (the consumer's reset, pause or stop) answers [`wait`](Self::wait) with `false`, so the
/// runtime takes every page as changed.
///
/// **Writers** (optional: both methods default to the plain ones). The runtime reports a copy's
/// own copy back or merge (a shadow's, or a copied host buffer's merge back of the GPU's writes,
/// `copied.rs`) naming the copy as its writer ([`wrote_as`](Self::wrote_as)), and asks
/// for that copy's copy in which pages changed by any writer but it
/// ([`changed_except`](Self::changed_except)): it already holds what it wrote, so a page only it
/// wrote since is no change to it. A writer is a [`NonZeroU64`] no other writer in the process
/// is ever given; 0 is the consumer's own writes and every write reported through
/// [`wrote`](Self::wrote), which no copy's question leaves out. The sources over the same host
/// bytes share their writers as their stamps. One way to keep it: per page, its latest stamp and
/// that write's writer, and the latest stamp of any other writer; `changed_except(since, w)`
/// counts the page where its latest writer is not `w` and its latest stamp is later than
/// `since`, or where it is and the other writers' latest stamp is.
///
/// **Threads.** Called on the runtime's threads, never under a lock the source's own calls
/// take: [`watch`](Self::watch) where a copy of the pages is made, [`request`](Self::request) on
/// the thread that commits a command buffer, [`wait`](Self::wait), [`changed`](Self::changed)
/// and [`changed_except`](Self::changed_except) on the thread that records its commands for the
/// GPU, and for copied host buffers on the thread that commits it, [`wrote`](Self::wrote) and
/// [`wrote_as`](Self::wrote_as) at the commit and at the completion. A [`Watch`](Self::Watch) may be dropped on a completion thread, so its drop takes
/// no lock the consumer holds while it calls the runtime.
///
/// [`HostMemory::with_dirty_pages`]: crate::HostMemory::with_dirty_pages
/// [`HostMemory::pointer`]: crate::HostMemory::pointer
pub trait DirtyPages: Send + Sync {
    /// A watch of a range of the memory's pages, ended by its drop.
    type Watch: Send + Sync;

    /// Starts watching the memory's bytes `range` (offsets from its pointer), for the watch's
    /// life: the consumer logs only watched pages, so a page no copy reads costs the guest
    /// nothing. [`changed`](Self::changed) reports watched pages alone; an unwatched page reads
    /// as changed at every stamp. The runtime watches the bytes of each copy it keeps, for the
    /// copy's life.
    fn watch(&self, range: Range<u64>) -> Self::Watch;

    /// A stamp a future publication reaches: every write made before this call shows in
    /// [`changed`](Self::changed) once [`wait`](Self::wait) on it returns `true`. Never blocks.
    /// The wait for it comes before the command buffer's commands are recorded: a publication
    /// that needs that command buffer's own work never comes within it.
    fn request(&self) -> DirtyStamp;

    /// Block until publication `stamp` exists, bounded (about 4 ms). `false`: the source cannot
    /// say (or is closed), so the runtime takes every page as changed (a whole copy in).
    fn wait(&self, stamp: DirtyStamp) -> bool;

    /// Append to `out` the byte runs, as offsets from the memory's pointer, of the watched pages
    /// stamped later than `since`, and of those whose continuous watch began after it.
    fn changed(&self, since: DirtyStamp, out: &mut Vec<Range<u64>>);

    /// The runtime wrote the memory's bytes `range` (offsets from its pointer) itself: from the
    /// GPU (a shadow's copy back, a merge, a copied host buffer's merge back, a command's write
    /// into the pages imported in place, at the commit and again at the completion) or from the
    /// CPU (a `replaceRegion` of a texture over them or of a heap's texture held in them, an
    /// argument encoder's write, `copyParameterDataToBuffer`), so other copies of the same pages
    /// see it as changed.
    fn wrote(&self, range: Range<u64>);

    /// [`wrote`](Self::wrote), naming its writer: the copy whose copy back or merge it is (the
    /// trait's **Writers**). By default, `wrote`.
    fn wrote_as(&self, range: Range<u64>, writer: NonZeroU64) {
        let _ = writer;
        self.wrote(range);
    }

    /// [`changed`](Self::changed), leaving out a page whose every write stamped later than
    /// `since` was `writer`'s ([`wrote_as`](Self::wrote_as)); a page another writer wrote since
    /// (the consumer among them), or one whose watch began after `since`, still counts (the
    /// trait's **Writers**). By default, `changed`.
    fn changed_except(&self, since: DirtyStamp, writer: NonZeroU64, out: &mut Vec<Range<u64>>) {
        let _ = writer;
        self.changed(since, out);
    }
}

/// A [`DirtyPages`] source of any watch type, as the runtime keeps it: its watches boxed.
pub(crate) trait DirtyHook: Send + Sync {
    /// [`DirtyPages::watch`], boxed.
    fn watch(&self, range: Range<u64>) -> Box<dyn Send + Sync>;
    /// [`DirtyPages::request`].
    fn request(&self) -> DirtyStamp;
    /// [`DirtyPages::wait`].
    fn wait(&self, stamp: DirtyStamp) -> bool;
    /// [`DirtyPages::wrote`].
    fn wrote_as_consumer(&self, range: Range<u64>);
    /// [`DirtyPages::wrote_as`].
    fn wrote_as(&self, range: Range<u64>, writer: NonZeroU64);
    /// [`DirtyPages::changed`].
    fn changed(&self, since: DirtyStamp, out: &mut Vec<Range<u64>>);
    /// [`DirtyPages::changed_except`].
    fn changed_except(&self, since: DirtyStamp, writer: NonZeroU64, out: &mut Vec<Range<u64>>);
}

/// A consumer's source, shared.
struct Hooked<D: ?Sized>(Arc<D>);

impl<D> DirtyHook for Hooked<D>
where
    D: DirtyPages + ?Sized,
    D::Watch: 'static,
{
    fn watch(&self, range: Range<u64>) -> Box<dyn Send + Sync> {
        Box::new(self.0.watch(range))
    }
    fn request(&self) -> DirtyStamp {
        self.0.request()
    }
    fn wait(&self, stamp: DirtyStamp) -> bool {
        self.0.wait(stamp)
    }
    fn wrote_as_consumer(&self, range: Range<u64>) {
        self.0.wrote(range);
    }
    fn wrote_as(&self, range: Range<u64>, writer: NonZeroU64) {
        self.0.wrote_as(range, writer);
    }
    fn changed(&self, since: DirtyStamp, out: &mut Vec<Range<u64>>) {
        self.0.changed(since, out);
    }
    fn changed_except(&self, since: DirtyStamp, writer: NonZeroU64, out: &mut Vec<Range<u64>>) {
        self.0.changed_except(since, writer, out);
    }
}

/// `source`, as the runtime keeps it.
pub(crate) fn hook<D>(source: Arc<D>) -> Arc<dyn DirtyHook>
where
    D: DirtyPages + ?Sized + 'static,
    D::Watch: 'static,
{
    Arc::new(Hooked(source))
}

impl fmt::Debug for dyn DirtyHook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DirtyPages")
    }
}

/// A consumer's [`DirtyPages`] as the runtime keeps it with the pages it came with: the source,
/// the host address its offsets count from, and the latest publication a wait on it found, so a
/// wait on an earlier one is not made again.
pub(crate) struct DirtySource {
    hook: Arc<dyn DirtyHook>,
    /// The memory's pointer, offset 0 of the source.
    pointer: u64,
    /// One past the latest stamp a wait returned `true` for; 0 for none.
    published: AtomicU64,
    /// One past the latest stamp a wait returned `false` for; 0 for none.
    unanswered: AtomicU64,
    /// One past the latest stamp asked of it ([`DirtySource::request`]); 0 for none.
    requested: AtomicU64,
}

impl DirtySource {
    /// `hook`, whose offsets count from `pointer`.
    pub(crate) fn new(hook: Arc<dyn DirtyHook>, pointer: u64) -> Arc<DirtySource> {
        Arc::new(DirtySource {
            hook,
            pointer,
            published: AtomicU64::new(0),
            unanswered: AtomicU64::new(0),
            requested: AtomicU64::new(0),
        })
    }

    /// [`DirtyPages::request`], counted in `tally`.
    pub(crate) fn request(&self, tally: &DirtyTally) -> DirtyStamp {
        let stamp = self.hook.request();
        self.requested
            .fetch_max(stamp.0.saturating_add(1), Ordering::AcqRel);
        tally.stamps.fetch_add(1, Ordering::Relaxed);
        stamp
    }

    /// The source's offset of the host address `address`; `None` below its pointer.
    pub(crate) fn offset_of(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.pointer)
    }

    /// Whether publication `stamp` exists: at once where a wait on it or a later one returned
    /// `true`, `false` at once where one on it or a later one returned `false`; else
    /// [`DirtyPages::wait`]'s answer for the latest stamp asked of the source, which every
    /// commit's stamp asked so far is at or below, so one wait serves every command buffer
    /// committed before it (counted in `tally`).
    pub(crate) fn wait(&self, stamp: DirtyStamp, tally: &DirtyTally) -> bool {
        if self.published.load(Ordering::Acquire) > stamp.0 {
            return true;
        }
        if self.unanswered.load(Ordering::Acquire) > stamp.0 {
            return false;
        }
        let latest = self.requested.load(Ordering::Acquire).saturating_sub(1);
        let target = DirtyStamp(latest.max(stamp.0));
        let started = std::time::Instant::now();
        let published = self.hook.wait(target);
        let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        tally.waits.fetch_add(1, Ordering::Relaxed);
        tally.wait_nanos.fetch_add(nanos, Ordering::Relaxed);
        let answered = match published {
            true => &self.published,
            false => {
                tally.waits_unanswered.fetch_add(1, Ordering::Relaxed);
                &self.unanswered
            }
        };
        answered.fetch_max(target.0.saturating_add(1), Ordering::AcqRel);
        published
    }

    /// [`DirtyPages::wrote`] of the source's offsets `range` by the runtime's own CPU (a
    /// `replaceRegion` of a texture over the pages, writer 0), where non-empty.
    pub(crate) fn wrote(&self, range: Range<u64>) {
        if range.start < range.end {
            self.hook.wrote_as_consumer(range);
        }
    }

    /// The source's offsets of the watched pages changed since `since` ([`DirtyPages::changed`]),
    /// by any writer but `writer` where given ([`DirtyPages::changed_except`]), into `out`.
    pub(crate) fn changed(
        &self,
        since: DirtyStamp,
        writer: Option<NonZeroU64>,
        out: &mut Vec<Range<u64>>,
    ) {
        match writer {
            Some(writer) => self.hook.changed_except(since, writer, out),
            None => self.hook.changed(since, out),
        }
    }
}

impl fmt::Debug for DirtySource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DirtySource")
            .field("pointer", &self.pointer)
            .finish()
    }
}

/// A shadow over logged host pages (`shadow.rs`), as its copies in ask the log: the source, the
/// shadow's writer, where its truth lies in the source, the watch over those bytes, and the stamp
/// as of which its image held every texel of the truth, where it holds one ([`DirtyShadows`]).
pub(crate) struct ShadowLog {
    copies: Weak<ShadowCopies>,
    source: Arc<DirtySource>,
    writer: NonZeroU64,
    /// The `VkBuffer` the truth lay in when the log was made, and the source's offset of that
    /// `VkBuffer`'s byte 0 (wrapping: it may lie below the pointer).
    raw: vk::Buffer,
    delta: u64,
    /// The truth's bytes, as the source's offsets.
    truth: Range<u64>,
    _watch: Box<dyn Send + Sync>,
    /// The stamp as of which the image holds every texel of the truth, but for the pages the log
    /// reports changed since (the module docs); `None` where it holds nothing a copy may count on.
    hold: Mutex<Option<DirtyStamp>>,
}

impl ShadowLog {
    /// The source's offsets of the `VkBuffer` bytes `range`.
    fn offsets(&self, range: Range<u64>) -> Range<u64> {
        range.start.wrapping_add(self.delta)..range.end.wrapping_add(self.delta)
    }

    /// Report a copy back or merge of the shadow into the `VkBuffer` bytes `range`
    /// ([`DirtyPages::wrote_as`], the shadow its writer).
    pub(crate) fn wrote_back(&self, range: Range<u64>) {
        if let Some(offsets) = self.written(range) {
            self.source.hook.wrote_as(offsets, self.writer);
        }
    }

    /// The source's offsets a copy back or merge into the `VkBuffer` bytes `range` writes, cut to
    /// the truth's; `None` where it writes none.
    fn written(&self, range: Range<u64>) -> Option<Range<u64>> {
        let range = self.offsets(range);
        let (start, end) = (
            range.start.max(self.truth.start),
            range.end.min(self.truth.end),
        );
        (start < end).then_some(start..end)
    }
}

/// The logs of a device's shadows over logged host pages, by the shadow's key: made at a shadow's
/// first copy in over such pages, and let go of once the shadow went.
#[derive(Default)]
pub(crate) struct DirtyShadows {
    logs: Mutex<Vec<Arc<ShadowLog>>>,
    /// Whether any log was ever made: a command buffer collects its writes only then.
    any: AtomicBool,
    tally: Arc<DirtyTally>,
    /// Whether any host pages of the device's came with a log ([`PagesLog`]): the runtime's CPU
    /// writes through a buffer's `contents` look for one only then.
    logged: AtomicBool,
    /// The logs of host pages imported in place (not copies), by the key their uses name ([`Imports`]),
    /// which the GPU's writes through them are reported to: a command buffer collects its writes
    /// once one is ([`DirtyShadows::collects`]).
    imports: Mutex<Imports>,
    imported: AtomicBool,
}

/// The logs of host pages imported in place, by the key of each buffer whose uses name them, held
/// weakly (the pages' import holds them): the GPU writes such pages in place, and the logs see none
/// of it but what the runtime reports ([`DirtyCommit::report_host_writes`]).
#[derive(Default)]
struct Imports {
    by_key: KeyMap<ResourceKey, Vec<Weak<PagesLog>>>,
    /// The keys past which the next registration lets go of those whose pages went.
    prune_at: usize,
}

/// The gated copies in, as `WorkCounts::dirty_copies_in_gated` and kin count them.
#[derive(Default)]
pub(crate) struct DirtyTally {
    pub(crate) gated: AtomicU64,
    pub(crate) whole: AtomicU64,
    pub(crate) skipped: AtomicU64,
    pub(crate) bytes_whole: AtomicU64,
    pub(crate) bytes_copied: AtomicU64,
    pub(crate) stamps: AtomicU64,
    pub(crate) waits: AtomicU64,
    pub(crate) waits_unanswered: AtomicU64,
    pub(crate) wait_nanos: AtomicU64,
}

/// The writers no other shadow or copy in the process is ever given.
static NEXT_WRITER: AtomicU64 = AtomicU64::new(1);

/// A writer no other shadow or copy in the process is ever given ([`DirtyPages::wrote_as`]).
pub(crate) fn new_writer() -> Option<NonZeroU64> {
    NonZeroU64::new(NEXT_WRITER.fetch_add(1, Ordering::Relaxed))
}

/// Where an object's bytes over host pages (a no-copy buffer's, or a heap's own buffer's) lie in
/// the logs of the writes to those pages: its memory's own (`HostMemory::with_dirty_pages`),
/// whose offsets count from its pointer, and its guest RAM's (`HostRam::with_dirty_pages`), whose
/// offsets are host addresses, through the runs of the RAM its pages are. The runtime reports to
/// both what it writes into the pages itself (a copy's merge back, a `replaceRegion`, the GPU's
/// writes through an import of them); a copy of the bytes asks one of them which pages changed
/// (`copied.rs`). Kept with the pages' import, which every buffer lying in it shares.
pub(crate) struct PagesLog {
    /// The host address of the object's byte 0 (its `contents`).
    pointer: u64,
    /// Its bytes.
    length: u64,
    logs: Vec<LogOf>,
}

/// One log of an object's pages ([`PagesLog`]).
struct LogOf {
    source: Arc<DirtySource>,
    /// The object's bytes in the source's offsets: runs, each its first byte of the object and
    /// the source's offsets of it, in the object's order, apart.
    runs: Box<[(u64, Range<u64>)]>,
}

impl LogOf {
    /// Each run of the object's bytes that `changed` (the source's offsets, sorted by start and
    /// apart) meets, to `each`, in the object's order.
    fn bytes_of(&self, changed: &[Range<u64>], mut each: impl FnMut(Range<u64>)) {
        for (first, offsets) in self.runs.iter() {
            let from = changed.partition_point(|run| run.end <= offsets.start);
            for run in changed[from..]
                .iter()
                .take_while(|run| run.start < offsets.end)
            {
                let (start, stop) = (run.start.max(offsets.start), run.end.min(offsets.end));
                if start < stop {
                    each(first + (start - offsets.start)..first + (stop - offsets.start));
                }
            }
        }
    }

    /// Each run of the source's offsets of the object's bytes `bytes`, to `each`.
    fn each(&self, bytes: &Range<u64>, mut each: impl FnMut(Range<u64>)) {
        for (first, offsets) in self.runs.iter() {
            let end = first + (offsets.end - offsets.start);
            let (start, stop) = (bytes.start.max(*first), bytes.end.min(end));
            if start < stop {
                each(offsets.start + (start - first)..offsets.start + (stop - first));
            }
        }
    }
}

impl PagesLog {
    /// The logs of the `length` bytes at host address `pointer`: `own`, their memory's log, and
    /// `ram`, their guest RAM's log with the runs of the RAM (host addresses) their import window
    /// is, in order, the bytes' first at `offset` of the window. `None` where neither is given or
    /// a log holds none of the bytes.
    pub(crate) fn new(
        (pointer, length): (u64, u64),
        own: Option<&Arc<DirtySource>>,
        ram: Option<(&Arc<DirtySource>, &[Range<u64>], u64)>,
    ) -> Option<Arc<PagesLog>> {
        let mut logs = Vec::new();
        if let Some(source) = own {
            if let Some(start) = source.offset_of(pointer) {
                logs.push(LogOf {
                    source: Arc::clone(source),
                    runs: Box::new([(0, start..start.saturating_add(length))]),
                });
            }
        }
        if let Some((source, window, offset)) = ram {
            let mut runs = Vec::new();
            let mut at = 0u64;
            for run in window {
                let size = run.end.saturating_sub(run.start);
                // The run's window bytes `at..at + size`, cut to the object's.
                let (start, stop) = (at.max(offset), (at + size).min(offset + length));
                if start < stop {
                    let host = run.start + (start - at)..run.start + (stop - at);
                    if let (Some(first), Some(last)) =
                        (source.offset_of(host.start), source.offset_of(host.end))
                    {
                        runs.push((start - offset, first..last));
                    }
                }
                at += size;
            }
            if !runs.is_empty() {
                logs.push(LogOf {
                    source: Arc::clone(source),
                    runs: runs.into_boxed_slice(),
                });
            }
        }
        (!logs.is_empty()).then(|| {
            Arc::new(PagesLog {
                pointer,
                length,
                logs,
            })
        })
    }

    /// Its logs' sources, in its order.
    pub(crate) fn sources(&self) -> impl Iterator<Item = &Arc<DirtySource>> {
        self.logs.iter().map(|log| &log.source)
    }

    /// A watch of every run of the object's bytes in each log ([`DirtyPages::watch`]), held
    /// while a copy of them asks the logs which pages changed.
    pub(crate) fn watch(&self) -> Vec<Box<dyn Send + Sync>> {
        (self.logs.iter())
            .flat_map(|log| {
                log.runs
                    .iter()
                    .map(|(_, offsets)| log.source.hook.watch(offsets.clone()))
            })
            .collect()
    }

    /// The object's bytes no log keeps the writes to, in order: every copy in compares them.
    pub(crate) fn unlogged(&self) -> Vec<Range<u64>> {
        let mut kept: Vec<Range<u64>> = (self.logs.iter())
            .flat_map(|log| log.runs.iter())
            .map(|(first, offsets)| *first..first + (offsets.end - offsets.start))
            .collect();
        kept.sort_by_key(|run| run.start);
        let mut unlogged = Vec::new();
        let mut at = 0;
        for run in kept {
            if at < run.start {
                unlogged.push(at..run.start.min(self.length));
            }
            at = at.max(run.end);
        }
        if at < self.length {
            unlogged.push(at..self.length);
        }
        unlogged
    }

    /// Each run of the object's bytes that `changed`, the runs log `index` (of [`sources`]'
    /// order) reported changed, sorted by start and apart, meets, to `each`.
    ///
    /// [`sources`]: Self::sources
    pub(crate) fn bytes_of(
        &self,
        index: usize,
        changed: &[Range<u64>],
        each: impl FnMut(Range<u64>),
    ) {
        if let Some(log) = self.logs.get(index) {
            log.bytes_of(changed, each);
        }
    }

    /// The runtime wrote the object's bytes `bytes` itself (writer 0, [`DirtyPages::wrote`]):
    /// reported to every log, so every copy of the pages takes them.
    pub(crate) fn wrote(&self, bytes: Range<u64>) {
        for log in &self.logs {
            log.each(&bytes, |run| log.source.wrote(run));
        }
    }

    /// [`wrote`](Self::wrote), but for the source offsets `but` names of each source: what a
    /// shadow over the same pages reports as its own write ([`DirtyCommit::report_host_writes`]).
    fn wrote_but(&self, bytes: Range<u64>, but: &[(&Arc<DirtySource>, Range<u64>)]) {
        for log in &self.logs {
            log.each(&bytes, |run| {
                let mut left = vec![run];
                for (_, apart) in but.iter().filter(|(s, _)| Arc::ptr_eq(s, &log.source)) {
                    left = (left.into_iter())
                        .flat_map(|run| {
                            let before = run.start..run.end.min(apart.start);
                            let after = run.start.max(apart.end)..run.end;
                            [before, after].into_iter().filter(|r| r.start < r.end)
                        })
                        .collect();
                }
                for run in left {
                    log.source.wrote(run);
                }
            });
        }
    }

    /// [`wrote`](Self::wrote) of the host addresses `addresses`, cut to the object's: a CPU write
    /// of the runtime's own through its `contents` (a `replaceRegion`, an argument encoder's).
    pub(crate) fn wrote_at(&self, addresses: Range<u64>) {
        let start = addresses.start.saturating_sub(self.pointer);
        let end = (addresses.end.saturating_sub(self.pointer)).min(self.length);
        if start < end {
            self.wrote(start..end);
        }
    }

    /// The runtime wrote the object's bytes `bytes` as `writer`, a copy's merge back
    /// ([`DirtyPages::wrote_as`]): reported to every log.
    pub(crate) fn wrote_as(&self, bytes: Range<u64>, writer: NonZeroU64) {
        for log in &self.logs {
            log.each(&bytes, |run| {
                if run.start < run.end {
                    log.source.hook.wrote_as(run, writer);
                }
            });
        }
    }
}

impl DirtyShadows {
    /// The gated copies in so far.
    pub(crate) fn tally(&self) -> &DirtyTally {
        &self.tally
    }

    /// Whether any shadow over logged host pages was copied in on the device.
    pub(crate) fn any(&self) -> bool {
        self.any.load(Ordering::Acquire)
    }

    /// Host pages of the device's came with a log ([`PagesLog`]).
    pub(crate) fn note_logged(&self) {
        self.logged.store(true, Ordering::Release);
    }

    /// Whether any host pages of the device's came with a log ([`PagesLog`]).
    pub(crate) fn logged(&self) -> bool {
        self.logged.load(Ordering::Acquire)
    }

    /// Whether a command buffer collects its commands' writes ([`DirtyCommit::note`]): a shadow
    /// over logged pages was copied in, or logged pages are imported in place.
    pub(crate) fn collects(&self) -> bool {
        self.any() || self.in_place()
    }

    /// Whether logged pages are imported in place ([`DirtyShadows::logged_in_place`]).
    pub(crate) fn in_place(&self) -> bool {
        self.imported.load(Ordering::Acquire)
    }

    /// Host pages imported in place whose writes `log` keeps lie under the uses of `key`, in the
    /// bytes `log` keeps (a buffer's own, or a heap's, whose buffers' uses name its bytes): the
    /// GPU's writes through it are reported to `log` ([`DirtyCommit::report_host_writes`]).
    pub(crate) fn logged_in_place(&self, key: ResourceKey, log: &Arc<PagesLog>) {
        let mut imports = lock(&self.imports);
        if imports.by_key.len() >= imports.prune_at {
            (imports.by_key).retain(|_, logs| {
                logs.retain(|log| log.strong_count() > 0);
                !logs.is_empty()
            });
            imports.prune_at = (2 * imports.by_key.len()).max(64);
        }
        let logs = imports.by_key.entry(key).or_default();
        if !logs
            .iter()
            .any(|kept| std::ptr::eq(kept.as_ptr(), Arc::as_ptr(log)))
        {
            logs.push(Arc::downgrade(log));
        }
        drop(imports);
        self.imported.store(true, Ordering::Release);
    }

    /// The log of `copies`, whose truth lies at `truth` of the `VkBuffer` `raw` in pages logged by
    /// `source` (`raw`'s byte `v` at the host address `v + base`): kept, or made now, its watch
    /// begun. A kept log of another source or `VkBuffer` (a rebind gave the buffer other pages)
    /// is made again.
    pub(crate) fn log(
        &self,
        copies: &Arc<ShadowCopies>,
        source: &Arc<DirtySource>,
        raw: vk::Buffer,
        base: u64,
        truth: Range<u64>,
    ) -> Option<Arc<ShadowLog>> {
        let delta = base.wrapping_sub(source.pointer);
        let offsets = truth.start.wrapping_add(delta)..truth.end.wrapping_add(delta);
        // A truth below the pointer, or wrapping, is no range of the source's.
        if offsets.start > offsets.end || base.checked_add(truth.start)? < source.pointer {
            return None;
        }
        let mut logs = lock(&self.logs);
        logs.retain(|log| log.copies.strong_count() > 0);
        let kept = logs.iter().position(|log| {
            std::ptr::eq(log.copies.as_ptr(), Arc::as_ptr(copies))
                && Arc::ptr_eq(&log.source, source)
                && log.raw == raw
                && log.truth == offsets
        });
        if let Some(at) = kept {
            return Some(Arc::clone(&logs[at]));
        }
        logs.retain(|log| !std::ptr::eq(log.copies.as_ptr(), Arc::as_ptr(copies)));
        let writer = NonZeroU64::new(NEXT_WRITER.fetch_add(1, Ordering::Relaxed))?;
        let log = Arc::new(ShadowLog {
            copies: Arc::downgrade(copies),
            source: Arc::clone(source),
            writer,
            raw,
            delta,
            truth: offsets.clone(),
            _watch: source.hook.watch(offsets),
            hold: Mutex::new(None),
        });
        logs.push(Arc::clone(&log));
        self.any.store(true, Ordering::Release);
        Some(log)
    }

    /// A command buffer whose commit was `commit` completed, `failed` or not: its shadows' copies
    /// back and merges are reported again ([`DirtyPages::wrote_as`]), and its writes into logged
    /// host pages imported in place ([`DirtyCommit::report_host_writes`]), now they ran, so a
    /// stamp asked between the commit and the GPU's write still finds them; where it failed, the
    /// holds its copies in set are let go of, its copies having perhaps not run.
    pub(crate) fn completed(&self, commit: &DirtyCommit, failed: bool) {
        for (log, range) in &commit.wrote {
            log.wrote_back(range.clone());
        }
        commit.report_host_writes();
        if !failed {
            return;
        }
        for gate in &commit.gates {
            let Some(decision) = *lock(&gate.decision) else {
                continue;
            };
            let mut hold = lock(&gate.log.hold);
            if *hold == Some(decision.stamp) {
                *hold = None;
            }
        }
    }

    /// The log kept for `copies`, where its copies in made one.
    pub(crate) fn kept(&self, copies: &ShadowCopies) -> Option<Arc<ShadowLog>> {
        let logs = lock(&self.logs);
        (logs.iter())
            .find(|log| std::ptr::eq(log.copies.as_ptr(), copies))
            .cloned()
    }

    /// A committed command buffer's `commit` ([`DirtyCommit`]), in commit order: its gates take
    /// the holds as they stand and a stamp each (no hold where it `waits` for an event or a
    /// semaphore: work committed after it may run before it), its writes let go of the holds of
    /// the shadows they meet, and the shadows its end left holding every texel of the truth
    /// (`held`) hold from their stamp on.
    pub(crate) fn committed(&self, commit: &DirtyCommit, held: &[Arc<ShadowCopies>], waits: bool) {
        let DirtyCommit {
            written,
            gates,
            wrote,
            ..
        } = commit;
        // One stamp per source and commit.
        let mut stamps: Vec<(Arc<DirtySource>, DirtyStamp)> = Vec::new();
        let mut stamp_of = |source: &Arc<DirtySource>| {
            if let Some((_, stamp)) = stamps.iter().find(|(s, _)| Arc::ptr_eq(s, source)) {
                return *stamp;
            }
            let stamp = source.request(&self.tally);
            stamps.push((Arc::clone(source), stamp));
            stamp
        };
        let mut synced: Vec<(Arc<ShadowLog>, DirtyStamp)> = Vec::new();
        for gate in gates {
            let since = match gate.written_before || waits {
                true => None,
                false => *lock(&gate.log.hold),
            };
            let stamp = stamp_of(&gate.log.source);
            *lock(&gate.decision) = Some(Decision { since, stamp });
            synced.push((Arc::clone(&gate.log), stamp));
        }
        for (log, range) in wrote {
            log.wrote_back(range.clone());
        }
        commit.report_host_writes();
        let logs: Vec<Arc<ShadowLog>> = lock(&self.logs).clone();
        for log in &logs {
            let Some(copies) = log.copies.upgrade() else {
                continue;
            };
            let meets =
                |used: &Use<ResourceKey>| used.key == copies.shadow_key || copies.meets(used);
            if written.iter().any(meets) {
                *lock(&log.hold) = None;
            }
        }
        for (log, stamp) in synced {
            let holds = log
                .copies
                .upgrade()
                .is_some_and(|copies| held.iter().any(|h| Arc::ptr_eq(h, &copies)));
            if holds {
                *lock(&log.hold) = Some(stamp);
            }
        }
    }
}

/// What a command buffer did that the logs' holds hear of at its commit
/// ([`DirtyShadows::committed`]): its commands' writes, its gated copies in, its shadows' copies
/// back and merges into logged pages, and its writes into logged pages imported in place.
#[derive(Default)]
pub(crate) struct DirtyCommit {
    written: Vec<Use<ResourceKey>>,
    gates: Vec<Arc<CopyInGate>>,
    wrote: Vec<(Arc<ShadowLog>, Range<u64>)>,
    /// Its writes were more than [`WRITES`]: `written` holds the first alone.
    overflowed: bool,
    /// Its writes into host pages imported in place whose memory is logged: each pages' logs,
    /// and the bytes written, in order, at most [`HOST_RANGES`] of them, past which the pages
    /// are written whole.
    host: Vec<(Arc<PagesLog>, Vec<Range<u64>>)>,
}

/// The most writes a command buffer keeps apart, past which they are one write of everything.
const WRITES: usize = 4096;

/// The most ranges of one host pages' a command buffer keeps apart, past which it wrote them
/// whole.
const HOST_RANGES: usize = 64;

impl DirtyCommit {
    /// Note a command's `uses`: its writes, each once; and of them, those into logged host pages
    /// imported in place (`shadows`' [`Imports`]).
    pub(crate) fn note(&mut self, uses: &[Use<ResourceKey>], shadows: &DirtyShadows) {
        let imports = shadows.in_place().then(|| lock(&shadows.imports));
        for used in uses.iter().filter(|used| !used.write.is_empty()) {
            let seen = (self.written.iter().rev().take(8))
                .any(|w| w.key == used.key && w.range == used.range);
            match (seen, self.written.len() < WRITES) {
                (true, _) => {}
                (false, true) => self.written.push(used.clone()),
                (false, false) => self.overflowed = true,
            }
            let Some(logs) = imports.as_ref().and_then(|i| i.by_key.get(&used.key)) else {
                continue;
            };
            let bytes = match &used.range {
                UseRange::Bytes(bytes) => bytes.clone(),
                // An image's levels and layers, which name no bytes: all of them.
                UseRange::Image { .. } => 0..u64::MAX,
            };
            for log in logs.iter().filter_map(Weak::upgrade) {
                self.wrote_host(log, bytes.clone());
            }
        }
    }

    /// Its commands' writes, for the merge back of copied host buffers (`copied.rs`): the keys
    /// they wrote, or every copy where it kept them not all apart.
    pub(crate) fn copies_written(&self, into: &mut Written) {
        if self.overflowed {
            into.wrote_all();
            return;
        }
        for used in &self.written {
            into.wrote(used.key);
        }
    }

    /// Note a write of `bytes` of the host pages `log` keeps the writes to.
    fn wrote_host(&mut self, log: Arc<PagesLog>, bytes: Range<u64>) {
        let at = match (self.host.iter()).position(|(kept, _)| Arc::ptr_eq(kept, &log)) {
            Some(at) => at,
            None => {
                self.host.push((log, Vec::new()));
                self.host.len() - 1
            }
        };
        let ranges = &mut self.host[at].1;
        let whole = 0..u64::MAX;
        if ranges.first() == Some(&whole) || ranges.iter().rev().take(8).any(|r| *r == bytes) {
            return;
        }
        match ranges.len() < HOST_RANGES {
            true => ranges.push(bytes),
            false => *ranges = vec![whole],
        }
    }

    /// Report its writes into logged host pages imported in place to their logs, as the
    /// consumer's writes (writer 0, [`DirtyPages::wrote`]): but for the bytes its shadows' copies
    /// back and merges report as their own ([`ShadowLog::wrote_back`]), which every other copy
    /// of the pages counts as changed already, and their own copies in must not. At the commit
    /// ([`DirtyShadows::committed`]), again at each of its batches' completion turns, before
    /// anything sees the batch complete (its event signals among them, `completion.rs`), and at
    /// its completion ([`DirtyShadows::completed`]): a GPU write after a copy's sync at a stamp is
    /// reported after it too, before work ordered after the write can look.
    pub(crate) fn report_host_writes(&self) {
        if self.host.is_empty() {
            return;
        }
        let shadows: Vec<(&Arc<DirtySource>, Range<u64>)> = (self.wrote.iter())
            .filter_map(|(log, range)| Some((&log.source, log.written(range.clone())?)))
            .collect();
        for (log, ranges) in &self.host {
            for bytes in ranges {
                log.wrote_but(bytes.clone(), &shadows);
            }
        }
    }

    /// Whether a write noted so far meets `copies`' truth or its shadow.
    fn wrote_over(&self, copies: &ShadowCopies) -> bool {
        self.written.len() >= WRITES
            || (self.written.iter()).any(|used| used.key == copies.shadow_key || copies.meets(used))
    }

    /// A gate for a copy in of `log`'s shadow `copies` recorded now ([`CopyInGate`]), counted in
    /// `shadows`' tally.
    pub(crate) fn gate(
        &mut self,
        (log, copies, texel): (Arc<ShadowLog>, &ShadowCopies, u64),
        shadows: &DirtyShadows,
    ) -> Arc<CopyInGate> {
        let gate = Arc::new(CopyInGate {
            log,
            texel,
            tally: Arc::clone(&shadows.tally),
            written_before: self.wrote_over(copies),
            decision: Mutex::new(None),
        });
        self.gates.push(Arc::clone(&gate));
        gate
    }

    /// Note a copy back or merge of `log`'s shadow into its `VkBuffer`'s bytes `range`.
    pub(crate) fn wrote_back(&mut self, log: Arc<ShadowLog>, range: Range<u64>) {
        self.wrote.push((log, range));
    }
}

/// A shadow's copy in recorded with what its commit decides ([`DirtyShadows::committed`]): the
/// stamp its log's hold stood at, where it held one, and the stamp it syncs to.
#[derive(Clone, Copy)]
struct Decision {
    since: Option<DirtyStamp>,
    stamp: DirtyStamp,
}

/// A copy in of a shadow over logged host pages, its regions decided as its batch is submitted
/// (`shadow::GatedCopyIn`): every region where its commit found no hold
/// or its log cannot say, else the rows of the pages changed since the hold.
pub(crate) struct CopyInGate {
    log: Arc<ShadowLog>,
    /// The bytes of one texel in the buffer.
    texel: u64,
    /// The command buffer wrote the truth or the shadow before the copy.
    written_before: bool,
    tally: Arc<DirtyTally>,
    decision: Mutex<Option<Decision>>,
}

/// A run of rows more than this many rows from the next is copied apart.
const ROW_GAP: u32 = 8;
/// The most regions a gated copy in records, past which it copies every region.
const REGIONS: usize = 1024;

impl CopyInGate {
    /// The regions of `whole` to copy now, into `out`, counted: `whole` itself where the decision
    /// holds no stamp, the wait cannot say, or the rows changed come to half the rows or more;
    /// none where no page changed.
    pub(crate) fn regions(
        &self,
        whole: &[vk::BufferImageCopy],
        out: &mut Vec<vk::BufferImageCopy>,
    ) {
        let (total, copied) = self.chosen(whole, out);
        let tally = &self.tally;
        tally.gated.fetch_add(1, Ordering::Relaxed);
        tally.bytes_whole.fetch_add(total, Ordering::Relaxed);
        tally.bytes_copied.fetch_add(copied, Ordering::Relaxed);
        if out.is_empty() {
            tally.skipped.fetch_add(1, Ordering::Relaxed);
        } else if copied == total {
            tally.whole.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// [`regions`](Self::regions), uncounted: the bytes `whole` and the chosen regions move.
    fn chosen(
        &self,
        whole: &[vk::BufferImageCopy],
        out: &mut Vec<vk::BufferImageCopy>,
    ) -> (u64, u64) {
        out.clear();
        let bytes = |regions: &[vk::BufferImageCopy]| -> u64 {
            (regions.iter())
                .map(|r| {
                    u64::from(r.image_extent.width)
                        * u64::from(r.image_extent.height)
                        * u64::from(r.image_extent.depth.max(1))
                        * self.texel
                })
                .sum()
        };
        let total = bytes(whole);
        let decision = *lock(&self.decision);
        let changed = decision.and_then(|Decision { since, stamp }| {
            let since = since?;
            if !self.log.source.wait(stamp, &self.tally) {
                return None;
            }
            let mut runs = Vec::new();
            (self.log.source.hook).changed_except(since, self.log.writer, &mut runs);
            Some(runs)
        });
        let Some(runs) = changed else {
            out.extend_from_slice(whole);
            return (total, total);
        };
        for region in whole {
            if !self.bands(region, &runs, out) {
                out.clear();
                out.extend_from_slice(whole);
                return (total, total);
            }
        }
        if out.len() > REGIONS {
            out.clear();
            out.extend_from_slice(whole);
            return (total, total);
        }
        (total, bytes(out))
    }

    /// The full-width bands of `region`'s rows that the source's changed `runs` meet, into
    /// `out`; `false` where the region is copied whole (rows changed come to half or more, or
    /// its shape is not one of rows).
    fn bands(
        &self,
        region: &vk::BufferImageCopy,
        runs: &[Range<u64>],
        out: &mut Vec<vk::BufferImageCopy>,
    ) -> bool {
        let extent = region.image_extent;
        if extent.depth > 1 || extent.height == 0 || extent.width == 0 {
            return false;
        }
        let row_texels = match region.buffer_row_length {
            0 => extent.width,
            row => row,
        };
        let pitch = u64::from(row_texels) * self.texel;
        let row_bytes = u64::from(extent.width) * self.texel;
        let first = region.buffer_offset.wrapping_add(self.log.delta);
        let height = u64::from(extent.height);
        let end = first + (height - 1) * pitch + row_bytes;
        let mut rows: Vec<(u32, u32)> = Vec::new();
        for run in runs {
            let (start, stop) = (run.start.max(first), run.end.min(end));
            if start >= stop {
                continue;
            }
            let top = (start - first) / pitch;
            let bottom = ((stop - first).div_ceil(pitch)).min(height);
            rows.push((top as u32, bottom as u32));
        }
        rows.sort_unstable();
        let mut bands: Vec<(u32, u32)> = Vec::new();
        for (top, bottom) in rows {
            match bands.last_mut() {
                Some(last) if top <= last.1 + ROW_GAP => last.1 = last.1.max(bottom),
                _ => bands.push((top, bottom)),
            }
        }
        let changed: u64 = bands.iter().map(|(t, b)| u64::from(b - t)).sum();
        if 2 * changed >= height {
            return false;
        }
        out.extend(bands.into_iter().map(|(top, bottom)| vk::BufferImageCopy {
            buffer_offset: region.buffer_offset + u64::from(top) * pitch,
            buffer_image_height: 0,
            image_offset: vk::Offset3D {
                y: region.image_offset.y + top as i32,
                ..region.image_offset
            },
            image_extent: vk::Extent3D {
                height: bottom - top,
                ..extent
            },
            ..*region
        }));
        true
    }
}
