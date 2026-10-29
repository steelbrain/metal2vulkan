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
//!   [`DirtyPages::wait`]). A copy then syncs to that stamp: every write made after it is stamped
//!   later.
//! - **The pages changed** ([`DirtyPages::changed_except`]) since the stamp a copy synced to, by
//!   any writer but that copy's own, are the only bytes it may lack: they are copied in, the rest
//!   skipped.
//! - **The runtime's own writes** into the pages (a shadow's copy back or merge) are reported
//!   ([`DirtyPages::wrote_as`]), naming the copy as their writer, so other copies of the same
//!   pages see them as changed and the copy's own copies in, under a log that keeps writers
//!   apart, do not.
//! - **Correctness first.** A wait that answers `false`, a source that cannot say, and a copy
//!   that holds nothing it may count on all copy every byte.
//!
//! [`HostMemory::with_dirty_pages`]: crate::HostMemory::with_dirty_pages

use super::device::lock;
use super::resource::ResourceKey;
use super::shadow::ShadowCopies;
use super::tracker::Use;
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
/// own copy back or merge naming the copy as its writer ([`wrote_as`](Self::wrote_as)), and asks
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
/// GPU, [`wrote`](Self::wrote) and [`wrote_as`](Self::wrote_as) at the commit and at the
/// completion. A [`Watch`](Self::Watch) may be dropped on a completion thread, so its drop takes
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

    /// The runtime wrote the memory's bytes `range` (offsets from its pointer) from the GPU (a
    /// shadow's copy back, a merge), so other copies of the same pages see it as changed.
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
    fn request(&self, tally: &DirtyTally) -> DirtyStamp {
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
    fn wait(&self, stamp: DirtyStamp, tally: &DirtyTally) -> bool {
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
        let range = self.offsets(range);
        let (start, end) = (
            range.start.max(self.truth.start),
            range.end.min(self.truth.end),
        );
        if start < end {
            self.source.hook.wrote_as(start..end, self.writer);
        }
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

/// The writers no other shadow in the process is ever given.
static NEXT_WRITER: AtomicU64 = AtomicU64::new(1);

impl DirtyShadows {
    /// The gated copies in so far.
    pub(crate) fn tally(&self) -> &DirtyTally {
        &self.tally
    }

    /// Whether any shadow over logged host pages was copied in on the device.
    pub(crate) fn any(&self) -> bool {
        self.any.load(Ordering::Acquire)
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
    /// back and merges are reported again ([`DirtyPages::wrote_as`]), now they ran, so a stamp
    /// asked between the commit and the GPU's write still finds them; where it failed, the holds
    /// its copies in set are let go of, its copies having perhaps not run.
    pub(crate) fn completed(&self, commit: &DirtyCommit, failed: bool) {
        for (log, range) in &commit.wrote {
            log.wrote_back(range.clone());
        }
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
/// ([`DirtyShadows::committed`]): its commands' writes, its gated copies in, and its shadows'
/// copies back and merges into logged pages.
#[derive(Default)]
pub(crate) struct DirtyCommit {
    written: Vec<Use<ResourceKey>>,
    gates: Vec<Arc<CopyInGate>>,
    wrote: Vec<(Arc<ShadowLog>, Range<u64>)>,
}

/// The most writes a command buffer keeps apart, past which they are one write of everything.
const WRITES: usize = 4096;

impl DirtyCommit {
    /// Note a command's `uses`: its writes, each once.
    pub(crate) fn note(&mut self, uses: &[Use<ResourceKey>]) {
        for used in uses.iter().filter(|used| !used.write.is_empty()) {
            let seen = (self.written.iter().rev().take(8))
                .any(|w| w.key == used.key && w.range == used.range);
            if !seen && self.written.len() < WRITES {
                self.written.push(used.clone());
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
