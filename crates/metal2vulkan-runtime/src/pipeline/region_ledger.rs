//! The device's ledger of the region `VkPipeline`s its compute variants keep: one bound across
//! every kernel's and tile function's variant, a released variant's kept instance included
//! (`kept.rs`), the least recently used going first.
//!
//! - **What it lists.** Each region `VkPipeline` a variant's memo keeps, or its making begun
//!   (`making.rs`), by the variant's instance and the region's key: every compute variant but a
//!   warm pass's, whose objects no instance keeps (`warm/pass.rs`). A memo's key kept, looked up,
//!   let go past the variant's own bound ([`REGION_PIPELINES`](super::REGION_PIPELINES)) or
//!   removed is told here as it happens ([`RegionWatch`]), under the memo's lock: the ledger's
//!   lock is the innermost, taken under a memo's or the kept variants' and never the other way
//!   round. A dispatch that takes the pipelines it took before without looking them up (its
//!   encoder's memo of what it planned, `compute.rs`) does not use them here.
//! - **Past its bound** ([`DEVICE_REGION_PIPELINES`]) the least recently used keys go
//!   (`WorkCounts::region_pipelines_evicted` counts them with each variant's own). A key whose
//!   variant lives is taken out of its memo where it is made at the next lookup of a region
//!   pipeline on the device ([`let_go`]), with no memo's lock held; a making in flight finishes
//!   and is kept, listed again. A key of a released variant's kept instance is destroyed where the
//!   kept variants next settle, on the program's thread and in its order, as every change of
//!   theirs is (`Kept::settle`): they keep only the region pipelines the ledger lists. Either way
//!   the `VkPipeline` is destroyed once its last holder lets go of it: a command in flight, or an
//!   encoder's memo of what a dispatch planned.
//! - **One copy each.** A region pipeline a variant's own bound or the ledger lets go is destroyed
//!   at its last release, never kept ([`RawPipeline::doom`]): a kept instance holds only what its
//!   variant's memo held when it was released, each listed once.

use super::kept::Instance;
use super::making::{KeyWatch, WeakMaking};
use super::memo::Lru;
use super::objects::RawPipeline;
use super::RegionKey;
use crate::device::{lock, DeviceShared};
use crate::work_counts::Counters;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The most region `VkPipeline`s the device's compute variants keep together, their kept
/// instances included (the module docs): twice what one variant keeps
/// ([`REGION_PIPELINES`](super::REGION_PIPELINES), whose docs give the measured working set and
/// what each costs on lavapipe and RADV), so one kernel's full memo leaves as many to the rest.
/// About 325 MiB of resident memory on lavapipe and at most about 258 MiB of GTT on RADV where it
/// is full of such pipelines. A count, not bytes: Vulkan has no query of what one `VkPipeline`
/// holds (`VK_EXT_memory_budget` reports a heap's use by the whole process, and
/// `VK_KHR_pipeline_executable_properties`' statistics are the driver's own, optional, and name no
/// memory).
pub(crate) const DEVICE_REGION_PIPELINES: usize = 2 * super::REGION_PIPELINES;

/// A listed region pipeline: its variant's instance and its key.
type Listed = (Instance, RegionKey);

/// The device's ledger (the module docs).
pub(crate) struct RegionLedger {
    state: Mutex<Ledger>,
    /// Whether keys went past the bound since the last [`let_go`]: a lookup that
    /// finds none takes no lock.
    due: AtomicBool,
    /// How many times keys went past the bound.
    times: AtomicU64,
    /// `times` as it was when the kept variants last destroyed the region pipelines it no longer
    /// lists (`Kept::settle`).
    pruned: AtomicU64,
    counts: Arc<Counters>,
}

struct Ledger {
    /// Each listed key, with its variant's memo, least recently used first.
    listed: Lru<Listed, WeakMaking<RegionKey, RawPipeline>>,
    bound: usize,
    /// Keys past the bound whose memos may still hold them made.
    past: Vec<(Listed, WeakMaking<RegionKey, RawPipeline>)>,
}

impl RegionLedger {
    /// An empty ledger of at most `bound` keys, counting what goes past it in `counts`.
    pub(crate) fn new(bound: usize, counts: Arc<Counters>) -> RegionLedger {
        RegionLedger {
            state: Mutex::new(Ledger {
                listed: Lru::counted(usize::MAX),
                bound,
                past: Vec::new(),
            }),
            due: AtomicBool::new(false),
            times: AtomicU64::new(0),
            pruned: AtomicU64::new(0),
            counts,
        }
    }

    /// `instance`'s `key`, kept by its variant's `memo`, now the most recently used: the least
    /// recently used past the bound go.
    fn kept(&self, listed: Listed, memo: &WeakMaking<RegionKey, RawPipeline>) {
        let mut state = lock(&self.state);
        state.listed.insert(listed, memo.clone(), 1);
        let mut gone = 0;
        while state.listed.count() > state.bound {
            let Some(oldest) = state.listed.pop_oldest() else {
                break;
            };
            state.past.push(oldest);
            gone += 1;
        }
        if gone > 0 {
            self.due.store(true, Ordering::Release);
            self.times.fetch_add(1, Ordering::AcqRel);
            self.counts.region_pipelines_evicted(gone);
        }
    }

    /// `listed` was looked up: now the most recently used.
    fn used(&self, listed: &Listed) {
        lock(&self.state).listed.get(listed);
    }

    /// `instance`'s `key`, no longer kept: whether it was listed.
    pub(super) fn forget(&self, instance: Instance, key: &RegionKey) -> bool {
        let listed = (instance, key.clone());
        let mut state = lock(&self.state);
        let was = state.listed.peek(&listed).is_some();
        state.listed.remove(&listed);
        was
    }

    /// Whether `instance`'s `key` is listed: what a kept instance may keep.
    pub(super) fn lists(&self, instance: Instance, key: &RegionKey) -> bool {
        (lock(&self.state).listed)
            .peek(&(instance, key.clone()))
            .is_some()
    }

    /// How many times keys went past the bound so far, where that is more than when the kept
    /// variants last destroyed what it no longer lists ([`pruned_at`](Self::pruned_at)).
    pub(super) fn prune_due(&self) -> Option<u64> {
        let times = self.times.load(Ordering::Acquire);
        (times != self.pruned.load(Ordering::Acquire)).then_some(times)
    }

    /// The kept variants destroyed what it no longer listed when keys had gone past the bound
    /// `times` times.
    pub(super) fn pruned_at(&self, times: u64) {
        self.pruned.store(times, Ordering::Release);
    }

    /// How many keys it lists.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn listed(&self) -> usize {
        lock(&self.state).listed.count()
    }

    /// Bound it at `bound` keys from now on: a test's, of a bound its kernels reach.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn bound_at(&self, bound: usize) {
        lock(&self.state).bound = bound;
    }

    /// Take the keys gone past the bound out of the memos that live and hold them made, each
    /// `VkPipeline` destroyed once its last holder lets go of it; a key listed again since stays.
    /// With no memo's lock held, nor the ledger's.
    fn let_go_past(&self) {
        if !self.due.swap(false, Ordering::AcqRel) {
            return;
        }
        loop {
            // One at a time, the ledger's lock let go before the memo's is taken.
            let next = lock(&self.state).past.pop();
            let Some(((instance, key), memo)) = next else {
                break;
            };
            if self.lists(instance, &key) {
                continue;
            }
            if let Some(made) = memo.forget(&key) {
                made.doom();
            }
        }
    }
}

/// Let go of what the device's ledger let go past its bound ([`RegionLedger`]), and where it let
/// any go since the kept variants last settled, settle them, destroying the kept region pipelines
/// it no longer lists (`Kept::settle_let_go`, which does nothing in a host step or a pipeline
/// pool's piece): where a dispatch looks up its region pipelines, with no memo's lock held nor the
/// kept variants'.
pub(super) fn let_go(shared: &DeviceShared) {
    shared.region_ledger.let_go_past();
    if shared.region_ledger.prune_due().is_some() {
        lock(&shared.kept).settle_let_go(shared);
    }
}

/// What a compute variant's memo of region pipelines tells the device's ledger, for its instance
/// (the module docs); a warm pass's variant, which has none, only counts what its memo lets go.
pub(super) struct RegionWatch {
    pub(super) instance: Option<Instance>,
    pub(super) ledger: Arc<RegionLedger>,
    pub(super) memo: WeakMaking<RegionKey, RawPipeline>,
}

impl KeyWatch<RegionKey, RawPipeline> for RegionWatch {
    fn kept(&self, key: &RegionKey) {
        if let Some(instance) = self.instance {
            self.ledger.kept((instance, key.clone()), &self.memo);
        }
    }

    fn used(&self, key: &RegionKey) {
        if let Some(instance) = self.instance {
            self.ledger.used(&(instance, key.clone()));
        }
    }

    fn evicted(&self, key: &RegionKey, made: Option<&Arc<RawPipeline>>) {
        if let Some(made) = made {
            made.doom();
        }
        // Counted once: a key the ledger let go first was counted then.
        let listed = match self.instance {
            Some(instance) => self.ledger.forget(instance, key),
            None => true,
        };
        if listed {
            self.ledger.counts.region_pipelines_evicted(1);
        }
    }

    fn removed(&self, key: &RegionKey) {
        if let Some(instance) = self.instance {
            self.ledger.forget(instance, key);
        }
    }
}
