//! The device's bounded memo of compiled pipelines, and the least-recently-used map every pipeline
//! memo of the runtime is.
//!
//! - **What it holds.** Translated functions by their [`FunctionCompileKey`], and compiled
//!   pipelines by their [`PipelineKey`], each beside a weak reference to the variant a live state
//!   made of it. Metal compiles a function once per process and gives every later state of it at
//!   once (measured on an M3 Max and an M4: 39 ms, then 20 us), as a new object with its own
//!   resource ID. So does the runtime: a state made again shares the live variant (its shader
//!   modules, layouts, plans and `VkPipeline`s), or makes the variant's objects again from the
//!   memo's pipeline, and is a state of its own either way.
//! - **Bounded.** The memo holds at most [`ShaderCacheOptions::memory_capacity`] bytes of
//!   artifacts and at most [`MEMO_ENTRIES`] entries; past either, the least recently used go. An
//!   entry is counted at its encoded length where it was encoded for the directory, and at what
//!   it retains ([`RenderPipeline::footprint`]) where it was kept without being encoded
//!   (`pipeline/sources.rs`) -- never at the zero bytes a skipped encoding returns, which would
//!   let 4096 entries of a few hundred kilobytes each sit in a memo that believed its bound
//!   untouched. It holds no Vulkan object, and no strong reference to a variant: those live as
//!   long as a state or a pending command holds them. A variant's own memos (its plans and
//!   `VkPipeline`s) and a state's other variants are [`Lru`]s too.
//!
//! [`ShaderCacheOptions::memory_capacity`]: crate::ShaderCacheOptions::memory_capacity

use super::render::RenderVariant;
use super::Variant;
use crate::device::lock;
use crate::key_hash::KeyMap;
use metal2vulkan::cache::PipelineKey;
use metal2vulkan::library::{CompiledFunction, FunctionCompileKey};
use metal2vulkan::pipeline::{ComputePipeline, RenderPipeline, StageEntries};
use std::collections::BTreeMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex, Weak};

/// The most entries the device's memo holds, whatever their sizes.
pub(crate) const MEMO_ENTRIES: usize = 4096;

/// A map that keeps at most `max_weight` of its entries' weights and at most `max_entries`
/// entries, removing the least recently used past either. A lookup or an insert is a use.
pub(crate) struct Lru<K, V> {
    /// Each entry's number: the stamp of the insert that kept it, found by the runtime's key
    /// hasher (`key_hash.rs`): a lookup of a variant's plan by bind shape is one per dispatch that
    /// settles.
    numbers: KeyMap<K, u64>,
    /// The entries, by number.
    entries: BTreeMap<u64, Slot<K, V>>,
    /// Each entry's number by its last use, oldest first.
    order: BTreeMap<u64, u64>,
    /// The next use's stamp.
    clock: u64,
    weight: u64,
    max_weight: u64,
    max_entries: usize,
    /// The uses that moved an entry in `order` ([`used`](Self::used)).
    #[cfg(test)]
    reordered: u64,
}

struct Slot<K, V> {
    key: K,
    value: V,
    weight: u64,
    used: u64,
}

/// An entry of an [`Lru`], to look up again without its key ([`Lru::get_at`]) while it stays:
/// its number, which no other entry of the map takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct At(u64);

impl<K: Clone + Eq + Hash, V> Lru<K, V> {
    /// An empty map bounded by `max_weight` and `max_entries`.
    pub(crate) fn new(max_weight: u64, max_entries: usize) -> Lru<K, V> {
        Lru {
            numbers: KeyMap::default(),
            entries: BTreeMap::new(),
            order: BTreeMap::new(),
            clock: 0,
            weight: 0,
            max_weight,
            max_entries,
            #[cfg(test)]
            reordered: 0,
        }
    }

    /// An empty map of at most `max_entries` entries, each of weight 1.
    pub(crate) fn counted(max_entries: usize) -> Lru<K, V> {
        Lru::new(u64::MAX, max_entries)
    }

    /// The value of `key`, now the most recently used.
    pub(crate) fn get(&mut self, key: &K) -> Option<&V> {
        self.get_mut(key).map(|value| &*value)
    }

    /// The value of `key`, now the most recently used, to change in place (its weight stays).
    fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let number = *self.numbers.get(key)?;
        self.used(number).map(|slot| &mut slot.value)
    }

    /// The value of `key`, now the most recently used, and where its entry lies.
    pub(crate) fn get_with_at(&mut self, key: &K) -> Option<(At, &V)> {
        let number = *self.numbers.get(key)?;
        let slot = self.used(number)?;
        Some((At(number), &slot.value))
    }

    /// The value of the entry at `at`, now the most recently used, while the entry stays: what
    /// [`get`](Self::get) of its key answers, and the same use of it.
    pub(crate) fn get_at(&mut self, at: At) -> Option<&V> {
        self.used(at.0).map(|slot| &slot.value)
    }

    /// The entry numbered `number`, now the most recently used. One that is already keeps its
    /// stamp and its place: no other entry was used since, so the order stands as a new stamp
    /// would leave it.
    fn used(&mut self, number: u64) -> Option<&mut Slot<K, V>> {
        let slot = self.entries.get_mut(&number)?;
        // The latest stamp given is the clock's last.
        if slot.used + 1 == self.clock {
            return Some(slot);
        }
        #[cfg(test)]
        {
            self.reordered += 1;
        }
        let stamp = self.clock;
        self.clock += 1;
        self.order.remove(&slot.used);
        slot.used = stamp;
        self.order.insert(stamp, number);
        Some(slot)
    }

    /// Keep `value` under `key` with `weight`, as the most recently used, replacing what `key`
    /// held; then remove the least recently used entries until both bounds hold. A value heavier
    /// than the whole bound is not kept.
    pub(crate) fn insert(&mut self, key: K, value: V, weight: u64) {
        self.remove(&key);
        if weight > self.max_weight || self.max_entries == 0 {
            return;
        }
        let stamp = self.clock;
        self.clock += 1;
        self.numbers.insert(key.clone(), stamp);
        let slot = Slot {
            key,
            value,
            weight,
            used: stamp,
        };
        self.entries.insert(stamp, slot);
        self.order.insert(stamp, stamp);
        self.weight += weight;
        while self.weight > self.max_weight || self.entries.len() > self.max_entries {
            let Some((_, oldest)) = self.order.pop_first() else {
                break;
            };
            if let Some(slot) = self.entries.remove(&oldest) {
                self.numbers.remove(&slot.key);
                self.weight -= slot.weight;
            }
        }
    }

    /// Remove `key`'s entry, if any.
    pub(crate) fn remove(&mut self, key: &K) {
        let Some(number) = self.numbers.remove(key) else {
            return;
        };
        if let Some(slot) = self.entries.remove(&number) {
            self.order.remove(&slot.used);
            self.weight -= slot.weight;
        }
    }

    /// How many uses moved an entry in the order: none that used the latest used again.
    #[cfg(test)]
    pub(crate) fn reordered(&self) -> u64 {
        self.reordered
    }

    /// The entries' total weight and their count, checked against the entries themselves.
    #[cfg(test)]
    pub(crate) fn footprint(&self) -> (u64, usize) {
        let weight: u64 = self.entries.values().map(|slot| slot.weight).sum();
        assert_eq!(weight, self.weight, "the tracked weight is the entries'");
        assert_eq!(self.order.len(), self.entries.len(), "one stamp per entry");
        assert_eq!(self.numbers.len(), self.entries.len(), "one number per key");
        (weight, self.entries.len())
    }
}

/// A memo entry's key.
#[derive(Clone, PartialEq, Eq, Hash)]
enum MemoKey {
    Function(FunctionCompileKey),
    Compute(PipelineKey),
    Render(PipelineKey),
}

/// A memo entry.
enum MemoValue {
    Function(CompiledFunction),
    Compute(Arc<ComputePipeline>, Weak<Variant>),
    Render(Arc<RenderPipeline>, Weak<RenderVariant>),
}

/// The device's memo of compiled pipelines.
pub(crate) struct PipelineMemo {
    pipelines: Mutex<Lru<MemoKey, MemoValue>>,
    /// The device's specialized stage entries, which every pipeline creation of the device shares
    /// ([`PipelineMemo::stage_entries`]).
    entries: StageEntries,
    /// What [`PipelineMemo::note_stage_entries`] has counted of `entries`' own totals, so a
    /// creation adds only what it moved.
    noted: Mutex<(u64, u64)>,
}

/// What the memo knows of a compiled pipeline: the variant a live state made of it, or only the
/// pipeline, whose objects are made again.
pub(crate) enum Remembered<P, V> {
    Live(Arc<V>),
    Compiled(Arc<P>),
}

impl PipelineMemo {
    /// An empty memo of at most `capacity` bytes and [`MEMO_ENTRIES`] entries.
    pub(crate) fn new(capacity: u64) -> PipelineMemo {
        PipelineMemo {
            pipelines: Mutex::new(Lru::new(capacity, MEMO_ENTRIES)),
            entries: StageEntries::new(),
            noted: Mutex::new((0, 0)),
        }
    }

    /// The device's specialized stage entries: one object every pipeline creation of the device
    /// gives to its checks, its keys and its compile, so a stage's function is decoded and
    /// specialized once for the device and not once for each state made of it. An entry is a
    /// property of the function, its constants, its linked functions and its dynamic libraries
    /// alone, never of the state's own fields, so a sweep of states over one function pair shares
    /// one entry a stage; the object is bounded, and the least recently used entries go.
    pub(crate) fn stage_entries(&self) -> &StageEntries {
        &self.entries
    }

    /// Count what the device's stage entries have specialized and spared since this last read it
    /// (`WorkCounts::stage_entries_specialized`, `stage_entries_shared`).
    pub(crate) fn note_stage_entries(&self, counts: &crate::work_counts::Counters) {
        let (made, found) = self.entries.counts();
        let mut noted = lock(&self.noted);
        counts.stage_entries(made.saturating_sub(noted.0), found.saturating_sub(noted.1));
        *noted = (made, found);
    }

    /// The function compiled under `key`.
    pub(crate) fn function(&self, key: &FunctionCompileKey) -> Option<CompiledFunction> {
        match lock(&self.pipelines).get(&MemoKey::Function(*key)) {
            Some(MemoValue::Function(function)) => Some(function.clone()),
            _ => None,
        }
    }

    /// Keep `function`, whose encoding is `bytes` long.
    pub(crate) fn keep_function(&self, function: &CompiledFunction, bytes: u64) {
        lock(&self.pipelines).insert(
            MemoKey::Function(*function.key()),
            MemoValue::Function(function.clone()),
            bytes,
        );
    }

    /// The compute pipeline compiled under `key`.
    pub(crate) fn compute(
        &self,
        key: &PipelineKey,
    ) -> Option<Remembered<ComputePipeline, Variant>> {
        match lock(&self.pipelines).get(&MemoKey::Compute(*key)) {
            Some(MemoValue::Compute(pipeline, live)) => Some(remembered(pipeline, live)),
            _ => None,
        }
    }

    /// Keep `pipeline`, whose encoding is `bytes` long.
    pub(crate) fn keep_compute(&self, pipeline: &Arc<ComputePipeline>, bytes: u64) {
        lock(&self.pipelines).insert(
            MemoKey::Compute(pipeline.key),
            MemoValue::Compute(Arc::clone(pipeline), Weak::new()),
            bytes,
        );
    }

    /// Remember `variant` as the live variant of its pipeline, if the memo holds the pipeline.
    pub(crate) fn live_compute(&self, variant: &Arc<Variant>) {
        let key = MemoKey::Compute(variant.pipeline.key);
        if let Some(MemoValue::Compute(_, live)) = lock(&self.pipelines).get_mut(&key) {
            *live = Arc::downgrade(variant);
        }
    }

    /// The render pipeline compiled under `key`.
    pub(crate) fn render(
        &self,
        key: &PipelineKey,
    ) -> Option<Remembered<RenderPipeline, RenderVariant>> {
        match lock(&self.pipelines).get(&MemoKey::Render(*key)) {
            Some(MemoValue::Render(pipeline, live)) => Some(remembered(pipeline, live)),
            _ => None,
        }
    }

    /// Keep `pipeline` at `weight`: its encoding's length where its creation encoded it for the
    /// directory, else what it retains (`RenderPipeline::footprint`). Never zero, whichever way
    /// the creation kept it, or the memo's byte bound would stop bounding it.
    pub(crate) fn keep_render(&self, pipeline: &Arc<RenderPipeline>, weight: u64) {
        lock(&self.pipelines).insert(
            MemoKey::Render(pipeline.key),
            MemoValue::Render(Arc::clone(pipeline), Weak::new()),
            weight,
        );
    }

    /// Remember `variant` as the live variant of its pipeline, if the memo holds the pipeline.
    pub(crate) fn live_render(&self, variant: &Arc<RenderVariant>) {
        let key = MemoKey::Render(variant.pipeline().key);
        if let Some(MemoValue::Render(_, live)) = lock(&self.pipelines).get_mut(&key) {
            *live = Arc::downgrade(variant);
        }
    }

    /// The bytes and entries the memo holds.
    #[cfg(test)]
    pub(crate) fn footprint(&self) -> (u64, usize) {
        lock(&self.pipelines).footprint()
    }
}

/// The live variant `live` names, or else `pipeline`. On a warm pass's threads always `pipeline`
/// (`warm/pass.rs`): a warm state holding a variant of the program's could be the last to let it
/// go, and so hand its objects to the kept variants at a moment the program's order does not
/// decide.
fn remembered<P, V>(pipeline: &Arc<P>, live: &Weak<V>) -> Remembered<P, V> {
    let live = match crate::warm::warming() {
        true => None,
        false => live.upgrade(),
    };
    match live {
        Some(variant) => Remembered::Live(variant),
        None => Remembered::Compiled(Arc::clone(pipeline)),
    }
}
