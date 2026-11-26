//! The device's compiled pipelines by their content ([`KernelKey`]), compute and render alike in
//! one map: every pipeline state of one content shares one translation, its shader modules and
//! their `VkPipeline`s, and a state of a content the map holds costs a lookup. The map holds at
//! most [`KERNEL_ENTRIES`], forgetting the least recently used past it. It bounds what it holds,
//! not what lives: a state keeps the content it took, which goes once its last state and the map
//! have let it go. The translator's keys name their kind (`compute` or `render`), so a key of
//! one kind never meets a content of the other.

use crate::counters::{Counters, Gauge, Metric};
use crate::device::lock;
use crate::pipeline::Compiled;
use crate::render::RenderCompiled;
use metal2vulkan::cache::PipelineKey;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

/// The entries the map holds at most.
pub(crate) const KERNEL_ENTRIES: usize = 1024;

/// What a pipeline state compiles: the translator's key of its descriptor
/// (`Device::compute_pipeline_key`: the kernel's AIR and entry, its constant values, linked
/// functions and dynamic libraries, its lowering inputs, the descriptor's fields that change the
/// compile, the device's profile and the translator's semantics version; or
/// `Device::queried_render_descriptor_key`, every field of a render pipeline descriptor with its
/// functions keyed as a kernel is; neither key decodes a function), and a compute descriptor's
/// `requiredThreadsPerThreadgroup`, which the state answers as its `maxTotalThreadsPerThreadgroup`
/// and the translator's key leaves out (`None` for a render pipeline); and for a compute state's
/// sRGB variant, its texture slots that hold an 8-bit sRGB texture's UNORM twin, bit `i` slot `i`
/// (`PipelineInner::srgb_variant`), 0 for every other content.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct KernelKey {
    pub(crate) pipeline: PipelineKey,
    pub(crate) required: Option<[u32; 3]>,
    pub(crate) srgb: u128,
}

/// What one content compiles to.
#[derive(Clone)]
pub(crate) enum Content {
    Kernel(Arc<Compiled>),
    Render(Arc<RenderCompiled>),
}

impl Content {
    /// The address of what it holds, which names it.
    fn at(&self) -> *const () {
        match self {
            Content::Kernel(compiled) => Arc::as_ptr(compiled).cast(),
            Content::Render(compiled) => Arc::as_ptr(compiled).cast(),
        }
    }
}

/// The device's map of compiled contents.
pub(crate) struct KernelMap {
    map: Mutex<Lru<KernelKey, Content>>,
}

impl KernelMap {
    /// An empty map of [`KERNEL_ENTRIES`] entries.
    pub(crate) fn new() -> KernelMap {
        KernelMap {
            map: Mutex::new(Lru::new(KERNEL_ENTRIES)),
        }
    }

    /// The compiled kernel of `key`, where the map holds it.
    pub(crate) fn find(&self, key: &KernelKey) -> Option<Arc<Compiled>> {
        match lock(&self.map).get(key)? {
            Content::Kernel(compiled) => Some(compiled),
            Content::Render(_) => None,
        }
    }

    /// The compiled render pipeline of `key`, where the map holds it.
    pub(crate) fn find_render(&self, key: &KernelKey) -> Option<Arc<RenderCompiled>> {
        match lock(&self.map).get(key)? {
            Content::Render(compiled) => Some(compiled),
            Content::Kernel(_) => None,
        }
    }

    /// The kernel the map holds under `key`: the one it holds already (another thread compiled
    /// the content first), or `compiled`, kept now. Past the cap, the least recently used entry
    /// is forgotten, and goes once no state holds it.
    pub(crate) fn keep(
        &self,
        key: KernelKey,
        compiled: Arc<Compiled>,
        counters: &Counters,
    ) -> Arc<Compiled> {
        match self.keep_content(key, Content::Kernel(Arc::clone(&compiled)), counters) {
            Content::Kernel(kept) => kept,
            Content::Render(_) => compiled,
        }
    }

    /// The render pipeline the map holds under `key`, as [`keep`](Self::keep) keeps a kernel.
    pub(crate) fn keep_render(
        &self,
        key: KernelKey,
        compiled: Arc<RenderCompiled>,
        counters: &Counters,
    ) -> Arc<RenderCompiled> {
        match self.keep_content(key, Content::Render(Arc::clone(&compiled)), counters) {
            Content::Render(kept) => kept,
            Content::Kernel(_) => compiled,
        }
    }

    fn keep_content(&self, key: KernelKey, content: Content, counters: &Counters) -> Content {
        // What the map lets go is dropped after its lock, as a content's end gives the reaper
        // work.
        let ours = content.at();
        let (kept, gone) = lock(&self.map).keep(key, content);
        if kept.at() != ours {
            counters.count(Metric::KernelMakesDropped);
        }
        // A new entry beside the others is one more; one in place of the oldest, or the entry
        // held already, leaves as many.
        if gone.is_none() {
            counters.rise(Gauge::KernelEntries);
        }
        drop(gone);
        kept
    }

    /// Forgets every entry, at the device's end.
    pub(crate) fn clear(&self, counters: &Counters) {
        let held = lock(&self.map).clear();
        for _ in &held {
            counters.fall(Gauge::KernelEntries);
        }
        drop(held);
    }

    /// For the tests: a map of `cap` entries in place of [`KERNEL_ENTRIES`], set before any state.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn set_cap(&self, cap: usize) {
        lock(&self.map).cap = cap;
    }
}

/// A map of at most `cap` entries, the least recently used forgotten first.
pub(crate) struct Lru<K, V> {
    cap: usize,
    clock: u64,
    entries: HashMap<K, (V, u64)>,
}

impl<K: Eq + Hash + Clone, V: Clone> Lru<K, V> {
    /// An empty map of at most `cap` entries.
    pub(crate) fn new(cap: usize) -> Lru<K, V> {
        Lru {
            cap,
            clock: 0,
            entries: HashMap::new(),
        }
    }

    /// The value of `key`, now the most recently used.
    pub(crate) fn get(&mut self, key: &K) -> Option<V> {
        self.clock += 1;
        let clock = self.clock;
        let (value, used) = self.entries.get_mut(key)?;
        *used = clock;
        Some(value.clone())
    }

    /// The value held under `key`, now the most recently used: the one held already, or `value`,
    /// kept now; and what that let go: `value` where `key` held one already, the least recently
    /// used entry's where keeping `value` passed the cap, else nothing.
    pub(crate) fn keep(&mut self, key: K, value: V) -> (V, Option<V>) {
        self.clock += 1;
        let clock = self.clock;
        if let Some((held, used)) = self.entries.get_mut(&key) {
            *used = clock;
            return (held.clone(), Some(value));
        }
        let mut forgotten = None;
        if self.entries.len() >= self.cap.max(1) {
            let oldest = (self.entries.iter())
                .min_by_key(|(_, (_, used))| *used)
                .map(|(key, _)| key.clone());
            if let Some((value, _)) = oldest.and_then(|oldest| self.entries.remove(&oldest)) {
                forgotten = Some(value);
            }
        }
        self.entries.insert(key, (value.clone(), clock));
        (value, forgotten)
    }

    /// Every value, the map left empty.
    pub(crate) fn clear(&mut self) -> Vec<V> {
        self.entries.drain().map(|(_, (value, _))| value).collect()
    }

    /// The entries held.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}
