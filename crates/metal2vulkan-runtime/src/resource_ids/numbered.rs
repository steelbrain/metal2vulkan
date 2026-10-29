//! Numbered samplers (the consumer's ask of 2026-10-08): the plain sampler IDs Metal's pipeline
//! creation takes for a pipeline's samplers, and what holds them.
//!
//! Measured on an Apple M4 (macOS 27.0.1), plain Metal, in two sets of probes (an M3 Max on macOS
//! 26.7 answered the first set's 108 rows the same):
//!
//! - **Keys.** A pipeline state, as it is made, numbers each of its functions' samplers (the
//!   compile layer's [`NumberedSampler`]s: the constexpr samplers by state, `Read` and `Write`)
//!   and, for a render or mesh render pipeline with a fragment function, the device's own sampler
//!   ([`SamplerKey::Device`]). The order is the device's, then the fragment function's, the vertex
//!   or mesh function's, and the object function's samplers: the reverse of the pipeline's
//!   stages. A compute or tile pipeline numbers its function's alone. A key is device-wide: one a
//!   live holder holds keeps its ID, and any other takes a plain sampler ID, handed out as a
//!   sampler state's is ([`Pool`](super::Pool)), so the sampler states made after it start past
//!   it. A binary archive's add numbers as the state it makes does.
//! - **Holders.** The pipeline state holds every key it numbered. Each of its functions' holds
//!   ([`FunctionHolds`]) hold that function's keys, the fragment function's the device's too.
//!   Every function object of the same function of a library holds them: function objects made
//!   before the pipeline or after it, as long as the library or one of them lives. So does every
//!   live library of the same content ([`LibraryHolds`]: libraries of equal bytes made while one
//!   lives share it; one made once none lives holds nothing numbered before). A function object
//!   of another function, a library of other content, and a vertex function object for the
//!   device's key hold nothing. A function object outlives its library's holds: a library of the
//!   same content made after them is another's. When a key's last holder goes, its ID goes back
//!   to plain allocation.
//! - **The device's for good.** A command queue, an indirect command buffer and a texture whose
//!   memory is its own keep the device's sampler for the device's life
//!   ([`ResourceIds::take_device_sampler`]): the numbered one's ID where a holder holds it, else a
//!   plain one. A pipeline made after it numbers it no more.
//!
//! The runtime finds a pipeline function's holds by its library's bytes and its index: the
//! function objects of a library the program made from the same parsed library
//! ([`Library::bytes`](metal2vulkan::library::Library::bytes)) are one function's, as Metal's
//! function objects of one library object are.

use super::ResourceIds;
use crate::device::lock;
use metal2vulkan::cache::Hash256;
use metal2vulkan::pipeline::NumberedSampler;
use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, Mutex, Weak};

/// What a pipeline state numbers: one ID per key while a holder holds it (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum SamplerKey {
    /// The device's own sampler, which a render pipeline with a fragment function numbers first.
    Device,
    /// One of a function's samplers.
    Function(NumberedSampler),
}

/// One key's plain sampler ID, held while a holder holds this: it goes back to plain allocation
/// with the last, unless the device keeps it for good.
pub(crate) struct Numbered {
    ids: Arc<Mutex<ResourceIds>>,
    key: SamplerKey,
    id: u64,
}

impl fmt::Debug for Numbered {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Numbered")
            .field("key", &self.key)
            .field("id", &self.id)
            .finish()
    }
}

impl Drop for Numbered {
    fn drop(&mut self) {
        let mut all = lock(&self.ids);
        // A pipeline made while this went numbered the key again: its entry stays.
        if all
            .numbered
            .get(&self.key)
            .is_some_and(|(_, held)| std::ptr::eq(held.as_ptr(), self))
        {
            all.numbered.remove(&self.key);
        }
        if self.key == SamplerKey::Device && all.device_sampler == Some(self.id) {
            return;
        }
        all.count_sampler_id(false);
        all.samplers.plain.give(self.id);
    }
}

/// What every live library of one content holds: its functions' holds, one per function index,
/// each holding what pipelines of the function numbered (the module docs).
pub(crate) struct LibraryHolds {
    ids: Arc<Mutex<ResourceIds>>,
    content: Hash256,
    functions: Mutex<BTreeMap<u32, Arc<FunctionHolds>>>,
}

impl LibraryHolds {
    /// The holds of a library of `content` made on `ids`' device: a live library's of that
    /// content, else new ones.
    pub(crate) fn of(ids: &Arc<Mutex<ResourceIds>>, content: Hash256) -> Arc<LibraryHolds> {
        let mut all = lock(ids);
        if let Some(holds) = all.libraries.get(&content).and_then(Weak::upgrade) {
            return holds;
        }
        let holds = Arc::new(LibraryHolds {
            ids: Arc::clone(ids),
            content,
            functions: Mutex::default(),
        });
        all.libraries.insert(content, Arc::downgrade(&holds));
        holds
    }

    /// The holds of a function object of this content's function `name`
    /// ([`function_name`]): every function object of the function shares them, and these hold
    /// them while they live.
    pub(crate) fn function(&self, name: (usize, u32)) -> Arc<FunctionHolds> {
        let holds = Arc::clone(
            lock(&self.functions)
                .entry(name.1)
                .or_insert_with(|| Arc::new(FunctionHolds::new(&self.ids))),
        );
        lock(&self.ids)
            .functions
            .insert(name, Arc::downgrade(&holds));
        let mut names = lock(&holds.names);
        if !names.contains(&name) {
            names.push(name);
        }
        drop(names);
        holds
    }
}

impl fmt::Debug for LibraryHolds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LibraryHolds").finish_non_exhaustive()
    }
}

impl Drop for LibraryHolds {
    fn drop(&mut self) {
        let mut all = lock(&self.ids);
        if all
            .libraries
            .get(&self.content)
            .is_some_and(|holds| std::ptr::eq(holds.as_ptr(), self))
        {
            all.libraries.remove(&self.content);
        }
    }
}

/// What one function of a library's function objects hold: the keys the pipelines of the
/// function numbered (the module docs).
pub(crate) struct FunctionHolds {
    ids: Arc<Mutex<ResourceIds>>,
    held: Mutex<BTreeMap<SamplerKey, Arc<Numbered>>>,
    /// The (library bytes, index) names a pipeline finds these by.
    names: Mutex<Vec<(usize, u32)>>,
}

impl FunctionHolds {
    fn new(ids: &Arc<Mutex<ResourceIds>>) -> FunctionHolds {
        FunctionHolds {
            ids: Arc::clone(ids),
            held: Mutex::default(),
            names: Mutex::default(),
        }
    }

    /// Hold `numbered`, each key once.
    fn hold(&self, numbered: &[Arc<Numbered>]) {
        let mut held = lock(&self.held);
        for n in numbered {
            held.entry(n.key).or_insert_with(|| Arc::clone(n));
        }
    }
}

impl fmt::Debug for FunctionHolds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FunctionHolds").finish_non_exhaustive()
    }
}

impl Drop for FunctionHolds {
    fn drop(&mut self) {
        let me: *const FunctionHolds = self;
        let mut all = lock(&self.ids);
        for name in lock(&self.names).iter() {
            if all
                .functions
                .get(name)
                .is_some_and(|holds| std::ptr::eq(holds.as_ptr(), me))
            {
                all.functions.remove(name);
            }
        }
    }
}

/// The name a pipeline finds `function`'s holds by: its parsed library's bytes, which the
/// library's clones share, and its index.
pub(crate) fn function_name(function: &metal2vulkan::library::Function) -> (usize, u32) {
    (
        function.library().bytes().as_ptr() as usize,
        function.index(),
    )
}

/// One function of a pipeline state being made ([`number`]), in Metal's order.
pub(crate) struct NumberedStage<'a> {
    /// The function's name ([`function_name`]), where it is a library's: its function objects
    /// and library hold what it numbers. A compiled function has none.
    pub(crate) function: Option<(usize, u32)>,
    /// Whether it numbers the device's own sampler first: a render pipeline's fragment function.
    pub(crate) device: bool,
    /// Its samplers, in order.
    pub(crate) samplers: &'a [NumberedSampler],
}

/// Number the samplers of `stages`, a pipeline state's functions in Metal's order, on `ids`'
/// device (the module docs): the keys the state holds, each once. Each function's holds take
/// its keys. None on a warm pass's threads, whose states the program never sees: the numbers
/// are the program's creations' to take, in its order (`warm/pass.rs`).
pub(crate) fn number(
    ids: &Arc<Mutex<ResourceIds>>,
    stages: &[NumberedStage<'_>],
) -> Vec<Arc<Numbered>> {
    if crate::warm::warming() {
        return Vec::new();
    }
    // What a lock-holding step upgrades leaves the step: a holder's last handle must not go while
    // the lock is held, since its drop takes the lock.
    let per_stage: Vec<(Option<Arc<FunctionHolds>>, Vec<Arc<Numbered>>)> = {
        let mut all = lock(ids);
        let mut per_stage = Vec::with_capacity(stages.len());
        for stage in stages {
            let keys = stage
                .device
                .then_some(SamplerKey::Device)
                .into_iter()
                .chain(stage.samplers.iter().map(|&s| SamplerKey::Function(s)));
            let mut numbered = Vec::new();
            for key in keys {
                if key == SamplerKey::Device && all.device_sampler.is_some() {
                    continue;
                }
                let live = all.numbered.get(&key).and_then(|(_, held)| held.upgrade());
                let n = match live {
                    Some(n) => n,
                    None => {
                        let id = all.samplers.take_plain();
                        all.count_sampler_id(true);
                        let n = Arc::new(Numbered {
                            ids: Arc::clone(ids),
                            key,
                            id,
                        });
                        all.numbered.insert(key, (id, Arc::downgrade(&n)));
                        n
                    }
                };
                numbered.push(n);
            }
            let holds = stage
                .function
                .and_then(|name| all.functions.get(&name).and_then(Weak::upgrade));
            per_stage.push((holds, numbered));
        }
        per_stage
    };
    let mut state: Vec<Arc<Numbered>> = Vec::new();
    for (holds, numbered) in &per_stage {
        if let Some(holds) = holds {
            holds.hold(numbered);
        }
        for n in numbered {
            if !state.iter().any(|s| Arc::ptr_eq(s, n)) {
                state.push(Arc::clone(n));
            }
        }
    }
    state
}
