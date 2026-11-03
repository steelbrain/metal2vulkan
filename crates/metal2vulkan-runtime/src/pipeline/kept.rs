//! Released variants' Vulkan objects, kept for a state made again.
//!
//! A program can release a pipeline state and later make the same one again, as CoreAnimation
//! does, and Metal then gives the state from its caches at once. States alive together already
//! share their variant (`memo.rs`). Once the last holder of a variant's objects or of one of its
//! `VkPipeline`s releases it, the object comes here instead of being destroyed: its shader
//! modules, immutable samplers, set layouts and pipeline layout, and each `VkPipeline` the
//! variant made. A variant of the same compiled pipeline made later (the memo's key, so nothing
//! new decides equality) takes them back, with every `VkPipeline` the earlier one made: no
//! driver call, and on MoltenVK no SPIR-V to MSL conversion, MSL compile, specialization or
//! pipeline compile.
//!
//! - **Exact.** A `VkPipeline` goes back only to the objects of the variant instance whose
//!   layout made it: each is kept under its instance, and an instance's objects taken back keep
//!   the instance. A pipeline released after its instance's objects left (taken back, or
//!   evicted) waits for them under its instance, as a pipeline released first does.
//! - **Bounded.** At most [`KEPT`] instances, the least recently released going first; an
//!   evicted instance's objects and pipelines are destroyed then. An instance keeps at most what
//!   its variant's memo of `VkPipeline`s held.
//! - **On the guest's threads, in a fixed order** (the owner's HITCH rule 1). A last release can
//!   happen on any thread: a handler thread when a command buffer lets its objects go, a pool
//!   thread whose making evicts a memo's entry, or a program's thread. A release only queues there
//!   ([`Kept::objects_released`], [`Kept::pipeline_released`]): nothing moves in the order, nothing
//!   is evicted, nothing is destroyed. The queue is applied where a variant is made
//!   ([`Kept::take`]), on the program's thread that creates a state or draws (a host step's making
//!   neither takes nor settles), in the order of the instances' numbers, which the program's calls
//!   gave them ([`Number`]: a host step's from its command buffer's commit): so which instance is
//!   evicted, and when its immutable samplers stop counting against the device's limit
//!   (`SamplerLimit`), follow the program's own order of creations and releases. What remains
//!   timing's is only whether a command buffer let a variant go before a creation, as before.
//! - **Teardown.** The device destroys what is left at its teardown (`DeviceShared::drop`), the
//!   queue's included.

use super::graphics::GraphicsKey;
use super::objects::{SamplerSet, StageObject};
use super::shared_layouts::SharedLayout;
use super::RegionKey;
use crate::device::DeviceShared;
use crate::release;
use ash::vk;
use metal2vulkan::cache::PipelineKey;
use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

/// The most released variant instances kept.
pub(crate) const KEPT: usize = 16;

/// The compiled pipeline a variant is of: the device memo's key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Compiled {
    Compute(PipelineKey),
    Render(PipelineKey),
}

/// A `VkPipeline`'s place in its variant: its memo's key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Place {
    Region(RegionKey),
    Graphics(Box<GraphicsKey>),
}

/// One variant instance: the objects made once, and every `VkPipeline` made with its layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Instance {
    pub(crate) compiled: Compiled,
    number: Number,
}

impl Instance {
    /// A new instance of `compiled`, numbered in the program's order ([`Number::next`]).
    pub(crate) fn new(compiled: Compiled) -> Instance {
        Instance {
            compiled,
            number: Number::next(),
        }
    }
}

/// The counter the program's calls draw from: each variant instance made in one (a state's
/// creation, a draw) and each command buffer's commit ([`StepNumbers::reserve`]).
static NEXT: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// The numbers of the host step this thread runs now, if any: its command buffer's draw,
    /// and how many instances its host steps made so far ([`StepNumbers::numbering`]).
    static STEP: Cell<Option<(u64, u32)>> = const { Cell::new(None) };
}

/// An instance's place in the order the kept variants apply releases in (the module docs):
/// `(at, 0)` for one made in the program's call, `at` drawn then; `(at, n)` for the `n`th one
/// the host steps of a command buffer made (from 1), `at` drawn at its commit. So an instance a
/// host step makes falls where its command buffer's commit fell among the program's calls, and
/// after the step's earlier ones, however the steps' thread is timed against the program's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Number {
    at: u64,
    step: u32,
}

impl Number {
    /// The next number on this thread: the running host step's next, or a new draw.
    fn next() -> Number {
        STEP.with(|step| match step.get() {
            Some((at, made)) => {
                step.set(Some((at, made + 1)));
                Number { at, step: made + 1 }
            }
            None => Number {
                at: NEXT.fetch_add(1, Ordering::Relaxed),
                step: 0,
            },
        })
    }
}

/// The numbers a command buffer's host steps give the instances they make: drawn at its commit,
/// in the program's call, and counted across its steps, which the steps' thread
/// (`scheduler/steps.rs`) runs one after another in the batches' order.
pub(crate) struct StepNumbers {
    at: u64,
    made: AtomicU32,
}

impl StepNumbers {
    /// The numbers of a command buffer committed now.
    pub(crate) fn reserve() -> StepNumbers {
        StepNumbers {
            at: NEXT.fetch_add(1, Ordering::Relaxed),
            made: AtomicU32::new(0),
        }
    }

    /// Run `f`, one host step of the command buffer, its instances numbered from these (until
    /// it returns or unwinds).
    pub(crate) fn numbering<T>(&self, f: impl FnOnce() -> T) -> T {
        /// Puts back the thread's numbers, keeping the count of those the step took.
        struct Restore<'a> {
            numbers: &'a StepNumbers,
            outer: Option<(u64, u32)>,
        }
        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                if let Some((_, made)) = STEP.with(|step| step.replace(self.outer)) {
                    self.numbers.made.store(made, Ordering::Relaxed);
                }
            }
        }
        let made = self.made.load(Ordering::Relaxed);
        let outer = STEP.with(|step| step.replace(Some((self.at, made))));
        let _restore = Restore {
            numbers: self,
            outer,
        };
        f()
    }
}

/// A released instance's objects. They own no Vulkan object of their own: their immutable
/// samplers, layouts and stages' modules are the device's (`objects::SamplerSet`,
/// `shared_layouts.rs`, `content_keyed.rs`). Dropping them lets go of their shares, and only the
/// device's maps destroy each, where a program's call settles it or at the device's teardown,
/// after these.
pub(crate) struct KeptObjects {
    pub(crate) stages: Vec<StageObject>,
    pub(crate) samplers: Arc<SamplerSet>,
    /// Its set layouts and pipeline layout, a share of the device's (`shared_layouts.rs`).
    pub(crate) layout: Arc<SharedLayout>,
    /// The digest the variant's makings' records are named by (`objects::variant_content`).
    pub(crate) content: metal2vulkan::cache::Hash256,
    /// Each stage's words' digest (`objects::words_digest`), in stage order.
    pub(crate) words: std::sync::Arc<[metal2vulkan::cache::Hash256]>,
}

/// A kept `VkPipeline`: its place, and whether its specialized notices wait for a draw's first
/// use of it (made at a state's creation and never drawn).
pub(crate) struct KeptPipeline {
    pub(crate) place: Place,
    pub(crate) raw: vk::Pipeline,
    pub(crate) unannounced: bool,
    /// The device's linked pipeline it is, with its libraries, kept with it (`libraries.rs`).
    pub(crate) linked: Option<super::libraries::Linked>,
}

impl KeptPipeline {
    /// Destroy it where it is the variant's own, a pipeline made whole: a linked one is the
    /// device's, which other variants may share, destroyed by its map once nothing holds it
    /// (`libraries.rs`), and dropping this lets go of its share.
    fn destroy(&self, shared: &DeviceShared) {
        if self.linked.is_none() {
            // SAFETY: released by its last holder, so no pending command uses it.
            unsafe { release::object(shared.vk(), self.raw) };
        }
    }
}

/// What a variant made again takes back: its instance, objects and pipelines.
pub(crate) struct TakenBack {
    pub(crate) instance: Instance,
    pub(crate) objects: KeptObjects,
    pub(crate) pipelines: Vec<KeptPipeline>,
}

/// One kept instance: its objects once their holder released them, and its pipelines released.
struct Entry {
    instance: Instance,
    objects: Option<KeptObjects>,
    pipelines: Vec<KeptPipeline>,
}

impl Entry {
    /// Destroy its pipelines; dropping it lets go of its objects' shares.
    fn destroy(&self, shared: &DeviceShared) {
        for pipeline in &self.pipelines {
            pipeline.destroy(shared);
        }
    }
}

/// A release from any thread, queued until a variant is made ([`Kept::take`]).
enum Arrival {
    Objects(Instance, KeptObjects),
    Pipeline(Instance, KeptPipeline),
}

impl Arrival {
    fn instance(&self) -> Instance {
        match self {
            Arrival::Objects(instance, _) | Arrival::Pipeline(instance, _) => *instance,
        }
    }

    /// Destroy its pipeline; dropping it lets go of its objects' shares.
    fn destroy(&self, shared: &DeviceShared) {
        if let Arrival::Pipeline(_, pipeline) = self {
            pipeline.destroy(shared);
        }
    }
}

/// The device's kept instances, the least recently released first, and the releases queued
/// since the last variant was made.
#[derive(Default)]
pub(crate) struct Kept {
    entries: Entries,
    arrived: Vec<Arrival>,
}

/// The kept instances, the least recently released first.
#[derive(Default)]
struct Entries(VecDeque<Entry>);

impl Entries {
    /// The entry of `instance`, now the most recently released, made where there is none; an
    /// entry of another instance of its compiled pipeline is destroyed, as is the oldest past
    /// [`KEPT`].
    fn entry(&mut self, shared: &DeviceShared, instance: Instance) -> Option<&mut Entry> {
        let entries = &mut self.0;
        let mut at = None;
        let mut i = 0;
        while i < entries.len() {
            match entries.get(i) {
                Some(e) if e.instance == instance => {
                    at = Some(i);
                    i += 1;
                }
                Some(e) if e.instance.compiled == instance.compiled => {
                    if let Some(other) = entries.remove(i) {
                        other.destroy(shared);
                    }
                }
                _ => i += 1,
            }
        }
        let entry = match at.and_then(|i| entries.remove(i)) {
            Some(entry) => entry,
            None => Entry {
                instance,
                objects: None,
                pipelines: Vec::new(),
            },
        };
        while entries.len() >= KEPT {
            if let Some(oldest) = entries.pop_front() {
                oldest.destroy(shared);
            }
        }
        entries.push_back(entry);
        entries.back_mut()
    }

    /// Apply `arrival`, a release by its last holder: an instance's objects, or one of its
    /// `VkPipeline`s.
    fn apply(&mut self, shared: &DeviceShared, arrival: Arrival) {
        match arrival {
            Arrival::Objects(instance, objects) => {
                // An earlier release's objects, or these where no entry takes them, are dropped.
                if let Some(entry) = self.entry(shared, instance) {
                    entry.objects = Some(objects);
                }
            }
            Arrival::Pipeline(instance, pipeline) => match self.entry(shared, instance) {
                // The instance made its place again after taking the first back: one is enough.
                Some(entry) if !entry.pipelines.iter().any(|p| p.place == pipeline.place) => {
                    entry.pipelines.push(pipeline);
                }
                _ => pipeline.destroy(shared),
            },
        }
    }
}

impl Kept {
    /// `instance`'s objects, released by their last holder on any thread: queued until a variant
    /// is made (the module docs).
    pub(crate) fn objects_released(&mut self, instance: Instance, objects: KeptObjects) {
        self.arrived.push(Arrival::Objects(instance, objects));
    }

    /// `instance`'s `VkPipeline` `pipeline`, released by its last holder on any thread: queued
    /// as [`objects_released`](Self::objects_released) says.
    pub(crate) fn pipeline_released(&mut self, instance: Instance, pipeline: KeptPipeline) {
        self.arrived.push(Arrival::Pipeline(instance, pipeline));
    }

    /// Apply the queued releases, in the order of their instances' numbers (each instance's in
    /// the order they came): on the thread that makes a variant.
    fn settle(&mut self, shared: &DeviceShared) {
        self.arrived
            .sort_by_key(|arrival| arrival.instance().number);
        for arrival in self.arrived.drain(..) {
            self.entries.apply(shared, arrival);
        }
    }

    /// The kept instance of `compiled` whose objects are here, taken back, once the queued
    /// releases are applied: on the program's thread that makes a variant (the module docs).
    pub(crate) fn take(&mut self, shared: &DeviceShared, compiled: Compiled) -> Option<TakenBack> {
        debug_assert!(
            !super::pool::in_piece(),
            "the kept variants settled in a pipeline pool's piece"
        );
        // A host step's variant takes nothing back and settles nothing: the queue waits for the
        // next variant made on the program's thread.
        if crate::exit_guard::in_host_step() {
            return None;
        }
        self.settle(shared);
        let entries = &mut self.entries.0;
        let at = entries
            .iter()
            .position(|e| e.instance.compiled == compiled && e.objects.is_some())?;
        let entry = entries.remove(at)?;
        Some(TakenBack {
            instance: entry.instance,
            objects: entry.objects?,
            pipelines: entry.pipelines,
        })
    }

    /// Destroy everything kept, and every release queued: the device's teardown.
    pub(crate) fn destroy(&mut self, shared: &DeviceShared) {
        while let Some(entry) = self.entries.0.pop_front() {
            entry.destroy(shared);
        }
        for arrival in self.arrived.drain(..) {
            arrival.destroy(shared);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Number, StepNumbers};

    /// An instance a host step makes is numbered from its command buffer's commit: after what the
    /// program made before the commit and the step's earlier instances, before what the program
    /// made after the commit, however late the step runs and on whichever thread; the count runs
    /// on across the command buffer's steps, an unwinding step included, and outside a step a
    /// thread draws a new number again.
    #[test]
    fn a_host_step_s_instances_follow_their_commit_in_the_program_s_order() {
        let before = Number::next();
        let committed = StepNumbers::reserve();
        let after = Number::next();
        // A thread of the runtime's (`exit_guard::spawn`), as the scheduler's is.
        let (made, outside) = crate::exit_guard::spawn(std::thread::Builder::new(), move || {
            let (first, second) = committed.numbering(|| (Number::next(), Number::next()));
            // A step that unwinds, its panic caught as the program's code's are.
            let unwound = std::cell::Cell::new(true);
            crate::exit_guard::program(|| {
                committed.numbering(|| {
                    if Number::next().step == 3 {
                        panic!("a step that unwinds");
                    }
                });
                unwound.set(false);
            });
            assert!(unwound.get(), "the step unwound");
            let fourth = committed.numbering(Number::next);
            ([first, second, fourth], Number::next())
        })
        .unwrap()
        .join()
        .unwrap();
        let later = Number::next();
        assert_eq!(made.map(|number| number.step), [1, 2, 4]);
        assert!(made.iter().all(|number| number.at == made[0].at));
        assert!(before < made[0] && made[0] < made[1] && made[1] < made[2]);
        assert!(made[2] < after && after < later);
        assert_eq!((before.step, after.step, outside.step), (0, 0, 0));
        assert!(outside > after, "outside a step, a new draw");
    }
}
