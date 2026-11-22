//! Compute pipeline states: the translator's kernel and its interface, and its `VkPipeline`s.
//!
//! The translator answers, lowering nothing, whether a kernel observes its threadgroup partition
//! (`Device::kernel_observes_partition`), where the cache directory holds neither a lowering of
//! its partial-threadgroups variant nor the answer a device before kept there, either of which
//! answers it. A kernel that does not observe it is lowered once,
//! as that variant, which runs its grid in tiles (`RuntimeFact::GridTiles`):
//! every direct dispatch runs in workgroups of 256 threads, each one tile of the grid whose sides
//! the translator's rule picks from the grid's extents (`grid_tile`) and its region carries, on
//! one `VkPipeline` made with the state. An image-only kernel's (`CompiledFunction::is_image_only`)
//! runs instead in workgroups of 16x16x1 over a grid one deep, on that one, and of a tile the
//! grid and the threadgroup size pick (`encoder::image_tile`) over a deeper one, on one
//! `VkPipeline` per tile made at its first dispatch. An indirect
//! dispatch runs at its own threadgroup size. A kernel that observes its partition is lowered
//! plainly, and as its partial-threadgroups variant with the state: every dispatch, a whole
//! grid, threadgroups, an indirect dispatch or a `dispatchThreads` grid that is not a whole number
//! of threadgroups, runs the variant at its own threadgroup size, all but the last unflagged; the
//! dispatches where the content has no variant or the variant does not take the threadgroup run
//! the plain module.
//! Each `VkPipeline` is keyed by its module, threadgroup size and threadgroup memory lengths.
//!
//! What a state compiles is its content ([`KernelKey`]), never the state: the translator's key
//! of its descriptor and its `requiredThreadsPerThreadgroup`. Every state of one content shares
//! one [`Compiled`]: its translation, its shader modules and their `VkPipeline`s, its emulated
//! variant and its notes. A module past the stage bound is kept as its words and no shader
//! module: each make of its `VkPipeline`s makes one of its own and destroys it once the create
//! returns (`crate::stage`). The device's map of kernels holds each content's (`kernel_map`), so a
//! state of a content the map holds costs a lookup; past its cap the map forgets the least
//! recently used, which a state that took it keeps. What stays a state's own is what the consumer
//! gives it: the indices of its numbered samplers and the constexpr samplers' elements it holds
//! at them.
//!
//! A kernel whose tier runs each SIMD group as one subgroup, or as a 32-lane cluster of one, runs
//! only threadgroups of whole subgroups. A threadgroup that is not whole subgroups (32 threads
//! where each SIMD group is half of a 64-lane subgroup included) runs the kernel's
//! emulated-SIMD-group variant, whose SIMD groups exchange through threadgroup memory, at its own
//! size, whose last SIMD group may be partly filled, as Metal runs it; that variant is lowered
//! with the state too.
//!
//! What a state's make is given is made at the make, as Metal compiles a state when it is made,
//! up to the driver's create: every module ([`Lowered`] holds each variant's: a shader module, or
//! a module past the stage bound's words), with every refusal. The make then answers, and the
//! one `VkPipeline` whose key the descriptor holds, the content's main pipeline (a tiling
//! kernel's tiles, an image-only kernel's square workgroups, and an observing kernel's at the
//! descriptor's `requiredThreadsPerThreadgroup`, or, with none, at `[N, 1, 1]` for the bound `N`
//! of threads its state declares), is made by the consumer's
//! [`ComputePipelineState::finish_pipeline`] call on a thread of its own, so the driver's
//! compile overlaps the program's work; a kernel past the stage bound makes it at the make, so
//! that the stage bound's refusal is the make's. Nothing the runtime records depends on when a
//! compile finishes: an encoding call whose `VkPipeline` is not made yet makes it, or waits for
//! the make in flight, then records, so there is one recorded form. Each such wait is a
//! first-use wait, counted with its time
//! ([`Metrics::first_use_waits`](crate::Metrics::first_use_waits)). No translation runs after the
//! make. Each make of a `VkPipeline` is kept in the device's counters as it ends
//! ([`Counters::variant_makes`]): its key, who started it (a `finish_pipeline` call, the state's
//! make, a [`ComputePipelineState::prepare_dispatch`] call or a first use) and the first-use
//! waits that waited for it.

use crate::arrays::BINDINGS;
use crate::cache;
use crate::counters::{Counters, FirstUse, Gauge, Metric, VariantMake, VariantStart};
use crate::device::{lock, Core};
use crate::encoder::{
    decide, size_checked, threadgroup_memory, Dispatch, Lengths, Module, BUFFER_SLOTS,
};
use crate::error::{Error, Gap, IndirectCommandsKind, KernelInputKind, Refusal};
use crate::hazard::Access;
use crate::kernel_map::KernelKey;
use crate::reaper::{Free, Job};
use crate::ring::CHUNK;
use crate::sampler::{canonical, constexpr_notes, hold_constexpr, release_constexpr};
use crate::stage::{Create, Heavy};
use crate::times::host_time;
use ash::vk;
use metal2vulkan::cache::Hash256;
use metal2vulkan::error::{Error as TranslationError, Feature};
use metal2vulkan::library::{CompiledFunction, FunctionCompileKey};
use metal2vulkan::metal::{BindingAccess, BindingType};
use metal2vulkan::pipeline::{
    arena_bytes, AccessSet, Axis, BindingSource, BufferKind, CombinedSampler, ComputePipeline,
    ComputePipelineDescriptor, DescriptorSetLayout, FunctionDescriptor, KernelVariant,
    LoweringInputs, NumberedSampler, PrecisionNote, PushField, PushLoc, RuntimeFact, SamplerRef,
    SlotClass, SlotInterface, SlotRef, SpecKey, Stage, StageEntries, StageSource, TextureAccess,
    DEEP_THREADGROUP, GRID_TILE_THREADS,
};
use metal2vulkan::Budget;
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError};

/// The push window: the parameters a kernel reads from push constants, `[0, 256)` on every device,
/// the floor's `maxPushConstantsSize` and the translator's window on `Device::from_queries`'
/// profile: a dispatch region and 28 buffer addresses. Fields past it go to a block in the
/// parameter ring, whose address is a field in the window.
pub(crate) const PUSH_BYTES: u32 = crate::floor::PUSH_CONSTANT_BYTES;

/// The stages of the layout's one push range, which every `vkCmdPushConstants` names: a
/// dispatch's and a draw's pushes are of one range over the compute, vertex and fragment stages.
pub(crate) const PUSH_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;

/// `MTLComputePipelineState`.
///
/// A cheap handle: clones share one state. A command buffer that dispatches with it keeps it until
/// it retires. States of one content (the kernel, its constant values and linked functions, and
/// the descriptor's fields that change the compile) share its translation and `VkPipeline`s;
/// each numbers its own samplers. It answers no `gpuResourceID`, which Metal's state has
/// (`MTLComputePipelineState.gpuResourceID`).
#[derive(Clone)]
pub struct ComputePipelineState {
    pub(crate) inner: Arc<PipelineInner>,
}

/// No Metal counterpart: the `VkPipeline` a dispatch of a compute state runs, as a key for a
/// consumer's pre-pass beside [`ComputePipelineState::prepare_dispatch`]
/// ([`ComputePipelineState::dispatch_variant`]). Two keys are equal exactly when their
/// dispatches run the same `VkPipeline`: of one compiled content, the same module, workgroup size
/// and threadgroup memory lengths. The states of one content share its `VkPipeline`s, so their
/// keys compare equal; keys of different contents never do, on any device, even once one content
/// is gone, so one memo serves every state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DispatchVariant {
    content: u64,
    module: Module,
    local: [u32; 3],
    lengths: Lengths,
}

impl DispatchVariant {
    /// Its `VkPipeline`'s key among its kernel's variants.
    fn key(&self) -> Key<'_> {
        (
            self.local,
            self.lengths.as_slice(),
            self.module == Module::Tiles,
        )
    }
}

#[cfg(test)]
impl DispatchVariant {
    /// For the tests: the key of content `content`'s whole module at one thread.
    pub(crate) fn of_content(content: u64) -> DispatchVariant {
        DispatchVariant {
            content,
            module: Module::Whole,
            local: [1, 1, 1],
            lengths: Lengths::default(),
        }
    }
}

/// Metal's `maxTotalThreadsPerThreadgroup` of a kernel nothing bounds: Apple's 1024, which only a
/// declared bound brings Metal's answer below
/// (`ComputePipelineMetalInfo::metal_max_total_threads_per_threadgroup`).
const METAL_MAX_THREADS: u32 = 1024;

/// The number the next [`Compiled`] takes: each content made has its own, never reused in the
/// process, which a [`DispatchVariant`] names it by.
static CONTENTS: AtomicU64 = AtomicU64::new(0);

pub(crate) struct PipelineInner {
    /// What the state's content compiles to, shared with every state of that content.
    pub(crate) compiled: Arc<Compiled>,
    /// The samplers the state numbers and the indices the consumer gave them.
    pub(crate) numbered: Numbered,
}

/// What one compute pipeline content ([`KernelKey`]) compiles to, shared by every state of it
/// and held by the device's map of kernels until the map forgets it.
pub(crate) struct Compiled {
    core: Arc<Core>,
    /// Its number ([`CONTENTS`]).
    id: u64,
    /// For a kernel that observes its threadgroup partition, its partial-threadgroups variant
    /// ([`Partial::Alone`]), or the plain module where the state keeps one beside the variant
    /// ([`keeps_plain`]), which runs the dispatches the variant does not; for a kernel that does
    /// not observe it, the module of indirect dispatches, and of every dispatch where it is the
    /// only one. Its interface numbers the state's samplers and answers Metal's state.
    pub(crate) whole: Kernel,
    /// How a direct dispatch runs.
    pub(crate) form: Form,
    /// The emulated-SIMD-group variant of a kernel whose `whole` module runs only whole SIMD
    /// groups, lowered with the state.
    pub(crate) emulated: Option<Lowered>,
    /// The translator's notes of where the kernel's answers may differ from Metal's, of the
    /// modules made with the content, and its constexpr samplers' (`constexpr_notes`).
    notes: Vec<PrecisionNote>,
    /// What Metal's state answers of the content, beside its threads.
    answers: Answers,
    /// The key of the content's main pipeline ([`Compiled::main_key`]), which a state's
    /// [`ComputePipelineState::finish_pipeline`] makes, or its creation for a kernel past the
    /// stage bound; none where its descriptor holds no key of a `VkPipeline`.
    main: Option<DispatchVariant>,
}

/// What Metal's compute state answers of a content beside its threads, from its whole module's
/// translation.
#[derive(Clone, Copy, Default)]
struct Answers {
    /// `staticThreadgroupMemoryLength` (`ComputePipelineMetalInfo`).
    static_threadgroup_memory_length: u32,
    /// `threadExecutionWidth` (`ComputePipelineMetalInfo`).
    thread_execution_width: u32,
    /// Whether the kernel declares an imageblock, as the compile's kept reflection answers it;
    /// `None` where the compile kept none.
    imageblock: Option<bool>,
    /// `requiredThreadsPerThreadgroup`: the descriptor's, zeros where it set none.
    required_threads_per_threadgroup: [u32; 3],
}

impl Answers {
    /// `pipeline`'s answers, with `imageblock`.
    fn of(pipeline: &ComputePipeline, imageblock: Option<bool>) -> Answers {
        Answers {
            static_threadgroup_memory_length: pipeline.metal_info.static_threadgroup_memory_length,
            thread_execution_width: pipeline.metal_info.thread_execution_width,
            imageblock,
            required_threads_per_threadgroup: [0; 3],
        }
    }
}

/// Whether the kernel `function`, compiled for `descriptor`, declares an imageblock: what the
/// compile kept of its reflection answers, decoding nothing; `None` where it kept none (a binding
/// whose answer is unmeasured).
pub(crate) fn declares_imageblock(
    descriptor: &ComputePipelineDescriptor,
    function: &CompiledFunction,
    budget: &Budget,
) -> Option<bool> {
    let mut compiled = descriptor.clone();
    compiled.kernel = StageSource::Compiled(function.clone());
    let reflection = compiled.reflection(budget).ok()?;
    let mut bindings = reflection.bindings().iter();
    Some(bindings.any(|binding| matches!(binding.binding_type(), BindingType::Imageblock)))
}

/// The samplers a state numbers ([`Interface::numbered`]), the index the consumer gave each, and
/// the constexpr samplers' elements the state holds at theirs.
#[derive(Default)]
pub(crate) struct Numbered {
    samplers: Vec<NumberedSampler>,
    pub(crate) ids: Vec<u32>,
    /// Each constexpr sampler the state holds the elements of, by its place in `samplers`.
    held: Vec<usize>,
}

/// Whether a state of the module `whole` holds every constexpr sampler use of `later`, a variant
/// of the same entry: the samplers it numbers are its, and each use it samples
/// through is one of its, as a state holds the uses of its module ([`number`]); and whether its
/// texture slots' accesses are `whole`'s, which the state answers
/// ([`ComputePipelineState::texture_accesses`]) for every dispatch's hazards.
fn covers(whole: &Interface, later: &Interface) -> bool {
    later.numbered == whole.numbered
        && later.textures == whole.textures
        && later.constexprs.iter().all(|(at, sampler)| {
            (whole.constexprs.iter())
                .any(|(held, used)| held == at && sampler_array(used) == sampler_array(sampler))
        })
}

/// The sampler array of a constexpr sampler's use: 0 plain, 1 comparison, 2 integer texture.
fn sampler_array(sampler: &CombinedSampler) -> usize {
    match (sampler.compare, sampler.integer) {
        (true, _) => 1,
        (false, true) => 2,
        (false, false) => 0,
    }
}

/// The state's numbered samplers: `sampler_ids` gives an index to each of its module `whole`'s,
/// and each constexpr sampler it samples through takes the device sampler of each use at its
/// index.
fn number(
    core: &Core,
    whole: &Interface,
    sampler_ids: impl FnOnce(&[NumberedSampler]) -> Result<Vec<u32>, Error>,
) -> Result<Numbered, Error> {
    let samplers = whole.numbered.clone();
    let unstated = || gap(KernelInputKind::ConstexprSampler);
    number_samplers(core, samplers, &whole.constexprs, sampler_ids, unstated)
}

/// A state's numbered samplers `samplers`: `sampler_ids` gives an index to each, and each
/// constexpr sampler of `uses` (by its place in `samplers`, with its use) takes the device sampler
/// of each use at its index. `gap` is the error of a use whose sampler the translator cannot
/// state.
pub(crate) fn number_samplers(
    core: &Core,
    samplers: Vec<NumberedSampler>,
    uses: &[(usize, CombinedSampler)],
    sampler_ids: impl FnOnce(&[NumberedSampler]) -> Result<Vec<u32>, Error>,
    gap: fn() -> Error,
) -> Result<Numbered, Error> {
    let ids = sampler_ids(&samplers)?;
    if ids.len() != samplers.len() {
        return Err(Error::Refused(Refusal::SamplerIds {
            numbered: samplers.len(),
            given: ids.len(),
        }));
    }
    let profile = core.translator.profile();
    let mut numbered = Numbered {
        samplers,
        ids,
        held: Vec::new(),
    };
    for (at, sampler) in numbered.samplers.iter().enumerate() {
        let NumberedSampler::Constexpr(words) = *sampler else {
            continue;
        };
        // Any refusal gives back the elements the constexpr samplers before this one took.
        let held = constexpr_uses(profile, uses, at, gap).and_then(|held| {
            if held.iter().all(Option::is_none) {
                return Ok(false);
            }
            hold_constexpr(core, numbered.ids[at], words, held).map(|()| true)
        });
        match held {
            Ok(true) => numbered.held.push(at),
            Ok(false) => {}
            Err(error) => {
                numbered.release(core);
                return Err(error);
            }
        }
    }
    Ok(numbered)
}

/// The device sampler of each sampler array the uses of `uses` at place `at` sample through.
fn constexpr_uses(
    profile: &metal2vulkan::device::DeviceProfile,
    uses: &[(usize, CombinedSampler)],
    at: usize,
    gap: fn() -> Error,
) -> Result<[Option<metal2vulkan::vk::SamplerDesc>; 3], Error> {
    let mut held = [None; 3];
    for (_, sampler) in uses.iter().filter(|(place, _)| *place == at) {
        let desc = sampler
            .constexpr_sampler(profile)
            .map_err(Error::translation)?
            .ok_or_else(gap)?;
        held[sampler_array(sampler)] = Some(canonical(desc));
    }
    Ok(held)
}

impl Numbered {
    /// Gives up the constexpr samplers' elements the state holds.
    pub(crate) fn release(&mut self, core: &Core) {
        for at in self.held.drain(..) {
            if let NumberedSampler::Constexpr(words) = self.samplers[at] {
                release_constexpr(core, self.ids[at], words);
            }
        }
    }
}

/// Which of a content's variants, each lowered with the state.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Which {
    /// [`Compiled::emulated`].
    Emulated,
    /// [`Form::Sized`]'s partial-threadgroups variant.
    Partial,
}

/// What the lowering of a kernel's variant gave at its state's make: its emulated-SIMD-group
/// variant (`LoweringInputs::emulated_simdgroups`), or an observing kernel's partial-threadgroups
/// variant (`LoweringInputs::partial_threadgroups`). A dispatch that needs it runs its module, or
/// reports why there is none.
pub(crate) enum Lowered {
    /// The module, and the translator's notes of it.
    Made(Box<Kernel>, Vec<PrecisionNote>),
    /// The translator refused the variant (`Unsupported`), now or on a device on the cache
    /// directory before (which keeps that it refused, not why): the need's own refusal, the
    /// variant's gap.
    Refused,
    /// The translator's lowering or compile of the variant failed otherwise, a translator bug or
    /// a limit among them: the need reports this error.
    Translation(TranslationError),
    /// The variant reads what the compute encoder does not supply ([`Gap::KernelInput`] of the
    /// kind).
    Unsupplied(KernelInputKind),
    /// `vkCreateShaderModule` failed.
    Failed(vk::Result),
}

/// How a state runs a direct dispatch.
pub(crate) enum Form {
    /// The kernel does not observe its threadgroup partition: `whole`, its partial-threadgroups
    /// variant, runs its grid in tiles (`RuntimeFact::GridTiles`): every direct dispatch in
    /// workgroups of `GRID_TILE_THREADS`, each a `grid_tile` tile of its grid, whatever the
    /// threadgroup size.
    Tiled,
    /// The kernel does not observe its partition and is image-only
    /// (`CompiledFunction::is_image_only`): `whole` runs a direct dispatch at `MultiRegion` 0,
    /// whatever the threadgroup size, in workgroups of [`SQUARE`] over a grid one deep and of its
    /// `encoder::image_tile` over a deeper one, a tile deeper than `DEEP_THREADGROUP` as one row
    /// ([`Constant::Deep`]).
    Square,
    /// The kernel observes its partition: a dispatch runs at its own threadgroup size, every one,
    /// direct or indirect, on the partial-threadgroups variant (`partial`), wherever the variant
    /// takes the dispatch's threadgroup and threadgroup memory (`encoder::decide`).
    Sized { partial: Partial },
}

/// An observing kernel's partial-threadgroups variant ([`Form::Sized`]).
pub(crate) enum Partial {
    /// The variant is the content's module, [`Compiled::whole`]: the make lowers no plain module
    /// beside it ([`keeps_plain`]).
    Alone,
    /// The variant, lowered with the state beside the plain module, which is the content's.
    Beside(Box<Lowered>),
    /// The content has no function to lower the variant from.
    Absent,
}

/// One translated module of a state, its interface, and the `VkPipeline`s made of it so far.
pub(crate) struct Kernel {
    code: Code,
    /// The module's entry point.
    entry: CString,
    pub(crate) interface: Interface,
    /// The `VkPipeline`s made so far, by their specialization.
    variants: Variants,
}

/// What a kernel's `VkPipeline`s are created from.
enum Code {
    /// A module at or under the stage bound: its shader module, made with the kernel and named
    /// by each of its `VkPipeline`s' creates.
    Module(vk::ShaderModule),
    /// A module past the stage bound: each `VkPipeline` is made from its helper's cache data
    /// alone, through a shader module of that make's own, made from the module's words and
    /// destroyed once the create returns (`crate::stage`). The kernel holds no shader module.
    Heavy(Heavy),
}

/// The variants of a kernel that each have a cell of their own.
const VARIANT_CELLS: usize = 8;

/// A kernel's `VkPipeline`s made so far, by their specialization. The first [`VARIANT_CELLS`]
/// made each fill a cell of their own, which a dispatch reads with no lock; any made past them go
/// in a list under the lock. A making holds the lock only to enter its variant as pending and to
/// keep what it made, so makes of different variants run at once and one variant is made once at
/// a time.
#[derive(Default)]
struct Variants {
    cells: [OnceLock<(Variant, vk::Pipeline)>; VARIANT_CELLS],
    more: Mutex<More>,
    /// Signalled when a make ends, made or failed: a dispatch waiting for its pending variant
    /// looks again.
    ended: Condvar,
}

/// What [`Variants`] keeps under its lock.
#[derive(Default)]
struct More {
    /// The variants made past the cells.
    made: Vec<(Variant, vk::Pipeline)>,
    /// The variants a thread is making now.
    pending: Vec<Making>,
}

/// A variant a thread is making now, and the first-use waits that wait for it
/// ([`VariantMake::waits`]): how many, and when the earliest began.
struct Making {
    variant: DispatchVariant,
    waits: u32,
    first_wait: u64,
}

impl More {
    fn pending(&self, key: Key<'_>) -> Option<usize> {
        (self.pending.iter()).position(|making| making.variant.key() == key)
    }
}

impl Variants {
    /// The pipeline of the variant of `size`, threadgroup memory `lengths` and `tiles`, where its
    /// cell holds it, read with no lock; `more`, the list past the cells, where they are all full.
    fn in_cells(&self, (size, lengths, tiles): Key<'_>) -> Result<vk::Pipeline, bool> {
        for cell in &self.cells {
            match cell.get() {
                Some((variant, pipeline)) if variant.is(size, lengths, tiles) => {
                    return Ok(*pipeline)
                }
                Some(_) => {}
                // Cells fill in order, under the lock: none after an empty one is full.
                None => return Err(false),
            }
        }
        Err(true)
    }

    /// The pipeline of the variant `key` where it is made, read under the lock `more` is.
    fn made(&self, more: &More, key: Key<'_>) -> Option<vk::Pipeline> {
        let (size, lengths, tiles) = key;
        match self.in_cells(key) {
            Ok(pipeline) => Some(pipeline),
            Err(false) => None,
            Err(true) => (more.made.iter())
                .find(|(variant, _)| variant.is(size, lengths, tiles))
                .map(|&(_, pipeline)| pipeline),
        }
    }

    /// Keeps `pipeline`, of `variant`, under the lock `more` is: in the first empty cell, else in
    /// `more`.
    fn keep(&self, more: &mut More, variant: Variant, pipeline: vk::Pipeline) {
        match self.cells.iter().find(|cell| cell.get().is_none()) {
            // Only a making fills a cell, under the lock: the empty one stays empty until this.
            Some(cell) => {
                let _ = cell.set((variant, pipeline));
            }
            None => more.made.push((variant, pipeline)),
        }
    }

    /// Every pipeline, taken out to give back.
    fn drain(&mut self) -> impl Iterator<Item = vk::Pipeline> + '_ {
        let more = self.more.get_mut().unwrap_or_else(PoisonError::into_inner);
        (self.cells.iter_mut())
            .filter_map(|cell| cell.take().map(|(_, pipeline)| pipeline))
            .chain(more.made.drain(..).map(|(_, pipeline)| pipeline))
    }
}

/// A variant entered as pending under [`Variants`]'s lock, for the thread that makes it, started
/// by `start` at `began`: the entry leaves when this ends, made or not, panicking included, the
/// make is kept in the device's counters ([`Counters::variant_makes`]), and the waiting dispatches
/// wake.
struct Pending<'a> {
    variants: &'a Variants,
    counters: &'a Counters,
    variant: DispatchVariant,
    start: VariantStart,
    began: u64,
    made: Option<(Variant, vk::Pipeline)>,
}

impl Drop for Pending<'_> {
    fn drop(&mut self) {
        let mut more = lock(&self.variants.more);
        let at = more.pending(self.variant.key());
        let making = at.map(|at| more.pending.swap_remove(at));
        let made = self.made.is_some();
        if let Some((variant, pipeline)) = self.made.take() {
            self.variants.keep(&mut more, variant, pipeline);
        }
        drop(more);
        let ended = host_time();
        let (waits, first_wait) = making.map_or((0, 0), |making| (making.waits, making.first_wait));
        self.counters.variant_made(|number| VariantMake {
            number,
            variant: self.variant,
            start: self.start,
            began: self.began,
            ended,
            made,
            waits,
            first_wait,
        });
        self.variants.ended.notify_all();
    }
}

/// A variant's key: its threadgroup size, its threadgroup memory lengths, whether it runs a
/// kernel's grid tiles.
pub(crate) type Key<'a> = ([u32; 3], &'a [u32], bool);

/// What the runtime needs of a kernel, from the translator.
pub(crate) struct Interface {
    /// Each buffer slot the kernel uses, with its access.
    pub(crate) buffers: Vec<(u32, Access)>,
    /// Each texture slot the kernel uses, with its access.
    pub(crate) textures: Vec<(u32, Access)>,
    /// Each texture slot the kernel reads through its storage image (`TextureSlot::storage`, a
    /// `read_write` slot it reads), which takes no swizzle: a swizzled texture bound there is
    /// `Gap::SwizzledReadWrite`.
    pub(crate) storage_reads: Vec<u32>,
    /// Each texture slot whose storage image the kernel declares without a format
    /// (`TextureSlot::declared_format`), whose reads and writes the bound texture's format answers
    /// without one: a slot declared with one (a texture atomic's) takes neither answer.
    pub(crate) formatless: Vec<u32>,
    /// The samplers a state of the kernel numbers, in Metal's order
    /// (`PipelineInterface::numbered_samplers`).
    pub(crate) numbered: Vec<NumberedSampler>,
    /// Each constexpr sampler the kernel samples through, by its place in
    /// [`Interface::numbered`], with its use.
    pub(crate) constexprs: Vec<(usize, CombinedSampler)>,
    /// Whether the kernel reads through an address that is not one of its bound slots.
    pub(crate) reads_unbound: bool,
    /// Whether the kernel writes through an address that is not one of its bound slots.
    pub(crate) writes_unbound: bool,
    /// Whether the kernel may reach a texture through its storage image through an address that
    /// is not one of its bound slots (`PipelineInterface::unbound_storage_textures`): an
    /// argument buffer's `access::write` or `access::read_write` texture, a resource ID.
    pub(crate) unbound_storage: bool,
    /// Whether the kernel writes memory another invocation or a later command can read
    /// (`PipelineInterface::writes_memory`): a dispatch of a kernel without it records nothing.
    pub(crate) has_effects: bool,
    /// Bytes of the push window the fields use.
    pub(crate) push_len: u32,
    /// The length of the ring block of the fields past the window, a multiple of 16; 0 where
    /// every field is in the window.
    pub(crate) block_len: u32,
    /// The offset in the window of the ring block's address, a `u64`, where there is a block.
    pub(crate) block_address: u32,
    /// The parameters.
    pub(crate) fields: Vec<Field>,
    /// Whether the kernel reads the stage-in region ([`FieldValue::StageInRegion`]), which a
    /// dispatch needs the encoder to have set.
    pub(crate) stage_in_region: bool,
    /// The bytes of each invocation's slot in the kernel's private arena
    /// (`RuntimeFact::PrivateArena`), where its private memory passes the translator's floor: a
    /// dispatch of it takes the runtime's arena (`crate::arena`, [`FieldValue::PrivateArena`]).
    pub(crate) private_arena: Option<u32>,
    /// The kernel's specialization constants: each one's id and what sets it.
    pub(crate) constants: Vec<(u32, Constant)>,
    /// The index of each threadgroup memory argument the kernel takes, in the order of
    /// [`Variant::threadgroup_lengths`].
    pub(crate) threadgroups: Vec<u32>,
    /// `maxTotalThreadsPerThreadgroup`.
    pub(crate) max_total_threads: u32,
    /// Metal's `maxTotalThreadsPerThreadgroup` for the kernel before this device's limits
    /// (`ComputePipelineMetalInfo::metal_max_total_threads_per_threadgroup`), or a required
    /// size's threads.
    pub(crate) metal_max_total_threads: u32,
    /// The threadgroup memory every threadgroup takes beside the lengths set: Metal's static part,
    /// or an emulated module's arena and scratch; and a predicated walk's vote word.
    pub(crate) memory: Memory,
    /// What every threadgroup's threads must be a multiple of: the subgroup size of a tier that
    /// runs each SIMD group as one subgroup, where the local size is flattened.
    pub(crate) multiples: Vec<u32>,
    /// The subgroup size the kernel's tier pins
    /// (`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`), where it pins one.
    pub(crate) required_subgroup_size: Option<u32>,
    /// The stage's create flags (`REQUIRE_FULL_SUBGROUPS`).
    pub(crate) stage_flags: vk::PipelineShaderStageCreateFlags,
}

/// The threadgroup memory of a module beside the lengths set.
#[derive(Clone, Copy)]
pub(crate) struct Memory {
    pub(crate) own: OwnMemory,
    /// A predicated walk's vote word: 4 bytes, or 0.
    pub(crate) vote: u32,
}

/// A module's own threadgroup memory.
#[derive(Clone, Copy)]
pub(crate) enum OwnMemory {
    /// Metal's static part (`staticThreadgroupMemoryLength`), the lengths set after it.
    Static(u32),
    /// An emulated module's (`RuntimeFact::SimdgroupEmulation`): its arena, whose static part is
    /// `arena`, holding the lengths set, and `per_simdgroup` bytes of scratch for each SIMD group
    /// of a threadgroup.
    Emulated {
        arena: Option<u32>,
        per_simdgroup: u32,
    },
}

impl Memory {
    /// The threadgroup memory a threadgroup of `threads` threads takes with the lengths `lengths`
    /// set, each rounded up to 16 bytes.
    pub(crate) fn bytes(&self, lengths: &[u32], threads: u64) -> u64 {
        let regions = lengths
            .iter()
            .map(|&length| u64::from(length).next_multiple_of(16))
            .sum::<u64>();
        let own = match self.own {
            OwnMemory::Static(bytes) => u64::from(bytes) + regions,
            OwnMemory::Emulated {
                arena,
                per_simdgroup,
            } => arena_bytes(arena, regions) + u64::from(per_simdgroup) * threads.div_ceil(32),
        };
        own + u64::from(self.vote)
    }
}

/// What sets a specialization constant.
#[derive(Clone, Copy)]
pub(crate) enum Constant {
    /// The threadgroup size on one axis, 0 to 2.
    Size(usize),
    /// The threadgroup's threads: the product of its size.
    Threads,
    /// The length of the threadgroup memory argument at this position of
    /// [`Interface::threadgroups`].
    ThreadgroupLength(usize),
    /// `MultiRegion`: 1 for a kernel's grid tiles ([`Variant::tiles`]), whose regions carry the
    /// tile; 0 for every other launch, whose region states the pipeline's own threadgroup size,
    /// so the kernel takes `L` from its workgroup size and reads no decode.
    MultiRegion,
    /// `DeepThreadgroup`: 1 where the threadgroup is deeper than `DEEP_THREADGROUP`, which then
    /// runs as one row of its threads, 0 for every other, whose workgroup is the threadgroup.
    Deep,
}

/// A parameter and where it goes.
pub(crate) struct Field {
    pub(crate) place: Place,
    pub(crate) value: FieldValue,
}

/// Where a parameter goes: a byte offset of the push window, or of the ring block.
#[derive(Clone, Copy)]
pub(crate) enum Place {
    Push(u32),
    Block(u32),
}

/// The word a buffer slot's [`FieldValue::BufferAddress`] parameter holds for `address`: the
/// address less 2^31, as the translator's identity domains read a buffer slot's `SlotGuestBase`
/// (`docs/API.md`, section 5). A module adds an access's 32-bit byte offset plus 2^31 to the word,
/// zero-extended, which is the address plus the offset read as signed, and adds 2^31 back where it
/// reads the address itself.
pub(crate) fn address_word(address: u64) -> u64 {
    address.wrapping_sub(1 << 31)
}

/// What a parameter holds.
#[derive(Clone, Copy)]
pub(crate) enum FieldValue {
    /// A `u64`: the [`address_word`] of the address of the buffer bound at the slot plus its
    /// offset, of the address of the slot's `set_bytes` copy, or of the nil buffer's address where
    /// nothing is bound. `mask` is the alignment the module takes that address to have, less 1
    /// (`RuntimeFact::AddressAlignment`), 0 where it takes none: a dispatch whose bound buffer's
    /// address it does not divide is refused (`Refusal::BufferOffsetAlignment`).
    BufferAddress { slot: u32, mask: u64 },
    /// Two `u32`s: the bytes of the buffer bound at the slot before its bind offset, then its
    /// length (a `set_bytes` copy: 0 and its length; nothing bound: 0 and 0), which the module
    /// checks each access of the slot at an offset that is not a constant against (the
    /// translator's `PushField::SlotBounds`).
    BufferBounds(u32),
    /// A `u64`: the address of the ring's zero range (`Ring::zero`), where a checked load outside
    /// its buffer reads (the translator's `PushField::ZeroAddress`).
    ZeroAddress,
    /// A `u64`: the `gpuResourceID` of the texture bound at the slot, 0 where none is.
    TextureId(u32),
    /// A `u64`: the `gpuResourceID` of the sampler state bound at the slot, 0 where none is.
    SamplerId(u32),
    /// A `u32`: the channel widths of the texture bound at the slot, as an integer write
    /// saturates to them (`metal::format::integer_channels_word`), 0 where none is.
    IntegerChannels(u32),
    /// A `u32`: how Metal's bicubic filter treats the texels of the texture bound at the slot
    /// (`metal::format::bicubic_texels_word`, under the swizzle it is sampled through),
    /// `metal::format::BICUBIC_TEXELS_UNBOUND` where none is; a texture of a format with no
    /// such treatment is refused at the dispatch or the draw ([`Gap::BicubicTexels`]).
    BicubicTexels(u32),
    /// A `u32`: the sampler index the consumer gave the constexpr sampler at this place of the
    /// state's numbered samplers (a render state's: the vertex function's, then the fragment
    /// function's).
    ConstexprSampler(usize),
    /// The dispatch's region, 32 bytes read as two `uvec4`: the first threadgroup of the dispatch
    /// in Metal's grid (0 for a whole dispatch) with the threadgroup size packed as
    /// `(x-1) | (y-1) << 10 | (z-1) << 20`, then Metal's threads per grid (exact for
    /// `dispatch_threads`, threadgroups times their size for `dispatch_threadgroups`, and zeros
    /// for an indirect dispatch, whose kernel takes the grid from its workgroup count) and 0.
    DispatchRegion,
    /// The compute encoder's stage-in region (`setStageInRegion:`), 32 bytes read as two
    /// `uvec4`: its origin and its size, each with a zero `w`.
    StageInRegion,
    /// A `u32` the content fixes: a post-tessellation vertex function's pulled layout's stride,
    /// the descriptor's.
    Word(u32),
    /// A `u64`: the address of the render encoder's tessellation factor buffer plus its offset
    /// (`setTessellationFactorBuffer:offset:instanceStride:`), which a post-tessellation vertex
    /// function's control stage reads each patch's factors from.
    TessellationFactors,
    /// A `u32`: the render encoder's alpha test reference (`setAlphaTestReferenceValue:`) as the
    /// 8-bit unorm code a fragment function under the private alpha test compares its color 0's
    /// alpha with (`metal2vulkan::bind::alpha_test_code`), Metal's 1 until the encoder sets it.
    AlphaTestReference,
    /// A `u32`: the index of the sampled-texture array element through which the fragment
    /// function reads colour attachment `n` of the pass (`PushField::FetchedColor`), the
    /// attachment's texture's second element: its view in the feedback layout, or of its image's
    /// shadow on a device without the attachment feedback loop (`crate::render_encoder`).
    FetchedColor(u32),
    /// A `u32`: the layers the draw's pass renders (`renderTargetArrayLength`, at least 1), at or
    /// past which a vertex function's `[[render_target_array_index]]` stores `Layer` 0
    /// (`PushField::RenderTargetArrayLength`).
    RenderTargetArrayLength,
    /// A `u64`: the address of a dispatch's slots in the private arena (`crate::arena`), each
    /// invocation's at its index in the dispatch times the slot's bytes
    /// (`PushField::PrivateArenaAddress`).
    PrivateArena,
    /// A `u32`: the viewports the render encoder set (`setViewports:count:`, at least 1), at or
    /// past which a vertex function's `[[viewport_array_index]]` stores `ViewportIndex` 0
    /// (`PushField::ViewportCount`).
    ViewportCount,
    /// A `u64`: the address of the execution range the vertex function of a state made for
    /// indirect command buffers tests its command against (`PushField::IndirectRange`): the
    /// ring's range of every command (`Ring::whole`) for a direct draw and an execution over a
    /// range the CPU gives, the range buffer's address plus its offset for an execution over a
    /// range read from a buffer.
    IndirectRange,
    /// A `u32`: the command of an execution the draw is (`PushField::IndirectSequence`), 0 for a
    /// direct draw; an execution's sequence index token pushes each command's own.
    IndirectSequence,
    /// A `u32`: the first command of the part of an execution the draw is
    /// (`PushField::IndirectBase`), 0 for a direct draw; an execution pushes each part's own.
    IndirectBase,
}

/// A `VkPipeline`'s specialization: the threadgroup size, the threadgroup memory lengths in the
/// order of [`Interface::threadgroups`], and whether it runs a kernel's grid tiles.
pub(crate) struct Variant {
    pub(crate) size: [u32; 3],
    pub(crate) threadgroup_lengths: Vec<u32>,
    /// The launches are a kernel's grid tiles (`MultiRegion` 1).
    pub(crate) tiles: bool,
}

impl Variant {
    /// Whether this is the variant of `size`, threadgroup memory `lengths` and `tiles`.
    fn is(&self, size: [u32; 3], lengths: &[u32], tiles: bool) -> bool {
        self.size == size && self.threadgroup_lengths == lengths && self.tiles == tiles
    }
}

/// The workgroup an image-only kernel that does not observe its threadgroup partition runs a grid
/// one deep in ([`Form::Square`]).
pub(crate) const SQUARE: [u32; 3] = [16, 16, 1];

impl crate::Device {
    /// `newComputePipelineStateWithDescriptor:options:reflection:error:`, without reflection.
    ///
    /// A state's content is its descriptor's kernel with its constant values, linked functions
    /// and dynamic libraries, the descriptor's fields that change the compile, and its
    /// `requiredThreadsPerThreadgroup`. Every state of one content shares its translation, its
    /// shader modules and their `VkPipeline`s: a state of a content the device holds (the last
    /// 1,024 contents used) costs a lookup, its samplers' numbering and nothing compiled. For a
    /// content the device does not hold, the translator compiles the kernel on the calling
    /// thread, which waits for it, within the translator's 20-second budget per lowering: a
    /// kernel that does not observe its threadgroup partition as its partial-threadgroups
    /// variant, whose main pipeline is its 256-wide tile's `VkPipeline`, or an image-only
    /// kernel's 16x16 workgroups' one, and one that does as its partial-threadgroups variant,
    /// which its dispatches run, and its plain module too only where the variant's predicated
    /// walk takes a vote word of threadgroup memory the plain module does not and the kernel
    /// takes a threadgroup memory length from the encoder, or its own memory with the word is
    /// past the device's `maxComputeSharedMemorySize`. Its main pipeline is the one `VkPipeline`
    /// its dispatches run, whole grids, ragged and indirect ones (the plain module's where the
    /// variant's threadgroup memory, the vote word with it, is past the device's), at the
    /// descriptor's `requiredThreadsPerThreadgroup` where it sets one, or else at `[N, 1, 1]`
    /// where the state declares a bound of threads (the descriptor's
    /// `maxTotalThreadsPerThreadgroup`, or the kernel's
    /// `[[max_total_threads_per_threadgroup(n)]]`), `N` the state's
    /// `maxTotalThreadsPerThreadgroup`, where the kernel takes no threadgroup memory length from
    /// the encoder. A kernel whose tier runs only whole SIMD groups has its
    /// emulated-SIMD-group variant lowered here too. The translator answers whether the kernel
    /// observes its partition without lowering, where the cache directory holds no lowering of
    /// the variant to answer it. The call makes every shader module and answers before the
    /// driver's create of the main pipeline, which the consumer's
    /// [`finish_pipeline`](ComputePipelineState::finish_pipeline) call makes after it, on a
    /// thread of its own, or else the first dispatch that runs it; a kernel past the stage bound
    /// makes its main pipeline here (Err below). Where the device has a cache directory
    /// ([`DeviceOptions::pipeline_cache_dir`]), a lowering a device made before is read from it
    /// in place of lowering again, and every `VkPipeline` is made through the pipeline cache it
    /// keeps there. The content's other `VkPipeline`s, of a threadgroup size and threadgroup
    /// memory lengths only a dispatch gives, are made by a
    /// [`prepare_dispatch`](ComputePipelineState::prepare_dispatch) call or by the first encoding
    /// call of any of its states that needs each ([`ComputePipelineState`]), which waits for it.
    ///
    /// [`DeviceOptions::pipeline_cache_dir`]: crate::DeviceOptions::pipeline_cache_dir
    ///
    /// The state numbers samplers as Metal's pipeline creation does
    /// (`PipelineInterface::numbered_samplers`): its constexpr samplers, then one for a texture
    /// read and one for a texture write. Once the kernel is lowered, `sampler_ids` takes them in
    /// Metal's order and answers the sampler index the consumer gives each, from its live
    /// [`IndexClass::Sampler`](crate::IndexClass::Sampler) grants. Each constexpr sampler the
    /// kernel samples through takes the device sampler of each use it has
    /// (`CombinedSampler::constexpr_sampler`, shared by create info as a sampler state's are) at
    /// its index of the sampler arrays before the call returns, and every dispatch pushes the
    /// index; states that number one constexpr sampler at one index share its elements, and the
    /// last of them to go gives them back. A constexpr sampler's index has one owner, as a
    /// texture's has ([`Device::new_texture`](crate::Device::new_texture)): a sampler state made
    /// there takes it, and a later state numbering the constexpr sampler there takes it back. The
    /// consumer keeps the index from the program's sampler states while a state numbering it
    /// there lives; the runtime does not check it. The read and write samplers take nothing at
    /// their indices. What `sampler_ids` answers as an error, the call answers; once it has
    /// answered, a failing call holds none of the indices.
    ///
    /// Err: [`Error::Translation`], the translator's typed refusal or a descriptor Metal refuses,
    /// naming the kernel; [`Gap::IndirectCommandBuffers`] of
    /// [`IndirectCommandsKind::ComputeCommands`] for a descriptor that sets
    /// `supportIndirectCommandBuffers`; [`Gap::KernelInput`] for a kernel that
    /// reads what the compute encoder does not supply yet; [`Refusal::SamplerIds`] where
    /// `sampler_ids` answers another count; [`Refusal::IndicesNotGranted`] for a constexpr
    /// sampler's index; [`Limit::DeviceSamplers`](crate::Limit::DeviceSamplers);
    /// [`Error::StageUnfilled`], whose request the consumer's helper fills before it makes the
    /// state again ([`Device::fill_stage`](crate::Device::fill_stage)), or [`Gap::StageSize`], for
    /// the main pipeline of a kernel past the stage bound; [`Error::Vulkan`] where a shader module
    /// cannot be made.
    pub fn new_compute_pipeline_state(
        &self,
        descriptor: &ComputePipelineDescriptor,
        sampler_ids: impl FnOnce(&[NumberedSampler]) -> Result<Vec<u32>, Error>,
    ) -> Result<ComputePipelineState, Error> {
        (self.compute_pipeline_state(descriptor, sampler_ids))
            .map_err(|error| error.of_stages([&descriptor.kernel]))
    }

    /// [`new_compute_pipeline_state`](Self::new_compute_pipeline_state), its translator's error
    /// naming no function.
    fn compute_pipeline_state(
        &self,
        descriptor: &ComputePipelineDescriptor,
        sampler_ids: impl FnOnce(&[NumberedSampler]) -> Result<Vec<u32>, Error>,
    ) -> Result<ComputePipelineState, Error> {
        if descriptor.support_indirect_command_buffers {
            return Err(Error::Unsupported(Gap::IndirectCommandBuffers {
                kind: IndirectCommandsKind::ComputeCommands,
            }));
        }
        let core = self.core();
        let key = KernelKey {
            pipeline: core.translator.compute_pipeline_key(descriptor),
            required: descriptor.required_threads_per_threadgroup,
        };
        let compiled = match core.kernels.find(&key) {
            Some(compiled) => compiled,
            None => {
                let compiled = Arc::new(Compiled::of(core, descriptor)?);
                core.kernels.keep(key, compiled, &core.counters)
            }
        };
        let numbered = number(core, &compiled.whole.interface, sampler_ids)?;
        Ok(ComputePipelineState {
            inner: Arc::new(PipelineInner { compiled, numbered }),
        })
    }
}

impl Compiled {
    /// The translation of `descriptor`, every variant lowered, with the `VkPipeline`s whose key
    /// the descriptor holds.
    fn of(core: &Arc<Core>, descriptor: &ComputePipelineDescriptor) -> Result<Compiled, Error> {
        let core = Arc::clone(core);
        let budget = Budget::standard();
        let translator = &core.translator;
        let required = descriptor.required_threads_per_threadgroup;
        let compiled = |function: CompiledFunction| {
            let mut compiled = descriptor.clone();
            compiled.kernel = StageSource::Compiled(function);
            translating(&core, || {
                translator.compile_compute_pipeline(&compiled, &budget)
            })
        };
        let (whole, notes, partial, image_only, source, answers) = match &descriptor.kernel {
            StageSource::Function(function) => {
                let entries = &core.entries;
                let plain = translating(&core, || {
                    translator.kernel_inputs_read(
                        function,
                        &descriptor.lowering_inputs(),
                        &budget,
                        entries,
                    )
                })
                .map_err(Error::translation)?;
                let source = (
                    function,
                    plain.kernel_variant(KernelVariant::EmulatedSimdgroups),
                );
                let inputs = plain.kernel_variant(KernelVariant::PartialThreadgroups);
                let lower =
                    |inputs: &LoweringInputs| lower(&core, function, inputs, &budget, entries);
                // The partial-threadgroups variant: what the cache directory holds, else lowered
                // where the kernel does not observe its partition; where the translator answers
                // that it observes it, it is lowered below, before the plain module the state
                // keeps beside it where it keeps one. The translator's refusal of the variant
                // (emulated simdgroups, a vote word its workgroup memory cannot hold) is kept like
                // a lowering, and so is its answer that the kernel observes it, which a device on
                // the directory then reads in place of asking.
                let variant_key = *translator.function_compile_key(function, &inputs).digest();
                let variant = if core.caches.refused(&variant_key) {
                    core.counters.count(Metric::TranslationsLoaded);
                    Found::Refused
                } else if let Some(kept) = kept(&core, function, &inputs) {
                    Found::Lowered(kept)
                } else if core.caches.observed(&variant_key) {
                    Found::Unlowered
                } else {
                    let observes = translating(&core, || {
                        translator.kernel_observes_partition(function, &inputs, &budget, entries)
                    })
                    .map_err(Error::translation);
                    core.note_entries();
                    if observes? {
                        cache::keep_observer(&core, &variant_key);
                        Found::Unlowered
                    } else {
                        match lower(&inputs) {
                            Ok(variant) => Found::Lowered(variant),
                            Err(TranslationError::Unsupported(refused))
                                if refused.feature == Feature::PartialThreadgroups =>
                            {
                                cache::keep_refusal(&core, &variant_key);
                                Found::Refused
                            }
                            Err(error) => return Err(Error::translation(error)),
                        }
                    }
                };
                match variant {
                    // A kernel that does not observe its partition runs its variant alone: in
                    // tiles where the variant states them, else at its own size.
                    Found::Lowered(variant) if !variant.observes_threadgroup_partition() => {
                        let image_only = variant.is_image_only();
                        let imageblock = declares_imageblock(descriptor, &variant, &budget);
                        let pipeline = compiled(variant).map_err(Error::translation)?;
                        let partial = (!grid_tiles(&pipeline)).then_some(Partial::Absent);
                        let answers = Answers::of(&pipeline, imageblock);
                        let (whole, notes) = made(&core, pipeline, required)?;
                        (whole, notes, partial, image_only, Some(source), answers)
                    }
                    // The kernel observes its partition, or its variant is refused: the variant,
                    // which alone is the content's module where it is made and the state keeps no
                    // plain module beside it (`keeps_plain`), else the plain module, and the
                    // variant beside it for the grids that need it.
                    variant => {
                        let max = core.limits.max_shared_memory;
                        let variant = link_variant(&core, descriptor, function, &inputs, variant)
                            .map(|(lowered, pipeline)| {
                                let imageblock = declares_imageblock(descriptor, &lowered, &budget);
                                let answers = Answers::of(&pipeline, imageblock);
                                (made_variant(&core, pipeline, required), answers)
                            });
                        match variant {
                            Ok((Lowered::Made(variant, notes), answers))
                                if !keeps_plain(&variant.interface, max) =>
                            {
                                let partial = Some(Partial::Alone);
                                (*variant, notes, partial, false, Some(source), answers)
                            }
                            variant => {
                                let mut variant =
                                    variant.map_or_else(|lowered| lowered, |made| made.0);
                                let plain = (|| {
                                    let plain = lower(&plain).map_err(Error::translation)?;
                                    let imageblock =
                                        declares_imageblock(descriptor, &plain, &budget);
                                    let plain = compiled(plain).map_err(Error::translation)?;
                                    let answers = Answers::of(&plain, imageblock);
                                    Ok::<_, Error>((made(&core, plain, required)?, answers))
                                })();
                                let ((whole, notes), answers) = match plain {
                                    Ok(plain) => plain,
                                    Err(error) => {
                                        variant.release(&core);
                                        return Err(error);
                                    }
                                };
                                let variant = beside(&core, variant, &whole.interface);
                                let partial = Some(Partial::Beside(Box::new(variant)));
                                (whole, notes, partial, false, Some(source), answers)
                            }
                        }
                    }
                }
            }
            // A kernel the consumer compiled: the variant it was compiled under.
            StageSource::Compiled(function) => {
                let pipeline = translating(&core, || {
                    translator.compile_compute_pipeline(descriptor, &budget)
                })
                .map_err(Error::translation)?;
                // Only a non-observer's partial variant runs its grid in tiles. It has no
                // emulated variant: the kernel keeps the variant it was compiled under. The
                // translator links a compiled kernel only for its own profile
                // (`ForeignCompiledFunction` otherwise), on which a non-observer's partial
                // variant states `GridTiles`: a module without it observes its partition or is
                // the plain module, and runs at its own size.
                let partial = (!grid_tiles(&pipeline)).then_some(Partial::Absent);
                let imageblock = declares_imageblock(descriptor, function, &budget);
                let answers = Answers::of(&pipeline, imageblock);
                let (whole, notes) = made(&core, pipeline, required)?;
                (
                    whole,
                    notes,
                    partial,
                    function.is_image_only(),
                    None,
                    answers,
                )
            }
        };
        let mut notes = notes;
        // The emulated-SIMD-group variant, lowered now, of a module that runs only whole SIMD
        // groups. It samples no constexpr sampler the whole module does not (`covers`).
        let emulated = source
            .filter(|_| !whole.interface.multiples.is_empty())
            .map(|(function, inputs)| {
                lower_variant(
                    &core,
                    descriptor,
                    function,
                    &inputs,
                    Found::Unlowered,
                    &whole.interface,
                )
            });
        let profile = core.translator.profile();
        notes.extend(constexpr_notes(profile, &whole.interface.constexprs));
        notes.sort_unstable();
        notes.dedup();
        // `partial` is `Some` for a kernel that runs at its own size: with its variant where it
        // has one.
        let form = match (partial, image_only) {
            (Some(partial), _) => Form::Sized { partial },
            (None, false) => Form::Tiled,
            (None, true) => Form::Square,
        };
        let answers = Answers {
            required_threads_per_threadgroup: required.unwrap_or([0; 3]),
            ..answers
        };
        let mut compiled = Compiled::new(core, whole, form, emulated, (notes, answers));
        compiled.main = compiled.main_key(descriptor);
        // A kernel past the stage bound makes its main pipeline here, so that a make the stage
        // bound refuses refuses the state's, which the consumer's helper then fills. Every other
        // error of the make is the dispatch's that makes it again, as a failed
        // `finish_pipeline`'s is.
        let main = compiled.main_kernel();
        if let Some((kernel, variant)) = main.filter(|(kernel, _)| kernel.is_heavy()) {
            match kernel.make_ahead(&compiled.core, variant, VariantStart::State) {
                Err(error) if past_stage_bound(&error) => return Err(error),
                Ok(_) | Err(_) => {}
            }
        }
        Ok(compiled)
    }

    /// The key of the content's main pipeline, the one `VkPipeline` whose key its descriptor
    /// holds, where it has one: a tiling kernel's tiles, or an image-only kernel's square
    /// workgroups, which its grids one deep run; for an observing kernel, the one its dispatches
    /// at the size its state gives run: the descriptor's `requiredThreadsPerThreadgroup`, the
    /// size its dispatches give, or, with none, `[N, 1, 1]` where the state declares a bound of
    /// threads, `N` its `maxTotalThreadsPerThreadgroup`. The bound is declared by the
    /// descriptor's `maxTotalThreadsPerThreadgroup`, or by the kernel's
    /// `[[max_total_threads_per_threadgroup(n)]]`, which alone brings Metal's answer below its
    /// 1024; the device's own limit is no declared bound. Its dispatches of that size run the
    /// partial-threadgroups variant's, which runs whole grids, ragged and indirect ones alike, or
    /// the whole module's where the content has no such variant or the variant's threadgroup
    /// memory, a predicated walk's vote word with it, is past the device's (`encoder::decide`);
    /// none where the module takes a threadgroup memory length from the encoder.
    fn main_key(&self, descriptor: &ComputePipelineDescriptor) -> Option<DispatchVariant> {
        let (module, size) = match &self.form {
            Form::Tiled => (Module::Tiles, [GRID_TILE_THREADS, 1, 1]),
            Form::Square => (Module::Whole, SQUARE),
            Form::Sized { .. } => {
                let interface = &self.whole.interface;
                let declared = descriptor.max_total_threads_per_threadgroup.is_some()
                    || interface.metal_max_total_threads < METAL_MAX_THREADS;
                let bound = declared.then_some([interface.max_total_threads, 1, 1]);
                let size = descriptor.required_threads_per_threadgroup.or(bound)?;
                let dispatch = Dispatch::Threadgroups([1; 3]);
                let (kernel, module, local, lengths) = self.dispatched(dispatch, size, &[])?;
                return (kernel.interface.threadgroups.is_empty())
                    .then(|| self.dispatch_key(module, local, lengths));
            }
        };
        let lengths = Lengths::zeros(self.whole.interface.threadgroups.len());
        Some(self.dispatch_key(module, size, lengths))
    }

    /// The content of `whole`, its form, emulated variant, notes and Metal's answers, counted
    /// live, with no main pipeline.
    fn new(
        core: Arc<Core>,
        whole: Kernel,
        form: Form,
        emulated: Option<Lowered>,
        (notes, answers): (Vec<PrecisionNote>, Answers),
    ) -> Compiled {
        core.counters.rise(Gauge::Kernels);
        Compiled {
            core,
            id: CONTENTS.fetch_add(1, Relaxed),
            whole,
            form,
            emulated,
            notes,
            answers,
            main: None,
        }
    }
}

/// Whether `pipeline`'s module runs its grid in tiles (`RuntimeFact::GridTiles`): the
/// partial-threadgroups variant of a kernel that does not observe its threadgroup partition.
fn grid_tiles(pipeline: &ComputePipeline) -> bool {
    pipeline
        .dependencies
        .iter()
        .any(|dependency| dependency.fact == RuntimeFact::GridTiles)
}

/// The lowering of the kernel `function` under `inputs` a device kept in the cache directory under
/// its key, which covers everything that changes the output, counted as read.
fn kept(
    core: &Core,
    function: &FunctionDescriptor,
    inputs: &LoweringInputs,
) -> Option<CompiledFunction> {
    let key = core.translator.function_compile_key(function, inputs);
    let kept = loaded(core, key.digest());
    if kept.is_some() {
        core.counters.count(Metric::TranslationsLoaded);
    }
    kept
}

/// What a state's make first finds of a kernel's variant.
enum Found {
    /// Lowered, or read from the cache directory.
    Lowered(CompiledFunction),
    /// The translator refused it, at this make or on a device on the cache directory before.
    Refused,
    /// Not lowered yet: the emulated-SIMD-group variant, which is lowered once the content's
    /// module is, or an observing kernel's partial-threadgroups one, lowered before the plain
    /// module its state keeps beside it where it keeps one ([`keeps_plain`]).
    Unlowered,
}

/// The variant of the kernel `function` under `inputs` (the emulated-SIMD-group or the
/// partial-threadgroups one), at its state's make: `found` where the make already lowered or read
/// it, else the lowering a device kept in the cache directory, or one made now and kept; the
/// translator's refusal is kept too. `whole` is the content's module, which a state numbers its
/// samplers from.
fn lower_variant(
    core: &Core,
    descriptor: &ComputePipelineDescriptor,
    function: &FunctionDescriptor,
    inputs: &LoweringInputs,
    found: Found,
    whole: &Interface,
) -> Lowered {
    let required = descriptor.required_threads_per_threadgroup;
    match link_variant(core, descriptor, function, inputs, found) {
        Ok((_, pipeline)) => beside(core, made_variant(core, pipeline, required), whole),
        Err(lowered) => lowered,
    }
}

/// [`lower_variant`]'s lowering and compile: the variant's lowering and its pipeline. Err: the
/// translator's refusal ([`Lowered::Refused`]) or failure ([`Lowered::Translation`]).
fn link_variant(
    core: &Core,
    descriptor: &ComputePipelineDescriptor,
    function: &FunctionDescriptor,
    inputs: &LoweringInputs,
    found: Found,
) -> Result<(CompiledFunction, ComputePipeline), Lowered> {
    let translator = &core.translator;
    let budget = Budget::standard();
    let compiled = match found {
        Found::Lowered(compiled) => compiled,
        Found::Refused => return Err(Lowered::Refused),
        Found::Unlowered => {
            let key = *translator.function_compile_key(function, inputs).digest();
            if core.caches.refused(&key) {
                core.counters.count(Metric::TranslationsLoaded);
                return Err(Lowered::Refused);
            }
            match lower(core, function, inputs, &budget, &core.entries) {
                Ok(compiled) => compiled,
                Err(TranslationError::Unsupported(_)) => {
                    cache::keep_refusal(core, &key);
                    return Err(Lowered::Refused);
                }
                Err(error) => return Err(Lowered::Translation(error)),
            }
        }
    };
    let mut descriptor = descriptor.clone();
    descriptor.kernel = StageSource::Compiled(compiled.clone());
    match translating(core, || {
        translator.compile_compute_pipeline(&descriptor, &budget)
    }) {
        Ok(pipeline) => Ok((compiled, pipeline)),
        Err(TranslationError::Unsupported(_)) => Err(Lowered::Refused),
        Err(error) => Err(Lowered::Translation(error)),
    }
}

/// A variant's `pipeline` as a module of the device, with the translator's notes of it, at a
/// state of `required` threads per threadgroup where it has one.
fn made_variant(core: &Core, pipeline: ComputePipeline, required: Option<[u32; 3]>) -> Lowered {
    let notes = pipeline.stage.precision_notes.clone();
    match Kernel::new(core, pipeline, required) {
        Ok(kernel) => Lowered::Made(Box::new(kernel), notes),
        Err(Unmade::Unsupplied(kind)) => Lowered::Unsupplied(kind),
        Err(Unmade::Failed(result)) => Lowered::Failed(result),
    }
}

/// The variant `lowered` beside the content's module `whole`: a module that samples a constexpr
/// sampler in a use the state does not hold reads what the encoder does not supply.
fn beside(core: &Core, lowered: Lowered, whole: &Interface) -> Lowered {
    match lowered {
        Lowered::Made(mut kernel, _) if !covers(whole, &kernel.interface) => {
            kernel.release(core);
            Lowered::Unsupplied(KernelInputKind::ConstexprSampler)
        }
        lowered => lowered,
    }
}

/// The content's module of `pipeline`, at a state of `required` threads per threadgroup where it
/// has one, and the translator's notes of it.
fn made(
    core: &Core,
    pipeline: ComputePipeline,
    required: Option<[u32; 3]>,
) -> Result<(Kernel, Vec<PrecisionNote>), Error> {
    let notes = pipeline.stage.precision_notes.clone();
    Ok((Kernel::new(core, pipeline, required)?, notes))
}

/// Whether an observing kernel's state keeps its plain module beside its partial-threadgroups
/// variant, whose module's interface is `variant`: where the variant's predicated walk takes a
/// vote word of threadgroup memory the plain module does not ([`Memory::vote`]), and the kernel
/// takes a threadgroup memory length from the encoder or its own memory with the word is past the
/// device's `max` bytes, so that a dispatch the plain module takes may find the variant past the
/// limit (`encoder::decide`). Every other dispatch either module takes, the variant takes.
fn keeps_plain(variant: &Interface, max: u32) -> bool {
    variant.memory.vote != 0
        && (!variant.threadgroups.is_empty() || variant.memory.bytes(&[], 0) > u64::from(max))
}

/// The lowering of `FunctionCompileKey` digest `key` a device kept in the cache directory, where
/// its bytes decode for that key; not counted.
pub(crate) fn loaded(core: &Core, key: &Hash256) -> Option<CompiledFunction> {
    let translator = &core.translator;
    let key = FunctionCompileKey::new(*key);
    (core.caches).translation(key.digest(), |bytes| {
        CompiledFunction::from_bytes_for_key(bytes, &key, translator).ok()
    })
}

/// The kernel `function` lowered under `inputs`: the lowering a device kept in the cache directory
/// ([`kept`]), or one made now and kept.
pub(crate) fn lower(
    core: &Core,
    function: &FunctionDescriptor,
    inputs: &LoweringInputs,
    budget: &Budget,
    entries: &StageEntries,
) -> Result<CompiledFunction, TranslationError> {
    if let Some(kept) = kept(core, function, inputs) {
        return Ok(kept);
    }
    let translator = &core.translator;
    let key = translator.function_compile_key(function, inputs);
    core.counters.count(Metric::Translations);
    let lowered = translating(core, || {
        translator.compile_function_from_entries(function, inputs, budget, entries)
    });
    core.note_entries();
    let lowered = lowered?;
    cache::keep_translation(core, key.digest(), lowered.to_bytes());
    Ok(lowered)
}

/// `call`, a call into the translator, its time counted ([`Metrics::translation_nanos`]).
///
/// [`Metrics::translation_nanos`]: crate::Metrics::translation_nanos
pub(crate) fn translating<T>(core: &Core, call: impl FnOnce() -> T) -> T {
    let began = host_time();
    let answer = call();
    core.counters.translation(host_time().saturating_sub(began));
    answer
}

/// The one `VkPipeline` that `create` makes through the cache it is given: the device's one
/// `VkPipelineCache` (`crate::cache`), which a creation before the reaper made it empty waits for.
/// `call` names `create`'s Vulkan call. Counted as made, and its call's time
/// (`Metrics::pipeline_creation_nanos`).
pub(crate) fn made_through_cache(
    core: &Core,
    call: &'static str,
    create: impl FnOnce(vk::PipelineCache) -> Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)>,
) -> Result<vk::Pipeline, Error> {
    let creation = cache::creation(core);
    let began = host_time();
    let made = create(creation.handle());
    core.counters
        .pipeline_creation(host_time().saturating_sub(began));
    drop(creation);
    let pipeline = made.map_err(|(_, result)| Error::Vulkan { call, result })?[0];
    core.counters.count(Metric::PipelinesCreated);
    core.counters.rise(Gauge::Pipelines);
    Ok(pipeline)
}

/// Whether `error` is a refusal of a kernel past the stage bound: a make that only the consumer's
/// helper, or nothing, lets succeed.
fn past_stage_bound(error: &Error) -> bool {
    matches!(
        error,
        Error::StageUnfilled(_) | Error::Unsupported(Gap::StageSize { .. })
    )
}

/// The runtime's view of the translator's kernel, or what the kernel reads that the compute
/// encoder does not supply ([`Gap::KernelInput`] of that kind), the first this finds.
pub(crate) fn interface(
    pipeline: &ComputePipeline,
    required_size: Option<[u32; 3]>,
) -> Result<Interface, KernelInputKind> {
    let translated = &pipeline.interface;
    let stage = &pipeline.stage;
    // Every argument is a pushed address, a pushed resource ID or index, or a specialization: no
    // descriptor but set 1's and no table the binder builds (the translator holds a kernel's
    // constant data in its module on this device's profile); the stage's flags are its
    // subgroups'.
    let stage_flags = vk::PipelineShaderStageCreateFlags::from_raw(stage.stage_flags.as_raw());
    let subgroups = vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS
        | vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE;
    let unsupplied = [
        (
            translated.needs_pointer_translation,
            KernelInputKind::PointerTranslation,
        ),
        (
            translated.needs_handle_translation,
            KernelInputKind::HandleTranslation,
        ),
        (!translated.stage_in.is_empty(), KernelInputKind::StageIn),
        (
            !pipeline.layout.sets.iter().all(of_the_device),
            KernelInputKind::DescriptorSet,
        ),
        (
            !subgroups.contains(stage_flags),
            KernelInputKind::StageFlags,
        ),
    ];
    if let Some(&(_, kind)) = unsupplied.iter().find(|(reads, _)| *reads) {
        return Err(kind);
    }
    // The bind-time facts the encoder supplies.
    let mut multiples = Vec::new();
    let mut aligned = Vec::new();
    let mut vote = 0;
    let mut private_arena = None;
    let mut own = OwnMemory::Static(pipeline.metal_info.static_threadgroup_memory_length);
    for dependency in &pipeline.dependencies {
        match dependency.fact {
            RuntimeFact::LocalSizeConstraint { multiple } => multiples.push(multiple),
            // The slot's address, a multiple of its argument's alignment.
            RuntimeFact::AddressAlignment {
                slot:
                    SlotRef {
                        stage: Stage::Kernel,
                        class: SlotClass::Buffer,
                        index,
                    },
                align,
            } if align.is_power_of_two() => aligned.push((index, u64::from(align) - 1)),
            // The walk's vote word: 4 bytes beside the threadgroup memory.
            RuntimeFact::PredicationVote => vote = 4,
            // Each invocation's slot of the private arena, which the encoder places
            // (`FieldValue::PrivateArena`).
            RuntimeFact::PrivateArena { slot_bytes } => private_arena = Some(slot_bytes),
            // The emulated tier's arena and scratch, in place of Metal's static part.
            RuntimeFact::SimdgroupEmulation {
                bytes_per_simdgroup,
                arena,
            } => {
                own = OwnMemory::Emulated {
                    arena,
                    per_simdgroup: bytes_per_simdgroup,
                }
            }
            RuntimeFact::EmulatedSimdgroups
            | RuntimeFact::ThreadsPerThreadgroup
            | RuntimeFact::ThreadgroupMemoryLength { .. }
            | RuntimeFact::LocalSizeLimits
            | RuntimeFact::DispatchGrid
            | RuntimeFact::IndirectGroupCount
            | RuntimeFact::PartialThreadgroups
            | RuntimeFact::GridTiles
            | RuntimeFact::DeviceAddresses { .. } => {}
            // The stage-in region, which the encoder pushes (`FieldValue::StageInRegion`).
            RuntimeFact::StageInRegion => {}
            // What a variant could relieve, which the module runs without: a predicated walk
            // that unit axes would make less divergent. Every dispatch runs on this module,
            // keeping its whole groups.
            RuntimeFact::UnitAxes { axes: 0, .. } => {}
            // What the translator checks against the descriptor when it compiles, and nothing on
            // the device reads.
            RuntimeFact::CallStackDepth { .. } | RuntimeFact::OpenClMode => {}
            // `RuntimeFact` is non-exhaustive: a fact this match does not name is one the encoder
            // does not supply, and the refusal names it.
            fact => {
                return Err(KernelInputKind::RuntimeFact { fact: fact.name() });
            }
        }
    }
    let mut buffers = Vec::new();
    let mut textures = Vec::new();
    let mut storage_reads = Vec::new();
    let mut formatless = Vec::new();
    for (slot, used) in &translated.slots {
        match (slot.stage, slot.class, used) {
            (Stage::Kernel, SlotClass::Buffer, SlotInterface::Buffer(buffer))
                if buffer.kind == BufferKind::Buffer =>
            {
                buffers.push((slot.index, access(buffer.access)));
            }
            (Stage::Kernel, SlotClass::Texture, SlotInterface::Texture(texture)) => {
                textures.push((slot.index, texture_access(texture.access)));
                if texture.storage && texture.access.read {
                    storage_reads.push(slot.index);
                }
                if texture.storage && !texture.declared_format {
                    formatless.push(slot.index);
                }
            }
            (Stage::Kernel, SlotClass::Sampler, SlotInterface::Sampler(_)) => {}
            _ => return Err(KernelInputKind::Slot),
        }
    }
    // The samplers a state of the kernel numbers, in Metal's order.
    let numbered: Vec<NumberedSampler> = translated
        .numbered_samplers
        .iter()
        .filter(|(stage, _)| *stage == Stage::Kernel)
        .flat_map(|(_, samplers)| samplers.iter().copied())
        .collect();
    let mut constexprs = Vec::new();
    let mut fields = Vec::new();
    let mut push_len = 0;
    let mut block_len = 0;
    let mut block_address = None;
    for placement in &translated.push {
        if placement.stage != Stage::Kernel {
            return Err(KernelInputKind::PushField);
        }
        push_len = placement
            .push_range
            .map_or(0, |(offset, size)| offset + size);
        block_len = placement.argument_table_size;
        for placed in &placement.fields {
            let place = match placed.loc {
                PushLoc::Push { offset } => Place::Push(offset),
                PushLoc::ArgumentTable { offset } => Place::Block(offset),
                _ => return Err(KernelInputKind::PushField),
            };
            let value = match (placed.field, place) {
                (PushField::ArgumentTableAddress, Place::Push(offset)) => {
                    block_address = Some(offset);
                    continue;
                }
                (
                    PushField::SlotGuestBase(SlotRef {
                        stage: Stage::Kernel,
                        class,
                        index,
                    }),
                    _,
                ) => match class {
                    SlotClass::Buffer => FieldValue::BufferAddress {
                        slot: index,
                        mask: aligned
                            .iter()
                            .find(|&&(slot, _)| slot == index)
                            .map_or(0, |&(_, mask)| mask),
                    },
                    SlotClass::Texture => FieldValue::TextureId(index),
                    SlotClass::Sampler => FieldValue::SamplerId(index),
                },
                (
                    PushField::TextureIntegerChannels(SlotRef {
                        stage: Stage::Kernel,
                        class: SlotClass::Texture,
                        index,
                    }),
                    _,
                ) => FieldValue::IntegerChannels(index),
                (
                    PushField::BicubicTexels(SlotRef {
                        stage: Stage::Kernel,
                        class: SlotClass::Texture,
                        index,
                    }),
                    _,
                ) => FieldValue::BicubicTexels(index),
                (PushField::ConstexprSampler(sampler), _) => {
                    let SamplerRef::Constexpr(words) = sampler.sampler else {
                        return Err(KernelInputKind::ConstexprSampler);
                    };
                    let at = numbered
                        .iter()
                        .position(|&numbered| numbered == NumberedSampler::Constexpr(words))
                        .ok_or(KernelInputKind::ConstexprSampler)?;
                    if sampler.unnormalized {
                        return Err(KernelInputKind::ConstexprSampler);
                    }
                    constexprs.push((at, sampler));
                    FieldValue::ConstexprSampler(at)
                }
                (PushField::DispatchRegion, _) => FieldValue::DispatchRegion,
                (PushField::StageInRegion, _) => FieldValue::StageInRegion,
                (
                    PushField::SlotBounds(SlotRef {
                        stage: Stage::Kernel,
                        class: SlotClass::Buffer,
                        index,
                    }),
                    _,
                ) => FieldValue::BufferBounds(index),
                (PushField::ZeroAddress, _) => FieldValue::ZeroAddress,
                (PushField::PrivateArenaAddress, _) => FieldValue::PrivateArena,
                _ => return Err(KernelInputKind::PushField),
            };
            fields.push(Field { place, value });
        }
    }
    // The block, where there is one, is in one ring chunk, and its address in the window.
    let block_address = match (block_len, block_address) {
        (0, None) => 0,
        (1.., Some(at)) if u64::from(block_len) <= CHUNK => at,
        _ => return Err(KernelInputKind::ArgumentBlock),
    };
    if push_len > PUSH_BYTES {
        return Err(KernelInputKind::PushBytes);
    }
    // The threadgroup memory arguments are the lengths the module specializes on, each by its
    // `[[threadgroup(n)]]` index.
    let threadgroups: Vec<u32> = stage
        .specialization
        .constants
        .iter()
        .filter_map(|constant| match constant.key {
            SpecKey::TgLength(index) => Some(index),
            _ => None,
        })
        .collect();
    // At most the lengths `setThreadgroupMemoryLength:atIndex:` sets, one at each of its indices,
    // which a dispatch holds inline (`Lengths`).
    if threadgroups.len() > BUFFER_SLOTS as usize {
        return Err(KernelInputKind::Specialization);
    }
    let mut constants = Vec::new();
    for constant in &stage.specialization.constants {
        if constant.size != 4 {
            return Err(KernelInputKind::Specialization);
        }
        let set = match constant.key {
            SpecKey::LocalSize(Axis::X) => Constant::Size(0),
            SpecKey::LocalSize(Axis::Y) => Constant::Size(1),
            SpecKey::LocalSize(Axis::Z) => Constant::Size(2),
            SpecKey::FlatLocalSize => Constant::Threads,
            SpecKey::TgLength(index) => Constant::ThreadgroupLength(
                threadgroups
                    .iter()
                    .position(|&argument| argument == index)
                    .ok_or(KernelInputKind::Specialization)?,
            ),
            SpecKey::MultiRegion => Constant::MultiRegion,
            SpecKey::DeepThreadgroup => Constant::Deep,
            _ => return Err(KernelInputKind::Specialization),
        };
        constants.push((constant.id, set));
    }
    // A multiple outside a flattened local size is a derivative group's, which a kernel over
    // buffers alone does not take.
    let flat = constants
        .iter()
        .any(|(_, set)| matches!(set, Constant::Threads));
    if !multiples.is_empty() && !flat {
        return Err(KernelInputKind::Workgroup);
    }
    // Every workgroup is within the floor's sizes: a flattened one is one row of the threadgroup,
    // and any other runs a threadgroup deeper than `DEEP_THREADGROUP` as one row, which a module
    // that cannot (compute derivatives) does not declare.
    let deep = constants
        .iter()
        .any(|(_, set)| matches!(set, Constant::Deep));
    if !flat && !deep {
        return Err(KernelInputKind::Workgroup);
    }
    let max_total_threads = match required_size {
        Some(size) => size.iter().product(),
        None => pipeline.metal_info.max_total_threads_per_threadgroup,
    };
    let metal_max_total_threads = match required_size {
        Some(size) => size.iter().product(),
        None => pipeline.metal_info.metal_max_total_threads_per_threadgroup,
    };
    Ok(Interface {
        buffers,
        textures,
        storage_reads,
        formatless,
        numbered,
        constexprs,
        reads_unbound: translated.reads_unbound_resources,
        writes_unbound: translated.writes_unbound_resources,
        unbound_storage: translated.unbound_storage_textures,
        has_effects: translated.writes_memory,
        push_len,
        block_len,
        block_address,
        stage_in_region: (fields.iter())
            .any(|field| matches!(field.value, FieldValue::StageInRegion)),
        private_arena,
        fields,
        constants,
        threadgroups,
        max_total_threads,
        metal_max_total_threads,
        memory: Memory { own, vote },
        multiples,
        required_subgroup_size: stage.required_subgroup_size,
        stage_flags,
    })
}

/// The refusal of a kernel that reads what the compute encoder does not supply, of `kind`.
fn gap(kind: KernelInputKind) -> Error {
    Error::Unsupported(Gap::KernelInput { kind })
}

/// Why [`Kernel::new`] made no module.
enum Unmade {
    /// The kernel reads what the compute encoder does not supply ([`Gap::KernelInput`] of the
    /// kind).
    Unsupplied(KernelInputKind),
    /// `vkCreateShaderModule` failed.
    Failed(vk::Result),
}

impl From<Unmade> for Error {
    fn from(unmade: Unmade) -> Error {
        match unmade {
            Unmade::Unsupplied(kind) => gap(kind),
            Unmade::Failed(result) => Error::Vulkan {
                call: "vkCreateShaderModule",
                result,
            },
        }
    }
}

/// Whether `set` of a pipeline's layout is the device's: set 1 the device's texture and sampler
/// arrays (those of them its functions read), any other set empty.
pub(crate) fn of_the_device(set: &DescriptorSetLayout) -> bool {
    match set.set {
        1 => set.bindings.iter().all(|binding| {
            let ty: vk::DescriptorType = binding.ty.into();
            binding.source == BindingSource::IdentityArray
                && BINDINGS.contains(&(binding.binding, ty, binding.count))
        }),
        _ => set.bindings.is_empty(),
    }
}

/// The hazard table's access of a texture slot a function samples, reads, writes, or several.
pub(crate) fn texture_access(used: TextureAccess) -> Access {
    match (used.sample || used.read, used.write) {
        (_, false) => Access::Read,
        (false, true) => Access::Write,
        (true, true) => Access::ReadWrite,
    }
}

/// The hazard table's access of a slot a function reads, writes, or both.
pub(crate) fn access(used: AccessSet) -> Access {
    match (used.read, used.write) {
        (_, false) => Access::Read,
        (false, true) => Access::Write,
        (true, true) => Access::ReadWrite,
    }
}

impl ComputePipelineState {
    /// No Metal counterpart: makes the state's main `VkPipeline` on the calling thread, through
    /// the device's pipeline cache, once the state's creation has answered. Metal compiles a
    /// compute state whole when the state is made. Here the creation does everything up to the
    /// driver's create (every lowering, every refusal and Metal's own errors, the shader modules
    /// and the samplers' numbering) and answers, and this call runs the driver's create, so a
    /// consumer can answer the program before the driver's compile ends and let that compile
    /// overlap the program's own work. The runtime starts no thread for it: the consumer calls it
    /// on a thread of its own after it has answered.
    ///
    /// The main pipeline is the one whose key the descriptor holds: a tiling kernel's tiles', an
    /// image-only kernel's square workgroups', or an observing kernel's at the descriptor's
    /// `requiredThreadsPerThreadgroup` or its declared bound of threads where the kernel takes no
    /// threadgroup memory length from the encoder
    /// ([`new_compute_pipeline_state`](crate::Device::new_compute_pipeline_state)). A state with
    /// none makes nothing here. A kernel past the stage bound makes its main pipeline at the
    /// creation, so that the stage bound's refusal is the creation's, and makes nothing here.
    ///
    /// It returns at once where the pipeline is made, or another thread is making it, so a
    /// second call makes nothing. A dispatch that needs the pipeline first waits for the make in
    /// flight, or makes it itself where none is: a first-use wait
    /// ([`Metrics::first_use_unfinished_waits`](crate::Metrics::first_use_unfinished_waits)),
    /// after which it records what it records with the pipeline made here. So a state this is
    /// never called for runs as one it was called for, its first dispatch making the pipeline.
    ///
    /// Counted as [`Metrics::finished_pipelines`](crate::Metrics::finished_pipelines), with their
    /// time.
    ///
    /// Err: [`Error::Vulkan`] where the driver's create failed, which Metal would have answered
    /// at the creation (Vulkan lets it fail for exhausted host or device memory, or with
    /// `VK_ERROR_UNKNOWN` or `VK_ERROR_VALIDATION_FAILED`), counted in
    /// [`Metrics::failed_finishes`](crate::Metrics::failed_finishes): the pipeline is not kept,
    /// the first dispatch that needs it makes it again, and that dispatch answers the driver's
    /// error where that make fails too.
    pub fn finish_pipeline(&self) -> Result<(), Error> {
        let compiled = &self.inner.compiled;
        let core = &compiled.core;
        let main = compiled.main_kernel();
        let Some((kernel, variant)) = main.filter(|(kernel, _)| !kernel.is_heavy()) else {
            return Ok(());
        };
        match kernel.make_ahead(core, variant, VariantStart::Finish) {
            Ok(Some(nanos)) => core.counters.finished(nanos),
            Ok(None) => {}
            Err(error) => {
                core.counters.finish_failed();
                return Err(error);
            }
        }
        Ok(())
    }

    /// Makes ahead, on the calling thread, what a dispatch of this state makes at its first use,
    /// so that the dispatch finds it made. Metal has no counterpart: this is a hook for a
    /// consumer's pre-pass. Metal compiles a compute state whole when the state is made, and so
    /// does the state's make here, every variant included, with its main pipeline made by
    /// [`finish_pipeline`](Self::finish_pipeline); but a kernel's `VkPipeline` depends on the
    /// threadgroup size and the threadgroup memory lengths, which arrive with the dispatch. A
    /// consumer that reads a command buffer's dispatches before it encodes them may call this for
    /// each, in their order, on threads of its own; the runtime starts none.
    ///
    /// `dispatch` and `threads_per_threadgroup` are the dispatch's own arguments, as the program
    /// passed them to [`dispatch_threads`](crate::ComputeCommandEncoder::dispatch_threads),
    /// [`dispatch_threadgroups`](crate::ComputeCommandEncoder::dispatch_threadgroups) or
    /// [`dispatch_threadgroups_with_indirect_buffer`](crate::ComputeCommandEncoder::dispatch_threadgroups_with_indirect_buffer);
    /// `threadgroup_memory_lengths` is what
    /// [`set_threadgroup_memory_length`](crate::ComputeCommandEncoder::set_threadgroup_memory_length)
    /// set on the encoder before it, as (index, length) pairs, each index once, an index not
    /// listed at 0.
    ///
    /// It decides what the dispatch runs as the dispatch decides it, then makes the `VkPipeline`
    /// of the dispatch's module, workgroup size, threadgroup memory lengths and grid tiles. It
    /// returns at once where that is made, or another thread is making it: a dispatch that needs
    /// it then waits for that thread alone. It makes nothing where the dispatch would refuse its
    /// threadgroup size, its variant or its threadgroup memory, or records nothing (a grid with no
    /// threads, a kernel that writes no memory), and it reports no other error than a kernel's past
    /// the stage bound: every other is the dispatch's own. What the dispatch checks of the
    /// encoder's bindings is not checked here. A make that failed is made again by the next call
    /// or dispatch that needs it.
    ///
    /// Counted as [`Metrics::made_ahead_variants`](crate::Metrics::made_ahead_variants).
    ///
    /// Err: the refusal the dispatch would meet of a kernel past the stage bound:
    /// [`Error::StageUnfilled`], whose request the consumer's helper fills before the dispatch is
    /// encoded ([`Device::fill_stage`](crate::Device::fill_stage)), or [`Gap::StageSize`].
    pub fn prepare_dispatch(
        &self,
        dispatch: Dispatch,
        threads_per_threadgroup: [u32; 3],
        threadgroup_memory_lengths: &[(u32, u32)],
    ) -> Result<(), Error> {
        let compiled = &self.inner.compiled;
        let (size, set) = (threads_per_threadgroup, threadgroup_memory_lengths);
        let Some((kernel, module, local, lengths)) = compiled.dispatched(dispatch, size, set)
        else {
            return Ok(());
        };
        let core = &compiled.core;
        let variant = compiled.dispatch_key(module, local, lengths);
        match kernel.make_ahead(core, variant, VariantStart::Ahead) {
            Ok(Some(nanos)) => core.counters.made_ahead(nanos),
            Err(error) if past_stage_bound(&error) => return Err(error),
            Ok(None) | Err(_) => {}
        }
        Ok(())
    }

    /// No Metal counterpart: a hook for a consumer's pre-pass, beside
    /// [`prepare_dispatch`](Self::prepare_dispatch), with its arguments. The key of the
    /// `VkPipeline` the dispatch runs, decided as the dispatch and `prepare_dispatch` decide it:
    /// dispatches whose keys are equal run one `VkPipeline`, so a pre-pass that calls
    /// `prepare_dispatch` once for each key it has not met makes each pipeline once. What the
    /// decision does not read does not change the key: the size of an observing kernel's grid
    /// beyond whether its last threadgroups are partial, or a tiling kernel's direct grid and
    /// threadgroup size.
    ///
    /// With the key, whether its `VkPipeline` is made, so a pre-pass can leave a made one alone:
    /// read from the kernel's first eight variants with no lock, and past them under the short
    /// lock of the kernel's list, which no make holds while the driver compiles. It may answer
    /// false for a key whose make ends after the read, and never answers true for a key not
    /// made. That bit is no answer of Metal's and moves as makes end: a state's main pipeline
    /// reads false until its [`finish_pipeline`](Self::finish_pipeline), a `prepare_dispatch`
    /// call or a dispatch has made it. The key does not move.
    ///
    /// None where the dispatch would refuse its threadgroup size, its variant or its threadgroup
    /// memory, or records nothing (a grid with no threads, a kernel that writes no memory): where
    /// `prepare_dispatch` makes nothing whatever was made before. It makes nothing and allocates
    /// nothing where it answers a key.
    pub fn dispatch_variant(
        &self,
        dispatch: Dispatch,
        threads_per_threadgroup: [u32; 3],
        threadgroup_memory_lengths: &[(u32, u32)],
    ) -> Option<(DispatchVariant, bool)> {
        let compiled = &self.inner.compiled;
        let (size, set) = (threads_per_threadgroup, threadgroup_memory_lengths);
        let (kernel, module, local, lengths) = compiled.dispatched(dispatch, size, set)?;
        let variant = compiled.dispatch_key(module, local, lengths);
        Some((variant, kernel.is_made(variant.key())))
    }

    /// `maxTotalThreadsPerThreadgroup`: the translator's answer for the device, which is Metal's
    /// limit or less where the device holds fewer threads in a workgroup; for a descriptor with
    /// `requiredThreadsPerThreadgroup`, that size's threads, as Metal answers. A dispatch of
    /// another size within it runs at the size it gives, where Metal's validation stops it.
    pub fn max_total_threads_per_threadgroup(&self) -> u32 {
        self.inner.compiled.whole.interface.max_total_threads
    }

    /// `staticThreadgroupMemoryLength`: the bytes of the kernel's statically sized threadgroup
    /// memory, as Metal reports them, rounded up to a multiple of 16 (the translator's
    /// `ComputePipelineMetalInfo::static_threadgroup_memory_length`). The threadgroup memory the
    /// runtime adds beside it (an emulated SIMD group's scratch, a vote word) is no part of
    /// Metal's, and is not counted.
    pub fn static_threadgroup_memory_length(&self) -> u32 {
        self.inner.compiled.answers.static_threadgroup_memory_length
    }

    /// `threadExecutionWidth`: 32 on every device, Metal's answer on Apple's GPUs (the
    /// translator's `ComputePipelineMetalInfo::thread_execution_width`). A kernel that observes
    /// its SIMD group runs it 32 wide on every device, and one that does not observe it cannot
    /// tell its width.
    pub fn thread_execution_width(&self) -> u32 {
        self.inner.compiled.answers.thread_execution_width
    }

    /// `imageblockMemoryLengthForDimensions:`: 0 for a kernel that declares no imageblock, as
    /// Metal answers, whatever the dimensions. A kernel that reaches an imageblock has no state
    /// here ([`Gap::KernelInput`]).
    ///
    /// Err: [`Gap::ImageblockLength`] for a kernel that declares an imageblock it never reaches,
    /// or whose compile kept no reflection to tell: its length is not measured against Metal's.
    pub fn imageblock_memory_length_for_dimensions(
        &self,
        _dimensions: [u32; 3],
    ) -> Result<u64, Error> {
        match self.inner.compiled.answers.imageblock {
            Some(false) => Ok(0),
            Some(true) | None => Err(Error::Unsupported(Gap::ImageblockLength)),
        }
    }

    /// `requiredThreadsPerThreadgroup`: the descriptor's size, `[0, 0, 0]` where it set none, as
    /// Metal answers (the conformance recordings `pipeline-facts-compute-plain` and
    /// `pipeline-facts-compute-described`, an Apple M3 Max under macOS 27.0.1).
    pub fn required_threads_per_threadgroup(&self) -> [u32; 3] {
        self.inner.compiled.answers.required_threads_per_threadgroup
    }

    /// `supportIndirectCommandBuffers`: false on every state the runtime makes, as a descriptor
    /// that asks for indirect command buffers is refused at the making
    /// ([`Gap::IndirectCommandBuffers`] of [`IndirectCommandsKind::ComputeCommands`]).
    pub fn support_indirect_command_buffers(&self) -> bool {
        false
    }

    /// No Metal counterpart as such: each texture slot the kernel uses, by index, ascending, with
    /// the access a dispatch of the state marks in its encoder's hazards for the texture bound
    /// there, read from the same interface: `ReadOnly` where the kernel samples, reads or only
    /// queries it, `WriteOnly` where it writes it and does not read it, `ReadWrite` where it does
    /// both (an atomic does both). It is what the kernel's operations do, not the argument's
    /// `access::` qualifier: a `read_write` slot the kernel only writes is `WriteOnly`. A texture
    /// an argument buffer, a heap or a resource ID reaches is in no slot. Metal's counterpart is
    /// reflection's `MTLBinding.access` of a texture binding, the declared access; the consumer
    /// builds reflection itself.
    pub fn texture_accesses(&self) -> Vec<(u32, BindingAccess)> {
        let interface = &self.inner.compiled.whole.interface;
        (interface.textures.iter())
            .map(|&(slot, access)| (slot, access.binding()))
            .collect()
    }

    /// No Metal counterpart as such: each buffer slot the kernel uses, by index, ascending, with
    /// the access a dispatch of the state marks in its encoder's hazards for the buffer bound
    /// there, read from the same interface: `WriteOnly` where the kernel writes the buffer and
    /// does not read it, `ReadWrite` where it does both (an atomic does both, and so does a store
    /// of part of a word, which reads the word), `ReadOnly` otherwise. It is what the kernel's
    /// operations do, not the argument's declaration: a `device` buffer the kernel only writes is
    /// `WriteOnly`. A buffer an argument buffer, a heap or an address reaches is in no slot, and so
    /// is a function table. Metal's counterpart is reflection's `MTLBinding.access` of a buffer
    /// binding, the declared access, which is `ReadOnly` for a `constant` or `const device`
    /// buffer and `ReadWrite` for a `device` one; the consumer builds reflection itself.
    pub fn buffer_accesses(&self) -> Vec<(u32, BindingAccess)> {
        let interface = &self.inner.compiled.whole.interface;
        (interface.buffers.iter())
            .map(|&(slot, access)| (slot, access.binding()))
            .collect()
    }

    /// No Metal counterpart: where the kernel's answers may differ from Metal's on this device,
    /// fixed per operation by the translator, for every module the state runs: those made with
    /// the state (an observing kernel's one-row module among them), its variants among them: the
    /// emulated-SIMD-group module that runs threadgroups of partial SIMD groups and an observing
    /// kernel's partial-threadgroups module. Metal's answers are its own GPU's. [`PrecisionNote::SwizzledOpaqueBlackBorder`] is among them where the kernel samples
    /// through a constexpr sampler whose address mode reads an opaque-black border (decided by
    /// the sampler's own constants, as
    /// [`SamplerState::precision_note`](crate::SamplerState::precision_note) is by its
    /// descriptor).
    pub fn precision_notes(&self) -> Vec<PrecisionNote> {
        let compiled = &self.inner.compiled;
        let mut notes = compiled.notes.clone();
        for which in [Which::Emulated, Which::Partial] {
            if let Some(Lowered::Made(_, more)) = compiled.variant(which) {
                notes.extend(more);
            }
        }
        notes.sort_unstable();
        notes.dedup();
        notes
    }

    /// For the device tests: a state of a hand-made kernel and its interface, run at each
    /// dispatch's own threadgroup size.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn for_test(
        device: &crate::Device,
        words: &[u32],
        interface: Interface,
    ) -> Result<ComputePipelineState, Error> {
        let core = Arc::clone(device.core());
        // SAFETY: the words are a valid SPIR-V module.
        let module = unsafe { Kernel::module(&core, words) }
            .map_err(Error::vulkan("vkCreateShaderModule"))?;
        let whole = Kernel {
            code: Code::Module(module),
            entry: c"main".to_owned(),
            interface,
            variants: Variants::default(),
        };
        let form = Form::Sized {
            partial: Partial::Absent,
        };
        let answers = Answers::default();
        let compiled = Compiled::new(core, whole, form, None, (Vec::new(), answers));
        Ok(ComputePipelineState {
            inner: Arc::new(PipelineInner {
                compiled: Arc::new(compiled),
                numbered: Numbered::default(),
            }),
        })
    }

    /// Whether two handles are one state.
    pub(crate) fn is(&self, other: &ComputePipelineState) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl PipelineInner {
    pub(crate) fn core(&self) -> &Arc<Core> {
        &self.compiled.core
    }
}

impl Compiled {
    /// The content's `which` variant, where it has one.
    fn variant(&self, which: Which) -> Option<&Lowered> {
        match (which, &self.form) {
            (Which::Emulated, _) => self.emulated.as_ref(),
            (
                Which::Partial,
                Form::Sized {
                    partial: Partial::Beside(lowered),
                },
            ) => Some(lowered),
            (Which::Partial, Form::Sized { .. } | Form::Tiled | Form::Square) => None,
        }
    }

    /// The module of the content's `which` variant, lowered with the state: [`Compiled::whole`]
    /// for a partial-threadgroups variant that is the content's module ([`Partial::Alone`]).
    ///
    /// Err: `refused` where the content has no such variant or the translator refused it
    /// ([`Lowered::Refused`]); [`Error::Translation`] where its lowering or compile failed
    /// otherwise; [`Gap::KernelInput`] where it reads what the encoder does not supply;
    /// [`Error::Vulkan`] where its shader module failed.
    pub(crate) fn variant_kernel(
        &self,
        which: Which,
        refused: impl FnOnce() -> Error,
    ) -> Result<&Kernel, Error> {
        if let (
            Which::Partial,
            Form::Sized {
                partial: Partial::Alone,
            },
        ) = (which, &self.form)
        {
            return Ok(&self.whole);
        }
        match self.variant(which) {
            Some(lowered) => lowered.kernel(refused),
            None => Err(refused()),
        }
    }

    /// The kernel whose module a dispatch of `module` runs, where the content has it made: the
    /// content's own for its whole module and its tiles, else its variant's.
    fn kernel_of(&self, module: Module) -> Option<&Kernel> {
        let which = match module {
            Module::Whole | Module::Tiles => return Some(&self.whole),
            Module::Partial => Which::Partial,
            Module::Emulated => Which::Emulated,
        };
        if let (
            Which::Partial,
            Form::Sized {
                partial: Partial::Alone,
            },
        ) = (which, &self.form)
        {
            return Some(&self.whole);
        }
        match self.variant(which)? {
            Lowered::Made(kernel, _) => Some(&**kernel),
            Lowered::Refused
            | Lowered::Translation(_)
            | Lowered::Unsupplied(_)
            | Lowered::Failed(_) => None,
        }
    }

    /// The content's main pipeline's kernel and key, where it has one ([`Compiled::main`]).
    fn main_kernel(&self) -> Option<(&Kernel, DispatchVariant)> {
        let variant = self.main?;
        Some((self.kernel_of(variant.module)?, variant))
    }

    /// Where a first use of `variant` waits ([`FirstUse`]): the content's main pipeline's
    /// ([`FirstUse::Unfinished`]), or a variant's no state of the content has run.
    pub(crate) fn first_use(&self, variant: DispatchVariant) -> FirstUse {
        match self.main == Some(variant) {
            true => FirstUse::Unfinished,
            false => FirstUse::Variant,
        }
    }

    /// The key of the dispatches that run `module` at workgroup size `local` with threadgroup
    /// memory `lengths`: its [`DispatchVariant`].
    pub(crate) fn dispatch_key(
        &self,
        module: Module,
        local: [u32; 3],
        lengths: Lengths,
    ) -> DispatchVariant {
        DispatchVariant {
            content: self.id,
            module,
            local,
            lengths,
        }
    }

    /// What a dispatch of `dispatch` at threadgroup size `size`, with the encoder's threadgroup
    /// memory lengths `set`, runs, decided as the dispatch decides it: its kernel, its module, its
    /// workgroup size and its kernel's threadgroup memory lengths. None where the dispatch refuses
    /// its threadgroup size, its variant or its threadgroup memory, or records nothing (a grid
    /// with no threads, a kernel that writes no memory).
    fn dispatched(
        &self,
        dispatch: Dispatch,
        size: [u32; 3],
        set: &[(u32, u32)],
    ) -> Option<(&Kernel, Module, [u32; 3], Lengths)> {
        let interface = &self.whole.interface;
        if size_checked(interface, size).is_err() || !interface.has_effects {
            return None;
        }
        let max = self.core.limits.max_shared_memory;
        let (choice, module, kernel) =
            decide(self, dispatch.direct(size), size, (set, max)).ok()?;
        if (choice.direct).is_some_and(|(_, groups, _)| groups.contains(&0)) {
            return None;
        }
        let lengths = threadgroup_memory(kernel, set, choice.local, max).ok()?;
        Some((kernel, module, choice.local, lengths))
    }
}

impl Lowered {
    /// Releases the module, where there is one.
    fn release(&mut self, core: &Core) {
        if let Lowered::Made(kernel, _) = self {
            kernel.release(core);
        }
    }

    /// The module, or the error a need of the variant reports: `refused` for the translator's
    /// refusal alone, so no other failure reads as the variant's gap.
    pub(crate) fn kernel(&self, refused: impl FnOnce() -> Error) -> Result<&Kernel, Error> {
        match self {
            Lowered::Made(kernel, _) => Ok(&**kernel),
            Lowered::Refused => Err(refused()),
            Lowered::Translation(error) => Err(Error::translation(error.clone())),
            &Lowered::Unsupplied(kind) => Err(gap(kind)),
            &Lowered::Failed(result) => Err(Error::Vulkan {
                call: "vkCreateShaderModule",
                result,
            }),
        }
    }
}

impl Kernel {
    /// The translator's `pipeline` as a module of the device, with its interface: its shader
    /// module made now, or, past the stage bound, its words kept for each make's own.
    fn new(
        core: &Core,
        pipeline: ComputePipeline,
        required_size: Option<[u32; 3]>,
    ) -> Result<Kernel, Unmade> {
        let interface = interface(&pipeline, required_size).map_err(Unmade::Unsupplied)?;
        let stage = pipeline.stage;
        let heavy = Heavy::of(&stage);
        // The translator's entry points hold no NUL; one that did would name no entry point.
        let entry = CString::new(stage.entry_point)
            .map_err(|_| Unmade::Unsupplied(KernelInputKind::EntryPoint))?;
        let code = match heavy {
            Some(heavy) => Code::Heavy(heavy),
            None => {
                // SAFETY: the words are the translator's SPIR-V, for the features device creation
                // enabled.
                let module = unsafe { Kernel::module(core, &stage.spirv) };
                Code::Module(module.map_err(Unmade::Failed)?)
            }
        };
        Ok(Kernel {
            code,
            entry,
            interface,
            variants: Variants::default(),
        })
    }

    /// A shader module of `words`.
    ///
    /// # Safety
    ///
    /// `words` are a valid SPIR-V module for the features device creation enabled.
    unsafe fn module(core: &Core, words: &[u32]) -> Result<vk::ShaderModule, vk::Result> {
        let info = vk::ShaderModuleCreateInfo::default().code(words);
        // SAFETY: the caller's contract.
        unsafe { core.device.create_shader_module(&info, None) }
    }

    /// For the device tests: whether the kernel holds a shader module of its own.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn holds_module(&self) -> bool {
        matches!(self.code, Code::Module(_))
    }

    /// Whether the kernel is past the stage bound: each of its `VkPipeline`s is made from its
    /// helper's cache data alone ([`Code::Heavy`]).
    fn is_heavy(&self) -> bool {
        matches!(self.code, Code::Heavy(_))
    }

    /// The `VkPipeline` of `variant`, made now where it is not yet: the calling thread waits for
    /// it, a first-use wait at `site` ([`Compiled::first_use`]), so what is recorded never
    /// depends on when a compile finishes. Where another thread is making it, the call waits for
    /// that make, also a first-use wait counted in that make's [`VariantMake::waits`], and makes
    /// it itself where that one failed. One of the kernel's first [`VARIANT_CELLS`] variants is
    /// found with no lock.
    pub(crate) fn pipeline(
        &self,
        core: &Core,
        variant: DispatchVariant,
        site: FirstUse,
    ) -> Result<vk::Pipeline, Error> {
        let variants = &self.variants;
        let key = variant.key();
        if let Ok(pipeline) = variants.in_cells(key) {
            return Ok(pipeline);
        }
        let began = host_time();
        let waited =
            |core: &Core| (core.counters).first_use_wait(site, host_time().saturating_sub(began));
        let mut more = lock(&variants.more);
        let mut waits = false;
        loop {
            if let Some(pipeline) = variants.made(&more, key) {
                if waits {
                    waited(core);
                }
                return Ok(pipeline);
            }
            let Some(at) = more.pending(key) else {
                break;
            };
            if !waits {
                let making = &mut more.pending[at];
                making.waits = making.waits.saturating_add(1);
                making.first_wait = match making.first_wait {
                    0 => began,
                    first => first.min(began),
                };
            }
            waits = true;
            more = (variants.ended.wait(more)).unwrap_or_else(PoisonError::into_inner);
        }
        let start = VariantStart::FirstUse;
        let pipeline = self.make_pending(core, more, variant, start, Some(began))?;
        waited(core);
        Ok(pipeline)
    }

    /// Whether the variant `key` is made: read from its cell with no lock, and past the cells under
    /// the list's lock, which no make holds while it compiles. False for a make that ends after
    /// the read.
    fn is_made(&self, key: Key<'_>) -> bool {
        let variants = &self.variants;
        match variants.in_cells(key) {
            Ok(_) => true,
            Err(false) => false,
            Err(true) => variants.made(&lock(&variants.more), key).is_some(),
        }
    }

    /// A make of `variant` ahead of its dispatches, started by `start`: none where it is made or a
    /// thread is making it; else made on the calling thread, and its nanoseconds where it was
    /// made, or the make's error.
    pub(crate) fn make_ahead(
        &self,
        core: &Core,
        variant: DispatchVariant,
        start: VariantStart,
    ) -> Result<Option<u64>, Error> {
        let variants = &self.variants;
        let key = variant.key();
        if variants.in_cells(key).is_ok() {
            return Ok(None);
        }
        let more = lock(&variants.more);
        if variants.made(&more, key).is_some() || more.pending(key).is_some() {
            return Ok(None);
        }
        let began = host_time();
        self.make_pending(core, more, variant, start, None)?;
        Ok(Some(host_time().saturating_sub(began)))
    }

    /// Makes `variant`, which `more`, the lock held, shows neither made nor pending, started by
    /// `start`: entered as pending, with the first-use wait that began at `wait` where a dispatch
    /// makes it, made with the lock released, kept, and kept in the device's counters
    /// ([`Counters::variant_makes`]).
    fn make_pending(
        &self,
        core: &Core,
        mut more: MutexGuard<'_, More>,
        variant: DispatchVariant,
        start: VariantStart,
        wait: Option<u64>,
    ) -> Result<vk::Pipeline, Error> {
        more.pending.push(Making {
            variant,
            waits: u32::from(wait.is_some()),
            first_wait: wait.unwrap_or(0),
        });
        drop(more);
        let mut pending = Pending {
            variants: &self.variants,
            counters: &core.counters,
            variant,
            start,
            began: host_time(),
            made: None,
        };
        let (size, lengths, tiles) = variant.key();
        let made = Variant {
            size,
            threadgroup_lengths: lengths.to_vec(),
            tiles,
        };
        let pipeline = self.make(core, &made)?;
        pending.made = Some((made, pipeline));
        Ok(pipeline)
    }

    fn make(&self, core: &Core, variant: &Variant) -> Result<vk::Pipeline, Error> {
        let constants = &self.interface.constants;
        let entries: Vec<vk::SpecializationMapEntry> = constants
            .iter()
            .enumerate()
            .map(|(index, &(id, _))| vk::SpecializationMapEntry {
                constant_id: id,
                offset: (index * 4) as u32,
                size: 4,
            })
            .collect();
        let values: Vec<u8> = constants
            .iter()
            .map(|&(_, set)| match set {
                Constant::Size(axis) => variant.size[axis],
                Constant::Threads => variant.size.iter().product(),
                Constant::ThreadgroupLength(at) => variant.threadgroup_lengths[at],
                Constant::MultiRegion => u32::from(variant.tiles),
                Constant::Deep => u32::from(variant.size[2] > DEEP_THREADGROUP),
            })
            .flat_map(u32::to_ne_bytes)
            .collect();
        let specialization = vk::SpecializationInfo::default()
            .map_entries(&entries)
            .data(&values);
        // The translator pins a size only for a stage the device lists in
        // `requiredSubgroupSizeStages`, and the encoder dispatches such a kernel only in whole
        // subgroups of it, so its full-subgroups flag holds.
        let mut required = self.interface.required_subgroup_size.map(|size| {
            vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
                .required_subgroup_size(size)
        });
        // The pipeline's one create, of `module` through `cache` with `flags`.
        let mut create =
            |module: vk::ShaderModule, cache: vk::PipelineCache, flags: vk::PipelineCreateFlags| {
                let mut stage = vk::PipelineShaderStageCreateInfo::default()
                    .flags(self.interface.stage_flags)
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(&self.entry)
                    .specialization_info(&specialization);
                if let Some(required) = &mut required {
                    stage = stage.push_next(required);
                }
                let info = vk::ComputePipelineCreateInfo::default()
                    .flags(flags)
                    .stage(stage)
                    .layout(core.layout);
                // SAFETY: the module, the layout and the cache are the device's, and the
                // specialization gives each of the kernel's constants a 4-byte value.
                unsafe { core.device.create_compute_pipelines(cache, &[info], None) }
            };
        match &self.code {
            Code::Module(module) => made_through_cache(core, "vkCreateComputePipelines", |cache| {
                create(*module, cache, vk::PipelineCreateFlags::empty())
            }),
            Code::Heavy(heavy) => {
                let ids: Vec<u32> = constants.iter().map(|&(id, _)| id).collect();
                let stage = Create {
                    entry: &self.entry,
                    flags: self.interface.stage_flags,
                    subgroup: self.interface.required_subgroup_size,
                    ids: &ids,
                    data: &values,
                };
                // The cache is made of the pipeline's entry alone, and the module of the
                // kernel's words for this create alone.
                crate::stage::made(core, heavy, &stage, |cache, module| {
                    let flags = vk::PipelineCreateFlags::FAIL_ON_PIPELINE_COMPILE_REQUIRED;
                    create(module, cache, flags)
                })
            }
        }
    }

    /// Gives its `VkPipeline`s, and a kernel's at or under the stage bound its module, to the
    /// reaper.
    fn release(&mut self, core: &Core) {
        for pipeline in self.variants.drain() {
            core.reaper.give(Job::Free(Free::Pipeline(pipeline)));
        }
        if let Code::Module(module) = self.code {
            core.reaper.give(Job::Free(Free::ShaderModule(module)));
        }
    }
}

impl Drop for PipelineInner {
    fn drop(&mut self) {
        self.numbered.release(&self.compiled.core);
    }
}

impl Drop for Compiled {
    fn drop(&mut self) {
        // No state holds it, so no command buffer that used it is left: each kept its state until
        // it retired.
        self.core.counters.fall(Gauge::Kernels);
        self.whole.release(&self.core);
        let partial = match &mut self.form {
            Form::Sized {
                partial: Partial::Beside(lowered),
            } => Some(&mut **lowered),
            Form::Sized { .. } | Form::Tiled | Form::Square => None,
        };
        for lowered in [self.emulated.as_mut(), partial].into_iter().flatten() {
            lowered.release(&self.core);
        }
    }
}

impl crate::kept::Uncount for ComputePipelineState {
    unsafe fn uncounted(&self) -> std::mem::ManuallyDrop<ComputePipelineState> {
        // SAFETY: the caller keeps the state alive past the copy's last use.
        let inner = unsafe { crate::kept::Uncount::uncounted(&self.inner) };
        std::mem::ManuallyDrop::new(ComputePipelineState {
            inner: std::mem::ManuallyDrop::into_inner(inner),
        })
    }
}

impl crate::kept::Handle for ComputePipelineState {
    /// The state itself: it holds no handle of the device.
    type Kept = ComputePipelineState;

    fn object(&self) -> &ComputePipelineState {
        self
    }

    fn address_of(object: &ComputePipelineState) -> usize {
        Arc::as_ptr(&object.inner) as *const () as usize
    }
}
