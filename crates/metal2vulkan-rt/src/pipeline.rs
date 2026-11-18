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
//! once, plainly: its whole grids, threadgroup dispatches and indirect dispatches run that module
//! at their own threadgroup size, and a `dispatchThreads` grid that is not a whole number of
//! threadgroups runs its partial-threadgroups variant at its size, lowered at the first such grid.
//! Each `VkPipeline` is keyed by its module, threadgroup size and threadgroup memory lengths.
//!
//! What a state compiles is its content ([`KernelKey`]), never the state: the translator's key
//! of its descriptor and its `requiredThreadsPerThreadgroup`. Every state of one content shares
//! one [`Compiled`]: its translation, its shader modules and their `VkPipeline`s, its emulated
//! variant and its notes. The device's map of kernels holds each content's (`kernel_map`), so a
//! state of a content the map holds costs a lookup; past its cap the map forgets the least
//! recently used, which a state that took it keeps. What stays a state's own is what the consumer
//! gives it: the indices of its numbered samplers and the constexpr samplers' elements it holds
//! at them.
//!
//! A kernel whose tier runs each SIMD group as one subgroup, or as a 32-lane cluster of one, runs
//! only threadgroups of whole subgroups. A threadgroup that is not whole subgroups (32 threads
//! where each SIMD group is half of a 64-lane subgroup included) runs the kernel's
//! emulated-SIMD-group variant, whose SIMD groups exchange through threadgroup memory, at its own
//! size, whose last SIMD group may be partly filled, as Metal runs it; that variant is lowered at
//! the first such dispatch.
//!
//! Nothing the runtime records depends on when a compile finishes: an encoding call whose
//! `VkPipeline` is not made yet makes it and waits for it, then records, so there is one recorded
//! form; one whose variant ([`Later`]: the emulated or the partial-threadgroups one) is not lowered
//! yet gives the lowering to the reaper and waits for it. Each such wait is a first-use wait,
//! counted with its time ([`Metrics::first_use_waits`](crate::Metrics::first_use_waits)). No
//! translation runs on an encoding thread, and no variant is lowered before a dispatch needs it.

use crate::arrays::BINDINGS;
use crate::cache;
use crate::counters::{Gauge, Metric};
use crate::device::{lock, Core};
use crate::error::{Error, Gap, Refusal};
use crate::hazard::Access;
use crate::kernel_map::KernelKey;
use crate::reaper::{Free, Job};
use crate::ring::CHUNK;
use crate::sampler::{canonical, hold_constexpr, release_constexpr};
use crate::times::host_time;
use ash::vk;
use metal2vulkan::error::{Error as TranslationError, Feature};
use metal2vulkan::library::CompiledFunction;
use metal2vulkan::pipeline::{
    arena_bytes, AccessSet, Axis, BindingSource, BufferKind, CombinedSampler, ComputePipeline,
    ComputePipelineDescriptor, DescriptorSetLayout, FunctionDescriptor, LoweringInputs,
    NumberedSampler, PrecisionNote, PushField, PushLoc, RuntimeFact, SamplerRef, SlotClass,
    SlotInterface, SlotRef, SpecKey, Stage, StageEntries, StageSource, TextureAccess,
    DEEP_THREADGROUP, GRID_TILE_THREADS,
};
use metal2vulkan::Budget;
use std::ffi::CString;
use std::sync::{Arc, Mutex, OnceLock};

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
/// each numbers its own samplers.
#[derive(Clone)]
pub struct ComputePipelineState {
    pub(crate) inner: Arc<PipelineInner>,
}

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
    /// The module of whole grids, threadgroup dispatches and indirect dispatches; for a kernel
    /// that does not observe its threadgroup partition, of every dispatch.
    pub(crate) whole: Kernel,
    /// How a direct dispatch runs.
    pub(crate) form: Form,
    /// The emulated-SIMD-group variant of a kernel whose `whole` module runs only whole SIMD
    /// groups, lowered at the first dispatch that needs it.
    pub(crate) emulated: Option<Later>,
    /// The translator's notes of where the kernel's answers may differ from Metal's, of the
    /// module made with the content.
    notes: Vec<PrecisionNote>,
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

/// Whether a state of the module `whole` holds every constexpr sampler use of `later`, a module
/// of the same entry lowered later: the samplers it numbers are its, and each use it samples
/// through is one of its, as a state holds the uses of its module ([`number`]).
fn covers(whole: &Interface, later: &Interface) -> bool {
    later.numbered == whole.numbered
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
    number_samplers(core, samplers, &whole.constexprs, sampler_ids, gap)
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
            .map_err(Error::Translation)?
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

/// A variant of a kernel the reaper lowers at the first dispatch that needs it, which waits for
/// it: its emulated-SIMD-group variant (`LoweringInputs::emulated_simdgroups`), or an observing
/// kernel's partial-threadgroups variant (`LoweringInputs::partial_threadgroups`). What it is
/// lowered from, and once lowered, the module or why there is none.
pub(crate) struct Later {
    descriptor: ComputePipelineDescriptor,
    function: FunctionDescriptor,
    inputs: LoweringInputs,
    pub(crate) lowered: OnceLock<Lowered>,
}

/// Which of a content's [`Later`] variants.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Which {
    /// [`Compiled::emulated`].
    Emulated,
    /// [`Form::Sized`]'s partial-threadgroups variant.
    Partial,
}

/// What the reaper's lowering of a [`Later`] variant gave.
pub(crate) enum Lowered {
    /// The module, and the translator's notes of it.
    Made(Box<Kernel>, Vec<PrecisionNote>),
    /// The translator refused the variant, or it reads what the compute encoder does not supply.
    Refused,
    /// `vkCreateShaderModule` failed.
    Failed(vk::Result),
    /// The device was quiesced before the reaper lowered it.
    Quiesced,
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
    /// The kernel observes its partition: a dispatch runs at its own threadgroup size, and a
    /// `dispatchThreads` grid that is not a whole number of threadgroups on the
    /// partial-threadgroups variant, lowered at the first such grid where the content has a
    /// function to lower it from.
    Sized { partial: Option<Box<Later>> },
}

/// One translated module of a state, its interface, and the `VkPipeline`s made of it so far.
pub(crate) struct Kernel {
    module: vk::ShaderModule,
    /// The module's entry point.
    entry: CString,
    pub(crate) interface: Interface,
    /// The `VkPipeline`s made so far, by their specialization.
    variants: Variants,
}

/// The variants of a kernel that each have a cell of their own.
const VARIANT_CELLS: usize = 8;

/// A kernel's `VkPipeline`s made so far, by their specialization. The first [`VARIANT_CELLS`]
/// made each fill a cell of their own, which a dispatch reads with no lock; any made past them go
/// in a list under the lock that every making takes, so that one variant is made once.
#[derive(Default)]
struct Variants {
    cells: [OnceLock<(Variant, vk::Pipeline)>; VARIANT_CELLS],
    more: Mutex<Vec<(Variant, vk::Pipeline)>>,
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

    /// Keeps `pipeline`, of `variant`, made under the lock `more` is: in the first empty cell,
    /// else in `more`.
    fn keep(
        &self,
        more: &mut Vec<(Variant, vk::Pipeline)>,
        variant: Variant,
        pipeline: vk::Pipeline,
    ) {
        match self.cells.iter().find(|cell| cell.get().is_none()) {
            // Only a making fills a cell, under the lock: the empty one stays empty until this.
            Some(cell) => {
                let _ = cell.set((variant, pipeline));
            }
            None => more.push((variant, pipeline)),
        }
    }

    /// Every pipeline, taken out to give back.
    fn drain(&mut self) -> impl Iterator<Item = vk::Pipeline> + '_ {
        let more = self
            .more
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        (self.cells.iter_mut())
            .filter_map(|cell| cell.take().map(|(_, pipeline)| pipeline))
            .chain(more.drain(..).map(|(_, pipeline)| pipeline))
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
    /// The kernel's specialization constants: each one's id and what sets it.
    pub(crate) constants: Vec<(u32, Constant)>,
    /// The index of each threadgroup memory argument the kernel takes, in the order of
    /// [`Variant::threadgroup_lengths`].
    pub(crate) threadgroups: Vec<u32>,
    /// `maxTotalThreadsPerThreadgroup`.
    pub(crate) max_total_threads: u32,
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

/// What a parameter holds.
#[derive(Clone, Copy)]
pub(crate) enum FieldValue {
    /// A `u64`: the address of the buffer bound at the slot plus its offset, the address of the
    /// slot's `set_bytes` copy, or 0 where nothing is bound. `mask` is the alignment the module
    /// takes that address to have, less 1 (`RuntimeFact::AddressAlignment`), 0 where it takes
    /// none: a dispatch whose bound buffer's address it does not divide is refused
    /// (`Refusal::BufferOffsetAlignment`).
    BufferAddress { slot: u32, mask: u64 },
    /// A `u64`: the `gpuResourceID` of the texture bound at the slot, 0 where none is.
    TextureId(u32),
    /// A `u64`: the `gpuResourceID` of the sampler state bound at the slot, 0 where none is.
    SamplerId(u32),
    /// A `u32`: the channel widths of the texture bound at the slot, as an integer write
    /// saturates to them (`metal::format::integer_channels_word`), 0 where none is.
    IntegerChannels(u32),
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
    /// thread, which waits for it, within the translator's 20-second budget per lowering, once:
    /// a kernel that does not observe its threadgroup partition as its partial-threadgroups
    /// variant, whose 256-wide tile's `VkPipeline`, or an image-only kernel's 16x16 workgroups'
    /// one, is made here too, and one that does as its plain module. The translator answers which
    /// without lowering, where the cache directory holds no lowering of the variant to answer it.
    /// Where the device has a cache directory ([`DeviceOptions::pipeline_cache_dir`]), a lowering a
    /// device made before is read from it in place of lowering again, and every `VkPipeline` is
    /// made from its file there where it has one. The content's other `VkPipeline`s are made
    /// later, each by the first encoding call of any of its states that needs it
    /// ([`ComputePipelineState`]), and so are its variants: an observing kernel's
    /// partial-threadgroups variant, and the emulated-SIMD-group variant of a kernel whose tier
    /// runs only whole SIMD groups, each lowered by the reaper, through the cache directory too,
    /// for the first dispatch that needs it, which waits for it.
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
    /// Err: [`Error::Translation`], the translator's typed refusal or a descriptor Metal refuses;
    /// [`Gap::IndirectCommandBuffers`]; [`Gap::KernelInput`] for a kernel that reads what the
    /// compute encoder does not supply yet; [`Refusal::SamplerIds`] where `sampler_ids` answers
    /// another count; [`Refusal::IndicesNotGranted`] for a constexpr sampler's index;
    /// [`Limit::DeviceSamplers`](crate::Limit::DeviceSamplers); [`Error::Vulkan`].
    pub fn new_compute_pipeline_state(
        &self,
        descriptor: &ComputePipelineDescriptor,
        sampler_ids: impl FnOnce(&[NumberedSampler]) -> Result<Vec<u32>, Error>,
    ) -> Result<ComputePipelineState, Error> {
        if descriptor.support_indirect_command_buffers {
            return Err(Error::Unsupported(Gap::IndirectCommandBuffers));
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
    /// The translation of `descriptor`, with the `VkPipeline` of a tiling kernel's tiles.
    fn of(core: &Arc<Core>, descriptor: &ComputePipelineDescriptor) -> Result<Compiled, Error> {
        let core = Arc::clone(core);
        let budget = Budget::standard();
        let translator = &core.translator;
        let compiled = |function: CompiledFunction| {
            let mut compiled = descriptor.clone();
            compiled.kernel = StageSource::Compiled(function);
            translator.compile_compute_pipeline(&compiled, &budget)
        };
        let (whole, partial, image_only, source) = match &descriptor.kernel {
            StageSource::Function(function) => {
                let entries = &core.entries;
                let plain = translator
                    .kernel_inputs_read(function, &descriptor.lowering_inputs(), &budget, entries)
                    .map_err(Error::Translation)?;
                let mut emulated = plain.clone();
                emulated.emulated_simdgroups = true;
                let source = (function.clone(), emulated);
                let mut inputs = plain.clone();
                inputs.partial_threadgroups = true;
                let lower =
                    |inputs: &LoweringInputs| lower(&core, function, inputs, &budget, entries);
                // The partial-threadgroups variant, where the kernel does not observe its
                // partition: what the cache directory holds, else lowered, unless the translator
                // answers that the kernel observes it. The translator's refusal of the variant
                // (emulated simdgroups, a vote word its workgroup memory cannot hold) is kept like
                // a lowering, and so is its answer that the kernel observes it, which a device
                // on the directory then reads in place of asking.
                let variant_key = *translator.function_compile_key(function, &inputs).digest();
                let variant = if core.caches.refused(&variant_key) {
                    core.counters.count(Metric::TranslationsLoaded);
                    None
                } else if let Some(kept) = kept(&core, function, &inputs) {
                    Some(kept)
                } else if core.caches.observed(&variant_key) {
                    None
                } else {
                    let observes = (translator)
                        .kernel_observes_partition(function, &inputs, &budget, entries)
                        .map_err(Error::Translation);
                    core.note_entries();
                    if observes? {
                        cache::keep_observer(&core, &variant_key);
                        None
                    } else {
                        match lower(&inputs) {
                            Ok(variant) => Some(variant),
                            Err(TranslationError::Unsupported(refused))
                                if refused.feature == Feature::GpuEncodedIndirectCommands =>
                            {
                                cache::keep_refusal(&core, &variant_key);
                                None
                            }
                            Err(error) => return Err(Error::Translation(error)),
                        }
                    }
                };
                match variant {
                    // A kernel that does not observe its partition runs its variant alone: in
                    // tiles where the variant states them, else at its own size.
                    Some(variant) if !variant.observes_threadgroup_partition() => {
                        let image_only = variant.is_image_only();
                        let pipeline = compiled(variant).map_err(Error::Translation)?;
                        let partial = (!grid_tiles(&pipeline)).then_some(None);
                        (pipeline, partial, image_only, Some(source))
                    }
                    // The kernel observes its partition, or its variant is refused: its plain
                    // module, the variant lowered at the first grid that needs it.
                    _ => {
                        let plain = lower(&plain).and_then(compiled);
                        let plain = plain.map_err(Error::Translation)?;
                        let partial = Box::new(Later {
                            descriptor: descriptor.clone(),
                            function: function.clone(),
                            inputs,
                            lowered: OnceLock::new(),
                        });
                        (plain, Some(Some(partial)), false, Some(source))
                    }
                }
            }
            // A kernel the consumer compiled: the variant it was compiled under.
            StageSource::Compiled(function) => {
                let pipeline = translator
                    .compile_compute_pipeline(descriptor, &budget)
                    .map_err(Error::Translation)?;
                // Only a non-observer's partial variant runs its grid in tiles. It has no
                // emulated variant: the kernel keeps the variant it was compiled under. The
                // translator links a compiled kernel only for its own profile
                // (`ForeignCompiledFunction` otherwise), on which a non-observer's partial
                // variant states `GridTiles`: a module without it observes its partition or is
                // the plain module, and runs at its own size.
                let partial = (!grid_tiles(&pipeline)).then_some(None);
                (pipeline, partial, function.is_image_only(), None)
            }
        };
        let required = descriptor.required_threads_per_threadgroup;
        let mut notes = whole.stage.precision_notes.clone();
        notes.sort_unstable();
        notes.dedup();
        let whole = Kernel::new(&core, whole, required)?;
        // A module that runs only whole SIMD groups has its emulated variant for the others.
        let emulated = source
            .filter(|_| !whole.interface.multiples.is_empty())
            .map(|(function, inputs)| Later {
                descriptor: descriptor.clone(),
                function,
                inputs,
                lowered: OnceLock::new(),
            });
        // `partial` is `Some` for a kernel that runs at its own size: with its variant to lower
        // where it has one.
        let form = match (partial, image_only) {
            (Some(partial), _) => Form::Sized { partial },
            (None, false) => Form::Tiled,
            (None, true) => Form::Square,
        };
        let compiled = Compiled::new(core, whole, form, emulated, notes);
        // A tiling kernel's grids one deep run on one `VkPipeline`, made with the state: its
        // tiles', or an image-only kernel's square workgroups'.
        let first = match compiled.form {
            Form::Tiled => Some(([GRID_TILE_THREADS, 1, 1], true)),
            Form::Square => Some((SQUARE, false)),
            Form::Sized { .. } => None,
        };
        if let Some((size, tiles)) = first {
            let variant = Variant {
                size,
                threadgroup_lengths: vec![0; compiled.whole.interface.threadgroups.len()],
                tiles,
            };
            let made = compiled.whole.make(&compiled.core, &variant)?;
            let variants = &compiled.whole.variants;
            variants.keep(&mut lock(&variants.more), variant, made);
        }
        Ok(compiled)
    }

    /// The content of `whole`, its form, emulated variant and notes, counted live.
    fn new(
        core: Arc<Core>,
        whole: Kernel,
        form: Form,
        emulated: Option<Later>,
        notes: Vec<PrecisionNote>,
    ) -> Compiled {
        core.counters.rise(Gauge::Kernels);
        Compiled {
            core,
            whole,
            form,
            emulated,
            notes,
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
    let translator = &core.translator;
    let key = translator.function_compile_key(function, inputs);
    let kept = (core.caches).translation(key.digest(), |bytes| {
        CompiledFunction::from_bytes_for_key(bytes, &key, translator).ok()
    });
    if kept.is_some() {
        core.counters.count(Metric::TranslationsLoaded);
    }
    kept
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
    let lowered = translator.compile_function_from_entries(function, inputs, budget, entries);
    core.note_entries();
    let lowered = lowered?;
    cache::keep_translation(core, key.digest(), lowered.to_bytes());
    Ok(lowered)
}

/// The one `VkPipeline` that `create` makes through the cache it is given: the device's one
/// `VkPipelineCache` (`crate::cache`), which a creation before the reaper made it empty waits for.
/// `call` names `create`'s Vulkan call. Counted as made.
pub(crate) fn made_through_cache(
    core: &Core,
    call: &'static str,
    create: impl FnOnce(vk::PipelineCache) -> Result<Vec<vk::Pipeline>, (Vec<vk::Pipeline>, vk::Result)>,
) -> Result<vk::Pipeline, Error> {
    let creation = cache::creation(core);
    let made = create(creation.handle());
    drop(creation);
    let pipeline = made.map_err(|(_, result)| Error::Vulkan { call, result })?[0];
    core.counters.count(Metric::PipelinesCreated);
    core.counters.rise(Gauge::Pipelines);
    Ok(pipeline)
}

/// The runtime's view of the translator's kernel, or [`Gap::KernelInput`] where the kernel reads
/// something the compute encoder does not supply.
fn interface(
    pipeline: &ComputePipeline,
    required_size: Option<[u32; 3]>,
) -> Result<Interface, Error> {
    let translated = &pipeline.interface;
    let stage = &pipeline.stage;
    // Every argument is a pushed address, a pushed resource ID or index, or a specialization: no
    // descriptor but set 1's, no table the binder builds, no constant data; the stage's flags are
    // its subgroups'.
    let stage_flags = vk::PipelineShaderStageCreateFlags::from_raw(stage.stage_flags.as_raw());
    let subgroups = vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS
        | vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE;
    if translated.needs_pointer_translation
        || translated.needs_handle_translation
        || translated.reaches_command_buffers
        || !translated.stage_in.is_empty()
        || !pipeline.layout.sets.iter().all(of_the_device)
        || stage.constant_blob.is_some()
        || !subgroups.contains(stage_flags)
    {
        return Err(gap());
    }
    // The bind-time facts the encoder supplies.
    let mut multiples = Vec::new();
    let mut aligned = Vec::new();
    let mut vote = 0;
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
            // What a variant could relieve, which the module runs without: a predicated walk
            // that unit axes would make less divergent, and lockstep barriers a one-row region
            // could skip. Every dispatch runs on this module, keeping its whole groups.
            RuntimeFact::UnitAxes { axes: 0, .. } | RuntimeFact::RowLockstep => {}
            // What the translator checks against the descriptor when it compiles, and nothing on
            // the device reads.
            RuntimeFact::CallStackDepth { .. } | RuntimeFact::OpenClMode => {}
            // `RuntimeFact` is non-exhaustive: a fact this match does not name is one the encoder
            // does not supply.
            _ => return Err(gap()),
        }
    }
    let mut buffers = Vec::new();
    let mut textures = Vec::new();
    for (slot, used) in &translated.slots {
        match (slot.stage, slot.class, used) {
            (Stage::Kernel, SlotClass::Buffer, SlotInterface::Buffer(buffer))
                if buffer.kind == BufferKind::Buffer =>
            {
                buffers.push((slot.index, access(buffer.access)));
            }
            (Stage::Kernel, SlotClass::Texture, SlotInterface::Texture(texture)) => {
                textures.push((slot.index, texture_access(texture.access)));
            }
            (Stage::Kernel, SlotClass::Sampler, SlotInterface::Sampler(_)) => {}
            _ => return Err(gap()),
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
            return Err(gap());
        }
        push_len = placement
            .push_range
            .map_or(0, |(offset, size)| offset + size);
        block_len = placement.argument_table_size;
        for placed in &placement.fields {
            let place = match placed.loc {
                PushLoc::Push { offset } => Place::Push(offset),
                PushLoc::ArgumentTable { offset } => Place::Block(offset),
                _ => return Err(gap()),
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
                (PushField::ConstexprSampler(sampler), _) => {
                    let SamplerRef::Constexpr(words) = sampler.sampler else {
                        return Err(gap());
                    };
                    let at = numbered
                        .iter()
                        .position(|&numbered| numbered == NumberedSampler::Constexpr(words))
                        .ok_or_else(gap)?;
                    if sampler.unnormalized {
                        return Err(gap());
                    }
                    constexprs.push((at, sampler));
                    FieldValue::ConstexprSampler(at)
                }
                (PushField::DispatchRegion, _) => FieldValue::DispatchRegion,
                _ => return Err(gap()),
            };
            fields.push(Field { place, value });
        }
    }
    // The block, where there is one, is in one ring chunk, and its address in the window.
    let block_address = match (block_len, block_address) {
        (0, None) => 0,
        (1.., Some(at)) if u64::from(block_len) <= CHUNK => at,
        _ => return Err(gap()),
    };
    if push_len > PUSH_BYTES {
        return Err(gap());
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
    let mut constants = Vec::new();
    for constant in &stage.specialization.constants {
        if constant.size != 4 {
            return Err(gap());
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
                    .ok_or_else(gap)?,
            ),
            SpecKey::MultiRegion => Constant::MultiRegion,
            SpecKey::DeepThreadgroup => Constant::Deep,
            _ => return Err(gap()),
        };
        constants.push((constant.id, set));
    }
    // A multiple outside a flattened local size is a derivative group's, which a kernel over
    // buffers alone does not take.
    let flat = constants
        .iter()
        .any(|(_, set)| matches!(set, Constant::Threads));
    if !multiples.is_empty() && !flat {
        return Err(gap());
    }
    // Every workgroup is within the floor's sizes: a flattened one is one row of the threadgroup,
    // and any other runs a threadgroup deeper than `DEEP_THREADGROUP` as one row, which a module
    // that cannot (compute derivatives) does not declare.
    let deep = constants
        .iter()
        .any(|(_, set)| matches!(set, Constant::Deep));
    if !flat && !deep {
        return Err(gap());
    }
    let max_total_threads = match required_size {
        Some(size) => size.iter().product(),
        None => pipeline.metal_info.max_total_threads_per_threadgroup,
    };
    Ok(Interface {
        buffers,
        textures,
        numbered,
        constexprs,
        reads_unbound: translated.reads_unbound_resources,
        writes_unbound: translated.writes_unbound_resources,
        has_effects: translated.writes_memory,
        push_len,
        block_len,
        block_address,
        fields,
        constants,
        threadgroups,
        max_total_threads,
        memory: Memory { own, vote },
        multiples,
        required_subgroup_size: stage.required_subgroup_size,
        stage_flags,
    })
}

/// The refusal of a kernel that reads what the compute encoder does not supply.
fn gap() -> Error {
    Error::Unsupported(Gap::KernelInput)
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
    /// `maxTotalThreadsPerThreadgroup`: the translator's answer for the device, which is Metal's
    /// limit or less where the device holds fewer threads in a workgroup; for a descriptor with
    /// `requiredThreadsPerThreadgroup`, that size's threads, as Metal answers. A dispatch of
    /// another size within it runs at the size it gives, where Metal's validation stops it.
    pub fn max_total_threads_per_threadgroup(&self) -> u32 {
        self.inner.compiled.whole.interface.max_total_threads
    }

    /// No Metal counterpart: where the kernel's answers may differ from Metal's on this device,
    /// fixed per operation by the translator, for every module the state runs: the one made with
    /// the state, and once a dispatch needed them, its variants: the emulated-SIMD-group module
    /// that runs threadgroups of partial SIMD groups and an observing kernel's
    /// partial-threadgroups module. Metal's answers are its own GPU's.
    pub fn precision_notes(&self) -> Vec<PrecisionNote> {
        let compiled = &self.inner.compiled;
        let mut notes = compiled.notes.clone();
        for which in [Which::Emulated, Which::Partial] {
            if let Some(Lowered::Made(_, more)) =
                (compiled.later(which)).and_then(|later| later.lowered.get())
            {
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
        let module = unsafe { Kernel::module(&core, words) }?;
        let whole = Kernel {
            module,
            entry: c"main".to_owned(),
            interface,
            variants: Variants::default(),
        };
        let compiled = Compiled::new(core, whole, Form::Sized { partial: None }, None, Vec::new());
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
    fn later(&self, which: Which) -> Option<&Later> {
        match (which, &self.form) {
            (Which::Emulated, _) => self.emulated.as_ref(),
            (Which::Partial, Form::Sized { partial }) => partial.as_deref(),
            (Which::Partial, Form::Tiled | Form::Square) => None,
        }
    }

    /// On the reaper: lowers the content's `which` variant, where it has one not lowered yet.
    pub(crate) fn lower_later(&self, which: Which) {
        if let Some(later) = self.later(which) {
            (later.lowered).get_or_init(|| later.lower(&self.core, &self.whole.interface));
        }
    }

    /// Where the reaper ended before lowering it: the content's `which` variant is not lowered.
    pub(crate) fn quiesce_later(&self, which: Which) {
        if let Some(later) = self.later(which) {
            later.lowered.get_or_init(|| Lowered::Quiesced);
        }
    }

    /// The module of the content's `which` variant: where it is not lowered yet, the calling
    /// thread gives the lowering to the reaper and waits for it, a first-use wait.
    ///
    /// Err: `refused` where the content has no such variant or the translator refused it;
    /// [`Error::Quiesced`]; [`Error::Vulkan`] where its shader module failed.
    pub(crate) fn later_kernel(
        self: &Arc<Self>,
        which: Which,
        refused: impl FnOnce() -> Error,
    ) -> Result<&Kernel, Error> {
        let Some(later) = self.later(which) else {
            return Err(refused());
        };
        let lowered = match later.lowered.get() {
            Some(lowered) => lowered,
            None => {
                let began = host_time();
                self.core.reaper.lower(Arc::downgrade(self), which);
                let lowered = later.lowered.wait();
                self.core
                    .counters
                    .first_use_wait(host_time().saturating_sub(began));
                lowered
            }
        };
        match lowered {
            Lowered::Made(kernel, _) => Ok(&**kernel),
            Lowered::Refused => Err(refused()),
            &Lowered::Failed(result) => Err(Error::Vulkan {
                call: "vkCreateShaderModule",
                result,
            }),
            Lowered::Quiesced => Err(Error::Quiesced),
        }
    }
}

impl Later {
    /// The variant lowered, on the reaper: the lowering a device kept in the cache directory, or
    /// one made now and kept; the translator's refusal is kept too. `whole` is the content's
    /// module, which a state numbers its samplers from.
    fn lower(&self, core: &Core, whole: &Interface) -> Lowered {
        let translator = &core.translator;
        let budget = Budget::standard();
        let key = *translator
            .function_compile_key(&self.function, &self.inputs)
            .digest();
        if core.caches.refused(&key) {
            core.counters.count(Metric::TranslationsLoaded);
            return Lowered::Refused;
        }
        let compiled = match lower(core, &self.function, &self.inputs, &budget, &core.entries) {
            Ok(compiled) => compiled,
            Err(TranslationError::Unsupported(_)) => {
                cache::keep_refusal(core, &key);
                return Lowered::Refused;
            }
            Err(_) => return Lowered::Refused,
        };
        let mut descriptor = self.descriptor.clone();
        descriptor.kernel = StageSource::Compiled(compiled);
        let Ok(pipeline) = translator.compile_compute_pipeline(&descriptor, &budget) else {
            return Lowered::Refused;
        };
        let notes = pipeline.stage.precision_notes.clone();
        match Kernel::new(core, pipeline, descriptor.required_threads_per_threadgroup) {
            // A module that samples a constexpr sampler in a use the state does not hold reads
            // what the encoder does not supply.
            Ok(mut kernel) if !covers(whole, &kernel.interface) => {
                kernel.release(core);
                Lowered::Refused
            }
            Ok(kernel) => Lowered::Made(Box::new(kernel), notes),
            Err(Error::Vulkan { result, .. }) => Lowered::Failed(result),
            Err(_) => Lowered::Refused,
        }
    }
}

impl Kernel {
    /// The translator's `pipeline` as a module of the device, with its interface.
    fn new(
        core: &Core,
        pipeline: ComputePipeline,
        required_size: Option<[u32; 3]>,
    ) -> Result<Kernel, Error> {
        let interface = interface(&pipeline, required_size)?;
        let stage = pipeline.stage;
        // The translator's entry points hold no NUL; one that did would name no entry point.
        let entry = CString::new(stage.entry_point).map_err(|_| gap())?;
        // SAFETY: the words are the translator's SPIR-V, for the features device creation enabled.
        let module = unsafe { Kernel::module(core, &stage.spirv) }?;
        Ok(Kernel {
            module,
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
    unsafe fn module(core: &Core, words: &[u32]) -> Result<vk::ShaderModule, Error> {
        let info = vk::ShaderModuleCreateInfo::default().code(words);
        // SAFETY: the caller's contract.
        unsafe { core.device.create_shader_module(&info, None) }
            .map_err(Error::vulkan("vkCreateShaderModule"))
    }

    /// The `VkPipeline` of the variant `key`, made now where it is not yet: the calling thread
    /// waits for it, a first-use wait, so what is recorded never depends on when a compile
    /// finishes. One of the kernel's first [`VARIANT_CELLS`] variants is found with no lock.
    pub(crate) fn pipeline(&self, core: &Core, key: Key<'_>) -> Result<vk::Pipeline, Error> {
        let variants = &self.variants;
        let full = match variants.in_cells(key) {
            Ok(pipeline) => return Ok(pipeline),
            Err(full) => full,
        };
        let mut more = lock(&variants.more);
        let (size, lengths, tiles) = key;
        let found = match full {
            true => (more.iter()).find(|(variant, _)| variant.is(size, lengths, tiles)),
            false => None,
        };
        if let Some(&(_, pipeline)) = found {
            return Ok(pipeline);
        }
        // Another thread may have made it since the cells were read.
        if let Ok(pipeline) = variants.in_cells(key) {
            return Ok(pipeline);
        }
        let variant = Variant {
            size,
            threadgroup_lengths: lengths.to_vec(),
            tiles,
        };
        let began = host_time();
        let pipeline = self.make(core, &variant)?;
        core.counters
            .first_use_wait(host_time().saturating_sub(began));
        variants.keep(&mut more, variant, pipeline);
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
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .flags(self.interface.stage_flags)
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(self.module)
            .name(&self.entry)
            .specialization_info(&specialization);
        if let Some(required) = &mut required {
            stage = stage.push_next(required);
        }
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(core.layout);
        made_through_cache(core, "vkCreateComputePipelines", |cache| {
            // SAFETY: the module, the layout and the cache are the device's, and the
            // specialization gives each of the kernel's constants a 4-byte value.
            unsafe { core.device.create_compute_pipelines(cache, &[info], None) }
        })
    }

    /// Gives the module and its `VkPipeline`s to the reaper.
    fn release(&mut self, core: &Core) {
        for pipeline in self.variants.drain() {
            core.reaper.give(Job::Free(Free::Pipeline(pipeline)));
        }
        core.reaper.give(Job::Free(Free::ShaderModule(self.module)));
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
            Form::Sized { partial } => partial.as_deref_mut(),
            Form::Tiled | Form::Square => None,
        };
        for later in [self.emulated.as_mut(), partial].into_iter().flatten() {
            if let Some(Lowered::Made(kernel, _)) = later.lowered.get_mut() {
                kernel.release(&self.core);
            }
        }
    }
}

impl crate::kept::Handle for ComputePipelineState {
    fn address(&self) -> usize {
        Arc::as_ptr(&self.inner) as *const () as usize
    }

    unsafe fn uncounted(&self) -> std::mem::ManuallyDrop<ComputePipelineState> {
        // SAFETY: the pointer is a live `Arc`'s; the handle made of it is never dropped, and the
        // caller keeps the object alive past its last use.
        let inner = unsafe { Arc::from_raw(Arc::as_ptr(&self.inner)) };
        std::mem::ManuallyDrop::new(ComputePipelineState { inner })
    }
}
