//! Compute pipeline states: the translator's kernel and its interface, and its `VkPipeline`s.
//!
//! The translator lowers every kernel first as its partial-threadgroups variant, through one
//! decode and specialization of its entry. A kernel that does not observe its threadgroup
//! partition keeps that module alone: every direct dispatch runs in tiles of 256 threads, one
//! `VkPipeline` per tile shape ([`tile`]), the 256-wide tile made with the state; an indirect
//! dispatch runs at its own threadgroup size. A kernel that observes its partition is lowered
//! once more, plainly: its whole grids, threadgroup dispatches and indirect dispatches run that
//! module at their own threadgroup size, and a `dispatchThreads` grid that is not a whole number
//! of threadgroups runs the partial-threadgroups variant at its size, where the translator made
//! one. Each `VkPipeline` is keyed by its module, threadgroup size and threadgroup memory lengths.
//!
//! Nothing the runtime records depends on when a compile finishes: an encoding call whose
//! `VkPipeline` is not made yet makes it and waits for it, then records, so there is one recorded
//! form. Each such wait is a first-use wait, counted with its time
//! ([`Metrics::first_use_waits`](crate::Metrics::first_use_waits)). No translation runs on an
//! encoding thread.

use crate::counters::{Gauge, Metric};
use crate::device::{lock, Core};
use crate::error::{Error, Gap};
use crate::hazard::Access;
use crate::reaper::{Free, Job};
use crate::ring::CHUNK;
use crate::times::host_time;
use ash::vk;
use metal2vulkan::error::{Error as TranslationError, Feature};
use metal2vulkan::library::CompiledFunction;
use metal2vulkan::pipeline::{
    AccessSet, Axis, BufferKind, ComputePipeline, ComputePipelineDescriptor, FunctionDescriptor,
    LoweringInputs, PrecisionNote, PushField, PushLoc, RuntimeFact, SlotClass, SlotInterface,
    SlotRef, SpecKey, Stage, StageEntries, StageSource,
};
use metal2vulkan::Budget;
use std::ffi::CString;
use std::sync::{Arc, Mutex};

/// The push window: the parameters a kernel reads from push constants, `[0, 128)` on every device,
/// as 128 bytes is what Vulkan guarantees. Fields past it go to a block in the parameter ring,
/// whose address is a field in the window.
pub(crate) const PUSH_BYTES: u32 = 128;

/// The threads of a tile: a kernel that does not observe its threadgroup partition runs every
/// direct dispatch in tiles of this many threads.
pub(crate) const TILE_THREADS: u32 = 256;

/// The tile a direct dispatch of a kernel that does not observe its threadgroup partition runs
/// over a grid of `threads`: 256 by 1 for a grid 256 threads wide or wider, and for a narrower
/// one the 256-thread tile whose width is the grid's width rounded down to a power of two, so at
/// most nine shapes per kernel.
pub(crate) fn tile(threads: [u32; 3]) -> [u32; 3] {
    let width = match threads[0] {
        0 => TILE_THREADS,
        wide if wide >= TILE_THREADS => TILE_THREADS,
        narrow => 1 << narrow.ilog2(),
    };
    [width, TILE_THREADS / width, 1]
}

/// `MTLComputePipelineState`.
///
/// A cheap handle: clones share one state. A command buffer that dispatches with it keeps it until
/// it retires.
#[derive(Clone)]
pub struct ComputePipelineState {
    pub(crate) inner: Arc<PipelineInner>,
}

pub(crate) struct PipelineInner {
    core: Arc<Core>,
    /// The module of whole grids, threadgroup dispatches and indirect dispatches; for a kernel
    /// that does not observe its threadgroup partition, of every dispatch.
    pub(crate) whole: Kernel,
    /// How a direct dispatch runs.
    pub(crate) form: Form,
    /// The translator's notes of where the kernel's answers may differ from Metal's, of every
    /// module the state runs.
    notes: Vec<PrecisionNote>,
}

/// How a state runs a direct dispatch.
pub(crate) enum Form {
    /// The kernel does not observe its threadgroup partition: `whole`, its partial-threadgroups
    /// variant, runs every direct dispatch in [`tile`]s, whatever the threadgroup size.
    Tiled,
    /// The kernel observes its partition: a dispatch runs at its own threadgroup size, and a
    /// `dispatchThreads` grid that is not a whole number of threadgroups on the
    /// partial-threadgroups variant, where the translator made one.
    Sized { partial: Option<Box<Kernel>> },
}

/// One translated module of a state, its interface, and the `VkPipeline`s made of it so far.
pub(crate) struct Kernel {
    module: vk::ShaderModule,
    /// The module's entry point.
    entry: CString,
    pub(crate) interface: Interface,
    /// The `VkPipeline`s made so far, by their specialization.
    variants: Mutex<Vec<(Variant, vk::Pipeline)>>,
}

/// What the runtime needs of a kernel, from the translator.
pub(crate) struct Interface {
    /// Each buffer slot the kernel uses, with its access.
    pub(crate) buffers: Vec<(u32, Access)>,
    /// Whether the kernel reads through an address that is not one of its bound slots.
    pub(crate) reads_unbound: bool,
    /// Whether the kernel writes through an address that is not one of its bound slots.
    pub(crate) writes_unbound: bool,
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
    /// The threadgroup memory every threadgroup takes beside the lengths set: Metal's static part
    /// and a predicated walk's vote word.
    pub(crate) static_threadgroup_bytes: u32,
    /// What every threadgroup's threads must be a multiple of: the subgroup size of a tier that
    /// runs each SIMD group as one subgroup, where the local size is flattened.
    pub(crate) multiples: Vec<u32>,
    /// The subgroup size the kernel's tier pins
    /// (`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo`), where it pins one.
    pub(crate) required_subgroup_size: Option<u32>,
    /// The stage's create flags (`REQUIRE_FULL_SUBGROUPS`).
    pub(crate) stage_flags: vk::PipelineShaderStageCreateFlags,
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
    /// slot's `set_bytes` copy, or 0 where nothing is bound.
    BufferAddress(u32),
    /// The dispatch's region, 32 bytes read as two `uvec4`: the first threadgroup of the dispatch
    /// in Metal's grid (0 for a whole dispatch) with the threadgroup size packed as
    /// `(x-1) | (y-1) << 10 | (z-1) << 20`, then Metal's threads per grid (exact for
    /// `dispatch_threads`, threadgroups times their size for `dispatch_threadgroups`, and zeros
    /// for an indirect dispatch, whose kernel takes the grid from its workgroup count) and 0.
    DispatchRegion,
}

/// A `VkPipeline`'s specialization: the threadgroup size and the threadgroup memory lengths in the
/// order of [`Interface::threadgroups`].
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct Variant {
    pub(crate) size: [u32; 3],
    pub(crate) threadgroup_lengths: Vec<u32>,
}

impl crate::Device {
    /// `newComputePipelineStateWithDescriptor:options:reflection:error:`, without reflection.
    ///
    /// The translator compiles the kernel on the calling thread, which waits for it, within the
    /// translator's 20-second budget per lowering: once for a kernel that does not observe its
    /// threadgroup partition, whose 256-wide tile's `VkPipeline` is made here too, and twice for
    /// one that does (its partial-threadgroups variant and its plain module). Where the device
    /// has a cache directory ([`DeviceOptions::pipeline_cache_dir`]), a lowering a device made
    /// before is read from it in place of lowering again, and every `VkPipeline` is made through
    /// the pipeline data it keeps. The state's other `VkPipeline`s are made later, each by the
    /// first encoding call that needs it ([`ComputePipelineState`]).
    ///
    /// [`DeviceOptions::pipeline_cache_dir`]: crate::DeviceOptions::pipeline_cache_dir
    ///
    /// Err: [`Error::Translation`], the translator's typed refusal or a descriptor Metal refuses;
    /// [`Gap::IndirectCommandBuffers`]; [`Gap::KernelInput`] for a kernel that reads what the
    /// compute encoder does not supply yet; [`Error::Vulkan`].
    pub fn new_compute_pipeline_state(
        &self,
        descriptor: &ComputePipelineDescriptor,
    ) -> Result<ComputePipelineState, Error> {
        if descriptor.support_indirect_command_buffers {
            return Err(Error::Unsupported(Gap::IndirectCommandBuffers));
        }
        let core = Arc::clone(self.core());
        let budget = Budget::standard();
        let translator = &core.translator;
        let compiled = |function: CompiledFunction| {
            let mut compiled = descriptor.clone();
            compiled.kernel = StageSource::Compiled(function);
            translator.compile_compute_pipeline(&compiled, &budget)
        };
        let (whole, partial, observes) = match &descriptor.kernel {
            StageSource::Function(function) => {
                let entries = StageEntries::new();
                let plain = translator
                    .kernel_inputs_read(function, &descriptor.lowering_inputs(), &budget, &entries)
                    .map_err(Error::Translation)?;
                let mut inputs = plain.clone();
                inputs.partial_threadgroups = true;
                let lower =
                    |inputs: &LoweringInputs| lower(&core, function, inputs, &budget, &entries);
                // The translator's refusal of the variant (emulated simdgroups, a vote word its
                // workgroup memory cannot hold), kept like a lowering: the kernel keeps its plain
                // module.
                let variant_key = *translator.function_compile_key(function, &inputs).digest();
                let variant = if core.caches.refused(&variant_key) {
                    core.counters.count(Metric::TranslationsLoaded);
                    None
                } else {
                    match lower(&inputs) {
                        Ok(variant) => Some(variant),
                        Err(TranslationError::Unsupported(refused))
                            if refused.feature == Feature::GpuEncodedIndirectCommands =>
                        {
                            core.caches.keep_refusal(&variant_key);
                            None
                        }
                        Err(error) => return Err(Error::Translation(error)),
                    }
                };
                match variant {
                    Some(variant) if !variant.observes_threadgroup_partition() => {
                        (compiled(variant).map_err(Error::Translation)?, None, false)
                    }
                    Some(variant) => {
                        let plain = lower(&plain).and_then(compiled);
                        let variant = compiled(variant);
                        (
                            plain.map_err(Error::Translation)?,
                            Some(variant.map_err(Error::Translation)?),
                            true,
                        )
                    }
                    None => {
                        let plain = lower(&plain).and_then(compiled);
                        (plain.map_err(Error::Translation)?, None, true)
                    }
                }
            }
            // A kernel the consumer compiled: the variant it was compiled under.
            StageSource::Compiled(function) => {
                let pipeline = translator
                    .compile_compute_pipeline(descriptor, &budget)
                    .map_err(Error::Translation)?;
                let partial = pipeline
                    .dependencies
                    .iter()
                    .any(|dependency| dependency.fact == RuntimeFact::PartialThreadgroups);
                // Only a non-observer's partial variant runs every dispatch in tiles.
                (
                    pipeline,
                    None,
                    !(partial && !function.observes_threadgroup_partition()),
                )
            }
        };
        let required = descriptor.required_threads_per_threadgroup;
        let mut notes = whole.stage.precision_notes.clone();
        if let Some(pipeline) = &partial {
            notes.extend(&pipeline.stage.precision_notes);
        }
        notes.sort_unstable();
        notes.dedup();
        let whole = Kernel::new(&core, whole, required)?;
        // Made whole before anything else can fail, so dropping it releases what was made.
        let mut inner = PipelineInner {
            core,
            whole,
            form: match observes {
                false => Form::Tiled,
                true => Form::Sized { partial: None },
            },
            notes,
        };
        if let Some(pipeline) = partial {
            let partial = Kernel::new(&inner.core, pipeline, required)?;
            inner.form = Form::Sized {
                partial: Some(Box::new(partial)),
            };
        }
        if let Form::Tiled = inner.form {
            let variant = Variant {
                size: [TILE_THREADS, 1, 1],
                threadgroup_lengths: vec![0; inner.whole.interface.threadgroups.len()],
            };
            let made = inner.whole.make(&inner.core, &variant)?;
            lock(&inner.whole.variants).push((variant, made));
        }
        Ok(ComputePipelineState {
            inner: Arc::new(inner),
        })
    }
}

/// The kernel `function` lowered under `inputs`: the lowering a device kept in the cache directory
/// under its key, which covers everything that changes the output, or one made now and kept.
fn lower(
    core: &Core,
    function: &FunctionDescriptor,
    inputs: &LoweringInputs,
    budget: &Budget,
    entries: &StageEntries,
) -> Result<CompiledFunction, TranslationError> {
    let translator = &core.translator;
    let key = translator.function_compile_key(function, inputs);
    let kept = core
        .caches
        .translation(key.digest())
        .and_then(|bytes| CompiledFunction::from_bytes_for_key(&bytes, &key, translator).ok());
    if let Some(kept) = kept {
        core.counters.count(Metric::TranslationsLoaded);
        return Ok(kept);
    }
    core.counters.count(Metric::Translations);
    let lowered = translator.compile_function_from_entries(function, inputs, budget, entries)?;
    core.caches
        .keep_translation(key.digest(), &lowered.to_bytes());
    Ok(lowered)
}

/// The runtime's view of the translator's kernel, or [`Gap::KernelInput`] where the kernel reads
/// something the compute encoder does not supply.
fn interface(
    pipeline: &ComputePipeline,
    required_size: Option<[u32; 3]>,
) -> Result<Interface, Error> {
    let translated = &pipeline.interface;
    let stage = &pipeline.stage;
    // Every argument is a pushed address or a specialization: no descriptor, no table the binder
    // builds, no constant data; the stage's flags are its subgroups'.
    let stage_flags = vk::PipelineShaderStageCreateFlags::from_raw(stage.stage_flags.as_raw());
    let subgroups = vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS
        | vk::PipelineShaderStageCreateFlags::ALLOW_VARYING_SUBGROUP_SIZE;
    if translated.needs_pointer_translation
        || translated.needs_handle_translation
        || translated.reaches_command_buffers
        || translated
            .numbered_samplers
            .iter()
            .any(|(_, samplers)| !samplers.is_empty())
        || !translated.stage_in.is_empty()
        || pipeline
            .layout
            .sets
            .iter()
            .any(|set| !set.bindings.is_empty())
        || stage.constant_blob.is_some()
        || !subgroups.contains(stage_flags)
    {
        return Err(gap());
    }
    // The bind-time facts the encoder supplies.
    let mut multiples = Vec::new();
    let mut vote = 0;
    for dependency in &pipeline.dependencies {
        match dependency.fact {
            RuntimeFact::LocalSizeConstraint { multiple } => multiples.push(multiple),
            // The walk's vote word: 4 bytes beside the threadgroup memory.
            RuntimeFact::PredicationVote => vote = 4,
            RuntimeFact::ThreadsPerThreadgroup
            | RuntimeFact::ThreadgroupMemoryLength { .. }
            | RuntimeFact::LocalSizeLimits
            | RuntimeFact::DispatchGrid
            | RuntimeFact::IndirectGroupCount
            | RuntimeFact::PartialThreadgroups
            | RuntimeFact::DeviceAddresses { .. } => {}
            _ => return Err(gap()),
        }
    }
    let mut buffers = Vec::new();
    for (slot, used) in &translated.slots {
        match (slot.stage, slot.class, used) {
            (Stage::Kernel, SlotClass::Buffer, SlotInterface::Buffer(buffer))
                if buffer.kind == BufferKind::Buffer =>
            {
                buffers.push((slot.index, access(buffer.access)));
            }
            _ => return Err(gap()),
        }
    }
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
                        class: SlotClass::Buffer,
                        index,
                    }),
                    _,
                ) => FieldValue::BufferAddress(index),
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
    let max_total_threads = match required_size {
        Some(size) => size.iter().product(),
        None => pipeline.metal_info.max_total_threads_per_threadgroup,
    };
    Ok(Interface {
        buffers,
        reads_unbound: translated.reads_unbound_resources,
        writes_unbound: translated.writes_unbound_resources,
        push_len,
        block_len,
        block_address,
        fields,
        constants,
        threadgroups,
        max_total_threads,
        static_threadgroup_bytes: pipeline.metal_info.static_threadgroup_memory_length + vote,
        multiples,
        required_subgroup_size: stage.required_subgroup_size,
        stage_flags,
    })
}

/// The refusal of a kernel that reads what the compute encoder does not supply.
fn gap() -> Error {
    Error::Unsupported(Gap::KernelInput)
}

/// The hazard table's access of a slot the kernel reads, writes, or both.
fn access(used: AccessSet) -> Access {
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
        self.inner.whole.interface.max_total_threads
    }

    /// No Metal counterpart: where the kernel's answers may differ from Metal's on this device,
    /// fixed per operation when the state is made, for every module the state runs. Metal's
    /// answers are its own GPU's.
    pub fn precision_notes(&self) -> &[PrecisionNote] {
        &self.inner.notes
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
        Ok(ComputePipelineState {
            inner: Arc::new(PipelineInner {
                core,
                whole: Kernel {
                    module,
                    entry: c"main".to_owned(),
                    interface,
                    variants: Mutex::new(Vec::new()),
                },
                form: Form::Sized { partial: None },
                notes: Vec::new(),
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
        &self.core
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
            variants: Mutex::new(Vec::new()),
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

    /// The `VkPipeline` of `variant`, made now where it is not yet: the calling thread waits for
    /// it, a first-use wait, so what is recorded never depends on when a compile finishes.
    pub(crate) fn pipeline(&self, core: &Core, variant: &Variant) -> Result<vk::Pipeline, Error> {
        let mut variants = lock(&self.variants);
        if let Some((_, pipeline)) = variants.iter().find(|(made, _)| made == variant) {
            return Ok(*pipeline);
        }
        let began = host_time();
        let pipeline = self.make(core, variant)?;
        core.counters
            .first_use_wait(host_time().saturating_sub(began));
        variants.push((variant.clone(), pipeline));
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
        let device = &core.device;
        let cache = core.caches.pipeline_cache;
        // SAFETY: the module, the layout and the cache are the device's, and the specialization
        // gives each of the kernel's constants a 4-byte value.
        let made = unsafe { device.create_compute_pipelines(cache, &[info], None) };
        let pipeline = made.map_err(|(_, result)| Error::Vulkan {
            call: "vkCreateComputePipelines",
            result,
        })?[0];
        core.counters.count(Metric::PipelinesCreated);
        core.counters.rise(Gauge::Pipelines);
        if core.caches.made() {
            core.reaper.wake();
        }
        Ok(pipeline)
    }

    /// Gives the module and its `VkPipeline`s to the reaper.
    fn release(&mut self, core: &Core) {
        let variants = self
            .variants
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (_, pipeline) in variants.drain(..) {
            core.reaper.give(Job::Free(Free::Pipeline(pipeline)));
        }
        core.reaper.give(Job::Free(Free::ShaderModule(self.module)));
    }
}

impl Drop for PipelineInner {
    fn drop(&mut self) {
        // No command buffer that used it is left: each kept the state until it retired.
        self.whole.release(&self.core);
        if let Form::Sized {
            partial: Some(partial),
        } = &mut self.form
        {
            partial.release(&self.core);
        }
    }
}
