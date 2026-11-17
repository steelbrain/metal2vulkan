//! Compute pipeline states: the translator's kernel and its interface, and its `VkPipeline`s, one
//! per threadgroup size and threadgroup memory lengths, each made by the encoding call that first
//! needs it.
//!
//! Nothing the runtime records depends on when a compile finishes: an encoding call whose
//! `VkPipeline` is not made yet makes it and waits for it, then records, so there is one recorded
//! form. Each such wait is a first-use wait, counted with its time
//! ([`Metrics::first_use_waits`](crate::Metrics::first_use_waits)).

use crate::counters::{Gauge, Metric};
use crate::device::{lock, Core};
use crate::error::{Error, Gap};
use crate::hazard::Access;
use crate::reaper::{Free, Job};
use crate::ring::CHUNK;
use crate::times::host_time;
use ash::vk;
use metal2vulkan::pipeline::{
    AccessSet, Axis, BufferKind, ComputePipeline, ComputePipelineDescriptor, PrecisionNote,
    PushField, PushLoc, RuntimeFact, SlotClass, SlotInterface, SlotRef, SpecKey, Stage,
};
use metal2vulkan::Budget;
use std::ffi::CString;
use std::sync::{Arc, Mutex};

/// The push window: the parameters a kernel reads from push constants, `[0, 128)` on every device,
/// as 128 bytes is what Vulkan guarantees. Fields past it go to a block in the parameter ring,
/// whose address is a field in the window.
pub(crate) const PUSH_BYTES: u32 = 128;

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
    module: vk::ShaderModule,
    /// The module's entry point.
    entry: CString,
    pub(crate) interface: Interface,
    /// The translator's notes of where the kernel's answers may differ from Metal's.
    notes: Vec<PrecisionNote>,
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
    /// translator's 20-second budget. Each call compiles anew: no cache keeps a state made before.
    /// The state's `VkPipeline`s are made later, each by the first encoding call that needs it
    /// ([`ComputePipelineState`]).
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
        let pipeline = core
            .translator
            .compile_compute_pipeline(descriptor, &Budget::standard())
            .map_err(Error::Translation)?;
        let interface = interface(&pipeline, descriptor.required_threads_per_threadgroup)?;
        let stage = pipeline.stage;
        // The translator's entry points hold no NUL; one that did would name no entry point.
        let entry = CString::new(stage.entry_point).map_err(|_| gap())?;
        let info = vk::ShaderModuleCreateInfo::default().code(&stage.spirv[..]);
        // SAFETY: the words are the translator's SPIR-V, for the features device creation enabled.
        let module = unsafe { core.device.create_shader_module(&info, None) }
            .map_err(Error::vulkan("vkCreateShaderModule"))?;
        Ok(ComputePipelineState {
            inner: Arc::new(PipelineInner {
                core,
                module,
                entry,
                interface,
                notes: stage.precision_notes,
                variants: Mutex::new(Vec::new()),
            }),
        })
    }
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
    // builds, no constant data, no pinned subgroup size.
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
        || stage.required_subgroup_size.is_some()
        || !stage.stage_flags.is_empty()
    {
        return Err(gap());
    }
    // The bind-time facts the encoder supplies.
    for dependency in &pipeline.dependencies {
        match dependency.fact {
            RuntimeFact::ThreadsPerThreadgroup
            | RuntimeFact::ThreadgroupMemoryLength { .. }
            | RuntimeFact::LocalSizeLimits
            | RuntimeFact::DispatchGrid
            | RuntimeFact::IndirectGroupCount
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
    let threadgroups: Vec<u32> = translated
        .threadgroup
        .iter()
        .map(|argument| argument.index)
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
        self.inner.interface.max_total_threads
    }

    /// No Metal counterpart: where the kernel's answers may differ from Metal's on this device,
    /// fixed per operation when the state is made. Metal's answers are its own GPU's.
    pub fn precision_notes(&self) -> &[PrecisionNote] {
        &self.inner.notes
    }

    /// For the device tests: a state of a hand-made kernel and its interface.
    #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
    pub(crate) fn for_test(
        device: &crate::Device,
        words: &[u32],
        interface: Interface,
    ) -> Result<ComputePipelineState, Error> {
        let core = Arc::clone(device.core());
        let info = vk::ShaderModuleCreateInfo::default().code(words);
        // SAFETY: the words are a valid SPIR-V module.
        let module = unsafe { core.device.create_shader_module(&info, None) }
            .map_err(Error::vulkan("vkCreateShaderModule"))?;
        Ok(ComputePipelineState {
            inner: Arc::new(PipelineInner {
                core,
                module,
                entry: c"main".to_owned(),
                interface,
                notes: Vec::new(),
                variants: Mutex::new(Vec::new()),
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

    /// The `VkPipeline` of `variant`, made now where it is not yet: the calling thread waits for
    /// it, a first-use wait, so what is recorded never depends on when a compile finishes.
    pub(crate) fn pipeline(&self, variant: &Variant) -> Result<vk::Pipeline, Error> {
        let mut variants = lock(&self.variants);
        if let Some((_, pipeline)) = variants.iter().find(|(made, _)| made == variant) {
            return Ok(*pipeline);
        }
        let began = host_time();
        let pipeline = self.make(variant)?;
        self.core
            .counters
            .first_use_wait(host_time().saturating_sub(began));
        variants.push((variant.clone(), pipeline));
        Ok(pipeline)
    }

    fn make(&self, variant: &Variant) -> Result<vk::Pipeline, Error> {
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
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(self.module)
            .name(&self.entry)
            .specialization_info(&specialization);
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.core.layout);
        let device = &self.core.device;
        // SAFETY: the module and the layout are the device's, and the specialization gives each
        // of the kernel's constants a 4-byte value.
        let made =
            unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &[info], None) };
        let pipeline = made.map_err(|(_, result)| Error::Vulkan {
            call: "vkCreateComputePipelines",
            result,
        })?[0];
        self.core.counters.count(Metric::PipelinesCreated);
        self.core.counters.rise(Gauge::Pipelines);
        Ok(pipeline)
    }
}

impl Drop for PipelineInner {
    fn drop(&mut self) {
        // No command buffer that used it is left: each kept the state until it retired.
        let variants = self
            .variants
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (_, pipeline) in variants.drain(..) {
            self.core.reaper.give(Job::Free(Free::Pipeline(pipeline)));
        }
        self.core
            .reaper
            .give(Job::Free(Free::ShaderModule(self.module)));
    }
}
