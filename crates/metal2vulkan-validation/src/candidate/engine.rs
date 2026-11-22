//! The engine candidate.
//!
//! The executor dogfoods the product's consumer path end to end: the opened Vulkan device is
//! queried (`DeviceDescription::query`, restricted to what the context enabled,
//! `DeviceDescription::with_enabled`) into a `Device`, a kernel case's AIR goes through
//! `Library` → `Function` → `compile_function` → `compile_compute_pipeline` with the inputs its
//! bind shape decides ([`variants`]), and the dispatch is planned by the binder
//! (`ComputePipeline::plan`, `BindingPlan::instantiate_dispatch`): descriptor writes (buffers,
//! textures, texel buffers, samplers), ranges, push bytes and holes, neutral stand-ins, the root
//! table, per-region specialization and group counts. A vertex case compiles and draws through
//! one `RenderPipeline` the same way, a fragment case and an observed vertex case with their
//! generated companion stage ([`graphics`]). The case's own function links the functions its
//! visible-function references name, from the modules the Metal oracle links
//! ([`linked_modules`]). Each element of a texture array is the texture at its own Metal slot
//! ([`resources::bound_textures`]), and an argument-buffer buffer of either executor is a used
//! resource whose address its owner holds ([`argument_buffers`]); an argument-buffer texture is a
//! used texture whose resource id its owner holds, a member of the plan's heap. Nothing about the
//! layout is re-derived here.
//!
//! Safety and evidence rules, in the order they apply:
//!
//! - admission ([`ExecutionRequest::EngineCandidate`]) checks the case and its S3 caps before
//!   any device is requested;
//! - the compiled modules carry no execution bound: a cyclic case runs as compiled on every device.
//!   A submitted shader that does not end is not cancelled: the finite device wait ends only the
//!   host's wait (the row fails), the per-run timeout ends only the host process, and the hang
//!   tripwire only refuses later runs; the OS or driver GPU timeout is what stops the shader;
//! - the `ExecutionStatus` buffer starts zero ([`ExecutionStatus::initial`]) and is read before
//!   any output ([`status_verdict`]): an indirect overflow or a resident-set miss is never
//!   evidence;
//! - the context runs under `VK_LAYER_KHRONOS_validation` (core and synchronization); any
//!   message it reports fails the row.
//!
//! This module is the device-free half (case mapping, the status rule, identity);
//! the Vulkan half is `platform::{compute,render,plan_resources}`. Every pipeline compiles in a
//! killable compile worker ([`compile_case`], [`compile_render_case`]), never in the process that
//! holds the device, and the device half of a tool's run is itself a child, the device worker
//! (`run_on_device`, `worker::device`): a driver fault ends the worker and comes back as a typed
//! `ExecutorError::DeviceCrash`.

use crate::acceptance::{LinkedLibrary, PipelineRequest, RequestFunction};
use crate::c3::{AnswerGap, ReviewedList, VenueProfile, Venues, VENUE_PROFILE};
use crate::case::out_of_bounds::{OutOfBoundsResource, OutOfBoundsTarget};
use crate::case::{
    ArgumentBufferSamplerResource, AuthoredCase, OutputSelection, SamplerResource, Stage,
};
use crate::check::admission::{
    admit, AdmittedCase, ExecutionRequest, ExecutorError, NotSupported, VenueMismatch,
};
use crate::check::interface::AirInterface;
use crate::check::CheckedCase;
use crate::companion::{Companion, Spelling};
use crate::coverage::ExecutedCensus;
use crate::cutover::List as _;
use crate::engine::{bounded_detail, Outcome, RowSource};
use crate::executor_contract::out_of_bounds::{self, OutOfBoundsRun, TargetRun};
use crate::gpu::wait::wait_timeout;
use crate::gpu::DeviceType;
use crate::hash::sha256_bytes;
use crate::library_module::ResolvedLinkedFunctions;
use crate::literal::{LiteralDepthStencil, LiteralResources, LiteralTexture};
use crate::metal::ORACLE_ABI;
use crate::observation::{
    may_be_noted, CandidateDependencies, CandidateObservation, CandidateStatus, CaseTiming,
    MetalObservation, RUNTIME_CRATE_FINGERPRINT, TRANSLATOR_FINGERPRINT,
};
use crate::observation_contract::ComparisonClass;
use crate::refusal::{Refusal, RefusalObservation};
use crate::requirement::{Executor, ToolingRequirement};
use crate::store::CorpusStore;
use crate::worker::compile::{
    compile_compute_request, compile_render_request, CompileWorker, NotCompiled, WorkerCompile,
    WorkerDevice,
};
use crate::worker::device::{DeviceInput, DeviceOutcome, DeviceRun, OpenedDevice, VenueRun};
use crate::worker::WorkerFailure;
use base64::Engine as _;
use metal2vulkan::bind::{
    BindShape, BindState, BoundBuffer, BufferRanges, Dispatch, DrawCall, ExecutionStatus,
    StageTable, StatusReport,
};
use metal2vulkan::cache::Hash256;
use metal2vulkan::device::DeviceProfile;
use metal2vulkan::error::{BindError, Error};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::RenderStages;
use metal2vulkan::pipeline::Stage as ShaderStage;
use metal2vulkan::pipeline::{ComputePipeline, RenderPipeline};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Duration;

pub mod argument_buffers;
pub mod artifact;
pub mod graphics;
pub mod guards;
pub mod mesh;
#[cfg(test)]
mod public_cases;
pub mod resources;
#[cfg(test)]
pub(crate) mod tests;
pub mod variants;

/// The executor ABI of engine candidate observations. Every recorded candidate row, the
/// authored compile census and the C3 records key on it, so a bump makes every recorded row stale.
///
/// Bump it whenever what the executor enables, binds, submits, reads back, or records changes, or
/// how it decides a match. A change no recorded row can observe (a case kind no earlier executor
/// ran, or one it refused) needs no bump; its commit says why. The commit that bumps it says what
/// changed; this comment keeps no history.
pub const ENGINE_EXECUTOR_ABI: &str = "vulkan-engine-v212";

/// One kernel case in the terms the binder takes: what this executor supports of an
/// [`AuthoredCase`] (buffers, device buffer arrays, argument-buffer buffers and textures,
/// textures, texture arrays, samplers, threadgroup lengths, function
/// constants, a `dispatchThreads` grid, and a buffer, argument-buffer buffer, texture or
/// argument-buffer texture output).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComputeCase {
    pub entry: String,
    pub constants: FunctionConstantValues,
    /// Initial bytes of each bound buffer, by Metal buffer index. Element `i` of a device buffer
    /// array at `binding` is the buffer bound at `binding + i`, as the Metal oracle binds it
    /// (`executor_contract::device_buffer_array_completeness_error`) and as the product reads a
    /// buffer table. An argument buffer's owner covers each of its embedded pointer fields
    /// ([`ComputeCase::argument_buffer_buffers`]).
    pub buffers: BTreeMap<u32, Vec<u8>>,
    /// Initial bytes of each argument-buffer buffer, by `(owner Metal index, field offset)`:
    /// a buffer the owner's argument buffer points to (loaded pointers). The run writes its guest
    /// address into the owner's bytes at the field offset, the pointer Metal's argument encoder
    /// writes there ([`argument_buffers::owner_bytes`]), and lists it as a used resource
    /// (`useResource`, read and write, as the oracle does), so the resident set reaches it.
    pub argument_buffer_buffers: BTreeMap<(u32, u32), Vec<u8>>,
    /// Each argument-buffer texture (initial contents included), by `(owner Metal index, field
    /// offset)`: a used texture whose resource id the run writes into the owner at the field, the
    /// `gpuResourceID` Metal's argument encoder writes there ([`argument_buffers::field_words`]),
    /// resident for reading and writing, so the plan's heap holds it and the kernel's handle table
    /// finds it.
    pub argument_buffer_textures: BTreeMap<(u32, u32), LiteralTexture>,
    /// Each argument-buffer sampler, by `(owner Metal index, field offset)`: a sampler state whose
    /// id ([`argument_buffers::sampler_ids`]) the run writes into the owner at the field, the
    /// `gpuResourceID` Metal's argument encoder writes there, and lists among the bind state's
    /// argument samplers, so the plan's sampler arrays hold it and the function's sampler table
    /// finds it.
    pub argument_buffer_samplers: BTreeMap<(u32, u32), ArgumentBufferSamplerResource>,
    /// Each bound texture (initial contents included), by Metal texture index: the case's textures
    /// and each element of its texture arrays at its own slot ([`resources::bound_textures`]).
    pub textures: BTreeMap<u32, LiteralTexture>,
    /// Each bound sampler, by Metal sampler index.
    pub samplers: BTreeMap<u32, SamplerResource>,
    /// `setThreadgroupMemoryLength:atIndex:`.
    pub threadgroup_lengths: BTreeMap<u32, u32>,
    pub dispatch: Dispatch,
    pub output: ComputeOutput,
    /// The case's out-of-bounds targets: each buffer target is allocated inside a guard of its
    /// reach, and the run records how the compiled module contains each target
    /// (`executor_contract::out_of_bounds::OutOfBoundsRun`).
    pub out_of_bounds: Vec<OutOfBoundsTarget>,
    /// The bound each driver creation of the case's own pipeline is held to
    /// (`budgets::creation_within`): the case's declared one (`AuthoredCase::creation_bound`), or
    /// the shared ceiling.
    pub creation_bound: Duration,
}

/// Which bytes of the run are the observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputeOutput {
    /// The empty byte sequence.
    None,
    /// `length` bytes at `offset` of the buffer bound at Metal index `binding`.
    Buffer {
        binding: u32,
        offset: u64,
        length: u64,
    },
    /// A region of the texture bound at Metal index `binding`, in the case's spelling
    /// ([`resources::select_texture`]).
    Texture {
        binding: u32,
        origin: [u32; 3],
        dimensions: [u32; 3],
    },
    /// `length` bytes at `offset` of the argument-buffer buffer at `field_offset` of the buffer
    /// bound at Metal index `owner`.
    ArgumentBufferBuffer {
        owner: u32,
        field_offset: u32,
        offset: u64,
        length: u64,
    },
    /// A region of the argument-buffer texture at `field_offset` of the buffer bound at Metal
    /// index `owner`, in the case's spelling ([`resources::select_texture`]).
    ArgumentBufferTexture {
        owner: u32,
        field_offset: u32,
        origin: [u32; 3],
        dimensions: [u32; 3],
    },
}

/// What one engine run produced on a device.
#[derive(Clone, Debug)]
pub struct ComputeRun {
    pub output: Vec<u8>,
    /// SHA-256 of the pipeline's SPIR-V bytes (before specialization), where the executor
    /// compiled it itself; `None` for a run through the runtime, which keeps its modules to
    /// itself.
    pub spv_sha256: Option<String>,
    /// [`engine_fingerprint`] of the profile the case compiled for, or [`runtime_fingerprint`]
    /// for a run through the runtime.
    pub translator_fingerprint: String,
    pub environment: serde_json::Value,
    /// The binder's execution status, read before the output; `None` for a run through the
    /// runtime, whose shaders keep no status.
    pub status: Option<StatusReport>,
    /// Derived from the executed modules' precision notes and strategy census.
    pub comparison: ComparisonClass,
    /// The executed modules' census and the class of the device that ran them.
    pub census: ExecutedCensus,
}

/// The device an attempt refused on, as the engine knows it before the worker has read its clock:
/// the facts a row on that device would have recorded.
///
/// The worker's reported form ([`OpenedDevice`]) is this plus what the attempt cost, and only the
/// worker's own `run_on_device` has measured that, so this carries no figure rather than a
/// placeholder for one. It is to [`OpenedDevice`] what [`ComputeRun`] is to [`DeviceRun`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RefusedOn {
    pub environment: serde_json::Value,
    pub translator_fingerprint: String,
    pub artifact: Option<artifact::ExecutedArtifact>,
}

/// What one run on an opened device came to: the run, or why it produced none; and, for a typed
/// refusal after the device opened, that device, whose facts the refusal is stored with.
pub struct RunAttempt {
    pub result: Result<ComputeRun, ExecutorError>,
    pub refused_on: Option<RefusedOn>,
}

impl RunAttempt {
    /// An attempt with no opened device behind it.
    pub fn of(result: Result<ComputeRun, ExecutorError>) -> Self {
        Self {
            result,
            refused_on: None,
        }
    }
}

/// Map `case` onto what this executor runs, or name what it does not implement yet. The families of
/// the shared vocabulary that stop this executor for a kernel case
/// (`executor_contract::case_requirements`: among them function tables, acceleration structures and
/// intersection-function tables, imageblocks, stage inputs, rate maps, sparse textures and texture
/// buffers, the runtime's limits, and every indirect command buffer and pipeline state an argument
/// buffer holds) are `NotSupported::Requirements`; it runs buffers, device buffer arrays,
/// argument-buffer buffers and textures, textures, texture arrays, samplers and threadgroup
/// lengths, and a case whose literals do not map onto them is
/// `NotSupported::NotImplemented` (graphics stages are `candidate::engine::graphics`). A case has no
/// indirect dispatch to map: its dispatch is a `dispatchThreads` grid, and indirect work is
/// authored only as indirect command buffers.
pub fn compute_case(case: &AuthoredCase) -> Result<ComputeCase, NotSupported> {
    let gap = |what: &str| NotSupported::NotImplemented(format!("engine candidate: {what}"));
    if case.stage != Stage::Kernel {
        return Err(gap(&format!(
            "a {:?} case is not a compute case (graphics::render_case maps it)",
            case.stage
        )));
    }
    require_case_families(case)?;
    for (present, what) in [
        (case.depth_stencil.is_some(), "a depth/stencil attachment"),
        (!case.vertex_inputs.is_empty(), "vertex inputs"),
        (case.draw.is_some() || case.tessellation.is_some(), "draws"),
    ] {
        if present {
            return Err(gap(what));
        }
    }
    let dispatch = case
        .dispatch
        .as_ref()
        .ok_or_else(|| gap("a kernel case without a dispatch"))?;
    let output =
        match case.output {
            OutputSelection::None => ComputeOutput::None,
            OutputSelection::Buffer {
                binding,
                offset,
                length,
            } => ComputeOutput::Buffer {
                binding,
                offset,
                length,
            },
            OutputSelection::Texture {
                binding,
                origin,
                dimensions,
            } => ComputeOutput::Texture {
                binding,
                origin,
                dimensions,
            },
            // The element is the texture bound at its slot (`resources::bound_textures`).
            OutputSelection::TextureArrayElement {
                binding,
                element,
                origin,
                dimensions,
            } => ComputeOutput::Texture {
                binding: resources::texture_array_slot(binding, element)
                    .ok_or_else(|| gap("a texture-array element slot overflows"))?,
                origin,
                dimensions,
            },
            OutputSelection::DeviceBufferArrayElement {
                binding,
                element,
                offset,
                length,
            } => ComputeOutput::Buffer {
                binding: array_element_slot(binding, element)
                    .ok_or_else(|| gap("a device-buffer-array element slot overflows"))?,
                offset,
                length,
            },
            OutputSelection::ArgumentBufferBuffer {
                buffer_binding,
                field_offset,
                offset,
                length,
            } => ComputeOutput::ArgumentBufferBuffer {
                owner: buffer_binding,
                field_offset,
                offset,
                length,
            },
            OutputSelection::ArgumentBufferTexture {
                buffer_binding,
                field_offset,
                origin,
                dimensions,
            } => ComputeOutput::ArgumentBufferTexture {
                owner: buffer_binding,
                field_offset,
                origin,
                dimensions,
            },
            _ => return Err(gap(
                "an output other than a buffer range, an argument-buffer buffer range, a region \
                 of a texture, a texture-array element, or an argument-buffer texture",
            )),
        };
    let resources =
        LiteralResources::prepare(case).map_err(|error| gap(&format!("resources: {error}")))?;
    let constants = function_constants(&resources)?;
    let mut buffers = BTreeMap::new();
    let elements = resources
        .device_buffer_arrays
        .into_iter()
        .flat_map(|array| {
            let binding = array.binding;
            array
                .elements
                .into_iter()
                .map(move |element| (array_element_slot(binding, element.index), element.bytes))
        });
    for (binding, bytes) in resources
        .buffers
        .into_iter()
        .map(|buffer| (Some(buffer.binding), buffer.bytes))
        .chain(elements)
    {
        let binding = binding.ok_or_else(|| gap("a device-buffer-array element slot overflows"))?;
        if buffers.insert(binding, bytes).is_some() {
            return Err(gap(&format!("buffer {binding} bound twice")));
        }
    }
    let argument_buffer_buffers =
        argument_buffers::embed(&mut buffers, &resources.argument_buffer_buffers)
            .map_err(|error| gap(&error))?;
    let argument_buffer_textures =
        argument_buffers::embed_textures(&mut buffers, &resources.argument_buffer_textures)
            .map_err(|error| gap(&error))?;
    let argument_buffer_samplers =
        argument_buffers::embed_samplers(&mut buffers, &case.argument_buffer_samplers)
            .map_err(|error| gap(&error))?;
    let textures = resources::bound_textures(&resources.textures, &resources.texture_arrays)
        .map_err(|error| gap(&error))?;
    let mut samplers = BTreeMap::new();
    for sampler in &case.samplers {
        if samplers.insert(sampler.binding, sampler.clone()).is_some() {
            return Err(gap(&format!("sampler {} bound twice", sampler.binding)));
        }
    }
    let mut threadgroup_lengths = BTreeMap::new();
    for memory in &case.threadgroup_memory {
        threadgroup_lengths.insert(memory.binding, memory.length);
    }
    Ok(ComputeCase {
        entry: case.entry.clone(),
        constants,
        buffers,
        argument_buffer_buffers,
        argument_buffer_textures,
        argument_buffer_samplers,
        textures,
        samplers,
        threadgroup_lengths,
        // `dispatch.grid` is in threads: the Metal oracle calls
        // `dispatchThreads:threadsPerThreadgroup:`.
        dispatch: Dispatch::Threads {
            threads: dispatch.grid,
            threads_per_group: dispatch.threads_per_threadgroup,
        },
        output,
        out_of_bounds: case
            .out_of_bounds
            .as_ref()
            .map(|declaration| declaration.targets.clone())
            .unwrap_or_default(),
        creation_bound: case.pipeline_creation_bound(),
    })
}

/// Refuse the families of `case`'s own literals that stop this executor
/// (`executor_contract::case_requirements`): argument-buffer intersection-function tables, linked
/// intersection functions, the runtime's limits (function tables, acceleration structures and
/// intersection-function tables, imageblocks, stage inputs, rate maps, sparse textures, texture
/// buffers and dynamic vertex strides among them), and the render executor's own (device buffer
/// arrays, a vertex case's depth/stencil attachment, a rasterization-disabled vertex case's render
/// targets; it draws a tessellation case's patches). The checker computed them with the source's
/// families ([`crate::check::CheckedShader::requirements`]), which [`prepare_case`] refuses; this
/// check keeps a case mapped without the checker to the same rule. (The compute executor binds
/// device buffer arrays, [`compute_case`]. Both executors bind argument-buffer buffers,
/// [`argument_buffers`], and texture arrays, each element the texture at its slot,
/// [`resources::bound_textures`], and argument-buffer textures and samplers. Visible-function
/// references are no resource: the case's function links them, [`linked_modules`].)
pub fn require_case_families(case: &AuthoredCase) -> Result<(), NotSupported> {
    crate::executor_contract::require_executor(
        &crate::executor_contract::case_requirements(case),
        Executor::EngineCandidate,
    )
    .map_err(NotSupported::Requirements)
}

/// The Metal buffer index of element `element` of the device buffer array at `binding`, or `None`
/// when it overflows.
fn array_element_slot(binding: u32, element: u32) -> Option<u32> {
    binding.checked_add(element)
}

/// The function-constant values a case's prepared `resources` supply, in the product's terms.
pub fn function_constants(
    resources: &LiteralResources,
) -> Result<FunctionConstantValues, NotSupported> {
    let mut constants = FunctionConstantValues::defaults();
    for constant in &resources.function_constants {
        let ty = constant
            .scalar_type
            .metal_data_type(constant.lanes)
            .ok_or_else(|| {
                NotSupported::NotImplemented(format!(
                    "engine candidate: function constant {} of type {:?}x{}",
                    constant.index, constant.scalar_type, constant.lanes
                ))
            })?;
        constants = constants.with_index(
            constant.index,
            FunctionConstantValue::new(ty, constant.bytes.clone()),
        );
    }
    Ok(constants)
}

/// The view base alignment the executor's bind states state (`BindState::view_base_alignment`):
/// 256, the largest `minUniformBufferOffsetAlignment` Vulkan allows, a multiple of every device's
/// view base alignment, which every bind offset the
/// executor binds, 0, is a multiple of in turn. Each of its states tells it apart at every buffer
/// slot ([`views_told_apart`]), so a uniform view's range starts at its bind offset in its plan
/// (`SpecKey::RangeAtBindOffset`) on every device, as in the runtime's plan of the same binds: the
/// runtime states its device's own alignment at the slots its pipeline reads through a uniform
/// view, and the plans of the two are the same.
pub const VIEW_BASE_ALIGNMENT: u32 = 256;

/// `state` telling [`VIEW_BASE_ALIGNMENT`] apart at every buffer slot of each of its stages. Each
/// stage table classes its buffers' lengths under `ranges` (`StageTable::buffer_ranges`), the
/// device's range limits (`BufferRanges::of`), as the runtime's tables class them, so a buffer's
/// shape states the length class the runtime's shape of the same bind states.
pub fn views_told_apart(mut state: BindState, ranges: BufferRanges) -> BindState {
    state.view_base_alignment = VIEW_BASE_ALIGNMENT;
    for table in state.stages.values_mut() {
        table.at_bind_views = u64::MAX;
        table.buffer_ranges = ranges;
    }
    state
}

/// The `BindState` of `case`: each buffer bound at offset 0 with its length, its guest address,
/// and its host device address when the device has one (`addresses` by Metal index), each texture
/// and sampler ([`resources`]), the threadgroup lengths, each argument-buffer buffer as a used
/// buffer (`embedded` by
/// `(owner, field offset)`; its consumer token is its position in
/// [`ComputeCase::argument_buffer_buffers`]), and each argument-buffer texture as a used texture
/// with its resource id (its token follows the buffers', [`argument_buffers::texture_tokens`]),
/// each made resident for reading and writing, as the Metal oracle's `useResource` does, and each
/// argument-buffer sampler as an argument sampler with its id
/// ([`argument_buffers::argument_samplers`]); telling the executor's view base alignment apart and
/// classing lengths under the device's range limits `ranges` ([`views_told_apart`]).
pub fn bind_state(
    case: &ComputeCase,
    addresses: &BTreeMap<u32, BufferAddress>,
    embedded: &BTreeMap<(u32, u32), BufferAddress>,
    ranges: BufferRanges,
) -> Result<BindState, String> {
    let mut table = StageTable::default();
    for (&index, bytes) in &case.buffers {
        let address = addresses.get(&index).copied().unwrap_or_default();
        let bound = BoundBuffer::Buffer {
            length: bytes.len() as u64,
            offset: 0,
            gpu_address: address.guest,
            device_address: address.host,
            attribute_stride: None,
        };
        table.buffers.insert(index, bound);
    }
    for (&index, texture) in &case.textures {
        table
            .textures
            .insert(index, resources::bound_texture(texture)?);
    }
    for (&index, sampler) in &case.samplers {
        table
            .samplers
            .insert(index, resources::bound_sampler(sampler));
    }
    table.threadgroup_lengths = case.threadgroup_lengths.clone();
    let mut state = BindState::default();
    state.stages.insert(ShaderStage::Kernel, table);
    for (token, (key, bytes)) in case.argument_buffer_buffers.iter().enumerate() {
        let address = embedded.get(key).copied().unwrap_or_default();
        state.used_resources.push(argument_buffers::used_buffer(
            token,
            bytes.len(),
            address,
            RenderStages::empty(),
        )?);
    }
    let tokens = texture_tokens(case)?;
    for (key, texture) in &case.argument_buffer_textures {
        state.used_resources.push(argument_buffers::used_texture(
            tokens[key],
            texture,
            RenderStages::empty(),
        )?);
    }
    state.argument_samplers =
        argument_buffers::argument_samplers(case.argument_buffer_samplers.iter())?;
    Ok(views_told_apart(state, ranges))
}

/// The consumer token of each argument-buffer texture of `case`, by its key
/// ([`argument_buffers::texture_tokens`]).
pub fn texture_tokens(case: &ComputeCase) -> Result<BTreeMap<(u32, u32), u32>, String> {
    argument_buffers::texture_tokens(
        case.argument_buffer_buffers.len(),
        case.argument_buffer_textures.keys().copied(),
    )
}

/// The id of each argument-buffer sampler of `case`, by its key
/// ([`argument_buffers::sampler_ids`]).
pub fn sampler_ids(case: &ComputeCase) -> Result<BTreeMap<(u32, u32), u64>, String> {
    argument_buffers::sampler_ids(case.argument_buffer_samplers.keys().copied())
}

/// A bound buffer's addresses: the Metal `gpuAddress` the guest sees and the host Vulkan device
/// address when the buffer has one. With device addresses an allocation's guest address is its
/// device address, read as a signed number, moved up by [`BufferAddress::GUEST_OFFSET`]
/// ([`BufferAddress::addressed`]): a Metal address whatever the device hands out, which the
/// product translates through the resident set as it does the runtime's own address space.
/// Without them the guest address is a distinct synthetic base.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferAddress {
    pub guest: u64,
    pub host: Option<u64>,
}

impl BufferAddress {
    /// How far above its device address, read as a signed number, an allocation's guest address
    /// lies: 2^47 + 2^32. Vulkan bounds no device address, and devices hand out different ones:
    /// NVIDIA's lie under 2^32 (0x5020000 for a vertex buffer on an RTX 3090 and 0x4af0000 on an
    /// RTX 5080 Laptop GPU, 615.71.09), below the lowest address that names a buffer
    /// (`BufferBinding::MIN_BUFFER_ADDRESS`, 2^32; below it a record's address is a nil buffer's
    /// offset, as on Apple silicon, whose buffers all lie above it), and RADV gives the upper half
    /// of its 48-bit space sign-extended (0xffff800100040000 on a Radeon 780M), where the records
    /// of indirect command buffers lie (`records_address`, from 2^63). Read as signed, every
    /// 48-bit sign-extended address and every address under 2^63 less this offset moves into
    /// [2^32, 2^63), where Metal's buffers lie. One offset for every allocation keeps their guest
    /// ranges as disjoint as their device ranges.
    pub const GUEST_OFFSET: u64 =
        (1 << 47) + metal2vulkan::bind::indirect_commands::BufferBinding::MIN_BUFFER_ADDRESS;

    /// The addresses of an allocation at device address `host`: its guest address
    /// [`GUEST_OFFSET`](Self::GUEST_OFFSET) above it, read as a signed number. Refused where that
    /// guest address would lie below 2^32 (a device address sign-extended past 48 bits) or reach
    /// 2^63, where the records of indirect command buffers and the synthetic windows of
    /// [`BufferAddress::synthetic_used`] lie.
    pub fn addressed(host: u64) -> Result<BufferAddress, String> {
        let lowest = metal2vulkan::bind::indirect_commands::BufferBinding::MIN_BUFFER_ADDRESS;
        match host
            .cast_signed()
            .checked_add(Self::GUEST_OFFSET.cast_signed())
        {
            Some(guest) if guest >= lowest.cast_signed() => Ok(BufferAddress {
                guest: guest.cast_unsigned(),
                host: Some(host),
            }),
            _ => Err(format!(
                "device address {host:#x} has no guest address in [2^32, 2^63), where Metal's \
                 buffers lie"
            )),
        }
    }

    /// The synthetic guest base of Metal buffer `index` on a device without device addresses:
    /// disjoint 4-GiB windows, never 0.
    pub fn synthetic(index: u32) -> BufferAddress {
        BufferAddress {
            guest: (u64::from(index) + 1) << 32,
            host: None,
        }
    }

    /// The synthetic guest base of the argument-buffer buffer with consumer token `token` on a
    /// device without device addresses: 4-GiB windows in the upper half of the address space,
    /// disjoint from every [`BufferAddress::synthetic`] window.
    pub fn synthetic_used(token: u32) -> BufferAddress {
        BufferAddress {
            guest: (1 << 63) | ((u64::from(token) + 1) << 32),
            host: None,
        }
    }
}

/// Read the `ExecutionStatus` bytes after the run, before any output ([`report_verdict`]).
pub fn status_verdict(bytes: &[u8; ExecutionStatus::BYTES]) -> Result<StatusReport, ExecutorError> {
    report_verdict(ExecutionStatus::decode(bytes))
}

/// The verdict on a run's status report, read before any output: a device replay that refused a
/// command, a clamped indirect count, or a resident-set miss means the run is not a semantic
/// observation, a failure.
pub fn report_verdict(report: StatusReport) -> Result<StatusReport, ExecutorError> {
    if report.replay_fault {
        return Err(ExecutorError::Failed(format!(
            "the device replay dispatched nothing for command(s) it refused ({:#x}); the output \
             is not evidence",
            report.replay_faults.0
        )));
    }
    if report.resident_miss || report.resident_misses != 0 {
        return Err(ExecutorError::Failed(format!(
            "{} access(es) missed the resident set; the output is not evidence",
            report.resident_misses
        )));
    }
    Ok(report)
}

/// The translator fingerprint of engine observations: the product and the profile the case
/// compiled for (the opened device's queried profile, policy included).
pub fn engine_fingerprint(profile: &DeviceProfile) -> String {
    engine_fingerprint_of(&profile.fingerprint())
}

/// [`engine_fingerprint`] of the profile whose fingerprint is `profile`: what the evidence
/// census recomputes from a row's recorded `profile_fingerprint`.
pub fn engine_fingerprint_of(profile: &Hash256) -> String {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"metal2vulkan-engine-candidate\0");
    bytes.extend_from_slice(TRANSLATOR_FINGERPRINT.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&metal2vulkan::cache::SEMANTICS_VERSION.to_le_bytes());
    bytes.extend_from_slice(profile);
    sha256_bytes(&bytes)
}

/// The translator fingerprint of a run through the runtime: the product, its semantics version
/// and the runtime crate's sources
/// ([`RUNTIME_CRATE_FINGERPRINT`]). The runtime derives what it compiles for from the device it
/// opened and answers none of it, so no profile is part of it; the row's slot and its recorded
/// device are.
pub fn runtime_fingerprint() -> String {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"metal2vulkan-engine-runtime\0");
    bytes.extend_from_slice(TRANSLATOR_FINGERPRINT.as_bytes());
    bytes.push(0);
    bytes.extend_from_slice(&metal2vulkan::cache::SEMANTICS_VERSION.to_le_bytes());
    bytes.extend_from_slice(RUNTIME_CRATE_FINGERPRINT.as_bytes());
    sha256_bytes(&bytes)
}

/// The bytes of each buffer a run through the runtime binds, by Metal index, as the Metal oracle
/// creates them (`metal::platform::buffers`): a bound buffer's authored bytes padded to the size
/// its argument declares ([`buffer_bytes_padded_to_declared_size`]), and a device buffer array's
/// element's as authored.
///
/// [`buffer_bytes_padded_to_declared_size`]: crate::executor_contract::buffer_bytes_padded_to_declared_size
fn runtime_buffers(
    authored: &AuthoredCase,
    reflection: &crate::air::reflect::ShaderInterface,
    buffers: &BTreeMap<u32, Vec<u8>>,
) -> BTreeMap<u32, Vec<u8>> {
    buffers
        .iter()
        .map(|(&index, bytes)| {
            let bound = authored
                .buffers
                .iter()
                .any(|buffer| buffer.binding == index);
            let bytes = match bound {
                true => crate::executor_contract::buffer_bytes_padded_to_declared_size(
                    bytes,
                    crate::executor_contract::reflected_buffer_declared_size(reflection, index),
                )
                .into_owned(),
                false => bytes.clone(),
            };
            (index, bytes)
        })
        .collect()
}

/// The simdgroup float sums and products a run through the runtime executes
/// (`check::interface::float_reductions`): the case's entry's, which its check read, and those of
/// each function it links from `linked`, read from its module.
fn runtime_reductions(
    interface: &AirInterface,
    linked: &[LinkedModule],
) -> Result<std::collections::BTreeSet<&'static str>, ExecutorError> {
    let mut reductions = interface.float_reductions.clone();
    for module in linked {
        let decoded = crate::check::loop_free::DecodedModule::from_bitcode(
            &module.module_sha256,
            &module.bitcode,
        )
        .map_err(|error| ExecutorError::Failed(format!("a linked module: {error}")))?;
        let roots =
            crate::check::interface::functions_named(decoded.module(), &module.library.functions);
        reductions.extend(crate::check::interface::float_reductions(
            decoded.module(),
            roots,
        ));
    }
    Ok(reductions)
}

/// A run through the runtime of `prepared`'s kernel case `compute`: the row's AIR and each
/// module it links, its buffers as the oracle makes them
/// ([`runtime_buffers`]), and the float reductions it executes ([`runtime_reductions`]), run by
/// `platform::execute_runtime`.
fn run_on_runtime(prepared: &PreparedCase, compute: &ComputeCase) -> RunAttempt {
    let primary = &prepared.checked.primary;
    let kernel = match RowSource::of_row(&primary.source).bitcode() {
        Ok(kernel) => kernel,
        Err(error) => return RunAttempt::of(Err(ExecutorError::Failed(error.to_string()))),
    };
    let reductions = match runtime_reductions(&primary.interface, &prepared.linked) {
        Ok(reductions) => reductions,
        Err(error) => return RunAttempt::of(Err(error)),
    };
    let buffers = runtime_buffers(&primary.case, &primary.reflection, &compute.buffers);
    let texture_usages = runtime_texture_usages(&primary.case, &primary.interface);
    let member_usages = runtime_member_usages(&primary.case, &primary.interface);
    super::platform::execute_runtime(
        prepared.device,
        &primary.case.case_id,
        compute,
        RuntimeKernel {
            bitcode: &kernel,
            linked: &prepared.linked,
            buffers: &buffers,
            texture_usages: &texture_usages,
            member_usages: &member_usages,
            reductions: &reductions,
        },
    )
}

/// The usage each argument-buffer texture of `case` is made with in a run through the runtime,
/// by its field, as the Metal oracle makes it (`metal::platform::argument_buffers`): its role's,
/// narrowed by the access its AIR member declares. A field with no texture member has none, and
/// the run fails, as the oracle's does.
fn runtime_member_usages(
    case: &AuthoredCase,
    interface: &AirInterface,
) -> BTreeMap<argument_buffers::FieldKey, metal2vulkan::metal::TextureUsage> {
    use crate::check::interface::{declared_writable, texture_shader_usage};
    use metal2vulkan_air::abi::ParamRole;
    case.argument_buffer_textures
        .iter()
        .filter_map(|texture| {
            let (_, member) = interface.member_at(texture.buffer_binding, texture.field_offset)?;
            let ParamRole::Texture(declared) = member.role else {
                return None;
            };
            let usage =
                texture_shader_usage(texture.role, Some(declared_writable(declared.access)));
            Some(((texture.buffer_binding, texture.field_offset), usage))
        })
        .collect()
}

/// The usage each texture slot of `case` is made with in a run through the runtime, as the Metal
/// oracle makes it (`metal::platform::textures`): its role's, narrowed by what the function
/// declares of the slot (`check::interface::texture_shader_usage`), with `ShaderAtomic` where the
/// function's texture there is atomic; an element of a texture array its array's, by the array's
/// declaration, at its slot ([`resources::texture_array_slot`]), never atomic. An array element
/// at a direct texture's slot replaces it, as the array's bind does ([`resources::bound_textures`]).
fn runtime_texture_usages(
    case: &AuthoredCase,
    interface: &AirInterface,
) -> BTreeMap<u32, metal2vulkan::metal::TextureUsage> {
    use crate::check::interface::{
        declared_texture_atomic, declared_texture_writable, texture_shader_usage,
    };
    use metal2vulkan::metal::TextureUsage;
    let mut usages = BTreeMap::new();
    for texture in &case.textures {
        let mut usage = texture_shader_usage(
            texture.role,
            declared_texture_writable(interface, texture.binding),
        );
        if declared_texture_atomic(interface, texture.binding) {
            usage |= TextureUsage::SHADER_ATOMIC;
        }
        usages.insert(texture.binding, usage);
    }
    for array in &case.texture_arrays {
        let usage = texture_shader_usage(
            array.role,
            declared_texture_writable(interface, array.binding),
        );
        let slots = (0..array.elements.len())
            .filter_map(|element| u32::try_from(element).ok())
            .filter_map(|element| resources::texture_array_slot(array.binding, element));
        for slot in slots {
            usages.insert(slot, usage);
        }
    }
    usages
}

/// What a run through the runtime makes its kernel and binds from, beside its mapped case: the
/// row's AIR, the modules it links, its buffers' bytes by Metal index (each bound buffer padded to
/// its declared size, as the oracle makes it), the usage of each texture slot it binds
/// (`runtime_texture_usages`) and of each argument-buffer texture by its field
/// (`runtime_member_usages`), and the float reductions it executes
/// (`check::interface::float_reductions`).
pub struct RuntimeKernel<'a> {
    pub bitcode: &'a [u8],
    pub linked: &'a [LinkedModule],
    pub buffers: &'a BTreeMap<u32, Vec<u8>>,
    pub texture_usages: &'a BTreeMap<u32, metal2vulkan::metal::TextureUsage>,
    pub member_usages: &'a BTreeMap<argument_buffers::FieldKey, metal2vulkan::metal::TextureUsage>,
    pub reductions: &'a std::collections::BTreeSet<&'static str>,
}

/// Whether `case` draws through the runtime
/// (`platform::execute_runtime_render`) in place of the executor's own context: a vertex or
/// fragment case of any sample count (`raster_sample_count`; each attachment of more samples
/// filled and read back sample by sample, `platform::runtime_multisample`), its own stage's
/// buffers, bound textures (made as a kernel case's run makes them, `platform::runtime_textures`)
/// and bound samplers, argument buffers of buffers, textures and samplers (their textures made so
/// too), function constants and linked functions, its vertex buffers, its color targets and a depth/stencil
/// attachment of one aspect, or none (a fragment case's pass of no attachment is the oracle's 1x1
/// at its samples), the encoder's depth-stencil state, cull mode and winding, and
/// one direct or indexed draw of primitives or one patch draw with no patch or control-point index
/// buffer (a post-tessellation vertex function's, its factors bound as the tessellation factor
/// buffer), observing a buffer, a texture, an argument-buffer member, a color target, the
/// depth/stencil attachment's aspect or nothing, whose out-of-bounds targets, if any, are attribute
/// buffers
/// (`runtime_bounds::drawn` reads their facts). Decided by the case's structure alone:
/// every field the mapping fills is named here, so a new one is decided when it is added.
pub fn runs_render_through_runtime(case: &graphics::RenderCase) -> bool {
    let graphics::RenderCase {
        vertex: _,
        fragment: _,
        vertex_buffers: _,
        color_targets: _,
        depth_stencil,
        depth_stencil_state: _,
        render_target_array_length,
        assembly,
        draw,
        index_buffer: _,
        out_of_bounds,
        cull_mode: _,
        output,
        companion: _,
        raster_sample_count: _,
    } = case;
    // The runtime renders into no separate depth and stencil textures in one pass
    // (`Gap::RenderPass` of `DepthStencilViews`), which the oracle makes of an attachment of both.
    let one_aspect = |attachment: &LiteralDepthStencil| {
        attachment.depth.is_none() || attachment.stencil.is_none()
    };
    depth_stencil.as_ref().is_none_or(one_aspect)
        && *render_target_array_length == 0
        && matches!(
            (assembly, draw),
            (
                graphics::Assembly::Primitives(_),
                DrawCall::Primitives { .. } | DrawCall::Indexed { .. }
            ) | (
                graphics::Assembly::Patches { .. },
                DrawCall::Patches {
                    patch_index: None,
                    control_point_index: None,
                    ..
                }
            )
        )
        && out_of_bounds
            .iter()
            .all(|target| matches!(target.resource, OutOfBoundsResource::AttributeBuffer { .. }))
        && matches!(
            output,
            graphics::RenderOutput::None
                | graphics::RenderOutput::Buffer { .. }
                | graphics::RenderOutput::Texture { .. }
                | graphics::RenderOutput::ArgumentBufferTexture { .. }
                | graphics::RenderOutput::ArgumentBufferBuffer { .. }
                | graphics::RenderOutput::RenderTarget { .. }
                | graphics::RenderOutput::Depth { .. }
                | graphics::RenderOutput::Stencil { .. }
        )
}

/// A draw through the runtime ([`runs_render_through_runtime`]) of `prepared`'s render case
/// `render`: the row's AIR, its generated companion's, each module its own function links, its
/// own stage's buffers as the oracle makes them ([`runtime_buffers`]), and the float reductions
/// it executes ([`runtime_reductions`]), drawn by `platform::execute_runtime_render`.
fn run_render_on_runtime(prepared: &PreparedCase, render: &graphics::RenderCase) -> RunAttempt {
    let primary = &prepared.checked.primary;
    let own = match RowSource::of_row(&primary.source).bitcode() {
        Ok(own) => own,
        Err(error) => return RunAttempt::of(Err(ExecutorError::Failed(error.to_string()))),
    };
    let companion = match render.companion.as_ref().map(Companion::opaque_bitcode) {
        Some(Err(error)) => return RunAttempt::of(Err(ExecutorError::Failed(error))),
        Some(Ok(bitcode)) => Some(bitcode),
        None => None,
    };
    let reductions = match runtime_reductions(&primary.interface, &prepared.linked) {
        Ok(reductions) => reductions,
        Err(error) => return RunAttempt::of(Err(error)),
    };
    let function = match (render.companion_stage(), &render.fragment) {
        (Some(ShaderStage::Vertex), Some(fragment)) => fragment,
        _ => &render.vertex,
    };
    let buffers = runtime_buffers(
        &primary.case,
        &primary.reflection,
        &function.resources.buffers,
    );
    let texture_usages = runtime_texture_usages(&primary.case, &primary.interface);
    let member_usages = runtime_member_usages(&primary.case, &primary.interface);
    super::platform::execute_runtime_render(
        prepared.device,
        &primary.case.case_id,
        render,
        RuntimeDraw {
            bitcode: &own,
            companion: companion.as_deref(),
            linked: &prepared.linked,
            buffers: &buffers,
            texture_usages: &texture_usages,
            member_usages: &member_usages,
            reductions: &reductions,
        },
    )
}

/// What a draw through the runtime makes its pipeline state and binds from, beside its mapped
/// case: the row's AIR, its generated companion's (`None` for a rasterization-disabled vertex
/// case), the modules its own function links, its own stage's buffers' bytes by Metal index (each
/// bound buffer padded to its declared size, as the oracle makes it), the usage of each texture
/// slot its own stage binds (`runtime_texture_usages`) and of each argument-buffer texture by its
/// field (`runtime_member_usages`), and the float reductions it executes
/// (`check::interface::float_reductions`).
pub struct RuntimeDraw<'a> {
    pub bitcode: &'a [u8],
    pub companion: Option<&'a [u8]>,
    pub linked: &'a [LinkedModule],
    pub buffers: &'a BTreeMap<u32, Vec<u8>>,
    pub texture_usages: &'a BTreeMap<u32, metal2vulkan::metal::TextureUsage>,
    pub member_usages: &'a BTreeMap<argument_buffers::FieldKey, metal2vulkan::metal::TextureUsage>,
    pub reductions: &'a std::collections::BTreeSet<&'static str>,
}

/// The suffix of an engine store slot.
const ENGINE_SLOT_SUFFIX: &str = "@engine";

/// The store slot of an engine observation for the operator's `environment_id`: its own slot, so
/// an engine row never replaces the old engine's row of the same environment.
pub fn engine_environment_id(environment_id: &str) -> String {
    engine_slot(environment_id, None)
}

/// The store slot of an engine observation for the operator's `environment_id` on a device
/// `VK_LAYER_KHRONOS_profiles` restricted to the committed profile `restricted_to`
/// (`ENV@engine@P`), or on the device as it is ([`engine_environment_id`]): a row run under a
/// profile never replaces a native row, nor one run under another profile.
pub fn engine_slot(environment_id: &str, restricted_to: Option<&str>) -> String {
    match restricted_to {
        None => format!("{environment_id}{ENGINE_SLOT_SUFFIX}"),
        Some(profile) => format!("{environment_id}{ENGINE_SLOT_SUFFIX}@{profile}"),
    }
}

/// Whether a candidate row's `environment_id` is an engine slot ([`engine_slot`], native or
/// restricted): the row kind the evidence census judges by the new executor's dependency rule.
/// Neither executor accepts an operator environment ID that already names a engine slot, so the
/// slot alone decides the kind.
pub fn is_engine_slot(environment_id: &str) -> bool {
    match environment_id.rsplit_once(ENGINE_SLOT_SUFFIX) {
        Some((_, restricted)) => {
            restricted.is_empty()
                || restricted
                    .strip_prefix('@')
                    .is_some_and(|profile| !profile.is_empty() && !profile.contains(['@', '+']))
        }
        None => false,
    }
}

/// How a compiled module contains each of `targets` (`out_of_bounds::containment`: its slot's
/// form and `profile`'s encodings), or `None` for a case without targets: the run's
/// out-of-bounds facts, before what its guards count (`TargetRun::escaped`,
/// `TargetRun::past_end`). A target whose accesses the module checks nowhere
/// (`out_of_bounds::Check::Unchecked`) refuses the run, so it is decided before anything is
/// submitted, before the executor's context executes.
pub(crate) fn out_of_bounds_run(
    profile: &DeviceProfile,
    module: &out_of_bounds::ModuleFacts<'_>,
    targets: &[OutOfBoundsTarget],
) -> Result<Option<OutOfBoundsRun>, ExecutorError> {
    if targets.is_empty() {
        return Ok(None);
    }
    let targets = targets
        .iter()
        .map(|target| {
            let contained = out_of_bounds::containment(target, module, profile)
                .map_err(ExecutorError::Failed)?;
            Ok(TargetRun {
                resource: target.resource,
                form: contained.form,
                mechanism: contained.mechanism,
                written: contained.written,
                escaped: None,
                past_end: None,
            })
        })
        .collect::<Result<Vec<_>, ExecutorError>>()?;
    let unchecked = targets
        .iter()
        .filter(|target| target.mechanism.check == out_of_bounds::Check::Unchecked)
        .map(|target| target.form)
        .collect::<BTreeSet<_>>();
    if !unchecked.is_empty() {
        return Err(NotSupported::Uncontained {
            unchecked,
            variant: None,
        }
        .into());
    }
    Ok(Some(OutOfBoundsRun { targets }))
}

/// [`out_of_bounds_run`] for a render case's draw: how `pipeline`'s module, compiled for
/// `device`, that runs the Metal vertex function (`graphics::vertex_module`) contains each of
/// `case`'s targets, its attributes fetched from `case`'s vertex buffers.
pub(crate) fn draw_out_of_bounds_run(
    device: &metal2vulkan::device::Device,
    pipeline: &RenderPipeline,
    case: &graphics::RenderCase,
) -> Result<Option<OutOfBoundsRun>, ExecutorError> {
    out_of_bounds_run(
        device.profile(),
        &out_of_bounds::ModuleFacts {
            stage: ShaderStage::Vertex,
            slots: &pipeline.interface.slots,
            attributes: &out_of_bounds::Fetch::of_vertex_input(
                &pipeline.vertex_input,
                case.vertex_buffers
                    .iter()
                    .map(|vertex| (vertex.location, vertex.buffer_index)),
            ),
            resident_roots: pipeline.interface.needs_pointer_translation,
            handle_table: pipeline.interface.needs_handle_translation,
            policy: *device.policy(),
            strategies: &graphics::vertex_module(pipeline)?.strategies,
        },
        &case.out_of_bounds,
    )
}

/// Whether a draw that read `padding` index positions past where Metal's fetch stops reading its
/// index buffer from a zero-padded copy of its indices contains them as `run`'s index buffer
/// target states: the padded copy, of at least one such position. The executor's context holds
/// the draw to it; a run without that target agrees with any draw.
pub(crate) fn index_padding_agrees(
    run: Option<&OutOfBoundsRun>,
    padding: Option<u64>,
) -> Result<(), ExecutorError> {
    let Some(target) = run.and_then(|run| {
        run.targets
            .iter()
            .find(|target| target.resource == OutOfBoundsResource::IndexBuffer)
    }) else {
        return Ok(());
    };
    match (target.mechanism.check, padding) {
        (out_of_bounds::Check::Padded, Some(positions)) if positions > 0 => Ok(()),
        _ => Err(ExecutorError::Failed(format!(
            "the draw reads {padding:?} padded index positions past the index buffer, not as {} \
             contains them",
            target.mechanism.label()
        ))),
    }
}

/// A product refusal as the executor reports it: a typed outcome, or a failure when the outcome
/// is a defect (panic, internal, invalid output).
pub fn product_refusal(error: &Error) -> ExecutorError {
    outcome_refusal(Outcome::from(error), bounded_detail(&error.to_string()))
}

/// [`product_refusal`] of an error a compile worker reported as its typed outcome and detail.
pub fn outcome_refusal(outcome: Outcome, detail: String) -> ExecutorError {
    refusal_at(outcome, detail, false)
}

/// [`product_refusal`] of the binder's typed refusal of what the case binds (its resources, its
/// dispatch): `at_bind`, which its refusal record keeps.
pub fn bound_refusal(error: &Error) -> ExecutorError {
    refusal_at(
        Outcome::from(error),
        bounded_detail(&error.to_string()),
        true,
    )
}

fn refusal_at(outcome: Outcome, detail: String, at_bind: bool) -> ExecutorError {
    if outcome.is_defect() {
        ExecutorError::Failed(format!("engine {}: {detail}", outcome.key()))
    } else {
        NotSupported::Product {
            outcome,
            detail,
            at_bind,
        }
        .into()
    }
}

/// A library the case's own function links functions from: the linked module's SHA-256 (by which a
/// table entry names it), its AIR bitcode (its retained blob, the module the Metal oracle loads, or
/// a public linked module's text assembled with `llvm-as`), and the functions linked from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkedModule {
    pub module_sha256: String,
    pub bitcode: Vec<u8>,
    pub library: LinkedLibrary,
}

/// The libraries the case's own function links, from the checked case's `linked` functions as the
/// Metal oracle links them (`ResolvedLinkedFunctions::libraries`): each module once, with every
/// function it supplies, table entries included. Linking is not where tables are refused: a case
/// with a linked intersection function is refused in the case mapping
/// (`ToolingRequirement::LinkedIntersectionFunction`, [`require_case_families`]), before it links.
pub fn linked_modules(
    linked: &ResolvedLinkedFunctions,
) -> Result<Vec<LinkedModule>, ExecutorError> {
    linked
        .libraries()
        .into_iter()
        .map(|library| {
            let module = library.module;
            // A retained module's blob, or a public linked module's text assembled.
            let bitcode = module.bitcode()?;
            Ok(LinkedModule {
                module_sha256: module.module_sha256.clone(),
                bitcode,
                library: LinkedLibrary {
                    functions: library.functions.into_iter().map(str::to_owned).collect(),
                },
            })
        })
        .collect()
}

/// The compile's libraries: the stage libraries `stages`, then each of `linked`'s, retained, in
/// order ([`crate::acceptance::PipelineRequest::libraries`]).
fn with_linked<'a>(stages: &[&RowSource<'a>], linked: &[LinkedModule]) -> Vec<RowSource<'a>> {
    let mut libraries: Vec<RowSource<'a>> = stages.iter().map(|&stage| stage.clone()).collect();
    libraries.extend(
        linked
            .iter()
            .map(|module| crate::acceptance::retained(&module.bitcode)),
    );
    libraries
}

/// `entry` with `constants`, linking the functions of `linked`.
fn linking(
    entry: &str,
    constants: &FunctionConstantValues,
    linked: &[LinkedModule],
) -> RequestFunction {
    RequestFunction {
        entry: entry.into(),
        constants: constants.clone(),
        linked: linked.iter().map(|module| module.library.clone()).collect(),
    }
}

/// The compile request for `case`'s kernel from `library` (its source row's,
/// [`RowSource::of_row`]; the worker assembles a row that retains no bitcode) with the lowering
/// inputs the bind shape `shape` and the case's dispatch decide (`PipelineRequest::BoundCompute`),
/// linking the functions of `linked`, and its libraries in [`PipelineRequest::libraries`] order: `library`, then each of
/// `linked`'s.
pub fn compute_request<'a>(
    library: &RowSource<'a>,
    linked: &[LinkedModule],
    case: &ComputeCase,
    shape: &BindShape,
) -> (PipelineRequest, Vec<RowSource<'a>>) {
    let request = PipelineRequest::BoundCompute {
        function: linking(&case.entry, &case.constants, linked),
        shape: shape.clone(),
        dispatch: Some(case.dispatch),
    };
    (request, with_linked(&[library], linked))
}

/// The source of `case`'s generated companion, when it has one: its opaque AIR text, which the
/// compile worker assembles.
pub fn companion_source(case: &graphics::RenderCase) -> Option<RowSource<'static>> {
    case.companion
        .as_ref()
        .map(|generated| RowSource::Text(Cow::Owned(generated.air_text(Spelling::Opaque))))
}

/// The compile request for `case`'s pipeline with the inputs the bind shape `shape` decides
/// (storage-texture formats of either stage, pulled vertex buffers,
/// `variants::compile_render_for_shape`), and its libraries in [`PipelineRequest::libraries`]
/// order: a vertex function alone from `library` as the Metal oracle's rasterization-disabled draw
/// (`PipelineRequest::VertexDraw` of [`graphics::render_template`]), or the case's function from
/// `library` and its generated companion's from `companion` ([`companion_source`]) in stage order,
/// drawn together (`PipelineRequest::Draw`); each list ends with `linked`'s libraries. The case's
/// own function (the fragment function a passthrough vertex companion draws, the vertex function an
/// observer fragment companion observes) links the functions of `linked`; the companion links
/// nothing. A fragment function that is no companion's (which only owned device tests build,
/// through [`compile_render`]) is a failure, as is a companion without its source.
pub fn render_request<'a>(
    library: &RowSource<'a>,
    companion: Option<&RowSource<'a>>,
    linked: &[LinkedModule],
    case: &graphics::RenderCase,
    shape: &BindShape,
) -> Result<(PipelineRequest, Vec<RowSource<'a>>), ExecutorError> {
    let template = graphics::render_template(case);
    let own =
        |function: &graphics::RenderFunction| linking(&function.entry, &function.constants, linked);
    let generated = |function: &graphics::RenderFunction| {
        RequestFunction::new(function.entry.as_str(), function.constants.clone())
    };
    match (&case.fragment, &case.companion, companion) {
        (None, None, None) => {
            let request = PipelineRequest::VertexDraw {
                function: own(&case.vertex),
                template,
                shape: Some(shape.clone()),
            };
            Ok((request, with_linked(&[library], linked)))
        }
        (Some(fragment), Some(generated_stage), Some(companion)) => {
            let (stages, vertex, fragment) = match generated_stage {
                Companion::PassthroughVertex(_) => {
                    ([companion, library], generated(&case.vertex), own(fragment))
                }
                Companion::ObserverFragment(_) => {
                    ([library, companion], own(&case.vertex), generated(fragment))
                }
                Companion::MeshStages(_) => {
                    return Err(ExecutorError::Failed(
                        "a render case's companion is a mesh render pipeline's stages".into(),
                    ))
                }
            };
            let request = PipelineRequest::Draw {
                vertex,
                fragment,
                template,
                shape: Some(shape.clone()),
            };
            Ok((request, with_linked(&stages, linked)))
        }
        _ => Err(ExecutorError::Failed(
            "the compile worker draws a fragment function only as a case's generated companion \
             or with one, from that companion's source"
                .into(),
        )),
    }
}

/// A stop at the compile worker's boundary, as the run's failure.
fn worker_stopped(failure: WorkerFailure) -> ExecutorError {
    ExecutorError::Failed(format!(
        "engine compile worker ({}): {}",
        failure.outcome().key(),
        failure.message()
    ))
}

/// What a compile worker's refusal means to the executor: the product's typed refusal is
/// [`outcome_refusal`]'s. The executor's requests carry every input they need, so a request the
/// tool does not build is a failure.
fn not_compiled(refusal: NotCompiled) -> ExecutorError {
    match refusal {
        NotCompiled::Product(outcome, detail) => outcome_refusal(outcome, detail),
        NotCompiled::Unmodelled(why) => {
            ExecutorError::Failed(format!("the compile worker built no inputs: {why}"))
        }
    }
}

/// Compile `case`'s kernel from `library` for `device` in a killable compile worker, as
/// [`compute_request`] asks: the product's typed refusal is [`outcome_refusal`]'s, and a stop at
/// the worker boundary (the deadline, the memory limit, a crashed worker, an artifact that does not
/// load for `device`) fails the run.
pub fn compile_case(
    worker: &CompileWorker,
    device: &WorkerDevice,
    library: &RowSource<'_>,
    linked: &[LinkedModule],
    case: &ComputeCase,
    shape: &BindShape,
) -> Result<ComputePipeline, ExecutorError> {
    let (request, libraries) = compute_request(library, linked, case, shape);
    compile_request(worker, device, &libraries, &request)
}

/// Compile `request`, a kernel's, from `libraries` in `worker`.
fn compile_request(
    worker: &CompileWorker,
    device: &WorkerDevice,
    libraries: &[RowSource<'_>],
    request: &PipelineRequest,
) -> Result<ComputePipeline, ExecutorError> {
    compute_attempt(worker, device, libraries, request)
        .result
        .map_err(ExecutorError::from)
}

/// Why a compile of one of a case's pipelines in a compile worker gave no pipeline: a stop at the
/// worker boundary, or the worker's refusal. The executor reads each as its [`ExecutorError`]
/// ([`From`]: `worker_stopped`, `not_compiled`); the authored compile census keeps them typed.
#[derive(Debug)]
pub enum CompileFailure {
    /// The worker boundary stopped the compile (the deadline, the memory limit, a crashed
    /// worker, an artifact that does not load for the device).
    Stopped(WorkerFailure),
    /// The worker compiled nothing: the product's typed outcome, or inputs the tool does not
    /// build.
    NotCompiled(NotCompiled),
    /// The compile's digest could not be made ([`artifact::CaseArtifact::record`]).
    Artifact(ExecutorError),
}

impl From<CompileFailure> for ExecutorError {
    fn from(failure: CompileFailure) -> ExecutorError {
        match failure {
            CompileFailure::Stopped(failure) => worker_stopped(failure),
            CompileFailure::NotCompiled(refusal) => not_compiled(refusal),
            CompileFailure::Artifact(error) => error,
        }
    }
}

/// One compile of a case's pipeline in a compile worker: the pipeline, or why there is none, and
/// what the parent measured (`None` when no worker was spawned).
pub struct CompileAttempt<P> {
    pub result: Result<P, CompileFailure>,
    pub measured: Option<crate::worker::Measured>,
}

impl<P> CompileAttempt<P> {
    /// The attempt a worker compile `compiled` gave.
    fn of(compiled: Result<WorkerCompile<P>, WorkerFailure>) -> Self {
        match compiled {
            Err(failure) => CompileAttempt {
                measured: failure.measured(),
                result: Err(CompileFailure::Stopped(failure)),
            },
            Ok(compiled) => CompileAttempt {
                measured: Some(compiled.measured),
                result: compiled.result.map_err(CompileFailure::NotCompiled),
            },
        }
    }
}

/// [`compile_request`] with its failure typed and its measure ([`CompileAttempt`]).
fn compute_attempt(
    worker: &CompileWorker,
    device: &WorkerDevice,
    libraries: &[RowSource<'_>],
    request: &PipelineRequest,
) -> CompileAttempt<ComputePipeline> {
    CompileAttempt::of(compile_compute_request(worker, device, libraries, request))
}

/// [`compile_render`] with its failure typed and its measure ([`CompileAttempt`]).
fn render_attempt(
    worker: &CompileWorker,
    device: &WorkerDevice,
    request: &PipelineRequest,
    libraries: &[RowSource<'_>],
) -> CompileAttempt<RenderPipeline> {
    CompileAttempt::of(compile_render_request(worker, device, libraries, request))
}

/// Compile `case`'s pipeline for `device` in a killable compile worker, as [`render_request`]
/// asks; the worker assembles a source that retains no bitcode (the companion's text, a public
/// fixture's). Refusals and stops are classified as for [`compile_case`].
pub fn compile_render_case(
    worker: &CompileWorker,
    device: &WorkerDevice,
    library: &RowSource<'_>,
    companion: Option<&RowSource<'_>>,
    linked: &[LinkedModule],
    case: &graphics::RenderCase,
    shape: &BindShape,
) -> Result<RenderPipeline, ExecutorError> {
    let (request, libraries) = render_request(library, companion, linked, case, shape)?;
    compile_render(worker, device, &request, &libraries)
}

/// Compile `request`, a draw, from `libraries` (in [`PipelineRequest::libraries`] order) for
/// `device` in a killable compile worker. [`compile_render_case`] asks for [`render_request`]'s
/// draw; the owned device tests ask for a vertex and a fragment function of two owned libraries,
/// which no authored case draws. Refusals and stops are classified as for [`compile_case`].
pub fn compile_render(
    worker: &CompileWorker,
    device: &WorkerDevice,
    request: &PipelineRequest,
    libraries: &[RowSource<'_>],
) -> Result<RenderPipeline, ExecutorError> {
    render_attempt(worker, device, request, libraries)
        .result
        .map_err(ExecutorError::from)
}

/// The `BindShape` the executor binds `case` with on a device that has device addresses exactly
/// when `addressed` and the range limits `ranges` (`BufferRanges::of`): [`bind_state`] with every
/// buffer's guest address and, when `addressed`, a host address (only whether a buffer has one is
/// part of the shape, not its value).
pub fn compute_shape(
    case: &ComputeCase,
    addressed: bool,
    ranges: BufferRanges,
) -> Result<BindShape, String> {
    let with_host = |mut address: BufferAddress| {
        address.host = addressed.then_some(address.guest);
        address
    };
    let addresses = case
        .buffers
        .keys()
        .map(|&index| (index, with_host(BufferAddress::synthetic(index))))
        .collect();
    let embedded = synthetic_used_addresses(case)?
        .into_iter()
        .map(|(key, address)| (key, with_host(address)))
        .collect();
    Ok(bind_state(case, &addresses, &embedded, ranges)?.shape())
}

/// The synthetic guest address of each argument-buffer buffer of `case`, by its key
/// ([`argument_buffers::synthetic_addresses`]): its [`BufferAddress::synthetic_used`].
pub fn synthetic_used_addresses(
    case: &ComputeCase,
) -> Result<BTreeMap<(u32, u32), BufferAddress>, String> {
    argument_buffers::synthetic_addresses(case.argument_buffer_buffers.keys().copied())
}

/// A binder refusal as the executor reports it: an `Unsupported` is the product's typed outcome,
/// an internal disagreement between pipeline and binder is a failure, and so is a dispatch that
/// asks for another variant than the one compiled for it; every other refusal of the bind state on
/// this device is [`NotSupported::Bind`].
pub fn bind_refusal(error: &BindError) -> ExecutorError {
    match error {
        BindError::Unsupported(unsupported) => {
            bound_refusal(&Error::Unsupported(unsupported.clone()))
        }
        BindError::Internal(_) | BindError::NeedsVariant { .. } => {
            ExecutorError::Failed(format!("binder: {error}"))
        }
        _ => NotSupported::Bind(bounded_detail(&error.to_string())).into(),
    }
}

/// Admit `case` for the engine candidate on `device_type`, then open that device and execute
/// it, compiling in `worker`.
pub fn execute(
    root: &Path,
    case: AuthoredCase,
    device_type: DeviceType,
    metal_environment_id: &str,
    environment_id: &str,
    worker: &CompileWorker,
) -> Result<CandidateObservation, ExecutorError> {
    let admitted = admit(
        root,
        case,
        ExecutionRequest::EngineCandidate {
            device: device_type,
        },
    )?;
    execute_case(
        device_type,
        root,
        admitted,
        metal_environment_id,
        environment_id,
        worker,
    )
}

/// Compile (in `worker`), bind, execute on `device`, and compare one admitted case against its
/// exact Metal observation: a kernel case as one dispatch (`platform::compute`), a vertex or
/// fragment case as one draw (`platform::render`, [`graphics`]). The admission must be for this device's type. The
/// observation is stored in the engine slot of `environment_id`
/// ([`engine_environment_id`]).
pub fn execute_case(
    device: DeviceType,
    root: &Path,
    admitted: AdmittedCase,
    metal_environment_id: &str,
    environment_id: &str,
    worker: &CompileWorker,
) -> Result<CandidateObservation, ExecutorError> {
    execute_prepared(
        root,
        prepare_case(device, admitted)?,
        metal_environment_id,
        environment_id,
        worker,
    )
}

/// An admitted case that passed the executor's device-free checks ([`prepare_case`]), or the
/// compile census's ([`prepare_compile`]).
pub struct PreparedCase {
    device: DeviceType,
    checked: CheckedCase,
    mapped: Mapped,
    /// The libraries the case's own function links ([`linked_modules`]).
    linked: Vec<LinkedModule>,
}

/// An admitted case in the terms of the executor that runs it (boxed: the two differ in size).
enum Mapped {
    Compute(Box<ComputeCase>),
    Render(Box<graphics::RenderCase>),
    Mesh(Box<mesh::MeshCase>),
}

/// The executor's device-free checks of a case admitted for `device`: the admission's device, the
/// tooling requirements the checker found that stop this executor
/// ([`CheckedCase::requirements`], of the case and of every pipeline template, the source's
/// families, such as a kernel's implicit imageblock, and the case's own), what the executor maps (a
/// kernel case [`compute_case`] maps, a graphics case [`graphics::render_case`] maps with the
/// checker's generated companion, an object or mesh case [`mesh::mesh_case`] maps), and the
/// libraries its function links ([`linked_modules`]). A
/// refusal here needs neither a device nor a Metal observation, so a case this executor refuses
/// device-free never reaches the GPU.
pub fn prepare_case(
    device: DeviceType,
    admitted: AdmittedCase,
) -> Result<PreparedCase, ExecutorError> {
    prepare(device, admitted, false)
}

/// [`prepare_case`] for the authored compile census (`authored_compile`): a case whose only stop
/// is the mesh render pipeline the executor does not draw
/// ([`ToolingRequirement::MeshRenderPipeline`]) is prepared too, its draw mapped
/// ([`mesh::mesh_case`]), so [`case_compile`] plans the pipeline a run of it would compile. A
/// run's preparation refuses such a case before any device, its device worker's included.
pub fn prepare_compile(
    device: DeviceType,
    admitted: AdmittedCase,
) -> Result<PreparedCase, ExecutorError> {
    prepare(device, admitted, true)
}

/// [`prepare_case`], or with `compile_only` [`prepare_compile`].
fn prepare(
    device: DeviceType,
    admitted: AdmittedCase,
    compile_only: bool,
) -> Result<PreparedCase, ExecutorError> {
    let checked = admitted.into_checked_for(ExecutionRequest::EngineCandidate { device })?;
    let mut requirements = checked.requirements();
    if compile_only {
        requirements.remove(&ToolingRequirement::MeshRenderPipeline);
    }
    crate::executor_contract::require_executor(&requirements, Executor::EngineCandidate)
        .map_err(NotSupported::Requirements)?;
    let case = &checked.primary.case;
    let mapped = match case.stage {
        Stage::Kernel => Mapped::Compute(Box::new(compute_case(case)?)),
        Stage::Vertex | Stage::Fragment => Mapped::Render(Box::new(graphics::render_case(
            case,
            &checked.primary.interface,
            checked.primary.entry_ll(),
            checked.primary.companion.as_ref(),
        )?)),
        Stage::Object | Stage::Mesh => Mapped::Mesh(Box::new(mesh::mesh_case(
            case,
            checked.primary.companion.as_ref(),
        )?)),
    };
    let linked = linked_modules(&checked.primary.linked_functions)?;
    Ok(PreparedCase {
        device,
        checked,
        mapped,
        linked,
    })
}

/// The one pipeline a run of a prepared case compiles, with the request and libraries the executor
/// builds for it.
pub struct CaseCompile<'a> {
    pipeline: PlannedPipeline<'a>,
}

/// A planned compile's request, its libraries in [`PipelineRequest::libraries`] order, and the
/// bind shape the run plans it for: a kernel's, or a draw's (a case's own, or a mesh draw's,
/// `mesh::mesh_request`).
enum PlannedPipeline<'a> {
    Compute {
        request: PipelineRequest,
        libraries: Vec<RowSource<'a>>,
        shape: BindShape,
    },
    Render {
        request: PipelineRequest,
        libraries: Vec<RowSource<'a>>,
        shape: BindShape,
    },
}

impl CaseCompile<'_> {
    /// Compile in `worker` for `device` as the executor compiles it and record it in `artifact` as
    /// the executor's run records it (the pipeline's content and its plan for the run's bind
    /// shape), keeping only whether it compiled and what the worker measured: the pipeline is
    /// dropped here.
    pub fn compile(
        &self,
        worker: &CompileWorker,
        device: &WorkerDevice,
        artifact: &mut artifact::CaseArtifact,
    ) -> (Result<(), CompileFailure>, Option<crate::worker::Measured>) {
        let mut record = |pipeline: &dyn artifact::Planned, shape| {
            artifact
                .record(pipeline, device.device(), shape)
                .map_err(CompileFailure::Artifact)
        };
        match &self.pipeline {
            PlannedPipeline::Compute {
                request,
                libraries,
                shape,
            } => {
                let attempt = compute_attempt(worker, device, libraries, request);
                let result = attempt.result.and_then(|pipeline| record(&pipeline, shape));
                (result, attempt.measured)
            }
            PlannedPipeline::Render {
                request,
                libraries,
                shape,
            } => {
                let attempt = render_attempt(worker, device, request, libraries);
                let result = attempt.result.and_then(|pipeline| record(&pipeline, shape));
                (result, attempt.measured)
            }
        }
    }
}

/// The pipeline a run of `prepared` compiles on `device` before it reads any device output, with
/// the request the executor builds for it: the case's own pipeline ([`compute_request`] for the
/// bind shape [`compute_shape`] gives, [`render_request`] with its generated companion for
/// [`graphics::render_shape`]'s, or a mesh draw's, `mesh::mesh_request`). The bind shapes depend
/// on the device only through whether it has device addresses, which the executor reads from what
/// its context enabled and this from `device`'s description: the same fact for an opened device,
/// whose description is restricted to what was enabled; and through its profile's range limits
/// (`BufferRanges::of`), which both read from the device's profile. The authored compile census
/// compiles it on each committed profile; the executor builds the same request through
/// [`compile_case`] and [`compile_render_case`].
pub fn case_compile<'a>(
    prepared: &'a PreparedCase,
    device: &WorkerDevice,
) -> Result<CaseCompile<'a>, ExecutorError> {
    use metal2vulkan::device::VkFeature;
    let addressed = device.description().has(VkFeature::BufferDeviceAddress);
    let ranges = BufferRanges::of(device.device().profile().limits());
    let library = RowSource::of_row(&prepared.checked.primary.source);
    let linked = prepared.linked.as_slice();
    let pipeline = match &prepared.mapped {
        Mapped::Compute(compute) => {
            let shape = compute_shape(compute, addressed, ranges).map_err(ExecutorError::Failed)?;
            let (request, libraries) = compute_request(&library, linked, compute, &shape);
            PlannedPipeline::Compute {
                request,
                libraries,
                shape,
            }
        }
        Mapped::Render(render) => {
            let shape =
                graphics::render_shape(render, addressed, ranges).map_err(ExecutorError::Failed)?;
            let companion = companion_source(render);
            let (request, libraries) =
                render_request(&library, companion.as_ref(), linked, render, &shape)?;
            PlannedPipeline::Render {
                request,
                libraries,
                shape,
            }
        }
        Mapped::Mesh(mesh) => {
            let shape = mesh::mesh_shape(mesh, addressed, ranges)?;
            let companion = mesh
                .companion
                .as_ref()
                .map(|generated| RowSource::Text(Cow::Owned(generated.air_text(Spelling::Opaque))));
            let (request, libraries) = mesh::mesh_request(&library, companion, mesh, &shape)?;
            PlannedPipeline::Render {
                request,
                libraries,
                shape,
            }
        }
    };
    Ok(CaseCompile { pipeline })
}

/// The device worker's half of [`execute_prepared`] (`worker::device`): admit and prepare `input`'s
/// case again, require the parent's checked input, then open the device, compile in a compile
/// worker of this executable, bind, execute, and read back.
///
/// A run recorded under a C3 venue first holds the device to the venue's profile
/// ([`checked_venue`]): on any difference nothing runs, and the run or refusal of a device that is
/// the profile's records it ([`VenueProfile::record`]).
pub(crate) fn run_on_device(input: &DeviceInput) -> DeviceOutcome {
    let venue = match checked_venue(input) {
        Ok(venue) => venue,
        Err(error) => {
            return DeviceOutcome {
                result: Err(error),
                refused_on: None,
            }
        }
    };
    let recorded = |mut environment: serde_json::Value| {
        if let (Some(venue), Some(object)) = (&venue, environment.as_object_mut()) {
            object.insert(VENUE_PROFILE.into(), venue.record());
        }
        environment
    };
    // What the attempt cost in time, read around the attempt itself: the case's compiles, its
    // device work and its readback, with the worker's own admission and preparation of the case
    // outside it, so the figure is the case on the device and nothing else. The clock is the one
    // the pipeline-creation ceiling is checked against, so the two can never disagree.
    //
    // The outcome is built here, inside the arm that read the clock, so a run and a refusal on an
    // opened device each take the figure from it by construction: a case that never reached a
    // device (its preparation failed) is the only outcome without one, and it is neither.
    match prepare_on_device(input) {
        Ok((prepared, worker)) => {
            let clock = crate::budgets::CreationClock::start();
            let attempt = run_on_context(&prepared, &worker);
            let timing = CaseTiming::of_one(clock.read());
            DeviceOutcome {
                result: attempt.result.map(|run| DeviceRun {
                    output: run.output,
                    spv_sha256: run.spv_sha256,
                    translator_fingerprint: run.translator_fingerprint,
                    environment: recorded(run.environment),
                    comparison: run.comparison,
                    census: run.census,
                    timing,
                }),
                refused_on: attempt.refused_on.map(|opened| OpenedDevice {
                    environment: recorded(opened.environment),
                    translator_fingerprint: opened.translator_fingerprint,
                    artifact: opened.artifact,
                    timing,
                }),
            }
        }
        Err(error) => DeviceOutcome {
            result: Err(error),
            refused_on: None,
        },
    }
}

/// The committed profile of the C3 venue `input`'s run records under, held to the device the run
/// opens (`platform::venue_check`): `None` for a run no venue's, the profile when the device's
/// identity is one the venue allows and its translation key is the profile's, and
/// [`ExecutorError::VenueMismatch`] naming every difference otherwise. Description members that
/// differ while the key holds are no translation's input: they are written to stderr (which the
/// runner relays) and the run proceeds.
fn checked_venue(input: &DeviceInput) -> Result<Option<VenueProfile>, ExecutorError> {
    let Some(venue) = &input.venue else {
        return Ok(None);
    };
    let profile = VenueProfile::committed(&venue.profile)?;
    let check = super::platform::venue_check(input.device, &profile, &venue.allowed)?;
    if !check.stops.is_empty() {
        return Err(ExecutorError::VenueMismatch(VenueMismatch {
            environment_id: venue.environment_id.clone(),
            profile: venue.profile.clone(),
            differences: check.stops,
        }));
    }
    if !check.unread.is_empty() {
        eprintln!(
            "venue {}: the device differs from profile {} in {} member(s) no translation reads \
             (the run proceeds): {}",
            venue.environment_id,
            venue.profile,
            check.unread.len(),
            check.unread.join("; ")
        );
    }
    Ok(Some(profile))
}

/// The device worker's admission and preparation of `input`'s case, checked against the parent's
/// input digest, and the compile worker it compiles in.
fn prepare_on_device(input: &DeviceInput) -> Result<(PreparedCase, CompileWorker), ExecutorError> {
    let admitted = admit(
        &input.root,
        input.case.clone(),
        ExecutionRequest::EngineCandidate {
            device: input.device,
        },
    )?;
    let prepared = prepare_case(input.device, admitted)?;
    if prepared.checked.primary.input_sha256 != input.input_sha256 {
        return Err(format!(
            "the device worker checked case {} as input {}, its parent as {}: the corpus changed \
             between the two",
            input.case.case_id, prepared.checked.primary.input_sha256, input.input_sha256
        )
        .into());
    }
    Ok((prepared, CompileWorker::current()?))
}

/// Whether the executor's own context refuses a case whose argument buffers hold `members`: a
/// texture buffer among them, which a run through the runtime makes over a buffer of its bytes
/// (`platform::runtime_textures`) and the binder's planned heap holds no member of (the
/// translator's `bind::heap`), so a handle loaded from its field would miss
/// ([`argument_buffers::used_texture`]).
pub fn own_context_refuses_members<'a>(
    mut members: impl Iterator<Item = &'a LiteralTexture>,
) -> bool {
    members.any(|texture| texture.texture_type == crate::case::TextureType::Buffer)
}

/// Compile (in `worker`), bind, and execute one prepared case on the executor's context: a
/// kernel case as one dispatch, a vertex or fragment case as one draw. The run's row or refusal
/// records what it compiled, by content ([`artifact::CaseArtifact`]).
fn run_on_context(prepared: &PreparedCase, worker: &CompileWorker) -> RunAttempt {
    let case = &prepared.checked.primary.case;
    // The worker assembles a row that retains no bitcode (a public fixture), and a companion.
    let library = RowSource::of_row(&prepared.checked.primary.source);
    let linked = &prepared.linked;
    // A case the runtime runs, run so in this process's default run: no profiles-layer
    // restriction. Every kernel case the engine maps is one (`compute_case` dispatches it by
    // threads). It records no artifact: the runtime keeps what it compiled to itself.
    if let Mapped::Compute(compute) = &prepared.mapped {
        if super::requested_restriction().is_none() {
            return run_on_runtime(prepared, compute);
        }
    }
    if let Mapped::Render(render) = &prepared.mapped {
        if runs_render_through_runtime(render) && super::requested_restriction().is_none() {
            return run_render_on_runtime(prepared, render);
        }
    }
    let refused = match &prepared.mapped {
        Mapped::Compute(compute) => {
            own_context_refuses_members(compute.argument_buffer_textures.values())
        }
        Mapped::Render(render) => own_context_refuses_members(
            render
                .stages()
                .flat_map(|(_, function)| function.resources.argument_buffer_textures.values()),
        ),
        // A mesh draw's functions take no argument buffer here (`mesh::mesh_shape`).
        Mapped::Mesh(_) => false,
    };
    if refused {
        return RunAttempt::of(Err(ExecutorError::NotSupported(
            NotSupported::NotImplemented(
                "engine candidate: a texture buffer an argument buffer holds, off the runtime: \
                 the binder's planned heap holds no texture buffer member"
                    .into(),
            ),
        )));
    }
    let artifact = RefCell::new(artifact::CaseArtifact::default());
    let mut attempt = match &prepared.mapped {
        Mapped::Compute(compute) => super::platform::execute_engine(
            prepared.device,
            &case.case_id,
            compute,
            |target, shape| {
                let compiled = compile_case(worker, target, &library, linked, compute, shape)?;
                artifact
                    .borrow_mut()
                    .record(&compiled, target.device(), shape)?;
                Ok(compiled)
            },
        ),
        Mapped::Render(render) => {
            let companion = companion_source(render);
            super::platform::execute_engine_render(
                prepared.device,
                &case.case_id,
                render,
                |target, shape| {
                    let compiled = compile_render_case(
                        worker,
                        target,
                        &library,
                        companion.as_ref(),
                        linked,
                        render,
                        shape,
                    )?;
                    artifact
                        .borrow_mut()
                        .record(&compiled, target.device(), shape)?;
                    Ok(compiled)
                },
            )
        }
        // A run's preparation refuses a mesh case before it maps (`prepare_case`); only the
        // compile census prepares one (`prepare_compile`).
        Mapped::Mesh(_) => RunAttempt::of(Err(NotSupported::Requirements(
            [ToolingRequirement::MeshRenderPipeline].into(),
        )
        .into())),
    };
    let executed = artifact.into_inner().executed();
    if let Ok(run) = &mut attempt.result {
        run.census.artifact = Some(executed.clone());
    }
    if let Some(opened) = &mut attempt.refused_on {
        opened.artifact = Some(executed);
    }
    attempt
}

/// [`execute_case`] of a case [`prepare_case`] accepted: the device half in a device worker
/// (`run_on_device`), the comparison and the row here.
pub fn execute_prepared(
    root: &Path,
    prepared: PreparedCase,
    metal_environment_id: &str,
    environment_id: &str,
    worker: &CompileWorker,
) -> Result<CandidateObservation, ExecutorError> {
    if metal_environment_id.trim().is_empty() || environment_id.trim().is_empty() {
        return Err("Metal and candidate environment IDs must not be empty".into());
    }
    if is_engine_slot(environment_id) {
        return Err(format!(
            "environment ID {environment_id:?} already names an engine slot; pass the \
             operator's environment ID"
        )
        .into());
    }
    super::validate_host()?;
    let PreparedCase {
        device, checked, ..
    } = prepared;
    let case = &checked.primary.case;
    let store = CorpusStore::new(root);
    // Only the case's Metal shard is read: every row of the case is there.
    let metal = MetalObservation::current(
        store.read_metal_of_case(case)?,
        case,
        metal_environment_id,
        ORACLE_ABI,
    )?;
    // The device half runs in a device worker: a driver fault ends the worker, and comes back as
    // `ExecutorError::DeviceCrash`, never as this process's death.
    let restricted_to = super::requested_restriction();
    // A run on the device as it is under a C3 venue's environment is that venue's evidence, held
    // to its committed profile; a restricted run's slot is no venue's.
    let venue = match restricted_to {
        Some(_) => None,
        None => Venues::load(&Venues::committed_path())?
            .venue_of(environment_id)
            .map(|venue| VenueRun {
                environment_id: environment_id.into(),
                profile: venue.profile.clone(),
                allowed: venue
                    .devices
                    .iter()
                    .map(|device| device.identity.clone())
                    .collect(),
            }),
    };
    let input = DeviceInput {
        root: root.to_path_buf(),
        case: case.clone(),
        device,
        input_sha256: checked.primary.input_sha256.clone(),
        wait_timeout_ms: u64::try_from(wait_timeout().as_millis()).unwrap_or(u64::MAX),
        restricted_to,
        venue,
        source_index: crate::source::source_index(),
    };
    let DeviceOutcome { result, refused_on } = crate::worker::device::execute(worker, &input);
    let run = match result {
        Ok(run) => run,
        // A typed refusal on an opened device is evidence too: stored in the slot the row would
        // have taken, with that device's facts.
        Err(ExecutorError::NotSupported(not_supported)) => {
            if let Some(opened) = refused_on {
                store.upsert_refusal(RefusalObservation {
                    case_id: case.case_id.clone(),
                    air_sha256: case.air_sha256.clone(),
                    input_sha256: input.input_sha256.clone(),
                    translator_fingerprint: opened.translator_fingerprint,
                    environment_id: engine_slot(environment_id, input.restricted_to.as_deref()),
                    environment: opened.environment,
                    executor_abi: ENGINE_EXECUTOR_ABI.into(),
                    refusal: Refusal::of(&not_supported),
                    artifact: opened.artifact,
                    timing: Some(opened.timing),
                })?;
            }
            return Err(not_supported.into());
        }
        Err(error) => return Err(error),
    };
    let slot = engine_slot(environment_id, run.census.restricted_to.as_deref());
    let candidate_output_sha256 = sha256_bytes(&run.output);
    let golden = base64::engine::general_purpose::STANDARD
        .decode(&metal.output_b64)
        .map_err(|error| format!("decode the Metal observation's output: {error}"))?;
    // A case with out-of-bounds targets is judged by their containment property.
    let (status, held_open_by) = if crate::executor_contract::out_of_bounds::matches(
        case,
        &run.comparison,
        &golden,
        &run.output,
        run.census.out_of_bounds.as_ref(),
    ) {
        (CandidateStatus::Match, None)
    } else {
        difference_status(
            &checked,
            candidate_output_sha256 != metal.metal_output_sha256,
            &run.comparison,
            run.census.out_of_bounds.as_ref(),
            || ReviewedList::load(&ReviewedList::committed_path()),
        )?
    };
    let observation = CandidateObservation {
        case_id: case.case_id.clone(),
        air_sha256: case.air_sha256.clone(),
        input_sha256: checked.primary.input_sha256.clone(),
        golden_output_sha256: metal.metal_output_sha256.clone(),
        spv_sha256: run.spv_sha256.clone(),
        translator_fingerprint: run.translator_fingerprint.clone(),
        candidate_output_sha256,
        output_b64: base64::engine::general_purpose::STANDARD.encode(&run.output),
        environment_id: slot.clone(),
        environment: run.environment,
        executor_abi: ENGINE_EXECUTOR_ABI.into(),
        comparison: run.comparison.clone(),
        census: Some(run.census.clone()),
        timing: Some(run.timing),
        status,
        held_open_by,
    };
    // The row contract first, so a refusal (such as a row under an unrequested profiles layer)
    // reports its own reason.
    observation.validate_content()?;
    let dependencies = CandidateDependencies {
        case,
        metal: &metal,
        spv_sha256: run.spv_sha256.as_deref(),
        translator_fingerprint: &run.translator_fingerprint,
        environment_id: &slot,
        executor_abi: ENGINE_EXECUTOR_ABI,
    };
    if !observation.dependency_matches(&dependencies) {
        return Err("constructed candidate observation failed its own dependency check".into());
    }
    store.upsert_candidate(observation.clone())?;
    Ok(observation)
}

/// The status of a run of `checked` whose output does not match Metal's under its class
/// `comparison` (`differs`: its bytes differ; `run`: its out-of-bounds facts), and the answer gap
/// that holds it open: a noted difference where it may be one ([`may_be_noted`]), unless the C3
/// reviewed list (`list`, read only then) holds the case under an answer gap its structure meets
/// ([`AnswerGap::met_by`]), which keeps it a mismatch (the user's decision of 2026-10-06). Which
/// venues' devices have the gap is C3's to judge; here the case's difference is a mismatch on every
/// device, which hides none.
pub fn difference_status(
    checked: &CheckedCase,
    differs: bool,
    comparison: &ComparisonClass,
    run: Option<&OutOfBoundsRun>,
    list: impl FnOnce() -> Result<ReviewedList, String>,
) -> Result<(CandidateStatus, Option<AnswerGap>), String> {
    if !may_be_noted(differs, comparison, run) {
        return Ok((CandidateStatus::Mismatch, None));
    }
    let case_id = &checked.primary.case.case_id;
    Ok(
        match list()?.holding_open(case_id, |gap| gap.met_by(checked)) {
            Some(gap) => (CandidateStatus::Mismatch, Some(gap)),
            None => (CandidateStatus::NotedDifference, None),
        },
    )
}

/// The bytes of `output` from the final buffer contents and the final texture contents, by Metal
/// index, and the final argument-buffer buffer and texture contents, by `(owner, field offset)`.
pub fn select_output(
    output: ComputeOutput,
    buffers: &BTreeMap<u32, Vec<u8>>,
    embedded: &BTreeMap<(u32, u32), Vec<u8>>,
    textures: &BTreeMap<u32, LiteralTexture>,
    embedded_textures: &BTreeMap<(u32, u32), LiteralTexture>,
) -> Result<Vec<u8>, String> {
    match output {
        ComputeOutput::None => Ok(Vec::new()),
        ComputeOutput::Texture {
            binding,
            origin,
            dimensions,
        } => {
            let texture = textures
                .get(&binding)
                .ok_or_else(|| format!("output texture {binding} is not bound"))?;
            resources::select_texture(texture, origin, dimensions)
        }
        ComputeOutput::Buffer {
            binding,
            offset,
            length,
        } => {
            let bytes = buffers
                .get(&binding)
                .ok_or_else(|| format!("output buffer {binding} is not bound"))?;
            select_range(bytes, offset, length, &format!("buffer {binding}"))
        }
        ComputeOutput::ArgumentBufferBuffer {
            owner,
            field_offset,
            offset,
            length,
        } => {
            let what = format!("argument-buffer buffer {owner}+{field_offset}");
            let bytes = embedded
                .get(&(owner, field_offset))
                .ok_or_else(|| format!("output {what} is not bound"))?;
            select_range(bytes, offset, length, &what)
        }
        ComputeOutput::ArgumentBufferTexture {
            owner,
            field_offset,
            origin,
            dimensions,
        } => {
            let texture = embedded_textures
                .get(&(owner, field_offset))
                .ok_or_else(|| {
                    format!("output argument-buffer texture {owner}+{field_offset} is not bound")
                })?;
            resources::select_texture(texture, origin, dimensions)
        }
    }
}

/// `length` bytes at `offset` of `bytes`, the final contents of `what`.
fn select_range(bytes: &[u8], offset: u64, length: u64, what: &str) -> Result<Vec<u8>, String> {
    let start = usize::try_from(offset).map_err(|_| "output offset overflows")?;
    let end = offset
        .checked_add(length)
        .and_then(|end| usize::try_from(end).ok())
        .ok_or("output range overflows")?;
    bytes.get(start..end).map(<[u8]>::to_vec).ok_or_else(|| {
        format!(
            "output range {offset}+{length} exceeds {what} ({} bytes)",
            bytes.len()
        )
    })
}
