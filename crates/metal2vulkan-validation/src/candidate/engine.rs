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
//! visible-function references and tables name, from the modules the Metal oracle links
//! ([`linked_modules`]), and each visible-function table is a buffer of the function's stage
//! holding the product's table words with the handles the compile gave those functions
//! ([`tables`]). Each element of a texture array is the texture at its own Metal slot
//! ([`resources::bound_textures`]), and an argument-buffer buffer of either executor is a used
//! resource whose address its owner holds ([`argument_buffers`]); an argument-buffer texture is a
//! used texture whose resource id its owner holds, a member of the plan's heap, and a command
//! buffer an opaque id in its owner. A tile kernel over render targets compiles through
//! `compile_tile_pipeline` and dispatches over the render area's tiles, its attachments bound as
//! the plan's tile attachments ([`TilePass`]). Nothing about the layout is re-derived here.
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
use crate::c3::{VenueProfile, Venues, VENUE_PROFILE};
use crate::case::indirect_commands::{CommandBufferResource, HostStageInRegion};
use crate::case::out_of_bounds::{OutOfBoundsResource, OutOfBoundsTarget};
use crate::case::{
    AccelerationStructureKind, ArgumentBufferSamplerResource, AuthoredCase,
    ImplicitImageblockCoverage, IntersectionFunctionTableEntry, IntersectionFunctionTableResource,
    OutputSelection, RateMapLiteral, ResourceRole, SamplerResource, Stage, TextureType,
};
use crate::check::admission::{
    admit, AdmittedCase, ExecutionRequest, ExecutorError, NotSupported, VenueMismatch,
};
use crate::check::interface::AirInterface;
use crate::check::CheckedCase;
use crate::companion::{Companion, Spelling};
use crate::coverage::{ExecutedCensus, ExecutionClass, Leg};
use crate::engine::{bounded_detail, Outcome, RowSource};
use crate::executor_contract::out_of_bounds::{self, OutOfBoundsRun, TargetRun};
use crate::executor_contract::stage_inputs;
use crate::gpu::wait::wait_timeout;
use crate::gpu::DeviceType;
use crate::hash::sha256_bytes;
use crate::library_module::ResolvedLinkedFunctions;
use crate::literal::{LiteralResources, LiteralTexture};
use crate::metal::ORACLE_ABI;
use crate::observation::{
    Backend, CandidateDependencies, CandidateObservation, CandidateStatus, CaseTiming,
    MetalObservation, TRANSLATOR_FINGERPRINT,
};
use crate::observation_contract::ComparisonClass;
use crate::refusal::{Refusal, RefusalObservation};
use crate::requirement::Executor;
use crate::store::CorpusStore;
use crate::worker::compile::{
    compile_compute_request, compile_render_request, CompileWorker, NotCompiled, WorkerDevice,
};
use crate::worker::device::{DeviceInput, DeviceOutcome, DeviceRun, OpenedDevice, VenueRun};
use crate::worker::WorkerFailure;
use base64::Engine as _;
use metal2vulkan::bind::{
    BindShape, BindState, BoundAccelerationStructure, BoundBuffer, Dispatch, ExecutionStatus,
    FunctionTableWords, ImageblockSize, MetalInstance, ReplayVariant, StageTable, StatusReport,
};
use metal2vulkan::cache::Hash256;
use metal2vulkan::device::DeviceProfile;
use metal2vulkan::error::{BindError, Error};
use metal2vulkan::library::{FunctionConstantValue, FunctionConstantValues};
use metal2vulkan::metal::{PixelFormat, RenderStages};
use metal2vulkan::pipeline::Stage as ShaderStage;
use metal2vulkan::pipeline::StageInputDescriptor;
use metal2vulkan::pipeline::{ComputePipeline, RenderPipeline};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

pub mod argument_buffers;
pub mod artifact;
pub mod graphics;
pub mod guards;
pub mod imageblock;
pub mod mesh;
#[cfg(test)]
mod public_cases;
pub mod resources;
pub mod runtime_run;
pub mod tables;
#[cfg(test)]
pub(crate) mod tests;
pub mod variants;

use tables::{function_tables, table_buffers, FunctionTable, TableBuffers};

/// The executor ABI of engine candidate observations. Every recorded candidate row, the
/// authored compile census and the C3 records key on it, so a bump makes every recorded row stale.
///
/// Bump it whenever what the executor enables, binds, submits, reads back, or records changes, or
/// how it decides a match. A change no recorded row can observe (a case kind no earlier executor
/// ran, or one it refused) needs no bump; its commit says why. The commit that bumps it says what
/// changed; this comment keeps no history.
pub const ENGINE_EXECUTOR_ABI: &str = "vulkan-engine-v181";

/// One kernel case in the terms the binder takes: what this executor supports of an
/// [`AuthoredCase`] (buffers, device buffer arrays, argument-buffer buffers and textures, command
/// buffers and pipeline states, textures, texture buffers, texture arrays, samplers, visible-function tables,
/// threadgroup lengths, an explicit imageblock's size, stage inputs, function constants, a
/// `dispatchThreads` grid or a tile pass's tiles, and a buffer, argument-buffer buffer, texture,
/// argument-buffer texture, or tile pass render-target output).
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
    /// Each bound buffer holding a rasterization-rate map's parameter data, by Metal buffer index:
    /// its bytes in [`ComputeCase::buffers`] are the product's data of the map, which the executor
    /// binds; the runtime path makes the map with the runtime and copies its data into the buffer,
    /// as the Metal oracle does with Metal's.
    pub rate_maps: BTreeMap<u32, RateMapLiteral>,
    /// Initial bytes of each argument-buffer buffer, by `(owner Metal index, field offset)`:
    /// a buffer the owner's argument buffer points to (loaded pointers). The run writes its guest
    /// address into the owner's bytes at the field offset, the pointer Metal's argument encoder
    /// writes there ([`argument_buffers::owner_bytes`]), and lists it as a used resource
    /// (`useResource`, read and write, as the oracle does), so the resident set reaches it.
    pub argument_buffer_buffers: BTreeMap<(u32, u32), Vec<u8>>,
    /// Each argument-buffer buffer holding a rasterization-rate map's parameter data, by `(owner
    /// Metal index, field offset)`, as [`ComputeCase::rate_maps`] is for a bound buffer.
    pub argument_buffer_rate_maps: BTreeMap<(u32, u32), RateMapLiteral>,
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
    /// Each pipeline state, by `(owner Metal index, field offset)`: the case's token for it, the
    /// word the owner holds at the field ([`argument_buffers::embed_objects`]). The product reads
    /// the word only as an encoder's operand, the pipeline a command records.
    pub objects: BTreeMap<(u32, u32), u64>,
    /// Each command buffer, by `(owner Metal index, field offset)`: its record storage is the
    /// argument-buffer buffer at the key ([`ComputeCase::argument_buffer_buffers`]), initialized
    /// under its descriptor, at its ID's records address rather than its own, and its owner holds
    /// its ID ([`records_ids`]); the run checks its final records
    /// ([`argument_buffers::validate_records`]).
    pub command_buffers: BTreeMap<(u32, u32), CommandBufferResource>,
    /// Each bound texture (initial contents included), by Metal texture index: the case's textures
    /// and each element of its texture arrays at its own slot ([`resources::bound_textures`]).
    pub textures: BTreeMap<u32, LiteralTexture>,
    /// Each bound sampler, by Metal sampler index.
    pub samplers: BTreeMap<u32, SamplerResource>,
    /// Each visible-function table, by Metal buffer index: bound as a buffer of
    /// [`FunctionTable::byte_len`] bytes, whose words the compile's handles decide
    /// ([`CompiledCase::tables`]).
    pub function_tables: BTreeMap<u32, FunctionTable>,
    /// Each acceleration structure, by Metal buffer index: built on the device as the Metal oracle
    /// builds it, and bound as the plan's `WriteSource::AccelerationStructure` names it. An
    /// intersection function table is a buffer of [`ComputeCase::buffers`] holding the product's
    /// table words ([`intersection_table_words`]).
    pub acceleration_structures: BTreeMap<u32, CaseStructure>,
    /// `setThreadgroupMemoryLength:atIndex:`.
    pub threadgroup_lengths: BTreeMap<u32, u32>,
    /// `setImageblockWidth:height:`: the dimensions of the case's explicit imageblock, which the
    /// kernel stages in threadgroup memory; a tile pass's tile size.
    pub imageblock: Option<[u32; 2]>,
    /// The render pass of a tile kernel over render targets: the kernel compiles as a tile function
    /// under its attachments' formats, and [`ComputeCase::dispatch`] covers its render area in
    /// tiles of [`ComputeCase::imageblock`].
    pub tile: Option<TilePass>,
    /// The compute pipeline's stage-input descriptor when the case has stage inputs: the kernel
    /// compiles under it, and the dispatch's stage-in region is [`ComputeCase::stage_in_region`] or
    /// the grid's (`executor_contract::stage_inputs`). Each input's data is in
    /// [`ComputeCase::buffers`] at its slot.
    pub stage_input: Option<StageInputDescriptor>,
    /// The case's authored stage-in region (`Dispatch::stage_in_region`), which a dispatch with
    /// stage inputs sets in place of its grid's; `None` for the grid's, and for an executed
    /// command's child, whose region is its own grid's.
    pub stage_in_region: Option<HostStageInRegion>,
    /// The stride each stage input of a dynamic layout's data is bound with, by its slot
    /// (`stage_inputs::bound_strides`): the stride form's, as the oracle binds it; every other
    /// buffer is bound with the plain form.
    pub attribute_strides: BTreeMap<u32, u32>,
    pub dispatch: Dispatch,
    pub output: ComputeOutput,
    /// The case's out-of-bounds targets: each buffer target is allocated inside a guard of its
    /// reach, and the run records how the compiled module contains each target
    /// (`executor_contract::out_of_bounds::OutOfBoundsRun`).
    pub out_of_bounds: Vec<OutOfBoundsTarget>,
    /// The bound each driver creation of the case's own pipeline is held to
    /// (`budgets::creation_within`): the case's declared one (`AuthoredCase::creation_bound`), or
    /// the shared ceiling. An executed command's pipelines, its templates', keep the ceiling.
    pub creation_bound: Duration,
}

/// The render pass a tile kernel runs in, as the Metal oracle encodes it (`metal::platform::tile`):
/// a pass of `area` (`renderTargetWidth`/`renderTargetHeight`, the dispatch grid's width and
/// height) whose color attachments load and store, single-sampled and unlayered.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TilePass {
    pub area: [u32; 2],
    /// The whole tiles over `area`: the render area the tile function specializes
    /// (`BindState::render_area`), as the runtime's dispatch takes it.
    pub tiles: [u32; 2],
    /// Each color attachment with its initial contents, by color index: a 2D texture of the
    /// attachment's authored format and extent (the render area).
    pub attachments: BTreeMap<u32, LiteralTexture>,
    /// Each of `attachments` over the whole tiles, as the executor binds it: its texels from the
    /// origin and each texel past them the nearest one, clamped to their edge, as Metal's tile
    /// memory holds a loaded attachment's cells past it when the pass begins. The tile function's
    /// cells past `area` are this texture's texels there; a render-target output selects its rows
    /// from the origin, as the oracle selects them from the attachment.
    pub tiled: BTreeMap<u32, LiteralTexture>,
}

impl TilePass {
    /// Each attachment's pixel format by color index: the tile pipeline's
    /// `color_attachment_formats`.
    pub fn formats(&self) -> BTreeMap<u32, PixelFormat> {
        self.attachments
            .iter()
            .map(|(&index, texture)| (index, resources::pixel_format(texture.format)))
            .collect()
    }
}

/// One acceleration structure of a case, as the Metal oracle builds it
/// (`metal::platform::acceleration`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaseStructure {
    /// A primitive structure of one triangle geometry (opaque, duplicate intersection function
    /// calls allowed: Metal's defaults) over `vertices`: tightly packed little-endian `float3`
    /// vertices, three per triangle.
    Primitive { vertices: Vec<u8> },
    /// An instance structure of `instances` instances of the primitive structure of
    /// [`AccelerationStructureKind::INSTANCED_TRIANGLE`], each with the identity transform, the
    /// opaque option, mask `0xFFFFFFFF`, table offset 0, and no user id.
    Instances { instances: u32 },
}

impl CaseStructure {
    /// The Metal instance every instance of [`CaseStructure::Instances`] is.
    pub fn instance() -> MetalInstance {
        let mut instance = MetalInstance::new();
        instance.options = INSTANCE_OPTION_OPAQUE;
        instance.mask = u32::MAX;
        instance
    }
}

/// `MTLAccelerationStructureInstanceOptionOpaque`.
const INSTANCE_OPTION_OPAQUE: u32 = 4;

/// The words of intersection function table `table`: the product's table layout
/// (`FunctionTableWords`), each opaque triangle entry the reserved word the translation accepts.
/// A linked intersection function is not implemented: the product refuses a stage that may link
/// one.
pub fn intersection_table_words(
    table: &IntersectionFunctionTableResource,
) -> Result<Vec<u8>, String> {
    let mut words = FunctionTableWords::new(table.size);
    for entry in &table.entries {
        match entry {
            IntersectionFunctionTableEntry::OpaqueTriangle { index, .. } => {
                if !words.set_opaque_triangle(*index) {
                    return Err(format!(
                        "intersection function table {} entry {index} is past its size {}",
                        table.binding, table.size
                    ));
                }
            }
            IntersectionFunctionTableEntry::Linked { .. } => {
                return Err("linked intersection functions".into())
            }
        }
    }
    Ok(words.words().iter().flat_map(|w| w.to_le_bytes()).collect())
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
    /// A region of the tile pass's color attachment `index`, tightly packed rows as the Metal
    /// oracle reads a render target.
    RenderTarget {
        index: u32,
        origin: [u32; 2],
        dimensions: [u32; 2],
    },
}

/// What one engine run produced on a device.
#[derive(Clone, Debug)]
pub struct ComputeRun {
    pub output: Vec<u8>,
    /// SHA-256 of the pipeline's SPIR-V bytes (before specialization).
    pub spv_sha256: String,
    /// [`engine_fingerprint`] of the profile the case compiled for.
    pub translator_fingerprint: String,
    pub environment: serde_json::Value,
    pub status: StatusReport,
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
/// (`executor_contract::case_requirements`: argument-buffer intersection-function tables and linked
/// intersection functions) are `NotSupported::Requirements`; it runs buffers, device buffer arrays,
/// argument-buffer buffers and textures, command buffers and the indirect compute commands they
/// record, pipeline states (each the case's token in its owner), textures (texture buffers
/// included), texture arrays, samplers, visible-function tables, acceleration structures,
/// intersection function tables of opaque triangle entries, threadgroup lengths, an explicit
/// imageblock, stage inputs, and a tile kernel's render pass ([`TilePass`]), and a case whose
/// literals do not map onto them is `NotSupported::NotImplemented` (graphics stages are
/// `candidate::engine::graphics`). `interface` is the checked entry's AIR-layer interface,
/// whose stage inputs take the slots the Metal oracle binds their data at
/// (`executor_contract::stage_inputs`). A case has no indirect dispatch to map: its dispatch is a
/// `dispatchThreads` grid, and indirect work is authored only as indirect command buffers.
pub fn compute_case(
    case: &AuthoredCase,
    interface: &AirInterface,
) -> Result<ComputeCase, NotSupported> {
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
            OutputSelection::RenderTarget {
                index,
                origin,
                dimensions,
            } => ComputeOutput::RenderTarget {
                index,
                origin,
                dimensions,
            },
            _ => return Err(gap(
                "an output other than a buffer range, an argument-buffer buffer range, a region \
                 of a texture, a texture-array element, or an argument-buffer texture, or a tile \
                 pass's render target",
            )),
        };
    let resources =
        LiteralResources::prepare(case).map_err(|error| gap(&format!("resources: {error}")))?;
    let tile = tile_pass(case, &resources)?;
    if matches!(output, ComputeOutput::RenderTarget { .. }) && tile.is_none() {
        return Err(gap(
            "a render-target output of a kernel that runs in no tile pass",
        ));
    }
    let constants = function_constants(&resources)?;
    let rate_maps = resources
        .buffers
        .iter()
        .filter_map(|buffer| Some((buffer.binding, buffer.rasterization_rate_map.clone()?)))
        .collect();
    let argument_buffer_rate_maps = resources
        .argument_buffer_buffers
        .iter()
        .filter_map(|buffer| {
            let map = buffer.rasterization_rate_map.clone()?;
            Some(((buffer.buffer_binding, buffer.field_offset), map))
        })
        .collect();
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
    let mut argument_buffer_buffers =
        argument_buffers::embed(&mut buffers, &resources.argument_buffer_buffers)
            .map_err(|error| gap(&error))?;
    let argument_buffer_textures =
        argument_buffers::embed_textures(&mut buffers, &resources.argument_buffer_textures)
            .map_err(|error| gap(&error))?;
    let argument_buffer_samplers =
        argument_buffers::embed_samplers(&mut buffers, &case.argument_buffer_samplers)
            .map_err(|error| gap(&error))?;
    let objects = argument_buffers::embed_objects(
        &mut buffers,
        &mut argument_buffer_buffers,
        &case.command_buffers,
        &case.pipeline_states,
    )
    .map_err(|error| gap(&error))?;
    let textures = resources::bound_textures(&resources.textures, &resources.texture_arrays)
        .map_err(|error| gap(&error))?;
    let mut samplers = BTreeMap::new();
    for sampler in &case.samplers {
        if samplers.insert(sampler.binding, sampler.clone()).is_some() {
            return Err(gap(&format!("sampler {} bound twice", sampler.binding)));
        }
    }
    for table in &case.intersection_function_tables {
        let words = intersection_table_words(table).map_err(|error| gap(&error))?;
        if buffers.insert(table.binding, words).is_some() {
            return Err(gap(&format!(
                "buffer {} is also an intersection function table",
                table.binding
            )));
        }
    }
    let mut acceleration_structures = BTreeMap::new();
    for structure in &case.acceleration_structures {
        let mapped = match structure.kind {
            AccelerationStructureKind::Primitive => CaseStructure::Primitive {
                vertices: base64::engine::general_purpose::STANDARD
                    .decode(
                        structure
                            .primitive_triangles_b64
                            .as_deref()
                            .unwrap_or_default(),
                    )
                    .map_err(|error| gap(&format!("structure {}: {error}", structure.binding)))?,
            },
            AccelerationStructureKind::Instance => CaseStructure::Instances {
                instances: u32::try_from(structure.child_references.len())
                    .map_err(|_| gap(&format!("structure {} instances", structure.binding)))?,
            },
        };
        if buffers.contains_key(&structure.binding)
            || acceleration_structures
                .insert(structure.binding, mapped)
                .is_some()
        {
            return Err(gap(&format!(
                "acceleration structure {} bound twice",
                structure.binding
            )));
        }
    }
    let function_tables = function_tables(&case.visible_function_tables)?;
    if let Some(binding) = function_tables
        .keys()
        .find(|binding| buffers.contains_key(binding))
    {
        return Err(gap(&format!(
            "buffer {binding} is also a visible function table"
        )));
    }
    let mut threadgroup_lengths = BTreeMap::new();
    for memory in &case.threadgroup_memory {
        threadgroup_lengths.insert(memory.binding, memory.length);
    }
    // Each stage input's data is its own buffer, at the slot the oracle binds it at.
    let fetches = stage_inputs::authored(&case.kernel_stage_inputs, interface)
        .map_err(|error| gap(&format!("stage inputs: {error}")))?;
    if tile.is_some() && !fetches.is_empty() {
        return Err(gap(
            "stage inputs of a tile function (a tile pipeline has no stage-input descriptor)",
        ));
    }
    for fetch in &fetches {
        let bytes = resources
            .kernel_stage_inputs
            .iter()
            .find(|input| input.location == fetch.location)
            .map(|input| input.bytes.clone())
            .ok_or_else(|| gap(&format!("stage input {} has no data", fetch.location)))?;
        if buffers.insert(fetch.slot, bytes).is_some() || function_tables.contains_key(&fetch.slot)
        {
            return Err(gap(&format!(
                "stage input {}'s buffer {} is bound twice",
                fetch.location, fetch.slot
            )));
        }
    }
    Ok(ComputeCase {
        entry: case.entry.clone(),
        constants,
        buffers,
        rate_maps,
        argument_buffer_buffers,
        argument_buffer_rate_maps,
        argument_buffer_textures,
        argument_buffer_samplers,
        objects: objects.pipeline_tokens,
        command_buffers: objects.command_buffers,
        textures,
        samplers,
        function_tables,
        acceleration_structures,
        threadgroup_lengths,
        imageblock: case.imageblock.as_ref().map(|i| i.dimensions),
        stage_input: stage_inputs::descriptor(&fetches),
        stage_in_region: dispatch.stage_in_region,
        attribute_strides: stage_inputs::bound_strides(&fetches),
        dispatch: match &tile {
            // One threadgroup per tile of the render area (`dispatchThreadsPerTile:`).
            Some(pass) => {
                let [tile_width, tile_height] = case
                    .imageblock
                    .as_ref()
                    .map(|imageblock| imageblock.dimensions)
                    .ok_or_else(|| gap("a tile pass without a tile size"))?;
                if tile_width == 0 || tile_height == 0 {
                    return Err(gap("a tile pass of an empty tile"));
                }
                Dispatch::Threadgroups {
                    groups: [
                        pass.area[0].div_ceil(tile_width),
                        pass.area[1].div_ceil(tile_height),
                        1,
                    ],
                    threads_per_group: dispatch.threads_per_threadgroup,
                }
            }
            // `dispatch.grid` is in threads: the Metal oracle calls
            // `dispatchThreads:threadsPerThreadgroup:`.
            None => Dispatch::Threads {
                threads: dispatch.grid,
                threads_per_group: dispatch.threads_per_threadgroup,
            },
        },
        tile,
        output,
        out_of_bounds: case
            .out_of_bounds
            .as_ref()
            .map(|declaration| declaration.targets.clone())
            .unwrap_or_default(),
        creation_bound: case.pipeline_creation_bound(),
    })
}

/// The render pass of `case` when it is a tile kernel (its implicit imageblock's
/// `implicit_coverage`), as the Metal oracle encodes it (`metal::platform::tile`): a single-sample,
/// unlayered pass of the dispatch grid's width and height, each render target a color attachment
/// that loads and stores, of its authored format, extent, and contents (`resources`, the case's
/// prepared literals). Render targets of another kernel are not implemented, and neither is a grid
/// of more than one plane (the oracle's pass has none).
fn tile_pass(
    case: &AuthoredCase,
    resources: &LiteralResources,
) -> Result<Option<TilePass>, NotSupported> {
    let gap = |what: &str| NotSupported::NotImplemented(format!("engine candidate: {what}"));
    let coverage = case
        .imageblock
        .as_ref()
        .and_then(|imageblock| imageblock.implicit_coverage);
    match coverage {
        None if case.render_targets.is_empty() => return Ok(None),
        None => return Err(gap("render targets of a kernel that is no tile function")),
        // The oracle's coverage prepass writes no color, so each attachment's authored contents
        // are its initial pixels.
        Some(ImplicitImageblockCoverage::FullSingleSample) => {}
    }
    let grid = case
        .dispatch
        .as_ref()
        .ok_or_else(|| gap("a tile kernel without a dispatch"))?
        .grid;
    if grid[2] != 1 {
        return Err(gap("a tile pass of a grid of more than one plane"));
    }
    let area = [grid[0], grid[1]];
    let tile = case
        .imageblock
        .as_ref()
        .map(|imageblock| imageblock.dimensions)
        .filter(|&[width, height]| width > 0 && height > 0)
        .ok_or_else(|| gap("a tile pass of no tile size"))?;
    if area.contains(&0) {
        return Err(gap("a tile pass of no pixels"));
    }
    let tiles = [0, 1].map(|axis| area[axis].div_ceil(tile[axis]).saturating_mul(tile[axis]));
    let mut attachments = BTreeMap::new();
    let mut tiled = BTreeMap::new();
    for target in &resources.render_targets {
        if target.dimensions != area {
            return Err(gap(&format!(
                "render target {} of {:?} in a pass of {area:?}",
                target.index, target.dimensions
            )));
        }
        let texture = LiteralTexture {
            binding: target.index,
            role: ResourceRole::InOut,
            texture_type: TextureType::D2,
            format: target.format,
            dimensions: [area[0], area[1], 1],
            sample_count: 1,
            bytes: target.bytes.clone(),
            sparse: None,
            mip_levels: 1,
        };
        let over_tiles = LiteralTexture {
            dimensions: [tiles[0], tiles[1], 1],
            bytes: clamped_to_edge(&target.bytes, target.format.bytes_per_pixel(), area, tiles)
                .ok_or_else(|| gap(&format!("render target {}'s bytes", target.index)))?,
            ..texture.clone()
        };
        tiled.insert(target.index, over_tiles);
        if attachments.insert(target.index, texture).is_some() {
            return Err(gap(&format!("render target {} bound twice", target.index)));
        }
    }
    Ok(Some(TilePass {
        area,
        tiles,
        attachments,
        tiled,
    }))
}

/// `bytes`, the rows of `from` texels of `texel` bytes, as an image of `to` texels (each axis at
/// least `from`'s): each texel past `from` the nearest one inside, clamped to its edge. `None`
/// where `bytes` is not that image.
fn clamped_to_edge(bytes: &[u8], texel: usize, from: [u32; 2], to: [u32; 2]) -> Option<Vec<u8>> {
    let [width, height] = from.map(|n| n as usize);
    if width == 0 || height == 0 || bytes.len() != width * height * texel {
        return None;
    }
    let [across, down] = to.map(|n| n as usize);
    let mut out = Vec::with_capacity(across * down * texel);
    for y in 0..down {
        let row = y.min(height - 1) * width;
        for x in 0..across {
            let at = (row + x.min(width - 1)) * texel;
            out.extend_from_slice(bytes.get(at..at + texel)?);
        }
    }
    Some(out)
}

/// Refuse the families of `case`'s own literals that stop this executor
/// (`executor_contract::case_requirements`): argument-buffer intersection-function tables, linked
/// intersection functions, and the render executor's own (device buffer arrays, acceleration
/// structures and intersection function tables, a vertex case's depth/stencil attachment, a
/// rasterization-disabled vertex case's render targets; it draws a tessellation case's patches, and
/// runs a fragment function's explicit imageblock). The checker computed them with the source's
/// families ([`crate::check::CheckedShader::requirements`]), which [`prepare_case`] refuses; this
/// check keeps a case mapped without the checker to the same rule. (The compute executor binds
/// device buffer arrays, runs kernel stage inputs, [`compute_case`], and builds acceleration
/// structures and binds intersection function tables of opaque triangle entries. Both executors
/// bind argument-buffer buffers, [`argument_buffers`], and texture arrays, each element the texture
/// at its slot, [`resources::bound_textures`], argument-buffer textures and command buffers, each
/// command buffer its record storage, and pipeline states, each its token in its owner; the compute
/// executor executes a kernel case's commands. Visible-function references are no resource: the
/// case's function links them, [`linked_modules`]. Both executors bind visible-function tables,
/// [`tables`].)
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
/// view base alignment (`BindState::view_base_alignment_of`), which every bind offset the
/// executor binds, 0, is a multiple of in turn. Each of its states tells it apart at every buffer
/// slot ([`views_told_apart`]), so a uniform view's range starts at its bind offset in its plan
/// (`SpecKey::RangeAtBindOffset`) on every device, as in the runtime's plan of the same binds: the
/// runtime states its device's own alignment at the slots its pipeline reads through a uniform
/// view, and the plans of the two are the same, so the runtime path's artifact digest is the
/// executor's and the authored compile census's.
pub const VIEW_BASE_ALIGNMENT: u32 = 256;

/// `state` telling [`VIEW_BASE_ALIGNMENT`] apart at every buffer slot of each of its stages.
pub fn views_told_apart(mut state: BindState) -> BindState {
    state.view_base_alignment = VIEW_BASE_ALIGNMENT;
    for table in state.stages.values_mut() {
        table.at_bind_views = u64::MAX;
    }
    state
}

/// The `BindState` of `case`: each buffer bound at offset 0 with its length, its guest address,
/// and its host device address when the device has one (`addresses` by Metal index), each
/// visible-function table the same way as a buffer of its table words' length (`bind::table`), each
/// texture and sampler ([`resources`]), the threadgroup lengths, the imageblock size (a tile pass's
/// tile size), a tile pass's render area, the stage-in region of a case with stage inputs, each
/// instance structure's instance count, each argument-buffer buffer as a used buffer (`embedded` by
/// `(owner, field offset)`; its consumer token is its position in
/// [`ComputeCase::argument_buffer_buffers`]), and each argument-buffer texture as a used texture
/// with its resource id (its token follows the buffers', [`argument_buffers::texture_tokens`]),
/// each made resident for reading and writing, as the Metal oracle's `useResource` does, and each
/// argument-buffer sampler as an argument sampler with its id
/// ([`argument_buffers::argument_samplers`]); telling the executor's view base alignment apart
/// ([`views_told_apart`]).
pub fn bind_state(
    case: &ComputeCase,
    addresses: &BTreeMap<u32, BufferAddress>,
    embedded: &BTreeMap<(u32, u32), BufferAddress>,
) -> Result<BindState, String> {
    let mut table = StageTable::default();
    let lengths = case
        .buffers
        .iter()
        .map(|(&index, bytes)| (index, bytes.len() as u64))
        .chain(
            case.function_tables
                .iter()
                .map(|(&index, function_table)| (index, function_table.byte_len())),
        );
    for (index, length) in lengths {
        let address = addresses.get(&index).copied().unwrap_or_default();
        let bound = BoundBuffer::Buffer {
            length,
            offset: 0,
            gpu_address: address.guest,
            device_address: address.host,
            attribute_stride: case.attribute_strides.get(&index).copied(),
        };
        if table.buffers.insert(index, bound).is_some() {
            return Err(format!("buffer {index} is also a visible function table"));
        }
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
    // The instance count of each instance structure, as the oracle builds it: what a function's
    // `get_instance_count()` reads.
    for (&index, structure) in &case.acceleration_structures {
        if let CaseStructure::Instances { instances } = structure {
            table
                .acceleration_structures
                .insert(index, BoundAccelerationStructure::new(*instances));
        }
    }
    let mut state = BindState::default();
    state.stages.insert(ShaderStage::Kernel, table);
    if let Some([width, height]) = case.imageblock {
        state.imageblock_size = ImageblockSize::new(width, height);
    }
    // The pass the oracle encodes the tile dispatch in: unlayered, so its render target array
    // length stays the default.
    if let Some(pass) = &case.tile {
        state.render_area = pass.tiles;
    }
    // The oracle sets the stage-in region of a case with stage inputs, the case's or its
    // grid's, and no other.
    if let (Some(_), Dispatch::Threads { threads, .. }) = (&case.stage_input, case.dispatch) {
        state.stage_in_region = Some(stage_inputs::region(case.stage_in_region, threads));
    }
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
    Ok(views_told_apart(state))
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

/// The `gpuResourceID` of each command buffer of `case`, by its key, in key order
/// ([`argument_buffers::records_ids`]): the word its owner holds, which the product resolves to its
/// record storage.
pub fn records_ids(case: &ComputeCase) -> Result<BTreeMap<(u32, u32), u64>, String> {
    argument_buffers::records_ids(case.command_buffers.keys().copied())
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
/// observation, a failure; a read of an argument the function constants removed
/// ([`absent_argument_refusal`]) is the translation's typed refusal.
pub fn report_verdict(report: StatusReport) -> Result<StatusReport, ExecutorError> {
    if report.absent_argument_read {
        return Err(absent_argument_refusal());
    }
    if report.replay_fault {
        return Err(ExecutorError::Failed(format!(
            "the device replay dispatched nothing for command(s) it refused ({:#x}); the output \
             is not evidence",
            report.replay_faults.0
        )));
    }
    if report.overflow {
        return Err(ExecutorError::Failed(
            "the indirect clamp reduced a group count; the output is not evidence".into(),
        ));
    }
    if report.resident_miss || report.resident_misses != 0 {
        return Err(ExecutorError::Failed(format!(
            "{} access(es) missed the resident set; the output is not evidence",
            report.resident_misses
        )));
    }
    Ok(report)
}

/// The refusal of a run in which a function read an argument its function constants removed
/// (`StatusBits::ABSENT_ARGUMENT_READ`, which the runtime's execution report states): Metal gives
/// the read no value, so the translation took zero for it, and nothing the run wrote is evidence.
/// It is the product's typed refusal of the case, `FunctionConstantArgument`, not representable: no
/// Vulkan run gives Metal's undefined answer.
pub fn absent_argument_refusal() -> ExecutorError {
    let error =
        metal2vulkan::error::Error::from(metal2vulkan::error::Unsupported::not_representable(
            metal2vulkan::error::Feature::FunctionConstantArgument,
        ));
    refusal_at(
        Outcome::from(&error),
        "a function read an argument its function constants removed, which Metal gives no \
         value: the run read zero"
            .into(),
        false,
    )
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

/// The suffix of an engine store slot.
const ENGINE_SLOT_SUFFIX: &str = "@engine";

/// The suffix of a robustness-off engine slot.
const ROBUSTNESS_OFF_SLOT_SUFFIX: &str = "+robustness-off";

/// The store slot of an engine observation for the operator's `environment_id` in the stated
/// leg: its own slot, so an engine row never replaces the old engine's row of the same
/// environment.
pub fn engine_environment_id(environment_id: &str) -> String {
    engine_slot(environment_id, None, Leg::Stated)
}

/// The store slot of an engine observation for the operator's `environment_id` on a device
/// `VK_LAYER_KHRONOS_profiles` restricted to the committed profile `restricted_to`
/// (`ENV@engine@P`), or on the device as it is ([`engine_environment_id`]), in robustness
/// leg `leg` (`+robustness-off` for that leg): a row run under a profile never replaces a native
/// row, nor one run under another profile, nor one of the other leg.
pub fn engine_slot(environment_id: &str, restricted_to: Option<&str>, leg: Leg) -> String {
    let leg = match leg {
        Leg::Stated => "",
        Leg::RobustnessOff => ROBUSTNESS_OFF_SLOT_SUFFIX,
    };
    match restricted_to {
        None => format!("{environment_id}{ENGINE_SLOT_SUFFIX}{leg}"),
        Some(profile) => format!("{environment_id}{ENGINE_SLOT_SUFFIX}@{profile}{leg}"),
    }
}

/// Whether a candidate row's `environment_id` is an engine slot ([`engine_slot`], native or
/// restricted, in either leg): the row kind the evidence census judges by the new executor's
/// dependency rule. Neither executor accepts an operator environment ID that already names a
/// engine slot, so the slot alone decides the kind.
pub fn is_engine_slot(environment_id: &str) -> bool {
    match environment_id.rsplit_once(ENGINE_SLOT_SUFFIX) {
        Some((_, rest)) => {
            let restricted = rest
                .strip_suffix(ROBUSTNESS_OFF_SLOT_SUFFIX)
                .unwrap_or(rest);
            restricted.is_empty()
                || restricted
                    .strip_prefix('@')
                    .is_some_and(|profile| !profile.is_empty() && !profile.contains(['@', '+']))
        }
        None => false,
    }
}

/// Whether a candidate row's `environment_id` is an engine slot of the robustness-off leg
/// ([`engine_slot`] with [`Leg::RobustnessOff`]): the slot a row whose census ran in that leg
/// is stored in, and no other row.
pub fn is_robustness_off_slot(environment_id: &str) -> bool {
    is_engine_slot(environment_id) && environment_id.ends_with(ROBUSTNESS_OFF_SLOT_SUFFIX)
}

/// How a compiled module contains each of `targets` (`out_of_bounds::containment`: its slot's
/// form and `profile`'s encodings) on a device that executes as `class`, or `None` for a case
/// without targets: the run's out-of-bounds facts, before what its guards count
/// (`TargetRun::escaped`, `TargetRun::past_end`). A target whose form the device leaves
/// uncontained under that encoding refuses the run, so it is decided before anything is submitted:
/// by the executor's own context before it executes, and by the runtime path before it commits
/// (`runtime_run`).
pub(crate) fn out_of_bounds_run(
    class: Option<ExecutionClass>,
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
                escape_cause: None,
                read_cause: None,
            })
        })
        .collect::<Result<Vec<_>, ExecutorError>>()?;
    let uncontained = out_of_bounds::uncontained(&targets, class);
    if !uncontained.is_empty() {
        return Err(NotSupported::Uncontained(uncontained).into());
    }
    Ok(Some(OutOfBoundsRun { targets }))
}

/// [`out_of_bounds_run`] for a render case's draw: how `pipeline`'s module that runs the Metal
/// vertex function (`graphics::vertex_module`) contains each of `case`'s targets, its attributes
/// fetched from `case`'s vertex buffers. The one decision the executor's own context and the
/// runtime path (`runtime_run`) take for a draw.
pub(crate) fn draw_out_of_bounds_run(
    class: Option<ExecutionClass>,
    profile: &DeviceProfile,
    pipeline: &RenderPipeline,
    case: &graphics::RenderCase,
) -> Result<Option<OutOfBoundsRun>, ExecutorError> {
    out_of_bounds_run(
        class,
        profile,
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
            strategies: &graphics::vertex_module(pipeline)?.strategies,
        },
        &case.out_of_bounds,
    )
}

/// Whether a draw that read `padding` index positions past where Metal's fetch stops reading its
/// index buffer from a zero-padded copy of its indices contains them as `run`'s index buffer
/// target states: the padded copy, of at least one such position. The executor's context and the
/// runtime path (the draw's execution record) both hold the draw to it; a run without that target
/// agrees with any draw.
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
/// a tile kernel's as a tile pipeline under its attachments' formats, linking the functions of
/// `linked`, and its libraries in [`PipelineRequest::libraries`] order: `library`, then each of
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
        stage_input: case.stage_input.clone(),
        tile: case.tile.as_ref().map(TilePass::formats),
        replay: false,
    };
    (request, with_linked(&[library], linked))
}

/// The verdict of a device replay whose preprocess refused commands for `faults`, `what` naming the
/// command buffer. The product's typed limits refuse the row (`not_supported{bind}`): no registered
/// variant has the command's pipeline and lengths and the threadgroup size of one of its regions
/// (its own, or an edge's), the command has more regions than the execution records, its grid
/// exceeds `2^32 - 1` threads, or a merged draw replay's commands draw `2^62` vertices or more,
/// where the host replay of v72 ran the command. Every other reason
/// fails it, as the host replay failed: the records passed the host contract's decode before the
/// replay (`validate_records`: no encoding fault, malformed record, draw, missing pipeline, or
/// state set while inherited), every allocation of the run is resident and the execution's capacity
/// is the storage's, so an encoding, record, action, pipeline, buffers, or range fault is a case
/// the host replay fails too (a dynamic stride) or a preprocess that disagrees with the host
/// contract.
pub fn replay_fault_verdict(what: &str, faults: metal2vulkan::bind::ReplayFaults) -> ExecutorError {
    use metal2vulkan::bind::ReplayFaults as F;
    let names = [
        (F::ENCODING, "encoding"),
        (F::ACTION, "a draw"),
        (F::PIPELINE, "pipeline"),
        (F::BUFFERS, "buffers"),
        (F::VARIANT, "no registered variant"),
        (F::GROUP_COUNT, "a grid past 2^32 - 1 threads"),
        (F::RECORD, "a malformed record"),
        (F::RANGE, "a command past the storage's capacity"),
        (F::REGIONS, "more regions than the execution records"),
        (F::BARRIER, "a set barrier between draws"),
        (F::VERTICES, "a merged draw of 2^62 vertices or more"),
    ];
    let named: Vec<&str> = names
        .iter()
        .filter(|(bit, _)| faults.contains(*bit))
        .map(|(_, name)| *name)
        .collect();
    let reasons = format!("{:#x}: {}", faults.0, named.join(", "));
    let limits = F::VARIANT.0 | F::GROUP_COUNT.0 | F::REGIONS.0 | F::VERTICES.0;
    if faults.0 != 0 && faults.0 & !limits == 0 {
        NotSupported::Bind(format!(
            "{what}: the device replay dispatched nothing for command(s) the registered variants \
             do not run ({reasons})"
        ))
        .into()
    } else {
        ExecutorError::Failed(format!(
            "{what}: the device replay refused command(s) the host contract admits or a run \
             cannot bind ({reasons}); the output is not evidence"
        ))
    }
}

/// The dispatch of a device-replay variant for a threadgroup of `threads_per_group` threads:
/// indirect, its group counts the replay's arguments, so it plans on the pipeline variant every
/// such dispatch runs on (`dispatch_variant`). The arguments' buffer is the replay's scratch, which
/// no plan names.
pub fn replay_dispatch(threads_per_group: [u32; 3]) -> Dispatch {
    Dispatch::ThreadgroupsIndirect {
        indirect: metal2vulkan::bind::IndirectRef::new(0, 0),
        threads_per_group,
    }
}

/// The compile request for the device-replay variant of the template `case` (its dispatch a
/// [`replay_dispatch`]) for the bind shape `shape`, which binds no kernel buffer:
/// [`compute_request`] with `replay` set, so the kernel compiles with
/// `LoweringInputs::replay_bindings`. A replayed template has no stage input and no tile pass
/// ([`replay_templates`]).
pub fn replay_request<'a>(
    library: &RowSource<'a>,
    linked: &[LinkedModule],
    case: &ComputeCase,
    shape: &BindShape,
) -> (PipelineRequest, Vec<RowSource<'a>>) {
    let request = PipelineRequest::BoundCompute {
        function: linking(&case.entry, &case.constants, linked),
        shape: shape.clone(),
        dispatch: Some(case.dispatch),
        stage_input: None,
        tile: None,
        replay: true,
    };
    (request, with_linked(&[library], linked))
}

/// One allocation of a compute run that a replayed command may bind or reach through a pointer, by
/// what it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunAllocation {
    /// The case buffer at this Metal index.
    Buffer(u32),
    /// The buffer of the visible-function table at this Metal index.
    FunctionTable(u32),
    /// The argument-buffer buffer at this key (its owner's index and field offset).
    ArgumentBuffer((u32, u32)),
}

/// Every allocation of `case`'s run that a replayed command may bind or reach, in the order the
/// run lists them, each its token in the device replay's resident set ([`replay_resident`]):
/// each case buffer, each visible-function table's buffer, then each argument-buffer buffer,
/// with the length the guest sees (not the padding). The compute executor's run lists its
/// allocations by it, and the authored compile census its variants' resident set.
pub fn run_allocations(case: &ComputeCase) -> Vec<(RunAllocation, u64)> {
    case.buffers
        .iter()
        .map(|(&index, bytes)| (RunAllocation::Buffer(index), bytes.len() as u64))
        .chain(
            case.function_tables
                .iter()
                .map(|(&index, table)| (RunAllocation::FunctionTable(index), table.byte_len())),
        )
        .chain(
            case.argument_buffer_buffers
                .iter()
                .map(|(&key, bytes)| (RunAllocation::ArgumentBuffer(key), bytes.len() as u64)),
        )
        .collect()
}

/// The bind state the device replay's variants start from: each of `allocations` (its address and
/// the length the guest sees; its token its position) resident for reading and writing, as the
/// Metal oracle's `useResource` makes every allocation of the run resident.
pub fn replay_resident(
    allocations: impl IntoIterator<Item = (BufferAddress, u64)>,
) -> Result<BindState, String> {
    let mut resident = BindState::default();
    for (token, (address, length)) in allocations.into_iter().enumerate() {
        resident.used_resources.push(argument_buffers::used_buffer(
            token,
            length as usize,
            address,
            RenderStages::empty(),
        )?);
    }
    Ok(resident)
}

/// The bind state the registered replay variant `variant` is compiled and planned for: `resident`
/// ([`replay_resident`]) with the variant's threadgroup lengths and no kernel buffer.
pub fn replay_variant_state(resident: &BindState, variant: &ReplayVariant) -> BindState {
    let mut state = resident.clone();
    let mut table = StageTable::default();
    for (slot, &length) in variant.threadgroup_memory_lengths.iter().enumerate() {
        if length != 0 {
            table.threadgroup_lengths.insert(slot as u32, length);
        }
    }
    state.stages.insert(ShaderStage::Kernel, table);
    state
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
                form: variants::DrawForm::Bound,
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

/// A case's compiled pipeline and the bytes of its visible-function tables' buffers
/// ([`tables::table_buffers`] of the handles the compile gave the linked functions), which the run
/// binds at their stages and Metal buffer indices.
#[derive(Clone, Debug)]
pub struct CompiledCase<P> {
    pub pipeline: P,
    pub tables: TableBuffers,
}

/// Compile `case`'s kernel from `library` for `device` in a killable compile worker, as
/// [`compute_request`] asks, and encode its tables with the handles the worker returned: the
/// product's typed refusal is [`outcome_refusal`]'s, and a stop at the worker boundary (the
/// deadline, the memory limit, a crashed worker, an artifact that does not load for `device`,
/// handles that do not answer the request) fails the run.
pub fn compile_case(
    worker: &CompileWorker,
    device: &WorkerDevice,
    library: &RowSource<'_>,
    linked: &[LinkedModule],
    case: &ComputeCase,
    shape: &BindShape,
) -> Result<CompiledCase<ComputePipeline>, ExecutorError> {
    let (request, libraries) = compute_request(library, linked, case, shape);
    compile_request(worker, device, &libraries, &request, case, linked)
}

/// Compile `request` (of `case`'s kernel, linking `linked`) from `libraries` in `worker`, with
/// its visible-function tables' bytes.
fn compile_request(
    worker: &CompileWorker,
    device: &WorkerDevice,
    libraries: &[RowSource<'_>],
    request: &PipelineRequest,
    case: &ComputeCase,
    linked: &[LinkedModule],
) -> Result<CompiledCase<ComputePipeline>, ExecutorError> {
    compute_attempt(worker, device, libraries, request, case, linked)
        .result
        .map_err(ExecutorError::from)
}

/// Why a compile of one of a case's pipelines in a compile worker gave no pipeline: a stop at the
/// worker boundary, the worker's refusal, or table handles that do not encode. The executor reads
/// each as its [`ExecutorError`] ([`From`]: `worker_stopped`, `not_compiled`); the authored compile
/// census keeps them typed.
#[derive(Debug)]
pub enum CompileFailure {
    /// The worker boundary stopped the compile (the deadline, the memory limit, a crashed
    /// worker, an artifact that does not load for the device, handles that do not answer the
    /// request).
    Stopped(WorkerFailure),
    /// The worker compiled nothing: the product's typed outcome, or inputs the tool does not
    /// build.
    NotCompiled(NotCompiled),
    /// The handles the compile returned do not encode the case's visible-function tables.
    Tables(ExecutorError),
    /// The compile's digest could not be made ([`artifact::CaseArtifact::record`]).
    Artifact(ExecutorError),
}

impl From<CompileFailure> for ExecutorError {
    fn from(failure: CompileFailure) -> ExecutorError {
        match failure {
            CompileFailure::Stopped(failure) => worker_stopped(failure),
            CompileFailure::NotCompiled(refusal) => not_compiled(refusal),
            CompileFailure::Tables(error) | CompileFailure::Artifact(error) => error,
        }
    }
}

/// One compile of a case's pipeline in a compile worker: the pipeline with its tables, or why
/// there is none, and what the parent measured (`None` when no worker was spawned).
pub struct CompileAttempt<P> {
    pub result: Result<CompiledCase<P>, CompileFailure>,
    pub measured: Option<crate::worker::Measured>,
}

/// [`compile_request`] with its failure typed and its measure ([`CompileAttempt`]).
fn compute_attempt(
    worker: &CompileWorker,
    device: &WorkerDevice,
    libraries: &[RowSource<'_>],
    request: &PipelineRequest,
    case: &ComputeCase,
    linked: &[LinkedModule],
) -> CompileAttempt<ComputePipeline> {
    match compile_compute_request(worker, device, libraries, request) {
        Err(failure) => CompileAttempt {
            measured: failure.measured(),
            result: Err(CompileFailure::Stopped(failure)),
        },
        Ok(compiled) => CompileAttempt {
            measured: Some(compiled.measured),
            result: compiled
                .result
                .map_err(CompileFailure::NotCompiled)
                .and_then(|compiled| {
                    Ok(CompiledCase {
                        tables: compute_tables(case, linked, &compiled.handles)
                            .map_err(CompileFailure::Tables)?,
                        pipeline: compiled.pipeline,
                    })
                }),
        },
    }
}

/// [`compile_render`] with its failure typed and its measure ([`CompileAttempt`]).
fn render_attempt(
    worker: &CompileWorker,
    device: &WorkerDevice,
    request: &PipelineRequest,
    libraries: &[RowSource<'_>],
    case: &graphics::RenderCase,
    linked: &[LinkedModule],
) -> CompileAttempt<RenderPipeline> {
    match compile_render_request(worker, device, libraries, request) {
        Err(failure) => CompileAttempt {
            measured: failure.measured(),
            result: Err(CompileFailure::Stopped(failure)),
        },
        Ok(compiled) => CompileAttempt {
            measured: Some(compiled.measured),
            result: compiled
                .result
                .map_err(CompileFailure::NotCompiled)
                .and_then(|compiled| {
                    Ok(CompiledCase {
                        tables: render_tables(case, linked, &compiled.handles)
                            .map_err(CompileFailure::Tables)?,
                        pipeline: compiled.pipeline,
                    })
                }),
        },
    }
}

/// The table buffers of `case`'s kernel, whose request function (the one of a
/// `PipelineRequest::BoundCompute`) linked `linked` with `handles`.
pub fn compute_tables(
    case: &ComputeCase,
    linked: &[LinkedModule],
    handles: &[crate::acceptance::LinkedHandles],
) -> Result<TableBuffers, ExecutorError> {
    let kernel = handles
        .first()
        .ok_or("the kernel's compile returned no handles")?;
    table_buffers(ShaderStage::Kernel, &case.function_tables, linked, kernel)
}

/// The table buffers of `case`'s stages, whose request functions (in stage order, as
/// [`render_request`] builds them) linked with `handles`: the case's own function links `linked`,
/// a generated companion nothing.
pub fn render_tables(
    case: &graphics::RenderCase,
    linked: &[LinkedModule],
    handles: &[crate::acceptance::LinkedHandles],
) -> Result<TableBuffers, ExecutorError> {
    let stages = case.stages().collect::<Vec<_>>();
    if stages.len() != handles.len() {
        return Err(format!(
            "the draw's compile returned handles for {} functions, not {}",
            handles.len(),
            stages.len()
        )
        .into());
    }
    let mut buffers = TableBuffers::new();
    for ((stage, function), handles) in stages.into_iter().zip(handles) {
        let links = if case.companion_stage() == Some(stage) {
            &[][..]
        } else {
            linked
        };
        buffers.extend(table_buffers(
            stage,
            &function.resources.function_tables,
            links,
            handles,
        )?);
    }
    Ok(buffers)
}

/// Compile `case`'s pipeline for `device` in a killable compile worker, as [`render_request`] asks,
/// and encode its tables with the handles the worker returned ([`render_tables`]); the worker
/// assembles a source that retains no bitcode (the companion's text, a public fixture's). Refusals
/// and stops are classified as for [`compile_case`].
pub fn compile_render_case(
    worker: &CompileWorker,
    device: &WorkerDevice,
    library: &RowSource<'_>,
    companion: Option<&RowSource<'_>>,
    linked: &[LinkedModule],
    case: &graphics::RenderCase,
    shape: &BindShape,
) -> Result<CompiledCase<RenderPipeline>, ExecutorError> {
    let (request, libraries) = render_request(library, companion, linked, case, shape)?;
    compile_render(worker, device, &request, &libraries, case, linked)
}

/// Compile `request`, a draw of `case` whose own function links `linked`, from `libraries` (in
/// [`PipelineRequest::libraries`] order) for `device` in a killable compile worker, and encode
/// its tables with the handles the worker returned ([`render_tables`]). [`compile_render_case`]
/// asks for [`render_request`]'s draw; the owned device tests ask for a vertex and a fragment
/// function of two owned libraries, which no authored case draws. Refusals and stops are
/// classified as for [`compile_case`].
pub fn compile_render(
    worker: &CompileWorker,
    device: &WorkerDevice,
    request: &PipelineRequest,
    libraries: &[RowSource<'_>],
    case: &graphics::RenderCase,
    linked: &[LinkedModule],
) -> Result<CompiledCase<RenderPipeline>, ExecutorError> {
    render_attempt(worker, device, request, libraries, case, linked)
        .result
        .map_err(ExecutorError::from)
}

/// The `BindShape` the executor binds `case` with on a device that has device addresses exactly
/// when `addressed`: [`bind_state`] with every buffer's guest address and, when `addressed`, a
/// host address (only whether a buffer has one is part of the shape, not its value).
pub fn compute_shape(case: &ComputeCase, addressed: bool) -> Result<BindShape, String> {
    let with_host = |mut address: BufferAddress| {
        address.host = addressed.then_some(address.guest);
        address
    };
    let addresses = case
        .buffers
        .keys()
        .chain(case.function_tables.keys())
        .map(|&index| (index, with_host(BufferAddress::synthetic(index))))
        .collect();
    let embedded = synthetic_used_addresses(case)?
        .into_iter()
        .map(|(key, address)| (key, with_host(address)))
        .collect();
    Ok(bind_state(case, &addresses, &embedded)?.shape())
}

/// The synthetic guest address of each argument-buffer buffer of `case`, by its key
/// ([`argument_buffers::synthetic_addresses`]): a command buffer's record storage at its records
/// address ([`records_ids`]), every other buffer at its [`BufferAddress::synthetic_used`].
pub fn synthetic_used_addresses(
    case: &ComputeCase,
) -> Result<BTreeMap<(u32, u32), BufferAddress>, String> {
    argument_buffers::synthetic_addresses(
        case.argument_buffer_buffers.keys().copied(),
        &records_ids(case)?,
    )
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
    backend: Backend,
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
        backend,
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
    backend: Backend,
    metal_environment_id: &str,
    environment_id: &str,
    worker: &CompileWorker,
) -> Result<CandidateObservation, ExecutorError> {
    execute_prepared(
        root,
        prepare_case(device, admitted)?,
        backend,
        metal_environment_id,
        environment_id,
        worker,
    )
}

/// An admitted case that passed the executor's device-free checks ([`prepare_case`]).
pub struct PreparedCase {
    device: DeviceType,
    checked: CheckedCase,
    mapped: Mapped,
    /// The libraries the case's own function links ([`linked_modules`]).
    linked: Vec<LinkedModule>,
    /// The template of each pipeline token the case's executed commands may run, as the
    /// executor's own context replays them ([`replay_templates`]), or why that context cannot: its
    /// refusal of the case, which the runtime's replay does not share.
    templates: Result<BTreeMap<u64, ReplayTemplate>, ExecutorError>,
}

impl PreparedCase {
    /// The templates the executor's own context replays, or its refusal of the case: a case the
    /// runtime hands back to that context fails there.
    fn templates(&self) -> Result<&BTreeMap<u64, ReplayTemplate>, ExecutorError> {
        self.templates.as_ref().map_err(Clone::clone)
    }
}

/// A pipeline template an executed command may run: its checked case (by id in
/// [`CheckedCase::pipeline_templates`]), mapped, and the libraries it links.
pub struct ReplayTemplate {
    case_id: String,
    mapped: TemplateCase,
    linked: Vec<LinkedModule>,
}

/// What a replayed command runs: a compute command's dispatch, or a render command's draw of
/// `primitive`; or, for the device replay, the template's replay variant for a threadgroup of
/// `threads_per_group` threads, which every command selecting it dispatches indirectly; or, for
/// the device replay of render commands, the render template's draw replay variant, `merged` into
/// one draw or drawn once per command, which every command of the execution draws indirectly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChildCommand {
    Dispatch(Dispatch),
    Draw {
        draw: metal2vulkan::bind::DrawCall,
        primitive: crate::case::Primitive,
    },
    ReplayVariant {
        threads_per_group: [u32; 3],
    },
    DrawReplay {
        merged: bool,
    },
}

/// A template compiled for one replayed command: a compute pipeline for a dispatch, and for a
/// draw the template's render case (with the command's draw) and its pipeline.
pub enum CompiledChild {
    Compute(Box<CompiledCase<ComputePipeline>>),
    Render(Box<graphics::RenderCase>, Box<CompiledCase<RenderPipeline>>),
}

/// A replay template's case in the terms of the executor that runs it: a compute template a
/// dispatch runs, or a render template a draw runs.
#[derive(Clone, Debug)]
pub enum TemplateCase {
    Compute(Box<ComputeCase>),
    Render(Box<graphics::RenderCase>),
}

/// The template of each pipeline token an executed command of `checked`'s case may select: each
/// pipeline state's and each execution's parent pipeline's, by token, of a family some executed
/// command buffer runs. A compute template is mapped only when compute commands execute, a render
/// template only when render commands do, as the checker requires of render templates only then
/// (`check::check_render_replay_templates`): a command never selects a template of the other
/// family, so such a literal is no template here. A replayed compute template binds buffers and
/// threadgroup memory only, and runs no stage input; a replayed render template is a
/// rasterization-disabled vertex function (which the checker requires) that binds buffers only,
/// with no vertex input. Every other resource of its own, and an object or mesh template, whose
/// mesh draws only the runtime replays (`runtime_run::template_states`), is the refusal of the
/// executor's context, before any device: of the case itself where it runs there, and kept for a
/// run the runtime hands back where the runtime runs it (`PreparedCase::templates`).
pub fn replay_templates(
    checked: &CheckedCase,
) -> Result<BTreeMap<u64, ReplayTemplate>, ExecutorError> {
    let case = &checked.primary.case;
    let executed = || {
        case.command_buffers
            .iter()
            .filter(|command| command.execute.is_some())
    };
    let executes_compute = executed().any(|command| !command.is_render());
    let executes_render = executed().any(|command| command.is_render());
    let parents = executed().filter_map(|command| {
        let template = command.execute.as_ref()?.pipeline.as_ref()?;
        Some((template.token, &template.case_id))
    });
    let mut templates = BTreeMap::new();
    for (token, case_id) in case
        .pipeline_states
        .iter()
        .map(|state| (state.token, &state.case_id))
        .chain(parents)
    {
        let template = checked
            .pipeline_templates
            .get(case_id)
            .ok_or_else(|| format!("pipeline template {case_id} was not checked"))?;
        let executes_family = match template.case.stage {
            Stage::Kernel => executes_compute,
            Stage::Vertex | Stage::Fragment | Stage::Object | Stage::Mesh => executes_render,
        };
        if !executes_family {
            continue;
        }
        let mapped = match template.case.stage {
            Stage::Kernel => {
                let mapped = compute_case(&template.case, &template.interface)?;
                let replayable = mapped.textures.is_empty()
                    && mapped.samplers.is_empty()
                    && mapped.function_tables.is_empty()
                    && mapped.acceleration_structures.is_empty()
                    && mapped.argument_buffer_buffers.is_empty()
                    && mapped.argument_buffer_textures.is_empty()
                    && mapped.argument_buffer_samplers.is_empty()
                    && mapped.objects.is_empty()
                    && mapped.imageblock.is_none()
                    && mapped.stage_input.is_none();
                if !replayable {
                    return Err(NotSupported::NotImplemented(format!(
                        "engine candidate: the replayed template {case_id} binds more than \
                         buffers and threadgroup memory"
                    ))
                    .into());
                }
                TemplateCase::Compute(Box::new(mapped))
            }
            Stage::Vertex | Stage::Fragment => {
                let mapped = graphics::render_case(
                    &template.case,
                    &template.interface,
                    template.entry_ll(),
                    template.companion.as_ref(),
                    template.reflection.fragment_imageblock.as_ref(),
                )?;
                let resources = &mapped.vertex.resources;
                // An index buffer or an out-of-bounds target is the template's own draw's; the
                // replayed draw is the command's, which binds neither.
                let replayable = mapped.fragment.is_none()
                    && mapped.companion.is_none()
                    && mapped.vertex_buffers.is_empty()
                    && mapped.index_buffer.is_none()
                    && mapped.out_of_bounds.is_empty()
                    && mapped.depth_stencil.is_none()
                    && mapped.imageblock.is_none()
                    && matches!(mapped.assembly, graphics::Assembly::Primitives(_))
                    && resources.textures.is_empty()
                    && resources.samplers.is_empty()
                    && resources.function_tables.is_empty()
                    && resources.argument_buffer_buffers.is_empty()
                    && resources.argument_buffer_textures.is_empty()
                    && resources.argument_buffer_samplers.is_empty()
                    && resources.objects.is_empty()
                    && resources.command_buffers.is_empty();
                if !replayable {
                    return Err(NotSupported::NotImplemented(format!(
                        "engine candidate: the replayed template {case_id} is not a \
                         rasterization-disabled vertex function that binds buffers only"
                    ))
                    .into());
                }
                TemplateCase::Render(Box::new(mapped))
            }
            Stage::Object | Stage::Mesh => {
                return Err(NotSupported::NotImplemented(format!(
                    "engine candidate: the replayed template {case_id} is an object or mesh \
                     function, whose mesh draws only the runtime replays"
                ))
                .into())
            }
        };
        templates.insert(
            token,
            ReplayTemplate {
                case_id: case_id.clone(),
                mapped,
                linked: linked_modules(&template.linked_functions)?,
            },
        );
    }
    Ok(templates)
}

/// The variants the device replay registers for `case`'s executed compute commands, deduplicated in
/// a fixed order: each compute template of `templates` with the threadgroup size of its own
/// dispatch and its threadgroup lengths, and each dispatch the host encodes into an executed
/// compute command buffer with its pipeline (its own, or the execution's parent pipeline where the
/// commands inherit it), threadgroup lengths, and the threadgroup size of each of its regions (its
/// own, and a thread grid's edges, which run the remaining threads, [`region_sizes`]). Where an
/// executed compute command buffer has a command in its range whose action the host does not
/// encode, the producer may encode a thread grid of any size, so each compute template also
/// registers every edge size of its own threadgroup size (each size no larger on any axis,
/// [`edge_sizes`]), in token order, while the variants stay within the product's
/// `MAX_REPLAY_VARIANTS`; a template whose edges do not fit keeps its own size, and a ragged
/// command of it faults `VARIANT`. Registration is the consumer's: a command that selects another
/// variant dispatches nothing, and the run fails on its replay fault. A template threadgroup length
/// at an index no record holds is the executor's refusal.
pub fn replay_variants(
    case: &ComputeCase,
    templates: &BTreeMap<u64, ReplayTemplate>,
) -> Result<Vec<ReplayVariant>, ExecutorError> {
    use crate::case::indirect_commands::HostAction;
    const LENGTHS: usize = crate::air::reflect::THREADGROUP_BUFFER_ARGUMENT_COUNT_USIZE;
    let lengths = |entries: &mut dyn Iterator<Item = (u32, u32)>| {
        let mut out = [0u32; LENGTHS];
        for (index, length) in entries {
            *out.get_mut(index as usize).ok_or_else(|| {
                NotSupported::NotImplemented(format!(
                    "engine candidate: a replayed threadgroup length at index {index}"
                ))
            })? = length;
        }
        Ok::<_, ExecutorError>(out)
    };
    let mut registered = std::collections::BTreeSet::new();
    // Each compute template's own variant, in token order.
    let mut own = Vec::new();
    for (&token, template) in templates {
        let TemplateCase::Compute(mapped) = &template.mapped else {
            continue;
        };
        let threads_per_group = match mapped.dispatch {
            Dispatch::Threadgroups {
                threads_per_group, ..
            }
            | Dispatch::Threads {
                threads_per_group, ..
            }
            | Dispatch::ThreadgroupsIndirect {
                threads_per_group, ..
            } => threads_per_group,
            other => {
                return Err(NotSupported::NotImplemented(format!(
                    "engine candidate: a replayed template's dispatch {other:?}"
                ))
                .into())
            }
        };
        let lengths = lengths(&mut mapped.threadgroup_lengths.iter().map(|(&i, &l)| (i, l)))?;
        registered.insert((token, threads_per_group, lengths));
        own.push((token, threads_per_group, lengths));
    }
    let mut gpu_encoded = false;
    for command in case.command_buffers.values() {
        let Some(execution) = command.execute.as_ref().filter(|_| !command.is_render()) else {
            continue;
        };
        let [start, end] = command.execution_range;
        gpu_encoded |= (start..end).any(|index| {
            !command
                .commands
                .iter()
                .any(|host| host.index == index && host.action.is_some())
        });
        let parent = execution.pipeline.as_ref().map(|template| template.token);
        for host in &command.commands {
            let sizes = match host.action {
                Some(HostAction::DispatchThreadgroups {
                    threads_per_threadgroup,
                    ..
                }) => vec![threads_per_threadgroup],
                Some(HostAction::DispatchThreads {
                    grid,
                    threads_per_threadgroup,
                }) => region_sizes(grid, threads_per_threadgroup),
                Some(
                    HostAction::DrawPrimitives { .. }
                    | HostAction::DrawIndexedPrimitives { .. }
                    | HostAction::DrawPatches { .. },
                )
                | None => continue,
            };
            let token = if command.inherit_pipeline_state {
                parent
            } else {
                host.pipeline
            };
            let Some(token) = token else {
                continue;
            };
            let lengths = lengths(
                &mut host
                    .threadgroup_memory_lengths
                    .iter()
                    .map(|l| (l.index, l.length)),
            )?;
            registered.extend(sizes.into_iter().map(|size| (token, size, lengths)));
        }
    }
    if gpu_encoded {
        let most = metal2vulkan::bind::MAX_REPLAY_VARIANTS as usize;
        for (token, threads_per_group, lengths) in own {
            let count: u64 = threads_per_group.iter().map(|&a| u64::from(a)).product();
            if count > most as u64 {
                continue;
            }
            let edges: Vec<_> = edge_sizes(threads_per_group)
                .into_iter()
                .map(|size| (token, size, lengths))
                .filter(|variant| !registered.contains(variant))
                .collect();
            if registered.len() + edges.len() <= most {
                registered.extend(edges);
            }
        }
    }
    registered
        .into_iter()
        .map(|(token, threads_per_group, lengths)| {
            let pipeline = std::num::NonZeroU64::new(token)
                .ok_or("a replayed pipeline token of 0 names no pipeline")?;
            Ok(ReplayVariant::new(pipeline, threads_per_group, lengths))
        })
        .collect()
}

/// The regions a device replay of `commands` commands over `variants` registered variants records
/// per command: `MAX_REPLAY_REGIONS`, or as many as keep its dispatches, `commands * regions *
/// variants`, within `MAX_REPLAY_DISPATCHES`. So every range executor v92 ran (one dispatch per
/// command and variant, at most 16,384 x 64 = 2^20) still runs, and a command that needs more
/// regions than a long range records is a typed `REGIONS` fault, never a refused range. Both of the
/// device replay's paths record the same regions.
pub(crate) fn replay_regions(commands: u32, variants: usize) -> u32 {
    use metal2vulkan::bind::{MAX_REPLAY_DISPATCHES, MAX_REPLAY_REGIONS};
    let per_region = u64::from(commands.max(1)) * (variants.max(1) as u64);
    let fit = MAX_REPLAY_DISPATCHES / per_region;
    u32::try_from(fit.min(u64::from(MAX_REPLAY_REGIONS)))
        .expect("at most MAX_REPLAY_REGIONS")
        .max(1)
}

/// The threadgroup sizes of the regions of a thread grid `grid` in threadgroups of `local`: per
/// axis, the whole threadgroups' size where the grid holds one, and the remaining threads where it
/// is not a multiple. A grid with a zero axis runs nothing and needs no variant; it registers
/// `local`, as a host-encoded dispatch always has.
pub fn region_sizes(grid: [u32; 3], local: [u32; 3]) -> Vec<[u32; 3]> {
    if grid.contains(&0) || local.contains(&0) {
        return vec![local];
    }
    let axis = |i: usize| {
        let mut sizes = Vec::with_capacity(2);
        if grid[i] >= local[i] {
            sizes.push(local[i]);
        }
        if !grid[i].is_multiple_of(local[i]) {
            sizes.push(grid[i] % local[i]);
        }
        sizes
    };
    let (xs, ys, zs) = (axis(0), axis(1), axis(2));
    let mut out = Vec::with_capacity(xs.len() * ys.len() * zs.len());
    for &z in &zs {
        for &y in &ys {
            for &x in &xs {
                out.push([x, y, z]);
            }
        }
    }
    out
}

/// Every threadgroup size an edge of a thread grid in threadgroups of `local` can run: each size of
/// 1 to `local`'s threads on every axis, `local` itself among them.
pub fn edge_sizes(local: [u32; 3]) -> Vec<[u32; 3]> {
    let mut out = Vec::new();
    for z in 1..=local[2] {
        for y in 1..=local[1] {
            for x in 1..=local[0] {
                out.push([x, y, z]);
            }
        }
    }
    out
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
    let checked = admitted.into_checked_for(ExecutionRequest::EngineCandidate { device })?;
    crate::executor_contract::require_executor(&checked.requirements(), Executor::EngineCandidate)
        .map_err(NotSupported::Requirements)?;
    let case = &checked.primary.case;
    let mapped = match case.stage {
        Stage::Kernel => Mapped::Compute(Box::new(compute_case(case, &checked.primary.interface)?)),
        Stage::Vertex | Stage::Fragment => Mapped::Render(Box::new(graphics::render_case(
            case,
            &checked.primary.interface,
            checked.primary.entry_ll(),
            checked.primary.companion.as_ref(),
            checked.primary.reflection.fragment_imageblock.as_ref(),
        )?)),
        Stage::Object | Stage::Mesh => Mapped::Mesh(Box::new(mesh::mesh_case(
            case,
            checked.primary.companion.as_ref(),
        )?)),
    };
    let linked = linked_modules(&checked.primary.linked_functions)?;
    let templates = replay_templates(&checked);
    let prepared = PreparedCase {
        device,
        checked,
        mapped,
        linked,
        templates,
    };
    // The runtime replays every template through its own pipeline states, so a template the
    // executor's context cannot replay refuses device-free only a case that runs on that context; a
    // case the runtime runs meets the refusal only if the runtime hands it back.
    if let (Err(refused), runtime_run::Route::Executor(_)) =
        (&prepared.templates, runtime_run::route(&prepared))
    {
        return Err(refused.clone());
    }
    Ok(prepared)
}

/// What one compile of [`case_compiles`] builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompileSubject {
    /// The case's own pipeline: its kernel's, or its draw's with its generated companion.
    Case,
    /// The device-replay variant of the compute template of pipeline token `token` for a
    /// threadgroup of `threads_per_group` threads.
    ReplayVariant {
        token: u64,
        threads_per_group: [u32; 3],
    },
}

/// One pipeline a run of a prepared case compiles, with the request and libraries the executor
/// builds for it.
pub struct CaseCompile<'a> {
    pub subject: CompileSubject,
    pipeline: PlannedPipeline<'a>,
}

/// A planned compile's request, its libraries in [`PipelineRequest::libraries`] order, what its
/// tables are encoded from, and the bind shape the run plans it for.
enum PlannedPipeline<'a> {
    Compute {
        request: PipelineRequest,
        libraries: Vec<RowSource<'a>>,
        // Boxed: a mapped case is large beside a render case's reference.
        case: Box<Cow<'a, ComputeCase>>,
        linked: &'a [LinkedModule],
        shape: BindShape,
    },
    Render {
        request: PipelineRequest,
        libraries: Vec<RowSource<'a>>,
        case: &'a graphics::RenderCase,
        linked: &'a [LinkedModule],
        shape: BindShape,
    },
    /// A mesh draw's pipeline, whose functions link nothing (`mesh::mesh_request`).
    Mesh {
        request: PipelineRequest,
        libraries: Vec<RowSource<'a>>,
        shape: BindShape,
    },
}

impl CaseCompile<'_> {
    /// Compile in `worker` for `device` as the executor compiles it (its tables encoded from the
    /// handles the worker returned) and record it in `artifact` as the executor's run records it
    /// (the pipeline's content, its plan for the run's bind shape, its tables), keeping only
    /// whether it compiled and what the worker measured: the pipeline is dropped here.
    pub fn compile(
        &self,
        worker: &CompileWorker,
        device: &WorkerDevice,
        artifact: &mut artifact::CaseArtifact,
    ) -> (Result<(), CompileFailure>, Option<crate::worker::Measured>) {
        let mut record = |pipeline: &dyn artifact::Planned, tables: &TableBuffers, shape| {
            artifact
                .record(self.subject, pipeline, tables, device.device(), shape)
                .map_err(CompileFailure::Artifact)
        };
        match &self.pipeline {
            PlannedPipeline::Compute {
                request,
                libraries,
                case,
                linked,
                shape,
            } => {
                let attempt = compute_attempt(worker, device, libraries, request, case, linked);
                let result = attempt
                    .result
                    .and_then(|compiled| record(&compiled.pipeline, &compiled.tables, shape));
                (result, attempt.measured)
            }
            PlannedPipeline::Render {
                request,
                libraries,
                case,
                linked,
                shape,
            } => {
                let attempt = render_attempt(worker, device, request, libraries, case, linked);
                let result = attempt
                    .result
                    .and_then(|compiled| record(&compiled.pipeline, &compiled.tables, shape));
                (result, attempt.measured)
            }
            PlannedPipeline::Mesh {
                request,
                libraries,
                shape,
            } => match compile_render_request(worker, device, libraries, request) {
                Err(failure) => {
                    let measured = failure.measured();
                    (Err(CompileFailure::Stopped(failure)), measured)
                }
                Ok(compiled) => {
                    let result = compiled
                        .result
                        .map_err(CompileFailure::NotCompiled)
                        .and_then(|c| record(&c.pipeline, &TableBuffers::new(), shape));
                    (result, Some(compiled.measured))
                }
            },
        }
    }
}

/// Every pipeline a run of `prepared` compiles on `device` before it reads any device output, in
/// the order the run compiles them, and how many of its executed command buffers compile what no
/// device-free compile builds there.
pub struct CaseCompiles<'a> {
    pub compiles: Vec<CaseCompile<'a>>,
    /// The executed command buffers (with a non-empty execution range) whose commands' pipelines no
    /// device-free compile builds on this device: a render command buffer, whose host replay
    /// compiles each command's pipeline for the bind shape of the records the producer writes on
    /// the device, and whose draw replay (`platform::compute::draw_replay`) compiles its
    /// template's variant for the bind state of the execution's encoder, as the runtime path's
    /// record leaves both uncovered; and a compute one the host replay runs, where the product sets
    /// no device replay up or the case registers no variant (a case the runtime runs whose
    /// templates the executor's context does not replay registers none).
    pub host_replayed: usize,
}

/// The pipelines a run of `prepared` compiles on `device` before it reads any device output,
/// with the requests the executor builds for them: the case's own pipeline ([`compute_request`] for
/// the bind shape [`compute_shape`] gives, or [`render_request`] with its generated companion for
/// [`graphics::render_shape`]'s), and, where the device replay runs an executed compute command
/// buffer (the product sets it up on `device` and the case registers variants), each registered
/// variant (`replay_variant_request` for the bind state [`replay_variant_state`] of the run's
/// resident allocations, [`run_allocations`]). The bind shapes depend on the device only through
/// whether it has device addresses, which the executor reads from what its context enabled and this
/// from `device`'s description: the same fact for an opened device, whose description is restricted
/// to what was enabled. The authored compile census compiles these on each committed profile; the
/// executor builds the same requests through [`compile_case`], [`compile_render_case`], and
/// `compile_child`.
pub fn case_compiles<'a>(
    prepared: &'a PreparedCase,
    device: &WorkerDevice,
) -> Result<CaseCompiles<'a>, ExecutorError> {
    use metal2vulkan::device::VkFeature;
    let addressed = device.description().has(VkFeature::BufferDeviceAddress);
    let executed = |command: &CommandBufferResource| {
        command.execute.is_some() && command.execution_range[0] != command.execution_range[1]
    };
    let library = RowSource::of_row(&prepared.checked.primary.source);
    let linked = prepared.linked.as_slice();
    let mut compiles = Vec::new();
    let mut host_replayed = 0;
    match &prepared.mapped {
        Mapped::Compute(compute) => {
            // A case the runtime runs replays its templates through the runtime's own pipeline
            // states (`prepare_case`), so the executor's context's refusal of them is not that
            // run's: it registers no variant here, and each executed command buffer is uncovered.
            let registered = match prepared.templates() {
                Ok(templates) => replay_variants(compute, templates)?,
                Err(_) if runtime_run::route(prepared) == runtime_run::Route::Runtime => Vec::new(),
                Err(refused) => return Err(refused),
            };
            let shape = compute_shape(compute, addressed).map_err(ExecutorError::Failed)?;
            let (request, libraries) = compute_request(&library, linked, compute, &shape);
            compiles.push(CaseCompile {
                subject: CompileSubject::Case,
                pipeline: PlannedPipeline::Compute {
                    request,
                    libraries,
                    case: Box::new(Cow::Borrowed(compute.as_ref())),
                    linked,
                    shape,
                },
            });
            // Whether the product sets the device replay up on this device, asked only of a case
            // whose executed compute commands could take it (`device_replay::replay_commands`).
            let mut set_up = None;
            let mut on_device = false;
            for command in compute.command_buffers.values().filter(|c| executed(c)) {
                if command.is_render() || registered.is_empty() {
                    host_replayed += 1;
                    continue;
                }
                let runs = match set_up {
                    Some(runs) => runs,
                    None => {
                        let runs =
                            match metal2vulkan::bind::replay_preprocess_module(device.device()) {
                                Ok(_) => true,
                                Err(Error::Unsupported(_)) => false,
                                Err(error) => return Err(product_refusal(&error)),
                            };
                        set_up = Some(runs);
                        runs
                    }
                };
                if runs {
                    on_device = true;
                } else {
                    host_replayed += 1;
                }
            }
            if on_device {
                let resident =
                    replay_resident(run_allocations(compute).into_iter().enumerate().map(
                        |(token, (_, length))| {
                            let address = BufferAddress::synthetic_used(token as u32);
                            (
                                BufferAddress {
                                    host: Some(address.guest),
                                    ..address
                                },
                                length,
                            )
                        },
                    ))
                    .map_err(ExecutorError::Failed)?;
                for variant in &registered {
                    let token = variant.pipeline.get();
                    let threads_per_group = variant.threads_per_threadgroup;
                    let shape = replay_variant_state(&resident, variant).shape();
                    let variant =
                        replay_variant_request(prepared, token, threads_per_group, &shape)?;
                    compiles.push(CaseCompile {
                        subject: CompileSubject::ReplayVariant {
                            token,
                            threads_per_group,
                        },
                        pipeline: PlannedPipeline::Compute {
                            request: variant.request,
                            libraries: variant.libraries,
                            case: Box::new(Cow::Owned(variant.child)),
                            linked: variant.linked,
                            shape,
                        },
                    });
                }
            }
        }
        Mapped::Render(render) => {
            let shape = graphics::render_shape(render, addressed).map_err(ExecutorError::Failed)?;
            let companion = companion_source(render);
            let (request, libraries) =
                render_request(&library, companion.as_ref(), linked, render, &shape)?;
            compiles.push(CaseCompile {
                subject: CompileSubject::Case,
                pipeline: PlannedPipeline::Render {
                    request,
                    libraries,
                    case: render.as_ref(),
                    linked,
                    shape,
                },
            });
            host_replayed = render
                .command_buffers()
                .filter(|(_, command)| executed(command))
                .count();
        }
        // The pipeline the runtime compiles in the run itself (`runtime_run`), for its own
        // stage's resources.
        Mapped::Mesh(mesh) => {
            let shape = mesh::mesh_shape(mesh, addressed)?;
            let companion = mesh
                .companion
                .as_ref()
                .map(|generated| RowSource::Text(Cow::Owned(generated.air_text(Spelling::Opaque))));
            let (request, libraries) = mesh::mesh_request(&library, companion, mesh, &shape)?;
            compiles.push(CaseCompile {
                subject: CompileSubject::Case,
                pipeline: PlannedPipeline::Mesh {
                    request,
                    libraries,
                    shape,
                },
            });
        }
    }
    Ok(CaseCompiles {
        compiles,
        host_replayed,
    })
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
            let attempt = run_prepared(&prepared, input.backend, &worker);
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

/// Run one prepared case on its device by the path [`runtime_run::route`] gives it: through the
/// runtime ([`runtime_run::run`]), or on the executor's own context ([`run_on_context`]) while the
/// runtime path waits for prerequisites of the case's family, or when the runtime raised a typed
/// gap before it committed (the bridge). A row or refusal of the executor's context records why the
/// runtime did not run it.
fn run_prepared(prepared: &PreparedCase, backend: Backend, worker: &CompileWorker) -> RunAttempt {
    let why = match runtime_run::route(prepared) {
        runtime_run::Route::Runtime => match runtime_run::run(prepared, backend) {
            runtime_run::RuntimeOutcome::Ran(attempt) => return *attempt,
            runtime_run::RuntimeOutcome::Gap(gap) => runtime_run::ExecutorRun::Gap(gap),
        },
        runtime_run::Route::Executor(prerequisites) => {
            runtime_run::ExecutorRun::WaitsFor(prerequisites)
        }
    };
    let mut attempt = run_on_context(prepared, backend, worker);
    let environments = attempt
        .result
        .as_mut()
        .ok()
        .map(|run| &mut run.environment)
        .into_iter()
        .chain(
            attempt
                .refused_on
                .as_mut()
                .map(|opened| &mut opened.environment),
        );
    for environment in environments {
        if let Some(object) = environment.as_object_mut() {
            for (key, value) in runtime_run::executor_members(&why) {
                object.insert(key.into(), value);
            }
        }
    }
    attempt
}

/// Compile (in `worker`), bind, and execute one prepared case on the executor's own context: a
/// kernel case as one dispatch, a vertex or fragment case as one draw. The run's row or refusal
/// records what it compiled, by content ([`artifact::CaseArtifact`]).
fn run_on_context(prepared: &PreparedCase, backend: Backend, worker: &CompileWorker) -> RunAttempt {
    let case = &prepared.checked.primary.case;
    // A template this context cannot replay refuses the case here, before any device work.
    let templates = match prepared.templates() {
        Ok(templates) => templates,
        Err(refused) => return RunAttempt::of(Err(refused)),
    };
    // The worker assembles a row that retains no bitcode (a public fixture), and a companion.
    let library = RowSource::of_row(&prepared.checked.primary.source);
    let linked = &prepared.linked;
    let artifact = RefCell::new(artifact::CaseArtifact::default());
    // Each executed command's template, compiled for the command's bind shape and dispatch or
    // draw.
    let children =
        |target: &WorkerDevice, token: u64, shape: &BindShape, command: &ChildCommand| {
            let compiled = compile_child(prepared, worker, target, (token, shape, command))?;
            record_child(
                &mut artifact.borrow_mut(),
                target,
                (token, shape, command),
                &compiled,
            )?;
            Ok(compiled)
        };
    let mut attempt = match &prepared.mapped {
        Mapped::Compute(compute) => {
            let registered = match replay_variants(compute, templates) {
                Ok(registered) => registered,
                Err(error) => return RunAttempt::of(Err(error)),
            };
            super::platform::execute_engine(
                prepared.device,
                &case.case_id,
                compute,
                |target, shape| {
                    let compiled = compile_case(worker, target, &library, linked, compute, shape)?;
                    artifact.borrow_mut().record(
                        CompileSubject::Case,
                        &compiled.pipeline,
                        &compiled.tables,
                        target.device(),
                        shape,
                    )?;
                    Ok(compiled)
                },
                children,
                &registered,
                backend,
            )
        }
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
                    artifact.borrow_mut().record(
                        CompileSubject::Case,
                        &compiled.pipeline,
                        &compiled.tables,
                        target.device(),
                        shape,
                    )?;
                    Ok(compiled)
                },
                children,
                backend,
            )
        }
        // Only the runtime path draws a mesh render pipeline; the executor's context runs a
        // mesh case only when the runtime handed it back (the bridge).
        Mapped::Mesh(_) => RunAttempt::of(Err(NotSupported::NotImplemented(
            "engine candidate: the executor's own context draws no mesh render pipeline".into(),
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

/// Record a compiled child in the run's artifact: a device-replay variant by its subject, with the
/// preprocess module the device replay runs it under; a host replay's pipeline (for a record a
/// producer wrote on the device) and a draw replay's variant (for the bind state of the execution's
/// encoder) as ones the census cannot compile, as the runtime path records every pipeline its
/// executed commands ran.
fn record_child(
    artifact: &mut artifact::CaseArtifact,
    target: &WorkerDevice,
    (token, shape, command): (u64, &BindShape, &ChildCommand),
    compiled: &CompiledChild,
) -> Result<(), ExecutorError> {
    match (command, compiled) {
        (&ChildCommand::ReplayVariant { threads_per_group }, CompiledChild::Compute(compiled)) => {
            artifact.record(
                CompileSubject::ReplayVariant {
                    token,
                    threads_per_group,
                },
                &compiled.pipeline,
                &compiled.tables,
                target.device(),
                shape,
            )?;
            artifact.record_replay_module(target.device())
        }
        _ => {
            artifact.record_uncovered();
            Ok(())
        }
    }
}

/// Compile the template of `token` for one command the case executes, in `worker`, for the bind
/// shape `shape`: a compute template for a dispatch, with the command's dispatch, or its replay
/// variant; a render template for a draw, with the command's draw (its class stays the template's
/// own, as Metal compiles it), or its draw replay variant for an execution's commands. A token
/// without a template, or a template of the other family, fails the run.
fn compile_child(
    prepared: &PreparedCase,
    worker: &CompileWorker,
    target: &WorkerDevice,
    (token, shape, command): (u64, &BindShape, &ChildCommand),
) -> Result<CompiledChild, ExecutorError> {
    let template = prepared.templates()?.get(&token).ok_or_else(|| {
        ExecutorError::Failed(format!(
            "an executed command's pipeline token {token} names no template"
        ))
    })?;
    let source = RowSource::of_row(&prepared.checked.pipeline_templates[&template.case_id].source);
    // A render template's vertex function alone, its variant in `form`.
    let render_child = |child: graphics::RenderCase, form: variants::DrawForm| {
        let (mut request, libraries) =
            render_request(&source, None, &template.linked, &child, shape)?;
        let PipelineRequest::VertexDraw { form: formed, .. } = &mut request else {
            return Err(ExecutorError::Failed(
                "an executed command's render template has a fragment stage".into(),
            ));
        };
        *formed = form;
        let compiled = compile_render(
            worker,
            target,
            &request,
            &libraries,
            &child,
            &template.linked,
        )?;
        Ok(CompiledChild::Render(Box::new(child), Box::new(compiled)))
    };
    match (&template.mapped, command) {
        (TemplateCase::Compute(mapped), ChildCommand::Dispatch(dispatch)) => {
            let child = ComputeCase {
                dispatch: *dispatch,
                stage_in_region: None,
                ..(**mapped).clone()
            };
            compile_case(worker, target, &source, &template.linked, &child, shape)
                .map(|c| CompiledChild::Compute(Box::new(c)))
        }
        (TemplateCase::Compute(_), &ChildCommand::ReplayVariant { threads_per_group }) => {
            let variant = replay_variant_request(prepared, token, threads_per_group, shape)?;
            compile_request(
                worker,
                target,
                &variant.libraries,
                &variant.request,
                &variant.child,
                variant.linked,
            )
            .map(|c| CompiledChild::Compute(Box::new(c)))
        }
        // In the large-buffer form, as the runtime's host step draws an executed render command.
        (TemplateCase::Render(mapped), &ChildCommand::Draw { draw, .. }) => {
            let child = graphics::RenderCase {
                draw,
                ..(**mapped).clone()
            };
            render_child(child, variants::DrawForm::Large)
        }
        // In a draw replay's form, as the runtime's GPU form draws an execution's commands
        // (`platform::compute::draw_replay`).
        (TemplateCase::Render(mapped), &ChildCommand::DrawReplay { merged }) => {
            let form = match merged {
                true => variants::DrawForm::Merged,
                false => variants::DrawForm::Replay,
            };
            render_child((**mapped).clone(), form)
        }
        _ => Err(ExecutorError::Failed(format!(
            "an executed command's pipeline token {token} names a template of the other command \
             family"
        ))),
    }
}

/// The compile of the device-replay variant of the compute template of pipeline token `token` for
/// a threadgroup of `threads_per_group` threads, bound as `shape`: the template's case with a
/// [`replay_dispatch`], and [`replay_request`]'s request and libraries for it. The executor's
/// compile of a variant ([`compile_child`]) and the authored compile census ([`case_compiles`])
/// build it here. A token without a template, or with a render template, fails the run.
struct VariantRequest<'a> {
    child: ComputeCase,
    request: PipelineRequest,
    libraries: Vec<RowSource<'a>>,
    linked: &'a [LinkedModule],
}

fn replay_variant_request<'a>(
    prepared: &'a PreparedCase,
    token: u64,
    threads_per_group: [u32; 3],
    shape: &BindShape,
) -> Result<VariantRequest<'a>, ExecutorError> {
    let template = prepared.templates()?.get(&token).ok_or_else(|| {
        ExecutorError::Failed(format!(
            "an executed command's pipeline token {token} names no template"
        ))
    })?;
    let TemplateCase::Compute(mapped) = &template.mapped else {
        return Err(ExecutorError::Failed(format!(
            "an executed command's pipeline token {token} names a template of the other command \
             family"
        )));
    };
    let source = RowSource::of_row(&prepared.checked.pipeline_templates[&template.case_id].source);
    let child = ComputeCase {
        dispatch: replay_dispatch(threads_per_group),
        stage_in_region: None,
        ..(**mapped).clone()
    };
    let (request, libraries) = replay_request(&source, &template.linked, &child, shape);
    Ok(VariantRequest {
        child,
        request,
        libraries,
        linked: &template.linked,
    })
}

/// [`execute_case`] of a case [`prepare_case`] accepted: the device half in a device worker
/// (`run_on_device`), the comparison and the row here. Without generated commands
/// (`candidate::run_without_generated_commands`) the row is compared and returned but not stored:
/// it measures another path, and the slot is the default's.
pub fn execute_prepared(
    root: &Path,
    prepared: PreparedCase,
    backend: Backend,
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
    super::validate_backend_host(backend)?;
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
            .venue_of(environment_id, backend)
            .map(|venue| VenueRun {
                environment_id: environment_id.into(),
                profile: venue.profile.clone(),
                allowed: venue.identities_compared_with(metal_environment_id),
            }),
    };
    let input = DeviceInput {
        root: root.to_path_buf(),
        case: case.clone(),
        device,
        backend,
        input_sha256: checked.primary.input_sha256.clone(),
        wait_timeout_ms: u64::try_from(wait_timeout().as_millis()).unwrap_or(u64::MAX),
        restricted_to,
        leg: super::requested_leg(),
        generated_commands: super::requested_generated_commands(),
        venue,
        source_index: crate::source::source_index(),
    };
    let DeviceOutcome { result, refused_on } = crate::worker::device::execute(worker, &input);
    let run = match result {
        Ok(run) => run,
        // A typed refusal on an opened device is evidence too: stored in the slot the row would
        // have taken, with that device's facts. Without generated commands nothing is stored, as
        // for a row.
        Err(ExecutorError::NotSupported(not_supported)) => {
            if let Some(opened) = refused_on.filter(|_| input.generated_commands) {
                store.upsert_refusal(RefusalObservation {
                    case_id: case.case_id.clone(),
                    air_sha256: case.air_sha256.clone(),
                    input_sha256: input.input_sha256.clone(),
                    translator_fingerprint: opened.translator_fingerprint,
                    backend,
                    environment_id: engine_slot(
                        environment_id,
                        input.restricted_to.as_deref(),
                        input.leg,
                    ),
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
    let leg = run
        .census
        .leg
        .ok_or("the device worker's census names no robustness leg")?;
    let slot = engine_slot(environment_id, run.census.restricted_to.as_deref(), leg);
    let candidate_output_sha256 = sha256_bytes(&run.output);
    let golden = base64::engine::general_purpose::STANDARD
        .decode(&metal.output_b64)
        .map_err(|error| format!("decode the Metal observation's output: {error}"))?;
    // A case with out-of-bounds targets is judged by their containment property.
    let status = if crate::executor_contract::out_of_bounds::matches(
        case,
        &run.comparison,
        &golden,
        &run.output,
        run.census.out_of_bounds.as_ref(),
    ) {
        CandidateStatus::Match
    } else {
        CandidateStatus::Mismatch
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
        backend,
        environment_id: slot.clone(),
        environment: run.environment,
        executor_abi: ENGINE_EXECUTOR_ABI.into(),
        comparison: run.comparison.clone(),
        census: Some(run.census.clone()),
        timing: Some(run.timing),
        status,
    };
    // The row contract first, so a refusal (such as a row under an unrequested profiles layer)
    // reports its own reason.
    observation.validate_content()?;
    let dependencies = CandidateDependencies {
        case,
        metal: &metal,
        spv_sha256: &run.spv_sha256,
        translator_fingerprint: &run.translator_fingerprint,
        backend,
        environment_id: &slot,
        executor_abi: ENGINE_EXECUTOR_ABI,
    };
    if !observation.dependency_matches(&dependencies) {
        return Err("constructed candidate observation failed its own dependency check".into());
    }
    // A run without generated commands (`--without-generated-commands`) measures the device
    // replay's other path on a device whose default run takes the generated-commands path:
    // compared and returned, never stored over the default's evidence in the same slot.
    if super::requested_generated_commands() {
        store.upsert_candidate(observation.clone())?;
    }
    Ok(observation)
}

/// The bytes of `output` from the final buffer contents and the final texture contents, by Metal
/// index, the final argument-buffer buffer and texture contents, by `(owner, field offset)`, and
/// the final contents of a tile pass's color attachments, by color index.
pub fn select_output(
    output: ComputeOutput,
    buffers: &BTreeMap<u32, Vec<u8>>,
    embedded: &BTreeMap<(u32, u32), Vec<u8>>,
    textures: &BTreeMap<u32, LiteralTexture>,
    embedded_textures: &BTreeMap<(u32, u32), LiteralTexture>,
    attachments: &BTreeMap<u32, LiteralTexture>,
) -> Result<Vec<u8>, String> {
    match output {
        ComputeOutput::None => Ok(Vec::new()),
        // Rows as the Metal oracle reads a render target (`LiteralRenderTarget::select`).
        ComputeOutput::RenderTarget {
            index,
            origin,
            dimensions,
        } => {
            let attachment = attachments
                .get(&index)
                .ok_or_else(|| format!("output render target {index} is not attached"))?;
            crate::literal::select_tightly_packed_2d(
                &attachment.bytes,
                [attachment.dimensions[0], attachment.dimensions[1]],
                origin,
                dimensions,
                attachment.format.bytes_per_pixel(),
            )
        }
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
